# evaluate_agent.py
"""
Benchmark evaluator for the multi-agent framework.

Produces <alias>_agent_results.json — directly comparable to the
zero-shot <alias>_eval_results.json from evaluate.py.

CLI examples
────────────
# Evaluate one model with 3 agent runs per item, 3 ReAct iters per node
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview

# Multiple models
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o

# More samples for tighter pass@1 estimate (expensive)
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --n-samples 10

# Wider ReAct budget
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --max-iter 5

# Re-run specific items (ignores cached agent results)
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --rescore pr29633

# Re-run everything
python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --rescore
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Union

from dotenv import load_dotenv
from openai import OpenAI

# ── Reuse everything that doesn't change ──────────────────────────
from evaluate import (
    ROOT,
    CONFIG_FILE,
    N_SAMPLES,
    RunConfig,
    make_client,
    load_existing_results,
    load_or_build_weights,
    load_test_files,
    domain_weighted_codebleu,
    run_test_with_candidate,
    clone_and_install,
    pass_at_k,
    print_summary,
)
from orchestrator import Orchestrator

load_dotenv()

DEFAULT_MODELS   = ["GLM5:z-ai/glm-5"]
DEFAULT_N        = 3     # agent runs are expensive — default is lower than zero-shot
DEFAULT_MAX_ITER = 3     # ReAct iterations per sub-task


# ─────────────────────── Result file helpers ──────────────────────

def _agent_results_path(cfg: RunConfig) -> Path:
    return ROOT / f"{cfg.alias}_agent_results.json"


def _load_existing_agent_results(cfg: RunConfig) -> Dict[str, dict]:
    path = _agent_results_path(cfg)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["name"]: r for r in data.get("results", []) if r.get("status") == "OK"}


# ─────────────────────── Per-item evaluation ──────────────────────

def evaluate_item_agent(
    cfg:            RunConfig,
    client:         OpenAI,
    item_path_str:  str,
    domain_weights: dict,
    existing:       Dict[str, dict],
    n_samples:      int,
    max_iter:       int,
    rescore:        bool = False,
) -> dict:
    print("\n" + "=" * 70)
    print(f"[{cfg.alias}] Agent eval: {item_path_str}")

    item_dir       = ROOT / "manual_corpus" / item_path_str
    meta           = json.loads((item_dir / "metadata.json").read_text())
    prompt         = (item_dir / "prompt.txt").read_text().strip()
    reference      = (item_dir / meta["after_file"]).read_text()
    test_files     = load_test_files(item_dir, meta)
    name           = meta["name"]
    repo_url       = meta["repo_url"]
    checkout       = meta.get("checkout", "master")
    target_relpath = meta["target_relpath"]

    # Check if we can skip
    if not rescore and name in existing:
        cached = existing[name]
        if cached.get("n_samples", 0) >= n_samples:
            print(f"  [SKIP] Already have {cached['n_samples']} agent samples.")
            _print_cached_summary(cached)
            return cached

    pass_results:    List[bool] = []
    codebleu_scores: List[dict] = []
    generations:     List[dict] = []

    print(f"  Cloning {repo_url} for agent evaluation ...")

    with tempfile.TemporaryDirectory(prefix=f"{name}_{cfg.alias}_agent_") as tmp:
        workdir  = Path(tmp)
        repo_dir, err = clone_and_install(repo_url, checkout, workdir)

        if repo_dir is None:
            print(f"  [ERROR] Repo setup: {err}")
            return {
                "item": item_path_str, "name": name,
                "status": "ERROR", "error": err,
            }

        for sample_idx in range(n_samples):
            print(f"\n  ── Agent sample {sample_idx + 1}/{n_samples} "
                  f"{'─' * 40}")

            # Inside evaluate_item_agent(), replace the Orchestrator instantiation:

            # Build the trace directory for this specific sample
            # Layout: agent_traces/<alias>/<item_name>/sample_<n>/
            trace_root = ROOT / "agent_traces"
            sample_save_dir = (
                trace_root
                / cfg.alias
                / name
                / f"sample_{sample_idx + 1}"
            )

            orchestrator = Orchestrator(
                cfg            = cfg,
                client         = client,
                repo_dir       = repo_dir,
                target_relpath = target_relpath,
                test_files     = test_files,
                max_iter       = max_iter,
                verbose        = True,
                run_save_dir   = sample_save_dir,   # ← NEW
            )

            t0     = time.time()
            result = orchestrator.solve(prompt)
            elapsed= round(time.time() - t0, 1)

            final_code = result["final_code"]

            # ── Score with full benchmark test suite ───────────────
            if final_code.strip():
                passed, pytest_out, failures = run_test_with_candidate(
                    repo_dir, target_relpath, final_code, test_files
                )
            else:
                passed, pytest_out, failures = False, "empty generation", []

            cb = domain_weighted_codebleu(
                final_code, reference, domain_weights, passed=passed
            )

            pass_results.append(passed)
            codebleu_scores.append(cb)

            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"\n  sample {sample_idx+1:>2}  {status}"
                f"  codebleu={cb['codebleu']:.4f}"
                f"  domain_codebleu={cb['domain_codebleu']:.4f}"
                f"  elapsed={elapsed}s  nodes={result['n_nodes']}"
            )

            generations.append({
                "sample":      sample_idx,
                "passed":      passed,
                "code_length": len(final_code),
                "failures":    failures,
                "pytest_tail": pytest_out[-300:] if pytest_out else "",
                "codebleu":    cb,
                "elapsed_s":   elapsed,
                "n_nodes":     result["n_nodes"],
                "tcrg":        result["tcrg"],
                # Per-node summary (full code omitted to keep file sizes manageable)
                "node_summary": [
                    {
                        "node_id":      nh["node_id"],
                        "node_title":   nh["node_title"],
                        "n_iters":      nh["n_iters"],
                        "final_passed": nh["final_passed"],
                    }
                    for nh in result.get("node_history", [])
                ],
            })

            # Rate-limit buffer between samples
            time.sleep(2.0)

    n_passed = sum(pass_results)
    p1       = pass_at_k(n_samples, n_passed, k=1)
    avg_cb   = {
        key: round(sum(s[key] for s in codebleu_scores) / n_samples, 4)
        for key in codebleu_scores[0]
    }

    print(
        f"\n  ► {name}  pass@1={p1:.3f}"
        f"  avg_domain_codebleu={avg_cb['domain_codebleu']:.4f}"
        f"  ({n_passed}/{n_samples} passed)"
    )

    return {
        "item":         item_path_str,
        "name":         name,
        "model":        cfg.model_id,
        "alias":        cfg.alias,
        "status":       "OK",
        "mode":         "agent",
        "n_samples":    n_samples,
        "max_iter":     max_iter,
        "n_passed":     n_passed,
        "pass_at_1":    round(p1, 4),
        "avg_codebleu": avg_cb,
        "per_sample":   generations,
    }


def _print_cached_summary(r: dict) -> None:
    for s in r.get("per_sample", []):
        status = "✓ PASS" if s["passed"] else "✗ FAIL"
        cb     = s["codebleu"]
        print(
            f"    [cached] sample {s['sample']+1:>2}  {status}"
            f"  codebleu={cb['codebleu']:.4f}"
            f"  domain_codebleu={cb['domain_codebleu']:.4f}"
        )


# ─────────────────────── Comparison table ─────────────────────────

def print_comparison(
    cfg:            RunConfig,
    agent_results:  dict,
    zeroshot_results: Dict[str, dict],
) -> None:
    """Side-by-side zero-shot vs agent scores."""
    w = 82
    print(f"\n{'='*w}")
    print(f"  COMPARISON  —  {cfg.alias}  (zero-shot  vs  agent)")
    print(
        f"{'PR':<12} {'ZS Pass@1':>10} {'ZS Dom-CB':>10}"
        f"  {'AG Pass@1':>10} {'AG Dom-CB':>10}  {'Δ Pass@1':>9}"
    )
    print(f"{'-'*w}")

    total_delta_p1 = 0.0
    ok_count       = 0

    for r in agent_results.get("results", []):
        if r["status"] != "OK":
            continue
        name   = r["name"]
        ag_p1  = r["pass_at_1"]
        ag_dcb = r["avg_codebleu"]["domain_codebleu"]

        zs     = zeroshot_results.get(name, {})
        zs_p1  = zs.get("pass_at_1", 0.0)
        zs_dcb = zs.get("avg_codebleu", {}).get("domain_codebleu", 0.0)

        delta        = ag_p1 - zs_p1
        total_delta_p1 += delta
        ok_count     += 1

        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(
            f"{name:<12}"
            f"{zs_p1:>10.3f}{zs_dcb:>10.4f}"
            f"  {ag_p1:>10.3f}{ag_dcb:>10.4f}"
            f"  {marker}{delta:>+8.3f}"
        )

    if ok_count:
        avg_delta = total_delta_p1 / ok_count
        print(f"{'-'*w}")
        print(f"{'AVG DELTA':<12}{'':>20}  {'':>20}  {avg_delta:>+9.3f}")
    print(f"{'='*w}\n")


# ─────────────────────── CLI ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Agent-based physics benchmark evaluator",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python evaluate_agent.py --models gemini:google/gemini-3-flash-preview
  python evaluate_agent.py --models gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o
  python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --n-samples 10 --max-iter 5
  python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --rescore pr29633
  python evaluate_agent.py --models gemini:google/gemini-3-flash-preview --rescore
        """,
    )
    p.add_argument(
        "--models", nargs="+", metavar="ALIAS:MODEL_ID",
        default = DEFAULT_MODELS,
        help    = "One or more models as alias:model/id",
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N,
        help=f"Agent runs per benchmark item (default {DEFAULT_N})",
    )
    p.add_argument(
        "--max-iter", type=int, default=DEFAULT_MAX_ITER,
        help=f"ReAct iterations per sub-task (default {DEFAULT_MAX_ITER})",
    )
    p.add_argument(
        "--rescore", nargs="*", metavar="PR",
        help="Ignore cached agent results. No args = rescore all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse rescore set
    if args.rescore is None:
        rescore_set: Union[Set[str], str] = set()
    elif len(args.rescore) == 0:
        rescore_set = "__all__"
    else:
        rescore_set = set(args.rescore)

    if not CONFIG_FILE.exists():
        print(f"[ERROR] {CONFIG_FILE} not found.")
        sys.exit(1)

    config = json.loads(CONFIG_FILE.read_text())
    items  = config["items"]
    cfgs   = [RunConfig.from_spec(s) for s in args.models]

    print("[INFO] Mode      : agent")
    print(f"[INFO] Models    : {[(c.alias, c.model_id) for c in cfgs]}")
    print(f"[INFO] Items     : {len(items)}")
    print(f"[INFO] N samples : {args.n_samples}")
    print(f"[INFO] Max iter  : {args.max_iter} ReAct iters/node")

    domain_weights = load_or_build_weights(config)
    client         = make_client()

    for cfg in cfgs:
        print(f"\n{'#'*72}")
        print(f"#  Agent model: {cfg.alias}  ({cfg.model_id})")
        print(f"{'#'*72}")

        existing_agent = _load_existing_agent_results(cfg)
        existing_zs    = load_existing_results(cfg)       # zero-shot for comparison

        all_results: dict = {
            "model":     cfg.model_id,
            "alias":     cfg.alias,
            "mode":      "agent",
            "n_samples": args.n_samples,
            "max_iter":  args.max_iter,
            "results":   [],
        }

        for item in items:
            item_name  = item.split("/")[-1]
            do_rescore = (
                rescore_set == "__all__"
                or item_name in rescore_set
            )

            res = evaluate_item_agent(
                cfg            = cfg,
                client         = client,
                item_path_str  = item,
                domain_weights = domain_weights,
                existing       = existing_agent,
                n_samples      = args.n_samples,
                max_iter       = args.max_iter,
                rescore        = do_rescore,
            )
            all_results["results"].append(res)

            out_path = _agent_results_path(cfg)
            out_path.write_text(json.dumps(all_results, indent=2))
            print(f"  [INFO] Saved → {out_path}")

        print_summary(f"{cfg.alias} [agent]", all_results)
        print_comparison(cfg, all_results, existing_zs)


if __name__ == "__main__":
    main()
