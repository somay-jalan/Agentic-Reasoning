# evaluate.py
"""
Multi-model evaluator. Evaluates one or more LLMs via OpenRouter on each PR prompt.

CLI examples
────────────
# Run two models with friendly aliases
python evaluate.py --models gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o

# Only recompute BLEU scores from already-cached samples (no API, no pytest)
python evaluate.py --models gemini:google/gemini-3-flash-preview --bleu-only

# Rescore specific PRs (re-run pytest, ignore results JSON) for all listed models
python evaluate.py --models gemini:google/gemini-3-flash-preview --rescore pr29633 pr29550

# Rescore everything
python evaluate.py --models gemini:google/gemini-3-flash-preview --rescore

Cache layout
────────────
  cache/
    <alias>/                        ← e.g. "gemini", "gpt4o"
      <pr_name>/
        sample_0.py
        ...
        sample_9.py
        metadata.json

Results files
─────────────
  <alias>_eval_results.json         ← e.g. gemini_eval_results.json
"""

import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Union

from openai import OpenAI
from codebleu import calc_codebleu
from dotenv import load_dotenv
import argparse

# ─────────────────────────── CONFIG ───────────────────────────────
load_dotenv()

ROOT        = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "manual_corpus/benchmark_config.json"
WEIGHT_FILE = ROOT / "domain_weights.json"
CACHE_DIR   = ROOT / "cache"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
N_SAMPLES          = 10

HTTP_REFERER = "https://github.com/your-repo"
APP_TITLE    = "Physics-Benchmark-Eval"

# Default model when none is specified on the CLI
DEFAULT_MODELS = ["gemini:google/gemini-3-flash-preview"]


# ──────────────────────── RUN CONFIG ──────────────────────────────

@dataclass
class RunConfig:
    """Everything that varies between models lives here."""
    model_id:     str          # OpenRouter model string
    alias:        str          # friendly name used for cache/results naming
    cache_dir:    Path         # ROOT/cache/<alias>
    results_file: Path         # ROOT/<alias>_eval_results.json

    @classmethod
    def from_spec(cls, spec: str) -> "RunConfig":
        """
        Parse 'alias:model/id' or bare 'model/id' (alias = last path component).
        Examples:
            gemini:google/gemini-3-flash-preview
            openai/gpt-4o                         → alias 'gpt-4o'
        """
        if ":" in spec:
            alias, model_id = spec.split(":", 1)
        else:
            model_id = spec
            alias    = spec.split("/")[-1]

        alias = alias.strip()
        return cls(
            model_id     = model_id.strip(),
            alias        = alias,
            cache_dir    = CACHE_DIR / alias,
            results_file = ROOT / f"{alias}_eval_results.json",
        )


# ─────────────────────────── CACHE ────────────────────────────────

def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def load_existing_results(cfg: RunConfig) -> dict:
    if not cfg.results_file.exists():
        return {}
    data     = json.loads(cfg.results_file.read_text())
    existing = {}
    for r in data.get("results", []):
        if r.get("status") == "OK":
            existing[r["name"]] = r
    print(f"[{cfg.alias}] Loaded existing results for: {list(existing.keys())}")
    return existing


def get_scored_samples(existing: dict, name: str) -> List[Optional[dict]]:
    if name not in existing:
        return [None] * N_SAMPLES
    per_sample = existing[name].get("per_sample", [])
    scored     = {s["sample"]: s for s in per_sample}
    return [scored.get(i) for i in range(N_SAMPLES)]


def cache_dir_for(cfg: RunConfig, name: str) -> Path:
    d = cfg.cache_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached_samples(
    cfg: RunConfig, name: str, prompt: str, n: int
) -> Tuple[List[Optional[str]], bool]:
    d      = cache_dir_for(cfg, name)
    meta_p = d / "metadata.json"
    phash  = prompt_hash(prompt)

    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("prompt_hash") != phash or meta.get("model") != cfg.model_id:
            print(f"  [{cfg.alias}] Prompt/model changed for {name} — invalidating cache.")
            for f in d.glob("sample_*.py"):
                f.unlink()
            meta_p.unlink()

    samples = []
    all_hit = True
    for i in range(n):
        p = d / f"sample_{i}.py"
        if p.exists():
            samples.append(p.read_text(encoding="utf-8"))
        else:
            samples.append(None)
            all_hit = False

    return samples, all_hit


def save_sample_to_cache(cfg: RunConfig, name: str, index: int, code: str, prompt: str):
    d = cache_dir_for(cfg, name)
    (d / f"sample_{index}.py").write_text(code, encoding="utf-8")
    meta_p = d / "metadata.json"
    meta   = {
        "model":       cfg.model_id,
        "alias":       cfg.alias,
        "prompt_hash": prompt_hash(prompt),
        "n_samples":   N_SAMPLES,
        "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_p.write_text(json.dumps(meta, indent=2))


def show_cache_status(config: dict, cfgs: List[RunConfig]):
    print("\n── Cache status ──────────────────────────────────────────────")
    for cfg in cfgs:
        print(f"  Model alias: {cfg.alias}  ({cfg.model_id})")
        for item_str in config["items"]:
            item_dir = ROOT / "manual_corpus" / item_str
            meta     = json.loads((item_dir / "metadata.json").read_text())
            name     = meta["name"]
            d        = cfg.cache_dir / name
            if not d.exists():
                print(f"    {name:<20}  no cache")
                continue
            found = len(list(d.glob("sample_*.py")))
            print(f"    {name:<20}  {found}/{N_SAMPLES} samples cached")
    print("──────────────────────────────────────────────────────────────\n")


# ─────────────────────── OPENROUTER CLIENT ────────────────────────

def make_client() -> OpenAI:
    return OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key  = OPENROUTER_API_KEY,
    )


def openrouter_generate(cfg: RunConfig, client: OpenAI, prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model    = cfg.model_id,
                messages = [
                    {
                        "role":    "system",
                        "content": (
                            "You are an expert Python developer. "
                            "When asked to fix or implement code, output ONLY "
                            "a single Python code block and nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature = 0.8,
                max_tokens  = 65536,
                extra_headers = {
                    "HTTP-Referer": HTTP_REFERER,
                    "X-Title":      APP_TITLE,
                },
                extra_body={
                    "provider": {
                    "sort": "throughput"
                    }
                },
            )
            raw = response.choices[0].message.content or ""
            return extract_python_code(raw)

        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [WARN] OpenRouter call failed (attempt {attempt+1}/{retries}): {exc}")
            print(f"         Retrying in {wait}s ...")
            time.sleep(wait)

    print("  [ERROR] All retries exhausted — returning empty string.")
    return ""


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ──────────────────── DOMAIN WEIGHTS (AST-based) ──────────────────

CATEGORY_BASE_WEIGHT = {
    "class_instantiation": 3.0,
    "method_call":         2.5,
    "attribute_access":    2.0,
    "function_call":       2.0,
    "exception_type":      1.8,
    "decorator":           1.5,
    "import_name":         1.3,
    "comparison_target":   1.2,
    "name_load":           1.0,
    "string_constant":     0.5,
}


class DomainTokenExtractor(ast.NodeVisitor):
    def __init__(self):
        self.tokens: List[Tuple[str, str]] = []

    def visit_Import(self, node):
        for alias in node.names:
            self.tokens.append((alias.name.split(".")[0], "import_name"))
            if alias.asname:
                self.tokens.append((alias.asname, "import_name"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.tokens.append((node.module.split(".")[0], "import_name"))
        for alias in node.names:
            self.tokens.append((alias.name, "import_name"))
            if alias.asname:
                self.tokens.append((alias.asname, "import_name"))
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            cat = "class_instantiation" if func.id[0].isupper() else "function_call"
            self.tokens.append((func.id, cat))
        elif isinstance(func, ast.Attribute):
            self.tokens.append((func.attr, "method_call"))
            if isinstance(func.value, ast.Name):
                self.tokens.append((func.value.id, "attribute_access"))
            elif isinstance(func.value, ast.Attribute):
                self.tokens.append((func.value.attr, "attribute_access"))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.tokens.append((node.attr, "attribute_access"))
            if isinstance(node.value, ast.Name):
                self.tokens.append((node.value.id, "attribute_access"))
        self.generic_visit(node)

    def visit_Raise(self, node):
        if node.exc:
            exc = node.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                self.tokens.append((exc.func.id, "exception_type"))
            elif isinstance(exc, ast.Name):
                self.tokens.append((exc.id, "exception_type"))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type:
            if isinstance(node.type, ast.Name):
                self.tokens.append((node.type.id, "exception_type"))
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        self.tokens.append((elt.id, "exception_type"))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self.tokens.append((dec.id, "decorator"))
            elif isinstance(dec, ast.Attribute):
                self.tokens.append((dec.attr, "decorator"))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self.tokens.append((dec.id, "decorator"))
            elif isinstance(dec, ast.Attribute):
                self.tokens.append((dec.attr, "decorator"))
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.tokens.append((base.id, "class_instantiation"))
        self.generic_visit(node)

    def visit_Assert(self, node):
        if isinstance(node.test, ast.Compare):
            for comp in node.test.comparators:
                for child in ast.walk(comp):
                    if isinstance(child, ast.Name):
                        self.tokens.append((child.id, "comparison_target"))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and len(node.id) > 1:
            self.tokens.append((node.id, "name_load"))
        self.generic_visit(node)


def extract_tokens(source: str) -> List[Tuple[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ext = DomainTokenExtractor()
    ext.visit(tree)
    return ext.tokens


def build_domain_weights(ref_files: List[Path]) -> Dict[str, float]:
    from collections import Counter
    N          = len(ref_files)
    doc_weighted: List[Dict[str, float]] = []
    doc_totals:   List[float]            = []
    doc_freq:     Counter                = Counter()

    for path in ref_files:
        source   = path.read_text(encoding="utf-8")
        tokens   = extract_tokens(source)
        weighted: Dict[str, float] = defaultdict(float)
        total    = 0.0
        seen     = set()
        for (tok, cat) in tokens:
            w = CATEGORY_BASE_WEIGHT.get(cat, 1.0)
            weighted[tok] += w
            total += w
            seen.add(tok)
        for tok in seen:
            doc_freq[tok] += 1
        doc_weighted.append(dict(weighted))
        doc_totals.append(max(total, 1.0))

    raw: Dict[str, float] = {}
    for tok in doc_freq:
        idf     = math.log((N + 1) / (doc_freq[tok] + 1)) + 1.0
        max_tfi = max(
            doc_weighted[d].get(tok, 0.0) / doc_totals[d] * idf
            for d in range(N)
        )
        raw[tok] = max_tfi

    if not raw:
        return {}
    v_min = min(raw.values())
    v_max = max(raw.values())
    span  = v_max - v_min if v_max != v_min else 1.0
    return {
        tok: round(0.1 + (v - v_min) / span * 2.9, 4)
        for tok, v in raw.items()
    }


def load_or_build_weights(config: dict) -> Dict[str, float]:
    if WEIGHT_FILE.exists():
        print(f"[INFO] Loading domain weights from {WEIGHT_FILE}")
        return json.loads(WEIGHT_FILE.read_text())

    print("[INFO] Building domain weights from reference files ...")
    ref_files = []
    for item_str in config["items"]:
        item_dir = ROOT / "manual_corpus" / item_str
        meta     = json.loads((item_dir / "metadata.json").read_text())
        after_f  = item_dir / meta["after_file"]
        if after_f.exists():
            ref_files.append(after_f)

    weights = build_domain_weights(ref_files)
    WEIGHT_FILE.write_text(json.dumps(weights, indent=2, sort_keys=True))
    print(f"[INFO] Saved {len(weights)} weights → {WEIGHT_FILE}")
    return weights


# ──────────────────── DOMAIN CODEBLEU ─────────────────────────────

def domain_weighted_codebleu(
    hypothesis:     str,
    reference:      str,
    domain_weights: Dict[str, float],
    passed:         Optional[bool] = None,
) -> dict:
    empty = {
        "codebleu":             0.0,
        "ngram_match":          0.0,
        "weighted_ngram_match": 0.0,
        "syntax_match":         0.0,
        "dataflow_match":       0.0,
        "domain_delta":         0.0,
        "domain_codebleu_raw":  0.0,
        "execution_penalty":    1.0,
        "domain_codebleu":      0.0,
        "n_domain_tokens":      0,
    }
    if not hypothesis.strip():
        return empty

    try:
        result = calc_codebleu(
            references  = [reference],
            predictions = [hypothesis],
            lang        = "python",
            weights     = (0.25, 0.25, 0.25, 0.25),
        )
    except Exception as e:
        print(f"  [WARN] calc_codebleu failed: {e}")
        return empty

    base_cb  = result["codebleu"]
    tokens   = extract_tokens(hypothesis)
    tok_w    = [domain_weights.get(tok, 1.0) for (tok, _) in tokens]
    n_tokens = len(tok_w)

    if tok_w:
        hyp_mean    = sum(tok_w) / n_tokens
        global_mean = sum(domain_weights.values()) / max(len(domain_weights), 1)
        global_std  = (
            sum((v - global_mean) ** 2 for v in domain_weights.values())
            / max(len(domain_weights), 1)
        ) ** 0.5
        raw_delta    = (hyp_mean - global_mean) / max(global_std, 1e-9)
        clamped      = max(-1.0, min(1.0, raw_delta))
        domain_delta = clamped * 0.15
    else:
        domain_delta = 0.0

    domain_raw   = max(0.0, min(1.0, base_cb + domain_delta))
    exec_penalty = 1.0
    domain_final = round(domain_raw * exec_penalty, 4)

    return {
        "codebleu":             round(base_cb, 4),
        "ngram_match":          round(result.get("ngram_match_score", 0), 4),
        "weighted_ngram_match": round(result.get("weighted_ngram_match_score", 0), 4),
        "syntax_match":         round(result.get("syntax_match_score", 0), 4),
        "dataflow_match":       round(result.get("dataflow_match_score", 0), 4),
        "domain_delta":         round(domain_delta, 4),
        "domain_codebleu_raw":  round(domain_raw, 4),
        "execution_penalty":    exec_penalty,
        "domain_codebleu":      domain_final,
        "n_domain_tokens":      n_tokens,
    }


# ───────────────────── PASS@K ESTIMATOR ───────────────────────────

def pass_at_k(n: int, c: int, k: int = 1) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ───────────────────── REPO HELPERS ───────────────────────────────

def run_cmd(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def clone_and_install(repo_url: str, checkout: str, workdir: Path):
    repo_dir = workdir / "repo"
    r = run_cmd(["git", "clone", "--depth=1", repo_url, str(repo_dir)])
    if r.returncode != 0:
        return None, f"clone failed: {r.stderr[:300]}"
    if checkout and checkout not in ("master", "main"):
        r2 = run_cmd(["git", "checkout", checkout], cwd=repo_dir)
        if r2.returncode != 0:
            return None, f"checkout failed: {r2.stderr[:300]}"
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
             "pip", "setuptools", "wheel"])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    r3 = run_cmd(
        [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
        cwd=repo_dir,
    )
    if r3.returncode != 0:
        return None, f"pip install -e . failed: {r3.stderr[:300]}"
    return repo_dir, None


def load_test_files(item_dir: Path, meta: dict) -> List[Path]:
    if "test_files" in meta:
        paths = [item_dir / f for f in meta["test_files"]]
    elif "test_file" in meta:
        paths = [item_dir / meta["test_file"]]
    else:
        raise KeyError("metadata.json must have 'test_file' or 'test_files'")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Test files not found: {missing}")
    return paths


def parse_failed_tests(pytest_stdout: str, source_file: str = "unknown") -> List[dict]:
    failures    = []
    test_blocks = re.split(r"_{5,}\s+(\S+)\s+_{5,}", pytest_stdout)
    i = 1
    while i + 1 < len(test_blocks):
        test_name = test_blocks[i].strip()
        body      = test_blocks[i + 1]
        i        += 2
        if test_name in ("short test summary info",):
            continue
        loc_match  = re.search(
            r"([\w/]+\.py):(\d+):\s*(\w+(?:Error|Exception|Warning))", body)
        location   = (f"{loc_match.group(1)}:{loc_match.group(2)}"
                      if loc_match else "unknown")
        error_type = loc_match.group(3) if loc_match else "unknown"
        executed   = re.findall(r"^>\s+(.+)$", body, re.MULTILINE)
        assertion  = executed[0].strip() if executed else ""
        e_lines    = re.findall(r"^E\s+(.+)$", body, re.MULTILINE)
        explanation= "\n".join(e_lines[:6])
        lines      = [l for l in body.splitlines() if l.strip()]
        snippet    = "\n".join(lines[-5:]) if lines else ""
        failures.append({
            "test_file":   source_file,
            "test_name":   test_name,
            "assertion":   assertion,
            "location":    location,
            "error_type":  error_type,
            "explanation": explanation,
            "snippet":     snippet,
        })
    return failures


def _collection_error(stdout: str, source_file: str) -> List[dict]:
    error_lines = []
    capture     = False
    for line in stdout.splitlines():
        if "ERROR collecting" in line:
            capture = True
        if capture:
            error_lines.append(line)
        if capture and line.strip() == "":
            break
    return [{
        "test_file":   source_file,
        "test_name":   "COLLECTION_ERROR",
        "assertion":   "pytest could not import/collect the test file",
        "location":    "unknown",
        "error_type":  "ImportError/CollectionError",
        "explanation": "\n".join(error_lines[:8]),
        "snippet":     "",
    }]


def format_failures(failures: List[dict], indent: str = "      ") -> str:
    if not failures:
        return f"{indent}(no structured failure info parsed)"
    lines        = []
    current_file = None
    for f in failures:
        tf = f.get("test_file", "unknown")
        if tf != current_file:
            lines.append(f"{indent}── {tf} {'─'*(40-len(tf))}")
            current_file = tf
        lines.append(
            f"{indent}  ✗ {f['test_name']}  [{f['error_type']} @ {f['location']}]")
        if f["assertion"]:
            lines.append(f"{indent}    → {f['assertion']}")
        for el in f["explanation"].splitlines()[:4]:
            lines.append(f"{indent}      {el}")
    return "\n".join(lines)


def run_test_with_candidate(
    repo_dir:       Path,
    target_relpath: str,
    candidate_code: str,
    test_files:     List[Path],
) -> Tuple[bool, str, List[dict]]:
    try:
        ast.parse(candidate_code)
    except SyntaxError as e:
        msg = f"SyntaxError: {e}"
        print(f"      [SYNTAX ERROR] line {e.lineno}: {e.msg}")
        return False, msg, [{
            "test_name":   "SYNTAX_CHECK",
            "test_file":   "n/a",
            "assertion":   str(e),
            "location":    f"line {e.lineno}",
            "error_type":  "SyntaxError",
            "explanation": (candidate_code.splitlines()[max(0, (e.lineno or 1) - 1)]
                            if candidate_code else ""),
            "snippet":     "",
        }]

    target = repo_dir / target_relpath
    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, backup)

    all_passed   = True
    all_stdout   = []
    all_failures = []

    try:
        target.write_text(candidate_code, encoding="utf-8")
        env      = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(repo_dir) + os.pathsep + existing if existing else str(repo_dir)
        )

        for test_file in test_files:
            print(f"        running {test_file.name} ...", end=" ", flush=True)
            r = run_cmd(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                cwd=ROOT, env=env,
            )
            file_passed  = r.returncode == 0
            all_passed   = all_passed and file_passed
            all_stdout.append(f"\n{'='*60}\n{test_file.name}\n{'='*60}\n{r.stdout}")

            if file_passed:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                failures = parse_failed_tests(r.stdout, source_file=test_file.name)
                if not failures and "ERROR collecting" in r.stdout:
                    failures = _collection_error(r.stdout, test_file.name)
                all_failures.extend(failures)

        return all_passed, "\n".join(all_stdout), all_failures

    except Exception as e:
        return False, str(e), [{
            "test_name":  "EXCEPTION",
            "test_file":  "n/a",
            "assertion":  str(e),
            "location":   "",
            "error_type": type(e).__name__,
            "explanation":"",
            "snippet":    "",
        }]
    finally:
        shutil.move(backup, target)


# ────────────────── BLEU-ONLY RECALCULATION ───────────────────────

def recalculate_bleu_only(
    cfg:            RunConfig,
    item_path_str:  str,
    domain_weights: Dict[str, float],
    existing:       dict,
) -> dict:
    """
    Reads samples directly from cache (no API calls, no pytest).
    Preserves existing pass/fail results if available; marks as None otherwise.
    Use this when you just want to update CodeBLEU numbers after changing weights.
    """
    item_dir  = ROOT / "manual_corpus" / item_path_str
    meta      = json.loads((item_dir / "metadata.json").read_text())
    prompt    = (item_dir / "prompt.txt").read_text().strip()
    reference = (item_dir / meta["after_file"]).read_text()
    name      = meta["name"]

    print(f"\n[{cfg.alias}] BLEU-only recalc: {name}")

    cached_samples, _ = load_cached_samples(cfg, name, prompt, N_SAMPLES)

    # Carry over existing pass/fail results if present
    old_per_sample = {}
    if name in existing:
        for s in existing[name].get("per_sample", []):
            old_per_sample[s["sample"]] = s

    generations:     list = []
    codebleu_scores: list = []
    pass_results:    list = []

    for i, code in enumerate(cached_samples):
        if code is None:
            print(f"  sample {i+1:>2}  [NO CACHE] — skipping")
            # Insert a zero-score placeholder so indexing stays consistent
            cb = domain_weighted_codebleu("", reference, domain_weights)
            passed = old_per_sample.get(i, {}).get("passed", None)
            generations.append({
                "sample":      i,
                "source":      "missing",
                "code_length": 0,
                "passed":      passed,
                "failures":    [],
                "pytest_tail": "",
                "codebleu":    cb,
                "bleu_only":   True,
            })
            pass_results.append(False)
            codebleu_scores.append(cb)
            continue

        # Reuse existing pass result if available
        passed = old_per_sample.get(i, {}).get("passed", None)
        cb     = domain_weighted_codebleu(code, reference, domain_weights, passed=passed)

        pass_tag = "✓" if passed else ("✗" if passed is False else "?")
        print(
            f"  sample {i+1:>2}  {pass_tag}"
            f"  codebleu={cb['codebleu']:.4f}"
            f"  delta={cb['domain_delta']:+.4f}"
            f"  domain_codebleu={cb['domain_codebleu']:.4f}"
            f"  (pass={'kept' if passed is not None else 'unknown'})"
        )

        codebleu_scores.append(cb)
        pass_results.append(passed if passed is not None else False)
        generations.append({
            **old_per_sample.get(i, {}),
            "sample":    i,
            "codebleu":  cb,
            "bleu_only": True,
        })

    n_passed = sum(1 for p in pass_results if p)
    p1       = pass_at_k(N_SAMPLES, n_passed, k=1)
    avg_cb   = {
        key: round(sum(s[key] for s in codebleu_scores) / N_SAMPLES, 4)
        for key in codebleu_scores[0]
    }

    print(
        f"  ► {name}  pass@1={p1:.3f}  "
        f"avg_domain_codebleu={avg_cb['domain_codebleu']:.4f}"
        f"  (BLEU-only, no pytest ran)"
    )

    return {
        "item":         item_path_str,
        "name":         name,
        "model":        cfg.model_id,
        "alias":        cfg.alias,
        "status":       "OK",
        "n_samples":    N_SAMPLES,
        "n_passed":     n_passed,
        "pass_at_1":    round(p1, 4),
        "avg_codebleu": avg_cb,
        "per_sample":   generations,
    }


# ──────────────────────── PER-ITEM EVAL ───────────────────────────

def evaluate_item(
    cfg:            RunConfig,
    client:         OpenAI,
    item_path_str:  str,
    domain_weights: Dict[str, float],
    existing:       dict,
    rescore:        bool = False,
) -> dict:
    print("\n" + "=" * 70)
    flag = " [RESCORING]" if rescore else ""
    print(f"[{cfg.alias}] Evaluating: {item_path_str}{flag}")

    item_dir       = ROOT / "manual_corpus" / item_path_str
    meta           = json.loads((item_dir / "metadata.json").read_text())
    prompt         = (item_dir / "prompt.txt").read_text().strip()
    reference      = (item_dir / meta["after_file"]).read_text()
    test_files     = load_test_files(item_dir, meta)
    name           = meta["name"]
    repo_url       = meta["repo_url"]
    checkout       = meta.get("checkout", "master")
    target_relpath = meta["target_relpath"]

    cached_samples, _ = load_cached_samples(cfg, name, prompt, N_SAMPLES)
    scored_samples    = (
        [None] * N_SAMPLES
        if rescore
        else get_scored_samples(existing, name)
    )

    needs_scoring = [i for i in range(N_SAMPLES) if scored_samples[i] is None]
    needs_api     = [i for i in range(N_SAMPLES) if cached_samples[i] is None]

    print(f"  Samples needing API call : {needs_api or 'none'}")
    print(f"  Samples needing scoring  : {needs_scoring or 'none'}")

    if not needs_api and not needs_scoring:
        print(f"  [SKIP] All {N_SAMPLES} samples already scored.")
        r = existing[name]
        for s in r["per_sample"]:
            status_str = "✓ PASS" if s["passed"] else "✗ FAIL"
            cb         = s["codebleu"]
            print(
                f"    [results✓] sample {s['sample']+1:>2}  {status_str}"
                f"  codebleu={cb['codebleu']:.4f}"
                f"  domain_codebleu={cb['domain_codebleu']:.4f}"
            )
        return r

    # Fetch any missing API samples
    samples: List[str] = []
    for i, cached in enumerate(cached_samples):
        if cached is not None:
            samples.append(cached)
        else:
            print(f"  Calling API for sample {i+1}/{N_SAMPLES} ...", end=" ", flush=True)
            code = openrouter_generate(cfg, client, prompt)
            save_sample_to_cache(cfg, name, i, code, prompt)
            samples.append(code)
            print("done")
            time.sleep(1.5)

    print(f"  Cloning {repo_url} for {len(needs_scoring)} sample(s) ...")
    generations:     list = []
    codebleu_scores: list = []
    pass_results:    list = []

    with tempfile.TemporaryDirectory(prefix=f"{name}_{cfg.alias}_") as tmp:
        workdir  = Path(tmp)
        repo_dir, err = clone_and_install(repo_url, checkout, workdir)

        if repo_dir is None:
            print(f"  [ERROR] Repo setup failed: {err}")
            return {
                "item":   item_path_str,
                "name":   name,
                "status": "ERROR",
                "error":  err,
            }

        for i, code in enumerate(samples):
            if scored_samples[i] is not None:
                s          = scored_samples[i]
                passed     = s["passed"]
                cb         = s["codebleu"]
                pass_results.append(passed)
                codebleu_scores.append(cb)
                generations.append(s)
                status_str = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"    [results✓] sample {i+1:>2}  {status_str}"
                    f"  codebleu={cb['codebleu']:.4f}"
                    f"  domain_codebleu={cb['domain_codebleu']:.4f}"
                    f"  (from results JSON)"
                )
                continue

            if code.strip():
                passed, pytest_out, failures = run_test_with_candidate(
                    repo_dir, target_relpath, code, test_files
                )
            else:
                passed, pytest_out, failures = False, "empty generation", []

            cb = domain_weighted_codebleu(code, reference, domain_weights, passed=passed)
            pass_results.append(passed)
            codebleu_scores.append(cb)

            src        = "cache" if cached_samples[i] is not None else "api  "
            status_str = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"    [{src}] sample {i+1:>2}  {status_str}"
                f"  codebleu={cb['codebleu']:.4f}"
                f"  delta={cb['domain_delta']:+.4f}"
                f"  domain_codebleu={cb['domain_codebleu']:.4f}"
            )

            if not passed:
                if failures:
                    print(format_failures(failures))
                else:
                    for line in pytest_out.splitlines():
                        if "FAILED" in line or "ERROR" in line:
                            print(f"      {line.strip()}")

            generations.append({
                "sample":      i,
                "source":      src.strip(),
                "code_length": len(code),
                "passed":      passed,
                "failures":    failures,
                "pytest_tail": pytest_out[-500:] if pytest_out else "",
                "codebleu":    cb,
            })

    n_passed = sum(pass_results)
    p1       = pass_at_k(N_SAMPLES, n_passed, k=1)
    avg_cb   = {
        key: round(sum(s[key] for s in codebleu_scores) / N_SAMPLES, 4)
        for key in codebleu_scores[0]
    }

    print(
        f"\n  ► {name}  pass@1={p1:.3f}  "
        f"avg_domain_codebleu={avg_cb['domain_codebleu']:.4f}  "
        f"({n_passed}/{N_SAMPLES} passed)"
    )

    return {
        "item":         item_path_str,
        "name":         name,
        "model":        cfg.model_id,
        "alias":        cfg.alias,
        "status":       "OK",
        "n_samples":    N_SAMPLES,
        "n_passed":     n_passed,
        "pass_at_1":    round(p1, 4),
        "avg_codebleu": avg_cb,
        "per_sample":   generations,
    }


# ─────────────────────── SUMMARY TABLE ────────────────────────────

def print_summary(alias: str, all_results: dict):
    print("\n" + "=" * 72)
    print(f"  Model: {alias}")
    print(f"{'PR':<12} {'Pass@1':>8} {'CodeBLEU':>10} {'Domain-CB':>12} {'Boost':>8} {'Pass':>8}")
    print("-" * 72)

    total_p1 = total_cb = total_dcb = 0.0
    ok_count = 0

    for r in all_results["results"]:
        if r["status"] != "OK":
            print(f"{r['name']:<12}  ERROR: {r.get('error', '?')}")
            continue
        ok_count += 1
        p1  = r["pass_at_1"]
        cb  = r["avg_codebleu"]["codebleu"]
        dcb = r["avg_codebleu"]["domain_codebleu"]
        bst = r["avg_codebleu"].get("domain_boost", r["avg_codebleu"].get("domain_delta", 0))
        n   = r["n_passed"]
        total_p1 += p1; total_cb += cb; total_dcb += dcb
        print(f"{r['name']:<12}{p1:>8.3f}{cb:>10.4f}{dcb:>12.4f}{bst:>8.4f}  {n}/{N_SAMPLES}")

    if ok_count:
        print("-" * 72)
        print(
            f"{'AVERAGE':<12}"
            f"{total_p1/ok_count:>8.3f}"
            f"{total_cb/ok_count:>10.4f}"
            f"{total_dcb/ok_count:>12.4f}"
        )
    print("=" * 72)


# ────────────────────────────── MAIN ──────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-model physics coding benchmark evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # run two models
  python evaluate.py --models gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o

  # only recompute BLEU from cached samples (no API calls, no pytest)
  python evaluate.py --models gemini:google/gemini-3-flash-preview --bleu-only

  # rescore specific PRs (re-run pytest, ignore results JSON)
  python evaluate.py --models gemini:google/gemini-3-flash-preview --rescore pr29633 pr29550

  # rescore everything for all listed models
  python evaluate.py --models gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o --rescore
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        metavar="ALIAS:MODEL_ID",
        default=DEFAULT_MODELS,
        help=(
            "One or more models to evaluate, each as 'alias:model/id'. "
            "The alias is used for cache directory and results file naming. "
            "Example: gemini:google/gemini-3-flash-preview gpt4o:openai/gpt-4o"
        ),
    )

    parser.add_argument(
        "--rescore",
        nargs="*",
        metavar="PR",
        help=(
            "Re-run pytest and recompute scores, ignoring the results JSON. "
            "Omit PR names to rescore everything: --rescore "
            "Or name specific items: --rescore pr29633 pr29550"
        ),
    )

    parser.add_argument(
        "--bleu-only",
        action="store_true",
        help=(
            "Skip all API calls and pytest. Read cached samples and recompute "
            "CodeBLEU scores only. Existing pass/fail results are preserved."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate mutual exclusivity ────────────────────────────────
    if args.bleu_only and args.rescore is not None:
        print("[ERROR] --bleu-only and --rescore are mutually exclusive.")
        sys.exit(1)

    # ── Parse rescore set ──────────────────────────────────────────
    if args.rescore is None:
        rescore_set: Union[Set[str], str] = set()
    elif len(args.rescore) == 0:
        rescore_set = "__all__"
    else:
        rescore_set = set(args.rescore)

    if rescore_set == "__all__":
        print("[INFO] --rescore: will rescore ALL items.")
    elif rescore_set:
        print(f"[INFO] --rescore: will rescore {rescore_set}")

    if args.bleu_only:
        print("[INFO] --bleu-only: recomputing CodeBLEU from cache, no pytest.")

    # ── Load benchmark config ──────────────────────────────────────
    if not CONFIG_FILE.exists():
        print(f"[ERROR] Missing {CONFIG_FILE}")
        sys.exit(1)

    config = json.loads(CONFIG_FILE.read_text())
    items  = config["items"]

    # ── Parse model specs ──────────────────────────────────────────
    cfgs = [RunConfig.from_spec(s) for s in args.models]

    print(f"[INFO] Models  : {[(c.alias, c.model_id) for c in cfgs]}")
    print(f"[INFO] Items   : {len(items)}")
    print(f"[INFO] Samples : {N_SAMPLES} per item")

    show_cache_status(config, cfgs)

    domain_weights = load_or_build_weights(config)
    client         = make_client()

    # ── Evaluate each model ────────────────────────────────────────
    for cfg in cfgs:
        print(f"\n{'#'*72}")
        print(f"#  Model: {cfg.alias}  ({cfg.model_id})")
        print(f"{'#'*72}")

        existing    = load_existing_results(cfg)
        all_results = {
            "model":   cfg.model_id,
            "alias":   cfg.alias,
            "n":       N_SAMPLES,
            "results": [],
        }

        for item in items:
            item_name  = item.split("/")[-1]
            do_rescore = (rescore_set == "__all__" or item_name in rescore_set)

            if args.bleu_only:
                res = recalculate_bleu_only(cfg, item, domain_weights, existing)
            else:
                res = evaluate_item(
                    cfg, client, item, domain_weights, existing, rescore=do_rescore
                )

            all_results["results"].append(res)
            cfg.results_file.write_text(json.dumps(all_results, indent=2))
            print(f"  [INFO] Saved → {cfg.results_file}")

        print_summary(cfg.alias, all_results)


if __name__ == "__main__":
    main()