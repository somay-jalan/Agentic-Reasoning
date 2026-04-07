# orchestrator.py

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from agent.planner  import PlannerAgent
from agent.coder    import CoderAgent
from agent.critic   import CriticAgent
from agent.debugger import DebugAgent
from tools.tool_suite import ToolSuite
from react_loop import react_loop, DEFAULT_MAX_ITER
from agent.utils import strip_code_fences
import ast as _ast


class Orchestrator:
    def __init__(
        self,
        cfg,
        client:         OpenAI,
        repo_dir:       Optional[Path] = None,
        target_relpath: Optional[str]  = None,
        test_files:     Optional[List[Path]] = None,
        max_iter:       int  = DEFAULT_MAX_ITER,
        verbose:        bool = True,
        run_save_dir:   Optional[Path] = None,  # ← NEW: root save dir for this run
    ):
        self.cfg          = cfg
        self.verbose      = verbose
        self.max_iter     = max_iter
        self.run_save_dir = run_save_dir        # e.g. agent_traces/gemini/pr29550/sample_1/

        self.planner  = PlannerAgent(cfg, client)
        self.coder    = CoderAgent(cfg, client)
        self.critic   = CriticAgent(cfg, client)
        self.debugger = DebugAgent(cfg, client)

        # Full tool suite — used at assembly time only
        self.full_tool_suite = ToolSuite(
            repo_dir       = repo_dir,
            target_relpath = target_relpath,
            test_files     = test_files,
        )

    def solve(self, prompt: str) -> dict:
        out = {
            "final_code":   "",
            "tcrg":         {},
            "node_history": [],
            "n_nodes":      0,
        }

        # ── 1. Plan ────────────────────────────────────────────────
        if self.verbose:
            print(f"\n  [{self.cfg.alias}] Planner: decomposing problem ...")

        tcrg = self.planner.plan(prompt)
        out["tcrg"]    = tcrg
        out["n_nodes"] = len(tcrg.get("nodes", []))
        task_type      = tcrg.get("task_type", "feature")

        # Save TCRG immediately
        if self.run_save_dir:
            self.run_save_dir.mkdir(parents=True, exist_ok=True)
            (self.run_save_dir / "tcrg.json").write_text(
                json.dumps(tcrg, indent=2), encoding="utf-8"
            )
            (self.run_save_dir / "prompt.txt").write_text(
                prompt, encoding="utf-8"
            )

        if not tcrg.get("nodes"):
            print("  [Orchestrator] Empty TCRG — aborting.")
            return out

        if self.verbose:
            print(f"  [{self.cfg.alias}] TCRG: task_type={task_type} "
                  f"({out['n_nodes']} nodes):")
            for node in tcrg["nodes"]:
                deps = ", ".join(node["dependencies"]) or "—"
                print(f"    {node['id']}: {node['title']}  (deps: {deps})")

        # ── 2. Topological sort ────────────────────────────────────
        ordered = _topological_sort(tcrg["nodes"])

        # ── 3. Choose loop-level tool suite ────────────────────────
        # Bugfixes: no pytest inside the loop (partial files always fail)
        # Features: full suite inside the loop
        if task_type == "bugfix":
            loop_tool_suite = ToolSuite()   # syntax + AST + physics only
        else:
            loop_tool_suite = self.full_tool_suite

        # ── 4. ReAct loop per node ─────────────────────────────────
        node_outputs: Dict[str, str] = {}

        for node in ordered:
            if self.verbose:
                print(f"\n  [{self.cfg.alias}] Node {node['id']}: {node['title']}")

            context  = _build_context(node, node_outputs)

            # Per-node save directory
            node_save_dir = None
            if self.run_save_dir:
                node_save_dir = self.run_save_dir / node["id"]
                node_save_dir.mkdir(parents=True, exist_ok=True)
                # Save node spec
                (node_save_dir / "node_spec.json").write_text(
                    json.dumps(node, indent=2), encoding="utf-8"
                )
                # Save context code the Coder will receive
                if context.strip():
                    (node_save_dir / "context.py").write_text(
                        context, encoding="utf-8"
                    )

            best_code, history = react_loop(
                subtask      = node,
                context_code = context,
                coder        = self.coder,
                critic       = self.critic,
                debugger     = self.debugger,
                tool_suite   = loop_tool_suite,
                max_iter     = self.max_iter,
                verbose      = self.verbose,
                save_dir     = node_save_dir,   # ← wire through
                original_prompt  = prompt, 
            )

            node_outputs[node["id"]] = best_code

            # Save per-node summary
            if node_save_dir:
                node_summary = {
                    "node_id":         node["id"],
                    "node_title":      node["title"],
                    "n_iterations":    len(history),
                    "final_passed":    (
                        history[-1]["tool_report"].get("all_passed", False)
                        if history else False
                    ),
                    "best_code_length": len(best_code),
                    "iteration_scores": [
                        {
                            "iteration":  h["iteration"],
                            "role":       h["role"],
                            "code_length":h["code_length"],
                            "all_passed": h["tool_report"].get("all_passed", False),
                            "severity":   h["critique"].get("severity", "—"),
                            "n_fixes":    len(h["critique"].get("actionable_fixes", [])),
                        }
                        for h in history
                    ],
                }
                (node_save_dir / "node_summary.json").write_text(
                    json.dumps(node_summary, indent=2), encoding="utf-8"
                )

            out["node_history"].append({
                "node_id":    node["id"],
                "node_title": node["title"],
                "n_iters":    len(history),
                "final_passed": (
                    history[-1]["tool_report"].get("all_passed", False)
                    if history else False
                ),
                "history": history,
            })

        # ── 5. Assemble ────────────────────────────────────────────
        final_code = _assemble(ordered, node_outputs, task_type)
        out["final_code"] = final_code
        # Validate assembled file syntax before scoring
        if final_code.strip():
            try:
                _ast.parse(final_code)
                if self.verbose:
                    print(f"  [{self.cfg.alias}] Assembly syntax: ✓ OK")
            except SyntaxError as exc:
                if self.verbose:
                    print(
                        f"  [{self.cfg.alias}] Assembly syntax: ✗ FAIL "
                        f"line {exc.lineno}: {exc.msg}"
                    )
                # Save the bad assembly for inspection
                if self.run_save_dir:
                    (self.run_save_dir / "final_assembled_BROKEN.py").write_text(
                        final_code, encoding="utf-8"
                    )
                # Fall back to the last node's output directly
                for node in reversed(ordered):
                    fallback = node_outputs.get(node["id"], "").strip()
                    if fallback:
                        try:
                            _ast.parse(fallback)
                            if self.verbose:
                                print(
                                    f"  [{self.cfg.alias}] Falling back to "
                                    f"node {node['id']} output ({len(fallback)} chars)."
                                )
                            out["final_code"] = fallback
                            if self.run_save_dir:
                                (self.run_save_dir / "final_assembled.py").write_text(
                                    fallback, encoding="utf-8"
                                )
                            break
                        except SyntaxError:
                            continue
        # Save final assembled file and run summary
        if self.run_save_dir and final_code.strip():
            (self.run_save_dir / "final_assembled.py").write_text(
                final_code, encoding="utf-8"
            )
            run_summary = {
                "task_type":       task_type,
                "n_nodes":         out["n_nodes"],
                "final_code_length": len(final_code),
                "nodes": [
                    {
                        "id":           nh["node_id"],
                        "title":        nh["node_title"],
                        "n_iters":      nh["n_iters"],
                        "final_passed": nh["final_passed"],
                    }
                    for nh in out["node_history"]
                ],
            }
            (self.run_save_dir / "run_summary.json").write_text(
                json.dumps(run_summary, indent=2), encoding="utf-8"
            )

        if self.verbose:
            print(
                f"\n  [{self.cfg.alias}] Assembly complete "
                f"({len(final_code)} chars)."
            )

        return out


# ─────────────────────── Graph utilities ──────────────────────────

def _topological_sort(nodes: list) -> list:
    """Kahn's algorithm — falls back to original order on cycle detection."""
    id_to_node = {n["id"]: n for n in nodes}
    in_degree  = {n["id"]: 0 for n in nodes}
    children   = {n["id"]: [] for n in nodes}

    for node in nodes:
        for dep in node.get("dependencies", []):
            if dep in children:
                children[dep].append(node["id"])
                in_degree[node["id"]] += 1

    queue  = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        nid = queue.pop(0)
        result.append(id_to_node[nid])
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(nodes):
        print("  [Orchestrator] WARNING: cycle in TCRG — using original node order.")
        return nodes

    return result


def _build_context(node: dict, node_outputs: Dict[str, str]) -> str:
    """Concatenate accepted code from all declared dependency nodes."""
    parts = []
    for dep_id in node.get("dependencies", []):
        code = strip_code_fences(node_outputs.get(dep_id, "").strip())
        if code:
            parts.append(f"# ── output of sub-task {dep_id} ──────────────\n{code}")
    return "\n\n".join(parts)


# orchestrator.py

def _is_complete_file(code: str) -> bool:
    """
    Returns True if the code looks like a complete Python file
    rather than a fragment (a few functions or a class addition).

    Heuristics:
      - Has a module-level docstring or multiple import blocks
      - Has more than N lines
      - Contains class definitions that appear in multiple nodes
    """
    if not code.strip():
        return False

    lines = code.strip().splitlines()

    # A fragment is typically short
    if len(lines) < 50:
        return False

    # Count top-level constructs — a fragment has very few
    import_count = sum(
        1 for l in lines
        if l.startswith("import ") or l.startswith("from ")
    )
    class_count = sum(
        1 for l in lines
        if l.startswith("class ")
    )
    def_count = sum(
        1 for l in lines
        if l.startswith("def ")
    )

    # A complete file has many imports, many classes/functions
    # A fragment has 0-2 imports and 1-3 top-level definitions
    if import_count >= 5 and (class_count + def_count) >= 5:
        return True

    return False


def _assemble(
    ordered_nodes: list,
    node_outputs:  Dict[str, str],
    task_type:     str = "feature",
) -> str:
    # Bugfix: always return the last node's complete file
    if task_type == "bugfix":
        for node in reversed(ordered_nodes):
            code = node_outputs.get(node["id"], "").strip()
            if code:
                return code
        return ""

    # Feature: check if nodes are outputting complete files or fragments
    non_empty = [
        node_outputs.get(n["id"], "").strip()
        for n in ordered_nodes
        if node_outputs.get(n["id"], "").strip()
    ]

    if not non_empty:
        return ""

    # If any node output looks like a complete file, use the last
    # complete file — don't concatenate
    complete_file_outputs = [c for c in non_empty if _is_complete_file(c)]

    if complete_file_outputs:
        # The last node's complete file is the most up-to-date version
        return complete_file_outputs[-1]

    # True fragment assembly — original logic for when nodes
    # really do output only their piece
    import_lines:  List[str] = []
    body_sections: List[str] = []

    for node in ordered_nodes:
        code = node_outputs.get(node["id"], "").strip()
        if not code:
            continue

        lines   = code.splitlines()
        imports = [l for l in lines if l.startswith(("import ", "from "))]
        body    = [l for l in lines if not l.startswith(("import ", "from "))]

        import_lines.extend(imports)
        body_sections.append(
            f"# {'─'*58}\n"
            f"# Node {node['id']}: {node['title']}\n"
            f"# {'─'*58}\n"
            + "\n".join(body)
        )

    seen_imports:   set       = set()
    unique_imports: List[str] = []
    for line in import_lines:
        if line not in seen_imports:
            seen_imports.add(line)
            unique_imports.append(line)

    header = "\n".join(unique_imports)
    body   = "\n\n".join(body_sections)
    return f"{header}\n\n{body}".strip()
