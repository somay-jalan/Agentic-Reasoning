# react_loop.py
"""
ReAct loop for a single TCRG sub-task.

Each iteration follows the cycle:
    THINK  →  ACT (generate / debug)  →  OBSERVE (run tools)  →  THINK ...

The loop exits early when all tool checks pass, or exhausts its budget.
"""

import time
from typing import List, Tuple, Optional

from agent.coder    import CoderAgent
from agent.critic   import CriticAgent
from agent.debugger import DebugAgent
from tools.tool_suite import ToolSuite, ToolReport
from pathlib import Path
import json
from agent.utils import strip_code_fences

DEFAULT_MAX_ITER = 3


def react_loop(
    subtask:      dict,
    context_code: str,
    coder:        CoderAgent,
    critic:       CriticAgent,
    debugger:     DebugAgent,
    tool_suite:   ToolSuite,
     original_prompt: str,
    max_iter:     int  = DEFAULT_MAX_ITER,
    verbose:      bool = True,
    save_dir:     Optional[Path] = None,   # ← NEW: where to write per-iteration files
   
) -> Tuple[str, List[dict]]:
    """
    Run the ReAct loop for one TCRG sub-task.

    If save_dir is provided, writes after every iteration:
        <save_dir>/
            iter_1_generated.py
            iter_1_tool_report.json
            iter_1_critique.json
            iter_2_debugged.py
            iter_2_tool_report.json
            iter_2_critique.json
            ...
            best.py
    """
    history:    List[dict] = []
    best_code:  str        = ""
    best_score: int        = -1
    code:       str        = ""

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, max_iter + 1):
        tag = f"[ReAct {subtask['id']} {iteration}/{max_iter}]"

        # ── THINK + ACT ────────────────────────────────────────────
        if iteration == 1:
            if verbose:
                print(f"  {tag} Coder generating ...")
            code = coder.code(subtask, context_code, original_prompt )
            role = "generated"
        else:
            prev = history[-1]
            if verbose:
                print(f"  {tag} Debugger fixing "
                      f"(severity={prev['critique'].get('severity','?')}) ...")
            code = debugger.fix(
                subtask     = subtask,
                code        = prev["code"],
                critique    = prev["critique"],
                tool_report = prev["tool_report"],
                original_prompt  = original_prompt,
            )
            role = "debugged"
        code = strip_code_fences(code)
        if not code.strip():
            if verbose:
                print(f"  {tag} WARNING: empty generation — skipping.")
            entry = _entry(iteration, role, "", {}, {})
            history.append(entry)
            if save_dir:
                _save_iteration(save_dir, iteration, role, "", {}, {})
            continue

        # ── OBSERVE ────────────────────────────────────────────────
        if verbose:
            print(f"  {tag} Running tool suite ...", end=" ", flush=True)

        report: ToolReport = tool_suite.run(code)

        if verbose:
            print(report.summary())

        # Track best code
        score = sum([
            report.syntax_ok,
            report.physics_ok,
            report.pytest_ok,
        ])
        if score > best_score or not best_code:
            best_score = score
            best_code  = code

        # ── THINK: early exit ──────────────────────────────────────
        if report.all_passed:
            if verbose:
                print(f"  {tag} ✓ All checks passed — accepting.")
            critique = {"severity": "none", "actionable_fixes": []}
            entry    = _entry(iteration, role, code, report.to_dict(), critique)
            history.append(entry)
            if save_dir:
                _save_iteration(save_dir, iteration, role, code,
                                report.to_dict(), critique)
                _save_best(save_dir, code)
            return code, history

        # ── THINK: Critique ────────────────────────────────────────
        if verbose:
            print(f"  {tag} Critic reviewing ...")

        critique = critic.review(subtask, code, report.to_dict())
        entry    = _entry(iteration, role, code, report.to_dict(), critique)
        history.append(entry)

        if save_dir:
            _save_iteration(save_dir, iteration, role, code,
                            report.to_dict(), critique)

        if verbose:
            n_fixes = len(critique.get("actionable_fixes", []))
            print(f"  {tag} Critique: severity={critique.get('severity','?')} "
                  f"{n_fixes} fix(es).")

        time.sleep(1.0)

    # Budget exhausted
    if verbose:
        print(f"  [ReAct {subtask['id']}] Budget exhausted — returning best code.")

    if save_dir:
        _save_best(save_dir, best_code)

    return best_code or code, history


# ─────────────────────── Helpers ──────────────────────────────────

def _entry(
    iteration:   int,
    role:        str,
    code:        str,
    tool_report: dict,
    critique:    dict,
) -> dict:
    return {
        "iteration":   iteration,
        "role":        role,           # "generated" | "debugged"
        "code":        code,           # full code stored now
        "code_length": len(code),
        "tool_report": tool_report,
        "critique":    critique,
    }


def _save_iteration(
    save_dir:    Path,
    iteration:   int,
    role:        str,
    code:        str,
    tool_report: dict,
    critique:    dict,
) -> None:
    prefix = f"iter_{iteration}_{role}"

    # Python file
    if code.strip():
        (save_dir / f"{prefix}.py").write_text(code, encoding="utf-8")

    # Tool report
    (save_dir / f"{prefix}_tool_report.json").write_text(
        json.dumps(tool_report, indent=2), encoding="utf-8"
    )

    # Critique
    (save_dir / f"{prefix}_critique.json").write_text(
        json.dumps(critique, indent=2), encoding="utf-8"
    )


def _save_best(save_dir: Path, code: str) -> None:
    if code.strip():
        (save_dir / "best.py").write_text(code, encoding="utf-8")