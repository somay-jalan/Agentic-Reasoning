# tools/physics_sanity.py
"""
Tool 3 — Physics sanity checker.

Runs the candidate code in a subprocess with a timeout and checks:
  1. Does it execute without error?
  2. Does the output contain NaN or Inf?
  3. Do any printed energy-like quantities stay conserved?
  4. Does the code remain stable when iteration counts are doubled?
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional
from agent.utils import strip_code_fences

TIMEOUT_SECONDS  = 12
STABILITY_TIMEOUT = 20      # doubled iterations may take longer


# ─────────────────────── Public API ───────────────────────────────

def check_physics_sanity(code: str) -> dict:
    """
    Returns:
        passed            : bool
        ran_successfully  : bool
        nan_inf_detected  : bool
        stability_ok      : bool | None  (None = not tested)
        issues            : list[str]
        stdout_snippet    : str
    """
    result: dict = {
        "passed":             False,
        "ran_successfully":   False,
        "nan_inf_detected":   False,
        "stability_ok":       None,
        "issues":             [],
        "stdout_snippet":     "",
    }

    if not code.strip():
        code = strip_code_fences(code)
        result["issues"].append("Empty code — nothing to run.")
        return result

    # Gate 1: must parse
    try:
        ast.parse(code)
    except SyntaxError as exc:
        result["issues"].append(f"SyntaxError prevents execution: {exc}")
        return result

    # Gate 2: run in subprocess
    run = _run_subprocess(code, timeout=TIMEOUT_SECONDS)
    result["ran_successfully"] = run["success"]
    result["stdout_snippet"]   = run["stdout"][:600]

    if not run["success"]:
        result["issues"].append(
            f"Runtime error: {run['error'][:300]}"
        )
        return result

    stdout = run["stdout"]

    # Check 3: NaN / Inf in output
    if re.search(r"\bnan\b|\binf\b", stdout, re.IGNORECASE):
        result["nan_inf_detected"] = True
        result["issues"].append(
            "NaN or Inf in program output — numerical instability or "
            "division by zero in the physics computation."
        )

    # Check 4: energy conservation heuristic
    energy_check = _check_energy_conservation(stdout)
    if energy_check["detected"] and not energy_check["conserved"]:
        result["issues"].append(
            f"Possible energy non-conservation: printed energy values vary by "
            f"{energy_check['variation_pct']:.1f}% "
            f"(threshold 1% for conservative systems)."
        )

    # Check 5: stability under doubled iteration count
    stability = _check_stability(code)
    result["stability_ok"] = stability["ok"]
    if not stability["ok"]:
        result["issues"].append(stability["message"])

    result["passed"] = (
        result["ran_successfully"]
        and not result["nan_inf_detected"]
        and result["stability_ok"] is not False
        and len(result["issues"]) == 0
    )
    return result


# ─────────────────────── Subprocess runner ────────────────────────

def _run_subprocess(code: str, timeout: int = TIMEOUT_SECONDS) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        tmp = fh.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output = True,
            text           = True,
            timeout        = timeout,
            env            = {**os.environ},
        )
        return {
            "success": proc.returncode == 0,
            "stdout":  proc.stdout,
            "stderr":  proc.stderr,
            "error":   proc.stderr[-400:] if proc.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout":  "",
            "stderr":  "",
            "error":   f"Timed out after {timeout}s — possible infinite loop.",
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": "", "stderr": "", "error": str(exc),
        }
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ─────────────────────── Conservation check ───────────────────────

def _check_energy_conservation(stdout: str) -> dict:
    """
    Scan stdout for lines mentioning energy-like quantities.
    If multiple values are found, check their relative spread.
    """
    result = {"detected": False, "conserved": True, "variation_pct": 0.0}

    pattern = re.compile(
        r"(?:total[_ ]?energy|kinetic|potential|hamiltonian|E\s*=|KE\s*=|PE\s*=)"
        r".*?(-?\d[\d.]*(?:e[+\-]?\d+)?)",
        re.IGNORECASE,
    )
    values = [float(m.group(1)) for m in pattern.finditer(stdout)]

    if len(values) < 2:
        return result

    result["detected"] = True
    mean = sum(values) / len(values)
    if abs(mean) < 1e-15:
        return result   # avoid divide-by-zero

    variation = (max(values) - min(values)) / abs(mean) * 100.0
    result["variation_pct"] = variation
    result["conserved"]     = variation < 1.0
    return result


# ─────────────────────── Stability check ──────────────────────────

def _check_stability(code: str) -> dict:
    """
    Try to double the iteration count and re-run.
    If output blows up (NaN/Inf) or code crashes, flag instability.
    """
    doubled = _double_iteration_counts(code)
    if doubled == code:
        # Could not find any integer iteration variable to double
        return {"ok": True, "message": ""}

    run = _run_subprocess(doubled, timeout=STABILITY_TIMEOUT)
    if not run["success"]:
        return {
            "ok":      False,
            "message": (
                "Code fails with doubled iteration count — likely numerical "
                f"instability. Error: {run['error'][:200]}"
            ),
        }
    if re.search(r"\bnan\b|\binf\b", run["stdout"], re.IGNORECASE):
        return {
            "ok":      False,
            "message": (
                "NaN/Inf appears when running for longer duration — "
                "numerical instability grows over time."
            ),
        }
    return {"ok": True, "message": ""}


def _double_iteration_counts(code: str) -> str:
    """
    Find module-level `name = <int>` assignments where name looks like
    an iteration count and double the value.
    Returns original code if nothing was changed.
    """
    count_names = {
        "n_steps", "N_steps", "num_steps", "nsteps",
        "n_iter",  "N_iter",  "num_iter",  "niters",
        "steps",   "N",       "n",         "iterations",
        "max_iter","maxiter",
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines   = code.splitlines()
    changed = False

    for node in tree.body:          # only top-level assignments
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in count_names
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
                and node.value.value > 0
            ):
                old_val  = node.value.value
                new_val  = old_val * 2
                line_idx = node.lineno - 1
                # Replace only the first occurrence of the integer on that line
                lines[line_idx] = lines[line_idx].replace(
                    str(old_val), str(new_val), 1
                )
                changed = True

    return "\n".join(lines) if changed else code
