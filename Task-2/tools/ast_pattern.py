# tools/ast_pattern.py
"""
Tool 2 — Physics-specific AST bad-pattern checker.

Each check function takes an ast.AST and returns a list of issue strings.
New checks can be added by decorating with @_pattern.
"""

import ast
from typing import List, Callable
from agent.utils import strip_code_fences

# Registry of all pattern-check functions
_PATTERNS: List[Callable[[ast.AST], List[str]]] = []


def _pattern(fn: Callable) -> Callable:
    _PATTERNS.append(fn)
    return fn


# ─────────────────────── Pattern definitions ──────────────────────

@_pattern
def math_module_on_arrays(tree: ast.AST) -> List[str]:
    """
    math.sqrt / math.sin / math.cos / math.exp used where the argument
    is likely an array — should use numpy equivalents.
    """
    issues  = []
    bad_fns = {"sqrt", "sin", "cos", "exp", "log", "pi", "fabs"}

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in bad_fns
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            continue

        # Only flag if the argument is not a plain numeric literal
        if node.args and not isinstance(node.args[0], ast.Constant):
            issues.append(
                f"Line ~{node.lineno}: math.{node.func.attr}() called on a "
                f"non-literal argument — use numpy.{node.func.attr}() to support arrays."
            )
    return issues


@_pattern
def floating_point_accumulation(tree: ast.AST) -> List[str]:
    """
    Detects `x = x + small_val` inside a loop — causes floating-point
    drift in long simulations. Suggests numpy.cumsum or pre-computed arrays.
    """
    issues = []
    seen   = set()

    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for node in ast.walk(loop):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            val  = node.value
            if isinstance(val, ast.BinOp) and isinstance(val.op, ast.Add):
                lhs_is_name = isinstance(val.left,  ast.Name) and val.left.id  == name
                rhs_is_name = isinstance(val.right, ast.Name) and val.right.id == name
                if (lhs_is_name or rhs_is_name) and name not in seen:
                    seen.add(name)
                    issues.append(
                        f"Line ~{node.lineno}: '{name} = {name} + ...' accumulation "
                        f"inside loop causes floating-point drift. Consider "
                        f"numpy.cumsum or pre-computed index arrays."
                    )
    return issues


@_pattern
def angular_frequency_missing_2pi(tree: ast.AST) -> List[str]:
    """
    Variables named omega / angular_freq assigned without any 2*pi factor.
    """
    issues    = []
    omega_ids = {"omega", "angular_freq", "angular_frequency"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id in omega_ids):
                continue
            rhs_dump = ast.dump(node.value)
            if "pi" not in rhs_dump and "Pi" not in rhs_dump:
                issues.append(
                    f"Line ~{node.lineno}: '{target.id}' assigned without a 2π "
                    f"factor. Angular frequency should be ω = 2π·f."
                )
    return issues


@_pattern
def floor_division_in_formula(tree: ast.AST) -> List[str]:
    """
    Floor division '//' inside an expression — almost always a physics bug.
    """
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            issues.append(
                f"Line ~{getattr(node, 'lineno', '?')}: Floor division '//' "
                f"truncates to integer — this is almost certainly a bug in a "
                f"physics formula. Replace with '/'."
            )
    return issues


@_pattern
def magic_number_timestep(tree: ast.AST) -> List[str]:
    """
    dt / timestep assigned as a bare numeric literal without a comment.
    Timesteps should be derived from physical scales or documented.
    """
    issues  = []
    dt_ids  = {"dt", "delta_t", "time_step", "timestep", "h"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in dt_ids
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
            ):
                issues.append(
                    f"Line ~{node.lineno}: '{target.id} = {node.value.value}' "
                    f"is a magic number. Derive it from physical scales "
                    f"(e.g. T/N_steps) or document its units."
                )
    return issues


@_pattern
def wavefunction_without_normalization(tree: ast.AST) -> List[str]:
    """
    A wavefunction variable updated inside a loop but never normalized.
    """
    issues   = []
    wf_ids   = {"psi", "wavefunction", "phi", "state_vector", "state"}
    norm_ids = {"norm", "normalize", "linalg", "normalized"}

    # Check if any wavefunction name exists at all
    all_names = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id in wf_ids
    }
    if not all_names:
        return issues

    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        loop_dump = ast.dump(loop)
        wf_in_loop   = any(name in loop_dump for name in wf_ids)
        norm_in_loop = any(ind  in loop_dump for ind  in norm_ids)
        if wf_in_loop and not norm_in_loop:
            issues.append(
                f"Line ~{getattr(loop, 'lineno', '?')}: Wavefunction/state vector "
                f"updated inside loop but no normalization call found. "
                f"Add normalization after each time step."
            )
            break   # one warning per tree is enough

    return issues


@_pattern
def integer_range_for_physics_loop(tree: ast.AST) -> List[str]:
    """
    range(N) used to iterate over physics time steps where numpy.linspace
    would be more appropriate for continuous time variables.
    """
    issues = []
    time_ids = {"t", "time", "tau"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # for t in range(...) pattern
        target = node.target
        iter_  = node.iter
        if (
            isinstance(target, ast.Name)
            and target.id in time_ids
            and isinstance(iter_, ast.Call)
            and isinstance(iter_.func, ast.Name)
            and iter_.func.id == "range"
        ):
            issues.append(
                f"Line ~{node.lineno}: 'for {target.id} in range(...)' uses integer "
                f"steps for a time variable. Use numpy.linspace for continuous time."
            )
    return issues


# ─────────────────────── Public API ───────────────────────────────

def check_ast_patterns(code: str) -> dict:
    """
    Run all registered pattern checks against the given code.

    Returns:
        passed : bool   — True only if zero issues found
        issues : list   — strings describing each problem
    """
    code = strip_code_fences(code) 
    result: dict = {"passed": True, "issues": []}

    if not code.strip():
        result["passed"] = False
        result["issues"].append("Empty code.")
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["passed"] = False
        result["issues"].append(f"SyntaxError prevents AST analysis: {exc}")
        return result

    all_issues: List[str] = []
    for check_fn in _PATTERNS:
        try:
            all_issues.extend(check_fn(tree))
        except Exception:
            pass    # never let a pattern checker crash the pipeline

    # Deduplicate while preserving order
    seen: set = set()
    deduped   = []
    for issue in all_issues:
        if issue not in seen:
            seen.add(issue)
            deduped.append(issue)

    result["issues"] = deduped
    result["passed"] = len(deduped) == 0
    return result
