# tools/syntax_checker.py
"""
Tool 1 — Syntax and import structure checker.
Pure static analysis, no execution.
"""

import ast
from typing import List


def check_syntax(code: str) -> dict:
    """
    Returns:
        passed        : bool
        syntax_errors : list of strings
        warnings      : list of non-fatal observations
    """
    result: dict = {
        "passed":        False,
        "syntax_errors": [],
        "warnings":      [],
    }

    if not code.strip():
        result["syntax_errors"].append("Code string is empty.")
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["syntax_errors"].append(
            f"SyntaxError at line {exc.lineno}: {exc.msg} "
            f"(text: {exc.text!r})"
        )
        return result

    result["passed"] = True
    _check_import_warnings(tree, result)
    return result


def _check_import_warnings(tree: ast.AST, result: dict) -> None:
    """Emit non-fatal warnings about likely missing imports."""
    imported: set = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            for alias in node.names:
                imported.add(alias.asname or alias.name)

    # Warn if numpy array operations appear without numpy import
    numpy_attrs = {
        "linspace", "zeros", "ones", "array", "arange",
        "exp", "sin", "cos", "sqrt", "dot", "cross",
        "linalg", "fft", "cumsum",
    }
    uses_numpy_attr = any(
        isinstance(n, ast.Attribute) and n.attr in numpy_attrs
        for n in ast.walk(tree)
    )
    if uses_numpy_attr and "numpy" not in imported and "np" not in imported:
        result["warnings"].append(
            "NumPy array operations detected but numpy is not explicitly imported."
        )

    # Warn if scipy used without import
    uses_scipy = any(
        isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id in {"scipy", "sp"}
        for n in ast.walk(tree)
    )
    if uses_scipy and "scipy" not in imported and "sp" not in imported:
        result["warnings"].append(
            "scipy attribute access detected but scipy is not explicitly imported."
        )
