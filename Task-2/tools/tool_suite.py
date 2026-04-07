# tools/tool_suite.py
"""
Aggregates all four tools into a single ToolSuite that the ReAct loop
calls after every code generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tools.syntax_checker import check_syntax
from tools.ast_pattern    import check_ast_patterns
from tools.physics_sanity import check_physics_sanity


@dataclass
class ToolReport:
    syntax:       dict = field(default_factory=dict)
    ast_patterns: dict = field(default_factory=dict)
    physics:      dict = field(default_factory=dict)
    pytest:       dict = field(default_factory=dict)    # empty if repo not available

    # ── Derived properties ─────────────────────────────────────────

    @property
    def syntax_ok(self) -> bool:
        return self.syntax.get("passed", False)

    @property
    def pytest_ok(self) -> bool:
        # True when not run OR when run and passing
        return self.pytest.get("passed", True)

    @property
    def physics_ok(self) -> bool:
        return self.physics.get("passed", False)

    @property
    def all_passed(self) -> bool:
        return self.syntax_ok and self.physics_ok and self.pytest_ok

    @property
    def has_critical_failure(self) -> bool:
        """
        Critical = syntax error or pytest failure.
        AST pattern warnings and physics sanity issues are non-critical
        (the code may still be functionally correct).
        """
        return not self.syntax_ok or not self.pytest_ok

    def to_dict(self) -> dict:
        return {
            "all_passed":           self.all_passed,
            "has_critical_failure": self.has_critical_failure,
            "syntax":               self.syntax,
            "ast_patterns":         self.ast_patterns,
            "physics":              self.physics,
            "pytest":               self.pytest,
        }

    def summary(self) -> str:
        parts = []
        if not self.syntax_ok:
            errs = self.syntax.get("syntax_errors", [])
            parts.append(f"SYNTAX({len(errs)} err)")
        if self.ast_patterns.get("issues"):
            n = len(self.ast_patterns["issues"])
            parts.append(f"AST({n} warn)")
        if not self.physics_ok:
            n = len(self.physics.get("issues", []))
            parts.append(f"PHYSICS({n} issue)")
        if not self.pytest_ok:
            n = len(self.pytest.get("failures", []))
            parts.append(f"PYTEST({n} fail)")
        return " | ".join(parts) if parts else "ALL PASS"


class ToolSuite:
    """
    Runs all tools in order.  Pytest is only run when repo_dir,
    target_relpath, and test_files are provided (i.e. during the final
    benchmark evaluation, not during intermediate node checks).
    """

    def __init__(
        self,
        repo_dir:       Optional[Path] = None,
        target_relpath: Optional[str]  = None,
        test_files:     Optional[List[Path]] = None,
    ):
        self.repo_dir       = repo_dir
        self.target_relpath = target_relpath
        self.test_files     = test_files or []

    def run(self, code: str) -> ToolReport:
        report = ToolReport()

        # ── Tool 1: Syntax ─────────────────────────────────────────
        report.syntax = check_syntax(code)
        if not report.syntax["passed"]:
            # Remaining tools require parseable code — short-circuit
            skipped = {"passed": False, "issues": ["Skipped: syntax error upstream."]}
            report.ast_patterns = skipped
            report.physics      = skipped
            return report

        # ── Tool 2: AST pattern analysis ───────────────────────────
        report.ast_patterns = check_ast_patterns(code)

        # ── Tool 3: Physics sanity (subprocess) ────────────────────
        report.physics = check_physics_sanity(code)

        # ── Tool 4: Pytest (only when repo is available) ───────────
        if self.repo_dir and self.target_relpath and self.test_files:
            report.pytest = self._run_pytest(code)

        return report

    # ── Internal ───────────────────────────────────────────────────

    def _run_pytest(self, code: str) -> dict:
        try:
            from evaluate import run_test_with_candidate
            passed, stdout, failures = run_test_with_candidate(
                self.repo_dir,
                self.target_relpath,
                code,
                self.test_files,
            )
            return {
                "passed":   passed,
                "failures": failures,
                "tail":     stdout[-400:] if stdout else "",
            }
        except Exception as exc:
            return {
                "passed":   False,
                "failures": [{"test_name": "TOOL_ERROR", "explanation": str(exc)}],
                "tail":     "",
            }

