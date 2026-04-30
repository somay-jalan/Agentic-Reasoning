# modules.py
import dspy
from signatures import (
    FinalAnswerJudge,
    NoToyCaseJudge,
    NoLogicalGapJudge,
    NoApproximationErrorJudge,
    NoCalculationErrorJudge,
)

# Ground truth answers (used by FinalAnswerJudge)
GROUND_TRUTH: dict[str, str] = {
    "cauchy_schwarz": "A: ≤",
    "triangle":       "A: ≤",
    "jensens":        "A: ≤",
    "bernoullis":     "B: ≥",
    "youngs":         "A: ≤",
    "chebyshevs":     "B: ≥",
    "markovs":        "A: ≤",
}


class IneqMathJudgeAgent(dspy.Module):
    """
    Runs all 5 IneqMath judges on a generated proof:
      Judge 1: Final Answer Judge  (is the answer correct?)
      Judge 2: NTC — No Toy Case
      Judge 3: NLG — No Logical Gap
      Judge 4: NAE — No Approximation Error
      Judge 5: NCE — No Calculation Error

    Overall verdict = PASS only if ALL 5 judges return PASS.
    This mirrors the IneqMath paper's evaluation protocol exactly.
    """
    def __init__(self):
        super().__init__()
        self.final_answer_judge = dspy.ChainOfThought(FinalAnswerJudge)
        self.ntc_judge          = dspy.ChainOfThought(NoToyCaseJudge)
        self.nlg_judge          = dspy.ChainOfThought(NoLogicalGapJudge)
        self.nae_judge          = dspy.ChainOfThought(NoApproximationErrorJudge)
        self.nce_judge          = dspy.ChainOfThought(NoCalculationErrorJudge)

    def forward(self,
                inequality_key: str,
                problem_statement: str,
                predicted_answer: str,
                proof: str) -> dict:

        ground_truth = GROUND_TRUTH.get(inequality_key, "UNKNOWN")

        # ── Judge 1: Final Answer ─────────────────────────────────────────
        j1 = self.final_answer_judge(
            problem_statement=problem_statement,
            predicted_answer=predicted_answer,
            ground_truth_option=ground_truth,
        )

        # ── Judge 2: NTC ──────────────────────────────────────────────────
        j2 = self.ntc_judge(
            problem_statement=problem_statement,
            proof=proof,
        )

        # ── Judge 3: NLG ──────────────────────────────────────────────────
        j3 = self.nlg_judge(
            problem_statement=problem_statement,
            proof=proof,
        )

        # ── Judge 4: NAE ──────────────────────────────────────────────────
        j4 = self.nae_judge(
            problem_statement=problem_statement,
            proof=proof,
        )

        # ── Judge 5: NCE ──────────────────────────────────────────────────
        j5 = self.nce_judge(
            problem_statement=problem_statement,
            proof=proof,
        )

        # ── Aggregate ─────────────────────────────────────────────────────
        judges = {
            "final_answer": {"verdict": j1.verdict.strip().upper(), "reason": j1.reason},
            "ntc":          {"verdict": j2.verdict.strip().upper(), "confidence": j2.confidence, "reason": j2.reason},
            "nlg":          {"verdict": j3.verdict.strip().upper(), "confidence": j3.confidence, "reason": j3.reason},
            "nae":          {"verdict": j4.verdict.strip().upper(), "confidence": j4.confidence, "reason": j4.reason},
            "nce":          {"verdict": j5.verdict.strip().upper(), "confidence": j5.confidence, "reason": j5.reason},
        }

        overall = "PASS" if all(j["verdict"] == "PASS" for j in judges.values()) else "FAIL"

        return {"judges": judges, "overall_verdict": overall}