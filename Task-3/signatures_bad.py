# signatures_bad.py
# Deliberately bad signatures — vague instructions, no examples of what
# PASS/FAIL means, no guidance on what to look for.
# Used as the unoptimized baseline to compare against GEPA-optimized versions.

import dspy


class FinalAnswerJudge(dspy.Signature):
    """
    Check if the answer is correct.
    """
    problem_statement: str = dspy.InputField(
        desc="The problem."
    )
    predicted_answer: str = dspy.InputField(
        desc="The predicted answer."
    )
    ground_truth_option: str = dspy.InputField(
        desc="The correct answer."
    )
    verdict: str = dspy.OutputField(
        desc="PASS or FAIL."
    )
    reason: str = dspy.OutputField(
        desc="Why."
    )


class NoToyCaseJudge(dspy.Signature):
    """
    Check the proof has any toy case to prove the final, PASS if not.
    """
    problem_statement: str = dspy.InputField(
        desc="The problem."
    )
    proof: str = dspy.InputField(
        desc="The proof."
    )
    verdict: str = dspy.OutputField(
        desc="PASS or FAIL."
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level."
    )
    reason: str = dspy.OutputField(
        desc="Why."
    )


class NoLogicalGapJudge(dspy.Signature):
    """
    Check the proof has any logical gap.
    """
    problem_statement: str = dspy.InputField(
        desc="The problem."
    )
    proof: str = dspy.InputField(
        desc="The proof."
    )
    verdict: str = dspy.OutputField(
        desc="PASS or FAIL."
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level."
    )
    reason: str = dspy.OutputField(
        desc="Why."
    )


class NoApproximationErrorJudge(dspy.Signature):
    """
    Check the proof if uses some wrong approximation to prove, PASS if not.
    """
    problem_statement: str = dspy.InputField(
        desc="The problem."
    )
    proof: str = dspy.InputField(
        desc="The proof."
    )
    verdict: str = dspy.OutputField(
        desc="PASS or FAIL."
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level."
    )
    reason: str = dspy.OutputField(
        desc="Why."
    )


class NoCalculationErrorJudge(dspy.Signature):
    """
    Check the proof has some calculation error.
    """
    problem_statement: str = dspy.InputField(
        desc="The problem."
    )
    proof: str = dspy.InputField(
        desc="The proof."
    )
    verdict: str = dspy.OutputField(
        desc="PASS or FAIL."
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level."
    )
    reason: str = dspy.OutputField(
        desc="Why."
    )