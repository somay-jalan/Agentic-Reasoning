# signatures.py
import dspy


class FinalAnswerJudge(dspy.Signature):
    """
    Judge whether the final answer (inequality sign / bound) predicted
    in the proof matches the correct answer for the problem.
    A solution is correct only if the predicted relation symbol or bound
    exactly matches the ground truth.
    """
    problem_statement: str = dspy.InputField(
        desc="The original mathematical problem with options."
    )
    predicted_answer: str = dspy.InputField(
        desc="The answer section (PART 1) extracted from the model's response, "
             "containing the selected option and stated inequality."
    )
    ground_truth_option: str = dspy.InputField(
        desc="The correct option letter (e.g. 'A') and its symbol (e.g. '≤')."
    )
    verdict: str = dspy.OutputField(
        desc="PASS if the predicted answer matches ground truth, FAIL otherwise."
    )
    reason: str = dspy.OutputField(
        desc="One sentence explaining why it passes or fails."
    )


class NoToyCaseJudge(dspy.Signature):
    """
    NTC Judge — No Toy Case.
    Detect whether the proof draws a general conclusion from a special
    or degenerate case (e.g., setting n=1, or using a=b=c to prove
    something for all a,b,c). This is a logical flaw.
    A proof PASSES if it does NOT use a toy/special case to justify
    a general claim. It FAILS if it does.
    """
    problem_statement: str = dspy.InputField(
        desc="The original mathematical problem being proved."
    )
    proof: str = dspy.InputField(
        desc="The full step-by-step proof to evaluate."
    )
    verdict: str = dspy.OutputField(
        desc="PASS if no toy-case flaw is found, FAIL if a toy case is used "
             "to draw a general conclusion."
    )
    confidence: str = dspy.OutputField(
        desc="HIGH, MEDIUM, or LOW."
    )
    reason: str = dspy.OutputField(
        desc="One sentence identifying where the toy-case flaw occurs, "
             "or confirming none was found."
    )


class NoLogicalGapJudge(dspy.Signature):
    """
    NLG Judge — No Logical Gap.
    Detect whether the proof contains any unjustified logical leaps —
    steps where the conclusion does not follow clearly from what was
    established before, or where a key intermediate step is missing.
    A proof PASSES if every step follows logically from the previous.
    It FAILS if there is at least one logical gap.
    """
    problem_statement: str = dspy.InputField(
        desc="The original mathematical problem being proved."
    )
    proof: str = dspy.InputField(
        desc="The full step-by-step proof to evaluate."
    )
    verdict: str = dspy.OutputField(
        desc="PASS if no logical gaps are found, FAIL if a gap exists."
    )
    confidence: str = dspy.OutputField(
        desc="HIGH, MEDIUM, or LOW."
    )
    reason: str = dspy.OutputField(
        desc="One sentence identifying the logical gap, or confirming none found."
    )


class NoApproximationErrorJudge(dspy.Signature):
    """
    NAE Judge — No Approximation Error.
    Detect whether the proof makes any unjustified approximations —
    treating an approximate bound as an exact equality, or replacing
    an expression with a looser one without proper justification.
    A proof PASSES if all bounds and equalities are exact and justified.
    It FAILS if any approximation is used incorrectly.
    """
    problem_statement: str = dspy.InputField(
        desc="The original mathematical problem being proved."
    )
    proof: str = dspy.InputField(
        desc="The full step-by-step proof to evaluate."
    )
    verdict: str = dspy.OutputField(
        desc="PASS if no approximation errors are found, FAIL if one exists."
    )
    confidence: str = dspy.OutputField(
        desc="HIGH, MEDIUM, or LOW."
    )
    reason: str = dspy.OutputField(
        desc="One sentence identifying the approximation error, "
             "or confirming none found."
    )


class NoCalculationErrorJudge(dspy.Signature):
    """
    NCE Judge — No Calculation Error.
    Detect whether the proof contains any arithmetic or algebraic
    calculation mistakes — wrong expansions, incorrect simplifications,
    sign errors, or wrong numeric values.
    A proof PASSES if all calculations are correct.
    It FAILS if any calculation mistake is found.
    """
    problem_statement: str = dspy.InputField(
        desc="The original mathematical problem being proved."
    )
    proof: str = dspy.InputField(
        desc="The full step-by-step proof to evaluate."
    )
    verdict: str = dspy.OutputField(
        desc="PASS if no calculation errors are found, FAIL if one exists."
    )
    confidence: str = dspy.OutputField(
        desc="HIGH, MEDIUM, or LOW."
    )
    reason: str = dspy.OutputField(
        desc="One sentence identifying the calculation error, "
             "or confirming none found."
    )