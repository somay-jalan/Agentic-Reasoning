# training_examples.py
import json
import random
import dspy
from pathlib import Path
from modules import GROUND_TRUTH


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC POSITIVE PROOFS — correct, rigorous, all judges should PASS
# ═════════════════════════════════════════════════════════════════════════════

POSITIVE_PROOFS = [
    {
        "id": "pos_am_gm_2",
        "problem": "Let a, b ≥ 0. Show that (a+b)/2 ≥ √(ab).",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Consider the expression (√a - √b)². Since it is a square, (√a - √b)² ≥ 0.
Step 2: Expand: (√a - √b)² = a - 2√(ab) + b ≥ 0.
Step 3: Rearrange: a + b ≥ 2√(ab).
Step 4: Divide both sides by 2 (positive): (a+b)/2 ≥ √(ab).
Step 5: Equality holds iff √a = √b, i.e., a = b. ∎
""",
    },
    {
        "id": "pos_squares_nonneg",
        "problem": "Let x be any real number. Show that x² ≥ 0.",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: For any real x, consider two cases: x ≥ 0 and x < 0.
Step 2: Case x ≥ 0: x² = x · x ≥ 0 since the product of two non-negatives is non-negative.
Step 3: Case x < 0: x² = x · x = (-|x|)·(-|x|) = |x|² ≥ 0.
Step 4: In both cases x² ≥ 0.
Step 5: Equality holds iff x = 0. ∎
""",
    },
    {
        "id": "pos_triangle_reals",
        "problem": "For real numbers a, b, show |a + b| ≤ |a| + |b|.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Note that for any real x, -|x| ≤ x ≤ |x|.
Step 2: Therefore: -|a| ≤ a ≤ |a| and -|b| ≤ b ≤ |b|.
Step 3: Adding: -(|a|+|b|) ≤ a+b ≤ |a|+|b|.
Step 4: By definition of absolute value, |a+b| ≤ |a|+|b|.
Step 5: Equality holds iff a and b have the same sign. ∎
""",
    },
    {
        "id": "pos_bernoulli_induction",
        "problem": "Let x > -1 and n ∈ ℕ, n ≥ 1. Show (1+x)ⁿ ≥ 1 + nx.",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Base case n=1: (1+x)¹ = 1+x = 1+1·x. Holds with equality.
Step 2: Inductive hypothesis: assume (1+x)ᵏ ≥ 1+kx for some k ≥ 1.
Step 3: Multiply both sides by (1+x) > 0: (1+x)ᵏ⁺¹ ≥ (1+kx)(1+x).
Step 4: Expand right side: (1+kx)(1+x) = 1 + x + kx + kx² = 1+(k+1)x + kx².
Step 5: Since kx² ≥ 0, we get (1+x)ᵏ⁺¹ ≥ 1+(k+1)x.
Step 6: By induction, (1+x)ⁿ ≥ 1+nx for all n ≥ 1. ∎
""",
    },
    {
        "id": "pos_cauchy_schwarz_2d",
        "problem": "For reals a₁,a₂,b₁,b₂ show (a₁b₁+a₂b₂)² ≤ (a₁²+a₂²)(b₁²+b₂²).",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Expand the right side minus left side: (a₁²+a₂²)(b₁²+b₂²) - (a₁b₁+a₂b₂)².
Step 2: = a₁²b₁² + a₁²b₂² + a₂²b₁² + a₂²b₂² - a₁²b₁² - 2a₁b₁a₂b₂ - a₂²b₂².
Step 3: = a₁²b₂² - 2a₁a₂b₁b₂ + a₂²b₁².
Step 4: = (a₁b₂ - a₂b₁)².
Step 5: Since (a₁b₂ - a₂b₁)² ≥ 0, we have (a₁²+a₂²)(b₁²+b₂²) ≥ (a₁b₁+a₂b₂)².
Step 6: Equality holds iff a₁b₂ = a₂b₁, i.e., a₁/a₂ = b₁/b₂. ∎
""",
    },
    {
        "id": "pos_markov",
        "problem": "Let X ≥ 0 with E[X] finite, a > 0. Show P(X ≥ a) ≤ E[X]/a.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Define indicator 𝟏{X ≥ a} = 1 if X ≥ a, else 0.
Step 2: Pointwise: a·𝟏{X ≥ a} ≤ X. 
        When X ≥ a: a·1 = a ≤ X. When X < a: a·0 = 0 ≤ X since X ≥ 0.
Step 3: Take expectations of both sides (monotonicity of expectation):
        a·E[𝟏{X ≥ a}] ≤ E[X].
Step 4: Use E[𝟏{X ≥ a}] = P(X ≥ a):
        a·P(X ≥ a) ≤ E[X].
Step 5: Divide by a > 0: P(X ≥ a) ≤ E[X]/a. ∎
""",
    },
    {
        "id": "pos_jensen_convex",
        "problem": "Let f be convex, λ₁,λ₂ ≥ 0, λ₁+λ₂=1. Show f(λ₁x₁+λ₂x₂) ≤ λ₁f(x₁)+λ₂f(x₂).",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: This is precisely the definition of convexity of f.
Step 2: A function f is convex iff for all x₁,x₂ and λ ∈ [0,1]:
        f(λx₁+(1-λ)x₂) ≤ λf(x₁)+(1-λ)f(x₂).
Step 3: Set λ = λ₁ and (1-λ) = λ₂. Since λ₁+λ₂=1 and both ≥ 0, λ₁ ∈ [0,1].
Step 4: Substituting: f(λ₁x₁+λ₂x₂) ≤ λ₁f(x₁)+λ₂f(x₂).
Step 5: Equality holds iff x₁=x₂ or f is linear on [x₁,x₂]. ∎
""",
    },
    {
        "id": "pos_young_special",
        "problem": "For a,b ≥ 0, show ab ≤ a²/2 + b²/2.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Consider (a-b)² ≥ 0 (square of a real number).
Step 2: Expand: a² - 2ab + b² ≥ 0.
Step 3: Rearrange: a² + b² ≥ 2ab.
Step 4: Divide by 2: a²/2 + b²/2 ≥ ab.
Step 5: This is Young's inequality with p=q=2. Equality holds iff a=b. ∎
""",
    },
    {
        "id": "pos_chebyshev_2terms",
        "problem": "Let a₁ ≥ a₂ and b₁ ≥ b₂. Show 2(a₁b₁+a₂b₂) ≥ (a₁+a₂)(b₁+b₂).",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Expand right side: (a₁+a₂)(b₁+b₂) = a₁b₁+a₁b₂+a₂b₁+a₂b₂.
Step 2: So we need: 2(a₁b₁+a₂b₂) ≥ a₁b₁+a₁b₂+a₂b₁+a₂b₂.
Step 3: Equivalently: a₁b₁+a₂b₂ ≥ a₁b₂+a₂b₁.
Step 4: Rearrange: a₁b₁-a₁b₂-a₂b₁+a₂b₂ ≥ 0.
Step 5: Factor: (a₁-a₂)(b₁-b₂) ≥ 0.
Step 6: Since a₁ ≥ a₂ and b₁ ≥ b₂, both factors ≥ 0. Product ≥ 0. ∎
""",
    },
    {
        "id": "pos_power_mean",
        "problem": "For positive reals a,b show (a+b)/2 ≥ 2ab/(a+b).",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Cross multiply (both sides positive since a,b > 0):
        (a+b)² ≥ 4ab.
Step 2: Expand left side: a² + 2ab + b² ≥ 4ab.
Step 3: Rearrange: a² - 2ab + b² ≥ 0.
Step 4: Factor: (a-b)² ≥ 0.
Step 5: This is always true. Equality holds iff a=b. ∎
""",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC NEGATIVE PROOFS — deliberately flawed, specific judge should FAIL
# ═════════════════════════════════════════════════════════════════════════════

# ── NTC negatives (toy case used to prove general claim) ──────────────────────
NTC_NEGATIVES = [
    {
        "id": "neg_ntc_1",
        "problem": "For all n ≥ 1 and x > -1, show (1+x)ⁿ ≥ 1 + nx.",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Let x = 0. Then (1+0)ⁿ = 1 = 1 + n·0. Equality holds.
Step 2: Let x = 1. Then (1+1)ⁿ = 2ⁿ and 1+n·1 = 1+n. For n=2: 4 ≥ 3. ✓
Step 3: Since it works for x=0 and x=1, by continuity it works for all x > -1.
Step 4: Therefore (1+x)ⁿ ≥ 1+nx for all x > -1. ∎
""",
        "expected_ntc": "FAIL",
    },
    {
        "id": "neg_ntc_2",
        "problem": "For reals a₁,…,aₙ and b₁,…,bₙ show (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²).",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Let n=2, a₁=1, a₂=0, b₁=1, b₂=0.
Step 2: LHS = (1·1+0·0)² = 1. RHS = (1+0)(1+0) = 1. Equality holds.
Step 3: Let n=2, a₁=1, a₂=1, b₁=1, b₂=0.
Step 4: LHS = 1, RHS = 2. 1 ≤ 2. ✓
Step 5: These examples confirm the inequality holds in general. ∎
""",
        "expected_ntc": "FAIL",
    },
    {
        "id": "neg_ntc_3",
        "problem": "For a,b ≥ 0 show ab ≤ a²/2 + b²/2.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Set a = b = 1. Then ab = 1 and a²/2 + b²/2 = 1. Equality holds.
Step 2: Set a = 2, b = 1. Then ab = 2 and a²/2 + b²/2 = 2.5. 2 ≤ 2.5 ✓.
Step 3: Since equality holds at a=b and the inequality holds for a≠b,
        the inequality ab ≤ a²/2 + b²/2 is verified. ∎
""",
        "expected_ntc": "FAIL",
    },
]

# ── NLG negatives (logical gap — conclusion does not follow from steps) ────────
NLG_NEGATIVES = [
    {
        "id": "neg_nlg_1",
        "problem": "For vectors u,v in ℝⁿ show ‖u+v‖ ≤ ‖u‖ + ‖v‖.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: ‖u+v‖² = ⟨u+v, u+v⟩.
Step 2: = ⟨u,u⟩ + 2⟨u,v⟩ + ⟨v,v⟩ = ‖u‖² + 2⟨u,v⟩ + ‖v‖².
Step 3: Therefore ‖u+v‖ ≤ ‖u‖ + ‖v‖. ∎
""",
        "expected_nlg": "FAIL",
    },
    {
        "id": "neg_nlg_2",
        "problem": "For X ≥ 0 with finite E[X] and a > 0, show P(X ≥ a) ≤ E[X]/a.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: X is a non-negative random variable.
Step 2: E[X] is its expected value.
Step 3: By the properties of probability and expectation, P(X ≥ a) ≤ E[X]/a. ∎
""",
        "expected_nlg": "FAIL",
    },
    {
        "id": "neg_nlg_3",
        "problem": "For a,b ≥ 0, p,q > 1 with 1/p+1/q=1 show ab ≤ aᵖ/p + bᵍ/q.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: The function f(t) = eᵗ is convex.
Step 2: By convexity, the weighted average is bounded.
Step 3: Setting appropriate values gives ab ≤ aᵖ/p + bᵍ/q. ∎
""",
        "expected_nlg": "FAIL",
    },
]

# ── NAE negatives (approximation used as exact) ───────────────────────────────
NAE_NEGATIVES = [
    {
        "id": "neg_nae_1",
        "problem": "For a,b ≥ 0 show ab ≤ aᵖ/p + bᵍ/q where 1/p+1/q=1.",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: By AM-GM, (aᵖ/p + bᵍ/q) ≈ √(aᵖ · bᵍ) for large values.
Step 2: √(aᵖ · bᵍ) = a^(p/2) · b^(q/2) ≈ ab when p,q ≈ 2.
Step 3: Therefore ab ≤ aᵖ/p + bᵍ/q approximately, and the approximation
        is tight enough to establish the inequality in general. ∎
""",
        "expected_nae": "FAIL",
    },
    {
        "id": "neg_nae_2",
        "problem": "For all n ≥ 1 and x > -1, show (1+x)ⁿ ≥ 1 + nx.",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: For small x, (1+x)ⁿ ≈ 1 + nx + n(n-1)x²/2 by Taylor expansion.
Step 2: Since n(n-1)x²/2 ≥ 0 approximately, we have (1+x)ⁿ ≈ 1+nx.
Step 3: The higher order terms are negligible, so (1+x)ⁿ ≥ 1+nx. ∎
""",
        "expected_nae": "FAIL",
    },
    {
        "id": "neg_nae_3",
        "problem": "For a,b ≥ 0 show (a+b)/2 ≥ √(ab).",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: For a,b close to each other, √(ab) ≈ (a+b)/2.
Step 2: When a and b differ significantly, (a+b)/2 is clearly larger.
Step 3: In all cases (a+b)/2 ≥ √(ab) approximately, and the approximation
        confirms the inequality. ∎
""",
        "expected_nae": "FAIL",
    },
]

# ── NCE negatives (calculation error in algebra) ──────────────────────────────
NCE_NEGATIVES = [
    {
        "id": "neg_nce_1",
        "problem": "For reals a,b show (a+b)² ≥ 0.",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Expand (a+b)² = a² + b².
Step 2: Since a² ≥ 0 and b² ≥ 0, we have a² + b² ≥ 0.
Step 3: Therefore (a+b)² ≥ 0. ∎
""",
        "expected_nce": "FAIL",
    },
    {
        "id": "neg_nce_2",
        "problem": "For reals a₁,a₂,b₁,b₂ show (a₁b₁+a₂b₂)² ≤ (a₁²+a₂²)(b₁²+b₂²).",
        "answer": "A: ≤",
        "ground_truth": "A: ≤",
        "proof": """
Step 1: Expand (a₁²+a₂²)(b₁²+b₂²) = a₁²b₁² + a₁²b₂² + a₂²b₁² + a₂²b₂².
Step 2: Expand (a₁b₁+a₂b₂)² = a₁²b₁² + a₂²b₂².
Step 3: Difference = a₁²b₂² + a₂²b₁².
Step 4: Since a₁²b₂² + a₂²b₁² ≥ 0, the inequality holds. ∎
""",
        "expected_nce": "FAIL",
    },
    {
        "id": "neg_nce_3",
        "problem": "For a,b ≥ 0 show (a+b)/2 ≥ √(ab).",
        "answer": "B: ≥",
        "ground_truth": "B: ≥",
        "proof": """
Step 1: Square both sides: (a+b)²/4 ≥ ab.
Step 2: Expand: (a+b)²/4 = (a² + b²)/4.
Step 3: So we need a² + b² ≥ 4ab.
Step 4: This is equivalent to (a-b)² ≥ 2ab which is true since squares are non-negative. ∎
""",
        "expected_nce": "FAIL",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def make_example(data: dict, label: str,
                 expected_fa="PASS", expected_ntc="PASS",
                 expected_nlg="PASS", expected_nae="PASS",
                 expected_nce="PASS") -> dspy.Example:

    # Build a richer predicted_answer that looks like a real PART 1 response
    # so FinalAnswerJudge doesn't fail it for being "just a symbol"
    option   = data["answer"]          # e.g. "B: ≥"
    symbol   = option.split(":")[1].strip() if ":" in option else option
    problem  = data["problem"]

    # Extract the relation from the problem (the blank being filled)
    rich_answer = (
        f"Option {option}.\n"
        f"The correct relation is {symbol}.\n"
        f"This can be verified from the problem statement. "
        f"Equality holds under the specific conditions of the problem."
    )

    return dspy.Example(
        inequality_key=data["id"],
        problem_statement=data["problem"],
        predicted_answer=rich_answer,       # ← richer, not just "B: ≥"
        proof=data["proof"],
        ground_truth_option=data["ground_truth"],
        expected_fa=expected_fa,
        expected_ntc=expected_ntc,
        expected_nlg=expected_nlg,
        expected_nae=expected_nae,
        expected_nce=expected_nce,
        label=label,
    ).with_inputs(
        "inequality_key",
        "problem_statement",
        "predicted_answer",
        "proof",
    )


def build_synthetic_examples() -> list[dspy.Example]:
    """Builds all synthetic train/val examples."""
    examples = []

    # Positives
    for p in POSITIVE_PROOFS:
        examples.append(make_example(p, label="positive"))

    # NTC negatives
    for n in NTC_NEGATIVES:
        examples.append(make_example(
            n, label="negative_ntc", expected_ntc="FAIL"
        ))

    # NLG negatives
    for n in NLG_NEGATIVES:
        examples.append(make_example(
            n, label="negative_nlg", expected_nlg="FAIL"
        ))

    # NAE negatives
    for n in NAE_NEGATIVES:
        examples.append(make_example(
            n, label="negative_nae", expected_nae="FAIL"
        ))

    # NCE negatives
    for n in NCE_NEGATIVES:
        examples.append(make_example(
            n, label="negative_nce", expected_nce="FAIL"
        ))

    return examples   # 10 pos + 3+3+3+3 neg = 22 total


# ═════════════════════════════════════════════════════════════════════════════
# TEST SET — real inequality proofs + 1 negative per inequality
# ═════════════════════════════════════════════════════════════════════════════

# One flawed proof per real inequality — each targets a different judge
REAL_NEGATIVES = {
    "cauchy_schwarz": {
        "label": "negative_nce",
        "proof": """
Step 1: Consider P(t) = Σ(aᵢt + bᵢ)².
Step 2: Expand: P(t) = (Σaᵢ²)t² + 2(ΣaᵢbI)t + (Σbᵢ²).
Step 3: For P(t) ≥ 0, discriminant D = (2ΣaᵢbI)² - (Σaᵢ²)(Σbᵢ²) ≤ 0.
Step 4: Therefore (ΣaᵢbI)² ≤ (Σaᵢ²)(Σbᵢ²). ∎
""",
        # Error in Step 3: D = B²-4AC = 4(ΣaᵢbI)² - 4(Σaᵢ²)(Σbᵢ²)
        # missing factor of 4, so D ≤ 0 gives wrong bound
        "expected_nce": "FAIL",
    },
    "triangle": {
        "label": "negative_nlg",
        "proof": """
Step 1: ‖u+v‖² = ‖u‖² + 2⟨u,v⟩ + ‖v‖².
Step 2: We know ⟨u,v⟩ is bounded.
Step 3: Therefore ‖u+v‖ ≤ ‖u‖ + ‖v‖. ∎
""",
        "expected_nlg": "FAIL",
    },
    "jensens": {
        "label": "negative_ntc",
        "proof": """
Step 1: Let n=2, x₁=0, x₂=1, λ₁=λ₂=0.5. f(0.5) ≤ 0.5f(0)+0.5f(1). ✓ by convexity.
Step 2: This confirms Jensen's for n=2.
Step 3: For general n, the same argument applies by analogy. ∎
""",
        "expected_ntc": "FAIL",
    },
    "bernoullis": {
        "label": "negative_nae",
        "proof": """
Step 1: For small x, eˣ ≈ 1+x, so (1+x)ⁿ ≈ eⁿˣ ≈ 1+nx.
Step 2: The approximation shows (1+x)ⁿ ≥ 1+nx for all x > -1.
Step 3: Therefore Bernoulli's inequality holds. ∎
""",
        "expected_nae": "FAIL",
    },
    "youngs": {
        "label": "negative_nce",
        "proof": """
Step 1: Apply AM-GM with weights 1/p and 1/q:
        (1/p)·aᵖ + (1/q)·bᵍ ≥ (aᵖ)^(1/p) · (bᵍ)^(1/q).
Step 2: (aᵖ)^(1/p) · (bᵍ)^(1/q) = a · b^(q/p).
Step 3: Therefore ab ≤ aᵖ/p + bᵍ/q. ∎
""",
        # Error: (bᵍ)^(1/q) = b, not b^(q/p)
        "expected_nce": "FAIL",
    },
    "chebyshevs": {
        "label": "negative_ntc",
        "proof": """
Step 1: For n=2: 2(a₁b₁+a₂b₂) ≥ (a₁+a₂)(b₁+b₂). Verified by example.
Step 2: For n=3: checked numerically with a=(3,2,1), b=(3,2,1). Holds.
Step 3: Pattern holds for all n by the same reasoning. ∎
""",
        "expected_ntc": "FAIL",
    },
    "markovs": {
        "label": "negative_nlg",
        "proof": """
Step 1: X is non-negative with finite mean E[X].
Step 2: Intuitively, large values of X are less likely.
Step 3: Therefore P(X ≥ a) ≤ E[X]/a. ∎
""",
        "expected_nlg": "FAIL",
    },
}


def build_testset(results_dir: Path) -> list[dspy.Example]:
    testset = []

    for json_file in sorted(results_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        key  = data.get("inequality_key", "")
        if not data.get("proof"):
            continue

        # ── Fix: JSON stores "prompt" not "problem_statement" ─────────────
        # "prompt" is the full prompt sent to Gemini (includes problem + instructions)
        # We extract just the problem block from it, or just use the full prompt
        # as context — the judges benefit from seeing the full structured problem.
        problem_statement = (
            data.get("problem_statement") or   # try new field first
            data.get("prompt") or              # fall back to full prompt
            ""
        )

        if not problem_statement:
            print(f"  ⚠️  No problem_statement or prompt in {json_file.name}")
            continue

        # ── Positive ──────────────────────────────────────────────────────
        pos = dspy.Example(
            inequality_key=key,
            problem_statement=problem_statement,
            predicted_answer=data.get("answer", ""),
            proof=data.get("proof", ""),
            ground_truth_option=GROUND_TRUTH.get(key, "UNKNOWN"),
            expected_fa="PASS",
            expected_ntc="PASS",
            expected_nlg="PASS",
            expected_nae="PASS",
            expected_nce="PASS",
            label="positive",
        ).with_inputs(
            "inequality_key", "problem_statement",
            "predicted_answer", "proof",
        )
        testset.append(pos)
        print(f"  ✅ Test positive : {data.get('inequality_name', key)}")

        # ── Negative ──────────────────────────────────────────────────────
        if key in REAL_NEGATIVES:
            neg_data  = REAL_NEGATIVES[key]
            neg_label = neg_data["label"]

            neg = dspy.Example(
                inequality_key=f"{key}_neg",
                problem_statement=problem_statement,   # same prompt as positive
                predicted_answer=GROUND_TRUTH.get(key, "UNKNOWN"),
                proof=neg_data["proof"],
                ground_truth_option=GROUND_TRUTH.get(key, "UNKNOWN"),
                expected_fa="PASS",
                expected_ntc=neg_data.get("expected_ntc", "PASS"),
                expected_nlg=neg_data.get("expected_nlg", "PASS"),
                expected_nae=neg_data.get("expected_nae", "PASS"),
                expected_nce=neg_data.get("expected_nce", "PASS"),
                label=neg_label,
            ).with_inputs(
                "inequality_key", "problem_statement",
                "predicted_answer", "proof",
            )
            testset.append(neg)
            print(f"  🔴 Test negative : {key} ({neg_label})")

    return testset


def load_examples(
    results_dir: Path,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """
    Returns trainset, valset, testset.

    trainset : synthetic examples only (for GEPA reflection)
    valset   : synthetic examples only (for GEPA candidate scoring)
    testset  : real proofs + 1 negative per inequality (for final eval)
    """
    random.seed(seed)

    # ── Build synthetic pool (22 examples) ───────────────────────────────
    synthetic = build_synthetic_examples()
    random.shuffle(synthetic)

    # Stratified split of synthetic: ensure each label type in both splits
    by_label: dict[str, list] = {}
    for ex in synthetic:
        by_label.setdefault(ex.label, []).append(ex)

    trainset, valset = [], []
    for label, exs in by_label.items():
        random.shuffle(exs)
        mid = len(exs) // 2 + len(exs) % 2   # train gets extra if odd
        trainset.extend(exs[:mid])
        valset.extend(exs[mid:])

    random.shuffle(trainset)
    random.shuffle(valset)

    # ── Build real testset ────────────────────────────────────────────────
    testset = build_testset(results_dir)
    random.shuffle(testset)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n  {'Split':<8} {'Size':>5}   Label distribution")
    print(f"  {'─'*60}")
    for split_name, split in [("train", trainset), ("val", valset), ("test", testset)]:
        from collections import Counter
        dist = Counter(e.label for e in split)
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {split_name:<8} {len(split):>5}   {dist_str}")

    return trainset, valset, testset