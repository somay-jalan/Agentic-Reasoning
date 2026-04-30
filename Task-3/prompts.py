# prompts.py
# ─────────────────────────────────────────────────────────────────────────────
# IneqMath-style prompts: raw mathematical expressions only.
# The model is NOT told the inequality name or how to prove it.
# It must:
#   PART 1 – Determine the correct inequality sign / constant
#   PART 2 – Generate a structured proof entirely on its own
#   PART 3 – List theorems it used
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a rigorous mathematical reasoning assistant.
You will be given a mathematical expression or relationship.
You must determine the correct answer, prove it with a well-structured
step-by-step proof, and list all theorems you used.
You are NOT told what method to use — figure it out yourself.
Always respond in exactly three labeled sections. Never skip a section."""


def build_prompt(expression_block: str,
                 answer_instruction: str,
                 proof_instruction: str,
                 theorem_instruction: str) -> str:
    return f"""
==============================================================
MATHEMATICAL PROBLEM
==============================================================

{expression_block}

--------------------------------------------------------------
You must respond in EXACTLY the three sections below.
Do NOT merge sections. Label each section exactly as shown.
--------------------------------------------------------------

## PART 1 – ANSWER
{answer_instruction}

## PART 2 – PROOF
{proof_instruction}

## PART 3 – THEOREMS & LEMMAS USED
{theorem_instruction}
""".strip()


# ── Shared instructions (same for all) ───────────────────────────────────────

PROOF_INSTRUCTION = """
Provide a complete, rigorous, self-contained proof for your answer above.
Structure your proof as clearly numbered steps. Each step must:
  - Be a single, atomic mathematical claim or algebraic manipulation
  - Explicitly justify why the claim holds
  - State any intermediate result before using it in the next step
Do NOT skip steps. Do NOT assume the reader knows what technique you are using.
"""

THEOREM_INSTRUCTION = """
List every theorem, lemma, algebraic identity, or mathematical property
that you used anywhere in PART 2.
For each entry provide:
  • Name of the theorem / property
  • Precise statement of it as you used it in this proof
  • The step number(s) in PART 2 where you applied it
"""


# ─────────────────────────────────────────────────────────────────────────────
# A) Cauchy-Schwarz
# ─────────────────────────────────────────────────────────────────────────────
CAUCHY_SCHWARZ_PROMPT = build_prompt(

    expression_block="""
Let a₁, a₂, …, aₙ and b₁, b₂, …, bₙ be real numbers.

Consider the following relation:

    ( Σᵢ aᵢ · bᵢ )²     ( )     ( Σᵢ aᵢ² ) · ( Σᵢ bᵢ² )

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact condition under which equality holds.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# B) Triangle Inequality
# ─────────────────────────────────────────────────────────────────────────────
TRIANGLE_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let u and v be vectors in ℝⁿ, and let ‖·‖ denote the standard Euclidean norm.

Consider the following relation:

    ‖u + v‖     ( )     ‖u‖ + ‖v‖

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact condition under which equality holds.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# C) Jensen's Inequality
# ─────────────────────────────────────────────────────────────────────────────
JENSENS_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let f : ℝ → ℝ be a convex function.
Let x₁, x₂, …, xₙ ∈ ℝ and let λ₁, λ₂, …, λₙ ≥ 0 with Σᵢ λᵢ = 1.

Consider the following relation:

    f( Σᵢ λᵢ xᵢ )     ( )     Σᵢ λᵢ f(xᵢ)

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds, and state what changes
if f is concave instead of convex.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact condition under which equality holds.
State how the answer changes when f is concave.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# D) Bernoulli's Inequality
# ─────────────────────────────────────────────────────────────────────────────
BERNOULLIS_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let x > −1 be a real number and let n be a positive integer (n ≥ 1).

Consider the following relation:

    (1 + x)ⁿ     ( )     1 + n·x

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact conditions under which equality holds.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# E) Young's Inequality
# ─────────────────────────────────────────────────────────────────────────────
YOUNGS_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let a, b ≥ 0 be real numbers.
Let p > 1 and q > 1 be real numbers satisfying 1/p + 1/q = 1.

Consider the following relation:

    a · b     ( )     aᵖ/p  +  bᵍ/q

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact condition under which equality holds.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# F) Chebyshev's Sum Inequality
# ─────────────────────────────────────────────────────────────────────────────
CHEBYSHEVS_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let a₁ ≥ a₂ ≥ … ≥ aₙ and b₁ ≥ b₂ ≥ … ≥ bₙ be real numbers
(both sequences sorted in the same order).

Consider the following relation:

    n · ( Σᵢ aᵢ bᵢ )     ( )     ( Σᵢ aᵢ ) · ( Σᵢ bᵢ )

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds, and what happens when
the two sequences are sorted in opposite orders.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact conditions under which equality holds.
State how the answer changes if the sequences are oppositely ordered.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# G) Markov's Inequality
# ─────────────────────────────────────────────────────────────────────────────
MARKOVS_INEQUALITY_PROMPT = build_prompt(

    expression_block="""
Let X be a non-negative random variable with finite expectation E[X].
Let a > 0 be a real constant.

Consider the following relation:

    P( X ≥ a )     ( )     E[X] / a

Options:
  (A) ≤    (B) ≥    (C) =    (D) <    (E) >    (F) None of the above

Determine the correct relation to fill in the blank ( ).
Also determine exactly when equality holds.
""",

    answer_instruction="""
State which option (A–F) is correct and write the complete, precise
mathematical statement using the correct inequality or equality sign.
State the exact condition under which equality holds.
""",

    proof_instruction=PROOF_INSTRUCTION,
    theorem_instruction=THEOREM_INSTRUCTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
ALL_PROMPTS: dict[str, tuple[str, str]] = {
    "cauchy_schwarz": ("Cauchy-Schwarz Inequality",   CAUCHY_SCHWARZ_PROMPT),
    "triangle":       ("Triangle Inequality",         TRIANGLE_INEQUALITY_PROMPT),
    "jensens":        ("Jensen's Inequality",         JENSENS_INEQUALITY_PROMPT),
    "bernoullis":     ("Bernoulli's Inequality",      BERNOULLIS_INEQUALITY_PROMPT),
    "youngs":         ("Young's Inequality",          YOUNGS_INEQUALITY_PROMPT),
    "chebyshevs":     ("Chebyshev's Sum Inequality",  CHEBYSHEVS_INEQUALITY_PROMPT),
    "markovs":        ("Markov's Inequality",         MARKOVS_INEQUALITY_PROMPT),
}