# IneqMath Proof Agent — LLM-Based Mathematical Inequality Verification with GEPA Optimization

A DSPy-based agentic system that generates and verifies proofs of mathematical
inequalities, replicating the IneqMath evaluation protocol with GEPA-based
prompt optimization.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [How It Works](#how-it-works)
   - [Step 1 — Proof Generation](#step-1--proof-generation)
   - [Step 2 — Granular Verification](#step-2--granular-verification)
   - [Step 3 — GEPA Prompt Optimization](#step-3--gepa-prompt-optimization)
5. [The Five Judges (IneqMath Protocol)](#the-five-judges-ineqmath-protocol)
6. [Prompt Design](#prompt-design)
   - [Optimized Signatures](#optimized-signatures)
   - [Unoptimized (Bad) Signatures](#unoptimized-bad-signatures)
7. [Metrics](#metrics)
8. [Running the Pipeline](#running-the-pipeline)
9. [Results and Analysis](#results-and-analysis)
    - [Baseline](#baseline)
    - [GEPA Optimization](#gepa-optimization)
    - [Per-Judge Accuracy](#per-judge-accuracy)
    - [Head-to-Head Comparison](#head-to-head-comparison)
    

---

## Project Overview

This project replicates and extends the
[IneqMath](https://ineqmath.github.io/) evaluation framework, which uses
LLM agents to verify mathematical proofs at a granular level.

**What the system does:**

1. Uses **Gemini 2.0 Flash** (via OpenRouter) to generate structured proofs
   for seven classical inequalities. Note that I used all 7 because for 2 the optimization was not working well and used a older gemini version because the problems are quite trivial and newer models were working too good.
2. Runs **five independent DSPy judge modules** on each proof, detecting
   specific error types (wrong answer, toy case, logical gap, approximation
   error, calculation error).
3. Applies **GEPA** (Gradient-free Evolutionary Prompt Adaptation), DSPy's
   automatic prompt optimizer, to improve the judge signatures.
4. Compares the optimized system against an intentionally weak baseline
   (`signatures_bad.py`) to demonstrate measurable prompt quality impact.

**Inequalities covered:**
Cauchy-Schwarz, Triangle, Jensen's, Bernoulli's, Young's,
Chebyshev's Sum, and Markov's Inequality.

---

## Repository Structure

```
.
├── .env                    # The API key for open router
├── config.py               # API key read, model name, global constants
├── prompts.py              # Structured prompts for all 7 inequalities
├── generator.py            # Calls Gemini, parses PART 1/2/3 response
├── run_all.py              # Generates proofs for all inequalities → JSON
├── signatures.py           # Optimized DSPy judge signatures (good)
├── signatures_bad.py       # Deliberately weak signatures (baseline)
├── modules.py              # IneqMathJudgeAgent — orchestrates 5 judges
├── run_dspy.py             # Loads saved proofs, runs judges, saves results
├── metrics.py              # Partial-credit and strict binary metrics
├── training_examples.py    # Synthetic train/val/test examples
├── optimize.py             # Full GEPA optimization pipeline
└── results/
    ├── google_gemini-2.0-flash-001/               # Raw generated proofs
    ├── google_gemini-2.0-flash-001_verified/      # Verified proofs (good sigs)
    ├── google_gemini-2.0-flash-001_verified_bad/  # Verified proofs (bad sigs)
    ├── google_gemini-2.0-flash-001_optimized/     # GEPA output (good sigs)
    └── google_gemini-2.0-flash-001_optimized_bad/ # GEPA output (bad sigs)
```



---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ineqmath-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install dspy-ai openai python-dotenv

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key:
#   OPENROUTER_API_KEY=sk-or-...
```

**.env file:**
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

---

## How It Works

### Step 1 — Proof Generation

`run_all.py` calls `generator.py` for each inequality. The generator sends
a structured prompt (from `prompts.py`) to Gemini and parses the response
into three labeled sections:

| Section | Content |
|---------|---------|
| **PART 1 — ANSWER** | The chosen inequality option (A–F) and equality conditions |
| **PART 2 — PROOF** | Numbered step-by-step proof |
| **PART 3 — THEOREMS** | Every theorem or lemma used, with step references |

The prompt is intentionally anonymous — the model is **not told the
inequality name**. It must identify the correct relation symbol from the
raw mathematical expression and prove it independently.

**Example prompt structure (Cauchy-Schwarz):**
```
MATHEMATICAL PROBLEM
====================
Let a₁, …, aₙ and b₁, …, bₙ be real numbers.
Consider: ( Σᵢ aᵢ · bᵢ )²  ( )  ( Σᵢ aᵢ² ) · ( Σᵢ bᵢ² )
Options: (A) ≤  (B) ≥  (C) =  ...

## PART 1 – ANSWER
## PART 2 – PROOF
## PART 3 – THEOREMS & LEMMAS USED
```

Results are saved to `results/google_gemini-2.0-flash-001/<key>.json`.

---

### Step 2 — Granular Verification

`run_dspy.py` loads the saved proofs and passes each one through
`IneqMathJudgeAgent` (in `modules.py`). The agent runs five independent
`dspy.ChainOfThought` judges in sequence and aggregates the verdicts.

The overall verdict is **PASS** only if **all five judges** return PASS —
matching the strict IneqMath protocol.

---

### Step 3 — GEPA Prompt Optimization

`optimize.py` runs GEPA on the `IneqMathJudgeAgent`. GEPA works by:

1. **Evaluating** the current system on the training set.
2. **Generating feedback** via `IneqMathGEPAMetric` — targeted per-judge
   feedback identifying exact error types (false positive vs. false negative).
3. **Proposing new candidate prompts** using a higher-temperature reflection
   LM (temperature = 0.9).
4. **Scoring candidates** on the validation set.
5. **Selecting the best** using Pareto-optimal candidate selection.

GEPA optimizes each judge's prompt independently (predictor-level mode),
then evaluates the full program at the end (program-level mode).

---

## The Five Judges (IneqMath Protocol)

Each judge is a `dspy.ChainOfThought` module with a typed `dspy.Signature`.

| Judge | Code Key | What It Detects |
|-------|----------|-----------------|
| **Final Answer Judge** | `final_answer` | Does the model pick the correct inequality symbol (≤, ≥, =)? |
| **No Toy Case (NTC)** | `ntc` | Does the proof use a special/degenerate case (e.g. n=1, a=b=0) to prove a general claim? |
| **No Logical Gap (NLG)** | `nlg` | Does every proof step follow logically from the previous, with no unjustified leaps? |
| **No Approximation Error (NAE)** | `nae` | Are all bounds exact and justified? No approximations treated as equalities? |
| **No Calculation Error (NCE)** | `nce` | Are all algebraic expansions and arithmetic steps correct? |

**Scoring:**
- Each judge returns `PASS` or `FAIL`.
- The **overall verdict** is `PASS` only if all five judges agree.
- For optimization, **partial credit** is given per judge (1.0 / 5 per correct judge).
- For final evaluation, **strict binary** scoring is used (1.0 only if all five correct).

---

## Prompt Design

### Optimized Signatures (`signatures.py`)

Each signature has a detailed docstring explaining the judge's responsibility,
explicit PASS/FAIL semantics, and precise field descriptions. Example (NTC):

```python
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
        desc="PASS if no toy-case flaw is found, FAIL if a toy case is used to draw a general conclusion."
    )
    confidence: str = dspy.OutputField(desc="HIGH, MEDIUM, or LOW.")
    reason: str = dspy.OutputField(
        desc="One sentence identifying where the toy-case flaw occurs, or confirming none was found."
    )
```

### Unoptimized (Bad) Signatures (`signatures_bad.py`)

These are intentionally vague — minimal docstrings, no definitions of
PASS/FAIL, no guidance on what to look for. Used as the unoptimized baseline:

```python
class NoToyCaseJudge(dspy.Signature):
    """
    Check the proof has any toy case to prove the final, PASS if not.
    """
    problem_statement: str = dspy.InputField(desc="The problem.")
    proof: str = dspy.InputField(desc="The proof.")
    verdict: str = dspy.OutputField(desc="PASS or FAIL.")
    confidence: str = dspy.OutputField(desc="Confidence level.")
    reason: str = dspy.OutputField(desc="Why.")
```

The difference between these two is the **independent variable** in the
prompt optimization experiment.

---

## Metrics

Two metrics are defined in `metrics.py`:

### `ineqmath_metric` (Partial Credit)
Used during GEPA training and predictor-level optimization:
```
score = (number of judges with correct verdict) / 5
```
Range: 0.0 to 1.0 in steps of 0.2.

### `ineqmath_strict` (Binary)
Used for final evaluation on the test set:
```
score = 1.0  if ALL 5 judges correct
score = 0.0  otherwise
```

---

## Running the Pipeline

```bash
# Step 1: Generate proofs for all 7 inequalities
python run_all.py

# Step 2: Run the 5 judges on the generated proofs (good signatures)
python run_dspy.py

# Step 3: Run GEPA optimization (good signatures)
python optimize.py

# To run with bad signatures, swap the import in modules.py:
# from signatures_bad import ... instead of from signatures import ...
# Then re-run run_dspy.py and optimize.py
```

---

## Results and Analysis

### Baseline

Both experiments (good and bad signatures) share the same baseline, loaded
from the `run_dspy.py` JSON outputs:

Note that bad signatures are made such that they should fail atleast one of these judges, these are made to test that the judge doesn't just pass all. Here the ones without _neg tag is the outputs generated by gemini for testing and _neg are the ones deliberately fail atleast one of the judges. 

| Example | Label | Baseline Score |
|---------|-------|---------------|
| markovs | none | 1.00 |
| cauchy_schwarz | none | 1.00 |
| jensens | none | 1.00 |
| youngs | none | 1.00 |
| triangle | none | 1.00 |
| bernoullis | none | 1.00 |
| chebyshevs | none | 0.00 |
| chebyshevs_neg | negative_ntc | 0.00 |
| markovs_neg | negative_nlg | 0.00 |
| youngs_neg | negative_nce | 0.00 |
| jensens_neg | negative_ntc | 0.00 |
| cauchy_schwarz_neg | negative_nce | 0.00 |
| bernoullis_neg | negative_nae | 0.00 |
| triangle_neg | negative_nlg | 0.00 |
| **AVERAGE** | | **0.4286** |


Label None means that we know the ground truth if the output is correct or not.

---

### GEPA Optimization 
Now given these samples we can't optimize as we need a metric to optimize and for gemini outputs we don't know the ground truth. So to tackle this I create a train and val set which contains correct proofs and wrong proofs for each judge to optimize prompt for.

**Positive Example:**  
Proof for AM-GM Inequality  
Step 1: Consider the expression (√a - √b)². Since it is a square, (√a - √b)² ≥ 0.  
Step 2: Expand: (√a - √b)² = a - 2√(ab) + b ≥ 0.  
Step 3: Rearrange: a + b ≥ 2√(ab).  
Step 4: Divide both sides by 2 (positive): (a+b)/2 ≥ √(ab).  
Step 5: Equality holds iff √a = √b, i.e., a = b.   

**Negative Example:**  
Problem - For all n ≥ 1 and x > -1, show (1+x)ⁿ ≥ 1 + nx.  
Step 1: Let x = 0. Then (1+0)ⁿ = 1 = 1 + n·0. Equality holds.  
Step 2: Let x = 1. Then (1+1)ⁿ = 2ⁿ and 1+n·1 = 1+n. For n=2: 4 ≥ 3. ✓  
Step 3: Since it works for x=0 and x=1, by continuity it works for all x > -1.  
Step 4: Therefore (1+x)ⁿ ≥ 1+nx for all x > -1.   
This should fail for No Toy Case Judge.

**METRIC**

Now for the metric to optimize, note that now for eachy problem I know which judges to pass and which judges to fail so I use the metric for each prompt, number of judges correctly classifying / number of judges. So while optimizing for a particular prompt if it corrects that judge then this would up for prompts hence maximizing for out GEPA metric

Now to test how good or bad prompts work, I create a good signatures which is basically good judge prompts and bad signatures which is bad judge prompts.  
Note that baseline is same as the critea for if the proof is correct is if all judges approve for these trivial problems its quite easy to differentiate that even with a very vague judge prompt.

**GEPA configuration:**
- Optimizer: `dspy.GEPA`
- Max full evaluations: 5
- Reflection minibatch size: 10
- Candidate selection: Pareto
- Reflection LM temperature: 0.9
- Seed: 42

Same for bad signatures and good signatures.


---

### Per-Judge Accuracy

Per-judge accuracy on the 7 negative examples in the test set:

#### Good Signatures

| Judge | Baseline | Optimized | Δ |
|-------|----------|-----------|---|
| FA (Final Answer) | 0.86 | **1.00** | +0.14 ⬆ |
| NTC (No Toy Case) | 0.43 | **0.57** | +0.14 ⬆ |
| NLG (No Logical Gap) | 0.43 | 0.43 | +0.00 |
| NAE (No Approx Error) | 0.29 | 0.29 | +0.00 |
| NCE (No Calc Error) | 0.14 | 0.14 | +0.00 |

#### Bad Signatures

| Judge | Baseline | Optimized | Δ |
|-------|----------|-----------|---|
| FA (Final Answer) | 0.86 | **1.00** | +0.14 ⬆ |
| NTC (No Toy Case) | 0.29 | **0.71** | **+0.43 ⬆** |
| NLG (No Logical Gap) | 0.43 | 0.43 | +0.00 |
| NAE (No Approx Error) | 0.29 | 0.29 | +0.00 |
| NCE (No Calc Error) | 0.29 | 0.29 | +0.00 |

The NTC judge shows the most dramatic improvement under GEPA, especially
with bad signatures, where NTC accuracy jumps from 0.29 → 0.71 (+0.43).
Both FA and NTC improve; NLG, NAE, and NCE remain flat.

This shows that after optimization alignments of judges have worked better however to be noted that due to low number of optimizations due to API budget issues, not all judges are fixed.

Now following are more results which I honestly feel are not that significant because again due to the way I have calculated the final yes no based on if all 5 judges pass, it makes it quite difficult for a bad review to pass. However there is one significant change I will talk below.

---

### Good Signatures



**Candidate scores discovered during optimization:**

| Candidate | Score | Discovered at Call |
|-----------|-------|--------------------|
| 0 (initial) | 0.8000 | 0 |
| **1 (BEST)** | **0.8444** | 29 |
| 2 | 0.8000 | 98 |

The best candidate achieved **0.8444** on the validation set — an improvement
of **+0.0444** over the initial prompt.

**Test set — Overall scores:**

| Example | Label | Baseline | Optimized | Δ |
|---------|-------|----------|-----------|---|
| markovs | none | 1.00 | 1.00 | +0.00 |
| cauchy_schwarz | none | 1.00 | 1.00 | +0.00 |
| jensens | none | 1.00 | **0.00** | **-1.00** |
| youngs | none | 1.00 | 1.00 | +0.00 |
| triangle | none | 1.00 | 1.00 | +0.00 |
| bernoullis | none | 1.00 | 1.00 | +0.00 |
| chebyshevs | none | 0.00 | 0.00 | +0.00 |
| cauchy_schwarz_neg | negative_nce | 0.00 | **1.00** | **+1.00** |
| chebyshevs_neg | negative_ntc | 0.00 | 0.00 | +0.00 |
| markovs_neg | negative_nlg | 0.00 | 0.00 | +0.00 |
| youngs_neg | negative_nce | 0.00 | 0.00 | +0.00 |
| jensens_neg | negative_ntc | 0.00 | 0.00 | +0.00 |
| bernoullis_neg | negative_nae | 0.00 | 0.00 | +0.00 |
| triangle_neg | negative_nlg | 0.00 | 0.00 | +0.00 |
| **AVERAGE** | | **0.4286** | **0.4286** | **+0.00** |

The overall average is unchanged (+0.00), but notable shifts occur:
`cauchy_schwarz_neg` is **fixed** (+1.00) while `jensens` **regresses**
(-1.00), resulting in a net zero on the strict binary metric.  For jensens we don't know what the truth is so it can fail but cauchy_schwarz_neg should not pass we know it is wrong so there is issue with the way judges have been optimized, in my understanding it is because that a judge nce judges alignment is very bad and after fixing others I think it is performing bad but other judges are not calling it as it was not their job, this testcase is passing.

The new prompts are in results/google_gemini-2.0-flash-001_optimized/optimized_judge_agent.json  I am not pasting them here are they are too big and are overwhelming the readability of the README.



---

### GEPA Optimization — Bad Signatures

**Candidate scores with bad signatures:**

| Candidate | Score | Discovered at Call |
|-----------|-------|--------------------|
| 0 | 0.7333 | 0 |
| 1 | 0.7333 | 29 |
| **2 (BEST)** | **0.7778** | 58 |
| 3 | 0.7778 | 127 |

Best validation score: **0.7778**, lower than the good-signature run
(0.8444), confirming that better initial signatures give GEPA a stronger
starting point.

For Test-set we see absolutely no change in the results with the baseline. 

The new prompts are in results/google_gemini-2.0-flash-001_optimized_bad/optimized_judge_agent.json  I am not pasting them here are they are too big and are overwhelming the readability of the README.


---

### Head-to-Head Comparison

| Metric | Bad Sigs (Baseline) | Good Sigs (Baseline) | Bad Sigs + GEPA | Good Sigs + GEPA |
|--------|--------------------|--------------------|-----------------|-----------------|
| Val score (best candidate) | 0.7333 | 0.8000 | 0.7778 | **0.8444** |
| Total metric calls | 136 | 127 | 136 | 127 |
| Candidates proposed | 4 | 3 | 4 | 3 |
| Test avg (strict) | 0.4286 | 0.4286 | 0.4286 | 0.4286 |

The key signal is in the **validation score**, not the test average:
- Good signatures + GEPA achieve **0.8444** on val vs. **0.7778** for bad
  signatures + GEPA.
- This gap (**+0.0666**) demonstrates that prompt quality matters even
  when GEPA can optimize both systems.



## Citation

This project is based on the IneqMath evaluation framework:
> **IneqMath: A Benchmark for LLM Mathematical Reasoning on Inequalities**
> https://ineqmath.github.io/

DSPy and GEPA:
> Khattab et al., *DSPy: Compiling Declarative Language Model Calls into
> Self-Improving Pipelines*, 2023.
> https://dspy.ai/
