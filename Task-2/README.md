# PhysicsPy-Bench: A Domain-Specific LLM Coding Benchmark and Agentic Framework for Scientific Python

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Domain](#2-domain)
3. [Benchmark Construction](#3-benchmark-construction)
4. [Extended CodeBLEU Score](#4-extended-codebleu-score)
5. [Models Evaluated](#5-models-evaluated)
6. [Zero-Shot Evaluation Results](#6-zero-shot-evaluation-results)
7. [Agentic Framework](#7-agentic-framework)
8. [Agent Evaluation Results](#8-agent-evaluation-results)
9. [Project Structure](#9-project-structure)
10. [Installation](#10-installation)
11. [Usage](#11-usage)
12. [Reproducing Results](#12-reproducing-results)
13. [References](#13-references)

---

## 1. Project Overview

PhysicsPy-Bench is an end-to-end LLM coding benchmark and multi-agent
evaluation framework targeting scientific Python programming. It was
constructed from real GitHub pull requests drawn from two widely used
scientific Python libraries:

- **SymPy** (https://github.com/sympy/sympy) : a pure-Python computer
  algebra system covering symbolic mathematics, classical mechanics,
  control theory, group theory, and special functions.
- **QuTiP** (https://github.com/qutip/qutip) : the Quantum Toolbox in
  Python, a library for simulating open quantum systems, quantum circuits,
  and quantum information processing.

Together these libraries span the full range of scientific Python
programming: symbolic computation, numerical simulation, quantum mechanics,
and control systems. PRs were selected to cover both bug fixes and feature
additions across both codebases.

The project has three components:

- **Benchmark**: 7 real-world PRs with zero-shot prompts, reference
  solutions, and pytest test suites — sourced from both SymPy and QuTiP.
- **Extended CodeBLEU**: A domain-weighted variant of CodeBLEU that assigns
  higher importance to physics- and mathematics-specific Python identifiers
  drawn from both libraries.
- **Agentic Framework**: A multi-agent ReAct system inspired by VerilogCoder
   [(arxiv:2408.08927)](https://arxiv.org/abs/2408.08927) with a Planner, Coder, Critic, and Debug agent
  operating over a Task-Conditioned Reasoning Graph (TCRG).

---

## 2. Domain

**Language**: Python 3  
**Domain**: Scientific Simulation and Physics Programming  
**Source libraries**:
- SymPy — https://github.com/sympy/sympy
- QuTiP — https://github.com/qutip/qutip

### 2.1 SymPy

SymPy is a pure-Python computer algebra system used across physics,
engineering, applied mathematics, and scientific computing. It implements:

- Classical and quantum mechanics (`sympy.physics.mechanics`,
  `sympy.physics.quantum`)
- Control systems theory (`sympy.physics.control`)
- Special functions: trigonometric, Bessel, hypergeometric, orthogonal
  polynomials
- Group theory and abstract algebra (`sympy.combinatorics`, `sympy.groups`)
- Symbolic integration, differentiation, and series expansion
- Numerical evaluation via `lambdify` and `mpmath`

### 2.2 QuTiP

QuTiP is the standard Python library for quantum physics simulation. It
implements:

- Quantum state representation (`Qobj`, `ket`, `bra`, density matrices)
- Time evolution of open quantum systems (Lindblad master equation,
  Monte Carlo wave function method)
- Quantum gates and quantum circuit simulation
- Expectation value computation and measurement
- Quantum information: entropy, fidelity, partial trace, concurrence
- Visualization: Bloch sphere, Wigner functions, Husimi Q-functions

### 2.3 Why These Two Libraries

Both libraries were chosen because:

1. They have large, well-maintained public repositories with structured,
   descriptive PRs.
2. Correctness is objectively verifiable: symbolic results match analytical
   answers; quantum simulation results match known physical results.
3. The codebases share a rich vocabulary of domain-specific identifiers
   (`lambdify`, `Qobj`, `mesolve`, `bode_magnitude`, `FpGroup`,
   `_eval_derivative_n_times`) that make domain-specific token weighting
   meaningful and measurable.
4. Existing test suites serve directly as pass/fail oracles without
   requiring manual test authoring.
5. Together they cover complementary aspects of scientific Python:
   SymPy for symbolic/algebraic computation and QuTiP for numerical
   quantum simulation.

### 2.4 PR Categories Covered

| PR         | Library | Category                      | Task Type |
|------------|---------|-------------------------------|-----------|
| pr29550    | SymPy   | Control systems (Bode plot)   | Bug fix   |
| pr29633    | SymPy   | Symbolic integration          | Bug fix   |
| pr29394    | SymPy   | Trigonometric nth derivative  | Feature   |
| pr29093    | SymPy   | Group theory (FpGroup kernel) | Feature   |
| pr29263    | SymPy   | Fractional Intergral          | Bug fix   |
| pr29369    | SymPy   | Linear Programming            | Bug fix   |
| pr2835     | QuTiP   | Quantum Pauli basis           | Bug fix   |

---

## 3. Benchmark Construction

### 3.1 Data Collection

Pull requests were selected from both repositories according to the
following criteria:

- Closed and merged PRs with a clear, self-contained description
- The change is localised to one or two Python files (to make evaluation
  tractable)
- An existing or PR-added test file covers the changed functionality
- The PR description contains enough context for a zero-shot prompt

Each benchmark item lives under `manual_corpus/<suite>/<pr_name>/` and
contains:

```
manual_corpus/
    test-1/                          ← SymPy PRs
        pr29550/     
            metadata.json            ← repo URL, checkout, target file, test files
            prompt.txt               ← zero-shot prompt sent to the model
            before.py                ← file state before the PR
            after.py                 ← reference solution (ground truth)
            fix.diff                 ← diff between after and before file
            test_control_plots.py    ← test file where before fails and after passes all the test, included in the merge pr
        pr29633/ ...
        pr29394/ ...
        pr29369/ ...
        pr29093/ ...
        pr29263/ ...

    test-2/                     ← QuTiP PRs
        pr2835/
            metadata.json
            prompt.txt
            before.py
            after.py
            fix.diff
            test_superop_reps.py
```

### 3.2 Prompt Construction

Two prompt templates are used depending on task type. Both templates
include the full source file so the model has complete context.

**Bug fix template**:
```
You are an expert Python physicist and scientific programmer.

TASK:
The following issue is present in the <library> library. You are
provided with the error and the file containing the error. Figure out
the error and fix the file. Provide the complete Python file in a
```python``` code block. You should provide the whole file.

ERROR (as defined by the user):
<error description, reproduction script, and expected result>

PYTHON FILE WITH ERROR:
<full contents of before.py>
```

**Feature template**:
```
You are an expert Python physicist and scientific programmer.

TASK:
The following feature is to be added to the <library> library. You are
provided with the file where the feature should be added. Figure out
how to add the feature. Provide the complete Python file in a
```python``` code block. You should provide the whole file.

FEATURE (as defined by the user):
<feature description from PR>

PYTHON FILE IN WHICH YOU NEED TO ADD THE FEATURE:
<full contents of before.py>
```

### 3.3 Benchmark Configuration

`manual_corpus/benchmark_config.json` lists all items from both libraries:

```json
{
  "items": [
    "test-1/pr29550",
    "test-1/pr29633",
    "test-1/pr29394",
    "test-1/pr29369",
    "test-1/pr29093",
    "test-1/pr29263",
    "test-2/pr2835"
  ]
}
```

Each item's `metadata.json` specifies the source library:

```json
{
  "name":           "pr29550",
  "repo_url":       "https://github.com/sympy/sympy.git",
  "checkout":       "master",
  "target_relpath": "sympy/physics/control/control_plots.py",
  "before_file":    "before.py",
  "after_file":     "after.py",
  "test_files":     ["test.py"]
}
```

```json
{
  "name": "pr2835",
  "repo_url": "https://github.com/qutip/qutip",
  "checkout":       "master",
  "target_relpath": "qutip/core/superop_reps.py",
  "before_file":    "before.py",
  "after_file":     "after.py",
  "test_files":     ["test.py"]
}
```

---

## 4. Extended CodeBLEU Score

Standard CodeBLEU (Ren et al., 2020) measures code similarity via n-gram
match, weighted n-gram match, AST match, and data-flow match with equal
token weighting. For scientific Python spanning two physics libraries, a
generic weighting treats `import numpy` the same as `mesolve` or
`bode_magnitude_numerical_data`, which undervalues domain correctness.

### 4.1 Domain Weight Construction

Domain weights are built from the reference (`after.py`) files of all
benchmark items across both libraries using an AST-based token extractor.
Each token receives a base importance weight based on its syntactic role:


| Role                  | Base Weight | Example identifiers                          |
|-----------------------|-------------|----------------------------------------------|
| `class_instantiation` | 3.0         | `Qobj()`, `Symbol()`, `TransferFunction()`   |
| `method_call`         | 2.5         | `.doit()`, `.ptrace()`, `.simplify()`        |
| `attribute_access`    | 2.0         | `qutip.mesolve`, `sp.integrate`              |
| `function_call`       | 2.0         | `lambdify`, `mesolve`, `calc_codebleu`       |
| `exception_type`      | 1.8         | `NotImplementedError`, `QutipError`          |
| `decorator`           | 1.5         | `@cacheit`, `@property`                      |
| `import_name`         | 1.3         | `sympy`, `qutip`, `numpy`, `scipy`           |
| `comparison_target`   | 1.2         | Test assertions on physical quantities       |
| `name_load`           | 1.0         | General variable references                  |
| `string_constant`     | 0.5         | Low-value string literals                    |

TF-IDF(Term Frequency — Inverse Document Frequency) is applied across all reference 
files from both libraries to upweight tokens that are domain-specific (rare across files)
and downweight generic tokens (common across all files). The final per-token weight is
normalised to the range [0.1, 3.0].

Because weights are computed from both SymPy and QuTiP reference files
together, identifiers specific to either library (e.g. `Qobj`, `mesolve`,
`FpGroup`, `bode_magnitude`) receive high weights, while identifiers
common to all scientific Python (`numpy`, `array`, `dtype`) receive lower
weights. This means the extended CodeBLEU correctly rewards generated code
that uses library-appropriate APIs rather than generic numeric code that
happens to produce a similar result.

Weights are saved to `domain_weights.json` after first computation and
reused across all evaluation runs.

### 4.2 Domain-Weighted CodeBLEU Formula

```
domain_codebleu = clamp(base_codebleu + domain_delta, 0, 1)

domain_delta = clamp((hyp_mean - global_mean) / global_std, -1, 1) × 0.15

where:
  base_codebleu  = standard CodeBLEU(hypothesis, reference)
  hyp_mean       = mean domain weight of tokens in the hypothesis
  global_mean    = mean domain weight across all tokens in weight table
  global_std     = standard deviation of all domain weights
```

The domain delta is additive and bounded to ±0.15 so that:
- A high base score cannot be inflated to 1.0 purely by using the right
  library names
- A low base score cannot be rescued purely by domain alignment
- The score genuinely reflects both structural similarity and
  domain-appropriate API usage

### 4.3 Metrics Reported Per Item

| Metric                  | Description                                              |
|-------------------------|----------------------------------------------------------|
| `codebleu`              | Standard CodeBLEU (equal weights, no domain adjustment)  |
| `ngram_match`           | Unweighted n-gram precision                              |
| `weighted_ngram_match`  | n-gram precision weighted by token frequency             |
| `syntax_match`          | AST subtree match score                                  |
| `dataflow_match`        | Data flow graph match score                              |
| `domain_delta`          | Additive domain alignment bonus/penalty (±0.15 max)      |
| `domain_codebleu`       | Final domain-weighted score                              |
| `pass_at_1`             | Estimated probability that one sample passes all tests   |

### 4.4 Pass@1 Estimation

Following Chen et al. (2021), Pass@1 is estimated without bias from
n independent samples:

```
Pass@1 = c / n

where c = number of samples that pass all pytest tests
      n = total samples (10 for zero-shot, 3 for agent)
```

---

## 5. Models Evaluated

### 5.1 Zero-Shot Models (via OpenRouter)

All models are accessed via the OpenRouter API
(https://openrouter.ai) using the OpenAI-compatible endpoint.
Temperature 0.8 is used for all zero-shot generation to encourage
diversity across the 10 samples per item.

| Alias    | Model ID                            | Provider  | Access         |
|----------|-------------------------------------|-----------|----------------|
| `gemini` | `google/gemini-3-flash-preview`     | Google    | Closed Source  |
| `GLM5`   | `z-ai/glm-5`                        | Z.AI      | Open Source    |
| `qwen`   | `qwen/qwen3-coder-next`             | Alibaba   | Open Source    |

### 5.2 Agent Models

The z-ai/Glm-5 model is used inside the agentic framework. Temperature
varies by agent role to balance consistency with creativity:

| Agent    | Temperature | Rationale                              |
|----------|-------------|----------------------------------------|
| Planner  | 0.2         | Deterministic task decomposition       |
| Coder    | 0.8         | Creative code generation               |
| Critic   | 0.2         | Consistent structured analysis         |
| Debugger | 0.4         | Grounded but flexible fixing           |

---

## 6. Zero-Shot Evaluation Results 

Results are stored in `<alias>_eval_results.json`.

### 6.1 Pass@1 (number of samples is 10) by Library

**SymPy PRs**

| PR       | gemini | GLM5  | qwen  |
|----------|--------|-------|-------|
| pr29550  | 0.200  | 0.900 | 0.300 |
| pr29633  | 0.900  | 0.600 | 0.000 |
| pr29394  | 0.800  | 0.400 | 0.500 |
| pr29369  | 0.100  | 0.700 | 0.000 |
| pr29093  | 0.000  | 0.000 | 0.000 |
| pr29263  | 0.000  | 0.000 | 0.000 |
| **AVG**  | 0.333  | 0.433 | 0.133 |

**QuTiP PRs**

| PR         | gemini | GLM5  | qwen  |
|------------|--------|-------|-------|
| pr2835     | 0.000  | 0.000 | 0.000 |
| **AVG**    | 0.000  | 0.000 | 0.000 |


**Combined Average**
|       | gemini | GLM5  | qwen  |
|-------|--------|-------|-------|
|**AVG**| 0.286  | 0.371 | 0.114 |

Results populated after running `evaluate.py`.

### 6.2 Domain CodeBLEU by Library

**SymPy PRs**

| PR       | gemini | GLM5   | qwen   |
|----------|--------|------- |------- |
| pr29550  | 0.9737 | 0.9867 | 0.4806 |
| pr29633  | 1.0000 | 0.8163 | 0.3240 |
| pr29394  | 0.9430 | 1.0000 | 0.7859 |
| pr29369  | 1.0000 | 0.7512 | 0.1542 |
| pr29093  | 0.9932 | 1.0000 | 1.0000 |
| pr29263  | 0.9684 | 0.9737 | 0.1504 |
| **AVG**  | 0.9797 | 0.9113 | 0.4925 |

**QuTiP PRs**

| PR         | gemini | GLM5   | qwen   |
|------------|--------|------- |------- |
| pr2835     | 1.0000 | 1.0000 | 1.0000 |
| **AVG**    | 1.0000 | 1.0000 | 1.0000 |

**Combined Average**

|         | gemini | GLM5   | qwen   |
|---------|--------|--------|------- |
| **AVG** | 0.9826 | 0.9326 | 0.5564 |

---

## 7. Agentic Framework

The agentic framework adapts the VerilogCoder multi-agent architecture
[(Tsai et al., arxiv:2408.08927)](https://arxiv.org/abs/2408.08927) 
 to scientific Python code generation
across both SymPy and QuTiP. The paper proposes three core ideas:
graph-based task planning (TCRG), role-separated LLM agents, and a
tool-assisted ReAct debugging loop.

### 7.1 Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │              Orchestrator               │
                    │                                         │
  Prompt ──────────►│  ┌──────────┐                           │
  (SymPy or         │  │ Planner  │── TCRG (DAG of sub-tasks) │
   QuTiP file)      │  └──────────┘                           │
                    │       │                                 │
                    │       ▼  (for each node, in topo order) │
                    │  ┌─────────────────────────────────┐    │
                    │  │         ReAct Loop              │    │
                    │  │                                 │    │
                    │  │  ┌───────┐    ┌─────────────┐   │    │
                    │  │  │ Coder │    │  Tool Suite │   │    │
                    │  │  └───┬───┘    │  1. Syntax  │   │    │
                    │  │      │code    │  2. AST Pat │   │    │
                    │  │      └───────►│  3. Physics │   │    │
                    │  │               │  4. pytest  │   │    │
                    │  │               └──────┬──────┘   │    │
                    │  │                      │report    │    │
                    │  │               ┌──────▼───────┐  │    │
                    │  │               │    Critic    │  │    │
                    │  │               └──────┬───────┘  │    │
                    │  │                      │critique  │    │
                    │  │               ┌──────▼───────┐  │    │
                    │  │               │   Debugger   │  │    │
                    │  │               └──────┬───────┘  │    │
                    │  │                      │fixed code│    │
                    │  │               (loop until pass  │    │
                    │  │                or budget done)  │    │
                    │  └─────────────────────────────────┘    │
                    │       │  best code per node             │
                    │       ▼                                 │
                    │  ┌──────────┐                           │
                    │  │Assembler │── final_assembled.py      │
                    │  └──────────┘                           │
                    └─────────────────────────────────────────┘
```

### 7.2 Planner Agent — Task-Conditioned Reasoning Graph

The Planner classifies every incoming prompt as `bugfix` or `feature`
before decomposing, by scanning for keyword signals:

- **bugfix keywords**: `issue`, `error`, `fix`, `bug`, `reproduce`,
  `incorrect`, `wrong`, `fails`, `exception`, `traceback`
- **feature keywords**: `feature`, `implement`, `add`, `create`,
  `support`, `introduce`, `extend`, `new`

**Bugfix tasks** produce exactly 1 node (2 at most). The node description
names the exact function to modify, quotes the reported error, and requires
the complete modified file to be returned — not a fragment.

**Feature tasks** produce 3–5 nodes decomposed by logical component:
data structures/constants → core algorithm → integration layer →
validation → assembly.

This classification handles both SymPy and QuTiP tasks uniformly — a QuTiP
`mesolve` bug fix is handled identically to a SymPy `bode_magnitude` bug
fix at the planning level.

### 7.3 Coder Agent

Receives one TCRG node at a time plus the full original prompt (which
contains the complete source file from either SymPy or QuTiP). The
system prompt requires the complete file to be returned rather than
just the changed fragment, which is essential for both libraries since their test suites import the entire module.

### 7.4 Tool Suite — Four Layers of Feedback

After every Coder or Debugger call, the Tool Suite runs in order,
short-circuiting on syntax failure:

**Tool 1 — Syntax Checker** (`tools/syntax_checker.py`)  
`ast.parse()` plus import completeness warnings. Detects missing
`numpy`, `scipy`, `qutip`, or `sympy` imports.

**Tool 2 — AST Pattern Checker** (`tools/ast_pattern.py`)  
Seven physics-specific bad-pattern detectors applicable to both libraries:
- `math.sqrt/sin/cos` on array arguments (should use numpy equivalents)
- Floating-point accumulation `x = x + dx` inside loops
- Angular frequency assigned without 2π factor
- Floor division `//` inside physics formulas
- Magic number timesteps without documented derivation
- Wavefunction/state vector updated in loop without normalisation
- Integer `range()` used for continuous time variable

**Tool 3 — Physics Sanity Checker** (`tools/physics_sanity.py`)  
Executes the code in a subprocess (12-second timeout) and checks:
- Code runs without runtime error
- Output contains no NaN or Inf values
- Printed energy-like or expectation-value quantities stay stable
- Code remains stable when iteration counts are doubled

**Tool 4 — pytest** (`tools/tool_suite.py`)  
Runs the benchmark test suite against the candidate file using the
cloned SymPy or QuTiP repository as the installed package. Only
enabled at the assembly stage for bugfix tasks.

### 7.5 Critic Agent

Produces a structured JSON critique with:
- `physical_errors`: physics-correctness problems
- `numerical_errors`: algorithmic or numerical problems  
- `missing_cases`: absent edge cases or validations
- `severity`: `low | medium | high`
- `actionable_fixes`: concrete numbered fix instructions

Includes a regex-based fallback parser for models that wrap JSON in
prose or use non-standard formatting.

### 7.6 Debug Agent

Receives the full original prompt (with source file), current code,
structured critique, and tool report. Produces a corrected complete
file addressing every item in `actionable_fixes`.

### 7.7 ReAct Loop

```
for iteration in 1..max_iter:
    THINK  → What does the spec require?
    ACT    → Coder (iter 1) or Debugger (iter 2+)
    STRIP  → Remove markdown fences from output
    OBSERVE → Run Tool Suite → structured ToolReport
    if all_passed:
        accept and break
    THINK  → Critic produces structured critique
    (next iteration)
return best code seen (tracked by tool pass score)
```

### 7.8 Assembler

Detects whether nodes output complete files or true fragments:
- **Complete file detected** (≥50 lines, ≥5 imports, ≥5 top-level
  definitions): returns the last node's output — no concatenation
- **Fragment outputs**: de-duplicates imports and concatenates body
  sections with node headers
- **Bugfix tasks**: always returns the last node's output directly

After assembly, a syntax check validates the result. On failure, the
assembler falls back to the last syntactically valid node output.

### 7.9 Trace Saving

Every intermediate output is saved to disk in real time:

```
agent_traces/<alias>/<pr_name>/sample_<n>/
    prompt.txt                    ← exact prompt sent to planner
    tcrg.json                     ← planner output with task_type
    run_summary.json              ← per-node iteration scores
    final_assembled.py            ← what gets scored
    <node_id>/
        node_spec.json            ← TCRG node definition
        context.py                ← dependency code the coder saw
        node_summary.json         ← iteration score table
        iter_1_generated.py       ← coder output
        iter_1_tool_report.json
        iter_1_critique.json
        iter_2_debugged.py        ← debugger output
        ...
        best.py                   ← highest-scoring iteration
```

---

## 8. Agent Evaluation Results

Results are stored in `<alias>_agent_results.json`. A comparison table
against zero-shot scores is printed at the end of each run.

### 8.1 Pass@1 Comparison (3 samples as 10 would consume too many credits)


PR            ZS Pass@1  ZS Dom-CB   AG Pass@1  AG Dom-CB   Δ Pass@1
----------------------------------------------------------------------------------
pr29550          0.900    0.9867       1.000    0.9883  ▲  +0.100
pr29633          0.600    0.8163       1.000    1.0000  ▲  +0.400
pr29394          0.400    1.0000       1.000    1.0000  ▲  +0.600
pr29369          0.700    0.7512       1.000    1.0000  ▲  +0.300
pr29093          0.000    1.0000       0.333    1.0000  ▲  +0.333
pr29263          0.000    0.9737       0.000    0.9748     +0.000
pr2835           0.000    1.0000       0.000    1.0000     +0.000

**SymPy PRs — GLM5**

| PR       | Zero-Shot | Agent | Δ Pass@1|
|----------|-----------|-------|---------|
| pr29550  | 0.900     | 1.000 | +0.100  |
| pr29633  | 0.600     | 1.000 | +0.400  |
| pr29394  | 0.400     | 1.000 | +0.600  |
| pr29369  | 0.700     | 1.000 | +0.300  |
| pr29093  | 0.000     | 0.333 | +0.333  | 
| pr29263  | 0.000     | 0.000 |  0.000  | 

**QuTiP PRs — GLM5**

| PR         | Zero-Shot | Agent | Δ Pass@1|
|------------|-----------|-------|---------|
| pr2835     | 0.000     | 0.000 | 0       |

**Combined Average**

|          | Zero-Shot | Agent | Δ Pass@1 |
|----------|-----------|-------|----------|
|**AVG**   | 0.371     | 0.619 | +0.248   |

Results populated after running `evaluate_agent.py`.

### 8.2 Expected Improvement Conditions

The agent framework is expected to improve Pass@1 when:
- Zero-shot code runs but is numerically wrong (physics sanity checker
  catches what pytest misses)
- The task is a multi-component feature requiring layered assembly
- The model makes consistent, fixable mistakes across zero-shot samples

The agent is not expected to improve when:
- Zero-shot already achieves Pass@1 ≥ 0.9
- The error requires library-internal knowledge the model does not have

---

## 9. Project Structure
```
.
├── README.md
├── requirements.txt
│
├── evaluate.py                  ← zero-shot multi-model evaluator
├── evaluate_agent.py            ← agent evaluator
├── orchestrator.py              ← multi-agent pipeline controller
├── react_loop.py                ← ReAct iteration loop
├── domain_weights.py            ← domain weight construction script
├── collect_corpus.py            ← PR collection and corpus builder
├── check.py                     ← quick sanity check utility
│
├── agent/
│   ├── __init__.py
│   ├── utils.py                 ← strip_code_fences and shared helpers
│   ├── planner.py               ← Planner agent (TCRG + task_type)
│   ├── coder.py                 ← Coder agent
│   ├── critic.py                ← Critic agent
│   └── debugger.py              ← Debug agent
│
├── tools/
│   ├── __init__.py
│   ├── syntax_checker.py        ← Tool 1: ast.parse + import warnings
│   ├── ast_pattern.py           ← Tool 2: physics bad-pattern detection
│   ├── physics_sanity.py        ← Tool 3: subprocess execution + checks
│   └── tool_suite.py            ← Tool 4: pytest runner + aggregator
│
├── manual_corpus/
│   ├── benchmark_config.json    ← lists all benchmark items
│   ├── benchmark_config.json
│   ├── benchmark_results.json
│   ├── benchmark_runner.py
│   ├── extract_pr.sh            ← shell script to extract PR diffs
│   ├── sympy/                   ← raw SymPy PR data
│   ├── qutip/                   ← raw QuTiP PR data
│   ├── test-1/                  ← SymPy benchmark items
│   │   ├── pr29550/
│   │   │   ├── metadata.json
│   │   │   ├── prompt.txt
│   │   │   ├── before.py
│   │   │   ├── after.py
│   │   │   └── test_*.py
│   │   └── ...
│   └── test-2/                  ← QuTiP benchmark items
│       ├── qutip-pr1/
│       │   ├── metadata.json
│       │   ├── prompt.txt
│       │   ├── before.py
│       │   ├── after.py
│       │   └── test_*.py
│       └── ...
│
├── cache/                       ← zero-shot API response cache
│   ├── gemini/
│   ├── GLM5/
│   └── Qwen3_coder/
│
├── agent_traces/                ← full agent processing traces
│   └── GLM5/
│
├── domain_weights.json          ← computed domain weights (auto-generated)
│
├── gemini_eval_results.json     ← zero-shot results
├── GLM5_eval_results.json
├── Qwen3_coder_eval_results.json
└── GLM5_agent_results.json      ← agent results
```
## 10. Installation

```bash
# Clone the repository
git clone <repo-url>
cd repo_name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
openai>=1.0.0
codebleu>=0.6.0
python-dotenv>=1.0.0
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.0.0
sympy>=1.12
qutip>=5.0.0
```

Set your API key in .env:
```bash
#.env
# OPENROUTER_API_KEY=your_key_here
```

---

## 11. Usage

### Zero-Shot Evaluation

```bash
# Single model
python evaluate.py --models GLM5:z-ai/glm-5

# Multiple models
python evaluate.py \
  --models gemini:google/gemini-flash-1.5 \
           GLM5:z-ai/glm-5 \
           qwen:qwen/qwen-2.5-coder-32b-instruct

# Recompute CodeBLEU from cached samples only (no API calls, no pytest)
python evaluate.py --models GLM5:z-ai/glm-5 --bleu-only

# Rescore specific PRs (re-run pytest, ignore cached scores)
python evaluate.py --models GLM5:z-ai/glm-5 --rescore pr29550 qutip-pr1

# Rescore everything
python evaluate.py --models GLM5:z-ai/glm-5 --rescore
```

### Agent Evaluation

```bash
# Default settings (3 samples, 3 ReAct iters/node)
python evaluate_agent.py --models GLM5:z-ai/glm-5

# Multiple models
python evaluate_agent.py \
  --models gemini:google/gemini-flash-1.5 \
           GLM5:z-ai/glm-5

# More samples for tighter Pass@1 estimate
python evaluate_agent.py --models GLM5:z-ai/glm-5 --n-samples 10

# Wider ReAct budget
python evaluate_agent.py --models GLM5:z-ai/glm-5 --max-iter 5

# Rescore specific PRs
python evaluate_agent.py --models GLM5:z-ai/glm-5 --rescore pr29394 qutip-pr2

# Rescore all
python evaluate_agent.py --models GLM5:z-ai/glm-5 --rescore
```

---

## 12. Reproducing Results

```bash
# Step 1: Zero-shot evaluation — all three models, 10 samples each
# Domain weights are built automatically on first run
python evaluate.py \
  --models gemini:google/gemini-flash-1.5 \
           GLM5:z-ai/glm-5 \
           qwen:qwen/qwen-2.5-coder-32b-instruct

# Step 2: Agent evaluation — all three models, 3 samples each
python evaluate_agent.py \
  --models gemini:google/gemini-flash-1.5 \
           GLM5:z-ai/glm-5 \
           qwen:qwen/qwen-2.5-coder-32b-instruct \
  --n-samples 3 \
  --max-iter 3

# Results files produced:
#   domain_weights.json
#   gemini_eval_results.json     ← zero-shot
#   GLM5_eval_results.json
#   qwen_eval_results.json
#   gemini_agent_results.json    ← agent
#   GLM5_agent_results.json
#   qwen_agent_results.json
#   agent_traces/                ← full processing traces per sample
```

---

## 13. References

- Ren, S. et al. (2020). CodeBLEU: a Method for Automatic Evaluation of
  Code Synthesis. arXiv:2009.10297

- Chen, M. et al. (2021). Evaluating Large Language Models Trained on
  Code. arXiv:2107.03374

- Tsai, Y. et al. (2024). VerilogCoder: Autonomous Verilog Coding Agents
  with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform
  Tracing Tool. arXiv:2408.08927

- SymPy Development Team (2023). SymPy: Python library for symbolic
  mathematics. https://www.sympy.org

- Johansson, J.R. et al. (2013). QuTiP 2: A Python framework for the
  dynamics of open quantum systems. Computer Physics Communications 184,
  1234–1240. https://qutip.org

- OpenRouter API. https://openrouter.ai

---

## License

This project is released under the MIT License.

The SymPy source files used as benchmark inputs are licensed under the
BSD 3-Clause License (https://github.com/sympy/sympy/blob/master/LICENSE).

The QuTiP source files used as benchmark inputs are licensed under the
BSD 3-Clause License (https://github.com/qutip/qutip/blob/master/LICENSE.txt).