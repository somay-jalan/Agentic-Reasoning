# optimize.py
# Correct GEPA implementation based on actual dspy.GEPA API docs.

import json
import dspy
from dspy.teleprompt.gepa import GEPA
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric, ScoreWithFeedback
from typing import Optional, Union
from pathlib import Path

from config import (OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    GEMINI_MODEL, MAX_TOKENS, TEMPERATURE, MODEL_FOLDER)
from modules import IneqMathJudgeAgent
from training_examples import load_examples
from metrics import ineqmath_metric, ineqmath_strict


# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR          = Path("results") / MODEL_FOLDER
VERIFIED_DIR         = Path("results") / f"{MODEL_FOLDER}_verified"
OPTIMIZED_DIR        = Path("results") / f"{MODEL_FOLDER}_optimized"
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZED_MODEL_PATH = OPTIMIZED_DIR / "optimized_judge_agent.json"


# ── DSPy setup ────────────────────────────────────────────────────────────────
def configure_dspy():
    lm = dspy.LM(
        model=f"openai/{GEMINI_MODEL}",
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        cache=False,
    )
    dspy.configure(lm=lm)
    print(f"✅ DSPy configured: {GEMINI_MODEL}\n", flush=True)


# ── Agent wrapper ─────────────────────────────────────────────────────────────
class JudgeAgentWrapper(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = IneqMathJudgeAgent()

    def forward(self,
                inequality_key: str,
                problem_statement: str,
                predicted_answer: str,
                proof: str) -> dspy.Prediction:

        result = self.agent(
            inequality_key=inequality_key,
            problem_statement=problem_statement,
            predicted_answer=predicted_answer,
            proof=proof,
        )
        return dspy.Prediction(
            judges=result["judges"],
            overall_verdict=result["overall_verdict"],
        )


# ── GEPA Feedback Metric ──────────────────────────────────────────────────────
# GEPAFeedbackMetric is a Protocol — implement __call__ directly.
#
# GEPA calls this with pred_name = the name of the specific predictor
# currently being optimized (e.g. "agent.ntc_judge", "agent.nlg_judge").
# We use pred_name to give TARGETED feedback for that specific judge
# rather than generic program-level feedback.
#
# Return options:
#   float                                    → score only, GEPA makes up feedback
#   dspy.Prediction(score=..., feedback=...) → score + targeted feedback

# Maps DSPy predictor names to our judge keys and expected verdict fields
# Map predictor name → judge key
PREDICTOR_TO_JUDGE = {
    "agent.final_answer_judge.predict": ("final_answer", "expected_fa",  "Final Answer Judge"),
    "agent.ntc_judge.predict":          ("ntc",          "expected_ntc", "No Toy Case Judge"),
    "agent.nlg_judge.predict":          ("nlg",          "expected_nlg", "No Logical Gap Judge"),
    "agent.nae_judge.predict":          ("nae",          "expected_nae", "No Approximation Error Judge"),
    "agent.nce_judge.predict":          ("nce",          "expected_nce", "No Calculation Error Judge"),
}


class IneqMathGEPAMetric:
    def __init__(self):
        self.call_count = 0

    def __call__(self, gold, pred, trace=None,
                 pred_name=None, pred_trace=None):

        self.call_count += 1
        call_id = self.call_count

        if isinstance(pred, dspy.Prediction):
            judges = pred.judges if hasattr(pred, "judges") else {}
        elif isinstance(pred, dict):
            judges = pred.get("judges", {})
        else:
            print(f"  [Call {call_id}] ❌ Could not parse prediction", flush=True)
            return dspy.Prediction(score=0.0,
                feedback="Could not parse.") if pred_name else 0.0

        # ── Mode 1: program-level ─────────────────────────────────────────
        if pred_name is None:
            score = ineqmath_metric(gold, {"judges": judges})
            print(
                f"\n  [Call {call_id}] MODE=program-level"
                f"\n    example  : {gold.inequality_key} ({gold.label})"
                f"\n    trace    : {'optimization' if trace else 'final_eval'}"
                f"\n    score    : {score:.2f}"
            )
            return score

        # ── Mode 2: predictor-level ───────────────────────────────────────
        judge_info = PREDICTOR_TO_JUDGE.get(pred_name)

        if judge_info is None:
            score = ineqmath_metric(gold, {"judges": judges})
            print(f"  [Call {call_id}] ⚠️  Unknown predictor: {pred_name}", flush=True)
            return dspy.Prediction(score=score, feedback=f"Unknown: {pred_name}")

        judge_key, expected_field, judge_name = judge_info

        # Per-judge score
        per_judge_score = ineqmath_metric(
            gold, {"judges": judges},
            judge_key=judge_key
        )

        # What the judge actually said
        judge_output     = judges.get(judge_key, {})
        predicted        = judge_output.get("verdict", "").upper().strip()
        expected         = getattr(gold, expected_field, "PASS").upper().strip()
        predicted_binary = "PASS" if "PASS" in predicted else "FAIL"
        judge_correct    = predicted_binary == expected
        label            = getattr(gold, "label", "unknown")
        reason           = judge_output.get("reason", "")[:80]

        # ── This is the key print — shows if improvement is detectable ────
        print(
            f"\n  [Call {call_id}] MODE=predictor-level"
            f"\n    predictor : {pred_name}"
            f"\n    example   : {gold.inequality_key} ({label})"
            f"\n    trace     : {'optimization' if trace else 'final_eval'}"
            f"\n    predicted : {predicted_binary}"
            f"\n    expected  : {expected}"
            f"\n    correct   : {'✅' if judge_correct else '❌'}"
            f"\n    score     : {per_judge_score:.2f}  ← per-judge score"
            f"\n    reason    : {reason}"
        )

        # Build feedback
        if judge_correct:
            feedback = (
                f"{judge_name} correctly predicted {predicted_binary} "
                f"(expected {expected}) on '{label}'. Reason: '{reason}'."
            )
        else:
            error_type = (
                "FALSE POSITIVE — said PASS, should be FAIL"
                if expected == "FAIL"
                else
                "FALSE NEGATIVE — said FAIL, should be PASS"
            )
            feedback = (
                f"{judge_name} WRONG on '{label}'.\n"
                f"Error     : {error_type}\n"
                f"Predicted : {predicted_binary}\n"
                f"Expected  : {expected}\n"
                f"Reason    : '{reason}'\n"
                f"Proof     : '{getattr(gold, 'proof', '')[:300]}...'\n\n"
                + (
                    f"Make {judge_name} MORE sensitive to '{label}' flaws."
                    if expected == "FAIL"
                    else
                    f"Make {judge_name} MORE precise, stop flagging valid proofs."
                )
            )

        return dspy.Prediction(score=per_judge_score, feedback=feedback)
    


# ── Load baseline from saved JSONs ────────────────────────────────────────────
def load_baseline_scores(testset: list, agent: JudgeAgentWrapper) -> list[float]:
    scores = []
    print("="*60, flush=True)
    print("  BASELINE — loaded from saved run_dspy.py JSONs", flush=True)
    print("="*60, flush=True)

    for ex in testset:
        key   = ex.inequality_key
        label = ex.label

        # Positive → key.json, Negative → key_label.json
        json_path = (
            VERIFIED_DIR / f"{key}.json"
            if label == "positive"
            else VERIFIED_DIR / f"{key}_{label}.json"
        )

        if json_path.exists():
            # ── Load from disk (works for both positive and negative) ─────
            data   = json.loads(json_path.read_text(encoding="utf-8"))
            judges = data.get("verification", {}).get("judges", {})
            score  = ineqmath_strict(ex, {"judges": judges}) if judges else 0.0
            print(f"  [{label:<15}] {key:<20} score={score:.2f}  (from JSON)", flush=True)

        else:
            # ── Not saved yet → run agent and save ────────────────────────
            print(f"  [{label:<15}] {key:<20} not found, running agent...", flush=True)
            try:
                pred  = agent(
                    inequality_key=ex.inequality_key,
                    problem_statement=ex.problem_statement,
                    predicted_answer=ex.predicted_answer,
                    proof=ex.proof,
                )
                score = ineqmath_strict(ex, pred)   # final eval → strict

                save_data = {
                    "inequality_key":    key,
                    "inequality_name":   key.replace("_", " ").title(),
                    "label":             label,
                    "model":             GEMINI_MODEL,
                    "problem_statement": ex.problem_statement,
                    "proof":             ex.proof,
                    "verification": {
                        "judges":          pred.judges,
                        "overall_verdict": pred.overall_verdict,
                    },
                    "baseline_score": round(score, 4),
                }
                json_path.write_text(
                    json.dumps(save_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"             💾 Saved → {json_path.name}", flush=True)

            except Exception as e:
                print(f"  [{label:<15}] {key:<20} ERROR: {e} → 0.0", flush=True)
                score = 0.0

        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  📊 Baseline avg (test): {avg:.4f}\n", flush=True)
    return scores



# ── Parse and save GEPA detailed_results ─────────────────────────────────────
def safe_serialize(obj):
    """Recursively convert any non-JSON-serializable object to string."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(i) for i in obj]
    elif isinstance(obj, set):
        return sorted([safe_serialize(i) for i in obj])
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_gepa_stats(optimized_agent, testset: list):
    if not hasattr(optimized_agent, "detailed_results"):
        print("  ⚠️  No detailed_results found", flush=True)
        return

    r = optimized_agent.detailed_results

    print("\n" + "="*60, flush=True)
    print("  GEPA OPTIMIZATION STATS", flush=True)
    print("="*60, flush=True)
    print(f"  Total metric calls    : {r.total_metric_calls}", flush=True)
    print(f"  Full val evaluations  : {r.num_full_val_evals}", flush=True)
    print(f"  Candidates proposed   : {len(r.candidates)}", flush=True)
    print(f"  Best candidate index  : {r.best_idx}", flush=True)
    print(f"  Best candidate score  : {r.val_aggregate_scores[r.best_idx]:.4f}", flush=True)
    print(f"  Log dir               : {r.log_dir}", flush=True)

    print(f"\n  All candidate scores:", flush=True)
    for i, (score, disc) in enumerate(
        zip(r.val_aggregate_scores, r.discovery_eval_counts)
    ):
        marker = " ← BEST" if i == r.best_idx else ""
        print(f"    Candidate {i:2d}: score={score:.4f}  "
              f"discovered at metric_call={disc}{marker}")

    print(f"\n  Per-val-instance best candidates:", flush=True)
    for t, best in enumerate(r.per_val_instance_best_candidates):
        ex_key = testset[t].inequality_key if t < len(testset) else f"ex_{t}"
        best_str = str(sorted(best)) if isinstance(best, set) else str(best)
        print(f"    {ex_key:<20} best candidate: {best_str}", flush=True)

    # ── Build stats dict and sanitize everything ──────────────────────────
    stats = safe_serialize({
        "total_metric_calls":              r.total_metric_calls,
        "num_full_val_evals":              r.num_full_val_evals,
        "num_candidates":                  len(r.candidates),
        "best_idx":                        r.best_idx,
        "best_score":                      r.val_aggregate_scores[r.best_idx],
        "val_aggregate_scores":            r.val_aggregate_scores,
        "val_subscores":                   r.val_subscores,
        "discovery_eval_counts":           r.discovery_eval_counts,
        "per_val_instance_best_candidates": list(r.per_val_instance_best_candidates),
        "best_candidate":                  r.best_candidate,
        "log_dir":                         r.log_dir,
        "seed":                            r.seed,
    })

    stats_path = OPTIMIZED_DIR / "gepa_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n  💾 GEPA stats → {stats_path}", flush=True)


def evaluate_per_judge(agent, testset: list, label_prefix: str) -> dict:
    """
    Runs agent on testset and returns per-judge accuracy broken down
    by example and judge type.
    """
    results = {}

    for ex in testset:
        key = ex.inequality_key
        try:
            pred = agent(
                inequality_key=ex.inequality_key,
                problem_statement=ex.problem_statement,
                predicted_answer=ex.predicted_answer,
                proof=ex.proof,
            )
            judges = pred.judges if hasattr(pred, "judges") else {}
        except Exception as e:
            print(f"  ❌ Error on {key}: {e}", flush=True)
            judges = {}

        judge_map = {
            "final_answer": "expected_fa",
            "ntc":          "expected_ntc",
            "nlg":          "expected_nlg",
            "nae":          "expected_nae",
            "nce":          "expected_nce",
        }

        per_judge = {}
        for judge_key, expected_field in judge_map.items():
            predicted        = judges.get(judge_key, {}).get("verdict", "").upper()
            expected         = getattr(ex, expected_field, "PASS").upper()
            predicted_binary = "PASS" if "PASS" in predicted else "FAIL"
            correct          = predicted_binary == expected
            per_judge[judge_key] = {
                "predicted": predicted_binary,
                "expected":  expected,
                "correct":   correct,
            }

        results[key] = {
            "label":     ex.label,
            "per_judge": per_judge,
        }

    return results


def print_per_judge_comparison(baseline_results: dict,
                                optimized_results: dict,
                                testset: list):
    """
    Prints a per-judge breakdown for negative examples only,
    showing baseline vs optimized verdict for each judge.
    """
    judge_names = {
        "final_answer": "FA",
        "ntc":          "NTC",
        "nlg":          "NLG",
        "nae":          "NAE",
        "nce":          "NCE",
    }

    neg_examples = [ex for ex in testset if ex.label != "positive"]

    print("\n" + "="*60, flush=True)
    print("  PER-JUDGE BREAKDOWN — NEGATIVE EXAMPLES ONLY", flush=True)
    print("="*60, flush=True)
    print(f"\n  {'Example':<22} {'Label':<16} {'Judge':<6} "
          f"{'Expected':<10} {'Baseline':<12} {'Optimized':<12} {'Δ'}")
    print(f"  {'─'*80}", flush=True)

    # Track per-judge aggregate improvement
    judge_correct_baseline  = {j: 0 for j in judge_names}
    judge_correct_optimized = {j: 0 for j in judge_names}
    total_neg = len(neg_examples)

    for ex in neg_examples:
        key = ex.inequality_key
        b   = baseline_results.get(key, {}).get("per_judge", {})
        o   = optimized_results.get(key, {}).get("per_judge", {})

        first_row = True
        for judge_key, judge_short in judge_names.items():
            b_info = b.get(judge_key, {})
            o_info = o.get(judge_key, {})

            b_correct = b_info.get("correct", False)
            o_correct = o_info.get("correct", False)
            b_pred    = b_info.get("predicted", "?")
            o_pred    = o_info.get("predicted", "?")
            expected  = b_info.get("expected", "?")

            judge_correct_baseline[judge_key]  += int(b_correct)
            judge_correct_optimized[judge_key] += int(o_correct)

            # Change indicator
            if o_correct and not b_correct:
                delta = "⬆️ FIXED"
            elif not o_correct and b_correct:
                delta = "⬇️ BROKE"
            elif o_correct and b_correct:
                delta = "✅ same"
            else:
                delta = "❌ same"

            # Only print example name and label on first judge row
            ex_col    = key if first_row else ""
            label_col = ex.label if first_row else ""
            first_row = False

            b_icon = "✅" if b_correct else "❌"
            o_icon = "✅" if o_correct else "❌"

            print(f"  {ex_col:<22} {label_col:<16} {judge_short:<6} "
                  f"{expected:<10} "
                  f"{b_icon} {b_pred:<9} "
                  f"{o_icon} {o_pred:<9} "
                  f"{delta}")

        print(f"  {'─'*80}", flush=True)

    # Per-judge aggregate
    print(f"\n  PER-JUDGE ACCURACY ON NEGATIVES ({total_neg} examples)", flush=True)
    print(f"  {'Judge':<6} {'Baseline':>10} {'Optimized':>10} {'Δ':>8}", flush=True)
    print(f"  {'─'*40}", flush=True)
    for judge_key, judge_short in judge_names.items():
        b_acc = judge_correct_baseline[judge_key]  / total_neg if total_neg else 0
        o_acc = judge_correct_optimized[judge_key] / total_neg if total_neg else 0
        delta = o_acc - b_acc
        arrow = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")
        print(f"  {judge_short:<6} {b_acc:>10.2f} {o_acc:>10.2f} "
              f"{delta:>+8.2f} {arrow}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_gepa_optimization():
    print("Loading training/test examples...", flush=True)
    trainset, valset, testset = load_examples(RESULTS_DIR)  

    if not trainset:
        print("❌ No examples found. Run run_all.py first.", flush=True)
        return

    # Need a baseline agent instance for running negatives
    baseline_agent  = JudgeAgentWrapper()
    baseline_scores = load_baseline_scores(testset, baseline_agent)
    baseline_avg    = sum(baseline_scores) / len(baseline_scores) \
                      if baseline_scores else 0.0

    # ── GEPA ──────────────────────────────────────────────────────────────
    print("="*60, flush=True)
    print("  GEPA OPTIMIZATION", flush=True)
    print("="*60, flush=True)

    reflection_lm = dspy.LM(
        model=f"openai/{GEMINI_MODEL}",
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        max_tokens=MAX_TOKENS,
        temperature=0.9,               # ← higher = more creative mutations
        cache=False,
    )

    optimizer = GEPA(
        metric=IneqMathGEPAMetric(),
        max_full_evals = 5,
        reflection_minibatch_size=10,
        candidate_selection_strategy="pareto",
        reflection_lm=reflection_lm, 
        skip_perfect_score=True,
        add_format_failure_as_feedback=True,
        use_merge=True,
        max_merge_invocations=3,
        num_threads=4,
        failure_score=0.0,
        perfect_score=1.0,
        log_dir=str(OPTIMIZED_DIR / "gepa_logs"),
        track_stats=True,              # enables detailed_results
        seed=42,
    )

    optimized_agent = optimizer.compile(
        JudgeAgentWrapper(),
        trainset=trainset,
        valset=valset,
    )

    # ── Parse GEPA stats ──────────────────────────────────────────────────
    save_gepa_stats(optimized_agent, testset)

    # ── Evaluate optimized on TEST set ────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("  OPTIMIZED evaluation on TEST set", flush=True)
    print("="*60, flush=True)
    optimized_scores = []

    for ex in testset:
        try:
            pred  = optimized_agent(
                inequality_key=ex.inequality_key,
                problem_statement=ex.problem_statement,
                predicted_answer=ex.predicted_answer,
                proof=ex.proof,
            )
            score = ineqmath_strict(ex, pred)
            optimized_scores.append(score)
            print(f"  [{ex.label:<15}] {ex.inequality_key:<20} score={score:.2f}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            optimized_scores.append(0.0)

    optimized_avg = sum(optimized_scores) / len(optimized_scores) \
                    if optimized_scores else 0.0
    
    # ── Load baseline per-judge from saved JSONs ──────────────────────────
    print("\n  Computing per-judge baseline from saved JSONs...", flush=True)
    baseline_per_judge = {}
    for ex in testset:
        key       = ex.inequality_key
        label     = ex.label
        json_path = (
            VERIFIED_DIR / f"{key}.json"
            if label == "positive"
            else VERIFIED_DIR / f"{key}_{label}.json"
        )
        if json_path.exists():
            data   = json.loads(json_path.read_text(encoding="utf-8"))
            judges = data.get("verification", {}).get("judges", {})
            judge_map = {
                "final_answer": "expected_fa",
                "ntc":          "expected_ntc",
                "nlg":          "expected_nlg",
                "nae":          "expected_nae",
                "nce":          "expected_nce",
            }
            per_judge = {}
            for judge_key, expected_field in judge_map.items():
                predicted        = judges.get(judge_key, {}).get("verdict", "").upper()
                expected         = getattr(ex, expected_field, "PASS").upper()
                predicted_binary = "PASS" if "PASS" in predicted else "FAIL"
                per_judge[judge_key] = {
                    "predicted": predicted_binary,
                    "expected":  expected,
                    "correct":   predicted_binary == expected,
                }
            baseline_per_judge[key] = {"label": label, "per_judge": per_judge}

    # ── Evaluate optimized per-judge ──────────────────────────────────────
    print("  Computing per-judge optimized results...", flush=True)
    optimized_per_judge = evaluate_per_judge(optimized_agent, testset,
                                             label_prefix="optimized")

    # ── Print comparison ──────────────────────────────────────────────────
    delta = optimized_avg - baseline_avg
    arrow = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")

    print("\n" + "="*60, flush=True)
    print("  FINAL RESULTS — Overall", flush=True)
    print("="*60, flush=True)
    print(f"  {'Example':<20} {'Label':<15} {'Baseline':>10} "
          f"{'Optimized':>10} {'Δ':>8}")
    print(f"  {'─'*65}", flush=True)
    for ex, b, o in zip(testset, baseline_scores, optimized_scores):
        print(f"  {ex.inequality_key:<20} {ex.label:<15} "
              f"{b:>10.2f} {o:>10.2f} {o-b:>+8.2f}")
    print(f"  {'─'*65}", flush=True)
    print(f"  {'AVERAGE':<20} {'':<15} "
          f"{baseline_avg:>10.4f} {optimized_avg:>10.4f} "
          f"{delta:>+8.4f}  {arrow}")

    # ── Per-judge breakdown for negatives ─────────────────────────────────
    print_per_judge_comparison(baseline_per_judge, optimized_per_judge, testset)

    # ── Final comparison ──────────────────────────────────────────────────
    delta = optimized_avg - baseline_avg
    arrow = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")

    print("\n" + "="*60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"  {'Example':<20} {'Label':<15} {'Baseline':>10} {'Optimized':>10} {'Δ':>8}", flush=True)
    print(f"  {'─'*65}", flush=True)
    for ex, b, o in zip(testset, baseline_scores, optimized_scores):
        print(f"  {ex.inequality_key:<20} {ex.label:<15} "
              f"{b:>10.2f} {o:>10.2f} {o-b:>+8.2f}")
    print(f"  {'─'*65}", flush=True)
    print(f"  {'AVERAGE':<20} {'':<15} "
          f"{baseline_avg:>10.4f} {optimized_avg:>10.4f} {delta:>+8.4f}  {arrow}")

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "model":             GEMINI_MODEL,
        "optimizer":         "dspy.GEPA",
        "baseline_source":   "run_dspy.py saved JSONs",
        "baseline_avg":      round(baseline_avg, 4),
        "optimized_avg":     round(optimized_avg, 4),
        "delta":             round(delta, 4),
        "per_example": [
            {
                "inequality_key":  ex.inequality_key,
                "label":           ex.label,
                "baseline_score":  round(b, 4),
                "optimized_score": round(o, 4),
                "delta":           round(o - b, 4),
            }
            for ex, b, o in zip(testset, baseline_scores, optimized_scores)
        ],
    }
    report_path = OPTIMIZED_DIR / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    optimized_agent.save(str(OPTIMIZED_MODEL_PATH))
    print(f"\n  💾 Report          → {report_path}", flush=True)
    print(f"  💾 Optimized agent → {OPTIMIZED_MODEL_PATH}", flush=True)

    return optimized_agent


if __name__ == "__main__":
    configure_dspy()
    run_gepa_optimization()