# metrics.py
# Strict metric: 1.0 if ALL judges correct, 0.0 otherwise.
# No partial credit — mirrors IneqMath "overall PASS" definition.

# metrics.py

def ineqmath_metric(example, prediction, judge_key=None) -> float:
    """Partial credit: correct judges / total judges. Used during optimization."""
    
    if isinstance(prediction, dict):
        judges = prediction.get("judges", {})
    elif hasattr(prediction, "judges"):
        judges = prediction.judges
    else:
        return 0.0

    if not judges:
        return 0.0

    judge_map = {
        "final_answer": "expected_fa",
        "ntc":          "expected_ntc",
        "nlg":          "expected_nlg",
        "nae":          "expected_nae",
        "nce":          "expected_nce",
    }

    # ── Per-judge scoring (during GEPA predictor optimization) ────────────
    if judge_key is not None:
        expected_field   = judge_map.get(judge_key)
        predicted        = judges.get(judge_key, {}).get("verdict", "").upper()
        expected         = getattr(example, expected_field, "PASS").upper()
        predicted_binary = "PASS" if "PASS" in predicted else "FAIL"
        return 1.0 if predicted_binary == expected else 0.0

    # ── All-judges scoring ────────────────────────────────────────────────
    scores = []
    for jk, ef in judge_map.items():
        predicted        = judges.get(jk, {}).get("verdict", "").upper()
        expected         = getattr(example, ef, "PASS").upper()
        predicted_binary = "PASS" if "PASS" in predicted else "FAIL"
        scores.append(1.0 if predicted_binary == expected else 0.0)

    return sum(scores) / len(scores)


def ineqmath_strict(example, prediction) -> float:
    """Strict binary: 1.0 only if ALL judges correct. Used for final eval."""
    score = ineqmath_metric(example, prediction)
    return 1.0 if score == 1.0 else 0.0


def make_metric_for_optimizer():
    def metric(example, prediction, trace=None) -> float:
        return ineqmath_metric(example, prediction, trace)
    return metric



def make_metric_for_optimizer():
    def metric(example, prediction, trace=None) -> float:
        return ineqmath_metric(example, prediction, trace)
    return metric