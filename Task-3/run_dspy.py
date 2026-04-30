# run_dspy.py
# Load saved proofs → run all 5 IneqMath judges via DSPy → save results.

import json
import time
import dspy
from pathlib import Path

from config import (OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    GEMINI_MODEL, MAX_TOKENS, TEMPERATURE, MODEL_FOLDER)
from modules import IneqMathJudgeAgent


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
    print(f"✅ DSPy configured: {GEMINI_MODEL}\n")


def load_results(results_dir: Path) -> list[dict]:
    proofs = []
    for json_file in sorted(results_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if data.get("proof"):
            proofs.append(data)
            print(f"  📂 Loaded: {json_file.name}")
        else:
            print(f"  ⚠️  Skipped (no proof): {json_file.name}")
    return proofs


def print_judge_results(name: str, result: dict):
    judges = result["judges"]
    print(f"\n  Judges for: {name}")
    print(f"  {'─'*50}")

    icons = {"PASS": "✅", "FAIL": "❌"}
    rows = [
        ("Final Answer", judges["final_answer"]),
        ("NTC (No Toy Case)",          judges["ntc"]),
        ("NLG (No Logical Gap)",       judges["nlg"]),
        ("NAE (No Approx Error)",      judges["nae"]),
        ("NCE (No Calc Error)",        judges["nce"]),
    ]
    for label, j in rows:
        icon = icons.get(j["verdict"], "❓")
        conf = f"({j['confidence']})" if "confidence" in j else ""
        print(f"  {icon} {label:<28} {j['verdict']} {conf}")
        print(f"       → {j['reason'][:90]}")

    overall_icon = icons.get(result["overall_verdict"], "❓")
    print(f"\n  {overall_icon} OVERALL: {result['overall_verdict']}")
    print(f"  {'─'*50}")


def run_all():
    results_dir  = Path("results") / MODEL_FOLDER
    verified_dir = Path("results") / f"{MODEL_FOLDER}_verified_bad"
    verified_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from : {results_dir.resolve()}")
    print(f"Saving to    : {verified_dir.resolve()}\n")

    proofs = load_results(results_dir)
    agent  = IneqMathJudgeAgent()

    summary = []

    for proof_data in proofs:
        key  = proof_data["inequality_key"]
        name = proof_data["inequality_name"]

        print(f"\n{'='*60}")
        print(f"  Verifying: {name}")
        print(f"{'='*60}")

        try:
            result = agent(
                inequality_key=key,
                problem_statement=proof_data.get("problem_statement", ""),
                predicted_answer=proof_data.get("answer", ""),
                proof=proof_data.get("proof", ""),
            )
            print_judge_results(name, result)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            result = {"error": str(e), "overall_verdict": "ERROR"}

        # Save merged result
        save_data = {**proof_data, "verification": result}
        out_path  = verified_dir / f"{key}.json"
        out_path.write_text(
            json.dumps(save_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  💾 Saved → {out_path}")

        summary.append({
            "inequality":     name,
            "overall_verdict": result.get("overall_verdict", "ERROR"),
            "judges":          result.get("judges", {}),
        })

        time.sleep(1.5)

    # ── Print final summary table ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Inequality':<30} {'FA':<6} {'NTC':<6} {'NLG':<6} {'NAE':<6} {'NCE':<6} {'OVERALL'}")
    print(f"  {'─'*70}")
    for s in summary:
        j = s.get("judges", {})
        def v(key): return "✅" if j.get(key, {}).get("verdict") == "PASS" else "❌"
        icon = "✅" if s["overall_verdict"] == "PASS" else "❌"
        print(f"  {s['inequality']:<30} {v('final_answer'):<6} {v('ntc'):<6} "
              f"{v('nlg'):<6} {v('nae'):<6} {v('nce'):<6} {icon} {s['overall_verdict']}")

    print(f"\n✅ Done. Results in: {verified_dir.resolve()}")


if __name__ == "__main__":
    configure_dspy()
    run_all()