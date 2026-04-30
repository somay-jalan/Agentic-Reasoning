# run_all.py
import json
import time
from pathlib import Path
from generator import generate_proof
from config import GEMINI_MODEL
from prompts import ALL_PROMPTS

# ── Save under results/<model_name>/ ─────────────────────────────────────────
# Sanitize model name for use as a folder name
# e.g. "google/gemini-2.0-flash-001" → "google_gemini-2.0-flash-001"
MODEL_FOLDER  = GEMINI_MODEL.replace("/", "_")
RESULTS_DIR   = Path("results") / MODEL_FOLDER
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_all():
    print(f"\nModel  : {GEMINI_MODEL}")
    print(f"Saving : {RESULTS_DIR.resolve()}\n")

    for key, (name, prompt) in ALL_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"  Generating: {name}")
        print(f"{'='*60}")

        result   = generate_proof(key, name, prompt)
        out_path = RESULTS_DIR / f"{key}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Saved → {out_path}")

        time.sleep(1.5)

    print(f"\n✅ All results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    run_all()