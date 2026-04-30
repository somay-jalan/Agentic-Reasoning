# generator.py
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT
from config import (OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    GEMINI_MODEL, MAX_TOKENS, TEMPERATURE)
load_dotenv()

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


def call_gemini(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        extra_headers={
            "HTTP-Referer": "https://ineqmath-agent.local",
            "X-Title":      "IneqMath Proof Agent",
        },
    )
    return response.choices[0].message.content


def parse_response(raw: str) -> dict:
    def extract(pattern, text):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return {
        "answer":   extract(r"##\s*PART\s*1[^\n]*\n(.*?)(?=##\s*PART\s*2|$)", raw),
        "proof":    extract(r"##\s*PART\s*2[^\n]*\n(.*?)(?=##\s*PART\s*3|$)", raw),
        "theorems": extract(r"##\s*PART\s*3[^\n]*\n(.*?)$", raw),
    }


def generate_proof(key: str, name: str, prompt: str) -> dict:
    raw    = call_gemini(prompt)
    parsed = parse_response(raw)

    result = {
        "inequality_key":  key,
        "inequality_name": name,
        "model":           GEMINI_MODEL,
        "prompt":          prompt,
        "raw_response":    raw,
        "answer":          parsed["answer"],
        "proof":           parsed["proof"],
        "theorems":        parsed["theorems"],
        "parse_success":   all(parsed.values()),
    }

    status = "✅" if result["parse_success"] else "⚠️  partial parse"
    print(f"  {status}  ({len(raw)} chars)")
    return result