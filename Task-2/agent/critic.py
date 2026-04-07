# agent/critic.py
"""
Critic Agent — reviews generated code and produces a structured critique
that the Debug Agent can act on.
"""

import json
import re
from openai import OpenAI

CRITIC_SYSTEM = """\
You are a physics code reviewer. You will receive:
  1. A sub-task specification
  2. Generated Python code
  3. A structured tool report (syntax errors, AST pattern issues,
     physics sanity check results, pytest failures)

Produce a concise structured critique that a debugger can act on directly.

Output ONLY valid JSON in exactly this format — no markdown, no extra text:
{
  "physical_errors":  ["list of physics-correctness problems"],
  "numerical_errors": ["list of numerical / algorithmic problems"],
  "missing_cases":    ["edge cases or validations that are absent"],
  "severity":         "low | medium | high",
  "actionable_fixes": ["concrete, specific things the debugger must change"]
}

Severity guide:
  high   — syntax errors, wrong physics equations, pytest failures
  medium — numerical instability, missing edge cases, wrong units
  low    — style issues, missing docstrings, suboptimal variable names
"""

CRITIC_USER = """\
Sub-task specification
──────────────────────
Title       : {title}
Description : {description}
Expected    : {expected_output}

Generated code:
```python
{code}
```

Tool report:
{tool_report_json}

Provide your structured critique as JSON.
"""


class CriticAgent:
    def __init__(self, cfg, client: OpenAI):
        self.cfg    = cfg
        self.client = client

    def review(
        self,
        subtask:     dict,
        code:        str,
        tool_report: dict,
        retries:     int = 3,
    ) -> dict:
        """
        Returns a critique dict with keys:
            physical_errors, numerical_errors, missing_cases,
            severity, actionable_fixes
        """
        user_msg = CRITIC_USER.format(
            title           = subtask["title"],
            description     = subtask["description"],
            expected_output = subtask["expected_output"],
            code            = code,
            tool_report_json= json.dumps(tool_report, indent=2),
        )

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.cfg.model_id,
                    messages = [
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature = 0.2,
                    max_tokens  = 65536,
                    extra_headers = {
                        "HTTP-Referer": "https://github.com/your-repo",
                        "X-Title":      "PhysicsBenchmark-Agent",
                    },
                    extra_body={
                        "provider": {
                        "sort": "throughput"
                        }
                    },
                )
                raw      = resp.choices[0].message.content or ""
                critique = self._parse_critique(raw)
                if critique:
                    return critique
                print(f"  [Critic] Attempt {attempt+1}: JSON parse failed — retrying.")

            except Exception as exc:
                print(f"  [Critic] Attempt {attempt+1} API error: {exc}")

        # Safe fallback so the debug loop can still proceed
        return {
            "physical_errors":  [],
            "numerical_errors": [],
            "missing_cases":    [],
            "severity":         "unknown",
            "actionable_fixes": [
                "Review all tool report errors and fix them systematically."
            ],
        }

    def _parse_critique(self, raw: str) -> dict:
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        
        # Try direct parse first
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > 0:
            try:
                data = json.loads(raw[start:end])
                if "severity" in data and "actionable_fixes" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # GLM5 fallback: extract fields with regex
        severity_match = re.search(
            r'"?severity"?\s*:\s*"?(low|medium|high|unknown)"?', raw, re.IGNORECASE
        )
        fixes_match = re.search(
            r'"?actionable_fixes"?\s*:\s*\[([^\]]*)\]', raw, re.DOTALL
        )
        
        if severity_match:
            severity = severity_match.group(1).lower()
            fixes    = []
            if fixes_match:
                # Extract quoted strings from the list
                fixes = re.findall(r'"([^"]+)"', fixes_match.group(1))
            
            return {
                "physical_errors":  [],
                "numerical_errors": [],
                "missing_cases":    [],
                "severity":         severity,
                "actionable_fixes": fixes or ["Fix all errors shown in the tool report."],
            }

        return {}  # genuine failure — caller uses fallback


