# agent/debugger.py
"""
Debug Agent — rewrites code given a structured critique and tool report.
"""

import json
import re
from openai import OpenAI

DEBUGGER_SYSTEM = """\
You are an expert Python physics debugger. You receive:
  1. The sub-task specification (what the code must do)
  2. The current (broken) code
  3. A structured critique from a physics reviewer
  4. A full tool report (syntax errors, AST pattern warnings,
     physics sanity check results, pytest failures)

Your job: output a corrected version that addresses EVERY item in
'actionable_fixes' and fixes all errors in the tool report.

Rules:
- Output ONLY a single Python code block (```python ... ```)
- Fix physics errors before numerical errors before style issues
- Preserve function and class signatures the sub-task requires
- Do not remove existing correct logic — only fix what is broken
- If a pytest failure shows an expected value, make your code
  produce exactly that value
- Never leave TODO comments
"""

DEBUGGER_USER = """\
Original task prompt (contains the file you must work with):
────────────────────────────────────────────────────────────
{original_prompt}
────────────────────────────────────────────────────────────

Sub-task specification:
Title       : {title}
Description : {description}
Expected    : {expected_output}

Current (broken) code:
```python
{code}
```

Structured critique:
{critique_json}

Tool report:
{tool_report_json}

Fix the code above. The original file is in the task prompt.
Do not invent functions that don't exist in the original.
Output the COMPLETE corrected file — one Python code block.
"""


class DebugAgent:
    def __init__(self, cfg, client: OpenAI):
        self.cfg    = cfg
        self.client = client

    
    def fix(
        self,
        subtask:         dict,
        code:            str,
        critique:        dict,
        tool_report:     dict,
        original_prompt: str = "",    # ← ADD THIS
        retries:         int = 3,
    ) -> str:
        """
        Returns corrected code, or the original if all retries fail.
        """
        user_msg = DEBUGGER_USER.format(
            original_prompt  = original_prompt,
            title            = subtask["title"],
            description      = subtask["description"],
            expected_output  = subtask["expected_output"],
            code             = code,
            critique_json    = json.dumps(critique,     indent=2),
            tool_report_json = json.dumps(tool_report,  indent=2),
        )

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.cfg.model_id,
                    messages = [
                        {"role": "system", "content": DEBUGGER_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature = 0.4,      # slightly creative but grounded
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
                raw = resp.choices[0].message.content or ""
                return self._extract_code(raw)

            except Exception as exc:
                print(f"  [Debugger] Attempt {attempt+1} failed: {exc}")

        # Return original so the pipeline doesn't lose all progress
        return code

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
