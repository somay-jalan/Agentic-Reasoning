# agent/coder.py
"""
Coder Agent — implements one TCRG sub-task given context code from
dependency nodes.
"""

import re
from openai import OpenAI

CODER_SYSTEM = """\
You are an expert Python physics developer implementing one sub-task from
a larger physics simulation pipeline.

Rules:
- Output ONLY a single Python code block (```python ... ```)
- Do NOT repeat code that is already in the dependency context — only add
  new functions, classes, or module-level constants.
- Use numpy / scipy / standard scientific Python as needed.
- Include docstrings that state units for every physical quantity.
- Use physically meaningful variable names (e.g. 'omega' not 'w',
  'hbar' not 'h', 'epsilon_0' not 'e0').
- Handle numerical edge cases: zero division, singular matrices,
  near-zero denominators.
- Never leave a TODO comment — implement everything fully.
"""

CODER_USER_NO_CONTEXT = """\
Original task prompt (contains the file you must work with):
────────────────────────────────────────────────────────────
{original_prompt}
────────────────────────────────────────────────────────────

Sub-task to implement (first node — no prior code):
Title       : {title}
Description : {description}
Expected    : {expected_output}

The original file is in the task prompt above.
Work from that file. Do not invent a different file structure.
Output one Python code block containing the COMPLETE file.
"""

CODER_USER_WITH_CONTEXT = """\
Original task prompt (contains the file you must work with):
────────────────────────────────────────────────────────────
{original_prompt}
────────────────────────────────────────────────────────────

Code already written by dependency sub-tasks (extend this):
```python
{context_code}
```

Sub-task to implement:
Title       : {title}
Description : {description}
Expected    : {expected_output}

Output one Python code block containing the COMPLETE file.
"""


class CoderAgent:
    def __init__(self, cfg, client: OpenAI):
        self.cfg    = cfg
        self.client = client

    def code(
        self,
        subtask:         dict,
        context_code:    str = "",
        original_prompt: str = "",    # ← ADD THIS
        retries:         int = 3,
    ) -> str:
        """
        Generate code for one sub-task.

        Parameters
        ----------
        subtask      : TCRG node dict (id, title, description, expected_output)
        context_code : concatenated code from all dependency nodes
        """
        if context_code.strip():
            user_msg = CODER_USER_WITH_CONTEXT.format(
                original_prompt = original_prompt,
                title           = subtask["title"],
                description     = subtask["description"],
                expected_output = subtask["expected_output"],
                context_code    = context_code,
            )
        else:
            user_msg = CODER_USER_NO_CONTEXT.format(
                original_prompt = original_prompt,
                title           = subtask["title"],
                description     = subtask["description"],
                expected_output = subtask["expected_output"],
            )

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.cfg.model_id,
                    messages = [
                        {"role": "system", "content": CODER_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature = 0.8,
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
                print(f"  [Coder] Attempt {attempt+1} failed: {exc}")

        return ""

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
