# agent/planner.py
"""
Planner Agent — decomposes a physics programming prompt into a
Task-Conditioned Reasoning Graph (TCRG): a DAG of ordered sub-tasks.
"""

import json
import re
from openai import OpenAI

PLANNER_SYSTEM = """\
You are a physics software architect. Your job is to decompose a physics
programming task into an ordered sequence of sub-tasks represented as a
directed acyclic graph (TCRG).

════════════════════════════════════════════════════════════════
STEP 1 — CLASSIFY THE TASK TYPE
════════════════════════════════════════════════════════════════
Read the prompt carefully and classify it as one of:

  "bugfix"  — prompt contains any of:
                "issue", "error", "fix", "bug", "reproduce",
                "incorrect", "wrong", "fails", "exception",
                "traceback", "expected result", "actual result"

  "feature" — prompt contains any of:
                "feature", "implement", "add", "create",
                "support", "introduce", "extend", "new"

If both signals appear, prefer "bugfix".
Store this as the "task_type" field in your output.

════════════════════════════════════════════════════════════════
STEP 2 — DECOMPOSE ACCORDING TO TASK TYPE
════════════════════════════════════════════════════════════════

── IF task_type = "bugfix" ─────────────────────────────────────

Produce EXACTLY 1 node. 2 nodes only if the fix spans two
clearly separate functions that cannot be changed together.
NEVER produce more than 2 nodes for a bugfix.

Node rules for bugfix:
  - The description MUST name the exact function(s) to modify.
  - The description MUST quote or paraphrase the error / wrong
    behaviour directly from the prompt.
  - The description MUST state the correct expected behaviour.
  - The expected_output MUST say: "The COMPLETE modified Python
    file with the fix applied. Not a fragment. Not just the
    fixed function. The entire file from the first import to
    the last line."
  - NEVER create separate nodes for: imports, validation,
    assembly, test cases, or helper utilities. All of that
    belongs inside the single fix node.

── IF task_type = "feature" ────────────────────────────────────

Produce between 3 and 5 nodes covering:
  (1) Physical constants, data structures, type definitions
  (2) Core algorithm / numerical kernel
  (3) Integration / orchestration layer
  (4) Validation against a known analytical limit or test case
  (5) Final assembly with if __name__ == '__main__': block
      (only if the feature is a standalone script)

Node rules for feature:
  - Keep nodes cohesive: one physical concept or component
    per node.
  - Dependencies must form a DAG — no cycles.
  - Each node's expected_output must name concrete
    function/class signatures that downstream nodes can rely on.

════════════════════════════════════════════════════════════════
STEP 3 — OUTPUT FORMAT
════════════════════════════════════════════════════════════════
Output ONLY valid JSON — no markdown fences, no explanation,
no text before or after the JSON object:

{
  "task_type": "bugfix | feature",
  "nodes": [
    {
      "id": "n1",
      "title": "Short descriptive title",
      "description": "Detailed description of exactly what this
                      node must produce. For bugfix: name the
                      function, quote the error, state the fix.
                      For feature: describe the component.",
      "dependencies": [],
      "expected_output": "Concrete description of the code this
                          node must return."
    }
  ]
}

"""

PLANNER_USER = """\
Task prompt:
{prompt}

If this is a bugfix task, each node description MUST specify:
1. The exact function(s) to modify (by name)
2. What the incorrect behavior is
3. What the correct behavior should be
4. That the node must return the COMPLETE modified file, not a fragment

Decompose into a TCRG. Output only the JSON object.
"""


class PlannerAgent:
    def __init__(self, cfg, client: OpenAI):
        self.cfg    = cfg
        self.client = client

    def plan(self, prompt: str, retries: int = 3) -> dict:
        """
        Returns a TCRG dict: {"nodes": [...]}
        Falls back to a single-node graph if all retries fail.
        """
        user_msg = PLANNER_USER.format(prompt=prompt)

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.cfg.model_id,
                    messages = [
                        {"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature = 0.2,      # low temp: planning needs consistency
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
                raw  = resp.choices[0].message.content or ""
                tcrg = self._parse_tcrg(raw)
                if tcrg:
                    return tcrg
                print(f"  [Planner] Attempt {attempt+1}: JSON parse failed — retrying.")

            except Exception as exc:
                print(f"  [Planner] Attempt {attempt+1} API error: {exc}")

        print("  [Planner] All retries failed — falling back to single-node graph.")
        return self._fallback_graph(prompt)

    # ── Parsing ────────────────────────────────────────────────────

    def _parse_tcrg(self, raw: str) -> dict:
        """Extract and validate the TCRG JSON from an LLM response."""
        # Strip markdown fences if the model added them
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}

        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return {}

        nodes = data.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            return {}

        required_keys = {"id", "title", "description", "dependencies", "expected_output"}
        for node in nodes:
            if not required_keys.issubset(node.keys()):
                return {}

        return data

    def _fallback_graph(self, prompt: str) -> dict:
        """Single-node fallback: degrades to near-zero-shot behaviour."""
        return {
            "nodes": [{
                "id":             "n1",
                "title":          "Full implementation",
                "description":    prompt,
                "dependencies":   [],
                "expected_output": "Complete, working physics Python implementation.",
            }]
        }
