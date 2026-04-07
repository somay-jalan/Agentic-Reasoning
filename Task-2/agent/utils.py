# agent/utils.py

import re


def strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles all of these cases:
`````python\n...\n```
````\n...\n```
```python...```   (no newline after opening fence)
        leading/trailing whitespace around fences
    """
    if not text:
        return text

    text = text.strip()

    # Remove opening fence: ```python or ``` with optional whitespace/newline
    text = re.sub(r'^```(?:python)?\s*\n?', '', text)

    # Remove closing fence: ``` at end with optional whitespace
    text = re.sub(r'\n?```\s*$', '', text)

    return text.strip()