"""JSON extraction and repair for LLM structured output."""
import json
import re
from typing import Any, Optional


def extract_json_block(text: str) -> Optional[Any]:
    """Extract JSON from markdown code fence or raw braces."""
    if not text:
        return None

    # Try markdown fence first
    match = re.search(r'```json\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if match:
        result = repair_json(match.group(1))
        if result is not None:
            return result

    # Try raw JSON (find outermost braces/brackets)
    # Sort by which delimiter appears first in the text
    delimiters = [('{', '}'), ('[', ']')]
    delimiters.sort(key=lambda d: text.find(d[0]) if text.find(d[0]) != -1 else float('inf'))
    for start_char, end_char in delimiters:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    result = repair_json(text[start:i+1])
                    if result is not None:
                        return result
                    break

    return None


def repair_json(text: str) -> Optional[Any]:
    """Attempt to parse JSON with common LLM error repairs."""
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix single quotes
    fixed2 = fixed.replace("'", '"')
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    return None
