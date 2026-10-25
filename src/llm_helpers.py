"""
src/llm_helpers.py
==================
Pure helper functions and state abstractions for LLM agent tournaments.
Independent of cloud SDKs (Modal) so that core parsing and state logic
can be tested and run in any environment with zero external dependencies.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

VALID_ACTIONS: List[str] = ["ROCK", "PAPER", "SCISSORS"]
COUNTER_ACTIONS = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}


def determine_public_state(previous_actions: Optional[List[str]]) -> str:
    """
    Computes the 5-state low-cardinality public summary for round t
    based strictly on plurality of previous actions.
    Returns: "START", "ROCK", "PAPER", "SCISSORS", or "TIE".
    """
    if not previous_actions:
        return "START"
    counts = {"ROCK": 0, "PAPER": 0, "SCISSORS": 0}
    for a in previous_actions:
        if a in counts:
            counts[a] += 1
    max_c = max(counts.values())
    top = [k for k, v in counts.items() if v == max_c]
    if len(top) == 1:
        return top[0]
    return "TIE"


def parse_action(raw_text: str) -> Optional[str]:
    """
    Robustly parses an action from model response.
    Never defaults to ROCK silently. Returns None if invalid or ambiguous.
    """
    if not raw_text:
        return None
    clean = raw_text.strip()
    if "```" in clean:
        parts = clean.split("```")
        clean = parts[1].replace("json", "").strip() if len(parts) > 1 else parts[0].strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            act = str(data.get("action", "")).upper().strip()
            if act in VALID_ACTIONS:
                return act
    except Exception:
        pass

    # Secondary regex search for {"action": "..."}
    match = re.search(r'["\']action["\']\s*:\s*["\'](ROCK|PAPER|SCISSORS)["\']', clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: exact single word action match (reject if ambiguous)
    words = re.findall(r"\b(ROCK|PAPER|SCISSORS)\b", clean.upper())
    if len(set(words)) == 1:
        return words[0]

    return None
