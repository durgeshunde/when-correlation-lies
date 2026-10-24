"""
Unit tests for the exploratory Modal LLM pilot helper functions and state abstraction.
"""

import pytest
from scripts.modal_llm_pilot import determine_public_state, parse_action, VALID_ACTIONS, COUNTER_ACTIONS


def test_determine_public_state_start():
    assert determine_public_state(None) == "START"
    assert determine_public_state([]) == "START"


def test_determine_public_state_plurality():
    # Strict plurality
    assert determine_public_state(["ROCK", "ROCK", "PAPER", "SCISSORS"]) == "ROCK"
    assert determine_public_state(["PAPER", "PAPER", "PAPER", "ROCK"]) == "PAPER"
    assert determine_public_state(["SCISSORS", "SCISSORS", "ROCK", "PAPER"]) == "SCISSORS"


def test_determine_public_state_tie():
    # 2 vs 2 tie
    assert determine_public_state(["ROCK", "ROCK", "PAPER", "PAPER"]) == "TIE"
    assert determine_public_state(["ROCK", "ROCK", "SCISSORS", "SCISSORS"]) == "TIE"
    assert determine_public_state(["PAPER", "PAPER", "SCISSORS", "SCISSORS"]) == "TIE"


def test_counter_actions_validity():
    for act in VALID_ACTIONS:
        assert act in COUNTER_ACTIONS
        counter = COUNTER_ACTIONS[act]
        assert counter in VALID_ACTIONS
        assert counter != act


def test_parse_action_valid():
    assert parse_action('{"action": "ROCK"}') == "ROCK"
    assert parse_action('{"action": "PAPER"}') == "PAPER"
    assert parse_action('{"action": "SCISSORS"}') == "SCISSORS"
    assert parse_action('```json\n{"action": "ROCK"}\n```') == "ROCK"
    assert parse_action('{"action": "rock"}') == "ROCK"


def test_parse_action_fallback_and_invalid():
    assert parse_action("I choose ROCK for this round.") == "ROCK"
    # Never silently default to ROCK if invalid
    assert parse_action("I surrender and pass.") is None
    assert parse_action("") is None
    assert parse_action("invalid output") is None
    # Ambiguous choices should return None
    assert parse_action("Either ROCK or PAPER") is None
