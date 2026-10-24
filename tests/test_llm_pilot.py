"""
Unit tests for the exploratory Modal LLM pilot helper functions and state abstraction.
"""

import pytest
from scripts.modal_llm_pilot import determine_public_state, VALID_ACTIONS, COUNTER_ACTIONS


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
