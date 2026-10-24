"""
test_environment.py
==================
Unit tests for the RPS game environment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.environment import (
    RPSGame, RPSGameState, compute_payoffs,
    ROCK, PAPER, SCISSORS, N_AGENTS, PAYOFF_MATRIX
)


class TestPayoffs:
    """Test payoff calculation on hand-computed examples."""

    def test_rock_beats_scissors(self):
        """Rock vs Scissors: winner gets +3, loser gets -3 (vs 3 opponents)."""
        # All 4 agents: agent 0 plays Rock, all others play Scissors
        actions = [ROCK, SCISSORS, SCISSORS, SCISSORS]
        payoffs = compute_payoffs(actions)
        # Agent 0 (Rock) beats 3 Scissors: +3
        assert payoffs[0] == 3.0
        # Each Scissors loses to Rock: -1 (1 match), but beats no one here
        # Actually: Scissors loses to Rock (+1 for Rock = -1 for Scissors)
        for p in payoffs[1:]:
            assert p == -1.0

    def test_all_same_action_zero_payoffs(self):
        """All agents playing same action → all payoffs = 0."""
        for action in [ROCK, PAPER, SCISSORS]:
            actions = [action] * N_AGENTS
            payoffs = compute_payoffs(actions)
            assert all(p == 0.0 for p in payoffs), f"Expected zero payoffs, got {payoffs}"

    def test_paper_beats_rock(self):
        """Paper beats Rock."""
        # Agent 0 plays Paper, rest play Rock
        actions = [PAPER, ROCK, ROCK, ROCK]
        payoffs = compute_payoffs(actions)
        assert payoffs[0] == 3.0  # beats 3 rocks
        for p in payoffs[1:]:
            assert p == -1.0  # each rock loses to paper

    def test_scissors_beats_paper(self):
        """Scissors beats Paper."""
        actions = [SCISSORS, PAPER, PAPER, PAPER]
        payoffs = compute_payoffs(actions)
        assert payoffs[0] == 3.0
        for p in payoffs[1:]:
            assert p == -1.0

    def test_payoffs_sum_to_zero(self):
        """Total payoffs across agents must sum to zero (zero-sum game)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            actions = rng.integers(0, 3, 4).tolist()
            payoffs = compute_payoffs(actions)
            assert abs(sum(payoffs)) < 1e-9, f"Non-zero sum: {payoffs}"

    def test_invalid_action_raises(self):
        """Invalid action should raise AssertionError."""
        with pytest.raises(AssertionError):
            compute_payoffs([0, 1, 2, 3])  # 3 is invalid


class TestRPSGame:
    """Tests for the game orchestrator."""

    def test_correct_round_count(self):
        """Game runs exactly n_rounds rounds."""
        game = RPSGame(n_rounds=10)
        from src.agents import make_condition_a_agents
        agents = make_condition_a_agents(base_seed=0)
        history = game.run(agents)
        assert len(history) == 10

    def test_state_transitions(self):
        """State at round t+1 should be previous round's actions."""
        game = RPSGame(n_rounds=5)
        from src.agents import make_condition_a_agents
        agents = make_condition_a_agents(base_seed=1)
        history = game.run(agents)
        
        # Round 0 should have START state
        assert history[0]["state_feature"] == ("START",)
        
        # Round 1's state should be round 0's actions
        assert tuple(history[0]["actions"]) == history[1]["state_feature"]

    def test_reset_clears_history(self):
        """reset() should clear the game history."""
        game = RPSGame(n_rounds=5)
        from src.agents import make_condition_a_agents
        agents = make_condition_a_agents(base_seed=2)
        game.run(agents)
        game.reset()
        assert len(game.history) == 0

    def test_seed_reproducibility(self):
        """Same seed produces identical history."""
        from src.agents import make_condition_a_agents
        
        game1 = RPSGame(n_rounds=20)
        agents1 = make_condition_a_agents(base_seed=99)
        history1 = game1.run(agents1)
        
        game2 = RPSGame(n_rounds=20)
        agents2 = make_condition_a_agents(base_seed=99)
        history2 = game2.run(agents2)
        
        assert [r["actions"] for r in history1] == [r["actions"] for r in history2]

    def test_actions_are_valid(self):
        """All actions in game history must be in {0, 1, 2}."""
        game = RPSGame(n_rounds=50)
        from src.agents import make_condition_a_agents
        agents = make_condition_a_agents(base_seed=3)
        history = game.run(agents)
        
        for record in history:
            for action in record["actions"]:
                assert action in (ROCK, PAPER, SCISSORS), f"Invalid action {action}"

    def test_is_done_after_run(self):
        """Game should be done after running all rounds."""
        game = RPSGame(n_rounds=5)
        from src.agents import make_condition_a_agents
        agents = make_condition_a_agents(base_seed=4)
        game.run(agents)
        assert game.is_done

    def test_step_raises_when_done(self):
        """Step on a finished game should raise RuntimeError."""
        game = RPSGame(n_rounds=1)
        game.step([ROCK, ROCK, ROCK, ROCK])
        with pytest.raises(RuntimeError):
            game.step([ROCK, ROCK, ROCK, ROCK])


class TestRPSGameState:
    """Tests for the state object."""

    def test_start_state_hashable(self):
        """State feature must be hashable for use as dict key."""
        state = RPSGameState(round_idx=0)
        d = {state.to_feature(): "value"}  # should not raise

    def test_non_start_state_hashable(self):
        """Non-start state with action tuple must be hashable."""
        state = RPSGameState(round_idx=1, prev_actions=(0, 1, 2, 0))
        d = {state.to_feature(): "value"}  # should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
