"""
environment.py
==============
Four-agent iterated Rock-Paper-Scissors (RPS) game environment.

State representation
--------------------
In a fully observable simultaneous game (as required by Bonjour et al. 2022),
the public state S at round t is the joint action profile from round t-1:
    S_t = (a1_{t-1}, a2_{t-1}, a3_{t-1}, a4_{t-1})

For round 0, the state is a sentinel 'START' token.

Actions
-------
0 = Rock, 1 = Paper, 2 = Scissors

Payoffs
-------
Standard zero-sum RPS payoffs for each pairwise matchup.
In a four-agent game, agent i's payoff = sum of pairwise outcomes vs all others.

    win  -> +1
    draw ->  0
    loss -> -1
"""

from __future__ import annotations
from typing import Optional
import numpy as np

# Action constants
ROCK = 0
PAPER = 1
SCISSORS = 2
ACTION_NAMES = {0: "R", 1: "P", 2: "S"}
N_ACTIONS = 3
N_AGENTS = 4

# Payoff matrix: PAYOFF[a][b] = payoff for agent playing action a against agent playing b
# Rows = my action, Cols = opponent action
PAYOFF_MATRIX = np.array([
    # R   P   S
    [ 0, -1,  1],  # R
    [ 1,  0, -1],  # P
    [-1,  1,  0],  # S
], dtype=float)


def compute_payoffs(actions: list[int]) -> list[float]:
    """Compute payoffs for all agents given a joint action profile.
    
    Args:
        actions: List of N_AGENTS integers in {ROCK, PAPER, SCISSORS}.
        
    Returns:
        List of floats, one per agent.
    """
    assert len(actions) == N_AGENTS, f"Expected {N_AGENTS} actions, got {len(actions)}"
    for a in actions:
        assert a in (ROCK, PAPER, SCISSORS), f"Invalid action: {a}"
    
    payoffs = []
    for i in range(N_AGENTS):
        total = 0.0
        for j in range(N_AGENTS):
            if i != j:
                total += PAYOFF_MATRIX[actions[i], actions[j]]
        payoffs.append(total)
    return payoffs


class RPSGameState:
    """Immutable snapshot of public game state."""
    
    START_TOKEN = "START"
    
    def __init__(self, round_idx: int, prev_actions: Optional[tuple[int, ...]] = None):
        """
        Args:
            round_idx: Current round index (0-based).
            prev_actions: Joint action profile from previous round (None for round 0).
        """
        self.round_idx = round_idx
        self.prev_actions = prev_actions  # None = START
    
    @property
    def is_start(self) -> bool:
        return self.prev_actions is None
    
    def to_feature(self) -> tuple:
        """Return a hashable feature for the state, suitable as a dict key."""
        if self.is_start:
            return (self.START_TOKEN,)
        return tuple(self.prev_actions)
    
    def __repr__(self) -> str:
        if self.is_start:
            return f"RPSGameState(round={self.round_idx}, state=START)"
        names = [ACTION_NAMES[a] for a in self.prev_actions]
        return f"RPSGameState(round={self.round_idx}, prev={''.join(names)})"


class RPSGame:
    """Orchestrates a multi-round RPS game with four agents.
    
    This class:
    - Does NOT contain agent logic (agents are separate).
    - Tracks round history.
    - Exposes public state to agents uniformly.
    - Computes and records payoffs.
    """
    
    def __init__(self, n_rounds: int, rng: Optional[np.random.Generator] = None):
        self.n_rounds = n_rounds
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()
    
    def reset(self):
        self.round_idx = 0
        self.history: list[dict] = []
        self._current_state = RPSGameState(round_idx=0, prev_actions=None)
    
    @property
    def current_state(self) -> RPSGameState:
        return self._current_state
    
    @property
    def is_done(self) -> bool:
        return self.round_idx >= self.n_rounds
    
    def step(self, actions: list[int]) -> dict:
        """Execute one round given joint agent actions.
        
        Args:
            actions: List of N_AGENTS integers.
            
        Returns:
            Round record dict (state, actions, payoffs, round_idx).
        """
        if self.is_done:
            raise RuntimeError("Game is done; call reset() to start a new game.")
        
        state_feature = self._current_state.to_feature()
        payoffs = compute_payoffs(actions)
        
        record = {
            "round_idx": self.round_idx,
            "state_feature": state_feature,
            "actions": list(actions),
            "payoffs": payoffs,
        }
        self.history.append(record)
        
        # Advance state
        self.round_idx += 1
        if self.round_idx < self.n_rounds:
            self._current_state = RPSGameState(
                round_idx=self.round_idx,
                prev_actions=tuple(actions)
            )
        
        return record
    
    def run(self, agent_fns: list) -> list[dict]:
        """Run the full game using a list of agent callable objects.
        
        Args:
            agent_fns: List of N_AGENTS callables, each accepting
                       (state: RPSGameState, agent_id: int) -> int.
                       
        Returns:
            List of round records.
        """
        assert len(agent_fns) == N_AGENTS
        self.reset()
        
        while not self.is_done:
            state = self.current_state
            actions = [
                agent_fn(state, agent_id=i)
                for i, agent_fn in enumerate(agent_fns)
            ]
            self.step(actions)
        
        return self.history
