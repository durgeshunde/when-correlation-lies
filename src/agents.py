"""
src/agents.py
=============
Agent implementations for the four experimental conditions:

Condition A: Independent baseline
    Each agent independently samples uniformly from {Rock, Paper, Scissors}.
    Each agent has an isolated RNG seeded separately.
    No shared state, no coordination.

Condition B: Shared-context negative control
    Agents independently play a state-conditioned best-response policy to the
    most common action in the observable public state S, with independent noise.
    Actions are conditionally independent given S, but unconditionally correlated
    due to shared public history.

Condition C: Valid known-coordination positive control
    A designated pair (agents 0 and 1) shares a private coordination signal Z_t in {0, 1, 2}
    that is unobservable to non-coordinating agents and hidden from the detector/auditor.
    When coordinating (alpha = 1.0), both agents play Z_t.
    Non-coordinating agents (2 and 3) play independent state-conditioned best responses.
    True population CMI I(A_0; A_1 | S) > 0; true CMI for other pairs = 0.

Condition D: Coordination-strength sweep
    Pair (0, 1) follows the private coordination signal Z_t with probability alpha in [0, 1],
    and falls back to independent best response with probability 1 - alpha.
    Enables empirical tracing of the detector's sensitivity boundary.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable
from .environment import RPSGameState, ROCK, PAPER, SCISSORS, N_ACTIONS


class UniformRandomAgent:
    """Condition A: Uniformly random action selection with isolated RNG."""

    def __init__(self, agent_id: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.rng = rng

    def __call__(self, state: RPSGameState, private_signal: Optional[int] = None) -> int:
        return int(self.rng.integers(0, N_ACTIONS))

    def __repr__(self) -> str:
        return f"UniformRandomAgent(id={self.agent_id})"


class SharedContextAgent:
    """Condition B (and non-coordinating background agent in C/D).
    
    Given public state S, independently chooses best response to the dominant prior action
    with independent noise epsilon.
    Conditionally independent given S: P(A_i, A_j | S) = P(A_i | S) P(A_j | S).
    """

    def __init__(self, agent_id: int, rng: np.random.Generator, noise: float = 0.10):
        self.agent_id = agent_id
        self.rng = rng
        self.noise = noise

    def _state_policy(self, state: RPSGameState) -> int:
        if state.is_start:
            return ROCK
        prev = state.prev_actions
        counts = [prev.count(a) for a in range(N_ACTIONS)]
        dominant = int(np.argmax(counts))
        # Best response: R -> P (1), P -> S (2), S -> R (0)
        beats = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
        return beats[dominant]

    def __call__(self, state: RPSGameState, private_signal: Optional[int] = None) -> int:
        if self.rng.random() < self.noise:
            return int(self.rng.integers(0, N_ACTIONS))
        return self._state_policy(state)

    def __repr__(self) -> str:
        return f"SharedContextAgent(id={self.agent_id}, noise={self.noise})"


class CoordinatedAgent:
    """Conditions C and D: Member of the coordinating pair.
    
    Observes public state S. When the round's joint coordination gate G_t is active,
    receives the shared private coordination signal Z_t in {0, 1, 2} and plays it.
    When the gate is inactive (private_signal is None), falls back to an independent
    state-conditioned best response with noise.
    
    This guarantees that the pair coordinates jointly with probability alpha,
    rather than independently with probability alpha^2.
    """

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        noise: float = 0.10,
    ):
        self.agent_id = agent_id
        self.rng = rng
        self.alpha = alpha
        self.noise = noise

    def _state_policy(self, state: RPSGameState) -> int:
        if state.is_start:
            return ROCK
        prev = state.prev_actions
        counts = [prev.count(a) for a in range(N_ACTIONS)]
        dominant = int(np.argmax(counts))
        beats = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
        return beats[dominant]

    def __call__(self, state: RPSGameState, private_signal: Optional[int] = None) -> int:
        # If the joint coordination gate G_t was active, follow private signal Z_t
        if private_signal is not None:
            return int(private_signal)
        
        # Fallback: independent best response with noise
        if self.rng.random() < self.noise:
            return int(self.rng.integers(0, N_ACTIONS))
        return self._state_policy(state)

    def __repr__(self) -> str:
        return f"CoordinatedAgent(id={self.agent_id}, alpha={self.alpha})"


def make_condition_a_agents(base_seed: int) -> list:
    """Condition A: 4 independent uniform random agents with isolated seeds."""
    return [
        UniformRandomAgent(
            agent_id=i,
            rng=np.random.default_rng([base_seed, i, 0x1111])
        )
        for i in range(4)
    ]


def make_condition_b_agents(base_seed: int, noise: float = 0.10) -> list:
    """Condition B: 4 shared-context agents, conditionally independent given S."""
    return [
        SharedContextAgent(
            agent_id=i,
            rng=np.random.default_rng([base_seed, i, 0x2222]),
            noise=noise,
        )
        for i in range(4)
    ]


def make_condition_c_agents(
    base_seed: int,
    coordinated_pair: tuple[int, int] = (0, 1),
    noise: float = 0.10,
) -> list:
    """Condition C: Agents (0, 1) coordinate on private signal (alpha = 1.0); others independent."""
    agents = []
    for i in range(4):
        if i in coordinated_pair:
            agents.append(CoordinatedAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0x3333]),
                alpha=1.0,
                noise=noise,
            ))
        else:
            agents.append(SharedContextAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0x4444]),
                noise=noise,
            ))
    return agents


def make_condition_d_agents(
    base_seed: int,
    alpha: float,
    coordinated_pair: tuple[int, int] = (0, 1),
    noise: float = 0.10,
) -> list:
    """Condition D: Partial coordination sweep with probability alpha in [0, 1]."""
    agents = []
    for i in range(4):
        if i in coordinated_pair:
            agents.append(CoordinatedAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0x5555]),
                alpha=alpha,
                noise=noise,
            ))
        else:
            agents.append(SharedContextAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0x6666]),
                noise=noise,
            ))
    return agents
