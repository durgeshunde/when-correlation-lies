"""
agents.py
=========
Agent implementations for the four conditions:

Condition A: Independent baseline
    Each agent independently samples uniformly (or from a fixed policy) 
    given the state. No shared random state, no communication.

Condition B: Shared-context negative control
    Agents independently use a deterministic or stochastic policy 
    conditioned on the shared public state. Actions may appear correlated 
    because both agents observe the same context — but they are independent
    conditional on the state.

Condition C: Known-coordination positive control
    A designated pair (e.g., agents 0 and 1) uses a coordination mechanism:
    they both deterministically respond to the same coordination signal 
    derived from the public state. This is a controlled intervention.
    The other agents remain independent.

Condition D: Coordination-strength sweep
    A parameter alpha in [0,1] controls the probability that the coordinating
    pair follows the coordination policy vs. falling back to independent play.
    alpha=1.0 is full coordination (Condition C); alpha=0.0 is independent.

All agents are STATELESS between rounds — they only use the current public 
state. This matches the paper's fully observable simultaneous game assumption.

No agent has any private memory, shared random generator, or hidden channel
UNLESS explicitly documented as part of the experimental condition.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from .environment import RPSGameState, ROCK, PAPER, SCISSORS, N_ACTIONS, ACTION_NAMES


class UniformRandomAgent:
    """Condition A: Uniformly random, stateless, no shared RNG with other agents.
    
    Each agent has its own independent RNG seeded separately.
    """
    
    def __init__(self, agent_id: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.rng = rng
    
    def __call__(self, state: RPSGameState, agent_id: int) -> int:
        return int(self.rng.integers(0, N_ACTIONS))
    
    def __repr__(self) -> str:
        return f"UniformRandomAgent(id={self.agent_id})"


class SharedContextAgent:
    """Condition B: Policy conditioned on shared public state.
    
    Strategy: respond to the most common action in previous round 
    (or uniform if tied/START), using a fixed lookup with independent noise.
    
    The key property: given the public state S, each agent independently 
    samples from P(A_i | S). Actions can be correlated unconditionally 
    because of the shared state, but are independent conditional on S.
    
    This is the *negative control* — it should expose the shared-context
    confounding problem in naive detectors.
    """
    
    def __init__(self, agent_id: int, rng: np.random.Generator, noise: float = 0.1):
        """
        Args:
            agent_id: This agent's ID.
            rng: Independent RNG for this agent.
            noise: Probability of choosing a random action instead of the 
                   state-conditioned policy. Must be independent of other agents.
        """
        self.agent_id = agent_id
        self.rng = rng
        self.noise = noise
    
    def _state_policy(self, state: RPSGameState) -> int:
        """Deterministic policy given state (before noise injection)."""
        if state.is_start:
            return ROCK  # fixed default for START
        
        # "Best response" heuristic: play the action that beats the most-played 
        # action of others in the previous round.
        prev = state.prev_actions
        counts = [prev.count(a) for a in range(N_ACTIONS)]
        dominant = int(np.argmax(counts))
        # Play the action that beats dominant: R->P, P->S, S->R
        beats = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
        return beats[dominant]
    
    def __call__(self, state: RPSGameState, agent_id: int) -> int:
        if self.rng.random() < self.noise:
            return int(self.rng.integers(0, N_ACTIONS))
        return self._state_policy(state)
    
    def __repr__(self) -> str:
        return f"SharedContextAgent(id={self.agent_id}, noise={self.noise})"


class CoordinatedAgent:
    """Condition C: One agent from the coordinating pair.
    
    Coordination mechanism: Both coordinated agents use the SAME deterministic
    function of the public state to choose their action. This is an explicit
    coordination signal through a shared computational rule — not a shared RNG.
    
    The coordination function is:
        coord_action = (hash(state_feature) % N_ACTIONS)
    
    This produces correlated actions between the two coordinated agents that
    CANNOT be explained by conditional independence given the state, because
    both agents are playing the SAME action rather than independently sampling
    from a state-conditioned distribution.
    
    Both coordinated agents play the same action in each round.
    The other two agents are SharedContextAgents (independent given state).
    
    Args:
        alpha: Probability of following the coordination rule.
               alpha=1.0 = full coordination (Condition C).
               alpha < 1.0 = partial coordination (Condition D).
    """
    
    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        noise: float = 0.1,
    ):
        self.agent_id = agent_id
        self.rng = rng
        self.alpha = alpha
        self.noise = noise  # fallback noise when not coordinating
    
    def _coordination_action(self, state: RPSGameState) -> int:
        """Deterministic coordination rule applied to the shared public state."""
        feature = state.to_feature()
        # Use a stable hash of the feature tuple
        h = hash(feature) & 0x7FFFFFFF  # ensure non-negative
        return int(h % N_ACTIONS)
    
    def __call__(self, state: RPSGameState, agent_id: int) -> int:
        if self.rng.random() < self.alpha:
            return self._coordination_action(state)
        # Fall back to independent noisy best-response
        if state.is_start:
            return int(self.rng.integers(0, N_ACTIONS))
        prev = state.prev_actions
        counts = [prev.count(a) for a in range(N_ACTIONS)]
        dominant = int(np.argmax(counts))
        beats = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
        policy_action = beats[dominant]
        if self.rng.random() < self.noise:
            return int(self.rng.integers(0, N_ACTIONS))
        return policy_action
    
    def __repr__(self) -> str:
        return f"CoordinatedAgent(id={self.agent_id}, alpha={self.alpha})"


def make_condition_a_agents(base_seed: int) -> list:
    """Condition A: Four independent uniform random agents.
    
    Each agent receives a DIFFERENT seed to ensure no shared random state.
    """
    return [
        UniformRandomAgent(
            agent_id=i,
            rng=np.random.default_rng([base_seed, i, 0xCAFE])
        )
        for i in range(4)
    ]


def make_condition_b_agents(base_seed: int, noise: float = 0.1) -> list:
    """Condition B: Four shared-context agents, independent conditional on state.
    
    Each agent has its own independent RNG.
    """
    return [
        SharedContextAgent(
            agent_id=i,
            rng=np.random.default_rng([base_seed, i, 0xBEEF]),
            noise=noise,
        )
        for i in range(4)
    ]


def make_condition_c_agents(
    base_seed: int,
    coordinated_pair: tuple[int, int] = (0, 1),
    noise: float = 0.1,
) -> list:
    """Condition C: Agents 0 & 1 coordinate; agents 2 & 3 are independent.
    
    Args:
        coordinated_pair: The (i, j) pair that uses the coordination rule.
        noise: Noise for the non-coordinating fallback (Condition C uses alpha=1.0).
    """
    agents = []
    for i in range(4):
        if i in coordinated_pair:
            agents.append(CoordinatedAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0xDEAD]),
                alpha=1.0,
                noise=noise,
            ))
        else:
            agents.append(SharedContextAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0xBEEF]),
                noise=noise,
            ))
    return agents


def make_condition_d_agents(
    base_seed: int,
    alpha: float,
    coordinated_pair: tuple[int, int] = (0, 1),
    noise: float = 0.1,
) -> list:
    """Condition D: Partial coordination sweep.
    
    Args:
        alpha: Coordination probability in [0, 1].
    """
    agents = []
    for i in range(4):
        if i in coordinated_pair:
            agents.append(CoordinatedAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0xDEAD]),
                alpha=alpha,
                noise=noise,
            ))
        else:
            agents.append(SharedContextAgent(
                agent_id=i,
                rng=np.random.default_rng([base_seed, i, 0xBEEF]),
                noise=noise,
            ))
    return agents
