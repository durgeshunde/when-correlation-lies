"""
trajectories.py
===============
Trajectory generation, logging, and loading.

A trajectory is a complete game run: a sequence of (state, actions, payoffs)
tuples for one experimental condition and seed.

JSONL format for raw trajectories
----------------------------------
Each line is a round record. The file begins with a header line.
Ground-truth labels are stored in a SEPARATE evaluation manifest.

Ground-truth separation
-----------------------
The evaluation manifest stores the coordinating pair identity.
The detector code NEVER reads from the evaluation manifest;
it only receives round records (public state + actions).
"""

from __future__ import annotations
import json
import hashlib
import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from .environment import RPSGame, N_AGENTS
from .agents import (
    make_condition_a_agents,
    make_condition_b_agents,
    make_condition_c_agents,
    make_condition_d_agents,
)

# Sentinel value for no coordinating pair (conditions A and B)
NO_PAIR = None
COORDINATED_PAIR = (0, 1)  # default; must match agent construction


def run_trajectory(
    condition: str,
    seed: int,
    n_rounds: int,
    alpha: float = 1.0,
    coordinated_pair: tuple[int, int] = COORDINATED_PAIR,
    noise: float = 0.1,
) -> dict:
    """Run one trajectory and return a structured record.
    
    Args:
        condition: One of 'A', 'B', 'C', 'D'.
        seed: Random seed for reproducibility.
        n_rounds: Number of rounds to play.
        alpha: Coordination strength (used only for condition D).
        coordinated_pair: The pair that coordinates (conditions C and D).
        noise: Policy noise level.
        
    Returns:
        Dict with keys:
            - 'meta': trajectory metadata.
            - 'rounds': list of round records (detector-visible).
            - 'ground_truth': dict with the true coordinating pair (MUST be
                             stored separately and not fed to detectors).
    """
    rng = np.random.default_rng(seed)
    game = RPSGame(n_rounds=n_rounds, rng=rng)
    
    if condition == 'A':
        agents = make_condition_a_agents(base_seed=seed)
        gt_pair = None
    elif condition == 'B':
        agents = make_condition_b_agents(base_seed=seed, noise=noise)
        gt_pair = None
    elif condition == 'C':
        agents = make_condition_c_agents(
            base_seed=seed,
            coordinated_pair=coordinated_pair,
            noise=noise,
        )
        gt_pair = list(coordinated_pair)
    elif condition == 'D':
        agents = make_condition_d_agents(
            base_seed=seed,
            alpha=alpha,
            coordinated_pair=coordinated_pair,
            noise=noise,
        )
        gt_pair = list(coordinated_pair)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
    
    history = game.run(agents)
    
    run_id = _make_run_id(condition, seed, n_rounds, alpha)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    meta = {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "n_rounds": n_rounds,
        "alpha": alpha if condition == 'D' else None,
        "noise": noise,
        "coordinated_pair_config": list(coordinated_pair) if condition in ('C', 'D') else None,
        "n_agents": N_AGENTS,
        "timestamp": timestamp,
        "game": "iterated_rps",
        "version": "1.0",
    }
    
    ground_truth = {
        "run_id": run_id,
        "condition": condition,
        "coordinating_pair": gt_pair,
        "note": "Ground-truth only. Must not be consumed by detectors.",
    }
    
    return {
        "meta": meta,
        "rounds": history,
        "ground_truth": ground_truth,
    }


def save_trajectory(traj: dict, data_dir: Path, overwrite: bool = False) -> tuple[Path, Path]:
    """Save trajectory and ground-truth manifest to separate files.
    
    Args:
        traj: Output from run_trajectory().
        data_dir: Root data directory. Creates subdirs automatically.
        overwrite: If False, raises if files exist.
        
    Returns:
        Tuple of (trajectory_path, ground_truth_path).
    """
    run_id = traj["meta"]["run_id"]
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = data_dir / "eval_labels"  # separate from detector inputs
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    traj_path = raw_dir / f"{run_id}.jsonl"
    gt_path = eval_dir / f"{run_id}_gt.json"
    
    if not overwrite and traj_path.exists():
        raise FileExistsError(f"Trajectory already exists: {traj_path}")
    
    # Write detector-visible trajectory (meta + rounds, NO ground truth)
    with open(traj_path, "w") as f:
        f.write(json.dumps(traj["meta"]) + "\n")
        for record in traj["rounds"]:
            # Convert state_feature tuple to list for JSON serialization
            r = dict(record)
            r["state_feature"] = list(r["state_feature"])
            f.write(json.dumps(r) + "\n")
    
    # Write ground-truth manifest separately
    with open(gt_path, "w") as f:
        json.dump(traj["ground_truth"], f, indent=2)
    
    return traj_path, gt_path


def load_trajectory(traj_path: Path) -> tuple[dict, list[dict]]:
    """Load trajectory from JSONL file (no ground truth).
    
    Returns:
        Tuple of (meta dict, list of round records).
    """
    with open(traj_path) as f:
        lines = f.readlines()
    
    meta = json.loads(lines[0])
    rounds = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            r = json.loads(line)
            # Restore state_feature as tuple for hashing
            r["state_feature"] = tuple(r["state_feature"])
            rounds.append(r)
    
    return meta, rounds


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground-truth manifest."""
    with open(gt_path) as f:
        return json.load(f)


def _make_run_id(condition: str, seed: int, n_rounds: int, alpha: float) -> str:
    """Create a stable unique run ID."""
    payload = f"cond={condition}_seed={seed}_rounds={n_rounds}_alpha={alpha:.4f}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"{condition}_{seed:04d}_{n_rounds}r_{h}"


def generate_dataset(
    conditions: list[str],
    seeds: list[int],
    n_rounds: int,
    data_dir: Path,
    alphas: Optional[list[float]] = None,
    coordinated_pair: tuple[int, int] = COORDINATED_PAIR,
    noise: float = 0.1,
    overwrite: bool = False,
) -> list[dict]:
    """Generate and save a full dataset of trajectories.
    
    Args:
        conditions: List of condition codes (subset of ['A', 'B', 'C', 'D']).
        seeds: List of random seeds for independent replications.
        n_rounds: Number of rounds per trajectory.
        data_dir: Root data directory.
        alphas: Alpha values for condition D. If None and D in conditions, uses [0.25, 0.5, 0.75, 1.0].
        coordinated_pair: Which pair coordinates (conditions C and D).
        noise: Policy noise.
        overwrite: Whether to overwrite existing files.
        
    Returns:
        List of dicts with run metadata and file paths.
    """
    if alphas is None:
        alphas = [0.25, 0.5, 0.75, 1.0]
    
    manifest = []
    
    for condition in conditions:
        if condition == 'D':
            for alpha in alphas:
                for seed in seeds:
                    traj = run_trajectory(
                        condition=condition,
                        seed=seed,
                        n_rounds=n_rounds,
                        alpha=alpha,
                        coordinated_pair=coordinated_pair,
                        noise=noise,
                    )
                    try:
                        traj_path, gt_path = save_trajectory(traj, data_dir, overwrite=overwrite)
                        status = "saved"
                    except FileExistsError:
                        traj_path = data_dir / "raw" / f"{traj['meta']['run_id']}.jsonl"
                        gt_path = data_dir / "eval_labels" / f"{traj['meta']['run_id']}_gt.json"
                        status = "existed"
                    
                    manifest.append({
                        "run_id": traj["meta"]["run_id"],
                        "condition": condition,
                        "seed": seed,
                        "n_rounds": n_rounds,
                        "alpha": alpha,
                        "traj_path": str(traj_path),
                        "gt_path": str(gt_path),
                        "status": status,
                    })
        else:
            for seed in seeds:
                traj = run_trajectory(
                    condition=condition,
                    seed=seed,
                    n_rounds=n_rounds,
                    coordinated_pair=coordinated_pair,
                    noise=noise,
                )
                try:
                    traj_path, gt_path = save_trajectory(traj, data_dir, overwrite=overwrite)
                    status = "saved"
                except FileExistsError:
                    traj_path = data_dir / "raw" / f"{traj['meta']['run_id']}.jsonl"
                    gt_path = data_dir / "eval_labels" / f"{traj['meta']['run_id']}_gt.json"
                    status = "existed"
                
                manifest.append({
                    "run_id": traj["meta"]["run_id"],
                    "condition": condition,
                    "seed": seed,
                    "n_rounds": n_rounds,
                    "alpha": None,
                    "traj_path": str(traj_path),
                    "gt_path": str(gt_path),
                    "status": status,
                })
    
    return manifest
