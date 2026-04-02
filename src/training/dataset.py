from __future__ import annotations

from typing import Any

import numpy as np

# (state, policy, value)
TrajectoryStep = tuple[Any, list[float], float]
Trajectory = list[TrajectoryStep]
Dataset = dict[str, np.ndarray]


def build_dataset(trajectories: list[Trajectory]) -> Dataset:
    """Aggregate trajectories from one or more workers into a numpy dataset."""
    states = []
    policies = []
    values = []

    for trajectory in trajectories:
        for state, policy, value in trajectory:
            states.append(state)
            policies.append(policy)
            values.append(value)

    return {
        "states": np.array(states),
        "policies": np.array(policies, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
    }


def save_dataset(dataset: Dataset, path: str) -> None:
    """Save dataset to a single .npz file at *path*."""
    np.savez(path, **dataset)
