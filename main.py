"""Entry point: run self-play episodes and save dataset."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.envs import SokobanWrapper
from src.agents import PUCTAgent
from src.selfplay import run_episode, run_parallel
from src.training import build_dataset, save_dataset

# --- Config ---
N_SIMULATIONS = 20
C = 1.0
EPISODES = 4
WORKERS = 2
OUTPUT_PATH = "dataset.npz"


def env_fn() -> SokobanWrapper:
    return SokobanWrapper(dim_room=(5, 5), max_steps=50, num_boxes=1)


def agent_fn() -> PUCTAgent:
    return PUCTAgent(env_fn(), n_simulations=N_SIMULATIONS, c=C)


def main() -> None:
    print(f"Running {EPISODES} episodes with {WORKERS} workers ...")

    if WORKERS > 1:
        trajectories = run_parallel(env_fn, agent_fn, episodes=EPISODES, workers=WORKERS)
    else:
        env = env_fn()
        agent = agent_fn()
        trajectories = [run_episode(env, agent) for _ in range(EPISODES)]

    total_steps = sum(len(t) for t in trajectories)
    print(f"Collected {len(trajectories)} trajectories, {total_steps} total steps.")

    dataset = build_dataset(trajectories)
    print(f"Dataset shapes — states: {dataset['states'].shape}, "
          f"policies: {dataset['policies'].shape}, "
          f"values: {dataset['values'].shape}")

    save_dataset(dataset, OUTPUT_PATH)
    print(f"Dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
