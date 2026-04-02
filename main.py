"""Entry point: run self-play episodes and save dataset."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.envs import MiniGridWrapper
from src.agents import PUCTAgent
from src.selfplay import run_episode, run_parallel
from src.training import build_dataset, save_dataset

# --- Config ---
ENV_ID = "MiniGrid-Empty-5x5-v0"
N_SIMULATIONS = 20
C = 1.0
EPISODES = 4
WORKERS = 2
OUTPUT_PATH = "dataset.npz"


def env_fn() -> MiniGridWrapper:
    return MiniGridWrapper(env_id=ENV_ID)


def agent_fn() -> PUCTAgent:
    return PUCTAgent(env_fn(), n_simulations=N_SIMULATIONS, c=C)


def main() -> None:
    print(f"Env: {ENV_ID}")
    print(f"Episodes: {EPISODES} | Workers: {WORKERS} | Simulations/step: {N_SIMULATIONS}")

    if WORKERS > 1:
        trajectories = run_parallel(env_fn, agent_fn, episodes=EPISODES, workers=WORKERS)
    else:
        env = env_fn()
        agent = agent_fn()
        trajectories = [run_episode(env, agent) for _ in range(EPISODES)]

    total_steps = sum(len(t) for t in trajectories)
    print(f"Collected {len(trajectories)} trajectories, {total_steps} total steps.")

    dataset = build_dataset(trajectories)
    print(f"Dataset — states: {dataset['states'].shape}, "
          f"policies: {dataset['policies'].shape}, "
          f"values: {dataset['values'].shape}")

    save_dataset(dataset, OUTPUT_PATH)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
