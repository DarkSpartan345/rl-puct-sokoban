from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  — registers MiniGrid envs


class MiniGridWrapper:
    """Minimal wrapper around MiniGrid for the PUCT-MCTS framework.

    State: flattened 1-D float32 vector (image 7×7×3 + direction).
    All actions in the discrete action space are always considered valid.
    """

    def __init__(self, env_id: str = "MiniGrid-Empty-5x5-v0") -> None:
        self._env = gym.make(env_id)
        self._n_actions: int = self._env.action_space.n

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return self._flatten(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        obs, reward, terminated, truncated, _ = self._env.step(action)
        return self._flatten(obs), float(reward), terminated, truncated

    def get_actions(self, state: Any) -> list[int]:  # noqa: ARG002
        return list(range(self._n_actions))

    @staticmethod
    def _flatten(obs: dict) -> np.ndarray:
        image = obs["image"].astype(np.float32).flatten()
        direction = np.array([obs["direction"]], dtype=np.float32)
        return np.concatenate([image, direction])
