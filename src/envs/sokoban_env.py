from __future__ import annotations

import warnings
from typing import Any

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gym_sokoban  # noqa: F401
    from gym_sokoban.envs import SokobanEnv


class SokobanWrapper:
    """Minimal wrapper around gym-sokoban's SokobanEnv.

    Adapts the old gym API (obs, reward, done, info) to the interface
    expected by the framework: reset() → state, step() → (state, reward, terminated, truncated).
    """

    def __init__(self, dim_room: tuple[int, int] = (5, 5), max_steps: int = 100, num_boxes: int = 1) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._env = SokobanEnv(dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes)
        self._n_actions: int = self._env.action_space.n

    def reset(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = self._env.reset()
        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        state, reward, done, _info = self._env.step(action)
        return state, float(reward), bool(done), False

    def get_actions(self, state: Any) -> list[int]:  # noqa: ARG002
        return list(range(self._n_actions))
