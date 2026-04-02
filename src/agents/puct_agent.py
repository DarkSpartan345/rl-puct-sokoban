from __future__ import annotations

import random
from typing import Any, Protocol

from src.mcts import PUCTMCTS


class Env(Protocol):
    def step(self, action: int) -> tuple[Any, float, bool, bool]: ...
    def get_actions(self, state: Any) -> list[int]: ...


class PUCTAgent:
    def __init__(self, env: Env, n_simulations: int = 100, c: float = 1.0) -> None:
        self._env = env
        self._n_simulations = n_simulations
        self._c = c
        self._mcts = PUCTMCTS(n_simulations=n_simulations, c=c)

    def select_action(self, state: Any) -> int:
        policy = self.get_policy(state)
        actions = self._env.get_actions(state)
        return random.choices(actions, weights=policy, k=1)[0]

    def get_policy(self, state: Any) -> list[float]:
        policy, _ = self._mcts.search(state, self._env)
        return policy

    def reset_tree(self) -> None:
        self._mcts = PUCTMCTS(n_simulations=self._n_simulations, c=self._c)