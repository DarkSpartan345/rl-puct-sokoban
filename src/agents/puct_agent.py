from __future__ import annotations

import random
from typing import Any, Callable, Protocol

from src.mcts import PUCTMCTS

PriorFn = Callable[[Any], tuple[list[float], float]]


class Env(Protocol):
    def step(self, action: int) -> tuple[Any, float, bool, bool]: ...
    def get_actions(self, state: Any) -> list[int]: ...


class PUCTAgent:
    def __init__(
        self,
        env: Env,
        n_simulations: int = 100,
        c: float = 1.0,
        prior_fn: PriorFn | None = None,
    ) -> None:
        self._env = env
        self._n_simulations = n_simulations
        self._c = c
        self._prior_fn = prior_fn
        self._mcts = self._make_mcts()

    def select_action(self, state: Any) -> int:
        policy = self.get_policy(state)
        actions = self._env.get_actions(state)
        return random.choices(actions, weights=policy, k=1)[0]

    def get_policy(self, state: Any) -> list[float]:
        policy, _ = self._mcts.search(state, self._env)
        return policy

    def reset_tree(self) -> None:
        self._mcts = self._make_mcts()

    def set_prior_fn(self, prior_fn: PriorFn | None) -> None:
        self._prior_fn = prior_fn
        self._mcts = self._make_mcts()

    def _make_mcts(self) -> PUCTMCTS:
        return PUCTMCTS(
            n_simulations=self._n_simulations,
            c=self._c,
            prior_fn=self._prior_fn,
        )
