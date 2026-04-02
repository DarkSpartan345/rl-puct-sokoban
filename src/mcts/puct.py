from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable, Protocol

from .node import Node


class Env(Protocol):
    def step(self, action: int) -> tuple[Any, float, bool, bool]: ...
    def get_actions(self, state: Any) -> list[int]: ...


# prior_fn: state → (policy: list[float], value: float)
PriorFn = Callable[[Any], tuple[list[float], float]]


class PUCTMCTS:
    def __init__(
        self,
        n_simulations: int = 100,
        c: float = 1.0,
        prior_fn: PriorFn | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.c = c
        self.prior_fn = prior_fn  # None → prior uniforme + rollout aleatorio
        self._root: Node | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, state: Any, env: Env) -> tuple[list[float], float]:
        self._root = Node(state)
        actions = env.get_actions(state)

        for _ in range(self.n_simulations):
            sim_env = copy.deepcopy(env)
            self._simulate(self._root, sim_env)

        policy = self._get_policy(self._root, actions)
        value = self._root.Q if self._root.N > 0 else 0.0
        return policy, value

    def update(self, action: int, next_state: Any) -> None:
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None
            self._root.state = next_state
        else:
            self._root = Node(next_state)

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _simulate(self, root: Node, env: Env) -> None:
        node, accumulated_reward = self._select(root, env)
        if node is None:
            return

        value = accumulated_reward + self._evaluate(node, env)
        self._backpropagate(node, value)

    def _select(self, node: Node, env: Env) -> tuple[Node | None, float]:
        """PUCT selection hasta nodo no visitado; devuelve (nodo, recompensa acumulada)."""
        accumulated = 0.0

        while node.N > 0:
            actions = env.get_actions(node.state)
            if not actions:
                return node, accumulated

            action = self._best_action(node, actions)

            if action not in node.children:
                # Primer visit a esta acción: expandir
                next_state, reward, terminated, truncated = env.step(action)
                child = Node(next_state, parent=node, prior=1.0 / len(actions))
                node.children[action] = child
                accumulated += reward
                if terminated or truncated:
                    self._backpropagate(child, accumulated)
                    return None, 0.0
                return child, accumulated

            next_state, reward, terminated, truncated = env.step(action)
            node = node.children[action]
            node.state = next_state
            accumulated += reward

            if terminated or truncated:
                self._backpropagate(node, accumulated)
                return None, 0.0

        return node, accumulated

    def _evaluate(self, node: Node, env: Env) -> float:
        """Evalúa el nodo: con red neuronal o con rollout aleatorio."""
        if self.prior_fn is not None:
            policy, value = self.prior_fn(node.state)
            actions = env.get_actions(node.state)
            # Expandir hijos con prior de la red
            for i, a in enumerate(actions):
                if a not in node.children:
                    prior = policy[i] if i < len(policy) else 1.0 / len(actions)
                    node.children[a] = Node(state=None, parent=node, prior=prior)
            return value
        else:
            # Expandir con prior uniforme
            actions = env.get_actions(node.state)
            if actions:
                for a in actions:
                    if a not in node.children:
                        node.children[a] = Node(state=None, parent=node, prior=1.0 / len(actions))
            return self._rollout(node, env)

    def _best_action(self, node: Node, actions: list[int]) -> int:
        sqrt_parent = math.sqrt(node.N + 1)
        best_score = -float("inf")
        best_action = actions[0]

        for action in actions:
            child = node.children.get(action)
            q = child.Q if child else 0.0
            n_child = child.N if child else 0
            prior = child.P if child else 1.0 / len(actions)
            score = q + self.c * prior * sqrt_parent / (1 + n_child)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _rollout(self, node: Node, env: Env) -> float:
        state = node.state
        total = 0.0
        while True:
            actions = env.get_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state, reward, terminated, truncated = env.step(action)
            total += reward
            if terminated or truncated:
                break
        return total

    def _backpropagate(self, node: Node | None, value: float) -> None:
        while node is not None:
            node.update(value)
            node = node.parent

    def _get_policy(self, root: Node, actions: list[int]) -> list[float]:
        visits = [root.children[a].N if a in root.children else 0 for a in actions]
        total = sum(visits)
        if total == 0:
            return [1.0 / len(actions)] * len(actions)
        return [v / total for v in visits]
