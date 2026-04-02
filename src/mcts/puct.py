from __future__ import annotations

import copy
import math
import random
from typing import Any, Protocol

from .node import Node


class Env(Protocol):
    def step(self, action: int) -> tuple[Any, float, bool, bool]: ...
    def get_actions(self, state: Any) -> list[int]: ...


class PUCTMCTS:
    def __init__(self, n_simulations: int = 100, c: float = 1.0) -> None:
        self.n_simulations = n_simulations
        self.c = c
        self._root: Node | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, state: Any, env: Env) -> tuple[list[float], float]:
        """Run MCTS from *state* and return (policy, value).

        policy  — visit counts normalised over legal actions.
        value   — mean value of the root node after search.
        """
        self._root = Node(state)
        actions = env.get_actions(state)

        for _ in range(self.n_simulations):
            sim_env = copy.deepcopy(env)
            self._simulate(self._root, sim_env)

        policy = self._get_policy(self._root, actions)
        value = self._root.Q if self._root.N > 0 else 0.0
        return policy, value

    def update(self, action: int, next_state: Any) -> None:
        """Advance the root to the child reached by *action*."""
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
        node, depth_reward = self._select(root, env)

        if node is None:
            return

        terminal_value = self._rollout(node, env)
        value = depth_reward + terminal_value
        self._backpropagate(node, value)

    def _select(self, node: Node, env: Env) -> tuple[Node | None, float]:
        """Walk the tree using PUCT until a leaf; return (leaf, accumulated reward)."""
        accumulated = 0.0

        while not node.is_leaf():
            actions = env.get_actions(node.state)
            if not actions:
                return None, accumulated

            action = self._best_action(node, actions)
            child = node.children.get(action)

            if child is None:
                # Expand this action
                next_state, reward, terminated, truncated = env.step(action)
                child = Node(next_state, parent=node, prior=1.0 / len(actions))
                node.children[action] = child
                accumulated += reward
                if terminated or truncated:
                    return child, accumulated
                return child, accumulated

            next_state, reward, terminated, truncated = env.step(action)
            accumulated += reward
            node = child

            if terminated or truncated:
                return node, accumulated

        # Leaf node — expand all legal actions
        actions = env.get_actions(node.state)
        if not actions:
            return node, accumulated

        n_actions = len(actions)
        for a in actions:
            node.children[a] = Node(state=None, parent=node, prior=1.0 / n_actions)

        # Pick first child to evaluate
        action = actions[0]
        next_state, reward, terminated, truncated = env.step(action)
        child = node.children[action]
        child.state = next_state
        accumulated += reward
        return child, accumulated

    def _best_action(self, node: Node, actions: list[int]) -> int:
        """Select action maximising PUCT score."""
        sqrt_parent = math.sqrt(node.N + 1)
        best_score = -float("inf")
        best_action = actions[0]

        for action in actions:
            child = node.children.get(action)
            if child is None:
                q = 0.0
                n_child = 0
                prior = 1.0 / len(actions)
            else:
                q = child.Q
                n_child = child.N
                prior = child.P

            u = self.c * prior * sqrt_parent / (1 + n_child)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _rollout(self, node: Node, env: Env) -> float:
        """Random rollout from *node*; returns cumulative reward."""
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

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def _get_policy(self, root: Node, actions: list[int]) -> list[float]:
        visits = [root.children[a].N if a in root.children else 0 for a in actions]
        total = sum(visits)
        if total == 0:
            n = len(actions)
            return [1.0 / n] * n
        return [v / total for v in visits]
