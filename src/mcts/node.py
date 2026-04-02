from __future__ import annotations
from typing import Any


class Node:
    def __init__(
        self,
        state: Any,
        parent: Node | None = None,
        prior: float = 1.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: dict[int, Node] = {}

        self.N: int = 0      # visit count
        self.W: float = 0.0  # total value
        self.Q: float = 0.0  # mean value  (W / N)
        self.P: float = prior  # prior probability

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update(self, value: float) -> None:
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
