from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .mlp import PolicyValueNet


class PriorFn:
    """Callable picklable que envuelve PolicyValueNet para usarse como prior en MCTS."""

    def __init__(self, model: PolicyValueNet, input_dim: int, device: str = "cpu") -> None:
        self._state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self._input_dim = input_dim
        self._n_actions = model.policy_head.out_features
        self._hidden_dims: list[int] = [
            layer.out_features
            for layer in model.backbone
            if isinstance(layer, torch.nn.Linear)
        ]
        self._device = device
        self._model: PolicyValueNet | None = None  # cargado lazy en el worker

    def _load(self) -> None:
        if self._model is None:
            m = PolicyValueNet(self._input_dim, self._n_actions, self._hidden_dims)
            m.load_state_dict(self._state_dict)
            m.eval()
            self._model = m.to(self._device)

    def __call__(self, state: np.ndarray) -> tuple[list[float], float]:
        self._load()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            log_policy, value = self._model(x)
        policy = log_policy.exp().squeeze(0).cpu().tolist()
        return policy, float(value.item())


def make_prior_fn(model: PolicyValueNet, input_dim: int, device: str = "cpu") -> PriorFn:
    return PriorFn(model, input_dim=input_dim, device=device)
