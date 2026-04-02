from __future__ import annotations

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """MLP con dos cabezas: policy y value.

    policy head: distribución sobre acciones (log-softmax).
    value head:  escalar estimado del retorno (tanh).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        log_policy = torch.log_softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return log_policy, value
