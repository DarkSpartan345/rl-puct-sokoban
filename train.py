"""Entrena PolicyValueNet con el dataset generado por main.py."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.network import PolicyValueNet

# --- Config ---
DATASET_PATH = "dataset.npz"
MODEL_PATH = "model.pt"
INPUT_DIM = 148       # 7×7×3 imagen + 1 dirección
N_ACTIONS = 7
HIDDEN_DIMS = [128, 128]
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path: str) -> TensorDataset:
    data = np.load(path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    policies = torch.tensor(data["policies"], dtype=torch.float32)
    values = torch.tensor(data["values"], dtype=torch.float32)
    return TensorDataset(states, policies, values)


def train() -> None:
    print(f"Device: {DEVICE}")

    dataset = load_dataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PolicyValueNet(INPUT_DIM, N_ACTIONS, HIDDEN_DIMS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss = total_policy_loss = total_value_loss = 0.0
        model.train()

        for states, policies, values in loader:
            states = states.to(DEVICE)
            policies = policies.to(DEVICE)
            values = values.to(DEVICE)

            log_policy_pred, value_pred = model(states)

            # cross-entropy: -sum(target * log_pred)
            policy_loss = -(policies * log_policy_pred).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(value_pred, values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        n = len(loader)
        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"loss {total_loss/n:.4f} | "
            f"policy {total_policy_loss/n:.4f} | "
            f"value {total_value_loss/n:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado → {MODEL_PATH}")


if __name__ == "__main__":
    train()
