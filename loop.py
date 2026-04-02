"""Loop iterativo AlphaZero: self-play → train → actualizar prior → repetir."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.envs import MiniGridWrapper
from src.agents import PUCTAgent
from src.network import PolicyValueNet, make_prior_fn
from src.network.inference import PriorFn
from src.selfplay import run_episode, run_parallel
from src.training import build_dataset, save_dataset

# --- Config ---
ENV_ID = "MiniGrid-Empty-5x5-v0"
INPUT_DIM = 148
N_ACTIONS = 7
HIDDEN_DIMS = [128, 128]

N_ITERATIONS = 5
EPISODES_PER_ITER = 8
WORKERS = 2
N_SIMULATIONS = 20
C = 1.0

EPOCHS_PER_ITER = 10
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model.pt"
DATASET_PATH = "dataset_latest.npz"

# prior global compartido entre iteraciones (picklable)
_current_prior: PriorFn | None = None


# ------------------------------------------------------------------
# Funciones top-level (picklables por ProcessPoolExecutor)
# ------------------------------------------------------------------

def _env_fn() -> MiniGridWrapper:
    return MiniGridWrapper(env_id=ENV_ID)


def _agent_fn() -> PUCTAgent:
    return PUCTAgent(_env_fn(), n_simulations=N_SIMULATIONS, c=C, prior_fn=_current_prior)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def self_play(prior_fn: PriorFn | None, episodes: int) -> list:
    global _current_prior
    _current_prior = prior_fn

    if WORKERS > 1:
        return run_parallel(_env_fn, _agent_fn, episodes=episodes, workers=WORKERS)
    else:
        env = _env_fn()
        agent = _agent_fn()
        return [run_episode(env, agent) for _ in range(episodes)]


def train_model(model: PolicyValueNet, dataset_path: str) -> None:
    data = np.load(dataset_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    policies = torch.tensor(data["policies"], dtype=torch.float32)
    values = torch.tensor(data["values"], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(states, policies, values),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    for epoch in range(1, EPOCHS_PER_ITER + 1):
        total = 0.0
        for s, p, v in loader:
            s, p, v = s.to(DEVICE), p.to(DEVICE), v.to(DEVICE)
            log_p_pred, v_pred = model(s)
            loss = -(p * log_p_pred).sum(-1).mean() + nn.functional.mse_loss(v_pred, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"    epoch {epoch:>2}/{EPOCHS_PER_ITER} loss {total/len(loader):.4f}")


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE} | Iterations: {N_ITERATIONS} | "
          f"Episodes/iter: {EPISODES_PER_ITER} | Simulations: {N_SIMULATIONS}\n")

    model = PolicyValueNet(INPUT_DIM, N_ACTIONS, HIDDEN_DIMS).to(DEVICE)
    prior_fn: PriorFn | None = None  # iteración 1: prior uniforme + rollout

    for it in range(1, N_ITERATIONS + 1):
        mode = "red neuronal" if prior_fn else "prior uniforme"
        print(f"=== Iteración {it}/{N_ITERATIONS} | prior: {mode} ===")

        # 1. Self-play
        print(f"  Self-play ({EPISODES_PER_ITER} episodios)...")
        trajectories = self_play(prior_fn, EPISODES_PER_ITER)
        steps = sum(len(t) for t in trajectories)
        print(f"  Recolectados {steps} steps.")

        # 2. Dataset
        dataset = build_dataset(trajectories)
        save_dataset(dataset, DATASET_PATH)

        # 3. Entrenamiento
        print(f"  Entrenando ({EPOCHS_PER_ITER} épocas)...")
        train_model(model, DATASET_PATH)

        # 4. Actualizar prior para la siguiente iteración
        prior_fn = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)
        print()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Loop completado. Modelo guardado → {MODEL_PATH}")


if __name__ == "__main__":
    main()
