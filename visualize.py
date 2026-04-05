"""Visualización del agente entrenado: métricas, política y video."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import imageio.v3 as iio
import torch
import gymnasium as gym
import minigrid  # noqa: F401

from src.network import PolicyValueNet, make_prior_fn
from src.envs import MiniGridWrapper

# --- Config ---
MODEL_PATH = "model.pt"
METRICS_PATH = "metrics.json"
ENV_ID = "MiniGrid-Empty-5x5-v0"
INPUT_DIM = 148
N_ACTIONS = 7
HIDDEN_DIMS = [128, 128]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "viz_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


# ------------------------------------------------------------------
# Carga del modelo
# ------------------------------------------------------------------

def load_model() -> PolicyValueNet:
    model = PolicyValueNet(INPUT_DIM, N_ACTIONS, HIDDEN_DIMS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def flatten_obs(obs: dict) -> np.ndarray:
    image = obs["image"].astype(np.float32).flatten()
    direction = np.array([obs["direction"]], dtype=np.float32)
    return np.concatenate([image, direction])


# ------------------------------------------------------------------
# 1. Curvas de entrenamiento
# ------------------------------------------------------------------

def plot_training_curves(metrics: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Métricas de Entrenamiento por Iteración", fontsize=14, fontweight="bold")

    iterations = [m["iteration"] for m in metrics]

    # Loss final por iteración
    axes[0, 0].plot(iterations, [m["final_loss"] for m in metrics], "o-", color="#e74c3c", linewidth=2)
    axes[0, 0].set_title("Loss Final por Iteración")
    axes[0, 0].set_xlabel("Iteración")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Curvas de loss por época (todas las iteraciones superpuestas)
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    for i, m in enumerate(metrics):
        axes[0, 1].plot(m["epoch_losses"], color=colors[i], linewidth=1.5,
                        label=f"Iter {m['iteration']}")
    axes[0, 1].set_title("Loss por Época (todas las iteraciones)")
    axes[0, 1].set_xlabel("Época")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Retorno medio por iteración
    axes[1, 0].bar(iterations, [m["mean_return"] for m in metrics], color="#3498db", alpha=0.8)
    axes[1, 0].plot(iterations, [m["max_return"] for m in metrics], "D--", color="#e67e22",
                    label="max return")
    axes[1, 0].set_title("Retorno por Iteración")
    axes[1, 0].set_xlabel("Iteración")
    axes[1, 0].set_ylabel("Retorno")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Longitud media de episodio
    axes[1, 1].plot(iterations, [m["mean_episode_length"] for m in metrics],
                    "s-", color="#2ecc71", linewidth=2)
    axes[1, 1].set_title("Longitud Media de Episodio")
    axes[1, 1].set_xlabel("Iteración")
    axes[1, 1].set_ylabel("Pasos")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ------------------------------------------------------------------
# 2. Visualización de política
# ------------------------------------------------------------------

def plot_policy(model: PolicyValueNet, env_raw: gym.Env) -> None:
    prior_fn = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)

    obs, _ = env_raw.reset()
    state = flatten_obs(obs)
    policy, value = prior_fn(state)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Política del Agente en Estado Inicial", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 2, 1])

    # Panel 1: render del entorno
    ax_env = fig.add_subplot(gs[0])
    frame = env_raw.render()
    ax_env.imshow(frame)
    ax_env.set_title("Estado Actual")
    ax_env.axis("off")

    # Panel 2: barras de política
    ax_pol = fig.add_subplot(gs[1])
    colors = ["#e74c3c" if p == max(policy) else "#3498db" for p in policy]
    bars = ax_pol.bar(ACTION_NAMES, policy, color=colors, alpha=0.85, edgecolor="white")
    ax_pol.set_title("Distribución de Política π(a|s)")
    ax_pol.set_ylabel("Probabilidad")
    ax_pol.set_ylim(0, 1)
    ax_pol.grid(True, axis="y", alpha=0.3)
    for bar, p in zip(bars, policy):
        ax_pol.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{p:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: valor estimado
    ax_val = fig.add_subplot(gs[2])
    color = "#2ecc71" if value >= 0 else "#e74c3c"
    ax_val.barh(["V(s)"], [value], color=color, alpha=0.85)
    ax_val.set_xlim(-1, 1)
    ax_val.set_title("Valor Estimado V(s)")
    ax_val.axvline(0, color="gray", linewidth=0.8)
    ax_val.text(value + 0.05 * np.sign(value) if abs(value) < 0.9 else value - 0.15,
                0, f"{value:.3f}", va="center", fontsize=12, fontweight="bold")
    ax_val.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "policy_state0.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ------------------------------------------------------------------
# 3. Evolución de política a lo largo de un episodio
# ------------------------------------------------------------------

def plot_policy_episode(model: PolicyValueNet, env_raw: gym.Env, max_steps: int = 20) -> None:
    prior_fn = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)
    obs, _ = env_raw.reset()

    policies_over_time = []
    values_over_time = []
    steps = 0

    for _ in range(max_steps):
        state = flatten_obs(obs)
        policy, value = prior_fn(state)
        policies_over_time.append(policy)
        values_over_time.append(value)

        action = int(np.argmax(policy))
        obs, _, terminated, truncated, _ = env_raw.step(action)
        steps += 1
        if terminated or truncated:
            break

    policies_arr = np.array(policies_over_time)  # (T, n_actions)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"Evolución de Política y Valor a lo largo del Episodio ({steps} pasos)",
                 fontsize=13, fontweight="bold")

    # Heatmap de política
    im = axes[0].imshow(policies_arr.T, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_yticks(range(N_ACTIONS))
    axes[0].set_yticklabels(ACTION_NAMES)
    axes[0].set_xlabel("Paso")
    axes[0].set_title("π(a|s) por paso")
    plt.colorbar(im, ax=axes[0], label="Probabilidad")

    # Valor estimado
    axes[1].plot(values_over_time, "o-", color="#e74c3c", linewidth=2, markersize=5)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].fill_between(range(len(values_over_time)), values_over_time, alpha=0.2, color="#e74c3c")
    axes[1].set_xlabel("Paso")
    axes[1].set_ylabel("V(s)")
    axes[1].set_title("Valor Estimado por Paso")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "policy_episode.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ------------------------------------------------------------------
# 4. Entropía de política (cuánto explora el agente)
# ------------------------------------------------------------------

def plot_policy_entropy(model: PolicyValueNet, n_episodes: int = 10) -> None:
    prior_fn = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)
    wrapper = MiniGridWrapper(env_id=ENV_ID)

    entropies = []
    for _ in range(n_episodes):
        state = wrapper.reset()
        for _ in range(50):
            policy, _ = prior_fn(state)
            p = np.array(policy) + 1e-8
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
            action = int(np.argmax(policy))
            state, _, terminated, truncated = wrapper.step(action)
            if terminated or truncated:
                break

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(entropies, alpha=0.6, color="#9b59b6", linewidth=1)
    ax.axhline(np.log(N_ACTIONS), color="gray", linestyle="--",
               label=f"Entropía máxima (uniforme) = {np.log(N_ACTIONS):.2f}")
    ax.axhline(np.mean(entropies), color="#e74c3c", linestyle="-",
               label=f"Media = {np.mean(entropies):.2f}")
    ax.set_title("Entropía de la Política H(π) — cuánto explora el agente")
    ax.set_xlabel("Paso (todos los episodios concatenados)")
    ax.set_ylabel("Entropía (nats)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "policy_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ------------------------------------------------------------------
# 5. Video del agente jugando
# ------------------------------------------------------------------

def record_video(model: PolicyValueNet, env_raw: gym.Env,
                 max_steps: int = 100, fps: int = 4) -> None:
    prior_fn = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)
    import cv2
    obs, _ = env_raw.reset()

    # Usar prior para obtener política del estado inicial y generar el primer frame con overlay
    prior_fn2 = make_prior_fn(model, input_dim=INPUT_DIM, device=DEVICE)
    state0 = flatten_obs(obs)
    p0, v0 = prior_fn2(state0)
    f0 = _add_overlay(env_raw.render(), 0, "-", 0.0, v0, p0)
    frames = [f0]

    total_reward = 0.0
    for step in range(max_steps):
        state = flatten_obs(obs)
        policy, value = prior_fn(state)
        action = int(np.argmax(policy))
        obs, reward, terminated, truncated, _ = env_raw.step(action)
        total_reward += reward
        frame = env_raw.render()

        # Overlay: paso, acción, recompensa, valor
        frame = _add_overlay(frame, step + 1, ACTION_NAMES[action], total_reward, value, policy)
        frames.append(frame)

        if terminated or truncated:
            # Congelar último frame 1 segundo
            frames.extend([frames[-1]] * fps)
            break

    path = os.path.join(OUTPUT_DIR, "agent_play.mp4")
    iio.imwrite(path, np.array(frames, dtype=np.uint8), fps=fps)
    print(f"  → {path}  ({len(frames)} frames, reward={total_reward:.2f})")


_FRAME_H = 160
_FRAME_W = 160
_PANEL_W = 160
_TOTAL_W = _FRAME_W + _PANEL_W
_DPI = 80


def _add_overlay(
    frame: np.ndarray,
    step: int,
    action_name: str,
    total_reward: float,
    value: float,
    policy: list[float],
) -> np.ndarray:
    """Dibuja un panel de información junto al frame del entorno."""
    fig = plt.figure(figsize=(_TOTAL_W / _DPI, _FRAME_H / _DPI), dpi=_DPI)
    fig.patch.set_facecolor("#1a1a2e")

    env_frac = _FRAME_W / _TOTAL_W
    panel_frac = _PANEL_W / _TOTAL_W

    # Imagen del entorno (redimensionada si es necesario)
    import cv2
    frame_resized = cv2.resize(frame, (_FRAME_W, _FRAME_H))

    ax_env = fig.add_axes([0, 0, env_frac, 1])
    ax_env.imshow(frame_resized)
    ax_env.axis("off")

    # Panel derecho
    ax_panel = fig.add_axes([env_frac + 0.01, 0.05, panel_frac - 0.02, 0.9])
    ax_panel.set_facecolor("#16213e")
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")

    ax_panel.text(0.5, 0.97, f"Paso {step}", ha="center", va="top",
                  color="white", fontsize=9, fontweight="bold")
    ax_panel.text(0.5, 0.88, f"Acción: {action_name}", ha="center", va="top",
                  color="#f39c12", fontsize=8)
    ax_panel.text(0.5, 0.80, f"Reward: {total_reward:.2f}", ha="center", va="top",
                  color="#2ecc71", fontsize=8)
    val_color = "#2ecc71" if value >= 0 else "#e74c3c"
    ax_panel.text(0.5, 0.72, f"V(s): {value:.3f}", ha="center", va="top",
                  color=val_color, fontsize=8)

    ax_panel.text(0.5, 0.63, "π(a|s)", ha="center", va="top",
                  color="white", fontsize=7)
    bar_colors = ["#e74c3c" if p == max(policy) else "#3498db" for p in policy]
    max_p = max(policy) if max(policy) > 0 else 1.0
    for i, (p, name, c) in enumerate(zip(policy, ACTION_NAMES, bar_colors)):
        y = 0.57 - i * 0.076
        ax_panel.barh([y], [p / max_p * 0.85], left=0.05, height=0.055,
                      color=c, alpha=0.85)
        ax_panel.text(0.02, y, name[:3], va="center", color="white", fontsize=5.5)
        ax_panel.text(0.93, y, f"{p:.2f}", va="center", ha="right",
                      color="white", fontsize=5.5)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    result = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = load_model()

    print("\n[1/5] Curvas de entrenamiento...")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        plot_training_curves(metrics)
    else:
        print(f"  {METRICS_PATH} no encontrado — ejecuta loop.py primero.")

    env_render = gym.make(ENV_ID, render_mode="rgb_array")

    print("\n[2/5] Política en estado inicial...")
    plot_policy(model, env_render)

    print("\n[3/5] Evolución de política en episodio...")
    plot_policy_episode(model, env_render)

    print("\n[4/5] Entropía de política...")
    plot_policy_entropy(model)

    print("\n[5/5] Grabando video del agente...")
    env_render_video = gym.make(ENV_ID, render_mode="rgb_array")
    record_video(model, env_render_video)

    print(f"\nTodo guardado en {OUTPUT_DIR}/")
    print("  training_curves.png  — curvas de loss y retorno")
    print("  policy_state0.png    — política en estado inicial")
    print("  policy_episode.png   — heatmap de política en episodio")
    print("  policy_entropy.png   — entropía H(π) a lo largo del tiempo")
    print("  agent_play.mp4       — video del agente jugando")


if __name__ == "__main__":
    main()
