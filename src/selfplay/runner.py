from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from src.agents import PUCTAgent

# Trajectory step before value assignment: (state, policy, reward)
Step = tuple[Any, list[float], float]
# Final trajectory step with return: (state, policy, value)
TrajectoryStep = tuple[Any, list[float], float]
Trajectory = list[TrajectoryStep]


def run_episode(env: Any, agent: PUCTAgent) -> Trajectory:
    """Run one episode and return a trajectory of (state, policy, value)."""
    agent.reset_tree()
    state = env.reset()

    buffer: list[Step] = []

    while True:
        policy = agent.get_policy(state)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated = env.step(action)
        buffer.append((state, policy, reward))
        state = next_state
        if terminated or truncated:
            break

    return _assign_returns(buffer)


def run_parallel(
    env_fn: Callable[[], Any],
    agent_fn: Callable[[], PUCTAgent],
    episodes: int,
    workers: int,
) -> list[Trajectory]:
    """Run *episodes* episodes in parallel using *workers* processes."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_episode_worker, env_fn, agent_fn)
            for _ in range(episodes)
        ]
        return [f.result() for f in as_completed(futures)]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _assign_returns(buffer: list[Step]) -> Trajectory:
    """Replace per-step rewards with the cumulative return (sum of future rewards)."""
    G = 0.0
    trajectory: Trajectory = []
    for state, policy, reward in reversed(buffer):
        G += reward
        trajectory.append((state, policy, G))
    trajectory.reverse()
    return trajectory


def _run_episode_worker(
    env_fn: Callable[[], Any],
    agent_fn: Callable[[], PUCTAgent],
) -> Trajectory:
    """Top-level function (picklable) used by ProcessPoolExecutor."""
    return run_episode(env_fn(), agent_fn())
