from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

from rsl_rl.train_isaac import make_isaaclab_vec_env


def _is_discrete_action_space(space: Any) -> bool:
    return hasattr(space, "n") and isinstance(getattr(space, "n"), int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal Isaac Lab env smoke test (via skrl wrapper)")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--strict", action="store_true", help="Fail hard if dependencies are missing")
    args = parser.parse_args(argv)

    try:
        env = make_isaaclab_vec_env(task_name=args.task, num_envs=args.num_envs, headless=args.headless)
    except ModuleNotFoundError as err:
        if args.strict:
            raise
        print(f"[test_env] Skipping (missing deps): {err}")
        return 0

    # Try to introspect wrapped env spaces (best-effort)
    wrapped = getattr(env, "_env", None)
    action_space = getattr(wrapped, "action_space", None)

    print("[test_env] num_envs:", env.num_envs)
    print("[test_env] num_actions:", env.num_actions)
    print("[test_env] device:", env.device)
    print("[test_env] action_space:", type(action_space).__name__ if action_space is not None else None)

    obs = env.get_observations()
    print("[test_env] obs keys:", list(obs.keys()))
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            print(f"[test_env] obs[{k}] shape:", tuple(v.shape), "dtype:", v.dtype, "device:", v.device)

    # Roll a few steps
    for i in range(args.steps):
        if action_space is not None and _is_discrete_action_space(action_space):
            # Discrete: sample integer actions in [0, n)
            n = int(action_space.n)
            actions = torch.randint(low=0, high=n, size=(env.num_envs, 1), device=env.device, dtype=torch.long)
        else:
            actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2.0 - 1.0

        obs, rewards, dones, extras = env.step(actions)

        if i == 0:
            print("[test_env] step() returns:")
            print("  obs type:", type(obs).__name__, "keys:", list(obs.keys()))
            print("  rewards shape:", tuple(rewards.shape), "dtype:", rewards.dtype)
            print("  dones shape:", tuple(dones.shape), "dtype:", dones.dtype)
            print("  extras keys:", list(extras.keys()))

        # Basic sanity checks
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert rewards.shape[0] == env.num_envs
        assert dones.shape[0] == env.num_envs

    print("[test_env] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
