from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from rsl_rl.train_isaac import make_isaaclab_vec_env


def _is_discrete_action_space(space: Any) -> bool:
    return hasattr(space, "n") and isinstance(getattr(space, "n"), int)


@pytest.mark.isaaclab
def test_isaaclab_env_smoke() -> None:
    # Isaac Lab is heavy and typically not present in CI. Opt-in only.
    if os.getenv("RSL_RL_RUN_ISAACLAB_TESTS", "0") != "1":
        pytest.skip("Set RSL_RL_RUN_ISAACLAB_TESTS=1 to run Isaac Lab env tests")

    # skrl is required (and Isaac Lab must be installed for the loader to work)
    pytest.importorskip("skrl")

    task = os.getenv("ISAACLAB_TASK", "Isaac-Cartpole-Direct-v0")
    num_envs = int(os.getenv("ISAACLAB_NUM_ENVS", "4"))

    try:
        env = make_isaaclab_vec_env(task_name=task, num_envs=num_envs, headless=True, show_cfg=False)
    except Exception as err:
        pytest.skip(f"Isaac Lab env could not be created: {err}")

    obs = env.get_observations()
    assert "policy" in obs, f"obs keys: {list(obs.keys())}"

    wrapped = getattr(env, "_env", None)
    action_space = getattr(wrapped, "action_space", None)

    if action_space is not None and _is_discrete_action_space(action_space):
        n = int(action_space.n)
        actions = torch.randint(low=0, high=n, size=(env.num_envs, 1), device=env.device, dtype=torch.long)
    else:
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    obs2, rewards, dones, extras = env.step(actions)

    assert "policy" in obs2
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(dones, torch.Tensor)
    assert rewards.shape[0] == env.num_envs
    assert dones.shape[0] == env.num_envs
    assert isinstance(extras, dict)
    assert "time_outs" in extras
