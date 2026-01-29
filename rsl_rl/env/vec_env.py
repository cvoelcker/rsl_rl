# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that the same type of
    action is applied to all environments and the same type of observation is returned from all environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length.

    The maximum episode length can be a scalar or a tensor. If it is a scalar, it is the same for all environments.
    If it is a tensor, it is the maximum episode length for each environment. This is useful for dynamic episode
    lengths.
    """

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device | str
    """Device to use."""

    cfg: dict | object
    """Configuration object."""

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            The observations from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            observations: Observations from the environment.
            rewards: Rewards from the environment. Shape: (num_envs,)
            dones: Done flags from the environment. Shape: (num_envs,)
            extras: Extra information from the environment.

        Observations:
            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps the available observation groups to observation sets. The observation sets
            (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "policy": Specified observation groups are used as input to the actor/student network.
            - "critic": Specified observation groups are used as input to the critic network.
            - "teacher": Specified observation groups are used as input to the teacher network.
            - "rnd_state": Specified observation groups are used as input to the RND network.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`.

        Extras:
            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that
               happen due to time limits and not due to the environment reaching a terminal state. This is useful
               for environments that have a fixed episode length.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.
        """
        raise NotImplementedError


def _prod(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def _space_num_actions(space: Any) -> int:
    # gymnasium.spaces.Box
    if hasattr(space, "shape") and space.shape is not None:
        return _prod(space.shape)
    # gymnasium.spaces.Discrete
    if hasattr(space, "n"):
        return 1
    raise TypeError(f"Unsupported action space type: {type(space)}")


@dataclass
class _UnwrappedFallback:
    step_dt: float = 1.0


class SkrlIsaacLabVecEnv(VecEnv):
    """Adapter from skrl's Isaac Lab wrapper API to rsl_rl's `VecEnv` API.

    rsl_rl expects:
    - `get_observations()` returning a `tensordict.TensorDict` with named observation groups.
    - `step()` returning `(obs: TensorDict, rewards: Tensor, dones: Tensor, extras: dict)`.

    skrl wrappers expose a gym-like API:
    - `reset() -> (obs, info)`
    - `step(actions) -> (obs, reward, terminated, truncated, info)`
    """

    def __init__(self, env: Any):
        self._env = env

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = getattr(env, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = getattr(env, "cfg", {})
        self.unwrapped = getattr(env, "unwrapped", _UnwrappedFallback())

        self.num_actions = _space_num_actions(getattr(env, "action_space"))

        max_ep = getattr(env, "max_episode_length", None)
        if max_ep is None:
            max_ep = getattr(env, "max_episode_steps", None)
        if max_ep is None:
            max_ep = 1
        self.max_episode_length = max_ep
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        obs, _info = env.reset()
        self._obs = obs

    def _to_obs_tensordict(self, obs: Any) -> TensorDict:
        if isinstance(obs, TensorDict):
            return obs

        if isinstance(obs, dict):
            out: dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                if not isinstance(v, torch.Tensor):
                    continue
                out[str(k)] = v
            if not out:
                raise TypeError("Observation dict contained no torch.Tensors")
            batch_size = [next(iter(out.values())).shape[0]]
            return TensorDict(out, batch_size=batch_size, device=next(iter(out.values())).device)

        if isinstance(obs, torch.Tensor):
            batch_size = [obs.shape[0]] if obs.ndim >= 1 else [1]
            return TensorDict({"policy": obs, "critic": obs}, batch_size=batch_size, device=obs.device)

        raise TypeError(f"Unsupported observation type from env: {type(obs)}")

    def get_observations(self) -> TensorDict:
        return self._to_obs_tensordict(self._obs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=self.device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.as_tensor(terminated, device=self.device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.as_tensor(truncated, device=self.device)

        dones = (terminated | truncated).to(torch.int32)

        self.episode_length_buf += 1
        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.episode_length_buf[done_ids] = 0

        self._obs = obs

        extras: dict[str, Any] = {}
        if isinstance(info, dict):
            if isinstance(info.get("log"), dict):
                extras["log"] = info["log"]
            if isinstance(info.get("episode"), dict):
                extras["episode"] = info["episode"]
            extras["info"] = info

        extras["time_outs"] = truncated

        return self._to_obs_tensordict(obs), rewards, dones, extras
