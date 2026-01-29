from __future__ import annotations

import os
from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import nn


def save_policy_checkpoint(
    path: str,
    *,
    model_state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a minimal policy checkpoint.

    The file format is compatible with `OnPolicyRunner.load`'s expectations in that it stores
    `model_state_dict`, but it intentionally omits optimizer / RND state.
    """

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload: dict[str, Any] = {"model_state_dict": model_state_dict}
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, path)


def load_policy_checkpoint(path: str, *, map_location: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either a full training checkpoint or a minimal policy checkpoint.

    Returns:
        (model_state_dict, metadata)
    """

    payload: dict[str, Any] = torch.load(path, weights_only=False, map_location=map_location)

    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint at '{path}' has no 'model_state_dict' key.")

    metadata = payload.get("metadata")
    if metadata is None:
        # Backward-compatible: training checkpoints store these at the top-level.
        metadata = {
            "iter": payload.get("iter"),
            "infos": payload.get("infos"),
        }

    assert isinstance(metadata, dict)
    return payload["model_state_dict"], metadata


class FlatPolicy(nn.Module):
    """A TorchScript-friendly policy that maps flat actor observations to actions."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        actor_obs_normalizer: nn.Module,
        state_dependent_std: bool,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.actor_obs_normalizer = actor_obs_normalizer
        self.state_dependent_std = state_dependent_std

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        actor_obs = self.actor_obs_normalizer(actor_obs)
        out = self.actor(actor_obs)
        if self.state_dependent_std:
            return out[..., 0, :]
        return out


def export_policy_as_torchscript(
    policy: nn.Module,
    *,
    example_obs: TensorDict,
    out_path: str,
    device: str = "cpu",
) -> torch.jit.ScriptModule:
    """Export a policy as a TorchScript module.

    This currently supports policies where:
    - `policy.is_recurrent == False`
    - `policy.get_actor_obs(example_obs)` returns a single flat `torch.Tensor`
    - the policy exposes `actor` and `actor_obs_normalizer` attributes

    The exported module signature is: `actions = policy(actor_obs)`.
    """

    if bool(getattr(policy, "is_recurrent", False)):
        raise NotImplementedError("TorchScript export is not implemented for recurrent policies.")

    if not hasattr(policy, "get_actor_obs"):
        raise TypeError(f"Policy type {type(policy)} has no get_actor_obs().")

    actor_obs = policy.get_actor_obs(example_obs)  # type: ignore[attr-defined]
    if not isinstance(actor_obs, torch.Tensor):
        raise NotImplementedError(
            "TorchScript export currently supports only policies whose get_actor_obs() returns a single Tensor. "
            f"Got type: {type(actor_obs)}"
        )

    if not hasattr(policy, "actor") or not hasattr(policy, "actor_obs_normalizer"):
        raise TypeError(
            "TorchScript export expects policy to have 'actor' and 'actor_obs_normalizer' attributes. "
            f"Got: {type(policy)}"
        )

    actor: nn.Module = getattr(policy, "actor")
    actor_obs_normalizer: nn.Module = getattr(policy, "actor_obs_normalizer")
    state_dependent_std: bool = bool(getattr(policy, "state_dependent_std", False))

    wrapper = FlatPolicy(
        actor=actor,
        actor_obs_normalizer=actor_obs_normalizer,
        state_dependent_std=state_dependent_std,
    )

    wrapper = wrapper.to(device)
    wrapper.eval()

    example_actor_obs = actor_obs.to(device)

    traced = cast(torch.jit.ScriptModule, torch.jit.trace(wrapper, example_actor_obs, strict=False))
    try:
        traced = cast(torch.jit.ScriptModule, torch.jit.freeze(traced))
    except Exception:
        # Freezing is optional; tracing already produced a runnable module.
        pass

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    traced.save(out_path)

    return traced
