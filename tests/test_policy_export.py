import os
import tempfile

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.utils.policy_export import (
    export_policy_as_torchscript,
    load_policy_checkpoint,
    save_policy_checkpoint,
)


def test_save_and_load_policy_checkpoint_roundtrip() -> None:
    state_dict = {"a": torch.randn(3, 4)}
    metadata = {"iter": 123, "foo": "bar"}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "policy.pt")
        save_policy_checkpoint(path, model_state_dict=state_dict, metadata=metadata)
        loaded_state_dict, loaded_metadata = load_policy_checkpoint(path)

    assert loaded_state_dict["a"].shape == (3, 4)
    assert loaded_metadata["iter"] == 123
    assert loaded_metadata["foo"] == "bar"


def test_export_policy_as_torchscript_actor_critic() -> None:
    batch = 2
    actor_dim = 11
    critic_dim = 7
    num_actions = 4

    obs = TensorDict(
        {
            "policy_obs": torch.randn(batch, actor_dim),
            "critic_obs": torch.randn(batch, critic_dim),
        },
        batch_size=[batch],
    )

    obs_groups = {"policy": ["policy_obs"], "critic": ["critic_obs"]}
    policy = ActorCritic(obs, obs_groups, num_actions=num_actions, actor_obs_normalization=False)
    policy.eval()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "policy_jit.pt")
        export_policy_as_torchscript(policy, example_obs=obs, out_path=out_path, device="cpu")

        loaded = torch.jit.load(out_path)
        x = torch.randn(batch, actor_dim)
        y = loaded(x)

    assert tuple(y.shape) == (batch, num_actions)
