# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


class Transition:
    """Storage for a single state transition."""

    def __init__(self) -> None:
        self.observations: TensorDict | None = None
        self.actions: torch.Tensor | None = None
        self.privileged_actions: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.soft_rewards: torch.Tensor | None = None
        self.clean_returns: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
        self.truncations: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.actions_log_prob: torch.Tensor
        self.next_actions_log_prob: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self.action_sigma: torch.Tensor | None = None
        self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)

    def clear(self) -> None:
        self.__init__()


class RolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, num_envs, *value.shape[1:], device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.soft_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.truncations = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.curr_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.clean_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        self.filled_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_filled_index = 0

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        if transition.soft_rewards is not None:
            self.soft_rewards[self.step].copy_(transition.soft_rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if transition.truncations is not None:
            self.truncations[self.step].copy_(transition.truncations.view(-1, 1))
        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values.view(-1, 1))
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

        # mark as filled up to the current step for all environments (since we add transitions for all envs at once)
        self.filled_indices[:transition.observations.shape[0]] = 1
        self.max_filled_index = max(self.max_filled_index, transition.observations.shape[0])

    # For distillation
    def generator(self) -> Generator:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        # Flatten all data
        # select only from valid rows (where filled_indices is 1) to avoid including uninitialized data in the batches
        observations = self.observations[:, :self.max_filled_index].flatten(0, 1)
        actions = self.actions[:, :self.max_filled_index].flatten(0, 1)
        values = self.values[:, :self.max_filled_index].flatten(0, 1)
        returns = self.returns[:, :self.max_filled_index].flatten(0, 1)
        clean_returns = self.clean_returns[:, :self.max_filled_index].flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob[:, :self.max_filled_index].flatten(0, 1)
        advantages = self.advantages[:, :self.max_filled_index].flatten(0, 1)
        old_mu = self.mu[:, :self.max_filled_index].flatten(0, 1)
        old_sigma = self.sigma[:, :self.max_filled_index].flatten(0, 1)

        termination = self.dones[:, :self.max_filled_index].flatten(0, 1)
        dones = termination

        # Mask out truncated transitions (they have invalid value targets)
        truncation_mask = ~self.truncations[:, :self.max_filled_index].flatten(0, 1).squeeze(-1).bool()
        valid_indices = torch.nonzero(truncation_mask, as_tuple=False).squeeze(-1)
        # valid_indices = torch.ones_like(valid_indices)
        num_valid = valid_indices.shape[0]

        # Compute batch sizes based on valid transitions
        mini_batch_size = num_valid // num_mini_batches

        for epoch in range(num_epochs):
            # Shuffle valid indices each epoch
            perm = torch.randperm(num_valid, requires_grad=False, device=self.device)
            shuffled_indices = valid_indices[perm]

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                if i == num_mini_batches - 1:  # Last batch takes the rest
                    stop = num_valid
                batch_idx = shuffled_indices[start:stop]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                clean_returns_batch = clean_returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                dones_batch = dones[batch_idx]

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # Yield the mini-batch (no valid_mask needed since we filtered truncations)
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    clean_returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    dones_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                    None,  # valid_mask - None for feedforward since truncations are filtered out
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        # For recurrent training, we create a mask for truncated transitions
        # so they don't contribute to losses while maintaining trajectory structure
        truncation_mask = self.truncations.squeeze(-1).bool()
        # Mask for valid (non-truncated) transitions: 1.0 for valid, 0.0 for truncated
        valid_mask = (~truncation_mask).float().unsqueeze(-1)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                clean_returns_batch = self.clean_returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                valid_mask_batch = valid_mask[:, start:stop]

                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                hidden_state_a_batch = [
                    saved_hidden_state
                    .permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_a
                ]
                hidden_state_c_batch = [
                    saved_hidden_state
                    .permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_c
                ]
                # Remove the tuple for GRU
                hidden_state_a_batch = (
                    hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                )
                hidden_state_c_batch = (
                    hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                )

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    clean_returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                    valid_mask_batch,
                )

                first_traj = last_traj

    def clear(self) -> None:
        self.step = 0

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        if hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden states to match the LSTM format
        hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize hidden states if needed
        if self.saved_hidden_state_a is None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        # Copy the states
        for i in range(len(hidden_state_a)):
            self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])
            self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])


class HybridRolloutStorage:
    """Storage for hybrid on-policy off-policy replay buffers."""

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        num_off_policy_trajectories: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        
        self.rollout_storage = RolloutStorage(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )
        self.off_policy_storage = RolloutStorage(
            training_type="rl",
            num_envs=num_off_policy_trajectories,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )
        self.off_policy_add_index = 0

        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env


    def add_transition(self, transition: Transition) -> None:
        self.rollout_storage.add_transition(transition)

    def _copy_env_columns(self, src, dst, src_env_slice, dst_env_indices) -> None:
        """Copy env-dimension columns from src storage to dst storage.

        Tensors in RolloutStorage are shaped [T, N, ...]. This helper selects
        env columns src[:, src_env_slice] and writes them to dst[:, dst_env_indices].
        TensorDict observations are handled key-by-key since fancy env-dim indexing
        does not support in-place assignment via TensorDict directly.
        """
        for key in src.observations.keys():
            dst.observations[key][:, dst_env_indices] = src.observations[key][:, src_env_slice]
        dst.actions[:, dst_env_indices] = src.actions[:, src_env_slice]
        dst.rewards[:, dst_env_indices] = src.rewards[:, src_env_slice]
        dst.dones[:, dst_env_indices] = src.dones[:, src_env_slice]
        dst.truncations[:, dst_env_indices] = src.truncations[:, src_env_slice]
        dst.values[:, dst_env_indices] = src.values[:, src_env_slice]
        dst.actions_log_prob[:, dst_env_indices] = src.actions_log_prob[:, src_env_slice]
        dst.mu[:, dst_env_indices] = src.mu[:, src_env_slice]
        dst.sigma[:, dst_env_indices] = src.sigma[:, src_env_slice]
        dst.returns[:, dst_env_indices] = src.returns[:, src_env_slice]
        dst.clean_returns[:, dst_env_indices] = src.clean_returns[:, src_env_slice]
        dst.advantages[:, dst_env_indices] = src.advantages[:, src_env_slice]

    def select_for_off_policy_storage(self) -> None:
        if self.off_policy_storage.max_filled_index < self.off_policy_storage.num_envs:
            # Buffer not yet full: fill new slots up to capacity.
            n_new = min(
                self.off_policy_storage.num_envs - self.off_policy_storage.max_filled_index,
                self.rollout_storage.num_envs,
            )
            dst_start = self.off_policy_storage.max_filled_index
            add_indices = torch.arange(dst_start, dst_start + n_new, device=self.off_policy_storage.device)
            self._copy_env_columns(
                src=self.rollout_storage,
                dst=self.off_policy_storage,
                src_env_slice=slice(0, n_new),
                dst_env_indices=add_indices,
            )
            self.off_policy_storage.filled_indices[add_indices] = 1
            self.off_policy_storage.max_filled_index = dst_start + n_new
            self.off_policy_add_index = (dst_start + n_new) % self.off_policy_storage.num_envs

        else:
            # Buffer full: overwrite circularly.
            # TODO: smarter eviction policy (e.g. prioritised replay) can be added here.
            add_indices = (
                torch.arange(self.rollout_storage.num_envs, device=self.off_policy_storage.device)
                + self.off_policy_add_index
            ) % self.off_policy_storage.num_envs
            self._copy_env_columns(
                src=self.rollout_storage,
                dst=self.off_policy_storage,
                src_env_slice=slice(None),  # all rollout envs
                dst_env_indices=add_indices,
            )
            self.off_policy_add_index = (
                self.off_policy_add_index + self.rollout_storage.num_envs
            ) % self.off_policy_storage.num_envs

    def clear(self) -> None:
        self.rollout_storage.clear()

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        # get's half a batch from on-policy data and half from off-policy data

        #compute minibatch size in on-policy storage
        on_policy_minibatch_size = self.rollout_storage.num_envs * self.rollout_storage.num_transitions_per_env // num_mini_batches
        # now compute how many minibatches we need to obtain from the off-policy storage to match the on-policy batch size
        num_off_policy_batches = self.off_policy_storage.max_filled_index * self.off_policy_storage.num_transitions_per_env // on_policy_minibatch_size

        on_policy_generator = self.rollout_storage.mini_batch_generator(num_mini_batches, num_epochs)
        off_policy_generator = self.off_policy_storage.mini_batch_generator(num_off_policy_batches, num_epochs)
        for on_policy_batch, off_policy_batch in zip(on_policy_generator, off_policy_generator):
            # merge batches
            obs_batch = torch.cat([on_policy_batch[0], off_policy_batch[0]], dim=0)
            actions_batch = torch.cat([on_policy_batch[1], off_policy_batch[1]], dim=0)
            target_values_batch = torch.cat([on_policy_batch[2], off_policy_batch[2]], dim=0)
            advantages_batch = torch.cat([on_policy_batch[3], off_policy_batch[3]],
                                        dim=0)
            returns_batch = torch.cat([on_policy_batch[4], off_policy_batch[4]], dim=0)
            clean_returns_batch = torch.cat([on_policy_batch[5], off_policy_batch[5]], dim=0)
            old_actions_log_prob_batch = torch.cat([on_policy_batch[6], off_policy_batch[6]], dim=0)
            old_mu_batch = torch.cat([on_policy_batch[7], off_policy_batch[7]], dim=0)
            old_sigma_batch = torch.cat([on_policy_batch[8], off_policy_batch[8]], dim=0)
            dones_batch = torch.cat([on_policy_batch[9], off_policy_batch[9]], dim=0)
            # Hidden states, masks and valid_mask are None for feedforward policies.
            # TODO: to support recurrent policies here, RolloutStorage.mini_batch_generator
            # (used by both ppo.py and reppo.py) must be updated to yield non-None hidden
            # states and masks, and the concatenation logic below must be updated accordingly.
            hidden_states_batch = (None, None)
            masks_batch = None
            valid_mask_batch = None
            yield (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                clean_returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                dones_batch,
                hidden_states_batch,
                masks_batch,
                valid_mask_batch,
                (torch.cat([
                    torch.ones(on_policy_batch[0].shape[0], dtype=torch.bool, device=obs_batch.device),
                    torch.zeros(off_policy_batch[0].shape[0], dtype=torch.bool, device=obs_batch.device)
                ]))
            )
