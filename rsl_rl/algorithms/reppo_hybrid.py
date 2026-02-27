# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticRecurrent, ActorQ, ActorQCNN, ActorQRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, Transition
from rsl_rl.storage.rollout_storage import HybridRolloutStorage
from rsl_rl.utils import string_to_callable

from time import time


class REPPO:
    """Relative Entropy Pathwise Policy Optimization algorithm ()."""

    policy: ActorQ | ActorQCNN | ActorQRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorQ | ActorQCNN | ActorQRecurrent,
        storage: HybridRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 0.001,
        temp_learning_rate: float | None = None,
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        target_entropy: float = -1.0,
        schedule: str = "adaptive",
        run_td3: bool=False,
        device: str = "cpu",
        num_kl_samples: int = 4,
        use_relative_kl: bool = False,
        reverse_kl: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        actor_lr_multiplier: float = 1.0,
        max_inf_batch_size: int = 81920,
        **kwargs,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.reverse_kl = reverse_kl
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # old policy (for KL computation)
        self.old_policy = copy.deepcopy(self.policy)
        self.old_policy.to(self.device)
        self.old_policy.eval()

        # Create separate optimizers for actor, temperature, and critic
        actor_params = list(self.policy.actor.parameters())
        if hasattr(self.policy, "std"):
            actor_params.append(self.policy.std)
        if hasattr(self.policy, "log_std"):
            actor_params.append(self.policy.log_std)
        temp_params = [self.policy.log_alpha_temp, self.policy.log_alpha_kl]
        critic_params = (
            list(self.policy.critic.parameters())
            + list(self.policy.critic_embedding_layer.parameters())
            + list(self.policy.norm.parameters())
        )
        self.actor_optimizer = optim.AdamW(actor_params, lr=learning_rate * actor_lr_multiplier, weight_decay=1e-3)
        if temp_learning_rate is None:
            temp_learning_rate = learning_rate
        self.temp_optimizer = optim.Adam(temp_params, lr=temp_learning_rate)
        self.critic_optimizer = optim.Adam(critic_params, lr=learning_rate)

        # Add storage
        self.storage = storage
        self.transition = Transition()
        
        # REPPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.target_entropy = target_entropy * self.policy.num_actions
        if use_relative_kl:
            self.target_entropy = target_entropy
        self.learning_rate = learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.schedule = schedule
        self.run_td3 = run_td3
        self.num_kl_samples = num_kl_samples

        self.use_relative_kl = use_relative_kl
        self.max_inf_batch_size = max_inf_batch_size

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs, self.transition.actions).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = extras.get("terminations", torch.zeros_like(dones)).to(self.device)
        # Store truncations (time_outs) separately for proper handling in compute_returns
        # Note: We do NOT bootstrap here because obs is post-reset and causally unrelated
        # to the truncated transition. Truncation bootstrap is handled in compute_returns.
        self.transition.truncations = extras.get("time_outs", torch.zeros_like(dones)).to(self.device)

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
 
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def process_buffer_transitions(self) -> None:
        # compute effective batches from rollout length
        self.storage.select_for_off_policy_storage()
        with torch.no_grad():
            transition_batch_size = self.max_inf_batch_size // self.storage.num_transitions_per_env
            num_op_transition_batches = self.storage.rollout_storage.num_envs // transition_batch_size
            num_offp_transition_batches = self.storage.off_policy_storage.max_filled_index // transition_batch_size

            for op_batch_idx in range(num_op_transition_batches):
                obs = self.storage.rollout_storage.observations[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size]
                rewards = self.storage.rollout_storage.rewards[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size]
                dones = self.storage.rollout_storage.dones[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size]
                truncations = self.storage.rollout_storage.truncations[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size]

                self.policy.act(obs)
                op_actions = self.policy.distribution.sample().detach()
                # evaluate/log_prob return [T, B]; unsqueeze to [T, B, 1] to match storage shape
                op_log_prob = self.policy.distribution.log_prob(op_actions).sum(-1).unsqueeze(-1).detach()
                returns = self.policy.evaluate(obs, op_actions).unsqueeze(-1).detach()
                rewards = rewards + truncations * self.gamma * returns
                soft_rewards = rewards - (1.0 - dones) * self.policy.alpha_temp.detach() * op_log_prob
                self.storage.rollout_storage.soft_rewards[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size] = soft_rewards
                self.storage.rollout_storage.values[:, op_batch_idx * transition_batch_size:(op_batch_idx + 1) * transition_batch_size] = returns

            num_offp_filled = self.storage.off_policy_storage.max_filled_index
            num_offp_transition_batches = (num_offp_filled + transition_batch_size - 1) // transition_batch_size
            for offp_batch_idx in range(num_offp_transition_batches):
                offp_obs = self.storage.off_policy_storage.observations[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size]
                offp_rewards = self.storage.off_policy_storage.rewards[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size]
                offp_dones = self.storage.off_policy_storage.dones[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size]
                offp_truncations = self.storage.off_policy_storage.truncations[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size]
                offp_actions = self.storage.off_policy_storage.actions[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size]

                self.policy.act(offp_obs)
                offp_actions = self.policy.distribution.sample().detach()
                offp_log_prob = self.policy.distribution.log_prob(offp_actions).sum(-1).unsqueeze(-1).detach()
                offp_returns = self.policy.evaluate(offp_obs, offp_actions).unsqueeze(-1).detach()
                offp_rewards = offp_rewards + offp_truncations * self.gamma * offp_returns
                offp_soft_rewards = offp_rewards - (1.0 - offp_dones) * self.policy.alpha_temp.detach() * offp_log_prob
                self.storage.off_policy_storage.soft_rewards[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size] = offp_soft_rewards
                self.storage.off_policy_storage.values[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size] = offp_returns
                curr_log_prob = self.policy.distribution.log_prob(offp_actions).sum(-1, keepdim=True).detach()
                self.storage.off_policy_storage.curr_log_prob[:, offp_batch_idx * transition_batch_size:(offp_batch_idx + 1) * transition_batch_size] = curr_log_prob

            self.storage.rollout_storage.truncations[-1] = 1.0
            self.storage.off_policy_storage.truncations[-1] = 1.0


    def compute_returns(self, obs: TensorDict) -> None:
        self.process_buffer_transitions()
        # First iterate over the on-policy transitions
        st = self.storage.rollout_storage
        last_values = st.values[-1]
        recurr_value = 0.0

        td_target = last_values
        for step in reversed(range(st.num_transitions_per_env-1)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # True terminals should zero out the bootstrap.
            is_truncated = st.truncations[step].bool()
            next_is_not_terminal = 1.0 - st.dones[step]
            lambda_sum = (1.0 - self.lam) * next_values + self.lam * td_target
            if self.run_td3:
                td_target = st.rewards[step] + next_is_not_terminal * self.gamma * lambda_sum
            else:
                td_target = st.soft_rewards[step] + next_is_not_terminal * self.gamma * lambda_sum
            st.returns[step] = td_target
            td_target = torch.where(is_truncated, st.values[step],  td_target)
            recurr_value = next_is_not_terminal * (self.gamma * recurr_value + st.rewards[step])
            st.clean_returns[step] = recurr_value

        # Now iterate over the off-policy transitions
        st = self.storage.off_policy_storage
        last_values = st.values[-1]
        recurr_value = 0.0

        td_target = last_values
        for step in reversed(range(st.num_transitions_per_env-1)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # True terminals should zero out the bootstrap.
            is_truncated = st.truncations[step].bool()
            next_is_not_terminal = 1.0 - st.dones[step]
            offp_correction = torch.clamp(torch.exp(st.curr_log_prob[step] - st.actions_log_prob[step]), min=0.0, max=1.0)
            lam = self.lam * offp_correction
            lambda_sum = (1.0 - lam) * next_values + lam * td_target
            if self.run_td3:
                td_target = st.rewards[step] + next_is_not_terminal * self.gamma * lambda_sum
            else:
                td_target = st.soft_rewards[step] + next_is_not_terminal * self.gamma * lambda_sum
            st.returns[step] = td_target
            td_target = torch.where(is_truncated, st.values[step],  td_target)
            recurr_value = next_is_not_terminal * (self.gamma * recurr_value + st.rewards[step])
            st.clean_returns[step] = recurr_value

    def _compute_storage_metrics(self, st) -> dict:
        """Compute diagnostic metrics from a RolloutStorage instance."""
        is_true_terminal = st.dones.bool() & ~st.truncations.bool()
        is_orphan_truncation = st.truncations.bool() & ~st.dones.bool()
        is_double_terminal = st.dones.bool() & st.truncations.bool()
        return {
            "returns_mean": st.returns.mean().item(),
            "returns_std": st.returns.std().item(),
            "returns_min": st.returns.min().item(),
            "returns_max": st.returns.max().item(),
            "clean_returns_mean": st.clean_returns[0].mean().item(),
            "clean_returns_std": st.clean_returns[0].std().item(),
            "clean_returns_min": st.clean_returns[0].min().item(),
            "clean_returns_max": st.clean_returns[0].max().item(),
            "rewards_mean": st.rewards.mean().item(),
            "rewards_std": st.rewards.std().item(),
            "soft_rewards_mean": st.soft_rewards.mean().item(),
            "num_true_terminations": is_true_terminal.sum().item(),
            "num_truncations": st.truncations.sum().item(),
            "num_orphan_truncations": is_orphan_truncation.sum().item(),
            "num_double_terminations": is_double_terminal.sum().item(),
            "action_min": st.actions.min().item(),
            "action_max": st.actions.max().item(),
            "action_mean": st.actions.mean().item(),
        }

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_kl_divergence = 0
        mean_kl_exceed_frac = 0
        mean_kl_std = 0
        mean_kl_max = 0
        mean_q_value_std = 0
        mean_actor_grad_norm = 0
        mean_critic_grad_norm = 0
        mean_action_std_mean = 0
        mean_action_std_min = 0
        mean_action_std_max = 0
        mean_action_mean_abs = 0
        mean_primary_policy_loss = 0
        mean_value_pred_error = 0
        mean_value_pred_std = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Compute per-storage diagnostics before the update
        op_m = self._compute_storage_metrics(self.storage.rollout_storage)
        offp_m = self._compute_storage_metrics(self.storage.off_policy_storage)

        # checkpoint critic params
        critic_params = copy.deepcopy((
            list(self.policy.critic.parameters())
            + list(self.policy.critic_embedding_layer.parameters())
            + list(self.policy.norm.parameters())
        ))

        with torch.no_grad():
            self.old_policy.load_state_dict(self.policy.state_dict())

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for i, (
            obs_batch,
            actions_batch,
            _,
            _,
            returns_batch,
            _,
            _,
            _,
            _,
            dones_batch,
            hidden_states_batch,
            masks_batch,
            valid_mask,
            off_policy_marker,
        ) in enumerate(generator):
            # Build minibatch dict
            minibatch = {
                "obs_batch": obs_batch,
                "actions_batch": actions_batch,
                "returns_batch": returns_batch,
                "hidden_states_batch": hidden_states_batch,
                "masks_batch": masks_batch,
                "valid_mask": valid_mask,
                "dones_batch": dones_batch,
                "off_policy_marker": off_policy_marker,
            }

            # Update critic
            critic_metrics = self.update_critic(minibatch)

            # Update actor
            # if i >= self.num_learning_epochs * self.num_mini_batches // 2:
            actor_metrics = self.update_actor(minibatch)
            mean_entropy += actor_metrics["entropy"]
            mean_surrogate_loss += actor_metrics["actor_loss"]
            mean_kl_divergence += actor_metrics["kl_divergence"]
            mean_kl_exceed_frac += actor_metrics["kl_exceed_frac"]
            mean_kl_std += actor_metrics["kl_std"]
            mean_kl_max += actor_metrics["kl_max"]
            mean_q_value_std += actor_metrics["on_policy_values_std"]
            mean_actor_grad_norm += actor_metrics["actor_grad_norm"]
            mean_action_std_mean += actor_metrics["action_std_mean"]
            mean_action_std_min += actor_metrics["action_std_min"]
            mean_action_std_max += actor_metrics["action_std_max"]
            mean_action_mean_abs += actor_metrics["action_mean_abs"]
            mean_primary_policy_loss += actor_metrics["primary_policy_loss"]

            mean_value_loss += critic_metrics["value_loss"]
            mean_value_pred_error += critic_metrics["value_prediction_error"]
            mean_value_pred_std += critic_metrics["value_pred_std"]
            mean_critic_grad_norm += critic_metrics["critic_grad_norm"]
            

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_kl_divergence /= num_updates
        mean_kl_exceed_frac /= num_updates
        mean_kl_std /= num_updates
        mean_kl_max /= num_updates
        mean_q_value_std /= num_updates
        mean_actor_grad_norm /= num_updates
        mean_critic_grad_norm /= num_updates
        mean_action_std_mean /= num_updates
        mean_action_std_min /= num_updates
        mean_action_std_max /= num_updates
        mean_action_mean_abs /= num_updates
        mean_primary_policy_loss /= num_updates
        mean_value_pred_error /= num_updates
        mean_value_pred_std /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            # Core losses
            "loss/value": mean_value_loss,
            "loss/surrogate": mean_surrogate_loss,
            "loss/primary_policy": mean_primary_policy_loss,

            # Entropy/KL metrics
            "policy/entropy": mean_entropy,
            "policy/kl_divergence": mean_kl_divergence,
            "policy/kl_exceed_frac": mean_kl_exceed_frac,
            "policy/kl_std": mean_kl_std,
            "policy/kl_max": mean_kl_max,
            "policy/alpha_temp": self.policy.alpha_temp.item(),
            "policy/alpha_kl": self.policy.alpha_kl.item(),

            # On-policy value/return statistics
            "value/returns_mean": op_m["returns_mean"],
            "value/returns_std": op_m["returns_std"],
            "value/returns_min": op_m["returns_min"],
            "value/returns_max": op_m["returns_max"],
            "value/clean_returns_mean": op_m["clean_returns_mean"],
            "value/clean_returns_std": op_m["clean_returns_std"],
            "value/clean_returns_min": op_m["clean_returns_min"],
            "value/clean_returns_max": op_m["clean_returns_max"],
            "value/q_std": mean_q_value_std,
            "value/pred_error": mean_value_pred_error,
            "value/pred_std": mean_value_pred_std,
            "optim/learning_rate": self.learning_rate,

            # Critic stats
            "value/critic_enc_dec_error": critic_metrics["enc_dec_error"],
            "value/critic_q_min": critic_metrics["min_q"],
            "value/critic_q_max": critic_metrics["max_q"],
            "value/critic_q_mean": critic_metrics["mean_q"],
            "value/critic_pred_error": critic_metrics["value_prediction_error"],

            # On-policy reward statistics
            "env/rewards_mean": op_m["rewards_mean"],
            "env/rewards_std": op_m["rewards_std"],
            "env/soft_rewards_mean": op_m["soft_rewards_mean"],

            # On-policy termination statistics
            "env/true_terminations": op_m["num_true_terminations"],
            "env/truncations": op_m["num_truncations"],
            "env/orphan_truncations": op_m["num_orphan_truncations"],
            "env/double_terminations": op_m["num_double_terminations"],

            # Off-policy statistics
            "off_policy/returns_mean": offp_m["returns_mean"],
            "off_policy/rewards_mean": offp_m["rewards_mean"],
            "off_policy/soft_rewards_mean": offp_m["soft_rewards_mean"],
            "off_policy/true_terminations": offp_m["num_true_terminations"],
            "off_policy/truncations": offp_m["num_truncations"],

            # Gradient statistics
            "optim/actor_grad_norm": mean_actor_grad_norm,
            "optim/critic_grad_norm": mean_critic_grad_norm,

            # Obs stats
            "env/obs_min": critic_metrics["min_obs"],
            "env/obs_max": critic_metrics["max_obs"],
            "env/obs_norm_min": critic_metrics["min_norm_obs"],
            "env/obs_norm_max": critic_metrics["max_norm_obs"],

            # Policy distribution statistics
            "policy/action_std_mean": mean_action_std_mean,
            "policy/action_std_min": mean_action_std_min,
            "policy/action_std_max": mean_action_std_max,
            "policy/action_mean_abs": mean_action_mean_abs,
            "policy/action_min": op_m["action_min"],
            "policy/action_max": op_m["action_max"],
            "policy/action_mean": op_m["action_mean"],
        }
        if self.rnd:
            loss_dict["aux/rnd_loss"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["aux/symmetry_loss"] = mean_symmetry_loss

        self.storage.clear()

        # get updated critic params
        updated_critic_params = (
            list(self.policy.critic.parameters())
            + list(self.policy.critic_embedding_layer.parameters())
            + list(self.policy.norm.parameters())
        )
        # merge param dicts with ema
        for p, op in zip(updated_critic_params, critic_params):
            fast_polyak(op, p, tau=0.0)

        return loss_dict

    def update_actor(self, minibatch: dict) -> dict:
        off_policy_mask = minibatch["off_policy_marker"]
        obs_batch = minibatch["obs_batch"][off_policy_mask.bool()]
        hidden_states_batch = minibatch["hidden_states_batch"][off_policy_mask.bool()]
        masks_batch = minibatch["masks_batch"][off_policy_mask.bool()]
        valid_mask = minibatch["valid_mask"][off_policy_mask.bool()]

        self.policy.act(obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        predicted_actions = predicted_policy.rsample()
        on_policy_values = self.policy.evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)

        entropy = -predicted_policy.log_prob(predicted_actions).sum(-1)
        entropy_loss = self.policy.alpha_temp.detach() * entropy
        primary_policy_loss = -(on_policy_values + entropy_loss).squeeze()

        # Compute KL loss
        if self.reverse_kl:
            log_prob_new = -entropy
            self.old_policy.act(obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
            log_prob_old = old_policy_distribution.log_prob(predicted_actions).sum(-1)
            kl_divergence = (log_prob_new - log_prob_old)
            
        else:
            with torch.no_grad():
                self.old_policy.act(obs_batch, hidden_states_batch, masks_batch)
                old_policy_distribution = self.old_policy.distribution
                old_policy_actions = old_policy_distribution.sample((self.num_kl_samples,))
                log_prob_old = old_policy_distribution.log_prob(old_policy_actions)
            log_prob_new = predicted_policy.log_prob(old_policy_actions)
            kl_divergence = (log_prob_old - log_prob_new).sum(-1).mean(0)
        kl_loss = self.policy.alpha_kl.detach() * kl_divergence

        constraint_ok = (kl_divergence < self.desired_kl)
        policy_loss = torch.where(
            constraint_ok.detach(),
            primary_policy_loss,
            kl_loss - entropy_loss,
        )
        # policy_loss = primary_policy_loss.mean() + kl_loss.mean()
        policy_loss = policy_loss.mean()
        entropy_mean = entropy.mean()
        kl_mean = kl_divergence.mean()

        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl_mean_reduced = kl_mean
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean_reduced, op=torch.distributed.ReduceOp.SUM)
                    kl_mean_reduced /= self.gpu_world_size

                target_kl = 0.5 * self.desired_kl
                if kl_mean_reduced > target_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean_reduced < target_kl / 2.0 and kl_mean_reduced > 0.0:
                    self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        temp_target_loss = self.policy.alpha_temp * (entropy_mean - self.target_entropy).detach().mean()
        if self.run_td3:
            temp_target_loss = 0.0
        kl_target_loss = self.policy.alpha_kl * (self.desired_kl - kl_mean).detach().mean()

        self.actor_optimizer.zero_grad()
        self.temp_optimizer.zero_grad()
        actor_loss = policy_loss + temp_target_loss + kl_target_loss
        actor_loss.backward()

        # Compute gradient statistics before clipping
        actor_grad_norm = 0.0
        for p in self.policy.actor.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.data.norm(2).item() ** 2
        actor_grad_norm = actor_grad_norm ** 0.5

        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.temp_optimizer.step()

        # Compute diagnostics
        kl_exceeds_threshold = (kl_divergence >= self.desired_kl).float()
        kl_exceed_frac = kl_exceeds_threshold.mean()
        q_value_std = on_policy_values.std()
        q_value_mean = on_policy_values.view(-1).mean()
        q_value_min = on_policy_values.min()
        q_value_max = on_policy_values.max()
        kl_std = kl_divergence.std().item()
        kl_max = kl_divergence.max().item()
        # Policy distribution statistics
        action_std_mean = self.policy.action_std.mean().item()
        action_std_min = self.policy.action_std.min().item()
        action_std_max = self.policy.action_std.max().item()
        action_mean_abs = self.policy.action_mean.abs().mean().item()

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy_mean.item() if valid_mask is not None else entropy.mean().item(),
            "kl_divergence": kl_mean.item() if valid_mask is not None else kl_divergence.mean().item(),
            "kl_exceed_frac": kl_exceed_frac.item(),
            "kl_std": kl_std,
            "kl_max": kl_max,
            "alpha_temp": self.policy.alpha_temp.item(),
            "alpha_kl": self.policy.alpha_kl.item(),
            "on_policy_values_mean": on_policy_values.mean().item(),
            "on_policy_values_std": q_value_std.item(),
            "actor_grad_norm": actor_grad_norm,
            "action_std_mean": action_std_mean,
            "action_std_min": action_std_min,
            "action_std_max": action_std_max,
            "action_mean_abs": action_mean_abs,
            "primary_policy_loss": primary_policy_loss.mean().item(),
            "q_value_mean": q_value_mean.item(),
            "q_value_min": q_value_min.item(),
            "q_value_max": q_value_max.item(),
        }

    def update_critic(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        returns_batch = minibatch["returns_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        value_prediction, value_logits = self.policy.evaluate(
            obs_batch, actions_batch, hidden_states_batch, masks_batch, return_logits=True
        )

        embedded_returns = self.policy.hlgauss_embed(returns_batch.view(-1)).view(value_logits.shape).detach()
        per_sample_loss = -((embedded_returns * torch.log_softmax(value_logits, dim=-1)).sum(-1))
        
        value_loss = per_sample_loss.mean()

        self.critic_optimizer.zero_grad()
        value_loss.backward()

        # Compute gradient statistics before clipping
        critic_grad_norm = 0.0
        for p in self.policy.critic.parameters():
            if p.grad is not None:
                critic_grad_norm += p.grad.data.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5

        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        value_prediction_error = ((value_prediction.view(-1) - returns_batch.view(-1)) ** 2).mean().item()
        decoded_returns = self.policy.hlgauss_decode(torch.log(embedded_returns)).detach()
        enc_dec_error = (decoded_returns.view(-1) - returns_batch.view(-1)).abs().mean().item()

        # Additional critic diagnostics
        value_pred_std = value_prediction.std().item()
        returns_batch_std = returns_batch.std().item()

        # obs min and max
        obs = self.policy.get_critic_obs(obs_batch)
        min_obs = obs.min()
        max_obs = obs.max()
        min_norm_obs = self.policy.critic_obs_normalizer(obs).min()
        max_norm_obs = self.policy.critic_obs_normalizer(obs).max()

        return {
            "value_loss": value_loss.item(),
            "value_prediction_error": value_prediction_error,
            "enc_dec_error": enc_dec_error,
            "critic_grad_norm": critic_grad_norm,
            "value_pred_std": value_pred_std,
            "returns_batch_std": returns_batch_std,
            "min_obs": min_obs,
            "max_obs": max_obs,
            "min_norm_obs": min_norm_obs,
            "max_norm_obs": max_norm_obs,
            "min_q": value_prediction.min().item(),
            "max_q": value_prediction.max().item(),
            "mean_q": value_prediction.mean().item(),
        }

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def _set_critic_grad(self, requires_grad: bool) -> None:
        """Enable or disable gradient computation for critic parameters.

        This is used to prevent the policy loss from updating critic parameters
        while still allowing gradients to flow through the critic to the actor.

        Args:
            requires_grad: Whether to enable gradient computation for critic parameters.
        """
        for param in self.policy.critic.parameters():
            param.requires_grad = requires_grad
        for param in self.policy.critic_embedding_layer.parameters():
            param.requires_grad = requires_grad
        for param in self.policy.norm.parameters():
            param.requires_grad = requires_grad
        if self.policy.is_recurrent:
            for param in self.policy.memory_c.parameters():
                param.requires_grad = requires_grad

def fast_polyak(critic1, critic2, tau):
    for param, target_param in zip(critic1, critic2):
        target_param.data.copy_((1-tau)*target_param.data + tau*param.data)