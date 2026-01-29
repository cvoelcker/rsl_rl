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

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent
from rsl_rl.modules import ActorQ, ActorQCNN, ActorQRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, Transition
from rsl_rl.utils import string_to_callable


class REPPO:
    """Relative Entropy Pathwise Policy Optimization algorithm ()."""

    policy: ActorQ | ActorQCNN | ActorQRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorQ | ActorQCNN | ActorQRecurrent,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        target_entropy: float = -1.0,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

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

        # Create the optimizer
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.95))

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
        self.learning_rate = learning_rate

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs, self.transition.actions).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
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
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.truncations = extras["time_outs"].to(self.device) * 1.0

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * self.transition.values * extras["time_outs"].to(self.device)
        self.transition.soft_rewards = (
            self.transition.rewards - self.gamma * self.policy.alpha_temp * self.transition.actions_log_prob
        )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        last_action = self.policy.act(obs).detach()
        last_values = self.policy.evaluate(obs, last_action).detach().view(-1, 1)
        # Compute returns and advantages
        recurr_value = last_values
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta_1 = next_is_not_terminal * self.gamma * next_values
            delta_n = next_is_not_terminal * self.gamma * recurr_value
            recurr_value = st.soft_rewards[step] + (1 - self.lam) * delta_1 + self.lam * delta_n
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = recurr_value

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

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
            truncations_batch,
            _,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in enumerate(generator):

            # Build minibatch dict
            minibatch = {
                "obs_batch": obs_batch,
                "actions_batch": actions_batch,
                "returns_batch": returns_batch,
                "truncations_batch": truncations_batch,
                "hidden_states_batch": hidden_states_batch,
                "masks_batch": masks_batch,
                "old_mean": old_mean_batch,
                "old_std": old_std_batch,
            }

            # Update critic
            critic_metrics = self.update_critic(minibatch)

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
            truncations_batch,
            _,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) in enumerate(generator):

            # Build minibatch dict
            minibatch = {
                "obs_batch": obs_batch,
                "actions_batch": actions_batch,
                "returns_batch": returns_batch,
                "truncations_batch": truncations_batch,
                "hidden_states_batch": hidden_states_batch,
                "masks_batch": masks_batch,
                "old_mean": old_mean_batch,
                "old_std": old_std_batch,
            }
            
            # Update actor
            actor_metrics = self.update_actor(minibatch)

            # Store the losses
            mean_value_loss += critic_metrics["value_loss"]
            mean_entropy += actor_metrics["entropy"]
            mean_surrogate_loss += actor_metrics["actor_loss"]

        print("value prediction error: ", critic_metrics["value_prediction_error"])
        print("enc dec error: ", critic_metrics["enc_dec_error"])
        print("on policy values mean: ", actor_metrics["on_policy_values_mean"])
        print("entropy: ", actor_metrics["entropy"])
        print("kl divergence: ", actor_metrics["kl_divergence"])
        print("entropy target: ", self.target_entropy)
        print("alpha temp: ", self.policy.alpha_temp.item())
        print("alpha kl: ", self.policy.alpha_kl.item())

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict
    
    def update_actor(self, minibatch: dict) -> dict:
        """Update the actor network.
        
        Args:
            minibatch: Dictionary containing the minibatch data with keys:
                - obs_batch: Observations
                - actions_batch: Actions
                - returns_batch: Returns
                - truncations_batch: Truncation flags
                - hidden_states_batch: Hidden states (for recurrent policies)
                - masks_batch: Masks (for recurrent policies)
        
        Returns:
            Dictionary containing actor metrics.
        """
        obs_batch = minibatch["obs_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        # Get current actor outputs
        self.policy.act(obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        predicted_actions = predicted_policy.rsample()
        on_policy_values = self.policy.evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)

        # Temperature component
        entropy = -predicted_policy.log_prob(predicted_actions).sum(-1)
        entropy_loss = self.policy.alpha_temp.detach() * entropy
        primary_policy_loss = -(on_policy_values + entropy_loss)

        # KL computation
        self.old_policy._update_distribution(obs_batch, minibatch["old_mean"], minibatch["old_std"])
        old_policy_distribution = self.old_policy.distribution
        old_policy_actions = old_policy_distribution.sample((1,))
        log_prob_old = old_policy_distribution.log_prob(old_policy_actions).detach()
        log_prob_new = predicted_policy.log_prob(old_policy_actions)
        kl_divergence = (log_prob_old - log_prob_new).sum(-1).mean(0)

        # Clipped Policy Loss
        policy_loss = torch.where(
            (kl_divergence < self.desired_kl).detach(),
            primary_policy_loss,
            self.policy.alpha_kl.detach() * kl_divergence,
        ).mean()

        # Temperature updates
        temp_target_loss = self.policy.alpha_temp * (entropy.mean() - self.target_entropy).detach()
        kl_target_loss = self.policy.alpha_kl * (self.desired_kl - kl_divergence.mean()).detach()

        # Freeze critic parameters to prevent policy loss from updating them
        self._set_critic_grad(False)
        self.optimizer.zero_grad()
        actor_loss = policy_loss + temp_target_loss + kl_target_loss
        actor_loss.backward()
        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Unfreeze critic parameters
        self._set_critic_grad(True)

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "kl_divergence": kl_divergence.mean().item(),
            "on_policy_values_mean": on_policy_values.mean().item(),
        }

    def update_critic(self, minibatch: dict) -> dict:
        """Update the critic network.
        
        Args:
            minibatch: Dictionary containing the minibatch data with keys:
                - obs_batch: Observations
                - actions_batch: Actions
                - returns_batch: Returns
                - truncations_batch: Truncation flags
                - hidden_states_batch: Hidden states (for recurrent policies)
                - masks_batch: Masks (for recurrent policies)
        
        Returns:
            Dictionary containing critic metrics.
        """
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        returns_batch = minibatch["returns_batch"]
        truncations_batch = minibatch["truncations_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        # Get current critic outputs
        value_prediction, value_logits = self.policy.evaluate(
            obs_batch, actions_batch, hidden_states_batch, masks_batch, return_logits=True
        )

        # VALUE LOSS
        # Embed the returns
        embedded_returns = self.policy.hlgauss_embed(returns_batch.view(-1)).view(value_logits.shape).detach()

        # Compute loss
        value_loss = -(
            (1.0 - truncations_batch.view(-1))
            * (embedded_returns * torch.log_softmax(value_logits, dim=-1)).sum(-1)
        ).mean()

        # Compute the gradients
        self.optimizer.zero_grad()
        value_loss.backward()
        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Compute metrics for logging
        value_prediction_error = (value_prediction.view(-1) - returns_batch.view(-1)).abs().mean().item()
        decoded_returns = self.policy.hlgauss_decode(torch.log(embedded_returns)).detach()
        enc_dec_error = (decoded_returns.view(-1) - returns_batch.view(-1)).abs().mean().item()

        return {
            "value_loss": value_loss.item(),
            "value_prediction_error": value_prediction_error,
            "enc_dec_error": enc_dec_error,
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
