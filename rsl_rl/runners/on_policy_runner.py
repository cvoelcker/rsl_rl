# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from torch import nn
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import PPO, REPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    ActorQ,
    ActorQCNN,
    ActorQRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.policy_export import export_policy_as_torchscript, save_policy_checkpoint
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        # Resolve action scaling configuration
        self.action_scale_cfg = self._resolve_action_scale_config()

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self.init_at_random_ep_len = self.cfg.get("init_at_random_ep_len", False)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.randomize_num_steps(self.env.max_episode_steps)  # randomize episode lengths after evaluation for better exploration

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        it = 0
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            for _ in range(self.cfg["num_steps_per_env"]):
                # Sample actions
                with torch.inference_mode():
                    actions = self.alg.act(obs)
                    # scale actions
                    if self.action_scale_cfg["scale_actions"] and self.action_scale_cfg["method"] == "runner":
                        actions = self._scale_actions(actions)

                # Step the environment
                obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                rewards = rewards.squeeze(-1)
                dones = dones.squeeze(-1)
                extras["time_outs"] = extras["time_outs"].squeeze(-1)
                # Move to device
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                # Process the step
                self.alg.process_env_step(obs, rewards, dones, extras)
                # Extract intrinsic rewards (only for logging)
                intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                # Book keeping
                self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

            stop = time.time()
            collect_time = stop - start
            start = stop

            # Compute returns
            self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
            
            if it % self.cfg["eval_interval"] == 0:
                self.eval(it)
                obs = self.env.get_observations().to(self.device)

        # Save the final model after training
        self.eval(it)
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            # "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def save_policy(self, path: str, infos: dict | None = None) -> None:
        """Save a minimal checkpoint containing only the trained policy (for inference)."""

        metadata = {
            "iter": self.current_learning_iteration,
            "infos": infos,
            "obs_groups": self.cfg.get("obs_groups"),
            "num_actions": getattr(self.env, "num_actions", None),
            "policy_class": type(self.alg.policy).__name__,
            "policy_module": type(self.alg.policy).__module__,
        }
        save_policy_checkpoint(path, model_state_dict=self.alg.policy.state_dict(), metadata=metadata)
        self.logger.save_model(path, self.current_learning_iteration)

    def eval(self, it: int):
        """Run evaluation rollouts with the current policy."""
        self.eval_mode()  # switch to eval mode (for dropout for example)
        obs = self.env.reset().to(self.device)
        done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        returns = torch.zeros(self.env.num_envs, device=self.device)
        lengths = torch.zeros(self.env.num_envs, device=self.device)
        while not done.all():
            with torch.inference_mode():
                actions = self.alg.policy.act_inference(obs)
            if self.action_scale_cfg["scale_actions"] and self.action_scale_cfg["method"] == "runner":
                actions = self._scale_actions(actions)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
            done = done | dones.squeeze(-1)
            # book keeping
            returns += rewards.squeeze(-1) * (1.0 - done)  # only add rewards for environments that are not done
            lengths += (1.0 - done)  # only add to length for environments that are not done
        self.logger.log_eval(it, {"eval_return": returns.mean().item(), "eval_length": lengths.mean().item()})
        if self.init_at_random_ep_len:
            self.env.reset()  # reset the env after evaluation
            self.env.randomize_num_steps(self.env.max_episode_steps)  # randomize episode lengths after evaluation for better exploration
            self.logger.update_episode_lengths(self.env.episode_length_buf)

    def export_policy_jit(self, path: str, *, device: str = "cpu") -> None:
        """Export the policy as a TorchScript module (flat actor obs -> actions).

        See `rsl_rl.utils.policy_export.export_policy_as_torchscript` for current limitations.
        """

        self.eval_mode()
        obs = self.env.get_observations().to(self.device)
        export_policy_as_torchscript(self.alg.policy, example_obs=obs, out_path=path, device=device)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        return default_sets

    def _resolve_action_scale_config(self) -> dict[str, str | bool]:
        scale_actions = bool(self.alg_cfg.get("scale_actions", False))
        method = self.alg_cfg.get("action_scale_method")
        bounds = self.alg_cfg.get("action_scale_bounds")

        # Legacy compatibility
        if method is None or bounds is None:
            if self.alg_cfg.get("use_env_action_bounds_in_runner", False):
                method = "runner"
                bounds = "env"
            elif self.alg_cfg.get("use_env_action_bounds", False):
                method = "policy"
                bounds = "env"
            else:
                method = method or "runner"
                bounds = bounds or "provided"

        if method not in ("runner", "policy"):
            raise ValueError(f"Unknown action_scale_method '{method}'. Expected 'runner' or 'policy'.")
        if bounds not in ("provided", "env"):
            raise ValueError(f"Unknown action_scale_bounds '{bounds}'. Expected 'provided' or 'env'.")

        self.alg_cfg["action_scale_method"] = method
        self.alg_cfg["action_scale_bounds"] = bounds

        return {"scale_actions": scale_actions, "method": method, "bounds": bounds}

    def _resolve_action_bounds(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if self.action_scale_cfg["bounds"] == "env":
            if not (hasattr(self.env, "action_low") and hasattr(self.env, "action_high")):
                raise ValueError("Environment does not provide action_low/action_high for action scaling.")
            return self.env.action_low, self.env.action_high

        lower = self.alg_cfg.get("action_lower_bound", -1.0)
        upper = self.alg_cfg.get("action_upper_bound", 1.0)
        return lower, upper

    def _to_bound_tensor(
        self, value: torch.Tensor | float | list, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            tensor_value = torch.tensor(value, device=device, dtype=dtype)
            return torch.sign(tensor_value) * torch.ceil(tensor_value.abs())
        return torch.tensor(value, device=device, dtype=dtype)

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        lower, upper = self._resolve_action_bounds()
        lower_t = self._to_bound_tensor(lower, actions.device, actions.dtype)
        upper_t = self._to_bound_tensor(upper, actions.device, actions.dtype)
        return actions * (upper_t - lower_t) / 2.0 + (upper_t + lower_t) / 2.0

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPO | REPPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Propagate action bounds to policy when scaling is handled by the policy
        if self.action_scale_cfg["scale_actions"] and self.action_scale_cfg["method"] == "policy":
            action_low, action_high = self._resolve_action_bounds()
            self.policy_cfg["action_low"] = action_low
            self.policy_cfg["action_high"] = action_high

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: nn.Module = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO | REPPO = alg_class(
            actor_critic, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
