# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, TanhNormal
from rsl_rl.utils.hl_gauss import HLGaussLayer, embed_targets

from .actor_q import ActorQ


class ActorQCNN(ActorQ):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        num_critic_bins: int = 201,
        vmin: float = -10.0,
        vmax: float = 10.0,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        distribution_type: str = "normal",
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("ActorQCNN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs]))
        super(ActorQ, self).__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_2d = []
        actor_in_channels_2d = []
        self.actor_obs_groups_2d = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.actor_obs_groups_2d.append(obs_group)
                actor_in_dims_2d.append(obs[obs_group].shape[2:4])
                actor_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
        num_critic_obs_1d = num_actions
        self.critic_obs_groups_1d = []
        critic_in_dims_2d = []
        critic_in_channels_2d = []
        self.critic_obs_groups_2d = []
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.critic_obs_groups_2d.append(obs_group)
                critic_in_dims_2d.append(obs[obs_group].shape[2:4])
                critic_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        # Assert that there are 2D observations
        assert self.actor_obs_groups_2d or self.critic_obs_groups_2d, (
            "No 2D observations are provided. If this is intentional, use the ActorQ module instead."
        )

        # save activation
        self.activation = activation
        if self.activation == "selu":
            self.activation_fn = F.selu
        elif self.activation == "relu":
            self.activation_fn = F.relu
        elif self.activation == "elu":
            self.activation_fn = F.elu
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # Actor CNN
        if self.actor_obs_groups_2d:
            # Resolve the actor CNN configuration
            assert actor_cnn_cfg is not None, "An actor CNN configuration is required for 2D actor observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {group: actor_cnn_cfg for group in self.actor_obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D actor observations."
            )

            # Create CNNs for each 2D actor observation
            self.actor_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.actor_obs_groups_2d):
                self.actor_cnns[obs_group] = CNN(
                    input_dim=actor_in_dims_2d[idx],
                    input_channels=actor_in_channels_2d[idx],
                    **actor_cnn_cfg[obs_group],
                )
                print(f"Actor CNN for {obs_group}: {self.actor_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.actor_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.actor_cnns[obs_group].output_dim)
                else:
                    raise ValueError("The output of the actor CNN must be flattened before passing it to the MLP.")
        else:
            self.actor_cnns = None
            encoding_dim = 0

        # Actor MLP
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization (only for 1D actor observations)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # policy distribution
        self.distribution_type = distribution_type
        if self.distribution_type not in ["normal", "tanh"]:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}. Should be 'normal' or 'tanh'.")

        # Critic CNN
        if self.critic_obs_groups_2d:
            # Resolve the critic CNN configuration
            assert critic_cnn_cfg is not None, "A critic CNN configuration is required for 2D critic observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {group: critic_cnn_cfg for group in self.critic_obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D critic observations."
            )

            # Create CNNs for each 2D critic observation
            self.critic_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.critic_obs_groups_2d):
                self.critic_cnns[obs_group] = CNN(
                    input_dim=critic_in_dims_2d[idx],
                    input_channels=critic_in_channels_2d[idx],
                    **critic_cnn_cfg[obs_group],
                )
                print(f"Critic CNN for {obs_group}: {self.critic_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.critic_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.critic_cnns[obs_group].output_dim)
                else:
                    raise ValueError("The output of the critic CNN must be flattened before passing it to the MLP.")
        else:
            self.critic_cnns = None
            encoding_dim = 0

        # Critic MLP with Q-function (takes obs + action)
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.critic = MLP(num_critic_obs_1d + encoding_dim, critic_hidden_dims[-1], critic_hidden_dims[:-1], activation)
        self.critic_embedding_layer = HLGaussLayer(
            in_features=critic_hidden_dims[-1],
            min_value=vmin,
            max_value=vmax,
            num_bins=num_critic_bins,
            sigma=0.75,
        )
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization (only for 1D critic observations)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.num_actions = num_actions
        self.distribution = None

        # Trainable temperature parameters
        self.log_alpha_temp = nn.Parameter(torch.log(torch.tensor(init_alpha_temp)))
        self.log_alpha_kl = nn.Parameter(torch.log(torch.tensor(init_alpha_kl)))

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def alpha_temp(self) -> torch.Tensor:
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        return self.log_alpha_kl.exp()

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(mlp_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(mlp_obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        if self.distribution_type == "normal":
            self.distribution = Normal(mean, std)
        elif self.distribution_type == "tanh":
            self.distribution = TanhNormal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        if self.state_dependent_std:
            return self.actor(mlp_obs)[..., 0, :]
        else:
            return self.actor(mlp_obs)

    def evaluate(
        self, obs: TensorDict, act: Tensor, return_logits: bool = False, **kwargs: dict[str, Any]
    ) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        # Concatenate action to 1D observations
        mlp_obs = torch.cat([mlp_obs, act], dim=-1)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # Encode the 2D critic observations
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        embeddings = self.critic(mlp_obs)
        return self.critic_embedding_layer(self.activation_fn(embeddings), return_logits=return_logits)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.actor_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.critic_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
