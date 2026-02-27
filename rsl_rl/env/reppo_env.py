from typing import Optional

import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

class IsaacLabEnv:
    """Wrapper for IsaacLab environments to be compatible with MuJoCo Playground"""

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
        terrain_levels: str | int = "auto",
    ):
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.curriculum = None
        if (
            getattr(env_cfg, "scene", None) is not None
            and getattr(env_cfg.scene, "terrain", None) is not None
            and getattr(env_cfg.scene.terrain, "terrain_generator", None) is not None
        ):
            terrain_generator = env_cfg.scene.terrain.terrain_generator
            if terrain_levels == "auto":
                env_cfg.scene.terrain.max_init_terrain_level = terrain_generator.num_rows - 1
            elif isinstance(terrain_levels, int):
                env_cfg.scene.terrain.max_init_terrain_level = terrain_levels
            else:
                raise ValueError("terrain_levels must be 'auto' or an int.")
            env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.seed = seed
        self.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

        pos_limit = self.envs.unwrapped.scene["robot"].data.soft_joint_pos_limits
        pos_low = pos_limit[0, :, 0]
        pos_high = pos_limit[0, :, 1]
        margin = 0.1 * (pos_high - pos_low)
        self.action_low = pos_low - margin
        self.action_high = pos_high + margin

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.unwrapped.single_observation_space
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.unwrapped.single_observation_space[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]

        obs, _ = self.envs.reset()
        self._obs = obs
        
        self.device = device

        self.cfg = {
            "task_name": task_name,
            "device": device,
            "num_envs": num_envs,
            "seed": seed,
            "action_bounds": action_bounds,
            "terrain_levels": terrain_levels,
        }

    def reset(self, random_start_init: bool = False) -> torch.Tensor:
        obs_dict, _ = self.envs.reset()
        self._obs = obs_dict
        # NOTE: decorrelate episode horizons like RSL‑RL
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        self.episode_length_buf = self.envs.unwrapped.episode_length_buf
        return self.get_observations()
    
    def randomize_num_steps(self, max_episode_steps: int) -> None:
        """Randomize episode horizons by setting the episode_length_buf to random values in [0, max_episode_steps)."""
        self.envs.unwrapped.episode_length_buf = torch.randint(
            high=int(max_episode_steps),
            size=(self.num_envs,),
            device=self.device,
        )
        self.episode_length_buf = self.envs.unwrapped.episode_length_buf
        self.episode_length_buf = self.envs.unwrapped.episode_length_buf
    
    def get_observations(self):
        obs = self._obs
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

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # if self.action_bounds is not None:
        #     actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions)
        self._obs = obs_dict
        dones = (terminations | truncations).to(dtype=torch.long)
        obs = self.get_observations()
        infos["time_outs"] = truncations
        infos["terminations"] = terminations
        self.episode_length_buf = self.envs.unwrapped.episode_length_buf
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        return obs, rew, dones, infos

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )
