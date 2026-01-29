from __future__ import annotations

import copy
import os
from typing import Any, Callable, Iterable, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rsl_rl.env import VecEnv
from rsl_rl.env.vec_env import SkrlIsaacLabVecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import string_to_callable


def make_isaaclab_vec_env(
	task_name: str,
	num_envs: int | None = None,
	headless: bool | None = None,
	show_cfg: bool = True,
	wrapper: str = "isaaclab",
	verbose: bool = True,
) -> VecEnv:
	"""Create an Isaac Lab environment via skrl and adapt it to `rsl_rl.env.VecEnv`.

	Uses:
	- `skrl.envs.loaders.torch.load_isaaclab_env`
	- `skrl.envs.wrappers.torch.wrap_env`

	Note: Isaac Lab itself must be installed and typically requires launching via `isaaclab -p ...`.
	"""
	try:
		from skrl.envs.loaders.torch import load_isaaclab_env
		from skrl.envs.wrappers.torch import wrap_env
	except ModuleNotFoundError as err:
		raise ModuleNotFoundError(
			"Missing dependency `skrl`. Install it (and Isaac Lab) to use the default Isaac Lab env."
		) from err

	base_env = load_isaaclab_env(
		task_name=task_name,
		num_envs=num_envs,
		headless=headless,
		show_cfg=show_cfg,
	)
	wrapped_env = wrap_env(base_env, wrapper=wrapper, verbose=verbose)
	return SkrlIsaacLabVecEnv(wrapped_env)


def _resolve_device(device: str) -> str:
	"""Resolve device for single- or multi-GPU training.

	`OnPolicyRunner` expects `device == f"cuda:{LOCAL_RANK}"` when `WORLD_SIZE > 1`.
	"""
	world_size = int(os.getenv("WORLD_SIZE", "1"))
	if world_size <= 1:
		return device

	local_rank = int(os.getenv("LOCAL_RANK", "0"))
	expected = f"cuda:{local_rank}"

	# If the user passed a generic CUDA device, auto-map to the expected rank.
	if device.startswith("cuda") and device != expected:
		return expected

	return device


def resolve_device(device: str) -> str:
	"""Public wrapper for device resolution (single- or multi-GPU)."""
	return _resolve_device(device)


def _set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _make_env(cfg: DictConfig) -> VecEnv:
	if "env" not in cfg or cfg.env is None:
		raise ValueError("Missing `env` section in config.")
	if cfg.env.get("make") is None:
		raise ValueError(
			"Missing `env.make` in config. Provide a callable string `module:attribute` that returns a VecEnv."
		)

	make_fn = string_to_callable(str(cfg.env.make))
	kwargs: dict[str, Any] = {}
	if cfg.env.get("kwargs") is not None:
		kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)  # type: ignore[assignment]
		assert isinstance(kwargs, dict)

	env = make_fn(**kwargs)
	if not isinstance(env, VecEnv):
		# Many external envs implement VecEnv but don't subclass it directly; we still require the interface.
		missing = [
			name
			for name in ("num_envs", "num_actions", "device", "cfg", "get_observations", "step")
			if not hasattr(env, name)
		]
		if missing:
			raise TypeError(f"env.make returned an object that does not look like a VecEnv. Missing: {missing}")
	return env  # type: ignore[return-value]


def make_env(cfg: DictConfig) -> VecEnv:
	"""Public wrapper for constructing the VecEnv from config."""
	return _make_env(cfg)


def train(cfg: DictConfig, env: VecEnv | None = None) -> OnPolicyRunner | DistillationRunner:
	"""Train via `OnPolicyRunner` (or `DistillationRunner`) from a Hydra config."""
	if env is None:
		env = _make_env(cfg)

	runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
	assert isinstance(runner_cfg, dict)
	runner_cfg = copy.deepcopy(runner_cfg)

	# Extract runner selection (kept for compatibility with existing yaml examples)
	runner_class_name = str(runner_cfg.pop("class_name", "OnPolicyRunner"))
	runner_cls: type[OnPolicyRunner | DistillationRunner]
	if runner_class_name == "OnPolicyRunner":
		runner_cls = OnPolicyRunner
	elif runner_class_name == "DistillationRunner":
		runner_cls = DistillationRunner
	else:
		raise ValueError(f"Unknown runner.class_name='{runner_class_name}'.")

	device = _resolve_device(str(cfg.device))

	seed = int(runner_cfg.get("seed", 0))
	if seed:
		_set_seed(seed)

	log_dir = cfg.get("log_dir")
	log_dir = None if log_dir in (None, "null") else str(log_dir)

	runner = runner_cls(env=env, train_cfg=runner_cfg, log_dir=log_dir, device=device)

	# Resume if requested
	resume_path = cfg.get("resume")
	if resume_path not in (None, "null"):
		runner.load(str(resume_path), load_optimizer=bool(cfg.get("load_optimizer", True)))

	num_learning_iterations = cfg.get("num_learning_iterations")
	if num_learning_iterations in (None, "null"):
		num_learning_iterations = int(runner_cfg["max_iterations"])
	else:
		num_learning_iterations = int(num_learning_iterations)

	runner.learn(
		num_learning_iterations=num_learning_iterations,
		init_at_random_ep_len=bool(cfg.get("init_at_random_ep_len", False)),
	)
	return runner


@hydra.main(version_base=None, config_path="config", config_name="train_isaac")
def main(cfg: DictConfig) -> None:
	"""Hydra entrypoint.

	Config is loaded from `rsl_rl/config/train_isaac.yaml`.
	"""
	# Print resolved config for reproducibility
	print(OmegaConf.to_yaml(cfg, resolve=True))
	train(cfg)


if __name__ == "__main__":
	main()
