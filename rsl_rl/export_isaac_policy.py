from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.train_isaac import make_env, resolve_device


def _make_runner(cfg: DictConfig):
    env = make_env(cfg)

    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    assert isinstance(runner_cfg, dict)

    runner_class_name = str(runner_cfg.pop("class_name", "OnPolicyRunner"))
    if runner_class_name == "OnPolicyRunner":
        runner_cls = OnPolicyRunner
    elif runner_class_name == "DistillationRunner":
        runner_cls = DistillationRunner
    else:
        raise ValueError(f"Unknown runner.class_name='{runner_class_name}'.")

    device = resolve_device(str(cfg.device))
    return runner_cls(env=env, train_cfg=runner_cfg, log_dir=None, device=device)


@hydra.main(version_base=None, config_path="config", config_name="train_isaac")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for exporting a trained policy.

    Usage example:
        isaaclab -p -m rsl_rl.export_isaac_policy \\
          resume=/abs/path/to/model_1000.pt \\
          export_path=/abs/path/to/policy.pt \\
          export_jit=true \\
          export_jit_path=/abs/path/to/policy_jit.pt \\
          export_map_location=cpu
    """

    resume = cfg.get("resume")
    if resume in (None, "null"):
        raise ValueError("Missing `resume`. Provide the checkpoint to export from (e.g. resume=/path/to/model_1000.pt).")

    export_path = cfg.get("export_path")
    if export_path in (None, "null"):
        export_path = os.path.join(os.getcwd(), "policy.pt")

    export_jit = bool(cfg.get("export_jit", False))
    export_jit_path = cfg.get("export_jit_path")
    if export_jit and export_jit_path in (None, "null"):
        export_jit_path = os.path.splitext(str(export_path))[0] + "_jit.pt"

    export_map_location = cfg.get("export_map_location")
    export_map_location = None if export_map_location in (None, "null") else str(export_map_location)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    runner = _make_runner(cfg)
    runner.load(str(resume), load_optimizer=False, map_location=export_map_location)

    runner.save_policy(str(export_path))
    print(f"Exported minimal policy checkpoint to: {export_path}")

    if export_jit:
        runner.export_policy_jit(str(export_jit_path), device=str(export_map_location or runner.device))
        print(f"Exported TorchScript policy to: {export_jit_path}")


if __name__ == "__main__":
    main()
