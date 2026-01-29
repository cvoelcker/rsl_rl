from __future__ import annotations

import argparse
import os

from rsl_rl.utils.policy_export import load_policy_checkpoint, save_policy_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a minimal policy checkpoint from a full training checkpoint. "
            "This does NOT construct an environment and can run outside Isaac Sim containers."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint (e.g. model_1000.pt)")
    parser.add_argument("--out", required=True, help="Output path for minimal policy checkpoint")
    parser.add_argument(
        "--map-location",
        default=None,
        help="Optional torch.load map_location (e.g. cpu, cuda:0)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_state_dict, metadata = load_policy_checkpoint(args.checkpoint, map_location=args.map_location)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_policy_checkpoint(out_path, model_state_dict=model_state_dict, metadata=metadata)
    print(f"Wrote minimal policy checkpoint: {out_path}")


if __name__ == "__main__":
    main()
