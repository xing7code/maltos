"""
example:
PYTHONPATH=. python tools/upload_wandb_checkpoint.py \
  --checkpoint-dir checkpoints/llama_50m_dp2_tp2_sp_zero3 \
  --steps 1000 2000 3000 \
  --project maltos \
  --entity xing7-org \
  --artifact-prefix llama-50m-dp2-tp2-sp-zero3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local checkpoint directory to W&B as an artifact.")
    parser.add_argument("--checkpoint-dir", required=True, type=str, help="Checkpoint root directory")
    parser.add_argument("--steps", required=True, type=int, nargs="+", help="Checkpoint steps to upload")
    parser.add_argument("--project", required=True, type=str, help="W&B project")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--run-id", type=str, default=None, help="Existing W&B run id to attach artifacts to")
    parser.add_argument("--resume", type=str, default="allow", choices=("allow", "must", "never", "auto"))
    parser.add_argument("--run-name", type=str, default=None, help="Temporary W&B run name for the upload")
    parser.add_argument("--artifact-prefix", type=str, default=None, help="Artifact name prefix. Defaults to checkpoint root name")
    parser.add_argument("--artifact-type", type=str, default="checkpoint")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Extra artifact alias. Can be passed multiple times",
    )
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_dir)
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint root directory does not exist: {checkpoint_root}")

    import wandb

    artifact_prefix = _sanitize_artifact_name(args.artifact_prefix or checkpoint_root.name)
    init_kwargs = {
        "project": args.project,
        "entity": args.entity,
        "name": args.run_name or f"upload-{artifact_prefix}",
        "job_type": "checkpoint-upload",
    }
    if args.run_id is not None:
        init_kwargs["id"] = args.run_id
        init_kwargs["resume"] = args.resume
        if args.run_name is None:
            init_kwargs.pop("name")
    run = wandb.init(**init_kwargs)
    try:
        for step in args.steps:
            checkpoint_dir = checkpoint_root / f"step_{step:08d}"
            _validate_checkpoint_dir(checkpoint_dir)
            artifact_name = f"{artifact_prefix}-step-{step:08d}"
            aliases = [*args.alias, f"step_{step}"]
            if step == max(args.steps):
                aliases.append("latest")
            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                metadata={"step": step, "path": str(checkpoint_dir), "files": _count_files(checkpoint_dir)},
            )
            artifact.add_dir(str(checkpoint_dir))
            logged_artifact = run.log_artifact(artifact, aliases=aliases)
            if args.wait:
                logged_artifact.wait()
            print(f"uploaded {checkpoint_dir} to {args.project}/{artifact_name} aliases={aliases}")
    finally:
        run.finish()


def _validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint step directory does not exist: {checkpoint_dir}")
    manifest = checkpoint_dir / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"checkpoint manifest not found: {manifest}")


def _count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def _sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-._")
    if not sanitized:
        raise ValueError(f"invalid artifact name: {name!r}")
    return sanitized


if __name__ == "__main__":
    main()
