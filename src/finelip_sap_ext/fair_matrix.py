"""Controlled FineLIP versus FineLIP+SAP stage matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from finelip_sap_ext.eval_flickr_both import atomic_write_json


@dataclass(frozen=True)
class FairConfig:
    method: str
    seed: int
    dataset: str
    backbone: str
    experiment: str
    epochs: int
    max_samples: int | None
    run_eval: bool
    use_sap: bool


def _pair(stage: str, seed: int, epochs: int, max_samples: int | None, run_eval: bool):
    common = {
        "seed": seed,
        "dataset": "dci",
        "backbone": "ViT-B/16",
        "epochs": epochs,
        "max_samples": max_samples,
        "run_eval": run_eval,
    }
    return (
        FairConfig(
            method="finelip",
            experiment=f"fair_{stage}_dci_b16_seed{seed}_finelip",
            use_sap=False,
            **common,
        ),
        FairConfig(
            method="finelip_sap",
            experiment=f"fair_{stage}_dci_b16_seed{seed}_finelip_sap",
            use_sap=True,
            **common,
        ),
    )


def configs_for_stage(stage: str) -> tuple[FairConfig, ...]:
    if stage == "smoke":
        return _pair(stage, seed=71, epochs=1, max_samples=32, run_eval=False)
    if stage == "pilot":
        return _pair(stage, seed=71, epochs=6, max_samples=None, run_eval=True)
    if stage == "replicate":
        return (
            *_pair(stage, seed=72, epochs=6, max_samples=None, run_eval=True),
            *_pair(stage, seed=73, epochs=6, max_samples=None, run_eval=True),
        )
    raise ValueError("stage must be one of: smoke, pilot, replicate")


def protocol_metadata(stage: str, config: FairConfig) -> dict:
    controlled = {
        "stage": stage,
        "seed": config.seed,
        "dataset": config.dataset,
        "backbone": config.backbone,
        "epochs": config.epochs,
        "max_samples": config.max_samples,
        "run_eval": config.run_eval,
        "gpus": 2,
        "physical_global_batch": 32,
        "per_rank_batch": 16,
        "accumulation_steps": 4,
        "effective_batch": 128,
        "optimizer": "AdamW",
        "learning_rate": 1e-6,
        "cross_net_learning_rate": 2e-4,
        "weight_decay": 1e-2,
        "warmup_length": 200,
        "scheduler_mode": "optimizer_update",
        "evaluation_batch_size": 32,
        "finegrain_chunk_size": 4,
    }
    canonical = json.dumps(
        controlled, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "profile": "fair-ablation",
        "stage": stage,
        "method": config.method,
        "seed": config.seed,
        "dataset": config.dataset,
        "backbone": config.backbone,
        "experiment": config.experiment,
        "use_sap": config.use_sap,
        "controlled_protocol": controlled,
        "controlled_protocol_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_protocol_metadata(path: Path | str, metadata: dict) -> None:
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != metadata:
            raise ValueError(
                f"refusing to replace different protocol metadata: {path}"
            )
        return
    atomic_write_json(path, metadata)


def _line(config: FairConfig) -> str:
    max_samples = "" if config.max_samples is None else str(config.max_samples)
    return "|".join(
        (
            config.method,
            str(config.seed),
            config.dataset,
            config.backbone,
            config.experiment,
            str(config.epochs),
            max_samples,
            "1" if config.run_eval else "0",
            "1" if config.use_sap else "0",
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--metadata-output")
    args = parser.parse_args(argv)
    configs = configs_for_stage(args.stage)
    if args.all:
        for config in configs:
            print(_line(config))
        return 0
    if not 0 <= args.index < len(configs):
        raise SystemExit(f"array index must be 0-{len(configs) - 1}")
    config = configs[args.index]
    if args.metadata_output:
        write_protocol_metadata(
            args.metadata_output,
            protocol_metadata(args.stage, config),
        )
    print(_line(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

