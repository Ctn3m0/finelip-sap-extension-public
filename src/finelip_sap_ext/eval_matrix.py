"""Single source of truth for the eight legacy checkpoint evaluations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str
    result_name: str
    dataset: str
    backbone: str


EVAL_CONFIGS = (
    EvalConfig("experiments/dci_b16/ckpt/dci_b16_32_epoch_6_finelip-B.pt", "dci_b16_global_fused.json", "dci", "ViT-B/16"),
    EvalConfig("experiments/dci_l14/ckpt/dci_l14_32_epoch_6_finelip-L.pt", "dci_l14_global_fused.json", "dci", "ViT-L/14"),
    EvalConfig("experiments/docci_b16/ckpt/docci_b16_32_epoch_6_finelip-B.pt", "docci_b16_global_fused.json", "docci", "ViT-B/16"),
    EvalConfig("experiments/docci_l14/ckpt/docci_l14_32_epoch_6_finelip-L.pt", "docci_l14_global_fused.json", "docci", "ViT-L/14"),
    EvalConfig("experiments/dci_b16_sap/ckpt/dci_b16_sap_32_epoch_6_finelip-B.pt", "dci_b16_sap_global_fused.json", "dci", "ViT-B/16"),
    EvalConfig("experiments/dci_l14_sap/ckpt/dci_l14_sap_32_epoch_6_finelip-L.pt", "dci_l14_sap_global_fused.json", "dci", "ViT-L/14"),
    EvalConfig("experiments/docci_b16_sap/ckpt/docci_b16_sap_32_epoch_6_finelip-B.pt", "docci_b16_sap_global_fused.json", "docci", "ViT-B/16"),
    EvalConfig("experiments/docci_l14_sap/ckpt/docci_l14_sap_32_epoch_6_finelip-L.pt", "docci_l14_sap_global_fused.json", "docci", "ViT-L/14"),
)


def config_for_index(index: int) -> EvalConfig:
    if not 0 <= index < len(EVAL_CONFIGS):
        raise ValueError(f"Slurm array index must be 0-{len(EVAL_CONFIGS) - 1}; got {index}")
    return EVAL_CONFIGS[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int)
    mode.add_argument("--all-checkpoints", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.all_checkpoints:
        for item in EVAL_CONFIGS:
            print(item.checkpoint)
        return 0
    item = config_for_index(args.index)
    print(f"{item.checkpoint}|{item.result_name}|{item.dataset}|{item.backbone}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

