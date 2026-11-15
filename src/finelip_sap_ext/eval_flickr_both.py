"""Evaluate Flickr30k retrieval with global and FineLIP fused scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

try:
    import torch
except ImportError:  # Allow file-safety and release tests without local PyTorch.
    torch = None


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for retrieval evaluation")
    return torch


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(
    path: os.PathLike[str] | str,
    payload: dict[str, Any],
    *,
    allow_replace: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not allow_replace:
        raise FileExistsError(f"refusing to replace existing result: {destination}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if allow_replace:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(
                    f"refusing to replace existing result: {destination}"
                ) from None
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_replace_destination(path: os.PathLike[str] | str) -> Path:
    """Allow replacement only for one of the eight isolated new-format results."""
    from finelip_sap_ext.eval_matrix import EVAL_CONFIGS

    destination = Path(path)
    allowed_names = {config.result_name for config in EVAL_CONFIGS}
    if (
        destination.parent.name != "flickr30k_global_and_fused"
        or destination.name not in allowed_names
    ):
        raise ValueError(
            "replacement is restricted to the eight new-format files under "
            "results/flickr30k_global_and_fused"
        )
    return destination


def retrieval_metrics(logits_i2t, captions_per_image: int = 5) -> dict[str, float]:
    framework = _require_torch()
    if captions_per_image < 1:
        raise ValueError("captions_per_image must be at least 1")
    if logits_i2t.ndim != 2:
        raise ValueError("retrieval logits must be a matrix")
    num_images, num_captions = logits_i2t.shape
    if num_captions != num_images * captions_per_image:
        raise ValueError(
            f"expected {captions_per_image} captions per image; "
            f"got shape {tuple(logits_i2t.shape)}"
        )

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        topk = logits_i2t.topk(min(k, num_captions), dim=1).indices
        targets = framework.arange(num_images).view(-1, 1) * captions_per_image
        positives = targets + framework.arange(captions_per_image).view(1, -1)
        hit = (topk.unsqueeze(-1) == positives.unsqueeze(1)).any(dim=(1, 2))
        metrics[f"image_to_text_R@{k}"] = hit.float().mean().item()

    logits_t2i = logits_i2t.T
    text_targets = framework.arange(num_captions) // captions_per_image
    for k in (1, 5, 10):
        topk = logits_t2i.topk(min(k, num_images), dim=1).indices
        hit = (topk == text_targets.view(-1, 1)).any(dim=1)
        metrics[f"text_to_image_R@{k}"] = hit.float().mean().item()
    return metrics


def validate_logits(
    global_logits,
    finegrain_logits,
    expected_images: int = 1000,
    expected_captions: int = 5000,
) -> None:
    framework = _require_torch()
    expected = (expected_images, expected_captions)
    for label, logits in (
        ("global", global_logits),
        ("fine-grained", finegrain_logits),
    ):
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"expected logits shape {expected}; got {label} {tuple(logits.shape)}"
            )
        if not framework.isfinite(logits).all().item():
            raise ValueError(f"non-finite {label} logits")


def fuse_logits(global_logits, finegrain_logits):
    return 0.8 * global_logits + 0.2 * finegrain_logits


def build_result_payload(
    *,
    checkpoint: os.PathLike[str] | str,
    num_images: int,
    num_captions: int,
    global_metrics: dict[str, float],
    fused_metrics: dict[str, float],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": "flickr30k_karpathy_test_1k",
        "num_images": num_images,
        "num_captions": num_captions,
        "fusion": {"global_weight": 0.8, "finegrain_weight": 0.2},
        "global": global_metrics,
        "finegrain_fused": fused_metrics,
    }
    if run_metadata is not None:
        result["run_metadata"] = run_metadata
    return result


class FlickrBothEvaluator:
    def __init__(self, args):
        framework = _require_torch()
        from model import finelip

        self.args = args
        self.finelip = finelip
        self.device = framework.device("cuda:0")
        self.model, self.preprocess = finelip.load(
            args.ckpt_path,
            device="cpu",
            run_finelip=True,
        )
        self.model = self.model.float().to(self.device).eval()

    def encode(self, loader):
        framework = _require_torch()
        from tqdm import tqdm

        image_features = []
        text_features = []
        fine_image_features = []
        fine_text_features = []
        text_lengths = []

        with framework.no_grad():
            for images, captions_by_position in tqdm(
                loader, desc="Encoding Flickr30k"
            ):
                images = images.to(self.device, non_blocking=True)
                image_full = self.model.encode_image_full(images)
                image_global = image_full[:, 0, :]
                image_global = image_global / image_global.norm(
                    dim=-1, keepdim=True
                )

                batch_global = []
                batch_full = []
                batch_lengths = []
                for caption_group in captions_by_position:
                    tokens = self.finelip.tokenize(
                        list(caption_group), truncate=True
                    ).to(self.device)
                    text_full = (
                        self.model.encode_text_full(tokens)
                        @ self.model.text_projection
                    )
                    text_global = text_full[
                        framework.arange(text_full.shape[0], device=self.device),
                        tokens.argmax(dim=-1),
                    ]
                    text_global = text_global / text_global.norm(
                        dim=-1, keepdim=True
                    )
                    batch_global.append(text_global)
                    batch_full.append(text_full.cpu())
                    batch_lengths.append(
                        framework.count_nonzero(tokens, dim=1).cpu()
                    )

                text_features.append(
                    framework.stack(batch_global, dim=1).reshape(
                        -1, batch_global[0].shape[-1]
                    )
                )
                image_features.append(image_global)
                fine_image_features.append(image_full)
                fine_text_features.append(
                    framework.stack(batch_full, dim=1).reshape(
                        -1, batch_full[0].shape[-2], batch_full[0].shape[-1]
                    )
                )
                text_lengths.append(
                    framework.stack(batch_lengths, dim=1).reshape(-1)
                )

            image_global_all = framework.cat(image_features)
            text_global_all = framework.cat(text_features)
            global_logits = (image_global_all @ text_global_all.T).cpu()

            fine_images = framework.cat(fine_image_features)
            fine_texts = framework.cat(fine_text_features)
            lengths = framework.cat(text_lengths)
            fine_logits = framework.empty_like(global_logits)
            chunk = self.args.finegrain_chunk_size
            for start in tqdm(
                range(0, len(fine_texts), chunk),
                desc="Fine-grained scoring",
            ):
                stop = min(start + chunk, len(fine_texts))
                fine_logits[:, start:stop] = self.model.cross_net.forward_dual_aggr(
                    fine_images,
                    fine_texts[start:stop].to(self.device),
                    lengths[start:stop].to(self.device),
                ).cpu()
        return global_logits, fine_logits

    def run(self) -> dict[str, Any]:
        framework = _require_torch()
        from torch.utils.data import DataLoader
        from train.eval_data import Flickr30kTestDataset

        dataset = Flickr30kTestDataset(self.args.flickr_root, self.preprocess)
        num_images = len(dataset)
        num_captions = num_images * 5
        if (num_images, num_captions) != (1000, 5000):
            raise ValueError(
                "expected Flickr30k test split with 1000 images and 5000 "
                f"captions; got {num_images} images and {num_captions} captions"
            )
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0,
        )
        global_logits, finegrain_logits = self.encode(loader)
        validate_logits(global_logits, finegrain_logits)
        fused_logits = fuse_logits(global_logits, finegrain_logits)
        run_metadata = None
        if self.args.metadata_json:
            with open(self.args.metadata_json, encoding="utf-8") as stream:
                run_metadata = json.load(stream)
            if not isinstance(run_metadata, dict):
                raise ValueError("run metadata must be a JSON object")
        result = build_result_payload(
            checkpoint=self.args.ckpt_path,
            num_images=num_images,
            num_captions=num_captions,
            global_metrics=retrieval_metrics(global_logits),
            fused_metrics=retrieval_metrics(fused_logits),
            run_metadata=run_metadata,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if self.args.output:
            if self.args.allow_replace:
                validate_replace_destination(self.args.output)
            atomic_write_json(
                self.args.output,
                result,
                allow_replace=self.args.allow_replace,
            )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one FineLIP checkpoint with global and fused scoring"
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--flickr_root", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--finegrain_chunk_size", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--metadata_json")
    parser.add_argument("--allow_replace", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    if args.finegrain_chunk_size < 1:
        raise SystemExit("--finegrain_chunk_size must be at least 1")
    FlickrBothEvaluator(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
