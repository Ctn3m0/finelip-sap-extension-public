# FineLIP + SAP research extension

This is an unofficial research extension for controlled FineLIP and FineLIP+SAP experiments. It is not the official FineLIP repository and does not redistribute the FineLIP source tree.

Obtain FineLIP separately from the [official upstream repository](https://github.com/tiiuae/FineLIP). The component installer in `patches-or-installer/` verifies the supported revision and exact upstream file hashes, then copies only extension-owned modules.

**Scope:** this public artifact stores the project-authored SAP, subcaption, scheduler, evaluator, profile, and validation code. It intentionally does **not** patch upstream training/model files and is not, by itself, a runnable FineLIP+SAP trainer. The runnable integration remains in the checksum-verified private/server patch until redistribution permission for upstream-derived integration is established. Do not describe a component-only install as a reproduced training run.

## Where SAP is implemented

- `src/finelip_sap_ext/sap.py`: Subcaption-Aggregated Patch attention pooling and loss.
- `src/finelip_sap_ext/subcaptions.py`: deterministic construction and controlled merging of subcaptions.
- `patches-or-installer/install_into_finelip.py`: version/hash-checked component installer for a separately obtained FineLIP checkout; it does not install private training hooks.
- `src/finelip_sap_ext/eval_flickr_both.py`: one-pass global and FineLIP fused Flickr30k evaluation.
- `src/finelip_sap_ext/scheduler_updates.py`: explicit legacy and optimizer-update scheduler semantics.

The literal FineLIP+SAP objective used here is FineLIP loss plus weight-1 SAP, at most 15 subcaptions, adjacent merge probability 0.5, and local-rank sigmoid contrastive SAP. It is not the complete MulCLIP objective: Global loss, Local Calibration, and WPR are not included.

In the private/server integration, `train/train.py` constructs subcaptions and passes them into `model/model_finelip.py`, which invokes `src/finelip_sap_ext/sap.py`. Those upstream-derived integration files are deliberately excluded here.

## Reproducibility profiles

- `profiles/legacy-reproduction/` records the completed runs, including their different effective batches and legacy microstep-indexed scheduler. It is not valid for attributing differences solely to SAP.
- `profiles/fair-ablation/` pins the physical batch, accumulation, optimizer, learning rates, scheduler clock, seeds, preprocessing, and evaluation. `--use_sap` is the sole method switch.

## Upstream and research attribution

FineLIP:

```bibtex
@InProceedings{Asokan_2025_CVPR,
  author = {Asokan, Mothilal and Wu, Kebin and Albreiki, Fatima},
  title = {FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  year = {2025},
  pages = {14495--14504}
}
```

SAP source research:

```bibtex
@inproceedings{truong2026mulclip,
  author = {Truong, Chau and Ta Quang, Hieu and Liu, Zhenzhen and Le, Dung D.},
  title = {MulCLIP: A Multi-level Alignment Framework for Enhancing Fine-grained Long-context CLIP},
  booktitle = {40th Conference on Neural Information Processing Systems},
  year = {2026}
}
```

See [MulCLIP](https://github.com/truongchau2602/MulCLIP) for the research project's announced code location.

## Licensing status

At the recorded inspection date, no LICENSE file was observed in the official FineLIP repository. GitHub's [licensing documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) explains that public visibility alone is not an open-source license. Accordingly, this repository contains only project-authored extension material and requires FineLIP to be obtained separately.

No license has yet been selected for this extension. See `LICENSE-PENDING.md`. This repository does not grant rights to upstream FineLIP code. This is not legal advice.

