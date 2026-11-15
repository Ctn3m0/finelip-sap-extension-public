# Fair ablation profile

This profile is for new controlled experiments only. FineLIP and FineLIP+SAP share the same physical batch, gradient accumulation, effective batch, AdamW groups and learning rates, warmup/cosine update clock, seed, dataset split, preprocessing, dataloader construction, and dual evaluation. The sole method switch is `--use_sap` with the fixed SAP parameters in `profile.json`.

Partial final accumulation windows are normalized by their actual sample count. Resume is fail-closed and experiment-specific; cross-method or cross-seed checkpoint reuse is rejected.

Run the stages in order and inspect each gate before spending more compute:

1. `smoke`: 32 deterministic DCI B/16 samples, seed 71, one epoch, no Flickr evaluation.
2. `pilot`: full DCI B/16 pair, seed 71, six epochs, both evaluation modes.
3. `replicate`: the same pair for seeds 72 and 73.
4. Expand to DOCCI or L/14 only after the paired DCI B/16 results and LR/update traces are validated.

