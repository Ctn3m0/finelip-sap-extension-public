# Scientific protocol

Use `legacy-reproduction` only to reproduce and interpret completed runs. Those FineLIP and FineLIP+SAP runs used different accumulation and effective batches, and their original evaluation modes differed; they do not isolate SAP causally.

Use `fair-ablation` for new comparisons. It pins two GPUs, physical global batch 32, accumulation 4, effective batch 128, AdamW groups and learning rates, optimizer-update-indexed warmup/cosine scheduling, paired seeds, and identical global/fused evaluation. The only method switch is SAP.

The minimum sequence is a 32-sample DCI B/16 smoke pair, one full DCI B/16 seed-71 pilot pair, then paired seeds 72 and 73. Expand to DOCCI or L/14 only after validating finite losses, exact update counts, matching LR traces, controlled-protocol hashes, and paired retrieval differences.

The component installer verifies a separately obtained FineLIP Git revision and exact supported file hashes, then copies extension-owned modules into that checkout. It deliberately does not include, generate, or overwrite the upstream FineLIP training/model source. Therefore the public component artifact alone must not be presented as a runnable reproduction; apply the private/server integration only in an authorized working checkout and preserve upstream attribution and terms.

