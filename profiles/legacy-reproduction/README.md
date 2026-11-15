# Legacy reproduction profile

This profile preserves the completed experiment behavior. It deliberately does not change accumulation or scheduler semantics and is retained so the original checkpoints and logs can be interpreted and reproduced.

FineLIP used effective batch 128, while FineLIP+SAP used effective batch 512. The legacy scheduler was indexed by dataloader microsteps after optimizer updates even though its nominal duration was expressed in optimizer updates. Therefore these runs are not a causal SAP ablation.

Use `scripts/submit_eval_all_global_fused.sh` to compare the existing checkpoints under identical evaluation without retraining them.

