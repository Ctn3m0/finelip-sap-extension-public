"""Pure optimizer-update counting for the fair-ablation scheduler."""

from __future__ import annotations


SCHEDULER_MODES = ("legacy_microstep", "optimizer_update")


def microbatch_loss_scale(
    batch_index: int,
    num_microbatches: int,
    accumulation_steps: int,
    regular_batch_size: int,
    current_batch_size: int,
    final_batch_size: int,
) -> float:
    """Return sample-weighted scaling within the current accumulation window."""
    if not 0 <= batch_index < num_microbatches:
        raise ValueError("batch_index is outside the epoch")
    if (
        accumulation_steps < 1
        or regular_batch_size < 1
        or current_batch_size < 1
        or final_batch_size < 1
    ):
        raise ValueError("accumulation and batch sizes must be positive")
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    window_stop = min(window_start + accumulation_steps, num_microbatches)
    batch_sizes = [regular_batch_size] * (window_stop - window_start)
    if window_stop == num_microbatches:
        batch_sizes[-1] = final_batch_size
    return current_batch_size / sum(batch_sizes)


def updates_per_epoch(num_microbatches: int, accumulation_steps: int) -> int:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be at least 1")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    return (num_microbatches + accumulation_steps - 1) // accumulation_steps


def total_optimizer_updates(
    epochs: int,
    num_microbatches: int,
    accumulation_steps: int,
) -> int:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    return epochs * updates_per_epoch(num_microbatches, accumulation_steps)


def validate_scheduler_mode(mode: str) -> str:
    if mode not in SCHEDULER_MODES:
        choices = ", ".join(SCHEDULER_MODES)
        raise ValueError(f"scheduler mode must be one of: {choices}")
    return mode


def scheduler_total_steps(
    mode: str,
    epochs: int,
    num_microbatches: int,
    accumulation_steps: int,
):
    validate_scheduler_mode(mode)
    if mode == "optimizer_update":
        return total_optimizer_updates(epochs, num_microbatches, accumulation_steps)
    if epochs < 1 or num_microbatches < 1 or accumulation_steps < 1:
        raise ValueError("epochs, num_microbatches, and accumulation_steps must be positive")
    return (epochs * num_microbatches) / accumulation_steps


def restore_optimizer_update(checkpoint: dict, mode: str) -> int:
    validate_scheduler_mode(mode)
    if "optimizer_update" not in checkpoint:
        if mode == "optimizer_update":
            raise ValueError(
                "optimizer_update scheduler resume requires an optimizer_update "
                "counter in the checkpoint sidecar"
            )
        return 0
    value = checkpoint["optimizer_update"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("optimizer_update counter must be a non-negative integer")
    return value


def validate_fair_resume_metadata(checkpoint: dict, expected: dict) -> None:
    """Reject cross-method/seed/experiment resumes in the fair profile."""
    required = ("profile", "method", "seed", "experiment", "scheduler_mode")
    for key in required:
        if checkpoint.get(key) != expected.get(key):
            raise ValueError(
                f"fair resume {key} mismatch: checkpoint={checkpoint.get(key)!r}, "
                f"expected={expected.get(key)!r}"
            )


def perform_optimizer_update(
    *,
    mode: str,
    scheduler,
    optimizer,
    microstep_index: int,
    optimizer_update: int,
) -> int:
    validate_scheduler_mode(mode)
    if mode == "optimizer_update":
        scheduler(optimizer_update)
    optimizer.step()
    optimizer.zero_grad()
    if mode == "legacy_microstep":
        scheduler(microstep_index)
    return optimizer_update + 1

