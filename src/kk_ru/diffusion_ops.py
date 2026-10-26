from __future__ import annotations

import torch
import torch.nn.functional as F


def corrupt_targets(
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    timestep: torch.Tensor,
    total_steps: int,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = timestep.to(torch.float32).unsqueeze(1) / total_steps
    masked = (torch.rand(target_ids.shape, device=target_ids.device) < probability) & target_mask
    missing = ~masked.any(dim=1)
    candidates = torch.rand(target_ids.shape, device=target_ids.device)
    candidates = candidates.masked_fill(~target_mask, -1)
    selected = candidates.argmax(dim=1)
    rows = torch.arange(target_ids.shape[0], device=target_ids.device)
    masked[rows[missing], selected[missing]] = True
    noisy = target_ids.masked_fill(masked, mask_id)
    return noisy, masked


def masked_token_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    masked: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.float().transpose(1, 2), target_ids, reduction="none"
    )
    weights = masked.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def target_length_labels(target_mask: torch.Tensor) -> torch.Tensor:
    lengths = target_mask.sum(dim=1)
    return lengths - 1


def reveal_tokens(
    noisy_ids: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_id: int,
    steps_remaining: int,
) -> torch.Tensor:
    if steps_remaining <= 0:
        raise ValueError("steps_remaining must be positive")
    probabilities = logits.float().softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    masked = (noisy_ids == mask_id) & valid_mask
    remaining = masked.sum(dim=1)
    reveal_count = torch.div(
        remaining + steps_remaining - 1,
        steps_remaining,
        rounding_mode="floor",
    )
    ranked = confidence.masked_fill(~masked, -torch.inf).argsort(dim=1, descending=True)
    ranks = torch.arange(noisy_ids.shape[1], device=noisy_ids.device).unsqueeze(0)
    reveal_rank = ranks < reveal_count.unsqueeze(1)
    reveal_rank = reveal_rank.expand_as(noisy_ids)
    selected = torch.zeros_like(masked).scatter(1, ranked, reveal_rank)
    return torch.where(selected, predictions, noisy_ids)
