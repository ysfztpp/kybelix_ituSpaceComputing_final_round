from __future__ import annotations

from collections import defaultdict

import torch


def _prefix_argmin(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_cost = torch.empty_like(values)
    prefix_arg = torch.empty(values.shape[0], dtype=torch.long)
    best_cost = float("inf")
    best_index = 0
    for index in range(values.shape[0]):
        current = float(values[index])
        if current < best_cost:
            best_cost = current
            best_index = index
        prefix_cost[index] = best_cost
        prefix_arg[index] = best_index
    return prefix_cost, prefix_arg


def monotonic_viterbi_decode(
    stage_logits: torch.Tensor,
    point_ids: torch.Tensor,
    query_doys: torch.Tensor,
) -> torch.Tensor:
    """Decode non-decreasing stage sequences independently for each point.

    The decoded path minimizes the summed negative log-probability subject to a
    monotonic phenology constraint. This keeps stage order consistent across
    query dates for the same point without changing the model logits.
    """

    if stage_logits.ndim != 2:
        raise ValueError(f"stage_logits must be 2D, got shape {tuple(stage_logits.shape)}")
    if stage_logits.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)

    log_probs = torch.log_softmax(stage_logits.detach().float().cpu(), dim=1)
    costs = -log_probs
    point_ids_cpu = point_ids.detach().cpu().reshape(-1).tolist()
    query_doys_cpu = query_doys.detach().float().cpu().reshape(-1).tolist()
    decoded = torch.empty(stage_logits.shape[0], dtype=torch.long)

    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row_index, (point_id, query_doy) in enumerate(zip(point_ids_cpu, query_doys_cpu)):
        grouped[int(point_id)].append((float(query_doy), row_index))

    for _point_id, group in grouped.items():
        ordered = [row_index for _query_doy, row_index in sorted(group, key=lambda item: (item[0], item[1]))]
        group_costs = costs[ordered]
        steps, stage_count = group_costs.shape
        backptr = torch.zeros((steps, stage_count), dtype=torch.long)
        dp = group_costs[0].clone()
        for step in range(1, steps):
            prefix_cost, prefix_arg = _prefix_argmin(dp)
            dp = group_costs[step] + prefix_cost
            backptr[step] = prefix_arg
        sequence = torch.empty(steps, dtype=torch.long)
        state = int(dp.argmin().item())
        for step in range(steps - 1, -1, -1):
            sequence[step] = state
            state = int(backptr[step, state].item())
        for ordered_index, row_index in enumerate(ordered):
            decoded[row_index] = sequence[ordered_index]
    return decoded


def transition_viterbi_decode(
    stage_logits: torch.Tensor,
    point_ids: torch.Tensor,
    query_doys: torch.Tensor,
    stay_cost: float = 0.0,
    step_cost: float = 0.05,
    skip_cost: float = 0.35,
    backward_cost: float = 1.25,
) -> torch.Tensor:
    """Decode with a soft phenology transition prior instead of a hard monotonic constraint."""

    if stage_logits.ndim != 2:
        raise ValueError(f"stage_logits must be 2D, got shape {tuple(stage_logits.shape)}")
    if stage_logits.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)

    log_probs = torch.log_softmax(stage_logits.detach().float().cpu(), dim=1)
    emission_costs = -log_probs
    stage_count = emission_costs.shape[1]
    transition = torch.full((stage_count, stage_count), fill_value=skip_cost, dtype=torch.float32)
    for previous in range(stage_count):
        for current in range(stage_count):
            delta = current - previous
            if delta == 0:
                transition[previous, current] = stay_cost
            elif delta == 1:
                transition[previous, current] = step_cost
            elif delta > 1:
                transition[previous, current] = skip_cost * float(delta)
            else:
                transition[previous, current] = backward_cost * float(abs(delta))

    point_ids_cpu = point_ids.detach().cpu().reshape(-1).tolist()
    query_doys_cpu = query_doys.detach().float().cpu().reshape(-1).tolist()
    decoded = torch.empty(stage_logits.shape[0], dtype=torch.long)

    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row_index, (point_id, query_doy) in enumerate(zip(point_ids_cpu, query_doys_cpu)):
        grouped[int(point_id)].append((float(query_doy), row_index))

    for _point_id, group in grouped.items():
        ordered = [row_index for _query_doy, row_index in sorted(group, key=lambda item: (item[0], item[1]))]
        group_costs = emission_costs[ordered]
        steps = group_costs.shape[0]
        backptr = torch.zeros((steps, stage_count), dtype=torch.long)
        dp = group_costs[0].clone()
        for step in range(1, steps):
            expanded = dp.unsqueeze(1) + transition
            best_prev_cost, best_prev_state = expanded.min(dim=0)
            dp = group_costs[step] + best_prev_cost
            backptr[step] = best_prev_state
        sequence = torch.empty(steps, dtype=torch.long)
        state = int(dp.argmin().item())
        for step in range(steps - 1, -1, -1):
            sequence[step] = state
            state = int(backptr[step, state].item())
        for ordered_index, row_index in enumerate(ordered):
            decoded[row_index] = sequence[ordered_index]
    return decoded


def maybe_decode_stages(
    stage_logits: torch.Tensor,
    point_ids: torch.Tensor,
    query_doys: torch.Tensor,
    mode: str | None = None,
) -> torch.Tensor:
    mode_normalized = str(mode or "none").strip().lower()
    if mode_normalized in {"", "none", "off", "disabled"}:
        return stage_logits.detach().argmax(dim=1).cpu()
    if mode_normalized in {"monotonic", "monotonic_viterbi", "pointwise_monotonic"}:
        return monotonic_viterbi_decode(stage_logits, point_ids, query_doys)
    if mode_normalized in {"transition_viterbi", "soft_transition_viterbi", "stage_transition"}:
        return transition_viterbi_decode(stage_logits, point_ids, query_doys)
    raise ValueError(f"unsupported stage_postprocess mode: {mode}")
