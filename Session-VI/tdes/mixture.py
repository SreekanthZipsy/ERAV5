"""Curriculum mixture schedule with protected floors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class StageSpec:
    name: str
    start_frac: float
    end_frac: float
    weights: dict[str, float]  # lane -> weight (sum ~ 1)
    seq_len: int


# Aligns with Session-V plan, scaled to demo lanes.
STAGES: list[StageSpec] = [
    StageSpec(
        "B0_foundation",
        0.0,
        0.35,
        {"web": 0.35, "indic": 0.18, "code": 0.15, "stem": 0.12, "reasoning": 0.08, "agentic": 0.05, "india": 0.0},
        seq_len=32,
    ),
    StageSpec(
        "B1_skills",
        0.35,
        0.65,
        {"web": 0.25, "code": 0.22, "indic": 0.18, "stem": 0.12, "reasoning": 0.10, "agentic": 0.08, "india": 0.0},
        seq_len=32,
    ),
    StageSpec(
        "B2_lean",
        0.65,
        0.90,
        {"web": 0.18, "code": 0.22, "indic": 0.16, "reasoning": 0.14, "agentic": 0.14, "stem": 0.10, "india": 0.0},
        seq_len=48,
    ),
    StageSpec(
        "B3_pre_anneal",
        0.90,
        1.01,
        {"web": 0.15, "code": 0.20, "agentic": 0.16, "reasoning": 0.16, "indic": 0.15, "stem": 0.10, "india": 0.0},
        seq_len=48,
    ),
]

# Protected floors — selector may not go below these batch token shares.
PROTECTED_FLOORS = {
    "indic": 0.12,
    "agentic": 0.03,
    "code": 0.10,
}


def stage_for_progress(progress: float) -> StageSpec:
    for s in STAGES:
        if s.start_frac <= progress < s.end_frac:
            return s
    return STAGES[-1]


def normalize_weights(w: dict[str, float], available_lanes: set[str]) -> dict[str, float]:
    filtered = {k: v for k, v in w.items() if k in available_lanes and v > 0}
    s = sum(filtered.values()) or 1.0
    return {k: v / s for k, v in filtered.items()}


def enforce_floors(weights: dict[str, float], floors: dict[str, float] | None = None) -> dict[str, float]:
    """Raise lanes to floors, then renormalize remaining mass without breaching floors."""
    floors = floors or PROTECTED_FLOORS
    w = {k: float(v) for k, v in weights.items()}
    for lane, fl in floors.items():
        if fl > 0:
            w[lane] = max(w.get(lane, 0.0), fl)

    active_floors = {k: floors[k] for k in w if floors.get(k, 0.0) > 0}
    floor_sum = sum(active_floors.values())
    if floor_sum >= 1.0:
        return {k: active_floors.get(k, 0.0) / floor_sum for k in w}

    # Lock floor lanes at exactly their floor; give leftover to non-floor lanes.
    out = {k: 0.0 for k in w}
    for k, fl in active_floors.items():
        out[k] = fl
    remaining = 1.0 - floor_sum
    free = [k for k in w if k not in active_floors]
    if not free:
        return out
    free_mass = sum(max(weights.get(k, 0.0), 0.0) for k in free) or float(len(free))
    for k in free:
        out[k] = remaining * (max(weights.get(k, 0.0), 0.0) / free_mass)
    # Fix float drift on free lanes only
    drift = 1.0 - sum(out.values())
    out[free[0]] += drift
    return out


def compile_schedule(total_steps: int, available_lanes: set[str]) -> list[dict]:
    schedule = []
    for step in range(total_steps):
        progress = step / max(total_steps, 1)
        stage = stage_for_progress(progress)
        weights = normalize_weights(stage.weights, available_lanes)
        weights = enforce_floors(weights)
        schedule.append(
            {
                "step": step,
                "progress": progress,
                "stage": stage.name,
                "seq_len": stage.seq_len,
                "weights": weights,
            }
        )
    return schedule


def iter_lane_quota(weights: dict[str, float], n_slots: int) -> Iterator[str]:
    """Deterministic lane assignment for n_slots using largest-remainder method."""
    raw = {k: weights[k] * n_slots for k in weights}
    base = {k: int(v) for k, v in raw.items()}
    rem = sorted(((raw[k] - base[k], k) for k in weights), reverse=True)
    assigned = sum(base.values())
    i = 0
    while assigned < n_slots and rem:
        base[rem[i % len(rem)][1]] += 1
        assigned += 1
        i += 1
    for lane, count in sorted(base.items()):
        for _ in range(count):
            yield lane
