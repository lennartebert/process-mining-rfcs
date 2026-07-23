"""Event-log simulation helpers for custom process structures."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd


def sample_parallel_iteration_trace(model: dict) -> list[str]:
    """Sample one full interleaved pass through parallel structures."""
    structure_states = []
    ready_queue = []
    for structure_index, gates in enumerate(model["parallel_structures"]):
        state = {"gates": gates, "gate_pointer": 0}
        structure_states.append(state)
        if state["gates"]:
            ready_queue.append(structure_index)

    sampled_trace = []
    while ready_queue:
        queue_index = random.randrange(len(ready_queue))
        structure_index = ready_queue.pop(queue_index)
        state = structure_states[structure_index]
        current_gate = state["gates"][state["gate_pointer"]]
        selected_activity = random.choices(
            population=current_gate["activities"],
            weights=current_gate["probabilities"],
            k=1,
        )[0]
        sampled_trace.append(selected_activity)
        state["gate_pointer"] += 1
        if state["gate_pointer"] < len(state["gates"]):
            ready_queue.append(structure_index)
    return sampled_trace


def simulate_event_log(
    model: dict,
    N: int,
    p: float = 0.2,
    seed: int | None = 123,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """Generate a synthetic event log from a process model configuration."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0.0 and 1.0.")
    if seed is not None:
        random.seed(seed)
    if start_time is None:
        start_time = datetime.now()

    events = []
    current_time = start_time
    for case_index in range(1, N + 1):
        case_id = f"case_{case_index}"
        trace = [model["start_activity"]]
        should_run_another_iteration = True
        while should_run_another_iteration:
            should_run_another_iteration = False
            trace.extend(sample_parallel_iteration_trace(model))
            trace.append(model["end_activity"])
            if random.random() < p:
                trace.append("loop_back")
                should_run_another_iteration = True

        for activity in trace:
            events.append(
                {"case": case_id, "activity": activity, "timestamp": current_time}
            )
            current_time += timedelta(seconds=random.randint(1, 20))
    return pd.DataFrame(events)
