"""BPMN creation and rendering helpers for custom process structures."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pm4py.objects.bpmn.obj import BPMN
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

from .model_definition import make_xor_probabilities
from .types import XorProbabilityDistribution


def build_bpmn_from_instruction(
    n: int,
    o: int,
    m: int,
    q: XorProbabilityDistribution,
    alpha: float,
    p: float,
) -> BPMN:
    """Build a BPMN graph matching one synthetic model configuration."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if o < 2:
        raise ValueError("o must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    model = BPMN(name="Parallel XOR Loop Model")
    start_event = BPMN.StartEvent(name="Start")
    final_end_event = BPMN.EndEvent(name="Process Completed")
    and_split = BPMN.ParallelGateway(name="Parallel Split")
    and_join = BPMN.ParallelGateway(name="Parallel Join")
    model.add_node(start_event)
    model.add_node(final_end_event)
    model.add_node(and_split)
    model.add_node(and_join)
    model.add_flow(BPMN.SequenceFlow(start_event, and_split))

    option_probs = make_xor_probabilities(o=o, q=q, alpha=alpha)
    for structure_index in range(1, m + 1):
        previous_node = and_split
        for gate_index in range(1, n + 1):
            split_gateway = BPMN.ExclusiveGateway(
                name=f"Structure {structure_index} XOR Gate {gate_index} Split"
            )
            join_gateway = BPMN.ExclusiveGateway(
                name=f"Structure {structure_index} XOR Gate {gate_index} Join"
            )
            model.add_node(split_gateway)
            model.add_node(join_gateway)
            model.add_flow(BPMN.SequenceFlow(previous_node, split_gateway))
            for option_index in range(1, o + 1):
                option_probability = option_probs[option_index - 1]
                task = BPMN.Task(
                    name=(
                        f"Structure {structure_index} Gate {gate_index} "
                        f"Choice {option_index} (p={option_probability:.2f})"
                    )
                )
                model.add_node(task)
                model.add_flow(BPMN.SequenceFlow(split_gateway, task))
                model.add_flow(BPMN.SequenceFlow(task, join_gateway))
            previous_node = join_gateway
        model.add_flow(BPMN.SequenceFlow(previous_node, and_join))

    loop_decision = BPMN.ExclusiveGateway(name="Loop Decision")
    complete_process_task = BPMN.Task(name=f"Complete Process (p={1.0 - p:.2f})")
    loop_back_task = BPMN.Task(name=f"Loop Back (p={p:.2f})")
    model.add_node(loop_decision)
    model.add_node(complete_process_task)
    model.add_node(loop_back_task)
    model.add_flow(BPMN.SequenceFlow(and_join, loop_decision))
    model.add_flow(BPMN.SequenceFlow(loop_decision, complete_process_task))
    model.add_flow(BPMN.SequenceFlow(loop_decision, loop_back_task))
    model.add_flow(BPMN.SequenceFlow(complete_process_task, final_end_event))
    model.add_flow(BPMN.SequenceFlow(loop_back_task, and_split))
    return model


def render_bpmn_bytes(bpmn_model: BPMN, image_format: str = "png", dpi: int = 300) -> bytes:
    """Render BPMN model as bytes using pm4py visualization backend."""
    gviz = bpmn_visualizer.apply(
        bpmn_model,
        parameters={"format": image_format, "dpi": dpi},
    )
    with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    bpmn_visualizer.save(gviz, str(tmp_path))
    content = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return content
