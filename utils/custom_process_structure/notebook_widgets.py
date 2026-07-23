"""ipywidgets helpers for interactive simulation notebooks."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator

import ipywidgets as widgets
from IPython.display import Image, Markdown, display

from .types import ActivitySetOverlap, ControlFlowSettings, XorProbabilityDistribution

CONTROL_STYLE = {"description_width": "220px"}
CONTROL_LAYOUT = widgets.Layout(width="600px")
GLOBAL_RANDOM_SEED = 123


def make_N_slider(value: int = 10_000, description: str = "Num. cases (N)") -> widgets.IntSlider:
    return widgets.IntSlider(
        value=value,
        min=100,
        max=100_000,
        step=100,
        description=description,
        style=CONTROL_STYLE,
        layout=CONTROL_LAYOUT,
    )


def make_control_flow_sliders(
    defaults: ControlFlowSettings | None = None,
) -> dict[str, widgets.Widget]:
    """Build control-flow sliders; keys match parameter symbols."""
    defaults = defaults or {}
    return {
        "m": widgets.IntSlider(
            value=defaults.get("m", 2),
            min=1,
            max=10,
            step=1,
            description="Num. parallel paths (m)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
        ),
        "n": widgets.IntSlider(
            value=defaults.get("n", 2),
            min=1,
            max=10,
            step=1,
            description="Num. XOR gates (n)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
        ),
        "o": widgets.IntSlider(
            value=defaults.get("o", 3),
            min=2,
            max=12,
            step=1,
            description="Choices per XOR gate (o)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
        ),
        "p": widgets.FloatSlider(
            value=defaults.get("p", 0.2),
            min=0.0,
            max=0.9,
            step=0.01,
            description="Loop probability (p)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
            readout_format=".2f",
        ),
        "q": widgets.Dropdown(
            options=["equal", "majority", "power_law"],
            value=defaults.get("q", "power_law"),
            description="XOR probability distribution (q)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
        ),
        "alpha": widgets.FloatSlider(
            value=defaults.get("alpha", 1.5),
            min=0.5,
            max=3.0,
            step=0.1,
            description="Power-law exponent (α)",
            style=CONTROL_STYLE,
            layout=CONTROL_LAYOUT,
            readout_format=".1f",
        ),
    }


def make_overlap_selector(value: ActivitySetOverlap = "overlapping") -> widgets.Dropdown:
    return widgets.Dropdown(
        options=["overlapping", "non_overlapping"],
        value=value,
        description="Overlapping activity sets (s)",
        style=CONTROL_STYLE,
        layout=CONTROL_LAYOUT,
    )


def collect_control_flow(widgets_dict: dict[str, widgets.Widget]) -> ControlFlowSettings:
    """Read control-flow widget values."""
    q_value: XorProbabilityDistribution = widgets_dict["q"].value
    return {
        "m": widgets_dict["m"].value,
        "n": widgets_dict["n"].value,
        "o": widgets_dict["o"].value,
        "p": widgets_dict["p"].value,
        "q": q_value,
        "alpha": widgets_dict["alpha"].value,
    }


def control_flow_widget_list(widgets_dict: dict[str, widgets.Widget]) -> list[widgets.Widget]:
    """Ordered widget list for VBox display."""
    return [
        widgets_dict["m"],
        widgets_dict["n"],
        widgets_dict["o"],
        widgets_dict["p"],
        widgets_dict["q"],
        widgets_dict["alpha"],
    ]


def _image_if_present(png: bytes | None, caption: str):
    if png:
        return [Markdown(f"**{caption}**"), Image(data=png)]
    return [Markdown(f"*{caption} — not available (Graphviz `dot` missing?)*")]


def iter_simple_displays(result: dict) -> Iterator:
    """Display objects for one simple experiment (call ``display`` in the notebook)."""
    yield from _image_if_present(result.get("bpmn_png"), "BPMN")
    yield Markdown("**RFC**")
    yield Image(data=result["rfc_png"])


def iter_two_process_displays(result: dict, saved_paths: dict | None = None) -> Iterator:
    """Display objects for composition/drift (call ``display`` in the notebook)."""
    yield from _image_if_present(result.get("bpmn_a_png"), "BPMN — process a")
    yield from _image_if_present(result.get("bpmn_b_png"), "BPMN — process b")
    yield Markdown("**RFC — process a**")
    yield Image(data=result["rfc_a_png"])
    yield Markdown("**RFC — process b**")
    yield Image(data=result["rfc_b_png"])
    yield Markdown("**RFC — combined**")
    yield Image(data=result["rfc_png"])
    if saved_paths:
        yield Markdown(
            f"Saved BPMN a: `{saved_paths['bpmn_pdf_a']}`  \n"
            f"Saved BPMN b: `{saved_paths['bpmn_pdf_b']}`  \n"
            f"Saved RFC a: `{saved_paths['rfc_pdf_a']}`  \n"
            f"Saved RFC b: `{saved_paths['rfc_pdf_b']}`  \n"
            f"Saved RFC combined: `{saved_paths['rfc_pdf_combined']}`"
        )


def display_in_output(output: widgets.Output, render_fn: Callable[[], None]) -> None:
    """Run *render_fn* inside *output* and show tracebacks on failure."""
    with output:
        output.clear_output(wait=True)
        try:
            render_fn()
        except Exception:
            traceback.print_exc()


def show_objects(objects: Iterator) -> None:
    """Display an iterable of IPython display objects (use inside ``display_in_output``)."""
    for obj in objects:
        display(obj)
