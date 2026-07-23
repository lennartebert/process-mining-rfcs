"""Model definition helpers for synthetic parallel XOR process structures."""

from __future__ import annotations

from .types import XorProbabilityDistribution


def make_xor_probabilities(
    o: int,
    q: XorProbabilityDistribution = "equal",
    majority_share: float = 0.8,
    alpha: float = 1.5,
) -> list[float]:
    """Build one probability vector for an XOR gate."""
    if o < 2:
        raise ValueError("o must be at least 2.")

    if q == "equal":
        return [1.0 / o] * o
    if q == "majority":
        if not 0 < majority_share < 1:
            raise ValueError("majority_share must be between 0 and 1.")
        remaining = 1.0 - majority_share
        return [majority_share] + [remaining / (o - 1)] * (o - 1)
    if q == "power_law":
        weights = [(rank + 1) ** (-alpha) for rank in range(o)]
        total = sum(weights)
        return [weight / total for weight in weights]
    raise ValueError(f"Unknown probability mode: {q}")


def create_parallel_xor_model(
    n: int,
    o: int,
    m: int,
    q: XorProbabilityDistribution = "equal",
    majority_share: float = 0.8,
    alpha: float = 1.5,
    activity_prefix: str | None = None,
) -> dict:
    """Create a serial-XOR-in-parallel structure model configuration."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    if m < 1:
        raise ValueError("m must be at least 1.")

    prefix = f"{activity_prefix}_" if activity_prefix else ""

    parallel_structures = []
    for structure_index in range(1, m + 1):
        gates = []
        for gate_index in range(1, n + 1):
            activities = [
                f"{prefix}parallel_{structure_index}_xor_{gate_index}_option_{option_index}"
                for option_index in range(1, o + 1)
            ]
            probabilities = make_xor_probabilities(
                o=o,
                q=q,
                majority_share=majority_share,
                alpha=alpha,
            )
            gates.append(
                {
                    "gate": f"{prefix}parallel_{structure_index}_xor_{gate_index}",
                    "activities": activities,
                    "probabilities": probabilities,
                }
            )
        parallel_structures.append(gates)

    return {
        "start_activity": f"{prefix}start",
        "end_activity": f"{prefix}parallel_block_complete",
        "parallel_structures": parallel_structures,
        "q": q,
        "majority_share": majority_share,
        "alpha": alpha,
    }
