"""Anchor catalogue -- brief section 3.3, transcribed as structured data.

This is the ONLY valid source of anchor capacity in the system (Determinism
Boundary, plan section 3/14). The LLM/Reasoner may request a named anchor
type; it can never supply or override a capacity value.
"""
from __future__ import annotations

from liftagent.schema import AnchorType

STRENGTH_COLUMNS_MPA: tuple[int, ...] = (15, 25, 35)

CATALOGUE: dict[str, AnchorType] = {
    "ARL-30": AnchorType(
        name="ARL-30", clutch="RU-30",
        capacity_kn={15: 35, 25: 45, 35: 50}, v_zul_kn=25,
        min_edge_mm=350, min_axis_mm=700,
        min_wall_axial_mm=140, min_wall_transverse_mm=200,
    ),
    "ARL-42": AnchorType(
        name="ARL-42", clutch="RU-42",
        capacity_kn={15: 60, 25: 75, 35: 80}, v_zul_kn=40,
        min_edge_mm=500, min_axis_mm=1000,
        min_wall_axial_mm=160, min_wall_transverse_mm=240,
    ),
    "ARL-52": AnchorType(
        name="ARL-52", clutch="RU-52",
        capacity_kn={15: 90, 25: 115, 35: 125}, v_zul_kn=60,
        min_edge_mm=650, min_axis_mm=1300,
        min_wall_axial_mm=200, min_wall_transverse_mm=300,
    ),
    "CFS-WAL-30": AnchorType(
        name="CFS-WAL-30", clutch="WAL",
        capacity_kn={15: 30, 25: 37, 35: 39}, v_zul_kn=20,
        min_edge_mm=300, min_axis_mm=600,
        min_wall_axial_mm=150, min_wall_transverse_mm=220,
    ),
    "HAL-TPA-5.0": AnchorType(
        name="HAL-TPA-5.0", clutch="TPA",
        capacity_kn={15: 40, 25: 48, 35: 50}, v_zul_kn=35,
        min_edge_mm=400, min_axis_mm=800,
        min_wall_axial_mm=150, min_wall_transverse_mm=210,
    ),
}

# Deterministic try-order for the MockReasoner / ITERATE loop: candidate anchor
# first (the brief's own worked example), then escalate by capacity.
CANDIDATE_TRY_ORDER: tuple[str, ...] = (
    "ARL-42", "ARL-52", "HAL-TPA-5.0", "ARL-30", "CFS-WAL-30",
)


class UnknownAnchorType(ValueError):
    pass


def get_anchor(anchor_type: str) -> AnchorType:
    try:
        return CATALOGUE[anchor_type]
    except KeyError as exc:
        raise UnknownAnchorType(anchor_type) from exc


def nearest_available_column(achieved_mpa: float) -> int | None:
    """Highest tabulated strength column <= the achieved concrete strength.

    Returns None if the achieved strength is below every tabulated column
    (i.e. below the first-lift/15 MPa floor) -- callers must treat that as a
    hard stop (brief 3.6: "no lift below the specified first-lift strength"),
    never silently fall back to a lower, unsupported column.
    """
    eligible = [c for c in STRENGTH_COLUMNS_MPA if c <= achieved_mpa]
    return max(eligible) if eligible else None


def lookup_capacity(anchor_type: str, strength_column_mpa: int) -> float:
    anchor = get_anchor(anchor_type)
    if strength_column_mpa not in anchor.capacity_kn:
        raise ValueError(
            f"{anchor_type} has no tabulated capacity at {strength_column_mpa} MPa; "
            f"valid columns are {STRENGTH_COLUMNS_MPA}"
        )
    return anchor.capacity_kn[strength_column_mpa]
