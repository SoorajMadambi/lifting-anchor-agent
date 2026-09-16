"""F1: load the Appendix A JSON (required) and normalise it, optionally
merging in a third geometry source parsed from IFC (optional).

This module only builds typed data and detects conflicts; it performs no
engineering arithmetic (that lives in engineering.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from liftagent.engineering import parse_concrete_class
from liftagent.schema import (
    ConcreteSpec, ElementInput, ExtractionStatus, GeometrySource, Opening,
    ProductionSpec, SourceRole, SourceType,
)

GEOMETRY_CONFLICT_TOLERANCE_MM = 25.0


def load_raw(json_path: str | Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _openings_from_raw(raw_openings: list[dict]) -> tuple[Opening, ...]:
    return tuple(
        Opening(id=o["id"], x_mm=o["x"], width_mm=o["width"], sill_mm=o["sill"], height_mm=o["height"])
        for o in raw_openings
    )


def _approval_source(raw: dict) -> GeometrySource:
    s = raw["sources"]["approval_design"]
    return GeometrySource(
        source_type=SourceType.APPROVAL_JSON, source_name="approval_design",
        role=SourceRole.AUTHORITATIVE_DESIGN,
        length_mm=s["length_mm"], height_mm=s["height_mm"], thickness_mm=s["thickness_mm"],
        openings=_openings_from_raw(s.get("openings_mm", [])),
        extraction_status=ExtractionStatus.SUCCESS,
        provenance_note="Appendix A approval design geometry (JSON, required input).",
    )


def _ifc_export_source(raw: dict) -> GeometrySource:
    s = raw["sources"]["ifc_export"]
    return GeometrySource(
        source_type=SourceType.IFC_EXPORT_JSON, source_name="ifc_export",
        role=SourceRole.EXPORT,
        length_mm=s["length_mm"], height_mm=s["height_mm"], thickness_mm=s["thickness_mm"],
        openings=_openings_from_raw(s.get("openings_mm", [])),
        extraction_status=ExtractionStatus.SUCCESS,
        provenance_note="Appendix A IFC/export summary field (JSON, required input; "
                         "NOT independently re-parsed -- see 'ifc_geometry' for that).",
    )


def _concrete_spec(raw: dict) -> ConcreteSpec:
    c = raw["concrete"]
    cyl, cube = parse_concrete_class(c["class"])
    return ConcreteSpec(
        class_label=c["class"], density_kg_per_m3=c["density_kg_per_m3"],
        first_lift_strength_mpa=c["first_lift_strength_mpa"],
        cylinder_strength_mpa=cyl, cube_strength_mpa=cube,
    )


def _production_spec(raw: dict) -> ProductionSpec:
    p = raw["production"]
    return ProductionSpec(
        cast_orientation=p["cast_orientation"], mould=p["mould"],
        turn_method=p["turn_method"], storage=p["storage"],
    )


def build_element_input(raw: dict, ifc_geometry: GeometrySource | None = None) -> ElementInput:
    sources = [_approval_source(raw), _ifc_export_source(raw)]
    if ifc_geometry is not None:
        sources.append(ifc_geometry)
    return ElementInput(
        element_id=raw["element_id"], element_type=raw["type"], description=raw["description"],
        geometry_sources=tuple(sources),
        concrete=_concrete_spec(raw),
        reinforcement_note=raw.get("reinforcement", {}).get("note", ""),
        production=_production_spec(raw),
        handling_states_required=tuple(raw["handling_states_required"]),
    )


def detect_geometry_conflict(sources: tuple[GeometrySource, ...],
                              tolerance_mm: float = GEOMETRY_CONFLICT_TOLERANCE_MM) -> list[str]:
    """Pairwise-compares every successfully-extracted geometry source.

    Returns a list of human-readable discrepancy descriptions; an empty list
    means all sources agree within tolerance. Sources with a FAILED
    extraction_status are skipped (they contribute no geometry, not a
    disagreement) rather than fabricating a comparison against absent data.
    """
    usable = [s for s in sources if s.extraction_status != ExtractionStatus.FAILED
              and s.length_mm is not None]
    discrepancies: list[str] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            if abs(a.length_mm - b.length_mm) > tolerance_mm:
                discrepancies.append(
                    f"{a.source_name} length {a.length_mm:.0f}mm vs {b.source_name} length {b.length_mm:.0f}mm"
                )
            # Only compare opening counts when BOTH sides reliably extracted openings --
            # a PARTIAL source (e.g. ifc_geometry) has openings=() meaning "not
            # extracted", not "confirmed zero", and must not be compared as if it were.
            if a.extraction_status == ExtractionStatus.SUCCESS and b.extraction_status == ExtractionStatus.SUCCESS:
                if len(a.openings) != len(b.openings):
                    discrepancies.append(
                        f"{a.source_name} has {len(a.openings)} opening(s) vs {b.source_name} has {len(b.openings)}"
                    )
    return discrepancies
