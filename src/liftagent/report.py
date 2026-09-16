"""F4: structured JSON report (brief section 7 / plan section 28) and a
human-readable, rule-cited summary for the terminal (plan section 32).

Presentation-only translation of agent.py's machine-readable `reason` codes
lives here -- this module never re-derives WHY a status was reached, it only
formats what agent.py already decided.
"""
from __future__ import annotations

import json

from liftagent.schema import AgentResult, Candidate, CheckResult, GeometrySource, Opening

REASON_MESSAGES = {
    "UNRESOLVED_GEOMETRY_CONFLICT": "Geometry conflict -- placement unresolved",
    "TURN_METHOD_UNCONFIRMED": "Turn method unconfirmed",
    "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED": "Supplementary reinforcement not confirmed",
    "NO_CANDIDATE_SATISFIES_REQUIREMENTS": "No catalogue candidate satisfies requirements",
    "ITERATION_IN_PROGRESS": "Iteration in progress",
}


def _round(x: float | None, n: int = 2) -> float | None:
    return None if x is None else round(x, n)


def _opening_dict(o: Opening) -> dict:
    return {"id": o.id, "x_mm": o.x_mm, "width_mm": o.width_mm, "sill_mm": o.sill_mm, "height_mm": o.height_mm}


def _geometry_source_dict(s: GeometrySource) -> dict:
    return {
        "source_type": s.source_type.value, "source_name": s.source_name, "role": s.role.value,
        "length_mm": s.length_mm, "height_mm": s.height_mm, "thickness_mm": s.thickness_mm,
        "openings": [_opening_dict(o) for o in s.openings],
        "extraction_status": s.extraction_status.value, "provenance_note": s.provenance_note,
    }


def _check_dict(c: CheckResult) -> dict:
    return {
        "check_name": c.check_name, "state": c.state.value, "scope": c.scope.value,
        "demand": _round(c.demand), "capacity": _round(c.capacity), "utilisation": _round(c.utilisation, 3),
        "governing_input": c.governing_input, "rule_reference": c.rule_reference,
        "explanation": c.explanation, "governing": c.governing,
        "handling_state": c.handling_state, "anchor_id": c.anchor_id,
        # non-empty => this PASS/FAIL is conditional/reference only, not a verified acceptance
        # (e.g. tabulated capacity while supplementary reinforcement is UNKNOWN/FAIL)
        "conditional_on": c.conditional_on,
    }


def _candidate_dict(c: Candidate | None) -> dict | None:
    if c is None:
        return None
    return {
        "candidate_id": c.candidate_id, "anchor_type": c.anchor_type, "clutch": c.clutch,
        "x1_mm": _round(c.x1_mm, 1), "x2_mm": _round(c.x2_mm, 1),
        "anchors": [
            {"id": a.id, "type": a.anchor_type, "clutch": a.clutch, "x_mm": _round(a.x_mm, 1), "y_mm": _round(a.y_mm, 1)}
            for a in c.anchors
        ],
        "checks": [_check_dict(ch) for ch in c.checks],
        "governing_check": _check_dict(c.governing_check) if c.governing_check is not None else None,
        "rig": (None if c.rig is None else {"spreader_required": c.rig.spreader_required, "reason": c.rig.reason}),
    }


def to_json_dict(result: AgentResult) -> dict:
    return {
        "element_id": result.element_id,
        "status": result.status.value,
        "reason": result.reason,
        "geometry_sources": [_geometry_source_dict(s) for s in result.geometry_sources],
        "selected_geometry": result.selected_geometry,
        "cog": (None if result.cog is None else {
            "x_mm": _round(result.cog.x_mm, 1), "y_mm": _round(result.cog.y_mm, 1),
            "source": result.cog.source, "method": result.cog.method, "note": result.cog.note,
        }),
        "cog_by_source": [
            {"source": c.source, "x_mm": _round(c.x_mm, 1), "y_mm": _round(c.y_mm, 1)}
            for c in result.cog_by_source
        ],
        "self_weight": (None if result.self_weight is None else {
            "value_kN": _round(result.self_weight.value_kn), "method": result.self_weight.method,
            "source": result.self_weight.source, "reference_value_kN": result.self_weight.reference_value_kn,
            "discrepancy_kN": _round(result.self_weight.discrepancy_kn), "warning": result.self_weight.warning,
            "density_kg_per_m3": result.self_weight.density_kg_per_m3,
            "concrete_class": result.self_weight.concrete_class,
        }),
        "self_weight_by_source": [
            {"source": s.source, "value_kN": _round(s.value_kn)} for s in result.self_weight_by_source
        ],
        # Never both populated: resolved_candidate is non-null ONLY when
        # status == ACCEPT_PROVISIONAL; every other status carries the same
        # computed candidate strictly as illustrative_candidate instead.
        "resolved_candidate": _candidate_dict(result.resolved_candidate),
        "illustrative_candidate": _candidate_dict(result.illustrative_candidate),
        "rfis": [{"id": r.id, "message": r.message, "rule_reference": r.rule_reference} for r in result.rfis],
        "rule_trace": [{"step": t.step, "result": t.result, **t.data} for t in result.rule_trace],
        "assumptions": list(result.assumptions),
        "requires_human_signoff": result.requires_human_signoff,
        "engineering_core_version": result.engineering_core_version,
        "rule_set_version": result.rule_set_version,
        "catalogue_version": result.catalogue_version,
    }


def to_json(result: AgentResult, indent: int = 2) -> str:
    return json.dumps(to_json_dict(result), indent=indent)


def render_summary(result: AgentResult) -> str:
    resolved = result.status.value == "ACCEPT_PROVISIONAL"
    candidate = result.resolved_candidate if resolved else result.illustrative_candidate

    lines: list[str] = []
    lines.append(f"{result.element_id} LIFTING ANCHOR ANALYSIS")
    lines.append("-" * 34)
    lines.append(f"STATUS: {result.status.value}")
    if result.reason is not None:
        lines.append(f"Reason: {result.reason} -- {REASON_MESSAGES.get(result.reason, result.reason)}")
    lines.append("")
    lines.append("Geometry sources:")
    for s in result.geometry_sources:
        if s.length_mm is None:
            lines.append(f"  {s.source_name:<22} ({s.role.value}) -- extraction {s.extraction_status.value}")
        else:
            openings_str = f"{len(s.openings)} opening(s)" if s.extraction_status.value == "SUCCESS" else "openings UNKNOWN"
            lines.append(f"  {s.source_name:<22} ({s.role.value}) {s.length_mm:.0f} mm / {openings_str}")
    lines.append("")
    lines.append("Self-weight (derived from geometry x density -- reference figure below is NOT authoritative):")
    for sw in result.self_weight_by_source:
        lines.append(f"  {sw.source:<22} {sw.value_kn:6.1f} kN")
    if result.self_weight is not None:
        lines.append(f"  {'(brief reference)':<22} {result.self_weight.reference_value_kn:6.1f} kN")
    lines.append("")
    if candidate is not None:
        if resolved:
            lines.append("Anchors (RESOLVED):")
        else:
            lines.append("Anchors (ILLUSTRATIVE CANDIDATE -- NOT A RESOLVED PLACEMENT):")
        for a in candidate.anchors:
            lines.append(f"  {a.id}: {a.anchor_type} @ x={a.x_mm:.0f}mm, y={a.y_mm:.0f}mm (clutch {a.clutch})")
        lines.append("")
        if candidate.rig is not None:
            rig = candidate.rig
            lines.append(f"Rig: {'SPREADER REQUIRED' if rig.spreader_required else 'direct slings permitted'} -- {rig.reason}")
        if candidate.governing_check is not None and candidate.governing_check.utilisation is not None:
            gc = candidate.governing_check
            label = "Governing check" if resolved else "Illustrative check"
            tag = " [CONDITIONAL]" if gc.conditional_on else ""
            lines.append(f"{label}: {gc.handling_state}, anchor {gc.anchor_id}, utilisation {gc.utilisation:.2f}{tag}")
            if gc.conditional_on:
                lines.append(f"  CONDITIONAL: {gc.conditional_on}")
            if not resolved:
                lines.append("  NOTE: not used for final acceptance because the result is not resolved (see Reason above).")
    lines.append("")
    lines.append("RFIs:" if result.rfis else "RFIs: none")
    for i, rfi in enumerate(result.rfis, 1):
        lines.append(f"  {i}. {rfi.message}")
    lines.append("")
    lines.append(f"Human sign-off required: {'YES' if result.requires_human_signoff else 'NO'}")
    return "\n".join(lines)
