from __future__ import annotations

from liftagent import ingest
from liftagent.agent import run_agent
from liftagent.reasoner import MockReasoner
from liftagent.schema import SourceRole, SourceType, Status
from tests.conftest import make_element_input, make_geometry_source, make_production


def test_no_conflict_when_sources_agree():
    sources = (
        make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4700, role=SourceRole.EXPORT, name="ifc_export"),
    )
    discrepancies = ingest.detect_geometry_conflict(sources)
    assert discrepancies == []


def test_conflict_detected_on_length_mismatch():
    sources = (
        make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4300, role=SourceRole.EXPORT, name="ifc_export"),
    )
    discrepancies = ingest.detect_geometry_conflict(sources)
    assert len(discrepancies) == 1
    assert "4700" in discrepancies[0] and "4300" in discrepancies[0]


def test_three_way_conflict_reports_all_disagreeing_pairs():
    sources = (
        make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4300, role=SourceRole.EXPORT, name="ifc_export"),
        make_geometry_source(length_mm=4750, role=SourceRole.DERIVED_REFERENCE, name="ifc_geometry",
                              source_type=SourceType.IFC_PARSED),
    )
    discrepancies = ingest.detect_geometry_conflict(sources)
    # approval vs export, approval vs ifc_geometry, export vs ifc_geometry (all differ by > tolerance)
    assert len(discrepancies) == 3


def test_partial_extraction_source_excluded_from_opening_count_comparison():
    from liftagent.schema import ExtractionStatus, Opening
    door = Opening(id="door", x_mm=1150, width_mm=1100, sill_mm=0, height_mm=2100)
    sources = (
        make_geometry_source(length_mm=4700, openings=(door,), role=SourceRole.AUTHORITATIVE_DESIGN,
                              name="approval_design"),
        make_geometry_source(length_mm=4700, openings=(), role=SourceRole.DERIVED_REFERENCE,
                              name="ifc_geometry", source_type=SourceType.IFC_PARSED,
                              status=ExtractionStatus.PARTIAL),
    )
    discrepancies = ingest.detect_geometry_conflict(sources)
    assert discrepancies == []  # lengths agree; opening count not compared because ifc_geometry is PARTIAL


def test_agent_holds_on_geometry_conflict():
    sources = (
        make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4300, role=SourceRole.EXPORT, name="ifc_export"),
    )
    element = make_element_input(geometry_sources=sources, production=make_production("tilting_table"))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    assert result.selected_geometry is None
    assert result.resolved_candidate is None  # safety invariant: unresolved geometry -> never a resolved placement
    assert any("geometry" in r.message.lower() or "discrepancy" in r.message.lower() for r in result.rfis)


def test_agent_holds_on_unconfirmed_turn_method_even_with_clean_geometry():
    sources = (make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    element = make_element_input(geometry_sources=sources, production=make_production("UNCONFIRMED"))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert any("turn method" in r.message.lower() for r in result.rfis)


def test_agent_still_computes_useful_results_despite_hold():
    """Brief 3.6 hard stop: emit a provisional result + RFI, not a blank refusal."""
    sources = (make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    element = make_element_input(geometry_sources=sources, production=make_production("UNCONFIRMED"))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.resolved_candidate is None  # never presented as an accepted placement
    assert result.illustrative_candidate is not None
    assert len(result.illustrative_candidate.anchors) == 2  # a placement was still computed
    assert result.cog is not None
    assert result.self_weight is not None
    assert result.requires_human_signoff is True
