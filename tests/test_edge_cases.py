from __future__ import annotations

import pytest

from liftagent import engineering, ingest
from liftagent.agent import run_agent
from liftagent.catalogue import get_anchor
from liftagent.ifc_geometry import extract_wc001_geometry
from liftagent.reasoner import MockReasoner
from liftagent.schema import CheckState, ExtractionStatus, Opening, SourceRole, Status
from tests.conftest import WC001_JSON, make_element_input, make_geometry_source, make_production


def _clean_element(**overrides):
    """A single, unambiguous geometry source + confirmed turn method, so
    tests can isolate one specific edge case instead of always hitting the
    geometry-conflict/turn-method HOLDs exercised in test_conflict.py."""
    sources = overrides.pop("geometry_sources", (
        make_geometry_source(length_mm=4700, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    production = overrides.pop("production", make_production("tilting_table"))
    return make_element_input(geometry_sources=sources, production=production, **overrides)


def test_off_centre_cog_shifts_anchors_asymmetrically():
    opening = Opening(id="big", x_mm=3200, width_mm=1200, sill_mm=0, height_mm=2000)
    element = _clean_element(geometry_sources=(
        make_geometry_source(length_mm=4700, openings=(opening,), role=SourceRole.AUTHORITATIVE_DESIGN,
                              name="approval_design"),
    ))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL  # resolved -- clean geometry, confirmed turn+reinforcement
    assert len(result.resolved_candidate.anchors) == 2
    x1, x2 = sorted(a.x_mm for a in result.resolved_candidate.anchors)
    mid = (x1 + x2) / 2
    # mid should track the (off-centre) CoG, not the panel's geometric mid-length (2350mm)
    assert mid == pytest.approx(result.cog.x_mm, abs=1.0)
    assert mid != pytest.approx(2350.0, abs=5.0)


def test_panel_too_thin_for_every_catalogue_anchor_rejects():
    element = _clean_element(geometry_sources=(
        make_geometry_source(length_mm=4700, thickness_mm=100.0, role=SourceRole.AUTHORITATIVE_DESIGN,
                              name="approval_design"),
    ))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.REJECT


def test_missing_turn_method_value_holds():
    element = _clean_element(production=make_production(""))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert any("turn method" in r.message.lower() for r in result.rfis)


def test_unconfirmed_reinforcement_holds_with_unknown_check():
    element = _clean_element()
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=None)
    assert result.status == Status.HOLD
    assert result.reason == "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED"
    assert result.resolved_candidate is None
    reinforcement_checks = [c for c in result.illustrative_candidate.checks if c.check_name == "supplementary_reinforcement"]
    assert reinforcement_checks and reinforcement_checks[-1].state == CheckState.UNKNOWN


def test_explicitly_denied_reinforcement_also_holds():
    element = _clean_element()
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=False)
    assert result.status == Status.HOLD
    assert result.resolved_candidate is None
    reinforcement_checks = [c for c in result.illustrative_candidate.checks if c.check_name == "supplementary_reinforcement"]
    assert reinforcement_checks and reinforcement_checks[-1].state == CheckState.FAIL


def test_capacity_checks_are_flagged_conditional_when_reinforcement_unconfirmed():
    """A bare axial_capacity PASS must never be read as a verified acceptance
    while brief 3.3's Zulage precondition is UNKNOWN -- it should still show
    PASS/FAIL (the number itself is still correct and useful), but flagged
    as conditional so the report can't be misread as an unconditional pass."""
    element = _clean_element()
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=None)
    capacity_checks = [c for c in result.illustrative_candidate.checks if c.check_name == "axial_capacity"]
    assert capacity_checks  # sanity: some capacity checks were actually run
    assert all(c.conditional_on != "" for c in capacity_checks)
    assert any(c.state == CheckState.PASS for c in capacity_checks)  # the raw PASS/FAIL is preserved, not forced to UNKNOWN
    # the overall result must still HOLD regardless of any individual capacity PASS
    assert result.status == Status.HOLD


def test_capacity_checks_are_unconditional_when_reinforcement_confirmed():
    element = _clean_element()
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    capacity_checks = [c for c in result.resolved_candidate.checks if c.check_name == "axial_capacity"]
    assert capacity_checks
    assert all(c.conditional_on == "" for c in capacity_checks)


def test_clean_input_with_everything_confirmed_accepts():
    element = _clean_element()
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.requires_human_signoff is True  # never a final release, even on accept


def test_json_only_run_succeeds_without_ifc(wc001_element):
    result = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    assert result is not None
    assert len(result.geometry_sources) == 2  # approval_design + ifc_export only, no ifc_geometry


def test_ifc_extraction_failure_degrades_gracefully_instead_of_crashing():
    failed_source = extract_wc001_geometry("this/path/does/not/exist.ifc")
    assert failed_source.extraction_status == ExtractionStatus.FAILED
    assert failed_source.length_mm is None

    raw = ingest.load_raw(WC001_JSON)
    element = ingest.build_element_input(raw, ifc_geometry=failed_source)
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result is not None  # did not crash
    # the FAILED source is still listed (for audit) but excluded from self-weight/CoG-by-source
    assert any(s.source_name == "ifc_geometry" and s.extraction_status == ExtractionStatus.FAILED
               for s in result.geometry_sources)
    assert all(sw.source != "ifc_geometry" for sw in result.self_weight_by_source)


def test_mismatched_clutch_check_fails_in_isolation():
    """The hard stop is enforced by construction (the Agent always derives
    clutch from catalogue.py, never accepts an externally supplied one) --
    this unit-tests the underlying deterministic check directly to prove it
    correctly flags a mismatch if it were ever reachable."""
    anchor = get_anchor("ARL-42")
    check = engineering.check_clutch_match(anchor, clutch_used="RU-30")  # wrong clutch for ARL-42
    assert check.state == CheckState.FAIL
    matching = engineering.check_clutch_match(anchor, clutch_used="RU-42")
    assert matching.state == CheckState.PASS
