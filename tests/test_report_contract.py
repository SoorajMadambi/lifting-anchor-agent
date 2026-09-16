"""TEST 9 -- Auditable engineering report / output contract.

Proves the final structured report (report.to_json_dict() / AgentResult) and
the human-readable summary (report.render_summary()) carry enough
information for a human engineer to understand, audit, and challenge the
agent's decision -- without relying on the reasoning/explanation layer.
This is about the OUTPUT CONTRACT, not the engineering maths (see
test_end_to_end.py / Test 8) and not the Reasoner boundary (see
test_reasoner_security.py / Test 7, reused directly in 9O below).

Two scenarios throughout:
  A. wc001_result   -- the real WC001 input (JSON + parsed IFC), which
     correctly HOLDs on its own unresolved geometry conflict.
  B. resolved_result -- Test 8's conflict-free fixture, which correctly
     reaches ACCEPT_PROVISIONAL.

No production code was changed to make these tests pass. Two genuine, minor
output-contract gaps were discovered during inspection and are called out at
the bottom of this file (and reported separately) rather than silently
patched: (1) AgentResult carries no field for the concrete density/class
actually used in a run, and (2) agent.py's ASSUMPTIONS text is a static
module-level constant that names "C32/40" unconditionally, even for a run
whose actual input concrete class is different.
"""
from __future__ import annotations

import json

import pytest

from liftagent import catalogue, engineering, ingest, report
from liftagent.agent import run_agent
from liftagent.reasoner import EvilReasoner, MockReasoner
from liftagent.schema import CheckState, Status
from tests.conftest import WC001_IFC, WC001_JSON, make_concrete
from tests.test_end_to_end import _resolved_element, _run_resolved


@pytest.fixture
def wc001_result():
    """The real WC001 input, with all three geometry sources (JSON + the
    parsed IFC file) -- exactly what `liftagent run data/wc001.json --ifc
    data/ifc/WC001.ifc` produces. Skips gracefully if ifcopenshell isn't
    installed, matching test_ifc_geometry.py's existing convention."""
    pytest.importorskip("ifcopenshell", reason="ifcopenshell not installed")
    from liftagent.ifc_geometry import extract_wc001_geometry

    raw = ingest.load_raw(WC001_JSON)
    ifc_source = extract_wc001_geometry(str(WC001_IFC))
    element = ingest.build_element_input(raw, ifc_geometry=ifc_source)
    return run_agent(element, MockReasoner())  # reinforcement_confirmed left None -- the honest, real WC001 case


@pytest.fixture
def resolved_result():
    return _run_resolved()


def _assert_capacity_check_is_fully_auditable(c):
    """The 7 audit questions from Test 9I, all answerable from one CheckResult."""
    assert c.demand is not None                                              # 1. demand
    assert c.capacity is not None                                            # 2. capacity used
    assert c.anchor_id in ("A1", "A2")                                       # 4. which anchor
    assert c.handling_state != ""                                            # (needed for Q3, below)
    assert c.utilisation == pytest.approx(c.demand / c.capacity, rel=1e-9)   # 5. utilisation
    assert c.state in (CheckState.PASS, CheckState.FAIL, CheckState.UNKNOWN)  # 6. pass/fail
    assert isinstance(c.conditional_on, str)                                 # 7. conditional-on-another-check


# --------------------------------------------------------------------------
# 9A -- top-level decision
# --------------------------------------------------------------------------

def test_9a_wc001_top_level_decision(wc001_result):
    assert wc001_result.element_id == "WC001"
    assert wc001_result.status == Status.HOLD
    assert wc001_result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    assert wc001_result.requires_human_signoff is True


def test_9a_resolved_top_level_decision(resolved_result):
    assert resolved_result.status == Status.ACCEPT_PROVISIONAL
    assert resolved_result.reason is None
    assert resolved_result.requires_human_signoff is True
    # ACCEPT_PROVISIONAL can never be mistaken for final sign-off: the field
    # that would represent that (requires_human_signoff) is explicitly True,
    # and there is no separate "final"/"approved" status value anywhere in
    # the Status enum for a caller to confuse this with.
    assert "FINAL" not in {s.value for s in Status}
    assert "APPROVED" not in {s.value for s in Status}


# --------------------------------------------------------------------------
# 9B -- geometry provenance (structured, not string-matched)
# --------------------------------------------------------------------------

def test_9b_geometry_provenance_structured(wc001_result):
    by_name = {s.source_name: s for s in wc001_result.geometry_sources}
    assert {"approval_design", "ifc_export", "ifc_geometry"} <= set(by_name)

    for name, s in by_name.items():
        assert s.source_type is not None       # source identifier / kind
        assert s.role is not None               # provenance/role (AUTHORITATIVE_DESIGN/EXPORT/DERIVED_REFERENCE)
        assert s.extraction_status is not None
        if s.extraction_status.value == "SUCCESS":
            assert s.length_mm is not None and s.height_mm is not None and s.thickness_mm is not None

    # the conflict itself, as STRUCTURED data: two sources genuinely
    # disagree on length -- not inferred from any rendered string
    lengths = {s.source_name: s.length_mm for s in wc001_result.geometry_sources if s.length_mm is not None}
    assert len(set(lengths.values())) > 1
    assert lengths["approval_design"] != lengths["ifc_export"]

    geometry_rfis = [r for r in wc001_result.rfis if r.id == "RFI-GEOMETRY"]
    assert geometry_rfis  # the conflict is also surfaced as a structured, identifiable RFI


# --------------------------------------------------------------------------
# 9C -- selected geometry / fail-closed behaviour
# --------------------------------------------------------------------------

def test_9c_wc001_never_silently_selects_a_geometry(wc001_result):
    assert wc001_result.selected_geometry is None
    assert wc001_result.resolved_candidate is None
    # neither the 4700mm nor the 4300mm reading is silently promoted to "the" geometry
    lengths = {s.length_mm for s in wc001_result.geometry_sources if s.length_mm is not None}
    assert 4700.0 in lengths or any(abs(l - 4700.0) < 1 for l in lengths)
    assert 4300.0 in lengths


def test_9c_resolved_fixture_has_explicit_selected_geometry(resolved_result):
    assert resolved_result.selected_geometry is not None
    matching = [s for s in resolved_result.geometry_sources if s.source_name == resolved_result.selected_geometry]
    assert len(matching) == 1
    assert matching[0].role.value == "AUTHORITATIVE_DESIGN"  # explicit provenance, not just a bare name


# --------------------------------------------------------------------------
# 9D -- self-weight audit trail (provenance, not just the number)
# --------------------------------------------------------------------------

def test_9d_self_weight_audit_trail_wc001(wc001_result):
    sw = wc001_result.self_weight
    assert sw is not None
    assert sw.value_kn > 0
    assert sw.method == "geometry_volume_x_density"   # calculation method
    assert sw.source != ""                             # source geometry it was computed from
    assert sw.reference_value_kn == 72.6                # the brief's figure is retained, distinctly...
    assert sw.value_kn != pytest.approx(72.6, abs=1.0)  # ...and never silently substituted as the computed value
    assert sw.warning != ""                             # the discrepancy is explicitly surfaced
    assert sw.discrepancy_kn == pytest.approx(sw.value_kn - sw.reference_value_kn, rel=1e-9)

    # (Test 9 follow-up) the actual calculation INPUTS are now auditable too,
    # not just the method name and output value.
    assert sw.density_kg_per_m3 == 2400.0
    assert sw.concrete_class == "C32/40"

    # per-source table gives further audit granularity beyond the single headline number
    named = {s.source for s in wc001_result.self_weight_by_source}
    assert {"approval_design", "ifc_export"} <= named

    reconciliation_rfis = [r for r in wc001_result.rfis if r.id == "RFI-SELF-WEIGHT"]
    assert reconciliation_rfis


def test_9d_self_weight_audit_trail_resolved(resolved_result):
    sw = resolved_result.self_weight
    assert sw.value_kn > 0
    assert sw.method == "geometry_volume_x_density"
    assert sw.source != ""
    assert sw.reference_value_kn == 72.6
    # (Test 9 follow-up) actual density/concrete-class inputs for the resolved fixture
    assert sw.density_kg_per_m3 == 2400.0
    assert sw.concrete_class == "C32/40"


def test_9d_self_weight_report_json_exposes_density_and_concrete_class(resolved_result):
    """Issue 2: expose density_kg_per_m3 and concrete_class in the
    structured (JSON-serialisable) self_weight report, not just on the
    dataclass -- this is what an external auditor actually reads."""
    as_dict = report.to_json_dict(resolved_result)
    sw_dict = as_dict["self_weight"]
    assert sw_dict["density_kg_per_m3"] == 2400.0
    assert sw_dict["concrete_class"] == "C32/40"


# --------------------------------------------------------------------------
# 9E -- CoG audit trail
# --------------------------------------------------------------------------

def test_9e_cog_audit_trail_wc001(wc001_result):
    cog = wc001_result.cog
    assert cog is not None
    assert cog.x_mm is not None and cog.y_mm is not None
    assert cog.source != ""
    assert cog.method != ""
    assert "curtain" in cog.note.lower() or "bias" in cog.note.lower()  # qualitative one-sided-curtain note

    # source-specific CoG remains available (per source, for audit/comparison)
    by_source = {c.source for c in wc001_result.cog_by_source}
    assert {"approval_design", "ifc_export"} <= by_source


def test_9e_cog_audit_trail_resolved(resolved_result):
    cog = resolved_result.cog
    assert cog.x_mm == pytest.approx(2350.0, abs=0.5)
    assert cog.y_mm == pytest.approx(1500.0, abs=0.5)
    assert cog.source != ""


# --------------------------------------------------------------------------
# 9F -- anchor placement
# --------------------------------------------------------------------------

def test_9f_wc001_illustrative_placement_is_clearly_labelled_not_resolved(wc001_result):
    assert wc001_result.resolved_candidate is None
    c = wc001_result.illustrative_candidate
    assert c is not None
    assert c.candidate_id and c.anchor_type and c.clutch
    assert len(c.anchors) == 2
    x1, x2 = sorted(a.x_mm for a in c.anchors)
    assert (x2 - x1) > 0  # spacing is derivable/present
    edge_checks = [ch for ch in c.checks if ch.check_name == "edge_distance"]
    assert edge_checks  # edge distances are present (via the checks, per Test Design Principle 5)
    midpoint = (x1 + x2) / 2.0
    assert midpoint == pytest.approx(wc001_result.cog.x_mm, abs=1.0)  # CoG relationship is present/verifiable

    # the mutually-exclusive resolved/illustrative split on AgentResult IS the
    # "trial/illustrative/resolved" flag -- there is no separate placement
    # this WC001 result contains that could be mistaken for a final one.
    assert wc001_result.resolved_candidate is None
    assert wc001_result.illustrative_candidate is c


def test_9f_resolved_placement_is_explicitly_the_resolved_candidate(resolved_result):
    assert resolved_result.illustrative_candidate is None
    c = resolved_result.resolved_candidate
    assert c is not None
    assert len(c.anchors) == 2
    assert c.x1_mm < c.x2_mm


# --------------------------------------------------------------------------
# 9G -- rig decision
# --------------------------------------------------------------------------

def test_9g_rig_decision_structured(wc001_result, resolved_result):
    for result, candidate in (
        (wc001_result, wc001_result.illustrative_candidate),
        (resolved_result, resolved_result.resolved_candidate),
    ):
        assert candidate.rig is not None
        assert candidate.rig.spreader_required is True  # 180mm panel vs every catalogue anchor's transverse min-wall
        assert candidate.rig.reason != ""
        assert "180" in candidate.rig.reason  # the panel/anchor constraint is named, not just asserted


# --------------------------------------------------------------------------
# 9H -- handling states
# --------------------------------------------------------------------------

def test_9h_all_handling_states_represented_wc001(wc001_result):
    """WC001's turn_method is UNCONFIRMED -> both TURN_TILTING_TABLE and the
    conservative TURN_FREE_CRANE must both still be represented."""
    candidate = wc001_result.illustrative_candidate
    states_in_checks = {c.handling_state for c in candidate.checks if c.handling_state}
    assert {"DEMOULD", "STORAGE", "ROAD_TRANSPORT", "ERECTION", "TURN_TILTING_TABLE", "TURN_FREE_CRANE"} <= states_in_checks

    states_in_trace = {t.step.removeprefix("load_") for t in wc001_result.rule_trace if t.step.startswith("load_")}
    assert {"DEMOULD", "STORAGE", "ROAD_TRANSPORT", "ERECTION", "TURN_TILTING_TABLE", "TURN_FREE_CRANE"} <= states_in_trace

    for state in states_in_checks:
        state_checks = [c for c in candidate.checks if c.handling_state == state]
        assert any(c.check_name == "reaction_nonnegative" for c in state_checks)  # reactions
        assert any(c.check_name == "axial_capacity" for c in state_checks)        # capacity check + utilisation + status


def test_9h_all_handling_states_represented_resolved(resolved_result):
    candidate = resolved_result.resolved_candidate
    states_in_checks = {c.handling_state for c in candidate.checks if c.handling_state}
    assert {"DEMOULD", "STORAGE", "ROAD_TRANSPORT", "ERECTION", "TURN_TILTING_TABLE"} <= states_in_checks


# --------------------------------------------------------------------------
# 9I -- capacity / utilisation auditability
# --------------------------------------------------------------------------

def test_9i_resolved_capacity_check_is_fully_auditable_and_unconditional(resolved_result):
    gc = resolved_result.resolved_candidate.governing_check
    _assert_capacity_check_is_fully_auditable(gc)
    assert gc.conditional_on == ""  # reinforcement confirmed for this fixture -> not conditional

    # Q3 (which strength column) is provably recoverable: only real catalogue
    # columns can reproduce the reported capacity for this anchor.
    matching_columns = [mpa for mpa in (15, 25, 35)
                         if catalogue.lookup_capacity(resolved_result.resolved_candidate.anchor_type, mpa) == gc.capacity]
    assert matching_columns


def test_9i_wc001_capacity_check_remains_conditional(wc001_result):
    gc = wc001_result.illustrative_candidate.governing_check
    _assert_capacity_check_is_fully_auditable(gc)
    assert gc.conditional_on != ""  # reinforcement UNKNOWN -> capacity NOT presented as verified
    assert "reinforcement" in gc.conditional_on.lower()


# --------------------------------------------------------------------------
# 9J -- governing check
# --------------------------------------------------------------------------

def test_9j_governing_check_matches_the_deterministic_maximum(resolved_result):
    candidate = resolved_result.resolved_candidate
    axial_checks = [c for c in candidate.checks if c.check_name == "axial_capacity" and c.utilisation is not None]
    max_utilisation = max(c.utilisation for c in axial_checks)

    gc = candidate.governing_check
    assert gc is not None
    assert gc.governing is True
    assert gc.demand is not None and gc.capacity is not None and gc.utilisation is not None
    assert gc.anchor_id in ("A1", "A2")
    assert gc.utilisation == pytest.approx(max_utilisation, rel=1e-9)

    # do not over-constrain in case of a tie -- the governing check must be
    # ONE of the checks tied at the maximum, not necessarily a specific one
    tied = [c for c in axial_checks if abs(c.utilisation - max_utilisation) < 1e-9]
    assert any(c.handling_state == gc.handling_state and c.anchor_id == gc.anchor_id for c in tied)


# --------------------------------------------------------------------------
# 9K -- RFIs (semantic presence via stable id, not brittle string equality)
# --------------------------------------------------------------------------

def test_9k_wc001_rfi_categories_present(wc001_result):
    rfi_ids = {r.id for r in wc001_result.rfis}
    assert {"RFI-GEOMETRY", "RFI-TURN-METHOD", "RFI-SUPPLEMENTARY-REINFORCEMENT", "RFI-SELF-WEIGHT"} <= rfi_ids
    for r in wc001_result.rfis:
        assert r.message != ""
        assert r.rule_reference != ""  # every RFI cites a rule, not just a bare claim


# --------------------------------------------------------------------------
# 9L -- rule trace (key engineering decision stages, not every call)
# --------------------------------------------------------------------------

def test_9l_rule_trace_covers_the_key_decision_stages(resolved_result):
    steps = {t.step for t in resolved_result.rule_trace}
    assert "self_weight" in steps
    assert "cog" in steps
    assert "anchor_trial" in steps
    assert "anchor_shift_to_cog" in steps
    assert {"min_wall_axial", "edge_distance", "axis_spacing"} <= steps      # geometric constraints
    assert "rig_decision" in steps
    assert "reactions" in steps
    assert any(s.startswith("load_") for s in steps)                        # handling-state loads
    assert any(s.startswith("capacity_") for s in steps)                    # capacity checks
    assert "strength_column" in steps                                       # concrete-strength selection

    # "candidate outcome" and "final invariant guard" are represented at the
    # top level of the report (status/reason), not as their own trace
    # entries -- enforce_invariants() is a pure function with no ToolBox
    # calls to log, so this is the correct/expected place to look for them.
    assert resolved_result.status == Status.ACCEPT_PROVISIONAL
    assert resolved_result.reason is None


# --------------------------------------------------------------------------
# 9M -- assumptions
# --------------------------------------------------------------------------

def test_9m_key_assumptions_are_visible(resolved_result):
    text = " ".join(resolved_result.assumptions).lower()
    assert "cube-strength" in text or "cube strength" in text     # concrete strength interpretation
    assert "72.6" in " ".join(resolved_result.assumptions)         # density/self-weight reference reconciliation
    assert "spreader" in text or "vertical sling" in text          # rig/spreader assumption

    # the one-sided curtain treatment is represented, but on CogResult.note
    # rather than in the assumptions list -- per Test Design Principle 5,
    # test the existing representation instead of assuming it's duplicated
    # into `assumptions` too.
    assert "curtain" in resolved_result.cog.note.lower() or "bias" in resolved_result.cog.note.lower()


def test_9m_wc001_unresolved_turn_method_is_visible(wc001_result):
    rfi_messages = " ".join(r.message for r in wc001_result.rfis).lower()
    assert "turn method" in rfi_messages and "unconfirmed" in rfi_messages


def test_9m_concrete_strength_assumption_describes_c32_40():
    """Issue 1 regression: the resolved fixture's actual class is C32/40 ->
    the assumption must name C32/40 and its 40 MPa cube strength."""
    result = _run_resolved()
    assert result.self_weight.concrete_class == "C32/40"
    strength_assumption = next(a for a in result.assumptions if "cube-strength" in a.lower())
    assert "C32/40" in strength_assumption
    assert "40 MPa" in strength_assumption


def test_9m_concrete_strength_assumption_describes_c25_30_not_c32_40():
    """Issue 1 regression: a DIFFERENT input concrete class (C25/30) must
    produce an assumption naming C25/30 and its 30 MPa cube strength -- and
    must NOT still claim C32/40, which was the bug Test 9 discovered."""
    result = _run_resolved(concrete=make_concrete("C25/30", density=2400.0, first_lift=15.0))
    assert result.status == Status.ACCEPT_PROVISIONAL  # sanity: this run still resolves cleanly
    assert result.self_weight.concrete_class == "C25/30"

    strength_assumption = next(a for a in result.assumptions if "cube-strength" in a.lower())
    assert "C25/30" in strength_assumption
    assert "30 MPa" in strength_assumption
    assert "C25/30 is interpreted" in strength_assumption  # describes THIS run, not the WC001 default
    assert "C32/40 is interpreted" not in strength_assumption


# --------------------------------------------------------------------------
# 9N -- human sign-off, for BOTH outcomes
# --------------------------------------------------------------------------

def test_9n_human_signoff_required_for_hold(wc001_result):
    assert wc001_result.requires_human_signoff is True


def test_9n_human_signoff_required_for_accept_provisional(resolved_result):
    assert resolved_result.requires_human_signoff is True


# --------------------------------------------------------------------------
# 9O -- no hidden Reasoner authority (reuses Test 7's evidence, at the report level)
# --------------------------------------------------------------------------

def test_9o_report_identical_regardless_of_reasoner_resolved_case():
    mock_result = _run_resolved()
    evil_result = run_agent(_resolved_element(), EvilReasoner(), reinforcement_confirmed=True)
    assert report.to_json_dict(mock_result) == report.to_json_dict(evil_result)


def test_9o_report_identical_regardless_of_reasoner_wc001_case(wc001_result):
    from liftagent.ifc_geometry import extract_wc001_geometry
    raw = ingest.load_raw(WC001_JSON)
    ifc_source = extract_wc001_geometry(str(WC001_IFC))
    element = ingest.build_element_input(raw, ifc_geometry=ifc_source)
    evil_result = run_agent(element, EvilReasoner())
    assert report.to_json_dict(wc001_result) == report.to_json_dict(evil_result)


# --------------------------------------------------------------------------
# 9P -- JSON serialisation contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["wc001_result", "resolved_result"])
def test_9p_json_serialisable_and_round_trips_top_level_fields(request, fixture_name):
    result = request.getfixturevalue(fixture_name)
    as_dict = report.to_json_dict(result)
    text = json.dumps(as_dict)  # must not raise
    round_tripped = json.loads(text)

    required_top_level = {
        "element_id", "status", "reason", "geometry_sources", "selected_geometry",
        "cog", "self_weight", "resolved_candidate", "illustrative_candidate",
        "rfis", "rule_trace", "assumptions", "requires_human_signoff",
    }
    assert required_top_level <= set(round_tripped)
    assert round_tripped["element_id"] == as_dict["element_id"]
    assert round_tripped["status"] == as_dict["status"]


# --------------------------------------------------------------------------
# 9Q -- human-readable summary
# --------------------------------------------------------------------------

def test_9q_render_summary_wc001(wc001_result):
    text = report.render_summary(wc001_result)
    assert "WC001" in text
    assert "HOLD" in text
    assert "UNRESOLVED_GEOMETRY_CONFLICT" in text or "Geometry conflict" in text
    assert "ILLUSTRATIVE CANDIDATE" in text and "NOT A RESOLVED PLACEMENT" in text
    # "final" legitimately appears only in disclaiming/negating contexts (e.g.
    # "NOT used for final acceptance", "before finalising placement") -- what
    # must never appear is an affirmative claim of approval/sign-off.
    assert "APPROVED" not in text.upper()
    assert "SIGNED OFF" not in text.upper()
    assert "not used for final acceptance" in text.lower()
    assert "Human sign-off required: YES" in text


def test_9q_render_summary_resolved(resolved_result):
    text = report.render_summary(resolved_result)
    assert "ACCEPT_PROVISIONAL" in text
    assert "Human sign-off required: YES" in text
    assert "APPROVED" not in text.upper()


# --------------------------------------------------------------------------
# 9R -- no false certainty
# --------------------------------------------------------------------------

def test_9r_wc001_cannot_be_read_as_resolved_approved_or_verified(wc001_result):
    # geometry NOT resolved
    assert wc001_result.selected_geometry is None
    # placement NOT approved
    assert wc001_result.resolved_candidate is None
    assert wc001_result.illustrative_candidate is not None  # still shown, but not as an accepted placement
    # supplementary reinforcement NOT verified
    reinforcement_checks = [c for c in wc001_result.illustrative_candidate.checks
                             if c.check_name == "supplementary_reinforcement"]
    assert reinforcement_checks and reinforcement_checks[-1].state == CheckState.UNKNOWN
    # capacity NOT presented as unconditionally verified while reinforcement is UNKNOWN
    axial_checks = [c for c in wc001_result.illustrative_candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks and all(c.conditional_on != "" for c in axial_checks)
    # final engineering sign-off NOT complete
    assert wc001_result.status != Status.ACCEPT_PROVISIONAL
    assert wc001_result.requires_human_signoff is True
