"""Bounded 'move in' position iteration -- brief §3.5 step 11.

Proves the §3.5 audit's identified gap is closed NARROWLY: when the trial
position (a=0.207L, shifted to CoG) fails edge distance or axis spacing, the
deterministic engineering core attempts exactly one analytically-derived
inward position (engineering.find_feasible_inward_position) before giving up
on that catalogue candidate -- not a general optimiser, not a loop, and
never something the Reasoner can influence (it never sees a position at
all). Catalogue-candidate iteration (trying the next anchor type) remains
completely separate and untouched.

This is NOT about the Reasoner/LLM (see test_reasoner_security.py) and NOT
about the general engineering formulas (see test_end_to_end.py) -- it's
specifically about the new position-search mechanism itself.
"""
from __future__ import annotations

import pytest

from liftagent import catalogue, report
from liftagent.agent import run_agent
from liftagent.reasoner import EvilReasoner, MockReasoner
from liftagent.schema import CheckState, SourceRole, Status
from tests.conftest import make_geometry_source, make_production
from tests.test_end_to_end import _resolved_element, _run_resolved


def _run_at_length(length_mm: float, thickness_mm: float = 180.0, **overrides):
    sources = overrides.pop("geometry_sources", (
        make_geometry_source(length_mm=length_mm, height_mm=3000.0, thickness_mm=thickness_mm,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    element = _resolved_element(geometry_sources=sources, **overrides)
    return run_agent(element, MockReasoner(), reinforcement_confirmed=True)


# --------------------------------------------------------------------------
# TEST A -- the 1200mm case from the §3.5 audit
# --------------------------------------------------------------------------

def test_a_1200mm_case_resolves_via_bounded_move_in():
    """L=1200mm: trial a=0.207*1200~=248.4mm fails ARL-42's (and ARL-52's,
    HAL-TPA-5.0's, ARL-30's) min_edge_mm outright, and the bounded search is
    infeasible for those larger anchors too -- but CFS-WAL-30 (min_edge=300,
    min_axis=600) has exactly one feasible position: moving both anchors in
    to x=300mm/x=900mm satisfies min_edge_mm (300>=300) AND min_axis_mm
    (600>=600) simultaneously. Hand-verified: see the §3.5 follow-up audit."""
    result = _run_at_length(1200.0)
    candidate = result.resolved_candidate
    assert candidate is not None, f"expected ACCEPT_PROVISIONAL, got {result.status}/{result.reason}"
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert candidate.anchor_type == "CFS-WAL-30"

    pi = candidate.position_iteration
    assert pi is not None
    assert pi.attempted is True                                   # initial position was attempted
    assert pi.attempts[0].result == CheckState.FAIL                # ...and it failed
    assert "edge_distance" in pi.attempts[0].failed_checks
    assert len(pi.attempts) >= 2                                    # a fallback position was attempted
    assert pi.attempts[1].result == CheckState.PASS
    assert (pi.selected_x1_mm, pi.selected_x2_mm) != (pi.initial_x1_mm, pi.initial_x2_mm)  # selected != initial
    assert pi.selected_x1_mm == pytest.approx(300.0, abs=0.1)
    assert pi.selected_x2_mm == pytest.approx(900.0, abs=0.1)

    # the FINAL candidate position is the selected one, and it satisfies edge/axis
    assert candidate.x1_mm == pytest.approx(300.0, abs=0.1)
    assert candidate.x2_mm == pytest.approx(900.0, abs=0.1)
    edge_check = next(c for c in candidate.checks if c.check_name == "edge_distance")
    axis_check = next(c for c in candidate.checks if c.check_name == "axis_spacing")
    assert edge_check.state == CheckState.PASS
    assert axis_check.state == CheckState.PASS

    # complete engineering evaluation was performed at the final position --
    # every handling state has both a reaction and a capacity check, and the
    # capacity traces to catalogue.py (never fabricated).
    handling_states_seen = {c.handling_state for c in candidate.checks if c.handling_state}
    assert {"DEMOULD", "STORAGE", "ROAD_TRANSPORT", "ERECTION", "TURN_TILTING_TABLE"} <= handling_states_seen
    axial_checks = [c for c in candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks
    valid_capacities = set(catalogue.get_anchor("CFS-WAL-30").capacity_kn.values())
    assert all(c.capacity in valid_capacities for c in axial_checks)
    assert all(c.state == CheckState.PASS for c in axial_checks)


def test_a_1200mm_case_infeasible_anchors_are_not_silently_accepted():
    """ARL-42 (tried first) has NO feasible position at L=1200mm (its
    min_edge=500 requires moving in further than min_axis=1000 allows) --
    confirm the search correctly reports infeasible for it, rather than
    somehow finding a wrong/fabricated position."""
    result = _run_at_length(1200.0)
    arl42_position_search = [t for t in result.rule_trace
                              if t.step == "position_search" and t.data.get("anchor") == "ARL-42"]
    assert arl42_position_search
    assert arl42_position_search[0].result == "INFEASIBLE"


# --------------------------------------------------------------------------
# TEST B -- no unnecessary iteration
# --------------------------------------------------------------------------

def test_b_normal_passing_case_needs_no_position_iteration():
    result = _run_resolved()
    assert result.status == Status.ACCEPT_PROVISIONAL
    candidate = result.resolved_candidate

    pi = candidate.position_iteration
    assert pi is not None
    assert pi.attempted is False                     # initial position passes -> no iteration needed
    assert pi.attempts == ()
    assert pi.selected_x1_mm is None                   # nothing was "selected" via search

    # selected position equals the original trial+CoG-shift position
    assert candidate.x1_mm == pytest.approx(972.9, abs=0.1)
    assert candidate.x2_mm == pytest.approx(3727.1, abs=0.1)

    # existing Test 8 regression quantities are unchanged by this feature
    assert result.self_weight.value_kn == pytest.approx(59.7547, abs=0.001)
    assert result.cog.x_mm == pytest.approx(2350.0, abs=0.5)
    assert result.cog.y_mm == pytest.approx(1500.0, abs=0.5)
    assert candidate.anchor_type == "ARL-42"
    assert candidate.governing_check.handling_state == "DEMOULD"
    assert candidate.governing_check.anchor_id == "A1"
    assert candidate.governing_check.utilisation == pytest.approx(0.7648, abs=0.001)
    assert result.requires_human_signoff is True


# --------------------------------------------------------------------------
# TEST C -- position iteration cannot bypass hard stops
# --------------------------------------------------------------------------

def test_c_position_iteration_cannot_rescue_a_geometry_conflict(wc001_element):
    result = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    assert result.resolved_candidate is None
    # the illustrative candidate may or may not have needed position iteration --
    # that's irrelevant, because the hard stop is decided independently of it
    assert result.status != Status.ACCEPT_PROVISIONAL


def test_c_position_iteration_cannot_rescue_unconfirmed_turn_method():
    result = _run_at_length(1200.0, production=make_production("UNCONFIRMED"))
    # this is the SAME geometry as TEST A (which resolves to ACCEPT_PROVISIONAL
    # when turn method is confirmed) -- proving the hard stop, not the geometry,
    # is what's blocking acceptance here.
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"
    assert result.resolved_candidate is None


# --------------------------------------------------------------------------
# TEST D -- position search remains deterministic
# --------------------------------------------------------------------------

def test_d_position_search_is_deterministic():
    r1 = _run_at_length(1200.0)
    r2 = _run_at_length(1200.0)
    assert r1.status == r2.status == Status.ACCEPT_PROVISIONAL
    assert (r1.resolved_candidate.x1_mm, r1.resolved_candidate.x2_mm) == \
           (r2.resolved_candidate.x1_mm, r2.resolved_candidate.x2_mm)
    # full structured-report equality -- no randomness, no dict-ordering dependence
    assert report.to_json_dict(r1) == report.to_json_dict(r2)


# --------------------------------------------------------------------------
# TEST E -- position iteration does not break candidate iteration
# --------------------------------------------------------------------------

def test_e_candidate_iteration_still_works_alongside_position_search():
    """Reuses the existing ARL-42-fails-on-capacity / ARL-52-passes fixture
    (test_agent.py / test_agent_workflow.py) -- at L=6000mm/t=220mm, edge and
    axis spacing both pass comfortably for every anchor, so position search
    is never triggered; the ONLY iteration mechanism exercised here is
    catalogue-candidate iteration, proving the new code path doesn't
    interfere with the old one."""
    result = _run_at_length(6000.0, thickness_mm=220.0)
    assert result.status == Status.ACCEPT_PROVISIONAL
    candidate = result.resolved_candidate
    assert candidate.anchor_type == "ARL-52"
    assert candidate.position_iteration.attempted is False  # never needed here

    tried_anchor_types = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchor_types == {"ARL-42", "ARL-52"}  # both candidates were genuinely evaluated


# --------------------------------------------------------------------------
# TEST F -- invalid/no-feasible position
# --------------------------------------------------------------------------

def test_f_no_feasible_position_terminates_cleanly_and_rejects():
    """L=800mm: even CFS-WAL-30 (the most permissive catalogue anchor) has no
    position that clears both min_edge_mm and min_axis_mm simultaneously --
    every candidate's search must terminate (not hang) and report
    infeasible, and the workflow must still reach a final REJECT."""
    result = _run_at_length(800.0)
    assert result.status == Status.REJECT
    assert result.reason == "NO_CANDIDATE_SATISFIES_REQUIREMENTS"
    assert result.resolved_candidate is None

    # every catalogue candidate was tried (the search terminated for each,
    # rather than hanging on any one of them)
    tried_anchor_types = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchor_types == set(catalogue.CANDIDATE_TRY_ORDER)

    position_search_entries = [t for t in result.rule_trace if t.step == "position_search"]
    assert len(position_search_entries) == len(catalogue.CANDIDATE_TRY_ORDER)  # one attempt per candidate
    assert all(t.result == "INFEASIBLE" for t in position_search_entries)

    last = result.illustrative_candidate
    assert last is not None
    assert last.position_iteration.attempted is True
    assert last.position_iteration.selected_x1_mm is None  # no valid position was found


def test_f_evil_reasoner_cannot_exploit_the_infeasible_search_path():
    """An EvilReasoner hallucinating a nonexistent anchor, run against the
    same genuinely-infeasible L=800mm geometry, must still land on the same
    correct REJECT via the same deterministic fallback -- position search
    infeasibility is a deterministic-engine fact, not something a Reasoner
    can talk its way around."""
    element = _resolved_element(geometry_sources=(
        make_geometry_source(length_mm=800.0, height_mm=3000.0, thickness_mm=180.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    result = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.REJECT
    assert result.resolved_candidate is None
