"""Bounded 'move in' position iteration -- brief §3.5 step 11.

Proves the §3.5 audit's identified gap is closed NARROWLY: when the trial
position (a=0.207L, shifted to CoG) fails any position-dependent check --
edge distance, axis spacing, or opening-void clash -- the deterministic
engineering core attempts exactly one analytically-derived inward position
(engineering.find_feasible_inward_position) before giving up on that
catalogue candidate -- not a general optimiser, not a loop, and never
something the Reasoner can influence (it never sees a position at all).
Catalogue-candidate iteration (trying the next anchor type) remains
completely separate and untouched.

A second, later F3 audit (see DESIGN_NOTE.md) established that
reaction_nonnegative/axial_capacity are algebraically INVARIANT to spacing
as long as the pair stays CoG-centered (which every placement here always
is) -- so this bounded fallback is only ever able to affect the three
position-dependent checks above, never capacity/reaction failures. TEST G/H
below cover that widened trigger and pin the invariance as intended
behaviour, not a gap.

A THIRD audit found that even the widened fallback still returned only the
minimum edge/axis-feasible position, with no opening-awareness at all --
so a fallback that itself landed inside an opening was rejected outright,
even when other feasible positions existed further along the same
edge/axis-feasible range. `find_feasible_inward_position()` now derives
that full feasible range and an opening-aware position within it
analytically (closed-form interval math, still no search/optimiser, still
at most one candidate position per anchor) -- see
test_engineering.py's `test_interval_*` tests for the formula-level proof,
and TEST K below for the same fix exercised end-to-end through a real
catalogue anchor and the full agent loop.

This is NOT about the Reasoner/LLM (see test_reasoner_security.py) and NOT
about the general engineering formulas (see test_end_to_end.py) -- it's
specifically about the new position-search mechanism itself.
"""
from __future__ import annotations

import pytest

from liftagent import catalogue, report
from liftagent.agent import run_agent
from liftagent.reasoner import EvilReasoner, MockReasoner
from liftagent.schema import CheckState, Opening, SourceRole, Status
from tests.conftest import make_geometry_source, make_production
from tests.test_agent_workflow import _AlternateDeterministicReasoner
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


# --------------------------------------------------------------------------
# TEST G -- F3 gap: opening-void clash must be able to trigger the fallback
# --------------------------------------------------------------------------

# L=4700mm, H=3000mm, t=180mm (matches the brief's WC001 dimensions). A
# single opening reaching the top edge sits directly under where ARL-42's
# trial position (a=0.207L, shifted to CoG) lands, but clear of where the
# bounded "move in" fallback lands -- hand-verified against the actual
# engineering functions (not asserted blind): trial lo=988.4mm sits inside
# the opening's [800,1200]mm x-range; the fallback moves to lo=531.0mm,
# clear of it, while edge_distance (500.0>=500) and axis_spacing both still
# pass there.
_OPENING_UNDER_TRIAL_POSITION = Opening(id="O1", x_mm=800.0, width_mm=400.0, sill_mm=2600.0, height_mm=400.0)


def _run_with_opening_under_trial_position():
    sources = (make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
                                     openings=(_OPENING_UNDER_TRIAL_POSITION,),
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    element = _resolved_element(geometry_sources=sources)
    return run_agent(element, MockReasoner(), reinforcement_confirmed=True)


def test_g_opening_clash_at_trial_position_triggers_placement_fallback():
    """Before this fix, opening_void_clash could never trigger the bounded
    'move in' search (only edge_distance/axis_spacing could) -- a candidate
    that cleared edge/axis but landed inside an opening was never retried.
    This is the concrete F3 gap the second audit identified."""
    result = _run_with_opening_under_trial_position()
    candidate = result.resolved_candidate
    assert candidate is not None, f"expected ACCEPT_PROVISIONAL, got {result.status}/{result.reason}"
    assert candidate.anchor_type == "ARL-42"

    pi = candidate.position_iteration
    assert pi is not None
    assert pi.attempted is True
    assert len(pi.attempts) == 2                                # bounded: trial + exactly one fallback

    trial_attempt, fallback_attempt = pi.attempts
    assert trial_attempt.result == CheckState.FAIL
    assert trial_attempt.failed_checks == ("opening_void_clash",)  # edge/axis passed at trial
    assert fallback_attempt.result == CheckState.PASS
    assert fallback_attempt.failed_checks == ()

    # the fallback position was actually selected
    assert (candidate.x1_mm, candidate.x2_mm) == (pi.selected_x1_mm, pi.selected_x2_mm)
    assert (candidate.x1_mm, candidate.x2_mm) != (pi.initial_x1_mm, pi.initial_x2_mm)

    # the final candidate proceeded through the normal downstream checks --
    # every handling state has a capacity check tracing to catalogue.py.
    axial_checks = [c for c in candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks
    valid_capacities = set(catalogue.get_anchor("ARL-42").capacity_kn.values())
    assert all(c.capacity in valid_capacities for c in axial_checks)

    # plumb is preserved at the selected position, same as the edge/axis case
    assert (candidate.x1_mm + candidate.x2_mm) / 2 == pytest.approx(result.cog.x_mm, abs=0.01)

    # only ONE opening_void_clash CheckResult survives on the final
    # candidate (from the selected position) -- not two, even though it was
    # evaluated twice internally.
    clash_checks = [c for c in candidate.checks if c.check_name == "opening_void_clash"]
    assert len(clash_checks) == 1
    assert clash_checks[0].state == CheckState.PASS


def test_g_opening_clash_search_is_deterministic():
    r1 = _run_with_opening_under_trial_position()
    r2 = _run_with_opening_under_trial_position()
    assert r1.status == r2.status == Status.ACCEPT_PROVISIONAL
    assert report.to_json_dict(r1) == report.to_json_dict(r2)


# --------------------------------------------------------------------------
# TEST H -- capacity/reaction failures never trigger placement iteration
# (the CoG-centered two-anchor statics model makes reactions invariant to
# spacing, so this is intentional, not a gap -- see DESIGN_NOTE.md)
# --------------------------------------------------------------------------

def test_h_capacity_only_failure_does_not_trigger_a_placement_fallback():
    """L=6000mm/t=220mm (the existing ARL-42-fails-on-capacity fixture, TEST
    E): ARL-42's trial position clears every position-dependent check
    (edge/axis/opening) but fails axial_capacity outright. Confirm NO
    position_search was ever logged for ARL-42 -- the fallback is never
    invoked for a check it structurally cannot fix."""
    result = _run_at_length(6000.0, thickness_mm=220.0)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.resolved_candidate.anchor_type == "ARL-52"  # ARL-42 was tried and failed first

    arl42_capacity_fails = [t for t in result.rule_trace
                             if t.step.startswith("capacity_") and t.result == "FAIL"]
    assert arl42_capacity_fails  # confirms ARL-42 genuinely failed on capacity, not geometry

    arl42_position_search = [t for t in result.rule_trace
                              if t.step == "position_search" and t.data.get("anchor") == "ARL-42"]
    assert arl42_position_search == []  # the fallback was never even attempted for ARL-42


# --------------------------------------------------------------------------
# TEST I -- the bounded fallback never exceeds two placement attempts per
# anchor candidate, across every scenario in this file
# --------------------------------------------------------------------------

def test_i_no_anchor_candidate_ever_generates_more_than_two_placement_attempts():
    scenarios = [
        _run_at_length(1200.0),                          # TEST A: geometry-only fallback (edge/axis)
        _run_at_length(800.0),                            # TEST F: infeasible fallback
        _run_with_opening_under_trial_position(),          # TEST G: opening-clash fallback
        _run_at_length(6000.0, thickness_mm=220.0),        # TEST H: capacity-only, no fallback at all
    ]
    for result in scenarios:
        attempts_by_anchor: dict[str, int] = {}
        for entry in result.rule_trace:
            if entry.step == "edge_distance":
                anchor = entry.data.get("anchor")
                attempts_by_anchor[anchor] = attempts_by_anchor.get(anchor, 0) + 1
        assert attempts_by_anchor, "expected at least one edge_distance trace entry"
        assert all(count <= 2 for count in attempts_by_anchor.values()), attempts_by_anchor


# --------------------------------------------------------------------------
# TEST J -- placement iteration is reasoner-independent (determinism
# boundary): the Reasoner is consulted only to pick an anchor TYPE
# (reasoner.py); position generation, feasibility checks, and selection
# live entirely in engineering.py/agent.py and never consult it. For a
# given chosen anchor, the entire placement-iteration record -- ordered
# attempts, selected position, and the full structured report -- must be
# byte-identical no matter which Reasoner implementation chose it.
# --------------------------------------------------------------------------

def test_j_placement_iteration_is_identical_across_independent_reasoners():
    """MockReasoner, EvilReasoner (whose hallucinated anchor is rejected by
    validator.py and falls back to the same deterministic try-order), and
    _AlternateDeterministicReasoner (test_agent_workflow.py -- an
    independent implementation with unrelated wording) all resolve to the
    same first anchor type for both fixtures below. Covers both fallback
    triggers: geometry-only (1200mm, TEST A) and opening-clash (TEST G)."""
    element_geometry_fallback = _resolved_element(geometry_sources=(
        make_geometry_source(length_mm=1200.0, height_mm=3000.0, thickness_mm=180.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    element_opening_fallback = _resolved_element(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
                              openings=(_OPENING_UNDER_TRIAL_POSITION,),
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))

    for element in (element_geometry_fallback, element_opening_fallback):
        a = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
        b = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
        c = run_agent(element, _AlternateDeterministicReasoner(), reinforcement_confirmed=True)

        assert a.status == Status.ACCEPT_PROVISIONAL
        assert a.resolved_candidate.anchor_type == b.resolved_candidate.anchor_type == \
               c.resolved_candidate.anchor_type

        # the placement-iteration record itself -- ordered attempts, selected
        # position -- not just the final engineering numbers
        assert a.resolved_candidate.position_iteration == b.resolved_candidate.position_iteration
        assert a.resolved_candidate.position_iteration == c.resolved_candidate.position_iteration

        # the full structured report is byte-identical -- AgentResult carries
        # no Reasoner-authored text at all (explain() is a separate, optional
        # call never folded into the report; see test_11g in
        # test_agent_workflow.py for the same property on the wider result).
        report_a, report_b, report_c = report.to_json_dict(a), report.to_json_dict(b), report.to_json_dict(c)
        assert report_a == report_b == report_c


# --------------------------------------------------------------------------
# TEST K -- the opening-aware interval fix, exercised end-to-end (3rd F3
# audit): a real catalogue anchor (ARL-42) whose SINGLE bounded fallback
# position would itself land inside an opening under the OLD single-point
# formula, but the interval search finds a further, opening-clear position
# within the same edge/axis-feasible range -- still exactly one fallback
# attempt at the agent level, still re-verified by the real checks.
# --------------------------------------------------------------------------

# L=4700mm/H=3000mm/t=180mm (WC001 scale). O1 makes ARL-42's TRIAL position
# clash (same opening as TEST G); O2 is a second, small opening placed
# exactly straddling the OLD fallback formula's answer (hand-verified
# against the real engineering functions, not asserted blind) -- forcing
# the interval search to find a position further along the feasible range.
_OPENING_BLOCKS_TRIAL = Opening(id="O1", x_mm=800.0, width_mm=400.0, sill_mm=2600.0, height_mm=400.0)
_OPENING_BLOCKS_OLD_FALLBACK = Opening(id="O2", x_mm=505.49497847919656, width_mm=20.0,
                                        sill_mm=2600.0, height_mm=400.0)


def _run_with_opening_blocking_both_trial_and_old_fallback():
    sources = (make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
                                     openings=(_OPENING_BLOCKS_TRIAL, _OPENING_BLOCKS_OLD_FALLBACK),
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    element = _resolved_element(geometry_sources=sources)
    return run_agent(element, MockReasoner(), reinforcement_confirmed=True)


def test_k_interval_search_finds_a_position_past_the_old_single_point_fallback():
    result = _run_with_opening_blocking_both_trial_and_old_fallback()
    candidate = result.resolved_candidate
    assert candidate is not None, f"expected ACCEPT_PROVISIONAL, got {result.status}/{result.reason}"
    assert candidate.anchor_type == "ARL-42"

    pi = candidate.position_iteration
    assert pi is not None
    assert pi.attempted is True
    assert len(pi.attempts) == 2  # still bounded: trial + exactly one (now opening-aware) fallback

    trial_attempt, fallback_attempt = pi.attempts
    assert trial_attempt.result == CheckState.FAIL
    assert fallback_attempt.result == CheckState.PASS
    assert fallback_attempt.failed_checks == ()
    # the fallback position is NOT the old single-point answer (515.49...) --
    # the interval search moved past the second opening too
    assert fallback_attempt.x1_mm > 525.0

    # re-verified by the real, unchanged checks -- never trusted from the
    # interval math alone
    edge_check = next(c for c in candidate.checks if c.check_name == "edge_distance")
    axis_check = next(c for c in candidate.checks if c.check_name == "axis_spacing")
    clash_check = next(c for c in candidate.checks if c.check_name == "opening_void_clash")
    assert edge_check.state == axis_check.state == clash_check.state == CheckState.PASS

    # plumb preserved at the selected position
    assert (candidate.x1_mm + candidate.x2_mm) / 2 == pytest.approx(result.cog.x_mm, abs=1e-6)


def test_k_interval_search_is_identical_across_independent_reasoners():
    """Same reasoner-independence property as TEST J, specifically for the
    interval-derived (not single-point) fallback."""
    element = _resolved_element(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
                              openings=(_OPENING_BLOCKS_TRIAL, _OPENING_BLOCKS_OLD_FALLBACK),
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    a = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    b = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
    c = run_agent(element, _AlternateDeterministicReasoner(), reinforcement_confirmed=True)

    assert a.status == Status.ACCEPT_PROVISIONAL
    assert a.resolved_candidate.position_iteration == b.resolved_candidate.position_iteration
    assert a.resolved_candidate.position_iteration == c.resolved_candidate.position_iteration
    assert report.to_json_dict(a) == report.to_json_dict(b) == report.to_json_dict(c)


def test_k_interval_search_result_is_deterministic_across_repeated_runs():
    r1 = _run_with_opening_blocking_both_trial_and_old_fallback()
    r2 = _run_with_opening_blocking_both_trial_and_old_fallback()
    assert r1.status == r2.status == Status.ACCEPT_PROVISIONAL
    assert report.to_json_dict(r1) == report.to_json_dict(r2)
