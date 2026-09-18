"""TEST 8 -- End-to-end deterministic engineering regression.

Proves the complete agent pipeline (input -> validation -> resolved geometry
-> self-weight -> CoG -> trial placement -> CoG shift -> geometric
constraints -> rig -> handling states -> reactions -> catalogue capacity ->
utilisation -> governing check -> candidate outcome -> invariant guard ->
report) produces deterministic, numerically correct results for a known,
conflict-free scenario. This is NOT about the Reasoner/LLM -- MockReasoner
is used throughout and every test here would produce the identical result
with any Reasoner, because none of these numbers ever pass through one
(see tests/test_reasoner_security.py for that boundary).

The fixture below is deliberately NOT data/wc001.json: WC001's own Appendix A
data intentionally has an unresolved geometry conflict, an UNCONFIRMED turn
method, and unconfirmed supplementary reinforcement (all correctly produce
HOLD -- see test_conflict.py / test_edge_cases.py), which would prevent
exercising the successful ACCEPT_PROVISIONAL path this file is about. This
fixture reuses WC001's own dimensions/concrete class/density (4700x3000x180mm,
C32/40, 2400 kg/m3, 15 MPa first-lift) with every safety-critical input
explicitly resolved instead of left ambiguous.
"""
from __future__ import annotations

import pytest

from liftagent import catalogue, engineering, report
from liftagent.agent import run_agent
from liftagent.reasoner import MockReasoner
from liftagent.schema import CheckState, SourceRole, Status
from tests.conftest import make_concrete, make_element_input, make_geometry_source, make_production


def _resolved_element(**overrides):
    """The Test 8 fixture: single geometry source (no conflict), confirmed
    turn method, WC001's own concrete spec. `reinforcement_confirmed` is
    NOT a field here -- in the real system it's a separate run_agent()
    argument (see cli.py's --reinforcement-confirmed flag), so callers pass
    it to _run_resolved() below, not to this function."""
    sources = overrides.pop("geometry_sources", (
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0, openings=(),
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    production = overrides.pop("production", make_production("tilting_table"))
    concrete = overrides.pop("concrete", make_concrete("C32/40", density=2400.0, first_lift=15.0))
    return make_element_input(geometry_sources=sources, production=production, concrete=concrete, **overrides)


def _run_resolved(reinforcement_confirmed=True, **overrides):
    element = _resolved_element(**overrides)
    return run_agent(element, MockReasoner(), reinforcement_confirmed=reinforcement_confirmed)


# --------------------------------------------------------------------------
# 8A -- deterministic repeatability
# --------------------------------------------------------------------------

def test_8a_deterministic_repeatability():
    r1 = _run_resolved()
    r2 = _run_resolved()

    assert r1.status == r2.status == Status.ACCEPT_PROVISIONAL
    c1, c2 = r1.resolved_candidate, r2.resolved_candidate
    assert c1.anchor_type == c2.anchor_type
    assert [a.x_mm for a in c1.anchors] == [a.x_mm for a in c2.anchors]
    assert (r1.cog.x_mm, r1.cog.y_mm) == (r2.cog.x_mm, r2.cog.y_mm)
    assert r1.self_weight.value_kn == r2.self_weight.value_kn

    reactions1 = sorted((c.handling_state, c.anchor_id, c.demand) for c in c1.checks if c.check_name == "reaction_nonnegative")
    reactions2 = sorted((c.handling_state, c.anchor_id, c.demand) for c in c2.checks if c.check_name == "reaction_nonnegative")
    assert reactions1 == reactions2

    caps1 = sorted((c.handling_state, c.anchor_id, c.capacity, c.utilisation) for c in c1.checks if c.check_name == "axial_capacity")
    caps2 = sorted((c.handling_state, c.anchor_id, c.capacity, c.utilisation) for c in c2.checks if c.check_name == "axial_capacity")
    assert caps1 == caps2

    assert c1.governing_check.handling_state == c2.governing_check.handling_state
    assert c1.governing_check.anchor_id == c2.governing_check.anchor_id
    assert c1.governing_check.utilisation == c2.governing_check.utilisation

    # Full structured-report comparison -- there are no timestamps or
    # randomly-generated IDs anywhere in this report (candidate_id is a
    # deterministic f"candidate-{anchor_type}" string), so this must be
    # byte-for-byte identical, not just "close".
    assert report.to_json_dict(r1) == report.to_json_dict(r2)


# --------------------------------------------------------------------------
# 8B -- self-weight regression (never silently replaced by the 72.6kN reference)
# --------------------------------------------------------------------------

def test_8b_self_weight_regression():
    element = _resolved_element()
    result = _run_resolved()

    L, H, t = 4700.0, 3000.0, 180.0
    density = element.concrete.density_kg_per_m3
    # Independently known physics (volume x density x g for a solid slab),
    # not a copy of engineering.compute_self_weight()'s implementation --
    # only engineering.GRAVITY (the production constant) is reused so the
    # expected value tracks the real constant rather than a hardcoded 9.81.
    expected_kn = (L / 1000.0) * (H / 1000.0) * (t / 1000.0) * density * engineering.GRAVITY / 1000.0

    assert result.self_weight.value_kn == pytest.approx(expected_kn, rel=1e-6)
    assert result.self_weight.value_kn == pytest.approx(59.75, abs=0.05)  # pinned regression value
    assert result.self_weight.reference_value_kn == 72.6  # the brief's figure is still reported...
    assert result.self_weight.value_kn != pytest.approx(72.6, abs=1.0)  # ...but never used as the computed value


# --------------------------------------------------------------------------
# 8C -- CoG regression (symmetric solid panel -> geometric centroid)
# --------------------------------------------------------------------------

def test_8c_cog_regression():
    result = _run_resolved()
    assert result.cog.x_mm == pytest.approx(2350.0, abs=0.5)
    assert result.cog.y_mm == pytest.approx(1500.0, abs=0.5)


# --------------------------------------------------------------------------
# 8D -- anchor position regression
# --------------------------------------------------------------------------

def test_8d_anchor_position_regression():
    result = _run_resolved()
    candidate = result.resolved_candidate
    assert candidate is not None
    L = 4700.0

    # Cross-check against the actual production algorithm (not hardcoded
    # expected values) -- brief 3.5 steps 5-6: trial at 0.207L, then shift to CoG.
    trial_x1, trial_x2 = engineering.trial_placement(L)
    expected_x1, expected_x2 = engineering.shift_to_cog(trial_x1, trial_x2, L, result.cog.x_mm)

    assert candidate.x1_mm == pytest.approx(expected_x1, abs=0.1)
    assert candidate.x2_mm == pytest.approx(expected_x2, abs=0.1)

    assert candidate.x1_mm < candidate.x2_mm
    assert 0.0 < candidate.x1_mm < L
    assert 0.0 < candidate.x2_mm < L
    assert (candidate.x2_mm - candidate.x1_mm) == pytest.approx(expected_x2 - expected_x1, abs=0.1)

    midpoint = (candidate.x1_mm + candidate.x2_mm) / 2.0
    assert midpoint == pytest.approx(result.cog.x_mm, abs=0.5)  # symmetric panel -> symmetric about the CoG


# --------------------------------------------------------------------------
# 8E -- two-anchor equilibrium (verify the equations, not "reasonableness")
# --------------------------------------------------------------------------

def test_8e_two_anchor_equilibrium():
    element = _resolved_element()
    result = _run_resolved()
    candidate = result.resolved_candidate
    x1, x2, xc = candidate.x1_mm, candidate.x2_mm, result.cog.x_mm

    reaction_by_state_anchor = {
        (c.handling_state, c.anchor_id): c.demand
        for c in candidate.checks if c.check_name == "reaction_nonnegative"
    }
    handling_states = engineering.enumerate_handling_states(element.production)
    z = 1.0  # vertical slings -- documented assumption, same as agent.py

    assert handling_states  # sanity
    for state in handling_states:
        r1 = reaction_by_state_anchor[(state.name, "A1")]
        r2 = reaction_by_state_anchor[(state.name, "A2")]
        w = engineering.total_load_for_state(state, result.self_weight.value_kn, z, 4700.0, 3000.0)

        assert (r1 + r2) == pytest.approx(w, rel=1e-6)
        assert (r1 * x1 + r2 * x2) == pytest.approx(w * xc, rel=1e-6)
        # symmetric fixture: CoG is centred between the anchors -> equal reactions
        assert r1 == pytest.approx(r2, rel=1e-6)


# --------------------------------------------------------------------------
# 8F -- capacity source (catalogue.py only, correct strength column, matching clutch)
# --------------------------------------------------------------------------

def test_8f_capacity_source_traces_to_the_catalogue():
    element = _resolved_element()
    result = _run_resolved()
    candidate = result.resolved_candidate

    catalogue_anchor = catalogue.get_anchor(candidate.anchor_type)
    assert candidate.clutch == catalogue_anchor.clutch  # no mismatched anchor/clutch

    clutch_checks = [c for c in candidate.checks if c.check_name == "clutch_match"]
    assert clutch_checks and all(c.state == CheckState.PASS for c in clutch_checks)

    axial_checks = {(c.handling_state, c.anchor_id): c for c in candidate.checks if c.check_name == "axial_capacity"}
    for state in engineering.enumerate_handling_states(element.production):
        expected_column = engineering.select_strength_column(element.concrete, state.strength_requirement)
        assert expected_column is not None
        # C32/40 -> cube strength 40 MPa: unlocks the top (35 MPa) column for "full"
        # states, matches the specified 15 MPa first-lift strength exactly otherwise.
        assert expected_column == (35 if state.strength_requirement == "full" else 15)

        expected_capacity = catalogue.lookup_capacity(candidate.anchor_type, expected_column)
        for anchor_id in ("A1", "A2"):
            check = axial_checks[(state.name, anchor_id)]
            assert check.capacity == expected_capacity  # sourced from catalogue.py, nothing else


# --------------------------------------------------------------------------
# 8G -- utilisation regression
# --------------------------------------------------------------------------

def test_8g_utilisation_regression():
    result = _run_resolved()
    axial_checks = [c for c in result.resolved_candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks
    for c in axial_checks:
        assert c.utilisation == pytest.approx(c.demand / c.capacity, rel=1e-9)
        if c.state == CheckState.PASS:
            assert c.utilisation <= 1.0


# --------------------------------------------------------------------------
# 8H -- governing check is the true max-utilisation check
# --------------------------------------------------------------------------

def test_8h_governing_check_is_the_true_max_utilisation():
    result = _run_resolved()
    candidate = result.resolved_candidate
    axial_checks = [c for c in candidate.checks if c.check_name == "axial_capacity" and c.utilisation is not None]
    assert axial_checks

    max_utilisation = max(c.utilisation for c in axial_checks)
    assert candidate.governing_check is not None
    assert candidate.governing_check.governing is True
    assert candidate.governing_check.utilisation == pytest.approx(max_utilisation, rel=1e-9)

    # if several checks tie at the max (within float tolerance), the governing
    # one must be one of them -- preserves the existing "first max wins" rule
    # (_finalize_governing in agent.py) without over-asserting which specific
    # one that is beyond what the implementation actually guarantees.
    tied = [c for c in axial_checks if abs(c.utilisation - max_utilisation) < 1e-9]
    assert any(c.handling_state == candidate.governing_check.handling_state
               and c.anchor_id == candidate.governing_check.anchor_id for c in tied)


# --------------------------------------------------------------------------
# 8I -- final acceptance still requires human sign-off
# --------------------------------------------------------------------------

def test_8i_final_acceptance_still_requires_human_signoff():
    result = _run_resolved()
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.reason is None
    assert result.resolved_candidate is not None
    assert result.illustrative_candidate is None
    # ACCEPT_PROVISIONAL is not final engineering sign-off -- the brief's
    # hard stop 12 ("the agent itself never signs off") always applies.
    assert result.requires_human_signoff is True


# --------------------------------------------------------------------------
# 8J -- negative regression: change exactly ONE safety-critical input at a time
# --------------------------------------------------------------------------

def test_8j1_unconfirmed_reinforcement_holds():
    result = _run_resolved(reinforcement_confirmed=None)
    assert result.status == Status.HOLD
    assert result.reason == "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED"
    assert result.resolved_candidate is None


def test_8j2_unconfirmed_turn_method_holds():
    result = _run_resolved(production=make_production("UNCONFIRMED"))
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"
    assert result.resolved_candidate is None


def test_8j3_invalid_edge_distance_genuinely_infeasible_rejects():
    """L=800mm -> trial edge distance a=0.207*800~=165.6mm, below every
    catalogue anchor's min_edge_mm (smallest is 300mm). Unlike the L=1200mm
    case (see tests/test_position_iteration.py TEST A), this length is short
    enough that even CFS-WAL-30's bounded 'move in' search cannot find a
    single position that clears both min_edge_mm (300mm) and min_axis_mm
    (600mm) simultaneously -- moving in far enough for edge distance leaves
    only 800-2*300=200mm of spacing, well under the 600mm required. So this
    remains a genuine REJECT even with position iteration in place."""
    result = _run_resolved(geometry_sources=(
        make_geometry_source(length_mm=800.0, height_mm=3000.0, thickness_mm=180.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    assert result.status == Status.REJECT
    assert result.illustrative_candidate is not None
    edge_checks = [c for c in result.illustrative_candidate.checks if c.check_name == "edge_distance"]
    assert edge_checks and edge_checks[0].state == CheckState.FAIL
    # position iteration was attempted but could not find a feasible position
    pi = result.illustrative_candidate.position_iteration
    assert pi is not None and pi.attempted is True
    assert pi.selected_x1_mm is None and pi.selected_x2_mm is None


def test_8j4_invalid_axis_spacing_is_detected_by_the_deterministic_check():
    """IMPLEMENTATION FINDING (see Test 8 report): the trial-placement
    formula (a = 0.207*L, symmetric, spacing = L - 2a = 0.586*L, invariant to
    the CoG shift) combined with the catalogue's fixed 2:1 min_axis:min_edge
    ratio for every anchor makes an axis_spacing-only failure (edge PASS,
    axis FAIL) mathematically unreachable through the full pipeline for any
    input length: whenever a >= min_edge, spacing = 2.83*a is always >=
    2.83*min_edge > 2*min_edge = min_axis. This unit-tests the deterministic
    RULE directly with a contrived spacing to prove it correctly fails when
    genuinely given one -- independent of whether the current placement
    algorithm can reach that exact state end-to-end."""
    anchor = catalogue.get_anchor("ARL-42")  # min_edge_mm=500, min_axis_mm=1000
    results = engineering.check_edge_axis_wall(anchor, x1_mm=600.0, x2_mm=1400.0, length_mm=4700.0, thickness_mm=180.0)
    axis_check = next(c for c in results if c.check_name == "axis_spacing")
    edge_check = next(c for c in results if c.check_name == "edge_distance")
    assert edge_check.state == CheckState.PASS   # edge = 600mm >= 500mm required -- isolated
    assert axis_check.state == CheckState.FAIL    # spacing = 800mm < 1000mm required


def test_8j5_insufficient_anchor_capacity_rejects():
    """L=12000mm keeps edge/axis/wall margins generous for every catalogue
    anchor, but inflates self-weight/transport load beyond even the largest
    anchor's (ARL-52) tabulated capacity -- isolating capacity as the
    failure driver, not geometry."""
    result = _run_resolved(geometry_sources=(
        make_geometry_source(length_mm=12000.0, height_mm=3000.0, thickness_mm=180.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    assert result.status == Status.REJECT
    assert result.reason == "NO_CANDIDATE_SATISFIES_REQUIREMENTS"

    tried_anchor_types = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchor_types == set(catalogue.CANDIDATE_TRY_ORDER)  # every candidate exhausted

    last = result.illustrative_candidate
    assert last.anchor_type == catalogue.CANDIDATE_TRY_ORDER[-1]
    placement_checks = [c for c in last.checks if c.check_name in ("edge_distance", "axis_spacing", "min_wall_axial")]
    assert placement_checks and all(c.state == CheckState.PASS for c in placement_checks)  # geometry is fine...
    capacity_fails = [c for c in last.checks if c.check_name == "axial_capacity" and c.state == CheckState.FAIL]
    assert capacity_fails  # ...it's capacity that's insufficient
