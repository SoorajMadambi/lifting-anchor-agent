from __future__ import annotations

import math

import pytest

from liftagent import catalogue, engineering
from liftagent.schema import AnchorType, CheckState, Opening, ProductionSpec
from tests.conftest import make_concrete, make_geometry_source


# --------------------------------------------------------------------------
# Concrete strength interpretation
# --------------------------------------------------------------------------

def test_parse_concrete_class():
    cyl, cube = engineering.parse_concrete_class("C32/40")
    assert cyl == 32.0
    assert cube == 40.0


def test_select_strength_column_full_uses_cube_strength_and_unlocks_35mpa():
    concrete = make_concrete("C32/40")
    # cube strength 40 >= 35 -> unlocks the top catalogue column, matching the
    # brief's own worked example (72.6/80 = 0.91 at "full strength").
    assert engineering.select_strength_column(concrete, "full") == 35


def test_select_strength_column_first_lift_matches_exactly():
    concrete = make_concrete("C32/40", first_lift=15.0)
    assert engineering.select_strength_column(concrete, "first_lift") == 15


def test_select_strength_column_below_first_lift_floor_returns_none():
    concrete = make_concrete("C32/40", first_lift=10.0)  # below every tabulated column
    assert engineering.select_strength_column(concrete, "first_lift") is None


# --------------------------------------------------------------------------
# Self-weight
# --------------------------------------------------------------------------

def test_self_weight_matches_hand_calc_no_openings():
    geom = make_geometry_source(length_mm=4300, height_mm=3000, thickness_mm=180, openings=())
    r = engineering.compute_self_weight(geom, density_kg_per_m3=2400.0)
    # volume = 4.3*3*0.18 = 2.322 m3; mass = 5572.8 kg; G = mass*9.81/1000
    expected = 2.322 * 2400.0 * 9.81 / 1000.0
    assert r.value_kn == pytest.approx(expected, rel=1e-6)


def test_self_weight_subtracts_opening_area():
    door = Opening(id="door", x_mm=1150, width_mm=1100, sill_mm=0, height_mm=2100)
    window = Opening(id="window", x_mm=3050, width_mm=1200, sill_mm=950, height_mm=1000)
    geom = make_geometry_source(length_mm=4700, height_mm=3000, thickness_mm=180, openings=(door, window))
    r = engineering.compute_self_weight(geom, density_kg_per_m3=2400.0)
    net_area_m2 = 4.7 * 3.0 - (1.1 * 2.1) - (1.2 * 1.0)
    expected = net_area_m2 * 0.18 * 2400.0 * 9.81 / 1000.0
    assert r.value_kn == pytest.approx(expected, rel=1e-6)
    assert r.value_kn < 60  # sanity: well below the brief's un-reconciled 72.6kN reference


def test_self_weight_flags_discrepancy_from_reference_figure():
    geom = make_geometry_source(length_mm=4700, height_mm=3000, thickness_mm=180)
    r = engineering.compute_self_weight(geom, density_kg_per_m3=2400.0, reference_value_kn=72.6)
    assert r.warning != ""
    assert r.discrepancy_kn != 0


# --------------------------------------------------------------------------
# Centre of gravity
# --------------------------------------------------------------------------

def test_cog_centred_with_no_openings():
    geom = make_geometry_source(length_mm=4700, height_mm=3000, thickness_mm=180, openings=())
    r = engineering.compute_cog(geom)
    assert r.x_mm == pytest.approx(2350.0)
    assert r.y_mm == pytest.approx(1500.0)


def test_cog_shifts_off_centre_with_asymmetric_openings():
    # A single large opening near the right end pulls the CoG to the left.
    opening = Opening(id="big", x_mm=3500, width_mm=1000, sill_mm=0, height_mm=2000)
    geom = make_geometry_source(length_mm=4700, height_mm=3000, thickness_mm=180, openings=(opening,))
    r = engineering.compute_cog(geom)
    assert r.x_mm < 2350.0  # pulled left, away from the opening
    assert "curtain" in r.note.lower() or "bias" in r.note.lower()  # qualitative note present, no invented number


# --------------------------------------------------------------------------
# Trial placement / CoG shift
# --------------------------------------------------------------------------

def test_trial_placement_matches_brief_worked_example():
    x1, x2 = engineering.trial_placement(4700.0)
    assert x1 == pytest.approx(972.9, abs=0.1)
    assert x2 == pytest.approx(3727.1, abs=0.1)


def test_shift_to_cog_centred_cog_leaves_positions_unchanged():
    x1, x2 = engineering.trial_placement(4700.0)
    nx1, nx2 = engineering.shift_to_cog(x1, x2, 4700.0, cog_x_mm=2350.0)
    assert nx1 == pytest.approx(x1)
    assert nx2 == pytest.approx(x2)


def test_shift_to_cog_off_centre_shifts_both_anchors_equally():
    x1, x2 = engineering.trial_placement(4700.0)
    nx1, nx2 = engineering.shift_to_cog(x1, x2, 4700.0, cog_x_mm=2000.0)  # cog left of mid-length
    delta = 2000.0 - 2350.0
    assert nx1 == pytest.approx(x1 + delta)
    assert nx2 == pytest.approx(x2 + delta)
    assert (nx2 - nx1) == pytest.approx(x2 - x1)  # spacing preserved


# --------------------------------------------------------------------------
# Two-anchor statics / equilibrium
# --------------------------------------------------------------------------

def test_reactions_equilibrium_for_arbitrary_cog():
    W, x1, x2, xc = 100.0, 900.0, 3700.0, 1800.0
    r1, r2 = engineering.compute_reactions(W, x1, x2, xc)
    assert (r1 + r2) == pytest.approx(W)
    assert (r1 * x1 + r2 * x2) == pytest.approx(W * xc)


def test_reactions_symmetric_case_splits_evenly():
    r1, r2 = engineering.compute_reactions(100.0, 973.0, 3727.0, cog_x_mm=2350.0)
    assert r1 == pytest.approx(r2)
    assert r1 == pytest.approx(50.0)


def test_reactions_off_centre_cog_gives_unequal_reactions():
    r1, r2 = engineering.compute_reactions(100.0, 973.0, 3727.0, cog_x_mm=2000.0)
    assert r1 != pytest.approx(r2)
    assert max(r1, r2) > 50.0  # the anchor nearer the CoG carries more


# --------------------------------------------------------------------------
# Catalogue lookup
# --------------------------------------------------------------------------

def test_catalogue_lookup_capacity():
    assert catalogue.lookup_capacity("ARL-42", 35) == 80.0
    assert catalogue.lookup_capacity("ARL-42", 15) == 60.0


def test_catalogue_unknown_anchor_raises():
    with pytest.raises(catalogue.UnknownAnchorType):
        catalogue.get_anchor("NOT-A-REAL-ANCHOR")


def test_nearest_available_column():
    assert catalogue.nearest_available_column(40) == 35
    assert catalogue.nearest_available_column(20) == 15
    assert catalogue.nearest_available_column(10) is None


# --------------------------------------------------------------------------
# Edge / axis / wall-thickness checks
# --------------------------------------------------------------------------

def test_edge_axis_wall_all_pass_for_wc001_nominal_geometry():
    anchor = catalogue.get_anchor("ARL-42")
    results = engineering.check_edge_axis_wall(anchor, 973.0, 3727.0, 4700.0, 180.0)
    assert all(r.state == CheckState.PASS for r in results)


def test_min_wall_axial_fails_when_panel_too_thin_for_anchor():
    # ARL-52 needs 200mm axial min-wall; this panel is only 180mm.
    anchor = catalogue.get_anchor("ARL-52")
    results = engineering.check_edge_axis_wall(anchor, 973.0, 3727.0, 4700.0, 180.0)
    wall_check = next(r for r in results if r.check_name == "min_wall_axial")
    assert wall_check.state == CheckState.FAIL


def test_edge_distance_fails_when_anchor_too_close_to_end():
    anchor = catalogue.get_anchor("ARL-42")  # min edge 500mm
    results = engineering.check_edge_axis_wall(anchor, 100.0, 3727.0, 4700.0, 180.0)
    edge_check = next(r for r in results if r.check_name == "edge_distance")
    assert edge_check.state == CheckState.FAIL


def test_axis_spacing_fails_for_anchors_too_close_together():
    anchor = catalogue.get_anchor("ARL-42")  # min axis 1000mm
    results = engineering.check_edge_axis_wall(anchor, 2000.0, 2500.0, 4700.0, 180.0)
    axis_check = next(r for r in results if r.check_name == "axis_spacing")
    assert axis_check.state == CheckState.FAIL


# --------------------------------------------------------------------------
# Rig decision
# --------------------------------------------------------------------------

def test_spreader_mandatory_for_thin_panel_matches_brief():
    anchor = catalogue.get_anchor("ARL-42")
    rig = engineering.decide_rig(anchor, thickness_mm=180.0)
    assert rig.spreader_required is True
    assert "240" in rig.reason and "180" in rig.reason


def test_spreader_not_required_for_thick_panel():
    anchor = catalogue.get_anchor("ARL-42")
    rig = engineering.decide_rig(anchor, thickness_mm=300.0)
    assert rig.spreader_required is False


def test_all_catalogue_anchors_require_spreader_on_wc001_thickness():
    # Documented finding: every catalogue anchor's transverse min-wall exceeds
    # WC001's 180mm thickness, so the rig decision is anchor-independent here.
    for name in catalogue.CATALOGUE:
        anchor = catalogue.get_anchor(name)
        rig = engineering.decide_rig(anchor, thickness_mm=180.0)
        assert rig.spreader_required is True, name


# --------------------------------------------------------------------------
# Capacity / utilisation -- regression against the brief's own worked example
# --------------------------------------------------------------------------

def test_capacity_check_matches_brief_worked_example_utilisation():
    anchor = catalogue.get_anchor("ARL-42")
    state = engineering.enumerate_handling_states(ProductionSpec(
        cast_orientation="flat", mould="x", turn_method="tilting_table", storage="x",
    ))
    transport = next(s for s in state if s.name == "ROAD_TRANSPORT")
    # brief's Figure 1: G=72.6kN, psi=2.0, z=1, n=2 -> F_per_anchor = G = 72.6kN
    demand_kn = 72.6
    strength_col = engineering.select_strength_column(make_concrete("C32/40"), transport.strength_requirement)
    check = engineering.check_capacity(anchor, transport, demand_kn, strength_col, anchor_id="A1")
    assert strength_col == 35
    assert check.capacity == 80.0
    assert check.utilisation == pytest.approx(0.9075, abs=0.001)
    assert check.state == CheckState.PASS


def test_capacity_check_fails_when_demand_exceeds_capacity():
    anchor = catalogue.get_anchor("ARL-30")
    state = engineering.enumerate_handling_states(ProductionSpec(
        cast_orientation="flat", mould="x", turn_method="tilting_table", storage="x",
    ))[0]
    check = engineering.check_capacity(anchor, state, demand_kn=1000.0, strength_column_mpa=35, anchor_id="A1")
    assert check.state == CheckState.FAIL


def test_capacity_check_unknown_when_strength_column_unresolved():
    anchor = catalogue.get_anchor("ARL-42")
    state = engineering.enumerate_handling_states(ProductionSpec(
        cast_orientation="flat", mould="x", turn_method="tilting_table", storage="x",
    ))[0]
    check = engineering.check_capacity(anchor, state, demand_kn=10.0, strength_column_mpa=None, anchor_id="A1")
    assert check.state == CheckState.UNKNOWN
    assert check.scope.value == "PROBLEM"


# --------------------------------------------------------------------------
# Handling states
# --------------------------------------------------------------------------

def test_enumerate_handling_states_runs_both_turn_cases_when_unconfirmed():
    prod = ProductionSpec(cast_orientation="flat", mould="x", turn_method="UNCONFIRMED", storage="x")
    states = engineering.enumerate_handling_states(prod)
    names = {s.name for s in states}
    assert "TURN_TILTING_TABLE" in names
    assert "TURN_FREE_CRANE" in names


def test_enumerate_handling_states_runs_single_turn_case_when_confirmed():
    prod = ProductionSpec(cast_orientation="flat", mould="x", turn_method="tilting_table", storage="x")
    states = engineering.enumerate_handling_states(prod)
    names = {s.name for s in states}
    assert "TURN_TILTING_TABLE" in names
    assert "TURN_FREE_CRANE" not in names


def test_demould_load_includes_adhesion_term():
    prod = ProductionSpec(cast_orientation="flat", mould="x", turn_method="tilting_table", storage="x")
    demould = next(s for s in engineering.enumerate_handling_states(prod) if s.name == "DEMOULD")
    f = engineering.total_load_for_state(demould, self_weight_kn=44.9, z=1.0, length_mm=4700, height_mm=3000)
    a_f = 4.7 * 3.0
    expected = 44.9 * 1.3 + 1.0 * a_f * 1.0
    assert f == pytest.approx(expected)


def test_sling_angle_factor_matches_vertical_case():
    assert engineering.sling_angle_factor(0.0) == pytest.approx(1.0)


def test_sling_angle_factor_rejects_angle_beyond_limit():
    with pytest.raises(ValueError):
        engineering.sling_angle_factor(45.0)


# --------------------------------------------------------------------------
# find_feasible_inward_position() -- opening-aware closed-form interval
# search (2nd F3 audit). A synthetic AnchorType with round min_edge_mm=200/
# min_axis_mm=400 is used so the exact numbers below are hand-verifiable,
# rather than tied to a specific catalogue row.
# --------------------------------------------------------------------------

_INTERVAL_TEST_ANCHOR = AnchorType(
    name="TEST-200-400", clutch="X", capacity_kn={15: 1, 25: 1, 35: 1}, v_zul_kn=1,
    min_edge_mm=200.0, min_axis_mm=400.0, min_wall_axial_mm=1.0, min_wall_transverse_mm=1.0,
)
_L, _H = 1000.0, 3000.0  # panel length/height for every case below


def _find(cog_x_mm: float, openings: tuple[Opening, ...]):
    return engineering.find_feasible_inward_position(_INTERVAL_TEST_ANCHOR, _L, cog_x_mm, openings, _H)


def _assert_clear_and_feasible(result, openings):
    """Re-verifies a returned position with the REAL, unchanged check
    functions -- the analytical derivation is never trusted alone."""
    assert result is not None
    x1, x2 = result
    edge_axis = engineering.check_edge_axis_wall(_INTERVAL_TEST_ANCHOR, x1, x2, _L, thickness_mm=999.0)
    assert all(c.state == CheckState.PASS for c in edge_axis if c.check_name in ("edge_distance", "axis_spacing"))
    clash = engineering.check_opening_void_clash(x1, x2, openings, _H)
    assert clash.state == CheckState.PASS, clash.explanation


def test_interval_1_exact_repro_finds_position_past_the_blocking_opening():
    """The scenario that motivated this fix: edge/axis-feasible a in [200,300],
    an opening at [190,210] blocks the minimum (a=200), but a=211..300 are
    valid -- the old single-point formula rejected this candidate outright."""
    opening = Opening(id="O1", x_mm=190.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (opening,))
    _assert_clear_and_feasible(result, (opening,))
    x1, x2 = result
    assert 210.0 < x1 <= 300.0  # strictly past the opening, within the axis-spacing ceiling
    assert (x1 + x2) / 2 == pytest.approx(500.0)  # plumb preserved


def test_interval_2_opening_blocking_the_entire_interval_returns_none():
    opening = Opening(id="O2", x_mm=150.0, width_mm=200.0, sill_mm=2600.0, height_mm=400.0)  # covers [150,350]
    assert _find(500.0, (opening,)) is None


def test_interval_3_middle_block_leaves_the_original_minimum_untouched():
    """An opening blocking only the MIDDLE of the feasible range (leaving
    a_min=200 itself clear) must not perturb the answer -- proves the
    smallest remaining position is chosen even when a larger disjoint
    remaining sub-interval also exists further right."""
    opening = Opening(id="O3", x_mm=240.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (opening,))
    _assert_clear_and_feasible(result, (opening,))
    assert result == (200.0, 800.0)  # unchanged from the no-opening answer


def test_interval_4_overlapping_openings_are_handled_as_their_union():
    """Two openings whose blocked sub-ranges overlap (200-215 and 210-230)
    must be treated as one combined obstruction, not two independent ones."""
    opening_a = Opening(id="O4a", x_mm=200.0, width_mm=15.0, sill_mm=2600.0, height_mm=400.0)
    opening_b = Opening(id="O4b", x_mm=210.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (opening_a, opening_b))
    _assert_clear_and_feasible(result, (opening_a, opening_b))
    x1, _ = result
    assert x1 > 230.0  # past BOTH openings' combined span, not just the first


def test_interval_5_opening_blocking_only_the_x2_side_is_detected():
    """An opening near the panel's other end (blocking x2, not x1) must be
    discovered too -- exercises the x2-side half of the formula. The
    unmodified x1=200 candidate does NOT clash directly (200 isn't inside
    [790,810]), but its CoG-symmetric partner x2=800 does -- so the search
    must still escape, moving x1 as a side effect of keeping the pair
    exactly CoG-centered while nudging x2 clear."""
    opening = Opening(id="O5", x_mm=790.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (opening,))
    _assert_clear_and_feasible(result, (opening,))
    x1, x2 = result
    assert (x1, x2) != (200.0, 800.0)  # the naive unblocked-x1 answer would have clashed via x2
    assert x2 < 790.0  # nudged clear of the opening's lower edge


def test_interval_6_openings_affecting_both_anchors_are_both_resolved():
    """Opening A blocks x1 near the minimum; opening B independently blocks
    the x2 that results from escaping A -- proves the search keeps
    re-checking after an escape rather than stopping after the first fix,
    and that nudging x2 doesn't silently re-open a clash on x1's side."""
    opening_a = Opening(id="OA", x_mm=190.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    opening_b = Opening(id="OB", x_mm=785.0, width_mm=10.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (opening_a, opening_b))
    _assert_clear_and_feasible(result, (opening_a, opening_b))


def test_interval_7_nonzero_cog_offset_is_handled_correctly():
    """The formulas must not silently assume a centred CoG."""
    opening = Opening(id="O7", x_mm=290.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(600.0, (opening,))  # off-centre CoG
    _assert_clear_and_feasible(result, (opening,))
    x1, x2 = result
    assert (x1 + x2) / 2 == pytest.approx(600.0)


def test_interval_8_malformed_opening_is_ignored_by_the_search_but_still_fails_the_real_check():
    """A malformed opening (non-positive width) can't be solved for a
    meaningful blocked range, so the search doesn't try to avoid it -- but
    the caller's mandatory re-verification (check_opening_void_clash) still
    unconditionally FAILs on it, exactly as for any other candidate. This
    search must never fabricate a position that papers over that."""
    bad = Opening(id="BAD", x_mm=100.0, width_mm=-5.0, sill_mm=2600.0, height_mm=400.0)
    result = _find(500.0, (bad,))
    assert result == (200.0, 800.0)  # search itself is unaffected (correctly can't reason about it)
    clash = engineering.check_opening_void_clash(*result, (bad,), _H)
    assert clash.state == CheckState.FAIL  # but the real check still conservatively fails


def test_interval_9_opening_not_reaching_the_top_edge_is_irrelevant():
    low_opening = Opening(id="LOW", x_mm=190.0, width_mm=20.0, sill_mm=0.0, height_mm=100.0)
    assert _find(500.0, (low_opening,)) == (200.0, 800.0)


def test_interval_10_no_openings_reduces_to_the_original_single_point_answer():
    assert _find(500.0, ()) == (200.0, 800.0)


def test_interval_11_search_terminates_quickly_even_with_many_openings():
    """Bounded-behaviour proof: many small, scattered, irrelevant openings
    must not turn this into an unbounded search -- termination is guaranteed
    by x1 only ever increasing, checked here against a stress case."""
    import time
    many_openings = tuple(
        Opening(id=f"O{i}", x_mm=float(1000 + i * 5), width_mm=2.0, sill_mm=2600.0, height_mm=400.0)
        for i in range(200)  # all far outside the feasible [200,300]/[700,800] bands -- irrelevant
    )
    start = time.monotonic()
    result = _find(500.0, many_openings)
    elapsed = time.monotonic() - start
    assert result == (200.0, 800.0)  # none of them are actually relevant
    assert elapsed < 1.0  # not a stepped/open-ended search


def test_interval_12_result_is_deterministic_across_repeated_calls():
    opening = Opening(id="O1", x_mm=190.0, width_mm=20.0, sill_mm=2600.0, height_mm=400.0)
    r1 = _find(500.0, (opening,))
    r2 = _find(500.0, (opening,))
    assert r1 == r2  # exact float equality, not just approx -- pure function, no hidden state


def test_capacity_invariant_to_cog_centered_spacing():
    """Documents/pins the capacity-invariance property this whole feature
    deliberately does NOT try to exploit (see DESIGN_NOTE.md): for any
    CoG-centered pair (x1+x2)/2==cog, compute_reactions() always splits the
    load exactly 50/50, regardless of spacing -- so placement iteration can
    never rescue a capacity failure for the same anchor type."""
    cog = 500.0
    total_load = 100.0
    for x1, x2 in [(200.0, 800.0), (350.0, 650.0), (490.0, 510.0)]:
        assert (x1 + x2) / 2 == pytest.approx(cog)
        r1, r2 = engineering.compute_reactions(total_load, x1, x2, cog)
        assert r1 == pytest.approx(50.0)
        assert r2 == pytest.approx(50.0)
