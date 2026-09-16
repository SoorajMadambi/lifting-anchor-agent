from __future__ import annotations

import math

import pytest

from liftagent import catalogue, engineering
from liftagent.schema import CheckState, Opening, ProductionSpec
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
