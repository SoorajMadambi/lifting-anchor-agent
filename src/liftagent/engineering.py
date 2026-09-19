"""Deterministic engineering core -- brief section 3.

Every function here is pure and side-effect-free. NOTHING in this module may
be influenced by an LLM/Reasoner (Determinism Boundary, plan section 3). All
functions are unit tested in tests/test_engineering.py.
"""
from __future__ import annotations

import math

from liftagent import catalogue
from liftagent.schema import (
    AnchorType, CheckResult, CheckScope, CheckState, CogResult, ConcreteSpec,
    GeometrySource, HandlingState, Opening, ProductionSpec, ReactionResult,
    RigResult, SelfWeightResult,
)

GRAVITY = 9.81
ADHESION_KN_PER_M2 = 1.0        # q_adh, brief 3.1
XI_ADH_FLAT_SOLID = 1.0          # xi_adh for a flat solid panel
MAX_SLING_ANGLE_DEG = 30.0       # brief 3.4 hard limit


# --------------------------------------------------------------------------
# Concrete strength interpretation (isolated policy function -- plan section 13)
# --------------------------------------------------------------------------

def parse_concrete_class(class_label: str) -> tuple[float, float]:
    """"C32/40" -> (cylinder_mpa=32.0, cube_mpa=40.0)."""
    try:
        cyl_str, cube_str = class_label.lstrip("Cc").split("/")
        return float(cyl_str), float(cube_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot parse concrete class {class_label!r} as 'Cxx/yy'") from exc


def select_strength_column(concrete: ConcreteSpec, requirement: str) -> int | None:
    """Isolated strength-column policy (brief 3.2).

    ASSUMPTION (documented, see DESIGN_NOTE.md): the catalogue's "@35 MPa"
    column is matched against the CUBE strength (the second number in a
    "Cxx/yy" class label), not the cylinder strength. This is the only
    reading consistent with the brief's own worked example: C32/40 at
    "full strength" unlocks the 35 MPa column (72.6/80 = 0.91 utilisation
    against ARL-42's 80 kN "@35" capacity) even though the cylinder strength
    (32 MPa) alone would not.

    requirement: "first_lift" uses the specified/measured first-lift
    strength; "full" uses the cube strength. Returns None if the achieved
    strength is below every tabulated column (brief hard stop: no lift below
    the specified first-lift strength).
    """
    if requirement == "first_lift":
        achieved = concrete.first_lift_strength_mpa
    elif requirement == "full":
        achieved = concrete.cube_strength_mpa
    else:
        raise ValueError(f"Unknown strength requirement {requirement!r}")
    return catalogue.nearest_available_column(achieved)


# --------------------------------------------------------------------------
# Handling states (brief 3.2 / 3.5 step 1)
# --------------------------------------------------------------------------

# The only turn methods this POC actually recognises as CONFIRMED. Any other
# value -- "UNCONFIRMED", None, empty, or an unrecognised/garbage string --
# must be treated as not confirmed (agent.py's turn_confirmed check uses this
# same constant, so the two can never drift apart; see Test 10 finding: a
# bare `not in ("UNCONFIRMED", None, "")` check previously let ANY unknown
# string silently count as confirmed).
KNOWN_TURN_METHODS: tuple[str, ...] = ("tilting_table", "free_crane")


def enumerate_handling_states(production: ProductionSpec) -> tuple[HandlingState, ...]:
    states: list[HandlingState] = [
        HandlingState("DEMOULD", psi_dyn=1.3, strength_requirement="first_lift",
                       adhesion=True, note="early age -- concrete is weakest here"),
        HandlingState("STORAGE", psi_dyn=1.0, strength_requirement="first_lift",
                       note="anchors ~unloaded per brief; checked as a static bound only, "
                            "not a distinct dynamic case (documented assumption)"),
        HandlingState("ROAD_TRANSPORT", psi_dyn=2.0, strength_requirement="full",
                       note="usually governs the anchor"),
        HandlingState("ERECTION", psi_dyn=1.4, strength_requirement="full"),
    ]
    turn_method = production.turn_method
    if turn_method == "UNCONFIRMED":
        states.append(HandlingState("TURN_TILTING_TABLE", psi_dyn=1.1,
                                     strength_requirement="first_lift",
                                     note="gentle if on a table"))
        states.append(HandlingState("TURN_FREE_CRANE", psi_dyn=1.4,
                                     strength_requirement="first_lift",
                                     note="conservative weak-axis check run because "
                                          "turn_method is UNCONFIRMED (brief 3.6 hard stop)"))
    elif turn_method == "tilting_table":
        states.append(HandlingState("TURN_TILTING_TABLE", psi_dyn=1.1,
                                     strength_requirement="first_lift"))
    elif turn_method == "free_crane":
        states.append(HandlingState("TURN_FREE_CRANE", psi_dyn=1.4,
                                     strength_requirement="first_lift",
                                     note="also bends the panel weak-axis"))
    else:
        states.append(HandlingState("TURN_FREE_CRANE", psi_dyn=1.4,
                                     strength_requirement="first_lift",
                                     note=f"unrecognised turn_method {turn_method!r}; "
                                          f"treated conservatively as free-crane"))
    return tuple(states)


# --------------------------------------------------------------------------
# Self-weight (plan section 8)
# --------------------------------------------------------------------------

def compute_self_weight(geometry: GeometrySource, density_kg_per_m3: float,
                         reference_value_kn: float = 72.6, concrete_class: str = "") -> SelfWeightResult:
    if geometry.length_mm is None or geometry.height_mm is None or geometry.thickness_mm is None:
        raise ValueError("geometry source is missing dimensions; cannot compute self-weight")
    if density_kg_per_m3 <= 0:
        # Test 10 finding: a non-positive density silently produced a zero
        # (or negative) self-weight, zeroing every non-adhesion load and
        # letting every capacity check trivially PASS -> a false
        # ACCEPT_PROVISIONAL. Concrete density is a required physical input,
        # same as the panel dimensions checked above.
        raise ValueError(f"density_kg_per_m3 must be positive, got {density_kg_per_m3!r}")
    gross_area_mm2 = geometry.length_mm * geometry.height_mm
    opening_area_mm2 = sum(o.width_mm * o.height_mm for o in geometry.openings)
    net_area_m2 = (gross_area_mm2 - opening_area_mm2) / 1_000_000.0
    volume_m3 = net_area_m2 * (geometry.thickness_mm / 1000.0)
    mass_kg = volume_m3 * density_kg_per_m3
    value_kn = mass_kg * GRAVITY / 1000.0
    discrepancy_kn = value_kn - reference_value_kn
    warning = ""
    if reference_value_kn and abs(discrepancy_kn) > 0.05 * reference_value_kn:
        warning = (
            f"Computed self-weight ({value_kn:.1f} kN) does not reconcile with the brief's "
            f"reference figure ({reference_value_kn:.1f} kN); the reference value's provenance "
            f"cannot be established from the supplied geometry+density, so it is NOT used as an "
            f"authoritative input (see DESIGN_NOTE.md)."
        )
    return SelfWeightResult(
        value_kn=value_kn, method="geometry_volume_x_density", source=geometry.source_name,
        reference_value_kn=reference_value_kn, discrepancy_kn=discrepancy_kn,
        density_kg_per_m3=density_kg_per_m3, concrete_class=concrete_class, warning=warning,
    )


# --------------------------------------------------------------------------
# Centre of gravity (plan section 9)
# --------------------------------------------------------------------------

def compute_cog(geometry: GeometrySource) -> CogResult:
    if geometry.length_mm is None or geometry.height_mm is None:
        raise ValueError("geometry source is missing dimensions; cannot compute CoG")
    L, H = geometry.length_mm, geometry.height_mm
    panel_area = L * H
    panel_cx, panel_cy = L / 2.0, H / 2.0

    net_area = panel_area
    moment_x = panel_area * panel_cx
    moment_y = panel_area * panel_cy
    for o in geometry.openings:
        a = o.width_mm * o.height_mm
        net_area -= a
        moment_x -= a * o.centroid_x_mm
        moment_y -= a * o.centroid_y_from_bottom_mm

    if net_area <= 0:
        raise ValueError("net panel area is non-positive; openings exceed panel area")

    x_cog = moment_x / net_area
    y_cog = moment_y / net_area
    return CogResult(
        x_mm=x_cog, y_mm=y_cog, source=geometry.source_name,
        note=(
            "One-sided reinforcement curtain may introduce a qualitative mass bias "
            "(input note: 'curtain steps down up the building'); no quantified offset is "
            "provided, so no numerical correction was applied."
        ),
    )


# --------------------------------------------------------------------------
# Trial placement (plan sections 15-16)
# --------------------------------------------------------------------------

BALANCE_POINT_FRACTION = 0.207


def trial_placement(length_mm: float) -> tuple[float, float]:
    a = BALANCE_POINT_FRACTION * length_mm
    return a, length_mm - a


def shift_to_cog(x1_trial: float, x2_trial: float, length_mm: float, cog_x_mm: float) -> tuple[float, float]:
    delta = cog_x_mm - length_mm / 2.0
    return x1_trial + delta, x2_trial + delta


# --------------------------------------------------------------------------
# Two-anchor statics (plan section 10)
# --------------------------------------------------------------------------

def compute_reactions(total_load_kn: float, x1_mm: float, x2_mm: float, cog_x_mm: float) -> tuple[float, float]:
    if x1_mm == x2_mm:
        raise ValueError("anchor positions coincide; cannot solve statics")
    r1 = total_load_kn * (x2_mm - cog_x_mm) / (x2_mm - x1_mm)
    r2 = total_load_kn * (cog_x_mm - x1_mm) / (x2_mm - x1_mm)
    return r1, r2


# --------------------------------------------------------------------------
# Loads per handling state (brief 3.1)
# --------------------------------------------------------------------------

def sling_angle_factor(beta_deg: float) -> float:
    if beta_deg < 0 or beta_deg > MAX_SLING_ANGLE_DEG:
        raise ValueError(f"sling angle {beta_deg} deg exceeds the {MAX_SLING_ANGLE_DEG} deg hard limit (brief 3.4)")
    return 1.0 / math.cos(math.radians(beta_deg))


def total_load_for_state(state: HandlingState, self_weight_kn: float, z: float,
                          length_mm: float, height_mm: float) -> float:
    if state.adhesion:
        a_f_m2 = (length_mm / 1000.0) * (height_mm / 1000.0)
        return self_weight_kn * state.psi_dyn + ADHESION_KN_PER_M2 * a_f_m2 * XI_ADH_FLAT_SOLID
    return self_weight_kn * state.psi_dyn * z


# --------------------------------------------------------------------------
# Rig decision (plan section 19)
# --------------------------------------------------------------------------

def decide_rig(anchor: AnchorType, thickness_mm: float, beta_deg: float = 0.0) -> RigResult:
    thickness_forces_spreader = thickness_mm < anchor.min_wall_transverse_mm
    angle_forces_spreader = beta_deg > MAX_SLING_ANGLE_DEG
    if thickness_forces_spreader:
        return RigResult(
            spreader_required=True,
            reason=(
                f"Panel thickness {thickness_mm:.0f} mm is below {anchor.name}'s "
                f"{anchor.min_wall_transverse_mm:.0f} mm transverse minimum wall requirement."
            ),
        )
    if angle_forces_spreader:
        return RigResult(spreader_required=True,
                          reason=f"Sling angle {beta_deg} deg exceeds the {MAX_SLING_ANGLE_DEG} deg limit.")
    return RigResult(spreader_required=False, reason="Direct slings permitted: angle and transverse wall both within limits.")


# --------------------------------------------------------------------------
# Placement constraint checks (plan section 17) -- CANDIDATE scope
# --------------------------------------------------------------------------

def check_edge_axis_wall(anchor: AnchorType, x1_mm: float, x2_mm: float,
                          length_mm: float, thickness_mm: float) -> list[CheckResult]:
    lo, hi = min(x1_mm, x2_mm), max(x1_mm, x2_mm)
    edge_distance = min(lo, length_mm - hi)
    axis_spacing = hi - lo

    results = [
        CheckResult(
            check_name="min_wall_axial", state=(CheckState.PASS if thickness_mm >= anchor.min_wall_axial_mm else CheckState.FAIL),
            scope=CheckScope.CANDIDATE, demand=thickness_mm, capacity=anchor.min_wall_axial_mm,
            rule_reference="brief 3.3 min wall (axial)",
            explanation=f"panel thickness {thickness_mm:.0f}mm vs {anchor.name} min axial wall {anchor.min_wall_axial_mm:.0f}mm",
        ),
        CheckResult(
            check_name="edge_distance", state=(CheckState.PASS if edge_distance >= anchor.min_edge_mm else CheckState.FAIL),
            scope=CheckScope.CANDIDATE, demand=edge_distance, capacity=anchor.min_edge_mm,
            rule_reference="brief 3.3 min edge distance",
            explanation=f"edge distance {edge_distance:.0f}mm vs {anchor.name} min edge {anchor.min_edge_mm:.0f}mm",
        ),
        CheckResult(
            check_name="axis_spacing", state=(CheckState.PASS if axis_spacing >= anchor.min_axis_mm else CheckState.FAIL),
            scope=CheckScope.CANDIDATE, demand=axis_spacing, capacity=anchor.min_axis_mm,
            rule_reference="brief 3.3 min axis spacing",
            explanation=f"axis spacing {axis_spacing:.0f}mm vs {anchor.name} min axis {anchor.min_axis_mm:.0f}mm",
        ),
    ]
    return results


def find_feasible_inward_position(anchor: AnchorType, length_mm: float, cog_x_mm: float,
                                   openings: tuple[Opening, ...], height_mm: float) -> tuple[float, float] | None:
    """Bounded deterministic 'move in' search -- brief 3.5 step 11's "move in"
    remedy, and nothing more. NOT an optimiser and NOT a stepped/arbitrary
    search: this analytically derives the smallest feasible x1 (the
    edge/axis-feasible interval for the inward offset, with every
    top-edge-reaching opening's blocked sub-range removed) and its exact
    CoG-symmetric partner x2 -- the same "closest to the edge" bias as
    before, just aware of openings too. min_edge_mm/min_axis_mm come
    straight from catalogue.py; opening bounds come straight from the
    Opening dataclass; nothing is invented. When no opening affects the
    feasible interval, this reduces to exactly the prior single-point
    answer (x1 = min_edge_mm + |delta| + delta).

    Why plumb is preserved by construction: x2 is always computed as
    `2*cog_x_mm - x1`, never independently -- so the pair's MIDPOINT is
    always exactly cog_x_mm by construction, for any x1 this function
    produces. The caller re-derives reactions via the normal
    compute_reactions() call to confirm this (never assumed, never a 50/50
    split), exactly as for the original trial position.

    Floating-point boundary handling -- WHY THIS WORKS IN COORDINATE SPACE,
    NOT OFFSET SPACE: since check_opening_void_clash()'s x-overlap test is
    inclusive (`<=`) on both ends, the feasible region is open immediately
    past a blocking opening -- there is no real-number smallest value
    there, so `math.nextafter(boundary, math.inf/-math.inf)` is used to
    select the smallest strictly-clear representable float. This is a
    NUMERICAL boundary technique only -- it asserts no physical
    construction clearance from the opening, and no such tolerance value is
    invented or reused from elsewhere in this codebase (see DESIGN_NOTE.md).
    Critically, the nudge is always applied directly to whichever
    coordinate (x1 or x2) the clashing opening actually constrains, and the
    OTHER coordinate is always freshly recomputed as `2*cog_x_mm -
    (that coordinate)` -- never round-tripped through an intermediate
    offset. Converting a tiny nudge on the SMALLER-magnitude side (e.g. x1)
    through a large-magnitude subtraction (e.g. length_mm - x1) can lose it
    entirely to floating-point rounding (verified empirically); nudging
    whichever coordinate is actually being checked, and only ever deriving
    the other one from it, avoids that loss.

    Returns the new (x1_mm, x2_mm) if a feasible position remains after
    accounting for min_edge_mm, min_axis_mm, and every opening, else None.
    Does NOT check min_wall_axial or capacity -- neither depends on
    position (see DESIGN_NOTE.md's capacity-invariance note), so the caller
    must still re-check them. This function's result is ONLY a candidate --
    the caller MUST re-verify it with the real, unchanged
    check_edge_axis_wall()/check_opening_void_clash() before accepting it;
    this analytical derivation is never itself trusted as the final source
    of feasibility. If floating-point propagation ever left a residual
    clash despite the handling above, the caller's mandatory re-check
    catches it and the candidate fails closed -- no further retry or
    invented tolerance is introduced here.
    """
    delta = cog_x_mm - length_mm / 2.0
    a_min = anchor.min_edge_mm + abs(delta)                # smallest inward move clearing min_edge_mm
    a_max = (length_mm - anchor.min_axis_mm) / 2.0          # largest inward move still clearing min_axis_mm

    if a_min > a_max:
        return None  # no single position clears both constraints for this anchor
    if a_min >= length_mm / 2.0:
        return None  # would push anchors past the panel midpoint -- not a valid inward move

    relevant_openings = [o for o in openings
                          if o.width_mm > 0 and o.height_mm > 0
                          and (o.sill_mm + o.height_mm) >= height_mm]
    # Malformed openings (non-positive width/height) contribute no interval
    # here -- they cannot be solved for a meaningful x-range -- and are
    # instead caught unconditionally by check_opening_void_clash()'s own
    # conservative-FAIL path when the caller re-verifies the selected
    # position; a position generated here can never make that FAIL go away.

    x1 = a_min + delta
    x1_ceiling = a_max + delta

    # Re-check every opening until a full pass changes nothing, or until a
    # fixed, input-size-bounded number of passes is exhausted -- never an
    # open-ended/stepped search. x1 only ever increases across escapes (each
    # opening's blocked range is finite), which guarantees termination
    # within, at most, one pass per opening.
    changed = True
    passes = 0
    max_passes = 2 * len(relevant_openings) + 1
    while changed and passes <= max_passes:
        changed = False
        passes += 1
        for o in relevant_openings:
            lo_o, hi_o = o.x_mm, o.x_mm + o.width_mm
            if lo_o <= x1 <= hi_o:
                x1 = math.nextafter(hi_o, math.inf)
                changed = True
                continue
            x2 = 2.0 * cog_x_mm - x1
            if lo_o <= x2 <= hi_o:
                x2 = math.nextafter(lo_o, -math.inf)
                x1 = 2.0 * cog_x_mm - x2
                changed = True

    if x1 > x1_ceiling:
        return None  # every remaining position is either opening-blocked or past the axis-spacing bound

    x2 = 2.0 * cog_x_mm - x1
    return x1, x2


# --------------------------------------------------------------------------
# Clash checks (plan section 18) -- opening void is deterministic PASS/FAIL;
# 3D reinforcement clash is UNKNOWN unless explicitly confirmed in the input.
# --------------------------------------------------------------------------

def check_opening_void_clash(x1_mm: float, x2_mm: float, openings: tuple[Opening, ...],
                              height_mm: float) -> CheckResult:
    """Anchors sit at the panel's top edge (brief Figure 1). A void can only
    clash with an anchor if it reaches that top edge AND overlaps in x.

    A malformed opening (non-positive width/height -- Test 10 finding) makes
    the x-overlap test `o.x_mm <= x <= o.x_mm + o.width_mm` an empty/backwards
    range that can never be true, silently reporting "no clash" regardless of
    the real anchor position. Rather than evaluate a geometrically meaningless
    opening, conservatively FAIL: we cannot verify the anchor is clear of it.
    """
    malformed = [o for o in openings if o.width_mm <= 0 or o.height_mm <= 0]
    if malformed:
        return CheckResult(
            check_name="opening_void_clash", state=CheckState.FAIL, scope=CheckScope.CANDIDATE,
            rule_reference="brief 3.5 step 8",
            explanation=(f"opening(s) {[o.id for o in malformed]} have non-positive width/height and cannot be "
                         "verified clear of the anchor positions; conservatively treated as a clash"),
        )
    for o in openings:
        top_from_top_mm = height_mm - (o.sill_mm + o.height_mm)
        reaches_top = top_from_top_mm <= 0
        for x in (x1_mm, x2_mm):
            if reaches_top and o.x_mm <= x <= o.x_mm + o.width_mm:
                return CheckResult(
                    check_name="opening_void_clash", state=CheckState.FAIL, scope=CheckScope.CANDIDATE,
                    rule_reference="brief 3.5 step 8",
                    explanation=f"anchor at x={x:.0f}mm lands inside opening {o.id!r} which reaches the panel top edge",
                )
    return CheckResult(
        check_name="opening_void_clash", state=CheckState.PASS, scope=CheckScope.CANDIDATE,
        rule_reference="brief 3.5 step 8", explanation="no anchor position overlaps an opening reaching the top edge",
    )


def check_clutch_match(anchor: AnchorType, clutch_used: str) -> CheckResult:
    """Brief 3.3: 'every anchor needs its matching clutch (same row)'; brief
    3.6 hard stop: 'no mismatched anchor/clutch'. In this architecture the
    Agent always derives clutch_used from anchor.clutch (catalogue.py) rather
    than accepting an externally-supplied clutch, so this check always PASSes
    in a real run -- the mismatch is prevented by construction. It is kept as
    an explicit, independently unit-tested deterministic check (rather than
    an unreachable no-op) so the hard stop is provably enforced, not just
    assumed impossible.
    """
    matches = clutch_used == anchor.clutch
    return CheckResult(
        check_name="clutch_match", state=(CheckState.PASS if matches else CheckState.FAIL),
        scope=CheckScope.PROBLEM,
        rule_reference="brief 3.3 matching clutch / 3.6 hard stop",
        explanation=(f"clutch {clutch_used!r} matches {anchor.name}'s required {anchor.clutch!r}" if matches
                     else f"clutch {clutch_used!r} does NOT match {anchor.name}'s required {anchor.clutch!r}"),
    )


def check_supplementary_reinforcement(confirmed: bool | None) -> CheckResult:
    """PASS only if the input explicitly confirms the Zulage/supplementary
    reinforcement is placed (Figure 2). Absent or None => UNKNOWN, which is a
    PROBLEM-scope hard stop (brief 3.6: capacity is invalid without it).
    """
    if confirmed is True:
        return CheckResult(
            check_name="supplementary_reinforcement", state=CheckState.PASS, scope=CheckScope.PROBLEM,
            rule_reference="brief 3.3 / Figure 2 / 3.6 hard stop",
            explanation="input explicitly confirms supplementary reinforcement is placed",
        )
    if confirmed is False:
        return CheckResult(
            check_name="supplementary_reinforcement", state=CheckState.FAIL, scope=CheckScope.PROBLEM,
            rule_reference="brief 3.3 / Figure 2 / 3.6 hard stop",
            explanation="input explicitly states supplementary reinforcement is NOT placed; tabulated capacity is invalid",
        )
    return CheckResult(
        check_name="supplementary_reinforcement", state=CheckState.UNKNOWN, scope=CheckScope.PROBLEM,
        rule_reference="brief 3.3 / Figure 2 / 3.6 hard stop",
        explanation="input does not confirm whether the anchor's supplementary reinforcement (Zulage) is placed; "
                    "tabulated capacity cannot be claimed without it",
    )


# --------------------------------------------------------------------------
# Capacity checks (plan section 24) -- CANDIDATE scope
# --------------------------------------------------------------------------

def check_capacity(anchor: AnchorType, state: HandlingState, demand_kn: float,
                    strength_column_mpa: int | None, anchor_id: str) -> CheckResult:
    if strength_column_mpa is None:
        return CheckResult(
            check_name="axial_capacity", state=CheckState.UNKNOWN, scope=CheckScope.PROBLEM,
            handling_state=state.name, anchor_id=anchor_id,
            rule_reference="brief 3.6 hard stop (no lift below first-lift strength)",
            explanation="achieved concrete strength is below every tabulated column for this state",
        )
    capacity = catalogue.lookup_capacity(anchor.name, strength_column_mpa)
    utilisation = demand_kn / capacity if capacity else float("inf")
    return CheckResult(
        check_name="axial_capacity", state=(CheckState.PASS if demand_kn <= capacity else CheckState.FAIL),
        scope=CheckScope.CANDIDATE, demand=demand_kn, capacity=capacity, utilisation=utilisation,
        handling_state=state.name, anchor_id=anchor_id,
        rule_reference="brief 3.3 F <= N_zul",
        explanation=f"{state.name}: demand {demand_kn:.1f}kN vs {anchor.name}@{strength_column_mpa}MPa capacity {capacity:.1f}kN",
    )
