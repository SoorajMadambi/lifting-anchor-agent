"""Agent orchestration loop (plan section 22) + the final Invariant Guard
(plan section 25).

The Agent drives the deterministic tools directly -- the workflow below is
fixed control flow, not something an LLM narrates step by step. The Reasoner
is consulted at exactly one point: choosing the next catalogue candidate to
try after the current one fails (or none, if the first candidate passes).
Whatever the Reasoner returns is validated (validator.py) and then the
final status is *always* recomputed from the raw deterministic check
results by enforce_invariants() -- the Reasoner's own opinion of the status
is never read, let alone trusted.
"""
from __future__ import annotations

from dataclasses import replace

from liftagent import catalogue, engineering, ingest
from liftagent.reasoner import AgentAction, Reasoner, ReasonerDecision, TryAnchorAction
from liftagent.schema import (
    AgentResult, AnchorPlacement, Candidate, CheckResult, CheckScope, CheckState,
    ElementInput, PositionAttempt, PositionIterationResult, RFI, Status,
)
from liftagent.tools import ToolBox
from liftagent.validator import ActionRejected, validate_decision

# Machine-readable reason codes (agent.py is the sole authority on these --
# report.py/visualize.py only translate them for display, never re-derive them).
REASON_UNRESOLVED_GEOMETRY = "UNRESOLVED_GEOMETRY_CONFLICT"
REASON_TURN_METHOD_UNCONFIRMED = "TURN_METHOD_UNCONFIRMED"
REASON_REINFORCEMENT_UNCONFIRMED = "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED"
REASON_NO_CANDIDATE_SATISFIES_REQUIREMENTS = "NO_CANDIDATE_SATISFIES_REQUIREMENTS"

# Static assumptions -- true regardless of the specific run's concrete class.
# The concrete-strength assumption is NOT here because it names a specific
# class/MPa value and must describe the actual run -- see
# _concrete_strength_assumption() below (Test 9 follow-up: this used to be a
# static string hardcoding "C32/40", which was factually wrong for any other
# input class).
STATIC_ASSUMPTIONS: tuple[str, ...] = (
    "Self-weight is derived deterministically from geometry x density, not from the brief's indicative "
    "72.6 kN figure -- that figure's provenance cannot be established from the supplied geometry, so it "
    "is reported as a reference value only, never used as an authoritative engineering input.",
    "No sling/rigging angle is provided in the input; the engineering core assumes vertical slings "
    "(z=1.0), which is consistent with the rig decision -- a spreader is mandatory for every catalogue "
    "anchor against this panel's 180mm thickness, so beta=0 always applies here regardless.",
    "The POC does not perform continuous anchor-position optimisation: the placement policy is "
    "trial-at-0.207L then shift-to-CoG, with a single bounded 'move in' fallback (brief 3.5 step 11) "
    "if that shifted position fails edge distance, axis spacing, or opening-void clearance -- a "
    "closed-form interval derivation (the edge/axis-feasible range for the inward offset, minus every "
    "top-edge-reaching opening's blocked sub-interval) against this anchor's own catalogue "
    "min_edge_mm/min_axis_mm and the panel's openings, never a stepped search or an optimiser; the "
    "resulting candidate is always re-verified against the real, unchanged checks before being used. "
    "Anchor-capacity and reaction failures are never addressed by trying a different spacing: the "
    "CoG-centered two-anchor statics model makes reactions invariant to spacing, so only "
    "position-dependent checks can be affected by this fallback; capacity failures are handled by "
    "trying the next catalogue candidate instead. If no feasible fallback position exists, or another "
    "constraint (wall thickness, capacity) fails regardless, the candidate fails and the next "
    "catalogue candidate is tried (brief 3.5 steps 5-11).",
    "Existing 'Wire Loop Box' cast-in proxies found in WC001.ifc are not lifting anchors (no matching "
    "row in the brief 3.3 catalogue) and are excluded from the engineering calculation.",
    "Opening voids were not extracted from WC001.ifc's raw faceted-BREP geometry (see ifc_geometry.py); "
    "that source's opening count is treated as UNKNOWN, not confirmed-zero.",
)


def _concrete_strength_assumption(concrete) -> str:
    """Generated from THIS run's actual concrete.class_label/cube_strength_mpa
    (already present on ConcreteSpec/ElementInput) -- never a hardcoded class.
    Preserves the existing strength-column selection logic (engineering.py's
    select_strength_column) unchanged; this only describes it."""
    return (
        f"{concrete.class_label} is interpreted using the cube-strength value "
        f"({concrete.cube_strength_mpa:.0f} MPa) when matching the lifting-anchor catalogue strength "
        "columns (brief 3.2/3.3) -- the only reading consistent with the brief's own C32/40 worked "
        "example (72.6/80 = 0.91 against ARL-42's '@35 MPa' column). See DESIGN_NOTE.md."
    )


def _usable_sources(element: ElementInput):
    return [s for s in element.geometry_sources if s.length_mm is not None]


def _finalize_governing(checks: list[CheckResult]) -> tuple[list[CheckResult], CheckResult | None]:
    """Marks the single worst/binding axial_capacity check as governing=True
    IN PLACE within the returned checks list, and returns that SAME object as
    the governing check -- guaranteeing `checks` and `governing_check` can
    never disagree, because they share one object rather than two
    independently-constructed copies (see the governing-flag inconsistency
    this fixes: a prior version re-derived the governing check separately via
    dataclasses.replace() without writing the result back into `checks`)."""
    best_idx: int | None = None
    best_utilisation: float | None = None
    for i, c in enumerate(checks):
        if c.check_name == "axial_capacity" and c.utilisation is not None:
            if best_utilisation is None or c.utilisation > best_utilisation:
                best_idx, best_utilisation = i, c.utilisation
    if best_idx is None:
        return checks, None
    governing = replace(checks[best_idx], governing=True)
    updated = list(checks)
    updated[best_idx] = governing
    return updated, governing


def _mark_capacity_conditional(checks: list[CheckResult], reinforcement_check: CheckResult) -> list[CheckResult]:
    """A tabulated axial_capacity PASS is not an unconditional acceptance
    while the anchor's supplementary reinforcement (Zulage, Figure 2) is
    UNKNOWN or FAIL -- brief 3.3: 'the tabulated capacity is only valid if
    the anchor's supplementary reinforcement is placed'. This does NOT
    change the check's PASS/FAIL state (the raw demand-vs-catalogue-capacity
    comparison is still correct and worth keeping as a reference number,
    and changing it to UNKNOWN would make the candidate-search loop exhaust
    the whole catalogue for no engineering reason, since HOLD already fires
    independently via reinforcement_state in enforce_invariants). It only
    annotates WHY that PASS/FAIL cannot yet be read as verified/final.
    Updates the SAME list in place -- see _finalize_governing for why that
    discipline matters (a prior bug came from constructing a second,
    divergent copy instead)."""
    if reinforcement_check.state == CheckState.PASS:
        return checks
    note = (f"conditional on supplementary reinforcement (currently {reinforcement_check.state.value}): "
            "the tabulated capacity is not claimable as verified until Figure 2's Zulage is confirmed placed")
    updated = list(checks)
    for i, c in enumerate(updated):
        if c.check_name == "axial_capacity":
            updated[i] = replace(c, conditional_on=note, explanation=f"{c.explanation} [CONDITIONAL: {note}]")
    return updated


# All three of these depend on x1/x2 (brief 3.3 edge/axis, 3.5 step 8 opening
# clash) -- unlike reaction_nonnegative/axial_capacity, which are algebraically
# invariant to spacing as long as the pair stays CoG-centered (see
# DESIGN_NOTE.md "Bounded position iteration"), so only these three can ever
# be fixed by trying a different placement of the SAME anchor.
POSITION_DEPENDENT_CHECK_NAMES = ("edge_distance", "axis_spacing", "opening_void_clash")


def _position_dependent_checks(tools: ToolBox, anchor, lo: float, hi: float, length_mm: float,
                                thickness_mm: float, openings, height_mm: float) -> tuple[list[CheckResult], list[str]]:
    """Every check whose PASS/FAIL depends on the candidate x1/x2, evaluated
    together at one position -- used identically for the trial position and
    the bounded 'move in' fallback, so both are judged by the same complete
    set of position-dependent checks (not edge/axis alone)."""
    checks = tools.check_edge_axis_wall(anchor, lo, hi, length_mm, thickness_mm)
    checks.append(tools.check_opening_void_clash(lo, hi, openings, height_mm))
    failed = [c.check_name for c in checks
              if c.check_name in POSITION_DEPENDENT_CHECK_NAMES and c.state == CheckState.FAIL]
    return checks, failed


def _evaluate_candidate(tools: ToolBox, anchor_type: str, authoritative, concrete, handling_states,
                         reinforcement_confirmed: bool | None):
    anchor = tools.get_anchor(anchor_type)
    length_mm, thickness_mm, height_mm = authoritative.length_mm, authoritative.thickness_mm, authoritative.height_mm

    self_weight = tools.compute_self_weight(authoritative, concrete.density_kg_per_m3, concrete.class_label)
    cog = tools.compute_cog(authoritative)

    x1_trial, x2_trial = tools.trial_placement(length_mm)
    x1_trial, x2_trial = tools.shift_to_cog(x1_trial, x2_trial, length_mm, cog.x_mm)
    lo_trial, hi_trial = min(x1_trial, x2_trial), max(x1_trial, x2_trial)

    # brief 3.5 step 7/11: enforce edge/axis/opening-clash at the trial
    # position first. If (and only if) any of those position-dependent
    # checks fail, attempt the bounded "move in" search (step 11) -- at
    # most one fallback position, never a search or optimiser -- before
    # giving up on this candidate.
    position_checks, trial_failed = _position_dependent_checks(
        tools, anchor, lo_trial, hi_trial, length_mm, thickness_mm, authoritative.openings, height_mm)

    lo, hi = lo_trial, hi_trial
    position_iteration = PositionIterationResult(attempted=False)

    if trial_failed:
        reason = f"initial trial position (a=0.207L) fails {', '.join(trial_failed)} for {anchor.name}"
        trial_attempt = PositionAttempt(x1_mm=lo_trial, x2_mm=hi_trial, result=CheckState.FAIL,
                                         failed_checks=tuple(trial_failed))
        feasible = tools.find_feasible_inward_position(
            anchor, length_mm, cog.x_mm, authoritative.openings, height_mm)

        if feasible is not None:
            new_lo, new_hi = feasible
            new_checks, still_failed = _position_dependent_checks(
                tools, anchor, new_lo, new_hi, length_mm, thickness_mm, authoritative.openings, height_mm)
            fallback_attempt = PositionAttempt(
                x1_mm=new_lo, x2_mm=new_hi,
                result=(CheckState.FAIL if still_failed else CheckState.PASS),
                failed_checks=tuple(still_failed),
            )
            if not still_failed:
                lo, hi = new_lo, new_hi
                position_checks = new_checks
            position_iteration = PositionIterationResult(
                attempted=True, reason=reason,
                initial_x1_mm=lo_trial, initial_x2_mm=hi_trial,
                attempts=(trial_attempt, fallback_attempt),
                selected_x1_mm=(lo if not still_failed else None),
                selected_x2_mm=(hi if not still_failed else None),
                method=("closed-form interval search: derives the full edge/axis-feasible interval "
                        "for the inward offset, subtracts every top-edge-reaching opening's "
                        "blocked sub-interval(s), and selects the smallest remaining feasible "
                        "position (brief 3.5 step 11 'move in'); not an optimiser or a stepped "
                        "search -- still at most one candidate position is tried, and it is always "
                        "re-verified against the real, unchanged checks above."),
            )
        else:
            position_iteration = PositionIterationResult(
                attempted=True, reason=reason,
                initial_x1_mm=lo_trial, initial_x2_mm=hi_trial,
                attempts=(trial_attempt,),
                selected_x1_mm=None, selected_x2_mm=None,
                method=("closed-form interval search: no position satisfies min_edge_mm, "
                        "min_axis_mm, and every opening's clearance simultaneously "
                        "(brief 3.5 step 11 'move in')."),
            )

    checks: list[CheckResult] = list(position_checks)

    rig = tools.decide_rig(anchor, thickness_mm)
    z = 1.0  # vertical slings -- see STATIC_ASSUMPTIONS

    anchors = (
        AnchorPlacement(id="A1", anchor_type=anchor.name, clutch=anchor.clutch, x_mm=lo, y_mm=0.0),
        AnchorPlacement(id="A2", anchor_type=anchor.name, clutch=anchor.clutch, x_mm=hi, y_mm=0.0),
    )

    for state in handling_states:
        f_total = tools.total_load_for_state(state, self_weight.value_kn, z, length_mm, height_mm)
        r1, r2 = tools.compute_reactions(f_total, lo, hi, cog.x_mm)
        strength_col = tools.select_strength_column(concrete, state.strength_requirement)
        for anchor_id, r in (("A1", r1), ("A2", r2)):
            checks.append(CheckResult(
                check_name="reaction_nonnegative", state=(CheckState.PASS if r >= 0 else CheckState.FAIL),
                scope=CheckScope.CANDIDATE, demand=r, handling_state=state.name, anchor_id=anchor_id,
                rule_reference="brief 3.5 step 4 (plumb lift, CoG between the picks)",
                explanation=f"{state.name} {anchor_id}: reaction {r:.2f}kN (must be >=0 for a stable 2-point lift)",
            ))
            checks.append(tools.check_capacity(anchor, state, max(r, 0.0), strength_col, anchor_id))

    reinforcement_check = tools.check_supplementary_reinforcement(reinforcement_confirmed)
    checks = _mark_capacity_conditional(checks, reinforcement_check)
    checks.append(reinforcement_check)
    checks.append(tools.check_clutch_match(anchor, anchor.clutch))  # always matches by construction; see engineering.py

    candidate_checks = [c for c in checks if c.scope == CheckScope.CANDIDATE]
    passed = all(c.state == CheckState.PASS for c in candidate_checks)
    return anchors, checks, rig, self_weight, cog, passed, position_iteration


def enforce_invariants(*, geometry_resolved: bool, turn_confirmed: bool, reinforcement_state: CheckState,
                        candidate_passed: bool, candidates_exhausted: bool) -> tuple[Status, str | None]:
    """The final safety boundary (plan section 25). Recomputes status --
    and WHY -- purely from deterministic facts. Nothing the Reasoner said is
    read here; status and reason are computed together, from the same
    priority order, so they can never drift apart or contradict each other.

    Invariant: status == ACCEPT_PROVISIONAL if and only if reason is None.
    Every other status always carries an explicit machine-readable reason.
    """
    if not geometry_resolved:
        return Status.HOLD, REASON_UNRESOLVED_GEOMETRY
    if not turn_confirmed:
        return Status.HOLD, REASON_TURN_METHOD_UNCONFIRMED
    if reinforcement_state != CheckState.PASS:
        return Status.HOLD, REASON_REINFORCEMENT_UNCONFIRMED
    if candidate_passed:
        return Status.ACCEPT_PROVISIONAL, None
    if candidates_exhausted:
        return Status.REJECT, REASON_NO_CANDIDATE_SATISFIES_REQUIREMENTS
    return Status.ITERATE, "ITERATION_IN_PROGRESS"


def run_agent(element: ElementInput, reasoner: Reasoner,
              reinforcement_confirmed: bool | None = None) -> AgentResult:
    sources = _usable_sources(element)
    if not sources:
        raise ValueError("no usable geometry source available (all sources failed extraction)")

    conflicts = ingest.detect_geometry_conflict(element.geometry_sources)
    geometry_resolved = len(conflicts) == 0
    # Whitelist, not a blacklist (Test 10 finding): an unrecognised turn_method
    # string must NOT be silently treated as confirmed just because it isn't
    # literally "UNCONFIRMED"/None/"".
    turn_confirmed = element.production.turn_method in engineering.KNOWN_TURN_METHODS

    authoritative = next(
        (s for s in sources if s.role.value == "AUTHORITATIVE_DESIGN"), sources[0]
    )
    handling_states = engineering.enumerate_handling_states(element.production)

    tools = ToolBox()
    available = list(catalogue.CANDIDATE_TRY_ORDER)
    tried: list[str] = []
    last_failure_reason = ""

    last_candidate: Candidate | None = None
    last_self_weight = None
    last_cog = None
    candidate_passed = False
    candidates_exhausted = False

    for _ in range(len(available)):
        decision = reasoner.decide_next_candidate(tried, available, last_failure_reason)
        try:
            decision = validate_decision(decision, tried, available)
        except ActionRejected:
            # A rejected decision is NEVER trusted or acted on -- fall back to
            # the deterministic try-order instead of stopping the run.
            remaining = [a for a in available if a not in tried]
            if not remaining:
                candidates_exhausted = True
                break
            decision = ReasonerDecision(action=AgentAction.TRY_ANCHOR,
                                         try_anchor=TryAnchorAction(anchor_type=remaining[0]))

        if decision.action == AgentAction.ESCALATE_HOLD:
            candidates_exhausted = True
            break

        anchor_type = decision.try_anchor.anchor_type
        tried.append(anchor_type)

        anchors, checks, rig, self_weight, cog, passed, position_iteration = _evaluate_candidate(
            tools, anchor_type, authoritative, element.concrete, handling_states, reinforcement_confirmed,
        )
        last_self_weight, last_cog = self_weight, cog
        checks, governing_check = _finalize_governing(checks)

        lo, hi = min(a.x_mm for a in anchors), max(a.x_mm for a in anchors)
        last_candidate = Candidate(
            candidate_id=f"candidate-{anchor_type}", anchor_type=anchor_type,
            clutch=anchors[0].clutch, x1_mm=lo, x2_mm=hi,
            anchors=anchors, checks=tuple(checks), governing_check=governing_check, rig=rig,
            position_iteration=position_iteration,
        )

        if passed:
            candidate_passed = True
            break
        fails = [c for c in checks if c.scope == CheckScope.CANDIDATE and c.state == CheckState.FAIL]
        last_failure_reason = "; ".join(f"{c.check_name}: {c.explanation}" for c in fails[:3])
    else:
        candidates_exhausted = True

    last_checks = last_candidate.checks if last_candidate is not None else ()
    reinforcement_checks = [c for c in last_checks if c.check_name == "supplementary_reinforcement"]
    reinforcement_state = reinforcement_checks[-1].state if reinforcement_checks else CheckState.UNKNOWN

    status, reason = enforce_invariants(
        geometry_resolved=geometry_resolved, turn_confirmed=turn_confirmed,
        reinforcement_state=reinforcement_state, candidate_passed=candidate_passed,
        candidates_exhausted=candidates_exhausted,
    )

    # An illustrative anchor position must never become an accepted placement
    # just because the calculation succeeded -- resolved_candidate is ONLY
    # ever populated alongside ACCEPT_PROVISIONAL; every other status keeps
    # the same computed Candidate strictly on the illustrative side, and a
    # Reasoner/LLM has no channel through which it could move it across.
    resolved_candidate = last_candidate if status == Status.ACCEPT_PROVISIONAL else None
    illustrative_candidate = last_candidate if status != Status.ACCEPT_PROVISIONAL else None

    rfis: list[RFI] = []
    if not geometry_resolved:
        rfis.append(RFI(id="RFI-GEOMETRY", rule_reference="brief 3.6 hard stop 1",
                         message="Resolve geometry discrepancy before finalising placement: " + "; ".join(conflicts)))
    if not turn_confirmed:
        rfis.append(RFI(id="RFI-TURN-METHOD", rule_reference="brief 3.6 hard stop 1",
                         message="Confirm turn method (currently UNCONFIRMED). The conservative free-crane "
                                 "weak-axis case has been run in the meantime, per brief 3.5 step 1."))
    if reinforcement_state != CheckState.PASS:
        rfis.append(RFI(id="RFI-SUPPLEMENTARY-REINFORCEMENT", rule_reference="brief 3.3 / Figure 2 / 3.6 hard stop",
                         message="Confirm the anchor's supplementary reinforcement (Zulage, Figure 2) is detailed "
                                 "and placed at each anchor location; the tabulated capacity is invalid without it."))
    if last_self_weight is not None and last_self_weight.warning:
        rfis.append(RFI(id="RFI-SELF-WEIGHT", rule_reference="brief Figure 1",
                         message="Explain/reconcile computed self-weight against the brief's reference value: "
                                 + last_self_weight.warning))

    self_weight_by_source = tuple(
        engineering.compute_self_weight(s, element.concrete.density_kg_per_m3,
                                         concrete_class=element.concrete.class_label)
        for s in sources
    )
    cog_by_source = tuple(engineering.compute_cog(s) for s in sources)

    return AgentResult(
        element_id=element.element_id,
        status=status,
        reason=reason,
        geometry_sources=element.geometry_sources,
        selected_geometry=(authoritative.source_name if geometry_resolved else None),
        cog=last_cog,
        self_weight=last_self_weight,
        resolved_candidate=resolved_candidate,
        illustrative_candidate=illustrative_candidate,
        rfis=tuple(rfis),
        rule_trace=tuple(tools.trace),
        assumptions=(_concrete_strength_assumption(element.concrete),) + STATIC_ASSUMPTIONS,
        cog_by_source=cog_by_source,
        self_weight_by_source=self_weight_by_source,
    )
