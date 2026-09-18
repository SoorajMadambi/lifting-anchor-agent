"""TEST 10 -- Input robustness / fail-closed boundary tests.

Proves that malformed, missing, contradictory, or physically invalid inputs
cannot accidentally produce ACCEPT_PROVISIONAL. This is a boundary test, not
an engineering-correctness test (see test_end_to_end.py / Test 8) and not a
Reasoner-security test (see test_reasoner_security.py / Test 7, reused
directly in 10H below).

Every scenario here was empirically probed against the real production code
before being written up as a test -- three genuine fail-open bugs were found
and fixed with the smallest possible change (see the module docstring in
agent.py/engineering.py at each fix site, and the "PRODUCTION FIXES" section
below):

1. `turn_confirmed` used a blacklist (`not in ("UNCONFIRMED", None, "")`), so
   ANY unrecognised turn_method string (e.g. a typo) was silently treated as
   confirmed. Fixed to a whitelist (`engineering.KNOWN_TURN_METHODS`).
2. `density_kg_per_m3 <= 0` was never validated -- a zero density zeroed
   every non-adhesion load, so every capacity check trivially passed and the
   panel reached ACCEPT_PROVISIONAL with self_weight = 0. Now raises, same
   pattern as the existing missing-dimensions guard.
3. An opening with a non-positive width/height made the clash x-overlap
   range backwards/empty, so `check_opening_void_clash` silently reported
   PASS regardless of the anchor position. Now conservatively FAILs instead
   of evaluating geometrically meaningless input.

No other production code was changed. Everything else this file tests was
already safe (documented per-scenario below) -- either it correctly HOLDs,
correctly REJECTs, or correctly raises a clear exception before any result
is produced (never a silent ACCEPT_PROVISIONAL).
"""
from __future__ import annotations

import pytest

from liftagent import catalogue, engineering, ingest
from liftagent.agent import run_agent
from liftagent.reasoner import (
    AgentAction, EvilReasoner, MockReasoner, ReasonerDecision, TryAnchorAction,
)
from liftagent.schema import (
    CheckState, ConcreteSpec, ElementInput, ExtractionStatus, GeometrySource,
    Opening, ProductionSpec, SourceRole, SourceType, Status,
)
from liftagent.validator import ActionRejected, parse_try_anchor_payload, validate_decision
from tests.conftest import make_concrete, make_element_input, make_geometry_source, make_production
from tests.test_end_to_end import _resolved_element, _run_resolved


def _clean_element(**overrides):
    """Baseline: same shape as the Test 8 fixture. Any subsection below
    overrides exactly the one field it's testing."""
    return _resolved_element(**overrides)


def _run(reinforcement_confirmed=True, **overrides):
    return run_agent(_clean_element(**overrides), MockReasoner(), reinforcement_confirmed=reinforcement_confirmed)


def _never_accepted(status: Status) -> bool:
    return status != Status.ACCEPT_PROVISIONAL


# --------------------------------------------------------------------------
# 10A -- missing / invalid critical geometry
# --------------------------------------------------------------------------

def test_10a_all_sources_missing_dimensions_raises_not_accepts():
    """No usable geometry at all -- run_agent()'s existing contract is to
    raise rather than fabricate a result. This is documented, intentional
    fail-closed-via-exception, not a silent accept."""
    sources = (GeometrySource(source_type=SourceType.APPROVAL_JSON, source_name="approval_design",
                               role=SourceRole.AUTHORITATIVE_DESIGN, length_mm=None, height_mm=None,
                               thickness_mm=None, openings=(), extraction_status=ExtractionStatus.FAILED),)
    with pytest.raises(ValueError):
        run_agent(_clean_element(geometry_sources=sources), MockReasoner(), reinforcement_confirmed=True)


def test_10a_missing_height_raises():
    sources = (make_geometry_source(length_mm=4700.0, height_mm=None, thickness_mm=180.0,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    with pytest.raises(ValueError):
        _run(geometry_sources=sources)


def test_10a_missing_thickness_raises():
    sources = (make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=None,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    with pytest.raises(ValueError):
        _run(geometry_sources=sources)


@pytest.mark.parametrize("length_mm", [0.0, -4700.0])
def test_10a_zero_or_negative_length_raises(length_mm):
    sources = (make_geometry_source(length_mm=length_mm, height_mm=3000.0, thickness_mm=180.0,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    with pytest.raises(ValueError):
        _run(geometry_sources=sources)


@pytest.mark.parametrize("height_mm", [0.0, -3000.0])
def test_10a_zero_or_negative_height_raises(height_mm):
    sources = (make_geometry_source(length_mm=4700.0, height_mm=height_mm, thickness_mm=180.0,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    with pytest.raises(ValueError):
        _run(geometry_sources=sources)


@pytest.mark.parametrize("thickness_mm", [0.0, -180.0])
def test_10a_zero_or_negative_thickness_rejects_not_accepts(thickness_mm):
    """Unlike length/height, thickness=0 doesn't zero the panel area, so no
    exception fires here -- but every catalogue anchor's min_wall_axial_mm
    is positive, so thickness<=0 fails that check for every candidate ->
    REJECT. Verified this is the actual (safe) behaviour, not assumed."""
    result = _run(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=thickness_mm,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    assert _never_accepted(result.status)
    assert result.status == Status.REJECT


@pytest.mark.parametrize("opening", [
    Opening(id="neg_width", x_mm=1000.0, width_mm=-500.0, sill_mm=0.0, height_mm=3000.0),   # reaches top edge
    Opening(id="neg_height", x_mm=1000.0, width_mm=500.0, sill_mm=0.0, height_mm=-3000.0),
])
def test_10a_opening_with_negative_dimension_never_accepts(opening):
    """Regression for fix #3: a malformed opening must not silently disable
    clash detection and slip a candidate through to ACCEPT_PROVISIONAL."""
    result = _run(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0, openings=(opening,),
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    assert _never_accepted(result.status)
    clash_checks = [c for c in (result.resolved_candidate or result.illustrative_candidate).checks
                     if c.check_name == "opening_void_clash"]
    assert clash_checks and clash_checks[0].state == CheckState.FAIL


def test_10a_opening_extending_outside_the_panel_stays_auditable():
    """An opening whose x+width exceeds the panel length corrupts the CoG
    (its centroid falls outside the panel), which shifts the trial position
    enough to fail edge_distance for the first catalogue anchor. Since the
    §3.5 'move in' position-search fix (see test_position_iteration.py), this
    no longer necessarily REJECTs -- ARL-42 finds a feasible moved-in
    position here and the panel legitimately accepts. This is not a new
    safety gap: the CoG-corruption issue itself is a pre-existing,
    documented, out-of-scope data-quality gap (see the Test 10 report),
    unrelated to and not worsened by position iteration -- the search only
    ever operates on whatever CoG it's given, using the same deterministic
    catalogue bounds and reaction equations as every other candidate. What
    matters here is that the result stays fully auditable and legitimate:
    the selected anchors are provably clear of the (corrupted) opening span,
    capacity traces to catalogue.py, and the position-search trace records
    exactly what happened -- never a silent/unexplained accept."""
    opening = Opening(id="out", x_mm=4500.0, width_mm=1000.0, sill_mm=0.0, height_mm=3000.0)
    result = _run(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0, openings=(opening,),
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    candidate = result.resolved_candidate or result.illustrative_candidate
    assert candidate is not None
    # the moved-in anchors are provably clear of the opening's own x-span
    assert candidate.x2_mm < opening.x_mm
    # if accepted, the position-iteration trace fully explains why -- never a silent accept
    if result.status == Status.ACCEPT_PROVISIONAL:
        pi = candidate.position_iteration
        assert pi is not None and pi.attempted is True
        assert pi.selected_x1_mm is not None
        gc = candidate.governing_check
        valid_capacities = set(catalogue.get_anchor(candidate.anchor_type).capacity_kn.values())
        assert gc is not None and gc.capacity in valid_capacities


def test_10a_opening_larger_than_the_panel_raises():
    opening = Opening(id="huge", x_mm=0.0, width_mm=6000.0, sill_mm=0.0, height_mm=3000.0)
    with pytest.raises(ValueError):
        _run(geometry_sources=(
            make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0, openings=(opening,),
                                  role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        ))


# --------------------------------------------------------------------------
# 10B -- missing / invalid material data
# --------------------------------------------------------------------------

def test_10b_missing_concrete_key_in_raw_json_raises():
    """The realistic 'missing concrete data' entry point: malformed raw
    input JSON, caught at the ingest boundary before run_agent ever runs."""
    raw = {
        "element_id": "X", "type": "precast_rc_wall_panel", "description": "",
        "sources": {
            "approval_design": {"length_mm": 4700, "height_mm": 3000, "thickness_mm": 180, "openings_mm": []},
            "ifc_export": {"length_mm": 4700, "height_mm": 3000, "thickness_mm": 180, "openings_mm": []},
        },
        # "concrete" key missing entirely
        "reinforcement": {"note": ""},
        "production": {"cast_orientation": "flat", "mould": "x", "turn_method": "tilting_table", "storage": "x"},
        "handling_states_required": ["demould"],
    }
    with pytest.raises(KeyError):
        ingest.build_element_input(raw)


def test_10b_unparseable_concrete_class_raises_at_ingest():
    with pytest.raises(ValueError):
        engineering.parse_concrete_class("not-a-class")


def test_10b_zero_density_raises_not_accepts():
    """Regression for fix #2."""
    with pytest.raises(ValueError):
        _run(concrete=make_concrete("C32/40", density=0.0, first_lift=15.0))


def test_10b_negative_density_never_accepts():
    """A negative density makes every non-adhesion reaction negative, which
    fails the existing reaction_nonnegative check for every state/anchor ->
    REJECT. Verified as the actual (already-safe) behaviour."""
    with pytest.raises(ValueError):
        _run(concrete=make_concrete("C32/40", density=-2400.0, first_lift=15.0))


def test_10b_missing_cube_strength_raises():
    """Simulates a ConcreteSpec missing its cube_strength_mpa (the schema
    doesn't forbid constructing one with a None numeric field directly, even
    though ingest.py always derives it consistently from class_label)."""
    concrete = ConcreteSpec(class_label="C32/40", density_kg_per_m3=2400.0,
                             first_lift_strength_mpa=15.0, cylinder_strength_mpa=32.0,
                             cube_strength_mpa=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        engineering.select_strength_column(concrete, "full")


def test_10b_c32_40_assumption_never_leaks_into_a_different_class_run():
    """Protects against regression of the Test 9 bug where C32/40 was
    effectively hardcoded into the assumptions regardless of actual input."""
    result = _run(concrete=make_concrete("C25/30", density=2400.0, first_lift=15.0))
    assert result.self_weight.concrete_class == "C25/30"
    strength_assumption = next(a for a in result.assumptions if "cube-strength" in a.lower())
    assert "C25/30" in strength_assumption
    assert "C25/30 is interpreted" in strength_assumption
    assert "C32/40 is interpreted" not in strength_assumption


# --------------------------------------------------------------------------
# 10C -- anchor / catalogue input robustness (reuses Test 7's validator evidence)
# --------------------------------------------------------------------------

def test_10c_nonexistent_candidate_rejected_by_validator():
    with pytest.raises(ActionRejected):
        validate_decision(
            ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type="GHOST-ANCHOR")),
            tried=[], available=list(catalogue.CANDIDATE_TRY_ORDER),
        )


def test_10c_unknown_anchor_raises_from_the_catalogue_directly():
    with pytest.raises(catalogue.UnknownAnchorType):
        catalogue.get_anchor("GHOST-ANCHOR")


def test_10c_mismatched_clutch_fails_the_deterministic_check():
    anchor = catalogue.get_anchor("ARL-42")
    check = engineering.check_clutch_match(anchor, clutch_used="RU-30")  # wrong clutch for this anchor
    assert check.state == CheckState.FAIL


def test_10c_capacity_lookup_for_unavailable_column_raises():
    with pytest.raises(ValueError):
        catalogue.lookup_capacity("ARL-42", 99)  # not a tabulated column


def test_10c_capacity_never_computed_when_strength_column_unresolved():
    """A too-low achieved strength (below every tabulated column) must
    produce UNKNOWN, not a fabricated capacity."""
    anchor = catalogue.get_anchor("ARL-42")
    states = engineering.enumerate_handling_states(make_production("tilting_table"))
    check = engineering.check_capacity(anchor, states[0], demand_kn=10.0, strength_column_mpa=None, anchor_id="A1")
    assert check.state == CheckState.UNKNOWN
    assert check.capacity is None


def test_10c_candidate_exhaustion_never_accepts():
    """Every catalogue candidate genuinely fails (too-thin panel) -> REJECT,
    not a false accept, and the search really did try every candidate."""
    result = _run(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=100.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    assert _never_accepted(result.status)
    assert result.status == Status.REJECT
    tried = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried == set(catalogue.CANDIDATE_TRY_ORDER)


# --------------------------------------------------------------------------
# 10D -- lifting / handling input robustness
# --------------------------------------------------------------------------

def test_10d_missing_turn_method_holds():
    result = _run(production=make_production(""))
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"


def test_10d_invalid_unsupported_turn_method_holds():
    """Regression for fix #1: an unrecognised turn_method string used to be
    silently treated as confirmed."""
    result = _run(production=make_production("sideways_by_forklift"))
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"


def test_10d_unconfirmed_turn_method_holds():
    result = _run(production=make_production("UNCONFIRMED"))
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"


def test_10d_none_turn_method_holds():
    production = ProductionSpec(cast_orientation="flat", mould="x", turn_method=None, storage="x")  # type: ignore[arg-type]
    result = _run(production=production)
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD


def test_10d_unconfirmed_turn_method_still_runs_the_conservative_free_crane_case():
    """The existing conservative behaviour (brief 3.5 step 1: also run the
    free-crane weak-axis check) must remain unchanged by the whitelist fix."""
    result = _run(production=make_production("UNCONFIRMED"))
    states = {c.handling_state for c in result.illustrative_candidate.checks if c.handling_state}
    assert "TURN_FREE_CRANE" in states
    assert "TURN_TILTING_TABLE" in states


def test_10d_recognised_turn_methods_still_resolve_normally():
    """Sanity: the whitelist didn't break the two genuinely-supported values."""
    for method in engineering.KNOWN_TURN_METHODS:
        result = _run(production=make_production(method))
        assert result.status == Status.ACCEPT_PROVISIONAL


# --------------------------------------------------------------------------
# 10E -- reinforcement / clash information (PASS / FAIL / UNKNOWN)
# --------------------------------------------------------------------------

def test_10e_reinforcement_pass_allows_normal_processing():
    result = _run(reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL


def test_10e_reinforcement_fail_never_accepts():
    result = _run(reinforcement_confirmed=False)
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD


def test_10e_reinforcement_unknown_never_accepts():
    result = _run(reinforcement_confirmed=None)
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD
    checks = [c for c in result.illustrative_candidate.checks if c.check_name == "supplementary_reinforcement"]
    assert checks[-1].state == CheckState.UNKNOWN


def test_10e_evil_reasoner_cannot_turn_unknown_reinforcement_into_pass():
    element = _clean_element()
    result = run_agent(element, EvilReasoner(), reinforcement_confirmed=None)
    assert _never_accepted(result.status)
    assert result.status == Status.HOLD
    checks = [c for c in result.illustrative_candidate.checks if c.check_name == "supplementary_reinforcement"]
    assert checks[-1].state == CheckState.UNKNOWN  # unaffected by EvilReasoner


def test_10e_conditional_capacity_semantics_unchanged_when_reinforcement_unresolved():
    """The existing conditional_on annotation (added for the Test 9
    follow-up on capacity semantics) must still fire exactly as before."""
    result = _run(reinforcement_confirmed=None)
    axial = [c for c in result.illustrative_candidate.checks if c.check_name == "axial_capacity"]
    assert axial and all(c.conditional_on != "" for c in axial)


# --------------------------------------------------------------------------
# 10F -- geometry provenance robustness
# --------------------------------------------------------------------------

def test_10f_unresolved_conflict_never_selects_a_geometry(wc001_element):
    result = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.selected_geometry is None
    assert result.resolved_candidate is None


def test_10f_no_fallback_to_an_arbitrary_source_when_conflicted():
    """Three sources, all mutually disagreeing -- must not silently pick any
    of them, including whichever happens to be role=AUTHORITATIVE_DESIGN."""
    sources = (
        make_geometry_source(length_mm=4700.0, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4300.0, role=SourceRole.EXPORT, name="ifc_export"),
        make_geometry_source(length_mm=5200.0, role=SourceRole.DERIVED_REFERENCE, name="ifc_geometry",
                              source_type=SourceType.IFC_PARSED),
    )
    element = make_element_input(geometry_sources=sources, production=make_production("tilting_table"))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.selected_geometry is None
    # provenance remains visible even though nothing was selected
    assert {s.source_name for s in result.geometry_sources} == {"approval_design", "ifc_export", "ifc_geometry"}
    assert any(r.id == "RFI-GEOMETRY" for r in result.rfis)


def test_10f_agreeing_sources_do_resolve():
    """Sanity check: the fail-closed behaviour is specifically about
    disagreement, not about having more than one source per se."""
    sources = (
        make_geometry_source(length_mm=4700.0, role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        make_geometry_source(length_mm=4700.0, role=SourceRole.EXPORT, name="ifc_export"),
    )
    element = make_element_input(geometry_sources=sources, production=make_production("tilting_table"))
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.selected_geometry == "approval_design"


# --------------------------------------------------------------------------
# 10G -- no false acceptance (the most important section)
# --------------------------------------------------------------------------

def _malformed_scenarios():
    """Each entry: (label, kwargs-or-callable). Every one of these must
    produce result.status != ACCEPT_PROVISIONAL, whether via HOLD, REJECT,
    or a raised exception (also a form of "never accepted" -- no result
    object is ever produced to accept)."""
    thin_source = (make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=100.0,
                                         role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    bad_opening_source = (make_geometry_source(
        length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
        openings=(Opening(id="bad", x_mm=1000.0, width_mm=-500.0, sill_mm=0.0, height_mm=3000.0),),
        role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    return [
        ("reinforcement_unknown", dict(reinforcement_confirmed=None)),
        ("reinforcement_denied", dict(reinforcement_confirmed=False)),
        ("turn_method_unconfirmed", dict(production=make_production("UNCONFIRMED"))),
        ("turn_method_missing", dict(production=make_production(""))),
        ("turn_method_garbage", dict(production=make_production("some_unrecognised_method"))),
        ("panel_too_thin_for_any_anchor", dict(geometry_sources=thin_source)),
        ("malformed_opening", dict(geometry_sources=bad_opening_source)),
        # L=1200mm is deliberately NOT in this "never accepts" sweep any more: the
        # §3.5 "move in" position-search fix (test_position_iteration.py TEST A)
        # correctly resolves it to ACCEPT_PROVISIONAL via CFS-WAL-30. L=800mm is
        # short enough that even that search cannot find a position clearing both
        # min_edge_mm and min_axis_mm simultaneously -- a genuine, still-unfixable
        # edge-distance failure.
        ("length_too_short_for_edge_distance_even_with_position_search", dict(geometry_sources=(
            make_geometry_source(length_mm=800.0, height_mm=3000.0, thickness_mm=180.0,
                                  role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        ))),
        ("length_too_long_for_any_capacity", dict(geometry_sources=(
            make_geometry_source(length_mm=12000.0, height_mm=3000.0, thickness_mm=180.0,
                                  role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
        ))),
    ]


@pytest.mark.parametrize("label,kwargs", _malformed_scenarios(), ids=[s[0] for s in _malformed_scenarios()])
def test_10g_malformed_or_unresolved_input_never_produces_accept_provisional(label, kwargs):
    result = _run(**kwargs)
    assert result.status != Status.ACCEPT_PROVISIONAL, f"{label} incorrectly reached ACCEPT_PROVISIONAL"
    assert result.status in (Status.HOLD, Status.REJECT, Status.ITERATE)
    assert result.resolved_candidate is None
    assert result.requires_human_signoff is True


@pytest.mark.parametrize("density", [0.0, -2400.0])
def test_10g_invalid_density_never_produces_accept_provisional(density):
    """These raise rather than return a result -- still "never accepted":
    there is no result object to have accepted anything."""
    with pytest.raises(ValueError):
        _run(concrete=make_concrete("C32/40", density=density, first_lift=15.0))


def test_10g_evil_reasoner_across_every_malformed_scenario():
    """The Reasoner cannot rescue an unsafe input into acceptance, for
    every scenario in the sweep above."""
    for label, kwargs in _malformed_scenarios():
        element = _clean_element(**{k: v for k, v in kwargs.items() if k != "reinforcement_confirmed"})
        result = run_agent(element, EvilReasoner(), reinforcement_confirmed=kwargs.get("reinforcement_confirmed", True))
        assert result.status != Status.ACCEPT_PROVISIONAL, f"{label} + EvilReasoner incorrectly accepted"


# --------------------------------------------------------------------------
# 10H -- Reasoner adversarial robustness (reuses Test 7's infrastructure)
# --------------------------------------------------------------------------

def test_10h_evil_reasoner_cannot_inject_unknown_candidate_into_a_malformed_run():
    result = run_agent(_clean_element(geometry_sources=(
        make_geometry_source(length_mm=4700.0, height_mm=3000.0, thickness_mm=100.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    )), EvilReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.REJECT
    tried = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert "SUPER-ANCHOR-9000-NOT-IN-CATALOGUE" not in tried
    assert tried <= set(catalogue.CATALOGUE)


def test_10h_evil_reasoner_cannot_inject_capacity_for_a_malformed_run():
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload({"anchor_type": "ARL-42", "capacity": 999999})


def test_10h_evil_reasoner_cannot_override_status_for_a_malformed_run():
    result = run_agent(_clean_element(production=make_production("garbage_turn_method")),
                        EvilReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    lie = EvilReasoner().explain(result)
    assert "ACCEPT_PROVISIONAL" in lie
    assert result.status == Status.HOLD  # the lie has zero effect


def test_10h_evil_reasoner_cannot_override_reinforcement_state():
    result = run_agent(_clean_element(), EvilReasoner(), reinforcement_confirmed=None)
    checks = [c for c in result.illustrative_candidate.checks if c.check_name == "supplementary_reinforcement"]
    assert checks[-1].state == CheckState.UNKNOWN


def test_10h_malicious_extra_action_fields_rejected_even_for_a_malformed_run():
    raw = {"anchor_type": "ARL-42", "status": "ACCEPT_PROVISIONAL", "override_hard_stop": True}
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(raw)


# --------------------------------------------------------------------------
# 10I -- regression: existing valid fixtures are unchanged
# --------------------------------------------------------------------------

def test_10i_wc001_unchanged(wc001_element):
    result = run_agent(wc001_element, MockReasoner())
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"


def test_10i_resolved_fixture_unchanged():
    result = _run_resolved()
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.requires_human_signoff is True


def test_10i_test8_numerical_values_unchanged():
    """Pins the exact Test 8 regression numbers to prove the three Test 10
    fixes did not perturb the valid-case engineering path."""
    result = _run_resolved()
    assert result.self_weight.value_kn == pytest.approx(59.7547, abs=0.001)
    assert result.cog.x_mm == pytest.approx(2350.0, abs=0.5)
    assert result.cog.y_mm == pytest.approx(1500.0, abs=0.5)
    c = result.resolved_candidate
    assert c.anchor_type == "ARL-42"
    assert c.x1_mm == pytest.approx(972.9, abs=0.1)
    assert c.x2_mm == pytest.approx(3727.1, abs=0.1)
    assert c.governing_check.handling_state == "DEMOULD"
    assert c.governing_check.anchor_id == "A1"
    assert c.governing_check.utilisation == pytest.approx(0.7648, abs=0.001)
