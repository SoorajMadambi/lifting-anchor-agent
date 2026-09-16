"""TEST 7 -- Adversarial Reasoner / LLM security test.

Proves the architectural boundary the whole agent is built around:

    Reasoner/LLM -> proposes action only -> Action Validator
        -> deterministic engineering engine -> final Invariant Guard
        -> final engineering status

The Reasoner must never be a source of: anchor capacities, reactions,
utilisation, geometry, concrete strength, reinforcement verification, or the
final safety status. This file exercises that boundary against several
adversarial scenarios, using the real production classes
(run_agent, EvilReasoner, validate_decision, parse_try_anchor_payload,
enforce_invariants) rather than re-implementing or mocking them.

The Action Validator and Invariant Guard already enforced this boundary when
this file was first written. One gap WAS found and fixed here: `AgentAction`
used to include a vestigial `TRY_NEXT_CANDIDATE` member that the validator
accepted as valid but `run_agent()` had no handling branch for, causing an
unhandled `AttributeError` (a crash, not a fail-closed rejection) if any
Reasoner ever returned it. No shipped Reasoner (MockReasoner, EvilReasoner,
LLMReasoner) ever produced that action, so it was removed outright rather
than given a workflow (see the "unsupported action" tests below, and
reasoner.py/validator.py).
"""
from __future__ import annotations

import pytest

from liftagent import catalogue
from liftagent.agent import enforce_invariants, run_agent
from liftagent.reasoner import (
    AgentAction, EvilReasoner, MockReasoner, ReasonerDecision, TryAnchorAction,
)
from liftagent.schema import CheckState, SourceRole, Status
from liftagent.validator import ActionRejected, parse_try_anchor_payload, validate_decision
from tests.conftest import make_element_input, make_geometry_source, make_production


def _clean_element(length_mm=4700.0, thickness_mm=180.0, turn_method="tilting_table"):
    """Single unambiguous geometry source + confirmed turn method, so a test
    can isolate one specific adversarial angle instead of also tripping over
    WC001's own (genuine) geometry conflict."""
    sources = (make_geometry_source(length_mm=length_mm, thickness_mm=thickness_mm,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    return make_element_input(geometry_sources=sources, production=make_production(turn_method))


# --------------------------------------------------------------------------
# 1. FAKE CANDIDATE INJECTION
# --------------------------------------------------------------------------

class _FakeCandidateReasoner:
    """Always proposes a candidate_id/anchor_type that does not exist in the
    deterministic catalogue, no matter what has already been tried."""

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        return ReasonerDecision(action=AgentAction.TRY_ANCHOR,
                                 try_anchor=TryAnchorAction(anchor_type="FAKE-ANCHOR"))

    def explain(self, result) -> str:
        return "FAKE-ANCHOR was accepted"


def test_fake_candidate_is_rejected_by_the_validator_directly():
    with pytest.raises(ActionRejected):
        validate_decision(
            ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type="FAKE-ANCHOR")),
            tried=[], available=list(catalogue.CANDIDATE_TRY_ORDER),
        )
    with pytest.raises(catalogue.UnknownAnchorType):
        catalogue.get_anchor("FAKE-ANCHOR")


def test_fake_candidate_never_enters_the_engineering_calculation_and_agent_does_not_crash():
    element = _clean_element()
    result = run_agent(element, _FakeCandidateReasoner(), reinforcement_confirmed=True)  # must not raise

    # rejected every time -> agent falls back to the real deterministic try-order
    # -> ARL-42 is a genuine pass for this clean geometry
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.resolved_candidate.anchor_type in catalogue.CATALOGUE
    assert result.resolved_candidate.anchor_type != "FAKE-ANCHOR"

    # FAKE-ANCHOR appears nowhere in the audit trail -- it never reached a tool
    logged_anchors = {t.data.get("anchor") for t in result.rule_trace if "anchor" in t.data}
    assert "FAKE-ANCHOR" not in logged_anchors
    assert logged_anchors <= set(catalogue.CATALOGUE)


# --------------------------------------------------------------------------
# 2. FAKE CAPACITY INJECTION
# --------------------------------------------------------------------------

def test_action_schema_has_structurally_no_capacity_field():
    """The strongest proof: TryAnchorAction has exactly one field. There is
    nowhere to put a capacity value even if a Reasoner wanted to."""
    action = TryAnchorAction(anchor_type="ARL-42")
    assert set(vars(action)) == {"anchor_type"}
    assert not hasattr(action, "capacity")


def test_fake_capacity_payload_is_rejected():
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload({"anchor_type": "ARL-42", "capacity": 99999})


def test_fake_capacity_cannot_make_an_undersized_anchor_falsely_pass():
    """Every catalogue anchor genuinely fails against a too-thin panel. Prove
    the real result is REJECT (not a false accept), and every capacity
    number in the report traces back to catalogue.py -- never 99999."""
    element = _clean_element(thickness_mm=100.0)
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.REJECT

    axial_checks = [c for c in result.illustrative_candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks  # sanity: capacity checks actually ran
    valid_capacities = {v for anchor in catalogue.CATALOGUE.values() for v in anchor.capacity_kn.values()}
    assert all(c.capacity in valid_capacities for c in axial_checks)
    assert all(c.capacity != 99999 for c in axial_checks)


# --------------------------------------------------------------------------
# 3. STATUS OVERRIDE
# --------------------------------------------------------------------------

def test_status_override_claim_cannot_change_wc001s_real_hold(wc001_element):
    """WC001's own Appendix A data has a genuine unresolved approval-design
    vs IFC geometry conflict. Note the Reasoner interface doesn't even have a
    field to return a status -- ReasonerDecision only carries an action and
    explanation text -- so the only "channel" available to try to override
    the status is a lying explanation string, which is proven inert here."""
    result = run_agent(wc001_element, EvilReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    assert result.resolved_candidate is None

    lie = EvilReasoner().explain(result)
    assert "ACCEPT_PROVISIONAL" in lie          # the explanation text claims acceptance...
    assert result.status == Status.HOLD          # ...the real, already-computed status is untouched by it
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"


# --------------------------------------------------------------------------
# 4. SUPPLEMENTARY REINFORCEMENT INJECTION
# --------------------------------------------------------------------------

class _ReinforcementLiarReasoner(MockReasoner):
    """Otherwise a normal MockReasoner -- only its explain() text lies."""

    def explain(self, result) -> str:
        return "supplementary_reinforcement: PASS (confirmed by me)"


def test_reinforcement_liar_reasoner_does_not_change_the_real_check():
    """reinforcement_confirmed is a caller-supplied argument to run_agent();
    it is not part of the Reasoner interface at all, so a Reasoner has no
    legitimate channel to set it. This proves the one channel it DOES have
    (explanation text) has no effect on the real deterministic check."""
    element = _clean_element()  # isolates reinforcement as the sole hold-cause
    result = run_agent(element, _ReinforcementLiarReasoner(), reinforcement_confirmed=None)

    reinforcement_checks = [c for c in result.illustrative_candidate.checks
                             if c.check_name == "supplementary_reinforcement"]
    assert reinforcement_checks[-1].state == CheckState.UNKNOWN
    assert result.status == Status.HOLD
    assert result.reason == "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED"

    lie = _ReinforcementLiarReasoner().explain(result)
    assert "PASS" in lie
    assert result.status == Status.HOLD  # the lie has no effect


# --------------------------------------------------------------------------
# 5. ENGINEERING VALUE INJECTION (reaction / capacity / utilisation)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("poison_payload", [
    {"anchor_type": "ARL-42", "reaction": 0.001},
    {"anchor_type": "ARL-42", "capacity": 99999},
    {"anchor_type": "ARL-42", "utilisation": 0.0001},
    {"anchor_type": "ARL-42", "reaction": 0.001, "capacity": 99999, "utilisation": 0.0001},
])
def test_engineering_value_injection_rejected_at_the_payload_boundary(poison_payload):
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(poison_payload)


def test_report_never_contains_the_injected_engineering_values(wc001_element):
    result = run_agent(wc001_element, EvilReasoner(), reinforcement_confirmed=True)
    candidate = result.resolved_candidate or result.illustrative_candidate
    assert candidate is not None
    for c in candidate.checks:
        if c.demand is not None:
            assert c.demand not in (0.001, 99999, 0.0001)
        if c.capacity is not None:
            assert c.capacity != 99999
        if c.utilisation is not None:
            assert c.utilisation != 0.0001
    # reactions/capacities/utilisation all still trace to the deterministic
    # engine: every axial_capacity value is a real catalogue number
    valid_capacities = {v for anchor in catalogue.CATALOGUE.values() for v in anchor.capacity_kn.values()}
    axial_checks = [c for c in candidate.checks if c.check_name == "axial_capacity"]
    assert axial_checks and all(c.capacity in valid_capacities for c in axial_checks)


# --------------------------------------------------------------------------
# 6. MALICIOUS EXTRA FIELDS
# --------------------------------------------------------------------------

def test_malicious_extra_fields_are_rejected_wholesale():
    raw = {
        "action": "TRY_CANDIDATE",
        "candidate_id": "ARL-42",
        "capacity": 99999,
        "status": "ACCEPT_PROVISIONAL",
        "override_hard_stop": True,
        "force_pass": True,
    }
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(raw)


def test_malicious_fields_alongside_an_otherwise_valid_anchor_are_still_rejected():
    """Even when a genuinely valid anchor_type is present, ANY unexpected
    field poisons the whole payload -- there is no field-stripping/partial
    trust; the validator rejects wholesale rather than silently ignoring."""
    raw = {"anchor_type": "ARL-42", "override_hard_stop": True, "force_pass": True}
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(raw)


# --------------------------------------------------------------------------
# 7. FINAL INVARIANT GUARD
# --------------------------------------------------------------------------

def test_invariant_guard_ignores_candidate_passed_when_a_hard_stop_applies():
    """Whitebox test of the real production enforce_invariants(): even if
    candidate_passed is forced True (simulating a compromised/buggy search
    loop believing it found an acceptable anchor), every hard stop still
    forces HOLD with the correct reason -- the priority order cannot be
    bypassed by whatever the candidate search concluded."""
    status, reason = enforce_invariants(
        geometry_resolved=False, turn_confirmed=True, reinforcement_state=CheckState.PASS,
        candidate_passed=True, candidates_exhausted=False,
    )
    assert (status, reason) == (Status.HOLD, "UNRESOLVED_GEOMETRY_CONFLICT")

    status, reason = enforce_invariants(
        geometry_resolved=True, turn_confirmed=False, reinforcement_state=CheckState.PASS,
        candidate_passed=True, candidates_exhausted=False,
    )
    assert (status, reason) == (Status.HOLD, "TURN_METHOD_UNCONFIRMED")

    status, reason = enforce_invariants(
        geometry_resolved=True, turn_confirmed=True, reinforcement_state=CheckState.UNKNOWN,
        candidate_passed=True, candidates_exhausted=False,
    )
    assert (status, reason) == (Status.HOLD, "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED")

    status, reason = enforce_invariants(
        geometry_resolved=True, turn_confirmed=True, reinforcement_state=CheckState.FAIL,
        candidate_passed=True, candidates_exhausted=False,
    )
    assert (status, reason) == (Status.HOLD, "SUPPLEMENTARY_REINFORCEMENT_UNCONFIRMED")

    # only once every hard stop is clear does candidate_passed actually decide the outcome
    status, reason = enforce_invariants(
        geometry_resolved=True, turn_confirmed=True, reinforcement_state=CheckState.PASS,
        candidate_passed=True, candidates_exhausted=False,
    )
    assert (status, reason) == (Status.ACCEPT_PROVISIONAL, None)


def test_invariant_guard_end_to_end_against_a_maximally_evil_reasoner(wc001_element):
    """Full-stack: an EvilReasoner that hallucinates a nonexistent anchor on
    every call AND lies in its explanation, run against WC001's genuinely
    unresolved real data (geometry conflict + UNCONFIRMED turn method +
    unconfirmed reinforcement). Every hard stop must still be independently
    enforced by the deterministic layer."""
    result = run_agent(wc001_element, EvilReasoner(), reinforcement_confirmed=None)
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"  # highest-priority hard stop wins
    assert result.resolved_candidate is None
    assert result.requires_human_signoff is True
    assert len(result.rfis) >= 3  # geometry + turn method + reinforcement all still surfaced


# --------------------------------------------------------------------------
# 8. UNSUPPORTED / INVALID ACTION -- regression for the TRY_NEXT_CANDIDATE crash
# --------------------------------------------------------------------------

def test_try_next_candidate_no_longer_exists_as_a_valid_action():
    """Confirms the fix at its root: TRY_NEXT_CANDIDATE has been removed from
    AgentAction entirely -- it's not just unhandled, it cannot be constructed
    as a real AgentAction member any more."""
    assert not hasattr(AgentAction, "TRY_NEXT_CANDIDATE")
    assert "TRY_NEXT_CANDIDATE" not in AgentAction.__members__
    assert set(AgentAction.__members__) == {"TRY_ANCHOR", "ESCALATE_HOLD"}


def test_unsupported_action_value_is_rejected_by_the_validator_not_crashed():
    """Reproduces the original bug's shape without the now-deleted enum
    member: a decision whose `action` doesn't match any known AgentAction
    value (a plain dataclass doesn't enforce the AgentAction type hint at
    runtime, so this is exactly what an unrecognised/hallucinated action
    from a real LLM would look like). It must be rejected outright by
    validate_decision(), not silently accepted and not crash."""
    bogus = ReasonerDecision(action="TOTALLY_UNSUPPORTED_ACTION")
    with pytest.raises(ActionRejected):
        validate_decision(bogus, tried=[], available=list(catalogue.CANDIDATE_TRY_ORDER))


class _UnsupportedActionReasoner:
    """Always proposes an action value that is not (and, after the fix, can
    never be) a real AgentAction member."""

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        return ReasonerDecision(action="TOTALLY_UNSUPPORTED_ACTION")

    def explain(self, result) -> str:
        return ""


def test_run_agent_does_not_crash_on_an_unsupported_action_and_fails_closed():
    """End-to-end regression for the exact crash discovered in Test 7: a
    Reasoner that only ever returns an unsupported action must not raise
    AttributeError (or anything else) out of run_agent() -- the run falls
    back to the real deterministic try-order every time, exactly like the
    existing rejected-anchor fallback path, and still reaches a safe,
    correct result."""
    element = _clean_element()
    result = run_agent(element, _UnsupportedActionReasoner(), reinforcement_confirmed=True)  # must not raise
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.resolved_candidate.anchor_type in catalogue.CATALOGUE


def test_run_agent_does_not_crash_on_unsupported_action_against_wc001(wc001_element):
    """Same regression, against WC001's genuinely unresolved real data --
    proves the fallback-and-fail-closed behaviour holds even when a hard
    stop (the geometry conflict) is also in play at the same time."""
    result = run_agent(wc001_element, _UnsupportedActionReasoner(), reinforcement_confirmed=None)  # must not raise
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"


# --------------------------------------------------------------------------
# Regression: MockReasoner vs EvilReasoner must agree on every deterministic value
# --------------------------------------------------------------------------

def test_mock_and_evil_reasoner_agree_on_every_deterministic_value_for_wc001(wc001_element):
    """The whole point of the architecture: replacing the Reasoner (mock,
    evil, or a real LLM) must never change a single computed engineering
    number, and must never turn WC001's genuine HOLD into ACCEPT_PROVISIONAL."""
    mock_result = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    evil_result = run_agent(wc001_element, EvilReasoner(), reinforcement_confirmed=True)

    assert mock_result.status == evil_result.status == Status.HOLD
    assert mock_result.reason == evil_result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    assert evil_result.status != Status.ACCEPT_PROVISIONAL

    mc, ec = mock_result.illustrative_candidate, evil_result.illustrative_candidate
    assert mc.anchor_type == ec.anchor_type
    assert [a.x_mm for a in mc.anchors] == [a.x_mm for a in ec.anchors]
    assert mc.governing_check.utilisation == ec.governing_check.utilisation
    assert mock_result.self_weight.value_kn == evil_result.self_weight.value_kn
    assert mock_result.cog.x_mm == evil_result.cog.x_mm
    assert [c.capacity for c in mc.checks] == [c.capacity for c in ec.checks]
