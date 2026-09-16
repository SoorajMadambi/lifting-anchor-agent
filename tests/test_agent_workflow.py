"""TEST 11 -- End-to-end agent workflow / Reasoner orchestration.

Proves the system is actually an agent architecture, not a deterministic
script with a Reasoner bolted on for show:

    INPUT -> deterministic observations/tools -> Reasoner -> typed action
    -> Action Validator -> deterministic engineering evaluation
    -> candidate iteration if required -> Invariant Guard -> final report

The Reasoner decides WHICH permitted candidate to try next; it never
performs or overrides engineering calculations. This file is about
orchestration/wiring, not engineering correctness (Test 8) or adversarial
security (Test 7) or the output contract (Test 9) -- it reuses all three
rather than re-deriving them.

Note on terminology: the brief for this test describes the action as
`TRY_CANDIDATE(candidate_id)`. The real, already-shipped action is
`TRY_ANCHOR(anchor_type)` (reasoner.py) -- there is no separate
`TRY_CANDIDATE` action in this architecture (Test 7 removed a vestigial,
unused `TRY_NEXT_CANDIDATE` member for exactly this kind of reason: an
action nothing produces is dead weight). Every test below uses the real
`TRY_ANCHOR`/`anchor_type` action, per "use the existing architecture, do
not redesign Reasoner interfaces."

No production code was changed for this file -- every workflow property it
checks was already true of the existing implementation.
"""
from __future__ import annotations

import inspect

import pytest

from liftagent import catalogue
from liftagent.agent import run_agent
from liftagent.reasoner import (
    AgentAction, EscalateHoldAction, EvilReasoner, MockReasoner, ReasonerDecision, TryAnchorAction,
)
from liftagent.schema import SourceRole, Status
from liftagent.validator import ActionRejected, parse_try_anchor_payload, validate_decision
from tests.conftest import make_geometry_source, make_production
from tests.test_end_to_end import _resolved_element, _run_resolved


# --------------------------------------------------------------------------
# 11A -- Reasoner receives deterministic context
# --------------------------------------------------------------------------

class _RecordingReasoner:
    """Wraps MockReasoner's real decision logic but records every call's
    arguments, so the test can inspect exactly what context the Agent gives
    the Reasoner to decide with."""

    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
        self._mock = MockReasoner()

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        self.calls.append((tuple(tried), tuple(available), last_failure_reason))
        return self._mock.decide_next_candidate(tried, available, last_failure_reason)

    def explain(self, result) -> str:
        return self._mock.explain(result)


def test_11a_reasoner_receives_deterministic_candidate_context():
    reasoner = _RecordingReasoner()
    result = run_agent(_resolved_element(), reasoner, reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert reasoner.calls  # the Reasoner was actually invoked, not bypassed

    tried0, available0, reason0 = reasoner.calls[0]
    # "available candidate IDs" -- real catalogue anchor types, not invented ones
    assert set(available0) == set(catalogue.CANDIDATE_TRY_ORDER)
    # each ID is deterministic data resolving to a full record (type + clutch +
    # capacities + geometric constraints) via catalogue.py -- the Reasoner is
    # handed the KEY, not asked to invent the engineering facts behind it
    for anchor_type in available0:
        anchor = catalogue.get_anchor(anchor_type)
        assert anchor.clutch  # candidate anchor type + clutch are real, deterministic
    assert tried0 == ()          # candidate evaluation state: nothing tried yet on the first call
    assert reason0 == ""         # no failure information yet on the first call


# --------------------------------------------------------------------------
# 11B -- valid Reasoner action is evaluated by the deterministic engine
# --------------------------------------------------------------------------

class _FixedChoiceReasoner:
    """Always selects one specific anchor. Its explanation text is a
    deliberate lie about the engineering result, to prove that text is
    never read as data."""

    def __init__(self, anchor_type: str):
        self.anchor_type = anchor_type

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        return ReasonerDecision(action=AgentAction.TRY_ANCHOR,
                                 try_anchor=TryAnchorAction(anchor_type=self.anchor_type),
                                 explanation="I have calculated a capacity of 99999 kN")

    def explain(self, result) -> str:
        return "final capacity was 99999 kN, utilisation 0.0001"  # a lie


def test_11b_valid_reasoner_action_is_evaluated_by_the_deterministic_engine():
    result = run_agent(_resolved_element(), _FixedChoiceReasoner("ARL-42"), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.resolved_candidate.anchor_type == "ARL-42"

    gc = result.resolved_candidate.governing_check
    # the real number comes from catalogue.py + the deterministic engine, not the Reasoner's text
    assert gc.capacity == catalogue.lookup_capacity("ARL-42", 15)
    assert gc.utilisation == pytest.approx(0.7648, abs=0.001)
    assert gc.capacity != 99999
    assert gc.utilisation != pytest.approx(0.0001, abs=1e-5)


# --------------------------------------------------------------------------
# 11C -- invalid candidate action: rejected, never evaluated, never the
# reason a result becomes ACCEPT_PROVISIONAL
# --------------------------------------------------------------------------

class _AlwaysInvalidCandidateReasoner:
    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        return ReasonerDecision(action=AgentAction.TRY_ANCHOR,
                                 try_anchor=TryAnchorAction(anchor_type="does_not_exist"))

    def explain(self, result) -> str:
        return ""


def test_11c_invalid_candidate_rejected_and_cannot_cause_wc001_to_accept(wc001_element):
    with pytest.raises(ActionRejected):
        validate_decision(
            ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type="does_not_exist")),
            tried=[], available=list(catalogue.CANDIDATE_TRY_ORDER),
        )

    result = run_agent(wc001_element, _AlwaysInvalidCandidateReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD  # cannot become ACCEPT_PROVISIONAL because of the invalid action
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"
    tried_anchors = {t.data.get("anchor") for t in result.rule_trace if "anchor" in t.data}
    assert "does_not_exist" not in tried_anchors  # the fake candidate was never evaluated


# --------------------------------------------------------------------------
# 11D -- Reasoner cannot inject engineering values via the action payload
# --------------------------------------------------------------------------

def test_11d_reasoner_cannot_inject_engineering_values_via_the_action_payload():
    poisoned = {
        "anchor_type": "ARL-42", "capacity": 999999, "reaction": 0.001, "cog": 2350.0,
        "utilisation": 0.0001, "self_weight": 1.0, "status": "ACCEPT_PROVISIONAL",
        "governing_check": "PASS",
    }
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(poisoned)

    # the ONLY thing a Reasoner can legitimately supply is anchor_type; every
    # engineering value in the final report still traces to catalogue.py
    result = _run_resolved()
    gc = result.resolved_candidate.governing_check
    assert gc.capacity == catalogue.lookup_capacity(result.resolved_candidate.anchor_type, 15)
    assert gc.capacity != 999999


# --------------------------------------------------------------------------
# 11E -- Reasoner cannot override status
# --------------------------------------------------------------------------

def test_11e_reasoner_cannot_override_status_for_unconfirmed_turn_method():
    element = _resolved_element(production=make_production("UNCONFIRMED"))
    result = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.HOLD
    assert result.reason == "TURN_METHOD_UNCONFIRMED"

    lie = EvilReasoner().explain(result)
    assert "ACCEPT_PROVISIONAL" in lie  # the explanation claims acceptance...
    assert result.status == Status.HOLD  # ...the real, already-computed status is unaffected


# --------------------------------------------------------------------------
# 11F -- candidate iteration: candidate 1 evaluated before candidate 2 is
# even selected, using Test 8's own geometry (no new engineering rules)
# --------------------------------------------------------------------------

class _TwoStepReasoner:
    """Deliberately picks ARL-42 first, then ARL-52 -- proving the ORDER of
    evaluation, not just the eventual outcome."""

    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        self.calls.append((tuple(tried), last_failure_reason != ""))
        if "ARL-42" not in tried:
            return ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type="ARL-42"))
        return ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type="ARL-52"))

    def explain(self, result) -> str:
        return ""


def test_11f_candidate_iteration_evaluates_candidate_1_before_candidate_2_is_selected():
    # sized exactly as tests/test_agent.py's test_iterate_advances_to_a_second_candidate_when_first_fails:
    # ARL-42 (tried first) genuinely fails capacity; ARL-52 (tried second) genuinely passes.
    element = _resolved_element(geometry_sources=(
        make_geometry_source(length_mm=6000.0, height_mm=3000.0, thickness_mm=220.0,
                              role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),
    ))
    reasoner = _TwoStepReasoner()
    result = run_agent(element, reasoner, reinforcement_confirmed=True)

    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.resolved_candidate.anchor_type == "ARL-52"  # candidate 2's real deterministic result

    assert len(reasoner.calls) == 2
    assert reasoner.calls[0] == ((), False)       # first call: nothing tried, no failure yet
    assert reasoner.calls[1][0] == ("ARL-42",)     # second call proves candidate 1 was evaluated FIRST
    assert reasoner.calls[1][1] is True             # ...with a real deterministic failure reason produced for it

    tried_anchors = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchors == {"ARL-42", "ARL-52"}   # both evaluations used the deterministic engine; nothing fabricated


# --------------------------------------------------------------------------
# 11G -- Reasoner independence / determinism (Reasoner != engineering calculator)
# --------------------------------------------------------------------------

class _AlternateDeterministicReasoner:
    """A completely different implementation from MockReasoner, with
    different explanation wording, but the same try-order logic -- an
    'equivalent valid candidate selection' per the brief."""

    def decide_next_candidate(self, tried, available, last_failure_reason=""):
        remaining = [a for a in available if a not in tried]
        if remaining:
            return ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type=remaining[0]),
                                     explanation="alternate reasoner implementation, unrelated wording")
        return ReasonerDecision(action=AgentAction.ESCALATE_HOLD, escalate=EscalateHoldAction(reason="exhausted"))

    def explain(self, result) -> str:
        return "a completely different explanation string than MockReasoner produces"


def test_11g_two_independent_reasoners_produce_identical_engineering_quantities():
    element = _resolved_element()
    a = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    b = run_agent(element, _AlternateDeterministicReasoner(), reinforcement_confirmed=True)

    assert a.status == b.status == Status.ACCEPT_PROVISIONAL
    assert a.self_weight.value_kn == b.self_weight.value_kn
    assert (a.cog.x_mm, a.cog.y_mm) == (b.cog.x_mm, b.cog.y_mm)
    assert (a.resolved_candidate.x1_mm, a.resolved_candidate.x2_mm) == \
           (b.resolved_candidate.x1_mm, b.resolved_candidate.x2_mm)

    caps_a = sorted((c.handling_state, c.anchor_id, c.capacity, c.utilisation)
                     for c in a.resolved_candidate.checks if c.check_name == "axial_capacity")
    caps_b = sorted((c.handling_state, c.anchor_id, c.capacity, c.utilisation)
                     for c in b.resolved_candidate.checks if c.check_name == "axial_capacity")
    assert caps_a == caps_b  # every reaction/capacity/utilisation is identical

    assert a.resolved_candidate.governing_check.utilisation == b.resolved_candidate.governing_check.utilisation
    # AgentResult itself carries no Reasoner-authored text at all (explain()
    # is a separate, optional call -- see reasoner.py) -- only the two
    # Reasoners' own decorative explanation strings were ever allowed to
    # differ, and neither result object stores them.


# --------------------------------------------------------------------------
# 11H -- EvilReasoner regression: cannot force WC001 acceptance end-to-end
# --------------------------------------------------------------------------

def test_11h_evil_reasoner_cannot_force_wc001_acceptance(wc001_element):
    result = run_agent(wc001_element, EvilReasoner(), reinforcement_confirmed=None)
    assert result.status == Status.HOLD
    assert result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"  # highest-priority hard stop still wins
    assert result.resolved_candidate is None
    assert result.requires_human_signoff is True


# --------------------------------------------------------------------------
# 11I -- human sign-off: ACCEPT_PROVISIONAL is never autonomous final approval
# --------------------------------------------------------------------------

def test_11i_human_signoff_required_even_on_accept_provisional():
    result = _run_resolved()
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.requires_human_signoff is True
    # the Reasoner has no channel to touch this field at all -- it isn't part
    # of ReasonerDecision, and every EvilReasoner test elsewhere confirms it


# --------------------------------------------------------------------------
# 11J -- MockReasoner alone is sufficient; no LLM/API dependency
# --------------------------------------------------------------------------

def test_11j_mock_reasoner_workflow_is_self_sufficient_without_any_llm_dependency(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # structural proof: MockReasoner's own source never references anthropic/an API client
    assert "anthropic" not in inspect.getsource(MockReasoner).lower()

    result = run_agent(_resolved_element(), MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.requires_human_signoff is True


# --------------------------------------------------------------------------
# 11K -- regression
# --------------------------------------------------------------------------

def test_11k_wc001_and_resolved_fixture_outcomes_and_test8_numbers_unchanged(wc001_element):
    wc001_result = run_agent(wc001_element, MockReasoner())
    assert wc001_result.status == Status.HOLD
    assert wc001_result.reason == "UNRESOLVED_GEOMETRY_CONFLICT"

    resolved = _run_resolved()
    assert resolved.status == Status.ACCEPT_PROVISIONAL
    assert resolved.requires_human_signoff is True

    assert resolved.self_weight.value_kn == pytest.approx(59.7547, abs=0.001)
    assert resolved.cog.x_mm == pytest.approx(2350.0, abs=0.5)
    assert resolved.cog.y_mm == pytest.approx(1500.0, abs=0.5)
    c = resolved.resolved_candidate
    assert c.x1_mm == pytest.approx(972.9, abs=0.1)
    assert c.x2_mm == pytest.approx(3727.1, abs=0.1)
    assert c.governing_check.handling_state == "DEMOULD"
    assert c.governing_check.anchor_id == "A1"
    assert c.governing_check.utilisation == pytest.approx(0.7648, abs=0.001)
