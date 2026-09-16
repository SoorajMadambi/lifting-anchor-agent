from __future__ import annotations

import pytest

from liftagent import catalogue, report
from liftagent.agent import run_agent
from liftagent.reasoner import (
    AgentAction, EvilReasoner, MockReasoner, ReasonerDecision, TryAnchorAction,
)
from liftagent.schema import SourceRole, Status
from liftagent.validator import ActionRejected, parse_try_anchor_payload, validate_decision
from tests.conftest import make_element_input, make_geometry_source, make_production


def _clean_element(length_mm=4700.0, thickness_mm=180.0, turn_method="tilting_table"):
    sources = (make_geometry_source(length_mm=length_mm, thickness_mm=thickness_mm,
                                     role=SourceRole.AUTHORITATIVE_DESIGN, name="approval_design"),)
    return make_element_input(geometry_sources=sources, production=make_production(turn_method))


# --------------------------------------------------------------------------
# End-to-end / repeatability
# --------------------------------------------------------------------------

def test_end_to_end_run_produces_a_result(wc001_element):
    result = run_agent(wc001_element, MockReasoner())
    assert result.element_id == "WC001"
    assert result.status in Status
    # report + visualization must not crash on the real result
    assert report.to_json(result)
    from liftagent.visualize import render_svg
    assert "<svg" in render_svg(result)


def test_governing_check_is_consistent_with_the_matching_entry_in_checks(wc001_element):
    """Regression test: governing_check and the corresponding entry inside
    checks must be the SAME object/value, never two independently-derived
    copies that can disagree on the `governing` flag."""
    result = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    candidate = result.resolved_candidate or result.illustrative_candidate
    assert candidate is not None
    assert candidate.governing_check is not None

    matches = [
        c for c in candidate.checks
        if c.check_name == candidate.governing_check.check_name
        and c.handling_state == candidate.governing_check.handling_state
        and c.anchor_id == candidate.governing_check.anchor_id
    ]
    assert len(matches) == 1
    assert matches[0] == candidate.governing_check
    assert matches[0].governing is True

    # exactly one check in the whole candidate is flagged governing
    governing_flagged = [c for c in candidate.checks if c.governing]
    assert len(governing_flagged) == 1


def test_deterministic_repeatability(wc001_element):
    r1 = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    r2 = run_agent(wc001_element, MockReasoner(), reinforcement_confirmed=True)
    assert r1.status == r2.status == Status.HOLD  # wc001.json HOLDs on its own geometry conflict
    assert r1.resolved_candidate is None and r2.resolved_candidate is None
    assert [a.x_mm for a in r1.illustrative_candidate.anchors] == [a.x_mm for a in r2.illustrative_candidate.anchors]
    assert r1.illustrative_candidate.governing_check.utilisation == r2.illustrative_candidate.governing_check.utilisation


# --------------------------------------------------------------------------
# ITERATE advancement / REJECT exhaustion
# --------------------------------------------------------------------------

def test_iterate_advances_to_a_second_candidate_when_first_fails():
    # Sized so ARL-42 (tried first) fails capacity, but ARL-52 (tried second)
    # passes -- see tests/test_agent.py module docstring-equivalent reasoning
    # in the PR description / DESIGN_NOTE.md for the hand-worked numbers.
    element = _clean_element(length_mm=6000.0, thickness_mm=220.0)
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.ACCEPT_PROVISIONAL
    assert result.reason is None
    assert result.resolved_candidate.anchor_type == "ARL-52"
    assert result.illustrative_candidate is None  # mutually exclusive with resolved_candidate
    tried_anchor_types = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchor_types == {"ARL-42", "ARL-52"}  # proves both were actually attempted


def test_reject_fires_only_once_every_candidate_is_exhausted():
    element = _clean_element(thickness_mm=100.0)  # too thin for every catalogue anchor's axial min-wall
    result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    assert result.status == Status.REJECT
    assert result.reason == "NO_CANDIDATE_SATISFIES_REQUIREMENTS"
    assert result.resolved_candidate is None
    tried_anchor_types = {t.data.get("anchor") for t in result.rule_trace if t.step == "rig_decision"}
    assert tried_anchor_types == set(catalogue.CANDIDATE_TRY_ORDER)  # every candidate was tried
    assert result.illustrative_candidate.anchor_type == catalogue.CANDIDATE_TRY_ORDER[-1]  # last one attempted


# --------------------------------------------------------------------------
# Adversarial: EvilReasoner cannot override the invariant guard
# --------------------------------------------------------------------------

def test_evil_reasoner_hallucinated_anchor_is_rejected_and_run_still_succeeds():
    element = _clean_element()
    mock_result = run_agent(element, MockReasoner(), reinforcement_confirmed=True)
    evil_result = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
    # the hallucinated action is rejected every time -> falls back to the same
    # deterministic try-order MockReasoner uses -> identical engineering result
    assert evil_result.status == mock_result.status == Status.ACCEPT_PROVISIONAL
    assert [a.x_mm for a in evil_result.resolved_candidate.anchors] == [a.x_mm for a in mock_result.resolved_candidate.anchors]


def test_evil_reasoner_explanation_lie_does_not_affect_status():
    element = _clean_element(thickness_mm=100.0)  # a genuinely REJECT-worthy case
    result = run_agent(element, EvilReasoner(), reinforcement_confirmed=True)
    lie = EvilReasoner().explain(result)
    assert "ACCEPT_PROVISIONAL" in lie  # the explanation text lies
    assert result.status == Status.REJECT  # but the real status is unaffected by that text


def test_evil_reasoner_action_is_actually_rejected_by_the_validator():
    with pytest.raises(ActionRejected):
        validate_decision(
            ReasonerDecision(action=AgentAction.TRY_ANCHOR,
                              try_anchor=TryAnchorAction(anchor_type="SUPER-ANCHOR-9000-NOT-IN-CATALOGUE")),
            tried=[], available=list(catalogue.CANDIDATE_TRY_ORDER),
        )


# --------------------------------------------------------------------------
# Adversarial: tool-result / engineering-number injection via a raw payload
# (simulating untrusted JSON from a real LLM tool call)
# --------------------------------------------------------------------------

def test_injection_test_a_malformed_payload_with_fabricated_capacity_is_rejected():
    raw = {"anchor_type": "ARL-42", "capacity": 9999}
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(raw)


@pytest.mark.parametrize("poison_field", ["force", "cog", "utilisation", "dynamic_factor"])
def test_injection_fabricated_engineering_fields_are_rejected(poison_field):
    raw = {"anchor_type": "ARL-42", poison_field: 123456}
    with pytest.raises(ActionRejected):
        parse_try_anchor_payload(raw)


def test_injection_test_b_valid_payload_is_accepted_and_capacity_comes_from_catalogue():
    raw = {"anchor_type": "ARL-42"}
    action = parse_try_anchor_payload(raw)  # does not raise
    assert action.anchor_type == "ARL-42"
    # the ONLY valid capacity source is catalogue.py, never the payload
    assert catalogue.lookup_capacity(action.anchor_type, 35) == 80.0
