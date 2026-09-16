"""Action Validator (plan section 25): the checkpoint every Reasoner-selected
action must pass before a deterministic tool runs. Two lines of defense:

1. Schema: TryAnchorAction has exactly one field (reasoner.py) -- there is
   structurally nowhere to put a fabricated engineering number.
2. This module: rejects any *raw* payload (e.g. untrusted JSON from a real
   LLM tool call) that carries unexpected fields, references an anchor not
   in catalogue.py, or is invalid for the current deterministic state.
"""
from __future__ import annotations

from liftagent import catalogue
from liftagent.reasoner import AgentAction, ReasonerDecision, TryAnchorAction

ALLOWED_TRY_ANCHOR_FIELDS = {"anchor_type"}


class ActionRejected(Exception):
    pass


def parse_try_anchor_payload(raw: dict) -> TryAnchorAction:
    """Interpret an untrusted raw dict (e.g. a real LLM's JSON tool call) as
    a TryAnchorAction. This is the realistic adversarial-input boundary --
    see tests/test_agent.py for the injection tests exercising this."""
    extra = set(raw) - ALLOWED_TRY_ANCHOR_FIELDS
    if extra:
        raise ActionRejected(f"TRY_ANCHOR payload carries unexpected field(s): {sorted(extra)}")
    anchor_type = raw.get("anchor_type")
    if not isinstance(anchor_type, str) or not anchor_type:
        raise ActionRejected("TRY_ANCHOR payload missing a valid 'anchor_type' string")
    return TryAnchorAction(anchor_type=anchor_type)


def validate_decision(decision: ReasonerDecision, tried: list[str], available: list[str]) -> ReasonerDecision:
    """Validate a (already-typed) ReasonerDecision against catalogue/state
    before the Agent acts on it. Raises ActionRejected on anything invalid;
    callers must not fall back to trusting the rejected decision."""
    if decision.action == AgentAction.TRY_ANCHOR:
        if decision.try_anchor is None:
            raise ActionRejected("TRY_ANCHOR action missing its payload")
        extra_fields = set(vars(decision.try_anchor)) - ALLOWED_TRY_ANCHOR_FIELDS
        if extra_fields:
            raise ActionRejected(f"TRY_ANCHOR payload carries unexpected field(s): {sorted(extra_fields)}")
        anchor_type = decision.try_anchor.anchor_type
        if anchor_type not in available:
            raise ActionRejected(f"{anchor_type!r} is not a permitted catalogue candidate for this run")
        if anchor_type in tried:
            raise ActionRejected(f"{anchor_type!r} has already been tried")
        try:
            catalogue.get_anchor(anchor_type)
        except catalogue.UnknownAnchorType as exc:
            raise ActionRejected(f"{anchor_type!r} does not exist in catalogue.py") from exc
        return decision
    if decision.action == AgentAction.ESCALATE_HOLD:
        return decision
    # Any other action -- including an unsupported/invalid value a Reasoner
    # might return (e.g. a removed action, a typo, or a hallucinated one) --
    # is rejected here rather than silently falling through to code that
    # assumes a specific payload shape. This is what makes an unrecognised
    # action fail closed instead of crashing run_agent() with an
    # AttributeError (see tests/test_security_adversarial.py).
    raise ActionRejected(f"Unrecognised action {decision.action!r}")
