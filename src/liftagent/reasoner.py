"""Bounded Reasoner: the only component allowed to be an LLM.

It may return exactly one typed AgentAction, plus optional explanation text
derived from already-computed results. It NEVER returns an engineering
number or an authoritative status (Determinism Boundary, plan section 3).

TryAnchorAction has exactly one field, anchor_type -- there is no capacity,
force, cog, utilisation, or dynamic_factor field anywhere in this module for
a Reasoner to populate, by construction. See validator.py for the second
line of defense against a malformed/untrusted raw payload (e.g. JSON from a
real LLM tool call).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AgentAction(str, Enum):
    TRY_ANCHOR = "TRY_ANCHOR"
    ESCALATE_HOLD = "ESCALATE_HOLD"


@dataclass(frozen=True)
class TryAnchorAction:
    anchor_type: str


@dataclass(frozen=True)
class EscalateHoldAction:
    reason: str


@dataclass(frozen=True)
class ReasonerDecision:
    action: AgentAction
    try_anchor: TryAnchorAction | None = None
    escalate: EscalateHoldAction | None = None
    explanation: str = ""


class Reasoner(Protocol):
    def decide_next_candidate(self, tried: list[str], available: list[str],
                               last_failure_reason: str = "") -> ReasonerDecision: ...

    def explain(self, result) -> str: ...


class MockReasoner:
    """Deterministic, offline, reproducible. Used by default (plan section 23)."""

    def decide_next_candidate(self, tried: list[str], available: list[str],
                               last_failure_reason: str = "") -> ReasonerDecision:
        remaining = [a for a in available if a not in tried]
        if remaining:
            return ReasonerDecision(
                action=AgentAction.TRY_ANCHOR,
                try_anchor=TryAnchorAction(anchor_type=remaining[0]),
                explanation=f"Trying next catalogue candidate {remaining[0]!r} in capacity order.",
            )
        return ReasonerDecision(
            action=AgentAction.ESCALATE_HOLD,
            escalate=EscalateHoldAction(reason="All catalogue candidates exhausted."),
            explanation="No remaining permitted catalogue candidates to try.",
        )

    def explain(self, result) -> str:
        resolved = result.status.value == "ACCEPT_PROVISIONAL"
        parts = [f"Status: {result.status.value}."]
        if result.reason is not None:
            parts.append(f"Reason: {result.reason}.")
        candidate = result.resolved_candidate if resolved else result.illustrative_candidate
        gc = candidate.governing_check if candidate is not None else None
        if gc is not None and gc.utilisation is not None:
            label = "Governing check" if resolved else "Illustrative check (not final)"
            parts.append(f"{label}: {gc.handling_state} anchor {gc.anchor_id}, utilisation {gc.utilisation:.2f}.")
        for rfi in result.rfis:
            parts.append(f"RFI: {rfi.message}")
        return " ".join(p for p in parts if p)


class EvilReasoner:
    """Adversarial test double simulating a hallucinating/malicious LLM: it
    always names a bogus anchor type that does not exist in catalogue.py (or
    is not in the permitted candidate list), and its explanation text always
    lies and claims success. Used only in tests/test_agent.py to prove:

    1. The Action Validator rejects the hallucinated action outright (it is
       never trusted or acted on) -- the Agent falls back to the
       deterministic try-order instead, so the final engineering result is
       identical to what MockReasoner would have produced.
    2. explain()'s lie has zero effect on result.status -- that string is
       purely decorative, never parsed as data (Determinism Boundary,
       plan section 3).
    """

    def decide_next_candidate(self, tried: list[str], available: list[str],
                               last_failure_reason: str = "") -> ReasonerDecision:
        return ReasonerDecision(
            action=AgentAction.TRY_ANCHOR,
            try_anchor=TryAnchorAction(anchor_type="SUPER-ANCHOR-9000-NOT-IN-CATALOGUE"),
            explanation="(evil) hallucinating a miracle anchor",
        )

    def explain(self, result) -> str:
        return "ACCEPT_PROVISIONAL -- everything passed (this is a lie; see EvilReasoner)"


class LLMReasoner:
    """Optional. Lazily imports anthropic and only activates when
    ANTHROPIC_API_KEY is set. Even with a real model behind it, this class's
    return type is still exactly ReasonerDecision -- the same typed,
    number-free action space as MockReasoner (plan section 23)."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; use MockReasoner instead")
        import anthropic  # lazy import -- optional dependency
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def decide_next_candidate(self, tried: list[str], available: list[str],
                               last_failure_reason: str = "") -> ReasonerDecision:
        remaining = [a for a in available if a not in tried]
        if not remaining:
            return ReasonerDecision(
                action=AgentAction.ESCALATE_HOLD,
                escalate=EscalateHoldAction(reason="All catalogue candidates exhausted."),
            )
        prompt = (
            "You are choosing the next lifting-anchor catalogue candidate to try for a precast "
            "wall panel. Already-tried candidates (all failed): "
            f"{tried}. Remaining permitted candidates: {remaining}. "
            f"Reason the last candidate failed: {last_failure_reason!r}. "
            "Reply with EXACTLY one candidate name copied verbatim from the remaining list, "
            "nothing else -- no numbers, no units, no explanation."
        )
        resp = self._client.messages.create(
            model=self._model, max_tokens=20, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        chosen = text if text in remaining else remaining[0]  # never trust free text outside the allowed set
        return ReasonerDecision(action=AgentAction.TRY_ANCHOR, try_anchor=TryAnchorAction(anchor_type=chosen),
                                 explanation=f"LLM selected {chosen!r}.")

    def explain(self, result) -> str:
        prompt = (
            "Explain this deterministic lifting-anchor engineering result to a checking engineer "
            "in 3-5 plain sentences. Use ONLY the facts given below -- do not invent, recompute, "
            "or restate any number differently than given.\n\n"
            f"Status: {result.status.value}\n"
            f"RFIs: {[r.message for r in result.rfis]}\n"
            f"Assumptions: {list(result.assumptions)}\n"
        )
        resp = self._client.messages.create(
            model=self._model, max_tokens=400, messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
