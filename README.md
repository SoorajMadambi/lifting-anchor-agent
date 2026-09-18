# Lifting-Anchor Placement Agent (WC001)

A proof-of-concept AI agent that places lifting anchors on a precast wall panel, checks
capacity across six handling states, and produces an auditable, explainable result --
failing closed whenever an input is missing, ambiguous, or contradictory.

Built for the BuildTwin technical assessment. See [`DESIGN_NOTE.md`](DESIGN_NOTE.md) for the
architecture, the determinism boundary, and documented assumptions.

**The determinism boundary, in one paragraph:** the agent uses deterministic engineering rules
to calculate geometry, CoG, reactions, handling loads, catalogue capacity, and pass/fail
results -- that layer is authoritative, and it performs its own bounded design iteration
internally (see below). The AI Reasoner is deliberately bounded to orchestration: it can select
which permitted anchor candidate to try next and produce explanations, but it cannot supply or
override any engineering number, position, or the final status. `MockReasoner` (the default) is
fully deterministic and offline; `LLMReasoner` is optional and only activates if
`ANTHROPIC_API_KEY` is set -- either way, the engineering result is identical.

**Two independent, separate iteration mechanisms exist, neither touched by the Reasoner:**
1. **Bounded deterministic design iteration** -- if the trial anchor position (0.207L, shifted to
   the centre of gravity) fails minimum edge distance or axis spacing, the engineering core
   attempts exactly one analytically-derived "move in" position (from the anchor's own catalogue
   limits) before giving up on that candidate. Not a search or optimiser -- see `DESIGN_NOTE.md`.
2. **Catalogue candidate iteration** -- if a candidate still fails (position iteration included),
   the next permitted anchor type is tried, in a fixed deterministic order.

## What's required vs optional

```
Required: the JSON input (data/wc001.json)
Optional: --ifc (a third geometry source parsed from a real IFC file)
Optional: ANTHROPIC_API_KEY (a real LLM reasoner instead of the deterministic MockReasoner)
```

The deterministic engineering result is identical either way. Deleting `reasoner.py` and
unsetting `ANTHROPIC_API_KEY` does not change a single computed number -- see
["Most Important Implementation Principle"](DESIGN_NOTE.md#guiding-principle).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"       # core + pytest
pip install -e ".[ifc]"       # optional: enables --ifc (installs ifcopenshell)
pip install -e ".[llm]"       # optional: enables a real LLM reasoner (installs anthropic)
```

## Run

```bash
python -m liftagent run data/wc001.json
```

or, with the optional third geometry source parsed from the real IFC file:

```bash
python -m liftagent run data/wc001.json --ifc "data/ifc/WC001.ifc"
```

Writes `output/wc001_result.json` (the structured result, brief section 7 schema) and
`output/wc001_elevation.svg` (a 2D elevation with the computed anchor placement), and prints
a human-readable summary to the terminal, e.g.:

```
WC001 LIFTING ANCHOR ANALYSIS
----------------------------------
STATUS: HOLD
Reason: UNRESOLVED_GEOMETRY_CONFLICT -- Geometry conflict -- placement unresolved

Geometry sources:
  approval_design        (AUTHORITATIVE_DESIGN) 4700 mm / 2 opening(s)
  ifc_export             (EXPORT) 4300 mm / 0 opening(s)
  ifc_geometry           (DERIVED_REFERENCE) 4700 mm / openings UNKNOWN

Anchors (ILLUSTRATIVE CANDIDATE -- NOT A RESOLVED PLACEMENT):
  A1: ARL-42 @ x=967mm, y=0mm (clutch RU-42)
  A2: ARL-42 @ x=3722mm, y=0mm (clutch RU-42)

Illustrative check: DEMOULD, anchor A1, utilisation 0.60 [CONDITIONAL]
  NOTE: not used for final acceptance because the result is not resolved (see Reason above).

RFIs:
  1. Resolve geometry discrepancy before finalising placement: ...
  2. Confirm turn method (currently UNCONFIRMED). ...
  3. Confirm the anchor's supplementary reinforcement (Zulage, Figure 2) ...
  4. Explain/reconcile computed self-weight against the brief's reference value: ...

Human sign-off required: YES
```

No API key is required for any of this. `data/wc001.json` is Appendix A transcribed verbatim
and is deliberately unresolved (a genuine geometry conflict, an UNCONFIRMED turn method, and no
confirmation that supplementary reinforcement is placed) -- so the WC001 run **correctly HOLDs**,
per brief section 3.6's hard stops, rather than confidently emitting a wrong number. See the
`tests/test_edge_cases.py::test_clean_input_with_everything_confirmed_accepts` test for a clean
input that reaches `ACCEPT_PROVISIONAL`.

To explicitly confirm/deny the supplementary reinforcement (rather than leaving it UNKNOWN):

```bash
python -m liftagent run data/wc001.json --reinforcement-confirmed true
```

## Tests

```bash
pytest -q
```

**217 tests**, covering:

- deterministic engineering: self-weight, CoG, equilibrium/reaction calculations, trial
  placement + CoG shift, catalogue capacity lookups, utilisation, and the governing check
  (`test_engineering.py`, `test_end_to_end.py`)
- the bounded "move in" position-iteration fallback -- including the exact case that motivated
  it, proof it's skipped when not needed, proof it can't bypass hard stops, determinism, and
  proof it doesn't interfere with catalogue candidate iteration (`test_position_iteration.py`)
- geometry conflict handling and provenance -- WC001's real approval-vs-IFC disagreement, and
  never silently selecting a geometry when sources disagree (`test_conflict.py`)
- IFC extraction from the real `WC001.ifc` file (`test_ifc_geometry.py` -- skipped gracefully,
  not failed, if the optional `ifc` extra isn't installed; see below)
- the structured report/audit contract -- every field a human reviewer needs to audit the
  decision, for both the WC001 (`HOLD`) and a conflict-free (`ACCEPT_PROVISIONAL`) fixture
  (`test_report_contract.py`)
- malformed/invalid input fail-closed behaviour -- missing/zero/negative geometry, invalid
  concrete data, unsupported turn methods, malformed openings, catalogue edge cases -- proving
  unsafe input can never produce a false `ACCEPT_PROVISIONAL` (`test_input_robustness.py`)
- adversarial Reasoner / tool-payload-injection tests -- an `EvilReasoner` and raw
  untrusted-JSON payloads attempting to inject a fake capacity/force/CoG/utilisation/status/
  reinforcement-state, all rejected by the Action Validator or ignored by the Invariant Guard
  (`test_reasoner_security.py`)
- end-to-end agent workflow -- candidate iteration order, two independent Reasoners producing
  identical engineering numbers, and confirmation the Reasoner is never authoritative over any
  engineering calculation (`test_agent_workflow.py`, `test_agent.py`)
- other edge cases: off-centre CoG, a too-thin panel, missing/unconfirmed inputs
  (`test_edge_cases.py`)

Without the optional `ifc` extra installed, IFC-dependent tests skip (not fail) and the suite
reports fewer passing tests -- this is expected, documented behaviour, not a broken run.

## Deterministic mode vs optional LLM mode

By default the agent's bounded `Reasoner` (used only to pick the next catalogue candidate to
try -- never to move an anchor, which is the engineering core's own bounded fallback, not a
Reasoner decision) is `MockReasoner` -- fully deterministic, offline, reproducible. If
`ANTHROPIC_API_KEY` is set, `LLMReasoner` is used instead; it can only ever return the same small
typed action (which anchor to try next) or explanation text -- never an engineering number, a
position, or a status. See `DESIGN_NOTE.md` for why.

## Safety disclaimer

This is a proof-of-concept on synthetic assessment data only. Lifting design is safety-critical;
nothing produced here is for real fabrication. Every result requires human review and
countersignature (`requires_human_signoff: true`, always) -- the system never provides
real-world engineering approval, and `ACCEPT_PROVISIONAL` is never a final sign-off.
