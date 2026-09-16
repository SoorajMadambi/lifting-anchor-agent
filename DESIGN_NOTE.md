# Design Note -- Lifting-Anchor Placement Agent (WC001)

## Architecture

```
INPUTS (JSON required, IFC optional)
      |
INGEST + PROVENANCE  ->  CONFLICT DETECTOR  ->  GEOMETRY CANDIDATES
      |
   AGENT WORKFLOW  (bounded deterministic control flow + typed Reasoner choice, agent.py)
      |
   calls typed tools (tools.py) ------------------------------+
      |                                                       |
   ENGINEERING CORE (engineering.py, catalogue.py)            |
   CoG . statics . loads . capacity . placement . rig         |
      |                                                       |
   DETERMINISTIC RESULTS                                      |
      |                                                       |
   REASONER (only when a genuine choice exists: which     <---+
   catalogue candidate to try next)
      |
   TYPED ACTION ONLY (AgentAction: TRY_ANCHOR / ESCALATE_HOLD)
      |
   ACTION VALIDATOR (validator.py -- rejects anything malformed
   or not in catalogue.py; never trusted if rejected)
      |
   INVARIANT GUARD (enforce_invariants(), agent.py -- recomputes
   the final status purely from deterministic check results;
   the Reasoner's own opinion of the status is never read)
      |
   REPORT (JSON, brief section 7 schema) + SVG elevation
```

**Agent** = orchestrator of deterministic tools. **Reasoner** = bounded decision/explanation
component, typed-action-only. **Engineering core** = sole source of engineering truth.
**Invariant guard** = final safety boundary.

Every calculated `Candidate` is split by `enforce_invariants()`'s status into exactly one of two
mutually-exclusive fields: `resolved_candidate` (only when `status == ACCEPT_PROVISIONAL`) or
`illustrative_candidate` (every other status) -- a HOLD/REJECT result still exposes a candidate
for audit, but it can never be read as accepted, since no other code path writes
`resolved_candidate`. `status`, `reason`, and this split are computed together from one priority
order, so they cannot drift apart. `requires_human_signoff` is always `True`, for every status --
the agent itself never signs off (brief hard stop 12).

## Determinism boundary

The LLM never performs engineering arithmetic or determines safety pass/fail/status.
`engineering.py`, `catalogue.py`, `ifc_geometry.py`, and `schema.py` are pure, deterministic,
side-effect-free, and unit tested. The Reasoner (`reasoner.py`) is consulted at exactly one point
-- choosing the next catalogue anchor to try -- and its action payload has exactly one field,
`anchor_type`; there is structurally nowhere to put a fabricated force, capacity, CoG, or
utilisation. `validator.py` rejects any untrusted raw payload carrying unexpected fields or an
anchor not in `catalogue.py`. `agent.py::enforce_invariants()` always recomputes the final status
from the raw deterministic checks -- the Reasoner's opinion is never read.

The adversarial and end-to-end workflow suites (`test_reasoner_security.py`,
`test_agent_workflow.py`, `test_agent.py`) prove this holds under attack: an `EvilReasoner` that
hallucinates a nonexistent anchor is rejected every time and falls back to the same deterministic
try-order `MockReasoner` uses; raw-payload injection tests confirm a smuggled capacity/force/CoG/
utilisation/status field is rejected outright; and two independent Reasoner implementations are
shown to produce byte-identical engineering results, proving `Reasoner != engineering calculator`.

For WC001 specifically, most workflow decisions turn out to be fully deterministic -- usually
exactly one candidate anchor is worth trying first. The Reasoner is an extension point for genuine
choice and exception handling, not a conversational wrapper around every calculation.

## Geometry conflict -- three disagreeing sources

Appendix A's `approval_design` (4700mm, 2 openings) and `ifc_export` (4300mm, 0 openings) already
disagree. This POC adds a third source, `ifc_geometry`, parsed directly from the real `WC001.ifc`
file via `ifcopenshell` (optional -- the JSON-only path needs no IFC dependency). **Finding:** the
real parsed geometry's length comes back as ~4700mm, agreeing with `approval_design`, not
`ifc_export` -- suggesting `ifc_export` may itself be the stale value. Opening voids could not be
reliably extracted from this file's raw faceted-BREP representation, so that source's opening
count is reported `UNKNOWN`, never fabricated as zero.

Any disagreement forces `status = HOLD, selected_geometry = null`. The engineering core is still
*evaluated* against the `AUTHORITATIVE_DESIGN` source so a checking engineer has something
concrete to inspect -- but that calculation is written only to `illustrative_candidate`, never
`resolved_candidate`, and both the JSON and SVG label it "not a resolved placement."

## Self-weight: derived, not copied from the figure

Appendix A's JSON has no `G` field. The brief's Figure 1 shows "G = 72.6 kN", but that doesn't
reconcile with any geometry+density combination given (net-of-openings `approval_design`: ~44.9
kN; gross `ifc_export`: ~54.7 kN). This POC derives self-weight deterministically from
`volume x density`, and reports 72.6 kN only as a reference value with an explicit warning when it
doesn't reconcile -- never as an authoritative input. Consequence: with the derived ~44.9 kN,
**demould** (not transport) governs WC001's illustrative candidate, since the adhesion term is a
larger share of a smaller self-weight -- a different governing case than the brief's own worked
example.

## Strength column: cube-strength interpretation, isolated

The catalogue's top column is "@35 MPa"; C32/40 has cylinder strength 32 MPa and cube strength 40
MPa. `select_strength_column()` uses the cube reading -- the only interpretation consistent with
the brief's own worked example (72.6/80 = 0.91 against ARL-42's 80kN "@35" column). Isolated in one
function so the convention can be changed in one place if a real engineer's interpretation
differs; a regression test pins it against the brief's own numbers.

## AI usage

**Building this project:** the entire repository -- architecture, engineering formulas, tests, and
this note -- was built in a single session with Claude Code (Claude Sonnet 5), from the assessment
brief and the two supplied IFC files, including cross-checking the brief's own numbers against
hand-derived values (finding the self-weight and strength-column issues above).

**At runtime:** AI is confined to the bounded Reasoner described above -- selecting which
candidate to try next and producing explanations. It is never used to decide any engineering
number; that boundary is what this whole design enforces and tests against.

## POC-grade vs production-grade

**POC:** two-anchor lifts only; no continuous anchor-position optimisation (a CoG-shifted trial
position that violates a constraint simply fails that candidate, the next catalogue anchor is
tried); opening-void clash is checked, but exact 3D reinforcement/trimmer clash is always
`UNKNOWN` and correctly forces HOLD rather than a fabricated PASS; vertical-sling assumption
(z=1.0), moot for WC001 since every catalogue anchor's transverse min-wall exceeds this panel's
180mm thickness; IFC opening extraction not attempted (would need polygon reconstruction from
tessellated facets).

**Production next steps:** 4-point lifts with a load-balancing traverse (n > 2); real
reinforcement/cast-in geometry for a genuine 3D clash check; a certified anchor catalogue instead
of the synthetic POC table; validated IFC/BIM integration; versioned rule-sets; a genuine
confidence/uncertainty model (deliberately not attempted -- this POC uses explicit provenance and
PASS/FAIL/UNKNOWN instead of fabricated confidence); an engineering approval workflow with an audit
trail; a chat affordance for "why is anchor 1 here?", answerable from the stored `rule_trace`.

## Guiding principle

The deterministic engineering workflow does not depend on an external LLM or API key: with
`MockReasoner` (the default) the workflow runs offline and reproducibly, and replacing the
Reasoner never changes a computed engineering result. The goal was never to demonstrate that an
LLM can do engineering -- it's to demonstrate that an AI agent can safely orchestrate deterministic
engineering software without ever being allowed to override it.
