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

The adversarial and end-to-end workflow test suites prove this holds under attack, not just in the
happy path: an `EvilReasoner` hallucinating a nonexistent anchor is rejected every time and falls
back to the same deterministic try-order `MockReasoner` uses; raw-payload injection tests confirm
a smuggled capacity/force/CoG/utilisation/status field is rejected outright; and two independent
Reasoner implementations produce byte-identical engineering results, proving
`Reasoner != engineering calculator`.

For WC001 specifically, most workflow decisions turn out to be fully deterministic -- usually
exactly one candidate anchor is worth trying first. The Reasoner is an extension point for genuine
choice, not a conversational wrapper around every calculation.

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

## IFC opening extraction: input-format limitation, not a capability gap

`WC001.ifc` contains 11 `IfcBuildingElementProxy` entities and zero `IfcWall`, `IfcOpeningElement`,
`IfcRelVoidsElement`, `IfcDoor`, or `IfcWindow` entities (confirmed via `ifcopenshell`) -- the panel
is raw, unstructured proxy geometry, not a wall with explicit voids, so `ifc_geometry.py` has no
`IfcRelVoidsElement` relationship to trace and no `IfcOpeningElement` to measure. Opening
extraction is possible when the IFC provides explicit wall/opening semantics; the supplied WC001
export does not, so opening information remains `UNKNOWN` for this source -- not a claim that the
panel has zero physical openings, only that opening geometry can't be deterministically recovered
from this file's current representation. Per the existing "never fabricate, report UNKNOWN"
invariant, the agent does not infer openings from the proxy's raw facets; `ifc_geometry`'s
length/height/thickness bounding-box extraction is unaffected. `UNKNOWN` opening information is
excluded from opening-count conflict comparisons (`ingest.py`) and, where a genuine mandatory
input remains unresolved, still contributes to the existing HOLD/RFI outcome -- fail-closed, never
assuming zero openings.

A separate wall-modeled IFC2x3 file (`IfcWallStandardCase` + `IfcOpeningElement` +
`IfcRelVoidsElement`, `IfcDoor`/`IfcWindow` via `IfcRelFillsElement`) was investigated separately
and confirmed deterministic opening extraction -- wall-local x/width/sill/height via
`ifcopenshell`'s geometry kernel and coordinate transforms -- is technically feasible when an IFC
does carry that structure. This is a known input-format limitation, not evidence that IFC opening
extraction is fundamentally impossible; supporting wall-modeled IFC input is a scoped future
enhancement (see "Production next steps" below), intentionally outside this MVP since WC001.ifc
itself has no such structure to extract from.

## Bounded position iteration ("move in")

A §3.5 audit found one gap: trial `a=0.207L` shifted to the CoG (steps 5-6) simply failed the
candidate if it violated a position-dependent constraint (`edge_distance`, `axis_spacing`, or --
since a second audit -- `opening_void_clash`), never attempting step 11's "move in" remedy.
`engineering.find_feasible_inward_position()` derives the smallest feasible x1 satisfying the
anchor's `min_edge_mm`/`min_axis_mm` (from `catalogue.py`) with every top-edge-reaching opening's
x-span excluded from that range too (a third audit found the fallback itself was opening-blind,
returning its answer even when that exact point sat inside an opening). x2 is always computed as
`2*cog_x_mm - x1`, never independently, so plumb (`(x1+x2)/2 == cog_x_mm`) holds **by
construction**; `compute_reactions()` still re-derives reactions, never assumed. Still a single
closed-form derivation of **at most one** fallback position (two placements total per anchor), not
a search or optimiser -- if none exists, the candidate fails and the next catalogue anchor is
tried. Where the result lands exactly on an opening's inclusive boundary, `math.nextafter()` picks
the nearest strictly-clear representable float -- a numerical-correctness technique only, no
physical clearance asserted -- applied to whichever coordinate (x1 or x2) the opening actually
constrains, never round-tripped through the other (that conversion can silently lose the nudge to
floating-point rounding, verified empirically). The candidate is never trusted directly: the real
`check_edge_axis_wall()`/`check_opening_void_clash()` always re-verify it, failing closed on any
residual clash. The Reasoner never sees a position -- entirely internal to the engineering core.
`Candidate.position_iteration` records what was attempted/failed/selected, for the same
auditability as every other decision. `Status.ITERATE` remains structurally unreachable as a final
status (intentionally -- internal to one candidate's evaluation, not an externally meaningful
state).

Capacity and `reaction_nonnegative` deliberately never trigger this fallback: since x2 is always
`2*cog_x_mm - x1`, `compute_reactions()` reduces to `r1 == r2 == F_total/2` regardless of spacing --
a capacity failure is never a placement problem for the same anchor, and is resolved the existing
way, by trying the next catalogue candidate.

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

**POC:** two-anchor lifts only; no continuous anchor-position optimisation -- only the single
bounded "move in" fallback described above, never a general search; opening-void clash is checked
geometrically (preventing an anchor from landing inside a modeled opening's void), but detailed 3D
clash with opening trimmers, cast-ins, and supplementary reinforcement is not modeled in this POC
and remains outside the deterministic geometry checks -- must be verified separately before
release, with no physical trimmer clearance assumed since the assessment specifies no such value
(`math.nextafter()`, described above, is purely numerical, never a construction-clearance
allowance) -- the agent never signs off or auto-releases regardless (`requires_human_signoff` is
always `True`, brief hard stop 12); vertical-sling assumption (z=1.0), moot for WC001 since every
catalogue anchor's transverse min-wall exceeds this panel's 180mm thickness; IFC opening
extraction not implemented for WC001.ifc's proxy representation (see "IFC opening extraction"
above) -- confirmed feasible for wall-modeled IFC input, not for this file's structure.

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
