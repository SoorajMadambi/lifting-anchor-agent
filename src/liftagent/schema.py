"""Typed data model for the lifting-anchor agent.

Every geometry source and every derived engineering quantity carries explicit
provenance. No field here represents a numeric confidence score -- there is no
uncertainty model in this POC, so none is fabricated (see DESIGN_NOTE.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    APPROVAL_JSON = "APPROVAL_JSON"
    IFC_EXPORT_JSON = "IFC_EXPORT_JSON"
    IFC_PARSED = "IFC_PARSED"


class SourceRole(str, Enum):
    AUTHORITATIVE_DESIGN = "AUTHORITATIVE_DESIGN"
    EXPORT = "EXPORT"
    DERIVED_REFERENCE = "DERIVED_REFERENCE"


class ExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class CheckState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class CheckScope(str, Enum):
    CANDIDATE = "CANDIDATE"
    PROBLEM = "PROBLEM"


class Status(str, Enum):
    ACCEPT_PROVISIONAL = "ACCEPT_PROVISIONAL"
    ITERATE = "ITERATE"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclass(frozen=True)
class Opening:
    id: str
    x_mm: float          # from the panel's left edge
    width_mm: float
    sill_mm: float        # from the panel bottom
    height_mm: float

    @property
    def area_m2(self) -> float:
        return (self.width_mm * self.height_mm) / 1_000_000.0

    @property
    def centroid_x_mm(self) -> float:
        return self.x_mm + self.width_mm / 2.0

    @property
    def centroid_y_from_bottom_mm(self) -> float:
        return self.sill_mm + self.height_mm / 2.0


@dataclass(frozen=True)
class GeometrySource:
    source_type: SourceType
    source_name: str
    role: SourceRole
    length_mm: float | None
    height_mm: float | None
    thickness_mm: float | None
    openings: tuple[Opening, ...]
    extraction_status: ExtractionStatus
    provenance_note: str = ""


@dataclass(frozen=True)
class ConcreteSpec:
    class_label: str                 # e.g. "C32/40"
    density_kg_per_m3: float
    first_lift_strength_mpa: float
    cylinder_strength_mpa: float     # derived from class_label
    cube_strength_mpa: float         # derived from class_label


@dataclass(frozen=True)
class ProductionSpec:
    cast_orientation: str
    mould: str
    turn_method: str                 # "UNCONFIRMED" | "tilting_table" | "free_crane"
    storage: str


@dataclass(frozen=True)
class ElementInput:
    element_id: str
    element_type: str
    description: str
    geometry_sources: tuple[GeometrySource, ...]
    concrete: ConcreteSpec
    reinforcement_note: str
    production: ProductionSpec
    handling_states_required: tuple[str, ...]


@dataclass(frozen=True)
class HandlingState:
    name: str                        # e.g. "ROAD_TRANSPORT"
    psi_dyn: float
    strength_requirement: str        # "first_lift" | "full"
    adhesion: bool = False
    note: str = ""


@dataclass(frozen=True)
class AnchorType:
    name: str
    clutch: str
    capacity_kn: dict[int, float]    # {15: .., 25: .., 35: ..} axial N_zul
    v_zul_kn: float                  # transverse capacity
    min_edge_mm: float
    min_axis_mm: float
    min_wall_axial_mm: float
    min_wall_transverse_mm: float


@dataclass(frozen=True)
class AnchorPlacement:
    id: str
    anchor_type: str
    clutch: str
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    state: CheckState
    scope: CheckScope
    demand: float | None = None
    capacity: float | None = None
    utilisation: float | None = None
    governing_input: str = ""
    rule_reference: str = ""
    explanation: str = ""
    governing: bool = False
    handling_state: str = ""
    anchor_id: str = ""
    conditional_on: str = ""  # non-empty => this PASS/FAIL is not an unconditional acceptance;
                               # e.g. a tabulated capacity PASS while supplementary reinforcement
                               # is UNKNOWN/FAIL is a conditional/reference number, not a verified one


@dataclass(frozen=True)
class ReactionResult:
    anchor_id: str
    handling_state: str
    value_kn: float
    method: str = "two_anchor_static_equilibrium"


@dataclass(frozen=True)
class RFI:
    id: str
    message: str
    rule_reference: str = ""


@dataclass(frozen=True)
class RuleTraceEntry:
    step: str
    result: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SelfWeightResult:
    value_kn: float
    method: str
    source: str
    reference_value_kn: float
    discrepancy_kn: float
    density_kg_per_m3: float    # the actual density used for THIS run's calculation
    concrete_class: str          # the actual concrete class label used for THIS run (e.g. "C32/40")
    warning: str = ""


@dataclass(frozen=True)
class CogResult:
    x_mm: float
    y_mm: float
    source: str
    method: str = "net_area_centroid"
    note: str = ""


@dataclass(frozen=True)
class RigResult:
    spreader_required: bool
    reason: str


@dataclass(frozen=True)
class PositionAttempt:
    """One position tried during the bounded 'move in' search (brief 3.5
    step 11). `result` reuses the existing three-state CheckState enum --
    no new state was introduced for this."""
    x1_mm: float
    x2_mm: float
    result: CheckState
    failed_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class PositionIterationResult:
    """Audit trail for the bounded 'move in' search. attempted=False means
    the initial trial position already satisfied edge distance and axis
    spacing, so no search was needed. `attempts` is always a small,
    analytically-derived, bounded list (0-2 entries) -- never an
    unbounded/looping search; this is a feasibility check, not an
    optimiser (see engineering.find_feasible_inward_position)."""
    attempted: bool
    reason: str = ""
    initial_x1_mm: float | None = None
    initial_x2_mm: float | None = None
    attempts: tuple[PositionAttempt, ...] = ()
    selected_x1_mm: float | None = None
    selected_x2_mm: float | None = None
    method: str = ""


@dataclass(frozen=True)
class Candidate:
    """One evaluated anchor candidate. Never presented as accepted on its own
    -- AgentResult splits every Candidate into either `resolved_candidate`
    (only when status == ACCEPT_PROVISIONAL) or `illustrative_candidate`
    (every other status), so a caller can never confuse "calculated" with
    "accepted" (see plan/user refinement: an illustrative anchor position
    must never become an accepted placement just because the maths ran)."""
    candidate_id: str
    anchor_type: str
    clutch: str
    x1_mm: float
    x2_mm: float
    anchors: tuple[AnchorPlacement, ...]
    checks: tuple[CheckResult, ...]
    governing_check: CheckResult | None
    rig: RigResult | None
    position_iteration: PositionIterationResult | None = None


@dataclass(frozen=True)
class AgentResult:
    element_id: str
    status: Status
    reason: str | None                          # e.g. "UNRESOLVED_GEOMETRY_CONFLICT"; None only for ACCEPT_PROVISIONAL
    geometry_sources: tuple[GeometrySource, ...]
    selected_geometry: str | None
    cog: CogResult | None
    self_weight: SelfWeightResult | None
    resolved_candidate: Candidate | None         # populated ONLY when status == ACCEPT_PROVISIONAL
    illustrative_candidate: Candidate | None      # populated for every other status; never an accepted placement
    rfis: tuple[RFI, ...]
    rule_trace: tuple[RuleTraceEntry, ...]
    assumptions: tuple[str, ...]
    cog_by_source: tuple[CogResult, ...] = ()
    self_weight_by_source: tuple[SelfWeightResult, ...] = ()
    requires_human_signoff: bool = True
    engineering_core_version: str = "0.1.0"
    rule_set_version: str = "assessment-v1"
    catalogue_version: str = "synthetic-poc-v1"
