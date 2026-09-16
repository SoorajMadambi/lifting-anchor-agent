"""Typed tool surface the Agent calls -- thin wrappers over engineering.py
that additionally log a structured RuleTraceEntry per call (plan section 24
and 29). The Reasoner never calls these directly; it only ever returns an
action for the Agent to act on (plan section 2/23).
"""
from __future__ import annotations

from liftagent import catalogue, engineering
from liftagent.schema import (
    AnchorType, CheckResult, CogResult, ConcreteSpec, GeometrySource,
    HandlingState, Opening, ReactionResult, RigResult, RuleTraceEntry,
    SelfWeightResult,
)


class ToolBox:
    def __init__(self) -> None:
        self.trace: list[RuleTraceEntry] = []

    def _log(self, step: str, result: str, **data) -> None:
        self.trace.append(RuleTraceEntry(step=step, result=result, data=data))

    def compute_self_weight(self, geometry: GeometrySource, density_kg_per_m3: float,
                             concrete_class: str = "") -> SelfWeightResult:
        r = engineering.compute_self_weight(geometry, density_kg_per_m3, concrete_class=concrete_class)
        self._log("self_weight", "COMPUTED", value_kN=round(r.value_kn, 2), source=r.source,
                   discrepancy_kN=round(r.discrepancy_kn, 2), density_kg_per_m3=r.density_kg_per_m3,
                   concrete_class=r.concrete_class)
        return r

    def compute_cog(self, geometry: GeometrySource) -> CogResult:
        r = engineering.compute_cog(geometry)
        self._log("cog", "COMPUTED", x_mm=round(r.x_mm, 1), y_mm=round(r.y_mm, 1), source=r.source)
        return r

    def trial_placement(self, length_mm: float) -> tuple[float, float]:
        x1, x2 = engineering.trial_placement(length_mm)
        self._log("anchor_trial", "COMPUTED", x1_mm=round(x1, 1), x2_mm=round(x2, 1))
        return x1, x2

    def shift_to_cog(self, x1: float, x2: float, length_mm: float, cog_x_mm: float) -> tuple[float, float]:
        nx1, nx2 = engineering.shift_to_cog(x1, x2, length_mm, cog_x_mm)
        self._log("anchor_shift_to_cog", "COMPUTED", x1_mm=round(nx1, 1), x2_mm=round(nx2, 1))
        return nx1, nx2

    def compute_reactions(self, total_load_kn: float, x1: float, x2: float, cog_x_mm: float) -> tuple[float, float]:
        r1, r2 = engineering.compute_reactions(total_load_kn, x1, x2, cog_x_mm)
        self._log("reactions", "COMPUTED", r1_kN=round(r1, 2), r2_kN=round(r2, 2))
        return r1, r2

    def check_edge_axis_wall(self, anchor: AnchorType, x1: float, x2: float,
                              length_mm: float, thickness_mm: float) -> list[CheckResult]:
        results = engineering.check_edge_axis_wall(anchor, x1, x2, length_mm, thickness_mm)
        for r in results:
            self._log(r.check_name, r.state.value, anchor=anchor.name)
        return results

    def check_opening_void_clash(self, x1: float, x2: float, openings: tuple[Opening, ...],
                                  height_mm: float) -> CheckResult:
        r = engineering.check_opening_void_clash(x1, x2, openings, height_mm)
        self._log(r.check_name, r.state.value)
        return r

    def check_supplementary_reinforcement(self, confirmed: bool | None) -> CheckResult:
        r = engineering.check_supplementary_reinforcement(confirmed)
        self._log(r.check_name, r.state.value)
        return r

    def check_clutch_match(self, anchor: AnchorType, clutch_used: str) -> CheckResult:
        r = engineering.check_clutch_match(anchor, clutch_used)
        self._log(r.check_name, r.state.value, anchor=anchor.name)
        return r

    def decide_rig(self, anchor: AnchorType, thickness_mm: float, beta_deg: float = 0.0) -> RigResult:
        r = engineering.decide_rig(anchor, thickness_mm, beta_deg)
        self._log("rig_decision", "COMPUTED", spreader_required=r.spreader_required, anchor=anchor.name)
        return r

    def check_capacity(self, anchor: AnchorType, state: HandlingState, demand_kn: float,
                        strength_column_mpa: int | None, anchor_id: str) -> CheckResult:
        r = engineering.check_capacity(anchor, state, demand_kn, strength_column_mpa, anchor_id)
        self._log(f"capacity_{state.name}_{anchor_id}", r.state.value,
                   demand_kN=round(demand_kn, 2), utilisation=(round(r.utilisation, 3) if r.utilisation is not None else None))
        return r

    def select_strength_column(self, concrete: ConcreteSpec, requirement: str) -> int | None:
        col = engineering.select_strength_column(concrete, requirement)
        self._log("strength_column", "COMPUTED", requirement=requirement, column_mpa=col)
        return col

    def total_load_for_state(self, state: HandlingState, self_weight_kn: float, z: float,
                              length_mm: float, height_mm: float) -> float:
        f = engineering.total_load_for_state(state, self_weight_kn, z, length_mm, height_mm)
        self._log(f"load_{state.name}", "COMPUTED", F_total_kN=round(f, 2))
        return f

    def get_anchor(self, anchor_type: str) -> AnchorType:
        return catalogue.get_anchor(anchor_type)
