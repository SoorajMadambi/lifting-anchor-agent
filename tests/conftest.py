from __future__ import annotations

from pathlib import Path

import pytest

from liftagent import ingest
from liftagent.schema import (
    ConcreteSpec, ElementInput, ExtractionStatus, GeometrySource, Opening,
    ProductionSpec, SourceRole, SourceType,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WC001_JSON = REPO_ROOT / "data" / "wc001.json"
WC001_IFC = REPO_ROOT / "data" / "ifc" / "WC001.ifc"


def make_geometry_source(*, length_mm=4700.0, height_mm=3000.0, thickness_mm=180.0,
                          openings=(), role=SourceRole.AUTHORITATIVE_DESIGN,
                          name="approval_design", status=ExtractionStatus.SUCCESS,
                          source_type=SourceType.APPROVAL_JSON) -> GeometrySource:
    return GeometrySource(
        source_type=source_type, source_name=name, role=role,
        length_mm=length_mm, height_mm=height_mm, thickness_mm=thickness_mm,
        openings=tuple(openings), extraction_status=status, provenance_note="test fixture",
    )


def make_concrete(class_label="C32/40", density=2400.0, first_lift=15.0) -> ConcreteSpec:
    from liftagent.engineering import parse_concrete_class
    cyl, cube = parse_concrete_class(class_label)
    return ConcreteSpec(class_label=class_label, density_kg_per_m3=density,
                         first_lift_strength_mpa=first_lift, cylinder_strength_mpa=cyl, cube_strength_mpa=cube)


def make_production(turn_method="tilting_table") -> ProductionSpec:
    return ProductionSpec(cast_orientation="flat", mould="tilting_table_or_battery",
                           turn_method=turn_method, storage="A-frame stillage, anchor-up")


def make_element_input(*, geometry_sources=None, concrete=None, production=None,
                        element_id="TEST001") -> ElementInput:
    if geometry_sources is None:
        geometry_sources = (make_geometry_source(),)
    return ElementInput(
        element_id=element_id, element_type="precast_rc_wall_panel", description="test element",
        geometry_sources=tuple(geometry_sources), concrete=(concrete or make_concrete()),
        reinforcement_note="", production=(production or make_production()),
        handling_states_required=("demould", "turn", "storage", "transport", "erection"),
    )


@pytest.fixture
def wc001_raw() -> dict:
    return ingest.load_raw(WC001_JSON)


@pytest.fixture
def wc001_element(wc001_raw) -> ElementInput:
    return ingest.build_element_input(wc001_raw)
