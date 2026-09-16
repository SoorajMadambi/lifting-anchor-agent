from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell", reason="ifcopenshell not installed; IFC tests skipped")

from liftagent.ifc_geometry import extract_wc001_geometry
from liftagent.schema import ExtractionStatus
from tests.conftest import WC001_IFC


def test_wc001_ifc_loads_and_finds_the_panel():
    source = extract_wc001_geometry(str(WC001_IFC))
    assert source.extraction_status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL)
    assert source.length_mm is not None


def test_wc001_ifc_dimensions_are_sane():
    source = extract_wc001_geometry(str(WC001_IFC))
    # a real precast wall panel: length in metres, height ~3m, thickness ~180mm
    assert 1000 < source.length_mm < 10000
    assert 1000 < source.height_mm < 5000
    assert 50 < source.thickness_mm < 500
    assert source.length_mm > source.height_mm > source.thickness_mm


def test_wc001_ifc_length_matches_approval_design_not_the_stale_export_field():
    """Documented finding (DESIGN_NOTE.md): the real parsed geometry agrees
    with approval_design's 4700mm, not ifc_export's 4300mm."""
    source = extract_wc001_geometry(str(WC001_IFC))
    assert source.length_mm == pytest.approx(4700.0, abs=25.0)


def test_opening_extraction_does_not_crash_and_is_marked_unknown():
    source = extract_wc001_geometry(str(WC001_IFC))
    assert source.openings == ()
    assert source.extraction_status == ExtractionStatus.PARTIAL
    assert "opening" in source.provenance_note.lower()


def test_missing_file_fails_gracefully():
    source = extract_wc001_geometry("no/such/file.ifc")
    assert source.extraction_status == ExtractionStatus.FAILED
    assert source.length_mm is None
