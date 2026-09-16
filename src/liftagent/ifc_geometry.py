"""Optional third geometry source: parse the real WC001.ifc file.

Kept deliberately small (plan section 26/34): extract a reliable bounding
box (length/height/thickness) via ifcopenshell's geometry kernel. Opening
voids are NOT reliably recoverable from this file's raw representation --
inspection shows the panel is faceted with almost entirely
IfcFaceOuterBound entries (one IfcFaceBound total), i.e. openings are cut by
tessellating separate boundary facets around the void rather than encoding a
proper inner face loop. Reconstructing exact opening rectangles from that
would require real polygon-reconstruction geometry work, which is out of
scope for this POC (plan section 26/34: "no unnecessary IFC features"). We
report PARTIAL extraction with a clear note rather than fabricate openings.

No engineering logic lives here -- this module only produces a
GeometrySource, exactly like the JSON sources in ingest.py.
"""
from __future__ import annotations

from liftagent.schema import ExtractionStatus, GeometrySource, SourceRole, SourceType


class IfcExtractionError(Exception):
    pass


def _find_wc001_proxy(ifc_file):
    proxies = ifc_file.by_type("IfcBuildingElementProxy")
    exact = [p for p in proxies if (p.Name or "") == "WC001:WC001"]
    if exact:
        return exact[0]
    candidates = [p for p in proxies if "WC001" in (p.Name or "") and "Wire Loop" not in (p.Name or "")]
    if candidates:
        return candidates[0]
    raise IfcExtractionError("No IfcBuildingElementProxy named 'WC001:WC001' found in WC001.ifc")


def extract_wc001_geometry(ifc_path: str) -> GeometrySource:
    """Best-effort extraction. Never raises for recoverable problems -- on
    failure it returns a GeometrySource with extraction_status=FAILED and a
    provenance_note, so the rest of the pipeline (JSON-only path) can still
    complete successfully (plan section 6/26).
    """
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as exc:
        return GeometrySource(
            source_type=SourceType.IFC_PARSED, source_name="ifc_geometry",
            role=SourceRole.DERIVED_REFERENCE,
            length_mm=None, height_mm=None, thickness_mm=None, openings=(),
            extraction_status=ExtractionStatus.FAILED,
            provenance_note=f"ifcopenshell is not installed ({exc}); IFC geometry source skipped.",
        )

    try:
        ifc_file = ifcopenshell.open(ifc_path)
        proxy = _find_wc001_proxy(ifc_file)

        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, proxy)
        verts = shape.geometry.verts
        xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]

        # ifcopenshell.geom always returns vertices in METRES, regardless of the
        # file's own declared length unit (confirmed empirically: this file
        # declares IfcSIUnit MILLI.METRE, but the returned x-extent for the
        # ~4700mm-long panel comes back as ~4.7) -- so convert with a fixed
        # *1000, not the file's declared unit scale.
        x_extent = (max(xs) - min(xs)) * 1000.0
        y_extent = (max(ys) - min(ys)) * 1000.0
        z_extent = (max(zs) - min(zs)) * 1000.0

        # IFC convention: world Z is vertical -> that extent is the panel height.
        # Of the remaining two (in-plane) extents, the larger is the length and
        # the smaller is the thickness (a wall panel is always far thinner than
        # it is long).
        height_mm = z_extent
        length_mm, thickness_mm = max(x_extent, y_extent), min(x_extent, y_extent)

        return GeometrySource(
            source_type=SourceType.IFC_PARSED, source_name="ifc_geometry",
            role=SourceRole.DERIVED_REFERENCE,
            length_mm=length_mm, height_mm=height_mm, thickness_mm=thickness_mm,
            openings=(),
            extraction_status=ExtractionStatus.PARTIAL,
            provenance_note=(
                "Bounding box (length/height/thickness) extracted from the real WC001.ifc geometry "
                "via ifcopenshell. Opening voids were NOT extracted: this file's proxy geometry is a "
                "faceted BREP with openings cut via tessellated boundary facets rather than a proper "
                "inner face loop, so opening count/position here is UNKNOWN, not confirmed-zero -- "
                "it is excluded from opening-count conflict comparisons (see ingest.py)."
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- any IFC/geometry failure degrades, never crashes the run
        return GeometrySource(
            source_type=SourceType.IFC_PARSED, source_name="ifc_geometry",
            role=SourceRole.DERIVED_REFERENCE,
            length_mm=None, height_mm=None, thickness_mm=None, openings=(),
            extraction_status=ExtractionStatus.FAILED,
            provenance_note=f"IFC extraction failed: {exc!r}",
        )
