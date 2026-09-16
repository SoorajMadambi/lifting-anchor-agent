"""F6: hand-built SVG elevation of the panel with the computed anchor
placement (plan section 30). No plotting dependency needed.

Visual hierarchy (most to least prominent): title -> STATUS -> Reason ->
the engineering drawing -> the illustrative-candidate disclaimer -> detail
panel. The HOLD/REJECT reason must never be visually subordinate to the
illustrative candidate it explains.
"""
from __future__ import annotations

from liftagent.report import REASON_MESSAGES
from liftagent.schema import AgentResult, GeometrySource, Status

_STATUS_COLOR = {
    Status.ACCEPT_PROVISIONAL: "#2f6b3f",
    Status.ITERATE: "#9c6a13",
    Status.HOLD: "#c0392b",
    Status.REJECT: "#c0392b",
}

_PAD = 60
_HEADER_H = 90
_SCALE = 0.12  # mm -> px


def _find_source(result: AgentResult) -> GeometrySource | None:
    name = result.cog.source if result.cog is not None else None
    for s in result.geometry_sources:
        if s.source_name == name:
            return s
    usable = [s for s in result.geometry_sources if s.length_mm is not None]
    return usable[0] if usable else None


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(result: AgentResult) -> str:
    resolved = result.status == Status.ACCEPT_PROVISIONAL
    candidate = result.resolved_candidate if resolved else result.illustrative_candidate

    source = _find_source(result)
    if source is None or source.length_mm is None:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200">'
            '<text x="20" y="100" font-family="Arial" font-size="16" fill="#c0392b">'
            "No usable geometry available to visualise.</text></svg>"
        )

    L, H = source.length_mm, source.height_mm
    panel_w, panel_h = L * _SCALE, H * _SCALE
    width = panel_w + 2 * _PAD + 260
    height = _HEADER_H + panel_h + 2 * _PAD + 80

    def px(x_mm: float) -> float:
        return _PAD + x_mm * _SCALE

    def py(y_from_top_mm: float) -> float:
        return _HEADER_H + _PAD + y_from_top_mm * _SCALE

    status_color = _STATUS_COLOR.get(result.status, "#1f2933")

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" font-family="Arial">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="white"/>',
    ]

    # ---- Header: title, STATUS, Reason -- the most prominent thing on the page ----
    parts.append(f'<text x="{_PAD}" y="26" font-size="13" fill="#52606d" letter-spacing="1">'
                 f'{result.element_id} -- LIFTING ANALYSIS</text>')
    parts.append(f'<text x="{_PAD}" y="54" font-size="26" font-weight="bold" fill="{status_color}">'
                 f'STATUS: {result.status.value}</text>')
    if result.reason is not None:
        reason_text = f'Reason: {REASON_MESSAGES.get(result.reason, result.reason)}'
        parts.append(f'<text x="{_PAD}" y="76" font-size="14" font-weight="bold" fill="{status_color}">'
                     f'{_xml_escape(reason_text)}</text>')
    parts.append(f'<line x1="0" y1="{_HEADER_H - 4}" x2="{width:.0f}" y2="{_HEADER_H - 4}" stroke="#d2d6dc" stroke-width="1"/>')

    # ---- The engineering drawing ----
    parts.append(
        f'<rect x="{px(0):.1f}" y="{py(0):.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" '
        f'fill="#f5f7fa" stroke="#1f2933" stroke-width="1.8"/>'
    )

    if not resolved:
        parts.append(
            f'<text x="{px(0):.1f}" y="{py(0) - 40:.1f}" font-size="11" fill="#c0392b" font-weight="bold">'
            f'ILLUSTRATIVE CANDIDATE ({_xml_escape(source.source_name)}) -- NOT A RESOLVED PLACEMENT</text>'
        )

    for o in source.openings:
        top_from_top = H - (o.sill_mm + o.height_mm)
        parts.append(
            f'<rect x="{px(o.x_mm):.1f}" y="{py(top_from_top):.1f}" width="{o.width_mm * _SCALE:.1f}" '
            f'height="{o.height_mm * _SCALE:.1f}" fill="#e4e7eb" stroke="#52606d" stroke-width="1" '
            f'stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{px(o.centroid_x_mm):.1f}" y="{py(top_from_top + o.height_mm / 2):.1f}" '
            f'font-size="10" fill="#52606d" text-anchor="middle" dominant-baseline="middle">{_xml_escape(o.id)}</text>'
        )

    if result.cog is not None:
        cx, cy_from_top = px(result.cog.x_mm), py(H - result.cog.y_mm)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy_from_top:.1f}" r="8" fill="white" stroke="#1f2933" stroke-width="1.4"/>')
        parts.append(
            f'<line x1="{cx:.1f}" y1="{py(0):.1f}" x2="{cx:.1f}" y2="{py(H):.1f}" '
            f'stroke="#1f2933" stroke-width="0.8" stroke-dasharray="3 3"/>'
        )
        parts.append(f'<text x="{cx + 12:.1f}" y="{cy_from_top:.1f}" font-size="10" fill="#1f2933">CoG</text>')

    anchors = candidate.anchors if candidate is not None else ()
    for a in anchors:
        ax = px(a.x_mm)
        ay = py(0)
        parts.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="#c0392b"/>')
        parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ax:.1f}" y2="{ay + 26:.1f}" stroke="#c0392b" stroke-width="2"/>')
        parts.append(f'<text x="{ax:.1f}" y="{ay - 10:.1f}" font-size="11" fill="#c0392b" text-anchor="middle" font-weight="bold">{a.id}</text>')
        parts.append(f'<text x="{ax:.1f}" y="{ay + 40:.1f}" font-size="9.5" fill="#52606d" text-anchor="middle">{_xml_escape(a.anchor_type)}</text>')

    if len(anchors) == 2:
        x1, x2 = sorted(a.x_mm for a in anchors)
        spacing_y = py(0) - 22
        parts.append(f'<line x1="{px(x1):.1f}" y1="{spacing_y:.1f}" x2="{px(x2):.1f}" y2="{spacing_y:.1f}" stroke="#9aa5b1" stroke-width="1"/>')
        parts.append(
            f'<text x="{px((x1 + x2) / 2):.1f}" y="{spacing_y - 5:.1f}" font-size="10" fill="#52606d" '
            f'text-anchor="middle">spacing {x2 - x1:.0f}mm</text>'
        )

    rig = candidate.rig if candidate is not None else None
    if rig is not None:
        parts.append(
            f'<text x="{px(0):.1f}" y="{py(H) + 22:.1f}" font-size="11" fill="#1f2933">'
            f'Rig: {"SPREADER REQUIRED" if rig.spreader_required else "direct slings"} -- {_xml_escape(rig.reason)}</text>'
        )

    # ---- Side info panel ----
    info_x = panel_w + _PAD + 30
    info_y = py(0) + 10
    parts.append(f'<rect x="{info_x - 10:.1f}" y="{py(0) - 10:.1f}" width="240" height="{panel_h + 20:.1f}" rx="6" fill="#fbfcfd" stroke="#9aa5b1" stroke-width="1"/>')
    parts.append(f'<text x="{info_x:.1f}" y="{info_y:.1f}" font-size="12" font-weight="bold" fill="#1f2933">{_xml_escape(result.element_id)} -- lifting data</text>')
    row = info_y + 22

    gc = candidate.governing_check if candidate is not None else None
    if gc is not None and gc.utilisation is not None:
        label = "governing check" if resolved else "illustrative check only"
        parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="10" font-weight="bold" fill="#52606d">{label}:</text>')
        row += 14
        parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="10" fill="#52606d">{gc.handling_state} / {gc.anchor_id}</text>')
        row += 16
        parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="10" fill="#c0392b">utilisation: {gc.utilisation:.2f}</text>')
        row += 14
        if gc.conditional_on:
            parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="8.5" fill="#9c6a13" font-weight="bold">CONDITIONAL (not verified):</text>')
            row += 12
            parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="8.5" fill="#9c6a13">reinforcement not confirmed -- see RFI</text>')
            row += 12
        if not resolved:
            parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="8.5" fill="#9c6a13">NOTE: not used for final acceptance</text>')
            row += 12
            parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="8.5" fill="#9c6a13">(result is not resolved -- see Reason).</text>')
            row += 12
        row += 8
    for rfi in result.rfis:
        wrapped = rfi.message if len(rfi.message) <= 46 else rfi.message[:43] + "..."
        parts.append(f'<text x="{info_x:.1f}" y="{row:.1f}" font-size="9" fill="#9c6a13">RFI: {_xml_escape(wrapped)}</text>')
        row += 14

    parts.append("</svg>")
    return "\n".join(parts)
