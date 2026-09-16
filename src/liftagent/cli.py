"""`python -m liftagent run data/wc001.json [--ifc data/ifc/WC001.ifc] [--out output/]`

JSON input is required; --ifc is optional (plan section 6/32). No API key is
required -- MockReasoner is used unless ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from liftagent import ingest, report, visualize
from liftagent.agent import run_agent
from liftagent.reasoner import MockReasoner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="liftagent")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the lifting-anchor agent on a JSON input")
    run_p.add_argument("json_path", help="Path to the Appendix A JSON input (required)")
    run_p.add_argument("--ifc", default=None, help="Optional path to WC001.ifc for the third geometry source")
    run_p.add_argument("--out", default="output", help="Output directory (default: output/)")
    run_p.add_argument("--reinforcement-confirmed", choices=["true", "false"], default=None,
                        help="Explicitly confirm/deny supplementary reinforcement is placed "
                             "(default: unconfirmed -> UNKNOWN -> HOLD, per brief 3.6)")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
    return 1


def _run(args: argparse.Namespace) -> int:
    raw = ingest.load_raw(args.json_path)

    ifc_geometry = None
    if args.ifc:
        from liftagent.ifc_geometry import extract_wc001_geometry
        ifc_geometry = extract_wc001_geometry(args.ifc)

    element = ingest.build_element_input(raw, ifc_geometry=ifc_geometry)

    reinforcement_confirmed = None
    if args.reinforcement_confirmed is not None:
        reinforcement_confirmed = args.reinforcement_confirmed == "true"

    if os.environ.get("ANTHROPIC_API_KEY"):
        from liftagent.reasoner import LLMReasoner
        reasoner = LLMReasoner()
    else:
        reasoner = MockReasoner()

    result = run_agent(element, reasoner, reinforcement_confirmed=reinforcement_confirmed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{element.element_id.lower()}_result.json"
    svg_path = out_dir / f"{element.element_id.lower()}_elevation.svg"
    json_path.write_text(report.to_json(result), encoding="utf-8")
    svg_path.write_text(visualize.render_svg(result), encoding="utf-8")

    print(report.render_summary(result))
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
