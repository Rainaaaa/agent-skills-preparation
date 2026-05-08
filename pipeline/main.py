#!/usr/bin/env python3
"""CLI dispatcher for the agent-skills-preparation pipeline.

Subcommands:

    normalize          canonical → normalized + (optional) scan-result filter
    build_phase1       normalized → full_cpt (Phase 1 / Structured CLM CPT)
    build_phase2_hcl   full_cpt → pl_hcl   (Phase 2 / Hierarchical Contrastive)

The two enrichment steps live in their own modules + own CLIs because
they have different dependency profiles (vLLM for T3a, pure CPU for T3c):

    python -m pipeline.enrich_t3a --help
    python -m pipeline.enrich_t3c --help

Legacy subcommands (`build_phase2`, `build_phase3`) from the older
data_preparation/ are intentionally NOT carried over — see the README's
deprecation notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline._shared import configure_logging, detect_default_workers, load_config
from pipeline.build_phase1 import run_phase1
from pipeline.build_phase2_hcl import run_phase2_hcl
from pipeline.normalize import run_normalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="agent-skills-preparation pipeline")
    subparsers = parser.add_subparsers(dest="command")

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", required=True, help="Path to config.yaml (or .json).")
    base.add_argument(
        "--workers", type=int, default=detect_default_workers(),
        help="Worker processes for CPU-heavy record transforms.",
    )
    base.add_argument(
        "--batch-size", type=int, default=128,
        help="Records per processing batch.",
    )
    base.add_argument(
        "--skip-parquet", action="store_true",
        help="Skip parquet export when pyarrow is unavailable.",
    )

    common = argparse.ArgumentParser(add_help=False, parents=[base])
    common.add_argument(
        "--normalized-version", required=True,
        help="Normalized dataset version, e.g. v1 or v2026_04_23.",
    )

    # normalize ---------------------------------------------------------
    normalize_parser = subparsers.add_parser(
        "normalize",
        parents=[common],
        help="Normalize canonical records (with optional scan-result filter).",
    )
    normalize_parser.add_argument(
        "--scan-results", default=None,
        help="Path to agent-skills-scanning unified_results.csv. If set, "
             "skills classified as malicious or misaligned are dropped before "
             "normalization. Falls back to config.filter.scan_results / "
             "$AGENTSKILLS_SCAN_RESULTS / no filter.",
    )
    normalize_parser.set_defaults(handler=handle_normalize)

    # build_phase1 ------------------------------------------------------
    phase_common = argparse.ArgumentParser(add_help=False, parents=[common])
    phase_common.add_argument(
        "--output-version", required=True,
        help="Output version name for the target downstream dataset.",
    )

    phase1_parser = subparsers.add_parser(
        "build_phase1",
        parents=[phase_common],
        help="Build the Phase 1 (Structured CLM CPT) full_cpt dataset.",
    )
    phase1_parser.set_defaults(handler=handle_phase1)

    # build_phase2_hcl --------------------------------------------------
    # Reads Phase 1 outputs (not normalized records) so it takes
    # --full-cpt-version instead of --normalized-version.
    phase2_hcl_parser = subparsers.add_parser(
        "build_phase2_hcl",
        parents=[base],
        help="Build the Phase 2 HCL pl_hcl dataset (T1, T2, T3b).",
    )
    phase2_hcl_parser.add_argument(
        "--full-cpt-version", required=True,
        help="Phase 1 output version to read positives from (e.g. full_cpt_v2).",
    )
    phase2_hcl_parser.add_argument(
        "--output-version", required=True,
        help="Output version for pl_hcl (e.g. pl_hcl_v1).",
    )
    phase2_hcl_parser.set_defaults(handler=handle_phase2_hcl)

    return parser


def handle_normalize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_normalize(
        config=config,
        normalized_version=args.normalized_version,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        skip_parquet=bool(args.skip_parquet),
        scan_results=args.scan_results,
    )


def handle_phase1(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_phase1(
        config=config,
        normalized_version=args.normalized_version,
        output_version=args.output_version,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        skip_parquet=bool(args.skip_parquet),
    )


def handle_phase2_hcl(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_phase2_hcl(
        config=config,
        full_cpt_version=args.full_cpt_version,
        output_version=args.output_version,
        skip_parquet=bool(args.skip_parquet),
    )


def main() -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
