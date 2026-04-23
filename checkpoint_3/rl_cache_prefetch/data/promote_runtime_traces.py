#!/usr/bin/env python3
"""
Promote runtime-collected traces from data/runs/<run_id>/ into canonical data/ files.

This script standardizes the workflow when you want evaluation/training to use
LMCache cold-pass traces (with runtime provenance columns) instead of synthetic
trace files.

Usage:
    python data/promote_runtime_traces.py
    python data/promote_runtime_traces.py --run-id 20260423T172836Z
    python data/promote_runtime_traces.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional


DATA_DIR = Path(__file__).resolve().parent
RUNS_DIR = DATA_DIR / "runs"

TRACE_NAMES = ("prefix", "rag", "nocontext", "multiturn")
REQUIRED_TRACE_COLUMNS = {
    "query_id",
    "query_text",
    "chunk_ids_needed",
    "runtime_chunk_ids",
    "chunk_event_source",
    "chunk_id_source",
    "runtime_event_count",
}


def list_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])


def infer_latest_run(run_dirs: Iterable[Path]) -> Optional[Path]:
    dirs = list(run_dirs)
    if not dirs:
        return None
    # Timestamp-based names sort lexicographically.
    return dirs[-1]


def validate_trace_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path.name}"

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_TRACE_COLUMNS - fieldnames
        if missing:
            return False, f"missing columns in {path.name}: {sorted(missing)}"

        first = next(reader, None)
        if first is None:
            return False, f"empty file: {path.name}"

        source = (first.get("chunk_event_source") or "").strip()
        if source != "lmcache_store_coldpass":
            return False, (
                f"unexpected chunk_event_source in {path.name}: {source!r} "
                f"(expected 'lmcache_store_coldpass')"
            )

    return True, "ok"


def promote_run(run_dir: Path) -> None:
    print(f"[promote] Using run directory: {run_dir}")

    trace_paths: dict[str, Path] = {}
    for name in TRACE_NAMES:
        trace_path = run_dir / f"traces_{name}.csv"
        ok, msg = validate_trace_file(trace_path)
        if not ok:
            raise RuntimeError(msg)
        trace_paths[name] = trace_path

    ttft_path = run_dir / "ttft_lookup.json"
    if not ttft_path.exists():
        raise RuntimeError(f"missing file: {ttft_path.name}")

    for name, src in trace_paths.items():
        dst = DATA_DIR / f"traces_{name}.csv"
        shutil.copy2(src, dst)
        print(f"[promote] {src.name} -> {dst.name}")

    ttft_dst = DATA_DIR / "ttft_lookup.json"
    shutil.copy2(ttft_path, ttft_dst)
    print(f"[promote] {ttft_path.name} -> {ttft_dst.name}")

    metadata = {
        "source_run": run_dir.name,
        "source_path": str(run_dir),
        "trace_names": list(TRACE_NAMES),
        "chunk_event_source": "lmcache_store_coldpass",
    }
    metadata_path = DATA_DIR / "runtime_trace_source.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[promote] wrote {metadata_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote LMCache runtime traces from data/runs to data/."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run directory name under data/runs (e.g. 20260423T172836Z)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available run directories and exit",
    )
    args = parser.parse_args()

    run_dirs = list_run_dirs()
    if args.list:
        if not run_dirs:
            print("[promote] No run directories found under data/runs")
            return
        print("[promote] Available runs:")
        for d in run_dirs:
            print(f"  - {d.name}")
        return

    if not run_dirs:
        raise RuntimeError("No run directories found under data/runs")

    if args.run_id:
        run_dir = RUNS_DIR / args.run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise RuntimeError(f"run-id not found: {args.run_id}")
    else:
        run_dir = infer_latest_run(run_dirs)
        assert run_dir is not None

    promote_run(run_dir)


if __name__ == "__main__":
    main()
