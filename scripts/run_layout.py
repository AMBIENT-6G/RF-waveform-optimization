#!/usr/bin/env python3
"""Shared helpers for results/<run-id> output routing and run manifests."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def manual_run_id() -> str:
    return f"manual_{timestamp_run_id()}"


def infer_run_id_from_path(path: Path | None) -> str | None:
    if path is None:
        return None

    name = path.name
    if len(name) >= 15 and RUN_ID_RE.match(name[:15]):
        return name[:15]

    for part in path.parts:
        if RUN_ID_RE.match(part):
            return part

    return None


def ensure_results_path(results_dir: Path, run_id: str, bucket: str) -> Path:
    if bucket not in {"raw", "tables", "plots", "meta"}:
        raise ValueError(f"Unsupported results bucket: {bucket!r}")
    target = results_dir / run_id / bucket
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_path_from_repo(path: Path, repo_root: Path) -> Path:
    """Resolve relative paths from repository root, keep absolute paths unchanged."""
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_output_path(
    explicit_output: Path | None,
    *,
    results_dir: Path,
    run_id: str,
    bucket: str,
    default_name: str,
) -> Path:
    if explicit_output is not None:
        if explicit_output.is_absolute():
            explicit_output.parent.mkdir(parents=True, exist_ok=True)
            return explicit_output
        target = ensure_results_path(results_dir, run_id, bucket) / explicit_output
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    return ensure_results_path(results_dir, run_id, bucket) / default_name


def write_manifest(
    *,
    results_dir: Path,
    run_id: str,
    script_name: str,
    argv: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    meta_dir = ensure_results_path(results_dir, run_id, "meta")
    manifest_path = meta_dir / "manifest.json"

    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}

    invocations = payload.get("invocations")
    if not isinstance(invocations, list):
        invocations = []
    invocation = {
        "timestamp_utc": utc_now_iso(),
        "script": script_name,
        "argv": argv,
    }
    if extra:
        invocation["extra"] = extra
    invocations.append(invocation)
    payload["invocations"] = invocations
    payload["run_id"] = run_id

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
