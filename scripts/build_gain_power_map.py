#!/usr/bin/env python3
"""Build a gain-to-power lookup by matching reference dBm levels to scope data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CSV = REPO_ROOT / "data" / "reference" / "harvester-chart-data.csv"
DEFAULT_SCOPE_JSONL = REPO_ROOT / "data" / "meas-tone-power-scope.jsonl"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "gain-power-map.csv"


def watts_to_dbm(power_w: float) -> float:
    if power_w <= 0.0:
        raise ValueError(f"Power must be positive to convert to dBm, got {power_w!r}")
    return 10.0 * math.log10(power_w / 1e-3)


def dbm_to_mw(power_dbm: float) -> float:
    return math.pow(10.0, power_dbm / 10.0)


def load_reference_levels(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "level_dbm" not in reader.fieldnames:
            raise ValueError(f"Reference CSV missing required 'level_dbm' column: {path}")
        return [float(row["level_dbm"]) for row in reader if row.get("level_dbm")]


def load_scope_gain_points(path: Path, tone: int) -> list[dict[str, float]]:
    gain_points: list[dict[str, float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            if int(record.get("tone", -1)) != tone:
                continue

            gain_db = float(record["gain_db"])
            readings = record.get("readings", [])
            scope_values = [
                float(reading["scope_power_w"])
                for reading in readings
                if isinstance(reading, dict) and reading.get("scope_power_w") is not None
            ]
            if not scope_values:
                raise ValueError(
                    f"No scope_power_w readings found for tone={tone}, gain={gain_db:g} at line {line_number}"
                )

            mean_power_w = sum(scope_values) / len(scope_values)
            gain_points.append(
                {
                    "gain_db": gain_db,
                    "scope_power_dbm": watts_to_dbm(mean_power_w),
                }
            )

    if not gain_points:
        raise ValueError(f"No records found for tone={tone} in {path}")

    gain_points.sort(key=lambda item: item["gain_db"])
    return gain_points


def build_gain_power_rows(reference_levels_dbm: list[float], gain_points: list[dict[str, float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for input_level_dbm in reference_levels_dbm:
        best_match = min(
            gain_points,
            key=lambda point: (abs(point["scope_power_dbm"] - input_level_dbm), point["gain_db"]),
        )
        rows.append(
            {
                "scope_power_dbm": best_match["scope_power_dbm"],
                "gain_db": best_match["gain_db"],
                "input_level_dbm": input_level_dbm,
                "input_level_mw": dbm_to_mw(input_level_dbm),
            }
        )

    return rows


def deduplicate_rows_by_gain(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    best_by_gain: dict[float, dict[str, float]] = {}

    for row in rows:
        gain_db = float(row["gain_db"])
        candidate_error = abs(float(row["scope_power_dbm"]) - float(row["input_level_dbm"]))
        current = best_by_gain.get(gain_db)

        if current is None:
            best_by_gain[gain_db] = row
            continue

        current_error = abs(float(current["scope_power_dbm"]) - float(current["input_level_dbm"]))
        if candidate_error < current_error:
            best_by_gain[gain_db] = row

    return sorted(best_by_gain.values(), key=lambda row: float(row["input_level_dbm"]))


def write_gain_power_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope_power_dbm", "gain_db", "input_level_dbm", "input_level_mw"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scope_power_dbm": f"{row['scope_power_dbm']:.12g}",
                    "gain_db": f"{row['gain_db']:.12g}",
                    "input_level_dbm": f"{row['input_level_dbm']:.12g}",
                    "input_level_mw": f"{row['input_level_mw']:.12g}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match each reference input level (dBm) to the closest tone-0 scope measurement "
            "mean power (converted from scope_power_w to dBm)."
        )
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE_CSV,
        help=f"Reference CSV with level_dbm column (default: {DEFAULT_REFERENCE_CSV})",
    )
    parser.add_argument(
        "--scope-jsonl",
        type=Path,
        default=DEFAULT_SCOPE_JSONL,
        help=f"Scope JSONL file with readings[].scope_power_w (default: {DEFAULT_SCOPE_JSONL})",
    )
    parser.add_argument(
        "--tone",
        type=int,
        default=0,
        help="Tone to use from the scope JSONL (default: 0)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_levels_dbm = load_reference_levels(args.reference_csv)
    gain_points = load_scope_gain_points(args.scope_jsonl, tone=args.tone)
    rows = build_gain_power_rows(reference_levels_dbm, gain_points)
    rows = deduplicate_rows_by_gain(rows)
    write_gain_power_csv(args.output_csv, rows)
    print(
        f"Wrote {len(rows)} rows to {args.output_csv} using tone={args.tone} "
        f"from {args.scope_jsonl.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
