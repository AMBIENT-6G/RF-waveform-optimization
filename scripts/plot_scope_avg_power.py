#!/usr/bin/env python3
"""Plot average scope-measured power versus configured TX gain for each tone."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_layout import infer_run_id_from_path, manual_run_id, resolve_output_path, write_manifest


MEASUREMENT_STEM_SUFFIX = "meas-tone-power-scope"
MEASUREMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
READING_KEY_CHOICES = ("auto", "scope_power_w", "pwr_pw")
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_results_dir(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_candidate = path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()

    if cwd_candidate.exists():
        return cwd_candidate
    if repo_candidate.exists():
        return repo_candidate
    return repo_candidate


def candidate_search_roots(search_dir: Path) -> list[Path]:
    if search_dir.is_absolute():
        return [search_dir.resolve()]

    roots = [search_dir.resolve(), (REPO_ROOT / search_dir).resolve()]
    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def is_measurement_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and path.stem.endswith(MEASUREMENT_STEM_SUFFIX)
    )


def measurement_timestamp(path: Path) -> datetime | None:
    if not path.stem.endswith(MEASUREMENT_STEM_SUFFIX):
        return None

    prefix = path.stem[: -len(MEASUREMENT_STEM_SUFFIX)].rstrip("_-")
    if not prefix:
        return None

    try:
        return datetime.strptime(prefix, MEASUREMENT_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def measurement_sort_key(path: Path) -> tuple[int, float]:
    timestamp = measurement_timestamp(path)
    if timestamp is not None:
        return (1, timestamp.timestamp())
    return (0, path.stat().st_mtime)


def discover_measurement_files(search_dir: Path = Path("results")) -> list[Path]:
    candidates: list[Path] = []
    for search_root in candidate_search_roots(search_dir):
        if not search_root.exists():
            continue
        candidates.extend(path.resolve() for path in search_root.rglob("*") if is_measurement_file(path))

    return sorted(candidates, key=measurement_sort_key, reverse=True)


def resolve_input_paths(input_path: Path | None, include_all: bool, search_dir: Path) -> list[Path]:
    if include_all:
        matches = discover_measurement_files(search_dir=search_dir)
        if not matches:
            raise FileNotFoundError(
                f"No measurement files found in {search_dir.resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
            )
        return matches

    if input_path is not None:
        resolved = input_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [resolved]

    matches = discover_measurement_files(search_dir=search_dir)
    if not matches:
        raise FileNotFoundError(
            f"No measurement files found in {search_dir.resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
        )
    return [matches[0]]


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return load_jsonl_records(text)

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        if {"tone", "gain_db", "readings"} <= parsed.keys():
            return [parsed]
        return flatten_mapping_records(parsed)

    raise ValueError(f"Unsupported JSON structure in {path}")


def load_jsonl_records(text: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        records.append(item)
    return records


def flatten_mapping_records(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, value in mapping.items():
        if not isinstance(value, list):
            continue

        tone, gain = parse_key_pair(key)
        records.append(
            {
                "tone": tone,
                "gain_db": gain,
                "readings": value,
            }
        )
    return records


def parse_key_pair(key: str) -> tuple[int, float]:
    cleaned = key.strip().strip("()")
    parts = [item.strip() for item in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse mapping key {key!r}; expected '(tone, gain)'")
    return int(parts[0]), float(parts[1])


def choose_reading_key(records: list[dict[str, Any]], requested_key: str) -> str:
    if requested_key != "auto":
        return requested_key

    for candidate in ("scope_power_w", "pwr_pw"):
        for record in records:
            readings = record.get("readings", [])
            if not isinstance(readings, list):
                continue
            for reading in readings:
                if isinstance(reading, dict) and reading.get(candidate) is not None:
                    return candidate

    raise ValueError(
        "Could not infer reading key. Pass --reading-key scope_power_w or --reading-key pwr_pw."
    )


def group_power_by_tone_gain(
    records: list[dict[str, Any]],
    reading_key: str,
) -> dict[int, dict[float, list[float]]]:
    grouped: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        if not isinstance(record, dict):
            continue

        tone = record.get("tone")
        gain = record.get("gain_db")
        readings = record.get("readings", [])

        try:
            tone_int = int(tone)
            gain_float = float(gain)
        except (TypeError, ValueError):
            continue

        if not isinstance(readings, list):
            continue

        for reading in readings:
            if not isinstance(reading, dict):
                continue
            power_value = reading.get(reading_key)
            if power_value is None:
                continue
            try:
                power_float = float(power_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(power_float):
                continue
            grouped[tone_int][gain_float].append(power_float)

    return grouped


def linear_power_to_dbm(power_values: np.ndarray, reading_key: str) -> np.ndarray:
    values = np.asarray(power_values, dtype=float)
    dbm = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0.0

    if reading_key == "scope_power_w":
        dbm[positive] = 10.0 * np.log10(values[positive]) + 30.0
        return dbm

    if reading_key == "pwr_pw":
        dbm[positive] = 10.0 * np.log10(values[positive]) - 90.0
        return dbm

    raise ValueError(f"Unsupported reading key for dBm conversion: {reading_key!r}")


def mean_linear_to_watts(mean_linear_value: float, reading_key: str) -> float:
    if reading_key == "scope_power_w":
        return mean_linear_value
    if reading_key == "pwr_pw":
        return mean_linear_value * 1e-12
    raise ValueError(f"Unsupported reading key for watt conversion: {reading_key!r}")


def build_calibration_rows(
    grouped: dict[int, dict[float, list[float]]],
    reading_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for tone, gain_map in sorted(grouped.items()):
        for gain in sorted(gain_map):
            power_values = np.asarray(gain_map[float(gain)], dtype=float)
            if power_values.size == 0:
                continue
            mean_linear = float(np.mean(power_values))
            mean_w = mean_linear_to_watts(mean_linear, reading_key=reading_key)
            mean_pw = mean_w * 1e12
            p_rf_dbm = float(10.0 * np.log10(mean_w) + 30.0) if mean_w > 0 else float("nan")
            rows.append(
                {
                    "tone": int(tone),
                    "tx_gain_db": float(gain),
                    "p_rf_dbm": p_rf_dbm,
                    "p_rf_w": mean_w,
                    "p_rf_pw": mean_pw,
                    "reading_count": int(power_values.size),
                }
            )

    return rows


def build_series(
    grouped: dict[int, dict[float, list[float]]],
    reading_key: str,
) -> dict[int, dict[str, np.ndarray]]:
    series: dict[int, dict[str, np.ndarray]] = {}

    for tone, gain_map in sorted(grouped.items()):
        gains = np.array(sorted(gain_map), dtype=float)
        mean_linear_values = []

        for gain in gains:
            power_values = np.asarray(gain_map[float(gain)], dtype=float)
            mean_linear_values.append(float(np.mean(power_values)))

        mean_linear = np.asarray(mean_linear_values, dtype=float)
        series[tone] = {
            "gains": gains,
            "mean_dbm": linear_power_to_dbm(mean_linear, reading_key=reading_key),
        }

    return series


def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Could not import matplotlib.pyplot. The local Matplotlib installation appears incompatible "
            "with the current NumPy version."
        ) from exc

    return plt


def default_output_name_for_input(input_path: Path) -> str:
    return f"{input_path.stem}_avg-power-vs-gain.png"


def default_csv_name_for_input(input_path: Path) -> str:
    return f"{input_path.stem}_tone_gain_prf.csv"


def write_calibration_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No calibration rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tone", "tx_gain_db", "p_rf_dbm", "p_rf_w", "p_rf_pw", "reading_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved calibration CSV to {path}")


def plot_series(series: dict[int, dict[str, np.ndarray]], output: Path):
    if not series:
        raise ValueError("No usable records found in the input file")

    plt = get_pyplot()
    fig, axis = plt.subplots(1, 1, figsize=(10, 6))
    color_map = plt.get_cmap("tab10")

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        color = color_map(index % color_map.N)
        axis.plot(
            tone_series["gains"],
            tone_series["mean_dbm"],
            color=color,
            marker="o",
            linewidth=1.8,
            label=f"Tone {tone}",
        )

    axis.set_title("Average measured power vs configured TX gain")
    axis.set_xlabel("Configured TX gain (dB)")
    axis.set_ylabel("Measured power (dBm)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Tone")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output}")
    return plt, fig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot average measured scope power versus configured TX gain for each tone"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Measurement file (.json or .jsonl). "
            "If omitted, uses the newest '*meas-tone-power-scope*' file."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every '*meas-tone-power-scope*' file found under --results-dir, newest first",
    )
    parser.add_argument(
        "--reading-key",
        default="auto",
        choices=READING_KEY_CHOICES,
        help="Reading field to aggregate (default: auto, preferring scope_power_w over pwr_pw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path for a single input file "
            "(default: results/<run-id>/plots/<input>_avg-power-vs-gain.png). Not allowed with --all."
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Calibration CSV output path for a single input file "
            "(default: results/<run-id>/tables/<input>_tone_gain_prf.csv). Not allowed with --all."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results root directory (default: results)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. If omitted, inferred from input path, else manual_<timestamp>.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure(s) without opening an interactive window",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.all and args.input is not None:
        raise ValueError("Do not pass an input file together with --all")
    if args.all and args.output is not None:
        raise ValueError("--output cannot be used with --all because each input file gets its own output name")
    if args.all and args.csv_output is not None:
        raise ValueError("--csv-output cannot be used with --all because each input file gets its own CSV output")

    results_dir = resolve_results_dir(args.results_dir)
    input_paths = resolve_input_paths(args.input, include_all=args.all, search_dir=results_dir)
    all_figures = []
    plt_module = None

    for input_path in input_paths:
        run_id = args.run_id or infer_run_id_from_path(input_path) or manual_run_id()
        output_path = resolve_output_path(
            args.output,
            results_dir=results_dir,
            run_id=run_id,
            bucket="plots",
            default_name=default_output_name_for_input(input_path),
        )
        csv_output_path = resolve_output_path(
            args.csv_output,
            results_dir=results_dir,
            run_id=run_id,
            bucket="tables",
            default_name=default_csv_name_for_input(input_path),
        )
        print(f"Processing {input_path}")
        records = load_records(input_path)
        reading_key = choose_reading_key(records, requested_key=args.reading_key)
        grouped = group_power_by_tone_gain(records, reading_key=reading_key)
        series = build_series(grouped, reading_key=reading_key)
        calibration_rows = build_calibration_rows(grouped, reading_key=reading_key)
        plt_module, figure = plot_series(series, output=output_path)
        write_calibration_csv(csv_output_path, rows=calibration_rows)
        all_figures.append(figure)

        manifest_path = write_manifest(
            results_dir=results_dir,
            run_id=run_id,
            script_name=Path(__file__).name,
            argv=sys.argv[1:],
            extra={
                "input": str(input_path),
                "output": str(output_path),
                "csv_output": str(csv_output_path),
                "reading_key": reading_key,
            },
        )
        print(f"Updated run manifest: {manifest_path}")

    if plt_module is not None:
        if args.no_show:
            for figure in all_figures:
                plt_module.close(figure)
        else:
            plt_module.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
