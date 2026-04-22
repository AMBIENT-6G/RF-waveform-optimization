#!/usr/bin/env python3
"""Plot power statistics versus configured gain for each waveform."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_layout import infer_run_id_from_path, manual_run_id, resolve_output_path, write_manifest


DEFAULT_PERCENTILES = (25.0, 75.0)
BUFFER_VOLTAGE_KEY = "buffer_voltage_mv"
MEASUREMENT_STEM_SUFFIX = "meas-tones-power"
MEASUREMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = REPO_ROOT / "data" / "reference" / "harvester-chart-data.csv"
EXPECTED_EFFICIENCY_CSV = REPO_ROOT / "data" / "reference" / "expected-efficiency.csv"
DEFAULT_GAIN_MAP_CSV = REPO_ROOT / "data" / "gain-power-map.csv"
WEIGHTS_DIR = REPO_ROOT / "data" / "weights"
LEGACY_RF_CALIBRATION_REGEX = re.compile(r".*tone_gain_prf\.csv$", re.IGNORECASE)


@dataclass
class CalibrationLookup:
    global_by_gain: dict[float, float]
    by_tone: dict[int, dict[float, float]]


def waveform_label_for_tone(tone: int) -> str:
    if tone == 0:
        return "DC"
    if tone == 1:
        return "NB"
    return f"N={tone}"


def pw_to_dbm(power_pw: np.ndarray | float) -> np.ndarray:
    values = np.asarray(power_pw, dtype=float)
    dbm = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0
    dbm[positive] = 10.0 * np.log10(values[positive]) - 90.0
    return dbm


def pw_to_watts(power_pw: np.ndarray | float) -> np.ndarray:
    return np.asarray(power_pw, dtype=float) * 1e-12


def watts_to_mw(power_w: np.ndarray | float) -> np.ndarray:
    return np.asarray(power_w, dtype=float) * 1e3


def dbm_to_watts(power_dbm: np.ndarray | float) -> np.ndarray:
    values = np.asarray(power_dbm, dtype=float)
    return 1e-3 * np.power(10.0, values / 10.0)


def gain_to_input_power_dbm(
    gain_db: np.ndarray | float,
    calibration_by_gain: dict[float, float] | None = None,
    *,
    debug: bool = False,
    debug_label: str = "",
) -> np.ndarray:
    values = np.asarray(gain_db, dtype=float)

    if calibration_by_gain is None:
        if debug:
            print(
                f"DEBUG gain_to_input_power_dbm[{debug_label}]: "
                "no calibration_by_gain provided; using inferred Pin = gain - 80 dBm."
            )
        return values - 80.0

    calibrated = np.full(values.shape, np.nan, dtype=float)
    flat_gain = values.reshape(-1)
    flat_calibrated = calibrated.reshape(-1)

    if debug:
        print(
            f"DEBUG gain_to_input_power_dbm[{debug_label}] calibration_by_gain="
            f"{dict(sorted(calibration_by_gain.items()))}"
        )

    for index, gain in enumerate(flat_gain):
        mapped = calibration_by_gain.get(gain_key(float(gain)))
        if mapped is None:
            continue
        flat_calibrated[index] = float(mapped)

    if debug:
        debug_pairs = [
            (float(gain), float(power))
            for gain, power in zip(flat_gain.tolist(), flat_calibrated.tolist())
            if np.isfinite(power)
        ]
        print(f"DEBUG gain_to_input_power_dbm[{debug_label}] mapped_points={debug_pairs}")

    return calibrated


def gain_key(gain_db: float) -> float:
    return round(float(gain_db), 6)


def choose_csv_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {name.strip().lower(): name for name in fieldnames if isinstance(name, str)}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_rf_calibration_csv(path: Path) -> CalibrationLookup:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"RF calibration CSV has no header row: {path}")

        tone_col = choose_csv_column(fieldnames, ("tone",))
        gain_col = choose_csv_column(fieldnames, ("tx_gain_db", "gain_db", "configured_gain_db"))
        power_col = choose_csv_column(
            fieldnames,
            ("input_level_dbm", "p_rf_dbm", "prf_dbm", "input_power_dbm", "pin_dbm"),
        )

        missing = []
        if gain_col is None:
            missing.append("tx_gain_db/gain_db")
        if power_col is None:
            missing.append("input_level_dbm/p_rf_dbm")
        if missing:
            raise ValueError(
                f"RF calibration CSV missing required column(s): {', '.join(missing)} in {path}"
            )

        by_tone: dict[int, dict[float, float]] = defaultdict(dict)
        global_by_gain: dict[float, float] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                gain = float(row[gain_col])  # type: ignore[index]
                power_dbm = float(row[power_col])  # type: ignore[index]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid RF calibration row {line_number} in {path}"
                ) from exc

            if not np.isfinite(power_dbm):
                continue
            if tone_col is not None:
                raw_tone = row.get(tone_col)
                if raw_tone not in (None, ""):
                    tone = int(float(raw_tone))
                    by_tone[tone][gain_key(gain)] = power_dbm
                    continue
            global_by_gain[gain_key(gain)] = power_dbm

    if not by_tone and not global_by_gain:
        raise ValueError(f"No usable calibration points found in {path}")
    return CalibrationLookup(
        global_by_gain=dict(global_by_gain),
        by_tone={tone: dict(gain_map) for tone, gain_map in by_tone.items()},
    )


def discover_rf_calibration_files(search_dir: Path = Path("results")) -> list[Path]:
    candidates: list[Path] = []

    if DEFAULT_GAIN_MAP_CSV.exists():
        candidates.append(DEFAULT_GAIN_MAP_CSV.resolve())

    search_root = search_dir.resolve()
    if search_root.exists():
        candidates.extend(
            path.resolve()
            for path in search_root.rglob("*.csv")
            if LEGACY_RF_CALIBRATION_REGEX.match(path.name)
        )
    if search_root != REPO_ROOT:
        candidates.extend(
            path.resolve()
            for path in REPO_ROOT.rglob("*.csv")
            if LEGACY_RF_CALIBRATION_REGEX.match(path.name)
        )

    unique_candidates: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        if path not in unique_candidates:
            unique_candidates.append(path)
    return unique_candidates


def build_input_power_lookup(
    series: dict[int, dict[str, Any]],
    rf_calibration: CalibrationLookup | None,
    debug_calibration: bool = False,
) -> tuple[dict[int, np.ndarray], bool]:
    input_power_dbm_by_tone: dict[int, np.ndarray] = {}
    used_calibration = False

    for tone in sorted(series):
        gains = np.asarray(series[tone]["gains"], dtype=float)
        waveform_label = str(series[tone].get("label", waveform_label_for_tone(tone)))
        inferred = gain_to_input_power_dbm(gains)

        if rf_calibration is None:
            input_power_dbm_by_tone[tone] = inferred
            continue

        tone_map = rf_calibration.by_tone.get(int(tone))
        if not tone_map and rf_calibration.global_by_gain:
            tone_map = rf_calibration.global_by_gain
        if not tone_map:
            print(
                f"Warning: {waveform_label} missing in RF calibration CSV; using inferred Pin = gain - 80 dBm."
            )
            input_power_dbm_by_tone[tone] = inferred
            continue

        calibrated = gain_to_input_power_dbm(
            gains,
            calibration_by_gain=tone_map,
            debug=debug_calibration,
            debug_label=waveform_label,
        )
        matched = int(np.count_nonzero(np.isfinite(calibrated)))

        if matched == 0:
            print(
                f"Warning: {waveform_label} has no gain matches in RF calibration CSV; using inferred Pin = gain - 80 dBm."
            )
            input_power_dbm_by_tone[tone] = inferred
            continue

        if matched < gains.size:
            print(
                f"Warning: {waveform_label} missing {gains.size - matched} calibrated gain point(s); "
                "dropping those points from input-power plots."
            )

        input_power_dbm_by_tone[tone] = calibrated
        used_calibration = True

    return input_power_dbm_by_tone, used_calibration


def parse_percentiles(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated percentiles, e.g. 25,75")

    percentiles = tuple(float(part) for part in parts)
    for percentile in percentiles:
        if not 0 <= percentile <= 100:
            raise argparse.ArgumentTypeError("Percentiles must be between 0 and 100")

    return percentiles


def is_measurement_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and path.stem.endswith(MEASUREMENT_STEM_SUFFIX)


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


def discover_measurement_files(
    search_dir: Path = Path("results"),
    *,
    allow_repo_fallback: bool = True,
) -> list[Path]:
    candidates: list[Path] = []
    for search_root in candidate_search_roots(search_dir):
        if not search_root.exists():
            continue
        candidates.extend(path.resolve() for path in search_root.rglob("*") if is_measurement_file(path))

    if not candidates and allow_repo_fallback:
        candidates.extend(path.resolve() for path in REPO_ROOT.rglob("*") if is_measurement_file(path))

    unique_candidates: list[Path] = []
    for path in sorted(candidates, key=measurement_sort_key, reverse=True):
        if path not in unique_candidates:
            unique_candidates.append(path)
    return unique_candidates


def resolve_input_paths(
    input_path: Path | None,
    include_all: bool,
    search_dir: Path,
    *,
    allow_repo_fallback: bool = True,
) -> list[Path]:
    if include_all:
        matches = discover_measurement_files(
            search_dir=search_dir,
            allow_repo_fallback=allow_repo_fallback,
        )
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

    matches = discover_measurement_files(
        search_dir=search_dir,
        allow_repo_fallback=allow_repo_fallback,
    )
    if not matches:
        raise FileNotFoundError(
            f"No measurement files found in {search_dir.resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
        )
    return [matches[0]]


def resolve_input_search_dir(results_dir: Path, run_id: str | None, input_path: Path | None) -> tuple[Path, bool]:
    if run_id is None or input_path is not None:
        return results_dir, True
    return results_dir / run_id, False


def default_output_name_for_input(input_path: Path) -> str:
    return f"{input_path.stem}.png"


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


def extract_waveform_labels(records: list[dict[str, Any]]) -> dict[int, str]:
    labels: dict[int, str] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        tone = record.get("tone")
        try:
            tone_int = int(tone)
        except (TypeError, ValueError):
            continue

        label = record.get("waveform_label")
        if isinstance(label, str) and label.strip():
            labels[tone_int] = label.strip()
        else:
            labels.setdefault(tone_int, waveform_label_for_tone(tone_int))

    return labels


def extract_bandwidths(records: list[dict[str, Any]]) -> dict[int, int]:
    bandwidths: dict[int, int] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        tone = record.get("tone")
        bw_khz = record.get("bw_khz")
        try:
            tone_int = int(tone)
            bw_int = int(float(bw_khz))
        except (TypeError, ValueError):
            continue

        bandwidths.setdefault(tone_int, bw_int)

    return bandwidths


def parse_key_pair(key: str) -> tuple[int, float]:
    cleaned = key.strip().strip("()")
    parts = [item.strip() for item in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse mapping key {key!r}; expected '(tone, gain)'")
    return int(parts[0]), float(parts[1])


def group_power_by_tone_gain(
    records: list[dict[str, Any]],
    power_key: str,
) -> dict[int, dict[float, list[float]]]:
    grouped: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        if not isinstance(record, dict):
            continue

        tone = record.get("tone")
        gain = record.get("gain_db")
        readings = record.get("readings", [])
        if tone is None or gain is None or not isinstance(readings, list):
            continue

        for reading in readings:
            if not isinstance(reading, dict):
                continue
            power_value = reading.get(power_key)
            if power_value is None:
                continue
            grouped[int(tone)][float(gain)].append(float(power_value))

    return grouped


def build_series(
    grouped: dict[int, dict[float, list[float]]],
    percentiles: tuple[float, float],
    waveform_labels: dict[int, str] | None = None,
) -> dict[int, dict[str, Any]]:
    series: dict[int, dict[str, Any]] = {}

    for tone, gain_map in sorted(grouped.items()):
        gains = np.array(sorted(gain_map), dtype=float)
        mean_values = []
        min_values = []
        max_values = []
        percentile_a_values = []
        percentile_b_values = []

        for gain in gains:
            power_values = np.asarray(gain_map[float(gain)], dtype=float)
            mean_values.append(float(np.mean(power_values)))
            min_values.append(float(np.min(power_values)))
            max_values.append(float(np.max(power_values)))
            percentile_a_values.append(float(np.percentile(power_values, percentiles[0])))
            percentile_b_values.append(float(np.percentile(power_values, percentiles[1])))

        series[tone] = {
            "label": (
                waveform_labels.get(tone, waveform_label_for_tone(tone))
                if waveform_labels is not None
                else waveform_label_for_tone(tone)
            ),
            "gains": gains,
            "mean": np.asarray(mean_values, dtype=float),
            "min": np.asarray(min_values, dtype=float),
            "max": np.asarray(max_values, dtype=float),
            f"p{percentiles[0]:g}": np.asarray(percentile_a_values, dtype=float),
            f"p{percentiles[1]:g}": np.asarray(percentile_b_values, dtype=float),
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


def make_axes():
    plt = get_pyplot()
    fig, axis = plt.subplots(1, 1, figsize=(10, 6))
    return plt, fig, axis


def save_figure(fig, output: Path | None) -> None:
    if output is None:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output}")


def marker_output_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    return output.with_name(f"{output.stem}_avg_markers{output.suffix}")


def efficiency_output_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    return output.with_name(f"{output.stem}_efficiency{output.suffix}")


def input_output_mw_output_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    return output.with_name(f"{output.stem}_input_output_mw_markers{output.suffix}")


def buffer_voltage_output_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    return output.with_name(f"{output.stem}_buffer_voltage{output.suffix}")


def load_reference_efficiency(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None

    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception:
        return None

    if data.size == 0:
        return None

    names = tuple(data.dtype.names or ())
    required = {"level_dbm", "efficiency"}
    if not required.issubset(names):
        return None

    level = np.atleast_1d(np.asarray(data["level_dbm"], dtype=float))
    efficiency = np.atleast_1d(np.asarray(data["efficiency"], dtype=float))
    order = np.argsort(level)
    return level[order], efficiency[order]


def resolve_weights_file(tone: int, bw_khz: int | None) -> Path | None:
    if tone < 2:
        return None

    if bw_khz is not None:
        candidate = WEIGHTS_DIR / f"weightsN{tone}BW{bw_khz}.mat"
        if candidate.exists():
            return candidate

    matches = sorted(WEIGHTS_DIR.glob(f"weightsN{tone}BW*.mat"))
    if len(matches) == 1:
        return matches[0]
    return None


def load_expected_efficiency_from_mat(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None

    try:
        import h5py
    except Exception:
        return None

    try:
        with h5py.File(path, "r") as handle:
            required = {"Pdc", "inPwrVec"}
            if not required.issubset(handle.keys()):
                return None

            pdc = np.atleast_1d(np.asarray(handle["Pdc"][()], dtype=float).squeeze())
            input_power_mw = np.atleast_1d(np.asarray(handle["inPwrVec"][()], dtype=float).squeeze())
    except Exception:
        return None

    return compute_expected_efficiency(input_power_mw=input_power_mw, output_power_mw=pdc)


def compute_expected_efficiency(
    input_power_mw: np.ndarray,
    output_power_mw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if output_power_mw.size != input_power_mw.size:
        return None

    input_power_dbm = np.full(input_power_mw.shape, np.nan, dtype=float)
    positive = input_power_mw > 0
    input_power_dbm[positive] = 10.0 * np.log10(input_power_mw[positive])
    efficiency_pct = np.divide(
        output_power_mw,
        input_power_mw,
        out=np.full_like(output_power_mw, np.nan, dtype=float),
        where=input_power_mw > 0,
    ) * 100.0

    valid = np.isfinite(input_power_dbm) & np.isfinite(efficiency_pct)
    if not np.any(valid):
        return None

    input_power_dbm = input_power_dbm[valid]
    efficiency_pct = efficiency_pct[valid]
    order = np.argsort(input_power_dbm)
    return input_power_dbm[order], efficiency_pct[order]


def load_expected_efficiency_from_csv(
    path: Path,
    tone: int,
    bw_khz: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None

    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception:
        return None

    if data.size == 0:
        return None

    names = tuple(data.dtype.names or ())
    required = {"tone", "bw_khz", "level_dbm", "efficiency"}
    if not required.issubset(names):
        return None

    tones = np.atleast_1d(np.asarray(data["tone"], dtype=int))
    bandwidths = np.atleast_1d(np.asarray(data["bw_khz"], dtype=int))
    levels = np.atleast_1d(np.asarray(data["level_dbm"], dtype=float))
    efficiencies = np.atleast_1d(np.asarray(data["efficiency"], dtype=float))

    mask = tones == int(tone)
    if bw_khz is not None:
        mask &= bandwidths == int(bw_khz)
    elif np.any(mask):
        unique_bandwidths = np.unique(bandwidths[mask])
        if unique_bandwidths.size == 1:
            mask &= bandwidths == int(unique_bandwidths[0])

    if not np.any(mask):
        return None

    levels = levels[mask]
    efficiencies = efficiencies[mask]
    valid = np.isfinite(levels) & np.isfinite(efficiencies)
    if not np.any(valid):
        return None

    levels = levels[valid]
    efficiencies = efficiencies[valid]
    order = np.argsort(levels)
    return levels[order], efficiencies[order]


def load_expected_efficiency(
    tone: int,
    bw_khz: int | None,
    weights_path: Path | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if weights_path is not None:
        expected = load_expected_efficiency_from_mat(weights_path)
        if expected is not None:
            return expected

    return load_expected_efficiency_from_csv(EXPECTED_EFFICIENCY_CSV, tone=tone, bw_khz=bw_khz)


def plot_series(
    series: dict[int, dict[str, Any]],
    percentiles: tuple[float, float],
    power_key: str,
    output: Path | None,
):
    if not series:
        raise ValueError("No usable records found in the input file")

    plt, fig, axis = make_axes()
    label_a = f"P{percentiles[0]:g}"
    label_b = f"P{percentiles[1]:g}"
    use_dbm = power_key == "pwr_pw"
    y_label = "Power (dBm)" if use_dbm else power_key
    title_suffix = "Power statistics vs configured gain (dBm)" if use_dbm else f"Power statistics vs configured gain ({power_key})"
    color_map = plt.get_cmap("tab10")

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        mean_values = pw_to_dbm(tone_series["mean"]) if use_dbm else tone_series["mean"]
        percentile_a_values = (
            pw_to_dbm(tone_series[f"p{percentiles[0]:g}"])
            if use_dbm
            else tone_series[f"p{percentiles[0]:g}"]
        )
        percentile_b_values = (
            pw_to_dbm(tone_series[f"p{percentiles[1]:g}"])
            if use_dbm
            else tone_series[f"p{percentiles[1]:g}"]
        )
        color = color_map(index % color_map.N)
        waveform_label = str(tone_series.get("label", waveform_label_for_tone(tone)))
        axis.plot(
            tone_series["gains"],
            mean_values,
            color=color,
            linewidth=2,
            linestyle="-",
            label=waveform_label,
        )
        axis.plot(
            tone_series["gains"],
            percentile_a_values,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=None,
        )
        axis.plot(
            tone_series["gains"],
            percentile_b_values,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=None,
        )

    fig.suptitle(title_suffix)
    axis.set_xlabel("Configured gain (dB)")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend(title=f"Solid: Average | Dashed: {label_a}/{label_b}")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def plot_average_markers(
    series: dict[int, dict[str, Any]],
    power_key: str,
    output: Path | None,
):
    if not series:
        raise ValueError("No usable records found in the input file")

    plt, fig, axis = make_axes()
    use_dbm = power_key == "pwr_pw"
    y_label = "Power (dBm)" if use_dbm else power_key
    title_suffix = "Average power vs configured gain (markers)" if use_dbm else f"Average {power_key} vs configured gain (markers)"
    color_map = plt.get_cmap("tab10")

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        mean_values = pw_to_dbm(tone_series["mean"]) if use_dbm else tone_series["mean"]
        color = color_map(index % color_map.N)
        waveform_label = str(tone_series.get("label", waveform_label_for_tone(tone)))
        axis.plot(
            tone_series["gains"],
            mean_values,
            color=color,
            marker="o",
            linestyle="None",
            label=waveform_label,
        )

    fig.suptitle(title_suffix)
    axis.set_xlabel("Configured gain (dB)")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Markers: Average")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def plot_efficiency_markers(
    series: dict[int, dict[str, Any]],
    percentiles: tuple[float, float],
    power_key: str,
    output: Path | None,
    input_power_dbm_by_tone: dict[int, np.ndarray] | None = None,
    bandwidth_khz_by_tone: dict[int, int] | None = None,
    use_calibrated_input: bool = False,
):
    if not series:
        raise ValueError("No usable records found in the input file")
    if power_key != "pwr_pw":
        print(f"Skipping efficiency plot: supported only for power_key='pwr_pw', got {power_key!r}")
        return None, None

    plt, fig, axis = make_axes()
    color_map = plt.get_cmap("tab10")
    _ = percentiles
    plotted_expected = False
    plotted_measured = False

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        waveform_label = str(tone_series.get("label", waveform_label_for_tone(tone)))
        if input_power_dbm_by_tone is not None and tone in input_power_dbm_by_tone:
            input_power_dbm = np.asarray(input_power_dbm_by_tone[tone], dtype=float)
        else:
            input_power_dbm = gain_to_input_power_dbm(tone_series["gains"])
        input_power_w = dbm_to_watts(input_power_dbm)
        output_power_w = pw_to_watts(tone_series["mean"])
        efficiency_pct = np.divide(
            output_power_w,
            input_power_w,
            out=np.full_like(output_power_w, np.nan, dtype=float),
            where=input_power_w > 0,
        ) * 100.0
        valid = np.isfinite(input_power_dbm) & np.isfinite(efficiency_pct)
        if not np.any(valid):
            print(f"Warning: skipping {waveform_label} in efficiency plot (no valid input-power points).")
            continue
        input_power_dbm = input_power_dbm[valid]
        efficiency_pct = efficiency_pct[valid]
        order = np.argsort(input_power_dbm)
        input_power_dbm = input_power_dbm[order]
        efficiency_pct = efficiency_pct[order]
        color = color_map(index % color_map.N)
        axis.plot(
            input_power_dbm,
            efficiency_pct,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
        axis.scatter(
            input_power_dbm,
            efficiency_pct,
            facecolors="none",
            edgecolors=color,
            marker="o",
            linewidths=1.5,
            label=waveform_label,
            zorder=3,
        )
        plotted_measured = True

        weights_path = resolve_weights_file(
            tone,
            None if bandwidth_khz_by_tone is None else bandwidth_khz_by_tone.get(tone),
        )
        expected = load_expected_efficiency(
            tone=tone,
            bw_khz=None if bandwidth_khz_by_tone is None else bandwidth_khz_by_tone.get(tone),
            weights_path=weights_path,
        )
        if expected is None:
            source_hint = str(weights_path) if weights_path is not None else str(EXPECTED_EFFICIENCY_CSV)
            print(f"Warning: could not load expected efficiency for {waveform_label} from {source_hint}")
            continue

        expected_input_dbm, expected_efficiency_pct = expected
        axis.scatter(
            expected_input_dbm,
            expected_efficiency_pct,
            marker="x",
            color=color,
            linewidths=2.5,
            s=90,
            zorder=4,
        )
        plotted_expected = True

    reference = load_reference_efficiency(REFERENCE_CSV)
    if reference is not None:
        ref_level_dbm, ref_efficiency = reference
        axis.plot(
            ref_level_dbm,
            ref_efficiency,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Narrowband ref",
            zorder=5,
        )
    if plotted_expected:
        axis.plot(
            [],
            [],
            color="black",
            marker="x",
            linestyle="None",
            markeredgewidth=2,
            label="Expected",
        )
    else:
        print(
            "Warning: no expected efficiency overlays were plotted. "
            f"Checked MAT weights under {WEIGHTS_DIR} and CSV fallback {EXPECTED_EFFICIENCY_CSV}."
        )

    title_prefix = "calibrated" if use_calibrated_input else "inferred"
    fig.suptitle(f"Efficiency vs {title_prefix} input power")
    axis.set_xlabel("Input RF power Pin (dBm)")
    axis.set_ylabel("Efficiency Pout/Pin (%)")
    axis.grid(True, alpha=0.3)
    legend_title = "o: Measured"
    if plotted_expected:
        legend_title += " | x: Expected"
    if not plotted_measured:
        legend_title = "x: Expected" if plotted_expected else "Waveform"
    axis.legend(title=legend_title)
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def plot_input_output_mw_markers(
    series: dict[int, dict[str, Any]],
    power_key: str,
    output: Path | None,
    input_power_dbm_by_tone: dict[int, np.ndarray] | None = None,
    use_calibrated_input: bool = False,
):
    if not series:
        raise ValueError("No usable records found in the input file")
    if power_key != "pwr_pw":
        print(f"Skipping input/output marker plot: supported only for power_key='pwr_pw', got {power_key!r}")
        return None, None

    plt, fig, axis = make_axes()
    color_map = plt.get_cmap("tab10")
    plotted = False

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        waveform_label = str(tone_series.get("label", waveform_label_for_tone(tone)))
        if input_power_dbm_by_tone is not None and tone in input_power_dbm_by_tone:
            input_power_dbm = np.asarray(input_power_dbm_by_tone[tone], dtype=float)
        else:
            input_power_dbm = gain_to_input_power_dbm(tone_series["gains"])
        input_power_mw = watts_to_mw(dbm_to_watts(input_power_dbm))
        output_power_mw = watts_to_mw(pw_to_watts(tone_series["mean"]))
        valid = np.isfinite(input_power_mw) & np.isfinite(output_power_mw) & (input_power_mw > 0)
        if not np.any(valid):
            print(f"Warning: skipping {waveform_label} in input/output plot (no valid input-power points).")
            continue
        input_power_mw = input_power_mw[valid]
        output_power_mw = output_power_mw[valid]
        order = np.argsort(input_power_mw)
        sorted_input_mw = input_power_mw[order]
        sorted_output_mw = output_power_mw[order]
        color = color_map(index % color_map.N)
        axis.plot(
            sorted_input_mw,
            sorted_output_mw,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label=f"{waveform_label} data",
        )

    title_prefix = "calibrated" if use_calibrated_input else "inferred"
    fig.suptitle(f"Output vs {title_prefix} input power (linear mW)")
    axis.set_xlabel("Input RF power Pin (mW)")
    axis.set_ylabel("Output DC power Pout (mW)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Data")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def plot_buffer_voltage_minmax_avg(
    series: dict[int, dict[str, Any]],
    output: Path | None,
):
    if not series:
        print(f"Skipping buffer voltage plot: no usable {BUFFER_VOLTAGE_KEY!r} readings found.")
        return None, None

    plt, fig, axis = make_axes()
    color_map = plt.get_cmap("tab10")

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        gains = np.asarray(tone_series["gains"], dtype=float)
        min_values = np.asarray(tone_series["min"], dtype=float)
        mean_values = np.asarray(tone_series["mean"], dtype=float)
        max_values = np.asarray(tone_series["max"], dtype=float)
        valid = (
            np.isfinite(gains)
            & np.isfinite(min_values)
            & np.isfinite(mean_values)
            & np.isfinite(max_values)
        )
        if not np.any(valid):
            continue

        gains = gains[valid]
        min_values = min_values[valid]
        mean_values = mean_values[valid]
        max_values = max_values[valid]
        order = np.argsort(gains)
        gains = gains[order]
        min_values = min_values[order]
        mean_values = mean_values[order]
        max_values = max_values[order]

        color = color_map(index % color_map.N)
        waveform_label = str(tone_series.get("label", waveform_label_for_tone(tone)))
        axis.fill_between(
            gains,
            min_values,
            max_values,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        axis.plot(
            gains,
            mean_values,
            color=color,
            marker="o",
            linewidth=1.8,
            label=waveform_label,
        )
        plotted = True

    if not plotted:
        print(f"Skipping buffer voltage plot: no finite {BUFFER_VOLTAGE_KEY!r} readings found.")
        return plt, fig

    fig.suptitle("Buffer voltage statistics vs configured gain")
    axis.set_xlabel("Configured gain (dB)")
    axis.set_ylabel("Buffer voltage (mV)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Line: average | Shaded: min-max")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot power statistics versus configured gain for each waveform")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Measurement file (.json, .jsonl, or mapping-style JSON). If omitted, uses the newest "
            "'*meas-tones-power*' file, scoped to --run-id when provided."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every matching '*meas-tones-power*' measurement file, scoped to --run-id when provided.",
    )
    parser.add_argument(
        "--power-key",
        default="pwr_pw",
        help="Reading field to aggregate and plot (default: pwr_pw)",
    )
    parser.add_argument(
        "--percentiles",
        type=parse_percentiles,
        default=DEFAULT_PERCENTILES,
        help="Two comma-separated percentiles to plot (default: 25,75)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for a single input file (default: <input>.png). Not allowed with --all.",
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
        help="Run identifier. With no input path, also searches inputs under results/<run-id>/.",
    )
    parser.add_argument(
        "--rf-calibration-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV used to calibrate input-power x-axes instead of Pin = gain - 80 dBm. "
            "Supports the current data/gain-power-map.csv format (gain_db,input_level_dbm) "
            "and legacy per-tone CSVs such as *tone_gain_prf.csv*."
        ),
    )
    parser.add_argument(
        "--debug-calibration",
        action="store_true",
        help="Print DEBUG output for calibration_by_gain mapping and matched points per tone.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.all and args.input is not None:
        raise ValueError("Do not pass an input file together with --all")
    if args.all and args.output is not None:
        raise ValueError("--output cannot be used with --all because each input file gets its own output name")

    results_dir = resolve_results_dir(args.results_dir)
    input_search_dir, allow_repo_fallback = resolve_input_search_dir(
        results_dir,
        args.run_id,
        args.input,
    )
    if args.input is None and args.run_id is not None:
        print(f"Searching measurement inputs in {input_search_dir}")
    input_paths = resolve_input_paths(
        args.input,
        include_all=args.all,
        search_dir=input_search_dir,
        allow_repo_fallback=allow_repo_fallback,
    )
    all_figures = []
    plt_module = None
    rf_calibration = None
    rf_calibration_path: Path | None = None
    if args.rf_calibration_csv is not None:
        rf_calibration_path = args.rf_calibration_csv.resolve()
        if not rf_calibration_path.exists():
            raise FileNotFoundError(f"RF calibration CSV not found: {args.rf_calibration_csv}")
    else:
        discovered = discover_rf_calibration_files(results_dir)
        if discovered:
            rf_calibration_path = discovered[0]
            print(f"Auto-discovered RF calibration CSV: {rf_calibration_path}")

    if rf_calibration_path is not None:
        rf_calibration = load_rf_calibration_csv(rf_calibration_path)
        print(f"Loaded RF calibration CSV: {rf_calibration_path}")
    else:
        print(
            "No RF calibration CSV provided or discovered; "
            "using inferred Pin = gain - 80 dBm."
        )

    for input_path in input_paths:
        run_id = args.run_id or infer_run_id_from_path(input_path) or manual_run_id()
        output_path = resolve_output_path(
            args.output,
            results_dir=results_dir,
            run_id=run_id,
            bucket="plots",
            default_name=default_output_name_for_input(input_path),
        )
        print(f"Processing {input_path}")
        records = load_records(input_path)
        waveform_labels = extract_waveform_labels(records)
        bandwidth_khz_by_tone = extract_bandwidths(records)
        grouped = group_power_by_tone_gain(records, power_key=args.power_key)
        series = build_series(grouped, percentiles=args.percentiles, waveform_labels=waveform_labels)
        voltage_grouped = group_power_by_tone_gain(records, power_key=BUFFER_VOLTAGE_KEY)
        voltage_series = build_series(
            voltage_grouped,
            percentiles=args.percentiles,
            waveform_labels=waveform_labels,
        )
        input_power_dbm_by_tone, used_calibration = build_input_power_lookup(
            series,
            rf_calibration=rf_calibration,
            debug_calibration=args.debug_calibration,
        )
        plt_module, band_fig = plot_series(
            series,
            percentiles=args.percentiles,
            power_key=args.power_key,
            output=output_path,
        )
        _, marker_fig = plot_average_markers(
            series,
            power_key=args.power_key,
            output=marker_output_path(output_path),
        )
        _, efficiency_fig = plot_efficiency_markers(
            series,
            percentiles=args.percentiles,
            power_key=args.power_key,
            output=efficiency_output_path(output_path),
            input_power_dbm_by_tone=input_power_dbm_by_tone,
            bandwidth_khz_by_tone=bandwidth_khz_by_tone,
            use_calibrated_input=used_calibration,
        )
        _, input_output_fig = plot_input_output_mw_markers(
            series,
            power_key=args.power_key,
            output=input_output_mw_output_path(output_path),
            input_power_dbm_by_tone=input_power_dbm_by_tone,
            use_calibrated_input=used_calibration,
        )
        _, buffer_voltage_fig = plot_buffer_voltage_minmax_avg(
            voltage_series,
            output=buffer_voltage_output_path(output_path),
        )
        all_figures.append(band_fig)
        all_figures.append(marker_fig)
        if efficiency_fig is not None:
            all_figures.append(efficiency_fig)
        if input_output_fig is not None:
            all_figures.append(input_output_fig)
        if buffer_voltage_fig is not None:
            all_figures.append(buffer_voltage_fig)

        manifest_path = write_manifest(
            results_dir=results_dir,
            run_id=run_id,
            script_name=Path(__file__).name,
            argv=sys.argv[1:],
            extra={
                "input": str(input_path),
                "output": str(output_path),
                "rf_calibration_csv": (
                    str(rf_calibration_path) if rf_calibration_path is not None else None
                ),
                "buffer_voltage_output": str(buffer_voltage_output_path(output_path)),
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
