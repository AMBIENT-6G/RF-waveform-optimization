#!/usr/bin/env python3
"""Plot power statistics versus configured gain for each tone."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PERCENTILES = (25.0, 75.0)
MEASUREMENT_STEM_SUFFIX = "meas-tones-power"
MEASUREMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
REFERENCE_CSV = Path(__file__).with_name("harvester-chart-data.csv")


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


def gain_to_input_power_dbm(gain_db: np.ndarray | float) -> np.ndarray:
    values = np.asarray(gain_db, dtype=float)
    return values - 80.0


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


def discover_measurement_files(search_dir: Path = Path(".")) -> list[Path]:
    candidates = [path.resolve() for path in search_dir.iterdir() if is_measurement_file(path)]
    return sorted(candidates, key=measurement_sort_key, reverse=True)


def resolve_input_paths(input_path: Path | None, include_all: bool) -> list[Path]:
    if include_all:
        matches = discover_measurement_files()
        if not matches:
            raise FileNotFoundError(
                f"No measurement files found in {Path('.').resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
            )
        return matches

    if input_path is not None:
        resolved = input_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [resolved]

    matches = discover_measurement_files()
    if not matches:
        raise FileNotFoundError(
            f"No measurement files found in {Path('.').resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
        )
    return [matches[0]]


def default_output_path_for_input(input_path: Path) -> Path:
    return input_path.with_suffix(".png")


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
) -> dict[int, dict[str, np.ndarray]]:
    series: dict[int, dict[str, np.ndarray]] = {}

    for tone, gain_map in sorted(grouped.items()):
        gains = np.array(sorted(gain_map), dtype=float)
        mean_values = []
        percentile_a_values = []
        percentile_b_values = []

        for gain in gains:
            power_values = np.asarray(gain_map[float(gain)], dtype=float)
            mean_values.append(float(np.mean(power_values)))
            percentile_a_values.append(float(np.percentile(power_values, percentiles[0])))
            percentile_b_values.append(float(np.percentile(power_values, percentiles[1])))

        series[tone] = {
            "gains": gains,
            "mean": np.asarray(mean_values, dtype=float),
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


def plot_series(
    series: dict[int, dict[str, np.ndarray]],
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
        axis.plot(
            tone_series["gains"],
            mean_values,
            color=color,
            linewidth=2,
            linestyle="-",
            label=f"Tone {tone}",
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
    series: dict[int, dict[str, np.ndarray]],
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
        axis.plot(
            tone_series["gains"],
            mean_values,
            color=color,
            marker="o",
            linestyle="None",
            label=f"Tone {tone}",
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
    series: dict[int, dict[str, np.ndarray]],
    percentiles: tuple[float, float],
    power_key: str,
    output: Path | None,
):
    if not series:
        raise ValueError("No usable records found in the input file")
    if power_key != "pwr_pw":
        print(f"Skipping efficiency plot: supported only for power_key='pwr_pw', got {power_key!r}")
        return None, None

    plt, fig, axis = make_axes()
    color_map = plt.get_cmap("tab10")
    label_a = f"P{percentiles[0]:g}"
    label_b = f"P{percentiles[1]:g}"

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        input_power_dbm = gain_to_input_power_dbm(tone_series["gains"])
        input_power_w = dbm_to_watts(input_power_dbm)
        output_power_w = pw_to_watts(tone_series["mean"])
        percentile_a_power_w = pw_to_watts(tone_series[f"p{percentiles[0]:g}"])
        percentile_b_power_w = pw_to_watts(tone_series[f"p{percentiles[1]:g}"])
        efficiency_pct = np.divide(
            output_power_w,
            input_power_w,
            out=np.full_like(output_power_w, np.nan, dtype=float),
            where=input_power_w > 0,
        ) * 100.0
        percentile_a_efficiency_pct = np.divide(
            percentile_a_power_w,
            input_power_w,
            out=np.full_like(percentile_a_power_w, np.nan, dtype=float),
            where=input_power_w > 0,
        ) * 100.0
        percentile_b_efficiency_pct = np.divide(
            percentile_b_power_w,
            input_power_w,
            out=np.full_like(percentile_b_power_w, np.nan, dtype=float),
            where=input_power_w > 0,
        ) * 100.0
        order = np.argsort(input_power_dbm)
        input_power_dbm = input_power_dbm[order]
        efficiency_pct = efficiency_pct[order]
        percentile_a_efficiency_pct = percentile_a_efficiency_pct[order]
        percentile_b_efficiency_pct = percentile_b_efficiency_pct[order]
        color = color_map(index % color_map.N)
        axis.plot(
            input_power_dbm,
            efficiency_pct,
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=f"Tone {tone}",
        )
        axis.fill_between(
            input_power_dbm,
            np.minimum(percentile_a_efficiency_pct, efficiency_pct),
            np.maximum(percentile_a_efficiency_pct, efficiency_pct),
            color=color,
            alpha=0.5,
            linewidth=0,
        )
        axis.fill_between(
            input_power_dbm,
            np.minimum(efficiency_pct, percentile_b_efficiency_pct),
            np.maximum(efficiency_pct, percentile_b_efficiency_pct),
            color=color,
            alpha=0.5,
            linewidth=0,
        )

    reference = load_reference_efficiency(REFERENCE_CSV)
    if reference is not None:
        ref_level_dbm, ref_efficiency = reference
        axis.plot(
            ref_level_dbm,
            ref_efficiency,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Measured",
            zorder=5,
        )

    fig.suptitle("Efficiency vs inferred input power")
    axis.set_xlabel("Input RF power Pin (dBm)")
    axis.set_ylabel("Efficiency Pout/Pin (%)")
    axis.grid(True, alpha=0.3)
    axis.legend(title=f"Line: Average | Fill: {label_a}/{label_b}")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def plot_input_output_mw_markers(
    series: dict[int, dict[str, np.ndarray]],
    power_key: str,
    output: Path | None,
):
    if not series:
        raise ValueError("No usable records found in the input file")
    if power_key != "pwr_pw":
        print(f"Skipping input/output marker plot: supported only for power_key='pwr_pw', got {power_key!r}")
        return None, None

    plt, fig, axis = make_axes()
    color_map = plt.get_cmap("tab10")

    for index, tone in enumerate(sorted(series)):
        tone_series = series[tone]
        input_power_dbm = gain_to_input_power_dbm(tone_series["gains"])
        input_power_mw = watts_to_mw(dbm_to_watts(input_power_dbm))
        output_power_mw = watts_to_mw(pw_to_watts(tone_series["mean"]))
        order = np.argsort(input_power_mw)
        color = color_map(index % color_map.N)
        axis.plot(
            input_power_mw[order],
            output_power_mw[order],
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label=f"Tone {tone}",
        )

    fig.suptitle("Output vs input power (linear mW)")
    axis.set_xlabel("Input RF power Pin (mW)")
    axis.set_ylabel("Output DC power Pout (mW)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Markers: Average")
    fig.tight_layout()
    save_figure(fig, output)
    return plt, fig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot power statistics versus configured gain for each tone")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Measurement file (.json, .jsonl, or mapping-style JSON). If omitted, uses the newest '*meas-tones-power*' file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every '*meas-tones-power*' measurement file in the current directory, newest first",
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

    input_paths = resolve_input_paths(args.input, include_all=args.all)
    all_figures = []
    plt_module = None

    for input_path in input_paths:
        output_path = args.output if args.output is not None else default_output_path_for_input(input_path)
        print(f"Processing {input_path}")
        records = load_records(input_path)
        grouped = group_power_by_tone_gain(records, power_key=args.power_key)
        series = build_series(grouped, percentiles=args.percentiles)
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
        )
        _, input_output_fig = plot_input_output_mw_markers(
            series,
            power_key=args.power_key,
            output=input_output_mw_output_path(output_path),
        )
        all_figures.append(band_fig)
        all_figures.append(marker_fig)
        if efficiency_fig is not None:
            all_figures.append(efficiency_fig)
        if input_output_fig is not None:
            all_figures.append(input_output_fig)

    if plt_module is not None:
        if args.no_show:
            for figure in all_figures:
                plt_module.close(figure)
        else:
            plt_module.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
