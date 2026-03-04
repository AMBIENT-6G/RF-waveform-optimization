#!/usr/bin/env python3
"""Plot power statistics versus configured gain for each tone."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PERCENTILES = (25.0, 75.0)


def parse_percentiles(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated percentiles, e.g. 25,75")

    percentiles = tuple(float(part) for part in parts)
    for percentile in percentiles:
        if not 0 <= percentile <= 100:
            raise argparse.ArgumentTypeError("Percentiles must be between 0 and 100")

    return percentiles


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


def make_axes(n_plots: int):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Could not import matplotlib.pyplot. The local Matplotlib installation appears incompatible "
            "with the current NumPy version."
        ) from exc

    n_cols = min(2, max(1, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    return fig, axes.reshape(-1)


def plot_series(
    series: dict[int, dict[str, np.ndarray]],
    percentiles: tuple[float, float],
    power_key: str,
    output: Path | None,
    show: bool,
) -> None:
    if not series:
        raise ValueError("No usable records found in the input file")

    fig, axes = make_axes(len(series))
    label_a = f"P{percentiles[0]:g}"
    label_b = f"P{percentiles[1]:g}"

    for axis, tone in zip(axes, sorted(series)):
        tone_series = series[tone]
        axis.plot(tone_series["gains"], tone_series["mean"], marker="o", linewidth=2, label="Average")
        axis.plot(tone_series["gains"], tone_series[f"p{percentiles[0]:g}"], marker="s", linewidth=2, label=label_a)
        axis.plot(tone_series["gains"], tone_series[f"p{percentiles[1]:g}"], marker="^", linewidth=2, label=label_b)
        axis.set_title(f"Tone {tone}")
        axis.set_xlabel("Configured gain (dB)")
        axis.set_ylabel(f"{power_key}")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes[len(series) :]:
        axis.remove()

    fig.suptitle(f"Power statistics vs configured gain ({power_key})")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot power statistics versus configured gain for each tone")
    parser.add_argument("input", type=Path, help="Measurement file (.json, .jsonl, or mapping-style JSON)")
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
        default=Path("meas-tones-power.png"),
        help="Output image path (default: meas-tones-power.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_records(args.input)
    grouped = group_power_by_tone_gain(records, power_key=args.power_key)
    series = build_series(grouped, percentiles=args.percentiles)
    plot_series(
        series,
        percentiles=args.percentiles,
        power_key=args.power_key,
        output=args.output,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
