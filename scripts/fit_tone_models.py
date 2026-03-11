#!/usr/bin/env python3
"""Read tone measurement data, extract Prf/Pdc, fit models, and plot results.

This is a standalone script that replaces the split flow from:
- model definitions
- fitting logic
- plotting logic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from run_layout import (
    infer_run_id_from_path,
    manual_run_id,
    resolve_output_path,
    write_manifest,
)


EPS = 1e-12
MEASUREMENT_STEM_SUFFIX = "meas-tones-power"
MEASUREMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
REPO_ROOT = Path(__file__).resolve().parents[1]


def get_curve_fit():
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        raise RuntimeError(
            "scipy.optimize.curve_fit is required. Install SciPy first, e.g. `pip install scipy`."
        ) from exc
    return curve_fit


def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Could not import matplotlib.pyplot. Install/repair Matplotlib in this environment."
        ) from exc
    return plt


def dbm_to_mw(values_dbm: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values_dbm, dtype=float)
    return np.power(10.0, values / 10.0)


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


def discover_measurement_files(search_dir: Path = Path("results")) -> list[Path]:
    search_root = search_dir.resolve()
    if not search_root.exists():
        return []
    candidates = [path.resolve() for path in search_root.rglob("*") if is_measurement_file(path)]
    if not candidates and search_root != REPO_ROOT:
        candidates = [path.resolve() for path in REPO_ROOT.rglob("*") if is_measurement_file(path)]
    return sorted(candidates, key=measurement_sort_key, reverse=True)


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        resolved = input_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return resolved
    matches = discover_measurement_files()
    if not matches:
        raise FileNotFoundError(
            f"No measurement files found in {Path('.').resolve()} matching '*{MEASUREMENT_STEM_SUFFIX}*.jsonl'"
        )
    return matches[0]


def parse_key_pair(key: str) -> tuple[int, float]:
    cleaned = key.strip().strip("()")
    parts = [item.strip() for item in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse mapping key {key!r}; expected '(tone, gain)'")
    return int(parts[0]), float(parts[1])


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
        records.append({"tone": tone, "gain_db": gain, "readings": value})
    return records


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


def power_to_mw(power_values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "pw":
        return power_values * 1e-9
    if unit == "w":
        return power_values * 1e3
    if unit == "mw":
        return power_values
    raise ValueError(f"Unsupported power unit: {unit}")


def extract_tone_points(
    records: list[dict[str, Any]],
    power_key: str = "pwr_pw",
    power_unit: str = "pw",
    gain_to_prf_offset_db: float = 80.0,
) -> dict[int, dict[str, np.ndarray]]:
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
            value = reading.get(power_key)
            if value is None:
                continue
            grouped[int(tone)][float(gain)].append(float(value))

    tone_data: dict[int, dict[str, np.ndarray]] = {}
    for tone, by_gain in sorted(grouped.items()):
        gains = np.array(sorted(by_gain.keys()), dtype=float)
        mean_raw = np.array([float(np.mean(by_gain[float(gain)])) for gain in gains], dtype=float)
        prf_dbm = gains - gain_to_prf_offset_db
        prf_mw = dbm_to_mw(prf_dbm)
        pdc_mw = power_to_mw(mean_raw, power_unit)
        valid = np.isfinite(prf_mw) & np.isfinite(pdc_mw) & (prf_mw > 0.0)
        prf_dbm = prf_dbm[valid]
        prf_mw = prf_mw[valid]
        pdc_mw = pdc_mw[valid]
        gains = gains[valid]
        if prf_mw.size == 0:
            continue
        order = np.argsort(prf_mw)
        tone_data[tone] = {
            "gain_db": gains[order],
            "prf_dbm": prf_dbm[order],
            "prf_mw": prf_mw[order],
            "pdc_mw": pdc_mw[order],
        }
    return tone_data


def linear_efficiency_model(prf_mw: np.ndarray | float, eta: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    return eta * prf


def polynomial_even_model(prf_mw: np.ndarray | float, a2: float, a4: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    return a2 * prf**2 + a4 * prf**4


def polynomial_cubic_model(prf_mw: np.ndarray | float, a1: float, a2: float, a3: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    return a1 * prf + a2 * prf**2 + a3 * prf**3


def logistic_model(prf_mw: np.ndarray | float, pmax: float, a: float, b: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    exponent = np.clip(-a * (prf - b), -700.0, 700.0)
    return pmax / (1.0 + np.exp(exponent))


def logistic_sigmoind_model(prf_mw: np.ndarray | float, pmax: float, a: float, b: float) -> np.ndarray:
    """PDC = Pmax / (1 + exp(-a*(PRF - b)))."""
    return logistic_model(prf_mw, pmax, a, b)


def paper_logistic_model(prf_mw: np.ndarray | float, psat: float, a: float, b: float) -> np.ndarray:
    """Normalized logistic model from the paper:

    Psi_dc = Psat / (1 + exp(-a*(Prf - b)))
    Omega  = 1 / (1 + exp(a*b))
    Pdc    = (Psi_dc - Psat*Omega) / (1 - Omega)
    """
    prf = np.asarray(prf_mw, dtype=float)
    psi_exponent = np.clip(-a * (prf - b), -700.0, 700.0)
    psi_dc = psat / (1.0 + np.exp(psi_exponent))
    omega_exponent = np.clip(a * b, -700.0, 700.0)
    omega = 1.0 / (1.0 + np.exp(omega_exponent))
    denominator = np.maximum(1.0 - omega, EPS)
    return (psi_dc - psat * omega) / denominator


def rational_saturation_model(prf_mw: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    prf_power = np.power(np.maximum(prf, 0.0), b)
    return (a * prf_power) / (1.0 + c * prf_power)


def power_law_offset_model(prf_mw: np.ndarray | float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    return alpha * np.power(np.maximum(prf, EPS), beta) + gamma


def exponential_model(prf_mw: np.ndarray | float, amp: float, slope: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    exponent = np.clip(slope * prf, -700.0, 700.0)
    return amp * np.expm1(exponent)


def piecewise_linear_model(prf_mw: np.ndarray | float, psens: float, eta: float, psat: float) -> np.ndarray:
    prf = np.asarray(prf_mw, dtype=float)
    active_span = np.maximum(psat - psens, EPS)
    return eta * np.clip(prf - psens, 0.0, active_span)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    func: Callable[..., np.ndarray]
    param_names: tuple[str, ...]
    initial_guess: Callable[[np.ndarray, np.ndarray], tuple[float, ...]]
    bounds: Callable[[np.ndarray, np.ndarray], tuple[tuple[float, ...], tuple[float, ...]]]


@dataclass
class FitResult:
    model_name: str
    param_names: tuple[str, ...]
    params: np.ndarray
    rmse: float
    r2: float
    success: bool
    error: str | None = None


def _x_stats(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    mask = np.isfinite(prf_mw) & np.isfinite(pdc_mw) & (prf_mw > 0.0)
    x = prf_mw[mask]
    y = pdc_mw[mask]
    if x.size == 0:
        x = np.array([1.0], dtype=float)
        y = np.array([0.0], dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(x_max - x_min, EPS)
    return x, y, x_min, x_max, x_span


def _least_squares_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray, powers: tuple[int, ...]) -> tuple[float, ...]:
    x, y, *_ = _x_stats(prf_mw, pdc_mw)
    design = np.column_stack([x**power for power in powers])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return tuple(float(value) for value in coeffs)


def _linear_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    x, y, *_ = _x_stats(prf_mw, pdc_mw)
    ratio = y / np.maximum(x, EPS)
    eta = float(np.nanmedian(np.clip(ratio, 0.0, np.inf)))
    if not np.isfinite(eta):
        eta = 0.1
    return (eta,)


def _linear_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((0.0,), (np.inf,))


def _poly_even_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    return _least_squares_guess(prf_mw, pdc_mw, (2, 4))


def _poly_even_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((-np.inf, -np.inf), (np.inf, np.inf))


def _poly_cubic_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    return _least_squares_guess(prf_mw, pdc_mw, (1, 2, 3))


def _poly_cubic_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))


def _logistic_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    x, y, _, _, x_span = _x_stats(prf_mw, pdc_mw)
    y_max = float(np.max(y)) if y.size else 1.0
    return (max(y_max, EPS), 4.0 / x_span, float(np.median(x)))


def _logistic_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _, _, x_min, x_max, _ = _x_stats(prf_mw, pdc_mw)
    return ((0.0, 1e-8, x_min), (np.inf, np.inf, x_max * 2.0))


def _paper_logistic_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    return _logistic_guess(prf_mw, pdc_mw)


def _paper_logistic_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _, _, _, x_max, _ = _x_stats(prf_mw, pdc_mw)
    return ((0.0, 1e-8, 0.0), (np.inf, np.inf, x_max * 2.0))


def _rational_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    _, y, _, x_max, _ = _x_stats(prf_mw, pdc_mw)
    y_max = float(np.max(y)) if y.size else 1.0
    b0 = 1.0
    a0 = max(y_max / max(x_max**b0, EPS), EPS)
    c0 = 1.0 / max(x_max**b0, EPS)
    return (a0, b0, c0)


def _rational_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((0.0, 0.0, 0.0), (np.inf, 10.0, np.inf))


def _power_law_offset_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    x, y, _, _, _ = _x_stats(prf_mw, pdc_mw)
    y_min = float(np.min(y)) if y.size else 0.0
    y_max = float(np.max(y)) if y.size else 1.0
    y_scale = max(y_max - y_min, abs(y_max), 1.0, EPS)
    gamma_candidates = np.unique(
        np.concatenate(
            (
                np.linspace(y_min - y_scale, y_min - 0.01 * y_scale, num=12),
                np.array([0.0, y_min - 0.1 * y_scale, y_min - 0.5 * y_scale]),
            )
        )
    )

    design = np.column_stack([np.ones_like(x), np.log(np.maximum(x, EPS))])
    best_guess: tuple[float, float, float] | None = None
    best_score = float("inf")
    for gamma0 in gamma_candidates:
        if gamma0 >= y_min:
            continue
        shifted = y - gamma0
        if not np.all(shifted > 0.0):
            continue
        coeffs, *_ = np.linalg.lstsq(design, np.log(shifted), rcond=None)
        alpha0 = float(np.exp(coeffs[0]))
        beta0 = float(np.clip(coeffs[1], 0.0, 10.0))
        if not np.isfinite(alpha0) or alpha0 <= 0.0 or not np.isfinite(beta0):
            continue
        pred = alpha0 * np.power(np.maximum(x, EPS), beta0) + gamma0
        score = float(np.sqrt(np.mean((y - pred) ** 2)))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_guess = (alpha0, beta0, float(gamma0))

    if best_guess is not None:
        return best_guess

    gamma0 = y_min - 0.1 * y_scale
    alpha0 = max((y_max - gamma0) / max(float(np.max(x)), EPS), EPS)
    return (alpha0, 1.0, gamma0)


def _power_law_offset_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((0.0, 0.0, -np.inf), (np.inf, 10.0, np.inf))


def _exponential_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    _, y, _, x_max, x_span = _x_stats(prf_mw, pdc_mw)
    y_max = float(np.max(y)) if y.size else 1.0
    slope0 = 1.0 / x_span
    amp0 = y_max / max(np.expm1(slope0 * x_max), EPS)
    return (max(amp0, EPS), slope0)


def _exponential_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _ = (prf_mw, pdc_mw)
    return ((0.0, 0.0), (np.inf, np.inf))


def _piecewise_guess(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[float, ...]:
    x, y, _, _, x_span = _x_stats(prf_mw, pdc_mw)
    psens0 = float(np.quantile(x, 0.2))
    psat0 = float(np.quantile(x, 0.8))
    if psat0 <= psens0:
        psat0 = psens0 + 0.5 * x_span
    y_min = float(np.min(y)) if y.size else 0.0
    y_max = float(np.max(y)) if y.size else 1.0
    eta0 = max((y_max - y_min) / max(psat0 - psens0, EPS), 0.0)
    return (psens0, eta0, psat0)


def _piecewise_bounds(prf_mw: np.ndarray, pdc_mw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _, _, x_min, x_max, _ = _x_stats(prf_mw, pdc_mw)
    return ((x_min, 0.0, x_min), (x_max, np.inf, x_max * 2.0))


MODEL_SPECS: dict[str, ModelSpec] = {
    "linear_efficiency": ModelSpec("linear_efficiency", linear_efficiency_model, ("eta",), _linear_guess, _linear_bounds),
    "polynomial_even": ModelSpec("polynomial_even", polynomial_even_model, ("a2", "a4"), _poly_even_guess, _poly_even_bounds),
    "polynomial_cubic": ModelSpec(
        "polynomial_cubic",
        polynomial_cubic_model,
        ("a1", "a2", "a3"),
        _poly_cubic_guess,
        _poly_cubic_bounds,
    ),
    "logistic": ModelSpec("logistic", logistic_model, ("Pmax", "a", "b"), _logistic_guess, _logistic_bounds),
    "logistic_sigmoind_model": ModelSpec(
        "logistic_sigmoind_model",
        logistic_sigmoind_model,
        ("Pmax", "a", "b"),
        _logistic_guess,
        _logistic_bounds,
    ),
    "paper_logistic": ModelSpec(
        "paper_logistic",
        paper_logistic_model,
        ("Psat", "a", "b"),
        _paper_logistic_guess,
        _paper_logistic_bounds,
    ),
    "rational_saturation": ModelSpec(
        "rational_saturation",
        rational_saturation_model,
        ("a", "b", "c"),
        _rational_guess,
        _rational_bounds,
    ),
    "power_law_offset": ModelSpec(
        "power_law_offset",
        power_law_offset_model,
        ("alpha", "beta", "gamma"),
        _power_law_offset_guess,
        _power_law_offset_bounds,
    ),
    "exponential": ModelSpec("exponential", exponential_model, ("A", "B"), _exponential_guess, _exponential_bounds),
    "piecewise_linear": ModelSpec(
        "piecewise_linear",
        piecewise_linear_model,
        ("Psens", "eta", "Psat"),
        _piecewise_guess,
        _piecewise_bounds,
    ),
}


def get_model_specs(model_names: list[str] | None) -> list[ModelSpec]:
    if model_names is None:
        return list(MODEL_SPECS.values())
    specs = []
    for name in model_names:
        key = name.strip()
        if key not in MODEL_SPECS:
            available = ", ".join(MODEL_SPECS.keys())
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        specs.append(MODEL_SPECS[key])
    return specs


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = y_true - y_pred
    return float(np.sqrt(np.mean(residual * residual)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - np.mean(y_true)
    sst = float(np.sum(centered * centered))
    if sst <= EPS:
        return float("nan")
    residual = y_true - y_pred
    sse = float(np.sum(residual * residual))
    return float(1.0 - sse / sst)


def fit_models_for_tone(
    prf_mw: np.ndarray,
    pdc_mw: np.ndarray,
    specs: list[ModelSpec],
    maxfev: int,
) -> list[FitResult]:
    curve_fit = get_curve_fit()
    results: list[FitResult] = []
    for spec in specs:
        p0 = np.asarray(spec.initial_guess(prf_mw, pdc_mw), dtype=float)
        lb, ub = spec.bounds(prf_mw, pdc_mw)
        bounds = (np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params, _ = curve_fit(spec.func, prf_mw, pdc_mw, p0=p0, bounds=bounds, maxfev=maxfev)
                pred = np.asarray(spec.func(prf_mw, *params), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise RuntimeError("Non-finite predictions")
            results.append(
                FitResult(
                    model_name=spec.name,
                    param_names=spec.param_names,
                    params=np.asarray(params, dtype=float),
                    rmse=rmse(pdc_mw, pred),
                    r2=r2_score(pdc_mw, pred),
                    success=True,
                    error=None,
                )
            )
        except Exception as exc:
            results.append(
                FitResult(
                    model_name=spec.name,
                    param_names=spec.param_names,
                    params=p0,
                    rmse=float("inf"),
                    r2=float("nan"),
                    success=False,
                    error=str(exc),
                )
            )
    return results


def format_tone_table(tone: int, results: list[FitResult]) -> str:
    ordered = sorted(results, key=lambda item: item.rmse)
    header = f"Tone {tone}"
    rows = [
        (item.model_name, "nan" if not np.isfinite(item.rmse) else f"{item.rmse:.6g}", "nan" if not np.isfinite(item.r2) else f"{item.r2:.6g}", "ok" if item.success else "failed")
        for item in ordered
    ]
    widths = [
        max(len("model"), *(len(row[0]) for row in rows)),
        max(len("RMSE_mW"), *(len(row[1]) for row in rows)),
        max(len("R2"), *(len(row[2]) for row in rows)),
        max(len("status"), *(len(row[3]) for row in rows)),
    ]
    lines = [header]
    lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(("model", "RMSE_mW", "R2", "status"))))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def save_extracted_csv(path: Path, tone_data: dict[int, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("tone,gain_db,Prf_dBm,Prf_mW,Pdc_mW\n")
        for tone in sorted(tone_data):
            data = tone_data[tone]
            for gain_db, prf_dbm, prf_mw, pdc_mw in zip(
                data["gain_db"], data["prf_dbm"], data["prf_mw"], data["pdc_mw"], strict=False
            ):
                handle.write(
                    f"{tone},{gain_db:.12g},{prf_dbm:.12g},{prf_mw:.12g},{pdc_mw:.12g}\n"
                )


def save_results_json(path: Path, by_tone_results: dict[int, list[FitResult]]) -> None:
    payload: dict[str, Any] = {"tones": {}}
    for tone in sorted(by_tone_results):
        entries = []
        for result in sorted(by_tone_results[tone], key=lambda item: item.rmse):
            entries.append(
                {
                    "model": result.model_name,
                    "success": result.success,
                    "rmse_mw": result.rmse,
                    "r2": result.r2,
                    "param_names": list(result.param_names),
                    "params": [float(value) for value in result.params],
                    "error": result.error,
                }
            )
        payload["tones"][str(tone)] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def per_tone_plot_path(base_output: Path, tone: int) -> Path:
    return base_output.with_name(f"{base_output.stem}_tone{tone}{base_output.suffix}")


def format_param_summary(param_names: tuple[str, ...], params: np.ndarray) -> str:
    parts = []
    for name, value in zip(param_names, params, strict=False):
        if np.isfinite(value):
            parts.append(f"{name}={value:.3g}")
        else:
            parts.append(f"{name}=nan")
    return "[" + ", ".join(parts) + "]"


def plot_fits(
    tone_data: dict[int, dict[str, np.ndarray]],
    by_tone_results: dict[int, list[FitResult]],
    output_path: Path,
    show: bool,
) -> None:
    plt = get_pyplot()
    tones = sorted(tone_data.keys())
    for tone in tones:
        fig, axis = plt.subplots(1, 1, figsize=(8.0, 5.6))
        data = tone_data[tone]
        prf_mw = data["prf_mw"]
        pdc_mw = data["pdc_mw"]
        axis.scatter(prf_mw, pdc_mw, color="black", s=24, alpha=0.75, label="Measured")
        x_min = float(np.min(prf_mw))
        x_max = float(np.max(prf_mw))
        if x_max <= x_min:
            x_max = x_min * 1.1
        grid = np.linspace(x_min, x_max, num=350)

        top_results = [entry for entry in sorted(by_tone_results[tone], key=lambda item: item.rmse) if entry.success][:3]
        for result in top_results:
            spec = MODEL_SPECS[result.model_name]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                y_grid = np.asarray(spec.func(grid, *result.params), dtype=float)
            valid = np.isfinite(y_grid)
            if np.count_nonzero(valid) < 2:
                continue
            axis.plot(
                grid[valid],
                y_grid[valid],
                linewidth=1.6,
                label=f"{result.model_name} ({result.rmse:.3g}) {format_param_summary(result.param_names, result.params)}",
            )

        axis.set_xlabel("Prf (mW)")
        axis.set_ylabel("Pdc (mW)")
        axis.set_title(f"Tone {tone}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)

        fig.tight_layout()
        tone_output = per_tone_plot_path(output_path, tone)
        tone_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(tone_output, dpi=170, bbox_inches="tight")
        print(f"Saved fit plot to {tone_output}")
        if show:
            plt.show()
        else:
            plt.close(fig)


def parse_model_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read tone measurement JSONL, extract Prf/Pdc, fit models, and plot fitting."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Measurement file (.json/.jsonl). If omitted, newest '*meas-tones-power*' is used.",
    )
    parser.add_argument("--power-key", default="pwr_pw", help="Reading field to use for output power (default: pwr_pw)")
    parser.add_argument(
        "--power-unit",
        choices=("pw", "w", "mw"),
        default="pw",
        help="Unit of --power-key values in measurement records (default: pw)",
    )
    parser.add_argument(
        "--gain-offset-db",
        type=float,
        default=80.0,
        help="Prf_dBm = gain_db - gain_offset_db (default: 80)",
    )
    parser.add_argument("--models", default=None, help="Comma-separated model list (default: all)")
    parser.add_argument("--maxfev", type=int, default=200_000, help="Max curve_fit evaluations per model")
    parser.add_argument("--csv-out", type=Path, default=None, help="Extracted Prf/Pdc CSV output path")
    parser.add_argument("--json-out", type=Path, default=None, help="Fit results JSON output path")
    parser.add_argument("--plot-out", type=Path, default=None, help="Fit plot output path")
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
    parser.add_argument("--no-show", action="store_true", help="Save plot without opening interactive window")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = resolve_input_path(args.input)
    run_id = args.run_id or infer_run_id_from_path(input_path) or manual_run_id()
    results_dir = args.results_dir.resolve()
    model_names = parse_model_list(args.models)
    specs = get_model_specs(model_names)

    records = load_records(input_path)
    tone_data = extract_tone_points(
        records,
        power_key=args.power_key,
        power_unit=args.power_unit,
        gain_to_prf_offset_db=args.gain_offset_db,
    )
    if not tone_data:
        raise ValueError("No usable tone data extracted from the measurement file.")

    csv_out = resolve_output_path(
        args.csv_out,
        results_dir=results_dir,
        run_id=run_id,
        bucket="tables",
        default_name=default_output_path(input_path, "Prf_Pdc_by_tone.csv").name,
    )
    json_out = resolve_output_path(
        args.json_out,
        results_dir=results_dir,
        run_id=run_id,
        bucket="tables",
        default_name=default_output_path(input_path, "tone_fit_results.json").name,
    )
    plot_out = resolve_output_path(
        args.plot_out,
        results_dir=results_dir,
        run_id=run_id,
        bucket="plots",
        default_name=default_output_path(input_path, "tone_fit_results.png").name,
    )

    save_extracted_csv(csv_out, tone_data)
    print(f"Saved extracted CSV to {csv_out}")

    by_tone_results: dict[int, list[FitResult]] = {}
    for tone in sorted(tone_data):
        data = tone_data[tone]
        tone_results = fit_models_for_tone(data["prf_mw"], data["pdc_mw"], specs, maxfev=args.maxfev)
        by_tone_results[tone] = tone_results
        print()
        print(format_tone_table(tone, tone_results))

    save_results_json(json_out, by_tone_results)
    print(f"\nSaved fit results JSON to {json_out}")
    plot_fits(tone_data, by_tone_results, output_path=plot_out, show=not args.no_show)
    manifest_path = write_manifest(
        results_dir=results_dir,
        run_id=run_id,
        script_name=Path(__file__).name,
        argv=sys.argv[1:],
        extra={
            "input": str(input_path),
            "csv_out": str(csv_out),
            "json_out": str(json_out),
            "plot_out": str(plot_out),
        },
    )
    print(f"Updated run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
