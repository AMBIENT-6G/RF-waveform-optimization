#!/usr/bin/env python3
"""Sweep tone/gain settings and log oscilloscope power measurements.

Behavior mirrors ``meas-tones-power.py``:
- sweep tone/gain combinations
- launch ``tx_waveform.py`` for each sweep
- wait before sampling
- sample power over a window
- append one JSON object per sweep to output JSONL

Power is read from ``TechtileScope.Scope.get_power_Watt()`` and stored as
``pwr_pw`` (pico-watts) in each reading.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_output_path() -> Path:
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{timestamp_prefix}_meas-tone-power-scope.jsonl")


def default_python_executable() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".venv" / "Scripts" / "python.exe",
        script_dir / ".venv" / "bin" / "python",
        script_dir / "venv" / "Scripts" / "python.exe",
        script_dir / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return sys.executable


def parse_tone_list(value: str) -> list[int]:
    tones = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        tone = int(stripped)
        if tone < 0:
            raise argparse.ArgumentTypeError("Tone values must be >= 0")
        tones.append(tone)

    if not tones:
        raise argparse.ArgumentTypeError("At least one tone must be provided")

    return tones


def build_gain_values(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("--gain-step must not be 0")

    if step > 0 and start > stop:
        raise ValueError("--gain-start must be <= --gain-stop when --gain-step is positive")

    if step < 0 and start < stop:
        raise ValueError("--gain-start must be >= --gain-stop when --gain-step is negative")

    gains = []
    current = start
    epsilon = abs(step) * 1e-9 + 1e-12

    if step > 0:
        while current <= stop + epsilon:
            gains.append(round(current, 10))
            current += step
    else:
        while current >= stop - epsilon:
            gains.append(round(current, 10))
            current += step

    return gains


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")


def load_scope_config(path: Path | None, key: str) -> Any | None:
    if path is None:
        return None

    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError(
            "--scope-settings requires PyYAML (pip install pyyaml)"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if isinstance(data, dict) and key in data:
        return data[key]

    return data


class ScopePowerMeter:
    def __init__(self, channel: int, config: Any | None):
        try:
            from TechtileScope import Scope  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import TechtileScope.Scope. Ensure TechtileScope is installed and available."
            ) from exc

        self.scope = (
            Scope(config=config) if config is not None else Scope("192.108.0.251")
        )
        self.channel = int(channel)

    def close(self) -> None:
        close_fn = getattr(self.scope, "close", None)
        if callable(close_fn):
            close_fn()

    def read_power_watt(self) -> float | None:
        values = self.scope.get_power_Watt()
        if values is None:
            return None

        if np.isscalar(values):
            candidate = values
        else:
            seq = list(values)
            if not seq:
                return None
            if self.channel >= len(seq):
                raise IndexError(
                    f"--scope-channel={self.channel} out of range for scope output with {len(seq)} channel(s)"
                )
            candidate = seq[self.channel]

        if candidate is None:
            return None

        power_w = float(candidate)
        if not np.isfinite(power_w):
            return None
        return power_w

    def get_measurement(self) -> dict[str, Any] | None:
        power_w = self.read_power_watt()
        if power_w is None:
            return None

        return {
            "timestamp_ms": round(time.time_ns() / 1e6),
            "scope_channel": self.channel,
            "scope_power_w": power_w,
            "pwr_pw": power_w * 1e12,
        }


def collect_measurements(
    meter: ScopePowerMeter,
    window_s: float,
    poll_interval_s: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + window_s
    measurements = []

    while time.monotonic() < deadline:
        measurement = meter.get_measurement()
        if measurement is not None:
            measurements.append(measurement)
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)

    return measurements


def launch_tx_process(
    python_executable: str,
    tx_script: Path,
    tone: int,
    bandwidth_khz: int,
    gain_db: float,
    duration_s: float,
) -> subprocess.Popen[bytes]:
    command = [
        python_executable,
        str(tx_script),
        "--tone",
        str(tone),
        "--bw",
        str(bandwidth_khz),
        "--gain",
        f"{gain_db:g}",
        "--duration",
        f"{duration_s:g}",
    ]
    print(f"Launching TX: {' '.join(command)}")
    return subprocess.Popen(command)


def wait_for_process(process: subprocess.Popen[bytes]) -> int:
    return process.wait()


def run_sweep(args: argparse.Namespace) -> int:
    scope_config = load_scope_config(args.scope_settings, args.scope_config_key)
    meter = ScopePowerMeter(channel=args.scope_channel, config=scope_config)
    tx_script = Path(__file__).with_name("tx_waveform.py").resolve()
    gains = build_gain_values(args.gain_start, args.gain_stop, args.gain_step)
    completed_sweeps = 0

    try:
        for tone in args.tones:
            for gain in gains:
                print(f"Starting sweep: tone={tone}, gain={gain:g} dB")
                process = launch_tx_process(
                    python_executable=args.python,
                    tx_script=tx_script,
                    tone=tone,
                    bandwidth_khz=args.bw,
                    gain_db=gain,
                    duration_s=args.tx_duration,
                )

                started_at = utc_now_iso()
                sweep_started = time.monotonic()
                try:
                    time.sleep(args.pre_measure_delay)
                    readings = collect_measurements(
                        meter,
                        window_s=args.measure_window,
                        poll_interval_s=args.scope_poll_interval,
                    )
                    exit_code = wait_for_process(process)
                except Exception:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise

                if exit_code != 0:
                    raise RuntimeError(
                        f"tx_waveform.py failed for tone={tone}, gain={gain:g} dB with exit code {exit_code}"
                    )

                sweep_duration_s = time.monotonic() - sweep_started
                record = {
                    "started_at": started_at,
                    "completed_at": utc_now_iso(),
                    "measurement_source": "scope",
                    "tone": tone,
                    "bw_khz": args.bw,
                    "gain_db": gain,
                    "tx_duration_s": args.tx_duration,
                    "pre_measure_delay_s": args.pre_measure_delay,
                    "measure_window_s": args.measure_window,
                    "scope_channel": args.scope_channel,
                    "scope_poll_interval_s": args.scope_poll_interval,
                    "sweep_duration_s": round(sweep_duration_s, 3),
                    "reading_count": len(readings),
                    "readings": readings,
                }
                append_jsonl(args.output, record)
                completed_sweeps += 1
                print(
                    f"Stored {len(readings)} readings for tone={tone}, gain={gain:g} dB "
                    f"to {args.output}"
                )
    finally:
        meter.close()

    print(f"Completed {completed_sweeps} sweeps.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure scope power while replaying TX waveforms")
    parser.add_argument(
        "--tones",
        type=parse_tone_list,
        default=[0, 1, 4, 8, 16, 32],
        help="Comma-separated tone list (default: 0,1,4,8,16,32)",
    )
    parser.add_argument("--bw", type=int, default=1000, help="Waveform bandwidth in kHz (default: 1000)")
    parser.add_argument("--gain-start", type=float, default=50.0, help="Start gain in dB (default: 50)")
    parser.add_argument("--gain-stop", type=float, default=85.0, help="Stop gain in dB, inclusive (default: 85)")
    parser.add_argument("--gain-step", type=float, default=0.2, help="Gain step in dB (default: 0.2)")
    parser.add_argument("--tx-duration", type=float, default=20.0, help="TX duration in seconds (default: 20)")
    parser.add_argument(
        "--pre-measure-delay",
        type=float,
        default=10.0,
        help="Delay after starting TX before sampling the scope (default: 10)",
    )
    parser.add_argument(
        "--measure-window",
        type=float,
        default=10.0,
        help="How long to collect scope readings per sweep in seconds (default: 10)",
    )
    parser.add_argument(
        "--scope-channel",
        type=int,
        default=0,
        help="Scope channel index used for power extraction from get_power_Watt() (default: 0)",
    )
    parser.add_argument(
        "--scope-poll-interval",
        type=float,
        default=0.1,
        help="Delay between scope reads in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--scope-settings",
        type=Path,
        default=None,
        help="Optional YAML settings file. If provided, its --scope-config-key section is passed to Scope(config=...).",
    )
    parser.add_argument(
        "--scope-config-key",
        default="scope",
        help="Key inside --scope-settings used as scope config (default: scope)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Append-only JSONL output path (default: <timestamp>_meas-tone-power-scope.jsonl)",
    )
    parser.add_argument(
        "--python",
        default=default_python_executable(),
        help=(
            "Python executable used to launch tx_waveform.py "
            "(default: local .venv/venv if present, else current interpreter)"
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.output is None:
        args.output = default_output_path()
    if args.bw <= 0:
        raise ValueError("--bw must be > 0")
    if args.tx_duration <= 0:
        raise ValueError("--tx-duration must be > 0")
    if args.pre_measure_delay < 0:
        raise ValueError("--pre-measure-delay must be >= 0")
    if args.measure_window <= 0:
        raise ValueError("--measure-window must be > 0")
    if args.scope_channel < 0:
        raise ValueError("--scope-channel must be >= 0")
    if args.scope_poll_interval < 0:
        raise ValueError("--scope-poll-interval must be >= 0")
    if args.scope_settings is not None and not args.scope_settings.exists():
        raise FileNotFoundError(f"--scope-settings file not found: {args.scope_settings}")
    if not Path(args.python).exists() and shutil.which(args.python) is None:
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    return run_sweep(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
