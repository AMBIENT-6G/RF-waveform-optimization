#!/usr/bin/env python3
"""Sweep tone/gain settings and log energy-profiler measurements.

Default behavior:
- tones: 0, 1, 4, 8, 16, 32 (0 = DC, 1 = NB)
- gains: auto-discovered from gain-tagged IQ files when available
- optional explicit gain sweeps require --gain-start, --gain-stop, and --gain-step together
- launches ``tx_waveform.py`` for each tone/gain combination
- waits 10 s before sampling the energy profiler
- records profiler readings for 10 s
- waits for the TX process to finish
- appends one JSON object per sweep to the output file
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import serial

from run_layout import resolve_output_path, timestamp_run_id, write_manifest


START_BYTE = 0x02
READING_FORMAT = ">IIII"
READING_PAYLOAD_SIZE = struct.calcsize(READING_FORMAT)
REPO_ROOT = Path(__file__).resolve().parents[1]
IQ_DIR = REPO_ROOT / "data" / "tx_iq"
TX_GAIN_REGEX = re.compile(r"_TXG(?P<gain>[-+]?(?:\d+\.?\d*|\.\d+))db(?:_|$)", re.IGNORECASE)


def xor_checksum(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_output_path() -> Path:
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{timestamp_prefix}_meas-tones-power.jsonl")


def default_python_executable() -> str:
    script_dir = REPO_ROOT
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

    return list(dict.fromkeys(tones))


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


def tone_label(tone: int) -> str:
    if tone == 0:
        return "DC"
    if tone == 1:
        return "NB"
    return f"N={tone}"


def waveform_kind(tone: int) -> str:
    if tone == 0:
        return "dc"
    if tone == 1:
        return "nb"
    return "multitone"


def waveform_glob(iq_dir: Path, tone: int, bandwidth_khz: int):
    if tone == 0:
        pattern = f"iq_dc_BW{bandwidth_khz}*.npz"
    elif tone == 1:
        pattern = f"iq_nb_BW{bandwidth_khz}*.npz"
    else:
        pattern = f"iq_N{tone}_BW{bandwidth_khz}*.npz"
    return sorted(iq_dir.glob(pattern))


def parse_tx_gain_from_iq_name(path: Path) -> float | None:
    match = TX_GAIN_REGEX.search(path.stem)
    if not match:
        return None
    try:
        return float(match.group("gain"))
    except ValueError:
        return None


def discover_available_gains(iq_dir: Path, bandwidth_khz: int, tones: list[int]) -> tuple[list[float], str]:
    multitone_tones = [tone for tone in tones if tone >= 2]
    if not multitone_tones:
        multitone_tones = sorted(
            {
                int(match.group("tone"))
                for path in iq_dir.glob(f"iq_N*_BW{bandwidth_khz}*.npz")
                for match in [re.search(r"iq_N(?P<tone>\d+)_BW", path.name)]
                if match is not None
            }
        )

    gain_sets: list[set[float]] = []
    contributing_tones: list[int] = []
    for tone in multitone_tones:
        gains = {
            gain
            for path in waveform_glob(iq_dir, tone, bandwidth_khz)
            for gain in [parse_tx_gain_from_iq_name(path)]
            if gain is not None and math.isfinite(gain)
        }
        if gains:
            gain_sets.append(gains)
            contributing_tones.append(tone)

    if gain_sets:
        shared = set.intersection(*gain_sets)
        if not shared:
            detail = {tone_label(tone): sorted(gains) for tone, gains in zip(contributing_tones, gain_sets, strict=False)}
            raise RuntimeError(
                f"No shared TX gain set found across BW={bandwidth_khz}kHz multitone IQ files: {detail}"
            )
        return sorted(shared), f"auto-discovered shared gains from {', '.join(tone_label(t) for t in contributing_tones)}"

    return [], "no gain-tagged multitone IQ files found"


def resolve_iq_file_for_request(
    iq_dir: Path,
    tone: int,
    bandwidth_khz: int,
    gain_db: float,
    *,
    closest_gain_match: bool = False,
) -> Path:
    matches = waveform_glob(iq_dir, tone, bandwidth_khz)
    if not matches:
        raise FileNotFoundError(
            f"No IQ file found for {tone_label(tone)}, BW={bandwidth_khz}kHz in {iq_dir.resolve()}"
        )

    if tone in (0, 1):
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple IQ files found for {tone_label(tone)}, BW={bandwidth_khz}kHz: {[p.name for p in matches]}"
            )
        return matches[0].resolve()

    exact = [
        path
        for path in matches
        if (gain := parse_tx_gain_from_iq_name(path)) is not None
        and math.isclose(gain, gain_db, rel_tol=0.0, abs_tol=1e-6)
    ]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        raise RuntimeError(
            f"Multiple IQ files found for {tone_label(tone)}, BW={bandwidth_khz}kHz, gain={gain_db:g} dB: "
            f"{[p.name for p in exact]}"
        )

    if closest_gain_match:
        tagged_matches = [
            (abs(gain - gain_db), gain, path)
            for path in matches
            for gain in [parse_tx_gain_from_iq_name(path)]
            if gain is not None
        ]
        if tagged_matches:
            tagged_matches.sort(key=lambda item: (item[0], item[1], item[2].name))
            return tagged_matches[0][2].resolve()

    if len(matches) == 1 and parse_tx_gain_from_iq_name(matches[0]) is None:
        return matches[0].resolve()

    available_gains = sorted(
        gain
        for path in matches
        for gain in [parse_tx_gain_from_iq_name(path)]
        if gain is not None
    )
    raise FileNotFoundError(
        f"No IQ file found for {tone_label(tone)}, BW={bandwidth_khz}kHz, gain={gain_db:g} dB. "
        f"Available tagged gains: {available_gains}"
    )


def build_sweep_plan(
    iq_dir: Path,
    bandwidth_khz: int,
    tones: list[int],
    gains: list[float],
    *,
    closest_gain_match: bool = False,
) -> list[dict[str, Any]]:
    plan = []
    for gain_db in gains:
        for tone in tones:
            iq_file = resolve_iq_file_for_request(
                iq_dir,
                tone,
                bandwidth_khz,
                gain_db,
                closest_gain_match=closest_gain_match,
            )
            plan.append(
                {
                    "tone": int(tone),
                    "gain_db": float(gain_db),
                    "waveform_kind": waveform_kind(tone),
                    "waveform_label": tone_label(tone),
                    "iq_file": str(iq_file),
                }
            )
    return plan


@dataclass
class EnergyProfiler:
    port: str
    baudrate: int
    timeout: float

    def __post_init__(self) -> None:
        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self) -> None:
        if self.serial_port.is_open:
            self.serial_port.close()

    def read_raw_values(self) -> tuple[int, int, int, int] | None:
        self.serial_port.reset_input_buffer()

        while True:
            start = self.serial_port.read(1)
            if not start:
                return None
            if start == bytes([START_BYTE]):
                break

        length_byte = self.serial_port.read(1)
        if len(length_byte) != 1:
            return None

        frame_length = length_byte[0]
        frame = self.serial_port.read(frame_length)
        if len(frame) != frame_length:
            return None

        payload = frame[:-1]
        received_checksum = frame[-1]

        if len(payload) != READING_PAYLOAD_SIZE:
            return None

        expected_checksum = xor_checksum(bytes([START_BYTE]) + length_byte + payload)
        if expected_checksum != received_checksum:
            return None

        return struct.unpack(READING_FORMAT, payload)

    def get_measurement(self) -> dict[str, int] | None:
        raw_values = self.read_raw_values()
        if raw_values is None:
            return None

        return {
            "timestamp_ms": round(time.time_ns() / 1e6),
            "buffer_voltage_mv": raw_values[0],
            "resistance": raw_values[1],
            "pwr_pw": raw_values[2],
            "pot_val": raw_values[3],
        }

    def set_target_voltage(self, value: int) -> None:
        command = bytearray()
        command.append(START_BYTE)
        command.append(0x02)
        command.append(0x04)
        command += struct.pack(">I", value)
        command.append(0xFF)

        time.sleep(0.1)
        self.serial_port.write(command)
        self.serial_port.flush()


def collect_measurements(profiler: EnergyProfiler, window_s: float) -> list[dict[str, int]]:
    deadline = time.monotonic() + window_s
    measurements = []

    while time.monotonic() < deadline:
        measurement = profiler.get_measurement()
        if measurement is not None:
            measurements.append(measurement)

    return measurements


def launch_tx_process(
    python_executable: str,
    tx_script: Path,
    tone: int,
    bandwidth_khz: int,
    gain_db: float,
    duration_s: float,
    *,
    closest_gain_match: bool = False,
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
    if closest_gain_match:
        command.append("--closest-gain-match")
    print(f"Launching TX: {' '.join(command)}")
    return subprocess.Popen(command)


def validate_gain_args(args: argparse.Namespace) -> tuple[bool, list[float] | None]:
    provided = {
        "gain_start": args.gain_start is not None,
        "gain_stop": args.gain_stop is not None,
        "gain_step": args.gain_step is not None,
    }
    if any(provided.values()) and not all(provided.values()):
        missing = [
            option
            for option, is_present in (
                ("--gain-start", provided["gain_start"]),
                ("--gain-stop", provided["gain_stop"]),
                ("--gain-step", provided["gain_step"]),
            )
            if not is_present
        ]
        raise ValueError(
            "If any explicit gain sweep argument is provided, all of "
            "--gain-start, --gain-stop, and --gain-step must be provided. "
            f"Missing: {', '.join(missing)}"
        )

    if all(provided.values()):
        return True, build_gain_values(args.gain_start, args.gain_stop, args.gain_step)

    return False, None


def wait_for_process(process: subprocess.Popen[bytes]) -> int:
    return process.wait()


def run_sweep(args: argparse.Namespace) -> int:
    profiler = EnergyProfiler(args.port, args.baudrate, args.serial_timeout)
    tx_script = Path(__file__).with_name("tx_waveform.py").resolve()
    if not IQ_DIR.exists():
        raise FileNotFoundError(f"IQ directory not found: {IQ_DIR.resolve()}")

    explicit_gain_range, explicit_gains = validate_gain_args(args)
    if explicit_gain_range:
        gains = explicit_gains or []
        gain_source = "explicit gain range"
        closest_gain_match = True
    else:
        auto_gains, gain_source = discover_available_gains(IQ_DIR, args.bw, args.tones)
        if not auto_gains:
            raise RuntimeError(
                "No shared gain-tagged multitone IQ files were found for the requested tones and bandwidth. "
                "Provide --gain-start, --gain-stop, and --gain-step to run an explicit gain sweep."
            )
        gains = auto_gains
        closest_gain_match = False

    sweep_plan = build_sweep_plan(
        IQ_DIR,
        args.bw,
        args.tones,
        gains,
        closest_gain_match=closest_gain_match,
    )
    completed_sweeps = 0

    try:
        if args.target_voltage is not None:
            profiler.set_target_voltage(args.target_voltage)
            print(f"Set EP target voltage to {args.target_voltage} mV")

        print(f"Using {len(gains)} gain value(s): {gains}")
        print(f"Gain source: {gain_source}")
        print(
            "Sweep order per gain: "
            + ", ".join(tone_label(tone) for tone in args.tones)
        )

        for entry in sweep_plan:
            tone = int(entry["tone"])
            gain = float(entry["gain_db"])
            print(
                f"Starting sweep: {entry['waveform_label']}, gain={gain:g} dB, iq_file={Path(entry['iq_file']).name}"
            )
            process = launch_tx_process(
                python_executable=args.python,
                tx_script=tx_script,
                tone=tone,
                bandwidth_khz=args.bw,
                gain_db=gain,
                duration_s=args.tx_duration,
                closest_gain_match=closest_gain_match,
            )

            started_at = utc_now_iso()
            sweep_started = time.monotonic()
            try:
                time.sleep(args.pre_measure_delay)
                readings = collect_measurements(profiler, args.measure_window)
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
                    f"tx_waveform.py failed for {entry['waveform_label']}, gain={gain:g} dB with exit code {exit_code}"
                )

            sweep_duration_s = time.monotonic() - sweep_started
            record = {
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "tone": tone,
                "waveform_kind": entry["waveform_kind"],
                "waveform_label": entry["waveform_label"],
                "iq_file": entry["iq_file"],
                "bw_khz": args.bw,
                "gain_db": gain,
                "tx_duration_s": args.tx_duration,
                "pre_measure_delay_s": args.pre_measure_delay,
                "measure_window_s": args.measure_window,
                "sweep_duration_s": round(sweep_duration_s, 3),
                "reading_count": len(readings),
                "readings": readings,
            }
            append_jsonl(args.output, record)
            completed_sweeps += 1
            print(
                f"Stored {len(readings)} readings for {entry['waveform_label']}, gain={gain:g} dB "
                f"to {args.output}"
            )
    finally:
        profiler.close()

    print(f"Completed {completed_sweeps} sweeps.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure energy-profiler power while replaying TX waveforms")
    parser.add_argument(
        "--tones",
        type=parse_tone_list,
        default=[0, 1, 4, 8, 16, 32],
        help="Comma-separated waveform list (0=DC, 1=NB, N>=2 multitone; default: 0,1,4,8,16,32)",
    )
    parser.add_argument("--bw", type=int, default=1000, help="Waveform bandwidth in kHz (default: 1000)")
    parser.add_argument(
        "--gain-start",
        type=float,
        default=None,
        help="Optional explicit sweep start gain in dB. Must be provided together with --gain-stop and --gain-step.",
    )
    parser.add_argument(
        "--gain-stop",
        type=float,
        default=None,
        help="Optional explicit sweep stop gain in dB, inclusive. Must be provided together with --gain-start and --gain-step.",
    )
    parser.add_argument(
        "--gain-step",
        type=float,
        default=None,
        help="Optional explicit sweep gain step in dB. Must be provided together with --gain-start and --gain-stop.",
    )
    parser.add_argument("--tx-duration", type=float, default=20.0, help="TX duration in seconds (default: 20)")
    parser.add_argument(
        "--pre-measure-delay",
        type=float,
        default=10.0,
        help="Delay after starting TX before sampling the profiler (default: 10)",
    )
    parser.add_argument(
        "--measure-window",
        type=float,
        default=10.0,
        help="How long to collect profiler readings per sweep in seconds (default: 10)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Energy-profiler serial port (default: ttyUSB0)",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=1.0,
        help="Serial read timeout in seconds (default: 1)",
    )
    parser.add_argument(
        "--target-voltage",
        type=int,
        default=None,
        help="Optional EP target voltage in mV (uint32). If set, sent once before the sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Append-only JSONL output path (relative paths resolve inside results/<run-id>/raw/)",
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
        help="Run identifier. If omitted, a timestamp run-id is generated.",
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
    run_id = args.run_id if args.run_id else timestamp_run_id()
    results_dir = args.results_dir.resolve()
    if args.output is None:
        args.output = resolve_output_path(
            None,
            results_dir=results_dir,
            run_id=run_id,
            bucket="raw",
            default_name="meas-tones-power.jsonl",
        )
    else:
        args.output = resolve_output_path(
            args.output,
            results_dir=results_dir,
            run_id=run_id,
            bucket="raw",
            default_name="meas-tones-power.jsonl",
        )
    if args.bw <= 0:
        raise ValueError("--bw must be > 0")
    if args.tx_duration <= 0:
        raise ValueError("--tx-duration must be > 0")
    if args.pre_measure_delay < 0:
        raise ValueError("--pre-measure-delay must be >= 0")
    if args.measure_window <= 0:
        raise ValueError("--measure-window must be > 0")
    if args.serial_timeout <= 0:
        raise ValueError("--serial-timeout must be > 0")
    if args.target_voltage is not None and not (0 <= args.target_voltage <= 0xFFFFFFFF):
        raise ValueError("--target-voltage must be in [0, 4294967295]")
    if not Path(args.python).exists() and shutil.which(args.python) is None:
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    manifest_path = write_manifest(
        results_dir=results_dir,
        run_id=run_id,
        script_name=Path(__file__).name,
        argv=sys.argv[1:],
        extra={"output": str(args.output)},
    )
    print(f"Updated run manifest: {manifest_path}")

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
