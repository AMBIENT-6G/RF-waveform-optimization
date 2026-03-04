#!/usr/bin/env python3
"""Sweep tone/gain settings and log energy-profiler measurements.

Default behavior mirrors the pseudocode in the original file:
- tones: 0, 4, 8, 16, 32
- gains: 40 dB to 80 dB in 1 dB steps
- launches ``tx_waveform.py`` for each tone/gain combination
- waits 2 s before sampling the energy profiler
- records profiler readings for 10 s
- waits for the TX process to finish
- appends one JSON object per sweep to the output file
"""

from __future__ import annotations

import argparse
import json
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


START_BYTE = 0x02
READING_FORMAT = ">IIII"
READING_PAYLOAD_SIZE = struct.calcsize(READING_FORMAT)


def xor_checksum(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    profiler = EnergyProfiler(args.port, args.baudrate, args.serial_timeout)
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
                        f"tx_waveform.py failed for tone={tone}, gain={gain:g} dB with exit code {exit_code}"
                    )

                sweep_duration_s = time.monotonic() - sweep_started
                record = {
                    "started_at": started_at,
                    "completed_at": utc_now_iso(),
                    "tone": tone,
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
                    f"Stored {len(readings)} readings for tone={tone}, gain={gain:g} dB "
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
        default=[0, 4, 8, 16, 32],
        help="Comma-separated tone list (default: 0,4,8,16,32)",
    )
    parser.add_argument("--bw", type=int, default=1000, help="Waveform bandwidth in kHz (default: 1000)")
    parser.add_argument("--gain-start", type=float, default=40.0, help="Start gain in dB (default: 40)")
    parser.add_argument("--gain-stop", type=float, default=80.0, help="Stop gain in dB, inclusive (default: 80)")
    parser.add_argument("--gain-step", type=float, default=1.0, help="Gain step in dB (default: 1)")
    parser.add_argument("--tx-duration", type=float, default=20.0, help="TX duration in seconds (default: 20)")
    parser.add_argument(
        "--pre-measure-delay",
        type=float,
        default=2.0,
        help="Delay after starting TX before sampling the profiler (default: 2)",
    )
    parser.add_argument(
        "--measure-window",
        type=float,
        default=10.0,
        help="How long to collect profiler readings per sweep in seconds (default: 10)",
    )
    parser.add_argument("--port", default="COM4", help="Energy-profiler serial port (default: COM4)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=1.0,
        help="Serial read timeout in seconds (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("meas-tones-power.jsonl"),
        help="Append-only JSONL output path (default: meas-tones-power.jsonl)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch tx_waveform.py (default: current interpreter)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
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
