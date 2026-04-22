#!/usr/bin/env python3
"""Sweep tone/gain settings and log energy-profiler measurements.

Default behavior:
- tones: 0, 1, 4, 8, 16, 32 (0 = DC, 1 = NB)
- gains: auto-discovered from gain-tagged IQ files when available
- optional explicit gain sweeps require --gain-start, --gain-stop, and --gain-step together
- waits for a localhost ZMQ ``tx_started`` event from ``tx_waveform.py`` before sampling
- derives measurement window as 90% of ``--tx-duration``
- launches ``tx_waveform.py`` for each tone/gain combination
- stops sampling early on a ZMQ ``tx_done`` event and trims the last 10 samples
- waits for the TX process to finish
- appends one JSON object per sweep to the output file
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import serial
import zmq

from run_layout import resolve_output_path, timestamp_run_id, write_manifest


START_BYTE = 0x02
READING_FORMAT = ">IIII"
READING_PAYLOAD_SIZE = struct.calcsize(READING_FORMAT)
REPO_ROOT = Path(__file__).resolve().parents[1]
IQ_DIR = REPO_ROOT / "data" / "tx_iq"
TX_GAIN_REGEX = re.compile(r"_TXG(?P<gain>[-+]?(?:\d+\.?\d*|\.\d+))db(?:_|$)", re.IGNORECASE)
TX_NOTIFY_HOST = "127.0.0.1"
MEASURE_WINDOW_FRACTION = 0.9
TX_NOTIFY_POLL_TIMEOUT_MS = 50
TX_DONE_TRIM_SAMPLES = 10
EP_SERIAL_POLL_TIMEOUT_S = 0.05
DEFAULT_TX_START_TIMEOUT_S = 60.0
SET_TARGET_VOLTAGE_CMD = 0x02
SET_TARGET_VOLTAGE_VALUE_SIZE = 0x04
DEFAULT_TARGET_VOLTAGE_ACK_TIMEOUT_S = 2.0
EP_ACK_CMD_RE = re.compile(r"CMD\s*=\s*0x(?P<cmd>[0-9A-Fa-f]{1,2})")
EP_ACK_VALUE_RE = re.compile(
    r"VALUE\s*=\s*(?P<value>\d+)\s*\(0x(?P<value_hex>[0-9A-Fa-f]{1,8})\)"
)


class TeeTextIO:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def resolve_run_log_path(
    explicit_log: Path | None,
    *,
    results_dir: Path,
    run_id: str,
) -> Path:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if explicit_log is None:
        target = run_dir / "measure_ep_power.log"
    elif explicit_log.is_absolute():
        target = explicit_log
    else:
        target = run_dir / explicit_log

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


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


def derive_measure_window_s(tx_duration_s: float) -> float:
    return float(tx_duration_s) * MEASURE_WINDOW_FRACTION


def trim_trailing_samples(
    readings: list[dict[str, int]],
    trim_count: int = TX_DONE_TRIM_SAMPLES,
) -> tuple[list[dict[str, int]], int]:
    trimmed = min(max(trim_count, 0), len(readings))
    if trimmed == 0:
        return list(readings), 0
    return list(readings[:-trimmed]), trimmed


def create_tx_notify_socket(context: zmq.Context[Any]) -> tuple[zmq.Socket[Any], str]:
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    port = socket.bind_to_random_port(f"tcp://{TX_NOTIFY_HOST}")
    return socket, f"tcp://{TX_NOTIFY_HOST}:{port}"


def recv_tx_event_nonblocking(socket: zmq.Socket[Any]) -> dict[str, Any] | None:
    try:
        event = socket.recv_json(flags=zmq.NOBLOCK)
    except zmq.Again:
        return None

    if not isinstance(event, dict):
        raise RuntimeError(f"Unexpected TX notify payload type: {type(event).__name__}")
    return event


def wait_for_tx_started(
    process: subprocess.Popen[bytes],
    notify_socket: zmq.Socket[Any],
    timeout_s: float,
) -> dict[str, Any]:
    poller = zmq.Poller()
    poller.register(notify_socket, zmq.POLLIN)
    deadline = time.monotonic() + timeout_s

    while True:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"tx_waveform.py exited with code {exit_code} before sending tx_started"
            )

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            pid = getattr(process, "pid", "unknown")
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s waiting for tx_started from tx_waveform.py "
                f"(pid={pid}). TX is still running but has not accepted its first payload; "
                "UHD startup or the first send may still be blocked."
            )

        timeout_ms = max(1, min(int(remaining_s * 1000), TX_NOTIFY_POLL_TIMEOUT_MS))
        if notify_socket not in dict(poller.poll(timeout_ms)):
            continue

        while True:
            event = recv_tx_event_nonblocking(notify_socket)
            if event is None:
                break

            event_name = str(event.get("event", ""))
            if event_name == "tx_started":
                return event
            if event_name == "tx_done":
                raise RuntimeError("Received tx_done before tx_started from tx_waveform.py")
            print(f"Warning: ignoring unexpected TX notify event before start: {event!r}")


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

    def _read_target_voltage_ack(
        self,
        *,
        expected_cmd: int,
        expected_value: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        original_timeout = self.serial_port.timeout
        short_timeout = 0.1 if original_timeout is None else min(float(original_timeout), 0.1)
        self.serial_port.timeout = short_timeout

        text = ""
        lines: list[str] = []
        ack_cmd: int | None = None
        ack_value: int | None = None
        ack_value_hex: int | None = None

        try:
            while time.monotonic() < deadline:
                waiting = getattr(self.serial_port, "in_waiting", 0)
                raw = self.serial_port.read(max(1, waiting))
                if not raw:
                    continue

                decoded = raw.decode("utf-8", errors="ignore")
                if not decoded:
                    continue

                text += decoded
                while "\n" in text:
                    line, text = text.split("\n", 1)
                    clean = line.strip()
                    if clean:
                        lines.append(clean)

                search_text = "\n".join([*lines, text])
                if ack_cmd is None:
                    cmd_match = EP_ACK_CMD_RE.search(search_text)
                    if cmd_match:
                        ack_cmd = int(cmd_match.group("cmd"), 16)

                if ack_value is None or ack_value_hex is None:
                    value_match = EP_ACK_VALUE_RE.search(search_text)
                    if value_match:
                        ack_value = int(value_match.group("value"), 10)
                        ack_value_hex = int(value_match.group("value_hex"), 16)

                if ack_cmd is not None and ack_value is not None and ack_value_hex is not None:
                    break
        finally:
            self.serial_port.timeout = original_timeout

        if text.strip():
            lines.append(text.strip())

        ack = {
            "expected_cmd": int(expected_cmd),
            "expected_value": int(expected_value),
            "cmd": ack_cmd,
            "value": ack_value,
            "value_hex": ack_value_hex,
            "lines": lines,
        }
        ack["ok"] = (
            ack_cmd == expected_cmd
            and ack_value == expected_value
            and ack_value_hex == expected_value
        )
        return ack

    def set_target_voltage(self, value: int, *, ack_timeout_s: float) -> dict[str, Any]:
        command = bytearray()
        command.append(START_BYTE)
        command.append(SET_TARGET_VOLTAGE_CMD)
        command.append(SET_TARGET_VOLTAGE_VALUE_SIZE)
        command += struct.pack(">I", value)
        command.append(0xFF)

        time.sleep(0.1)
        self.serial_port.reset_input_buffer()
        self.serial_port.write(command)
        self.serial_port.flush()

        ack = self._read_target_voltage_ack(
            expected_cmd=SET_TARGET_VOLTAGE_CMD,
            expected_value=value,
            timeout_s=ack_timeout_s,
        )
        if not ack["ok"]:
            seen = "; ".join(ack["lines"]) if ack["lines"] else "<no text ACK>"
            raise RuntimeError(
                "EP target voltage ACK mismatch: "
                f"expected CMD=0x{SET_TARGET_VOLTAGE_CMD:02X}, VALUE={value} (0x{value:08X}); "
                f"got CMD={ack['cmd']!r}, VALUE={ack['value']!r}, VALUE_HEX={ack['value_hex']!r}; "
                f"received: {seen}"
            )
        return ack


def collect_measurements_until_stop(
    profiler: EnergyProfiler,
    process: subprocess.Popen[bytes],
    notify_socket: zmq.Socket[Any],
    window_s: float,
) -> tuple[list[dict[str, int]], str, int]:
    deadline = time.monotonic() + window_s
    measurements: list[dict[str, int]] = []
    stop_reason = "window_elapsed"
    trimmed_count = 0
    original_timeout = profiler.serial_port.timeout
    short_timeout = EP_SERIAL_POLL_TIMEOUT_S if original_timeout is None else min(float(original_timeout), EP_SERIAL_POLL_TIMEOUT_S)
    profiler.serial_port.timeout = short_timeout

    try:
        while True:
            if time.monotonic() >= deadline:
                stop_reason = "window_elapsed"
                break

            while True:
                event = recv_tx_event_nonblocking(notify_socket)
                if event is None:
                    break
                event_name = str(event.get("event", ""))
                if event_name == "tx_done":
                    stop_reason = "tx_done"
                    break
                if event_name != "tx_started":
                    print(f"Warning: ignoring unexpected TX notify event during collection: {event!r}")
            if stop_reason == "tx_done":
                break

            if process.poll() is not None:
                stop_reason = "tx_process_exited"
                break

            measurement = profiler.get_measurement()
            if measurement is not None:
                measurements.append(measurement)

            while True:
                event = recv_tx_event_nonblocking(notify_socket)
                if event is None:
                    break
                event_name = str(event.get("event", ""))
                if event_name == "tx_done":
                    stop_reason = "tx_done"
                    break
                if event_name != "tx_started":
                    print(f"Warning: ignoring unexpected TX notify event during collection: {event!r}")
            if stop_reason == "tx_done":
                break

        if stop_reason == "tx_done":
            measurements, trimmed_count = trim_trailing_samples(measurements)
    finally:
        profiler.serial_port.timeout = original_timeout

    return measurements, stop_reason, trimmed_count


def launch_tx_process(
    python_executable: str,
    tx_script: Path,
    tone: int,
    bandwidth_khz: int,
    gain_db: float,
    duration_s: float,
    *,
    closest_gain_match: bool = False,
    tx_notify_endpoint: str | None = None,
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
    if tx_notify_endpoint is not None:
        command.extend(["--tx-notify-endpoint", tx_notify_endpoint])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Launching TX: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


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


def stream_process_output(
    process: subprocess.Popen[bytes],
    *,
    prefix: str,
) -> threading.Thread | None:
    if process.stdout is None:
        return None

    def worker() -> None:
        assert process.stdout is not None
        for raw_line in iter(process.stdout.readline, b""):
            text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                print(f"{prefix}{text}", flush=True)
        process.stdout.close()

    thread = threading.Thread(target=worker, name=f"{prefix.strip()} output", daemon=True)
    thread.start()
    return thread


def join_process_output_thread(thread: threading.Thread | None) -> None:
    if thread is not None:
        thread.join(timeout=2.0)


def run_sweep(args: argparse.Namespace) -> int:
    profiler = EnergyProfiler(args.port, args.baudrate, args.serial_timeout)
    zmq_context = zmq.Context()
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
        target_voltage_ack = None
        if args.target_voltage is not None:
            target_voltage_ack = profiler.set_target_voltage(
                args.target_voltage,
                ack_timeout_s=args.target_voltage_ack_timeout,
            )
            for ack_line in target_voltage_ack["lines"]:
                print(f"EP ACK: {ack_line}")
            print(
                "Set EP target voltage to "
                f"{args.target_voltage} mV; ACK CMD=0x{target_voltage_ack['cmd']:02X}, "
                f"VALUE={target_voltage_ack['value']} (0x{target_voltage_ack['value_hex']:08X})"
            )

        print(f"Using {len(gains)} gain value(s): {gains}")
        print(f"Gain source: {gain_source}")
        print(
            "Sweep order per gain: "
            + ", ".join(tone_label(tone) for tone in args.tones)
        )

        for entry in sweep_plan:
            tone = int(entry["tone"])
            gain = float(entry["gain_db"])
            measure_window_s = derive_measure_window_s(args.tx_duration)
            if measure_window_s <= 0:
                raise ValueError(
                    f"Derived measurement window must be > 0, got {measure_window_s!r}"
                )
            notify_socket, notify_endpoint = create_tx_notify_socket(zmq_context)
            print(
                f"Starting sweep: {entry['waveform_label']}, gain={gain:g} dB, iq_file={Path(entry['iq_file']).name}"
            )
            process: subprocess.Popen[bytes] | None = None
            tx_output_thread: threading.Thread | None = None
            try:
                process = launch_tx_process(
                    python_executable=args.python,
                    tx_script=tx_script,
                    tone=tone,
                    bandwidth_khz=args.bw,
                    gain_db=gain,
                    duration_s=args.tx_duration,
                    closest_gain_match=closest_gain_match,
                    tx_notify_endpoint=notify_endpoint,
                )
                tx_output_thread = stream_process_output(process, prefix="[tx_waveform] ")

                started_at = utc_now_iso()
                sweep_started = time.monotonic()
                try:
                    wait_for_tx_started(
                        process,
                        notify_socket,
                        args.tx_start_timeout,
                    )
                    print(
                        f"Received tx_started for {entry['waveform_label']}, gain={gain:g} dB; "
                        f"starting EP collection for up to {measure_window_s:.3f}s"
                    )
                    readings, stop_reason, trimmed_count = collect_measurements_until_stop(
                        profiler,
                        process,
                        notify_socket,
                        measure_window_s,
                    )
                    if stop_reason == "tx_done":
                        print(
                            f"Received tx_done while collecting {entry['waveform_label']}, gain={gain:g} dB; "
                            f"trimmed {trimmed_count} trailing sample(s)."
                        )
                    elif stop_reason == "tx_process_exited":
                        print(
                            f"TX process exited before tx_done for {entry['waveform_label']}, gain={gain:g} dB; "
                            "stopping EP collection without trim."
                        )
                    else:
                        print(
                            f"EP collection completed full derived window for {entry['waveform_label']}, "
                            f"gain={gain:g} dB."
                        )
                    exit_code = wait_for_process(process)
                    join_process_output_thread(tx_output_thread)
                except Exception:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    join_process_output_thread(tx_output_thread)
                    raise
            finally:
                notify_socket.close(0)

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
                "pre_measure_delay_s": 0.0,
                "measure_window_s": measure_window_s,
                "sweep_duration_s": round(sweep_duration_s, 3),
                "reading_count": len(readings),
                "readings": readings,
                "target_voltage_mv": args.target_voltage,
                "target_voltage_ack": target_voltage_ack,
            }
            append_jsonl(args.output, record)
            completed_sweeps += 1
            print(
                f"Stored {len(readings)} readings for {entry['waveform_label']}, gain={gain:g} dB "
                f"to {args.output}"
            )
    finally:
        profiler.close()
        zmq_context.term()

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
        "--tx-start-timeout",
        type=float,
        default=DEFAULT_TX_START_TIMEOUT_S,
        help=(
            "How long to wait for tx_waveform.py to emit tx_started over ZMQ before failing the sweep "
            f"(default: {DEFAULT_TX_START_TIMEOUT_S:g})"
        ),
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
        "--target-voltage-ack-timeout",
        type=float,
        default=DEFAULT_TARGET_VOLTAGE_ACK_TIMEOUT_S,
        help=(
            "Seconds to wait for the EP CMD/VALUE acknowledgement after --target-voltage "
            f"(default: {DEFAULT_TARGET_VOLTAGE_ACK_TIMEOUT_S:g})"
        ),
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
        "--log-file",
        type=Path,
        default=None,
        help="Text log path. Relative paths resolve inside results/<run-id>/ (default: measure_ep_power.log).",
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


def run_measurement(args: argparse.Namespace, *, run_id: str, results_dir: Path) -> int:
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
    if args.tx_start_timeout <= 0:
        raise ValueError("--tx-start-timeout must be > 0")
    if args.serial_timeout <= 0:
        raise ValueError("--serial-timeout must be > 0")
    if args.target_voltage_ack_timeout <= 0:
        raise ValueError("--target-voltage-ack-timeout must be > 0")
    if args.target_voltage is not None and not (0 <= args.target_voltage <= 0xFFFFFFFF):
        raise ValueError("--target-voltage must be in [0, 4294967295]")
    if not Path(args.python).exists() and shutil.which(args.python) is None:
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    manifest_path = write_manifest(
        results_dir=results_dir,
        run_id=run_id,
        script_name=Path(__file__).name,
        argv=sys.argv[1:],
        extra={"output": str(args.output), "log_file": str(args.log_file)},
    )
    print(f"Updated run manifest: {manifest_path}")

    return run_sweep(args)


def main() -> int:
    args = build_arg_parser().parse_args()
    run_id = args.run_id if args.run_id else timestamp_run_id()
    results_dir = args.results_dir.resolve()
    args.log_file = resolve_run_log_path(
        args.log_file,
        results_dir=results_dir,
        run_id=run_id,
    )

    with args.log_file.open("a", encoding="utf-8", buffering=1) as log_handle:
        stdout = TeeTextIO(sys.stdout, log_handle)
        stderr = TeeTextIO(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Logging to {args.log_file}")
            try:
                return run_measurement(args, run_id=run_id, results_dir=results_dir)
            except KeyboardInterrupt:
                print("Interrupted by user.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
