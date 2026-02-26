#!/usr/bin/env python3
"""Replay an exported IQ waveform NPZ on a USRP using UHD.

Required arguments are intentionally minimal:
- --tone: number of tones (N in filename)
- --bw: bandwidth in kHz (BW in filename)
- --gain: TX gain in dB
- --duration: approximate replay time in seconds
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np


IQ_DIR = Path("tx_iq")
IQ_BW_REGEX = re.compile(r"^iq_N(?P<n>\d+)_BW(?P<bw>\d+)kHz(?:_.*)?\.npz$")
# Use UHD defaults by default; aggressive frame settings can trigger USB NO_MEM on some hosts.
DEFAULT_UHD_ARGS = ""
DEFAULT_SEND_TIMEOUT_S = 10.0
DEFAULT_SPB = 4096
START_DELAY_S = 0.5
SETTLE_DELAY_S = 1.0
TX_CHANNEL = 0
TX_ANTENNA_PREFERRED = "TX/RX"
MAX_ZERO_SENDS = 16


def _set_with_channel(fn, value, channel: int) -> None:
    try:
        fn(value, channel)
    except TypeError:
        fn(value)


def _get_with_channel(fn, channel: int):
    try:
        return fn(channel)
    except TypeError:
        return fn()


def _make_tune_request(uhd_module, freq_hz: float):
    try:
        return uhd_module.types.TuneRequest(freq_hz)
    except Exception:
        try:
            return uhd_module.libpyuhd.types.tune_request(freq_hz)
        except Exception:
            return freq_hz


def _tx_send(tx_stream, samples, md, timeout_s: float) -> int:
    try:
        return int(tx_stream.send(samples, md, timeout_s))
    except TypeError:
        return int(tx_stream.send(samples, md))


def try_set_thread_priority(uhd_module) -> bool:
    candidates = [
        lambda: uhd_module.utils.set_thread_priority_safe(),
        lambda: uhd_module.set_thread_priority_safe(),
    ]
    for setter in candidates:
        try:
            result = setter()
            if result is None or bool(result):
                return True
        except Exception:
            continue
    return False


def _parse_tone_bw_from_iq_name(path: Path) -> tuple[int, int] | None:
    match = IQ_BW_REGEX.match(path.name)
    if not match:
        return None
    return int(match.group("n")), int(match.group("bw"))


def find_iq_file_for_tone_bw(iq_dir: Path, tone: int, bw_khz: int) -> Path:
    if not iq_dir.exists():
        raise FileNotFoundError(f"IQ directory not found: {iq_dir.resolve()}")

    matches = []
    for path in sorted(iq_dir.glob("iq_N*_BW*kHz*.npz")):
        parsed = _parse_tone_bw_from_iq_name(path)
        if parsed is None:
            continue
        n_tones, bw = parsed
        if n_tones == tone and bw == bw_khz:
            matches.append(path)

    if not matches:
        available = sorted(
            {
                parsed
                for path in iq_dir.glob("iq_N*_BW*kHz*.npz")
                for parsed in [_parse_tone_bw_from_iq_name(path)]
                if parsed is not None
            }
        )
        raise FileNotFoundError(
            f"No IQ file found for tone={tone}, bw={bw_khz}kHz in {iq_dir.resolve()} "
            f"(expected pattern iq_N<tone>_BW<bw>kHz_*.npz). "
            f"Available (tone,bw_kHz): {available}"
        )

    if len(matches) == 1:
        return matches[0].resolve()

    preferred = [path for path in matches if "_col0" in path.stem]
    if len(preferred) == 1:
        return preferred[0].resolve()

    raise RuntimeError(
        f"Multiple IQ files found for tone={tone}, bw={bw_khz}kHz: {[p.name for p in matches]}. "
        "Keep one file per tone/bw (or one _col0 file) in tx_iq/."
    )


def load_iq_file(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required_keys = ("iq", "sample_rate_hz", "center_freq_hz")
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise KeyError(f"{path.name} missing required key(s): {missing}")

        iq = np.asarray(data["iq"]).squeeze()
        sample_rate_hz = float(np.asarray(data["sample_rate_hz"]).squeeze())
        center_freq_hz = float(np.asarray(data["center_freq_hz"]).squeeze())

    if iq.ndim != 1:
        raise ValueError(f"{path.name} IQ must be 1D after squeeze, got shape={iq.shape}")
    if iq.size == 0:
        raise ValueError(f"{path.name} IQ array is empty")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"{path.name} sample_rate_hz must be > 0, got {sample_rate_hz!r}")
    if not np.isfinite(center_freq_hz) or center_freq_hz <= 0:
        raise ValueError(f"{path.name} center_freq_hz must be > 0, got {center_freq_hz!r}")

    if not np.iscomplexobj(iq):
        iq = iq.astype(np.float64) + 0j
    iq = np.ascontiguousarray(iq.astype(np.complex64, copy=False))

    peak = float(np.max(np.abs(iq)))
    avg_power = float(np.mean(np.abs(iq) ** 2))
    if peak > 1.0 + 1e-6:
        raise ValueError(
            f"{path.name} has |IQ| peak={peak:.6g} > 1.0. "
            "Re-export with proper normalization to avoid DAC overdrive."
        )

    return iq, sample_rate_hz, center_freq_hz, avg_power, peak


def _get_tx_antennas(usrp, channel: int) -> list[str]:
    try:
        antennas = usrp.get_tx_antennas(channel)
    except TypeError:
        antennas = usrp.get_tx_antennas()
    return list(antennas)


def setup_usrp(uhd_module, uhd_args: str, sample_rate_hz: float, center_freq_hz: float, tx_gain_db: float):
    try:
        usrp = uhd_module.usrp.MultiUSRP(uhd_args)
    except Exception as exc:
        raise RuntimeError(f"Failed to create UHD MultiUSRP with --uhd-args '{uhd_args}': {exc}") from exc

    _set_with_channel(usrp.set_tx_rate, sample_rate_hz, TX_CHANNEL)
    _set_with_channel(usrp.set_tx_freq, _make_tune_request(uhd_module, center_freq_hz), TX_CHANNEL)
    _set_with_channel(usrp.set_tx_gain, tx_gain_db, TX_CHANNEL)
    _set_with_channel(usrp.set_tx_bandwidth, sample_rate_hz, TX_CHANNEL)

    antennas = _get_tx_antennas(usrp, TX_CHANNEL)
    antenna = TX_ANTENNA_PREFERRED if TX_ANTENNA_PREFERRED in antennas else (
        antennas[0] if antennas else TX_ANTENNA_PREFERRED
    )
    _set_with_channel(usrp.set_tx_antenna, antenna, TX_CHANNEL)

    stream_args = uhd_module.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [TX_CHANNEL]
    tx_stream = usrp.get_tx_stream(stream_args)
    return usrp, tx_stream, antenna


def make_start_metadata(uhd_module, usrp, start_delay_s: float):
    md = uhd_module.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False
    if start_delay_s > 0:
        try:
            now = usrp.get_time_now().get_real_secs()
            md.time_spec = uhd_module.types.TimeSpec(now + start_delay_s)
            md.has_time_spec = True
        except Exception:
            # Fallback when timed metadata APIs differ by UHD version.
            time.sleep(start_delay_s)
            md.has_time_spec = False
    return md


def send_buffered(tx_stream, samples: np.ndarray, md, spb: int, send_timeout_s: float) -> None:
    offset = 0
    total = int(samples.size)
    first = True
    zero_sends = 0

    while offset < total:
        n = min(spb, total - offset)
        chunk = samples[offset : offset + n]
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk)

        sent = _tx_send(tx_stream, chunk, md, send_timeout_s)
        if sent < 0:
            raise RuntimeError(f"TX send failed: requested={n}, sent={sent}")
        if sent == 0:
            zero_sends += 1
            if zero_sends >= MAX_ZERO_SENDS:
                raise RuntimeError(
                    f"TX send stalled: sent 0 samples for {zero_sends} consecutive send() calls"
                )
            continue
        if sent != n:
            raise RuntimeError(
                f"TX send timed out/partial send: requested={n}, sent={sent}"
            )

        zero_sends = 0
        offset += sent

        if first:
            md.start_of_burst = False
            md.has_time_spec = False
            first = False


def send_eob(uhd_module, tx_stream) -> None:
    md = uhd_module.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = True
    md.has_time_spec = False

    empty = np.zeros(0, dtype=np.complex64)
    fallback = np.zeros(1, dtype=np.complex64)
    try:
        tx_stream.send(empty, md)
    except Exception:
        tx_stream.send(fallback, md)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay IQ waveform NPZ on a USRP using UHD")
    parser.add_argument("--tone", required=True, type=int, help="Tone count N (matches iq_N<N>_...)")
    parser.add_argument("--bw", required=True, type=int, help="Signal bandwidth in kHz (matches _BW<bw>kHz_)")
    parser.add_argument("--gain", required=True, type=float, help="TX gain in dB")
    parser.add_argument("--duration", default=10.0, type=float, help="Approximate replay duration in seconds")
    parser.add_argument("--uhd-args", default=DEFAULT_UHD_ARGS, type=str, help="UHD device args string")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.tone < 0:
        raise ValueError("--tone must be >= 0")
    if args.bw <= 0:
        raise ValueError("--bw must be > 0")
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")

    iq_file = find_iq_file_for_tone_bw(IQ_DIR, args.tone, args.bw)
    iq, sample_rate_hz, center_freq_hz, avg_power, peak = load_iq_file(iq_file)

    print(f"Selected IQ file: {iq_file}")
    print(f"IQ stats: samples={iq.size}, avg_power={avg_power:.6g}, peak={peak:.6g}")

    try:
        import uhd
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python UHD module `uhd`. Install UHD Python bindings."
        ) from exc

    rt_ok = try_set_thread_priority(uhd)
    print(f"Thread priority: {'realtime-enabled' if rt_ok else 'default'}")

    usrp, tx_stream, antenna = setup_usrp(uhd, args.uhd_args, sample_rate_hz, center_freq_hz, args.gain)
    time.sleep(SETTLE_DELAY_S)
    actual_rate = float(_get_with_channel(usrp.get_tx_rate, TX_CHANNEL))
    actual_freq = float(_get_with_channel(usrp.get_tx_freq, TX_CHANNEL))
    actual_gain = float(_get_with_channel(usrp.get_tx_gain, TX_CHANNEL))
    max_samps = int(tx_stream.get_max_num_samps())
    if max_samps <= 0:
        raise RuntimeError(f"Invalid tx_stream.get_max_num_samps()={max_samps}")
    spb = max(1, min(DEFAULT_SPB, max_samps, iq.size))

    if not np.isclose(actual_rate, sample_rate_hz, rtol=1e-3, atol=1.0):
        print(
            f"WARNING: requested sample_rate_hz={sample_rate_hz:.3f}, "
            f"device actual_rate={actual_rate:.3f}. Duration estimate uses actual rate."
        )

    # Use the hardware-coerced rate for repeat timing so requested duration is closer.
    n_repeats = max(1, int(math.ceil(args.duration * actual_rate / iq.size)))
    actual_duration = n_repeats * iq.size / actual_rate
    print(
        f"TX settings: rate={actual_rate:.3f} Sa/s, "
        f"freq={actual_freq / 1e6:.6f} MHz, gain={actual_gain:.2f} dB, antenna={antenna}"
    )
    print(
        f"Playback: requested={args.duration:.6f}s, actual={actual_duration:.6f}s, "
        f"repeats={n_repeats}, spb={spb}, streamer_max_samps={max_samps}, "
        f"start_delay={START_DELAY_S:.3f}s, uhd_args='{args.uhd_args}'"
    )

    md = make_start_metadata(uhd, usrp, START_DELAY_S)
    try:
        for _ in range(n_repeats):
            send_buffered(tx_stream, iq, md, spb, DEFAULT_SEND_TIMEOUT_S)
    finally:
        print("Stopping TX (sending end-of-burst)...")
        try:
            send_eob(uhd, tx_stream)
        except Exception as exc:
            print(f"WARNING: Failed to send EOB cleanly: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
