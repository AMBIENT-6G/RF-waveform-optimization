#!/usr/bin/env python3
"""Replay preprocessed IQ blocks on a USRP B210.

This script expects IQ files exported by the notebook in `tx_iq/`, then replays
one selected tone block for a fixed duration.

Arguments kept intentionally minimal:
- `--tone`    : which exported tone set to play (0, 4, 8, 16, 32)
- `--tx-gain` : TX gain in dB
- `--duration`: playback time in seconds
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np


IQ_DIR = Path("tx_iq")
ALLOWED_TONES = (0, 4, 8, 16, 32)

# Fixed radio settings for simplicity.
USRP_ARGS = ""
CHANNEL = 0
TX_ANTENNA = "TX/RX"
START_DELAY_S = 0.2


def _set_with_channel(fn, value, channel):
    try:
        fn(value, channel)
    except TypeError:
        fn(value)


def _get_with_channel(fn, channel):
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


def find_iq_file_for_tone(iq_dir: Path, tone: int) -> Path:
    if not iq_dir.exists():
        raise FileNotFoundError(f"IQ directory not found: {iq_dir.resolve()}")

    matches = sorted(iq_dir.glob(f"iq_N{tone}_*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No IQ file found for tone {tone} in {iq_dir.resolve()} "
            f"(expected pattern iq_N{tone}_*.npz)"
        )

    if len(matches) == 1:
        return matches[0].resolve()

    # Prefer col0 for multitone if several exports are present.
    preferred = [p for p in matches if "_col0" in p.stem]
    if len(preferred) == 1:
        return preferred[0].resolve()

    raise RuntimeError(
        f"Multiple IQ files found for tone {tone}: {[p.name for p in matches]}. "
        "Keep one file per tone (or only col0) in tx_iq/."
    )


def load_iq_file(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "iq" not in data.files:
            raise KeyError(f"{path} missing required array: iq")

        iq = np.asarray(data["iq"]).squeeze()
        fs_hz = float(np.asarray(data["sample_rate_hz"]).squeeze()) if "sample_rate_hz" in data.files else 20e6
        fc_hz = float(np.asarray(data["center_freq_hz"]).squeeze()) if "center_freq_hz" in data.files else 875e6

    if iq.ndim != 1:
        raise ValueError(f"{path} IQ must be 1D after squeeze, got shape {iq.shape}")
    if iq.size == 0:
        raise ValueError(f"{path} IQ array is empty")

    if not np.iscomplexobj(iq):
        iq = iq.astype(np.float64) + 0j
    iq = iq.astype(np.complex64, copy=False)

    peak = float(np.max(np.abs(iq)))
    power = float(np.mean(np.abs(iq) ** 2))
    if peak > 1.0 + 1e-6:
        raise ValueError(
            f"{path.name} has |IQ| peak={peak:.6g} > 1.0. "
            "Re-export IQ from notebook with proper normalization."
        )

    return iq, fs_hz, fc_hz, power, peak


def setup_usrp(uhd_module, sample_rate_hz: float, center_freq_hz: float, tx_gain_db: float):
    usrp = uhd_module.usrp.MultiUSRP(USRP_ARGS)

    _set_with_channel(usrp.set_tx_rate, sample_rate_hz, CHANNEL)
    _set_with_channel(usrp.set_tx_freq, _make_tune_request(uhd_module, center_freq_hz), CHANNEL)
    _set_with_channel(usrp.set_tx_gain, tx_gain_db, CHANNEL)
    _set_with_channel(usrp.set_tx_bandwidth, sample_rate_hz, CHANNEL)
    _set_with_channel(usrp.set_tx_antenna, TX_ANTENNA, CHANNEL)

    stream_args = uhd_module.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]
    tx_stream = usrp.get_tx_stream(stream_args)

    return usrp, tx_stream


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
            time.sleep(start_delay_s)
            md.has_time_spec = False

    return md


def send_buffered(tx_stream, samples: np.ndarray, md, max_samps: int) -> None:
    offset = 0
    first = True
    total = samples.size
    zero_sends = 0

    while offset < total:
        n = min(max_samps, total - offset)
        sent = int(tx_stream.send(samples[offset : offset + n], md))
        if sent < 0:
            raise RuntimeError(f"TX send failed: requested {n}, sent {sent}")
        if sent == 0:
            zero_sends += 1
            if zero_sends >= 16:
                raise RuntimeError(
                    f"TX send stalled: requested {n}, sent 0 for {zero_sends} consecutive attempts"
                )
            continue

        zero_sends = 0
        offset += sent

        # SOB/time_spec must only be attached to the first successfully queued samples.
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
    try:
        tx_stream.send(empty, md)
    except Exception:
        tx_stream.send(np.zeros(1, dtype=np.complex64), md)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay one exported IQ tone block on B210")
    p.add_argument("--tone", type=int, required=True, choices=ALLOWED_TONES, help="Tone set to play")
    p.add_argument("--tx-gain", type=float, required=True, help="TX gain in dB")
    p.add_argument("--duration", type=float, default=10.0, help="Playback duration in seconds")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")

    iq_file = find_iq_file_for_tone(IQ_DIR, args.tone)
    iq, sample_rate_hz, center_freq_hz, power, peak = load_iq_file(iq_file)

    print(f"Selected IQ file: {iq_file}")
    print(f"IQ stats: samples={iq.size}, avg_power={power:.6g}, peak={peak:.6g}")
    print(
        f"Replay settings: fs={sample_rate_hz:.3f} Sa/s, fc={center_freq_hz/1e6:.6f} MHz, "
        f"tx_gain={args.tx_gain:.2f} dB"
    )

    try:
        import uhd
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python UHD module `uhd`. Install UHD Python bindings."
        ) from exc

    usrp, tx_stream = setup_usrp(uhd, sample_rate_hz, center_freq_hz, args.tx_gain)

    actual_rate = _get_with_channel(usrp.get_tx_rate, CHANNEL)
    actual_freq = _get_with_channel(usrp.get_tx_freq, CHANNEL)
    actual_gain = _get_with_channel(usrp.get_tx_gain, CHANNEL)
    print(
        f"USRP configured: rate={actual_rate:.3f} Sa/s, "
        f"freq={actual_freq/1e6:.6f} MHz, gain={actual_gain:.2f} dB"
    )

    max_samps = int(tx_stream.get_max_num_samps())  
    n_repeats = max(1, int(math.ceil(args.duration * sample_rate_hz / iq.size)))
    actual_duration = n_repeats * iq.size / sample_rate_hz
    print(
        f"Playback: requested={args.duration:.6f}s, actual={actual_duration:.6f}s, "
        f"repeats={n_repeats}, max_samps_per_send={max_samps}"
    )

    md = make_start_metadata(uhd, usrp, START_DELAY_S)

    try:
        for _ in range(n_repeats):
            send_buffered(tx_stream, iq, md, max_samps)
    finally:
        print("Stopping TX (sending end-of-burst)...")
        try:
            send_eob(uhd, tx_stream)
        except Exception as exc:
            print(f"WARNING: Failed to send EOB cleanly: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
