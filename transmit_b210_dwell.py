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
ALLOWED_CHANNELS = (0, 1)

# Fixed radio settings for simplicity.
USRP_ARGS = ""
DEFAULT_CHANNEL = 0
TX_ANTENNA = "TX/RX"
START_DELAY_S = 0.2
DEFAULT_CHUNK_MULT = 64
DEFAULT_SEND_TIMEOUT_S = 0.5
DEFAULT_OVERSAMPLE = 2


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


def _set_tx_subdev_for_b210(uhd_module, usrp) -> str:
    # Keep both TX frontends active so channel 1 exists as index 1.
    subdev = "A:A A:B"
    try:
        spec = uhd_module.usrp.SubdevSpec(subdev)
        usrp.set_tx_subdev_spec(spec)
    except Exception:
        try:
            usrp.set_tx_subdev_spec(subdev)
        except Exception:
            return "default"
    return subdev


def _get_tx_antennas(usrp, channel: int):
    try:
        antennas = usrp.get_tx_antennas(channel)
    except TypeError:
        antennas = usrp.get_tx_antennas()
    return list(antennas)


def _get_tx_num_channels(usrp) -> int:
    try:
        return int(usrp.get_tx_num_channels())
    except Exception:
        return 1


def _tx_send(tx_stream, samples, md, timeout_s: float) -> int:
    try:
        return int(tx_stream.send(samples, md, timeout_s))
    except TypeError:
        return int(tx_stream.send(samples, md))


def _set_tx_dc_offset(usrp, offset: complex, channel: int) -> None:
    try:
        usrp.set_tx_dc_offset(offset, channel)
    except TypeError:
        usrp.set_tx_dc_offset(offset)


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


def remove_iq_dc(samples: np.ndarray) -> tuple[np.ndarray, complex]:
    dc = complex(np.mean(samples))
    out = (samples - dc).astype(np.complex64, copy=False)
    return out, dc


def oversample_periodic_iq(samples: np.ndarray, sample_rate_hz: float, factor: int) -> tuple[np.ndarray, float]:
    if factor <= 1:
        return samples.astype(np.complex64, copy=False), float(sample_rate_hz)

    x = np.asarray(samples, dtype=np.complex64).ravel()
    n = int(x.size)
    if n < 2:
        raise ValueError("Need at least 2 IQ samples for oversampling")

    m = int(n * factor)
    x_f = np.fft.fftshift(np.fft.fft(x, n=n))
    pad_total = m - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    y_f = np.pad(x_f, (pad_left, pad_right), mode="constant")

    # NumPy ifft uses 1/M scaling, so multiply by factor=M/N to preserve amplitude.
    y = np.fft.ifft(np.fft.ifftshift(y_f), n=m) * float(factor)
    return y.astype(np.complex64), float(sample_rate_hz) * float(factor)


def apply_baseband_freq_shift(samples: np.ndarray, sample_rate_hz: float, shift_hz: float) -> np.ndarray:
    if abs(shift_hz) < 1e-12:
        return samples
    n = np.arange(samples.size, dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * shift_hz * (n / float(sample_rate_hz))).astype(np.complex64)
    out = (samples * rot).astype(np.complex64, copy=False)
    return out


def setup_usrp(
    uhd_module,
    sample_rate_hz: float,
    tune_freq_hz: float,
    tx_gain_db: float,
    channels: list[int],
    tx_dc_offset: complex | None = None,
):
    usrp = uhd_module.usrp.MultiUSRP(USRP_ARGS)
    subdev = _set_tx_subdev_for_b210(uhd_module, usrp)
    num_channels = _get_tx_num_channels(usrp)
    if any(ch < 0 or ch >= num_channels for ch in channels):
        raise ValueError(
            f"Requested TX channels {channels}, but device exposes {num_channels} TX channel(s) "
            f"(valid: 0..{max(0, num_channels - 1)}; subdev_spec={subdev})"
        )

    channel_antennas = {}
    for channel in channels:
        antennas = _get_tx_antennas(usrp, channel)
        tx_antenna = TX_ANTENNA if TX_ANTENNA in antennas else (antennas[0] if antennas else TX_ANTENNA)

        _set_with_channel(usrp.set_tx_rate, sample_rate_hz, channel)
        _set_with_channel(usrp.set_tx_freq, _make_tune_request(uhd_module, tune_freq_hz), channel)
        _set_with_channel(usrp.set_tx_gain, tx_gain_db, channel)
        _set_with_channel(usrp.set_tx_bandwidth, sample_rate_hz, channel)
        _set_with_channel(usrp.set_tx_antenna, tx_antenna, channel)
        if tx_dc_offset is not None:
            _set_tx_dc_offset(usrp, tx_dc_offset, channel)
        channel_antennas[channel] = tx_antenna

    stream_args = uhd_module.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = channels
    tx_stream = usrp.get_tx_stream(stream_args)

    return usrp, tx_stream, subdev, channel_antennas


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


def send_buffered(
    tx_stream,
    samples: np.ndarray,
    md,
    max_samps: int,
    send_chunk_samps: int,
    send_timeout_s: float,
) -> None:
    offset = 0
    first = True
    total = int(samples.shape[-1]) if samples.ndim > 1 else int(samples.size)
    zero_sends = 0
    target_samps = max(max_samps, int(send_chunk_samps))

    while offset < total:
        n = min(target_samps, total - offset)
        chunk = samples[:, offset : offset + n] if samples.ndim > 1 else samples[offset : offset + n]
        if chunk.ndim > 1 and not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk)
        sent = _tx_send(tx_stream, chunk, md, send_timeout_s)
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


def send_eob(uhd_module, tx_stream, num_channels: int) -> None:
    md = uhd_module.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = True
    md.has_time_spec = False

    if num_channels > 1:
        empty = np.zeros((num_channels, 0), dtype=np.complex64)
        fallback = np.zeros((num_channels, 1), dtype=np.complex64)
    else:
        empty = np.zeros(0, dtype=np.complex64)
        fallback = np.zeros(1, dtype=np.complex64)

    try:
        tx_stream.send(empty, md)
    except Exception:
        tx_stream.send(fallback, md)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay one exported IQ tone block on B210")
    p.add_argument("--tone", type=int, required=True, choices=ALLOWED_TONES, help="Tone set to play")
    p.add_argument("--tx-gain", type=float, required=True, help="TX gain in dB")
    p.add_argument("--duration", type=float, default=10.0, help="Playback duration in seconds")
    p.add_argument(
        "--oversample",
        type=int,
        default=DEFAULT_OVERSAMPLE,
        help=(
            "Integer TX oversampling factor for edge protection. "
            "Upsamples IQ and scales USRP tx rate by this factor."
        ),
    )
    p.add_argument(
        "--remove-dc",
        action="store_true",
        help="Subtract complex mean from IQ before TX (helps suppress LO leakage)",
    )
    p.add_argument(
        "--lo-offset-hz",
        type=float,
        default=0.0,
        help=(
            "RF LO offset in Hz for leakage mitigation. "
            "TX is tuned to fc + offset and IQ is digitally shifted by -offset."
        ),
    )
    p.add_argument("--tx-dc-i", type=float, default=None, help="Manual TX DC correction, I component (-1.0..1.0)")
    p.add_argument("--tx-dc-q", type=float, default=None, help="Manual TX DC correction, Q component (-1.0..1.0)")
    p.add_argument(
        "--start-delay",
        type=float,
        default=START_DELAY_S,
        help="Timed TX start offset in seconds to allow host queue prefill",
    )
    p.add_argument(
        "--chunk-mult",
        type=int,
        default=DEFAULT_CHUNK_MULT,
        help="Requested send chunk size multiplier relative to UHD max_num_samps",
    )
    p.add_argument(
        "--send-timeout",
        type=float,
        default=DEFAULT_SEND_TIMEOUT_S,
        help="Timeout (seconds) for each tx_stream.send() call",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        choices=ALLOWED_CHANNELS,
        help="Single TX channel index on B210 (ignored when --both-channels is set)",
    )
    p.add_argument(
        "--both-channels",
        action="store_true",
        help="Transmit the same IQ stream on TX channels 0 and 1 simultaneously",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.start_delay < 0:
        raise ValueError("--start-delay must be >= 0")
    if args.chunk_mult <= 0:
        raise ValueError("--chunk-mult must be > 0")
    if args.send_timeout <= 0:
        raise ValueError("--send-timeout must be > 0")
    if args.oversample < 1:
        raise ValueError("--oversample must be >= 1")
    tx_channels = [0, 1] if args.both_channels else [args.channel]

    iq_file = find_iq_file_for_tone(IQ_DIR, args.tone)
    iq, sample_rate_hz, center_freq_hz, power, peak = load_iq_file(iq_file)
    raw_num_samples = int(iq.size)
    source_rate_hz = float(sample_rate_hz)
    if args.oversample > 1:
        iq, sample_rate_hz = oversample_periodic_iq(iq, sample_rate_hz, args.oversample)

    if abs(args.lo_offset_hz) >= 0.5 * sample_rate_hz:
        raise ValueError(
            f"|--lo-offset-hz| must be < fs/2 ({0.5 * sample_rate_hz:.3f} Hz), got {args.lo_offset_hz:.3f} Hz"
        )

    tx_dc_offset = None
    if (args.tx_dc_i is None) ^ (args.tx_dc_q is None):
        raise ValueError("Provide both --tx-dc-i and --tx-dc-q, or neither.")
    if args.tx_dc_i is not None and args.tx_dc_q is not None:
        tx_dc_offset = complex(float(args.tx_dc_i), float(args.tx_dc_q))

    print(f"Selected IQ file: {iq_file}")
    print(f"IQ stats (raw): samples={raw_num_samples}, avg_power={power:.6g}, peak={peak:.6g}")

    dc_removed = 0.0 + 0.0j
    if args.remove_dc:
        iq, dc_removed = remove_iq_dc(iq)

    if abs(args.lo_offset_hz) > 0:
        iq = apply_baseband_freq_shift(iq, sample_rate_hz, shift_hz=-args.lo_offset_hz)

    proc_power = float(np.mean(np.abs(iq) ** 2))
    proc_peak = float(np.max(np.abs(iq)))
    if proc_peak > 1.0 + 1e-6:
        scale = (1.0 - 1e-7) / proc_peak
        iq = (iq * scale).astype(np.complex64, copy=False)
        proc_power = float(np.mean(np.abs(iq) ** 2))
        proc_peak = float(np.max(np.abs(iq)))
        print(f"WARNING: IQ peak exceeded 1 after processing; scaled by {scale:.6g}")

    rf_tune_hz = center_freq_hz + args.lo_offset_hz
    print(
        f"Replay settings: fs_file={source_rate_hz:.3f} Sa/s, fs_tx={sample_rate_hz:.3f} Sa/s, "
        f"oversample={args.oversample}x, fc={center_freq_hz/1e6:.6f} MHz, "
        f"rf_tune={rf_tune_hz/1e6:.6f} MHz, lo_offset={args.lo_offset_hz:.3f} Hz, "
        f"tx_gain={args.tx_gain:.2f} dB, channels={tx_channels}, "
        f"start_delay={args.start_delay:.3f}s, chunk_mult={args.chunk_mult}, "
        f"send_timeout={args.send_timeout:.3f}s"
    )
    print(
        f"IQ stats (processed): avg_power={proc_power:.6g}, peak={proc_peak:.6g}, "
        f"dc_removed=({dc_removed.real:.6g}, {dc_removed.imag:.6g}), "
        f"manual_tx_dc_offset={tx_dc_offset if tx_dc_offset is not None else 'none'}"
    )

    try:
        import uhd
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python UHD module `uhd`. Install UHD Python bindings."
        ) from exc

    usrp, tx_stream, subdev, channel_antennas = setup_usrp(
        uhd, sample_rate_hz, rf_tune_hz, args.tx_gain, tx_channels, tx_dc_offset=tx_dc_offset
    )
    print(f"USRP configured: subdev={subdev}")
    for channel in tx_channels:
        actual_rate = _get_with_channel(usrp.get_tx_rate, channel)
        actual_freq = _get_with_channel(usrp.get_tx_freq, channel)
        actual_gain = _get_with_channel(usrp.get_tx_gain, channel)
        tx_antenna = channel_antennas.get(channel, TX_ANTENNA)
        effective_fc_hz = actual_freq - args.lo_offset_hz
        print(
            f"  ch{channel}: rate={actual_rate:.3f} Sa/s, "
            f"freq={actual_freq/1e6:.6f} MHz, effective_fc={effective_fc_hz/1e6:.6f} MHz, "
            f"gain={actual_gain:.2f} dB, "
            f"antenna={tx_antenna}"
        )

    max_samps = int(tx_stream.get_max_num_samps())
    send_chunk_samps = max_samps * args.chunk_mult
    n_repeats = max(1, int(math.ceil(args.duration * sample_rate_hz / iq.size)))
    actual_duration = n_repeats * iq.size / sample_rate_hz
    print(
        f"Playback: requested={args.duration:.6f}s, actual={actual_duration:.6f}s, "
        f"repeats={n_repeats}, max_samps_per_send={max_samps}, "
        f"requested_send_chunk_samps={send_chunk_samps}"
    )

    md = make_start_metadata(uhd, usrp, args.start_delay)
    tx_iq = np.vstack((iq, iq)) if len(tx_channels) > 1 else iq

    try:
        for _ in range(n_repeats):
            send_buffered(tx_stream, tx_iq, md, max_samps, send_chunk_samps, args.send_timeout)
    finally:
        print("Stopping TX (sending end-of-burst)...")
        try:
            send_eob(uhd, tx_stream, len(tx_channels))
        except Exception as exc:
            print(f"WARNING: Failed to send EOB cleanly: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
