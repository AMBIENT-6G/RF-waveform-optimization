#!/usr/bin/env python3
"""Replay preprocessed IQ blocks on a USRP B210.

This script expects IQ files exported by the notebook in `tx_iq/`, then replays
one selected tone block for a fixed duration.

Arguments kept intentionally minimal:
- `--tone`    : which exported tone set to play (0, 4, 8, 16, 32)
- `--bw`      : which exported signal bandwidth to play (in kHz)
- `--tx-gain` : TX gain in dB
- `--duration`: playback time in seconds
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np


IQ_DIR = Path("tx_iq")
ALLOWED_TONES = (0, 4, 8, 16, 32)
ALLOWED_CHANNELS = (0, 1)
IQ_BW_REGEX = re.compile(r"^iq_N(?P<n>\d+)_BW(?P<bw>\d+)kHz(?:_.*)?\.npz$")
ALLOWED_CPU_FORMATS = ("fc32", "sc16")
ALLOWED_OTW_FORMATS = ("sc16", "sc8")
SC16_DTYPE = np.dtype([("i", np.int16), ("q", np.int16)])

# Fixed radio settings for simplicity.
DEFAULT_USRP_ARGS = (
    "num_send_frames=1024,send_frame_size=32760"
)
DEFAULT_CHANNEL = 0
TX_ANTENNA = "TX/RX"
START_DELAY_S = 0.8
DEFAULT_CHUNK_MULT = 1
DEFAULT_SEND_TIMEOUT_S = 1.0
DEFAULT_CPU_FORMAT = "sc16"
DEFAULT_OTW_FORMAT = "sc16"


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


def try_set_thread_priority(uhd_module) -> bool:
    # Best effort only; fail open on platforms without RT priority privileges.
    candidates = [
        ("uhd.utils.set_thread_priority_safe", lambda: uhd_module.utils.set_thread_priority_safe()),
        ("uhd.set_thread_priority_safe", lambda: uhd_module.set_thread_priority_safe()),
    ]
    for _, setter in candidates:
        try:
            result = setter()
            # UHD helpers usually return bool, but some bindings may return None.
            if result is None or bool(result):
                return True
        except Exception:
            continue
    return False


def complex_to_sc16_buffer(samples: np.ndarray) -> np.ndarray:
    # Convert complex float IQ in [-1, 1] to packed int16 IQ pairs for CPU format sc16.
    s = np.asarray(samples)
    if s.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D IQ array for sc16 conversion, got ndim={s.ndim}")
    if not np.iscomplexobj(s):
        raise ValueError("sc16 conversion expects complex-valued samples")

    i = np.clip(np.rint(np.real(s) * 32767.0), -32768.0, 32767.0).astype(np.int16)
    q = np.clip(np.rint(np.imag(s) * 32767.0), -32768.0, 32767.0).astype(np.int16)
    out = np.empty(s.shape, dtype=SC16_DTYPE)
    out["i"] = i
    out["q"] = q
    return np.ascontiguousarray(out)


def _parse_tone_bw_from_iq_name(path: Path):
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
        if n_tones == int(tone) and bw == int(bw_khz):
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
            "(expected pattern iq_N<tone>_BW<bw>kHz_*.npz). "
            f"Available (tone,bw_kHz): {available}"
        )

    if len(matches) == 1:
        return matches[0].resolve()

    # Prefer col0 for multitone if several exports are present.
    preferred = [p for p in matches if "_col0" in p.stem]
    if len(preferred) == 1:
        return preferred[0].resolve()

    raise RuntimeError(
        f"Multiple IQ files found for tone={tone}, bw={bw_khz}kHz: {[p.name for p in matches]}. "
        "Keep one file per tone/bw (or only col0) in tx_iq/."
    )


def load_iq_file(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "iq" not in data.files:
            raise KeyError(f"{path} missing required array: iq")

        iq = np.asarray(data["iq"]).squeeze()
        fs_hz = float(np.asarray(data["sample_rate_hz"]).squeeze()) if "sample_rate_hz" in data.files else 20e6
        fs_source_hz = (
            float(np.asarray(data["sample_rate_source_hz"]).squeeze())
            if "sample_rate_source_hz" in data.files
            else fs_hz
        )
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

    return iq, fs_hz, fs_source_hz, fc_hz, power, peak


def setup_usrp(
    uhd_module,
    usrp_args: str,
    cpu_format: str,
    otw_format: str,
    spp: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    tx_gain_db: float,
    channels: list[int],
):
    usrp = uhd_module.usrp.MultiUSRP(usrp_args)
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
        _set_with_channel(usrp.set_tx_freq, _make_tune_request(uhd_module, center_freq_hz), channel)
        _set_with_channel(usrp.set_tx_gain, tx_gain_db, channel)
        _set_with_channel(usrp.set_tx_bandwidth, sample_rate_hz, channel)
        _set_with_channel(usrp.set_tx_antenna, tx_antenna, channel)
        channel_antennas[channel] = tx_antenna

    stream_args = uhd_module.usrp.StreamArgs(cpu_format, otw_format)
    if spp > 0:
        stream_args.args = f"spp={int(spp)}"
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
    # UHD streamer will not accept more than max_num_samps in one send().
    # Keep chunking bounded to avoid oversized Python->UHD buffer handoffs.
    target_samps = min(max_samps, max(1, int(send_chunk_samps)))

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


def send_eob(uhd_module, tx_stream, num_channels: int, cpu_format: str) -> None:
    md = uhd_module.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = True
    md.has_time_spec = False

    dtype = SC16_DTYPE if cpu_format == "sc16" else np.complex64
    if num_channels > 1:
        empty = np.zeros((num_channels, 0), dtype=dtype)
        fallback = np.zeros((num_channels, 1), dtype=dtype)
    else:
        empty = np.zeros(0, dtype=dtype)
        fallback = np.zeros(1, dtype=dtype)

    try:
        tx_stream.send(empty, md)
    except Exception:
        tx_stream.send(fallback, md)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay one exported IQ tone block on B210")
    p.add_argument("--tone", type=int, required=True, choices=ALLOWED_TONES, help="Tone set to play")
    p.add_argument(
        "--bw",
        type=int,
        required=True,
        help="Signal bandwidth in kHz (selects files named ..._BW<kHz>kHz_...)",
    )
    p.add_argument("--tx-gain", type=float, required=True, help="TX gain in dB")
    p.add_argument("--duration", type=float, default=10.0, help="Playback duration in seconds")
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
        help=(
            "Requested send chunk size multiplier relative to UHD max_num_samps "
            "(effective chunk is clamped to max_num_samps)"
        ),
    )
    p.add_argument(
        "--send-timeout",
        type=float,
        default=DEFAULT_SEND_TIMEOUT_S,
        help="Timeout (seconds) for each tx_stream.send() call",
    )
    p.add_argument(
        "--uhd-args",
        type=str,
        default=DEFAULT_USRP_ARGS,
        help=(
            "UHD device args string. Default increases TX transport buffering "
            "(num_send_frames/send_frame_size) to reduce underflows."
        ),
    )
    p.add_argument(
        "--cpu-format",
        type=str,
        default=DEFAULT_CPU_FORMAT,
        choices=ALLOWED_CPU_FORMATS,
        help="Host-side streamer sample format. sc16 reduces host bandwidth/CPU vs fc32.",
    )
    p.add_argument(
        "--otw-format",
        type=str,
        default=DEFAULT_OTW_FORMAT,
        choices=ALLOWED_OTW_FORMATS,
        help="UHD over-the-wire format. Use sc8 to reduce USB throughput at the cost of SNR.",
    )
    p.add_argument(
        "--spp",
        type=int,
        default=0,
        help="Optional samples-per-packet hint for UHD streamer (0 = UHD default).",
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
    if args.spp < 0:
        raise ValueError("--spp must be >= 0")
    if args.bw <= 0:
        raise ValueError("--bw must be > 0")
    tx_channels = [0, 1] if args.both_channels else [args.channel]

    iq_file = find_iq_file_for_tone_bw(IQ_DIR, args.tone, args.bw)
    iq, sample_rate_hz, sample_rate_source_hz, center_freq_hz, power, peak = load_iq_file(iq_file)
    expected_rate_hz = 2.0 * float(args.bw) * 1e3

    print(f"Selected IQ file: {iq_file}")
    print(f"IQ stats: samples={iq.size}, avg_power={power:.6g}, peak={peak:.6g}")
    if not np.isclose(sample_rate_hz, sample_rate_source_hz, rtol=1e-12, atol=0.0):
        print(
            f"Rate metadata: sample_rate_hz={sample_rate_hz:.3f}, "
            f"sample_rate_source_hz={sample_rate_source_hz:.3f}"
        )
    print(
        f"Replay settings: bw={args.bw} kHz, fs={sample_rate_hz:.3f} Sa/s, fc={center_freq_hz/1e6:.6f} MHz, "
        f"tx_gain={args.tx_gain:.2f} dB, channels={tx_channels}, "
        f"start_delay={args.start_delay:.3f}s, chunk_mult={args.chunk_mult}, "
        f"send_timeout={args.send_timeout:.3f}s, "
        f"cpu_format={args.cpu_format}, otw_format={args.otw_format}, spp={args.spp}, "
        f"uhd_args='{args.uhd_args}'"
    )
    if not np.isclose(sample_rate_source_hz, expected_rate_hz, rtol=1e-4, atol=1.0):
        print(
            f"WARNING: sample_rate_source_hz={sample_rate_source_hz:.3f} does not match 2*bw "
            f"({expected_rate_hz:.3f} for bw={args.bw} kHz)."
        )
    if args.otw_format == "sc16" and sample_rate_hz >= 40e6:
        print(
            "WARNING: sc16 at >=40 MS/s is host-heavy in Python. "
            "Use --otw-format sc8 to reduce USB and CPU load."
        )
    if args.cpu_format == "fc32" and sample_rate_hz >= 40e6:
        print(
            "WARNING: cpu_format=fc32 at >=40 MS/s is Python-heavy. "
            "Try --cpu-format sc16 with --otw-format sc8."
        )

    try:
        import uhd
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python UHD module `uhd`. Install UHD Python bindings."
        ) from exc

    rt_ok = try_set_thread_priority(uhd)
    print(f"Thread priority: {'realtime-enabled' if rt_ok else 'default (no RT privilege)'}")

    usrp, tx_stream, subdev, channel_antennas = setup_usrp(
        uhd,
        args.uhd_args,
        args.cpu_format,
        args.otw_format,
        args.spp,
        sample_rate_hz,
        center_freq_hz,
        args.tx_gain,
        tx_channels,
    )
    print(f"USRP configured: subdev={subdev}")
    for channel in tx_channels:
        actual_rate = _get_with_channel(usrp.get_tx_rate, channel)
        actual_freq = _get_with_channel(usrp.get_tx_freq, channel)
        actual_gain = _get_with_channel(usrp.get_tx_gain, channel)
        tx_antenna = channel_antennas.get(channel, TX_ANTENNA)
        print(
            f"  ch{channel}: rate={actual_rate:.3f} Sa/s, "
            f"freq={actual_freq/1e6:.6f} MHz, gain={actual_gain:.2f} dB, "
            f"antenna={tx_antenna}"
        )

    max_samps = int(tx_stream.get_max_num_samps())
    send_chunk_samps_req = max_samps * args.chunk_mult
    send_chunk_samps = min(max_samps, send_chunk_samps_req)
    n_repeats = max(1, int(math.ceil(args.duration * sample_rate_hz / iq.size)))
    actual_duration = n_repeats * iq.size / sample_rate_hz
    print(
        f"Playback: requested={args.duration:.6f}s, actual={actual_duration:.6f}s, "
        f"repeats={n_repeats}, max_samps_per_send={max_samps}, "
        f"requested_send_chunk_samps={send_chunk_samps_req}, "
        f"effective_send_chunk_samps={send_chunk_samps}"
    )

    md = make_start_metadata(uhd, usrp, args.start_delay)
    tx_iq = np.vstack((iq, iq)) if len(tx_channels) > 1 else iq
    if args.cpu_format == "sc16":
        tx_iq = complex_to_sc16_buffer(tx_iq)
        print("Prepared packed sc16 host buffer for TX.")

    try:
        for _ in range(n_repeats):
            send_buffered(tx_stream, tx_iq, md, max_samps, send_chunk_samps, args.send_timeout)
    finally:
        print("Stopping TX (sending end-of-burst)...")
        try:
            send_eob(uhd, tx_stream, len(tx_channels), args.cpu_format)
        except Exception as exc:
            print(f"WARNING: Failed to send EOB cleanly: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
