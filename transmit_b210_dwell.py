#!/usr/bin/env python3
"""Unified B210 TX script for IQ replay, multitone, and narrowband operation.

Mode selection is via `--tones`:
- `--tones 0`   : narrowband mode, transmit one sinusoid at `--tone-offset-hz`
- `--tones 4`   : load `weightsN4.mat`
- `--tones 8`   : load `weightsN8.mat`
- `--tones 16`  : load `weightsN16.mat`
- `--tones 32`  : load `weightsN32.mat`

Or provide preprocessed IQ files (recommended):
- `--iq-file path/to/iq_*.npz` (repeatable) and/or `--iq-glob "tx_iq/*.npz"`

For multitone modes, this script reads `fn` and `weights` from the selected MAT
file and synthesizes periodic baseband blocks for selected weight columns.

All modes support discrete TX-gain sweep, per-step dwell, and safe amplitude
control:
- Normalize average time-domain power to `--target-power`.
- Enforce `|IQ| <= --max-amplitude` (default 1.0) by additional scaling.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


ALLOWED_TONES = (0, 4, 8, 16, 32)


@dataclass
class WaveformData:
    source: Path
    fn_hz: np.ndarray
    weights: np.ndarray
    in_pwr_mw: Optional[np.ndarray]


@dataclass
class SynthResult:
    block: np.ndarray
    bins: np.ndarray
    quantized_fn_hz: np.ndarray
    freq_error_hz: np.ndarray
    scale: float
    raw_peak: float
    raw_power: float
    final_peak: float
    final_power: float
    peak_limited: bool


@dataclass
class SymbolBlock:
    label: str
    block: np.ndarray


def _to_1d(x: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(x).squeeze())


def _to_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).squeeze()
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def load_waveform_mat(path: Path) -> WaveformData:
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover - runtime dependency failure
        raise RuntimeError(
            "Could not import scipy.io.loadmat. Install compatible SciPy/NumPy versions."
        ) from exc

    mat = loadmat(path, squeeze_me=True)
    required = ["fn", "weights"]
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")

    fn = _to_1d(mat["fn"]).astype(float)
    weights = _to_2d(mat["weights"]).astype(np.complex128)
    in_pwr = _to_1d(mat["inPwrVec"]).astype(float) if "inPwrVec" in mat else None

    if weights.shape[0] != fn.size and weights.shape[1] == fn.size:
        weights = weights.T

    if weights.shape[0] != fn.size:
        raise ValueError(
            f"Tone count mismatch: fn has {fn.size}, weights has shape {weights.shape}"
        )
    if in_pwr is not None and weights.shape[1] != in_pwr.size:
        raise ValueError(
            f"Column mismatch: weights has {weights.shape[1]} cols, inPwrVec has {in_pwr.size}"
        )

    return WaveformData(source=path, fn_hz=fn, weights=weights, in_pwr_mw=in_pwr)


def parse_columns(expr: str, n_cols: int) -> List[int]:
    if expr.strip().lower() == "all":
        return list(range(n_cols))

    idxs: List[int] = []
    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"Column index {idx} out of range [0, {n_cols - 1}]")
        idxs.append(idx)

    if not idxs:
        raise ValueError("No valid columns selected.")

    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def parse_gain_values(args: argparse.Namespace) -> List[float]:
    has_explicit = bool(args.tx_gains.strip())
    has_range = any(x is not None for x in (args.gain_start, args.gain_stop, args.gain_step))

    if has_explicit and has_range:
        raise ValueError("Use either --tx-gains or --gain-start/--gain-stop/--gain-step, not both.")

    if has_explicit:
        gains = []
        for token in args.tx_gains.split(","):
            token = token.strip()
            if token:
                gains.append(float(token))
        if not gains:
            raise ValueError("No valid gain values in --tx-gains.")
        return gains

    if has_range:
        if None in (args.gain_start, args.gain_stop, args.gain_step):
            raise ValueError("--gain-start, --gain-stop, and --gain-step must all be set.")
        if args.gain_step == 0:
            raise ValueError("--gain-step must not be zero.")

        start = float(args.gain_start)
        stop = float(args.gain_stop)
        step = float(args.gain_step)
        direction = 1.0 if stop >= start else -1.0
        if step * direction <= 0:
            raise ValueError("--gain-step has wrong sign for the requested range.")

        gains: List[float] = []
        g = start
        tol = 1e-9
        if direction > 0:
            while g <= stop + tol:
                gains.append(float(g))
                g += step
        else:
            while g >= stop - tol:
                gains.append(float(g))
                g += step

        if not gains:
            raise ValueError("Gain sweep generated no values.")
        return gains

    return [float(args.tx_gain)]


def resolve_iq_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for p in args.iq_file:
        paths.append(p.expanduser())
    if args.iq_glob:
        paths.extend(sorted(Path(".").glob(args.iq_glob)))

    out: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if not rp.exists():
            raise FileNotFoundError(f"IQ file not found: {p}")
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def load_iq_block(path: Path):
    ext = path.suffix.lower()
    meta = {}

    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "iq" in data.files:
                arr = data["iq"]
            elif data.files:
                arr = data[data.files[0]]
            else:
                raise ValueError(f"{path} contains no arrays.")

            for key in (
                "sample_rate_hz",
                "center_freq_hz",
                "target_power",
                "max_amplitude",
                "n_tones",
                "column",
                "raw_power",
                "final_power",
                "raw_peak",
                "final_peak",
            ):
                if key in data.files:
                    v = np.asarray(data[key]).squeeze()
                    if v.ndim == 0:
                        meta[key] = float(v)
    elif ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported IQ file type: {path} (expected .npz or .npy)")

    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{path} IQ array must be 1D after squeeze, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{path} IQ array is empty.")

    if not np.iscomplexobj(arr):
        arr = arr.astype(np.float64) + 0j
    block = arr.astype(np.complex64, copy=False)
    return block, meta


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


def normalize_power_and_limit_peak(
    signal: np.ndarray,
    target_power: float,
    max_amplitude: float,
):
    if target_power <= 0:
        raise ValueError("target_power must be > 0")
    if max_amplitude <= 0:
        raise ValueError("max_amplitude must be > 0")

    raw_power = float(np.mean(np.abs(signal) ** 2))
    raw_peak = float(np.max(np.abs(signal)))
    if raw_power <= 0 or raw_peak <= 0:
        raise ValueError("Synthesis produced zero signal")

    scale = math.sqrt(target_power / raw_power)
    out = signal * scale
    final_peak = float(np.max(np.abs(out)))
    peak_limited = False

    if final_peak > max_amplitude:
        limiter = max_amplitude / final_peak
        out = out * limiter
        scale *= limiter
        final_peak = float(np.max(np.abs(out)))
        peak_limited = True

    if final_peak > max_amplitude:
        guard = (max_amplitude * (1.0 - 1e-7)) / final_peak
        out = out * guard
        scale *= guard
        final_peak = float(np.max(np.abs(out)))
        peak_limited = True

    final_power = float(np.mean(np.abs(out) ** 2))
    return out, scale, raw_peak, raw_power, final_peak, final_power, peak_limited


def synthesize_periodic_block(
    fn_hz: np.ndarray,
    weights_col: np.ndarray,
    sample_rate: float,
    block_size: int,
    target_power: float,
    max_amplitude: float,
) -> SynthResult:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    bins = np.rint((fn_hz / sample_rate) * block_size).astype(np.int64)
    quantized_fn = bins.astype(np.float64) * (sample_rate / block_size)
    freq_error = quantized_fn - fn_hz

    n = np.arange(block_size, dtype=np.float64)
    phase = (2j * np.pi / block_size) * np.outer(n, bins)
    block = np.exp(phase) @ weights_col

    (
        block_scaled,
        scale,
        raw_peak,
        raw_power,
        final_peak,
        final_power,
        peak_limited,
    ) = normalize_power_and_limit_peak(
        signal=block,
        target_power=target_power,
        max_amplitude=max_amplitude,
    )

    return SynthResult(
        block=block_scaled.astype(np.complex64),
        bins=bins,
        quantized_fn_hz=quantized_fn,
        freq_error_hz=freq_error,
        scale=scale,
        raw_peak=raw_peak,
        raw_power=raw_power,
        final_peak=final_peak,
        final_power=final_power,
        peak_limited=peak_limited,
    )


def setup_usrp(
    uhd_module,
    args: argparse.Namespace,
    initial_gain_db: float,
):
    usrp = uhd_module.usrp.MultiUSRP(args.usrp_args)

    _set_with_channel(usrp.set_tx_rate, args.sample_rate, args.channel)
    _set_with_channel(usrp.set_tx_freq, _make_tune_request(uhd_module, args.center_freq), args.channel)
    _set_with_channel(usrp.set_tx_gain, initial_gain_db, args.channel)
    if args.tx_bandwidth > 0:
        _set_with_channel(usrp.set_tx_bandwidth, args.tx_bandwidth, args.channel)
    if args.tx_antenna:
        _set_with_channel(usrp.set_tx_antenna, args.tx_antenna, args.channel)

    stream_args = uhd_module.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [args.channel]
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


def send_buffered(
    tx_stream,
    samples: np.ndarray,
    md,
    max_samps: int,
) -> None:
    offset = 0
    first = True
    total = samples.size

    while offset < total:
        n = min(max_samps, total - offset)
        chunk = samples[offset : offset + n]
        sent = tx_stream.send(chunk, md)
        if sent != n:
            raise RuntimeError(f"Short send: requested {n}, sent {sent}")
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
    try:
        tx_stream.send(empty, md)
    except Exception:
        one = np.zeros(1, dtype=np.complex64)
        tx_stream.send(one, md)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified B210 TX: narrowband or multitone with gain sweep")

    p.add_argument(
        "--iq-file",
        type=Path,
        action="append",
        default=[],
        help="Preprocessed IQ file (.npz or .npy). Can be repeated.",
    )
    p.add_argument(
        "--iq-glob",
        type=str,
        default="",
        help='Additional IQ file glob, e.g. "tx_iq/*.npz"',
    )
    p.add_argument(
        "--tones",
        type=int,
        choices=ALLOWED_TONES,
        default=None,
        help="0=narrowband sinusoid; 4/8/16/32 load weightsN{tones}.mat",
    )
    p.add_argument(
        "--tone-offset-hz",
        type=float,
        default=100e3,
        help="Used only when --tones 0: baseband tone offset from center frequency in Hz",
    )
    p.add_argument(
        "--mat-dir",
        type=Path,
        default=Path("."),
        help="Directory containing weightsN4/8/16/32.mat files",
    )
    p.add_argument(
        "--columns",
        type=str,
        default="0",
        help='Used for multitone modes: comma-separated column indices (e.g. "0,1") or "all"',
    )

    p.add_argument(
        "--dwell-seconds",
        type=float,
        default=2.0,
        help="Dwell time per symbol/gain step in seconds (rounded up to block boundaries)",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="How many passes over all selected symbols (0 = infinite)",
    )

    p.add_argument("--sample-rate", type=float, default=20e6, help="TX sample rate in Sa/s")
    p.add_argument("--center-freq", type=float, default=875e6, help="RF center frequency in Hz")
    p.add_argument(
        "--tx-bandwidth",
        type=float,
        default=20e6,
        help="Analog TX bandwidth in Hz (<=0 means do not set)",
    )

    p.add_argument(
        "--tx-gain",
        type=float,
        default=40.0,
        help="Single TX gain in dB (used if no gain sweep args are provided)",
    )
    p.add_argument(
        "--tx-gains",
        type=str,
        default="",
        help='Explicit comma-separated TX gain values in dB, e.g. "30,32,34"',
    )
    p.add_argument("--gain-start", type=float, default=None, help="Sweep start gain in dB")
    p.add_argument("--gain-stop", type=float, default=None, help="Sweep stop gain in dB")
    p.add_argument("--gain-step", type=float, default=None, help="Sweep step in dB")
    p.add_argument(
        "--gain-settle-seconds",
        type=float,
        default=0.05,
        help="Delay after changing TX gain before transmitting (seconds)",
    )

    p.add_argument(
        "--block-size",
        type=int,
        default=65536,
        help="Periodic block size in samples (power of two recommended)",
    )
    p.add_argument(
        "--target-power",
        type=float,
        default=0.8,
        help="Target average time-domain power E[|x|^2] before peak safety scaling",
    )
    p.add_argument(
        "--max-amplitude",
        type=float,
        default=1.0,
        help="Maximum allowed complex magnitude |x| (must be <= 1.0)",
    )
    p.add_argument(
        "--max-freq-error-hz",
        type=float,
        default=5.0,
        help="Warn if |tone quantization error| exceeds this value",
    )
    p.add_argument(
        "--strict-freq-grid",
        action="store_true",
        help="Abort if tone quantization error exceeds --max-freq-error-hz",
    )

    p.add_argument("--tx-antenna", type=str, default="TX/RX", help="TX antenna name")
    p.add_argument("--channel", type=int, default=0, help="USRP TX channel index")
    p.add_argument("--usrp-args", type=str, default="", help='UHD device args, e.g. "type=b200"')
    p.add_argument(
        "--start-delay",
        type=float,
        default=0.2,
        help="Timed-start delay in seconds (0 = start immediately)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load/synthesize/report symbol blocks, do not access UHD hardware",
    )

    return p


def build_symbol_blocks(args: argparse.Namespace) -> List[SymbolBlock]:
    symbol_blocks: List[SymbolBlock] = []

    iq_paths = resolve_iq_paths(args)
    if iq_paths:
        print(f"IQ replay mode with {len(iq_paths)} file(s)")
        for path in iq_paths:
            block, meta = load_iq_block(path)
            power = float(np.mean(np.abs(block) ** 2))
            peak = float(np.max(np.abs(block)))
            if peak > args.max_amplitude + 1e-6:
                raise ValueError(
                    f"{path.name} has |IQ| peak {peak:.6g} > max-amplitude {args.max_amplitude:.6g}. "
                    "Re-export/preprocess with lower amplitude."
                )

            msg = (
                f"[iq:{path.name}] power={power:.6g}, peak={peak:.6g}, "
                f"samples={block.size}"
            )
            if "sample_rate_hz" in meta and abs(meta["sample_rate_hz"] - args.sample_rate) > 1e-6:
                msg += f", NOTE sample_rate_hz in file={meta['sample_rate_hz']:.6g}"
            if "center_freq_hz" in meta and abs(meta["center_freq_hz"] - args.center_freq) > 1e-3:
                msg += f", NOTE center_freq_hz in file={meta['center_freq_hz']:.6g}"
            print(msg)

            symbol_blocks.append(SymbolBlock(label=f"iq:{path.name}", block=block))
        return symbol_blocks

    if args.tones is None:
        raise ValueError("Provide either IQ input (--iq-file/--iq-glob) or --tones.")

    if args.tones == 0:
        synth = synthesize_periodic_block(
            fn_hz=np.array([args.tone_offset_hz], dtype=float),
            weights_col=np.array([1.0 + 0.0j], dtype=np.complex128),
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            target_power=args.target_power,
            max_amplitude=args.max_amplitude,
        )
        max_abs_err = float(np.max(np.abs(synth.freq_error_hz)))
        if max_abs_err > args.max_freq_error_hz:
            msg = (
                f"narrowband max tone quantization error is {max_abs_err:.3f} Hz "
                f"(limit {args.max_freq_error_hz:.3f} Hz)"
            )
            if args.strict_freq_grid:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

        quantized_offset = float(synth.quantized_fn_hz[0])
        label = f"narrowband:offset={quantized_offset:.3f}Hz"
        print(
            f"[{label}] raw_power={synth.raw_power:.6g}, final_power={synth.final_power:.6g}, "
            f"raw_peak={synth.raw_peak:.6g}, final_peak={synth.final_peak:.6g}, "
            f"peak_limited={synth.peak_limited}, scale={synth.scale:.6g}, "
            f"freq_error={max_abs_err:.3f} Hz"
        )
        symbol_blocks.append(SymbolBlock(label=label, block=synth.block))
        return symbol_blocks

    mat_path = (args.mat_dir / f"weightsN{args.tones}.mat").resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"Expected MAT file not found: {mat_path}")

    wf = load_waveform_mat(mat_path)
    if wf.fn_hz.size != args.tones:
        print(
            f"WARNING: tones in file ({wf.fn_hz.size}) does not match --tones {args.tones}."
        )

    columns = parse_columns(args.columns, wf.weights.shape[1])
    print(
        f"[{wf.source.name}] tones={wf.fn_hz.size}, "
        f"weight_cols={wf.weights.shape[1]}, selected_cols={columns}"
    )

    for col in columns:
        synth = synthesize_periodic_block(
            fn_hz=wf.fn_hz,
            weights_col=wf.weights[:, col],
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            target_power=args.target_power,
            max_amplitude=args.max_amplitude,
        )
        max_abs_err = float(np.max(np.abs(synth.freq_error_hz)))

        if max_abs_err > args.max_freq_error_hz:
            msg = (
                f"{wf.source.name}:col{col} max tone quantization error is {max_abs_err:.3f} Hz "
                f"(limit {args.max_freq_error_hz:.3f} Hz)"
            )
            if args.strict_freq_grid:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

        label = f"{wf.source.name}:col{col}"
        msg = (
            f"  {label} raw_power={synth.raw_power:.6g}, final_power={synth.final_power:.6g}, "
            f"raw_peak={synth.raw_peak:.6g}, final_peak={synth.final_peak:.6g}, "
            f"peak_limited={synth.peak_limited}, scale={synth.scale:.6g}, "
            f"max|df|={max_abs_err:.3f} Hz"
        )
        if wf.in_pwr_mw is not None:
            pin_mw = float(wf.in_pwr_mw[col])
            pin_dbm = 10.0 * math.log10(pin_mw) if pin_mw > 0 else float("-inf")
            msg += f", inPwr={pin_mw:.6g} mW ({pin_dbm:.2f} dBm)"
        print(msg)

        symbol_blocks.append(SymbolBlock(label=label, block=synth.block))

    return symbol_blocks


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.dwell_seconds <= 0:
        raise ValueError("--dwell-seconds must be > 0")
    if args.block_size <= 0:
        raise ValueError("--block-size must be > 0")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.gain_settle_seconds < 0:
        raise ValueError("--gain-settle-seconds must be >= 0")
    if args.target_power <= 0:
        raise ValueError("--target-power must be > 0")
    if args.max_amplitude <= 0 or args.max_amplitude > 1.0:
        raise ValueError("--max-amplitude must be in (0, 1.0]")

    gain_values = parse_gain_values(args)
    if args.iq_file or args.iq_glob:
        print("Mode: IQ replay")
    else:
        print(f"Mode tones={args.tones}")
    print(f"Gain schedule ({len(gain_values)} step(s)): {gain_values}")

    symbol_blocks = build_symbol_blocks(args)
    if not symbol_blocks:
        raise RuntimeError("No symbol blocks selected for transmission.")

    if args.dry_run:
        print("Dry run complete (no UHD device opened).")
        return 0

    try:
        import uhd
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python UHD module `uhd`. "
            "Install UHD Python bindings on the TX machine."
        ) from exc

    usrp, tx_stream = setup_usrp(uhd, args, initial_gain_db=gain_values[0])

    actual_rate = _get_with_channel(usrp.get_tx_rate, args.channel)
    actual_freq = _get_with_channel(usrp.get_tx_freq, args.channel)
    actual_gain = _get_with_channel(usrp.get_tx_gain, args.channel)
    print(
        f"USRP configured: rate={actual_rate:.3f} Sa/s, "
        f"freq={actual_freq/1e6:.6f} MHz, initial_gain={actual_gain:.2f} dB"
    )

    max_samps = int(tx_stream.get_max_num_samps())
    print(f"TX streamer max samples per send: {max_samps}")

    dwell_blocks = max(1, int(math.ceil((args.dwell_seconds * args.sample_rate) / args.block_size)))
    dwell_time_actual = dwell_blocks * args.block_size / args.sample_rate
    print(
        f"Dwell per symbol/gain step: requested={args.dwell_seconds:.6f}s, "
        f"actual={dwell_time_actual:.6f}s ({dwell_blocks} blocks)"
    )

    stop = False

    def _handle_stop(signum, frame):  # noqa: ANN001, ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    md = make_start_metadata(uhd, usrp, args.start_delay)

    cycle = 0
    print("Starting TX. Press Ctrl+C to stop.")
    try:
        while not stop and (args.cycles == 0 or cycle < args.cycles):
            cycle += 1
            print(f"Cycle {cycle}" if args.cycles != 0 else f"Cycle {cycle} (infinite mode)")
            for symbol in symbol_blocks:
                if stop:
                    break

                print(f"  Symbol {symbol.label}")
                for gain in gain_values:
                    if stop:
                        break

                    _set_with_channel(usrp.set_tx_gain, gain, args.channel)
                    if args.gain_settle_seconds > 0:
                        time.sleep(args.gain_settle_seconds)
                    print(f"    gain={gain:.2f} dB: transmitting {dwell_blocks} block repeats")

                    for _ in range(dwell_blocks):
                        if stop:
                            break
                        send_buffered(tx_stream, symbol.block, md, max_samps)

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
    except KeyboardInterrupt:
        raise SystemExit(130)
