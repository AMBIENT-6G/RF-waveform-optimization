# RF Waveform Optimization

Pipeline to:
1. transmit waveform-based RF signals,
2. measure harvested power (scope or energy profiler),
3. plot sweep statistics,
4. fit RF->DC harvester models per tone.

## Project Layout

```text
scripts/
  build_gain_power_map.py
  measure_scope_power.py
  measure_ep_power.py
  plot_power_stats.py
  plot_scope_avg_power.py
  fit_tone_models.py
  tx_waveform.py
  run_layout.py

data/
  reference/
    harvester-chart-data.csv
  weights/
    *.mat
  gain-power-map.csv
  tx_iq/
    iq_dc_BW*.npz
    iq_nb_BW*.npz
    iq_N*_BW*_TXG*db.npz

results/
  <run-id>/
    raw/
    tables/
    plots/
    meta/manifest.json
  legacy_undated/
    raw/
    tables/
    plots/

notebooks/
  read_weights_mat_files.ipynb
```

Waveform selector convention:
- `tone=0` -> DC waveform
- `tone=1` -> narrowband waveform
- `tone>=2` -> multitone waveform with `N=tone` carriers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Results Contract

All output-producing scripts support:
- `--results-dir` (default: `results`)
- `--run-id` (optional)

Routing:
- raw measurements -> `results/<run-id>/raw/`
- derived tables/json -> `results/<run-id>/tables/`
- plots -> `results/<run-id>/plots/`

If `--run-id` is omitted:
- measurement scripts create a timestamp run-id (`YYYYMMDD_HHMMSS`)
- fit/plot scripts infer from input path if possible, else `manual_<timestamp>`

Each run also updates:
- `results/<run-id>/meta/manifest.json`

## End-to-End Workflow

### 0) Export IQ Files From MAT Weights

Use [notebooks/read_weights_mat_files.ipynb](/mnt/c/Users/Calle/OneDrive/Documenten/GitHub/RF-waveform-optimization/notebooks/read_weights_mat_files.ipynb) to:
- read `data/weights/weightsN*BW*.mat`
- interpret `BW1000` as `1000 kHz = 1 MHz`
- enforce source sample rate `fs = 2 * BW`
- map each `inPwrVec` level to the closest gain in `data/gain-power-map.csv`
- export `iq_dc_BW*.npz`, `iq_nb_BW*.npz`, and `iq_N*_BW*_TXG*db.npz`

The current repo already contains `data/gain-power-map.csv`. If you regenerate
that file from fresh scope measurements, rerun the notebook export afterwards so
the multitone IQ filenames and metadata stay aligned with the updated gain map.

### 1) Measure (Scope)

```bash
python3 scripts/measure_scope_power.py \
  --tones 0,1,4,8,16,32 \
  --bw 1000 \
  --gain-start 50 --gain-stop 85 --gain-step 0.2 \
  --tx-duration 20 \
  --run-id 20260304_172559
```

### 2) Build Gain-to-Power Map

```bash
python3 scripts/build_gain_power_map.py \
  --scope-jsonl results/20260304_172559/raw/meas-tone-power-scope.jsonl \
  --tone 0 \
  --output-csv data/gain-power-map.csv
```

This writes `data/gain-power-map.csv` with:
- `scope_power_dbm`
- `gain_db`
- `input_level_dbm`
- `input_level_mw`

After rebuilding the gain map, rerun the notebook export step if you want the
multitone `iq_N*_BW*_TXG*db.npz` files to reflect the new mapping.

### 3) Measure (Energy Profiler)

```bash
python3 scripts/measure_ep_power.py \
  --tones 0,1,4,8,16,32 \
  --bw 1000 \
  --gain-start 50 --gain-stop 80 --gain-step 0.2 \
  --tx-duration 20 \
  --run-id 20260304_172559
```

`measure_ep_power.py` auto-discovers the shared multitone gain set from
`data/tx_iq/` when gain-tagged IQ files are present, then sweeps in this order
for each gain:
- `DC`
- `NB`
- `N=4`
- `N=8`
- `N=16`
- `N=32`

In this auto-discovery mode, multitone sweeps use the exact shared tagged gains
present in `data/tx_iq/`.

If you want an explicit gain sweep, provide `--gain-start`, `--gain-stop`, and
`--gain-step` together. In that mode:
- all three gain arguments are required as a group
- `tone=0` and `tone=1` still use their single DC/NB IQ files
- `tone>=2` uses the closest gain-tagged `iq_N*_BW*_TXG*db.npz` file for each requested gain

Example: requesting `--gain-start 80 --gain-stop 80 --gain-step 1` for `tone=32`
will use `iq_N32_BW1000_TXG80.2db.npz` when that is the nearest tagged file.

During EP sweeps, `measure_ep_power.py` now waits for a localhost ZMQ
`tx_started` event from `tx_waveform.py` before sampling. The measurement window
is derived automatically as `0.9 * --tx-duration`.

If `tx_done` arrives while EP sampling is still active, collection stops
immediately and the last 10 EP samples are discarded. Use `--tx-start-timeout`
to control how long `measure_ep_power.py` waits for the `tx_started` event
before failing the sweep.

### 4) Plot Sweep Statistics

```bash
python3 scripts/plot_power_stats.py \
  results/20260304_172559/raw/meas-tones-power.jsonl \
  --run-id 20260304_172559 \
  --no-show
```

For scope-only sweeps:

```bash
python3 scripts/plot_scope_avg_power.py \
  results/20260304_172559/raw/meas-tone-power-scope.jsonl \
  --run-id 20260304_172559 \
  --no-show
```

### 5) Fit Harvester Models Per Tone

```bash
python3 scripts/fit_tone_models.py \
  results/20260304_172559/raw/meas-tones-power.jsonl \
  --run-id 20260304_172559 \
  --no-show
```

## Model List in `fit_tone_models.py`

- `linear_efficiency`
- `polynomial_even`
- `polynomial_cubic`
- `logistic`
- `logistic_sigmoind_model`
- `paper_logistic`
- `rational_saturation`
- `power_law_offset`
- `exponential`
- `piecewise_linear`

Plots show the best 3 models per tone (by RMSE).

## TX Replay Utility

```bash
python3 scripts/tx_waveform.py --tone 32 --bw 1000 --gain 80.2 --duration 20
python3 scripts/tx_waveform.py --tone 1 --bw 1000 --gain 80.2 --duration 20
python3 scripts/tx_waveform.py --tone 0 --bw 1000 --gain 80.2 --duration 20
```

This reads IQ files from `data/tx_iq/`.
- For multitone files, `--gain` selects the matching `iq_N*_BW*_TXG*db.npz`.
- Use `--closest-gain-match` to allow the nearest tagged multitone IQ file when no exact tagged gain exists.
  Example: `--tone 32 --bw 1000 --gain 80.0 --closest-gain-match` can select `iq_N32_BW1000_TXG80.2db.npz`.
- For `tone=0` and `tone=1`, the same configured gain is still applied on the USRP even though the file itself is not gain-tagged.

## Make Targets

```bash
make measure
make plot
make fit
make lint
make check
```
