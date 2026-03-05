# RF Waveform Optimization

Pipeline to:
1. transmit waveform-based RF signals,
2. measure harvested power (scope or energy profiler),
3. plot sweep statistics,
4. fit RF->DC harvester models per tone.

## Project Layout

```text
scripts/
  measure_scope_power.py
  measure_ep_power.py
  plot_power_stats.py
  fit_tone_models.py
  tx_waveform.py
  run_layout.py

data/
  reference/
    harvester-chart-data.csv
  weights/
    *.mat
  tx_iq/
    iq_N*_BW*.npz

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

### 1) Measure (Scope)

```bash
python3 scripts/measure_scope_power.py \
  --tones 0,1,4,8,16,32 \
  --bw 1000 \
  --gain-start 50 --gain-stop 85 --gain-step 0.2 \
  --tx-duration 20 \
  --run-id 20260304_172559
```

### 2) Measure (Energy Profiler)

```bash
python3 scripts/measure_ep_power.py \
  --tones 0,4,8,16,32 \
  --bw 1000 \
  --gain-start 50 --gain-stop 85 --gain-step 0.2 \
  --tx-duration 20 \
  --run-id 20260304_172559
```

### 3) Plot Sweep Statistics

```bash
python3 scripts/plot_power_stats.py \
  results/20260304_172559/raw/meas-tones-power.jsonl \
  --run-id 20260304_172559 \
  --no-show
```

### 4) Fit Harvester Models Per Tone

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
- `exponential`
- `piecewise_linear`

Plots show the best 3 models per tone (by RMSE).

## TX Replay Utility

```bash
python3 scripts/tx_waveform.py --tone 32 --bw 1000 --gain 80 --duration 20
```

This reads IQ files from `data/tx_iq/`.

## Make Targets

```bash
make measure
make plot
make fit
make lint
make check
```
