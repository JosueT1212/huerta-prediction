# Pure-LSTM Architecture Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-LSTM variant of `CNNRNN` (`num_cnn_blocks=0`), run the same 216-run grid search for inv3 and inv4 under this variant, and report whether it beats each greenhouse's current CNN+LSTM production winner on both R² and RMSE.

**Architecture:** Reuse `num_cnn_blocks=0` as the no-CNN toggle in the existing `CNNRNN` class — an empty `nn.Sequential` is already identity, so only `lstm.input_size` needs a conditional. New `hp_inv3_lstm.yaml`/`hp_inv4_lstm.yaml` configs (copies of the production yamls with `num_cnn_blocks: 0`) drive two new thin grid-search scripts that mirror `cnn_rnn_inv3.py`/`cnn_rnn_inv4.py` exactly, writing to separate `*_lstm` output paths.

**Tech Stack:** PyTorch, pandas, numpy, pytest — same stack as `Models/cnn_rnn_yield.py`.

## Global Constraints

- `HORIZON`, `seq_len` (inv3=4, inv4=2), `use_gap_norm`, `skip_first_weeks`, and all other retuned hyperparameters stay fixed — this compares architecture only.
- No changes to other model families (ARIMAX, XGBoost, SVR, ElasticNet).
- No automatic production refit or deployment swap — this plan stops at reporting the comparison.
- Current production winners to beat (both must hold, decided independently per greenhouse):
  - Inv3: R² > 0.5286 AND RMSE < 5661.0 kg
  - Inv4: R² > 0.5595 AND RMSE < 6572.9 kg
- Existing production configs/scripts/models (`hp_inv3.yaml`, `hp_inv4.yaml`, `cnn_rnn_inv3.py`, `cnn_rnn_inv4.py`, `production_cnn_rnn_inv*.pt`) must not be modified.

---

### Task 1: `CNNRNN` supports `num_cnn_blocks=0`

**Files:**
- Modify: `Models/cnn_rnn_yield.py:767-813` (`CNNRNN.__init__`)
- Test: `Models/tests/test_cnn_rnn_yield.py`

**Interfaces:**
- Consumes: existing `CNNRNN(n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding, num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden, n_out=1)` constructor signature — unchanged.
- Produces: `CNNRNN` instantiated with `num_cnn_blocks=0` now builds with `lstm.input_size = n_sensor + n_temporal` instead of raising a shape mismatch at `forward()`. No other task depends on new symbols from this task — later tasks only pass `num_cnn_blocks: 0` via yaml.

- [ ] **Step 1: Write the failing test**

Add to `Models/tests/test_cnn_rnn_yield.py` (near the other `CNNRNN`/`train_model` tests, e.g. after `test_train_model_track_best_false_ignores_patience_and_runs_all_epochs`):

```python
def test_cnnrnn_num_cnn_blocks_zero_runs_forward():
    """
    num_cnn_blocks=0 must skip the CNN entirely (empty nn.Sequential is
    identity) and feed x_sensor's raw n_sensor channels straight into the
    LSTM, concatenated with x_temporal. lstm.input_size must be
    n_sensor + n_temporal, not cnn_filters + n_temporal.
    """
    from cnn_rnn_yield import CNNRNN, DEVICE

    n_sensor, n_temporal, seq_len, batch = 5, 2, 4, 3

    model = CNNRNN(n_sensor=n_sensor, n_temporal=n_temporal,
                   cnn_filters=32, cnn_kernel_size=2, cnn_padding=1,
                   num_cnn_blocks=0, lstm_hidden=16, lstm_layers=1,
                   dropout=0.0, fc_hidden=8, n_out=1).to(DEVICE)

    assert len(model.cnn) == 0
    assert model.lstm.input_size == n_sensor + n_temporal

    x_sensor = torch.randn(batch, seq_len, n_sensor).to(DEVICE)
    x_temporal = torch.randn(batch, seq_len, n_temporal).to(DEVICE)
    out = model(x_sensor, x_temporal)
    assert out.shape == (batch, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && python -m pytest Models/tests/test_cnn_rnn_yield.py::test_cnnrnn_num_cnn_blocks_zero_runs_forward -v`
Expected: FAIL — `RuntimeError` from the LSTM (input size mismatch: `x` has `n_sensor` channels but `lstm.input_size` was built as `cnn_filters + n_temporal`).

- [ ] **Step 3: Implement `num_cnn_blocks=0` support**

In `Models/cnn_rnn_yield.py`, replace the fixed `lstm.input_size` inside `CNNRNN.__init__`:

```python
        cnn_blocks = []
        in_ch = n_sensor
        for _ in range(num_cnn_blocks):
            cnn_blocks.append(CNNBlock(in_ch, cnn_filters, cnn_kernel_size,
                                       cnn_padding, dropout))
            in_ch = cnn_filters
        self.cnn = nn.Sequential(*cnn_blocks)

        lstm_in = in_ch + n_temporal
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
```

`in_ch` already equals `n_sensor` when `num_cnn_blocks=0` (loop never executes) and `cnn_filters` otherwise — reusing it avoids a separate conditional. `forward()` needs no change: `self.cnn(x)` on an empty `nn.Sequential` returns `x` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && python -m pytest Models/tests/test_cnn_rnn_yield.py::test_cnnrnn_num_cnn_blocks_zero_runs_forward -v`
Expected: PASS

- [ ] **Step 5: Run full existing suite to confirm no regression**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && python -m pytest Models/tests/test_cnn_rnn_yield.py -v`
Expected: all tests pass (22/22 — 21 pre-existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add Models/cnn_rnn_yield.py Models/tests/test_cnn_rnn_yield.py
git commit -m "feat(model): support num_cnn_blocks=0 as pure-LSTM toggle in CNNRNN"
```

---

### Task 2: New hyperparameter configs for pure-LSTM

**Files:**
- Create: `Models/hp_inv3_lstm.yaml`
- Create: `Models/hp_inv4_lstm.yaml`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: two yaml files with the same key set as `hp_inv3.yaml`/`hp_inv4.yaml`, loaded by `yaml.safe_load` in Task 3's scripts as `HP` — same keys read there: `seq_len`, `skip_first_weeks`, `ramp_weeks`, `ramp_weight`, `use_gap_norm`, `cnn_filters`, `cnn_kernel_size`, `cnn_padding`, `num_cnn_blocks`, `lstm_hidden`, `lstm_layers`, `fc_hidden`, `dropout`, `learning_rate`, `weight_decay`, `corr_weight`, `epochs`, `patience`, `init_methods`, `seeds`.

- [ ] **Step 1: Create `Models/hp_inv3_lstm.yaml`**

Exact copy of `Models/hp_inv3.yaml` with only `num_cnn_blocks` changed from `1` to `0`:

```yaml
seq_len: 4
skip_first_weeks: 3
ramp_weeks: 4
ramp_weight: 0.5
alignment: positional
batch_size: 8
lag_features: []
include_rolling_mean: false
use_spline_feature: false
use_gap_norm: true
temporal_keep: [dias_desde_transplante, week_in_season]

# CNN
cnn_filters: 32
cnn_kernel_size: 2
cnn_padding: 1
num_cnn_blocks: 0

# RNN
lstm_hidden: 64
lstm_layers: 1
fc_hidden: 32

# Training
dropout: 0.3
learning_rate: 0.002
weight_decay: 0.0001
corr_weight: 0.8
epochs: 500
patience: 150
init_methods: [default, xavier, orthogonal, lecun]
seeds: [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
        11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        100, 200, 500, 1000]
```

- [ ] **Step 2: Create `Models/hp_inv4_lstm.yaml`**

Exact copy of `Models/hp_inv4.yaml` with only `num_cnn_blocks` changed from `1` to `0`:

```yaml
seq_len: 2
skip_first_weeks: 3
ramp_weeks: 4
ramp_weight: 0.5
alignment: positional
batch_size: 8
lag_features: []
include_rolling_mean: false
use_spline_feature: false
use_gap_norm: true
temporal_keep: [dias_desde_transplante, week_in_season]

# CNN
cnn_filters: 32
cnn_kernel_size: 2
cnn_padding: 1
num_cnn_blocks: 0

# RNN
lstm_hidden: 64
lstm_layers: 1
fc_hidden: 32

# Training
dropout: 0.3
learning_rate: 0.002
weight_decay: 0.0001
corr_weight: 0.5
epochs: 1000
patience: 250
init_methods: [default, xavier, orthogonal, lecun]
seeds: [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
        11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        100, 200, 500, 1000]
```

- [ ] **Step 3: Verify both files parse and only `num_cnn_blocks` differs from originals**

Run:
```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction
python -c "import yaml; yaml.safe_load(open('Models/hp_inv3_lstm.yaml')); yaml.safe_load(open('Models/hp_inv4_lstm.yaml')); print('OK')"
diff Models/hp_inv3.yaml Models/hp_inv3_lstm.yaml
diff Models/hp_inv4.yaml Models/hp_inv4_lstm.yaml
```
Expected: `OK`, then each `diff` shows exactly one changed line (`num_cnn_blocks: 1` → `num_cnn_blocks: 0`).

- [ ] **Step 4: Commit**

```bash
git add Models/hp_inv3_lstm.yaml Models/hp_inv4_lstm.yaml
git commit -m "feat(config): add pure-LSTM hyperparameter configs for inv3/inv4"
```

---

### Task 3: New grid-search scripts for pure-LSTM (inv3)

**Files:**
- Create: `Models/cnn_rnn_inv3_lstm.py`

**Interfaces:**
- Consumes: `Models/hp_inv3_lstm.yaml` (Task 2); `CNNRNN` (Task 1, called with `num_cnn_blocks=HP['num_cnn_blocks']` which is `0`); all functions imported from `cnn_rnn_yield` exactly as `cnn_rnn_inv3.py` does — `DEVICE, RESULTS_DIR, HORIZON, prepare_data, set_seed, init_weights, CNNRNN, train_model, evaluate_model, evaluate_model_q, compute_metrics, compute_phase_metrics, print_metrics, plot_results_per_greenhouse, plot_quantile_bands, compute_cptc_intervals`.
- Produces: `Models/results/inv3_lstm.log` (via stdout redirect at run time), `Models/results/cnn_rnn_inv3_lstm_metrics.csv`, `Models/results/cnn_rnn_inv3_lstm_predictions.npz`, `Models/results/best_cnn_rnn_inv3_lstm.pt`, `Models/results/cnn_rnn_inv3_lstm_results.png` (via `plot_results_per_greenhouse`'s existing per-`INV_ID` path convention — see Step 1 note on `INV_ID`).

- [ ] **Step 1: Create `Models/cnn_rnn_inv3_lstm.py`**

Copy `Models/cnn_rnn_inv3.py` verbatim, then apply exactly these changes: load `hp_inv3_lstm.yaml` instead of `hp_inv3.yaml`, and redirect every `results/*inv3*` output path to its `*inv3_lstm*` equivalent (metrics CSV, predictions npz, model checkpoint, pipeline pickle, print statements). `INV_ID` stays `3` (unchanged) since `plot_results_per_greenhouse`/`print_metrics` key off it for display labels only, not file paths — those are set explicitly via the local path variables already used in this script (`model_path`, `RESULTS_DIR / 'cnn_rnn_inv3_metrics.csv'`, etc.), so pointing those variables at `_lstm` filenames is sufficient to keep outputs fully separate from the CNN version.

```python
"""
Pure LSTM (no CNN) — Invernadero 3
===================================
Train: T13-T15 | Val: T16 | Test: T17
Architecture comparison against CNN+LSTM production winner (see
docs/superpowers/specs/2026-07-13-pure-lstm-comparison-design.md).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, evaluate_model_q,
    compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, plot_quantile_bands,
    compute_cptc_intervals,
)

INV_ID = 3

_hp_path = Path(__file__).parent / 'hp_inv3_lstm.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  Pure LSTM (no CNN) — Invernadero 3')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print(f'  HORIZON={HORIZON} (doc) | seq_len={HP["seq_len"]} | num_cnn_blocks={HP["num_cnn_blocks"]}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'pipeline_inv3_lstm.pkl')

    n_runs = len(HP['seeds']) * len(HP['init_methods'])
    print(f'\n  {n_runs} runs ({len(HP["seeds"])} seeds × {len(HP["init_methods"])} inits)')

    best_r2     = -float('inf')
    best_result = None
    top_runs    = []
    model_path  = RESULTS_DIR / 'best_cnn_rnn_inv3_lstm.pt'
    tmp_ckpt    = RESULTS_DIR / '_tmp_cnn_rnn_inv3_lstm.pt'

    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    for init_method in HP['init_methods']:
        for seed in HP['seeds']:
            set_seed(seed)
            model = CNNRNN(
                n_sensor=n_sensor_pca, n_temporal=n_temporal,
                cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
                cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
                lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
                dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
                n_out=_n_out,
            ).to(DEVICE)

            if init_method != 'default':
                model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, HP, tmp_ckpt, var_y_train=var_y_train)

            if HP.get('loss_type') == 'quantile':
                _quantiles = HP.get('quantiles', [0.1, 0.5, 0.9])
                y_true, q_preds, metrics = evaluate_model_q(model, test_loader, scaler_y, bc_lambda,
                                                             quantiles=_quantiles)
                y_pred = q_preds.get(0.5, list(q_preds.values())[0])
            else:
                y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
                q_preds = None
            r2 = metrics['R²']
            print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
            top_runs.append((r2, y_pred.copy()))

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'seed': seed, 'init_method': init_method,
                    'wis_test': wis_test,
                    'q_preds': q_preds,
                }
                torch.save(model.state_dict(), model_path)

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    # Ensemble top-20
    top_runs.sort(key=lambda x: x[0], reverse=True)
    top_k = min(20, len(top_runs))
    ensemble_pred    = np.mean([r[1] for r in top_runs[:top_k]], axis=0)
    ensemble_metrics = compute_metrics(best_result['y_true'], ensemble_pred)
    best_result['ensemble_pred']    = ensemble_pred
    best_result['ensemble_metrics'] = ensemble_metrics

    # Prediction Intervals: use q10/q90 directly in quantile mode, CPTC otherwise
    if HP.get('loss_type') == 'quantile':
        q_preds_best = best_result.get('q_preds', {})
        lower = q_preds_best.get(0.1, np.zeros(len(best_result['y_true'])))
        upper = q_preds_best.get(0.9, np.zeros(len(best_result['y_true'])))
        y_true_arr = best_result['y_true']
        covered = np.mean((y_true_arr >= lower) & (y_true_arr <= upper))
        avg_w = float(np.mean(upper - lower))
        val_cov = float(covered)
        print(f'  80% PI (q10–q90): coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')
    else:
        best_model = CNNRNN(
            n_sensor=n_sensor_pca, n_temporal=n_temporal,
            cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
            cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
            lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
            dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
            n_out=_n_out,
        ).to(DEVICE)
        best_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        y_val_true, y_val_pred, _ = evaluate_model(best_model, val_loader, scaler_y, bc_lambda)
        lower, upper, val_cov, avg_w = compute_cptc_intervals(y_val_true, y_val_pred, best_result['y_pred'])
        best_result['pi_lower'] = lower
        best_result['pi_upper'] = upper
        print(f'  90% PI (CPTC): val_coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')

    print(f'\n  >>> Best: [{best_result["init_method"]}] seed={best_result["seed"]}  R²={best_r2:.4f} <<<')
    print(f'  >>> Ensemble top-{top_k}: R²={ensemble_metrics["R²"]:.4f}, MAPE={ensemble_metrics["MAPE (%)"]:.2f}% <<<')
    print_metrics(INV_ID, best_result['metrics'], horizon=HORIZON)

    if HP.get('loss_type') == 'quantile' and best_result.get('q_preds') is not None:
        plot_quantile_bands(
            y_true=best_result['y_true'],
            q_preds=best_result['q_preds'],
            train_losses=best_result['train_losses'],
            val_losses=best_result['val_losses'],
            inv_id=INV_ID,
            metrics=best_result['metrics'],
            horizon=HORIZON,
            quantiles=HP.get('quantiles', [0.1, 0.5, 0.9]),
            wis=best_result.get('wis_test'),
            ramp_weeks=HP.get('ramp_weeks', 4),
        )
    else:
        plot_results_per_greenhouse({INV_ID: best_result}, horizon=HORIZON,
                                    intervals={INV_ID: (lower, upper)})

    # Top-25 R² statistics
    all_r2s = sorted([r[0] for r in top_runs], reverse=True)
    top25_r2s = all_r2s[:25]
    top25_mean = float(np.mean(top25_r2s))
    top25_std  = float(np.std(top25_r2s))
    print(f'\n  Top-25 R²: mean={top25_mean:.4f}, std={top25_std:.4f}')

    phase = compute_phase_metrics(
        best_result['y_true'], best_result['y_pred'], wis_test,
        ramp_weeks=HP.get('ramp_weeks', 4))

    metrics_rows = [
        {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
        {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
        {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
         'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
        {'invernadero': INV_ID, 'model': 'quantile_pi' if HP.get('loss_type') == 'quantile' else 'cptc_pi', 'pi_coverage': val_cov, 'pi_avg_width': avg_w,
         'R²': None, 'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    ]
    if 'ramp_up' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_ramp_up', **phase['ramp_up']})
    if 'peak' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_peak',    **phase['peak']})
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / 'cnn_rnn_inv3_lstm_metrics.csv', index=False)

    np.savez_compressed(
        RESULTS_DIR / 'cnn_rnn_inv3_lstm_predictions.npz',
        y_true=best_result['y_true'],
        y_pred=best_result['y_pred'],
        ensemble_pred=best_result['ensemble_pred'],
        pi_lower=lower,
        pi_upper=upper,
        week_keys=test_week_keys,
    )
    print(f'\nBest model → results/best_cnn_rnn_inv3_lstm.pt')
    print(f'Metrics    → results/cnn_rnn_inv3_lstm_metrics.csv')
    print(f'Predictions → results/cnn_rnn_inv3_lstm_predictions.npz')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Sanity-check the script parses and imports cleanly**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && python -c "import ast; ast.parse(open('Models/cnn_rnn_inv3_lstm.py').read())" && echo OK`
Expected: `OK`

- [ ] **Step 3: Confirm output paths are fully separate from the CNN version's**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && grep -n "RESULTS_DIR /" Models/cnn_rnn_inv3_lstm.py`
Expected: every path contains `_lstm` (i.e. `pipeline_inv3_lstm.pkl`, `best_cnn_rnn_inv3_lstm.pt`, `_tmp_cnn_rnn_inv3_lstm.pt`, `cnn_rnn_inv3_lstm_metrics.csv`, `cnn_rnn_inv3_lstm_predictions.npz`) — none collide with `cnn_rnn_inv3.py`'s paths.

- [ ] **Step 4: Commit**

```bash
git add Models/cnn_rnn_inv3_lstm.py
git commit -m "feat(model): add pure-LSTM grid-search script for inv3"
```

---

### Task 4: New grid-search script for pure-LSTM (inv4)

**Files:**
- Create: `Models/cnn_rnn_inv4_lstm.py`

**Interfaces:**
- Consumes: `Models/hp_inv4_lstm.yaml` (Task 2); same `cnn_rnn_yield` imports as Task 3.
- Produces: `Models/results/inv4_lstm.log` (via stdout redirect at run time), `Models/results/cnn_rnn_inv4_lstm_metrics.csv`, `Models/results/cnn_rnn_inv4_lstm_predictions.npz`, `Models/results/best_cnn_rnn_inv4_lstm.pt`, `Models/results/cnn_rnn_inv4_lstm_results.png`.

- [ ] **Step 1: Create `Models/cnn_rnn_inv4_lstm.py`**

Same structure as Task 3's `cnn_rnn_inv3_lstm.py`, with `INV_ID = 4`, hp file `hp_inv4_lstm.yaml`, `TRAIN_SEASONS = ['T13', 'T14', 'T15']`, `VAL_SEASON = 'T16'`, and every `inv3`/`inv3_lstm` path/string replaced with `inv4`/`inv4_lstm`:

```python
"""
Pure LSTM (no CNN) — Invernadero 4
===================================
Train: T13-T15 | Val: T16 | Test: T17
Architecture comparison against CNN+LSTM production winner (see
docs/superpowers/specs/2026-07-13-pure-lstm-comparison-design.md).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, evaluate_model_q,
    compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, plot_quantile_bands,
    compute_cptc_intervals,
)

INV_ID = 4

_hp_path = Path(__file__).parent / 'hp_inv4_lstm.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  Pure LSTM (no CNN) — Invernadero 4')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print(f'  HORIZON={HORIZON} (doc) | seq_len={HP["seq_len"]} | num_cnn_blocks={HP["num_cnn_blocks"]}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'pipeline_inv4_lstm.pkl')

    n_runs = len(HP['seeds']) * len(HP['init_methods'])
    print(f'\n  {n_runs} runs ({len(HP["seeds"])} seeds × {len(HP["init_methods"])} inits)')

    best_r2     = -float('inf')
    best_result = None
    top_runs    = []
    model_path  = RESULTS_DIR / 'best_cnn_rnn_inv4_lstm.pt'
    tmp_ckpt    = RESULTS_DIR / '_tmp_cnn_rnn_inv4_lstm.pt'

    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    for init_method in HP['init_methods']:
        for seed in HP['seeds']:
            set_seed(seed)
            model = CNNRNN(
                n_sensor=n_sensor_pca, n_temporal=n_temporal,
                cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
                cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
                lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
                dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
                n_out=_n_out,
            ).to(DEVICE)

            if init_method != 'default':
                model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, HP, tmp_ckpt, var_y_train=var_y_train)

            if HP.get('loss_type') == 'quantile':
                _quantiles = HP.get('quantiles', [0.1, 0.5, 0.9])
                y_true, q_preds, metrics = evaluate_model_q(model, test_loader, scaler_y, bc_lambda,
                                                             quantiles=_quantiles)
                y_pred = q_preds.get(0.5, list(q_preds.values())[0])
            else:
                y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
                q_preds = None
            r2 = metrics['R²']
            print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
            top_runs.append((r2, y_pred.copy()))

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'seed': seed, 'init_method': init_method,
                    'wis_test': wis_test,
                    'q_preds': q_preds,
                }
                torch.save(model.state_dict(), model_path)

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    # Ensemble top-20
    top_runs.sort(key=lambda x: x[0], reverse=True)
    top_k = min(20, len(top_runs))
    ensemble_pred    = np.mean([r[1] for r in top_runs[:top_k]], axis=0)
    ensemble_metrics = compute_metrics(best_result['y_true'], ensemble_pred)
    best_result['ensemble_pred']    = ensemble_pred
    best_result['ensemble_metrics'] = ensemble_metrics

    # Prediction Intervals: use q10/q90 directly in quantile mode, CPTC otherwise
    if HP.get('loss_type') == 'quantile':
        q_preds_best = best_result.get('q_preds', {})
        lower = q_preds_best.get(0.1, np.zeros(len(best_result['y_true'])))
        upper = q_preds_best.get(0.9, np.zeros(len(best_result['y_true'])))
        y_true_arr = best_result['y_true']
        covered = np.mean((y_true_arr >= lower) & (y_true_arr <= upper))
        avg_w = float(np.mean(upper - lower))
        val_cov = float(covered)
        print(f'  80% PI (q10–q90): coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')
    else:
        best_model = CNNRNN(
            n_sensor=n_sensor_pca, n_temporal=n_temporal,
            cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
            cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
            lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
            dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
            n_out=_n_out,
        ).to(DEVICE)
        best_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        y_val_true, y_val_pred, _ = evaluate_model(best_model, val_loader, scaler_y, bc_lambda)
        lower, upper, val_cov, avg_w = compute_cptc_intervals(y_val_true, y_val_pred, best_result['y_pred'])
        best_result['pi_lower'] = lower
        best_result['pi_upper'] = upper
        print(f'  90% PI (CPTC): val_coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')

    print(f'\n  >>> Best: [{best_result["init_method"]}] seed={best_result["seed"]}  R²={best_r2:.4f} <<<')
    print(f'  >>> Ensemble top-{top_k}: R²={ensemble_metrics["R²"]:.4f}, MAPE={ensemble_metrics["MAPE (%)"]:.2f}% <<<')
    print_metrics(INV_ID, best_result['metrics'], horizon=HORIZON)

    if HP.get('loss_type') == 'quantile' and best_result.get('q_preds') is not None:
        plot_quantile_bands(
            y_true=best_result['y_true'],
            q_preds=best_result['q_preds'],
            train_losses=best_result['train_losses'],
            val_losses=best_result['val_losses'],
            inv_id=INV_ID,
            metrics=best_result['metrics'],
            horizon=HORIZON,
            quantiles=HP.get('quantiles', [0.1, 0.5, 0.9]),
            wis=best_result.get('wis_test'),
            ramp_weeks=HP.get('ramp_weeks', 4),
        )
    else:
        plot_results_per_greenhouse({INV_ID: best_result}, horizon=HORIZON,
                                    intervals={INV_ID: (lower, upper)})

    # Top-25 R² statistics
    all_r2s = sorted([r[0] for r in top_runs], reverse=True)
    top25_r2s = all_r2s[:25]
    top25_mean = float(np.mean(top25_r2s))
    top25_std  = float(np.std(top25_r2s))
    print(f'\n  Top-25 R²: mean={top25_mean:.4f}, std={top25_std:.4f}')

    phase = compute_phase_metrics(
        best_result['y_true'], best_result['y_pred'], wis_test,
        ramp_weeks=HP.get('ramp_weeks', 4))

    metrics_rows = [
        {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
        {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
        {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
         'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
        {'invernadero': INV_ID, 'model': 'quantile_pi' if HP.get('loss_type') == 'quantile' else 'cptc_pi', 'pi_coverage': val_cov, 'pi_avg_width': avg_w,
         'R²': None, 'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    ]
    if 'ramp_up' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_ramp_up', **phase['ramp_up']})
    if 'peak' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_peak',    **phase['peak']})
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / 'cnn_rnn_inv4_lstm_metrics.csv', index=False)

    np.savez_compressed(
        RESULTS_DIR / 'cnn_rnn_inv4_lstm_predictions.npz',
        y_true=best_result['y_true'],
        y_pred=best_result['y_pred'],
        ensemble_pred=best_result['ensemble_pred'],
        pi_lower=lower,
        pi_upper=upper,
        week_keys=test_week_keys,
    )
    print(f'\nBest model → results/best_cnn_rnn_inv4_lstm.pt')
    print(f'Metrics    → results/cnn_rnn_inv4_lstm_metrics.csv')
    print(f'Predictions → results/cnn_rnn_inv4_lstm_predictions.npz')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Sanity-check the script parses and imports cleanly**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && python -c "import ast; ast.parse(open('Models/cnn_rnn_inv4_lstm.py').read())" && echo OK`
Expected: `OK`

- [ ] **Step 3: Confirm output paths are fully separate from the CNN version's**

Run: `cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction && grep -n "RESULTS_DIR /" Models/cnn_rnn_inv4_lstm.py`
Expected: every path contains `_lstm` (`pipeline_inv4_lstm.pkl`, `best_cnn_rnn_inv4_lstm.pt`, `_tmp_cnn_rnn_inv4_lstm.pt`, `cnn_rnn_inv4_lstm_metrics.csv`, `cnn_rnn_inv4_lstm_predictions.npz`).

- [ ] **Step 4: Commit**

```bash
git add Models/cnn_rnn_inv4_lstm.py
git commit -m "feat(model): add pure-LSTM grid-search script for inv4"
```

---

### Task 5: Run both grid searches and report the comparison

**Files:**
- No new source files. Produces run artifacts in `Models/results/`: `inv3_lstm.log`, `inv4_lstm.log`, `cnn_rnn_inv3_lstm_metrics.csv`, `cnn_rnn_inv4_lstm_metrics.csv`, `cnn_rnn_inv3_lstm_predictions.npz`, `cnn_rnn_inv4_lstm_predictions.npz`, `best_cnn_rnn_inv3_lstm.pt`, `best_cnn_rnn_inv4_lstm.pt`, `cnn_rnn_inv3_lstm_results.png`, `cnn_rnn_inv4_lstm_results.png`, `pipeline_inv3_lstm.pkl`, `pipeline_inv4_lstm.pkl`.

**Interfaces:**
- Consumes: `Models/cnn_rnn_inv3_lstm.py` and `Models/cnn_rnn_inv4_lstm.py` (Tasks 3-4) — run as standalone scripts, no imports by later code.
- Produces: a written comparison summary (see Step 3) — this is the deliverable a human reads; no other task depends on it programmatically.

- [ ] **Step 1: Run inv3 pure-LSTM grid search (216 runs), sequentially — do not run inv4 concurrently**

```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction/Models
nohup python cnn_rnn_inv3_lstm.py > results/inv3_lstm.log 2>&1 < /dev/null &
disown
```

Poll until finished (script exits — check via `tail -30 results/inv3_lstm.log` for the closing `Predictions →` line, or `ps aux | grep cnn_rnn_inv3_lstm` to confirm the process ended). Expected: log ends with:
```
Best model → results/best_cnn_rnn_inv3_lstm.pt
Metrics    → results/cnn_rnn_inv3_lstm_metrics.csv
Predictions → results/cnn_rnn_inv3_lstm_predictions.npz
```

- [ ] **Step 2: Run inv4 pure-LSTM grid search (216 runs), only after inv3 finished**

```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction/Models
nohup python cnn_rnn_inv4_lstm.py > results/inv4_lstm.log 2>&1 < /dev/null &
disown
```

Poll the same way. Expected: log ends with the equivalent `inv4_lstm` closing lines.

- [ ] **Step 3: Extract each run's best R²/RMSE and compare against the production winners**

```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction
grep ">>> Best:" Models/results/inv3_lstm.log
grep ">>> Best:" Models/results/inv4_lstm.log
python -c "
import pandas as pd
for inv in (3, 4):
    df = pd.read_csv(f'Models/results/cnn_rnn_inv{inv}_lstm_metrics.csv')
    row = df[df['model'] == 'best'].iloc[0]
    print(f'inv{inv} pure-LSTM best: R2={row[\"R²\"]:.4f}, RMSE={row[\"RMSE (kg)\"]:.1f} kg')
"
```

Compare each greenhouse's printed `R2`/`RMSE` against the Global Constraints thresholds:
- Inv3 pure-LSTM wins only if `R² > 0.5286 AND RMSE < 5661.0`.
- Inv4 pure-LSTM wins only if `R² > 0.5595 AND RMSE < 6572.9`.

Write the result (win/lose per greenhouse, with the actual numbers) into the plan's final report — this is the task's deliverable. No code changes result from a "win" per the spec's Outcome section (§5): report only, no automatic production refit or swap.

- [ ] **Step 4: Confirm existing production artifacts and CNN grid-search outputs are untouched**

```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction
git status --short Models/results/
```

Expected: only the new `*_lstm*` files appear as untracked/new — `best_cnn_rnn_inv3.pt`, `best_cnn_rnn_inv4.pt`, `production_cnn_rnn_inv3.pt`, `production_cnn_rnn_inv4.pt`, `cnn_rnn_inv3_metrics.csv`, `cnn_rnn_inv4_metrics.csv`, `inv3.log`, `inv4.log`, and their pipeline pickles show no diff.

- [ ] **Step 5: Commit the run artifacts and comparison result**

```bash
cd /Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction
git add Models/results/inv3_lstm.log Models/results/inv4_lstm.log \
        Models/results/cnn_rnn_inv3_lstm_metrics.csv Models/results/cnn_rnn_inv4_lstm_metrics.csv \
        Models/results/cnn_rnn_inv3_lstm_predictions.npz Models/results/cnn_rnn_inv4_lstm_predictions.npz \
        Models/results/best_cnn_rnn_inv3_lstm.pt Models/results/best_cnn_rnn_inv4_lstm.pt \
        Models/results/cnn_rnn_inv3_lstm_results.png Models/results/cnn_rnn_inv4_lstm_results.png \
        Models/results/pipeline_inv3_lstm.pkl Models/results/pipeline_inv4_lstm.pkl
git commit -m "chore(results): pure-LSTM grid-search results for inv3/inv4 comparison"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (model change) → Task 1. §2 (new configs) → Task 2. §3 (new scripts) → Tasks 3-4. §4 (run and compare) → Task 5. §5 (outcome: report only, no auto-refit) → Task 5 Step 3 explicitly states no code changes follow a win. Out-of-scope items (no HORIZON/seq_len/gap-norm changes, no other model families, no production swap) are respected — no task touches those.
- **Placeholder scan:** none found — every step has concrete code, exact paths, and exact commands.
- **Type consistency:** `CNNRNN.__init__` signature unchanged across Task 1 and Tasks 3-4's call sites (`n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding, num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden, n_out`) — matches existing `cnn_rnn_inv3.py`/`cnn_rnn_inv4.py` usage exactly, only `num_cnn_blocks` value differs (via yaml, not code).
