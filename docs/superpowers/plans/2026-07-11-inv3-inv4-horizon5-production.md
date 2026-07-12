# Inv3/Inv4 HORIZON=5 Retune + Production Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the CNN-RNN pipeline's target prediction horizon from 4 to 5 weeks for both greenhouses, fix inv4's per-season effective-horizon instability via gap normalization + per-greenhouse `seq_len` tuning, then produce final deployable ("production") model weights trained on all 5 seasons for both inv3 and inv4.

**Architecture:** All changes build on the existing `Models/cnn_rnn_yield.py` shared pipeline — no new abstractions. One constant change (`HORIZON`), two yaml hyperparameter tweaks (`use_gap_norm`, `seq_len`), two grid-search re-runs using the existing per-greenhouse scripts unchanged, then two new thin "production" scripts that reuse `prepare_data`/`train_model` exactly as-is with different season/epoch arguments.

**Tech Stack:** Python, PyTorch, pandas, scikit-learn (PCA/scalers), pytest.

## Global Constraints

- `HORIZON = 5` (was 4) applies to both greenhouses — spec: `docs/superpowers/specs/2026-07-11-inv3-inv4-horizon5-production-design.md` §1.
- `use_gap_norm: true` in both `hp_inv3.yaml` and `hp_inv4.yaml` — spec §2.
- `seq_len: 4` for inv3, `seq_len: 2` for inv4 — spec §2 (chosen so gap-norm trims every inv3 season to exactly `eff_horizon=5`; inv4's T16 remains 1 week short at `eff_horizon=4`, an accepted structural limit given `gap=6`).
- Production refit pools all 5 seasons (`T13`–`T17`) into training, fixed epoch count (no early stopping), single winning seed/init from the grid search — spec §4. Production weights saved to `Models/results/production_cnn_rnn_inv3.pt` / `_inv4.pt`, never overwriting the eval checkpoints `best_cnn_rnn_inv3.pt` / `_inv4.pt`.
- No changes to feature engineering, model architecture, or other model families (ARIMAX/XGBoost/SVR/ElasticNet) — spec "Out of scope".

---

### Task 1: HORIZON constant → 5

**Files:**
- Modify: `Models/cnn_rnn_yield.py:52`
- Test: `Models/tests/test_cnn_rnn_yield.py`

**Interfaces:**
- Produces: module-level `cnn_rnn_yield.HORIZON == 5`, consumed by `_apply_gap_norm_cnn` (default `horizon=HORIZON`) and by `cnn_rnn_inv3.py`/`cnn_rnn_inv4.py` (`print(...HORIZON...)`).

- [ ] **Step 1: Write the failing test**

Add to the end of `Models/tests/test_cnn_rnn_yield.py`:

```python
def test_horizon_is_5():
    """Target prediction horizon changed from 4 to 5 weeks (2026-07-11 retune)."""
    from cnn_rnn_yield import HORIZON
    assert HORIZON == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Models && python -m pytest tests/test_cnn_rnn_yield.py::test_horizon_is_5 -v`
Expected: FAIL — `assert 4 == 5`

- [ ] **Step 3: Change the constant**

In `Models/cnn_rnn_yield.py`, line 52, change:

```python
HORIZON = 4   # fixed prediction horizon (weeks); seq_len = gap - HORIZON + skip_first_weeks
```

to:

```python
HORIZON = 5   # fixed prediction horizon (weeks); per-season alignment via _apply_gap_norm_cnn (seq_len set per greenhouse in hp_inv*.yaml)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Models && python -m pytest tests/test_cnn_rnn_yield.py::test_horizon_is_5 -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `cd Models && python -m pytest tests/test_cnn_rnn_yield.py -v`
Expected: all tests PASS (no test in this file hardcodes `HORIZON=4` as an expected value — confirm by reading output).

- [ ] **Step 6: Commit**

```bash
git add Models/cnn_rnn_yield.py Models/tests/test_cnn_rnn_yield.py
git commit -m "feat(cnn-rnn): change target prediction horizon from 4 to 5 weeks"
```

---

### Task 2: Gap-norm regression test + per-greenhouse seq_len/use_gap_norm config

**Files:**
- Modify: `Models/hp_inv3.yaml:1,10`
- Modify: `Models/hp_inv4.yaml:1,10`
- Test: `Models/tests/test_cnn_rnn_yield.py`

**Interfaces:**
- Consumes: `cnn_rnn_yield._apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon)` (existing, unchanged — Task 1 produced `HORIZON=5` used as its default).
- Produces: `hp_inv3.yaml['seq_len'] == 4`, `hp_inv3.yaml['use_gap_norm'] == True`; `hp_inv4.yaml['seq_len'] == 2`, `hp_inv4.yaml['use_gap_norm'] == True`. Consumed by Task 3/4 (grid search re-run) and Task 5/6 (production scripts).

This task first characterizes (with a test) what `_apply_gap_norm_cnn` does for the exact `seq_len` values the design chose, confirming the row-trim counts match the spec's hand-computed table, then applies those values to the yaml configs.

- [ ] **Step 1: Write the regression tests**

Add to the end of `Models/tests/test_cnn_rnn_yield.py`:

```python
def _make_gap_norm_fixture(gaps, n_rows=20):
    """Build a synthetic sensor_df/prod_df pair for _apply_gap_norm_cnn tests.

    gaps: dict of {season: gap_weeks}. Each season gets n_rows sensor rows
    (indexed 0..n_rows-1 via the 'row' column) and a matching 1-row prod_df
    entry carrying 'sensor_gap'.
    """
    import pandas as pd
    sensor_df = pd.concat([
        pd.DataFrame({'temporada': [temp] * n_rows, 'row': range(n_rows)})
        for temp in gaps
    ], ignore_index=True)
    prod_df = pd.DataFrame({
        'temporada': list(gaps.keys()),
        'sensor_gap': list(gaps.values()),
    })
    return sensor_df, prod_df


def test_gap_norm_extra_skip_inv3_seq_len_4_horizon_5():
    """Inv3 gaps (11,10,10,10,11) with seq_len=4, horizon=5 all trim to
    eff_horizon=5 exactly — extra_skip = gap - seq_len - horizon."""
    from cnn_rnn_yield import _apply_gap_norm_cnn

    gaps = {'T13': 11, 'T14': 10, 'T15': 10, 'T16': 10, 'T17': 11}
    seq_len, horizon, n_rows = 4, 5, 20
    sensor_df, prod_df = _make_gap_norm_fixture(gaps, n_rows)

    out = _apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon=horizon)

    expected_extra_skip = {'T13': 2, 'T14': 1, 'T15': 1, 'T16': 1, 'T17': 2}
    for temp, skip in expected_extra_skip.items():
        kept = out[out['temporada'] == temp]
        assert len(kept) == n_rows - skip, (
            f'{temp}: expected {n_rows - skip} rows after trim, got {len(kept)}')
        assert kept['row'].iloc[0] == skip, (
            f'{temp}: expected first kept row index {skip}, got {kept["row"].iloc[0]}')
        eff_horizon = gaps[temp] - seq_len - skip
        assert eff_horizon == horizon, (
            f'{temp}: eff_horizon should be exactly {horizon}, got {eff_horizon}')


def test_gap_norm_extra_skip_inv4_seq_len_2_horizon_5():
    """Inv4 gaps (10,10,8,6,11) with seq_len=2, horizon=5: T13/T14/T15/T17
    trim to eff_horizon=5; T16 (gap=6) is untouched (extra_skip=0) and stays
    at its natural eff_horizon=4 — 1 week short of target, a structural limit."""
    from cnn_rnn_yield import _apply_gap_norm_cnn

    gaps = {'T13': 10, 'T14': 10, 'T15': 8, 'T16': 6, 'T17': 11}
    seq_len, horizon, n_rows = 2, 5, 20
    sensor_df, prod_df = _make_gap_norm_fixture(gaps, n_rows)

    out = _apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon=horizon)

    expected_extra_skip = {'T13': 3, 'T14': 3, 'T15': 1, 'T16': 0, 'T17': 4}
    expected_eff_horizon = {'T13': 5, 'T14': 5, 'T15': 5, 'T16': 4, 'T17': 5}
    for temp, skip in expected_extra_skip.items():
        kept = out[out['temporada'] == temp]
        assert len(kept) == n_rows - skip, (
            f'{temp}: expected {n_rows - skip} rows after trim, got {len(kept)}')
        eff_horizon = gaps[temp] - seq_len - skip
        assert eff_horizon == expected_eff_horizon[temp], (
            f'{temp}: expected eff_horizon={expected_eff_horizon[temp]}, got {eff_horizon}')
```

- [ ] **Step 2: Run tests to verify they pass immediately**

Run: `cd Models && python -m pytest tests/test_cnn_rnn_yield.py -k "gap_norm_extra_skip" -v`
Expected: both PASS. (`_apply_gap_norm_cnn` is existing, unmodified code — this step is a characterization check confirming the design doc's hand-computed trim table is correct, not a TDD red/green cycle.)

- [ ] **Step 3: Update `Models/hp_inv3.yaml`**

Change line 1 and line 10:

```yaml
seq_len: 4
```

```yaml
use_gap_norm: true
```

- [ ] **Step 4: Update `Models/hp_inv4.yaml`**

Change line 1 and line 10:

```yaml
seq_len: 2
```

```yaml
use_gap_norm: true
```

- [ ] **Step 5: Commit**

```bash
git add Models/tests/test_cnn_rnn_yield.py Models/hp_inv3.yaml Models/hp_inv4.yaml
git commit -m "feat(cnn-rnn): enable gap-norm, set seq_len=4 (inv3) / seq_len=2 (inv4) for horizon=5 alignment"
```

---

### Task 3: Re-run inv3 grid search, capture winning config

**Files:**
- Read-only: `Models/cnn_rnn_inv3.py` (unchanged — reads `hp_inv3.yaml`, which Task 2 already updated)
- Produces: `Models/results/inv3.log`, `Models/results/cnn_rnn_inv3_metrics.csv`, `Models/results/cnn_rnn_inv3_predictions.npz`, `Models/results/best_cnn_rnn_inv3.pt`, `Models/results/pipeline_inv3.pkl`

**Interfaces:**
- Produces: winning `(init_method, seed, early_stop_epoch)` triple, read from the log's `>>> Best: [...] seed=... <<<` line and the `Early stopping at epoch N` line immediately preceding it. This triple is consumed by Task 5.

- [ ] **Step 1: Run the grid search**

Run (from repo root): `python Models/cnn_rnn_inv3.py 2>&1 | tee Models/results/inv3.log`

Expected: prints `216 runs (54 seeds × 4 inits)` header, then per-run lines like `[lecun] s100: R²=0.xxxx, MAPE=xx.xx%`, ending with a `>>> Best: [init] seed=N  R²=0.xxxx <<<` line. This can take a long time (hours) — 216 sequential training runs.

- [ ] **Step 2: Identify the winning run's init/seed**

Run: `grep '>>> Best:' Models/results/inv3.log`

Expected output form: `  >>> Best: [init_method] seed=N  R²=0.xxxx <<<` — record `init_method` and `N`.

- [ ] **Step 3: Identify the winning run's early-stop epoch**

The winning run's `Early stopping at epoch E` line is the one immediately preceding its own `[init_method] sN: R²=...` line in the log (each run logs its own early-stop line right before its result line). Find it:

```bash
grep -B1 "\[$INIT\] s$SEED:" Models/results/inv3.log
```

(substitute `$INIT`/`$SEED` with the values from Step 2). Expected: two lines — `Early stopping at epoch E` followed by `[init] sN: R²=...`. Record `E`.

If the winning run did NOT early-stop (ran the full `epochs=500` from `hp_inv3.yaml`), there will be no `Early stopping` line for it — in that case use `E=500`.

- [ ] **Step 4: Confirm output artifacts exist**

Run: `ls -la Models/results/inv3.log Models/results/cnn_rnn_inv3_metrics.csv Models/results/cnn_rnn_inv3_predictions.npz Models/results/best_cnn_rnn_inv3.pt Models/results/pipeline_inv3.pkl`
Expected: all 5 files exist with a recent mtime.

- [ ] **Step 5: Commit the updated result artifacts**

```bash
git add Models/results/inv3.log Models/results/cnn_rnn_inv3_metrics.csv Models/results/cnn_rnn_inv3_predictions.npz Models/results/best_cnn_rnn_inv3.pt Models/results/cnn_rnn_inv3_results.png Models/results/pipeline_inv3.pkl
git commit -m "chore(inv3): grid search results under HORIZON=5, gap-norm, seq_len=4"
```

---

### Task 4: Re-run inv4 grid search, capture winning config

**Files:**
- Read-only: `Models/cnn_rnn_inv4.py` (unchanged — reads `hp_inv4.yaml`, which Task 2 already updated)
- Produces: `Models/results/inv4.log`, `Models/results/cnn_rnn_inv4_metrics.csv`, `Models/results/cnn_rnn_inv4_predictions.npz`, `Models/results/best_cnn_rnn_inv4.pt`

**Interfaces:**
- Produces: winning `(init_method, seed, early_stop_epoch)` triple for inv4, same extraction method as Task 3. Consumed by Task 6.

- [ ] **Step 1: Run the grid search**

Run (from repo root): `python Models/cnn_rnn_inv4.py 2>&1 | tee Models/results/inv4.log`

Expected: same structure as Task 3 Step 1, but `hp_inv4.yaml` has `epochs=1000`/`patience=250`, so this run takes noticeably longer.

- [ ] **Step 2: Identify the winning run's init/seed**

Run: `grep '>>> Best:' Models/results/inv4.log`

- [ ] **Step 3: Identify the winning run's early-stop epoch**

```bash
grep -B1 "\[$INIT\] s$SEED:" Models/results/inv4.log
```

Same fallback as Task 3 Step 3: if no `Early stopping` line precedes it, use `E=1000` (`hp_inv4.yaml`'s `epochs`).

- [ ] **Step 4: Confirm output artifacts exist**

Run: `ls -la Models/results/inv4.log Models/results/cnn_rnn_inv4_metrics.csv Models/results/cnn_rnn_inv4_predictions.npz Models/results/best_cnn_rnn_inv4.pt`
Expected: all 4 files exist with a recent mtime.

- [ ] **Step 5: Commit the updated result artifacts**

```bash
git add Models/results/inv4.log Models/results/cnn_rnn_inv4_metrics.csv Models/results/cnn_rnn_inv4_predictions.npz Models/results/best_cnn_rnn_inv4.pt Models/results/cnn_rnn_inv4_results.png
git commit -m "chore(inv4): grid search results under HORIZON=5, gap-norm, seq_len=2"
```

---

### Task 5: Inv3 production refit

**Files:**
- Create: `Models/cnn_rnn_inv3_production.py`
- Produces: `Models/results/production_cnn_rnn_inv3.pt`, `Models/results/production_pipeline_inv3.pkl`

**Interfaces:**
- Consumes: `cnn_rnn_yield.{DEVICE, RESULTS_DIR, HORIZON, prepare_data, set_seed, init_weights, CNNRNN, train_model}` (all existing, unchanged); Task 3's winning `(init_method, seed, early_stop_epoch)`.
- Produces: `Models/results/production_cnn_rnn_inv3.pt` (final deployable weights, trained on all 5 seasons), `Models/results/production_pipeline_inv3.pkl` (matching inference preprocessing pipeline).

- [ ] **Step 1: Write the production script**

Create `Models/cnn_rnn_inv3_production.py`. Replace `WINNING_INIT_METHOD`, `WINNING_SEED`, `WINNING_EPOCHS` with the values captured in Task 3 Steps 2–3 before running:

```python
"""
CNN-RNN — Invernadero 3 — PRODUCTION REFIT
===========================================
Trains the winning grid-search configuration on ALL 5 seasons (T13-T17)
pooled, with no holdout, for a fixed epoch count (no early stopping).
This produces the final deployable model — not an evaluation checkpoint.

Winning config source: Models/results/inv3.log (see docs/superpowers/plans/
2026-07-11-inv3-inv4-horizon5-production.md, Task 3).
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model,
)

INV_ID = 3

# ── Winning config from Task 3 (fill in from Models/results/inv3.log) ──
WINNING_INIT_METHOD = 'lecun'   # placeholder — replace with Task 3 Step 2 value
WINNING_SEED = 100              # placeholder — replace with Task 3 Step 2 value
WINNING_EPOCHS = 164            # placeholder — replace with Task 3 Step 3 value

_hp_path = Path(__file__).parent / 'hp_inv3.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

HP = copy.deepcopy(HP)
HP['epochs'] = WINNING_EPOCHS
HP['patience'] = WINNING_EPOCHS + 10  # never triggers before `epochs` completes

ALL_SEASONS = ['T13', 'T14', 'T15', 'T16', 'T17']


def main():
    print('=' * 60)
    print('  CNN-RNN Invernadero 3 — PRODUCTION REFIT')
    print(f'  Train: {ALL_SEASONS} (all data, no holdout)')
    print(f'  Config: init={WINNING_INIT_METHOD}, seed={WINNING_SEED}, epochs={WINNING_EPOCHS}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=ALL_SEASONS, val_season='T16',
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'production_pipeline_inv3.pkl')

    set_seed(WINNING_SEED)
    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    model = CNNRNN(
        n_sensor=n_sensor_pca, n_temporal=n_temporal,
        cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
        cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
        lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
        dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
        n_out=_n_out,
    ).to(DEVICE)

    if WINNING_INIT_METHOD != 'default':
        model = model.cpu(); init_weights(model, WINNING_INIT_METHOD); model = model.to(DEVICE)

    model_path = RESULTS_DIR / 'production_cnn_rnn_inv3.pt'
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, HP, model_path, var_y_train=var_y_train)

    print(f'\n  Trained {len(train_losses)} epochs (target was {WINNING_EPOCHS}).')
    print(f'  Final train loss: {train_losses[-1]:.6f}')
    print(f'\nProduction model  → {model_path}')
    print(f'Inference pipeline → {RESULTS_DIR / "production_pipeline_inv3.pkl"}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Fill in the winning config values**

Edit the three placeholder constants (`WINNING_INIT_METHOD`, `WINNING_SEED`, `WINNING_EPOCHS`) in `Models/cnn_rnn_inv3_production.py` using the values recorded in Task 3 Steps 2–3.

- [ ] **Step 3: Run the production refit**

Run (from repo root): `python Models/cnn_rnn_inv3_production.py`

Expected: prints the header, `prepare_data` logs (season sample counts — `Train samples` should now include all 5 seasons), periodic `Epoch .../<WINNING_EPOCHS>` lines from `train_model`, then `Trained <WINNING_EPOCHS> epochs (target was <WINNING_EPOCHS>).` confirming no early stop occurred, and the two output paths.

- [ ] **Step 4: Verify the saved model loads and has the expected shape**

Run:

```bash
cd Models && python -c "
import torch
from cnn_rnn_yield import CNNRNN
sd = torch.load('results/production_cnn_rnn_inv3.pt', map_location='cpu', weights_only=True)
print('Keys:', len(sd))
print('First key:', next(iter(sd)))
"
```

Expected: prints a nonzero key count and a key name like `cnn_blocks.0.conv1.parametrizations.weight.original0` (or similar `CNNRNN` submodule name) with no error.

- [ ] **Step 5: Confirm the production model path is distinct from the eval checkpoint**

Run: `ls -la Models/results/production_cnn_rnn_inv3.pt Models/results/best_cnn_rnn_inv3.pt`
Expected: two separate files with different mtimes — production refit did not overwrite the grid-search eval checkpoint.

- [ ] **Step 6: Commit**

```bash
git add Models/cnn_rnn_inv3_production.py Models/results/production_cnn_rnn_inv3.pt Models/results/production_pipeline_inv3.pkl
git commit -m "feat(inv3): production refit on all 5 seasons, fixed-epoch, no holdout"
```

---

### Task 6: Inv4 production refit

**Files:**
- Create: `Models/cnn_rnn_inv4_production.py`
- Produces: `Models/results/production_cnn_rnn_inv4.pt`, `Models/results/production_pipeline_inv4.pkl`

**Interfaces:**
- Consumes: same `cnn_rnn_yield` exports as Task 5; Task 4's winning `(init_method, seed, early_stop_epoch)`.
- Produces: `Models/results/production_cnn_rnn_inv4.pt`, `Models/results/production_pipeline_inv4.pkl`.

- [ ] **Step 1: Write the production script**

Create `Models/cnn_rnn_inv4_production.py` — identical structure to Task 5's script, with `INV_ID = 4`, `hp_inv4.yaml`, and pipeline filename swapped:

```python
"""
CNN-RNN — Invernadero 4 — PRODUCTION REFIT
===========================================
Trains the winning grid-search configuration on ALL 5 seasons (T13-T17)
pooled, with no holdout, for a fixed epoch count (no early stopping).
This produces the final deployable model — not an evaluation checkpoint.

Winning config source: Models/results/inv4.log (see docs/superpowers/plans/
2026-07-11-inv3-inv4-horizon5-production.md, Task 4).
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model,
)

INV_ID = 4

# ── Winning config from Task 4 (fill in from Models/results/inv4.log) ──
WINNING_INIT_METHOD = 'lecun'   # placeholder — replace with Task 4 Step 2 value
WINNING_SEED = 100              # placeholder — replace with Task 4 Step 2 value
WINNING_EPOCHS = 200             # placeholder — replace with Task 4 Step 3 value

_hp_path = Path(__file__).parent / 'hp_inv4.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

HP = copy.deepcopy(HP)
HP['epochs'] = WINNING_EPOCHS
HP['patience'] = WINNING_EPOCHS + 10  # never triggers before `epochs` completes

ALL_SEASONS = ['T13', 'T14', 'T15', 'T16', 'T17']


def main():
    print('=' * 60)
    print('  CNN-RNN Invernadero 4 — PRODUCTION REFIT')
    print(f'  Train: {ALL_SEASONS} (all data, no holdout)')
    print(f'  Config: init={WINNING_INIT_METHOD}, seed={WINNING_SEED}, epochs={WINNING_EPOCHS}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=ALL_SEASONS, val_season='T16',
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'production_pipeline_inv4.pkl')

    set_seed(WINNING_SEED)
    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    model = CNNRNN(
        n_sensor=n_sensor_pca, n_temporal=n_temporal,
        cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
        cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
        lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
        dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
        n_out=_n_out,
    ).to(DEVICE)

    if WINNING_INIT_METHOD != 'default':
        model = model.cpu(); init_weights(model, WINNING_INIT_METHOD); model = model.to(DEVICE)

    model_path = RESULTS_DIR / 'production_cnn_rnn_inv4.pt'
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, HP, model_path, var_y_train=var_y_train,
        track_best=False)

    print(f'\n  Trained {len(train_losses)} epochs (target was {WINNING_EPOCHS}).')
    print(f'  Final train loss: {train_losses[-1]:.6f}')
    print(f'\nProduction model  → {model_path}')
    print(f'Inference pipeline → {RESULTS_DIR / "production_pipeline_inv4.pkl"}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Fill in the winning config values**

Edit the three placeholder constants using the values recorded in Task 4 Steps 2–3.

- [ ] **Step 3: Run the production refit**

Run (from repo root): `python Models/cnn_rnn_inv4_production.py`

Expected: same structure as Task 5 Step 3.

- [ ] **Step 4: Verify the saved model loads and has the expected shape**

```bash
cd Models && python -c "
import torch
from cnn_rnn_yield import CNNRNN
sd = torch.load('results/production_cnn_rnn_inv4.pt', map_location='cpu', weights_only=True)
print('Keys:', len(sd))
print('First key:', next(iter(sd)))
"
```

Expected: same as Task 5 Step 4, no error.

- [ ] **Step 5: Confirm the production model path is distinct from the eval checkpoint**

Run: `ls -la Models/results/production_cnn_rnn_inv4.pt Models/results/best_cnn_rnn_inv4.pt`
Expected: two separate files with different mtimes.

- [ ] **Step 6: Commit**

```bash
git add Models/cnn_rnn_inv4_production.py Models/results/production_cnn_rnn_inv4.pt Models/results/production_pipeline_inv4.pkl
git commit -m "feat(inv4): production refit on all 5 seasons, fixed-epoch, no holdout"
```

---

### Task 7: Update CLAUDE.md §7 documentation

**Files:**
- Modify: `CLAUDE.md` (§7 "Horizonte de Predicción — Diseño seq_len")

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Replace the stale §7 section**

`CLAUDE.md`'s §7 currently documents `HORIZON=4`, `seq_len=6` fixed for both greenhouses, and `use_gap_norm` disabled. Replace its content with the actual post-retune state:

```markdown
## 7. Horizonte de Predicción — Diseño seq_len (actualizado 2026-07-11)

`HORIZON = 5` en `cnn_rnn_yield.py` es el horizonte objetivo real (semanas entre el
último dato del sensor en la ventana y la producción target). `use_gap_norm: true`
en ambos `hp_inv*.yaml` alinea el horizonte efectivo por temporada mediante
`_apply_gap_norm_cnn` (`extra_skip = max(0, gap - seq_len - HORIZON)`), recortando
solo cuando el gap es mayor a `seq_len + HORIZON` — nunca puede aumentar el horizonte
efectivo de una temporada con gap corto.

`seq_len` es fijo por invernadero (no varía por temporada — ver razón abajo):

| Invernadero | seq_len | Resultado |
|-------------|---------|-----------|
| Inv3        | 4       | Las 5 temporadas alcanzan eff_horizon=5 exacto (todos los gaps ≥ 10) |
| Inv4        | 2       | T13/T14/T15/T17 alcanzan eff_horizon=5; T16 (gap=6) queda en eff_horizon=4 — límite estructural, no corregible sin perder casi todo el contexto sensor (seq_len=1) |

**Por qué NO se usa seq_len variable por temporada:** con `seq_len` distinto por
temporada, las ventanas deben rellenarse con ceros (padding) hasta `max_seq_len`
para compartir forma en un batch. Un experimento previo con `seq_len = gap - HORIZON
+ skip_first_weeks` (HORIZON=4 entonces) dio `seq_len=3` en inv4 T16, con 5 de 8
filas de padding en la temporada de validación → R² muy bajo (~0.39). Se mantiene
`seq_len` fijo por invernadero; el ajuste por temporada lo hace únicamente
`_apply_gap_norm_cnn` recortando filas del sensor, sin padding.

Ver `docs/superpowers/specs/2026-07-11-inv3-inv4-horizon5-production-design.md`
para el diseño completo y `docs/superpowers/plans/2026-07-11-inv3-inv4-horizon5-production.md`
para el plan de implementación.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md §7 for HORIZON=5 retune and per-greenhouse seq_len"
```
