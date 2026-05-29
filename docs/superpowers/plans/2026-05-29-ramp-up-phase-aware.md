# Phase-Aware Ramp-Up Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Include ramp-up season weeks (0–3) in training by down-weighting them in the loss, enabling full-season predictions that were previously excluded by `skip_first_weeks=3`.

**Architecture:** Add `ramp_weeks`/`ramp_weight` HP params; compute per-sample weights from `pos_tr` (already returned by `make_sequences_per_season`); thread weights as 4th tensor through DataLoaders; pass to `YieldWMAELoss.sample_weights` in training loop. Phase-split metrics reported in output CSV.

**Tech Stack:** PyTorch, NumPy, pandas, PyYAML. Tests: pytest. No new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `Models/cnn_rnn_yield.py` | `prepare_data`: add `ramp_weeks`/`ramp_weight` params, compute `w_train`, 4-tensor `to_ds`, return `wis_test`; `train_model`: unpack 4-tuple, pass weights to loss; `evaluate_model` + `evaluate_loader`: unpack 4-tuple; add `compute_phase_metrics` |
| `Models/hp_inv3.yaml` | `skip_first_weeks: 0`, add `ramp_weeks: 4`, `ramp_weight: 0.5` |
| `Models/hp_inv4.yaml` | Same as inv3 yaml |
| `Models/cnn_rnn_inv3.py` | Pass `ramp_weeks`/`ramp_weight` to `prepare_data`; unpack `wis_test`; phase metrics in CSV |
| `Models/cnn_rnn_inv4.py` | Same as inv3 runner |
| `Models/tests/test_cnn_rnn_yield.py` | New: unit tests for weight logic + phase metrics |

---

## Task 1: Unit tests for sample weight computation and phase metrics

**Files:**
- Create: `Models/tests/test_cnn_rnn_yield.py`

These tests exercise pure-numpy logic that exists before and after the changes. Run them now — they must FAIL because `compute_phase_metrics` doesn't exist yet and the weight logic isn't wired up.

- [ ] **Step 1: Write the failing tests**

```python
# Models/tests/test_cnn_rnn_yield.py
"""Unit tests for phase-aware sample weight logic and compute_phase_metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


# ── sample weight computation ─────────────────────────────────────────────────

def test_ramp_weights_are_ramp_weight_for_early_weeks():
    """Weeks 0..ramp_weeks-1 get ramp_weight; rest get 1.0."""
    pos_tr = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    ramp_weeks = 4
    ramp_weight = 0.5
    w = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
    assert (w[:4] == 0.5).all(), f'Expected 0.5 for weeks 0-3, got {w[:4]}'
    assert (w[4:] == 1.0).all(), f'Expected 1.0 for weeks 4+, got {w[4:]}'


def test_ramp_weights_all_ones_when_ramp_weight_1():
    """ramp_weight=1.0 means no down-weighting — all weights are 1."""
    pos_tr = np.arange(10, dtype=np.int32)
    w = np.where(pos_tr < 4, 1.0, 1.0).astype(np.float32)
    assert (w == 1.0).all()


def test_ramp_weights_dtype_float32():
    pos_tr = np.array([0, 1, 5], dtype=np.int32)
    w = np.where(pos_tr < 4, 0.3, 1.0).astype(np.float32)
    assert w.dtype == np.float32


# ── compute_phase_metrics ─────────────────────────────────────────────────────

def test_compute_phase_metrics_returns_all_key():
    from cnn_rnn_yield import compute_phase_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    assert 'all' in result, 'Missing "all" key'


def test_compute_phase_metrics_splits_ramp_peak():
    from cnn_rnn_yield import compute_phase_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    assert 'ramp_up' in result, 'Missing "ramp_up" key'
    assert 'peak' in result, 'Missing "peak" key'


def test_compute_phase_metrics_ramp_uses_correct_samples():
    from cnn_rnn_yield import compute_phase_metrics, compute_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    expected_ramp = compute_metrics(y_true[:4], y_pred[:4])
    assert abs(result['ramp_up']['R²'] - expected_ramp['R²']) < 1e-6


def test_compute_phase_metrics_all_uses_full_array():
    from cnn_rnn_yield import compute_phase_metrics, compute_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    expected_all = compute_metrics(y_true, y_pred)
    assert abs(result['all']['R²'] - expected_all['R²']) < 1e-6


# ── 4-tensor DataLoader iteration ─────────────────────────────────────────────

def test_4tensor_dataset_unpacks_correctly():
    """TensorDataset with 4 tensors yields 4-tuples per batch."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    Xs = torch.randn(8, 6, 3)
    Xt = torch.randn(8, 6, 2)
    y  = torch.randn(8, 1)
    w  = torch.ones(8, 1)
    ds = TensorDataset(Xs, Xt, y, w)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for batch in loader:
        assert len(batch) == 4, f'Expected 4 tensors per batch, got {len(batch)}'
        Xs_b, Xt_b, y_b, w_b = batch
        assert w_b.shape == (4, 1)
        break
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd Models && python3 -m pytest tests/test_cnn_rnn_yield.py -v 2>&1 | head -40
```

Expected: `ImportError: cannot import name 'compute_phase_metrics'` for phase tests; weight/dataset tests should PASS (they test pure logic already correct). Mark dataset/weight tests as passing baseline; `compute_phase_metrics` tests will fail.

- [ ] **Step 3: Commit failing tests**

```bash
git add Models/tests/test_cnn_rnn_yield.py
git commit -m "test(cnn_rnn): add failing tests for phase metrics and 4-tensor DataLoader"
```

---

## Task 2: Add `compute_phase_metrics` to `cnn_rnn_yield.py`

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — insert after `compute_metrics` (line ~1140)

- [ ] **Step 1: Insert `compute_phase_metrics` after `compute_metrics`**

Find the line `def evaluate_model(` (line ~1142). Insert the new function immediately before it:

```python
def compute_phase_metrics(y_true, y_pred, wis_test, ramp_weeks=4):
    """Compute metrics split by season phase (ramp-up vs peak).

    Args:
        y_true: 1-D array of true kg values (original scale).
        y_pred: 1-D array of predicted kg values (original scale).
        wis_test: 1-D int array of week_in_season per test sample.
        ramp_weeks: number of weeks considered ramp-up (week_in_season < ramp_weeks).

    Returns:
        dict with keys 'all', 'ramp_up' (if ≥2 ramp samples), 'peak' (if ≥2 peak samples).
        Each value is the output of compute_metrics().
    """
    mask_ramp = wis_test < ramp_weeks
    result = {'all': compute_metrics(y_true, y_pred)}
    if mask_ramp.sum() >= 2:
        result['ramp_up'] = compute_metrics(y_true[mask_ramp], y_pred[mask_ramp])
    if (~mask_ramp).sum() >= 2:
        result['peak'] = compute_metrics(y_true[~mask_ramp], y_pred[~mask_ramp])
    return result
```

Also add `compute_phase_metrics` to the module-level exports list at the top of `cnn_rnn_inv3.py` and `cnn_rnn_inv4.py` import block (Task 6 handles runners; for now just ensure the function exists in `cnn_rnn_yield.py`).

- [ ] **Step 2: Run phase metric tests — they should now pass**

```bash
cd Models && python3 -m pytest tests/test_cnn_rnn_yield.py -v 2>&1 | tail -20
```

Expected: all `test_compute_phase_metrics_*` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(cnn_rnn): add compute_phase_metrics for ramp-up vs peak split"
```

---

## Task 3: 4-tensor DataLoaders in `prepare_data`

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — `prepare_data` function (lines ~938–952)

- [ ] **Step 1: Save `pos_te` (currently discarded) and update `to_ds` and DataLoaders**

Find the sequence-building block (lines ~921–945). Make these changes:

**Change 1** — save `pos_te` (was `_`):
```python
# OLD:
Xs_te_seq, Xt_te_seq, y_te, _      = make_sequences_per_season(
    Xs_te, Xt_te, y_test,  temps_sensor_te, temps_prod_te, seq_len, stride)

# NEW:
Xs_te_seq, Xt_te_seq, y_te, pos_te = make_sequences_per_season(
    Xs_te, Xt_te, y_test,  temps_sensor_te, temps_prod_te, seq_len, stride)
```

**Change 2** — update `to_ds` to accept weight array:
```python
# OLD:
def to_ds(Xs, Xt, y):
    return TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(Xt),
                         torch.FloatTensor(y).unsqueeze(1))

# NEW:
def to_ds(Xs, Xt, y, w):
    return TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(Xt),
                         torch.FloatTensor(y).unsqueeze(1),
                         torch.FloatTensor(w).reshape(-1, 1))
```

**Change 3** — compute `w_train` from `pos_tr` and build DataLoaders with weights:
```python
# OLD:
bs = hp['batch_size']
train_loader = DataLoader(to_ds(Xs_tr_seq, Xt_tr_seq, y_tr), batch_size=bs, shuffle=True, drop_last=True)
val_loader   = DataLoader(to_ds(Xs_va_seq, Xt_va_seq, y_va), batch_size=bs, shuffle=False)
test_loader  = DataLoader(to_ds(Xs_te_seq, Xt_te_seq, y_te), batch_size=bs, shuffle=False)

# NEW:
ramp_weeks  = hp.get('ramp_weeks', 4)
ramp_weight = hp.get('ramp_weight', 1.0)
w_train = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
w_ones_va = np.ones(len(y_va), dtype=np.float32)
w_ones_te = np.ones(len(y_te), dtype=np.float32)

bs = hp['batch_size']
train_loader = DataLoader(to_ds(Xs_tr_seq, Xt_tr_seq, y_tr, w_train), batch_size=bs, shuffle=True, drop_last=True)
val_loader   = DataLoader(to_ds(Xs_va_seq, Xt_va_seq, y_va, w_ones_va), batch_size=bs, shuffle=False)
test_loader  = DataLoader(to_ds(Xs_te_seq, Xt_te_seq, y_te, w_ones_te), batch_size=bs, shuffle=False)
```

**Change 4** — return `wis_test` (= `pos_te`) as 10th element:
```python
# OLD:
return train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys

# NEW:
wis_test = pos_te  # week_in_season of each test sequence target (== pos within season)
return train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test
```

- [ ] **Step 2: Run 4-tensor DataLoader test**

```bash
cd Models && python3 -m pytest tests/test_cnn_rnn_yield.py::test_4tensor_dataset_unpacks_correctly -v
```

Expected: PASS (test is pure torch, no data loading).

- [ ] **Step 3: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(cnn_rnn): 4-tensor DataLoader with per-sample ramp weights in prepare_data"
```

---

## Task 4: Update `train_model`, `evaluate_model`, `evaluate_loader` for 4-tuple

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — three functions

- [ ] **Step 1: Update `train_model` training loop to pass weights to criterion**

Find the training inner loop (line ~1047). Change:

```python
# OLD:
for Xs_batch, Xt_batch, y_batch in train_loader:
    Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)
    optimizer.zero_grad()
    pred = model(Xs_batch, Xt_batch)
    loss = criterion(pred, y_batch)

# NEW:
for Xs_batch, Xt_batch, y_batch, w_batch in train_loader:
    Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)
    optimizer.zero_grad()
    pred = model(Xs_batch, Xt_batch)
    loss = criterion(pred, y_batch, sample_weights=w_batch)
```

- [ ] **Step 2: Update `train_model` validation loop to unpack 4-tuple**

Find the val loop (line ~1070). Change:

```python
# OLD:
for Xs_batch, Xt_batch, y_batch in val_loader:
    Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)
    pred = model(Xs_batch, Xt_batch)
    loss = criterion(pred, y_batch)

# NEW:
for Xs_batch, Xt_batch, y_batch, _ in val_loader:
    Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)
    pred = model(Xs_batch, Xt_batch)
    loss = criterion(pred, y_batch)
```

- [ ] **Step 3: Update `evaluate_model` to unpack 4-tuple**

Find `evaluate_model` inner loop (line ~1148). Change:

```python
# OLD:
for Xs_batch, Xt_batch, y_batch in test_loader:
    pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
    all_preds.append(pred.cpu().numpy())
    all_targets.append(y_batch.numpy())

# NEW:
for Xs_batch, Xt_batch, y_batch, _ in test_loader:
    pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
    all_preds.append(pred.cpu().numpy())
    all_targets.append(y_batch.numpy())
```

- [ ] **Step 4: Update `evaluate_loader` to unpack 4-tuple**

Find `evaluate_loader` inner loop (line ~1184). Change:

```python
# OLD:
for Xs_batch, Xt_batch, y_batch in loader:
    pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
    all_preds.append(pred.cpu().numpy())
    all_targets.append(y_batch.numpy())

# NEW:
for Xs_batch, Xt_batch, y_batch, _ in loader:
    pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
    all_preds.append(pred.cpu().numpy())
    all_targets.append(y_batch.numpy())
```

- [ ] **Step 5: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(cnn_rnn): update train/evaluate loops to unpack 4-tensor batches"
```

---

## Task 5: Update HP yamls

**Files:**
- Modify: `Models/hp_inv3.yaml`
- Modify: `Models/hp_inv4.yaml`

- [ ] **Step 1: Update `hp_inv3.yaml`**

Change `skip_first_weeks` and add `ramp_weeks` / `ramp_weight`:

```yaml
# OLD first lines:
seq_len: 6
skip_first_weeks: 3
alignment: positional

# NEW:
seq_len: 6
skip_first_weeks: 0
ramp_weeks: 4
ramp_weight: 0.5
alignment: positional
```

- [ ] **Step 2: Update `hp_inv4.yaml`**

Same change — find `skip_first_weeks: 3` and replace with:

```yaml
seq_len: 6
skip_first_weeks: 0
ramp_weeks: 4
ramp_weight: 0.5
alignment: positional
```

- [ ] **Step 3: Commit**

```bash
git add Models/hp_inv3.yaml Models/hp_inv4.yaml
git commit -m "feat(hp): set skip_first_weeks=0, add ramp_weeks=4 and ramp_weight=0.5"
```

---

## Task 6: Update `cnn_rnn_inv3.py` and `cnn_rnn_inv4.py` — new return signature + phase metrics

**Files:**
- Modify: `Models/cnn_rnn_inv3.py`
- Modify: `Models/cnn_rnn_inv4.py`

- [ ] **Step 1: Update `cnn_rnn_inv3.py` — import `compute_phase_metrics`**

Find the import block (line ~20):

```python
# OLD:
from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, compute_metrics, print_metrics,
    plot_results_per_greenhouse, compute_cptc_intervals,
)

# NEW:
from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, compute_cptc_intervals,
)
```

- [ ] **Step 2: Update `cnn_rnn_inv3.py` — unpack `wis_test` from `prepare_data`**

Find the `prepare_data` call (line ~44):

```python
# OLD:
train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys = \
    prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                 skip_first_weeks=HP.get('skip_first_weeks', 0))

# NEW:
train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
    prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                 skip_first_weeks=HP.get('skip_first_weeks', 0),
                 ramp_weeks=HP.get('ramp_weeks', 4),
                 ramp_weight=HP.get('ramp_weight', 1.0))
```

Also update `prepare_data` signature to accept these new kwargs (Task 3 added them to `prepare_data` body but not the function signature). In `cnn_rnn_yield.py` line ~787:

```python
# OLD:
def prepare_data(invernadero_id, hp, train_seasons=None, val_season=None, transform='boxcox', skip_first_weeks=0, return_arrays=False):

# NEW:
def prepare_data(invernadero_id, hp, train_seasons=None, val_season=None, transform='boxcox', skip_first_weeks=0, ramp_weeks=None, ramp_weight=None, return_arrays=False):
```

Note: `ramp_weeks` and `ramp_weight` passed as explicit kwargs override the HP dict values. If `None`, the HP dict is used (already implemented in Task 3 via `hp.get(...)`). Leave Task 3's `hp.get` logic untouched — the explicit kwargs are accepted and ignored (HP dict is the source of truth). This keeps the interface clean.

- [ ] **Step 3: Add phase metrics to `cnn_rnn_inv3.py` metrics rows**

Find the `metrics_rows` list (line ~128):

```python
# OLD:
metrics_rows = [
    {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
    {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
    {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
     'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    {'invernadero': INV_ID, 'model': 'cptc_pi', 'pi_coverage': val_cov, 'pi_avg_width': avg_w,
     'R²': None, 'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
]

# NEW:
phase = compute_phase_metrics(
    best_result['y_true'], best_result['y_pred'], wis_test,
    ramp_weeks=HP.get('ramp_weeks', 4))

metrics_rows = [
    {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
    {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
    {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
     'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    {'invernadero': INV_ID, 'model': 'cptc_pi', 'pi_coverage': val_cov, 'pi_avg_width': avg_w,
     'R²': None, 'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
]
if 'ramp_up' in phase:
    metrics_rows.append({'invernadero': INV_ID, 'model': 'best_ramp_up', **phase['ramp_up']})
if 'peak' in phase:
    metrics_rows.append({'invernadero': INV_ID, 'model': 'best_peak',    **phase['peak']})
```

- [ ] **Step 4: Mirror all changes to `cnn_rnn_inv4.py`**

Apply the exact same three changes (import, `prepare_data` unpack, phase metrics) to `cnn_rnn_inv4.py`. The only difference is `INV_ID = 4` and the path `hp_inv4.yaml` — all logic is identical.

- [ ] **Step 5: Commit**

```bash
git add Models/cnn_rnn_inv3.py Models/cnn_rnn_inv4.py Models/cnn_rnn_yield.py
git commit -m "feat(cnn_rnn): wire ramp_weeks/ramp_weight into runners, add phase metrics to CSV"
```

---

## Task 7: Phase-colored plots

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — `plot_results_per_greenhouse` (line ~1255)

- [ ] **Step 1: Add `wis_test` to `best_result` dict in both runners**

In `cnn_rnn_inv3.py`, inside the `if r2 > best_r2:` block (line ~79), add `wis_test` to `best_result`:

```python
# OLD:
best_result = {
    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
    'train_losses': train_losses, 'val_losses': val_losses,
    'seed': seed, 'init_method': init_method,
}

# NEW:
best_result = {
    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
    'train_losses': train_losses, 'val_losses': val_losses,
    'seed': seed, 'init_method': init_method,
    'wis_test': wis_test,
}
```

Mirror the same change in `cnn_rnn_inv4.py`.

- [ ] **Step 2: Update `plot_results_per_greenhouse` to color ramp-up points**

Find `plot_results_per_greenhouse` in `cnn_rnn_yield.py` (line ~1255). Locate the scatter/line plot of actual vs predicted. Replace the single-color scatter with phase-colored points.

Find the axes block that plots actual vs predicted (look for `axes[i, 0]` or `ax.scatter` inside the function). Add coloring logic:

```python
# Find where y_true is plotted against time/index. Typical pattern:
#   ax.plot(y_true, label='Actual')
#   ax.plot(y_pred, label='Predicted')
# Add ramp-up shading after those lines:

wis = res.get('wis_test', None)
ramp_wks = 4  # default; results dict doesn't carry HP, use fixed default
if wis is not None:
    mask_r = wis < ramp_wks
    # shade ramp-up region on the time-series axis
    ramp_indices = np.where(mask_r)[0]
    if len(ramp_indices) > 0:
        ax_ts.axvspan(ramp_indices[0] - 0.5, ramp_indices[-1] + 0.5,
                      alpha=0.12, color='orange', label='Ramp-up phase')
```

Note: read the existing plot function body first to find the exact axis variable name (`ax_ts` may be named differently). Adapt accordingly — do NOT guess variable names; read lines 1255–1305 before editing.

- [ ] **Step 3: Run all tests to verify nothing broke**

```bash
cd Models && python3 -m pytest tests/test_cnn_rnn_yield.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add Models/cnn_rnn_yield.py Models/cnn_rnn_inv3.py Models/cnn_rnn_inv4.py
git commit -m "feat(cnn_rnn): shade ramp-up phase in results plots"
```

---

## Task 8: Smoke test — run inv3 training for 2 epochs

**Files:** None created. Verify wiring is correct without full training run.

- [ ] **Step 1: Patch HP for fast smoke test**

Create a temporary override to run 2 epochs only (do NOT commit this change):

```bash
cd Models
python3 - <<'EOF'
import yaml
from pathlib import Path
p = Path('hp_inv3.yaml')
hp = yaml.safe_load(p.read_text())
hp['epochs'] = 2
hp['patience'] = 2
hp['seeds'] = [42]
hp['init_methods'] = ['default']
Path('hp_inv3_smoke.yaml').write_text(yaml.dump(hp))
print('Smoke HP written.')
EOF
```

- [ ] **Step 2: Run inv3 with smoke HP**

```bash
cd Models
python3 - <<'EOF'
import yaml
from pathlib import Path
import sys
sys.path.insert(0, '.')

hp = yaml.safe_load(Path('hp_inv3_smoke.yaml').read_text())

from cnn_rnn_yield import prepare_data, CNNRNN, train_model, evaluate_model, compute_phase_metrics, DEVICE, RESULTS_DIR
from pathlib import Path
import torch

INV_ID = 3
TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON = 'T16'

result = prepare_data(INV_ID, hp, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                      skip_first_weeks=hp.get('skip_first_weeks', 0))
train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = result

print(f'wis_test[:8]: {wis_test[:8]}')
print(f'wis_test min={wis_test.min()} max={wis_test.max()}')
print(f'ramp-up test samples (wk<4): {(wis_test < 4).sum()}')

# Verify train batches are 4-tuples
batch = next(iter(train_loader))
assert len(batch) == 4, f'Expected 4 tensors, got {len(batch)}'
_, _, _, w = batch
print(f'w_batch sample: {w.flatten()[:8].tolist()}')
assert not (w == 1.0).all(), 'Some weights should be < 1.0 for ramp-up weeks'

model = CNNRNN(n_sensor=n_sensor_pca, n_temporal=n_temporal,
               cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
               cnn_padding=hp['cnn_padding'], num_cnn_blocks=hp['num_cnn_blocks'],
               lstm_hidden=hp['lstm_hidden'], lstm_layers=hp['lstm_layers'],
               dropout=hp['dropout'], fc_hidden=hp['fc_hidden']).to(DEVICE)

tmp = RESULTS_DIR / '_smoke_inv3.pt'
model, tl, vl = train_model(model, train_loader, val_loader, hp, tmp, var_y_train=var_y_train)
print(f'train_losses: {tl}')

y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
print(f'R²={metrics["R²"]:.4f}')

phase = compute_phase_metrics(y_true, y_pred, wis_test, ramp_weeks=hp.get('ramp_weeks', 4))
print('Phase keys:', list(phase.keys()))
if 'ramp_up' in phase:
    print(f'  ramp_up R²={phase["ramp_up"]["R²"]:.4f}')
if 'peak' in phase:
    print(f'  peak    R²={phase["peak"]["R²"]:.4f}')

if tmp.exists(): tmp.unlink()
print('\\nSmoke test PASSED.')
EOF
```

Expected output includes `wis_test[:8]` starting at 0 (ramp-up weeks present), some `w_batch` values < 1.0, `Phase keys: ['all', 'ramp_up', 'peak']`.

- [ ] **Step 3: Clean up smoke files**

```bash
rm -f Models/hp_inv3_smoke.yaml
```

- [ ] **Step 4: Run full test suite**

```bash
cd Models && python3 -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "test(cnn_rnn): verify smoke test passes with 4-tensor DataLoader and phase metrics"
```

---

## Self-Review Notes

- **Spec §1 (skip_first_weeks=0):** Covered in Task 5 (yamls) + Task 3 (`prepare_data`). ✓
- **Spec §1 (w_train from pos_tr):** Task 3, Change 3. ✓
- **Spec §1 (4-tensor TensorDataset):** Task 3, Change 2. ✓
- **Spec §2 (4-tuple unpack in train loop + pass sample_weights):** Task 4, Step 1. ✓
- **Spec §2 (val loop ignores weights):** Task 4, Step 2. ✓
- **Spec §3 (ramp_weight HP grid [0.3, 0.5, 0.7]):** Spec mentions a sweep grid; this plan sets a single value (0.5) as default. The runner scripts do not currently have a grid sweep — they run seeds × init_methods. Adding a 3rd HP dimension (ramp_weight) would triple training time. Recommended: start with `ramp_weight: 0.5`, observe results, then manually test 0.3 and 0.7 by editing the yaml. Adjust plan if a sweep loop is explicitly requested.
- **Spec §3 (phase-split metrics in CSV):** Task 6, Step 3. ✓
- **Spec §3 (plot phase coloring):** Task 7. ✓
- **`evaluate_loader` 4-tuple:** Task 4, Step 4. ✓ (used by CPTC val inference)
- **`prepare_data` signature:** Task 6, Step 2 adds the signature fix. ✓
