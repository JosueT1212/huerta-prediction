# ElasticNet Yield Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ElasticNet yield prediction model for Inv3 and Inv4 with the same preprocessing pipeline as CNN-RNN (Box-Cox, MinMax, PCA) plus per-season gap normalization.

**Architecture:** `elasticnet_yield.py` provides `prepare_data_flat()` which reuses CNN-RNN preprocessing internals and flattens sequences (rolling mean+std) into 30-feature vectors. Two runner scripts (`elasticnet_inv3.py`, `elasticnet_inv4.py`) do grid search over (alpha, l1_ratio) on val set and evaluate on test.

**Tech Stack:** scikit-learn (ElasticNet, MinMaxScaler, PCA), scipy (boxcox), numpy, pandas — all already installed.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `Models/elasticnet_yield.py` | Create | `prepare_data_flat()`, `flatten_sequences()`, `grid_search_elasticnet()`, `evaluate_elasticnet()`, `save_results()` |
| `Models/elasticnet_inv3.py` | Create | Inv3 runner: load HP, call flat pipeline, grid search, print + save results |
| `Models/elasticnet_inv4.py` | Create | Inv4 runner: identical to inv3 but INV_ID=4 |

---

## Task 1: `prepare_data_flat()` — core preprocessing

**Files:**
- Create: `Models/elasticnet_yield.py`

The function reuses CNN-RNN internals from `cnn_rnn_yield.py`. After building sequences, it applies per-season gap normalization (extra sensor rows dropped at start) then flattens each sequence into 30 features.

**Key import reference** (what to pull from `cnn_rnn_yield.py`):
- `RESULTS_DIR, HORIZON, SEASON_NAMES` — constants
- `build_dataset_for_greenhouse` — returns `(train_df, val_df, test_df, feature_cols, df_sensor_all)`
- `make_sequences_per_season` — returns `(Xs_seq, Xt_seq, y_seq, pos_seq)` given `(Xs, Xt, y, temps_sensor, temps_prod, seq_len)`
- `split_features` — returns `(sensor_cols, pheno_cols, temporal_cols)`

- [ ] **Step 1: Create `elasticnet_yield.py` with imports and `flatten_sequences()`**

```python
"""
ElasticNet — shared pipeline for Inv3 and Inv4.
Reuses cnn_rnn_yield.py preprocessing; adds flat feature extraction.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox as _boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from cnn_rnn_yield import (
    RESULTS_DIR, HORIZON, SEASON_NAMES,
    build_dataset_for_greenhouse,
    make_sequences_per_season,
    split_features,
    compute_metrics, print_metrics,
)


def flatten_sequences(Xs_seq, Xt_seq):
    """
    Flatten (N, seq_len, n_sensor+pheno) + (N, seq_len, n_temporal) → (N, 30).

    Per sample:
      mean over seq_len of Xs  → n_sensor_pheno values
      std  over seq_len of Xs  → n_sensor_pheno values
      last step of Xt          → n_temporal values
    """
    Xs_mean = Xs_seq.mean(axis=1)   # (N, n_sensor_pheno)
    Xs_std  = Xs_seq.std(axis=1)    # (N, n_sensor_pheno)
    Xt_last = Xt_seq[:, -1, :]      # (N, n_temporal)
    return np.hstack([Xs_mean, Xs_std, Xt_last])  # (N, 2*n_sp + n_t)
```

- [ ] **Step 2: Add `_apply_gap_normalization()` helper**

```python
def _apply_gap_normalization(sensor_df, prod_df, seq_len, target_horizon=HORIZON):
    """
    Per season: drop first max(0, gap - seq_len - target_horizon) rows from
    BOTH sensor and production frames so eff_horizon = target_horizon for all
    seasons where gap >= seq_len + target_horizon.

    Positional alignment is preserved: sensor[i] still pairs with prod[i]
    after trimming. Short-gap seasons (gap < seq_len + target_horizon) are
    left unchanged.

    Returns (sensor_df_norm, prod_df_norm) with same columns.
    """
    sensor_parts = []
    prod_parts   = []
    for temp in prod_df['temporada'].unique():
        s = sensor_df[sensor_df['temporada'] == temp].copy()
        p = prod_df[prod_df['temporada'] == temp].copy()
        if len(p) == 0 or 'sensor_gap' not in p.columns:
            sensor_parts.append(s)
            prod_parts.append(p)
            continue
        gap        = int(p['sensor_gap'].iloc[0])
        extra_skip = max(0, gap - seq_len - target_horizon)
        sensor_parts.append(s.iloc[extra_skip:].reset_index(drop=True))
        prod_parts.append(p.iloc[extra_skip:].reset_index(drop=True))
    return (pd.concat(sensor_parts, ignore_index=True),
            pd.concat(prod_parts,   ignore_index=True))
```

- [ ] **Step 3: Add `prepare_data_flat()`**

```python
def prepare_data_flat(invernadero_id, hp,
                      train_seasons=None, val_season=None):
    """
    Full preprocessing pipeline returning flat numpy arrays for ElasticNet.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, 30)
    y_train, y_val, y_test : np.ndarray, shape (N,)  — Box-Cox + MinMax scaled
    scaler_y               : fitted MinMaxScaler for y
    bc_lambda              : Box-Cox lambda (float)
    test_week_keys         : array of production week keys for test set
    """
    if train_seasons is None:
        train_seasons = ['T13', 'T14', 'T15']
    if val_season is None:
        val_season = 'T16'

    seq_len = hp['seq_len']
    skip    = hp.get('skip_first_weeks', 0)

    # ── 1. Build aligned dataframes ──
    train_df, val_df, test_df, feature_cols, df_sensor_all = \
        build_dataset_for_greenhouse(
            invernadero_id, horizon=0,
            lag_features=hp.get('lag_features', []),
            include_rolling_mean=hp.get('include_rolling_mean', False),
            train_seasons=train_seasons, val_season=val_season,
            alignment=hp.get('alignment', 'positional'),
            sensor_lead_weeks=hp.get('sensor_lead_weeks', None),
        )

    # ── 2. skip_first_weeks ──
    if skip > 0:
        train_df = train_df[train_df['week_in_season'] >= skip].reset_index(drop=True)
        val_df   = val_df[val_df['week_in_season']     >= skip].reset_index(drop=True)
        test_df  = test_df[test_df['week_in_season']   >= skip].reset_index(drop=True)

    sensor_cols, pheno_cols, temporal_cols = split_features(feature_cols)

    # ── 3. Source sensor frames ──
    if df_sensor_all is not None:
        sensor_tr = df_sensor_all[df_sensor_all['temporada'].isin(train_seasons)].reset_index(drop=True)
        sensor_va = df_sensor_all[df_sensor_all['temporada'] == val_season].reset_index(drop=True)
        sensor_te = df_sensor_all[df_sensor_all['temporada'] == 'T17'].reset_index(drop=True)
    else:
        sensor_tr, sensor_va, sensor_te = train_df, val_df, test_df

    avail_sensor   = [c for c in sensor_cols  if c in sensor_tr.columns]
    avail_pheno    = [c for c in pheno_cols   if c in sensor_tr.columns] or \
                     [c for c in pheno_cols   if c in train_df.columns]
    avail_temporal = [c for c in temporal_cols if c in sensor_tr.columns]

    # ── 4. Gap normalization per season ──
    sensor_tr, train_df = _apply_gap_normalization(sensor_tr, train_df, seq_len)
    sensor_va, val_df   = _apply_gap_normalization(sensor_va, val_df,   seq_len)
    sensor_te, test_df  = _apply_gap_normalization(sensor_te, test_df,  seq_len)

    # ── 5. Box-Cox + MinMax on y ──
    y_tr_raw = train_df['target'].values.astype(np.float32)
    y_va_raw = val_df['target'].values.astype(np.float32)
    y_te_raw = test_df['target'].values.astype(np.float32)

    y_tr_bc, bc_lambda = _boxcox(y_tr_raw + 1.0)
    y_va_bc = _boxcox(y_va_raw + 1.0, lmbda=bc_lambda)
    y_te_bc = _boxcox(y_te_raw + 1.0, lmbda=bc_lambda)

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler_y.fit_transform(y_tr_bc.reshape(-1, 1)).flatten().astype(np.float32)
    y_val   = scaler_y.transform(y_va_bc.reshape(-1, 1)).flatten().astype(np.float32)
    y_test  = scaler_y.transform(y_te_bc.reshape(-1, 1)).flatten().astype(np.float32)

    # ── 6. MinMax + PCA on sensor ──
    Xs_tr = sensor_tr[avail_sensor].values.astype(np.float32)
    Xs_va = sensor_va[avail_sensor].values.astype(np.float32)
    Xs_te = sensor_te[avail_sensor].values.astype(np.float32)

    scaler_Xs = MinMaxScaler(feature_range=(-1, 1))
    Xs_tr = scaler_Xs.fit_transform(Xs_tr)
    Xs_va = scaler_Xs.transform(Xs_va)
    Xs_te = scaler_Xs.transform(Xs_te)

    pca = PCA(n_components=0.95)
    Xs_tr = pca.fit_transform(Xs_tr)
    Xs_va = pca.transform(Xs_va)
    Xs_te = pca.transform(Xs_te)

    # ── 7. MinMax on pheno; hstack with sensor PCA ──
    if avail_pheno:
        _src_tr = sensor_tr if all(c in sensor_tr.columns for c in avail_pheno) else train_df
        _src_va = sensor_va if all(c in sensor_va.columns for c in avail_pheno) else val_df
        _src_te = sensor_te if all(c in sensor_te.columns for c in avail_pheno) else test_df
        scaler_Xp = MinMaxScaler(feature_range=(-1, 1))
        Xp_tr = scaler_Xp.fit_transform(_src_tr[avail_pheno].values.astype(np.float32))
        Xp_va = scaler_Xp.transform(_src_va[avail_pheno].values.astype(np.float32))
        Xp_te = scaler_Xp.transform(_src_te[avail_pheno].values.astype(np.float32))
        Xs_tr = np.hstack([Xs_tr, Xp_tr])
        Xs_va = np.hstack([Xs_va, Xp_va])
        Xs_te = np.hstack([Xs_te, Xp_te])

    # ── 8. MinMax on temporal ──
    Xt_tr = sensor_tr[avail_temporal].values.astype(np.float32)
    Xt_va = sensor_va[avail_temporal].values.astype(np.float32)
    Xt_te = sensor_te[avail_temporal].values.astype(np.float32)

    scaler_Xt = MinMaxScaler(feature_range=(-1, 1))
    Xt_tr = scaler_Xt.fit_transform(Xt_tr)
    Xt_va = scaler_Xt.transform(Xt_va)
    Xt_te = scaler_Xt.transform(Xt_te)

    # ── 9. Build sequences (per season) ──
    temps_s_tr = sensor_tr['temporada'].values
    temps_s_va = sensor_va['temporada'].values
    temps_s_te = sensor_te['temporada'].values
    temps_p_tr = train_df['temporada'].values
    temps_p_va = val_df['temporada'].values
    temps_p_te = test_df['temporada'].values

    Xs_tr_seq, Xt_tr_seq, y_tr_seq, _ = make_sequences_per_season(
        Xs_tr, Xt_tr, y_train, temps_s_tr, temps_p_tr, seq_len)
    Xs_va_seq, Xt_va_seq, y_va_seq, _ = make_sequences_per_season(
        Xs_va, Xt_va, y_val,   temps_s_va, temps_p_va, seq_len)
    Xs_te_seq, Xt_te_seq, y_te_seq, _ = make_sequences_per_season(
        Xs_te, Xt_te, y_test,  temps_s_te, temps_p_te, seq_len)

    # ── 10. Flatten sequences → (N, 30) ──
    X_train = flatten_sequences(Xs_tr_seq, Xt_tr_seq)
    X_val   = flatten_sequences(Xs_va_seq, Xt_va_seq)
    X_test  = flatten_sequences(Xs_te_seq, Xt_te_seq)

    print(f'  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')
    print(f'  Features: {X_train.shape[1]} '
          f'(PCA×2={pca.n_components_*2} + pheno×2={len(avail_pheno)*2} + temporal={len(avail_temporal)})')

    _has_wk = 'prod_week_key' in test_df.columns
    test_week_keys = test_df['prod_week_key'].values if _has_wk else np.arange(len(y_te_seq))

    return (X_train, X_val, X_test,
            y_tr_seq, y_va_seq, y_te_seq,
            scaler_y, bc_lambda, test_week_keys)
```

- [ ] **Step 4: Verify `prepare_data_flat()` runs and shapes are correct**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction/Models
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -c "
import yaml
from pathlib import Path
from elasticnet_yield import prepare_data_flat

with open('hp_inv3.yaml') as f:
    hp = yaml.safe_load(f)

X_tr, X_va, X_te, y_tr, y_va, y_te, sy, lam, wk = \
    prepare_data_flat(3, hp)

assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1], 'Feature dim mismatch'
assert len(X_tr) == len(y_tr), 'Train X/y length mismatch'
assert len(X_va) == len(y_va), 'Val X/y length mismatch'
assert len(X_te) == len(y_te), 'Test X/y length mismatch'
print('PASS — shapes:', X_tr.shape, X_va.shape, X_te.shape)
"
```

Expected output:
```
  X_train: (N, 30), X_val: (M, 30), X_test: (K, 30)
  Features: 30 (PCA×2=12 + pheno×2=16 + temporal=2)
PASS — shapes: (N, 30) (M, 30) (K, 30)
```
(N ≈ 100, M ≈ 36, K ≈ 34 — exact values depend on gap normalization trimming)

- [ ] **Step 5: Commit**

```bash
git add Models/elasticnet_yield.py
git commit -m "feat(elasticnet): add prepare_data_flat with gap normalization"
```

---

## Task 2: Grid search, evaluation, and save functions

**Files:**
- Modify: `Models/elasticnet_yield.py`

- [ ] **Step 1: Add `grid_search_elasticnet()`**

```python
def grid_search_elasticnet(X_train, y_train, X_val, y_val, scaler_y, bc_lambda):
    """
    Grid search over alpha x l1_ratio, scored by val R².
    Returns (best_model, best_params, best_val_r2).
    """
    from scipy.special import inv_boxcox

    alphas    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    best_r2     = -np.inf
    best_params = {}
    best_model  = None

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio,
                max_iter=10000, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_val)
            # Inverse transform: MinMax → Box-Cox → original kg
            y_pred_bc = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            if bc_lambda == 'log':
                y_pred_kg = np.expm1(y_pred_bc)
            else:
                y_pred_kg = inv_boxcox(np.clip(y_pred_bc, 0, None), bc_lambda) - 1.0
            y_val_bc = scaler_y.inverse_transform(
                y_val.reshape(-1, 1)).flatten()
            if bc_lambda == 'log':
                y_val_kg = np.expm1(y_val_bc)
            else:
                y_val_kg = inv_boxcox(np.clip(y_val_bc, 0, None), bc_lambda) - 1.0
            r2 = r2_score(y_val_kg, y_pred_kg)
            if r2 > best_r2:
                best_r2     = r2
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'val_r2': r2}
                best_model  = model

    print(f'  Best val R²={best_r2:.4f}  alpha={best_params["alpha"]}  '
          f'l1_ratio={best_params["l1_ratio"]}')
    return best_model, best_params, best_r2
```

- [ ] **Step 2: Add `evaluate_elasticnet()`**

```python
def evaluate_elasticnet(model, X_test, y_test, scaler_y, bc_lambda):
    """
    Evaluate on test set. Returns (y_true_kg, y_pred_kg, metrics_dict).
    """
    from scipy.special import inv_boxcox

    def inv_transform(y_scaled):
        y_bc = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        if bc_lambda == 'log':
            return np.expm1(y_bc)
        return inv_boxcox(np.clip(y_bc, 0, None), bc_lambda) - 1.0

    y_pred_scaled = model.predict(X_test)
    y_true = inv_transform(y_test)
    y_pred = inv_transform(y_pred_scaled)
    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics
```

- [ ] **Step 3: Add `save_results()`**

```python
def save_results(inv_id, best_params, metrics, y_true, y_pred, test_week_keys):
    """Save params JSON, metrics CSV, predictions NPZ, results plot."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Params JSON
    params_path = RESULTS_DIR / f'elasticnet_inv{inv_id}_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    # Metrics CSV
    rows = [{'invernadero': inv_id, 'model': 'best', **metrics}]
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / f'elasticnet_inv{inv_id}_metrics.csv', index=False)

    # Predictions NPZ
    np.savez_compressed(
        RESULTS_DIR / f'elasticnet_inv{inv_id}_predictions.npz',
        y_true=y_true, y_pred=y_pred, week_keys=test_week_keys,
    )

    # Results plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label='Actual', marker='o', ms=4)
    ax.plot(y_pred, label='ElasticNet', marker='s', ms=4, linestyle='--')
    ax.set_title(f'Invernadero {inv_id} — ElasticNet T17 Predictions')
    ax.set_xlabel('Week (test sequence)')
    ax.set_ylabel('kg')
    ax.legend()
    plt.tight_layout()
    plot_path = RESULTS_DIR / f'elasticnet_inv{inv_id}_results.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f'  Params  → {params_path}')
    print(f'  Metrics → results/elasticnet_inv{inv_id}_metrics.csv')
    print(f'  Preds   → results/elasticnet_inv{inv_id}_predictions.npz')
    print(f'  Plot    → {plot_path}')
```

- [ ] **Step 4: Commit**

```bash
git add Models/elasticnet_yield.py
git commit -m "feat(elasticnet): add grid search, evaluation, save functions"
```

---

## Task 3: Runner scripts

**Files:**
- Create: `Models/elasticnet_inv3.py`
- Create: `Models/elasticnet_inv4.py`

- [ ] **Step 1: Create `elasticnet_inv3.py`**

```python
"""
ElasticNet — Invernadero 3
==========================
Train: T13-T15 | Val: T16 | Test: T17
"""
import warnings
warnings.filterwarnings('ignore')

import yaml
from pathlib import Path

from elasticnet_yield import (
    prepare_data_flat, grid_search_elasticnet,
    evaluate_elasticnet, save_results,
)
from cnn_rnn_yield import HORIZON, print_metrics

INV_ID = 3

_hp_path = Path(__file__).parent / 'hp_inv3.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  ElasticNet Invernadero 3')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print('=' * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, bc_lambda, test_week_keys = \
        prepare_data_flat(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON)

    best_model, best_params, _ = grid_search_elasticnet(
        X_train, y_train, X_val, y_val, scaler_y, bc_lambda)

    y_true, y_pred, metrics = evaluate_elasticnet(
        best_model, X_test, y_test, scaler_y, bc_lambda)

    print_metrics(INV_ID, metrics, horizon=HORIZON)
    save_results(INV_ID, best_params, metrics, y_true, y_pred, test_week_keys)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create `elasticnet_inv4.py`**

```python
"""
ElasticNet — Invernadero 4
==========================
Train: T13-T15 | Val: T16 | Test: T17
"""
import warnings
warnings.filterwarnings('ignore')

import yaml
from pathlib import Path

from elasticnet_yield import (
    prepare_data_flat, grid_search_elasticnet,
    evaluate_elasticnet, save_results,
)
from cnn_rnn_yield import HORIZON, print_metrics

INV_ID = 4

_hp_path = Path(__file__).parent / 'hp_inv4.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  ElasticNet Invernadero 4')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print('=' * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, bc_lambda, test_week_keys = \
        prepare_data_flat(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON)

    best_model, best_params, _ = grid_search_elasticnet(
        X_train, y_train, X_val, y_val, scaler_y, bc_lambda)

    y_true, y_pred, metrics = evaluate_elasticnet(
        best_model, X_test, y_test, scaler_y, bc_lambda)

    print_metrics(INV_ID, metrics, horizon=HORIZON)
    save_results(INV_ID, best_params, metrics, y_true, y_pred, test_week_keys)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run inv3 and verify output**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction/Models
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 elasticnet_inv3.py
```

Expected output (values approximate):
```
============================================================
  ElasticNet Invernadero 3
  Train: ['T13', 'T14', 'T15'] | Val: T16 | Test: T17
============================================================
  X_train: (N, 30), X_val: (M, 30), X_test: (K, 30)
  Best val R²=0.XXXX  alpha=X.X  l1_ratio=X.X
=======================================================
  INVERNADERO 3 - MÉTRICAS (Test: T17, h=4 semanas)
=======================================================
  RMSE (kg)      :  XXXX
  R²             :  0.XXXX
  ...
```

Files created: `results/elasticnet_inv3_params.json`, `results/elasticnet_inv3_metrics.csv`, `results/elasticnet_inv3_predictions.npz`, `results/elasticnet_inv3_results.png`

- [ ] **Step 4: Run inv4 and verify output**

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 elasticnet_inv4.py
```

Same expected structure. Both runs exit 0.

- [ ] **Step 5: Commit**

```bash
git add Models/elasticnet_inv3.py Models/elasticnet_inv4.py \
        Models/elasticnet_yield.py \
        Models/results/elasticnet_inv3_metrics.csv \
        Models/results/elasticnet_inv4_metrics.csv \
        Models/results/elasticnet_inv3_predictions.npz \
        Models/results/elasticnet_inv4_predictions.npz \
        Models/results/elasticnet_inv3_results.png \
        Models/results/elasticnet_inv4_results.png \
        Models/results/elasticnet_inv3_params.json \
        Models/results/elasticnet_inv4_params.json
git commit -m "feat: add ElasticNet yield model for Inv3 and Inv4"
git push
```

---

## Self-Review

**Spec coverage:**
- ✅ File structure: `elasticnet_yield.py`, `elasticnet_inv3.py`, `elasticnet_inv4.py`
- ✅ Reuses CNN-RNN preprocessing: Box-Cox, MinMax, PCA
- ✅ Rolling mean+std feature extraction (30 features)
- ✅ Per-season gap normalization (`extra_skip` per season)
- ✅ skip_first_weeks=3 applied
- ✅ Grid search over 36 (alpha, l1_ratio) combos on val R²
- ✅ Params saved to JSON
- ✅ Metrics CSV same schema as CNN-RNN
- ✅ Predictions NPZ with y_true, y_pred, week_keys
- ✅ Results plot

**Type consistency:**
- `prepare_data_flat` returns `(X_tr, X_va, X_te, y_tr, y_va, y_te, scaler_y, bc_lambda, test_week_keys)` — used consistently in Task 3
- `grid_search_elasticnet` returns `(best_model, best_params, best_val_r2)` — used consistently
- `evaluate_elasticnet` returns `(y_true, y_pred, metrics)` — used consistently
- `save_results(inv_id, best_params, metrics, y_true, y_pred, test_week_keys)` — signature matches all call sites
