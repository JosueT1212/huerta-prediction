# SVR Improvements — Design Spec

**Date:** 2026-05-28  
**Greenhouses:** Invernadero 3 & 4  
**Goal:** Improve SVR yield prediction by enriching temporal features and expanding the hyperparameter search grid.

**Baseline:** Inv3 R²=0.413 MAPE=17.6%, Inv4 R²=0.109 MAPE=17.2%

---

## 1. Changes Overview

Three files modified, no new files:

| File | Change |
|------|--------|
| `Models/svr_yield.py` | `flatten_sequences` → 58 features; `grid_search_svr` → 245 combos |
| `Models/svr_inv3.py` | Add `StandardScaler` on flat X |
| `Models/svr_inv4.py` | Add `StandardScaler` on flat X |
| `Models/tests/test_svr_yield.py` | Update shape test, add slope test |

---

## 2. Feature Engineering — `flatten_sequences`

Add slope and last-value stats for `Xs_seq` (sensor+pheno). `Xt_seq` (temporal) keeps last-step only.

| Stat | Formula | Features |
|------|---------|---------|
| mean | `Xs_seq.mean(axis=1)` | n_sp |
| std | `Xs_seq.std(axis=1)` | n_sp |
| slope | `(Xs_seq[:, -1, :] - Xs_seq[:, 0, :]) / (seq_len - 1)` | n_sp |
| last | `Xs_seq[:, -1, :]` | n_sp |
| temporal last | `Xt_seq[:, -1, :]` | n_t |
| **Total** | | **4×n_sp + n_t = 58** (with n_sp=14, n_t=2) |

`seq_len - 1 = 5` for the current 6-step sequences. Slope is computed as simple first-to-last difference divided by steps — captures trend direction and magnitude without fitting a regression.

Updated signature docstring: `(N, seq_len, n_sp), (N, seq_len, n_t) → (N, 4*n_sp + n_t)`.

---

## 3. Feature Scaling — `StandardScaler` in runners

After `flatten_sequences`, apply `StandardScaler` fit on `X_train`, transform `X_val` and `X_test`.

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val   = scaler_X.transform(X_val)
X_test  = scaler_X.transform(X_test)
```

Placed in `svr_inv3.py` and `svr_inv4.py` after the three `flatten_sequences` calls, before `grid_search_svr`.

**Why:** mean, std, slope, and last-value stats have different scales even after upstream MinMax scaling. SVR with RBF kernel is sensitive to feature scale; StandardScaler ensures each feature has zero mean and unit variance across the 102 training samples.

The `scaler_X` is fit on `X_train` only — no leakage from val/test.

---

## 4. Hyperparameter Grid — `grid_search_svr`

245 combos (was 120):

| Param | Values | Count |
|-------|--------|-------|
| `C` | 0.1, 1, 10, 100, 1000, 5000, 10000 | 7 |
| `gamma` | `'scale'`, `'auto'`, 0.0001, 0.001, 0.01, 0.1, 1.0 | 7 |
| `epsilon` | 0.005, 0.01, 0.05, 0.1, 0.5 | 5 |

Progress print every 40 combos. All other `grid_search_svr` behavior unchanged: val R² in kg space, y_val_kg precomputed once, RuntimeError guard, returns `(best_model, best_params, best_r2)`.

---

## 5. Tests — `test_svr_yield.py`

Three test changes:

1. **Update `test_flatten_sequences_shape`**: expect `(10, 58)` not `(10, 30)`
2. **Update `test_flatten_sequences_last_step`**: temporal columns now at `X[:, -2:]` (unchanged — n_t=2 still at end)
3. **Add `test_flatten_sequences_slope`**:
   - Constant sequence → slope = 0
   - Linear ramp sequence → slope = known value

---

## 6. Success Criteria

- All tests pass (updated + new slope test)
- Both runner scripts execute without error
- Output files overwritten: `svr_inv{3,4}_metrics.csv`, `_params.json`, `_predictions.npz`, `_results.png`
- R² and MAPE reported; improvement over baseline (R²>0.413 for Inv3, R²>0.109 for Inv4) expected but not required — metrics may improve or reveal further issues
