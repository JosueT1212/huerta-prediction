# SVR Yield Prediction — Design Spec

**Date:** 2026-05-28
**Greenhouses:** Invernadero 3 & 4
**Goal:** Non-linear baseline model (SVR RBF kernel) to complement CNN-RNN and fill the gap above ElasticNet.

---

## 1. File Structure

```
Models/
  svr_yield.py          # shared: flatten_sequences(), grid_search_svr(), evaluate_svr(), save_results()
  svr_inv3.py           # INV_ID=3 runner
  svr_inv4.py           # INV_ID=4 runner
  results/
    svr_inv3_metrics.csv
    svr_inv3_predictions.npz
    svr_inv3_params.json
    svr_inv3_results.png
    svr_inv4_metrics.csv
    svr_inv4_predictions.npz
    svr_inv4_params.json
    svr_inv4_results.png
```

---

## 2. Pipeline

### Reused from `cnn_rnn_yield.py`
- `prepare_data()` — full CNN-RNN preprocessing pipeline:
  - `build_dataset_for_greenhouse()` with positional alignment, skip_first_weeks=3
  - `_apply_gap_norm_cnn()` — trims sensor frame per season so eff_horizon = HORIZON = 4
  - Box-Cox fit on train y
  - MinMaxScaler fit on train y
  - MinMaxScaler + PCA (95% variance, typically 19→6 components) on sensor features
  - MinMaxScaler on pheno + temporal features
  - Same splits: Train T13–T15 | Val T16 | Test T17
- All HP loaded from `hp_inv{id}.yaml` (seq_len=6, skip_first_weeks=3, lag_features=[], etc.)

### New in `svr_yield.py` — `flatten_sequences(Xs_seq, Xt_seq)`

Converts CNN-RNN sequences to flat feature vectors:

| Source | Stats | Features |
|--------|-------|---------|
| Sensor PCA + pheno combined `Xs_seq` (N, 6, n_sp) | mean + std over 6 steps | 2 × n_sp |
| Temporal `Xt_seq` (N, 6, n_t) | last step only | n_t |
| **Total** | | **2 × n_sp + n_t** (typically 30: 2×14 + 2) |

`n_sp` = PCA components + pheno columns (typically 6+8=14).
`n_t` = temporal columns (typically 2: dias_desde_transplante, week_in_season).

Returns:
- `X_train (102×30)`, `X_val (38×30)`, `X_test (36×30)` — numpy float32
- `y_train`, `y_val`, `y_test` — Box-Cox + MinMax transformed, numpy float32
- `scaler_y`, `bc_lambda` — for inverse transform at evaluation

---

## 3. Hyperparameter Tuning

Grid search over val set (T16), metric = R² in original kg space (full inverse transform applied):

```python
Cs       = [0.1, 1.0, 10.0, 100.0, 1000.0]
gammas   = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
epsilons = [0.01, 0.05, 0.1, 0.5]
# 120 combinations, deterministic
```

`sklearn.svm.SVR(kernel='rbf', max_iter=50000)`.

Best `(C, gamma, epsilon)` saved to `results/svr_inv{id}_params.json`.
Final model refit on train+val combined with best params, evaluated on test T17.

### Inverse transform helper (shared with ElasticNet)

```python
def _inverse_transform(y_scaled, scaler_y, bc_lambda):
    y_bc = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    if bc_lambda == 'log':
        return np.expm1(y_bc)
    from scipy.special import inv_boxcox
    return inv_boxcox(np.clip(y_bc, 0, None), bc_lambda) - 1.0
```

Val R² evaluated in kg space. `y_val_kg` precomputed once before the 120-combo loop.

---

## 4. Evaluation & Output

Metrics on T17 test (36 sequences), inverse Box-Cox applied to predictions:
- R², RMSE (kg), NSE, PBIAS (%), MAPE (%)

No ensemble (deterministic). No CPTC intervals.

Saved files per greenhouse:
- `svr_inv{id}_metrics.csv` — same schema as CNN-RNN metrics CSV
- `svr_inv{id}_predictions.npz` — `y_true`, `y_pred`, `week_keys`
- `svr_inv{id}_params.json` — `{"C": X, "gamma": Y, "epsilon": Z, "val_r2": W}`
- `svr_inv{id}_results.png` — actual vs predicted on T17 (single panel)

---

## 5. Runner Scripts

`svr_inv3.py` and `svr_inv4.py` are minimal:
1. Load `hp_inv{id}.yaml`
2. Call `prepare_data(invernadero_id, hp)` from `cnn_rnn_yield.py` → get sequences + scalers
3. Call `flatten_sequences(Xs_tr_seq, Xt_tr_seq)` → `X_train`, etc.
4. Call `grid_search_svr(X_train, y_train, X_val, y_val, scaler_y, bc_lambda)`
5. Print best params + val R²
6. Refit on combined train+val
7. Call `evaluate_svr(model, X_test, y_test, scaler_y, bc_lambda)` → metrics
8. Call `print_metrics(metrics)` + `save_results(inv_id, params, metrics, y_true, y_pred, week_keys)`

Mirror structure of `elasticnet_inv3.py`.

---

## 6. Success Criteria

- Scripts run without error on both greenhouses
- Grid search selects `(C, gamma, epsilon)` logged to JSON
- Metrics printed and saved; R² expected higher than ElasticNet (R²~0) due to RBF non-linearity
- No data leakage: all scalers/PCA/BC fit on train only, gap normalization applied per-season
