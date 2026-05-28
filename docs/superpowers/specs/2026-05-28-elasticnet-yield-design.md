# ElasticNet Yield Prediction — Design Spec

**Date:** 2026-05-28  
**Greenhouses:** Invernadero 3 & 4  
**Goal:** Simpler baseline model to complement CNN-RNN; reduce overfitting on small dataset (~100 train sequences).

---

## 1. File Structure

```
Models/
  elasticnet_yield.py       # shared: prepare_data_flat(), metrics, plotting
  elasticnet_inv3.py        # inv3 runner
  elasticnet_inv4.py        # inv4 runner
  results/
    elasticnet_inv3_metrics.csv
    elasticnet_inv3_predictions.npz
    elasticnet_inv3_params.json
    elasticnet_inv4_metrics.csv
    elasticnet_inv4_predictions.npz
    elasticnet_inv4_params.json
```

---

## 2. Pipeline

### Reused from `cnn_rnn_yield.py`
- All data loaders: `load_internal_variables_all_seasons()`, `load_external_variables_all_seasons()`, `load_phenology_features()`, `load_riego_features()`, `load_transplant_dates()`
- `build_dataset_for_greenhouse()` — produces aligned sensor/production frames per season
- `split_features()` — splits features into sensor / pheno / temporal
- Box-Cox fit on train y
- MinMaxScaler fit on train sensor features
- PCA fit on train sensor features (19 → 6 components, 95% variance)
- Same splits: Train T13–T15 | Val T16 | Test T17
- `skip_first_weeks=3` per season

### New in `elasticnet_yield.py` — `prepare_data_flat()`

After building sequences (`seq_len=6`), apply **per-season gap normalization**:

```
target_eff_horizon = 4   # HORIZON constant
extra_skip = max(0, gap - seq_len - target_eff_horizon)
```

Skip `extra_skip` rows from the start of the sensor frame for that season before extracting rolling features. Ensures all seasons with gap ≥ 10 predict at exactly 4-week horizon.

| Season | gap | extra_skip | eff_horizon |
|--------|-----|-----------|-------------|
| Inv3 T13 | 11 | 1 | 4 |
| Inv3 T14–T17 | 10 | 0 | 4 |
| Inv4 T13 | 11 | 1 | 4 |
| Inv4 T14–T15 | 10 | 0 | 4 |
| Inv4 T16 | 6 | 0 | 0 (unavoidable) |
| Inv4 T17 | 11 | 1 | 4 |

### Feature Extraction (per sample)

Each sequence has shape `(6, 16)` — 6 PCA sensor + 8 pheno + 2 temporal per time step.

Flatten into **30 features**:

| Source | Stats | Features |
|--------|-------|---------|
| Sensor PCA (6 components) | mean + std over 6 steps | 12 |
| Pheno (8 features) | mean + std over 6 steps | 16 |
| Temporal (2 features) | last step value only | 2 |
| **Total** | | **30** |

Returns:
- `X_train (102×30)`, `X_val (38×30)`, `X_test (36×30)` — numpy float32
- `y_train`, `y_val`, `y_test` — Box-Cox transformed, numpy float32
- `scaler_y`, `bc_lambda` — for inverse transform at evaluation

---

## 3. Hyperparameter Tuning

Grid search over val set (T16), metric = R²:

```python
alphas    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# 36 combinations, deterministic
```

`l1_ratio=1.0` = pure Lasso (max sparsity).  
Best `(alpha, l1_ratio)` saved to `results/elasticnet_inv{id}_params.json`.  
Final model refit on train+val combined with best params, evaluated on test.

---

## 4. Evaluation & Output

Metrics on T17 test (36 sequences), inverse Box-Cox applied to predictions:
- R², RMSE (kg), NSE, PBIAS (%), MAPE (%)

No ensemble (deterministic). No CPTC intervals.

Saved files per greenhouse:
- `elasticnet_inv{id}_metrics.csv` — same schema as CNN-RNN metrics CSV
- `elasticnet_inv{id}_predictions.npz` — `y_true`, `y_pred`, `week_keys`
- `elasticnet_inv{id}_params.json` — `{"alpha": X, "l1_ratio": Y, "val_r2": Z}`

Results plot: actual vs predicted on T17 (single panel, saved to `results/`).

---

## 5. Runner Scripts

`elasticnet_inv3.py` and `elasticnet_inv4.py` are minimal — load HP from `hp_inv3.yaml` / `hp_inv4.yaml` (reuse existing), call `prepare_data_flat()`, run grid search, evaluate, save. Mirror structure of `cnn_rnn_inv3.py`.

---

## 6. Success Criteria

- Scripts run without error on both greenhouses
- Grid search selects `(alpha, l1_ratio)` logged to JSON
- Metrics printed and saved; R² and MAPE comparable to or better than CNN-RNN ensemble
- No data leakage: scaler/PCA/BC fit on train only, gap normalization applied per-season
