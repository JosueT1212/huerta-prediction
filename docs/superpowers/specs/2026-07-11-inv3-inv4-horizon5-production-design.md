# Inv3 production retrain + Inv4 gap fix, HORIZON=5 for both

## Problem

- Inv3 needs a final deployable model trained on all available data (T13–T17), using the best hyperparameters already found via grid search.
- Inv4's effective prediction horizon varies wildly across seasons (0–5 weeks) because sensor→production gap swings 6–11 weeks and isn't normalized. This degrades generalization.
- Target horizon is being changed from 4 to 5 weeks for both greenhouses, and both need a fresh grid search under the new horizon before any production retrain.

## Current state (relevant facts)

- `Models/cnn_rnn_yield.py` defines `HORIZON = 4` (line 52), used by `_apply_gap_norm_cnn` (line 603) to compute `extra_skip = gap - seq_len - horizon`, trimming sensor-front rows per season so the effective horizon aligns to `HORIZON`.
- `hp_inv3.yaml` and `hp_inv4.yaml` both currently set `use_gap_norm: false`.
- `cnn_rnn_inv3.py` / `cnn_rnn_inv4.py` already run a full grid search: 4 init methods × 54 seeds = 216 runs, train on T13–T15, validate on T16 (early stopping on val loss), test on T17. Best run selected by val performance, reported metrics computed on T17.
- Current best (pre-change) results: inv3 best = lecun/seed=100, R²=0.6973 (test), early-stopped at epoch 164. Inv4 best = R²=0.6551 (test), from a run where gap-norm happened to be enabled at the time.
- Test season is hardcoded to `'T17'` in two places: `build_dataset_for_greenhouse` (line 587) and `prepare_data` (line 914). There is currently no way to pool all 5 seasons into training — this must change for the production-refit step.

## Design

### 1. Horizon constant

Change `Models/cnn_rnn_yield.py:52`: `HORIZON = 4` → `HORIZON = 5`. Applies to both greenhouses (both go through the same shared pipeline module).

### 2. Enable gap normalization

In both `Models/hp_inv3.yaml` and `Models/hp_inv4.yaml`, set `use_gap_norm: true`. This keeps all 5 seasons per greenhouse (no seasons dropped) — it trims each season's sensor-front rows so the effective horizon (`gap - seq_len`) aligns to the new `HORIZON=5` consistently across seasons, instead of the current per-season swing.

### 3. Grid search re-run (existing scripts, no code changes needed here)

Run `python Models/cnn_rnn_inv3.py` and `python Models/cnn_rnn_inv4.py` unchanged. Each does its existing 216-run search (train T13-15 / val T16 / test T17) under the new HORIZON=5 + gap-norm settings. This produces fresh `inv3.log` / `inv4.log`, `cnn_rnn_inv3_metrics.csv` / `cnn_rnn_inv4_metrics.csv`, and eval checkpoints `best_cnn_rnn_inv3.pt` / `best_cnn_rnn_inv4.pt`. These are validation artifacts, not the production models.

From each log, record the winning run's: init method, seed, and early-stop epoch number (needed for step 4).

### 4. Production refit (code change required)

**Code change:** make the test season configurable instead of hardcoded `'T17'`:
- `build_dataset_for_greenhouse(...)`: accept a `test_season` parameter (default `'T17'` to preserve existing grid-search behavior); when `test_season=None`, skip creating a separate test split — all seasons not in `val_season` go to train.
- `prepare_data(...)`: thread `test_season` through the same way; when `None`, skip building `test_df`/`sensor_te_df`/test `DataLoader` entirely.

**New production mode** (small addition per greenhouse — either a CLI flag on the existing scripts or a new thin script, e.g. `cnn_rnn_inv3_production.py` / `cnn_rnn_inv4_production.py`, reusing all shared logic from `cnn_rnn_yield.py`):
- `train_seasons = ['T13','T14','T15','T16','T17']`, `val_season = None`, `test_season = None` — all 5 seasons pooled into training, nothing held out.
- Hyperparameters fixed to the winning config from step 3 (init method, seed) — no seed/init sweep.
- No early stopping (no val set exists). Train for a **fixed number of epochs** equal to the winning grid-search run's early-stop epoch (e.g. inv3 was 164 in the pre-change run; re-derive from the new HORIZON=5 run's log).
- Save the resulting weights to a distinct path so they never clobber the eval checkpoints used for reporting metrics: `Models/results/production_cnn_rnn_inv3.pt`, `Models/results/production_cnn_rnn_inv4.pt`.
- No test-set metrics are computed or reported for the production model (data leakage — it trained on everything). Training-loss curve can still be logged for sanity-checking convergence.

### 5. Order of operations

For each greenhouse (inv3, inv4), in order:
1. Apply HORIZON=5 constant change (shared, one-time).
2. Set `use_gap_norm: true` in its `hp_inv*.yaml`.
3. Run grid search script, inspect log for winning seed/init/epoch.
4. Run production-refit with that config, all 5 seasons, fixed epochs.
5. Confirm `production_cnn_rnn_inv*.pt` saved.

## Out of scope

- No changes to feature engineering, architecture (CNN/LSTM structure), or the other model families (ARIMAX, XGBoost, SVR, ElasticNet).
- No changes to the live-inference pipeline design (separate spec) — this only produces the trained weights it would eventually consume.
- No re-derivation of `seq_len` (stays fixed at 6 per CLAUDE.md §7 rationale — variable seq_len caused zero-padding issues previously).
