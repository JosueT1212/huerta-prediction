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
- `make_sequences_per_season` (line 626) takes a single fixed `seq_len` int applied uniformly to every season within one greenhouse. `seq_len` stays fixed per greenhouse — CLAUDE.md §7 already established that varying `seq_len` per *season* causes zero-padding problems (inv4 T16 case). What varies per season instead is the *skip*, which `_apply_gap_norm_cnn` already computes automatically via `extra_skip = max(0, gap - seq_len - HORIZON)`.
- `extra_skip` can only trim rows *down* (shrink eff_horizon when gap is larger than `seq_len + HORIZON`) — it can never push eff_horizon up, since there's no sensor data earlier than a season's actual sensor start. So gap-norm is a ceiling, not a normalizer: seasons where `gap < seq_len + HORIZON` keep their natural (smaller) `eff_horizon = gap - seq_len`, no matter what.
- Analysis with `HORIZON=5`: at `seq_len=6` (old fixed value), inv4 T15 (gap=8) and T16 (gap=6) fall structurally short of horizon=5 (natural eff_horizon 2 and 0). Lowering `seq_len` shrinks the trim threshold (`seq_len+HORIZON`) and raises natural eff_horizon for short-gap seasons, closing the gap without touching per-season alignment logic:
  - `seq_len=6`: inv4 T15→2, T16→0.
  - `seq_len=4`: inv4 T15→4, T16→2. Inv3: all 5 seasons trim to exactly 5 (every inv3 gap ≥ 10 > threshold 9).
  - `seq_len=2`: inv4 T15→5 (trimmed, gap=8>threshold=7), T16→4 (untrimmed, gap=6<threshold=7, natural=6-2=4). Inv3 doesn't need this — seq_len=4 already gives full alignment.

## Design

### 1. Horizon constant

Change `Models/cnn_rnn_yield.py:52`: `HORIZON = 4` → `HORIZON = 5`. Applies to both greenhouses (both go through the same shared pipeline module).

### 2. Enable gap normalization + set per-greenhouse seq_len

In `Models/hp_inv3.yaml`: set `use_gap_norm: true`, `seq_len: 4`. All 5 inv3 seasons have gap ≥ 10, exceeding the trim threshold (`seq_len+HORIZON=9`), so every season gets trimmed to exactly `eff_horizon=5` — full alignment, no shortfall.

In `Models/hp_inv4.yaml`: set `use_gap_norm: true`, `seq_len: 2`. Trim threshold becomes `seq_len+HORIZON=7`. T13/T14/T15/T17 (gap ≥ 8) trim to `eff_horizon=5`; T16 (gap=6, below threshold) keeps its natural `eff_horizon = gap - seq_len = 4` — 1 week short of target, the closest achievable without going to `seq_len=1` (which would leave almost no sensor context per window).

Both changes keep all 5 seasons per greenhouse — no seasons dropped.

### 3. Grid search re-run (existing scripts, no code changes needed here)

Run `python Models/cnn_rnn_inv3.py` and `python Models/cnn_rnn_inv4.py` unchanged (they already read `seq_len` and `use_gap_norm` from their respective yaml). Each does its existing 216-run search (train T13-15 / val T16 / test T17) under the new HORIZON=5 + gap-norm + updated `seq_len` settings. This produces fresh `inv3.log` / `inv4.log`, `cnn_rnn_inv3_metrics.csv` / `cnn_rnn_inv4_metrics.csv`, and eval checkpoints `best_cnn_rnn_inv3.pt` / `best_cnn_rnn_inv4.pt`. These are validation artifacts, not the production models.

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
2. Set `use_gap_norm: true` and `seq_len` (4 for inv3, 2 for inv4) in its `hp_inv*.yaml`.
3. Run grid search script, inspect log for winning seed/init/epoch.
4. Run production-refit with that config, all 5 seasons, fixed epochs.
5. Confirm `production_cnn_rnn_inv*.pt` saved.

## Out of scope

- No changes to feature engineering, architecture (CNN/LSTM structure), or the other model families (ARIMAX, XGBoost, SVR, ElasticNet).
- No changes to the live-inference pipeline design (separate spec) — this only produces the trained weights it would eventually consume.
- No per-season `seq_len` within a greenhouse (stays fixed per greenhouse: 4 for inv3, 2 for inv4 — per CLAUDE.md §7 rationale against varying seq_len across seasons). Per-season horizon alignment is handled by `_apply_gap_norm_cnn`'s automatic skip, not by varying `seq_len` within a greenhouse. Inv4 T16 remains 1 week short of the horizon=5 target — an accepted structural limitation given its gap=6 is smaller than any workable `seq_len+HORIZON` threshold.
