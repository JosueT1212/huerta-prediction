# Inv3 production retrain + Inv4 gap fix, HORIZON=5 for both

> **Amendment (2026-07-12, post-implementation):** this spec's `extra_skip` formula
> (`max(0, gap - seq_len - HORIZON)`) turned out to be missing a `skip_first_weeks`
> term — see `_apply_gap_norm_cnn` in `Models/cnn_rnn_yield.py` and CLAUDE.md §7 for
> the corrected formula (`extra_skip = max(0, gap - seq_len - HORIZON + skip_first_weeks + 1)`).
> This means every claim below that inv4 T16 is "1 week short of horizon=5" or a
> "structural limitation" is **incorrect** — that shortfall was an artifact of the
> missing term. With the corrected formula, T16 also reaches exactly horizon=5, same
> as every other season in both greenhouses. The `seq_len` choices (inv3=4, inv4=2)
> and the rest of the design below still stand; only the T16-shortfall claims are
> superseded. See CLAUDE.md §7 for the authoritative current state.

## Problem

- Inv3 needs a final deployable model trained on all available data (T13–T17), using the best hyperparameters already found via grid search.
- Inv4's effective prediction horizon varies wildly across seasons (0–5 weeks) because sensor→production gap swings 6–11 weeks and isn't normalized. This degrades generalization.
- Target horizon is being changed from 4 to 5 weeks for both greenhouses, and both need a fresh grid search under the new horizon before any production retrain.

## Current state (relevant facts)

- `Models/cnn_rnn_yield.py` defines `HORIZON = 4` (line 52), used by `_apply_gap_norm_cnn` (line 603) to compute `extra_skip = gap - seq_len - horizon`, trimming sensor-front rows per season so the effective horizon aligns to `HORIZON`.
- `hp_inv3.yaml` and `hp_inv4.yaml` both currently set `use_gap_norm: false`.
- `cnn_rnn_inv3.py` / `cnn_rnn_inv4.py` already run a full grid search: 4 init methods × 54 seeds = 216 runs, train on T13–T15, validate on T16 (early stopping on val loss), test on T17. Best run selected by val performance, reported metrics computed on T17.
- Current best (pre-change) results: inv3 best = lecun/seed=100, R²=0.6973 (test), early-stopped at epoch 164. Inv4 best = R²=0.6551 (test), from a run where gap-norm happened to be enabled at the time.
- `build_dataset_for_greenhouse` and `prepare_data` already accept arbitrary `train_seasons` (list) and `val_season` (single season name) — `train_seasons=['T13',...,'T17']` pools all 5 seasons into `train_df` with no code change needed. `test_df` stays hardcoded to `'T17'`, which is fine for production: it becomes an unused loader (harmless, no metrics reported from it).
- `train_model` (line 1166) already reads `epochs` and `patience` from the `hp` dict passed in — a production `hp` copy with `epochs=<target>` and `patience=<target>+10` forces the loop to run for exactly `<target>` epochs without ever early-stopping, no code change needed.
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

### 4. Production refit (no code change to `cnn_rnn_yield.py` — new thin scripts only)

New scripts `Models/cnn_rnn_inv3_production.py` / `Models/cnn_rnn_inv4_production.py`, mirroring the structure of the existing `cnn_rnn_inv3.py`/`cnn_rnn_inv4.py` but simplified to a single run:
- `train_seasons = ['T13','T14','T15','T16','T17']` — all 5 seasons pooled into `train_df`. `val_season` stays `'T16'` (already inside `train_seasons`, so it's a redundant/overlapping subset used only to keep the existing `prepare_data`/`train_model` plumbing working — not a real holdout).
- A copy of the greenhouse's `hp_inv*.yaml` dict with `epochs` and `patience` overridden: `epochs = <target_epoch>` (the winning grid-search run's early-stop epoch from step 3), `patience = <target_epoch> + 10` (large enough that early stopping can never trigger before the loop naturally completes `epochs` iterations).
- Single seed/init: the winning `(init_method, seed)` pair from step 3 — no sweep loop.
- Save the resulting weights to a distinct path so they never clobber the eval checkpoints used for reporting metrics: `Models/results/production_cnn_rnn_inv3.pt`, `Models/results/production_cnn_rnn_inv4.pt`.
- Also save the inference pipeline (scalers, PCA, etc., fit on the full pooled training data) via `prepare_data(..., pipeline_path=RESULTS_DIR / 'production_pipeline_inv3.pkl')` (and `_inv4.pkl`) — this is the artifact the live-inference pipeline will actually consume.
- No test-set metrics are computed or reported (the "test" loader is technically built from `T17`, which is now inside the training data too — reporting metrics from it would be pure data leakage, so this step is skipped entirely). Training-loss curve can still be logged for sanity-checking convergence.

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
