# Ramp-up weight sweep — fix poor early-season predictions

Date: 2026-07-16

## Problem

Overall CNN-RNN production metrics look acceptable (R²=0.5286 inv3, 0.5595 inv4,
per §9 thresholds in CLAUDE.md), but the ramp-up phase (`week_in_season <
ramp_weeks=4`) is measured separately in `Models/results/cnn_rnn_inv{3,4}_metrics.csv`
as the `best_ramp_up` row, and is badly broken:

| Invernadero | best_ramp_up R² | best_ramp_up RMSE |
|---|---|---|
| Inv3 | -4.458387 | 7802.86 kg |
| Inv4 | -3.510031 | 8528.68 kg |

Negative R² here means the model does worse than simply predicting the mean
for those weeks. The overall metric hides this because it's dominated by
mid/peak-season weeks, which make up most of the season.

## Root cause

`Models/cnn_rnn_yield.py:1037`:

```python
w_train = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
```

`ramp_weight` is read from `hp_inv3.yaml` / `hp_inv4.yaml` — both currently set
to `0.5`. Ramp-up training samples (`week_in_season < ramp_weeks=4`) get half
the loss weight of normal weeks. The model is directly trained to care less
about getting these weeks right, which plausibly explains the deeply negative
`best_ramp_up` R².

## Approach

`ramp_weight` is a plain scalar read once per run from the yaml into
`prepare_data()`, before the existing seed/init grid loop
(`Models/cnn_rnn_inv3.py:50`, same pattern in `cnn_rnn_inv4.py`). Sweeping it
requires no code changes — only rerunning the existing, unmodified 216-run
grid search (54 seeds × 4 inits, §8 in CLAUDE.md) once per candidate value,
per invernadero.

### Sweep

For each invernadero (3 and 4) independently:

1. Set `ramp_weight` in `hp_inv{n}.yaml` to each candidate value in
   `{0.5, 1.0, 1.5, 2.0}` (0.5 is the current baseline, included for
   comparison).
2. Run `python Models/cnn_rnn_inv{n}.py` unchanged (train=T13-15, val=T16,
   test=T17 — same split as all prior grid searches).
3. Record the `best_ramp_up` and `best` rows from the resulting
   `Models/results/cnn_rnn_inv{n}_metrics.csv` for that candidate.

This produces 4 candidate result sets per invernadero (8 total grid search
runs, 216 seed/init combinations each — same per-run cost as any existing
grid search in this repo).

### Selection rule

Per invernadero, independently:

1. **Hard gate:** `best_ramp_up` R² must be `> 0`. Any candidate that doesn't
   clear this is discarded — a ramp-up fit worse than the mean is not
   acceptable regardless of overall performance.
2. **Among candidates that pass the gate:** pick the one maximizing overall
   `best` R² (equivalently, minimizing `best` RMSE). This is the same metric
   already used to judge production-readiness in §9 of CLAUDE.md — we are not
   introducing a new bar, just refusing to trade away the overall bar for a
   ramp-up fix.
3. If **no** candidate clears the ramp-up gate, stop and report — this would
   mean the loss-reweighting approach alone isn't sufficient, and picking
   `ramp_weeks`/architecture is out of scope for this design (see below).

### Production refit

Once a winning `ramp_weight` is selected per invernadero, follow the existing
§8 step 5 process unchanged: retrain once per invernadero grouping all five
seasons (T13-T17) as training data, no holdout, epoch count fixed from the
winning grid-search run's early-stop epoch. Output goes to
`Models/results/production_cnn_rnn_inv{3,4}.pt` as before — never overwrites
the evaluation checkpoints `best_cnn_rnn_inv{3,4}.pt`.

## Out of scope

- `seq_len`, `HORIZON_WEEKS`, gap-norm formula (`_apply_gap_norm_cnn`) — no
  changes. These were considered and explicitly deferred in favor of this
  narrower, cheaper fix (see conversation history 2026-07-16).
- `CURRENT_SEASON` constant / T18 anything in `scripts/live_inference.py` —
  untouched, unrelated to this fix.
- Changing `ramp_weeks` (the *number of weeks* considered ramp-up, currently
  4) — only the loss *weight* applied to those weeks is being tuned here.

## Testing

- Existing tests in `Models/tests/test_cnn_rnn_yield.py` covering
  `compute_phase_metrics` and `prepare_data`'s sample-weighting logic must
  still pass unchanged (no code touched, only yaml config values).
- After the sweep, manually verify the winning config's `best_ramp_up` and
  `best` rows in `Models/results/cnn_rnn_inv{3,4}_metrics.csv` satisfy the
  selection rule above before doing any production refit.
