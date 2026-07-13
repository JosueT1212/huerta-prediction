# Pure-LSTM architecture comparison (no CNN blocks)

## Problem

The CNN-RNN model's CNN blocks extract local temporal patterns over a sliding window before feeding the LSTM. Since the recent HORIZON=5 retune, `seq_len` shrank to 4 (inv3) and 2 (inv4) to hit the target forecast horizon (see `docs/superpowers/specs/2026-07-11-inv3-inv4-horizon5-production-design.md` and CLAUDE.md §7). With `seq_len=2`, a `cnn_kernel_size=2` convolution has almost no room to extract anything before the LSTM sees the result — the CNN's value at this window size is questionable, especially for inv4. A pure-LSTM architecture (skip the CNN entirely, feed sensor+temporal features straight into the LSTM) may perform equally well or better with fewer parameters, which matters given the small dataset (~150-200 training rows).

## Current state (relevant facts)

- `CNNRNN` (`Models/cnn_rnn_yield.py:767-813`): sensor features go through `num_cnn_blocks` `CNNBlock`s (1D conv + weight norm + ReLU + dropout + residual), producing `cnn_filters` channels; temporal features bypass the CNN and concatenate directly before the LSTM. `lstm.input_size = cnn_filters + n_temporal`.
- Current production winners (both under HORIZON=5, corrected gap-norm, from `docs/superpowers/plans/2026-07-11-inv3-inv4-horizon5-production.md`): inv3 R²=0.5286, RMSE=5661.0 kg (lecun/seed=21, `seq_len=4`); inv4 R²=0.5595, RMSE=6572.9 kg (xavier/seed=41, `seq_len=2`).
- `hp_inv3.yaml`/`hp_inv4.yaml` currently set `num_cnn_blocks: 1`.
- Grid-search scripts (`cnn_rnn_inv3.py`/`cnn_rnn_inv4.py`) already run 216 runs (54 seeds × 4 inits), train T13-15, val T16, test T17, reporting R²/RMSE/MAPE per run and a best-run summary. This pattern is reused unchanged for the comparison.

## Design

### 1. Model change: `num_cnn_blocks=0` means no CNN

In `CNNRNN.__init__` (`Models/cnn_rnn_yield.py`):
- When `num_cnn_blocks=0`, `self.cnn = nn.Sequential()` (empty — already the case, since the block-building loop simply doesn't execute).
- `lstm.input_size` becomes conditional: `cnn_filters + n_temporal` when `num_cnn_blocks > 0`, else `n_sensor + n_temporal`.
- `forward()` is unchanged — `self.cnn(x)` on an empty `nn.Sequential` is already identity, so `x` passes through with its original `n_sensor` channels when there are no CNN blocks.

No new hp key. `num_cnn_blocks=0` is the toggle.

### 2. New hyperparameter configs

`Models/hp_inv3_lstm.yaml`, `Models/hp_inv4_lstm.yaml` — exact copies of the current `hp_inv3.yaml`/`hp_inv4.yaml` (same `seq_len`, `use_gap_norm: true`, `skip_first_weeks`, all training hyperparameters) with `num_cnn_blocks: 0`. The now-unused `cnn_filters`/`cnn_kernel_size`/`cnn_padding` fields stay in the file — harmless, keeps the diff against the CNN version minimal and easy to read.

### 3. New grid-search scripts

`Models/cnn_rnn_inv3_lstm.py`, `Models/cnn_rnn_inv4_lstm.py` — copy the existing `cnn_rnn_inv3.py`/`cnn_rnn_inv4.py` structure exactly (same 216-run seed×init sweep, same train/val/test split, same metrics), changed only to:
- Load `hp_inv3_lstm.yaml`/`hp_inv4_lstm.yaml` instead of the CNN configs.
- Write to separate output paths: `results/inv3_lstm.log` / `inv4_lstm.log`, `cnn_rnn_inv3_lstm_metrics.csv` / `_inv4_lstm_metrics.csv`, `cnn_rnn_inv3_lstm_predictions.npz` / `_inv4_lstm_predictions.npz`, `best_cnn_rnn_inv3_lstm.pt` / `_inv4_lstm.pt`, `cnn_rnn_inv3_lstm_results.png` / `_inv4_lstm_results.png`.

Current production configs, scripts, and models (`hp_inv3.yaml`, `cnn_rnn_inv3.py`, `production_cnn_rnn_inv3.pt`, etc.) are completely untouched by this work.

### 4. Run and compare

Run both new scripts (full 216-run grid search each, sequentially — never two trainings competing for the same compute, per this session's established practice). Compare each greenhouse's best run against its current CNN+LSTM winner:

| | Current (CNN+LSTM) R² | Current RMSE | Pure-LSTM must beat |
|---|---|---|---|
| Inv3 | 0.5286 | 5661.0 kg | both R² > 0.5286 AND RMSE < 5661.0 |
| Inv4 | 0.5595 | 6572.9 kg | both R² > 0.5595 AND RMSE < 6572.9 |

Decided independently per greenhouse — one could win while the other doesn't.

### 5. Outcome

This piece of work stops at reporting which architecture wins per greenhouse. It does **not** automatically trigger a production refit or swap the deployed models — that's a follow-up decision once the comparison result is known.

## Out of scope

- No changes to `HORIZON`, `seq_len`, gap-norm, or any other retuned hyperparameter from the prior session's work — this compares architectures only, holding everything else fixed.
- No changes to other model families (ARIMAX, XGBoost, SVR, ElasticNet).
- No production refit or deployment swap — a separate decision after this comparison's results are in.
