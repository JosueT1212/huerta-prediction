# Design: Phase-Aware Ramp-Up Training for CNN-RNN Yield Model

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** `Models/cnn_rnn_yield.py`, `Models/cnn_rnn_inv3.py`, `Models/cnn_rnn_inv4.py`, `Models/hp_inv3.yaml`, `Models/hp_inv4.yaml`

---

## Problem

First 3–4 weeks of each season have near-zero production (ramp-up phase). Current config sets `skip_first_weeks=3`, excluding those weeks from train/val/test entirely. Model never sees ramp-up patterns → cannot predict them at deployment.

**Goal:** Full-season predictions including ramp-up weeks. Overall R² (all weeks) is the success metric. Some peak-accuracy trade-off is acceptable.

---

## Approach: Phase-Aware Sample Weighting

Include all season weeks in training. Down-weight ramp-up samples in the loss so they don't dominate gradient updates but are still learned.

Single model. No architectural change. Uses existing `YieldWMAELoss.sample_weights` hook.

---

## Section 1 — Data Pipeline

### `prepare_data` changes

New parameters:
- `ramp_weeks: int = 4` — number of weeks considered ramp-up phase (week_in_season < ramp_weeks)
- `ramp_weight: float = 1.0` — loss weight for ramp-up samples (< 1.0 down-weights them)

HP yaml changes:
```yaml
skip_first_weeks: 0   # was 3 — include all weeks
ramp_weeks: 4         # new
ramp_weight: 0.5      # new — sweep [0.3, 0.5, 0.7]
```

### `make_sequences_per_season` — no change needed

`week_in_season` is set as `np.arange(n)` per season, so it equals the row index `j` within the season. `make_sequences_per_season` already returns `seqs_pos` (= `j` per sequence). Therefore `wis_train = pos_tr` directly — no signature change required.

### Sample weight computation

After building train sequences, using existing `pos_tr` return:
```python
# pos_tr == week_in_season of each sequence's target row
w_train = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
```

### TensorDataset shape

All three datasets (train, val, test) become 4-tensors: `(Xs, Xt, y, w)`.  
Val and test get `w = ones` — unused in loss, uniform interface.

---

## Section 2 — Training Loop

Unpack 4-tuple in training loop:
```python
for Xs_batch, Xt_batch, y_batch, w_batch in train_loader:
    ...
    loss = criterion(pred, y_batch, sample_weights=w_batch)
```

Validation loop continues unpacking 4-tuple but calls loss without weights:
```python
for Xs_batch, Xt_batch, y_batch, _ in val_loader:
    ...
    loss = criterion(pred, y_batch)
```

`YieldWMAELoss.forward` already handles `sample_weights=None` — no change needed there.

`ramp_weight` only takes effect when `loss_type == 'wmae'` (default). NSE and Huber paths receive `w=ones` → no behavioral change for those loss types.

---

## Section 3 — HP Grid & Evaluation

### New HP grid dimension

Add `ramp_weight` sweep to existing runner grid:
```python
ramp_weights = hp.get('ramp_weight_grid', [0.5])  # override in yaml to sweep
```

Sweep values: `[0.3, 0.5, 0.7]`. Combined with existing seed/init_method grid.

### Evaluation split

After test inference, split predictions by `week_in_season`:
- **ramp_up**: weeks 0 to `ramp_weeks - 1`
- **peak**: weeks `ramp_weeks`+
- **all**: full season

`*_metrics.csv` gains three rows per model variant: `ramp_up`, `peak`, `all`.

### Plots

Scatter/line plots color ramp-up points differently (e.g., orange vs blue) to visually distinguish phase performance.

---

## Files Changed

| File | Change |
|------|--------|
| `Models/cnn_rnn_yield.py` | `prepare_data` builds `w_train` from existing `pos_tr`; 4-tensor TensorDataset; training loop unpacks 4-tuple; phase-split evaluation |
| `Models/hp_inv3.yaml` | `skip_first_weeks: 0`, add `ramp_weeks: 4`, `ramp_weight: 0.5` |
| `Models/hp_inv4.yaml` | Same as inv3 yaml |
| `Models/cnn_rnn_inv3.py` | Pass `ramp_weeks` / `ramp_weight` to `prepare_data` |
| `Models/cnn_rnn_inv4.py` | Same as inv3 runner |

---

## Success Criteria

- Ramp-up weeks (0–3) have reasonable RMSE — not catastrophically off
- Overall R² (all weeks) ≥ current R² on peak-only subset (R²=0.679 Inv3, 0.545 Inv4)
- Phase-split metrics visible in output CSV
