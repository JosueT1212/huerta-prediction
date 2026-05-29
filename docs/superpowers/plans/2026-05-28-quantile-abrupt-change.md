# Quantile Regression for Abrupt Yield Change Uncertainty

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-scalar FC head with a 3-quantile output (q=0.1, 0.5, 0.9) and train with pinball loss, giving uncertainty bands around predictions so abrupt yield changes appear as high-spread intervals rather than missed point forecasts.

**Architecture:** Add `n_out` param to `CNNRNN.__init__` (default=1); when `loss_type='quantile'`, set `n_out=3`. Add `PinballLoss` class. Add `evaluate_model_q` that treats q50 as the point forecast for metrics. Add `plot_quantile_bands` for the results figure. Runner scripts branch on `loss_type`.

**Tech Stack:** PyTorch, NumPy, pandas, matplotlib, PyYAML. Tests: pytest. No new dependencies.

**Prerequisite:** The ramp-up plan (`2026-05-29-ramp-up-phase-aware.md`) must be merged first — this plan assumes 4-tensor DataLoaders are in place (train batches yield `Xs, Xt, y, w` 4-tuples).

**Paper:** Wen et al. (2017) "A Multi-Horizon Quantile Recurrent Forecaster" — arXiv:1711.11053

---

## File Map

| File | Change |
|------|--------|
| `Models/cnn_rnn_yield.py` | Add `PinballLoss`; add `n_out` param to `CNNRNN`; add `evaluate_model_q`; add `plot_quantile_bands`; update `train_model` quantile branch |
| `Models/hp_inv3.yaml` | Add `loss_type: quantile`, `quantiles: [0.1, 0.5, 0.9]` |
| `Models/hp_inv4.yaml` | Same as inv3 yaml |
| `Models/cnn_rnn_inv3.py` | Import `evaluate_model_q`, `plot_quantile_bands`; branch on `loss_type` in runner loop |
| `Models/cnn_rnn_inv4.py` | Mirror inv3 changes |
| `Models/tests/test_quantile_yield.py` | Unit tests: pinball loss math, `CNNRNN` output shape with `n_out=3` |

---

## Task 1: Unit tests for `PinballLoss` and `CNNRNN` n_out

**Files:**
- Create: `Models/tests/test_quantile_yield.py`

These must FAIL — `PinballLoss` and `n_out` don't exist yet.

- [ ] **Step 1: Write failing tests**

```python
# Models/tests/test_quantile_yield.py
"""Unit tests for quantile (pinball) loss and multi-output CNNRNN."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest


# ── PinballLoss math ─────────────────────────────────────────────────────────

def test_pinball_loss_median_equals_mae_half():
    """PinballLoss at q=0.5 is 0.5 * MAE."""
    from cnn_rnn_yield import PinballLoss
    pred   = torch.tensor([[3.0], [5.0]])
    target = torch.tensor([[1.0], [8.0]])
    # errors: 2.0 (over), 3.0 (under) — q=0.5 both weigh 0.5
    # pinball = 0.5*(2+3)/2 = 1.25 == 0.5 * MAE(1,3)
    loss_fn = PinballLoss(quantiles=[0.5])
    loss = loss_fn(pred, target)
    expected = 0.5 * (2.0 + 3.0) / 2
    assert abs(loss.item() - expected) < 1e-5, f'Got {loss.item()}, expected {expected}'


def test_pinball_loss_q10_penalises_overestimates():
    """At q=0.1, over-predicting costs 0.1*err; under costs 0.9*err."""
    from cnn_rnn_yield import PinballLoss
    # pred=10, target=5 → over by 5  → loss = 0.1 * 5 = 0.5
    pred   = torch.tensor([[10.0]])
    target = torch.tensor([[ 5.0]])
    loss_fn = PinballLoss(quantiles=[0.1])
    loss = loss_fn(pred, target)
    assert abs(loss.item() - 0.5) < 1e-5, f'Got {loss.item()}'


def test_pinball_loss_q90_penalises_underestimates():
    """At q=0.9, under-predicting costs 0.9*err; over costs 0.1*err."""
    from cnn_rnn_yield import PinballLoss
    # pred=5, target=10 → under by 5 → loss = 0.9 * 5 = 4.5
    pred   = torch.tensor([[5.0]])
    target = torch.tensor([[10.0]])
    loss_fn = PinballLoss(quantiles=[0.9])
    loss = loss_fn(pred, target)
    assert abs(loss.item() - 4.5) < 1e-5, f'Got {loss.item()}'


def test_pinball_loss_three_quantiles_output_scalar():
    """PinballLoss with 3 quantiles on (batch,3) pred returns scalar."""
    from cnn_rnn_yield import PinballLoss
    pred   = torch.randn(8, 3)
    target = torch.randn(8, 1)
    loss_fn = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([]), f'Expected scalar, got {loss.shape}'


def test_pinball_loss_decreasing_quantile_order_raises():
    """Quantiles must be strictly increasing to produce valid intervals."""
    from cnn_rnn_yield import PinballLoss
    with pytest.raises(ValueError):
        PinballLoss(quantiles=[0.9, 0.5, 0.1])


# ── CNNRNN n_out ─────────────────────────────────────────────────────────────

def test_cnnrnn_default_output_shape():
    """Default CNNRNN (n_out=1) outputs (batch, 1)."""
    from cnn_rnn_yield import CNNRNN
    model = CNNRNN(n_sensor=5, n_temporal=3, cnn_filters=16, cnn_kernel_size=3,
                   cnn_padding=1, num_cnn_blocks=2, lstm_hidden=32, lstm_layers=1,
                   dropout=0.0, fc_hidden=16)
    Xs = torch.randn(4, 6, 5)
    Xt = torch.randn(4, 6, 3)
    out = model(Xs, Xt)
    assert out.shape == (4, 1), f'Expected (4,1), got {out.shape}'


def test_cnnrnn_n_out_3_output_shape():
    """CNNRNN with n_out=3 outputs (batch, 3)."""
    from cnn_rnn_yield import CNNRNN
    model = CNNRNN(n_sensor=5, n_temporal=3, cnn_filters=16, cnn_kernel_size=3,
                   cnn_padding=1, num_cnn_blocks=2, lstm_hidden=32, lstm_layers=1,
                   dropout=0.0, fc_hidden=16, n_out=3)
    Xs = torch.randn(4, 6, 5)
    Xt = torch.randn(4, 6, 3)
    out = model(Xs, Xt)
    assert out.shape == (4, 3), f'Expected (4,3), got {out.shape}'


def test_cnnrnn_n_out_default_is_1():
    """CNNRNN without n_out kwarg is backward-compatible (n_out=1)."""
    from cnn_rnn_yield import CNNRNN
    import inspect
    sig = inspect.signature(CNNRNN.__init__)
    default = sig.parameters.get('n_out', None)
    assert default is not None, 'n_out parameter missing from CNNRNN.__init__'
    assert default.default == 1, f'Expected default n_out=1, got {default.default}'
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd Models && python3 -m pytest tests/test_quantile_yield.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'PinballLoss'` for loss tests; `TypeError` for `n_out` tests.

- [ ] **Step 3: Commit failing tests**

```bash
git add Models/tests/test_quantile_yield.py
git commit -m "test(quantile): add failing tests for PinballLoss and CNNRNN n_out"
```

---

## Task 2: Add `PinballLoss` to `cnn_rnn_yield.py`

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — insert after `YieldWMAELoss` (line ~1017)

- [ ] **Step 1: Insert `PinballLoss` class after `YieldWMAELoss.forward`**

Find the line `def train_model(` (line ~1020). Insert immediately before it:

```python
class PinballLoss(nn.Module):
    """Pinball (quantile) loss for multi-quantile output.

    pred:   (batch, n_quantiles) — model outputs one column per quantile
    target: (batch, 1)           — single true value broadcast across quantiles
    quantiles: list of floats in strictly increasing order, e.g. [0.1, 0.5, 0.9]

    Loss = mean over batch and quantiles of pinball(q, err):
        err > 0 (over-pred) → q * err
        err < 0 (under-pred)→ (q - 1) * err
    """
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        quantiles = list(quantiles)
        if quantiles != sorted(quantiles):
            raise ValueError(f'quantiles must be strictly increasing, got {quantiles}')
        super().__init__()
        self.register_buffer('q', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, pred, target):
        # pred:   (batch, n_q)  target: (batch, 1)
        target_exp = target.expand_as(pred)     # (batch, n_q)
        err = target_exp - pred                 # positive = under-pred
        loss = torch.max(self.q * err, (self.q - 1) * err)
        return loss.mean()
```

- [ ] **Step 2: Run pinball loss tests**

```bash
cd Models && python3 -m pytest tests/test_quantile_yield.py -k "pinball" -v
```

Expected: all 5 `test_pinball_loss_*` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(quantile): add PinballLoss class to cnn_rnn_yield"
```

---

## Task 3: Add `n_out` param to `CNNRNN`

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — `CNNRNN.__init__` and `forward` (line ~712)

- [ ] **Step 1: Update `CNNRNN.__init__` signature and FC head**

Find `def __init__(self, n_sensor, n_temporal, ...` (line ~712). Change:

```python
# OLD:
def __init__(self, n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding,
             num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden):
    ...
    self.fc = nn.Sequential(
        nn.Linear(lstm_hidden, fc_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden, 1)
    )

# NEW:
def __init__(self, n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding,
             num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden, n_out=1):
    ...
    self.fc = nn.Sequential(
        nn.Linear(lstm_hidden, fc_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden, n_out)
    )
```

`forward` needs no change — output shape is already `(batch, n_out)` implicitly.

- [ ] **Step 2: Run CNNRNN n_out tests**

```bash
cd Models && python3 -m pytest tests/test_quantile_yield.py -k "cnnrnn" -v
```

Expected: all 3 `test_cnnrnn_*` tests PASS.

- [ ] **Step 3: Run full quantile test suite**

```bash
cd Models && python3 -m pytest tests/test_quantile_yield.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(quantile): add n_out param to CNNRNN (default=1, backward-compatible)"
```

---

## Task 4: Add `evaluate_model_q` and `plot_quantile_bands`

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — insert after `evaluate_loader` (line ~1195) and after `plot_results_per_greenhouse` (line ~1255)

- [ ] **Step 1: Insert `evaluate_model_q` after `evaluate_loader`**

Find the end of `evaluate_loader` (around line ~1200). Insert immediately after:

```python
def evaluate_model_q(model, test_loader, scaler_y, bc_lambda, quantiles=(0.1, 0.5, 0.9)):
    """Run quantile model inference. Returns q50 as point forecast for metrics.

    Returns:
        y_true: 1-D array (original kg scale)
        q_preds: dict with keys matching quantiles, each a 1-D array
        metrics: compute_metrics on (y_true, q50)
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for Xs_batch, Xt_batch, y_batch, _ in test_loader:
            pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_norm = np.concatenate(all_preds)        # (N, n_q)
    y_true_norm = np.concatenate(all_targets).flatten()

    def _inverse(arr_norm):
        arr = scaler_y.inverse_transform(arr_norm.reshape(-1, 1)).flatten()
        if bc_lambda == 'log':
            arr = np.expm1(arr)
        elif bc_lambda is not None:
            from scipy.special import inv_boxcox as _inv_boxcox
            if bc_lambda > 0:
                arr = np.clip(arr, -1.0 / bc_lambda + 1e-6, None)
            arr = _inv_boxcox(arr, bc_lambda) - 1.0
        return arr

    y_true = _inverse(y_true_norm)
    q_preds = {q: _inverse(preds_norm[:, i]) for i, q in enumerate(quantiles)}
    q50_idx = list(quantiles).index(0.5)
    y_pred_median = _inverse(preds_norm[:, q50_idx])
    metrics = compute_metrics(y_true, y_pred_median)
    return y_true, q_preds, metrics
```

- [ ] **Step 2: Insert `plot_quantile_bands` before `plot_results_per_greenhouse`**

Find `def plot_results_per_greenhouse(` (line ~1255). Insert immediately before it:

```python
def plot_quantile_bands(y_true, q_preds, train_losses, val_losses,
                        inv_id, metrics, horizon=6,
                        quantiles=(0.1, 0.5, 0.9), wis=None, ramp_weeks=4):
    """Plot loss curves + actual vs quantile bands for one greenhouse.

    Saves: Models/results/quantile_inv{inv_id}_results.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss curves ---
    ax_loss = axes[0]
    ax_loss.plot(train_losses, label='Train loss')
    ax_loss.plot(val_losses,   label='Val loss')
    ax_loss.set_title(f'Inv{inv_id} — Quantile Training Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.legend()

    # --- Quantile bands ---
    ax_q = axes[1]
    t = np.arange(len(y_true))

    # Shade ramp-up phase
    if wis is not None:
        mask_r = wis < ramp_weeks
        ramp_idx = np.where(mask_r)[0]
        if len(ramp_idx) > 0:
            ax_q.axvspan(ramp_idx[0] - 0.5, ramp_idx[-1] + 0.5,
                         alpha=0.10, color='orange', label='Ramp-up phase')

    q10 = q_preds.get(0.1)
    q50 = q_preds.get(0.5)
    q90 = q_preds.get(0.9)

    if q10 is not None and q90 is not None:
        ax_q.fill_between(t, q10, q90, alpha=0.25, color='steelblue', label='80% PI (q10–q90)')
    if q50 is not None:
        ax_q.plot(t, q50, color='steelblue', linewidth=1.5, label='Median (q50)')
    ax_q.plot(t, y_true, color='black', linewidth=1.2, linestyle='--', label='Actual')

    r2  = metrics.get('R²',       float('nan'))
    rmse = metrics.get('RMSE (kg)', float('nan'))
    ax_q.set_title(f'Inv{inv_id} — Quantile Predictions | R²={r2:.3f} RMSE={rmse:.1f}kg')
    ax_q.set_xlabel(f'Test week (h={horizon}w ahead)')
    ax_q.set_ylabel('kg')
    ax_q.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / f'quantile_inv{inv_id}_results.png'
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')
```

- [ ] **Step 3: Smoke test new functions import cleanly**

```bash
cd Models && python3 -c "
from cnn_rnn_yield import PinballLoss, evaluate_model_q, plot_quantile_bands, CNNRNN
import torch
m = CNNRNN(5,3,16,3,1,2,32,1,0.0,16, n_out=3)
print('CNNRNN n_out=3:', m(torch.randn(2,6,5), torch.randn(2,6,3)).shape)
loss_fn = PinballLoss([0.1, 0.5, 0.9])
print('PinballLoss:', loss_fn(torch.randn(4,3), torch.randn(4,1)).item())
print('OK')
"
```

Expected: `CNNRNN n_out=3: torch.Size([2, 3])`, loss value printed, `OK`.

- [ ] **Step 4: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(quantile): add evaluate_model_q and plot_quantile_bands"
```

---

## Task 5: Update `train_model` to select quantile loss branch

**Files:**
- Modify: `Models/cnn_rnn_yield.py` — `train_model` loss selection (line ~1022)

- [ ] **Step 1: Add quantile branch to criterion selection**

Find the criterion selection block (line ~1022):

```python
# OLD:
loss_type = hp.get('loss_type', 'wmae')
if loss_type == 'huber':
    criterion = nn.HuberLoss(delta=hp.get('huber_delta', 1.0))
elif loss_type == 'nse':
    criterion = NSELoss(corr_weight=hp.get('corr_weight', 0.8), var_y_train=var_y_train)
else:
    criterion = YieldWMAELoss(corr_weight=hp.get('corr_weight', 0.8),
                              power=hp.get('wmae_power', 1),
                              delta=hp.get('huber_delta', 0.7),
                              under_penalty=hp.get('under_penalty', 0.0))

# NEW:
loss_type = hp.get('loss_type', 'wmae')
if loss_type == 'huber':
    criterion = nn.HuberLoss(delta=hp.get('huber_delta', 1.0))
elif loss_type == 'nse':
    criterion = NSELoss(corr_weight=hp.get('corr_weight', 0.8), var_y_train=var_y_train)
elif loss_type == 'quantile':
    criterion = PinballLoss(quantiles=hp.get('quantiles', [0.1, 0.5, 0.9]))
else:
    criterion = YieldWMAELoss(corr_weight=hp.get('corr_weight', 0.8),
                              power=hp.get('wmae_power', 1),
                              delta=hp.get('huber_delta', 0.7),
                              under_penalty=hp.get('under_penalty', 0.0))
```

Note: the `PinballLoss.forward` signature is `(pred, target)` with no `sample_weights` argument. The `train_model` training loop currently calls `criterion(pred, y_batch, sample_weights=w_batch)` (after ramp-up plan). For the quantile branch, sample weights are not passed. Fix this by checking loss type before calling:

Find the training inner loop. Change:

```python
# OLD (after ramp-up plan):
loss = criterion(pred, y_batch, sample_weights=w_batch)

# NEW:
if loss_type == 'quantile':
    loss = criterion(pred, y_batch)
else:
    loss = criterion(pred, y_batch, sample_weights=w_batch)
```

Same fix for validation loop:

```python
# OLD:
for Xs_batch, Xt_batch, y_batch, _ in val_loader:
    ...
    loss = criterion(pred, y_batch)

# NEW: unchanged — val loop already doesn't pass sample_weights; no change needed.
```

- [ ] **Step 2: Verify no regression in existing loss types**

```bash
cd Models && python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Models/cnn_rnn_yield.py
git commit -m "feat(quantile): wire PinballLoss into train_model loss_type=quantile branch"
```

---

## Task 6: Update HP yamls

**Files:**
- Modify: `Models/hp_inv3.yaml`
- Modify: `Models/hp_inv4.yaml`

- [ ] **Step 1: Add quantile HP params to `hp_inv3.yaml`**

Open `Models/hp_inv3.yaml`. Add after the `loss_type` line (or add `loss_type` if absent):

```yaml
loss_type: quantile
quantiles: [0.1, 0.5, 0.9]
```

Do NOT remove `ramp_weeks` / `ramp_weight` added by the ramp-up plan.

- [ ] **Step 2: Mirror to `hp_inv4.yaml`**

Same two lines in `Models/hp_inv4.yaml`.

- [ ] **Step 3: Commit**

```bash
git add Models/hp_inv3.yaml Models/hp_inv4.yaml
git commit -m "feat(hp): set loss_type=quantile with [0.1,0.5,0.9] for inv3 and inv4"
```

---

## Task 7: Update runner scripts (`cnn_rnn_inv3.py`, `cnn_rnn_inv4.py`)

**Files:**
- Modify: `Models/cnn_rnn_inv3.py`
- Modify: `Models/cnn_rnn_inv4.py`

- [ ] **Step 1: Update `cnn_rnn_inv3.py` — import new functions**

Find the import block (line ~20). Add `evaluate_model_q` and `plot_quantile_bands`:

```python
# OLD:
from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, compute_cptc_intervals,
)

# NEW:
from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, evaluate_model_q,
    compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, plot_quantile_bands,
    compute_cptc_intervals,
)
```

- [ ] **Step 2: Update `CNNRNN` instantiation in `cnn_rnn_inv3.py`**

Find where `CNNRNN(...)` is constructed (line ~67). Change:

```python
# OLD:
model = CNNRNN(n_sensor=n_sensor_pca, n_temporal=n_temporal,
               cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
               cnn_padding=hp['cnn_padding'], num_cnn_blocks=hp['num_cnn_blocks'],
               lstm_hidden=hp['lstm_hidden'], lstm_layers=hp['lstm_layers'],
               dropout=hp['dropout'], fc_hidden=hp['fc_hidden']).to(DEVICE)

# NEW:
_n_out = len(hp.get('quantiles', [0.5])) if hp.get('loss_type') == 'quantile' else 1
model = CNNRNN(n_sensor=n_sensor_pca, n_temporal=n_temporal,
               cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
               cnn_padding=hp['cnn_padding'], num_cnn_blocks=hp['num_cnn_blocks'],
               lstm_hidden=hp['lstm_hidden'], lstm_layers=hp['lstm_layers'],
               dropout=hp['dropout'], fc_hidden=hp['fc_hidden'],
               n_out=_n_out).to(DEVICE)
```

- [ ] **Step 3: Update evaluation call in `cnn_rnn_inv3.py`**

Find where `evaluate_model(...)` is called inside the seed loop (line ~85). Wrap with a branch:

```python
# OLD:
y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)

# NEW:
if HP.get('loss_type') == 'quantile':
    _quantiles = HP.get('quantiles', [0.1, 0.5, 0.9])
    y_true, q_preds, metrics = evaluate_model_q(model, test_loader, scaler_y, bc_lambda,
                                                 quantiles=_quantiles)
    y_pred = q_preds.get(0.5, list(q_preds.values())[0])  # use median as point forecast
else:
    y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
    q_preds = None
```

Add `q_preds` to `best_result` dict:

```python
best_result = {
    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
    'train_losses': train_losses, 'val_losses': val_losses,
    'seed': seed, 'init_method': init_method,
    'wis_test': wis_test,
    'q_preds': q_preds,   # None if not quantile mode
}
```

- [ ] **Step 4: Update plot call in `cnn_rnn_inv3.py`**

Find where `plot_results_per_greenhouse(...)` is called (line ~130). Wrap with branch:

```python
# OLD:
plot_results_per_greenhouse([best_result], horizon=HORIZON)

# NEW:
if HP.get('loss_type') == 'quantile' and best_result.get('q_preds') is not None:
    plot_quantile_bands(
        y_true=best_result['y_true'],
        q_preds=best_result['q_preds'],
        train_losses=best_result['train_losses'],
        val_losses=best_result['val_losses'],
        inv_id=INV_ID,
        metrics=best_result['metrics'],
        horizon=HORIZON,
        quantiles=HP.get('quantiles', [0.1, 0.5, 0.9]),
        wis=best_result.get('wis_test'),
        ramp_weeks=HP.get('ramp_weeks', 4),
    )
else:
    plot_results_per_greenhouse([best_result], horizon=HORIZON)
```

- [ ] **Step 5: Mirror all four sub-steps to `cnn_rnn_inv4.py`**

Apply identical changes to `cnn_rnn_inv4.py` — only `INV_ID = 4` differs, all logic is the same.

- [ ] **Step 6: Commit**

```bash
git add Models/cnn_rnn_inv3.py Models/cnn_rnn_inv4.py
git commit -m "feat(quantile): wire n_out, evaluate_model_q, plot_quantile_bands into inv3/inv4 runners"
```

---

## Task 8: Smoke test — 2-epoch quantile run

**Files:** None created. Verify full pipeline without full training.

- [ ] **Step 1: Write smoke HP override**

```bash
cd Models && python3 - <<'EOF'
import yaml
from pathlib import Path
p = Path('hp_inv3.yaml')
hp = yaml.safe_load(p.read_text())
hp['epochs'] = 2
hp['patience'] = 2
hp['seeds'] = [42]
hp['init_methods'] = ['default']
Path('hp_inv3_q_smoke.yaml').write_text(yaml.dump(hp))
print('Smoke HP written:', hp.get('loss_type'), hp.get('quantiles'))
EOF
```

- [ ] **Step 2: Run smoke pipeline**

```bash
cd Models && python3 - <<'EOF'
import yaml, sys, torch
import numpy as np
from pathlib import Path
sys.path.insert(0, '.')

hp = yaml.safe_load(Path('hp_inv3_q_smoke.yaml').read_text())
from cnn_rnn_yield import (
    prepare_data, CNNRNN, PinballLoss, train_model,
    evaluate_model_q, plot_quantile_bands, DEVICE, RESULTS_DIR
)

INV_ID = 3
TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON = 'T16'

(train_loader, val_loader, test_loader,
 scaler_y, bc_lambda, n_sensor_pca, n_temporal,
 var_y_train, test_week_keys, wis_test) = prepare_data(
    INV_ID, hp, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
    skip_first_weeks=hp.get('skip_first_weeks', 0))

quantiles = hp.get('quantiles', [0.1, 0.5, 0.9])
n_out = len(quantiles)
model = CNNRNN(
    n_sensor=n_sensor_pca, n_temporal=n_temporal,
    cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
    cnn_padding=hp['cnn_padding'], num_cnn_blocks=hp['num_cnn_blocks'],
    lstm_hidden=hp['lstm_hidden'], lstm_layers=hp['lstm_layers'],
    dropout=hp['dropout'], fc_hidden=hp['fc_hidden'], n_out=n_out
).to(DEVICE)

print(f'Model n_out={n_out}')

tmp = RESULTS_DIR / '_smoke_q_inv3.pt'
model, tl, vl = train_model(model, train_loader, val_loader, hp, tmp, var_y_train=var_y_train)
print(f'train_losses: {tl}')

y_true, q_preds, metrics = evaluate_model_q(model, test_loader, scaler_y, bc_lambda, quantiles=quantiles)
print(f'q50 R²={metrics["R²"]:.4f}')
print(f'q_preds keys: {list(q_preds.keys())}')

plot_quantile_bands(y_true, q_preds, tl, vl, inv_id=INV_ID, metrics=metrics,
                    quantiles=quantiles, wis=wis_test, ramp_weeks=hp.get('ramp_weeks', 4))

if tmp.exists(): tmp.unlink()
print('\nSmoke test PASSED.')
EOF
```

Expected: model builds with `n_out=3`, trains 2 epochs, prints `q_preds keys: [0.1, 0.5, 0.9]`, saves `quantile_inv3_results.png`.

- [ ] **Step 3: Clean up**

```bash
rm -f Models/hp_inv3_q_smoke.yaml
```

- [ ] **Step 4: Run full test suite**

```bash
cd Models && python3 -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "test(quantile): smoke test passes with n_out=3, PinballLoss, quantile bands plot"
```

---

## Self-Review

| Spec requirement | Covered in |
|---|---|
| PinballLoss math (q10/q50/q90) | Task 2 |
| Strictly increasing quantile validation | Task 2 (raises ValueError) |
| CNNRNN n_out=3, backward-compat | Task 3 |
| loss_type='quantile' in train_model | Task 5 |
| evaluate_model_q → q50 as point forecast | Task 4 |
| plot_quantile_bands with ramp-up shading | Task 4 |
| HP yaml wired | Task 6 |
| Runner branch on loss_type | Task 7 |
| q_preds in best_result dict | Task 7 Step 3 |
| Smoke test full pipeline | Task 8 |
| sample_weights not passed to PinballLoss | Task 5 Step 1 |

**Placeholder scan:** None found.

**Type check:** `evaluate_model_q` returns `(y_true, q_preds, metrics)` — Task 7 Step 3 unpacks `y_true, q_preds, metrics`. Consistent. `q_preds` is `dict[float → np.ndarray]`. `plot_quantile_bands` accesses `q_preds.get(0.5)`. Consistent.
