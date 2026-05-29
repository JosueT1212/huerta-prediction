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
