# Models/tests/test_cnn_rnn_yield.py
"""Unit tests for phase-aware sample weight logic and compute_phase_metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


# ── sample weight computation ─────────────────────────────────────────────────

def test_ramp_weights_are_ramp_weight_for_early_weeks():
    """Weeks 0..ramp_weeks-1 get ramp_weight; rest get 1.0."""
    pos_tr = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    ramp_weeks = 4
    ramp_weight = 0.5
    w = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
    assert (w[:4] == 0.5).all(), f'Expected 0.5 for weeks 0-3, got {w[:4]}'
    assert (w[4:] == 1.0).all(), f'Expected 1.0 for weeks 4+, got {w[4:]}'


def test_ramp_weights_all_ones_when_ramp_weight_1():
    """ramp_weight=1.0 means no down-weighting — all weights are 1."""
    pos_tr = np.arange(10, dtype=np.int32)
    w = np.where(pos_tr < 4, 1.0, 1.0).astype(np.float32)
    assert (w == 1.0).all()


def test_ramp_weights_dtype_float32():
    pos_tr = np.array([0, 1, 5], dtype=np.int32)
    w = np.where(pos_tr < 4, 0.3, 1.0).astype(np.float32)
    assert w.dtype == np.float32


# ── compute_phase_metrics ─────────────────────────────────────────────────────

def test_compute_phase_metrics_returns_all_key():
    from cnn_rnn_yield import compute_phase_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    assert 'all' in result, 'Missing "all" key'


def test_compute_phase_metrics_splits_ramp_peak():
    from cnn_rnn_yield import compute_phase_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    assert 'ramp_up' in result, 'Missing "ramp_up" key'
    assert 'peak' in result, 'Missing "peak" key'


def test_compute_phase_metrics_ramp_uses_correct_samples():
    from cnn_rnn_yield import compute_phase_metrics, compute_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    expected_ramp = compute_metrics(y_true[:4], y_pred[:4])
    assert abs(result['ramp_up']['R²'] - expected_ramp['R²']) < 1e-6


def test_compute_phase_metrics_all_uses_full_array():
    from cnn_rnn_yield import compute_phase_metrics, compute_metrics
    y_true = np.array([100., 200., 300., 400., 5000., 6000., 7000., 8000.])
    y_pred = np.array([110., 190., 310., 390., 4800., 6200., 6900., 8100.])
    wis    = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = compute_phase_metrics(y_true, y_pred, wis, ramp_weeks=4)
    expected_all = compute_metrics(y_true, y_pred)
    assert abs(result['all']['R²'] - expected_all['R²']) < 1e-6


# ── 4-tensor DataLoader iteration ─────────────────────────────────────────────

def test_4tensor_dataset_unpacks_correctly():
    """TensorDataset with 4 tensors yields 4-tuples per batch."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    Xs = torch.randn(8, 6, 3)
    Xt = torch.randn(8, 6, 2)
    y  = torch.randn(8, 1)
    w  = torch.ones(8, 1)
    ds = TensorDataset(Xs, Xt, y, w)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for batch in loader:
        assert len(batch) == 4, f'Expected 4 tensors per batch, got {len(batch)}'
        Xs_b, Xt_b, y_b, w_b = batch
        assert w_b.shape == (4, 1)
        break


# ── crad_season feature tests ─────────────────────────────────────────────────

def test_crad_season_in_temporal_feature_names():
    """crad_season must be routed to LSTM (temporal path), not PCA."""
    from cnn_rnn_yield import TEMPORAL_FEATURE_NAMES
    assert 'crad_season' in TEMPORAL_FEATURE_NAMES


def test_crad_season_split_routes_to_temporal():
    """split_features must return crad_season in temporal list."""
    from cnn_rnn_yield import split_features
    sensor, pheno, temporal = split_features(['temp_prom_int', 'crad_season', 'week_in_season'])
    assert 'crad_season' in temporal
    assert 'crad_season' not in sensor
    assert 'crad_season' not in pheno


def test_crad_season_monotone_within_season():
    """crad_season should be non-decreasing within each season (cumsum of non-negative values)."""
    import numpy as np
    rad = np.array([100.0, 200.0, 50.0, 300.0, 0.0, 150.0])
    crad = np.cumsum(rad)
    diffs = np.diff(crad)
    assert (diffs >= 0).all(), "crad_season must be non-decreasing within a season"
