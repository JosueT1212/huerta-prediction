"""Unit tests for svr_yield shared functions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

# ── flatten_sequences ─────────────────────────────────────────────────────────
def test_flatten_sequences_shape():
    from svr_yield import flatten_sequences
    Xs = np.random.randn(10, 6, 14).astype(np.float32)
    Xt = np.random.randn(10, 6,  2).astype(np.float32)
    X  = flatten_sequences(Xs, Xt)
    assert X.shape == (10, 30), f'Expected (10, 30), got {X.shape}'

def test_flatten_sequences_dtype():
    from svr_yield import flatten_sequences
    Xs = np.random.randn(5, 6, 14).astype(np.float32)
    Xt = np.random.randn(5, 6,  2).astype(np.float32)
    assert flatten_sequences(Xs, Xt).dtype == np.float32

def test_flatten_sequences_last_step():
    """Temporal part must equal last time step of Xt."""
    from svr_yield import flatten_sequences
    Xs = np.zeros((3, 6, 14), dtype=np.float32)
    Xt = np.arange(36, dtype=np.float32).reshape(3, 6, 2)
    X  = flatten_sequences(Xs, Xt)
    np.testing.assert_array_equal(X[:, -2:], Xt[:, -1, :])

# ── _inverse_transform ────────────────────────────────────────────────────────
def test_inverse_transform_log():
    from svr_yield import _inverse_transform
    from sklearn.preprocessing import MinMaxScaler
    y_orig = np.array([100.0, 200.0, 300.0])
    y_log  = np.log1p(y_orig)
    scaler = MinMaxScaler()
    y_sc   = scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
    y_back = _inverse_transform(y_sc, scaler, 'log')
    np.testing.assert_allclose(y_back, y_orig, rtol=1e-4)

def test_inverse_transform_boxcox():
    from svr_yield import _inverse_transform
    from sklearn.preprocessing import MinMaxScaler
    from scipy.stats import boxcox as _bc
    y_orig = np.array([100.0, 200.0, 300.0])
    y_bc, lam = _bc(y_orig + 1.0)
    scaler = MinMaxScaler()
    y_sc   = scaler.fit_transform(y_bc.reshape(-1, 1)).flatten()
    y_back = _inverse_transform(y_sc, scaler, lam)
    np.testing.assert_allclose(y_back, y_orig, rtol=1e-4)

# ── compute_metrics ────────────────────────────────────────────────────────────
def test_compute_metrics_perfect():
    from svr_yield import compute_metrics
    y = np.array([100.0, 200.0, 300.0])
    m = compute_metrics(y, y)
    assert m['R²'] == pytest.approx(1.0)
    assert m['RMSE (kg)'] == pytest.approx(0.0)
    assert m['NSE'] == pytest.approx(1.0)
    assert m['PBIAS (%)'] == pytest.approx(0.0)
    assert m['MAPE (%)'] == pytest.approx(0.0)

# ── grid_search_svr ────────────────────────────────────────────────────────────
def test_grid_search_returns_best_model():
    from svr_yield import grid_search_svr
    from sklearn.preprocessing import MinMaxScaler
    rng = np.random.default_rng(42)
    X_tr = rng.random((20, 30)).astype(np.float32)
    X_va = rng.random((8,  30)).astype(np.float32)
    y_tr = rng.random(20).astype(np.float32)
    y_va = rng.random(8).astype(np.float32)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_tr.reshape(-1, 1))
    model, params, best_r2 = grid_search_svr(X_tr, y_tr, X_va, y_va, scaler_y, 'log')
    assert model is not None
    assert 'C' in params and 'gamma' in params and 'epsilon' in params and 'val_r2' in params
    assert isinstance(best_r2, float)

if __name__ == '__main__':
    import pytest as _pt; _pt.main([__file__, '-v'])
