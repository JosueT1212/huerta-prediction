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


# ── spline_yield feature tests ────────────────────────────────────────────────

def test_spline_yield_in_pheno_feature_names():
    """spline_yield must route through pheno path (post-PCA channel, not rotated)."""
    from cnn_rnn_yield import PHENO_FEATURE_NAMES
    assert 'spline_yield' in PHENO_FEATURE_NAMES


def test_spline_yield_split_routes_to_pheno():
    """split_features must put spline_yield in pheno list, not sensor or temporal."""
    from cnn_rnn_yield import split_features
    sensor, pheno, temporal = split_features(['temp_prom_int', 'spline_yield', 'week_in_season'])
    assert 'spline_yield' in pheno
    assert 'spline_yield' not in sensor
    assert 'spline_yield' not in temporal


def test_compute_spline_yield_no_leakage():
    """spline_yield on val/test must be computed from training mean curve only."""
    import numpy as np
    import pandas as pd
    from cnn_rnn_yield import compute_spline_yield_feature

    rng = np.random.default_rng(42)
    n_train = 37
    n_val = 20

    train_df = pd.DataFrame({
        'week_in_season': np.arange(n_train),
        'kg_reales': 1000.0 * np.sin(np.linspace(0, np.pi, n_train)) + rng.normal(0, 50, n_train),
    })
    # Val has different actual yields — spline_yield must be from train curve
    val_df = pd.DataFrame({
        'week_in_season': np.arange(n_val),
        'kg_reales': 2000.0 * np.sin(np.linspace(0, np.pi, n_val)) + rng.normal(0, 50, n_val),
    })
    test_df = pd.DataFrame({
        'week_in_season': np.arange(15),
        'kg_reales': np.zeros(15),
    })

    tr_out, va_out, te_out, *_ = compute_spline_yield_feature(train_df.copy(), val_df.copy(), test_df.copy())

    # spline_yield column must exist in all splits
    assert 'spline_yield' in tr_out.columns
    assert 'spline_yield' in va_out.columns
    assert 'spline_yield' in te_out.columns

    # spline_yield for week 0 on train and val should be same (same spline evaluated at w=0)
    assert abs(tr_out.loc[tr_out['week_in_season'] == 0, 'spline_yield'].iloc[0] -
               va_out.loc[va_out['week_in_season'] == 0, 'spline_yield'].iloc[0]) < 1.0

    # Spline should be non-negative at the peak (week ~n_train//2)
    mid = n_train // 2
    assert tr_out.loc[tr_out['week_in_season'] == mid, 'spline_yield'].iloc[0] > 0


def test_nse_early_stopping_selects_best_r2():
    """
    NSE-based stopping must select the checkpoint with lowest MSE/var(target).
    Demonstrate that the old scheme (MAE - corr_weight * r) can incorrectly prefer
    a 2x-scaled prediction (r=1, R²=-4.5) over a constant mean prediction (r=0, R²=0).
    NSE correctly prefers the better-R² prediction.
    """
    import numpy as np

    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    # pred_a: constant at mean — r=0, R²=0 (useless but not destructive)
    pred_a = np.full(5, targets.mean(), dtype=np.float64)
    # pred_b: 2x scaled — r=1.0, R²=-4.5 (high correlation, terrible accuracy)
    pred_b = (2.0 * targets).astype(np.float64)

    def pearson_r(p, t):
        p_ = p - p.mean(); t_ = t - t.mean()
        return float(np.sum(p_ * t_) / (np.sqrt(np.sum(p_**2) + 1e-8) * np.sqrt(np.sum(t_**2) + 1e-8)))

    def old_score(pred, target, corr_w=3.0):
        mae = float(np.mean(np.abs(pred - target)))
        r   = pearson_r(pred, target)
        return mae - corr_w * r

    def nse_score(pred, target):
        mse   = float(np.mean((pred - target) ** 2))
        var_t = float(np.var(target)) + 1e-8
        nse   = 1.0 - mse / var_t
        return -nse  # lower is better

    r_b = pearson_r(pred_b, targets)
    assert r_b > 0.999, f"pred_b should have r≈1, got {r_b}"

    # Old scheme (corr_weight=3) prefers pred_b — wrong!
    score_a_old = old_score(pred_a, targets, corr_w=3.0)
    score_b_old = old_score(pred_b, targets, corr_w=3.0)
    assert score_b_old < score_a_old, (
        f"Old scheme should prefer scaled pred_b (score={score_b_old:.3f}) "
        f"over mean pred_a (score={score_a_old:.3f})"
    )

    # NSE scheme correctly prefers pred_a (R²=0 > R²=-4.5)
    score_a_nse = nse_score(pred_a, targets)
    score_b_nse = nse_score(pred_b, targets)
    assert score_a_nse < score_b_nse, (
        f"NSE must prefer pred_a (score={score_a_nse:.3f}) "
        f"over scaled pred_b (score={score_b_nse:.3f})"
    )


def test_train_model_uses_nse_stopping(tmp_path):
    """
    train_model with WMAE loss must use NSE-based val_score so training completes
    without error and val_losses contains finite values.
    """
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from cnn_rnn_yield import CNNRNN, DEVICE, train_model

    torch.manual_seed(0)
    n_sensor, n_temporal, seq_len = 4, 2, 6
    batch = 8

    hp = {
        'epochs': 2, 'patience': 10, 'learning_rate': 1e-4,
        'weight_decay': 0.0, 'corr_weight': 0.5, 'loss_type': 'wmae',
        'ramp_weeks': 0, 'ramp_weight': 1.0,
    }

    Xs = torch.randn(batch, seq_len, n_sensor)
    Xt = torch.randn(batch, seq_len, n_temporal)
    y_norm = torch.linspace(0.1, 0.9, batch)
    w = torch.ones(batch)

    dataset = TensorDataset(Xs, Xt, y_norm, w)
    loader = DataLoader(dataset, batch_size=batch)

    model = CNNRNN(n_sensor=n_sensor, n_temporal=n_temporal,
                   cnn_filters=8, cnn_kernel_size=2, cnn_padding=1,
                   num_cnn_blocks=1, lstm_hidden=16, lstm_layers=1,
                   dropout=0.0, fc_hidden=8, n_out=1).to(DEVICE)

    ckpt = tmp_path / 'ckpt.pt'
    model_out, train_l, val_l = train_model(model, loader, loader, hp, ckpt, var_y_train=0.1)
    assert len(train_l) == 2
    assert len(val_l) == 2
    assert all(np.isfinite(v) for v in val_l)


def test_spline_yield_in_sensor_frame_after_prepare_data():
    """
    spline_yield must appear in the pheno path's actual CNN input (sensor_tr_df),
    not just in train_df. This catches the bug where spline_yield was only added
    to train_df but the pheno selector preferred sensor_tr_df.
    We test this by checking that spline_yield reaches the pheno columns list
    via split_features + avail_pheno logic.

    Proxy test: after prepare_data with use_spline_feature=True, the CNN input
    should have MORE channels than without it (spline adds 1 pheno channel).
    """
    import sys, yaml
    sys.path.insert(0, 'Models')
    import warnings
    warnings.filterwarnings('ignore')
    from cnn_rnn_yield import prepare_data

    with open('Models/hp_inv3.yaml') as f:
        HP = yaml.safe_load(f)

    HP_without = dict(HP); HP_without['use_spline_feature'] = False
    HP_with    = dict(HP); HP_with['use_spline_feature']    = True

    r_without = prepare_data(3, HP_without, train_seasons=['T13','T14','T15'], val_season='T16',
                             ramp_weeks=4, ramp_weight=0.5, return_arrays=True)
    r_with    = prepare_data(3, HP_with,    train_seasons=['T13','T14','T15'], val_season='T16',
                             ramp_weeks=4, ramp_weight=0.5, return_arrays=True)

    n_sensor_without = r_without[0].shape[2]  # Xs_tr.shape = (N, seq_len, n_channels)
    n_sensor_with    = r_with[0].shape[2]

    assert n_sensor_with == n_sensor_without + 1, (
        f"spline_yield should add 1 CNN channel: without={n_sensor_without}, with={n_sensor_with}. "
        f"If equal, spline_yield is being dropped from the pheno path."
    )


def test_horizon_is_5():
    """Target prediction horizon changed from 4 to 5 weeks (2026-07-11 retune)."""
    from cnn_rnn_yield import HORIZON
    assert HORIZON == 5


# ── gap-norm regression tests ────────────────────────────────────────────────

def _make_gap_norm_fixture(gaps, n_rows=20):
    """Build a synthetic sensor_df/prod_df pair for _apply_gap_norm_cnn tests.

    gaps: dict of {season: gap_weeks}. Each season gets n_rows sensor rows
    (indexed 0..n_rows-1 via the 'row' column) and a matching 1-row prod_df
    entry carrying 'sensor_gap'.
    """
    import pandas as pd
    sensor_df = pd.concat([
        pd.DataFrame({'temporada': [temp] * n_rows, 'row': range(n_rows)})
        for temp in gaps
    ], ignore_index=True)
    prod_df = pd.DataFrame({
        'temporada': list(gaps.keys()),
        'sensor_gap': list(gaps.values()),
    })
    return sensor_df, prod_df


def test_gap_norm_extra_skip_inv3_seq_len_4_horizon_5():
    """Inv3 gaps (11,10,10,10,11) with seq_len=4, horizon=5 all trim to
    eff_horizon=5 exactly — extra_skip = gap - seq_len - horizon."""
    from cnn_rnn_yield import _apply_gap_norm_cnn

    gaps = {'T13': 11, 'T14': 10, 'T15': 10, 'T16': 10, 'T17': 11}
    seq_len, horizon, n_rows = 4, 5, 20
    sensor_df, prod_df = _make_gap_norm_fixture(gaps, n_rows)

    out = _apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon=horizon)

    expected_extra_skip = {'T13': 2, 'T14': 1, 'T15': 1, 'T16': 1, 'T17': 2}
    for temp, skip in expected_extra_skip.items():
        kept = out[out['temporada'] == temp]
        assert len(kept) == n_rows - skip, (
            f'{temp}: expected {n_rows - skip} rows after trim, got {len(kept)}')
        assert kept['row'].iloc[0] == skip, (
            f'{temp}: expected first kept row index {skip}, got {kept["row"].iloc[0]}')
        eff_horizon = gaps[temp] - seq_len - skip
        assert eff_horizon == horizon, (
            f'{temp}: eff_horizon should be exactly {horizon}, got {eff_horizon}')


def test_gap_norm_extra_skip_inv4_seq_len_2_horizon_5():
    """Inv4 gaps (10,10,8,6,11) with seq_len=2, horizon=5: T13/T14/T15/T17
    trim to eff_horizon=5; T16 (gap=6) is untouched (extra_skip=0) and stays
    at its natural eff_horizon=4 — 1 week short of target, a structural limit."""
    from cnn_rnn_yield import _apply_gap_norm_cnn

    gaps = {'T13': 10, 'T14': 10, 'T15': 8, 'T16': 6, 'T17': 11}
    seq_len, horizon, n_rows = 2, 5, 20
    sensor_df, prod_df = _make_gap_norm_fixture(gaps, n_rows)

    out = _apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon=horizon)

    expected_extra_skip = {'T13': 3, 'T14': 3, 'T15': 1, 'T16': 0, 'T17': 4}
    expected_eff_horizon = {'T13': 5, 'T14': 5, 'T15': 5, 'T16': 4, 'T17': 5}
    for temp, skip in expected_extra_skip.items():
        kept = out[out['temporada'] == temp]
        assert len(kept) == n_rows - skip, (
            f'{temp}: expected {n_rows - skip} rows after trim, got {len(kept)}')
        eff_horizon = gaps[temp] - seq_len - skip
        assert eff_horizon == expected_eff_horizon[temp], (
            f'{temp}: expected eff_horizon={expected_eff_horizon[temp]}, got {eff_horizon}')
