import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from datetime import date

os.environ.setdefault("TRANSPLANT_DATE_INV3", "2026-01-05")


def _make_pipeline(n_sensor_cols=3, seq_len=6):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    n_components = 2
    scaler_Xs = MinMaxScaler(feature_range=(-1, 1))
    scaler_Xs.fit(np.random.rand(20, n_sensor_cols))
    pca = PCA(n_components=n_components)
    pca.fit(scaler_Xs.transform(np.random.rand(20, n_sensor_cols)))
    scaler_Xt = MinMaxScaler(feature_range=(-1, 1))
    scaler_Xt.fit(np.random.rand(20, 2))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(np.array([[0.0], [500.0]]))
    return {
        'scaler_Xs': scaler_Xs, 'pca': pca,
        'scaler_Xp': None, 'scaler_Xt': scaler_Xt,
        'scaler_y': scaler_y, 'bc_lambda': None,
        'sensor_cols': ['temp_prom_int', 'hr_prom_int', 'co2_ppm'],
        'pheno_cols': [],
        'temporal_cols': ['dias_desde_transplante', 'week_in_season'],
        'pheno_means': {}, 'seq_len': seq_len, 'transform': None,
    }


def _wide_rows(n_days=60):
    from datetime import timedelta
    rows = []
    base = date(2026, 4, 1)
    for d in range(n_days):
        day = base.fromordinal(base.toordinal() + d)
        rows.append({
            'fecha': day.isoformat(),
            'temp_prom_int': 24.0 + d * 0.01,
            'hr_prom_int': 65.0,
            'co2_ppm': 410.0,
        })
    return rows


def _pheno_obs(n_weeks=8):
    from datetime import timedelta
    rows = []
    base = date(2026, 4, 7)
    for w in range(n_weeks):
        week_date = (base + timedelta(weeks=w)).isoformat()
        for zona in range(1, 5):
            for planta in range(1, 5):
                rows.append({
                    'week_date': week_date, 'zona': zona, 'planta': planta,
                    'racimos_puestos': 10.0 + w,
                    'flores_racimo_abiertas': 8.0,
                    'racimos_en_planta': 5.0,
                    'cantidad_tomates': 30.0,
                    'racimo_en_cosecha': 2.0,
                    'tomates_maduros': 10.0,
                    'diametro_fruto_cm': 5.5,
                    'crecimiento_planta_cm': 3.2,
                })
    return rows


def test_aggregate_wide_to_weekly():
    from backend.live_features import aggregate_wide_to_weekly
    rows = _wide_rows(60)
    df = aggregate_wide_to_weekly(rows, ['temp_prom_int', 'hr_prom_int', 'co2_ppm'])
    assert len(df) > 0
    assert 'temp_prom_int' in df.columns
    assert 'year_week' in df.columns


def test_aggregate_pheno_to_weekly():
    from backend.live_features import aggregate_pheno_to_weekly
    rows = _pheno_obs(8)
    df = aggregate_pheno_to_weekly(rows)
    assert len(df) == 8
    assert 'racimos_puestos' in df.columns
    # mean of 16 identical values = same value
    assert abs(df['racimos_puestos'].iloc[0] - 10.0) < 0.01


def test_build_input_tensor_shape():
    from backend.live_features import build_input_tensor
    pipeline = _make_pipeline(n_sensor_cols=3, seq_len=6)
    wide_rows = _wide_rows(60)
    pheno_rows = _pheno_obs(8)
    with patch('backend.live_features.joblib.load', return_value=pipeline):
        xs, xt = build_input_tensor(
            3, wide_rows, pheno_rows,
            transplant_date=date(2026, 1, 5),
            pipeline_path=Path("Models/results/pipeline_inv3.pkl"),
        )
    assert xs.shape == (1, 6, 2)   # (batch, seq_len, n_pca_components)
    assert xt.shape == (1, 6, 2)   # (batch, seq_len, n_temporal)


def test_build_input_tensor_requires_explicit_transplant_date():
    from backend.live_features import build_input_tensor
    pipeline = _make_pipeline(n_sensor_cols=3, seq_len=6)
    wide_rows = _wide_rows(60)
    with patch('backend.live_features.joblib.load', return_value=pipeline):
        with pytest.raises(TypeError):
            build_input_tensor(4, wide_rows, [], pipeline_path="dummy.pkl")
