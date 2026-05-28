"""
ElasticNet — shared pipeline for Inv3 and Inv4.
Reuses cnn_rnn_yield.py preprocessing; adds flat feature extraction.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox as _boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from cnn_rnn_yield import (
    RESULTS_DIR, HORIZON, SEASON_NAMES,
    build_dataset_for_greenhouse,
    make_sequences_per_season,
    split_features,
    compute_metrics, print_metrics,
)


def flatten_sequences(Xs_seq, Xt_seq):
    """
    Flatten (N, seq_len, n_sensor+pheno) + (N, seq_len, n_temporal) → (N, 30).

    Per sample:
      mean over seq_len of Xs  → n_sensor_pheno values
      std  over seq_len of Xs  → n_sensor_pheno values
      last step of Xt          → n_temporal values
    """
    Xs_mean = Xs_seq.mean(axis=1)   # (N, n_sensor_pheno)
    Xs_std  = Xs_seq.std(axis=1)    # (N, n_sensor_pheno)
    Xt_last = Xt_seq[:, -1, :]      # (N, n_temporal)
    return np.hstack([Xs_mean, Xs_std, Xt_last])  # (N, 2*n_sp + n_t)


def _apply_gap_normalization(sensor_df, prod_df, seq_len, target_horizon=HORIZON):
    """
    Per season: drop first max(0, gap - seq_len - target_horizon) rows from
    BOTH sensor and production frames so eff_horizon = target_horizon for all
    seasons where gap >= seq_len + target_horizon.

    Positional alignment is preserved: sensor[i] still pairs with prod[i]
    after trimming. Short-gap seasons (gap < seq_len + target_horizon) are
    left unchanged.

    Returns (sensor_df_norm, prod_df_norm) with same columns.
    """
    sensor_parts = []
    prod_parts   = []
    for temp in prod_df['temporada'].unique():
        s = sensor_df[sensor_df['temporada'] == temp].copy()
        p = prod_df[prod_df['temporada'] == temp].copy()
        if len(p) == 0 or 'sensor_gap' not in p.columns:
            sensor_parts.append(s)
            prod_parts.append(p)
            continue
        gap        = int(p['sensor_gap'].iloc[0])
        extra_skip = max(0, gap - seq_len - target_horizon)
        sensor_parts.append(s.iloc[extra_skip:].reset_index(drop=True))
        prod_parts.append(p.iloc[extra_skip:].reset_index(drop=True))
    return (pd.concat(sensor_parts, ignore_index=True),
            pd.concat(prod_parts,   ignore_index=True))


def prepare_data_flat(invernadero_id, hp,
                      train_seasons=None, val_season=None):
    """
    Full preprocessing pipeline returning flat numpy arrays for ElasticNet.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, 30)
    y_train, y_val, y_test : np.ndarray, shape (N,)  — Box-Cox + MinMax scaled
    scaler_y               : fitted MinMaxScaler for y
    bc_lambda              : Box-Cox lambda (float)
    test_week_keys         : array of production week keys for test set
    """
    if train_seasons is None:
        train_seasons = ['T13', 'T14', 'T15']
    if val_season is None:
        val_season = 'T16'

    seq_len = hp['seq_len']
    skip    = hp.get('skip_first_weeks', 0)

    # ── 1. Build aligned dataframes ──
    train_df, val_df, test_df, feature_cols, df_sensor_all = \
        build_dataset_for_greenhouse(
            invernadero_id, horizon=0,
            lag_features=hp.get('lag_features', []),
            include_rolling_mean=hp.get('include_rolling_mean', False),
            train_seasons=train_seasons, val_season=val_season,
            alignment=hp.get('alignment', 'positional'),
            sensor_lead_weeks=hp.get('sensor_lead_weeks', None),
        )

    # ── 2. skip_first_weeks ──
    if skip > 0:
        train_df = train_df[train_df['week_in_season'] >= skip].reset_index(drop=True)
        val_df   = val_df[val_df['week_in_season']     >= skip].reset_index(drop=True)
        test_df  = test_df[test_df['week_in_season']   >= skip].reset_index(drop=True)

    sensor_cols, pheno_cols, temporal_cols = split_features(feature_cols)

    # ── 3. Source sensor frames ──
    if df_sensor_all is not None:
        sensor_tr = df_sensor_all[df_sensor_all['temporada'].isin(train_seasons)].reset_index(drop=True)
        sensor_va = df_sensor_all[df_sensor_all['temporada'] == val_season].reset_index(drop=True)
        sensor_te = df_sensor_all[df_sensor_all['temporada'] == 'T17'].reset_index(drop=True)
    else:
        sensor_tr, sensor_va, sensor_te = train_df, val_df, test_df

    avail_sensor   = [c for c in sensor_cols  if c in sensor_tr.columns]
    avail_pheno    = [c for c in pheno_cols   if c in sensor_tr.columns] or \
                     [c for c in pheno_cols   if c in train_df.columns]
    avail_temporal = [c for c in temporal_cols if c in sensor_tr.columns]

    # ── 4. Gap normalization per season ──
    sensor_tr, train_df = _apply_gap_normalization(sensor_tr, train_df, seq_len)
    sensor_va, val_df   = _apply_gap_normalization(sensor_va, val_df,   seq_len)
    sensor_te, test_df  = _apply_gap_normalization(sensor_te, test_df,  seq_len)

    # ── 5. Box-Cox + MinMax on y ──
    y_tr_raw = train_df['target'].values.astype(np.float32)
    y_va_raw = val_df['target'].values.astype(np.float32)
    y_te_raw = test_df['target'].values.astype(np.float32)

    y_tr_bc, bc_lambda = _boxcox(y_tr_raw + 1.0)
    y_va_bc = _boxcox(y_va_raw + 1.0, lmbda=bc_lambda)
    y_te_bc = _boxcox(y_te_raw + 1.0, lmbda=bc_lambda)

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler_y.fit_transform(y_tr_bc.reshape(-1, 1)).flatten().astype(np.float32)
    y_val   = scaler_y.transform(y_va_bc.reshape(-1, 1)).flatten().astype(np.float32)
    y_test  = scaler_y.transform(y_te_bc.reshape(-1, 1)).flatten().astype(np.float32)

    # ── 6. MinMax + PCA on sensor ──
    Xs_tr = sensor_tr[avail_sensor].values.astype(np.float32)
    Xs_va = sensor_va[avail_sensor].values.astype(np.float32)
    Xs_te = sensor_te[avail_sensor].values.astype(np.float32)

    scaler_Xs = MinMaxScaler(feature_range=(-1, 1))
    Xs_tr = scaler_Xs.fit_transform(Xs_tr)
    Xs_va = scaler_Xs.transform(Xs_va)
    Xs_te = scaler_Xs.transform(Xs_te)

    pca = PCA(n_components=0.95)
    Xs_tr = pca.fit_transform(Xs_tr)
    Xs_va = pca.transform(Xs_va)
    Xs_te = pca.transform(Xs_te)

    # ── 7. MinMax on pheno; hstack with sensor PCA ──
    if avail_pheno:
        _src_tr = sensor_tr if all(c in sensor_tr.columns for c in avail_pheno) else train_df
        _src_va = sensor_va if all(c in sensor_va.columns for c in avail_pheno) else val_df
        _src_te = sensor_te if all(c in sensor_te.columns for c in avail_pheno) else test_df
        scaler_Xp = MinMaxScaler(feature_range=(-1, 1))
        Xp_tr = scaler_Xp.fit_transform(_src_tr[avail_pheno].values.astype(np.float32))
        Xp_va = scaler_Xp.transform(_src_va[avail_pheno].values.astype(np.float32))
        Xp_te = scaler_Xp.transform(_src_te[avail_pheno].values.astype(np.float32))
        Xs_tr = np.hstack([Xs_tr, Xp_tr])
        Xs_va = np.hstack([Xs_va, Xp_va])
        Xs_te = np.hstack([Xs_te, Xp_te])

    # ── 8. MinMax on temporal ──
    Xt_tr = sensor_tr[avail_temporal].values.astype(np.float32)
    Xt_va = sensor_va[avail_temporal].values.astype(np.float32)
    Xt_te = sensor_te[avail_temporal].values.astype(np.float32)

    scaler_Xt = MinMaxScaler(feature_range=(-1, 1))
    Xt_tr = scaler_Xt.fit_transform(Xt_tr)
    Xt_va = scaler_Xt.transform(Xt_va)
    Xt_te = scaler_Xt.transform(Xt_te)

    # ── 9. Build sequences (per season) ──
    temps_s_tr = sensor_tr['temporada'].values
    temps_s_va = sensor_va['temporada'].values
    temps_s_te = sensor_te['temporada'].values
    temps_p_tr = train_df['temporada'].values
    temps_p_va = val_df['temporada'].values
    temps_p_te = test_df['temporada'].values

    Xs_tr_seq, Xt_tr_seq, y_tr_seq, _ = make_sequences_per_season(
        Xs_tr, Xt_tr, y_train, temps_s_tr, temps_p_tr, seq_len)
    Xs_va_seq, Xt_va_seq, y_va_seq, _ = make_sequences_per_season(
        Xs_va, Xt_va, y_val,   temps_s_va, temps_p_va, seq_len)
    Xs_te_seq, Xt_te_seq, y_te_seq, _ = make_sequences_per_season(
        Xs_te, Xt_te, y_test,  temps_s_te, temps_p_te, seq_len)

    # ── 10. Flatten sequences → (N, 30) ──
    X_train = flatten_sequences(Xs_tr_seq, Xt_tr_seq)
    X_val   = flatten_sequences(Xs_va_seq, Xt_va_seq)
    X_test  = flatten_sequences(Xs_te_seq, Xt_te_seq)

    print(f'  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')
    print(f'  Features: {X_train.shape[1]} '
          f'(PCA×2={pca.n_components_*2} + pheno×2={len(avail_pheno)*2} + temporal={len(avail_temporal)})')

    _has_wk = 'prod_week_key' in test_df.columns
    test_week_keys = test_df['prod_week_key'].values if _has_wk else np.arange(len(y_te_seq))

    return (X_train, X_val, X_test,
            y_tr_seq, y_va_seq, y_te_seq,
            scaler_y, bc_lambda, test_week_keys)
