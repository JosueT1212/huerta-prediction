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
    Flatten (N, seq_len, n_sensor+pheno) + (N, seq_len, n_temporal) → (N, F).
    F = 2*n_sensor_pheno + n_temporal (typically 30 with default PCA and 8 pheno cols).

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
    the SENSOR frame only, advancing its start so eff_horizon = target_horizon
    for seasons where gap >= seq_len + target_horizon.

    prod_df is returned unchanged — the gap is encoded in the length difference
    between sensor_df and prod_df, so only sensor needs trimming.

    Returns (sensor_df_norm, prod_df_unchanged).
    """
    sensor_parts = []
    for temp in prod_df['temporada'].unique():
        s = sensor_df[sensor_df['temporada'] == temp].copy()
        p = prod_df[prod_df['temporada'] == temp]
        if len(p) == 0 or 'sensor_gap' not in p.columns:
            sensor_parts.append(s)
            continue
        gap        = int(p['sensor_gap'].iloc[0])
        extra_skip = max(0, gap - seq_len - target_horizon)
        if extra_skip >= len(s):
            raise ValueError(
                f'Season {temp}: extra_skip={extra_skip} >= sensor rows={len(s)}. '
                f'gap={gap}, seq_len={seq_len}, target_horizon={target_horizon}')
        sensor_parts.append(s.iloc[extra_skip:].reset_index(drop=True))
    sensor_norm = pd.concat(sensor_parts, ignore_index=True)
    return sensor_norm, prod_df  # prod_df unchanged


def prepare_data_flat(invernadero_id, hp,
                      train_seasons=None, val_season=None):
    """
    Full preprocessing pipeline returning flat numpy arrays for ElasticNet.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, F) where F = 2*n_sensor_pheno + n_temporal
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

    # ── 10. Flatten sequences → (N, F) ──
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


def grid_search_elasticnet(X_train, y_train, X_val, y_val, scaler_y, bc_lambda):
    """
    Grid search over alpha x l1_ratio, scored by val R².
    Returns (best_model, best_params, best_val_r2).
    """
    alphas    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    best_r2     = -np.inf
    best_params = {}
    best_model  = None

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio,
                max_iter=10000, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_val)
            # Inverse transform: MinMax → Box-Cox → original kg
            y_pred_bc = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            if bc_lambda == 'log':
                y_pred_kg = np.expm1(y_pred_bc)
            else:
                y_pred_kg = inv_boxcox(np.clip(y_pred_bc, 0, None), bc_lambda) - 1.0
            y_val_bc = scaler_y.inverse_transform(
                y_val.reshape(-1, 1)).flatten()
            if bc_lambda == 'log':
                y_val_kg = np.expm1(y_val_bc)
            else:
                y_val_kg = inv_boxcox(np.clip(y_val_bc, 0, None), bc_lambda) - 1.0
            r2 = r2_score(y_val_kg, y_pred_kg)
            if r2 > best_r2:
                best_r2     = r2
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'val_r2': r2}
                best_model  = model

    print(f'  Best val R²={best_r2:.4f}  alpha={best_params["alpha"]}  '
          f'l1_ratio={best_params["l1_ratio"]}')
    return best_model, best_params, best_r2


def evaluate_elasticnet(model, X_test, y_test, scaler_y, bc_lambda):
    """
    Evaluate on test set. Returns (y_true_kg, y_pred_kg, metrics_dict).
    """
    def inv_transform(y_scaled):
        y_bc = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        if bc_lambda == 'log':
            return np.expm1(y_bc)
        return inv_boxcox(np.clip(y_bc, 0, None), bc_lambda) - 1.0

    y_pred_scaled = model.predict(X_test)
    y_true = inv_transform(y_test)
    y_pred = inv_transform(y_pred_scaled)
    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def save_results(inv_id, best_params, metrics, y_true, y_pred, test_week_keys):
    """Save params JSON, metrics CSV, predictions NPZ, results plot."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Params JSON
    params_path = RESULTS_DIR / f'elasticnet_inv{inv_id}_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    # Metrics CSV
    rows = [{'invernadero': inv_id, 'model': 'best', **metrics}]
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / f'elasticnet_inv{inv_id}_metrics.csv', index=False)

    # Predictions NPZ
    np.savez_compressed(
        RESULTS_DIR / f'elasticnet_inv{inv_id}_predictions.npz',
        y_true=y_true, y_pred=y_pred, week_keys=test_week_keys,
    )

    # Results plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label='Actual', marker='o', ms=4)
    ax.plot(y_pred, label='ElasticNet', marker='s', ms=4, linestyle='--')
    ax.set_title(f'Invernadero {inv_id} — ElasticNet T17 Predictions')
    ax.set_xlabel('Week (test sequence)')
    ax.set_ylabel('kg')
    ax.legend()
    plt.tight_layout()
    plot_path = RESULTS_DIR / f'elasticnet_inv{inv_id}_results.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f'  Params  → {params_path}')
    print(f'  Metrics → results/elasticnet_inv{inv_id}_metrics.csv')
    print(f'  Preds   → results/elasticnet_inv{inv_id}_predictions.npz')
    print(f'  Plot    → {plot_path}')
