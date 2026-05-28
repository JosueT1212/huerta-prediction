"""
SVR Yield Prediction — Shared Functions
========================================
flatten_sequences, grid_search_svr, evaluate_svr, save_results
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from scipy.special import inv_boxcox

RESULTS_DIR = Path(__file__).parent / 'results'


def flatten_sequences(Xs_seq, Xt_seq):
    """(N, seq_len, n_sp), (N, seq_len, n_t) → (N, 2*n_sp + n_t)

    Xs_seq: sensor PCA + pheno features concatenated (output of prepare_data CNN input channel)
    Xt_seq: temporal features
    Returns float32 array of shape (N, F) where F = 2*n_sp + n_t (typically 30).
    """
    Xs_mean = Xs_seq.mean(axis=1)    # (N, n_sp)
    Xs_std  = Xs_seq.std(axis=1)     # (N, n_sp)
    Xt_last = Xt_seq[:, -1, :]       # (N, n_t)
    return np.hstack([Xs_mean, Xs_std, Xt_last]).astype(np.float32)


def _inverse_transform(y_scaled, scaler_y, bc_lambda):
    """Invert MinMax scaling then Box-Cox (or log1p) to original kg space."""
    y_bc = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    if bc_lambda == 'log':
        return np.expm1(y_bc)
    return inv_boxcox(np.clip(y_bc, 0, None), bc_lambda) - 1.0


def compute_metrics(y_true, y_pred):
    """Return dict with R², RMSE (kg), NSE, PBIAS (%), MAPE (%)."""
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nse  = float(1 - np.sum((y_true - y_pred) ** 2) /
                 np.sum((y_true - np.mean(y_true)) ** 2))
    pbias = float(100 * np.sum(y_pred - y_true) / np.sum(y_true))
    mape  = float(100 * np.mean(
        np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))
    return {'R²': r2, 'RMSE (kg)': rmse, 'NSE': nse,
            'PBIAS (%)': pbias, 'MAPE (%)': mape}


def grid_search_svr(X_train, y_train, X_val, y_val, scaler_y, bc_lambda):
    """Grid search over 120 (C, gamma, epsilon) combos; select by val R² in kg space.

    Returns (best_model, best_params_dict, best_val_r2).
    Raises RuntimeError if all combos produce NaN R².
    """
    Cs       = [0.1, 1.0, 10.0, 100.0, 1000.0]
    gammas   = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
    epsilons = [0.01, 0.05, 0.1, 0.5]

    best_r2, best_params, best_model = -np.inf, {}, None
    y_val_kg = _inverse_transform(y_val, scaler_y, bc_lambda)

    import warnings
    total = len(Cs) * len(gammas) * len(epsilons)
    done  = 0
    for C in Cs:
        for gamma in gammas:
            for epsilon in epsilons:
                model = SVR(kernel='rbf', C=C, gamma=gamma,
                            epsilon=epsilon, max_iter=50000)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X_train, y_train)
                y_pred_kg = _inverse_transform(
                    model.predict(X_val), scaler_y, bc_lambda)
                r2 = r2_score(y_val_kg, y_pred_kg)
                if r2 > best_r2:
                    best_r2    = r2
                    best_params = {'C': C, 'gamma': gamma,
                                   'epsilon': epsilon, 'val_r2': float(r2)}
                    best_model  = model
                done += 1
                if done % 20 == 0:
                    print(f'  Grid search: {done}/{total}  best val R²={best_r2:.4f}')

    if best_model is None:
        raise RuntimeError('grid_search_svr: all 120 combos produced NaN R²')
    return best_model, best_params, best_r2


def evaluate_svr(model, X_test, y_test, scaler_y, bc_lambda):
    """Return y_true, y_pred (kg), metrics dict."""
    y_true = _inverse_transform(y_test,               scaler_y, bc_lambda)
    y_pred = _inverse_transform(model.predict(X_test), scaler_y, bc_lambda)
    return y_true, y_pred, compute_metrics(y_true, y_pred)


def save_results(inv_id, best_params, metrics, y_true, y_pred, week_keys):
    """Save params JSON, metrics CSV, predictions NPZ, results PNG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'svr_inv{inv_id}'

    # params JSON
    params_path = RESULTS_DIR / f'{tag}_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'  Params     → {params_path}')

    # metrics CSV
    row = {'invernadero': inv_id, 'model': 'best', **metrics}
    metrics_path = RESULTS_DIR / f'{tag}_metrics.csv'
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    print(f'  Metrics    → {metrics_path}')

    # predictions NPZ
    npz_path = RESULTS_DIR / f'{tag}_predictions.npz'
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, week_keys=week_keys)
    print(f'  Predictions→ {npz_path}')

    # results plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label='Actual',    marker='o', linewidth=1.5)
    ax.plot(y_pred, label='Predicted', marker='x', linewidth=1.5, linestyle='--')
    ax.set_title(f'SVR Inv{inv_id} — T17 Test  '
                 f'R²={metrics["R²"]:.3f}  MAPE={metrics["MAPE (%)"]:.1f}%')
    ax.set_xlabel('Week'); ax.set_ylabel('kg')
    ax.legend(); fig.tight_layout()
    png_path = RESULTS_DIR / f'{tag}_results.png'
    fig.savefig(png_path, dpi=150); plt.close(fig)
    print(f'  Plot       → {png_path}')
