"""
SVR — Invernadero 3
====================
Train: T13-T15 | Val: T16 | Test: T17
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from pathlib import Path
from sklearn.svm import SVR
import numpy as np

from cnn_rnn_yield import prepare_data, HORIZON, print_metrics
from svr_yield import (
    flatten_sequences, grid_search_svr,
    evaluate_svr, save_results,
)

INV_ID        = 3
TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'

_hp_path = Path(__file__).parent / 'hp_inv3.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)


def main():
    print('=' * 60)
    print('  SVR Invernadero 3')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print('=' * 60)

    (Xs_tr_seq, Xt_tr_seq, y_tr,
     Xs_va_seq, Xt_va_seq, y_va,
     Xs_te_seq, Xt_te_seq, y_te,
     scaler_y, bc_lambda, test_week_keys) = prepare_data(
        INV_ID, HP,
        train_seasons=TRAIN_SEASONS,
        val_season=VAL_SEASON,
        skip_first_weeks=HP.get('skip_first_weeks', 3),
        return_arrays=True,
    )

    X_train = flatten_sequences(Xs_tr_seq, Xt_tr_seq)
    X_val   = flatten_sequences(Xs_va_seq, Xt_va_seq)
    X_test  = flatten_sequences(Xs_te_seq, Xt_te_seq)
    print(f'  Flat features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    best_model, best_params, best_val_r2 = grid_search_svr(
        X_train, y_tr, X_val, y_va, scaler_y, bc_lambda)
    print(f'  Best params: C={best_params["C"]}, gamma={best_params["gamma"]}, '
          f'epsilon={best_params["epsilon"]}  val R²={best_val_r2:.4f}')

    # Refit on train+val combined
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_tr, y_va])
    final_model = SVR(kernel='rbf', C=best_params['C'],
                      gamma=best_params['gamma'],
                      epsilon=best_params['epsilon'], max_iter=50000)
    final_model.fit(X_tv, y_tv)

    y_true, y_pred, metrics = evaluate_svr(
        final_model, X_test, y_te, scaler_y, bc_lambda)

    print_metrics(INV_ID, metrics, horizon=HORIZON)
    save_results(INV_ID, best_params, metrics, y_true, y_pred, test_week_keys)


if __name__ == '__main__':
    main()
