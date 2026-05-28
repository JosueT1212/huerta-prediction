"""
ElasticNet — Invernadero 3
==========================
Train: T13-T15 | Val: T16 | Test: T17
"""
import warnings
warnings.filterwarnings('ignore')

import yaml
from pathlib import Path

from elasticnet_yield import (
    prepare_data_flat, grid_search_elasticnet,
    evaluate_elasticnet, save_results,
)
from cnn_rnn_yield import HORIZON, print_metrics

INV_ID = 3

_hp_path = Path(__file__).parent / 'hp_inv3.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  ElasticNet Invernadero 3')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print('=' * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, bc_lambda, test_week_keys = \
        prepare_data_flat(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON)

    best_model, best_params, _ = grid_search_elasticnet(
        X_train, y_train, X_val, y_val, scaler_y, bc_lambda)

    y_true, y_pred, metrics = evaluate_elasticnet(
        best_model, X_test, y_test, scaler_y, bc_lambda)

    print_metrics(INV_ID, metrics, horizon=HORIZON)
    save_results(INV_ID, best_params, metrics, y_true, y_pred, test_week_keys)


if __name__ == '__main__':
    main()
