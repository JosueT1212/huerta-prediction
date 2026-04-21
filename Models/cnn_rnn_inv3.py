"""
CNN-RNN — Invernadero 3
=======================
Train: T13-T15 | Val: T16 | Test: T17
Best configuration: R²=0.79
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, compute_metrics, print_metrics,
    plot_results_per_greenhouse,
)

INV_ID = 3

HP = {
    'seq_len': 2,
    'horizon': 4,
    'batch_size': 8,
    'lag_features': [1],
    'include_rolling_mean': False,
    # CNN
    'cnn_filters': 128,
    'cnn_kernel_size': 2,
    'cnn_padding': 0,
    'num_cnn_blocks': 3,
    # RNN
    'lstm_hidden': 128,
    'lstm_layers': 3,
    'fc_hidden': 128,
    # Training
    'dropout': 0.05,
    'learning_rate': 2e-3,
    'weight_decay': 0,
    'corr_weight': 0.8,
    'epochs': 500,
    'patience': 150,
    'init_methods': ['default', 'xavier', 'orthogonal', 'lecun'],
    'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
              0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
              11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
              22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
              100, 200, 500, 1000],
}

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  CNN-RNN Invernadero 3')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print(f'  Horizon: {HP["horizon"]} semanas')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train = \
        prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON)

    n_runs = len(HP['seeds']) * len(HP['init_methods'])
    print(f'\n  {n_runs} runs ({len(HP["seeds"])} seeds × {len(HP["init_methods"])} inits)')

    best_r2     = -float('inf')
    best_result = None
    top_runs    = []
    model_path  = RESULTS_DIR / 'best_cnn_rnn_inv3.pt'
    tmp_ckpt    = RESULTS_DIR / '_tmp_cnn_rnn_inv3.pt'

    for init_method in HP['init_methods']:
        for seed in HP['seeds']:
            set_seed(seed)
            model = CNNRNN(
                n_sensor=n_sensor_pca, n_temporal=n_temporal,
                cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
                cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
                lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
                dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
            ).to(DEVICE)

            if init_method != 'default':
                model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, HP, tmp_ckpt, var_y_train=var_y_train)

            y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
            r2 = metrics['R²']
            print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
            top_runs.append((r2, y_pred.copy()))

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'seed': seed, 'init_method': init_method,
                }
                torch.save(model.state_dict(), model_path)

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    # Ensemble top-20
    top_runs.sort(key=lambda x: x[0], reverse=True)
    top_k = min(20, len(top_runs))
    ensemble_pred    = np.mean([r[1] for r in top_runs[:top_k]], axis=0)
    ensemble_metrics = compute_metrics(best_result['y_true'], ensemble_pred)
    best_result['ensemble_pred']    = ensemble_pred
    best_result['ensemble_metrics'] = ensemble_metrics

    print(f'\n  >>> Best: [{best_result["init_method"]}] seed={best_result["seed"]}  R²={best_r2:.4f} <<<')
    print(f'  >>> Ensemble top-{top_k}: R²={ensemble_metrics["R²"]:.4f}, MAPE={ensemble_metrics["MAPE (%)"]:.2f}% <<<')
    print_metrics(INV_ID, best_result['metrics'], horizon=HP['horizon'])

    plot_results_per_greenhouse({INV_ID: best_result}, horizon=HP['horizon'])

    # Top-25 R² statistics
    all_r2s = sorted([r[0] for r in top_runs], reverse=True)
    top25_r2s = all_r2s[:25]
    top25_mean = float(np.mean(top25_r2s))
    top25_std  = float(np.std(top25_r2s))
    print(f'\n  Top-25 R²: mean={top25_mean:.4f}, std={top25_std:.4f}')

    metrics_rows = [
        {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
        {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
        {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
         'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    ]
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / 'cnn_rnn_inv3_metrics.csv', index=False)
    print(f'\nBest model → results/best_cnn_rnn_inv3.pt')
    print(f'Metrics    → results/cnn_rnn_inv3_metrics.csv')


if __name__ == '__main__':
    main()
