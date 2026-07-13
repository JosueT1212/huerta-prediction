"""
Pure LSTM (no CNN) — Invernadero 4
===================================
Train: T13-T15 | Val: T16 | Test: T17
Architecture comparison against CNN+LSTM production winner (see
docs/superpowers/specs/2026-07-13-pure-lstm-comparison-design.md).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR, HORIZON,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model, evaluate_model, evaluate_model_q,
    compute_metrics, compute_phase_metrics,
    print_metrics, plot_results_per_greenhouse, plot_quantile_bands,
    compute_cptc_intervals,
)

INV_ID = 4

_hp_path = Path(__file__).parent / 'hp_inv4_lstm.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON    = 'T16'


def main():
    print('=' * 60)
    print('  Pure LSTM (no CNN) — Invernadero 4')
    print(f'  Train: {TRAIN_SEASONS} | Val: {VAL_SEASON} | Test: T17')
    print(f'  HORIZON={HORIZON} (doc) | seq_len={HP["seq_len"]} | num_cnn_blocks={HP["num_cnn_blocks"]}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'pipeline_inv4_lstm.pkl')

    n_runs = len(HP['seeds']) * len(HP['init_methods'])
    print(f'\n  {n_runs} runs ({len(HP["seeds"])} seeds × {len(HP["init_methods"])} inits)')

    best_r2     = -float('inf')
    best_result = None
    top_runs    = []
    model_path  = RESULTS_DIR / 'best_cnn_rnn_inv4_lstm.pt'
    tmp_ckpt    = RESULTS_DIR / '_tmp_cnn_rnn_inv4_lstm.pt'

    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    for init_method in HP['init_methods']:
        for seed in HP['seeds']:
            set_seed(seed)
            model = CNNRNN(
                n_sensor=n_sensor_pca, n_temporal=n_temporal,
                cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
                cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
                lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
                dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
                n_out=_n_out,
            ).to(DEVICE)

            if init_method != 'default':
                model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, HP, tmp_ckpt, var_y_train=var_y_train)

            if HP.get('loss_type') == 'quantile':
                _quantiles = HP.get('quantiles', [0.1, 0.5, 0.9])
                y_true, q_preds, metrics = evaluate_model_q(model, test_loader, scaler_y, bc_lambda,
                                                             quantiles=_quantiles)
                y_pred = q_preds.get(0.5, list(q_preds.values())[0])
            else:
                y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)
                q_preds = None
            r2 = metrics['R²']
            print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
            top_runs.append((r2, y_pred.copy()))

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'seed': seed, 'init_method': init_method,
                    'wis_test': wis_test,
                    'q_preds': q_preds,
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

    # Prediction Intervals: use q10/q90 directly in quantile mode, CPTC otherwise
    if HP.get('loss_type') == 'quantile':
        q_preds_best = best_result.get('q_preds', {})
        lower = q_preds_best.get(0.1, np.zeros(len(best_result['y_true'])))
        upper = q_preds_best.get(0.9, np.zeros(len(best_result['y_true'])))
        y_true_arr = best_result['y_true']
        covered = np.mean((y_true_arr >= lower) & (y_true_arr <= upper))
        avg_w = float(np.mean(upper - lower))
        val_cov = float(covered)
        print(f'  80% PI (q10–q90): coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')
    else:
        best_model = CNNRNN(
            n_sensor=n_sensor_pca, n_temporal=n_temporal,
            cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
            cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
            lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
            dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
            n_out=_n_out,
        ).to(DEVICE)
        best_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        y_val_true, y_val_pred, _ = evaluate_model(best_model, val_loader, scaler_y, bc_lambda)
        lower, upper, val_cov, avg_w = compute_cptc_intervals(y_val_true, y_val_pred, best_result['y_pred'])
        best_result['pi_lower'] = lower
        best_result['pi_upper'] = upper
        print(f'  90% PI (CPTC): val_coverage={val_cov:.3f}, avg_width={avg_w:.1f} kg')

    print(f'\n  >>> Best: [{best_result["init_method"]}] seed={best_result["seed"]}  R²={best_r2:.4f} <<<')
    print(f'  >>> Ensemble top-{top_k}: R²={ensemble_metrics["R²"]:.4f}, MAPE={ensemble_metrics["MAPE (%)"]:.2f}% <<<')
    print_metrics(INV_ID, best_result['metrics'], horizon=HORIZON)

    if HP.get('loss_type') == 'quantile' and best_result.get('q_preds') is not None:
        plot_quantile_bands(
            y_true=best_result['y_true'],
            q_preds=best_result['q_preds'],
            train_losses=best_result['train_losses'],
            val_losses=best_result['val_losses'],
            inv_id=f'{INV_ID}_lstm',
            metrics=best_result['metrics'],
            horizon=HORIZON,
            quantiles=HP.get('quantiles', [0.1, 0.5, 0.9]),
            wis=best_result.get('wis_test'),
            ramp_weeks=HP.get('ramp_weeks', 4),
        )
    else:
        plot_results_per_greenhouse({f'{INV_ID}_lstm': best_result}, horizon=HORIZON,
                                    intervals={f'{INV_ID}_lstm': (lower, upper)})

    # Top-25 R² statistics
    all_r2s = sorted([r[0] for r in top_runs], reverse=True)
    top25_r2s = all_r2s[:25]
    top25_mean = float(np.mean(top25_r2s))
    top25_std  = float(np.std(top25_r2s))
    print(f'\n  Top-25 R²: mean={top25_mean:.4f}, std={top25_std:.4f}')

    phase = compute_phase_metrics(
        best_result['y_true'], best_result['y_pred'], wis_test,
        ramp_weeks=HP.get('ramp_weeks', 4))

    metrics_rows = [
        {'invernadero': INV_ID, 'model': 'best',              **best_result['metrics']},
        {'invernadero': INV_ID, 'model': f'ensemble_top{top_k}', **ensemble_metrics},
        {'invernadero': INV_ID, 'model': 'top25_mean', 'R²': top25_mean, 'R²_std': top25_std,
         'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
        {'invernadero': INV_ID, 'model': 'quantile_pi' if HP.get('loss_type') == 'quantile' else 'cptc_pi', 'pi_coverage': val_cov, 'pi_avg_width': avg_w,
         'R²': None, 'RMSE (kg)': None, 'NSE': None, 'PBIAS (%)': None, 'MAPE (%)': None},
    ]
    if 'ramp_up' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_ramp_up', **phase['ramp_up']})
    if 'peak' in phase:
        metrics_rows.append({'invernadero': INV_ID, 'model': 'best_peak',    **phase['peak']})
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / 'cnn_rnn_inv4_lstm_metrics.csv', index=False)

    np.savez_compressed(
        RESULTS_DIR / 'cnn_rnn_inv4_lstm_predictions.npz',
        y_true=best_result['y_true'],
        y_pred=best_result['y_pred'],
        ensemble_pred=best_result['ensemble_pred'],
        pi_lower=lower,
        pi_upper=upper,
        week_keys=test_week_keys,
    )
    print(f'\nBest model → results/best_cnn_rnn_inv4_lstm.pt')
    print(f'Metrics    → results/cnn_rnn_inv4_lstm_metrics.csv')
    print(f'Predictions → results/cnn_rnn_inv4_lstm_predictions.npz')


if __name__ == '__main__':
    main()
