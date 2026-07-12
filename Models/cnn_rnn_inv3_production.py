"""
CNN-RNN — Invernadero 3 — PRODUCTION REFIT
===========================================
Trains the winning grid-search configuration on ALL 5 seasons (T13-T17)
pooled, with no holdout, for a fixed epoch count (no early stopping).
This produces the final deployable model — not an evaluation checkpoint.

Winning config source: Models/results/inv3.log (see docs/superpowers/plans/
2026-07-11-inv3-inv4-horizon5-production.md, Task 3).
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import torch
import yaml
from pathlib import Path

from cnn_rnn_yield import (
    DEVICE, RESULTS_DIR,
    prepare_data, set_seed, init_weights,
    CNNRNN, train_model,
)

INV_ID = 3

# ── Winning config from Task 3 (Models/results/inv3.log) ──
WINNING_INIT_METHOD = 'lecun'
WINNING_SEED = 21
WINNING_EPOCHS = 166

_hp_path = Path(__file__).parent / 'hp_inv3.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

HP = copy.deepcopy(HP)
HP['epochs'] = WINNING_EPOCHS
HP['patience'] = WINNING_EPOCHS + 10  # never triggers before `epochs` completes

ALL_SEASONS = ['T13', 'T14', 'T15', 'T16', 'T17']


def main():
    print('=' * 60)
    print('  CNN-RNN Invernadero 3 — PRODUCTION REFIT')
    print(f'  Train: {ALL_SEASONS} (all data, no holdout)')
    print(f'  Config: init={WINNING_INIT_METHOD}, seed={WINNING_SEED}, epochs={WINNING_EPOCHS}')
    print('=' * 60)

    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=ALL_SEASONS, val_season='T16',
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'production_pipeline_inv3.pkl')

    set_seed(WINNING_SEED)
    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    model = CNNRNN(
        n_sensor=n_sensor_pca, n_temporal=n_temporal,
        cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
        cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
        lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
        dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
        n_out=_n_out,
    ).to(DEVICE)

    if WINNING_INIT_METHOD != 'default':
        model = model.cpu(); init_weights(model, WINNING_INIT_METHOD); model = model.to(DEVICE)

    model_path = RESULTS_DIR / 'production_cnn_rnn_inv3.pt'
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, HP, model_path, var_y_train=var_y_train)

    print(f'\n  Trained {len(train_losses)} epochs (target was {WINNING_EPOCHS}).')
    print(f'  Final train loss: {train_losses[-1]:.6f}')
    print(f'\nProduction model  → {model_path}')
    print(f'Inference pipeline → {RESULTS_DIR / "production_pipeline_inv3.pkl"}')


if __name__ == '__main__':
    main()
