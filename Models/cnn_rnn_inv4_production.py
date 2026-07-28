"""
CNN-RNN — Invernadero 4 — PRODUCTION REFIT (10-seed ensemble)
================================================================
Trains the winning grid-search configuration on ALL 5 seasons (T13-T17)
pooled, with no holdout, for a fixed epoch count (no early stopping) —
once per each of the top-10 seed/init pairs from the post-LOSO-fix
full 216-run grid search (run log:
/Users/josuetapiahernandez/.claude/jobs/27f9213d/tmp/inv4_full216_final.log).

Ensembling across seeds instead of shipping a single winning seed reduces
the seed-to-seed variance documented for this dataset (mean R²=0.49,
std=0.11 across 216 seeds — a single seed is not a reliable generalization
estimate at this sample size). See industry practice: FT-Transformer/TabM
report ensembles of ~15 seeds by default for small tabular data.

Saves a single .pt file containing a LIST of 10 state_dicts (not one
state_dict) — scripts/live_inference.py loads all 10 and averages predictions.
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

INV_ID = 4

# ── Top-10 (init, seed, epoch) from the post-LOSO-fix full 216-run grid ──
# epoch = early-stop epoch of that seed's original sweep run (or the max
# epoch cap if that seed ran to completion without early stopping).
ENSEMBLE_MEMBERS = [
    ('default',      23, 251),
    ('orthogonal',   39, 252),
    ('xavier',      500, 265),
    ('orthogonal',  200, 262),
    ('orthogonal',   28, 254),
    ('orthogonal',   20, 252),
    ('orthogonal',   22, 251),
    ('orthogonal',    1, 295),
    ('orthogonal',   13, 251),
    ('default',       3, 252),
]

_hp_path = Path(__file__).parent / 'hp_inv4.yaml'
with open(_hp_path) as f:
    HP = yaml.safe_load(f)

ALL_SEASONS = ['T13', 'T14', 'T15', 'T16', 'T17']


def main():
    print('=' * 60)
    print('  CNN-RNN Invernadero 4 — PRODUCTION REFIT (10-seed ensemble)')
    print(f'  Train: {ALL_SEASONS} (all data, no holdout)')
    print(f'  {len(ENSEMBLE_MEMBERS)} ensemble members')
    print('=' * 60)

    # Data pipeline (scalers, PCA, seq construction) is seed-independent —
    # fit once, reuse the same loaders for every ensemble member.
    train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test = \
        prepare_data(INV_ID, HP, train_seasons=ALL_SEASONS, val_season='T16',
                     skip_first_weeks=HP.get('skip_first_weeks', 0),
                     ramp_weeks=HP.get('ramp_weeks', 4),
                     ramp_weight=HP.get('ramp_weight', 1.0),
                     pipeline_path=RESULTS_DIR / 'production_pipeline_inv4.pkl')

    _n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    tmp_ckpt = RESULTS_DIR / '_tmp_production_inv4.pt'
    state_dicts = []

    for i, (init_method, seed, epochs) in enumerate(ENSEMBLE_MEMBERS):
        member_hp = copy.deepcopy(HP)
        member_hp['epochs'] = epochs
        member_hp['patience'] = epochs + 10  # unused when track_best=False

        set_seed(seed)
        model = CNNRNN(
            n_sensor=n_sensor_pca, n_temporal=n_temporal,
            cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
            cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
            lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
            dropout=HP['dropout'], fc_hidden=HP['fc_hidden'], fc_layers=HP.get('fc_layers', 1),
            n_out=_n_out,
        ).to(DEVICE)

        if init_method != 'default':
            model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, member_hp, tmp_ckpt, var_y_train=var_y_train,
            track_best=False)

        state_dicts.append(model.state_dict())
        print(f'  [{i+1}/{len(ENSEMBLE_MEMBERS)}] [{init_method}] seed={seed} epochs={epochs} '
              f'→ final train loss: {train_losses[-1]:.6f}')

    if tmp_ckpt.exists():
        tmp_ckpt.unlink()

    model_path = RESULTS_DIR / 'production_cnn_rnn_inv4.pt'
    torch.save(state_dicts, model_path)

    print(f'\nProduction ensemble ({len(state_dicts)} members) → {model_path}')
    print(f'Inference pipeline → {RESULTS_DIR / "production_pipeline_inv4.pkl"}')


if __name__ == '__main__':
    main()
