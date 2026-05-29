"""
Quantile-Win Analysis — All Seasons
=====================================
Uses best trained model to predict on T13–T17 for each invernadero.
Checks if "q90 wins early / q50 wins late" pattern is stable across seasons.

T16, T17 : use model's own train/val scaler (clean).
T13-T15  : use same scaler but pass each as test with other two as train.
           Scaler will differ slightly — acceptable for exploratory pattern check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import inv_boxcox

from cnn_rnn_yield import DEVICE, CNNRNN, prepare_data

RESULTS_DIR   = Path(__file__).parent / 'results'
QUANTILES     = [0.1, 0.5, 0.9]
LABELS        = ['q10', 'q50', 'q90']
COLORS        = ['#e74c3c', '#2980b9', '#27ae60']
ALL_SEASONS   = ['T13', 'T14', 'T15', 'T16', 'T17']


def load_hp(inv_id):
    p = Path(__file__).parent / f'hp_inv{inv_id}.yaml'
    with open(p) as f:
        return yaml.safe_load(f)


def build_model(HP, n_sensor_pca, n_temporal, n_out):
    m = CNNRNN(
        n_sensor=n_sensor_pca, n_temporal=n_temporal,
        cnn_filters=HP['cnn_filters'], cnn_kernel_size=HP['cnn_kernel_size'],
        cnn_padding=HP['cnn_padding'], num_cnn_blocks=HP['num_cnn_blocks'],
        lstm_hidden=HP['lstm_hidden'], lstm_layers=HP['lstm_layers'],
        dropout=HP['dropout'], fc_hidden=HP['fc_hidden'],
        n_out=n_out,
    ).to(DEVICE)
    return m


def inverse_transform(arr_norm, scaler_y, bc_lambda):
    arr_bc = scaler_y.inverse_transform(arr_norm.reshape(-1, 1)).flatten()
    if bc_lambda == 'log':
        return np.expm1(arr_bc)
    elif bc_lambda is not None:
        return inv_boxcox(np.clip(arr_bc, 0, None), bc_lambda) - 1.0
    return arr_bc


def infer_q(model, Xs, Xt, scaler_y, bc_lambda, quantiles):
    """(N,seq_len,n_sp),(N,seq_len,n_t) → dict {q: array(N)}"""
    model.eval()
    with torch.no_grad():
        out = model(
            torch.FloatTensor(Xs).to(DEVICE),
            torch.FloatTensor(Xt).to(DEVICE),
        ).cpu().numpy()   # (N, n_q)
    out = np.sort(out, axis=1)   # enforce q10 ≤ q50 ≤ q90
    return {q: inverse_transform(out[:, i], scaler_y, bc_lambda)
            for i, q in enumerate(quantiles)}


def analyse_inv(inv_id):
    HP    = load_hp(inv_id)
    n_out = len(HP.get('quantiles', [0.5])) if HP.get('loss_type') == 'quantile' else 1
    q_list = HP.get('quantiles', [0.5]) if HP.get('loss_type') == 'quantile' else [0.5]

    model_path = RESULTS_DIR / f'best_cnn_rnn_inv{inv_id}.pt'
    if not model_path.exists():
        print(f'Inv{inv_id}: no model file, skipping.')
        return

    # ── single prepare_data call — original split, return arrays ──────
    result = prepare_data(
        inv_id, HP,
        train_seasons=['T13', 'T14', 'T15'],
        val_season='T16',
        skip_first_weeks=HP.get('skip_first_weeks', 0),
        ramp_weeks=HP.get('ramp_weeks', 4),
        ramp_weight=HP.get('ramp_weight', 1.0),
        return_arrays=True,
    )
    # (Xs_tr,Xt_tr,y_tr, Xs_va,Xt_va,y_va, Xs_te,Xt_te,y_te, scaler_y, bc_lambda, week_keys)
    Xs_tr, Xt_tr, y_tr = result[0], result[1], result[2]
    Xs_va, Xt_va, y_va = result[3], result[4], result[5]
    Xs_te, Xt_te, y_te = result[6], result[7], result[8]
    scaler_y, bc_lambda = result[9], result[10]

    n_sensor_pca = Xs_tr.shape[2]
    n_temporal   = Xt_tr.shape[2]

    model = build_model(HP, n_sensor_pca, n_temporal, n_out)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'\nInv{inv_id}: model loaded  n_sensor={n_sensor_pca} n_temp={n_temporal}')

    # ── run inference on all splits ────────────────────────────────────
    # T13+T14+T15 combined (111 rows) — w_rel = sequential index, not per-season
    # T16 (41 rows) — clean single season
    # T17 (39 rows) — clean single season
    splits = {
        'T13-15 (train, mixed)': (Xs_tr, Xt_tr, y_tr),
        'T16 (val)':             (Xs_va, Xt_va, y_va),
        'T17 (test)':            (Xs_te, Xt_te, y_te),
    }

    season_results = {}
    for label, (Xs, Xt, y_norm) in splits.items():
        y_true  = inverse_transform(y_norm.flatten(), scaler_y, bc_lambda)
        q_preds = infer_q(model, Xs, Xt, scaler_y, bc_lambda, q_list)

        n     = len(y_true)
        w_rel = np.arange(n)   # T16/T17: index = relative week; train: sequential

        preds_stacked = np.stack([q_preds[q] for q in q_list], axis=1)
        abs_err = np.abs(preds_stacked - y_true[:, None])
        winners = np.argmin(abs_err, axis=1)

        season_results[label] = {
            'y_true': y_true, 'q_preds': q_preds,
            'winners': winners, 'w_rel': w_rel, 'n': n,
        }

        win_pct = [(winners == i).mean() * 100 for i in range(len(q_list))]
        print(f'  {label} (n={n}): ' +
              '  '.join(f'{l}={p:.0f}%' for l, p in zip(LABELS[:len(q_list)], win_pct)))

    if not season_results:
        return

    # ── combined plot ──────────────────────────────────────────────────
    n_seasons = len(season_results)
    fig, axes = plt.subplots(n_seasons, 1, figsize=(14, 3.5 * n_seasons), sharex=False)
    if n_seasons == 1:
        axes = [axes]

    for ax, (season, res) in zip(axes, season_results.items()):
        w = res['w_rel']
        y = res['y_true']
        winners = res['winners']
        q_preds = res['q_preds']
        q_keys  = list(q_preds.keys())

        ax.plot(w, y, 'k-', lw=2, label='y_true', zorder=5)
        if len(q_keys) >= 3:
            ax.fill_between(w, q_preds[q_keys[0]], q_preds[q_keys[-1]],
                            alpha=0.15, color=COLORS[1])
            ax.plot(w, q_preds[q_keys[1]], '--', color=COLORS[1], lw=1, label='q50')

        for i, (lbl, col) in enumerate(zip(LABELS[:len(q_keys)], COLORS)):
            idx = np.where(winners == i)[0]
            ax.scatter(w[idx], y[idx], color=col, s=60, zorder=6, label=f'{lbl} wins')

        ax.axvspan(-0.5, 4.5, alpha=0.07, color='orange')
        ax.set_title(f'Inv{inv_id} {season} — quantile wins  '
                     f'({" ".join(f"{l}={( winners==i).mean()*100:.0f}%" for i,l in enumerate(LABELS[:len(q_keys)]))})')
        ax.set_ylabel('kg'); ax.legend(ncol=4, fontsize=7)

    axes[-1].set_xlabel('Relative week in season')
    fig.suptitle(f'Invernadero {inv_id} — Quantile Win Pattern Across Seasons', fontsize=13, y=1.01)
    fig.tight_layout()
    out = RESULTS_DIR / f'quantile_win_allseasons_inv{inv_id}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot → {out}')

    # ── consistency check: does q90 dominate weeks 0-4 in all seasons? ─
    print(f'\n  Ramp-up (weeks 0–4) consistency check:')
    for season, res in season_results.items():
        ramp_mask = res['w_rel'] < 5
        if ramp_mask.sum() == 0:
            continue
        w_ramp = res['winners'][ramp_mask]
        pcts   = [(w_ramp == i).mean() for i in range(len(q_list))]
        dom    = LABELS[int(np.argmax(pcts))]
        print(f'    {season}: dominant={dom}  ' +
              '  '.join(f'{l}={p:.2f}' for l,p in zip(LABELS[:len(q_list)], pcts)))


def main():
    for inv_id in [3, 4]:
        analyse_inv(inv_id)


if __name__ == '__main__':
    main()
