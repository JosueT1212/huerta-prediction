"""
Empirical Quantile-Win Analysis
================================
For each test week: which quantile (q10, q50, q90) is closest to y_true?
Plot winner vs relative week index to check if a stable pattern exists.
Runs on any available invernadero predictions npz.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'
QUANTILES   = [0.1, 0.5, 0.9]
LABELS      = ['q10', 'q50', 'q90']
COLORS      = ['#e74c3c', '#2980b9', '#27ae60']


def load_inv(inv_id):
    path = RESULTS_DIR / f'cnn_rnn_inv{inv_id}_predictions.npz'
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'y_true':    d['y_true'],
        'q50':       d['y_pred'],
        'q10':       d['pi_lower'],
        'q90':       d['pi_upper'],
        'week_keys': d['week_keys'],
    }


def quantile_win_stats(data):
    """Return array of winner index (0=q10,1=q50,2=q90) per sample."""
    y = data['y_true']
    preds = np.stack([data['q10'], data['q50'], data['q90']], axis=1)  # (N,3)
    abs_err = np.abs(preds - y[:, None])                                # (N,3)
    return np.argmin(abs_err, axis=1)                                   # (N,)


def analyse(inv_id, data):
    n        = len(data['y_true'])
    w_rel    = np.arange(n)                 # T17 is single season → index = relative week
    winners  = quantile_win_stats(data)

    # ── summary stats ──────────────────────────────────────────────────
    print(f'\n=== Invernadero {inv_id} (n={n}) ===')
    for i, lbl in enumerate(LABELS):
        mask = winners == i
        pct  = mask.mean() * 100
        weeks_won = w_rel[mask]
        print(f'  {lbl}: {pct:5.1f}%  weeks: {sorted(weeks_won.tolist())}')

    # Phase split (ramp / plateau / late)
    ramp  = w_rel < 5
    late  = w_rel >= (n - 5)
    mid   = ~ramp & ~late
    for phase_name, mask in [('ramp(0-4)', ramp), ('plateau(5..)', mid), ('late(-5)', late)]:
        if mask.sum() == 0:
            continue
        w_phase  = winners[mask]
        dominant = LABELS[np.bincount(w_phase, minlength=3).argmax()]
        print(f'  {phase_name:14s}: dominant={dominant}  '
              f'q10={( w_phase==0).mean():.2f} q50={(w_phase==1).mean():.2f} q90={(w_phase==2).mean():.2f}')

    # ── plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    # Top: actual curves + winner highlighted
    ax = axes[0]
    ax.plot(w_rel, data['y_true'], 'k-', lw=2, label='y_true', zorder=5)
    ax.plot(w_rel, data['q50'],    '--', color=COLORS[1], lw=1.5, label='q50')
    ax.fill_between(w_rel, data['q10'], data['q90'],
                    alpha=0.15, color=COLORS[1], label='q10–q90 band')

    for i, (lbl, col) in enumerate(zip(LABELS, COLORS)):
        idx = np.where(winners == i)[0]
        ax.scatter(w_rel[idx], data['y_true'][idx],
                   color=col, s=80, zorder=6, label=f'{lbl} wins')

    ax.set_title(f'Invernadero {inv_id} — T17 Test: which quantile is closest to y_true?')
    ax.set_ylabel('kg'); ax.legend(ncol=3, fontsize=8)
    ax.axvspan(0, 4.5, alpha=0.05, color='orange', label='ramp-up')

    # Bottom: winner bar chart
    ax2 = axes[1]
    bottom = np.zeros(n)
    for i, (lbl, col) in enumerate(zip(LABELS, COLORS)):
        vals = (winners == i).astype(float)
        ax2.bar(w_rel, vals, bottom=bottom, color=col, label=lbl, alpha=0.85)
        bottom += vals

    ax2.set_ylabel('winner'); ax2.set_xlabel('Relative week in season (T17)')
    ax2.set_yticks([0, 1]); ax2.legend(ncol=3, fontsize=8)

    fig.tight_layout()
    out = RESULTS_DIR / f'quantile_win_inv{inv_id}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Plot → {out}')

    # ── interval coverage ──────────────────────────────────────────────
    y   = data['y_true']
    cov = np.mean((y >= data['q10']) & (y <= data['q90']))
    avg_w = np.mean(data['q90'] - data['q10'])
    print(f'  PI coverage (q10–q90): {cov:.3f}  avg_width={avg_w:.0f} kg')

    # ── MAE per quantile ───────────────────────────────────────────────
    for lbl, key in zip(LABELS, ['q10', 'q50', 'q90']):
        mae = np.mean(np.abs(data[lbl] - y))
        print(f'  MAE {lbl}: {mae:.1f} kg')


def main():
    found = False
    for inv_id in [3, 4]:
        data = load_inv(inv_id)
        if data is None:
            print(f'Inv{inv_id}: no predictions file yet, skipping.')
            continue
        found = True
        analyse(inv_id, data)

    if not found:
        print('No predictions files found in results/.')


if __name__ == '__main__':
    main()
