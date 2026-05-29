"""
Motor de inferencia para el demo (modo .npz).

Carga los arrays precalculados de cnn_rnn_inv{3,4}_predictions.npz y las
métricas de cnn_rnn_inv{3,4}_metrics.csv. No carga ni corre el modelo PyTorch.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'Models' / 'results'


def _week_key_to_iso(wk: int) -> dict:
    return {'year': int(wk // 100), 'iso_week': int(wk % 100), 'week_key': int(wk)}


def _load_npz(inv_id: int) -> dict:
    t0 = time.time()
    npz_path = RESULTS_DIR / f'cnn_rnn_inv{inv_id}_predictions.npz'
    d = np.load(npz_path, allow_pickle=True)

    y_true        = d['y_true'].astype(float)
    y_pred        = d['y_pred'].astype(float)
    ensemble_pred = d['ensemble_pred'].astype(float)
    pi_lower      = d['pi_lower'].astype(float)
    pi_upper      = d['pi_upper'].astype(float)
    week_keys     = d['week_keys'].astype(int)

    # Métricas desde CSV
    csv_path = RESULTS_DIR / f'cnn_rnn_inv{inv_id}_metrics.csv'
    metrics = {'R²': None, 'RMSE (kg)': None, 'MAPE (%)': None,
               'NSE': None, 'PBIAS (%)': None}
    pi_coverage = pi_avg_width = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        best = df[df['model'] == 'best']
        if not best.empty:
            row = best.iloc[0]
            for k in metrics:
                if k in row and not pd.isna(row[k]):
                    metrics[k] = float(row[k])
        cptc = df[df['model'] == 'cptc_pi']
        if not cptc.empty:
            row = cptc.iloc[0]
            if 'pi_coverage' in row and not pd.isna(row['pi_coverage']):
                pi_coverage = float(row['pi_coverage'])
            if 'pi_avg_width' in row and not pd.isna(row['pi_avg_width']):
                pi_avg_width = float(row['pi_avg_width'])

    elapsed = time.time() - t0
    r2 = metrics.get('R²') or 0.0
    mape = metrics.get('MAPE (%)') or 0.0
    print(f'[engine] inv{inv_id} listo en {elapsed:.2f}s — '
          f'R²={r2:.4f}, MAPE={mape:.2f}%  (fuente: .npz)')

    return {
        'inv_id': inv_id,
        'week_keys': week_keys,
        'y_true': y_true,
        'y_pred': y_pred,
        'ensemble_pred': ensemble_pred,
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
        'metrics': metrics,
        'pi_coverage': pi_coverage,
        'pi_avg_width': pi_avg_width,
    }


class InferenceEngine:
    def __init__(self):
        self._cache: dict[int, dict] = {}

    def get(self, inv_id: int, force: bool = False) -> dict:
        if force or inv_id not in self._cache:
            self._cache[inv_id] = _load_npz(inv_id)
        return self._cache[inv_id]

    def warm(self, inv_ids=(3, 4)) -> None:
        for i in inv_ids:
            try:
                self.get(i)
            except Exception as e:  # noqa: BLE001
                print(f'[engine] inv{i} FALLÓ: {e}')

    def loaded(self) -> list[int]:
        return sorted(self._cache.keys())

    def payload(self, inv_id: int) -> dict:
        s = self.get(inv_id)
        return {
            'inv_id': inv_id,
            'horizon_weeks': 6,
            'n_weeks': int(len(s['week_keys'])),
            'weeks': [_week_key_to_iso(int(w)) for w in s['week_keys']],
            'y_true': s['y_true'].tolist(),
            'y_pred': s['y_pred'].tolist(),
            'ensemble_pred': s['ensemble_pred'].tolist(),
            'pi_lower': s['pi_lower'].tolist(),
            'pi_upper': s['pi_upper'].tolist(),
        }

    def metrics(self, inv_id: int) -> dict:
        s = self.get(inv_id)
        m = {k: (None if v is None else float(v)) for k, v in s['metrics'].items()}
        return {
            'inv_id': inv_id,
            'source': 'npz_cache',
            'best': m,
            'cptc_pi': {
                'pi_coverage': s['pi_coverage'],
                'pi_avg_width': s['pi_avg_width'],
            },
        }

    def window(self, inv_id: int, cursor: int | None = None,
               horizon: int = 6, past: int = 5) -> dict:
        s = self.get(inv_id)
        n = int(len(s['week_keys']))
        if cursor is None:
            cursor = min(n // 2, n - horizon)
        cursor = max(0, min(cursor, n - 1))
        start_past = max(0, cursor - past)

        wk = [_week_key_to_iso(int(w)) for w in s['week_keys']]

        def row(i, observed):
            return {
                'week_key': wk[i]['week_key'],
                'year': wk[i]['year'],
                'iso_week': wk[i]['iso_week'],
                'observed': observed,
                'real': float(s['y_true'][i]) if observed else None,
                'pred': float(s['y_pred'][i]),
                'lo': float(s['pi_lower'][i]),
                'hi': float(s['pi_upper'][i]),
            }

        history = [row(i, observed=True) for i in range(start_past, cursor)]
        predictions = []
        for i in range(cursor, min(cursor + horizon, n)):
            r = row(i, observed=False)
            r['horizon'] = i - cursor + 1
            r['real'] = float(s['y_true'][i])
            predictions.append(r)

        return {
            'inv_id': inv_id,
            'cursor': cursor,
            'n_weeks': n,
            'horizon': horizon,
            'past': past,
            'history': history,
            'predictions': predictions,
        }


ENGINE = InferenceEngine()
