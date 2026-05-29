"""
Motor de inferencia para el demo.

Carga best_cnn_rnn_inv{3,4}.pt UNA sola vez al arrancar, corre el forward sobre
todo el test set T17 y cachea los arrays por invernadero. "Correr en la semana X"
es solo cortar (slice) esos arrays por un cursor — el modelo no se re-ejecuta.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'Models'
RESULTS_DIR = MODELS_DIR / 'results'
sys.path.insert(0, str(MODELS_DIR))

from cnn_rnn_yield import (  # noqa: E402
    DEVICE, CNNRNN, prepare_data, evaluate_model,
    evaluate_loader, compute_cptc_intervals,
)

TRAIN_SEASONS = ['T13', 'T14', 'T15']
VAL_SEASON = 'T16'


def _load_hp(inv_id: int) -> dict:
    with open(MODELS_DIR / f'hp_inv{inv_id}.yaml') as f:
        return yaml.safe_load(f)


def _build_model(hp: dict, n_sensor: int, n_temporal: int) -> CNNRNN:
    return CNNRNN(
        n_sensor=n_sensor, n_temporal=n_temporal,
        cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
        cnn_padding=hp['cnn_padding'], num_cnn_blocks=hp['num_cnn_blocks'],
        lstm_hidden=hp['lstm_hidden'], lstm_layers=hp['lstm_layers'],
        dropout=hp['dropout'], fc_hidden=hp['fc_hidden'],
    ).to(DEVICE)


def _week_key_to_iso(wk: int) -> dict:
    return {'year': int(wk // 100), 'iso_week': int(wk % 100), 'week_key': int(wk)}


class InferenceEngine:
    """Carga perezosa + cache por invernadero."""

    def __init__(self):
        self._cache: dict[int, dict] = {}

    # ── cómputo (1 vez por invernadero) ──────────────────────────────────
    def _run(self, inv_id: int) -> dict:
        t0 = time.time()
        hp = _load_hp(inv_id)
        (train_loader, val_loader, test_loader, scaler_y, bc_lambda,
         n_sensor, n_temporal, var_y_train, week_keys) = prepare_data(
            inv_id, hp, train_seasons=TRAIN_SEASONS, val_season=VAL_SEASON,
            skip_first_weeks=hp.get('skip_first_weeks', 0))

        model = _build_model(hp, n_sensor, n_temporal)
        weights_path = RESULTS_DIR / f'best_cnn_rnn_inv{inv_id}.pt'
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # Test T17 con el modelo cargado del .pt
        y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)

        # Val T16 → calibra intervalos CPTC al 90%
        y_val_true, y_val_pred = evaluate_loader(model, val_loader, scaler_y, bc_lambda)
        pi_lower, pi_upper, val_cov, pi_width = compute_cptc_intervals(
            y_val_true, y_val_pred, y_pred)

        elapsed = time.time() - t0
        print(f'[engine] inv{inv_id} listo en {elapsed:.1f}s — '
              f'R²={metrics["R²"]:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')

        return {
            'inv_id': inv_id,
            'weights_path': str(weights_path),
            'week_keys': np.asarray(week_keys),
            'y_true': np.asarray(y_true, dtype=float),
            'y_pred': np.asarray(y_pred, dtype=float),
            'pi_lower': np.asarray(pi_lower, dtype=float),
            'pi_upper': np.asarray(pi_upper, dtype=float),
            'metrics': metrics,
            'pi_coverage': float(val_cov),
            'pi_avg_width': float(pi_width),
            'hp': hp,
            'elapsed_s': elapsed,
        }

    def get(self, inv_id: int, force: bool = False) -> dict:
        if force or inv_id not in self._cache:
            self._cache[inv_id] = self._run(inv_id)
        return self._cache[inv_id]

    def warm(self, inv_ids=(3, 4)) -> None:
        for i in inv_ids:
            try:
                self.get(i)
            except Exception as e:  # noqa: BLE001
                print(f'[engine] inv{i} FALLÓ: {e}')

    def loaded(self) -> list[int]:
        return sorted(self._cache.keys())

    # ── serialización ────────────────────────────────────────────────────
    def payload(self, inv_id: int) -> dict:
        """Arrays completos T17 (forma que consume el dashboard)."""
        s = self.get(inv_id)
        n = int(len(s['week_keys']))
        yp = s['y_pred']
        return {
            'inv_id': inv_id,
            'horizon_weeks': 6,
            'n_weeks': n,
            'weights_path': s['weights_path'],
            'weeks': [_week_key_to_iso(int(w)) for w in s['week_keys']],
            'y_true': s['y_true'].tolist(),
            'y_pred': yp.tolist(),
            'ensemble_pred': yp.tolist(),  # demo en vivo usa el best (sin ensemble)
            'pi_lower': s['pi_lower'].tolist(),
            'pi_upper': s['pi_upper'].tolist(),
        }

    def metrics(self, inv_id: int) -> dict:
        s = self.get(inv_id)
        m = {k: (None if v is None else float(v)) for k, v in s['metrics'].items()}
        return {
            'inv_id': inv_id,
            'source': 'live_inference',  # recalculado desde el .pt, no del .npz
            'best': m,
            'cptc_pi': {
                'pi_coverage': s['pi_coverage'],
                'pi_avg_width': s['pi_avg_width'],
            },
        }

    def window(self, inv_id: int, cursor: int | None = None,
               horizon: int = 6, past: int = 5) -> dict:
        """Recorta la inferencia T17 alrededor de `cursor` (semana elegida).

        history     = `past` semanas observadas previas al cursor (real + pred)
        predictions = `horizon` semanas desde el cursor (pred + real revelado)
        """
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
                'real': s['y_true'][i] if observed else None,
                'pred': float(s['y_pred'][i]),
                'lo': float(s['pi_lower'][i]),
                'hi': float(s['pi_upper'][i]),
            }

        history = [row(i, observed=True) for i in range(start_past, cursor)]
        predictions = []
        for i in range(cursor, min(cursor + horizon, n)):
            r = row(i, observed=False)
            r['horizon'] = i - cursor + 1
            r['real'] = float(s['y_true'][i])  # disponible (test set) para evaluar el pronóstico
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
