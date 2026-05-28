"""Smoke test: prepare_data with return_arrays=True returns numpy arrays, not DataLoaders."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import yaml
from pathlib import Path
from cnn_rnn_yield import prepare_data

hp_path = Path(__file__).parent.parent / 'hp_inv3.yaml'
with open(hp_path) as f:
    HP = yaml.safe_load(f)

result = prepare_data(
    3, HP,
    train_seasons=['T13', 'T14', 'T15'], val_season='T16',
    skip_first_weeks=HP.get('skip_first_weeks', 3),
    return_arrays=True
)
assert len(result) == 12, f'Expected 12 return values, got {len(result)}'
Xs_tr_seq, Xt_tr_seq, y_tr, Xs_va_seq, Xt_va_seq, y_va, Xs_te_seq, Xt_te_seq, y_te, scaler_y, bc_lambda, week_keys = result
assert isinstance(Xs_tr_seq, np.ndarray), 'Xs_tr_seq must be ndarray'
assert Xs_tr_seq.ndim == 3, f'Expected 3D array, got {Xs_tr_seq.ndim}D'
assert len(y_tr) == len(Xs_tr_seq), 'y_tr and Xs_tr_seq must have same length'
assert len(week_keys) == len(y_te), f'week_keys/y_te mismatch: {len(week_keys)} vs {len(y_te)}'
print(f'PASS: shapes train={Xs_tr_seq.shape}, val={Xs_va_seq.shape}, test={Xs_te_seq.shape}, week_keys={len(week_keys)}')
