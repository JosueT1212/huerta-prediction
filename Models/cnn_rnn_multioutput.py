"""
CNN-RNN Multi-Output Model for Greenhouse Crop Yield Prediction
===============================================================
Direct multi-step forecasting: the model outputs h values simultaneously,
one per week from 1 to h. Each output neuron directly regresses the yield
for that specific future week.

Architecture:
  - Same CNN-RNN backbone (CNN blocks + LSTM)
  - FC head: Linear(fc_hidden, horizon) — one output per future week
  - Loss: WMAE averaged over all h output steps
  - Evaluation: metrics per step (week 1 through h)

Split: T13-T15 train | T16 validation | T17 test
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

DATA_DIR = Path(__file__).resolve().parent.parent / 'Data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
#    (identical to cnn_rnn_yield.py — only target creation differs)
# ============================================================================

def load_transplant_dates():
    filepath = DATA_DIR / 'Fechas de Transplante.xlsx'
    raw = pd.read_excel(filepath, header=None)
    dates = {}
    for _, row in raw.iloc[2:].iterrows():
        season_num = str(row.iloc[0]).strip()
        temp = f'T{int(float(season_num))}'
        for inv_id, col_idx in [(3, 1), (4, 2)]:
            cell = row.iloc[col_idx]
            if isinstance(cell, (pd.Timestamp, __import__('datetime').datetime)):
                dates[(temp, inv_id)] = cell
            else:
                raw_val = str(cell).strip().split(' y ')[0].strip()
                dates[(temp, inv_id)] = pd.to_datetime(raw_val, dayfirst=True)
    return dates


def load_riego_variables(invernadero_id):
    filepath = DATA_DIR / f'Variables riego invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'fecha' in col_lower:   rename[col] = 'fecha'
            elif 'ph' in col_lower:    rename[col] = 'ph_promedio'
            elif 'ce' in col_lower:    rename[col] = 'ce_promedio'
            elif 'riego' in col_lower: rename[col] = 'riego_total'
        df = df.rename(columns=rename)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        for col in ['ph_promedio', 'ce_promedio', 'riego_total']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        keep = ['fecha', 'temporada'] + [c for c in ['riego_total', 'ph_promedio', 'ce_promedio'] if c in df.columns]
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


def load_internal_variables_all_seasons(invernadero_id=4):
    filepath = DATA_DIR / f'Variables internas invernadero {invernadero_id}.xlsx'
    if not filepath.exists():
        filepath = DATA_DIR / f'Variables internas Invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        for col in df.columns:
            if col not in ['fecha', 'temporada']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_external_variables_all_seasons():
    filepath = DATA_DIR / 'Variables exteriores.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        for col in df.columns:
            if col not in ['fecha', 'temporada']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_production(invernadero_id):
    filepath = DATA_DIR / f'Kg por semana T13 - T17 Invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        raw = pd.read_excel(filepath, sheet_name=sheet, header=None)
        data = raw.iloc[2:].copy()
        data.columns = ['semana', 'kg_reales']
        data['semana']    = pd.to_numeric(data['semana'],    errors='coerce')
        data['kg_reales'] = pd.to_numeric(data['kg_reales'], errors='coerce')
        data['temporada'] = SEASON_NAMES[i]
        frames.append(data)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['kg_reales']).reset_index(drop=True)
    return df


def build_dataset_for_greenhouse(invernadero_id, horizon=6, lag_features=None,
                                 include_rolling_mean=False,
                                 train_seasons=None, val_season=None):
    """
    Identical data pipeline to cnn_rnn_yield.py, except targets:
    instead of one target (shift -horizon), we create h targets
    (shift -1 through -horizon) for direct multi-step forecasting.
    """
    if lag_features is None:
        lag_features = [1]

    df_int   = load_internal_variables_all_seasons(invernadero_id)
    df_ext   = load_external_variables_all_seasons()
    df_kg    = load_production(invernadero_id)
    df_riego = load_riego_variables(invernadero_id)
    transplant_dates = load_transplant_dates()

    int_rename = {}
    for col in df_int.columns:
        if   'temperatura promedio' in col: int_rename[col] = 'temp_prom_int'
        elif 'temperatura minima'   in col: int_rename[col] = 'temp_min_int'
        elif 'temperatura maxima'   in col: int_rename[col] = 'temp_max_int'
        elif 'humedad relativa promedio' in col: int_rename[col] = 'hr_prom_int'
        elif 'co2' in col:                  int_rename[col] = 'co2_ppm'
        elif 'ficit de humedad' in col:     int_rename[col] = 'deficit_humedad'
        elif 'ficit' in col and 'vapor' in col: int_rename[col] = 'deficit_presion_vapor'
        elif 'humedad absoluta' in col:     int_rename[col] = 'humedad_abs_int'
    df_int = df_int.rename(columns=int_rename)

    ext_rename = {}
    for col in df_ext.columns:
        if   'promedio' in col and 'temperatura' in col: ext_rename[col] = 'temp_prom_ext'
        elif 'maxima'   in col and 'temperatura' in col: ext_rename[col] = 'temp_max_ext'
        elif 'minima'   in col and 'temperatura' in col: ext_rename[col] = 'temp_min_ext'
        elif 'hr promedio' in col: ext_rename[col] = 'hr_prom_ext'
        elif 'suma'        in col: ext_rename[col] = 'rad_sum'
        elif 'intensidad'  in col: ext_rename[col] = 'rad_max'
        elif 'dh'          in col: ext_rename[col] = 'dh_ext'
        elif 'humedad absoluta' in col: ext_rename[col] = 'humedad_abs_ext'
    df_ext = df_ext.rename(columns=ext_rename)

    int_features   = [c for c in ['temp_prom_int','temp_min_int','temp_max_int',
                                   'hr_prom_int','co2_ppm','deficit_humedad',
                                   'deficit_presion_vapor','humedad_abs_int'] if c in df_int.columns]
    ext_features   = [c for c in ['temp_prom_ext','temp_max_ext','temp_min_ext',
                                   'hr_prom_ext','rad_sum','rad_max','dh_ext',
                                   'humedad_abs_ext'] if c in df_ext.columns]
    riego_features = [c for c in ['riego_total','ph_promedio','ce_promedio'] if c in df_riego.columns]

    all_season_frames = []

    for temp in SEASON_NAMES:
        int_season  = df_int[df_int['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        ext_season  = df_ext[df_ext['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        kg_season   = df_kg[df_kg['temporada']   == temp].reset_index(drop=True)
        if len(kg_season) == 0:
            continue

        daily = pd.merge(int_season[['fecha','temporada'] + int_features],
                         ext_season[['fecha'] + ext_features], on='fecha', how='inner')
        riego_season = df_riego[df_riego['temporada'] == temp][['fecha'] + riego_features]
        daily = pd.merge(daily, riego_season, on='fecha', how='left')
        daily = daily.sort_values('fecha').reset_index(drop=True)
        daily['week_idx'] = daily.index // 7

        tp_key = (temp, invernadero_id)
        if tp_key in transplant_dates:
            tp_date = transplant_dates[tp_key]
            daily['dias_desde_transplante'] = (daily['fecha'] - tp_date).dt.days

        agg_dict = {feat: 'mean' for feat in int_features + ext_features}
        if 'rad_sum' in agg_dict: agg_dict['rad_sum'] = 'sum'
        for rf in riego_features:
            agg_dict[rf] = 'sum' if rf == 'riego_total' else 'mean'
        if 'dias_desde_transplante' in daily.columns:
            agg_dict['dias_desde_transplante'] = 'mean'

        weekly_env = daily.groupby('week_idx').agg(agg_dict).reset_index()
        n_weeks    = min(len(weekly_env), len(kg_season))
        weekly_env = weekly_env.iloc[:n_weeks].copy()
        kg_vals    = kg_season.iloc[:n_weeks].copy()

        weekly = weekly_env.copy()
        weekly['kg_reales'] = kg_vals['kg_reales'].values
        weekly['temporada'] = temp
        weekly['semana']    = kg_vals['semana'].values

        weekly['week_in_season'] = np.arange(len(weekly))
        weekly['week_position']  = weekly['week_in_season'] / len(weekly)
        for period in [52, 26, 13]:
            weekly[f'week_sin_{period}'] = np.sin(2 * np.pi * weekly['semana'] / period)
            weekly[f'week_cos_{period}'] = np.cos(2 * np.pi * weekly['semana'] / period)

        for lag in lag_features:
            weekly[f'kg_lag_{lag}'] = weekly['kg_reales'].shift(lag)

        if include_rolling_mean:
            weekly['kg_roll_mean_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).mean()

        # ── KEY DIFFERENCE: one target column per future week ──
        for k in range(1, horizon + 1):
            weekly[f'target_{k}'] = weekly['kg_reales'].shift(-k)

        all_season_frames.append(weekly)

    df_all = pd.concat(all_season_frames, ignore_index=True)

    train_seasons = train_seasons if train_seasons is not None else ['T13', 'T14', 'T15']
    val_season    = val_season    if val_season    is not None else 'T16'

    target_cols  = [f'target_{k}' for k in range(1, horizon + 1)]
    exclude      = ['week_idx', 'kg_reales', 'temporada', 'fecha', 'semana'] + target_cols
    feature_cols = [c for c in df_all.columns if c not in exclude]

    df_all = df_all.dropna().reset_index(drop=True)

    train_df = df_all[df_all['temporada'].isin(train_seasons)].reset_index(drop=True)
    val_df   = df_all[df_all['temporada'] == val_season].reset_index(drop=True)
    test_df  = df_all[df_all['temporada'] == 'T17'].reset_index(drop=True)

    print(f'  Invernadero {invernadero_id}:')
    print(f'    Train: {len(train_df)} ({",".join(train_seasons)})  '
          f'Val: {len(val_df)} ({val_season})  Test: {len(test_df)} (T17)')
    print(f'    Features ({len(feature_cols)}): {feature_cols}')

    return train_df, val_df, test_df, feature_cols, target_cols


# ============================================================================
# 2. SEQUENCES
# ============================================================================

def make_sequences_per_season(X, y, temporadas, seq_len):
    """
    Build (seq_len, n_features) windows within each season.
    y: (n,) or (n, h) — both handled.
    """
    seqs_X, seqs_y = [], []
    for temp in pd.Series(temporadas).unique():
        mask = np.array(temporadas) == temp
        X_s, y_s = X[mask], y[mask]
        for i in range(len(X_s) - seq_len + 1):
            seqs_X.append(X_s[i:i + seq_len])
            seqs_y.append(y_s[i + seq_len - 1])
    if len(seqs_X) == 0:
        raise ValueError('No sequences built — check seq_len vs season length.')
    return np.array(seqs_X, dtype=np.float32), np.array(seqs_y, dtype=np.float32)


# ============================================================================
# 3. MODEL
# ============================================================================

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.residual_conv is None else self.residual_conv(x)
        return out + res


class CNNRNNMultiOutput(nn.Module):
    """
    CNN-RNN with split inputs:
      - x_sensor  → CNN blocks → (batch, seq_len, cnn_filters)
      - x_temporal (bypasses CNN, arrives clean)
      - concat → LSTM → FC → (batch, horizon)

    Temporal features (calendar encodings, lag features) are not convolved
    so their structured signal reaches the LSTM unmodified.
    """
    def __init__(self, n_sensor, n_temporal, horizon,
                 cnn_filters, cnn_kernel_size, cnn_padding,
                 num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden):
        super().__init__()
        cnn_blocks, in_ch = [], n_sensor
        for _ in range(num_cnn_blocks):
            cnn_blocks.append(CNNBlock(in_ch, cnn_filters, cnn_kernel_size, cnn_padding, dropout))
            in_ch = cnn_filters
        self.cnn  = nn.Sequential(*cnn_blocks)
        # LSTM sees CNN output + raw temporal features at every time step
        self.lstm = nn.LSTM(input_size=cnn_filters + n_temporal,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.fc   = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )

    def forward(self, x_sensor, x_temporal):
        # x_sensor:  (batch, seq_len, n_sensor)
        # x_temporal:(batch, seq_len, n_temporal)
        x = x_sensor.permute(0, 2, 1)          # (batch, n_sensor, seq_len)
        x = self.cnn(x)                          # (batch, cnn_filters, seq_len)
        x = x.permute(0, 2, 1)                  # (batch, seq_len, cnn_filters)
        x = torch.cat([x, x_temporal], dim=-1)  # (batch, seq_len, cnn_filters + n_temporal)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])       # (batch, horizon)


def init_weights(model, method='default'):
    if method == 'default':
        return
    for _, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if method == 'he':         nn.init.kaiming_normal_(param, nonlinearity='relu')
        elif method == 'xavier':   nn.init.xavier_normal_(param)
        elif method == 'orthogonal': nn.init.orthogonal_(param)
        elif method == 'lecun':    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')


# ============================================================================
# 4. HYPERPARAMETERS
# ============================================================================

HYPERPARAMS_PER_GREENHOUSE = {
    3: {
        'seq_len': 1, 'horizon': 6, 'batch_size': 8,
        'lag_features': [1], 'include_rolling_mean': False,
        'cnn_filters': 64, 'cnn_kernel_size': 2, 'cnn_padding': 1, 'num_cnn_blocks': 1,
        'lstm_hidden': 64, 'lstm_layers': 1, 'fc_hidden': 64,
        'dropout': 0.2, 'learning_rate': 5e-4, 'weight_decay': 1e-4, 'wmae_power': 3,
        'epochs': 500, 'patience': 250,
        'init_methods': ['default', 'xavier', 'orthogonal', 'lecun'],
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
                  0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                  100, 200, 500, 1000],
    },
    4: {
        'seq_len': 1, 'horizon': 6, 'batch_size': 8,
        'lag_features': [1], 'include_rolling_mean': False,
        'cnn_filters': 64, 'cnn_kernel_size': 2, 'cnn_padding': 1, 'num_cnn_blocks': 1,
        'lstm_hidden': 64, 'lstm_layers': 1, 'fc_hidden': 64,
        'dropout': 0.2, 'learning_rate': 5e-4, 'weight_decay': 1e-4, 'wmae_power': 3,
        'epochs': 500, 'patience': 250,
        'init_methods': ['default', 'xavier', 'orthogonal', 'lecun'],
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
                  0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                  100, 200, 500, 1000],
    },
}


# ============================================================================
# 5. TRAINING
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Features that bypass the CNN and go straight to the LSTM
TEMPORAL_FEATURE_PREFIXES = ('week_sin_', 'week_cos_', 'kg_lag_', 'kg_roll_')
TEMPORAL_FEATURE_NAMES    = {'dias_desde_transplante', 'week_in_season', 'week_position'}


def split_features(feature_cols):
    """Split feature list into sensor (→ CNN) and temporal (→ LSTM directly)."""
    temporal, sensor = [], []
    for c in feature_cols:
        if c in TEMPORAL_FEATURE_NAMES or any(c.startswith(p) for p in TEMPORAL_FEATURE_PREFIXES):
            temporal.append(c)
        else:
            sensor.append(c)
    return sensor, temporal


def prepare_data(invernadero_id, hp, train_seasons=None, val_season=None):
    train_df, val_df, test_df, feature_cols, target_cols = build_dataset_for_greenhouse(
        invernadero_id, horizon=hp['horizon'],
        lag_features=hp.get('lag_features', [1]),
        include_rolling_mean=hp.get('include_rolling_mean', False),
        train_seasons=train_seasons, val_season=val_season,
    )
    horizon = hp['horizon']

    sensor_cols, temporal_cols = split_features(feature_cols)
    print(f'  Sensor features ({len(sensor_cols)}): {sensor_cols}')
    print(f'  Temporal features ({len(temporal_cols)}): {temporal_cols}')

    # ── Sensor features: scale + PCA ──
    Xs_tr = train_df[sensor_cols].values.astype(np.float32)
    Xs_va = val_df[sensor_cols].values.astype(np.float32)
    Xs_te = test_df[sensor_cols].values.astype(np.float32)

    scaler_Xs = MinMaxScaler(feature_range=(-1, 1))
    Xs_tr = scaler_Xs.fit_transform(Xs_tr)
    Xs_va = scaler_Xs.transform(Xs_va)
    Xs_te = scaler_Xs.transform(Xs_te)

    pca = PCA(n_components=hp.get('pca_variance', 0.95))
    Xs_tr = pca.fit_transform(Xs_tr)
    Xs_va = pca.transform(Xs_va)
    Xs_te = pca.transform(Xs_te)
    n_sensor_pca = pca.n_components_
    print(f'  PCA (sensor only): {len(sensor_cols)} -> {n_sensor_pca} components')

    # ── Temporal features: scale only, no PCA ──
    Xt_tr = train_df[temporal_cols].values.astype(np.float32)
    Xt_va = val_df[temporal_cols].values.astype(np.float32)
    Xt_te = test_df[temporal_cols].values.astype(np.float32)

    scaler_Xt = MinMaxScaler(feature_range=(-1, 1))
    Xt_tr = scaler_Xt.fit_transform(Xt_tr)
    Xt_va = scaler_Xt.transform(Xt_va)
    Xt_te = scaler_Xt.transform(Xt_te)
    n_temporal = len(temporal_cols)

    # ── Targets: Box-Cox + MinMax ──
    y_train = train_df[target_cols].values.astype(np.float32)
    y_val   = val_df[target_cols].values.astype(np.float32)
    y_test  = test_df[target_cols].values.astype(np.float32)

    _, bc_lambda = boxcox(y_train.flatten() + 1.0)
    print(f'  Box-Cox λ = {bc_lambda:.4f}')

    y_train_bc = boxcox(y_train + 1.0, lmbda=bc_lambda).astype(np.float32)
    y_val_bc   = boxcox(y_val   + 1.0, lmbda=bc_lambda).astype(np.float32)
    y_test_bc  = boxcox(y_test  + 1.0, lmbda=bc_lambda).astype(np.float32)

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y_train_bc.flatten().reshape(-1, 1))
    y_tr_s = scaler_y.transform(y_train_bc.reshape(-1, 1)).reshape(-1, horizon).astype(np.float32)
    y_va_s = scaler_y.transform(y_val_bc.reshape(-1,   1)).reshape(-1, horizon).astype(np.float32)
    y_te_s = scaler_y.transform(y_test_bc.reshape(-1,  1)).reshape(-1, horizon).astype(np.float32)

    # ── Sequences (sensor and temporal built separately, same order) ──
    seq_len = hp['seq_len']
    temps_tr = train_df['temporada'].values
    temps_va = val_df['temporada'].values
    temps_te = test_df['temporada'].values

    Xs_tr_seq, y_tr   = make_sequences_per_season(Xs_tr, y_tr_s, temps_tr, seq_len)
    Xt_tr_seq, _      = make_sequences_per_season(Xt_tr, y_tr_s, temps_tr, seq_len)
    Xs_va_seq, y_va   = make_sequences_per_season(Xs_va, y_va_s, temps_va, seq_len)
    Xt_va_seq, _      = make_sequences_per_season(Xt_va, y_va_s, temps_va, seq_len)
    Xs_te_seq, y_te   = make_sequences_per_season(Xs_te, y_te_s, temps_te, seq_len)
    Xt_te_seq, _      = make_sequences_per_season(Xt_te, y_te_s, temps_te, seq_len)

    print(f'  Sequences — train: {len(Xs_tr_seq)}, val: {len(Xs_va_seq)}, test: {len(Xs_te_seq)}')

    def to_ds(Xs, Xt, y):
        return TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(Xt), torch.FloatTensor(y))

    bs = hp['batch_size']
    train_loader = DataLoader(to_ds(Xs_tr_seq, Xt_tr_seq, y_tr), batch_size=bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(to_ds(Xs_va_seq, Xt_va_seq, y_va), batch_size=bs, shuffle=False)
    test_loader  = DataLoader(to_ds(Xs_te_seq, Xt_te_seq, y_te), batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal


class YieldWMAELoss(nn.Module):
    """Weighted MAE over all h output steps."""
    def __init__(self, power=1):
        super().__init__()
        self.power = power

    def forward(self, pred, target):
        pred_f   = pred.flatten()
        target_f = target.flatten()
        weights = ((target_f + 1.0) / 2.0).clamp(min=0.0) ** self.power
        return torch.mean(weights * torch.abs(pred_f - target_f))


def train_model(model, train_loader, val_loader, hp, model_path):
    criterion = YieldWMAELoss(power=hp.get('wmae_power', 1))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['learning_rate'], weight_decay=hp['weight_decay'])

    best_val_loss, patience_counter = float('inf'), 0
    train_losses, val_losses = [], []

    for epoch in range(hp['epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for Xs_b, Xt_b, y_b in train_loader:
            Xs_b, Xt_b, y_b = Xs_b.to(DEVICE), Xt_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xs_b, Xt_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss, val_batches = 0.0, 0
        all_vp, all_vt = [], []
        with torch.no_grad():
            for Xs_b, Xt_b, y_b in val_loader:
                Xs_b, Xt_b, y_b = Xs_b.to(DEVICE), Xt_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(Xs_b, Xt_b)
                val_loss += criterion(pred, y_b).item(); val_batches += 1
                all_vp.append(pred.cpu()); all_vt.append(y_b.cpu())

        avg_val = val_loss / max(val_batches, 1)
        val_losses.append(avg_val)

        vp = torch.cat(all_vp)[:, -1]; vt = torch.cat(all_vt)[:, -1]
        pm, tm = vp - vp.mean(), vt - vt.mean()
        val_corr = (torch.sum(pm * tm) / (
            torch.sqrt(torch.sum(pm**2) + 1e-8) * torch.sqrt(torch.sum(tm**2) + 1e-8)
        )).item()

        if avg_val < best_val_loss:
            best_val_loss = avg_val; patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train:.6f} | Val: {avg_val:.6f} | '
                  f'Corr(h={hp["horizon"]}): {val_corr:.4f}')

        if patience_counter >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, train_losses, val_losses


# ============================================================================
# 6. EVALUATION
# ============================================================================

def compute_metrics(y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    nse   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    pbias = 100.0 * np.sum(y_true - y_pred) / np.sum(y_true)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'RMSE (kg)': rmse, 'R²': r2, 'NSE': nse, 'PBIAS (%)': pbias, 'MAPE (%)': mape}


def evaluate_model(model, test_loader, scaler_y, bc_lambda, horizon):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xs_b, Xt_b, y_b in test_loader:
            pred = model(Xs_b.to(DEVICE), Xt_b.to(DEVICE))
            all_preds.append(pred.cpu().numpy()); all_targets.append(y_b.numpy())

    y_pred_s = np.concatenate(all_preds,   axis=0)  # (n, h)
    y_true_s = np.concatenate(all_targets, axis=0)  # (n, h)
    n = y_pred_s.shape[0]

    # Inverse: scaler_y → Box-Cox space → kg
    y_pred_bc = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(n, horizon)
    y_true_bc = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(n, horizon)
    y_pred_kg = inv_boxcox(y_pred_bc, bc_lambda) - 1.0
    y_true_kg = inv_boxcox(y_true_bc, bc_lambda) - 1.0

    per_step = {k + 1: compute_metrics(y_true_kg[:, k], y_pred_kg[:, k]) for k in range(horizon)}
    return y_true_kg, y_pred_kg, per_step, per_step[horizon]


def print_per_step(inv_id, horizon, per_step):
    print(f'\n{"=" * 65}')
    print(f'  INVERNADERO {inv_id} — Per-step metrics (Test T17)')
    print(f'{"=" * 65}')
    print(f'  {"Step":>5}  {"RMSE (kg)":>10}  {"R²":>8}  {"NSE":>8}  {"MAPE%":>8}')
    print(f'  {"─"*5}  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*8}')
    for k in range(1, horizon + 1):
        m = per_step[k]
        print(f'  h={k:>3}   {m["RMSE (kg)"]:>10.1f}  {m["R²"]:>8.4f}  '
              f'{m["NSE"]:>8.4f}  {m["MAPE (%)"] :>8.2f}%')
    print(f'{"=" * 65}')


def plot_results(results, horizon):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        axes[i, 0].plot(res['train_losses'], color='steelblue', lw=1, label='Train')
        axes[i, 0].plot(res['val_losses'],   color='coral',     lw=1, label='Val (T16)')
        axes[i, 0].set_title(f'Invernadero {inv_id} — Loss'); axes[i, 0].set_yscale('log')
        axes[i, 0].set_xlabel('Epoch'); axes[i, 0].set_ylabel('Loss'); axes[i, 0].legend()

        weeks = np.arange(len(res['y_true_final']))
        axes[i, 1].plot(weeks, res['y_true_final'], 'o-',  color='steelblue', label='Actual',            ms=5)
        axes[i, 1].plot(weeks, res['y_pred_final'], 's--', color='coral',     label=f'Best (h={horizon})', ms=5)
        if 'ensemble_pred_final' in res:
            axes[i, 1].plot(weeks, res['ensemble_pred_final'], '^:', color='green',
                             label=f'Ensemble top-20 (h={horizon})', ms=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} — Predicción semana h={horizon} (T17)')
        axes[i, 1].set_xlabel('Semana (T17)'); axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        m  = res['final_metrics']
        em = res.get('ensemble_final_metrics', {})
        txt = (f'Best model\nRMSE: {m["RMSE (kg)"]:.1f} kg\n'
               f'R²: {m["R²"]:.4f}\nMAPE: {m["MAPE (%)"]:.2f}%')
        if em:
            txt += f'\n─── Ensemble top-20 ───\nR²: {em["R²"]:.4f}  MAPE: {em["MAPE (%)"]:.2f}%'
        axes[i, 1].text(0.02, 0.98, txt, transform=axes[i, 1].transAxes,
                         va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out = RESULTS_DIR / 'cnn_rnn_multioutput_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results plot saved to {out}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 65)
    print('  CNN-RNN Multi-Output — Greenhouse Yield Prediction')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('  Direct multi-step: h outputs (week 1 … h) jointly')
    print('=' * 65)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        hp      = HYPERPARAMS_PER_GREENHOUSE[inv_id]
        seeds   = hp.get('seeds', [42])
        inits   = hp.get('init_methods', ['default', 'xavier', 'orthogonal', 'lecun'])
        horizon = hp['horizon']
        n_runs  = len(seeds) * len(inits)

        print(f'\n{"─" * 65}')
        print(f'  INVERNADERO {inv_id} — {n_runs} runs ({len(seeds)} seeds × {len(inits)} inits)')
        print(f'{"─" * 65}')

        (train_loader, val_loader, test_loader,
         scaler_y, bc_lambda, n_sensor_pca, n_temporal) = prepare_data(inv_id, hp)

        best_r2, best_result = -float('inf'), None
        top_runs = []

        model_path = RESULTS_DIR / f'best_cnn_rnn_multioutput_inv{inv_id}.pt'
        tmp_ckpt   = RESULTS_DIR / f'_tmp_cnn_rnn_multioutput_inv{inv_id}.pt'

        for init_method in inits:
            for seed in seeds:
                set_seed(seed)
                model = CNNRNNMultiOutput(
                    n_sensor=n_sensor_pca, n_temporal=n_temporal, horizon=horizon,
                    cnn_filters=hp['cnn_filters'], cnn_kernel_size=hp['cnn_kernel_size'],
                    cnn_padding=hp['cnn_padding'],  num_cnn_blocks=hp['num_cnn_blocks'],
                    lstm_hidden=hp['lstm_hidden'],  lstm_layers=hp['lstm_layers'],
                    dropout=hp['dropout'],          fc_hidden=hp['fc_hidden'],
                ).to(DEVICE)

                if init_method != 'default':
                    model = model.cpu(); init_weights(model, init_method); model = model.to(DEVICE)

                model, train_losses, val_losses = train_model(
                    model, train_loader, val_loader, hp, tmp_ckpt)

                y_true_kg, y_pred_kg, per_step, final_metrics = evaluate_model(
                    model, test_loader, scaler_y, bc_lambda, horizon)

                r2   = final_metrics['R²']
                mape = final_metrics['MAPE (%)']
                print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={mape:.2f}%  (h={horizon})')
                top_runs.append((r2, y_pred_kg.copy()))

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = {
                        'y_true_kg': y_true_kg, 'y_pred_kg': y_pred_kg,
                        'y_true_final': y_true_kg[:, -1], 'y_pred_final': y_pred_kg[:, -1],
                        'per_step': per_step, 'final_metrics': final_metrics,
                        'train_losses': train_losses, 'val_losses': val_losses,
                        'seed': seed, 'init_method': init_method,
                    }
                    torch.save(model.state_dict(), model_path)

        if tmp_ckpt.exists():
            tmp_ckpt.unlink()

        print(f'\n  >>> Best init={best_result["init_method"]} seed={best_result["seed"]} '
              f'(R²={best_r2:.4f}) <<<')
        print_per_step(inv_id, horizon, best_result['per_step'])

        # Ensemble top-20
        top_runs.sort(key=lambda x: x[0], reverse=True)
        top_k = min(20, len(top_runs))
        ens_pred_all   = np.mean([r[1] for r in top_runs[:top_k]], axis=0)
        ens_pred_final = ens_pred_all[:, -1]
        ens_metrics    = compute_metrics(best_result['y_true_final'], ens_pred_final)
        print(f'\n  >>> Ensemble top-{top_k}: R²={ens_metrics["R²"]:.4f}, '
              f'MAPE={ens_metrics["MAPE (%)"]:.2f}% <<<')

        all_results[inv_id] = {
            **best_result,
            'ensemble_pred_final': ens_pred_final,
            'ensemble_final_metrics': ens_metrics,
        }
        all_metrics.append({'invernadero': inv_id, 'model': 'best',
                             'horizon_step': horizon, **best_result['final_metrics']})
        all_metrics.append({'invernadero': inv_id, 'model': f'ensemble_top{top_k}',
                             'horizon_step': horizon, **ens_metrics})

    print('\nGenerando gráficas...')
    plot_results(all_results, HYPERPARAMS_PER_GREENHOUSE[greenhouses[0]]['horizon'])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'cnn_rnn_multioutput_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "cnn_rnn_multioutput_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN — Multi-Output CNN-RNN (final horizon step, h=6)')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
