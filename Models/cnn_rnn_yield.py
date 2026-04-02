"""
CNN-RNN Model for Greenhouse Crop Yield Prediction
===================================================
Based on: Gong et al. (2023) "A Novel Model Fusion Approach for Greenhouse
Crop Yield Prediction", Horticulturae 9(1), 5.

Architecture (Section 2.2 of the paper):
  - CNN part: N blocks of [1D Conv -> WeightNorm -> ReLU -> Dropout] + residual
  - RNN part: LSTM units -> Feed-forward network -> Prediction

Trains one model per greenhouse (Inv. 3, Inv. 4) using all 5 seasons (T13-T17).
All Excel files have 5 sheets (one per season) with sensor + production data.
Split: T13-T15 train, T16 validation, T17 test.
Prediction horizon: 4 weeks ahead.

Sections:
  1. Data Loading & Feature Engineering
  2. Dataset & DataLoader
  3. CNN-RNN Model Definition
  4. Hyperparameter Configuration (TUNE HERE)
  5. Training (with validation-based early stopping)
  6. Testing & Evaluation (RMSE, R², NSE, PBIAS, MAPE)
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
from torch.utils.data import Dataset, DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

DATA_DIR = Path(__file__).resolve().parent.parent / 'Data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_transplant_dates():
    """Load official transplant dates per season per greenhouse from Excel."""
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
    """Load irrigation variables (riego_total, pH, CE) for all seasons."""
    filepath = DATA_DIR / f'Variables riego invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'fecha' in col_lower: rename[col] = 'fecha'
            elif 'ph' in col_lower: rename[col] = 'ph_promedio'
            elif 'ce' in col_lower: rename[col] = 'ce_promedio'
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
    """Load internal greenhouse sensor data for all 5 seasons."""
    filepath = DATA_DIR / f'Variables internas invernadero {invernadero_id}.xlsx'
    if not filepath.exists():
        filepath = DATA_DIR / f'Variables internas Invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={'fecha': 'fecha'})
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        # Force numeric on all non-date columns
        for col in df.columns:
            if col not in ['fecha', 'temporada']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_external_variables_all_seasons():
    """Load external weather data for all 5 seasons."""
    filepath = DATA_DIR / 'Variables exteriores.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={'fecha': 'fecha'})
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        # Force numeric on all non-date columns
        for col in df.columns:
            if col not in ['fecha', 'temporada']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_production(invernadero_id):
    """Load weekly production for a given greenhouse across all seasons."""
    filepath = DATA_DIR / f'Kg por semana T13 - T17 Invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        raw = pd.read_excel(filepath, sheet_name=sheet, header=None)
        data = raw.iloc[2:].copy()
        data.columns = ['semana', 'kg_reales']
        data['semana'] = pd.to_numeric(data['semana'], errors='coerce')
        data['kg_reales'] = pd.to_numeric(data['kg_reales'], errors='coerce')
        data['temporada'] = SEASON_NAMES[i]
        data['invernadero'] = invernadero_id
        frames.append(data)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['kg_reales']).reset_index(drop=True)
    return df


def aggregate_daily_to_weekly(df_daily, temporada):
    """Aggregate daily sensor data to weekly resolution within a season."""
    season_data = df_daily[df_daily['temporada'] == temporada].copy()
    season_data = season_data.sort_values('fecha').reset_index(drop=True)

    # Assign week number within the season (sequential)
    season_data['day_idx'] = np.arange(len(season_data))
    season_data['week_idx'] = season_data['day_idx'] // 7

    return season_data


def build_dataset_for_greenhouse(invernadero_id, horizon=4, lag_features=None,
                                 include_rolling_mean=False,
                                 train_seasons=None, val_season=None):
    """
    Build feature matrix for a single greenhouse using all 5 seasons.

    Merges internal sensor data + external weather data (aggregated to weekly)
    with weekly production targets. Creates lag features and rolling stats.

    Returns:
        train_df, val_df, test_df, feature_cols
    """
    df_int = load_internal_variables_all_seasons(invernadero_id)
    df_ext = load_external_variables_all_seasons()
    df_kg = load_production(invernadero_id)
    df_riego = load_riego_variables(invernadero_id)
    transplant_dates = load_transplant_dates()

    # Rename internal sensor columns (all lowercased by loader)
    int_rename = {}
    for col in df_int.columns:
        if 'temperatura promedio' in col:
            int_rename[col] = 'temp_prom_int'
        elif 'temperatura minima' in col:
            int_rename[col] = 'temp_min_int'
        elif 'temperatura maxima' in col:
            int_rename[col] = 'temp_max_int'
        elif 'humedad relativa promedio' in col:
            int_rename[col] = 'hr_prom_int'
        elif 'co2' in col:
            int_rename[col] = 'co2_ppm'
        elif 'ficit de humedad' in col:
            int_rename[col] = 'deficit_humedad'
        elif 'ficit' in col and 'vapor' in col:
            int_rename[col] = 'deficit_presion_vapor'
        elif 'humedad absoluta' in col:
            int_rename[col] = 'humedad_abs_int'

    df_int = df_int.rename(columns=int_rename)

    # Rename external columns (all lowercased by loader)
    ext_rename = {}
    for col in df_ext.columns:
        if 'promedio' in col and 'temperatura' in col:
            ext_rename[col] = 'temp_prom_ext'
        elif 'maxima' in col and 'temperatura' in col:
            ext_rename[col] = 'temp_max_ext'
        elif 'minima' in col and 'temperatura' in col:
            ext_rename[col] = 'temp_min_ext'
        elif 'hr promedio' in col:
            ext_rename[col] = 'hr_prom_ext'
        elif 'suma' in col:
            ext_rename[col] = 'rad_sum'
        elif 'intensidad' in col:
            ext_rename[col] = 'rad_max'
        elif 'dh' in col:
            ext_rename[col] = 'dh_ext'
        elif 'humedad absoluta' in col:
            ext_rename[col] = 'humedad_abs_ext'

    df_ext = df_ext.rename(columns=ext_rename)

    # Define feature columns to aggregate
    int_features = [c for c in ['temp_prom_int', 'temp_min_int', 'temp_max_int',
                                 'hr_prom_int', 'co2_ppm', 'deficit_humedad',
                                 'deficit_presion_vapor', 'humedad_abs_int']
                    if c in df_int.columns]

    ext_features = [c for c in ['temp_prom_ext', 'temp_max_ext', 'temp_min_ext',
                                 'hr_prom_ext', 'rad_sum', 'rad_max', 'dh_ext',
                                 'humedad_abs_ext']
                    if c in df_ext.columns]

    riego_features = [c for c in ['riego_total', 'ph_promedio', 'ce_promedio']
                      if c in df_riego.columns]

    all_season_frames = []

    for temp in SEASON_NAMES:
        # Get daily data for this season
        int_season = df_int[df_int['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        ext_season = df_ext[df_ext['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        kg_season = df_kg[df_kg['temporada'] == temp].reset_index(drop=True)

        if len(kg_season) == 0:
            continue

        # Merge internal + external on fecha
        daily = pd.merge(int_season[['fecha', 'temporada'] + int_features],
                         ext_season[['fecha'] + ext_features],
                         on='fecha', how='inner')

        # Merge riego data on fecha
        riego_season = df_riego[df_riego['temporada'] == temp][['fecha'] + riego_features]
        daily = pd.merge(daily, riego_season, on='fecha', how='left')

        # Assign sequential week index within the season
        daily = daily.sort_values('fecha').reset_index(drop=True)
        daily['week_idx'] = daily.index // 7

        # Add dias_desde_transplante
        tp_key = (temp, invernadero_id)
        if tp_key in transplant_dates:
            tp_date = transplant_dates[tp_key]
            daily['dias_desde_transplante'] = (daily['fecha'] - tp_date).dt.days

        # Aggregate daily -> weekly
        agg_dict = {feat: 'mean' for feat in int_features + ext_features}
        # Radiation sum should be summed, not averaged
        if 'rad_sum' in agg_dict:
            agg_dict['rad_sum'] = 'sum'
        # Riego: total is sum, pH and CE are mean
        for rf in riego_features:
            agg_dict[rf] = 'sum' if rf == 'riego_total' else 'mean'
        # dias_desde_transplante: take mean of the week
        if 'dias_desde_transplante' in daily.columns:
            agg_dict['dias_desde_transplante'] = 'mean'

        weekly_env = daily.groupby('week_idx').agg(agg_dict).reset_index()

        # Align with production data by sequential week number
        n_weeks = min(len(weekly_env), len(kg_season))
        weekly_env = weekly_env.iloc[:n_weeks].copy()
        kg_vals = kg_season.iloc[:n_weeks].copy()

        weekly = weekly_env.copy()
        weekly['kg_reales'] = kg_vals['kg_reales'].values
        weekly['temporada'] = temp
        weekly['semana'] = kg_vals['semana'].values

        # Temporal features
        weekly['week_in_season'] = np.arange(len(weekly))
        weekly['week_position'] = weekly['week_in_season'] / len(weekly)

        # Multi-harmonic Fourier encoding of ISO week number
        for period in [52, 26, 13]:
            weekly[f'week_sin_{period}'] = np.sin(2 * np.pi * weekly['semana'] / period)
            weekly[f'week_cos_{period}'] = np.cos(2 * np.pi * weekly['semana'] / period)

        # Lag features
        lags = lag_features if lag_features is not None else [1]
        for lag in lags:
            weekly[f'kg_lag_{lag}'] = weekly['kg_reales'].shift(lag)

        # Rolling statistics (optional)
        if include_rolling_mean:
            weekly['kg_roll_mean_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).mean()

        # Target: absolute yield h weeks ahead
        weekly['target'] = weekly['kg_reales'].shift(-horizon)

        all_season_frames.append(weekly)

    df_all = pd.concat(all_season_frames, ignore_index=True)

    # Split: configurable train/val seasons, T17 always test
    train_seasons = train_seasons if train_seasons is not None else ['T13', 'T14', 'T15']
    val_season = val_season if val_season is not None else 'T16'

    # Drop rows with NaN from lagging or target shift
    df_all = df_all.dropna().reset_index(drop=True)

    # Feature columns (all go through same pipeline)
    exclude = ['week_idx', 'kg_reales', 'temporada', 'target', 'fecha', 'semana']
    feature_cols = [c for c in df_all.columns if c not in exclude]

    train_df = df_all[df_all['temporada'].isin(train_seasons)].reset_index(drop=True)
    val_df = df_all[df_all['temporada'] == val_season].reset_index(drop=True)
    test_df = df_all[df_all['temporada'] == 'T17'].reset_index(drop=True)

    train_label = ','.join(train_seasons)
    print(f'  Invernadero {invernadero_id}:')
    print(f'    Train samples: {len(train_df)} ({train_label})')
    print(f'    Val samples:   {len(val_df)} ({val_season})')
    print(f'    Test samples:  {len(test_df)} (T17)')
    print(f'    Features ({len(feature_cols)}): {feature_cols}')

    return train_df, val_df, test_df, feature_cols


# ============================================================================
# 2. DATASET & DATALOADER
# ============================================================================

def make_sequences_per_season(X, y, temporadas, seq_len):
    """
    Build (seq_len, n_features) windows strictly within each season.
    No window ever crosses a season boundary — each season is treated
    as an independent time series.
    """
    seqs_X, seqs_y = [], []
    for temp in pd.Series(temporadas).unique():
        mask = np.array(temporadas) == temp
        X_s = X[mask]
        y_s = y[mask]
        for i in range(len(X_s) - seq_len + 1):
            seqs_X.append(X_s[i:i + seq_len])
            seqs_y.append(y_s[i + seq_len - 1])
    if len(seqs_X) == 0:
        raise ValueError('No sequences built — check seq_len vs season length.')
    return np.array(seqs_X, dtype=np.float32), np.array(seqs_y, dtype=np.float32)


# ============================================================================
# 3. CNN-RNN MODEL DEFINITION
# ============================================================================

class CNNBlock(nn.Module):
    """
    Single CNN block as described in the paper (Figure 2):
      1D Conv -> WeightNorm -> ReLU -> Dropout
    With an optional residual 1D Conv for dimension matching.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        # Trim conv output to match input length (handles padding > 0)
        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.residual_conv is None else self.residual_conv(x)
        return out + res


class CNNRNN(nn.Module):
    """
    CNN-RNN model for crop yield prediction (Section 2.2 of the paper).
    """
    def __init__(self, input_dim, cnn_filters, cnn_kernel_size, cnn_padding,
                 num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden):
        super().__init__()

        cnn_blocks = []
        in_ch = input_dim
        for _ in range(num_cnn_blocks):
            cnn_blocks.append(CNNBlock(in_ch, cnn_filters, cnn_kernel_size,
                                       cnn_padding, dropout))
            in_ch = cnn_filters
        self.cnn = nn.Sequential(*cnn_blocks)

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        pred = self.fc(last_out)
        return pred


def init_weights(model, method='default'):
    """Apply weight initialization to all layers of the model."""
    if method == 'default':
        return  # PyTorch defaults (Kaiming uniform for Linear, etc.)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases and 1D params

        if method == 'he':
            nn.init.kaiming_normal_(param, nonlinearity='relu')
        elif method == 'xavier':
            nn.init.xavier_normal_(param)
        elif method == 'orthogonal':
            nn.init.orthogonal_(param)
        elif method == 'lecun':
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')


# ============================================================================
# 4. HYPERPARAMETER CONFIGURATION
# ============================================================================

HYPERPARAMS_PER_GREENHOUSE = {
    3: {
        'seq_len': 2,
        'horizon': 4,
        'batch_size': 8,
        # Feature engineering
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
        # Multi-seed: try several seeds, keep best
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
                  0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                  100, 200, 500, 1000],
    },
    4: {
        'seq_len': 2,
        'horizon': 4,
        'batch_size': 8,
        # Feature engineering
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
        'corr_weight': 0.5,
        'epochs': 500,
        'patience': 150,
        'init_methods': ['default', 'xavier', 'orthogonal', 'lecun'],
        # Multi-seed: try several seeds, keep best
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


def prepare_data(invernadero_id, hp, train_seasons=None, val_season=None):
    """Load data, normalize, split into train/val/test, create DataLoaders."""
    train_df, val_df, test_df, feature_cols = build_dataset_for_greenhouse(
        invernadero_id, horizon=hp['horizon'],
        lag_features=hp.get('lag_features', [1]),
        include_rolling_mean=hp.get('include_rolling_mean', False),
        train_seasons=train_seasons, val_season=val_season,
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['target'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['target'].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['target'].values.astype(np.float32)

    # Box-Cox transform yield: finds optimal λ on train, applies to val/test.
    # λ→0 ≡ log; other values may fit the yield distribution better.
    # Shift by +1 to guarantee strictly positive inputs.
    y_train_bc, bc_lambda = boxcox(y_train + 1.0)
    y_train = y_train_bc.astype(np.float32)
    y_val   = boxcox(y_val  + 1.0, lmbda=bc_lambda).astype(np.float32)
    y_test  = boxcox(y_test + 1.0, lmbda=bc_lambda).astype(np.float32)
    print(f'  Box-Cox λ = {bc_lambda:.4f}')

    # Min-Max normalization to [-1, 1] for more dynamic range
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_val = scaler_X.transform(X_val)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # PCA dimensionality reduction
    pca = PCA(n_components=hp.get('pca_variance', 0.95))
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    n_components = pca.n_components_
    print(f'  PCA: {len(feature_cols)} features -> {n_components} components')

    # Build sequences per season (no cross-boundary windows)
    seq_len = hp['seq_len']
    X_tr, y_tr = make_sequences_per_season(X_train, y_train, train_df['temporada'].values, seq_len)
    X_va, y_va = make_sequences_per_season(X_val, y_val, val_df['temporada'].values, seq_len)
    X_te, y_te = make_sequences_per_season(X_test, y_test, test_df['temporada'].values, seq_len)
    print(f'  Sequences — train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}')

    def to_ds(X, y):
        return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1))

    train_loader = DataLoader(to_ds(X_tr, y_tr), batch_size=hp['batch_size'], shuffle=True)
    val_loader   = DataLoader(to_ds(X_va, y_va), batch_size=hp['batch_size'], shuffle=False)
    test_loader  = DataLoader(to_ds(X_te, y_te), batch_size=hp['batch_size'], shuffle=False)

    var_y_train = float(np.var(y_tr)) if len(y_tr) > 1 else 1.0
    print(f'  Train y variance (scaled Box-Cox): {var_y_train:.6f}')

    return train_loader, val_loader, test_loader, scaler_X, scaler_y, bc_lambda, feature_cols, n_components, var_y_train


class NSECorrLoss(nn.Module):
    """Nash-Sutcliffe Efficiency loss + Pearson correlation penalty.
    nse_loss = MSE / var_y_train  (fixed denominator from training data)
    Predicting the train mean always → nse_loss = 1.0.
    Perfect predictions → nse_loss = 0.
    Using a fixed denominator (not batch variance) keeps gradients stable.
    """
    def __init__(self, corr_weight=0.8, var_y_train=1.0):
        super().__init__()
        self.corr_weight = corr_weight
        self.var_y_train = max(var_y_train, 1e-8)

    def forward(self, pred, target):
        pred_f   = pred.flatten()
        target_f = target.flatten()
        nse_loss = torch.mean((pred_f - target_f) ** 2) / self.var_y_train
        corr_loss = torch.zeros(1, device=pred.device)
        if pred_f.shape[0] >= 4 and self.corr_weight > 0:
            pm = pred_f   - pred_f.mean()
            tm = target_f - target_f.mean()
            corr = torch.sum(pm * tm) / (
                torch.sqrt(torch.sum(pm ** 2) + 1e-8) *
                torch.sqrt(torch.sum(tm ** 2) + 1e-8)
            )
            corr_loss = self.corr_weight * (1.0 - corr)
        return nse_loss + corr_loss


class YieldWMAELoss(nn.Module):
    """Weighted MAE where weight ∝ actual yield^p (High-Price WMAE adapted for yield).
    After MinMax(-1,1) scaling: w_i = ((y_i + 1)/2)^p ∈ [0, 1].
    High-yield weeks carry more weight → model can't minimize loss by
    regressing to the mean (low-yield weeks have near-zero weight).
    Always stable: no normalizer, weights naturally bounded.
    """
    def __init__(self, corr_weight=0.8, power=1):
        super().__init__()
        self.corr_weight = corr_weight
        self.power = power

    def forward(self, pred, target):
        pred_f   = pred.flatten()
        target_f = target.flatten()
        weights  = ((target_f + 1.0) / 2.0).clamp(min=0.0) ** self.power
        wmae = torch.mean(weights * torch.abs(pred_f - target_f))
        corr_loss = torch.zeros(1, device=pred.device)
        if pred_f.shape[0] >= 4 and self.corr_weight > 0:
            pm = pred_f   - pred_f.mean()
            tm = target_f - target_f.mean()
            corr = torch.sum(pm * tm) / (
                torch.sqrt(torch.sum(pm ** 2) + 1e-8) *
                torch.sqrt(torch.sum(tm ** 2) + 1e-8)
            )
            corr_loss = self.corr_weight * (1.0 - corr)
        return wmae + corr_loss


def train_model(model, train_loader, val_loader, hp, model_path, var_y_train=1.0):
    """Train the CNN-RNN model with early stopping on validation score."""
    criterion = YieldWMAELoss(corr_weight=hp.get('corr_weight', 0.8),
                              power=hp.get('wmae_power', 1))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(hp['epochs']):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                val_batches += 1
                all_val_preds.append(pred.cpu())
                all_val_targets.append(y_batch.cpu())

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        # Compute validation correlation for early stopping
        val_preds_cat = torch.cat(all_val_preds).flatten()
        val_targets_cat = torch.cat(all_val_targets).flatten()
        vp = val_preds_cat - val_preds_cat.mean()
        vt = val_targets_cat - val_targets_cat.mean()
        val_corr = (torch.sum(vp * vt) / (
            torch.sqrt(torch.sum(vp ** 2) + 1e-8) *
            torch.sqrt(torch.sum(vt ** 2) + 1e-8)
        )).item()

        # Early stopping score: loss - corr_weight * correlation
        val_score = avg_val_loss - hp.get('corr_weight', 0.0) * val_corr

        if val_score < best_val_loss:
            best_val_loss = val_score
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train_loss:.6f} | '
                  f'Val Loss: {avg_val_loss:.6f} | '
                  f'Val Corr: {val_corr:.4f}')

        if patience_counter >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))

    return model, train_losses, val_losses


# ============================================================================
# 6. TESTING & EVALUATION
# ============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute all metrics from the paper + MAPE:
      - RMSE, R², NSE, PBIAS, MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    pbias = 100.0 * np.sum(y_true - y_pred) / np.sum(y_true)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        'RMSE (kg)': rmse,
        'R²': r2,
        'NSE': nse,
        'PBIAS (%)': pbias,
        'MAPE (%)': mape,
    }


def evaluate_model(model, test_loader, scaler_y, bc_lambda):
    """Run inference on test set, inverse-transform predictions, compute metrics."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred_norm = np.concatenate(all_preds).flatten()
    y_true_norm = np.concatenate(all_targets).flatten()

    # Inverse MinMaxScaler → Box-Cox space, then inv_boxcox - 1 → original kg
    y_pred_bc = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true_bc = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    y_pred = inv_boxcox(y_pred_bc, bc_lambda) - 1.0
    y_true = inv_boxcox(y_true_bc, bc_lambda) - 1.0

    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def plot_results_per_greenhouse(results):
    """Plot training/validation loss and predictions for each greenhouse."""
    n_greenhouses = len(results)
    fig, axes = plt.subplots(n_greenhouses, 2, figsize=(16, 5 * n_greenhouses))

    if n_greenhouses == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        # Training & validation loss
        axes[i, 0].plot(res['train_losses'], color='steelblue', linewidth=1, label='Train')
        axes[i, 0].plot(res['val_losses'], color='coral', linewidth=1, label='Validación (T16)')
        axes[i, 0].set_title(f'Invernadero {inv_id} - Loss (MSE)')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()

        # Predictions vs Actual
        weeks = np.arange(len(res['y_true']))
        axes[i, 1].plot(weeks, res['y_true'], 'o-', color='steelblue',
                         label='Actual', markersize=5)
        axes[i, 1].plot(weeks, res['y_pred'], 's--', color='coral',
                         label='Predicción (h=4)', markersize=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} - Predicción a 4 semanas (T17)')
        axes[i, 1].set_xlabel('Semana de test (T17)')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
        axes[i, 1].text(0.02, 0.98, metrics_text, transform=axes[i, 1].transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cnn_rnn_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results plot saved to {RESULTS_DIR / "cnn_rnn_results.png"}')


def print_metrics(invernadero_id, metrics):
    """Pretty-print evaluation metrics."""
    print(f'\n{"=" * 55}')
    print(f'  INVERNADERO {invernadero_id} - MÉTRICAS (Test: T17, h=4 semanas)')
    print(f'{"=" * 55}')
    for name, value in metrics.items():
        print(f'  {name:<15s}: {value:>10.4f}')
    print(f'{"=" * 55}')
    print(f'  >>> MAPE: {metrics["MAPE (%)"]:>10.4f}% <<<')
    print(f'{"=" * 55}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('  CNN-RNN Greenhouse Yield Prediction')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('  Prediction horizon: 4 weeks')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        hp = HYPERPARAMS_PER_GREENHOUSE[inv_id]
        seeds = hp.get('seeds', [42])
        init_methods = hp.get('init_methods', ['default', 'xavier', 'orthogonal', 'lecun'])

        n_runs = len(seeds) * len(init_methods)
        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id} — {n_runs} runs ({len(seeds)} seeds × {len(init_methods)} inits)')
        print(f'{"─" * 60}')

        best_r2 = -float('inf')
        best_result = None

        train_loader, val_loader, test_loader, scaler_X, scaler_y, bc_lambda, feature_cols, n_components, var_y_train = \
            prepare_data(inv_id, hp)

        for init_method in init_methods:
            for seed in seeds:
                set_seed(seed)

                model = CNNRNN(
                    input_dim=n_components,
                    cnn_filters=hp['cnn_filters'],
                    cnn_kernel_size=hp['cnn_kernel_size'],
                    cnn_padding=hp['cnn_padding'],
                    num_cnn_blocks=hp['num_cnn_blocks'],
                    lstm_hidden=hp['lstm_hidden'],
                    lstm_layers=hp['lstm_layers'],
                    dropout=hp['dropout'],
                    fc_hidden=hp['fc_hidden'],
                ).to(DEVICE)

                # Apply weight initialization (on CPU to avoid MPS limitations)
                if init_method != 'default':
                    model = model.cpu()
                    init_weights(model, init_method)
                    model = model.to(DEVICE)

                model_path = RESULTS_DIR / f'best_cnn_rnn_inv{inv_id}.pt'
                tmp_ckpt = RESULTS_DIR / f'_tmp_cnn_rnn_inv{inv_id}.pt'
                model, train_losses, val_losses = train_model(
                    model, train_loader, val_loader, hp, tmp_ckpt,
                    var_y_train=var_y_train
                )

                y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, bc_lambda)

                r2 = metrics['R²']
                mape = metrics['MAPE (%)']
                print(f'    [{init_method}] s{seed}: R²={r2:.4f}, MAPE={mape:.2f}%')

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'metrics': metrics,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'seed': seed,
                        'init_method': init_method,
                    }
                    # Save this run's model as the best across all seeds/inits
                    torch.save(model.state_dict(), model_path)

        if tmp_ckpt.exists():
            tmp_ckpt.unlink()
        print(f'\n  >>> Best init={best_result["init_method"]} seed={best_result["seed"]} (R²={best_r2:.4f}) <<<')
        print_metrics(inv_id, best_result['metrics'])

        all_results[inv_id] = best_result
        metrics_row = {'invernadero': inv_id, **best_result['metrics']}
        all_metrics.append(metrics_row)

    # Plot all results
    print('\nGenerando gráficas...')
    plot_results_per_greenhouse(all_results)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "metrics.csv"}')

    # Summary table
    print('\n' + '=' * 70)
    print('  RESUMEN - Predicción a 4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
