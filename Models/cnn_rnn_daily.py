"""
CNN-RNN Model at DAILY Resolution for Greenhouse Crop Yield Prediction
======================================================================
Instead of aggregating daily sensor data to weekly means, this model
ingests raw daily features directly.  The CNN layers extract local
(intra-week) patterns and the LSTM captures longer-range dynamics.

Each training sample is a window of `window_days` consecutive days of
raw sensor readings.  The target is the weekly kg production of the
week that ends on the last day of the window.

Architecture:
  CNN blocks (1-D over the time axis) → LSTM → FC → scalar prediction

Split: T13-T15 train, T16 validation, T17 test.
Prediction horizon: h weeks ahead (default 4).
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

DATA_DIR = Path(__file__).resolve().parent.parent / 'Data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_internal_variables_all_seasons():
    filepath = DATA_DIR / 'Variables internas invernadero 4.xlsx'
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
        data['semana'] = pd.to_numeric(data['semana'], errors='coerce')
        data['kg_reales'] = pd.to_numeric(data['kg_reales'], errors='coerce')
        data['temporada'] = SEASON_NAMES[i]
        data['invernadero'] = invernadero_id
        frames.append(data)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['kg_reales']).reset_index(drop=True)
    return df


# ============================================================================
# 2. DAILY-RESOLUTION DATASET BUILDER
# ============================================================================

def rename_columns(df_int, df_ext):
    """Standardise column names for internal and external dataframes."""
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

    return df_int, df_ext


def build_daily_dataset(invernadero_id, horizon=4):
    """
    Build a daily-resolution dataset for one greenhouse.

    Each row is a single day with raw sensor features.  A `week_idx`
    column indicates which production week the day belongs to, and
    `kg_target` is the production `horizon` weeks ahead of that week.

    Returns
    -------
    daily_all : DataFrame  (one row per day, all seasons concatenated)
    feature_cols : list[str]
    """
    df_int = load_internal_variables_all_seasons()
    df_ext = load_external_variables_all_seasons()
    df_kg = load_production(invernadero_id)

    df_int, df_ext = rename_columns(df_int, df_ext)

    int_features = [c for c in ['temp_prom_int', 'temp_min_int', 'temp_max_int',
                                 'hr_prom_int', 'co2_ppm', 'deficit_humedad',
                                 'deficit_presion_vapor', 'humedad_abs_int']
                    if c in df_int.columns]

    ext_features = [c for c in ['temp_prom_ext', 'temp_max_ext', 'temp_min_ext',
                                 'hr_prom_ext', 'rad_sum', 'rad_max', 'dh_ext',
                                 'humedad_abs_ext']
                    if c in df_ext.columns]

    feature_cols = int_features + ext_features
    all_frames = []

    for temp in SEASON_NAMES:
        int_s = df_int[df_int['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        ext_s = df_ext[df_ext['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        kg_s = df_kg[df_kg['temporada'] == temp].reset_index(drop=True)

        if len(kg_s) == 0:
            continue

        # Merge daily internal + external
        daily = pd.merge(int_s[['fecha', 'temporada'] + int_features],
                         ext_s[['fecha'] + ext_features],
                         on='fecha', how='inner')
        daily = daily.sort_values('fecha').reset_index(drop=True)

        # Assign each day to a production week (0-indexed)
        daily['week_idx'] = daily.index // 7
        daily['day_in_week'] = daily.index % 7  # 0=lun … 6=dom of that week

        # Map weekly kg to each day (the kg of the week the day belongs to)
        n_weeks = min(daily['week_idx'].max() + 1, len(kg_s))
        week_to_kg = kg_s.iloc[:n_weeks].set_index(kg_s.index)['kg_reales']

        daily = daily[daily['week_idx'] < n_weeks].copy()
        daily['kg_week'] = daily['week_idx'].map(week_to_kg)

        # Target: kg of the week that is `horizon` weeks ahead
        target_map = {}
        for w in range(n_weeks):
            future_w = w + horizon
            if future_w < n_weeks:
                target_map[w] = week_to_kg[future_w]
        daily['kg_target'] = daily['week_idx'].map(target_map)

        daily['temporada'] = temp

        # Add cyclical day-in-week encoding
        daily['day_sin'] = np.sin(2 * np.pi * daily['day_in_week'] / 7)
        daily['day_cos'] = np.cos(2 * np.pi * daily['day_in_week'] / 7)

        all_frames.append(daily)

    daily_all = pd.concat(all_frames, ignore_index=True)
    daily_all = daily_all.dropna(subset=['kg_target']).reset_index(drop=True)

    # Final feature list includes cyclical day encoding
    feature_cols = feature_cols + ['day_sin', 'day_cos']

    # Forward-fill any remaining NaN in sensor columns
    daily_all[feature_cols] = daily_all[feature_cols].ffill().bfill()

    print(f'  Invernadero {invernadero_id} (daily resolution):')
    print(f'    Total daily rows: {len(daily_all)}')
    print(f'    Features ({len(feature_cols)}): {feature_cols}')

    return daily_all, feature_cols


# ============================================================================
# 3. WINDOWED DATASET FOR PYTORCH
# ============================================================================

class DailyWindowDataset(Dataset):
    """
    Sliding window over daily data.

    Each sample is a contiguous block of `window_days` days of raw features.
    The target is `kg_target` of the LAST day in the window (i.e., the
    weekly production `horizon` weeks ahead of the week that day belongs to).

    Windows are only created within a single season (no cross-season windows).
    """
    def __init__(self, daily_df, feature_cols, window_days):
        self.window_days = window_days
        self.samples = []  # list of (features_array, target_scalar)

        for _, season_df in daily_df.groupby('temporada'):
            feats = season_df[feature_cols].values.astype(np.float32)
            targets = season_df['kg_target'].values.astype(np.float32)
            n = len(feats)
            for i in range(n - window_days + 1):
                x = feats[i:i + window_days]
                y = targets[i + window_days - 1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================================
# 4. MODEL (same CNN-RNN architecture, tuned for daily input)
# ============================================================================

class CNNBlock(nn.Module):
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
        out = self.dropout(self.relu(self.conv1(x)))
        res = x if self.residual_conv is None else self.residual_conv(x)
        return out + res


class CNNRNNDaily(nn.Module):
    """
    CNN-RNN that operates on daily-resolution sequences.

    Input shape: (batch, window_days, n_features)
    Output shape: (batch, 1)
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
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        # x: (batch, window_days, features)
        x = x.permute(0, 2, 1)       # (batch, features, window_days)
        x = self.cnn(x)              # (batch, cnn_filters, window_days)
        x = x.permute(0, 2, 1)       # (batch, window_days, cnn_filters)
        lstm_out, _ = self.lstm(x)    # (batch, window_days, lstm_hidden)
        last = lstm_out[:, -1, :]     # (batch, lstm_hidden)
        return self.fc(last)          # (batch, 1)


# ============================================================================
# 5. HYPERPARAMETERS
# ============================================================================

HYPERPARAMS = {
    # Data
    'window_days': 28,        # 4 weeks of daily data as input context
    'horizon': 4,             # predict kg 4 weeks ahead
    'pca_variance': 0.95,     # retain 95% of variance in PCA
    'batch_size': 8,

    # CNN — kernel_size=7 to capture weekly cycles
    'cnn_filters': 32,
    'cnn_kernel_size': 7,
    'cnn_padding': 3,
    'num_cnn_blocks': 2,

    # RNN
    'lstm_hidden': 32,
    'lstm_layers': 1,
    'fc_hidden': 16,

    # Training
    'dropout': 0.3,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 500,
    'patience': 60,

    'seed': 42,
}


# ============================================================================
# 6. TRAINING & EVALUATION
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(invernadero_id, hp):
    daily_all, feature_cols = build_daily_dataset(invernadero_id, horizon=hp['horizon'])

    train_seasons = ['T13', 'T14', 'T15']
    val_seasons = ['T16']
    test_seasons = ['T17']

    train_daily = daily_all[daily_all['temporada'].isin(train_seasons)].reset_index(drop=True)
    val_daily = daily_all[daily_all['temporada'].isin(val_seasons)].reset_index(drop=True)
    test_daily = daily_all[daily_all['temporada'].isin(test_seasons)].reset_index(drop=True)

    # Fit scaler on training data only
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(train_daily[feature_cols].values)
    scaler_y.fit(train_daily[['kg_target']].values)

    for df in [train_daily, val_daily, test_daily]:
        df[feature_cols] = scaler_X.transform(df[feature_cols].values)
        df['kg_target'] = scaler_y.transform(df[['kg_target']].values).flatten()

    # Apply PCA on scaled features (fit on training data only)
    pca = PCA(n_components=hp.get('pca_variance', 0.95))
    pca.fit(train_daily[feature_cols].values)

    n_components = pca.n_components_
    pca_cols = [f'pc{i+1}' for i in range(n_components)]

    print(f'    PCA: {len(feature_cols)} features → {n_components} components '
          f'({pca.explained_variance_ratio_.sum()*100:.1f}% varianza explicada)')

    for df in [train_daily, val_daily, test_daily]:
        pca_values = pca.transform(df[feature_cols].values)
        for j, col in enumerate(pca_cols):
            df[col] = pca_values[:, j]

    feature_cols = pca_cols

    train_ds = DailyWindowDataset(train_daily, feature_cols, hp['window_days'])
    val_ds = DailyWindowDataset(val_daily, feature_cols, hp['window_days'])
    test_ds = DailyWindowDataset(test_daily, feature_cols, hp['window_days'])

    print(f'    Train windows: {len(train_ds)}')
    print(f'    Val windows:   {len(val_ds)}')
    print(f'    Test windows:  {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler_X, scaler_y, feature_cols


def train_model(model, train_loader, val_loader, hp, model_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(hp['epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss, vb = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                loss = criterion(model(X_batch), y_batch)
                val_loss += loss.item()
                vb += 1
        avg_val = val_loss / max(vb, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train:.6f} | Val: {avg_val:.6f} | '
                  f'Best: {best_val_loss:.6f}')

        if patience_counter >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, train_losses, val_losses


def compute_metrics(y_true, y_pred):
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


def evaluate_model(model, test_loader, scaler_y):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred_n = np.concatenate(all_preds).flatten()
    y_true_n = np.concatenate(all_targets).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_n.reshape(-1, 1)).flatten()
    return y_true, y_pred, compute_metrics(y_true, y_pred)


def plot_results(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        axes[i, 0].plot(res['train_losses'], color='steelblue', lw=1, label='Train')
        axes[i, 0].plot(res['val_losses'], color='coral', lw=1, label='Val (T16)')
        axes[i, 0].set_title(f'Inv. {inv_id} - Loss (MSE)')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()

        t = np.arange(len(res['y_true']))
        axes[i, 1].plot(t, res['y_true'], 'o-', color='steelblue', ms=4, label='Actual')
        axes[i, 1].plot(t, res['y_pred'], 's--', color='coral', ms=4, label='Pred (h=4)')
        axes[i, 1].set_title(f'Inv. {inv_id} - Predicción diaria→semanal (T17)')
        axes[i, 1].set_xlabel('Window index (test)')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()
        txt = '\n'.join(f'{k}: {v:.4f}' for k, v in res['metrics'].items())
        axes[i, 1].text(0.02, 0.98, txt, transform=axes[i, 1].transAxes,
                         va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = RESULTS_DIR / 'cnn_rnn_daily_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {path}')


def print_metrics(inv_id, metrics):
    print(f'\n{"=" * 60}')
    print(f'  INVERNADERO {inv_id} - MÉTRICAS (Test: T17, h=4, daily input)')
    print(f'{"=" * 60}')
    for k, v in metrics.items():
        print(f'  {k:<15s}: {v:>10.4f}')
    print(f'{"=" * 60}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    hp = HYPERPARAMS
    set_seed(hp['seed'])

    print('=' * 65)
    print('  CNN-RNN (Daily Resolution) — Greenhouse Yield Prediction')
    print(f'  Window: {hp["window_days"]} days | Horizon: {hp["horizon"]} weeks')
    print(f'  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 65)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id}')
        print(f'{"─" * 60}')

        print('\n  [1/4] Cargando datos diarios...')
        train_loader, val_loader, test_loader, scaler_X, scaler_y, feat_cols = \
            prepare_data(inv_id, hp)

        print(f'\n  [2/4] Construyendo modelo CNN-RNN (daily)...')
        model = CNNRNNDaily(
            input_dim=len(feat_cols),
            cnn_filters=hp['cnn_filters'],
            cnn_kernel_size=hp['cnn_kernel_size'],
            cnn_padding=hp['cnn_padding'],
            num_cnn_blocks=hp['num_cnn_blocks'],
            lstm_hidden=hp['lstm_hidden'],
            lstm_layers=hp['lstm_layers'],
            dropout=hp['dropout'],
            fc_hidden=hp['fc_hidden'],
        ).to(DEVICE)

        params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {params:,}')

        print(f'\n  [3/4] Entrenando...')
        model_path = RESULTS_DIR / f'best_cnn_rnn_daily_inv{inv_id}.pt'
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, hp, model_path)

        print(f'\n  [4/4] Evaluando en T17...')
        y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y)
        print_metrics(inv_id, metrics)

        all_results[inv_id] = {
            'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics,
            'train_losses': train_losses, 'val_losses': val_losses,
        }
        all_metrics.append({'invernadero': inv_id, **metrics})

    print('\nGenerando gráficas...')
    plot_results(all_results)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'cnn_rnn_daily_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "cnn_rnn_daily_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN CNN-RNN (Daily) — Predicción a 4 semanas')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
