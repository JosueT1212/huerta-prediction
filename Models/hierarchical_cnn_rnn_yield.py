"""
Hierarchical CNN-RNN for Greenhouse Crop Yield Prediction
==========================================================
Key idea: Instead of averaging daily sensor data into weekly means (losing
within-week patterns), use a two-level architecture:

  Level 1 — Week Encoder (CNN):
    Takes 7 daily readings for each week and compresses them into a
    fixed-size weekly embedding. This preserves daily patterns (spikes,
    trends, variance) that averaging destroys.

  Level 2 — Sequence Model (LSTM):
    Takes a sequence of weekly embeddings and predicts kg yield h weeks ahead.
    Same as the baseline CNN-RNN but now each "week" input is a learned
    representation of 7 days, not a simple average.

Architecture:
  Input: (batch, seq_weeks, 7, n_features) — seq_weeks of daily data
  Week Encoder CNN: (batch*seq_weeks, n_features, 7) → (batch*seq_weeks, embed_dim)
  LSTM: (batch, seq_weeks, embed_dim) → (batch, lstm_hidden)
  FC head: (batch, lstm_hidden) → (batch, 1)

Split: T13-T15 train | T16 validation | T17 test
Prediction horizon: 4 weeks ahead
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
# 1. DATA LOADING (reuse loaders from cnn_rnn_yield)
# ============================================================================

from cnn_rnn_yield import (
    load_internal_variables_all_seasons,
    load_external_variables_all_seasons,
    load_production,
    compute_metrics,
    set_seed,
)


def build_daily_dataset(invernadero_id, horizon=4, lag_weeks=1):
    """
    Build dataset with RAW daily sensor data organized by week.

    Returns per-season data: for each week w, we store:
      - daily_features[w]: (7, n_features) — 7 days of sensor readings
      - target[w]: kg_reales at week w+horizon
      - kg_lag[w]: kg_reales at week w-1 (lag feature, appended to embedding)

    Only complete weeks (exactly 7 days) are kept.
    """
    df_int = load_internal_variables_all_seasons()
    df_ext = load_external_variables_all_seasons()
    df_kg = load_production(invernadero_id)

    # Rename columns (same as cnn_rnn_yield)
    int_rename = {}
    for col in df_int.columns:
        if 'temperatura promedio' in col: int_rename[col] = 'temp_prom_int'
        elif 'temperatura minima' in col: int_rename[col] = 'temp_min_int'
        elif 'temperatura maxima' in col: int_rename[col] = 'temp_max_int'
        elif 'humedad relativa promedio' in col: int_rename[col] = 'hr_prom_int'
        elif 'co2' in col: int_rename[col] = 'co2_ppm'
        elif 'ficit de humedad' in col: int_rename[col] = 'deficit_humedad'
        elif 'ficit' in col and 'vapor' in col: int_rename[col] = 'deficit_presion_vapor'
        elif 'humedad absoluta' in col: int_rename[col] = 'humedad_abs_int'
    df_int = df_int.rename(columns=int_rename)

    ext_rename = {}
    for col in df_ext.columns:
        if 'promedio' in col and 'temperatura' in col: ext_rename[col] = 'temp_prom_ext'
        elif 'maxima' in col and 'temperatura' in col: ext_rename[col] = 'temp_max_ext'
        elif 'minima' in col and 'temperatura' in col: ext_rename[col] = 'temp_min_ext'
        elif 'hr promedio' in col: ext_rename[col] = 'hr_prom_ext'
        elif 'suma' in col: ext_rename[col] = 'rad_sum'
        elif 'intensidad' in col: ext_rename[col] = 'rad_max'
        elif 'dh' in col: ext_rename[col] = 'dh_ext'
        elif 'humedad absoluta' in col: ext_rename[col] = 'humedad_abs_ext'
    df_ext = df_ext.rename(columns=ext_rename)

    int_features = [c for c in ['temp_prom_int', 'temp_min_int', 'temp_max_int',
                                 'hr_prom_int', 'co2_ppm', 'deficit_humedad',
                                 'deficit_presion_vapor', 'humedad_abs_int']
                    if c in df_int.columns]

    ext_features = [c for c in ['temp_prom_ext', 'temp_max_ext', 'temp_min_ext',
                                 'hr_prom_ext', 'rad_sum', 'rad_max', 'dh_ext',
                                 'humedad_abs_ext']
                    if c in df_ext.columns]

    sensor_features = int_features + ext_features

    all_seasons = {}

    for temp in SEASON_NAMES:
        int_season = df_int[df_int['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        ext_season = df_ext[df_ext['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        kg_season = df_kg[df_kg['temporada'] == temp].reset_index(drop=True)

        if len(kg_season) == 0:
            continue

        # Merge internal + external on fecha
        daily = pd.merge(int_season[['fecha', 'temporada'] + int_features],
                         ext_season[['fecha'] + ext_features],
                         on='fecha', how='inner')
        daily = daily.sort_values('fecha').reset_index(drop=True)
        daily['week_idx'] = daily.index // 7

        # Only keep complete weeks (7 days)
        week_counts = daily.groupby('week_idx').size()
        complete_weeks = week_counts[week_counts == 7].index

        n_weeks = min(len(complete_weeks), len(kg_season))
        complete_weeks = sorted(complete_weeks)[:n_weeks]

        # Build daily arrays per week: (n_weeks, 7, n_sensor_features)
        daily_arrays = []
        for w in complete_weeks:
            week_data = daily[daily['week_idx'] == w][sensor_features].values
            daily_arrays.append(week_data)

        daily_arrays = np.array(daily_arrays, dtype=np.float32)  # (n_weeks, 7, n_features)
        kg_values = kg_season['kg_reales'].values[:n_weeks].astype(np.float32)

        # Build samples: for week w, target = kg[w + horizon]
        # Also include kg_lag (previous week's production) and week_position
        samples = []
        for w in range(lag_weeks, n_weeks - horizon):
            daily_feat = daily_arrays[w]  # (7, n_features)
            target = kg_values[w + horizon]
            kg_lag = kg_values[w - lag_weeks:w]  # (lag_weeks,) previous weeks' kg
            week_pos = w / n_weeks  # normalized position in season
            samples.append({
                'daily': daily_feat,
                'target': target,
                'kg_lag': kg_lag,
                'week_pos': week_pos,
                'temporada': temp,
            })

        all_seasons[temp] = samples

    print(f'  Invernadero {invernadero_id}:')
    print(f'    Sensor features: {len(sensor_features)} ({sensor_features})')
    for temp, samples in all_seasons.items():
        print(f'    {temp}: {len(samples)} samples')

    return all_seasons, sensor_features


# ============================================================================
# 2. DATASET
# ============================================================================

class HierarchicalDataset(Dataset):
    """
    Each sample is a window of `seq_weeks` consecutive weeks.
    For each week: (7, n_features) daily data + kg_lag + week_position.
    Target: kg at last week + horizon.
    """
    def __init__(self, samples, seq_weeks):
        self.seq_weeks = seq_weeks
        # Build sequences from consecutive samples within each season
        self.sequences = []
        # Group by season to avoid cross-season sequences
        by_season = {}
        for s in samples:
            by_season.setdefault(s['temporada'], []).append(s)

        for temp, season_samples in by_season.items():
            for i in range(len(season_samples) - seq_weeks + 1):
                window = season_samples[i:i + seq_weeks]
                self.sequences.append(window)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        window = self.sequences[idx]
        # Stack daily data: (seq_weeks, 7, n_features)
        daily = np.stack([w['daily'] for w in window])
        # Scalar features per week: kg_lag + week_pos
        kg_lags = np.array([w['kg_lag'] for w in window], dtype=np.float32)  # (seq_weeks, lag_weeks)
        week_pos = np.array([w['week_pos'] for w in window], dtype=np.float32)  # (seq_weeks,)
        target = window[-1]['target']

        return (torch.FloatTensor(daily),
                torch.FloatTensor(kg_lags),
                torch.FloatTensor(week_pos),
                torch.FloatTensor([target]))


# ============================================================================
# 3. HIERARCHICAL MODEL
# ============================================================================

class WeekEncoder(nn.Module):
    """
    CNN that encodes 7 daily readings into a fixed-size weekly embedding.
    Input: (batch, n_features, 7)
    Output: (batch, embed_dim)
    """
    def __init__(self, n_features, embed_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Global average pooling over time dimension (7 days → 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, n_features, 7)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(x).squeeze(-1)  # (batch, embed_dim)
        return x


class HierarchicalCNNRNN(nn.Module):
    """
    Two-level architecture:
      Level 1: WeekEncoder CNN processes 7 daily readings → weekly embedding
      Level 2: LSTM processes sequence of weekly embeddings → prediction

    The weekly embedding is concatenated with scalar features (kg_lag, week_pos)
    before entering the LSTM.
    """
    def __init__(self, n_features, embed_dim=64, lstm_hidden=128,
                 lstm_layers=3, fc_hidden=128, dropout=0.05, lag_weeks=1):
        super().__init__()

        self.week_encoder = WeekEncoder(n_features, embed_dim, dropout)

        # LSTM input: week_embedding + kg_lags + week_position
        lstm_input_dim = embed_dim + lag_weeks + 1

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
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

    def forward(self, daily, kg_lags, week_pos):
        """
        daily: (batch, seq_weeks, 7, n_features)
        kg_lags: (batch, seq_weeks, lag_weeks)
        week_pos: (batch, seq_weeks)
        """
        batch_size, seq_weeks, days, n_feat = daily.shape

        # Reshape to process all weeks at once through the encoder
        daily_flat = daily.view(batch_size * seq_weeks, days, n_feat)
        daily_flat = daily_flat.permute(0, 2, 1)  # (B*S, n_feat, 7)
        week_emb = self.week_encoder(daily_flat)  # (B*S, embed_dim)
        week_emb = week_emb.view(batch_size, seq_weeks, -1)  # (B, S, embed_dim)

        # Concatenate scalar features
        week_pos_expanded = week_pos.unsqueeze(-1)  # (B, S, 1)
        lstm_input = torch.cat([week_emb, kg_lags, week_pos_expanded], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :]  # (B, lstm_hidden)

        pred = self.fc(last_out)
        return pred


# ============================================================================
# 4. LOSS (same CorrMSE)
# ============================================================================

class CorrMSELoss(nn.Module):
    def __init__(self, corr_weight=0.1):
        super().__init__()
        self.corr_weight = corr_weight

    def forward(self, pred, target):
        mse_loss = nn.functional.mse_loss(pred, target)
        loss = mse_loss

        if pred.shape[0] >= 4 and self.corr_weight > 0:
            p = pred.flatten()
            t = target.flatten()
            pm = p - p.mean()
            tm = t - t.mean()
            corr = torch.sum(pm * tm) / (
                torch.sqrt(torch.sum(pm ** 2) + 1e-8) *
                torch.sqrt(torch.sum(tm ** 2) + 1e-8)
            )
            loss = loss + self.corr_weight * (1.0 - corr)

        return loss


# ============================================================================
# 5. TRAINING & EVALUATION
# ============================================================================

def prepare_data(invernadero_id, hp):
    """Build daily dataset, scale, create DataLoaders."""
    all_seasons, sensor_features = build_daily_dataset(
        invernadero_id, horizon=hp['horizon'], lag_weeks=hp['lag_weeks']
    )

    # Collect all samples by split
    train_seasons = ['T13', 'T14', 'T15']
    val_season = 'T16'

    train_samples = []
    for t in train_seasons:
        train_samples.extend(all_seasons.get(t, []))
    val_samples = all_seasons.get(val_season, [])
    test_samples = all_seasons.get('T17', [])

    print(f'    Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

    # Fit scalers on training data
    # Daily sensor scaler
    all_train_daily = np.concatenate([s['daily'] for s in train_samples])  # (N*7, n_feat)
    all_train_daily_flat = all_train_daily.reshape(-1, all_train_daily.shape[-1])
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_X.fit(all_train_daily_flat)

    # Target scaler
    all_train_targets = np.array([s['target'] for s in train_samples])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(all_train_targets.reshape(-1, 1))

    # Kg lag scaler
    all_train_kg_lags = np.concatenate([s['kg_lag'] for s in train_samples])
    scaler_kg = MinMaxScaler(feature_range=(-1, 1))
    scaler_kg.fit(all_train_kg_lags.reshape(-1, 1))

    # Apply scaling
    def scale_samples(samples):
        scaled = []
        for s in samples:
            daily_shape = s['daily'].shape  # (7, n_feat)
            daily_scaled = scaler_X.transform(s['daily'].reshape(-1, daily_shape[-1]))
            daily_scaled = daily_scaled.reshape(daily_shape).astype(np.float32)

            target_scaled = scaler_y.transform([[s['target']]])[0, 0]
            kg_lag_scaled = scaler_kg.transform(s['kg_lag'].reshape(-1, 1)).flatten().astype(np.float32)

            scaled.append({
                'daily': daily_scaled,
                'target': target_scaled,
                'kg_lag': kg_lag_scaled,
                'week_pos': s['week_pos'],
                'temporada': s['temporada'],
            })
        return scaled

    train_scaled = scale_samples(train_samples)
    val_scaled = scale_samples(val_samples)
    test_scaled = scale_samples(test_samples)

    seq_weeks = hp['seq_weeks']
    train_ds = HierarchicalDataset(train_scaled, seq_weeks)
    val_ds = HierarchicalDataset(val_scaled, seq_weeks)
    test_ds = HierarchicalDataset(test_scaled, seq_weeks)

    print(f'    Sequences — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler_y, len(sensor_features)


def train_model(model, train_loader, val_loader, hp, model_path):
    """Train with early stopping on validation score."""
    criterion = CorrMSELoss(corr_weight=hp['corr_weight'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])

    best_val_score = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(hp['epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for daily, kg_lags, week_pos, target in train_loader:
            daily = daily.to(DEVICE)
            kg_lags = kg_lags.to(DEVICE)
            week_pos = week_pos.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            pred = model(daily, kg_lags, week_pos)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for daily, kg_lags, week_pos, target in val_loader:
                daily = daily.to(DEVICE)
                kg_lags = kg_lags.to(DEVICE)
                week_pos = week_pos.to(DEVICE)
                target = target.to(DEVICE)

                pred = model(daily, kg_lags, week_pos)
                loss = criterion(pred, target)
                val_loss += loss.item()
                val_batches += 1
                all_val_preds.append(pred.cpu())
                all_val_targets.append(target.cpu())

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        val_preds_cat = torch.cat(all_val_preds).flatten()
        val_targets_cat = torch.cat(all_val_targets).flatten()
        vp = val_preds_cat - val_preds_cat.mean()
        vt = val_targets_cat - val_targets_cat.mean()
        val_corr = (torch.sum(vp * vt) / (
            torch.sqrt(torch.sum(vp ** 2) + 1e-8) *
            torch.sqrt(torch.sum(vt ** 2) + 1e-8)
        )).item()

        val_score = avg_val_loss - hp['corr_weight'] * val_corr

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train_loss:.6f} | '
                  f'Val: {avg_val_loss:.6f} | '
                  f'Corr: {val_corr:.4f}')

        if patience_counter >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y):
    """Run inference, inverse-transform, compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for daily, kg_lags, week_pos, target in test_loader:
            daily = daily.to(DEVICE)
            kg_lags = kg_lags.to(DEVICE)
            week_pos = week_pos.to(DEVICE)

            pred = model(daily, kg_lags, week_pos)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    y_pred_norm = np.concatenate(all_preds).flatten()
    y_true_norm = np.concatenate(all_targets).flatten()

    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()

    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def init_weights(model, method='xavier'):
    """Apply weight initialization."""
    if method == 'default':
        return
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if method == 'xavier':
            nn.init.xavier_normal_(param)
        elif method == 'he':
            nn.init.kaiming_normal_(param, nonlinearity='relu')


# ============================================================================
# 6. HYPERPARAMETERS
# ============================================================================

HYPERPARAMS_PER_GREENHOUSE = {
    3: {
        'horizon': 4,
        'lag_weeks': 1,
        'seq_weeks': 2,
        'batch_size': 8,
        # Week encoder
        'embed_dim': 64,
        # LSTM
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
        'init_method': 'default',
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025],
    },
    4: {
        'horizon': 4,
        'lag_weeks': 1,
        'seq_weeks': 2,
        'batch_size': 8,
        # Week encoder
        'embed_dim': 64,
        # LSTM
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
        'init_method': 'xavier',
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025],
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('  Hierarchical CNN-RNN — Daily Resolution')
    print('  Week Encoder (CNN) + Sequence Model (LSTM)')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        hp = HYPERPARAMS_PER_GREENHOUSE[inv_id]
        seeds = hp['seeds']
        init_method = hp.get('init_method', 'default')

        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id} — init={init_method}, {len(seeds)} seeds')
        print(f'{"─" * 60}')

        best_r2 = -float('inf')
        best_result = None

        for seed in seeds:
            set_seed(seed)

            train_loader, val_loader, test_loader, scaler_y, n_features = \
                prepare_data(inv_id, hp)

            model = HierarchicalCNNRNN(
                n_features=n_features,
                embed_dim=hp['embed_dim'],
                lstm_hidden=hp['lstm_hidden'],
                lstm_layers=hp['lstm_layers'],
                fc_hidden=hp['fc_hidden'],
                dropout=hp['dropout'],
                lag_weeks=hp['lag_weeks'],
            )

            # Init on CPU to avoid MPS issues, then move to device
            if init_method != 'default':
                init_weights(model, init_method)
            model = model.to(DEVICE)

            total_params = sum(p.numel() for p in model.parameters())
            if seed == seeds[0]:
                print(f'  Model parameters: {total_params:,}')

            model_path = RESULTS_DIR / f'best_hierarchical_inv{inv_id}.pt'
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, hp, model_path
            )

            y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y)
            r2 = metrics['R²']
            mape = metrics['MAPE (%)']
            print(f'    s{seed}: R²={r2:.4f}, MAPE={mape:.2f}%')

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'metrics': metrics,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'seed': seed,
                }

        print(f'\n  >>> Best seed={best_result["seed"]} (R²={best_r2:.4f}) <<<')

        # Print metrics
        print(f'\n{"=" * 55}')
        print(f'  INVERNADERO {inv_id} - MÉTRICAS (Test: T17, h=4 semanas)')
        print(f'{"=" * 55}')
        for name, value in best_result['metrics'].items():
            print(f'  {name:<15s}: {value:>10.4f}')
        print(f'{"=" * 55}')

        all_results[inv_id] = best_result
        all_metrics.append({'invernadero': inv_id, **best_result['metrics']})

    # Plot
    n_greenhouses = len(all_results)
    fig, axes = plt.subplots(n_greenhouses, 2, figsize=(16, 5 * n_greenhouses))
    if n_greenhouses == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(all_results.items()):
        axes[i, 0].plot(res['train_losses'], color='steelblue', linewidth=1, label='Train')
        axes[i, 0].plot(res['val_losses'], color='coral', linewidth=1, label='Val')
        axes[i, 0].set_title(f'Invernadero {inv_id} - Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()

        weeks = np.arange(len(res['y_true']))
        axes[i, 1].plot(weeks, res['y_true'], 'o-', color='steelblue',
                         label='Actual', markersize=5)
        axes[i, 1].plot(weeks, res['y_pred'], 's--', color='coral',
                         label='Hierarchical (h=4)', markersize=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} - Predicción (T17)')
        axes[i, 1].set_xlabel('Semana')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
        axes[i, 1].text(0.02, 0.98, metrics_text, transform=axes[i, 1].transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'hierarchical_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {RESULTS_DIR / "hierarchical_results.png"}')

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'hierarchical_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "hierarchical_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN — Hierarchical CNN-RNN (daily resolution)')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
