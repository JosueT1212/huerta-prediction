"""
Feed-Forward Neural Network for Greenhouse Crop Yield Prediction
================================================================
Simple FFNN baseline: flat PCA-reduced weekly features → hidden layer (64) → prediction.

No sequences — each row is an independent sample (like XGBoost) but trained
with backpropagation via PyTorch.

Trains one model per greenhouse (Inv. 3, Inv. 4) using all 5 seasons.
Split: T13-T15 train, T16 validation, T17 test.
Prediction horizon: 4 weeks ahead.

Sections:
  1. Data Loading & Feature Engineering (shared with CNN-RNN / XGBoost)
  2. FFNN Model Definition
  3. Hyperparameter Configuration
  4. Training (with validation-based early stopping)
  5. Testing & Evaluation (RMSE, R², NSE, PBIAS, MAPE)
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
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_internal_variables_all_seasons():
    """Load internal greenhouse 4 sensor data for all 5 seasons."""
    filepath = DATA_DIR / 'Variables internas invernadero 4.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={'fecha': 'fecha'})
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
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


def build_dataset_for_greenhouse(invernadero_id, horizon=4):
    """
    Build feature matrix for a single greenhouse using all 5 seasons.

    Merges internal sensor data + external weather data (aggregated to weekly)
    with weekly production targets. Creates lag features and rolling stats.

    The target is kg_reales at time t+horizon (4-week ahead prediction).

    Returns:
        train_df, val_df, test_df, feature_cols
    """
    df_int = load_internal_variables_all_seasons()
    df_ext = load_external_variables_all_seasons()
    df_kg = load_production(invernadero_id)

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

        # Assign sequential week index within the season
        daily = daily.sort_values('fecha').reset_index(drop=True)
        daily['week_idx'] = daily.index // 7

        # Aggregate daily -> weekly
        agg_dict = {feat: 'mean' for feat in int_features + ext_features}
        # Radiation sum should be summed, not averaged
        if 'rad_sum' in agg_dict:
            agg_dict['rad_sum'] = 'sum'

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

        # Lag features
        for lag in range(1, 5):
            weekly[f'kg_lag_{lag}'] = weekly['kg_reales'].shift(lag)

        # Rolling statistics
        weekly['kg_roll_mean_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).mean()
        weekly['kg_roll_std_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).std().fillna(0)

        # Target: kg_reales h weeks ahead
        weekly['target'] = weekly['kg_reales'].shift(-horizon)

        all_season_frames.append(weekly)

    df_all = pd.concat(all_season_frames, ignore_index=True)

    # Drop rows with NaN from lagging or target shift
    df_all = df_all.dropna().reset_index(drop=True)

    # Feature columns
    exclude = ['week_idx', 'semana', 'kg_reales', 'temporada', 'target']
    feature_cols = [c for c in df_all.columns if c not in exclude]

    # Split: T13-T15 train, T16 validation, T17 test
    train_df = df_all[df_all['temporada'].isin(['T13', 'T14', 'T15'])].reset_index(drop=True)
    val_df = df_all[df_all['temporada'] == 'T16'].reset_index(drop=True)
    test_df = df_all[df_all['temporada'] == 'T17'].reset_index(drop=True)

    print(f'  Invernadero {invernadero_id}:')
    print(f'    Train samples: {len(train_df)} (T13-T15)')
    print(f'    Val samples:   {len(val_df)} (T16)')
    print(f'    Test samples:  {len(test_df)} (T17)')
    print(f'    Features ({len(feature_cols)}): {feature_cols}')

    return train_df, val_df, test_df, feature_cols


# ============================================================================
# 2. DATASET
# ============================================================================

class FlatDataset(Dataset):
    """Simple dataset for flat feature vectors (no sequences)."""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx].unsqueeze(0)


# ============================================================================
# 3. FFNN MODEL DEFINITION
# ============================================================================

class FFNN(nn.Module):
    """
    Feed-forward neural network: Input → Hidden(64, ReLU) → Output(1).
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 4. HYPERPARAMETER CONFIGURATION
# ============================================================================

HYPERPARAMS = {
    # Data
    'horizon': 4,              # Prediction horizon: 4 weeks ahead
    'batch_size': 4,

    # FFNN
    'hidden_dim': 64,

    # PCA
    'pca_variance': 0.95,      # Retain 95% of explained variance

    # Training
    'dropout': 0.3,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 500,
    'patience': 60,

    # Reproducibility
    'seed': 42,
}


# ============================================================================
# 5. TRAINING
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(invernadero_id, hp):
    """Load data, normalize, apply PCA, create DataLoaders."""
    train_df, val_df, test_df, feature_cols = build_dataset_for_greenhouse(
        invernadero_id, horizon=hp['horizon']
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['target'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['target'].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['target'].values.astype(np.float32)

    # Min-Max normalization to [0, 1]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_val = scaler_X.transform(X_val)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # PCA dimensionality reduction
    pca = PCA(n_components=hp['pca_variance'])
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    n_components = pca.n_components_
    print(f'  PCA: {len(feature_cols)} features -> {n_components} components '
          f'({hp["pca_variance"]*100:.0f}% variance explained)')

    train_ds = FlatDataset(X_train, y_train)
    val_ds = FlatDataset(X_val, y_val)
    test_ds = FlatDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler_X, scaler_y, feature_cols, n_components


def train_model(model, train_loader, val_loader, hp, model_path):
    """Train the FFNN with early stopping on validation loss."""
    criterion = nn.MSELoss()
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
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train_loss:.6f} | '
                  f'Val: {avg_val_loss:.6f} | '
                  f'Best Val: {best_val_loss:.6f}')

        if patience_counter >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, train_losses, val_losses


# ============================================================================
# 6. TESTING & EVALUATION
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute RMSE, R², NSE, PBIAS, MAPE."""
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
    """Run inference on test set, inverse-transform predictions, compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred_norm = np.concatenate(all_preds).flatten()
    y_true_norm = np.concatenate(all_targets).flatten()

    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()

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
    plt.savefig(RESULTS_DIR / 'ffnn_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results plot saved to {RESULTS_DIR / "ffnn_results.png"}')


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
    hp = HYPERPARAMS
    set_seed(hp['seed'])

    print('=' * 60)
    print('  FFNN Greenhouse Yield Prediction')
    print(f'  Hidden: {hp["hidden_dim"]} | PCA: {hp["pca_variance"]*100:.0f}%')
    print(f'  Train: T13-T15 | Val: T16 | Test: T17 | Horizon: {hp["horizon"]} semanas')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id}')
        print(f'{"─" * 60}')

        print('\n  [1/4] Cargando y preparando datos...')
        train_loader, val_loader, test_loader, scaler_X, scaler_y, feature_cols, n_components = \
            prepare_data(inv_id, hp)

        input_dim = n_components
        print(f'  Input dimension (after PCA): {input_dim}')

        print('\n  [2/4] Construyendo modelo FFNN...')
        model = FFNN(
            input_dim=input_dim,
            hidden_dim=hp['hidden_dim'],
            dropout=hp['dropout'],
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        print(f'  Model parameters: {total_params:,}')

        print('\n  [3/4] Entrenando...')
        model_path = RESULTS_DIR / f'best_ffnn_inv{inv_id}.pt'
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, hp, model_path
        )

        print('\n  [4/4] Evaluando en test (T17)...')
        y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y)
        print_metrics(inv_id, metrics)

        all_results[inv_id] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

        metrics_row = {'invernadero': inv_id, **metrics}
        all_metrics.append(metrics_row)

    # Plot all results
    print('\nGenerando gráficas...')
    plot_results_per_greenhouse(all_results)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'ffnn_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "ffnn_metrics.csv"}')

    # Summary table
    print('\n' + '=' * 70)
    print('  RESUMEN FFNN - Predicción a 4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
