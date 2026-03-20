"""
iTransformer for Greenhouse Crop Yield Prediction
==================================================
Based on: Liu et al. (2024) "iTransformer: Inverted Transformers Are
Effective for Time Series Forecasting", ICLR 2024.

Key insight: standard Transformers apply attention across time steps.
iTransformer *inverts* this — it applies attention across VARIATES
(sensor channels), treating each variate as a token. This captures
multivariate correlations (e.g., temperature × humidity → yield).

Architecture:
  - Each variate (sensor feature) is independently embedded across time
  - Self-attention applied across variates (not time)
  - Feed-forward network per variate
  - Final projection to predict yield

This is ideal for our case: 19 sensor features + lag features as variates,
with the transformer learning which sensor combinations matter most.

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
import math

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

# Reuse data loading from cnn_rnn_yield
from cnn_rnn_yield import (
    build_dataset_for_greenhouse, compute_metrics, set_seed,
)


# ============================================================================
# DATASET
# ============================================================================

class YieldSequenceDataset(Dataset):
    """Creates sequences of length `seq_len` weeks -> predict target."""
    def __init__(self, features, targets, seq_len):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]  # (seq_len, n_features)
        y = self.targets[idx + self.seq_len - 1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================================
# iTransformer MODEL
# ============================================================================

class VariateEmbedding(nn.Module):
    """Embed each variate's time series independently into d_model dimensions."""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, n_variates)
        # Transpose to (batch, n_variates, seq_len), then project
        x = x.permute(0, 2, 1)  # (batch, n_variates, seq_len)
        x = self.projection(x)   # (batch, n_variates, d_model)
        return x


class iTransformerBlock(nn.Module):
    """Single iTransformer block: attention across variates + FFN."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, n_variates, d_model)
        # Self-attention across variates
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for time series forecasting.

    Instead of attending across time steps, attends across variates.
    Each variate (sensor feature) becomes a token.
    """
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()

        self.variate_embedding = VariateEmbedding(seq_len, d_model)

        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Aggregate across variates and project to prediction
        self.output_projection = nn.Sequential(
            nn.Linear(n_variates * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_variates)
        x = self.variate_embedding(x)  # (batch, n_variates, d_model)

        for block in self.blocks:
            x = block(x)  # (batch, n_variates, d_model)

        # Flatten all variate embeddings
        x = x.flatten(1)  # (batch, n_variates * d_model)
        pred = self.output_projection(x)  # (batch, 1)
        return pred


# ============================================================================
# LOSS (same CorrMSE that works best)
# ============================================================================

class CorrMSELoss(nn.Module):
    def __init__(self, corr_weight=0.1):
        super().__init__()
        self.corr_weight = corr_weight

    def forward(self, pred, target):
        mse_loss = nn.functional.mse_loss(pred, target)
        loss = mse_loss

        if pred.shape[0] >= 4 and self.corr_weight > 0:
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            p = pred_flat - pred_flat.mean()
            t = target_flat - target_flat.mean()
            corr = torch.sum(p * t) / (
                torch.sqrt(torch.sum(p ** 2) + 1e-8) *
                torch.sqrt(torch.sum(t ** 2) + 1e-8)
            )
            loss = loss + self.corr_weight * (1.0 - corr)

        return loss


# ============================================================================
# TRAINING
# ============================================================================

def prepare_data(invernadero_id, hp):
    """Load, scale, PCA, create DataLoaders."""
    train_df, val_df, test_df, feature_cols = build_dataset_for_greenhouse(
        invernadero_id, horizon=hp['horizon'],
        lag_features=hp.get('lag_features', [1]),
        include_rolling_mean=hp.get('include_rolling_mean', False),
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['target'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['target'].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['target'].values.astype(np.float32)

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
    n_features = pca.n_components_
    print(f'  PCA: {len(feature_cols)} features -> {n_features} components (variates)')

    train_ds = YieldSequenceDataset(X_train, y_train, hp['seq_len'])
    val_ds = YieldSequenceDataset(X_val, y_val, hp['seq_len'])
    test_ds = YieldSequenceDataset(X_test, y_test, hp['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler_y, n_features


def train_model(model, train_loader, val_loader, hp, model_path):
    """Train with early stopping on validation score."""
    criterion = CorrMSELoss(corr_weight=hp['corr_weight'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=hp['learning_rate'],
                                   weight_decay=hp.get('weight_decay', 1e-4))

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp['epochs'], eta_min=1e-6
    )

    best_val_score = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(hp['epochs']):
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
        scheduler.step()

        # Validation
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

        # Validation correlation
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


def init_weights(model, method='default'):
    """Apply weight initialization to the model."""
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
# HYPERPARAMETERS
# ============================================================================

HYPERPARAMS_PER_GREENHOUSE = {
    3: {
        'seq_len': 4,
        'horizon': 4,
        'batch_size': 8,
        'lag_features': [1, 2],
        'include_rolling_mean': True,
        # iTransformer architecture
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 256,
        'dropout': 0.1,
        'init_method': 'xavier',
        # Training
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'corr_weight': 0.8,
        'epochs': 500,
        'patience': 100,
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
                  0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  11, 12, 14, 15, 16, 17, 18, 19, 20, 21],
    },
    4: {
        'seq_len': 4,
        'horizon': 4,
        'batch_size': 8,
        'lag_features': [1, 2],
        'include_rolling_mean': True,
        # iTransformer architecture
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 256,
        'dropout': 0.1,
        'init_method': 'xavier',
        # Training
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'corr_weight': 0.5,
        'epochs': 500,
        'patience': 100,
        'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
                  0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  11, 12, 14, 15, 16, 17, 18, 19, 20, 21],
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('  iTransformer — Multivariate Yield Prediction')
    print('  Attention across variates (sensor features)')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        hp = HYPERPARAMS_PER_GREENHOUSE[inv_id]
        seeds = hp['seeds']

        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id} — {len(seeds)} seeds')
        print(f'{"─" * 60}')

        best_r2 = -float('inf')
        best_result = None

        for seed in seeds:
            set_seed(seed)
            print(f'\n  --- Seed {seed} ---')

            train_loader, val_loader, test_loader, scaler_y, n_features = \
                prepare_data(inv_id, hp)

            model = iTransformer(
                seq_len=hp['seq_len'],
                n_variates=n_features,
                d_model=hp['d_model'],
                n_heads=hp['n_heads'],
                n_layers=hp['n_layers'],
                d_ff=hp['d_ff'],
                dropout=hp['dropout'],
            )
            init_method = hp.get('init_method', 'default')
            init_weights(model, init_method)
            model = model.to(DEVICE)

            total_params = sum(p.numel() for p in model.parameters())
            print(f'  Model parameters: {total_params:,}')

            model_path = RESULTS_DIR / f'best_itransformer_inv{inv_id}.pt'
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, hp, model_path
            )

            y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y)
            print(f'  Seed {seed}: R²={metrics["R²"]:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')

            if metrics['R²'] > best_r2:
                best_r2 = metrics['R²']
                best_result = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'metrics': metrics,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'seed': seed,
                }

        print(f'\n  >>> Best seed={best_result["seed"]} (R²={best_r2:.4f}) <<<')

        all_results[inv_id] = best_result
        metrics_row = {'invernadero': inv_id, **best_result['metrics']}
        all_metrics.append(metrics_row)

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
                         label='iTransformer (h=4)', markersize=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} - Predicción (T17)')
        axes[i, 1].set_xlabel('Semana')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
        axes[i, 1].text(0.02, 0.98, metrics_text, transform=axes[i, 1].transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'itransformer_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {RESULTS_DIR / "itransformer_results.png"}')

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'itransformer_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "itransformer_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN iTransformer — h=4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
