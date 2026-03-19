"""
CNN-RNN Hyperparameter Optimization with Optuna
================================================
Uses Bayesian optimization (TPE) to search for optimal hyperparameters
per greenhouse. Reuses model/data infrastructure from cnn_rnn_yield.py.

Search space:
  - Architecture: cnn_filters, num_cnn_blocks, lstm_hidden, lstm_layers, fc_hidden
  - Training: learning_rate, dropout, corr_weight, batch_size
  - Features: seq_len, lag config, rolling mean
  - Seed

Objective: maximize validation R² (proper ML practice — no test leakage).
After optimization, retrains best config and evaluates on test (T17).
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState

# Reuse everything from the main script
from cnn_rnn_yield import (
    DEVICE, DATA_DIR, RESULTS_DIR, SEASON_NAMES,
    build_dataset_for_greenhouse, YieldSequenceDataset,
    CNNRNN, CorrMSELoss, set_seed,
    compute_metrics, evaluate_model, plot_results_per_greenhouse, print_metrics,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

N_TRIALS = 60          # Number of Optuna trials per greenhouse
EPOCHS_PER_TRIAL = 300  # Reduced epochs for faster search
PATIENCE = 80           # Reduced patience for faster search


def prepare_data_from_hp(invernadero_id, hp):
    """Prepare data loaders from hyperparameter dict (same as prepare_data but importable)."""
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

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    n_components = pca.n_components_

    train_ds = YieldSequenceDataset(X_train, y_train, hp['seq_len'])
    val_ds = YieldSequenceDataset(X_val, y_val, hp['seq_len'])
    test_ds = YieldSequenceDataset(X_test, y_test, hp['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler_X, scaler_y, n_components


def train_for_optuna(model, train_loader, val_loader, hp, trial=None):
    """
    Train model and return best validation R².
    Supports Optuna pruning: reports intermediate val R² so bad trials die early.
    """
    criterion = CorrMSELoss(corr_weight=hp['corr_weight'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['learning_rate'],
                                 weight_decay=hp.get('weight_decay', 0))

    best_val_score = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS_PER_TRIAL):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            if torch.isnan(loss):
                return -10.0  # NaN → terrible score
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        all_val_preds = []
        all_val_targets = []
        val_loss = 0.0
        val_batches = 0
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

        # Validation correlation
        val_preds_cat = torch.cat(all_val_preds).flatten()
        val_targets_cat = torch.cat(all_val_targets).flatten()
        vp = val_preds_cat - val_preds_cat.mean()
        vt = val_targets_cat - val_targets_cat.mean()
        val_corr = (torch.sum(vp * vt) / (
            torch.sqrt(torch.sum(vp ** 2) + 1e-8) *
            torch.sqrt(torch.sum(vt ** 2) + 1e-8)
        )).item()

        # Validation R²
        val_r2 = 1.0 - (torch.sum((val_preds_cat - val_targets_cat) ** 2) /
                         torch.sum((val_targets_cat - val_targets_cat.mean()) ** 2)).item()

        # Combined score for early stopping
        val_score = avg_val_loss - hp['corr_weight'] * val_corr

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_r2 = val_r2
        else:
            patience_counter += 1

        # Optuna pruning: report every 20 epochs
        if trial is not None and (epoch + 1) % 20 == 0:
            trial.report(val_r2, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if patience_counter >= PATIENCE:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return best_val_r2


def create_objective(invernadero_id):
    """Create an Optuna objective function for a specific greenhouse."""

    def objective(trial):
        # --- Sample hyperparameters ---
        seed = trial.suggest_int('seed', 0, 9999)
        seq_len = trial.suggest_int('seq_len', 1, 4)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

        # Feature engineering
        use_lag_2 = trial.suggest_categorical('use_lag_2', [True, False])
        include_rolling_mean = trial.suggest_categorical('include_rolling_mean', [True, False])
        lag_features = [1, 2] if use_lag_2 else [1]

        # Architecture
        model_size = trial.suggest_categorical('model_size', [64, 128])
        num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 1, 3)
        lstm_layers = trial.suggest_int('lstm_layers', 1, 3)

        # Training
        learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.2)
        corr_weight = trial.suggest_float('corr_weight', 0.1, 1.5)

        hp = {
            'seq_len': seq_len,
            'horizon': 4,
            'batch_size': batch_size,
            'lag_features': lag_features,
            'include_rolling_mean': include_rolling_mean,
            'cnn_filters': model_size,
            'cnn_kernel_size': 2,
            'cnn_padding': 1,  # padding=1 preserves temporal dimension
            'num_cnn_blocks': num_cnn_blocks,
            'lstm_hidden': model_size,
            'lstm_layers': lstm_layers,
            'fc_hidden': model_size,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': 0,
            'corr_weight': corr_weight,
        }

        set_seed(seed)

        try:
            train_loader, val_loader, test_loader, scaler_X, scaler_y, n_components = \
                prepare_data_from_hp(invernadero_id, hp)

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

            val_r2 = train_for_optuna(model, train_loader, val_loader, hp, trial)
            return val_r2
        except Exception:
            return -10.0

    return objective


def retrain_best(invernadero_id, best_params):
    """Retrain with best params (full epochs + patience) and evaluate on test."""
    lag_features = [1, 2] if best_params['use_lag_2'] else [1]
    model_size = best_params['model_size']

    hp = {
        'seq_len': best_params['seq_len'],
        'horizon': 4,
        'batch_size': best_params['batch_size'],
        'lag_features': lag_features,
        'include_rolling_mean': best_params['include_rolling_mean'],
        'cnn_filters': model_size,
        'cnn_kernel_size': 2,
        'cnn_padding': 1,
        'num_cnn_blocks': best_params['num_cnn_blocks'],
        'lstm_hidden': model_size,
        'lstm_layers': best_params['lstm_layers'],
        'fc_hidden': model_size,
        'dropout': best_params['dropout'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': 0,
        'corr_weight': best_params['corr_weight'],
        'epochs': 500,
        'patience': 150,
    }

    set_seed(best_params['seed'])
    train_loader, val_loader, test_loader, scaler_X, scaler_y, n_components = \
        prepare_data_from_hp(invernadero_id, hp)

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model parameters: {total_params:,}')

    # Import full training function
    from cnn_rnn_yield import train_model
    model_path = RESULTS_DIR / f'best_cnn_rnn_inv{invernadero_id}.pt'
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, hp, model_path)

    y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y)

    return y_true, y_pred, metrics, train_losses, val_losses


def main():
    print('=' * 60)
    print('  CNN-RNN Hyperparameter Optimization with Optuna')
    print(f'  {N_TRIALS} trials per greenhouse (TPE + pruning)')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        print(f'\n{"=" * 60}')
        print(f'  OPTIMIZING INVERNADERO {inv_id}')
        print(f'{"=" * 60}')

        # Create and run Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=40),
            study_name=f'inv{inv_id}_optimization',
        )

        study.optimize(
            create_objective(inv_id),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        # Results summary
        best_trial = study.best_trial
        print(f'\n  Best trial #{best_trial.number}:')
        print(f'    Val R²: {best_trial.value:.4f}')
        print(f'    Params:')
        for key, val in best_trial.params.items():
            print(f'      {key}: {val}')

        # Count completed/pruned
        completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
        print(f'\n  Trials: {completed} completed, {pruned} pruned')

        # Retrain best config with full epochs
        print(f'\n  Retraining best config with full epochs...')
        y_true, y_pred, metrics, train_losses, val_losses = retrain_best(inv_id, best_trial.params)
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

        # Save best params
        params_df = pd.DataFrame([best_trial.params])
        params_df.to_csv(RESULTS_DIR / f'optuna_best_params_inv{inv_id}.csv', index=False)

    # Plot results
    print('\nGenerando gráficas...')
    plot_results_per_greenhouse(all_results)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "metrics.csv"}')

    # Summary
    print('\n' + '=' * 70)
    print('  RESUMEN OPTUNA - Predicción a 4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
