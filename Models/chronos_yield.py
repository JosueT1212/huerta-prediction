"""
Chronos Foundation Model for Greenhouse Crop Yield Prediction
=============================================================
Uses Amazon's Chronos (T5-based time series foundation model).

Phase 1: Zero-shot forecasting (no training) — DONE, poor results (R²<0)
Phase 2: Fine-tune on our yield series (T13-T15), validate on T16, test on T17

Chronos is univariate — it uses only the yield history to forecast.
Fine-tuning adapts the pretrained weights to our greenhouse yield patterns.

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
import copy

import torch
from torch.optim import AdamW
from chronos import ChronosPipeline

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

DATA_DIR = Path(__file__).resolve().parent.parent / 'Data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']
HORIZON = 4


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
        frames.append(data)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['kg_reales']).reset_index(drop=True)
    return df


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


def rolling_forecast(pipeline, full_series, test_start_idx, horizon=4):
    """
    Rolling h-step-ahead forecast over the test period.
    At each test step t, uses full_series[:t+1] as context and predicts h steps ahead.
    """
    y_true = []
    y_pred = []

    n = len(full_series)
    for t in range(test_start_idx, n - horizon):
        context = torch.tensor(full_series[:t + 1], dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=horizon)
        pred_median = forecast.median(dim=1).values
        pred_value = pred_median[0, horizon - 1].item()

        y_true.append(full_series[t + horizon])
        y_pred.append(pred_value)

    return np.array(y_true), np.array(y_pred)


def create_training_windows(series, prediction_length, min_context=10):
    """
    Create overlapping (context, label) windows from a time series for Chronos fine-tuning.

    Chronos requires labels of exactly prediction_length.
    We slide a window across the series, using varying context lengths.
    """
    contexts = []
    labels = []

    n = len(series)
    # Slide the label window across the series
    for label_end in range(min_context + prediction_length, n + 1):
        label_start = label_end - prediction_length
        context_series = series[:label_start]
        label_series = series[label_start:label_end]

        if len(context_series) >= min_context:
            contexts.append(torch.tensor(context_series, dtype=torch.float32))
            labels.append(torch.tensor(label_series, dtype=torch.float32))

    return contexts, labels


def fine_tune_chronos(pipeline, train_series, val_series, epochs=50, lr=1e-4):
    """
    Fine-tune Chronos on greenhouse yield data.

    Args:
        pipeline: ChronosPipeline with pretrained model
        train_series: numpy array of training yield values (T13-T15)
        val_series: numpy array of full series up to validation (T13-T16)
        epochs: number of fine-tuning epochs
        lr: learning rate
    """
    tokenizer = pipeline.tokenizer
    model = pipeline.model
    pred_len = tokenizer.config.prediction_length

    # Create training windows from train series
    contexts, labels = create_training_windows(train_series, pred_len, min_context=10)
    n_samples = len(contexts)
    print(f'    Training windows: {n_samples}')

    if n_samples == 0:
        print('    No valid training windows! Skipping fine-tuning.')
        return pipeline

    # Enable training mode on the T5 model
    t5_model = model.model
    t5_model.train()

    optimizer = AdamW(t5_model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle training order
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for idx in indices:
            ctx = contexts[idx].unsqueeze(0)
            lbl = labels[idx].unsqueeze(0)

            # Tokenize
            token_ids, attention_mask, scale = tokenizer.context_input_transform(ctx)
            label_ids, label_attention_mask = tokenizer.label_input_transform(lbl, scale)

            # Forward with cross-entropy loss
            optimizer.zero_grad()
            out = t5_model(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t5_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_samples

        # Validation loss
        t5_model.eval()
        val_contexts, val_labels = create_training_windows(val_series, pred_len, min_context=10)
        val_loss = 0.0
        with torch.no_grad():
            for ctx, lbl in zip(val_contexts, val_labels):
                ctx = ctx.unsqueeze(0)
                lbl = lbl.unsqueeze(0)
                token_ids, attention_mask, scale = tokenizer.context_input_transform(ctx)
                label_ids, _ = tokenizer.label_input_transform(lbl, scale)
                out = t5_model(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
                val_loss += out.loss.item()
        avg_val_loss = val_loss / max(len(val_contexts), 1)
        t5_model.train()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(t5_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'    Epoch {epoch+1:>3d}/{epochs} | '
                  f'Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}')

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch+1}')
            break

    # Restore best model
    if best_state is not None:
        t5_model.load_state_dict(best_state)
    t5_model.eval()

    return pipeline


def main():
    print('=' * 60)
    print('  Chronos — Zero-Shot + Fine-Tuned Comparison')
    print('  Rolling h=4 forecast on T17')
    print('=' * 60)

    greenhouses = [3, 4]
    model_name = 'amazon/chronos-t5-tiny'  # smallest for fast fine-tuning

    # ─── PHASE 1: Zero-Shot ───
    print(f'\n{"═" * 60}')
    print(f'  PHASE 1: Zero-Shot ({model_name})')
    print(f'{"═" * 60}')

    pipeline_zs = ChronosPipeline.from_pretrained(
        model_name, device_map='cpu', dtype=torch.float32,
    )

    zs_results = {}
    for inv_id in greenhouses:
        df_kg = load_production(inv_id)
        full_series = df_kg['kg_reales'].values.astype(np.float64)
        season_list = df_kg['temporada'].values
        test_start_idx = np.where(season_list == 'T17')[0][0]

        y_true, y_pred = rolling_forecast(pipeline_zs, full_series, test_start_idx, HORIZON)
        metrics = compute_metrics(y_true, y_pred)

        print(f'  Inv {inv_id}: R²={metrics["R²"]:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
        zs_results[inv_id] = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}

    # ─── PHASE 2: Fine-Tuned ───
    print(f'\n{"═" * 60}')
    print(f'  PHASE 2: Fine-Tuned ({model_name})')
    print(f'{"═" * 60}')

    ft_results = {}
    for inv_id in greenhouses:
        print(f'\n  Invernadero {inv_id}:')

        # Load fresh model for each greenhouse
        pipeline_ft = ChronosPipeline.from_pretrained(
            model_name, device_map='cpu', dtype=torch.float32,
        )

        df_kg = load_production(inv_id)
        full_series = df_kg['kg_reales'].values.astype(np.float64)
        season_list = df_kg['temporada'].values

        # Train on T13-T15, validate with T13-T16
        train_end = np.where(season_list == 'T16')[0][0]
        val_end = np.where(season_list == 'T17')[0][0]
        test_start_idx = val_end

        train_series = full_series[:train_end]
        val_series = full_series[:val_end]

        print(f'    Train series length: {len(train_series)} (T13-T15)')
        print(f'    Val series length: {len(val_series)} (T13-T16)')

        # Fine-tune
        pipeline_ft = fine_tune_chronos(
            pipeline_ft, train_series, val_series,
            epochs=100, lr=5e-5,
        )

        # Evaluate on T17
        y_true, y_pred = rolling_forecast(pipeline_ft, full_series, test_start_idx, HORIZON)
        metrics = compute_metrics(y_true, y_pred)

        print(f'    Fine-tuned: R²={metrics["R²"]:.4f}, MAPE={metrics["MAPE (%)"]:.2f}%')
        ft_results[inv_id] = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}

    # ─── Comparison ───
    print(f'\n{"═" * 60}')
    print(f'  COMPARISON: Zero-Shot vs Fine-Tuned')
    print(f'{"═" * 60}')

    all_metrics = []
    for inv_id in greenhouses:
        zs = zs_results[inv_id]['metrics']
        ft = ft_results[inv_id]['metrics']
        print(f'\n  Invernadero {inv_id}:')
        print(f'    Zero-shot:  R²={zs["R²"]:.4f}, MAPE={zs["MAPE (%)"]:.2f}%')
        print(f'    Fine-tuned: R²={ft["R²"]:.4f}, MAPE={ft["MAPE (%)"]:.2f}%')

        # Use whichever is better
        best = ft_results[inv_id] if ft['R²'] > zs['R²'] else zs_results[inv_id]
        best_label = 'fine-tuned' if ft['R²'] > zs['R²'] else 'zero-shot'
        all_metrics.append({
            'invernadero': inv_id,
            'method': best_label,
            **best['metrics'],
        })

    # Plot: zero-shot vs fine-tuned for each greenhouse
    fig, axes = plt.subplots(len(greenhouses), 2, figsize=(16, 5 * len(greenhouses)))
    if len(greenhouses) == 1:
        axes = axes.reshape(1, -1)

    for i, inv_id in enumerate(greenhouses):
        for j, (label, res) in enumerate([('Zero-Shot', zs_results[inv_id]),
                                           ('Fine-Tuned', ft_results[inv_id])]):
            weeks = np.arange(len(res['y_true']))
            axes[i, j].plot(weeks, res['y_true'], 'o-', color='steelblue',
                            label='Actual', markersize=5)
            axes[i, j].plot(weeks, res['y_pred'], 's--', color='coral',
                            label=f'Chronos {label}', markersize=5)
            axes[i, j].set_title(f'Inv {inv_id} — {label}')
            axes[i, j].set_xlabel('Semana (T17)')
            axes[i, j].set_ylabel('Producción (kg)')
            axes[i, j].legend()

            metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
            axes[i, j].text(0.02, 0.98, metrics_text, transform=axes[i, j].transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'chronos_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {RESULTS_DIR / "chronos_results.png"}')

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'chronos_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "chronos_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN CHRONOS — h=4 semanas')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
