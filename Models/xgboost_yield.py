"""
XGBoost Model for Greenhouse Crop Yield Prediction
===================================================
Tree-based baseline to compare against the CNN-RNN approach.

XGBoost receives flat feature vectors (no sequences) and does not
require normalization. Each row is an independent sample with 24
environmental + lag features → target (kg 4 weeks ahead).

Trains one model per greenhouse (Inv. 3, Inv. 4) using all 5 seasons.
Split: T13-T15 train, T16 validation, T17 test.
Prediction horizon: 4 weeks ahead.

Sections:
  1. Data Loading & Feature Engineering (same as CNN-RNN)
  2. Hyperparameter Configuration
  3. Training with XGBRegressor + early stopping on val (T16)
  4. Testing & Evaluation (RMSE, R², NSE, PBIAS, MAPE)
  5. Plots: prediction vs actual + feature importance
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

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
# 2. HYPERPARAMETER CONFIGURATION
# ============================================================================

HYPERPARAMS = {
    'horizon': 4,
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 30,
    'seed': 42,
}


# ============================================================================
# 3. TRAINING
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


def train_and_evaluate(invernadero_id, hp):
    """Train XGBoost and evaluate on test set for one greenhouse."""
    train_df, val_df, test_df, feature_cols = build_dataset_for_greenhouse(
        invernadero_id, horizon=hp['horizon']
    )

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    model = xgb.XGBRegressor(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=hp['learning_rate'],
        subsample=hp['subsample'],
        colsample_bytree=hp['colsample_bytree'],
        reg_alpha=hp['reg_alpha'],
        reg_lambda=hp['reg_lambda'],
        random_state=hp['seed'],
        verbosity=0,
        early_stopping_rounds=hp['early_stopping_rounds'],
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f'  Best iteration: {best_iter}')

    # Predictions
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Get training history for plotting
    evals_result = model.evals_result()
    train_rmse = evals_result['validation_0']['rmse']
    val_rmse = evals_result['validation_1']['rmse']

    return {
        'model': model,
        'y_true': y_test,
        'y_pred': y_pred,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
    }


# ============================================================================
# 4. PLOTS
# ============================================================================

def plot_results(results):
    """Plot training loss, predictions, and feature importance for each greenhouse."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(20, 5 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        # Training & validation RMSE
        axes[i, 0].plot(res['train_rmse'], color='steelblue', linewidth=1, label='Train')
        axes[i, 0].plot(res['val_rmse'], color='coral', linewidth=1, label='Validación (T16)')
        axes[i, 0].set_title(f'Invernadero {inv_id} - RMSE por iteración')
        axes[i, 0].set_xlabel('Boosting Round')
        axes[i, 0].set_ylabel('RMSE')
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

        # Feature importance
        model = res['model']
        importance = model.feature_importances_
        feature_cols = res['feature_cols']
        sorted_idx = np.argsort(importance)
        axes[i, 2].barh(range(len(sorted_idx)), importance[sorted_idx], color='steelblue')
        axes[i, 2].set_yticks(range(len(sorted_idx)))
        axes[i, 2].set_yticklabels([feature_cols[j] for j in sorted_idx], fontsize=8)
        axes[i, 2].set_title(f'Invernadero {inv_id} - Feature Importance')
        axes[i, 2].set_xlabel('Importance (gain)')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgboost_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results plot saved to {RESULTS_DIR / "xgboost_results.png"}')


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
    np.random.seed(hp['seed'])

    print('=' * 60)
    print('  XGBoost Greenhouse Yield Prediction')
    print(f'  Train: T13-T15 | Val: T16 | Test: T17 | Horizon: {hp["horizon"]} semanas')
    print('=' * 60)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id}')
        print(f'{"─" * 60}')

        print('\n  [1/2] Cargando datos y entrenando XGBoost...')
        res = train_and_evaluate(inv_id, hp)

        print('\n  [2/2] Evaluando en test (T17)...')
        print_metrics(inv_id, res['metrics'])

        all_results[inv_id] = res
        all_metrics.append({'invernadero': inv_id, **res['metrics']})

    # Plot all results
    print('\nGenerando gráficas...')
    plot_results(all_results)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'xgboost_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "xgboost_metrics.csv"}')

    # Summary table
    print('\n' + '=' * 70)
    print('  RESUMEN XGBoost - Predicción a 4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
