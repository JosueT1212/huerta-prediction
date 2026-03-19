"""
ARIMAX Model for Greenhouse Crop Yield Prediction
==================================================
Uses statsmodels SARIMAX to fit an ARIMAX model where:
  - Endogenous variable: weekly kg production (autoregressive lags 1-2)
  - Exogenous variables: weekly-aggregated sensor/weather features reduced via PCA

Pipeline (same as other Models scripts):
  1. Load internal + external sensor data and production targets
  2. Aggregate daily → weekly, merge with kg production
  3. Apply MinMax scaling + PCA on environmental features
  4. Fit ARIMAX per season split (Train: T13-T15, Val: T16, Test: T17)
  5. Evaluate with RMSE, R², NSE, PBIAS, MAPE

The ARIMAX order (p,d,q) uses p=2 to capture the 1-2 week autoregressive
structure requested. d and q are tuned via AIC on validation data.
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

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
# 2. COLUMN RENAMING
# ============================================================================

def rename_columns(df_int, df_ext):
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


# ============================================================================
# 3. WEEKLY DATASET BUILDER (with PCA on exogenous variables)
# ============================================================================

def build_weekly_dataset(invernadero_id):
    """
    Build weekly dataset for ARIMAX.

    Returns a DataFrame with columns:
      - kg_reales: endogenous variable (weekly production)
      - pc1, pc2, ...: PCA components of environmental features (exogenous)
      - temporada: season label for splitting

    Also returns the PCA and scaler objects for reference.
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

    env_features = int_features + ext_features
    all_season_frames = []

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

        # Aggregate daily → weekly
        agg_dict = {feat: 'mean' for feat in env_features}
        if 'rad_sum' in agg_dict:
            agg_dict['rad_sum'] = 'sum'

        weekly_env = daily.groupby('week_idx').agg(agg_dict).reset_index()

        # Align with production
        n_weeks = min(len(weekly_env), len(kg_season))
        weekly_env = weekly_env.iloc[:n_weeks].copy()
        kg_vals = kg_season.iloc[:n_weeks].copy()

        weekly = weekly_env.copy()
        weekly['kg_reales'] = kg_vals['kg_reales'].values
        weekly['temporada'] = temp
        weekly['semana'] = kg_vals['semana'].values

        all_season_frames.append(weekly)

    df_all = pd.concat(all_season_frames, ignore_index=True)

    # Forward-fill NaN in environmental columns
    df_all[env_features] = df_all[env_features].ffill().bfill()

    # Filter: remove rows where kg_reales is NaN or zero (no production data)
    df_all = df_all.dropna(subset=['kg_reales']).reset_index(drop=True)

    # Split for fitting scaler/PCA on training data only
    train_mask = df_all['temporada'].isin(['T13', 'T14', 'T15'])

    # MinMax scaling on environmental features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(df_all.loc[train_mask, env_features].values)

    env_scaled = scaler_X.transform(df_all[env_features].values)

    # PCA: retain 95% variance
    pca = PCA(n_components=0.95)
    pca.fit(env_scaled[train_mask.values])

    n_components = pca.n_components_
    pca_cols = [f'pc{i+1}' for i in range(n_components)]

    print(f'    PCA: {len(env_features)} features → {n_components} components '
          f'({pca.explained_variance_ratio_.sum()*100:.1f}% varianza explicada)')

    pca_values = pca.transform(env_scaled)
    for j, col in enumerate(pca_cols):
        df_all[col] = pca_values[:, j]

    return df_all, pca_cols, env_features, scaler_X, pca


# ============================================================================
# 4. ARIMAX ORDER SELECTION
# ============================================================================

HYPERPARAMS = {
    'max_p': 2,        # max AR order (1-2 weeks autoregressive)
    'd_range': [0, 1],  # differencing orders to try
    'max_q': 2,        # max MA order
    'seed': 42,
}


def select_arimax_order(endog_train, exog_train, endog_val, exog_val, hp):
    """
    Grid search over ARIMAX(p,d,q) orders using AIC on training data,
    then pick the model with lowest RMSE on validation data.

    Returns the best (p, d, q) tuple.
    """
    p_range = range(1, hp['max_p'] + 1)  # at least p=1 for autoregressive
    d_range = hp['d_range']
    q_range = range(0, hp['max_q'] + 1)

    best_order = (2, 0, 0)
    best_val_rmse = float('inf')

    print('    Buscando mejor orden ARIMAX(p,d,q)...')

    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = SARIMAX(
                endog_train,
                exog=exog_train,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)

            # Forecast on validation
            forecast = result.forecast(steps=len(endog_val), exog=exog_val)
            val_rmse = np.sqrt(mean_squared_error(endog_val, forecast))

            print(f'      ARIMAX({p},{d},{q}) → AIC={result.aic:.1f}, '
                  f'Val RMSE={val_rmse:.2f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_order = (p, d, q)

        except Exception:
            continue

    print(f'    → Mejor orden: ARIMAX{best_order} (Val RMSE={best_val_rmse:.2f})')
    return best_order


# ============================================================================
# 5. TRAINING & EVALUATION
# ============================================================================

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


def train_and_evaluate(invernadero_id, hp):
    """
    Full ARIMAX pipeline for one greenhouse:
      1. Build weekly data with PCA exogenous features
      2. Split into train/val/test
      3. Select best ARIMAX order on validation
      4. Refit on train+val, forecast on test
    """
    print(f'\n  [1/4] Cargando y preparando datos semanales...')
    df_all, pca_cols, env_features, scaler_X, pca = build_weekly_dataset(invernadero_id)

    # Split by season
    train_df = df_all[df_all['temporada'].isin(['T13', 'T14', 'T15'])].reset_index(drop=True)
    val_df = df_all[df_all['temporada'] == 'T16'].reset_index(drop=True)
    test_df = df_all[df_all['temporada'] == 'T17'].reset_index(drop=True)

    print(f'    Train: {len(train_df)} semanas (T13-T15)')
    print(f'    Val:   {len(val_df)} semanas (T16)')
    print(f'    Test:  {len(test_df)} semanas (T17)')

    endog_train = train_df['kg_reales'].values
    exog_train = train_df[pca_cols].values
    endog_val = val_df['kg_reales'].values
    exog_val = val_df[pca_cols].values
    endog_test = test_df['kg_reales'].values
    exog_test = test_df[pca_cols].values

    # Step 2: Select best order using validation
    print(f'\n  [2/4] Seleccionando orden ARIMAX...')
    best_order = select_arimax_order(endog_train, exog_train,
                                      endog_val, exog_val, hp)

    # Step 3: Refit on train + val combined for final test forecast
    print(f'\n  [3/4] Entrenando modelo final ARIMAX{best_order} (train+val)...')
    endog_trainval = np.concatenate([endog_train, endog_val])
    exog_trainval = np.concatenate([exog_train, exog_val], axis=0)

    final_model = SARIMAX(
        endog_trainval,
        exog=exog_trainval,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_result = final_model.fit(disp=False, maxiter=500)

    print(f'    AIC: {final_result.aic:.1f}')
    print(f'    BIC: {final_result.bic:.1f}')

    # Step 4: Forecast on test
    print(f'\n  [4/4] Evaluando en test (T17)...')
    y_pred = final_result.forecast(steps=len(endog_test), exog=exog_test)
    y_true = endog_test

    metrics = compute_metrics(y_true, y_pred)

    # Also get in-sample fitted values for train+val (for plotting)
    fitted_trainval = final_result.fittedvalues

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics,
        'order': best_order,
        'aic': final_result.aic,
        'bic': final_result.bic,
        'fitted_trainval': fitted_trainval,
        'endog_trainval': endog_trainval,
        'model_result': final_result,
        'pca_cols': pca_cols,
        'n_train': len(endog_train),
        'n_val': len(endog_val),
    }


# ============================================================================
# 6. PLOTS
# ============================================================================

def plot_results(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        # Left: In-sample fit (train+val)
        n_tv = len(res['endog_trainval'])
        t_tv = np.arange(n_tv)
        axes[i, 0].plot(t_tv, res['endog_trainval'], 'o-', color='steelblue',
                         ms=3, label='Actual (train+val)')
        axes[i, 0].plot(t_tv, res['fitted_trainval'], 's--', color='coral',
                         ms=3, label='Ajuste ARIMAX')
        # Mark train/val boundary
        n_train = res['n_train']
        axes[i, 0].axvline(x=n_train - 0.5, color='gray', ls=':', lw=1,
                            label=f'T15→T16 (sem {n_train})')
        axes[i, 0].set_title(f'Inv. {inv_id} - Ajuste in-sample '
                              f'ARIMAX{res["order"]}')
        axes[i, 0].set_xlabel('Semana (train+val)')
        axes[i, 0].set_ylabel('Producción (kg)')
        axes[i, 0].legend(fontsize=8)

        info = f'AIC: {res["aic"]:.1f}\nBIC: {res["bic"]:.1f}'
        axes[i, 0].text(0.02, 0.98, info, transform=axes[i, 0].transAxes,
                         va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # Right: Test forecast
        t_test = np.arange(len(res['y_true']))
        axes[i, 1].plot(t_test, res['y_true'], 'o-', color='steelblue',
                         ms=5, label='Actual')
        axes[i, 1].plot(t_test, res['y_pred'], 's--', color='coral',
                         ms=5, label=f'Predicción ARIMAX{res["order"]}')
        axes[i, 1].set_title(f'Inv. {inv_id} - Predicción (T17)')
        axes[i, 1].set_xlabel('Semana de test (T17)')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        txt = '\n'.join(f'{k}: {v:.4f}' for k, v in res['metrics'].items())
        axes[i, 1].text(0.02, 0.98, txt, transform=axes[i, 1].transAxes,
                         va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = RESULTS_DIR / 'arimax_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {path}')


def print_metrics(inv_id, metrics, order):
    print(f'\n{"=" * 60}')
    print(f'  INVERNADERO {inv_id} - MÉTRICAS ARIMAX{order} (Test: T17)')
    print(f'{"=" * 60}')
    for k, v in metrics.items():
        print(f'  {k:<15s}: {v:>10.4f}')
    print(f'{"=" * 60}')
    print(f'  >>> MAPE: {metrics["MAPE (%)"]:>10.4f}% <<<')
    print(f'{"=" * 60}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    hp = HYPERPARAMS
    np.random.seed(hp['seed'])

    print('=' * 65)
    print('  ARIMAX — Greenhouse Yield Prediction')
    print('  Endógena: kg_reales (AR lags 1-2)')
    print('  Exógenas: variables ambientales semanales (PCA)')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 65)

    greenhouses = [3, 4]
    all_results = {}
    all_metrics = []

    for inv_id in greenhouses:
        print(f'\n{"─" * 60}')
        print(f'  INVERNADERO {inv_id}')
        print(f'{"─" * 60}')

        res = train_and_evaluate(inv_id, hp)
        print_metrics(inv_id, res['metrics'], res['order'])

        all_results[inv_id] = res
        all_metrics.append({
            'invernadero': inv_id,
            'order': str(res['order']),
            **res['metrics'],
        })

    print('\nGenerando gráficas...')
    plot_results(all_results)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'arimax_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "arimax_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN ARIMAX — Predicción semanal')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
