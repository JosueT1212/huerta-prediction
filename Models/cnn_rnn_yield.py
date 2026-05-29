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
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
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

HORIZON = 4   # fixed prediction horizon (weeks); seq_len = gap - HORIZON + skip_first_weeks
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']

# Phenology feature columns (from Monitoreo de Fenologia files)
PHENO_FEATURE_COLS = [
    'RACIMOS PUESTOS',
    'FLORES EN RACIMO ABIERTAS',
    'CANTIDAD DE RACIMOS EN PLANTA',
    'CANTIDAD DE TOMATES',
    'Nº DE RACIMO EN COSECHA',
    'TOMATES MADUROS (COLOR 2)',
    'DIAMETRO DEL FRUTO cm',
    'CRECIMIENTO PLANTA (cm)',
]
PHENO_FEATURE_NAMES = set(PHENO_FEATURE_COLS)

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================


def load_phenology_features(invernadero_id):
    frames = []
    for i, season_num in enumerate(range(13, 18)):
        filepath = DATA_DIR / f'Monitoreo de Fenologia - Temporada {season_num}.xlsx'
        if not filepath.exists():
            continue
        df = pd.read_excel(filepath, sheet_name='BD TOMATE', header=1)
        df['INVERNADERO'] = pd.to_numeric(df['INVERNADERO'], errors='coerce')
        df = df[df['INVERNADERO'] == invernadero_id].copy()
        present = [c for c in PHENO_FEATURE_COLS if c in df.columns]
        for col in present:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        week_col = 'SEMANA DEL AÑO'
        weekly = df.groupby(week_col)[present].mean().reset_index()
        weekly = weekly.rename(columns={week_col: 'semana'})
        weekly['temporada'] = SEASON_NAMES[i]
        frames.append(weekly)
    if not frames:
        return pd.DataFrame(columns=['temporada', 'semana'] + PHENO_FEATURE_COLS)
    return pd.concat(frames, ignore_index=True)


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
                                 train_seasons=None, val_season=None,
                                 early_season_lag=0, resolution='weekly',
                                 alignment='iso', sensor_lead_weeks=None,
                                 seq_len=None):
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
    df_pheno = load_phenology_features(invernadero_id)
    pheno_feat_cols = [c for c in PHENO_FEATURE_COLS if c in df_pheno.columns]
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
    all_sensor_season_frames = []

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

        # Assign ISO calendar week key (year*100 + iso_week) so sensor aggregation
        # aligns with the calendar-week semana column in the production file.
        daily = daily.sort_values('fecha').reset_index(drop=True)
        _iso = daily['fecha'].dt.isocalendar()
        daily['iso_year'] = _iso['year'].values.astype(int)
        daily['iso_week'] = _iso['week'].values.astype(int)
        daily['week_key'] = daily['iso_year'] * 100 + daily['iso_week']

        # Add dias_desde_transplante
        tp_key = (temp, invernadero_id)
        if tp_key in transplant_dates:
            tp_date = transplant_dates[tp_key]
            daily['dias_desde_transplante'] = (daily['fecha'] - tp_date).dt.days

        # Aggregate daily -> weekly (by calendar week)
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

        if alignment == 'positional':
            # ── POSITIONAL alignment (intentional old behavior, correct sensor dates) ──
            # Build weekly_env with ISO aggregation so sensor dates are correct
            # (June/July start of season), then pair with production by position:
            # sensor row 0 (June/July) → production row 0 (September/October).
            # This intentionally feeds the model pre-production vegetative sensor
            # data for the first production weeks.
            weekly_env = daily.groupby('week_key').agg(agg_dict).reset_index()
            weekly_env = weekly_env.sort_values('week_key').reset_index(drop=True)

            # Build full sensor frame (all sensor rows including pre-production weeks)
            _sf = weekly_env.copy()
            _sf['temporada'] = temp
            _sf['week_in_season'] = np.arange(len(_sf))
            if pheno_feat_cols:
                _pheno_s = df_pheno[df_pheno['temporada'] == temp][['semana'] + pheno_feat_cols].copy()
                _pheno_s = _pheno_s.rename(columns={'semana': '_sensor_semana'})
                _sf['_sensor_semana'] = (_sf['week_key'] % 100).astype(int)
                _sf = _sf.merge(_pheno_s, on='_sensor_semana', how='left')
                _sf = _sf.drop(columns=['_sensor_semana'])
                for col in pheno_feat_cols:
                    _sf[col] = _sf[col].ffill().bfill().fillna(0)
            all_sensor_season_frames.append(_sf)

            # Keep production in original Excel order (chronological across year boundary)
            # DO NOT sort by semana — seasons span 52→1, so numeric sort reverses them.
            kg_sorted = kg_season.reset_index(drop=True)
            # Assign correct week_keys to production rows (year-transition safe)
            kg_semanas = kg_sorted['semana'].astype(int).values
            first_s = int(kg_semanas[0])
            match = daily[daily['iso_week'] == first_s]
            start_year = int(match['iso_year'].iloc[0]) if len(match) else int(daily['iso_year'].iloc[0])
            semana_years = []
            cur_year = start_year
            for i, s in enumerate(kg_semanas):
                if i > 0 and s < kg_semanas[i - 1]:
                    cur_year += 1
                semana_years.append(cur_year)
            kg_sorted = kg_sorted.copy()
            kg_sorted['week_key'] = [y * 100 + s for y, s in zip(semana_years, kg_semanas)]

            # Compute sensor offset so that sensor row 0 is `sensor_lead_weeks` before
            # the first production week.  When sensor_lead_weeks is None, start from
            # the very first a
            # vailable sensor week (maximum available lead).
            if sensor_lead_weeks is not None and sensor_lead_weeks > 0:
                wk_to_pos = {wk: i for i, wk in enumerate(weekly_env['week_key'])}
                first_prod_wk = kg_sorted['week_key'].iloc[0]
                prod_start_pos = wk_to_pos.get(first_prod_wk, 0)
                offset = max(0, prod_start_pos - sensor_lead_weeks)
            else:
                offset = 0  # start from the beginning of sensor data

            # Pair by position: sensor week (offset + i) ↔ production week i
            n = min(len(weekly_env) - offset, len(kg_sorted))
            # Document sensor→production gap per season
            _wk2pos = {wk: i for i, wk in enumerate(weekly_env['week_key'])}
            _prod_pos = _wk2pos.get(kg_sorted['week_key'].iloc[0], offset)
            _gap = _prod_pos - offset
            _s_wk = int(weekly_env['week_key'].iloc[offset]) % 100
            _p_wk = int(kg_sorted['week_key'].iloc[0]) % 100
            print(f'    {temp}: sensor ISO sem {_s_wk:02d} → prod ISO sem {_p_wk:02d}  gap={_gap} sem')
            weekly = weekly_env.iloc[offset:offset + n].reset_index(drop=True).copy()
            weekly['sensor_gap'] = _gap
            weekly['kg_reales']      = kg_sorted['kg_reales'].values[:n]
            weekly['semana']         = kg_sorted['semana'].values[:n]
            # Keep production week_key as reference; sensor week_key already in weekly from weekly_env
            weekly['prod_week_key']  = kg_sorted['week_key'].values[:n]
            weekly['temporada']      = temp
            weekly['week_in_season'] = np.arange(n)

            if pheno_feat_cols:
                pheno_season = df_pheno[df_pheno['temporada'] == temp][['semana'] + pheno_feat_cols].copy()
                pheno_season = pheno_season.rename(columns={'semana': '_sensor_semana'})
                weekly['_sensor_semana'] = (weekly['week_key'] % 100).astype(int)
                weekly = weekly.merge(pheno_season, on='_sensor_semana', how='left')
                weekly = weekly.drop(columns=['_sensor_semana'])
                for col in pheno_feat_cols:
                    weekly[col] = weekly[col].ffill().bfill()

            lags = lag_features if lag_features is not None else [1]
            for lag in lags:
                weekly[f'kg_lag_{lag}'] = weekly['kg_reales'].shift(lag)
            if include_rolling_mean:
                weekly['kg_roll_mean_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).mean()

            weekly['target'] = weekly['kg_reales'].shift(-horizon)

            if resolution == 'daily':
                # For daily: each sensor day i → production row i//7.
                # daily rows 0-6 (June/July week 0) → production row 0 (September),
                # inheriting its target and temporal features.
                sensor_daily_cols = [c for c in int_features + ext_features if c in daily.columns]
                for rc in riego_features:
                    if rc in daily.columns:
                        sensor_daily_cols.append(rc)

                # Assign positional week index to daily rows, then apply offset so
                # that daily rows from sensor week (offset) become production week 0.
                daily['pos_week'] = np.arange(len(daily)) // 7
                daily_trunc = daily[
                    (daily['pos_week'] >= offset) &
                    (daily['pos_week'] < offset + n)
                ].copy()
                daily_trunc['pos_week'] = daily_trunc['pos_week'] - offset  # remap to 0…n-1

                weekly_only_pos = [c for c in weekly.columns
                                   if c not in sensor_daily_cols
                                   and c not in ('temporada', 'week_key')]
                weekly['pos_week'] = np.arange(len(weekly))

                daily_frame = pd.merge(
                    daily_trunc[['fecha', 'temporada', 'pos_week'] + sensor_daily_cols],
                    weekly[['pos_week'] + weekly_only_pos],
                    on='pos_week', how='inner'
                ).sort_values('fecha').reset_index(drop=True)
                all_season_frames.append(daily_frame)
            else:
                all_season_frames.append(weekly)

        else:
            # ── ISO calendar alignment (default) ─────────────────────────────
            weekly_env = daily.groupby('week_key').agg(agg_dict).reset_index()

            # Build week_key for the production rows.
            # The semana column is ISO calendar week; the season may span two calendar years
            # (e.g. semana 32-52 in year Y, then semana 1-10 in year Y+1).
            # Detect the year transition whenever semana decreases.
            kg_semanas = kg_season['semana'].astype(int).values
            # Start year = iso_year of the first sensor day whose iso_week == first production semana
            first_s = int(kg_semanas[0])
            match = daily[daily['iso_week'] == first_s]
            start_year = int(match['iso_year'].iloc[0]) if len(match) else int(daily['iso_year'].iloc[0])
            semana_years = []
            cur_year = start_year
            for i, s in enumerate(kg_semanas):
                if i > 0 and s < kg_semanas[i - 1]:   # semana wrapped (52 → 1)
                    cur_year += 1
                semana_years.append(cur_year)
            kg_season = kg_season.copy()
            kg_season['week_key'] = [y * 100 + s for y, s in zip(semana_years, kg_semanas)]

            # Inner join: only keep weeks present in both sensor aggregates and production
            weekly = pd.merge(weekly_env, kg_season[['week_key', 'semana', 'kg_reales']],
                              on='week_key', how='inner').sort_values('week_key').reset_index(drop=True)
            weekly['temporada'] = temp

            # Temporal features
            weekly['week_in_season'] = np.arange(len(weekly))

            # Early-season sensor features: for each production week, include sensor
            # readings from `early_season_lag` weeks earlier (pre-production summer period).
            # Use positional shift in weekly_env (sorted by week_key) to avoid broken
            # week_key arithmetic across ISO year boundaries.
            if early_season_lag and early_season_lag > 0:
                sensor_feat_cols = [c for c in int_features + ext_features if c in weekly_env.columns]
                env_sorted = weekly_env.sort_values('week_key').reset_index(drop=True)
                wk_to_pos  = {wk: pos for pos, wk in enumerate(env_sorted['week_key'])}
                for col in sensor_feat_cols:
                    env_vals = env_sorted[col].values
                    def _lookup(wk, _ev=env_vals, _m=wk_to_pos, _lag=early_season_lag):
                        pos = _m.get(wk)
                        if pos is None:
                            return np.nan
                        early_pos = pos - _lag
                        return float(_ev[early_pos]) if early_pos >= 0 else np.nan
                    weekly[f'{col}_early'] = weekly['week_key'].map(_lookup)

            if pheno_feat_cols:
                pheno_season = df_pheno[df_pheno['temporada'] == temp][['semana'] + pheno_feat_cols].copy()
                pheno_season = pheno_season.rename(columns={'semana': '_sensor_semana'})
                weekly['_sensor_semana'] = (weekly['week_key'] % 100).astype(int)
                weekly = weekly.merge(pheno_season, on='_sensor_semana', how='left')
                weekly = weekly.drop(columns=['_sensor_semana'])
                for col in pheno_feat_cols:
                    weekly[col] = weekly[col].ffill().bfill()

            # Lag features
            lags = lag_features if lag_features is not None else [1]
            for lag in lags:
                weekly[f'kg_lag_{lag}'] = weekly['kg_reales'].shift(lag)

            # Rolling statistics (optional)
            if include_rolling_mean:
                weekly['kg_roll_mean_4'] = weekly['kg_reales'].shift(1).rolling(4, min_periods=1).mean()

            # Target: absolute yield h weeks ahead
            weekly['target'] = weekly['kg_reales'].shift(-horizon)

            if resolution == 'daily':
                # For daily modality: each production week expands into its 7 daily rows.
                # Daily rows inherit the target and temporal features from their parent week.
                # Sensor features come from the raw daily data (not weekly aggregates).
                sensor_daily_cols = [c for c in int_features + ext_features
                                     if c in daily.columns]
                for rc in riego_features:
                    if rc in daily.columns:
                        sensor_daily_cols.append(rc)

                # Columns to copy from weekly → daily (temporal + target, no sensor aggregates)
                weekly_only = [c for c in weekly.columns
                               if c not in sensor_daily_cols
                               and c not in ('fecha', 'temporada', 'iso_year', 'iso_week')]

                # daily already has week_key; merge to get temporal + target per day
                daily_frame = pd.merge(
                    daily[['fecha', 'temporada', 'week_key'] + sensor_daily_cols],
                    weekly[weekly_only],
                    on='week_key', how='inner'
                ).sort_values('fecha').reset_index(drop=True)

                all_season_frames.append(daily_frame)
            else:
                all_season_frames.append(weekly)

    df_all = pd.concat(all_season_frames, ignore_index=True)
    df_sensor_all = pd.concat(all_sensor_season_frames, ignore_index=True) if all_sensor_season_frames else None

    # Split: configurable train/val seasons, T17 always test
    train_seasons = train_seasons if train_seasons is not None else ['T13', 'T14', 'T15']

    val_season = val_season if val_season is not None else 'T16'

    # Drop rows with NaN from lagging or target shift
    df_all = df_all.dropna().reset_index(drop=True)

    # Feature columns (all go through same pipeline)
    exclude = ['week_idx', 'week_key', 'pos_week', 'prod_week_key', 'kg_reales',
               'temporada', 'target', 'fecha', 'semana', 'iso_year', 'iso_week', 'sensor_gap']
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

    return train_df, val_df, test_df, feature_cols, df_sensor_all


# ============================================================================
# 2. DATASET & DATALOADER
# ============================================================================

def _apply_gap_norm_cnn(sensor_df, prod_df, seq_len, horizon=HORIZON):
    """Trim sensor_df front per season so eff_horizon = gap - seq_len = horizon.

    extra_skip = max(0, gap - seq_len - horizon)
    prod_df is used only to read sensor_gap; it is NOT modified.
    sensor_df rows without a matching season in prod_df are kept unchanged.
    """
    parts = []
    for temp in sensor_df['temporada'].unique():
        s = sensor_df[sensor_df['temporada'] == temp].copy()
        p = prod_df[prod_df['temporada'] == temp]
        if len(p) == 0 or 'sensor_gap' not in p.columns:
            parts.append(s)
            continue
        gap = int(p['sensor_gap'].iloc[0])
        extra_skip = max(0, gap - seq_len - horizon)
        if extra_skip >= len(s):
            raise ValueError(
                f'Season {temp}: extra_skip={extra_skip} >= sensor rows={len(s)}.')
        parts.append(s.iloc[extra_skip:].reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def make_sequences_per_season(Xs, Xt, y, temporadas_s, temporadas_y, seq_len, stride=1):
    """
    Build sliding windows strictly within each season.
    seq_len: fixed integer (from hp['seq_len']). Effective horizon = gap - seq_len per season.
    For each production target j, uses sensor window [j : j+seq_len].

    stride=1 : dense sliding window (weekly resolution)
    stride=7 : one window per production week (daily resolution)

    Returns seqs_Xs, seqs_Xt, seqs_y, seqs_pos.
    """
    seqs_Xs, seqs_Xt, seqs_y, seqs_pos = [], [], [], []
    for temp in pd.Series(temporadas_y).unique():
        mask_s = np.array(temporadas_s) == temp
        mask_y = np.array(temporadas_y) == temp
        Xs_s = Xs[mask_s]
        Xt_s = Xt[mask_s]
        y_s  = y[mask_y]
        for j in range(0, len(y_s), stride):
            if j + seq_len > len(Xs_s):
                break
            seqs_Xs.append(Xs_s[j : j + seq_len])
            seqs_Xt.append(Xt_s[j : j + seq_len])
            seqs_y.append(y_s[j])
            seqs_pos.append(j)
    if len(seqs_Xs) == 0:
        raise ValueError('No sequences built — check seq_len vs season length.')
    return (np.array(seqs_Xs, dtype=np.float32),
            np.array(seqs_Xt, dtype=np.float32),
            np.array(seqs_y,  dtype=np.float32),
            np.array(seqs_pos, dtype=np.int32))


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
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        # Trim conv output to match input length (handles integer padding > 0).
        # With padding='same' output already equals input length — no trim needed.
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.residual_conv is None else self.residual_conv(x)
        return out + res


# Features that bypass the CNN and go straight to the LSTM (clean temporal signal)
TEMPORAL_FEATURE_PREFIXES = ('kg_lag_', 'kg_roll_')
TEMPORAL_FEATURE_NAMES    = {'dias_desde_transplante', 'week_in_season'}


def split_features(feature_cols):
    """Split feature list into sensor (→ CNN+PCA), pheno (→ concatenated with PCA), temporal (→ LSTM)."""
    temporal, pheno, sensor = [], [], []
    for c in feature_cols:
        if c in TEMPORAL_FEATURE_NAMES or any(c.startswith(p) for p in TEMPORAL_FEATURE_PREFIXES):
            temporal.append(c)
        elif c in PHENO_FEATURE_NAMES:
            pheno.append(c)
        else:
            sensor.append(c)
    return sensor, pheno, temporal


class CNNRNN(nn.Module):
    """
    CNN-RNN model for crop yield prediction.
    Sensor features go through the CNN; temporal features (calendar encodings,
    lag features) bypass the CNN and are concatenated at the LSTM input.

      x_sensor  → CNN → (batch, seq_len, cnn_filters)  ─┐
                                                          cat → LSTM → FC → (batch, 1)
      x_temporal (clean) ──────────────────────────────  ─┘
    """
    def __init__(self, n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding,
                 num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden):
        super().__init__()

        cnn_blocks = []
        in_ch = n_sensor
        for _ in range(num_cnn_blocks):
            cnn_blocks.append(CNNBlock(in_ch, cnn_filters, cnn_kernel_size,
                                       cnn_padding, dropout))
            in_ch = cnn_filters
        self.cnn = nn.Sequential(*cnn_blocks)

        self.lstm = nn.LSTM(
            input_size=cnn_filters + n_temporal,
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

    def forward(self, x_sensor, x_temporal):
        # x_sensor:  (batch, seq_len, n_sensor)
        # x_temporal:(batch, seq_len, n_temporal)
        x = x_sensor.permute(0, 2, 1)          # (batch, n_sensor, seq_len)
        x = self.cnn(x)                          # (batch, cnn_filters, seq_len)
        x = x.permute(0, 2, 1)                  # (batch, seq_len, cnn_filters)
        x = torch.cat([x, x_temporal], dim=-1)  # (batch, seq_len, cnn_filters + n_temporal)

        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


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
# Hyperparameters are defined in each greenhouse-specific script (cnn_rnn_inv3.py / cnn_rnn_inv4.py)


# ============================================================================
# 5. TRAINING
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(invernadero_id, hp, train_seasons=None, val_season=None, transform='boxcox', skip_first_weeks=0, return_arrays=False):
    """Load data, normalize, split into train/val/test, create DataLoaders."""
    train_df, val_df, test_df, feature_cols, df_sensor_all = build_dataset_for_greenhouse(
        invernadero_id, horizon=0,
        lag_features=hp.get('lag_features', [1]),
        include_rolling_mean=hp.get('include_rolling_mean', False),
        train_seasons=train_seasons, val_season=val_season,
        early_season_lag=hp.get('early_season_lag', 0),
        resolution=hp.get('resolution', 'weekly'),
        alignment=hp.get('alignment', 'positional'),
        sensor_lead_weeks=hp.get('sensor_lead_weeks', None),
    )

    if skip_first_weeks > 0:
        train_df = train_df[train_df['week_in_season'] >= skip_first_weeks].reset_index(drop=True)
        val_df   = val_df[val_df['week_in_season']     >= skip_first_weeks].reset_index(drop=True)
        test_df  = test_df[test_df['week_in_season']   >= skip_first_weeks].reset_index(drop=True)
        print(f'  Skipped first {skip_first_weeks} weeks → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')

    sensor_cols, pheno_cols, temporal_cols = split_features(feature_cols)
    if hp.get('temporal_keep'):
        temporal_cols = [c for c in temporal_cols if c in hp['temporal_keep']]
        print(f'  Temporal features kept ({len(temporal_cols)}): {temporal_cols}')
    else:
        print(f'  Temporal features ({len(temporal_cols)}): {temporal_cols}')
    print(f'  Sensor features ({len(sensor_cols)}): {sensor_cols}')
    print(f'  Phenology features ({len(pheno_cols)}): {pheno_cols}')

    _s = hp.get('scaler', 'robust' if hp.get('use_robust_scaler') else 'minmax')
    _scaler = {'minmax': lambda: MinMaxScaler(feature_range=(-1, 1)),
               'robust': RobustScaler,
               'standard': StandardScaler}.get(_s, lambda: MinMaxScaler(feature_range=(-1, 1)))

    # ── Targets: transform + scaler ──
    y_train = train_df['target'].values.astype(np.float32)
    y_val   = val_df['target'].values.astype(np.float32)
    y_test  = test_df['target'].values.astype(np.float32)

    if transform == 'log':
        y_train = np.log1p(y_train).astype(np.float32)
        y_val   = np.log1p(y_val).astype(np.float32)
        y_test  = np.log1p(y_test).astype(np.float32)
        bc_lambda = 'log'
        print('  Transform: log1p')
    else:
        from scipy.stats import boxcox as _boxcox
        y_train_bc, bc_lambda = _boxcox(y_train + 1.0)
        y_train = y_train_bc.astype(np.float32)
        y_val   = _boxcox(y_val  + 1.0, lmbda=bc_lambda).astype(np.float32)
        y_test  = _boxcox(y_test + 1.0, lmbda=bc_lambda).astype(np.float32)
        print(f'  Box-Cox λ = {bc_lambda:.4f}')

    scaler_y = _scaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # ── Sensor/temporal source: use full sensor frames when available ──
    if df_sensor_all is not None:
        sensor_tr_df = df_sensor_all[df_sensor_all['temporada'].isin(
            train_seasons if train_seasons is not None else ['T13', 'T14', 'T15'])].reset_index(drop=True)
        sensor_va_df = df_sensor_all[df_sensor_all['temporada'] == (val_season or 'T16')].reset_index(drop=True)
        sensor_te_df = df_sensor_all[df_sensor_all['temporada'] == 'T17'].reset_index(drop=True)
        # Gap normalisation: trim sensor front so eff_horizon = gap - seq_len = HORIZON
        _seq = hp.get('seq_len', 6)
        sensor_tr_df = _apply_gap_norm_cnn(sensor_tr_df, train_df, _seq)
        sensor_va_df = _apply_gap_norm_cnn(sensor_va_df, val_df,   _seq)
        sensor_te_df = _apply_gap_norm_cnn(sensor_te_df, test_df,  _seq)
        print(f'  Gap-norm applied (HORIZON={HORIZON}, seq_len={_seq}): sensor frames trimmed per season')
        # Only keep columns present in both sensor frames and feature_cols
        avail_sensor  = [c for c in sensor_cols  if c in sensor_tr_df.columns]
        avail_temporal = [c for c in temporal_cols if c in sensor_tr_df.columns]
    else:
        sensor_tr_df, sensor_va_df, sensor_te_df = train_df, val_df, test_df
        avail_sensor, avail_temporal = sensor_cols, temporal_cols

    # ── Sensor features: scale + PCA ──
    Xs_tr = sensor_tr_df[avail_sensor].values.astype(np.float32)
    Xs_va = sensor_va_df[avail_sensor].values.astype(np.float32)
    Xs_te = sensor_te_df[avail_sensor].values.astype(np.float32)

    scaler_Xs = _scaler()
    Xs_tr = scaler_Xs.fit_transform(Xs_tr)
    Xs_va = scaler_Xs.transform(Xs_va)
    Xs_te = scaler_Xs.transform(Xs_te)

    pca = PCA(n_components=hp.get('pca_variance', 0.95))
    Xs_tr = pca.fit_transform(Xs_tr)
    Xs_va = pca.transform(Xs_va)
    Xs_te = pca.transform(Xs_te)
    n_sensor_pca = pca.n_components_
    print(f'  PCA (sensor only): {len(avail_sensor)} -> {n_sensor_pca} components')

    # ── Phenology features: MinMax scale, concatenate with sensor PCA ──
    if pheno_cols:
        avail_pheno_s = [c for c in pheno_cols if c in sensor_tr_df.columns]
        avail_pheno_p = [c for c in pheno_cols if c in train_df.columns]
        _pheno_src_tr = sensor_tr_df if avail_pheno_s else train_df
        _pheno_src_va = sensor_va_df if avail_pheno_s else val_df
        _pheno_src_te = sensor_te_df if avail_pheno_s else test_df
        _pheno_cols   = avail_pheno_s if avail_pheno_s else avail_pheno_p
        if _pheno_cols:
            scaler_Xp = MinMaxScaler(feature_range=(-1, 1))
            Xp_tr = scaler_Xp.fit_transform(_pheno_src_tr[_pheno_cols].values.astype(np.float32))
            Xp_va = scaler_Xp.transform(_pheno_src_va[_pheno_cols].values.astype(np.float32))
            Xp_te = scaler_Xp.transform(_pheno_src_te[_pheno_cols].values.astype(np.float32))
            Xs_tr = np.hstack([Xs_tr, Xp_tr])
            Xs_va = np.hstack([Xs_va, Xp_va])
            Xs_te = np.hstack([Xs_te, Xp_te])
            print(f'  CNN input: {n_sensor_pca} PCA + {len(_pheno_cols)} pheno = {Xs_tr.shape[1]} channels')
    n_sensor_pca = Xs_tr.shape[1]

    # ── Temporal features ──
    Xt_tr = sensor_tr_df[avail_temporal].values.astype(np.float32)
    Xt_va = sensor_va_df[avail_temporal].values.astype(np.float32)
    Xt_te = sensor_te_df[avail_temporal].values.astype(np.float32)

    scaler_Xt = _scaler()
    Xt_tr = scaler_Xt.fit_transform(Xt_tr)
    Xt_va = scaler_Xt.transform(Xt_va)
    Xt_te = scaler_Xt.transform(Xt_te)
    n_temporal = len(avail_temporal)

    # ── Sequences ──
    seq_len = hp['seq_len']
    stride  = hp.get('stride', 7 if hp.get('resolution') == 'daily' else 1)
    print(f'  HORIZON={HORIZON} (doc) | seq_len={seq_len} | eff_horizon = gap - seq_len (per temporada, shown above)')
    temps_sensor_tr = sensor_tr_df['temporada'].values
    temps_sensor_va = sensor_va_df['temporada'].values
    temps_sensor_te = sensor_te_df['temporada'].values
    temps_prod_tr   = train_df['temporada'].values
    temps_prod_va   = val_df['temporada'].values
    temps_prod_te   = test_df['temporada'].values

    Xs_tr_seq, Xt_tr_seq, y_tr, pos_tr = make_sequences_per_season(
        Xs_tr, Xt_tr, y_train, temps_sensor_tr, temps_prod_tr, seq_len, stride)
    Xs_va_seq, Xt_va_seq, y_va, pos_va = make_sequences_per_season(
        Xs_va, Xt_va, y_val,   temps_sensor_va, temps_prod_va, seq_len, stride)
    Xs_te_seq, Xt_te_seq, y_te, pos_te = make_sequences_per_season(
        Xs_te, Xt_te, y_test,  temps_sensor_te, temps_prod_te, seq_len, stride)
    print(f'  Sequences — train: {len(Xs_tr_seq)}, val: {len(Xs_va_seq)}, test: {len(Xs_te_seq)}'
          + (f' (stride={stride})' if stride > 1 else ''))

    if return_arrays:
        _has_wk = 'prod_week_key' in test_df.columns
        _week_keys = test_df['prod_week_key'].values if _has_wk else np.arange(len(y_te))
        return (Xs_tr_seq, Xt_tr_seq, y_tr,
                Xs_va_seq, Xt_va_seq, y_va,
                Xs_te_seq, Xt_te_seq, y_te,
                scaler_y, bc_lambda, _week_keys)

    def to_ds(Xs, Xt, y, w):
        return TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(Xt),
                             torch.FloatTensor(y).unsqueeze(1),
                             torch.FloatTensor(w).reshape(-1, 1))

    ramp_weeks  = hp.get('ramp_weeks', 4)
    ramp_weight = hp.get('ramp_weight', 1.0)
    w_train = np.where(pos_tr < ramp_weeks, ramp_weight, 1.0).astype(np.float32)
    w_ones_va = np.ones(len(y_va), dtype=np.float32)
    w_ones_te = np.ones(len(y_te), dtype=np.float32)

    bs = hp['batch_size']
    train_loader = DataLoader(to_ds(Xs_tr_seq, Xt_tr_seq, y_tr, w_train), batch_size=bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(to_ds(Xs_va_seq, Xt_va_seq, y_va, w_ones_va), batch_size=bs, shuffle=False)
    test_loader  = DataLoader(to_ds(Xs_te_seq, Xt_te_seq, y_te, w_ones_te), batch_size=bs, shuffle=False)

    var_y_train = float(np.var(y_tr)) if len(y_tr) > 1 else 1.0
    print(f'  Train y variance (scaled {transform}): {var_y_train:.6f}')

    _has_wk = 'prod_week_key' in test_df.columns
    test_week_keys = test_df['prod_week_key'].values if _has_wk else np.arange(len(y_te))
    wis_test = pos_te  # week_in_season of each test sequence target (== pos within season)
    return train_loader, val_loader, test_loader, scaler_y, bc_lambda, n_sensor_pca, n_temporal, var_y_train, test_week_keys, wis_test


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
    """Weighted Huber — weight ∝ |target - mean(target)|^p, emphasizes both peaks and valleys.
    under_penalty > 0 penalizes underestimates extra: effective weight = 1 + under_penalty when pred < target."""
    def __init__(self, corr_weight=0.8, power=1, delta=0.7, under_penalty=0.0):
        super().__init__()
        self.corr_weight = corr_weight
        self.power = power
        self.delta = delta
        self.under_penalty = under_penalty

    def forward(self, pred, target, sample_weights=None):
        pred_f   = pred.flatten()
        target_f = target.flatten()
        weights = (torch.abs(target_f - target_f.mean()) + 1e-6) ** self.power
        weights = weights / weights.mean()
        err = torch.abs(pred_f - target_f)
        huber = torch.where(err <= self.delta,
                            0.5 * err ** 2 / self.delta,
                            err - 0.5 * self.delta)
        if self.under_penalty > 0.0:
            under_mask = (pred_f < target_f).float()
            weights = weights * (1.0 + self.under_penalty * under_mask)
        if sample_weights is not None:
            weights = weights * sample_weights.flatten().to(pred.device)
        wmae = torch.mean(weights * huber)
        corr_loss = torch.zeros(1, device=pred.device)
        if pred_f.shape[0] >= 4 and self.corr_weight > 0:
            pm = pred_f - pred_f.mean()
            tm = target_f - target_f.mean()
            corr = torch.sum(pm * tm) / (
                torch.sqrt(torch.sum(pm ** 2) + 1e-8) *
                torch.sqrt(torch.sum(tm ** 2) + 1e-8)
            )
            corr_loss = self.corr_weight * (1.0 - corr)
        return wmae + corr_loss


def train_model(model, train_loader, val_loader, hp, model_path, var_y_train=1.0):
    """Train the CNN-RNN model with early stopping on validation score."""
    loss_type = hp.get('loss_type', 'wmae')
    if loss_type == 'huber':
        criterion = nn.HuberLoss(delta=hp.get('huber_delta', 1.0))
    elif loss_type == 'nse_corr':
        criterion = NSECorrLoss(corr_weight=hp.get('corr_weight', 0.8),
                                var_y_train=var_y_train)
    else:
        criterion = YieldWMAELoss(corr_weight=hp.get('corr_weight', 0.8),
                                  power=hp.get('wmae_power', 1),
                                  under_penalty=hp.get('under_penalty', 0.0))
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

        for Xs_batch, Xt_batch, y_batch, w_batch in train_loader:
            Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(Xs_batch, Xt_batch)
            loss = criterion(pred, y_batch, sample_weights=w_batch)
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
            for Xs_batch, Xt_batch, y_batch, _ in val_loader:
                Xs_batch, Xt_batch, y_batch = Xs_batch.to(DEVICE), Xt_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(Xs_batch, Xt_batch)
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


def compute_phase_metrics(y_true, y_pred, wis_test, ramp_weeks=4):
    """Compute metrics split by season phase (ramp-up vs peak).

    Args:
        y_true: 1-D array of true kg values (original scale).
        y_pred: 1-D array of predicted kg values (original scale).
        wis_test: 1-D int array of week_in_season per test sample.
        ramp_weeks: number of weeks considered ramp-up (week_in_season < ramp_weeks).

    Returns:
        dict with keys 'all', 'ramp_up' (if >=2 ramp samples), 'peak' (if >=2 peak samples).
        Each value is the output of compute_metrics().
    """
    mask_ramp = wis_test < ramp_weeks
    result = {'all': compute_metrics(y_true, y_pred)}
    if mask_ramp.sum() >= 2:
        result['ramp_up'] = compute_metrics(y_true[mask_ramp], y_pred[mask_ramp])
    if (~mask_ramp).sum() >= 2:
        result['peak'] = compute_metrics(y_true[~mask_ramp], y_pred[~mask_ramp])
    return result


def evaluate_model(model, test_loader, scaler_y, bc_lambda, hist_te=None):
    """Run inference on test set, inverse-transform predictions, compute metrics."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for Xs_batch, Xt_batch, y_batch, _ in test_loader:
            pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred_norm = np.concatenate(all_preds).flatten()
    y_true_norm = np.concatenate(all_targets).flatten()

    # Inverse MinMaxScaler
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()

    if bc_lambda == 'log':
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)
    elif bc_lambda is not None:
        from scipy.special import inv_boxcox as _inv_boxcox
        if bc_lambda > 0:
            y_pred = np.clip(y_pred, -1.0 / bc_lambda + 1e-6, None)
        y_pred = _inv_boxcox(y_pred, bc_lambda) - 1.0
        y_true = _inv_boxcox(y_true, bc_lambda) - 1.0
    else:
        # Residual mode: add back historical average → actual kg
        if hist_te is not None:
            y_pred = y_pred + hist_te
            y_true = y_true + hist_te

    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def evaluate_loader(model, loader, scaler_y, bc_lambda):
    """Run inference on any DataLoader; return (y_true, y_pred) in original kg scale."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xs_batch, Xt_batch, y_batch, _ in loader:
            pred = model(Xs_batch.to(DEVICE), Xt_batch.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    y_pred_norm = np.concatenate(all_preds).flatten()
    y_true_norm = np.concatenate(all_targets).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    if bc_lambda == 'log':
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)
    elif bc_lambda is not None:
        from scipy.special import inv_boxcox as _inv_boxcox
        if bc_lambda > 0:
            y_pred = np.clip(y_pred, -1.0 / bc_lambda + 1e-6, None)
        y_pred = _inv_boxcox(y_pred, bc_lambda) - 1.0
        y_true = _inv_boxcox(y_true, bc_lambda) - 1.0
    return y_true, y_pred


def compute_cptc_intervals(y_cal_true, y_cal_pred, y_te_pred,
                           alpha=0.10, gamma=0.005, K=3):
    """
    Warm-starts K state-specific calibration sets from validation residuals,
    then applies CPTC online update to test predictions.
    Returns (lower, upper, val_coverage, avg_width).
    """
    cal_residuals = np.abs(y_cal_true - y_cal_pred)
    n_cal = len(cal_residuals)
    n_te  = len(y_te_pred)

    def _states(n):
        return np.minimum(np.arange(n) * K // max(n, 1), K - 1)

    cal_states = _states(n_cal)
    te_states  = _states(n_te)

    S = {k: list(cal_residuals[cal_states == k]) for k in range(K)}

    alpha_t = alpha
    lower, upper = [], []

    for t in range(n_te):
        k   = int(te_states[t])
        sk  = S[k]
        q   = np.quantile(sk, float(np.clip(1 - alpha_t, 0.01, 0.99))) if sk else 0.0
        lower.append(y_te_pred[t] - q)
        upper.append(y_te_pred[t] + q)
        alpha_t = alpha_t + gamma * (alpha - (1 if len(lower) > 1 else 0))

    lower = np.array(lower)
    upper = np.array(upper)

    val_coverage = float(np.mean(
        (y_cal_true >= (y_cal_pred - np.array([np.quantile(S[k], 1 - alpha)
                                               if S[k] else 0.0
                                               for k in cal_states]))) &
        (y_cal_true <= (y_cal_pred + np.array([np.quantile(S[k], 1 - alpha)
                                               if S[k] else 0.0
                                               for k in cal_states])))
    ))
    avg_width = float(np.mean(upper - lower))

    for i in range(n_cal):
        k  = int(cal_states[i])
        sk = cal_residuals[cal_states == k]
        S[k].append(float(cal_residuals[i]))

    return lower, upper, val_coverage, avg_width


def plot_results_per_greenhouse(results, horizon=6, intervals=None):
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
                         label=f'Mejor modelo (h={horizon})', markersize=5)
        if 'ensemble_pred' in res:
            axes[i, 1].plot(weeks, res['ensemble_pred'], '^:', color='green',
                             label='Ensemble top-20', markersize=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} - Predicción a {horizon} semanas (T17)')
        axes[i, 1].set_xlabel('Semana de test (T17)')
        axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()

        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
        if 'ensemble_metrics' in res:
            em = res['ensemble_metrics']
            metrics_text += f'\n─── Ensemble top-20 ───\nR²: {em["R²"]:.4f}  MAPE: {em["MAPE (%)"]:.2f}%'
        axes[i, 1].text(0.02, 0.98, metrics_text, transform=axes[i, 1].transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    inv_ids = list(results.keys())
    if len(inv_ids) == 1:
        fname = f'cnn_rnn_inv{inv_ids[0]}_results.png'
    else:
        fname = 'cnn_rnn_results.png'
    plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results plot saved to {RESULTS_DIR / fname}')


def print_metrics(invernadero_id, metrics, horizon=6):
    """Pretty-print evaluation metrics."""
    print(f'\n{"=" * 55}')
    print(f'  INVERNADERO {invernadero_id} - MÉTRICAS (Test: T17, h={horizon} semanas)')
    print(f'{"=" * 55}')
    for name, value in metrics.items():
        print(f'  {name:<15s}: {value:>10.4f}')
    print(f'{"=" * 55}')
    print(f'  >>> MAPE: {metrics["MAPE (%)"]:>10.4f}% <<<')
    print(f'{"=" * 55}')


# Training and evaluation logic is in cnn_rnn_inv3.py and cnn_rnn_inv4.py
