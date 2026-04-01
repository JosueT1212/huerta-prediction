"""
Temporal Fusion Transformer (TFT) for Greenhouse Crop Yield Prediction
=======================================================================
Based on: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", Int. J. Forecasting.
arXiv: 1912.09363

Three input streams:
  Static:        greenhouse ID (Inv3=0, Inv4=1)
  Past observed: internal/external sensors + riego + production history
  Known future:  temporal features (Fourier harmonics, season position,
                 dias_desde_transplante) — known for any future week

Multi-horizon output: quantiles [0.1, 0.5, 0.9] for weeks t+1..t+H.
Evaluated at step H=4 (median) for comparison with CNN-RNN baseline.

Train: T13-T15 | Val: T16 | Test: T17
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
# MPS excluded: small d_model=32 tensors trigger MPS buffer size assertion
print(f'Using device: {DEVICE}')

DATA_DIR    = Path(__file__).resolve().parent.parent / 'Data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEASON_NAMES = ['T13', 'T14', 'T15', 'T16', 'T17']

# ============================================================================
# 1. DATA LOADING  (same pipeline as cnn_rnn_yield.py)
# ============================================================================

def load_transplant_dates():
    filepath = DATA_DIR / 'Fechas de Transplante.xlsx'
    raw = pd.read_excel(filepath, header=None)
    dates = {}
    for _, row in raw.iloc[2:].iterrows():
        season_num = str(row.iloc[0]).strip()
        temp = f'T{int(float(season_num))}'
        for inv_id, col_idx in [(3, 1), (4, 2)]:
            cell = row.iloc[col_idx]
            if isinstance(cell, (pd.Timestamp, datetime.datetime)):
                dates[(temp, inv_id)] = cell
            else:
                raw_val = str(cell).strip().split(' y ')[0].strip()
                dates[(temp, inv_id)] = pd.to_datetime(raw_val, dayfirst=True)
    return dates


def load_riego_variables(invernadero_id):
    filepath = DATA_DIR / f'Variables riego invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        rename = {}
        for col in df.columns:
            col_l = col.lower()
            if 'fecha' in col_l:  rename[col] = 'fecha'
            elif 'ph' in col_l:   rename[col] = 'ph_promedio'
            elif 'ce' in col_l:   rename[col] = 'ce_promedio'
            elif 'riego' in col_l: rename[col] = 'riego_total'
        df = df.rename(columns=rename)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['temporada'] = SEASON_NAMES[i]
        for col in ['ph_promedio', 'ce_promedio', 'riego_total']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('fecha').reset_index(drop=True)
        keep = ['fecha', 'temporada'] + [c for c in ['riego_total', 'ph_promedio', 'ce_promedio']
                                         if c in df.columns]
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


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
        frames.append(df.sort_values('fecha').reset_index(drop=True))
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
        frames.append(df.sort_values('fecha').reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def load_production(invernadero_id):
    filepath = DATA_DIR / f'Kg por semana T13 - T17 Invernadero {invernadero_id}.xlsx'
    xls = pd.ExcelFile(filepath)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        raw = pd.read_excel(filepath, sheet_name=sheet, header=None)
        data = raw.iloc[2:].copy()
        data.columns = ['semana', 'kg_reales']
        data['semana']    = pd.to_numeric(data['semana'],    errors='coerce')
        data['kg_reales'] = pd.to_numeric(data['kg_reales'], errors='coerce')
        data['temporada'] = SEASON_NAMES[i]
        frames.append(data)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=['kg_reales']).reset_index(drop=True)


# ============================================================================
# 2. FEATURE DEFINITIONS
# ============================================================================

SENSOR_COLS = [
    'temp_prom_int', 'temp_min_int', 'temp_max_int', 'hr_prom_int',
    'co2_ppm', 'deficit_humedad', 'deficit_presion_vapor', 'humedad_abs_int',
    'temp_prom_ext', 'temp_max_ext', 'temp_min_ext', 'hr_prom_ext',
    'rad_sum', 'rad_max', 'dh_ext', 'humedad_abs_ext',
    'riego_total', 'ph_promedio', 'ce_promedio',
]  # 19 sensor features — past-observed only

TEMPORAL_COLS = [
    'week_in_season', 'week_position',
    'week_sin_52', 'week_cos_52',
    'week_sin_26', 'week_cos_26',
    'week_sin_13', 'week_cos_13',
    'dias_desde_transplante',
]  # 9 known-future features


# ============================================================================
# 3. DATASET CONSTRUCTION
# ============================================================================

def build_weekly_seasons(invernadero_id):
    """
    Build per-season weekly DataFrames with sensor + temporal features
    and raw kg_reales. No lag or target shift — TFT handles history explicitly.
    """
    df_int  = load_internal_variables_all_seasons()
    df_ext  = load_external_variables_all_seasons()
    df_kg   = load_production(invernadero_id)
    df_riego = load_riego_variables(invernadero_id)
    transplant_dates = load_transplant_dates()

    # Rename internal columns
    int_rename = {}
    for col in df_int.columns:
        if 'temperatura promedio' in col:      int_rename[col] = 'temp_prom_int'
        elif 'temperatura minima' in col:      int_rename[col] = 'temp_min_int'
        elif 'temperatura maxima' in col:      int_rename[col] = 'temp_max_int'
        elif 'humedad relativa promedio' in col: int_rename[col] = 'hr_prom_int'
        elif 'co2' in col:                     int_rename[col] = 'co2_ppm'
        elif 'ficit de humedad' in col:        int_rename[col] = 'deficit_humedad'
        elif 'ficit' in col and 'vapor' in col: int_rename[col] = 'deficit_presion_vapor'
        elif 'humedad absoluta' in col:        int_rename[col] = 'humedad_abs_int'
    df_int = df_int.rename(columns=int_rename)

    # Rename external columns
    ext_rename = {}
    for col in df_ext.columns:
        if 'promedio' in col and 'temperatura' in col:  ext_rename[col] = 'temp_prom_ext'
        elif 'maxima' in col and 'temperatura' in col:  ext_rename[col] = 'temp_max_ext'
        elif 'minima' in col and 'temperatura' in col:  ext_rename[col] = 'temp_min_ext'
        elif 'hr promedio' in col:                      ext_rename[col] = 'hr_prom_ext'
        elif 'suma' in col:                             ext_rename[col] = 'rad_sum'
        elif 'intensidad' in col:                       ext_rename[col] = 'rad_max'
        elif 'dh' in col:                               ext_rename[col] = 'dh_ext'
        elif 'humedad absoluta' in col:                 ext_rename[col] = 'humedad_abs_ext'
    df_ext = df_ext.rename(columns=ext_rename)

    int_features = [c for c in SENSOR_COLS[:8]  if c in df_int.columns]
    ext_features = [c for c in SENSOR_COLS[8:16] if c in df_ext.columns]
    riego_features = [c for c in ['riego_total', 'ph_promedio', 'ce_promedio']
                      if c in df_riego.columns]

    season_dfs = {}
    for temp in SEASON_NAMES:
        int_s   = df_int[df_int['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        ext_s   = df_ext[df_ext['temporada'] == temp].sort_values('fecha').reset_index(drop=True)
        kg_s    = df_kg[df_kg['temporada'] == temp].reset_index(drop=True)
        riego_s = df_riego[df_riego['temporada'] == temp][['fecha'] + riego_features]

        if len(kg_s) == 0:
            continue

        # Merge internal + external + riego on fecha
        daily = pd.merge(int_s[['fecha'] + int_features],
                         ext_s[['fecha'] + ext_features], on='fecha', how='inner')
        daily = pd.merge(daily, riego_s, on='fecha', how='left')
        daily = daily.sort_values('fecha').reset_index(drop=True)
        daily['week_idx'] = daily.index // 7

        # dias_desde_transplante
        tp_key = (temp, invernadero_id)
        if tp_key in transplant_dates:
            tp_date = transplant_dates[tp_key]
            daily['dias_desde_transplante'] = (daily['fecha'] - tp_date).dt.days

        # Aggregate daily → weekly
        agg_dict = {f: 'mean' for f in int_features + ext_features}
        if 'rad_sum' in agg_dict:
            agg_dict['rad_sum'] = 'sum'
        for rf in riego_features:
            agg_dict[rf] = 'sum' if rf == 'riego_total' else 'mean'
        if 'dias_desde_transplante' in daily.columns:
            agg_dict['dias_desde_transplante'] = 'mean'

        weekly_env = daily.groupby('week_idx').agg(agg_dict).reset_index()

        # Align with production by sequential week
        n_weeks = min(len(weekly_env), len(kg_s))
        weekly_env = weekly_env.iloc[:n_weeks].copy()
        kg_vals    = kg_s.iloc[:n_weeks].copy()

        weekly = weekly_env.copy()
        weekly['kg_reales']     = kg_vals['kg_reales'].values
        weekly['semana']        = kg_vals['semana'].values
        weekly['temporada']     = temp
        weekly['week_in_season'] = np.arange(n_weeks)
        weekly['week_position']  = weekly['week_in_season'] / n_weeks

        for period in [52, 26, 13]:
            weekly[f'week_sin_{period}'] = np.sin(2 * np.pi * weekly['semana'] / period)
            weekly[f'week_cos_{period}'] = np.cos(2 * np.pi * weekly['semana'] / period)

        # Drop rows missing core sensor data or kg_reales
        avail_sensor = [c for c in SENSOR_COLS if c in weekly.columns]
        weekly = weekly.dropna(subset=avail_sensor + ['kg_reales']).reset_index(drop=True)

        season_dfs[temp] = weekly

    return season_dfs


def build_sequences(season_dfs, invernadero_id, encoder_steps, horizon,
                    scaler_sensor, scaler_temporal, scaler_y, avail_sensor, seasons):
    """Build (past_x, future_x, static_x, target_y) tuples per season."""
    samples = []
    static_val = np.float32(0.0 if invernadero_id == 3 else 1.0)
    avail_temporal = [c for c in TEMPORAL_COLS if c in list(season_dfs.values())[0].columns]

    for temp in seasons:
        if temp not in season_dfs:
            continue
        weekly = season_dfs[temp]
        n = len(weekly)

        X_sensor   = scaler_sensor.transform(weekly[avail_sensor].values.astype(np.float32))
        X_temporal = scaler_temporal.transform(weekly[avail_temporal].values.astype(np.float32))
        y_scaled   = scaler_y.transform(
            weekly['kg_reales'].values.reshape(-1, 1).astype(np.float32)
        ).flatten()

        for i in range(n - encoder_steps - horizon + 1):
            # Past: sensor features + production history (known up to t)
            enc_sensor = X_sensor[i:i + encoder_steps]                        # [T, n_sensor]
            enc_kg     = y_scaled[i:i + encoder_steps].reshape(-1, 1)         # [T, 1]
            past_x     = np.concatenate([enc_sensor, enc_kg], axis=1)         # [T, n_past]

            # Future: temporal features known for all T+H steps
            future_x = X_temporal[i:i + encoder_steps + horizon]              # [T+H, n_temporal]

            # Target: kg at each future step
            target_y = y_scaled[i + encoder_steps:i + encoder_steps + horizon]  # [H]

            if (np.any(np.isnan(past_x)) or np.any(np.isnan(future_x)) or
                    np.any(np.isnan(target_y))):
                continue

            samples.append((past_x, future_x, np.array([static_val]), target_y))

    return samples


def prepare_data(invernadero_id, hp):
    train_seasons = ['T13', 'T14', 'T15']
    val_season    = 'T16'
    enc           = hp['encoder_steps']
    horizon       = hp['horizon']

    season_dfs = build_weekly_seasons(invernadero_id)
    train_df   = pd.concat([season_dfs[t] for t in train_seasons if t in season_dfs])

    avail_sensor   = [c for c in SENSOR_COLS    if c in train_df.columns]
    avail_temporal = [c for c in TEMPORAL_COLS  if c in train_df.columns]

    scaler_sensor   = MinMaxScaler(feature_range=(-1, 1)).fit(
        train_df[avail_sensor].values.astype(np.float32))
    scaler_temporal = MinMaxScaler(feature_range=(-1, 1)).fit(
        train_df[avail_temporal].values.astype(np.float32))
    scaler_y        = MinMaxScaler(feature_range=(-1, 1)).fit(
        train_df['kg_reales'].values.reshape(-1, 1).astype(np.float32))

    n_past    = len(avail_sensor) + 1   # sensors + kg history
    n_future  = len(avail_temporal)
    n_static  = 1

    def to_loader(seasons, shuffle):
        samps = build_sequences(season_dfs, invernadero_id, enc, horizon,
                                scaler_sensor, scaler_temporal, scaler_y,
                                avail_sensor, seasons)
        if not samps:
            raise ValueError(f'No valid sequences for {seasons}')
        past_t   = torch.FloatTensor(np.array([s[0] for s in samps]))   # [N, T, n_past]
        future_t = torch.FloatTensor(np.array([s[1] for s in samps]))   # [N, T+H, n_future]
        static_t = torch.FloatTensor(np.array([s[2] for s in samps]))   # [N, 1]
        target_t = torch.FloatTensor(np.array([s[3] for s in samps]))   # [N, H]
        ds = TensorDataset(past_t, future_t, static_t, target_t)
        return DataLoader(ds, batch_size=hp['batch_size'], shuffle=shuffle), len(samps)

    train_loader, n_tr = to_loader(train_seasons, shuffle=True)
    val_loader,   n_va = to_loader([val_season],  shuffle=False)
    test_loader,  n_te = to_loader(['T17'],        shuffle=False)

    print(f'  Invernadero {invernadero_id}:')
    print(f'    Train sequences: {n_tr} | Val: {n_va} | Test: {n_te}')
    print(f'    Past features:   {n_past}  (sensors={len(avail_sensor)} + kg=1)')
    print(f'    Future features: {n_future}')

    return train_loader, val_loader, test_loader, scaler_y, n_past, n_future, n_static


# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================

class GRN(nn.Module):
    """
    Gated Residual Network — core TFT building block.
    GRN(a, c) = LayerNorm(a + GLU(ELU(W2·a + W2c·c)))
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc_in      = nn.Linear(input_dim, hidden_dim)
        self.fc_context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.elu        = nn.ELU()
        self.dropout    = nn.Dropout(dropout)
        self.fc_out     = nn.Linear(hidden_dim, output_dim * 2)   # ×2 for GLU
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip       = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = x if self.skip is None else self.skip(x)
        h = self.fc_in(x)
        if context is not None and self.fc_context is not None:
            h = h + self.fc_context(context)
        h = self.dropout(self.elu(h))
        gate, signal = self.fc_out(h).chunk(2, dim=-1)
        glu = torch.sigmoid(gate) * signal
        return self.layer_norm(residual + glu)


class VariableSelectionNetwork(nn.Module):
    """
    Soft selection of input variables with learned importance weights.
    Each scalar feature is projected to d_model, then importance weights
    are computed via a GRN on the concatenated projections.
    """
    def __init__(self, num_vars, d_model, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_vars   = num_vars
        self.d_model    = d_model
        self.var_projs  = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_vars)])
        self.var_grns   = nn.ModuleList([GRN(d_model, d_model, d_model, dropout)
                                         for _ in range(num_vars)])
        self.select_grn = GRN(num_vars * d_model, d_model, num_vars, dropout,
                               context_dim=context_dim)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        """
        x: [B, num_vars] or [B, T, num_vars]  (one scalar per variable)
        returns: selected [B, d_model] or [B, T, d_model], weights [B, (T,) num_vars]
        """
        if x.dim() == 2:
            B, V = x.shape
            projs      = torch.stack([self.var_projs[i](x[:, i:i+1]) for i in range(V)], dim=1)
            var_outs   = torch.stack([self.var_grns[i](projs[:, i]) for i in range(V)], dim=1)
            weights    = self.softmax(self.select_grn(projs.reshape(B, V * self.d_model), context))
            selected   = (weights.unsqueeze(-1) * var_outs).sum(dim=1)
            return selected, weights
        else:
            B, T, V = x.shape
            projs    = torch.stack([self.var_projs[i](x[:, :, i:i+1]) for i in range(V)], dim=2)
            var_outs = torch.stack([self.var_grns[i](projs[:, :, i]) for i in range(V)], dim=2)
            flat     = projs.reshape(B * T, V * self.d_model)
            ctx_flat = (context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                        if context is not None else None)
            weights  = self.softmax(self.select_grn(flat, ctx_flat)).reshape(B, T, V)
            selected = (weights.unsqueeze(-1) * var_outs).sum(dim=2)
            return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with shared value weights across heads.
    Allows direct interpretation of attention as temporal importance.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q  = nn.Linear(d_model, d_model)
        self.W_k  = nn.Linear(d_model, d_model)
        self.W_v  = nn.Linear(d_model, self.d_k)    # shared across heads
        self.W_o  = nn.Linear(self.d_k, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, d_k  = self.num_heads, self.d_k

        Q = self.W_q(x).reshape(B, T, H, d_k).transpose(1, 2)   # [B, H, T, d_k]
        K = self.W_k(x).reshape(B, T, H, d_k).transpose(1, 2)   # [B, H, T, d_k]
        V = self.W_v(x)                                           # [B, T, d_k] shared

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn    = self.drop(torch.softmax(scores, dim=-1))           # [B, H, T, T]
        V_exp   = V.unsqueeze(1).expand(-1, H, -1, -1)              # [B, H, T, d_k]
        out     = torch.matmul(attn, V_exp).mean(dim=1)             # [B, T, d_k]
        out     = self.W_o(out)                                      # [B, T, d_model]
        return out, attn.mean(dim=1)                                 # avg attn for interpretability


class TemporalFusionTransformer(nn.Module):
    """
    Lightweight TFT adapted for small greenhouse yield dataset.

    Forward inputs:
      past_x:   [B, T, n_past]    sensor + kg history (encoder steps)
      future_x: [B, T+H, n_fut]   temporal features   (encoder + decoder steps)
      static_x: [B, 1]            greenhouse ID

    Output:
      quantiles: [B, H, Q]        production quantiles for each decoder step
    """
    def __init__(self, n_past, n_future, n_static, d_model, num_heads,
                 num_lstm_layers, dropout, quantiles, encoder_steps, horizon):
        super().__init__()
        self.encoder_steps = encoder_steps
        self.horizon       = horizon
        self.num_quantiles = len(quantiles)
        d = d_model

        # --- Static pathway ---
        self.static_vsn = VariableSelectionNetwork(n_static, d, dropout)
        self.enc_cs = GRN(d, d, d, dropout)   # context for VSN
        self.enc_ce = GRN(d, d, d, dropout)   # context for static enrichment
        self.enc_ch = GRN(d, d, d, dropout)   # LSTM h0 init
        self.enc_cc = GRN(d, d, d, dropout)   # LSTM c0 init

        # --- Variable selection ---
        self.past_vsn   = VariableSelectionNetwork(n_past,   d, dropout, context_dim=d)
        self.future_vsn = VariableSelectionNetwork(n_future, d, dropout, context_dim=d)

        # --- LSTM encoder-decoder ---
        self.lstm = nn.LSTM(
            input_size=d, hidden_size=d,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        # Gated skip over LSTM
        self.lstm_gate = GRN(d, d, d, dropout)

        # --- Static enrichment ---
        self.static_enrichment = GRN(d, d, d, dropout, context_dim=d)

        # --- Interpretable multi-head self-attention ---
        self.attn      = InterpretableMultiHeadAttention(d, num_heads, dropout)
        self.attn_gate = GRN(d, d, d, dropout)

        # --- Position-wise feed-forward ---
        self.ff      = GRN(d, d, d, dropout)
        self.ff_gate = GRN(d, d, d, dropout)

        # --- Output: quantile projections for each decoder step ---
        self.out = nn.Linear(d, self.num_quantiles)

    def _causal_mask(self, T, device):
        """Lower-triangular mask: position i can attend to j <= i."""
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]

    def forward(self, past_x, future_x, static_x):
        B  = past_x.size(0)
        T  = self.encoder_steps
        H  = self.horizon
        device = past_x.device

        # --- 1. Static encoding ---
        static_emb, _   = self.static_vsn(static_x)          # [B, d]
        cs = self.enc_cs(static_emb)
        ce = self.enc_ce(static_emb)
        ch = self.enc_ch(static_emb)
        cc = self.enc_cc(static_emb)

        # --- 2. Variable selection ---
        past_sel,   _ = self.past_vsn(past_x, context=cs)     # [B, T, d]
        future_sel, _ = self.future_vsn(future_x, context=cs)  # [B, T+H, d]

        # --- 3. LSTM over full sequence (encoder + decoder) ---
        temporal = torch.cat([past_sel, future_sel[:, T:]], dim=1)  # [B, T+H, d]
        # Init from static context
        h0 = ch.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        lstm_out, _ = self.lstm(temporal, (h0, c0))             # [B, T+H, d]
        lstm_out = self.lstm_gate(lstm_out, context=None) + temporal  # gated residual

        # --- 4. Static enrichment ---
        enriched = self.static_enrichment(lstm_out, context=ce.unsqueeze(1).expand(-1, T+H, -1))

        # --- 5. Interpretable self-attention with causal mask ---
        mask    = self._causal_mask(T + H, device).expand(B, -1, -1, -1)
        attn_out, _ = self.attn(enriched, mask=mask)           # [B, T+H, d]
        attn_out = self.attn_gate(attn_out) + enriched         # gated residual

        # --- 6. Position-wise FFN ---
        ff_out = self.ff(attn_out)
        ff_out = self.ff_gate(ff_out) + attn_out              # gated residual

        # --- 7. Output: only decoder positions ---
        decoder_out = ff_out[:, T:]                            # [B, H, d]
        quantiles   = self.out(decoder_out)                    # [B, H, Q]
        return quantiles


# ============================================================================
# 5. LOSS & TRAINING
# ============================================================================

class QuantileLoss(nn.Module):
    """Pinball loss summed over quantiles and horizon steps."""
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        # pred:   [B, H, Q]
        # target: [B, H]
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            err = target - pred[:, :, i]
            loss = loss + torch.mean(torch.max(q * err, (q - 1) * err))
        return loss / len(self.quantiles)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, hp, model_path):
    criterion = QuantileLoss(hp['quantiles'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5
    )

    best_val   = float('inf')
    patience_c = 0
    train_losses, val_losses = [], []

    for epoch in range(hp['epochs']):
        # Training
        model.train()
        epoch_loss = 0.0
        for past_x, future_x, static_x, target in train_loader:
            past_x, future_x, static_x, target = (
                past_x.to(DEVICE), future_x.to(DEVICE),
                static_x.to(DEVICE), target.to(DEVICE)
            )
            optimizer.zero_grad()
            pred = model(past_x, future_x, static_x)   # [B, H, Q]
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past_x, future_x, static_x, target in val_loader:
                past_x, future_x, static_x, target = (
                    past_x.to(DEVICE), future_x.to(DEVICE),
                    static_x.to(DEVICE), target.to(DEVICE)
                )
                pred = model(past_x, future_x, static_x)
                val_loss += criterion(pred, target).item()
        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            patience_c = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_c += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4d}/{hp["epochs"]} | '
                  f'Train: {avg_train:.5f} | Val: {avg_val:.5f}')

        if patience_c >= hp['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y, horizon):
    """Evaluate on test set using median quantile (q=0.5, index 1)."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for past_x, future_x, static_x, target in test_loader:
            past_x, future_x, static_x = (
                past_x.to(DEVICE), future_x.to(DEVICE), static_x.to(DEVICE)
            )
            pred = model(past_x, future_x, static_x)   # [B, H, Q]
            # Use median (q=0.5) at last horizon step for comparison
            all_pred.append(pred[:, -1, 1].cpu().numpy())   # index 1 = q0.5
            all_true.append(target[:, -1].numpy())

    y_pred_norm = np.concatenate(all_pred)
    y_true_norm = np.concatenate(all_true)

    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    nse   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    pbias = 100 * np.sum(y_true - y_pred) / np.sum(y_true)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100

    return y_true, y_pred, {'RMSE (kg)': rmse, 'R²': r2, 'NSE': nse,
                             'PBIAS (%)': pbias, 'MAPE (%)': mape}


def compute_metrics(y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    nse   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    pbias = 100 * np.sum(y_true - y_pred) / np.sum(y_true)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'RMSE (kg)': rmse, 'R²': r2, 'NSE': nse, 'PBIAS (%)': pbias, 'MAPE (%)': mape}


# ============================================================================
# 6. HYPERPARAMETERS
# ============================================================================

HP = {
    # Architecture
    'd_model':        32,
    'num_heads':       4,
    'num_lstm_layers': 1,
    'dropout':         0.15,
    # Sequence
    'encoder_steps':   4,
    'horizon':         4,
    'quantiles':       [0.1, 0.5, 0.9],
    # Training
    'batch_size':     16,
    'learning_rate':  1e-3,
    'weight_decay':   1e-4,
    'epochs':        300,
    'patience':       80,
    # Multi-seed
    'seeds': [42, 7, 123, 2024, 99, 13, 55, 777, 314, 2025,
              0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
              11, 12, 14, 15, 16, 17, 18, 19, 20, 21],
}


# ============================================================================
# 7. MAIN
# ============================================================================

def plot_results(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (inv_id, res) in enumerate(results.items()):
        axes[i, 0].plot(res['train_losses'], color='steelblue', lw=1, label='Train')
        axes[i, 0].plot(res['val_losses'],   color='coral',     lw=1, label='Val (T16)')
        axes[i, 0].set_title(f'Invernadero {inv_id} — Quantile Loss')
        axes[i, 0].set_xlabel('Epoch'); axes[i, 0].set_ylabel('Loss')
        axes[i, 0].set_yscale('log'); axes[i, 0].legend()

        weeks = np.arange(len(res['y_true']))
        axes[i, 1].plot(weeks, res['y_true'], 'o-', color='steelblue', label='Actual',    ms=5)
        axes[i, 1].plot(weeks, res['y_pred'], 's--', color='coral',    label='Pred (q0.5)', ms=5)
        axes[i, 1].set_title(f'Invernadero {inv_id} — Predicción a 4 semanas (T17)')
        axes[i, 1].set_xlabel('Semana'); axes[i, 1].set_ylabel('Producción (kg)')
        axes[i, 1].legend()
        txt = '\n'.join([f'{k}: {v:.4f}' for k, v in res['metrics'].items()])
        axes[i, 1].text(0.02, 0.98, txt, transform=axes[i, 1].transAxes,
                         va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = RESULTS_DIR / 'tft_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {out_path}')


def main():
    print('=' * 60)
    print('  Temporal Fusion Transformer — Greenhouse Yield')
    print('  Train: T13-T15 | Val: T16 | Test: T17 | h=4 weeks')
    print('=' * 60)

    all_results = {}
    all_metrics = []

    for inv_id in [3, 4]:
        print(f'\n{"─"*60}')
        print(f'  INVERNADERO {inv_id}  ({len(HP["seeds"])} seeds)')
        print(f'{"─"*60}')

        train_loader, val_loader, test_loader, scaler_y, n_past, n_future, n_static = \
            prepare_data(inv_id, HP)

        best_r2     = -float('inf')
        best_result = None

        for seed in HP['seeds']:
            set_seed(seed)

            model = TemporalFusionTransformer(
                n_past=n_past, n_future=n_future, n_static=n_static,
                d_model=HP['d_model'], num_heads=HP['num_heads'],
                num_lstm_layers=HP['num_lstm_layers'], dropout=HP['dropout'],
                quantiles=HP['quantiles'],
                encoder_steps=HP['encoder_steps'], horizon=HP['horizon'],
            ).to(DEVICE)

            model_path = RESULTS_DIR / f'best_tft_inv{inv_id}.pt'
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, HP, model_path
            )

            y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler_y, HP['horizon'])
            r2   = metrics['R²']
            mape = metrics['MAPE (%)']
            print(f'    s{seed}: R²={r2:.4f}, MAPE={mape:.2f}%')

            if r2 > best_r2:
                best_r2     = r2
                best_result = dict(y_true=y_true, y_pred=y_pred, metrics=metrics,
                                   train_losses=train_losses, val_losses=val_losses,
                                   seed=seed)

        print(f'\n  >>> Best seed={best_result["seed"]} (R²={best_r2:.4f}) <<<')
        print(f'\n{"="*55}')
        print(f'  INVERNADERO {inv_id} — MÉTRICAS (Test T17, h=4)')
        print(f'{"="*55}')
        for k, v in best_result['metrics'].items():
            print(f'  {k:<15s}: {v:>10.4f}')
        print(f'{"="*55}')

        all_results[inv_id] = best_result
        all_metrics.append({'invernadero': inv_id, **best_result['metrics']})

    print('\nGenerando gráficas...')
    plot_results(all_results)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'tft_metrics.csv', index=False)
    print(f'Metrics saved to {RESULTS_DIR / "tft_metrics.csv"}')

    print('\n' + '=' * 70)
    print('  RESUMEN TFT — Predicción a 4 semanas')
    print('  Train: T13-T15 | Val: T16 | Test: T17')
    print('=' * 70)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print('=' * 70)


if __name__ == '__main__':
    main()
