"""
Lectura de datos crudos para el dashboard (fenología + histórico de sensores).

No toca el modelo ni engine.py. Solo lee los Excel de Data/ y los expone como
series listas para graficar. Cachea en memoria por proceso (los Excel no cambian
en runtime del demo).

- Fenología: promedio semanal de las 8 variables con las que se entrena el modelo
  (mismas columnas que cnn_rnn_yield.PHENO_FEATURE_COLS).
- Histórico de sensores: promedio mensual por temporada (una serie por temporada).
"""
from __future__ import annotations

import glob
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'Data'

SEASONS = list(range(13, 18))            # T13..T17
SEASON_LABELS = {s: f'T{s}' for s in SEASONS}

# ── Fenología ───────────────────────────────────────────────────────────────
# Mismas columnas que entrena el modelo (cnn_rnn_yield.PHENO_FEATURE_COLS),
# con metadatos para el formulario "Insertar datos" y los KPIs.
PHENO_FIELDS = [
    {'col': 'RACIMOS PUESTOS',               'key': 'racimos_puestos',    'label': 'Racimos puestos',          'unit': 'conteo', 'min': 0, 'max': 40,  'step': 1},
    {'col': 'FLORES EN RACIMO ABIERTAS',     'key': 'flores_abiertas',    'label': 'Flores en racimo abiertas','unit': 'conteo', 'min': 0, 'max': 12,  'step': 1},
    {'col': 'CANTIDAD DE RACIMOS EN PLANTA', 'key': 'racimos_planta',     'label': 'Racimos en planta',        'unit': 'conteo', 'min': 0, 'max': 15,  'step': 1},
    {'col': 'CANTIDAD DE TOMATES',           'key': 'cantidad_tomates',   'label': 'Cantidad de tomates',      'unit': 'conteo', 'min': 0, 'max': 60,  'step': 1},
    {'col': 'Nº DE RACIMO EN COSECHA',       'key': 'racimo_cosecha',     'label': 'Nº de racimo en cosecha',  'unit': 'índice', 'min': 0, 'max': 40,  'step': 1},
    {'col': 'TOMATES MADUROS (COLOR 2)',     'key': 'tomates_maduros',    'label': 'Tomates maduros (color 2)','unit': 'conteo', 'min': 0, 'max': 10,  'step': 1},
    {'col': 'DIAMETRO DEL FRUTO cm',         'key': 'diametro_fruto',     'label': 'Diámetro del fruto',       'unit': 'cm',     'min': 0, 'max': 60,  'step': 0.1},
    {'col': 'CRECIMIENTO PLANTA (cm)',       'key': 'crecimiento_planta', 'label': 'Crecimiento de planta',    'unit': 'cm',     'min': 0, 'max': 50,  'step': 0.5},
]
PHENO_COLS = [f['col'] for f in PHENO_FIELDS]

# ── Sensores (histórico) ────────────────────────────────────────────────────
# var_key → (fuente, columna en el Excel). 'suelo' no tiene datos históricos.
SENSOR_VARS = {
    'temp': {'source': 'internas', 'col': 'Temperatura promedio',            'label': 'Temperatura',      'unit': '°C'},
    'hr':   {'source': 'internas', 'col': 'Humedad relativa promedio (%)',   'label': 'Humedad relativa', 'unit': '%'},
    'co2':  {'source': 'internas', 'col': 'CO2 [ppm]',                       'label': 'CO₂',              'unit': 'ppm'},
    'ce':   {'source': 'riego',    'col': 'CE promedio [mS/cm]',             'label': 'Conductividad (CE)','unit': 'mS/cm'},
    'par':  {'source': 'exterior', 'col': 'Intensidad de radiación maxima [W/m²]', 'label': 'Radiación (PAR)', 'unit': 'W/m²'},
}

MONTH_LABELS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']


def _norm(s: str) -> str:
    """minúsculas sin acentos para emparejar columnas de forma robusta."""
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return s.strip().lower()


def _find_file(patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = glob.glob(str(DATA_DIR / pat))
        if hits:
            return Path(hits[0])
    return None


def _internas_file(inv: int) -> Path | None:
    return _find_file([f'Variables internas [Ii]nvernadero {inv}.xlsx'])


def _riego_file(inv: int) -> Path | None:
    return _find_file([f'Variables riego [Ii]nvernadero {inv}.xlsx'])


def _exterior_file() -> Path | None:
    return _find_file(['Variables exteriores.xlsx'])


def _match_col(df: pd.DataFrame, target: str) -> str | None:
    nt = _norm(target)
    for c in df.columns:
        if _norm(c) == nt:
            return c
    return None


# ── Fenología ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def phenology_weekly(inv: int) -> dict:
    """Promedio semanal (sobre plantas) de las 8 vars de fenología, por temporada."""
    series = {f['key']: [] for f in PHENO_FIELDS}
    seasons_out = []
    for s in SEASONS:
        fp = DATA_DIR / f'Monitoreo de Fenologia - Temporada {s}.xlsx'
        if not fp.exists():
            continue
        df = pd.read_excel(fp, sheet_name='BD TOMATE', header=1)
        df['INVERNADERO'] = pd.to_numeric(df['INVERNADERO'], errors='coerce')
        df = df[df['INVERNADERO'] == inv].copy()
        if df.empty:
            continue
        week_col = _match_col(df, 'SEMANA DEL AÑO')
        present = {f['key']: _match_col(df, f['col']) for f in PHENO_FIELDS}
        for f in PHENO_FIELDS:
            c = present[f['key']]
            if c:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        agg_cols = [c for c in present.values() if c]
        weekly = df.groupby(week_col)[agg_cols].mean().reset_index().sort_values(week_col)
        seasons_out.append({
            'season': SEASON_LABELS[s],
            'weeks': [int(w) for w in weekly[week_col].tolist()],
            'values': {
                f['key']: [None if pd.isna(v) else round(float(v), 2)
                           for v in (weekly[present[f['key']]].tolist() if present[f['key']] else [])]
                for f in PHENO_FIELDS
            },
        })
    # Último valor capturado (de la última temporada con datos) por variable.
    latest = {}
    if seasons_out:
        last = seasons_out[-1]
        for f in PHENO_FIELDS:
            vals = [v for v in last['values'][f['key']] if v is not None]
            latest[f['key']] = vals[-1] if vals else None
    return {
        'inv_id': inv,
        'fields': PHENO_FIELDS,
        'seasons': seasons_out,
        'latest': latest,
        'latest_season': seasons_out[-1]['season'] if seasons_out else None,
    }


def validate_and_average(rows: list[dict]) -> dict:
    """Valida filas-por-planta y devuelve el promedio por variable (stub demo, sin guardar)."""
    if not rows:
        raise ValueError('Sin filas de plantas para procesar.')
    field_by_key = {f['key']: f for f in PHENO_FIELDS}
    sums = {k: 0.0 for k in field_by_key}
    counts = {k: 0 for k in field_by_key}
    errors = []
    for i, row in enumerate(rows, 1):
        for key, f in field_by_key.items():
            if key not in row or row[key] is None or row[key] == '':
                continue
            try:
                v = float(row[key])
            except (TypeError, ValueError):
                errors.append(f'Planta {i}: "{f["label"]}" no es numérico.')
                continue
            if v < f['min'] or v > f['max']:
                errors.append(f'Planta {i}: "{f["label"]}" = {v} fuera de rango [{f["min"]}, {f["max"]}].')
                continue
            sums[key] += v
            counts[key] += 1
    if errors:
        raise ValueError(' '.join(errors))
    averages = {k: (round(sums[k] / counts[k], 2) if counts[k] else None) for k in field_by_key}
    return {'n_plants': len(rows), 'averages': averages}


# ── Histórico de sensores ───────────────────────────────────────────────────
@lru_cache(maxsize=32)
def sensor_history(inv: int, var: str) -> dict:
    """Promedio mensual por temporada de una variable de sensor (una serie por temporada)."""
    if var not in SENSOR_VARS:
        raise KeyError(f'Variable de sensor desconocida: {var}')
    meta = SENSOR_VARS[var]
    if meta['source'] == 'internas':
        fp = _internas_file(inv)
    elif meta['source'] == 'riego':
        fp = _riego_file(inv)
    else:
        fp = _exterior_file()
    if fp is None or not fp.exists():
        return {'inv_id': inv, 'var': var, 'meta': meta, 'series': [], 'months': MONTH_LABELS}

    xl = pd.ExcelFile(fp)
    series = []
    for s in SEASONS:
        sheet = next((sh for sh in xl.sheet_names if str(s) in sh and 'emporada' in sh), None)
        if sheet is None:
            continue
        df = pd.read_excel(fp, sheet_name=sheet)
        date_col = next((c for c in df.columns if _norm(c).startswith('fecha')), None)
        val_col = _match_col(df, meta['col'])
        if date_col is None or val_col is None:
            continue
        df['_d'] = pd.to_datetime(df[date_col], errors='coerce')
        df['_v'] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=['_d', '_v'])
        if df.empty:
            continue
        df['_m'] = df['_d'].dt.month
        monthly = df.groupby('_m')['_v'].mean()
        # 12 posiciones (ene..dic); None donde la temporada no tiene ese mes.
        points = [None if m not in monthly.index else round(float(monthly[m]), 2)
                  for m in range(1, 13)]
        series.append({'season': SEASON_LABELS[s], 'points': points})
    return {'inv_id': inv, 'var': var, 'meta': meta, 'months': MONTH_LABELS, 'series': series}
