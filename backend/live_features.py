"""
Live feature engineering: Supabase rows → model-ready tensors for Inv3.

Sensor: sensor_readings_wide (daily) → aggregate weekly → scaler_Xs → PCA
Phenology: phenology_observations (per plant) → mean per week → scaler_Xp
Temporal: transplant_date + week → dias_desde_transplante, week_in_season → scaler_Xt
"""
import os
from datetime import date, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

PIPELINE_PATH = Path("Models/results/pipeline_inv3.pkl")
SUM_SENSORS = {"rad_sum", "riego_total"}

PHENO_DB_TO_MODEL = {
    "racimos_puestos":        "RACIMOS PUESTOS",
    "flores_racimo_abiertas": "FLORES EN RACIMO ABIERTAS",
    "racimos_en_planta":      "CANTIDAD DE RACIMOS EN PLANTA",
    "cantidad_tomates":       "CANTIDAD DE TOMATES",
    "racimo_en_cosecha":      "Nº DE RACIMO EN COSECHA",
    "tomates_maduros":        "TOMATES MADUROS (COLOR 2)",
    "diametro_fruto_cm":      "DIAMETRO DEL FRUTO cm",
    "crecimiento_planta_cm":  "CRECIMIENTO PLANTA (cm)",
}


def aggregate_wide_to_weekly(wide_rows: list[dict], sensor_cols: list[str]) -> pd.DataFrame:
    """Group sensor_readings_wide daily rows by ISO week → one row per week."""
    if not wide_rows:
        return pd.DataFrame(columns=["year_week"] + sensor_cols)

    df = pd.DataFrame(wide_rows)
    df["fecha"] = pd.to_datetime(df["fecha"])
    iso = df["fecha"].dt.isocalendar()
    df["year_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

    agg = {}
    for col in sensor_cols:
        if col not in df.columns:
            continue
        agg[col] = "sum" if col in SUM_SENSORS else "mean"

    weekly = df.groupby("year_week").agg(agg).reset_index()
    for col in sensor_cols:
        if col not in weekly.columns:
            weekly[col] = np.nan

    return weekly.sort_values("year_week").reset_index(drop=True)


def aggregate_pheno_to_weekly(pheno_rows: list[dict]) -> pd.DataFrame:
    """Average phenology_observations across all zona/planta per week.

    Matches load_phenology_features() training logic: .groupby('SEMANA DEL AÑO').mean()
    """
    if not pheno_rows:
        return pd.DataFrame()

    df = pd.DataFrame(pheno_rows)
    pheno_cols_db = list(PHENO_DB_TO_MODEL.keys())
    available = [c for c in pheno_cols_db if c in df.columns]
    if not available:
        return pd.DataFrame()

    weekly = df.groupby("week_date")[available].mean().reset_index()
    return weekly.sort_values("week_date").reset_index(drop=True)


def _compute_temporal(year_weeks: list[str], transplant_date: date) -> np.ndarray:
    rows = []
    for yw in year_weeks:
        try:
            monday = date.fromisoformat(f"{yw}-1")
        except Exception:
            monday = transplant_date
        dias = max(0, (monday - transplant_date).days + 3)
        wis = max(0, (monday - transplant_date).days // 7)
        rows.append([float(dias), float(wis)])
    return np.array(rows, dtype=np.float32)


def build_input_tensor(
    inv: int,
    wide_rows: list[dict],
    pheno_rows: list[dict],
    transplant_date: date | None = None,
    pipeline_path: Path = PIPELINE_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_sensor, x_temporal) tensors ready for model.forward().

    x_sensor:  (1, seq_len, n_pca + n_pheno)
    x_temporal:(1, seq_len, n_temporal)
    """
    pipeline = joblib.load(pipeline_path)
    seq_len = pipeline["seq_len"]
    sensor_cols = pipeline["sensor_cols"]
    pheno_cols = pipeline["pheno_cols"]   # model feature names (uppercase)
    scaler_Xs = pipeline["scaler_Xs"]
    pca = pipeline["pca"]
    scaler_Xp = pipeline["scaler_Xp"]
    scaler_Xt = pipeline["scaler_Xt"]
    pheno_means = pipeline["pheno_means"]

    if transplant_date is None:
        raw = os.environ.get("TRANSPLANT_DATE_INV3", "")
        transplant_date = date.fromisoformat(raw) if raw else date(2026, 1, 1)

    # ── Sensor: daily → weekly ─────────────────────────────────────────────
    weekly = aggregate_wide_to_weekly(wide_rows, sensor_cols)
    weekly = weekly.tail(seq_len).reset_index(drop=True)

    if len(weekly) < seq_len:
        pad_rows = seq_len - len(weekly)
        pad_data = {col: [0.0] * pad_rows for col in weekly.columns}
        pad = pd.DataFrame(pad_data)
        weekly = pd.concat([pad, weekly], ignore_index=True)

    Xs = weekly[sensor_cols].fillna(0.0).values.astype(np.float32)
    Xs = scaler_Xs.transform(Xs)
    Xs = pca.transform(Xs)

    # ── Phenology: per-plant → weekly mean → scale ─────────────────────────
    if pheno_cols and scaler_Xp is not None:
        pheno_weekly = aggregate_pheno_to_weekly(pheno_rows)

        pheno_lookup: dict[str, dict] = {}
        if not pheno_weekly.empty:
            for _, row in pheno_weekly.iterrows():
                wd = str(row["week_date"])
                pheno_lookup[wd] = {
                    PHENO_DB_TO_MODEL[k]: float(row[k])
                    for k in PHENO_DB_TO_MODEL
                    if k in row and not pd.isna(row[k])
                }

        Xp_rows = []
        for _, wrow in weekly.iterrows():
            yw = str(wrow.get("year_week", ""))
            try:
                monday = date.fromisoformat(f"{yw}-1").isoformat()
            except Exception:
                monday = ""
            vals = pheno_lookup.get(monday, {})
            wis = max(0, (date.fromisoformat(monday) - transplant_date).days // 7) if monday else 0
            row_vals = []
            for col in pheno_cols:
                if col in vals:
                    row_vals.append(vals[col])
                else:
                    row_vals.append(float(pheno_means.get(col, {}).get(wis, 0.0)))
            Xp_rows.append(row_vals)

        Xp = np.array(Xp_rows, dtype=np.float32)
        Xp = scaler_Xp.transform(Xp)
        Xs = np.hstack([Xs, Xp])

    # ── Temporal ───────────────────────────────────────────────────────────
    year_weeks = list(weekly["year_week"].fillna("").astype(str))
    Xt = _compute_temporal(year_weeks, transplant_date)
    Xt = scaler_Xt.transform(Xt)

    x_sensor = torch.FloatTensor(Xs).unsqueeze(0)
    x_temporal = torch.FloatTensor(Xt).unsqueeze(0)

    return x_sensor, x_temporal
