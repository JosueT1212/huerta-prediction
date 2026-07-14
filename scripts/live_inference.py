"""
Weekly inference cron script — runs every Sunday 8am UTC.

Railway Cron:
  Schedule: 0 8 * * 0
  Command:  python scripts/live_inference.py

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_KEY
Optional:
  DRY_RUN=1  — run models but skip Supabase write
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
import joblib
from supabase import create_client

from backend.live_features import build_input_tensor

# Model class verified via:
# grep -n "^class" Models/cnn_rnn_yield.py  → CNNRNN at line 757
from Models.cnn_rnn_yield import CNNRNN

RESULTS_DIR = ROOT / "Models" / "results"
HORIZON_WEEKS = 5
MODEL_VERSION = "cnn_rnn_v2_production"
HISTORICAL_MEAN_VERSION = "historical_mean_v1"
SKIP_FIRST_WEEKS = 3
CURRENT_SEASON = "T18"
INV_IDS = (3, 4)


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_in_season(target_date: date, transplant_date: date) -> int:
    return max(0, (target_date - transplant_date).days // 7)


def load_historical_mean(inv: int, wis: int) -> float | None:
    path = RESULTS_DIR / f"historical_mean_inv{inv}.json"
    with open(path) as f:
        means = json.load(f)
    return means.get(str(wis))


def get_transplant_date(supa, inv: int) -> date | None:
    resp = (
        supa.table("transplant_dates")
        .select("fecha")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        return None
    return date.fromisoformat(row["fecha"])


def run_inference_for_greenhouse(supa, inv: int, dry_run: bool) -> None:
    transplant_date = get_transplant_date(supa, inv)
    if transplant_date is None:
        print(f"  ERROR: no transplant_date set for invernadero {inv} — skipping. "
              f"Set it via PUT /transplant-date/{inv}.", file=sys.stderr)
        return

    predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS))
    wis_target = week_in_season(predicted_for, transplant_date)

    if wis_target < SKIP_FIRST_WEEKS:
        historical_kg = load_historical_mean(inv, wis_target)
        if historical_kg is not None:
            print(f"  Inv{inv} → week-in-season {wis_target} < {SKIP_FIRST_WEEKS} "
                  f"(ramp-up) → historical mean {historical_kg:.1f} kg")
            _upsert_prediction(supa, inv, predicted_for, historical_kg,
                                HISTORICAL_MEAN_VERSION, dry_run)
            return

    cutoff = (date.today() - timedelta(weeks=12)).isoformat()

    sensor_resp = (
        supa.table("sensor_readings_wide").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).execute()
    )
    riego_resp = (
        supa.table("riego_readings").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).execute()
    )
    # exterior_readings is global (no greenhouse_id column) — same weather feeds both invernaderos
    ext_resp = (
        supa.table("exterior_readings").select("*")
        .gte("fecha", cutoff).execute()
    )

    merged_by_fecha: dict[str, dict] = {}
    for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
        merged_by_fecha.setdefault(row["fecha"], {}).update(row)
    wide_rows = list(merged_by_fecha.values())
    print(f"  Inv{inv} sensor rows pulled: {len(sensor_resp.data or [])} sensores + "
          f"{len(riego_resp.data or [])} riego + {len(ext_resp.data or [])} exteriores "
          f"→ {len(wide_rows)} merged by fecha")

    pheno_resp = (
        supa.table("phenology_observations").select("*")
        .eq("greenhouse_id", inv).gte("week_date", cutoff).execute()
    )
    pheno_rows = pheno_resp.data or []

    pipeline_path = RESULTS_DIR / f"production_pipeline_inv{inv}.pkl"
    x_sensor, x_temporal = build_input_tensor(
        inv, wide_rows, pheno_rows,
        transplant_date=transplant_date,
        pipeline_path=pipeline_path,
    )

    pipeline = joblib.load(pipeline_path)
    scaler_y = pipeline["scaler_y"]
    bc_lambda = pipeline["bc_lambda"]
    transform = pipeline["transform"]

    with open(ROOT / "Models" / f"hp_inv{inv}.yaml") as f:
        hp = yaml.safe_load(f)

    model = CNNRNN(
        n_sensor=x_sensor.shape[-1],
        n_temporal=x_temporal.shape[-1],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        cnn_padding=hp["cnn_padding"],
        num_cnn_blocks=hp["num_cnn_blocks"],
        lstm_hidden=hp["lstm_hidden"],
        lstm_layers=hp["lstm_layers"],
        dropout=0.0,
        fc_hidden=hp["fc_hidden"],
    )
    model_path = RESULTS_DIR / f"production_cnn_rnn_inv{inv}.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        y_norm = model(x_sensor, x_temporal).squeeze().item()

    y_scaled = scaler_y.inverse_transform([[y_norm]])[0][0]

    if transform == "log":
        import numpy as np
        kg_predicted = float(np.expm1(y_scaled))
    elif transform == "boxcox" and bc_lambda is not None:
        from scipy.special import inv_boxcox
        kg_predicted = float(inv_boxcox(y_scaled, bc_lambda))
    else:
        kg_predicted = float(y_scaled)

    print(f"  Inv{inv} → {kg_predicted:.1f} kg for week {predicted_for}")
    _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)


def _upsert_prediction(supa, inv: int, predicted_for: date, kg_predicted: float,
                        model_version: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] would upsert inv{inv}: {predicted_for} → {kg_predicted:.1f} kg "
              f"({model_version})")
        return
    supa.table("predictions").upsert({
        "greenhouse_id": inv,
        "predicted_for": predicted_for.isoformat(),
        "kg_predicted": kg_predicted,
        "model_version": model_version,
        "season": CURRENT_SEASON,
    }, on_conflict="greenhouse_id,predicted_for").execute()
    print(f"  Inv{inv} written to predictions table.")


def main():
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    supa = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    succeeded = 0
    for inv in INV_IDS:
        try:
            run_inference_for_greenhouse(supa, inv, dry_run)
            succeeded += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: invernadero {inv} failed: {e}", file=sys.stderr)

    if succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
