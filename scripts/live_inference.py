"""
Weekly inference cron script — runs every Sunday 8am UTC.

Railway Cron:
  Schedule: 0 8 * * 0
  Command:  python scripts/live_inference.py

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_KEY, TRANSPLANT_DATE_INV3
Optional:
  DRY_RUN=1  — run model but skip Supabase write
"""
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

RESULTS_DIR   = ROOT / "Models" / "results"
INV_ID        = 3
MODEL_PATH    = RESULTS_DIR / "best_cnn_rnn_inv3.pt"
PIPELINE_PATH = RESULTS_DIR / "pipeline_inv3.pkl"
HORIZON_WEEKS = 5
MODEL_VERSION = "cnn_rnn_v1"


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def main():
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    transplant_date_str = os.environ.get("TRANSPLANT_DATE_INV3", "")
    if not transplant_date_str:
        print("ERROR: TRANSPLANT_DATE_INV3 not set", file=sys.stderr)
        sys.exit(1)
    transplant_date = date.fromisoformat(transplant_date_str)

    supa = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    cutoff = (date.today() - timedelta(weeks=12)).isoformat()

    wide_resp = (
        supa.table("sensor_readings_wide")
        .select("*")
        .eq("greenhouse_id", INV_ID)
        .gte("fecha", cutoff)
        .execute()
    )
    wide_rows = wide_resp.data or []
    print(f"  Wide sensor rows pulled: {len(wide_rows)}")

    pheno_resp = (
        supa.table("phenology_observations")
        .select("*")
        .eq("greenhouse_id", INV_ID)
        .gte("week_date", cutoff)
        .execute()
    )
    pheno_rows = pheno_resp.data or []
    print(f"  Phenology observations pulled: {len(pheno_rows)}")

    x_sensor, x_temporal = build_input_tensor(
        INV_ID, wide_rows, pheno_rows,
        transplant_date=transplant_date,
        pipeline_path=PIPELINE_PATH,
    )

    pipeline  = joblib.load(PIPELINE_PATH)
    scaler_y  = pipeline["scaler_y"]
    bc_lambda = pipeline["bc_lambda"]
    transform = pipeline["transform"]

    # Load hyperparams from hp_inv3.yaml
    # Keys used: cnn_filters, cnn_kernel_size, cnn_padding, num_cnn_blocks,
    #            lstm_hidden, lstm_layers, fc_hidden
    with open(ROOT / "Models" / "hp_inv3.yaml") as f:
        hp = yaml.safe_load(f)

    # CNNRNN __init__ signature (verified line 767-768 of cnn_rnn_yield.py):
    # (n_sensor, n_temporal, cnn_filters, cnn_kernel_size, cnn_padding,
    #  num_cnn_blocks, lstm_hidden, lstm_layers, dropout, fc_hidden, n_out=1)
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
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
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

    predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS)).isoformat()
    print(f"  Inv{INV_ID} → {kg_predicted:.1f} kg for week {predicted_for}")

    if dry_run:
        print("  [DRY RUN] skipping Supabase write")
        return

    supa.table("predictions").upsert({
        "greenhouse_id": INV_ID,
        "predicted_for": predicted_for,
        "kg_predicted":  kg_predicted,
        "model_version": MODEL_VERSION,
    }, on_conflict="greenhouse_id,predicted_for").execute()
    print("  Written to predictions table.")


if __name__ == "__main__":
    main()
