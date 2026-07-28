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

from backend.live_features import build_input_tensor, aggregate_wide_to_weekly

# Model class verified via:
# grep -n "^class" Models/cnn_rnn_yield.py  → CNNRNN at line 757
from Models.cnn_rnn_yield import CNNRNN

RESULTS_DIR = ROOT / "Models" / "results"
HORIZON_WEEKS = 5
MODEL_VERSION = "cnn_rnn_v2_production"
CURRENT_SEASON = "T18"
INV_IDS = (3, 4)


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_in_season(target_date: date, transplant_date: date) -> int:
    return max(0, (target_date - transplant_date).days // 7)


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


def get_harvest_start(supa, inv: int) -> date | None:
    """Admin-set fact — see harvest_start_dates table (010_harvest_start_dates.sql).

    NOT inferred from sensor_readings_wide: sensors start recording ~10-11
    weeks before harvest (CLAUDE.md §5) and the client's first upload of a
    season batches that entire backlog in one payload, so MIN(fecha) from
    sensor data resolves to the pre-harvest backlog date, not the season's
    actual week 1. See docs/superpowers/specs/2026-07-15-first-submission-backfill-design.md §1.
    """
    resp = (
        supa.table("harvest_start_dates")
        .select("fecha")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        return None
    return monday_of_week(date.fromisoformat(row["fecha"]))


def is_first_submission(supa, inv: int) -> bool:
    """True iff no predictions row exists yet for this greenhouse+season."""
    resp = (
        supa.table("predictions")
        .select("id")
        .eq("greenhouse_id", inv)
        .eq("season", CURRENT_SEASON)
        .limit(1)
        .execute()
    )
    return not (resp.data or [])


def _run_model_forward(supa, inv: int, transplant_date: date, as_of_date: date,
                        wis_target: int) -> float | None:
    """CNN-RNN ensemble forward pass for a sensor window ending at as_of_date.

    Returns None if the window doesn't have seq_len real (non-padded) weekly
    rows yet — the caller must not write a prediction built on fabricated
    zero-padding.

    wis_target: week-in-season of the PREDICTED week (not as_of_date) — used
    to add the historical mean back when the model was trained in residual
    target_mode (predicts kg - mean(week_in_season), not raw kg).
    """
    cutoff = (as_of_date - timedelta(weeks=12)).isoformat()
    upper = as_of_date.isoformat()

    sensor_resp = (
        supa.table("sensor_readings_wide").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).lte("fecha", upper).execute()
    )
    riego_resp = (
        supa.table("riego_readings").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).lte("fecha", upper).execute()
    )
    # exterior_readings is global (no greenhouse_id column) — same weather feeds both invernaderos
    ext_resp = (
        supa.table("exterior_readings").select("*")
        .gte("fecha", cutoff).lte("fecha", upper).execute()
    )

    merged_by_fecha: dict[str, dict] = {}
    for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
        merged_by_fecha.setdefault(row["fecha"], {}).update(row)
    wide_rows = list(merged_by_fecha.values())
    print(f"  Inv{inv} sensor rows pulled (as of {as_of_date}): {len(sensor_resp.data or [])} sensores + "
          f"{len(riego_resp.data or [])} riego + {len(ext_resp.data or [])} exteriores "
          f"→ {len(wide_rows)} merged by fecha")

    pheno_resp = (
        supa.table("phenology_observations").select("*")
        .eq("greenhouse_id", inv).gte("week_date", cutoff).lte("week_date", upper).execute()
    )
    pheno_rows = pheno_resp.data or []

    pipeline_path = RESULTS_DIR / f"production_pipeline_inv{inv}.pkl"
    pipeline = joblib.load(pipeline_path)
    seq_len = pipeline["seq_len"]
    sensor_cols = pipeline["sensor_cols"]

    weekly_check = aggregate_wide_to_weekly(wide_rows, sensor_cols)
    if len(weekly_check) < seq_len:
        print(f"  Inv{inv} → only {len(weekly_check)} weeks of sensor data as of {as_of_date}, "
              f"need {seq_len} — not enough history yet.")
        return None

    x_sensor, x_temporal = build_input_tensor(
        inv, wide_rows, pheno_rows,
        transplant_date=transplant_date,
        pipeline_path=pipeline_path,
    )

    scaler_y = pipeline["scaler_y"]
    bc_lambda = pipeline["bc_lambda"]
    transform = pipeline["transform"]

    with open(ROOT / "Models" / f"hp_inv{inv}.yaml") as f:
        hp = yaml.safe_load(f)

    model_path = RESULTS_DIR / f"production_cnn_rnn_inv{inv}.pt"
    # map_location="cpu": checkpoints were saved on Apple Silicon (mps
    # device) and Railway's container is CPU-only Linux — without this,
    # torch.load fails with "Storage device not recognized: mps".
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # 10-seed production ensemble: checkpoint is a LIST of state_dicts (one
    # per seed), not a single state_dict — run all members, average in kg
    # space (post inverse-transform, so it's correct even for nonlinear
    # transforms).
    state_dicts = checkpoint if isinstance(checkpoint, list) else [checkpoint]

    member_kg_preds = []
    for state_dict in state_dicts:
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
            fc_layers=hp.get("fc_layers", 1),
        )
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            y_norm = model(x_sensor, x_temporal).squeeze().item()

        y_scaled = scaler_y.inverse_transform([[y_norm]])[0][0]

        if transform == "log":
            import numpy as np
            kg_member = float(np.expm1(y_scaled))
        elif transform == "boxcox" and bc_lambda is not None:
            from scipy.special import inv_boxcox
            kg_member = float(inv_boxcox(y_scaled, bc_lambda))
        else:
            kg_member = float(y_scaled)

        member_kg_preds.append(kg_member)

    kg_predicted = sum(member_kg_preds) / len(member_kg_preds)

    if pipeline.get("target_mode") == "residual":
        mu_wis = pipeline.get("mu_wis") or {}
        global_mu = pipeline.get("global_mu")
        historical_baseline = mu_wis.get(str(wis_target), mu_wis.get(wis_target, global_mu))
        kg_predicted += historical_baseline

    return kg_predicted


def _backfill_first_submission(supa, inv: int, transplant_date: date, harvest_start: date,
                                dry_run: bool) -> bool:
    """First submission of the season: write every model week the current
    sensor backlog supports, starting from week-in-season 0 — no hardcoded
    row count (spec §3)."""
    wrote_any = False

    wis = 0
    while True:
        as_of_date = harvest_start + timedelta(weeks=wis - HORIZON_WEEKS)
        if as_of_date > date.today():
            print(f"  Inv{inv} → backfilled through week-in-season {wis - 1}; "
                  f"week-in-season {wis} would need sensor data from the future.")
            break
        kg_predicted = _run_model_forward(supa, inv, transplant_date, as_of_date, wis)
        if kg_predicted is None:
            print(f"  Inv{inv} → backfilled through week-in-season {wis - 1}; "
                  f"insufficient sensor history for week-in-season {wis}.")
            break
        predicted_for = harvest_start + timedelta(weeks=wis)
        _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)
        wrote_any = True
        wis += 1

    return wrote_any


def run_inference_for_greenhouse(supa, inv: int, dry_run: bool) -> bool:
    transplant_date = get_transplant_date(supa, inv)
    if transplant_date is None:
        print(f"  ERROR: no transplant_date set for invernadero {inv} — skipping. "
              f"Set it via PUT /transplant-date/{inv}.", file=sys.stderr)
        return False

    harvest_start = get_harvest_start(supa, inv)
    if harvest_start is None:
        print(f"  ERROR: no harvest_start set for invernadero {inv} — skipping. "
              f"Set it via PUT /harvest-start-date/{inv}.", file=sys.stderr)
        return False

    if is_first_submission(supa, inv):
        return _backfill_first_submission(supa, inv, transplant_date, harvest_start, dry_run)

    predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS))
    wis_target = week_in_season(predicted_for, harvest_start)

    kg_predicted = _run_model_forward(supa, inv, transplant_date, date.today(), wis_target)
    if kg_predicted is None:
        print(f"  ERROR: insufficient sensor history for invernadero {inv} — skipping.",
              file=sys.stderr)
        return False

    print(f"  Inv{inv} → {kg_predicted:.1f} kg for week {predicted_for}")
    _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)
    return True


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
            if run_inference_for_greenhouse(supa, inv, dry_run):
                succeeded += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: invernadero {inv} failed: {e}", file=sys.stderr)

    if succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
