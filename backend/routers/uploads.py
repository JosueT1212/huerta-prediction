import io
from typing import Annotated
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from backend.auth import get_current_user
from backend.supabase_client import service_client
from backend.lock_utils import touch_submission_lock, get_lock_status
from backend.inference_trigger import maybe_trigger_inference

router = APIRouter()

SENSOR_COLS = [
    "fecha", "temp_prom_int", "temp_min_int", "temp_max_int", "hr_prom_int",
    "co2_ppm", "deficit_humedad", "deficit_presion_vapor", "humedad_abs_int",
]
RIEGO_COLS = ["fecha", "riego_total", "ph_promedio", "ce_promedio"]
PHENOLOGY_COLS = [
    "fecha", "zona", "planta", "racimos_puestos", "flores_racimo_abiertas",
    "racimos_en_planta", "cantidad_tomates", "racimo_en_cosecha",
    "tomates_maduros", "diametro_fruto_cm", "crecimiento_planta_cm",
]
PRODUCTION_COLS = ["fecha", "kg_reales"]
EXTERIOR_COLS = [
    "fecha", "temp_prom_ext", "temp_max_ext", "temp_min_ext", "hr_prom_ext",
    "rad_sum", "rad_max", "dh_ext", "humedad_abs_ext",
]

FORM_TYPES = {
    "sensores": SENSOR_COLS,
    "riego": RIEGO_COLS,
    "fenologia": PHENOLOGY_COLS,
    "produccion": PRODUCTION_COLS,
    "exteriores": EXTERIOR_COLS,
}

PER_INV_TYPES = {"sensores", "riego", "fenologia", "produccion"}

EXTERIORES_GH_ID = 0  # sentinel: exteriores has no real invernadero (0 is never a real id)


def _require_form_type(form_type: str) -> list[str]:
    cols = FORM_TYPES.get(form_type)
    if cols is None:
        raise HTTPException(422, f"form_type inválido: {form_type}")
    return cols


def _table_for(form_type: str) -> str:
    return {
        "sensores": "sensor_readings_wide",
        "riego": "riego_readings",
        "fenologia": "phenology_observations",
        "produccion": "predictions",
    }[form_type]


def _date_str(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _parse_excel(contents: bytes, required_cols: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
    except Exception as e:
        raise HTTPException(422, f"No se pudo leer el archivo Excel: {e}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Columnas faltantes: {', '.join(missing)}")
    df = df[required_cols].dropna(how="all")
    df = df.where(pd.notna(df), None)
    return df


def _ingest_rows(df: pd.DataFrame, form_type: str, greenhouse_id: int | None):
    required_cols = FORM_TYPES[form_type]
    rows_inserted = rows_updated = rows_skipped = 0
    skipped_reasons: list[str] = []

    if form_type in ("sensores", "riego"):
        table = _table_for(form_type)
        for _, row in df.iterrows():
            record = {"greenhouse_id": greenhouse_id, "fecha": _date_str(row["fecha"])}
            record.update({c: row[c] for c in required_cols if c != "fecha"})
            service_client.table(table).upsert(
                record, on_conflict="greenhouse_id,fecha"
            ).execute()
            rows_inserted += 1

    elif form_type == "exteriores":
        for _, row in df.iterrows():
            record = {"fecha": _date_str(row["fecha"])}
            record.update({c: row[c] for c in required_cols if c != "fecha"})
            service_client.table("exterior_readings").upsert(
                record, on_conflict="fecha"
            ).execute()
            rows_inserted += 1

    elif form_type == "fenologia":
        for _, row in df.iterrows():
            try:
                zona = int(row["zona"])
                planta = int(row["planta"])
            except (ValueError, TypeError):
                raise HTTPException(
                    422,
                    f"Valores no numéricos en zona/planta para la fila con fecha "
                    f"{_date_str(row['fecha'])}: zona={row['zona']!r}, planta={row['planta']!r}",
                )
            record = {
                "greenhouse_id": greenhouse_id,
                "week_date": _date_str(row["fecha"]),
                "zona": zona,
                "planta": planta,
            }
            record.update({c: row[c] for c in required_cols if c not in ("fecha", "zona", "planta")})
            service_client.table("phenology_observations").upsert(
                record, on_conflict="greenhouse_id,week_date,zona,planta"
            ).execute()
            rows_inserted += 1

    elif form_type == "produccion":
        for _, row in df.iterrows():
            resp = (
                service_client.table("predictions")
                .update({"kg_actual": row["kg_reales"]})
                .eq("greenhouse_id", greenhouse_id)
                .eq("predicted_for", _date_str(row["fecha"]))
                .execute()
            )
            if resp.data:
                rows_updated += 1
            else:
                rows_skipped += 1
                skipped_reasons.append(
                    f"{_date_str(row['fecha'])}: no existe predicción para esa semana"
                )

    return rows_inserted, rows_updated, rows_skipped, skipped_reasons


@router.post("/uploads/{inv}/{form_type}")
async def upload_excel(
    inv: int,
    form_type: str,
    file: UploadFile,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /uploads/{form_type}")
    required_cols = _require_form_type(form_type)

    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, form_type, inv)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(inv, form_type)
        if form_type != "produccion":
            await run_in_threadpool(maybe_trigger_inference, form_type, inv)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/{inv}/{form_type}/history")
def upload_history(
    inv: int,
    form_type: str,
    limit: int = 200,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /uploads/{form_type}/history")
    _require_form_type(form_type)
    table = _table_for(form_type)
    order_col = {
        "sensores": "fecha", "riego": "fecha",
        "fenologia": "week_date", "produccion": "predicted_for",
    }[form_type]
    query = service_client.table(table).select("*").eq("greenhouse_id", inv)
    if form_type == "produccion":
        # predictions rows exist for every model-predicted week, not just the
        # ones the client actually submitted real kg for — a "capture
        # history" must only show weeks with a real value on file.
        query = query.not_.is_("kg_actual", "null")
    resp = query.order(order_col, desc=True).limit(limit).execute()
    return resp.data


@router.get("/submission-lock/{inv}/{form_type}")
def lock_status(
    inv: int,
    form_type: str,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /submission-lock/{form_type}")
    _require_form_type(form_type)
    return get_lock_status(inv, form_type)


@router.post("/uploads/exteriores")
async def upload_exteriores(
    file: UploadFile,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    required_cols = FORM_TYPES["exteriores"]

    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, "exteriores", None)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(EXTERIORES_GH_ID, "exteriores")
        await run_in_threadpool(maybe_trigger_inference, "exteriores", None)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/exteriores/history")
def upload_history_exteriores(
    limit: int = 200,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("exterior_readings")
        .select("*")
        .order("fecha", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


@router.get("/submission-lock/exteriores")
def lock_status_exteriores(
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    return get_lock_status(EXTERIORES_GH_ID, "exteriores")
