from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import Annotated
from backend.auth import get_current_user
from backend.inference_trigger import maybe_trigger_inference
from backend.supabase_client import service_client

router = APIRouter()


class HarvestStartDateRequest(BaseModel):
    fecha: str


@router.get("/harvest-start-date/{inv}")
def get_harvest_start_date(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("harvest_start_dates")
        .select("greenhouse_id, fecha, updated_at")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        raise HTTPException(404, f"No hay fecha de inicio de cosecha para invernadero {inv}")
    return row


@router.put("/harvest-start-date/{inv}")
async def put_harvest_start_date(
    inv: int,
    body: HarvestStartDateRequest,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    service_client.table("harvest_start_dates").upsert(
        {"greenhouse_id": inv, "fecha": body.fecha}, on_conflict="greenhouse_id"
    ).execute()
    # La fecha de inicio de cosecha no es una carga semanal (no participa en
    # uploads_complete_for_inv), pero la inferencia la requiere: si el set
    # semanal ya estaba completo cuando se guardó, el trigger de uploads ya
    # corrió sin ella y se saltó — este re-chequeo cubre ese caso (mismo
    # patrón que backend/routers/transplant_dates.py).
    await run_in_threadpool(maybe_trigger_inference, "harvest_start_date", inv)
    return {"ok": True}
