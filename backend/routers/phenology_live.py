from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter()


class PhenologyObservation(BaseModel):
    week_date: str
    zona: int
    planta: int
    racimos_puestos: float | None = None
    flores_racimo_abiertas: float | None = None
    racimos_en_planta: float | None = None
    cantidad_tomates: float | None = None
    racimo_en_cosecha: float | None = None
    tomates_maduros: float | None = None
    diametro_fruto_cm: float | None = None
    crecimiento_planta_cm: float | None = None


@router.post("/phenology-live/{inv}")
def upsert_phenology(
    inv: int,
    body: PhenologyObservation,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    row = {"greenhouse_id": inv, **body.model_dump()}
    service_client.table("phenology_observations").upsert(
        row, on_conflict="greenhouse_id,week_date,zona,planta"
    ).execute()
    return {"ok": True}


@router.get("/phenology-live/{inv}")
def list_phenology(
    inv: int,
    limit: int = Query(100, ge=1, le=500),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("phenology_observations")
        .select("*")
        .eq("greenhouse_id", inv)
        .order("week_date", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


@router.delete("/phenology-live/{inv}/{obs_id}")
def delete_phenology(
    inv: int,
    obs_id: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("phenology_observations")
        .delete()
        .eq("id", obs_id)
        .eq("greenhouse_id", inv)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Observación no encontrada")
    return {"ok": True}
