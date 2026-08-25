from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter()


class ActualKgRequest(BaseModel):
    # None clears a real production value the client entered by mistake
    kg_actual: float | None


@router.get("/live-predictions/{inv}")
def list_predictions(
    inv: int,
    limit: int = Query(20, ge=1, le=200),
    season: str | None = Query(None),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    query = (
        service_client.table("predictions")
        .select("id, greenhouse_id, predicted_for, predicted_at, kg_predicted, kg_actual, model_version, season")
        .eq("greenhouse_id", inv)
    )
    if season is not None:
        query = query.eq("season", season)
    resp = query.order("predicted_for", desc=True).limit(limit).execute()
    return resp.data


@router.patch("/live-predictions/{inv}/{prediction_id}")
def update_kg_actual(
    inv: int,
    prediction_id: int,
    body: ActualKgRequest,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    service_client.table("predictions").update(
        {"kg_actual": body.kg_actual}
    ).eq("id", prediction_id).eq("greenhouse_id", inv).execute()
    return {"ok": True}


