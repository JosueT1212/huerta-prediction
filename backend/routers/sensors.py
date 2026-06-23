from fastapi import APIRouter, Depends, Query
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter()


@router.get("/sensors/{inv}")
def get_sensors(
    inv: int,
    limit: int = Query(100, ge=1, le=1000),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("sensor_readings")
        .select("sensor_name, value, recorded_at")
        .eq("greenhouse_id", inv)
        .order("recorded_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data
