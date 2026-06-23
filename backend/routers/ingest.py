import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Annotated
from backend.supabase_client import service_client

router = APIRouter()
INGEST_TOKEN = os.environ["INGEST_TOKEN"]


def verify_ingest_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not authorization or authorization != f"Bearer {INGEST_TOKEN}":
        raise HTTPException(401, "Invalid ingest token")


class SensorReading(BaseModel):
    greenhouse_id: int
    sensor_name: str
    value: float
    recorded_at: datetime


@router.post("/ingest/sensors")
def ingest_sensor(
    reading: SensorReading,
    _: None = Depends(verify_ingest_token),
):
    service_client.table("sensor_readings").insert({
        "greenhouse_id": reading.greenhouse_id,
        "sensor_name": reading.sensor_name,
        "value": reading.value,
        "recorded_at": reading.recorded_at.isoformat(),
    }).execute()
    # Dual-write: upsert the single column into the wide daily table
    WIDE_COLS = {
        "temp_prom_int", "temp_min_int", "temp_max_int", "hr_prom_int",
        "co2_ppm", "riego_total", "ph_promedio", "ce_promedio",
        "temp_prom_ext", "temp_max_ext", "temp_min_ext", "rad_sum",
    }
    if reading.sensor_name in WIDE_COLS:
        service_client.table("sensor_readings_wide").upsert(
            {
                "greenhouse_id": reading.greenhouse_id,
                "fecha": reading.recorded_at.date().isoformat(),
                reading.sensor_name: reading.value,
            },
            on_conflict="greenhouse_id,fecha",
        ).execute()
    return {"ok": True}
