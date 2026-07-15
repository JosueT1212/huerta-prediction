from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from backend.supabase_client import service_client

LOCK_WINDOW = timedelta(days=7)


def last_submitted_at(greenhouse_id: int, form_type: str) -> datetime | None:
    resp = (
        service_client.table("submission_locks")
        .select("last_submitted_at")
        .eq("greenhouse_id", greenhouse_id)
        .eq("form_type", form_type)
        .execute()
    )
    if not resp.data:
        return None
    raw = resp.data[0]["last_submitted_at"]
    return datetime.fromisoformat(raw.replace("Z", "+00:00")) if isinstance(raw, str) else raw


def check_submission_lock(greenhouse_id: int, form_type: str) -> None:
    last = last_submitted_at(greenhouse_id, form_type)
    if last is None:
        return
    next_allowed = last + LOCK_WINDOW
    if datetime.now(timezone.utc) < next_allowed:
        raise HTTPException(429, detail={"next_allowed_at": next_allowed.isoformat()})


def touch_submission_lock(greenhouse_id: int, form_type: str) -> None:
    service_client.table("submission_locks").upsert(
        {
            "greenhouse_id": greenhouse_id,
            "form_type": form_type,
            "last_submitted_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="greenhouse_id,form_type",
    ).execute()


def get_lock_status(greenhouse_id: int, form_type: str) -> dict:
    last = last_submitted_at(greenhouse_id, form_type)
    if last is None:
        return {"locked": False, "next_allowed_at": None}
    next_allowed = last + LOCK_WINDOW
    locked = datetime.now(timezone.utc) < next_allowed
    return {"locked": locked, "next_allowed_at": next_allowed.isoformat() if locked else None}
