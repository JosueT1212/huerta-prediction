from fastapi import Depends, HTTPException, Header
from typing import Annotated
from backend.supabase_client import service_client


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    try:
        resp = service_client.auth.get_user(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    user = resp.user
    if user is None:
        raise HTTPException(401, "Invalid token")
    profile_resp = (
        service_client.table("profiles")
        .select("full_name, disabled")
        .eq("id", str(user.id))
        .single()
        .execute()
    )
    if not profile_resp.data:
        raise HTTPException(403, "User profile not found")
    if profile_resp.data["disabled"]:
        raise HTTPException(403, "Account disabled")
    return {"user_id": str(user.id), "full_name": profile_resp.data["full_name"]}
