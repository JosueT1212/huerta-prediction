from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter(prefix="/admin")


class CreateUserRequest(BaseModel):
    email: str
    full_name: str
    password: str


class UpdateUserRequest(BaseModel):
    full_name: str | None = None
    disabled: bool | None = None


@router.get("/users")
def list_users(
    _user: Annotated[dict, Depends(get_current_user)],
):
    resp = (
        service_client.table("profiles")
        .select("id, full_name, disabled, created_at")
        .execute()
    )
    return resp.data


@router.post("/users")
def create_user(
    body: CreateUserRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    create_resp = service_client.auth.admin.create_user({
        "email": body.email,
        "password": body.password,
        "email_confirm": True,
    })
    user_id = str(create_resp.user.id)
    service_client.table("profiles").upsert({
        "id": user_id,
        "full_name": body.full_name,
    }).execute()
    return {"ok": True, "user_id": user_id}


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UpdateUserRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    service_client.table("profiles").update(updates).eq("id", user_id).execute()
    return {"ok": True}
