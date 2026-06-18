# Auth, Sensor Ingest & Railway Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Supabase JWT auth (admin/operator roles), sensor ingest pipeline (Telegraf → FastAPI → Supabase), user management UI in dashboard, and Railway deployment via Dockerfile.

**Architecture:** FastAPI backend on Railway verifies Supabase JWTs on all endpoints. Supabase PostgreSQL stores auth users, profiles, greenhouse access, and sensor readings. Telegraf (or mock script) POSTs readings to `/ingest/sensors` using a static bearer token. Admin role gets a user management panel in the existing HTML dashboard.

**Tech Stack:** Python 3.11, FastAPI, supabase-py 2.x, python-jose, pytest, httpx, Supabase (PostgreSQL + Auth), Telegraf, Docker/Railway

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `backend/supabase_client.py` | Supabase client singletons (service role) |
| Create | `backend/auth.py` | FastAPI deps: JWT verify, role guard, greenhouse guard |
| Create | `backend/routers/__init__.py` | Package marker |
| Create | `backend/routers/ingest.py` | `POST /ingest/sensors` — machine-to-machine token auth |
| Create | `backend/routers/sensors.py` | `GET /sensors/{inv}` — recent readings from Supabase |
| Create | `backend/routers/admin.py` | Admin CRUD: list/create/update/disable users + greenhouse assignments |
| Create | `backend/tests/__init__.py` | Package marker |
| Create | `backend/tests/conftest.py` | Env vars, engine mock, Supabase mock fixtures |
| Create | `backend/tests/test_auth.py` | Unit tests for JWT dep, role guard, greenhouse guard |
| Create | `backend/tests/test_ingest.py` | Unit tests for ingest endpoint |
| Create | `backend/tests/test_admin.py` | Unit tests for admin CRUD routes |
| Create | `supabase/migrations/001_schema.sql` | profiles, greenhouse_access, sensor_readings tables + RLS |
| Modify | `backend/main.py` | Add `/config`, `/me` endpoints; mount routers; protect routes with JWT dep |
| Modify | `backend/requirements.txt` | Add supabase, python-jose, pytest, httpx |
| Modify | `demo/login.html` | Replace fake sessionStorage auth with real Supabase email+password |
| Modify | `demo/Demo Dashboard.html` | Add JWT to fetchJSON; add admin "Usuarios" view; fetch `/me` for role |
| Create | `scripts/mock_ingest.py` | Replay Excel sensor data to `/ingest/sensors` for pipeline testing |
| Create | `telegraf/telegraf.conf` | Telegraf config template (HTTP output → FastAPI) |
| Create | `Dockerfile` | Railway deployment image |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the file contents with:

```
fastapi>=0.110
uvicorn[standard]>=0.27
numpy>=1.24
pandas>=2.0
supabase>=2.3
python-jose[cryptography]>=3.3
pytest>=8.0
httpx>=0.27
pytest-asyncio>=0.23
```

- [ ] **Step 2: Install**

```bash
pip install -r backend/requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "feat(deps): add supabase, python-jose, test dependencies"
```

---

## Task 2: Supabase Client Singleton

**Files:**
- Create: `backend/supabase_client.py`

- [ ] **Step 1: Create `backend/supabase_client.py`**

```python
import os
from supabase import create_client, Client

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY: str = os.environ["SUPABASE_SERVICE_KEY"]

service_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
```

This module is imported by `auth.py`, `routers/ingest.py`, `routers/sensors.py`, and `routers/admin.py`. Single instance — do not call `create_client` elsewhere.

- [ ] **Step 2: Commit**

```bash
git add backend/supabase_client.py
git commit -m "feat(auth): add Supabase service client singleton"
```

---

## Task 3: SQL Migrations

**Files:**
- Create: `supabase/migrations/001_schema.sql`

- [ ] **Step 1: Create migration file**

```bash
mkdir -p supabase/migrations
```

Create `supabase/migrations/001_schema.sql`:

```sql
-- profiles: one row per user, extends auth.users
create table if not exists profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  role        text        not null check (role in ('admin', 'operator')),
  full_name   text,
  disabled    boolean     not null default false,
  created_at  timestamptz not null default now()
);

-- greenhouse_access: which greenhouses an operator can see
create table if not exists greenhouse_access (
  user_id       uuid references profiles(id) on delete cascade,
  greenhouse_id int  not null check (greenhouse_id in (3, 4)),
  primary key (user_id, greenhouse_id)
);

-- sensor_readings: time-series archive written by the ingest endpoint
create table if not exists sensor_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  recorded_at   timestamptz not null,
  sensor_name   text        not null,
  value         float8      not null
);

create index if not exists sensor_readings_gh_time
  on sensor_readings (greenhouse_id, recorded_at desc);

-- RLS: enable on sensor_readings
alter table sensor_readings enable row level security;

create policy "admin_full_access" on sensor_readings
  for select using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );

create policy "operator_own_greenhouses" on sensor_readings
  for select using (
    exists (
      select 1 from greenhouse_access
      where user_id = auth.uid()
        and greenhouse_id = sensor_readings.greenhouse_id
    )
  );
```

- [ ] **Step 2: Apply to Supabase**

Go to your Supabase project → SQL Editor → paste the contents of `supabase/migrations/001_schema.sql` → Run.

Verify in Table Editor: `profiles`, `greenhouse_access`, `sensor_readings` tables exist.

- [ ] **Step 3: Create first admin user manually**

In Supabase dashboard → Authentication → Users → Invite user → enter your email.

After confirming the email and setting a password, run in SQL Editor:
```sql
insert into profiles (id, role, full_name)
values ('<your-user-uuid>', 'admin', 'Tu Nombre');
```

Replace `<your-user-uuid>` with the UUID shown in the Users table.

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/001_schema.sql
git commit -m "feat(db): add profiles, greenhouse_access, sensor_readings schema + RLS"
```

---

## Task 4: JWT Auth Dependency (TDD)

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_auth.py`
- Create: `backend/auth.py`

- [ ] **Step 1: Create package markers**

```bash
touch backend/tests/__init__.py backend/routers/__init__.py
```

- [ ] **Step 2: Create `backend/tests/conftest.py`**

```python
import os
import pytest
from unittest.mock import MagicMock, patch

# Must be set before any backend imports
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("INGEST_TOKEN", "test-ingest-token")

@pytest.fixture(autouse=True)
def mock_engine(monkeypatch):
    mock = MagicMock()
    mock.warm.return_value = None
    mock.loaded.return_value = [3, 4]
    monkeypatch.setattr("backend.main.ENGINE", mock)
    return mock

@pytest.fixture
def mock_supa(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("backend.supabase_client.service_client", mock)
    monkeypatch.setattr("backend.auth.service_client", mock)
    monkeypatch.setattr("backend.routers.ingest.service_client", mock)
    monkeypatch.setattr("backend.routers.sensors.service_client", mock)
    monkeypatch.setattr("backend.routers.admin.service_client", mock)
    return mock

@pytest.fixture
def api_client(mock_supa):
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)

def make_user(user_id="uid-admin", role="admin", greenhouse_ids=None):
    """Build mock Supabase get_user + profile responses."""
    greenhouse_ids = greenhouse_ids or [3, 4]
    mock_user = MagicMock()
    mock_user.id = user_id
    profile_data = {
        "role": role,
        "disabled": False,
        "greenhouse_access": [{"greenhouse_id": g} for g in greenhouse_ids],
    }
    return mock_user, profile_data
```

- [ ] **Step 3: Write failing tests in `backend/tests/test_auth.py`**

```python
import pytest


def test_missing_auth_header_returns_401(api_client):
    r = api_client.get("/me")
    assert r.status_code == 401


def test_malformed_auth_header_returns_401(api_client):
    r = api_client.get("/me", headers={"Authorization": "NotBearer abc"})
    assert r.status_code == 401


def test_invalid_token_returns_401(api_client, mock_supa):
    from supabase.auth_client import AuthApiError
    mock_supa.auth.get_user.side_effect = AuthApiError("invalid", 401, {})
    r = api_client.get("/me", headers={"Authorization": "Bearer bad-token"})
    assert r.status_code == 401


def test_disabled_user_returns_403(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="operator")
    profile["disabled"] = True
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 403


def test_valid_admin_token_returns_user_info(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="admin")
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin"
    assert body["user_id"] == "uid-admin"


def test_operator_blocked_from_wrong_greenhouse(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="operator", greenhouse_ids=[3])
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/inference/4", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 403


def test_admin_can_access_any_greenhouse(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="admin")
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    # /inference/{inv} now returns ENGINE data — mock it
    r = api_client.get("/inference/4", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code != 403
```

- [ ] **Step 4: Run tests — expect failures**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction
python -m pytest backend/tests/test_auth.py -v 2>&1 | head -40
```

Expected: `ImportError` or `FAILED` — `/me` route and `auth.py` don't exist yet.

- [ ] **Step 5: Create `backend/auth.py`**

```python
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
        .select("role, disabled, greenhouse_access(greenhouse_id)")
        .eq("id", str(user.id))
        .single()
        .execute()
    )
    if not profile_resp.data:
        raise HTTPException(403, "User profile not found")
    profile = profile_resp.data
    if profile["disabled"]:
        raise HTTPException(403, "Account disabled")
    return {
        "user_id": str(user.id),
        "role": profile["role"],
        "greenhouse_ids": [
            g["greenhouse_id"] for g in profile.get("greenhouse_access", [])
        ],
    }


def require_admin(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    return current_user


def greenhouse_access_dep(
    inv: int,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    if current_user["role"] == "admin":
        return current_user
    if inv not in current_user["greenhouse_ids"]:
        raise HTTPException(403, f"No access to invernadero {inv}")
    return current_user


```

- [ ] **Step 6: Add `/me` and `/config` to `backend/main.py`**

In `backend/main.py`, add these imports at the top (after existing imports):
```python
from backend.auth import get_current_user, greenhouse_access_dep
from typing import Annotated
```

Add these two routes (after the `/health` route):
```python
@app.get('/config')
def config():
    """Public: return Supabase URL and anon key for the frontend."""
    return {
        'supabase_url': os.environ['SUPABASE_URL'],
        'supabase_anon_key': os.environ['SUPABASE_ANON_KEY'],
    }


@app.get('/me')
def me(current_user: Annotated[dict, Depends(get_current_user)]):
    return current_user
```

Add `import os` if not already present at the top of `backend/main.py`. Also add `SUPABASE_ANON_KEY` to the env vars needed.

- [ ] **Step 7: Run tests — expect pass**

```bash
python -m pytest backend/tests/test_auth.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/auth.py backend/main.py backend/tests/__init__.py backend/tests/conftest.py backend/tests/test_auth.py backend/routers/__init__.py
git commit -m "feat(auth): JWT middleware, get_current_user dep, /me + /config endpoints"
```

---

## Task 5: Protect Existing Inference Routes

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing test in `backend/tests/test_auth.py`**

Add to `test_auth.py`:

```python
def test_inference_requires_auth(api_client):
    r = api_client.get("/inference/3")
    assert r.status_code == 401

def test_metrics_requires_auth(api_client):
    r = api_client.get("/metrics/3")
    assert r.status_code == 401
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest backend/tests/test_auth.py::test_inference_requires_auth backend/tests/test_auth.py::test_metrics_requires_auth -v
```

Expected: FAIL — routes return 200 without auth.

- [ ] **Step 3: Protect routes in `backend/main.py`**

Find the `inference` route definition and add the dependency:
```python
@app.get('/inference/{inv}')
def inference(
    inv: int,
    _user: Annotated[dict, Depends(greenhouse_access_dep)],
):
    _check_inv(inv)
    return ENGINE.payload(inv)
```

Find `inference_window` and update:
```python
@app.get('/inference/{inv}/window')
def inference_window(
    inv: int,
    cursor: int | None = Query(None),
    horizon: int = Query(6, ge=1, le=12),
    past: int = Query(5, ge=0, le=20),
    _user: Annotated[dict, Depends(greenhouse_access_dep)] = None,
):
    _check_inv(inv)
    return ENGINE.window(inv, cursor=cursor, horizon=horizon, past=past)
```

Find `metrics` and update:
```python
@app.get('/metrics/{inv}')
def metrics(
    inv: int,
    _user: Annotated[dict, Depends(greenhouse_access_dep)],
):
    _check_inv(inv)
    return ENGINE.metrics(inv)
```

Find `predictions` (compat alias) and update the same way:
```python
@app.get('/predictions/{inv}')
def predictions(
    inv: int,
    _user: Annotated[dict, Depends(greenhouse_access_dep)],
):
    _check_inv(inv)
    return ENGINE.payload(inv)
```

- [ ] **Step 4: Run all auth tests**

```bash
python -m pytest backend/tests/test_auth.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_auth.py
git commit -m "feat(auth): protect inference/metrics routes with JWT + greenhouse access"
```

---

## Task 6: Sensor Ingest Endpoint (TDD)

**Files:**
- Create: `backend/routers/ingest.py`
- Create: `backend/tests/test_ingest.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_ingest.py`**

```python
import pytest


def test_ingest_rejects_no_token(api_client):
    r = api_client.post("/ingest/sensors", json={
        "greenhouse_id": 3,
        "sensor_name": "temp_interior",
        "value": 24.5,
        "recorded_at": "2026-06-18T10:00:00Z",
    })
    assert r.status_code == 401


def test_ingest_rejects_wrong_token(api_client):
    r = api_client.post("/ingest/sensors",
        json={
            "greenhouse_id": 3,
            "sensor_name": "temp_interior",
            "value": 24.5,
            "recorded_at": "2026-06-18T10:00:00Z",
        },
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_ingest_accepts_valid_token(api_client, mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    r = api_client.post("/ingest/sensors",
        json={
            "greenhouse_id": 3,
            "sensor_name": "temp_interior",
            "value": 24.5,
            "recorded_at": "2026-06-18T10:00:00Z",
        },
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_ingest_writes_to_supabase(api_client, mock_supa):
    from unittest.mock import call
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    api_client.post("/ingest/sensors",
        json={
            "greenhouse_id": 4,
            "sensor_name": "hr_interior",
            "value": 65.0,
            "recorded_at": "2026-06-18T11:00:00Z",
        },
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    mock_supa.table.assert_called_with("sensor_readings")
    inserted = mock_supa.table.return_value.insert.call_args[0][0]
    assert inserted["greenhouse_id"] == 4
    assert inserted["sensor_name"] == "hr_interior"
    assert inserted["value"] == 65.0


# add this import at top of file
from unittest.mock import MagicMock
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest backend/tests/test_ingest.py -v 2>&1 | head -20
```

Expected: `ImportError` — router doesn't exist yet.

- [ ] **Step 3: Create `backend/routers/ingest.py`**

```python
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
    return {"ok": True}
```

- [ ] **Step 4: Mount router in `backend/main.py`**

Add near the top of `backend/main.py` (after existing imports):
```python
from backend.routers import ingest as ingest_router
```

Add after `app = FastAPI(...)` and middleware setup:
```python
app.include_router(ingest_router.router)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest backend/tests/test_ingest.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/routers/ingest.py backend/tests/test_ingest.py backend/main.py
git commit -m "feat(ingest): POST /ingest/sensors with machine-to-machine token auth"
```

---

## Task 7: Sensors Read Endpoint (TDD)

**Files:**
- Create: `backend/routers/sensors.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing tests — add to `backend/tests/test_ingest.py`**

```python
def test_sensors_requires_jwt(api_client):
    r = api_client.get("/sensors/3")
    assert r.status_code == 401


def test_sensors_returns_readings(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="admin")
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    # Second table call: sensor_readings query
    readings_mock = MagicMock()
    readings_mock.execute.return_value.data = [
        {"sensor_name": "temp_interior", "value": 24.5, "recorded_at": "2026-06-18T10:00:00Z"}
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value = readings_mock
    r = api_client.get("/sensors/3", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest backend/tests/test_ingest.py::test_sensors_requires_jwt -v
```

Expected: FAIL — route doesn't exist.

- [ ] **Step 3: Create `backend/routers/sensors.py`**

```python
from fastapi import APIRouter, Depends, Query
from typing import Annotated
from backend.auth import greenhouse_access_dep
from backend.supabase_client import service_client

router = APIRouter()


@router.get("/sensors/{inv}")
def get_sensors(
    inv: int,
    limit: int = Query(100, ge=1, le=1000),
    _user: Annotated[dict, Depends(greenhouse_access_dep)] = None,
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
```

- [ ] **Step 4: Mount in `backend/main.py`**

```python
from backend.routers import sensors as sensors_router
# ...
app.include_router(sensors_router.router)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest backend/tests/test_ingest.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/routers/sensors.py backend/tests/test_ingest.py backend/main.py
git commit -m "feat(sensors): GET /sensors/{inv} — reads recent sensor_readings from Supabase"
```

---

## Task 8: Admin User Management API (TDD)

**Files:**
- Create: `backend/routers/admin.py`
- Create: `backend/tests/test_admin.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_admin.py`**

```python
import pytest
from unittest.mock import MagicMock


def _admin_headers(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="admin")
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def _operator_headers(api_client, mock_supa):
    from tests.conftest import make_user
    user, profile = make_user(role="operator", greenhouse_ids=[3])
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_list_users_requires_admin(api_client, mock_supa):
    headers = _operator_headers(api_client, mock_supa)
    r = api_client.get("/admin/users", headers=headers)
    assert r.status_code == 403


def test_list_users_returns_profiles(api_client, mock_supa):
    headers = _admin_headers(api_client, mock_supa)
    mock_supa.table.return_value.select.return_value.execute.return_value.data = [
        {"id": "uid-1", "role": "operator", "full_name": "Juan", "disabled": False,
         "created_at": "2026-01-01T00:00:00Z", "greenhouse_access": [{"greenhouse_id": 3}]}
    ]
    r = api_client.get("/admin/users", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_user_requires_admin(api_client, mock_supa):
    headers = _operator_headers(api_client, mock_supa)
    r = api_client.post("/admin/users", headers=headers, json={
        "email": "new@test.com", "full_name": "New User",
        "role": "operator", "greenhouse_ids": [3]
    })
    assert r.status_code == 403


def test_create_user_invites_and_inserts_profile(api_client, mock_supa):
    headers = _admin_headers(api_client, mock_supa)
    new_user = MagicMock()
    new_user.id = "new-uid"
    mock_supa.auth.admin.invite_user_by_email.return_value.user = new_user
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()

    r = api_client.post("/admin/users", headers=headers, json={
        "email": "juan@huerta.com",
        "full_name": "Juan García",
        "role": "operator",
        "greenhouse_ids": [3],
    })
    assert r.status_code == 200
    assert r.json()["user_id"] == "new-uid"
    mock_supa.auth.admin.invite_user_by_email.assert_called_once_with("juan@huerta.com")


def test_disable_user_requires_admin(api_client, mock_supa):
    headers = _operator_headers(api_client, mock_supa)
    r = api_client.patch("/admin/users/some-uid", headers=headers, json={"disabled": True})
    assert r.status_code == 403


def test_disable_user_updates_profile(api_client, mock_supa):
    headers = _admin_headers(api_client, mock_supa)
    mock_supa.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

    r = api_client.patch("/admin/users/target-uid", headers=headers, json={"disabled": True})
    assert r.status_code == 200
    mock_supa.table.assert_called_with("profiles")
    mock_supa.table.return_value.update.assert_called_with({"disabled": True})


def test_grant_greenhouse_access(api_client, mock_supa):
    headers = _admin_headers(api_client, mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()

    r = api_client.post("/admin/users/target-uid/greenhouses/4", headers=headers)
    assert r.status_code == 200


def test_revoke_greenhouse_access(api_client, mock_supa):
    headers = _admin_headers(api_client, mock_supa)
    mock_supa.table.return_value.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock()

    r = api_client.delete("/admin/users/target-uid/greenhouses/3", headers=headers)
    assert r.status_code == 200
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest backend/tests/test_admin.py -v 2>&1 | head -20
```

Expected: `ImportError` — admin router doesn't exist.

- [ ] **Step 3: Create `backend/routers/admin.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated
from backend.auth import require_admin
from backend.supabase_client import service_client

router = APIRouter(prefix="/admin")


class CreateUserRequest(BaseModel):
    email: str
    full_name: str
    role: str
    greenhouse_ids: list[int]


class UpdateUserRequest(BaseModel):
    role: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


@router.get("/users")
def list_users(
    _user: Annotated[dict, Depends(require_admin)],
):
    resp = (
        service_client.table("profiles")
        .select("id, role, full_name, disabled, created_at, greenhouse_access(greenhouse_id)")
        .execute()
    )
    return resp.data


@router.post("/users")
def create_user(
    body: CreateUserRequest,
    _user: Annotated[dict, Depends(require_admin)],
):
    invite_resp = service_client.auth.admin.invite_user_by_email(body.email)
    user_id = str(invite_resp.user.id)

    service_client.table("profiles").insert({
        "id": user_id,
        "role": body.role,
        "full_name": body.full_name,
    }).execute()

    for gh_id in body.greenhouse_ids:
        service_client.table("greenhouse_access").upsert({
            "user_id": user_id,
            "greenhouse_id": gh_id,
        }).execute()

    return {"ok": True, "user_id": user_id}


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UpdateUserRequest,
    _user: Annotated[dict, Depends(require_admin)],
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    service_client.table("profiles").update(updates).eq("id", user_id).execute()
    return {"ok": True}


@router.post("/users/{user_id}/greenhouses/{inv}")
def grant_greenhouse(
    user_id: str,
    inv: int,
    _user: Annotated[dict, Depends(require_admin)],
):
    service_client.table("greenhouse_access").upsert({
        "user_id": user_id,
        "greenhouse_id": inv,
    }).execute()
    return {"ok": True}


@router.delete("/users/{user_id}/greenhouses/{inv}")
def revoke_greenhouse(
    user_id: str,
    inv: int,
    _user: Annotated[dict, Depends(require_admin)],
):
    service_client.table("greenhouse_access").delete().eq(
        "user_id", user_id
    ).eq("greenhouse_id", inv).execute()
    return {"ok": True}
```

- [ ] **Step 4: Mount in `backend/main.py`**

```python
from backend.routers import admin as admin_router
# ...
app.include_router(admin_router.router)
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest backend/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/routers/admin.py backend/tests/test_admin.py backend/main.py
git commit -m "feat(admin): user management CRUD — list/create/update/disable + greenhouse access"
```

---

## Task 9: Update login.html with Supabase Auth

**Files:**
- Modify: `demo/login.html`

The current login does fake auth (`sessionStorage.setItem('jata_auth', '1')`). Replace with real Supabase email+password sign-in.

- [ ] **Step 1: Replace the `<script>` block at the bottom of `demo/login.html`**

The current script block starts at line ~194. Replace it entirely with:

```html
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js"></script>
<script>
  const form = document.getElementById('login-form');
  const hint = document.getElementById('hint');
  const btn  = form.querySelector('.btn');
  let sb;

  async function initSupabase() {
    const cfg = await fetch('/config').then(r => r.json());
    const { createClient } = supabase;
    sb = createClient(cfg.supabase_url, cfg.supabase_anon_key);
  }

  initSupabase().catch(() => {
    hint.textContent = 'Error de configuración. Contacta al administrador.';
    hint.classList.add('show');
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('username').value.trim();
    const pass  = document.getElementById('password').value;

    if (!email || !pass) {
      hint.textContent = 'Ingresa tu correo y contraseña.';
      hint.classList.add('show');
      return;
    }
    hint.classList.remove('show');
    btn.disabled = true;
    btn.textContent = 'Entrando…';

    const { data, error } = await sb.auth.signInWithPassword({
      email,
      password: pass,
    });

    if (error) {
      hint.textContent = 'Credenciales inválidas.';
      hint.classList.add('show');
      btn.disabled = false;
      btn.textContent = 'Entrar';
      return;
    }

    localStorage.setItem('sb_token', data.session.access_token);
    window.location.href = '/app';
  });
</script>
```

Note: the `id="username"` input already exists in the form (it holds the email). No HTML structure changes needed.

- [ ] **Step 2: Manual test**

```bash
uvicorn backend.main:app --port 8000 --reload
```

Set these env vars first (export in terminal):
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export SUPABASE_SERVICE_KEY="your-service-role-key"
export INGEST_TOKEN="some-secret-token"
```

Open `http://localhost:8000/`. Enter the admin email + password created in Task 3. Should redirect to `/app` and load the dashboard.

Enter wrong credentials — should show "Credenciales inválidas."

- [ ] **Step 3: Commit**

```bash
git add demo/login.html
git commit -m "feat(login): replace fake auth with Supabase email+password sign-in"
```

---

## Task 10: Update Dashboard HTML — JWT Headers + Admin Panel

**Files:**
- Modify: `demo/Demo Dashboard.html`

Two changes: (1) add JWT to all API calls, (2) add "Usuarios" admin view with user management UI.

- [ ] **Step 1: Replace the `sessionStorage` guard and `fetchJSON` function**

In `demo/Demo Dashboard.html`, find and replace:

Find (lines ~1705–1707):
```javascript
if (!sessionStorage.getItem('jata_auth')) {
  window.location.replace('/');
}
```

Replace with:
```javascript
const _token = localStorage.getItem('sb_token');
if (!_token) { window.location.replace('/'); }
```

Find (lines ~1751–1753):
```javascript
async function fetchJSON(path) {
  const r = await fetch(`${API_BASE}${path}`);
```

Replace with:
```javascript
async function fetchJSON(path) {
  const token = localStorage.getItem('sb_token');
  if (!token) { window.location.replace('/'); return null; }
  const r = await fetch(`${API_BASE}${path}`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (r.status === 401) {
    localStorage.removeItem('sb_token');
    window.location.replace('/');
    return null;
  }
```

- [ ] **Step 2: Fetch user role on load and store globally**

Find the `init()` async function call near the bottom of the script. Before it calls `fetchJSON`, add a role fetch. Locate where `init()` is defined and add at its very start:

```javascript
async function init() {
  // Fetch current user role
  const me = await fetchJSON('/me');
  if (!me) return;
  window._userRole = me.role;
  // Show admin nav item if admin
  if (me.role === 'admin') {
    document.querySelectorAll('[data-admin-only]').forEach(el => el.style.display = '');
  }

  // ... rest of existing init() code unchanged ...
```

- [ ] **Step 3: Add admin nav link in sidebar**

Find the sidebar `<aside>` element. Locate the last `<a class="sb-link" ...>` nav item. After it, add:

```html
<a class="sb-link" data-goto="view-usuarios" data-admin-only style="display:none">
  <span class="sb-icon">👥</span>Usuarios
</a>
```

- [ ] **Step 4: Add admin view section**

Before the closing `</div>` of `<div class="app">`, add this new view:

```html
<!-- ADMIN: User Management -->
<section class="view" id="view-usuarios">
  <div style="padding:32px;max-width:900px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
      <h2 style="font-size:1.4rem;font-weight:600">Gestión de Usuarios</h2>
      <button onclick="openNewUserModal()" style="background:#65a30d;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:0.9rem">+ Nuevo usuario</button>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem" id="users-table">
      <thead>
        <tr style="border-bottom:2px solid #e2e5e9">
          <th style="text-align:left;padding:10px 8px">Nombre</th>
          <th style="text-align:left;padding:10px 8px">Correo</th>
          <th style="text-align:left;padding:10px 8px">Rol</th>
          <th style="text-align:left;padding:10px 8px">Invernaderos</th>
          <th style="text-align:left;padding:10px 8px">Estado</th>
          <th style="padding:10px 8px"></th>
        </tr>
      </thead>
      <tbody id="users-tbody"></tbody>
    </table>
  </div>

  <!-- New User Modal -->
  <div id="new-user-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:1000;display:none;align-items:center;justify-content:center">
    <div style="background:#fff;border-radius:10px;padding:32px;width:420px;max-width:95vw">
      <h3 style="margin-bottom:20px;font-size:1.1rem">Nuevo usuario</h3>
      <label style="display:block;margin-bottom:12px;font-size:0.9rem">
        Correo electrónico<br>
        <input id="nu-email" type="email" style="width:100%;margin-top:4px;padding:8px;border:1px solid #d2d6db;border-radius:6px">
      </label>
      <label style="display:block;margin-bottom:12px;font-size:0.9rem">
        Nombre completo<br>
        <input id="nu-name" type="text" style="width:100%;margin-top:4px;padding:8px;border:1px solid #d2d6db;border-radius:6px">
      </label>
      <label style="display:block;margin-bottom:12px;font-size:0.9rem">
        Rol<br>
        <select id="nu-role" style="width:100%;margin-top:4px;padding:8px;border:1px solid #d2d6db;border-radius:6px">
          <option value="operator">Operador</option>
          <option value="admin">Admin</option>
        </select>
      </label>
      <label style="display:block;margin-bottom:20px;font-size:0.9rem">
        Invernaderos<br>
        <label style="margin-right:16px"><input type="checkbox" id="nu-inv3" value="3"> Inv 3</label>
        <label><input type="checkbox" id="nu-inv4" value="4"> Inv 4</label>
      </label>
      <div style="display:flex;gap:10px;justify-content:flex-end">
        <button onclick="closeNewUserModal()" style="padding:8px 18px;border:1px solid #d2d6db;background:#fff;border-radius:6px;cursor:pointer">Cancelar</button>
        <button onclick="submitNewUser()" style="padding:8px 18px;background:#65a30d;color:#fff;border:none;border-radius:6px;cursor:pointer">Invitar</button>
      </div>
      <p id="nu-error" style="color:red;margin-top:10px;font-size:0.85rem;display:none"></p>
    </div>
  </div>
</section>
```

- [ ] **Step 5: Add admin JS functions to the script block**

Before the closing `</script>` tag, add:

```javascript
// ── Admin: user management ─────────────────────────────────────────────────
async function loadUsers() {
  const users = await fetchJSON('/admin/users');
  if (!users) return;
  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = users.map(u => `
    <tr style="border-bottom:1px solid #e2e5e9">
      <td style="padding:10px 8px">${u.full_name || '—'}</td>
      <td style="padding:10px 8px;color:#565b63">${u.id.slice(0, 8)}…</td>
      <td style="padding:10px 8px">${u.role}</td>
      <td style="padding:10px 8px">${(u.greenhouse_access||[]).map(g => `Inv ${g.greenhouse_id}`).join(', ') || '—'}</td>
      <td style="padding:10px 8px">
        <span style="color:${u.disabled ? '#dc2626' : '#16a34a'}">${u.disabled ? 'Desactivado' : 'Activo'}</span>
      </td>
      <td style="padding:10px 8px">
        <button onclick="toggleUser('${u.id}', ${u.disabled})" style="font-size:0.8rem;padding:4px 10px;border:1px solid #d2d6db;background:#fff;border-radius:4px;cursor:pointer">
          ${u.disabled ? 'Activar' : 'Desactivar'}
        </button>
      </td>
    </tr>
  `).join('');
}

async function toggleUser(userId, currentlyDisabled) {
  const token = localStorage.getItem('sb_token');
  await fetch(`${API_BASE}/admin/users/${userId}`, {
    method: 'PATCH',
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ disabled: !currentlyDisabled }),
  });
  loadUsers();
}

function openNewUserModal() {
  document.getElementById('new-user-modal').style.display = 'flex';
}

function closeNewUserModal() {
  document.getElementById('new-user-modal').style.display = 'none';
  document.getElementById('nu-error').style.display = 'none';
}

async function submitNewUser() {
  const email   = document.getElementById('nu-email').value.trim();
  const name    = document.getElementById('nu-name').value.trim();
  const role    = document.getElementById('nu-role').value;
  const ghIds   = [3, 4].filter(n => document.getElementById(`nu-inv${n}`).checked);

  if (!email || !name) {
    const err = document.getElementById('nu-error');
    err.textContent = 'Correo y nombre son requeridos.';
    err.style.display = 'block';
    return;
  }

  const token = localStorage.getItem('sb_token');
  const r = await fetch(`${API_BASE}/admin/users`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, full_name: name, role, greenhouse_ids: ghIds }),
  });

  if (!r.ok) {
    const err = document.getElementById('nu-error');
    err.textContent = 'Error al crear usuario. Verifica el correo.';
    err.style.display = 'block';
    return;
  }

  closeNewUserModal();
  loadUsers();
}

// Load users when navigating to the admin view
document.querySelector('[data-goto="view-usuarios"]')?.addEventListener('click', loadUsers);
```

- [ ] **Step 6: Manual test**

```bash
uvicorn backend.main:app --port 8000 --reload
```

1. Login as admin at `http://localhost:8000/`
2. Sidebar should show "Usuarios" link
3. Click "Usuarios" → table loads with existing users
4. Click "+ Nuevo usuario" → modal opens → fill form → "Invitar" → user receives invite email
5. Login as operator — "Usuarios" link should NOT appear

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add JWT auth headers + admin Usuarios panel"
```

---

## Task 11: Mock Ingest Script

**Files:**
- Create: `scripts/mock_ingest.py`

- [ ] **Step 1: Create `scripts/mock_ingest.py`**

```python
"""
Replay Excel sensor data to /ingest/sensors for pipeline testing.

Usage:
    INGEST_TOKEN=your-token python scripts/mock_ingest.py --inv 3 --host http://localhost:8000
"""
import argparse
import os
import sys
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"

SENSOR_FILES = {
    3: DATA / "Variables internas Invernadero 3.xlsx",
    4: DATA / "Variables internas invernadero 4.xlsx",
}

# Map Excel columns to sensor_name values written to the DB
COLUMN_MAP = {
    "Temperatura Promedio": "temp_prom_int",
    "Temperatura Mínima":   "temp_min_int",
    "Temperatura Máxima":   "temp_max_int",
    "Humedad Relativa Promedio": "hr_prom_int",
    "CO2": "co2_ppm",
}


def load_sensor_data(inv: int) -> pd.DataFrame:
    path = SENSOR_FILES[inv]
    frames = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Fecha" not in df.columns:
            continue
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.dropna(subset=["Fecha"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def post_reading(host: str, token: str, inv: int, sensor_name: str, value: float, recorded_at: str):
    r = requests.post(
        f"{host}/ingest/sensors",
        json={
            "greenhouse_id": inv,
            "sensor_name": sensor_name,
            "value": float(value),
            "recorded_at": recorded_at,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    r.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inv", type=int, choices=[3, 4], required=True)
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("INGEST_TOKEN", "")
    if not token and not args.dry_run:
        print("ERROR: set INGEST_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    df = load_sensor_data(args.inv)
    if df.empty:
        print(f"No data found for inv {args.inv}")
        sys.exit(1)

    sent = 0
    for _, row in df.iterrows():
        ts = row["Fecha"].isoformat()
        for col, sensor_name in COLUMN_MAP.items():
            if col not in row or pd.isna(row[col]):
                continue
            if not args.dry_run:
                post_reading(args.host, token, args.inv, sensor_name, row[col], ts)
            sent += 1

    print(f"{'[DRY RUN] Would send' if args.dry_run else 'Sent'} {sent} readings for inv {args.inv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry-run**

```bash
python scripts/mock_ingest.py --inv 3 --dry-run
```

Expected output: `[DRY RUN] Would send N readings for inv 3` (N > 0).

- [ ] **Step 3: Test live ingest (requires running backend)**

```bash
uvicorn backend.main:app --port 8000 &
INGEST_TOKEN=some-secret-token python scripts/mock_ingest.py --inv 3 --host http://localhost:8000
```

Expected: `Sent N readings for inv 3` with no errors. Verify in Supabase Table Editor: `sensor_readings` table has rows.

- [ ] **Step 4: Commit**

```bash
git add scripts/mock_ingest.py
git commit -m "feat(scripts): Excel sensor replay script for ingest pipeline testing"
```

---

## Task 12: Telegraf Configuration

**Files:**
- Create: `telegraf/telegraf.conf`

- [ ] **Step 1: Create `telegraf/telegraf.conf`**

```toml
# Telegraf configuration for Huerta Prediction sensor pipeline.
# Replace [[inputs.*]] section with the plugin matching your hardware.
# Set TELEGRAF_INGEST_TOKEN env var before running.

[agent]
  interval          = "1m"
  round_interval    = true
  metric_batch_size = 100
  metric_buffer_limit = 10000
  collection_jitter = "0s"
  flush_interval    = "10s"
  flush_jitter      = "0s"
  precision         = ""
  quiet             = false

# ── INPUTS ────────────────────────────────────────────────────────────────
# Option A: MQTT broker (e.g. Campbell datalogger with MQTT output)
[[inputs.mqtt_consumer]]
  servers         = ["tcp://localhost:1883"]
  topics          = ["greenhouse/+/sensors"]
  data_format     = "json"
  # The topic segment between / maps to greenhouse_id tag
  topic_tag       = "greenhouse_id"

# Option B: HTTP listener (datalogger POSTs directly to Telegraf)
# [[inputs.http_listener_v2]]
#   service_address = ":8080"
#   data_format     = "json"

# Option C: File reader (manual CSV drops from datalogger software)
# [[inputs.file]]
#   files       = ["/data/sensors/*.csv"]
#   data_format = "csv"
#   csv_header_row_count = 1

# ── OUTPUTS ───────────────────────────────────────────────────────────────
[[outputs.http]]
  url         = "https://your-app.railway.app/ingest/sensors"
  method      = "POST"
  data_format = "json"
  [outputs.http.headers]
    Content-Type  = "application/json"
    Authorization = "Bearer ${TELEGRAF_INGEST_TOKEN}"

# Local file output for debugging (disable in production)
[[outputs.file]]
  files = ["stdout"]
  data_format = "json"
```

- [ ] **Step 2: Create `telegraf/README.md`**

```markdown
# Telegraf Setup

## Install

```bash
# macOS
brew install telegraf

# Ubuntu/Debian
sudo apt-get install telegraf
```

## Run

```bash
export TELEGRAF_INGEST_TOKEN="your-ingest-token"
telegraf --config telegraf/telegraf.conf
```

## Swap input plugin

Edit `telegraf.conf` — comment out `[[inputs.mqtt_consumer]]` and uncomment
the option that matches your hardware (HTTP listener, file, serial, etc.).
```

- [ ] **Step 3: Commit**

```bash
git add telegraf/telegraf.conf telegraf/README.md
git commit -m "feat(telegraf): add Telegraf config template for sensor pipeline"
```

---

## Task 13: Dockerfile + Railway Deployment

**Files:**
- Create: `Dockerfile`
- Create: `.env.example`

- [ ] **Step 1: Create `Dockerfile` at repo root**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy full repo (model weights, demo HTML, backend code)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `.env.example`**

```bash
# Copy to .env for local development (never commit .env)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-role-key-here
INGEST_TOKEN=choose-a-strong-random-secret
```

- [ ] **Step 3: Verify .env is in .gitignore**

```bash
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

- [ ] **Step 4: Test Docker build locally**

```bash
docker build -t huerta-prediction .
docker run -p 8000:8000 \
  -e SUPABASE_URL="https://your-project.supabase.co" \
  -e SUPABASE_ANON_KEY="your-anon-key" \
  -e SUPABASE_SERVICE_KEY="your-service-role-key" \
  -e INGEST_TOKEN="test-token" \
  huerta-prediction
```

Expected: container starts, logs show `[main] Modelos listos: [3, 4]`, `http://localhost:8000/health` returns `{"status":"ok",...}`.

- [ ] **Step 5: Deploy to Railway**

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
2. Select `JosueT1212/Huerta_Prediction`
3. Railway auto-detects `Dockerfile`
4. In Variables tab, add all 4 env vars from `.env.example` with real values
5. Deploy → wait for build (2–3 minutes)
6. Open the generated Railway URL → `https://your-app.railway.app/health` → verify `{"status":"ok"}`
7. Update `telegraf/telegraf.conf`: replace `your-app.railway.app` with the real Railway URL

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .env.example .gitignore telegraf/telegraf.conf
git commit -m "feat(deploy): Dockerfile + Railway deployment config"
```

---

## Run Full Test Suite

After all tasks complete, run:

```bash
python -m pytest backend/tests/ -v
```

Expected: all tests PASS with no failures.
