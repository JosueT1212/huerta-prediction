# Auth, Sensor Ingest & Railway Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Supabase JWT auth (no roles — all users see everything), sensor ingest pipeline (Telegraf → FastAPI → Supabase), user management UI, live sensor polling in dashboard, and Railway deployment via Dockerfile.

**Architecture:** FastAPI on Railway serves both the frontend HTML and the API. Supabase PostgreSQL stores profiles and sensor readings. All authenticated users have equal access to both greenhouses. Telegraf (or mock script) POSTs sensor readings to `/ingest/sensors` with a static token.

**Tech Stack:** Python 3.11, FastAPI, supabase-py 2.x, pytest, httpx, Supabase (PostgreSQL + Auth), Telegraf, Docker/Railway

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `backend/supabase_client.py` | Supabase client singleton (service role) |
| Create | `backend/auth.py` | `get_current_user` dep — JWT verify, disabled check |
| Create | `backend/routers/__init__.py` | Package marker |
| Create | `backend/routers/ingest.py` | `POST /ingest/sensors` — static token auth |
| Create | `backend/routers/sensors.py` | `GET /sensors/{inv}` — recent readings |
| Create | `backend/routers/admin.py` | `GET/POST/PATCH /admin/users` — any authenticated user |
| Create | `backend/tests/__init__.py` | Package marker |
| Create | `backend/tests/conftest.py` | Env vars, engine mock, Supabase mock fixtures |
| Create | `backend/tests/test_auth.py` | JWT verify, disabled user, protected routes |
| Create | `backend/tests/test_ingest.py` | Ingest token auth, Supabase write, sensors read |
| Create | `backend/tests/test_admin.py` | User list/create/update |
| Create | `supabase/migrations/001_schema.sql` | profiles + sensor_readings tables |
| Modify | `backend/main.py` | Mount routers, add `/config` + `/me`, protect routes |
| Modify | `backend/requirements.txt` | Add supabase, pytest, httpx |
| Modify | `demo/login.html` | Real Supabase email+password auth |
| Modify | `demo/Demo Dashboard.html` | JWT on all fetches, live sensor polling, Usuarios panel |
| Create | `scripts/mock_ingest.py` | Excel sensor replay for pipeline testing |
| Create | `telegraf/telegraf.conf` | Telegraf config template |
| Create | `Dockerfile` | Railway deployment |
| Create | `.env.example` | Document required env vars |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update `backend/requirements.txt`**

```
fastapi>=0.110
uvicorn[standard]>=0.27
numpy>=1.24
pandas>=2.0
supabase>=2.3
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 2: Install**

```bash
pip install -r backend/requirements.txt
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "feat(deps): add supabase and test dependencies"
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

- [ ] **Step 2: Commit**

```bash
git add backend/supabase_client.py
git commit -m "feat(auth): Supabase service client singleton"
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
-- profiles: one row per user, no roles
create table if not exists profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  full_name   text,
  disabled    boolean     not null default false,
  created_at  timestamptz not null default now()
);

-- sensor_readings: time-series archive
create table if not exists sensor_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  recorded_at   timestamptz not null,
  sensor_name   text        not null,
  value         float8      not null
);

create index if not exists sensor_readings_gh_time
  on sensor_readings (greenhouse_id, recorded_at desc);
```

- [ ] **Step 2: Apply to Supabase**

Supabase dashboard → SQL Editor → paste file contents → Run.

Verify in Table Editor: `profiles` and `sensor_readings` tables exist.

- [ ] **Step 3: Create first user manually**

Supabase dashboard → Authentication → Users → Invite user → enter your email.

After confirming invite and setting password, run in SQL Editor:
```sql
insert into profiles (id, full_name)
values ('<your-user-uuid>', 'Tu Nombre');
```

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/001_schema.sql
git commit -m "feat(db): profiles + sensor_readings schema"
```

---

## Task 4: JWT Auth Dependency (TDD)

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_auth.py`
- Create: `backend/auth.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Create package markers**

```bash
touch backend/tests/__init__.py backend/routers/__init__.py
```

- [ ] **Step 2: Create `backend/tests/conftest.py`**

```python
import os
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("INGEST_TOKEN", "test-ingest-token")


@pytest.fixture(autouse=True)
def mock_engine(monkeypatch):
    # engine.py now loads from .npz (not .pt) — warm() reads files; mock it out
    mock = MagicMock()
    mock.warm.return_value = None
    mock.loaded.return_value = [3, 4]
    mock.payload.return_value = {"inv_id": 3, "weeks": [], "y_true": [], "y_pred": []}
    mock.metrics.return_value = {"inv_id": 3, "source": "npz_cache", "best": {}, "cptc_pi": {}}
    mock.window.return_value = {"inv_id": 3, "cursor": 0, "history": [], "predictions": []}
    monkeypatch.setattr("backend.main.ENGINE", mock)
    return mock


@pytest.fixture(autouse=True)
def mock_data_api(monkeypatch):
    # data_api reads Excel files from Data/ — mock all three functions
    mock_pheno = {"inv_id": 3, "fields": [], "seasons": [], "latest": {}, "latest_season": "T17"}
    mock_hist = {"inv_id": 3, "var": "temp", "meta": {}, "months": [], "series": []}
    mock_avg = {"n_plants": 1, "averages": {}}
    monkeypatch.setattr("backend.main.data_api.phenology_weekly", lambda inv: mock_pheno)
    monkeypatch.setattr("backend.main.data_api.sensor_history", lambda inv, var: mock_hist)
    monkeypatch.setattr("backend.main.data_api.validate_and_average", lambda rows: mock_avg)
    return mock_pheno


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


def make_user(user_id="uid-1"):
    """Build mock Supabase get_user + profile responses."""
    mock_user = MagicMock()
    mock_user.id = user_id
    profile_data = {"disabled": False, "full_name": "Test User"}
    return mock_user, profile_data
```

- [ ] **Step 3: Write failing tests in `backend/tests/test_auth.py`**

```python
def test_missing_auth_header_returns_401(api_client):
    r = api_client.get("/me")
    assert r.status_code == 401


def test_malformed_header_returns_401(api_client):
    r = api_client.get("/me", headers={"Authorization": "NotBearer abc"})
    assert r.status_code == 401


def test_invalid_token_returns_401(api_client, mock_supa):
    from gotrue.errors import AuthApiError
    mock_supa.auth.get_user.side_effect = AuthApiError("invalid", 401, {})
    r = api_client.get("/me", headers={"Authorization": "Bearer bad-token"})
    assert r.status_code == 401


def test_disabled_user_returns_403(api_client, mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    profile["disabled"] = True
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 403


def test_valid_token_returns_user_info(api_client, mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 200
    assert r.json()["user_id"] == "uid-1"


def test_inference_requires_auth(api_client):
    r = api_client.get("/inference/3")
    assert r.status_code == 401


def test_metrics_requires_auth(api_client):
    r = api_client.get("/metrics/3")
    assert r.status_code == 401
```

- [ ] **Step 4: Run to confirm failures**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction
python -m pytest backend/tests/test_auth.py -v 2>&1 | head -30
```

Expected: `ImportError` or `FAILED` — `auth.py` and `/me` don't exist yet.

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
```

- [ ] **Step 6: Add `/config` and `/me` to `backend/main.py`**

Add imports after existing imports:
```python
import os
from backend.auth import get_current_user
from typing import Annotated
```

Add routes after the `/health` route:
```python
@app.get('/config')
def config():
    return {
        'supabase_url': os.environ['SUPABASE_URL'],
        'supabase_anon_key': os.environ['SUPABASE_ANON_KEY'],
    }


@app.get('/me')
def me(current_user: Annotated[dict, Depends(get_current_user)]):
    return current_user
```

- [ ] **Step 7: Run tests**

```bash
python -m pytest backend/tests/test_auth.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/auth.py backend/main.py backend/tests/__init__.py backend/tests/conftest.py backend/tests/test_auth.py backend/routers/__init__.py
git commit -m "feat(auth): JWT get_current_user dep, /me + /config endpoints"
```

---

## Task 5: Protect Existing Inference Routes

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_auth.py`

- [ ] **Step 1: Add new failing tests to `backend/tests/test_auth.py`**

```python
def test_inference_requires_auth(api_client):
    r = api_client.get("/inference/3")
    assert r.status_code == 401

def test_metrics_requires_auth(api_client):
    r = api_client.get("/metrics/3")
    assert r.status_code == 401

def test_phenology_get_requires_auth(api_client):
    r = api_client.get("/phenology/3")
    assert r.status_code == 401

def test_phenology_post_requires_auth(api_client):
    r = api_client.post("/phenology/3", json={"rows": []})
    assert r.status_code == 401

def test_sensor_history_requires_auth(api_client):
    r = api_client.get("/sensor-history/3?var=temp")
    assert r.status_code == 401
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest backend/tests/test_auth.py::test_inference_requires_auth backend/tests/test_auth.py::test_phenology_get_requires_auth -v
```

Expected: FAIL — routes return 200 without auth.

- [ ] **Step 3: Add JWT dep to all data routes in `backend/main.py`**

Find and update `inference`:
```python
@app.get('/inference/{inv}')
def inference(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    _check_inv(inv)
    return ENGINE.payload(inv)
```

Find and update `inference_window`:
```python
@app.get('/inference/{inv}/window')
def inference_window(
    inv: int,
    cursor: int | None = Query(None, description='Índice de semana T17 (0..n-1)'),
    horizon: int = Query(6, ge=1, le=12),
    past: int = Query(5, ge=0, le=20),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    _check_inv(inv)
    return ENGINE.window(inv, cursor=cursor, horizon=horizon, past=past)
```

Find and update `metrics`:
```python
@app.get('/metrics/{inv}')
def metrics(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    _check_inv(inv)
    return ENGINE.metrics(inv)
```

Find and update `predictions` (compat alias):
```python
@app.get('/predictions/{inv}')
def predictions(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    _check_inv(inv)
    return ENGINE.payload(inv)
```

Find and update `phenology` GET:
```python
@app.get('/phenology/{inv}')
def phenology(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    _check_inv(inv)
    return data_api.phenology_weekly(inv)
```

Find and update `submit_phenology` POST:
```python
@app.post('/phenology/{inv}')
def submit_phenology(
    inv: int,
    payload: PhenologySubmission,
    _user: Annotated[dict, Depends(get_current_user)],
):
    _check_inv(inv)
    try:
        result = data_api.validate_and_average(payload.rows)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        'inv_id': inv,
        'week': payload.week,
        'persisted': False,
        'note': 'Demo sin persistencia — los datos se guardarán en Supabase tras el deployment.',
        **result,
    }
```

Find and update `sensor_hist`:
```python
@app.get('/sensor-history/{inv}')
def sensor_hist(
    inv: int,
    var: str = Query(..., description='temp|hr|co2|ce|par'),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    _check_inv(inv)
    try:
        return data_api.sensor_history(inv, var)
    except KeyError as e:
        raise HTTPException(404, str(e))
```

- [ ] **Step 3: Run all auth tests**

```bash
python -m pytest backend/tests/test_auth.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat(auth): protect inference/metrics routes with JWT"
```

---

## Task 6: Sensor Ingest Endpoint (TDD)

**Files:**
- Create: `backend/routers/ingest.py`
- Create: `backend/tests/test_ingest.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_ingest.py`**

```python
from unittest.mock import MagicMock


def test_ingest_rejects_no_token(api_client):
    r = api_client.post("/ingest/sensors", json={
        "greenhouse_id": 3, "sensor_name": "temp_interior",
        "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z",
    })
    assert r.status_code == 401


def test_ingest_rejects_wrong_token(api_client):
    r = api_client.post("/ingest/sensors",
        json={"greenhouse_id": 3, "sensor_name": "temp_interior",
              "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_ingest_accepts_valid_token(api_client, mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    r = api_client.post("/ingest/sensors",
        json={"greenhouse_id": 3, "sensor_name": "temp_interior",
              "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z"},
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_ingest_writes_to_supabase(api_client, mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    api_client.post("/ingest/sensors",
        json={"greenhouse_id": 4, "sensor_name": "hr_interior",
              "value": 65.0, "recorded_at": "2026-06-22T11:00:00Z"},
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    mock_supa.table.assert_called_with("sensor_readings")
    inserted = mock_supa.table.return_value.insert.call_args[0][0]
    assert inserted["greenhouse_id"] == 4
    assert inserted["sensor_name"] == "hr_interior"
    assert inserted["value"] == 65.0
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest backend/tests/test_ingest.py -v 2>&1 | head -20
```

Expected: `ImportError` — router doesn't exist.

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

Add import after existing imports:
```python
from backend.routers import ingest as ingest_router
```

Add after `app.include_router` calls (or after middleware setup):
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
git commit -m "feat(ingest): POST /ingest/sensors with static token auth"
```

---

## Task 7: Sensors Read Endpoint (TDD)

**Files:**
- Create: `backend/routers/sensors.py`
- Modify: `backend/tests/test_ingest.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Add failing tests to `backend/tests/test_ingest.py`**

```python
def test_sensors_requires_jwt(api_client):
    r = api_client.get("/sensors/3")
    assert r.status_code == 401


def test_sensors_returns_list(api_client, mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile

    readings_mock = MagicMock()
    readings_mock.execute.return_value.data = [
        {"sensor_name": "temp_interior", "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z"}
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value = readings_mock

    r = api_client.get("/sensors/3", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
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
git commit -m "feat(sensors): GET /sensors/{inv} reads recent sensor_readings"
```

---

## Task 8: User Management API (TDD)

**Files:**
- Create: `backend/routers/admin.py`
- Create: `backend/tests/test_admin.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_admin.py`**

```python
from unittest.mock import MagicMock


def _auth_headers(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_list_users_requires_auth(api_client):
    r = api_client.get("/admin/users")
    assert r.status_code == 401


def test_list_users_returns_profiles(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    mock_supa.table.return_value.select.return_value.execute.return_value.data = [
        {"id": "uid-1", "full_name": "Juan", "disabled": False, "created_at": "2026-01-01T00:00:00Z"}
    ]
    r = api_client.get("/admin/users", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_user_requires_auth(api_client):
    r = api_client.post("/admin/users", json={
        "email": "new@test.com", "full_name": "New User"
    })
    assert r.status_code == 401


def test_create_user_invites_and_inserts_profile(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    new_user = MagicMock()
    new_user.id = "new-uid"
    mock_supa.auth.admin.invite_user_by_email.return_value.user = new_user
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()

    r = api_client.post("/admin/users", headers=headers, json={
        "email": "juan@huerta.com", "full_name": "Juan García"
    })
    assert r.status_code == 200
    assert r.json()["user_id"] == "new-uid"
    mock_supa.auth.admin.invite_user_by_email.assert_called_once_with("juan@huerta.com")


def test_disable_user_requires_auth(api_client):
    r = api_client.patch("/admin/users/some-uid", json={"disabled": True})
    assert r.status_code == 401


def test_disable_user_updates_profile(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    mock_supa.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    r = api_client.patch("/admin/users/target-uid", headers=headers, json={"disabled": True})
    assert r.status_code == 200
    mock_supa.table.assert_called_with("profiles")
    mock_supa.table.return_value.update.assert_called_with({"disabled": True})
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest backend/tests/test_admin.py -v 2>&1 | head -20
```

Expected: `ImportError` — router doesn't exist.

- [ ] **Step 3: Create `backend/routers/admin.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter(prefix="/admin")


class CreateUserRequest(BaseModel):
    email: str
    full_name: str


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
    invite_resp = service_client.auth.admin.invite_user_by_email(body.email)
    user_id = str(invite_resp.user.id)
    service_client.table("profiles").insert({
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
git commit -m "feat(admin): user management — list/invite/disable, any authenticated user"
```

---

## Task 9: Update login.html with Supabase Auth

**Files:**
- Modify: `demo/login.html`

- [ ] **Step 1: Replace the `<script>` block at the bottom of `demo/login.html`**

Current block (lines ~194–214) starts with `<script>` and ends `</script>`. Replace it entirely with:

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

    const { data, error } = await sb.auth.signInWithPassword({ email, password: pass });

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

- [ ] **Step 2: Manual test**

Set env vars and start backend:
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export SUPABASE_SERVICE_KEY="your-service-role-key"
export INGEST_TOKEN="some-secret-token"
uvicorn backend.main:app --port 8000 --reload
```

Open `http://localhost:8000/`. Login with the user created in Task 3. Should redirect to `/app`. Wrong credentials should show "Credenciales inválidas."

- [ ] **Step 3: Commit**

```bash
git add demo/login.html
git commit -m "feat(login): replace fake auth with Supabase email+password"
```

---

## Task 10: Update Dashboard HTML — JWT + Sensor Polling + Usuarios Panel

**Files:**
- Modify: `demo/Demo Dashboard.html`

Three changes: (1) JWT headers on all API calls, (2) live sensor polling on "Tiempo real" views, (3) "Usuarios" management panel.

- [ ] **Step 1: Replace session guard and update `fetchJSON`**

Find (line ~1892):
```javascript
if (!sessionStorage.getItem('jata_auth')) {
  window.location.replace('/');
}
```

Replace with:
```javascript
if (!localStorage.getItem('sb_token')) { window.location.replace('/'); }
```

Find `fetchJSON` function (line ~1938):
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

Also find the direct `fetch()` for phenology POST (line ~2842) — this bypasses `fetchJSON` so needs JWT added manually:

```javascript
    const r = await fetch(`${API_BASE}/phenology/${inv}`, {
```

Replace with:
```javascript
    const token = localStorage.getItem('sb_token');
    const r = await fetch(`${API_BASE}/phenology/${inv}`, {
```

And in the same `fetch()` options object, add the Authorization header. Find the `headers:` inside that fetch and add `'Authorization': \`Bearer ${token}\`` to it. If no `headers` key exists, add:
```javascript
      headers: { 'Authorization': `Bearer ${token}` },
```

- [ ] **Step 2: Add live sensor polling for "Tiempo real" views**

Add a `postJSON` helper near `fetchJSON`:
```javascript
async function postJSON(path, body) {
  const token = localStorage.getItem('sb_token');
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}
```

Add sensor polling functions before the closing `</script>`:
```javascript
// ── Live sensor polling ────────────────────────────────────────────────────
const SENSOR_LABELS = {
  temp_prom_int: 'Temp interior (°C)',
  hr_prom_int:   'Humedad interior (%)',
  co2_ppm:       'CO₂ (ppm)',
  temp_prom_ext: 'Temp exterior (°C)',
  rad_sum:       'Radiación (W/m²)',
  riego_total:   'Riego total (L)',
};

const _sensorIntervals = {};

async function startSensorPolling(inv) {
  if (_sensorIntervals[inv]) clearInterval(_sensorIntervals[inv]);
  await refreshSensors(inv);
  _sensorIntervals[inv] = setInterval(() => refreshSensors(inv), 30000);
}

function stopSensorPolling(inv) {
  if (_sensorIntervals[inv]) {
    clearInterval(_sensorIntervals[inv]);
    delete _sensorIntervals[inv];
  }
}

async function refreshSensors(inv) {
  const data = await fetchJSON(`/sensors/${inv}?limit=200`);
  if (!data) return;

  const grid = document.getElementById(`sensor-grid-${inv}`);
  if (!grid) return;

  // Group latest reading per sensor
  const latest = {};
  data.forEach(r => { if (!latest[r.sensor_name]) latest[r.sensor_name] = r; });

  grid.innerHTML = Object.entries(latest).map(([name, r]) => `
    <div style="background:#1e2a3a;border-radius:8px;padding:16px;min-width:140px">
      <div style="font-size:0.75rem;color:#9aa0a8;margin-bottom:6px">${SENSOR_LABELS[name] || name}</div>
      <div style="font-size:1.4rem;font-weight:600;color:#f4ede0">${r.value.toFixed(1)}</div>
      <div style="font-size:0.7rem;color:#9aa0a8;margin-top:4px">${new Date(r.recorded_at).toLocaleTimeString('es-MX')}</div>
    </div>
  `).join('') || '<p style="color:#9aa0a8">Sin lecturas recientes.</p>';
}
```

Wire polling to view navigation: find where `showView` is defined and add inside it:
```javascript
// start/stop sensor polling when entering/leaving live views
if (viewId === 'view-live3') startSensorPolling(3);
else stopSensorPolling(3);
if (viewId === 'view-live4') startSensorPolling(4);
else stopSensorPolling(4);
```

- [ ] **Step 3: Add "Usuarios" nav link**

Find the last `<a class="sb-link"` in the sidebar. After it add:
```html
<a class="sb-link" data-goto="view-usuarios">
  <span class="sb-icon">👥</span>Usuarios
</a>
```

- [ ] **Step 4: Add "Usuarios" view section**

Before the closing `</div>` of `<div class="app">`, add:

```html
<section class="view" id="view-usuarios">
  <div style="padding:32px;max-width:860px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
      <h2 style="font-size:1.4rem;font-weight:600">Gestión de Usuarios</h2>
      <button onclick="openNewUserModal()" style="background:#65a30d;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:0.9rem">+ Nuevo usuario</button>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem" id="users-table">
      <thead>
        <tr style="border-bottom:2px solid #e2e5e9">
          <th style="text-align:left;padding:10px 8px">Nombre</th>
          <th style="text-align:left;padding:10px 8px">Estado</th>
          <th style="text-align:left;padding:10px 8px">Creado</th>
          <th style="padding:10px 8px"></th>
        </tr>
      </thead>
      <tbody id="users-tbody"></tbody>
    </table>
  </div>

  <div id="new-user-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:1000;align-items:center;justify-content:center">
    <div style="background:#fff;border-radius:10px;padding:32px;width:400px;max-width:95vw">
      <h3 style="margin-bottom:20px;font-size:1.1rem">Nuevo usuario</h3>
      <label style="display:block;margin-bottom:12px;font-size:0.9rem">
        Correo electrónico<br>
        <input id="nu-email" type="email" style="width:100%;margin-top:4px;padding:8px;border:1px solid #d2d6db;border-radius:6px">
      </label>
      <label style="display:block;margin-bottom:20px;font-size:0.9rem">
        Nombre completo<br>
        <input id="nu-name" type="text" style="width:100%;margin-top:4px;padding:8px;border:1px solid #d2d6db;border-radius:6px">
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

- [ ] **Step 5: Add Usuarios JS before closing `</script>`**

```javascript
// ── Usuarios management ───────────────────────────────────────────────────
async function loadUsers() {
  const users = await fetchJSON('/admin/users');
  if (!users) return;
  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = users.map(u => `
    <tr style="border-bottom:1px solid #e2e5e9">
      <td style="padding:10px 8px">${u.full_name || '—'}</td>
      <td style="padding:10px 8px">
        <span style="color:${u.disabled ? '#dc2626' : '#16a34a'}">${u.disabled ? 'Desactivado' : 'Activo'}</span>
      </td>
      <td style="padding:10px 8px;color:#9aa0a8">${new Date(u.created_at).toLocaleDateString('es-MX')}</td>
      <td style="padding:10px 8px">
        <button onclick="toggleUser('${u.id}', ${u.disabled})"
          style="font-size:0.8rem;padding:4px 10px;border:1px solid #d2d6db;background:#fff;border-radius:4px;cursor:pointer">
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
  const email = document.getElementById('nu-email').value.trim();
  const name  = document.getElementById('nu-name').value.trim();
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
    body: JSON.stringify({ email, full_name: name }),
  });
  if (!r.ok) {
    const err = document.getElementById('nu-error');
    err.textContent = 'Error al crear usuario.';
    err.style.display = 'block';
    return;
  }
  closeNewUserModal();
  loadUsers();
}

document.querySelector('[data-goto="view-usuarios"]')
  ?.addEventListener('click', loadUsers);
```

- [ ] **Step 6: Manual test**

```bash
uvicorn backend.main:app --port 8000 --reload
```

1. Login → dashboard loads
2. All fetch calls return data (check browser console — no 401s)
3. Click "Usuarios" → table loads
4. Click "+ Nuevo usuario" → modal → fill form → "Invitar" → user receives invite email
5. Click "Tiempo real" for inv3 → sensor grid renders (empty until mock_ingest runs)

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): JWT auth, live sensor polling, Usuarios management panel"
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

COLUMN_MAP = {
    "Temperatura Promedio":       "temp_prom_int",
    "Temperatura Mínima":         "temp_min_int",
    "Temperatura Máxima":         "temp_max_int",
    "Humedad Relativa Promedio":  "hr_prom_int",
    "CO2":                        "co2_ppm",
}


def load_sensor_data(inv: int) -> pd.DataFrame:
    xl = pd.ExcelFile(SENSOR_FILES[inv])
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Fecha" not in df.columns:
            continue
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        frames.append(df.dropna(subset=["Fecha"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        print(f"No data for inv {args.inv}")
        sys.exit(1)

    sent = 0
    for _, row in df.iterrows():
        ts = row["Fecha"].isoformat()
        for col, sensor_name in COLUMN_MAP.items():
            if col not in row or pd.isna(row[col]):
                continue
            if not args.dry_run:
                requests.post(
                    f"{args.host}/ingest/sensors",
                    json={"greenhouse_id": args.inv, "sensor_name": sensor_name,
                          "value": float(row[col]), "recorded_at": ts},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                ).raise_for_status()
            sent += 1

    print(f"{'[DRY RUN] Would send' if args.dry_run else 'Sent'} {sent} readings for inv {args.inv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry-run**

```bash
python scripts/mock_ingest.py --inv 3 --dry-run
```

Expected: `[DRY RUN] Would send N readings for inv 3` with N > 0.

- [ ] **Step 3: Test live (requires running backend)**

```bash
INGEST_TOKEN=some-secret-token python scripts/mock_ingest.py --inv 3 --host http://localhost:8000
```

Then check Supabase Table Editor: `sensor_readings` table has rows. Open "Tiempo real" view in dashboard — sensor grid populates.

- [ ] **Step 4: Commit**

```bash
git add scripts/mock_ingest.py
git commit -m "feat(scripts): Excel sensor replay script for ingest pipeline testing"
```

---

## Task 12: Telegraf Configuration

**Files:**
- Create: `telegraf/telegraf.conf`
- Create: `telegraf/README.md`

- [ ] **Step 1: Create `telegraf/telegraf.conf`**

```toml
# Huerta Prediction — Telegraf config
# Set env var: TELEGRAF_INGEST_TOKEN=your-ingest-token
# Run: telegraf --config telegraf/telegraf.conf

[agent]
  interval            = "1m"
  round_interval      = true
  metric_batch_size   = 100
  metric_buffer_limit = 10000
  flush_interval      = "10s"
  quiet               = false

# ── INPUT: swap to match hardware ─────────────────────────────────────────
# Option A: MQTT broker
[[inputs.mqtt_consumer]]
  servers     = ["tcp://localhost:1883"]
  topics      = ["greenhouse/+/sensors"]
  data_format = "json"

# Option B: HTTP listener (datalogger POSTs to Telegraf)
# [[inputs.http_listener_v2]]
#   service_address = ":8080"
#   data_format     = "json"

# Option C: File drop (CSV from datalogger software)
# [[inputs.file]]
#   files       = ["/data/sensors/*.csv"]
#   data_format = "csv"
#   csv_header_row_count = 1

# ── OUTPUT ─────────────────────────────────────────────────────────────────
[[outputs.http]]
  url         = "https://your-app.railway.app/ingest/sensors"
  method      = "POST"
  data_format = "json"
  [outputs.http.headers]
    Content-Type  = "application/json"
    Authorization = "Bearer ${TELEGRAF_INGEST_TOKEN}"

# Debug: also log to stdout (disable in production)
[[outputs.file]]
  files       = ["stdout"]
  data_format = "json"
```

- [ ] **Step 2: Create `telegraf/README.md`**

```markdown
# Telegraf Setup

Install: `brew install telegraf` (macOS) or `sudo apt-get install telegraf` (Ubuntu)

Run:
```bash
export TELEGRAF_INGEST_TOKEN="your-ingest-token"
telegraf --config telegraf/telegraf.conf
```

Swap input: comment out `[[inputs.mqtt_consumer]]`, uncomment the option matching your hardware.
Update `url` in `[[outputs.http]]` to the production Railway URL when deployed.
```

- [ ] **Step 3: Commit**

```bash
git add telegraf/telegraf.conf telegraf/README.md
git commit -m "feat(telegraf): sensor pipeline config template"
```

---

## Task 13: Dockerfile + Deployment

**Files:**
- Create: `Dockerfile`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `.env.example`**

```bash
# Copy to .env for local dev — never commit .env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-role-key-here
INGEST_TOKEN=choose-a-strong-random-secret
```

- [ ] **Step 3: Ensure .env is gitignored**

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

Expected: container starts, `http://localhost:8000/health` returns `{"status":"ok",...}`.

- [ ] **Step 5: Deploy to Railway**

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
2. Select `JosueT1212/Huerta_Prediction`
3. Railway auto-detects `Dockerfile`
4. Variables tab → add all 4 env vars with real Supabase values
5. Deploy → wait ~3 minutes
6. Open Railway URL → `https://your-app.railway.app/health` → verify `{"status":"ok"}`
7. Update `telegraf/telegraf.conf` `url` to the real Railway URL

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .env.example .gitignore telegraf/telegraf.conf
git commit -m "feat(deploy): Dockerfile + Railway config"
```

---

## Run Full Test Suite

```bash
python -m pytest backend/tests/ -v
```

Expected: all tests PASS.
