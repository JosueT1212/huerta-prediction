# Insertar Datos Excel Upload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual Fenología/Producción real forms in "Insertar datos" with 3 independent weekly Excel uploads (Sensores/Fenología/Producción), each rate-limited to once per 7 days per greenhouse; remove per-row deletion everywhere; remove the dead "Tiempo real" live-sensor tab and its backend (`ingest.py`, `sensors.py`, `sensor_readings` table).

**Architecture:** New `backend/routers/uploads.py` handles `POST /uploads/{inv}/{form_type}` (parses xlsx via pandas, upserts into the matching table, enforces a `submission_locks` row) and `GET /uploads/{inv}/{form_type}/history` (Historial data) and `GET /submission-lock/{inv}/{form_type}` (lock status for the frontend). `demo/Demo Dashboard.html` gets its Insertar datos section rewritten to 4 tabs (Sensores/Fenología/Producción/Historial) using client-side xlsx parsing (SheetJS, new CDN dep) for upload previews, and the "Tiempo real" view + all delete UI removed.

**Tech Stack:** FastAPI, Supabase (postgres), pandas/openpyxl (already deps), vanilla JS + SheetJS (new CDN script tag, no build step, matches existing Chart.js CDN pattern).

## Global Constraints

- Weekly lock: rolling 7 days from `last_submitted_at`, hard block (backend 429 + frontend disables control before submit).
- Lock is per `(greenhouse_id, form_type)`, `form_type ∈ {sensores, fenologia, produccion}` — independent per type and per invernadero.
- Excel columns must match DB column names exactly (see per-task tables below) — no renaming/aliasing layer.
- Producción upload is **update-only**: never inserts a new `predictions` row (`kg_predicted` is `NOT NULL`); unmatched weeks go to `rows_skipped`.
- No deletion capability anywhere in Insertar datos after this change (Historial is read-only).
- Auth: every new/kept endpoint requires `Depends(get_current_user)` (`backend/auth.py`), same as existing routers.
- Follow existing router pattern exactly: `APIRouter()`, `service_client.table(...)`, `Annotated[dict, Depends(get_current_user)] = None` param style (see `backend/routers/phenology_live.py`).
- The weekly lock only engages when at least one row was actually written (`rows_inserted + rows_updated > 0`). An upload that skips every row (e.g. Inv4 producción — see below) must not burn the 7-day window on zero data saved.
- Producción upload for Invernadero 4 will skip 100% of rows today — the live-inference cron (`docs/superpowers/specs/2026-06-23-live-inference-pipeline-design.md`) only generates `predictions` rows for Inv3. This is an accepted pre-existing limitation (the old manual form had the same gap), not a bug to fix here.
- Re-uploading a file that changes a previously-saved row's values **silently overwrites** that row (upsert, no conflict check). Accepted: the weekly lock controls upload *cadence*, not data mutability — client owns their data.

---

## Task 0: Confirm green baseline

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Run the existing backend test suite before touching anything**

Run: `cd backend && python -m pytest tests/ -v`

**Confirmed result as of this plan being written: 10 pre-existing failures, 24 passed.** All 10 failures are the same root cause: `backend/auth.py`'s `get_current_user` calls `.maybe_single()` on the Supabase query chain, but the `_auth()` test helpers in `test_admin.py`, `test_phenology_live.py`, and `test_predictions.py` mock `.single()` instead — the unconfigured `.maybe_single()` mock returns a truthy `MagicMock`, `profile["disabled"]` then returns another truthy `MagicMock`, and `get_current_user` raises 403 "Account disabled". (`test_auth.py`'s 2 failures and `test_ingest.py`'s 1 are separate pre-existing issues.)

**This is out of scope for this plan** — it predates this feature and isn't something the excel-upload work touches or worsens. Do not fix it as part of this plan; a fix belongs in its own small unrelated PR. What matters here: `backend/tests/test_uploads.py` (Task 3) uses a **corrected** `_auth()` helper that mocks `.maybe_single()` — copy it exactly as written in Task 3, not the broken pattern from the other test files, or the new tests will inherit the same false-403 failures.

Record the baseline count (10 failed / 24 passed) so later "run full suite" steps can distinguish new regressions from this pre-existing, known-and-accepted set.

---

## Task 1: `submission_locks` table + drop narrow `sensor_readings` table

**Files:**
- Create: `supabase/migrations/006_submission_locks.sql`
- Create: `supabase/migrations/007_drop_sensor_readings.sql`

**Interfaces:**
- Produces: table `submission_locks(greenhouse_id int, form_type text, last_submitted_at timestamptz)`, primary key `(greenhouse_id, form_type)`. Consumed by Task 2 (`backend/lock_utils.py`).

- [ ] **Step 1: Write migration 006**

```sql
-- supabase/migrations/006_submission_locks.sql
create table if not exists submission_locks (
  greenhouse_id      int         not null,
  form_type          text        not null check (form_type in ('sensores','fenologia','produccion')),
  last_submitted_at  timestamptz not null default now(),
  primary key (greenhouse_id, form_type)
);
```

- [ ] **Step 2: Write migration 007**

```sql
-- supabase/migrations/007_drop_sensor_readings.sql
drop table if exists sensor_readings;
```

- [ ] **Step 3: Apply migrations to the linked Supabase project**

Run: `supabase db push` (from repo root; requires `supabase/.temp` link already configured per existing setup). If the CLI isn't linked in this environment, apply manually via the Supabase SQL editor and confirm both statements ran without error.

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/006_submission_locks.sql supabase/migrations/007_drop_sensor_readings.sql
git commit -m "feat(db): add submission_locks table, drop narrow sensor_readings table"
```

---

## Task 2: `backend/lock_utils.py` — shared weekly-lock helper

**Files:**
- Create: `backend/lock_utils.py`
- Test: `backend/tests/test_lock_utils.py`

**Interfaces:**
- Consumes: `backend.supabase_client.service_client` (existing).
- Produces:
  - `check_submission_lock(greenhouse_id: int, form_type: str) -> None` — raises `fastapi.HTTPException(429, detail={"next_allowed_at": "<isoformat>"})` if locked, else returns `None`. Used by Task 3.
  - `touch_submission_lock(greenhouse_id: int, form_type: str) -> None` — upserts `last_submitted_at = now()`. Used by Task 3.
  - `get_lock_status(greenhouse_id: int, form_type: str) -> dict` — returns `{"locked": bool, "next_allowed_at": str | None}`. Used by Task 3's `GET /submission-lock/{inv}/{form_type}`.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_lock_utils.py
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException


def _mock_row(last_submitted_at_iso):
    row = MagicMock()
    row.data = [{"last_submitted_at": last_submitted_at_iso}] if last_submitted_at_iso else []
    return row


def test_check_lock_raises_when_within_7_days(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    with pytest.raises(HTTPException) as exc_info:
        lock_utils.check_submission_lock(3, "sensores")
    assert exc_info.value.status_code == 429
    assert "next_allowed_at" in exc_info.value.detail


def test_check_lock_passes_when_no_prior_submission(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.check_submission_lock(3, "sensores")  # should not raise


def test_check_lock_passes_when_older_than_7_days(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(old)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.check_submission_lock(3, "sensores")  # should not raise


def test_touch_lock_upserts(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.touch_submission_lock(3, "sensores")
    mock.table.assert_called_with("submission_locks")
    mock.table.return_value.upsert.assert_called_once()
    kwargs = mock.table.return_value.upsert.call_args
    assert kwargs[0][0]["greenhouse_id"] == 3
    assert kwargs[0][0]["form_type"] == "sensores"
    assert kwargs[1]["on_conflict"] == "greenhouse_id,form_type"


def test_get_lock_status_locked(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    status = lock_utils.get_lock_status(3, "sensores")
    assert status["locked"] is True
    assert status["next_allowed_at"] is not None


def test_get_lock_status_unlocked_no_prior(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    status = lock_utils.get_lock_status(3, "sensores")
    assert status == {"locked": False, "next_allowed_at": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_lock_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.lock_utils'`

- [ ] **Step 3: Write implementation**

```python
# backend/lock_utils.py
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from backend.supabase_client import service_client

LOCK_WINDOW = timedelta(days=7)


def _fetch_last_submitted_at(greenhouse_id: int, form_type: str) -> datetime | None:
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
    last = _fetch_last_submitted_at(greenhouse_id, form_type)
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
    last = _fetch_last_submitted_at(greenhouse_id, form_type)
    if last is None:
        return {"locked": False, "next_allowed_at": None}
    next_allowed = last + LOCK_WINDOW
    locked = datetime.now(timezone.utc) < next_allowed
    return {"locked": locked, "next_allowed_at": next_allowed.isoformat() if locked else None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_lock_utils.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add backend/lock_utils.py backend/tests/test_lock_utils.py
git commit -m "feat(backend): add submission_locks weekly-lock helper"
```

---

## Task 3: `backend/routers/uploads.py` — Excel upload, history, lock-status endpoints

**Files:**
- Create: `backend/routers/uploads.py`
- Test: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `backend.lock_utils.check_submission_lock`, `touch_submission_lock`, `get_lock_status` (Task 2); `backend.auth.get_current_user`; `backend.supabase_client.service_client`.
- Produces:
  - `router = APIRouter()` with:
    - `POST /uploads/{inv}/{form_type}` (multipart `file: UploadFile`) → `{rows_in_file, rows_inserted, rows_updated, rows_skipped, skipped_reasons: list[str]}`
    - `GET /uploads/{inv}/{form_type}/history?limit=200` → list of raw rows from the matching table
    - `GET /submission-lock/{inv}/{form_type}` → `{locked, next_allowed_at}`
  - Consumed by Task 4 (main.py registration) and Task 8 (frontend).

**Per-`form_type` mapping (exact):**

| form_type | table | required excel columns | unique/match key | write mode |
|---|---|---|---|---|
| `sensores` | `sensor_readings_wide` | `fecha, temp_prom_int, temp_min_int, temp_max_int, hr_prom_int, co2_ppm, riego_total, ph_promedio, ce_promedio, temp_prom_ext, temp_max_ext, temp_min_ext, rad_sum` | `greenhouse_id, fecha` | upsert |
| `fenologia` | `phenology_observations` | `fecha, zona, planta, racimos_puestos, flores_racimo_abiertas, racimos_en_planta, cantidad_tomates, racimo_en_cosecha, tomates_maduros, diametro_fruto_cm, crecimiento_planta_cm` | `greenhouse_id, week_date, zona, planta` | upsert (`fecha`→`week_date`) |
| `produccion` | `predictions` | `fecha, kg_reales` | `greenhouse_id, predicted_for` | **update-only** (`fecha`→`predicted_for`, `kg_reales`→`kg_actual`); no match → skipped |

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_uploads.py
import io
from unittest.mock import MagicMock
import pandas as pd
import pytest


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    # backend/auth.py calls .maybe_single(), not .single() — mock the actual
    # chain used, unlike the pre-existing (broken) helper in
    # test_phenology_live.py/test_predictions.py, which mocks .single() and
    # makes every authed request in those files 403 (see Task 0 baseline).
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.read()


def _no_lock(mock_supa):
    # submission_locks select → no prior row → unlocked
    mock_supa.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []


def test_upload_requires_auth(api_client):
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 100}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", files=files)
    assert r.status_code == 401


def test_upload_rejects_invalid_form_type(api_client, mock_supa):
    headers = _auth(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01"}])
    files = {"file": ("x.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/bogus", headers=headers, files=files)
    assert r.status_code == 422


def test_upload_returns_429_when_locked(api_client, mock_supa):
    headers = _auth(mock_supa)
    from datetime import datetime, timezone
    mock_supa.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"last_submitted_at": datetime.now(timezone.utc).isoformat()}
    ]
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 100}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 429


def test_upload_rejects_missing_columns(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01"}])  # missing kg_reales
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 422
    assert "kg_reales" in r.text


def test_upload_sensores_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
        "temp_prom_ext": 20.0, "temp_max_ext": 26.0, "temp_min_ext": 15.0,
        "rad_sum": 300.0,
    }])
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_in_file"] == 1
    assert body["rows_inserted"] == 1
    mock_supa.table.assert_any_call("sensor_readings_wide")
    mock_supa.table.assert_any_call("submission_locks")


def test_upload_produccion_skips_unmatched_week_and_does_not_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    # predictions update returns no matched row (empty data) → skipped
    mock_supa.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 5000}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_updated"] == 0
    assert body["rows_skipped"] == 1
    # all rows skipped → lock must NOT engage (nothing was actually saved)
    mock_supa.table.assert_any_call("predictions")
    assert mock_supa.table.call_args_list[-1] != (("submission_locks",),)


def test_history_requires_auth(api_client):
    r = api_client.get("/uploads/3/fenologia/history")
    assert r.status_code == 401


def test_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"week_date": "2026-07-01", "zona": 1, "planta": 1}
    ]
    r = api_client.get("/uploads/3/fenologia/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_lock_status_endpoint_requires_auth(api_client):
    r = api_client.get("/submission-lock/3/sensores")
    assert r.status_code == 401


def test_lock_status_endpoint_returns_status(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/3/sensores", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: FAIL — route `/uploads/...` not found (404) since router doesn't exist/isn't registered yet.

- [ ] **Step 3: Write implementation**

```python
# backend/routers/uploads.py
import io
from typing import Annotated
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from backend.auth import get_current_user
from backend.supabase_client import service_client
from backend.lock_utils import check_submission_lock, touch_submission_lock, get_lock_status

router = APIRouter()

SENSOR_COLS = [
    "fecha", "temp_prom_int", "temp_min_int", "temp_max_int", "hr_prom_int",
    "co2_ppm", "riego_total", "ph_promedio", "ce_promedio", "temp_prom_ext",
    "temp_max_ext", "temp_min_ext", "rad_sum",
]
PHENOLOGY_COLS = [
    "fecha", "zona", "planta", "racimos_puestos", "flores_racimo_abiertas",
    "racimos_en_planta", "cantidad_tomates", "racimo_en_cosecha",
    "tomates_maduros", "diametro_fruto_cm", "crecimiento_planta_cm",
]
PRODUCTION_COLS = ["fecha", "kg_reales"]

FORM_TYPES = {
    "sensores": SENSOR_COLS,
    "fenologia": PHENOLOGY_COLS,
    "produccion": PRODUCTION_COLS,
}


def _require_form_type(form_type: str) -> list[str]:
    cols = FORM_TYPES.get(form_type)
    if cols is None:
        raise HTTPException(422, f"form_type inválido: {form_type}")
    return cols


def _table_for(form_type: str) -> str:
    return {
        "sensores": "sensor_readings_wide",
        "fenologia": "phenology_observations",
        "produccion": "predictions",
    }[form_type]


@router.post("/uploads/{inv}/{form_type}")
async def upload_excel(
    inv: int,
    form_type: str,
    file: UploadFile,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    required_cols = _require_form_type(form_type)
    check_submission_lock(inv, form_type)

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
    except Exception as e:
        raise HTTPException(422, f"No se pudo leer el archivo Excel: {e}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Columnas faltantes: {', '.join(missing)}")

    df = df[required_cols].dropna(how="all")
    rows_inserted = rows_updated = rows_skipped = 0
    skipped_reasons: list[str] = []

    if form_type == "sensores":
        for _, row in df.iterrows():
            record = {"greenhouse_id": inv, "fecha": _date_str(row["fecha"])}
            record.update({c: row[c] for c in required_cols if c != "fecha"})
            service_client.table("sensor_readings_wide").upsert(
                record, on_conflict="greenhouse_id,fecha"
            ).execute()
            rows_inserted += 1

    elif form_type == "fenologia":
        for _, row in df.iterrows():
            record = {
                "greenhouse_id": inv,
                "week_date": _date_str(row["fecha"]),
                "zona": int(row["zona"]),
                "planta": int(row["planta"]),
            }
            record.update({c: row[c] for c in required_cols if c not in ("fecha", "zona", "planta")})
            service_client.table("phenology_observations").upsert(
                record, on_conflict="greenhouse_id,week_date,zona,planta"
            ).execute()
            rows_inserted += 1

    elif form_type == "produccion":
        for _, row in df.iterrows():
            resp = (
                service_client.table("predictions")
                .update({"kg_actual": row["kg_reales"]})
                .eq("greenhouse_id", inv)
                .eq("predicted_for", _date_str(row["fecha"]))
                .execute()
            )
            if resp.data:
                rows_updated += 1
            else:
                rows_skipped += 1
                skipped_reasons.append(
                    f"{_date_str(row['fecha'])}: no existe predicción para esa semana"
                )

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(inv, form_type)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


def _date_str(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


@router.get("/uploads/{inv}/{form_type}/history")
def upload_history(
    inv: int,
    form_type: str,
    limit: int = 200,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    _require_form_type(form_type)
    table = _table_for(form_type)
    order_col = {"sensores": "fecha", "fenologia": "week_date", "produccion": "predicted_for"}[form_type]
    resp = (
        service_client.table(table)
        .select("*")
        .eq("greenhouse_id", inv)
        .order(order_col, desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


@router.get("/submission-lock/{inv}/{form_type}")
def lock_status(
    inv: int,
    form_type: str,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    _require_form_type(form_type)
    return get_lock_status(inv, form_type)
```

- [ ] **Step 4: Register the router in `main.py` and add its `conftest.py` patch**

This task must reach green standalone — register the router now rather than deferring to Task 4 (which only removes `ingest`/`sensors`).

In `backend/main.py`, add to the router imports block (after the `phenology_live` import, line 37):
```python
from backend.routers import uploads as uploads_router  # noqa: E402
```
and to the `include_router` block (after the `phenology_live_router.router` line, line 71):
```python
app.include_router(uploads_router.router)
```

In `backend/tests/conftest.py`, add to the `mock_supa` fixture (after the `phenology_live` patch line):
```python
    monkeypatch.setattr("backend.routers.uploads.service_client", mock)
    monkeypatch.setattr("backend.lock_utils.service_client", mock)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_uploads.py tests/test_lock_utils.py -v`
Expected: all pass — router is registered, `/uploads/*` and `/submission-lock/*` routes exist.

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `cd backend && python -m pytest tests/ -v`
Expected: same 10 pre-existing failures as the Task 0 baseline (unrelated `.maybe_single()` mock mismatch, not touched here), all new `test_uploads.py`/`test_lock_utils.py` tests passing, no other new failures. Total should read `10 failed, <24 + new test count> passed`.

- [ ] **Step 7: Commit**

```bash
git add backend/routers/uploads.py backend/tests/test_uploads.py backend/main.py backend/tests/conftest.py
git commit -m "feat(backend): add excel upload, history, and lock-status endpoints"
```

---

## Task 4: Remove ingest/sensors routers (dead live-sensor infra)

**Files:**
- Modify: `backend/main.py:33-37,67-71`
- Delete: `backend/routers/ingest.py`
- Delete: `backend/routers/sensors.py`
- Delete: `backend/tests/test_ingest.py`
- Modify: `backend/tests/conftest.py`

**Interfaces:**
- Consumes: nothing new (Task 3 already registered `uploads_router` independently).
- Produces: `app` no longer exposes `/ingest/sensors` or `/sensors/{inv}`.

- [ ] **Step 1: Remove ingest/sensors imports and registration from `main.py`**

Replace:
```python
from backend.routers import ingest as ingest_router  # noqa: E402
from backend.routers import sensors as sensors_router  # noqa: E402
from backend.routers import admin as admin_router  # noqa: E402
from backend.routers import predictions as predictions_router  # noqa: E402
from backend.routers import phenology_live as phenology_live_router  # noqa: E402
from backend.routers import uploads as uploads_router  # noqa: E402
```
with:
```python
from backend.routers import admin as admin_router  # noqa: E402
from backend.routers import predictions as predictions_router  # noqa: E402
from backend.routers import phenology_live as phenology_live_router  # noqa: E402
from backend.routers import uploads as uploads_router  # noqa: E402
```

Replace:
```python
app.include_router(ingest_router.router)
app.include_router(sensors_router.router)
app.include_router(admin_router.router)
app.include_router(predictions_router.router)
app.include_router(phenology_live_router.router)
app.include_router(uploads_router.router)
```
with:
```python
app.include_router(admin_router.router)
app.include_router(predictions_router.router)
app.include_router(phenology_live_router.router)
app.include_router(uploads_router.router)
```

- [ ] **Step 2: Delete dead router files and their test**

```bash
git rm backend/routers/ingest.py backend/routers/sensors.py backend/tests/test_ingest.py
```

- [ ] **Step 3: Update `backend/tests/conftest.py`**

In the `mock_supa` fixture, replace:
```python
    monkeypatch.setattr("backend.routers.ingest.service_client", mock)
    monkeypatch.setattr("backend.routers.sensors.service_client", mock)
    monkeypatch.setattr("backend.routers.admin.service_client", mock)
    monkeypatch.setattr("backend.routers.predictions.service_client", mock)
    monkeypatch.setattr("backend.routers.phenology_live.service_client", mock)
    monkeypatch.setattr("backend.routers.uploads.service_client", mock)
    monkeypatch.setattr("backend.lock_utils.service_client", mock)
```
with:
```python
    monkeypatch.setattr("backend.routers.admin.service_client", mock)
    monkeypatch.setattr("backend.routers.predictions.service_client", mock)
    monkeypatch.setattr("backend.routers.phenology_live.service_client", mock)
    monkeypatch.setattr("backend.routers.uploads.service_client", mock)
    monkeypatch.setattr("backend.lock_utils.service_client", mock)
```

Also remove the now-unused line `os.environ.setdefault("INGEST_TOKEN", "test-ingest-token")` — `INGEST_TOKEN` was only read by the deleted `ingest.py`.

- [ ] **Step 4: Run full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: `test_ingest.py` is gone, so its 1 pre-existing failure drops out — 9 pre-existing failures remain (unrelated `.maybe_single()` mock mismatch, still out of scope), no new failures, no import errors from the removed routers.

- [ ] **Step 5: Manually verify the server boots**

Run: `uvicorn backend.main:app --port 8000` (from repo root, needs `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` env set — reuse existing local `.env`/shell config).
Expected: starts cleanly, `GET /health` returns 200, `GET /ingest/sensors` and `GET /sensors/3` now 404.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/conftest.py
git commit -m "feat(backend): remove ingest/sensors routers"
```

---

## Task 5: Remove per-row delete endpoints (`phenology-live` DELETE, `live-predictions` kg-actual DELETE)

**Files:**
- Modify: `backend/routers/phenology_live.py:54-69`
- Modify: `backend/routers/predictions.py:44-59`
- Modify: `backend/tests/test_phenology_live.py`
- Modify: `backend/tests/test_predictions.py`

**Interfaces:**
- Produces: `phenology_live.router` no longer has `DELETE /phenology-live/{inv}/{obs_id}`; `predictions.router` no longer has `DELETE /live-predictions/{inv}/{prediction_id}/kg-actual`.

- [ ] **Step 1: Remove the DELETE endpoint from `phenology_live.py`**

Delete lines 54-69 (the `delete_phenology` function) and the now-unused `HTTPException` import if nothing else in the file uses it — check first: `list_phenology`/`upsert_phenology` don't raise `HTTPException`, so remove it from the `fastapi` import line too:

Replace:
```python
from fastapi import APIRouter, Depends, HTTPException, Query
```
with:
```python
from fastapi import APIRouter, Depends, Query
```

- [ ] **Step 2: Remove the DELETE endpoint from `predictions.py`**

Delete lines 44-59 (`clear_kg_actual`). Check `HTTPException` usage elsewhere in the file — it's not used by the remaining two endpoints, so also update the import:

Replace:
```python
from fastapi import APIRouter, Depends, HTTPException, Query
```
with:
```python
from fastapi import APIRouter, Depends, Query
```

- [ ] **Step 3: Remove/update corresponding tests**

In `backend/tests/test_phenology_live.py`, delete any `test_delete_phenology*` tests (none currently present per the file read during planning — confirm with `grep -n delete backend/tests/test_phenology_live.py` before editing; if present, remove them).

In `backend/tests/test_predictions.py`, delete any `test_clear_kg_actual*` / `test_delete*` tests. Run `grep -n "delete\|clear_kg" backend/tests/test_predictions.py` first and remove matching test functions.

- [ ] **Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_phenology_live.py tests/test_predictions.py -v`
Expected: same pre-existing failures as the Task 0 baseline for `test_post_phenology_upserts_observation`, `test_get_phenology_returns_list`, `test_list_predictions_returns_list`, `test_patch_kg_actual_updates_row` (the `.maybe_single()` mock mismatch, still out of scope for this plan); `test_*_requires_auth` tests pass; no references to the deleted endpoints remain in either file.

- [ ] **Step 5: Commit**

```bash
git add backend/routers/phenology_live.py backend/routers/predictions.py backend/tests/test_phenology_live.py backend/tests/test_predictions.py
git commit -m "feat(backend): remove per-row delete endpoints for phenology and kg_actual"
```

---

## Task 6: Frontend — remove "Tiempo real" tab and its JS

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Produces: no `view-live3`/`view-live4` sections, no sidebar links to them, no mock sensor JS. `refreshLiveData`, `renderForecastSection`, `renderMenuBadge`, `extendSliderWithLive` (forecast/prediction related, unrelated to live sensors) are **kept** — verify by grep before deleting anything with "live" in the name.

- [ ] **Step 1: Remove sidebar links**

Delete line 1239: `<button class="sb-link sb-sub" data-view="live3"><span class="ico">◉</span>Tiempo real</button>`
Delete line 1247: `<button class="sb-link sb-sub" data-view="live4"><span class="ico">◉</span>Tiempo real</button>`

- [ ] **Step 2: Remove the two view sections**

Delete the full `<section class="view" id="view-live3">...</section>` block (lines 1811-1849) and `<section class="view" id="view-live4">...</section>` block (lines 1851-1889), including the `<!-- ============ LIVE INV3/4 VIEW ============ -->` comments immediately above each.

- [ ] **Step 3: Remove mock sensor-grid JS**

Delete the block containing `sensorState`, `renderSensors`, `SENSOR_HIST_MAP` note (keep `SENSOR_HIST_MAP` only if used elsewhere — grep first: `grep -n SENSOR_HIST_MAP "demo/Demo Dashboard.html"`; it's also referenced by the sensor-history **modal** feature which stays, so if the grep shows other usages, keep the constant and delete only `renderSensors`/`sensorState`), `renderStatus`, `tickLive`, the initial-render/`setInterval` calls, `makeLiveChart`, `gen24`, and the two `makeLiveChart('chart-live3'...)`/`makeLiveChart('chart-live4'...)` calls — this spans roughly lines 2691-2794. Confirm exact bounds with:
```bash
grep -n "sensorState\|function renderSensors\|function renderStatus\|function tickLive\|function makeLiveChart\|function gen24\|makeLiveChart('chart-live" "demo/Demo Dashboard.html"
```
and delete each matched function body plus its immediate call sites.

- [ ] **Step 4: Remove `startSensorPolling`/`stopSensorPolling`/`refreshSensors` and their `showView` wiring**

In `showView` (around line 2213-2238), delete:
```javascript
  if (viewId === 'live3') startSensorPolling(3);
  else stopSensorPolling(3);
  if (viewId === 'live4') startSensorPolling(4);
  else stopSensorPolling(4);
```
Delete the `SENSOR_LABELS` constant, `_sensorIntervals`, `startSensorPolling`, `stopSensorPolling`, and `refreshSensors` functions (originally lines ~3421-3463) — these call the now-removed `GET /sensors/{inv}` endpoint.

- [ ] **Step 5: Verify no dangling references**

Run:
```bash
grep -n "view-live\|sensor-grid\|live3-ts\|live4-ts\|chart-live\|startSensorPolling\|stopSensorPolling\|refreshSensors\b\|/sensors/" "demo/Demo Dashboard.html"
```
Expected: no matches (the sensor-history **modal**'s endpoint is `/sensor-history/{inv}`, a different string — confirm it still appears and is untouched).

- [ ] **Step 6: Manual browser check**

Run: `uvicorn backend.main:app --port 8000`, open `http://localhost:8000/`, log in, confirm sidebar no longer shows "Tiempo real" under either invernadero, and no JS console errors on page load.

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): remove dead Tiempo real live-sensor tab"
```

---

## Task 7: Frontend — remove all delete UI (Historial row delete, Producción real "Borrar real")

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Produces: `#hist-confirm` panel, `#prod-delete-confirm` panel, `histDelete`/`histConfirm`, `prodDeleteKg`/`prodDeleteConfirm` all removed. `loadHistorial`'s per-row delete `<button>` column removed.

- [ ] **Step 1: Remove delete confirm panel markup**

Delete lines 1394-1400 (`<div class="ins-confirm-panel" id="hist-confirm" hidden>...</div>`) inside the old `ins-panel-hist` block — note this whole panel gets replaced wholesale in Task 9, so if Task 9 is done first this step is redundant; if done in this order, remove just the confirm-panel `<div>`, not the whole `ins-panel-hist` section yet.

Delete lines 1428-1434 (`<div class="ins-confirm-panel" id="prod-delete-confirm" hidden>...</div>`).

- [ ] **Step 2: Remove delete trigger button in `loadProdPredictions`**

In the `confirmed.map(p => ...)` template (around line 3198-3205), remove the `<button ... onclick="prodDeleteKg(...)">Borrar real</button>` line, leaving just the read-only summary row (predicted_for, kg_actual, kg_predicted).

- [ ] **Step 3: Remove delete trigger button in `loadHistorial`**

In the `rows.map(r => ...)` template (around line 3694-3709), remove the trailing `<td><button ... onclick="histDelete(...)">🗑</button></td>` cell and its column, and drop the corresponding `<th></th>` in the `hist-table` header (line ~1388: `<th></th>` after `Crec. (cm)`).

- [ ] **Step 4: Remove the JS functions**

Delete `prodDeleteKg` and `prodDeleteConfirm` (originally lines ~3638-3671, ending before `async function loadHistorial`).
Delete `histDelete` and `histConfirm` (originally lines ~3715-3748).

- [ ] **Step 5: Remove now-dead event listener wiring in `init()`**

Delete from `init()` (around lines 3393-3400):
```javascript
    document.getElementById('hist-confirm-btn').addEventListener('click', histConfirm);
    document.getElementById('hist-cancel').addEventListener('click', () => {
      document.getElementById('hist-confirm').hidden = true;
    });
    document.getElementById('prod-delete-confirm-btn').addEventListener('click', prodDeleteConfirm);
    document.getElementById('prod-delete-cancel').addEventListener('click', () => {
      document.getElementById('prod-delete-confirm').hidden = true;
    });
```
Also delete `document.getElementById('prod-delete-confirm').hidden = true;` inside `prodSubmit` (line 3237) — the element no longer exists.

- [ ] **Step 6: Verify no dangling references**

Run: `grep -n "histDelete\|histConfirm\|prodDeleteKg\|prodDeleteConfirm\|prod-delete-confirm\|hist-confirm" "demo/Demo Dashboard.html"`
Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): remove all per-row delete UI from Insertar datos"
```

---

## Task 8: Frontend — redesign Insertar datos: Sensores/Fenología/Producción upload tabs

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `POST /uploads/{inv}/{form_type}`, `GET /submission-lock/{inv}/{form_type}` (Task 3).
- Produces: 3 upload-tab UIs wired to those endpoints; replaces the old `ins-panel-pheno` (manual form) and `ins-panel-prod` (manual form) content. Old JS (`insSubmit`, `insConfirm`, `buildInsertForm`, `insAddRow`, `insRecompute`, `insBuildHeader`, `PHENO_KEY_TO_LIVE`, `prodSubmit`, `prodConfirm`, `loadProdPredictions`) is removed — the per-plant/per-week manual entry model no longer applies.

- [ ] **Step 1: Add SheetJS CDN script tag**

In the `<head>`, next to the existing Chart.js tag (line 10), add:
```html
<script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
```

- [ ] **Step 2: Replace the `ins-tabs` + 3 panel markup**

Replace the entire block from `<div class="ins-tabs">` through the end of `ins-panel-prod`'s closing `</div>` (originally lines 1339-1436, i.e. everything between the divider and `</section>`) with:

```html
    <div class="ins-tabs">
      <button class="ins-tab active" data-tab="sensores">Sensores</button>
      <button class="ins-tab" data-tab="fenologia">Fenología</button>
      <button class="ins-tab" data-tab="produccion">Producción</button>
      <button class="ins-tab" data-tab="hist">Historial</button>
    </div>

    <div class="panel insert-panel ins-panel" id="ins-panel-sensores">
      <div class="insert-toolbar">
        <input type="file" id="upload-sensores-file" accept=".xlsx" />
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="upload-sensores-btn" disabled>Enviar captura</button>
      </div>
      <div class="insert-hint" id="upload-sensores-lock-banner"></div>
      <div class="insert-hint" id="upload-sensores-preview"></div>
      <div class="ins-confirm-panel" id="upload-sensores-confirm1" hidden>
        <pre class="ins-confirm-summary" id="upload-sensores-confirm1-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-sensores-cancel1">Cancelar</button>
          <button class="btn-primary" id="upload-sensores-next1">Continuar</button>
        </div>
      </div>
      <div class="ins-confirm-panel" id="upload-sensores-confirm2" hidden>
        <pre class="ins-confirm-summary">Esta captura no podrá modificarse ni eliminarse.
Próximo envío disponible en 7 días.
¿Confirmar definitivamente?</pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-sensores-cancel2">Cancelar</button>
          <button class="btn-primary" id="upload-sensores-confirm-btn">Confirmar definitivamente</button>
        </div>
      </div>
      <div class="insert-result" id="upload-sensores-result"></div>
    </div>

    <div class="panel insert-panel ins-panel" id="ins-panel-fenologia" hidden>
      <div class="insert-toolbar">
        <input type="file" id="upload-fenologia-file" accept=".xlsx" />
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="upload-fenologia-btn" disabled>Enviar captura</button>
      </div>
      <div class="insert-hint" id="upload-fenologia-lock-banner"></div>
      <div class="insert-hint" id="upload-fenologia-preview"></div>
      <div class="ins-confirm-panel" id="upload-fenologia-confirm1" hidden>
        <pre class="ins-confirm-summary" id="upload-fenologia-confirm1-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-fenologia-cancel1">Cancelar</button>
          <button class="btn-primary" id="upload-fenologia-next1">Continuar</button>
        </div>
      </div>
      <div class="ins-confirm-panel" id="upload-fenologia-confirm2" hidden>
        <pre class="ins-confirm-summary">Esta captura no podrá modificarse ni eliminarse.
Próximo envío disponible en 7 días.
¿Confirmar definitivamente?</pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-fenologia-cancel2">Cancelar</button>
          <button class="btn-primary" id="upload-fenologia-confirm-btn">Confirmar definitivamente</button>
        </div>
      </div>
      <div class="insert-result" id="upload-fenologia-result"></div>
    </div>

    <div class="panel insert-panel ins-panel" id="ins-panel-produccion" hidden>
      <div class="insert-toolbar">
        <input type="file" id="upload-produccion-file" accept=".xlsx" />
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="upload-produccion-btn" disabled>Enviar captura</button>
      </div>
      <div class="insert-hint" id="upload-produccion-lock-banner"></div>
      <div class="insert-hint" id="upload-produccion-preview"></div>
      <div class="ins-confirm-panel" id="upload-produccion-confirm1" hidden>
        <pre class="ins-confirm-summary" id="upload-produccion-confirm1-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-produccion-cancel1">Cancelar</button>
          <button class="btn-primary" id="upload-produccion-next1">Continuar</button>
        </div>
      </div>
      <div class="ins-confirm-panel" id="upload-produccion-confirm2" hidden>
        <pre class="ins-confirm-summary">Esta captura no podrá modificarse ni eliminarse.
Próximo envío disponible en 7 días.
¿Confirmar definitivamente?</pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-produccion-cancel2">Cancelar</button>
          <button class="btn-primary" id="upload-produccion-confirm-btn">Confirmar definitivamente</button>
        </div>
      </div>
      <div class="insert-result" id="upload-produccion-result"></div>
    </div>

    <div class="panel insert-panel ins-panel" id="ins-panel-hist" hidden>
      <div class="ins-tabs" style="margin-bottom:12px">
        <button class="hist-subtab active" data-histtab="sensores">Sensores</button>
        <button class="hist-subtab" data-histtab="fenologia">Fenología</button>
        <button class="hist-subtab" data-histtab="produccion">Producción</button>
      </div>
      <div style="overflow-x:auto">
        <table class="insert-table" id="hist-table-sensores" style="min-width:900px"></table>
        <table class="insert-table" id="hist-table-fenologia" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-produccion" style="min-width:900px" hidden></table>
      </div>
      <div class="insert-result" id="hist-result"></div>
    </div>
```

- [ ] **Step 3: Add the generic upload-tab JS module**

Replace the entire old block from `function insBuildHeader` / `insAddRow` / `insRecompute` / `insSubmit` / `PHENO_KEY_TO_LIVE` / `insConfirm` / `buildInsertForm` / `initInsTabs` (originally lines ~2915-3170, i.e. everything between `renderPhenology` and `loadProdPredictions`) with:

```javascript
const UPLOAD_TYPES = {
  sensores:   { requiredCols: ['fecha','temp_prom_int','temp_min_int','temp_max_int','hr_prom_int','co2_ppm','riego_total','ph_promedio','ce_promedio','temp_prom_ext','temp_max_ext','temp_min_ext','rad_sum'] },
  fenologia:  { requiredCols: ['fecha','zona','planta','racimos_puestos','flores_racimo_abiertas','racimos_en_planta','cantidad_tomates','racimo_en_cosecha','tomates_maduros','diametro_fruto_cm','crecimiento_planta_cm'] },
  produccion: { requiredCols: ['fecha','kg_reales'] },
};

const uploadState = {}; // { [formType]: { file, rows, inv } }

function parseXlsxPreview(formType, file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const wb = XLSX.read(e.target.result, { type: 'array' });
        const sheet = wb.Sheets[wb.SheetNames[0]];
        const rows = XLSX.utils.sheet_to_json(sheet, { raw: false });
        resolve(rows);
      } catch (err) { reject(err); }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

async function refreshLockBanner(formType, inv) {
  const banner = document.getElementById(`upload-${formType}-lock-banner`);
  const btn = document.getElementById(`upload-${formType}-btn`);
  const token = localStorage.getItem('sb_token');
  try {
    const status = await fetch(`${API_BASE}/submission-lock/${inv}/${formType}`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => r.json());
    if (status.locked) {
      const date = new Date(status.next_allowed_at).toLocaleDateString('es-MX');
      banner.textContent = `Próximo envío disponible: ${date}`;
      banner.classList.add('bad');
      btn.disabled = true;
      document.getElementById(`upload-${formType}-file`).disabled = true;
    } else {
      banner.textContent = '';
      banner.classList.remove('bad');
      document.getElementById(`upload-${formType}-file`).disabled = false;
      btn.disabled = !uploadState[formType]?.rows?.length;
    }
  } catch (e) {
    banner.textContent = '';
  }
}

function wireUploadTab(formType) {
  const fileInput = document.getElementById(`upload-${formType}-file`);
  const btn = document.getElementById(`upload-${formType}-btn`);
  const preview = document.getElementById(`upload-${formType}-preview`);
  const result = document.getElementById(`upload-${formType}-result`);

  fileInput.addEventListener('change', async () => {
    result.textContent = ''; result.className = 'insert-result';
    preview.textContent = '';
    btn.disabled = true;
    const file = fileInput.files[0];
    if (!file) return;
    try {
      const rows = await parseXlsxPreview(formType, file);
      const required = UPLOAD_TYPES[formType].requiredCols;
      const missing = required.filter(c => !(rows[0] && c in rows[0]));
      if (missing.length) {
        preview.textContent = `⚠ Columnas faltantes: ${missing.join(', ')}`;
        preview.classList.add('bad');
        return;
      }
      preview.classList.remove('bad');
      const fechas = rows.map(r => r.fecha).filter(Boolean).sort();
      preview.textContent = `${rows.length} fila(s) detectada(s) · ${fechas[0]} → ${fechas[fechas.length - 1]}`;
      uploadState[formType] = { file, rows, inv: +document.getElementById('ins-inv').value };
      btn.disabled = false;
    } catch (e) {
      preview.textContent = '⚠ No se pudo leer el archivo: ' + e.message;
      preview.classList.add('bad');
    }
  });

  btn.addEventListener('click', () => {
    const state = uploadState[formType];
    if (!state) return;
    document.getElementById(`upload-${formType}-confirm1-summary`).textContent =
      `Invernadero ${state.inv} · ${state.rows.length} fila(s) en el archivo\nArchivo: ${state.file.name}`;
    document.getElementById(`upload-${formType}-confirm1`).hidden = false;
    btn.disabled = true;
  });

  document.getElementById(`upload-${formType}-cancel1`).addEventListener('click', () => {
    document.getElementById(`upload-${formType}-confirm1`).hidden = true;
    btn.disabled = false;
  });

  document.getElementById(`upload-${formType}-next1`).addEventListener('click', () => {
    document.getElementById(`upload-${formType}-confirm1`).hidden = true;
    document.getElementById(`upload-${formType}-confirm2`).hidden = false;
  });

  document.getElementById(`upload-${formType}-cancel2`).addEventListener('click', () => {
    document.getElementById(`upload-${formType}-confirm2`).hidden = true;
    btn.disabled = false;
  });

  document.getElementById(`upload-${formType}-confirm-btn`).addEventListener('click', async () => {
    const state = uploadState[formType];
    document.getElementById(`upload-${formType}-confirm2`).hidden = true;
    const token = localStorage.getItem('sb_token');
    const formData = new FormData();
    formData.append('file', state.file);
    try {
      const r = await fetch(`${API_BASE}/uploads/${state.inv}/${formType}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        if (r.status === 429) {
          throw new Error(`Ya se envió una captura esta semana. Próximo envío: ${new Date(j.detail?.next_allowed_at).toLocaleDateString('es-MX')}`);
        }
        throw new Error(j.detail || `Error ${r.status}`);
      }
      result.className = 'insert-result ok';
      result.textContent = `✓ ${j.rows_inserted + j.rows_updated} fila(s) guardadas` +
        (j.rows_skipped ? ` · ${j.rows_skipped} omitida(s): ${j.skipped_reasons.join('; ')}` : '');
      fileInput.value = '';
      preview.textContent = '';
      delete uploadState[formType];
      refreshLockBanner(formType, state.inv);
    } catch (e) {
      result.className = 'insert-result bad';
      result.textContent = '⚠ ' + e.message;
      btn.disabled = false;
    }
  });
}

function initInsTabs() {
  ['sensores', 'fenologia', 'produccion'].forEach(wireUploadTab);

  document.querySelectorAll('.ins-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ins-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      const inv = +document.getElementById('ins-inv').value;
      document.getElementById('ins-panel-sensores').hidden  = tab !== 'sensores';
      document.getElementById('ins-panel-fenologia').hidden = tab !== 'fenologia';
      document.getElementById('ins-panel-produccion').hidden = tab !== 'produccion';
      document.getElementById('ins-panel-hist').hidden = tab !== 'hist';
      if (tab === 'hist') loadHistorial(inv);
      else refreshLockBanner(tab, inv);
    });
  });

  document.getElementById('ins-inv').addEventListener('change', () => {
    const activeTab = document.querySelector('.ins-tab.active')?.dataset.tab;
    const inv = +document.getElementById('ins-inv').value;
    if (activeTab === 'hist') loadHistorial(inv);
    else if (activeTab) refreshLockBanner(activeTab, inv);
  });

  refreshLockBanner('sensores', +document.getElementById('ins-inv').value);
}
```

- [ ] **Step 4: Update `init()` wiring**

Replace (originally lines 3382-3400):
```javascript
    rawPheno[3] = ph3; rawPheno[4] = ph4;
    phenoFields = ph3.fields || [];
    renderPhenology(3); renderPhenology(4);
    buildInsertForm();
    initInsTabs();
    document.getElementById('prod-submit-btn').addEventListener('click', prodSubmit);
    document.getElementById('prod-confirm-btn').addEventListener('click', prodConfirm);
    document.getElementById('prod-cancel').addEventListener('click', () => {
      document.getElementById('prod-confirm').hidden = true;
      document.getElementById('prod-submit-btn').disabled = false;
    });
    document.getElementById('hist-confirm-btn').addEventListener('click', histConfirm);
    document.getElementById('hist-cancel').addEventListener('click', () => {
      document.getElementById('hist-confirm').hidden = true;
    });
    document.getElementById('prod-delete-confirm-btn').addEventListener('click', prodDeleteConfirm);
    document.getElementById('prod-delete-cancel').addEventListener('click', () => {
      document.getElementById('prod-delete-confirm').hidden = true;
    });
```
with:
```javascript
    rawPheno[3] = ph3; rawPheno[4] = ph4;
    phenoFields = ph3.fields || [];
    renderPhenology(3); renderPhenology(4);
    initInsTabs();
```

(Note: `phenoFields`/`renderPhenology` still feed the read-only KPI panel fenología cards on `view-kpi3`/`view-kpi4` — unrelated to Insertar datos, keep them. Verify with `grep -n "renderPhenology\|phenoFields" "demo/Demo Dashboard.html"` before removing anything beyond the exact block above.)

- [ ] **Step 5: Verify no dangling references to removed functions**

Run:
```bash
grep -n "insSubmit\|insConfirm\|buildInsertForm\|insAddRow\|insRecompute\|insBuildHeader\|PHENO_KEY_TO_LIVE\|prodSubmit\|prodConfirm\b\|loadProdPredictions\|ins-confirm\b\|prod-confirm\b\|prod-week-select\|prod-kg\b" "demo/Demo Dashboard.html"
```
Expected: no matches (all superseded by the generic upload-tab module).

- [ ] **Step 6: Manual browser check**

Run: `uvicorn backend.main:app --port 8000`, open dashboard, go to Insertar datos. For each of Sensores/Fenología/Producción: pick a small `.xlsx` with the right columns (build one ad hoc with `pandas.DataFrame(...).to_excel(...)` if no sample file handy), confirm preview shows row count, click through both confirm steps, confirm success message and that submitting again immediately shows the locked banner (backend 429).

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): replace manual forms with weekly excel upload tabs"
```

---

## Task 9: Frontend — Historial: 3 read-only sub-tables (Sensores/Fenología/Producción)

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `GET /uploads/{inv}/{form_type}/history` (Task 3).
- Produces: `loadHistorial(inv)` renders all 3 tables; `histDelete`/`histConfirm` stay removed (done in Task 7).

- [ ] **Step 1: Replace `loadHistorial` with a 3-table version**

Replace the old `async function loadHistorial(inv) { ... }` (already stripped of its delete button in Task 7, originally lines 3676-3713) with:

```javascript
const HIST_COLUMNS = {
  sensores: [
    ['fecha','Fecha'], ['temp_prom_int','Temp int.'], ['hr_prom_int','HR int.'],
    ['co2_ppm','CO2'], ['riego_total','Riego'], ['ph_promedio','pH'], ['ce_promedio','CE'],
  ],
  fenologia: [
    ['week_date','Semana'], ['zona','Zona'], ['planta','Planta'],
    ['racimos_puestos','Rac. puestos'], ['flores_racimo_abiertas','Flores'],
    ['racimos_en_planta','Rac. planta'], ['cantidad_tomates','Tomates'],
    ['racimo_en_cosecha','Rac. cosecha'], ['tomates_maduros','Maduros'],
    ['diametro_fruto_cm','Diám. (cm)'], ['crecimiento_planta_cm','Crec. (cm)'],
  ],
  produccion: [
    ['predicted_for','Semana'], ['kg_predicted','kg predicho'], ['kg_actual','kg real'],
  ],
};

async function loadHistTable(inv, formType) {
  const table = document.getElementById(`hist-table-${formType}`);
  const token = localStorage.getItem('sb_token');
  const cols = HIST_COLUMNS[formType];
  table.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:#9aa0a8">Cargando…</td></tr>`;
  try {
    const rows = await fetch(`${API_BASE}/uploads/${inv}/${formType}/history?limit=500`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

    if (!rows.length) {
      table.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:#9aa0a8">Sin datos registrados.</td></tr>`;
      return;
    }
    const fmt = v => (v == null ? '—' : v);
    table.innerHTML =
      `<thead><tr>${cols.map(([, label]) => `<th>${label}</th>`).join('')}</tr></thead>` +
      `<tbody>${rows.map(r => `<tr>${cols.map(([key]) => `<td>${fmt(r[key])}</td>`).join('')}</tr>`).join('')}</tbody>`;
  } catch (e) {
    table.innerHTML = `<tr><td colspan="${cols.length}" style="color:#dc2626">Error al cargar (${e.message})</td></tr>`;
  }
}

async function loadHistorial(inv) {
  const activeSub = document.querySelector('.hist-subtab.active')?.dataset.histtab || 'sensores';
  await loadHistTable(inv, activeSub);
}

function initHistSubtabs() {
  document.querySelectorAll('.hist-subtab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.hist-subtab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const sub = btn.dataset.histtab;
      ['sensores', 'fenologia', 'produccion'].forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
      const inv = +document.getElementById('ins-inv').value;
      loadHistTable(inv, sub);
    });
  });
}
```

- [ ] **Step 2: Wire `initHistSubtabs()` into `initInsTabs()`**

In the `initInsTabs` function from Task 8, add `initHistSubtabs();` as the first line of the function body.

- [ ] **Step 3: Verify no dangling references**

Run: `grep -n "hist-tbody\|HIST_COLUMNS\|loadHistTable\|initHistSubtabs" "demo/Demo Dashboard.html"`
Expected: `hist-tbody` no longer appears (old single-table id removed in Task 8's markup replacement); the three new functions appear exactly once each in their definitions plus their call sites.

- [ ] **Step 4: Manual browser check**

Run the server, go to Insertar datos → Historial. Confirm 3 sub-tabs (Sensores/Fenología/Producción) each load their respective table with no delete column, switching sub-tabs and switching invernadero both refetch correctly.

- [ ] **Step 5: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): Historial shows 3 read-only tables, one per data type"
```

---

## Task 10: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 9 pre-existing failures remain (the `.maybe_single()` mock mismatch flagged in Task 0, still explicitly out of scope for this plan — `test_ingest.py` itself is gone so that failure dropped out in Task 4), all other tests (including every new one added in Tasks 2 and 3) pass. If the count differs from 9, something in this plan's changes touched the pre-existing bug's blast radius — investigate before proceeding.

- [ ] **Step 2: Boot server and smoke-test every changed endpoint**

Run: `uvicorn backend.main:app --port 8000`. Using `curl` with a real bearer token (or the browser dev tools network tab while logged in):
- `GET /submission-lock/3/sensores` → 200 `{"locked": false, "next_allowed_at": null}` (on a fresh test greenhouse/type)
- `POST /uploads/3/sensores` with a valid small xlsx → 200, then immediately again → 429
- `GET /uploads/3/sensores/history` → 200, list includes the row just uploaded
- `GET /ingest/sensors`, `GET /sensors/3` → 404 (routes removed)
- `DELETE /phenology-live/3/1`, `DELETE /live-predictions/3/1/kg-actual` → 404 (routes removed)

- [ ] **Step 3: Full manual browser walkthrough**

Open the dashboard, walk through: Insertar datos → each of the 4 tabs for both Invernadero 3 and 4; confirm no "Tiempo real" sidebar entry; confirm no delete buttons anywhere in Insertar datos; confirm weekly lock banner appears after a successful upload and blocks a second upload of the same type/greenhouse.

- [ ] **Step 4: Final commit (if any cleanup was needed)**

```bash
git add -A
git commit -m "chore: final verification pass for excel-upload feature"
```
(Skip this commit if Step 1-3 found nothing to fix.)
