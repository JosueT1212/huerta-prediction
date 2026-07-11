# Split Sensores/Riego/Exteriores Uploads — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the shipped "Sensores" excel-upload into three types (Sensores/internas, Riego, Exteriores), fix each type's column list to match the actual trained model's 19 sensor features exactly, and fix the one downstream consumer (`scripts/live_inference.py`) that assumed the old single-table shape.

**Architecture:** Backend `backend/routers/uploads.py` gains 2 new form_types (`riego`, per-invernadero like sensores; `exteriores`, global with dedicated no-`{inv}` routes) backed by 2 new DB tables. Frontend gains 2 new upload tabs + 2 new Historial sub-tabs, reusing the existing generic upload-tab JS with a small global/per-invernadero branch. `scripts/live_inference.py` merges 3 table queries by `fecha` instead of reading one wide table.

**Tech Stack:** FastAPI, Supabase (postgres), pandas (existing deps, no new ones), vanilla JS (existing `demo/Demo Dashboard.html`, no build step).

## Global Constraints

- Authoritative column source: `Models/results/pipeline_inv3.pkl["sensor_cols"]` (verified by loading the pickle directly) — 19 features: 8 internas + 8 externas + 3 riego. Every column name below is copied verbatim from that list.
- **Sensores (internas)** — `fecha` + `temp_prom_int, temp_min_int, temp_max_int, hr_prom_int, co2_ppm, deficit_humedad, deficit_presion_vapor, humedad_abs_int`.
- **Riego** — `fecha` + `riego_total, ph_promedio, ce_promedio`.
- **Exteriores** — `fecha` + `temp_prom_ext, temp_max_ext, temp_min_ext, hr_prom_ext, rad_sum, rad_max, dh_ext, humedad_abs_ext`.
- Exteriores is **global** — no `greenhouse_id` column, no `{inv}` path segment on any of its endpoints. Sentinel `greenhouse_id = 0` used internally only for its `submission_locks` row (0 is never a real invernadero id — real ids are 3/4).
- Migrations are written but **never applied to live Supabase** by any task in this plan — that remains an explicit separate human action, per the prior excel-upload plan's precedent.
- Auth: every endpoint (existing and new) requires `Depends(get_current_user)`, matching the existing pattern in `backend/routers/uploads.py`.
- Follow the existing router/module patterns exactly — no framework changes, no new dependencies.

---

## Task 1: Migration `008_split_sensor_tables.sql`

**Files:**
- Create: `supabase/migrations/008_split_sensor_tables.sql`

**Interfaces:**
- Produces: `sensor_readings_wide` trimmed to internas-only (adds `deficit_humedad`, `deficit_presion_vapor`, `humedad_abs_int`; drops `temp_prom_ext`, `temp_max_ext`, `temp_min_ext`, `rad_sum`, `riego_total`, `ph_promedio`, `ce_promedio`). New table `riego_readings(greenhouse_id, fecha, riego_total, ph_promedio, ce_promedio)`, unique on `(greenhouse_id, fecha)`. New table `exterior_readings(fecha, temp_prom_ext, temp_max_ext, temp_min_ext, hr_prom_ext, rad_sum, rad_max, dh_ext, humedad_abs_ext)`, unique on `(fecha)` alone — **no `greenhouse_id` column**. `submission_locks.form_type` check constraint extended to include `'riego'` and `'exteriores'`. Consumed by Task 2/3 (`backend/routers/uploads.py`).

- [ ] **Step 1: Write the migration**

```sql
-- supabase/migrations/008_split_sensor_tables.sql

-- Sensores (internas) — trim sensor_readings_wide to internas-only, add the
-- 3 previously-missing columns, drop the 4 ext + 3 riego columns the
-- original 003_sensor_wide.sql incorrectly bundled in.
alter table sensor_readings_wide
  add column if not exists deficit_humedad float8,
  add column if not exists deficit_presion_vapor float8,
  add column if not exists humedad_abs_int float8,
  drop column if exists temp_prom_ext,
  drop column if exists temp_max_ext,
  drop column if exists temp_min_ext,
  drop column if exists rad_sum,
  drop column if exists riego_total,
  drop column if exists ph_promedio,
  drop column if exists ce_promedio;

-- Riego — new table, per-invernadero
create table if not exists riego_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  fecha         date        not null,
  riego_total   float8,
  ph_promedio   float8,
  ce_promedio   float8
);
create unique index if not exists riego_readings_gh_date
  on riego_readings (greenhouse_id, fecha);

-- Exteriores — new table, GLOBAL (no greenhouse_id column)
create table if not exists exterior_readings (
  id                bigserial primary key,
  fecha             date      not null,
  temp_prom_ext     float8,
  temp_max_ext      float8,
  temp_min_ext      float8,
  hr_prom_ext       float8,
  rad_sum           float8,
  rad_max           float8,
  dh_ext            float8,
  humedad_abs_ext   float8
);
create unique index if not exists exterior_readings_date
  on exterior_readings (fecha);

-- Extend submission_locks.form_type to cover the 2 new types
alter table submission_locks drop constraint if exists submission_locks_form_type_check;
alter table submission_locks add constraint submission_locks_form_type_check
  check (form_type in ('sensores','riego','exteriores','fenologia','produccion'));
```

- [ ] **Step 2: Do NOT apply to live Supabase**

Per the global constraints, this migration must not be run against the live project by this task. Note in your report that `supabase db push` (or manual SQL-editor application) is deferred to explicit human action.

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/008_split_sensor_tables.sql
git commit -m "feat(db): split sensor_readings_wide into sensores/riego/exteriores tables"
```

---

## Task 2: Backend — refactor uploads.py, fix Sensores columns, add Riego

**Files:**
- Modify: `backend/routers/uploads.py`
- Modify: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `backend.lock_utils.check_submission_lock`, `touch_submission_lock`, `get_lock_status` (unchanged, already in place).
- Produces:
  - `FORM_TYPES: dict[str, list[str]]` — now `{"sensores": SENSOR_COLS, "riego": RIEGO_COLS, "fenologia": PHENOLOGY_COLS, "produccion": PRODUCTION_COLS}` (exteriores added in Task 3).
  - `PER_INV_TYPES: set[str]` — `{"sensores", "riego", "fenologia", "produccion"}`. Consumed by Task 3 (exteriores is deliberately excluded).
  - `_parse_excel(contents: bytes, required_cols: list[str]) -> pd.DataFrame` — extracted helper (read + validate + NaN-clean). Consumed by Task 3.
  - `_ingest_rows(df: pd.DataFrame, form_type: str, greenhouse_id: int | None) -> tuple[int, int, int, list[str]]` — extracted helper (returns `rows_inserted, rows_updated, rows_skipped, skipped_reasons`). Consumed by Task 3 for the `exteriores` branch (already written here, `greenhouse_id=None` case).
  - `_table_for(form_type: str) -> str` — extended with `"riego": "riego_readings"`.

- [ ] **Step 1: Update the existing Sensores test to the new internas-only 8-column schema**

Replace (in `backend/tests/test_uploads.py`) the body of `test_upload_sensores_upserts_and_touches_lock`:

```python
def test_upload_sensores_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "deficit_humedad": 3.2, "deficit_presion_vapor": 0.8, "humedad_abs_int": 11.5,
    }])
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_in_file"] == 1
    assert body["rows_inserted"] == 1
    mock_supa.table.assert_any_call("sensor_readings_wide")
    mock_supa.table.assert_any_call("submission_locks")
```

And `test_upload_sensores_handles_blank_optional_cell`:

```python
def test_upload_sensores_handles_blank_optional_cell(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "deficit_humedad": 3.2, "deficit_presion_vapor": 0.8, "humedad_abs_int": None,
    }])
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1
```

- [ ] **Step 2: Add failing tests for Riego**

Append to `backend/tests/test_uploads.py`:

```python
def test_upload_riego_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
    }])
    files = {"file": ("riego.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/riego", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1
    mock_supa.table.assert_any_call("riego_readings")
    mock_supa.table.assert_any_call("submission_locks")


def test_riego_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"greenhouse_id": 3, "fecha": "2026-07-01", "riego_total": 120.0}
    ]
    r = api_client.get("/uploads/3/riego/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_riego_lock_status_returns_status(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/3/riego", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}
```

- [ ] **Step 3: Run tests to verify the new Riego tests fail**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: the 2 updated Sensores tests fail (still hitting the old 12-column `SENSOR_COLS`, so `deficit_humedad`/`deficit_presion_vapor`/`humedad_abs_int` are "missing columns" per the old code), and the 3 new Riego tests fail with 422 (`form_type inválido: riego`).

- [ ] **Step 4: Rewrite `backend/routers/uploads.py`**

Replace the entire file:

```python
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
    "co2_ppm", "deficit_humedad", "deficit_presion_vapor", "humedad_abs_int",
]
RIEGO_COLS = ["fecha", "riego_total", "ph_promedio", "ce_promedio"]
PHENOLOGY_COLS = [
    "fecha", "zona", "planta", "racimos_puestos", "flores_racimo_abiertas",
    "racimos_en_planta", "cantidad_tomates", "racimo_en_cosecha",
    "tomates_maduros", "diametro_fruto_cm", "crecimiento_planta_cm",
]
PRODUCTION_COLS = ["fecha", "kg_reales"]

FORM_TYPES = {
    "sensores": SENSOR_COLS,
    "riego": RIEGO_COLS,
    "fenologia": PHENOLOGY_COLS,
    "produccion": PRODUCTION_COLS,
}

PER_INV_TYPES = {"sensores", "riego", "fenologia", "produccion"}


def _require_form_type(form_type: str) -> list[str]:
    cols = FORM_TYPES.get(form_type)
    if cols is None:
        raise HTTPException(422, f"form_type inválido: {form_type}")
    return cols


def _table_for(form_type: str) -> str:
    return {
        "sensores": "sensor_readings_wide",
        "riego": "riego_readings",
        "fenologia": "phenology_observations",
        "produccion": "predictions",
    }[form_type]


def _date_str(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _parse_excel(contents: bytes, required_cols: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
    except Exception as e:
        raise HTTPException(422, f"No se pudo leer el archivo Excel: {e}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Columnas faltantes: {', '.join(missing)}")
    df = df[required_cols].dropna(how="all")
    df = df.where(pd.notna(df), None)
    return df


def _ingest_rows(df: pd.DataFrame, form_type: str, greenhouse_id: int | None):
    required_cols = FORM_TYPES[form_type]
    rows_inserted = rows_updated = rows_skipped = 0
    skipped_reasons: list[str] = []

    if form_type in ("sensores", "riego"):
        table = _table_for(form_type)
        for _, row in df.iterrows():
            record = {"greenhouse_id": greenhouse_id, "fecha": _date_str(row["fecha"])}
            record.update({c: row[c] for c in required_cols if c != "fecha"})
            service_client.table(table).upsert(
                record, on_conflict="greenhouse_id,fecha"
            ).execute()
            rows_inserted += 1

    elif form_type == "fenologia":
        for _, row in df.iterrows():
            try:
                zona = int(row["zona"])
                planta = int(row["planta"])
            except (ValueError, TypeError):
                raise HTTPException(
                    422,
                    f"Valores no numéricos en zona/planta para la fila con fecha "
                    f"{_date_str(row['fecha'])}: zona={row['zona']!r}, planta={row['planta']!r}",
                )
            record = {
                "greenhouse_id": greenhouse_id,
                "week_date": _date_str(row["fecha"]),
                "zona": zona,
                "planta": planta,
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
                .eq("greenhouse_id", greenhouse_id)
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

    return rows_inserted, rows_updated, rows_skipped, skipped_reasons


@router.post("/uploads/{inv}/{form_type}")
async def upload_excel(
    inv: int,
    form_type: str,
    file: UploadFile,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /uploads/{form_type}")
    required_cols = _require_form_type(form_type)
    check_submission_lock(inv, form_type)

    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, form_type, inv)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(inv, form_type)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/{inv}/{form_type}/history")
def upload_history(
    inv: int,
    form_type: str,
    limit: int = 200,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /uploads/{form_type}/history")
    _require_form_type(form_type)
    table = _table_for(form_type)
    order_col = {
        "sensores": "fecha", "riego": "fecha",
        "fenologia": "week_date", "produccion": "predicted_for",
    }[form_type]
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
    if form_type not in PER_INV_TYPES:
        raise HTTPException(422, f"{form_type} no usa invernadero; usa /submission-lock/{form_type}")
    _require_form_type(form_type)
    return get_lock_status(inv, form_type)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: all pass (updated Sensores tests + new Riego tests). `test_upload_rejects_invalid_form_type` (existing, posts to `/uploads/3/bogus`) should still pass — `bogus` is neither in `FORM_TYPES` nor `PER_INV_TYPES`, still 422.

- [ ] **Step 6: Run full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 9 pre-existing failures (documented `.maybe_single()` mock mismatch, unrelated, out of scope — see `docs/superpowers/plans/2026-07-09-insertar-datos-excel-upload.md` Task 0), all other tests including this task's pass.

- [ ] **Step 7: Commit**

```bash
git add backend/routers/uploads.py backend/tests/test_uploads.py
git commit -m "feat(backend): fix sensores column list, add riego upload type"
```

---

## Task 3: Backend — add global Exteriores upload

**Files:**
- Modify: `backend/routers/uploads.py`
- Modify: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `_parse_excel`, `_ingest_rows`, `FORM_TYPES`, `PER_INV_TYPES` (Task 2).
- Produces: `EXTERIOR_COLS` list; `FORM_TYPES["exteriores"]`; `EXTERIORES_GH_ID = 0` sentinel constant; `POST /uploads/exteriores`; `GET /uploads/exteriores/history`; `GET /submission-lock/exteriores`. `exteriores` is deliberately absent from `PER_INV_TYPES`.

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/test_uploads.py`:

```python
def test_upload_exteriores_requires_auth(api_client):
    df = pd.DataFrame([{"fecha": "2026-07-01", "temp_prom_ext": 20.0}])
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", files=files)
    assert r.status_code == 401


def test_upload_exteriores_upserts_without_greenhouse_id(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_ext": 20.0, "temp_max_ext": 26.0,
        "temp_min_ext": 15.0, "hr_prom_ext": 70.0, "rad_sum": 1800.0,
        "rad_max": 850.0, "dh_ext": 5.0, "humedad_abs_ext": 9.5,
    }])
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1
    mock_supa.table.assert_any_call("exterior_readings")
    # verify no greenhouse_id key was sent for the exterior_readings upsert
    upsert_call = next(
        c for c in mock_supa.table.return_value.upsert.call_args_list
    )
    assert "greenhouse_id" not in upsert_call[0][0]


def test_upload_exteriores_rejects_per_inv_route(api_client, mock_supa):
    headers = _auth(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01", "temp_prom_ext": 20.0}])
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/exteriores", headers=headers, files=files)
    assert r.status_code == 422


def test_exteriores_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"fecha": "2026-07-01", "temp_prom_ext": 20.0}
    ]
    r = api_client.get("/uploads/exteriores/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_exteriores_lock_status_uses_sentinel(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/exteriores", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}
    mock_supa.table.return_value.select.return_value.eq.assert_any_call("greenhouse_id", 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: FAIL — `/uploads/exteriores` route doesn't exist yet (404), `test_upload_exteriores_rejects_per_inv_route` currently would 422 via `_require_form_type` anyway (form_type not in `FORM_TYPES` yet) — confirm it still fails for the right reason (route/type doesn't exist) before moving on.

- [ ] **Step 3: Add Exteriores support to `backend/routers/uploads.py`**

Add near the top, after `PRODUCTION_COLS`:

```python
EXTERIOR_COLS = [
    "fecha", "temp_prom_ext", "temp_max_ext", "temp_min_ext", "hr_prom_ext",
    "rad_sum", "rad_max", "dh_ext", "humedad_abs_ext",
]
```

Add `"exteriores": EXTERIOR_COLS` to the `FORM_TYPES` dict (do **not** add to `PER_INV_TYPES`).

Add a sentinel constant right after `PER_INV_TYPES`:

```python
EXTERIORES_GH_ID = 0  # sentinel: exteriores has no real invernadero (0 is never a real id)
```

Add an `exteriores` branch to `_ingest_rows`, right before the `elif form_type == "fenologia":` line:

```python
    elif form_type == "exteriores":
        for _, row in df.iterrows():
            record = {"fecha": _date_str(row["fecha"])}
            record.update({c: row[c] for c in required_cols if c != "fecha"})
            service_client.table("exterior_readings").upsert(
                record, on_conflict="fecha"
            ).execute()
            rows_inserted += 1
```

Add 3 new routes at the end of the file:

```python
@router.post("/uploads/exteriores")
async def upload_exteriores(
    file: UploadFile,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    required_cols = FORM_TYPES["exteriores"]
    check_submission_lock(EXTERIORES_GH_ID, "exteriores")

    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, "exteriores", None)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(EXTERIORES_GH_ID, "exteriores")

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/exteriores/history")
def upload_history_exteriores(
    limit: int = 200,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("exterior_readings")
        .select("*")
        .order("fecha", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


@router.get("/submission-lock/exteriores")
def lock_status_exteriores(
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    return get_lock_status(EXTERIORES_GH_ID, "exteriores")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: all pass.

- [ ] **Step 5: Run full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 9 pre-existing failures (unchanged from Task 2), zero new failures.

- [ ] **Step 6: Commit**

```bash
git add backend/routers/uploads.py backend/tests/test_uploads.py
git commit -m "feat(backend): add global exteriores upload (no per-invernadero scope)"
```

---

## Task 4: Frontend — add Riego tab (per-invernadero)

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `POST /uploads/{inv}/riego`, `GET /uploads/{inv}/riego/history`, `GET /submission-lock/{inv}/riego` (Task 2).
- Produces: `PER_INV_UPLOAD_TYPES` array (new, replaces the 3 hardcoded `['sensores','fenologia','produccion']` lists scattered across `initInsTabs`); `UPLOAD_TYPES.riego`; `HIST_COLUMNS.riego`; `ins-panel-riego`; `hist-table-riego`. Consumed by Task 5 (which adds `GLOBAL_UPLOAD_TYPES`).

- [ ] **Step 1: Add the Riego tab button**

Find (grep `data-tab="sensores"` to confirm current location):
```html
    <div class="ins-tabs">
      <button class="ins-tab active" data-tab="sensores">Sensores</button>
      <button class="ins-tab" data-tab="fenologia">Fenología</button>
      <button class="ins-tab" data-tab="produccion">Producción</button>
      <button class="ins-tab" data-tab="hist">Historial</button>
    </div>
```
Replace with:
```html
    <div class="ins-tabs">
      <button class="ins-tab active" data-tab="sensores">Sensores</button>
      <button class="ins-tab" data-tab="riego">Riego</button>
      <button class="ins-tab" data-tab="fenologia">Fenología</button>
      <button class="ins-tab" data-tab="produccion">Producción</button>
      <button class="ins-tab" data-tab="hist">Historial</button>
    </div>
```

- [ ] **Step 2: Add the Riego panel markup**

Immediately after the closing `</div>` of `ins-panel-sensores` (grep `id="ins-panel-sensores"` to find it, its closing tag is right before the `<!-- next panel -->`-style comment or the `ins-panel-fenologia` opening tag), insert:

```html
    <div class="panel insert-panel ins-panel" id="ins-panel-riego" hidden>
      <div class="insert-toolbar">
        <input type="file" id="upload-riego-file" accept=".xlsx" />
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="upload-riego-btn" disabled>Enviar captura</button>
      </div>
      <div class="insert-hint" id="upload-riego-lock-banner"></div>
      <div class="insert-hint" id="upload-riego-preview"></div>
      <div class="ins-confirm-panel" id="upload-riego-confirm1" hidden>
        <pre class="ins-confirm-summary" id="upload-riego-confirm1-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-riego-cancel1">Cancelar</button>
          <button class="btn-primary" id="upload-riego-next1">Continuar</button>
        </div>
      </div>
      <div class="ins-confirm-panel" id="upload-riego-confirm2" hidden>
        <pre class="ins-confirm-summary">Esta captura no podrá modificarse ni eliminarse.
Próximo envío disponible en 7 días.
¿Confirmar definitivamente?</pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-riego-cancel2">Cancelar</button>
          <button class="btn-primary" id="upload-riego-confirm-btn">Confirmar definitivamente</button>
        </div>
      </div>
      <div class="insert-result" id="upload-riego-result"></div>
    </div>
```

- [ ] **Step 3: Add the Riego Historial sub-tab**

Find (grep `data-histtab="sensores"`):
```html
        <button class="hist-subtab active" data-histtab="sensores">Sensores</button>
        <button class="hist-subtab" data-histtab="fenologia">Fenología</button>
        <button class="hist-subtab" data-histtab="produccion">Producción</button>
```
Replace with:
```html
        <button class="hist-subtab active" data-histtab="sensores">Sensores</button>
        <button class="hist-subtab" data-histtab="riego">Riego</button>
        <button class="hist-subtab" data-histtab="fenologia">Fenología</button>
        <button class="hist-subtab" data-histtab="produccion">Producción</button>
```

Find (grep `id="hist-table-sensores"`):
```html
        <table class="insert-table" id="hist-table-sensores" style="min-width:900px"></table>
        <table class="insert-table" id="hist-table-fenologia" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-produccion" style="min-width:900px" hidden></table>
```
Replace with:
```html
        <table class="insert-table" id="hist-table-sensores" style="min-width:900px"></table>
        <table class="insert-table" id="hist-table-riego" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-fenologia" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-produccion" style="min-width:900px" hidden></table>
```

- [ ] **Step 4: Introduce `PER_INV_UPLOAD_TYPES` and add `riego` to `UPLOAD_TYPES`/`HIST_COLUMNS`**

Find (grep `const UPLOAD_TYPES = {`):
```javascript
const UPLOAD_TYPES = {
  sensores:   { requiredCols: ['fecha','temp_prom_int','temp_min_int','temp_max_int','hr_prom_int','co2_ppm','riego_total','ph_promedio','ce_promedio','temp_prom_ext','temp_max_ext','temp_min_ext','rad_sum'] },
  fenologia:  { requiredCols: ['fecha','zona','planta','racimos_puestos','flores_racimo_abiertas','racimos_en_planta','cantidad_tomates','racimo_en_cosecha','tomates_maduros','diametro_fruto_cm','crecimiento_planta_cm'] },
  produccion: { requiredCols: ['fecha','kg_reales'] },
};
```
Replace with (note the `sensores` column list is corrected to the 8 internas-only columns, matching the backend's new `SENSOR_COLS`):
```javascript
const PER_INV_UPLOAD_TYPES = ['sensores', 'riego', 'fenologia', 'produccion'];

const UPLOAD_TYPES = {
  sensores:   { requiredCols: ['fecha','temp_prom_int','temp_min_int','temp_max_int','hr_prom_int','co2_ppm','deficit_humedad','deficit_presion_vapor','humedad_abs_int'] },
  riego:      { requiredCols: ['fecha','riego_total','ph_promedio','ce_promedio'] },
  fenologia:  { requiredCols: ['fecha','zona','planta','racimos_puestos','flores_racimo_abiertas','racimos_en_planta','cantidad_tomates','racimo_en_cosecha','tomates_maduros','diametro_fruto_cm','crecimiento_planta_cm'] },
  produccion: { requiredCols: ['fecha','kg_reales'] },
};
```

Find (grep `const HIST_COLUMNS = {`):
```javascript
const HIST_COLUMNS = {
  sensores: [
    ['fecha','Fecha'], ['temp_prom_int','Temp int.'], ['hr_prom_int','HR int.'],
    ['co2_ppm','CO2'], ['riego_total','Riego'], ['ph_promedio','pH'], ['ce_promedio','CE'],
  ],
  fenologia: [
```
Replace the `sensores` entry (drop the riego-only columns it wrongly displayed) and add a `riego` entry right after it:
```javascript
const HIST_COLUMNS = {
  sensores: [
    ['fecha','Fecha'], ['temp_prom_int','Temp int.'], ['temp_min_int','Temp min.'],
    ['temp_max_int','Temp máx.'], ['hr_prom_int','HR int.'], ['co2_ppm','CO2'],
    ['deficit_humedad','Déficit hum.'], ['deficit_presion_vapor','Déficit vapor'],
    ['humedad_abs_int','Hum. absoluta'],
  ],
  riego: [
    ['fecha','Fecha'], ['riego_total','Riego total'], ['ph_promedio','pH'], ['ce_promedio','CE'],
  ],
  fenologia: [
```
(leave the rest of the object — `fenologia`/`produccion` entries — unchanged).

- [ ] **Step 5: Replace the 3 hardcoded `['sensores', 'fenologia', 'produccion']` lists with `PER_INV_UPLOAD_TYPES`**

In `wireUploadTab` registration (grep `].forEach(wireUploadTab)`):
```javascript
  ['sensores', 'fenologia', 'produccion'].forEach(wireUploadTab);
```
Replace with:
```javascript
  PER_INV_UPLOAD_TYPES.forEach(wireUploadTab);
```

In the tab-click panel-hide handler (grep `ins-panel-sensores').hidden`):
```javascript
      document.getElementById('ins-panel-sensores').hidden  = tab !== 'sensores';
      document.getElementById('ins-panel-fenologia').hidden = tab !== 'fenologia';
      document.getElementById('ins-panel-produccion').hidden = tab !== 'produccion';
      document.getElementById('ins-panel-hist').hidden = tab !== 'hist';
```
Replace with:
```javascript
      document.getElementById('ins-panel-sensores').hidden  = tab !== 'sensores';
      document.getElementById('ins-panel-riego').hidden     = tab !== 'riego';
      document.getElementById('ins-panel-fenologia').hidden = tab !== 'fenologia';
      document.getElementById('ins-panel-produccion').hidden = tab !== 'produccion';
      document.getElementById('ins-panel-hist').hidden = tab !== 'hist';
```

In the `#ins-inv` change-handler's staged-state-clearing loop (grep `Clear staged state for`):
```javascript
    ['sensores', 'fenologia', 'produccion'].forEach(formType => {
```
Replace with:
```javascript
    PER_INV_UPLOAD_TYPES.forEach(formType => {
```

- [ ] **Step 6: Add `riego` to the Historial sub-tab toggle list**

In `initHistSubtabs` (grep `['sensores', 'fenologia', 'produccion'].forEach(t =>`):
```javascript
      ['sensores', 'fenologia', 'produccion'].forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
```
Replace with:
```javascript
      [...PER_INV_UPLOAD_TYPES, 'exteriores'].forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
```

Note: `'exteriores'` is included here now (rather than left for Task 5) because this line becomes the single source of truth for "all Historial sub-tab ids" — Task 5 will add the `hist-table-exteriores` element and `HIST_COLUMNS.exteriores` entry that this list already anticipates. If Task 5 hasn't run yet, `document.getElementById('hist-table-exteriores')` on this line will return `null` and `.hidden = ...` on `null` throws — **verify Task 5 runs immediately after this task in the same session**, or temporarily leave this line as `[...PER_INV_UPLOAD_TYPES].forEach(...)` (without `'exteriores'`) and let Task 5 add it back. Prefer the latter (safer, task stays independently correct) — use:
```javascript
      PER_INV_UPLOAD_TYPES.forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
```

- [ ] **Step 7: Verify no leftover hardcoded per-inv-type lists**

Run: `grep -n "'sensores', 'fenologia', 'produccion'" "demo/Demo Dashboard.html"`
Expected: zero matches (all three replaced with `PER_INV_UPLOAD_TYPES`).

- [ ] **Step 8: No automated test suite — verify via `node --check` and structural checks**

Run: extract the `<script>` block and run `node --check` on it to confirm no syntax errors. Also run:
```bash
grep -n "ins-panel-riego\|upload-riego-\|hist-table-riego\|data-tab=\"riego\"\|data-histtab=\"riego\"" "demo/Demo Dashboard.html"
```
Expected: matches for the panel, all `upload-riego-*` element ids, the historial table, and both tab buttons.

- [ ] **Step 9: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): add Riego upload tab, fix Sensores column list"
```

---

## Task 5: Frontend — add Exteriores tab (global, no invernadero)

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `POST /uploads/exteriores`, `GET /uploads/exteriores/history`, `GET /submission-lock/exteriores` (Task 3); `PER_INV_UPLOAD_TYPES` (Task 4).
- Produces: `GLOBAL_UPLOAD_TYPES = ['exteriores']`; `UPLOAD_TYPES.exteriores` (with a `global: true` flag); `HIST_COLUMNS.exteriores`; `ins-panel-exteriores`; `hist-table-exteriores`. Modifies `wireUploadTab`, `refreshLockBanner`, `loadHistTable` to branch on `global`.

- [ ] **Step 1: Add the Exteriores tab button**

Find (from Task 4's Step 1 result):
```html
      <button class="ins-tab active" data-tab="sensores">Sensores</button>
      <button class="ins-tab" data-tab="riego">Riego</button>
      <button class="ins-tab" data-tab="fenologia">Fenología</button>
```
Replace with:
```html
      <button class="ins-tab active" data-tab="sensores">Sensores</button>
      <button class="ins-tab" data-tab="riego">Riego</button>
      <button class="ins-tab" data-tab="exteriores">Exteriores</button>
      <button class="ins-tab" data-tab="fenologia">Fenología</button>
```

- [ ] **Step 2: Add the Exteriores panel markup**

Immediately after the `ins-panel-riego` panel's closing `</div>` (added in Task 4), insert:

```html
    <div class="panel insert-panel ins-panel" id="ins-panel-exteriores" hidden>
      <div class="insert-hint" style="margin-bottom:8px">Aplica a ambos invernaderos (dato climático compartido).</div>
      <div class="insert-toolbar">
        <input type="file" id="upload-exteriores-file" accept=".xlsx" />
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="upload-exteriores-btn" disabled>Enviar captura</button>
      </div>
      <div class="insert-hint" id="upload-exteriores-lock-banner"></div>
      <div class="insert-hint" id="upload-exteriores-preview"></div>
      <div class="ins-confirm-panel" id="upload-exteriores-confirm1" hidden>
        <pre class="ins-confirm-summary" id="upload-exteriores-confirm1-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-exteriores-cancel1">Cancelar</button>
          <button class="btn-primary" id="upload-exteriores-next1">Continuar</button>
        </div>
      </div>
      <div class="ins-confirm-panel" id="upload-exteriores-confirm2" hidden>
        <pre class="ins-confirm-summary">Esta captura no podrá modificarse ni eliminarse.
Próximo envío disponible en 7 días.
¿Confirmar definitivamente?</pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="upload-exteriores-cancel2">Cancelar</button>
          <button class="btn-primary" id="upload-exteriores-confirm-btn">Confirmar definitivamente</button>
        </div>
      </div>
      <div class="insert-result" id="upload-exteriores-result"></div>
    </div>
```

- [ ] **Step 3: Add the Exteriores Historial sub-tab**

Find (from Task 4's Step 3 result):
```html
        <button class="hist-subtab active" data-histtab="sensores">Sensores</button>
        <button class="hist-subtab" data-histtab="riego">Riego</button>
        <button class="hist-subtab" data-histtab="fenologia">Fenología</button>
```
Replace with:
```html
        <button class="hist-subtab active" data-histtab="sensores">Sensores</button>
        <button class="hist-subtab" data-histtab="riego">Riego</button>
        <button class="hist-subtab" data-histtab="exteriores">Exteriores</button>
        <button class="hist-subtab" data-histtab="fenologia">Fenología</button>
```

Find (from Task 4's Step 3 result):
```html
        <table class="insert-table" id="hist-table-riego" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-fenologia" style="min-width:900px" hidden></table>
```
Replace with:
```html
        <table class="insert-table" id="hist-table-riego" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-exteriores" style="min-width:900px" hidden></table>
        <table class="insert-table" id="hist-table-fenologia" style="min-width:900px" hidden></table>
```

- [ ] **Step 4: Add `GLOBAL_UPLOAD_TYPES`, `UPLOAD_TYPES.exteriores`, `HIST_COLUMNS.exteriores`**

Find (from Task 4's Step 4 result):
```javascript
const PER_INV_UPLOAD_TYPES = ['sensores', 'riego', 'fenologia', 'produccion'];

const UPLOAD_TYPES = {
  sensores:   { requiredCols: ['fecha','temp_prom_int','temp_min_int','temp_max_int','hr_prom_int','co2_ppm','deficit_humedad','deficit_presion_vapor','humedad_abs_int'] },
  riego:      { requiredCols: ['fecha','riego_total','ph_promedio','ce_promedio'] },
  fenologia:  { requiredCols: ['fecha','zona','planta','racimos_puestos','flores_racimo_abiertas','racimos_en_planta','cantidad_tomates','racimo_en_cosecha','tomates_maduros','diametro_fruto_cm','crecimiento_planta_cm'] },
  produccion: { requiredCols: ['fecha','kg_reales'] },
};
```
Replace with:
```javascript
const PER_INV_UPLOAD_TYPES = ['sensores', 'riego', 'fenologia', 'produccion'];
const GLOBAL_UPLOAD_TYPES = ['exteriores'];

const UPLOAD_TYPES = {
  sensores:   { requiredCols: ['fecha','temp_prom_int','temp_min_int','temp_max_int','hr_prom_int','co2_ppm','deficit_humedad','deficit_presion_vapor','humedad_abs_int'] },
  riego:      { requiredCols: ['fecha','riego_total','ph_promedio','ce_promedio'] },
  exteriores: { requiredCols: ['fecha','temp_prom_ext','temp_max_ext','temp_min_ext','hr_prom_ext','rad_sum','rad_max','dh_ext','humedad_abs_ext'], global: true },
  fenologia:  { requiredCols: ['fecha','zona','planta','racimos_puestos','flores_racimo_abiertas','racimos_en_planta','cantidad_tomates','racimo_en_cosecha','tomates_maduros','diametro_fruto_cm','crecimiento_planta_cm'] },
  produccion: { requiredCols: ['fecha','kg_reales'] },
};
```

Find (from Task 4's Step 4 result, the `riego:` entry inside `HIST_COLUMNS`):
```javascript
  riego: [
    ['fecha','Fecha'], ['riego_total','Riego total'], ['ph_promedio','pH'], ['ce_promedio','CE'],
  ],
  fenologia: [
```
Replace with:
```javascript
  riego: [
    ['fecha','Fecha'], ['riego_total','Riego total'], ['ph_promedio','pH'], ['ce_promedio','CE'],
  ],
  exteriores: [
    ['fecha','Fecha'], ['temp_prom_ext','Temp ext.'], ['temp_max_ext','Temp máx.'],
    ['temp_min_ext','Temp mín.'], ['hr_prom_ext','HR ext.'], ['rad_sum','Radiación (suma)'],
    ['rad_max','Radiación (máx)'], ['dh_ext','Déficit hum.'], ['humedad_abs_ext','Hum. absoluta'],
  ],
  fenologia: [
```

- [ ] **Step 5: Branch `wireUploadTab` on `global`**

Find (grep `function wireUploadTab(formType) {`) and within it, the file-change handler's state assignment:
```javascript
      uploadState[formType] = { file, rows, inv: +document.getElementById('ins-inv').value };
```
Replace with:
```javascript
      const isGlobal = !!UPLOAD_TYPES[formType].global;
      uploadState[formType] = { file, rows, inv: isGlobal ? null : +document.getElementById('ins-inv').value };
```

Find the click handler that builds the confirm-1 summary:
```javascript
  btn.addEventListener('click', () => {
    const state = uploadState[formType];
    if (!state) return;
    document.getElementById(`upload-${formType}-confirm1-summary`).textContent =
      `Invernadero ${state.inv} · ${state.rows.length} fila(s) en el archivo\nArchivo: ${state.file.name}`;
    document.getElementById(`upload-${formType}-confirm1`).hidden = false;
    btn.disabled = true;
  });
```
Replace with:
```javascript
  btn.addEventListener('click', () => {
    const state = uploadState[formType];
    if (!state) return;
    const invLabel = state.inv == null ? '' : `Invernadero ${state.inv} · `;
    document.getElementById(`upload-${formType}-confirm1-summary`).textContent =
      `${invLabel}${state.rows.length} fila(s) en el archivo\nArchivo: ${state.file.name}`;
    document.getElementById(`upload-${formType}-confirm1`).hidden = false;
    btn.disabled = true;
  });
```

Find the final-confirm POST call:
```javascript
      const r = await fetch(`${API_BASE}/uploads/${state.inv}/${formType}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });
```
Replace with:
```javascript
      const uploadUrl = state.inv == null
        ? `${API_BASE}/uploads/${formType}`
        : `${API_BASE}/uploads/${state.inv}/${formType}`;
      const r = await fetch(uploadUrl, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });
```

No edit needed for the success-path lock-banner refresh call
(`refreshLockBanner(formType, state.inv);`, right after `delete
uploadState[formType];`) — it already passes `state.inv`, which is `null`
for `exteriores` per Step 5's file-change handler update. `refreshLockBanner`
itself is fixed to handle `inv == null` in the next step.

- [ ] **Step 6: Branch `refreshLockBanner` on `inv == null`**

Find:
```javascript
async function refreshLockBanner(formType, inv) {
  const banner = document.getElementById(`upload-${formType}-lock-banner`);
  const btn = document.getElementById(`upload-${formType}-btn`);
  const token = localStorage.getItem('sb_token');
  try {
    const status = await fetch(`${API_BASE}/submission-lock/${inv}/${formType}`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => r.json());
```
Replace with:
```javascript
async function refreshLockBanner(formType, inv) {
  const banner = document.getElementById(`upload-${formType}-lock-banner`);
  const btn = document.getElementById(`upload-${formType}-btn`);
  const token = localStorage.getItem('sb_token');
  const lockUrl = inv == null
    ? `${API_BASE}/submission-lock/${formType}`
    : `${API_BASE}/submission-lock/${inv}/${formType}`;
  try {
    const status = await fetch(lockUrl, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => r.json());
```

- [ ] **Step 7: Wire the Exteriores tab's initial lock-banner load and panel toggle**

Find (Task 4's panel-hide block):
```javascript
      document.getElementById('ins-panel-sensores').hidden  = tab !== 'sensores';
      document.getElementById('ins-panel-riego').hidden     = tab !== 'riego';
      document.getElementById('ins-panel-fenologia').hidden = tab !== 'fenologia';
```
Replace with:
```javascript
      document.getElementById('ins-panel-sensores').hidden  = tab !== 'sensores';
      document.getElementById('ins-panel-riego').hidden     = tab !== 'riego';
      document.getElementById('ins-panel-exteriores').hidden = tab !== 'exteriores';
      document.getElementById('ins-panel-fenologia').hidden = tab !== 'fenologia';
```

Find the same handler's tail:
```javascript
      if (tab === 'hist') loadHistorial(inv);
      else refreshLockBanner(tab, inv);
```
This already works unmodified for `exteriores` since `wireUploadTab`'s registration (next step) will call `refreshLockBanner('exteriores', null)` appropriately via the `PER_INV_UPLOAD_TYPES`/`GLOBAL_UPLOAD_TYPES` split — but this specific line passes `inv` (the current invernadero, never `null`) regardless of tab. Replace with:
```javascript
      if (tab === 'hist') loadHistorial(inv);
      else refreshLockBanner(tab, GLOBAL_UPLOAD_TYPES.includes(tab) ? null : inv);
```

Find the `wireUploadTab` registration line (from Task 4):
```javascript
  PER_INV_UPLOAD_TYPES.forEach(wireUploadTab);
```
Replace with:
```javascript
  [...PER_INV_UPLOAD_TYPES, ...GLOBAL_UPLOAD_TYPES].forEach(wireUploadTab);
```

There is also a second, separate call site with the identical bug, inside the
`#ins-inv` change handler (not the tab-click handler edited above). Find:
```javascript
    if (activeTab === 'hist') loadHistorial(inv);
    else if (activeTab) refreshLockBanner(activeTab, inv);
  });

  refreshLockBanner('sensores', +document.getElementById('ins-inv').value);
```
Replace with:
```javascript
    if (activeTab === 'hist') loadHistorial(inv);
    else if (activeTab) refreshLockBanner(activeTab, GLOBAL_UPLOAD_TYPES.includes(activeTab) ? null : inv);
  });

  refreshLockBanner('sensores', +document.getElementById('ins-inv').value);
```
(the trailing `refreshLockBanner('sensores', ...)` priming call is unchanged —
it only ever primes the `sensores` tab, which is per-invernadero and starts
`.active` by default, so it's unaffected by this fix.)

- [ ] **Step 8: Add `exteriores` to the `#ins-inv` change-handler's `PER_INV_UPLOAD_TYPES` iteration — no change needed**

The `#ins-inv` change handler (Task 4, Step 5) iterates `PER_INV_UPLOAD_TYPES`, which deliberately excludes `exteriores` — correct, since exteriores' staged upload has no invernadero to desync from. No edit required here; this step exists only to confirm that omission is intentional, not a gap.

- [ ] **Step 9: Restore the full Historial sub-tab toggle list**

Find (Task 4, Step 6 — the safer variant that used only `PER_INV_UPLOAD_TYPES`):
```javascript
      PER_INV_UPLOAD_TYPES.forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
```
Replace with:
```javascript
      [...PER_INV_UPLOAD_TYPES, ...GLOBAL_UPLOAD_TYPES].forEach(t =>
        document.getElementById(`hist-table-${t}`).hidden = t !== sub
      );
```

- [ ] **Step 10: Branch `loadHistTable`'s fetch URL on global types**

Find:
```javascript
async function loadHistTable(inv, formType) {
  const table = document.getElementById(`hist-table-${formType}`);
  const token = localStorage.getItem('sb_token');
  const cols = HIST_COLUMNS[formType];
  table.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:#9aa0a8">Cargando…</td></tr>`;
  try {
    const rows = await fetch(`${API_BASE}/uploads/${inv}/${formType}/history?limit=500`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
```
Replace with:
```javascript
async function loadHistTable(inv, formType) {
  const table = document.getElementById(`hist-table-${formType}`);
  const token = localStorage.getItem('sb_token');
  const cols = HIST_COLUMNS[formType];
  table.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:#9aa0a8">Cargando…</td></tr>`;
  const historyUrl = GLOBAL_UPLOAD_TYPES.includes(formType)
    ? `${API_BASE}/uploads/${formType}/history?limit=500`
    : `${API_BASE}/uploads/${inv}/${formType}/history?limit=500`;
  try {
    const rows = await fetch(historyUrl, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
```

- [ ] **Step 11: Verify**

Run:
```bash
grep -n "ins-panel-exteriores\|upload-exteriores-\|hist-table-exteriores\|data-tab=\"exteriores\"\|data-histtab=\"exteriores\"\|GLOBAL_UPLOAD_TYPES" "demo/Demo Dashboard.html"
```
Expected: matches for the panel, all `upload-exteriores-*` ids, the historial table, both tab buttons, and every `GLOBAL_UPLOAD_TYPES` usage site (definition + the 4 call sites edited in Steps 6/7/9/10).

Extract the `<script>` block and run `node --check` to confirm no syntax errors.

- [ ] **Step 12: Manual browser check (if available) or note the gap**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_KEY=test SUPABASE_ANON_KEY=test uvicorn backend.main:app --port 8000` from repo root, open `/app`, go to Insertar datos, confirm 6 tabs (Sensores/Riego/Exteriores/Fenología/Producción/Historial), confirm Exteriores shows the "aplica a ambos invernaderos" note and its lock banner doesn't change when switching `#ins-inv`. If no browser is available in your environment, state that clearly in your report — this is a known, previously-accepted verification gap for this codebase (see Task 10 of the prior excel-upload plan).

- [ ] **Step 13: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(frontend): add global Exteriores upload tab"
```

---

## Task 6: Fix `scripts/live_inference.py` to merge 3 tables

**Files:**
- Modify: `scripts/live_inference.py`

**Interfaces:**
- Consumes: `sensor_readings_wide` (internas, per-inv), `riego_readings` (new, per-inv, Task 1), `exterior_readings` (new, global, Task 1).
- Produces: `wide_rows: list[dict]` — same shape/contract `build_input_tensor` (`backend/live_features.py`, unchanged) already expects: each dict has `fecha` plus whichever of `pipeline["sensor_cols"]` keys are present; missing keys are tolerated (filled `NaN` by `aggregate_wide_to_weekly`).

- [ ] **Step 1: Replace the single wide-table query with 3 merged queries**

Find (grep `sensor_readings_wide` in `scripts/live_inference.py`):
```python
    wide_resp = (
        supa.table("sensor_readings_wide")
        .select("*")
        .eq("greenhouse_id", INV_ID)
        .gte("fecha", cutoff)
        .execute()
    )
    wide_rows = wide_resp.data or []
    print(f"  Wide sensor rows pulled: {len(wide_rows)}")
```
Replace with:
```python
    sensor_resp = (
        supa.table("sensor_readings_wide")
        .select("*")
        .eq("greenhouse_id", INV_ID)
        .gte("fecha", cutoff)
        .execute()
    )
    riego_resp = (
        supa.table("riego_readings")
        .select("*")
        .eq("greenhouse_id", INV_ID)
        .gte("fecha", cutoff)
        .execute()
    )
    # exterior_readings is global (no greenhouse_id column) — same weather feeds both invernaderos
    ext_resp = (
        supa.table("exterior_readings")
        .select("*")
        .gte("fecha", cutoff)
        .execute()
    )

    merged_by_fecha: dict[str, dict] = {}
    for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
        merged_by_fecha.setdefault(row["fecha"], {}).update(row)
    wide_rows = list(merged_by_fecha.values())
    print(f"  Sensor rows pulled: {len(sensor_resp.data or [])} sensores + "
          f"{len(riego_resp.data or [])} riego + {len(ext_resp.data or [])} exteriores "
          f"→ {len(wide_rows)} merged by fecha")
```

- [ ] **Step 2: Verify with a dry-run against a local/mocked check**

There's no existing automated test for `scripts/live_inference.py` (it's a cron entry point, run manually per its docstring). Verify by:
```bash
cd /path/to/repo && python3 -c "
import ast
with open('scripts/live_inference.py') as f:
    ast.parse(f.read())
print('syntax OK')
"
```
This confirms the script still parses correctly. A full dry-run (`DRY_RUN=1 python scripts/live_inference.py`) requires live Supabase credentials and `TRANSPLANT_DATE_INV3` — not available in this environment; note this in your report as a gap consistent with the rest of this codebase's manual-verification-only cron scripts.

- [ ] **Step 3: Commit**

```bash
git add scripts/live_inference.py
git commit -m "fix(scripts): merge sensor_readings_wide/riego_readings/exterior_readings by fecha"
```

---

## Task 7: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: 9 pre-existing failures (documented, unrelated `.maybe_single()` mock mismatch — unchanged from before this plan), all other tests pass including every new test added across Tasks 2 and 3.

- [ ] **Step 2: Boot server and smoke-test every new/changed endpoint**

Run: `uvicorn backend.main:app --port 8000` (dummy `SUPABASE_URL`/`SUPABASE_SERVICE_KEY`/`SUPABASE_ANON_KEY` env vars are fine for a route-existence/auth check, per this codebase's established pattern).
```bash
curl -s -o /dev/null -w "POST /uploads/3/riego (no auth) -> %{http_code}\n" -X POST http://localhost:8000/uploads/3/riego
curl -s -o /dev/null -w "POST /uploads/exteriores (no auth) -> %{http_code}\n" -X POST http://localhost:8000/uploads/exteriores
curl -s -o /dev/null -w "GET /uploads/exteriores/history (no auth) -> %{http_code}\n" http://localhost:8000/uploads/exteriores/history
curl -s -o /dev/null -w "GET /submission-lock/exteriores (no auth) -> %{http_code}\n" http://localhost:8000/submission-lock/exteriores
curl -s -o /dev/null -w "POST /uploads/3/exteriores (should reject, wrong route) -> %{http_code}\n" -X POST http://localhost:8000/uploads/3/exteriores
```
Expected: all `401` except the last, which should be `422` (exteriores rejected on the per-inv route) or `401` if auth is checked before the form_type validation in that route's dependency order — confirm which and note it, either is acceptable since both reject the request.

- [ ] **Step 3: Confirm served HTML contains all new UI pieces**

```bash
curl -s http://localhost:8000/app | grep -o 'ins-panel-riego\|ins-panel-exteriores\|hist-table-riego\|hist-table-exteriores\|data-tab="riego"\|data-tab="exteriores"' | sort -u
```
Expected: all 6 strings present.

- [ ] **Step 4: Stop the test server**

```bash
pkill -f "uvicorn backend.main:app --port 8000"
```

- [ ] **Step 5: Manual browser walkthrough (if available) or documented gap**

Open Insertar datos, walk through all 6 tabs for both Invernadero 3 and 4, confirm Riego and Exteriores upload flows work end to end with a real small `.xlsx`, confirm Exteriores' lock/history don't vary with the `#ins-inv` selector. If no browser is available, state this plainly — consistent with the prior plan's precedent, this is a known residual gap for the user to close.

- [ ] **Step 6: Final commit (only if verification found something to fix)**

```bash
git add -A
git commit -m "chore: final verification pass for split sensor uploads"
```
Skip if nothing needed fixing.
