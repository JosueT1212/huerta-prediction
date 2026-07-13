# First-Submission Window Coverage Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject sensores/riego/exteriores/fenología uploads at the API layer that don't supply enough real rows for `build_input_tensor` to fill a real (non-zero-padded) prediction window — both for a scope's first submission and for every submission after it. Fenología gets the same harness as sensores/riego despite having a historical-mean fallback, per explicit user instruction (2026-07-13) to apply it uniformly.

**Architecture:** One new function, `_check_window_coverage`, added to `backend/routers/uploads.py`. It runs an existence query per scope (`greenhouse_id`+table, or table-wide for exteriores) on the scope's DB date column (`fecha` for sensores/riego/exteriores, `week_date` for fenología) to decide first-vs-subsequent, then either counts distinct ISO-weeks in the uploaded file (first) or counts new distinct dates not already in the DB (subsequent), raising `HTTPException(422, ...)` before any row is written if coverage is short. Called from `upload_excel` and `upload_exteriores`, right after `_parse_excel` and before `_ingest_rows`.

**Tech Stack:** FastAPI, pandas, supabase-py (`service_client`), pytest with `unittest.mock.MagicMock` (existing `mock_supa` fixture in `backend/tests/conftest.py`).

## Global Constraints

- Scope: sensores, riego, exteriores, fenología. Producción is explicitly out of scope (`_check_window_coverage` must no-op for it — not a model input).
- Fenología's DB date column is `week_date` (`phenology_observations.week_date`), not `fecha` — the existence/subsequent queries for that scope must filter/select on `week_date`. The uploaded file's own column is still `fecha` in all cases (`_parse_excel` output).
- Fenología uses the same per-invernadero thresholds as sensores/riego (`MIN_WEEKS_FIRST_BY_INV`, `MIN_NEW_DAYS_SUBSEQUENT`) — no separate constant.
- `HORIZON = 5`, `SEQ_LEN_BY_INV = {3: 4, 4: 2}` — copied constants, not runtime-loaded, per spec §1.
- `MIN_WEEKS_FIRST_BY_INV = {3: 9, 4: 7}` (`seq_len + HORIZON` per invernadero).
- `MIN_WEEKS_FIRST_EXTERIORES = 9` (`max()` across invernaderos).
- `MIN_NEW_DAYS_SUBSEQUENT = 7`.
- First-submission rejection message (exact, spec §2):
  `"Not sufficient data for first submission: se requieren al menos {required} semanas de datos, el archivo cubre {n} semana(s)."`
- Subsequent-submission rejection message (exact, spec §3):
  `"Not sufficient data: se requieren al menos 7 días nuevos de datos, el archivo aporta {n} día(s) nuevo(s)."`
- Check order in the request: auth → form_type validity → submission lock → column validation (`_parse_excel`) → **window-coverage check (new)** → ingest → touch lock. No reordering of the existing lock/parse calls.
- No contiguity requirement on subsequent submissions — plain new-distinct-day count (spec §3, YAGNI).
- Unknown `greenhouse_id` (not in `SEQ_LEN_BY_INV`) must not 500. The route doesn't validate `inv` today and this spec doesn't add that validation (out of scope) — `_check_window_coverage` must fail open (skip the first-submission check) for an unmapped `greenhouse_id`, preserving today's permissive behavior.
- Zero new frontend code — rejections surface through the existing generic `⚠ ' + e.message` error path (spec §5).

---

### Task 1: Add `_check_window_coverage` and wire it into both upload endpoints

**Files:**
- Modify: `backend/routers/uploads.py`
- Test: `backend/tests/test_uploads.py`

**Interfaces:**
- Produces: `_check_window_coverage(form_type: str, greenhouse_id: int | None, table: str, date_col: str, df: pd.DataFrame) -> None`, raising `HTTPException(422, str)` on insufficient coverage, returning `None` (no-op) for `form_type` outside `("sensores", "riego", "exteriores", "fenologia")` or for an unmapped `greenhouse_id` on the first-submission path.
- Produces: `_date_col_for(form_type: str) -> str` — `"week_date"` for `fenologia`, `"fecha"` otherwise.
- Consumes: `service_client` (`backend.supabase_client.service_client`, already imported in `uploads.py`), `_date_str` (`backend/routers/uploads.py:56`), `_table_for` (`backend/routers/uploads.py:47`), `HTTPException` (already imported).

- [ ] **Step 1: Write the failing test — sensores first-submission rejects short coverage**

Add to `backend/tests/test_uploads.py`, after the existing `_no_lock` helper (after line 29):

```python
def _weekly_dates(n: int, start: str = "2026-03-02") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, periods=n, freq="7D")]


def _sensor_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "deficit_humedad": 3.2, "deficit_presion_vapor": 0.8, "humedad_abs_int": 11.5,
    } for d in _weekly_dates(n, start)])


def _riego_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
    } for d in _weekly_dates(n, start)])


def _exterior_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "temp_prom_ext": 20.0, "temp_max_ext": 26.0, "temp_min_ext": 15.0,
        "hr_prom_ext": 70.0, "rad_sum": 1800.0, "rad_max": 850.0, "dh_ext": 5.0,
        "humedad_abs_ext": 9.5,
    } for d in _weekly_dates(n, start)])


def _phenology_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "zona": 1, "planta": 1, "racimos_puestos": 1,
        "flores_racimo_abiertas": 1, "racimos_en_planta": 1, "cantidad_tomates": 1,
        "racimo_en_cosecha": 1, "tomates_maduros": 1, "diametro_fruto_cm": 1.0,
        "crecimiento_planta_cm": 1.0,
    } for d in _weekly_dates(n, start)])


def _first_submission(mock_supa, per_inv: bool = True):
    if per_inv:
        mock_supa.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    else:
        mock_supa.table.return_value.select.return_value.limit.return_value.execute.return_value.data = []


def _prior_submission(mock_supa, per_inv: bool = True):
    existing_row = [{"fecha": "2020-01-01"}]
    if per_inv:
        mock_supa.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = existing_row
    else:
        mock_supa.table.return_value.select.return_value.limit.return_value.execute.return_value.data = existing_row


def _matched_dates(mock_supa, dates: list[str], per_inv: bool = True, date_col: str = "fecha"):
    data = [{date_col: d} for d in dates]
    if per_inv:
        mock_supa.table.return_value.select.return_value.in_.return_value.eq.return_value.execute.return_value.data = data
    else:
        mock_supa.table.return_value.select.return_value.in_.return_value.execute.return_value.data = data


def test_upload_sensores_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(3)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_sensores_first_submission_rejects_insufficient_weeks -v`
Expected: FAIL — response is `200`, not `422` (no window-coverage check exists yet).

- [ ] **Step 3: Add constants to `backend/routers/uploads.py`**

Insert after `EXTERIORES_GH_ID = 0  # sentinel: exteriores has no real invernadero (0 is never a real id)` (line 37), before `def _require_form_type`:

```python
# Sourced from Models/hp_inv3.yaml / Models/hp_inv4.yaml (seq_len) and
# Models/cnn_rnn_yield.py (HORIZON = 5) — copied as constants here rather
# than loaded at runtime, to keep the upload path free of the
# torch/joblib/pipeline-loading dependency chain that backend/engine.py and
# scripts/live_inference.py carry.
HORIZON = 5
SEQ_LEN_BY_INV = {3: 4, 4: 2}
MIN_WEEKS_FIRST_BY_INV = {inv: SEQ_LEN_BY_INV[inv] + HORIZON for inv in SEQ_LEN_BY_INV}
MIN_WEEKS_FIRST_EXTERIORES = max(MIN_WEEKS_FIRST_BY_INV.values())
MIN_NEW_DAYS_SUBSEQUENT = 7
```

- [ ] **Step 4: Add `_check_window_coverage` to `backend/routers/uploads.py`**

Insert after `_ingest_rows` (after line 139, before the `@router.post("/uploads/{inv}/{form_type}")` decorator on line 142):

```python
def _date_col_for(form_type: str) -> str:
    return "week_date" if form_type == "fenologia" else "fecha"


def _check_window_coverage(
    form_type: str, greenhouse_id: int | None, table: str, date_col: str, df: pd.DataFrame
) -> None:
    if form_type not in ("sensores", "riego", "exteriores", "fenologia"):
        return

    existence_query = service_client.table(table).select(date_col)
    if greenhouse_id is not None:
        existence_query = existence_query.eq("greenhouse_id", greenhouse_id)
    existing = existence_query.limit(1).execute()

    dates = pd.to_datetime(df["fecha"])

    if not existing.data:
        if form_type == "exteriores":
            required = MIN_WEEKS_FIRST_EXTERIORES
        else:
            required = MIN_WEEKS_FIRST_BY_INV.get(greenhouse_id)
            if required is None:
                return
        iso = dates.dt.isocalendar()
        n_weeks = iso[["year", "week"]].drop_duplicates().shape[0]
        if n_weeks < required:
            raise HTTPException(
                422,
                f"Not sufficient data for first submission: se requieren al menos "
                f"{required} semanas de datos, el archivo cubre {n_weeks} semana(s).",
            )
        return

    date_strs = sorted({_date_str(d) for d in dates})
    subsequent_query = service_client.table(table).select(date_col).in_(date_col, date_strs)
    if greenhouse_id is not None:
        subsequent_query = subsequent_query.eq("greenhouse_id", greenhouse_id)
    matched = subsequent_query.execute()
    existing_dates = {row[date_col] for row in matched.data}
    new_days = len(set(date_strs) - existing_dates)
    if new_days < MIN_NEW_DAYS_SUBSEQUENT:
        raise HTTPException(
            422,
            f"Not sufficient data: se requieren al menos {MIN_NEW_DAYS_SUBSEQUENT} "
            f"días nuevos de datos, el archivo aporta {new_days} día(s) nuevo(s).",
        )
```

- [ ] **Step 5: Wire the check into `upload_excel`**

In `upload_excel` (`backend/routers/uploads.py:142-167`), change:

```python
    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, form_type, inv)
```

to:

```python
    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    _check_window_coverage(form_type, inv, _table_for(form_type), _date_col_for(form_type), df)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, form_type, inv)
```

Note: `_table_for` doesn't have an `"exteriores"` key, but `upload_excel` only ever receives `form_type` from `PER_INV_TYPES` (`sensores`, `riego`, `fenologia`, `produccion`), so `_table_for(form_type)` always resolves here. `produccion` still passes through this call, but `_check_window_coverage` no-ops for it immediately (not in the four checked form_types).

- [ ] **Step 6: Wire the check into `upload_exteriores`**

In `upload_exteriores` (`backend/routers/uploads.py:208-229`), change:

```python
    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, "exteriores", None)
```

to:

```python
    contents = await file.read()
    df = _parse_excel(contents, required_cols)
    _check_window_coverage("exteriores", None, "exterior_readings", "fecha", df)
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, "exteriores", None)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_sensores_first_submission_rejects_insufficient_weeks -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/routers/uploads.py backend/tests/test_uploads.py
git commit -m "feat: reject sensores/riego/exteriores uploads with short window coverage"
```

---

### Task 2: Fix the 5 pre-existing tests broken by the new check

**Context:** `test_upload_sensores_upserts_and_touches_lock`, `test_upload_sensores_handles_blank_optional_cell`, `test_upload_riego_upserts_and_touches_lock`, `test_upload_exteriores_upserts_without_greenhouse_id`, and `test_upload_fenologia_rejects_non_numeric_zona` each upload a single-row DataFrame to inv3 (or global, for exteriores). Task 1's check now runs unconditionally for these four form types and rejects a 1-week first submission (inv3/inv4 require 9/7 weeks; exteriores requires 9). These tests must mock a satisfied first-submission (`existing.data = []` + enough distinct weeks) to keep testing what they were built to test — ingestion/upsert/lock/validation behavior — independent of the coverage check. Without this, `test_upload_fenologia_rejects_non_numeric_zona` doesn't even get the 422 it expects for the right reason: the unmocked existence-query default (`MagicMock`, truthy) routes it into the subsequent-submission branch, which then crashes iterating an unconfigured `matched.data` — it must be explicitly forced onto the first-submission path with sufficient weeks so the zona-validation 422 (raised later, inside `_ingest_rows`) is what actually fires.

**Files:**
- Modify: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `_first_submission`, `_sensor_rows`, `_riego_rows`, `_exterior_rows`, `_phenology_rows` (Task 1, `backend/tests/test_uploads.py`).

- [ ] **Step 1: Run the full suite to confirm the 5 pre-existing failures**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: FAIL — `test_upload_sensores_upserts_and_touches_lock`, `test_upload_sensores_handles_blank_optional_cell`, `test_upload_riego_upserts_and_touches_lock`, `test_upload_exteriores_upserts_without_greenhouse_id` all return 422 instead of 200; `test_upload_fenologia_rejects_non_numeric_zona` errors (crashes iterating unconfigured mock data) instead of cleanly returning the expected 422.

- [ ] **Step 2: Fix `test_upload_sensores_upserts_and_touches_lock`**

Replace (current body, `backend/tests/test_uploads.py:68-84`):

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

with:

```python
def test_upload_sensores_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(9)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_in_file"] == 9
    assert body["rows_inserted"] == 9
    mock_supa.table.assert_any_call("sensor_readings_wide")
    mock_supa.table.assert_any_call("submission_locks")
```

- [ ] **Step 3: Fix `test_upload_sensores_handles_blank_optional_cell`**

Replace (current body, `backend/tests/test_uploads.py:118-131`):

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

with:

```python
def test_upload_sensores_handles_blank_optional_cell(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(9)  # inv3 requires 9 distinct weeks on first submission
    df.loc[df.index[-1], "humedad_abs_int"] = None
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9
```

- [ ] **Step 4: Fix `test_upload_riego_upserts_and_touches_lock`**

Replace (current body, `backend/tests/test_uploads.py:161-173`):

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
```

with:

```python
def test_upload_riego_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _riego_rows(9)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("riego.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/riego", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9
    mock_supa.table.assert_any_call("riego_readings")
    mock_supa.table.assert_any_call("submission_locks")
```

- [ ] **Step 5: Fix `test_upload_exteriores_upserts_without_greenhouse_id`**

Replace (current body, `backend/tests/test_uploads.py:201-219`):

```python
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
```

with:

```python
def test_upload_exteriores_upserts_without_greenhouse_id(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa, per_inv=False)
    df = _exterior_rows(9)  # exteriores requires 9 distinct weeks on first submission
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9
    mock_supa.table.assert_any_call("exterior_readings")
    # verify no greenhouse_id key was sent for the exterior_readings upsert
    upsert_call = next(
        c for c in mock_supa.table.return_value.upsert.call_args_list
    )
    assert "greenhouse_id" not in upsert_call[0][0]
```

- [ ] **Step 6: Fix `test_upload_fenologia_rejects_non_numeric_zona`**

Replace (current body, `backend/tests/test_uploads.py:103-115`):

```python
def test_upload_fenologia_rejects_non_numeric_zona(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "zona": "abc", "planta": 1,
        "racimos_puestos": 1, "flores_racimo_abiertas": 1,
        "racimos_en_planta": 1, "cantidad_tomates": 1, "racimo_en_cosecha": 1,
        "tomates_maduros": 1, "diametro_fruto_cm": 1.0, "crecimiento_planta_cm": 1.0,
    }])
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "zona" in r.text
```

with:

```python
def test_upload_fenologia_rejects_non_numeric_zona(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _phenology_rows(9)  # inv3 requires 9 distinct weeks on first submission
    df.loc[df.index[-1], "zona"] = "abc"
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "zona" in r.text
```

- [ ] **Step 7: Run the full suite to verify all 5 pass**

Run: `cd backend && python -m pytest tests/test_uploads.py -v`
Expected: PASS — all tests green (the new test from Task 1 and the 5 fixed tests).

- [ ] **Step 8: Commit**

```bash
git add backend/tests/test_uploads.py
git commit -m "fix: unblock pre-existing upload tests under the new window-coverage check"
```

---

### Task 3: Add remaining coverage — subsequent-submission, exteriores/fenología first-submission, unknown-inv fail-open, date-match correctness

**Files:**
- Modify: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `_first_submission`, `_prior_submission`, `_matched_dates`, `_weekly_dates`, `_sensor_rows`, `_exterior_rows`, `_phenology_rows` (Task 1).

- [ ] **Step 1: Write failing test — sensores subsequent-submission rejects insufficient new days**

Add to `backend/tests/test_uploads.py`:

```python
def test_upload_sensores_subsequent_rejects_insufficient_new_days(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    _matched_dates(mock_supa, [])  # none of the uploaded dates are already in the DB
    df = _sensor_rows(3)  # 3 new days < 7 required
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 422
    assert "días nuevos" in r.text
```

- [ ] **Step 2: Run test to verify it passes (implementation already exists from Task 1)**

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_sensores_subsequent_rejects_insufficient_new_days -v`
Expected: PASS

- [ ] **Step 3: Add test — exteriores first-submission rejects insufficient weeks**

Add to `backend/tests/test_uploads.py`:

```python
def test_upload_exteriores_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa, per_inv=False)
    df = _exterior_rows(5)  # exteriores requires 9 distinct weeks
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text
```

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_exteriores_first_submission_rejects_insufficient_weeks -v`
Expected: PASS

- [ ] **Step 4: Add test — new-distinct-day set-difference matches on the DB date-string format**

This confirms `_check_window_coverage`'s subsequent-path diff (`set(date_strs) - existing_dates`) actually excludes a matched date rather than silently accepting everything, using the exact `"%Y-%m-%d"` string format Postgres `date` columns return via postgrest (matching what `_date_str` produces on the write side). Add to `backend/tests/test_uploads.py`:

```python
def test_upload_sensores_subsequent_excludes_already_present_dates(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    dates = _weekly_dates(8)
    _matched_dates(mock_supa, [dates[0]])  # 1 of the 8 uploaded dates already exists
    df = _sensor_rows(8)  # 8 uploaded - 1 already-present = 7 new days, exactly the minimum
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 8
```

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_sensores_subsequent_excludes_already_present_dates -v`
Expected: PASS

- [ ] **Step 5: Add test — fenología first-submission rejects insufficient weeks**

Add to `backend/tests/test_uploads.py`:

```python
def test_upload_fenologia_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _phenology_rows(3)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text
```

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_fenologia_first_submission_rejects_insufficient_weeks -v`
Expected: PASS

- [ ] **Step 6: Add test — fenología subsequent check queries `week_date`, not `fecha`**

This confirms `_date_col_for("fenologia")` is actually wired through to the DB query (a wrong column name would silently query `week_date` on the DB but still work off `df["fecha"]` for the file side — this test locks in that `matched.data` keyed by `"week_date"` is read correctly). Add to `backend/tests/test_uploads.py`:

```python
def test_upload_fenologia_subsequent_uses_week_date_column(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    dates = _weekly_dates(8)
    _matched_dates(mock_supa, [dates[0]], date_col="week_date")  # 1 of 8 already present
    df = _phenology_rows(8)  # 8 uploaded - 1 already-present = 7 new days, exactly the minimum
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 8
```

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_fenologia_subsequent_uses_week_date_column -v`
Expected: PASS

- [ ] **Step 7: Add test — unmapped invernadero fails open (no 500) on first submission**

Add to `backend/tests/test_uploads.py`:

```python
def test_upload_sensores_unmapped_inv_skips_window_check(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(1)  # inv 99 isn't in SEQ_LEN_BY_INV -> check is skipped, not a 500
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/99/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1
```

Run: `cd backend && python -m pytest tests/test_uploads.py::test_upload_sensores_unmapped_inv_skips_window_check -v`
Expected: PASS

- [ ] **Step 8: Run the full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS — all tests green, no regressions in `test_predictions.py`, `test_phenology_live.py`, or elsewhere (Task 1/2/3 touch only `uploads.py` and `test_uploads.py`).

- [ ] **Step 9: Commit**

```bash
git add backend/tests/test_uploads.py
git commit -m "test: cover subsequent-submission, exteriores/fenologia, and unmapped-inv paths for window coverage"
```
