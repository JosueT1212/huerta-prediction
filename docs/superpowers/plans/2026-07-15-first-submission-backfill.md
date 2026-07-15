# First-Submission Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the first upload-triggered inference call of a season, backfill every ramp-up week (historical mean) plus every model-predictable week the current sensor backlog supports, instead of writing a single row and leaving weeks 1-N unfilled until the cron catches up.

**Architecture:** `harvest_start` moves from an inferred value (buggy: was `MIN(sensor fecha)`, resolved to the pre-harvest backlog date, not the production season's actual week 1) to an explicit admin-set fact in a new `harvest_start_dates` Supabase table, mirroring the existing `transplant_dates` table/router pattern. `scripts/live_inference.py`'s single "pull sensor data, build tensor, run model" block is extracted into `_run_model_forward(..., as_of_date)` so it can be called with a truncated window ending in the past, not just "today" — this lets the first-submission path replay several weeks' worth of model calls in one run. First-submission detection becomes "no `predictions` rows yet for this greenhouse+season" (decoupled from `harvest_start`).

**Tech Stack:** Python 3.11, FastAPI, Supabase (postgres), pytest, PyTorch (CNN-RNN inference), pandas.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-first-submission-backfill-design.md` (read this first — it has the full rationale for every decision below).
- `SKIP_FIRST_WEEKS` (live-inference ramp-up threshold) changes from 3 to 4 — this is a live-pipeline constant, NOT the training-time `skip_first_weeks: 3` in `Models/hp_inv{3,4}.yaml`, which stays untouched (CLAUDE.md §7-8 constant, out of scope).
- No hardcoded row count for the backfill — loop until `as_of_date > date.today()` or the sensor window needs padding. Do not reintroduce a fixed "5 weeks" or "6 weeks" assumption anywhere in code or tests.
- Do not rely on `MIN_WEEKS_FIRST_BY_INV` (`backend/routers/uploads.py`) as a sufficiency guarantee for truncated windows — check padding at runtime per `as_of_date` instead (spec §3a).
- T18 seed value for `harvest_start_dates`: `2026-07-27` (Monday of ISO week 31, 2026) for both Inv3 and Inv4 — this is a manual data step (Task 5), not a code change.
- Existing single-row-per-week behavior for non-first submissions must not change (same `predicted_for` formula, same mean-vs-model decision logic).

---

## File Structure

- `supabase/migrations/010_harvest_start_dates.sql` — new table, same shape as `transplant_dates`.
- `backend/routers/harvest_start_dates.py` — new router, GET/PUT `/harvest-start-date/{inv}`, identical pattern to `backend/routers/transplant_dates.py`.
- `backend/main.py` — register the new router.
- `backend/tests/test_harvest_start_dates.py` — new test file, mirrors `backend/tests/test_transplant_dates.py`.
- `backend/tests/conftest.py` — add the new router's `service_client` to the `mock_supa` fixture's monkeypatch list.
- `scripts/compute_historical_means.py` — `N_POSITIONS` 3 → 4.
- `scripts/tests/test_compute_historical_means.py` — add a test asserting `N_POSITIONS == 4`.
- `Models/results/historical_mean_inv{3,4}.json` — regenerated with keys `"0".."3"` (currently `"0".."2"`).
- `scripts/live_inference.py` — `get_harvest_start` reanchored to `harvest_start_dates`, new `is_first_submission()`, model-forward logic extracted into `_run_model_forward(..., as_of_date)`, `run_inference_for_greenhouse` gains a first-submission backfill branch, `SKIP_FIRST_WEEKS` 3 → 4.
- `scripts/tests/test_live_inference.py` — rewritten to match the new function signatures and backfill behavior.
- `demo/Demo Dashboard.html` — one more date-picker + save button in the existing `ins-panel-transplante` panel, calling `PUT /harvest-start-date/{inv}`.

---

## Task 1: `harvest_start_dates` table + CRUD router

**Files:**
- Create: `supabase/migrations/010_harvest_start_dates.sql`
- Create: `backend/routers/harvest_start_dates.py`
- Modify: `backend/main.py` (register router)
- Modify: `backend/tests/conftest.py` (add router to `mock_supa` monkeypatch list)
- Create: `backend/tests/test_harvest_start_dates.py`

**Interfaces:**
- Produces: `GET /harvest-start-date/{inv}` → `{greenhouse_id, fecha, updated_at}` (404 if unset); `PUT /harvest-start-date/{inv}` body `{"fecha": "YYYY-MM-DD"}` → `{"ok": true}`, upserts `harvest_start_dates` and calls `maybe_trigger_inference("harvest_start_date", inv)`.
- Consumes: `backend.auth.get_current_user` (existing dependency), `backend.supabase_client.service_client`, `backend.inference_trigger.maybe_trigger_inference` (existing, unchanged — `form_type` string is opaque to it except for the `== "exteriores"` special case).

- [ ] **Step 1: Write the failing test file**

Create `backend/tests/test_harvest_start_dates.py`:

```python
from unittest.mock import MagicMock


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_get_harvest_start_date_requires_auth(api_client):
    r = api_client.get("/harvest-start-date/3")
    assert r.status_code == 401


def test_get_harvest_start_date_returns_row(api_client, mock_supa):
    headers = _auth(mock_supa)
    calls = {"n": 0}

    def execute_data():
        calls["n"] += 1
        # 1st call = auth profile lookup, 2nd call = harvest_start_date lookup
        if calls["n"] == 1:
            return MagicMock(data={"disabled": False, "full_name": "Test User"})
        else:
            return MagicMock(data={"greenhouse_id": 3, "fecha": "2026-07-27", "updated_at": "2026-07-27T08:00:00Z"})

    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = execute_data
    r = api_client.get("/harvest-start-date/3", headers=headers)
    assert r.status_code == 200
    assert r.json()["fecha"] == "2026-07-27"


def test_get_harvest_start_date_404_when_unset(api_client, mock_supa):
    headers = _auth(mock_supa)
    calls = {"n": 0}

    def maybe_single_data():
        calls["n"] += 1
        return {"disabled": False, "full_name": "Test User"} if calls["n"] == 1 else None

    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = (
        lambda: MagicMock(data=maybe_single_data())
    )
    r = api_client.get("/harvest-start-date/3", headers=headers)
    assert r.status_code == 404


def test_put_harvest_start_date_requires_auth(api_client):
    r = api_client.put("/harvest-start-date/3", json={"fecha": "2026-07-27"})
    assert r.status_code == 401


def test_put_harvest_start_date_upserts(api_client, mock_supa, monkeypatch):
    from backend.routers import harvest_start_dates as hsd_router
    monkeypatch.setattr(hsd_router, "maybe_trigger_inference", MagicMock())
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.put("/harvest-start-date/3", headers=headers, json={"fecha": "2026-07-27"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_supa.table.assert_called_with("harvest_start_dates")
    mock_supa.table.return_value.upsert.assert_called_with(
        {"greenhouse_id": 3, "fecha": "2026-07-27"}, on_conflict="greenhouse_id"
    )


def test_put_harvest_start_date_triggers_inference_check(api_client, mock_supa, monkeypatch):
    from backend.routers import harvest_start_dates as hsd_router
    trigger_mock = MagicMock()
    monkeypatch.setattr(hsd_router, "maybe_trigger_inference", trigger_mock)
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.put("/harvest-start-date/4", headers=headers, json={"fecha": "2026-07-28"})
    assert r.status_code == 200
    trigger_mock.assert_called_once_with("harvest_start_date", 4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest backend/tests/test_harvest_start_dates.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'backend.routers.harvest_start_dates'` (router doesn't exist yet), and route-not-found 404s where 401 is expected.

- [ ] **Step 3: Create the migration**

Create `supabase/migrations/010_harvest_start_dates.sql`:

```sql
-- supabase/migrations/010_harvest_start_dates.sql

create table if not exists harvest_start_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);
```

- [ ] **Step 4: Create the router**

Create `backend/routers/harvest_start_dates.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import Annotated
from backend.auth import get_current_user
from backend.inference_trigger import maybe_trigger_inference
from backend.supabase_client import service_client

router = APIRouter()


class HarvestStartDateRequest(BaseModel):
    fecha: str


@router.get("/harvest-start-date/{inv}")
def get_harvest_start_date(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("harvest_start_dates")
        .select("greenhouse_id, fecha, updated_at")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        raise HTTPException(404, f"No hay fecha de inicio de cosecha para invernadero {inv}")
    return row


@router.put("/harvest-start-date/{inv}")
async def put_harvest_start_date(
    inv: int,
    body: HarvestStartDateRequest,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    service_client.table("harvest_start_dates").upsert(
        {"greenhouse_id": inv, "fecha": body.fecha}, on_conflict="greenhouse_id"
    ).execute()
    # La fecha de inicio de cosecha no es una carga semanal (no participa en
    # uploads_complete_for_inv), pero la inferencia la requiere: si el set
    # semanal ya estaba completo cuando se guardó, el trigger de uploads ya
    # corrió sin ella y se saltó — este re-chequeo cubre ese caso (mismo
    # patrón que backend/routers/transplant_dates.py).
    await run_in_threadpool(maybe_trigger_inference, "harvest_start_date", inv)
    return {"ok": True}
```

- [ ] **Step 5: Register the router in `backend/main.py`**

Modify `backend/main.py` — add the import next to the `transplant_dates_router` import (around line 37):

```python
from backend.routers import transplant_dates as transplant_dates_router  # noqa: E402
from backend.routers import harvest_start_dates as harvest_start_dates_router  # noqa: E402
```

And register it next to the transplant-dates registration (around line 71):

```python
app.include_router(transplant_dates_router.router)
app.include_router(harvest_start_dates_router.router)
```

- [ ] **Step 6: Add the router to the `mock_supa` fixture**

Modify `backend/tests/conftest.py` — add one line after the `transplant_dates` monkeypatch (line 58):

```python
    monkeypatch.setattr("backend.routers.transplant_dates.service_client", mock)
    monkeypatch.setattr("backend.routers.harvest_start_dates.service_client", mock)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest backend/tests/test_harvest_start_dates.py -v`
Expected: `6 passed`

- [ ] **Step 8: Run the full backend test suite to check for regressions**

Run: `pytest backend/tests/ -v`
Expected: all tests pass (existing `test_transplant_dates.py` etc. unaffected).

- [ ] **Step 9: Commit**

```bash
git add supabase/migrations/010_harvest_start_dates.sql backend/routers/harvest_start_dates.py backend/main.py backend/tests/conftest.py backend/tests/test_harvest_start_dates.py
git commit -m "feat(backend): add harvest_start_dates table + GET/PUT /harvest-start-date/{inv}"
```

---

## Task 2: Extend ramp-up mean coverage from 3 to 4 weeks

**Files:**
- Modify: `scripts/compute_historical_means.py:24`
- Modify: `scripts/tests/test_compute_historical_means.py`
- Modify (regenerated data): `Models/results/historical_mean_inv3.json`, `Models/results/historical_mean_inv4.json`

**Interfaces:**
- Consumes: `Models.cnn_rnn_yield.load_production(inv)` (existing, unchanged).
- Produces: `Models/results/historical_mean_inv{3,4}.json` now has keys `"0","1","2","3"` (was `"0","1","2"`) — Task 3 depends on the `"3"` key existing for the live-inference ramp-up branch to have 4 full weeks of mean coverage.

- [ ] **Step 1: Write the failing test**

Modify `scripts/tests/test_compute_historical_means.py` — add at the end:

```python
def test_n_positions_is_four():
    from scripts.compute_historical_means import N_POSITIONS
    assert N_POSITIONS == 4


def test_historical_means_by_position_covers_four_weeks_by_default():
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13"] * 5,
        "kg_reales": [1.0, 2.0, 3.0, 4.0, 999.0],
    })
    means = historical_means_by_position(df)  # uses module default N_POSITIONS
    assert means == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest scripts/tests/test_compute_historical_means.py -v`
Expected: `test_n_positions_is_four` FAILS (`N_POSITIONS == 3`), `test_historical_means_by_position_covers_four_weeks_by_default` FAILS (only 3 keys returned).

- [ ] **Step 3: Change the constant**

Modify `scripts/compute_historical_means.py:24`:

```python
N_POSITIONS = 4
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest scripts/tests/test_compute_historical_means.py -v`
Expected: all pass (4 tests total).

- [ ] **Step 5: Regenerate the historical mean JSON files**

Run: `python scripts/compute_historical_means.py`
Expected output: two lines like `Inv3 → .../historical_mean_inv3.json: {0: ..., 1: ..., 2: ..., 3: ...}` / same for Inv4 — confirm both dicts now have 4 keys (`0` through `3`), not 3.

- [ ] **Step 6: Verify the regenerated files on disk**

Run: `cat Models/results/historical_mean_inv3.json Models/results/historical_mean_inv4.json`
Expected: both files contain keys `"0"`, `"1"`, `"2"`, `"3"`.

- [ ] **Step 7: Commit**

```bash
git add scripts/compute_historical_means.py scripts/tests/test_compute_historical_means.py Models/results/historical_mean_inv3.json Models/results/historical_mean_inv4.json
git commit -m "feat(scripts): extend historical-mean ramp-up coverage from 3 to 4 weeks"
```

---

## Task 3: `harvest_start` reanchor + first-submission backfill in `live_inference.py`

This is the core behavior change. It rewrites `get_harvest_start`, adds
`is_first_submission`, extracts the model-forward-pass block into a
reusable `_run_model_forward(..., as_of_date)` with a padding-based
sufficiency check, and rewrites `run_inference_for_greenhouse` to backfill
on first submission.

**Files:**
- Modify: `scripts/live_inference.py` (full rewrite of the functions listed below — the file is small, ~230 lines, and every function is coupled to the same `run_inference_for_greenhouse` flow, so the file is replaced wholesale in Step 3 rather than patched function-by-function)
- Modify: `scripts/tests/test_live_inference.py` (full rewrite — old tests target function signatures that no longer exist)

**Interfaces:**
- Consumes: `backend.live_features.build_input_tensor(inv, wide_rows, pheno_rows, transplant_date, pipeline_path)` (existing, unchanged signature — confirmed it does not read "today" internally, just tails the last `seq_len` rows it's given); `backend.live_features.aggregate_wide_to_weekly(wide_rows, sensor_cols)` (existing, used here only for the padding check); `Models.cnn_rnn_yield.CNNRNN` (existing, unchanged).
- Produces:
  - `get_harvest_start(supa, inv: int) -> date | None` (signature changed — no longer takes `transplant_date`)
  - `is_first_submission(supa, inv: int) -> bool`
  - `_run_model_forward(supa, inv: int, transplant_date: date, as_of_date: date) -> float | None` (returns `None` when the sensor window ending at `as_of_date` has fewer than `seq_len` real weekly rows)
  - `_backfill_first_submission(supa, inv: int, transplant_date: date, harvest_start: date, dry_run: bool) -> bool`
  - `run_inference_for_greenhouse(supa, inv: int, dry_run: bool) -> bool` (same signature, new internal branching)

- [ ] **Step 1: Write the failing test file**

Replace `scripts/tests/test_live_inference.py` entirely:

```python
import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def test_monday_of_week():
    from scripts.live_inference import monday_of_week
    assert monday_of_week(date(2026, 7, 15)) == date(2026, 7, 13)  # Wednesday -> Monday
    assert monday_of_week(date(2026, 7, 13)) == date(2026, 7, 13)  # already Monday


def test_week_in_season():
    from scripts.live_inference import week_in_season
    transplant = date(2026, 5, 20)
    assert week_in_season(date(2026, 5, 20), transplant) == 0
    assert week_in_season(date(2026, 5, 27), transplant) == 1
    assert week_in_season(date(2026, 6, 24), transplant) == 5


def test_load_historical_mean_returns_value_for_ramp_up_week(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 1) == 150.0


def test_load_historical_mean_returns_none_outside_ramp_up(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 4) is None


def test_run_inference_for_greenhouse_returns_false_when_no_transplant_date():
    from scripts.live_inference import run_inference_for_greenhouse

    mock_supa = MagicMock()
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value.data = None

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False


def test_run_inference_for_greenhouse_returns_false_when_no_harvest_start():
    from scripts.live_inference import run_inference_for_greenhouse

    mock_supa = MagicMock()
    transplant_response = MagicMock()
    transplant_response.data = {"fecha": "2026-05-15"}
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

    harvest_response = MagicMock()
    harvest_response.data = None
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = harvest_response

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False
    mock_supa.table("predictions").upsert.assert_not_called()


def test_get_harvest_start_reads_harvest_start_dates_table():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = {"greenhouse_id": 3, "fecha": "2026-07-27", "updated_at": "2026-07-27T08:00:00Z"}
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = resp

    result = get_harvest_start(mock_supa, 3)
    assert result == date(2026, 7, 27)


def test_get_harvest_start_returns_none_when_unset():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = None
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = resp

    result = get_harvest_start(mock_supa, 3)
    assert result is None


def test_is_first_submission_true_when_no_predictions_rows():
    from scripts.live_inference import is_first_submission

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = []
    mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = resp

    assert is_first_submission(mock_supa, 3) is True


def test_is_first_submission_false_when_predictions_rows_exist():
    from scripts.live_inference import is_first_submission

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = [{"id": 1}]
    mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = resp

    assert is_first_submission(mock_supa, 3) is False


def test_run_inference_for_greenhouse_non_first_submission_uses_historical_mean(tmp_path):
    from scripts.live_inference import run_inference_for_greenhouse

    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))

    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        mock_supa = MagicMock()
        transplant_date = date.today() - timedelta(days=5)
        # harvest_start far enough in the future that wis_target < SKIP_FIRST_WEEKS (4)
        harvest_start = date.today() + timedelta(days=20)

        transplant_response = MagicMock()
        transplant_response.data = {"fecha": transplant_date.isoformat()}
        mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

        harvest_response = MagicMock()
        harvest_response.data = {"fecha": harvest_start.isoformat()}
        mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = harvest_response

        predictions_select_resp = MagicMock()
        predictions_select_resp.data = [{"id": 1}]  # not first submission
        mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = predictions_select_resp

        result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
        assert result is True
        mock_supa.table("predictions").upsert.assert_called_once()
        upsert_call = mock_supa.table("predictions").upsert.call_args[0][0]
        assert upsert_call["model_version"] == "historical_mean_v1"


def test_backfill_first_submission_writes_mean_weeks_and_stops_at_padding(monkeypatch):
    from scripts.live_inference import _backfill_first_submission, HORIZON_WEEKS, SKIP_FIRST_WEEKS

    mock_supa = MagicMock()
    transplant_date = date(2026, 5, 15)
    harvest_start = date(2026, 7, 27)  # week 31

    monkeypatch.setattr(
        "scripts.live_inference.load_historical_mean",
        lambda inv, wis: {0: 100.0, 1: 150.0, 2: 200.0, 3: 250.0}.get(wis),
    )

    calls = []

    def fake_run_model_forward(supa, inv, td, as_of_date):
        calls.append(as_of_date)
        # First model call succeeds (enough backlog), second is padded (not enough).
        return 5000.0 if len(calls) == 1 else None

    monkeypatch.setattr("scripts.live_inference._run_model_forward", fake_run_model_forward)
    # Freeze "today" far enough ahead that the loop's ceiling isn't what stops it —
    # the padding check (fake_run_model_forward returning None) should stop it instead.
    monkeypatch.setattr("scripts.live_inference.date", _FakeDate)

    result = _backfill_first_submission(mock_supa, 3, transplant_date, harvest_start, dry_run=False)
    assert result is True

    upsert_calls = mock_supa.table("predictions").upsert.call_args_list
    # SKIP_FIRST_WEEKS (4) mean rows + exactly 1 model row (the 2nd model call returned None and stopped the loop)
    assert len(upsert_calls) == SKIP_FIRST_WEEKS + 1
    model_versions = [c.args[0]["model_version"] for c in upsert_calls]
    assert model_versions[:SKIP_FIRST_WEEKS] == ["historical_mean_v1"] * SKIP_FIRST_WEEKS
    assert model_versions[SKIP_FIRST_WEEKS] == "cnn_rnn_v2_production"
    # First model call's as_of_date = harvest_start + (SKIP_FIRST_WEEKS - HORIZON_WEEKS) weeks
    assert calls[0] == harvest_start + timedelta(weeks=SKIP_FIRST_WEEKS - HORIZON_WEEKS)


class _FakeDate(date):
    @classmethod
    def today(cls):
        return date(2026, 12, 31)  # far enough ahead that the ceiling never triggers first
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest scripts/tests/test_live_inference.py -v`
Expected: multiple failures — `get_harvest_start(mock_supa, 3)` called with 2 args instead of 3 in old signature, `is_first_submission` / `_backfill_first_submission` / `_run_model_forward` don't exist yet, `harvest_start_dates` table never queried by current code.

- [ ] **Step 3: Rewrite `scripts/live_inference.py`**

Replace the full file contents:

```python
"""
Weekly inference cron script — runs every Sunday 8am UTC.

Railway Cron:
  Schedule: 0 8 * * 0
  Command:  python scripts/live_inference.py

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_KEY
Optional:
  DRY_RUN=1  — run models but skip Supabase write
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
import joblib
from supabase import create_client

from backend.live_features import build_input_tensor, aggregate_wide_to_weekly

# Model class verified via:
# grep -n "^class" Models/cnn_rnn_yield.py  → CNNRNN at line 757
from Models.cnn_rnn_yield import CNNRNN

RESULTS_DIR = ROOT / "Models" / "results"
HORIZON_WEEKS = 5
MODEL_VERSION = "cnn_rnn_v2_production"
HISTORICAL_MEAN_VERSION = "historical_mean_v1"
SKIP_FIRST_WEEKS = 4
CURRENT_SEASON = "T18"
INV_IDS = (3, 4)


def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_in_season(target_date: date, transplant_date: date) -> int:
    return max(0, (target_date - transplant_date).days // 7)


def load_historical_mean(inv: int, wis: int) -> float | None:
    path = RESULTS_DIR / f"historical_mean_inv{inv}.json"
    with open(path) as f:
        means = json.load(f)
    return means.get(str(wis))


def get_transplant_date(supa, inv: int) -> date | None:
    resp = (
        supa.table("transplant_dates")
        .select("fecha")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        return None
    return date.fromisoformat(row["fecha"])


def get_harvest_start(supa, inv: int) -> date | None:
    """Admin-set fact — see harvest_start_dates table (010_harvest_start_dates.sql).

    NOT inferred from sensor_readings_wide: sensors start recording ~10-11
    weeks before harvest (CLAUDE.md §5) and the client's first upload of a
    season batches that entire backlog in one payload, so MIN(fecha) from
    sensor data resolves to the pre-harvest backlog date, not the season's
    actual week 1. See docs/superpowers/specs/2026-07-15-first-submission-backfill-design.md §1.
    """
    resp = (
        supa.table("harvest_start_dates")
        .select("fecha")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        return None
    return date.fromisoformat(row["fecha"])


def is_first_submission(supa, inv: int) -> bool:
    """True iff no predictions row exists yet for this greenhouse+season."""
    resp = (
        supa.table("predictions")
        .select("id")
        .eq("greenhouse_id", inv)
        .eq("season", CURRENT_SEASON)
        .limit(1)
        .execute()
    )
    return not (resp.data or [])


def _run_model_forward(supa, inv: int, transplant_date: date, as_of_date: date) -> float | None:
    """CNN-RNN forward pass for a sensor window ending at as_of_date.

    Returns None if the window doesn't have seq_len real (non-padded) weekly
    rows yet — the caller must not write a prediction built on fabricated
    zero-padding.
    """
    cutoff = (as_of_date - timedelta(weeks=12)).isoformat()
    upper = as_of_date.isoformat()

    sensor_resp = (
        supa.table("sensor_readings_wide").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).lte("fecha", upper).execute()
    )
    riego_resp = (
        supa.table("riego_readings").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).lte("fecha", upper).execute()
    )
    # exterior_readings is global (no greenhouse_id column) — same weather feeds both invernaderos
    ext_resp = (
        supa.table("exterior_readings").select("*")
        .gte("fecha", cutoff).lte("fecha", upper).execute()
    )

    merged_by_fecha: dict[str, dict] = {}
    for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
        merged_by_fecha.setdefault(row["fecha"], {}).update(row)
    wide_rows = list(merged_by_fecha.values())
    print(f"  Inv{inv} sensor rows pulled (as of {as_of_date}): {len(sensor_resp.data or [])} sensores + "
          f"{len(riego_resp.data or [])} riego + {len(ext_resp.data or [])} exteriores "
          f"→ {len(wide_rows)} merged by fecha")

    pheno_resp = (
        supa.table("phenology_observations").select("*")
        .eq("greenhouse_id", inv).gte("week_date", cutoff).lte("week_date", upper).execute()
    )
    pheno_rows = pheno_resp.data or []

    pipeline_path = RESULTS_DIR / f"production_pipeline_inv{inv}.pkl"
    pipeline = joblib.load(pipeline_path)
    seq_len = pipeline["seq_len"]
    sensor_cols = pipeline["sensor_cols"]

    weekly_check = aggregate_wide_to_weekly(wide_rows, sensor_cols)
    if len(weekly_check) < seq_len:
        print(f"  Inv{inv} → only {len(weekly_check)} weeks of sensor data as of {as_of_date}, "
              f"need {seq_len} — not enough history yet.")
        return None

    x_sensor, x_temporal = build_input_tensor(
        inv, wide_rows, pheno_rows,
        transplant_date=transplant_date,
        pipeline_path=pipeline_path,
    )

    scaler_y = pipeline["scaler_y"]
    bc_lambda = pipeline["bc_lambda"]
    transform = pipeline["transform"]

    with open(ROOT / "Models" / f"hp_inv{inv}.yaml") as f:
        hp = yaml.safe_load(f)

    model = CNNRNN(
        n_sensor=x_sensor.shape[-1],
        n_temporal=x_temporal.shape[-1],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        cnn_padding=hp["cnn_padding"],
        num_cnn_blocks=hp["num_cnn_blocks"],
        lstm_hidden=hp["lstm_hidden"],
        lstm_layers=hp["lstm_layers"],
        dropout=0.0,
        fc_hidden=hp["fc_hidden"],
    )
    model_path = RESULTS_DIR / f"production_cnn_rnn_inv{inv}.pt"
    # map_location="cpu": checkpoints were saved on Apple Silicon (mps
    # device) and Railway's container is CPU-only Linux — without this,
    # torch.load fails with "Storage device not recognized: mps".
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        y_norm = model(x_sensor, x_temporal).squeeze().item()

    y_scaled = scaler_y.inverse_transform([[y_norm]])[0][0]

    if transform == "log":
        import numpy as np
        kg_predicted = float(np.expm1(y_scaled))
    elif transform == "boxcox" and bc_lambda is not None:
        from scipy.special import inv_boxcox
        kg_predicted = float(inv_boxcox(y_scaled, bc_lambda))
    else:
        kg_predicted = float(y_scaled)

    return kg_predicted


def _backfill_first_submission(supa, inv: int, transplant_date: date, harvest_start: date,
                                dry_run: bool) -> bool:
    """First submission of the season: write every ramp-up mean week, then
    every model week the current sensor backlog supports — no hardcoded
    row count (spec §3)."""
    wrote_any = False

    for wis in range(SKIP_FIRST_WEEKS):
        historical_kg = load_historical_mean(inv, wis)
        if historical_kg is None:
            print(f"  ERROR: missing historical_mean_inv{inv}.json[\"{wis}\"] — skipping that week.",
                  file=sys.stderr)
            continue
        predicted_for = harvest_start + timedelta(weeks=wis)
        _upsert_prediction(supa, inv, predicted_for, historical_kg, HISTORICAL_MEAN_VERSION, dry_run)
        wrote_any = True

    wis = SKIP_FIRST_WEEKS
    while True:
        as_of_date = harvest_start + timedelta(weeks=wis - HORIZON_WEEKS)
        if as_of_date > date.today():
            print(f"  Inv{inv} → backfilled through week-in-season {wis - 1}; "
                  f"week-in-season {wis} would need sensor data from the future.")
            break
        kg_predicted = _run_model_forward(supa, inv, transplant_date, as_of_date)
        if kg_predicted is None:
            print(f"  Inv{inv} → backfilled through week-in-season {wis - 1}; "
                  f"insufficient sensor history for week-in-season {wis}.")
            break
        predicted_for = harvest_start + timedelta(weeks=wis)
        _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)
        wrote_any = True
        wis += 1

    return wrote_any


def run_inference_for_greenhouse(supa, inv: int, dry_run: bool) -> bool:
    transplant_date = get_transplant_date(supa, inv)
    if transplant_date is None:
        print(f"  ERROR: no transplant_date set for invernadero {inv} — skipping. "
              f"Set it via PUT /transplant-date/{inv}.", file=sys.stderr)
        return False

    harvest_start = get_harvest_start(supa, inv)
    if harvest_start is None:
        print(f"  ERROR: no harvest_start set for invernadero {inv} — skipping. "
              f"Set it via PUT /harvest-start-date/{inv}.", file=sys.stderr)
        return False

    if is_first_submission(supa, inv):
        return _backfill_first_submission(supa, inv, transplant_date, harvest_start, dry_run)

    predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS))
    wis_target = week_in_season(predicted_for, harvest_start)

    if wis_target < SKIP_FIRST_WEEKS:
        historical_kg = load_historical_mean(inv, wis_target)
        if historical_kg is not None:
            print(f"  Inv{inv} → week-in-season {wis_target} < {SKIP_FIRST_WEEKS} "
                  f"(ramp-up) → historical mean {historical_kg:.1f} kg")
            _upsert_prediction(supa, inv, predicted_for, historical_kg,
                                HISTORICAL_MEAN_VERSION, dry_run)
            return True

    kg_predicted = _run_model_forward(supa, inv, transplant_date, date.today())
    if kg_predicted is None:
        print(f"  ERROR: insufficient sensor history for invernadero {inv} — skipping.",
              file=sys.stderr)
        return False

    print(f"  Inv{inv} → {kg_predicted:.1f} kg for week {predicted_for}")
    _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)
    return True


def _upsert_prediction(supa, inv: int, predicted_for: date, kg_predicted: float,
                        model_version: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] would upsert inv{inv}: {predicted_for} → {kg_predicted:.1f} kg "
              f"({model_version})")
        return
    supa.table("predictions").upsert({
        "greenhouse_id": inv,
        "predicted_for": predicted_for.isoformat(),
        "kg_predicted": kg_predicted,
        "model_version": model_version,
        "season": CURRENT_SEASON,
    }, on_conflict="greenhouse_id,predicted_for").execute()
    print(f"  Inv{inv} written to predictions table.")


def main():
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    supa = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    succeeded = 0
    for inv in INV_IDS:
        try:
            if run_inference_for_greenhouse(supa, inv, dry_run):
                succeeded += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: invernadero {inv} failed: {e}", file=sys.stderr)

    if succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest scripts/tests/test_live_inference.py -v`
Expected: all tests pass.

- [ ] **Step 5: Update `backend/inference_trigger.py`'s call site**

No code change needed — `run_inference_for_greenhouse(service_client, check_inv, dry_run=False)` (`backend/inference_trigger.py:59`) already calls with the same `(supa, inv, dry_run)` signature, unchanged.

Run: `pytest backend/tests/test_inference_trigger.py -v`
Expected: all pass (no regressions — this file mocks `run_inference_for_greenhouse` itself, doesn't touch its internals).

- [ ] **Step 6: Run the full test suite**

Run: `pytest backend/tests/ scripts/tests/ -v`
Expected: all tests pass, no regressions in `test_live_features.py`, `test_uploads.py`, `test_inference_trigger.py`.

- [ ] **Step 7: Commit**

```bash
git add scripts/live_inference.py scripts/tests/test_live_inference.py
git commit -m "feat(scripts): first-submission backfill — reanchor harvest_start, dynamic ramp-up+model loop"
```

---

## Task 4: Dashboard — harvest-start-date admin picker

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `GET /harvest-start-date/{inv}`, `PUT /harvest-start-date/{inv}` (Task 1).
- No JS test framework exists in this repo for the dashboard (confirmed: no `.spec.js`/Playwright test files under version control) — this task is verified manually in-browser, same as the rest of `demo/Demo Dashboard.html`'s existing features (see `demo/CLAUDE.md` §8 "Verificado en navegador (Playwright)" — ad hoc, not a committed suite).

- [ ] **Step 1: Add the second date-picker to the existing transplant-date panel**

Modify `demo/Demo Dashboard.html` — the `ins-panel-transplante` panel (lines 1160-1169) currently has one field. Add a second field block right after the existing one, inside the same `<div style="display:flex;...">`:

```html
    <div class="panel insert-panel ins-panel" id="ins-panel-transplante" hidden>
      <div style="display:flex;align-items:flex-end;gap:8px;flex-wrap:wrap">
        <div>
          <label style="display:block;font-size:0.8rem;margin-bottom:4px">Fecha de transplante</label>
          <input type="date" id="transplant-date-input" />
        </div>
        <button class="btn-primary" id="transplant-date-save">Guardar</button>
        <div>
          <label style="display:block;font-size:0.8rem;margin-bottom:4px">Fecha de inicio de cosecha</label>
          <input type="date" id="harvest-start-date-input" />
        </div>
        <button class="btn-primary" id="harvest-start-date-save">Guardar</button>
      </div>
      <div class="insert-result" id="transplant-date-result"></div>
      <div class="insert-result" id="harvest-start-date-result"></div>
    </div>
```

- [ ] **Step 2: Add the refresh + init JS functions**

Modify `demo/Demo Dashboard.html` — add right after `refreshTransplantDateTab` (after line 2911):

```javascript
async function refreshHarvestStartDateTab(inv) {
  const input = document.getElementById('harvest-start-date-input');
  const result = document.getElementById('harvest-start-date-result');
  if (!input) return;
  input.value = '';
  if (result) result.textContent = '';
  const data = await fetchJSON(`/harvest-start-date/${inv}`).catch(() => null);
  if (data) input.value = data.fecha;
}

function initHarvestStartDateForm() {
  document.getElementById('harvest-start-date-save')?.addEventListener('click', async () => {
    const inv = +document.getElementById('ins-inv').value;
    const val = document.getElementById('harvest-start-date-input').value;
    const result = document.getElementById('harvest-start-date-result');
    if (!val) return;
    const token = localStorage.getItem('sb_token');
    try {
      const r = await fetch(`${API_BASE}/harvest-start-date/${inv}`, {
        method: 'PUT',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ fecha: val }),
      });
      if (!r.ok) throw new Error(`Error ${r.status}`);
      if (result) { result.className = 'insert-result ok'; result.textContent = '✓ Fecha guardada'; }
    } catch (e) {
      if (result) { result.className = 'insert-result bad'; result.textContent = '⚠ ' + e.message; }
    }
  });
}
```

- [ ] **Step 3: Wire the refresh call into the existing tab-switch and inv-switch handlers**

Modify `demo/Demo Dashboard.html:2952` (inside `initInsTabs`'s tab-click handler):

```javascript
      if (tab === 'hist') loadHistorial(inv);
      else if (tab === 'transplante') { refreshTransplantDateTab(inv); refreshHarvestStartDateTab(inv); }
      else refreshLockBanner(tab, GLOBAL_UPLOAD_TYPES.includes(tab) ? null : inv);
```

Modify `demo/Demo Dashboard.html:2973` (inside the `ins-inv` change handler):

```javascript
    if (activeTab === 'hist') loadHistorial(inv);
    else if (activeTab === 'transplante') { refreshTransplantDateTab(inv); refreshHarvestStartDateTab(inv); }
    else if (activeTab) refreshLockBanner(activeTab, GLOBAL_UPLOAD_TYPES.includes(activeTab) ? null : inv);
```

- [ ] **Step 4: Call `initHarvestStartDateForm()` at startup**

Modify `demo/Demo Dashboard.html:3096` — next to the existing `initTransplantDateForm()` call:

```javascript
    initTransplantDateForm();
    initHarvestStartDateForm();
```

- [ ] **Step 5: Manual verification in the browser**

Run: `uvicorn backend.main:app --port 8000 --reload` (from repo root, per `demo/CLAUDE.md` §3)

In the browser at `http://localhost:8000/`:
1. Log in (any credentials — cosmetic gate), go to "Insertar datos" → tab "Fecha de transplante".
2. Confirm both date inputs render side by side, each with its own "Guardar" button.
3. Set a harvest-start date, click its "Guardar", confirm "✓ Fecha guardada" appears and `GET /harvest-start-date/3` (Network tab) returns it on tab reload.
4. Switch `ins-inv` selector to invernadero 4 while on this tab, confirm both fields reset/refetch for inv4 (not stale inv3 values).

Expected: no console errors, both pickers independently persist per greenhouse.

- [ ] **Step 6: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add harvest-start-date admin picker alongside transplant-date"
```

---

## Task 5: Seed `harvest_start_dates` for T18 (manual data step)

This is not a code change — it's the operational step required before T18
go-live, called out explicitly in the spec (§"Files touched").

- [ ] **Step 1: Set harvest_start for Inv3**

Via the dashboard picker (Task 4) or `curl`:

```bash
curl -X PUT "$API_BASE/harvest-start-date/3" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"fecha": "2026-07-27"}'
```

Expected: `{"ok": true}`.

- [ ] **Step 2: Set harvest_start for Inv4**

```bash
curl -X PUT "$API_BASE/harvest-start-date/4" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"fecha": "2026-07-27"}'
```

Expected: `{"ok": true}`.

- [ ] **Step 3: Verify both are readable**

```bash
curl "$API_BASE/harvest-start-date/3" -H "Authorization: Bearer $TOKEN"
curl "$API_BASE/harvest-start-date/4" -H "Authorization: Bearer $TOKEN"
```

Expected: both return `{"greenhouse_id": <inv>, "fecha": "2026-07-27", "updated_at": "..."}`.

No commit — this is a Supabase data write, not a code change.
