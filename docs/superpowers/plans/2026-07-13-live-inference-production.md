# Live Inference → Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the weekly live-inference pipeline production-ready: switch to
the production-refit CNN-RNN models, support both greenhouses, move
transplant-date config from env vars into Supabase, add a historical-mean
fallback for the first 3 ramp-up weeks of a season, and surface live weekly
predictions in the dashboard as a new "T18" season tab — without touching
the existing T13–T17 sales-demo replay.

**Architecture:** `scripts/live_inference.py` (Railway weekly cron) loops
over both greenhouses, reading transplant date from a new `transplant_dates`
Supabase table, loading `production_cnn_rnn_inv{3,4}.pt` +
`production_pipeline_inv{3,4}.pkl`, falling back to a precomputed
historical-mean JSON for ramp-up weeks, and upserting into `predictions`
(now carrying a `season` column). The dashboard adds season tabs per
greenhouse view; the existing "Predicción en vivo" cards move from the T17
view into the new T18 tab and stop blending into the T17 replay arrays.

**Tech Stack:** FastAPI (`backend/`), Supabase (Postgres), PyTorch (model),
vanilla JS/HTML dashboard (`demo/Demo Dashboard.html`), pytest +
`unittest.mock` for backend tests.

## Global Constraints

- Spanish column/table names and labels stay in Spanish (project convention).
- `Models/cnn_rnn_yield.py` and other training scripts are NOT modified —
  this plan only touches `scripts/`, `backend/`, `supabase/migrations/`, and
  `demo/Demo Dashboard.html`.
- `backend/engine.py`, `/inference/{inv}`, `/inference/{inv}/window`,
  `/metrics/{inv}`, and the T17 slider/PI-band/kg-m²-hero dashboard UI are
  untouched.
- `best_cnn_rnn_inv{3,4}.pt`, `pipeline_inv3.pkl`, and the `.npz`/
  `_metrics.csv` eval artifacts stay on disk, unused by the live path but
  still used by grid-search/eval scripts.
- New backend routes follow the existing pattern in
  `backend/routers/phenology_live.py` / `predictions.py`: JWT auth via
  `Depends(get_current_user)`, `service_client` from
  `backend/supabase_client.py`.
- Tests follow `backend/tests/conftest.py` conventions: `mock_supa` fixture
  patches `service_client` on each router module; `api_client` gives a
  `TestClient`.

---

### Task 1: `transplant_dates` table + `predictions.season` column

**Files:**
- Create: `supabase/migrations/009_transplant_dates.sql`

**Interfaces:**
- Produces: table `transplant_dates(greenhouse_id int primary key, fecha date not null, updated_at timestamptz not null default now())`; column `predictions.season text` (nullable, no default — set explicitly on every insert going forward).

- [ ] **Step 1: Write the migration file**

```sql
-- supabase/migrations/009_transplant_dates.sql

create table if not exists transplant_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);

alter table predictions add column if not exists season text;
```

- [ ] **Step 2: Apply it locally / verify syntax**

Run: `cat supabase/migrations/009_transplant_dates.sql | psql "$SUPABASE_DB_URL" -f -`

(If no local Postgres connection is available in this environment, skip
execution and just confirm the file parses as valid SQL by eye — the
migration will be applied via the Supabase dashboard SQL editor per the
project's existing convention, same as migrations 001–008.)

Expected: `CREATE TABLE` then `ALTER TABLE` printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/009_transplant_dates.sql
git commit -m "feat(db): add transplant_dates table and predictions.season column"
```

---

### Task 2: `transplant_dates` router (GET/PUT)

**Files:**
- Create: `backend/routers/transplant_dates.py`
- Modify: `backend/main.py:37` (add import), `backend/main.py:69` (register router)
- Test: `backend/tests/test_transplant_dates.py`

**Interfaces:**
- Consumes: `backend.auth.get_current_user`, `backend.supabase_client.service_client` (same pattern as `phenology_live.py`).
- Produces: `GET /transplant-date/{inv}` → `{greenhouse_id, fecha, updated_at}` (404 if no row); `PUT /transplant-date/{inv}` body `{"fecha": "2026-05-20"}` → `{"ok": true}`. Router object `router` (import as `transplant_dates_router`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_transplant_dates.py
from unittest.mock import MagicMock


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_get_transplant_date_requires_auth(api_client):
    r = api_client.get("/transplant-date/3")
    assert r.status_code == 401


def test_get_transplant_date_returns_row(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
        "greenhouse_id": 3, "fecha": "2026-05-20", "updated_at": "2026-05-20T08:00:00Z"
    }
    r = api_client.get("/transplant-date/3", headers=headers)
    assert r.status_code == 200
    assert r.json()["fecha"] == "2026-05-20"


def test_get_transplant_date_404_when_unset(api_client, mock_supa):
    headers = _auth(mock_supa)
    calls = {"n": 0}

    def maybe_single_data():
        calls["n"] += 1
        # 1st call = auth profile lookup, 2nd call = transplant_date lookup
        return {"disabled": False, "full_name": "Test User"} if calls["n"] == 1 else None

    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = (
        lambda: MagicMock(data=maybe_single_data())
    )
    r = api_client.get("/transplant-date/3", headers=headers)
    assert r.status_code == 404


def test_put_transplant_date_requires_auth(api_client):
    r = api_client.put("/transplant-date/3", json={"fecha": "2026-05-20"})
    assert r.status_code == 401


def test_put_transplant_date_upserts(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.put("/transplant-date/3", headers=headers, json={"fecha": "2026-05-20"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_supa.table.assert_called_with("transplant_dates")
    mock_supa.table.return_value.upsert.assert_called_with(
        {"greenhouse_id": 3, "fecha": "2026-05-20"}, on_conflict="greenhouse_id"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest backend/tests/test_transplant_dates.py -v`
Expected: `ModuleNotFoundError: No module named 'backend.routers.transplant_dates'` (or import error), all tests FAIL/ERROR.

- [ ] **Step 3: Add `mock_supa` patch target for the new router**

Modify `backend/tests/conftest.py` — inside the `mock_supa` fixture, add:

```python
    monkeypatch.setattr("backend.routers.transplant_dates.service_client", mock)
```

right after the `phenology_live` line (`backend/tests/conftest.py:56`).

- [ ] **Step 4: Write the router**

```python
# backend/routers/transplant_dates.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated
from backend.auth import get_current_user
from backend.supabase_client import service_client

router = APIRouter()


class TransplantDateRequest(BaseModel):
    fecha: str


@router.get("/transplant-date/{inv}")
def get_transplant_date(
    inv: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("transplant_dates")
        .select("greenhouse_id, fecha, updated_at")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        raise HTTPException(404, f"No hay fecha de transplante para invernadero {inv}")
    return row


@router.put("/transplant-date/{inv}")
def put_transplant_date(
    inv: int,
    body: TransplantDateRequest,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    service_client.table("transplant_dates").upsert(
        {"greenhouse_id": inv, "fecha": body.fecha}, on_conflict="greenhouse_id"
    ).execute()
    return {"ok": True}
```

- [ ] **Step 5: Register the router in `backend/main.py`**

Modify `backend/main.py:37` (import block, after the `uploads_router` import):

```python
from backend.routers import transplant_dates as transplant_dates_router  # noqa: E402
```

Modify `backend/main.py:69` (after `app.include_router(uploads_router.router)`):

```python
app.include_router(transplant_dates_router.router)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest backend/tests/test_transplant_dates.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/routers/transplant_dates.py backend/main.py backend/tests/conftest.py backend/tests/test_transplant_dates.py
git commit -m "feat(api): add GET/PUT /transplant-date/{inv} endpoints"
```

---

### Task 3: `season` query param on `/live-predictions/{inv}`

**Files:**
- Modify: `backend/routers/predictions.py:14-28`
- Test: `backend/tests/test_predictions.py`

**Interfaces:**
- Consumes: none new.
- Produces: `GET /live-predictions/{inv}?season=T18` filters rows to `season = 'T18'` in addition to existing `greenhouse_id`/`limit` filtering. `season` omitted → unfiltered (all seasons), matching current behavior.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_predictions.py`:

```python
def test_list_predictions_filters_by_season(api_client, mock_supa):
    headers = _auth(mock_supa)
    chain = mock_supa.table.return_value.select.return_value.eq.return_value
    chain.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {
            "id": 2, "greenhouse_id": 3, "predicted_for": "2026-08-04",
            "predicted_at": "2026-07-27T08:00:00Z", "kg_predicted": 12.0,
            "kg_actual": None, "model_version": "cnn_rnn_v2_production",
            "season": "T18",
        }
    ]
    r = api_client.get("/live-predictions/3?season=T18", headers=headers)
    assert r.status_code == 200
    assert r.json()[0]["season"] == "T18"
    mock_supa.table.return_value.select.return_value.eq.return_value.eq.assert_called_with("season", "T18")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_predictions.py::test_list_predictions_filters_by_season -v`
Expected: FAIL — extra `.eq("season", ...)` call never happens (current code has only one `.eq()` for `greenhouse_id`), so the assert on `chain.eq` fails / mock chain shape mismatch.

- [ ] **Step 3: Update the endpoint**

Modify `backend/routers/predictions.py` (replace lines 14–28):

```python
@router.get("/live-predictions/{inv}")
def list_predictions(
    inv: int,
    limit: int = Query(20, ge=1, le=200),
    season: str | None = Query(None),
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    query = (
        service_client.table("predictions")
        .select("id, greenhouse_id, predicted_for, predicted_at, kg_predicted, kg_actual, model_version, season")
        .eq("greenhouse_id", inv)
    )
    if season is not None:
        query = query.eq("season", season)
    resp = query.order("predicted_for", desc=True).limit(limit).execute()
    return resp.data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_predictions.py -v`
Expected: all tests PASS (including the pre-existing `test_list_predictions_returns_list`, whose mock chain `select().eq().order().limit().execute()` still resolves since `season=None` skips the extra `.eq()`).

- [ ] **Step 5: Commit**

```bash
git add backend/routers/predictions.py backend/tests/test_predictions.py
git commit -m "feat(api): add season filter to GET /live-predictions/{inv}"
```

---

### Task 4: Historical-mean precompute script

**Files:**
- Create: `scripts/compute_historical_means.py`
- Create (generated output, committed): `Models/results/historical_mean_inv3.json`, `Models/results/historical_mean_inv4.json`
- Test: `scripts/tests/test_compute_historical_means.py`

**Interfaces:**
- Consumes: `Models.cnn_rnn_yield.load_production(invernadero_id)` → DataFrame with columns `semana, kg_reales, temporada, invernadero` (one row per week per season, dropna'd).
- Produces: function `historical_means_by_position(df: pd.DataFrame, n_positions: int = 3) -> dict[int, float]`; JSON files `Models/results/historical_mean_inv{3,4}.json` shaped `{"0": <float>, "1": <float>, "2": <float>}`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/tests/test_compute_historical_means.py
import pandas as pd


def test_historical_means_by_position_averages_first_n_rows_per_season():
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13", "T13", "T13", "T14", "T14", "T14"],
        "kg_reales": [10.0, 20.0, 30.0, 30.0, 40.0, 50.0],
    })
    means = historical_means_by_position(df, n_positions=3)
    assert means == {0: 20.0, 1: 30.0, 2: 40.0}


def test_historical_means_by_position_ignores_rows_past_n(): 
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13"] * 5,
        "kg_reales": [1.0, 2.0, 3.0, 999.0, 999.0],
    })
    means = historical_means_by_position(df, n_positions=3)
    assert means == {0: 1.0, 1: 2.0, 2: 3.0}
```

Also create `scripts/tests/__init__.py` (empty file) so pytest discovers the package.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts/tests/test_compute_historical_means.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.compute_historical_means'`.

- [ ] **Step 3: Write the script**

```python
# scripts/compute_historical_means.py
"""
One-off: compute mean kg for the first `skip_first_weeks` positional weeks
of each season, per greenhouse, from the historical Kg Excel files.

These weeks are excluded from CNN-RNN training (see CLAUDE.md §8) — the
model was never trained to predict them, so live inference falls back to
this historical mean instead. Re-run manually only if the Kg Excel data
changes.

Usage: python scripts/compute_historical_means.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Models"))

from cnn_rnn_yield import load_production  # noqa: E402

RESULTS_DIR = ROOT / "Models" / "results"
N_POSITIONS = 3


def historical_means_by_position(df: pd.DataFrame, n_positions: int = N_POSITIONS) -> dict[int, float]:
    df = df.copy()
    df["pos"] = df.groupby("temporada").cumcount()
    subset = df[df["pos"] < n_positions]
    return subset.groupby("pos")["kg_reales"].mean().round(2).to_dict()


def main():
    for inv in (3, 4):
        df = load_production(inv)
        means = historical_means_by_position(df)
        out_path = RESULTS_DIR / f"historical_mean_inv{inv}.json"
        with open(out_path, "w") as f:
            json.dump({str(k): float(v) for k, v in means.items()}, f, indent=2)
        print(f"  Inv{inv} → {out_path}: {means}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts/tests/test_compute_historical_means.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Run the script to generate the committed JSON files**

Run: `python scripts/compute_historical_means.py`
Expected output: two lines like `  Inv3 → .../historical_mean_inv3.json: {0: ..., 1: ..., 2: ...}` and same for Inv4. Verify both JSON files now exist under `Models/results/`.

- [ ] **Step 6: Commit**

```bash
git add scripts/compute_historical_means.py scripts/tests/__init__.py scripts/tests/test_compute_historical_means.py Models/results/historical_mean_inv3.json Models/results/historical_mean_inv4.json
git commit -m "feat: precompute historical-mean fallback for season ramp-up weeks"
```

---

### Task 5: Fix `live_features.py` inv-blind defaults

**Files:**
- Modify: `backend/live_features.py:16` (module constant), `backend/live_features.py:109-111` (transplant date fallback)
- Test: `backend/tests/test_live_features.py`

**Interfaces:**
- Consumes: none new.
- Produces: `build_input_tensor(inv, wide_rows, pheno_rows, transplant_date, pipeline_path)` — `transplant_date` and `pipeline_path` become effectively required (no inv-blind defaults); callers (Task 6) always pass both explicitly.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_live_features.py`:

```python
def test_build_input_tensor_requires_explicit_transplant_date():
    from backend.live_features import build_input_tensor
    import pytest as pt
    pipeline = _make_pipeline(n_sensor_cols=3, seq_len=6)
    wide_rows = _wide_rows(60)
    with patch('backend.live_features.joblib.load', return_value=pipeline):
        with pt.raises(TypeError):
            build_input_tensor(4, wide_rows, [], pipeline_path="dummy.pkl")
```

This documents the fix: `transplant_date` has no default anymore, so
omitting it raises `TypeError`, instead of silently falling back to
`TRANSPLANT_DATE_INV3`/`date(2026,1,1)` regardless of which `inv` was
passed.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_live_features.py::test_build_input_tensor_requires_explicit_transplant_date -v`
Expected: FAIL — no `TypeError` raised, because `transplant_date=None` is currently accepted and defaulted internally.

- [ ] **Step 3: Remove the inv-blind defaults**

Modify `backend/live_features.py`. Replace the module-level default (line 16):

```python
PIPELINE_PATH = Path("Models/results/pipeline_inv3.pkl")
```

Delete this line entirely — `pipeline_path` is now a required keyword-only
argument (see signature change below).

Replace the transplant-date fallback block (currently lines ~108-111):

```python
    if transplant_date is None:
        raw = os.environ.get("TRANSPLANT_DATE_INV3", "")
        transplant_date = date.fromisoformat(raw) if raw else date(2026, 1, 1)
```

with nothing — delete it. Change the function signature from:

```python
def build_input_tensor(
    inv: int,
    wide_rows: list[dict],
    pheno_rows: list[dict],
    transplant_date: date | None = None,
    pipeline_path: Path = PIPELINE_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
```

to:

```python
def build_input_tensor(
    inv: int,
    wide_rows: list[dict],
    pheno_rows: list[dict],
    transplant_date: date,
    pipeline_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
```

Also remove the now-unused `import os` if nothing else in the file uses
`os` (check with `grep -n "os\." backend/live_features.py` — if no other
usage, delete the `import os` line).

- [ ] **Step 4: Update existing call site in `test_build_input_tensor_shape`**

`backend/tests/test_live_features.py::test_build_input_tensor_shape`
already passes `transplant_date=date(2026, 1, 5)` and relies on the
`PIPELINE_PATH` module default via `patch('backend.live_features.joblib.load', ...)`
— that patch means the real path value no longer matters (joblib.load is
mocked regardless of the path argument), but the call must now also pass
`pipeline_path` explicitly since it has no default. Update the call:

```python
        xs, xt = build_input_tensor(
            3, wide_rows, pheno_rows,
            transplant_date=date(2026, 1, 5),
            pipeline_path=Path("Models/results/pipeline_inv3.pkl"),
        )
```

Add `from pathlib import Path` to the top of the test file if not already
imported (check `backend/tests/test_live_features.py:1-6` — it isn't).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest backend/tests/test_live_features.py -v`
Expected: all tests PASS, including the new `TypeError` test.

- [ ] **Step 6: Commit**

```bash
git add backend/live_features.py backend/tests/test_live_features.py
git commit -m "fix: remove inv-blind pipeline/transplant-date defaults from build_input_tensor"
```

---

### Task 6: Rewrite `scripts/live_inference.py` — production models, inv4, transplant dates, historical-mean fallback, season

**Files:**
- Modify: `scripts/live_inference.py` (full rewrite)
- Test: `scripts/tests/test_live_inference.py`

**Interfaces:**
- Consumes: `backend.live_features.build_input_tensor(inv, wide_rows, pheno_rows, transplant_date, pipeline_path)` (Task 5); `Models.cnn_rnn_yield.CNNRNN`; `scripts/compute_historical_means.py`'s output JSON files (Task 4); `transplant_dates` table (Task 1/2).
- Produces: pure helper functions `monday_of_week(d)`, `week_in_season(target_date, transplant_date)`, `load_historical_mean(inv, wis) -> float | None` — all unit-testable without Supabase/torch. `main()` orchestrates per-greenhouse inference using these.

- [ ] **Step 1: Write the failing tests for the pure helpers**

```python
# scripts/tests/test_live_inference.py
import json
from datetime import date
from unittest.mock import patch


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
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 1) == 150.0


def test_load_historical_mean_returns_none_outside_ramp_up(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 3) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts/tests/test_live_inference.py -v`
Expected: FAIL — `week_in_season` and `load_historical_mean` don't exist yet in `scripts/live_inference.py`.

- [ ] **Step 3: Rewrite `scripts/live_inference.py`**

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

from backend.live_features import build_input_tensor

# Model class verified via:
# grep -n "^class" Models/cnn_rnn_yield.py  → CNNRNN at line 757
from Models.cnn_rnn_yield import CNNRNN

RESULTS_DIR = ROOT / "Models" / "results"
HORIZON_WEEKS = 5
MODEL_VERSION = "cnn_rnn_v2_production"
HISTORICAL_MEAN_VERSION = "historical_mean_v1"
SKIP_FIRST_WEEKS = 3
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


def run_inference_for_greenhouse(supa, inv: int, dry_run: bool) -> None:
    transplant_date = get_transplant_date(supa, inv)
    if transplant_date is None:
        print(f"  ERROR: no transplant_date set for invernadero {inv} — skipping. "
              f"Set it via PUT /transplant-date/{inv}.", file=sys.stderr)
        return

    predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS))
    wis_target = week_in_season(predicted_for, transplant_date)

    if wis_target < SKIP_FIRST_WEEKS:
        historical_kg = load_historical_mean(inv, wis_target)
        if historical_kg is not None:
            print(f"  Inv{inv} → week-in-season {wis_target} < {SKIP_FIRST_WEEKS} "
                  f"(ramp-up) → historical mean {historical_kg:.1f} kg")
            _upsert_prediction(supa, inv, predicted_for, historical_kg,
                                HISTORICAL_MEAN_VERSION, dry_run)
            return

    cutoff = (date.today() - timedelta(weeks=12)).isoformat()

    sensor_resp = (
        supa.table("sensor_readings_wide").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).execute()
    )
    riego_resp = (
        supa.table("riego_readings").select("*")
        .eq("greenhouse_id", inv).gte("fecha", cutoff).execute()
    )
    # exterior_readings is global (no greenhouse_id column) — same weather feeds both invernaderos
    ext_resp = (
        supa.table("exterior_readings").select("*")
        .gte("fecha", cutoff).execute()
    )

    merged_by_fecha: dict[str, dict] = {}
    for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
        merged_by_fecha.setdefault(row["fecha"], {}).update(row)
    wide_rows = list(merged_by_fecha.values())
    print(f"  Inv{inv} sensor rows pulled: {len(sensor_resp.data or [])} sensores + "
          f"{len(riego_resp.data or [])} riego + {len(ext_resp.data or [])} exteriores "
          f"→ {len(wide_rows)} merged by fecha")

    pheno_resp = (
        supa.table("phenology_observations").select("*")
        .eq("greenhouse_id", inv).gte("week_date", cutoff).execute()
    )
    pheno_rows = pheno_resp.data or []

    pipeline_path = RESULTS_DIR / f"production_pipeline_inv{inv}.pkl"
    x_sensor, x_temporal = build_input_tensor(
        inv, wide_rows, pheno_rows,
        transplant_date=transplant_date,
        pipeline_path=pipeline_path,
    )

    pipeline = joblib.load(pipeline_path)
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
    model.load_state_dict(torch.load(model_path, weights_only=True))
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

    print(f"  Inv{inv} → {kg_predicted:.1f} kg for week {predicted_for}")
    _upsert_prediction(supa, inv, predicted_for, kg_predicted, MODEL_VERSION, dry_run)


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
            run_inference_for_greenhouse(supa, inv, dry_run)
            succeeded += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: invernadero {inv} failed: {e}", file=sys.stderr)

    if succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts/tests/test_live_inference.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Manual dry-run smoke check**

Run (requires `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` env vars pointing at a
real or staging project with at least one `transplant_dates` row):
`DRY_RUN=1 python scripts/live_inference.py`
Expected: for each greenhouse, either a `[DRY RUN] would upsert ...` line or
an `ERROR: no transplant_date set ...` line — no unhandled traceback.

- [ ] **Step 6: Commit**

```bash
git add scripts/live_inference.py scripts/tests/test_live_inference.py
git commit -m "feat: production models, inv4 support, transplant dates from Supabase, ramp-up fallback"
```

---

### Task 7: Dashboard — season tabs (T17 / T18) per greenhouse view

**Files:**
- Modify: `demo/Demo Dashboard.html` (CSS ~line 1004; markup for `view-inv3` ~line 1325, `view-inv4` ~line 1426; JS ~lines 3050-3139)

**Interfaces:**
- Consumes: `GET /live-predictions/{inv}?season=T18` (Task 3), `GET /transplant-date/{inv}` / `PUT /transplant-date/{inv}` (Task 2).
- Produces: two new DOM containers per greenhouse — `season-panel-t17-{inv}` (wraps existing content) and `season-panel-t18-{inv}` (new). No changes to existing element IDs inside the T17 panel.

- [ ] **Step 1: Add season-tab CSS**

Add after the existing `.ins-tab, .hist-subtab` block (`demo/Demo Dashboard.html:1004-1023`):

```css
  .season-tabs { display: flex; gap: 8px; margin-bottom: 20px; }
  .season-tab {
    padding: 8px 18px; border-radius: 999px; border: 1px solid rgba(0,0,0,.12);
    background: transparent; font-size: 0.85rem; font-weight: 600; cursor: pointer;
  }
  .season-tab.active { background: var(--ink, #1e2a3a); color: #f4ede0; border-color: transparent; }
  .season-tab:not(.active):hover { background: rgba(0,0,0,.04); }
```

- [ ] **Step 2: Wrap `view-inv3` content in season panels**

Modify `demo/Demo Dashboard.html` around line 1325. Immediately after the
opening `<section class="view" id="view-inv3">` and its `detail-header` +
divider (keep those two elements outside/above the tabs — they're generic
page chrome, not T17-specific), insert season tabs and wrap the rest:

```html
    <div class="season-tabs">
      <button class="season-tab active" data-season="t17" data-inv="3">T17 (histórico)</button>
      <button class="season-tab" data-season="t18" data-inv="3">T18 (en vivo)</button>
    </div>

    <div id="season-panel-t17-3">
      <!-- existing .sim-bar, .detail-grid content stays here unchanged -->
```

Then, right before the closing `</section>` of `view-inv3` (currently
preceded by the "Predicción en vivo" block at lines 1416-1421), replace:

```html
    <div style="margin-top:32px">
      <h3 style="font-size:1.1rem;font-weight:600;margin-bottom:16px">Predicción en vivo</h3>
      <div id="forecast-cards-3" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px">
        <p style="color:#9aa0a8">Cargando predicciones…</p>
      </div>
    </div>

  </section>
```

with:

```html
    </div><!-- /season-panel-t17-3 -->

    <div id="season-panel-t18-3" hidden>
      <div class="panel">
        <div class="panel-title">Predicción en vivo — T18</div>
        <div id="forecast-cards-3" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
          <p style="color:#9aa0a8">Cargando predicciones…</p>
        </div>
      </div>
    </div>

  </section>
```

Repeat the equivalent edit for `view-inv4` (around line 1426 for the
opening wrap, and the `forecast-cards-4` block for the closing wrap),
using `data-inv="4"` and `season-panel-t17-4` / `season-panel-t18-4`.

- [ ] **Step 3: Wire season-tab click handling in JS**

Add near `initInsTabs()` (called from `init()`), a new function, and call
it from `init()` alongside the other `wire*`/`init*` calls:

```javascript
function initSeasonTabs() {
  document.querySelectorAll('.season-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const inv = btn.dataset.inv;
      const season = btn.dataset.season;
      document.querySelectorAll(`.season-tab[data-inv="${inv}"]`).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`season-panel-t17-${inv}`).hidden = season !== 't17';
      document.getElementById(`season-panel-t18-${inv}`).hidden = season !== 't18';
    });
  });
}
```

Find the `init()` function's body (search for `initInsTabs()` call) and add
`initSeasonTabs();` on the next line.

- [ ] **Step 4: Retarget live-prediction JS to the T18 panel, drop T17-array blending**

Modify `demo/Demo Dashboard.html` — delete the `extendSliderWithLive`
function entirely (lines ~3079-3125) and its call site inside
`refreshLiveData()`. Replace the whole block from
`renderForecastSection` (line 3050) through the end of `refreshLiveData()`
(line 3139) with:

```javascript
// ── Live Prediction Display (T18 tab) ─────────────────────────────────────
function renderForecastSection(inv, data) {
  const cards = document.getElementById(`forecast-cards-${inv}`);
  if (!cards) return;
  cards.innerHTML = data.length
    ? data.map(d => `
        <div style="background:#1e2a3a;border-radius:8px;padding:16px;min-width:150px;text-align:center">
          <div style="font-size:0.75rem;color:#9aa0a8;margin-bottom:6px">
            ${new Date(d.predicted_for + 'T12:00:00').toLocaleDateString('es-MX',{month:'short',day:'numeric'})}
          </div>
          <div style="font-size:1.4rem;font-weight:600;color:#f4ede0">${Math.round(d.kg_predicted).toLocaleString('es-MX')} kg</div>
          <div style="font-size:0.75rem;color:#9aa0a8;margin-top:6px">
            ${d.kg_actual == null ? 'Sin registrar' : `Real: ${Math.round(d.kg_actual).toLocaleString('es-MX')} kg`}
          </div>
          ${d.kg_actual == null ? `<button class="btn-ghost register-actual-btn" data-inv="${inv}" data-id="${d.id}" style="margin-top:8px;font-size:0.75rem">Registrar producción real</button>` : ''}
        </div>`).join('')
    : '<p style="color:#9aa0a8">Sin predicciones de T18 todavía.</p>';

  cards.querySelectorAll('.register-actual-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const kgStr = prompt('Producción real (kg):');
      if (kgStr == null) return;
      const kg = parseFloat(kgStr);
      if (Number.isNaN(kg)) return;
      const token = localStorage.getItem('sb_token');
      await fetch(`${API_BASE}/live-predictions/${btn.dataset.inv}/${btn.dataset.id}`, {
        method: 'PATCH',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ kg_actual: kg }),
      });
      refreshLiveData();
    });
  });
}

function renderMenuBadge(inv, data) {
  const el = document.getElementById(`menu-live-badge-${inv}`);
  if (!el) return;
  const next = data.find(d => d.kg_actual == null);
  if (!next) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML =
    `<span style="color:#3b82f6;font-size:0.75rem;font-weight:600">◉ Próxima</span>` +
    `<span style="font-size:0.85rem;color:#f4ede0;margin-left:8px">` +
    `${next.predicted_for} · ${Math.round(next.kg_predicted).toLocaleString('es-MX')} kg</span>`;
}

async function refreshLiveData() {
  for (const inv of [3, 4]) {
    const data = await fetchJSON(`/live-predictions/${inv}?season=T18&limit=20`);
    if (!data) continue;
    renderForecastSection(inv, data);
    renderMenuBadge(inv, data);
  }
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshLiveData();
});
```

- [ ] **Step 5: Add a transplant-date settings form**

In `view-usuarios` (the existing admin/users view, `demo/Demo Dashboard.html:1775`),
add a small section — read the existing markup structure of that view first
(`Read demo/Demo Dashboard.html` offset 1775 limit 60) to match its panel
style, then add:

```html
<div class="panel" style="margin-top:24px">
  <div class="panel-title">Fecha de transplante</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <div>
      <label style="display:block;font-size:0.8rem;margin-bottom:4px">Invernadero 3</label>
      <input type="date" id="transplant-date-3" />
      <button class="btn-primary" id="transplant-date-save-3" style="margin-left:8px">Guardar</button>
    </div>
    <div>
      <label style="display:block;font-size:0.8rem;margin-bottom:4px">Invernadero 4</label>
      <input type="date" id="transplant-date-4" />
      <button class="btn-primary" id="transplant-date-save-4" style="margin-left:8px">Guardar</button>
    </div>
  </div>
</div>
```

And the corresponding JS (add near `initSeasonTabs`, call from `init()`):

```javascript
async function initTransplantDateForm() {
  for (const inv of [3, 4]) {
    const data = await fetchJSON(`/transplant-date/${inv}`).catch(() => null);
    const input = document.getElementById(`transplant-date-${inv}`);
    if (data && input) input.value = data.fecha;

    document.getElementById(`transplant-date-save-${inv}`)?.addEventListener('click', async () => {
      const val = document.getElementById(`transplant-date-${inv}`).value;
      if (!val) return;
      const token = localStorage.getItem('sb_token');
      await fetch(`${API_BASE}/transplant-date/${inv}`, {
        method: 'PUT',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ fecha: val }),
      });
    });
  }
}
```

Add `initTransplantDateForm();` alongside the other `init*()` calls in
`init()`. Note `fetchJSON` currently treats a 404 response as a thrown
`Error` (`if (!r.ok) throw ...`) — the `.catch(() => null)` handles the
expected 404-when-unset case from Task 2 gracefully (empty input, no crash).

- [ ] **Step 6: Manual browser verification**

Run: `uvicorn backend.main:app --port 8000 --reload` (from repo root, with
`backend/.env`-equivalent env vars set), open `http://localhost:8000/app`.

Check in browser:
- `view-inv3` and `view-inv4` show a "T17 (histórico)" / "T18 (en vivo)" tab
  pair; T17 is active by default and looks identical to before (slider,
  chart, kg/m² KPIs).
- Clicking "T18 (en vivo)" hides the T17 panel and shows the (possibly
  empty) live-predictions cards.
- `view-usuarios` shows the two transplant-date inputs; entering a date and
  clicking "Guardar" does not error in the browser console (Network tab
  shows `PUT /transplant-date/{inv}` → 200, after auth is set up).

- [ ] **Step 7: Commit**

```bash
git add demo/"Demo Dashboard.html"
git commit -m "feat(dashboard): add T17/T18 season tabs, retarget live predictions off T17 replay, add transplant-date form"
```

---

### Task 8: Full test suite + spec cross-check

**Files:** none new — verification only.

- [ ] **Step 1: Run the full backend test suite**

Run: `cd backend && python -m pytest -v` (or `pytest backend/tests -v` from repo root, matching whatever the existing CI/dev convention is — check for a `pytest.ini`/`pyproject.toml` `testpaths` setting first with `grep -rn testpaths . --include=*.ini --include=*.toml` if unsure)

Expected: all tests pass, including every test added in Tasks 2, 3, 5.

- [ ] **Step 2: Run the scripts test suite**

Run: `pytest scripts/tests -v`
Expected: all tests pass, including every test added in Tasks 4 and 6.

- [ ] **Step 3: Cross-check against the spec**

Re-read `docs/superpowers/specs/2026-07-13-live-inference-production-design.md`
section by section and confirm each is implemented:
- Section 1 (production artifacts) → Task 6
- Section 2 (inv4 loop, `build_input_tensor` fix) → Tasks 5, 6
- Section 3 (transplant_dates table/endpoints) → Tasks 1, 2, 6
- Section 4 (historical-mean fallback) → Tasks 4, 6
- Section 5 (T17/T18 dashboard tabs, season column, no T17-array blending) → Tasks 1, 3, 7
- "What is NOT in scope" → confirm `backend/engine.py` and its endpoints
  were not touched (`git diff main --stat` should show no changes under
  `backend/engine.py`, `backend/main.py`'s `/inference`/`/metrics` routes,
  or `Models/`).

- [ ] **Step 4: Commit any fixes found during cross-check**

If Step 3 surfaces a gap, fix it, re-run the relevant test file, and commit
with a message describing exactly what was missing.
