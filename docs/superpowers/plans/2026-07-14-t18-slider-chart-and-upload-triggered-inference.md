# T18 Slider/Chart Parity + Upload-Triggered Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the T18 ("en vivo") panel in the Predicción tab the same slider/chart/table exploration UX as T17, and make the backend fire live inference automatically once a week's non-production data (sensores + riego + fenología + exteriores) is fully uploaded, instead of waiting for the Sunday cron.

**Architecture:** Frontend: new slider + Chart.js line chart + prediction table inside `season-panel-t18-{3,4}` in `demo/Demo Dashboard.html`, driven by the already-fetched `rawLive[inv]` array (`GET /live-predictions/{inv}?season=T18`). Backend: a new `backend/inference_trigger.py` module checks `submission_locks` freshness across the 4 required upload types and, when all are within the 7-day window, calls the existing `scripts.live_inference.run_inference_for_greenhouse` synchronously from inside the upload request handler.

**Tech Stack:** FastAPI (Python) backend, vanilla JS + Chart.js 4.4 frontend (no build step), Supabase (postgrest client), pytest for backend tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-t18-slider-chart-and-upload-triggered-inference-design.md`
- `produccion` never gates the inference trigger — only sensores, riego, fenologia (per invernadero) + exteriores (global).
- Weekly Railway cron (`scripts/live_inference.py` standalone script) is NOT modified — it stays as the fallback.
- `run_inference_for_greenhouse(supa, inv, dry_run)` (`scripts/live_inference.py:87`) is reused as-is, no signature changes.
- T18 slider/chart/table must not touch `season-panel-t17-{inv}` markup or JS (`renderInv`, `buildSeriesFromAPI`, `wireSimControls` stay untouched).
- Follow existing repo test patterns: mock `service_client` via the `mock_supa` fixture (`backend/tests/conftest.py`), not real Supabase calls.
- Run backend tests with `./.venv/bin/python -m pytest backend/tests/ -q` (confirmed working: torch/joblib available in `.venv`).

---

### Task 1: Expose `last_submitted_at` from `backend/lock_utils.py`

**Files:**
- Modify: `backend/lock_utils.py:8-19,22-28,42-48`
- Test: `backend/tests/test_lock_utils.py`

**Interfaces:**
- Produces: `last_submitted_at(greenhouse_id: int, form_type: str) -> datetime | None` (public — was private `_fetch_last_submitted_at`). Task 2 imports this.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_lock_utils.py` (append at end of file):

```python
def test_last_submitted_at_returns_datetime_when_present(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    result = lock_utils.last_submitted_at(3, "sensores")
    assert result is not None
    assert result.tzinfo is not None


def test_last_submitted_at_returns_none_when_absent(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    assert lock_utils.last_submitted_at(3, "sensores") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest backend/tests/test_lock_utils.py -q`
Expected: FAIL — `AttributeError: module 'backend.lock_utils' has no attribute 'last_submitted_at'`

- [ ] **Step 3: Rename the function in `backend/lock_utils.py`**

Replace:
```python
def _fetch_last_submitted_at(greenhouse_id: int, form_type: str) -> datetime | None:
```
with:
```python
def last_submitted_at(greenhouse_id: int, form_type: str) -> datetime | None:
```

Replace (in `check_submission_lock`):
```python
def check_submission_lock(greenhouse_id: int, form_type: str) -> None:
    last = _fetch_last_submitted_at(greenhouse_id, form_type)
```
with:
```python
def check_submission_lock(greenhouse_id: int, form_type: str) -> None:
    last = last_submitted_at(greenhouse_id, form_type)
```

Replace (in `get_lock_status`):
```python
def get_lock_status(greenhouse_id: int, form_type: str) -> dict:
    last = _fetch_last_submitted_at(greenhouse_id, form_type)
```
with:
```python
def get_lock_status(greenhouse_id: int, form_type: str) -> dict:
    last = last_submitted_at(greenhouse_id, form_type)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/tests/test_lock_utils.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/lock_utils.py backend/tests/test_lock_utils.py
git commit -m "refactor(backend): expose lock_utils.last_submitted_at publicly

Needed by the upcoming upload-triggered inference check, which reads
last-submission timestamps across 4 form types independently of the
existing lock/unlock logic.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `backend/inference_trigger.py` — completeness check + trigger

**Files:**
- Create: `backend/inference_trigger.py`
- Test: `backend/tests/test_inference_trigger.py`

**Interfaces:**
- Consumes: `backend.lock_utils.last_submitted_at(greenhouse_id, form_type) -> datetime | None` (Task 1), `backend.lock_utils.LOCK_WINDOW` (existing, `timedelta(days=7)`), `scripts.live_inference.run_inference_for_greenhouse(supa, inv, dry_run) -> bool` (existing, unmodified).
- Produces: `uploads_complete_for_inv(inv: int) -> bool`, `maybe_trigger_inference(form_type: str, inv: int | None) -> None`. Task 3 imports both (well, only `maybe_trigger_inference` — `uploads_complete_for_inv` is used internally and by tests).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_inference_trigger.py`:

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


def _stub_locks(monkeypatch, times: dict):
    """times: {(greenhouse_id, form_type): iso_str_or_None}"""
    from backend import inference_trigger

    def fake_last_submitted_at(gh_id, form_type):
        iso = times.get((gh_id, form_type))
        return datetime.fromisoformat(iso) if iso else None

    monkeypatch.setattr(inference_trigger, "last_submitted_at", fake_last_submitted_at)


def test_uploads_complete_for_inv_true_when_all_four_recent(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): now, (3, "riego"): now, (3, "fenologia"): now,
        (0, "exteriores"): now,
    })
    assert uploads_complete_for_inv(3) is True


def test_uploads_complete_for_inv_false_when_one_missing(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): now, (3, "riego"): now, (3, "fenologia"): None,
        (0, "exteriores"): now,
    })
    assert uploads_complete_for_inv(3) is False


def test_uploads_complete_for_inv_false_when_expired(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): old, (3, "riego"): now, (3, "fenologia"): now,
        (0, "exteriores"): now,
    })
    assert uploads_complete_for_inv(3) is False


def test_maybe_trigger_inference_runs_when_complete(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: True)
    run_mock = MagicMock(return_value=True)
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("fenologia", 3)

    run_mock.assert_called_once()
    assert run_mock.call_args[0][1] == 3
    assert run_mock.call_args[1]["dry_run"] is False


def test_maybe_trigger_inference_skips_when_incomplete(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: False)
    run_mock = MagicMock()
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("sensores", 3)

    run_mock.assert_not_called()


def test_maybe_trigger_inference_checks_both_invs_for_exteriores(monkeypatch):
    from backend import inference_trigger
    checked = []
    monkeypatch.setattr(
        inference_trigger, "uploads_complete_for_inv",
        lambda inv: checked.append(inv) or True,
    )
    run_mock = MagicMock()
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("exteriores", None)

    assert checked == [3, 4]
    assert run_mock.call_count == 2


def test_maybe_trigger_inference_swallows_inference_errors(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: True)
    run_mock = MagicMock(side_effect=RuntimeError("boom"))
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("riego", 4)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/tests/test_inference_trigger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.inference_trigger'`

- [ ] **Step 3: Write `backend/inference_trigger.py`**

```python
"""Fires live inference right after the last required weekly upload lands.

Each of the 4 required types (sensores/riego/fenologia per invernadero,
exteriores global) can only be submitted once per LOCK_WINDOW — a repeat
submission is rejected by check_submission_lock() with 429 before it ever
reaches this module. That means uploads_complete_for_inv() can only flip
from False to True on the exact upload call that completes the set; no
extra "was it already complete" bookkeeping is needed here.
"""
from datetime import datetime, timezone
import sys

from backend.lock_utils import LOCK_WINDOW, last_submitted_at

# Mirrors the sentinel in backend/routers/uploads.py (EXTERIORES_GH_ID) —
# exteriores has no real invernadero, 0 is never a real greenhouse id.
EXTERIORES_GH_ID = 0

REQUIRED_PER_INV_TYPES = ("sensores", "riego", "fenologia")


def uploads_complete_for_inv(inv: int) -> bool:
    """True if sensores+riego+fenologia (this inv) + exteriores (global)
    were all submitted within the current LOCK_WINDOW."""
    now = datetime.now(timezone.utc)
    for form_type in REQUIRED_PER_INV_TYPES:
        ts = last_submitted_at(inv, form_type)
        if ts is None or now - ts > LOCK_WINDOW:
            return False
    ts = last_submitted_at(EXTERIORES_GH_ID, "exteriores")
    if ts is None or now - ts > LOCK_WINDOW:
        return False
    return True


def maybe_trigger_inference(form_type: str, inv: int | None) -> None:
    """Call after a successful upload that touched a submission lock.
    Runs inference for any invernadero whose required-upload set just
    became complete. Callers must not invoke this for form_type ==
    "produccion" — production never gates inference."""
    from backend.supabase_client import service_client
    from scripts.live_inference import run_inference_for_greenhouse

    invs_to_check = (3, 4) if form_type == "exteriores" else (inv,)
    for check_inv in invs_to_check:
        if not uploads_complete_for_inv(check_inv):
            continue
        try:
            run_inference_for_greenhouse(service_client, check_inv, dry_run=False)
        except Exception as e:  # noqa: BLE001
            print(
                f"  ERROR: upload-triggered inference failed for invernadero "
                f"{check_inv}: {e}",
                file=sys.stderr,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/tests/test_inference_trigger.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/inference_trigger.py backend/tests/test_inference_trigger.py
git commit -m "feat(backend): add upload-completeness check + inference trigger

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire the trigger into `backend/routers/uploads.py`

**Files:**
- Modify: `backend/routers/uploads.py:1-9,217-228,280-291`
- Test: `backend/tests/test_uploads.py`

**Interfaces:**
- Consumes: `maybe_trigger_inference(form_type: str, inv: int | None) -> None` (Task 2).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_uploads.py`:

```python
def test_upload_sensores_triggers_inference_check(api_client, mock_supa, monkeypatch):
    from backend.routers import uploads as uploads_router
    trigger_mock = MagicMock()
    monkeypatch.setattr(uploads_router, "maybe_trigger_inference", trigger_mock)
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(9)
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    trigger_mock.assert_called_once_with("sensores", 3)


def test_upload_produccion_does_not_trigger_inference(api_client, mock_supa, monkeypatch):
    from backend.routers import uploads as uploads_router
    trigger_mock = MagicMock()
    monkeypatch.setattr(uploads_router, "maybe_trigger_inference", trigger_mock)
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    mock_supa.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{"id": 1}]
    df = pd.DataFrame([{"fecha": "2026-03-02", "kg_reales": 100.0}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 200
    trigger_mock.assert_not_called()


def test_upload_exteriores_triggers_inference_check(api_client, mock_supa, monkeypatch):
    from backend.routers import uploads as uploads_router
    trigger_mock = MagicMock()
    monkeypatch.setattr(uploads_router, "maybe_trigger_inference", trigger_mock)
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa, per_inv=False)
    df = _exterior_rows(9)
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 200
    trigger_mock.assert_called_once_with("exteriores", None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/tests/test_uploads.py -q`
Expected: FAIL — `AttributeError: <module 'backend.routers.uploads'> does not have the attribute 'maybe_trigger_inference'`

- [ ] **Step 3: Add the import**

In `backend/routers/uploads.py`, replace:
```python
from backend.lock_utils import check_submission_lock, touch_submission_lock, get_lock_status
```
with:
```python
from backend.lock_utils import check_submission_lock, touch_submission_lock, get_lock_status
from backend.inference_trigger import maybe_trigger_inference
```

- [ ] **Step 4: Wire the trigger in `upload_excel`**

Replace:
```python
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
```
with:
```python
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, form_type, inv)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(inv, form_type)
        if form_type != "produccion":
            maybe_trigger_inference(form_type, inv)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/{inv}/{form_type}/history")
```

- [ ] **Step 5: Wire the trigger in `upload_exteriores`**

Replace:
```python
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
```
with:
```python
    rows_inserted, rows_updated, rows_skipped, skipped_reasons = _ingest_rows(df, "exteriores", None)

    if rows_inserted + rows_updated > 0:
        touch_submission_lock(EXTERIORES_GH_ID, "exteriores")
        maybe_trigger_inference("exteriores", None)

    return {
        "rows_in_file": len(df),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "rows_skipped": rows_skipped,
        "skipped_reasons": skipped_reasons,
    }


@router.get("/uploads/exteriores/history")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/tests/test_uploads.py -q`
Expected: PASS (31 tests)

- [ ] **Step 7: Run the full backend suite**

Run: `./.venv/bin/python -m pytest backend/tests/ scripts/tests/ -q`
Expected: PASS, no regressions

- [ ] **Step 8: Commit**

```bash
git add backend/routers/uploads.py backend/tests/test_uploads.py
git commit -m "feat(backend): trigger live inference right after uploads complete

Fires scripts.live_inference.run_inference_for_greenhouse synchronously
once sensores+riego+fenologia+exteriores are all submitted within the
same 7-day window. produccion is excluded — it's a verification value,
not a model input. Weekly Railway cron stays as fallback.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: T18 slider/chart/table markup in `demo/Demo Dashboard.html`

**Files:**
- Modify: `demo/Demo Dashboard.html:1427-1434` (inv3 `season-panel-t18-3`), `demo/Demo Dashboard.html:1537-1544` (inv4 `season-panel-t18-4`)

**Interfaces:**
- Produces DOM ids Task 5's JS binds to: `live-sim-bar-{inv}`, `simweek{inv}-t18`, `simcur{inv}-t18`, `prev{inv}-t18`, `slider{inv}-t18`, `next{inv}-t18`, `live-detail-grid-{inv}`, `chart-inv{inv}-t18`, `tbody-inv{inv}-t18`, `live-empty-{inv}`.
- Reuses existing CSS classes only (`.sim-bar`, `.sim-head`, `.sim-kicker`, `.sim-week`, `.sim-pos`, `.sim-track-row`, `.sim-step`, `.sim-slider`, `.detail-grid`, `.panel`, `.panel-dark`, `.chart-container`, `.pred-table-wrap`, `.pred-header`, `.click-hint`) — no new CSS needed.

- [ ] **Step 1: Insert markup for inv3**

In `demo/Demo Dashboard.html`, replace:
```html
    <div id="season-panel-t18-3" hidden>
      <div class="panel">
        <div class="panel-title">Predicción en vivo — T18</div>
        <div id="forecast-cards-3" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
          <p style="color:#9aa0a8">Cargando predicciones…</p>
        </div>
      </div>
    </div>

  </section>

  <!-- ============ INV4 DETAIL VIEW ============ -->
```
with:
```html
    <div id="season-panel-t18-3" hidden>
      <div class="sim-bar" id="live-sim-bar-3" hidden>
        <div class="sim-head">
          <div>
            <div class="sim-kicker">◆ Explorar predicciones T18 · arrastra para elegir la semana</div>
            <div class="sim-week" id="simweek3-t18">—</div>
          </div>
          <div class="sim-pos">Semana <b id="simcur3-t18">—</b></div>
        </div>
        <div class="sim-track-row">
          <button class="sim-step" id="prev3-t18" title="Semana anterior">‹</button>
          <input type="range" class="sim-slider" id="slider3-t18" min="0" max="0" value="0" />
          <button class="sim-step" id="next3-t18" title="Semana siguiente">›</button>
        </div>
      </div>

      <div class="detail-grid" id="live-detail-grid-3" hidden>
        <div class="panel panel-dark">
          <div class="panel-title">Predicción vs Real · T18</div>
          <div class="chart-container"><canvas id="chart-inv3-t18"></canvas></div>
        </div>
        <div class="panel pred-table-wrap" data-goto="hist3">
          <div class="pred-header">
            <div class="panel-title" style="margin:0">Semanas T18</div>
            <div class="click-hint">Ver histórico →</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Semana</th>
                <th class="num">Predicho (kg)</th>
                <th class="num">Real (kg)</th>
              </tr>
            </thead>
            <tbody id="tbody-inv3-t18"></tbody>
          </table>
        </div>
      </div>

      <p id="live-empty-3" style="color:#9aa0a8">Aún no hay predicciones para T18.</p>

      <div class="panel">
        <div class="panel-title">Predicción en vivo — T18</div>
        <div id="forecast-cards-3" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
          <p style="color:#9aa0a8">Cargando predicciones…</p>
        </div>
      </div>
    </div>

  </section>

  <!-- ============ INV4 DETAIL VIEW ============ -->
```

- [ ] **Step 2: Insert markup for inv4**

In `demo/Demo Dashboard.html`, replace:
```html
    <div id="season-panel-t18-4" hidden>
      <div class="panel">
        <div class="panel-title">Predicción en vivo — T18</div>
        <div id="forecast-cards-4" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
          <p style="color:#9aa0a8">Cargando predicciones…</p>
        </div>
      </div>
    </div>

  </section>

  <!-- ============ KPI INV3 VIEW ============ -->
```
with:
```html
    <div id="season-panel-t18-4" hidden>
      <div class="sim-bar" id="live-sim-bar-4" hidden>
        <div class="sim-head">
          <div>
            <div class="sim-kicker">◆ Explorar predicciones T18 · arrastra para elegir la semana</div>
            <div class="sim-week" id="simweek4-t18">—</div>
          </div>
          <div class="sim-pos">Semana <b id="simcur4-t18">—</b></div>
        </div>
        <div class="sim-track-row">
          <button class="sim-step" id="prev4-t18" title="Semana anterior">‹</button>
          <input type="range" class="sim-slider" id="slider4-t18" min="0" max="0" value="0" />
          <button class="sim-step" id="next4-t18" title="Semana siguiente">›</button>
        </div>
      </div>

      <div class="detail-grid" id="live-detail-grid-4" hidden>
        <div class="panel panel-dark">
          <div class="panel-title">Predicción vs Real · T18</div>
          <div class="chart-container"><canvas id="chart-inv4-t18"></canvas></div>
        </div>
        <div class="panel pred-table-wrap" data-goto="hist4">
          <div class="pred-header">
            <div class="panel-title" style="margin:0">Semanas T18</div>
            <div class="click-hint">Ver histórico →</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Semana</th>
                <th class="num">Predicho (kg)</th>
                <th class="num">Real (kg)</th>
              </tr>
            </thead>
            <tbody id="tbody-inv4-t18"></tbody>
          </table>
        </div>
      </div>

      <p id="live-empty-4" style="color:#9aa0a8">Aún no hay predicciones para T18.</p>

      <div class="panel">
        <div class="panel-title">Predicción en vivo — T18</div>
        <div id="forecast-cards-4" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
          <p style="color:#9aa0a8">Cargando predicciones…</p>
        </div>
      </div>
    </div>

  </section>

  <!-- ============ KPI INV3 VIEW ============ -->
```

- [ ] **Step 3: Sanity-check the HTML is well-formed**

Run: `grep -c '<section class="view"' "demo/Demo Dashboard.html"` and `grep -c '</section>' "demo/Demo Dashboard.html"`
Expected: both counts equal (unchanged from before this edit — this step only adds `<div>`s, no new `<section>`).

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add T18 slider/chart/table markup, mirrors T17 layout

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: T18 slider/chart/table JS

**Files:**
- Modify: `demo/Demo Dashboard.html` (JS `<script>` block — insert after `wireSimControls` at line 2377, and wire calls into `init()` around line 2860-2877 and `refreshLiveData()` around line 3035-3053)

**Interfaces:**
- Consumes: `rawLive[inv]` (existing, populated by `loadLiveSeason`), `palette.accent`/`palette.cream` (existing, `demo/Demo Dashboard.html:1843`), `charts`/`destroyChart` (existing, `2102-2103`), `chartBaseOpts` (existing, `2191`), `fmt`/`fmtLiveWeekLabel`/`setText` (existing), `PAST_WINDOW` (existing, `= 5`).
- Produces: `buildLiveSeriesFromRaw(rows, cursor)`, `buildLivePredTable(tbody, data)`, `makeLiveDetailChart(ctxId, history, predictions, color)`, `updateLiveSliderFill(inv, cursor, n)`, `renderLiveInv(inv, cursor)`, `stepLiveInv(inv, delta)`, `wireLiveSimControls(inv)`.

- [ ] **Step 1: Add the series builder, table builder, and chart function**

In `demo/Demo Dashboard.html`, after the `wireSimControls` function (ends line 2377, right before the `FENOLOGÍA` section comment at line 2379), insert:

```javascript
/* =========================================================
   T18 (EN VIVO) — slider/chart/tabla, paralelo a T17 pero
   sobre rawLive[inv] (sin PI bounds, longitud variable).
========================================================= */
const cursorLiveInv = { 3: null, 4: null };

/* Recorta rawLive[inv] alrededor del cursor: past=5 semanas antes,
   el resto (cursor→fin) como "predictions" (semanas ya predichas,
   algunas con kg_actual ya registrado). */
function buildLiveSeriesFromRaw(rows, cursor) {
  const startPast = Math.max(0, cursor - PAST_WINDOW);
  const history = [];
  for (let i = startPast; i < cursor; i++) {
    history.push({
      week: fmtLiveWeekLabel(rows[i].predicted_for),
      real: rows[i].kg_actual,
      pred: rows[i].kg_predicted,
    });
  }
  const predictions = [];
  for (let i = cursor; i < rows.length; i++) {
    predictions.push({
      week: fmtLiveWeekLabel(rows[i].predicted_for),
      real: rows[i].kg_actual,
      pred: rows[i].kg_predicted,
    });
  }
  return { history, predictions };
}

function buildLivePredTable(tbody, data) {
  if (!tbody) return;
  tbody.innerHTML = data.map(r => `
    <tr>
      <td class="label">${r.week}</td>
      <td class="num">${fmt(r.pred)}</td>
      <td class="num">${fmt(r.real)}</td>
    </tr>
  `).join('');
}

/* Como makeDetailChart pero sin forzar "Real" a null en las filas
   futuras — T18 puede tener kg_actual ya registrado dentro del rango
   de "predictions" (semana pasada re-visitada con el slider). */
function makeLiveDetailChart(ctxId, history, predictions, color) {
  destroyChart(ctxId);
  const el = document.getElementById(ctxId);
  if (!el) return;
  const ctx = el.getContext('2d');
  const allRows = [...history, ...predictions];
  const labels = allRows.map(r => r.week);
  const realData = allRows.map(r => r.real);
  const predData = allRows.map(r => r.pred);

  charts[ctxId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Predicción',
          data: predData,
          borderColor: color,
          borderWidth: 2,
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          pointBackgroundColor: color,
          pointRadius: 3,
          tension: 0.35,
          fill: false,
        },
        {
          label: 'Real',
          data: realData,
          borderColor: palette.cream,
          borderWidth: 2.5,
          backgroundColor: 'transparent',
          pointBackgroundColor: palette.cream,
          pointRadius: 4,
          pointHoverRadius: 7,
          tension: 0.3,
          fill: false,
        },
      ],
    },
    options: chartBaseOpts(false),
  });
}

function updateLiveSliderFill(inv, cursor, n) {
  const sl = document.getElementById(`slider${inv}-t18`);
  if (!sl) return;
  const pct = n > 1 ? (cursor / (n - 1)) * 100 : 0;
  sl.style.setProperty('--fill', pct.toFixed(1) + '%');
}

/* Repinta slider+chart+tabla T18 para un invernadero en la semana `cursor`.
   cursor=null → última semana con predicción (lo más reciente). */
function renderLiveInv(inv, cursor) {
  const rows = rawLive[inv] || [];
  const n = rows.length;
  const simBar = document.getElementById(`live-sim-bar-${inv}`);
  const grid = document.getElementById(`live-detail-grid-${inv}`);
  const empty = document.getElementById(`live-empty-${inv}`);

  if (!n) {
    if (simBar) simBar.hidden = true;
    if (grid) grid.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  if (simBar) simBar.hidden = false;
  if (grid) grid.hidden = false;
  if (empty) empty.hidden = true;

  if (cursor == null) cursor = cursorLiveInv[inv] ?? (n - 1);
  cursor = Math.max(0, Math.min(cursor, n - 1));
  cursorLiveInv[inv] = cursor;

  const { history, predictions } = buildLiveSeriesFromRaw(rows, cursor);
  buildLivePredTable(document.getElementById(`tbody-inv${inv}-t18`), predictions);
  makeLiveDetailChart(`chart-inv${inv}-t18`, history, predictions, palette.accent);

  const sl = document.getElementById(`slider${inv}-t18`);
  if (sl) {
    sl.max = n - 1;
    sl.value = cursor;
    sl.disabled = n <= 1;
  }
  updateLiveSliderFill(inv, cursor, n);
  setText(`simweek${inv}-t18`, fmtLiveWeekLabel(rows[cursor].predicted_for));
  setText(`simcur${inv}-t18`, `${cursor + 1} / ${n}`);
}

function stepLiveInv(inv, delta) {
  const n = (rawLive[inv] || []).length;
  if (!n) return;
  renderLiveInv(inv, Math.max(0, Math.min((cursorLiveInv[inv] ?? (n - 1)) + delta, n - 1)));
}

function wireLiveSimControls(inv) {
  const sl = document.getElementById(`slider${inv}-t18`);
  if (sl) sl.addEventListener('input', () => renderLiveInv(inv, +sl.value));
  document.getElementById(`prev${inv}-t18`)?.addEventListener('click', () => stepLiveInv(inv, -1));
  document.getElementById(`next${inv}-t18`)?.addEventListener('click', () => stepLiveInv(inv, +1));
}
```

- [ ] **Step 2: Wire it into `init()`**

In `demo/Demo Dashboard.html`, replace:
```javascript
    fillMenuKPIs(3);
    fillMenuKPIs(4);
    buildLiveHistTable(document.getElementById('tbody-hist3'), rawLive[3] || []);
    buildLiveHistTable(document.getElementById('tbody-hist4'), rawLive[4] || []);
    refreshLiveData();
```
with:
```javascript
    fillMenuKPIs(3);
    fillMenuKPIs(4);
    buildLiveHistTable(document.getElementById('tbody-hist3'), rawLive[3] || []);
    buildLiveHistTable(document.getElementById('tbody-hist4'), rawLive[4] || []);
    wireLiveSimControls(3);
    wireLiveSimControls(4);
    renderLiveInv(3, null);
    renderLiveInv(4, null);
    refreshLiveData();
```

- [ ] **Step 3: Wire it into `refreshLiveData()` so the slider stays in sync with polling**

In `demo/Demo Dashboard.html`, replace:
```javascript
      await loadLiveSeason(inv);
      fillMenuKPIs(inv);
      buildLiveHistTable(document.getElementById(`tbody-hist${inv}`), rawLive[inv] || []);
    }
    renderYieldGoal(3); renderYieldGoal(4);
```
with:
```javascript
      await loadLiveSeason(inv);
      fillMenuKPIs(inv);
      buildLiveHistTable(document.getElementById(`tbody-hist${inv}`), rawLive[inv] || []);
      renderLiveInv(inv, cursorLiveInv[inv]);
    }
    renderYieldGoal(3); renderYieldGoal(4);
```

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): wire T18 slider/chart/table JS, synced with live polling

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Manual verification in the browser

**Files:** none (verification only)

- [ ] **Step 1: Start the backend**

Run: `cd "/Users/josuetapiahernandez/Documents/JATA/Huerta_Prediction" && ./.venv/bin/uvicorn backend.main:app --port 8000 --reload`
Expected: log line `Application startup complete` with no errors, and `Uvicorn running on http://0.0.0.0:8000`.

- [ ] **Step 2: Open the dashboard and log in**

Open `http://localhost:8000/` in a browser, log in with any username/password (cosmetic gate per `demo/CLAUDE.md` §3), land on `/app`.

- [ ] **Step 3: Check T18 panel — has data case**

Click sidebar "Predicción" → Invernadero 3, click the "T18 (en vivo)" season tab.
Expected: if `rawLive[3]` has ≥1 row (check via `GET http://localhost:8000/live-predictions/3?season=T18` in a new tab), the slider bar, "Predicción vs Real · T18" chart, and "Semanas T18" table are visible; dragging the slider updates the chart/table/week label; clicking the table panel navigates to Histórico (T18) tab.
If `rawLive[3]` is empty: confirm the "Aún no hay predicciones para T18." message shows instead of a broken/empty chart.

- [ ] **Step 4: Repeat for Invernadero 4**

Same checks as Step 3 but for `view-inv4` / `slider4-t18` / `chart-inv4-t18`.

- [ ] **Step 5: Check browser console for errors**

Open DevTools console (F12). Expected: no red errors related to `renderLiveInv`, `makeLiveDetailChart`, `Cannot read properties of null`, or Chart.js canvas reuse warnings, while switching season tabs and dragging both sliders.

- [ ] **Step 6: Verify upload-triggered inference (backend only, no UI)**

With the backend running, use the "Insertar datos" tab to upload sensores/riego/fenología/exteriores `.xlsx` files (one invernadero) covering ≥9 weeks each (or use existing test fixtures under `Data/sample_uploads/` if present). After the last of the 4 required types is submitted, check the `uvicorn` terminal log for either a successful inference line (`Inv{n} → ... kg for week ...`) or an `ERROR: upload-triggered inference failed for invernadero {n}: ...` line — either confirms the trigger fired. Confirm it does NOT fire again on a subsequent `producción` upload for the same week (no new inference log line after that specific upload).
