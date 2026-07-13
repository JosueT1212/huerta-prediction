# Live Inference → Production — Design Spec

**Date:** 2026-07-13
**Status:** Approved

---

## Overview

Bring the live weekly inference pipeline (`scripts/live_inference.py` +
`backend/live_features.py`) up to production readiness. It currently:

- runs Invernadero 3 only (hardcoded `INV_ID = 3`)
- loads the eval checkpoint (`best_cnn_rnn_inv3.pt` / `pipeline_inv3.pkl`),
  not the production refit (`production_cnn_rnn_inv3.pt` /
  `production_pipeline_inv3.pkl`)
- reads transplant date from a Railway env var
  (`TRANSPLANT_DATE_INV3`), with no equivalent for Inv4
- has no fallback for the first `skip_first_weeks=3` weeks of a new season,
  which the model was never trained to predict (see CLAUDE.md §8)
- has no dashboard surface at all for the live weekly predictions it writes
  to the `predictions` table — `/live-predictions/{inv}` exists and works,
  but nothing in the dashboard calls it

This spec covers the changes needed to close all of the above.

Supabase schema is already aligned to the current split-table design
(migration `008_split_sensor_tables.sql`: `sensor_readings_wide`,
`riego_readings`, global `exterior_readings`) — `live_inference.py` already
queries these correctly. No further schema change needed there.

Note: `backend/engine.py` and its `/inference`, `/inference/window`,
`/metrics`, `/predictions` endpoints are a separate, unrelated system — the
sales-demo replay of the static T13–T17 eval snapshot (slider, PI bands,
kg/m² hero chart; see `demo/CLAUDE.md` §5). Out of scope here, see Section 5
and "What is NOT in scope" below.

---

## Section 1: Model artifacts — switch to production refit

`scripts/live_inference.py` and `backend/live_features.py` switch from the
eval checkpoint to the production refit, per greenhouse:

| | Before | After |
|---|---|---|
| Model weights | `Models/results/best_cnn_rnn_inv3.pt` | `Models/results/production_cnn_rnn_inv{inv}.pt` |
| Pipeline | `Models/results/pipeline_inv3.pkl` | `Models/results/production_pipeline_inv{inv}.pkl` |
| `model_version` | `"cnn_rnn_v1"` | `"cnn_rnn_v2_production"` |

Both production artifacts already exist on disk for inv3 and inv4 (see
`docs/superpowers/plans/2026-07-11-inv3-inv4-horizon5-production.md`). No
retraining needed.

`best_cnn_rnn_inv3.pt` / `pipeline_inv3.pkl` stay on disk unchanged — still
used by grid-search/eval scripts.

---

## Section 2: Invernadero 4 support

`scripts/live_inference.py` restructured to loop `for inv in (3, 4)`, with
everything currently hardcoded to `INV_ID = 3` made per-inv:

- model path, pipeline path, `hp_inv{inv}.yaml`
- sensor/riego/phenology queries already filter `.eq("greenhouse_id", inv)` —
  just need to run inside the loop
- `exterior_readings` query stays global (no `greenhouse_id` filter), shared
  across both iterations

`backend/live_features.py::build_input_tensor(inv, ...)` currently accepts
`inv` but never uses it — `PIPELINE_PATH` defaults to a hardcoded
`pipeline_inv3.pkl` and the transplant-date env var fallback is
`TRANSPLANT_DATE_INV3` regardless of `inv`. This is a latent bug (silently
wrong if ever called with `inv=4`) and gets fixed as part of this change:
pipeline path and transplant date are always passed in explicitly by the
caller, no per-function hardcoded default tied to inv3.

One inv failing (e.g. missing transplant date, see Section 4) does not block
the other — each iteration is wrapped so a failure logs and continues to the
next greenhouse. Script exits 0 if at least one greenhouse succeeded, non-zero
only if all failed.

---

## Section 3: Transplant date — move from env var to Supabase

New table:

```sql
-- supabase/migrations/009_transplant_dates.sql
create table if not exists transplant_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);
```

New endpoints (`backend/routers/transplant_dates.py`, pattern matching
`phenology_live.py`):

```
GET /transplant-date/{inv}   → { greenhouse_id, fecha, updated_at } or 404 if unset
PUT /transplant-date/{inv}   → upsert { fecha }; any authenticated user (JWT)
```

Small dashboard addition: one date-picker + save button per greenhouse
(admin/settings area), calling `PUT /transplant-date/{inv}`.

`TRANSPLANT_DATE_INV3` / `TRANSPLANT_DATE_INV4` env vars are removed.
`scripts/live_inference.py` reads the transplant date for each greenhouse
from `transplant_dates` at the start of each iteration.

**Missing transplant date** (table has no row for that `greenhouse_id`): that
greenhouse's inference is skipped for this run, with an `ERROR`-level log
line identifying which greenhouse and why. The other greenhouse still runs
and writes its prediction normally.

---

## Section 4: Historical-mean fallback for ramp-up weeks

Per CLAUDE.md §8: the first `skip_first_weeks=3` weeks of each new season
were excluded from training (harvest ramp-up, unreliable) — the model was
never trained to predict them and must not be used there. Those weeks need a
historical-mean fallback instead, implemented in the live pipeline (not the
training repo).

**One-off precompute script** — `scripts/compute_historical_means.py`:
- Reads `Data/Kg por semana T13 - T17 Invernadero {3,4}.xlsx`
- For each greenhouse, takes the first 3 weekly kg values of each season
  (position 0, 1, 2 — same positional convention as `skip_first_weeks` in
  training), averages each position across T13–T17
- Writes `Models/results/historical_mean_inv{3,4}.json`:
  ```json
  { "0": 123.4, "1": 156.7, "2": 189.2 }
  ```
- Run once now; re-run manually only if the underlying Kg data changes.

**At inference time**, for each greenhouse:
```
wis_target = (predicted_for - transplant_date).days // 7
if wis_target < 3:
    kg_predicted = historical_mean_inv{inv}.json[str(wis_target)]
    model_version = "historical_mean_v1"
else:
    kg_predicted = <CNN-RNN forward pass>
    model_version = "cnn_rnn_v2_production"
```
Both paths upsert into `predictions` the same way — `model_version`
distinguishes provenance for later debugging/auditing.

---

## Section 5: Dashboard — add a live "T18" season section, leave the T17 demo replay untouched

**Correction from an earlier draft of this section:** `backend/engine.py` /
`/inference/{inv}` / `/inference/{inv}/window` / `/metrics/{inv}` are **not**
a disposable eval artifact — they are the sales-demo replay of the full
T13–T17 slider (kg/m² hero chart, PI bands, KPI cards) documented in
`demo/CLAUDE.md` §5. None of that is touched by this spec.

`/live-predictions/{inv}` (GET/PATCH, `backend/routers/predictions.py`)
already exists, already reads/writes the `predictions` table directly, and
already updates weekly once `live_inference.py` runs both greenhouses — it
just isn't wired into any dashboard view yet. This section adds that UI.

**Season tracking.** `predictions` gets a `season` column:

```sql
-- part of supabase/migrations/009_transplant_dates.sql
alter table predictions add column if not exists season text;
```

`scripts/live_inference.py` sets `season` on every upsert from a constant,
`CURRENT_SEASON = "T18"`. Bumping to a future season (T19, …) is a one-line
constant change plus a new `transplant_dates` row — no other code changes
required.

**Dashboard.** Each greenhouse view (`view-inv3`, `view-inv4`) gets season
tabs above the existing content:

- **"T17" tab** (default) — exactly the existing slider/chart/kg-m²-hero
  content, unchanged.
- **"T18" tab** (new) — simple table/cards fed by
  `GET /live-predictions/{inv}?season=T18`: `predicted_for`, `kg_predicted`,
  `kg_actual` (or "—" if null), one row per week, growing weekly. A
  "Registrar producción real" button per row missing `kg_actual` opens a
  small form → `PATCH /live-predictions/{inv}/{id}`. No slider, no PI bands
  — there are only ever a handful of rows.

`GET /live-predictions/{inv}` gets an optional `season` query param
(defaults to returning all seasons, dashboard always passes `T18` explicitly
for this new tab).

Right after go-live, the T18 tab is nearly empty (one row per week as
`live_inference.py` runs) — that's expected, it fills in over the season.

---

## What is NOT in scope

- Retraining models on live data (weights stay fixed at the T13–T17
  production refit)
- Prediction intervals / confidence bands for live predictions
- Email/push notifications
- Changes to `Models/cnn_rnn_yield.py` or other training scripts
- Deleting `best_cnn_rnn_inv{3,4}.pt` / `pipeline_inv3.pkl` / the `.npz` +
  `_metrics.csv` eval artifacts — they remain for grid-search/eval use
- Touching `backend/engine.py`, `/inference/{inv}`, `/inference/{inv}/window`,
  `/metrics/{inv}`, or any T17-replay dashboard UI (slider, PI bands, kg/m²
  hero) — that is the sales-demo replay and stays exactly as-is
- Season rollover automation (T19 and beyond) — bumping `CURRENT_SEASON` and
  adding a `transplant_dates` row is a manual step when the next season starts

---

## Files touched (summary)

- `scripts/live_inference.py` — loop over both invs, production artifacts,
  transplant date from Supabase, historical-mean fallback
- `backend/live_features.py` — fix `inv`-blind pipeline/transplant-date
  defaults
- `scripts/compute_historical_means.py` — new, one-off
- `Models/results/historical_mean_inv{3,4}.json` — new, generated output
- `supabase/migrations/009_transplant_dates.sql` — new `transplant_dates`
  table + `predictions.season` column
- `backend/routers/transplant_dates.py` — new router
- `backend/routers/predictions.py` — add `season` query param to
  `GET /live-predictions/{inv}`; no removal
- Dashboard (`demo/Demo Dashboard.html`) — transplant-date form; new T17/T18
  season tabs per greenhouse view; new T18 tab UI (table/cards + "registrar
  producción real"). T17 tab and all existing `/inference`-backed content
  unchanged.
