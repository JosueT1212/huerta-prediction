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
and writes its prediction normally. Same skip behavior applies if a
transplant date exists but no sensor data has been uploaded yet this season
(see Section 4's corrected `harvest_start` derivation) — there's nothing to
run the model or the fallback on yet.

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

**Correction (post-implementation review, 2026-07-14):** the trigger below
originally used `wis_target = (predicted_for - transplant_date).days // 7`.
This is wrong — `transplant_date` marks when the plant went in the ground,
~10-11 weeks *before* harvest starts (CLAUDE.md §5), while the historical
means are indexed by weeks-since-harvest-*started* (`skip_first_weeks`
drops the first 3 *production* rows in training, not the first 3 weeks
post-transplant). Using `transplant_date` directly fires the fallback on
the wrong calendar weeks entirely.

**Corrected trigger, based on when the client actually starts uploading
data each season:** operationally, the client uploads no live sensor data
before harvest begins for a season — the first upload of a new season IS
harvest starting, and by the window-coverage validation
(`backend/routers/uploads.py`, `MIN_WEEKS_FIRST_BY_INV`) that first upload
already carries ≥ seq_len+horizon weeks of history. So "week harvest
started" = the earliest `fecha` present in `sensor_readings_wide` for that
greenhouse at or after the current `transplant_dates.fecha` (transplant
date still marks the season boundary, just not the harvest-start
reference point itself).

```
harvest_start = MIN(fecha) FROM sensor_readings_wide
                WHERE greenhouse_id = inv AND fecha >= transplant_date
# None if no data uploaded yet this season → skip this greenhouse (same as missing transplant_date)

wis_target = (predicted_for - harvest_start).days // 7
if wis_target < 3:
    kg_predicted = historical_mean_inv{inv}.json[str(wis_target)]
    model_version = "historical_mean_v1"
else:
    kg_predicted = <CNN-RNN forward pass>
    model_version = "cnn_rnn_v2_production"
```

`transplant_date` is still passed to `build_input_tensor` unchanged for the
model's temporal features (`dias_desde_transplante`, `week_in_season`) —
only the ramp-up *trigger* changes to use `harvest_start` instead.

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
already exists and already reads/writes the `predictions` table directly.
The dashboard already has live-prediction JS wired to it
(`refreshLiveData()`, `renderForecastSection()`, `renderMenuBadge()`,
`extendSliderWithLive()` — `demo/Demo Dashboard.html` lines ~3050–3139) —
but today `extendSliderWithLive()` appends each new live week directly onto
the same `rawInv[inv]` arrays that feed the T17 slider/chart, blending live
data into the historical replay with no season boundary.

This spec **removes that blending**: `extendSliderWithLive()` (and its
call in `refreshLiveData()`) is deleted. Live data moves into its own T18
tab instead of extending the T17 arrays. `renderForecastSection()` and
`renderMenuBadge()` are kept and retargeted at the new T18 tab's containers
instead of the old `forecast-cards-{inv}` / `menu-live-badge-{inv}` spots
(which lived directly on the T17 view) — see below.

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
tabs wrapping the existing content:

- **"T17" tab** (default) — exactly the existing slider/chart/kg-m²-hero
  content and markup, unchanged, minus the two elements that move to the
  T18 tab (see below).
- **"T18" tab** (new) — the relocated `forecast-cards-{inv}` (upcoming
  predictions) and `menu-live-badge-{inv}` pieces, now fed by
  `GET /live-predictions/{inv}?season=T18` instead of `?limit=20`: a card per
  row showing `predicted_for`, `kg_predicted`, and `kg_actual` ("—" if null),
  growing weekly. A "Registrar producción real" button per row missing
  `kg_actual` opens a small form → `PATCH /live-predictions/{inv}/{id}`. No
  slider, no PI bands — there are only ever a handful of rows.

`GET /live-predictions/{inv}` gets an optional `season` query param
(defaults to returning all seasons if omitted; dashboard always passes
`T18` explicitly for this tab).

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
