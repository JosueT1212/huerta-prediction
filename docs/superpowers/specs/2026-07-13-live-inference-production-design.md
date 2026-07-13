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
- feeds a dashboard (`backend/engine.py` + `/inference`, `/inference/window`,
  `/metrics`, `/predictions` endpoints) that replays a static, one-time T17
  eval snapshot (`.npz` + `_metrics.csv`) instead of live data

This spec covers the changes needed to close all of the above and retire the
now-redundant `/live-predictions/{inv}` endpoint pair.

Supabase schema is already aligned to the current split-table design
(migration `008_split_sensor_tables.sql`: `sensor_readings_wide`,
`riego_readings`, global `exterior_readings`) — `live_inference.py` already
queries these correctly. No further schema change needed there.

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

## Section 5: Dashboard — replace static eval replay with live data

`backend/engine.py` currently loads a static, one-time T17 eval snapshot
(`cnn_rnn_inv{3,4}_predictions.npz` + `_metrics.csv`) into memory at startup
and serves it unchanged (dashboard's "accuracy/history" view: windowed
scrubber, R²/RMSE/MAPE cards, PI bands).

This is rewritten to query the `predictions` table directly (no in-memory
cache needed — data volume is one row per greenhouse per week):

- `GET /inference/{inv}` — full prediction history for the greenhouse from
  `predictions`, ordered by `predicted_for`
- `GET /inference/{inv}/window` — windowed slice (cursor/horizon/past) over
  the same live data
- `GET /metrics/{inv}` — R²/RMSE/MAPE computed on the fly from rows where
  `kg_actual is not null` (need ≥2 such rows to compute; return nulls if
  fewer)
- `GET /predictions/{inv}` — compat alias, same as `/inference/{inv}`

Prediction-interval bands (`pi_lower`/`pi_upper`/`pi_coverage`/
`pi_avg_width`) are dropped — the live pipeline produces point estimates
only, no PI is computed at inference time. Dashboard UI elements depending on
PI bands are removed or hidden.

**`/live-predictions/{inv}` (GET/PATCH, `backend/routers/predictions.py`) is
removed** — redundant once the above endpoints read live data directly. The
"enter real production" (`kg_actual`) write path is folded into the
consolidated endpoints; exact request/response shape is decided at
implementation/plan time, keeping the same underlying `predictions` table
update.

Right after go-live, `predictions` will have very few rows per greenhouse
(no historical eval curve shown) — dashboard displays only what's actually in
the table, however sparse, and fills in week by week as `live_inference.py`
runs and users enter `kg_actual`.

---

## What is NOT in scope

- Retraining models on live data (weights stay fixed at the T13–T17
  production refit)
- Prediction intervals / confidence bands for live predictions
- Email/push notifications
- Changes to `Models/cnn_rnn_yield.py` or other training scripts
- Deleting `best_cnn_rnn_inv{3,4}.pt` / `pipeline_inv3.pkl` / the `.npz` +
  `_metrics.csv` eval artifacts — they remain for grid-search/eval use

---

## Files touched (summary)

- `scripts/live_inference.py` — loop over both invs, production artifacts,
  transplant date from Supabase, historical-mean fallback
- `backend/live_features.py` — fix `inv`-blind pipeline/transplant-date
  defaults
- `scripts/compute_historical_means.py` — new, one-off
- `Models/results/historical_mean_inv{3,4}.json` — new, generated output
- `supabase/migrations/009_transplant_dates.sql` — new table
- `backend/routers/transplant_dates.py` — new router
- `backend/engine.py` — rewritten to source from `predictions` table
- `backend/main.py` — no route signature changes, same 4 endpoints, new
  backing implementation
- `backend/routers/predictions.py` — `/live-predictions/{inv}` removed
- Dashboard (`demo/Demo Dashboard.html` or equivalent) — transplant-date
  form, PI-band UI removed, "enter kg_actual" wiring updated to new endpoint
  shape
