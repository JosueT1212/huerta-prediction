# Split Sensores/Riego/Exteriores Uploads — Design

**Date:** 2026-07-11
**Status:** Approved by user, pending implementation plan

## Context

The excel-upload feature (`docs/superpowers/specs/2026-07-09-insertar-datos-excel-upload-design.md`)
shipped a "Sensores" upload bundling internas + riego + a partial subset of
externas into one `.xlsx` type, written to `sensor_readings_wide`. Auditing
the real `Data/` files and the actual trained pipeline
(`Models/results/pipeline_inv3.pkl`'s `sensor_cols`, ground truth — not the
training script's comments) found two problems:

1. **Column mapping was incomplete.** The pipeline needs 19 sensor features:
   8 internas + 8 externas + 3 riego (verified by loading the pickle
   directly). The shipped `SENSOR_COLS` only covered internas partially (5/8)
   and externas partially (4/8).
2. **Externas data isn't per-invernadero.** `Data/Variables exteriores.xlsx`
   is one shared weather file — unlike internas/riego, which are genuinely
   per-greenhouse. Bundling it into a per-invernadero upload was a design
   mismatch, not just a missing-column bug.

This spec splits the single "Sensores" upload into three: **Sensores**
(internas), **Riego**, and **Exteriores** (global), fixes the column lists
to match the trained pipeline exactly, and fixes the one downstream
consumer (`scripts/live_inference.py`) that assumed the old single-table
shape.

## 1. Authoritative column lists

Verified directly against `pipeline_inv3.pkl["sensor_cols"]`:

```python
['temp_prom_int', 'temp_min_int', 'temp_max_int', 'hr_prom_int', 'co2_ppm',
 'deficit_humedad', 'deficit_presion_vapor', 'humedad_abs_int',
 'temp_prom_ext', 'temp_max_ext', 'temp_min_ext', 'hr_prom_ext',
 'rad_sum', 'rad_max', 'dh_ext', 'humedad_abs_ext',
 'riego_total', 'ph_promedio', 'ce_promedio']
```

**Sensores (internas)** — `fecha` + 8:
`temp_prom_int, temp_min_int, temp_max_int, hr_prom_int, co2_ppm, deficit_humedad, deficit_presion_vapor, humedad_abs_int`

**Riego** — `fecha` + 3:
`riego_total, ph_promedio, ce_promedio`

**Exteriores** — `fecha` + 8:
`temp_prom_ext, temp_max_ext, temp_min_ext, hr_prom_ext, rad_sum, rad_max, dh_ext, humedad_abs_ext`

These map 1:1 to `Models/cnn_rnn_yield.py`'s rename rules for
`Variables internas invernadero {inv}.xlsx`, `Variables riego invernadero
{inv}.xlsx`, and `Variables exteriores.xlsx` respectively (lines 239-292),
confirming the client's real Excel files already carry the data needed —
the gap was purely in the upload feature's column lists, not the source data.

## 2. Database changes (new migration `008_split_sensor_tables.sql`)

```sql
-- Sensores (internas) — trim sensor_readings_wide to internas-only,
-- add the 3 previously-missing columns, drop the 4 ext + 3 riego columns
-- the original 003_sensor_wide.sql incorrectly bundled in (no real data
-- expected yet — excel-upload feature is pre-launch, migrations 006/007
-- not yet applied to live Supabase).
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
```

`submission_locks.form_type` check constraint extends to
`in ('sensores','riego','exteriores','fenologia','produccion')`. Exteriores
locks use `greenhouse_id = 0` internally (a sentinel — 0 is never a real
invernadero id) so the existing `NOT NULL` schema on `submission_locks`
doesn't need to change.

## 3. Backend (`backend/routers/uploads.py`)

- `FORM_TYPES` dict extends with `riego` and `exteriores` entries (required
  columns per §1).
- `POST /uploads/{inv}/riego` and its history/lock endpoints follow the
  existing per-invernadero pattern exactly (same as sensores/fenologia).
- `POST /uploads/exteriores` and `GET /uploads/exteriores/history` — **no
  `{inv}` path segment**. Internally: upsert into `exterior_readings` on
  `fecha` (no `greenhouse_id` in the record); lock check/touch uses
  `check_submission_lock(0, "exteriores")` / `touch_submission_lock(0,
  "exteriores")` (reusing `lock_utils.py` unchanged, just with the sentinel).
  `GET /submission-lock/exteriores` similarly has no `{inv}` segment.

## 4. Frontend (`demo/Demo Dashboard.html`)

- Insertar datos gains two more tabs: **Riego** (identical structure to
  Sensores/Fenología/Producción, per-invernadero) and **Exteriores** (same
  upload/preview/two-step-confirm flow, but hides the `#ins-inv` selector's
  effect for this tab and shows a static note: "Aplica a ambos
  invernaderos" instead of an invernadero-specific lock banner).
- `UPLOAD_TYPES` gains `riego` and `exteriores` entries with their
  `requiredCols`. The generic `wireUploadTab(formType)` from the existing
  feature already handles per-type behavior via `formType` — for
  `exteriores`, the POST/GET calls omit the `{inv}` path segment (small
  branch in the fetch URL construction, not a structural rewrite).
- Historial gains 2 more sub-tabs (Riego, Exteriores) — 5 total. Exteriores'
  sub-tab also has no invernadero filter.

## 5. `scripts/live_inference.py` fix

Currently:
```python
wide_resp = supa.table("sensor_readings_wide").select("*").eq("greenhouse_id", INV_ID)...
```
expects internas+riego+externas merged in one row per date — true before
this split, false after.

Fix: three queries, merged by `fecha` before calling `build_input_tensor`
(which needs **no changes** — `aggregate_wide_to_weekly` already just reads
whichever `pipeline["sensor_cols"]` keys are present in each row dict, and
tolerates missing ones via `NaN`-fill):

```python
sensor_resp = supa.table("sensor_readings_wide").select("*").eq("greenhouse_id", INV_ID).gte("fecha", cutoff).execute()
riego_resp  = supa.table("riego_readings").select("*").eq("greenhouse_id", INV_ID).gte("fecha", cutoff).execute()
ext_resp    = supa.table("exterior_readings").select("*").gte("fecha", cutoff).execute()  # no greenhouse_id filter

merged = {}
for row in (sensor_resp.data or []) + (riego_resp.data or []) + (ext_resp.data or []):
    merged.setdefault(row["fecha"], {}).update(row)
wide_rows = list(merged.values())
```

## Out of scope

- Backfilling historical `Data/` Excel content into these tables (the
  trained `.pt` models already encode T13-T17 history; this pipeline is for
  ongoing weekly client uploads going forward).
- Applying migrations `006`/`007`/`008` to the live Supabase project —
  still deferred to explicit human action, per the prior spec's decision.
- Inv4 support for `live_inference.py` — out of scope for this fix,
  unrelated pre-existing limitation (Inv3-only cron, per
  `2026-06-23-live-inference-pipeline-design.md`).
