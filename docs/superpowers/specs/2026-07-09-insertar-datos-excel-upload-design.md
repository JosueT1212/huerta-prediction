# Insertar datos: Excel-upload weekly submission — Design

**Date:** 2026-07-09
**Status:** Approved by user, pending implementation plan

## Context

"Insertar datos" currently has 3 tabs: **Fenología** (manual per-plant form),
**Producción real** (manual kg_actual form), **Historial** (read-only
phenology table with per-row delete). No Excel upload exists anywhere in the
codebase — `backend/` has no `UploadFile`/multipart usage; `.xlsx` files are
only read server-side from `Data/` as static training data.

The client will submit weekly data via Excel instead of the manual forms.
This spec replaces the manual forms with 3 independent upload flows and
removes now-dead live-sensor infrastructure.

## Decisions carried over vs. discarded

During brainstorming the scope pivoted from "restrict the manual forms to
once/week" to "build Excel upload, restricted to once/week." These earlier
decisions are **discarded** (they applied to the manual-form framing):
tab-level lock tied to form submit clicks, delete-resets-lock question,
double-confirm-on-form-submit.

These **carry over**, reframed for file upload: hard block (frontend +
backend), rolling 7-day window, per-greenhouse **and per-data-type** lock,
remove deletion entirely, remove "Tiempo real" tab, remove `ingest.py`.

## 1. Upload templates

Three clean single-sheet `.xlsx` templates (not the legacy multi-sheet
`Data/` training files, which have merged headers and dozens of unrelated
sheets — unsuitable to hand a client). One row per record; client appends
new rows weekly and re-uploads the whole file.

**Sensores** — mirrors `sensor_readings_wide`:
```
fecha, temp_prom_int, temp_min_int, temp_max_int, hr_prom_int, co2_ppm,
riego_total, ph_promedio, ce_promedio, temp_prom_ext, temp_max_ext,
temp_min_ext, rad_sum
```

**Fenología** — mirrors `phenology_observations` (exact DB column names):
```
fecha, zona, planta, racimos_puestos, flores_racimo_abiertas,
racimos_en_planta, cantidad_tomates, racimo_en_cosecha, tomates_maduros,
diametro_fruto_cm, crecimiento_planta_cm
```
`fecha` maps to `week_date`.

**Producción** — mirrors `predictions.kg_actual`:
```
fecha, kg_reales
```
`fecha` maps to `predicted_for`. Rows are **update-only**: a matching
`predictions` row (greenhouse_id, predicted_for) must already exist —
created by the live-inference cron, per
`docs/superpowers/specs/2026-06-23-live-inference-pipeline-design.md`.
`predictions.kg_predicted` is `not null`, so this endpoint never inserts a
new prediction row; a `fecha` with no matching row is reported in
`rows_skipped` with reason "no existe predicción para esa semana".

## 2. Backend: `backend/routers/uploads.py` (new)

- `POST /uploads/{inv}/{form_type}` — multipart `UploadFile`,
  `form_type ∈ {sensores, fenologia, produccion}`.
- Order of operations:
  1. Check `submission_locks` for `(inv, form_type)`. If `now() - last_submitted_at < 7 days` → `429 {next_allowed_at}`.
  2. Parse with `pandas.read_excel(file, sheet_name=0)`.
  3. Validate required columns present and typed correctly → `422` with
     row-level errors if malformed (no lock consumed on failure).
  4. Upsert into the target table on its existing unique constraint:
     - Sensores → `sensor_readings_wide` (`sensor_wide_gh_date`)
     - Fenología → `phenology_observations` (existing unique idx on
       `greenhouse_id, week_date, zona, planta`)
     - Producción → `predictions.kg_actual` via existing
       `predictions_gh_week` unique idx
     Re-uploading the same file is a no-op for previously-seen rows — only
     new rows change anything. **The weekly lock is a client-requested
     cadence control, not a data-integrity mechanism**; upsert already makes
     re-upload safe on its own.
  5. On success, upsert `submission_locks(greenhouse_id, form_type,
     last_submitted_at = now())`.
  6. Response: `{rows_in_file, rows_inserted, rows_updated, rows_skipped}`.
- `GET /submission-lock/{inv}/{form_type}` → `{locked: bool,
  next_allowed_at: timestamptz | null}` for frontend polling.

## 3. `submission_locks` table (new migration `006_submission_locks.sql`)

```sql
create table submission_locks (
  greenhouse_id int not null,
  form_type     text not null check (form_type in ('sensores','fenologia','produccion')),
  last_submitted_at timestamptz not null default now(),
  primary key (greenhouse_id, form_type)
);
```

## 4. Frontend — Insertar datos redesigned

Tabs become: **Sensores**, **Fenología**, **Producción**, **Historial**.

Each upload tab:
1. File picker (`.xlsx` only).
2. Client-side parse preview: row count + date/week range detected.
3. Confirm panel (step 1): summary of what will be sent.
4. Second explicit confirm (step 2): "Esta captura no podrá modificarse ni
   eliminarse. Próximo envío disponible en 7 días. ¿Confirmar
   definitivamente?"
5. POST to `/uploads/{inv}/{form_type}`. Show `rows_inserted` /
   `rows_skipped` result. `429` response re-fetches lock state and disables
   the tab with a "próximo envío: DATE" banner (handles race condition where
   lock changed between page load and submit).

Locked state (checked on tab load via `GET /submission-lock`): disables
file picker and submit button, shows countdown/date banner.

**Historial** — three read-only sub-views (Sensores / Fenología /
Producción), each listing rows from its respective table, ordered most
recent first. **No delete anywhere** — this replaces the old per-row delete
UI and `DELETE /phenology-live/{inv}/{obs_id}` /
`DELETE /live-predictions/{inv}/{id}/kg-actual` endpoints, which are removed.

## 5. Removals (bundled cleanup, unrelated to upload feature but requested together)

- **Frontend**: delete `view-live3` / `view-live4` sections
  (`demo/Demo Dashboard.html:1812-1892`ish) and their sidebar links
  (`:1239`, `:1247`), `sensor-grid` JS wiring, `live{3,4}-ts` refs. The
  sensor-history **modal** (`/sensor-history/{inv}`, "ver histórico" button
  on KPI cards) is unrelated — reads historical `Data/` Excel, not live
  sensors — and stays.
- **Backend**: delete `backend/routers/ingest.py` and
  `backend/routers/sensors.py`; remove their imports and
  `app.include_router(...)` calls in `backend/main.py:33-37,67-71`.
- **Supabase**: migration `007_drop_sensor_readings.sql` drops only the
  narrow `sensor_readings` table. **`sensor_readings_wide` is kept** — it's
  now the Sensores upload target.
- **Note for the record**: `backend/engine.py` (demo predictions endpoint)
  serves from cached `.npz`/`.pt` and is unaffected by this change. But
  `scripts/live_inference.py` (weekly cron, see
  `docs/superpowers/specs/2026-06-23-live-inference-pipeline-design.md`)
  reads `sensor_readings_wide` directly to run live Inv3 inference — it has
  no dependency on `ingest.py` (table-only read), so removing `ingest.py` is
  safe, but that cron currently assumes rows arrive via continuous daily
  ingest rather than a weekly bulk upload. Whether that assumption still
  holds is out of scope here — tracked as a **separate spec** (cadence
  mismatch between weekly bulk upload and the cron's rolling-window read).

## Out of scope

- Automatic model retraining or live-swap from uploaded sensor data.
- Admin override / manual unlock of the weekly restriction.
- Historical bulk-import of pre-existing `Data/` files through this pipeline
  (that data is already baked into the trained `.pt` models).
