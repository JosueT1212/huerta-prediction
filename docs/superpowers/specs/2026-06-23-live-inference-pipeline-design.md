# Live Weekly Inference Pipeline — Design Spec

**Date:** 2026-06-23
**Status:** Approved

---

## Overview

Automated weekly inference pipeline that pulls live sensor data from Supabase, runs feature engineering, and feeds the CNN-RNN model to produce a 4-week-ahead kg prediction for **Invernadero 3 only**. Invernadero 4 is excluded — its sensor→production gap varies 0–5 weeks across seasons, making a reliable fixed horizon impossible without per-season calibration.

**Horizon:** `predicted_for = inference_date + 5 weeks`. Derived from Inv3's fixed framework: `gap(10–11) - seq_len(6) = eff_horizon(4–5)`. Upper end used (5 weeks).

No roles — any authenticated user can view predictions and enter actual production values.

---

## Architecture

```
Every Sunday 8am UTC (Railway Cron)
  └─→ scripts/live_inference.py
        ├─ Pull sensor_readings (last 12 weeks) from Supabase
        ├─ Aggregate hourly → weekly mean per sensor
        ├─ Pull last 4 kg_predicted rows from predictions table (lag features)
        ├─ Apply scaler_inv{3,4}.pkl + pca_inv{3,4}.pkl
        ├─ CNN-RNN forward pass → 1 prediction point (4 weeks ahead)
        └─ Upsert row into predictions table

FastAPI
  ├─ GET  /live-predictions/{inv}       → last N predictions (for dashboard)
  └─ PATCH /live-predictions/{inv}/{id} → set kg_actual (user enters real production)

Dashboard
  ├─ 4-week forecast cards (next 4 predicted weeks)
  ├─ Forecast history chart (predicted line + actual dots)
  └─ "Registrar producción real" modal → PATCH endpoint
```

---

## Section 1: Training Script Changes

`Models/cnn_rnn_yield.py` must save scaler and PCA after fitting on T13–T15 training data:

```python
import joblib

# After fitting scaler and PCA on train data:
joblib.dump(scaler, f"Models/results/scaler_inv{inv_id}.pkl")
joblib.dump(pca,    f"Models/results/pca_inv{inv_id}.pkl")
```

These `.pkl` files are baked into the Railway Docker image at build time. They must be committed to the repo.

Re-run training after this change to generate the pkl files:
```bash
python3 Models/cnn_rnn_yield.py
```

---

## Section 2: Database Schema

Apply via Supabase dashboard → SQL Editor:

```sql
-- supabase/migrations/002_predictions.sql

create table if not exists predictions (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  predicted_for date        not null,   -- target week (inference_date + 4 weeks)
  predicted_at  timestamptz not null default now(),
  kg_predicted  float8      not null,
  kg_actual     float8,                 -- null until user enters real production
  model_version text        not null default 'cnn_rnn_v1'
);

create unique index if not exists predictions_gh_week
  on predictions (greenhouse_id, predicted_for);
```

`predicted_for` is the Monday of the target week.
`kg_actual` is null until the user registers real production via the dashboard.

---

## Section 3: Feature Engineering Bridge

New file `backend/live_features.py` — pulls sensor data from Supabase and produces a model-ready input tensor.

### Sensor column mapping

| `sensor_name` in sensor_readings | Feature column |
|---|---|
| `temp_prom_int` | `temp_prom_int` |
| `temp_min_int` | `temp_min_int` |
| `temp_max_int` | `temp_max_int` |
| `hr_prom_int` | `hr_prom_int` |
| `co2_ppm` | `co2_ppm` |
| `temp_prom_ext` | `temp_prom_ext` |
| `rad_sum` | `rad_sum` (sum, not mean) |
| `riego_total` | `riego_total` (sum, not mean) |

### Pipeline steps

1. Pull `sensor_readings` rows for `greenhouse_id` covering last 12 ISO weeks
2. Group by ISO week → mean per sensor (sum for `rad_sum`, `riego_total`)
3. Use partial mean if week is incomplete (Option A — any readings present)
4. Pull last 4 `kg_predicted` values from `predictions` table → build `kg_lag_1..4`
5. Load `scaler_inv{inv}.pkl` → transform feature matrix
6. Load `pca_inv{inv}.pkl` → reduce dimensions
7. Build sequence tensor matching `seq_len` from `hp_inv{inv}.yaml`
8. Return tensor ready for `model.forward()`

---

## Section 4: Inference Script

`scripts/live_inference.py` — Railway Cron entry point.

```
For each greenhouse in [3, 4]:
  1. Call live_features.build_input_tensor(inv)
  2. Load model weights from Models/results/best_cnn_rnn_inv{inv}.pt
  3. model.eval() → forward pass → scalar kg prediction
  4. predicted_for = today + 5 weeks (Monday of that week)
  5. Upsert into predictions table (on conflict greenhouse_id+predicted_for: update kg_predicted)
  6. Log result
```

Requires env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.

### Railway Cron configuration

Set in Railway dashboard → your service → **Cron**:
```
Schedule: 0 8 * * 0
Command:  python scripts/live_inference.py
```

---

## Section 5: API Endpoints

### GET /live-predictions/{inv}

Returns last N predictions for a greenhouse. Protected by JWT.

Query param: `limit` (default 20).

Response:
```json
[
  {
    "id": 1,
    "greenhouse_id": 3,
    "predicted_for": "2026-07-14",
    "predicted_at": "2026-06-22T08:00:00Z",
    "kg_predicted": 247.3,
    "kg_actual": null,
    "model_version": "cnn_rnn_v1"
  }
]
```

### PATCH /live-predictions/{inv}/{id}

Updates `kg_actual` for a prediction row. Any authenticated user can submit. Protected by JWT.

Request body:
```json
{ "kg_actual": 241.0 }
```

Response: `{ "ok": true }`

---

## Section 6: Dashboard Changes

### Remove
- Loss curves (train/val loss chart) from inv3 and inv4 views

### Add to each greenhouse view — "Predicción en vivo" section

**4-week forecast cards:**
One card per predicted week showing `predicted_for` date and `kg_predicted`. Loaded from `GET /live-predictions/{inv}?limit=4`.

**Forecast history chart:**
Line chart (Chart.js or existing charting library):
- Blue line: `kg_predicted` per week over time
- Green dots: `kg_actual` where not null
- X-axis: `predicted_for` date
- Y-axis: kg

**"Registrar producción real" button:**
Opens modal → user selects week from dropdown (past predictions with null `kg_actual`) → enters kg → `PATCH /live-predictions/{inv}/{id}` → chart refreshes.

---

## What is NOT in scope

- Retraining model on live data (model weights stay fixed at T13–T17 training)
- Email/push notifications when prediction is ready
- Confidence intervals (dashboard shows point estimate only)

---

## Next Steps (future work)

**Inv4 horizon calibration:** Inv4's sensor→production gap varies 0–5 weeks across seasons (vs stable 10–11 for Inv3). Before live inference can be enabled for Inv4, the training framework needs revision — either a per-season variable `seq_len`, or a fixed gap alignment strategy that doesn't collapse to near-zero effective horizon in seasons like T16 (gap=6, seq_len=6 → eff_horizon=0). Once fixed and retrained, Inv4 can be added to `live_inference.py` with its own `predicted_for` offset.

---

## Section 7: Phenology Input (user-entered, stored in Supabase)

Phenology features require manual field observation — they cannot come from sensors. Users enter them weekly via the dashboard.

### New Supabase table

```sql
-- supabase/migrations/003_phenology_live.sql

create table if not exists phenology_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  week_date     date        not null,   -- Monday of the observation week
  recorded_at   timestamptz not null default now(),
  racimos_puestos          float8,
  flores_racimo_abiertas   float8,
  racimos_en_planta        float8,
  cantidad_tomates         float8,
  racimo_en_cosecha        float8,
  tomates_maduros          float8,
  diametro_fruto_cm        float8,
  crecimiento_planta_cm    float8
);

create unique index if not exists phenology_gh_week
  on phenology_readings (greenhouse_id, week_date);
```

Column names map 1-to-1 with `PHENO_FEATURE_COLS` in `cnn_rnn_yield.py`:

| DB column | Model feature |
|---|---|
| `racimos_puestos` | `RACIMOS PUESTOS` |
| `flores_racimo_abiertas` | `FLORES EN RACIMO ABIERTAS` |
| `racimos_en_planta` | `CANTIDAD DE RACIMOS EN PLANTA` |
| `cantidad_tomates` | `CANTIDAD DE TOMATES` |
| `racimo_en_cosecha` | `Nº DE RACIMO EN COSECHA` |
| `tomates_maduros` | `TOMATES MADUROS (COLOR 2)` |
| `diametro_fruto_cm` | `DIAMETRO DEL FRUTO cm` |
| `crecimiento_planta_cm` | `CRECIMIENTO PLANTA (cm)` |

### API endpoints

```
POST  /phenology-live/{inv}        → insert/upsert weekly phenology row (user submits form)
GET   /phenology-live/{inv}        → list recent rows (for display + inference use)
```

Both protected by JWT. Any authenticated user can submit.

### Dashboard — "Fenología semanal" form

In each greenhouse view, a weekly entry form:
- Week selector (defaults to current week)
- One numeric input per phenology field (8 fields)
- Submit → `POST /phenology-live/{inv}`

### How live_features.py uses it

When building the input tensor for Sunday inference:
1. Pull `phenology_readings` for the last `seq_len` weeks
2. If a week has a row → use values directly
3. If a week is missing → fill with training mean for that week-in-season (fallback only)

This way inference degrades gracefully if the user misses a week, but uses real data when available.
