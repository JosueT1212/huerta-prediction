# E2E Inference Smoke Test Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A single runnable script that simulates the full Inv3 live inference pipeline — from fake sensor POST to prediction written to Supabase — so the developer can confirm the dashboard shows a new prediction with the correct date.

**Architecture:** `scripts/test_e2e_inv3.py` drives the full path: ingest fake sensor readings → wide table → feature engineering → CNN-RNN forward pass → predictions table. No mocking, no DRY_RUN — real DB writes, real model. Asserts output is not NaN/Inf. Developer verifies visually on the dashboard.

**Tech Stack:** Python 3.11, requests, supabase-py, existing `backend/live_features.py`, `scripts/live_inference.py`, `Models/results/pipeline_inv3.pkl`, `Models/results/best_cnn_rnn_inv3.pt`

---

## Component Map

| File | Role |
|------|------|
| `scripts/test_e2e_inv3.py` | New — orchestrates the full smoke test |
| `backend/routers/ingest.py` | Existing — receives POST /ingest/sensors |
| `backend/live_features.py` | Existing — feature engineering |
| `scripts/live_inference.py` | Existing — inference + DB write (called via import) |

---

## Sensor Values

Synthetic readings based on T17 Inv3 seasonal averages:

| sensor_name | synthetic value |
|-------------|----------------|
| temp_prom_int | 22.5 |
| temp_min_int | 18.0 |
| temp_max_int | 28.0 |
| hr_prom_int | 65.0 |
| co2_ppm | 450.0 |
| riego_total | 12.0 |
| ph_promedio | 6.2 |
| ce_promedio | 3.1 |
| temp_prom_ext | 20.0 |
| temp_max_ext | 26.0 |
| temp_min_ext | 14.0 |
| rad_sum | 18500.0 |

Sent for each of the last 7 calendar days (one row per sensor per day = 84 POST requests).

---

## Script Flow

```
1. Parse env vars: INGEST_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY,
   TRANSPLANT_DATE_INV3, RAILWAY_URL (default http://localhost:8000)

2. For each of last 7 days × 12 sensors:
   POST {RAILWAY_URL}/ingest/sensors
   body: {greenhouse_id: 3, sensor_name, value, recorded_at: day T 12:00:00}
   header: Authorization: Bearer {INGEST_TOKEN}
   → print progress dots, raise on non-2xx

3. Import and call live_inference.main() with DRY_RUN=0
   (sets os.environ before import to control behaviour)

4. Query predictions table for greenhouse_id=3, order by predicted_at desc, limit 1
   Print: predicted_for date, kg_predicted, predicted_at timestamp

5. assert not (math.isnan(kg_predicted) or math.isinf(kg_predicted))
   assert kg_predicted > 0

6. Print: "PASS — check dashboard for prediction dated {predicted_for}"
   Exit 0 on pass, 1 on any failure
```

---

## Environment Variables

| Var | Required | Description |
|-----|----------|-------------|
| `INGEST_TOKEN` | yes | Bearer token for /ingest/sensors |
| `SUPABASE_URL` | yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | yes | Service role key (bypasses RLS) |
| `TRANSPLANT_DATE_INV3` | yes | ISO date e.g. 2025-05-22 |
| `RAILWAY_URL` | no | Default: http://localhost:8000 |

---

## Usage

```bash
INGEST_TOKEN=xxx \
SUPABASE_URL=https://xxx.supabase.co \
SUPABASE_SERVICE_KEY=xxx \
TRANSPLANT_DATE_INV3=2025-05-22 \
RAILWAY_URL=https://huertaprediction-production.up.railway.app \
python3 scripts/test_e2e_inv3.py
```

Expected output:
```
Ingesting 84 sensor readings to https://huertaprediction-production.up.railway.app...
............................................................................
Running live inference...
  Wide sensor rows pulled: 7
  Phenology observations pulled: 0
  Inv3 → 41823.4 kg for week 2026-06-30
  Written to predictions table.
Latest prediction: predicted_for=2026-06-30 kg=41823.4 predicted_at=2026-06-24T...
PASS — check dashboard for prediction dated 2026-06-30
```

---

## Out of Scope

- Phenology data ingest (pheno rows stay empty; model pads with `pheno_means`)
- Automated scheduling (this is a manual trigger)
- Inv4 (Inv3 only — pipeline_inv4.pkl does not exist yet)
