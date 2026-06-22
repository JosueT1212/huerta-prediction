# Auth, Sensor Ingest & Railway Deployment — Design Spec

**Date:** 2026-06-18 (updated 2026-06-22)
**Status:** Approved

---

## Overview

Three subsystems wired together into a production-ready stack:

1. **Supabase** — auth (email+password), long-term sensor archive. No roles — all authenticated users see all greenhouses equally.
2. **Railway** — single service running FastAPI. Serves both the frontend HTML and the inference/ingest API. Demo deployment; production migrates to Hetzner VPS (same Dockerfile).
3. **Telegraf** — on-premises sensor collector writing to FastAPI ingest (mocked with Excel replay script until physical sensors exist).

No InfluxDB. No Grafana. Supabase PostgreSQL is the single data store for auth and sensor readings.

---

## Architecture

```
Physical sensors (future) → Telegraf (HTTP output) ─┐
Excel mock script (now)   ─────────────────────────→ POST /ingest/sensors
                                                       │
                                               FastAPI (Railway)
                                               ├── demo/login.html  (GET /)
                                               ├── demo/Demo Dashboard.html  (GET /app)
                                               ├── CNN-RNN model (.pt baked in image)
                                               ├── JWT verification (Supabase)
                                               └── Supabase client (reads + writes)
                                                       │
                                               Supabase PostgreSQL
                                               ├── auth.users (managed by Supabase)
                                               ├── profiles (id, full_name, disabled)
                                               └── sensor_readings

Browser → login.html → Supabase Auth → JWT → FastAPI → inference/sensor data
```

---

## Section 1: Supabase Auth & Database Schema

### Auth Flow

1. Browser loads `login.html` (served by FastAPI `GET /`)
2. User submits email + password → Supabase Auth returns JWT
3. Frontend stores JWT in `localStorage`, sends as `Authorization: Bearer <jwt>` on all API calls
4. FastAPI `get_current_user` dep verifies JWT via Supabase, fetches profile
5. All routes require a valid JWT — no role checks, all users see all data

### SQL Schema

```sql
-- profiles: extends auth.users (no role column)
create table if not exists profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  full_name   text,
  disabled    boolean     not null default false,
  created_at  timestamptz not null default now()
);

-- sensor_readings: time-series archive written by the ingest endpoint
create table if not exists sensor_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  recorded_at   timestamptz not null,
  sensor_name   text        not null,
  value         float8      not null
);

create index if not exists sensor_readings_gh_time
  on sensor_readings (greenhouse_id, recorded_at desc);
```

No RLS policies needed on `sensor_readings` — all authenticated users see all data. The service role key (used by FastAPI) bypasses RLS for writes.

---

## Section 2: Sensor Ingest Pipeline

### Telegraf Configuration

Telegraf runs on-premises on the datalogger machine. Uses HTTP output plugin to POST to FastAPI.

```toml
[[inputs.mqtt_consumer]]       # swap for actual hardware input plugin
  servers = ["tcp://localhost:1883"]
  topics  = ["greenhouse/+/sensors"]

[[outputs.http]]
  url         = "https://your-app.railway.app/ingest/sensors"
  method      = "POST"
  data_format = "json"
  [outputs.http.headers]
    Authorization = "Bearer ${TELEGRAF_INGEST_TOKEN}"
    Content-Type  = "application/json"
```

### Ingest Endpoint (FastAPI)

Machine-to-machine auth uses a static bearer token (`INGEST_TOKEN` env var), not a Supabase JWT.

```
POST /ingest/sensors
Authorization: Bearer <INGEST_TOKEN>

{ "greenhouse_id": 3, "sensor_name": "temp_interior", "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z" }
```

### Excel Mock Script

`scripts/mock_ingest.py` — replays existing Excel sensor files row-by-row to `/ingest/sensors` for pipeline testing before physical sensors exist.

---

## Section 3: Railway Deployment

### What runs on Railway

Single service: FastAPI (`backend/main.py`) + CNN-RNN model weights (`.pt` baked into image) + frontend HTML files (`demo/`).

FastAPI serves:
- `GET /` → `demo/login.html`
- `GET /app` → `demo/Demo Dashboard.html`
- All API endpoints

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables (Railway dashboard)

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Public anon key |
| `SUPABASE_SERVICE_KEY` | Service role key — used for writes and admin user creation |
| `INGEST_TOKEN` | Static secret for Telegraf machine-to-machine auth |

### FastAPI Changes Required

| Change | File |
|---|---|
| Add `backend/supabase_client.py` | Supabase client singleton |
| Add `backend/auth.py` | `get_current_user` dep (JWT verify only, no roles) |
| Add `backend/routers/ingest.py` | `POST /ingest/sensors` |
| Add `backend/routers/sensors.py` | `GET /sensors/{inv}` |
| Add `backend/routers/admin.py` | User management CRUD (any authenticated user) |
| Modify `backend/main.py` | Mount routers, add `/config` + `/me`, protect routes |
| Modify `demo/login.html` | Real Supabase auth (replace fake sessionStorage) |
| Modify `demo/Demo Dashboard.html` | JWT headers on fetch, live sensor polling, user management panel |

### Deployment stages

| Stage | Platform | Cost |
|---|---|---|
| Demo | Railway Hobby | ~$5/mo (covered by credit) |
| Production | Hetzner CX22 VPS | ~$4.50/mo — same Dockerfile, add nginx + SSL |

---

## Section 4: User Management

Any authenticated user can manage accounts. No admin-only restriction.

### Routes

```
GET   /admin/users                → list all users (profiles)
POST  /admin/users                → invite new user by email (Supabase invite flow)
PATCH /admin/users/{user_id}      → update full_name or disabled flag
```

### Frontend panel ("Usuarios" view in dashboard)

- Table: name, status (active/disabled), created date
- "Nuevo usuario" button → modal: email + full name → POST /admin/users
- Per-row toggle: disable / re-enable account

FastAPI calls Supabase Admin API (`SUPABASE_SERVICE_KEY`) to create `auth.users` entry via invite, then inserts `profiles` row.

---

## What is NOT in scope

- Grafana (not needed — dashboard polls `/sensors/{inv}` directly)
- Role-based access control
- Sensor data feeding into live CNN-RNN inference (model still runs on historical T17; live inference is a future task)
- Multi-greenhouse expansion beyond inv3/inv4

---

## Migration Path (Demo → Production)

1. Provision Hetzner CX22 ($4.50/mo)
2. Install Docker + nginx on the server
3. `docker pull` or `git clone` + `docker build` — same Dockerfile
4. Set same 4 env vars
5. Add nginx reverse proxy + Let's Encrypt SSL (free)
6. Point domain DNS to Hetzner IP
7. Total cost: ~$4.50/mo + Supabase free tier
