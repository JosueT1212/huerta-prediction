# Auth, Sensor Ingest & Railway Deployment — Design Spec

**Date:** 2026-06-18
**Status:** Approved

---

## Overview

Three subsystems wired together into a production-ready stack:

1. **Supabase** — auth (email+password), role-based access, long-term sensor archive
2. **Railway** — FastAPI backend serving inference API + sensor ingest endpoint
3. **Telegraf** — sensor collector writing to FastAPI ingest (mocked with Excel replay script until physical sensors exist)

No InfluxDB. Supabase PostgreSQL is the single data store for both auth and sensor readings. Appropriate for greenhouse scale (10 sensors, hourly/daily readings).

---

## Architecture

```
Physical sensors (future) → Telegraf (HTTP output) ─┐
Excel mock script (now)   ─────────────────────────→ POST /ingest/sensors
                                                       │
                                                   FastAPI (Railway)
                                                   ├── CNN-RNN model (.pt baked in image)
                                                   ├── JWT middleware (Supabase verify)
                                                   └── Supabase client (reads + writes)
                                                       │
                                               Supabase PostgreSQL
                                               ├── auth.users (managed)
                                               ├── profiles
                                               ├── greenhouse_access
                                               └── sensor_readings

Browser → login.html → Supabase Auth → JWT → FastAPI → inference/sensor data
```

---

## Section 1: Supabase Auth & Database Schema

### Auth Flow

1. Browser loads `login.html` (served by FastAPI `GET /`)
2. User submits email + password → Supabase Auth returns JWT
3. Frontend stores JWT in `localStorage`, sends as `Authorization: Bearer <jwt>` on all API calls
4. FastAPI middleware verifies JWT using Supabase public key
5. Middleware fetches `profiles` row for `user_id` → injects `role` + `greenhouse_ids` into request state
6. Route guards enforce: operators only access their assigned greenhouses; admins access all

### Roles

| Role | Permissions |
|---|---|
| `admin` | All greenhouses, user management (future), full metrics |
| `operator` | Only greenhouses listed in `greenhouse_access` |

### SQL Schema

```sql
-- profiles: extends auth.users with role
create table profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  role        text not null check (role in ('admin', 'operator')),
  full_name   text,
  created_at  timestamptz default now()
);

-- greenhouse_access: operator → greenhouse mapping
create table greenhouse_access (
  user_id       uuid references profiles(id) on delete cascade,
  greenhouse_id int  not null check (greenhouse_id in (3, 4)),
  primary key (user_id, greenhouse_id)
);

-- sensor_readings: time-series archive (synced from Telegraf via ingest endpoint)
create table sensor_readings (
  id            bigserial primary key,
  greenhouse_id int         not null,
  recorded_at   timestamptz not null,
  sensor_name   text        not null,
  value         float8      not null
);

create index on sensor_readings (greenhouse_id, recorded_at desc);
```

### Row-Level Security

```sql
-- sensor_readings: admin sees all; operator sees only assigned greenhouses
alter table sensor_readings enable row level security;

create policy "admin full access" on sensor_readings
  for select using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );

create policy "operator own greenhouses" on sensor_readings
  for select using (
    exists (
      select 1 from greenhouse_access
      where user_id = auth.uid() and greenhouse_id = sensor_readings.greenhouse_id
    )
  );
```

---

## Section 2: Sensor Ingest Pipeline

### Telegraf Configuration

Telegraf runs on-premises (on the datalogger machine or a local server). Uses HTTP output plugin to POST batches to FastAPI.

```toml
# telegraf.conf

# INPUT: swap plugin to match actual sensor interface when hardware arrives
[[inputs.mqtt_consumer]]
  servers = ["tcp://localhost:1883"]
  topics  = ["greenhouse/+/sensors"]

# Alternative inputs (uncomment as needed):
# [[inputs.modbus]]
# [[inputs.serial]]
# [[inputs.http_listener_v2]]

[[outputs.http]]
  url             = "https://your-app.railway.app/ingest/sensors"
  method          = "POST"
  data_format     = "json"
  [outputs.http.headers]
    Authorization = "Bearer ${TELEGRAF_INGEST_TOKEN}"
    Content-Type  = "application/json"
```

### Ingest Endpoint (FastAPI)

Machine-to-machine auth uses a static bearer token (env var `INGEST_TOKEN`), not a Supabase JWT.

```
POST /ingest/sensors
Authorization: Bearer <INGEST_TOKEN>
Content-Type: application/json

{
  "greenhouse_id": 3,
  "sensor_name": "temp_interior",
  "value": 24.5,
  "recorded_at": "2026-06-18T10:00:00Z"
}
```

Response: `{"ok": true}`

### Excel Mock Script

`scripts/mock_ingest.py` — reads existing Excel sensor files and replays them row-by-row to `/ingest/sensors`. Allows full pipeline testing before physical sensors exist.

---

## Section 3: Railway Deployment

### Service

Single Railway service running FastAPI via Uvicorn.

**Dockerfile** (new file at repo root):
```dockerfile
FROM python/3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Model weights** — `.pt` files in `Models/results/` are committed to the repo and baked into the Docker image at build time. No object storage needed (~50 MB total).

### Environment Variables (Railway dashboard)

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Public anon key (safe to use in backend for auth verify) |
| `SUPABASE_SERVICE_KEY` | Service role key (bypasses RLS — write sensor_readings) |
| `INGEST_TOKEN` | Static secret for Telegraf machine-to-machine auth |

### FastAPI Changes Required

| Change | File | Notes |
|---|---|---|
| JWT middleware | `backend/main.py` | Verify Supabase JWT, inject role + greenhouse_ids |
| Greenhouse access guard | `backend/main.py` | Decorator/dep for operator routes |
| `POST /ingest/sensors` | `backend/main.py` | Token auth, write to Supabase |
| `GET /sensors/{inv}` | `backend/main.py` | Read recent sensor_readings from Supabase |
| Protect `/inference/{inv}` | `backend/main.py` | Require JWT + greenhouse access |
| Protect `/metrics/{inv}` | `backend/main.py` | Same |
| Supabase client setup | `backend/supabase_client.py` | New file — init supabase-py client |

### New Python Dependencies

```
supabase>=2.0
python-jose[cryptography]>=3.3
```

Add to `backend/requirements.txt`.

---

## What is NOT in scope

- Grafana (deferred — no live sensor dashboard until sensors exist)
- User management UI in dashboard (admin creates users directly in Supabase dashboard for now)
- Sensor data feeding into CNN-RNN inference (model still runs on historical T17 data; live inference is a separate future task)
- Multi-greenhouse expansion beyond inv3/inv4

---

## Migration Path

When physical sensors go live:
1. Swap Telegraf input plugin to match hardware interface
2. Verify `mock_ingest.py` replay matches real payload shape
3. If sensor frequency exceeds hourly → add `sensor_readings` partitioning by month
4. If query latency degrades → add TimescaleDB extension on Supabase (one-click in Supabase dashboard)
