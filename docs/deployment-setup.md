# Deployment & Configuration Guide

## Environment Variables

### FastAPI (Railway)

| Variable | Description | Where to get it |
|---|---|---|
| `SUPABASE_URL` | Project URL | Supabase dashboard → Settings → API → Project URL |
| `SUPABASE_ANON_KEY` | Public key for browser auth | Supabase dashboard → Settings → API → anon public |
| `SUPABASE_SERVICE_KEY` | Secret admin key — server only, never expose | Supabase dashboard → Settings → API → service_role |
| `INGEST_TOKEN` | Static secret for Telegraf → FastAPI auth | Generate: `openssl rand -hex 32` |

Add to Railway: dashboard → your service → **Variables** tab.

For local dev: copy `.env.example` → `.env` and fill in values.

```bash
cp .env.example .env
source .env
uvicorn backend.main:app --port 8000 --reload
```

### Key explanations

- **`SUPABASE_ANON_KEY`** — safe to expose in browser. Used by `login.html` to sign in users. Respects Row Level Security.
- **`SUPABASE_SERVICE_KEY`** — server-side only. Bypasses RLS. Used by FastAPI to verify JWTs, write sensor readings, create/disable users. Never send to frontend.
- **`INGEST_TOKEN`** — shared secret between FastAPI and Telegraf. FastAPI verifies it; Telegraf sends it. Same value on both machines.

---

## Sensor Pipeline Setup (Telegraf)

### 1. Install Telegraf on the datalogger machine

```bash
# Ubuntu/Debian
sudo apt-get install telegraf

# macOS
brew install telegraf
```

### 2. Set the ingest token

Same value as `INGEST_TOKEN` on the FastAPI side:

```bash
export TELEGRAF_INGEST_TOKEN=your-secret-here

# Persistent across reboots (Linux):
echo "TELEGRAF_INGEST_TOKEN=your-secret-here" | sudo tee -a /etc/environment
```

### 3. Choose input plugin in `telegraf/telegraf.conf`

Uncomment the block matching your hardware:

- **MQTT broker** — `[[inputs.mqtt_consumer]]` (active by default)
- **HTTP listener** — `[[inputs.http_listener_v2]]` (datalogger POSTs to Telegraf)
- **File/CSV drop** — `[[inputs.file]]`

### 4. Update the Railway URL in `telegraf/telegraf.conf`

```toml
[[outputs.http]]
  url = "https://your-actual-app.railway.app/ingest/sensors"
```

### 5. Run Telegraf

```bash
telegraf --config telegraf/telegraf.conf
```

Once running, sensor readings appear in the **Tiempo real** view in the dashboard (polls every 30 seconds).

---

## Mock Ingest (no physical sensors)

Replay historical Excel data to test the full pipeline:

```bash
INGEST_TOKEN=your-secret python3.11 scripts/mock_ingest.py --inv 3 --host http://localhost:8000

# Dry run (no actual POST):
python3.11 scripts/mock_ingest.py --inv 3 --dry-run
```

---

## Supabase Database Setup

Apply migration manually via Supabase dashboard → SQL Editor:

```sql
-- File: supabase/migrations/001_schema.sql
create table if not exists profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  full_name   text,
  disabled    boolean     not null default false,
  created_at  timestamptz not null default now()
);

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

---

## Railway Deployment

Dockerfile is at repo root. Railway auto-detects it.

1. Connect repo to Railway
2. Add the 4 env vars (see above)
3. Deploy — Railway builds the image and runs:
   ```
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
