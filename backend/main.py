"""
Backend FastAPI del demo Huerta Prediction.

- Sirve el dashboard (mismo origen → sin problemas de CORS/file://).
- Carga los .pt al arrancar (warm) y corre la inferencia T17 una sola vez.
- Expone la inferencia por invernadero y por ventana (semana elegida).

Arranque:
    uvicorn backend.main:app --port 8000 --reload
Abrir:
    http://localhost:8000/
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Permite `uvicorn backend.main:app` (repo root en path) y `uvicorn main:app` (dentro de backend/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ENGINE  # noqa: E402
import data_api  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from backend.auth import get_current_user  # noqa: E402
from backend.routers import ingest as ingest_router  # noqa: E402
from backend.routers import sensors as sensors_router  # noqa: E402
from backend.routers import admin as admin_router  # noqa: E402
from backend.routers import predictions as predictions_router  # noqa: E402
from backend.routers import phenology_live as phenology_live_router  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / 'demo'
HTML_FILE = DEMO_DIR / 'Demo Dashboard.html'
LOGIN_FILE = DEMO_DIR / 'login.html'

MODEL_VERSION = 'CNN-RNN v3.2'
SYNC_DATE = '28 May 2026'
HORIZON_WEEKS = 6
INV_IDS = (3, 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[main] Cargando modelos y corriendo inferencia T17 (1 vez)…')
    ENGINE.warm(INV_IDS)
    print(f'[main] Modelos listos: {ENGINE.loaded()}')
    yield


app = FastAPI(title='Huerta Prediction — Demo', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # solo dev local; restringir antes de producción
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(ingest_router.router)
app.include_router(sensors_router.router)
app.include_router(admin_router.router)
app.include_router(predictions_router.router)
app.include_router(phenology_live_router.router)


def _check_inv(inv: int):
    if inv not in INV_IDS:
        raise HTTPException(404, f'Invernadero {inv} no existe (usa 3 o 4)')


# ── Login + Dashboard ──────────────────────────────────────────────────────
@app.get('/')
def login():
    if not LOGIN_FILE.exists():
        raise HTTPException(500, f'No se encontró {LOGIN_FILE}')
    return FileResponse(LOGIN_FILE)


@app.get('/app')
def index():
    if not HTML_FILE.exists():
        raise HTTPException(500, f'No se encontró {HTML_FILE}')
    return FileResponse(HTML_FILE)


# ── Estado ─────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_version': MODEL_VERSION,
        'sync_date': SYNC_DATE,
        'horizon_weeks': HORIZON_WEEKS,
        'greenhouses_loaded': ENGINE.loaded(),
    }


@app.get('/config')
def config():
    return {
        'supabase_url': os.environ['SUPABASE_URL'],
        'supabase_anon_key': os.environ['SUPABASE_ANON_KEY'],
    }


@app.get('/me')
def me(current_user: Annotated[dict, Depends(get_current_user)]):
    return current_user


@app.get('/greenhouses')
def greenhouses():
    return [{'id': i, 'name': f'Invernadero {i}'} for i in INV_IDS]


# ── Inferencia (modelos cargados en backend) ───────────────────────────────
@app.get('/inference/{inv}')
def inference(inv: int, current_user: Annotated[dict, Depends(get_current_user)]):
    """Arrays completos T17 calculados desde best_cnn_rnn_inv{inv}.pt."""
    _check_inv(inv)
    return ENGINE.payload(inv)


@app.get('/inference/{inv}/window')
def inference_window(
    inv: int,
    cursor: int | None = Query(None, description='Índice de semana T17 (0..n-1)'),
    horizon: int = Query(6, ge=1, le=12),
    past: int = Query(5, ge=0, le=20),
):
    """Recorte alrededor de la semana elegida — alimenta el slider/Play del demo."""
    _check_inv(inv)
    return ENGINE.window(inv, cursor=cursor, horizon=horizon, past=past)


@app.get('/metrics/{inv}')
def metrics(inv: int, current_user: Annotated[dict, Depends(get_current_user)]):
    """Métricas de la inferencia real (best + intervalos CPTC)."""
    _check_inv(inv)
    return ENGINE.metrics(inv)


# Compat: el dashboard original llamaba /predictions/{inv}
@app.get('/predictions/{inv}')
def predictions(inv: int):
    _check_inv(inv)
    return ENGINE.payload(inv)


# ── Fenología (KPIs + formulario "Insertar datos") ────────────────────────
@app.get('/phenology/{inv}')
def phenology(inv: int, current_user: Annotated[dict, Depends(get_current_user)]):
    """Serie semanal (promedio) de las 8 variables de fenología que entrena el modelo."""
    _check_inv(inv)
    return data_api.phenology_weekly(inv)


class PhenologySubmission(BaseModel):
    week: int | None = None
    rows: list[dict]  # una fila por planta, claves = field['key']


@app.post('/phenology/{inv}')
def submit_phenology(inv: int, payload: PhenologySubmission, current_user: Annotated[dict, Depends(get_current_user)]):
    """Demo: valida filas-por-planta y devuelve el promedio calculado (NO persiste)."""
    _check_inv(inv)
    try:
        result = data_api.validate_and_average(payload.rows)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        'inv_id': inv,
        'week': payload.week,
        'persisted': False,
        'note': 'Demo sin persistencia — los datos se guardarán en Supabase tras el deployment.',
        **result,
    }


# ── Histórico de sensores (vista "Tiempo real" → "ver histórico") ──────────
@app.get('/sensor-history/{inv}')
def sensor_hist(inv: int, current_user: Annotated[dict, Depends(get_current_user)], var: str = Query(..., description='temp|hr|co2|ce|par')):
    """Promedio mensual por temporada (una serie por temporada T13–T17)."""
    _check_inv(inv)
    try:
        return data_api.sensor_history(inv, var)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ── Estáticos del demo (assets sueltos si los hubiera) ─────────────────────
if DEMO_DIR.exists():
    app.mount('/demo', StaticFiles(directory=str(DEMO_DIR)), name='demo')
