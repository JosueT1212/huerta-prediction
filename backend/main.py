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

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Permite `uvicorn backend.main:app` (repo root en path) y `uvicorn main:app` (dentro de backend/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ENGINE  # noqa: E402

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


@app.get('/greenhouses')
def greenhouses():
    return [{'id': i, 'name': f'Invernadero {i}'} for i in INV_IDS]


# ── Inferencia (modelos cargados en backend) ───────────────────────────────
@app.get('/inference/{inv}')
def inference(inv: int):
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
def metrics(inv: int):
    """Métricas de la inferencia real (best + intervalos CPTC)."""
    _check_inv(inv)
    return ENGINE.metrics(inv)


# Compat: el dashboard original llamaba /predictions/{inv}
@app.get('/predictions/{inv}')
def predictions(inv: int):
    _check_inv(inv)
    return ENGINE.payload(inv)


# ── Estáticos del demo (assets sueltos si los hubiera) ─────────────────────
if DEMO_DIR.exists():
    app.mount('/demo', StaticFiles(directory=str(DEMO_DIR)), name='demo')
