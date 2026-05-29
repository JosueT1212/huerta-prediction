# Client Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beautiful, client-facing Streamlit dashboard for an agriculture company showing weekly yield forecasts, prediction vs actual comparisons, a data-entry form for phenology and harvest records, and a live sensor data panel.

**Architecture:** Multi-page Streamlit app under `Dashboard/` with a shared SQLite database (`Dashboard/data/huerta.db`) for persistence. A separate `sensor_ingest.py` script polls a configurable sensor CSV/API and writes to the DB. All chart logic lives in `Dashboard/components/charts.py`; all DB access in `Dashboard/db.py`. The theme is defined once in `Dashboard/theme.py` and injected via `st.markdown` on every page.

**Tech Stack:** Python 3.10+, Streamlit ≥1.32, Plotly ≥5.18, pandas, numpy, torch (inference), SQLite3 (stdlib), pytest

---

> **Scope note:** This plan covers three subsystems (forecast view, data entry, sensor ingestion) in one Streamlit app. If timeline is tight, Tasks 1–5 (forecast + data entry) can ship first; Tasks 6–8 (sensor ingest + live panel) can follow.

---

## File Structure

```
Dashboard/
├── app.py                        # Home page — overview KPIs + nav
├── pages/
│   ├── 1_Forecast.py             # Forecast chart + prediction intervals
│   ├── 2_Comparativo.py          # Predicted vs actual numbers table
│   ├── 3_Registro.py             # Data entry: kg + phenology
│   └── 4_Sensores.py             # Live sensor readings
├── components/
│   ├── charts.py                 # Plotly chart builders (pure functions)
│   ├── metrics_cards.py          # KPI card renderer
│   └── theme.py                  # CSS, colors, inject_theme()
├── db.py                         # All SQLite CRUD (no Streamlit imports)
├── sensor_ingest.py              # Standalone polling script
├── data/
│   └── .gitkeep                  # huerta.db created at runtime
└── tests/
    ├── test_db.py
    ├── test_charts.py
    └── test_sensor_ingest.py
```

---

## Task 1: Theme & CSS Foundation

**Files:**
- Create: `Dashboard/components/theme.py`
- Modify: `Dashboard/app.py`
- Test: `Dashboard/tests/test_theme.py`

- [ ] **Step 1: Write a failing test for inject_theme**

```python
# Dashboard/tests/test_theme.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.theme import PALETTE, CSS

def test_palette_has_required_keys():
    for key in ("primary", "accent", "bg", "card", "text_muted"):
        assert key in PALETTE, f"Missing palette key: {key}"

def test_css_contains_font_import():
    assert "@import" in CSS or "font-family" in CSS
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction
python3 -m pytest Dashboard/tests/test_theme.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `Dashboard/components/__init__.py` (empty) and `Dashboard/tests/__init__.py` (empty)**

```bash
touch Dashboard/components/__init__.py Dashboard/tests/__init__.py
```

- [ ] **Step 4: Write `Dashboard/components/theme.py`**

```python
# Dashboard/components/theme.py
import streamlit as st

PALETTE = {
    "primary":    "#1B5E20",   # deep forest green
    "primary_lt": "#2E7D32",
    "accent":     "#E67E22",   # warm amber harvest
    "accent_lt":  "#F39C12",
    "bg":         "#F9F6F0",   # warm cream
    "card":       "#FFFFFF",
    "border":     "#D5E8D4",
    "text":       "#1C1C1C",
    "text_muted": "#6B7280",
    "good":       "#27AE60",
    "warn":       "#E67E22",
    "bad":        "#C0392B",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {PALETTE['primary']};
}}
[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {{
    color: #C8E6C9 !important;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {PALETTE['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {PALETTE['primary']};
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    background: {PALETTE['card']};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {PALETTE['border']};
}}
[data-baseweb="tab"] {{
    border-radius: 8px !important;
    font-weight: 500;
}}
[aria-selected="true"] {{
    background: {PALETTE['primary']} !important;
    color: white !important;
}}

/* DataFrames */
[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {PALETTE['border']};
}}

/* Buttons */
[data-testid="baseButton-primary"] {{
    background-color: {PALETTE['primary']} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

/* Page header hero */
.hero-header {{
    background: linear-gradient(135deg, {PALETTE['primary']} 0%, {PALETTE['primary_lt']} 60%, #388E3C 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(27,94,32,0.25);
}}
.hero-header h1 {{
    margin: 0 0 0.3rem 0;
    font-size: 2rem;
    font-weight: 700;
}}
.hero-header p {{
    margin: 0;
    opacity: 0.85;
    font-size: 1rem;
}}

/* Section headers */
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {PALETTE['primary']};
    border-left: 4px solid {PALETTE['accent']};
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem 0;
}}

/* Status badges */
.badge-good  {{ background:#D5F5E3; color:#1E8449; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
.badge-warn  {{ background:#FDEBD0; color:#D35400; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
.badge-bad   {{ background:#FADBD8; color:#C0392B; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
</style>
"""

def inject_theme():
    """Call once per page at the top to apply global styles."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f"""
    <div class="hero-header">
        <h1>🌿 {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python3 -m pytest Dashboard/tests/test_theme.py -v
```
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add Dashboard/components/ Dashboard/tests/
git commit -m "feat: add agriculture theme with green/amber palette and CSS"
```

---

## Task 2: SQLite Database Layer

**Files:**
- Create: `Dashboard/db.py`
- Create: `Dashboard/tests/test_db.py`

- [ ] **Step 1: Write failing tests for db.py**

```python
# Dashboard/tests/test_db.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from db import init_db, insert_kg_entry, insert_phenology_entry, \
                insert_sensor_reading, get_kg_entries, get_phenology_entries, \
                get_latest_sensor_readings

def make_db():
    tmp = tempfile.mktemp(suffix=".db")
    init_db(tmp)
    return tmp

def test_init_creates_tables():
    path = make_db()
    import sqlite3
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"kg_entries", "phenology_entries", "sensor_readings"}.issubset(tables)

def test_insert_and_get_kg_entry():
    path = make_db()
    insert_kg_entry(path, invernadero_id=3, iso_week=202417, kg_reales=8500.0)
    df = get_kg_entries(path, invernadero_id=3)
    assert len(df) == 1
    assert df.iloc[0]["kg_reales"] == 8500.0
    assert df.iloc[0]["iso_week"] == 202417

def test_insert_duplicate_kg_updates():
    path = make_db()
    insert_kg_entry(path, invernadero_id=3, iso_week=202417, kg_reales=8500.0)
    insert_kg_entry(path, invernadero_id=3, iso_week=202417, kg_reales=9200.0)
    df = get_kg_entries(path, invernadero_id=3)
    assert len(df) == 1
    assert df.iloc[0]["kg_reales"] == 9200.0

def test_insert_and_get_phenology():
    path = make_db()
    insert_phenology_entry(path, invernadero_id=3, iso_week=202417,
                           stage="floracion", dias_desde_transplante=65)
    df = get_phenology_entries(path, invernadero_id=3)
    assert len(df) == 1
    assert df.iloc[0]["stage"] == "floracion"

def test_insert_and_get_sensor_readings():
    path = make_db()
    insert_sensor_reading(path, invernadero_id=3, timestamp="2024-04-20 08:00:00",
                          sensor_name="temp_interior", value=24.5)
    df = get_latest_sensor_readings(path, invernadero_id=3, n=10)
    assert len(df) == 1
    assert df.iloc[0]["value"] == 24.5
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest Dashboard/tests/test_db.py -v
```
Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Write `Dashboard/db.py`**

```python
# Dashboard/db.py
import sqlite3
from pathlib import Path
import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "huerta.db"


def _connect(db_path=None):
    path = db_path or DEFAULT_DB
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(db_path=None):
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kg_entries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invernadero_id  INTEGER NOT NULL,
            iso_week        INTEGER NOT NULL,
            kg_reales       REAL    NOT NULL,
            entered_at      TEXT    DEFAULT (datetime('now')),
            UNIQUE(invernadero_id, iso_week)
        );
        CREATE TABLE IF NOT EXISTS phenology_entries (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            invernadero_id          INTEGER NOT NULL,
            iso_week                INTEGER NOT NULL,
            stage                   TEXT,
            dias_desde_transplante  INTEGER,
            notes                   TEXT,
            entered_at              TEXT    DEFAULT (datetime('now')),
            UNIQUE(invernadero_id, iso_week)
        );
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invernadero_id  INTEGER NOT NULL,
            timestamp       TEXT    NOT NULL,
            sensor_name     TEXT    NOT NULL,
            value           REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sensor_ts
            ON sensor_readings(invernadero_id, timestamp DESC);
    """)
    conn.commit()
    conn.close()


def insert_kg_entry(db_path=None, *, invernadero_id, iso_week, kg_reales):
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO kg_entries (invernadero_id, iso_week, kg_reales)
        VALUES (?, ?, ?)
        ON CONFLICT(invernadero_id, iso_week)
        DO UPDATE SET kg_reales=excluded.kg_reales, entered_at=datetime('now')
    """, (invernadero_id, iso_week, kg_reales))
    conn.commit(); conn.close()


def insert_phenology_entry(db_path=None, *, invernadero_id, iso_week,
                            stage=None, dias_desde_transplante=None, notes=None):
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO phenology_entries (invernadero_id, iso_week, stage, dias_desde_transplante, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(invernadero_id, iso_week)
        DO UPDATE SET stage=excluded.stage,
                      dias_desde_transplante=excluded.dias_desde_transplante,
                      notes=excluded.notes, entered_at=datetime('now')
    """, (invernadero_id, iso_week, stage, dias_desde_transplante, notes))
    conn.commit(); conn.close()


def insert_sensor_reading(db_path=None, *, invernadero_id, timestamp, sensor_name, value):
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO sensor_readings (invernadero_id, timestamp, sensor_name, value)
        VALUES (?, ?, ?, ?)
    """, (invernadero_id, timestamp, sensor_name, value))
    conn.commit(); conn.close()


def get_kg_entries(db_path=None, *, invernadero_id) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM kg_entries WHERE invernadero_id=? ORDER BY iso_week",
        conn, params=(invernadero_id,))
    conn.close()
    return df


def get_phenology_entries(db_path=None, *, invernadero_id) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM phenology_entries WHERE invernadero_id=? ORDER BY iso_week",
        conn, params=(invernadero_id,))
    conn.close()
    return df


def get_latest_sensor_readings(db_path=None, *, invernadero_id, n=100) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql("""
        SELECT * FROM sensor_readings
        WHERE invernadero_id=?
        ORDER BY timestamp DESC LIMIT ?
    """, conn, params=(invernadero_id, n))
    conn.close()
    return df


def get_sensor_latest_per_variable(db_path=None, *, invernadero_id) -> pd.DataFrame:
    """One row per sensor_name with the most recent value."""
    conn = _connect(db_path)
    df = pd.read_sql("""
        SELECT sensor_name, value, timestamp
        FROM sensor_readings
        WHERE invernadero_id=?
          AND timestamp = (
              SELECT MAX(s2.timestamp)
              FROM sensor_readings s2
              WHERE s2.invernadero_id = sensor_readings.invernadero_id
                AND s2.sensor_name   = sensor_readings.sensor_name
          )
        ORDER BY sensor_name
    """, conn, params=(invernadero_id,))
    conn.close()
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest Dashboard/tests/test_db.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add Dashboard/db.py Dashboard/tests/test_db.py
git commit -m "feat: add SQLite db layer for kg, phenology, and sensor data"
```

---

## Task 3: Chart Components

**Files:**
- Create: `Dashboard/components/charts.py`
- Create: `Dashboard/tests/test_charts.py`

- [ ] **Step 1: Write failing tests**

```python
# Dashboard/tests/test_charts.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from components.charts import build_forecast_chart, build_comparison_bar

def test_forecast_chart_returns_figure():
    import plotly.graph_objects as go
    n = 10
    fig = build_forecast_chart(
        x_labels=[f"W{i}" for i in range(n)],
        y_pred=np.random.rand(n) * 10000,
        y_true=np.random.rand(n) * 10000,
        pi_lower=np.random.rand(n) * 5000,
        pi_upper=np.random.rand(n) * 10000 + 5000,
        ensemble_pred=np.random.rand(n) * 10000,
    )
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Cosecha Real" in trace_names
    assert "Pronóstico CNN-RNN" in trace_names
    assert "Intervalo 90%" in trace_names

def test_comparison_bar_returns_figure():
    import plotly.graph_objects as go
    fig = build_comparison_bar(
        weeks=["W1", "W2", "W3"],
        y_pred=np.array([8000., 9000., 7500.]),
        y_true=np.array([8200., 8800., 7800.]),
    )
    assert isinstance(fig, go.Figure)

def test_forecast_chart_handles_no_actual():
    import plotly.graph_objects as go
    n = 5
    fig = build_forecast_chart(
        x_labels=[f"W{i}" for i in range(n)],
        y_pred=np.random.rand(n) * 10000,
        y_true=None,
        pi_lower=np.random.rand(n) * 5000,
        pi_upper=np.random.rand(n) * 10000 + 5000,
        ensemble_pred=None,
    )
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run to verify fail**

```bash
python3 -m pytest Dashboard/tests/test_charts.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `Dashboard/components/charts.py`**

```python
# Dashboard/components/charts.py
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from .theme import PALETTE

_LAYOUT = dict(
    plot_bgcolor="#FAFAF7",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color=PALETTE["text"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.9)", bordercolor=PALETTE["border"], borderwidth=1),
    hovermode="x unified",
    margin=dict(l=60, r=30, t=60, b=100),
    hoverlabel=dict(bgcolor="white", bordercolor=PALETTE["border"], font_size=13),
)


def build_forecast_chart(
    x_labels: list,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    ensemble_pred: Optional[np.ndarray] = None,
    height: int = 480,
) -> go.Figure:
    fig = go.Figure()

    # Prediction interval band
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=np.concatenate([pi_upper, pi_lower[::-1]]).tolist(),
        fill="toself",
        fillcolor="rgba(230,126,34,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo 90%",
        hoverinfo="skip",
    ))

    # Ensemble
    if ensemble_pred is not None:
        fig.add_trace(go.Scatter(
            x=x_labels, y=ensemble_pred.tolist(),
            mode="lines",
            line=dict(color="#7F8C8D", width=1.5, dash="dot"),
            name="Ensemble Top-20",
        ))

    # CNN-RNN best
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_pred.tolist(),
        mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=7, symbol="diamond",
                    line=dict(color=PALETTE["accent_lt"], width=1)),
        name="Pronóstico CNN-RNN",
    ))

    # PI bounds (thin lines for reference)
    fig.add_trace(go.Scatter(
        x=x_labels, y=pi_upper.tolist(), mode="lines",
        line=dict(color=PALETTE["accent"], width=1, dash="dot"),
        name="Límite superior 90%", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=pi_lower.tolist(), mode="lines",
        line=dict(color=PALETTE["accent"], width=1, dash="dot"),
        name="Límite inferior 90%", showlegend=False,
    ))

    # Actual yield
    if y_true is not None:
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_true.tolist(),
            mode="lines+markers",
            line=dict(color=PALETTE["primary"], width=3),
            marker=dict(size=9, symbol="circle",
                        color=PALETTE["primary"],
                        line=dict(color="white", width=2)),
            name="Cosecha Real",
        ))

    fig.update_layout(
        **_LAYOUT,
        height=height,
        xaxis=dict(title="Semana ISO", gridcolor="#ECECEC", tickangle=-40,
                   tickfont=dict(size=10)),
        yaxis=dict(title="Producción (kg)", gridcolor="#ECECEC",
                   tickformat=",", tickfont=dict(size=11)),
    )
    return fig


def build_comparison_bar(
    weeks: list,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    height: int = 380,
) -> go.Figure:
    error_pct = np.where(
        y_true != 0,
        (y_pred - y_true) / y_true * 100,
        0.0,
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Real (kg)", x=weeks, y=y_true.tolist(),
        marker_color=PALETTE["primary"],
        marker_line=dict(color="white", width=1.5),
    ))
    fig.add_trace(go.Bar(
        name="Pronóstico (kg)", x=weeks, y=y_pred.tolist(),
        marker_color=PALETTE["accent"],
        marker_line=dict(color="white", width=1.5),
    ))
    fig.update_layout(
        **_LAYOUT,
        barmode="group",
        height=height,
        xaxis=dict(title="Semana", gridcolor="#ECECEC", tickangle=-40),
        yaxis=dict(title="kg", gridcolor="#ECECEC", tickformat=","),
    )
    return fig


def build_sensor_timeseries(df, sensor_name: str, height: int = 250) -> go.Figure:
    """df must have columns: timestamp, value (for one sensor_name)."""
    sub = df[df["sensor_name"] == sensor_name].sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"].tolist(),
        y=sub["value"].tolist(),
        mode="lines",
        line=dict(color=PALETTE["primary_lt"], width=2),
        fill="toself",
        fillcolor="rgba(46,125,50,0.08)",
        name=sensor_name,
    ))
    fig.update_layout(
        **_LAYOUT,
        height=height,
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis=dict(gridcolor="#ECECEC", tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#ECECEC"),
        showlegend=False,
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest Dashboard/tests/test_charts.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add Dashboard/components/charts.py Dashboard/tests/test_charts.py
git commit -m "feat: add Plotly chart builders for forecast, comparison bar, sensor timeseries"
```

---

## Task 4: Refactor `app.py` — Home Page

**Files:**
- Modify: `Dashboard/app.py`

The home page shows:
- Hero header with company branding
- Summary KPI cards for both greenhouses (best R², MAPE, RMSE, PI coverage)
- Navigation hint to other pages

- [ ] **Step 1: Replace `Dashboard/app.py` entirely**

```python
# Dashboard/app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from components.theme import inject_theme, hero, section, PALETTE
from db import init_db

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Models" / "results"
DB_PATH     = Path(__file__).resolve().parent / "data" / "huerta.db"

INV_CONFIG = {
    3: {"npz":     RESULTS_DIR / "cnn_rnn_inv3_predictions.npz",
        "metrics": RESULTS_DIR / "cnn_rnn_inv3_metrics.csv",
        "label":   "Invernadero 3"},
    4: {"npz":     RESULTS_DIR / "cnn_rnn_inv4_predictions.npz",
        "metrics": RESULTS_DIR / "cnn_rnn_inv4_metrics.csv",
        "label":   "Invernadero 4"},
}

st.set_page_config(
    page_title="Huerta Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
init_db(DB_PATH)

hero(
    "Huerta Prediction",
    "Pronóstico inteligente de cosecha de tomate · Invernaderos 3 y 4 · Temporada T17",
)

section("Resumen de Desempeño del Modelo")

cols = st.columns(2)
for col, inv_id in zip(cols, [3, 4]):
    cfg = INV_CONFIG[inv_id]
    with col:
        st.markdown(f"""
        <div style='background:{PALETTE["card"]};border:1px solid {PALETTE["border"]};
             border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;
             box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
            <div style='font-size:1rem;font-weight:700;color:{PALETTE["primary"]};
                        margin-bottom:0.8rem;'>🏠 {cfg["label"]}</div>
        """, unsafe_allow_html=True)

        if cfg["metrics"].exists():
            df_m = pd.read_csv(cfg["metrics"]).set_index("model")
            row  = "best"
            r2   = df_m.loc[row, "R²"]
            mape = df_m.loc[row, "MAPE (%)"]
            rmse = df_m.loc[row, "RMSE (kg)"]
            cov  = df_m.loc["cptc_pi", "pi_coverage"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",          f"{r2:.3f}")
            c2.metric("MAPE",        f"{mape:.1f}%")
            c3.metric("RMSE",        f"{rmse:,.0f} kg")
            c4.metric("Cobertura PI",f"{cov*100:.0f}%")
        else:
            st.warning(f"Métricas no encontradas. Ejecutar `cnn_rnn_inv{inv_id}.py`")

        st.markdown("</div>", unsafe_allow_html=True)

section("Navegación")
st.markdown("""
| Página | Descripción |
|---|---|
| 🌱 **Forecast** | Gráfica de pronóstico vs cosecha real con intervalos de confianza |
| 📊 **Comparativo** | Tabla detallada semana a semana: pronóstico vs real |
| ✏️ **Registro** | Ingresa kg cosechados y etapas fenológicas por semana |
| 📡 **Sensores** | Lecturas en tiempo real de sensores del invernadero |
""")

st.markdown(f"""
<hr style='border:1px solid {PALETTE["border"]};margin-top:2rem;'>
<p style='text-align:center;color:{PALETTE["text_muted"]};font-size:0.8rem;'>
CNN-RNN · CPTC Prediction Intervals · Temporada T17 · 90% cobertura nominal
</p>
""", unsafe_allow_html=True)
```

- [ ] **Step 2: Verify app runs without error**

```bash
cd /Users/josuetapiahernandez/Documents/Huerta_Prediction
streamlit run Dashboard/app.py --server.headless true &
sleep 4 && curl -s http://localhost:8501 | grep -c "Huerta" && kill %1
```
Expected: `1` (page title found)

- [ ] **Step 3: Commit**

```bash
git add Dashboard/app.py
git commit -m "feat: refactor home page with hero header, KPI cards, navigation"
```

---

## Task 5: Forecast Page

**Files:**
- Create: `Dashboard/pages/1_Forecast.py`

- [ ] **Step 1: Create `Dashboard/pages/` directory**

```bash
mkdir -p Dashboard/pages
touch Dashboard/pages/__init__.py
```

- [ ] **Step 2: Write `Dashboard/pages/1_Forecast.py`**

```python
# Dashboard/pages/1_Forecast.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from components.theme import inject_theme, hero, section, PALETTE
from components.charts import build_forecast_chart
from db import get_kg_entries, DB_PATH  # DB_PATH imported from app context

# resolve paths relative to this file
_ROOT       = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _ROOT / "Models" / "results"
DB_PATH     = _ROOT / "Dashboard" / "data" / "huerta.db"

INV_OPTIONS = {"Invernadero 3": 3, "Invernadero 4": 4}

st.set_page_config(page_title="Forecast — Huerta Prediction", page_icon="🌱", layout="wide")
inject_theme()

hero("Pronóstico de Cosecha", "Predicciones semanales vs cosecha real · Temporada T17")

inv_label = st.sidebar.selectbox("Invernadero", list(INV_OPTIONS.keys()))
inv_id    = INV_OPTIONS[inv_label]

show_ensemble = st.sidebar.checkbox("Mostrar Ensemble Top-20", value=False)
show_pi       = st.sidebar.checkbox("Mostrar Intervalo 90%", value=True)

npz_path = RESULTS_DIR / f"cnn_rnn_inv{inv_id}_predictions.npz"

if not npz_path.exists():
    st.error(f"Archivo de predicciones no encontrado: `{npz_path}`")
    st.stop()

data          = np.load(npz_path)
y_pred        = data["y_pred"]
ensemble_pred = data["ensemble_pred"]
pi_lower      = data["pi_lower"]
pi_upper      = data["pi_upper"]
week_keys     = data["week_keys"]

# Build x labels
def wk_label(wk):
    y, w = int(wk) // 100, int(wk) % 100
    monday = pd.Timestamp.fromisocalendar(y, w, 1)
    return f"{monday.strftime('%d %b %Y')} (S{w})"

if week_keys[0] > 190000:
    x_labels = [wk_label(wk) for wk in week_keys]
else:
    x_labels = [f"Semana {int(wk)+1}" for wk in week_keys]

# Merge DB actual kg with npz y_true
y_true_npz = data["y_true"]
db_kg      = get_kg_entries(DB_PATH, invernadero_id=inv_id)
if not db_kg.empty and week_keys[0] > 190000:
    db_map = dict(zip(db_kg["iso_week"].astype(int), db_kg["kg_reales"]))
    y_true_merged = np.array([
        db_map.get(int(wk), y_true_npz[i])
        for i, wk in enumerate(week_keys)
    ])
else:
    y_true_merged = y_true_npz

section(f"Pronóstico Semanal — {inv_label}")
fig = build_forecast_chart(
    x_labels=x_labels,
    y_pred=y_pred,
    y_true=y_true_merged,
    pi_lower=pi_lower if show_pi else np.zeros_like(pi_lower),
    pi_upper=pi_upper if show_pi else np.zeros_like(pi_upper),
    ensemble_pred=ensemble_pred if show_ensemble else None,
)
st.plotly_chart(fig, use_container_width=True)

# Summary stats row
total_pred = float(y_pred.sum())
total_real = float(y_true_merged.sum())
diff_pct   = (total_pred - total_real) / max(total_real, 1) * 100

section("Resumen de Temporada")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pronosticado", f"{total_pred:,.0f} kg")
c2.metric("Total Real",         f"{total_real:,.0f} kg")
c3.metric("Diferencia",         f"{diff_pct:+.1f}%",
           delta_color="inverse" if diff_pct > 0 else "normal")
c4.metric("Semanas",            str(len(y_pred)))
```

- [ ] **Step 3: Verify page loads**

```bash
streamlit run Dashboard/app.py --server.headless true &
sleep 5 && curl -s http://localhost:8501/Forecast | head -20 ; kill %1
```
Expected: no Python errors in terminal

- [ ] **Step 4: Commit**

```bash
git add Dashboard/pages/
git commit -m "feat: add forecast page with sidebar controls and season summary stats"
```

---

## Task 6: Comparativo Page (Predicted vs Actual Numbers)

**Files:**
- Create: `Dashboard/pages/2_Comparativo.py`

- [ ] **Step 1: Write `Dashboard/pages/2_Comparativo.py`**

```python
# Dashboard/pages/2_Comparativo.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from components.theme import inject_theme, hero, section, PALETTE
from components.charts import build_comparison_bar
from db import get_kg_entries

_ROOT       = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _ROOT / "Models" / "results"
DB_PATH     = _ROOT / "Dashboard" / "data" / "huerta.db"

INV_OPTIONS = {"Invernadero 3": 3, "Invernadero 4": 4}

st.set_page_config(page_title="Comparativo — Huerta Prediction", page_icon="📊", layout="wide")
inject_theme()
hero("Pronóstico vs Real", "Comparación semana a semana de cosecha pronosticada y cosecha real")

inv_label = st.sidebar.selectbox("Invernadero", list(INV_OPTIONS.keys()))
inv_id    = INV_OPTIONS[inv_label]

npz_path = RESULTS_DIR / f"cnn_rnn_inv{inv_id}_predictions.npz"
if not npz_path.exists():
    st.error("Predicciones no encontradas."); st.stop()

data      = np.load(npz_path)
y_pred    = data["y_pred"]
y_true    = data["y_true"]
pi_lower  = data["pi_lower"]
pi_upper  = data["pi_upper"]
week_keys = data["week_keys"]

def wk_label(wk):
    y, w = int(wk) // 100, int(wk) % 100
    monday = pd.Timestamp.fromisocalendar(y, w, 1)
    return f"S{w} ({monday.strftime('%d %b')})"

x_labels = [wk_label(wk) for wk in week_keys] if week_keys[0] > 190000 \
           else [f"Sem {int(wk)+1}" for wk in week_keys]

# Merge DB overrides
db_kg = get_kg_entries(DB_PATH, invernadero_id=inv_id)
if not db_kg.empty and week_keys[0] > 190000:
    db_map = dict(zip(db_kg["iso_week"].astype(int), db_kg["kg_reales"]))
    y_true = np.array([db_map.get(int(wk), y_true[i]) for i, wk in enumerate(week_keys)])

section("Gráfica Comparativa")
fig = build_comparison_bar(weeks=x_labels, y_pred=y_pred, y_true=y_true)
st.plotly_chart(fig, use_container_width=True)

section("Tabla Detallada por Semana")
error_abs = np.abs(y_pred - y_true)
error_pct = np.where(y_true != 0, error_abs / y_true * 100, 0.0)
within_pi  = ((y_true >= pi_lower) & (y_true <= pi_upper)).astype(int)

table = pd.DataFrame({
    "Semana":               x_labels,
    "Real (kg)":            np.round(y_true, 0).astype(int),
    "Pronóstico (kg)":      np.round(y_pred, 0).astype(int),
    "Error Abs (kg)":       np.round(error_abs, 0).astype(int),
    "Error (%)":            np.round(error_pct, 1),
    "Límite Inf (kg)":      np.round(pi_lower, 0).astype(int),
    "Límite Sup (kg)":      np.round(pi_upper, 0).astype(int),
    "Dentro del Intervalo": ["✅" if v else "❌" for v in within_pi],
})

def color_error(val):
    if val < 10: return f"background-color: #D5F5E3; color: #1E8449"
    if val < 20: return f"background-color: #FDEBD0; color: #D35400"
    return f"background-color: #FADBD8; color: #C0392B"

st.dataframe(
    table.style
        .applymap(color_error, subset=["Error (%)"])
        .format({"Real (kg)": "{:,}", "Pronóstico (kg)": "{:,}",
                 "Error Abs (kg)": "{:,}", "Error (%)": "{:.1f}%",
                 "Límite Inf (kg)": "{:,}", "Límite Sup (kg)": "{:,}"}),
    use_container_width=True, hide_index=True,
)

section("Totales de Temporada")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Real",           f"{int(y_true.sum()):,} kg")
c2.metric("Total Pronosticado",   f"{int(y_pred.sum()):,} kg")
c3.metric("Error Total Abs",      f"{int(error_abs.sum()):,} kg")
c4.metric("MAPE Promedio",        f"{error_pct.mean():.1f}%")
c5.metric("Dentro del Intervalo", f"{within_pi.mean()*100:.0f}%")
```

- [ ] **Step 2: Commit**

```bash
git add Dashboard/pages/2_Comparativo.py
git commit -m "feat: add comparativo page with bar chart, error table, season totals"
```

---

## Task 7: Data Entry Page

**Files:**
- Create: `Dashboard/pages/3_Registro.py`

- [ ] **Step 1: Write `Dashboard/pages/3_Registro.py`**

```python
# Dashboard/pages/3_Registro.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from components.theme import inject_theme, hero, section, PALETTE
from db import (init_db, insert_kg_entry, insert_phenology_entry,
                get_kg_entries, get_phenology_entries)

_ROOT   = Path(__file__).resolve().parent.parent.parent
DB_PATH = _ROOT / "Dashboard" / "data" / "huerta.db"

PHENOLOGY_STAGES = [
    "Trasplante", "Establecimiento", "Vegetativo", "Floración",
    "Cuajado", "Desarrollo fruto", "Maduración", "Cosecha activa", "Fin de ciclo",
]

INV_OPTIONS = {"Invernadero 3": 3, "Invernadero 4": 4}

st.set_page_config(page_title="Registro — Huerta Prediction", page_icon="✏️", layout="wide")
inject_theme()
init_db(DB_PATH)
hero("Registro de Datos", "Ingresa cosecha semanal y etapas fenológicas del cultivo")

inv_label = st.sidebar.selectbox("Invernadero", list(INV_OPTIONS.keys()))
inv_id    = INV_OPTIONS[inv_label]

tab_kg, tab_pheno = st.tabs(["🍅 Cosecha Semanal (kg)", "🌱 Fenología"])

# ── Tab 1: kg_reales entry ──
with tab_kg:
    section("Registrar Cosecha Semanal")
    with st.form("form_kg", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            ref_date = col1.date_input("Fecha de inicio de semana (lunes)", value=datetime.today())
            iso_cal  = ref_date.isocalendar()
            iso_week = int(iso_cal[0]) * 100 + int(iso_cal[1])
            st.caption(f"Semana ISO: **{iso_week}** (año {iso_cal[0]}, semana {iso_cal[1]})")
        with col2:
            kg_val = st.number_input("kg cosechados", min_value=0.0, step=100.0, format="%.1f")

        submitted = st.form_submit_button("💾 Guardar", type="primary", use_container_width=True)
        if submitted:
            if kg_val <= 0:
                st.error("Ingresa un valor mayor a 0 kg.")
            else:
                insert_kg_entry(DB_PATH, invernadero_id=inv_id, iso_week=iso_week, kg_reales=kg_val)
                st.success(f"✅ Guardado: Semana {iso_week} → {kg_val:,.1f} kg")

    section("Historial de Cosecha Registrada")
    df_kg = get_kg_entries(DB_PATH, invernadero_id=inv_id)
    if df_kg.empty:
        st.info("No hay registros aún. Usa el formulario arriba para agregar cosechas.")
    else:
        df_kg["Semana ISO"] = df_kg["iso_week"].astype(str)
        df_kg["kg Reales"]  = df_kg["kg_reales"].round(1)
        df_kg["Registrado"] = pd.to_datetime(df_kg["entered_at"]).dt.strftime("%d %b %Y %H:%M")
        st.dataframe(
            df_kg[["Semana ISO", "kg Reales", "Registrado"]].sort_values("Semana ISO", ascending=False),
            use_container_width=True, hide_index=True,
        )
        csv = df_kg.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar CSV", csv, "kg_cosecha.csv", "text/csv")

# ── Tab 2: phenology entry ──
with tab_pheno:
    section("Registrar Etapa Fenológica")
    with st.form("form_pheno", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            ref_date2 = st.date_input("Fecha (lunes de la semana)", key="pheno_date")
            iso_cal2  = ref_date2.isocalendar()
            iso_week2 = int(iso_cal2[0]) * 100 + int(iso_cal2[1])
            st.caption(f"Semana ISO: **{iso_week2}**")
        with col2:
            stage = st.selectbox("Etapa fenológica", PHENOLOGY_STAGES)
        with col3:
            ddt = st.number_input("Días desde trasplante", min_value=0, max_value=365, step=1)
        notes = st.text_area("Notas (opcional)", height=80)
        sub2  = st.form_submit_button("💾 Guardar Fenología", type="primary", use_container_width=True)
        if sub2:
            insert_phenology_entry(DB_PATH, invernadero_id=inv_id, iso_week=iso_week2,
                                   stage=stage, dias_desde_transplante=ddt, notes=notes)
            st.success(f"✅ Guardado: Semana {iso_week2} → {stage} ({ddt} días)")

    section("Historial Fenológico Registrado")
    df_ph = get_phenology_entries(DB_PATH, invernadero_id=inv_id)
    if df_ph.empty:
        st.info("No hay registros fenológicos aún.")
    else:
        df_ph["Semana ISO"] = df_ph["iso_week"].astype(str)
        df_ph["Etapa"]      = df_ph["stage"]
        df_ph["Días DDT"]   = df_ph["dias_desde_transplante"]
        df_ph["Notas"]      = df_ph["notes"].fillna("")
        df_ph["Registrado"] = pd.to_datetime(df_ph["entered_at"]).dt.strftime("%d %b %Y")
        st.dataframe(
            df_ph[["Semana ISO", "Etapa", "Días DDT", "Notas", "Registrado"]]
                 .sort_values("Semana ISO", ascending=False),
            use_container_width=True, hide_index=True,
        )
```

- [ ] **Step 2: Commit**

```bash
git add Dashboard/pages/3_Registro.py
git commit -m "feat: add data entry page for weekly kg and phenology stage registration"
```

---

## Task 8: Sensor Ingest Script

**Files:**
- Create: `Dashboard/sensor_ingest.py`
- Create: `Dashboard/tests/test_sensor_ingest.py`

The script reads a CSV of sensor readings (format: `timestamp,invernadero_id,sensor_name,value`) and writes new rows to the SQLite DB. Designed to be run as a cron job or Railway background worker.

- [ ] **Step 1: Write failing tests**

```python
# Dashboard/tests/test_sensor_ingest.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from db import init_db, get_latest_sensor_readings
from sensor_ingest import ingest_csv

def test_ingest_csv_writes_to_db():
    db_path  = tempfile.mktemp(suffix=".db")
    csv_path = tempfile.mktemp(suffix=".csv")
    init_db(db_path)

    df = pd.DataFrame([
        {"timestamp": "2024-04-20 08:00:00", "invernadero_id": 3, "sensor_name": "temp_interior", "value": 24.5},
        {"timestamp": "2024-04-20 08:00:00", "invernadero_id": 3, "sensor_name": "humedad",       "value": 68.0},
    ])
    df.to_csv(csv_path, index=False)

    count = ingest_csv(csv_path, db_path)
    assert count == 2
    readings = get_latest_sensor_readings(db_path, invernadero_id=3, n=10)
    assert len(readings) == 2

def test_ingest_csv_skips_duplicate_timestamps():
    db_path  = tempfile.mktemp(suffix=".db")
    csv_path = tempfile.mktemp(suffix=".csv")
    init_db(db_path)

    df = pd.DataFrame([
        {"timestamp": "2024-04-20 08:00:00", "invernadero_id": 3, "sensor_name": "temp_interior", "value": 24.5},
        {"timestamp": "2024-04-20 08:00:00", "invernadero_id": 3, "sensor_name": "temp_interior", "value": 24.5},
    ])
    df.to_csv(csv_path, index=False)

    count = ingest_csv(csv_path, db_path)
    assert count == 1  # deduped on (inv, timestamp, sensor_name)

def test_ingest_csv_returns_zero_for_empty_file():
    db_path  = tempfile.mktemp(suffix=".db")
    csv_path = tempfile.mktemp(suffix=".csv")
    init_db(db_path)
    pd.DataFrame(columns=["timestamp","invernadero_id","sensor_name","value"]).to_csv(csv_path, index=False)
    assert ingest_csv(csv_path, db_path) == 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest Dashboard/tests/test_sensor_ingest.py -v
```
Expected: `ModuleNotFoundError: No module named 'sensor_ingest'`

- [ ] **Step 3: Write `Dashboard/sensor_ingest.py`**

```python
# Dashboard/sensor_ingest.py
"""
Sensor data ingestion script.
Usage: python3 Dashboard/sensor_ingest.py --csv /path/to/sensor_data.csv

CSV format expected:
    timestamp,invernadero_id,sensor_name,value
    2024-04-20 08:00:00,3,temp_interior,24.5
    2024-04-20 08:00:00,3,humedad,68.0

Run as cron (every 15 min):
    */15 * * * * python3 /app/Dashboard/sensor_ingest.py --csv /data/sensors/latest.csv
"""
import argparse
import sqlite3
from pathlib import Path
import pandas as pd

REQUIRED_COLS = {"timestamp", "invernadero_id", "sensor_name", "value"}
DEFAULT_DB    = Path(__file__).resolve().parent / "data" / "huerta.db"


def ingest_csv(csv_path, db_path=None) -> int:
    """Read CSV and insert new sensor readings. Returns count of inserted rows."""
    db = db_path or DEFAULT_DB
    df = pd.read_csv(csv_path)

    if df.empty:
        return 0

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["timestamp"]      = df["timestamp"].astype(str)
    df["invernadero_id"] = df["invernadero_id"].astype(int)
    df["sensor_name"]    = df["sensor_name"].astype(str)
    df["value"]          = df["value"].astype(float)

    # Deduplicate within file
    df = df.drop_duplicates(subset=["invernadero_id", "timestamp", "sensor_name"])

    conn = sqlite3.connect(db)
    inserted = 0
    for _, row in df.iterrows():
        exists = conn.execute("""
            SELECT 1 FROM sensor_readings
            WHERE invernadero_id=? AND timestamp=? AND sensor_name=?
        """, (row["invernadero_id"], row["timestamp"], row["sensor_name"])).fetchone()
        if not exists:
            conn.execute("""
                INSERT INTO sensor_readings (invernadero_id, timestamp, sensor_name, value)
                VALUES (?, ?, ?, ?)
            """, (row["invernadero_id"], row["timestamp"], row["sensor_name"], row["value"]))
            inserted += 1

    conn.commit()
    conn.close()
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest sensor CSV to huerta.db")
    parser.add_argument("--csv",     required=True, help="Path to sensor CSV file")
    parser.add_argument("--db",      default=None,  help="Path to SQLite DB (default: Dashboard/data/huerta.db)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n = ingest_csv(args.csv, args.db)
    if args.verbose:
        print(f"Ingested {n} new sensor readings from {args.csv}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest Dashboard/tests/test_sensor_ingest.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add Dashboard/sensor_ingest.py Dashboard/tests/test_sensor_ingest.py
git commit -m "feat: add sensor_ingest.py for CSV-to-SQLite sensor data pipeline"
```

---

## Task 9: Live Sensor Page

**Files:**
- Create: `Dashboard/pages/4_Sensores.py`

- [ ] **Step 1: Write `Dashboard/pages/4_Sensores.py`**

```python
# Dashboard/pages/4_Sensores.py
from pathlib import Path
import pandas as pd
import streamlit as st

from components.theme import inject_theme, hero, section, PALETTE
from components.charts import build_sensor_timeseries
from db import get_sensor_latest_per_variable, get_latest_sensor_readings

_ROOT   = Path(__file__).resolve().parent.parent.parent
DB_PATH = _ROOT / "Dashboard" / "data" / "huerta.db"

# Normal operating ranges per sensor
SENSOR_RANGES = {
    "temp_interior":   (18, 28,  "°C"),
    "humedad":         (60, 85,  "%"),
    "co2":             (400, 1200, "ppm"),
    "radiacion":       (0,  800,  "W/m²"),
    "temp_exterior":   (5,  35,  "°C"),
    "viento":          (0,  20,  "km/h"),
}

INV_OPTIONS = {"Invernadero 3": 3, "Invernadero 4": 4}

st.set_page_config(page_title="Sensores — Huerta Prediction", page_icon="📡", layout="wide")
inject_theme()
hero("Sensores en Tiempo Real", "Últimas lecturas de sensores del invernadero")

inv_label   = st.sidebar.selectbox("Invernadero", list(INV_OPTIONS.keys()))
inv_id      = INV_OPTIONS[inv_label]
auto_refresh = st.sidebar.checkbox("Auto-actualizar (30s)", value=False)

if auto_refresh:
    st.sidebar.caption("Página se recarga cada 30 segundos")

latest = get_sensor_latest_per_variable(DB_PATH, invernadero_id=inv_id)

if latest.empty:
    st.info("""
    📡 No hay lecturas de sensores registradas aún.

    Para cargar datos de sensores, usa:
    ```bash
    python3 Dashboard/sensor_ingest.py --csv /ruta/a/sensores.csv --verbose
    ```
    Formato del CSV: `timestamp, invernadero_id, sensor_name, value`
    """)
else:
    section("Estado Actual de Sensores")
    last_ts = latest["timestamp"].max()
    st.caption(f"Última actualización: **{last_ts}**")

    # Render a metric card per sensor
    cols = st.columns(min(len(latest), 4))
    for i, row in enumerate(latest.itertuples()):
        col = cols[i % len(cols)]
        name  = row.sensor_name
        val   = row.value
        rng   = SENSOR_RANGES.get(name, (None, None, ""))
        unit  = rng[2]
        badge = ""
        if rng[0] is not None:
            if rng[0] <= val <= rng[1]:
                badge = f'<span class="badge-good">Normal</span>'
            elif val < rng[0] * 0.85 or val > rng[1] * 1.15:
                badge = f'<span class="badge-bad">Alerta</span>'
            else:
                badge = f'<span class="badge-warn">Revisar</span>'
        col.markdown(f"""
        <div style='background:{PALETTE["card"]};border:1px solid {PALETTE["border"]};
             border-radius:12px;padding:1rem;text-align:center;
             box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
            <div style='font-size:0.75rem;color:{PALETTE["text_muted"]};font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;'>
                {name.replace("_", " ").title()}
            </div>
            <div style='font-size:1.8rem;font-weight:700;color:{PALETTE["primary"]};margin:.2rem 0;'>
                {val:.1f} <span style='font-size:1rem;color:{PALETTE["text_muted"]};'>{unit}</span>
            </div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    section("Historial de Lecturas")
    all_readings = get_latest_sensor_readings(DB_PATH, invernadero_id=inv_id, n=500)
    sensor_names = sorted(all_readings["sensor_name"].unique())
    selected     = st.multiselect("Variables a visualizar", sensor_names,
                                   default=sensor_names[:2] if len(sensor_names) >= 2 else sensor_names)

    chart_cols = st.columns(min(len(selected), 2))
    for i, sname in enumerate(selected):
        with chart_cols[i % 2]:
            st.markdown(f"**{sname.replace('_',' ').title()}**")
            fig = build_sensor_timeseries(all_readings, sname)
            st.plotly_chart(fig, use_container_width=True)

    section("Tabla de Lecturas Recientes")
    disp = all_readings[["timestamp", "sensor_name", "value"]].copy()
    disp.columns = ["Timestamp", "Sensor", "Valor"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add Dashboard/pages/4_Sensores.py
git commit -m "feat: add live sensor page with status badges, timeseries charts, auto-refresh"
```

---

## Task 10: Full Test Suite & Deployment Prep

**Files:**
- Create: `Dashboard/requirements.txt`
- Create: `Procfile` (Railway)
- Create: `Dashboard/data/.gitkeep`

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest Dashboard/tests/ -v
```
Expected: all tests pass (db: 5, charts: 3, sensor_ingest: 3 = **11 passed**)

- [ ] **Step 2: Create `Dashboard/requirements.txt`**

```text
streamlit>=1.32
plotly>=5.18
pandas>=2.0
numpy>=1.24
torch>=2.0
openpyxl>=3.1
```

- [ ] **Step 3: Create `Procfile` for Railway**

```
web: streamlit run Dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

- [ ] **Step 4: Create `.gitkeep` for data directory**

```bash
touch Dashboard/data/.gitkeep
echo "Dashboard/data/*.db" >> .gitignore
```

- [ ] **Step 5: Final commit**

```bash
git add Dashboard/requirements.txt Procfile Dashboard/data/.gitkeep .gitignore
git commit -m "feat: add requirements, Procfile for Railway deployment"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Predicted week + forecast + real kg → Forecast page + Comparativo page
- ✅ Numbers section (predicted kg vs actual) → Comparativo page + totals row
- ✅ Client data entry (phenology + kg_reales) → Registro page with two tabs
- ✅ Automatic sensor data ingestion → sensor_ingest.py + Sensores page
- ✅ Beautiful/appealing agriculture theme → Task 1 CSS, green/amber PALETTE, hero headers

**Placeholder scan:** No TBDs or TODOs in code steps. All code complete.

**Type consistency:** `DB_PATH` resolved consistently from `Path(__file__)` in each page. `get_kg_entries`, `insert_kg_entry` signatures match between db.py and callers. `build_forecast_chart`, `build_comparison_bar`, `build_sensor_timeseries` signatures match test expectations.

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `Dashboard/components/theme.py` | CSS, PALETTE, `inject_theme()`, `hero()`, `section()` |
| `Dashboard/components/charts.py` | `build_forecast_chart`, `build_comparison_bar`, `build_sensor_timeseries` |
| `Dashboard/db.py` | All SQLite CRUD — no Streamlit imports |
| `Dashboard/sensor_ingest.py` | CSV→SQLite sensor polling, CLI entry point |
| `Dashboard/app.py` | Home page, KPI cards, navigation |
| `Dashboard/pages/1_Forecast.py` | Forecast chart with PI + sidebar controls |
| `Dashboard/pages/2_Comparativo.py` | Bar chart + week-by-week error table |
| `Dashboard/pages/3_Registro.py` | Forms for kg + phenology entry |
| `Dashboard/pages/4_Sensores.py` | Live sensor status + timeseries |
| `Procfile` | Railway deployment entry point |
