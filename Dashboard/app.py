"""
Huerta Prediction — Client Dashboard
Dark theme, agriculture-grade forecasting interface
Launch: python3 -m streamlit run Dashboard/app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Models" / "results"

INV_CFG = {
    3: {
        "npz":     RESULTS_DIR / "cnn_rnn_inv3_predictions.npz",
        "metrics": RESULTS_DIR / "cnn_rnn_inv3_metrics.csv",
        "label":   "Invernadero 3",
        "color":   "#3FB950",
    },
    4: {
        "npz":     RESULTS_DIR / "cnn_rnn_inv4_predictions.npz",
        "metrics": RESULTS_DIR / "cnn_rnn_inv4_metrics.csv",
        "label":   "Invernadero 4",
        "color":   "#58A6FF",
    },
}

# ── Dark theme palette ─────────────────────────────────────────────────────
BG       = "#0D1117"
CARD     = "#161B22"
CARD2    = "#1C2128"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"
GREEN    = "#3FB950"
GREEN_DIM= "#238636"
AMBER    = "#F0883E"
AMBER_DIM= "#9E6A03"
BLUE     = "#58A6FF"
RED      = "#F85149"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Huerta Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* Hide default Streamlit header/footer */
#MainMenu, footer, header {{ visibility: hidden; }}

/* Sidebar dark */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}

/* Remove default padding */
.block-container {{ padding-top: 1rem !important; padding-bottom: 0 !important; }}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    background: {CARD} !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid {BORDER} !important;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 6px !important;
    color: {MUTED} !important;
    font-weight: 500 !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: {GREEN_DIM} !important;
    color: white !important;
}}

/* DataFrames */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] thead tr th {{
    background: {CARD2} !important;
    color: {MUTED} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}
[data-testid="stDataFrame"] tbody tr td {{
    background: {CARD} !important;
    color: {TEXT} !important;
    border-bottom: 1px solid {BORDER} !important;
}}

/* Plotly charts background */
.js-plotly-plot {{ border-radius: 10px; overflow: hidden; }}

/* Selectbox / widgets */
[data-testid="stSelectbox"] > div, [data-testid="stCheckbox"] {{
    color: {TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def wk_to_label(wk: int) -> str:
    y, w = int(wk) // 100, int(wk) % 100
    monday = pd.Timestamp.fromisocalendar(y, w, 1)
    return f"{monday.strftime('%d %b')} (S{w})"


def load_inv(inv_id: int):
    cfg = INV_CFG[inv_id]
    data = np.load(cfg["npz"])
    y_true    = data["y_true"]
    y_pred    = data["y_pred"]
    ensemble  = data["ensemble_pred"]
    pi_lower  = data["pi_lower"]
    pi_upper  = data["pi_upper"]
    wk_keys   = data["week_keys"]
    x_labels  = [wk_to_label(wk) for wk in wk_keys] if wk_keys[0] > 190000 \
                else [f"Sem {int(k)+1}" for k in wk_keys]
    metrics = None
    if cfg["metrics"].exists():
        try:
            metrics = pd.read_csv(cfg["metrics"]).set_index("model")
        except Exception:
            pass
    return y_true, y_pred, ensemble, pi_lower, pi_upper, x_labels, metrics


def make_forecast_fig(x, y_true, y_pred, ensemble, pi_lower, pi_upper,
                      show_ensemble, color):
    fig = go.Figure()

    # PI band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=np.concatenate([pi_upper, pi_lower[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(240,136,62,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo 90%", hoverinfo="skip",
    ))

    # PI bounds
    for arr, name in [(pi_upper, "Límite sup"), (pi_lower, "Límite inf")]:
        fig.add_trace(go.Scatter(
            x=x, y=arr.tolist(), mode="lines",
            line=dict(color=AMBER, width=1, dash="dot"),
            name=name, showlegend=False,
        ))

    # Ensemble
    if show_ensemble:
        fig.add_trace(go.Scatter(
            x=x, y=ensemble.tolist(), mode="lines",
            line=dict(color=MUTED, width=1.5, dash="dot"),
            name="Ensemble top-20",
        ))

    # Prediction
    fig.add_trace(go.Scatter(
        x=x, y=y_pred.tolist(), mode="lines+markers",
        line=dict(color=AMBER, width=2.5),
        marker=dict(size=7, symbol="diamond", color=AMBER,
                    line=dict(color=BG, width=1.5)),
        name="Pronóstico CNN-RNN",
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=x, y=y_true.tolist(), mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=9, color=color, line=dict(color=BG, width=2)),
        name="Cosecha Real",
    ))

    fig.update_layout(
        plot_bgcolor=CARD, paper_bgcolor=CARD,
        font=dict(family="Inter", color=TEXT),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=12),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font_size=12, font_color=TEXT),
        margin=dict(l=60, r=20, t=50, b=80),
        height=460,
        xaxis=dict(
            gridcolor=BORDER, tickangle=-35,
            tickfont=dict(size=10, color=MUTED),
            title=dict(text="Semana ISO · T17", font=dict(color=MUTED, size=11)),
            showline=True, linecolor=BORDER,
        ),
        yaxis=dict(
            gridcolor=BORDER,
            tickfont=dict(size=11, color=MUTED),
            title=dict(text="Producción (kg)", font=dict(color=MUTED, size=11)),
            tickformat=",",
            showline=True, linecolor=BORDER,
        ),
    )
    return fig


def make_error_bar_fig(x, y_true, y_pred):
    error = y_pred - y_true
    colors = [GREEN if abs(e)/max(t,1)*100 < 15
              else AMBER if abs(e)/max(t,1)*100 < 25 else RED
              for e, t in zip(error, y_true)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=error.tolist(),
        marker_color=colors,
        marker_line=dict(color=BG, width=1),
        name="Error (kg)",
        hovertemplate="%{y:,.0f} kg<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="solid"))
    fig.update_layout(
        plot_bgcolor=CARD, paper_bgcolor=CARD,
        font=dict(family="Inter", color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        margin=dict(l=60, r=20, t=30, b=80),
        height=240,
        xaxis=dict(gridcolor=BORDER, tickangle=-35,
                   tickfont=dict(size=9, color=MUTED), showline=True, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10, color=MUTED),
                   title=dict(text="Error (kg)", font=dict(color=MUTED, size=10)),
                   tickformat=",", showline=True, linecolor=BORDER),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font_color=TEXT),
    )
    return fig


def card(label: str, value: str, sub: str = "", color: str = TEXT):
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
         padding:1rem 1.2rem;'>
        <div style='font-size:0.7rem;font-weight:600;color:{MUTED};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem;'>
            {label}
        </div>
        <div style='font-size:1.55rem;font-weight:700;color:{color};line-height:1.1;'>
            {value}
        </div>
        {f'<div style="font-size:0.75rem;color:{MUTED};margin-top:.2rem;">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style='margin:1.5rem 0 .75rem 0;'>
        <div style='font-size:0.95rem;font-weight:600;color:{TEXT};
                    border-left:3px solid {GREEN};padding-left:.65rem;'>
            {title}
        </div>
        {f'<div style="font-size:0.78rem;color:{MUTED};padding-left:.9rem;margin-top:.1rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# ── Top nav bar ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
     padding:.6rem 1.5rem;background:{CARD};border-bottom:1px solid {BORDER};
     margin:-1rem -4rem 1.5rem -4rem;'>
    <div style='display:flex;align-items:center;gap:.7rem;'>
        <span style='font-size:1.35rem;'>🌿</span>
        <span style='font-size:1.05rem;font-weight:700;color:{TEXT};'>Huerta Prediction</span>
        <span style='background:{GREEN_DIM};color:white;font-size:0.65rem;font-weight:600;
                     padding:2px 8px;border-radius:20px;letter-spacing:.04em;'>LIVE</span>
    </div>
    <div style='font-size:0.78rem;color:{MUTED};'>
        CNN-RNN · Temporada T17 · 28 semanas
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────
def render_greenhouse(inv_id: int):
    cfg   = INV_CFG[inv_id]
    color = cfg["color"]

    if not cfg["npz"].exists():
        st.error(f"Predicciones no encontradas. Ejecutar `cnn_rnn_inv{inv_id}.py`")
        return

    y_true, y_pred, ensemble, pi_lower, pi_upper, x_labels, metrics = load_inv(inv_id)

    error_pct = np.abs(y_pred - y_true) / np.where(y_true == 0, 1, y_true) * 100
    within_pi = ((y_true >= pi_lower) & (y_true <= pi_upper)).mean()

    # ── KPI row ──
    r2, mape, rmse, cov_pi = None, None, None, None
    if metrics is not None:
        try:
            r2   = metrics.loc["best", "R²"]
            mape = metrics.loc["best", "MAPE (%)"]
            rmse = metrics.loc["best", "RMSE (kg)"]
            cov_pi = metrics.loc["cptc_pi", "pi_coverage"] if "cptc_pi" in metrics.index else None
        except KeyError:
            pass

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: card("R² (Test T17)", f"{r2:.3f}" if r2 else "—",
                  "Coef. determinación", color)
    with c2: card("MAPE", f"{mape:.1f}%" if mape else "—",
                  "Error porcentual medio")
    with c3: card("RMSE", f"{rmse:,.0f} kg" if rmse else "—",
                  "Error cuadrático medio")
    with c4: card("Cobertura PI 90%", f"{cov_pi*100:.0f}%" if cov_pi else "—",
                  "CPTC prediction intervals")
    with c5: card("Total Pronosticado", f"{y_pred.sum():,.0f} kg",
                  "Suma temporada T17", AMBER)
    with c6: card("Total Real", f"{y_true.sum():,.0f} kg",
                  "Suma temporada T17", color)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Controls row ──
    col_ctrl, col_info = st.columns([3, 1])
    with col_ctrl:
        show_ensemble = st.checkbox("Mostrar Ensemble Top-20", value=False, key=f"ens_{inv_id}")
    with col_info:
        diff = y_pred.sum() - y_true.sum()
        diff_pct = diff / max(y_true.sum(), 1) * 100
        color_diff = GREEN if abs(diff_pct) < 5 else AMBER if abs(diff_pct) < 15 else RED
        st.markdown(f"""
        <div style='text-align:right;padding:.3rem 0;'>
            <span style='font-size:0.75rem;color:{MUTED};'>Diferencia total: </span>
            <span style='font-size:0.9rem;font-weight:600;color:{color_diff};'>
                {diff_pct:+.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Main forecast chart ──
    section_header("Pronóstico vs Cosecha Real",
                   "Predicciones CNN-RNN con intervalo de confianza 90% (CPTC)")
    fig = make_forecast_fig(x_labels, y_true, y_pred, ensemble,
                            pi_lower, pi_upper, show_ensemble, color)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Error chart ──
    section_header("Error de Predicción por Semana",
                   "Verde < 15% · Ámbar 15–25% · Rojo > 25%")
    fig_err = make_error_bar_fig(x_labels, y_true, y_pred)
    st.plotly_chart(fig_err, use_container_width=True, config={"displayModeBar": False})

    # ── Table ──
    section_header("Tabla de Resultados Semana a Semana")
    within_arr = ((y_true >= pi_lower) & (y_true <= pi_upper))
    df = pd.DataFrame({
        "Semana":             x_labels,
        "Real (kg)":          np.round(y_true).astype(int),
        "Pronóstico (kg)":    np.round(y_pred).astype(int),
        "Error (kg)":         np.round(y_pred - y_true).astype(int),
        "Error (%)":          np.round(error_pct, 1),
        "Límite Inf (kg)":    np.round(pi_lower).astype(int),
        "Límite Sup (kg)":    np.round(pi_upper).astype(int),
        "En Intervalo":       ["✅" if v else "❌" for v in within_arr],
    })

    def style_error(v):
        if v < 10:  return "background-color:#0D2E16;color:#3FB950"
        if v < 20:  return "background-color:#2D1B00;color:#F0883E"
        return "background-color:#2D0F0F;color:#F85149"

    st.dataframe(
        df.style
          .map(style_error, subset=["Error (%)"])
          .format({"Real (kg)": "{:,}", "Pronóstico (kg)": "{:,}",
                   "Error (kg)": "{:+,}", "Error (%)": "{:.1f}%",
                   "Límite Inf (kg)": "{:,}", "Límite Sup (kg)": "{:,}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ── Season summary footer ──
    st.markdown(f"""
    <div style='display:flex;gap:1rem;margin-top:1rem;'>
        <div style='flex:1;background:{CARD};border:1px solid {BORDER};
             border-radius:8px;padding:.75rem 1rem;text-align:center;'>
            <div style='font-size:.68rem;color:{MUTED};font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;'>Semanas totales</div>
            <div style='font-size:1.4rem;font-weight:700;color:{TEXT};'>{len(y_pred)}</div>
        </div>
        <div style='flex:1;background:{CARD};border:1px solid {BORDER};
             border-radius:8px;padding:.75rem 1rem;text-align:center;'>
            <div style='font-size:.68rem;color:{MUTED};font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;'>MAPE temporada</div>
            <div style='font-size:1.4rem;font-weight:700;color:{AMBER};'>{error_pct.mean():.1f}%</div>
        </div>
        <div style='flex:1;background:{CARD};border:1px solid {BORDER};
             border-radius:8px;padding:.75rem 1rem;text-align:center;'>
            <div style='font-size:.68rem;color:{MUTED};font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;'>Dentro del intervalo</div>
            <div style='font-size:1.4rem;font-weight:700;color:{GREEN};'>{within_pi*100:.0f}%</div>
        </div>
        <div style='flex:1;background:{CARD};border:1px solid {BORDER};
             border-radius:8px;padding:.75rem 1rem;text-align:center;'>
            <div style='font-size:.68rem;color:{MUTED};font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;'>Mejor semana (error)</div>
            <div style='font-size:1.4rem;font-weight:700;color:{GREEN};'>
                {error_pct.min():.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
tab3, tab4 = st.tabs(["🏠 Invernadero 3", "🏠 Invernadero 4"])
with tab3:
    render_greenhouse(3)
with tab4:
    render_greenhouse(4)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:2rem;padding:1rem;border-top:1px solid {BORDER};
     display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-size:.75rem;color:{MUTED};'>
        CNN-RNN · CPTC Prediction Intervals · Temporada T17 · 90% cobertura nominal
    </span>
    <span style='font-size:.75rem;color:{MUTED};'>
        Huerta Prediction © 2025
    </span>
</div>
""", unsafe_allow_html=True)
