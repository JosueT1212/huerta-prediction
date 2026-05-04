"""
Greenhouse Crop Yield Prediction — Client Dashboard
Launch: streamlit run Dashboard/app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Models" / "results"

INV_CONFIG = {
    "Invernadero 3": {"npz": RESULTS_DIR / "cnn_rnn_inv3_predictions.npz",
                      "metrics": RESULTS_DIR / "cnn_rnn_inv3_metrics.csv", "id": 3},
    "Invernadero 4": {"npz": RESULTS_DIR / "cnn_rnn_inv4_predictions.npz",
                      "metrics": RESULTS_DIR / "cnn_rnn_inv4_metrics.csv", "id": 4},
}

st.set_page_config(page_title="Huerta Prediction — Yield Dashboard",
                   page_icon="🌿", layout="wide")

st.markdown("""
<div style='text-align:center; padding:1.2rem 0 0.5rem 0;'>
    <h1 style='font-size:2.4rem; color:#2E7D32; margin-bottom:0.1rem;'>🌿 Huerta Prediction</h1>
    <p style='color:#555; font-size:1.1rem; margin-top:0;'>
        CNN-RNN Tomato Yield Forecast · Greenhouse Performance Dashboard
    </p>
    <hr style='border:1px solid #C8E6C9; margin-top:0.8rem;'>
</div>
""", unsafe_allow_html=True)


def week_key_to_label(wk: int) -> str:
    """Convert YYYY*100+W (e.g. 202432) to 'Aug 05, 2024 (W32)'."""
    year, week = int(wk) // 100, int(wk) % 100
    monday = pd.Timestamp.fromisocalendar(year, week, 1)
    return f"{monday.strftime('%b %d, %Y')} (W{week})"


def render_greenhouse(cfg: dict):
    npz_path, csv_path = cfg["npz"], cfg["metrics"]

    if not npz_path.exists():
        st.error(f"Predictions file not found: `{npz_path}`\n\n"
                 f"Re-run: `python3 Models/cnn_rnn_inv{cfg['id']}.py`")
        return

    data = np.load(npz_path)
    y_true        = data["y_true"]
    y_pred        = data["y_pred"]
    ensemble_pred = data["ensemble_pred"]
    pi_lower      = data["pi_lower"]
    pi_upper      = data["pi_upper"]
    week_keys     = data["week_keys"]

    if week_keys[0] > 190000:
        x_labels = [week_key_to_label(wk) for wk in week_keys]
    else:
        x_labels = [f"Week {int(wk)+1}" for wk in week_keys]

    metrics_df = pd.read_csv(csv_path).set_index("model")
    r2   = metrics_df.loc["best", "R²"]
    mape = metrics_df.loc["best", "MAPE (%)"]
    rmse = metrics_df.loc["best", "RMSE (kg)"]
    cov  = metrics_df.loc["cptc_pi", "pi_coverage"]

    st.markdown("#### Model Performance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (Best Model)",  f"{r2:.3f}",       help="Closer to 1 is better.")
    k2.metric("MAPE (%)",         f"{mape:.1f}%",     help="Mean Absolute % Error on T17.")
    k3.metric("RMSE (kg)",        f"{rmse:,.0f} kg",  help="Root Mean Squared Error.")
    k4.metric("90% PI Coverage",  f"{cov*100:.1f}%",  help="Actual values inside the 90% band.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Actual vs. Predicted — Test Season (T17)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=np.concatenate([pi_upper, pi_lower[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Prediction Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=ensemble_pred.tolist(), mode="lines",
        line=dict(color="#43A047", width=2, dash="dot"),
        name="Ensemble (Top-20)",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_pred.tolist(), mode="lines+markers",
        line=dict(color="#FB8C00", width=2.5, dash="dash"),
        marker=dict(size=5), name="Best Prediction",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_true.tolist(), mode="lines+markers",
        line=dict(color="#1565C0", width=3),
        marker=dict(size=7, symbol="circle"), name="Actual Yield",
    ))
    fig.update_layout(
        xaxis=dict(title="ISO Week (Test Season T17)", gridcolor="#E0E0E0",
                   tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title="Tomato Yield (kg)", gridcolor="#E0E0E0", tickformat=","),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=500,
        margin=dict(l=60, r=30, t=50, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Results by Week")
    error_pct = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)) * 100
    table_df = pd.DataFrame({
        "Week":             x_labels,
        "Actual (kg)":      np.round(y_true, 1),
        "Predicted (kg)":   np.round(y_pred, 1),
        "Lower Bound (kg)": np.round(pi_lower, 1),
        "Upper Bound (kg)": np.round(pi_upper, 1),
        "Error (%)":        np.round(error_pct, 2),
    })
    st.dataframe(
        table_df.style
            .format({"Actual (kg)": "{:,.1f}", "Predicted (kg)": "{:,.1f}",
                     "Lower Bound (kg)": "{:,.1f}", "Upper Bound (kg)": "{:,.1f}",
                     "Error (%)": "{:.2f}%"})
            .background_gradient(subset=["Error (%)"], cmap="RdYlGn_r", vmin=0, vmax=30),
        use_container_width=True, hide_index=True,
    )


tab3, tab4 = st.tabs(["🏠 Invernadero 3", "🏠 Invernadero 4"])
with tab3:
    render_greenhouse(INV_CONFIG["Invernadero 3"])
with tab4:
    render_greenhouse(INV_CONFIG["Invernadero 4"])

st.markdown("""
<hr style='margin-top:2rem; border:1px solid #E0E0E0;'>
<p style='text-align:center; color:#9E9E9E; font-size:0.85rem;'>
    CNN-RNN model · CPTC Prediction Intervals · Test season T17 · 90% nominal coverage
</p>
""", unsafe_allow_html=True)
