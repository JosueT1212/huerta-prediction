# Huerta Prediction

Weekly tomato yield forecasting for commercial greenhouses. A CNN-RNN model
predicts production (kg/week) **5 weeks ahead** from in-greenhouse sensor
readings and external weather, served through a FastAPI backend and a live
client dashboard.

## What it does

Two greenhouses (invernaderos 3 and 4), five growing seasons of history
(T13–T17). Sensors start logging ~10–11 weeks before harvest begins each
season; the model exploits that natural sensor→harvest gap to forecast forward
without leaking future data.

- **Model** — 1D CNN blocks (weight norm, residual connections) feeding LSTM
  layers, with a feed-forward head. Box-Cox target transform, MinMax scaling,
  PCA at 95% retained variance.
- **Horizon** — 5 weeks, held exactly across all five seasons via a per-season
  gap-normalization step (`_apply_gap_norm_cnn`).
- **Split** — train T13–T15, validate T16, test T17.
- **Selection** — 216-run grid search (54 seeds × 4 inits) per greenhouse, then
  a single production refit over all five seasons at the best configuration.
- **Evaluation** — RMSE, R², NSE, PBIAS, MAPE.

## Stack

| Layer | Tech |
|---|---|
| Model | PyTorch (CNN-RNN), scikit-learn, statsmodels, XGBoost |
| Backend | FastAPI, Python 3.11 |
| Data | Supabase (PostgreSQL + Auth) |
| Ingest | Telegraf → `/ingest/sensors` |
| Deploy | Docker on Railway |
| Dashboard | Server-rendered HTML + Chart.js |

## Layout

```
Models/          training scripts, hyperparameter configs, saved weights
backend/         FastAPI app, auth, routers, inference engine
scripts/         live inference, ingest utilities
Dashboard/       client-facing dashboard
supabase/        SQL migrations
telegraf/        sensor collector config
docs/            design specs and implementation plans
EDA/             exploratory analysis
Report/          LaTeX write-up
```

## Baselines compared

ARIMAX, XGBoost, ElasticNet, SVR, FFNN, TFT, iTransformer, Chronos, a
hierarchical model, and a pure-LSTM ablation (no CNN blocks). CNN+LSTM wins on
both greenhouses — see `Models/results/*_metrics.csv`.

## A note on the data

**The greenhouse data is not included in this repository.** Sensor readings,
weekly production figures, phenology monitoring, and irrigation logs are the
grower's proprietary records and are kept private. `Data/` and the derived
artifacts that embed real production values (prediction archives, fitted
scalers, actual-vs-predicted plots) are excluded.

Code that reads them — `Models/*.py`, `backend/data_api.py` — expects `Data/`
with the Excel files described in `CLAUDE.md`. Trained model weights
(`Models/results/*.pt`) and aggregate metrics (`*_metrics.csv`) are included,
so the architecture and results are fully inspectable.

## Running

```bash
pip install -r requirements.txt

python Models/cnn_rnn_yield.py      # train (requires Data/)
uvicorn backend.main:app --reload   # backend + dashboard
```

Environment variables: see `.env.example` and `docs/deployment-setup.md`.
