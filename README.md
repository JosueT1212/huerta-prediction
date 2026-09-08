# Huerta Prediction

[![tests](https://github.com/JosueT1212/huerta-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/JosueT1212/huerta-prediction/actions/workflows/tests.yml)

Weekly tomato yield forecasting for commercial greenhouses. A CNN-RNN model
predicts production (kg/week) **5 weeks ahead** from in-greenhouse sensor
readings and external weather, served through a FastAPI backend and a live
client dashboard.

## Results (test season T17, 5-week horizon)

| Greenhouse | Model | R² | RMSE (kg) | MAPE (%) |
|---|---|---|---|---|
| Inv 3 | **CNN+LSTM** | **0.70** | **4 806** | **12.4** |
| Inv 3 | Pure LSTM (no CNN blocks) | 0.45 | 6 098 | 14.5 |
| Inv 3 | SVR | 0.41 | 6 316 | 17.6 |
| Inv 3 | ElasticNet | 0.04 | 8 084 | 19.8 |
| Inv 4 | **CNN+LSTM** | **0.68** | **6 015** | **14.1** |
| Inv 4 | Pure LSTM (no CNN blocks) | 0.50 | 6 994 | 14.1 |
| Inv 4 | SVR | 0.11 | 9 348 | 17.2 |
| Inv 4 | ElasticNet | −0.03 | 10 052 | 17.4 |

Full metrics (NSE, PBIAS, top-20 ensemble, 90% conformal interval coverage)
in `Models/results/*_metrics.csv`. All models share the same split, horizon and
gap normalization; the pure-LSTM row is the same network with `num_cnn_blocks=0`.

## What it does

Two greenhouses (invernaderos 3 and 4), five growing seasons of history
(T13–T17). Sensors start logging ~10–11 weeks before harvest begins each
season; the model exploits that natural sensor→harvest gap to forecast forward
without leaking future data.

- **Model** — 1D CNN blocks (weight norm, residual connections) feeding LSTM
  layers, with a feed-forward head. Residual target (delta vs. previous week),
  focal loss, Box-Cox transform, MinMax scaling, PCA at 95% retained variance.
- **Horizon** — 5 weeks, held exactly across all five seasons via a per-season
  gap-normalization step (`_apply_gap_norm_cnn`).
- **Split** — train T13–T15, validate T16, test T17.
- **Selection** — 216-run grid search (54 seeds × 4 inits) per greenhouse, then
  a single production refit over all five seasons at the best configuration.
- **Serving** — growers upload weekly Excel files; the backend rebuilds the
  feature window, runs the production `.pt`, and stores predictions in
  Supabase. Telegraf can push live sensor readings to `/ingest/sensors`.

## Where to look

The repo is a full product (model + backend + dashboard + infra). If you have
ten minutes, read this slice:

1. `backend/engine.py` — loads the production checkpoint and runs inference on a feature window.
2. `scripts/live_inference.py` — walks a live season week by week, stops at the edge of real data.
3. `backend/live_features.py` — rebuilds the training-time feature pipeline from uploaded rows.
4. `Models/cnn_rnn_yield.py` — training script: data loading, gap normalization, `CNNRNN`, grid search.

## Try it without the data

The greenhouse data is private (see below), so the training and live paths need
credentials. Two things run standalone:

- **Static dashboard mockup** — open `Dashboard/demo.html` in a browser. The
  client UI (overview, per-greenhouse forecast, confidence bands, season
  history) with placeholder numbers, no backend needed. The real dashboard
  (`demo/Demo Dashboard.html`) is served by FastAPI and reads live predictions.
- **Tests** — `pip install -r backend/requirements.txt && pytest`. 138 unit
  tests cover the feature pipeline, inference trigger, upload validation,
  auth, and model helpers. Two tests that need `Data/` skip automatically.

The live dashboard runs on Railway for the grower; a demo login is available on
request.

## Stack

| Layer | Tech |
|---|---|
| Model | PyTorch (CNN-RNN), scikit-learn |
| Backend | FastAPI, Python 3.11 |
| Data | Supabase (PostgreSQL + Auth) |
| Ingest | Telegraf → `/ingest/sensors` |
| Deploy | Docker on Railway |
| Dashboard | Server-rendered HTML + Chart.js |

## Layout

```
Models/          training scripts, hyperparameter configs, saved weights, metrics
backend/         FastAPI app, auth, routers, inference engine, tests
scripts/         live inference, Excel replay for ingest testing
Dashboard/       static UI mockup (demo.html) and an earlier Streamlit prototype
demo/            production dashboard HTML + login page, served by backend/main.py
supabase/        SQL migrations
telegraf/        sensor collector config
docs/            design specs and implementation plans
Report/          LaTeX write-up
```

## Who wrote what

Huerta Prediction is a co-founded project. This repository is Josué Tapia
Hernández's work: the data pipeline, the CNN-RNN model and all baselines, the
grid search and production refit, the FastAPI backend, the live inference
pipeline, the Supabase schema, the deployment, and the tests (284 of 291
commits). Co-founders contributed early iterations of the static demo
dashboard HTML.

## A note on the data

**The greenhouse data is not included in this repository.** Sensor readings,
weekly production figures, phenology monitoring, and irrigation logs are the
grower's proprietary records and are kept private. `Data/` and the derived
artifacts that embed real production values (prediction archives, fitted
scalers, actual-vs-predicted plots) are excluded.

Code that reads them — `Models/*.py`, `backend/data_api.py`,
`scripts/mock_ingest.py` — expects `Data/` with the Excel files described in
`CLAUDE.md`. Trained model weights (`Models/results/*.pt`) and aggregate
metrics (`*_metrics.csv`) are included, so the architecture and results are
fully inspectable.

## Running

```bash
pip install -r backend/requirements.txt

pytest                              # no data needed
python Models/cnn_rnn_inv3.py       # train (requires Data/)
uvicorn backend.main:app --reload   # backend + dashboard (requires Supabase env)
```

Environment variables: see `.env.example` and `docs/deployment-setup.md`.

## License

MIT — see `LICENSE`.
