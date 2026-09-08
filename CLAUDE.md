# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Huerta Prediction: weekly tomato yield forecasting (kg/week) for two commercial
greenhouses (invernaderos 3 and 4). A CNN-RNN forecasts 5 weeks ahead from
in-greenhouse sensors and external weather. Served by a FastAPI backend with a
Supabase database and a client dashboard. See `README.md` for results.

## Data (private, not in this repo)

`Data/` holds the grower's Excel files. It is gitignored. Column names and
sheet names are in Spanish and must be preserved.

- `Variables internas Invernadero {3,4}.xlsx` — daily in-greenhouse sensors, one sheet per season
- `Variables exteriores.xlsx` — daily external weather
- `Kg por semana T13 - T17 Invernadero {3,4}.xlsx` — weekly production targets
- `Monitoreo de Fenologia - Temporada {13..17}.xlsx` — phenology monitoring
- `Variables riego invernadero {3,4}.xlsx` — irrigation

Seasons T13–T17. Split: train T13–T15, validate T16, test T17.

## Layout

```
Models/cnn_rnn_yield.py       shared training library: loading, feature engineering,
                              gap normalization, CNNRNN model, grid search, metrics
Models/cnn_rnn_inv{3,4}.py    per-greenhouse grid search entry points
Models/cnn_rnn_inv{3,4}_production.py   refit on all five seasons -> production .pt
Models/cnn_rnn_inv{3,4}_lstm.py         pure-LSTM ablation (num_cnn_blocks=0)
Models/{elasticnet,svr}_*.py  baselines
Models/hp_inv{3,4}.yaml       hyperparameters per greenhouse
Models/results/               .pt weights, *_metrics.csv, logs (npz/pkl/png excluded)
Models/tests/                 unit tests for model helpers

backend/main.py               FastAPI app, routers mounted here
backend/engine.py             load production checkpoint, run inference on a feature window
backend/live_features.py      rebuild training-time features from uploaded rows
backend/inference_trigger.py  decide when a week's uploads are complete and run inference
backend/data_api.py           read Data/ Excel for the historical dashboard
backend/auth.py               Supabase JWT verification
backend/routers/              uploads, predictions, admin, phenology, transplant/harvest dates
backend/tests/                pytest suite

scripts/live_inference.py     walk a live season week by week and store predictions
scripts/mock_ingest.py        replay Excel sensor rows to /ingest/sensors (needs Data/)
supabase/                     SQL migrations
telegraf/telegraf.conf        sensor collector config
demo/                         production dashboard HTML + login page (served by main.py)
Dashboard/demo.html           static UI mockup with placeholder data
docs/superpowers/             design specs and implementation plans
```

## Commands

```bash
pip install -r backend/requirements.txt
pytest                                  # 138 tests, no data or credentials needed
python Models/cnn_rnn_inv3.py           # grid search (needs Data/)
python Models/cnn_rnn_inv3_production.py
uvicorn backend.main:app --reload       # needs .env (see .env.example)
```

Tests that need `Data/` skip when it is absent.

## Model notes

- `HORIZON = 5` in `cnn_rnn_yield.py` is the true forecast horizon: weeks
  between the last sensor row in the window and the target production week.
- `seq_len` is fixed per greenhouse in `hp_inv*.yaml` (inv3 = 4, inv4 = 5).
  Per-season variable `seq_len` was tried and rejected: it requires zero
  padding, which hurt validation R² badly.
- `use_gap_norm: true` enables `_apply_gap_norm_cnn`, which trims rows from the
  start of each season's sensor series so every season hits exactly
  `HORIZON`. `make_sequences_per_season` pairs sensor row j with production
  row j by position, not by calendar week. Because `skip_first_weeks` trims
  production independently, the trim formula must include it:
  `extra_skip = max(0, gap - seq_len - HORIZON + skip_first_weeks + 1)`.
  Omitting that term silently shifts the real horizon. Verify with real
  `week_key` dates, not arithmetic, after touching this code.
- Current config: `skip_first_weeks: 0`, `target_mode: residual`, focal loss.
  The model trains on harvest ramp-up weeks; there is no historical-mean
  fallback in live inference.
- Model selection: 216-run grid search (54 seeds × 4 inits) per greenhouse.
  Production weights are a single refit over T13–T17 at the best config with a
  fixed epoch count, saved to `Models/results/production_cnn_rnn_inv*.pt`.
  Never overwrite `best_cnn_rnn_inv*.pt` (the evaluation checkpoints).
- Pure-LSTM ablation lost on both greenhouses; `num_cnn_blocks=1` stays in
  production. Scripts and results are kept for reproducibility.

## Live inference

`scripts/live_inference.py` advances week by week while real data exists. The
most recent real row in the window must be within 6 days of `as_of_date`, or
the loop stops. Without that guard the loop drifted months past the last upload
on a frozen window.
