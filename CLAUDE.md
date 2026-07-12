# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Greenhouse crop yield prediction project ("Huerta Prediction"). Predicts weekly kg production for greenhouses (invernaderos 3 & 4) using internal sensor data and external environmental variables. CNN-RNN models forecast 5 weeks ahead (h=5, see §7-8 below for the horizon retune history).

## Data

All data lives in `Data/` as Excel (.xlsx) files:
- **Internal variables**: daily sensor readings from inside greenhouses (temperature, humidity, CO2, etc.) — e.g., `Variables internas invernadero 4.xlsx`
- **External variables**: daily weather/environmental conditions — `Variables exteriores.xlsx`
- **Production targets**: weekly kg output per greenhouse — `Kg por semana T13 - T17 Invernadero *.xlsx`

Data spans 5 seasons (T13–T17). Data is in Spanish — column names and labels must be preserved in their original language.

**Data split**: Train (T13–T15) | Validation (T16) | Test (T17)

## Running Code

```bash
# Run a model training script
python Models/cnn_rnn_yield.py
python Models/cnn_rnn_daily.py
python Models/arimax_yield.py
python Models/xgboost_yield.py

# Run EDA notebook
jupyter notebook EDA/eda.ipynb
```

All model scripts are standalone — each loads data, trains, evaluates, and saves results to `Models/results/`.

## Dependencies

No `requirements.txt` exists. Key libraries:
- `pandas`, `numpy`, `matplotlib`, `scikit-learn` (MinMaxScaler, PCA, metrics)
- `torch` (PyTorch) — CNN-RNN models
- `statsmodels` — ARIMAX model
- `xgboost` — XGBoost model
- `openpyxl` — Excel file reading

## Model Architecture

Four model approaches, all predicting weekly kg with the same train/val/test split:

| Model | Script | Resolution | Key Idea |
|-------|--------|-----------|----------|
| CNN-RNN (weekly) | `cnn_rnn_yield.py` | Weekly aggregates | 1D CNN blocks + LSTM, sequence length 7 weeks |
| CNN-RNN (daily) | `cnn_rnn_daily.py` | Raw daily data | Same architecture, 28-day sliding windows, larger network |

### Common pipeline across models
1. Load Excel data by greenhouse ID (each season is a separate sheet)
2. Merge internal + external sensor data on date
3. Aggregate daily → weekly (mean for most features, sum for radiation) — except daily CNN-RNN
4. Feature engineering: lag features (1–4 weeks), rolling statistics
5. MinMax scaling (0–1), PCA (95% variance retained)
6. Train with early stopping on validation loss
7. Evaluate on T17 test set: RMSE, R², NSE, PBIAS, MAPE

### CNN-RNN architecture detail
- **CNN blocks**: 1D convolution + weight normalization + ReLU + dropout + residual connections
- **RNN**: LSTM layers
- **FC**: Feed-forward head for final prediction
- Models saved as `.pt` files in `Models/results/`

## Results

All outputs go to `Models/results/`:
- `*.pt` — PyTorch model weights (per greenhouse)
- `*_metrics.csv` — evaluation metrics per greenhouse
- `*_results.png` — loss curves and actual vs predicted plots

# Reglas de Interacción para Claude Code (Ahorro Estricto de Contexto)

## 1. Prohibido leer archivos completos
NUNCA uses `read_file` para leer un script completo (como `inv3.py` o `inv4.py`) a menos que yo te lo ordene explícitamente. Es un desperdicio de tokens.

## 2. Uso obligatorio de herramientas de búsqueda
Para entender el código, tu flujo de trabajo DEBE ser:
- Usa `grep` o herramientas de búsqueda por palabras clave para encontrar la función o variable específica.
- Una vez localizada la línea, usa la lectura con parámetros `offset` y `limit` para leer SOLO el bloque de código relevante (máximo 50-100 líneas).

## 3. Respuestas concisas
No me des explicaciones largas ni repitas el código que ya escribimos. Dime exactamente qué cambiaste y ejecuta la prueba.

## 4. Contexto del Proyecto Actual
Estamos modificando un modelo CNN-RNN para predicción de rendimiento en invernaderos. Scripts clave: `inv3.py` e `inv4.py` y siempre los logs en inv3.log e inv.log respectivamente.

## 5. Alineación Sensor vs Producción (Invernadero 3)

Los sensores empiezan a registrar ~10–11 semanas antes de que comience la cosecha cada temporada. Con `alignment='positional'`, sensor_row_j y prod_row_j están en el mismo índice pero NO son la misma semana calendario.

| Temporada | Sensor inicia (sem ISO) | Producción inicia (sem) | Brecha |
|-----------|------------------------|------------------------|--------|
| T13       | 26                     | 37                     | +11 sem |
| T14       | 25                     | 35                     | +10 sem |
| T15       | 23                     | 33                     | +10 sem |
| T16       | 21                     | 31                     | +10 sem |
| T17       | 21                     | 32                     | +11 sem |

**Fórmula del horizonte de predicción:**

```
horizon = first_prod_week - seq_len_weeks
```

Con `seq_len=34` y `first_prod_week≈37` (T13): horizon ≈ 6 semanas.

**Cambio clave en `make_sequences_per_season`:** el target usa `y_s[i]` (inicio de la ventana), NO `y_s[i + seq_len - 1]`. Esto aprovecha la brecha natural sensor→producción para predecir hacia adelante sin filtración de datos.

---

## 6. Dashboard — Opciones de Visualización para Cliente

### Pipeline objetivo (sensores en vivo)

```
Sensores físicos → Collector → InfluxDB → Grafana (live viz)
                                   ↑
                     CNN-RNN inference script
                     (escribe predicciones → InfluxDB)
```

### Collector (capa que recibe datos del sensor y escribe a InfluxDB)

| Tipo de sensor | Collector recomendado |
|---|---|
| MQTT broker | Telegraf (plugin MQTT input) |
| HTTP POST desde datalogger | Telegraf HTTP listener o Flask |
| Serial/USB | Telegraf serial plugin o Python custom |
| Excel manual (situación actual) | Script Python con cliente InfluxDB |
| Campbell/Onset datalogger | Telegraf + config output datalogger |

**Telegraf** es el agente oficial de InfluxData — soporta 200+ inputs, cero código para sensores estándar.

### Opciones de stack completo

| Stack | Costo | Pros | Contras |
|---|---|---|---|
| **InfluxDB OSS + Grafana OSS** (self-hosted VPS) | ~$5–10/mes (DigitalOcean/Hetzner) | Control total, sin límites, costo fijo | Necesita server, mantenimiento |
| **InfluxDB Cloud + Grafana Cloud** (managed) | Gratis hasta límites; uso real ~$10–30/mes | Sin infra, escalable | Límites en free tier, datos en nube externa |
| **InfluxDB Cloud Free tier** | Gratis | Fácil inicio | 30 días retención, 5 MB/5 min write |
| **Grafana Cloud Free tier** | Gratis | 10k series, 14 días retención | Límite series/retención |
| **Docker local** (dev/demo) | Gratis | Corre en Mac ahora mismo | Solo local, no accesible cliente |
| **Streamlit (actual)** | Gratis (Community Cloud) | Ya funciona, deploy en minutos | No live data, no alertas |

### Tamaño estimado datos

- Imágenes Docker (InfluxDB + Grafana + Telegraf): ~600 MB
- Datos: 10 sensores @ 5 min → ~50 MB/año
- Histórico actual (T13–T17 Excel): < 10 MB total

### Decisión actual

Live sensors planeados → stack recomendado: **InfluxDB OSS + Grafana OSS en VPS**.
Por ahora (sin sensores live): mantener Streamlit para demo cliente + preparar `docker-compose.yml` local para desarrollo del pipeline.
---

## 7. Horizonte de Predicción — Diseño seq_len (actualizado 2026-07-12)

`HORIZON = 5` en `cnn_rnn_yield.py` es el horizonte objetivo real (semanas entre el
último dato del sensor en la ventana y la producción target).

`seq_len` es fijo **por invernadero** (no varía por temporada — ver razón abajo):

| Invernadero | seq_len | Resultado |
|-------------|---------|-----------|
| Inv3        | 4       | Las 5 temporadas alcanzan horizonte real = 5 exacto |
| Inv4        | 2       | Las 5 temporadas (incluyendo T16, gap=6) alcanzan horizonte real = 5 exacto |

**Por qué NO se usa seq_len variable por temporada:** con `seq_len` distinto por
temporada, las ventanas deben rellenarse con ceros (padding) hasta `max_seq_len`
para compartir forma en un batch. Un experimento previo con `seq_len = gap - HORIZON
+ skip_first_weeks` (HORIZON=4 entonces) dio `seq_len=3` en inv4 T16, con 5 de 8
filas de padding en la temporada de validación → R² muy bajo (~0.39). Se mantiene
`seq_len` fijo por invernadero; el ajuste por temporada lo hace únicamente
`_apply_gap_norm_cnn` recortando filas del sensor, sin padding.

### `use_gap_norm` y el bug de `skip_first_weeks` (crítico — leer antes de tocar `_apply_gap_norm_cnn`)

`use_gap_norm: true` en ambos `hp_inv*.yaml` activa `_apply_gap_norm_cnn`, que recorta
filas del INICIO de la serie de sensores por temporada para alinear el horizonte real.

`make_sequences_per_season` empareja fila `j` del sensor (ya recortado) con fila `j`
de producción (ya filtrada por `skip_first_weeks`, que recorta las primeras
`skip_first_weeks` semanas de cosecha) **por posición**, no por semana calendario.
Como ambos recortes son independientes, la fórmula de `extra_skip` DEBE incluir
`skip_first_weeks`, o el horizonte real se desincroniza silenciosamente:

```
extra_skip = max(0, gap - seq_len - HORIZON + skip_first_weeks + 1)
```

Se verificó empíricamente (con fechas reales `week_key`, no solo aritmética) que
omitir el término `skip_first_weeks + 1` produce un horizonte real de
`HORIZON + skip_first_weeks + 1` (9 semanas con `skip_first_weeks=3`, `HORIZON=5`)
en **todas** las temporadas de ambos invernaderos — no solo un caso aislado. Este bug
existía desde antes de este retune (skip_first_weeks=3 ya estaba en los yaml), pero
nunca se manifestó porque `use_gap_norm` estaba en `false` hasta ahora.

Este mismo bug también explicaba el "límite estructural" de T16 en inv4 (antes
documentado como horizonte máximo=4, 1 semana corta) — era un artefacto de la
fórmula rota, no una limitación real de datos: con la fórmula corregida, T16
también alcanza horizonte=5 exacto.

## 8. Plan final de entrenamiento de modelos de producción (2026-07-12)

Ver diseño completo en `docs/superpowers/specs/2026-07-11-inv3-inv4-horizon5-production-design.md`
y plan de implementación en `docs/superpowers/plans/2026-07-11-inv3-inv4-horizon5-production.md`.

Resumen del pipeline aplicado a **ambos** invernaderos:
1. `HORIZON=4→5` en `cnn_rnn_yield.py` (constante compartida).
2. `use_gap_norm: true` + `seq_len` fijo por invernadero (inv3=4, inv4=2) en cada `hp_inv*.yaml`.
3. Fix de `_apply_gap_norm_cnn` para incluir `skip_first_weeks` (ver §7 arriba).
4. Grid search completo (216 runs: 54 seeds × 4 inits) por invernadero, train=T13-15,
   val=T16, test=T17 — igual que antes, solo bajo la config corregida.
5. Refit de producción: se re-entrena UNA sola vez por invernadero con el mejor
   seed/init encontrado en el paso 4, agrupando las 5 temporadas (T13-T17) como
   entrenamiento (sin holdout), número de épocas fijo (= época de early-stop del
   mejor run del paso 4, sin early stopping real ya que no hay set de validación).
   Pesos finales → `Models/results/production_cnn_rnn_inv3.pt` / `_inv4.pt`
   (nunca sobrescriben los checkpoints de evaluación `best_cnn_rnn_inv*.pt`).

### Nota para el pipeline de inferencia en vivo (importante)

Las primeras `skip_first_weeks=3` semanas de cada temporada NO tienen predicción
del modelo (se descartan del entrenamiento/evaluación por ser semanas de rampa de
cosecha, poco confiables). **En inferencia real, esas primeras 3 semanas de cada
temporada nueva deben inferirse usando la media histórica** (media de esas mismas
semanas en temporadas pasadas), no con el modelo CNN-RNN — el modelo nunca fue
entrenado para predecir esas semanas y no debe usarse ahí. Esto debe implementarse
en el pipeline de inferencia en vivo (ver spec `docs/superpowers/specs/2026-06-23-live-inference-pipeline-design.md`),
no en este repo de entrenamiento.
