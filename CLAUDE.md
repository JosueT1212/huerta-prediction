# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Greenhouse crop yield prediction project ("Huerta Prediction"). Predicts weekly kg production for greenhouses (invernaderos 3 & 4) using internal sensor data and external environmental variables. All models forecast 4 weeks ahead (h=4).

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

## 7. Horizonte de Predicción — Diseño seq_len

`HORIZON = 4` es una constante de documentación en `cnn_rnn_yield.py`. Representa el horizonte objetivo (semanas entre el último dato del sensor en la ventana y la producción target).

`seq_len` es fijo por invernadero en `hp_inv*.yaml` (actualmente = 6). El horizonte efectivo real es `gap - seq_len` y varía ±1 semana entre temporadas según el gap sensor→producción:

| Invernadero | gap típico | seq_len | eff_horizon |
|-------------|-----------|---------|-------------|
| Inv3        | 10–11     | 6       | 4–5 sem     |
| Inv4        | 6–11      | 6       | 0–5 sem     |

Los logs muestran `gap` por temporada y la línea `HORIZON=4 (doc) | seq_len=6 | eff_horizon = gap - seq_len`.

**Por qué NO se usa seq_len variable por temporada:**
Con `seq_len = gap - HORIZON + skip_first_weeks`, inv4 T16 (gap=6, skip=1) da `seq_len=3`, que al hacer padding hasta `max_seq_len=8` rellena 5 de 8 filas con ceros. El modelo veía mayormente ceros en la temporada de validación → R² muy bajo (~0.39). Decisión: `seq_len` fijo, horizonte efectivo varía ligeramente entre temporadas.
