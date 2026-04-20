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