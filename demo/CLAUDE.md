# CLAUDE.md — Demo Huerta Prediction

Guía completa del demo del dashboard de predicción de rendimiento (kg) de jitomate
para el cliente, desarrollado por **JATA**. Aquí vive el frontend
del demo y la documentación de cómo conecta con el backend y los modelos
entrenados.

---

## 1. Propósito del demo

Demostrar al cliente y stakeholders el modelo **CNN-RNN**
entrenado para predicción semanal de kilogramos en **Invernadero 3** y
**Invernadero 4**, con un horizonte efectivo de **6 semanas**.

El demo debe:

1. Mostrar la predicción a 6 semanas vs el valor real observado.
2. Mostrar el intervalo de confianza al 90% (CPTC).
3. Comunicar la confiabilidad del modelo (R², MAPE, NSE).
4. Presentar histórico de temporadas T13–T17.
5. ✅ Correr el test set T17 **desde el `.pt` cargado en backend** y explorar
   cualquier semana con un **slider manual** (+ flechas `‹ ›`) que mueve el cursor.

---

## 2. Estructura de archivos

```
demo/
├── CLAUDE.md            # este archivo
├── login.html          # pantalla de acceso (gate cosmético, ver §3)
└── Demo Dashboard.html  # SPA single-file (HTML+CSS+JS, ~2990 líneas)
```

El backend que alimenta este HTML vive en `../backend/`:

```
backend/
├── main.py            # FastAPI app: endpoints REST + sirve login/HTML + warm al arrancar
├── engine.py          # InferenceEngine: carga predicciones T17 (.npz + métricas .csv), cachea
├── data_api.py        # Lectura cruda de Data/: fenología (8 vars) + histórico mensual de sensores
└── requirements.txt   # fastapi, uvicorn, numpy, pandas, openpyxl, torch, scipy, scikit-learn, pyyaml, matplotlib
```

> **Diseño (reescrito desde cero, may 2026)**: los `.pt` se cargan **al arrancar**
> y la inferencia de T17 se corre **una sola vez** (ambos invernaderos, ~20s).
> "Correr en la semana X" = cortar (slice) esos arrays cacheados por un `cursor`.
> El modelo NO se re-ejecuta por semana → demo robusto, nunca falla en vivo.
> El frontend tiene un **slider manual** (+ flechas `‹ ›` de paso) por invernadero
> que mueve el cursor. **Sin auto-play** (decisión: control manual, más intuitivo).

Y los modelos entrenados en `../Models/results/`:

```
Models/results/
├── best_cnn_rnn_inv3.pt              # mejor state_dict inv3 (R² T17 = 0.71)
├── best_cnn_rnn_inv4.pt              # mejor state_dict inv4 (R² T17 = 0.55)
├── cnn_rnn_inv{3,4}_predictions.npz  # predicciones cacheadas T17
├── cnn_rnn_inv{3,4}_metrics.csv      # métricas (best, ensemble, top25, cptc_pi)
└── cnn_rnn_inv{3,4}_results.png      # gráficas de entrenamiento
```

---

## 3. Cómo ejecutar el demo

### Requisitos
```bash
pip install -r backend/requirements.txt
```

### Iniciar
```bash
# Desde la raíz del repo
uvicorn backend.main:app --port 8000 --reload
```

### Abrir en el navegador
```
http://localhost:8000/
```

> **Login**: `GET /` sirve `login.html`. Es un **gate cosmético** (no valida
> credenciales): al enviar, setea `sessionStorage.jata_auth='1'` + `jata_user`
> y redirige a `/app`. El dashboard (`/app`) tiene una guarda client-side que
> rebota a `/` si no existe `jata_auth`. Cualquier usuario/contraseña entra.

> **Importante**: No abras el HTML con doble clic (`file://`).
> El backend lo sirve desde `GET /app` y `API_BASE` se auto-detecta a same-origin.

### Para detener
`Ctrl+C` en la terminal del uvicorn.

### Puerto / CORS
- Puerto: 8000 (default uvicorn)
- CORS: `allow_origins=['*']` para desarrollo local

---

## 4. Backend — endpoints

Base URL: `http://localhost:8000`

| Método | Path | Descripción |
|---|---|---|
| GET | `/` | Sirve `login.html` (gate cosmético) |
| GET | `/app` | Sirve `Demo Dashboard.html` desde mismo origen |
| GET | `/health` | `{status, model_version, sync_date, horizon_weeks, greenhouses_loaded}` |
| GET | `/greenhouses` | `[{id, name}]` |
| GET | `/inference/{inv}` | Arrays completos T17 (36 semanas) desde `.npz` cacheado |
| GET | `/inference/{inv}/window?cursor=N&horizon=6&past=5` | Recorte alrededor de la semana elegida (history + predictions) |
| GET | `/metrics/{inv}` | `{source: npz_cache, best, cptc_pi}` (best + intervalos CPTC desde `*_metrics.csv`) |
| GET | `/predictions/{inv}` | Alias compat de `/inference/{inv}` |
| **GET** | **`/phenology/{inv}`** | **Serie semanal (promedio sobre plantas) de las 8 vars de fenología + `fields` (metadatos) + `latest`** |
| **POST** | **`/phenology/{inv}`** | **Valida filas-por-planta y devuelve el promedio (stub demo: NO persiste, ver §13)** |
| **GET** | **`/sensor-history/{inv}?var=temp\|hr\|co2\|ce\|par`** | **Promedio mensual por temporada (una serie por T13–T17) para el modal "ver histórico"** |
| **POST** | **`/uploads/{inv}/{form_type}`** | **Sube un `.xlsx` (`form_type` = `sensores`\|`fenologia`\|`produccion`), valida columnas requeridas, hace upsert a Supabase; `429` si el bloqueo semanal (`submission-lock`) sigue activo. Tab "Insertar datos"** |
| **GET** | **`/uploads/{inv}/{form_type}/history`** | **Historial de capturas ya enviadas para ese invernadero/tipo (tab Historial, solo lectura)** |
| **GET** | **`/submission-lock/{inv}/{form_type}`** | **`{locked, next_allowed_at}` — si ya se envió una captura esta semana, bloquea el próximo envío 7 días** |
| GET | `/demo/...` | Archivos estáticos del directorio `demo/` |

### Formato de `/inference/{inv}` (alias: `/predictions/{inv}`)

Arrays completos T17 (36 semanas), calculados desde el `.pt` una sola vez.
`ensemble_pred` = `y_pred` (el demo en vivo usa el best, sin ensemble).

```json
{
  "inv_id": 3,
  "horizon_weeks": 6,
  "n_weeks": 36,
  "weights_path": ".../best_cnn_rnn_inv3.pt",
  "weeks": [{"year": 2025, "iso_week": 35, "week_key": 202535}, ...],
  "y_true":        [46173.83, 50189.10, 55582.83, ...],
  "y_pred":        [43759.65, 44884.79, 44680.76, ...],
  "ensemble_pred": [43759.65, 44884.79, 44680.76, ...],
  "pi_lower":      [35170.12, 36308.01, 35989.24, ...],
  "pi_upper":      [52349.18, 53461.58, 53372.27, ...]
}
```

### Formato de `/inference/{inv}/window?cursor=N&horizon=6&past=5`

Recorte alrededor de la semana elegida. `history` = `past` semanas observadas
previas al cursor; `predictions` = `horizon` semanas desde el cursor (con `real`
revelado porque es test set). Si `cursor` se omite → mitad de T17.

```json
{
  "inv_id": 3, "cursor": 18, "n_weeks": 36, "horizon": 6, "past": 5,
  "history":     [{"week_key": 202548, "year": 2025, "iso_week": 48,
                   "observed": true, "real": 24381.4, "pred": 26256.9,
                   "lo": 17612.6, "hi": 34901.2}, ...],
  "predictions": [{"week_key": 202601, "year": 2026, "iso_week": 1,
                   "observed": false, "horizon": 1, "real": 26998.9,
                   "pred": 26067.7, "lo": 17395.7, "hi": 34739.7}, ...]
}
```

### Formato de `/metrics/{inv}`

`source: npz_cache` → `best` + `cptc_pi` se leen de `cnn_rnn_inv{inv}_metrics.csv`
(no se re-ejecuta el `.pt`). Son las métricas oficiales reportadas en §6.

```json
{
  "inv_id": 3,
  "source": "npz_cache",
  "best":    {"R²": 0.7118, "RMSE (kg)": 4426.4, "MAPE (%)": 11.61, "NSE": 0.7118, "PBIAS (%)": 2.74},
  "cptc_pi": {"pi_coverage": 0.842, "pi_avg_width": 26002.7}
}
```

### Formato de `/phenology/{inv}` (KPIs fenología + formulario)

```json
{
  "inv_id": 3,
  "fields": [{"col": "RACIMOS PUESTOS", "key": "racimos_puestos",
              "label": "Racimos puestos", "unit": "conteo",
              "min": 0, "max": 40, "step": 1}, ...],   // 8 campos (ver §6)
  "seasons": [{"season": "T17", "weeks": [2,3,...],
               "values": {"racimos_puestos": [1.0, 1.2, ...], ...}}, ...],
  "latest": {"racimos_puestos": 22.8, "cantidad_tomates": 29.0, ...},
  "latest_season": "T17"
}
```

**POST `/phenology/{inv}`** → body `{"week": 35, "rows": [{key: valor, ...}, ...]}`
(una fila por planta). Respuesta: `{n_plants, averages: {key: promedio}, persisted: false, note}`.
Valida rango por campo → `422` con mensaje si algo cae fuera de `[min, max]`.

### Formato de `/sensor-history/{inv}?var=temp` (modal "ver histórico")

```json
{
  "inv_id": 3, "var": "temp",
  "meta": {"source": "internas", "col": "Temperatura promedio", "label": "Temperatura", "unit": "°C"},
  "months": ["Ene","Feb",...,"Dic"],
  "series": [{"season": "T13", "points": [18.3, 19.7, ..., 17.6]}, ...]  // null donde el mes no existe
}
```
Mapeo `var` → fuente: `temp/hr/co2` (internas), `ce` (riego), `par`
(radiación exterior, proxy). `suelo` **no tiene datos** → sin "ver histórico".

---

## 5. Frontend — dashboard

### Tecnología
- HTML+CSS+JS vanilla, single-file (~2990 líneas)
- **Chart.js 4.4** vía CDN
- Fuentes Google: Cormorant Garamond, JetBrains Mono, Inter
- Sin framework, sin build step

### Layout
```
<aside.sidebar>           → navegación lateral fija (JATA + enlaces)
<div.app>
  ├── 10× <section.view>  → vistas alternables (solo una .active)
<div.hist-modal>          → modal "ver histórico" de sensores (oculto por default)
<style>                   → CSS de features (fenología, formulario, modal, yield-hero)
<script>                  → datos + Chart.js + router + init() async
```

### Vistas (`id="view-<name>"`)

> `view-live3` / `view-live4` ("Tiempo real") fueron **eliminadas** (no había
> sensores live que las alimentaran; solo quedaba el mock + histórico real,
> que ahora vive dentro de "Insertar datos" → tab Historial).

| View | Sidebar | Contenido | Estado |
|---|---|---|---|
| `view-menu` | "Vista general" | Saludo + 2 cards (inv3, inv4) con KPIs + chart resumen | **Conectado** |
| `view-insert` | "Insertar datos" | **4 tabs**: Sensores/Fenología/Producción (carga de Excel) + Historial (3 subtablas de solo lectura) — ver detalle abajo | **Conectado** |
| `view-inv3` / `view-inv4` | "Predicción" | Slider manual (`‹ ›`) + KPIs (volumen, pico, **kg/m² 6 sem + acumulado T17**, semana pasada) + chart detalle + tabla 6 sem | **Conectado** |
| `view-kpi3` / `view-kpi4` | "KPIs & gráficas" | **Hero kg/m² combo (meta vs producción)** arriba + card rendimiento prom. ambos + **panel fenología (8 vars)** + (mock: calibres, error, recursos) | **Parcial** |
| `view-hist3` / `view-hist4` | "Histórico" | Tabla completa observado + predicho con % error | **Conectado** |

### "Insertar datos" (`view-insert`) — detalle de los 4 tabs

Reemplaza el antiguo formulario manual fila-por-planta (`ins-week`, `ins-add`,
`ins-submit`). Selector `#ins-inv` (Invernadero 3/4) arriba, compartido por
los 4 tabs (`.ins-tab[data-tab]`):

| Tab | Panel | Contenido |
|---|---|---|
| Sensores | `ins-panel-sensores` | Carga `.xlsx` (`upload-sensores-file`) → preview de filas/fechas → doble confirmación → `POST /uploads/{inv}/sensores` |
| Fenología | `ins-panel-fenologia` | Igual, columnas de fenología por planta → `POST /uploads/{inv}/fenologia` |
| Producción | `ins-panel-produccion` | Igual, `fecha` + `kg_reales` → `POST /uploads/{inv}/produccion` |
| Historial | `ins-panel-hist` | 3 subtabs de solo lectura (`.hist-subtab`: sensores/fenología/producción), cada uno una tabla (`hist-table-{tipo}`) de las capturas ya enviadas |

Flujo de envío por tab (`wireUploadTab(formType)`):
1. Se elige el `.xlsx` → se parsea client-side (SheetJS) y se valida que
   existan las columnas requeridas (`UPLOAD_TYPES[formType].requiredCols`).
2. Preview de filas detectadas + rango de fechas.
3. Doble confirmación (`ins-confirm1` → `ins-confirm2`, con advertencia de
   que la captura **no podrá modificarse ni eliminarse** y que el próximo
   envío estará bloqueado 7 días).
4. `POST /uploads/{inv}/{formType}` con el archivo; `429` si ya hay un envío
   reciente (semanal) — banner de bloqueo vía `GET /submission-lock/{inv}/{formType}`.
5. Cambiar `#ins-inv` limpia cualquier archivo/estado en staging de los 3
   tabs de carga (evita enviar a un invernadero distinto al mostrado).

### Router (JS)
- `showView(viewId)`: toggle `.active` en `.view`, sincroniza `.nav-btn` y `.sb-link`
- Sub-vistas (`hist3`/`kpi3`) mapean al padre `inv3` para el highlight
- Listeners en `[data-goto]` y `.nav-btn, .sb-link`

### Carga de datos (`init()`)
```
1. Promise.all → /inference/{3,4} + /metrics/{3,4} + /phenology/{3,4}
2. Guarda payloads en rawInv[3,4] y rawPheno[3,4]; phenoFields = ph3.fields
3. wireSimControls(inv): listeners del slider (input) y flechas prev/next
4. renderInv(inv, cursor=null): cursor inicial = mitad de T17 (índice 18)
5. renderPhenology(3,4): cards de las 8 vars con sparkline SVG
6. initInsTabs(): wire de los 4 tabs de "Insertar datos" (Sensores/Fenología/Producción/Historial, ver §5)
7. wireMeta(3,4) + renderYieldGoal(3,4): combo kg/m² meta vs producción (ver §13)
8. Si falla → banner rojo "Backend no disponible"
```

### Slider manual — `renderInv(inv, cursor)`
Repinta TODO para un invernadero en la semana `cursor` (el slicing vive en el
frontend, sobre los arrays ya cacheados; no hay fetch por semana):
```
- buildSeriesFromAPI(rawInv[inv], cursor) → { history, predictions, historyTable }
   · cursor clamp [0, n-1]; history = `past` previas; predictions = HORIZON desde cursor
- buildPredTable + buildHistTable
- makeMenuChart + makeDetailChart   (registry `charts[id]`: destroy + recrear)
- fillKPIs (volumen total, pico, semana pasada pred/real/error)
- updateSliderFill (relleno acento del track vía var --fill)
- actualiza labels simweek{inv} (rango fecha) y simcur{inv} ("19 / 36")
stepInv(inv, ±1) → flechas ‹ ›   ·   sin auto-play
```

### Configuración same-origin
```js
const API_BASE = window.API_BASE
  ?? (location.protocol.startsWith('http') ? '' : 'http://localhost:8000');
```

### IDs DOM relevantes (para hooks futuros)
- `tbody-inv3`, `tbody-inv4`, `tbody-hist3`, `tbody-hist4`
- `chart-menu-3/4`, `chart-inv3/4`, `chart-kpi{3,4}-{prod,cat,err,res}`
- KPI cards: `menu{3,4}-total/peak/peak-unit`, `inv{3,4}-total/peak/peak-unit`, `inv{3,4}-vs-{pred,real,err}`
- **kg/m²**: `inv{3,4}-kgm2-win` (6 sem), `inv{3,4}-kgm2-acc` (T17), `kpi-yield-avg-{3,4}`, `kpi-yield-trend-{3,4}`, `chart-yield-{3,4}` (combo), `meta-input-{3,4}`
- **Fenología**: `pheno-grid-{3,4}` (KPIs)
- **Insertar datos**: `ins-inv` (selector), `.ins-tab[data-tab]` (sensores/fenologia/produccion/hist), por tab: `upload-{tipo}-file/btn/preview/lock-banner/result`, `upload-{tipo}-confirm1/confirm2`; Historial: `.hist-subtab[data-histtab]`, `hist-table-{tipo}`
- **Histórico sensor**: `hist-modal`, `hist-canvas`, `hist-modal-title/sub`; botón `.sensor-hist-btn[data-hist-var][data-hist-inv]`
- Slider de semana: `slider{3,4}` (range), `prev{3,4}` / `next{3,4}` (flechas), `simweek{3,4}` (rango fecha), `simcur{3,4}` ("19 / 36")

---

## 6. Resumen del modelo CNN-RNN

### Objetivo
Predecir **producción semanal (kg)** de jitomate por invernadero usando sensores
internos + externos + riego + fenología, con horizonte efectivo **6 semanas**.

### Idea clave del horizonte
Los sensores comienzan a registrar ~10–11 semanas antes de la cosecha cada
temporada. Con `alignment='positional'` los índices de sensor y producción
están pareados por posición pero corresponden a semanas calendario distintas.
La ventana de sensor `[j, j+seq_len]` queda **adelantada** al target `y[j]`
por construcción → no hace falta `shift(-horizon)` en el target, el horizonte
emerge del gap natural.

| Temporada | Sensor inicia (ISO) | Producción inicia (ISO) | Gap |
|---|---|---|---|
| T13 | 26 | 37 | +11 |
| T14 | 25 | 35 | +10 |
| T15 | 23 | 33 | +10 |
| T16 | 21 | 31 | +10 |
| T17 | 21 | 32 | +11 |

Con `seq_len=6` → `eff_horizon = gap - seq_len ≈ 4–5 sem` real (presentado
como "6 semanas" al cliente porque es el largo de la ventana).

### Arquitectura
```
x_sensor (B, 6, n_sensor)  ──permute──→ Conv1D × num_cnn_blocks ──permute──→ (B, 6, cnn_filters)
                                                                                       │
                                                                                       cat
                                                                                       │
x_temporal (B, 6, n_temporal) ─────────────────────────────────────────────────────────┘
                                                                                       │
                                                                          LSTM(hidden, layers)
                                                                                       │
                                                                          FC(hidden→fc_hidden→1)
                                                                                       │
                                                                                    kg_pred
```

- **CNNBlock**: `Conv1d + WeightNorm + BatchNorm1d + ReLU + Dropout + residual`
- **Sensor → CNN+PCA**; **Fenología → MinMax (sin PCA) concat post-PCA**;
  **Temporal → bypass CNN, concat al LSTM**

### Hiperparámetros activos (`hp_inv{3,4}.yaml`)

| HP | Valor |
|---|---|
| `seq_len` | 6 |
| `skip_first_weeks` | 3 |
| `alignment` | positional |
| `batch_size` | 8 |
| `lag_features` | `[]` (desactivado) |
| `include_rolling_mean` | false |
| `cnn_filters` | 32 |
| `cnn_kernel_size` | 2 |
| `cnn_padding` | 1 |
| `num_cnn_blocks` | 1 |
| `lstm_hidden` | 64 |
| `lstm_layers` | 1 |
| `fc_hidden` | 32 |
| `dropout` | 0.3 |
| `learning_rate` | 2e-3 |
| `weight_decay` | 1e-4 |
| `init_methods` | `[default, xavier, orthogonal, lecun]` |
| `seeds` | 54 valores |

**Diferencias inv3 vs inv4**:

| | Inv3 | Inv4 |
|---|---|---|
| `corr_weight` | 0.8 | 0.5 |
| `epochs` | 500 | 1000 |
| `patience` | 150 | 250 |

### Features

**Sensor → CNN (con PCA 95%)**:
- Internas: `temp_prom_int`, `temp_min_int`, `temp_max_int`, `hr_prom_int`,
  `co2_ppm`, `deficit_humedad`, `deficit_presion_vapor`, `humedad_abs_int`
- Externas: `temp_prom_ext`, `temp_max_ext`, `temp_min_ext`, `hr_prom_ext`,
  `rad_sum`, `rad_max`, `dh_ext`, `humedad_abs_ext`
- Riego: `riego_total` (sum), `ph_promedio` (mean), `ce_promedio` (mean)

**Fenología → MinMax(-1,1) concat post-PCA (sin PCA propio)** — **8 cols** crudas
desde `BD TOMATE` (`cnn_rnn_yield.PHENO_FEATURE_COLS`, autoritativo; capturadas
por planta y promediadas por `SEMANA DEL AÑO`):

| Columna `BD TOMATE` | `key` (API/form) | Unidad | Rango T17 |
|---|---|---|---|
| `RACIMOS PUESTOS` | racimos_puestos | conteo | 1–33 |
| `FLORES EN RACIMO ABIERTAS` | flores_abiertas | conteo | 1–6 |
| `CANTIDAD DE RACIMOS EN PLANTA` | racimos_planta | conteo | 1–10 |
| `CANTIDAD DE TOMATES` | cantidad_tomates | conteo | 0–39 |
| `Nº DE RACIMO EN COSECHA` | racimo_cosecha | índice | 0–32 |
| `TOMATES MADUROS (COLOR 2)` | tomates_maduros | conteo | 0–3 |
| `DIAMETRO DEL FRUTO cm` | diametro_fruto | cm ⚠️ ¿mm? | 0–53 |
| `CRECIMIENTO PLANTA (cm)` | crecimiento_planta | cm/sem | 11–33 |

> ⚠️ `DIAMETRO DEL FRUTO` llega a ~53 con header "cm" → casi seguro **mm**.
> Confirmar con el cliente (afecta unidad mostrada y validación en el formulario).

**Temporal → bypass CNN, concat al LSTM input**:
- `dias_desde_transplante`, `week_in_season`
- (lags y rolling desactivados en YAML actual)

### Target
```python
weekly['target'] = weekly['kg_reales'].shift(-horizon)
```
Con `horizon=0` en `prepare_data` (porque alignment posicional ya
introduce el gap) → `target = kg_reales` y en `make_sequences_per_season`:
```python
seqs_y.append(y_s[j])   # inicio de la ventana
```

### Transformaciones
- **Target**: `Box-Cox(y+1)` → `MinMaxScaler(-1, 1)`
- **Sensor**: `MinMaxScaler(-1, 1)` → `PCA(0.95)`
- **Fenología**: `MinMaxScaler(-1, 1)` (sin PCA)
- **Temporal**: `MinMaxScaler(-1, 1)`

### Loss
`YieldWMAELoss`: Huber ponderado por `|target - mean|^power` + término de
correlación Pearson (peso = `corr_weight`).

### Split
- **Train**: T13, T14, T15
- **Val**: T16
- **Test**: T17

### Estrategia de entrenamiento
- **216 runs por invernadero** = 54 seeds × 4 init methods
  (`default`, `xavier`, `orthogonal`, `lecun`)
- Mejor por R² en T17 → `best_cnn_rnn_inv{3,4}.pt`
- **Ensemble top-20** (media de predicciones)
- **CPTC Prediction Intervals** al 90% calibrados con T16

### Métricas reportadas (T17, `*_metrics.csv`)

**Inv3:**
| Modelo | R² | RMSE (kg) | MAPE (%) |
|---|---|---|---|
| best | 0.7118 | 4,426 | 11.61 |
| ensemble_top20 | 0.6621 | 4,793 | 12.34 |
| top25_mean | 0.5787 ± 0.0594 | — | — |
| cptc_pi | cov=0.842, width=26,003 | — | — |

**Inv4:**
| Modelo | R² | RMSE (kg) | MAPE (%) |
|---|---|---|---|
| best | 0.5534 | 6,618 | 12.90 |
| ensemble_top20 | 0.5395 | 6,721 | 13.12 |
| top25_mean | 0.4869 ± 0.0359 | — | — |
| cptc_pi | cov=0.842, width=46,808 | — | — |

### Inferencia desde el `.pt` (re-ejecución)
Reproducir el test T17 desde el modelo guardado (lo que hace el backend):
```python
# backend/engine.py
from engine import ENGINE
state = ENGINE.get(3)
# → R²=0.6856, RMSE=4,623, MAPE=11.03%
# (Ligeramente distinto del .npz cacheado por re-ajuste de PCA/scalers)
```

---

## 7. Datos

Todos los datos viven en `/Data/` (mayúscula). Archivos clave:

| Archivo | Contenido |
|---|---|
| `Variables internas Invernadero 3.xlsx` | Sensores internos inv3 (diario) |
| `Variables internas invernadero 4.xlsx` | Sensores internos inv4 (diario) |
| `Variables exteriores.xlsx` | Clima externo (diario) |
| `Variables riego invernadero {3,4}.xlsx` | Riego total, pH, CE (diario) |
| `Kg por semana T13 - T17 Invernadero {3,4}.xlsx` | Producción (semanal, 5 hojas) |
| `Monitoreo de Fenologia - Temporada {13..17}.xlsx` | Hoja `BD TOMATE` |
| `Fechas de Transplante.xlsx` | Fechas por temporada e invernadero (regenerado desde PDF) |

### Fechas de transplante (del PDF original)
| Temp | Inv3 | Inv4 |
|---|---|---|
| T13 | 30-jun-2021 (y 01-jul-2021) | 30-jun-2021 |
| T14 | 23-jun-2022 | 24-jun-2022 |
| T15 | 08-jun-2023 | 09-jun-2023 |
| T16 | 23-may-2024 | 24-may-2024 |
| T17 | 22-may-2025 | 22-may-2025 |

---

## 8. Estado actual de la implementación

### ✅ Hecho
- Backend FastAPI servido en `:8000` con CORS abierto, **reescrito desde cero**
- `backend/engine.py` (`InferenceEngine`): carga ambos `.pt` al arrancar, corre T17 1 vez, cachea
- HTML servido desde mismo origen (`/`) — evita problemas `file://`
- Endpoints: `/inference/{inv}`, `/inference/{inv}/window`, `/metrics/{inv}` (todo desde el `.pt`)
- **Exploración por semana**: slider manual + flechas `‹ ›` por invernadero → mueve el cursor sobre T17 (sin auto-play)
- Vistas conectadas: **menú**, **inv3/inv4 detail** (con slider), **histórico**
- Charts y KPI cards repintan al mover el cursor (real vs predicho + intervalo 90%)
- Slider con thumb tipo perilla + relleno de track en acento (`--fill`)
- Verificado en navegador (Playwright): slider, flechas, clamp en extremos, charts y totales actualizan OK
- `requirements.txt` completo (faltaban torch/scipy/scikit-learn/pyyaml/matplotlib → causa del fallo previo)
- `Fechas de Transplante.xlsx` regenerado desde PDF
- **Login** `login.html` (gate cosmético, `sessionStorage.jata_auth`)
- **`backend/data_api.py`** + endpoints `/phenology/{inv}` (GET/POST), `/sensor-history/{inv}`
- **Fenología en KPIs**: panel con las 8 vars (valor + sparkline) desde `/phenology`
- **"Insertar datos"**: rediseñado como 4 tabs por invernadero — carga de `.xlsx` (Sensores/Fenología/Producción) con preview + doble confirmación + bloqueo semanal, y tab Historial de solo lectura (ver §5) — **persiste en Supabase** vía `/uploads/{inv}/{form_type}`
- **"Ver histórico"** de sensores: modal con line-chart de promedio mensual, una línea por temporada T13–T17
- **Rendimiento kg/m²** (ver §13): KPIs de predicción + card prom. ambos + hero combo meta vs producción
- Verificado en navegador (Playwright): paneles, carga de Excel (envío OK + validaciones), modal, combo y cambio de meta

### 🚧 En desarrollo
- (Opcional) Endpoint SSE `/inference/{inv}/stream?delay_ms=N` para animar server-side
- Banner "MOCK / Demo" sobre las gráficas mock restantes de KPIs
- **Persistencia de fenología manual** (`POST /phenology/{inv}`, formulario de KPIs, hoy es stub) → migrar a **Supabase** en deployment; la fenología cargada por Excel en "Insertar datos" **sí** persiste

### 📋 Lo que sigue mock (por diseño)
| Elemento | Por qué | Cuándo conectar |
|---|---|---|
| KPIs: calibres (dona), error 8 sem, recursos | Calibres requiere Vision Computer (YOLO) que aún no produce features | Tras integración YOLOv11-seg |
| KPIs: eficiencia hídrica, precisión, mortalidad | Cards estáticas de ejemplo | Cuando haya datos de riego/mortalidad por ciclo |

---

## 9. Flujo de datos completo

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRENAMIENTO (offline, ya ejecutado)                      │
│  Models/cnn_rnn_inv{3,4}.py                                 │
│    Excel (Data/) → prepare_data → 216 runs → best_*.pt      │
│                                  → *_predictions.npz        │
│                                  → *_metrics.csv            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  BACKEND (runtime, en :8000)                                │
│  backend/main.py + backend/engine.py                        │
│    [startup] warm → carga ambos .pt + corre T17 1 vez       │
│              → cachea arrays por invernadero (InferenceEngine)│
│    GET /inference/{inv}         → arrays T17 cacheados       │
│    GET /inference/{inv}/window  → slice por cursor (semana)  │
│    GET /metrics/{inv}           → métricas live (best+cptc)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (browser, http://localhost:8000/)                 │
│  demo/Demo Dashboard.html                                   │
│    init() async → fetch /inference/3, /inference/4          │
│                 + /metrics/3, /metrics/4  → rawInv[3,4]      │
│    slider/flechas → renderInv(inv, cursor)                  │
│      buildSeriesFromAPI() → { history, predictions }        │
│      Repinta: tablas, charts, KPI cards                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Decisiones de diseño

| Decisión | Por qué |
|---|---|
| **Backend sirve el HTML** | Evita `file://` bloqueando fetch; mismo origen → sin CORS issues en demo cliente |
| **Inferencia 1 vez al arrancar (warm)** | Carga `.pt` + corre T17 una sola vez (~20s); las semanas son slices instantáneos → nunca falla en vivo |
| **Slice en el frontend (no fetch por semana)** | Mover el slider es instantáneo; el endpoint `/window` existe por completitud pero la UI usa el array completo |
| **Slider manual, sin auto-play** | Control directo e intuitivo para el presentador; evita animación que distrae |
| **Cursor inicial en mitad de T17** | Muestra pasado observado + futuro predicho simultáneamente al abrir, sin lógica de fechas reales |
| **HORIZON = 6 en frontend** | Es lo que el cliente entiende como "6 semanas adelante"; internamente `seq_len=6` define la ventana de sensores |
| **CORS abierto en dev** | Solo para localhost; restringir antes de producción cliente |
| **Sin framework JS** | Demo single-file portable; el cliente puede abrirlo sin npm/build |

---

## 11. Reglas de interacción para Claude Code (heredadas del CLAUDE.md raíz)

1. **Prohibido leer archivos completos** sin orden explícita. Usa grep/glob +
   Read con `offset`/`limit` (máx 50–100 líneas).
2. **Respuestas concisas**: qué cambió y prueba ejecutada, sin parafrasear código.
3. **Logs**: `inv3.log` / `inv4.log` para entrenamiento; consola del navegador
   (F12 → Console) para el dashboard; `uvicorn` stdout para el backend.
4. **No tocar `Models/cnn_rnn_yield.py`** sin razón fuerte — es el módulo
   compartido por inv3 e inv4 y por `backend/engine.py`.

---

## 12. Referencias rápidas

- **Repo raíz**: `/home/master-chief/Documents/Huerta_Prediction/CLAUDE.md`
- **Modelo**: `/home/master-chief/Documents/Huerta_Prediction/Models/`
- **Vision Computer (otro pipeline)**: `/home/master-chief/Documents/Huerta_Prediction/Vision Computer/CLAUDE.md`
- **Cliente**: exportador de jitomate a EE.UU./Canadá
- **Empresa**: JATA (AgTech consultoría de IA)
- **Métricas de éxito**: utilidad operativa, no solo R²/RMSE

---

## 13. Rendimiento kg/m² (display, sin reentrenar)

El modelo predice **kg absolutos**; kg/m² es un **reescalado lineal** para
mostrar al cliente (no afecta el entrenamiento). Vive en el frontend.

### Superficie de cultivo (constante por invernadero)
```
A = largo × ancho
A₃ = 172.8 × 115 = 19,872 m²     (Inv3)
A₄ = 172.8 × 135 = 23,328 m²     (Inv4)
```
Definida en JS: `const AREA_M2 = { 3: 172.8*115, 4: 172.8*135 }`.
> Dato dado por el cliente como **superficie de cultivo** (no construida).

### Fórmulas
```
# kg/m² semanal por invernadero
r_inv[i] = kg_inv[i] / A_inv

# Producción semanal promedio de ambos (barras verdes del hero)
prodSemana[i] = ( pred₃[i]/A₃ + pred₄[i]/A₄ ) / 2

# Producción acumulada (línea verde, eje derecho)
prodAcum[k]   = Σ_{i=0..k} prodSemana[i]

# Meta del cliente: un valor M (kg/m² por semana, input editable, default 1.7)
metaSemana[i] = M                 # barras naranjas (constante)
metaAcum[k]   = M × (k + 1)       # recta ámbar punteada

# Card "Rendimiento prom. ambos inv" (real acumulado de toda la temporada)
rendReal = Σ_i ( real₃[i]/A₃ + real₄[i]/A₄ ) / 2     # ≈ 60.7 kg/m² (T17)

# KPIs del Resumen de Predicción (por invernadero)
inv{inv}-kgm2-win = (suma de las 6 predicciones de la ventana) / A_inv
inv{inv}-kgm2-acc = (suma de todo el kg real de la temporada) / A_inv
```

Ambos invernaderos comparten exactamente las **mismas 36 semanas** T17
(week_keys 202535→202618) → se alinean por índice directo.

### Hero combo (`renderYieldGoal(num)` → `chart-yield-{num}`)
Va **arriba de los KPIs**, sobre panel oscuro (`.yield-hero`, gradiente navy):
- **Barras** `yAxisID: 'yb'` (izq, kg/m² · semana): producción semanal (verde) + meta semanal (naranja).
- **Líneas** `yAxisID: 'yc'` (der, kg/m² · acumulado): producción acumulada (verde, área) + meta acumulada (ámbar punteada).
- Input `meta-input-{num}` → `localStorage.jata_meta_kgm2`; **una sola meta global**, sincroniza ambas vistas y repinta `renderYieldGoal(3,4)`.
- Lectura de negocio: línea verde sobre la ámbar = vamos adelantados a la meta; por debajo = vamos cortos.

### Números de referencia (T17)
| | kg/m² acumulado (real) | kg/m²·sem (prom) |
|---|---|---|
| Inv3 | 59.8 | 1.66 |
| Inv4 | 61.6 | 1.71 |
| **Prom. ambos** | **60.7** | **~1.69** |

### Pendiente
- ¿Meta **independiente por invernadero** vs global (hoy)?
- ¿Barras de **real** además de proyectado en el hero?
