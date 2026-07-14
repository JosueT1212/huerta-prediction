# Vista general / KPIs / Histórico → datos de temporada en vivo (T18)

Date: 2026-07-14

## Problem

`demo/Demo Dashboard.html` tiene tres secciones (Vista general, KPIs & gráficas,
Histórico) que hoy leen exclusivamente del snapshot cacheado de T17
(`/metrics`, `/phenology`, `/sensor-history`, npz de inferencia). El único
lugar que ya muestra datos de la temporada activa (T18) es el tab
`season-panel-t18-{inv}` dentro del detalle de invernadero (introducido en
commit `79056098`). El cliente quiere que las tres secciones reflejen la
temporada en vivo, no el replay de T17.

## Scope

- Reescritura de frontend (`demo/Demo Dashboard.html`) para las 3 secciones.
- Sin cambios de backend: los endpoints necesarios ya existen
  (`/live-predictions/{inv}?season=`, `/phenology-live/{inv}`).
- Sin auto-detección de temporada activa: `"T18"` queda como literal
  hardcodeado, igual que ya existe en `scripts/live_inference.py:38`
  (`CURRENT_SEASON`) y en `refreshLiveData()` del propio dashboard.
- Fenología en vivo SÍ entra en alcance (tabla `phenology_observations` y
  endpoint `/phenology-live/{inv}` ya existen, no es trabajo desde cero).
- Curva de meta kg/m²·semana (agronómica) NO cambia — se reutiliza igual
  que en T17, solo cambia la serie proyectado/real.
- Tarjetas mock de KPIs (calibres/error/recursos, ver `demo/CLAUDE.md` §5)
  se eliminan por completo (markup, estilos y JS asociado, ambos
  invernaderos) — no se dejan ocultas ni se reemplazan por dato real.
- Fuera de alcance: modal "ver histórico" de sensores (sigue leyendo
  Excel T13–T17, sin dato vivo de sensores en este spec).

## Data sources (ya existentes, sin cambios de backend)

- `GET /live-predictions/{inv}?season=T18` — filas `{predicted_for,
  kg_predicted, kg_actual, ...}` por semana. Fuente única de verdad para
  totales, pico, serie proyectado/real, y tabla histórico.
- `GET /phenology-live/{inv}` — última observación por zona/planta desde
  Supabase `phenology_observations`. Fuente para panel de fenología en KPIs.

## Design por sección

### Vista general (`data-view="menu"`)
`menu3-total` / `menu3-peak` (y equivalentes inv4) dejan de leer el
snapshot cacheado de `/inference/{inv}` (T17) y se calculan sumando/
maximizando sobre las filas de `/live-predictions/{inv}?season=T18`
(usa `kg_actual` si existe, si no `kg_predicted`). Se calcula en una
función compartida, llamada tanto en `init()` como en el poll existente
de `refreshLiveData()`.

### KPIs & gráficas (`kpi3`/`kpi4`)
- Hero `inv{n}-kgm2-win` / `inv{n}-kgm2-acc`: mismo cálculo `total/area`
  y `seasonReal/area` (usa `AREA_M2[inv]` ya definido), pero `total` y
  `seasonReal` ahora vienen de la misma función compartida de totales
  vivos (no de `/metrics`).
- Panel combo meta-vs-proyectado: curva de meta se mantiene igual
  (valores T17 reutilizados). Serie proyectado/real se reconstruye desde
  las filas semanales de `/live-predictions`.
- Panel de fenología: cambia su fuente de `/phenology/{inv}` (Excel
  estático) a `/phenology-live/{inv}` (Supabase, T18).
- Tarjetas mock (calibres/error/recursos): eliminadas por completo del
  markup/CSS/JS, ambos invernaderos.

### Histórico (`hist3`/`hist4`)
Tabla predicho vs real vs %error se reconstruye desde
`/live-predictions/{inv}?season=T18` en vez del array npz cacheado.
Modal "ver histórico" de sensores queda sin cambios (fuera de alcance).

## Empty state

Si la temporada T18 aún no tiene filas (temporada recién iniciada, sin
primera carga), las tres secciones muestran el mismo estado vacío que ya
usa `season-panel-t18-{inv}` — no deben romperse ni mostrar `NaN`/vacío
sin mensaje.

## Testing

Manual: levantar backend + dashboard, comparar números de Vista
general/KPIs/Histórico contra los ya mostrados en `season-panel-t18`
(deben coincidir, misma fuente). Probar caso de T18 sin filas (empty
state). Probar ambos invernaderos (3 y 4).
