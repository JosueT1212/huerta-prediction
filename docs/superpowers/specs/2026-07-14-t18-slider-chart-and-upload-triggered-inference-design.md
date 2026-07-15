# T18: slider/gráfica/tabla en tab Predicción + inferencia disparada por carga de datos

Date: 2026-07-14

## Problem

El tab "Predicción" (`view-inv3`/`view-inv4`) tiene dos sub-vistas por
temporada: `season-panel-t17-{inv}` (histórico, con slider manual + chart +
tabla "Próximas Semanas") y `season-panel-t18-{inv}` (en vivo), pero esta
última hoy solo muestra `forecast-cards-{inv}` (tarjetas simples), sin
slider ni chart ni tabla equivalente al panel T17. El cliente quiere paridad
visual entre T17 y T18 dentro de este tab.

Además, hoy la inferencia de T18 solo corre por cron semanal
(`scripts/live_inference.py`, Railway, domingos 8am UTC). El cliente quiere
que la inferencia corra automáticamente en cuanto toda la data de la semana
esté cargada (excepto producción, que es un dato posterior/de verificación,
no un insumo del modelo).

## Scope

- Frontend: agregar slider + chart + tabla "Próximas Semanas" (clicable →
  Histórico) dentro de `season-panel-t18-{3,4}`, backed by los datos ya
  cargados en `rawLive[inv]` (`/live-predictions/{inv}?season=T18`). No se
  toca `season-panel-t17-{inv}` ni la vista Histórico (`view-hist3/4`, ya
  lee T18 en vivo desde commit `64f4b9aa`).
- Backend: trigger de inferencia inmediato tras el upload que completa el
  set de datos requerido (sensores + riego + fenología por invernadero +
  exteriores global) para la semana en curso. Cron semanal se mantiene como
  respaldo (no se modifica `scripts/live_inference.py` como script standalone,
  solo se expone su lógica para ser invocada desde el backend).
- Fuera de alcance: cambios al script de cron en sí, cambios a
  `season-panel-t17`, autodetección de temporada activa (T18 sigue
  hardcodeado, igual que en el resto del dashboard).

## Frontend design

### Datos

`rawLive[inv]` ya existe (poblado por `loadLiveSeason(inv)`,
`demo/Demo Dashboard.html:2311-2314`), ordenado ascendente por
`predicted_for`. Cada fila: `{id, predicted_for, kg_predicted, kg_actual,
model_version, season}`. A diferencia de T17 (`rawInv[inv]`), no trae
`pi_lower`/`pi_upper` ni longitud fija de 36 semanas — el arreglo crece con
cada carga semanal.

### Slider

Nuevo slider `slider3-t18`/`slider4-t18` dentro de `season-panel-t18-{inv}`,
mismo markup que el `.sim-bar` de T17 (`sim-kicker`, `sim-week`, `sim-pos`,
botones `‹ ›`). Rango: `min=0`, `max=rawLive[inv].length-1`,
`value=rawLive[inv].length-1` (última semana con predicción por default,
en vez de la mitad como T17 — en T18 "lo más reciente" es lo relevante, no
el punto medio). Si `rawLive[inv].length === 0`, el panel muestra el mismo
empty-state ya usado hoy ("Cargando predicciones…" / mensaje vacío) y el
slider no se renderiza.

Reutiliza el patrón de clamp + `updateSliderFill` de `wireSimControls`
(`2372-2377`), pero como una función nueva `wireLiveSimControls(inv)` — no
se puede compartir literalmente `wireSimControls` porque el id de los
elementos y el arreglo fuente difieren.

### Chart y tabla

Nueva función `renderLiveInv(inv, cursor)` (paralela a `renderInv`,
`2343-2364`, pero sin PI bounds):
- Chart `chart-inv3-t18`/`chart-inv4-t18` (nuevo `<canvas>`): ventana de
  `past=5` semanas antes del cursor + el cursor mismo, línea predicho +
  línea real (con huecos donde `kg_actual` es `null`, sin intervalo de
  confianza ya que T18 no lo tiene).
- Tabla "Próximas Semanas" (`tbody-inv3-t18`/`tbody-inv4-t18`), mismo
  layout que la de T17 salvo sin columna "Intervalo" (T18 no tiene PI) —
  columnas: Semana, Predicho (kg), Real (kg) si existe. Click en el panel
  (`data-goto="hist3"`, igual que T17) navega a Histórico.

### Integración con `refreshLiveData`

`refreshLiveData()` (`2035-2053`) ya llama `loadLiveSeason(inv)`; se agrega
ahí mismo la llamada a `renderLiveInv(inv, cursor actual o length-1)` para
que el slider/chart/tabla T18 se mantengan sincronizados con el polling
existente (visibilitychange), igual que Vista general/Histórico.

## Backend design

### Detección de "set completo" para la semana en curso

`backend/lock_utils.get_lock_status(greenhouse_id, form_type)` ya expone
`last_submitted_at` (indirectamente, vía `_fetch_last_submitted_at`, hoy no
público — se expone). Se define en `backend/lock_utils.py` una nueva
función:

```python
def uploads_complete_for_inv(inv: int) -> bool:
    """True si sensores+riego+fenologia (inv) + exteriores (global)
    tienen last_submitted_at dentro de la misma ventana semanal actual."""
```

Regla de "misma ventana semanal": los 4 `last_submitted_at` deben caer
dentro de los últimos `LOCK_WINDOW` (7 días) counting from `now()` — mismo
criterio que ya usa el lock para bloquear reenvíos, sin inventar una nueva
noción de "semana".

### Trigger tras upload

En `backend/routers/uploads.py`, tanto `upload_excel` (línea 202) como
`upload_exteriores` (línea 269), después de `touch_submission_lock(...)`:

```python
if rows_inserted + rows_updated > 0:
    touch_submission_lock(inv, form_type)
    _maybe_trigger_inference(inv if form_type != "exteriores" else None, form_type)
```

`_maybe_trigger_inference`:
- Si `form_type == "exteriores"`: exteriores es global, así que se checa
  para **ambos** invernaderos (3 y 4).
- Para cada invernadero a chequear: si `uploads_complete_for_inv(inv)` es
  `True` **y** no lo era antes de este upload (evita relanzar en cada envío
  subsecuente de la misma semana — se determina comparando si el
  `form_type` recién tocado era el único que faltaba), llama
  `scripts.live_inference.run_inference_for_greenhouse(supa, inv,
  dry_run=False)` de forma síncrona dentro del request.
- Excepciones de la inferencia se capturan y loguean (`print(...,
  file=sys.stderr)`, mismo patrón que `scripts/live_inference.py:217-218`)
  pero **no** hacen fallar la respuesta del upload — la carga de datos ya
  se guardó correctamente, un fallo de inferencia es secundario y el cron
  semanal lo cubre como red de seguridad.

### Reuso de `live_inference.py`

`run_inference_for_greenhouse(supa, inv, dry_run)` (`scripts/live_inference.py:87`)
ya es una función pura reusable — no requiere cambios. Se importa desde
`backend/routers/uploads.py`. Dado que `backend/main.py` ya importa la
cadena torch/joblib/CNNRNN al arrancar (warm de T17), el costo de import
adicional en `uploads.py` es marginal (mismos módulos ya cargados en
memoria del proceso).

`produccion` NO participa en `uploads_complete_for_inv` (no es insumo del
modelo, es dato de verificación posterior) — confirma el requisito del
cliente ("excepting production registration").

## Empty / edge states

- T18 sin ninguna predicción aún: panel muestra mensaje vacío existente
  ("Cargando predicciones…" ajustado a "Aún no hay predicciones para T18"
  si el fetch retorna `[]`), sin slider.
- Solo 1 semana de datos: slider con `min=max=0`, deshabilitado
  (mismo patrón visual que un slider de un solo valor — sin flechas
  funcionales).
- Trigger de inferencia disparado dos veces por una carrera entre uploads
  concurrentes: idempotente por diseño (`run_inference_for_greenhouse`
  hace `upsert` sobre `predictions` keyed por `greenhouse_id,predicted_for`
  — una segunda ejecución simplemente sobreescribe con el mismo resultado).

## Testing

- Manual: cargar Excel de prueba en los 4 tipos (sensores, riego,
  fenología, exteriores) para un invernadero vía `/uploads/...` y
  confirmar (log de backend) que la inferencia se dispara solo después del
  último tipo faltante, no en cada carga.
- Manual: mover el slider T18 en el dashboard, confirmar que chart/tabla
  se actualizan y que el click en el panel navega a Histórico.
- Manual: probar temporada T18 vacía (sin datos) → sin errores en consola,
  empty-state visible.
- Backend: test unitario para `uploads_complete_for_inv` con locks
  mockeados (todos dentro de ventana / uno vencido / uno ausente).
