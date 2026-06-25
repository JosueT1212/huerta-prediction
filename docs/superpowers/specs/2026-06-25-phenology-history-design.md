# Phenology History — Design Spec

**Goal:** Add a "Historial" tab to "Insertar datos" that shows all inserted phenology observations in a table and lets the user delete rows (with confirmation) if entered incorrectly.

**Scope:** One new backend endpoint + frontend tab in `demo/Demo Dashboard.html`. No changes to existing Fenología or Producción real tabs.

---

## Component Map

| File | Change |
|------|--------|
| `backend/routers/phenology_live.py` | Add `DELETE /phenology-live/{inv}/{id}` |
| `demo/Demo Dashboard.html` | Add "Historial" tab + panel + JS |

---

## Backend

### `DELETE /phenology-live/{inv}/{id}`

```
DELETE /phenology-live/{inv}/{id}
Authorization: Bearer <token>
```

Deletes the row matching `id` AND `greenhouse_id = inv` (double-guard prevents cross-inv deletes). Returns `{"ok": true}`. Returns 404 if row not found.

`id` is the Supabase auto-generated bigserial primary key, returned by `GET /phenology-live/{inv}` via `select("*")`.

---

## Frontend

### Tab bar

"Insertar datos" view gains a third tab button:

```
[ Fenología ]  [ Producción real ]  [ Historial ]
```

Same `ins-tabs` CSS + `initInsTabs()` pattern already in place. New tab ID: `ins-tab-hist`. New panel ID: `ins-panel-hist`.

### Inv selector

Reuses existing `ins-inv` `<select>` (shared across all tabs). Historial loads data for the selected inv on tab open and on inv change (same pattern as `loadProdPredictions`).

### Panel content

Fetches `GET /phenology-live/{inv}?limit=500`. Renders a scrollable table:

| Semana | Zona | Planta | Rac. puestos | Flores | Rac. planta | Tomates | Rac. cosecha | Maduros | Diám. (cm) | Crec. (cm) | — |
|--------|------|--------|---|---|---|---|---|---|---|---|---|

- Sorted by `week_date` desc, then `zona` asc, then `planta` asc
- Numeric cells show `—` for null values
- Last column: trash button `🗑` per row
- Empty state: `<p>Sin observaciones registradas.</p>`

### Delete confirmation

Clicking 🗑 on a row stores `{id, inv, week_date, zona, planta}` in a dataset on a confirm panel div (`id="hist-confirm"`), then shows it below the table:

```
Eliminar observación
Invernadero 3 · Semana 2026-06-23 · Zona 2 · Planta 3

[ Cancelar ]   [ Confirmar eliminación ]
```

"Confirmar eliminación" fires `DELETE /phenology-live/{inv}/{id}` with `Authorization: Bearer <token>`. On success: hides confirm panel, re-fetches table. On error: shows red message inside confirm panel.

Exact same confirm/cancel UI pattern as `ins-confirm` / `prod-confirm` already in the file.

---

## Auth

`localStorage.getItem('sb_token')` — Bearer token, same pattern as all other fetch calls in the dashboard.

---

## Error Handling

- 401 → "Sesión expirada — recarga la página"
- 404 → "Observación no encontrada (ya fue eliminada)"
- Network error → "Error de red — intenta de nuevo"
- All errors shown inside `hist-result` span below the confirm panel; confirm panel hides on any error/success

---

---

## kg_reales Delete (Producción real tab extension)

The "Producción real" tab already shows a dropdown of predictions with `✓ real: X kg` for confirmed weeks. Add a delete (clear) action so the user can remove a wrongly entered kg_actual.

### Backend

```
DELETE /live-predictions/{inv}/{id}/kg-actual
Authorization: Bearer <token>
```

Sets `kg_actual = NULL` on the prediction row (does not delete the prediction itself — only clears the real value). Returns `{"ok": true}`.

### Frontend

In `loadProdPredictions`, each option that already has `kg_actual` gets a visible "Borrar real" button next to it in the panel (not inside the `<select>` — a separate list row). On click: same confirm pattern:

```
Borrar producción real
Invernadero 3 · Semana 2026-07-27
kg real registrado: 37,200 kg

[ Cancelar ]   [ Confirmar ]
```

On confirm: fires the DELETE endpoint, re-fetches `loadProdPredictions` + `refreshLiveData`.

---

## Out of Scope

- Editing a row (delete + re-insert is the workflow for phenology)
- Bulk delete
- Pagination (limit=500 covers realistic use)
- Inv4 separate from inv3 — same shared inv selector handles both
