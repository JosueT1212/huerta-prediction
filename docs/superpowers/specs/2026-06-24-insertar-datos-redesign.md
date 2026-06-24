# Insertar Datos — Redesign Spec

**Goal:** Restructure the "Insertar datos" view into two tabs — **Fenología** and **Producción real** — wiring both to live Supabase endpoints with a confirmation step before any DB write.

**Scope:** Frontend only (`demo/Demo Dashboard.html`). No new backend routes needed — all required endpoints already exist.

---

## Component Map

| File | Change |
|------|--------|
| `demo/Demo Dashboard.html` | Refactor `view-insert` section + add tab JS logic + wire endpoints + confirmation UI |

---

## Layout

Single sidebar entry "Insertar datos" (`view-insert`). Inside, a shared inv selector at top + two-tab bar:

```
Invernadero: [ 3 ▾ ]

[ Fenología ]  [ Producción real ]
───────────────────────────────────
<active panel>
```

Tab switching: JS toggles `.active` on panels. No new views or sidebar entries.

---

## Tab 1: Fenología

### Form fields (per-plant row)
Each row requires: `zona` (int, 1–4), `planta` (int, 1–4), plus the 8 pheno vars:

| field key | label | unit | range |
|-----------|-------|------|-------|
| `racimos_puestos` | Racimos puestos | conteo | 0–40 |
| `flores_racimo_abiertas` | Flores en racimo abiertas | conteo | 0–10 |
| `racimos_en_planta` | Racimos en planta | conteo | 0–15 |
| `cantidad_tomates` | Cantidad de tomates | conteo | 0–50 |
| `racimo_en_cosecha` | Racimo en cosecha | índice | 0–35 |
| `tomates_maduros` | Tomates maduros | conteo | 0–10 |
| `diametro_fruto_cm` | Diámetro fruto (cm) | cm | 0–6 |
| `crecimiento_planta_cm` | Crecimiento planta (cm) | cm | 0–40 |

### Date field
Replace "Semana del año" (ISO week number) with a **date picker** (`week_date`, ISO date string — Monday of the observed week). Backend upserts on `greenhouse_id, week_date, zona, planta`.

### Confirmation step
Clicking "Enviar captura" does NOT fire immediately. A confirmation panel appears:

```
Invernadero 3 · 2026-06-23
Plantas a enviar: 4

  Zona 1 · Planta 1: racimos=12, flores=3, ...
  Zona 1 · Planta 2: racimos=11, flores=4, ...
  ...

[ Cancelar ]   [ Confirmar envío ]
```

"Confirmar envío" fires one `POST /phenology-live/{inv}` per plant row (sequential), then shows ✓ or error.

### Endpoint
```
POST /phenology-live/{inv}
Authorization: Bearer <token>
Body: { week_date, zona, planta, racimos_puestos, flores_racimo_abiertas,
        racimos_en_planta, cantidad_tomates, racimo_en_cosecha,
        tomates_maduros, diametro_fruto_cm, crecimiento_planta_cm }
```
One request per plant. Upserts on conflict `(greenhouse_id, week_date, zona, planta)`.

---

## Tab 2: Producción real

### Form
```
Semana:    [ 2026-07-27 — pred: 34,894 kg ▾ ]   ← from GET /live-predictions/{inv}
kg reales: [ _________ ] kg
[ Registrar ]
```

Dropdown populated on tab open (or inv change) via `GET /live-predictions/{inv}`. Each option shows `predicted_for` date + `kg_predicted`. Options with `kg_actual` already set show ✓ and existing value.

### Confirmation step
```
Invernadero 3 · Semana 2026-07-27
Registrar kg reales: 37,200 kg
(predicción fue: 34,894 kg)

[ Cancelar ]   [ Confirmar ]
```

"Confirmar" fires `PATCH /live-predictions/{inv}/{id}`. On success, dropdown repopulates.

### Endpoint
```
PATCH /live-predictions/{inv}/{prediction_id}
Authorization: Bearer <token>
Body: { kg_actual: float }
```

---

## Auth
Both endpoints require JWT. Frontend already stores token in `sessionStorage` (key: `jata_token` or equivalent). All fetch calls must include `Authorization: Bearer <token>` header.

> **Check:** Verify sessionStorage key name used by existing live-predictions calls in `demo/Demo Dashboard.html` before implementing.

---

## Error Handling
- Network error → show red banner inside panel, do not clear form
- 422 validation error → show field-level hint from response detail
- 401 → show "Sesión expirada — recarga la página"

---

## Out of Scope
- Triggering inference after fenología submit (cron handles it)
- Sensor data entry (hardware-only)
- Inv4 phenology (form supports both invs via shared selector — no extra work)
