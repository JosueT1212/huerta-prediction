# Live Prediction Display & Refresh Design

**Goal:** Wire the live prediction data (from `/live-predictions/{inv}`) into all dashboard views — menu badges, inv3/inv4 forecast sections, T17 slider extension — and keep it fresh on tab focus. Remove duplicate kg_reales modal and inline fenología forms (now handled by "Insertar datos" tabs).

**Scope:** Frontend only (`demo/Demo Dashboard.html`). No new backend routes needed — existing `/inference/{inv}` (T17 arrays) and `/live-predictions/{inv}` (future predictions) together provide all required data.

---

## Refresh Architecture

### `refreshLiveData()`

Single async function — one fetch per inv, data passed directly to renderers. No global cache.

```js
async function refreshLiveData() {
  for (const inv of [3, 4]) {
    const data = await fetchJSON(`/live-predictions/${inv}?limit=20`);
    if (!data) continue;
    renderForecastSection(inv, data);
    renderMenuBadge(inv, data);
  }
}
```

Called from:
1. `init()` — after warm data loads, as final step
2. `showView(viewId)` — every navigation (replaces per-view `loadForecast` calls)
3. `visibilitychange` — tab focus restore:
   ```js
   document.addEventListener('visibilitychange', () => {
     if (!document.hidden) refreshLiveData();
   });
   ```

### Replace `loadForecast(inv)`

Delete `loadForecast`. Replace with `renderForecastSection(inv, data)` which accepts data as parameter instead of fetching:

```js
function renderForecastSection(inv, data) {
  // same render logic as current loadForecast — forecast cards + Chart.js line chart
  // upcoming = data.filter(d => !d.kg_actual).slice(0, 4)
  // chart: all rows sorted by predicted_for
}
```

---

## Menu View — Live Prediction Badge

Each inv card on the menu view gets a live prediction row below existing T17 KPIs.

**DOM target:** Two new placeholder divs added inside existing menu inv cards:
- `id="menu-live-badge-3"` (inv3 card)
- `id="menu-live-badge-4"` (inv4 card)

**`renderMenuBadge(inv, data)`:**
- Finds the most recent prediction with `kg_actual == null`
- If found: renders `◉ Próxima · {predicted_for} · {kg_predicted} kg`
- If none: hides the badge div

```js
function renderMenuBadge(inv, data) {
  const el = document.getElementById(`menu-live-badge-${inv}`);
  if (!el) return;
  const next = data.find(d => d.kg_actual == null);
  if (!next) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML =
    `<span style="color:#3b82f6;font-size:0.75rem;font-weight:600">◉ Próxima</span>` +
    `<span style="font-size:0.85rem;color:#f4ede0;margin-left:8px">` +
    `${next.predicted_for} · ${Math.round(next.kg_predicted).toLocaleString('es-MX')} kg</span>`;
}
```

---

## Removals

### Old kg_reales modal (replaced by "Insertar datos" → Producción real tab)

Remove from HTML:
- `<div id="actual-modal" ...>` block (~lines 3680+)

Remove from JS:
- `openActualModal(inv)` function
- `closeActualModal()` function
- `submitActualKg()` function
- `let _actualInv = 3` variable

Remove from inv3/inv4 views:
- "Registrar producción real" `<button onclick="openActualModal(3)">` and same for inv4

### Inline fenología forms (replaced by "Insertar datos" → Fenología tab)

Remove from inv3 view (~lines 1510–1545):
- "Fenología semanal" `<h3>` + date/zona/planta selects + 8 `ph3-*` inputs + "Guardar observación" button + `pheno-counter-3` + `pheno-msg-3`

Remove from inv4 view (same pattern, `ph4-*` IDs):
- Same block

Remove from JS:
- `submitPhenology(inv)` function (~line 3652)
- Any DOM references to `pheno-week-{inv}`, `pheno-zona-{inv}`, `pheno-planta-{inv}`, `ph3-*`, `ph4-*`, `pheno-counter-*`, `pheno-msg-*`

---

## Component Map

| File | Change |
|------|--------|
| `demo/Demo Dashboard.html` | Add `refreshLiveData`, `renderForecastSection`, `renderMenuBadge`; wire `visibilitychange`; add `menu-live-badge-{3,4}` divs; remove old modal + inline pheno forms + their JS |

---

## Error Handling

- `fetchJSON` returns `null` on failure — `refreshLiveData` skips render on null (existing pattern)
- Badge hidden if no upcoming predictions
- Forecast section shows "Sin predicciones próximas." if empty (existing pattern)

---

## T17 Slider Extension with Live Predictions

The slider currently shows 36 T17 test weeks. Live predictions append future weeks to the right, making the chart a continuous timeline: `T17 observed → live future`.

### Data merging

`refreshLiveData()` also calls `extendSliderWithLive(inv, data)`. This function appends live prediction weeks to `rawInv[inv]` arrays:

```js
function extendSliderWithLive(inv, data) {
  const raw = rawInv[inv];
  if (!raw) return;

  // Derive week key from predicted_for ISO date string
  function toWeekKey(dateStr) {
    const d = new Date(dateStr);
    // ISO week: use the same format as T17 week_key = year * 100 + iso_week
    const jan4 = new Date(d.getFullYear(), 0, 4);
    const isoWeek = Math.ceil(((d - jan4) / 86400000 + jan4.getDay() + 1) / 7);
    return d.getFullYear() * 100 + isoWeek;
  }

  // Only append weeks not already in T17
  const existingKeys = new Set(raw.weeks.map(w => w.week_key));
  const newRows = [...data]
    .sort((a, b) => a.predicted_for > b.predicted_for ? 1 : -1)
    .filter(d => !existingKeys.has(toWeekKey(d.predicted_for)));

  if (!newRows.length) return;

  // Append to raw arrays (y_true = null for future, pi = null)
  newRows.forEach(d => {
    const wk = toWeekKey(d.predicted_for);
    raw.weeks.push({ week_key: wk, iso_week: wk % 100, year: Math.floor(wk / 100) });
    raw.y_true.push(d.kg_actual ?? null);
    raw.y_pred.push(d.kg_predicted);
    raw.pi_lower.push(null);
    raw.pi_upper.push(null);
  });

  raw.n_weeks = raw.weeks.length;

  // Extend slider range
  const slider = document.getElementById(`slider${inv}`);
  if (slider) slider.max = raw.n_weeks - 1;
}
```

### Slider behavior for live weeks
- `y_true = kg_actual` (if recorded) or `null` (no real data yet) — chart omits null points
- `pi_lower / pi_upper = null` — no confidence interval shown for live predictions
- Slider thumb can navigate into live weeks; the detail chart and table render them as "Predicción en vivo" (not test-set history)
- Live weeks visually distinguished: dashed line style for `y_pred` in the live range (Chart.js `borderDash`)

### `refreshLiveData()` updated

```js
async function refreshLiveData() {
  for (const inv of [3, 4]) {
    const data = await fetchJSON(`/live-predictions/${inv}?limit=20`);
    if (!data) continue;
    renderForecastSection(inv, data);
    renderMenuBadge(inv, data);
    extendSliderWithLive(inv, data);
  }
}
```

---

## Out of Scope

- Polling (tab focus refresh sufficient for weekly-cadence inference)
- Backend changes (all data available from existing endpoints)
