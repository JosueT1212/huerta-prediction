# Live Prediction Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `/live-predictions/{inv}` into menu badges, inv3/inv4 forecast sections, and T17 slider extension; remove dead duplicate kg_reales modal and inline fenología forms.

**Architecture:** Single `refreshLiveData()` function fetches live predictions for both invs, passes data to `renderForecastSection`, `renderMenuBadge`, and `extendSliderWithLive`. Called on `init()`, every navigation, and tab focus restore (`visibilitychange`). No backend changes — all data from existing endpoints.

**Tech Stack:** Vanilla JS, Chart.js (already loaded), single-file SPA (`demo/Demo Dashboard.html`)

## Global Constraints

- Single file: `demo/Demo Dashboard.html` — all changes in this file only
- No backend changes
- Spanish copy unchanged — do not translate or reword existing labels
- Auth: `localStorage.getItem('sb_token')` — Bearer token pattern (match existing code)
- `fetchJSON(path)` returns `null` on failure — always guard with `if (!data) continue/return`
- Chart instances tracked in `_forecastCharts[inv]` — call `.destroy()` before re-creating

---

## File Map

| File | Changes |
|------|---------|
| `demo/Demo Dashboard.html` | Task 1: remove dead HTML/JS · Task 2: add refresh arch + menu badges · Task 3: add slider extension |

---

### Task 1: Remove Dead Code

**Files:**
- Modify: `demo/Demo Dashboard.html` (HTML + JS sections)

**Interfaces:**
- Consumes: nothing (pure deletion)
- Produces: `loadForecast` gone (Tasks 2–3 add replacement); `_forecastCharts` gone (Task 2 re-adds it)

- [ ] **Step 1: Remove "Registrar producción real" button from inv3 view**

In `demo/Demo Dashboard.html`, find and delete lines 1502–1507 (the entire `<div style="margin-top:12px">` block containing the `openActualModal(3)` button):

```html
<!-- DELETE this block (lines ~1502-1507) -->
      <div style="margin-top:12px">
        <button onclick="openActualModal(3)"
          style="font-size:0.85rem;padding:6px 14px;border:1px solid #d2d6db;background:#fff;border-radius:6px;cursor:pointer">
          Registrar producción real
        </button>
      </div>
```

- [ ] **Step 2: Remove inline fenología form from inv3 view**

Delete lines 1510–1545 — the entire `<div style="margin-top:32px;max-width:640px">` block that starts with `<h3>Fenología semanal</h3>` and ends with `</div>` after `pheno-msg-3`. Exact old string to delete:

```html
    <div style="margin-top:32px;max-width:640px">
      <h3 style="font-size:1.1rem;font-weight:600;margin-bottom:16px">Fenología semanal</h3>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">
        <label style="font-size:0.85rem">Semana (lunes)<br>
          <input id="pheno-week-3" type="date" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
        </label>
        <label style="font-size:0.85rem">Zona<br>
          <select id="pheno-zona-3" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
            <option>1</option><option>2</option><option>3</option><option>4</option>
          </select>
        </label>
        <label style="font-size:0.85rem">Planta<br>
          <select id="pheno-planta-3" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
            <option>1</option><option>2</option><option>3</option><option>4</option>
          </select>
        </label>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        <label style="font-size:0.85rem">Racimos puestos<br><input id="ph3-racimos_puestos" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Flores en racimo abiertas<br><input id="ph3-flores_racimo_abiertas" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Racimos en planta<br><input id="ph3-racimos_en_planta" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Cantidad de tomates<br><input id="ph3-cantidad_tomates" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Racimo en cosecha<br><input id="ph3-racimo_en_cosecha" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Tomates maduros<br><input id="ph3-tomates_maduros" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Diámetro fruto (cm)<br><input id="ph3-diametro_fruto_cm" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Crecimiento planta (cm)<br><input id="ph3-crecimiento_planta_cm" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
      </div>
      <div style="display:flex;align-items:center;gap:16px">
        <button onclick="submitPhenology(3)"
          style="background:#65a30d;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:0.9rem">
          Guardar observación
        </button>
        <span id="pheno-counter-3" style="font-size:0.85rem;color:#9aa0a8"></span>
      </div>
      <p id="pheno-msg-3" style="margin-top:8px;font-size:0.85rem;color:#16a34a;display:none">Guardado.</p>
    </div>
```

- [ ] **Step 3: Remove "Registrar producción real" button from inv4 view**

Delete lines ~1644–1649 (same pattern as inv3, `openActualModal(4)`):

```html
<!-- DELETE this block -->
      <div style="margin-top:12px">
        <button onclick="openActualModal(4)"
          style="font-size:0.85rem;padding:6px 14px;border:1px solid #d2d6db;background:#fff;border-radius:6px;cursor:pointer">
          Registrar producción real
        </button>
      </div>
```

- [ ] **Step 4: Remove inline fenología form from inv4 view**

Delete lines ~1652–1687 — same structure as inv3 but with `ph4-*` IDs and `submitPhenology(4)`:

```html
    <div style="margin-top:32px;max-width:640px">
      <h3 style="font-size:1.1rem;font-weight:600;margin-bottom:16px">Fenología semanal</h3>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">
        <label style="font-size:0.85rem">Semana (lunes)<br>
          <input id="pheno-week-4" type="date" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
        </label>
        <label style="font-size:0.85rem">Zona<br>
          <select id="pheno-zona-4" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
            <option>1</option><option>2</option><option>3</option><option>4</option>
          </select>
        </label>
        <label style="font-size:0.85rem">Planta<br>
          <select id="pheno-planta-4" style="margin-top:4px;padding:6px;border:1px solid #d2d6db;border-radius:6px">
            <option>1</option><option>2</option><option>3</option><option>4</option>
          </select>
        </label>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        <label style="font-size:0.85rem">Racimos puestos<br><input id="ph4-racimos_puestos" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Flores en racimo abiertas<br><input id="ph4-flores_racimo_abiertas" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Racimos en planta<br><input id="ph4-racimos_en_planta" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Cantidad de tomates<br><input id="ph4-cantidad_tomates" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Racimo en cosecha<br><input id="ph4-racimo_en_cosecha" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Tomates maduros<br><input id="ph4-tomates_maduros" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Diámetro fruto (cm)<br><input id="ph4-diametro_fruto_cm" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
        <label style="font-size:0.85rem">Crecimiento planta (cm)<br><input id="ph4-crecimiento_planta_cm" type="number" step="0.1" style="width:100%;padding:6px;border:1px solid #d2d6db;border-radius:6px;margin-top:4px"></label>
      </div>
      <div style="display:flex;align-items:center;gap:16px">
        <button onclick="submitPhenology(4)"
          style="background:#65a30d;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:0.9rem">
          Guardar observación
        </button>
        <span id="pheno-counter-4" style="font-size:0.85rem;color:#9aa0a8"></span>
      </div>
      <p id="pheno-msg-4" style="margin-top:8px;font-size:0.85rem;color:#16a34a;display:none">Guardado.</p>
    </div>
```

- [ ] **Step 5: Remove loadForecast + _forecastCharts from JS**

Delete lines ~3554–3593 — the `_forecastCharts` const and `loadForecast` async function:

```js
// DELETE: lines ~3554-3593
const _forecastCharts = {};

async function loadForecast(inv) {
  const data = await fetchJSON(`/live-predictions/${inv}?limit=20`);
  if (!data) return;

  const upcoming = data.filter(d => !d.kg_actual).slice(0, 4);
  const cards = document.getElementById(`forecast-cards-${inv}`);
  if (cards) {
    cards.innerHTML = upcoming.length
      ? upcoming.map(d => `
          <div style="background:#1e2a3a;border-radius:8px;padding:16px;min-width:130px;text-align:center">
            <div style="font-size:0.75rem;color:#9aa0a8;margin-bottom:6px">
              ${new Date(d.predicted_for).toLocaleDateString('es-MX',{month:'short',day:'numeric'})}
            </div>
            <div style="font-size:1.4rem;font-weight:600;color:#f4ede0">${d.kg_predicted.toFixed(0)} kg</div>
          </div>`).join('')
      : '<p style="color:#9aa0a8">Sin predicciones próximas.</p>';
  }

  const ctx = document.getElementById(`forecast-chart-${inv}`);
  if (!ctx) return;
  const sorted = [...data].sort((a,b) => a.predicted_for > b.predicted_for ? 1 : -1);
  if (_forecastCharts[inv]) _forecastCharts[inv].destroy();
  _forecastCharts[inv] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sorted.map(d => d.predicted_for),
      datasets: [
        { label: 'Predicción (kg)', data: sorted.map(d => d.kg_predicted),
          borderColor: '#3b82f6', backgroundColor: 'transparent', tension: 0.3 },
        { label: 'Real (kg)', data: sorted.map(d => d.kg_actual),
          borderColor: '#16a34a', backgroundColor: '#16a34a',
          pointStyle: 'circle', pointRadius: 5, showLine: false },
      ],
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } },
               scales: { y: { beginAtZero: false } } },
  });
}
```

- [ ] **Step 6: Remove _actualInv + openActualModal + closeActualModal + submitActualKg from JS**

Delete lines ~3595–3633:

```js
// DELETE: lines ~3595-3633
let _actualInv = 3;
async function openActualModal(inv) {
  _actualInv = inv;
  const data = await fetchJSON(`/live-predictions/${inv}?limit=20`);
  // ... (full block)
}
function closeActualModal() { ... }
async function submitActualKg() { ... }
```

Find and delete by matching the literal text starting from `let _actualInv = 3;` through the closing `}` of `submitActualKg`.

- [ ] **Step 7: Remove PHENO_FIELDS + loadPhenoCounter + submitPhenology from JS**

Delete lines ~3636–3677:

```js
// DELETE: lines ~3636-3677
const PHENO_FIELDS = [
  'racimos_puestos','flores_racimo_abiertas','racimos_en_planta',
  'cantidad_tomates','racimo_en_cosecha','tomates_maduros',
  'diametro_fruto_cm','crecimiento_planta_cm',
];

async function loadPhenoCounter(inv) { ... }
async function submitPhenology(inv) { ... }
```

- [ ] **Step 8: Remove actual-modal HTML div**

Delete lines ~3680–3695 — the entire `<div id="actual-modal" ...>` block at the bottom of the file.

- [ ] **Step 9: Remove loadForecast calls from showView**

In `showView` function (~lines 2289–2291), delete the two loadForecast calls:

Old (delete these two lines):
```js
  if (viewId === 'inv3') loadForecast(3);
  if (viewId === 'inv4') loadForecast(4);
```

The `showView` function closing brace `}` on line 2292 stays.

- [ ] **Step 10: Verify no references remain**

Open `demo/Demo Dashboard.html` and search (Ctrl+F or grep) for each of these — none should appear:
- `loadForecast`
- `openActualModal`
- `closeActualModal`
- `submitActualKg`
- `_actualInv`
- `submitPhenology`
- `loadPhenoCounter`
- `PHENO_FIELDS`
- `actual-modal`
- `pheno-week-3`
- `pheno-week-4`
- `ph3-`
- `ph4-`

Run check:
```bash
grep -n "loadForecast\|openActualModal\|closeActualModal\|submitActualKg\|_actualInv\|submitPhenology\|loadPhenoCounter\|actual-modal\|pheno-week-3\|pheno-week-4\|ph3-\|ph4-" "demo/Demo Dashboard.html"
```
Expected: no output.

- [ ] **Step 11: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "refactor: remove dead modal, inline pheno forms, and loadForecast (replaced in next task)"
```

---

### Task 2: Refresh Architecture + Menu Badges

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `fetchJSON(path)` (existing) · `rawInv[inv]` (existing global) · `_forecastCharts` (re-added here)
- Produces: `renderForecastSection(inv, data)` · `renderMenuBadge(inv, data)` · `refreshLiveData()` — called by Task 3's `extendSliderWithLive`

- [ ] **Step 1: Add menu-live-badge div to inv3 menu card**

In `demo/Demo Dashboard.html`, find the inv3 menu card chart line:

```html
        <div class="gh-chart"><canvas id="chart-menu-3"></canvas></div>
```

Replace with:

```html
        <div class="gh-chart"><canvas id="chart-menu-3"></canvas></div>
        <div id="menu-live-badge-3" hidden style="padding:8px 0 4px;display:flex;align-items:center;gap:6px;border-top:1px solid rgba(0,0,0,0.08);margin-top:8px"></div>
```

- [ ] **Step 2: Add menu-live-badge div to inv4 menu card**

Find the inv4 menu card chart line:

```html
        <div class="gh-chart"><canvas id="chart-menu-4"></canvas></div>
```

Replace with:

```html
        <div class="gh-chart"><canvas id="chart-menu-4"></canvas></div>
        <div id="menu-live-badge-4" hidden style="padding:8px 0 4px;display:flex;align-items:center;gap:6px;border-top:1px solid rgba(0,0,0,0.08);margin-top:8px"></div>
```

- [ ] **Step 3: Add _forecastCharts + renderForecastSection + renderMenuBadge + refreshLiveData**

Find the `</script>` tag that closes the main script block (the `</script>` that appears right before `<div id="actual-modal"` was — now right before end of file or next script-less block). Add these functions just before that `</script>`:

```js
// ── Live Prediction Display ───────────────────────────────────────────────
const _forecastCharts = {};

function renderForecastSection(inv, data) {
  const upcoming = data.filter(d => !d.kg_actual).slice(0, 4);
  const cards = document.getElementById(`forecast-cards-${inv}`);
  if (cards) {
    cards.innerHTML = upcoming.length
      ? upcoming.map(d => `
          <div style="background:#1e2a3a;border-radius:8px;padding:16px;min-width:130px;text-align:center">
            <div style="font-size:0.75rem;color:#9aa0a8;margin-bottom:6px">
              ${new Date(d.predicted_for + 'T12:00:00').toLocaleDateString('es-MX',{month:'short',day:'numeric'})}
            </div>
            <div style="font-size:1.4rem;font-weight:600;color:#f4ede0">${Math.round(d.kg_predicted).toLocaleString('es-MX')} kg</div>
          </div>`).join('')
      : '<p style="color:#9aa0a8">Sin predicciones próximas.</p>';
  }

  const ctx = document.getElementById(`forecast-chart-${inv}`);
  if (!ctx) return;
  const sorted = [...data].sort((a, b) => a.predicted_for > b.predicted_for ? 1 : -1);
  if (_forecastCharts[inv]) _forecastCharts[inv].destroy();
  _forecastCharts[inv] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sorted.map(d => d.predicted_for),
      datasets: [
        { label: 'Predicción (kg)', data: sorted.map(d => d.kg_predicted),
          borderColor: '#3b82f6', backgroundColor: 'transparent', tension: 0.3 },
        { label: 'Real (kg)', data: sorted.map(d => d.kg_actual ?? null),
          borderColor: '#16a34a', backgroundColor: '#16a34a',
          pointStyle: 'circle', pointRadius: 5, showLine: false },
      ],
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } },
               scales: { y: { beginAtZero: false } } },
  });
}

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

async function refreshLiveData() {
  for (const inv of [3, 4]) {
    const data = await fetchJSON(`/live-predictions/${inv}?limit=20`);
    if (!data) continue;
    renderForecastSection(inv, data);
    renderMenuBadge(inv, data);
  }
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshLiveData();
});
```

- [ ] **Step 4: Wire refreshLiveData into showView**

In `showView`, where the two `loadForecast` lines were removed (after the sensor polling block), add one call:

Find in `showView`:
```js
  // start/stop sensor polling when entering/leaving live views
  if (viewId === 'live3') startSensorPolling(3);
  else stopSensorPolling(3);
  if (viewId === 'live4') startSensorPolling(4);
  else stopSensorPolling(4);
}
```

Replace with:
```js
  // start/stop sensor polling when entering/leaving live views
  if (viewId === 'live3') startSensorPolling(3);
  else stopSensorPolling(3);
  if (viewId === 'live4') startSensorPolling(4);
  else stopSensorPolling(4);

  refreshLiveData();
}
```

- [ ] **Step 5: Wire refreshLiveData into init()**

In `init()`, after the last setup line before the `} catch` block, add `refreshLiveData()`:

Find:
```js
    wireMeta(3); wireMeta(4);
    renderYieldGoal(3); renderYieldGoal(4);

    console.log('[Huerta] backend OK',
```

Replace with:
```js
    wireMeta(3); wireMeta(4);
    renderYieldGoal(3); renderYieldGoal(4);

    refreshLiveData();

    console.log('[Huerta] backend OK',
```

- [ ] **Step 6: Verify in browser**

Open `demo/Demo Dashboard.html` via the backend (`http://localhost:8000`). In browser console:
```js
// Should not throw ReferenceError
refreshLiveData();
```

Navigate to Invernadero 3 view — `forecast-cards-3` should show prediction cards (or "Sin predicciones próximas." if empty).

Navigate to Menu — `menu-live-badge-3` and `menu-live-badge-4` divs should exist in DOM:
```js
document.getElementById('menu-live-badge-3') // should return element, not null
document.getElementById('menu-live-badge-4') // same
```

Verify deleted elements gone:
```js
document.getElementById('actual-modal')     // null
document.querySelector('[onclick="openActualModal(3)"]') // null
document.getElementById('pheno-week-3')     // null
```

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat: add refreshLiveData, renderForecastSection, renderMenuBadge — live predictions wired"
```

---

### Task 3: T17 Slider Extension

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `rawInv[inv]` — object with shape `{ weeks: [{week_key, iso_week, year}], y_true: number[], y_pred: number[], pi_lower: number[], pi_upper: number[], n_weeks: number }` · `refreshLiveData()` from Task 2
- Produces: `extendSliderWithLive(inv, data)` — modifies `rawInv[inv]` in-place, extends slider `max`

- [ ] **Step 1: Add extendSliderWithLive function**

In `demo/Demo Dashboard.html`, find the `// ── Live Prediction Display` comment block added in Task 2. After `refreshLiveData` and before the `visibilitychange` listener, insert:

```js
function extendSliderWithLive(inv, data) {
  const raw = rawInv[inv];
  if (!raw) return;

  function toWeekKey(dateStr) {
    const d = new Date(dateStr + 'T12:00:00');
    const year = d.getFullYear();
    const jan4 = new Date(year, 0, 4);
    const isoWeek = Math.ceil(((d - jan4) / 86400000 + jan4.getDay() + 1) / 7);
    return year * 100 + isoWeek;
  }

  const existingKeys = new Set(raw.weeks.map(w => w.week_key));
  const newRows = [...data]
    .sort((a, b) => a.predicted_for > b.predicted_for ? 1 : -1)
    .filter(d => !existingKeys.has(toWeekKey(d.predicted_for)));

  if (!newRows.length) return;

  newRows.forEach(d => {
    const wk = toWeekKey(d.predicted_for);
    raw.weeks.push({ week_key: wk, iso_week: wk % 100, year: Math.floor(wk / 100) });
    raw.y_true.push(d.kg_actual ?? null);
    raw.y_pred.push(d.kg_predicted);
    raw.pi_lower.push(null);
    raw.pi_upper.push(null);
  });

  raw.n_weeks = raw.weeks.length;

  const slider = document.getElementById(`slider${inv}`);
  if (slider) slider.max = raw.n_weeks - 1;
}
```

- [ ] **Step 2: Call extendSliderWithLive from refreshLiveData**

Find `refreshLiveData` (added in Task 2):

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

Replace with:

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

- [ ] **Step 3: Verify in browser**

After backend is running (`uvicorn backend.main:app --port 8000`), open the dashboard and navigate to Invernadero 3. In browser console:

```js
// After page loads, check if live weeks were appended
rawInv[3].n_weeks  // should be > 36 if any live predictions exist in DB
rawInv[3].weeks.slice(-3)  // should show future week_keys (202630+)

// Navigate slider to last position
const sl = document.getElementById('slider3');
sl.max  // should match rawInv[3].n_weeks - 1
sl.value = sl.max;
sl.dispatchEvent(new Event('input'));
// chart should render the live prediction week
```

If DB has zero live predictions, `n_weeks` stays 36 and the slider is unchanged — this is correct behavior (no error).

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat: extend T17 slider with live prediction weeks via extendSliderWithLive"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|------------------|------|
| `refreshLiveData()` — fetch both invs, pass to renderers | Task 2 Step 3 |
| Called from `init()` | Task 2 Step 5 |
| Called from `showView()` (every navigation) | Task 2 Step 4 |
| `visibilitychange` tab focus | Task 2 Step 3 |
| `renderForecastSection(inv, data)` — replaces `loadForecast` | Task 2 Step 3 |
| `renderMenuBadge(inv, data)` | Task 2 Step 3 |
| `menu-live-badge-{3,4}` divs in HTML | Task 2 Steps 1–2 |
| Remove `actual-modal` HTML | Task 1 Step 8 |
| Remove `openActualModal`, `closeActualModal`, `submitActualKg`, `_actualInv` | Task 1 Step 6 |
| Remove "Registrar producción real" buttons | Task 1 Steps 1, 3 |
| Remove inline fenología forms (inv3 + inv4) | Task 1 Steps 2, 4 |
| Remove `submitPhenology`, `loadPhenoCounter` | Task 1 Step 7 |
| Remove `loadForecast` | Task 1 Step 5 |
| `extendSliderWithLive(inv, data)` | Task 3 Step 1 |
| Called from `refreshLiveData` | Task 3 Step 2 |
| `y_true = kg_actual ?? null` for live weeks | Task 3 Step 1 |
| `pi_lower/pi_upper = null` for live weeks | Task 3 Step 1 |
| Slider `max` extended | Task 3 Step 1 |

All spec requirements covered. No gaps.

**Type consistency:**
- `renderForecastSection(inv: number, data: array)` — Task 2 defines, no downstream callers outside `refreshLiveData`
- `renderMenuBadge(inv: number, data: array)` — same
- `extendSliderWithLive(inv: number, data: array)` — Task 3 defines, called in Task 3 Step 2 with same signature

All consistent.

**Placeholder scan:** No TBDs. All code blocks complete.
