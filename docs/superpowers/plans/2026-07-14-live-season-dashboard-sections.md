# Vista general / KPIs / Histórico → datos de temporada en vivo (T18) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire Vista general, KPIs & gráficas, and Histórico in `demo/Demo Dashboard.html` to read live T18 data (`/live-predictions`, `/phenology-live`) instead of the cached T17 snapshot, and delete all remaining mock KPI panels.

**Architecture:** Frontend-only change to the single-file vanilla JS SPA `demo/Demo Dashboard.html`. No backend changes — `/live-predictions/{inv}?season=` and `/phenology-live/{inv}` already exist and return the needed data. A new client-side cache (`rawLive`, `rawLivePheno`) is populated once in `init()` and reused by the three sections; the existing T17-cache-driven "Predicción" slider view (`view-inv3`/`view-inv4`, `renderInv()`) is left untouched.

**Tech Stack:** Vanilla JS, Chart.js 4.4 (CDN), FastAPI backend (unchanged), Supabase (unchanged).

## Global Constraints

- Season is hardcoded to the literal `"T18"` — no auto-detection of current season (matches existing `CURRENT_SEASON` in `scripts/live_inference.py:38`).
- No backend changes. Only `demo/Demo Dashboard.html` is touched.
- `/live-predictions/{inv}?season=T18` used with `limit=200` for season-wide totals (not the existing `limit=20` fetch used by `refreshLiveData()` for forecast cards — that stays separate, unrelated to this work).
- `demo/Demo Dashboard.html` has no automated test runner (per `demo/CLAUDE.md`, verification is manual/Playwright in-browser). Each task's verification step is a manual browser check via `curl`/dev console or Playwright, not `pytest`.
- Reuse existing helpers: `fetchJSON`, `fmtKg`, `fmt`, `fmtRange`, `setText`, `errorBucket`, `AREA_M2`, `getMeta`, `sparklineSVG`, `charts` registry. Do not duplicate them.
- Empty-state pattern to replicate (from `renderForecastSection`, `Demo Dashboard.html:3141-3153`): ternary on `data.length`, fallback message `Sin ... de T18 todavía.`

---

### Task 1: Live-season data loader + shared helpers

**Files:**
- Modify: `demo/Demo Dashboard.html` (near `const rawInv = {};` at line 2541, and inside `init()` at lines 3012-3053)

**Interfaces:**
- Produces: `rawLive` (object, `{3: Array|null, 4: Array|null}`, each array item shaped like a `/live-predictions` row: `{id, predicted_for, kg_predicted, kg_actual, season, ...}`, sorted ascending by `predicted_for`), `async function loadLiveSeason(inv)`, `function liveKg(row)`, `function liveTotalsFor(inv)` (returns `{rows, total, peak}`), `function fmtWeekLabel(dateStr)`.
- Consumes: existing `fetchJSON(path)` helper.

- [ ] **Step 1: Add the shared live-data globals and helpers**

Insert right after the existing `const rawInv    = {};` declaration at `demo/Demo Dashboard.html:2541`:

```js
/* ── Live season (T18) data — shared by Vista general / KPIs / Histórico ── */
const rawLive = { 3: null, 4: null };

async function loadLiveSeason(inv) {
  const rows = await fetchJSON(`/live-predictions/${inv}?season=T18&limit=200`);
  rawLive[inv] = (rows || []).slice().sort((a, b) => a.predicted_for.localeCompare(b.predicted_for));
}

function liveKg(row) {
  return row.kg_actual != null ? row.kg_actual : (row.kg_predicted || 0);
}

function liveTotalsFor(inv) {
  const rows = rawLive[inv] || [];
  const total = rows.reduce((a, r) => a + liveKg(r), 0);
  let peak = null;
  rows.forEach(r => { if (!peak || liveKg(r) > liveKg(peak)) peak = r; });
  return { rows, total, peak };
}

function fmtWeekLabel(dateStr) {
  return new Date(dateStr + 'T12:00:00').toLocaleDateString('es-MX', { month: 'short', day: 'numeric' });
}
```

- [ ] **Step 2: Wire the loader into `init()`**

In `init()` (`demo/Demo Dashboard.html:3014-3021`), add `loadLiveSeason(3)` and `loadLiveSeason(4)` to the existing `Promise.all`. Change:

```js
    const [preds3, preds4, metrics3, metrics4, ph3, ph4] = await Promise.all([
      fetchJSON('/inference/3'),
      fetchJSON('/inference/4'),
      fetchJSON('/metrics/3'),
      fetchJSON('/metrics/4'),
      fetchJSON('/phenology/3'),
      fetchJSON('/phenology/4'),
    ]);
```

to:

```js
    const [preds3, preds4, metrics3, metrics4] = await Promise.all([
      fetchJSON('/inference/3'),
      fetchJSON('/inference/4'),
      fetchJSON('/metrics/3'),
      fetchJSON('/metrics/4'),
      loadLiveSeason(3),
      loadLiveSeason(4),
    ]);
```

(`ph3`/`ph4`/`/phenology/{inv}` fetch removal and replacement is handled in Task 4 — for this task, just drop them from the destructuring/array since `loadLiveSeason` returns `undefined` and writes to `rawLive` as a side effect.)

- [ ] **Step 3: Manual verification**

Run `uvicorn backend.main:app --port 8000 --reload` from repo root, open `http://localhost:8000/app` in a browser, open DevTools console, and run:

```js
rawLive[3]
```

Expected: an array (possibly empty if no T18 rows yet exist), not `null`/`undefined`, with items sorted ascending by `predicted_for` if non-empty.

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add live T18 season data loader"
```

---

### Task 2: Vista general — live totals on menu cards

**Files:**
- Modify: `demo/Demo Dashboard.html` (`fillKPIs` at lines 2505-2538, `init()`, `renderForecastSection`'s PATCH handler at lines 3155-3169)

**Interfaces:**
- Consumes: `rawLive`, `liveTotalsFor(inv)`, `liveKg(row)`, `fmtWeekLabel(dateStr)`, `fmtKg`, `setText` (Task 1).
- Produces: `function fillMenuKPIs(inv)` — sets `menu{inv}-total`, `menu{inv}-peak`, `menu{inv}-peak-unit` from live data. Later tasks call this after any live-data mutation.

- [ ] **Step 1: Extract the menu-card lines out of `fillKPIs` into a new live-sourced function**

In `fillKPIs` (`demo/Demo Dashboard.html:2505-2538`), delete these 3 lines (they currently source Vista general's cards from the T17 cache — `inv{inv}-total`/`inv{inv}-peak`/`inv{inv}-peak-unit`, the Predicción-view equivalents, stay untouched):

```js
  setText(`menu${inv}-total`, fmtKg(total));
  setText(`menu${inv}-peak`,  fmtKg(peak.pred));
  setText(`menu${inv}-peak-unit`, `kg · ${peak.horizon}`);
```

Add a new function right after `fillKPIs` (after line 2538, before `const rawInv    = {};`):

```js
function fillMenuKPIs(inv) {
  const { rows, total, peak } = liveTotalsFor(inv);
  if (!rows.length) {
    setText(`menu${inv}-total`, '—');
    setText(`menu${inv}-peak`, '—');
    setText(`menu${inv}-peak-unit`, 'Sin datos de T18 todavía');
    return;
  }
  setText(`menu${inv}-total`, fmtKg(total));
  setText(`menu${inv}-peak`, fmtKg(liveKg(peak)));
  setText(`menu${inv}-peak-unit`, `kg · ${fmtWeekLabel(peak.predicted_for)}`);
}
```

- [ ] **Step 2: Call `fillMenuKPIs` from `init()` and after registering actual kg**

In `init()`, after the `loadLiveSeason(3)`/`loadLiveSeason(4)` calls resolve (i.e. after the `Promise.all` from Task 1 Step 2), add:

```js
    fillMenuKPIs(3);
    fillMenuKPIs(4);
```

Place this right before the existing `refreshLiveData();` call (`demo/Demo Dashboard.html:3041`).

In the `register-actual-btn` click handler inside `renderForecastSection` (`demo/Demo Dashboard.html:3155-3169`), after the existing `refreshLiveData();` call, add a re-fetch + re-render so totals reflect the newly-registered actual:

```js
      await loadLiveSeason(btn.dataset.inv);
      fillMenuKPIs(Number(btn.dataset.inv));
      refreshLiveData();
```

(replacing the standalone `refreshLiveData();` line at 3167 with the 3 lines above).

- [ ] **Step 3: Manual verification**

Reload `http://localhost:8000/app`. In DevTools console:

```js
fillMenuKPIs(3); fillMenuKPIs(4);
document.getElementById('menu3-total').textContent
```

Expected: matches `rawLive[3]` sum (or `'—'` if `rawLive[3]` is empty). Visually confirm Vista general cards no longer show the old T17 total (compare against previous git stash if unsure).

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): Vista general cards read live T18 totals"
```

---

### Task 3: KPIs & gráficas — hero combo chart on live data

**Files:**
- Modify: `demo/Demo Dashboard.html` (`renderYieldGoal` at lines 2939-2997, `init()` at line 3039)

**Interfaces:**
- Consumes: `rawLive`, `liveKg(row)`, `fmtWeekLabel(dateStr)`, `AREA_M2`, `getMeta(num)` (all pre-existing except `rawLive`/`liveKg` from Task 1).
- Produces: `renderYieldGoal(num)` now reads `rawLive` instead of `rawInv`; same call signature, same DOM targets (`chart-yield-{num}`, `kpi-yield-avg-{num}`, `kpi-yield-trend-{num}`).

- [ ] **Step 1: Rewrite the data-sourcing half of `renderYieldGoal`**

Replace lines 2939-2958 of `demo/Demo Dashboard.html`:

```js
function renderYieldGoal(num) {
  if (!rawInv[3] || !rawInv[4]) return;
  const a3 = AREA_M2[3], a4 = AREA_M2[4];
  const p3 = rawInv[3].y_pred, p4 = rawInv[4].y_pred;
  const t3 = rawInv[3].y_true, t4 = rawInv[4].y_true;
  const weeks = rawInv[3].weeks;
  const n = Math.min(p3.length, p4.length);

  const meta = getMeta(num);
  let cum = 0, cumReal = 0;
  const projWeek = [], projCum = [], metaWeek = [], metaCum = [];
  for (let i = 0; i < n; i++) {
    const w = ((p3[i] / a3) + (p4[i] / a4)) / 2;   // producción proyectada de la semana i (prom. ambos)
    cum     += w;
    cumReal += (((t3[i] || 0) / a3) + ((t4[i] || 0) / a4)) / 2;
    projWeek.push(+w.toFixed(3));
    projCum.push(+cum.toFixed(3));
    metaWeek.push(+meta.toFixed(3));
    metaCum.push(+(meta * (i + 1)).toFixed(3));
  }
  const labels = weeks.slice(0, n).map(w => `S${w.iso_week}`);
```

with:

```js
function renderYieldGoal(num) {
  if (!rawLive[3] || !rawLive[4]) return;
  const a3 = AREA_M2[3], a4 = AREA_M2[4];
  const byDate3 = Object.fromEntries(rawLive[3].map(r => [r.predicted_for, r]));
  const byDate4 = Object.fromEntries(rawLive[4].map(r => [r.predicted_for, r]));
  const weeks = Array.from(new Set([
    ...rawLive[3].map(r => r.predicted_for),
    ...rawLive[4].map(r => r.predicted_for),
  ])).sort();
  const n = weeks.length;

  const meta = getMeta(num);
  let cum = 0, cumReal = 0;
  const projWeek = [], projCum = [], metaWeek = [], metaCum = [];
  for (let i = 0; i < n; i++) {
    const r3 = byDate3[weeks[i]], r4 = byDate4[weeks[i]];
    const w = (((r3 ? liveKg(r3) : 0) / a3) + ((r4 ? liveKg(r4) : 0) / a4)) / 2;
    cum     += w;
    cumReal += (((r3?.kg_actual || 0) / a3) + ((r4?.kg_actual || 0) / a4)) / 2;
    projWeek.push(+w.toFixed(3));
    projCum.push(+cum.toFixed(3));
    metaWeek.push(+meta.toFixed(3));
    metaCum.push(+(meta * (i + 1)).toFixed(3));
  }
  const labels = weeks.map(w => fmtWeekLabel(w));
```

The rest of the function (chart construction, `kpi-yield-avg-{num}`/`kpi-yield-trend-{num}` at lines 2960-2997) is unchanged — it already only references `n`, `labels`, `projWeek`, `projCum`, `metaWeek`, `metaCum`, `cumReal`, all of which keep the same names/shapes.

Note the meta curve stays untouched (still `meta * (i + 1)`), matching the approved design decision to reuse the same agronomic target regardless of season.

- [ ] **Step 2: Reposition the `init()` call**

`renderYieldGoal(3); renderYieldGoal(4);` at `demo/Demo Dashboard.html:3039` already runs after `rawInv` is set; it now depends on `rawLive` instead, which is populated by `loadLiveSeason` in the same `Promise.all` (Task 1 Step 2), so it resolves before this line runs — no reordering needed. Leave the call in place.

- [ ] **Step 3: Manual verification**

Reload `http://localhost:8000/app`, navigate to "KPIs & gráficas" for Invernadero 3. Confirm the hero chart renders without console errors and `kpi-yield-avg-3` shows a live-derived number (or `0.0`/blank chart if `rawLive[3]`/`rawLive[4]` are both empty — not a crash).

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): KPI hero combo chart reads live T18 weekly data"
```

---

### Task 4: KPIs & gráficas — live phenology panel

**Files:**
- Modify: `demo/Demo Dashboard.html` (`rawPheno`/`phenoFields` globals and `renderPhenology` at lines 2593-2627, `init()` at lines 3019-3020/3032-3034)

**Interfaces:**
- Consumes: `sparklineSVG(values)` (pre-existing, `demo/Demo Dashboard.html:2598`), `fetchJSON`.
- Produces: `const LIVE_PHENO_FIELDS` (array of `{key, label, unit}`, 8 entries matching `phenology_observations` columns), `async function loadLivePhenology(inv)`, `function renderLivePhenology(inv)`.

- [ ] **Step 1: Replace `rawPheno`/`phenoFields`/`renderPhenology` with the live equivalents**

Replace lines 2593-2627 of `demo/Demo Dashboard.html`:

```js
const rawPheno = { 3: null, 4: null };
let phenoFields = [];
const SEASON_COLORS = ['#94a3b8', '#a78bfa', '#38bdf8', '#fb923c', '#65a30d'];

/* Sparkline SVG simple a partir de una serie de valores */
function sparklineSVG(values) {
  const v = values.filter(x => x !== null && !isNaN(x));
  if (v.length < 2) return '';
  const w = 120, h = 26, min = Math.min(...v), max = Math.max(...v), rng = (max - min) || 1;
  const pts = v.map((y, i) => {
    const x = (i / (v.length - 1)) * w;
    const yy = h - ((y - min) / rng) * (h - 4) - 2;
    return `${x.toFixed(1)},${yy.toFixed(1)}`;
  }).join(' ');
  return `<svg class="pc-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="${palette.accent}" stroke-width="1.6"/></svg>`;
}

function renderPhenology(inv) {
  const data = rawPheno[inv];
  const grid = document.getElementById(`pheno-grid-${inv}`);
  if (!data || !grid) return;
  const lastSeason = data.seasons[data.seasons.length - 1];
  grid.innerHTML = data.fields.map(f => {
    const val = data.latest[f.key];
    const series = lastSeason ? lastSeason.values[f.key] : [];
    return `
      <div class="pheno-card">
        <div class="pc-label">${f.label}</div>
        <div class="pc-value">${val === null || val === undefined ? '—' : val}<small>${f.unit}</small></div>
        ${sparklineSVG(series || [])}
        <div class="pc-foot">${data.latest_season || ''} · promedio semanal</div>
      </div>`;
  }).join('');
}
```

with:

```js
const rawLivePheno = { 3: null, 4: null };
const SEASON_COLORS = ['#94a3b8', '#a78bfa', '#38bdf8', '#fb923c', '#65a30d'];

/* Sparkline SVG simple a partir de una serie de valores */
function sparklineSVG(values) {
  const v = values.filter(x => x !== null && !isNaN(x));
  if (v.length < 2) return '';
  const w = 120, h = 26, min = Math.min(...v), max = Math.max(...v), rng = (max - min) || 1;
  const pts = v.map((y, i) => {
    const x = (i / (v.length - 1)) * w;
    const yy = h - ((y - min) / rng) * (h - 4) - 2;
    return `${x.toFixed(1)},${yy.toFixed(1)}`;
  }).join(' ');
  return `<svg class="pc-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="${palette.accent}" stroke-width="1.6"/></svg>`;
}

/* Mapea columnas de phenology_observations (Supabase, T18) → label/unit para mostrar */
const LIVE_PHENO_FIELDS = [
  { key: 'racimos_puestos',          label: 'Racimos puestos',           unit: 'conteo' },
  { key: 'flores_racimo_abiertas',   label: 'Flores en racimo abiertas', unit: 'conteo' },
  { key: 'racimos_en_planta',        label: 'Racimos en planta',         unit: 'conteo' },
  { key: 'cantidad_tomates',         label: 'Cantidad de tomates',       unit: 'conteo' },
  { key: 'racimo_en_cosecha',        label: 'Nº de racimo en cosecha',   unit: 'índice' },
  { key: 'tomates_maduros',          label: 'Tomates maduros (color 2)', unit: 'conteo' },
  { key: 'diametro_fruto_cm',        label: 'Diámetro del fruto',        unit: 'cm' },
  { key: 'crecimiento_planta_cm',    label: 'Crecimiento de planta',     unit: 'cm' },
];

async function loadLivePhenology(inv) {
  const rows = await fetchJSON(`/phenology-live/${inv}?limit=500`);
  rawLivePheno[inv] = rows || [];
}

/* Agrupa observaciones por week_date y promedia cada variable (sobre zona/planta) */
function livePhenoWeeklyAvg(inv) {
  const rows = rawLivePheno[inv] || [];
  const byWeek = {};
  rows.forEach(r => { (byWeek[r.week_date] ||= []).push(r); });
  const weeks = Object.keys(byWeek).sort();
  const series = {};
  LIVE_PHENO_FIELDS.forEach(f => {
    series[f.key] = weeks.map(w => {
      const vals = byWeek[w].map(r => r[f.key]).filter(v => v != null);
      return vals.length ? vals.reduce((a, v) => a + v, 0) / vals.length : null;
    });
  });
  return { weeks, series };
}

function renderLivePhenology(inv) {
  const grid = document.getElementById(`pheno-grid-${inv}`);
  if (!grid) return;
  const { weeks, series } = livePhenoWeeklyAvg(inv);
  if (!weeks.length) {
    grid.innerHTML = '<p style="color:#9aa0a8">Sin fenología de T18 todavía.</p>';
    return;
  }
  grid.innerHTML = LIVE_PHENO_FIELDS.map(f => {
    const vals = series[f.key];
    const last = [...vals].reverse().find(v => v != null);
    return `
      <div class="pheno-card">
        <div class="pc-label">${f.label}</div>
        <div class="pc-value">${last == null ? '—' : last.toFixed(1)}<small>${f.unit}</small></div>
        ${sparklineSVG(vals)}
        <div class="pc-foot">T18 · promedio semanal</div>
      </div>`;
  }).join('');
}
```

- [ ] **Step 2: Wire the new loader/renderer into `init()`**

Complete the `Promise.all` edit started in Task 1 Step 2 — the destructuring should now be:

```js
    const [preds3, preds4, metrics3, metrics4] = await Promise.all([
      fetchJSON('/inference/3'),
      fetchJSON('/inference/4'),
      fetchJSON('/metrics/3'),
      fetchJSON('/metrics/4'),
      loadLiveSeason(3),
      loadLiveSeason(4),
      loadLivePhenology(3),
      loadLivePhenology(4),
    ]);
```

Replace (`demo/Demo Dashboard.html:3032-3034`):

```js
    rawPheno[3] = ph3; rawPheno[4] = ph4;
    phenoFields = ph3.fields || [];
    renderPhenology(3); renderPhenology(4);
```

with:

```js
    renderLivePhenology(3); renderLivePhenology(4);
```

- [ ] **Step 3: Manual verification**

Reload `http://localhost:8000/app`, navigate to "KPIs & gráficas" for Invernadero 3, confirm the "Fenología" panel shows either 8 cards with live values/sparklines, or the `Sin fenología de T18 todavía.` empty state — no console errors, no reference to `rawPheno`/`phenoFields` left (`grep -n "rawPheno\|phenoFields" "demo/Demo Dashboard.html"` returns nothing).

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): KPI phenology panel reads live T18 Supabase data"
```

---

### Task 5: Histórico — live weekly table

**Files:**
- Modify: `demo/Demo Dashboard.html` (`renderInv` at lines 2553-2569, near `buildHistTable` at lines 2137-2163, `init()`)

**Interfaces:**
- Consumes: `rawLive`, `errorBucket(err)`, `fmt(n)`, `fmtWeekLabel(dateStr)` (pre-existing / Task 1).
- Produces: `function buildLiveHistTable(tbody, rows)`.

- [ ] **Step 1: Decouple Histórico's tbody from the T17 slider**

In `renderInv(inv, cursor)` (`demo/Demo Dashboard.html:2553-2569`), delete this line (it currently repaints Histórico every time the Predicción slider moves, with T17 data):

```js
  buildHistTable(document.getElementById(`tbody-hist${inv}`), s.fullHistoryTable);
```

`renderInv` keeps populating `tbody-inv{inv}` (Predicción's own 6-week table, via `buildPredTable`) — only the `tbody-hist{inv}` line is removed.

- [ ] **Step 2: Add `buildLiveHistTable`**

Add a new function right after `buildHistTable` (after line 2163 of `demo/Demo Dashboard.html`). The table has 6 columns matching `view-hist3`/`view-hist4`'s `<thead>` (`Semana, Estatus, Real (kg), Predicho (kg), Intervalo, Error`) — live rows have no prediction interval (not computed by `live_inference.py`), so the Intervalo column always shows `—`:

```js
function buildLiveHistTable(tbody, rows) {
  if (!tbody) return;
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--ink-muted)">Sin datos de T18 todavía.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const isObs = r.kg_actual != null;
    const real = r.kg_actual;
    const pred = r.kg_predicted;
    let errHtml = '<td class="num"><span style="color:var(--ink-muted);opacity:0.4">—</span></td>';
    if (isObs && pred) {
      const err = ((pred - real) / real) * 100;
      const errKg = pred - real;
      const cls = errorBucket(err);
      const sign = err >= 0 ? '+' : '';
      const signKg = errKg >= 0 ? '+' : '';
      errHtml = `<td class="num">
        <span class="error-cell ${cls}">${sign}${err.toFixed(1)}%</span>
        <div style="font-size:11px;color:var(--ink-muted);margin-top:3px;font-variant-numeric:tabular-nums;">${signKg}${Math.round(errKg).toLocaleString('en-US')} kg</div>
      </td>`;
    }
    return `
      <tr class="${isObs ? 'observed-row' : 'future'}">
        <td class="label">${fmtWeekLabel(r.predicted_for)}</td>
        <td class="status ${isObs ? 'observed' : ''}">${isObs ? 'Observado' : 'Predicho'}</td>
        <td class="num">${fmt(real)}</td>
        <td class="num">${fmt(pred)}</td>
        <td class="num">—</td>
        ${errHtml}
      </tr>`;
  }).join('');
}
```

- [ ] **Step 3: Call it from `init()` and after registering actual kg**

In `init()`, right after the `fillMenuKPIs(3); fillMenuKPIs(4);` lines added in Task 2 Step 2, add:

```js
    buildLiveHistTable(document.getElementById('tbody-hist3'), rawLive[3] || []);
    buildLiveHistTable(document.getElementById('tbody-hist4'), rawLive[4] || []);
```

In the `register-actual-btn` handler (`demo/Demo Dashboard.html`, edited in Task 2 Step 2), after `fillMenuKPIs(Number(btn.dataset.inv));`, add:

```js
      buildLiveHistTable(document.getElementById(`tbody-hist${btn.dataset.inv}`), rawLive[btn.dataset.inv] || []);
```

- [ ] **Step 4: Manual verification**

Reload `http://localhost:8000/app`, navigate to "Histórico" for Invernadero 3. Confirm the table shows live T18 rows (or the empty-state row if `rawLive[3]` is empty) and does NOT change when moving the Predicción slider on `view-inv3` (open that view first, move the slider, then revisit Histórico — table must be unaffected by the slider).

- [ ] **Step 5: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): Histórico table reads live T18 predictions independent of T17 slider"
```

---

### Task 6: Delete all mock KPI panels

**Files:**
- Modify: `demo/Demo Dashboard.html` — HTML in `view-kpi3` (lines 1589-1640) and `view-kpi4` (lines 1685-1736), CSS (lines 831, 888-909 region), JS (lines 2325-2492 region)

**Interfaces:** None — pure deletion, no new interfaces. The one real card (`kpi-yield-avg-{num}`/`kpi-yield-trend-{num}`, already wired by `renderYieldGoal` in Task 3) is preserved.

- [ ] **Step 1: Delete the 3 mock `.kpi-stat` cards, keeping the real one**

In `view-kpi3`'s `.kpi-stat-grid` (`demo/Demo Dashboard.html:1589-1614`), delete the 3 mock cards (Eficiencia hídrica, Precisión modelo, Mortalidad), keeping the grid wrapper and the first (real) card:

```html
    <div class="kpi-stat-grid">
      <div class="kpi-stat">
        <div class="kpi-stat-label">Rendimiento · prom. ambos inv</div>
        <div class="kpi-stat-value" id="kpi-yield-avg-3">—</div>
        <div class="kpi-stat-unit">kg/m² acumulado T17</div>
        <div class="kpi-stat-trend trend-flat" id="kpi-yield-trend-3">— real vs proyectado</div>
      </div>
    </div>
```

Repeat identically for `view-kpi4`'s `.kpi-stat-grid` (`demo/Demo Dashboard.html:1685-1710` region), swapping `-3` suffixes for `-4`.

- [ ] **Step 2: Delete the mock chart panels and their wrapping grids**

In `view-kpi3`, delete the entire `.charts-grid` block (`demo/Demo Dashboard.html:1616-1627`, contains `chart-kpi3-prod` and `chart-kpi3-cat`) and the entire `.charts-grid-2` block (lines 1629-1640, contains `chart-kpi3-err` and `chart-kpi3-res`) — both grids only ever held mock panels, so remove the wrapping `<div class="charts-grid">...</div>` / `<div class="charts-grid-2">...</div>` entirely, not just their children.

Repeat identically for `view-kpi4`'s equivalent blocks (`chart-kpi4-prod`, `chart-kpi4-cat`, `chart-kpi4-err`, `chart-kpi4-res`).

- [ ] **Step 3: Delete the now-dead chart factory functions and their invocations**

Delete `makeProdChart`, `makeCatChart`, `makeErrChart`, `makeResChart`, and `tooltipOpts` function definitions (`demo/Demo Dashboard.html:2325-2472` region — run `grep -n "^function makeProdChart\|^function makeCatChart\|^function makeErrChart\|^function makeResChart\|^function tooltipOpts"` first to get exact current line numbers, since earlier edits in this plan may have shifted them) — verify via `grep -n "tooltipOpts\|makeProdChart\|makeCatChart\|makeErrChart\|makeResChart"` that these have no other callers before deleting (per the exploration done during planning, they don't).

Delete the 8 invocation lines calling them with hardcoded mock arrays (`demo/Demo Dashboard.html:2475-2478` and `2484-2487` region):

```js
makeProdChart('chart-kpi3-prod', [42100, 45800, 47200, 46500, 48900, 47500, 49200, 51000, 46800, 48000, 50100, 52400]);
makeCatChart('chart-kpi3-cat', [58, 32, 10]);
makeErrChart('chart-kpi3-err', [6.2, 4.8, 3.9, 5.4, 4.0, 3.2, 4.1, 3.5]);
makeResChart('chart-kpi3-res', ...);
makeProdChart('chart-kpi4-prod', [40200, 43500, 44900, 44100, 46800, 45200, 46500, 49800, 44100, 45500, 47800, 49200]);
makeCatChart('chart-kpi4-cat', [48, 38, 14]);
makeErrChart('chart-kpi4-err', [7.1, 6.4, 5.8, 6.9, 5.2, 7.4, 6.1, 8.2]);
makeResChart('chart-kpi4-res', ...);
```

- [ ] **Step 4: Delete the now-unused CSS**

Delete the `.charts-grid`, `.charts-grid-2`, `.panel.chart-panel`, `.chart-panel .panel-title`, `.chart-md`, `.chart-sm` rules (`demo/Demo Dashboard.html:888-909` region) — confirm via `grep -n "charts-grid\|chart-panel\|chart-md\|chart-sm"` that no HTML references remain after Steps 1-2 before deleting. Keep `.kpi-stat-grid`/`.kpi-stat` CSS (line 831 region) — still used by the one real card.

- [ ] **Step 5: Manual verification**

Reload `http://localhost:8000/app`, navigate to "KPIs & gráficas" for both invernaderos. Confirm: only the hero combo chart, the single "Rendimiento · prom. ambos inv" stat card, and the fenología panel remain — no broken layout, no console errors about missing canvas elements. Run `grep -n "chart-kpi3-prod\|chart-kpi3-cat\|chart-kpi3-err\|chart-kpi3-res\|Eficiencia hídrica\|Precisión modelo\|Mortalidad" "demo/Demo Dashboard.html"` and confirm no matches.

- [ ] **Step 6: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "refactor(dashboard): remove mock KPI panels (calibre/error/recursos/producción/eficiencia/precisión/mortalidad)"
```
