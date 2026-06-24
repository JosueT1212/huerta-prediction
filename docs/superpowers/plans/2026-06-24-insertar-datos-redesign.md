# Insertar Datos Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the "Insertar datos" view into two tabs — Fenología (wired to `/phenology-live/{inv}`) and Producción real (wired to `PATCH /live-predictions/{inv}/{id}`) — each with a confirmation step before any DB write.

**Architecture:** Frontend-only changes to `demo/Demo Dashboard.html`. Replace the single-form panel with a shared inv selector + tab bar + two panels. No new backend routes. Auth via `localStorage.getItem('sb_token')` (existing pattern).

**Tech Stack:** Vanilla JS, HTML, CSS — no build step. Chart.js already loaded. All fetch calls use `Authorization: Bearer <token>` header.

## Global Constraints

- File to modify: `demo/Demo Dashboard.html` only
- Auth token: `localStorage.getItem('sb_token')` — used in every fetch
- API base: `API_BASE` (already defined in the file as same-origin or `http://localhost:8000`)
- Key mismatch: `phenoFields` (from `/phenology/{inv}`) uses short keys (`flores_abiertas`, `racimos_planta`, `racimo_cosecha`, `diametro_fruto`, `crecimiento_planta`); `/phenology-live/{inv}` expects longer names (`flores_racimo_abiertas`, `racimos_en_planta`, `racimo_en_cosecha`, `diametro_fruto_cm`, `crecimiento_planta_cm`). Map at send time.
- Confirmation: show summary panel; require explicit "Confirmar" click before any POST/PATCH fires
- No tests (vanilla HTML file) — verify manually in browser after each task

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `demo/Demo Dashboard.html` lines 1277–1320 | Replace HTML of `view-insert` section |
| Modify | `demo/Demo Dashboard.html` ~1173 (`<style>`) | Add CSS for tabs + confirmation panel |
| Modify | `demo/Demo Dashboard.html` ~2961–3036 (insertar JS block) | Update JS functions for fenología tab |
| Modify | `demo/Demo Dashboard.html` ~3125–3145 (`init()`) | Add `loadProdPredictions` + `initInsTabs` calls |

---

### Task 1: Replace `view-insert` HTML with tabs layout

**Files:**
- Modify: `demo/Demo Dashboard.html` lines 1277–1320

**Interfaces:**
- Produces: DOM IDs consumed by Tasks 2 & 3:
  - `ins-inv` (select, shared)
  - `ins-date` (date input, replaces `ins-week`)
  - `ins-add`, `ins-thead`, `ins-tbody`, `ins-tfoot`, `ins-hint`, `ins-submit`, `ins-result` (fenología panel)
  - `ins-confirm`, `ins-confirm-summary` (fenología confirmation)
  - `ins-panel-pheno`, `ins-panel-prod` (tab panels)
  - `prod-week-select`, `prod-kg`, `prod-submit-btn`, `prod-result` (producción real panel)
  - `prod-confirm`, `prod-confirm-summary` (producción real confirmation)
  - `.ins-tab` buttons with `data-tab="pheno"` and `data-tab="prod"`

- [ ] **Step 1: Replace the `view-insert` section HTML**

In `demo/Demo Dashboard.html`, find the block from line 1277 to line 1320 (inclusive). Replace it with:

```html
  <!-- ============ INSERTAR DATOS VIEW ============ -->
  <section class="view" id="view-insert">
    <div class="detail-header">
      <div>
        <div class="breadcrumb">Menú · <span>Insertar datos</span></div>
        <h1 class="page-title">Insertar <em>datos</em></h1>
        <div class="ornament">Fenología por planta · Producción real semanal</div>
      </div>
      <div class="page-subtitle">
        <label class="insert-field" style="flex-direction:row;align-items:center;gap:8px">
          <span>Invernadero</span>
          <select id="ins-inv">
            <option value="3">Invernadero 3</option>
            <option value="4">Invernadero 4</option>
          </select>
        </label>
      </div>
    </div>
    <div class="divider"></div>

    <div class="ins-tabs">
      <button class="ins-tab active" data-tab="pheno">Fenología</button>
      <button class="ins-tab" data-tab="prod">Producción real</button>
    </div>

    <!-- ── Fenología panel ── -->
    <div class="panel insert-panel ins-panel" id="ins-panel-pheno">
      <div class="insert-toolbar">
        <label class="insert-field">
          <span>Fecha de observación</span>
          <input id="ins-date" type="date" />
        </label>
        <div class="insert-spacer"></div>
        <button class="btn-ghost" id="ins-add">+ Agregar planta</button>
      </div>

      <div class="insert-table-wrap">
        <table class="insert-table" id="ins-table">
          <thead id="ins-thead"></thead>
          <tbody id="ins-tbody"></tbody>
          <tfoot id="ins-tfoot"></tfoot>
        </table>
      </div>

      <div class="insert-actions">
        <div class="insert-hint" id="ins-hint">Agrega plantas y captura sus mediciones. El promedio se calcula automáticamente.</div>
        <button class="btn-primary" id="ins-submit">Enviar captura</button>
      </div>

      <div class="ins-confirm-panel" id="ins-confirm" hidden>
        <pre class="ins-confirm-summary" id="ins-confirm-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="ins-cancel">Cancelar</button>
          <button class="btn-primary" id="ins-confirm-btn">Confirmar envío</button>
        </div>
      </div>
      <div class="insert-result" id="ins-result"></div>
    </div>

    <!-- ── Producción real panel ── -->
    <div class="panel insert-panel ins-panel" id="ins-panel-prod" hidden>
      <div class="insert-toolbar" style="flex-wrap:wrap;gap:16px">
        <label class="insert-field">
          <span>Semana (predicción)</span>
          <select id="prod-week-select" style="min-width:280px">
            <option disabled selected>Cargando…</option>
          </select>
        </label>
        <label class="insert-field">
          <span>kg reales</span>
          <input id="prod-kg" type="number" min="0" step="1" placeholder="ej. 38000" style="width:140px" />
        </label>
        <div class="insert-spacer"></div>
        <button class="btn-primary" id="prod-submit-btn">Registrar</button>
      </div>

      <div class="ins-confirm-panel" id="prod-confirm" hidden>
        <pre class="ins-confirm-summary" id="prod-confirm-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="prod-cancel">Cancelar</button>
          <button class="btn-primary" id="prod-confirm-btn">Confirmar</button>
        </div>
      </div>
      <div class="insert-result" id="prod-result"></div>
    </div>
  </section>
```

- [ ] **Step 2: Add CSS for tabs and confirmation panel**

Find the closing `</style>` tag (line ~1173). Insert before it:

```css
  /* ── Insertar datos tabs ── */
  .ins-tabs {
    display: flex;
    gap: 2px;
    margin: 0 0 -1px 0;
    padding: 0 var(--pad, 24px);
  }
  .ins-tab {
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    background: transparent;
    color: var(--ink-muted);
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .ins-tab.active {
    background: var(--surface, #fff);
    color: var(--ink, #1e2a3a);
    border-color: var(--border, #d2d6db);
  }
  .ins-tab:not(.active):hover { background: rgba(0,0,0,.04); }
  .ins-panel { border-top-left-radius: 0; }
  .ins-confirm-panel {
    margin-top: 16px;
    padding: 16px;
    background: var(--surface-alt, #f7f5f0);
    border: 1px solid var(--border, #d2d6db);
    border-radius: 8px;
  }
  .ins-confirm-summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    margin: 0 0 12px 0;
    color: var(--ink, #1e2a3a);
  }
  .ins-confirm-actions { display: flex; gap: 10px; justify-content: flex-end; }
```

- [ ] **Step 3: Verify HTML renders correctly**

Start backend: `uvicorn backend.main:app --port 8000 --reload`
Open `http://localhost:8000/` → login → navigate to "Insertar datos".
Expected: tab bar visible with "Fenología" active, "Producción real" tab clickable. No JS errors in console.

- [ ] **Step 4: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add tabs layout to insertar datos view"
```

---

### Task 2: Fenología tab — JS update

**Files:**
- Modify: `demo/Demo Dashboard.html` (~2961–3036, insertar JS block)

**Interfaces:**
- Consumes: `ins-inv`, `ins-date`, `ins-tbody`, `ins-thead`, `ins-tfoot`, `ins-hint`, `ins-submit`, `ins-result`, `ins-confirm`, `ins-confirm-summary`, `ins-cancel`, `ins-confirm-btn` (Task 1)
- Consumes: `phenoFields` global (populated in `init()` from `/phenology/{inv}`)
- Consumes: `API_BASE`, `localStorage.getItem('sb_token')`
- Produces: working fenología form wired to `/phenology-live/{inv}` with confirmation step

**Key mismatch mapping** (form key → live endpoint field name):
```
flores_abiertas     → flores_racimo_abiertas
racimos_planta      → racimos_en_planta
racimo_cosecha      → racimo_en_cosecha
diametro_fruto      → diametro_fruto_cm
crecimiento_planta  → crecimiento_planta_cm
(all others: same key)
```

- [ ] **Step 1: Replace `insBuildHeader` to add Zona and Planta columns**

Find this function (starts ~line 2961):
```js
function insBuildHeader() {
  const thead = document.getElementById('ins-thead');
  thead.innerHTML = '<tr><th>Planta</th>' +
    phenoFields.map(f => `<th>${f.label}<span class="th-unit">${f.unit}</span></th>`).join('') +
    '<th></th></tr>';
}
```

Replace with:
```js
function insBuildHeader() {
  const thead = document.getElementById('ins-thead');
  thead.innerHTML = '<tr><th>#</th><th>Zona</th><th>Planta</th>' +
    phenoFields.map(f => `<th>${f.label}<span class="th-unit">${f.unit}</span></th>`).join('') +
    '<th></th></tr>';
}
```

- [ ] **Step 2: Replace `insAddRow` to include zona and planta selects**

Find `insAddRow` (~line 2968). Replace the entire function:
```js
function insAddRow() {
  const tbody = document.getElementById('ins-tbody');
  const id = ++insRowSeq;
  const tr = document.createElement('tr');
  tr.dataset.row = id;
  const zonaOpts  = [1,2,3,4].map(n => `<option value="${n}">${n}</option>`).join('');
  const plantaOpts = [1,2,3,4].map(n => `<option value="${n}">${n}</option>`).join('');
  tr.innerHTML =
    `<td>#${tbody.children.length + 1}</td>` +
    `<td><select data-key="zona" style="width:52px">${zonaOpts}</select></td>` +
    `<td><select data-key="planta" style="width:52px">${plantaOpts}</select></td>` +
    phenoFields.map(f =>
      `<td><input type="number" data-key="${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" placeholder="—"></td>`
    ).join('') +
    `<td><span class="row-del" title="Quitar">✕</span></td>`;
  tbody.appendChild(tr);
  tr.querySelectorAll('input').forEach(inp => inp.addEventListener('input', insRecompute));
  tr.querySelector('.row-del').addEventListener('click', () => { tr.remove(); insRenumber(); insRecompute(); });
  insRenumber();
  insRecompute();
}
```

- [ ] **Step 3: Replace `insSubmit` with confirmation-showing version**

Find `async function insSubmit()` (~line 3003). Replace entire function:
```js
function insSubmit() {
  const inv     = +document.getElementById('ins-inv').value;
  const dateVal = document.getElementById('ins-date').value;
  const result  = document.getElementById('ins-result');
  result.className = 'insert-result'; result.textContent = '';

  if (!dateVal) {
    result.className = 'insert-result bad';
    result.textContent = 'Selecciona una fecha de observación.';
    return;
  }

  const rows = [];
  document.querySelectorAll('#ins-tbody tr').forEach(tr => {
    const zona   = +tr.querySelector('[data-key="zona"]').value;
    const planta = +tr.querySelector('[data-key="planta"]').value;
    const pheno  = {};
    let hasData  = false;
    tr.querySelectorAll('input[data-key]').forEach(inp => {
      if (inp.value !== '') { pheno[inp.dataset.key] = parseFloat(inp.value); hasData = true; }
    });
    if (hasData) rows.push({ zona, planta, ...pheno });
  });

  if (!rows.length) {
    result.className = 'insert-result bad';
    result.textContent = 'Captura al menos una planta con datos.';
    return;
  }

  const summary = `Invernadero ${inv} · ${dateVal}\nPlantas: ${rows.length}\n\n` +
    rows.map(r =>
      `  Z${r.zona}·P${r.planta}: ` +
      phenoFields.filter(f => r[f.key] != null).map(f => `${f.label.split(' ')[0]}=${r[f.key]}`).join(', ')
    ).join('\n');

  document.getElementById('ins-confirm-summary').textContent = summary;
  document.getElementById('ins-confirm').hidden = false;
  document.getElementById('ins-submit').disabled = true;

  const confirmEl = document.getElementById('ins-confirm');
  confirmEl.dataset.inv   = inv;
  confirmEl.dataset.date  = dateVal;
  confirmEl.dataset.rows  = JSON.stringify(rows);
}
```

- [ ] **Step 4: Add `insConfirm` function (after `insSubmit`)**

Insert the following new function immediately after `insSubmit`:

```js
const PHENO_KEY_TO_LIVE = {
  flores_abiertas:     'flores_racimo_abiertas',
  racimos_planta:      'racimos_en_planta',
  racimo_cosecha:      'racimo_en_cosecha',
  diametro_fruto:      'diametro_fruto_cm',
  crecimiento_planta:  'crecimiento_planta_cm',
};

async function insConfirm() {
  const confirmEl = document.getElementById('ins-confirm');
  const inv       = +confirmEl.dataset.inv;
  const weekDate  = confirmEl.dataset.date;
  const rows      = JSON.parse(confirmEl.dataset.rows);
  const result    = document.getElementById('ins-result');
  const token     = localStorage.getItem('sb_token');

  confirmEl.hidden = true;
  document.getElementById('ins-submit').disabled = false;

  try {
    for (const row of rows) {
      const body = { week_date: weekDate, zona: row.zona, planta: row.planta };
      phenoFields.forEach(f => {
        if (row[f.key] != null) {
          const liveKey = PHENO_KEY_TO_LIVE[f.key] ?? f.key;
          body[liveKey] = row[f.key];
        }
      });
      const r = await fetch(`${API_BASE}/phenology-live/${inv}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.detail || `Error ${r.status}`);
      }
    }
    result.className = 'insert-result ok';
    result.textContent = `✓ ${rows.length} planta(s) guardadas · ${weekDate}`;
    document.getElementById('ins-tbody').innerHTML = '';
    insRowSeq = 0;
    insAddRow(); insAddRow();
  } catch (e) {
    result.className = 'insert-result bad';
    result.textContent = '⚠ ' + e.message;
  }
}
```

- [ ] **Step 5: Update `buildInsertForm` to wire new confirm/cancel buttons**

Find `buildInsertForm` (~line 3029):
```js
function buildInsertForm() {
  if (!phenoFields.length) return;
  insBuildHeader();
  document.getElementById('ins-tbody').innerHTML = '';
  document.getElementById('ins-add').addEventListener('click', insAddRow);
  document.getElementById('ins-submit').addEventListener('click', insSubmit);
  insAddRow(); insAddRow();   // arranca con 2 filas
}
```

Replace with:
```js
function buildInsertForm() {
  if (!phenoFields.length) return;
  insBuildHeader();
  document.getElementById('ins-tbody').innerHTML = '';
  document.getElementById('ins-add').addEventListener('click', insAddRow);
  document.getElementById('ins-submit').addEventListener('click', insSubmit);
  document.getElementById('ins-confirm-btn').addEventListener('click', insConfirm);
  document.getElementById('ins-cancel').addEventListener('click', () => {
    document.getElementById('ins-confirm').hidden = true;
    document.getElementById('ins-submit').disabled = false;
  });
  insAddRow(); insAddRow();
}
```

- [ ] **Step 6: Add `initInsTabs` function (after `buildInsertForm`)**

Insert after `buildInsertForm`:
```js
function initInsTabs() {
  document.querySelectorAll('.ins-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ins-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.getElementById('ins-panel-pheno').hidden = tab !== 'pheno';
      document.getElementById('ins-panel-prod').hidden  = tab !== 'prod';
      if (tab === 'prod') loadProdPredictions(+document.getElementById('ins-inv').value);
    });
  });
  document.getElementById('ins-inv').addEventListener('change', () => {
    const activeTab = document.querySelector('.ins-tab.active')?.dataset.tab;
    if (activeTab === 'prod') loadProdPredictions(+document.getElementById('ins-inv').value);
  });
}
```

- [ ] **Step 7: Verify fenología tab in browser**

Navigate to "Insertar datos" → Fenología tab.
- Table header shows: `# | Zona | Planta | Racimos puestos | …`
- Each row has Zona (1–4) and Planta (1–4) dropdowns
- Leave date empty → click "Enviar captura" → error "Selecciona una fecha"
- Fill date + at least one plant row → click "Enviar captura" → confirmation panel appears with summary
- Click "Cancelar" → panel hides, button re-enables
- No console errors

- [ ] **Step 8: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): wire fenología tab to /phenology-live with confirmation"
```

---

### Task 3: Producción real tab — JS + wire into init()

**Files:**
- Modify: `demo/Demo Dashboard.html` (~3125–3145, `init()` async function)

**Interfaces:**
- Consumes: `ins-inv`, `prod-week-select`, `prod-kg`, `prod-submit-btn`, `prod-result`, `prod-confirm`, `prod-confirm-summary`, `prod-cancel`, `prod-confirm-btn` (Task 1)
- Consumes: `API_BASE`, `localStorage.getItem('sb_token')`
- Produces: working producción real form wired to `GET /live-predictions/{inv}` + `PATCH /live-predictions/{inv}/{id}`

- [ ] **Step 1: Add `loadProdPredictions` function (after `initInsTabs`)**

```js
async function loadProdPredictions(inv) {
  const select = document.getElementById('prod-week-select');
  select.innerHTML = '<option disabled selected>Cargando…</option>';
  try {
    const token = localStorage.getItem('sb_token');
    const preds = await fetch(`${API_BASE}/live-predictions/${inv}`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

    if (!preds.length) {
      select.innerHTML = '<option disabled selected>Sin predicciones disponibles</option>';
      return;
    }
    select.innerHTML = preds.map(p =>
      `<option value="${p.id}" data-kg-pred="${p.kg_predicted}">` +
      `${p.predicted_for} — pred: ${Math.round(p.kg_predicted).toLocaleString('es-MX')} kg` +
      (p.kg_actual != null ? ` ✓ real: ${Math.round(p.kg_actual).toLocaleString('es-MX')} kg` : '') +
      `</option>`
    ).join('');
  } catch (e) {
    select.innerHTML = `<option disabled selected>Error al cargar (${e.message})</option>`;
  }
}
```

- [ ] **Step 2: Add `prodSubmit` function (after `loadProdPredictions`)**

```js
function prodSubmit() {
  const inv    = +document.getElementById('ins-inv').value;
  const select = document.getElementById('prod-week-select');
  const kgInput = document.getElementById('prod-kg');
  const result  = document.getElementById('prod-result');
  result.className = 'insert-result'; result.textContent = '';

  const predId  = select.value;
  const kgVal   = parseFloat(kgInput.value);
  const opt     = select.options[select.selectedIndex];
  const predDate = opt ? opt.text.split(' — ')[0] : '?';
  const kgPred   = parseFloat(opt?.dataset.kgPred || 0);

  if (!predId || select.selectedIndex < 0 || opt?.disabled) {
    result.className = 'insert-result bad'; result.textContent = 'Selecciona una semana.'; return;
  }
  if (isNaN(kgVal) || kgVal <= 0) {
    result.className = 'insert-result bad'; result.textContent = 'Ingresa kg reales válidos (> 0).'; return;
  }

  document.getElementById('prod-confirm-summary').textContent =
    `Invernadero ${inv} · Semana ${predDate}\n` +
    `Registrar kg reales: ${Math.round(kgVal).toLocaleString('es-MX')} kg\n` +
    `(predicción fue: ${Math.round(kgPred).toLocaleString('es-MX')} kg)`;
  document.getElementById('prod-confirm').hidden = false;
  document.getElementById('prod-submit-btn').disabled = true;

  const confirmEl = document.getElementById('prod-confirm');
  confirmEl.dataset.inv    = inv;
  confirmEl.dataset.predId = predId;
  confirmEl.dataset.kg     = kgVal;
}
```

- [ ] **Step 3: Add `prodConfirm` function (after `prodSubmit`)**

```js
async function prodConfirm() {
  const confirmEl = document.getElementById('prod-confirm');
  const inv    = +confirmEl.dataset.inv;
  const predId = +confirmEl.dataset.predId;
  const kg     = parseFloat(confirmEl.dataset.kg);
  const result = document.getElementById('prod-result');
  const token  = localStorage.getItem('sb_token');

  confirmEl.hidden = true;
  document.getElementById('prod-submit-btn').disabled = false;

  try {
    const r = await fetch(`${API_BASE}/live-predictions/${inv}/${predId}`, {
      method: 'PATCH',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ kg_actual: kg }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `Error ${r.status}`);
    }
    result.className = 'insert-result ok';
    result.textContent = `✓ kg reales registrados · ${Math.round(kg).toLocaleString('es-MX')} kg`;
    document.getElementById('prod-kg').value = '';
    await loadProdPredictions(inv);
  } catch (e) {
    result.className = 'insert-result bad';
    result.textContent = '⚠ ' + e.message;
  }
}
```

- [ ] **Step 4: Wire prod tab buttons**

In `init()`, find the call to `buildInsertForm()` (~line 3144). Add three lines after it:

```js
    buildInsertForm();
    initInsTabs();
    document.getElementById('prod-submit-btn').addEventListener('click', prodSubmit);
    document.getElementById('prod-confirm-btn').addEventListener('click', prodConfirm);
    document.getElementById('prod-cancel').addEventListener('click', () => {
      document.getElementById('prod-confirm').hidden = true;
      document.getElementById('prod-submit-btn').disabled = false;
    });
```

- [ ] **Step 5: Verify producción real tab end-to-end**

Navigate to "Insertar datos" → click "Producción real" tab.
Expected:
- Dropdown loads weeks from `/live-predictions/3` (e.g. `2026-07-27 — pred: 34,895 kg`)
- Select a week, enter kg (e.g. 37000), click "Registrar" → confirmation panel shows with correct summary
- Click "Cancelar" → panel hides, button re-enables
- Click "Confirmar" → `PATCH /live-predictions/3/{id}` fires → success message → dropdown reloads with ✓ next to updated week
- Change inv to 4 → dropdown reloads for inv4
- Switch back to Fenología tab → fenología form still works

Check: no 401 errors (token is present), no 422 errors (kg > 0).

- [ ] **Step 6: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add producción real tab with kg_actual PATCH + confirmation"
```
