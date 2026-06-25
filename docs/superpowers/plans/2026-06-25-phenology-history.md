# Phenology History & kg_reales Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Historial" tab to "Insertar datos" showing all inserted phenology observations with per-row delete, and add a "Borrar real" action on the Producción real tab for wrongly entered kg_actual.

**Architecture:** Task 1 adds two backend DELETE endpoints. Task 2 adds the Historial tab (HTML + JS). Task 3 extends the Producción real tab with a delete list for confirmed kg_actual rows.

**Tech Stack:** FastAPI (Python), Supabase-py, vanilla JS, single-file SPA (`demo/Demo Dashboard.html`)

## Global Constraints

- Auth: `localStorage.getItem('sb_token')` — `Authorization: Bearer <token>` header on every fetch
- Spanish copy unchanged — all labels in Spanish
- Confirm panel pattern: same `ins-confirm-panel` / `ins-confirm-summary` / `ins-confirm-actions` CSS classes already in file
- `API_BASE` constant already defined in dashboard — all fetch paths relative to it
- No new backend dependencies — existing `service_client`, `get_current_user`, FastAPI patterns
- Single dashboard file: `demo/Demo Dashboard.html`

---

## File Map

| File | Task | Change |
|------|------|--------|
| `backend/routers/phenology_live.py` | 1 | Add `DELETE /phenology-live/{inv}/{id}` |
| `backend/routers/predictions.py` | 1 | Add `DELETE /live-predictions/{inv}/{id}/kg-actual` (sets NULL) |
| `demo/Demo Dashboard.html` | 2 | Add Historial tab + panel HTML, `loadHistorial`, `histDelete`, `histConfirm`, update `initInsTabs` |
| `demo/Demo Dashboard.html` | 3 | Add confirmed-predictions list to prod panel, `prodDeleteKg`, `prodDeleteConfirm`, update `loadProdPredictions` and `initInsTabs` |

---

### Task 1: Backend DELETE Endpoints

**Files:**
- Modify: `backend/routers/phenology_live.py`
- Modify: `backend/routers/predictions.py`

**Interfaces:**
- Produces:
  - `DELETE /phenology-live/{inv}/{id}` → `{"ok": true}` or 404
  - `DELETE /live-predictions/{inv}/{id}/kg-actual` → `{"ok": true}` or 404

- [ ] **Step 1: Add DELETE to phenology_live.py**

Open `backend/routers/phenology_live.py`. Add after the `list_phenology` function:

```python
@router.delete("/phenology-live/{inv}/{obs_id}")
def delete_phenology(
    inv: int,
    obs_id: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("phenology_observations")
        .delete()
        .eq("id", obs_id)
        .eq("greenhouse_id", inv)
        .execute()
    )
    if not resp.data:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Observación no encontrada")
    return {"ok": True}
```

- [ ] **Step 2: Add clear-kg-actual to predictions.py**

Open `backend/routers/predictions.py`. Add after `update_kg_actual`:

```python
@router.delete("/live-predictions/{inv}/{prediction_id}/kg-actual")
def clear_kg_actual(
    inv: int,
    prediction_id: int,
    _user: Annotated[dict, Depends(get_current_user)] = None,
):
    resp = (
        service_client.table("predictions")
        .update({"kg_actual": None})
        .eq("id", prediction_id)
        .eq("greenhouse_id", inv)
        .execute()
    )
    if not resp.data:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return {"ok": True}
```

- [ ] **Step 3: Verify endpoints register correctly**

Start backend locally:
```bash
uvicorn backend.main:app --port 8000 --reload
```

Check routes appear:
```bash
curl -s http://localhost:8000/openapi.json | python3 -m json.tool | grep -E '"delete"' | head -5
```
Expected: lines containing `"delete"` for both routes.

- [ ] **Step 4: Commit**

```bash
git add backend/routers/phenology_live.py backend/routers/predictions.py
git commit -m "feat(api): add DELETE endpoints for phenology rows and kg_actual"
```

---

### Task 2: Historial Tab — HTML + JS

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `GET /phenology-live/{inv}?limit=500` (existing) · `DELETE /phenology-live/{inv}/{obs_id}` (Task 1)
- Produces: `loadHistorial(inv)` · `histDelete(id, inv, weekDate, zona, planta)` · `histConfirm()`

**Context — existing tab pattern to follow:**
- Tab buttons are `.ins-tab` with `data-tab` attribute, inside `.ins-tabs` div (line ~1339)
- Panels are `.panel.insert-panel.ins-panel` with `id="ins-panel-{name}"` and `hidden` attribute (line ~1345, 1379)
- `initInsTabs()` wires tab clicks and controls panel visibility (line ~3117)
- The confirm panel uses `class="ins-confirm-panel"` with `id="{name}-confirm"` (line ~1395)

- [ ] **Step 1: Add "Historial" tab button**

Find in `demo/Demo Dashboard.html`:
```html
      <button class="ins-tab active" data-tab="pheno">Fenología</button>
      <button class="ins-tab" data-tab="prod">Producción real</button>
```

Replace with:
```html
      <button class="ins-tab active" data-tab="pheno">Fenología</button>
      <button class="ins-tab" data-tab="prod">Producción real</button>
      <button class="ins-tab" data-tab="hist">Historial</button>
```

- [ ] **Step 2: Add Historial panel HTML**

Find in `demo/Demo Dashboard.html`:
```html
    <div class="panel insert-panel ins-panel" id="ins-panel-prod" hidden>
```

Insert this block BEFORE that line:

```html
    <div class="panel insert-panel ins-panel" id="ins-panel-hist" hidden>
      <div style="overflow-x:auto">
        <table class="insert-table" id="hist-table" style="min-width:900px">
          <thead>
            <tr>
              <th>Semana</th><th>Zona</th><th>Planta</th>
              <th>Rac. puestos</th><th>Flores</th><th>Rac. planta</th>
              <th>Tomates</th><th>Rac. cosecha</th><th>Maduros</th>
              <th>Diám. (cm)</th><th>Crec. (cm)</th><th></th>
            </tr>
          </thead>
          <tbody id="hist-tbody"></tbody>
        </table>
      </div>
      <div class="ins-confirm-panel" id="hist-confirm" hidden>
        <pre class="ins-confirm-summary" id="hist-confirm-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="hist-cancel">Cancelar</button>
          <button class="btn-primary" id="hist-confirm-btn" style="background:#dc2626">Confirmar eliminación</button>
        </div>
      </div>
      <div class="insert-result" id="hist-result"></div>
    </div>
```

- [ ] **Step 3: Add loadHistorial, histDelete, histConfirm JS**

Find the closing `</script>` tag at end of file (~line 3595). Insert BEFORE it:

```js
// ── Historial de fenología ────────────────────────────────────────────────
async function loadHistorial(inv) {
  const tbody = document.getElementById('hist-tbody');
  const result = document.getElementById('hist-result');
  tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:#9aa0a8">Cargando…</td></tr>';
  result.textContent = '';
  const token = localStorage.getItem('sb_token');
  try {
    const rows = await fetch(`${API_BASE}/phenology-live/${inv}?limit=500`, {
      headers: { 'Authorization': `Bearer ${token}` },
    }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:#9aa0a8">Sin observaciones registradas.</td></tr>';
      return;
    }

    const fmt = v => (v == null ? '—' : v);
    rows.sort((a, b) => b.week_date > a.week_date ? 1 : b.week_date < a.week_date ? -1 : (a.zona - b.zona) || (a.planta - b.planta));
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>${r.week_date}</td>
        <td>${r.zona}</td>
        <td>${r.planta}</td>
        <td>${fmt(r.racimos_puestos)}</td>
        <td>${fmt(r.flores_racimo_abiertas)}</td>
        <td>${fmt(r.racimos_en_planta)}</td>
        <td>${fmt(r.cantidad_tomates)}</td>
        <td>${fmt(r.racimo_en_cosecha)}</td>
        <td>${fmt(r.tomates_maduros)}</td>
        <td>${fmt(r.diametro_fruto_cm)}</td>
        <td>${fmt(r.crecimiento_planta_cm)}</td>
        <td><button class="btn-ghost" style="padding:2px 8px;font-size:0.8rem;color:#dc2626"
            onclick="histDelete(${r.id},${inv},'${r.week_date}',${r.zona},${r.planta})">🗑</button></td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="12" style="color:#dc2626">Error al cargar (${e.message})</td></tr>`;
  }
}

function histDelete(id, inv, weekDate, zona, planta) {
  document.getElementById('hist-result').textContent = '';
  const confirmEl = document.getElementById('hist-confirm');
  confirmEl.dataset.id   = id;
  confirmEl.dataset.inv  = inv;
  document.getElementById('hist-confirm-summary').textContent =
    `Eliminar observación\nInvernadero ${inv} · Semana ${weekDate} · Zona ${zona} · Planta ${planta}`;
  confirmEl.hidden = false;
}

async function histConfirm() {
  const confirmEl = document.getElementById('hist-confirm');
  const id    = +confirmEl.dataset.id;
  const inv   = +confirmEl.dataset.inv;
  const result = document.getElementById('hist-result');
  confirmEl.hidden = true;
  const token = localStorage.getItem('sb_token');
  try {
    const r = await fetch(`${API_BASE}/phenology-live/${inv}/${id}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || (r.status === 401 ? 'Sesión expirada — recarga la página' : `Error ${r.status}`));
    }
    result.className = 'insert-result ok';
    result.textContent = '✓ Observación eliminada';
    await loadHistorial(inv);
  } catch (e) {
    result.className = 'insert-result bad';
    result.textContent = '⚠ ' + e.message;
  }
}
```

- [ ] **Step 4: Update initInsTabs to wire Historial tab**

Find `initInsTabs()`:
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

Replace with:
```js
function initInsTabs() {
  document.querySelectorAll('.ins-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ins-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      const inv = +document.getElementById('ins-inv').value;
      document.getElementById('ins-panel-pheno').hidden = tab !== 'pheno';
      document.getElementById('ins-panel-prod').hidden  = tab !== 'prod';
      document.getElementById('ins-panel-hist').hidden  = tab !== 'hist';
      if (tab === 'prod')  loadProdPredictions(inv);
      if (tab === 'hist')  loadHistorial(inv);
    });
  });
  document.getElementById('ins-inv').addEventListener('change', () => {
    const activeTab = document.querySelector('.ins-tab.active')?.dataset.tab;
    const inv = +document.getElementById('ins-inv').value;
    if (activeTab === 'prod') loadProdPredictions(inv);
    if (activeTab === 'hist') loadHistorial(inv);
  });
}
```

- [ ] **Step 5: Wire hist-confirm and hist-cancel buttons in init()**

Find in `init()`:
```js
    document.getElementById('prod-confirm-btn').addEventListener('click', prodConfirm);
    document.getElementById('prod-cancel').addEventListener('click', () => {
      document.getElementById('prod-confirm').hidden = true;
      document.getElementById('prod-submit-btn').disabled = false;
    });
```

Add after that block:
```js
    document.getElementById('hist-confirm-btn').addEventListener('click', histConfirm);
    document.getElementById('hist-cancel').addEventListener('click', () => {
      document.getElementById('hist-confirm').hidden = true;
    });
```

- [ ] **Step 6: Verify in browser**

Open dashboard, go to "Insertar datos". Confirm three tabs visible: Fenología · Producción real · Historial. Click Historial — table should load (or show "Sin observaciones"). Click 🗑 on a row — confirm panel appears with correct summary. Cancel — panel hides. Confirm delete — row disappears.

Browser console check:
```js
document.getElementById('ins-panel-hist') // should return element, not null
document.getElementById('hist-confirm')    // should return element, not null
```

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add Historial tab with phenology table and delete confirmation"
```

---

### Task 3: Producción real — Delete kg_actual

**Files:**
- Modify: `demo/Demo Dashboard.html`

**Interfaces:**
- Consumes: `DELETE /live-predictions/{inv}/{id}/kg-actual` (Task 1) · `loadProdPredictions(inv)` (existing) · `refreshLiveData()` (existing)
- Produces: `prodDeleteKg(id, inv, predictedFor, kgActual)` · `prodDeleteConfirm()`

**Context:** The Producción real panel (`ins-panel-prod`) currently has: dropdown (`prod-week-select`) + kg input + Registrar button + confirm panel (`prod-confirm`) + result div (`prod-result`). The new "Borrar real" list sits between the dropdown/form and the existing confirm panel, showing only predictions that already have `kg_actual`.

- [ ] **Step 1: Add confirmed-predictions list container to prod panel HTML**

Find in `demo/Demo Dashboard.html`:
```html
      <div class="ins-confirm-panel" id="prod-confirm" hidden>
```

Insert BEFORE that line:
```html
      <div id="prod-confirmed-list" style="margin-top:16px"></div>
```

- [ ] **Step 2: Add prod-delete confirm panel HTML**

The existing `prod-confirm` is for registering. Add a separate confirm for deleting. Find:
```html
      <div class="insert-result" id="prod-result"></div>
    </div>
```

Replace with:
```html
      <div class="ins-confirm-panel" id="prod-delete-confirm" hidden>
        <pre class="ins-confirm-summary" id="prod-delete-summary"></pre>
        <div class="ins-confirm-actions">
          <button class="btn-ghost" id="prod-delete-cancel">Cancelar</button>
          <button class="btn-primary" id="prod-delete-confirm-btn" style="background:#dc2626">Confirmar</button>
        </div>
      </div>
      <div class="insert-result" id="prod-result"></div>
    </div>
```

- [ ] **Step 3: Update loadProdPredictions to also render confirmed list**

Find `loadProdPredictions` function. The current function sets `select.innerHTML`. After setting `select.innerHTML`, add rendering for the confirmed list.

Replace the entire `loadProdPredictions` function:
```js
async function loadProdPredictions(inv) {
  const select = document.getElementById('prod-week-select');
  const confirmedList = document.getElementById('prod-confirmed-list');
  select.innerHTML = '<option disabled selected>Cargando…</option>';
  if (confirmedList) confirmedList.innerHTML = '';
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

    const confirmed = preds.filter(p => p.kg_actual != null);
    if (confirmedList) {
      confirmedList.innerHTML = confirmed.length
        ? `<div style="font-size:0.75rem;font-weight:600;color:#9aa0a8;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em">Producción registrada</div>` +
          confirmed.map(p => `
            <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-top:1px solid rgba(0,0,0,0.06);font-size:0.85rem">
              <span style="min-width:90px;color:#6b7280">${p.predicted_for}</span>
              <span style="font-weight:600">${Math.round(p.kg_actual).toLocaleString('es-MX')} kg</span>
              <span style="color:#9aa0a8">pred: ${Math.round(p.kg_predicted).toLocaleString('es-MX')} kg</span>
              <button class="btn-ghost" style="margin-left:auto;padding:2px 10px;font-size:0.8rem;color:#dc2626"
                onclick="prodDeleteKg(${p.id},${inv},'${p.predicted_for}',${p.kg_actual})">Borrar real</button>
            </div>`).join('')
        : '';
    }
  } catch (e) {
    select.innerHTML = `<option disabled selected>Error al cargar (${e.message})</option>`;
  }
}
```

- [ ] **Step 4: Add prodDeleteKg and prodDeleteConfirm JS**

Find the `// ── Historial de fenología` comment added in Task 2. Insert BEFORE it:

```js
// ── Borrar kg_actual ──────────────────────────────────────────────────────
function prodDeleteKg(id, inv, predictedFor, kgActual) {
  document.getElementById('prod-result').textContent = '';
  document.getElementById('prod-confirm').hidden = true;
  const confirmEl = document.getElementById('prod-delete-confirm');
  confirmEl.dataset.id  = id;
  confirmEl.dataset.inv = inv;
  document.getElementById('prod-delete-summary').textContent =
    `Borrar producción real\nInvernadero ${inv} · Semana ${predictedFor}\nkg real registrado: ${Math.round(kgActual).toLocaleString('es-MX')} kg`;
  confirmEl.hidden = false;
}

async function prodDeleteConfirm() {
  const confirmEl = document.getElementById('prod-delete-confirm');
  const id    = +confirmEl.dataset.id;
  const inv   = +confirmEl.dataset.inv;
  const result = document.getElementById('prod-result');
  confirmEl.hidden = true;
  const token = localStorage.getItem('sb_token');
  try {
    const r = await fetch(`${API_BASE}/live-predictions/${inv}/${id}/kg-actual`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || (r.status === 401 ? 'Sesión expirada — recarga la página' : `Error ${r.status}`));
    }
    result.className = 'insert-result ok';
    result.textContent = '✓ Producción real eliminada';
    await loadProdPredictions(inv);
    refreshLiveData();
  } catch (e) {
    result.className = 'insert-result bad';
    result.textContent = '⚠ ' + e.message;
  }
}
```

- [ ] **Step 5: Wire prod-delete confirm/cancel in init()**

Find in `init()`:
```js
    document.getElementById('hist-confirm-btn').addEventListener('click', histConfirm);
    document.getElementById('hist-cancel').addEventListener('click', () => {
      document.getElementById('hist-confirm').hidden = true;
    });
```

Add after:
```js
    document.getElementById('prod-delete-confirm-btn').addEventListener('click', prodDeleteConfirm);
    document.getElementById('prod-delete-cancel').addEventListener('click', () => {
      document.getElementById('prod-delete-confirm').hidden = true;
    });
```

- [ ] **Step 6: Verify in browser**

Open "Insertar datos" → "Producción real". Select inv with a confirmed kg_actual. Confirm "Producción registrada" section appears below dropdown showing the week + actual + "Borrar real" button. Click "Borrar real" — confirm panel appears with correct summary. Cancel — panel hides. Confirm — row disappears from list, dropdown updates to remove ✓, `refreshLiveData` updates forecast section.

- [ ] **Step 7: Commit**

```bash
git add "demo/Demo Dashboard.html"
git commit -m "feat(dashboard): add Borrar real action on Producción real tab"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|------------------|------|
| `DELETE /phenology-live/{inv}/{id}` backend endpoint | Task 1 Step 1 |
| `DELETE /live-predictions/{inv}/{id}/kg-actual` backend endpoint | Task 1 Step 2 |
| "Historial" tab button in ins-tabs | Task 2 Step 1 |
| Historial panel with scrollable table | Task 2 Step 2 |
| `loadHistorial(inv)` — fetch + render rows | Task 2 Step 3 |
| `histDelete(...)` — show confirm panel | Task 2 Step 3 |
| `histConfirm()` — fires DELETE, re-fetches table | Task 2 Step 3 |
| `initInsTabs` updated — hist panel hide/show + loadHistorial on tab/inv change | Task 2 Step 4 |
| hist-confirm / hist-cancel wired in init() | Task 2 Step 5 |
| `prod-confirmed-list` container in prod panel | Task 3 Step 1 |
| `prod-delete-confirm` panel in prod panel | Task 3 Step 2 |
| `loadProdPredictions` renders confirmed list with "Borrar real" buttons | Task 3 Step 3 |
| `prodDeleteKg(...)` — show delete confirm | Task 3 Step 4 |
| `prodDeleteConfirm()` — fires DELETE, re-fetches + refreshLiveData | Task 3 Step 4 |
| prod-delete-confirm / prod-delete-cancel wired in init() | Task 3 Step 5 |
| Error: 401 → "Sesión expirada" | Task 2 Step 3, Task 3 Step 4 |
| Error: 404 → backend raises HTTPException | Task 1 Steps 1–2 |

All spec requirements covered. No gaps.

**Type consistency:** `histDelete(id, inv, weekDate, zona, planta)` defined in Step 3, called from inline `onclick` in same step — consistent. `prodDeleteKg(id, inv, predictedFor, kgActual)` defined in Step 4, called from inline `onclick` in Step 3 — consistent.

**Placeholder scan:** None found.
