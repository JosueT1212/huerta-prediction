# First-Submission Window Coverage Validation — Design

**Date:** 2026-07-13
**Status:** Approved by user, pending implementation plan

## Context

The excel-upload feature (sensores/riego/exteriores/fenologia/produccion,
`docs/superpowers/specs/2026-07-09-insertar-datos-excel-upload-design.md`,
`docs/superpowers/specs/2026-07-11-split-sensor-uploads-design.md`) currently
validates only that required columns are present. It does not validate that
enough *rows* exist for the live-inference pipeline
(`scripts/live_inference.py` → `backend/live_features.py`) to build a real
prediction window.

`build_input_tensor`'s `aggregate_wide_to_weekly` takes `.tail(seq_len)` of
the aggregated weekly sensor data and **silently zero-pads** if fewer than
`seq_len` weeks exist:

```python
if len(weekly) < seq_len:
    pad_rows = seq_len - len(weekly)
    pad_data = {col: [0.0] * pad_rows for col in weekly.columns}
    pad = pd.DataFrame(pad_data)
    weekly = pd.concat([pad, weekly], ignore_index=True)
```

Zero-padded weeks are not "missing data" to the model — they're indistinguishable
from real (post-MinMaxScaler) sensor readings, so a short first upload produces
a confident-looking but meaningless first prediction, with no error anywhere.

This spec adds a harness that rejects an upload before writing anything if it
doesn't supply enough real data for a sound prediction — for the first
submission, and for every submission after it.

## Scope

Sensores, Riego, Exteriores, Fenología. Fenología has a historical-mean
fallback (`pheno_means`, keyed by week-in-season) for any week without real
observations, so short coverage there degrades gracefully rather than
silently corrupting the input — but per explicit user instruction
(2026-07-13), the harness is applied uniformly across all four upload
types regardless, using the same per-invernadero thresholds as
sensores/riego. Producción is the prediction target, not a model input
feature — it has no "window" to fill, and stays out of scope.

## 1. Required constants (new, in `backend/routers/uploads.py`)

Sourced from `Models/hp_inv3.yaml` / `Models/hp_inv4.yaml` (`seq_len`) and
`Models/cnn_rnn_yield.py` (`HORIZON = 5`) — copied as constants here rather
than loaded at runtime, to keep the upload path free of the
torch/joblib/pipeline-loading dependency chain that `backend/engine.py` and
`scripts/live_inference.py` carry. A comment in the code points back to
these two source files as ground truth.

```python
HORIZON = 5
SEQ_LEN_BY_INV = {3: 4, 4: 2}
MIN_WEEKS_FIRST_BY_INV = {inv: SEQ_LEN_BY_INV[inv] + HORIZON for inv in SEQ_LEN_BY_INV}
# {3: 9, 4: 7}
MIN_WEEKS_FIRST_EXTERIORES = max(MIN_WEEKS_FIRST_BY_INV.values())  # 9
MIN_NEW_DAYS_SUBSEQUENT = 7
```

**Why `seq_len + HORIZON` and not just `seq_len`:** confirmed with the user
this is a deliberate safety margin beyond the mechanical minimum
`build_input_tensor` needs (which is `seq_len` weeks) — not derived from the
training-time `_apply_gap_norm_cnn` formula (`gap - seq_len - HORIZON +
skip_first_weeks + 1`), which is a per-season alignment fix with no
equivalent in continuous live uploads (no season boundary to measure `gap`
against). `gap` and `skip_first_weeks` are explicitly out of scope for this
check.

**Exteriores uses `max()` across invernaderos** since it's global (no
`greenhouse_id`) but feeds every invernadero's model — using the largest
requirement guarantees sufficiency for all of them, including Inv4 once it
gets a pipeline (`Models/results/pipeline_inv4.pkl` doesn't exist yet, but
`hp_inv4.yaml` does — see `docs/superpowers/specs/2026-06-23-live-inference-pipeline-design.md`
for why Inv4 is currently excluded from live inference).

**Fenología uses `week_date` as its DB date column** (not `fecha` —
`phenology_observations.week_date`, see `_ingest_rows`'s fenología branch in
`backend/routers/uploads.py`), so the existence/subsequent queries for that
scope filter on `week_date` while the uploaded file's column stays `fecha`
(the upload-side column name is unchanged; only the DB-column used for the
coverage query differs by form_type).

## 2. First-submission check

"First" = no existing rows for that scope: `(greenhouse_id, table)` for
sensores/riego/fenología, table-wide for exteriores (global). Detected by a
cheap existence query (`select {date_col} ... limit 1`) before validating
row count.

Count **distinct ISO-weeks** (`year`, `week` from `fecha`) in the uploaded
file. If below the required minimum, reject with 422 before any row is
written:

```
"Not sufficient data for first submission: se requieren al menos {required}
semanas de datos, el archivo cubre {n} semana(s)."
```

## 3. Subsequent-submission check

Not "first" (prior rows exist for that scope). Count distinct `fecha`
values in the uploaded file that are **not already present** in the DB for
that scope (a set-difference against existing `fecha`s, not ISO-week
grouping — matches the user's "7 days" framing literally). If fewer than 7,
reject with 422:

```
"Not sufficient data: se requieren al menos 7 días nuevos de datos, el
archivo aporta {n} día(s) nuevo(s)."
```

This is **in addition to** the existing `submission_locks` 7-day cadence
gate (unchanged) — that gate stops submitting *too often*; this stops
submitting *too little* once the window reopens.

Not required: contiguity (no gap-free-run requirement between the new days
and existing coverage) — a simple new-distinct-day count, per YAGNI.

## 4. Integration point

Both checks run in `upload_excel` / `upload_exteriores`, immediately after
`_parse_excel` (column validation, NaN-cleaning) and immediately after the
existing `check_submission_lock` call, before `_ingest_rows` writes
anything — so a rejection is atomic: zero rows touched, matching the
existing missing-columns-422 behavior.

Order of checks in the request: auth → form_type validity → submission
lock → column validation (`_parse_excel`) → **window-coverage check (new)**
→ ingest → touch lock.

## 5. Frontend

No new UI. The rejection reason (`HTTPException(422, "...")`) surfaces
through the existing generic error path already wired for every upload tab
(`result.textContent = '⚠ ' + e.message`).

## Out of scope

- Producción window-coverage validation (not a model input).
- Any change to `build_input_tensor`'s zero-padding behavior itself — this
  spec prevents the *conditions* that trigger it via upload-time validation,
  it doesn't change what happens if padding occurs anyway (e.g. via direct
  DB manipulation outside the upload API).
- Contiguity/gap-free-run validation on subsequent submissions.
- Inv4 live-inference support (tracked separately, `MIN_WEEKS_FIRST_BY_INV[4]`
  is defined now for when it ships, per the "future-proof" decision already
  made for `PER_INV_TYPES` in the split-sensor-uploads work).
