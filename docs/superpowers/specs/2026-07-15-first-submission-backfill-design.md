# First-Submission Backfill — Design Spec

**Date:** 2026-07-15
**Status:** Approved

---

## Overview

T18 transplant dates: Inv3 = 2026-05-15, Inv4 = 2026-05-16. First submission of
the season will land in ISO week 31 (2026), and per the client that first
submission IS the first week of production. Today, `scripts/live_inference.py`
writes exactly one `predictions` row per triggering upload (see
`docs/superpowers/specs/2026-07-13-live-inference-production-design.md` §4 and
`2026-07-14-t18-slider-chart-and-upload-triggered-inference-design.md`). On the
very first submission this means only one row gets written, and the
weeks-since-harvest-start indexing (`wis_target`) that decides mean-vs-model
never revisits earlier weeks — the ramp-up weeks (Section 2) currently never
get a row at all.

This spec fixes that, and fixes a pre-existing bug in how "week 1 of
production" is anchored that the fix depends on.

---

## Section 1: `harvest_start` — stored explicitly in Supabase, not inferred

**Problem:** `get_harvest_start()` (`scripts/live_inference.py:71-84`) returns
`MIN(fecha)` from `sensor_readings_wide` for that greenhouse. Per CLAUDE.md §5,
sensors start recording ~10-11 weeks *before* harvest, and the client's first
upload of a season batches that entire backlog in one payload — so
`MIN(fecha)` resolves to ~week 20-21, not week 31. `wis_target` (the index the
mean-vs-model decision and the season's "week N" numbering are both built on)
is computed relative to this — as it stands today, in production, the ramp-up
mean fallback added in the 2026-07-13 spec effectively never fires, and no
part of the pipeline currently notices.

**Fix:** stop inferring `harvest_start` at all — store it explicitly in
Supabase as an admin-set fact, same pattern as `transplant_dates`
(2026-07-13 spec §3), rather than derive it from sensor rows or from
`predictions`. New table:

```sql
-- part of supabase/migrations/010_harvest_start_dates.sql
create table if not exists harvest_start_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);
```

New endpoints (`backend/routers/harvest_start_dates.py`, identical pattern to
`transplant_dates.py`):

```
GET /harvest-start-date/{inv}   → { greenhouse_id, fecha, updated_at } or 404 if unset
PUT /harvest-start-date/{inv}   → upsert { fecha }; any authenticated user (JWT)
```

Small dashboard addition alongside the existing transplant-date picker: one
more date-picker + save button per greenhouse (admin/settings area), calling
`PUT /harvest-start-date/{inv}`.

`get_harvest_start()` becomes a direct read:

```python
def get_harvest_start(supa, inv: int) -> date | None:
    resp = (
        supa.table("harvest_start_dates")
        .select("fecha")
        .eq("greenhouse_id", inv)
        .maybe_single()
        .execute()
    )
    row = resp.data if resp is not None else None
    if not row:
        return None
    return date.fromisoformat(row["fecha"])
```

**Missing harvest_start_dates row:** same skip behavior as a missing
`transplant_dates` row (2026-07-13 spec §3) — that greenhouse's inference is
skipped for this run, `ERROR`-level log line, other greenhouse unaffected.
This replaces the earlier "`None` means first submission" signal (previous
draft of this spec) — first-submission detection moves entirely to Section 3
(whether `predictions` has rows yet for this greenhouse+season), decoupled
from whether `harvest_start` is known.

**For T18:** admin sets `harvest_start_dates` to `2026-07-27` (Monday of ISO
week 31, 2026) for both Inv3 and Inv4 before/at go-live — a manual step,
same as setting `transplant_dates` (2026-07-13 spec §3 "manual step when the
next season starts").

`transplant_date` is unchanged as an input to `build_input_tensor`'s temporal
features (`dias_desde_transplante`) — only the ramp-up/week-numbering anchor
moves off of sensor data (and off of `predictions`) onto this new stored
field.

---

## Section 2: Extend ramp-up mean coverage from 3 to 4 weeks

- `SKIP_FIRST_WEEKS = 3 → 4` in `scripts/live_inference.py`.
- `scripts/compute_historical_means.py`: currently averages positions 0, 1, 2
  (see `Models/results/historical_mean_inv{3,4}.json`, keys `"0","1","2"`).
  Extend to also compute position `"3"` (4th positional week), same method
  (mean of that position's weekly kg across T13-T17). Re-run once to
  regenerate both JSON files.

This is a standalone data/config change — `SEQ_LEN_BY_INV` /
`skip_first_weeks: 3` in `Models/hp_inv{3,4}.yaml` (the *training*-time
constant, CLAUDE.md §7-8) is untouched; this only affects the *live inference*
ramp-up threshold, which is allowed to diverge from the training-time skip
value per the 2026-07-13 spec's design (the live fallback is a live-pipeline
concern, not a training concern).

---

## Section 3: First-submission backfill (dynamic — as many rows as data supports)

**Detection:** decoupled from Section 1 now that `harvest_start` is a stored
fact rather than an inferred one. A call to `run_inference_for_greenhouse` is
the first submission of the season for that greenhouse iff there is no
`predictions` row yet for `(greenhouse_id, season=CURRENT_SEASON)`:

```python
def is_first_submission(supa, inv: int) -> bool:
    resp = (
        supa.table("predictions")
        .select("id")
        .eq("greenhouse_id", inv)
        .eq("season", CURRENT_SEASON)
        .limit(1)
        .execute()
    )
    return not (resp.data or [])
```

`harvest_start` itself is read from `harvest_start_dates` (Section 1) —
required for both the first-submission and non-first-submission paths; if
unset, skip this greenhouse this run (Section 1's missing-row behavior),
same as a missing `transplant_dates` row.

**On first submission**, instead of writing one row, write as many as can
legitimately be computed right now — no hardcoded row count. Today (week 29,
2026-07-15) is 2 weeks before T18's actual first submission (week 31), so
the exact number is not knowable ahead of time and must not be hardcoded
into the pipeline; it depends on how much sensor backlog the client's first
upload actually contains, which varies by greenhouse and by season.

1. **Ramp-up weeks** — for `wis` in `0 .. SKIP_FIRST_WEEKS-1`: always write
   from `historical_mean_inv{inv}.json[str(wis)]`. These have no sensor-data
   dependency, so all `SKIP_FIRST_WEEKS` of them are always written.
2. **Model weeks** — starting at `wis = SKIP_FIRST_WEEKS`, loop upward one
   week at a time:
   - `as_of_date = harvest_start + timedelta(weeks=wis - HORIZON_WEEKS)`
   - `predicted_for = harvest_start + timedelta(weeks=wis)`
   - If `as_of_date > date.today()`: stop — there's no sensor data that far
     in the future yet (this is the natural ceiling; it's what makes the
     normal non-first-submission path only ever produce one row per week,
     since `as_of_date` catches up to `today` one week at a time).
   - Otherwise call `_run_model_forward(..., as_of_date=as_of_date)`. If the
     window it builds needs padding (fewer than `seq_len` real weekly rows
     before `as_of_date` — see Section 3a below) — stop, this greenhouse's
     backlog doesn't reach this far back. Log an `INFO` line with how many
     weeks were backfilled and continue the outer per-greenhouse loop
     (Section 2 of the 2026-07-13 spec) — this is not an error.
   - Otherwise write the row and continue to `wis + 1`.

So the number of rows written on first submission is
`SKIP_FIRST_WEEKS + (however many consecutive model weeks the backlog and
today's date support)` — could be as few as `SKIP_FIRST_WEEKS` (mean only, if
the trigger fires exactly on harvest_start day with a thin backlog) or more,
naturally, with no upper bound imposed by the code itself.

**Non-first submissions** (`is_first_submission` is `False`): unchanged — one
row, `predicted_for = monday_of_week(today + HORIZON_WEEKS)`, model or mean
chosen the same way as today, just using the stored `harvest_start`.

### Refactor: `as_of_date`-parameterized model forward pass

`build_input_tensor` (`backend/live_features.py`) already takes pre-filtered
`wide_rows`/`pheno_rows` and does not read "today" internally — it just tails
the last `seq_len` weekly rows it's given. So truncating the window is a
caller-side filter, not a `live_features.py` change.

Extract the "pull sensor/riego/pheno data, build tensor, run model" block
(`scripts/live_inference.py:112-191`, currently inline) into:

```python
def _run_model_forward(supa, inv: int, transplant_date: date,
                        pipeline_path: Path, hp: dict, as_of_date: date) -> float:
    """Returns kg_predicted for a sensor window ending at as_of_date."""
    cutoff = (as_of_date - timedelta(weeks=12)).isoformat()
    upper = as_of_date.isoformat()
    # sensor_resp / riego_resp / ext_resp / pheno_resp queries: add
    # .lte("fecha", upper) alongside the existing .gte("fecha", cutoff)
    ...
    # (rest unchanged: merge by fecha, build_input_tensor, load model, forward pass,
    #  inverse-transform kg_predicted)
    return kg_predicted
```

Callers:
- Normal path: `_run_model_forward(..., as_of_date=date.today())`,
  `predicted_for = monday_of_week(date.today() + timedelta(weeks=HORIZON_WEEKS))`.
- First-submission wis=4 row: `_run_model_forward(..., as_of_date=harvest_start - timedelta(weeks=1))`,
  `predicted_for = harvest_start + timedelta(weeks=4)`.
- First-submission wis=5 row: `_run_model_forward(..., as_of_date=date.today())`,
  `predicted_for = harvest_start + timedelta(weeks=5)` (same value as the
  normal-path formula would give on this same day).

`pheno_resp` query also gains `.lte("week_date", upper)` for the same reason.

### Section 3a: Data sufficiency — checked at runtime, not assumed

Earlier drafts of this spec assumed `MIN_WEEKS_FIRST_BY_INV`
(`backend/routers/uploads.py` — the upload-time check that the first upload
carries `seq_len + HORIZON` weeks of history relative to *today*) guarantees
enough backlog for every truncated window this section needs. That's an
upload-time check anchored to *today*, not to each individual `as_of_date`
this loop constructs — it says nothing about whether a window ending a
week (or more) earlier than today has `seq_len` real rows. Relying on it
here would silently break the moment `MIN_WEEKS_FIRST_BY_INV`'s formula or
the client's actual backlog habits change. Instead, sufficiency is checked
directly against the data actually pulled for each `as_of_date`:

`_run_model_forward` returns a sentinel (e.g. `None`) instead of a float
when `aggregate_wide_to_weekly(...)` (`backend/live_features.py:29-50`),
after filtering to `fecha ≤ as_of_date`, yields fewer than `seq_len` weekly
rows — i.e. exactly the condition that today triggers zero-padding
(`backend/live_features.py:110-114`). The Section 3 loop stops as soon as it
sees this sentinel, rather than writing a prediction built on padded
(fabricated) input.

---

## Edge cases

- **Concurrent triggers race:** two uploads completing near-simultaneously
  could both observe `is_first_submission() == True` and both run the
  first-submission branch. Harmless — every write is an `upsert` keyed on
  `(greenhouse_id, predicted_for)`; duplicate work, identical final state.
- **`harvest_start_dates` unset:** greenhouse skipped this run, same as a
  missing `transplant_dates` row (Section 1) — logged at `ERROR`, other
  greenhouse unaffected. Must be set before T18 go-live for both invs.
- **Missing historical mean key:** if `historical_mean_inv{inv}.json` is
  missing a required key (shouldn't happen post Section 2, but defensively),
  skip that one row with an `ERROR`-level log line and continue with the
  rest — matches the existing "log and continue" pattern for the two-inv
  loop (2026-07-13 spec §2).
- **Future seasons (T19+):** no special-casing — `CURRENT_SEASON` bump +
  new `transplant_dates` row is the only manual step (unchanged from
  2026-07-13 spec), and the first upload of that new season naturally
  re-triggers this same first-submission branch since `predictions` has no
  rows yet for that `(greenhouse_id, season)` pair.

---

## What is NOT in scope

- Changing `Models/hp_inv{3,4}.yaml` `skip_first_weeks` (training-time
  constant) — only the live-inference `SKIP_FIRST_WEEKS` constant changes.
- Retraining or any change to `Models/cnn_rnn_yield.py`.
- Dashboard changes — the T18 tab already renders whatever rows exist in
  `predictions` for the season; six rows appearing at once instead of
  trickling in over six weeks requires no frontend change.
- Backfilling seasons that already have partial `predictions` rows under the
  old single-row-per-week behavior (T18 has none yet — go-live is in the
  future relative to this spec).

---

## Files touched (summary)

- `supabase/migrations/010_harvest_start_dates.sql` — new `harvest_start_dates`
  table.
- `backend/routers/harvest_start_dates.py` — new router, mirrors
  `transplant_dates.py` (GET/PUT `/harvest-start-date/{inv}`).
- `scripts/live_inference.py` — `get_harvest_start` reads
  `harvest_start_dates` directly (no more inference from sensor data or
  `predictions`), new `is_first_submission()` helper, `SKIP_FIRST_WEEKS`
  3→4, model-forward-pass extracted into
  `_run_model_forward(..., as_of_date) -> float | None`, first-submission
  branch looping until `as_of_date` exceeds today or the window needs
  padding, instead of writing 1 fixed row.
- `scripts/compute_historical_means.py` — add position `"3"`.
- `Models/results/historical_mean_inv{3,4}.json` — regenerated with 4 keys
  (`"0".."3"`) instead of 3.
- Dashboard (`demo/Demo Dashboard.html`) — one more date-picker + save
  button per greenhouse (admin/settings area, alongside the existing
  transplant-date picker), calling `PUT /harvest-start-date/{inv}`.
- **Manual step, not code:** set `harvest_start_dates.fecha = 2026-07-27`
  for Inv3 and Inv4 before/at T18 go-live.
