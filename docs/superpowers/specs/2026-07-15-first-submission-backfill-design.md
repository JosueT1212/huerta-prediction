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
never revisits earlier weeks — production weeks 1–4 (ramp-up) currently never
get a row at all.

This spec fixes that, and fixes a pre-existing bug in how "week 1 of
production" is anchored that the fix depends on.

---

## Section 1: Fix `harvest_start` anchoring (pre-existing bug)

**Problem:** `get_harvest_start()` (`scripts/live_inference.py:71-84`) returns
`MIN(fecha)` from `sensor_readings_wide` for that greenhouse. Per CLAUDE.md §5,
sensors start recording ~10-11 weeks *before* harvest, and the client's first
upload of a season batches that entire backlog in one payload — so
`MIN(fecha)` resolves to ~week 20-21, not week 31. `wis_target` (the index the
mean-vs-model decision and the season's "week N" numbering are both built on)
is computed relative to this — as it stands today, in production, the ramp-up
mean fallback added in the 2026-07-13 spec effectively never fires, and no
part of the pipeline currently notices.

**Fix:** anchor `harvest_start` to the first `predicted_for` value already
written for this greenhouse+season, not to sensor data:

```python
def get_harvest_start(supa, inv: int, season: str) -> date | None:
    resp = (
        supa.table("predictions")
        .select("predicted_for")
        .eq("greenhouse_id", inv)
        .eq("season", season)
        .order("predicted_for")
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None  # no predictions yet this season → this call is the first submission
    return date.fromisoformat(rows[0]["predicted_for"])
```

`transplant_date` is unchanged as an input to `build_input_tensor`'s temporal
features (`dias_desde_transplante`) — only the ramp-up/week-numbering anchor
moves off of sensor data.

When `get_harvest_start` returns `None`, the caller is in the first-submission
case (Section 3) and uses `monday_of_week(date.today())` as the season's
`harvest_start` for that run.

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

## Section 3: First-submission backfill (6 rows in one trigger call)

**Detection:** a call to `run_inference_for_greenhouse` is the first
submission of the season for that greenhouse iff `get_harvest_start` (Section
1) returns `None`.

**On first submission**, instead of writing one row, write six:

| wis | Calendar week | Source | How |
|---|---|---|---|
| 0 | week 31 | historical mean | `historical_mean_inv{inv}.json["0"]` |
| 1 | week 32 | historical mean | `historical_mean_inv{inv}.json["1"]` |
| 2 | week 33 | historical mean | `historical_mean_inv{inv}.json["2"]` |
| 3 | week 34 | historical mean | `historical_mean_inv{inv}.json["3"]` |
| 4 | week 35 | model | forward pass, sensor window truncated to `fecha ≤ week 30` |
| 5 | week 36 | model | forward pass, freshest sensor window (`fecha ≤ today`) — identical to today's existing normal-path call |

`harvest_start = monday_of_week(date.today())` for this run (week 31).
`predicted_for` for each row = `harvest_start + wis weeks`.

The wis=5/week-36 row is not an extra feature — it's what today's unmodified
single-call logic already produces (`predicted_for = today + HORIZON_WEEKS`).
It's included here because the refactor below runs it as part of the same
first-submission branch rather than special-casing it away; a non-first-run
next week would otherwise recompute the same row via upsert regardless.

**Non-first submissions** (`get_harvest_start` returns a date): unchanged —
one row, `predicted_for = monday_of_week(today + HORIZON_WEEKS)`, model or
mean chosen the same way as today, just using the corrected `harvest_start`.

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

### Data sufficiency for the truncated (week 30) window

`MIN_WEEKS_FIRST_BY_INV` (`backend/routers/uploads.py`) already requires the
first upload to carry `seq_len + HORIZON` weeks of history relative to
*today* (week 31) before it's accepted. Since the backlog spans ~10-11 weeks
before week 31 (CLAUDE.md §5), a window ending at week 30 (one week earlier)
still has `seq_len` trailing weeks available for both Inv3 (`seq_len=4`) and
Inv4 (`seq_len=2`) — no additional validation needed.

---

## Edge cases

- **Concurrent triggers race:** two uploads completing near-simultaneously
  could both observe `get_harvest_start() is None` and both run the
  first-submission branch. Harmless — every write is an `upsert` keyed on
  `(greenhouse_id, predicted_for)`; duplicate work, identical final state.
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

- `scripts/live_inference.py` — `get_harvest_start` reanchored to
  `predictions` table, `SKIP_FIRST_WEEKS` 3→4, model-forward-pass extracted
  into `_run_model_forward(..., as_of_date)`, first-submission branch writing
  6 rows instead of 1.
- `scripts/compute_historical_means.py` — add position `"3"`.
- `Models/results/historical_mean_inv{3,4}.json` — regenerated with 4 keys
  (`"0".."3"`) instead of 3.
