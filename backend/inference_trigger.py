"""Fires live inference right after the last required weekly upload lands.

Completeness requires all four required types (sensores/riego/fenologia per
invernadero, exteriores global) to have been submitted within the current
ISO week — i.e. each timestamp must be at or after Monday 00:00 UTC of the
week containing `now`. This anchors freshness to the calendar week instead
of a rolling window: a set that straddles a week boundary (e.g. three types
uploaded late last week, the fourth uploaded early this week) is never
considered complete here, and is instead picked up by the Sunday cron
fallback. Within a single ISO week, the four required types can each only
be submitted once (a repeat submission is rejected by
check_submission_lock() with 429 before it ever reaches this module), so
uploads_complete_for_inv() flips from False to True exactly once per
complete week, on the upload call that completes the set — no extra "was it
already complete" bookkeeping is needed here.
"""
from datetime import datetime, time, timedelta, timezone
import sys

from backend.lock_utils import last_submitted_at

# Mirrors the sentinel in backend/routers/uploads.py (EXTERIORES_GH_ID) —
# exteriores has no real invernadero, 0 is never a real greenhouse id.
EXTERIORES_GH_ID = 0

REQUIRED_PER_INV_TYPES = ("sensores", "riego", "fenologia")


def uploads_complete_for_inv(inv: int) -> bool:
    """True if sensores+riego+fenologia (this inv) + exteriores (global)
    were all submitted within the current ISO week (Mon 00:00 UTC onward)."""
    now = datetime.now(timezone.utc)
    monday = datetime.combine(
        now.date() - timedelta(days=now.weekday()), time.min, tzinfo=timezone.utc
    )
    for form_type in REQUIRED_PER_INV_TYPES:
        ts = last_submitted_at(inv, form_type)
        if ts is None or ts < monday:
            return False
    ts = last_submitted_at(EXTERIORES_GH_ID, "exteriores")
    if ts is None or ts < monday:
        return False
    return True


def maybe_trigger_inference(form_type: str, inv: int | None) -> None:
    """Call after a successful upload that touched a submission lock.
    Runs inference for any invernadero whose required-upload set just
    became complete. Callers must not invoke this for form_type ==
    "produccion" — production never gates inference."""
    invs_to_check = (3, 4) if form_type == "exteriores" else (inv,)
    for check_inv in invs_to_check:
        if not uploads_complete_for_inv(check_inv):
            continue
        try:
            from backend.supabase_client import service_client
            from scripts.live_inference import run_inference_for_greenhouse

            run_inference_for_greenhouse(service_client, check_inv, dry_run=False)
        except Exception as e:  # noqa: BLE001
            # Broad on purpose: a broken/missing optional dependency (e.g.
            # matplotlib, imported transitively by Models/cnn_rnn_yield.py)
            # must never surface as a 500 on the upload request — the
            # upload's data is already committed by the time this runs.
            print(
                f"  ERROR: upload-triggered inference failed for invernadero "
                f"{check_inv}: {e}",
                file=sys.stderr,
            )
