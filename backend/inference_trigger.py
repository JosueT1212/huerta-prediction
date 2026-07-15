"""Fires live inference right after the last required weekly upload lands.

Each of the 4 required types (sensores/riego/fenologia per invernadero,
exteriores global) can only be submitted once per LOCK_WINDOW — a repeat
submission is rejected by check_submission_lock() with 429 before it ever
reaches this module. That means uploads_complete_for_inv() can only flip
from False to True on the exact upload call that completes the set; no
extra "was it already complete" bookkeeping is needed here.
"""
from datetime import datetime, timezone
import sys

from backend.lock_utils import LOCK_WINDOW, last_submitted_at

# Mirrors the sentinel in backend/routers/uploads.py (EXTERIORES_GH_ID) —
# exteriores has no real invernadero, 0 is never a real greenhouse id.
EXTERIORES_GH_ID = 0

REQUIRED_PER_INV_TYPES = ("sensores", "riego", "fenologia")


def uploads_complete_for_inv(inv: int) -> bool:
    """True if sensores+riego+fenologia (this inv) + exteriores (global)
    were all submitted within the current LOCK_WINDOW."""
    now = datetime.now(timezone.utc)
    for form_type in REQUIRED_PER_INV_TYPES:
        ts = last_submitted_at(inv, form_type)
        if ts is None or now - ts > LOCK_WINDOW:
            return False
    ts = last_submitted_at(EXTERIORES_GH_ID, "exteriores")
    if ts is None or now - ts > LOCK_WINDOW:
        return False
    return True


def maybe_trigger_inference(form_type: str, inv: int | None) -> None:
    """Call after a successful upload that touched a submission lock.
    Runs inference for any invernadero whose required-upload set just
    became complete. Callers must not invoke this for form_type ==
    "produccion" — production never gates inference."""
    from backend.supabase_client import service_client
    from scripts.live_inference import run_inference_for_greenhouse

    invs_to_check = (3, 4) if form_type == "exteriores" else (inv,)
    for check_inv in invs_to_check:
        if not uploads_complete_for_inv(check_inv):
            continue
        try:
            run_inference_for_greenhouse(service_client, check_inv, dry_run=False)
        except Exception as e:  # noqa: BLE001
            print(
                f"  ERROR: upload-triggered inference failed for invernadero "
                f"{check_inv}: {e}",
                file=sys.stderr,
            )
