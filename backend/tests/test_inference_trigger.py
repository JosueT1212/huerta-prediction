from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock


def _stub_locks(monkeypatch, times: dict):
    """times: {(greenhouse_id, form_type): iso_str_or_None}"""
    from backend import inference_trigger

    def fake_last_submitted_at(gh_id, form_type):
        iso = times.get((gh_id, form_type))
        return datetime.fromisoformat(iso) if iso else None

    monkeypatch.setattr(inference_trigger, "last_submitted_at", fake_last_submitted_at)


def test_uploads_complete_for_inv_true_when_all_four_recent(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): now, (3, "riego"): now, (3, "fenologia"): now,
        (0, "exteriores"): now,
    })
    assert uploads_complete_for_inv(3) is True


def test_uploads_complete_for_inv_false_when_one_missing(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): now, (3, "riego"): now, (3, "fenologia"): None,
        (0, "exteriores"): now,
    })
    assert uploads_complete_for_inv(3) is False


def test_uploads_complete_for_inv_false_when_expired(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc)
    monday = datetime.combine(
        now.date() - timedelta(days=now.weekday()), time.min, tzinfo=timezone.utc
    )
    before_monday = (monday - timedelta(hours=1)).isoformat()
    _stub_locks(monkeypatch, {
        (3, "sensores"): before_monday, (3, "riego"): now.isoformat(),
        (3, "fenologia"): now.isoformat(), (0, "exteriores"): now.isoformat(),
    })
    assert uploads_complete_for_inv(3) is False


def test_uploads_complete_for_inv_true_at_monday_boundary(monkeypatch):
    from backend.inference_trigger import uploads_complete_for_inv
    now = datetime.now(timezone.utc)
    monday = datetime.combine(
        now.date() - timedelta(days=now.weekday()), time.min, tzinfo=timezone.utc
    )
    _stub_locks(monkeypatch, {
        (3, "sensores"): monday.isoformat(), (3, "riego"): now.isoformat(),
        (3, "fenologia"): now.isoformat(), (0, "exteriores"): now.isoformat(),
    })
    assert uploads_complete_for_inv(3) is True


def test_maybe_trigger_inference_runs_when_complete(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: True)
    run_mock = MagicMock(return_value=True)
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("fenologia", 3)

    run_mock.assert_called_once()
    assert run_mock.call_args[0][1] == 3
    assert run_mock.call_args[1]["dry_run"] is False


def test_maybe_trigger_inference_skips_when_incomplete(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: False)
    run_mock = MagicMock()
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("sensores", 3)

    run_mock.assert_not_called()


def test_maybe_trigger_inference_checks_both_invs_for_exteriores(monkeypatch):
    from backend import inference_trigger
    checked = []
    monkeypatch.setattr(
        inference_trigger, "uploads_complete_for_inv",
        lambda inv: checked.append(inv) or True,
    )
    run_mock = MagicMock()
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("exteriores", None)

    assert checked == [3, 4]
    assert run_mock.call_count == 2


def test_maybe_trigger_inference_swallows_inference_errors(monkeypatch):
    from backend import inference_trigger
    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: True)
    run_mock = MagicMock(side_effect=RuntimeError("boom"))
    import scripts.live_inference as live_inference
    monkeypatch.setattr(live_inference, "run_inference_for_greenhouse", run_mock)

    inference_trigger.maybe_trigger_inference("riego", 4)  # must not raise


def test_maybe_trigger_inference_swallows_import_errors(monkeypatch):
    # Regression: a missing optional dependency (matplotlib, imported
    # transitively via Models/cnn_rnn_yield.py) must not crash the upload
    # request — the ModuleNotFoundError happens at import time, before the
    # run_inference_for_greenhouse call, and must still be caught.
    from backend import inference_trigger
    import builtins

    monkeypatch.setattr(inference_trigger, "uploads_complete_for_inv", lambda inv: True)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scripts.live_inference":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    inference_trigger.maybe_trigger_inference("sensores", 3)  # must not raise
