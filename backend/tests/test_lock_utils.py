from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException


def _mock_row(last_submitted_at_iso):
    row = MagicMock()
    row.data = [{"last_submitted_at": last_submitted_at_iso}] if last_submitted_at_iso else []
    return row


def test_check_lock_raises_when_within_7_days(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    with pytest.raises(HTTPException) as exc_info:
        lock_utils.check_submission_lock(3, "sensores")
    assert exc_info.value.status_code == 429
    assert "next_allowed_at" in exc_info.value.detail


def test_check_lock_passes_when_no_prior_submission(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.check_submission_lock(3, "sensores")  # should not raise


def test_check_lock_passes_when_older_than_7_days(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(old)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.check_submission_lock(3, "sensores")  # should not raise


def test_touch_lock_upserts(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    monkeypatch.setattr(lock_utils, "service_client", mock)

    lock_utils.touch_submission_lock(3, "sensores")
    mock.table.assert_called_with("submission_locks")
    mock.table.return_value.upsert.assert_called_once()
    kwargs = mock.table.return_value.upsert.call_args
    assert kwargs[0][0]["greenhouse_id"] == 3
    assert kwargs[0][0]["form_type"] == "sensores"
    assert kwargs[1]["on_conflict"] == "greenhouse_id,form_type"


def test_get_lock_status_locked(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    status = lock_utils.get_lock_status(3, "sensores")
    assert status["locked"] is True
    assert status["next_allowed_at"] is not None


def test_get_lock_status_unlocked_no_prior(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    status = lock_utils.get_lock_status(3, "sensores")
    assert status == {"locked": False, "next_allowed_at": None}


def test_get_lock_status_unlocked_when_older_than_7_days(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(old)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    status = lock_utils.get_lock_status(3, "sensores")
    assert status == {"locked": False, "next_allowed_at": None}


def test_last_submitted_at_returns_datetime_when_present(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(recent)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    result = lock_utils.last_submitted_at(3, "sensores")
    assert result is not None
    assert result.tzinfo is not None


def test_last_submitted_at_returns_none_when_absent(monkeypatch):
    from backend import lock_utils
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = _mock_row(None)
    monkeypatch.setattr(lock_utils, "service_client", mock)

    assert lock_utils.last_submitted_at(3, "sensores") is None
