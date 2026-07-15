import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def test_monday_of_week():
    from scripts.live_inference import monday_of_week
    assert monday_of_week(date(2026, 7, 15)) == date(2026, 7, 13)  # Wednesday -> Monday
    assert monday_of_week(date(2026, 7, 13)) == date(2026, 7, 13)  # already Monday


def test_week_in_season():
    from scripts.live_inference import week_in_season
    transplant = date(2026, 5, 20)
    assert week_in_season(date(2026, 5, 20), transplant) == 0
    assert week_in_season(date(2026, 5, 27), transplant) == 1
    assert week_in_season(date(2026, 6, 24), transplant) == 5


def test_load_historical_mean_returns_value_for_ramp_up_week(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 1) == 150.0


def test_load_historical_mean_returns_none_outside_ramp_up(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 4) is None


def test_run_inference_for_greenhouse_returns_false_when_no_transplant_date():
    from scripts.live_inference import run_inference_for_greenhouse

    mock_supa = MagicMock()
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value.data = None

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False


def test_run_inference_for_greenhouse_returns_false_when_no_harvest_start():
    from scripts.live_inference import run_inference_for_greenhouse

    mock_supa = MagicMock()
    transplant_response = MagicMock()
    transplant_response.data = {"fecha": "2026-05-15"}
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

    harvest_response = MagicMock()
    harvest_response.data = None
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = harvest_response

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False
    mock_supa.table("predictions").upsert.assert_not_called()


def test_get_harvest_start_reads_harvest_start_dates_table():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = {"greenhouse_id": 3, "fecha": "2026-07-27", "updated_at": "2026-07-27T08:00:00Z"}
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = resp

    result = get_harvest_start(mock_supa, 3)
    assert result == date(2026, 7, 27)


def test_get_harvest_start_returns_none_when_unset():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = None
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = resp

    result = get_harvest_start(mock_supa, 3)
    assert result is None


def test_is_first_submission_true_when_no_predictions_rows():
    from scripts.live_inference import is_first_submission

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = []
    mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = resp

    assert is_first_submission(mock_supa, 3) is True


def test_is_first_submission_false_when_predictions_rows_exist():
    from scripts.live_inference import is_first_submission

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = [{"id": 1}]
    mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = resp

    assert is_first_submission(mock_supa, 3) is False


def test_run_inference_for_greenhouse_non_first_submission_uses_historical_mean(tmp_path):
    from scripts.live_inference import run_inference_for_greenhouse

    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0, "3": 250.0}))

    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        mock_supa = MagicMock()
        transplant_date = date.today() - timedelta(days=5)
        # harvest_start far enough in the future that wis_target < SKIP_FIRST_WEEKS (4)
        harvest_start = date.today() + timedelta(days=20)

        transplant_response = MagicMock()
        transplant_response.data = {"fecha": transplant_date.isoformat()}
        mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

        harvest_response = MagicMock()
        harvest_response.data = {"fecha": harvest_start.isoformat()}
        mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = harvest_response

        predictions_select_resp = MagicMock()
        predictions_select_resp.data = [{"id": 1}]  # not first submission
        mock_supa.table("predictions").select("id").eq("greenhouse_id", 3).eq("season", "T18").limit(1).execute.return_value = predictions_select_resp

        result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
        assert result is True
        mock_supa.table("predictions").upsert.assert_called_once()
        upsert_call = mock_supa.table("predictions").upsert.call_args[0][0]
        assert upsert_call["model_version"] == "historical_mean_v1"


def test_backfill_first_submission_writes_mean_weeks_and_stops_at_padding(monkeypatch):
    from scripts.live_inference import _backfill_first_submission, HORIZON_WEEKS, SKIP_FIRST_WEEKS

    mock_supa = MagicMock()
    transplant_date = date(2026, 5, 15)
    harvest_start = date(2026, 7, 27)  # week 31

    monkeypatch.setattr(
        "scripts.live_inference.load_historical_mean",
        lambda inv, wis: {0: 100.0, 1: 150.0, 2: 200.0, 3: 250.0}.get(wis),
    )

    calls = []

    def fake_run_model_forward(supa, inv, td, as_of_date):
        calls.append(as_of_date)
        # First model call succeeds (enough backlog), second is padded (not enough).
        return 5000.0 if len(calls) == 1 else None

    monkeypatch.setattr("scripts.live_inference._run_model_forward", fake_run_model_forward)
    # Freeze "today" far enough ahead that the loop's ceiling isn't what stops it —
    # the padding check (fake_run_model_forward returning None) should stop it instead.
    monkeypatch.setattr("scripts.live_inference.date", _FakeDate)

    result = _backfill_first_submission(mock_supa, 3, transplant_date, harvest_start, dry_run=False)
    assert result is True

    upsert_calls = mock_supa.table("predictions").upsert.call_args_list
    # SKIP_FIRST_WEEKS (4) mean rows + exactly 1 model row (the 2nd model call returned None and stopped the loop)
    assert len(upsert_calls) == SKIP_FIRST_WEEKS + 1
    model_versions = [c.args[0]["model_version"] for c in upsert_calls]
    assert model_versions[:SKIP_FIRST_WEEKS] == ["historical_mean_v1"] * SKIP_FIRST_WEEKS
    assert model_versions[SKIP_FIRST_WEEKS] == "cnn_rnn_v2_production"
    # First model call's as_of_date = harvest_start + (SKIP_FIRST_WEEKS - HORIZON_WEEKS) weeks
    assert calls[0] == harvest_start + timedelta(weeks=SKIP_FIRST_WEEKS - HORIZON_WEEKS)


class _FakeDate(date):
    @classmethod
    def today(cls):
        return date(2026, 12, 31)  # far enough ahead that the ceiling never triggers first
