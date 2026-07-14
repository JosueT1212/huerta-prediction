import json
from datetime import date
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
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 1) == 150.0


def test_load_historical_mean_returns_none_outside_ramp_up(tmp_path):
    from scripts.live_inference import load_historical_mean
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0}))
    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        assert load_historical_mean(3, 3) is None


def test_run_inference_for_greenhouse_returns_false_when_no_transplant_date():
    from scripts.live_inference import run_inference_for_greenhouse

    mock_supa = MagicMock()
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value.data = None

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False


def test_get_harvest_start_returns_earliest_fecha():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [{"fecha": "2026-06-01"}]
    (
        mock_supa.table("sensor_readings_wide")
        .select("fecha")
        .eq("greenhouse_id", 3)
        .gte("fecha", "2026-05-20")
        .order("fecha")
        .limit(1)
        .execute
    ).return_value = mock_response

    result = get_harvest_start(mock_supa, 3, date(2026, 5, 20))
    assert result == date(2026, 6, 1)


def test_get_harvest_start_returns_none_when_no_rows():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    mock_response = MagicMock()
    mock_response.data = []
    (
        mock_supa.table("sensor_readings_wide")
        .select("fecha")
        .eq("greenhouse_id", 3)
        .gte("fecha", "2026-05-20")
        .order("fecha")
        .limit(1)
        .execute
    ).return_value = mock_response

    result = get_harvest_start(mock_supa, 3, date(2026, 5, 20))
    assert result is None


def test_run_inference_for_greenhouse_returns_true_with_historical_mean(tmp_path, monkeypatch):
    from scripts.live_inference import run_inference_for_greenhouse, HORIZON_WEEKS
    from datetime import timedelta

    # Mock RESULTS_DIR and create historical mean file
    fake_json = tmp_path / "historical_mean_inv3.json"
    fake_json.write_text(json.dumps({"0": 100.0, "1": 150.0, "2": 200.0}))

    with patch("scripts.live_inference.RESULTS_DIR", tmp_path):
        # Mock Supabase to return a transplant date; harvest_start is what actually
        # drives wis_target now, so it (not transplant_date) is set such that
        # wis_target < 3.
        # predicted_for = today + HORIZON_WEEKS (5 weeks)
        # We want: (predicted_for - harvest_start).days // 7 < 3
        # So: harvest_start > predicted_for - 21 days = today + (5*7 - 21) days = today + 14 days
        mock_supa = MagicMock()
        transplant_date = date.today() - timedelta(days=5)  # season boundary, in the past
        harvest_start = date.today() + timedelta(days=20)  # ~3 weeks in the future
        transplant_response = MagicMock()
        transplant_response.data = {"fecha": transplant_date.isoformat()}
        mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

        harvest_response = MagicMock()
        harvest_response.data = [{"fecha": harvest_start.isoformat()}]
        (
            mock_supa.table("sensor_readings_wide")
            .select("fecha")
            .eq("greenhouse_id", 3)
            .gte("fecha", transplant_date.isoformat())
            .order("fecha")
            .limit(1)
            .execute
        ).return_value = harvest_response

        result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
        assert result is True

        # Verify that upsert was called (prediction was written)
        mock_supa.table("predictions").upsert.assert_called_once()


def test_run_inference_for_greenhouse_returns_false_when_no_harvest_start():
    from scripts.live_inference import run_inference_for_greenhouse
    from datetime import timedelta

    mock_supa = MagicMock()
    transplant_date = date.today() - timedelta(days=5)
    transplant_response = MagicMock()
    transplant_response.data = {"fecha": transplant_date.isoformat()}
    mock_supa.table("transplant_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = transplant_response

    harvest_response = MagicMock()
    harvest_response.data = []
    (
        mock_supa.table("sensor_readings_wide")
        .select("fecha")
        .eq("greenhouse_id", 3)
        .gte("fecha", transplant_date.isoformat())
        .order("fecha")
        .limit(1)
        .execute
    ).return_value = harvest_response

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False

    # No model/historical-mean work should happen — no prediction written
    mock_supa.table("predictions").upsert.assert_not_called()
