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


def test_get_last_predicted_wis_none_when_no_predictions_rows():
    from scripts.live_inference import get_last_predicted_wis

    mock_supa = MagicMock()
    resp = MagicMock()
    resp.data = []
    (mock_supa.table("predictions").select("predicted_for").eq("greenhouse_id", 3)
     .eq("season", "T18").order("predicted_for", desc=True).limit(1).execute.return_value) = resp

    harvest_start = date(2026, 7, 27)
    assert get_last_predicted_wis(mock_supa, 3, harvest_start) is None


def test_get_last_predicted_wis_returns_wis_of_latest_row():
    from scripts.live_inference import get_last_predicted_wis

    mock_supa = MagicMock()
    resp = MagicMock()
    harvest_start = date(2026, 7, 27)  # wis 0
    resp.data = [{"predicted_for": (harvest_start + timedelta(weeks=3)).isoformat()}]
    (mock_supa.table("predictions").select("predicted_for").eq("greenhouse_id", 3)
     .eq("season", "T18").order("predicted_for", desc=True).limit(1).execute.return_value) = resp

    assert get_last_predicted_wis(mock_supa, 3, harvest_start) == 3


def test_run_inference_for_greenhouse_starts_from_zero_when_no_predictions_yet(monkeypatch):
    from scripts.live_inference import run_inference_for_greenhouse, HORIZON_WEEKS

    mock_supa = MagicMock()
    transplant_date = date(2026, 5, 15)
    harvest_start = date(2026, 7, 27)  # week 31

    monkeypatch.setattr(
        "scripts.live_inference.get_transplant_date", lambda supa, inv: transplant_date
    )
    monkeypatch.setattr(
        "scripts.live_inference.get_harvest_start", lambda supa, inv: harvest_start
    )
    monkeypatch.setattr(
        "scripts.live_inference.get_last_predicted_wis", lambda supa, inv, hs: None
    )

    calls = []

    def fake_run_model_forward(supa, inv, td, as_of_date, wis_target):
        calls.append(as_of_date)
        # First model call succeeds (enough backlog), second is padded (not enough).
        return 5000.0 if len(calls) == 1 else None

    monkeypatch.setattr("scripts.live_inference._run_model_forward", fake_run_model_forward)

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is True

    upsert_calls = mock_supa.table("predictions").upsert.call_args_list
    # Exactly 1 model row (the 2nd model call returned None and stopped the loop)
    assert len(upsert_calls) == 1
    model_versions = [c.args[0]["model_version"] for c in upsert_calls]
    assert model_versions == ["cnn_rnn_v2_production"]
    # First model call's as_of_date = harvest_start - HORIZON_WEEKS weeks (wis starts at 0)
    assert calls[0] == harvest_start - timedelta(weeks=HORIZON_WEEKS)


def test_run_inference_for_greenhouse_resumes_after_last_predicted_wis(monkeypatch):
    from scripts.live_inference import run_inference_for_greenhouse, HORIZON_WEEKS

    mock_supa = MagicMock()
    transplant_date = date(2026, 5, 15)
    harvest_start = date(2026, 7, 27)  # week 31

    monkeypatch.setattr(
        "scripts.live_inference.get_transplant_date", lambda supa, inv: transplant_date
    )
    monkeypatch.setattr(
        "scripts.live_inference.get_harvest_start", lambda supa, inv: harvest_start
    )
    # Last prediction already written was for week-in-season 4 — resume at wis 5,
    # regardless of what date.today() is.
    monkeypatch.setattr(
        "scripts.live_inference.get_last_predicted_wis", lambda supa, inv, hs: 4
    )

    calls = []

    def fake_run_model_forward(supa, inv, td, as_of_date, wis_target):
        calls.append(wis_target)
        return None  # no new data yet — nothing to predict beyond wis 4

    monkeypatch.setattr("scripts.live_inference._run_model_forward", fake_run_model_forward)

    result = run_inference_for_greenhouse(mock_supa, 3, dry_run=False)
    assert result is False
    assert calls == [5]
    mock_supa.table("predictions").upsert.assert_not_called()


def test_get_harvest_start_normalizes_non_monday_to_monday():
    from scripts.live_inference import get_harvest_start

    mock_supa = MagicMock()
    resp = MagicMock()
    # 2026-07-29 is a Wednesday; the Monday of that week is 2026-07-27.
    resp.data = {"greenhouse_id": 3, "fecha": "2026-07-29", "updated_at": "2026-07-29T08:00:00Z"}
    mock_supa.table("harvest_start_dates").select("fecha").eq("greenhouse_id", 3).maybe_single().execute.return_value = resp

    result = get_harvest_start(mock_supa, 3)
    assert result == date(2026, 7, 27)


def test_run_model_forward_returns_none_when_not_enough_weekly_history():
    from scripts.live_inference import _run_model_forward

    mock_supa = MagicMock()
    # Only one day of sensor data merged by fecha -> aggregates to far fewer
    # than seq_len weekly rows, so the padding guard must fire before any
    # attempt to build tensors or load a model.
    sensor_resp = MagicMock()
    sensor_resp.data = [{"fecha": "2026-07-20", "greenhouse_id": 3, "temp": 20.0}]
    empty_resp = MagicMock()
    empty_resp.data = []

    def table_side_effect(name):
        m = MagicMock()
        if name == "sensor_readings_wide":
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = sensor_resp
        else:
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
            m.select.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
        return m

    mock_supa.table.side_effect = table_side_effect

    with patch(
        "scripts.live_inference.joblib.load",
        return_value={"seq_len": 4, "sensor_cols": ["temp"]},
    ):
        result = _run_model_forward(mock_supa, 3, date(2026, 5, 15), date(2026, 7, 27), 4)

    assert result is None


def test_run_model_forward_returns_none_when_latest_real_data_is_stale():
    from scripts.live_inference import _run_model_forward

    mock_supa = MagicMock()
    # Plenty of real daily rows (would satisfy any seq_len), but every one of
    # them is from over a week before as_of_date — e.g. as_of_date has kept
    # advancing week over week while uploads stopped. The recency guard must
    # fire before the seq_len/weekly-count check even runs, so this must
    # short-circuit with no need to mock joblib.load / pipeline files at all.
    sensor_resp = MagicMock()
    sensor_resp.data = [
        {"fecha": f"2026-06-{d:02d}", "greenhouse_id": 3, "temp": 20.0} for d in range(1, 29)
    ]
    empty_resp = MagicMock()
    empty_resp.data = []

    def table_side_effect(name):
        m = MagicMock()
        if name == "sensor_readings_wide":
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = sensor_resp
        else:
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
            m.select.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
        return m

    mock_supa.table.side_effect = table_side_effect

    # as_of_date is over a month past the latest real row (2026-06-28).
    result = _run_model_forward(mock_supa, 3, date(2026, 5, 15), date(2026, 8, 15), 4)

    assert result is None


def test_run_model_forward_proceeds_past_guards_when_data_is_recent(monkeypatch):
    from scripts.live_inference import _run_model_forward

    mock_supa = MagicMock()
    as_of_date = date(2026, 8, 10)
    sensor_resp = MagicMock()
    sensor_resp.data = [
        {"fecha": f"2026-07-{d:02d}", "greenhouse_id": 3, "temp": 20.0} for d in range(14, 32)
    ] + [{"fecha": f"2026-08-{d:02d}", "greenhouse_id": 3, "temp": 20.0} for d in range(1, 11)]
    empty_resp = MagicMock()
    empty_resp.data = []

    def table_side_effect(name):
        m = MagicMock()
        if name == "sensor_readings_wide":
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = sensor_resp
        else:
            m.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
            m.select.return_value.gte.return_value.lte.return_value.execute.return_value = empty_resp
        return m

    mock_supa.table.side_effect = table_side_effect

    sentinel = RuntimeError("reached model stage — guards did not block")

    def fake_build_input_tensor(*a, **k):
        raise sentinel

    monkeypatch.setattr("scripts.live_inference.build_input_tensor", fake_build_input_tensor)

    with patch(
        "scripts.live_inference.joblib.load",
        return_value={"seq_len": 2, "sensor_cols": ["temp"]},
    ):
        try:
            _run_model_forward(mock_supa, 3, date(2026, 5, 15), as_of_date, 4)
            assert False, "expected the sentinel RuntimeError to propagate"
        except RuntimeError as e:
            assert e is sentinel
