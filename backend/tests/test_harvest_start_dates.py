from unittest.mock import MagicMock


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_get_harvest_start_date_requires_auth(api_client):
    r = api_client.get("/harvest-start-date/3")
    assert r.status_code == 401


def test_get_harvest_start_date_returns_row(api_client, mock_supa):
    headers = _auth(mock_supa)
    calls = {"n": 0}

    def execute_data():
        calls["n"] += 1
        # 1st call = auth profile lookup, 2nd call = harvest_start_date lookup
        if calls["n"] == 1:
            return MagicMock(data={"disabled": False, "full_name": "Test User"})
        else:
            return MagicMock(data={"greenhouse_id": 3, "fecha": "2026-07-27", "updated_at": "2026-07-27T08:00:00Z"})

    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = execute_data
    r = api_client.get("/harvest-start-date/3", headers=headers)
    assert r.status_code == 200
    assert r.json()["fecha"] == "2026-07-27"


def test_get_harvest_start_date_404_when_unset(api_client, mock_supa):
    headers = _auth(mock_supa)
    calls = {"n": 0}

    def maybe_single_data():
        calls["n"] += 1
        return {"disabled": False, "full_name": "Test User"} if calls["n"] == 1 else None

    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = (
        lambda: MagicMock(data=maybe_single_data())
    )
    r = api_client.get("/harvest-start-date/3", headers=headers)
    assert r.status_code == 404


def test_put_harvest_start_date_requires_auth(api_client):
    r = api_client.put("/harvest-start-date/3", json={"fecha": "2026-07-27"})
    assert r.status_code == 401


def test_put_harvest_start_date_upserts(api_client, mock_supa, monkeypatch):
    from backend.routers import harvest_start_dates as hsd_router
    monkeypatch.setattr(hsd_router, "maybe_trigger_inference", MagicMock())
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.put("/harvest-start-date/3", headers=headers, json={"fecha": "2026-07-27"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_supa.table.assert_called_with("harvest_start_dates")
    mock_supa.table.return_value.upsert.assert_called_with(
        {"greenhouse_id": 3, "fecha": "2026-07-27"}, on_conflict="greenhouse_id"
    )


def test_put_harvest_start_date_triggers_inference_check(api_client, mock_supa, monkeypatch):
    from backend.routers import harvest_start_dates as hsd_router
    trigger_mock = MagicMock()
    monkeypatch.setattr(hsd_router, "maybe_trigger_inference", trigger_mock)
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.put("/harvest-start-date/4", headers=headers, json={"fecha": "2026-07-28"})
    assert r.status_code == 200
    trigger_mock.assert_called_once_with("harvest_start_date", 4)
