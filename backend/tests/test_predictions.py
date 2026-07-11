from unittest.mock import MagicMock


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_list_predictions_requires_auth(api_client):
    r = api_client.get("/live-predictions/3")
    assert r.status_code == 401


def test_list_predictions_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {
            "id": 1,
            "greenhouse_id": 3,
            "predicted_for": "2026-07-28",
            "predicted_at": "2026-06-23T08:00:00Z",
            "kg_predicted": 240.0,
            "kg_actual": None,
            "model_version": "cnn_rnn_v1",
        }
    ]
    r = api_client.get("/live-predictions/3", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_patch_kg_actual_requires_auth(api_client):
    r = api_client.patch("/live-predictions/3/1", json={"kg_actual": 235.0})
    assert r.status_code == 401


def test_patch_kg_actual_updates_row(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock()
    r = api_client.patch("/live-predictions/3/1", headers=headers, json={"kg_actual": 235.0})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_supa.table.assert_called_with("predictions")
    mock_supa.table.return_value.update.assert_called_with({"kg_actual": 235.0})
