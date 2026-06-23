def test_missing_auth_header_returns_401(api_client):
    r = api_client.get("/me")
    assert r.status_code == 401


def test_malformed_header_returns_401(api_client):
    r = api_client.get("/me", headers={"Authorization": "NotBearer abc"})
    assert r.status_code == 401


def test_invalid_token_returns_401(api_client, mock_supa):
    from gotrue.errors import AuthApiError
    mock_supa.auth.get_user.side_effect = AuthApiError("invalid", 401, {})
    r = api_client.get("/me", headers={"Authorization": "Bearer bad-token"})
    assert r.status_code == 401


def test_disabled_user_returns_403(api_client, mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    profile["disabled"] = True
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 403


def test_valid_token_returns_user_info(api_client, mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    r = api_client.get("/me", headers={"Authorization": "Bearer valid-token"})
    assert r.status_code == 200
    assert r.json()["user_id"] == "uid-1"


def test_inference_requires_auth(api_client):
    r = api_client.get("/inference/3")
    assert r.status_code == 401


def test_metrics_requires_auth(api_client):
    r = api_client.get("/metrics/3")
    assert r.status_code == 401


def test_phenology_get_requires_auth(api_client):
    r = api_client.get("/phenology/3")
    assert r.status_code == 401


def test_phenology_post_requires_auth(api_client):
    r = api_client.post("/phenology/3", json={"rows": []})
    assert r.status_code == 401


def test_sensor_history_requires_auth(api_client):
    r = api_client.get("/sensor-history/3?var=temp")
    assert r.status_code == 401
