from unittest.mock import MagicMock

SAMPLE_OBS = {
    "week_date": "2026-06-16",
    "zona": 1,
    "planta": 2,
    "racimos_puestos": 12.0,
    "flores_racimo_abiertas": 8.0,
    "racimos_en_planta": 5.0,
    "cantidad_tomates": 30.0,
    "racimo_en_cosecha": 2.0,
    "tomates_maduros": 10.0,
    "diametro_fruto_cm": 5.5,
    "crecimiento_planta_cm": 3.2,
}


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_post_phenology_requires_auth(api_client):
    r = api_client.post("/phenology-live/3", json=SAMPLE_OBS)
    assert r.status_code == 401


def test_post_phenology_upserts_observation(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    r = api_client.post("/phenology-live/3", headers=headers, json=SAMPLE_OBS)
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_supa.table.assert_called_with("phenology_observations")


def test_get_phenology_requires_auth(api_client):
    r = api_client.get("/phenology-live/3")
    assert r.status_code == 401


def test_get_phenology_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [SAMPLE_OBS]
    r = api_client.get("/phenology-live/3", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)
