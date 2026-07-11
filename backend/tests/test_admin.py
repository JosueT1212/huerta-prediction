from unittest.mock import MagicMock


def _auth_headers(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def test_list_users_requires_auth(api_client):
    r = api_client.get("/admin/users")
    assert r.status_code == 401


def test_list_users_returns_profiles(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    mock_supa.table.return_value.select.return_value.execute.return_value.data = [
        {"id": "uid-1", "full_name": "Juan", "disabled": False, "created_at": "2026-01-01T00:00:00Z"}
    ]
    r = api_client.get("/admin/users", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_user_requires_auth(api_client):
    r = api_client.post("/admin/users", json={
        "email": "new@test.com", "full_name": "New User"
    })
    assert r.status_code == 401


def test_create_user_invites_and_inserts_profile(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    new_user = MagicMock()
    new_user.id = "new-uid"
    mock_supa.auth.admin.create_user.return_value.user = new_user
    mock_supa.table.return_value.upsert.return_value.execute.return_value = MagicMock()

    r = api_client.post("/admin/users", headers=headers, json={
        "email": "juan@huerta.com", "full_name": "Juan García", "password": "s3cret-pw"
    })
    assert r.status_code == 200
    assert r.json()["user_id"] == "new-uid"
    mock_supa.auth.admin.create_user.assert_called_once_with({
        "email": "juan@huerta.com", "password": "s3cret-pw", "email_confirm": True,
    })


def test_disable_user_requires_auth(api_client):
    r = api_client.patch("/admin/users/some-uid", json={"disabled": True})
    assert r.status_code == 401


def test_disable_user_updates_profile(api_client, mock_supa):
    headers = _auth_headers(mock_supa)
    mock_supa.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    r = api_client.patch("/admin/users/target-uid", headers=headers, json={"disabled": True})
    assert r.status_code == 200
    mock_supa.table.assert_called_with("profiles")
    mock_supa.table.return_value.update.assert_called_with({"disabled": True})
