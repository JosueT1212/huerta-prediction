import io
from unittest.mock import MagicMock
import pandas as pd
import pytest


def _auth(mock_supa):
    from backend.tests.conftest import make_user
    user, profile = make_user()
    mock_supa.auth.get_user.return_value.user = user
    # backend/auth.py calls .maybe_single(), not .single() — mock the actual
    # chain used, unlike the pre-existing (broken) helper in
    # test_phenology_live.py/test_predictions.py, which mocks .single() and
    # makes every authed request in those files 403 (see Task 0 baseline).
    mock_supa.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = profile
    return {"Authorization": "Bearer valid-token"}


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.read()


def _no_lock(mock_supa):
    # submission_locks select → no prior row → unlocked
    mock_supa.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []


def test_upload_requires_auth(api_client):
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 100}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", files=files)
    assert r.status_code == 401


def test_upload_rejects_invalid_form_type(api_client, mock_supa):
    headers = _auth(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01"}])
    files = {"file": ("x.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/bogus", headers=headers, files=files)
    assert r.status_code == 422


def test_upload_returns_429_when_locked(api_client, mock_supa):
    headers = _auth(mock_supa)
    from datetime import datetime, timezone
    mock_supa.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"last_submitted_at": datetime.now(timezone.utc).isoformat()}
    ]
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 100}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 429


def test_upload_rejects_missing_columns(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01"}])  # missing kg_reales
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 422
    assert "kg_reales" in r.text


def test_upload_sensores_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
        "temp_prom_ext": 20.0, "temp_max_ext": 26.0, "temp_min_ext": 15.0,
        "rad_sum": 300.0,
    }])
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_in_file"] == 1
    assert body["rows_inserted"] == 1
    mock_supa.table.assert_any_call("sensor_readings_wide")
    mock_supa.table.assert_any_call("submission_locks")


def test_upload_produccion_skips_unmatched_week_and_does_not_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    # predictions update returns no matched row (empty data) → skipped
    mock_supa.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    df = pd.DataFrame([{"fecha": "2026-07-01", "kg_reales": 5000}])
    files = {"file": ("prod.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/produccion", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_updated"] == 0
    assert body["rows_skipped"] == 1
    # all rows skipped → lock must NOT engage (nothing was actually saved)
    mock_supa.table.assert_any_call("predictions")
    assert mock_supa.table.call_args_list[-1] != (("submission_locks",),)


def test_upload_fenologia_rejects_non_numeric_zona(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "zona": "abc", "planta": 1,
        "racimos_puestos": 1, "flores_racimo_abiertas": 1,
        "racimos_en_planta": 1, "cantidad_tomates": 1, "racimo_en_cosecha": 1,
        "tomates_maduros": 1, "diametro_fruto_cm": 1.0, "crecimiento_planta_cm": 1.0,
    }])
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "zona" in r.text


def test_upload_sensores_handles_blank_optional_cell(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    df = pd.DataFrame([{
        "fecha": "2026-07-01", "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
        "temp_prom_ext": 20.0, "temp_max_ext": 26.0, "temp_min_ext": None,
        "rad_sum": 300.0,
    }])
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1


def test_history_requires_auth(api_client):
    r = api_client.get("/uploads/3/fenologia/history")
    assert r.status_code == 401


def test_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"week_date": "2026-07-01", "zona": 1, "planta": 1}
    ]
    r = api_client.get("/uploads/3/fenologia/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_lock_status_endpoint_requires_auth(api_client):
    r = api_client.get("/submission-lock/3/sensores")
    assert r.status_code == 401


def test_lock_status_endpoint_returns_status(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/3/sensores", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}
