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


def _weekly_dates(n: int, start: str = "2026-03-02") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, periods=n, freq="7D")]


def _sensor_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "temp_prom_int": 22.1, "temp_min_int": 18.0,
        "temp_max_int": 27.0, "hr_prom_int": 65.0, "co2_ppm": 410.0,
        "deficit_humedad": 3.2, "deficit_presion_vapor": 0.8, "humedad_abs_int": 11.5,
    } for d in _weekly_dates(n, start)])


def _riego_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "riego_total": 120.0, "ph_promedio": 6.1, "ce_promedio": 2.3,
    } for d in _weekly_dates(n, start)])


def _exterior_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "temp_prom_ext": 20.0, "temp_max_ext": 26.0, "temp_min_ext": 15.0,
        "hr_prom_ext": 70.0, "rad_sum": 1800.0, "rad_max": 850.0, "dh_ext": 5.0,
        "humedad_abs_ext": 9.5,
    } for d in _weekly_dates(n, start)])


def _phenology_rows(n: int, start: str = "2026-03-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "fecha": d, "zona": 1, "planta": 1, "racimos_puestos": 1,
        "flores_racimo_abiertas": 1, "racimos_en_planta": 1, "cantidad_tomates": 1,
        "racimo_en_cosecha": 1, "tomates_maduros": 1, "diametro_fruto_cm": 1.0,
        "crecimiento_planta_cm": 1.0,
    } for d in _weekly_dates(n, start)])


def _first_submission(mock_supa, per_inv: bool = True):
    if per_inv:
        mock_supa.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    else:
        mock_supa.table.return_value.select.return_value.limit.return_value.execute.return_value.data = []


def _prior_submission(mock_supa, per_inv: bool = True):
    existing_row = [{"fecha": "2020-01-01"}]
    if per_inv:
        mock_supa.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = existing_row
    else:
        mock_supa.table.return_value.select.return_value.limit.return_value.execute.return_value.data = existing_row


def _matched_dates(mock_supa, dates: list[str], per_inv: bool = True, date_col: str = "fecha"):
    data = [{date_col: d} for d in dates]
    if per_inv:
        mock_supa.table.return_value.select.return_value.in_.return_value.eq.return_value.execute.return_value.data = data
    else:
        mock_supa.table.return_value.select.return_value.in_.return_value.execute.return_value.data = data


def test_upload_sensores_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(3)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text


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
    _first_submission(mock_supa)
    df = _sensor_rows(9)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_in_file"] == 9
    assert body["rows_inserted"] == 9
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
    _first_submission(mock_supa)
    df = _phenology_rows(9)  # inv3 requires 9 distinct weeks on first submission
    df["zona"] = df["zona"].astype(object)
    df.loc[df.index[-1], "zona"] = "abc"
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "zona" in r.text


def test_upload_sensores_handles_blank_optional_cell(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(9)  # inv3 requires 9 distinct weeks on first submission
    df.loc[df.index[-1], "humedad_abs_int"] = None
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9


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


def test_upload_riego_upserts_and_touches_lock(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _riego_rows(9)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("riego.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/riego", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9
    mock_supa.table.assert_any_call("riego_readings")
    mock_supa.table.assert_any_call("submission_locks")


def test_riego_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"greenhouse_id": 3, "fecha": "2026-07-01", "riego_total": 120.0}
    ]
    r = api_client.get("/uploads/3/riego/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_riego_lock_status_returns_status(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/3/riego", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}


def test_upload_exteriores_requires_auth(api_client):
    df = pd.DataFrame([{"fecha": "2026-07-01", "temp_prom_ext": 20.0}])
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", files=files)
    assert r.status_code == 401


def test_upload_exteriores_upserts_without_greenhouse_id(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa, per_inv=False)
    df = _exterior_rows(9)  # exteriores requires 9 distinct weeks on first submission
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 9
    mock_supa.table.assert_any_call("exterior_readings")
    # verify no greenhouse_id key was sent for the exterior_readings upsert
    upsert_call = next(
        c for c in mock_supa.table.return_value.upsert.call_args_list
    )
    assert "greenhouse_id" not in upsert_call[0][0]


def test_upload_exteriores_rejects_per_inv_route(api_client, mock_supa):
    headers = _auth(mock_supa)
    df = pd.DataFrame([{"fecha": "2026-07-01", "temp_prom_ext": 20.0}])
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/exteriores", headers=headers, files=files)
    assert r.status_code == 422


def test_exteriores_history_returns_list(api_client, mock_supa):
    headers = _auth(mock_supa)
    mock_supa.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"fecha": "2026-07-01", "temp_prom_ext": 20.0}
    ]
    r = api_client.get("/uploads/exteriores/history", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_exteriores_lock_status_uses_sentinel(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    r = api_client.get("/submission-lock/exteriores", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"locked": False, "next_allowed_at": None}
    mock_supa.table.return_value.select.return_value.eq.assert_any_call("greenhouse_id", 0)


def test_upload_sensores_subsequent_rejects_insufficient_new_days(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    _matched_dates(mock_supa, [])  # none of the uploaded dates are already in the DB
    df = _sensor_rows(3)  # 3 new days < 7 required
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 422
    assert "días nuevos" in r.text


def test_upload_exteriores_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa, per_inv=False)
    df = _exterior_rows(5)  # exteriores requires 9 distinct weeks
    files = {"file": ("ext.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/exteriores", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text


def test_upload_sensores_subsequent_excludes_already_present_dates(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    dates = _weekly_dates(8)
    _matched_dates(mock_supa, [dates[0]])  # 1 of the 8 uploaded dates already exists
    df = _sensor_rows(8)  # 8 uploaded - 1 already-present = 7 new days, exactly the minimum
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 8


def test_upload_fenologia_first_submission_rejects_insufficient_weeks(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _phenology_rows(3)  # inv3 requires 9 distinct weeks on first submission
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 422
    assert "semanas" in r.text


def test_upload_fenologia_subsequent_uses_week_date_column(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _prior_submission(mock_supa)
    dates = _weekly_dates(8)
    _matched_dates(mock_supa, [dates[0]], date_col="week_date")  # 1 of 8 already present
    df = _phenology_rows(8)  # 8 uploaded - 1 already-present = 7 new days, exactly the minimum
    files = {"file": ("fenologia.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/3/fenologia", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 8


def test_upload_sensores_unmapped_inv_skips_window_check(api_client, mock_supa):
    headers = _auth(mock_supa)
    _no_lock(mock_supa)
    _first_submission(mock_supa)
    df = _sensor_rows(1)  # inv 99 isn't in SEQ_LEN_BY_INV -> check is skipped, not a 500
    files = {"file": ("sensores.xlsx", _xlsx_bytes(df), "application/octet-stream")}
    r = api_client.post("/uploads/99/sensores", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows_inserted"] == 1
