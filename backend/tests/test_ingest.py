from unittest.mock import MagicMock


def test_ingest_rejects_no_token(api_client):
    r = api_client.post("/ingest/sensors", json={
        "greenhouse_id": 3, "sensor_name": "temp_interior",
        "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z",
    })
    assert r.status_code == 401


def test_ingest_rejects_wrong_token(api_client):
    r = api_client.post("/ingest/sensors",
        json={"greenhouse_id": 3, "sensor_name": "temp_interior",
              "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_ingest_accepts_valid_token(api_client, mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    r = api_client.post("/ingest/sensors",
        json={"greenhouse_id": 3, "sensor_name": "temp_interior",
              "value": 24.5, "recorded_at": "2026-06-22T10:00:00Z"},
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_ingest_writes_to_supabase(api_client, mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.return_value = MagicMock()
    api_client.post("/ingest/sensors",
        json={"greenhouse_id": 4, "sensor_name": "hr_interior",
              "value": 65.0, "recorded_at": "2026-06-22T11:00:00Z"},
        headers={"Authorization": "Bearer test-ingest-token"},
    )
    mock_supa.table.assert_called_with("sensor_readings")
    inserted = mock_supa.table.return_value.insert.call_args[0][0]
    assert inserted["greenhouse_id"] == 4
    assert inserted["sensor_name"] == "hr_interior"
    assert inserted["value"] == 65.0
