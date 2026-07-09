import os
import sys
import types
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")

# ── Inject stub modules BEFORE backend.main is imported ────────────────────
# engine.py depends on numpy/pandas which aren't in the test venv.
# Stub the module and the ENGINE singleton so the import succeeds.
_engine_mock = MagicMock()
_engine_mock.warm.return_value = None
_engine_mock.loaded.return_value = [3, 4]
_engine_mock.payload.return_value = {"inv_id": 3, "weeks": [], "y_true": [], "y_pred": []}
_engine_mock.metrics.return_value = {"inv_id": 3, "source": "npz_cache", "best": {}, "cptc_pi": {}}
_engine_mock.window.return_value = {"inv_id": 3, "cursor": 0, "history": [], "predictions": []}

_engine_module = types.ModuleType("engine")
_engine_module.ENGINE = _engine_mock
sys.modules.setdefault("engine", _engine_module)

# data_api.py reads Excel files — stub it too
_data_api_mock = MagicMock()
_data_api_mock.phenology_weekly.return_value = {
    "inv_id": 3, "fields": [], "seasons": [], "latest": {}, "latest_season": "T17"
}
_data_api_mock.sensor_history.return_value = {
    "inv_id": 3, "var": "temp", "meta": {}, "months": [], "series": []
}
_data_api_mock.validate_and_average.return_value = {"n_plants": 1, "averages": {}}
sys.modules.setdefault("data_api", _data_api_mock)


@pytest.fixture(autouse=True)
def mock_engine(monkeypatch):
    monkeypatch.setattr("backend.main.ENGINE", _engine_mock)
    return _engine_mock


@pytest.fixture(autouse=True)
def mock_data_api(monkeypatch):
    monkeypatch.setattr("backend.main.data_api", _data_api_mock)
    return _data_api_mock


@pytest.fixture
def mock_supa(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("backend.supabase_client.service_client", mock)
    monkeypatch.setattr("backend.auth.service_client", mock)
    monkeypatch.setattr("backend.routers.admin.service_client", mock)
    monkeypatch.setattr("backend.routers.predictions.service_client", mock)
    monkeypatch.setattr("backend.routers.phenology_live.service_client", mock)
    monkeypatch.setattr("backend.routers.uploads.service_client", mock)
    monkeypatch.setattr("backend.lock_utils.service_client", mock)
    return mock


@pytest.fixture
def api_client(mock_supa):
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def make_user(user_id="uid-1"):
    """Build mock Supabase get_user + profile responses."""
    mock_user = MagicMock()
    mock_user.id = user_id
    profile_data = {"disabled": False, "full_name": "Test User"}
    return mock_user, profile_data
