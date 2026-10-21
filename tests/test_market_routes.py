"""Flask API validation, job submission, and archive access without inference."""

from unittest.mock import Mock

from flask import Flask
import pandas as pd
import pytest

from market.binance import BinanceError
from market.service import MarketService, atomic_json
import webui.market as market_routes


IDENT = "a" * 32


@pytest.fixture
def api(monkeypatch):
    service = Mock()
    service.submit.return_value = IDENT
    service.job.return_value = {"id": IDENT, "status": "queued", "progress": 0}
    service.client.symbols.return_value = ["BTCUSDT", "ETHUSDT"]
    service.candles.return_value = [{"time": 1735689600, "open": 100, "high": 102, "low": 99, "close": 101}]
    service.forecasts.return_value = []
    monkeypatch.setattr(market_routes, "service", service)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(market_routes.blueprint)
    return app.test_client(), service


@pytest.mark.parametrize("kind", ["forecast", "backtest"])
def test_submit_returns_202_job_id_and_status_endpoint(api, kind):
    client, service = api
    response = client.post(f"/api/market/{kind}", json={"symbol": "eth/usdt", "interval": "4h", "sample_count": 100, "days": 365})
    assert response.status_code == 202
    assert response.get_json() == {"job_id": IDENT}
    service.submit.assert_called_once_with(kind, {
        "symbol": "ETHUSDT", "interval": "4h", "sample_count": 100,
        "fee_bps": 10.0, "slippage_bps": 5.0,
    }, days=365)
    status = client.get(f"/api/market/jobs/{IDENT}")
    assert status.status_code == 200
    assert status.get_json()["status"] == "queued"


@pytest.mark.parametrize("body", [
    None, [], {"symbol": ""}, {"interval": "7m"},
    {"sample_count": 0}, {"sample_count": 501}, {"sample_count": True},
    {"sample_count": 2.5}, {"sample_count": float("inf")},
    {"fee_bps": -1}, {"fee_bps": float("nan")},
    {"slippage_bps": 101}, {"days": 0}, {"days": 731},
    {"days": True}, {"days": 2.5}, {"days": float("inf")},
])
def test_invalid_arguments_are_json_400_and_never_submit_work(api, body):
    client, service = api
    response = client.post("/api/market/forecast", json=body)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)
    service.submit.assert_not_called()


def test_unknown_job_kind_is_404(api):
    client, service = api
    assert client.post("/api/market/trade", json={}).status_code == 404
    service.submit.assert_not_called()


def test_symbols_candles_and_archive_filters(api):
    client, service = api
    response = client.get("/api/market/symbols")
    assert response.status_code == 200
    assert response.get_json()["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert "1M" in response.get_json()["intervals"]
    response = client.get(f"/api/market/candles?symbol=ETHUSDT&interval=4h&forecast_id={IDENT}")
    assert response.status_code == 200
    assert len(response.get_json()["candles"]) == 1
    service.candles.assert_called_once_with("ETHUSDT", "4h", IDENT)
    response = client.get("/api/market/forecasts?symbol=ETHUSDT&interval=4h")
    assert response.get_json() == {"forecasts": []}
    service.forecasts.assert_called_once_with("ETHUSDT", "4h")


def test_live_candle_errors_are_actionable_json_and_invalid_interval_is_400(api):
    client, service = api
    service.candles.side_effect = BinanceError("Binance public API HTTP 451: unavailable in this region")
    response = client.get("/api/market/candles?symbol=BTCUSDT&interval=1h")
    assert response.status_code == 502
    assert "451" in response.get_json()["error"]
    assert client.get("/api/market/candles?interval=7m").status_code == 400


def real_archive_service(tmp_path, monkeypatch):
    service = MarketService(directory=tmp_path, client=Mock(), engine=Mock())
    service.client.fetch_candles.return_value = pd.DataFrame({
        "timestamps": [pd.Timestamp("2025-01-01", tz="UTC")],
        "open": [100], "high": [102], "low": [99], "close": [101],
    })
    monkeypatch.setattr(market_routes, "service", service)
    return service


def test_archive_ids_cannot_escape_result_directory(api, tmp_path, monkeypatch):
    client, _ = api
    service = real_archive_service(tmp_path, monkeypatch)
    try:
        for path in [
            "/api/market/forecasts/not-a-result-id",
            "/api/market/jobs/not-a-result-id",
            "/api/market/candles?forecast_id=../../README.md",
        ]:
            response = client.get(path)
            assert response.status_code == 400
            assert "Invalid result ID" in response.get_json()["error"]
        response = client.get("/api/market/forecasts/..%2F..%2FREADME.md")
        assert response.status_code == 404
        assert not list(tmp_path.rglob("*.json"))
    finally:
        service.executor.shutdown(wait=False)


def test_missing_saved_forecast_and_job_are_json_404(api, tmp_path, monkeypatch):
    client, _ = api
    service = real_archive_service(tmp_path, monkeypatch)
    try:
        for resource in ("forecasts", "jobs"):
            response = client.get(f"/api/market/{resource}/{IDENT}")
            assert response.status_code == 404
            assert response.get_json() == {"error": "Saved result not found."}
    finally:
        service.executor.shutdown(wait=False)


def test_saved_forecast_can_be_read_and_interrupted_job_reports_error(api, tmp_path, monkeypatch):
    client, _ = api
    service = real_archive_service(tmp_path, monkeypatch)
    try:
        result = {"id": IDENT, "symbol": "BTCUSDT", "sample_count": 100}
        service.archive(result)
        assert client.get(f"/api/market/forecasts/{IDENT}").get_json() == result
        atomic_json(service._path("jobs", IDENT), {"id": IDENT, "status": "running", "progress": .4})
        response = client.get(f"/api/market/jobs/{IDENT}")
        assert response.status_code == 200
        assert response.get_json()["status"] == "error"
        assert "Server stopped" in response.get_json()["error"]
    finally:
        service.executor.shutdown(wait=False)
