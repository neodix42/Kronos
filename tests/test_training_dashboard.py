"""Dashboard comparisons use the same past-only context without editing data."""

import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
import yaml

from finetune_csv.dashboard import create_app, validated_frame


def candles(count=10):
    return pd.DataFrame(dict(timestamps=pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
                             open=np.arange(count) + 100., high=np.arange(count) + 103.,
                             low=np.arange(count) + 98., close=np.arange(count) + 101.,
                             volume=np.full(count, 10.), amount=np.full(count, 1000.)))


class FakeEngine:
    def __init__(self, model_path, tokenizer_path, **kwargs):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.kwargs = kwargs
        self.calls = []

    def sample(self, history, timestamps, *, progress, **sampling):
        self.calls.append((history.copy(), timestamps, sampling))
        progress(1.)
        values = history[["open", "high", "low", "close", "volume", "amount"]].iloc[-1].to_numpy()
        return np.broadcast_to(values, (sampling["sample_count"], len(timestamps), 6)).copy()

    def close(self):
        pass


@pytest.fixture
def api(tmp_path):
    source = tmp_path / "candles.csv"
    candles().to_csv(source, index=False)
    original = tmp_path / "original"
    tokenizer = tmp_path / "tokenizer"
    trained = tmp_path / "trained" / "basemodel" / "best_model"
    for path in (original, tokenizer, trained):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
    config = dict(
        data=dict(data_path=str(source), lookback_window=4, predict_window=2, max_context=8,
                  train_ratio=.7, val_ratio=.1, test_ratio=.2),
        training=dict(seed=42),
        model_paths=dict(pretrained_predictor=str(original), pretrained_tokenizer=str(tokenizer),
                         finetuned_tokenizer=str(tokenizer), base_save_path=str(tmp_path / "trained")),
        experiment=dict(exchange="bybit", category="spot", symbol="BTCUSDT", interval="1h",
                        train_tokenizer=False),
        evaluation=dict(sample_count=3, seed=17, temperature=.8, top_p=.95, top_k=5),
        device=dict(use_cuda=False),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    app = create_app(config_path, engine_factory=FakeEngine,
                     live_fetcher=lambda market, lookback: candles(12))
    app.config["TESTING"] = True
    service = app.extensions["training_dashboard"]
    yield app.test_client(), service
    service.close()


def completed_job(client, service, *, live=False):
    response = client.post("/api/forecast", json=dict(refresh_live=live))
    assert response.status_code == 202
    ident = response.get_json()["job_id"]
    service.executor.submit(lambda: None).result(timeout=5)
    return client.get(f"/api/jobs/{ident}").get_json()


def test_fixed_market_dashboard_and_snapshot_have_no_model_load(api):
    client, service = api
    page = client.get("/")
    assert page.status_code == 200
    assert b"Your trained model" in page.data
    assert b'id="interval"' not in page.data
    assert client.get("/static/vendor/lightweight-charts.standalone.production.js").status_code == 200
    status = client.get("/api/status").get_json()
    assert status["fixed_timeframe"] is True
    assert status["trained_ready"] is True
    assert status["market"] == dict(exchange="bybit", category="spot", symbol="BTCUSDT", interval="1h")
    data = client.get("/api/data").get_json()
    assert data["rows"] == 10
    assert data["candles"][-1]["close"] == 110.
    assert service.engines == {}


def test_missing_evaluation_does_not_fabricate_metrics(api):
    client, service = api
    body = client.get("/api/report").get_json()
    assert body["available"] is False
    assert "report" not in body
    service.report_directory.mkdir(parents=True)
    service.report_path.write_text(json.dumps(matching_report(service)))
    body = client.get("/api/report").get_json()
    assert body["available"] is True
    assert body["report"]["coverage"]["evaluated_windows"] == 3


def matching_report(service):
    return dict(source_csv_sha256=service._file_hash(service.data_path), sampling=service.sampling(),
                interval=service.market["interval"],
                execution_rules=dict(lookback=service.config.lookback_window, horizon=service.config.predict_window),
                coverage=dict(evaluated_windows=3), models={
                    kind: dict(checkpoint_hash=service._checkpoint_hash(checkpoint),
                               tokenizer_hash=service._checkpoint_hash(tokenizer))
                    for kind, checkpoint, tokenizer in (("original", service.original, service.original_tokenizer),
                                                        ("finetuned", service.checkpoint, service.trained_tokenizer))})


@pytest.mark.parametrize("change", ["data", "model", "tokenizer", "sampling"])
def test_stale_reports_are_hidden_after_experiment_inputs_change(api, change):
    client, service = api
    service.report_directory.mkdir(parents=True)
    service.report_path.write_text(json.dumps(matching_report(service)))
    if change == "data":
        with service.data_path.open("a") as stream:
            stream.write("\n")
    elif change == "model":
        (service.checkpoint / "weights").write_text("changed")
    elif change == "tokenizer":
        (service.trained_tokenizer / "config.json").write_text('{"changed": true}')
    else:
        service.evaluation["sample_count"] = 5
    result = client.get("/api/report").get_json()
    assert result["available"] is False
    assert result["stale"] is True
    assert "report" not in result


def test_snapshot_checks_frozen_csv_hash_before_inference(api):
    client, service = api
    service.settings["data"]["sha256"] = "0" * 64
    response = client.get("/api/data")
    assert response.status_code == 400
    assert "SHA256 mismatch" in response.get_json()["error"]
    job = completed_job(client, service)
    assert job["status"] == "failed"
    assert "SHA256 mismatch" in job["error"]
    assert not service.engines


def test_changed_checkpoint_is_reloaded_but_unchanged_original_is_reused(api):
    client, service = api
    completed_job(client, service)
    trained, original = service.engines["finetuned"], service.engines["original"]
    (service.checkpoint / "weights").write_text("new checkpoint")
    completed_job(client, service)
    assert trained is not service.engines["finetuned"]
    assert original is service.engines["original"]


def test_both_models_get_identical_past_context_and_sampling_and_are_cached(api):
    client, service = api
    before = service.data_path.read_bytes()
    job = completed_job(client, service)
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["context_length"] == 4
    assert result["horizon"] == 2
    assert len(result["models"]["finetuned"]["paths"]) == 3
    trained, original = service.engines["finetuned"], service.engines["original"]
    left, times, sampling = trained.calls[0]
    pd.testing.assert_frame_equal(left, original.calls[0][0])
    assert len(left) == 4
    assert left["timestamps"].iloc[-1] < times[0]
    assert sampling == dict(sample_count=3, seed=17, temperature=.8, top_p=.95, top_k=5)
    assert sampling == original.calls[0][2]
    assert trained.kwargs["device"] == "cpu"
    assert trained.tokenizer_path == original.tokenizer_path
    assert trained.model_path != original.model_path
    assert service.data_path.read_bytes() == before
    completed_job(client, service)
    assert trained is service.engines["finetuned"]
    assert len(trained.calls) == 2
    assert client.get("/api/latest-forecast").get_json()["forecast"]["source"] == "frozen_dataset"


def test_live_forecast_and_actual_refresh_never_modify_snapshot_or_report(api):
    client, service = api
    before = service.data_path.read_bytes()
    received = []

    def live(market, lookback):
        received.append((market, lookback))
        return candles(12)

    service.live_fetcher = live
    job = completed_job(client, service, live=True)
    assert job["status"] == "completed"
    assert job["result"]["source"] == "live"
    assert job["result"]["history"][-1]["close"] == 112.
    frozen_forecast = service.latest_forecast
    data = client.get("/api/data?source=live").get_json()
    assert data["rows"] == 12
    assert data["training_data_unchanged"] is True
    assert received == [(service.market, 4), (service.market, 4)]
    assert service.data_path.read_bytes() == before
    assert not service.report_path.exists()
    assert service.latest_forecast is frozen_forecast


@pytest.mark.parametrize("body", [None, [], {"refresh_live": 1}, {"refresh_live": "yes"}, {"interval": "1d"}, {"model": "other"}])
def test_invalid_forecast_requests_do_not_submit_work(api, body):
    client, service = api
    assert client.post("/api/forecast", json=body).status_code == 400
    assert service.jobs == {}


def test_cross_origin_forecast_is_rejected(api):
    client, service = api
    response = client.post("/api/forecast", json={}, headers={"Origin": "https://example.com"})
    assert response.status_code == 403


class FakePaperTrader:
    def __init__(self, service):
        self.service = service
        self.enabled = False
        self.calls = []

    def status(self):
        return dict(enabled=self.enabled, cash=10000., equity=10000., model="finetuned",
                    market=self.service.market, position=None, trades=[])

    def enable(self):
        self.calls.append("start")
        self.enabled = True

    def pause(self):
        self.calls.append("pause")
        self.enabled = False

    def close_position(self):
        self.calls.append("close")

    def close(self):
        self.calls.append("shutdown")


def test_paper_account_controls_use_trained_experiment_and_leave_data_untouched(api):
    client, service = api
    service.paper_factory = FakePaperTrader
    before = service.data_path.read_bytes()
    assert service.paper_trader is None
    account = client.get("/api/paper/status").get_json()
    assert account["cash"] == 10000. and account["enabled"] is False
    assert account["model"] == "finetuned" and account["market"] == service.market
    assert client.post("/api/paper/start", json={}).get_json()["enabled"] is True
    assert client.post("/api/paper/pause", json={}).get_json()["enabled"] is False
    assert client.post("/api/paper/close", json={}).status_code == 200
    assert service.paper_trader.calls == ["start", "pause", "close"]
    assert service.data_path.read_bytes() == before
    assert not service.report_path.exists() and service.engines == {}


@pytest.mark.parametrize("action,body", [("start", None), ("start", []), ("start", {"fee_bps": 0}),
                                        ("pause", {"enabled": False}), ("close", {"price": 1})])
def test_paper_controls_reject_price_and_rule_overrides_before_creating_account(api, action, body):
    client, service = api
    service.paper_factory = FakePaperTrader
    assert client.post("/api/paper/" + action, json=body).status_code == 400
    assert service.paper_trader is None


def test_paper_mutations_require_same_origin_and_unknown_actions_are_rejected(api):
    client, service = api
    service.paper_factory = FakePaperTrader
    assert client.post("/api/paper/start", json={}, headers={"Origin": "https://example.com"}).status_code == 403
    assert client.post("/api/paper/reset", json={}).status_code == 404
    assert service.paper_trader is None


def test_paper_close_errors_are_visible_and_do_not_change_enabled_state(api):
    client, service = api
    service.paper_factory = FakePaperTrader

    def cannot_close():
        raise RuntimeError("No fresh public quote is available")

    service.paper().close_position = cannot_close
    response = client.post("/api/paper/close", json={})
    assert response.status_code == 409
    assert "fresh public quote" in response.get_json()["error"]
    assert service.paper_trader.enabled is False
    assert service.jobs == {}


def test_running_forecast_cannot_queue_unbounded_gpu_jobs(api):
    client, service = api
    started, finish = threading.Event(), threading.Event()

    def blocking(*args):
        started.set()
        finish.wait(timeout=5)
        return candles(12)

    service.live_fetcher = blocking
    try:
        assert client.post("/api/forecast", json={"refresh_live": True}).status_code == 202
        assert started.wait(timeout=2)
        assert client.post("/api/forecast", json={}).status_code == 409
    finally:
        finish.set()
        service.executor.submit(lambda: None).result(timeout=5)


def test_gapped_or_unfinished_context_fails_before_loading_models(api):
    client, service = api
    frame = candles()
    frame = frame.drop(index=8)
    frame.to_csv(service.data_path, index=False)
    job = completed_job(client, service)
    assert job["status"] == "failed"
    assert "missing candles" in job["error"]
    assert not service.engines
    future = candles()
    future["timestamps"] = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h"), periods=10, freq="h")
    future.to_csv(service.data_path, index=False)
    job = completed_job(client, service)
    assert job["status"] == "failed"
    assert "unfinished" in job["error"]
    assert not service.engines


def test_job_and_source_validation_and_live_api_failure_are_json(api):
    client, service = api
    assert client.get("/api/jobs/not-a-job").status_code == 400
    assert client.get("/api/jobs/" + "a" * 32).status_code == 404
    assert client.get("/api/data?source=other").status_code == 400

    def fail(*args):
        raise RuntimeError("Public exchange API unavailable")

    service.live_fetcher = fail
    response = client.get("/api/data?source=live")
    assert response.status_code == 502
    assert "unavailable" in response.get_json()["error"]


@pytest.mark.parametrize("mutation", ["duplicate", "reversed", "nan", "bounds", "negative"])
def test_invalid_candles_are_rejected(mutation):
    frame = candles()
    if mutation == "duplicate":
        frame.loc[1, "timestamps"] = frame.loc[0, "timestamps"]
    elif mutation == "reversed":
        frame = frame.iloc[::-1]
    elif mutation == "nan":
        frame.loc[0, "volume"] = float("nan")
    elif mutation == "bounds":
        frame.loc[0, "high"] = 1.
    else:
        frame.loc[0, "close"] = -1.
    with pytest.raises(ValueError):
        validated_frame(frame)
