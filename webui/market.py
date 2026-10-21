"""Binance research dashboard routes; no authenticated/order endpoints."""

from flask import Blueprint, jsonify, render_template, request

from market.binance import BinanceError, SUPPORTED_INTERVALS
from market.service import MarketService, integer_setting, settings

blueprint = Blueprint("market", __name__)
service = MarketService()


@blueprint.route("/market")
def dashboard():
    return render_template("market.html")


@blueprint.route("/api/market/symbols")
def symbols():
    return jsonify(symbols=service.client.symbols(), intervals=list(SUPPORTED_INTERVALS))


@blueprint.route("/api/market/candles")
def candles():
    config = settings(request.args.to_dict())
    return jsonify(candles=service.candles(config["symbol"], config["interval"], request.args.get("forecast_id")))


@blueprint.route("/api/market/forecasts")
def forecasts():
    return jsonify(forecasts=service.forecasts(request.args.get("symbol"), request.args.get("interval")))


@blueprint.route("/api/market/forecasts/<ident>")
def forecast(ident):
    return jsonify(service.read_forecast(ident))


@blueprint.route("/api/market/backtests/latest")
def latest_backtest():
    config = settings(request.args.to_dict())
    return jsonify(result=service.latest_backtest(config["symbol"], config["interval"]))


@blueprint.route("/api/market/<kind>", methods=["POST"])
def submit(kind):
    if kind not in ("forecast", "backtest"):
        return jsonify(error="Unknown research job."), 404
    body = request.get_json(silent=True)
    config = settings(body)
    days = integer_setting(body.get("days", 365), "days", 1, 730)
    return jsonify(job_id=service.submit(kind, config, days=days)), 202


@blueprint.route("/api/market/jobs/<ident>")
def job(ident):
    return jsonify(service.job(ident))


@blueprint.errorhandler(ValueError)
@blueprint.errorhandler(TypeError)
def invalid_request(error):
    return jsonify(error=str(error)), 400


@blueprint.errorhandler(FileNotFoundError)
def missing_result(error):
    return jsonify(error="Saved result not found."), 404


@blueprint.errorhandler(BinanceError)
def upstream_error(error):
    return jsonify(error=str(error)), 502
