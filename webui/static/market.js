/* Kronos market laboratory. Chart timestamps and displayed dates use UTC. */
'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { forecast: null, candles: [], chart: null, candleSeries: null, forecastSeries: [], markers: null, equityChart: null, activeJob: null, refreshTimer: null, marketRevision: 0, backtestVersion: 0, busy: false };
  const priceFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 8 });
  const colors = { green: '#91bea4', red: '#cb8992', cyan: '#7dc9df', muted: '#8e9bad', line: '#263447' };
  const intervalLabels = { '1m':'1 minute','3m':'3 minutes','5m':'5 minutes','15m':'15 minutes','30m':'30 minutes','1h':'1 hour','2h':'2 hours','4h':'4 hours','6h':'6 hours','8h':'8 hours','12h':'12 hours','1d':'1 day','3d':'3 days','1w':'1 week','1M':'1 month' };
  const isNumber = (n) => n !== null && n !== undefined && n !== '' && Number.isFinite(Number(n));
  const number = (n) => isNumber(n) ? Number(n) : null;
  const price = (n) => isNumber(n) ? priceFormatter.format(Number(n)) : '—';
  const percent = (n, signed = false) => isNumber(n) ? `${signed && Number(n) > 0 ? '+' : ''}${(Number(n) * 100).toFixed(1)}%` : '—';
  const date = (value, compact = false) => {
    if (value === undefined || value === null || value === '') return '—';
    const d = typeof value === 'number' ? new Date(value * (value > 1e12 ? 1 : 1000)) : new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toISOString().replace('T', ' ').slice(0, compact ? 10 : 16);
  };
  const time = (value) => typeof value === 'number' ? Math.floor(value > 1e12 ? value / 1000 : value) : Math.floor(new Date(value).getTime() / 1000);
  const marketKey = () => `${$('symbol').value.trim().toUpperCase()}|${$('interval').value}`;
  const market = () => ({ symbol: $('symbol').value.trim().toUpperCase(), interval: $('interval').value, sample_count: 100, fee_bps: Number($('fee').value), slippage_bps: Number($('slippage').value) });
  const error = (message) => { $('error').textContent = message || ''; $('error').hidden = !message; };
  const text = (id, value) => { $(id).textContent = value === null || value === undefined ? '—' : String(value); };
  const signedColor = (id, value) => {
    $(id).classList.toggle('text-positive', isNumber(value) && Number(value) > 0);
    $(id).classList.toggle('text-negative', isNumber(value) && Number(value) < 0);
  };
  const pretty = (value) => String(value || '').replace(/_/g, ' ').toLowerCase().replace(/^./, (c) => c.toUpperCase());

  async function request(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { 'Accept': 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
    let body;
    try { body = await response.json(); } catch (_) { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok || body.error) throw new Error(typeof body.error === 'string' ? body.error : body.message || `Request failed (${response.status}).`);
    return body;
  }

  function chartOptions(element) {
    return {
      autoSize: true,
      layout: { background: { type: 'solid', color: '#111923' }, textColor: '#8191a6', fontFamily: 'Inter, system-ui, sans-serif', fontSize: 10, attributionLogo: true },
      grid: { vertLines: { color: '#1b273530' }, horzLines: { color: '#25314166' } },
      rightPriceScale: { borderColor: '#253141', scaleMargins: { top: 0.12, bottom: 0.1 } },
      timeScale: { borderColor: '#253141', timeVisible: true, secondsVisible: false, rightOffset: 4, barSpacing: 7 },
      crosshair: { mode: 0, vertLine: { color: '#63778a', labelBackgroundColor: '#304256' }, horzLine: { color: '#63778a', labelBackgroundColor: '#304256' } },
      localization: { locale: 'en-US', timeFormatter: (t) => typeof t === 'number' ? date(t) + ' UTC' : String(t) },
    };
  }

  function initCharts() {
    if (!window.LightweightCharts) throw new Error('The chart library could not load. Restart the server after setup, then refresh this page.');
    state.chart = LightweightCharts.createChart($('price-chart'), chartOptions($('price-chart')));
    state.candleSeries = state.chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: colors.green, downColor: colors.red, wickUpColor: colors.green, wickDownColor: colors.red, borderVisible: false, lastValueVisible: true, priceLineVisible: false,
    });
    state.markers = LightweightCharts.createSeriesMarkers(state.candleSeries, []);
  }

  function normalizeCandles(items) {
    const map = new Map();
    (items || []).forEach((c) => {
      const t = time(c.time);
      if (Number.isFinite(t) && ['open','high','low','close'].every((k) => isNumber(c[k]))) map.set(t, { time: t, open: Number(c.open), high: Number(c.high), low: Number(c.low), close: Number(c.close) });
    });
    return [...map.values()].sort((a,b) => a.time-b.time);
  }

  function displayCandles(items, fit = false) {
    state.candles = normalizeCandles(items);
    state.candleSeries.setData(state.candles);
    const last = state.candles.at(-1);
    $('chart-empty').hidden = Boolean(last);
    const m = market();
    $('chart-heading').replaceChildren(document.createTextNode(m.symbol + ' '));
    const span = document.createElement('span'); span.className = 'market-interval'; span.textContent = `/ ${m.interval}`; $('chart-heading').append(span);
    text('chart-status', last ? `${state.candles.length} closed candles · latest open ${date(last.time)} UTC · refreshed ${new Date().toISOString().slice(11,19)} UTC` : 'No closed candles available');
    if (fit) fitPriceChart();
  }

  function fitPriceChart() {
    if (!state.candles.length) return;
    const context = state.forecast ? normalizeCandles(state.forecast.history).slice(-80) : state.candles.slice(-100);
    const future = state.forecast && state.forecast.quantiles ? state.forecast.quantiles.at(-1) : null;
    let end = future ? time(future.time) : state.candles.at(-1).time;
    if (future) {
      const qs = state.forecast.quantiles;
      const step = qs.length > 1 ? time(qs[1].time) - time(qs[0].time) : 86400;
      end = Math.max(end, Math.min(state.candles.at(-1).time, end + step * 12));
    }
    if (end > context[0].time) state.chart.timeScale().setVisibleRange({ from: context[0].time, to: end });
    else state.chart.timeScale().fitContent();
  }

  function clearProjection() {
    state.forecastSeries.forEach((series) => state.chart.removeSeries(series));
    state.forecastSeries = [];
    if (state.markers) state.markers.setMarkers([]);
  }

  function addLine(data, settings) {
    if (!data.length) return;
    const series = state.chart.addSeries(LightweightCharts.LineSeries, { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, ...settings });
    series.setData(data);
    state.forecastSeries.push(series);
  }

  function projection() {
    clearProjection();
    const f = state.forecast;
    if (!f) return;
    const h = normalizeCandles(f.history);
    const origin = h.at(-1);
    const start = origin ? [{ time: origin.time, value: origin.close }] : [];
    const paths = Array.isArray(f.paths) ? f.paths : [];
    paths.slice(0, Number($('path-count').value)).forEach((p) => {
      const points = (Array.isArray(p) ? p : p.points || []).map((v) => ({ time: time(v.time), value: Number(v.value ?? v.close) })).filter((v) => Number.isFinite(v.time) && Number.isFinite(v.value));
      addLine([...start, ...points], { color: 'rgba(125,201,223,0.14)', lineWidth: 1 });
    });
    const qs = f.quantiles || [];
    ['p10','p90','p50'].forEach((key) => {
      const points = qs.map((q) => ({ time: time(q.time), value: Number(q[key]) })).filter((v) => Number.isFinite(v.time) && Number.isFinite(v.value));
      addLine([...start, ...points], { color: key === 'p50' ? colors.cyan : '#678c9b', lineWidth: key === 'p50' ? 2 : 1, lineStyle: key === 'p50' ? 0 : 2, lastValueVisible: key === 'p50' });
    });
    if (origin) state.markers.setMarkers([{ time: origin.time, position: 'aboveBar', color: '#b5d5e3', shape: 'arrowDown', text: 'Forecast origin' }]);
  }

  function showForecast(f) {
    state.forecast = f;
    displayCandles([...(f.history || []), ...state.candles], false);
    projection(); fitPriceChart();
    const a = f.analysis || {};
    const regime = a.regime || 'UNAVAILABLE';
    const direction = a.direction || '—';
    text('regime-heading', regime);
    text('direction', direction);
    $('direction').className = 'badge ' + (/up|bull/i.test(direction) ? 'positive' : /down|bear/i.test(direction) ? 'negative' : 'neutral');
    text('regime-description', /trend/i.test(regime) ? 'Most sampled paths agree on a direction under the configured rule. Agreement can still be wrong.' : 'Sampled paths do not show enough directional agreement for a trend entry under the configured rule.');
    text('up-probability', percent(a.up_probability)); text('down-probability', percent(a.down_probability));
    $('probability-fill').style.width = `${Math.max(0, Math.min(1, number(a.up_probability) || 0)) * 100}%`;
    $('probability-fill').parentElement.dataset.ready = 'true';
    text('agreement', percent(a.agreement));
    text('median-return', percent(a.terminal_return_p50, true)); signedColor('median-return', a.terminal_return_p50);
    text('return-range', `${percent(a.terminal_return_p10, true)} to ${percent(a.terminal_return_p90, true)}`);
    const p = f.plan || {};
    text('plan-action', p.action || 'WAIT'); $('plan-action').className = 'badge ' + (p.action === 'BUY' ? 'positive' : 'neutral');
    text('entry-price', price(p.entry_estimate ?? p.entry));
    text('stop-price', price(p.stop_price ?? p.stop_loss));
    text('target-price', price(p.take_profit_price ?? p.take_profit));
    const reasons = Array.isArray(p.reasons) ? [...p.reasons] : p.reason ? [p.reason] : [];
    [p.entry_rule, p.exit_rule, p.stop_rule].filter(Boolean).forEach((r) => { if (!reasons.includes(r)) reasons.push(r); });
    $('plan-reasons').replaceChildren(...reasons.map((r) => { const li = document.createElement('li'); li.textContent = r; return li; }));
    text('plan-extra', `${isNumber(p.position_fraction) ? `Position: ${percent(p.position_fraction)} of equity. ` : ''}${isNumber(p.risk_fraction) ? `Planned stop risk: ${percent(p.risk_fraction)} of equity. ` : ''}Horizon: ${f.horizon || 24} candles. No orders are placed.`);
    text('forecast-origin', `Generated ${date(f.generated_at)} UTC · frozen after ${date(f.as_of ?? f.history?.at(-1)?.time)} UTC · ${f.sample_count || 100} paths`);
    $('saved-forecast').value = String(f.id || '');
  }

  async function loadSymbols() {
    const data = await request('/api/market/symbols');
    $('symbol-list').replaceChildren(...(data.symbols || []).map((s) => { const option = document.createElement('option'); option.value = typeof s === 'string' ? s : s.symbol; return option; }));
    if (Array.isArray(data.intervals) && data.intervals.length) {
      const selected = $('interval').value;
      $('interval').replaceChildren(...data.intervals.map((i) => { const option = document.createElement('option'); option.value = i; option.textContent = intervalLabels[i] || i; return option; }));
      $('interval').value = data.intervals.includes(selected) ? selected : data.intervals[0];
    }
  }

  async function loadCandles(fit = false, quiet = false) {
    const key = marketKey(), revision = state.marketRevision, m = market();
    $('refresh-button').disabled = true;
    try {
      const saved = state.forecast?.id ? `&forecast_id=${encodeURIComponent(state.forecast.id)}` : '';
      const data = await request(`/api/market/candles?symbol=${encodeURIComponent(m.symbol)}&interval=${encodeURIComponent(m.interval)}${saved}`);
      if (key !== marketKey() || revision !== state.marketRevision) return;
      const frozenHistory = state.forecast?.history || [];
      displayCandles([...frozenHistory, ...state.candles, ...(data.candles || [])], fit);
      if (!quiet) error('');
    } catch (e) { error(`Could not refresh candles: ${e.message}`); }
    finally { if (revision === state.marketRevision) $('refresh-button').disabled = false; }
  }

  async function loadSavedForecast(id) {
    const key = marketKey(), revision = state.marketRevision;
    const f = await request(`/api/market/forecasts/${encodeURIComponent(id)}`);
    if (key !== marketKey() || revision !== state.marketRevision || $('saved-forecast').value !== id) return;
    showForecast(f.forecast || f);
    await loadCandles(false, true);
  }

  async function loadArchive(selectId = null, loadLatest = false) {
    const key = marketKey(), revision = state.marketRevision, m = market();
    const data = await request(`/api/market/forecasts?symbol=${encodeURIComponent(m.symbol)}&interval=${encodeURIComponent(m.interval)}`);
    if (key !== marketKey() || revision !== state.marketRevision) return;
    const items = data.forecasts || [];
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = items.length ? 'Select a saved forecast…' : 'No saved forecasts for this market yet';
    $('saved-forecast').replaceChildren(placeholder, ...items.map((f) => {
      const option = document.createElement('option'); option.value = String(f.id); option.textContent = `${date(f.generated_at)} UTC · ${f.symbol} ${f.interval}`; return option;
    }));
    $('saved-forecast').value = String(selectId || state.forecast?.id || '');
    if (loadLatest && !state.forecast && items.length) {
      const id = String(items[0].id);
      $('saved-forecast').value = id;
      await loadSavedForecast(id);
    }
  }

  async function loadLatestBacktest() {
    const key = marketKey(), revision = state.marketRevision, version = state.backtestVersion, m = market();
    const data = await request(`/api/market/backtests/latest?symbol=${encodeURIComponent(m.symbol)}&interval=${encodeURIComponent(m.interval)}`);
    if (key === marketKey() && revision === state.marketRevision && version === state.backtestVersion && data.result) showBacktest(data.result);
  }

  function setBusy(value) {
    state.busy = value;
    ['forecast-button','backtest-button','symbol','interval','fee','slippage','saved-forecast'].forEach((id) => { $(id).disabled = value; });
  }

  function updateJob(job, kind) {
    text('job-label', `${kind === 'forecast' ? 'Sampling 100 possible futures' : 'Running walk-forward backtest'}${job.status === 'queued' ? ' · queued' : ''}`);
    const progress = Math.max(0, Math.min(1, number(job.progress) || 0));
    $('job-progress').value = progress; text('job-percentage', `${Math.round(progress * 100)}%`);
    text('job-message', job.message || (kind === 'forecast' ? 'Kronos-small is processing the latest 512 closed candles.' : 'Replaying historical forecast windows without future data.'));
  }

  async function runJob(kind) {
    if (state.busy || !$('market-form').reportValidity()) return;
    if (kind === 'backtest') state.backtestVersion += 1;
    error(''); setBusy(true); $('job-panel').hidden = false; updateJob({ progress: 0 }, kind);
    try {
      const payload = market(); if (kind === 'backtest') payload.days = 365;
      const created = await request(`/api/market/${kind}`, { method: 'POST', body: JSON.stringify(payload) });
      state.activeJob = created.job_id;
      if (!state.activeJob) throw new Error('The server did not return a job ID.');
      while (state.activeJob) {
        const job = await request(`/api/market/jobs/${encodeURIComponent(state.activeJob)}`);
        updateJob(job, kind);
        if (job.status === 'error' || job.status === 'failed') throw new Error(job.error || job.message || 'The job failed.');
        if (job.status === 'complete' || job.status === 'completed') {
          if (!job.result) throw new Error('The completed job has no result.');
          if (kind === 'forecast') { showForecast(job.result); await loadArchive(job.result.id); }
          else showBacktest(job.result);
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
      }
    } catch (e) { error(e.message); }
    finally { state.activeJob = null; setBusy(false); $('job-panel').hidden = true; }
  }

  function showBacktest(result) {
    const s = result.summary || {};
    $('backtest-empty').hidden = true; $('backtest-results').hidden = false;
    text('bt-return', percent(s.net_return, true)); signedColor('bt-return', s.net_return);
    text('bt-benchmark', percent(s.benchmark_return, true)); signedColor('bt-benchmark', s.benchmark_return);
    text('bt-drawdown', percent(s.max_drawdown)); text('bt-winrate', percent(s.win_rate));
    text('bt-trades', s.trades ?? (result.trades || []).length); text('bt-fees', price(s.total_fees));
    const coverage = result.coverage || {};
    const m = market();
    const rules = result.rules || {};
    text('backtest-coverage', `${result.symbol || m.symbol} · ${result.interval || m.interval} · ${date(coverage.first_evaluated_candle ?? coverage.start, true)} to ${date(coverage.last_evaluated_candle ?? coverage.end, true)} UTC${isNumber(coverage.evaluation_candles) ? ` · ${coverage.evaluation_candles} evaluated candles` : ''} · assumed fees ${rules.fee_bps_per_side ?? m.fee_bps} bps + slippage ${rules.slippage_bps_per_fill ?? m.slippage_bps} bps per side`);
    if (!state.equityChart) state.equityChart = LightweightCharts.createChart($('equity-chart'), chartOptions($('equity-chart')));
    if (state.equitySeries) state.equityChart.removeSeries(state.equitySeries);
    state.equitySeries = state.equityChart.addSeries(LightweightCharts.AreaSeries, { lineColor: '#a4e0bb', topColor: '#a4e0bb28', bottomColor: '#a4e0bb00', lineWidth: 2, priceLineVisible: false });
    const points = new Map();
    (result.equity_curve || []).forEach((p) => { const t = time(p.time), v = Number(p.equity ?? p.value); if (Number.isFinite(t) && Number.isFinite(v)) points.set(t, { time: t, value: v }); });
    state.equitySeries.setData([...points.values()].sort((a,b) => a.time-b.time)); state.equityChart.timeScale().fitContent();
    const trades = result.trades || [];
    $('trade-rows').replaceChildren(...trades.map((t) => {
      const row = document.createElement('tr');
      const pnl = t.pnl ?? t.net_pnl;
      const values = [date(t.entry_time), date(t.exit_time), price(t.entry_price), price(t.exit_price), price(pnl), pretty(t.exit_reason ?? t.reason)];
      values.forEach((v, i) => { const td = document.createElement('td'); td.textContent = v; if (i === 4 && isNumber(pnl)) td.className = Number(pnl) >= 0 ? 'text-positive' : 'text-negative'; row.append(td); }); return row;
    }));
    if (!trades.length) { const row = document.createElement('tr'), td = document.createElement('td'); td.colSpan = 6; td.textContent = 'No trades met the entry rules in this period.'; row.append(td); $('trade-rows').append(row); }
    let notes = Array.isArray(result.notes) ? result.notes.join(' ') : result.notes || '';
    if (Array.isArray(result.warnings)) notes += ' ' + result.warnings.join(' ');
    text('backtest-notes', notes || 'Backtests use fixed fees and adverse slippage with candle-level fills. Stops take priority when both stop and target are touched in one candle. Actual execution costs and liquidity may differ.');
  }

  async function changeMarket() {
    if (state.busy) return;
    state.marketRevision += 1; state.candles = []; state.forecast = null; clearProjection(); state.candleSeries.setData([]); $('chart-empty').hidden = false;
    text('forecast-origin', 'Refresh actuals to compare without changing the prediction.');
    text('regime-heading', 'Waiting for forecast'); text('direction', '—'); $('direction').className = 'badge neutral';
    ['up-probability','down-probability','agreement','median-return','return-range','entry-price','stop-price','target-price'].forEach((id) => text(id, '—'));
    $('probability-fill').style.width = '0'; $('probability-fill').parentElement.dataset.ready = 'false';
    text('plan-action', 'NO FORECAST'); $('plan-action').className = 'badge neutral';
    text('regime-description', 'Path agreement describes how these samples cluster. It does not establish what the market will do.');
    const li = document.createElement('li'); li.textContent = 'Generate a forecast to evaluate the fixed strategy rules.'; $('plan-reasons').replaceChildren(li);
    text('plan-extra', 'No exchange connection. This dashboard never places orders.');
    $('backtest-empty').hidden = false; $('backtest-results').hidden = true;
    const results = await Promise.allSettled([loadCandles(true), loadArchive(null, true), loadLatestBacktest()]);
    results.forEach((r) => { if (r.status === 'rejected') error(r.reason.message); });
  }

  async function boot() {
    try { initCharts(); } catch (e) { error(e.message); $('forecast-button').disabled = true; $('backtest-button').disabled = true; return; }
    $('market-form').addEventListener('submit', (event) => { event.preventDefault(); runJob('forecast'); });
    $('backtest-button').addEventListener('click', () => runJob('backtest'));
    $('refresh-button').addEventListener('click', () => loadCandles(false));
    $('path-count').addEventListener('change', projection);
    let resizeTimer;
    window.addEventListener('resize', () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => { fitPriceChart(); if (state.equityChart) state.equityChart.timeScale().fitContent(); }, 150);
    });
    ['symbol','interval'].forEach((id) => $(id).addEventListener('change', changeMarket));
    $('symbol').addEventListener('blur', () => { $('symbol').value = $('symbol').value.trim().toUpperCase(); });
    $('auto-refresh').addEventListener('change', () => {
      if (state.refreshTimer) window.clearInterval(state.refreshTimer);
      state.refreshTimer = $('auto-refresh').checked ? window.setInterval(() => { if (!document.hidden && !$('refresh-button').disabled) loadCandles(false, true); }, 60000) : null;
    });
    $('saved-forecast').addEventListener('change', async () => {
      const id = $('saved-forecast').value;
      if (!id) return;
      try { error(''); await loadSavedForecast(id); }
      catch (e) { error(`Could not load the saved forecast: ${e.message}`); }
    });
    try { await loadSymbols(); } catch (e) { error(`Symbol list unavailable: ${e.message} You can still type a Binance symbol.`); }
    const results = await Promise.allSettled([loadCandles(true, true), loadArchive(null, true), loadLatestBacktest()]);
    results.forEach((r) => { if (r.status === 'rejected') error(r.reason.message); });
  }
  boot();
})();
