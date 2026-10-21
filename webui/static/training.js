'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { chart: null, candles: null, projections: [], forecast: null, actuals: [], status: null, busy: false, timer: null };
  const text = (id, value) => { $(id).textContent = value ?? '—'; };
  const error = (message) => { text('error', message); $('error').hidden = !message; };
  const numeric = (n) => typeof n === 'number' && Number.isFinite(n);
  const num = (n) => numeric(n) ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 5 }).format(n) : '—';
  const pct = (n) => numeric(n) ? `${(n * 100).toFixed(2)}%` : '—';
  const date = (v) => {
    if (v === null || v === undefined) return '—';
    const d = new Date(typeof v === 'number' ? v * 1000 : v);
    return Number.isNaN(d.getTime()) ? '—' : d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  };
  async function request(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}) } });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }
  function initChart() {
    if (!window.LightweightCharts) throw new Error('The local chart library could not load. Refresh this page.');
    state.chart = LightweightCharts.createChart($('price-chart'), {
      autoSize: true, layout: { background: { type: 'solid', color: '#111923' }, textColor: '#8e9bad', attributionLogo: true },
      grid: { vertLines: { color: '#1b273530' }, horzLines: { color: '#25314166' } },
      rightPriceScale: { borderColor: '#253141', scaleMargins: { top: .12, bottom: .1 } },
      timeScale: { borderColor: '#253141', timeVisible: true, rightOffset: 6 },
      localization: { timeFormatter: date },
    });
    state.candles = state.chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: '#91bea4', downColor: '#cb8992', wickUpColor: '#91bea4', wickDownColor: '#cb8992', borderVisible: false, priceLineVisible: false,
    });
    let chartWidth = $('price-chart').clientWidth;
    new ResizeObserver((entries) => {
      const width = entries[0].contentRect.width;
      if (width !== chartWidth) { chartWidth = width; requestAnimationFrame(fit); }
    }).observe($('price-chart'));
  }
  function fit() {
    const origin = state.forecast?.origin_time;
    const start = state.actuals.filter((c) => !origin || c.time <= origin).slice(-90)[0]?.time;
    const end = state.forecast?.models?.finetuned?.quantiles?.at(-1)?.time || state.actuals.at(-1)?.time;
    if (start && end && end > start) state.chart.timeScale().setVisibleRange({ from: start, to: end });
    else state.chart.timeScale().fitContent();
  }
  function displayActuals(items, merge = false) {
    const map = new Map((merge ? state.actuals : []).map((c) => [c.time, c]));
    for (const c of items || []) if (numeric(c.time)) map.set(c.time, c);
    state.actuals = [...map.values()].sort((a, b) => a.time - b.time);
    state.candles.setData(state.actuals.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    text('chart-status', `${state.actuals.length} displayed candles · latest open ${date(state.actuals.at(-1)?.time)}`);
  }
  function line(points, color, width = 1, dashed = false) {
    const s = state.chart.addSeries(LightweightCharts.LineSeries, { color, lineWidth: width, lineStyle: dashed ? 2 : 0, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(points); state.projections.push(s);
  }
  function projection() {
    state.projections.forEach((s) => state.chart.removeSeries(s)); state.projections = [];
    if (!state.forecast) return;
    const f = state.forecast;
    const last = f.history.at(-1);
    const start = last ? [{ time: last.time, value: last.close }] : [];
    for (const [kind, color, pathColor] of [['finetuned', '#a4e0bb', 'rgba(164,224,187,.12)'], ['original', '#e1b57c', 'rgba(225,181,124,.10)']]) {
      const model = f.models[kind];
      model.paths.slice(0, Number($('path-count').value)).forEach((p) => line([...start, ...p], pathColor));
      for (const key of ['p10', 'p90', 'p50']) line([...start, ...model.quantiles.map((q) => ({ time: q.time, value: q[key] }))], color, key === 'p50' ? 2 : 1, key !== 'p50');
    }
  }
  function showForecast(f) {
    state.forecast = f; displayActuals(f.history, true); projection(); fit();
    text('forecast-origin', `${f.source === 'live' ? 'Live' : 'Dataset'} origin ${date(f.origin_time)} · ${f.sampling.sample_count} paths per model`);
    text('forecast-note', `${f.normalization} Forecasts remain fixed when actual candles refresh. Percentile bounds describe generated samples; they are not calibrated confidence intervals.`);
  }
  function busy(value) {
    state.busy = value;
    for (const id of ['forecast-snapshot', 'forecast-live']) $(id).disabled = value || !state.status?.trained_ready;
  }
  async function startForecast(refreshLive) {
    if (state.busy) return;
    error(''); busy(true); $('job-panel').hidden = false;
    try {
      const { job_id: id } = await request('/api/forecast', { method: 'POST', body: JSON.stringify({ refresh_live: refreshLive }) });
      for (;;) {
        const job = await request(`/api/jobs/${id}`);
        text('job-label', job.message); text('job-percentage', `${Math.round(job.progress * 100)}%`); $('job-progress').value = job.progress;
        if (job.status === 'failed') throw new Error(job.error || job.message);
        if (job.status === 'completed') { showForecast(job.result); break; }
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    } catch (e) { error(e.message); }
    finally { busy(false); $('job-panel').hidden = true; }
  }
  async function refreshActuals() {
    $('refresh-actuals').disabled = true;
    try { const data = await request('/api/data?source=live'); displayActuals(data.candles, true); error(''); }
    catch (e) { error(`Actual candles could not refresh: ${e.message}`); }
    finally { $('refresh-actuals').disabled = false; }
  }
  function metricRows(report) {
    const rows = [
      ['OHLC mean absolute error', (m) => num(m.forecast_metrics?.ohlc?.mae)],
      ['Close root mean square error', (m) => num(m.forecast_metrics?.close?.rmse)],
      ['Close mean absolute percentage error', (m) => pct(m.forecast_metrics?.close?.mape)],
      ['Terminal directional accuracy', (m) => pct(m.forecast_metrics?.terminal_directional_accuracy)],
      ['Net return after costs', (m) => pct(m.backtest?.net_return)],
      ['Maximum drawdown', (m) => pct(m.backtest?.max_drawdown)],
      ['Closed trades', (m) => num(Array.isArray(m.backtest?.trades) ? m.backtest.trades.length : m.backtest?.trade_count ?? m.backtest?.trades)],
      ['Fees paid (quote currency)', (m) => num(m.backtest?.total_fees)],
      ['Slippage cost (quote currency)', (m) => num(m.backtest?.total_slippage)],
    ];
    $('metric-rows').replaceChildren(...rows.map(([label, getter]) => {
      const row = document.createElement('tr');
      for (const value of [label, getter(report?.models?.finetuned || {}), getter(report?.models?.original || {})]) {
        const td = document.createElement('td'); td.textContent = value; row.append(td);
      }
      return row;
    }));
  }
  async function loadReport() {
    try {
      const result = await request('/api/report');
      metricRows(result.report);
      if (!result.available) { text('evaluation-coverage', result.message); text('execution-rules', ''); $('trade-rules').replaceChildren(); return; }
      const report = result.report; const c = report.coverage || {};
      text('evaluation-coverage', `Test opens ${date(c.test_start)} to ${date(c.test_end_exclusive)} (exclusive) · ${num(c.evaluated_windows)} / ${num(c.scheduled_windows)} scheduled windows · ${pct(c.window_coverage)} coverage.`);
      const rules = report.execution_rules || {}; const sampling = report.sampling || {};
      text('execution-rules', `Matched sampling: ${sampling.sample_count ?? '—'} paths · seed ${sampling.seed ?? '—'} · temperature ${sampling.temperature ?? 1} · top-p ${sampling.top_p ?? .9}. Initial equity: ${num(report.models?.finetuned?.backtest?.initial_equity)} quote currency. Costs per side: ${rules.fee_bps ?? rules.fee_bps_per_side ?? '—'} bps fees + ${rules.slippage_bps ?? rules.slippage_bps_per_fill ?? '—'} bps slippage.`);
      $('trade-rules').replaceChildren(...[['Entry', rules.entry], ['Stop', rules.stop], ['Exit', rules.exit]].filter(([, value]) => value).map(([label, value]) => {
        const div = document.createElement('div'); const dt = document.createElement('dt'); const dd = document.createElement('dd');
        dt.textContent = label; dd.textContent = value; div.append(dt, dd); return div;
      }));
      const skipped = ['gap', 'incomplete', 'limit'].map((key) => `${c[`skipped_${key}_windows`] ?? 0} ${key}`).join(', ');
      text('evaluation-notes', `Skipped windows: ${skipped}. Evaluation uses a fixed chronological test period. Original pretraining may have included historical market data from this period; this is a holdout for your local experiment.`);
    } catch (e) { error(`Evaluation report could not load: ${e.message}`); }
  }
  async function init() {
    try {
      initChart(); state.status = await request('/api/status');
      const s = state.status; const m = s.market;
      const label = `${m.exchange.toUpperCase()} ${m.category} · ${m.symbol} / ${m.interval}`;
      text('market-label', label); text('dataset-market', label); text('chart-heading', `${m.symbol} / ${m.interval}`);
      text('sampling-label', `${s.lookback} context · ${s.horizon} forecast · ${s.sample_count} paths`);
      text('checkpoint-status', s.trained_ready ? 'TRAINED CHECKPOINT FOUND' : 'WAITING FOR TRAINING');
      $('checkpoint-status').className = `badge ${s.trained_ready ? 'positive' : 'neutral'}`;
      text('checkpoint-path', s.trained_checkpoint); busy(false);
      const data = await request('/api/data'); displayActuals(data.candles); fit();
      text('dataset-coverage', `${num(data.rows)} candles · ${date(data.first_open)} → ${date(data.last_open)}`);
      await loadReport();
      const latest = await request('/api/latest-forecast'); if (latest.forecast) showForecast(latest.forecast);
    } catch (e) { error(e.message); }
  }
  $('forecast-snapshot').addEventListener('click', () => startForecast(false));
  $('forecast-live').addEventListener('click', () => startForecast(true));
  $('refresh-actuals').addEventListener('click', refreshActuals);
  $('refresh-report').addEventListener('click', loadReport);
  $('path-count').addEventListener('change', projection);
  $('auto-refresh').addEventListener('change', () => { clearInterval(state.timer); state.timer = $('auto-refresh').checked ? setInterval(refreshActuals, 60000) : null; });
  init();
})();
