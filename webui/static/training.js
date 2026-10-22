'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { chart: null, candles: null, projections: [], forecast: null, actuals: [], status: null, busy: false, timer: null,
    paper: null, paperBusy: false, paperPolling: false, paperTimer: null, paperMarkers: null, paperLines: [],
    lastPaperOrigin: null, paperForecastLoading: false, paperVersion: 0 };
  const text = (id, value) => { $(id).textContent = value ?? '—'; };
  const error = (message) => { text('error', message); $('error').hidden = !message; };
  const numeric = (n) => typeof n === 'number' && Number.isFinite(n);
  const num = (n) => numeric(n) ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 5 }).format(n) : '—';
  const pct = (n) => numeric(n) ? `${(n * 100).toFixed(2)}%` : '—';
  const quantity = (n) => numeric(n) ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 8 }).format(n) : '—';
  const seconds = (value) => typeof value === 'number' ? Math.floor(value > 1e11 ? value / 1000 : value) : Math.floor(new Date(value).getTime() / 1000);
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
    state.paperMarkers = LightweightCharts.createSeriesMarkers(state.candles, []);
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
    paperChart();
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
      const model = f.models?.[kind];
      if (!model) continue;
      (model.paths || []).slice(0, Number($('path-count').value)).forEach((p) => line([...start, ...p], pathColor));
      for (const key of ['p10', 'p90', 'p50']) line([...start, ...(model.quantiles || []).map((q) => ({ time: q.time, value: q[key] }))], color, key === 'p50' ? 2 : 1, key !== 'p50');
    }
  }
  function showForecast(f) {
    state.forecast = f; displayActuals(f.history, true); projection(); fit();
    const paired = Boolean(f.models?.original);
    $('original-legend').hidden = !paired;
    text('chart-eyebrow', paired ? 'CANDLES & MATCHED FORECASTS' : 'CANDLES & TRAINED FORECAST');
    const source = f.source === 'paper_live' ? 'Automatic paper' : String(f.source).includes('live') ? 'Live' : 'Dataset';
    text('forecast-origin', `${source} origin ${date(f.origin_time)} · ${f.sampling.sample_count} paths${paired ? ' per model' : ' · trained checkpoint'}`);
    text('forecast-note', `${f.normalization || 'Normalization uses only the historical context; future candle values are never supplied.'} ${paired ? '' : 'Trained checkpoint only. Use “Compare from live candles” for a matched original-model comparison. '}Forecasts remain fixed when actual candles refresh. Percentile bounds describe generated samples; they are not calibrated confidence intervals.`);
    if (f.source === 'paper_live') state.lastPaperOrigin = String(seconds(f.origin_time));
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
  function paperMessage(message) {
    text('paper-error', message || ''); $('paper-error').hidden = !message;
  }
  function paperButtons() {
    const p = state.paper;
    $('paper-start').disabled = state.paperBusy || !p || Boolean(p.enabled) || !state.status?.trained_ready;
    $('paper-pause').disabled = state.paperBusy || !p?.enabled;
    $('paper-close').disabled = state.paperBusy || !p?.position;
    for (const id of ['paper-start', 'paper-pause', 'paper-close']) $(id).setAttribute('aria-busy', String(state.paperBusy));
  }
  function paperValue(id, value) {
    text(id, num(value));
    $(id).classList.toggle('text-positive', numeric(value) && value > 0);
    $(id).classList.toggle('text-negative', numeric(value) && value < 0);
  }
  function paperTradeRows(trades) {
    const rows = (trades || []).slice().reverse().slice(0, 100).map((trade) => {
      const row = document.createElement('tr');
      const values = [
        `${date(trade.entry_time)}\n${date(trade.exit_time)}`,
        `${num(trade.entry_price)} → ${num(trade.exit_price)}`,
        quantity(trade.quantity), num(trade.pnl ?? trade.net_pnl),
        `${num(trade.fees ?? trade.total_fees)} / ${num(trade.slippage ?? trade.total_slippage)}`,
        String(trade.exit_reason || trade.reason || '—').replaceAll('_', ' '),
      ];
      values.forEach((value, index) => {
        const cell = document.createElement('td'); cell.textContent = value;
        if (index === 3) cell.className = (trade.pnl ?? trade.net_pnl) > 0 ? 'text-positive' : (trade.pnl ?? trade.net_pnl) < 0 ? 'text-negative' : '';
        row.append(cell);
      });
      return row;
    });
    if (!rows.length) {
      const row = document.createElement('tr'); const cell = document.createElement('td');
      cell.colSpan = 6; cell.className = 'paper-empty'; cell.textContent = 'No closed virtual trades yet. Open positions and skipped signals appear above.';
      row.append(cell); rows.push(row);
    }
    $('paper-trade-rows').replaceChildren(...rows);
  }
  function renderPaper(body) {
    const p = body.paper || (typeof body.status === 'object' ? body.status : body);
    state.paper = { ...(p.account || {}), ...p };
    const paper = state.paper;
    const enabled = Boolean(paper.enabled);
    text('paper-status', enabled ? (paper.forecast_running ? 'PAPER ON · SAMPLING' : 'AUTOMATIC PAPER ON') : 'NEW ENTRIES PAUSED');
    $('paper-status').className = `badge ${enabled ? 'positive' : 'neutral'}`;
    const unit = paper.quote_currency || 'quote currency';
    text('paper-equity', `${num(paper.equity)} ${unit}`); text('paper-cash', num(paper.cash));
    paperValue('paper-realized', paper.realized_pnl); paperValue('paper-unrealized', paper.unrealized_pnl);
    text('paper-costs', `${num(paper.total_fees)} / ${num(paper.total_slippage)}`);
    text('paper-quote', num(paper.last_quote?.price));
    $('paper-quote').title = `Quote ${date(paper.last_quote?.timestamp ?? paper.last_quote?.time)} · ${paper.last_quote?.source || 'public exchange API'}`;
    const decision = paper.latest_decision || {};
    text('paper-action', paper.forecast_running ? 'Sampling the trained checkpoint…' : decision.action || (enabled ? 'Waiting for a completed candle' : 'New entries paused'));
    text('paper-reason', decision.reason || (enabled ? 'The server will evaluate the next complete candle. No entry is forced.' : 'Open positions remain under management. Start automatic paper trading to allow new entries.'));
    text('paper-decision-time', decision.decision_time ? `Decision ${date(decision.decision_time)} · context through ${date(decision.origin_time)}` : 'No trained-model decision recorded yet.');
    const interval = paper.market?.interval || state.status?.market?.interval || 'configured';
    text('paper-next-candle', `Checks each new ${interval} candle · one position · up to ${state.status?.horizon || 24} candles. Next candle boundary: ${date(paper.next_candle_at)}.`);
    const position = paper.position;
    text('paper-position-title', position ? `LONG · ${quantity(position.quantity)}` : 'Flat · no open position');
    const details = position ? [
      ['Entry', `${num(position.entry_price)} · ${date(position.entry_time)}`],
      ['Stop', num(position.stop_price)], ['Target', num(position.take_profit_price ?? position.target_price)],
      ['Time exit', date(position.horizon_end ?? position.horizon_end_time ?? position.exit_deadline)],
    ] : [['Starting virtual balance', `${num(paper.initial_equity)} ${unit}`], ['Closed virtual trades', String(paper.trades?.length || 0)]];
    $('paper-position-details').replaceChildren(...details.map(([label, value]) => {
      const div = document.createElement('div'); const dt = document.createElement('dt'); const dd = document.createElement('dd');
      dt.textContent = label; dd.textContent = value; div.append(dt, dd); return div;
    }));
    const rules = paper.rules || {};
    text('paper-rules', `Long only · ${pct(rules.consensus_threshold ?? .7)} sampled directional agreement · minimum reward/risk ${num(rules.minimum_reward_risk ?? 1.5)} · ${num(rules.fee_bps ?? rules.fee_bps_per_side ?? 10)} bps fee + ${num(rules.slippage_bps ?? rules.slippage_bps_per_fill ?? 5)} bps slippage per fill. ${paper.backtest_difference || 'Paper entries use the first observed public quote after the forecast is ready; the historical backtest assumes the next candle open.'} Pausing entries keeps open positions under management.`);
    text('paper-updated', `Account saved locally · checked ${date(paper.last_poll)} · browser may close`);
    paperMessage(paper.last_error || ''); paperTradeRows(paper.trades); paperButtons(); paperChart();
    refreshPaperForecast(paper);
  }
  async function refreshPaperForecast(paper) {
    const rawOrigin = paper.latest_decision?.origin_time;
    const origin = rawOrigin === null || rawOrigin === undefined ? null : String(seconds(rawOrigin));
    if (origin === null || origin === state.lastPaperOrigin || paper.forecast_running || state.paperForecastLoading || state.busy) return;
    state.paperForecastLoading = true;
    try {
      const result = await request('/api/latest-forecast');
      state.lastPaperOrigin = origin;
      if (result.forecast?.source === 'paper_live' && String(seconds(result.forecast.origin_time)) === origin) showForecast(result.forecast);
    } catch (e) { paperMessage(`The latest paper forecast could not load: ${e.message}`); }
    finally { state.paperForecastLoading = false; }
  }
  function paperChart() {
    if (!state.candles || !state.paperMarkers) return;
    for (const line of state.paperLines) state.candles.removePriceLine(line);
    state.paperLines = [];
    const actuals = state.actuals;
    if (!actuals.length || !state.paper) { state.paperMarkers.setMarkers([]); return; }
    const opens = actuals.map((c) => c.time);
    const first = opens[0]; const last = opens.at(-1);
    const step = opens.length > 1 ? last - opens.at(-2) : 0;
    const bucket = (raw) => {
      const stamp = seconds(raw);
      if (!Number.isFinite(stamp) || stamp < first || stamp >= last + step) return null;
      let left = 0; let right = opens.length;
      while (left < right) { const middle = Math.floor((left + right) / 2); if (opens[middle] <= stamp) left = middle + 1; else right = middle; }
      return opens[Math.max(0, left - 1)];
    };
    const markers = [];
    const append = (time, entry, label) => {
      const candle = bucket(time);
      if (candle !== null) markers.push({ time: candle, position: entry ? 'belowBar' : 'aboveBar', color: entry ? '#a4e0bb' : '#e1b57c', shape: entry ? 'arrowUp' : 'arrowDown', text: label });
    };
    for (const trade of state.paper.trades || []) {
      append(trade.entry_time, true, 'PAPER BUY'); append(trade.exit_time, false, 'PAPER EXIT');
    }
    const position = state.paper.position;
    if (position) append(position.entry_time, true, 'PAPER BUY · OPEN');
    state.paperMarkers.setMarkers(markers.sort((a, b) => a.time - b.time));
    // Future fills wait until their actual candle is available. Old account
    // levels never stretch a chart displaying an earlier dataset snapshot.
    if (position && bucket(position.entry_time) !== null) {
      for (const [price, title, color] of [[position.entry_price, 'Paper entry', '#a4e0bb'], [position.stop_price, 'Paper stop', '#ed8d95'], [position.take_profit_price ?? position.target_price, 'Paper target', '#e1b57c']]) {
        if (numeric(price) && price > 0) state.paperLines.push(state.candles.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title }));
      }
    }
  }
  async function loadPaper() {
    if (state.paperPolling || state.paperBusy) return;
    state.paperPolling = true;
    const version = state.paperVersion;
    try { const result = await request('/api/paper/status'); if (version === state.paperVersion) renderPaper(result); }
    catch (e) { paperMessage(`Paper account could not refresh: ${e.message}. Last displayed account is retained.`); }
    finally { state.paperPolling = false; }
  }
  async function paperAction(action) {
    if (state.paperBusy) return;
    state.paperVersion += 1; state.paperBusy = true; paperButtons();
    text('paper-updated', action === 'close' ? 'Closing the virtual position at a current public quote…' : action === 'pause' ? 'Pausing new entries…' : 'Starting the local paper runner…');
    try { renderPaper(await request(`/api/paper/${action}`, { method: 'POST', body: '{}' })); }
    catch (e) { paperMessage(`Paper action failed: ${e.message}`); }
    finally { state.paperBusy = false; paperButtons(); }
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
      text('checkpoint-path', s.trained_checkpoint); busy(false); paperButtons();
      const data = await request('/api/data'); displayActuals(data.candles); fit();
      text('dataset-coverage', `${num(data.rows)} candles · ${date(data.first_open)} → ${date(data.last_open)}`);
      await loadReport();
      const latest = await request('/api/latest-forecast'); if (latest.forecast) showForecast(latest.forecast);
      if ($('auto-refresh').checked) refreshActuals();
    } catch (e) { error(e.message); }
  }
  $('forecast-snapshot').addEventListener('click', () => startForecast(false));
  $('forecast-live').addEventListener('click', () => startForecast(true));
  $('refresh-actuals').addEventListener('click', refreshActuals);
  $('refresh-report').addEventListener('click', loadReport);
  $('path-count').addEventListener('change', projection);
  $('auto-refresh').addEventListener('change', () => { clearInterval(state.timer); state.timer = $('auto-refresh').checked ? setInterval(refreshActuals, 60000) : null; });
  $('paper-start').addEventListener('click', () => paperAction('start'));
  $('paper-pause').addEventListener('click', () => paperAction('pause'));
  $('paper-close').addEventListener('click', () => paperAction('close'));
  state.paperTimer = setInterval(loadPaper, 5000);
  state.timer = $('auto-refresh').checked ? setInterval(refreshActuals, 60000) : null;
  init();
  loadPaper();
})();
