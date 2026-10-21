<div align="center">
  <h2><b>Kronos: A Foundation Model for the Language of Financial Markets </b></h2>
</div>


<div align="center">

</a> 
<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://shiyu-coder.github.io/Kronos-demo/"> <img src="https://img.shields.io/badge/🚀-Live_Demo-brightgreen" alt="Live Demo"> </a>
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://zdoc.app/de/shiyu-coder/Kronos">Deutsch</a> | 
  <a href="https://zdoc.app/es/shiyu-coder/Kronos">Español</a> | 
  <a href="https://zdoc.app/fr/shiyu-coder/Kronos">Français</a> | 
  <a href="https://zdoc.app/ja/shiyu-coder/Kronos">日本語</a> | 
  <a href="https://zdoc.app/ko/shiyu-coder/Kronos">한국어</a> | 
  <a href="https://zdoc.app/pt/shiyu-coder/Kronos">Português</a> | 
  <a href="https://zdoc.app/ru/shiyu-coder/Kronos">Русский</a> | 
  <a href="https://zdoc.app/zh/shiyu-coder/Kronos">中文</a>
</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos is the **first open-source foundation model** for financial candlesticks (K-lines), 
> trained on data from over **45 global exchanges**.


</div>

## 📰 News
*   🚩 **[2025.11.10]** Kronos has been accpeted by AAAI 2026.
*   🚩 **[2025.08.17]** We have released the scripts for fine-tuning! Check them out to adapt Kronos to your own tasks.
*   🚩 **[2025.08.02]** Our paper is now available on [arXiv](https://arxiv.org/abs/2508.02739)!

<p align="center">

## 📜 Introduction

**Kronos** is a family of decoder-only foundation models, pre-trained specifically for the "language" of financial markets—K-line sequences. Unlike general-purpose TSFMs, Kronos is designed to handle the unique, high-noise characteristics of financial data. It leverages a novel two-stage framework: 
1. A specialized tokenizer first quantizes continuous, multi-dimensional K-line data (OHLCV) into **hierarchical discrete tokens**. 
2. A large, autoregressive Transformer is then pre-trained on these tokens, enabling it to serve as a unified model for diverse quantitative tasks.

<p align="center">
    <img src="figures/overview.png" alt="" align="center" width="700px" />
</p>

## ✨ Live Demo 
We have set up a live demo to visualize Kronos's forecasting results. The webpage showcases a forecast for the **BTC/USDT** trading pair over the next 24 hours. 

**👉 [Access the Live Demo Here](https://shiyu-coder.github.io/Kronos-demo/)** 

## 📦 Model Zoo 
We release a family of pre-trained models with varying capacities to suit different computational and application needs. All models are readily accessible from the Hugging Face Hub.

| Model        | Tokenizer                                                                       | Context length | Params  | Open-source                                                               |
|--------------|---------------------------------------------------------------------------------| -------------- | ------ |---------------------------------------------------------------------------|
| Kronos-mini  | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)     | 2048           | 4.1M   | ✅ [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)  |
| Kronos-small | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 24.7M  | ✅ [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) |
| Kronos-base  | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 102.3M | ✅ [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)   |
| Kronos-large | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 499.2M | ❌                                                                         |


## 🚀 Getting Started

### Installation

1. Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

### Local setup on this machine

The project is installed in `.venv` with Python 3.12 and CUDA-enabled PyTorch
2.7.1. The Web UI, example, test, CSV training, and Qlib dependencies are
included. Kronos-mini, Kronos-small, Kronos-base, and both matching tokenizers
are downloaded to the standard Hugging Face cache.

From the repository root:

```bash
source .venv/bin/activate
python examples/prediction_example.py
```

The two basic prediction examples use `tests/data/regression_input.csv`, a
bundled historical sample, and work from any directory. The local `data/`
directory links to that sample and the bundled Alibaba CSV for the Web UI.

The Web UI uses **http://127.0.0.1:7070**. The initial background launch used
a temporary user service, which can be inspected with:

```bash
systemctl --user status kronos-webui
```

The temporary service is removed when stopped or when the user session ends.
To start the Web UI after stopping it or rebooting, use:

```bash
./start-webui.sh
```

To stop the Web UI and release its loaded models:

```bash
./stop-webui.sh
```

This stops this checkout's user service and direct launches, including custom
ports. Running it again when the Web UI is already stopped is safe.

The homepage loads the Binance forecast dashboard, which loads Kronos-small
automatically when forecasting. In the CSV workspace at `/csv`, select a
model and CUDA, then click Load Model after a fresh server start. The launcher
binds to localhost, disables debug/reload, and uses the project environment
automatically. Set `PORT=7071` to select another port.

The bundled CSV training configuration uses public model IDs, local paths,
and a batch size of 2 for the local GPU. To start training when wanted:

```bash
cd finetune_csv
../.venv/bin/python train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

The public Qlib daily A-share dataset is installed at `data/qlib/cn_data`.
This historical demo covers 1999-11-10 through 2020-09-25. The settings in
`finetune/config.py` use that local path and matching train/validation/test
ranges; replace the data and update the ranges to work with more recent data.
Comet logging is disabled until account details are configured. Training has
not been started.

Qlib preprocessing is complete: `data/processed_datasets/` contains the
train, validation, and test pickle files. To regenerate them from the
repository root, run `.venv/bin/python finetune/qlib_data_preprocess.py`.

To recreate the environment with the already installed `uv` tool:

```bash
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-local.txt
mkdir -p data
ln -s ../tests/data/regression_input.csv data/regression_input.csv
ln -s ../finetune_csv/data/HK_ali_09988_kline_5min_all.csv data/HK_ali_09988_kline_5min_all.csv
```

The CUDA wheel command follows [PyTorch's official installation instructions](https://pytorch.org/get-started/previous-versions/#v271).
The local requirements pin NumPy and PyArrow to compatible versions and avoid
obsolete MLflow metadata. Installed versions are recorded in
`outputs/installed-packages.txt`.

Verification:

```bash
.venv/bin/python -m pip check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_kronos_regression.py -q
```

All four regression tests passed. A real GPU forecast through the Web UI was
also verified; its response is saved in `outputs/setup-webui-forecast.json`.

### Download training candles from Binance or Bybit

Use the installed environment automatically through this executable script:

```bash
./download_data.sh bybit BTCUSDT 1h 2021-01-01
./download_data.sh binance BTCUSDT 1h 2021-01-01
```

Arguments are exchange, exchange symbol, native candle interval, and start date
(`YYYY-MM-DD`, UTC). Downloads run through exchange server time and include
only completed candles whose opens are on or after the start date. No account
or API key is required. The downloader does not load or train a model.

Both exchanges default to **spot**, matching the dashboard. For Bybit USDT/USDC
contracts, add `--category linear`; inverse contracts use `--category inverse`.
The category is explicit because Bybit's API otherwise defaults to contracts.

```bash
./download_data.sh bybit BTCUSDT 1h 2021-01-01 --category linear
./download_data.sh binance ETHUSDT 4h 2024-01-01 --end-date 2025-01-01 --output data/my_eth.csv
```

`--end-date` is exclusive. Explicit relative output paths are relative to the
calling directory; default outputs always go to this repository's `data/`.

For example, the first command creates
`data/bybit_spot_BTCUSDT_1h_2021-01-01.csv` with exactly these columns:

```text
timestamps,open,high,low,close,volume,amount
```

Timestamps are UTC candle opens. `volume` uses base-asset units and `amount`
uses quote-asset turnover; inverse Bybit units are normalized accordingly.
Rows are sorted oldest first and duplicate timestamps are checked. Supported
Bybit intervals are `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M`.
Binance additionally supports `1s, 8h, 3d`. `1M` means calendar month, while
`1m` means minute.

If early history is unavailable, candles are missing inside the range, or the
available history ends early, the command prints a **NOTICE** with the actual
coverage. It preserves gaps and never invents candles. A `.metadata.json`
sidecar records coverage and notices. Historical Binance candles with shortened
durations are retained and reported, including their original close times in
the metadata. No data produces an explicit notice and
a nonzero exit status; API failures also return nonzero. Existing CSVs remain
unchanged if downloading or validating fails. Successful reruns replace the
CSV and metadata with the new snapshot.

The downloader streams API pages to temporary disk storage so large exports
can exceed the dashboard's in-memory candle cap. CSVs in `data/` appear in the
CSV workspace at `/csv`.

API references: [Binance public data](https://developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only)
and [Bybit public kline API](https://bybit-exchange.github.io/docs/v5/market/kline).

### Binance forecasts, market regimes, and walk-forward backtests

Start the local server with `./start-webui.sh`, then open
**http://localhost:7070/** to see the BTCUSDT actual and forecast prices.
The dashboard also works at `/market`. Stop it with `./stop-webui.sh`.
The CSV workspace is available at `/csv`.

The market dashboard downloads Binance spot OHLCV and quote-asset volume
(`amount`) through the [public market-data API](https://developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only),
without an account or API key. It uses UTC, normal Binance candles and excludes
the current unfinished candle. These correspond to the same Binance spot feed
on TradingView; another venue, futures, synthetic candles, or timezone can differ.
Choose a listed pair such as `BTCUSDT` or `ETHUSDT` and a native Binance interval.
Markets must have at least 512 contiguous closed candles: weekly/monthly pairs
often do not have enough history. A 24-candle horizon means 24 hours on `1h`,
and 24 days on `1d`.

Each forecast loads pinned Kronos-small/tokenizer checkpoints, conditions on
512 candles, and keeps **100 separate 24-candle futures**. Upstream
`sample_count=100` averages those paths internally; this workflow instead batches
independent series with `sample_count=1` and preserves all 100 outputs. Small
inference batches bound memory use; CPU is supported, while GPU is faster.
The chart displays the forecast median, p10/p90 bounds, optional individual
paths, and actual candles using locally hosted [Lightweight Charts](https://tradingview.github.io/lightweight-charts/docs).

The default rules are fixed before each backtest:

- **Regime:** TRENDING when at least 70% of terminal sampled closes agree on
  direction relative to the last observed close; otherwise CHOPPY.
- **Entry:** spot long only, when TRENDING/BULLISH and estimated net reward/risk
  is at least 1.5 after costs. Bearish and choppy scenarios hold cash.
- **Stop:** the wider of 1.5 ATR14 or the terminal p10 price below the reference.
  **Target:** the forecast median terminal close. Exit after at most 24 candles.
- **Sizing:** estimated stop risk of 1% of equity, capped at a 25% allocation;
  no leverage. Gaps can exceed the planned risk budget.
- **Execution:** enter at the next bar open, rechecking frozen levels and
  reward/risk; charge fees and adverse slippage on both sides. Stop gaps fill
  at the worse open; if stop and target are both touched, assume the stop first.

These sample fractions and spreads are **not calibrated probabilities or risk
bounds**. Fees default to an assumed 10 bps per side and adverse slippage to
5 bps per fill; edit them for the market/account you intend to study. The app
does not know your actual commission tier or realized fills and places no orders.
The backtest uses candle-level fills, constant costs, and no inference latency
or order-book/market-impact model. Rules are deterministic; no external LLM or
paid API is called to classify the regime.

“Run 1-year backtest” fetches 365 days plus the 512-candle warmup, performs a
causal walk-forward replay every 24 candles, marks equity at every candle close,
and compares net returns with a cost-adjusted buy-and-hold benchmark. The last
partial window closes at the end of the available test history. Shorter
intervals require many more model runs; one-second annual history exceeds the
one-million-candle request cap. Forecast paths are cached to resume repeated
runs, and completed reports record coverage, checkpoints, rules, and costs.

To observe forecasts over several days, select a saved forecast and enable
“Refresh actuals every 60s.” Actual candles update while the prediction remains
frozen. Saved forecasts, jobs, source candles, sample caches, and backtests
persist under `outputs/market/`; closing the page does not delete them. A newly
generated forecast creates a separate snapshot. Reopen the dashboard to load
the most recent forecast and backtest for the selected market.

The same workflow is available from the repository root without a browser:

```bash
.venv/bin/python -m market forecast --symbol BTCUSDT --interval 1h
.venv/bin/python -m market backtest --symbol BTCUSDT --interval 1d --days 365 --fee-bps 10 --slippage-bps 5
# CPU / smaller-memory machine:
.venv/bin/python -m market forecast --symbol ETHUSDT --interval 1h --device cpu --batch-size 2
```

Test the data, sampling, API and execution rules with
`.venv/bin/python -m pytest tests -q`.

### 📈 Making Forecasts

Forecasting with Kronos is straightforward using the `KronosPredictor` class. It handles data preprocessing, normalization, prediction, and inverse normalization, allowing you to get from raw data to forecasts in just a few lines of code.

**Important Note**: The `max_context` for `Kronos-small` and `Kronos-base` is **512**. This is the maximum sequence length the model can process. For optimal performance, it is recommended that your input data length (i.e., `lookback`) does not exceed this limit. The `KronosPredictor` will automatically handle truncation for longer contexts.

Here is a step-by-step guide to making your first forecast.

#### 1. Load the Tokenizer and Model

First, load a pre-trained Kronos model and its corresponding tokenizer from the Hugging Face Hub.

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

#### 2. Instantiate the Predictor

Create an instance of `KronosPredictor`, passing the model, tokenizer, and desired device.

```python
# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

#### 3. Prepare Input Data

The `predict` method requires three main inputs:
-   `df`: A pandas DataFrame containing the historical K-line data. It must include columns `['open', 'high', 'low', 'close']`. `volume` and `amount` are optional.
-   `x_timestamp`: A pandas Series of timestamps corresponding to the historical data in `df`.
-   `y_timestamp`: A pandas Series of timestamps for the future periods you want to predict.

```python
import pandas as pd

# Load your data
df = pd.read_csv("./tests/data/regression_input.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length
lookback = 400
pred_len = 120

# Prepare inputs for the predictor
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

#### 4. Generate Forecasts 

Call the `predict` method to generate forecasts. You can control the sampling process with parameters like `T`, `top_p`, and `sample_count` for probabilistic forecasting.

```python
# Generate predictions
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature for sampling
    top_p=0.9,      # Nucleus sampling probability
    sample_count=1  # Number of forecast paths to generate and average
)

print("Forecasted Data Head:")
print(pred_df.head())
```

The `predict` method returns a pandas DataFrame containing the forecasted values for `open`, `high`, `low`, `close`, `volume`, and `amount`, indexed by the `y_timestamp` you provided.

For efficient processing of multiple time series, Kronos provides a `predict_batch` method that enables parallel prediction on multiple datasets simultaneously. This is particularly useful when you need to forecast multiple assets or time periods at once.

```python
# Prepare multiple datasets for batch prediction
df_list = [df1, df2, df3]  # List of DataFrames
x_timestamp_list = [x_ts1, x_ts2, x_ts3]  # List of historical timestamps
y_timestamp_list = [y_ts1, y_ts2, y_ts3]  # List of future timestamps

# Generate batch predictions
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# pred_df_list contains prediction results in the same order as input
for i, pred_df in enumerate(pred_df_list):
    print(f"Predictions for series {i}:")
    print(pred_df.head())
```

**Important Requirements for Batch Prediction:**
- All series must have the same historical length (lookback window)
- All series must have the same prediction length (`pred_len`)
- Each DataFrame must contain the required columns: `['open', 'high', 'low', 'close']`
- `volume` and `amount` columns are optional and will be filled with zeros if missing

The `predict_batch` method leverages GPU parallelism for efficient processing and automatically handles normalization and denormalization for each series independently.

#### 5. Example and Visualization

For a complete, runnable script that includes data loading, prediction, and plotting, please see [`examples/prediction_example.py`](examples/prediction_example.py).

Running this script will generate a plot comparing the ground truth data against the model's forecast, similar to the one shown below:

<p align="center">
    <img src="figures/prediction_example.png" alt="Forecast Example" align="center" width="600px" />
</p>

Additionally, we provide a script that makes predictions without Volume and Amount data, which can be found in [`examples/prediction_wo_vol_example.py`](examples/prediction_wo_vol_example.py).


## 🔧 Finetuning on Your Own Data (A-Share Market Example)

We provide a complete pipeline for finetuning Kronos on your own datasets. As an example, we demonstrate how to use [Qlib](https://github.com/microsoft/qlib) to prepare data from the Chinese A-share market and conduct a simple backtest.

> **Disclaimer:** This pipeline is intended as a demonstration to illustrate the finetuning process. It is a simplified example and not a production-ready quantitative trading system. A robust quantitative strategy requires more sophisticated techniques, such as portfolio optimization and risk factor neutralization, to achieve stable alpha.

The finetuning process is divided into four main steps:

1.  **Configuration**: Set up paths and hyperparameters.
2.  **Data Preparation**: Process and split your data using Qlib.
3.  **Model Finetuning**: Finetune the Tokenizer and the Predictor models.
4.  **Backtesting**: Evaluate the finetuned model's performance.

### Prerequisites

1.  First, ensure you have all dependencies from `requirements.txt` installed.
2.  This pipeline relies on `qlib`. Please install it:
    ```shell
      pip install pyqlib
    ```
3.  You will need to prepare your Qlib data. Follow the [official Qlib guide](https://github.com/microsoft/qlib) to download and set up your data locally. The example scripts assume you are using daily frequency data.

### Step 1: Configure Your Experiment

All settings for data, training, and model paths are centralized in `finetune/config.py`. Before running any scripts, please **modify the following paths** according to your environment:

*   `qlib_data_path`: Path to your local Qlib data directory.
*   `dataset_path`: Directory where the processed train/validation/test pickle files will be saved.
*   `save_path`: Base directory for saving model checkpoints.
*   `backtest_result_path`: Directory for saving backtesting results.
*   `pretrained_tokenizer_path` and `pretrained_predictor_path`: Paths to the pre-trained models you want to start from (can be local paths or Hugging Face model names).

You can also adjust other parameters like `instrument`, `train_time_range`, `epochs`, and `batch_size` to fit your specific task. If you don't use [Comet.ml](https://www.comet.com/), set `use_comet = False`.

### Step 2: Prepare the Dataset

Run the data preprocessing script. This script will load raw market data from your Qlib directory, process it, split it into training, validation, and test sets, and save them as pickle files.

```shell
python finetune/qlib_data_preprocess.py
```

After running, you will find `train_data.pkl`, `val_data.pkl`, and `test_data.pkl` in the directory specified by `dataset_path` in your config.

### Step 3: Run the Finetuning

The finetuning process consists of two stages: finetuning the tokenizer and then the predictor. Both training scripts are designed for multi-GPU training using `torchrun`.

#### 3.1 Finetune the Tokenizer

This step adjusts the tokenizer to the data distribution of your specific domain.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

The best tokenizer checkpoint will be saved to the path configured in `config.py` (derived from `save_path` and `tokenizer_save_folder_name`).

#### 3.2 Finetune the Predictor

This step finetunes the main Kronos model for the forecasting task.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

The best predictor checkpoint will be saved to the path configured in `config.py`.

### Step 4: Evaluate with Backtesting

Finally, run the backtesting script to evaluate your finetuned model. This script loads the models, performs inference on the test set, generates prediction signals (e.g., forecasted price change), and runs a simple top-K strategy backtest.

```shell
# Specify the GPU for inference
python finetune/qlib_test.py --device cuda:0
```

The script will output a detailed performance analysis in your console and generate a plot showing the cumulative return curves of your strategy against the benchmark, similar to the one below:

<p align="center">
    <img src="figures/backtest_result_example.png" alt="Backtest Example" align="center" width="700px" />
</p>

### 💡 From Demo to Production: Important Considerations

*   **Raw Signals vs. Pure Alpha**: The signals generated by the model in this demo are raw predictions. In a real-world quantitative workflow, these signals would typically be fed into a portfolio optimization model. This model would apply constraints to neutralize exposure to common risk factors (e.g., market beta, style factors like size and value), thereby isolating the **"pure alpha"** and improving the strategy's robustness.
*   **Data Handling**: The provided `QlibDataset` is an example. For different data sources or formats, you will need to adapt the data loading and preprocessing logic.
*   **Strategy and Backtesting Complexity**: The simple top-K strategy used here is a basic starting point. Production-level strategies often incorporate more complex logic for portfolio construction, dynamic position sizing, and risk management (e.g., stop-loss/take-profit rules). Furthermore, a high-fidelity backtest should meticulously model transaction costs, slippage, and market impact to provide a more accurate estimate of real-world performance.

> **📝 AI-Generated Comments**: Please note that many of the code comments within the `finetune/` directory were generated by an AI assistant (Gemini 2.5 Pro) for explanatory purposes. While they aim to be helpful, they may contain inaccuracies. We recommend treating the code itself as the definitive source of logic.

## 📖 Citation

If you use Kronos in your research, we would appreciate a citation to our [paper](https://arxiv.org/abs/2508.02739):

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## 📜 License 
This project is licensed under the [MIT License](./LICENSE).










