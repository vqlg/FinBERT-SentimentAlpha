# FinBert-SentimentALPHA

An NLP-driven quantitative finance model that generates long/short equity signals by blending FinBERT sentiment analysis with technical indicators. Supports both live signal generation and historical backtesting.

![Dashboard](dashboard.png)

## How it works

The pipeline has five stages:

1. **News ingestion** — fetches headlines from SEC EDGAR 8-K filings (backtest) or live sources via Alpaca/AlphaVantage/GDELT (live mode)
2. **NLP scoring** — scores each headline with a 75% FinBERT + 25% VADER ensemble, producing a signed confidence score in [-1, +1]
3. **Signal construction** — builds five orthogonal components, each cross-sectionally z-scored:
   - `level` (0.25): smoothed EWMA sentiment level
   - `momentum` (0.25): multi-horizon sentiment momentum (3/7/15-day weighted diffs)
   - `surprise` (0.10): sentiment deviation from its 20-day moving average
   - `rsi` (0.25): Wilder RSI mapped to a signed signal
   - `macd` (0.15): MACD histogram normalised by price
4. **VIX regime filter** — scales signals to 50% at VIX > 25, flat at VIX > 35
5. **Portfolio construction** — tercile long-short, vol-targeted to 15% annualised, dollar-neutral with slot rotation to reduce turnover

## Quickstart

```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

**Live mode** (prints ranked signals + recommended positions for today):

```bash
python main.py live
python main.py live --tickers AAPL MSFT NVDA GOOGL
```

**Backtest mode** (runs over 2022–2024 by default, plots performance dashboard):

```bash
python main.py backtest
python main.py backtest --tickers AAPL MSFT AMZN --start 2023-01-01 --end 2024-12-31
```

## Configuration

All parameters live in `config.py`. Key knobs:

| Config class | Field | Default | Description |
|---|---|---|---|
| `NLPConfig` | `finbert_weight` | 0.75 | FinBERT share of ensemble score |
| `SignalConfig` | `decay_halflife` | 3.0 days | News impact decay |
| `SignalConfig` | `vix_exit_threshold` | 35.0 | VIX level at which signals go flat |
| `PortfolioConfig` | `vol_target` | 0.15 | Target annualised portfolio volatility |
| `PortfolioConfig` | `max_position` | 0.25 | Single-name position cap |
| `BacktestConfig` | `transaction_cost_bps` | 10 | Round-trip transaction cost |

Device selection (`NLPConfig.device`) is `"auto"` by default — uses MPS on Apple Silicon, CUDA on GPU machines, and CPU otherwise.

## Optional API keys

Set these environment variables to unlock additional data sources in live mode:

| Variable | Source | Used for |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | [Alpaca](https://alpaca.markets) | Real-time news feed |
| `FINNHUB_API_KEY` | [Finnhub](https://finnhub.io) | Insider sentiment (MSPR signal) |
| `ALPHAVANTAGE_API_KEY` | [Alpha Vantage](https://www.alphavantage.co) | News sentiment backup |

Without API keys the pipeline falls back to GDELT (public) for news and omits insider sentiment.

## Project structure

```
├── main.py                  # CLI entry point (live / backtest)
├── config.py                # All hyperparameters in one place
├── visualize.py             # Matplotlib/Seaborn dashboard
└── src/
    ├── data/
    │   ├── news.py          # News aggregator (routes to available sources)
    │   ├── edgar_news.py    # SEC EDGAR 8-K filing scraper (backtest)
    │   ├── alpaca_news.py   # Alpaca news API
    │   ├── alphavantage_news.py
    │   ├── gdelt_news.py    # GDELT public news feed
    │   ├── historical_news.py
    │   ├── prices.py        # yfinance price + VIX fetch
    │   └── insider.py       # Finnhub insider transaction sentiment
    ├── nlp/
    │   └── sentiment.py     # FinBERTScorer, VADERScorer, EnsembleSentimentScorer
    ├── signals/
    │   └── generator.py     # build_signals(), RSI, MACD, rolling z-score
    ├── portfolio/
    │   └── optimizer.py     # signals_to_weights() — vol targeting + slot rotation
    └── backtest/
        ├── engine.py        # Event-driven backtest loop
        └── metrics.py       # Sharpe, max drawdown, Calmar, turnover
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU is fine; MPS/CUDA speeds up FinBERT batching significantly)
- See `requirements.txt` for full dependency list

## Research references

- Tetlock (2007) — news sentiment and idiosyncratic returns
- Bollen et al. (2011) — Twitter mood and stock market
- Trading-R1 (arXiv:2509.11420) — multi-horizon sentiment momentum weights
- Alpha-R1 (arXiv:2512.23515) — sentiment + technical indicator ablation (Sharpe 1.15 → 1.62)
