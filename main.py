"""
SentimentQuant — NLP-driven quantitative finance model.

Two modes:
  python main.py live      [--tickers AAPL MSFT ...]
  python main.py backtest  [--tickers AAPL MSFT ...] [--start 2022-01-01] [--end 2024-12-31]

Backtest mode uses a synthetic sentiment proxy (idiosyncratic price returns) to
validate the signal construction and portfolio methodology.  For a real historical
backtest, plug in a news API (e.g. Alpaca News, NewsAPI.org) and replace the
synthetic_daily_sentiment() call in run_backtest() with fetch_all_news() + scorer.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config import CFG
from src.data.news import fetch_all_news
from src.data.prices import fetch_prices, fetch_vix
from src.nlp.sentiment import EnsembleSentimentScorer
from src.signals.generator import build_signals, aggregate_daily_sentiment
from src.portfolio.optimizer import signals_to_weights
from src.backtest.engine import run_backtest
from visualize import plot_dashboard, plot_live_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
console = Console()


# ---------------------------------------------------------------------------
# LIVE MODE
# ---------------------------------------------------------------------------

def run_live(tickers=None):
    """
    Full-pipeline live mode: blends a 120-day synthetic sentiment baseline with
    real scored news, runs the 5-step signal pipeline, and outputs vol-scaled
    position weights alongside current price and VIX context.
    """
    tickers = tickers or CFG.backtest.tickers
    console.rule("[bold blue]SentimentQuant — Live Signal Mode")

    # Need ≥ zscore_window (60) trading days of history for the rolling z-scores.
    # 120 calendar days ≈ 85 trading days — enough warmup with room to spare.
    history_start = (pd.Timestamp.now() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    history_end = pd.Timestamp.now().strftime("%Y-%m-%d")

    console.print("[cyan]Fetching 120-day price history for signal warmup…")
    prices, _ = fetch_prices(tickers, history_start, history_end)
    vix = fetch_vix(history_start, history_end)

    if prices.empty:
        console.print("[red]Price fetch failed. Check tickers/internet.")
        return

    prices = prices.dropna(axis=1, thresh=40)
    active_tickers = prices.columns.tolist()

    # Synthetic sentiment baseline: idiosyncratic price returns proxy for news sentiment.
    # Provides the historical depth the z-score and momentum windows need.
    synthetic_sent = _synthetic_daily_sentiment(prices)

    # Real news fetch and NLP scoring
    console.print(f"[cyan]Fetching real news for {len(active_tickers)} tickers…")
    news_df = fetch_all_news(active_tickers, use_cache=True)
    scored = pd.DataFrame()

    if not news_df.empty:
        sources = news_df["source"].value_counts().to_dict()
        console.print(f"[green]{len(news_df)} headlines  |  sources: {sources}")
        console.print("[cyan]Scoring with FinBERT 75% + VADER 25%…")
        scorer_obj = EnsembleSentimentScorer(CFG.nlp)
        scored = scorer_obj.score_dataframe(news_df)
        real_sent = aggregate_daily_sentiment(scored)

        # Real scored sentiment overrides synthetic wherever dates overlap
        blended = synthetic_sent.merge(
            real_sent.rename(columns={"raw_sentiment": "_real"}),
            on=["date", "ticker"],
            how="left",
        )
        blended["raw_sentiment"] = blended["_real"].fillna(blended["raw_sentiment"])
        blended = blended.drop(columns="_real")
    else:
        console.print("[yellow]No news retrieved — using synthetic sentiment proxy.")
        blended = synthetic_sent

    # Full 5-step signal pipeline
    console.print(
        "[cyan]Running full pipeline "
        "(EWMA → level/momentum/surprise → VIX filter → cross-sectional z-score)…"
    )
    signals = build_signals(blended, prices, vix, CFG.signal)
    weights = signals_to_weights(signals, prices, CFG.portfolio)

    # Latest row
    latest_date = signals.index[-1]
    latest_signals = signals.loc[latest_date].dropna()
    latest_weights = weights.loc[latest_date]

    # VIX regime label
    current_vix = float(vix.dropna().iloc[-1]) if not vix.empty else None
    if current_vix is not None:
        if current_vix > CFG.signal.vix_exit_threshold:
            vix_str = f"[bold red]HIGH ({current_vix:.1f}) — signals FLAT[/bold red]"
        elif current_vix > CFG.signal.vix_scale_threshold:
            vix_str = f"[yellow]ELEVATED ({current_vix:.1f}) — signals at 50%[/yellow]"
        else:
            vix_str = f"[green]NORMAL ({current_vix:.1f}) — full signal[/green]"
    else:
        vix_str = "N/A"

    # Price and intraday % change from the already-fetched price DataFrame
    latest_prices = prices.iloc[-1]
    daily_changes = prices.pct_change().iloc[-1] * 100
    headline_counts = scored.groupby("ticker").size().to_dict() if not scored.empty else {}

    # Output table
    ranked = latest_signals.sort_values(ascending=False)
    table = Table(
        title=f"Live Signals — {latest_date.date()}",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Rank",      justify="center", width=5)
    table.add_column("Ticker",    justify="center", width=8)
    table.add_column("Signal",    justify="right",  width=8)
    table.add_column("Weight",    justify="right",  width=9)
    table.add_column("Price",     justify="right",  width=9)
    table.add_column("Today",     justify="right",  width=8)
    table.add_column("Headlines", justify="right",  width=11)
    table.add_column("Action",    justify="center", width=10)

    for rank, (ticker, sig) in enumerate(ranked.items(), 1):
        w = latest_weights.get(ticker, 0.0)
        price = latest_prices.get(ticker)
        chg = daily_changes.get(ticker)
        n_hl = headline_counts.get(ticker, 0)

        price_str = f"${price:.2f}" if price is not None and not np.isnan(price) else "—"
        chg_str   = f"{chg:+.1f}%" if chg is not None and not np.isnan(chg) else "—"
        chg_color = "green" if chg and chg > 0 else "red" if chg and chg < 0 else "white"

        if w > 0.02:
            action, w_color = "[bold green]LONG[/bold green]", "green"
        elif w < -0.02:
            action, w_color = "[bold red]SHORT[/bold red]", "red"
        else:
            action, w_color = "[yellow]FLAT[/yellow]", "yellow"

        table.add_row(
            str(rank), ticker,
            f"{sig:+.3f}",
            f"[{w_color}]{w * 100:+.1f}%[/{w_color}]",
            price_str,
            f"[{chg_color}]{chg_str}[/{chg_color}]",
            str(n_hl),
            action,
        )

    console.print(table)
    console.print(f"\nVIX: {vix_str}")
    console.print(f"Signal date: [cyan]{latest_date.date()}[/cyan]")

    # Key headlines for the strongest active positions
    if not scored.empty:
        active_longs  = sorted(
            [t for t in latest_weights.index if latest_weights[t] > 0.02],
            key=lambda t: latest_weights[t], reverse=True,
        )[:2]
        active_shorts = sorted(
            [t for t in latest_weights.index if latest_weights[t] < -0.02],
            key=lambda t: latest_weights[t],
        )[:2]

        if active_longs or active_shorts:
            console.print("\n[bold]Key headlines for active positions:[/bold]")
            for t in active_longs + active_shorts:
                w_val = latest_weights[t]
                direction = "LONG" if w_val > 0 else "SHORT"
                color = "green" if w_val > 0 else "red"
                top_news = scored[scored["ticker"] == t].nlargest(3, "finbert_score")
                console.print(f"\n  [bold {color}]{t}[/bold {color}] ({direction}, {w_val * 100:+.1f}%)")
                for _, h in top_news.iterrows():
                    hc = "green" if h["finbert_score"] > 0 else "red"
                    console.print(f"    [{hc}]{h['finbert_score']:+.3f}[/{hc}]  {h['headline'][:90]}")

    signals_plot_df = pd.DataFrame({"ticker": latest_signals.index, "signal": latest_signals.values})
    plot_live_signals(signals_plot_df, scored)
    return latest_signals, latest_weights, scored


# ---------------------------------------------------------------------------
# BACKTEST MODE
# ---------------------------------------------------------------------------

def _synthetic_daily_sentiment(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a synthetic daily sentiment proxy from idiosyncratic price returns.

    Method: market-adjusted return (stock return minus equal-weight index) is used
    as a proxy for news-driven sentiment.  The logic follows Tetlock (2007): abnormal
    returns tend to be associated with unusual news flow, so idiosyncratic return
    direction approximates the sentiment of that day's headlines.

    tanh(20 * idio_return) maps returns into [-1, +1] with a soft threshold that
    treats anything beyond ±5% as strongly positive/negative.

    REPLACE THIS with real historical news data for production use.
    """
    returns = prices.pct_change().dropna(how="all")
    market_ret = returns.mean(axis=1)
    idio = returns.subtract(market_ret, axis=0)

    # Smooth with short lag to simulate the delay between price moves and reporting
    smoothed = idio.ewm(span=3).mean().shift(1)

    long_df = (
        smoothed.apply(np.tanh)
        .mul(20)                       # amplify small idio returns to sentiment scale
        .clip(-1, 1)
        .stack()
        .reset_index()
    )
    long_df.columns = ["date", "ticker", "raw_sentiment"]
    return long_df.dropna()


def run_backtest_mode(tickers=None, start=None, end=None):
    cfg = CFG.backtest
    tickers = tickers or cfg.tickers
    start = start or cfg.start_date
    end = end or cfg.end_date

    console.rule("[bold blue]SentimentQuant — Backtest Mode")
    console.print(
        "[yellow]Using synthetic sentiment proxy (idiosyncratic returns).\n"
        "[yellow]For production: replace _synthetic_daily_sentiment() with real news API.\n"
    )

    console.print(f"[cyan]Fetching price data {start} → {end} for {len(tickers)} tickers…")
    prices, _ = fetch_prices(tickers, start, end)
    vix = fetch_vix(start, end)

    if prices.empty:
        console.print("[red]Price fetch returned empty DataFrame. Check tickers/dates.")
        sys.exit(1)

    # Remove tickers with insufficient history (< 200 trading days)
    prices = prices.dropna(axis=1, thresh=200)
    active_tickers = prices.columns.tolist()
    if not active_tickers:
        console.print("[red]No tickers with sufficient price history.")
        sys.exit(1)
    console.print(f"[green]Active tickers after history filter: {active_tickers}")

    # Build synthetic sentiment → signals → weights → backtest
    console.print("[cyan]Building synthetic daily sentiment…")
    daily_sentiment = _synthetic_daily_sentiment(prices)

    console.print("[cyan]Constructing composite signals (level + momentum + surprise)…")
    signals = build_signals(daily_sentiment, prices, vix, CFG.signal)

    console.print("[cyan]Sizing positions (tercile long-short, vol-scaled, dollar-neutral)…")
    weights = signals_to_weights(signals, prices, CFG.portfolio)

    console.print("[cyan]Running backtest…")
    result = run_backtest(weights, prices, cfg)

    # Benchmark
    try:
        spy, _ = fetch_prices(["SPY"], start, end)
        benchmark = spy.squeeze()
    except Exception:
        benchmark = None

    # Print metrics
    m = result["metrics"]
    table = Table(title="Strategy Performance", header_style="bold cyan", show_lines=True)
    table.add_column("Metric", style="bold", width=28)
    table.add_column("Value", justify="right", width=12)
    skip = {"monthly_returns"}
    for k, v in m.items():
        if k not in skip:
            table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    # Benchmark comparison
    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        rf_d = (1 + cfg.risk_free_rate) ** (1 / 252) - 1
        bench_sharpe = float(
            (bench_ret.mean() - rf_d) / bench_ret.std() * np.sqrt(252)
        )
        alpha = m["sharpe_ratio"] - bench_sharpe
        color = "green" if alpha > 0 else "red"
        console.print(f"\nSPY Sharpe: [yellow]{bench_sharpe:.3f}[/yellow]")
        console.print(f"Sharpe alpha vs SPY: [{color}]{alpha:+.3f}[/{color}]")

    # Daily turnover
    avg_turnover = result["turnover"].mean()
    console.print(f"Average daily turnover: [cyan]{avg_turnover:.3f}[/cyan]  "
                  f"(ann. TC ≈ [cyan]{avg_turnover * 252 * cfg.transaction_cost_bps:.0f}[/cyan] bps)")

    plot_dashboard(result, benchmark)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentimentQuant")
    parser.add_argument("mode", choices=["live", "backtest"])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(args.tickers)
    else:
        run_backtest_mode(args.tickers, args.start, args.end)
