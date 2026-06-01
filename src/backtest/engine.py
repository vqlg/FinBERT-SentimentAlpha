import numpy as np
import pandas as pd

from .metrics import compute_metrics


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cfg,
) -> dict:
    """
    Vectorised single-pass backtester.

    Critical design choice: weights[t] uses only information available at
    close of day t (sentiment scored from that day's news).  Positions are
    entered at the OPEN of day t+1 — modelled here as close t+1 for
    simplicity, introducing at most one day of execution slippage.

    This 1-day lag is the standard no-lookahead-bias convention in the
    academic event-study and NLP-trading literature.
    """
    # 1-day lag: signal from day t → position held on day t+1
    positions = weights.shift(1)

    # Align tickers
    common = positions.columns.intersection(prices.columns)
    positions = positions[common].reindex(prices.index).fillna(0.0)
    returns = prices[common].pct_change().fillna(0.0)

    # Gross PnL
    gross_pnl = (positions * returns).sum(axis=1)

    # Transaction costs applied on daily turnover (sum of absolute weight changes)
    turnover = positions.diff().abs().sum(axis=1)
    cost = turnover * (cfg.transaction_cost_bps / 10_000)

    net_pnl = gross_pnl - cost
    portfolio_value = (1 + net_pnl).cumprod() * cfg.initial_capital

    metrics = compute_metrics(net_pnl, cfg.risk_free_rate)

    return {
        "portfolio_value": portfolio_value,
        "net_returns": net_pnl,
        "gross_returns": gross_pnl,
        "turnover": turnover,
        "positions": positions,
        "metrics": metrics,
    }
