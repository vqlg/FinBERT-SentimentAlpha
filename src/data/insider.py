"""
Insider sentiment from Finnhub (free tier: 60 req/min).

Requires FINNHUB_API_KEY environment variable.  When the key is absent the
module returns an empty DataFrame and the caller falls back gracefully.

The key signal is MSPR (Monthly Share Purchase Ratio):
  MSPR = net_buy_value / (total_buy_value + total_sell_value)
  MSPR ∈ [-1, +1]:  +1 = pure buying, -1 = pure selling, 0 = neutral.

Insiders have private information about their own companies; their net buying
activity is an orthogonal signal to news sentiment and price technicals.
"""

import logging
import os
import time
from datetime import date, timedelta
from typing import List
from urllib.request import Request, urlopen
import json

from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"


def _get_api_key() -> Optional[str]:
    return os.environ.get("FINNHUB_API_KEY")


def _finnhub_get(endpoint: str, params: dict) -> Optional[dict]:
    key = _get_api_key()
    if not key:
        return None
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_BASE_URL}/{endpoint}?{qs}&token={key}"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("Finnhub request failed (%s %s): %s", endpoint, params, exc)
        return None


def fetch_insider_sentiment(
    tickers: List[str],
    lookback_days: int = 90,
    rate_limit_secs: float = 1.1,
) -> pd.DataFrame:
    """
    Fetch monthly insider sentiment (MSPR + net change) for each ticker.

    Returns a long-format DataFrame with columns:
      ticker, date (first of month), mspr, change

    If FINNHUB_API_KEY is not set, returns an empty DataFrame so callers
    can degrade gracefully without crashing.
    """
    if not _get_api_key():
        logger.info("FINNHUB_API_KEY not set — skipping insider sentiment.")
        return pd.DataFrame(columns=["ticker", "date", "mspr", "change"])

    end = date.today()
    start = end - timedelta(days=lookback_days)
    rows = []

    for ticker in tickers:
        data = _finnhub_get(
            "stock/insider-sentiment",
            {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()},
        )
        if data and "data" in data:
            for entry in data["data"]:
                rows.append({
                    "ticker": ticker,
                    "date": pd.Timestamp(f"{entry['year']}-{entry['month']:02d}-01"),
                    "mspr": float(entry.get("mspr", 0.0)),
                    "change": float(entry.get("change", 0.0)),
                })
        time.sleep(rate_limit_secs)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "mspr", "change"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def insider_to_daily_signal(
    insider_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Convert monthly insider MSPR to a daily wide signal aligned to the price index.

    Monthly values are forward-filled across the trading days of that month until
    the next report, capped at 90 days to prevent very stale data from persisting.
    """
    if insider_df.empty:
        return pd.DataFrame(index=price_index)

    # Pivot to wide (date = first-of-month, columns = tickers)
    wide = insider_df.pivot(index="date", columns="ticker", values="mspr")

    # Reindex to daily price dates and forward-fill (max 90 days = ~3 months)
    daily = wide.reindex(price_index).ffill(limit=90)
    return daily.fillna(0.0)
