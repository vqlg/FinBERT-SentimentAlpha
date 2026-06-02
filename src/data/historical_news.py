"""
Finnhub historical company news fetcher.

Fetches in monthly chunks and caches each month to
  .cache/finnhub_{TICKER}_{YYYY}_{MM}.json
so subsequent backtest runs are instant (only uncached months hit the API).

Free-tier rate limit: 60 calls/minute.  With the default 1.1s sleep between
calls, 10 tickers × 36 months (2022-2024) ≈ 7 minutes on first run.
Every subsequent run reads from cache and completes in seconds.

Usage:
    export FINNHUB_API_KEY=your_key
    python main.py backtest
"""

import calendar
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import List
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
_BASE_URL  = "https://finnhub.io/api/v1"


def _api_key() -> str:
    return os.environ.get("FINNHUB_API_KEY", "")


def _fetch_month(ticker: str, year: int, month: int):
    """Return list of articles on success, None on network/API failure."""
    start = date(year, month, 1).isoformat()
    end   = date(year, month, calendar.monthrange(year, month)[1]).isoformat()
    url   = (
        f"{_BASE_URL}/company-news"
        f"?symbol={ticker}&from={start}&to={end}&token={_api_key()}"
    )
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode()) or []
    except Exception as exc:
        logger.warning("Finnhub news failed %s %d-%02d: %s", ticker, year, month, exc)
        return None


def fetch_finnhub_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    rate_limit_secs: float = 1.1,
) -> pd.DataFrame:
    """
    Return all Finnhub company headlines for *tickers* between *start_date*
    and *end_date* (inclusive, ISO format YYYY-MM-DD).

    Already-cached months are read from disk; new months are fetched from the
    API and written to cache before returning.

    Returns a DataFrame with columns [ticker, date, headline, source] —
    the same schema as fetch_all_news() so it drops straight into the
    existing scoring and signal pipeline.

    Returns an empty DataFrame (not an error) when FINNHUB_API_KEY is unset.
    """
    if not _api_key():
        logger.warning("FINNHUB_API_KEY not set — cannot fetch Finnhub historical news.")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    _CACHE_DIR.mkdir(exist_ok=True)

    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    # Build the ordered list of (year, month) tuples that span the range
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1

    total   = len(tickers) * len(months)
    call_no = 0
    rows: list[dict] = []

    for ticker in tickers:
        for year, month in months:
            call_no += 1
            cache_path = _CACHE_DIR / f"finnhub_{ticker}_{year}_{month:02d}.json"

            if cache_path.exists():
                with open(cache_path) as fh:
                    articles = json.load(fh)
            else:
                logger.info(
                    "[%d/%d] Fetching Finnhub news  %s  %d-%02d …",
                    call_no, total, ticker, year, month,
                )
                articles = _fetch_month(ticker, year, month)
                if articles is None:
                    # Network/API failure — don't cache so the month is retried next run
                    articles = []
                else:
                    with open(cache_path, "w") as fh:
                        json.dump(articles, fh)
                time.sleep(rate_limit_secs)

            for art in articles:
                ts = art.get("datetime", 0)
                if not ts:
                    continue
                pub_date = pd.Timestamp(ts, unit="s").normalize()
                headline = (art.get("headline") or "").strip()
                if headline:
                    rows.append({
                        "ticker":   ticker,
                        "date":     pub_date,
                        "headline": headline,
                        "source":   "finnhub",
                    })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Clip to the exact requested range, deduplicate syndicated stories
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df   = df[mask].drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)

    logger.info(
        "Finnhub historical news: %d headlines across %d tickers (%s → %s)",
        len(df), df["ticker"].nunique(), start_date, end_date,
    )
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
