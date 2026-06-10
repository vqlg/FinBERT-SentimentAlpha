"""
Alpaca News API fetcher for FinBERT-SentimentAlpha backtesting.

Requires ALPACA_API_KEY and ALPACA_API_SECRET env vars.
Free account at alpaca.markets — no credit card required.

Free tier: 200 calls/min, historical data since 2016.
The 15-minute real-time delay does not apply to historical backtesting.

All tickers are batched into a single request per month, then paginated
automatically using next_page_token (50 articles per page).

Cache: .cache/alpaca_{YYYY}_{MM}.json — one file per calendar month,
all tickers combined.  If you change your ticker list, delete the relevant
month files so they are re-fetched with the updated symbol set.

First run: ~3-5 min for 10 tickers × 36 months depending on article volume.
Subsequent runs: instant (cache).
"""

import calendar
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
_BASE_URL  = "https://data.alpaca.markets/v1beta1/news"


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_API_SECRET", ""),
        "Accept":              "application/json",
    }


def _api_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET"))


def _fetch_month(tickers: List[str], year: int, month: int) -> Optional[list]:
    """
    Fetch all news articles for tickers in the given month, paginating
    automatically.  Returns a list of raw article dicts, or None on failure.
    """
    last_day   = calendar.monthrange(year, month)[1]
    start_iso  = f"{year}-{month:02d}-01T00:00:00Z"
    end_iso    = f"{year}-{month:02d}-{last_day:02d}T23:59:59Z"

    articles: list = []
    page_token: Optional[str] = None

    while True:
        params: dict = {
            "symbols":         ",".join(tickers),
            "start":           start_iso,
            "end":             end_iso,
            "limit":           50,
            "sort":            "asc",
            "include_content": "false",
        }
        if page_token:
            params["page_token"] = page_token

        url = f"{_BASE_URL}?{urlencode(params)}"
        try:
            req = Request(url, headers=_headers())
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            articles.extend(data.get("news") or [])
            page_token = data.get("next_page_token")
            if not page_token:
                break
        except Exception as exc:
            logger.warning("Alpaca news failed %d-%02d: %s", year, month, exc)
            return None

    return articles


def fetch_alpaca_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    rate_limit_secs: float = 0.35,
) -> pd.DataFrame:
    """
    Return Alpaca news headlines for *tickers* between *start_date* and
    *end_date* (ISO format YYYY-MM-DD).

    Fetches monthly with all tickers batched per call and paginates
    automatically.  Cached months are read from disk; uncached months
    hit the API and are written to cache.

    Returns DataFrame with columns [ticker, date, headline, source].
    """
    if not _api_configured():
        logger.warning(
            "ALPACA_API_KEY / ALPACA_API_SECRET not set — "
            "cannot fetch Alpaca news.  Set both env vars and retry."
        )
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    _CACHE_DIR.mkdir(exist_ok=True)

    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1

    rows: list[dict] = []
    total = len(months)

    for idx, (year, month) in enumerate(months, 1):
        cache_path = _CACHE_DIR / f"alpaca_{year}_{month:02d}.json"

        if cache_path.exists():
            with open(cache_path) as fh:
                articles = json.load(fh)
        else:
            logger.info(
                "[%d/%d] Fetching Alpaca news  %d-%02d …",
                idx, total, year, month,
            )
            articles = _fetch_month(tickers, year, month)
            if articles is None:
                articles = []  # network failure — skip cache so month is retried
            else:
                with open(cache_path, "w") as fh:
                    json.dump(articles, fh)
            if idx < total:
                time.sleep(rate_limit_secs)

        ticker_set = set(tickers)
        for art in articles:
            headline = (art.get("headline") or "").strip()
            created  = art.get("created_at", "")
            symbols  = art.get("symbols") or []
            if not headline or not created:
                continue
            try:
                pub_date = pd.Timestamp(created).normalize().tz_localize(None)
            except Exception:
                continue
            for ticker in symbols:
                if ticker in ticker_set:
                    rows.append({
                        "ticker":   ticker,
                        "date":     pub_date,
                        "headline": headline,
                        "source":   "alpaca",
                    })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df   = df[mask].drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)

    logger.info(
        "Alpaca news: %d headlines across %d tickers (%s → %s)",
        len(df), df["ticker"].nunique(), start_date, end_date,
    )
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
