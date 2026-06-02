"""
GDELT 2.0 Doc API — free historical news for backtesting.

No API key required.  GDELT indexes news from thousands of sources worldwide
going back to 2015.  We query by ticker + company name in monthly chunks,
cache each month to .cache/gdelt_{TICKER}_{YYYY}_{MM}.json, and return a
DataFrame with the same schema as fetch_finnhub_historical_news so it drops
straight into the existing pipeline.

GDELT caps results at 250 per request.  Monthly chunks are sufficient for
most tickers; high-coverage names (AAPL, TSLA) may hit the cap in busy months
but still provide strong signal density.

Rate limit: ~1 request per second (GDELT's informal guideline).
First run: ~7 min for 10 tickers × 36 months.  Subsequent runs use cache.
"""

import calendar
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
_BASE_URL  = "https://api.gdeltproject.org/api/v2/doc/doc"

_COMPANY_NAMES = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
    "GOOGL": "Alphabet",
    "GOOG":  "Alphabet",
    "META":  "Meta",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "JPM":   "JPMorgan",
    "V":     "Visa",
    "MA":    "Mastercard",
    "BRK.B": "Berkshire Hathaway",
    "UNH":   "UnitedHealth",
    "JNJ":   "Johnson Johnson",
    "XOM":   "ExxonMobil",
    "BAC":   "Bank of America",
    "WMT":   "Walmart",
    "PG":    "Procter Gamble",
    "HD":    "Home Depot",
    "CVX":   "Chevron",
}


def _gdelt_query(ticker: str, start_dt: str, end_dt: str) -> Optional[list]:
    """
    Fetch up to 250 articles from GDELT for ticker/company between start_dt and end_dt.
    start_dt / end_dt: "YYYYMMDDHHMMSS".  Returns None on persistent failure.
    Retries up to 3 times with exponential backoff on 429 rate-limit responses.
    """
    company = _COMPANY_NAMES.get(ticker, ticker)
    query   = f'"{company}" OR "{ticker}" sourcelang:english'
    params  = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    250,
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
        "format":        "json",
        "sort":          "DateDesc",
    }
    url = f"{_BASE_URL}?{urlencode(params)}"
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "FinBERT-SentimentAlpha/1.0"})
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
                return data.get("articles") or []
        except Exception as exc:
            wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
            if attempt < 2:
                logger.warning(
                    "GDELT query failed (%s %s→%s): %s — retrying in %ds",
                    ticker, start_dt, end_dt, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "GDELT query failed (%s %s→%s): %s — giving up",
                    ticker, start_dt, end_dt, exc,
                )
    return None


def _parse_seendate(seendate: str) -> Optional[pd.Timestamp]:
    """Parse GDELT seendate format: '20220115T120000Z'."""
    try:
        return pd.Timestamp(seendate, tz="UTC").normalize().tz_localize(None)
    except Exception:
        return None


def fetch_gdelt_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    rate_limit_secs: float = 3.0,
) -> pd.DataFrame:
    """
    Return GDELT headlines for *tickers* between *start_date* and *end_date*.

    Fetches in monthly chunks and caches each month so subsequent runs are
    instant.  Returns a DataFrame with columns [ticker, date, headline, source]
    — same schema as fetch_finnhub_historical_news.
    """
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

    total   = len(tickers) * len(months)
    call_no = 0
    rows: list[dict] = []

    for ticker in tickers:
        for year, month in months:
            call_no += 1
            cache_path = _CACHE_DIR / f"gdelt_{ticker}_{year}_{month:02d}.json"

            if cache_path.exists():
                with open(cache_path) as fh:
                    articles = json.load(fh)
            else:
                logger.info(
                    "[%d/%d] Fetching GDELT news  %s  %d-%02d …",
                    call_no, total, ticker, year, month,
                )
                last_day  = calendar.monthrange(year, month)[1]
                start_dt  = f"{year}{month:02d}01000000"
                end_dt    = f"{year}{month:02d}{last_day:02d}235959"
                articles  = _gdelt_query(ticker, start_dt, end_dt)

                if articles is None:
                    articles = []  # network failure — skip cache so the month is retried
                else:
                    with open(cache_path, "w") as fh:
                        json.dump(articles, fh)

                time.sleep(rate_limit_secs)

            for art in articles:
                title    = (art.get("title") or "").strip()
                seendate = art.get("seendate", "")
                if not title or not seendate:
                    continue
                pub_date = _parse_seendate(seendate)
                if pub_date is None:
                    continue
                rows.append({
                    "ticker":   ticker,
                    "date":     pub_date,
                    "headline": title,
                    "source":   "gdelt",
                })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df        = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df   = df[mask].drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)

    logger.info(
        "GDELT historical news: %d headlines across %d tickers (%s → %s)",
        len(df), df["ticker"].nunique(), start_date, end_date,
    )
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
