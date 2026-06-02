"""
Alpha Vantage NEWS_SENTIMENT — free historical financial news for backtesting.

Requires ALPHAVANTAGE_API_KEY env var (free at alphavantage.co).

Free tier: 25 calls/day, 5 calls/minute.  We batch all tickers into one
request per quarter, so a 3-year backtest needs only 12 API calls and
completes in ~3 minutes on first run.  Subsequent runs use cache.

Each article includes per-ticker relevance scores.  We filter to articles
where relevance_score >= 0.3 to reduce noise from tangential mentions.

Cache: .cache/alphavantage_{YYYY}_Q{N}.json
"""

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
_BASE_URL  = "https://www.alphavantage.co/query"

_QUARTER_RANGES = [
    ("01-01", "03-31"),
    ("04-01", "06-30"),
    ("07-01", "09-30"),
    ("10-01", "12-31"),
]


def _api_key() -> str:
    return os.environ.get("ALPHAVANTAGE_API_KEY", "")


def _fetch_quarter(tickers: List[str], year: int, quarter: int) -> Optional[list]:
    """
    Fetch up to 1000 articles for all tickers in one quarter.
    Returns list of article dicts, or None on failure.
    """
    start_m, end_m = _QUARTER_RANGES[quarter - 1]
    time_from = f"{year}{start_m.replace('-', '')}T0000"
    time_to   = f"{year}{end_m.replace('-', '')}T2359"
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ",".join(tickers),
        "time_from": time_from,
        "time_to":   time_to,
        "limit":     1000,
        "sort":      "EARLIEST",
        "apikey":    _api_key(),
    }
    url = f"{_BASE_URL}?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "FinBERT-SentimentAlpha/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if "Information" in data:
            logger.warning("Alpha Vantage API message: %s", data["Information"])
            return None
        if "Note" in data:
            logger.warning("Alpha Vantage note: %s", data["Note"])
            return None
        if "Error Message" in data:
            logger.warning("Alpha Vantage error: %s", data["Error Message"])
            return None
        return data.get("feed") or []
    except Exception as exc:
        logger.warning("Alpha Vantage query failed (%d Q%d): %s", year, quarter, exc)
        return None


def fetch_alphavantage_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    relevance_threshold: float = 0.3,
    rate_limit_secs: float = 13.0,  # 5 calls/min free tier → 12s minimum
) -> pd.DataFrame:
    """
    Return Alpha Vantage headlines for *tickers* between *start_date* and *end_date*.

    Fetches quarterly with all tickers batched per call.  A 3-year backtest
    uses 12 API calls, well within the free tier's 25 calls/day limit.

    Articles are filtered by per-ticker relevance_score >= relevance_threshold
    to drop tangential mentions.

    Returns DataFrame with columns [ticker, date, headline, source].
    """
    if not _api_key():
        logger.warning("ALPHAVANTAGE_API_KEY not set — cannot fetch historical news.")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    _CACHE_DIR.mkdir(exist_ok=True)

    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    # Alpha Vantage free tier covers roughly the last 2 years of news.
    # Warn early rather than silently burning API calls that will return nothing.
    today         = date.today()
    earliest_free = today.replace(year=today.year - 2)
    if start < earliest_free:
        logger.warning(
            "Alpha Vantage free tier covers roughly the last 2 years "
            "(%s → today). Requested start %s is outside that window — "
            "quarters before %s will return no articles. "
            "Upgrade to a paid plan or shorten the backtest window.",
            earliest_free, start, earliest_free,
        )
    if end > today:
        logger.warning(
            "Requested end date %s is in the future — Alpha Vantage "
            "will reject those quarters with 'Invalid inputs'.",
            end,
        )

    quarters = []
    for y in range(start.year, end.year + 1):
        for q, (start_m, end_m) in enumerate(_QUARTER_RANGES, 1):
            q_start = date.fromisoformat(f"{y}-{start_m}")
            q_end   = date.fromisoformat(f"{y}-{end_m}")
            if q_start > end or q_end < start:
                continue
            quarters.append((y, q))

    total = len(quarters)
    rows: list[dict] = []

    for idx, (year, quarter) in enumerate(quarters, 1):
        cache_path = _CACHE_DIR / f"alphavantage_{year}_Q{quarter}.json"

        if cache_path.exists():
            with open(cache_path) as fh:
                articles = json.load(fh)
        else:
            logger.info(
                "[%d/%d] Fetching Alpha Vantage news  %d Q%d …",
                idx, total, year, quarter,
            )
            articles = _fetch_quarter(tickers, year, quarter)

            if articles is None:
                articles = []  # failure — skip cache so the quarter is retried next run
            else:
                with open(cache_path, "w") as fh:
                    json.dump(articles, fh)

            if idx < total:
                time.sleep(rate_limit_secs)

        for art in articles:
            title    = (art.get("title") or "").strip()
            time_pub = art.get("time_published", "")
            if not title or not time_pub:
                continue
            try:
                pub_date = pd.Timestamp(time_pub).normalize()
            except Exception:
                continue

            for ts in art.get("ticker_sentiment", []):
                ticker = ts.get("ticker", "")
                if ticker not in tickers:
                    continue
                try:
                    relevance = float(ts.get("relevance_score", 0))
                except (ValueError, TypeError):
                    relevance = 0.0
                if relevance < relevance_threshold:
                    continue
                rows.append({
                    "ticker":   ticker,
                    "date":     pub_date,
                    "headline": title,
                    "source":   "alphavantage",
                })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df   = df[mask].drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)

    logger.info(
        "Alpha Vantage historical news: %d headlines across %d tickers (%s → %s)",
        len(df), df["ticker"].nunique(), start_date, end_date,
    )
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
