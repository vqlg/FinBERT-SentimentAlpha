import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.request import Request, urlopen

import feedparser
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------

def _fetch_finviz(ticker: str) -> pd.DataFrame:
    """Scrape recent headlines from Finviz (~2 weeks of history)."""
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    try:
        req = Request(url=url, headers=_HEADERS)
        html = BeautifulSoup(urlopen(req, timeout=10), features="html.parser")
        table = html.find(id="news-table")
        if not table:
            return pd.DataFrame()

        rows, current_date = [], None
        for row in table.findAll("tr"):
            anchor = row.select_one("a.tab-link-news")
            date_cell = row.td
            if not anchor or not date_cell:
                continue

            parts = date_cell.text.strip().split()
            # Finviz omits the date on rows that share a date with the row above;
            # only the first row of a new date has the full "MMM-DD-YY HH:MMxM" string.
            if parts and ":" not in parts[0]:
                for fmt in ("%b-%d-%y", "%b %d, %Y"):
                    try:
                        current_date = datetime.strptime(" ".join(parts[:2]), fmt).date()
                        break
                    except ValueError:
                        pass
                else:
                    current_date = date.today()
            if current_date is None:
                current_date = date.today()

            rows.append({
                "ticker": ticker,
                "date": current_date,
                "headline": anchor.text.strip(),
                "source": "finviz",
            })
        return pd.DataFrame(rows)
    except (URLError, Exception) as exc:
        logger.warning("Finviz failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_yahoo_rss(ticker: str) -> pd.DataFrame:
    """Pull headlines from Yahoo Finance RSS feed — free and reliable."""
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    try:
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries:
            pub = entry.get("published_parsed")
            pub_date = datetime(*pub[:6]).date() if pub else date.today()
            title = entry.get("title", "").strip()
            if title:
                rows.append({
                    "ticker": ticker,
                    "date": pub_date,
                    "headline": title,
                    "source": "yahoo_rss",
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("Yahoo RSS failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_yfinance_news(ticker: str) -> pd.DataFrame:
    """Use yfinance's built-in news endpoint (~20 recent articles)."""
    try:
        articles = yf.Ticker(ticker).news or []
        rows = []
        for a in articles:
            ts = a.get("providerPublishTime", 0)
            pub_date = datetime.fromtimestamp(ts).date() if ts else date.today()
            title = a.get("title", "").strip()
            if title:
                rows.append({
                    "ticker": ticker,
                    "date": pub_date,
                    "headline": title,
                    "source": "yfinance",
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_all_news(
    tickers: List[str],
    rate_limit_secs: float = 0.6,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch headlines from all three sources and deduplicate.

    Cache writes to .cache/news_<date>.json so repeated runs on the same day
    don't re-scrape.  Set use_cache=False to force a fresh pull.
    """
    cache_path = _CACHE_DIR / f"news_{date.today().isoformat()}.json"
    if use_cache and cache_path.exists():
        logger.info("Loading news from cache: %s", cache_path)
        df = pd.read_json(cache_path, orient="records", convert_dates=["date"])
        return df[df["ticker"].isin(tickers)].reset_index(drop=True)

    frames = []
    for ticker in tickers:
        for fn in (_fetch_finviz, _fetch_yahoo_rss, _fetch_yfinance_news):
            df = fn(ticker)
            if not df.empty:
                frames.append(df)
        time.sleep(rate_limit_secs)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    # Same story often syndicated across sources — deduplicate on headline text.
    combined = combined.drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)
    combined = combined.sort_values(["ticker", "date"])

    if use_cache:
        _CACHE_DIR.mkdir(exist_ok=True)
        combined.to_json(cache_path, orient="records", date_format="iso")
        logger.info("News cached to %s", cache_path)

    return combined
