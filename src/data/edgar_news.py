"""
SEC EDGAR 8-K filing fetcher for FinBERT-SentimentAlpha backtesting.

Completely free, no API key required. EDGAR is the SEC's public database
of company filings. 8-K filings are material event disclosures companies
must submit within 4 business days of significant events:

  Item 2.02  Results of Operations    — earnings releases (highest signal)
  Item 7.01  Reg FD Disclosure        — forward-looking guidance
  Item 5.02  Executive Changes        — CEO/CFO appointments or departures
  Item 1.01  Material Agreement       — major contracts and partnerships
  Item 2.01  M&A Completion           — acquisitions and dispositions
  Item 2.06  Material Impairment      — write-downs (bearish)
  Item 4.02  Non-Reliance             — accounting restatements (very bearish)

Exhibit titles (press release headlines like "Apple Reports Record Revenue")
are fetched from filing index pages and scored with FinBERT. Item-type
descriptions serve as fallback when no exhibit title is available.

EDGAR rate guideline: max 10 req/sec; we use 0.15s between calls (~6/sec).

Cache:
  .cache/edgar_ciks.json              — ticker→CIK map (refreshed weekly)
  .cache/edgar_{TICKER}_8k.json      — 8-K filing list per ticker (permanent)
  .cache/edgar_idx_{acc}.txt         — exhibit title per filing (permanent)
"""

import json
import logging
import time
from datetime import date
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
_HEADERS   = {"User-Agent": "FinBERT-SentimentAlpha research@sentimentalpha.com"}

_COMPANY_NAMES: Dict[str, str] = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
    "GOOGL": "Alphabet",
    "GOOG":  "Alphabet",
    "META":  "Meta",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "JPM":   "JPMorgan Chase",
    "V":     "Visa",
    "MA":    "Mastercard",
    "BRK.B": "Berkshire Hathaway",
    "UNH":   "UnitedHealth",
    "JNJ":   "Johnson & Johnson",
    "XOM":   "ExxonMobil",
    "BAC":   "Bank of America",
    "WMT":   "Walmart",
    "PG":    "Procter & Gamble",
    "HD":    "Home Depot",
    "CVX":   "Chevron",
}

_ITEM_LABELS: Dict[str, str] = {
    "1.01": "enters material agreement",
    "1.02": "terminates material agreement",
    "1.05": "reports cybersecurity incident",
    "2.01": "completes acquisition or disposition",
    "2.02": "reports financial results",
    "2.03": "incurs direct financial obligation",
    "2.05": "announces restructuring charges",
    "2.06": "records material impairment",
    "3.01": "receives exchange delisting notice",
    "4.02": "issues non-reliance on financial statements",
    "5.01": "reports change in control",
    "5.02": "executive departure or appointment",
    "7.01": "issues Reg FD guidance disclosure",
    "8.01": "reports other material event",
}


def _get(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug("EDGAR GET failed (%s): %s", url, exc)
        return None


def _cache_stale(path: Path, max_days: int) -> bool:
    if not path.exists():
        return True
    age = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
    return age >= max_days


def _get_ciks() -> Dict[str, str]:
    """Return {ticker: zero-padded-CIK}. Refreshed weekly."""
    cache_path = _CACHE_DIR / "edgar_ciks.json"
    if not _cache_stale(cache_path, max_days=7):
        with open(cache_path) as fh:
            return json.load(fh)

    raw = _get("https://www.sec.gov/files/company_tickers.json")
    if not raw:
        return {}
    data = json.loads(raw)
    ciks = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(ciks, fh)
    return ciks


def _get_8k_filings(ticker: str, cik: str) -> List[Dict]:
    """
    Return all 8-K filings for ticker from EDGAR submissions.
    Cached permanently — historical 8-Ks never change.
    Each dict: {accession, filing_date, items, cik}
    """
    cache_path = _CACHE_DIR / f"edgar_{ticker}_8k.json"
    if cache_path.exists():
        with open(cache_path) as fh:
            return json.load(fh)

    filings: List[Dict] = []

    def _extract(block: Dict) -> None:
        accs  = block.get("accessionNumber", [])
        dates = block.get("filingDate", [])
        forms = block.get("form", [])
        items = block.get("items", [])
        for acc, d, form, itm in zip_longest(accs, dates, forms, items, fillvalue=""):
            if form == "8-K":
                filings.append({
                    "accession":   acc,
                    "filing_date": d,
                    "items":       itm,
                    "cik":         cik,
                })

    raw = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not raw:
        return []
    data = json.loads(raw)
    _extract(data.get("filings", {}).get("recent", {}))

    for extra_file in data.get("filings", {}).get("files", []):
        time.sleep(0.15)
        extra_raw = _get(f"https://data.sec.gov/submissions/{extra_file['name']}")
        if extra_raw:
            _extract(json.loads(extra_raw))

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(filings, fh)
    return filings


def _get_exhibit_title(cik: str, accession: str) -> Optional[str]:
    """
    Fetch the Exhibit 99.x description from the 8-K filing index page.
    Returns the exhibit description (often the PR headline), or None.
    Cached permanently.
    """
    acc_no_dash = accession.replace("-", "")
    cache_path  = _CACHE_DIR / f"edgar_idx_{acc_no_dash}.txt"
    if cache_path.exists():
        text = cache_path.read_text().strip()
        return text or None

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{acc_no_dash}/{accession}-index.htm"
    )
    raw   = _get(url)
    title = None

    if raw:
        soup = BeautifulSoup(raw, "html.parser")
        for row in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 4:
                continue
            doc_type = cells[3]
            desc     = cells[1]
            if (
                doc_type.startswith("EX-99")
                and desc
                and len(desc) > 10
                and desc.lower() not in ("exhibit", "press release", "financial statements")
            ):
                title = desc
                break

    cache_path.write_text(title or "")
    return title


def _build_headline(ticker: str, items_str: str, exhibit_title: Optional[str]) -> str:
    """Prefer exhibit title (PR headline); fall back to item-based description."""
    if exhibit_title:
        return exhibit_title

    company   = _COMPANY_NAMES.get(ticker, ticker)
    item_nums = [i.strip() for i in items_str.split(",") if i.strip() != "9.01"]
    for num in item_nums:
        label = _ITEM_LABELS.get(num)
        if label:
            return f"{company} {label}"
    return f"{company} files 8-K disclosure"


def fetch_edgar_historical_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    rate_limit_secs: float = 0.15,
) -> pd.DataFrame:
    """
    Return SEC 8-K filing headlines for *tickers* between *start_date* and *end_date*.

    No API key required. Results cached to .cache/ — first run takes ~2 minutes
    for a 3-year backtest; all subsequent runs are instant.

    Returns DataFrame with columns [ticker, date, headline, source].
    """
    _CACHE_DIR.mkdir(exist_ok=True)

    start   = date.fromisoformat(start_date)
    end     = date.fromisoformat(end_date)
    cik_map = _get_ciks()
    rows: List[Dict] = []

    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            logger.warning("EDGAR: no CIK found for %s — skipping", ticker)
            continue

        all_filings = _get_8k_filings(ticker, cik)
        in_range = [
            f for f in all_filings
            if f["filing_date"] and start <= date.fromisoformat(f["filing_date"]) <= end
        ]

        if not in_range:
            logger.info("EDGAR: no 8-K filings for %s in %s → %s", ticker, start, end)
            continue

        logger.info(
            "EDGAR: fetching exhibit titles for %d 8-K filings (%s) …",
            len(in_range), ticker,
        )
        for filing in in_range:
            time.sleep(rate_limit_secs)
            exhibit  = _get_exhibit_title(cik, filing["accession"])
            headline = _build_headline(ticker, filing.get("items", ""), exhibit)
            rows.append({
                "ticker":   ticker,
                "date":     pd.Timestamp(filing["filing_date"]),
                "headline": headline,
                "source":   "edgar",
            })

    if not rows:
        logger.warning("EDGAR: no 8-K filings found for any ticker in date range.")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df   = df[mask].drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)

    logger.info(
        "EDGAR: %d 8-K filings across %d tickers (%s → %s)",
        len(df), df["ticker"].nunique(), start_date, end_date,
    )
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
