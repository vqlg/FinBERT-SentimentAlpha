import logging
from typing import List, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: List[str], start: str, end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close_prices, volume) DataFrames, columns=tickers, index=trading days."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = raw["Volume"]
    else:
        # Single ticker: yfinance returns flat columns
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    logger.info("Fetched prices for %d tickers (%s → %s)", len(close.columns), start, end)
    return close, volume


def fetch_vix(start: str, end: str) -> pd.Series:
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"].squeeze().rename("VIX")
