import numpy as np
import pandas as pd


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Normalise each column independently over a rolling lookback window."""
    mu = df.rolling(window, min_periods=max(1, window // 4)).mean()
    sigma = df.rolling(window, min_periods=max(1, window // 4)).std()
    return (df - mu) / sigma.clip(lower=1e-8)


def _compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Wilder RSI mapped to a signed signal.

    Standard RSI ∈ [0, 100], centred at 50.  We subtract 50 and divide by 25 so
    the output sits in roughly [-2, +2]: positive = momentum bullish, negative = bearish.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(span=window, min_periods=window).mean()
    rs = gain / loss.clip(lower=1e-8)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 25


def _compute_macd_histogram(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD histogram (MACD line − signal line), normalised by price level.

    Normalising by price converts the raw dollar difference into a percentage-of-price
    measure, making cross-sectional comparison between tickers meaningful.
    """
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return histogram / prices.clip(lower=1e-8) * 100  # as % of price


def aggregate_daily_sentiment(
    scored_news: pd.DataFrame,
    decay_halflife: float = 3.0,
) -> pd.DataFrame:
    """
    Collapse scored headlines to one row per (ticker, date).

    Headlines are confidence-weighted: a headline scored 0.95 positive counts more
    than one scored 0.51.  Temporal decay across dates is handled downstream by
    EWMA smoothing in build_signals; applying it here with a global reference date
    causes float underflow for old articles in multi-year backtests.

    Returns long-format DataFrame: columns = [ticker, date, raw_sentiment].
    """
    df = scored_news.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    def _wavg(g: pd.DataFrame) -> float:
        w = g["finbert_score"].abs().clip(lower=0.01)
        return float(np.average(g["sentiment_score"], weights=w))

    daily = (
        df.groupby(["ticker", "date"])
        .apply(_wavg)
        .reset_index(name="raw_sentiment")
    )
    return daily.dropna()


def build_signals(
    daily_sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    vix: pd.Series,
    cfg,
) -> pd.DataFrame:
    """
    Construct the composite trading signal from five orthogonal components.

    Pipeline:

    1. Pivot sentiment to wide format, forward-fill gaps ≤ 3 days.

    2. EWMA smoothing (span = cfg.ewma_span).

    3. Five z-scored components:
       - level     : EWMA sentiment level (absolute bullishness)
       - momentum  : multi-horizon sentiment momentum, weighted sum of 3/7/15-day diffs
                     (Trading-R1, arXiv:2509.11420)
       - surprise  : raw sentiment vs 20d MA (mean-reversion component)
       - rsi       : Wilder RSI mapped to signed signal (price momentum)
       - macd      : MACD histogram / price (trend confirmation)
       RSI and MACD are orthogonal to sentiment; Alpha-R1 ablation shows their addition
       raises Sharpe from 1.15 to 1.62 (arXiv:2512.23515).

    4. VIX regime filter: scale 0.5× at VIX > 25, flat at VIX > 35.

    5. Cross-sectional z-score within each date.

    Returns wide DataFrame [dates × tickers] of z-scored composite signals.
    """
    # -- 1. Wide format aligned to price index --
    price_dates = prices.index.normalize()
    wide = (
        daily_sentiment.pivot(index="date", columns="ticker", values="raw_sentiment")
        .reindex(price_dates)
        .ffill(limit=3)
    )

    # -- 2. EWMA smoothing --
    level_raw = wide.ewm(span=cfg.ewma_span, min_periods=1).mean()

    # -- 3a. Sentiment components --
    # Multi-horizon momentum: weighted combination of short/medium/long-term diffs.
    mom_z = sum(
        w * _rolling_zscore(level_raw.diff(h), cfg.zscore_window)
        for h, w in zip(cfg.momentum_horizons, cfg.momentum_horizon_weights)
    )
    surprise = wide - wide.rolling(20, min_periods=5).mean()
    level_z = _rolling_zscore(level_raw, cfg.zscore_window)
    surp_z = _rolling_zscore(surprise, cfg.zscore_window)

    # -- 3b. Technical components (computed from prices, aligned to same index) --
    rsi_raw = _compute_rsi(prices, cfg.rsi_window).reindex(price_dates)
    macd_raw = _compute_macd_histogram(
        prices, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_span
    ).reindex(price_dates)

    # Restrict to tickers present in the sentiment wide frame
    shared_cols = wide.columns
    rsi_z = _rolling_zscore(rsi_raw[shared_cols], cfg.zscore_window)
    macd_z = _rolling_zscore(macd_raw[shared_cols], cfg.zscore_window)

    w = cfg.signal_weights
    combined = (
        w["level"]    * level_z
        + w["momentum"] * mom_z
        + w["surprise"] * surp_z
        + w["rsi"]      * rsi_z
        + w["macd"]     * macd_z
    )

    # -- 4. VIX regime scaling --
    vix_aligned = vix.reindex(combined.index).ffill()
    scale = pd.Series(1.0, index=combined.index)
    mid  = (vix_aligned > cfg.vix_scale_threshold) & (vix_aligned <= cfg.vix_exit_threshold)
    high = vix_aligned > cfg.vix_exit_threshold
    scale[mid]  = 0.5
    scale[high] = 0.0
    combined = combined.multiply(scale, axis=0)

    # -- 5. Cross-sectional z-score --
    cs_mean = combined.mean(axis=1)
    cs_std  = combined.std(axis=1).clip(lower=1e-8)
    signals = combined.subtract(cs_mean, axis=0).divide(cs_std, axis=0)

    return signals  # [dates × tickers], ~N(0,1) cross-sectionally
