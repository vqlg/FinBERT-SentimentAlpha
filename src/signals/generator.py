import numpy as np
import pandas as pd


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Normalise each column independently over a rolling lookback window."""
    mu = df.rolling(window, min_periods=max(1, window // 4)).mean()
    sigma = df.rolling(window, min_periods=max(1, window // 4)).std()
    return (df - mu) / sigma.clip(lower=1e-8)


def aggregate_daily_sentiment(scored_news: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse scored headlines to one row per (ticker, date).

    Headlines are confidence-weighted: a headline scored 0.95 positive
    counts more than one scored 0.51. This aligns with practitioner usage
    where model confidence proxies for news salience.

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
    return daily


def build_signals(
    daily_sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    vix: pd.Series,
    cfg,
) -> pd.DataFrame:
    """
    Construct the composite trading signal.

    Pipeline (each step motivated by academic literature):

    1. Pivot to wide format, forward-fill gaps ≤ 3 days.
       Rationale: markets trade on stale news when no fresh news arrives.

    2. EWMA smoothing (span = cfg.ewma_span).
       Rationale: reduces noise; confirmed by Tetlock (2007) to increase
       predictive power of smoothed vs raw sentiment.

    3. Three orthogonal components, each z-scored over cfg.zscore_window:
       - level    : EWMA sentiment level (absolute bullishness)
       - momentum : level.diff(cfg.momentum_window) (regime shifts)
       - surprise : raw - 20d MA (deviation from expected baseline)

    4. VIX regime filter.
       Rationale: Bollen (2011) and recent LLM research both show sentiment
       alpha collapses when implied volatility is elevated.

    5. Cross-sectional z-score within each date.
       Rationale: removes common-factor drift; makes signal directly
       comparable across tickers for portfolio ranking.

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

    # -- 3. Components --
    momentum = level_raw.diff(cfg.momentum_window)
    surprise = wide - wide.rolling(20, min_periods=5).mean()

    level_z = _rolling_zscore(level_raw, cfg.zscore_window)
    mom_z = _rolling_zscore(momentum, cfg.zscore_window)
    surp_z = _rolling_zscore(surprise, cfg.zscore_window)

    w = cfg.signal_weights
    combined = w["level"] * level_z + w["momentum"] * mom_z + w["surprise"] * surp_z

    # -- 4. VIX regime scaling --
    vix_aligned = vix.reindex(combined.index).ffill()
    scale = pd.Series(1.0, index=combined.index)
    mid = (vix_aligned > cfg.vix_scale_threshold) & (vix_aligned <= cfg.vix_exit_threshold)
    high = vix_aligned > cfg.vix_exit_threshold
    scale[mid] = 0.5
    scale[high] = 0.0
    combined = combined.multiply(scale, axis=0)

    # -- 5. Cross-sectional z-score --
    cs_mean = combined.mean(axis=1)
    cs_std = combined.std(axis=1).clip(lower=1e-8)
    signals = combined.subtract(cs_mean, axis=0).divide(cs_std, axis=0)

    return signals  # [dates × tickers], ~N(0,1) cross-sectionally
