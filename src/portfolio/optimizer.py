import numpy as np
import pandas as pd


def signals_to_weights(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """
    Convert z-scored signals into daily portfolio weights.

    Construction methodology:

    1. Tercile ranking: top 1/3 → long (+1), bottom 1/3 → short (-1), mid → flat.
       Tercile (vs quintile or continuous) is favoured in recent research for
       robustness — it avoids overfitting to precise signal magnitudes.

    2. Volatility scaling: divide each raw weight by that stock's realised vol,
       then target the portfolio to cfg.vol_target annualised volatility.
       This equalises risk contribution across names so that a high-vol name
       like TSLA doesn't dominate a low-vol name like JPM.

    3. Dollar neutralisation: rescale longs and shorts independently so that
       sum(longs) = 1 and sum(shorts) = -1. Removes market beta so the
       strategy earns pure sentiment alpha.

    4. Hard cap: clip any position at ±cfg.max_position.

    Returns a DataFrame of same shape as signals (dates × tickers), values in [-1, +1].
    """
    returns = prices.pct_change()
    # Annualised realised vol; clip at 5% floor to prevent infinite sizing
    realised_vol = (
        returns.rolling(cfg.vol_lookback, min_periods=5)
        .std()
        .mul(np.sqrt(252))
        .clip(lower=0.05)
    )

    weight_rows = []
    for dt, sig_row in signals.iterrows():
        valid = sig_row.dropna()
        if valid.empty:
            weight_rows.append(pd.Series(0.0, index=signals.columns, name=dt))
            continue

        n = len(valid)
        k = max(1, n // 3)  # tercile size

        ranked = valid.rank(ascending=True)
        raw = pd.Series(0.0, index=valid.index)
        raw[ranked >= n - k + 1] = 1.0   # long tercile
        raw[ranked <= k] = -1.0           # short tercile

        # Volatility scaling
        if dt in realised_vol.index:
            vol = realised_vol.loc[dt].reindex(valid.index).fillna(0.20)
        else:
            vol = pd.Series(0.20, index=valid.index)

        scaled = raw / vol

        # Dollar neutralisation
        long_sum = scaled[scaled > 0].sum()
        short_sum = scaled[scaled < 0].sum()
        if long_sum > 0:
            scaled[scaled > 0] /= long_sum
        if short_sum < 0:
            scaled[scaled < 0] /= abs(short_sum)

        # Vol targeting: rescale entire book so expected portfolio vol ≈ vol_target
        # Approximate: assume equal-weight correlations of 0.3 among positions
        n_pos = (raw != 0).sum()
        if n_pos > 0:
            approx_port_vol = vol[raw != 0].mean() / np.sqrt(n_pos) * np.sqrt(1 + 0.3 * (n_pos - 1))
            if approx_port_vol > 0:
                scaled *= cfg.vol_target / approx_port_vol

        # Hard position cap
        scaled = scaled.clip(lower=-cfg.max_position, upper=cfg.max_position)

        weight_rows.append(scaled.reindex(signals.columns).fillna(0.0).rename(dt))

    weights = pd.DataFrame(weight_rows)
    weights.index.name = "date"
    return weights
