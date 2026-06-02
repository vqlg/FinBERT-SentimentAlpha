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

    2. Volatility scaling: divide each raw weight by that stock's realised vol,
       then target the portfolio to cfg.vol_target annualised volatility.

    3. Dollar neutralisation: rescale longs and shorts so sum(longs)=1, sum(shorts)=-1.

    4. Hard cap: clip any position at ±cfg.max_position.

    5. Slot rotation (when cfg.slot_rotation=True): divide capital into H=cfg.holding_period
       independent sub-portfolios.  On day t, only slot (t mod H) is rebalanced; the other
       H-1 slots carry forward unchanged.  This reduces average daily turnover by H× while
       maintaining full coverage — the key practical improvement from Alpha-R1 (arXiv:2512.23515).

    Returns a DataFrame of same shape as signals (dates × tickers), values in [-1, +1].
    """
    returns = prices.pct_change()
    realised_vol = (
        returns.rolling(cfg.vol_lookback, min_periods=5)
        .std()
        .mul(np.sqrt(252))
        .clip(lower=0.05)
    )

    def _compute_target(sig_row: pd.Series, dt) -> pd.Series:
        """Compute the full-rebalance target weight vector for one date."""
        valid = sig_row.dropna()
        if valid.empty:
            return pd.Series(0.0, index=signals.columns)

        n = len(valid)
        k = max(1, n // 3)

        ranked = valid.rank(ascending=True)
        raw = pd.Series(0.0, index=valid.index)
        raw[ranked >= n - k + 1] = 1.0
        raw[ranked <= k] = -1.0

        if dt in realised_vol.index:
            vol = realised_vol.loc[dt].reindex(valid.index).fillna(0.20)
        else:
            vol = pd.Series(0.20, index=valid.index)

        scaled = raw / vol

        long_sum = scaled[scaled > 0].sum()
        short_sum = scaled[scaled < 0].sum()
        if long_sum > 0:
            scaled[scaled > 0] /= long_sum
        if short_sum < 0:
            scaled[scaled < 0] /= abs(short_sum)

        n_pos = (raw != 0).sum()
        if n_pos > 0:
            approx_port_vol = vol[raw != 0].mean() / np.sqrt(n_pos) * np.sqrt(1 + 0.3 * (n_pos - 1))
            if approx_port_vol > 0:
                scaled *= cfg.vol_target / approx_port_vol

        scaled = scaled.clip(lower=-cfg.max_position, upper=cfg.max_position)
        return scaled.reindex(signals.columns).fillna(0.0)

    if not cfg.slot_rotation:
        # Original daily-rebalance path
        rows = [
            _compute_target(signals.loc[dt], dt).rename(dt)
            for dt in signals.index
        ]
        weights = pd.DataFrame(rows)
        weights.index.name = "date"
        return weights

    # --- Slot rotation ---
    # Pre-compute full target weights for every date, then apply the rotation mask.
    # On day i, only slot (i mod H) updates; the other slots carry the previous weight
    # for that slot forward.  Each slot is an independent sub-portfolio, so the combined
    # daily portfolio is the average of all H slots at each time step.
    H = cfg.holding_period
    dates = signals.index
    n_dates = len(dates)

    # Shape: (n_dates, n_tickers)
    target_matrix = np.zeros((n_dates, len(signals.columns)))
    for i, dt in enumerate(dates):
        target_matrix[i] = _compute_target(signals.loc[dt], dt).values

    # slots[s, i, :] = weight vector for slot s at date i
    slots = np.zeros((H, n_dates, len(signals.columns)))
    for i in range(n_dates):
        active_slot = i % H
        for s in range(H):
            if s == active_slot:
                slots[s, i] = target_matrix[i]
            elif i > 0:
                slots[s, i] = slots[s, i - 1]  # carry forward
            # else: slot starts at zero

    # Combined portfolio = average across slots
    combined = slots.mean(axis=0)
    weights = pd.DataFrame(combined, index=dates, columns=signals.columns)
    weights.index.name = "date"
    return weights
