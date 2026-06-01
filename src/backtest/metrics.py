import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Full suite of risk-adjusted performance metrics."""
    ann = 252
    rf_daily = (1 + risk_free_rate) ** (1 / ann) - 1
    excess = returns - rf_daily

    ann_return = float((1 + returns.mean()) ** ann - 1)
    ann_vol = float(returns.std() * np.sqrt(ann))
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann)) if excess.std() > 0 else 0.0

    downside = returns[returns < rf_daily]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(ann))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Omega ratio: probability-weighted ratio of gains to losses above/below RF
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    omega = float(gains / losses) if losses > 0 else float("inf")

    win_rate = float((returns > 0).mean())
    avg_win = float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0
    avg_loss = float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    monthly = returns.resample("ME").apply(lambda x: float((1 + x).prod() - 1))

    return {
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "omega_ratio": round(omega, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "monthly_returns": monthly,
    }
