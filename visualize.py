import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def plot_dashboard(result: dict, benchmark=None, save_path: str = "dashboard.png"):
    """4-panel strategy performance dashboard."""
    pv = result["portfolio_value"]
    net = result["net_returns"]
    m = result["metrics"]

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("SentimentQuant — Performance Dashboard", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # -- Panel 1: Cumulative value --
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pv.index, pv.values, label="Strategy", color="#2196F3", linewidth=1.6)
    if benchmark is not None:
        bm = benchmark.reindex(pv.index).ffill().dropna()
        bm_norm = bm / bm.iloc[0] * pv.iloc[0]
        ax1.plot(bm_norm.index, bm_norm.values, label="SPY (benchmark)",
                 color="#FF9800", linewidth=1.1, alpha=0.85, linestyle="--")
    ax1.set_title("Portfolio Value")
    ax1.set_ylabel("USD")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # -- Panel 2: Drawdown --
    ax2 = fig.add_subplot(gs[1, 0])
    cum = (1 + net).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#EF5350", alpha=0.55, label="Drawdown")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("%")

    # -- Panel 3: Rolling 63-day Sharpe --
    ax3 = fig.add_subplot(gs[1, 1])
    roll_sharpe = net.rolling(63).mean() / net.rolling(63).std() * np.sqrt(252)
    ax3.plot(roll_sharpe.index, roll_sharpe.values, color="#66BB6A", linewidth=1.2)
    ax3.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax3.axhline(1, color="#66BB6A", linewidth=0.6, linestyle=":", alpha=0.6, label="Sharpe=1")
    ax3.set_title("Rolling 63-Day Sharpe")
    ax3.legend(fontsize=8)

    # -- Panel 4: Monthly returns heatmap --
    ax4 = fig.add_subplot(gs[2, :])
    monthly = m["monthly_returns"] * 100
    monthly.index = pd.to_datetime(monthly.index)
    year_month = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = year_month.pivot(index="year", columns="month", values="ret")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        ax=ax4, linewidths=0.4, cbar_kws={"label": "Return (%)"},
        annot_kws={"size": 8},
    )
    ax4.set_title("Monthly Returns (%)")
    ax4.set_xlabel("")
    ax4.set_ylabel("")

    # -- Metrics strip at bottom --
    summary = (
        f"Return: {m['annualized_return_pct']:.1f}%  |  "
        f"Vol: {m['annualized_vol_pct']:.1f}%  |  "
        f"Sharpe: {m['sharpe_ratio']:.2f}  |  "
        f"Sortino: {m['sortino_ratio']:.2f}  |  "
        f"Calmar: {m['calmar_ratio']:.2f}  |  "
        f"MaxDD: {m['max_drawdown_pct']:.1f}%  |  "
        f"Omega: {m['omega_ratio']:.2f}  |  "
        f"Win Rate: {m['win_rate_pct']:.1f}%"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", alpha=0.9))

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Dashboard saved → {save_path}")
    plt.show()


def plot_live_signals(signals_df: pd.DataFrame, scored_news: pd.DataFrame, save_path: str = "live_signals.png"):
    """Bar chart of current signals + FinBERT score distribution per ticker."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("SentimentQuant — Live Signals", fontsize=12, fontweight="bold")

    # Signal bars
    colors = ["#4CAF50" if v > 0 else "#EF5350" for v in signals_df["signal"]]
    bars = ax1.bar(signals_df["ticker"], signals_df["signal"], color=colors, alpha=0.85, edgecolor="white", width=0.6)
    ax1.axhline(0, color="white", linewidth=0.6)
    for bar, val in zip(bars, signals_df["signal"]):
        offset = 0.04 * (1 if val >= 0 else -1)
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{val:.2f}", ha="center", fontsize=8.5,
                 va="bottom" if val >= 0 else "top")
    ax1.set_title("Cross-Sectional Sentiment Signal (z-scored)")
    ax1.set_ylabel("Signal")
    ax1.set_ylim(signals_df["signal"].min() - 0.5, signals_df["signal"].max() + 0.5)

    # Score distribution boxplot
    if not scored_news.empty and "finbert_score" in scored_news.columns:
        tickers = signals_df["ticker"].tolist()
        groups = [scored_news.loc[scored_news["ticker"] == t, "finbert_score"].values for t in tickers]
        groups = [g for g in groups if len(g) > 0]
        valid_tickers = [t for t, g in zip(tickers, [scored_news.loc[scored_news["ticker"] == t, "finbert_score"].values for t in tickers]) if len(g) > 0]
        if groups:
            bp = ax2.boxplot(groups, labels=valid_tickers, patch_artist=True, medianprops={"color": "white", "linewidth": 1.5})
            for patch, grp in zip(bp["boxes"], groups):
                patch.set_facecolor("#4CAF50" if np.median(grp) > 0 else "#EF5350")
                patch.set_alpha(0.65)
    ax2.axhline(0, color="grey", linestyle="--", alpha=0.6)
    ax2.set_title("FinBERT Score Distribution by Ticker")
    ax2.set_ylabel("FinBERT Score [-1, +1]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Live signals chart saved → {save_path}")
    plt.show()
