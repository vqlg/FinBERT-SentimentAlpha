from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class NLPConfig:
    # ProsusAI/finbert trained on Financial PhraseBank achieves ~97% on FPB benchmark,
    # outperforming yiyanghkust/finbert-tone for general financial headline classification.
    finbert_model: str = "ProsusAI/finbert"
    batch_size: int = 32
    max_length: int = 128
    device: str = "auto"   # "auto" | "cpu" | "cuda" | "mps"
    finbert_weight: float = 0.75
    vader_weight: float = 0.25


@dataclass
class SignalConfig:
    # Sentiment decay: news impact on returns has ~3-day half-life for individual stocks
    # (Tetlock 2007; Bollen et al. 2011). Longer for macro regimes (20-60 days).
    decay_halflife: float = 3.0
    ewma_span: int = 5
    # Multi-horizon sentiment momentum — 3/7/15-day windows capture short, medium, and
    # slow regime shifts in sentiment. Weights from Trading-R1 (arXiv:2509.11420).
    momentum_horizons: List[int] = field(default_factory=lambda: [3, 7, 15])
    momentum_horizon_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.2])
    zscore_window: int = 60         # Rolling window for time-series z-score
    min_headlines: int = 1

    # RSI and MACD parameters (Wilder RSI + classic MACD).
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_span: int = 9

    # Weights across all five components. Sentiment dominates (60%), technicals add 40%.
    # Alpha-R1 ablation shows news+price → Sharpe 1.62 vs news-only 1.15 (arXiv:2512.23515).
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        "level":    0.25,   # Smoothed absolute sentiment level
        "momentum": 0.25,   # Multi-horizon sentiment momentum
        "surprise": 0.10,   # Sentiment deviation from slow MA
        "rsi":      0.25,   # Price RSI (orthogonal to sentiment)
        "macd":     0.15,   # MACD histogram (trend confirmation)
    })

    # Research shows sentiment alpha degrades significantly during high-vol regimes.
    # Scale down at VIX > 25, go flat at VIX > 35.
    vix_scale_threshold: float = 25.0
    vix_exit_threshold: float = 35.0


@dataclass
class PortfolioConfig:
    vol_target: float = 0.15      # 15% annualized portfolio volatility target
    vol_lookback: int = 20
    max_position: float = 0.25    # Hard cap: no single name > 25% of book
    dollar_neutral: bool = True   # Long book = short book (removes market beta)
    # Tercile split preferred over continuous signal for out-of-sample robustness;
    # continuous sizing overfits to signal magnitude calibration (research consensus).
    tercile_split: bool = True
    # Slot rotation: divide capital into H sub-portfolios, rebalance 1/H per day.
    # Reduces daily turnover by 5x with no loss of coverage (Alpha-R1, arXiv:2512.23515).
    slot_rotation: bool = True
    holding_period: int = 5


@dataclass
class BacktestConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 10.0   # 10bps/side is realistic for retail execution
    risk_free_rate: float = 0.05
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "JPM", "V", "MA",
    ])


@dataclass
class Config:
    nlp: NLPConfig = field(default_factory=NLPConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


CFG = Config()
