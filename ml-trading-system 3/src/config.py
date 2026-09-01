"""
Configuration for the ML Trading System.

All tunable parameters live in one dataclass so the pipeline is
reproducible: a single TradingConfig instance fully determines a run
(data window, features, labeling, training, risk management, execution).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TradingConfig:
    """Central configuration for the trading pipeline."""

    # --- Data ---
    start_date: str = '2008-01-01'
    end_date: str = '2025-12-31'
    tickers: Optional[List[str]] = None

    # --- Feature engineering ---
    lookback_periods: Optional[List[int]] = None
    use_volume_features: bool = True
    use_volatility_regime: bool = True
    use_market_internals: bool = True

    # --- Triple-barrier labeling (see label_generator.py) ---
    min_holding_period: int = 5    # minimum days before checking exit
    max_holding_period: int = 20   # maximum days to hold before forced exit
    profit_target: float = 0.03    # +3% profit barrier
    stop_loss: float = 0.02        # -2% stop-loss barrier

    # --- Neural network training ---
    batch_size: int = 1024
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping: int = 10

    # --- Ensemble / hyperparameter search ---
    n_models: int = 5
    use_optuna: bool = True
    n_trials: int = 50

    # --- Risk management ---
    max_position_pct: float = 0.15   # max 15% of portfolio per position
    max_portfolio_pct: float = 0.70  # max 70% of portfolio invested at once
    kelly_fraction: float = 0.20     # fractional (conservative) Kelly sizing

    # --- Execution simulation ---
    slippage_bps: float = 10
    commission_bps: float = 5
    min_volume_usd: float = 1_000_000
    confidence_threshold: float = 0.5
    use_stop_loss: bool = True
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.05

    # --- Backtest ---
    initial_capital: float = 100_000
    rebalance_frequency: int = 1  # daily

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                # Major index ETFs
                'SPY', 'QQQ', 'IWM', 'DIA',
                # Sector ETFs
                'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU',
                # Commodity / bond ETFs
                'GLD', 'SLV', 'TLT', 'IEF', 'HYG',
                # Mega-cap stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
                'JPM', 'V', 'MA', 'WMT', 'JNJ', 'PG',
            ]
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]


def get_default_config() -> TradingConfig:
    """Return a TradingConfig with default settings."""
    return TradingConfig()
