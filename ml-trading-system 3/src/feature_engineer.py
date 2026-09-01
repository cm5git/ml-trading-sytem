"""
Feature engineering for the trading system.

Every feature is computed with backward-looking windows only
(`.rolling()`, `.shift()`, `.diff()`, `.ewm()`), so no feature at time t
uses information from t+1 onward. This is the first of two places
look-ahead bias is deliberately avoided — the second is the triple
barrier labeling in label_generator.py.
"""

from typing import List

import numpy as np
import pandas as pd

from .config import TradingConfig


class FeatureEngineer:
    """Create technical-indicator features with no look-ahead leakage."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from OHLCV data.

        Args:
            df: DataFrame with Open/High/Low/Close/Volume columns.

        Returns:
            DataFrame with added feature columns.
        """
        df = df.copy()

        # Returns and volume change at multiple horizons
        for period in self.config.lookback_periods:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
            df[f'volume_change_{period}d'] = df['Volume'].pct_change(period)

        # Realized volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}d'] = df['Close'].pct_change().rolling(period).std()
            df[f'volume_std_{period}d'] = df['Volume'].rolling(period).std()

        # Intraday price patterns
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

        # Moving averages and price-to-MA ratios
        for period in [10, 20, 50, 200]:
            df[f'ma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_to_ma_{period}'] = df['Close'] / df[f'ma_{period}']

        # RSI
        for period in [14, 28]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Volume-based features
        if self.config.use_volume_features:
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
            df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()

        # Volatility regime: short-term vol relative to long-term vol
        if self.config.use_volatility_regime:
            vol_20 = df['Close'].pct_change().rolling(20).std()
            vol_60 = df['Close'].pct_change().rolling(60).std()
            df['vol_regime'] = vol_20 / (vol_60 + 1e-10)

        # Momentum and rate of change
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return feature column names, excluding metadata/target/raw-MA columns."""
        exclude_cols = [
            'Ticker', 'Label', 'ForwardReturn', 'HoldPeriod',
            'Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume',
        ]
        return [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith('ma_')  # keep price_to_ma ratios, drop raw MAs
            and pd.api.types.is_numeric_dtype(df[col])
        ]
