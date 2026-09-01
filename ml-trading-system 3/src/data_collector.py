"""
Market data collection and quality validation.

Pulls OHLCV data from Yahoo Finance and applies quality gates before it
ever reaches feature engineering: minimum history length, required
columns, a liquidity floor, and a missing-data ceiling.
"""

from typing import Dict, List

import pandas as pd
import yfinance as yf

from .config import TradingConfig


class DataCollector:
    """Fetch and validate OHLCV data for a list of tickers."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def fetch_data(self, tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch and validate market data.

        Args:
            tickers: Ticker symbols to fetch. Defaults to config.tickers.

        Returns:
            Dict mapping ticker -> validated OHLCV DataFrame.
        """
        if tickers is None:
            tickers = self.config.tickers

        print(f"Fetching market data for {len(tickers)} instruments...")

        data_dict: Dict[str, pd.DataFrame] = {}
        failed = []

        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                    auto_adjust=True,
                )

                # yfinance can return MultiIndex columns for some requests
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Need at least a year of history to build meaningful features
                if len(df) < 252:
                    failed.append(f"{ticker}: insufficient data ({len(df)} days)")
                    continue

                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    failed.append(f"{ticker}: missing required columns")
                    continue

                df['DollarVolume'] = df['Close'] * df['Volume']

                # Liquidity filter — avoid thinly traded instruments
                mean_dollar_vol = df['DollarVolume'].mean()
                if mean_dollar_vol < self.config.min_volume_usd:
                    failed.append(f"{ticker}: low volume (${mean_dollar_vol:,.0f})")
                    continue

                # Data quality filter — reject if too much is missing
                nan_pct = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
                if nan_pct > 0.05:
                    failed.append(f"{ticker}: too many NaNs ({nan_pct:.1%})")
                    continue

                df = df.fillna(method='ffill').fillna(method='bfill')
                df['Ticker'] = ticker
                data_dict[ticker] = df

            except Exception as e:
                failed.append(f"{ticker}: {str(e)[:50]}")

        print(f"Successfully loaded: {len(data_dict)} instruments")
        if failed:
            print(f"Failed to load: {len(failed)} instruments")
            for f in failed[:10]:
                print(f"   - {f}")

        return data_dict

    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return per-ticker summary statistics (date range, volume, return)."""
        summaries = []
        for ticker, df in data_dict.items():
            summaries.append({
                'ticker': ticker,
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'n_days': len(df),
                'avg_volume': df['Volume'].mean(),
                'avg_dollar_volume': df['DollarVolume'].mean(),
                'avg_price': df['Close'].mean(),
                'total_return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1),
            })
        return pd.DataFrame(summaries).sort_values('ticker')
