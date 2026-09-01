"""
Triple Barrier labeling (Lopez de Prado, "Advances in Financial Machine
Learning").

Standard "predict tomorrow's return" labeling introduces look-ahead
bias and doesn't reflect how a trade actually plays out. Triple Barrier
labeling instead simulates entering a trade at each point in time and
checks which of three exits comes first:

  - Profit barrier:  price rises to the profit target
  - Stop barrier:     price falls to the stop-loss level
  - Time barrier:     neither is hit within max_holding_period days

The label is 1 ("good trade") if the profit barrier is hit, or if the
time barrier is hit with a return above a small positive threshold; 0
otherwise. This only ever uses realized future prices to construct the
label — the model itself is never given this information.
"""

import numpy as np
import pandas as pd

from .config import TradingConfig


class LabelGenerator:
    """Generate labels using the Triple Barrier method."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each row based on realized future price action.

        Args:
            df: DataFrame with a 'Close' column, sorted by date.

        Returns:
            DataFrame with added Label, ForwardReturn, HoldPeriod columns.
            Rows without enough future data to evaluate are dropped.
        """
        df = df.copy()

        labels = []
        forward_returns = []
        hold_periods = []

        for i in range(len(df)):
            if i >= len(df) - self.config.max_holding_period:
                labels.append(-1)  # not enough future data — filtered out below
                forward_returns.append(np.nan)
                hold_periods.append(np.nan)
                continue

            entry_price = df['Close'].iloc[i]

            max_return = -np.inf
            exit_period = self.config.max_holding_period
            hit_profit = False
            hit_stop = False

            for j in range(self.config.min_holding_period, self.config.max_holding_period + 1):
                if i + j >= len(df):
                    break

                future_price = df['Close'].iloc[i + j]
                ret = (future_price - entry_price) / entry_price

                if ret >= self.config.profit_target:
                    max_return = ret
                    exit_period = j
                    hit_profit = True
                    break

                if ret <= -self.config.stop_loss:
                    max_return = ret
                    exit_period = j
                    hit_stop = True
                    break

                if ret > max_return:
                    max_return = ret
                    exit_period = j

            if hit_profit:
                label = 1
            elif hit_stop:
                label = 0
            else:
                # Time barrier hit — require at least a modest positive return
                label = 1 if max_return > 0.01 else 0

            labels.append(label)
            forward_returns.append(max_return)
            hold_periods.append(exit_period)

        df['Label'] = labels
        df['ForwardReturn'] = forward_returns
        df['HoldPeriod'] = hold_periods

        return df[df['Label'] != -1].copy()

    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """Summary statistics about the generated labels for a sanity check."""
        return {
            'total_samples': len(df),
            'positive_count': int((df['Label'] == 1).sum()),
            'negative_count': int((df['Label'] == 0).sum()),
            'positive_pct': (df['Label'] == 1).mean() * 100,
            'avg_forward_return': df['ForwardReturn'].mean(),
            'avg_hold_period': df['HoldPeriod'].mean(),
            'profit_hit_pct': (df['ForwardReturn'] >= self.config.profit_target).mean() * 100,
            'stop_hit_pct': (df['ForwardReturn'] <= -self.config.stop_loss).mean() * 100,
        }
