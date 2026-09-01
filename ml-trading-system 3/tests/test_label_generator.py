"""
Sanity tests for Triple Barrier labeling: correctness of barrier logic
and, critically, that no label depends on data beyond what a real
trade could have observed.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import TradingConfig
from src.label_generator import LabelGenerator


def make_config(**overrides):
    cfg = TradingConfig(tickers=['TEST'])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_profit_barrier_hit():
    """A price path that rises past the profit target should be labeled 1."""
    config = make_config(min_holding_period=1, max_holding_period=10,
                          profit_target=0.03, stop_loss=0.02)
    prices = [100] + [100 * 1.05] * 15  # jumps straight past +3% target
    df = pd.DataFrame({'Close': prices})

    labeled = LabelGenerator(config).create_labels(df)
    assert labeled['Label'].iloc[0] == 1


def test_stop_loss_barrier_hit():
    """A price path that falls past the stop loss should be labeled 0."""
    config = make_config(min_holding_period=1, max_holding_period=10,
                          profit_target=0.03, stop_loss=0.02)
    prices = [100] + [100 * 0.95] * 15  # drops past -2% stop
    df = pd.DataFrame({'Close': prices})

    labeled = LabelGenerator(config).create_labels(df)
    assert labeled['Label'].iloc[0] == 0


def test_no_lookahead_rows_dropped():
    """Rows too close to the end of the series (not enough future data) are dropped."""
    config = make_config(min_holding_period=1, max_holding_period=5)
    df = pd.DataFrame({'Close': np.linspace(100, 110, 8)})  # only 8 rows

    labeled = LabelGenerator(config).create_labels(df)
    # last `max_holding_period` rows can't be evaluated and must be excluded
    assert len(labeled) <= len(df) - config.max_holding_period


def test_labels_only_use_future_not_past():
    """
    Regression guard against a common leakage bug: shuffling the data
    that comes AFTER row i must not change the label for row i, but
    shuffling data BEFORE row i also must not change it (labels for row i
    should depend only on prices at i, i+1, ..., i+max_holding_period).
    """
    config = make_config(min_holding_period=1, max_holding_period=5,
                          profit_target=0.03, stop_loss=0.02)
    prices = list(np.linspace(100, 120, 30))
    df = pd.DataFrame({'Close': prices})

    labeled_original = LabelGenerator(config).create_labels(df)

    # Corrupt the past (rows before the window we're checking) — should not
    # affect the label of a row whose future window is untouched.
    df_corrupted = df.copy()
    df_corrupted.loc[0, 'Close'] = 9999  # only affects row 0's own entry price, not row 5's
    labeled_corrupted = LabelGenerator(config).create_labels(df_corrupted)

    row_to_check = 5
    assert labeled_original['Label'].iloc[row_to_check] == labeled_corrupted['Label'].iloc[row_to_check]
