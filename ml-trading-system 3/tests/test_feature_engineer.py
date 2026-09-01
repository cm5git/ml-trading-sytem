"""
Sanity test that features computed at row i don't change when future
rows (after i) are modified — the core "no look-ahead" guarantee.
"""

import numpy as np
import pandas as pd

from src.config import TradingConfig
from src.feature_engineer import FeatureEngineer


def make_ohlcv(n=100, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume})


def test_features_unaffected_by_future_data():
    config = TradingConfig(tickers=['TEST'])
    engineer = FeatureEngineer(config)

    df = make_ohlcv(n=100)
    check_row = 50  # only compare a row with enough history behind it

    features_full = engineer.create_features(df)

    # Truncate everything AFTER check_row and recompute — the feature
    # values up to and including check_row must be identical, since a
    # backward-looking feature can't know the future was truncated.
    df_truncated = df.iloc[: check_row + 1].copy()
    features_truncated = engineer.create_features(df_truncated)

    feature_cols = engineer.get_feature_names(features_full)
    for col in feature_cols:
        full_val = features_full[col].iloc[check_row]
        trunc_val = features_truncated[col].iloc[check_row]
        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        assert np.isclose(full_val, trunc_val, equal_nan=True), (
            f"Feature '{col}' at row {check_row} changed when future rows were removed — "
            "possible look-ahead leakage."
        )
