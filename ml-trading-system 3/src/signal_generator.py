"""
Generate actionable trading signals from the trained ensemble on
current (most recent) market data — separate from backtesting, this is
what you'd run to get "what should I trade today" output.
"""

import pandas as pd


class SignalGenerator:
    """Turn ensemble predictions on the latest data into ranked trade signals."""

    def __init__(self, ensemble, scaler, feature_cols, config):
        self.ensemble = ensemble
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.config = config

    def calculate_position_size(self, confidence: float, volatility: float) -> float:
        kelly_size = self.config.kelly_fraction * confidence
        vol_adjusted = kelly_size / (volatility + 0.01)
        return min(vol_adjusted, self.config.max_position_pct)

    def generate_signals(self, current_data_dict, top_n: int = 10):
        """
        Score the latest row of features for each ticker and return the
        top_n highest-confidence BUY signals.
        """
        signals = []

        for ticker, df in current_data_dict.items():
            latest = df[self.feature_cols].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest)

            prob = self.ensemble.predict(latest_scaled)[0]
            pred = int(prob > 0.5)

            current_price = df['Close'].iloc[-1]
            volatility = df.get('volatility_20d', pd.Series([0.02])).iloc[-1]
            volume = df['Volume'].iloc[-1]
            dollar_volume = current_price * volume

            if pred == 1 and prob >= self.config.confidence_threshold:
                position_size = self.calculate_position_size(prob, volatility)
                profit_target = current_price * (1 + self.config.profit_target)
                stop_loss = current_price * (1 - self.config.stop_loss)

                signals.append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'current_price': current_price,
                    'confidence': prob,
                    'position_size_pct': position_size * 100,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'volatility': volatility,
                    'dollar_volume': dollar_volume,
                    'expected_return': self.config.profit_target,
                    'risk_reward': self.config.profit_target / self.config.stop_loss,
                })

        return sorted(signals, key=lambda x: x['confidence'], reverse=True)[:top_n]

    def format_signals_report(self, signals) -> str:
        """Render signals as a readable text report."""
        if not signals:
            return "No trading signals generated at this time."

        lines = ["=" * 80, "TRADING SIGNALS - Top Opportunities", "=" * 80, ""]
        for i, sig in enumerate(signals, 1):
            lines += [
                f"#{i} {sig['ticker']}",
                f"   Action:           {sig['action']}",
                f"   Current Price:    ${sig['current_price']:.2f}",
                f"   Confidence:       {sig['confidence']:.1%}",
                f"   Position Size:    {sig['position_size_pct']:.1f}% of portfolio",
                f"   Profit Target:    ${sig['profit_target']:.2f} (+{sig['expected_return']:.1%})",
                f"   Stop Loss:        ${sig['stop_loss']:.2f} (-{self.config.stop_loss:.1%})",
                f"   Risk/Reward:      {sig['risk_reward']:.2f}x",
                f"   Volatility:       {sig['volatility']:.1%}",
                f"   Daily Volume:     ${sig['dollar_volume']:,.0f}",
                "-" * 80,
            ]
        return "\n".join(lines)
