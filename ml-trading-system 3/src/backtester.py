"""
Realistic backtesting engine.

Simulates actually trading the model's signals day by day, including:
  - Kelly-criterion-based position sizing (fractional Kelly, volatility-adjusted)
  - Transaction costs (commission + slippage, in basis points)
  - Fixed and (optional) trailing stop-losses, plus a max-holding-period exit
  - Portfolio-level exposure caps

This is what turns classification metrics (AUC, F1) into trading metrics
(Sharpe, Sortino, Calmar, drawdown, win rate, profit factor).
"""

from typing import Dict

import numpy as np
import pandas as pd

from .config import TradingConfig


class RealisticBacktester:
    """Simulates trading the model's signals with realistic execution costs."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_position_size(self, confidence: float, volatility: float, capital: float) -> float:
        """Fractional Kelly sizing, scaled down for higher volatility, capped at max_position_pct."""
        kelly_size = self.config.kelly_fraction * confidence
        vol_adjusted_size = kelly_size / (volatility + 0.01)
        position_size = min(vol_adjusted_size, self.config.max_position_pct)
        return capital * position_size

    def apply_transaction_costs(self, shares: float, price: float, is_buy: bool = True) -> float:
        """Commission + slippage in basis points."""
        gross_amount = shares * price
        commission = gross_amount * (self.config.commission_bps / 10000)
        slippage = gross_amount * (self.config.slippage_bps / 10000)
        return commission + abs(slippage)

    def run_backtest(self, test_df: pd.DataFrame, predictions: np.ndarray, confidences: np.ndarray) -> Dict:
        """
        Simulate trading day by day over the test period.

        Args:
            test_df: Test-period DataFrame (must include Ticker, Close, volatility_20d).
            predictions: Binary buy/no-buy predictions, aligned to test_df rows.
            confidences: Predicted probabilities, aligned to test_df rows.

        Returns:
            Dict of performance metrics (see _calculate_metrics).
        """
        print("Running realistic backtest...")

        capital = self.config.initial_capital
        cash = capital
        positions = {}  # ticker -> {shares, entry_price, entry_date, highest_price, confidence}

        portfolio_values = [capital]
        dates, trades, daily_returns = [], [], []

        unique_dates = sorted(test_df.index.unique())

        for date in unique_dates:
            day_data = test_df.loc[date]
            if isinstance(day_data, pd.Series):
                day_data = pd.DataFrame([day_data])

            portfolio_value = cash
            positions_to_close = []

            for ticker, pos in positions.items():
                ticker_data = day_data[day_data['Ticker'] == ticker]
                if len(ticker_data) == 0:
                    portfolio_value += pos['shares'] * pos['entry_price']
                    continue

                current_price = ticker_data['Close'].iloc[0]
                portfolio_value += pos['shares'] * current_price

                if current_price > pos.get('highest_price', pos['entry_price']):
                    pos['highest_price'] = current_price

                if self.config.use_stop_loss:
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    if pnl_pct <= -self.config.stop_loss:
                        positions_to_close.append((ticker, current_price, 'stop_loss'))
                        continue
                    if self.config.use_trailing_stop:
                        trailing_loss = (current_price - pos['highest_price']) / pos['highest_price']
                        if trailing_loss <= -self.config.trailing_stop_pct:
                            positions_to_close.append((ticker, current_price, 'trailing_stop'))
                            continue

                days_held = (date - pos['entry_date']).days
                if days_held >= self.config.max_holding_period:
                    positions_to_close.append((ticker, current_price, 'max_hold'))

            for ticker, exit_price, reason in positions_to_close:
                pos = positions[ticker]
                entry_value = pos['shares'] * pos['entry_price']
                exit_value = pos['shares'] * exit_price
                costs = self.apply_transaction_costs(pos['shares'], exit_price, is_buy=False)
                pnl = exit_value - entry_value - costs
                cash += exit_value - costs

                trades.append({
                    'date': date, 'ticker': ticker, 'action': 'sell',
                    'shares': pos['shares'], 'entry_price': pos['entry_price'],
                    'exit_price': exit_price, 'pnl': pnl, 'pnl_pct': pnl / entry_value,
                    'reason': reason, 'days_held': (date - pos['entry_date']).days,
                })
                del positions[ticker]

            current_exposure = sum(
                pos['shares'] * day_data[day_data['Ticker'] == t]['Close'].iloc[0]
                for t, pos in positions.items()
                if len(day_data[day_data['Ticker'] == t]) > 0
            ) / portfolio_value

            if current_exposure < self.config.max_portfolio_pct:
                for idx, row in day_data.iterrows():
                    ticker = row['Ticker']
                    if ticker in positions:
                        continue

                    sample_idx = test_df.index.get_loc(idx)
                    if isinstance(sample_idx, slice):
                        sample_idx = sample_idx.start

                    pred = predictions[sample_idx]
                    confidence = confidences[sample_idx]

                    if pred == 1 and confidence >= self.config.confidence_threshold:
                        volatility = row.get('volatility_20d', 0.02)
                        position_dollars = self.calculate_position_size(confidence, volatility, portfolio_value)

                        if position_dollars > cash * 0.95:
                            continue

                        entry_price = row['Close']
                        shares = int(position_dollars / entry_price)
                        if shares == 0:
                            continue

                        costs = self.apply_transaction_costs(shares, entry_price, is_buy=True)
                        total_cost = shares * entry_price + costs

                        if total_cost <= cash:
                            cash -= total_cost
                            positions[ticker] = {
                                'shares': shares, 'entry_price': entry_price,
                                'entry_date': date, 'stop_loss': entry_price * (1 - self.config.stop_loss),
                                'highest_price': entry_price, 'confidence': confidence,
                            }
                            trades.append({
                                'date': date, 'ticker': ticker, 'action': 'buy',
                                'shares': shares, 'price': entry_price, 'confidence': confidence,
                                'position_size': total_cost / portfolio_value,
                            })

            portfolio_values.append(portfolio_value)
            dates.append(date)
            if len(portfolio_values) > 1:
                daily_returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])

        if positions:
            print(f"Closing {len(positions)} remaining positions at end of test period...")
            for ticker, pos in positions.items():
                final_data = test_df[test_df['Ticker'] == ticker].iloc[-1]
                exit_price = final_data['Close']
                entry_value = pos['shares'] * pos['entry_price']
                exit_value = pos['shares'] * exit_price
                costs = self.apply_transaction_costs(pos['shares'], exit_price, is_buy=False)
                pnl = exit_value - entry_value - costs

                trades.append({
                    'date': test_df.index[-1], 'ticker': ticker, 'action': 'sell',
                    'shares': pos['shares'], 'entry_price': pos['entry_price'],
                    'exit_price': exit_price, 'pnl': pnl, 'pnl_pct': pnl / entry_value,
                    'reason': 'end_of_test',
                })

        return self._calculate_metrics(portfolio_values, dates, trades, daily_returns)

    def _calculate_metrics(self, portfolio_values, dates, trades, daily_returns) -> Dict:
        """Compute return, risk, and trade-level performance metrics."""
        portfolio_values = np.array(portfolio_values)
        total_return = (portfolio_values[-1] / self.config.initial_capital) - 1

        n_days = len(portfolio_values)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1

        daily_returns = np.array(daily_returns)
        if len(daily_returns) > 0:
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        else:
            volatility = sharpe_ratio = sortino_ratio = 0

        cumulative = portfolio_values / self.config.initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']

        profitable_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0

        wins = [t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in sell_trades if t.get('pnl', 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        avg_hold_period = (
            np.mean([t['days_held'] for t in sell_trades if 'days_held' in t]) if sell_trades else 0
        )

        return {
            'portfolio_values': portfolio_values,
            'dates': dates,
            'trades': trades,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(buy_trades),
            'avg_hold_period': avg_hold_period,
            'final_equity': portfolio_values[-1],
        }
