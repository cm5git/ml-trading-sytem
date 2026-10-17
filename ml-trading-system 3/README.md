# ML Trading System

End-to-end machine learning pipeline for equity/ETF signal generation: data collection, leak-free feature engineering, Triple Barrier labeling, a 4-model ensemble (LightGBM, XGBoost, CatBoost, and a neural network), walk-forward validation, and a realistic backtester with transaction costs and Kelly-criterion position sizing.

Originally developed as part of an MSc final project studying validation methodology for financial ML (see `docs/` for the literature review that motivated the design choices).

## Why this exists

Most "ML trading bot" tutorials online have a fatal flaw: the validation is quietly leaking future information into the training process, so backtest results look great and mean nothing. This project is built around avoiding that, specifically:

- **Features** use only backward-looking windows (`rolling`, `shift`, `ewm`) — a feature at time *t* never uses data from *t+1* onward.
- **Labels** are generated with [Triple Barrier labeling](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) (López de Prado), which only uses *realized future prices* to construct the training target — never the current features.
- **Validation** uses walk-forward splits (`TimeSeriesSplit`), never random k-fold, so the model is always evaluated on data that comes after its training window — mirroring real deployment.
- These properties are covered by tests in `tests/` — see [Testing](#testing).

## Architecture

```
Data Collection (Yahoo Finance, 30 liquid instruments)
        |
Feature Engineering (60+ technical indicators, backward-looking only)
        |
Triple Barrier Labeling (profit / stop-loss / time exit)
        |
Chronological train / val / test split -> SMOTE (train only) -> RobustScaler
        |
Ensemble Training
   |-- LightGBM   (Optuna-tuned)
   |-- XGBoost    (fixed hyperparameters)
   |-- CatBoost   (fixed hyperparameters)
   +-- Neural net (residual blocks + one attention layer over features)
        |
Walk-Forward Validation (TimeSeriesSplit, 5 folds)
        |
Realistic Backtest (transaction costs, Kelly sizing, stop-losses)
        |
Live Signal Generation
```

## Methodology

### Triple Barrier Labeling

Predicting "will price go up tomorrow" ignores how a trade actually plays out. Triple Barrier labeling instead simulates entering a trade at each point in time and checks which exit comes first:

- **Profit barrier** — price rises to the target (+3% by default)
- **Stop barrier** — price falls to the stop-loss (-2% by default)
- **Time barrier** — neither is hit within the max holding period (20 days)

The label is 1 if the profit barrier is hit (or the time barrier is hit with a modest positive return), 0 otherwise. See `src/label_generator.py`.

### Walk-Forward Validation

Standard k-fold cross-validation shuffles data randomly — for time series, that means training on the future and testing on the past. `WalkForwardAnalysis` (`src/walk_forward.py`) instead uses `sklearn.model_selection.TimeSeriesSplit`, producing folds where the test window always comes after the training window. Performance is reported as mean ± std across folds — a large spread across folds is a sign of an unstable model, which matters more than any single fold's number.

### Ensemble

Four models vote by averaging predicted probabilities. Only **LightGBM** is currently tuned with Optuna (searching `num_leaves`, `learning_rate`, feature/bagging fractions, and L1/L2 regularization against validation AUC); XGBoost and CatBoost use fixed hyperparameters. Extending Optuna search to all three gradient boosting models is a natural next step (see [Possible Extensions](#possible-extensions)).

### Neural Network

`TradingNet` (`src/models/neural_network.py`) is a feedforward network with residual blocks for depth and a single self-attention layer applied across the *feature* dimension of one day's feature vector. It does not process a sequence of time steps and has no positional encoding — it is not a sequence Transformer in the GPT/BERT sense, despite the resemblance in components.

### Backtesting

`RealisticBacktester` (`src/backtester.py`) simulates trading the ensemble's signals day by day, including:
- Commission (5 bps) + slippage (10 bps) on every trade
- Fractional Kelly position sizing, scaled down by volatility, capped at 15% per position and 70% total exposure
- Fixed stop-loss, optional trailing stop, and a max-holding-period forced exit

Output metrics: total/annualized return, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor.

## Quick Start

```bash
git clone https://github.com/<cm5git>/ml-trading-system.git
cd ml-trading-system

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py
```

`run_pipeline.py` runs the full sequence end to end: data collection → features → labels → ensemble training → backtest → walk-forward validation → live signals. GPU-specific parameters (`device='gpu'`, `task_type='GPU'`) assume GPU-enabled builds of LightGBM/XGBoost/CatBoost (developed on a Colab A100); switch them to CPU equivalents (`device='cpu'`, remove `task_type`) if running without a GPU.

See `notebooks/01_quick_start.ipynb` for a lighter walkthrough of just the data → features → labels stages, with visualizations.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Tests currently cover the two components where leakage is most likely to silently creep in:
- `test_label_generator.py` — barrier logic correctness, and a regression test that a label depends only on its own future window, not on data before it.
- `test_feature_engineer.py` — that no feature's value changes when future rows are removed (the core "no look-ahead" guarantee).

## Project Structure

```
ml-trading-system/
├── src/
│   ├── config.py            # Central TradingConfig dataclass
│   ├── data_collector.py    # Yahoo Finance fetch + quality validation
│   ├── feature_engineer.py  # 60+ backward-looking technical features
│   ├── label_generator.py   # Triple Barrier labeling
│   ├── backtester.py        # Realistic backtest engine
│   ├── signal_generator.py  # Live signal generation
│   ├── walk_forward.py      # Walk-forward validation
│   ├── utils.py              # Device/memory helpers
│   └── models/
│       ├── neural_network.py  # Residual + attention feedforward net
│       ├── trainer.py         # PyTorch training loop
│       └── ensemble.py        # LightGBM + XGBoost + CatBoost + NN ensemble
├── notebooks/
│   └── 01_quick_start.ipynb
├── tests/
│   ├── test_label_generator.py
│   └── test_feature_engineer.py
├── run_pipeline.py           # End-to-end entry point
├── requirements.txt
└── docs/
    └── literature_review.pdf # Background research on adaptive/RL trading ML
```

## Possible Extensions

- **Monte Carlo / bootstrap resampling** of the backtested trade sequence, to check whether performance is robust or a product of one lucky historical path.
- **Purged cross-validation** (López de Prado) to remove overlapping-label leakage between adjacent folds.
- Optuna tuning extended to XGBoost and CatBoost, not just LightGBM.
- Self-supervised / meta-learning / reinforcement-learning approaches (MAML for cross-asset adaptation, PPO for portfolio management) as a longer-term research direction — see `docs/literature_review.pdf`.

## Disclaimer

This is a research and educational project. Past performance does not guarantee future results. Not financial advice. Trading involves substantial risk of loss.

## Acknowledgments

- Marcos López de Prado — *Advances in Financial Machine Learning* (Triple Barrier labeling, walk-forward and purged validation methodology)
- LightGBM, XGBoost, and CatBoost teams
- PyTorch team
- Yahoo Finance for market data access

## License

MIT — see [LICENSE](LICENSE).

## Contact

Chirag Modi — chiragmodi446@gmail.com — [LinkedIn](https://linkedin.com/in/chirag-modi-8b23722ab)
