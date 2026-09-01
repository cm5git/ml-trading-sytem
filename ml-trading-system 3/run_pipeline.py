"""
End-to-end pipeline: data -> features -> labels -> ensemble training ->
walk-forward validation -> backtest -> live signals.

This mirrors the original Colab notebook the project was developed in,
reorganized into the `src/` package. Run with a GPU available for
LightGBM/XGBoost/CatBoost GPU training and mixed-precision PyTorch
training; see README for CPU-only adjustments.

    python run_pipeline.py
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.utils import get_device, clear_memory
from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.label_generator import LabelGenerator
from src.models import TradingNet, TradingDataset, Trainer, EnsemblePredictor
from src.backtester import RealisticBacktester
from src.signal_generator import SignalGenerator
from src.walk_forward import WalkForwardAnalysis


def main():
    config = get_default_config()
    device = get_device()

    # 1. Data collection
    collector = DataCollector(config)
    raw_data = collector.fetch_data()
    if not raw_data:
        raise RuntimeError("No data loaded — check network access / ticker list.")

    # 2. Feature engineering
    engineer = FeatureEngineer(config)
    processed_data = {ticker: engineer.create_features(df) for ticker, df in raw_data.items()}

    # 3. Labeling (Triple Barrier)
    label_gen = LabelGenerator(config)
    labeled_data = {ticker: label_gen.create_labels(df) for ticker, df in processed_data.items()}

    combined_df = pd.concat(labeled_data.values(), axis=0).sort_index().dropna()
    feature_cols = engineer.get_feature_names(combined_df)
    print(f"Total samples: {len(combined_df):,} | Features: {len(feature_cols)}")

    # 4. Temporal train/val/test split (chronological, never shuffled)
    n = len(combined_df)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    train_df = combined_df.iloc[:train_end]
    val_df = combined_df.iloc[train_end:val_end]
    test_df = combined_df.iloc[val_end:]

    X_train, y_train = train_df[feature_cols].values, train_df['Label'].values
    X_val, y_val = val_df[feature_cols].values, val_df['Label'].values
    X_test, y_test = test_df[feature_cols].values, test_df['Label'].values

    # 5. Balance + scale
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    clear_memory()

    # 6. Neural net
    train_loader = DataLoader(TradingDataset(X_train_scaled, y_train_bal),
                               batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TradingDataset(X_val_scaled, y_val),
                             batch_size=config.batch_size * 2, shuffle=False)

    model = TradingNet(input_dim=len(feature_cols), hidden_dims=[512, 256, 128], dropout=0.3)
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
    clear_memory()

    # 7. Ensemble (NN + LightGBM + XGBoost + CatBoost)
    ensemble = EnsemblePredictor(config)
    ensemble.models['nn'] = trainer
    ensemble.train_lgb(X_train_scaled, y_train_bal, X_val_scaled, y_val)
    clear_memory()
    ensemble.train_xgb(X_train_scaled, y_train_bal, X_val_scaled, y_val)
    clear_memory()
    ensemble.train_catboost(X_train_scaled, y_train_bal, X_val_scaled, y_val)
    clear_memory()

    # 8. Test-set predictions + backtest
    test_probs = ensemble.predict(X_test_scaled, method='average')
    test_preds = (test_probs > 0.5).astype(int)

    backtester = RealisticBacktester(config)
    results = backtester.run_backtest(test_df, test_preds, test_probs)

    print("\nBACKTEST RESULTS")
    print(f"Total Return:      {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2%}")
    print(f"Win Rate:          {results['win_rate']:.2%}")
    print(f"Profit Factor:     {results['profit_factor']:.2f}")

    # 9. Walk-forward validation across the full dataset
    wfa = WalkForwardAnalysis(config)
    wfa.run_analysis(combined_df, feature_cols, n_splits=5)

    # 10. Live signals from the most recent data
    signal_gen = SignalGenerator(ensemble, scaler, feature_cols, config)
    current_signals = signal_gen.generate_signals(processed_data, top_n=10)
    print(signal_gen.format_signals_report(current_signals))


if __name__ == "__main__":
    main()
