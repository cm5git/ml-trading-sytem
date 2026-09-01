"""
Walk-forward validation.

Standard k-fold cross-validation shuffles data randomly, which for time
series means training on the future and testing on the past — a
serious form of leakage. `TimeSeriesSplit` instead produces expanding
train windows that always precede the test window in time, mirroring
how the model would actually be used once deployed: only ever trained
on the past to predict the immediate future.

This module retrains a fresh, quick LightGBM model per fold (not the
full tuned ensemble, for speed) purely to check how stable performance
is across different time periods — a large drop or high variance
across folds is a sign of an unstable or overfit model, not just a
final "does it work" check.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb


class WalkForwardAnalysis:
    """Test model stability across sequential time windows."""

    def __init__(self, config):
        self.config = config

    def run_analysis(self, combined_df: pd.DataFrame, feature_cols: List[str], n_splits: int = 5) -> pd.DataFrame:
        print("Running walk-forward analysis...")
        combined_df = combined_df.sort_index()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(combined_df), 1):
            print(f"Fold {fold}/{n_splits}")

            train_data = combined_df.iloc[train_idx]
            test_data = combined_df.iloc[test_idx]

            X_train = train_data[feature_cols].values
            y_train = train_data['Label'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['Label'].values

            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_bal)
            X_test_scaled = scaler.transform(X_test)

            train_set = lgb.Dataset(X_train_scaled, label=y_train_bal)
            params = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                'verbose': -1, 'device': 'gpu',
            }
            model = lgb.train(
                params, train_set, num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=0)],
            )

            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
            f1 = f1_score(y_test, y_pred)

            test_returns = [
                test_data['ForwardReturn'].iloc[i]
                for i in range(len(y_pred))
                if y_pred[i] == 1 and y_pred_proba[i] > 0.65
                and i < len(test_data) and not pd.isna(test_data['ForwardReturn'].iloc[i])
            ]
            avg_return = np.mean(test_returns) if test_returns else 0

            results.append({
                'fold': fold, 'train_samples': len(X_train), 'test_samples': len(X_test),
                'accuracy': accuracy, 'auc': auc, 'f1': f1,
                'avg_return': avg_return, 'n_trades': len(test_returns),
            })
            print(f"  accuracy={accuracy:.3f} auc={auc:.3f} f1={f1:.3f} "
                  f"avg_return={avg_return:.2%} trades={len(test_returns)}")

        results_df = pd.DataFrame(results)
        print("Walk-Forward Summary")
        print(f"Average AUC:      {results_df['auc'].mean():.3f} +/- {results_df['auc'].std():.3f}")
        print(f"Average Accuracy: {results_df['accuracy'].mean():.3f} +/- {results_df['accuracy'].std():.3f}")
        print(f"Average F1:       {results_df['f1'].mean():.3f} +/- {results_df['f1'].std():.3f}")
        print(f"Average Return:   {results_df['avg_return'].mean():.2%} +/- {results_df['avg_return'].std():.2%}")
        print(f"Total Trades:     {results_df['n_trades'].sum()}")

        return results_df
