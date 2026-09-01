"""
Ensemble of four models: a neural net (trained separately via Trainer),
LightGBM, XGBoost, and CatBoost. Final prediction is the average of the
models' predicted probabilities.

Only LightGBM is tuned with Optuna in this version — its objective
function searches num_leaves, learning_rate, feature/bagging fractions,
and L1/L2 regularization, maximizing validation AUC across n_trials.
XGBoost and CatBoost currently use fixed, hand-set hyperparameters
rather than an Optuna search of their own; that's a natural next
improvement (see README "Possible Extensions").

GPU parameters (device='gpu'/'cuda', task_type='GPU') assume
GPU-enabled builds of LightGBM/XGBoost/CatBoost and a CUDA-capable
machine (this was originally developed and run on a Colab A100). Set
them to CPU equivalents if running without a GPU.
"""

import numpy as np
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


class EnsemblePredictor:
    """Trains and combines LightGBM, XGBoost, CatBoost, and (externally) a neural net."""

    def __init__(self, config):
        self.config = config
        self.models = {}

    def train_lgb(self, X_train, y_train, X_val, y_val):
        """Train LightGBM, tuning hyperparameters with Optuna."""
        print("Training LightGBM (Optuna search)...")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'verbose': -1,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params, train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            preds = model.predict(X_val)
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
        print(f"Best validation AUC: {study.best_value:.4f}")

        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        })

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            best_params, train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        self.models['lgb'] = model
        return model

    def train_xgb(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with fixed hyperparameters."""
        print("Training XGBoost...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.1,
            'min_child_weight': 5,
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        self.models['xgb'] = model
        return model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost with fixed hyperparameters."""
        print("Training CatBoost...")

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_strength=1,
            bagging_temperature=1,
            task_type='GPU',
            devices='0',
            verbose=100,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

        self.models['catboost'] = model
        return model

    def predict(self, X, method: str = 'average'):
        """
        Combine predictions from every trained model.

        Args:
            X: Feature matrix (already scaled).
            method: 'average', 'median', or 'max' across model probabilities.
        """
        predictions = []

        if 'nn' in self.models:
            predictions.append(self.models['nn'].predict(X))
        if 'lgb' in self.models:
            predictions.append(self.models['lgb'].predict(X))
        if 'xgb' in self.models:
            predictions.append(self.models['xgb'].predict(xgb.DMatrix(X)))
        if 'catboost' in self.models:
            predictions.append(self.models['catboost'].predict_proba(X)[:, 1])

        predictions = np.array(predictions)

        if method == 'median':
            return np.median(predictions, axis=0)
        if method == 'max':
            return predictions.max(axis=0)
        return predictions.mean(axis=0)  # default: average
