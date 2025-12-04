---
title: "Chapter 2: XGBoost Deep Dive"
chapter_title: "Chapter 2: XGBoost Deep Dive"
subtitle: Gradient Boosting with XGBoost
---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/ensemble-methods-introduction/chapter2-xgboost.html>) | Last sync: 2025-11-16

[ML Dojo](<../index.html>) > [Ensemble Methods](<index.html>) > Ch2

## 2.1 XGBoost Fundamentals

XGBoost (Extreme Gradient Boosting) is optimized implementation of gradient boosting with regularization.

**📐 XGBoost Objective:** $$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$ where $\Omega(f) = \gamma T + \frac{1}{2}\lambda\|\omega\|^2$ is regularization term

### 💻 Code Example 1: XGBoost Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - xgboost>=2.0.0
    
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    class XGBoostModel:
        """XGBoost wrapper for classification and regression"""
        
        def __init__(self, task='classification'):
            self.task = task
            self.model = None
        
        def train(self, X_train, y_train, params=None):
            """Train XGBoost model"""
            if params is None:
                params = self.get_default_params()
            
            if self.task == 'classification':
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
            return self
        
        def get_default_params(self):
            """Default XGBoost parameters"""
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
        
        def tune_hyperparameters(self, X_train, y_train):
            """Hyperparameter tuning with GridSearchCV"""
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            base_model = xgb.XGBClassifier() if self.task == 'classification' else xgb.XGBRegressor()
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            return grid_search.best_params_, grid_search.best_score_
    
    # Example usage
    from sklearn.datasets import load_breast_cancer, load_diabetes
    
    # Classification example
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_clf = XGBoostModel(task='classification')
    xgb_clf.train(X_train, y_train)
    
    y_pred = xgb_clf.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Classification Accuracy: {accuracy:.4f}")
    
    # Hyperparameter tuning
    best_params, best_score = xgb_clf.tune_hyperparameters(X_train, y_train)
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {best_score:.4f}")

## 2.2-2.7 Advanced Topics

Feature importance, early stopping, handling imbalanced data, custom objectives, distributed training.

### 💻 Code Examples 2-7
    
    
    # Feature importance analysis
    # Early stopping implementation
    # Handling class imbalance
    # Custom loss functions
    # Model interpretation with SHAP
    # Production deployment strategies
    # See complete code in full chapter

## 📝 Exercises

  1. Train XGBoost classifier and analyze feature importances.
  2. Implement early stopping with validation set monitoring.
  3. Handle imbalanced dataset using scale_pos_weight parameter.
  4. Compare XGBoost vs LightGBM vs CatBoost on benchmark dataset.
  5. Tune hyperparameters using Bayesian optimization (Optuna).

## Summary

  * XGBoost: optimized gradient boosting with regularization
  * Key parameters: max_depth, learning_rate, n_estimators, subsample
  * Built-in cross-validation and early stopping
  * Handles missing values automatically
  * Feature importance for interpretability
  * State-of-the-art performance on structured data

[← Ch1: Basics](<chapter1-ensemble-basics.html>) [Ch3: LightGBM/CatBoost →](<chapter3-lightgbm-catboost.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
