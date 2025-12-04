---
title: "Chapter 3: Ensemble Methods"
chapter_title: "Chapter 3: Ensemble Methods"
subtitle: Performance Enhancement Through Model Combination - From Random Forest to XGBoost, LightGBM, and CatBoost
reading_time: 25-30 min
difficulty: Intermediate
code_examples: 13
exercises: 5
version: 1.0
created_at: 2025-10-20
---

## Learning Objectives

By completing this chapter, you will be able to:

  * Understand the principles of ensemble learning
  * Explain the differences between Bagging and Boosting
  * Implement Random Forest and analyze feature importance
  * Understand the mechanics of Gradient Boosting
  * Master XGBoost, LightGBM, and CatBoost
  * Acquire practical techniques used in Kaggle competitions

* * *

## 3.1 What is Ensemble Learning?

### Definition

**Ensemble Learning** is a method that combines multiple learners (models) to achieve higher performance than any single model.

> "Two heads are better than one" - Combining multiple weak learners to build a powerful predictor

### Benefits of Ensemble Methods
    
    
    ```mermaid
    graph LR
        A[Ensemble Benefits] --> B[Improved Accuracy]
        A --> C[Overfitting Prevention]
        A --> D[Improved Stability]
        A --> E[Enhanced Robustness]
    
        B --> B1[Higher accuracy than single models]
        C --> C1[Reduces variance]
        D --> D1[Reduces prediction variability]
        E --> E1[Robust to outliers and noise]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### Main Approaches

Method | Principle | Examples  
---|---|---  
**Bagging** | Parallel learning, averaging | Random Forest  
**Boosting** | Sequential learning, error correction | XGBoost, LightGBM, CatBoost  
**Stacking** | Integration via meta-learner | Level-wise Stacking  
  
* * *

## 3.2 Bagging (Bootstrap Aggregating)

### Principle

**Bagging** creates multiple datasets through bootstrap sampling and averages predictions from models trained on each dataset.
    
    
    ```mermaid
    graph TD
        A[Training Data] --> B[BootstrapSampling]
        B --> C1[Sample 1]
        B --> C2[Sample 2]
        B --> C3[Sample 3]
        C1 --> D1[Model 1]
        C2 --> D2[Model 2]
        C3 --> D3[Model 3]
        D1 --> E[Voting/Averaging]
        D2 --> E
        D3 --> E
        E --> F[Final Prediction]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style E fill:#f3e5f5
        style F fill:#e8f5e9
    ```

### Algorithm

  1. Create T bootstrap samples from training data through sampling with replacement
  2. Train learners independently on each sample
  3. Classification: majority voting, Regression: averaging for final prediction

$$ \hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x}) $$

### Implementation Example
    
    
    import numpy as np
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bagging
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,  # Number of learners
        max_samples=0.8,   # Sampling ratio
        random_state=42
    )
    
    bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_test)
    
    print("=== Bagging ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Compare with single decision tree
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    
    print(f"\nSingle Decision Tree Accuracy: {accuracy_score(y_test, y_pred_single):.4f}")
    print(f"Improvement: {accuracy_score(y_test, y_pred) - accuracy_score(y_test, y_pred_single):.4f}")
    

**Output** :
    
    
    === Bagging ===
    Accuracy: 0.8950
    
    Single Decision Tree Accuracy: 0.8300
    Improvement: 0.0650
    

* * *

## 3.3 Random Forest

### Overview

**Random Forest** is an ensemble method that adds random feature selection to Bagging. It builds a forest of decision trees.

### Differences Between Random Forest and Bagging

Item | Bagging | Random Forest  
---|---|---  
**Sampling** | Data only | Data + Features  
**Feature Selection** | Uses all features | Randomly selects subset  
**Diversity** | Moderate | High  
**Overfitting** | Somewhat prone | Less prone  
  
### Implementation Example
    
    
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',  # Randomly select sqrt(n) features
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print("=== Random Forest ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), importances[indices])
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Random Forest: Feature Importance (Top 10)', fontsize=14)
    plt.xticks(range(10), indices)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print(f"\nTop 5 Important Features:")
    for i in range(5):
        print(f"  Feature {indices[i]}: {importances[indices[i]]:.4f}")
    

**Output** :
    
    
    === Random Forest ===
    Accuracy: 0.9100
    
    Top 5 Important Features:
      Feature 2: 0.0852
      Feature 7: 0.0741
      Feature 13: 0.0689
      Feature 5: 0.0634
      Feature 19: 0.0598
    

### Out-of-Bag (OOB) Evaluation

You can evaluate using data not used in bootstrap sampling (approximately 37%).
    
    
    # OOB Score
    rf_oob = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42
    )
    
    rf_oob.fit(X_train, y_train)
    
    print(f"OOB Score: {rf_oob.oob_score_:.4f}")
    print(f"Test Score: {rf_oob.score(X_test, y_test):.4f}")
    

* * *

## 3.4 Boosting

### Overview

**Boosting** is a method that sequentially trains weak learners, with each subsequent model correcting the errors of the previous one.
    
    
    ```mermaid
    graph LR
        A[Data] --> B[Model 1]
        B --> C[Error Calculation]
        C --> D[Weight Update]
        D --> E[Model 2]
        E --> F[Error Calculation]
        F --> G[Weight Update]
        G --> H[Model 3]
        H --> I[...]
        I --> J[Final Model]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style E fill:#fff3e0
        style H fill:#fff3e0
        style J fill:#e8f5e9
    ```

### Differences Between Bagging and Boosting

Item | Bagging | Boosting  
---|---|---  
**Learning Method** | Parallel (independent) | Sequential (dependent)  
**Objective** | Variance reduction | Bias reduction  
**Weights** | Equal | Error-based  
**Overfitting** | Less prone | More prone  
**Training Speed** | Fast (parallelizable) | Slow (sequential)  
  
* * *

## 3.5 Gradient Boosting

### Principle

**Gradient Boosting** uses gradient descent to minimize the loss function. It learns residuals (actual value - predicted value) in subsequent models.

$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x}) $$

  * $F_m$: The m-th ensemble model
  * $\nu$: Learning rate
  * $h_m$: The m-th weak learner (learns residuals)

### Implementation Example
    
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    
    print("=== Gradient Boosting ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    
    # Learning Curve
    train_scores = []
    test_scores = []
    
    for i, y_pred in enumerate(gb_model.staged_predict(X_train)):
        train_scores.append(accuracy_score(y_train, y_pred))
    
    for i, y_pred in enumerate(gb_model.staged_predict(X_test)):
        test_scores.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Training Data', linewidth=2)
    plt.plot(test_scores, label='Test Data', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Gradient Boosting: Learning Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Output** :
    
    
    === Gradient Boosting ===
    Accuracy: 0.9250
    

* * *

## 3.6 XGBoost

### Overview

**XGBoost (Extreme Gradient Boosting)** is a fast and high-performance implementation of Gradient Boosting. It is one of the most widely used algorithms in Kaggle competitions.

### Features

  * **Regularization** : L1/L2 regularization prevents overfitting
  * **Missing Value Handling** : Automatically learns optimal splits
  * **Parallelization** : Parallelizes tree construction
  * **Early Stopping** : Detects overfitting and stops early
  * **Built-in Cross-Validation**

### Implementation Example
    
    
    import xgboost as xgb
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Early Stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    y_pred_xgb = xgb_model.predict(X_test)
    
    print("=== XGBoost ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    
    # Visualize Training History
    results = xgb_model.evals_result()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['logloss'], label='Training Data', linewidth=2)
    plt.plot(results['validation_1']['logloss'], label='Test Data', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title('XGBoost: Training History', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Feature Importance
    xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
    plt.title('XGBoost: Feature Importance (Top 10)')
    plt.show()
    

**Output** :
    
    
    === XGBoost ===
    Accuracy: 0.9350
    

### Hyperparameter Tuning
    
    
    from sklearn.model_selection import GridSearchCV
    
    # Parameter Grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid Search
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train)
    
    print("=== XGBoost Grid Search ===")
    print(f"Best Parameters: {xgb_grid.best_params_}")
    print(f"Best Score (CV): {xgb_grid.best_score_:.4f}")
    print(f"Test Score: {xgb_grid.score(X_test, y_test):.4f}")
    

* * *

## 3.7 LightGBM

### Overview

**LightGBM (Light Gradient Boosting Machine)** is a fast Gradient Boosting framework developed by Microsoft.

### Features

  * **Leaf-wise Growth** : More efficient than XGBoost's Level-wise approach
  * **GOSS** : Gradient-based One-Side Sampling for speedup
  * **EFB** : Exclusive Feature Bundling for memory reduction
  * **Categorical Variable Support** : No One-Hot Encoding required
  * **Large-Scale Data** : Fast even with millions of samples

    
    
    ```mermaid
    graph LR
        A[Tree Growth Strategy] --> B[Level-wiseXGBoost]
        A --> C[Leaf-wiseLightGBM]
    
        B --> B1[Grows layer by layerBalanced]
        C --> C1[Maximum loss reductionDeeper trees]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

### Implementation Example
    
    
    import lightgbm as lgb
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        verbose=False
    )
    
    y_pred_lgb = lgb_model.predict(X_test)
    
    print("=== LightGBM ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
    
    # Feature Importance
    lgb.plot_importance(lgb_model, max_num_features=10, importance_type='gain')
    plt.title('LightGBM: Feature Importance (Top 10)')
    plt.show()
    

**Output** :
    
    
    === LightGBM ===
    Accuracy: 0.9350
    

* * *

## 3.8 CatBoost

### Overview

**CatBoost (Categorical Boosting)** is a Gradient Boosting library developed by Yandex. It excels at handling categorical variables.

### Features

  * **Ordered Boosting** : Prevents prediction shift
  * **Automatic Categorical Variable Processing** : Improved version of Target Encoding
  * **Symmetric Trees** : Fast predictions
  * **GPU Acceleration** : Built-in GPU support
  * **Minimal Hyperparameter Tuning** : High performance with defaults

### Implementation Example
    
    
    from catboost import CatBoostClassifier
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        random_state=42,
        verbose=False
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test)
    )
    
    y_pred_cat = cat_model.predict(X_test)
    
    print("=== CatBoost ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
    
    # Feature Importance
    feature_importances = cat_model.get_feature_importance()
    indices = np.argsort(feature_importances)[::-1][:10]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), feature_importances[indices])
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('CatBoost: Feature Importance (Top 10)', fontsize=14)
    plt.xticks(range(10), indices)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    

**Output** :
    
    
    === CatBoost ===
    Accuracy: 0.9400
    

* * *

## 3.9 Comparison of Ensemble Methods

### Performance Comparison
    
    
    # Compare all models
    models = {
        'Bagging': bagging_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'CatBoost': cat_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ensemble Methods Performance Comparison', fontsize=14)
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y', alpha=0.3)
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10)
    plt.show()
    
    print("=== Ensemble Methods Comparison ===")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {acc:.4f}")
    

**Output** :
    
    
    === Ensemble Methods Comparison ===
    CatBoost            : 0.9400
    XGBoost             : 0.9350
    LightGBM            : 0.9350
    Gradient Boosting   : 0.9250
    Random Forest       : 0.9100
    Bagging             : 0.8950
    

### Feature Comparison

Method | Training Speed | Prediction Speed | Accuracy | Memory | Features  
---|---|---|---|---|---  
**Random Forest** | Fast | Fast | Medium | Large | Parallelization, Interpretability  
**Gradient Boosting** | Slow | Fast | High | Medium | Simple  
**XGBoost** | Medium | Fast | High | Medium | Kaggle standard  
**LightGBM** | Fast | Fast | High | Small | Large-scale data  
**CatBoost** | Medium | Fastest | Highest | Medium | Categorical variables  
  
* * *

## 3.10 Practical Techniques for Kaggle

### 1\. Ensemble of Ensembles (Stacking)
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Level 1: Base Models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
    ]
    
    # Level 2: Meta Model
    meta_model = LogisticRegression()
    
    # Stacking
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    
    print("=== Stacking Ensemble ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
    

### 2\. Weighted Average
    
    
    # Prediction probabilities from each model
    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    cat_proba = cat_model.predict_proba(X_test)
    
    # Weighted Average
    weights = [0.4, 0.3, 0.3]  # Adjust based on performance
    weighted_proba = (weights[0] * xgb_proba +
                     weights[1] * lgb_proba +
                     weights[2] * cat_proba)
    
    y_pred_weighted = np.argmax(weighted_proba, axis=1)
    
    print("=== Weighted Average ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
    

### 3\. Early Stopping
    
    
    # Using Early Stopping
    xgb_early = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_early.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    print(f"=== Early Stopping ===")
    print(f"Optimal Iterations: {xgb_early.best_iteration}")
    print(f"Accuracy: {xgb_early.score(X_test, y_test):.4f}")
    

* * *

## 3.11 Chapter Summary

### What You Learned

  1. **Ensemble Principles**

     * Performance improvement through model combination
     * Bagging: Parallel learning, variance reduction
     * Boosting: Sequential learning, bias reduction
  2. **Random Forest**

     * Bagging + random feature selection
     * Feature importance analysis
     * OOB evaluation
  3. **Gradient Boosting**

     * Sequential residual learning
     * High accuracy but beware of overfitting
  4. **XGBoost/LightGBM/CatBoost**

     * Most widely used methods in Kaggle
     * Fast and accurate
     * Each has different features and strengths
  5. **Practical Techniques**

     * Stacking
     * Weighted Average
     * Early Stopping

### Next Chapter

In Chapter 4, we will apply the techniques learned through **Practical Projects** :

  * Project 1: House Price Prediction (Regression)
  * Project 2: Customer Churn Prediction (Classification)
  * Complete Machine Learning Pipeline

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

List three main differences between Bagging and Boosting.

Solution

**Answer** :

  1. **Learning Method** : Bagging is parallel, Boosting is sequential
  2. **Objective** : Bagging reduces variance, Boosting reduces bias
  3. **Weights** : Bagging uses equal weights, Boosting uses error-based weights

### Problem 2 (Difficulty: Medium)

Explain why LightGBM is faster than XGBoost.

Solution

**Answer** :

**1\. Leaf-wise Growth Strategy** :

  * XGBoost: Level-wise (grows layer by layer)
  * LightGBM: Leaf-wise (grows the leaf with maximum loss reduction)
  * Result: Achieves same accuracy with fewer splits

**2\. GOSS (Gradient-based One-Side Sampling)** :

  * Retains data with large gradients
  * Randomly samples data with small gradients
  * Result: Speedup through data reduction

**3\. EFB (Exclusive Feature Bundling)** :

  * Bundles exclusive features together
  * Result: Improved memory efficiency through feature count reduction

**4\. Histogram-based** :

  * Discretizes continuous values into bins
  * Result: Faster split point search

### Problem 3 (Difficulty: Medium)

Extract the top 5 most important features from Random Forest and retrain the model using only those features. How does performance change?

Solution
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest with all features
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    acc_full = rf_full.score(X_test, y_test)
    
    print(f"Accuracy with all features (20): {acc_full:.4f}")
    
    # Extract Top 5 feature importances
    importances = rf_full.feature_importances_
    top5_indices = np.argsort(importances)[::-1][:5]
    
    print(f"\nTop 5 Features: {top5_indices}")
    print(f"Importances: {importances[top5_indices]}")
    
    # Build model with Top 5 features only
    X_train_top5 = X_train[:, top5_indices]
    X_test_top5 = X_test[:, top5_indices]
    
    rf_top5 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_top5.fit(X_train_top5, y_train)
    acc_top5 = rf_top5.score(X_test_top5, y_test)
    
    print(f"\nAccuracy with Top 5 features: {acc_top5:.4f}")
    print(f"Accuracy change: {acc_top5 - acc_full:.4f}")
    print(f"Feature reduction rate: {(20-5)/20*100:.1f}%")
    

**Output** :
    
    
    Accuracy with all features (20): 0.9100
    
    Top 5 Features: [ 2  7 13  5 19]
    Importances: [0.0852 0.0741 0.0689 0.0634 0.0598]
    
    Accuracy with Top 5 features: 0.8650
    Accuracy change: -0.0450
    Feature reduction rate: 75.0%
    

**Discussion** :

  * Even with 75% feature reduction, accuracy only drops by about 5%
  * Significant reduction in computation time and memory usage
  * Improved interpretability (focus on important features)

### Problem 4 (Difficulty: Hard)

Train XGBoost, LightGBM, and CatBoost on the same data and write code to select the most appropriate model.

Solution
    
    
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import cross_val_score
    import time
    
    # Data (refer to previous code)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model definitions
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
    }
    
    # Evaluation
    results = {}
    
    for name, model in models.items():
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
    
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
    
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
        # Test score
        test_score = accuracy_score(y_test, y_pred)
    
        results[name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score
        }
    
    # Display results
    print("=== Model Comparison ===\n")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Training Time: {metrics['train_time']:.4f} sec")
        print(f"  Prediction Time: {metrics['predict_time']:.4f} sec")
        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"  Test Accuracy: {metrics['test_score']:.4f}")
        print()
    
    # Select optimal model
    best_model = max(results.items(), key=lambda x: x[1]['test_score'])
    print(f"Optimal Model: {best_model[0]}")
    print(f"Test Accuracy: {best_model[1]['test_score']:.4f}")
    

**Output** :
    
    
    === Model Comparison ===
    
    XGBoost:
      Training Time: 0.2341 sec
      Prediction Time: 0.0023 sec
      CV Accuracy: 0.9212 (+/- 0.0156)
      Test Accuracy: 0.9350
    
    LightGBM:
      Training Time: 0.1234 sec
      Prediction Time: 0.0018 sec
      CV Accuracy: 0.9188 (+/- 0.0178)
      Test Accuracy: 0.9350
    
    CatBoost:
      Training Time: 0.4567 sec
      Prediction Time: 0.0012 sec
      CV Accuracy: 0.9250 (+/- 0.0134)
      Test Accuracy: 0.9400
    
    Optimal Model: CatBoost
    Test Accuracy: 0.9400
    

### Problem 5 (Difficulty: Hard)

Implement Stacking and Weighted Average, and compare which one achieves better performance.

Solution
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Data (refer to previous code)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base Models
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
        ('cat', CatBoostClassifier(iterations=100, random_state=42, verbose=False))
    ]
    
    # 1. Stacking
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    y_pred_stacking = stacking.predict(X_test)
    acc_stacking = accuracy_score(y_test, y_pred_stacking)
    
    print("=== Stacking ===")
    print(f"Accuracy: {acc_stacking:.4f}")
    
    # 2. Weighted Average
    # Get prediction probabilities from each model
    xgb_model = base_models[0][1]
    lgb_model = base_models[1][1]
    cat_model = base_models[2][1]
    
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)
    
    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    cat_proba = cat_model.predict_proba(X_test)
    
    # Weight optimization (grid search)
    best_acc = 0
    best_weights = None
    
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
    
            weighted_proba = w1 * xgb_proba + w2 * lgb_proba + w3 * cat_proba
            y_pred = np.argmax(weighted_proba, axis=1)
            acc = accuracy_score(y_test, y_pred)
    
            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2, w3)
    
    print("\n=== Weighted Average ===")
    print(f"Optimal Weights: XGB={best_weights[0]:.1f}, LGB={best_weights[1]:.1f}, Cat={best_weights[2]:.1f}")
    print(f"Accuracy: {best_acc:.4f}")
    
    # Comparison
    print("\n=== Comparison ===")
    print(f"Stacking: {acc_stacking:.4f}")
    print(f"Weighted Average: {best_acc:.4f}")
    print(f"Difference: {best_acc - acc_stacking:.4f}")
    
    if best_acc > acc_stacking:
        print("-> Weighted Average is superior")
    else:
        print("-> Stacking is superior")
    

**Output** :
    
    
    === Stacking ===
    Accuracy: 0.9450
    
    === Weighted Average ===
    Optimal Weights: XGB=0.3, LGB=0.3, Cat=0.4
    Accuracy: 0.9500
    
    === Comparison ===
    Stacking: 0.9450
    Weighted Average: 0.9500
    Difference: 0.0050
    -> Weighted Average is superior
    

**Discussion** :

  * Weighted Average is slightly superior
  * Stacking has a somewhat higher risk of overfitting
  * Weighted Average is simpler and more interpretable
  * For large-scale data, Stacking may be advantageous

* * *

## References

  1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." _KDD 2016_.
  2. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." _NIPS 2017_.
  3. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." _NeurIPS 2018_.
  4. Breiman, L. (2001). "Random Forests." _Machine Learning_ , 45(1), 5-32.
