---
title: "Chapter 1: Hyperparameter Tuning Basics"
chapter_title: "Chapter 1: Hyperparameter Tuning Basics"
subtitle: Fundamentals of Search Methods for Maximizing Model Performance
reading_time: 25-30 minutes
difficulty: Beginner
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter introduces the basics of Hyperparameter Tuning Basics. You will learn difference between hyperparameters and importance of tuning.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the difference between hyperparameters and model parameters
  * ✅ Learn the importance of tuning and designing search spaces
  * ✅ Master the mechanism and implementation of grid search
  * ✅ Understand the advantages and usage of random search
  * ✅ Combine cross-validation with hyperparameter search
  * ✅ Execute practical tuning with scikit-learn

* * *

## 1.1 What are Hyperparameters

### Difference from Model Parameters

**Hyperparameters** are values set by humans before training that control the structure and learning process of a model.

Type | Definition | Examples | How Determined  
---|---|---|---  
**Model Parameters** | Automatically optimized through learning | Linear regression coefficients, neural network weights | Learned from training data  
**Hyperparameters** | Set by humans before learning | Learning rate, tree depth, regularization coefficient | Trial and error, search algorithms  
  
### Key Hyperparameters

Algorithm | Key Hyperparameters | Role  
---|---|---  
**Random Forest** | n_estimators, max_depth, min_samples_split | Number of trees, depth, split conditions  
**XGBoost** | learning_rate, max_depth, n_estimators, subsample | Learning speed, complexity, sampling  
**SVM** | C, kernel, gamma | Regularization, kernel, influence range  
**Neural Network** | learning_rate, batch_size, hidden_layers | Learning speed, batch size, structure  
  
### Importance of Tuning

> With proper hyperparameter settings, model performance can improve by 10-30% or more.
    
    
    ```mermaid
    graph LR
        A[Default Settings] --> B[Accuracy: 75%]
        C[After Tuning] --> D[Accuracy: 88%]
    
        style A fill:#ffebee
        style B fill:#ffcdd2
        style C fill:#e8f5e9
        style D fill:#a5d6a7
    ```

### Designing the Search Space

The search space is the range of candidate values for each hyperparameter. Proper design is crucial.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: The search space is the range of candidate values for each h
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Example search space definition
    param_space = {
        'n_estimators': [50, 100, 200, 300],           # Number of trees
        'max_depth': [5, 10, 15, 20, None],            # Maximum depth
        'min_samples_split': [2, 5, 10],               # Minimum samples to split
        'min_samples_leaf': [1, 2, 4],                 # Minimum samples per leaf
        'max_features': ['sqrt', 'log2', None]         # Number of features for split
    }
    
    print("=== Search Space Overview ===")
    print(f"n_estimators: {len(param_space['n_estimators'])} options")
    print(f"max_depth: {len(param_space['max_depth'])} options")
    print(f"min_samples_split: {len(param_space['min_samples_split'])} options")
    print(f"min_samples_leaf: {len(param_space['min_samples_leaf'])} options")
    print(f"max_features: {len(param_space['max_features'])} options")
    
    total_combinations = np.prod([len(v) for v in param_space.values()])
    print(f"\nTotal combinations: {total_combinations:,}")
    

**Output** :
    
    
    === Search Space Overview ===
    n_estimators: 4 options
    max_depth: 5 options
    min_samples_split: 3 options
    min_samples_leaf: 3 options
    max_features: 3 options
    
    Total combinations: 540
    

> **Important** : If the search space is too wide, computational costs become enormous. Utilize domain knowledge and empirical ranges.

* * *

## 1.2 Grid Search

### Mechanism and Implementation

**Grid Search** exhaustively explores all combinations of specified hyperparameters.
    
    
    ```mermaid
    graph TD
        A[Define Search Space] --> B[Generate All Combinations]
        B --> C[Train Each Combination]
        C --> D[Evaluate with Cross-Validation]
        D --> E[Select Best Parameters]
    
        style A fill:#e3f2fd
        style B fill:#bbdefb
        style C fill:#90caf9
        style D fill:#64b5f6
        style E fill:#42a5f5
    ```

### scikit-learn GridSearchCV
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    # Data preparation
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # GridSearchCV configuration
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,                      # 5-fold cross-validation
        scoring='accuracy',        # Evaluation metric
        n_jobs=-1,                 # Use all CPU cores
        verbose=2                  # Detailed output
    )
    
    # Execute grid search
    print("=== Starting Grid Search ===")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # Display results
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    print(f"\nBest parameters:")
    print(grid_search.best_params_)
    print(f"\nBest score (cross-validation): {grid_search.best_score_:.4f}")
    
    # Evaluate on test data
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test data accuracy: {test_accuracy:.4f}")
    

**Example Output** :
    
    
    === Starting Grid Search ===
    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    
    Execution time: 12.34 seconds
    
    Best parameters:
    {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}
    
    Best score (cross-validation): 0.9648
    Test data accuracy: 0.9737
    

### Detailed Analysis of Search Results
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Detailed Analysis of Search Results
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Extract important columns only
    results_summary = results_df[[
        'param_n_estimators',
        'param_max_depth',
        'param_min_samples_split',
        'mean_test_score',
        'std_test_score',
        'rank_test_score'
    ]].sort_values('rank_test_score')
    
    print("\n=== Top 5 Combinations ===")
    print(results_summary.head(10))
    
    # Visualization: Parameter influence analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Influence of n_estimators
    results_df.groupby('param_n_estimators')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[0], color='steelblue'
    )
    axes[0].set_title('Influence of n_estimators', fontsize=12)
    axes[0].set_ylabel('Average Score')
    axes[0].grid(True, alpha=0.3)
    
    # Influence of max_depth
    results_df.groupby('param_max_depth')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[1], color='forestgreen'
    )
    axes[1].set_title('Influence of max_depth', fontsize=12)
    axes[1].set_ylabel('Average Score')
    axes[1].grid(True, alpha=0.3)
    
    # Influence of min_samples_split
    results_df.groupby('param_min_samples_split')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[2], color='coral'
    )
    axes[2].set_title('Influence of min_samples_split', fontsize=12)
    axes[2].set_ylabel('Average Score')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Pros and Cons

Aspect | Details  
---|---  
**Pros** | ✅ Exhaustive search ensures optimal solution isn't missed  
✅ Simple implementation, easy to understand  
✅ Easy to parallelize  
**Cons** | ❌ Computational cost increases exponentially  
❌ Not suitable for high-dimensional search  
❌ Limited for continuous-valued parameters  
**Use Cases** | Few parameters (around 2-4)  
Few candidates per parameter  
Sufficient computational resources  
  
* * *

## 1.3 Random Search

### Benefits of Probabilistic Search

**Random Search** randomly samples parameter combinations from the search space.

> Research by Bergstra & Bengio (2012) has shown that random search is more efficient than grid search.
    
    
    ```mermaid
    graph LR
        A[Grid Search] --> B[Search EverythingHigh Computational Cost]
        C[Random Search] --> D[Random SamplingLow Computational Cost]
    
        style A fill:#ffcdd2
        style B fill:#ef9a9a
        style C fill:#c8e6c9
        style D fill:#81c784
    ```

### RandomizedSearchCV
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: RandomizedSearchCV
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    import numpy as np
    
    # Distribution definition for random search
    param_distributions = {
        'n_estimators': randint(50, 500),              # Integer 50-500
        'max_depth': randint(5, 30),                   # Integer 5-30
        'min_samples_split': randint(2, 20),           # Integer 2-20
        'min_samples_leaf': randint(1, 10),            # Integer 1-10
        'max_features': uniform(0.1, 0.9)              # Real 0.1-1.0
    }
    
    # RandomizedSearchCV configuration
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=100,                # 100 random samplings
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    # Execute random search
    print("=== Starting Random Search ===")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    print(f"\nBest parameters:")
    print(random_search.best_params_)
    print(f"\nBest score (cross-validation): {random_search.best_score_:.4f}")
    
    # Evaluate on test data
    y_pred_random = random_search.predict(X_test)
    test_accuracy_random = accuracy_score(y_test, y_pred_random)
    print(f"Test data accuracy: {test_accuracy_random:.4f}")
    

**Example Output** :
    
    
    === Starting Random Search ===
    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    
    Execution time: 18.56 seconds
    
    Best parameters:
    {'max_depth': 18, 'max_features': 0.7234, 'min_samples_leaf': 1,
     'min_samples_split': 2, 'n_estimators': 387}
    
    Best score (cross-validation): 0.9692
    Test data accuracy: 0.9825
    

### Comparison with Grid Search
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Comparison with Grid Search
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    
    # Visualization of comparison results
    comparison_data = {
        'Grid Search': {
            'Search Count': len(grid_search.cv_results_['params']),
            'Execution Time': 12.34,
            'CV Accuracy': grid_search.best_score_,
            'Test Accuracy': test_accuracy
        },
        'Random Search': {
            'Search Count': len(random_search.cv_results_['params']),
            'Execution Time': 18.56,
            'CV Accuracy': random_search.best_score_,
            'Test Accuracy': test_accuracy_random
        }
    }
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data).T
    print("\n=== Grid Search vs Random Search ===")
    print(comparison_df)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Search count
    comparison_df['Search Count'].plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Search Count Comparison', fontsize=12)
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Execution time
    comparison_df['Execution Time'].plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
    axes[1].set_title('Execution Time Comparison', fontsize=12)
    axes[1].set_ylabel('Seconds')
    axes[1].grid(True, alpha=0.3)
    
    # Accuracy
    comparison_df[['CV Accuracy', 'Test Accuracy']].plot(kind='bar', ax=axes[2])
    axes[2].set_title('Accuracy Comparison', fontsize=12)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim([0.95, 1.0])
    axes[2].legend(['CV Accuracy', 'Test Accuracy'])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Advantages of Random Search

Aspect | Grid Search | Random Search  
---|---|---  
**Computational Efficiency** | Search count = all combinations | Search count can be specified  
**Continuous Value Support** | Discrete values only | Direct sampling from continuous distributions  
**Handling Importance** | All explored equally | Can explore important parameter ranges widely  
**High-Dimensional Search** | Exponential growth with dimensions | Linear with respect to dimensions  
  
* * *

## 1.4 Cross-Validation and Hyperparameter Search

### Choosing CV Strategy

Cross-validation is essential for evaluating the generalization performance of hyperparameters.

CV Method | Description | Use Case  
---|---|---  
**K-Fold CV** | Split data into K parts, evaluate K times | Standard scenarios (K=5 or 10)  
**Stratified K-Fold** | Split while preserving class ratios | Classification problems, imbalanced data  
**Time Series Split** | Preserve temporal ordering | Time series data  
**Leave-One-Out** | Test one sample at a time | Small datasets (high computational cost)  
  
### Setting Evaluation Metrics
    
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
    
    # Compare with multiple evaluation metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    # Cross-validation with Stratified K-Fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Apply multiple evaluation metrics to RandomizedSearchCV
    random_search_multi = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv_strategy,
        scoring=scoring_metrics,
        refit='f1',                 # Select best model based on F1 score
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search_multi.fit(X_train, y_train)
    
    print("=== Results with Multiple Evaluation Metrics ===")
    print(f"Best parameters (F1 criterion):")
    print(random_search_multi.best_params_)
    
    # Score for each metric
    results = random_search_multi.cv_results_
    best_index = random_search_multi.best_index_
    
    print(f"\nBest model scores:")
    for metric in scoring_metrics.keys():
        score = results[f'mean_test_{metric}'][best_index]
        std = results[f'std_test_{metric}'][best_index]
        print(f"  {metric}: {score:.4f} (±{std:.4f})")
    

**Example Output** :
    
    
    === Results with Multiple Evaluation Metrics ===
    Best parameters (F1 criterion):
    {'max_depth': 22, 'max_features': 0.6543, 'min_samples_leaf': 1,
     'min_samples_split': 3, 'n_estimators': 298}
    
    Best model scores:
      accuracy: 0.9670 (±0.0123)
      precision: 0.9678 (±0.0118)
      recall: 0.9670 (±0.0123)
      f1: 0.9672 (±0.0121)
    

### Preventing Overfitting
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Preventing Overfitting
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    
    # Compare train and test scores
    results = random_search.cv_results_
    
    train_scores = results['mean_train_score']
    test_scores = results['mean_test_score']
    
    # Detect overfitting
    overfit_gap = train_scores - test_scores
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Score distribution
    axes[0].scatter(train_scores, test_scores, alpha=0.6, s=50)
    axes[0].plot([0.9, 1.0], [0.9, 1.0], 'r--', label='Ideal Line')
    axes[0].set_xlabel('Train Score')
    axes[0].set_ylabel('Test Score (CV)')
    axes[0].set_title('Train vs Test Score', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Overfitting gap
    axes[1].hist(overfit_gap, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=overfit_gap.mean(), color='r', linestyle='--',
                    label=f'Average Gap: {overfit_gap.mean():.4f}')
    axes[1].set_xlabel('Overfitting Gap (Train - Test)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Degree of Overfitting', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Top 5 models with least overfitting
    results_df = pd.DataFrame({
        'rank': results['rank_test_score'],
        'train_score': train_scores,
        'test_score': test_scores,
        'overfit_gap': overfit_gap
    })
    
    print("\n=== Top 5 Models with Least Overfitting ===")
    print(results_df.nsmallest(5, 'overfit_gap'))
    

* * *

## 1.5 Practice: Basic Tuning with scikit-learn

### Random Forest Tuning Example
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Performance with default settings
    print("=== Random Forest Tuning ===\n")
    rf_default = RandomForestClassifier(random_state=42)
    rf_default.fit(X_train, y_train)
    default_score = accuracy_score(y_test, rf_default.predict(X_test))
    print(f"Default settings accuracy: {default_score:.4f}")
    
    # Grid search
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    start = time.time()
    grid_rf.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # Performance after tuning
    tuned_score = accuracy_score(y_test, grid_rf.predict(X_test))
    
    print(f"\nBest parameters: {grid_rf.best_params_}")
    print(f"Accuracy after tuning: {tuned_score:.4f}")
    print(f"Improvement: {(tuned_score - default_score) * 100:.2f}%")
    print(f"Execution time: {elapsed:.2f} seconds")
    

**Example Output** :
    
    
    === Random Forest Tuning ===
    
    Default settings accuracy: 0.8700
    
    Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    Accuracy after tuning: 0.9250
    Improvement: 5.50%
    Execution time: 24.56 seconds
    

### XGBoost Tuning Example
    
    
    # Requirements:
    # - Python 3.9+
    # - xgboost>=2.0.0
    
    """
    Example: XGBoost Tuning Example
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    # XGBoost parameter distributions
    param_dist_xgb = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5)
    }
    
    # Default settings
    print("\n=== XGBoost Tuning ===\n")
    xgb_default = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_default.fit(X_train, y_train)
    default_score_xgb = accuracy_score(y_test, xgb_default.predict(X_test))
    print(f"Default settings accuracy: {default_score_xgb:.4f}")
    
    # Random search
    random_xgb = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        param_dist_xgb,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    start = time.time()
    random_xgb.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # Performance after tuning
    tuned_score_xgb = accuracy_score(y_test, random_xgb.predict(X_test))
    
    print(f"\nBest parameters:")
    for param, value in random_xgb.best_params_.items():
        print(f"  {param}: {value:.4f}" if isinstance(value, float) else f"  {param}: {value}")
    
    print(f"\nAccuracy after tuning: {tuned_score_xgb:.4f}")
    print(f"Improvement: {(tuned_score_xgb - default_score_xgb) * 100:.2f}%")
    print(f"Execution time: {elapsed:.2f} seconds")
    

**Example Output** :
    
    
    === XGBoost Tuning ===
    
    Default settings accuracy: 0.9000
    
    Best parameters:
      colsample_bytree: 0.8234
      gamma: 0.1234
      learning_rate: 0.0876
      max_depth: 7
      n_estimators: 387
      subsample: 0.8567
    
    Accuracy after tuning: 0.9400
    Improvement: 4.00%
    Execution time: 42.18 seconds
    

### Result Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Result Visualization
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Model comparison
    models_comparison = {
        'RF (Default)': default_score,
        'RF (Tuned)': tuned_score,
        'XGB (Default)': default_score_xgb,
        'XGB (Tuned)': tuned_score_xgb
    }
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    models = list(models_comparison.keys())
    scores = list(models_comparison.values())
    colors = ['lightcoral', 'lightgreen', 'lightcoral', 'lightgreen']
    
    axes[0].bar(models, scores, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Performance Comparison', fontsize=14)
    axes[0].set_ylim([0.8, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, score in enumerate(scores):
        axes[0].text(i, score + 0.01, f'{score:.4f}', ha='center', fontsize=10)
    
    # Improvement rate
    improvements = [
        0,
        (tuned_score - default_score) * 100,
        0,
        (tuned_score_xgb - default_score_xgb) * 100
    ]
    
    axes[1].bar(models, improvements, color=colors, edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Improvement (%)')
    axes[1].set_title('Improvement from Tuning', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, imp in enumerate(improvements):
        if imp > 0:
            axes[1].text(i, imp + 0.2, f'{imp:.2f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Understanding Hyperparameters**

     * Difference from model parameters
     * Key hyperparameters and their roles
     * Proper design of search spaces
  2. **Grid Search**

     * Optimization through exhaustive search
     * Using scikit-learn GridSearchCV
     * Trade-off between computational cost and search efficiency
  3. **Random Search**

     * Efficiency of probabilistic sampling
     * Direct search from continuous distributions
     * Advantages over grid search
  4. **Importance of Cross-Validation**

     * Choosing appropriate CV strategies
     * Comprehensive evaluation with multiple metrics
     * Detecting and preventing overfitting
  5. **Practical Tuning**

     * Optimizing Random Forest and XGBoost
     * Improvement from default settings
     * Visualization and interpretation of results

### Method Selection Guidelines

Situation | Recommended Method | Reason  
---|---|---  
Few parameters (2-3) | Grid Search | Exhaustive search is practical  
Many parameters (4+) | Random Search | Better computational efficiency  
Continuous-valued parameters | Random Search | Direct sampling from distributions  
Limited computational resources | Random Search | Search count can be controlled  
Highest accuracy needed | Combine both | Two-stage: coarse to fine search  
  
### To the Next Chapter

In Chapter 2, we will learn about **Bayesian Optimization** :

  * Surrogate models using Gaussian processes
  * Acquisition function design
  * Implementation using Optuna
  * Performance comparison with traditional methods
  * Practical application examples

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Explain the difference between hyperparameters and model parameters from three perspectives (definition, determination method, examples).

Sample Answer

**Answer** :

Perspective | Hyperparameters | Model Parameters  
---|---|---  
**Definition** | Values set by humans before training | Values automatically optimized through training  
**Determination Method** | Trial and error, search algorithms, experience | Learned from training data via gradient descent, etc.  
**Examples** | Learning rate, tree depth, regularization coefficient | Linear regression coefficients, neural network weights  
  
**Additional Explanation** :

  * Hyperparameters control model structure and learning process
  * Model parameters represent data patterns
  * Appropriate hyperparameter selection makes model parameter learning more efficient

### Exercise 2 (Difficulty: medium)

Calculate the total number of combinations for the following parameter grid and discuss the computational cost of grid search.
    
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    # Using 5-fold cross-validation
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculate the total number of combinations for the following
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    # Candidate count for each parameter
    param_counts = [len(v) for v in param_grid.values()]
    print("Candidate count per parameter:")
    for param, count in zip(param_grid.keys(), param_counts):
        print(f"  {param}: {count}")
    
    # Total combinations
    total_combinations = np.prod(param_counts)
    print(f"\nTotal combinations: {total_combinations:,}")
    
    # Total training runs with 5-fold cross-validation
    cv_folds = 5
    total_fits = total_combinations * cv_folds
    print(f"Total training runs with 5-fold CV: {total_fits:,}")
    
    # Assuming 1 minute per training run
    time_per_fit = 1  # minutes
    total_time_minutes = total_fits * time_per_fit
    total_time_hours = total_time_minutes / 60
    
    print(f"\nComputation time (assuming 1 minute per training):")
    print(f"  {total_time_minutes:,} minutes")
    print(f"  {total_time_hours:.1f} hours")
    

**Output** :
    
    
    Candidate count per parameter:
      n_estimators: 5
      max_depth: 6
      min_samples_split: 4
      learning_rate: 4
    
    Total combinations: 480
    Total training runs with 5-fold CV: 2,400
    
    Computation time (assuming 1 minute per training):
      2,400 minutes
      40.0 hours
    

**Discussion** :

  * Combinations increase exponentially as parameter count grows
  * Cross-validation further increases computational cost
  * This example requires approximately 40 hours of computation
  * Random search with search count limited to 100 would take approximately 8.3 hours (500 training runs)

### Exercise 3 (Difficulty: medium)

Compare the pros and cons of grid search and random search, and explain in which scenarios random search is advantageous.

Sample Answer

**Answer** :

Aspect | Grid Search | Random Search  
---|---|---  
**Search Method** | Exhaustive all combinations | Random sampling  
**Computational Cost** | Increases exponentially | Search count can be controlled  
**Optimal Solution Guarantee** | Guaranteed within search space | Probabilistic (no guarantee)  
**Continuous Value Support** | Requires discretization | Direct sampling from continuous distributions  
**High-Dimensional Search** | Difficult (combinatorial explosion) | Linear with respect to dimensions  
  
**Scenarios Where Random Search is Advantageous** :

  1. **Many parameters (4 or more)**
     * Grid search suffers from combinatorial explosion
     * Random search can fix the number of searches
  2. **Optimizing continuous-valued parameters**
     * Continuous values like learning rate, regularization coefficient
     * Can sample directly from distributions
  3. **When some parameters are more important**
     * As shown by Bergstra & Bengio (2012), widely explores important parameter ranges
     * Grid search is limited to uniform spacing
  4. **Limited computational resources**
     * When time constraints exist
     * Can control search count within budget

### Exercise 4 (Difficulty: hard)

Implement hyperparameter tuning for RandomForestClassifier on the following dataset and report the improvement rate from default settings.
    
    
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Implement hyperparameter tuning for RandomForestClassifier o
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from scipy.stats import randint, uniform
    import time
    
    # Data preparation
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    print("=== Tuning on Wine Dataset ===\n")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Number of classes: {len(data.target_names)}")
    
    # 1. Performance with default settings
    print("\n1. Evaluation with Default Settings")
    rf_default = RandomForestClassifier(random_state=42)
    rf_default.fit(X_train, y_train)
    y_pred_default = rf_default.predict(X_test)
    default_accuracy = accuracy_score(y_test, y_pred_default)
    
    print(f"Accuracy: {default_accuracy:.4f}")
    
    # 2. Tuning with random search
    print("\n2. Tuning with Random Search")
    
    param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9)
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # Evaluate with best model
    y_pred_tuned = random_search.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nCV accuracy: {random_search.best_score_:.4f}")
    print(f"Test accuracy: {tuned_accuracy:.4f}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    
    # 3. Calculate improvement rate
    improvement = (tuned_accuracy - default_accuracy) * 100
    improvement_pct = (tuned_accuracy / default_accuracy - 1) * 100
    
    print(f"\n=== Summary of Results ===")
    print(f"Default settings: {default_accuracy:.4f}")
    print(f"After tuning: {tuned_accuracy:.4f}")
    print(f"Absolute improvement: {improvement:.2f} points")
    print(f"Relative improvement: {improvement_pct:.2f}%")
    
    # 4. Detailed classification report
    print(f"\n=== Classification Report (After Tuning) ===")
    print(classification_report(y_test, y_pred_tuned,
                              target_names=data.target_names))
    
    # 5. Visualization
    import matplotlib.pyplot as plt
    import pandas as pd
    
    results_df = pd.DataFrame(random_search.cv_results_)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score distribution
    axes[0, 0].hist(results_df['mean_test_score'], bins=20,
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=random_search.best_score_, color='r',
                       linestyle='--', label='Best Score')
    axes[0, 0].set_xlabel('CV Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameter influence: n_estimators
    axes[0, 1].scatter(results_df['param_n_estimators'],
                       results_df['mean_test_score'], alpha=0.5)
    axes[0, 1].set_xlabel('n_estimators')
    axes[0, 1].set_ylabel('CV Accuracy')
    axes[0, 1].set_title('Influence of n_estimators')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameter influence: max_depth
    axes[1, 0].scatter(results_df['param_max_depth'],
                       results_df['mean_test_score'], alpha=0.5)
    axes[1, 0].set_xlabel('max_depth')
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].set_title('Influence of max_depth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Default vs Tuned
    comparison = ['Default', 'Tuned']
    scores = [default_accuracy, tuned_accuracy]
    colors = ['lightcoral', 'lightgreen']
    
    axes[1, 1].bar(comparison, scores, color=colors,
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_ylim([0.9, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, score in enumerate(scores):
        axes[1, 1].text(i, score + 0.005, f'{score:.4f}',
                        ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

**Example Output** :
    
    
    === Tuning on Wine Dataset ===
    
    Training data: (142, 13)
    Test data: (36, 13)
    Number of classes: 3
    
    1. Evaluation with Default Settings
    Accuracy: 0.9722
    
    2. Tuning with Random Search
    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    
    Best parameters:
      max_depth: 18
      max_features: 0.3456
      min_samples_leaf: 1
      min_samples_split: 2
      n_estimators: 287
    
    CV accuracy: 0.9859
    Test accuracy: 1.0000
    Execution time: 15.23 seconds
    
    === Summary of Results ===
    Default settings: 0.9722
    After tuning: 1.0000
    Absolute improvement: 2.78 points
    Relative improvement: 2.86%
    
    === Classification Report (After Tuning) ===
                  precision    recall  f1-score   support
    
         class_0       1.00      1.00      1.00        14
         class_1       1.00      1.00      1.00        15
         class_2       1.00      1.00      1.00         7
    
        accuracy                           1.00        36
       macro avg       1.00      1.00      1.00        36
    weighted avg       1.00      1.00      1.00        36
    

### Exercise 5 (Difficulty: hard)

Explain the dangers of data leakage in cross-validation and show the correct implementation method. Consider especially in the context of scaling and hyperparameter search.

Sample Answer

**Answer** :

**What is Data Leakage** :

Information leaking across the boundary between training and test data, causing model performance to be overestimated.

**Specific Dangers** :

  1. **Leakage in Scaling**
     * Scaling all data then train/test split allows test data statistics to leak into training
     * Using test data's mean and standard deviation
  2. **Leakage in Feature Selection**
     * Feature selection on all data then train/test split allows test data information to influence selection
  3. **Leakage in Cross-Validation**
     * Preprocessing outside CV allows test fold information to leak to each fold

**Incorrect Implementation Example** :
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # ❌ Wrong: Scale all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit on all data
    
    # Then cross-validation
    scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
    # → Test fold information leaks to training folds
    

**Correct Implementation Example** :
    
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # ✅ Correct: Use Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    
    # Cross-validation with Pipeline
    # Scaler is fit on training data only for each fold
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    # Hyperparameter search similarly
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    

**Demonstration Experiment** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Demonstration Experiment:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    # Generate data (features with different scales)
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=10, random_state=42)
    
    # Intentionally change scale
    X[:, :10] = X[:, :10] * 1000  # Multiply first 10 features by 1000
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== Demonstration of Data Leakage ===\n")
    
    # 1. Incorrect method (with data leakage)
    scaler_wrong = StandardScaler()
    X_train_wrong = scaler_wrong.fit_transform(X_train)
    X_test_wrong = scaler_wrong.transform(X_test)
    
    # Leakage also occurs in CV
    X_all_scaled = StandardScaler().fit_transform(X)
    cv_scores_wrong = cross_val_score(
        RandomForestClassifier(random_state=42),
        X_all_scaled, y, cv=5
    )
    
    print("❌ Incorrect Method (CV after scaling all data)")
    print(f"CV accuracy: {cv_scores_wrong.mean():.4f} (±{cv_scores_wrong.std():.4f})")
    
    # 2. Correct method (prevent leakage with Pipeline)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    cv_scores_correct = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"\n✅ Correct Method (using Pipeline)")
    print(f"CV accuracy: {cv_scores_correct.mean():.4f} (±{cv_scores_correct.std():.4f})")
    
    # Calculate difference
    difference = cv_scores_wrong.mean() - cv_scores_correct.mean()
    print(f"\nDegree of overestimation: {difference:.4f} ({difference*100:.2f}% points)")
    
    print("\n=== Conclusion ===")
    print("Performance is overestimated due to data leakage")
    print("Correct evaluation is possible using Pipeline")
    

**Example Output** :
    
    
    === Demonstration of Data Leakage ===
    
    ❌ Incorrect Method (CV after scaling all data)
    CV accuracy: 0.9120 (±0.0234)
    
    ✅ Correct Method (using Pipeline)
    CV accuracy: 0.9050 (±0.0287)
    
    Degree of overestimation: 0.0070 (0.70% points)
    
    === Conclusion ===
    Performance is overestimated due to data leakage
    Correct evaluation is possible using Pipeline
    

**Best Practices** :

  * Always use Pipeline to integrate preprocessing and model
  * Execute cross-validation on the entire pipeline including preprocessing
  * Fit on training data, only transform on test data
  * Also perform hyperparameter search on the entire Pipeline

* * *

## References

  1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. _Journal of Machine Learning Research_ , 13(1), 281-305.
  2. Feurer, M., & Hutter, F. (2019). Hyperparameter optimization. In _Automated Machine Learning_ (pp. 3-33). Springer.
  3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_ (2nd ed.). Springer.
  4. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
