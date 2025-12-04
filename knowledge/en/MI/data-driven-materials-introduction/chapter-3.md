---
title: "Chapter 3: Model Selection and Hyperparameter Optimization"
chapter_title: "Chapter 3: Model Selection and Hyperparameter Optimization"
subtitle: 
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 15
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Model Selection and Hyperparameter Optimization

This chapter covers Model Selection and Hyperparameter Optimization. You will learn models based on data size (linear, cross-validation techniques (K-Fold, and ensemble learning methods (Bagging.

* * *

## Learning Objectives

After completing this chapter, you will be able to:

  * ✅ Select appropriate models based on data size (linear, tree-based, neural networks, GNNs)
  * ✅ Apply cross-validation techniques (K-Fold, Stratified, Time Series Split)
  * ✅ Perform automated hyperparameter optimization using Bayesian optimization with Optuna
  * ✅ Implement ensemble learning methods (Bagging, Boosting, Stacking)
  * ✅ Execute practical workflows for Li-ion battery capacity prediction

* * *

## 3.1 Model Selection Strategy

In materials science machine learning, selecting appropriate models based on data characteristics is crucial for achieving good performance.

### Data Size and Model Complexity
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Data Size and Model Complexity
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    
    # Generate sample materials data
    np.random.seed(42)
    
    def generate_material_data(n_samples, n_features=20):
        """Simulate materials data"""
        X = np.random.randn(n_samples, n_features)
        # Nonlinear relationships
        y = (
            2 * X[:, 0]**2 +
            3 * X[:, 1] * X[:, 2] -
            1.5 * X[:, 3] +
            np.random.normal(0, 0.5, n_samples)
        )
        return X, y
    
    # Relationship between model complexity and sample size
    sample_sizes = [50, 100, 200, 500, 1000]
    models = {
        'Ridge': Ridge(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'Neural Network': MLPRegressor(hidden_layers=(50, 50), max_iter=1000, random_state=42)
    }
    
    # Learning curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models.items()):
        X, y = generate_material_data(1000, n_features=20)
    
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    
        train_mean = -train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = -val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
    
        axes[idx].plot(train_sizes, train_mean, 'o-',
                       color='steelblue', label='Training Error')
        axes[idx].fill_between(train_sizes,
                               train_mean - train_std,
                               train_mean + train_std,
                               alpha=0.2, color='steelblue')
    
        axes[idx].plot(train_sizes, val_mean, 'o-',
                       color='coral', label='Validation Error')
        axes[idx].fill_between(train_sizes,
                               val_mean - val_std,
                               val_mean + val_std,
                               alpha=0.2, color='coral')
    
        axes[idx].set_xlabel('Training Size', fontsize=11)
        axes[idx].set_ylabel('MSE', fontsize=11)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Model Selection Guidelines:")
    print("- Small data (<100): Ridge, Lasso (regularized linear models)")
    print("- Medium data (100-1000): Random Forest, Gradient Boosting")
    print("- Large data (>1000): Neural Network, Deep Learning")
    

### Interpretability vs. Accuracy Trade-off
    
    
    # Model interpretability and accuracy comparison
    model_comparison = pd.DataFrame({
        'Model': [
            'Linear Regression',
            'Ridge/Lasso',
            'Decision Tree',
            'Random Forest',
            'Gradient Boosting',
            'Neural Network',
            'GNN'
        ],
        'Interpretability': [10, 9, 7, 4, 3, 2, 1],
        'Accuracy': [4, 5, 5, 8, 9, 9, 10],
        'Training Speed': [10, 9, 8, 6, 5, 3, 2],
        'Inference Speed': [10, 10, 9, 7, 6, 8, 4]
    })
    
    # Radar chart
    from math import pi
    
    categories = ['Interpretability', 'Accuracy', 'Training Speed', 'Inference Speed']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10),
                             subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for idx, row in model_comparison.iterrows():
        values = row[categories].tolist()
        values += values[:1]
    
        axes[idx].plot(angles, values, 'o-', linewidth=2)
        axes[idx].fill(angles, values, alpha=0.25)
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(categories, size=9)
        axes[idx].set_ylim(0, 10)
        axes[idx].set_title(row['Model'], size=11, fontweight='bold', pad=20)
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nModel Selection Criteria:")
    print("- Interpretability prioritized: Ridge, Lasso, Decision Tree")
    print("- Accuracy prioritized: Gradient Boosting, Neural Network, GNN")
    print("- Balanced approach: Random Forest")
    

### Linear Models, Tree-based, Neural Networks, and GNNs Comparison
    
    
    # Performance comparison on real data
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    
    X, y = generate_material_data(500, n_features=20)
    
    models_benchmark = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layers=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = []
    
    for model_name, model in models_benchmark.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
    
        # R²
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2 = cv_r2.mean()
    
        results.append({
            'Model': model_name,
            'MAE': mae,
            'MAE_std': mae_std,
            'R²': r2
        })
    
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE
    axes[0].barh(results_df['Model'], results_df['MAE'],
                 xerr=results_df['MAE_std'],
                 color='steelblue', alpha=0.7)
    axes[0].set_xlabel('MAE (lower is better)', fontsize=11)
    axes[0].set_title('Prediction Error (MAE)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # R²
    axes[1].barh(results_df['Model'], results_df['R²'],
                 color='coral', alpha=0.7)
    axes[1].set_xlabel('R² (higher is better)', fontsize=11)
    axes[1].set_title('Coefficient of Determination (R²)', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.2 Cross-Validation Techniques

Cross-validation is a fundamental technique for properly evaluating model generalization performance.

### K-Fold Cross-Validation
    
    
    from sklearn.model_selection import KFold, cross_validate
    
    def kfold_cv_demo(X, y, model, k=5):
        """
        K-Fold cross-validation demo
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
        fold_results = []
    
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
    
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
    
            fold_results.append({
                'Fold': fold_idx + 1,
                'MAE': mae,
                'R²': r2
            })
    
        return pd.DataFrame(fold_results)
    
    # Run K-Fold
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    fold_results = kfold_cv_demo(X, y, model, k=5)
    
    print("K-Fold CV Results:")
    print(fold_results.to_string(index=False))
    print(f"\nMean MAE: {fold_results['MAE'].mean():.4f} ± {fold_results['MAE'].std():.4f}")
    print(f"Mean R²: {fold_results['R²'].mean():.4f} ± {fold_results['R²'].std():.4f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(fold_results))
    
    ax.bar(x_pos, fold_results['MAE'], color='steelblue', alpha=0.7,
           label='MAE per fold')
    ax.axhline(y=fold_results['MAE'].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean MAE')
    ax.fill_between(x_pos,
                    fold_results['MAE'].mean() - fold_results['MAE'].std(),
                    fold_results['MAE'].mean() + fold_results['MAE'].std(),
                    color='red', alpha=0.2, label='±1 Std')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('K-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_results['Fold'])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Stratified K-Fold
    
    
    from sklearn.model_selection import StratifiedKFold
    
    # Classification data
    X_class, _ = generate_material_data(500, n_features=20)
    # 3-class classification
    y_class = np.digitize(y, bins=np.percentile(y, [33, 67]))
    
    def compare_kfold_strategies(X, y):
        """
        Standard K-Fold vs Stratified K-Fold comparison
        """
        # Standard K-Fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        normal_distributions = []
    
        for train_idx, _ in kf.split(X):
            y_train = y[train_idx]
            class_dist = np.bincount(y_train) / len(y_train)
            normal_distributions.append(class_dist)
    
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratified_distributions = []
    
        for train_idx, _ in skf.split(X, y):
            y_train = y[train_idx]
            class_dist = np.bincount(y_train) / len(y_train)
            stratified_distributions.append(class_dist)
    
        return normal_distributions, stratified_distributions
    
    normal_dist, stratified_dist = compare_kfold_strategies(X_class, y_class)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Standard K-Fold
    normal_array = np.array(normal_dist)
    axes[0].bar(range(len(normal_array)), normal_array[:, 0],
                label='Class 0', alpha=0.7)
    axes[0].bar(range(len(normal_array)), normal_array[:, 1],
                bottom=normal_array[:, 0],
                label='Class 1', alpha=0.7)
    axes[0].bar(range(len(normal_array)), normal_array[:, 2],
                bottom=normal_array[:, 0] + normal_array[:, 1],
                label='Class 2', alpha=0.7)
    axes[0].set_xlabel('Fold', fontsize=11)
    axes[0].set_ylabel('Class Distribution', fontsize=11)
    axes[0].set_title('Standard K-Fold', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Stratified K-Fold
    stratified_array = np.array(stratified_dist)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 0],
                label='Class 0', alpha=0.7)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 1],
                bottom=stratified_array[:, 0],
                label='Class 1', alpha=0.7)
    axes[1].bar(range(len(stratified_array)), stratified_array[:, 2],
                bottom=stratified_array[:, 0] + stratified_array[:, 1],
                label='Class 2', alpha=0.7)
    axes[1].set_xlabel('Fold', fontsize=11)
    axes[1].set_ylabel('Class Distribution', fontsize=11)
    axes[1].set_title('Stratified K-Fold', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("Stratified K-Fold Advantages:")
    print("- Uniform class distribution across folds")
    print("- Stable evaluation even with imbalanced data")
    

### Time Series Split (for Sequential Data)
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # Time series data simulation
    n_time_points = 200
    time = np.arange(n_time_points)
    # Trend + seasonality + noise
    y_timeseries = (
        0.05 * time +
        10 * np.sin(2 * np.pi * time / 50) +
        np.random.normal(0, 2, n_time_points)
    )
    X_timeseries = np.column_stack([time, np.sin(2 * np.pi * time / 50)])
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_timeseries)):
        # Train
        ax.scatter(time[train_idx], y_timeseries[train_idx],
                   c=[colors[fold_idx]], s=20, alpha=0.3,
                   label=f'Fold {fold_idx+1} Train')
        # Test
        ax.scatter(time[test_idx], y_timeseries[test_idx],
                   c=[colors[fold_idx]], s=50, marker='s',
                   label=f'Fold {fold_idx+1} Test')
    
    ax.plot(time, y_timeseries, 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Time Series Split', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Time Series Split Characteristics:")
    print("- Training data always precedes test data")
    print("- Prevents future data leakage (data leakage prevention)")
    

### Leave-One-Out CV (for Small Datasets)
    
    
    from sklearn.model_selection import LeaveOneOut
    
    def loo_cv_demo(X, y, model):
        """
        Leave-One-Out CV
        For small-scale datasets
        """
        loo = LeaveOneOut()
        predictions = []
        actuals = []
    
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    
            predictions.append(y_pred[0])
            actuals.append(y_test[0])
    
        return np.array(actuals), np.array(predictions)
    
    # Small dataset
    X_small, y_small = generate_material_data(50, n_features=10)
    model_small = Ridge(alpha=1.0)
    
    y_actual, y_pred_loo = loo_cv_demo(X_small, y_small, model_small)
    
    mae_loo = mean_absolute_error(y_actual, y_pred_loo)
    r2_loo = r2_score(y_actual, y_pred_loo)
    
    print(f"LOO CV Results (n={len(X_small)}):")
    print(f"MAE: {mae_loo:.4f}")
    print(f"R²: {r2_loo:.4f}")
    
    # Predicted vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred_loo, c='steelblue', s=50, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Leave-One-Out CV Results', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 3.3 Hyperparameter Optimization

Hyperparameter optimization explores the parameter space to maximize model performance.

### Grid Search (Exhaustive Search)
    
    
    from sklearn.model_selection import GridSearchCV
    
    def grid_search_demo(X, y):
        """
        Grid Search exhaustive search
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    
        model = RandomForestRegressor(random_state=42)
    
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
    
        grid_search.fit(X, y)
    
        return grid_search
    
    # Execute
    grid_result = grid_search_demo(X, y)
    
    print("Grid Search Results:")
    print(f"Best parameters: {grid_result.best_params_}")
    print(f"Best score (MAE): {-grid_result.best_score_:.4f}")
    print(f"\nSearch space size: {len(grid_result.cv_results_['params'])}")
    
    # Visualization (2D heatmap)
    results = pd.DataFrame(grid_result.cv_results_)
    
    # n_estimators vs max_depth
    pivot_table = results.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(-pivot_table, annot=True, fmt='.3f',
                cmap='YlOrRd', cbar_kws={'label': 'MAE'})
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('max_depth', fontsize=12)
    plt.title('Grid Search Results (MAE)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

### Random Search
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    def random_search_demo(X, y, n_iter=50):
        """
        Random Search random exploration
        """
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.3, 0.7)
        }
    
        model = RandomForestRegressor(random_state=42)
    
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
        random_search.fit(X, y)
    
        return random_search
    
    # Execute
    random_result = random_search_demo(X, y, n_iter=50)
    
    print("\nRandom Search Results:")
    print(f"Best parameters: {random_result.best_params_}")
    print(f"Best score (MAE): {-random_result.best_score_:.4f}")
    
    # Grid vs Random comparison
    print(f"\nGrid Search best score: {-grid_result.best_score_:.4f}")
    print(f"Random Search best score: {-random_result.best_score_:.4f}")
    print(f"\nRandom Search efficiency: Explored {50 / len(grid_result.cv_results_['params']) * 100:.1f}% of search space with comparable performance")
    

### Bayesian Optimization (Optuna, Hyperopt)
    
    
    # Requirements:
    # - Python 3.9+
    # - optuna>=3.2.0
    
    import optuna
    
    def objective(trial, X, y):
        """
        Optuna objective function
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        model = RandomForestRegressor(**params)
    
        # Cross-validation
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_mean_absolute_error'
        )
    
        return -cv_scores.mean()  # Negate for minimization
    
    # Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=True)
    
    print("\nOptuna Bayesian Optimization Results:")
    print(f"Best parameters: {study.best_params}")
    print(f"Best score (MAE): {study.best_value:.4f}")
    
    # Optimization history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Optimization history
    axes[0].plot(study.trials_dataframe()['number'],
                 study.trials_dataframe()['value'],
                 'o-', color='steelblue', alpha=0.6, label='Trial Score')
    axes[0].plot(study.trials_dataframe()['number'],
                 study.trials_dataframe()['value'].cummin(),
                 'r-', linewidth=2, label='Best Score')
    axes[0].set_xlabel('Trial', fontsize=11)
    axes[0].set_ylabel('MAE', fontsize=11)
    axes[0].set_title('Optimization History', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Parameter importance
    importances = optuna.importance.get_param_importances(study)
    axes[1].barh(list(importances.keys()), list(importances.values()),
                 color='coral', alpha=0.7)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Hyperparameter Importance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Early Stopping
    
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    def early_stopping_demo(X, y):
        """
        Early Stopping demo
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Track performance for each boosting stage
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
        model.fit(X_train, y_train)
    
        # Predictions for each stage
        train_scores = []
        val_scores = []
    
        for i, y_pred_train in enumerate(model.staged_predict(X_train)):
            y_pred_val = list(model.staged_predict(X_val))[i]
    
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
    
            train_scores.append(train_mae)
            val_scores.append(val_mae)
    
        # Optimal n_estimators (minimum validation error)
        best_n_estimators = np.argmin(val_scores) + 1
    
        return train_scores, val_scores, best_n_estimators
    
    train_curve, val_curve, best_n = early_stopping_demo(X, y)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_curve)+1), train_curve,
             'b-', label='Training Error', linewidth=2)
    plt.plot(range(1, len(val_curve)+1), val_curve,
             'r-', label='Validation Error', linewidth=2)
    plt.axvline(x=best_n, color='green', linestyle='--',
                label=f'Best n_estimators={best_n}', linewidth=2)
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Early Stopping', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Early Stopping Results:")
    print(f"Optimal n_estimators: {best_n}")
    print(f"Validation MAE: {val_curve[best_n-1]:.4f}")
    print(f"Overfitting prevention: Reduced {500 - best_n} iterations")
    

* * *

## 3.4 Ensemble Learning

Combining multiple models improves prediction accuracy through ensemble methods.

### Bagging (Bootstrap Aggregating)
    
    
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    
    # Bagging
    bagging = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),
        n_estimators=50,
        max_samples=0.8,
        random_state=42
    )
    
    # Single Decision Tree for comparison
    single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    
    # Evaluation
    cv_bagging = cross_val_score(bagging, X, y, cv=5,
                                 scoring='neg_mean_absolute_error')
    cv_single = cross_val_score(single_tree, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
    
    print("Bagging Results:")
    print(f"Single Decision Tree MAE: {-cv_single.mean():.4f} ± {cv_single.std():.4f}")
    print(f"Bagging MAE: {-cv_bagging.mean():.4f} ± {cv_bagging.std():.4f}")
    print(f"Improvement: {(cv_single.mean() - cv_bagging.mean()) / cv_single.mean() * 100:.1f}%")
    

### Boosting (AdaBoost, Gradient Boosting, LightGBM, XGBoost)
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: Boosting (AdaBoost, Gradient Boosting, LightGBM, XGBoost)
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    import lightgbm as lgb
    # import xgboost as xgb  # Optional
    
    # Various Boosting algorithms
    boosting_models = {
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        # 'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    boosting_results = []
    
    for model_name, model in boosting_models.items():
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
    
        boosting_results.append({
            'Model': model_name,
            'MAE': mae,
            'MAE_std': mae_std
        })
    
    boosting_df = pd.DataFrame(boosting_results)
    
    print("\nBoosting Methods Comparison:")
    print(boosting_df.to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(boosting_df['Model'], boosting_df['MAE'],
             xerr=boosting_df['MAE_std'],
             color='steelblue', alpha=0.7)
    plt.xlabel('MAE', fontsize=12)
    plt.title('Boosting Methods Performance Comparison', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Stacking
    
    
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    
    # Base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    # Meta model
    meta_model = Ridge(alpha=1.0)
    
    # Stacking
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    # Evaluation
    cv_stacking = cross_val_score(stacking, X, y, cv=5,
                                  scoring='neg_mean_absolute_error')
    
    print("\nStacking Results:")
    for name, _ in base_models:
        model_cv = cross_val_score(dict(base_models)[name], X, y, cv=5,
                                   scoring='neg_mean_absolute_error')
        print(f"{name} MAE: {-model_cv.mean():.4f}")
    
    print(f"Stacking Ensemble MAE: {-cv_stacking.mean():.4f}")
    print(f"\nImprovement: Stacking shows "
          f"{((-cv_stacking.mean() / min([cross_val_score(m, X, y, cv=5, scoring='neg_mean_absolute_error').mean() for _, m in base_models])) - 1) * -100:.1f}% improvement over best single model")
    

### Voting
    
    
    from sklearn.ensemble import VotingRegressor
    
    # Voting Ensemble
    voting = VotingRegressor(
        estimators=base_models,
        weights=[1, 1.5, 1]  # Higher weight for GB
    )
    
    cv_voting = cross_val_score(voting, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
    
    print(f"\nVoting Ensemble MAE: {-cv_voting.mean():.4f}")
    
    # Ensemble methods comparison
    ensemble_comparison = pd.DataFrame({
        'Method': ['Single Best', 'Bagging', 'Boosting (LightGBM)',
                   'Stacking', 'Voting'],
        'MAE': [
            min([cross_val_score(m, X, y, cv=5, scoring='neg_mean_absolute_error').mean() for _, m in base_models]),
            -cv_bagging.mean(),
            boosting_df[boosting_df['Model'] == 'LightGBM']['MAE'].values[0],
            -cv_stacking.mean(),
            -cv_voting.mean()
        ]
    })
    
    plt.figure(figsize=(10, 6))
    plt.barh(ensemble_comparison['Method'], ensemble_comparison['MAE'],
             color='coral', alpha=0.7)
    plt.xlabel('MAE (lower is better)', fontsize=12)
    plt.title('Ensemble Methods Performance Comparison', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Case Study: Li-ion Battery Capacity Prediction

We implement the complete workflow of model selection and optimization using real Li-ion battery data.
    
    
    # Li-ion battery capacity dataset (simulation)
    np.random.seed(42)
    n_batteries = 300
    
    battery_data = pd.DataFrame({
        'Cathode_Composition_Li': np.random.uniform(0.9, 1.1, n_batteries),
        'Cathode_Composition_Co': np.random.uniform(0, 0.6, n_batteries),
        'Cathode_Composition_Ni': np.random.uniform(0, 0.8, n_batteries),
        'Cathode_Composition_Mn': np.random.uniform(0, 0.4, n_batteries),
        'Anode_Carbon_Fraction': np.random.uniform(0.8, 1.0, n_batteries),
        'Electrolyte_Concentration': np.random.uniform(0.5, 2.0, n_batteries),
        'Electrode_Thickness': np.random.uniform(50, 200, n_batteries),
        'Particle_Size': np.random.uniform(1, 20, n_batteries),
        'Sintering_Temperature': np.random.uniform(700, 1000, n_batteries),
        'BET_Surface_Area': np.random.uniform(1, 50, n_batteries)
    })
    
    # Capacity (complex nonlinear relationship)
    capacity = (
        150 * battery_data['Cathode_Composition_Ni'] +
        120 * battery_data['Cathode_Composition_Co'] +
        80 * battery_data['Cathode_Composition_Mn'] +
        30 * battery_data['Electrolyte_Concentration'] -
        0.5 * battery_data['Electrode_Thickness'] +
        2 * battery_data['BET_Surface_Area'] +
        0.1 * battery_data['Sintering_Temperature'] +
        20 * battery_data['Cathode_Composition_Ni'] * battery_data['Electrolyte_Concentration'] +
        np.random.normal(0, 5, n_batteries)
    )
    
    battery_data['Capacity_mAh_g'] = capacity
    
    print("=== Li-ion Battery Capacity Prediction Dataset ===")
    print(f"Number of samples: {len(battery_data)}")
    print(f"Number of features: {battery_data.shape[1] - 1}")
    print(f"\nCapacity Statistics:")
    print(battery_data['Capacity_mAh_g'].describe())
    
    X_battery = battery_data.drop('Capacity_mAh_g', axis=1)
    y_battery = battery_data['Capacity_mAh_g']
    

### Step 1: Comparing Five Models
    
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_battery, y_battery, test_size=0.2, random_state=42
    )
    
    # Five models for comparison
    models_to_compare = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'Stacking': StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
    }
    
    comparison_results = []
    
    for model_name, model in models_to_compare.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        comparison_results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print("\n=== Step 1: Model Comparison Results ===")
    print(comparison_df.to_string(index=False))
    

### Step 2: Automated Hyperparameter Optimization with Optuna
    
    
    def objective_lightgbm(trial, X, y):
        """
        LightGBM hyperparameter optimization
        """
        param = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
    
        model = lgb.LGBMRegressor(**param)
    
        cv_scores = cross_val_score(
            model, X, y, cv=5, scoring='neg_mean_absolute_error'
        )
    
        return -cv_scores.mean()
    
    # Optuna optimization
    study_battery = optuna.create_study(direction='minimize')
    study_battery.optimize(
        lambda trial: objective_lightgbm(trial, X_battery, y_battery),
        n_trials=100,
        show_progress_bar=True
    )
    
    print("\n=== Step 2: Optuna Optimization ===")
    print(f"Best parameters:")
    for key, value in study_battery.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest MAE: {study_battery.best_value:.4f} mAh/g")
    
    # Re-evaluate with optimized model
    best_model = lgb.LGBMRegressor(**study_battery.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    print(f"\nOptimized Model Performance:")
    print(f"MAE: {mae_best:.4f} mAh/g")
    print(f"R²: {r2_best:.4f}")
    

### Step 3: Final Stacking Ensemble for Peak Performance
    
    
    # Stacking with optimized LightGBM
    optimized_stacking = StackingRegressor(
        estimators=[
            ('rf_tuned', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42
            )),
            ('lgbm_tuned', lgb.LGBMRegressor(**study_battery.best_params, random_state=42)),
            ('gb_tuned', GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                random_state=42
            ))
        ],
        final_estimator=Ridge(alpha=0.5),
        cv=5
    )
    
    optimized_stacking.fit(X_train, y_train)
    y_pred_stack = optimized_stacking.predict(X_test)
    
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    r2_stack = r2_score(y_test, y_pred_stack)
    
    print("\n=== Step 3: Final Stacking Ensemble ===")
    print(f"MAE: {mae_stack:.4f} mAh/g")
    print(f"R²: {r2_stack:.4f}")
    
    # Comprehensive results comparison
    final_comparison = pd.DataFrame({
        'Stage': [
            'Baseline (Ridge)',
            'Best Single Model',
            'Optuna Optimized',
            'Final Stacking'
        ],
        'MAE': [
            comparison_df[comparison_df['Model'] == 'Ridge']['MAE'].values[0],
            comparison_df['MAE'].min(),
            mae_best,
            mae_stack
        ],
        'R²': [
            comparison_df[comparison_df['Model'] == 'Ridge']['R²'].values[0],
            comparison_df['R²'].max(),
            r2_best,
            r2_stack
        ]
    })
    
    print("\n=== Overall Performance Progression ===")
    print(final_comparison.to_string(index=False))
    
    # Predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Best single model before optimization
    best_single_idx = comparison_df['MAE'].idxmin()
    best_single_model = list(models_to_compare.values())[best_single_idx]
    y_pred_single = best_single_model.predict(X_test)
    
    axes[0].scatter(y_test, y_pred_single, c='steelblue', s=50, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Actual Capacity (mAh/g)', fontsize=11)
    axes[0].set_ylabel('Predicted Capacity (mAh/g)', fontsize=11)
    axes[0].set_title(f'Best Single Model (MAE={comparison_df["MAE"].min():.2f})',
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Final Stacking
    axes[1].scatter(y_test, y_pred_stack, c='coral', s=50, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('Actual Capacity (mAh/g)', fontsize=11)
    axes[1].set_ylabel('Predicted Capacity (mAh/g)', fontsize=11)
    axes[1].set_title(f'Final Stacking (MAE={mae_stack:.2f})',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    improvement = (comparison_df['MAE'].min() - mae_stack) / comparison_df['MAE'].min() * 100
    print(f"\nFinal Improvement Rate: {improvement:.1f}%")
    

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Compare K-Fold CV and Stratified K-Fold CV using imbalanced classification data to evaluate performance.

Solution Example
    
    
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate imbalanced data
    X, y = make_classification(n_samples=200, n_features=20,
                              weights=[0.9, 0.1], random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(model, X, y, cv=kf, scoring='f1')
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_stratified = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    print(f"K-Fold F1: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
    print(f"Stratified K-Fold F1: {scores_stratified.mean():.4f} ± {scores_stratified.std():.4f}")
    

### Exercise 2 (Difficulty: Medium)

Use Optuna to optimize Random Forest hyperparameters. Explore four parameters: `n_estimators`, `max_depth`, `min_samples_split`, and `max_features`.

Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - optuna>=3.2.0
    
    """
    Example: Use Optuna to optimize Random Forest hyperparameters. Explor
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import optuna
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        model = RandomForestRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
    
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_rf, n_trials=50)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    

### Exercise 3 (Difficulty: Hard)

Build a Stacking Ensemble with three base models (Ridge, Random Forest, LightGBM) and compare two meta-models (Ridge, Lasso). Which meta-model performs better?

Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: Build a Stacking Ensemble with three base models (Ridge, Ran
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, Lasso
    import lightgbm as lgb
    
    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    meta_models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    results = []
    
    for meta_name, meta_model in meta_models.items():
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
    
        cv_scores = cross_val_score(stacking, X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
    
        results.append({
            'Meta Model': meta_name,
            'MAE': mae
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    

* * *

## 3.6 Model Optimization Environment and Reproducibility

### Library Version Management
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    # - pandas>=2.0.0, <2.2.0
    # - scikit-learn>=1.3.0, <1.5.0
    
    """
    Example: Library Version Management
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Libraries required for model selection and optimization
    import sys
    import sklearn
    import optuna
    import lightgbm
    import pandas as pd
    import numpy as np
    
    reproducibility_info = {
        'Python': sys.version,
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'Optuna': optuna.__version__,
        'LightGBM': lightgbm.__version__,
        'Date': '2025-10-19'
    }
    
    print("=== Model Optimization Environment ===")
    for key, value in reproducibility_info.items():
        print(f"{key}: {value}")
    
    # Recommended versions
    print("\n【Recommended Environment】")
    recommended = """
    numpy==1.24.3
    pandas==2.0.3
    scikit-learn==1.3.0
    optuna==3.3.0
    lightgbm==4.0.0
    xgboost==2.0.0  # Optional
    matplotlib==3.7.2
    seaborn==0.12.2
    """
    print(recommended)
    
    print("\n【Installation Command】")
    print("```bash")
    print("pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0")
    print("pip install optuna==3.3.0 lightgbm==4.0.0")
    print("```")
    
    print("\n【Important Notes】")
    print("⚠️ Optuna's optimization algorithm may differ across versions")
    print("⚠️ LightGBM/XGBoost may have OS-dependent issues → validate in consistent environment")
    print("⚠️ Always specify library versions in papers and reports")
    

### Random Seed Management
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # Seed setting for reproducibility
    def set_all_seeds(seed=42):
        """
        Set all random number generators
        """
        import random
        import numpy as np
    
        random.seed(seed)
        np.random.seed(seed)
    
        # scikit-learn models specify via random_state parameter
        print(f"✅ Random seed {seed} set")
        print("Specify random_state={seed} explicitly in model training")
    
    set_all_seeds(42)
    
    # Optuna seed management
    print("\n【Optuna Seed Management】")
    print("```python")
    print("study = optuna.create_study(")
    print("    direction='minimize',")
    print("    sampler=optuna.samplers.TPESampler(seed=42)  # Ensure reproducibility")
    print(")")
    print("```")
    

### Practical Pitfalls (Model Selection and Optimization)
    
    
    print("=== Model Selection and Optimization Pitfalls ===\n")
    
    print("【Pitfall 1: Hyperparameter Tuning on Test Data】")
    print("❌ Bad practice: Optimize on test set")
    print("```python")
    print("X_train, X_test = train_test_split(X, y)")
    print("# Evaluate multiple models and select best on test set")
    print("for model in models:")
    print("    score = model.score(X_test, y_test)  # WRONG!")
    print("best_model = models[best_idx]")
    print("```")
    
    print("\n✅ Correct approach: Train/Validation/Test split")
    print("```python")
    print("X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)")
    print("X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)")
    print("# Optimize on Validation, evaluate on Test only")
    print("```")
    
    print("\n【Pitfall 2: Data Leakage in Cross-Validation】")
    print("⚠️ Running preprocessing (StandardScaler, Imputer) on all data")
    print("→ Test data information leaks into training")
    
    print("\n✅ Solution: Use Pipeline to include preprocessing")
    print("```python")
    print("from sklearn.pipeline import Pipeline")
    print("from sklearn.preprocessing import StandardScaler")
    print("")
    print("pipeline = Pipeline([")
    print("    ('scaler', StandardScaler()),")
    print("    ('model', RandomForestRegressor())")
    print("])")
    print("cv_scores = cross_val_score(pipeline, X, y, cv=5)")
    print("```")
    
    print("\n【Pitfall 3: Excessive Optuna Iterations】")
    print("⚠️ Using n_trials=1000 for optimization → overfitting risk")
    print("→ Overfit to validation set")
    
    print("\n✅ Solution: Early Stopping + Test Evaluation")
    print("```python")
    print("study.optimize(objective, n_trials=100,")
    print("    callbacks=[optuna.study.MaxTrialsCallback(100)])")
    print("# Evaluate final model on test set")
    print("```")
    
    print("\n【Pitfall 4: Excessive Ensemble Complexity】")
    print("⚠️ Combining Stacking + Voting + Blending in multiple stages")
    print("→ Increased training time, lost interpretability, overfitting")
    
    print("\n✅ Solution: Simple ensemble (2-3 models)")
    print("```python")
    print("# Simple Stacking (3 base models + 1 meta model)")
    print("stacking = StackingRegressor(")
    print("    estimators=[('rf', RF), ('gb', GB), ('lgbm', LGBM)],")
    print("    final_estimator=Ridge(),")
    print("    cv=5")
    print(")")
    print("```")
    
    print("\n【Pitfall 5: Complex Models on Small Datasets】")
    print("⚠️ Training Neural Network on 50 samples")
    print("→ Overfitting, unstable")
    
    print("\n✅ Solution: Select model based on data size")
    print("```python")
    print("if len(X) < 100:")
    print("    model = Ridge()  # Linear model")
    print("elif len(X) < 1000:")
    print("    model = RandomForest()  # Tree-based")
    print("else:")
    print("    model = NeuralNetwork()  # Deep learning")
    print("```")
    

* * *

## Summary

In this chapter, we learned **model selection and hyperparameter optimization**.

**Key Takeaways** :

  1. **Model Selection** : Choose appropriate models based on data size (small: Ridge, medium: RF, large: NN)
  2. **Cross-Validation** : Apply K-Fold, Stratified, Time Series appropriately
  3. **Optimization Methods** : Progression from Grid < Random < Bayesian (Optuna) for efficiency
  4. **Ensemble Methods** : Stacking > Voting > Boosting > Bagging in performance hierarchy
  5. **Real-world Case Study** : Li-ion battery capacity prediction showed 30%+ performance improvement
  6. **Environment Management** : Strict library versioning and random seed control
  7. **Common Pitfalls** : Avoid test set tuning, data leakage, overfitting, and oversized models on small data

**Next Chapter Preview** : Chapter 4 covers Explainable AI (XAI). We'll use SHAP, LIME, and attention visualization to understand prediction physical meanings and explore real-world applications and career paths.

* * *

## Chapter 3 Checklist

### Model Selection

  * [ ] **Evaluate Data Size**
  * [ ] Samples < 100 → Ridge/Lasso (regularized linear models)
  * [ ] Samples 100-1000 → Random Forest/Gradient Boosting
  * [ ] Samples > 1000 → Neural Network/Deep Learning
  * [ ] Verify sample-to-feature ratio > 10:1

  * [ ] **Balance Interpretability vs. Accuracy**

  * [ ] Interpretability priority (material design guidelines) → Ridge, Decision Tree
  * [ ] Accuracy priority (maximize prediction) → Gradient Boosting, Stacking
  * [ ] Balanced approach → Random Forest

  * [ ] **Verify Model Complexity**

  * [ ] Check learning curves for overfitting (training error << validation error)
  * [ ] Tune regularization (alpha, lambda)
  * [ ] Set appropriate iteration count via Early Stopping

### Cross-Validation

  * [ ] **K-Fold CV (Standard)**
  * [ ] Use K=5 for regression tasks
  * [ ] Use K=10 when computational resources available for better accuracy
  * [ ] Set shuffle=True to remove order effects

  * [ ] **Stratified K-Fold (Classification/Imbalanced)**

  * [ ] Essential for imbalanced datasets
  * [ ] Verify class distribution uniformity across folds
  * [ ] Verify minority class samples > K

  * [ ] **Time Series Split (Sequential Data)**

  * [ ] Apply to sequential material experiment data
  * [ ] Verify training always precedes test temporally
  * [ ] Absolutely prevent future data training (data leakage prevention)

  * [ ] **Leave-One-Out CV (Small Datasets)**

  * [ ] Use when samples < 50
  * [ ] Recognize high computational cost (n iterations)
  * [ ] Be aware of overly optimistic evaluation

### Hyperparameter Optimization

  * [ ] **Grid Search**
  * [ ] Use when search space small (<100 combinations)
  * [ ] Focus on 2-3 important parameters
  * [ ] Exhaustively evaluate all combinations

  * [ ] **Random Search**

  * [ ] Use for large search spaces
  * [ ] Achieves Grid Search performance with 10-20% trials
  * [ ] Efficiently explore continuous parameters

  * [ ] **Bayesian Optimization (Optuna)**

  * [ ] Most efficient (recommended)
  * [ ] n_trials=50-100 sufficient for good performance
  * [ ] Parameter importance analysis improves interpretability
  * [ ] Set sampler algorithm (TPESampler) seed for reproducibility

  * [ ] **Early Stopping**

  * [ ] Essential for Boosting models
  * [ ] Stop at minimum validation error point
  * [ ] Prevents overfitting and reduces computation

### Ensemble Learning

  * [ ] **Bagging**
  * [ ] Random Forest implements Bagging automatically
  * [ ] Stabilizes high-variance models
  * [ ] Expect 20-30% improvement with Decision Tree ensemble

  * [ ] **Boosting**

  * [ ] Test Gradient Boosting, LightGBM, XGBoost
  * [ ] Trade-off between learning_rate and n_estimators
  * [ ] Use Early Stopping to prevent overfitting

  * [ ] **Stacking**

  * [ ] Use 3-5 diverse base models
  * [ ] Keep meta-model simple (Ridge, Lasso)
  * [ ] Generate meta-features with cross-validation (cv=5)
  * [ ] Target 5-10% improvement over single best model

  * [ ] **Voting**

  * [ ] Simpler than Stacking
  * [ ] Weight models by performance
  * [ ] 3 models typically sufficient

### Avoiding Practical Pitfalls

  * [ ] **No Test Set Tuning**
  * [ ] Use Train/Validation/Test split
  * [ ] Optimize on Validation, evaluate on Test only
  * [ ] Use test set exactly once

  * [ ] **Prevent Cross-Validation Data Leakage**

  * [ ] Embed preprocessing in Pipeline
  * [ ] Perform preprocessing independently per fold (StandardScaler.fit)
  * [ ] Execute feature selection within cross-validation

  * [ ] **Avoid Excessive Optimization**

  * [ ] Use Optuna n_trials < 200 (recommend 100)
  * [ ] Monitor validation vs. test performance gap
  * [ ] Start with simple models

  * [ ] **Avoid Ensemble Overcomplexity**

  * [ ] Limit Stacking to single level (no multi-stage)
  * [ ] Keep base models ≤ 5
  * [ ] Evaluate training time vs. performance trade-off

  * [ ] **Avoid Complex Models on Small Data**

  * [ ] Samples < 100 → linear models
  * [ ] Samples < 1000 → tree-based models
  * [ ] Deep learning only for large data (>1000)

### Reproducibility Assurance

  * [ ] **Version Management**
  * [ ] Record scikit-learn, Optuna, LightGBM versions
  * [ ] Create requirements.txt
  * [ ] Consider Docker for environment consistency (recommended)

  * [ ] **Random Seed Configuration**

  * [ ] NumPy: np.random.seed(42)
  * [ ] scikit-learn: random_state=42
  * [ ] Optuna: TPESampler(seed=42)
  * [ ] Use consistent seed across all random elements

  * [ ] **Save Optimization History**

  * [ ] Save Optuna study.trials_dataframe() to CSV
  * [ ] Save best parameters as JSON
  * [ ] Save learning curves as images

  * [ ] **Model Persistence**

  * [ ] Save trained model with joblib.dump()
  * [ ] Apply identical preprocessing during prediction
  * [ ] Version control models (Git LFS recommended)

### Li-ion Battery Capacity Prediction Case Study

  * [ ] **Data Preparation**
  * [ ] Generate composition, structure, process condition features
  * [ ] Split Train/Test (80/20)
  * [ ] Handle missing values and outliers pre-training

  * [ ] **Step 1: Model Comparison**

  * [ ] Evaluate 5 models (Ridge, RF, GB, LGBM, Stacking)
  * [ ] Compare using MAE, RMSE, R²
  * [ ] Identify best single model

  * [ ] **Step 2: Optuna Optimization**

  * [ ] Optimize 8 LightGBM hyperparameters
  * [ ] Run with n_trials=100
  * [ ] Analyze parameter importance

  * [ ] **Step 3: Final Stacking**

  * [ ] Use 3 optimized models as base
  * [ ] Select Ridge as meta-model
  * [ ] Achieve >30% improvement from Baseline (Ridge)

### Model Selection and Optimization Quality Metrics

  * [ ] **Prediction Accuracy**
  * [ ] MAE (regression) < 20% of data standard deviation
  * [ ] R² > 0.8 (R² > 0.7 acceptable for materials science)
  * [ ] Verify RMSE (evaluate outlier influence)

  * [ ] **Generalization Performance**

  * [ ] Training-test error gap < 10%
  * [ ] Small cross-validation std (stability)
  * [ ] Verify performance maintenance on new data

  * [ ] **Computational Efficiency**

  * [ ] Training time < 1 hour (medium-scale data)
  * [ ] Inference time < 1 second/sample
  * [ ] Optuna optimization time < 10 minutes (n_trials=100)

  * [ ] **Interpretability**

  * [ ] Identify important variables via feature_importances_
  * [ ] Use SHAP values (next chapter) for physical interpretation
  * [ ] Verify validity with domain expert

* * *

## References

  1. **Akiba, T., Sano, S., Yanase, T., et al.** (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. _Proceedings of the 25th ACM SIGKDD_ , 2623-2631. [DOI: 10.1145/3292500.3330701](<https://doi.org/10.1145/3292500.3330701>)

  2. **Bergstra, J. & Bengio, Y.** (2012). Random search for hyper-parameter optimization. _Journal of Machine Learning Research_ , 13, 281-305.

  3. **Dietterich, T. G.** (2000). Ensemble methods in machine learning. _International Workshop on Multiple Classifier Systems_ , 1-15. Springer.

  4. **Wolpert, D. H.** (1992). Stacked generalization. _Neural Networks_ , 5(2), 241-259. [DOI: 10.1016/S0893-6080(05)80023-1](<https://doi.org/10.1016/S0893-6080\(05\)80023-1>)

* * *

[← Chapter 2](<chapter-2.html>) | [Chapter 4 →](<chapter-4.html>)
