---
title: "Chapter 2: Hyperparameter Optimization"
chapter_title: "Chapter 2: Hyperparameter Optimization"
subtitle: The Core of AutoML - Automatically Searching for Optimal Settings
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
version: 1.0
created_at: "by:"
---

This chapter covers Hyperparameter Optimization. You will learn fundamentals of hyperparameter optimization, efficient hyperparameter search using Optuna, and Leverage Bayesian optimization with Hyperopt.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the fundamentals of hyperparameter optimization and search strategies
  * ✅ Implement efficient hyperparameter search using Optuna
  * ✅ Leverage Bayesian optimization with Hyperopt
  * ✅ Execute distributed hyperparameter optimization with Ray Tune
  * ✅ Understand and apply advanced HPO techniques (ASHA, PBT, Hyperband)
  * ✅ Select and implement optimization strategies for actual models

* * *

## 2.1 Fundamentals of HPO

### What are Hyperparameters?

**Hyperparameters** are parameters that control the model's learning process and must be set before training begins.

> "While model parameters are optimized during training, hyperparameters are set by humans before training"

### Examples of Hyperparameters

Algorithm | Key Hyperparameters | Impact  
---|---|---  
**Random Forest** | n_estimators, max_depth, min_samples_split | Performance, computational cost, overfitting  
**Gradient Boosting** | learning_rate, n_estimators, max_depth | Convergence speed, performance, overfitting  
**SVM** | C, kernel, gamma | Decision boundary, generalization performance  
**Neural Networks** | learning_rate, batch_size, hidden_units | Convergence, performance, computational efficiency  
  
### Defining the Search Space

The search space defines the range and distribution of possible values for each hyperparameter.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: The search space defines the range and distribution of possi
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Example search space
    search_space = {
        # Integer: number of trees (50 to 500)
        'n_estimators': (50, 500),
    
        # Integer: tree depth (3 to 20)
        'max_depth': (3, 20),
    
        # Float (log scale): learning rate
        'learning_rate': (1e-4, 1e-1, 'log'),
    
        # Categorical: boosting type
        'boosting_type': ['gbdt', 'dart', 'goss'],
    
        # Float (linear scale): regularization parameter
        'reg_alpha': (0.0, 10.0),
    }
    
    print("=== Search Space Definition ===")
    for param, space in search_space.items():
        print(f"{param}: {space}")
    

### Grid Search vs Random Search

#### Grid Search

Exhaustively searches all possible combinations.
    
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import accuracy_score
    import time
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Grid Search space
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    print("=== Grid Search ===")
    print(f"Number of combinations to search: {3 * 4 * 3} = 36 combinations")
    
    # Execute Grid Search
    start_time = time.time()
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    
    # Results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = accuracy_score(y_test, grid_search.predict(X_test))
    
    print(f"\nBest parameters: {best_params}")
    print(f"CV accuracy: {best_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Execution time: {grid_time:.2f} seconds")
    

**Output** :
    
    
    === Grid Search ===
    Number of combinations to search: 3 * 4 * 3 = 36 combinations
    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    
    Best parameters: {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}
    CV accuracy: 0.9162
    Test accuracy: 0.9200
    Execution time: 12.34 seconds
    

#### Random Search

Randomly samples combinations to search.
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # Random Search space (specified by distributions)
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 25),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.3, 0.7)  # Range from 0.3 to 1.0
    }
    
    print("\n=== Random Search ===")
    print(f"Number of random trials: 50")
    
    # Execute Random Search
    start_time = time.time()
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions,
        n_iter=50,  # 50 random trials
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    
    # Results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    test_score = accuracy_score(y_test, random_search.predict(X_test))
    
    print(f"\nBest parameters: {best_params}")
    print(f"CV accuracy: {best_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Execution time: {random_time:.2f} seconds")
    
    # Comparison
    print(f"\n=== Grid vs Random Comparison ===")
    print(f"Grid Search: {grid_time:.2f} seconds for accuracy {test_score:.4f}")
    print(f"Random Search: {random_time:.2f} seconds for accuracy {test_score:.4f}")
    print(f"Time reduction: {(1 - random_time/grid_time)*100:.1f}%")
    

### Classification of Search Strategies
    
    
    ```mermaid
    graph TD
        A[HPO Strategies] --> B[Simple Search]
        A --> C[Adaptive Search]
        A --> D[Multi-stage Search]
    
        B --> B1[Grid Search]
        B --> B2[Random Search]
    
        C --> C1[Bayesian Optimization]
        C --> C2[Evolutionary Algorithms]
        C --> C3[Bandit Algorithms]
    
        D --> D1[Hyperband]
        D --> D2[ASHA]
        D --> D3[PBT]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### Early Stopping

When performance stops improving during training, early stopping terminates training to save computational resources.
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: When performance stops improving during training, early stop
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    
    # Early Stopping example with LightGBM
    print("\n=== Early Stopping Demo ===")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameter settings
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    # With Early Stopping
    print("\nWith Early Stopping:")
    model_es = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    print(f"Actual training rounds: {model_es.best_iteration}")
    
    # Without Early Stopping
    print("\nWithout Early Stopping:")
    model_no_es = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    # Comparison
    pred_es = (model_es.predict(X_test) > 0.5).astype(int)
    pred_no_es = (model_no_es.predict(X_test) > 0.5).astype(int)
    
    print(f"\nAccuracy (Early Stopping): {accuracy_score(y_test, pred_es):.4f}")
    print(f"Accuracy (200 rounds): {accuracy_score(y_test, pred_no_es):.4f}")
    print(f"Computation time reduction: {(1 - model_es.best_iteration/200)*100:.1f}%")
    

* * *

## 2.2 Optuna

### Features of Optuna

**Optuna** is a next-generation hyperparameter optimization framework.

Feature | Description | Benefit  
---|---|---  
**Define-by-run API** | Dynamically define search space | Flexible and intuitive code  
**Pruning** | Early termination of unpromising trials | Significant time savings  
**Parallelization** | Execute multiple trials simultaneously | Speed improvement  
**Visualization** | Detailed visualization of optimization process | Easy understanding and diagnosis  
  
### Study and Trial

Optuna's core concepts:

  * **Study** : Manages the entire optimization task
  * **Trial** : Individual trial (one hyperparameter configuration)

    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    
    """
    Example: Optuna's core concepts:
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # Prepare data
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42
        }
    
        # Train and evaluate model
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
        return score
    
    # Create Study and optimize
    print("=== Optuna Basic Example ===")
    study = optuna.create_study(
        direction='maximize',  # Maximize
        study_name='random_forest_optimization'
    )
    
    # Execute optimization
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Results
    print(f"\nBest accuracy: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Statistics
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    

### Samplers

Optuna supports multiple sampling strategies.

#### TPE (Tree-structured Parzen Estimator)

The default sampler, a type of Bayesian optimization.
    
    
    from optuna.samplers import TPESampler
    
    # TPE sampler
    print("\n=== TPE Sampler ===")
    study_tpe = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    study_tpe.optimize(objective, n_trials=30)
    
    print(f"TPE best accuracy: {study_tpe.best_value:.4f}")
    

#### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

A sampler based on evolutionary strategies.
    
    
    from optuna.samplers import CmaEsSampler
    
    # CMA-ES sampler
    print("\n=== CMA-ES Sampler ===")
    study_cmaes = optuna.create_study(
        direction='maximize',
        sampler=CmaEsSampler(seed=42)
    )
    study_cmaes.optimize(objective, n_trials=30)
    
    print(f"CMA-ES best accuracy: {study_cmaes.best_value:.4f}")
    

### Pruning Strategies

By terminating unpromising trials early, computational time can be significantly reduced.
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: By terminating unpromising trials early, computational time 
    
    Purpose: Demonstrate optimization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from optuna.pruners import MedianPruner
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X, y = make_classification(n_samples=5000, n_features=50,
                              n_informative=30, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Objective function with pruning
    def objective_with_pruning(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
    
        # LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # Pruning callback
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')
    
        # Training
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[pruning_callback, lgb.log_evaluation(period=0)]
        )
    
        # Evaluation
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return accuracy
    
    # Study with Pruner
    print("\n=== Pruning Demo ===")
    study_pruning = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study_pruning.optimize(objective_with_pruning, n_trials=30, timeout=60)
    
    # Statistics
    n_complete = len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    print(f"\nBest accuracy: {study_pruning.best_value:.4f}")
    print(f"Completed trials: {n_complete}")
    print(f"Pruned trials: {n_pruned}")
    print(f"Pruning rate: {n_pruned/(n_complete+n_pruned)*100:.1f}%")
    

### Complete Optuna Example

A complete optimization example using Optuna.
    
    
    # Requirements:
    # - Python 3.9+
    # - optuna>=3.2.0
    
    """
    Example: A complete optimization example using Optuna.
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier
    import warnings
    warnings.filterwarnings('ignore')
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Objective function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
    
        model = GradientBoostingClassifier(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    
        return score
    
    # Create Study
    print("\n=== Complete Optuna Example ===")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name='breast_cancer_optimization'
    )
    
    # Execute optimization
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    # Display results
    print(f"\n=== Optimization Results ===")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Parameter importance
    print(f"\n=== Parameter Importance ===")
    importances = optuna.importance.get_param_importances(study)
    for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {importance:.4f}")
    
    # Visualizations are displayed in actual environment
    # plot_optimization_history(study).show()
    # plot_param_importances(study).show()
    # plot_parallel_coordinate(study).show()
    
    print("\n✓ Optimization complete")
    

* * *

## 2.3 Hyperopt

### Tree-structured Parzen Estimator (TPE)

**Hyperopt** is a Bayesian optimization framework using the TPE algorithm.
    
    
    # Requirements:
    # - Python 3.9+
    # - hyperopt>=0.2.7
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Hyperoptis a Bayesian optimization framework using the TPE a
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification
    import numpy as np
    
    # Prepare data
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # Define search space
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.uniform('max_features', 0.3, 1.0)
    }
    
    # Objective function (Hyperopt minimizes, so return negative accuracy)
    def objective(params):
        # Convert to integer types
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
    
        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
        return {'loss': -score, 'status': STATUS_OK}
    
    # Execute optimization
    print("=== Hyperopt TPE ===")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    print(f"\nBest parameters: {best}")
    print(f"Best accuracy: {-min([trial['result']['loss'] for trial in trials.trials]):.4f}")
    
    # Optimization history
    losses = [trial['result']['loss'] for trial in trials.trials]
    print(f"\nNumber of trials: {len(trials.trials)}")
    print(f"Best score progression:")
    best_so_far = []
    for i, loss in enumerate(losses):
        if i == 0:
            best_so_far.append(loss)
        else:
            best_so_far.append(min(best_so_far[-1], loss))
    print(f"  Start: {-best_so_far[0]:.4f}")
    print(f"  End: {-best_so_far[-1]:.4f}")
    print(f"  Improvement: {(-best_so_far[-1] + best_so_far[0]):.4f}")
    

### Search Space Definition

Hyperopt supports flexible search space definitions.
    
    
    # Requirements:
    # - Python 3.9+
    # - hyperopt>=0.2.7
    
    """
    Example: Hyperopt supports flexible search space definitions.
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from hyperopt import hp
    
    # Various distribution definitions
    search_space_detailed = {
        # Uniform distribution (continuous)
        'uniform_param': hp.uniform('uniform_param', 0.0, 1.0),
    
        # Uniform distribution (discrete)
        'quniform_param': hp.quniform('quniform_param', 10, 100, 5),  # 10, 15, 20, ...
    
        # Log uniform distribution
        'loguniform_param': hp.loguniform('loguniform_param', np.log(0.001), np.log(1.0)),
    
        # Normal distribution
        'normal_param': hp.normal('normal_param', 0, 1),
    
        # Categorical
        'choice_param': hp.choice('choice_param', ['option1', 'option2', 'option3']),
    
        # Conditional search space
        'classifier_type': hp.choice('classifier_type', [
            {
                'type': 'random_forest',
                'n_estimators': hp.quniform('rf_n_estimators', 50, 300, 1),
                'max_depth': hp.quniform('rf_max_depth', 3, 20, 1)
            },
            {
                'type': 'gradient_boosting',
                'n_estimators': hp.quniform('gb_n_estimators', 50, 300, 1),
                'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.01), np.log(0.3))
            }
        ])
    }
    
    print("=== Hyperopt Search Space Examples ===")
    for key, value in search_space_detailed.items():
        print(f"{key}: {value}")
    

### Trials Database

The Trials object stores all trial history.
    
    
    # Requirements:
    # - Python 3.9+
    # - hyperopt>=0.2.7
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: The Trials object stores all trial history.
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from hyperopt import Trials
    import pandas as pd
    
    # Detailed Trials analysis
    print("\n=== Trials Detailed Analysis ===")
    
    # Convert to DataFrame
    trials_df = pd.DataFrame([
        {
            'trial_id': i,
            'loss': trial['result']['loss'],
            **{k: v[0] if isinstance(v, (list, np.ndarray)) else v
               for k, v in trial['misc']['vals'].items() if v}
        }
        for i, trial in enumerate(trials.trials)
    ])
    
    print("\nTop 5 trials:")
    print(trials_df.nsmallest(5, 'loss')[['trial_id', 'loss', 'n_estimators', 'max_depth']])
    
    print("\nParameter statistics:")
    print(trials_df.describe())
    

### Hyperopt Integration

An example of integrating Hyperopt with machine learning libraries.
    
    
    # Requirements:
    # - Python 3.9+
    # - hyperopt>=0.2.7
    # - lightgbm>=4.0.0
    
    """
    Example: An example of integrating Hyperopt with machine learning lib
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Prepare data
    X, y = make_classification(n_samples=5000, n_features=50,
                              n_informative=30, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Search space for LightGBM
    lgb_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'num_leaves': hp.quniform('num_leaves', 20, 200, 1),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 10.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0),
    }
    
    # Objective function
    def lgb_objective(params):
        # Convert to integer types
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_samples'] = int(params['min_child_samples'])
    
        # LightGBM parameters
        lgb_params = {
            **params,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
        }
    
        # Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # Training
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
    
        # Evaluation
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return {'loss': -accuracy, 'status': STATUS_OK}
    
    # Optimization
    print("\n=== Hyperopt + LightGBM Integration ===")
    lgb_trials = Trials()
    best_lgb = fmin(
        fn=lgb_objective,
        space=lgb_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=lgb_trials,
        rstate=np.random.default_rng(42)
    )
    
    print(f"\nBest accuracy: {-min([trial['result']['loss'] for trial in lgb_trials.trials]):.4f}")
    print(f"Best parameters:")
    for key, value in best_lgb.items():
        print(f"  {key}: {value}")
    

* * *

## 2.4 Ray Tune

### Tune API

**Ray Tune** is a library for distributed hyperparameter optimization.
    
    
    # Requirements:
    # - Python 3.9+
    # - ray>=2.5.0
    
    """
    Example: Ray Tuneis a library for distributed hyperparameter optimiza
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # Prepare data
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, random_state=42)
    
    # Training function
    def train_model(config):
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=42
        )
    
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    
        # Report results to Ray Tune
        tune.report(accuracy=score)
    
    # Define search space
    config = {
        'n_estimators': tune.randint(50, 300),
        'max_depth': tune.randint(3, 20),
        'min_samples_split': tune.randint(2, 20)
    }
    
    # Execute optimization
    print("=== Ray Tune Basic Example ===")
    analysis = tune.run(
        train_model,
        config=config,
        num_samples=20,  # Number of trials
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    # Results
    best_config = analysis.get_best_config(metric='accuracy', mode='max')
    print(f"\nBest parameters: {best_config}")
    print(f"Best accuracy: {analysis.best_result['accuracy']:.4f}")
    

### Schedulers

#### ASHA (Async Successive Halving Algorithm)

ASHA terminates low-performing trials early and concentrates resources on promising trials.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - ray>=2.5.0
    
    """
    Example: ASHA terminates low-performing trials early and concentrates
    
    Purpose: Demonstrate simulation and statistical methods
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from ray.tune.schedulers import ASHAScheduler
    from ray import tune
    import numpy as np
    
    # Training function (iteration-aware)
    def train_with_iterations(config):
        # Simulation: performance improves with iterations
        base_score = np.random.rand()
    
        for iteration in range(config['max_iterations']):
            # Simulate learning curve
            score = base_score + (1 - base_score) * (1 - np.exp(-iteration / 20))
            score += np.random.randn() * 0.01  # Add noise
    
            # Report
            tune.report(accuracy=score, iteration=iteration)
    
    # ASHA scheduler
    asha_scheduler = ASHAScheduler(
        time_attr='iteration',
        metric='accuracy',
        mode='max',
        max_t=100,  # Maximum iterations
        grace_period=10,  # Minimum iterations to run
        reduction_factor=3  # Reduction factor
    )
    
    # Search space
    config_asha = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'max_iterations': 100
    }
    
    print("\n=== ASHA Scheduler ===")
    analysis_asha = tune.run(
        train_with_iterations,
        config=config_asha,
        num_samples=30,
        scheduler=asha_scheduler,
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    print(f"\nBest accuracy: {analysis_asha.best_result['accuracy']:.4f}")
    print(f"Number of completed trials: {len(analysis_asha.trials)}")
    

#### PBT (Population Based Training)

PBT dynamically adjusts hyperparameters using a population-based evolutionary approach.
    
    
    from ray.tune.schedulers import PopulationBasedTraining
    
    # PBT scheduler
    pbt_scheduler = PopulationBasedTraining(
        time_attr='iteration',
        metric='accuracy',
        mode='max',
        perturbation_interval=5,  # Perturbation interval
        hyperparam_mutations={
            'learning_rate': lambda: np.random.uniform(1e-4, 1e-1),
            'batch_size': [16, 32, 64, 128]
        }
    )
    
    print("\n=== PBT Scheduler ===")
    print("PBT dynamically adjusts hyperparameters")
    print("- Copy settings from high-performing models to others")
    print("- Add small variations to hyperparameters")
    print("- Progress optimization across the entire population")
    

### Integration with PyTorch/TensorFlow

Ray Tune integrates seamlessly with PyTorch and TensorFlow.
    
    
    # Requirements:
    # - Python 3.9+
    # - ray>=2.5.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Ray Tune integrates seamlessly with PyTorch and TensorFlow.
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Simple neural network
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Prepare data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Training function
    def train_pytorch(config):
        # Build model
        model = SimpleNet(
            input_size=20,
            hidden_size=config['hidden_size'],
            output_size=2
        )
    
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()
    
        # DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
    
        # Training loop
        for epoch in range(10):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
    
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
            accuracy = correct / total
            tune.report(accuracy=accuracy, loss=total_loss/len(train_loader))
    
    # Search space
    pytorch_config = {
        'hidden_size': tune.choice([32, 64, 128, 256]),
        'lr': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([16, 32, 64])
    }
    
    print("\n=== Ray Tune + PyTorch Integration ===")
    analysis_pytorch = tune.run(
        train_pytorch,
        config=pytorch_config,
        num_samples=10,
        resources_per_trial={'cpu': 1},
        verbose=1
    )
    
    print(f"\nBest accuracy: {analysis_pytorch.best_result['accuracy']:.4f}")
    print(f"Best configuration: {analysis_pytorch.get_best_config(metric='accuracy', mode='max')}")
    

### Distributed HPO

Ray Tune automatically distributes processing across multiple CPUs/GPUs.
    
    
    # Requirements:
    # - Python 3.9+
    # - ray>=2.5.0
    
    """
    Example: Ray Tune automatically distributes processing across multipl
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import ray
    
    # Initialize Ray (using multiple CPUs)
    ray.init(num_cpus=4, ignore_reinit_error=True)
    
    # Distributed execution configuration
    distributed_config = {
        'n_estimators': tune.randint(50, 300),
        'max_depth': tune.randint(3, 20),
        'min_samples_split': tune.randint(2, 20)
    }
    
    print("\n=== Distributed Hyperparameter Optimization ===")
    print("Parallel execution on 4 CPU cores")
    
    # Parallel execution
    analysis_distributed = tune.run(
        train_model,
        config=distributed_config,
        num_samples=40,
        resources_per_trial={'cpu': 1},  # 1 CPU per trial
        verbose=1
    )
    
    print(f"\nBest accuracy: {analysis_distributed.best_result['accuracy']:.4f}")
    print(f"Total trials: {len(analysis_distributed.trials)}")
    
    # Cleanup
    ray.shutdown()
    

* * *

## 2.5 Advanced HPO Techniques

### Bayesian Optimization

Bayesian optimization leverages past trial results to select the next search point.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Bayesian optimization leverages past trial results to select
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm
    import numpy as np
    
    class BayesianOptimizer:
        def __init__(self, bounds, n_init=5):
            self.bounds = bounds
            self.n_init = n_init
            self.X_obs = []
            self.y_obs = []
            self.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
    
        def acquisition_function(self, X, xi=0.01):
            """Expected Improvement (EI)"""
            mu, sigma = self.gp.predict(X, return_std=True)
    
            if len(self.y_obs) == 0:
                return np.zeros_like(mu)
    
            mu_best = np.max(self.y_obs)
    
            with np.errstate(divide='warn'):
                imp = mu - mu_best - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
    
            return ei
    
        def propose_location(self):
            """Propose next search point"""
            if len(self.X_obs) < self.n_init:
                # Random sampling
                return np.random.uniform(self.bounds[0], self.bounds[1])
    
            # Maximize Acquisition Function
            X_random = np.random.uniform(
                self.bounds[0], self.bounds[1], size=(1000, 1)
            )
            ei = self.acquisition_function(X_random)
            return X_random[np.argmax(ei)]
    
        def observe(self, X, y):
            """Record observation"""
            self.X_obs.append(X)
            self.y_obs.append(y)
    
            if len(self.X_obs) >= self.n_init:
                self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
    
    # Test function (optimization target)
    def test_function(x):
        """1D test function"""
        return -(x - 2) ** 2 + 5 + np.random.randn() * 0.1
    
    # Execute Bayesian optimization
    print("=== Bayesian Optimization Demo ===")
    optimizer = BayesianOptimizer(bounds=(0, 5), n_init=3)
    
    for i in range(20):
        # Propose next search point
        x_next = optimizer.propose_location()
    
        # Evaluate
        y_next = test_function(x_next[0])
    
        # Record observation
        optimizer.observe(x_next, y_next)
    
        if i % 5 == 0:
            print(f"Iteration {i}: x={x_next[0]:.3f}, y={y_next:.3f}")
    
    # Results
    best_idx = np.argmax(optimizer.y_obs)
    print(f"\nBest solution: x={optimizer.X_obs[best_idx][0]:.3f}, y={optimizer.y_obs[best_idx]:.3f}")
    print(f"True optimum: x=2.0, y=5.0")
    

### Population-based Training (PBT)

PBT simultaneously trains multiple models and shares good configurations.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from copy import deepcopy
    
    class PBTOptimizer:
        def __init__(self, population_size=10, perturbation_factor=0.2):
            self.population_size = population_size
            self.perturbation_factor = perturbation_factor
            self.population = []
    
        def initialize_population(self, param_ranges):
            """Initialize population"""
            for _ in range(self.population_size):
                individual = {
                    'params': {
                        key: np.random.uniform(low, high)
                        for key, (low, high) in param_ranges.items()
                    },
                    'score': 0.0,
                    'history': []
                }
                self.population.append(individual)
    
        def exploit_and_explore(self, param_ranges):
            """Exploit (copy good configurations) and Explore (perturbation)"""
            # Sort by performance
            self.population.sort(key=lambda x: x['score'], reverse=True)
    
            # Copy top performers to bottom 20%
            cutoff = int(0.2 * self.population_size)
            for i in range(self.population_size - cutoff, self.population_size):
                # Randomly select from top performers and copy
                source = np.random.randint(0, cutoff)
                self.population[i]['params'] = deepcopy(
                    self.population[source]['params']
                )
    
                # Add perturbation to parameters (Explore)
                for key in self.population[i]['params']:
                    low, high = param_ranges[key]
                    current = self.population[i]['params'][key]
    
                    # Random increase/decrease
                    factor = 1 + np.random.uniform(
                        -self.perturbation_factor,
                        self.perturbation_factor
                    )
                    new_value = current * factor
    
                    # Clip to range
                    self.population[i]['params'][key] = np.clip(
                        new_value, low, high
                    )
    
        def step(self, eval_fn, param_ranges):
            """Execute one step"""
            # Evaluate each individual
            for individual in self.population:
                score = eval_fn(individual['params'])
                individual['score'] = score
                individual['history'].append(score)
    
            # Exploit & Explore
            self.exploit_and_explore(param_ranges)
    
    # Evaluation function (simulation)
    def evaluate_params(params):
        """Evaluate parameters (simulation)"""
        # Optimal values: learning_rate=0.1, batch_size=32
        lr_score = 1 - abs(params['learning_rate'] - 0.1)
        bs_score = 1 - abs(params['batch_size'] - 32) / 64
        return (lr_score + bs_score) / 2 + np.random.randn() * 0.05
    
    # Execute PBT
    print("\n=== Population-based Training Demo ===")
    pbt = PBTOptimizer(population_size=10)
    param_ranges = {
        'learning_rate': (0.001, 0.3),
        'batch_size': (16, 128)
    }
    
    pbt.initialize_population(param_ranges)
    
    for step in range(20):
        pbt.step(evaluate_params, param_ranges)
    
        if step % 5 == 0:
            best = max(pbt.population, key=lambda x: x['score'])
            print(f"Step {step}: Best score={best['score']:.3f}, params={best['params']}")
    
    # Final results
    best_individual = max(pbt.population, key=lambda x: x['score'])
    print(f"\nBest parameters: {best_individual['params']}")
    print(f"Best score: {best_individual['score']:.3f}")
    

### Hyperband

Hyperband tries many configurations with various budgets (number of iterations).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import math
    
    class HyperbandOptimizer:
        def __init__(self, max_iter=81, eta=3):
            self.max_iter = max_iter
            self.eta = eta
            self.logeta = lambda x: math.log(x) / math.log(self.eta)
            self.s_max = int(self.logeta(self.max_iter))
            self.B = (self.s_max + 1) * self.max_iter
    
        def run(self, get_config_fn, eval_fn):
            """Execute Hyperband"""
            results = []
    
            for s in reversed(range(self.s_max + 1)):
                n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
                r = self.max_iter * self.eta ** (-s)
    
                print(f"\nBracket s={s}: n={n} configs, r={r:.1f} iterations")
    
                # Generate n configurations
                configs = [get_config_fn() for _ in range(n)]
    
                # Successive Halving
                for i in range(s + 1):
                    n_i = n * self.eta ** (-i)
                    r_i = r * self.eta ** i
    
                    print(f"  Round {i}: {int(n_i)} configs, {int(r_i)} iterations each")
    
                    # Evaluate
                    scores = [eval_fn(config, int(r_i)) for config in configs]
    
                    # Record results
                    for config, score in zip(configs, scores):
                        results.append({
                            'config': config,
                            'score': score,
                            'iterations': int(r_i)
                        })
    
                    # Select top performers
                    if i < s:
                        indices = np.argsort(scores)[-int(n_i / self.eta):]
                        configs = [configs[i] for i in indices]
    
            return results
    
    # Configuration generation function
    def get_random_config():
        return {
            'learning_rate': np.random.uniform(0.001, 0.3),
            'batch_size': np.random.choice([16, 32, 64, 128])
        }
    
    # Evaluation function (iteration-dependent)
    def evaluate_config(config, iterations):
        # Calculate performance based on distance from optimal
        lr_score = 1 - abs(config['learning_rate'] - 0.1)
        bs_score = 1 - abs(config['batch_size'] - 32) / 64
        base_score = (lr_score + bs_score) / 2
    
        # Performance improves with more iterations (learning curve)
        improvement = 1 - np.exp(-iterations / 20)
    
        return base_score * improvement + np.random.randn() * 0.01
    
    # Execute Hyperband
    print("=== Hyperband Demo ===")
    hyperband = HyperbandOptimizer(max_iter=81, eta=3)
    results = hyperband.run(get_random_config, evaluate_config)
    
    # Best result
    best_result = max(results, key=lambda x: x['score'])
    print(f"\n=== Best Result ===")
    print(f"Configuration: {best_result['config']}")
    print(f"Score: {best_result['score']:.4f}")
    print(f"Iterations: {best_result['iterations']}")
    print(f"\nTotal evaluations: {len(results)}")
    

### Multi-fidelity Optimization

Multi-fidelity optimization leverages low-cost approximate evaluations.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    class MultiFidelityOptimizer:
        def __init__(self, fidelity_levels=[0.1, 0.3, 0.5, 1.0]):
            self.fidelity_levels = fidelity_levels
            self.evaluations = {level: [] for level in fidelity_levels}
    
        def evaluate_at_fidelity(self, config, fidelity, true_fn):
            """Evaluate at specified fidelity"""
            # Low fidelity is fast but less accurate
            # High fidelity is slow but more accurate
    
            true_score = true_fn(config)
            noise = (1 - fidelity) * 0.2  # More noise at lower fidelity
    
            observed_score = true_score + np.random.randn() * noise
    
            return observed_score
    
        def optimize(self, param_ranges, true_fn, n_total_evals=100):
            """Multi-fidelity optimization"""
            # Budget allocation: many at low fidelity, few at high fidelity
            eval_counts = {
                0.1: int(0.5 * n_total_evals),
                0.3: int(0.3 * n_total_evals),
                0.5: int(0.15 * n_total_evals),
                1.0: int(0.05 * n_total_evals)
            }
    
            all_configs = []
    
            # Evaluate at each fidelity level
            for fidelity in self.fidelity_levels:
                n_evals = eval_counts[fidelity]
    
                if fidelity == self.fidelity_levels[0]:
                    # Lowest fidelity: random sampling
                    configs = [
                        {key: np.random.uniform(low, high)
                         for key, (low, high) in param_ranges.items()}
                        for _ in range(n_evals)
                    ]
                else:
                    # Top performers from previous fidelity evaluated at next fidelity
                    prev_results = sorted(
                        self.evaluations[self.fidelity_levels[self.fidelity_levels.index(fidelity) - 1]],
                        key=lambda x: x['score'],
                        reverse=True
                    )
                    configs = [r['config'] for r in prev_results[:n_evals]]
    
                # Evaluate
                for config in configs:
                    score = self.evaluate_at_fidelity(config, fidelity, true_fn)
                    self.evaluations[fidelity].append({
                        'config': config,
                        'score': score,
                        'fidelity': fidelity
                    })
                    all_configs.append((config, score, fidelity))
    
            # Return best result at highest fidelity
            best = max(
                self.evaluations[1.0],
                key=lambda x: x['score']
            )
    
            return best, all_configs
    
    # True objective function
    def true_objective(config):
        lr_score = 1 - abs(config['learning_rate'] - 0.1)
        bs_score = 1 - abs(config['batch_size'] - 32) / 64
        return (lr_score + bs_score) / 2
    
    # Execute multi-fidelity optimization
    print("\n=== Multi-fidelity Optimization Demo ===")
    mf_optimizer = MultiFidelityOptimizer()
    param_ranges = {
        'learning_rate': (0.001, 0.3),
        'batch_size': (16, 128)
    }
    
    best, all_evals = mf_optimizer.optimize(param_ranges, true_objective, n_total_evals=100)
    
    print(f"\nBest configuration: {best['config']}")
    print(f"Best score: {best['score']:.4f}")
    print(f"\nEvaluations by fidelity level:")
    for fidelity in mf_optimizer.fidelity_levels:
        print(f"  Fidelity {fidelity}: {len(mf_optimizer.evaluations[fidelity])} evaluations")
    

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **HPO Fundamentals**

     * Defining search spaces and choosing distributions
     * Comparison of Grid Search and Random Search
     * Efficiency improvement with Early Stopping
  2. **Optuna**

     * Flexible implementation with Define-by-run API
     * Advanced samplers like TPE and CMA-ES
     * Significant time savings with Pruning
     * Visualization and diagnostic features
  3. **Hyperopt**

     * TPE-based Bayesian optimization
     * Flexible search space definition
     * Detailed history management with Trials
  4. **Ray Tune**

     * Distributed hyperparameter optimization
     * Advanced schedulers like ASHA and PBT
     * Integration with PyTorch/TensorFlow
     * Scalable parallel execution
  5. **Advanced HPO Techniques**

     * Bayesian Optimization: Leveraging past trials
     * Population-based Training: Dynamic configuration adjustment
     * Hyperband: Search with diverse budgets
     * Multi-fidelity: Leveraging low-cost evaluations

### HPO Framework Selection Guide

Framework | Best Use Case | Strengths | Weaknesses  
---|---|---|---  
**Optuna** | General-purpose HPO, research | Flexible, feature-rich, visualization | Limited distributed capability  
**Hyperopt** | Medium-scale HPO, complex search spaces | Mature, stable | Somewhat dated design  
**Ray Tune** | Large-scale distributed HPO, DL | Scalable, integration | Complex configuration  
**scikit-learn** | Simple HPO | Easy, standard | Limited functionality  
  
### HPO Strategy Selection Criteria

Situation | Recommended Strategy | Reason  
---|---|---  
Small search (<10 parameters) | Grid Search | Exhaustive and easy to understand  
Medium search (10-20 parameters) | Random Search, TPE | Efficient and practical  
Large search (>20 parameters) | Bayesian Opt, ASHA | Efficient even in high dimensions  
Expensive evaluation function | Bayesian Opt | Optimize with few trials  
Cheap evaluation function | Random Search, Hyperband | Many trials possible  
Distributed environment available | Ray Tune + ASHA/PBT | Speed up with parallelization  
  
### Next Chapter

In Chapter 3, we will learn about **Neural Architecture Search (NAS)** :

  * Fundamentals and motivation for NAS
  * Search space design
  * DARTS, ENAS, NASNet
  * Efficient NAS methods
  * Practical NAS implementation

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the main differences between Grid Search and Random Search, and describe the strengths and weaknesses of each.

Sample Answer

**Answer** :

**Grid Search** :

  * Search method: Exhaustively tries all combinations
  * Strengths: Completely searches the space, high reproducibility
  * Weaknesses: Combinatorial explosion with more parameters, high computational cost

**Random Search** :

  * Search method: Randomly samples parameters
  * Strengths: Efficient even in high dimensions, easier to find important parameters
  * Weaknesses: No guarantee of optimal solution, difficult to determine number of trials

**When to use which** :

Situation | Recommendation  
---|---  
Few parameters (<5) | Grid Search  
Many parameters (>5) | Random Search  
Abundant computational resources | Grid Search  
Limited computational resources | Random Search  
  
### Problem 2 (Difficulty: medium)

Use Optuna to implement hyperparameter optimization for LightGBM. Enable Pruning and optimize at least 5 hyperparameters.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - optuna>=3.2.0
    
    """
    Example: Use Optuna to implement hyperparameter optimization for Ligh
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    from optuna.pruners import MedianPruner
    import lightgbm as lgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }
    
        # Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
        # Pruning callback
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, 'binary_logloss'
        )
    
        # Training
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[pruning_callback, lgb.log_evaluation(period=0)]
        )
    
        # Evaluation
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, (preds > 0.5).astype(int))
    
        return accuracy
    
    # Create Study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Execute optimization
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Results
    print(f"\n=== Optimization Results ===")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Statistics
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nComplete: {n_complete}, Pruned: {n_pruned}, Pruning rate: {n_pruned/(n_complete+n_pruned)*100:.1f}%")
    

### Problem 3 (Difficulty: medium)

Explain the role of "Acquisition Function" in Bayesian optimization. Also, describe the differences between Expected Improvement (EI) and Upper Confidence Bound (UCB).

Sample Answer

**Answer** :

**Role of Acquisition Function** :

  * Purpose: Determine the next point to evaluate
  * Function: Balance exploration and exploitation
  * Input: Current Gaussian process model (predicted mean and variance)
  * Output: "Utility" score for each candidate point

**Expected Improvement (EI)** :

  * Definition: Expected value of improvement over current best
  * Formula: $EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$
  * Characteristics: Emphasizes certain improvement, conservative
  * Strengths: Stable convergence, theoretical guarantees
  * Weaknesses: Prone to local optima

**Upper Confidence Bound (UCB)** :

  * Definition: Predicted mean + uncertainty bonus
  * Formula: $UCB(x) = \mu(x) + \kappa \sigma(x)$
  * Characteristics: Explores regions with high uncertainty
  * Strengths: Promotes exploration, global optimization
  * Weaknesses: Requires tuning parameter $\kappa$

**When to use which** :

Situation | Recommendation  
---|---  
Expensive evaluation, emphasize certainty | EI  
Emphasize exploration, global optimization | UCB  
High noise | EI  
Smooth objective function | Either works  
  
### Problem 4 (Difficulty: hard)

Explain how the ASHA scheduler works and describe why it is more efficient compared to regular Random Search.

Sample Answer

**Answer** :

**How ASHA Works** :

  1. **Basic Idea**

     * Try many configurations with limited resources (iterations)
     * Terminate poorly performing configurations early
     * Concentrate resources on promising configurations
  2. **Algorithm**

     * Set rungs (stages): e.g., [10, 30, 90, 270] iterations
     * At each rung, only top 1/η (e.g., top 1/3 if η=3) proceed to next
     * Eventually, only a few configurations run to maximum iterations
  3. **Asynchronous Execution**

     * Each trial progresses independently
     * Promotion decision when reaching a rung
     * Efficient resource utilization

**Comparison with Random Search** :

Aspect | Random Search | ASHA  
---|---|---  
Resource allocation | Equal to all trials | Concentrated on promising trials  
Early stopping | None | Yes (terminates poor performers)  
Parallelization | Simple | Efficient asynchronously  
Total computation time | N × max_iter | ≈ N × min_iter + few × max_iter  
  
**Reasons for Efficiency** :

  1. **Reducing Wasted Computation**

     * Terminate obviously bad configurations early
     * Only fully train promising configurations
  2. **Balance of Exploration and Exploitation**

     * Try diverse configurations (exploration)
     * Focus on good configurations (exploitation)
  3. **Theoretical Guarantees**

     * Low probability of missing optimal configuration
     * Logarithmic growth in computation

**Example** :
    
    
    Random Search: 81 trials × 100 iterations = 8,100 compute units
    
    ASHA (η=3):
    - Rung 0: 81 trials × 1 iteration = 81
    - Rung 1: 27 trials × 3 iterations = 81
    - Rung 2: 9 trials × 9 iterations = 81
    - Rung 3: 3 trials × 27 iterations = 81
    - Rung 4: 1 trial × 81 iterations = 81
    Total: 405 compute units
    
    Reduction rate: (8,100 - 405) / 8,100 = 95%
    

### Problem 5 (Difficulty: hard)

Explain the main differences between Population-based Training (PBT) and Bayesian optimization, and describe situations where each is appropriate.

Sample Answer

**Answer** :

**PBT Characteristics** :

  * Train multiple models simultaneously
  * Dynamically adjust hyperparameters during training
  * Copy good configurations to other models (Exploit)
  * Add perturbations to copied configurations (Explore)
  * Evolutionary approach (similar to genetic algorithms)

**Bayesian Optimization Characteristics** :

  * Train models sequentially one at a time
  * Hyperparameters fixed before training
  * Model with Gaussian process from past trial history
  * Probabilistically select next optimal search point
  * Theoretical optimization approach

**Main Differences** :

Aspect | PBT | Bayesian Optimization  
---|---|---  
**Parallelism** | High (train entire population simultaneously) | Limited (sequential execution basic)  
**Dynamic adjustment** | Yes (change during training) | No (fixed before training)  
**Computational efficiency** | High (parallel execution) | Medium (fewer trials)  
**Theoretical guarantees** | Weak (heuristic) | Strong (convergence guarantees)  
**Implementation complexity** | High (population management needed) | Medium (existing implementations available)  
**Application scope** | Long training (DL) | Expensive evaluation functions  
  
**Appropriate Situations** :

**When to use PBT** :

  * Long training like deep learning
  * Dynamic adjustments like learning rate schedules are important
  * Abundant parallel computing resources
  * Wide search space
  * Examples: Reinforcement learning, large-scale neural network training

**When to use Bayesian Optimization** :

  * Single evaluation very expensive
  * Smooth evaluation function
  * Want to optimize with few trials
  * Need theoretical guarantees
  * Examples: Simulation optimization, expensive experimental setups

**Hybrid Approach** :

  * Explore initial configurations with Bayesian optimization
  * Fine-tune around good configurations with PBT
  * Leverage advantages of both

* * *

## References

  1. Bergstra, J., & Bengio, Y. (2012). _Random search for hyper-parameter optimization_. Journal of Machine Learning Research, 13(1), 281-305.
  2. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). _Optuna: A next-generation hyperparameter optimization framework_. KDD.
  3. Bergstra, J., Yamins, D., & Cox, D. (2013). _Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures_. ICML.
  4. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). _Hyperband: A novel bandit-based approach to hyperparameter optimization_. JMLR.
  5. Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Kavukcuoglu, K. (2017). _Population based training of neural networks_. arXiv preprint arXiv:1711.09846.
  6. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2015). _Taking the human out of the loop: A review of Bayesian optimization_. Proceedings of the IEEE, 104(1), 148-175.
  7. Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J. E., & Stoica, I. (2018). _Tune: A research platform for distributed model selection and training_. arXiv preprint arXiv:1807.05118.
