---
title: "Chapter 3: LightGBM and CatBoost"
chapter_title: "Chapter 3: LightGBM and CatBoost"
subtitle: Next-Generation Gradient Boosting - Acceleration and Categorical Variable Handling
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 9
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers LightGBM and CatBoost. You will learn Implementing LightGBM, Understanding CatBoost's Ordered Boosting, and Implementing CatBoost.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understanding LightGBM's acceleration techniques (GOSS, EFB, Histogram-based)
  * ✅ Implementing LightGBM and performing parameter tuning
  * ✅ Understanding CatBoost's Ordered Boosting and Categorical Features processing
  * ✅ Implementing CatBoost and selecting encoding strategies
  * ✅ Comparing characteristics of XGBoost, LightGBM, and CatBoost to make informed choices

* * *

## 3.1 LightGBM - Acceleration Mechanisms

### What is LightGBM?

**LightGBM (Light Gradient Boosting Machine)** is a fast and efficient gradient boosting framework developed by Microsoft.

> As its "Light" name suggests, it is lighter and faster than XGBoost, making it suitable for large-scale datasets.

### Key Technical Innovations

#### 1\. Histogram-based Algorithm

Significantly reduces computational complexity by discretizing (binning) continuous values.

Method | Complexity | Memory | Accuracy  
---|---|---|---  
**Pre-sorted** (XGBoost) | $O(n \log n)$ | High | High  
**Histogram-based** (LightGBM) | $O(n \times k)$ | Low | Nearly equivalent  
  
$k$: Number of bins (typically 255), $n$: Number of data points
    
    
    ```mermaid
    graph LR
        A[Continuous Value Data] --> B[Histogramming]
        B --> C[Discretize into 255 bins]
        C --> D[Fast Split Search]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#f3e5f5
        style D fill:#c8e6c9
    ```

#### 2\. GOSS (Gradient-based One-Side Sampling)

**GOSS** accelerates training by emphasizing data with large gradients and sampling data with small gradients.

Algorithm:

  1. Sort data by absolute gradient value
  2. Keep all top $a\%$ (large gradients)
  3. Randomly sample $b\%$ from remaining $(1-a)\%$
  4. Adjust weight of sampled data by $(1-a)/b$

#### 3\. EFB (Exclusive Feature Bundling)

**EFB** reduces dimensionality by bundling mutually exclusive features (never non-zero simultaneously).

Example: One-Hot Encoded features
    
    
    color_red:   [1, 0, 0, 1, 0]
    color_blue:  [0, 1, 0, 0, 1]
    color_green: [0, 0, 1, 0, 0]
    → Can be merged into a single feature
    

### Leaf-wise vs Level-wise Growth Strategy

Strategy | Description | Used By | Advantages | Disadvantages  
---|---|---|---|---  
**Level-wise** | Depth-first splitting of all nodes | XGBoost | Balanced trees | Splits even low-gain nodes  
**Leaf-wise** | Split leaf with maximum gain | LightGBM | Efficient, high accuracy | Prone to overfitting  
      
    
    ```mermaid
    graph TD
        A[Level-wise: XGBoost] --> B1[Level 1: Split all]
        B1 --> C1[Level 2: Split all]
    
        D[Leaf-wise: LightGBM] --> E1[Split only max gain node]
        E1 --> F1[Split next max gain node]
    
        style A fill:#e3f2fd
        style D fill:#f3e5f5
    ```

* * *

## 3.2 LightGBM Implementation

### Basic Usage
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Basic Usage
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import lightgbm as lgb
    
    # Generate data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build LightGBM model
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("=== LightGBM Basic Implementation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    

**Output** :
    
    
    === LightGBM Basic Implementation ===
    Accuracy: 0.9350
    AUC: 0.9712
    

### Important Parameters

Parameter | Description | Recommended Values  
---|---|---  
`num_leaves` | Maximum number of leaves in tree | 31-255 (default: 31)  
`max_depth` | Maximum tree depth (overfitting control) | 3-10 (default: -1=unlimited)  
`learning_rate` | Learning rate | 0.01-0.1  
`n_estimators` | Number of trees | 100-1000  
`min_child_samples` | Minimum samples per leaf | 20-100  
`subsample` | Data sampling ratio | 0.7-1.0  
`colsample_bytree` | Feature sampling ratio | 0.7-1.0  
`reg_alpha` | L1 regularization | 0-1  
`reg_lambda` | L2 regularization | 0-1  
  
### Early Stopping and Validation
    
    
    # Further split training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train with early stopping
    model_early = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )
    
    model_early.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    print(f"\n=== Early Stopping ===")
    print(f"Optimal iteration count: {model_early.best_iteration_}")
    print(f"Validation AUC: {model_early.best_score_['valid_0']['auc']:.4f}")
    
    # Evaluate on test data
    y_pred_early = model_early.predict(X_test)
    accuracy_early = accuracy_score(y_test, y_pred_early)
    print(f"Test Accuracy: {accuracy_early:.4f}")
    

### Feature Importance Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Feature Importance Visualization
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance Top 10 ===")
    print(importance_df.head(10))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain')
    plt.title('LightGBM Feature Importance (Gain)', fontsize=14)
    plt.tight_layout()
    plt.show()
    

### GPU Support
    
    
    # GPU usage (CUDA environment required)
    model_gpu = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        device='gpu',  # Use GPU
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=42
    )
    
    # Train (accelerated on GPU)
    # model_gpu.fit(X_train, y_train)
    
    print("\n=== GPU Support ===")
    print("LightGBM supports GPU (CUDA), enabling 10-30x speedup on large datasets")
    print("Enable with device='gpu' parameter")
    

> **Note** : To use the GPU version, LightGBM must be built with GPU support.

* * *

## 3.3 CatBoost - Ordered Boosting and Categorical Variable Handling

### What is CatBoost?

**CatBoost (Categorical Boosting)** is a gradient boosting framework developed by Yandex, featuring automatic handling of categorical variables.

### Key Technical Innovations

#### 1\. Ordered Boosting

**Ordered Boosting** is a technique to prevent prediction shift.

**Problem** : Traditional boosting calculates gradients and trains on the same data, making overfitting likely.

**Solution** :

  1. Randomly permute the data
  2. For each sample $i$, use only samples $1, ..., i-1$ for prediction
  3. Build multiple models with different orderings

    
    
    ```mermaid
    graph LR
        A[Traditional Boosting] --> B[Train on all data]
        B --> C[Predict on same data]
        C --> D[Prediction shift occurs]
    
        E[Ordered Boosting] --> F[Train only on past data]
        F --> G[Predict on future data]
        G --> H[Prevent prediction shift]
    
        style D fill:#ffebee
        style H fill:#c8e6c9
    ```

#### 2\. Automatic Processing of Categorical Features

CatBoost automatically encodes categorical variables.

**Target Statistics** calculation:

$$ \text{TS}(x_i) = \frac{\sum_{j=1}^{i-1} \mathbb{1}_{x_j = x_i} \cdot y_j + a \cdot P}{\sum_{j=1}^{i-1} \mathbb{1}_{x_j = x_i} + a} $$ 

  * $x_i$: Category value
  * $y_j$: Target value
  * $a$: Smoothing parameter
  * $P$: Prior probability

This approach offers the following advantages:

  * No need for One-Hot Encoding
  * Handles high cardinality (many categories)
  * Prevents target leakage

### Symmetric Trees (Oblivious Trees)

CatBoost uses **symmetric trees** (Oblivious Decision Trees).

Characteristic | Regular Decision Tree | Symmetric Tree (CatBoost)  
---|---|---  
Split Condition | Different at each node | Same condition at same level  
Structure | Asymmetric | Perfectly symmetric  
Overfitting | Prone | Resistant  
Prediction Speed | Normal | Very fast  
  
* * *

## 3.4 CatBoost Implementation

### Basic Usage
    
    
    # Requirements:
    # - Python 3.9+
    # - catboost>=1.2.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Basic Usage
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from catboost import CatBoostClassifier
    
    # Generate data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build CatBoost model
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        random_seed=42,
        verbose=0
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("=== CatBoost Basic Implementation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    

**Output** :
    
    
    === CatBoost Basic Implementation ===
    Accuracy: 0.9365
    AUC: 0.9721
    

### Handling Categorical Variables
    
    
    # Generate dataset with categorical variables
    np.random.seed(42)
    n = 5000
    
    df = pd.DataFrame({
        'num_feature1': np.random.randn(n),
        'num_feature2': np.random.uniform(0, 100, n),
        'cat_feature1': np.random.choice(['A', 'B', 'C', 'D'], n),
        'cat_feature2': np.random.choice(['Low', 'Medium', 'High'], n),
        'cat_feature3': np.random.choice([f'Cat_{i}' for i in range(50)], n)  # High cardinality
    })
    
    # Target variable (depends on categories)
    df['target'] = (
        (df['cat_feature1'].isin(['A', 'B'])) &
        (df['num_feature1'] > 0) &
        (df['num_feature2'] > 50)
    ).astype(int)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Specify categorical variable columns
    cat_features = ['cat_feature1', 'cat_feature2', 'cat_feature3']
    
    # Split train/test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== Data with Categorical Variables ===")
    print(f"Data shape: {X.shape}")
    print(f"Categorical variables: {cat_features}")
    print(f"\nUnique counts for each category:")
    for col in cat_features:
        print(f"  {col}: {X[col].nunique()}")
    
    # Train with CatBoost (automatic categorical processing)
    model_cat = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,  # Specify categorical variables
        random_seed=42,
        verbose=0
    )
    
    model_cat.fit(X_train, y_train)
    
    # Evaluate
    y_pred_cat = model_cat.predict(X_test)
    y_proba_cat = model_cat.predict_proba(X_test)[:, 1]
    
    accuracy_cat = accuracy_score(y_test, y_pred_cat)
    auc_cat = roc_auc_score(y_test, y_proba_cat)
    
    print(f"\n=== Categorical Variable Processing Results ===")
    print(f"Accuracy: {accuracy_cat:.4f}")
    print(f"AUC: {auc_cat:.4f}")
    print("✓ Handles high cardinality without One-Hot Encoding")
    

### Encoding Strategies

CatBoost supports multiple encoding modes:

Mode | Description | Use Case  
---|---|---  
`Ordered` | Ordered Target Statistics | Prevent overfitting (default)  
`GreedyLogSum` | Greedy log sum | Large-scale data  
`OneHot` | One-Hot Encoding | Low cardinality (≤10)  
      
    
    # Requirements:
    # - Python 3.9+
    # - catboost>=1.2.0
    
    """
    Example: CatBoost supports multiple encoding modes:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Compare encoding strategies
    from catboost import Pool
    
    # Create CatBoost Pool (efficient data structure)
    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )
    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_features
    )
    
    # Different encoding strategies
    strategies = {
        'Ordered': 'Ordered',
        'GreedyLogSum': 'GreedyLogSum',
        'OneHot': {'one_hot_max_size': 10}  # One-Hot for cardinality≤10
    }
    
    print("\n=== Encoding Strategy Comparison ===")
    for name, strategy in strategies.items():
        model_strategy = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features,
            random_seed=42,
            verbose=0
        )
    
        if name == 'OneHot':
            model_strategy.set_params(**strategy)
    
        model_strategy.fit(train_pool)
        y_pred = model_strategy.predict(test_pool)
        accuracy = accuracy_score(y_test, y_pred)
    
        print(f"{name:15s}: Accuracy = {accuracy:.4f}")
    

### Early Stopping and Validation
    
    
    # Further split training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train with early stopping
    model_early = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100
    )
    
    model_early.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    print(f"\n=== Early Stopping ===")
    print(f"Optimal iteration count: {model_early.get_best_iteration()}")
    print(f"Best score: {model_early.get_best_score()}")
    
    # Evaluate on test data
    y_pred_early = model_early.predict(X_test)
    accuracy_early = accuracy_score(y_test, y_pred_early)
    print(f"Test Accuracy: {accuracy_early:.4f}")
    

* * *

## 3.5 Comparison of XGBoost, LightGBM, and CatBoost

### Algorithm Characteristics Comparison

Characteristic | XGBoost | LightGBM | CatBoost  
---|---|---|---  
**Developer** | Tianqi Chen (DMLC) | Microsoft | Yandex  
**Splitting Algorithm** | Pre-sorted | Histogram-based | Histogram-based  
**Tree Growth Strategy** | Level-wise | Leaf-wise | Level-wise (symmetric trees)  
**Speed** | Normal | Fast | Somewhat slow  
**Memory Efficiency** | Normal | High efficiency | Normal  
**Categorical Processing** | Manual encoding required | Manual encoding required | Automatic processing  
**Overfitting Resistance** | High | Medium (caution with Leaf-wise) | Very high  
**GPU Support** | Yes | Yes | Yes  
**Hyperparameter Tuning** | Somewhat complex | Somewhat complex | Simple  
  
### Performance Comparison Experiment
    
    
    # Requirements:
    # - Python 3.9+
    # - catboost>=1.2.0
    # - lightgbm>=4.0.0
    # - xgboost>=2.0.0
    
    """
    Example: Performance Comparison Experiment
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import time
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Generate large-scale data
    X_large, y_large = make_classification(
        n_samples=50000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    
    X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(
        X_large, y_large, test_size=0.2, random_state=42
    )
    
    # Common parameters
    common_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    # Define models
    models = {
        'XGBoost': XGBClassifier(**common_params, verbosity=0),
        'LightGBM': LGBMClassifier(**common_params, verbose=-1),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0
        )
    }
    
    print("=== Performance Comparison (50,000 samples, 50 features) ===\n")
    results = []
    
    for name, model in models.items():
        # Measure training time
        start_time = time.time()
        model.fit(X_train_lg, y_train_lg)
        train_time = time.time() - start_time
    
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test_lg)
        y_proba = model.predict_proba(X_test_lg)[:, 1]
        pred_time = time.time() - start_time
    
        # Evaluate
        accuracy = accuracy_score(y_test_lg, y_pred)
        auc = roc_auc_score(y_test_lg, y_proba)
    
        results.append({
            'Model': name,
            'Train Time (s)': train_time,
            'Predict Time (s)': pred_time,
            'Accuracy': accuracy,
            'AUC': auc
        })
    
        print(f"{name}:")
        print(f"  Training time: {train_time:.3f} seconds")
        print(f"  Prediction time: {pred_time:.3f} seconds")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}\n")
    
    # Display results as DataFrame
    results_df = pd.DataFrame(results)
    print("=== Results Summary ===")
    print(results_df.to_string(index=False))
    

### Memory Usage Comparison
    
    
    import sys
    
    print("\n=== Memory Usage Estimation ===")
    for name, model in models.items():
        # Model memory size (approximate)
        model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
        print(f"{name:10s}: Approx. {model_size:.2f} MB")
    
    print("\nCharacteristics:")
    print("• LightGBM: Minimum memory through histogramming")
    print("• XGBoost: Medium memory with pre-sorted method")
    print("• CatBoost: Compact with symmetric trees")
    

### Usage Guidelines

Situation | Recommended | Reason  
---|---|---  
**Large-scale data ( >1M rows)** | LightGBM | Fastest, low memory  
**Many categorical variables** | CatBoost | Automatic processing, high accuracy  
**High cardinality** | CatBoost | Target Statistics  
**Overfitting concerns** | CatBoost | Ordered Boosting  
**Balanced performance** | XGBoost | Stable, proven track record  
**Speed priority** | LightGBM | Leaf-wise + Histogram  
**Accuracy priority** | CatBoost | Overfitting resistance  
**Limited tuning time** | CatBoost | Good default performance  
**GPU acceleration** | All supported | Choose based on environment  
  
### Practical Selection Flowchart
    
    
    ```mermaid
    graph TD
        A[Gradient boosting needed] --> B{Many categorical variables?}
        B -->|Yes| C[CatBoost]
        B -->|No| D{Data size?}
        D -->|Large-scale >1M rows| E[LightGBM]
        D -->|Small-medium scale| F{What to prioritize?}
        F -->|Speed| E
        F -->|Accuracy| C
        F -->|Balance| G[XGBoost]
    
        style C fill:#c8e6c9
        style E fill:#fff9c4
        style G fill:#e1bee7
    ```

* * *

## 3.6 Chapter Summary

### What We Learned

  1. **LightGBM Technical Innovations**

     * Histogram-based Algorithm: Reduced computational complexity
     * GOSS: Gradient-based sampling
     * EFB: Bundling exclusive features
     * Leaf-wise growth: Efficient tree construction
  2. **LightGBM Implementation**

     * Fast and efficient training
     * Further acceleration through GPU support
     * Flexible tuning with abundant parameters
  3. **CatBoost Technical Innovations**

     * Ordered Boosting: Prevent prediction shift
     * Automatic categorical variable processing
     * Symmetric trees: Overfitting resistance and fast prediction
  4. **CatBoost Implementation**

     * Direct handling of categorical variables
     * High cardinality support
     * High performance with default parameters
  5. **Comparison of Three Tools**

     * XGBoost: Balance and proven track record
     * LightGBM: Speed and memory efficiency
     * CatBoost: Categorical processing and accuracy

### Selection Points

Priority Item | First Choice | Second Choice  
---|---|---  
Training Speed | LightGBM | XGBoost  
Prediction Accuracy | CatBoost | XGBoost  
Memory Efficiency | LightGBM | CatBoost  
Categorical Processing | CatBoost | -  
Ease of Tuning | CatBoost | XGBoost  
Stability | XGBoost | CatBoost  
  
### Next Steps

  * Automatic hyperparameter tuning (Optuna, Hyperopt)
  * Combining ensemble methods (stacking, blending)
  * Integration with feature engineering
  * Improving model interpretability (SHAP, LIME)

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain each of the three main acceleration techniques in LightGBM (Histogram-based, GOSS, EFB).

Solution

**Answer** :

  1. **Histogram-based Algorithm**

     * Description: Discretizes continuous values into a fixed number of bins (typically 255)
     * Effect: Reduces complexity from $O(n \log n)$ to $O(n \times k)$
     * Advantages: Improved memory efficiency, faster split search
  2. **GOSS (Gradient-based One-Side Sampling)**

     * Description: Prioritizes data with large gradients
     * Procedure: Keep all top $a\%$ gradients + sample $b\%$ from remainder
     * Advantages: Acceleration through data reduction, maintains accuracy
  3. **EFB (Exclusive Feature Bundling)**

     * Description: Bundles exclusive features (never non-zero simultaneously)
     * Example: Merge One-Hot Encoded variables into one
     * Advantages: Acceleration through feature reduction

### Problem 2 (Difficulty: medium)

Explain the differences between Level-wise (XGBoost) and Leaf-wise (LightGBM) tree growth strategies, and describe their respective advantages and disadvantages.

Solution

**Answer** :

**Level-wise** :

  * Strategy: Depth-first, split all nodes at same level
  * Advantages: Balanced trees, less prone to overfitting
  * Disadvantages: Inefficient as it splits even low-gain nodes
  * Used By: XGBoost, CatBoost

**Leaf-wise** :

  * Strategy: Split only the leaf with maximum gain
  * Advantages: Efficient, high accuracy, fast
  * Disadvantages: Prone to overfitting (depth limit important)
  * Used By: LightGBM

**Comparison Table** :

Aspect | Level-wise | Leaf-wise  
---|---|---  
Efficiency | Normal | High  
Accuracy | Stable | High but watch overfitting  
Tree Shape | Symmetric | Asymmetric  
Overfitting Resistance | High | Medium (depth limit needed)  
  
### Problem 3 (Difficulty: medium)

Explain why CatBoost's Ordered Boosting can prevent prediction shift.

Solution

**Answer** :

**Prediction Shift Problem** :

Traditional boosting has the following issues:

  1. Calculate gradients on all data
  2. Train next weak learner on same data
  3. Overfit to training data (seeing same data for prediction and training)
  4. Performance degradation on test data

**Ordered Boosting Solution** :

  1. **Data Ordering** : Randomly permute the data
  2. **Use Only Past Data** : For sample $i$, use only samples $1, ..., i-1$
  3. **Validate on Future Data** : Don't predict on data used for training
  4. **Multiple Models** : Build multiple models with different orderings and average

**Effects** :

  * Same conditions for training and testing (predict only from past data)
  * Prevention of prediction shift
  * Improved generalization performance
  * Overfitting suppression

**Formula** :

Prediction $\hat{y}_i$ for sample $i$ is:

$$ \hat{y}_i = M(\\{(x_j, y_j)\\}_{j=1}^{i-1}) $$ 

That is, use model $M$ trained only on data before $i$.

### Problem 4 (Difficulty: hard)

Train LightGBM and CatBoost on the following data and compare their performance. Pay attention to differences in categorical variable processing methods.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Train LightGBM and CatBoost on the following data and compar
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n = 10000
    
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'cat2': np.random.choice([f'Cat_{i}' for i in range(100)], n),  # High cardinality
        'target': np.random.choice([0, 1], n)
    })
    

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - catboost>=1.2.0
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Train LightGBM and CatBoost on the following data and compar
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    # Generate data
    np.random.seed(42)
    n = 10000
    
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'cat2': np.random.choice([f'Cat_{i}' for i in range(100)], n),
    })
    
    # Generate target (depends on categories)
    df['target'] = (
        (df['cat1'].isin(['A', 'B'])) &
        (df['num1'] > 0)
    ).astype(int)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== Data Information ===")
    print(f"Sample count: {n}")
    print(f"Categorical variables:")
    print(f"  cat1: {X['cat1'].nunique()} unique values")
    print(f"  cat2: {X['cat2'].nunique()} unique values (high cardinality)")
    
    # ===== LightGBM: Requires Label Encoding =====
    print("\n=== LightGBM (Using Label Encoding) ===")
    
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    
    # Label Encoding
    le_cat1 = LabelEncoder()
    le_cat2 = LabelEncoder()
    
    X_train_lgb['cat1'] = le_cat1.fit_transform(X_train_lgb['cat1'])
    X_test_lgb['cat1'] = le_cat1.transform(X_test_lgb['cat1'])
    
    X_train_lgb['cat2'] = le_cat2.fit_transform(X_train_lgb['cat2'])
    # Handle unknown categories in test data
    X_test_lgb['cat2'] = X_test_lgb['cat2'].map(
        {v: k for k, v in enumerate(le_cat2.classes_)}
    ).fillna(-1).astype(int)
    
    model_lgb = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    model_lgb.fit(X_train_lgb, y_train)
    y_pred_lgb = model_lgb.predict(X_test_lgb)
    y_proba_lgb = model_lgb.predict_proba(X_test_lgb)[:, 1]
    
    acc_lgb = accuracy_score(y_test, y_pred_lgb)
    auc_lgb = roc_auc_score(y_test, y_proba_lgb)
    
    print(f"Accuracy: {acc_lgb:.4f}")
    print(f"AUC: {auc_lgb:.4f}")
    print("Processing: Numerization via Label Encoding (no ordinal information)")
    
    # ===== CatBoost: Can handle categorical variables directly =====
    print("\n=== CatBoost (Automatic Categorical Processing) ===")
    
    cat_features = ['cat1', 'cat2']
    
    model_cat = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        verbose=0
    )
    
    model_cat.fit(X_train, y_train)
    y_pred_cat = model_cat.predict(X_test)
    y_proba_cat = model_cat.predict_proba(X_test)[:, 1]
    
    acc_cat = accuracy_score(y_test, y_pred_cat)
    auc_cat = roc_auc_score(y_test, y_proba_cat)
    
    print(f"Accuracy: {acc_cat:.4f}")
    print(f"AUC: {auc_cat:.4f}")
    print("Processing: Automatic encoding via Target Statistics")
    
    # ===== Comparison =====
    print("\n=== Comparison Results ===")
    comparison = pd.DataFrame({
        'Model': ['LightGBM', 'CatBoost'],
        'Accuracy': [acc_lgb, acc_cat],
        'AUC': [auc_lgb, auc_cat],
        'Categorical Handling': ['Manual (Label Encoding)', 'Automatic (Target Statistics)']
    })
    print(comparison.to_string(index=False))
    
    print("\n=== Discussion ===")
    print("• LightGBM: Label Encoding with no ordinal information (suboptimal)")
    print("• CatBoost: Meaningful encoding via Target Statistics")
    print("• CatBoost advantageous for high cardinality")
    print("• One-Hot Encoding impractical due to dimension explosion (100 categories)")
    

### Problem 5 (Difficulty: hard)

For each of the following situations, choose the most optimal among XGBoost, LightGBM, and CatBoost, and explain your reasoning:

  1. Dataset with 100 million rows and 100 features
  2. 5 high-cardinality variables with 100 categories each
  3. Small dataset (10,000 rows) where you want to maximize accuracy

Solution

**Answer** :

**1\. Dataset with 100 million rows and 100 features**

  * **Recommended** : LightGBM
  * **Reason** : 
    * Fastest with Histogram-based Algorithm
    * Further acceleration through GOSS data sampling
    * Best memory efficiency (essential for large-scale data)
    * Can be further accelerated with GPU support
  * **Alternative** : XGBoost (GPU mode) is an option but slower than LightGBM

**2\. 5 high-cardinality variables with 100 categories each**

  * **Recommended** : CatBoost
  * **Reason** : 
    * Automatically processes categorical variables (Target Statistics)
    * No need for One-Hot Encoding (avoids 100 categories×5 = 500 dimension explosion)
    * Prevents target leakage with Ordered Boosting
    * Design optimized for high cardinality
  * **Issues with Other Options** : 
    * XGBoost: Label Encoding has meaningless ordinal information, One-Hot causes dimension explosion
    * LightGBM: Same issues

**3\. Small dataset (10,000 rows) to maximize accuracy**

  * **Recommended** : CatBoost
  * **Reason** : 
    * Ordered Boosting prevents overfitting and provides high generalization performance
    * Proven accuracy on small datasets
    * Excellent default parameters (reduces tuning time)
    * Stable learning with symmetric trees
    * Speed is not an issue (small data)
  * **Alternative** : XGBoost (good balance and stability)
  * **LightGBM Issue** : Leaf-wise strategy prone to overfitting on small datasets

**Summary Table** :

Situation | First Choice | Second Choice | Key Factor  
---|---|---|---  
Large-scale data | LightGBM | XGBoost (GPU) | Speed, memory  
High cardinality | CatBoost | - | Automatic categorical processing  
Small-scale, high accuracy | CatBoost | XGBoost | Overfitting resistance  
  
* * *

## References

  1. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." _Advances in Neural Information Processing Systems_ 30.
  2. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." _Advances in Neural Information Processing Systems_ 31.
  3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." _Proceedings of the 22nd ACM SIGKDD_.
  4. Microsoft LightGBM Documentation: <https://lightgbm.readthedocs.io/>
  5. Yandex CatBoost Documentation: <https://catboost.ai/docs/>
