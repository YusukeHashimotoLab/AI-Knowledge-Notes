---
title: "Chapter 1: Fundamentals of Ensemble Learning"
chapter_title: "Chapter 1: Fundamentals of Ensemble Learning"
subtitle: Improving Prediction Accuracy through Model Combination - Principles of Bagging, Boosting, and Stacking
reading_time: 20-25 minutes
difficulty: Beginner to Intermediate
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers the fundamentals of Fundamentals of Ensemble Learning, which what is ensemble learning?. You will learn principles of ensemble learning, concept of bias-variance decomposition, and bagging (Random Forest.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the principles of ensemble learning
  * ✅ Explain the concept of bias-variance decomposition
  * ✅ Implement bagging (Random Forest, Extra Trees)
  * ✅ Understand the mechanisms of boosting (AdaBoost, Gradient Boosting)
  * ✅ Master stacking and meta-learners
  * ✅ Determine when to use each method

* * *

## 1.1 What is Ensemble Learning?

### Definition

**Ensemble Learning** is a machine learning approach that combines multiple weak learners to construct a more powerful prediction model.

> "Combining multiple models achieves higher performance than a single model"

### Why Combine Multiple Models?
    
    
    ```mermaid
    graph LR
        A[Single Model Limitations] --> B[Prone to overfitting]
        A --> C[High bias]
        A --> D[Sensitive to noise]
    
        E[Ensemble] --> F[Reduce variance]
        E --> G[Reduce bias]
        E --> H[Improve stability]
    
        style A fill:#ffebee
        style E fill:#e8f5e9
    ```

### Effectiveness of Ensembles

**Example** : When three models each predict independently with 70% accuracy

Accuracy through majority voting:

$$ P(\text{correct}) = P(\text{2 or more correct}) = \binom{3}{2}(0.7)^2(0.3) + \binom{3}{3}(0.7)^3 = 0.784 $$

Achieves higher accuracy (78.4%) than a single model (70%)!

### Bias-Variance Decomposition

Prediction error can be decomposed as follows:

$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

Component | Meaning | Solution  
---|---|---  
**Bias** | Error due to model simplification | Complex models, boosting  
**Variance** | Sensitivity to training data variation | Bagging, averaging  
**Irreducible Error** | Noise inherent in data | Cannot be reduced  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: $$
    \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Ir
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    
    # Generate data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Predict with decision trees of different depths
    depths = [1, 3, 10]
    plt.figure(figsize=(15, 4))
    
    for i, depth in enumerate(depths, 1):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
    
        X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_pred = model.predict(X_plot)
    
        plt.subplot(1, 3, i)
        plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
        plt.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Depth={depth}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Depth={depth}: {"High Bias" if depth==1 else "High Variance" if depth==10 else "Balanced"}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.2 Bagging

### Overview

**Bagging (Bootstrap Aggregating)** is a method that trains multiple models using bootstrap sampling of training data, then averages predictions (regression) or uses majority voting (classification).

### Algorithm

  1. Generate $B$ **bootstrap samples** from training data
  2. Train models independently on each sample
  3. Aggregate predictions: 
     * Regression: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)$
     * Classification: Majority voting

    
    
    ```mermaid
    graph TD
        A[Training Data] --> B1[Bootstrap 1]
        A --> B2[Bootstrap 2]
        A --> B3[Bootstrap 3]
    
        B1 --> M1[Model 1]
        B2 --> M2[Model 2]
        B3 --> M3[Model 3]
    
        M1 --> AGG[Aggregation]
        M2 --> AGG
        M3 --> AGG
    
        AGG --> PRED[Final Prediction]
    
        style A fill:#e3f2fd
        style AGG fill:#fff3e0
        style PRED fill:#e8f5e9
    ```

### Random Forest

**Random Forest** is a method that combines bagging with random feature selection.

**Features** :

  * Selects optimal split from a randomly chosen subset of features at each split
  * Reduces correlation between models and improves diversity

    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Single decision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print("=== Effect of Bagging ===")
    print(f"Decision Tree (single): {dt_acc:.4f}")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Improvement: {(rf_acc - dt_acc):.4f}")
    

**Output** :
    
    
    === Effect of Bagging ===
    Decision Tree (single): 0.8600
    Random Forest: 0.9250
    Improvement: 0.0650
    

### Extra Trees

**Extra Trees (Extremely Randomized Trees)** is an even more randomized version of Random Forest.

**Differences** :

  * Split thresholds are also chosen randomly
  * Does not use bootstrap sampling (uses all data)

    
    
    from sklearn.ensemble import ExtraTreesClassifier
    
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_train, y_train)
    et_acc = accuracy_score(y_test, et.predict(X_test))
    
    print("=== Extra Trees vs Random Forest ===")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Extra Trees: {et_acc:.4f}")
    

* * *

## 1.3 Boosting

### Overview

**Boosting** is a method that sequentially trains weak learners, with each subsequent model correcting the errors of previous models.
    
    
    ```mermaid
    graph LR
        A[Data] --> M1[Model 1]
        M1 --> W1[Update Weights]
        W1 --> M2[Model 2]
        M2 --> W2[Update Weights]
        W2 --> M3[Model 3]
        M3 --> F[Weighted Sum]
    
        style A fill:#e3f2fd
        style F fill:#e8f5e9
    ```

### AdaBoost

**AdaBoost (Adaptive Boosting)** sequentially trains models while increasing the weights of misclassified samples.

**Algorithm** :

  1. Initialize weights for all samples: $w_i = \frac{1}{m}$
  2. For each iteration $t = 1, ..., T$: 
     * Train weak learner $h_t$ on weighted data
     * Error rate: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
     * Model weight: $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
     * Update sample weights
  3. Final prediction: $H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$

    
    
    from sklearn.ensemble import AdaBoostClassifier
    
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada.predict(X_test))
    
    print("=== AdaBoost ===")
    print(f"Accuracy: {ada_acc:.4f}")
    
    # Accuracy progression with iterations
    from sklearn.metrics import accuracy_score
    
    n_trees = [1, 5, 10, 25, 50, 100]
    train_scores = []
    test_scores = []
    
    for n in n_trees:
        ada_temp = AdaBoostClassifier(n_estimators=n, random_state=42)
        ada_temp.fit(X_train, y_train)
        train_scores.append(ada_temp.score(X_train, y_train))
        test_scores.append(ada_temp.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, train_scores, 'o-', label='Training data', linewidth=2)
    plt.plot(n_trees, test_scores, 's-', label='Test data', linewidth=2)
    plt.xlabel('Number of weak learners', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('AdaBoost: Relationship between Number of Learners and Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### Gradient Boosting Fundamentals

**Gradient Boosting** is a method that adds models in the direction of the gradient of the loss function.

**Algorithm** :

  1. Initial prediction: $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{m} L(y_i, \gamma)$
  2. For each iteration $t = 1, ..., T$: 
     * Compute residuals (negative gradient): $r_i = -\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$
     * Train weak learner $h_t$ on residuals
     * Update model: $F_t(x) = F_{t-1}(x) + \nu \cdot h_t(x)$

    
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    
    print("=== Gradient Boosting ===")
    print(f"Accuracy: {gb_acc:.4f}")
    

* * *

## 1.4 Stacking

### Overview

**Stacking** is a method where a meta-learner makes final predictions using the predictions of multiple different models (base models) as input.
    
    
    ```mermaid
    graph TD
        A[Training Data] --> M1[Model 1: Logistic Regression]
        A --> M2[Model 2: Random Forest]
        A --> M3[Model 3: SVM]
    
        M1 --> P1[Prediction 1]
        M2 --> P2[Prediction 2]
        M3 --> P3[Prediction 3]
    
        P1 --> META[Meta-learner]
        P2 --> META
        P3 --> META
    
        META --> FINAL[Final Prediction]
    
        style A fill:#e3f2fd
        style META fill:#fff3e0
        style FINAL fill:#e8f5e9
    ```

### Meta-learner

The meta-learner learns using the predictions of base models as features.

**Common meta-learners** :

  * Logistic Regression
  * Ridge Regression
  * Neural Networks

### Cross-Validation Strategy

To prevent overfitting, **K-Fold cross-validation** is used to generate base model predictions.
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Base models
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=50, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
    ]
    
    # Stacking
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    stack.fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stack.predict(X_test))
    
    print("=== Stacking ===")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Extra Trees: {et_acc:.4f}")
    print(f"AdaBoost: {ada_acc:.4f}")
    print(f"Stacking: {stack_acc:.4f}")
    

* * *

## 1.5 Comparison and Selection

### Performance Comparison

Method | Variance Reduction | Bias Reduction | Parallelization | Training Speed  
---|---|---|---|---  
**Bagging** | ✓ | - | Possible | Fast  
**Random Forest** | ✓✓ | - | Possible | Fast  
**AdaBoost** | - | ✓ | Not possible | Moderate  
**Gradient Boosting** | - | ✓✓ | Not possible | Slow  
**Stacking** | ✓ | ✓ | Possible | Slow  
  
### Application Scenarios

Situation | Recommended Method | Reason  
---|---|---  
**High Variance Models** | Bagging, Random Forest | Effectively reduces variance  
**High Bias Models** | Boosting | Learns complex patterns  
**Large-scale Data** | Random Forest | Parallelizable and fast  
**Imbalanced Data** | AdaBoost | Focuses on misclassified samples  
**Pursuing Best Performance** | Stacking, GB | Integrates strengths of multiple methods  
      
    
    # Comparison of all methods
    results = {
        'Decision Tree': dt_acc,
        'Random Forest': rf_acc,
        'Extra Trees': et_acc,
        'AdaBoost': ada_acc,
        'Gradient Boosting': gb_acc,
        'Stacking': stack_acc
    }
    
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(methods, accuracies, color=['#e74c3c', '#3498db', '#2ecc71',
                                               '#f39c12', '#9b59b6', '#1abc9c'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Performance Comparison of Ensemble Methods', fontsize=14)
    plt.xticks(rotation=15, ha='right')
    plt.ylim([0.8, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Display values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Final Results ===")
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:20s}: {acc:.4f}")
    

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Principles of Ensemble Learning**

     * Improved performance through model combination
     * Bias-variance decomposition
  2. **Bagging**

     * Bootstrap sampling and averaging
     * Random Forest: Improved diversity through random feature selection
     * Extra Trees: Further randomization
  3. **Boosting**

     * AdaBoost: Focuses on misclassified samples
     * Gradient Boosting: Optimization in gradient direction
  4. **Stacking**

     * Integration through meta-learner
     * Overfitting prevention through cross-validation
  5. **Selection Criteria**

     * Bagging: Variance reduction, parallelization
     * Boosting: Bias reduction, high accuracy
     * Stacking: Pursuing best performance

### To the Next Chapter

In Chapter 2, we will learn about **advanced gradient boosting** :

  * XGBoost
  * LightGBM
  * CatBoost
  * Hyperparameter tuning

* * *

## Practice Problems

### Problem 1 (Difficulty: easy)

Explain the difference between bias and variance, and name the ensemble methods that reduce each.

Sample Answer

**Bias** :

  * Error due to model simplification
  * High error even on training data
  * Example: Predicting non-linear data with a linear model

**Variance** :

  * Sensitivity to training data variation
  * Good on training data but poor on test data
  * Example: Overfitting with deep decision trees

**Reduction Methods** :

  * **Variance Reduction** : Bagging, Random Forest (reduces variance through averaging)
  * **Bias Reduction** : Boosting (learns complex patterns sequentially)

### Problem 2 (Difficulty: medium)

Name two differences between Random Forest and Extra Trees, and explain the characteristics of each.

Sample Answer

**Difference 1: Sampling**

  * **Random Forest** : Bootstrap sampling (sampling with replacement)
  * **Extra Trees** : Uses all data (no sampling)

**Difference 2: Splitting Method**

  * **Random Forest** : Selects optimal split from random feature subset
  * **Extra Trees** : Randomly selects both features and thresholds

**Characteristics** :

  * **Random Forest** : Variance reduction, takes some time to train
  * **Extra Trees** : Faster, improved diversity through further randomization

### Problem 3 (Difficulty: medium)

Explain from an algorithmic perspective why the weights of misclassified samples increase in AdaBoost.

Sample Answer

**Reason** :

  * AdaBoost is designed so that each iteration **focuses on samples the previous model struggled with**
  * By increasing the weights of misclassified samples, the next model tries to classify them correctly

**Algorithm** :
    
    
    Weight update for misclassified sample i:
    w_i ← w_i * exp(α_t)
    
    Weight update for correctly classified sample j:
    w_j ← w_j * exp(-α_t)
    
    where α_t = 0.5 * ln((1 - ε_t) / ε_t) > 0
    

**Effect** :

  * Weak learners sequentially learn difficult samples
  * Eventually forms complex decision boundaries

### Problem 4 (Difficulty: hard)

Explain from an overfitting perspective why K-Fold cross-validation is used in stacking.

Sample Answer

**Problem** : If base models are trained on the entire training data and predictions are generated on the same data:

  * Meta-learner overfits to training data
  * Base model predictions are optimized for "previously seen data"

**K-Fold Cross-Validation Solution** :
    
    
    1. Split data into K folds
    2. For each fold k:
       - Train base models on all folds except fold k
       - Generate predictions for fold k (predictions on unseen data)
    3. Combine predictions from all folds to train meta-learner
    

**Effect** :

  * Input to meta-learner is "predictions on unseen data"
  * Improved generalization performance
  * Prevents overfitting

**Implementation Example** :
    
    
    StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5  # 5-Fold cross-validation
    )
    

### Problem 5 (Difficulty: hard)

Implement and compare Random Forest and Gradient Boosting using the iris dataset. Report training time and accuracy.

Sample Answer
    
    
    import time
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest
    print("=== Random Forest ===")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    
    rf_acc = rf.score(X_test, y_test)
    cv_rf = cross_val_score(rf, X, y, cv=5).mean()
    
    print(f"Training time: {rf_time:.4f} seconds")
    print(f"Test accuracy: {rf_acc:.4f}")
    print(f"Cross-validation accuracy: {cv_rf:.4f}")
    
    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    start = time.time()
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_time = time.time() - start
    
    gb_acc = gb.score(X_test, y_test)
    cv_gb = cross_val_score(gb, X, y, cv=5).mean()
    
    print(f"Training time: {gb_time:.4f} seconds")
    print(f"Test accuracy: {gb_acc:.4f}")
    print(f"Cross-validation accuracy: {cv_gb:.4f}")
    
    # Comparison
    print("\n=== Comparison ===")
    print(f"Accuracy: RF={rf_acc:.4f} vs GB={gb_acc:.4f}")
    print(f"Training time: RF={rf_time:.4f}s vs GB={gb_time:.4f}s")
    print(f"Speed ratio: GB/RF = {gb_time/rf_time:.2f}x")
    

**Example Output** :
    
    
    === Random Forest ===
    Training time: 0.0523 seconds
    Test accuracy: 1.0000
    Cross-validation accuracy: 0.9533
    
    === Gradient Boosting ===
    Training time: 0.1245 seconds
    Test accuracy: 1.0000
    Cross-validation accuracy: 0.9467
    
    === Comparison ===
    Accuracy: RF=1.0000 vs GB=1.0000
    Training time: RF=0.0523s vs GB=0.1245s
    Speed ratio: GB/RF = 2.38x
    

**Analysis** :

  * Accuracy is nearly equivalent
  * Random Forest trains faster (parallelizable)
  * Both methods achieve high accuracy on this small, simple dataset

* * *

## References

  1. Breiman, L. (1996). _Bagging predictors_. Machine Learning, 24(2), 123-140.
  2. Breiman, L. (2001). _Random forests_. Machine Learning, 45(1), 5-32.
  3. Freund, Y., & Schapire, R. E. (1997). _A decision-theoretic generalization of on-line learning and an application to boosting_. Journal of Computer and System Sciences, 55(1), 119-139.
  4. Friedman, J. H. (2001). _Greedy function approximation: A gradient boosting machine_. Annals of Statistics, 1189-1232.
