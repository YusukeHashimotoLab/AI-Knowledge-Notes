---
title: "Chapter 2: SHAP (SHapley Additive exPlanations)"
chapter_title: "Chapter 2: SHAP (SHapley Additive exPlanations)"
subtitle: Unified Feature Importance Based on Game Theory
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
---

This chapter covers SHAP (SHapley Additive exPlanations). You will learn mathematical formulation, algorithms such as TreeSHAP, and and visualize using the SHAP library.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the concept and properties of Shapley values in game theory
  * ✅ Explain the mathematical formulation and axiomatic properties of SHAP values
  * ✅ Understand algorithms such as TreeSHAP and KernelSHAP
  * ✅ Implement and visualize using the SHAP library
  * ✅ Use interpretation methods such as Waterfall, Force, and Summary plots
  * ✅ Understand the application scope and limitations of SHAP

* * *

## 2.1 Theory of Shapley Values

### Foundations of Game Theory

**Shapley values** are a concept from cooperative game theory that fairly evaluates the contribution of players.

#### Definition of Cooperative Games

A cooperative game $(N, v)$ is defined as follows:

  * $N = \\{1, 2, \ldots, n\\}$: Set of players
  * $v: 2^N \rightarrow \mathbb{R}$: Characteristic function (coalitional value function)
  * $v(S)$: Value of coalition $S \subseteq N$

Conditions:

  * $v(\emptyset) = 0$ (value of empty set is 0)
  * $v(N)$: Total value when all players cooperate

#### Application to Machine Learning

In machine learning interpretation problems:

  * **Players** : Features
  * **Value** : Prediction
  * **Coalition** : Subset of features

    
    
    ```mermaid
    graph TB
        GameTheory["Cooperative Game TheoryN: Players, v: Value function"] --> Shapley["Shapley ValuesFair contribution distribution"]
        Shapley --> ML["Machine Learning InterpretationFeature contributions"]
        ML --> SHAP["SHAPUnified interpretation framework"]
    
        style GameTheory fill:#b3e5fc
        style Shapley fill:#c5e1a5
        style ML fill:#fff9c4
        style SHAP fill:#ffab91
    ```

### Definition of Shapley Values

The **Shapley value** of player $i$ is defined as the average of contributions to all possible coalitions:

$$ \phi_i(v) = \sum_{S \subseteq N \setminus \\{i\\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \\{i\\}) - v(S) \right] $$ 

Where:

  * $S$: Coalition not containing player $i$
  * $v(S \cup \\{i\\}) - v(S)$: Marginal contribution of player $i$
  * $\frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!}$: Weight (considering all orderings)

### Properties (Axioms) of Shapley Values

Shapley values are the **unique solution** satisfying the following four axioms:

Axiom | Mathematical Expression | Meaning  
---|---|---  
**Efficiency** | $\sum_{i=1}^{n} \phi_i(v) = v(N)$ | Sum of all players' contributions = Total value  
**Symmetry** | $v(S \cup \\{i\\}) = v(S \cup \\{j\\})$ implies $\phi_i = \phi_j$ | Same contribution gets same reward  
**Dummy** | $v(S \cup \\{i\\}) = v(S)$ implies $\phi_i = 0$ | No contribution means zero reward  
**Additivity** | $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$ | Independent games are decomposable  
  
### Computational Complexity

Exact computation of Shapley values requires evaluating $2^n$ coalitions, resulting in **exponential time complexity** $O(2^n)$.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from itertools import combinations
    
    def shapley_value_exact(n, value_function):
        """
        Exact computation of Shapley values (only for small number of features)
    
        Args:
            n: Number of features
            value_function: Function returning value for subset S
    
        Returns:
            shapley_values: (n,) Shapley values
        """
        players = list(range(n))
        shapley_values = np.zeros(n)
    
        # For each player
        for i in players:
            # Other players excluding player i
            others = [p for p in players if p != i]
    
            # For all possible coalitions S
            for r in range(len(others) + 1):
                for S in combinations(others, r):
                    S = list(S)
    
                    # Marginal contribution: v(S ∪ {i}) - v(S)
                    marginal_contribution = value_function(S + [i]) - value_function(S)
    
                    # Weight: |S|! * (n - |S| - 1)! / n!
                    weight = (np.math.factorial(len(S)) *
                             np.math.factorial(n - len(S) - 1) /
                             np.math.factorial(n))
    
                    shapley_values[i] += weight * marginal_contribution
    
        return shapley_values
    
    
    # Simple example: 3 features
    print("=== Exact Computation of Shapley Values ===")
    n = 3
    
    # Example value function (linear model)
    def value_func(S):
        """Prediction value for feature subset S"""
        # Simplified: feature weights are [1, 2, 3]
        weights = np.array([1.0, 2.0, 3.0])
        if len(S) == 0:
            return 0.0
        return weights[S].sum()
    
    # Compute Shapley values
    shapley = shapley_value_exact(n, value_func)
    
    print(f"Number of features: {n}")
    print(f"Shapley values: {shapley}")
    print(f"Sum: {shapley.sum()} (= v(N) = {value_func([0, 1, 2])})")
    print("\n→ Satisfies efficiency axiom: sum = total value")
    

### Computational Complexity Problem
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Computational Complexity Problem
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Visualization of computational complexity
    n_features = np.arange(1, 21)
    num_coalitions = 2 ** n_features
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(n_features, num_coalitions, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Number of Coalitions (log scale)', fontsize=12)
    plt.title('Computational Complexity of Exact Shapley Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Reference lines
    plt.axhline(y=1e6, color='r', linestyle='--', label='1 million')
    plt.axhline(y=1e9, color='orange', linestyle='--', label='1 billion')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('shapley_complexity.png', dpi=150, bbox_inches='tight')
    print("Saved complexity figure: shapley_complexity.png")
    plt.close()
    
    print("\n=== Computational Complexity Examples ===")
    for n in [5, 10, 15, 20, 30]:
        coalitions = 2 ** n
        print(f"Features {n:2d}: {coalitions:,} coalitions")
    
    print("\n→ Computation becomes infeasible as features increase")
    print("→ Approximation algorithms (KernelSHAP, TreeSHAP, etc.) are necessary")
    

* * *

## 2.2 SHAP Values

### Additive Feature Attribution

**SHAP (SHapley Additive exPlanations)** is a framework that applies Shapley values to machine learning interpretation.

The explanation model $g$ for prediction $f(x)$ has the following form:

$$ g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i $$ 

Where:

  * $z' \in \\{0, 1\\}^M$: Simplified input (presence/absence of features)
  * $\phi_i$: SHAP value (contribution) of feature $i$
  * $\phi_0 = \mathbb{E}[f(X)]$: Base value (overall expectation)

### Definition of SHAP Values

The SHAP value of feature $i$ for input $x$ is:

$$ \phi_i(f, x) = \sum_{S \subseteq F \setminus \\{i\\}} \frac{|S|! \cdot (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \\{i\\}) - f_x(S) \right] $$ 

Where:

  * $F = \\{1, 2, \ldots, M\\}$: Set of all features
  * $f_x(S)$: Prediction using only feature subset $S$ (others replaced with expected values)

### Computing SHAP Values

In practice, missing features are computed by **conditioning on expected values** :

$$ f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S] $$ 

Different SHAP algorithms exist depending on how this expectation is computed.
    
    
    ```mermaid
    graph TB
        SHAP["SHAPUnified Framework"] --> TreeSHAP["TreeSHAPDecision tree modelsPolynomial time"]
        SHAP --> KernelSHAP["KernelSHAPAny modelSampling approximation"]
        SHAP --> DeepSHAP["DeepSHAPDeep learningGradient-based"]
        SHAP --> LinearSHAP["LinearSHAPLinear modelsAnalytical computation"]
    
        style SHAP fill:#ffab91
        style TreeSHAP fill:#c5e1a5
        style KernelSHAP fill:#fff9c4
        style DeepSHAP fill:#b3e5fc
    ```

### TreeSHAP Algorithm

**TreeSHAP** is an efficient algorithm for tree-based models (decision trees, random forests, XGBoost, LightGBM, etc.).

Features:

  * Complexity: $O(TLD^2)$ ($T$: number of trees, $L$: number of leaves, $D$: depth)
  * Computes exact Shapley values
  * Accelerated using tree structure

### KernelSHAP

**KernelSHAP** is an approximation algorithm applicable to any model.

Idea:

  1. Randomly sample feature subsets
  2. Compute predictions for each subset
  3. Estimate SHAP values via weighted linear regression

Weight function:

$$ \pi_{x}(z') = \frac{(M-1)}{\binom{M}{|z'|} |z'|(M - |z'|)} $$ 

* * *

## 2.3 SHAP Visualization

### Waterfall Plots

**Waterfall plots** display the contribution of each feature from the base value to the final prediction value for a single sample, like a waterfall.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: Waterfall plotsdisplay the contribution of each feature from
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    
    # Load dataset
    print("=== Preparing Breast Cancer Dataset ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    print(f"Data size: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Train model
    print("\n=== Training Random Forest ===")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)
    
    train_acc = model.score(X, y)
    print(f"Training accuracy: {train_acc:.4f}")
    
    # Create TreeExplainer
    print("\n=== Creating TreeExplainer ===")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for one sample
    sample_idx = 0
    shap_values = explainer(X.iloc[[sample_idx]])
    
    print(f"SHAP values for sample {sample_idx}:")
    print(f"  Shape: {shap_values.values.shape}")
    print(f"  Base value: {shap_values.base_values[0]:.4f}")
    print(f"  Prediction: {shap_values.base_values[0] + shap_values.values[0].sum():.4f}")
    
    # Create waterfall plot
    print("\n=== Creating Waterfall Plot ===")
    shap.plots.waterfall(shap_values[0], show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
    print("Saved waterfall plot: shap_waterfall.png")
    plt.close()
    

### Force Plots

**Force plots** visualize single-sample explanations similar to waterfall plots, but display positive and negative contributions side by side.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - shap>=0.42.0
    
    """
    Example: Force plotsvisualize single-sample explanations similar to w
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Creating Force Plot ===")
    
    # Force plot (static version)
    shap.plots.force(
        shap_values.base_values[0],
        shap_values.values[0],
        X.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_force.png', dpi=150, bbox_inches='tight')
    print("Saved force plot: shap_force.png")
    plt.close()
    
    print("\n→ Red: Features increasing prediction")
    print("→ Blue: Features decreasing prediction")
    

### Summary Plots

**Summary plots** aggregate SHAP values across all samples to visualize the importance and effects of each feature.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - shap>=0.42.0
    
    """
    Example: Summary plotsaggregate SHAP values across all samples to vis
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Creating Summary Plot ===")
    
    # Compute SHAP values for all samples (use subset if time-consuming)
    shap_values_all = explainer(X[:100])  # First 100 samples
    
    # Summary plot (bee swarm plot)
    shap.summary_plot(shap_values_all, X[:100], show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved summary plot: shap_summary.png")
    plt.close()
    
    print("\n→ Each point is one sample")
    print("→ X-axis: SHAP value (contribution)")
    print("→ Color: Feature value (red=high, blue=low)")
    

### Bar Plots (Feature Importance)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - shap>=0.42.0
    
    """
    Example: Bar Plots (Feature Importance)
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Bar Plot (Feature Importance) ===")
    
    # Sort by mean absolute SHAP value
    shap.plots.bar(shap_values_all, show=False)
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
    print("Saved bar plot: shap_bar.png")
    plt.close()
    
    # Manual computation
    mean_abs_shap = np.abs(shap_values_all.values).mean(axis=0)
    feature_importance = sorted(
        zip(X.columns, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nFeature importance (top 10):")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
    

### Dependence Plots

**Dependence plots** show the relationship between a feature's value and its SHAP value in a scatter plot.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - shap>=0.42.0
    
    """
    Example: Dependence plotsshow the relationship between a feature's va
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Creating Dependence Plot ===")
    
    # Select the most important feature
    top_feature = feature_importance[0][0]
    print(f"Analysis target: {top_feature}")
    
    # Dependence plot
    shap.dependence_plot(
        top_feature,
        shap_values_all.values,
        X[:100],
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
    print("Saved dependence plot: shap_dependence.png")
    plt.close()
    
    print("\n→ X-axis: Feature value")
    print("→ Y-axis: SHAP value (contribution of that feature)")
    print("→ Can discover non-linear relationships and interactions")
    

* * *

## 2.4 Practical Use of SHAP Library

### TreeExplainer (Tree-based Models)

**TreeExplainer** is the most efficient explainer for tree-based models (RandomForest, XGBoost, LightGBM, CatBoost, etc.).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    # - xgboost>=2.0.0
    
    """
    Example: TreeExplaineris the most efficient explainer for tree-based 
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import xgboost as xgb
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Data preparation (Boston Housing - regression task)
    print("=== Boston Housing Dataset ===")
    # Note: load_boston is deprecated, using California housing as alternative
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    
    print(f"Data size: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    print("\n=== Training XGBoost Model ===")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    
    # Using TreeExplainer
    print("\n=== Using TreeExplainer ===")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer(X_test)
    
    print(f"SHAP values shape: {shap_values.values.shape}")
    print(f"Base value: {shap_values.base_values[0]:.4f}")
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig('xgboost_shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved summary plot: xgboost_shap_summary.png")
    plt.close()
    

### LinearExplainer (Linear Models)

**LinearExplainer** is an analytical explainer for linear models (linear regression, logistic regression, etc.).
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: LinearExplaineris an analytical explainer for linear models 
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("\n=== LinearExplainer (Logistic Regression) ===")
    
    # Iris dataset
    X, y = load_iris(return_X_y=True, as_frame=True)
    # Simplify to binary classification
    X = X[y != 2]
    y = y[y != 2]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
    
    # LinearExplainer
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)
    
    print(f"SHAP values shape: {shap_values.values.shape}")
    
    # Waterfall plot (1 sample)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig('linear_shap_waterfall.png', dpi=150, bbox_inches='tight')
    print("Saved waterfall plot: linear_shap_waterfall.png")
    plt.close()
    

### KernelExplainer (Any Model)

**KernelExplainer** can be applied to any model (including black-box models) but has high computational cost.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: KernelExplainercan be applied to any model (including black-
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_wine
    import numpy as np
    
    print("\n=== KernelExplainer (Any Model) ===")
    
    # Wine dataset
    X, y = load_wine(return_X_y=True, as_frame=True)
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    print(f"Training accuracy: {model.score(X, y):.4f}")
    
    # KernelExplainer (using small sample due to high computational cost)
    print("\n=== Creating KernelExplainer (background data: 50 samples) ===")
    background = shap.sample(X, 50)  # Background data
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Compute SHAP values (only 3 samples)
    print("Computing SHAP values (this may take time)...")
    test_samples = X.iloc[:3]
    shap_values = explainer.shap_values(test_samples)
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    print("→ (number of classes, samples, features)")
    
    # Force plot (class 0)
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        test_samples.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('kernel_shap_force.png', dpi=150, bbox_inches='tight')
    print("Saved force plot: kernel_shap_force.png")
    plt.close()
    
    print("\n→ KernelSHAP is slow but applicable to any model")
    

### DeepExplainer (Neural Networks)

**DeepExplainer** is an efficient explainer for deep learning models (TensorFlow, PyTorch, etc.).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: DeepExplaineris an efficient explainer for deep learning mod
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import shap
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("\n=== DeepExplainer (PyTorch Neural Network) ===")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch data
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    
    # Simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNN(input_dim=20)
    
    # Training (simplified)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
        print(f"Test accuracy: {accuracy:.4f}")
    
    # DeepExplainer
    print("\n=== Using DeepExplainer ===")
    background = X_train_t[:100]  # Background data
    explainer = shap.DeepExplainer(model, background)
    
    # Compute SHAP values
    test_samples = X_test_t[:10]
    shap_values = explainer.shap_values(test_samples)
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # Summary plot (class 1)
    shap.summary_plot(
        shap_values[1],
        test_samples.numpy(),
        show=False
    )
    plt.tight_layout()
    plt.savefig('deep_shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved summary plot: deep_shap_summary.png")
    plt.close()
    

* * *

## 2.5 Applications and Limitations of SHAP

### Model Diagnosis

SHAP can be used to diagnose model problems:
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: SHAP can be used to diagnose model problems:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("=== Model Diagnosis with SHAP ===")
    
    # Create data with intentional bias
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    
    # Only features 0 and 1 are truly important
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Add noise feature with high correlation
    X[:, 2] = X[:, 0] + np.random.randn(n_samples) * 0.1  # Correlated with feature 0
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"Training accuracy: {model.score(X, y):.4f}")
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Feature importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), mean_abs_shap)
    plt.xlabel('Feature Index')
    plt.ylabel('Mean |SHAP value|')
    plt.title('Feature Importance via SHAP')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shap_diagnosis.png', dpi=150, bbox_inches='tight')
    print("Saved diagnosis result: shap_diagnosis.png")
    plt.close()
    
    print("\nFeature importance:")
    for i, importance in enumerate(mean_abs_shap):
        print(f"  Feature {i}: {importance:.4f}")
    
    print("\n→ Feature 2 (correlated noise) may be misidentified as important")
    print("→ SHAP discovers multicollinearity problems")
    

### Feature Selection

SHAP can be used to select truly important features:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: SHAP can be used to select truly important features:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import cross_val_score
    
    print("\n=== Feature Selection with SHAP ===")
    
    # Generate data (20 features, only 5 are important)
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=5,
        noise=10.0,
        random_state=42
    )
    
    # Initial model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Compute feature importance with SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Sort by importance
    feature_importance = sorted(
        enumerate(mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("Feature importance (SHAP):")
    for i, (idx, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. Feature {idx:2d}: {importance:.4f}")
    
    # Retrain with top k features
    for k in [5, 10, 15, 20]:
        top_features = [idx for idx, _ in feature_importance[:k]]
        X_selected = X[:, top_features]
    
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X_selected, y, cv=5, scoring='r2'
        )
    
        print(f"\nTop {k:2d} features: CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    
    print("\n→ Top 5 features are sufficient for good performance")
    print("→ Efficient feature selection with SHAP")
    

### Computational Cost

The computational cost of SHAP varies greatly depending on the algorithm and model:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: The computational cost of SHAP varies greatly depending on t
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import numpy as np
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("\n=== SHAP Computational Cost Comparison ===")
    
    # Measure computation time with varying data sizes
    results = []
    
    for n_samples in [100, 500, 1000, 2000]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            random_state=42
        )
    
        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
    
        # TreeExplainer
        explainer_tree = shap.TreeExplainer(model)
        start = time.time()
        shap_values_tree = explainer_tree(X)
        time_tree = time.time() - start
    
        # KernelExplainer (only for small samples)
        if n_samples <= 500:
            background = shap.sample(X, 50)
            explainer_kernel = shap.KernelExplainer(model.predict_proba, background)
            test_samples = X[:10]  # Only 10 samples
            start = time.time()
            shap_values_kernel = explainer_kernel.shap_values(test_samples)
            time_kernel = time.time() - start
        else:
            time_kernel = np.nan
    
        results.append((n_samples, time_tree, time_kernel))
        print(f"Samples {n_samples:4d}: TreeSHAP={time_tree:.3f}s, KernelSHAP={time_kernel:.3f}s")
    
    print("\n→ TreeSHAP is fast even for large-scale data")
    print("→ KernelSHAP is practical only for small-scale data")
    

### Interpretation Caveats

Caveat | Explanation | Mitigation  
---|---|---  
**Multicollinearity** | SHAP values unstable among correlated features | Use with correlation analysis, pre-select features  
**Background data selection** | KernelSHAP results depend on background data | Select representative samples, validate with multiple backgrounds  
**Extrapolation** | Reliability decreases outside training data range | Use with prediction confidence intervals  
**Causality** | SHAP values indicate correlation, not causation | Use with causal inference methods  
  
* * *

## Exercises

**Exercise 1: Comparing TreeSHAP and KernelSHAP**

Compute SHAP values for the same dataset and model using both TreeSHAP and KernelSHAP, and compare the results. How closely do they match?
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: Compute SHAP values for the same dataset and model using bot
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    from sklearn.ensemble import RandomForestClassifier
    
    # Exercise: Compare with scatter plot
    # Expected: Nearly identical, but KernelSHAP slightly differs due to approximation
    

**Exercise 2: Investigating Impact of Multicollinearity**

Create intentionally correlated features and investigate how they are interpreted by SHAP.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: Create intentionally correlated features and investigate how
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import shap
    
    # Exercise: Create X1 and X2 (X2 = X1 + noise)
    # Exercise: Train model and analyze with SHAP
    # Exercise: Compare SHAP values of X1 and X2
    # Analysis: SHAP values are distributed among correlated features
    

**Exercise 3: Anomaly Detection with SHAP Values**

Compare the distribution of SHAP values between normal and anomalous samples to identify features causing anomalies.
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: Compare the distribution of SHAP values between normal and a
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    from sklearn.ensemble import IsolationForest
    
    # Exercise: Create normal and anomalous data
    # Exercise: Analyze SHAP values of anomalous samples
    # Hint: Utilize summary plots and dependence plots
    

**Exercise 4: SHAP Application on Time Series Data**

Apply SHAP to time series data (e.g., predicting next day from past N days) and analyze which time points are important.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: Apply SHAP to time series data (e.g., predicting next day fr
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import shap
    
    # Exercise: Create lag features (t-1, t-2, ..., t-N)
    # Exercise: Analyze importance of each time point with SHAP
    # Expected: Recent time points are generally more important
    

**Exercise 5: Comparing SHAP and PFI (Permutation Feature Importance)**

Compare feature importance by SHAP with Permutation Feature Importance and analyze the differences.
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: Compare feature importance by SHAP with Permutation Feature 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import shap
    from sklearn.inspection import permutation_importance
    
    # Exercise: Train model
    # Exercise: Analyze correlation between the two importances
    # Exercise: Investigate features with large differences
    # Analysis: SHAP is local, PFI is global importance
    

* * *

## Summary

In this chapter, we learned the theory and practice of SHAP (SHapley Additive exPlanations).

### Key Points

  * **Shapley values** : Fair contribution evaluation based on cooperative game theory
  * **Axiomatic properties** : Unique solution satisfying efficiency, symmetry, dummy, and additivity
  * **SHAP** : Unified framework applying Shapley values to machine learning interpretation
  * **TreeSHAP** : Efficient exact computation for tree-based models
  * **KernelSHAP** : Approximation method applicable to any model
  * **Visualization** : Multifaceted analysis with Waterfall, Force, Summary, Dependence plots
  * **Applications** : Model diagnosis, feature selection, anomaly detection, etc.
  * **Limitations** : Attention needed for computational cost, multicollinearity, causal interpretation

### Advantages and Limitations of SHAP

Aspect | Advantages | Limitations  
---|---|---  
**Theoretical Foundation** | Solid game theory basis | Computational complexity problem (exact computation difficult)  
**Consistency** | Consistent interpretation through axiomatic properties | Unstable with multicollinearity  
**Application Scope** | Applicable to any model | Computational cost varies greatly by model  
**Interpretability** | Both local explanation and global importance | Risk of confusing correlation with causation  
  
### Next Steps

In the next chapter, we will learn about **Integrated Gradients** and **Attention mechanisms**. We will acquire more advanced interpretation techniques, including deep learning model interpretation methods, gradient-based attribution analysis, and visualization of attention mechanisms in Transformer models.

### References

  * Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." _NeurIPS_.
  * Shapley, L. S. (1953). "A value for n-person games." _Contributions to the Theory of Games_ , 2(28), 307-317.
  * Lundberg, S. M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." _Nature Machine Intelligence_ , 2(1), 2522-5839.
  * Molnar, C. (2022). "Interpretable Machine Learning." <https://christophm.github.io/interpretable-ml-book/>
