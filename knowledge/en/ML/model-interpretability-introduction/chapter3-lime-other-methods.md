---
title: "Chapter 3: LIME and Other Interpretation Methods"
chapter_title: "Chapter 3: LIME and Other Interpretation Methods"
subtitle: Diverse Approaches for Local and Global Interpretation
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers LIME and Other Interpretation Methods. You will learn principles of LIME and Utilize cutting-edge methods such as Anchors.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the principles of LIME and local linear approximation mechanisms
  * ✅ Calculate model-agnostic feature importance using Permutation Importance
  * ✅ Visualize feature effects using Partial Dependence Plots (PDP)
  * ✅ Utilize cutting-edge methods such as Anchors and counterfactual explanations
  * ✅ Make informed choices between methods like SHAP vs LIME
  * ✅ Understand the trade-off between computational cost and interpretation accuracy

* * *

## 3.1 LIME (Local Interpretable Model-agnostic Explanations)

### What is LIME

**LIME (Local Interpretable Model-agnostic Explanations)** is a method that explains individual predictions of any black-box model by approximating them with locally interpretable models.

> "Even complex models can be approximated by simple linear models around a single point"

### Basic Principles of LIME
    
    
    ```mermaid
    graph LR
        A[Original Data Point] --> B[Neighborhood Sampling]
        B --> C[Black-box Model Prediction]
        C --> D[Distance Weighting]
        D --> E[Linear Model Approximation]
        E --> F[Feature Importance]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#c8e6c9
    ```

#### Mathematical Formulation

LIME solves the following optimization problem:

$$ \xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

  * $f$: Black-box model
  * $g$: Interpretable model (e.g., linear model)
  * $\mathcal{L}$: Loss function (difference between predictions of $f$ and $g$)
  * $\pi_x$: Weight based on distance from the original data point $x$
  * $\Omega(g)$: Penalty for model complexity

### LIME Implementation: Tabular Data
    
    
    # Requirements:
    # - Python 3.9+
    # - lime>=0.2.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: LIME Implementation: Tabular Data
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from lime import lime_tabular
    
    # Data preparation
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Black-box model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    
    # Create LIME Explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    
    # Explain one sample
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    
    # Generate explanation
    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    print("\n=== LIME Explanation ===")
    print(f"Predicted class: {data.target_names[model.predict([sample])[0]]}")
    print(f"Prediction probability: {model.predict_proba([sample])[0]}")
    print("\nFeature contributions:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:+.4f}")
    
    # Visualization
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    Model accuracy: 0.965
    
    === LIME Explanation ===
    Predicted class: benign
    Prediction probability: [0.03 0.97]
    
    Feature contributions:
      worst concave points <= 0.10: +0.2845
      worst radius <= 13.43: +0.1234
      worst perimeter <= 86.60: +0.0987
      mean concave points <= 0.05: +0.0765
      worst area <= 549.20: +0.0543
    

> **Important** : LIME provides local explanations and does not represent the overall model behavior.

### LIME Sampling Method

#### Sampling Process
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Sampling Process
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Simple 2D data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Sample to explain
    sample = X[0]
    
    # LIME-style sampling (generate neighborhood with normal distribution)
    n_samples = 1000
    noise_scale = 0.5
    
    # Sample around the instance
    samples = np.random.normal(
        loc=sample,
        scale=noise_scale,
        size=(n_samples, 2)
    )
    
    # Predict with black-box model
    predictions = model.predict_proba(samples)[:, 1]
    
    # Calculate distances (Euclidean distance)
    distances = np.sqrt(np.sum((samples - sample)**2, axis=1))
    
    # Kernel weights (inversely proportional to distance)
    kernel_width = 0.75
    weights = np.exp(-(distances**2) / (kernel_width**2))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original data space
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                    alpha=0.5, edgecolors='black')
    axes[0].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2,
                    label='Sample to explain')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('Original Data Space', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sampled points
    scatter = axes[1].scatter(samples[:, 0], samples[:, 1],
                             c=predictions, cmap='coolwarm',
                             alpha=0.4, s=20, edgecolors='none')
    axes[1].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title('Sampled Neighborhood Points', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Prediction probability')
    
    # Weighted points
    scatter2 = axes[2].scatter(samples[:, 0], samples[:, 1],
                              c=predictions, cmap='coolwarm',
                              alpha=weights, s=weights*100, edgecolors='none')
    axes[2].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2)
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].set_title('Distance Weighting (closer = larger)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[2], label='Prediction probability')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of samples: {n_samples}")
    print(f"Average distance: {distances.mean():.3f}")
    print(f"Min/Max weights: {weights.min():.4f} / {weights.max():.4f}")
    

### Complete LIME Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics.pairwise import rbf_kernel
    
    class SimpleLIME:
        """Simple LIME implementation"""
    
        def __init__(self, kernel_width=0.75, n_samples=5000):
            self.kernel_width = kernel_width
            self.n_samples = n_samples
    
        def explain_instance(self, model, instance, X_train,
                            feature_names=None, n_features=10):
            """
            Explain an individual instance
    
            Parameters:
            -----------
            model : Trained model (requires predict_proba method)
            instance : Sample to explain (1D array)
            X_train : Training data (for statistics)
            feature_names : List of feature names
            n_features : Number of top features to return
    
            Returns:
            --------
            explanations : List of features and importance
            """
            # Neighborhood sampling
            samples = self._sample_around_instance(instance, X_train)
    
            # Model predictions
            predictions = model.predict_proba(samples)[:, 1]
    
            # Distance-based weight calculation
            distances = np.sqrt(np.sum((samples - instance)**2, axis=1))
            weights = np.exp(-(distances**2) / (self.kernel_width**2))
    
            # Linear model approximation
            linear_model = Ridge(alpha=1.0)
            linear_model.fit(samples, predictions, sample_weight=weights)
    
            # Get feature importance
            feature_importance = linear_model.coef_
    
            # Set feature names
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(instance))]
    
            # Sort by importance
            sorted_idx = np.argsort(np.abs(feature_importance))[::-1][:n_features]
    
            explanations = [
                (feature_names[idx], feature_importance[idx])
                for idx in sorted_idx
            ]
    
            return explanations, linear_model.score(samples, predictions,
                                                   sample_weight=weights)
    
        def _sample_around_instance(self, instance, X_train):
            """Sample around the instance"""
            # Use training data statistics
            means = X_train.mean(axis=0)
            stds = X_train.std(axis=0)
    
            # Sample with normal distribution
            samples = np.random.normal(
                loc=instance,
                scale=stds * 0.5,  # Scale adjustment
                size=(self.n_samples, len(instance))
            )
    
            return samples
    
    # Usage example
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Data preparation
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Explain with SimpleLIME
    lime_explainer = SimpleLIME(kernel_width=0.75, n_samples=5000)
    sample = X_test[0]
    
    explanations, r2_score = lime_explainer.explain_instance(
        model=model,
        instance=sample,
        X_train=X_train,
        feature_names=data.feature_names,
        n_features=10
    )
    
    print("=== SimpleLIME Explanation ===")
    print(f"Local model R² score: {r2_score:.3f}")
    print("\nFeature importance:")
    for feature, importance in explanations:
        print(f"  {feature}: {importance:+.4f}")
    

* * *

## 3.2 Permutation Importance

### What is Permutation Importance

**Permutation Importance** is a model-agnostic feature importance calculation method that measures the decrease in model performance when each feature is randomly shuffled.

#### Algorithm

  1. Calculate baseline model performance
  2. For each feature: 
     * Shuffle the values of that feature
     * Recalculate model performance
     * Performance decrease = importance
  3. Restore and move to the next feature

    
    
    ```mermaid
    graph TD
        A[Original Data] --> B[Baseline Performance Measurement]
        B --> C[Shuffle Feature 1]
        C --> D[Performance Measurement]
        D --> E[Importance = Performance Decrease]
        E --> F[Shuffle Feature 2]
        F --> G[...]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#c8e6c9
    ```

### Implementation with scikit-learn
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Implementation with scikit-learn
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # Data preparation (diabetes dataset)
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Baseline performance
    baseline_score = model.score(X_test, y_test)
    print(f"Baseline R² score: {baseline_score:.3f}")
    
    # Calculate Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,  # Repeat shuffling 30 times
        random_state=42,
        n_jobs=-1
    )
    
    # Organize results
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n=== Permutation Importance ===")
    print(importance_df)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance_mean'],
            xerr=importance_df['importance_std'],
            align='center', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² decrease)')
    ax.set_title('Permutation Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

### Custom Implementation: Permutation Importance
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from sklearn.metrics import r2_score, accuracy_score
    
    def custom_permutation_importance(model, X, y, metric='r2', n_repeats=10):
        """
        Custom implementation of Permutation Importance
    
        Parameters:
        -----------
        model : Trained model
        X : Features (DataFrame or ndarray)
        y : Target
        metric : Evaluation metric ('r2' or 'accuracy')
        n_repeats : Number of shuffle repetitions per feature
    
        Returns:
        --------
        importances : Feature importance (mean and standard deviation)
        """
        X_array = X.values if hasattr(X, 'values') else X
        n_features = X_array.shape[1]
    
        # Select metric function
        if metric == 'r2':
            score_func = r2_score
            predictions = model.predict(X_array)
        elif metric == 'accuracy':
            score_func = accuracy_score
            predictions = model.predict(X_array)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
        # Baseline score
        baseline_score = score_func(y, predictions)
    
        # Calculate importance for each feature
        importances = np.zeros((n_features, n_repeats))
    
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Shuffle feature
                X_permuted = X_array.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
    
                # Predict and evaluate
                if metric == 'r2':
                    perm_predictions = model.predict(X_permuted)
                else:
                    perm_predictions = model.predict(X_permuted)
    
                perm_score = score_func(y, perm_predictions)
    
                # Score decrease = importance
                importances[feature_idx, repeat] = baseline_score - perm_score
    
        # Calculate statistics
        result = {
            'importances_mean': importances.mean(axis=1),
            'importances_std': importances.std(axis=1),
            'importances': importances
        }
    
        return result
    
    # Usage example
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate with custom implementation
    custom_perm = custom_permutation_importance(
        model, X_test, y_test,
        metric='accuracy',
        n_repeats=30
    )
    
    # Organize results
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': custom_perm['importances_mean'],
        'importance_std': custom_perm['importances_std']
    }).sort_values('importance_mean', ascending=False)
    
    print("=== Custom Permutation Importance ===")
    print(results_df.head(10))
    

### Comparison of Permutation Importance and SHAP
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - shap>=0.42.0
    
    """
    Example: Comparison of Permutation Importance and SHAP
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    import shap
    
    # Data and model
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 1. Permutation Importance
    perm_imp = permutation_importance(
        model, X, y, n_repeats=30, random_state=42
    )
    
    # 2. SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # 3. Tree Feature Importance (for comparison)
    tree_importance = model.feature_importances_
    
    # Comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = [
        ('Permutation\nImportance', perm_imp.importances_mean),
        ('SHAP\nImportance', shap_importance),
        ('Tree Feature\nImportance', tree_importance)
    ]
    
    for ax, (title, importance) in zip(axes, methods):
        sorted_idx = np.argsort(importance)
        y_pos = np.arange(len(sorted_idx))
    
        ax.barh(y_pos, importance[sorted_idx], alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(X.columns[sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis
    print("=== Correlation Between Methods ===")
    comparison_df = pd.DataFrame({
        'Permutation': perm_imp.importances_mean,
        'SHAP': shap_importance,
        'Tree': tree_importance
    })
    print(comparison_df.corr())
    

* * *

## 3.3 Partial Dependence Plots (PDP)

### What is PDP

**Partial Dependence Plot** is a method that visualizes the average effect of features on model predictions.

#### Mathematical Definition

Partial dependence function for feature $x_S$:

$$ \hat{f}_{x_S}(x_S) = \mathbb{E}_{x_C}[\hat{f}(x_S, x_C)] = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_S, x_C^{(i)}) $$

  * $x_S$: Target feature
  * $x_C$: Other features
  * $\hat{f}$: Model's prediction function

### 1D PDP Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 1D PDP Implementation
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import PartialDependenceDisplay
    
    # Data preparation
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate and visualize PDP
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features_to_plot = ['age', 'bmi', 's5', 'bp', 's1', 's3']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes.flatten()[idx]
    
        # Display PDP
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            ax=ax, kind='average'
        )
    
        ax.set_title(f'PDP: {feature}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Partial Dependence Plot generation complete ===")
    print("Visualized average effect of each feature")
    

### 2D PDP (Interaction Visualization)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 2D PDP (Interaction Visualization)
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    
    # Visualize interaction between two features
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2D PDP example 1: bmi vs s5
    display1 = PartialDependenceDisplay.from_estimator(
        model, X, features=[('bmi', 's5')],
        ax=axes[0], kind='average'
    )
    axes[0].set_title('2D PDP: BMI vs S5 (Interaction)', fontsize=14)
    
    # 2D PDP example 2: age vs bmi
    display2 = PartialDependenceDisplay.from_estimator(
        model, X, features=[('age', 'bmi')],
        ax=axes[1], kind='average'
    )
    axes[1].set_title('2D PDP: Age vs BMI (Interaction)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

### ICE (Individual Conditional Expectation)

**ICE** visualizes conditional expectations for each sample and captures heterogeneity.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: ICEvisualizes conditional expectations for each sample and c
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    
    # ICE plot (individual conditional expectation)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    features_for_ice = ['bmi', 's5', 'bp']
    
    for ax, feature in zip(axes, features_for_ice):
        # ICE plot (individual=True)
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            kind='individual',  # Individual lines
            ax=ax,
            subsample=50,  # Display only 50 samples
            random_state=42
        )
    
        # Overlay PDP
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            kind='average',  # Average line
            ax=ax,
            line_kw={'color': 'red', 'linewidth': 3, 'label': 'PDP (average)'}
        )
    
        ax.set_title(f'ICE + PDP: {feature}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ICE Plot Explanation ===")
    print("- Thin lines: Individual sample conditional expectations (ICE)")
    print("- Thick red line: Average effect (PDP)")
    print("- Line variance = heterogeneity (individual differences)")
    

### Custom PDP Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def compute_partial_dependence(model, X, feature_idx, grid_resolution=50):
        """
        Compute partial dependence
    
        Parameters:
        -----------
        model : Trained model
        X : Feature data
        feature_idx : Target feature index
        grid_resolution : Grid resolution
    
        Returns:
        --------
        grid_values : Grid point values
        pd_values : Partial dependence values
        """
        X_array = X.values if hasattr(X, 'values') else X
    
        # Generate grid over target feature range
        feature_min = X_array[:, feature_idx].min()
        feature_max = X_array[:, feature_idx].max()
        grid_values = np.linspace(feature_min, feature_max, grid_resolution)
    
        # Calculate partial dependence
        pd_values = []
    
        for grid_value in grid_values:
            # Fix target feature to grid_value for all samples
            X_modified = X_array.copy()
            X_modified[:, feature_idx] = grid_value
    
            # Average of predictions
            predictions = model.predict(X_modified)
            pd_values.append(predictions.mean())
    
        return grid_values, np.array(pd_values)
    
    # Usage example
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate PDP with custom implementation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (ax, feature) in enumerate(zip(axes.flatten(), X.columns[:6])):
        grid, pd_vals = compute_partial_dependence(
            model, X, feature_idx=idx, grid_resolution=100
        )
    
        ax.plot(grid, pd_vals, linewidth=2, color='blue')
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'Custom PDP: {feature}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.4 Other Interpretation Methods

### Anchors

**Anchors** is a method that finds minimal rule sets that guarantee predictions.

> "If these conditions are met, the same prediction will occur with over 95% probability"
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: "If these conditions are met, the same prediction will occur
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from anchor import anchor_tabular
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Data preparation
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create Anchors Explainer
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=data.target_names,
        feature_names=data.feature_names,
        train_data=X_train.values
    )
    
    # Generate explanation
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    
    explanation = explainer.explain_instance(
        data_row=sample,
        classifier_fn=model.predict,
        threshold=0.95  # 95% confidence
    )
    
    print("=== Anchors Explanation ===")
    print(f"Prediction: {data.target_names[model.predict([sample])[0]]}")
    print(f"\nAnchor (precision={explanation.precision():.2f}):")
    print('AND'.join(explanation.names()))
    print(f"\nCoverage: {explanation.coverage():.2%}")
    

### Counterfactual Explanations

**Counterfactual Explanations** show "what needs to change for the prediction to change".
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Counterfactual Explanationsshow "what needs to change for th
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Data preparation
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    def find_counterfactual(model, instance, target_class,
                           X_train, max_iterations=1000,
                           step_size=0.1):
        """
        Search for counterfactual explanation (simple gradient-based)
    
        Parameters:
        -----------
        model : Trained model
        instance : Original instance
        target_class : Target class
        X_train : Training data (for range reference)
        max_iterations : Maximum iterations
        step_size : Step size
    
        Returns:
        --------
        counterfactual : Counterfactual instance
        changes : Change details
        """
        counterfactual = instance.copy()
    
        for iteration in range(max_iterations):
            # Current prediction
            pred_class = model.predict([counterfactual])[0]
    
            if pred_class == target_class:
                break
    
            # Randomly select and change a feature
            feature_idx = np.random.randint(0, len(counterfactual))
    
            # Random change within training data range
            feature_range = X_train.iloc[:, feature_idx]
            new_value = np.random.uniform(
                feature_range.min(),
                feature_range.max()
            )
    
            counterfactual[feature_idx] = new_value
    
        # Calculate changes
        changes = {}
        for idx, (orig, cf) in enumerate(zip(instance, counterfactual)):
            if not np.isclose(orig, cf):
                changes[X.columns[idx]] = {
                    'original': orig,
                    'counterfactual': cf,
                    'change': cf - orig
                }
    
        return counterfactual, changes
    
    # Usage example
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    original_pred = model.predict([sample])[0]
    target = 1 - original_pred  # Opposite class
    
    counterfactual, changes = find_counterfactual(
        model, sample, target, X_train,
        max_iterations=5000
    )
    
    print("=== Counterfactual Explanation ===")
    print(f"Original prediction: {data.target_names[original_pred]}")
    print(f"Target prediction: {data.target_names[target]}")
    print(f"After counterfactual: {data.target_names[model.predict([counterfactual])[0]]}")
    print(f"\nFeatures requiring changes (top 5):")
    
    sorted_changes = sorted(
        changes.items(),
        key=lambda x: abs(x[1]['change']),
        reverse=True
    )[:5]
    
    for feature, change_info in sorted_changes:
        print(f"\n{feature}:")
        print(f"  Original value: {change_info['original']:.2f}")
        print(f"  Changed to: {change_info['counterfactual']:.2f}")
        print(f"  Change amount: {change_info['change']:+.2f}")
    

### Feature Ablation

**Feature Ablation** measures the performance change when features are removed.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Feature Ablationmeasures the performance change when feature
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    
    # Data preparation
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Baseline performance
    baseline_score = r2_score(y, model.predict(X))
    
    # Performance when each feature is removed
    ablation_results = []
    
    for feature in X.columns:
        # Remove feature
        X_ablated = X.drop(columns=[feature])
    
        # Train new model
        model_ablated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_ablated.fit(X_ablated, y)
    
        # Measure performance
        score = r2_score(y, model_ablated.predict(X_ablated))
        importance = baseline_score - score
    
        ablation_results.append({
            'feature': feature,
            'score_without': score,
            'importance': importance
        })
    
    # Organize results
    ablation_df = pd.DataFrame(ablation_results).sort_values(
        'importance', ascending=False
    )
    
    print("=== Feature Ablation Results ===")
    print(f"Baseline R²: {baseline_score:.3f}\n")
    print(ablation_df)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(ablation_df))
    ax.barh(y_pos, ablation_df['importance'], alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ablation_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² decrease when removed)')
    ax.set_title('Feature Ablation Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Method Comparison and Best Practices

### SHAP vs LIME

Aspect | SHAP | LIME  
---|---|---  
**Theoretical Foundation** | Game theory (Shapley values) | Local linear approximation  
**Consistency** | High (with mathematical guarantees) | Low (depends on sampling)  
**Computational Cost** | High (especially KernelSHAP) | Moderate  
**Interpretation Granularity** | Both local and global | Mainly local  
**Model Agnosticism** | Completely agnostic | Completely agnostic  
**Reproducibility** | High | Moderate (random sampling)  
**Ease of Use** | Very high | High  
  
### Global vs Local Interpretation

Method | Type | Use Case  
---|---|---  
**LIME** | Local | Individual prediction explanation  
**SHAP** | Both | Individual and overall understanding  
**Permutation Importance** | Global | Overall feature importance  
**PDP/ICE** | Global | Average feature effect  
**Anchors** | Local | Rule-based explanation  
**Counterfactuals** | Local | Change suggestions  
  
### Computational Cost Comparison
    
    
    # Requirements:
    # - Python 3.9+
    # - lime>=0.2.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - shap>=0.42.0
    
    """
    Example: Computational Cost Comparison
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import time
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import shap
    from lime import lime_tabular
    
    # Data preparation
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Measure time for explaining 1 sample
    sample = X_test.iloc[0].values
    
    results = {}
    
    # SHAP TreeExplainer
    start = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:10])
    results['SHAP (Tree)'] = time.time() - start
    
    # SHAP KernelExplainer (slow)
    start = time.time()
    explainer_kernel = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_train, 50)
    )
    shap_kernel = explainer_kernel.shap_values(X_test.iloc[:5])
    results['SHAP (Kernel)'] = time.time() - start
    
    # LIME
    start = time.time()
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    for i in range(10):
        exp = lime_explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=10
        )
    results['LIME'] = time.time() - start
    
    # Permutation Importance
    start = time.time()
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42
    )
    results['Permutation'] = time.time() - start
    
    # Display results
    print("=== Computation Time Comparison (10 samples) ===")
    for method, duration in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:20s}: {duration:6.2f} seconds")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    times = list(results.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.barh(methods, times, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Computational Cost Comparison of Interpretation Methods', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Display values
    for bar, time_val in zip(bars, times):
        ax.text(time_val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{time_val:.2f}s', va='center')
    
    plt.tight_layout()
    plt.show()
    

### Selection Guide

Situation | Recommended Method | Reason  
---|---|---  
Detailed explanation of individual predictions | SHAP (Tree) | Fast with consistency  
Individual explanation of any model | LIME, KernelSHAP | Model agnostic  
Overall feature importance | Permutation, SHAP | Global understanding  
Visualize feature effects | PDP/ICE | Intuitive understanding  
Rule-based explanation | Anchors | If-then format  
Change suggestions | Counterfactuals | Actionable advice  
Limited computation time | LIME, Tree SHAP | Relatively fast  
High accuracy required | SHAP | Theoretical guarantees  
  
### Best Practices

  1. **Use Multiple Methods**

     * SHAP (global) + LIME (local) for multi-faceted understanding
     * PDP (average) + ICE (individual) to capture heterogeneity
  2. **Consider Computational Resources**

     * Production environment: TreeSHAP, LIME
     * Research/analysis: KernelSHAP, use all methods
  3. **Integrate Domain Knowledge**

     * Validate interpretation results with business knowledge
     * Investigate unnatural explanations
  4. **Visualization Considerations**

     * Concise for non-experts
     * Detailed for specialists
  5. **Ensure Reproducibility**

     * Fix random_state
     * Save and share explanations

* * *

## 3.6 Chapter Summary

### What We Learned

  1. **LIME**

     * Black-box interpretation through local linear approximation
     * Sampling and weighting mechanisms
     * Implementation and visualization methods
  2. **Permutation Importance**

     * Model-agnostic feature importance
     * Measuring performance decrease through shuffling
     * Differences from SHAP and Tree Importance
  3. **Partial Dependence Plots**

     * Visualizing average feature effects
     * Understanding interactions with 2D PDP
     * Capturing heterogeneity with ICE
  4. **Other Methods**

     * Anchors: Rule-based explanations
     * Counterfactuals: Change suggestions
     * Feature Ablation: Importance measurement through removal
  5. **Method Selection**

     * Characteristic comparison of SHAP vs LIME
     * Trade-off between computational cost and accuracy
     * Optimal method selection by situation

### Summary of Key Method Characteristics

Method | Strengths | Weaknesses | Application Scenarios  
---|---|---|---  
**LIME** | Easy to understand, fast | Unstable, local only | Simple individual prediction explanation  
**SHAP** | Theoretical guarantees, consistency | Computational cost (Kernel) | When precise interpretation is needed  
**Permutation** | Simple, intuitive | Unstable with correlated features | Understanding overall importance  
**PDP/ICE** | Intuitive visualization | Limitations with interactions | Understanding feature effects  
**Anchors** | Rule format, clear | Coverage limitations | Rule-based explanation  
  
### Next Chapter

In Chapter 4, we will learn about **Image and Text Data Interpretation** :

  * Image interpretation with Grad-CAM
  * Attention mechanism visualization
  * BERT model interpretation
  * Integrated Gradients
  * Practical application examples

* * *

## Exercises

### Problem 1 (Difficulty: easy)

List and explain three main differences between LIME and SHAP.

Sample Answer

**Answer** :

  1. **Theoretical Foundation**

     * LIME: Local linear approximation (sampling + linear model)
     * SHAP: Shapley values from game theory (axiomatic approach)
  2. **Consistency and Reproducibility**

     * LIME: Results may vary between runs due to sampling dependency
     * SHAP: Mathematically consistent results are guaranteed
  3. **Application Scope**

     * LIME: Mainly local explanations (individual samples)
     * SHAP: Both local and global (individual + overall importance)

**Selection Guidelines** :

  * Speed priority/simple explanation → LIME
  * Accuracy priority/theoretical guarantees → SHAP
  * Ideally, use both for multi-faceted understanding

### Problem 2 (Difficulty: medium)

Manually implement Permutation Importance and compare with scikit-learn results.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Manually implement Permutation Importance and compare with s
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    
    # Data preparation
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Manual implementation
    def manual_permutation_importance(model, X, y, n_repeats=10):
        """Manual implementation of Permutation Importance"""
        baseline_score = accuracy_score(y, model.predict(X))
        n_features = X.shape[1]
        importances = np.zeros((n_features, n_repeats))
    
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Shuffle feature
                X_permuted = X.copy()
                X_permuted.iloc[:, feature_idx] = np.random.permutation(
                    X_permuted.iloc[:, feature_idx]
                )
    
                # Calculate score
                perm_score = accuracy_score(y, model.predict(X_permuted))
                importances[feature_idx, repeat] = baseline_score - perm_score
    
        return {
            'importances_mean': importances.mean(axis=1),
            'importances_std': importances.std(axis=1)
        }
    
    # Calculate with manual implementation
    manual_result = manual_permutation_importance(
        model, X_test, y_test, n_repeats=30
    )
    
    # scikit-learn implementation
    sklearn_result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42
    )
    
    # Comparison
    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'Manual_Mean': manual_result['importances_mean'],
        'Manual_Std': manual_result['importances_std'],
        'Sklearn_Mean': sklearn_result.importances_mean,
        'Sklearn_Std': sklearn_result.importances_std
    }).sort_values('Manual_Mean', ascending=False)
    
    print("=== Permutation Importance Comparison ===")
    print(comparison_df)
    
    # Correlation check
    correlation = np.corrcoef(
        manual_result['importances_mean'],
        sklearn_result.importances_mean
    )[0, 1]
    print(f"\nCorrelation between manual and sklearn implementation: {correlation:.4f}")
    print("(Closer to 1 = better match)")
    

**Sample Output** :
    
    
    === Permutation Importance Comparison ===
                       Feature  Manual_Mean  Manual_Std  Sklearn_Mean  Sklearn_Std
    2       petal length (cm)     0.3156      0.0289        0.3200       0.0265
    3        petal width (cm)     0.2933      0.0312        0.2867       0.0298
    0       sepal length (cm)     0.0089      0.0145        0.0111       0.0134
    1        sepal width (cm)     0.0067      0.0123        0.0044       0.0098
    
    Correlation between manual and sklearn implementation: 0.9987
    (Closer to 1 = better match)
    

### Problem 3 (Difficulty: medium)

Explain the difference between Partial Dependence Plot and ICE plot, and describe when ICE is useful.

Sample Answer

**Answer** :

**Differences** :

  * **PDP (Partial Dependence Plot)**

    * Shows average effect across all samples
    * Formula: $\hat{f}_{PDP}(x_s) = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_s, x_c^{(i)})$
    * Represented by a single line
  * **ICE (Individual Conditional Expectation)**

    * Shows effect for each sample individually
    * Formula: $\hat{f}^{(i)}_{ICE}(x_s) = \hat{f}(x_s, x_c^{(i)})$
    * n lines (one per sample)

**When ICE is Useful** :

  1. **Detecting Heterogeneity**

     * When subgroups have different effects
     * Example: Age effect differs by gender
  2. **Discovering Interactions**

     * Complex interactions between features
     * Hidden by averaging in PDP
  3. **Understanding Non-linearity**

     * Individual patterns are diverse
     * Average appears simple but is actually complex

**Visualization Example** :
    
    
    from sklearn.inspection import PartialDependenceDisplay
    
    # PDP only (average)
    PartialDependenceDisplay.from_estimator(
        model, X, features=['age'], kind='average'
    )
    
    # ICE + PDP (individual + average)
    PartialDependenceDisplay.from_estimator(
        model, X, features=['age'],
        kind='both',  # Display both
        subsample=50
    )
    

### Problem 4 (Difficulty: hard)

Apply LIME, SHAP, and Permutation Importance to the following data and compare the results. If feature importance rankings differ, consider the reasons.
    
    
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - lime>=0.2.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - shap>=0.42.0
    
    """
    Example: Apply LIME, SHAP, and Permutation Importance to the followin
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    import shap
    from lime import lime_tabular
    
    # Data preparation
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 1. SHAP (global)
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_test)
    shap_importance = np.abs(shap_values[1]).mean(axis=0)
    
    # 2. LIME (average across multiple samples)
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    
    lime_importances = []
    for i in range(min(50, len(X_test))):  # 50 samples
        exp = explainer_lime.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=len(X.columns)
        )
        weights = dict(exp.as_list())
        # Extract feature names (remove condition parts)
        feature_weights = {}
        for key, val in weights.items():
            feature_name = key.split('<=')[0].split('>')[0].strip()
            if feature_name in X.columns:
                feature_weights[feature_name] = abs(val)
        lime_importances.append(feature_weights)
    
    # Average LIME importance
    lime_importance_mean = pd.DataFrame(lime_importances).mean().reindex(X.columns).fillna(0)
    
    # 3. Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42
    )
    
    # Integrate results
    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': shap_importance,
        'LIME': lime_importance_mean.values,
        'Permutation': perm_importance.importances_mean
    })
    
    # Normalize (for comparison)
    for col in ['SHAP', 'LIME', 'Permutation']:
        comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
    
    # Ranking
    comparison_df['SHAP_Rank'] = comparison_df['SHAP'].rank(ascending=False)
    comparison_df['LIME_Rank'] = comparison_df['LIME'].rank(ascending=False)
    comparison_df['Perm_Rank'] = comparison_df['Permutation'].rank(ascending=False)
    
    print("=== Feature Importance Comparison of 3 Methods (Top 10) ===\n")
    top_features = comparison_df.nlargest(10, 'SHAP')[
        ['Feature', 'SHAP', 'LIME', 'Permutation',
         'SHAP_Rank', 'LIME_Rank', 'Perm_Rank']
    ]
    print(top_features)
    
    # Correlation analysis
    print("\n=== Correlation Between Methods ===")
    correlation_matrix = comparison_df[['SHAP', 'LIME', 'Permutation']].corr()
    print(correlation_matrix)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, method in zip(axes, ['SHAP', 'LIME', 'Permutation']):
        sorted_df = comparison_df.sort_values(method, ascending=True).tail(10)
    
        ax.barh(range(len(sorted_df)), sorted_df[method],
                alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Feature'])
        ax.set_xlabel('Normalized Importance')
        ax.set_title(f'{method}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Rank difference analysis
    print("\n=== Features with Large Rank Differences ===")
    comparison_df['Rank_Std'] = comparison_df[
        ['SHAP_Rank', 'LIME_Rank', 'Perm_Rank']
    ].std(axis=1)
    
    disagreement = comparison_df.nlargest(5, 'Rank_Std')[
        ['Feature', 'SHAP_Rank', 'LIME_Rank', 'Perm_Rank', 'Rank_Std']
    ]
    print(disagreement)
    
    print("\n=== Discussion ===")
    print("""
    Reasons for ranking differences:
    
    1. **Differences in What is Measured**
       - SHAP: Contribution of each feature (game theory)
       - LIME: Coefficients of local linear approximation
       - Permutation: Performance decrease when shuffled
    
    2. **Local vs Global**
       - LIME: Local (around selected samples)
       - SHAP/Permutation: More global
    
    3. **Feature Correlation**
       - In highly correlated feature groups, importance is distributed differently by method
       - Permutation does not consider correlations
    
    4. **Differences in Calculation Methods**
       - SHAP: Considers all feature combinations
       - Permutation: Evaluates each independently
       - LIME: Sampling-based (with randomness)
    
    Recommendation: Use multiple methods for multi-faceted interpretation
    """)
    

### Problem 5 (Difficulty: hard)

Implement an algorithm that uses Counterfactual Explanations to find minimal changes that alter the prediction. Consider how to evaluate the validity of the changes.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    class CounterfactualExplainer:
        """Generate minimal counterfactual explanations"""
    
        def __init__(self, model, X_train):
            self.model = model
            self.X_train = X_train
            self.feature_ranges = {
                'min': X_train.min(axis=0),
                'max': X_train.max(axis=0),
                'median': X_train.median(axis=0)
            }
    
        def find_minimal_counterfactual(self, instance, target_class,
                                       max_iterations=1000,
                                       change_penalty=0.1):
            """
            Search for counterfactual achieving target class with minimal changes
    
            Parameters:
            -----------
            instance : Original instance
            target_class : Target class
            max_iterations : Maximum iterations
            change_penalty : Penalty for changes
    
            Returns:
            --------
            best_counterfactual : Best counterfactual instance
            changes : Change details
            metadata : Metadata
            """
            best_counterfactual = None
            best_distance = float('inf')
    
            current = instance.copy()
    
            for iteration in range(max_iterations):
                # Current prediction
                pred_class = self.model.predict([current])[0]
    
                if pred_class == target_class:
                    # Target achieved: calculate distance
                    distance = self._compute_distance(instance, current)
    
                    if distance < best_distance:
                        best_distance = distance
                        best_counterfactual = current.copy()
    
                # Randomly change one feature
                feature_idx = np.random.randint(0, len(current))
    
                # Change within feasible range
                feature_range = (
                    self.feature_ranges['min'][feature_idx],
                    self.feature_ranges['max'][feature_idx]
                )
    
                # Prefer values close to original (normal distribution)
                new_value = np.random.normal(
                    loc=instance[feature_idx],
                    scale=(feature_range[1] - feature_range[0]) * 0.1
                )
    
                # Clip to range
                new_value = np.clip(new_value, feature_range[0], feature_range[1])
                current[feature_idx] = new_value
    
            if best_counterfactual is None:
                return None, None, {'success': False}
    
            # Analyze changes
            changes = self._analyze_changes(instance, best_counterfactual)
    
            # Metadata
            metadata = {
                'success': True,
                'distance': best_distance,
                'n_changes': len(changes),
                'validity': self._check_validity(best_counterfactual)
            }
    
            return best_counterfactual, changes, metadata
    
        def _compute_distance(self, instance1, instance2):
            """L2 distance (normalized)"""
            # Normalize each feature by range
            ranges = self.feature_ranges['max'] - self.feature_ranges['min']
            normalized_diff = (instance1 - instance2) / ranges
            return np.sqrt(np.sum(normalized_diff**2))
    
        def _analyze_changes(self, original, counterfactual, threshold=0.01):
            """Analyze changes"""
            changes = {}
            for idx, (orig, cf) in enumerate(zip(original, counterfactual)):
                relative_change = abs((cf - orig) / (orig + 1e-10))
    
                if relative_change > threshold:
                    changes[idx] = {
                        'original': orig,
                        'counterfactual': cf,
                        'absolute_change': cf - orig,
                        'relative_change': relative_change
                    }
            return changes
    
        def _check_validity(self, instance):
            """Validity check (within range)"""
            within_range = (
                (instance >= self.feature_ranges['min']).all() and
                (instance <= self.feature_ranges['max']).all()
            )
            return within_range
    
    # Usage example
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Counterfactual Explainer
    cf_explainer = CounterfactualExplainer(model, X_train)
    
    # Try with test sample
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    original_pred = model.predict([sample])[0]
    target_class = 1 - original_pred
    
    print("=== Counterfactual Explanation ===")
    print(f"Original prediction: {data.target_names[original_pred]}")
    print(f"Target prediction: {data.target_names[target_class]}\n")
    
    counterfactual, changes, metadata = cf_explainer.find_minimal_counterfactual(
        sample, target_class, max_iterations=5000
    )
    
    if metadata['success']:
        print(f"✓ Counterfactual instance found")
        print(f"  Distance: {metadata['distance']:.4f}")
        print(f"  Number of changes: {metadata['n_changes']}")
        print(f"  Validity: {metadata['validity']}")
    
        cf_pred = model.predict([counterfactual])[0]
        print(f"  Prediction after counterfactual: {data.target_names[cf_pred]}")
    
        print(f"\nRequired changes (top 5):")
        sorted_changes = sorted(
            changes.items(),
            key=lambda x: abs(x[1]['absolute_change']),
            reverse=True
        )[:5]
    
        for idx, change_info in sorted_changes:
            feature_name = X.columns[idx]
            print(f"\n{feature_name}:")
            print(f"  Original: {change_info['original']:.2f}")
            print(f"  Changed to: {change_info['counterfactual']:.2f}")
            print(f"  Change: {change_info['absolute_change']:+.2f} "
                  f"({change_info['relative_change']:.1%})")
    
        # Validity evaluation
        print("\n=== Validity Evaluation ===")
        print("1. Feasibility: Within training data range")
        print(f"   → {metadata['validity']}")
    
        print("\n2. Minimality: Small number of changes")
        print(f"   → {metadata['n_changes']}/{len(sample)} features changed")
    
        print("\n3. Actionability: Are changes executable")
        print("   → Requires validation with domain knowledge")
        print("   (Example: Making age younger is impossible)")
    
        print("\n4. Proximity: Close to original instance")
        print(f"   → Normalized distance: {metadata['distance']:.4f}")
    
    else:
        print("✗ Counterfactual instance not found")
    
    print("\n=== Summary of Evaluation Criteria ===")
    print("""
    Validity evaluation of counterfactual explanations:
    
    1. **Feasibility**
       - Exists within training data distribution
       - Physically/logically possible values
    
    2. **Minimality**
       - Minimal number of features changed
       - Minimal magnitude of changes
    
    3. **Actionability**
       - Features that can actually be changed
       - Realistic cost
    
    4. **Proximity**
       - Close to original instance
       - Easy to interpret
    
    5. **Diversity**
       - Can present multiple solutions
       - User has choices
    """)
    

* * *

## References

  1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. _KDD_.
  2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. _NeurIPS_.
  3. Molnar, C. (2022). _Interpretable Machine Learning_ (2nd ed.). Available at: https://christophm.github.io/interpretable-ml-book/
  4. Goldstein, A., et al. (2015). Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation. _Journal of Computational and Graphical Statistics_.
  5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision Model-Agnostic Explanations. _AAAI_.
  6. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual Explanations without Opening the Black Box. _Harvard Journal of Law & Technology_.
  7. Breiman, L. (2001). Random Forests. _Machine Learning_ , 45(1), 5-32.
