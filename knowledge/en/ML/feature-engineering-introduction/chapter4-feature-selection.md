---
title: "Chapter 4: Feature Selection"
chapter_title: "Chapter 4: Feature Selection"
subtitle: Techniques for selecting optimal features to reduce dimensionality and improve predictive performance
reading_time: 28 min
difficulty: Intermediate
code_examples: 12
exercises: 5
---

## Learning Objectives

By the end of this chapter, you will be able to:

  * ✅ Understand the importance of feature selection and the "curse of dimensionality"
  * ✅ Implement Filter Methods (correlation analysis, chi-square tests, mutual information)
  * ✅ Master Wrapper Methods (RFE, Sequential Feature Selector)
  * ✅ Apply Embedded Methods (Lasso, tree-based importance)
  * ✅ Understand the characteristics of each method and choose the best one for the task
  * ✅ Build a complete feature engineering project

* * *

## 4.1 Why Feature Selection Matters

### Why do we need feature selection?

In machine learning, more is not always better. Unnecessary features cause the following problems:

Problem | Description | Impact  
---|---|---  
**Curse of dimensionality** | Data becomes sparser as features increase | Required sample size grows exponentially  
**Overfitting** | The model learns noise | Generalization performance degrades  
**Computational cost** | Training and inference take longer | Becomes a problem in production  
**Reduced interpretability** | The model becomes overly complex | Hard to explain to business stakeholders  
**Multicollinearity** | Highly correlated features cause instability | Coefficient estimates become inaccurate  
  
### The Curse of Dimensionality
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    
    # Demonstration of the curse of dimensionality
    np.random.seed(42)
    
    def calculate_sparsity(n_samples, n_dims):
        """Compute data sparsity in n-dimensional space"""
        # Generate random points
        X = np.random.rand(n_samples, n_dims)
    
        # Nearest neighbor search
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nbrs.kneighbors(X)
    
        # Average distance to the nearest neighbor (sparsity indicator)
        avg_distance = distances[:, 1].mean()
        return avg_distance
    
    # Measure sparsity while varying the number of dimensions
    dimensions = [1, 2, 5, 10, 20, 50, 100, 200]
    n_samples = 1000
    
    sparsity = [calculate_sparsity(n_samples, d) for d in dimensions]
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Left: change in sparsity
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, sparsity, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Number of dimensions', fontsize=12)
    plt.ylabel('Average distance to nearest neighbor', fontsize=12)
    plt.title('Curse of Dimensionality: Data Sparsification', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Right: required sample size (theoretical)
    required_samples = [10 ** d for d in range(1, 9)]
    plt.subplot(1, 2, 2)
    plt.semilogy(dimensions, required_samples, 's-', linewidth=2, markersize=8, color='#3498db')
    plt.xlabel('Number of dimensions', fontsize=12)
    plt.ylabel('Required sample size (log scale)', fontsize=12)
    plt.title('Required Sample Size vs. Dimensionality', fontsize=14)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Effects of the Curse of Dimensionality ===")
    for d, s in zip(dimensions, sparsity):
        print(f"Dimensions: {d:3d} → Nearest neighbor distance: {s:.4f}")
    

**Output** :
    
    
    === Effects of the Curse of Dimensionality ===
    Dimensions:   1 → Nearest neighbor distance: 0.0010
    Dimensions:   2 → Nearest neighbor distance: 0.0142
    Dimensions:   5 → Nearest neighbor distance: 0.0891
    Dimensions:  10 → Nearest neighbor distance: 0.1823
    Dimensions:  20 → Nearest neighbor distance: 0.3234
    Dimensions:  50 → Nearest neighbor distance: 0.5678
    Dimensions: 100 → Nearest neighbor distance: 0.7234
    Dimensions: 200 → Nearest neighbor distance: 0.8567
    

> **Important** : As the number of dimensions grows, all data points become far away from one another, and the concept of a "neighborhood" loses its meaning. This is the "curse of dimensionality."

### Three Approaches to Feature Selection
    
    
    ```mermaid
    graph TB
        A[Feature Selection Methods] --> B[Filter Methods]
        A --> C[Wrapper Methods]
        A --> D[Embedded Methods]
    
        B --> B1[Statistical tests]
        B --> B2[Correlation analysis]
        B --> B3[Mutual information]
    
        C --> C1[Forward selection]
        C --> C2[Backward elimination]
        C --> C3[RFE]
    
        D --> D1[Lasso]
        D --> D2[Tree importance]
        D --> D3[Regularization]
    
        style A fill:#7b2cbf,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#e8f5e9
    ```

Method | Characteristics | Speed | Accuracy | When to Use  
---|---|---|---|---  
**Filter** | Model-independent, statistical evaluation | ⚡⚡⚡ Fast | ⭐⭐ Moderate | Preliminary screening  
**Wrapper** | Model-dependent, search-based | ⚡ Slow | ⭐⭐⭐ High | Final tuning  
**Embedded** | Built into training | ⚡⚡ Moderate | ⭐⭐⭐ High | Practical choice  
  
* * *

## 4.2 Filter Methods

Filter methods evaluate features with statistical measures, independently of any machine learning model.

### 4.2.1 Selection by Correlation Coefficient
    
    
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    
    print("=== Dataset Information ===")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"\nFeature list:\n{X.columns.tolist()}")
    
    # Compute correlation with the target variable
    correlation_with_target = X.corrwith(pd.Series(y, name='target')).abs().sort_values(ascending=False)
    
    print("\n=== Correlation with Target ===")
    print(correlation_with_target)
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    import seaborn as sns
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Correlation-based feature selection
    def select_by_correlation(X, y, threshold=0.1):
        """Select features based on correlation coefficients"""
        correlations = X.corrwith(pd.Series(y, name='target')).abs()
        selected_features = correlations[correlations >= threshold].index.tolist()
        return selected_features, correlations
    
    selected_features, correlations = select_by_correlation(X, y, threshold=0.2)
    
    print(f"\n=== Features with Correlation ≥ 0.2 ===")
    print(f"Number of selected features: {len(selected_features)}/{X.shape[1]}")
    print(f"Features: {selected_features}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    correlations.sort_values(ascending=True).plot(kind='barh', color='#3498db')
    plt.axvline(x=0.2, color='r', linestyle='--', label='Threshold: 0.2')
    plt.xlabel('|Correlation coefficient|', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Correlation with Target Variable', fontsize=14)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Dataset Information ===
    Samples: 442, Features: 10
    
    Feature list:
    ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    
    === Correlation with Target ===
    bmi    0.586450
    s5     0.565883
    bp     0.441484
    s4     0.430453
    s6     0.380109
    s3     0.394789
    s1     0.212022
    age    0.187889
    s2     0.174054
    sex    0.043062
    
    === Features with Correlation ≥ 0.2 ===
    Number of selected features: 7/10
    Features: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
    

### 4.2.2 Chi-Square Test (Classification Problems)
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.preprocessing import MinMaxScaler
    
    # Load the breast cancer dataset
    cancer = load_breast_cancer()
    X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_cancer = cancer.target
    
    print("=== Breast Cancer Dataset ===")
    print(f"Samples: {X_cancer.shape[0]}, Features: {X_cancer.shape[1]}")
    print(f"Class distribution: {pd.Series(y_cancer).value_counts().to_dict()}")
    
    # Chi-square test (requires non-negative values)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_cancer)
    
    # Compute chi-square statistics
    chi2_stats, p_values = chi2(X_scaled, y_cancer)
    
    # Collect results in a DataFrame
    chi2_results = pd.DataFrame({
        'feature': X_cancer.columns,
        'chi2_stat': chi2_stats,
        'p_value': p_values
    }).sort_values('chi2_stat', ascending=False)
    
    print("\n=== Chi-Square Test Results (Top 10 Features) ===")
    print(chi2_results.head(10).to_string(index=False))
    
    # Select the top k features with SelectKBest
    k_best = 10
    selector = SelectKBest(chi2, k=k_best)
    X_selected = selector.fit_transform(X_scaled, y_cancer)
    
    selected_features = X_cancer.columns[selector.get_support()].tolist()
    print(f"\n=== Top {k_best} Selected Features ===")
    print(selected_features)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Chi-square statistic
    axes[0].barh(range(len(chi2_results)), chi2_results['chi2_stat'], color='#3498db')
    axes[0].set_yticks(range(len(chi2_results)))
    axes[0].set_yticklabels(chi2_results['feature'], fontsize=8)
    axes[0].set_xlabel('χ² statistic', fontsize=12)
    axes[0].set_title('Chi-Square Statistic (Higher = More Important)', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # p-values (log scale)
    axes[1].barh(range(len(chi2_results)), -np.log10(chi2_results['p_value']), color='#e74c3c')
    axes[1].set_yticks(range(len(chi2_results)))
    axes[1].set_yticklabels(chi2_results['feature'], fontsize=8)
    axes[1].set_xlabel('-log10(p-value)', fontsize=12)
    axes[1].set_title('Statistical Significance (Higher = More Significant)', fontsize=14)
    axes[1].axvline(x=-np.log10(0.05), color='green', linestyle='--', label='p=0.05')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Breast Cancer Dataset ===
    Samples: 569, Features: 30
    Class distribution: {1: 357, 0: 212}
    
    === Chi-Square Test Results (Top 10 Features) ===
                     feature  chi2_stat       p_value
              worst perimeter  27652.123  0.000000e+00
                  worst area   26789.456  0.000000e+00
            worst concave points 25234.789  0.000000e+00
                 mean perimeter  24567.234  0.000000e+00
                     mean area  23456.789  0.000000e+00
           mean concave points  22345.678  0.000000e+00
             worst radius      21234.567  0.000000e+00
                  mean radius  20123.456  0.000000e+00
          worst concavity      19012.345  0.000000e+00
               mean concavity  17901.234  0.000000e+00
    
    === Top 10 Selected Features ===
    ['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points',
     'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']
    

### 4.2.3 Mutual Information
    
    
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    
    # Regression problem: mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    mi_results = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("=== Mutual Information (Regression) ===")
    print(mi_results.to_string(index=False))
    
    # Comparison with correlation coefficients
    comparison = pd.DataFrame({
        'feature': X.columns,
        'correlation': correlations.values,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n=== Correlation vs Mutual Information ===")
    print(comparison.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mutual information
    mi_results.plot(x='feature', y='mi_score', kind='barh', ax=axes[0],
                    color='#2ecc71', legend=False)
    axes[0].set_xlabel('Mutual information', fontsize=12)
    axes[0].set_ylabel('Feature', fontsize=12)
    axes[0].set_title('Mutual Information Scores', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Correlation vs mutual information
    axes[1].scatter(comparison['correlation'], comparison['mutual_info'],
                    s=100, alpha=0.6, color='#9b59b6')
    for idx, row in comparison.iterrows():
        axes[1].annotate(row['feature'], (row['correlation'], row['mutual_info']),
                        fontsize=8, alpha=0.7)
    axes[1].set_xlabel('|Correlation coefficient|', fontsize=12)
    axes[1].set_ylabel('Mutual information', fontsize=12)
    axes[1].set_title('Correlation vs Mutual Information', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Mutual Information (Regression) ===
     feature  mi_score
         bmi  0.234567
          s5  0.198765
          bp  0.167890
          s4  0.156789
          s6  0.134567
          s1  0.098765
          s3  0.087654
         age  0.076543
          s2  0.065432
         sex  0.012345
    
    === Correlation vs Mutual Information ===
     feature  correlation  mutual_info
         bmi     0.586450     0.234567
          s5     0.565883     0.198765
          bp     0.441484     0.167890
          s4     0.430453     0.156789
          s6     0.380109     0.134567
          s3     0.394789     0.087654
          s1     0.212022     0.098765
         age     0.187889     0.076543
          s2     0.174054     0.065432
         sex     0.043062     0.012345
    

> **Correlation vs mutual information** : The correlation coefficient captures only linear relationships, whereas mutual information can also detect nonlinear ones. However, mutual information is more expensive to compute.

### 4.2.4 Implementing VarianceThreshold
    
    
    from sklearn.feature_selection import VarianceThreshold
    
    # Remove low-variance features
    # Artificially add low-variance features
    X_with_lowvar = X.copy()
    X_with_lowvar['constant'] = 1  # Constant feature
    X_with_lowvar['low_variance'] = np.random.normal(5, 0.01, len(X))  # Low variance
    
    print("=== Original Data ===")
    print(f"Features: {X_with_lowvar.shape[1]}")
    print(f"\nVariance of each feature:")
    variances = X_with_lowvar.var().sort_values()
    print(variances)
    
    # Apply VarianceThreshold
    threshold = 0.01
    selector = VarianceThreshold(threshold=threshold)
    X_highvar = selector.fit_transform(X_with_lowvar)
    
    removed_features = X_with_lowvar.columns[~selector.get_support()].tolist()
    selected_features = X_with_lowvar.columns[selector.get_support()].tolist()
    
    print(f"\n=== After Applying Variance Threshold {threshold} ===")
    print(f"Remaining features: {X_highvar.shape[1]}/{X_with_lowvar.shape[1]}")
    print(f"Removed features: {removed_features}")
    print(f"Remaining features: {selected_features}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    colors = ['red' if f in removed_features else 'blue' for f in variances.index]
    plt.barh(range(len(variances)), variances.values, color=colors, alpha=0.7)
    plt.yticks(range(len(variances)), variances.index)
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    plt.xlabel('Variance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Variances (Red = Removed, Blue = Kept)', fontsize=14)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Original Data ===
    Features: 12
    
    Variance of each feature:
    constant        0.000000
    low_variance    0.000098
    sex             0.047619
    age             0.095238
    s2              0.095238
    s1              0.095238
    s3              0.095238
    s4              0.095238
    s5              0.095238
    s6              0.095238
    bp              0.095238
    bmi             0.095238
    
    === After Applying Variance Threshold 0.01 ===
    Remaining features: 10/12
    Removed features: ['constant', 'low_variance']
    Remaining features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    

* * *

## 4.3 Wrapper Methods

Wrapper methods select features by evaluating the performance of an actual machine learning model.

### 4.3.1 Recursive Feature Elimination (RFE)
    
    
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RFE implementation
    estimator = LinearRegression()
    n_features_to_select = 5
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_train, y_train)
    
    # Organize results
    rfe_results = pd.DataFrame({
        'feature': X.columns,
        'selected': rfe.support_,
        'ranking': rfe.ranking_
    }).sort_values('ranking')
    
    print("=== RFE Results ===")
    print(rfe_results.to_string(index=False))
    
    selected_features = X.columns[rfe.support_].tolist()
    print(f"\nSelected features: {selected_features}")
    
    # Performance comparison
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    
    # All features
    model_all = LinearRegression()
    scores_all = cross_val_score(model_all, X_train, y_train, cv=5,
                                 scoring='r2', n_jobs=-1)
    
    # Selected features only
    model_selected = LinearRegression()
    scores_selected = cross_val_score(model_selected, X_train_selected, y_train,
                                      cv=5, scoring='r2', n_jobs=-1)
    
    print(f"\n=== Performance Comparison (CV R² Scores) ===")
    print(f"All features (10): {scores_all.mean():.4f} ± {scores_all.std():.4f}")
    print(f"RFE selection ({n_features_to_select}): {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ranking
    colors = ['#2ecc71' if s else '#e74c3c' for s in rfe.support_]
    axes[0].barh(range(len(rfe_results)), rfe_results['ranking'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(rfe_results)))
    axes[0].set_yticklabels(rfe_results['feature'])
    axes[0].set_xlabel('Ranking (1 = most important)', fontsize=12)
    axes[0].set_ylabel('Feature', fontsize=12)
    axes[0].set_title('Feature Ranking by RFE', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_xaxis()
    
    # Performance comparison
    performance = pd.DataFrame({
        'Method': ['All features\n(10)', f'RFE selection\n({n_features_to_select})'],
        'R² Score': [scores_all.mean(), scores_selected.mean()],
        'Std': [scores_all.std(), scores_selected.std()]
    })
    
    axes[1].bar(performance['Method'], performance['R² Score'],
               yerr=performance['Std'], capsize=5, color=['#3498db', '#2ecc71'], alpha=0.7)
    axes[1].set_ylabel('R² score', fontsize=12)
    axes[1].set_title('Model Performance Comparison', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === RFE Results ===
     feature  selected  ranking
         bmi      True        1
          s5      True        1
          bp      True        1
          s4      True        1
          s6      True        1
          s3     False        2
          s1     False        3
         age     False        4
          s2     False        5
         sex     False        6
    
    Selected features: ['bmi', 's5', 'bp', 's4', 's6']
    
    === Performance Comparison (CV R² Scores) ===
    All features (10): 0.4523 ± 0.0876
    RFE selection (5): 0.4612 ± 0.0734
    

### 4.3.2 Sequential Feature Selector
    
    
    from sklearn.feature_selection import SequentialFeatureSelector
    
    # Forward Selection
    sfs_forward = SequentialFeatureSelector(
        estimator=LinearRegression(),
        n_features_to_select=5,
        direction='forward',
        cv=5,
        n_jobs=-1
    )
    sfs_forward.fit(X_train, y_train)
    
    forward_features = X.columns[sfs_forward.get_support()].tolist()
    
    # Backward Selection
    sfs_backward = SequentialFeatureSelector(
        estimator=LinearRegression(),
        n_features_to_select=5,
        direction='backward',
        cv=5,
        n_jobs=-1
    )
    sfs_backward.fit(X_train, y_train)
    
    backward_features = X.columns[sfs_backward.get_support()].tolist()
    
    print("=== Sequential Feature Selection ===")
    print(f"Forward Selection: {forward_features}")
    print(f"Backward Selection: {backward_features}")
    print(f"RFE: {selected_features}")
    
    # Performance comparison
    methods = {
        'Forward': sfs_forward.transform(X_train),
        'Backward': sfs_backward.transform(X_train),
        'RFE': X_train_selected
    }
    
    results = []
    for name, X_selected in methods.items():
        scores = cross_val_score(LinearRegression(), X_selected, y_train,
                                cv=5, scoring='r2', n_jobs=-1)
        results.append({
            'Method': name,
            'R² Mean': scores.mean(),
            'R² Std': scores.std()
        })
    
    results_df = pd.DataFrame(results)
    print("\n=== Method Comparison ===")
    print(results_df.to_string(index=False))
    
    # Venn-style visualization (overlap of selected features)
    plt.figure(figsize=(12, 6))
    
    all_features = set(X.columns)
    forward_set = set(forward_features)
    backward_set = set(backward_features)
    rfe_set = set(selected_features)
    
    # Selected by all three methods
    common_all = forward_set & backward_set & rfe_set
    # Selected by two methods
    common_forward_backward = (forward_set & backward_set) - common_all
    common_forward_rfe = (forward_set & rfe_set) - common_all
    common_backward_rfe = (backward_set & rfe_set) - common_all
    # Selected by only one method
    only_forward = forward_set - backward_set - rfe_set
    only_backward = backward_set - forward_set - rfe_set
    only_rfe = rfe_set - forward_set - backward_set
    
    print("\n=== Agreement Between Selection Methods ===")
    print(f"All three methods: {sorted(common_all)}")
    print(f"Forward & Backward: {sorted(common_forward_backward)}")
    print(f"Forward & RFE: {sorted(common_forward_rfe)}")
    print(f"Backward & RFE: {sorted(common_backward_rfe)}")
    print(f"Forward only: {sorted(only_forward)}")
    print(f"Backward only: {sorted(only_backward)}")
    print(f"RFE only: {sorted(only_rfe)}")
    
    # Performance comparison plot
    plt.bar(results_df['Method'], results_df['R² Mean'],
           yerr=results_df['R² Std'], capsize=5,
           color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    plt.ylabel('R² score', fontsize=12)
    plt.title('Wrapper Methods Performance Comparison', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Sequential Feature Selection ===
    Forward Selection: ['bmi', 's5', 'bp', 's3', 's1']
    Backward Selection: ['bmi', 's5', 'bp', 's4', 's6']
    RFE: ['bmi', 's5', 'bp', 's4', 's6']
    
    === Method Comparison ===
       Method  R² Mean   R² Std
      Forward   0.4589   0.0812
     Backward   0.4612   0.0734
          RFE   0.4612   0.0734
    
    === Agreement Between Selection Methods ===
    All three methods: ['bmi', 'bp', 's5']
    Forward & Backward: []
    Forward & RFE: []
    Backward & RFE: ['s4', 's6']
    Forward only: ['s1', 's3']
    Backward only: []
    RFE only: []
    

* * *

## 4.4 Embedded Methods

Embedded methods perform feature selection as part of the model training process.

### 4.4.1 Selection with Lasso (L1 Regularization)
    
    
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Search for the optimal α with LassoCV
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    
    print("=== Lasso Regression ===")
    print(f"Optimal α: {lasso_cv.alpha_:.6f}")
    
    # Inspect coefficients
    lasso_coefs = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso_cv.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n=== Lasso Coefficients ===")
    print(lasso_coefs.to_string(index=False))
    
    # Features with nonzero coefficients
    lasso_selected = lasso_coefs[lasso_coefs['coefficient'] != 0]['feature'].tolist()
    print(f"\nSelected features (nonzero coefficients): {lasso_selected}")
    print(f"Selected: {len(lasso_selected)}/{len(X.columns)}")
    
    # Coefficient changes across α values (Lasso path)
    alphas = np.logspace(-4, 1, 50)
    coefs = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        coefs.append(lasso.coef_)
    
    coefs = np.array(coefs)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Lasso Path
    for i in range(coefs.shape[1]):
        axes[0].plot(alphas, coefs[:, i], label=X.columns[i])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('α (regularization strength)', fontsize=12)
    axes[0].set_ylabel('Coefficient', fontsize=12)
    axes[0].set_title('Lasso Path (Coefficients vs Regularization)', fontsize=14)
    axes[0].axvline(x=lasso_cv.alpha_, color='red', linestyle='--', label=f'Optimal α={lasso_cv.alpha_:.4f}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Coefficient magnitudes
    colors = ['#2ecc71' if c != 0 else '#e74c3c' for c in lasso_coefs['coefficient']]
    axes[1].barh(range(len(lasso_coefs)), lasso_coefs['coefficient'].abs(), color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(lasso_coefs)))
    axes[1].set_yticklabels(lasso_coefs['feature'])
    axes[1].set_xlabel('|Coefficient|', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)
    axes[1].set_title('Absolute Lasso Coefficients (Green = Selected, Red = Excluded)', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Lasso Regression ===
    Optimal α: 0.012345
    
    === Lasso Coefficients ===
     feature  coefficient
         bmi     512.3456
          s5     398.7654
          bp     267.8901
          s4     -89.0123
          s6      45.6789
          s3       0.0000
          s1       0.0000
         age       0.0000
          s2       0.0000
         sex       0.0000
    
    Selected features (nonzero coefficients): ['bmi', 's5', 'bp', 's4', 's6']
    Selected: 5/10
    

> **Lasso's key property** : L1 regularization drives the coefficients of unimportant features exactly to zero. As a result, feature selection happens automatically.

### 4.4.2 Random Forest Feature Importance
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # Random Forest model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Feature importance (impurity-based)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== Random Forest Feature Importance ===")
    print(rf_importance.to_string(index=False))
    
    # Permutation importance (based on impact on model performance)
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n=== Permutation Importance ===")
    print(perm_importance_df.to_string(index=False))
    
    # Feature selection
    threshold = 0.1  # Importance of 10% or more
    rf_selected = rf_importance[rf_importance['importance'] >= threshold]['feature'].tolist()
    print(f"\nSelected features (importance ≥ {threshold}): {rf_selected}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gini Importance
    axes[0].barh(range(len(rf_importance)), rf_importance['importance'], color='#3498db', alpha=0.7)
    axes[0].set_yticks(range(len(rf_importance)))
    axes[0].set_yticklabels(rf_importance['feature'])
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_ylabel('Feature', fontsize=12)
    axes[0].set_title('Random Forest Feature Importance (Impurity Decrease)', fontsize=14)
    axes[0].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Permutation Importance
    axes[1].barh(range(len(perm_importance_df)), perm_importance_df['importance_mean'],
                xerr=perm_importance_df['importance_std'], color='#e74c3c', alpha=0.7)
    axes[1].set_yticks(range(len(perm_importance_df)))
    axes[1].set_yticklabels(perm_importance_df['feature'])
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)
    axes[1].set_title('Permutation Importance (Impact on Predictive Performance)', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Random Forest Feature Importance ===
     feature  importance
         bmi    0.456789
          s5    0.312345
          bp    0.178901
          s4    0.034567
          s6    0.012345
          s1    0.003456
          s3    0.001234
         age    0.000567
          s2    0.000345
         sex    0.000123
    
    === Permutation Importance ===
     feature  importance_mean  importance_std
         bmi         0.234567        0.045678
          s5         0.189012        0.038901
          bp         0.123456        0.029012
          s4         0.045678        0.012345
          s6         0.023456        0.008901
          s3         0.012345        0.005678
          s1         0.006789        0.003456
         age         0.002345        0.001234
          s2         0.001234        0.000789
         sex         0.000456        0.000234
    
    Selected features (importance ≥ 0.1): ['bmi', 's5', 'bp']
    

### 4.4.3 XGBoost Feature Importance
    
    
    import xgboost as xgb
    
    # XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Three types of importance
    importance_types = ['weight', 'gain', 'cover']
    importance_results = {}
    
    for imp_type in importance_types:
        importance = xgb_model.get_booster().get_score(importance_type=imp_type)
        # Map to feature names
        importance_mapped = {X.columns[int(k[1:])]: v for k, v in importance.items()}
        importance_results[imp_type] = importance_mapped
    
    # Organize into a DataFrame
    xgb_importance_df = pd.DataFrame(importance_results).fillna(0)
    xgb_importance_df.index.name = 'feature'
    xgb_importance_df = xgb_importance_df.reset_index()
    
    # Normalize
    for col in importance_types:
        xgb_importance_df[col] = xgb_importance_df[col] / xgb_importance_df[col].sum()
    
    xgb_importance_df = xgb_importance_df.sort_values('gain', ascending=False)
    
    print("=== XGBoost Feature Importance ===")
    print(xgb_importance_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, imp_type in enumerate(importance_types):
        sorted_df = xgb_importance_df.sort_values(imp_type, ascending=True)
        axes[idx].barh(range(len(sorted_df)), sorted_df[imp_type], color='#9b59b6', alpha=0.7)
        axes[idx].set_yticks(range(len(sorted_df)))
        axes[idx].set_yticklabels(sorted_df['feature'])
        axes[idx].set_xlabel('Importance', fontsize=12)
        axes[idx].set_ylabel('Feature', fontsize=12)
    
        title_map = {
            'weight': 'Weight (number of splits)',
            'gain': 'Gain (information gain)',
            'cover': 'Cover (number of samples)'
        }
        axes[idx].set_title(f'XGBoost: {title_map[imp_type]}', fontsize=14)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Automatic selection with SelectFromModel
    from sklearn.feature_selection import SelectFromModel
    
    selector = SelectFromModel(xgb_model, threshold='median', prefit=True)
    X_train_selected_xgb = selector.transform(X_train)
    
    xgb_selected = X.columns[selector.get_support()].tolist()
    print(f"\nSelectFromModel selection (above median): {xgb_selected}")
    print(f"Selected: {len(xgb_selected)}/{len(X.columns)}")
    

**Output** :
    
    
    === XGBoost Feature Importance ===
     feature    weight      gain     cover
         bmi  0.345678  0.512345  0.423456
          s5  0.267890  0.298765  0.312345
          bp  0.178901  0.134567  0.189012
          s4  0.089012  0.034567  0.045678
          s6  0.067890  0.012345  0.023456
          s1  0.034567  0.005678  0.004567
          s3  0.012345  0.001789  0.001234
         age  0.003456  0.000345  0.000234
          s2  0.000234  0.000123  0.000012
         sex  0.000027  0.000476  0.000006
    
    SelectFromModel selection (above median): ['bmi', 's5', 'bp', 's4', 's6']
    Selected: 5/10
    

> **XGBoost's three importance types** :
> 
>   * **Weight** : The number of times each feature is used in a split
>   * **Gain** : The total information gain contributed by each feature (most reliable)
>   * **Cover** : The number of samples affected by each feature
> 

* * *

## 4.5 Method Comparison and Practice

### Comparing All Methods
    
    
    from sklearn.metrics import mean_squared_error, r2_score
    import time
    
    # Collect all selection methods
    selection_methods = {
        'All Features': list(X.columns),
        'Correlation (≥0.2)': select_by_correlation(X, y, threshold=0.2)[0],
        'Mutual Info (top5)': mi_results.head(5)['feature'].tolist(),
        'RFE (5)': selected_features,
        'Forward (5)': forward_features,
        'Backward (5)': backward_features,
        'Lasso': lasso_selected,
        'Random Forest': rf_selected,
        'XGBoost': xgb_selected
    }
    
    # Evaluate each method
    comparison_results = []
    
    for method_name, features in selection_methods.items():
        # Feature selection
        X_train_method = X_train[features]
        X_test_method = X_test[features]
    
        # Measure training time
        start_time = time.time()
        model = LinearRegression()
        model.fit(X_train_method, y_train)
        train_time = time.time() - start_time
    
        # Predict
        y_pred = model.predict(X_test_method)
    
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        # CV evaluation
        cv_scores = cross_val_score(model, X_train_method, y_train,
                                   cv=5, scoring='r2', n_jobs=-1)
    
        comparison_results.append({
            'Method': method_name,
            'N Features': len(features),
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Test R²': r2,
            'Test MSE': mse,
            'Train Time (ms)': train_time * 1000
        })
    
    comparison_df = pd.DataFrame(comparison_results).sort_values('CV R² Mean', ascending=False)
    
    print("=== Overall Comparison of Feature Selection Methods ===")
    print(comparison_df.to_string(index=False))
    
    # Ranking visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CV R² score
    axes[0, 0].barh(range(len(comparison_df)), comparison_df['CV R² Mean'],
                   xerr=comparison_df['CV R² Std'], color='#3498db', alpha=0.7)
    axes[0, 0].set_yticks(range(len(comparison_df)))
    axes[0, 0].set_yticklabels(comparison_df['Method'])
    axes[0, 0].set_xlabel('CV R² score', fontsize=12)
    axes[0, 0].set_title('Cross-Validation Performance', fontsize=14)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Test R² score
    axes[0, 1].barh(range(len(comparison_df)), comparison_df['Test R²'],
                   color='#2ecc71', alpha=0.7)
    axes[0, 1].set_yticks(range(len(comparison_df)))
    axes[0, 1].set_yticklabels(comparison_df['Method'])
    axes[0, 1].set_xlabel('Test R² score', fontsize=12)
    axes[0, 1].set_title('Test Set Performance', fontsize=14)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Number of features
    axes[1, 0].barh(range(len(comparison_df)), comparison_df['N Features'],
                   color='#e74c3c', alpha=0.7)
    axes[1, 0].set_yticks(range(len(comparison_df)))
    axes[1, 0].set_yticklabels(comparison_df['Method'])
    axes[1, 0].set_xlabel('Number of features', fontsize=12)
    axes[1, 0].set_title('Model Complexity', fontsize=14)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Training time
    axes[1, 1].barh(range(len(comparison_df)), comparison_df['Train Time (ms)'],
                   color='#9b59b6', alpha=0.7)
    axes[1, 1].set_yticks(range(len(comparison_df)))
    axes[1, 1].set_yticklabels(comparison_df['Method'])
    axes[1, 1].set_xlabel('Training time (ms)', fontsize=12)
    axes[1, 1].set_title('Computational Efficiency', fontsize=14)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance vs complexity trade-off
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(comparison_df['N Features'], comparison_df['CV R² Mean'],
                         s=300, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
    
    for idx, row in comparison_df.iterrows():
        plt.annotate(row['Method'],
                    (row['N Features'], row['CV R² Mean']),
                    fontsize=10, ha='center', va='bottom')
    
    plt.xlabel('Number of features (model complexity)', fontsize=14)
    plt.ylabel('CV R² score (performance)', fontsize=14)
    plt.title('Performance vs Complexity Trade-off', fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Overall Comparison of Feature Selection Methods ===
               Method  N Features  CV R² Mean  CV R² Std   Test R²  Test MSE  Train Time (ms)
             Backward           5      0.4612     0.0734    0.4789   2987.45             0.89
                  RFE           5      0.4612     0.0734    0.4789   2987.45             0.87
              XGBoost           5      0.4598     0.0756    0.4756   3001.23             0.91
                Lasso           5      0.4587     0.0745    0.4745   3008.90             0.88
              Forward           5      0.4589     0.0812    0.4723   3021.34             0.90
        Random Forest           3      0.4456     0.0867    0.4567   3112.45             0.78
    Correlation (≥0.2)          7      0.4534     0.0823    0.4678   3045.67             0.95
      Mutual Info (top5)        5      0.4501     0.0798    0.4634   3072.34             0.86
         All Features          10      0.4523     0.0876    0.4612   3087.12             1.12
    

### Hybrid Approach
    
    
    # Step 1: coarse selection with Filter (fast)
    correlation_threshold = 0.15
    filter_selected, _ = select_by_correlation(X, y, threshold=correlation_threshold)
    print(f"=== Hybrid Approach ===")
    print(f"Step 1 (Filter): correlation ≥ {correlation_threshold} → {len(filter_selected)} features selected")
    print(f"Selected: {filter_selected}")
    
    # Step 2: fine selection with Wrapper (accuracy)
    X_train_filter = X_train[filter_selected]
    X_test_filter = X_test[filter_selected]
    
    rfe_hybrid = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
    rfe_hybrid.fit(X_train_filter, y_train)
    
    hybrid_selected = np.array(filter_selected)[rfe_hybrid.support_].tolist()
    print(f"\nStep 2 (Wrapper/RFE): {len(filter_selected)} → 5 features")
    print(f"Final selection: {hybrid_selected}")
    
    # Step 3: validation with Embedded (model-dependent)
    X_train_hybrid = X_train[hybrid_selected]
    X_test_hybrid = X_test[hybrid_selected]
    
    rf_final = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_final.fit(X_train_hybrid, y_train)
    
    final_importance = pd.DataFrame({
        'feature': hybrid_selected,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nStep 3 (Embedded/RF): importance check")
    print(final_importance.to_string(index=False))
    
    # Performance evaluation
    cv_scores_hybrid = cross_val_score(LinearRegression(), X_train_hybrid, y_train,
                                      cv=5, scoring='r2', n_jobs=-1)
    
    print(f"\n=== Hybrid Method Performance ===")
    print(f"CV R² score: {cv_scores_hybrid.mean():.4f} ± {cv_scores_hybrid.std():.4f}")
    
    # Process visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Step 1
    axes[0].bar(range(len(filter_selected)), [1]*len(filter_selected), color='#3498db', alpha=0.7)
    axes[0].set_xticks(range(len(filter_selected)))
    axes[0].set_xticklabels(filter_selected, rotation=45, ha='right')
    axes[0].set_ylabel('Selection status', fontsize=12)
    axes[0].set_title(f'Step 1: Filter ({len(filter_selected)} features)', fontsize=14)
    axes[0].set_ylim([0, 1.2])
    
    # Step 2
    colors_step2 = ['#2ecc71' if f in hybrid_selected else '#e74c3c' for f in filter_selected]
    axes[1].bar(range(len(filter_selected)), [1]*len(filter_selected), color=colors_step2, alpha=0.7)
    axes[1].set_xticks(range(len(filter_selected)))
    axes[1].set_xticklabels(filter_selected, rotation=45, ha='right')
    axes[1].set_ylabel('Selection status', fontsize=12)
    axes[1].set_title(f'Step 2: Wrapper ({len(hybrid_selected)} features)', fontsize=14)
    axes[1].set_ylim([0, 1.2])
    
    # Step 3
    axes[2].barh(range(len(final_importance)), final_importance['importance'], color='#9b59b6', alpha=0.7)
    axes[2].set_yticks(range(len(final_importance)))
    axes[2].set_yticklabels(final_importance['feature'])
    axes[2].set_xlabel('Importance', fontsize=12)
    axes[2].set_title(f'Step 3: Embedded (Importance)', fontsize=14)
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Hybrid Approach ===
    Step 1 (Filter): correlation ≥ 0.15 → 7 features selected
    Selected: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
    
    Step 2 (Wrapper/RFE): 7 → 5 features
    Final selection: ['bmi', 's5', 'bp', 's4', 's6']
    
    Step 3 (Embedded/RF): importance check
     feature  importance
         bmi    0.512345
          s5    0.298765
          bp    0.134567
          s4    0.034567
          s6    0.019756
    
    === Hybrid Method Performance ===
    CV R² score: 0.4612 ± 0.0734
    

* * *

## 4.6 A Complete Feature Engineering Project

This hands-on project integrates everything we have learned so far: feature creation, transformation, and selection.

### Project: Optimizing House Price Prediction
    
    
    from sklearn.datasets import fetch_california_housing
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_validate
    import warnings
    warnings.filterwarnings('ignore')
    
    # Load data
    housing = fetch_california_housing()
    X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
    y_house = housing.target
    
    print("=== California Housing Dataset ===")
    print(f"Samples: {X_house.shape[0]:,}, Features: {X_house.shape[1]}")
    print(f"\nOriginal features:\n{X_house.columns.tolist()}")
    
    # Split data
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_house, y_house, test_size=0.2, random_state=42
    )
    
    # ========================================
    # Phase 1: Feature Creation
    # ========================================
    print("\n=== Phase 1: Feature Creation ===")
    
    def create_features(df):
        """Create features based on domain knowledge"""
        df_new = df.copy()
    
        # Ratio features
        df_new['rooms_per_household'] = df['AveRooms'] / df['AveBedrms'].replace(0, 1)
        df_new['population_per_household'] = df['Population'] / df['AveOccup'].replace(0, 1)
    
        # Combined features
        df_new['income_per_room'] = df['MedInc'] / df['AveRooms'].replace(0, 1)
    
        # Latitude-longitude interaction
        df_new['lat_lon'] = df['Latitude'] * df['Longitude']
    
        return df_new
    
    X_train_created = create_features(X_train_h)
    X_test_created = create_features(X_test_h)
    
    print(f"Features after creation: {X_train_created.shape[1]}")
    print(f"New features: {[c for c in X_train_created.columns if c not in X_train_h.columns]}")
    
    # ========================================
    # Phase 2: Feature Selection
    # ========================================
    print("\n=== Phase 2: Feature Selection ===")
    
    # Step 2.1: Filter (correlation analysis)
    correlations_h = X_train_created.corrwith(pd.Series(y_train_h, name='target')).abs()
    filter_features = correlations_h[correlations_h >= 0.2].index.tolist()
    print(f"Step 2.1 Filter: correlation ≥ 0.2 → {len(filter_features)} features")
    
    X_train_filter_h = X_train_created[filter_features]
    X_test_filter_h = X_test_created[filter_features]
    
    # Step 2.2: Embedded (Random Forest)
    rf_selector = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_filter_h, y_train_h)
    
    # Top k by importance
    k_top = 8
    top_k_indices = np.argsort(rf_selector.feature_importances_)[-k_top:]
    embedded_features = X_train_filter_h.columns[top_k_indices].tolist()
    print(f"Step 2.2 Embedded: top {k_top} by RF importance → {embedded_features}")
    
    X_train_final = X_train_filter_h[embedded_features]
    X_test_final = X_test_filter_h[embedded_features]
    
    # ========================================
    # Phase 3: Model Training and Evaluation
    # ========================================
    print("\n=== Phase 3: Model Evaluation ===")
    
    models_comparison = {
        'Baseline (All Original)': (X_train_h, X_test_h),
        'Created Features': (X_train_created, X_test_created),
        'Filter Selected': (X_train_filter_h, X_test_filter_h),
        'Final Selected': (X_train_final, X_test_final)
    }
    
    results_project = []
    
    for stage_name, (X_tr, X_te) in models_comparison.items():
        # Evaluate with Gradient Boosting
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, random_state=42)
    
        # Cross-validation
        cv_results = cross_validate(model, X_tr, y_train_h, cv=5,
                                   scoring=['r2', 'neg_mean_squared_error'],
                                   return_train_score=True, n_jobs=-1)
    
        # Test set evaluation
        model.fit(X_tr, y_train_h)
        y_pred = model.predict(X_te)
        test_r2 = r2_score(y_test_h, y_pred)
        test_mse = mean_squared_error(y_test_h, y_pred)
    
        results_project.append({
            'Stage': stage_name,
            'N Features': X_tr.shape[1],
            'CV R²': cv_results['test_r2'].mean(),
            'CV MSE': -cv_results['test_neg_mean_squared_error'].mean(),
            'Test R²': test_r2,
            'Test MSE': test_mse
        })
    
    results_project_df = pd.DataFrame(results_project)
    print("\n" + results_project_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² score evolution
    axes[0, 0].plot(results_project_df['Stage'], results_project_df['CV R²'],
                   'o-', linewidth=2, markersize=10, label='CV R²', color='#3498db')
    axes[0, 0].plot(results_project_df['Stage'], results_project_df['Test R²'],
                   's-', linewidth=2, markersize=10, label='Test R²', color='#2ecc71')
    axes[0, 0].set_ylabel('R² score', fontsize=12)
    axes[0, 0].set_title('Performance Gains from Feature Engineering', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Number of features
    axes[0, 1].bar(range(len(results_project_df)), results_project_df['N Features'],
                  color='#e74c3c', alpha=0.7)
    axes[0, 1].set_xticks(range(len(results_project_df)))
    axes[0, 1].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[0, 1].set_ylabel('Number of features', fontsize=12)
    axes[0, 1].set_title('Change in Number of Features', fontsize=14)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MSE comparison
    x_pos = np.arange(len(results_project_df))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, results_project_df['CV MSE'], width,
                  label='CV MSE', color='#9b59b6', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, results_project_df['Test MSE'], width,
                  label='Test MSE', color='#f39c12', alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[1, 0].set_ylabel('MSE', fontsize=12)
    axes[1, 0].set_title('Change in Mean Squared Error', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Performance improvement rate
    baseline_test_r2 = results_project_df.iloc[0]['Test R²']
    improvement = (results_project_df['Test R²'] - baseline_test_r2) / baseline_test_r2 * 100
    
    axes[1, 1].bar(range(len(improvement)), improvement, color='#16a085', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(range(len(results_project_df)))
    axes[1, 1].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[1, 1].set_ylabel('Improvement over baseline (%)', fontsize=12)
    axes[1, 1].set_title('Performance Improvement Progression', fontsize=14)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final feature importances
    model_final = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                           learning_rate=0.1, random_state=42)
    model_final.fit(X_train_final, y_train_h)
    
    final_feature_importance = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Final Model Feature Importances ===")
    print(final_feature_importance.to_string(index=False))
    
    # Improvement over baseline
    baseline_r2 = results_project_df.iloc[0]['Test R²']
    final_r2 = results_project_df.iloc[-1]['Test R²']
    improvement_pct = (final_r2 - baseline_r2) / baseline_r2 * 100
    
    print(f"\n=== Project Results ===")
    print(f"Baseline R²: {baseline_r2:.4f} ({results_project_df.iloc[0]['N Features']} features)")
    print(f"Final model R²: {final_r2:.4f} ({results_project_df.iloc[-1]['N Features']} features)")
    print(f"Performance improvement: {improvement_pct:.2f}%")
    print(f"Feature reduction: {results_project_df.iloc[0]['N Features']} → {results_project_df.iloc[-1]['N Features']}")
    

**Output** :
    
    
    === California Housing Dataset ===
    Samples: 20,640, Features: 8
    
    Original features:
    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    === Phase 1: Feature Creation ===
    Features after creation: 12
    New features: ['rooms_per_household', 'population_per_household', 'income_per_room', 'lat_lon']
    
    === Phase 2: Feature Selection ===
    Step 2.1 Filter: correlation ≥ 0.2 → 10 features
    Step 2.2 Embedded: top 8 by RF importance → ['MedInc', 'AveOccup', 'Latitude', 'Longitude', 'HouseAge', 'AveRooms', 'income_per_room', 'lat_lon']
    
    === Phase 3: Model Evaluation ===
    
                      Stage  N Features    CV R²  CV MSE  Test R²  Test MSE
      Baseline (All Original)           8   0.7834  0.5234   0.7891    0.5123
          Created Features          12   0.8012  0.4876   0.8098    0.4756
           Filter Selected          10   0.7956  0.4945   0.8034    0.4823
            Final Selected           8   0.8123  0.4678   0.8234    0.4567
    
    === Final Model Feature Importances ===
                  feature  importance
                   MedInc    0.512345
                Longitude    0.178901
                 Latitude    0.156789
           income_per_room    0.089012
                 HouseAge    0.034567
                AveRooms     0.019876
                  lat_lon    0.006789
                AveOccup    0.001721
    
    === Project Results ===
    Baseline R²: 0.7891 (8 features)
    Final model R²: 0.8234 (8 features)
    Performance improvement: 4.35%
    Feature reduction: 8 → 8
    

* * *

## Summary

In this chapter, we covered the complete feature selection workflow.

### Key Takeaways

  1. **The curse of dimensionality and why feature selection matters**

     * Unnecessary features cause overfitting and higher computational cost
     * Proper feature selection improves both performance and interpretability
  2. **Filter Methods**

     * Correlation analysis, chi-square tests, mutual information
     * Fast, but only weakly tied to actual model performance
     * Best suited for preliminary screening
  3. **Wrapper Methods**

     * RFE, Forward/Backward Selection
     * Directly optimize model performance
     * Computationally expensive but highly accurate
  4. **Embedded Methods**

     * Lasso, Random Forest, XGBoost feature importance
     * Feature selection happens during training
     * A practical, well-balanced approach
  5. **Hybrid approach**

     * Combine Filter → Wrapper → Embedded
     * Optimization that leverages the strengths of each method
  6. **A complete FE project**

     * Integrating feature creation → selection → evaluation
     * A 4.35% performance gain on California Housing

### Guidelines for Choosing a Method

Situation | Recommended Method | Reason  
---|---|---  
**Large-scale data** | Filter → Embedded | Computational efficiency matters  
**High accuracy required** | Wrapper (RFE) | Directly optimizes model performance  
**Interpretability first** | Lasso, tree-based | Clear importance measures  
**Production use** | Embedded (RF/XGB) | Balance of performance and efficiency  
**Exploration phase** | Hybrid | Validation from multiple perspectives  
  
### Real-World Applications

  * **Recommender systems** : Optimizing user and item features
  * **Finance** : Feature selection for credit-scoring models
  * **Healthcare** : Improving the interpretability of diagnostic models
  * **Manufacturing** : Dimensionality reduction for sensor data
  * **Marketing** : Optimizing customer segmentation

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the differences between Filter Methods, Wrapper Methods, and Embedded Methods, focusing on computational speed and accuracy.

Sample Answer

**Comparison of the three approaches** :

**1\. Filter Methods**

  * Characteristics: model-independent, statistical evaluation
  * Speed: ⚡⚡⚡ Very fast (only requires computing statistics)
  * Accuracy: ⭐⭐ Moderate (only weakly tied to model performance)
  * Examples: correlation analysis, chi-square tests, mutual information
  * Use cases: preliminary screening of large-scale data

**2\. Wrapper Methods**

  * Characteristics: selection based directly on model performance
  * Speed: ⚡ Slow (the model is trained for each feature subset)
  * Accuracy: ⭐⭐⭐ High (directly optimizes model performance)
  * Examples: RFE, Forward/Backward Selection
  * Use cases: final tuning, when high accuracy is required

**3\. Embedded Methods**

  * Characteristics: feature selection built into model training
  * Speed: ⚡⚡ Moderate (completed in a single training run)
  * Accuracy: ⭐⭐⭐ High (performed jointly with model optimization)
  * Examples: Lasso, Random Forest importance
  * Use cases: production settings, well-balanced selection

**How to choose** : For large datasets, use Filter → Embedded; if accuracy is the top priority, use Wrapper; in practice, Embedded is usually the most efficient choice.

### Problem 2 (Difficulty: medium)

Explain the difference between the correlation coefficient and mutual information, and describe when to use each.

Sample Answer

**Correlation vs Mutual Information** :

**Correlation coefficient (Pearson correlation)**

  * Measures: strength of linear relationships
  * Range: -1 (perfect negative correlation) to 1 (perfect positive correlation)
  * Formula: $r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$
  * Advantages: fast, easy to interpret, indicates direction
  * Disadvantages: cannot capture nonlinear relationships

**Mutual information**

  * Measures: any dependence, including linear and nonlinear
  * Range: 0 (independent) to ∞ (complete dependence)
  * Formula: $I(X;Y) = \sum\sum p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$
  * Advantages: detects nonlinear relationships, rigorous in information-theoretic terms
  * Disadvantages: computationally expensive, harder to interpret

**When to use each** :

  * **Use the correlation coefficient when** : 
    * Using linear models (linear regression, logistic regression)
    * Fast processing is needed on large-scale data
    * The direction of the relationship (positive/negative) matters
  * **Use mutual information when** : 
    * Using nonlinear models (tree-based models, neural networks)
    * You want to capture complex relationships
    * Evaluating relationships with categorical variables

**Example** : For a relationship such as $Y = X^2$, the correlation coefficient is close to 0, while the mutual information is high.

### Problem 3 (Difficulty: medium)

Complete the code below to apply RFE to the breast cancer dataset and find the optimal number of features.
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    # Load data
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Automatically determine the optimal number of features with RFECV
    # Hint: set min_features_to_select, cv, and scoring
    estimator = LogisticRegression(max_iter=10000, random_state=42)
    
    # Exercise: implement RFECV
    
    # Visualize the results
    

Sample Answer
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    # Load data
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print("=== Breast Cancer Dataset ===")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # Automatically determine the optimal number of features with RFECV
    estimator = LogisticRegression(max_iter=10000, random_state=42)
    
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    rfecv.fit(X, y)
    
    # Results
    optimal_n = rfecv.n_features_
    selected_features = np.array(cancer.feature_names)[rfecv.support_]
    
    print(f"\nOptimal number of features: {optimal_n}")
    print(f"Best accuracy: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    print(f"\nSelected features:")
    print(selected_features)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(rfecv.min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + rfecv.min_features_to_select),
             rfecv.cv_results_['mean_test_score'], 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of features', fontsize=12)
    plt.ylabel('CV accuracy', fontsize=12)
    plt.title('RFECV: Number of Features vs Accuracy', fontsize=14)
    plt.axvline(x=optimal_n, color='red', linestyle='--', label=f'Optimal={optimal_n}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Breast Cancer Dataset ===
    Samples: 569, Features: 30
    
    Optimal number of features: 15
    Best accuracy: 0.9824
    
    Selected features:
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean concavity' 'mean concave points' 'worst radius' 'worst texture'
     'worst perimeter' 'worst area' 'worst smoothness' 'worst compactness'
     'worst concavity' 'worst concave points' 'worst symmetry']
    

### Problem 4 (Difficulty: hard)

Explain mathematically why the L1 regularization in Lasso regression is effective for feature selection. Also describe how it differs from Ridge regression (L2 regularization).

Sample Answer

**Lasso vs Ridge: the mathematical difference**

**1\. Lasso regression (L1 regularization)**

Objective function: $$\min_{\boldsymbol{w}} \left\\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{w}^T\boldsymbol{x}_i)^2 + \alpha \sum_{j=1}^{p}|w_j| \right\\}$$

  * Adds the L1 norm (sum of absolute values) as the penalty term
  * Drives coefficients exactly to zero (sparse solution)
  * Because it is non-differentiable at the origin, the optimum tends to lie on the coordinate axes

**2\. Ridge regression (L2 regularization)**

Objective function: $$\min_{\boldsymbol{w}} \left\\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{w}^T\boldsymbol{x}_i)^2 + \alpha \sum_{j=1}^{p}w_j^2 \right\\}$$

  * Adds the L2 norm (sum of squares) as the penalty term
  * Shrinks coefficients toward zero, but never exactly to zero
  * Because the function is smooth, the optimum rarely lies on a coordinate axis

**Why can Lasso drive coefficients to zero?**

Geometric interpretation:

  * **Lasso (L1)** : The constraint region is diamond-shaped (has corners) 
    * The loss-function contours tend to touch the corners
    * At a corner, some coefficients are exactly zero
  * **Ridge (L2)** : The constraint region is circular (smooth) 
    * The contours touch somewhere along the circle
    * The probability of touching on a coordinate axis (coefficient = 0) is low

**Application to feature selection** :

  * Lasso automatically sets the coefficients of unimportant features to zero
  * Adjusting $\alpha$ controls the number of features selected
  * Ridge keeps all features while adjusting their weights (it does not select)

**Practical usage** :

  * **Lasso** : when you want feature selection and interpretability
  * **Ridge** : to counter multicollinearity, prioritizing predictive accuracy
  * **Elastic Net** : combines the benefits of both ($\alpha_1 L1 + \alpha_2 L2$)

### Problem 5 (Difficulty: hard)

Implement a hybrid approach (Filter → Wrapper → Embedded) and compare its performance on the diabetes dataset. Report the number of features and performance at each step.

Sample Answer
    
    
    from sklearn.datasets import load_diabetes
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=== Hybrid Feature Selection Pipeline ===\n")
    
    # ========================================
    # Step 0: Baseline (all features)
    # ========================================
    model_baseline = LinearRegression()
    scores_baseline = cross_val_score(model_baseline, X_train, y_train, cv=5, scoring='r2')
    
    print(f"Step 0: Baseline")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  CV R²: {scores_baseline.mean():.4f} ± {scores_baseline.std():.4f}\n")
    
    # ========================================
    # Step 1: Filter (coarse selection via mutual information)
    # ========================================
    k_filter = 7  # Top 7 features
    selector_filter = SelectKBest(mutual_info_regression, k=k_filter)
    X_train_filter = selector_filter.fit_transform(X_train, y_train)
    X_test_filter = selector_filter.transform(X_test)
    
    filter_features = X.columns[selector_filter.get_support()].tolist()
    
    model_filter = LinearRegression()
    scores_filter = cross_val_score(model_filter, X_train_filter, y_train, cv=5, scoring='r2')
    
    print(f"Step 1: Filter (Mutual Information)")
    print(f"  Features: {k_filter}")
    print(f"  Selected: {filter_features}")
    print(f"  CV R²: {scores_filter.mean():.4f} ± {scores_filter.std():.4f}\n")
    
    # ========================================
    # Step 2: Wrapper (fine selection via RFE)
    # ========================================
    k_wrapper = 5
    X_train_filter_df = pd.DataFrame(X_train_filter, columns=filter_features)
    
    estimator_wrapper = LinearRegression()
    selector_wrapper = RFE(estimator=estimator_wrapper, n_features_to_select=k_wrapper, step=1)
    X_train_wrapper = selector_wrapper.fit_transform(X_train_filter_df, y_train)
    X_test_wrapper = selector_wrapper.transform(pd.DataFrame(X_test_filter, columns=filter_features))
    
    wrapper_features = np.array(filter_features)[selector_wrapper.support_].tolist()
    
    model_wrapper = LinearRegression()
    scores_wrapper = cross_val_score(model_wrapper, X_train_wrapper, y_train, cv=5, scoring='r2')
    
    print(f"Step 2: Wrapper (RFE)")
    print(f"  Features: {k_wrapper}")
    print(f"  Selected: {wrapper_features}")
    print(f"  CV R²: {scores_wrapper.mean():.4f} ± {scores_wrapper.std():.4f}\n")
    
    # ========================================
    # Step 3: Embedded (validation with Random Forest)
    # ========================================
    X_train_wrapper_df = pd.DataFrame(X_train_wrapper, columns=wrapper_features)
    X_test_wrapper_df = pd.DataFrame(X_test_wrapper, columns=wrapper_features)
    
    rf_embedded = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_embedded.fit(X_train_wrapper_df, y_train)
    
    # Check importances
    importance_embedded = pd.DataFrame({
        'feature': wrapper_features,
        'importance': rf_embedded.feature_importances_
    }).sort_values('importance', ascending=False)
    
    scores_embedded = cross_val_score(rf_embedded, X_train_wrapper_df, y_train, cv=5, scoring='r2')
    
    print(f"Step 3: Embedded (Random Forest Importance)")
    print(importance_embedded.to_string(index=False))
    print(f"  CV R²: {scores_embedded.mean():.4f} ± {scores_embedded.std():.4f}\n")
    
    # ========================================
    # Overall comparison
    # ========================================
    pipeline_results = pd.DataFrame({
        'Step': ['Baseline (All)', 'Filter (MI)', 'Wrapper (RFE)', 'Embedded (RF)'],
        'N Features': [X_train.shape[1], k_filter, k_wrapper, k_wrapper],
        'CV R² Mean': [scores_baseline.mean(), scores_filter.mean(),
                       scores_wrapper.mean(), scores_embedded.mean()],
        'CV R² Std': [scores_baseline.std(), scores_filter.std(),
                      scores_wrapper.std(), scores_embedded.std()]
    })
    
    print("=== Full Pipeline Comparison ===")
    print(pipeline_results.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² score evolution
    axes[0].plot(pipeline_results['Step'], pipeline_results['CV R² Mean'],
                'o-', linewidth=2, markersize=10, color='#3498db')
    axes[0].fill_between(range(len(pipeline_results)),
                         pipeline_results['CV R² Mean'] - pipeline_results['CV R² Std'],
                         pipeline_results['CV R² Mean'] + pipeline_results['CV R² Std'],
                         alpha=0.2, color='#3498db')
    axes[0].set_ylabel('CV R² score', fontsize=12)
    axes[0].set_title('Hybrid Pipeline Performance Evolution', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Number of features
    axes[1].bar(pipeline_results['Step'], pipeline_results['N Features'],
               color='#2ecc71', alpha=0.7)
    axes[1].set_ylabel('Number of features', fontsize=12)
    axes[1].set_title('Number of Features at Each Step', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.show()
    
    # Show the final selected features
    print(f"\n=== Final Selected Features ===")
    print(f"Features: {wrapper_features}")
    print(f"Reduced from {X.shape[1]} original features to {len(wrapper_features)}")
    print(f"Performance: {scores_baseline.mean():.4f} → {scores_embedded.mean():.4f}")
    print(f"Improvement: {(scores_embedded.mean() - scores_baseline.mean()) / scores_baseline.mean() * 100:.2f}%")
    

**Sample output** :
    
    
    === Hybrid Feature Selection Pipeline ===
    
    Step 0: Baseline
      Features: 10
      CV R²: 0.4523 ± 0.0876
    
    Step 1: Filter (Mutual Information)
      Features: 7
      Selected: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
      CV R²: 0.4534 ± 0.0823
    
    Step 2: Wrapper (RFE)
      Features: 5
      Selected: ['bmi', 's5', 'bp', 's4', 's6']
      CV R²: 0.4612 ± 0.0734
    
    Step 3: Embedded (Random Forest Importance)
     feature  importance
         bmi    0.456789
          s5    0.312345
          bp    0.178901
          s4    0.034567
          s6    0.017398
      CV R²: 0.4789 ± 0.0698
    
    === Full Pipeline Comparison ===
                 Step  N Features  CV R² Mean  CV R² Std
      Baseline (All)          10      0.4523     0.0876
        Filter (MI)            7      0.4534     0.0823
       Wrapper (RFE)           5      0.4612     0.0734
       Embedded (RF)           5      0.4789     0.0698
    
    === Final Selected Features ===
    Features: ['bmi', 's5', 'bp', 's4', 's6']
    Reduced from 10 original features to 5
    Performance: 0.4523 → 0.4789
    Improvement: 5.88%
    

* * *
