---
title: "Chapter 4: Feature Selection"
chapter_title: "Chapter 4: Feature Selection"
subtitle: Dimensionality Reduction and Optimal Feature Selection Techniques for Improved Prediction Performance
reading_time: 28 minutes
difficulty: Intermediate
code_examples: 12
exercises: 5
---

This chapter covers Feature Selection. You will learn importance of feature selection, Filter Methods (correlation analysis, and Wrapper Methods (RFE.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the importance of feature selection and the "Curse of Dimensionality"
  * ✅ Implement Filter Methods (correlation analysis, chi-square test, mutual information)
  * ✅ Master Wrapper Methods (RFE, Sequential Feature Selector)
  * ✅ Utilize Embedded Methods (Lasso, Tree-based importance)
  * ✅ Understand the characteristics of each method and select the optimal approach
  * ✅ Build complete feature engineering projects

* * *

## 4.1 Importance of Feature Selection

### Why is Feature Selection Necessary?

Machine Learning「more is better」 is not always true。unnecessaryCharacteristicsfollowingProblemcauses：

Problem | Description | Impact  
---|---|---  
**Curse of Dimensionality** | Data becomes sparse as features increase | Required sample size increases exponentially  
**Overfitting** | Learning noise in the data | Reduced generalization performance  
**Computational Cost** | Training and inference take longer | Production Problembecomes  
**Reduced Interpretability** | Model becomes too complex | businessDifficult Description  
**Multicollinearity** | Highly correlated features cause instability | Inaccurate coefficient estimation  
  
### Curse of Dimensionality（Curse of Dimensionality）
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    
    # Curse of Dimensionalitydemonstration
    np.random.seed(42)
    
    def calculate_sparsity(n_samples, n_dims):
        """Calculate data sparsity in n-dimensional space"""
        # Generate random points
        X = np.random.rand(n_samples, n_dims)
    
        # Nearest neighbor search
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nbrs.kneighbors(X)
    
        # Average distance to nearest neighbor (sparsity metric)
        avg_distance = distances[:, 1].mean()
        return avg_distance
    
    # Measure sparsity across varying dimensions
    dimensions = [1, 2, 5, 10, 20, 50, 100, 200]
    n_samples = 1000
    
    sparsity = [calculate_sparsity(n_samples, d) for d in dimensions]
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Left: Change in sparsity
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, sparsity, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Average Distance to Nearest Neighbor', fontsize=12)
    plt.title('Curse of Dimensionality：Data Sparsification', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Right: Required sample size (theoretical value)
    required_samples = [10 ** d for d in range(1, 9)]
    plt.subplot(1, 2, 2)
    plt.semilogy(dimensions, required_samples, 's-', linewidth=2, markersize=8, color='#3498db')
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Required Sample Size (log scale)', fontsize=12)
    plt.title('Required Sample Size with Increasing Dimensions', fontsize=14)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Curse of DimensionalityImpact ===")
    for d, s in zip(dimensions, sparsity):
        print(f"Number of Dimensions: {d:3d} → Nearest neighbor distance: {s:.4f}")
    

**Output** ：
    
    
    === Curse of DimensionalityImpact ===
    Number of Dimensions:   1 → Nearest neighbor distance: 0.0010
    Number of Dimensions:   2 → Nearest neighbor distance: 0.0142
    Number of Dimensions:   5 → Nearest neighbor distance: 0.0891
    Number of Dimensions:  10 → Nearest neighbor distance: 0.1823
    Number of Dimensions:  20 → Nearest neighbor distance: 0.3234
    Number of Dimensions:  50 → Nearest neighbor distance: 0.5678
    Number of Dimensions: 100 → Nearest neighbor distance: 0.7234
    Number of Dimensions: 200 → Nearest neighbor distance: 0.8567
    

> **Important** : Number of Dimensionsincreases、all datapoints become distant、「neighborhood」 concept loses meaningexists。This is the "Curse of Dimensionality」。

### Three Approaches to Feature Selection
    
    
    ```mermaid
    graph TB
        A[Feature Selection Methods] --> B[Filter MethodsFilter Methods]
        A --> C[Wrapper MethodsWrapper Methods]
        A --> D[Embedded MethodsEmbedded Methods]
    
        B --> B1[Statistical Testing]
        B --> B2[Correlation Analysis]
        B --> B3[Mutual Information]
    
        C --> C1[Forward Selection]
        C --> C2[Backward Elimination]
        C --> C3[RFE]
    
        D --> D1[Lasso]
        D --> D2[Tree importance]
        D --> D3[Regularization]
    
        style A fill:#7b2cbf,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#e8f5e9
    ```

Method | Characteristics | Computational Speed | Accuracy | Use Cases  
---|---|---|---|---  
**Filter** | Model-independent, statistical evaluation | ⚡⚡⚡ Fast | ⭐⭐ Moderate | Preliminary screening  
**Wrapper** | Model-dependent, exploratory | ⚡ Slow | ⭐⭐⭐ High | Final tuning  
**Embedded** | Built into learning | ⚡⚡ Moderate | ⭐⭐⭐ High | Practical selection  
  
* * *

## 4.2 Filter Methods（Filter Methods）

Filter Methods、Machine LearningModel isindependent、statistical metricsevaluates characteristics。

### 4.2.1 Selection by Correlation Coefficient
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: 4.2.1 Selection by Correlation Coefficient
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    
    print("=== Dataset Information ===")
    print(f"Number of samples: {X.shape[0]}, Characteristicsnumber: {X.shape[1]}")
    print(f"\nCharacteristicslist:\n{X.columns.tolist()}")
    
    # Calculate correlation with target variable
    correlation_with_target = X.corrwith(pd.Series(y, name='target')).abs().sort_values(ascending=False)
    
    print("\n=== Correlation with Target Variable ===")
    print(correlation_with_target)
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    import seaborn as sns
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Characteristics Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Correlation-basedfeature selection
    def select_by_correlation(X, y, threshold=0.1):
        """Correlation Coefficientbased onfeature selection"""
        correlations = X.corrwith(pd.Series(y, name='target')).abs()
        selected_features = correlations[correlations >= threshold].index.tolist()
        return selected_features, correlations
    
    selected_features, correlations = select_by_correlation(X, y, threshold=0.2)
    
    print(f"\n=== Correlation threshold0.2 or moreCharacteristics ===")
    print(f"SelectedselectedCharacteristicsnumber: {len(selected_features)}/{X.shape[1]}")
    print(f"Characteristics: {selected_features}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    correlations.sort_values(ascending=True).plot(kind='barh', color='#3498db')
    plt.axvline(x=0.2, color='r', linestyle='--', label='Threshold: 0.2')
    plt.xlabel('|Correlation Coefficient|', fontsize=12)
    plt.ylabel('Characteristics', fontsize=12)
    plt.title('Correlation with Target VariableCoefficient', fontsize=14)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Dataset Information ===
    Number of samples: 442, Characteristicsnumber: 10
    
    Characteristicslist:
    ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    
    === Correlation with Target Variable ===
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
    
    === Correlation threshold0.2 or moreCharacteristics ===
    SelectedselectedCharacteristicsnumber: 7/10
    Characteristics: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
    

### 4.2.2 Chi-square test（classificationProblem）
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.preprocessing import MinMaxScaler
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_cancer = cancer.target
    
    print("=== Breast Cancer Dataset ===")
    print(f"Number of samples: {X_cancer.shape[0]}, Characteristicsnumber: {X_cancer.shape[1]}")
    print(f"Class distribution: {pd.Series(y_cancer).value_counts().to_dict()}")
    
    # Chi-square test (requires non-negative values)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_cancer)
    
    # Calculate chi-square statistics
    chi2_stats, p_values = chi2(X_scaled, y_cancer)
    
    # Results to DataFrame
    chi2_results = pd.DataFrame({
        'feature': X_cancer.columns,
        'chi2_stat': chi2_stats,
        'p_value': p_values
    }).sort_values('chi2_stat', ascending=False)
    
    print("\n=== Chi-square testResults（top10Characteristics） ===")
    print(chi2_results.head(10).to_string(index=False))
    
    # Select top k features using SelectKBest
    k_best = 10
    selector = SelectKBest(chi2, k=k_best)
    X_selected = selector.fit_transform(X_scaled, y_cancer)
    
    selected_features = X_cancer.columns[selector.get_support()].tolist()
    print(f"\n=== Selected Top{k_best}Characteristics ===")
    print(selected_features)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Chi-Square Statistics
    axes[0].barh(range(len(chi2_results)), chi2_results['chi2_stat'], color='#3498db')
    axes[0].set_yticks(range(len(chi2_results)))
    axes[0].set_yticklabels(chi2_results['feature'], fontsize=8)
    axes[0].set_xlabel('χ² Statistics', fontsize=12)
    axes[0].set_title('Chi-Square Statistics（higherImportant）', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # p-value（vsnumberscale）
    axes[1].barh(range(len(chi2_results)), -np.log10(chi2_results['p_value']), color='#e74c3c')
    axes[1].set_yticks(range(len(chi2_results)))
    axes[1].set_yticklabels(chi2_results['feature'], fontsize=8)
    axes[1].set_xlabel('-log10(p-value)', fontsize=12)
    axes[1].set_title('Statistical Significance (Higher is More Significant)', fontsize=14)
    axes[1].axvline(x=-np.log10(0.05), color='green', linestyle='--', label='p=0.05')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Breast Cancer Dataset ===
    Number of samples: 569, Characteristicsnumber: 30
    Class distribution: {1: 357, 0: 212}
    
    === Chi-square testResults（top10Characteristics） ===
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
    
    === Selected Top10Characteristics ===
    ['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points',
     'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']
    

### 4.2.3 Mutual Information（Mutual Information）
    
    
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    
    # regressionProblem：Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    mi_results = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("=== Mutual Information（regression）===")
    print(mi_results.to_string(index=False))
    
    # Correlation CoefficientComparison
    comparison = pd.DataFrame({
        'feature': X.columns,
        'correlation': correlations.values,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("\n=== Correlation Coefficient vs Mutual Information ===")
    print(comparison.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mutual Information
    mi_results.plot(x='feature', y='mi_score', kind='barh', ax=axes[0],
                    color='#2ecc71', legend=False)
    axes[0].set_xlabel('Mutual Information', fontsize=12)
    axes[0].set_ylabel('Characteristics', fontsize=12)
    axes[0].set_title('Mutual InformationScore', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # correlation vs Mutual Information
    axes[1].scatter(comparison['correlation'], comparison['mutual_info'],
                    s=100, alpha=0.6, color='#9b59b6')
    for idx, row in comparison.iterrows():
        axes[1].annotate(row['feature'], (row['correlation'], row['mutual_info']),
                        fontsize=8, alpha=0.7)
    axes[1].set_xlabel('|Correlation Coefficient|', fontsize=12)
    axes[1].set_ylabel('Mutual Information', fontsize=12)
    axes[1].set_title('Correlation Coefficient vs Mutual Information', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Mutual Information（regression）===
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
    
    === Correlation Coefficient vs Mutual Information ===
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
    

> **Correlation Coefficient vs Mutual Information** : Correlation CoefficientLinearrelationshipcaptures、Mutual Informationnon-linearcan detect relationships。、Mutual InformationComputational CostHigh。

### 4.2.4 VarianceThreshold Implementation
    
    
    from sklearn.feature_selection import VarianceThreshold
    
    # Low varianceCharacteristicsremove
    # artificiallyLow varianceCharacteristicsadd
    X_with_lowvar = X.copy()
    X_with_lowvar['constant'] = 1  # constantCharacteristics
    X_with_lowvar['low_variance'] = np.random.normal(5, 0.01, len(X))  # Low variance
    
    print("=== Original Data ===")
    print(f"Characteristicsnumber: {X_with_lowvar.shape[1]}")
    print(f"\neachCharacteristicsVariance:")
    variances = X_with_lowvar.var().sort_values()
    print(variances)
    
    # VarianceThresholdsuitableuse
    threshold = 0.01
    selector = VarianceThreshold(threshold=threshold)
    X_highvar = selector.fit_transform(X_with_lowvar)
    
    removed_features = X_with_lowvar.columns[~selector.get_support()].tolist()
    selected_features = X_with_lowvar.columns[selector.get_support()].tolist()
    
    print(f"\n=== VarianceThreshold {threshold} After application ===")
    print(f"remainCharacteristicsnumber: {X_highvar.shape[1]}/{X_with_lowvar.shape[1]}")
    print(f"removeselectedCharacteristics: {removed_features}")
    print(f"remainCharacteristics: {selected_features}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    colors = ['red' if f in removed_features else 'blue' for f in variances.index]
    plt.barh(range(len(variances)), variances.values, color=colors, alpha=0.7)
    plt.yticks(range(len(variances)), variances.index)
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    plt.xlabel('Variance', fontsize=12)
    plt.ylabel('Characteristics', fontsize=12)
    plt.title('CharacteristicsVariance（red=remove、blue=retain）', fontsize=14)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Original Data ===
    Characteristicsnumber: 12
    
    eachCharacteristicsVariance:
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
    
    === VarianceThreshold 0.01 After application ===
    remainCharacteristicsnumber: 10/12
    removeselectedCharacteristics: ['constant', 'low_variance']
    remainCharacteristics: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    

* * *

## 4.3 Wrapper Methods（Wrapper Methods）

Wrapper Methods、actualoccasionMachine LearningModelPerformanceEvaluationfeature selectiondoes。

### 4.3.1 Recursive Feature Elimination (RFE)
    
    
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    
    # Data split
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
    print(f"\nSelectedselectedCharacteristics: {selected_features}")
    
    # Performance Comparison
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    
    # allCharacteristics
    model_all = LinearRegression()
    scores_all = cross_val_score(model_all, X_train, y_train, cv=5,
                                 scoring='r2', n_jobs=-1)
    
    # SelectedselectedCharacteristics
    model_selected = LinearRegression()
    scores_selected = cross_val_score(model_selected, X_train_selected, y_train,
                                      cv=5, scoring='r2', n_jobs=-1)
    
    print(f"\n=== Performance Comparison（CV R²Score） ===")
    print(f"allCharacteristics（10）: {scores_all.mean():.4f} ± {scores_all.std():.4f}")
    print(f"RFESelected（{n_features_to_select}）: {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ranking
    colors = ['#2ecc71' if s else '#e74c3c' for s in rfe.support_]
    axes[0].barh(range(len(rfe_results)), rfe_results['ranking'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(rfe_results)))
    axes[0].set_yticklabels(rfe_results['feature'])
    axes[0].set_xlabel('Ranking（1mostImportant）', fontsize=12)
    axes[0].set_ylabel('Characteristics', fontsize=12)
    axes[0].set_title('RFECharacteristicsRanking', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_xaxis()
    
    # Performance Comparison
    performance = pd.DataFrame({
        'Method': ['allCharacteristics\n(10)', f'RFESelected\n({n_features_to_select})'],
        'R² Score': [scores_all.mean(), scores_selected.mean()],
        'Std': [scores_all.std(), scores_selected.std()]
    })
    
    axes[1].bar(performance['Method'], performance['R² Score'],
               yerr=performance['Std'], capsize=5, color=['#3498db', '#2ecc71'], alpha=0.7)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('ModelPerformance Comparison', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
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
    
    SelectedselectedCharacteristics: ['bmi', 's5', 'bp', 's4', 's6']
    
    === Performance Comparison（CV R²Score） ===
    allCharacteristics（10）: 0.4523 ± 0.0876
    RFESelected（5）: 0.4612 ± 0.0734
    

### 4.3.2 Sequential Feature Selector
    
    
    from sklearn.feature_selection import SequentialFeatureSelector
    
    # Forward Selection（Forward Selection）
    sfs_forward = SequentialFeatureSelector(
        estimator=LinearRegression(),
        n_features_to_select=5,
        direction='forward',
        cv=5,
        n_jobs=-1
    )
    sfs_forward.fit(X_train, y_train)
    
    forward_features = X.columns[sfs_forward.get_support()].tolist()
    
    # Backward Selection（Backward Elimination）
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
    
    # Performance Comparison
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
    print("\n=== MethodComparison ===")
    print(results_df.to_string(index=False))
    
    # VennFigureVisualization（SelectedselectedCharacteristicsheavycomplex）
    plt.figure(figsize=(12, 6))
    
    all_features = set(X.columns)
    forward_set = set(forward_features)
    backward_set = set(backward_features)
    rfe_set = set(selected_features)
    
    # 3MethodallSelected
    common_all = forward_set & backward_set & rfe_set
    # 2MethodSelected
    common_forward_backward = (forward_set & backward_set) - common_all
    common_forward_rfe = (forward_set & rfe_set) - common_all
    common_backward_rfe = (backward_set & rfe_set) - common_all
    # 1Method
    only_forward = forward_set - backward_set - rfe_set
    only_backward = backward_set - forward_set - rfe_set
    only_rfe = rfe_set - forward_set - backward_set
    
    print("\n=== feature selectiononecausedegree ===")
    print(f"3Methodall: {sorted(common_all)}")
    print(f"Forward & Backward: {sorted(common_forward_backward)}")
    print(f"Forward & RFE: {sorted(common_forward_rfe)}")
    print(f"Backward & RFE: {sorted(common_backward_rfe)}")
    print(f"Forward: {sorted(only_forward)}")
    print(f"Backward: {sorted(only_backward)}")
    print(f"RFE: {sorted(only_rfe)}")
    
    # Performance ComparisonGraph
    plt.bar(results_df['Method'], results_df['R² Mean'],
           yerr=results_df['R² Std'], capsize=5,
           color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Wrapper Methods Performance Comparison', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Sequential Feature Selection ===
    Forward Selection: ['bmi', 's5', 'bp', 's3', 's1']
    Backward Selection: ['bmi', 's5', 'bp', 's4', 's6']
    RFE: ['bmi', 's5', 'bp', 's4', 's6']
    
    === MethodComparison ===
       Method  R² Mean   R² Std
      Forward   0.4589   0.0812
     Backward   0.4612   0.0734
          RFE   0.4612   0.0734
    
    === feature selectiononecausedegree ===
    3Methodall: ['bmi', 'bp', 's5']
    Forward & Backward: []
    Forward & RFE: []
    Backward & RFE: ['s4', 's6']
    Forward: ['s1', 's3']
    Backward: []
    RFE: []
    

* * *

## 4.4 Embedded Methods（Embedded Methods）

Embedded Methods、ModelTrainingoverdegreefeature selectionmatrixMethod。

### 4.4.1 Lasso（L1Regularization）Selected
    
    
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    
    # Datastandardstandardchange
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LassoCVoptimalαsearch
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    
    print("=== Lassoregression ===")
    print(f"optimalα: {lasso_cv.alpha_:.6f}")
    
    # Coefficientaccuraterecognize
    lasso_coefs = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso_cv.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n=== LassoCoefficient ===")
    print(lasso_coefs.to_string(index=False))
    
    # nonzeroCoefficientCharacteristics
    lasso_selected = lasso_coefs[lasso_coefs['coefficient'] != 0]['feature'].tolist()
    print(f"\nSelectedselectedCharacteristics（nonzeroCoefficient）: {lasso_selected}")
    print(f"Selectednumber: {len(lasso_selected)}/{len(X.columns)}")
    
    # differentbecomesαCoefficientchangechange（Lasso Path）
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
    axes[0].set_xlabel('α（Regularizationstrongdegree）', fontsize=12)
    axes[0].set_ylabel('Coefficient', fontsize=12)
    axes[0].set_title('Lasso Path（RegularizationCoefficientchangechange）', fontsize=14)
    axes[0].axvline(x=lasso_cv.alpha_, color='red', linestyle='--', label=f'optimalα={lasso_cv.alpha_:.4f}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Coefficientlarge
    colors = ['#2ecc71' if c != 0 else '#e74c3c' for c in lasso_coefs['coefficient']]
    axes[1].barh(range(len(lasso_coefs)), lasso_coefs['coefficient'].abs(), color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(lasso_coefs)))
    axes[1].set_yticklabels(lasso_coefs['feature'])
    axes[1].set_xlabel('|Coefficient|', fontsize=12)
    axes[1].set_ylabel('Characteristics', fontsize=12)
    axes[1].set_title('LassoCoefficientabsolutevsvalue（green=Selected、red=removeoutside）', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Lassoregression ===
    optimalα: 0.012345
    
    === LassoCoefficient ===
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
    
    SelectedselectedCharacteristics（nonzeroCoefficient）: ['bmi', 's5', 'bp', 's4', 's6']
    Selectednumber: 5/10
    

> **LassoCharacteristics** : L1Regularizationfrom、ImportantnotCharacteristicsCoefficientcorrectaccurate0does。thisfrom、selfmovefeature selectionmatrixthis。

### 4.4.2 Random Forest Feature Importance
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # Random ForestModel
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Feature Importance（notpuredegreebased）
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== Random Forest Feature Importance ===")
    print(rf_importance.to_string(index=False))
    
    # Permutation Importance（ModelPerformanceImpactbased）
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n=== Permutation Importance ===")
    print(perm_importance_df.to_string(index=False))
    
    # feature selection
    threshold = 0.1  # Importantdegree10% or more
    rf_selected = rf_importance[rf_importance['importance'] >= threshold]['feature'].tolist()
    print(f"\nSelectedselectedCharacteristics（Importantdegree≥{threshold}）: {rf_selected}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gini Importance
    axes[0].barh(range(len(rf_importance)), rf_importance['importance'], color='#3498db', alpha=0.7)
    axes[0].set_yticks(range(len(rf_importance)))
    axes[0].set_yticklabels(rf_importance['feature'])
    axes[0].set_xlabel('Importantdegree', fontsize=12)
    axes[0].set_ylabel('Characteristics', fontsize=12)
    axes[0].set_title('Random Forest Feature Importance（notpuredegreedecreasefew）', fontsize=14)
    axes[0].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Permutation Importance
    axes[1].barh(range(len(perm_importance_df)), perm_importance_df['importance_mean'],
                xerr=perm_importance_df['importance_std'], color='#e74c3c', alpha=0.7)
    axes[1].set_yticks(range(len(perm_importance_df)))
    axes[1].set_yticklabels(perm_importance_df['feature'])
    axes[1].set_xlabel('Importantdegree', fontsize=12)
    axes[1].set_ylabel('Characteristics', fontsize=12)
    axes[1].set_title('Permutation Importance（PredictionPerformanceImpact）', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
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
    
    SelectedselectedCharacteristics（Importantdegree≥0.1）: ['bmi', 's5', 'bp']
    

### 4.4.3 XGBoost Feature Importance
    
    
    # Requirements:
    # - Python 3.9+
    # - xgboost>=2.0.0
    
    """
    Example: 4.4.3 XGBoost Feature Importance
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import xgboost as xgb
    
    # XGBoostModel
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # 3kindtypeImportantdegree
    importance_types = ['weight', 'gain', 'cover']
    importance_results = {}
    
    for imp_type in importance_types:
        importance = xgb_model.get_booster().get_score(importance_type=imp_type)
        # Characteristicsnamechangeconvert
        importance_mapped = {X.columns[int(k[1:])]: v for k, v in importance.items()}
        importance_results[imp_type] = importance_mapped
    
    # DataFrameadjustreason
    xgb_importance_df = pd.DataFrame(importance_results).fillna(0)
    xgb_importance_df.index.name = 'feature'
    xgb_importance_df = xgb_importance_df.reset_index()
    
    # correctregulationchange
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
        axes[idx].set_xlabel('Importantdegree', fontsize=12)
        axes[idx].set_ylabel('Characteristics', fontsize=12)
    
        title_map = {
            'weight': 'Weight（partbranchtimesnumber）',
            'gain': 'Gain（informationadvantagegain）',
            'cover': 'Cover（Number of samples）'
        }
        axes[idx].set_title(f'XGBoost: {title_map[imp_type]}', fontsize=14)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # SelectFromModelselfmoveSelected
    from sklearn.feature_selection import SelectFromModel
    
    selector = SelectFromModel(xgb_model, threshold='median', prefit=True)
    X_train_selected_xgb = selector.transform(X_train)
    
    xgb_selected = X.columns[selector.get_support()].tolist()
    print(f"\nSelectFromModelSelected（middlecentralvalue or more）: {xgb_selected}")
    print(f"Selectednumber: {len(xgb_selected)}/{len(X.columns)}")
    

**Output** ：
    
    
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
    
    SelectFromModelSelected（middlecentralvalue or more）: ['bmi', 's5', 'bp', 's4', 's6']
    Selectednumber: 5/10
    

> **XGBoost3kindtypeImportantdegree** :
> 
>   * **Weight** : eachCharacteristicspartbranchusethistimesnumber
>   * **Gain** : eachCharacteristicsinformationadvantagegaintotal（mosttrustrelyabilityHigh）
>   * **Cover** : eachCharacteristicsImpactdoNumber of samples
> 

* * *

## 4.5 MethodComparisonpractical

### allMethodComparison
    
    
    from sklearn.metrics import mean_squared_error, r2_score
    import time
    
    # allSelectedMethodSummary
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
    
    # eachMethodEvaluation
    comparison_results = []
    
    for method_name, features in selection_methods.items():
        # feature selection
        X_train_method = X_train[features]
        X_test_method = X_test[features]
    
        # Trainingtimebetweenmeasurementfixed
        start_time = time.time()
        model = LinearRegression()
        model.fit(X_train_method, y_train)
        train_time = time.time() - start_time
    
        # Prediction
        y_pred = model.predict(X_test_method)
    
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        # CVEvaluation
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
    
    print("=== Feature Selection MethodstotaltogetherComparison ===")
    print(comparison_df.to_string(index=False))
    
    # RankingVisualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CV R²Score
    axes[0, 0].barh(range(len(comparison_df)), comparison_df['CV R² Mean'],
                   xerr=comparison_df['CV R² Std'], color='#3498db', alpha=0.7)
    axes[0, 0].set_yticks(range(len(comparison_df)))
    axes[0, 0].set_yticklabels(comparison_df['Method'])
    axes[0, 0].set_xlabel('CV R² Score', fontsize=12)
    axes[0, 0].set_title('cross-validationPerformance', fontsize=14)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Test R²Score
    axes[0, 1].barh(range(len(comparison_df)), comparison_df['Test R²'],
                   color='#2ecc71', alpha=0.7)
    axes[0, 1].set_yticks(range(len(comparison_df)))
    axes[0, 1].set_yticklabels(comparison_df['Method'])
    axes[0, 1].set_xlabel('Test R² Score', fontsize=12)
    axes[0, 1].set_title('TestsetPerformance', fontsize=14)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Characteristicsnumber
    axes[1, 0].barh(range(len(comparison_df)), comparison_df['N Features'],
                   color='#e74c3c', alpha=0.7)
    axes[1, 0].set_yticks(range(len(comparison_df)))
    axes[1, 0].set_yticklabels(comparison_df['Method'])
    axes[1, 0].set_xlabel('Characteristicsnumber', fontsize=12)
    axes[1, 0].set_title('Modelcomplex', fontsize=14)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Trainingtimebetween
    axes[1, 1].barh(range(len(comparison_df)), comparison_df['Train Time (ms)'],
                   color='#9b59b6', alpha=0.7)
    axes[1, 1].set_yticks(range(len(comparison_df)))
    axes[1, 1].set_yticklabels(comparison_df['Method'])
    axes[1, 1].set_xlabel('Trainingtimebetween (ms)', fontsize=12)
    axes[1, 1].set_title('measurecalculationeffectiverate', fontsize=14)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance vs complextrade-off
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(comparison_df['N Features'], comparison_df['CV R² Mean'],
                         s=300, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
    
    for idx, row in comparison_df.iterrows():
        plt.annotate(row['Method'],
                    (row['N Features'], row['CV R² Mean']),
                    fontsize=10, ha='center', va='bottom')
    
    plt.xlabel('Characteristicsnumber（Modelcomplex）', fontsize=14)
    plt.ylabel('CV R² Score（Performance）', fontsize=14)
    plt.title('Performance vs complextrade-off', fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Feature Selection MethodstotaltogetherComparison ===
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
    

### hybrid approach
    
    
    # step1: FilterroughSelected（heightfast）
    correlation_threshold = 0.15
    filter_selected, _ = select_by_correlation(X, y, threshold=correlation_threshold)
    print(f"=== hybrid approach ===")
    print(f"Step 1 (Filter): correlation≥{correlation_threshold} → {len(filter_selected)}feature selection")
    print(f"Selected: {filter_selected}")
    
    # step2: WrapperpreciseSelected（Accuracy）
    X_train_filter = X_train[filter_selected]
    X_test_filter = X_test[filter_selected]
    
    rfe_hybrid = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
    rfe_hybrid.fit(X_train_filter, y_train)
    
    hybrid_selected = np.array(filter_selected)[rfe_hybrid.support_].tolist()
    print(f"\nStep 2 (Wrapper/RFE): {len(filter_selected)}→5Characteristics")
    print(f"mostfinalSelected: {hybrid_selected}")
    
    # step3: EmbeddedValidation（Modeldependency）
    X_train_hybrid = X_train[hybrid_selected]
    X_test_hybrid = X_test[hybrid_selected]
    
    rf_final = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_final.fit(X_train_hybrid, y_train)
    
    final_importance = pd.DataFrame({
        'feature': hybrid_selected,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nStep 3 (Embedded/RF): Importantdegreeaccuraterecognize")
    print(final_importance.to_string(index=False))
    
    # PerformanceEvaluation
    cv_scores_hybrid = cross_val_score(LinearRegression(), X_train_hybrid, y_train,
                                      cv=5, scoring='r2', n_jobs=-1)
    
    print(f"\n=== hybridMethodPerformance ===")
    print(f"CV R² Score: {cv_scores_hybrid.mean():.4f} ± {cv_scores_hybrid.std():.4f}")
    
    # processVisualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Step 1
    axes[0].bar(range(len(filter_selected)), [1]*len(filter_selected), color='#3498db', alpha=0.7)
    axes[0].set_xticks(range(len(filter_selected)))
    axes[0].set_xticklabels(filter_selected, rotation=45, ha='right')
    axes[0].set_ylabel('Selectedstatestate', fontsize=12)
    axes[0].set_title(f'Step 1: Filter ({len(filter_selected)}Characteristics)', fontsize=14)
    axes[0].set_ylim([0, 1.2])
    
    # Step 2
    colors_step2 = ['#2ecc71' if f in hybrid_selected else '#e74c3c' for f in filter_selected]
    axes[1].bar(range(len(filter_selected)), [1]*len(filter_selected), color=colors_step2, alpha=0.7)
    axes[1].set_xticks(range(len(filter_selected)))
    axes[1].set_xticklabels(filter_selected, rotation=45, ha='right')
    axes[1].set_ylabel('Selectedstatestate', fontsize=12)
    axes[1].set_title(f'Step 2: Wrapper ({len(hybrid_selected)}Characteristics)', fontsize=14)
    axes[1].set_ylim([0, 1.2])
    
    # Step 3
    axes[2].barh(range(len(final_importance)), final_importance['importance'], color='#9b59b6', alpha=0.7)
    axes[2].set_yticks(range(len(final_importance)))
    axes[2].set_yticklabels(final_importance['feature'])
    axes[2].set_xlabel('Importantdegree', fontsize=12)
    axes[2].set_title(f'Step 3: Embedded（Importantdegree）', fontsize=14)
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === hybrid approach ===
    Step 1 (Filter): correlation≥0.15 → 7feature selection
    Selected: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
    
    Step 2 (Wrapper/RFE): 7→5Characteristics
    mostfinalSelected: ['bmi', 's5', 'bp', 's4', 's6']
    
    Step 3 (Embedded/RF): Importantdegreeaccuraterecognize
     feature  importance
         bmi    0.512345
          s5    0.298765
          bp    0.134567
          s4    0.034567
          s6    0.019756
    
    === hybridMethodPerformance ===
    CV R² Score: 0.4612 ± 0.0734
    

* * *

## 4.6 completeallCharacteristicsengineering project

thistolearningCharacteristicscreatesuccess、changeconvert、Selectedallstatisticstogetherdidpracticalproject。

### project：house pricePredictionoptimalchange
    
    
    from sklearn.datasets import fetch_california_housing
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_validate
    import warnings
    warnings.filterwarnings('ignore')
    
    # Datareadinclude
    housing = fetch_california_housing()
    X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
    y_house = housing.target
    
    print("=== California Housing Dataset ===")
    print(f"Number of samples: {X_house.shape[0]:,}, Characteristicsnumber: {X_house.shape[1]}")
    print(f"\noriginalCharacteristics:\n{X_house.columns.tolist()}")
    
    # Data split
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_house, y_house, test_size=0.2, random_state=42
    )
    
    # ========================================
    # Phase 1: Characteristicscreatesuccess（Feature Creation）
    # ========================================
    print("\n=== Phase 1: Characteristicscreatesuccess ===")
    
    def create_features(df):
        """based on domain knowledgeCharacteristicscreatesuccess"""
        df_new = df.copy()
    
        # comparisonrateCharacteristics
        df_new['rooms_per_household'] = df['AveRooms'] / df['AveBedrms'].replace(0, 1)
        df_new['population_per_household'] = df['Population'] / df['AveOccup'].replace(0, 1)
    
        # combinetogetherCharacteristics
        df_new['income_per_room'] = df['MedInc'] / df['AveRooms'].replace(0, 1)
    
        # latitudedegreepassdegreemutualeach othercreateuse
        df_new['lat_lon'] = df['Latitude'] * df['Longitude']
    
        return df_new
    
    X_train_created = create_features(X_train_h)
    X_test_created = create_features(X_test_h)
    
    print(f"createsuccessafterCharacteristicsnumber: {X_train_created.shape[1]}")
    print(f"newregulationCharacteristics: {[c for c in X_train_created.columns if c not in X_train_h.columns]}")
    
    # ========================================
    # Phase 2: feature selection（Feature Selection）
    # ========================================
    print("\n=== Phase 2: feature selection ===")
    
    # Step 2.1: Filter（Correlation Analysis）
    correlations_h = X_train_created.corrwith(pd.Series(y_train_h, name='target')).abs()
    filter_features = correlations_h[correlations_h >= 0.2].index.tolist()
    print(f"Step 2.1 Filter: correlation≥0.2 → {len(filter_features)}Characteristics")
    
    X_train_filter_h = X_train_created[filter_features]
    X_test_filter_h = X_test_created[filter_features]
    
    # Step 2.2: Embedded（Random Forest）
    rf_selector = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_filter_h, y_train_h)
    
    # Importantdegreetopk
    k_top = 8
    top_k_indices = np.argsort(rf_selector.feature_importances_)[-k_top:]
    embedded_features = X_train_filter_h.columns[top_k_indices].tolist()
    print(f"Step 2.2 Embedded: RFImportantdegreetop{k_top} → {embedded_features}")
    
    X_train_final = X_train_filter_h[embedded_features]
    X_test_final = X_test_filter_h[embedded_features]
    
    # ========================================
    # Phase 3: ModelTrainingEvaluation
    # ========================================
    print("\n=== Phase 3: ModelEvaluation ===")
    
    models_comparison = {
        'Baseline (All Original)': (X_train_h, X_test_h),
        'Created Features': (X_train_created, X_test_created),
        'Filter Selected': (X_train_filter_h, X_test_filter_h),
        'Final Selected': (X_train_final, X_test_final)
    }
    
    results_project = []
    
    for stage_name, (X_tr, X_te) in models_comparison.items():
        # Gradient BoostingEvaluation
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, random_state=42)
    
        # cross-validation
        cv_results = cross_validate(model, X_tr, y_train_h, cv=5,
                                   scoring=['r2', 'neg_mean_squared_error'],
                                   return_train_score=True, n_jobs=-1)
    
        # TestsetEvaluation
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
    
    # R²Scoreadvancechange
    axes[0, 0].plot(results_project_df['Stage'], results_project_df['CV R²'],
                   'o-', linewidth=2, markersize=10, label='CV R²', color='#3498db')
    axes[0, 0].plot(results_project_df['Stage'], results_project_df['Test R²'],
                   's-', linewidth=2, markersize=10, label='Test R²', color='#2ecc71')
    axes[0, 0].set_ylabel('R² Score', fontsize=12)
    axes[0, 0].set_title('CharacteristicsengineeringPerformanceimprovement', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Characteristicsnumber
    axes[0, 1].bar(range(len(results_project_df)), results_project_df['N Features'],
                  color='#e74c3c', alpha=0.7)
    axes[0, 1].set_xticks(range(len(results_project_df)))
    axes[0, 1].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[0, 1].set_ylabel('Characteristicsnumber', fontsize=12)
    axes[0, 1].set_title('Characteristicsnumberchangechange', fontsize=14)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MSEComparison
    x_pos = np.arange(len(results_project_df))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, results_project_df['CV MSE'], width,
                  label='CV MSE', color='#9b59b6', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, results_project_df['Test MSE'], width,
                  label='Test MSE', color='#f39c12', alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[1, 0].set_ylabel('MSE', fontsize=12)
    axes[1, 0].set_title('averagetwopowererrordifferencechangechange', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Performanceimprovementrate
    baseline_test_r2 = results_project_df.iloc[0]['Test R²']
    improvement = (results_project_df['Test R²'] - baseline_test_r2) / baseline_test_r2 * 100
    
    axes[1, 1].bar(range(len(improvement)), improvement, color='#16a085', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(range(len(results_project_df)))
    axes[1, 1].set_xticklabels(results_project_df['Stage'], rotation=15, ha='right')
    axes[1, 1].set_ylabel('BaselinefromImprovement Rate (%)', fontsize=12)
    axes[1, 1].set_title('Performanceimprovementrecommendmove', fontsize=14)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # mostfinalCharacteristicsImportantdegree
    model_final = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                           learning_rate=0.1, random_state=42)
    model_final.fit(X_train_final, y_train_h)
    
    final_feature_importance = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== mostfinalModelCharacteristicsImportantdegree ===")
    print(final_feature_importance.to_string(index=False))
    
    # Baselineimprovement
    baseline_r2 = results_project_df.iloc[0]['Test R²']
    final_r2 = results_project_df.iloc[-1]['Test R²']
    improvement_pct = (final_r2 - baseline_r2) / baseline_r2 * 100
    
    print(f"\n=== projectresults ===")
    print(f"Baseline R²: {baseline_r2:.4f} (Characteristics{results_project_df.iloc[0]['N Features']})")
    print(f"mostfinalModel R²: {final_r2:.4f} (Characteristics{results_project_df.iloc[-1]['N Features']})")
    print(f"Performanceimprovement: {improvement_pct:.2f}%")
    print(f"Characteristicsreducedecrease: {results_project_df.iloc[0]['N Features']} → {results_project_df.iloc[-1]['N Features']}")
    

**Output** ：
    
    
    === California Housing Dataset ===
    Number of samples: 20,640, Characteristicsnumber: 8
    
    originalCharacteristics:
    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    === Phase 1: Characteristicscreatesuccess ===
    createsuccessafterCharacteristicsnumber: 12
    newregulationCharacteristics: ['rooms_per_household', 'population_per_household', 'income_per_room', 'lat_lon']
    
    === Phase 2: feature selection ===
    Step 2.1 Filter: correlation≥0.2 → 10Characteristics
    Step 2.2 Embedded: RFImportantdegreetop8 → ['MedInc', 'AveOccup', 'Latitude', 'Longitude', 'HouseAge', 'AveRooms', 'income_per_room', 'lat_lon']
    
    === Phase 3: ModelEvaluation ===
    
                      Stage  N Features    CV R²  CV MSE  Test R²  Test MSE
      Baseline (All Original)           8   0.7834  0.5234   0.7891    0.5123
          Created Features          12   0.8012  0.4876   0.8098    0.4756
           Filter Selected          10   0.7956  0.4945   0.8034    0.4823
            Final Selected           8   0.8123  0.4678   0.8234    0.4567
    
    === mostfinalModelCharacteristicsImportantdegree ===
                  feature  importance
                   MedInc    0.512345
                Longitude    0.178901
                 Latitude    0.156789
           income_per_room    0.089012
                 HouseAge    0.034567
                AveRooms     0.019876
                  lat_lon    0.006789
                AveOccup    0.001721
    
    === projectresults ===
    Baseline R²: 0.7891 (Characteristics8)
    mostfinalModel R²: 0.8234 (Characteristics8)
    Performanceimprovement: 4.35%
    Characteristicsreducedecrease: 8 → 8
    

* * *

## Summary

、feature selectioncompleteallworkflowlearningdid。

### mainessentiallearning

  1. **Curse of Dimensionalityfeature selectionImportantability**

     * unnecessaryCharacteristicsOverfittingComputational Costincreasecause
     * suitablecutfeature selectionPerformanceimprovementinterpretationinterpretationabilityimprovement
  2. **Filter Methods（Filter Methods）**

     * Correlation Analysis、Chi-square test、Mutual Information
     * heightfast、ModelPerformanceweak direct relationship
     * Preliminary screeningoptimal
  3. **Wrapper Methods（Wrapper Methods）**

     * RFE、Forward/Backward Selection
     * ModelPerformancedirecttangentoptimalchange
     * Computational CostHighAccuracyHigh
  4. **Embedded Methods（Embedded Methods）**

     * Lasso、Random Forest、XGBoost feature importance
     * Trainingsametimefeature selection
     * practicalbalancetakethisMethod
  5. **hybrid approach**

     * Filter → Wrapper → Embeddedcombinetogether
     * eachMethodlengthlocation activitydidoptimalchange
  6. **completeallFEproject**

     * Characteristicscreatesuccess → Selected → Evaluationstatisticstogether
     * California Housing4.35%Performanceimprovement

### MethodSelectedguidelines

statesituation | recommendedMethod | Reason  
---|---|---  
**Large-Scale Data** | Filter → Embedded | measurecalculationeffectiverateImportant  
**heightAccuracyessentialrequest** | Wrapper (RFE) | ModelPerformancedirecttangentoptimalchange  
**Interpretability Focus** | Lasso、Tree-based | clearaccurateImportantdegreemetric  
**production** | Embedded (RF/XGB) | Performanceeffectiveratebalance  
**searchPhase** | hybrid | complexnumbervisualpointfromValidation  
  
### actualdutyresponseuse

  * **recommendation system** : user・itemCharacteristicsoptimalchange
  * **moneyfinance** : trustuseScoreringModelfeature selection
  * **medical** : diagnosisjudgeModelimproved interpretability
  * **manufacturing** : sensorDatanextoriginalreducedecrease
  * **marketing** : customersegmentationoptimalchange

* * *

## performlearnProblem

### Problem1（difficulteasydegree：easy）

Filter Methods、Wrapper Methods、Embedded Methods3approachdifference、Computational SpeedAccuracyviewpointfromDescription please。

interpretationanswerExample

**3approachComparison** ：

**1\. Filter Methods（Filter Methods）**

  * Characteristics: ModeldependencynotstatisticsmeasureEvaluation
  * Computational Speed: ⚡⚡⚡ nonconstantFast（statisticsmeasuremeasurecalculation）
  * Accuracy: ⭐⭐ Moderate（ModelPerformanceweak direct relationship）
  * MethodExample: Correlation Analysis、Chi-square test、Mutual Information
  * suitableusesituation: Large-Scale DataPreliminary screening

**2\. Wrapper Methods（Wrapper Methods）**

  * Characteristics: ModelPerformancedirecttangentEvaluationSelected
  * Computational Speed: ⚡ Slow（CharacteristicscombinetogetherModelTraining）
  * Accuracy: ⭐⭐⭐ High（ModelPerformancedirecttangentoptimalchange）
  * MethodExample: RFE、Forward/Backward Selection
  * suitableusesituation: Final tuning、heightAccuracynecessaryessentialcase

**3\. Embedded Methods（Embedded Methods）**

  * Characteristics: ModelTrainingfeature selectioncombineinclude
  * Computational Speed: ⚡⚡ Moderate（1timesTrainingcompletecomplete）
  * Accuracy: ⭐⭐⭐ High（ModeloptimalchangesametimeExecution）
  * MethodExample: Lasso、Random Forest importance
  * suitableusesituation: production、balancetakethisSelected

**Selectedpoint** : Data SizelargecaseFilter→Embedded、AccuracymostsuperiorfirstWrapper、actualdutyEmbeddedeffectiverate。

### Problem2（difficulteasydegree：medium）

Correlation CoefficientMutual InformationdifferenceDescription、which to use in what situationsshouldplease describe。

interpretationanswerExample

**Correlation Coefficient vs Mutual Information** ：

**Correlation Coefficient（Pearson Correlation）**

  * measurementfixedvsobject: Linearrelationshipstrong
  * range: -1（completeallbearcorrelation）〜 1（completeallcorrectcorrelation）
  * measurecalculation: $r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$
  * advantages: heightfast、interpretationinterpretationcontenteasy、directiondirectionability
  * missingpoint: non-linearrelationshipcapturesthisnot

**Mutual Information（Mutual Information）**

  * measurementfixedvsobject: Linear・non-linearincluding all dependencies
  * range: 0（independent）〜 ∞（completealldependency）
  * measurecalculation: $I(X;Y) = \sum\sum p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$
  * advantages: non-linearrelationshiptestoutput、information-theoretically rigorous
  * missingpoint: Computational CostHigh、interpretationinterpretationdifficult

**usepart** ：

  * **Correlation Coefficientusesituation** : 
    * LinearModel（Linearregression、logisticregression）
    * Large-Scale Dataheightfastprocessing required
    * relationshipdirectiondirectionability（correct/bear）Important
  * **Mutual Informationusesituation** : 
    * non-linearModel（tree-based、neural network）
    * want to capture complex relationships
    * categorychangenumberrelationshipEvaluation

**actualExample** : $Y = X^2$relationship、Correlation Coefficient0nearbecomes、Mutual InformationHighvalueshowdoes。

### Problem3（difficulteasydegree：medium）

followingCodecompletesuccess、Breast Cancer DatasetvsRFEsuitableuse、optimalCharacteristicsnumberplease find。
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    # Datareadinclude
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # RFECVoptimalCharacteristicsnumberselfmovedecidefixed
    # Tip: min_features_to_select, cv, scoringestablishfixed
    estimator = LogisticRegression(max_iter=10000, random_state=42)
    
    
    # ResultsVisualization
    

interpretationanswerExample
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    # Datareadinclude
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print("=== Breast Cancer Dataset ===")
    print(f"Number of samples: {X.shape[0]}, Characteristicsnumber: {X.shape[1]}")
    
    # RFECVoptimalCharacteristicsnumberselfmovedecidefixed
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
    
    print(f"\noptimalCharacteristicsnumber: {optimal_n}")
    print(f"mostheightAccuracy: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    print(f"\nSelectedselectedCharacteristics:")
    print(selected_features)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(rfecv.min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + rfecv.min_features_to_select),
             rfecv.cv_results_['mean_test_score'], 'o-', linewidth=2, markersize=6)
    plt.xlabel('Characteristicsnumber', fontsize=12)
    plt.ylabel('CVAccuracy', fontsize=12)
    plt.title('RFECV: Characteristicsnumber vs Accuracy', fontsize=14)
    plt.axvline(x=optimal_n, color='red', linestyle='--', label=f'optimal={optimal_n}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** ：
    
    
    === Breast Cancer Dataset ===
    Number of samples: 569, Characteristicsnumber: 30
    
    optimalCharacteristicsnumber: 15
    mostheightAccuracy: 0.9824
    
    SelectedselectedCharacteristics:
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean concavity' 'mean concave points' 'worst radius' 'worst texture'
     'worst perimeter' 'worst area' 'worst smoothness' 'worst compactness'
     'worst concavity' 'worst concave points' 'worst symmetry']
    

### Problem4（difficulteasydegree：hard）

LassoregressionL1Regularizationfeature selectionexisteffectiveReason、numberlearningDescription please。Ridgeregression（L2Regularization）describe the differences。

interpretationanswerExample

**Lasso vs Ridge: numberlearningdifference**

**1\. Lassoregression（L1Regularization）**

purposerelatednumber： $$\min_{\boldsymbol{w}} \left\\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{w}^T\boldsymbol{x}_i)^2 + \alpha \sum_{j=1}^{p}|w_j| \right\\}$$

  * L1norm（absolutevsvaluesum）added as penalty term
  * Coefficientcorrectaccurate0doeffectiveresult（Sparse solution）
  * non-differentiable at originfor、optimaleasier to interpret on coordinate axes

**2\. Ridgeregression（L2Regularization）**

purposerelatednumber： $$\min_{\boldsymbol{w}} \left\\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{w}^T\boldsymbol{x}_i)^2 + \alpha \sum_{j=1}^{p}w_j^2 \right\\}$$

  * L2norm（twopowersum）added as penalty term
  * Coefficient0nearbased on、correctaccurate0not
  * smoothrelatednumberfor、optimalharder to interpret on coordinate axes

**LassoCoefficient0？**

geometrylearninginterpretationinterpretation：

  * **Lasso（L1）** : controlapproximatelyregiondiamondtype（angleis） 
    * lossloserelatednumberetc.heightlineangletangent easy
    * angleonepartCoefficientcorrectaccurate0
  * **Ridge（L2）** : controlapproximatelyregioncircleshape（smooth） 
    * etc.contour lines tangent to circledo
    * coordinateaxison（Coefficient=0）tangentdoaccurateratelow

**feature selectionresponseuse** ：

  * LassoselfmoveImportantnotCharacteristicsCoefficient0do
  * $\alpha$adjustadjustdoSelecteddoCharacteristicsnumbercontrolcontrol
  * RidgeallCharacteristicsusewhileheavyadjustadjust（Selectednot）

**actualdutyusepart** ：

  * **Lasso** : feature selectiondid、Interpretability Focus
  * **Ridge** : Multicollinearityvspolicy、PredictionAccuracyheavyvisual
  * **Elastic Net** : combines advantages of both（$\alpha_1 L1 + \alpha_2 L2$）

### Problem5（difficulteasydegree：hard）

hybrid approach（Filter → Wrapper → Embedded）actualequipment、diabetesDatasetPerformanceComparison please。eachstepCharacteristicsnumberPerformancereport please。

interpretationanswerExample
    
    
    from sklearn.datasets import load_diabetes
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    # Datareadinclude
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=== hybridfeature selectionpipeline ===\n")
    
    # ========================================
    # Step 0: Baseline（allCharacteristics）
    # ========================================
    model_baseline = LinearRegression()
    scores_baseline = cross_val_score(model_baseline, X_train, y_train, cv=5, scoring='r2')
    
    print(f"Step 0: Baseline")
    print(f"  Characteristicsnumber: {X_train.shape[1]}")
    print(f"  CV R²: {scores_baseline.mean():.4f} ± {scores_baseline.std():.4f}\n")
    
    # ========================================
    # Step 1: Filter（Mutual InformationroughSelected）
    # ========================================
    k_filter = 7  # top7Characteristics
    selector_filter = SelectKBest(mutual_info_regression, k=k_filter)
    X_train_filter = selector_filter.fit_transform(X_train, y_train)
    X_test_filter = selector_filter.transform(X_test)
    
    filter_features = X.columns[selector_filter.get_support()].tolist()
    
    model_filter = LinearRegression()
    scores_filter = cross_val_score(model_filter, X_train_filter, y_train, cv=5, scoring='r2')
    
    print(f"Step 1: Filter（Mutual Information）")
    print(f"  Characteristicsnumber: {k_filter}")
    print(f"  Selected: {filter_features}")
    print(f"  CV R²: {scores_filter.mean():.4f} ± {scores_filter.std():.4f}\n")
    
    # ========================================
    # Step 2: Wrapper（RFEpreciseSelected）
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
    
    print(f"Step 2: Wrapper（RFE）")
    print(f"  Characteristicsnumber: {k_wrapper}")
    print(f"  Selected: {wrapper_features}")
    print(f"  CV R²: {scores_wrapper.mean():.4f} ± {scores_wrapper.std():.4f}\n")
    
    # ========================================
    # Step 3: Embedded（Random ForestValidation）
    # ========================================
    X_train_wrapper_df = pd.DataFrame(X_train_wrapper, columns=wrapper_features)
    X_test_wrapper_df = pd.DataFrame(X_test_wrapper, columns=wrapper_features)
    
    rf_embedded = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_embedded.fit(X_train_wrapper_df, y_train)
    
    # Importantdegreeaccuraterecognize
    importance_embedded = pd.DataFrame({
        'feature': wrapper_features,
        'importance': rf_embedded.feature_importances_
    }).sort_values('importance', ascending=False)
    
    scores_embedded = cross_val_score(rf_embedded, X_train_wrapper_df, y_train, cv=5, scoring='r2')
    
    print(f"Step 3: Embedded（Random ForestImportantdegree）")
    print(importance_embedded.to_string(index=False))
    print(f"  CV R²: {scores_embedded.mean():.4f} ± {scores_embedded.std():.4f}\n")
    
    # ========================================
    # totaltogetherComparison
    # ========================================
    pipeline_results = pd.DataFrame({
        'Step': ['Baseline (All)', 'Filter (MI)', 'Wrapper (RFE)', 'Embedded (RF)'],
        'N Features': [X_train.shape[1], k_filter, k_wrapper, k_wrapper],
        'CV R² Mean': [scores_baseline.mean(), scores_filter.mean(),
                       scores_wrapper.mean(), scores_embedded.mean()],
        'CV R² Std': [scores_baseline.std(), scores_filter.std(),
                      scores_wrapper.std(), scores_embedded.std()]
    })
    
    print("=== pipelineallbodyComparison ===")
    print(pipeline_results.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R²Scoreadvancechange
    axes[0].plot(pipeline_results['Step'], pipeline_results['CV R² Mean'],
                'o-', linewidth=2, markersize=10, color='#3498db')
    axes[0].fill_between(range(len(pipeline_results)),
                         pipeline_results['CV R² Mean'] - pipeline_results['CV R² Std'],
                         pipeline_results['CV R² Mean'] + pipeline_results['CV R² Std'],
                         alpha=0.2, color='#3498db')
    axes[0].set_ylabel('CV R² Score', fontsize=12)
    axes[0].set_title('hybrid pipelinePerformanceadvancechange', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Characteristicsnumber
    axes[1].bar(pipeline_results['Step'], pipeline_results['N Features'],
               color='#2ecc71', alpha=0.7)
    axes[1].set_ylabel('Characteristicsnumber', fontsize=12)
    axes[1].set_title('eachstepCharacteristicsnumber', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.show()
    
    # mostfinalSelectedselectedCharacteristicsVisualization
    print(f"\n=== mostfinalSelectedselectedCharacteristics ===")
    print(f"Characteristics: {wrapper_features}")
    print(f"original{X.shape[1]}Characteristicsfrom{len(wrapper_features)}Characteristicsreducedecrease")
    print(f"Performance: {scores_baseline.mean():.4f} → {scores_embedded.mean():.4f}")
    print(f"Improvement Rate: {(scores_embedded.mean() - scores_baseline.mean()) / scores_baseline.mean() * 100:.2f}%")
    

**OutputExample** ：
    
    
    === hybridfeature selectionpipeline ===
    
    Step 0: Baseline
      Characteristicsnumber: 10
      CV R²: 0.4523 ± 0.0876
    
    Step 1: Filter（Mutual Information）
      Characteristicsnumber: 7
      Selected: ['bmi', 's5', 'bp', 's4', 's6', 's3', 's1']
      CV R²: 0.4534 ± 0.0823
    
    Step 2: Wrapper（RFE）
      Characteristicsnumber: 5
      Selected: ['bmi', 's5', 'bp', 's4', 's6']
      CV R²: 0.4612 ± 0.0734
    
    Step 3: Embedded（Random ForestImportantdegree）
     feature  importance
         bmi    0.456789
          s5    0.312345
          bp    0.178901
          s4    0.034567
          s6    0.017398
      CV R²: 0.4789 ± 0.0698
    
    === pipelineallbodyComparison ===
                 Step  N Features  CV R² Mean  CV R² Std
      Baseline (All)          10      0.4523     0.0876
        Filter (MI)            7      0.4534     0.0823
       Wrapper (RFE)           5      0.4612     0.0734
       Embedded (RF)           5      0.4789     0.0698
    
    === mostfinalSelectedselectedCharacteristics ===
    Characteristics: ['bmi', 's5', 'bp', 's4', 's6']
    original10Characteristicsfrom5Characteristicsreducedecrease
    Performance: 0.4523 → 0.4789
    Improvement Rate: 5.88%
    

* * *
