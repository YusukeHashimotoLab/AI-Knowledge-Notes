---
title: "Chapter 3: Machine Learning-Based Anomaly Detection"
chapter_title: "Chapter 3: Machine Learning-Based Anomaly Detection"
subtitle: Anomaly Detection with Isolation Forest, LOF, and One-Class SVM
reading_time: 70-80 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
version: 1.0
created_at: "by:"
---

# Chapter 3: Machine Learning-Based Anomaly Detection

This chapter covers Machine Learning. You will learn algorithmic principles of Isolation Forest.

## Learning Objectives

  * Understand the algorithmic principles of Isolation Forest
  * Detect local anomalies using LOF (Local Outlier Factor)
  * Master boundary learning of normal data with One-Class SVM
  * Apply DBSCAN and other methods to anomaly detection
  * Learn implementation methods for ensemble anomaly detection

**Reading Time** : 70-80 minutes

* * *

## 3.1 Isolation Forest

Isolation Forest is an anomaly detection algorithm that exploits the property that anomalous data is easier to "isolate" than normal data. Proposed by Liu et al. in 2008, it can be effectively applied to high-dimensional data.

### 3.1.1 Algorithm Principles

**Basic Idea:**

  * Anomalous data is rare and has different feature values from normal data
  * When repeatedly splitting with randomly selected features, anomalous data becomes isolated earlier
  * The shorter the number of splits until isolation (path length), the higher the anomaly score

**Algorithm Steps:**
    
    
    1. Randomly select a feature
    2. Randomly choose a split point between the minimum and maximum values of that feature
    3. Divide data into two groups
    4. Recursively repeat steps 1-3 for each group
    5. Record the path length until each data point is isolated
    6. Build multiple trees (forest) and calculate anomaly score from average path length
    

### 3.1.2 Path Length and Anomaly Score

**Path Length:**

Let $h(x)$ be the number of splits until data point $x$ is isolated. Normal data is isolated at deeper positions (larger $h(x)$), while anomalous data is isolated at shallower positions (smaller $h(x)$).

**Anomaly Score Calculation:**

$$ s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}} $$ 

Where:

  * $E[h(x)]$: Average path length across multiple trees
  * $c(n)$: Normalization constant for average path length at sample size $n$
  * $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ (where $H(i)$ is the harmonic number)

**Score Interpretation:**

  * $s \approx 1$: Anomalous (clear anomaly)
  * $s \approx 0.5$: Normal (average path length)
  * $s < 0.5$: Normal (deeper than average position)

### 3.1.3 Hyperparameter Tuning

**Main Parameters:**

Parameter | Description | Recommended Value  
---|---|---  
`n_estimators` | Number of trees | 100-200 (default: 100)  
`max_samples` | Number of samples to draw for each tree | 256 (default: auto)  
`contamination` | Proportion of anomalies | 0.1 (depends on data)  
`max_features` | Number of features to consider for each split | 1.0 (all features)  
  
**Parameter Selection Guidelines:**

  * `n_estimators`: More provides stability but increases computational cost (100-200 is sufficient)
  * `max_samples`: 256 is recommended (paper default), reduce for large-scale data to speed up
  * `contamination`: Set to known anomaly rate if available, otherwise use 0.1

### 3.1.4 scikit-learn Implementation

**Basic Implementation:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Basic Implementation:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    
    # Generate sample data (normal data + anomalous data)
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
    X_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))  # Anomalous data
    X = np.vstack([X_normal, X_anomaly])
    
    # Isolation Forest model
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.1,  # Assume 10% is anomalous
        random_state=42
    )
    
    # Train and predict
    y_pred = iso_forest.fit_predict(X)  # -1: Anomaly, 1: Normal
    scores = iso_forest.score_samples(X)  # Anomaly score (lower is more anomalous)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('Isolation Forest: Anomaly Detection Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', edgecolors='k')
    plt.title('Isolation Forest: Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly Score')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of detected anomalies: {np.sum(y_pred == -1)}")
    print(f"Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
    

**Sample Output:**
    
    
    Number of detected anomalies: 32
    Anomaly score range: [-0.234, 0.178]
    

**Application to Real Data (Credit Card Fraud Detection):**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Application to Real Data (Credit Card Fraud Detection):
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Load data (hypothetical example)
    # Actual data can be obtained from Kaggle Credit Card Fraud Detection
    # URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
    
    # Generate sample data (substitute for real data)
    np.random.seed(42)
    n_normal = 1000
    n_fraud = 50
    
    # Normal transactions (small amount, many transactions, geographically concentrated)
    normal_features = np.random.randn(n_normal, 5) * [10, 5, 2, 1, 0.5]
    normal_labels = np.zeros(n_normal)
    
    # Fraudulent transactions (large amount, few transactions, geographically dispersed)
    fraud_features = np.random.randn(n_fraud, 5) * [50, 1, 10, 5, 3] + [100, 0, 50, 20, 10]
    fraud_labels = np.ones(n_fraud)
    
    X = np.vstack([normal_features, fraud_features])
    y = np.hstack([normal_labels, fraud_labels])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Isolation Forest (train on normal data only)
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Assume 5% is fraudulent
        random_state=42
    )
    
    # Train on training data
    iso_forest.fit(X_train)
    
    # Predict on test data
    y_pred = iso_forest.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (fraud)
    
    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    

**Sample Output:**
    
    
    Confusion Matrix:
    [[285  15]
     [  3  12]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
          Normal       0.99      0.95      0.97       300
           Fraud       0.44      0.80      0.57        15
    
        accuracy                           0.94       315
       macro avg       0.72      0.88      0.77       315
    weighted avg       0.96      0.94      0.95       315
    

* * *

## 3.2 LOF (Local Outlier Factor)

LOF is a method for detecting anomalies based on the local density of each data point. Proposed by Breunig et al. in 2000.

### 3.2.1 Density-Based Anomaly Detection

**Basic Principle:**

  * Normal data exists in high-density regions
  * Anomalous data exists in low-density regions
  * Calculate anomaly score by comparing density of each point with density of neighboring points

**Why "Local":**

  * Can detect anomalies that cannot be detected with global density
  * Effective when multiple clusters with different densities exist
  * Calculates relative anomaly score considering each point's neighborhood

### 3.2.2 Local Reachability Density

**k-distance:**

Let $d_k(p)$ be the distance from point $p$ to the k-th nearest point.

**Reachability Distance:**

$$ \text{reach-dist}_k(p, o) = \max\\{d_k(o), d(p, o)\\} $$ 

  * $d(p, o)$: Actual distance between points $p$ and $o$
  * When neighbor $o$ is in a dense region, the reachability distance has a lower bound of $d_k(o)$

**Local Reachability Density (LRD):**

$$ \text{LRD}_k(p) = \frac{1}{\frac{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}{|N_k(p)|}} $$ 

  * $N_k(p)$: Set of k-nearest neighbors of point $p$
  * Inverse of average reachability distance = density

### 3.2.3 LOF Score Calculation

**LOF (Local Outlier Factor):**

$$ \text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{LRD}_k(o)}{\text{LRD}_k(p)}}{|N_k(p)|} $$ 

**Score Interpretation:**

  * $\text{LOF} \approx 1$: Normal (similar density to neighbors)
  * $\text{LOF} \gg 1$: Anomalous (lower density than neighbors)
  * $\text{LOF} < 1$: Normal (higher density than neighbors)

Generally, $\text{LOF} > 1.5$ is considered anomalous.

### 3.2.4 Complete Implementation Example

**Basic Implementation:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Basic Implementation:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.datasets import make_moons
    
    # Generate sample data (moon-shaped data + outliers)
    np.random.seed(42)
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X_outliers = np.random.uniform(low=-1, high=2, size=(20, 2))
    X = np.vstack([X, X_outliers])
    
    # LOF model
    lof = LocalOutlierFactor(
        n_neighbors=20,  # Number of neighbors
        contamination=0.1,  # Anomaly rate
        novelty=False  # Use True for new data prediction
    )
    
    # Prediction
    y_pred = lof.fit_predict(X)  # -1: Anomaly, 1: Normal
    scores = lof.negative_outlier_factor_  # Negative anomaly score (lower is more anomalous)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('LOF: Anomaly Detection Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', edgecolors='k')
    plt.title('LOF: Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Negative Outlier Factor')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of detected anomalies: {np.sum(y_pred == -1)}")
    print(f"Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
    

**Impact of n_neighbors Parameter:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Impact of n_neighbors Parameter:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor
    
    # Data generation
    np.random.seed(42)
    X_normal = np.random.randn(200, 2) * 0.5
    X_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # Compare with different n_neighbors
    n_neighbors_list = [5, 20, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n_neighbors in enumerate(n_neighbors_list):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        y_pred = lof.fit_predict(X)
    
        axes[idx].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
        axes[idx].set_title(f'LOF (n_neighbors={n_neighbors})')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
        anomaly_count = np.sum(y_pred == -1)
        axes[idx].text(0.05, 0.95, f'Anomalies: {anomaly_count}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    

**Anomaly Detection for New Data (novelty=True):**
    
    
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.model_selection import train_test_split
    
    # Data preparation
    np.random.seed(42)
    X_train = np.random.randn(500, 2) * 0.5  # Normal data only
    X_test_normal = np.random.randn(100, 2) * 0.5
    X_test_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X_test = np.vstack([X_test_normal, X_test_outliers])
    
    # LOF (novelty=True: new data prediction mode)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_train)  # Train on normal data only
    
    # Prediction on new data
    y_pred = lof.predict(X_test)
    scores = lof.score_samples(X_test)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.3, label='Training Data', color='blue')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm',
                edgecolors='k', s=100, label='Test Data')
    plt.title('LOF: Anomaly Detection for New Data (novelty=True)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    plt.show()
    
    print(f"Number of detected anomalies: {np.sum(y_pred == -1)}/{len(y_pred)}")
    

* * *

## 3.3 One-Class SVM

One-Class SVM is a method that learns the boundary of normal data and detects data outside that boundary as anomalies.

### 3.3.1 Maximum Margin Hyperplane

**Basic Principle:**

  * Find the hyperplane that best separates normal data from the origin
  * Maximize the margin between the hyperplane and data points
  * Learn non-linear boundaries with the kernel trick

**Mathematical Definition:**

Decision function:

$$ f(x) = \text{sign}(w \cdot \phi(x) - \rho) $$ 

  * $w$: Normal vector
  * $\phi(x)$: Feature vector after kernel transformation
  * $\rho$: Bias term

Optimization problem:

$$ \min_{w, \rho, \xi} \frac{1}{2} \|w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho $$ 

Constraints:

$$ w \cdot \phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0 $$ 

### 3.3.2 Kernel Trick

**Linear Kernel:**

$$ K(x, x') = x \cdot x' $$ 

  * Fast, easy to interpret
  * Applied to linearly separable data

**RBF (Gaussian) Kernel:**

$$ K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right) $$ 

  * Can learn non-linear boundaries
  * Most commonly used kernel
  * $\gamma$: Kernel width (larger values create more complex boundaries)

**Polynomial Kernel:**

$$ K(x, x') = (\gamma x \cdot x' + r)^d $$ 

  * Polynomial boundary of degree $d$
  * More constrained than RBF

### 3.3.3 nu Parameter

**Meaning of nu:**

$\nu \in (0, 1]$ controls the upper and lower bounds of the following two quantities:

  * **Upper bound** on the fraction of anomalies in training data
  * **Lower bound** on the fraction of support vectors

**Recommended Values:**

  * $\nu = 0.1$: Assume 10% is anomalous
  * $\nu = 0.05$: Assume 5% is anomalous
  * $\nu = 0.01$: Assume 1% is anomalous

**Notes:**

  * Setting $\nu$ too small will result in almost no anomalies being detected
  * Setting $\nu$ too large will result in normal data also being classified as anomalous
  * Should be set based on domain knowledge or prior anomaly rate

### 3.3.4 scikit-learn Implementation

**Basic Implementation:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Basic Implementation:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    
    # Data generation
    np.random.seed(42)
    X_train = np.random.randn(200, 2) * 0.5  # Normal data
    X_test_normal = np.random.randn(50, 2) * 0.5
    X_test_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X_test = np.vstack([X_test_normal, X_test_outliers])
    
    # One-Class SVM
    oc_svm = OneClassSVM(
        kernel='rbf',  # RBF kernel
        gamma='auto',  # gamma = 1 / n_features
        nu=0.1  # Assume 10% is anomalous
    )
    
    # Training
    oc_svm.fit(X_train)
    
    # Prediction
    y_pred_train = oc_svm.predict(X_train)
    y_pred_test = oc_svm.predict(X_test)
    scores_test = oc_svm.decision_function(X_test)
    
    # Visualize decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolors='k', label='Training')
    plt.title('One-Class SVM: Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm',
                edgecolors='k', s=100)
    plt.title('One-Class SVM: Test Data Prediction')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of anomalies in training data: {np.sum(y_pred_train == -1)}/{len(y_pred_train)}")
    print(f"Number of anomalies in test data: {np.sum(y_pred_test == -1)}/{len(y_pred_test)}")
    

**Impact of gamma Parameter:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Impact of gamma Parameter:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    
    # Data generation
    np.random.seed(42)
    X_train = np.random.randn(200, 2) * 0.5
    
    # Compare with different gamma values
    gamma_list = [0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, gamma in enumerate(gamma_list):
        oc_svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=0.1)
        oc_svm.fit(X_train)
    
        # Decision boundary
        xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
        Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        axes[idx].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        axes[idx].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        axes[idx].scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolors='k')
        axes[idx].set_title(f'One-Class SVM (gamma={gamma})')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.4 Other Machine Learning Methods

### 3.4.1 DBSCAN (Density-Based Clustering)

**Principle:**

  * Detect high-density regions as clusters
  * Consider points that don't belong to any cluster as noise (anomalies)
  * No need to specify the number of clusters in advance

**Main Parameters:**

  * `eps`: Neighborhood radius (distance threshold)
  * `min_samples`: Minimum number of neighbors to become a core point

**Implementation Example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation Example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    
    # Data generation
    np.random.seed(42)
    X_cluster1 = np.random.randn(100, 2) * 0.3 + [0, 0]
    X_cluster2 = np.random.randn(100, 2) * 0.3 + [3, 3]
    X_outliers = np.random.uniform(low=-2, high=5, size=(20, 2))
    X = np.vstack([X_cluster1, X_cluster2, X_outliers])
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    # Label -1 represents noise (anomalies)
    outliers = labels == -1
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X[~outliers, 0], X[~outliers, 1], c=labels[~outliers],
                cmap='viridis', edgecolors='k', label='Clusters')
    plt.scatter(X[outliers, 0], X[outliers, 1], c='red', marker='x',
                s=100, label='Outliers (Anomalies)')
    plt.title('DBSCAN: Density-Based Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
    print(f"Number of clusters detected: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Number of anomalies: {np.sum(outliers)}")
    

### 3.4.2 Elliptic Envelope

**Principle:**

  * Assume normal distribution and estimate data center and covariance
  * Detect anomalies with Mahalanobis distance
  * Suppress influence of outliers with robust estimation (Minimum Covariance Determinant)

**Implementation Example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation Example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.covariance import EllipticEnvelope
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Data generation
    np.random.seed(42)
    X_normal = np.random.randn(200, 2)
    X_outliers = np.random.uniform(low=-5, high=5, size=(10, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # Elliptic Envelope
    elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
    y_pred = elliptic.fit_predict(X)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('Elliptic Envelope: Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    plt.show()
    
    print(f"Number of detected anomalies: {np.sum(y_pred == -1)}")
    

### 3.4.3 Robust Covariance

**Minimum Covariance Determinant (MCD):**

  * Search for subset that minimizes the determinant of covariance matrix
  * Robust estimation against outliers
  * Used for Mahalanobis distance calculation

    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: Minimum Covariance Determinant (MCD):
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.covariance import MinCovDet
    import numpy as np
    
    # Data generation
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:5] = X[:5] + 5  # Add outliers
    
    # MCD estimation
    mcd = MinCovDet(random_state=42)
    mcd.fit(X)
    
    # Calculate Mahalanobis distance
    distances = mcd.mahalanobis(X)
    
    # Anomaly detection (95th percentile of chi-square distribution)
    from scipy import stats
    threshold = stats.chi2.ppf(0.95, df=2)
    outliers = distances > threshold
    
    print(f"Number of anomalies: {np.sum(outliers)}")
    print(f"Distance threshold: {threshold:.2f}")
    

### 3.4.4 PyOD Library

**PyOD (Python Outlier Detection)** is a specialized library for anomaly detection providing over 40 algorithms.

**Installation:**
    
    
    pip install pyod
    

**Usage Example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Usage Example:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    from pyod.utils.utility import standardizer
    import numpy as np
    
    # Data generation
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=2,
        contamination=0.1, random_state=42
    )
    
    # Data standardization
    X_train = standardizer(X_train)
    X_test = standardizer(X_test)
    
    # Compare multiple models
    models = {
        'KNN': KNN(contamination=0.1),
        'IForest': IForest(contamination=0.1, random_state=42),
        'LOF': LOF(contamination=0.1)
    }
    
    for name, model in models.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)
    
        # Evaluation (with hypothetical ground truth labels)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, scores)
    
        print(f"{name}:")
        print(f"  AUC-ROC: {auc:.3f}")
        print(f"  Number of detected anomalies: {np.sum(y_pred == 1)}")
        print()
    

**Sample Output:**
    
    
    KNN:
      AUC-ROC: 0.892
      Number of detected anomalies: 10
    
    IForest:
      AUC-ROC: 0.915
      Number of detected anomalies: 10
    
    LOF:
      AUC-ROC: 0.903
      Number of detected anomalies: 10
    

* * *

## 3.5 Ensemble Anomaly Detection

By combining multiple anomaly detection algorithms, more accurate and stable anomaly detection becomes possible.

### 3.5.1 Feature Bagging

**Principle:**

  * Randomly select feature subsets
  * Train anomaly detection models on each subset
  * Aggregate predictions from multiple models

**Implementation Example:**
    
    
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    from sklearn.metrics import roc_auc_score
    
    # Data generation
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=10,
        contamination=0.1, random_state=42
    )
    
    # Feature Bagging (base model: LOF)
    fb = FeatureBagging(
        base_estimator=LOF(),
        n_estimators=10,  # Number of models
        contamination=0.1,
        random_state=42
    )
    
    # Training and prediction
    fb.fit(X_train)
    y_pred = fb.predict(X_test)
    scores = fb.decision_function(X_test)
    
    # Evaluation
    auc = roc_auc_score(y_test, scores)
    print(f"Feature Bagging AUC-ROC: {auc:.3f}")
    print(f"Number of detected anomalies: {np.sum(y_pred == 1)}")
    

### 3.5.2 Model Averaging

**Principle:**

  * Train multiple different algorithms
  * Average anomaly scores from each model
  * More robust prediction than single models

**Implementation Example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation Example:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from pyod.models.combination import average, maximization
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    import numpy as np
    
    # Data generation
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=5,
        contamination=0.1, random_state=42
    )
    
    # Train multiple models
    models = [
        KNN(contamination=0.1),
        IForest(contamination=0.1, random_state=42),
        LOF(contamination=0.1)
    ]
    
    # Calculate scores for each model
    scores_list = []
    for model in models:
        model.fit(X_train)
        scores = model.decision_function(X_test)
        scores_list.append(scores)
    
    scores_array = np.array(scores_list)
    
    # Score aggregation (average)
    scores_avg = average(scores_array)
    
    # Score aggregation (maximum)
    scores_max = maximization(scores_array)
    
    # Evaluation
    from sklearn.metrics import roc_auc_score
    auc_avg = roc_auc_score(y_test, scores_avg)
    auc_max = roc_auc_score(y_test, scores_max)
    
    print(f"Average Combination AUC-ROC: {auc_avg:.3f}")
    print(f"Maximum Combination AUC-ROC: {auc_max:.3f}")
    

### 3.5.3 Isolation-Based Ensemble

**LSCP (Locally Selective Combination in Parallel):**

  * Select locally optimal model for each test sample
  * Weight models based on performance in neighborhood
  * Higher accuracy than global averaging

    
    
    from pyod.models.lscp import LSCP
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    
    # Data generation
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=5,
        contamination=0.1, random_state=42
    )
    
    # List of base models
    detector_list = [
        KNN(),
        IForest(random_state=42),
        LOF()
    ]
    
    # LSCP
    lscp = LSCP(detector_list, contamination=0.1, random_state=42)
    lscp.fit(X_train)
    
    # Prediction
    y_pred = lscp.predict(X_test)
    scores = lscp.decision_function(X_test)
    
    # Evaluation
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, scores)
    print(f"LSCP AUC-ROC: {auc:.3f}")
    print(f"Number of detected anomalies: {np.sum(y_pred == 1)}")
    

### 3.5.4 Complete Pipeline Example

**Data Preprocessing → Multiple Model Training → Ensemble → Evaluation:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Data Preprocessing → Multiple Model Training → Ensemble → Ev
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.combination import average
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    
    # Data generation (substitute for real data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    contamination = 0.05
    
    # Normal data
    X_normal = np.random.randn(int(n_samples * (1 - contamination)), n_features)
    # Anomalous data
    X_anomaly = np.random.uniform(low=-5, high=5, size=(int(n_samples * contamination), n_features))
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Data standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'KNN': KNN(contamination=contamination),
        'IForest': IForest(contamination=contamination, random_state=42),
        'LOF': LOF(contamination=contamination),
        'OCSVM': OCSVM(contamination=contamination)
    }
    
    scores_dict = {}
    predictions_dict = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled)
        scores = model.decision_function(X_test_scaled)
        y_pred = model.predict(X_test_scaled)
    
        scores_dict[name] = scores
        predictions_dict[name] = y_pred
    
        auc = roc_auc_score(y_test, scores)
        print(f"{name} AUC-ROC: {auc:.3f}")
    
    # Ensemble (average)
    scores_list = [scores_dict[name] for name in models.keys()]
    scores_ensemble = average(np.array(scores_list))
    auc_ensemble = roc_auc_score(y_test, scores_ensemble)
    print(f"\nEnsemble AUC-ROC: {auc_ensemble:.3f}")
    
    # ROC curve visualization
    plt.figure(figsize=(10, 6))
    
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    
    # Ensemble ROC
    fpr_ens, tpr_ens, _ = roc_curve(y_test, scores_ensemble)
    plt.plot(fpr_ens, tpr_ens, 'k--', linewidth=2,
             label=f'Ensemble (AUC={auc_ensemble:.3f})')
    
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Multiple Models and Ensemble')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

* * *

## 3.6 Summary

### What We Learned in This Chapter

  1. **Isolation Forest:**

     * Anomaly detection by random isolation
     * Calculate anomaly score from path length
     * Hyperparameters (n_estimators, max_samples, contamination)
     * Effective for high-dimensional data
  2. **LOF (Local Outlier Factor):**

     * Anomaly detection based on local density
     * Reachability distance and local reachability density
     * LOF score calculation and interpretation
     * Handles clusters with different densities
  3. **One-Class SVM:**

     * Boundary learning with maximum margin hyperplane
     * Kernel trick (RBF, linear, polynomial)
     * Anomaly rate control with nu parameter
     * Learning non-linear boundaries
  4. **Other Methods:**

     * DBSCAN (density-based clustering)
     * Elliptic Envelope
     * Robust Covariance
     * PyOD library (over 40 algorithms)
  5. **Ensemble Anomaly Detection:**

     * Feature Bagging (feature subsets)
     * Model Averaging (score averaging)
     * Isolation-Based Ensemble (LSCP)
     * Improved accuracy by combining multiple models

### Method Selection Guide

Method | Application Scenarios | Advantages | Disadvantages  
---|---|---|---  
Isolation Forest | High-dimensional data, large-scale data | Fast, scalable | Parameter tuning required  
LOF | Clusters with different densities | Detects local anomalies | High computational cost  
One-Class SVM | Non-linear boundaries, theoretical guarantees | Robust, theoretical foundation | Slow on large-scale data  
DBSCAN | Clustering + anomaly detection | No need to specify number of clusters | Sensitive to parameters  
Ensemble | When high accuracy is needed | Robust, high accuracy | Increased computational cost  
  
### Next Steps

In Chapter 4, we will learn deep learning-based anomaly detection:

  * Autoencoder (reconstruction error-based)
  * VAE (Variational Autoencoder)
  * GAN (Generative Adversarial Network)
  * LSTM Autoencoder (time-series anomaly detection)
  * Transformer (Attention mechanism)

* * *

## Exercises

**Question 1:** In Isolation Forest, should a data point with anomaly score $s(x, n) = 0.8$ be classified as anomalous? Answer with reasoning.

**Answer:**

Yes, it should be classified as anomalous.

**Reasoning:**

  * Anomaly score $s \approx 1$ indicates a clear anomaly
  * $s \approx 0.5$ is normal (average path length)
  * $s = 0.8$ is close to 1, meaning it is isolated earlier than usual
  * Generally, $s > 0.6$ is used as the threshold for anomalies

**Question 2:** If the LOF score is $\text{LOF}_k(p) = 2.5$, is this point anomalous? Also, explain what this score means.

**Answer:**

Yes, it is anomalous.

**Meaning:**

  * $\text{LOF} \approx 1$ indicates similar density to neighbors (normal)
  * $\text{LOF} > 1$ indicates lower density than neighbors (possibly anomalous)
  * $\text{LOF} = 2.5$ indicates that this point's density is approximately 1/2.5 of the average neighbor density
  * Since $\text{LOF} > 1.5$ is generally considered anomalous, 2.5 is a clear anomaly

**Question 3:** If the One-Class SVM nu parameter is set to 0.05, what percentage of training data will be classified as anomalous? Also, explain the impact of increasing nu.

**Answer:**

At most 5% of training data will be classified as anomalous.

**Impact of Increasing nu:**

  * $\nu = 0.1$: At most 10% classified as anomalous
  * $\nu = 0.2$: At most 20% classified as anomalous
  * Increasing nu results in more data being classified as anomalous
  * Risk of misclassifying normal data as anomalous increases (increased false positives)
  * Sensitivity of anomaly detection increases, but accuracy may decrease

**Question 4:** When performing anomaly detection with DBSCAN, how should the eps and min_samples parameters be selected? Describe three specific selection methods.

**Answer:**

  1. **K-Distance Graph Method:**

     * Calculate the k-th nearest neighbor distance for each point (k is the min_samples candidate)
     * Sort distances in descending order and plot
     * Select the point where the distance increases sharply (elbow point) as eps
  2. **Domain Knowledge-Based Selection:**

     * Estimate appropriate neighborhood size from data characteristics
     * Example: For 2D data, min_samples=4; for high dimensions, min_samples=2×dimensions
     * Adjust eps according to data scale
  3. **Grid Search:**

     * Try multiple combinations of (eps, min_samples)
     * Evaluate with silhouette score or number of clusters
     * Select the optimal combination

**Question 5:** In ensemble anomaly detection, explain the differences between Feature Bagging and Model Averaging, and discuss which situations each is effective in (within 300 characters).

**Sample Answer:**

Feature Bagging trains multiple models on feature subsets, addressing correlation and redundancy among features in high-dimensional data. It is effective when there are many correlated features. Model Averaging aggregates predictions from different algorithms (KNN, Isolation Forest, LOF, etc.), leveraging the strengths of each method. It is effective when the data characteristics are unclear and the optimal method is unknown in advance. Feature Bagging enhances diversity of the same algorithm, while Model Averaging utilizes algorithmic diversity. In practice, combining both can build a more robust anomaly detection system.

* * *

## References

  1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." _IEEE International Conference on Data Mining (ICDM)_.
  2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). "LOF: Identifying Density-Based Local Outliers." _ACM SIGMOD International Conference on Management of Data_.
  3. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). "Estimating the Support of a High-Dimensional Distribution." _Neural Computation_ , 13(7), 1443-1471.
  4. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters." _KDD_ , 96(34), 226-231.
  5. Zhao, Y., Nasrullah, Z., & Li, Z. (2019). "PyOD: A Python Toolbox for Scalable Outlier Detection." _Journal of Machine Learning Research_ , 20(96), 1-7.

* * *

**Next Chapter** : [Chapter 4: Deep Learning-Based Anomaly Detection](<chapter4-deep-learning-anomaly.html>)

**License** : This content is provided under the CC BY 4.0 license.
