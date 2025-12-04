---
title: "Chapter 1: Anomaly Detection Fundamentals"
chapter_title: "Chapter 1: Anomaly Detection Fundamentals"
subtitle: Basic Concepts and Task Design of Anomaly Detection
reading_time: 25-30 minutes
difficulty: Beginner
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers the fundamentals of Anomaly Detection Fundamentals, which what is anomaly detection?. You will learn Distinguish between Point, task classification, and evaluation metrics.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the definition and types of anomaly detection
  * ✅ Distinguish between Point, Contextual, and Collective anomalies
  * ✅ Understand task classification and selection criteria for anomaly detection
  * ✅ Select appropriate evaluation metrics
  * ✅ Use representative datasets and visualization techniques
  * ✅ Understand challenges and countermeasures in anomaly detection

* * *

## 1.1 What is Anomaly Detection?

### Definition of Anomalies

**Anomaly Detection** is a task that identifies data points that significantly deviate from normal patterns.

> "Anomaly" refers to observations with rare and unexpected patterns that differ from the majority of normal data.

### Three Types of Anomalies

Type | Description | Example  
---|---|---  
**Point Anomaly** | Individual data points are anomalous | Sudden high-value credit card transaction  
**Contextual Anomaly** | Anomalous only in specific context | 35°C temperature is normal in summer, anomalous in winter  
**Collective Anomaly** | Collection of data forms anomalous pattern | Continuous abnormal waveform in ECG  
  
### Applications of Anomaly Detection

#### 1\. Fraud Detection

  * Credit card fraud detection
  * Insurance claim fraud detection
  * Money laundering detection

#### 2\. Manufacturing

  * Defective product detection
  * Equipment failure prediction
  * Quality control anomaly detection

#### 3\. Healthcare

  * Early disease detection
  * Tumor detection in medical images
  * Vital sign anomaly detection

#### 4\. IT Systems (Cybersecurity & Operations)

  * Network intrusion detection
  * Server failure prediction
  * Abnormal traffic detection

### Business Value of Anomaly Detection
    
    
    ```mermaid
    graph LR
        A[Anomaly Detection] --> B[Cost Reduction]
        A --> C[Risk Mitigation]
        A --> D[Revenue Growth]
    
        B --> B1[Preventive Maintenance Before Failures]
        B --> B2[Early Defect Detection]
    
        C --> C1[Security Breach Prevention]
        C --> C2[Fraudulent Transaction Prevention]
    
        D --> D1[Downtime Reduction]
        D --> D2[Customer Satisfaction Improvement]
    
        style A fill:#7b2cbf,color:#fff
        style B fill:#e8f5e9
        style C fill:#fff3e0
        style D fill:#e3f2fd
    ```

### Example: Basic Anomaly Detection
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example: Basic Anomaly Detection
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    # Generate normal data
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1,
                             cluster_std=1.0, center_box=(0, 0))
    
    # Add anomalous data (3 types)
    # 1. Point Anomaly: distant points
    point_anomalies = np.array([[8, 8], [-8, -8], [8, -8]])
    
    # 2. Contextual Anomaly: within normal range but contextually anomalous
    # (e.g., out-of-season values in time series data)
    contextual_anomalies = np.array([[2, 2], [-2, -2]])
    
    # 3. Collective Anomaly: anomalous as a group
    collective_anomalies = np.random.normal(loc=[5, 5], scale=0.3, size=(10, 2))
    
    # Combine all data
    X_all = np.vstack([X_normal, point_anomalies,
                       contextual_anomalies, collective_anomalies])
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.scatter(X_normal[:, 0], X_normal[:, 1],
                c='blue', alpha=0.5, s=50, label='Normal Data', edgecolors='black')
    plt.scatter(point_anomalies[:, 0], point_anomalies[:, 1],
                c='red', s=200, marker='X', label='Point Anomaly',
                edgecolors='black', linewidths=2)
    plt.scatter(contextual_anomalies[:, 0], contextual_anomalies[:, 1],
                c='orange', s=200, marker='s', label='Contextual Anomaly',
                edgecolors='black', linewidths=2)
    plt.scatter(collective_anomalies[:, 0], collective_anomalies[:, 1],
                c='purple', s=100, marker='^', label='Collective Anomaly',
                edgecolors='black', linewidths=2)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Three Types of Anomalies', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Data Statistics ===")
    print(f"Normal Data: {len(X_normal)} samples")
    print(f"Point Anomaly: {len(point_anomalies)} samples")
    print(f"Contextual Anomaly: {len(contextual_anomalies)} samples")
    print(f"Collective Anomaly: {len(collective_anomalies)} samples")
    print(f"Anomaly Rate: {(len(point_anomalies) + len(contextual_anomalies) + len(collective_anomalies)) / len(X_all) * 100:.1f}%")
    

> **Important** : In anomaly detection, normal data forms the vast majority, while anomalous data is rare (typically 1-5%).

* * *

## 1.2 Task Classification of Anomaly Detection

### 1\. Classification by Learning Method

Type | Label Information | Use Case | Algorithm Examples  
---|---|---|---  
**Supervised Learning** | Both normal and anomaly labels | Abundant labeled data | Random Forest, SVM  
**Semi-supervised Learning** | Normal labels only | Only normal data labeled | One-Class SVM, Autoencoder  
**Unsupervised Learning** | No labels | Label acquisition difficult | Isolation Forest, LOF, DBSCAN  
  
### 2\. Novelty Detection vs Outlier Detection

Type | Training Data | Purpose | Example  
---|---|---|---  
**Novelty Detection** | Normal data only | Detection of new patterns | New malware detection  
**Outlier Detection** | Mixed normal and anomalies | Anomaly detection in existing data | Noise removal in sensor data  
  
### 3\. Online vs Offline Detection

Type | Processing Timing | Characteristics | Application Examples  
---|---|---|---  
**Online Detection**  
(Real-time) | Upon data arrival | Low latency, incremental updates | Network intrusion detection  
**Offline Detection**  
(Batch processing) | Batch processing | High accuracy, global optimization | Monthly report anomaly analysis  
  
### Task Selection Decision Flow
    
    
    ```mermaid
    graph TD
        A[Anomaly Detection Task Design] --> B{Label Data Available?}
        B -->|Both available| C[Supervised Learning]
        B -->|Normal only| D[Semi-supervised Learning / Novelty Detection]
        B -->|None| E[Unsupervised Learning / Outlier Detection]
    
        C --> F{Real-time?}
        D --> F
        E --> F
    
        F -->|Yes| G[Online Detection]
        F -->|No| H[Offline Detection]
    
        G --> I[Method Selection]
        H --> I
    
        style A fill:#7b2cbf,color:#fff
        style C fill:#e8f5e9
        style D fill:#fff3e0
        style E fill:#ffebee
        style G fill:#e3f2fd
        style H fill:#f3e5f5
    ```

### Example: Comparison of Three Approaches
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example: Comparison of Three Approaches
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import classification_report, accuracy_score
    
    # Generate data (imbalanced data)
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                               n_redundant=2, n_classes=2, weights=[0.95, 0.05],
                               flip_y=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("=== Data Distribution ===")
    print(f"Training Data: {len(y_train)} samples (Normal: {(y_train==0).sum()}, Anomaly: {(y_train==1).sum()})")
    print(f"Test Data: {len(y_test)} samples (Normal: {(y_test==0).sum()}, Anomaly: {(y_test==1).sum()})")
    
    # 1. Supervised learning (with labels)
    print("\n=== 1. Supervised Learning ===")
    clf_supervised = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_supervised.fit(X_train, y_train)
    y_pred_supervised = clf_supervised.predict(X_test)
    acc_supervised = accuracy_score(y_test, y_pred_supervised)
    print(f"Accuracy: {acc_supervised:.3f}")
    print(classification_report(y_test, y_pred_supervised, target_names=['Normal', 'Anomaly']))
    
    # 2. Semi-supervised learning (train with normal data only)
    print("\n=== 2. Semi-supervised Learning (Novelty Detection) ===")
    X_train_normal = X_train[y_train == 0]  # Normal data only
    clf_novelty = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    clf_novelty.fit(X_train_normal)
    y_pred_novelty = clf_novelty.predict(X_test)
    # One-Class SVM output: 1=normal, -1=anomaly → convert to 0=normal, 1=anomaly
    y_pred_novelty = (y_pred_novelty == -1).astype(int)
    acc_novelty = accuracy_score(y_test, y_pred_novelty)
    print(f"Accuracy: {acc_novelty:.3f}")
    print(classification_report(y_test, y_pred_novelty, target_names=['Normal', 'Anomaly']))
    
    # 3. Unsupervised learning (no labels)
    print("\n=== 3. Unsupervised Learning (Outlier Detection) ===")
    clf_unsupervised = IsolationForest(contamination=0.05, random_state=42)
    clf_unsupervised.fit(X_train)
    y_pred_unsupervised = clf_unsupervised.predict(X_test)
    # Isolation Forest output: 1=normal, -1=anomaly → convert to 0=normal, 1=anomaly
    y_pred_unsupervised = (y_pred_unsupervised == -1).astype(int)
    acc_unsupervised = accuracy_score(y_test, y_pred_unsupervised)
    print(f"Accuracy: {acc_unsupervised:.3f}")
    print(classification_report(y_test, y_pred_unsupervised, target_names=['Normal', 'Anomaly']))
    
    # Comparison summary
    print("\n=== Accuracy Comparison ===")
    print(f"Supervised Learning:   {acc_supervised:.3f}")
    print(f"Semi-supervised Learning: {acc_novelty:.3f}")
    print(f"Unsupervised Learning:   {acc_unsupervised:.3f}")
    

> **Important** : Supervised learning provides the highest accuracy but requires labeled data. In real business scenarios, select methods considering the cost of label acquisition.

* * *

## 1.3 Evaluation Metrics

### Class Imbalance Problem

In anomaly detection, normal data overwhelmingly outnumber anomalies, making accuracy alone insufficient.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: In anomaly detection, normal data overwhelmingly outnumber a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Example: 95% normal, 5% anomaly data
    y_true = np.array([0]*95 + [1]*5)
    
    # Bad predictor: predicts everything as normal
    y_pred_bad = np.array([0]*100)
    
    # Good predictor: correctly detects anomalies
    y_pred_good = np.concatenate([np.array([0]*95), np.array([1]*5)])
    
    print("=== Bad Predictor (Predicts Everything as Normal) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_bad):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred_bad, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred_bad, zero_division=0):.3f}")
    print(f"F1: {f1_score(y_true, y_pred_bad, zero_division=0):.3f}")
    
    print("\n=== Good Predictor (Correctly Detects Anomalies) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_good):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred_good):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred_good):.3f}")
    print(f"F1: {f1_score(y_true, y_pred_good):.3f}")
    

**Output** :
    
    
    === Bad Predictor (Predicts Everything as Normal) ===
    Accuracy: 0.950
    Precision: 0.000
    Recall: 0.000
    F1: 0.000
    
    === Good Predictor (Correctly Detects Anomalies) ===
    Accuracy: 1.000
    Precision: 1.000
    Recall: 1.000
    F1: 1.000
    

> **Lesson** : Even with 95% accuracy, the predictor may not detect a single anomaly.

### Confusion Matrix and Key Metrics

Metric | Formula | Meaning  
---|---|---  
**Precision** | $\frac{TP}{TP + FP}$ | Proportion of true anomalies among predicted anomalies  
**Recall** | $\frac{TP}{TP + FN}$ | Proportion of actual anomalies detected  
**F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of Precision and Recall  
**ROC-AUC** | Area under ROC curve | Threshold-independent overall performance  
**PR-AUC** | Area under PR curve | More appropriate than ROC-AUC for imbalanced data  
  
### ROC-AUC vs PR-AUC
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: ROC-AUC vs PR-AUC
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    # Generate imbalanced data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.95, 0.05],
                               random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get prediction probabilities
    y_scores = clf.predict_proba(X_test)[:, 1]
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC curve
    axes[0].plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                 label='Random Prediction')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    axes[1].plot(recall, precision, color='green', lw=2,
                 label=f'PR Curve (AUC = {pr_auc:.3f})')
    baseline = (y_test == 1).sum() / len(y_test)
    axes[1].axhline(y=baseline, color='gray', lw=1, linestyle='--',
                    label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Evaluation Metrics ===")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"Anomaly Data Ratio: {baseline:.3f}")
    

> **Important** : For imbalanced data, PR-AUC is a more appropriate metric than ROC-AUC. ROC-AUC tends to be overly optimistic due to the large proportion of normal data.

### Domain-Specific Evaluation Metrics

Domain | Emphasized Metric | Reason  
---|---|---  
**Medical Diagnosis** | Recall (High) | Minimize false negatives (reduce FN)  
**Spam Filter** | Precision (High) | Minimize false positives (reduce FP)  
**Fraud Detection** | F1, PR-AUC | Balance emphasis  
**Predictive Maintenance** | Recall (High) | Prevent failure oversight  
  
* * *

## 1.4 Datasets and Visualization

### Synthetic Datasets
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Synthetic Datasets
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs, make_moons
    from scipy.stats import multivariate_normal
    
    np.random.seed(42)
    
    # Dataset 1: Gaussian distribution
    X_gaussian, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0,
                               center_box=(0, 0), random_state=42)
    outliers_gaussian = np.random.uniform(low=-8, high=8, size=(15, 2))
    X1 = np.vstack([X_gaussian, outliers_gaussian])
    y1 = np.array([0]*300 + [1]*15)
    
    # Dataset 2: Crescent shape
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    outliers_moons = np.random.uniform(low=-2, high=3, size=(15, 2))
    X2 = np.vstack([X_moons, outliers_moons])
    y2 = np.array([0]*300 + [1]*15)
    
    # Dataset 3: Donut shape
    theta = np.linspace(0, 2*np.pi, 300)
    r = 3 + np.random.normal(0, 0.3, 300)
    X_donut = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    outliers_donut = np.random.normal(0, 1, size=(15, 2))
    X3 = np.vstack([X_donut, outliers_donut])
    y3 = np.array([0]*300 + [1]*15)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    datasets = [
        (X1, y1, 'Gaussian Distribution'),
        (X2, y2, 'Crescent Shape'),
        (X3, y3, 'Donut Shape')
    ]
    
    for ax, (X, y, title) in zip(axes, datasets):
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6,
                   s=50, label='Normal', edgecolors='black')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.9,
                   s=150, marker='X', label='Anomaly', edgecolors='black', linewidths=2)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Synthetic Dataset Characteristics ===")
    print("1. Gaussian Distribution: Linearly separable, suitable for statistical methods")
    print("2. Crescent Shape: Nonlinear pattern, complex boundaries")
    print("3. Donut Shape: Density-based methods are effective")
    

### Real-world Datasets

Dataset | Domain | Samples | Anomaly Rate  
---|---|---|---  
**Credit Card Fraud** | Finance | 284,807 | 0.17%  
**KDD Cup 99** | Network | 4,898,431 | 19.7%  
**MNIST (Anomaly Detection Version)** | Image | 70,000 | Variable  
**Thyroid Disease** | Medical | 3,772 | 2.5%  
**NASA Bearing** | Manufacturing | Time Series | Variable  
  
### Visualization Techniques

#### 1\. Visualization by Dimensionality Reduction
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 1. Visualization by Dimensionality Reduction
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Generate high-dimensional data (20 dimensions)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.95, 0.05],
                               random_state=42)
    
    # Dimensionality reduction by PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Dimensionality reduction by t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA
    axes[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', alpha=0.6,
                    s=50, label='Normal', edgecolors='black')
    axes[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', alpha=0.9,
                    s=150, marker='X', label='Anomaly', edgecolors='black', linewidths=2)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    axes[0].set_title('PCA Visualization', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    axes[1].scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='blue', alpha=0.6,
                    s=50, label='Normal', edgecolors='black')
    axes[1].scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='red', alpha=0.9,
                    s=150, marker='X', label='Anomaly', edgecolors='black', linewidths=2)
    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Dimensionality Reduction Comparison ===")
    print(f"PCA Cumulative Variance Ratio (2 components): {pca.explained_variance_ratio_.sum():.2%}")
    print("t-SNE: Excellent at preserving nonlinear structure (emphasizes local structure)")
    

#### 2\. Anomaly Score Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 2. Anomaly Score Visualization
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    
    # Generate data
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0,
                             center_box=(0, 0), random_state=42)
    X_outliers = np.random.uniform(low=-8, high=8, size=(15, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # Calculate anomaly scores with Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)
    anomaly_scores = -clf.score_samples(X)  # Convert negative values to positive
    
    # Calculate scores on grid (for heatmap)
    xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    Z = -clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.7)
    axes[0].scatter(X[:, 0], X[:, 1], c=anomaly_scores, cmap='RdYlBu_r',
                    s=50, edgecolors='black', linewidths=1)
    plt.colorbar(contour, ax=axes[0], label='Anomaly Score')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].set_title('Anomaly Score Heatmap', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(anomaly_scores, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=np.percentile(anomaly_scores, 95), color='red',
                    linestyle='--', linewidth=2, label='95th Percentile (Threshold)')
    axes[1].set_xlabel('Anomaly Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Anomaly Score Statistics ===")
    print(f"Minimum: {anomaly_scores.min():.3f}")
    print(f"Maximum: {anomaly_scores.max():.3f}")
    print(f"Mean: {anomaly_scores.mean():.3f}")
    print(f"95th Percentile (Threshold Candidate): {np.percentile(anomaly_scores, 95):.3f}")
    

* * *

## 1.5 Challenges in Anomaly Detection

### 1\. Label Scarcity

**Problem** : Labeling anomalous data is costly and difficult

**Countermeasures** :

  * Unsupervised learning (Isolation Forest, LOF)
  * Semi-supervised learning (One-Class SVM, Autoencoder)
  * Active Learning (label only important samples)
  * Weak Supervision (utilize noisy labels)

### 2\. High Dimensionality

**Problem** : Curse of dimensionality (distances lose meaning)

**Countermeasures** :

  * Dimensionality reduction (PCA, Autoencoder)
  * Feature selection (use important features only)
  * Subspace methods

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Countermeasures:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    
    # Experiment on curse of dimensionality
    dimensions = [2, 5, 10, 20, 50, 100, 200]
    avg_distances = []
    
    np.random.seed(42)
    for d in dimensions:
        # Generate random points
        X = np.random.uniform(0, 1, size=(100, d))
        # Calculate pairwise distances
        distances = pdist(X, metric='euclidean')
        avg_distances.append(distances.mean())
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, avg_distances, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Average Euclidean Distance', fontsize=12)
    plt.title('Curse of Dimensionality: Relationship Between Dimensions and Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Curse of Dimensionality ===")
    for d, dist in zip(dimensions, avg_distances):
        print(f"Dimensions {d:3d}: Average Distance = {dist:.3f}")
    print("\n→ In high dimensions, all points appear equidistant (distances lose meaning)")
    

### 3\. Concept Drift

**Problem** : Normal patterns change over time

**Countermeasures** :

  * Online Learning (incremental updates)
  * Sliding Window (retrain on recent data)
  * Ensemble Methods (models from multiple time periods)
  * Adaptive Thresholds (dynamic threshold adjustment)

### 4\. Interpretability

**Problem** : Difficult to explain why something was classified as anomalous

**Countermeasures** :

  * Rule-based methods
  * Feature importance
  * SHAP values (Shapley value-based explanations)
  * Attention mechanisms

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Countermeasures:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.tree import DecisionTreeClassifier
    
    # Sample data
    np.random.seed(42)
    X_normal = np.random.normal(0, 1, size=(100, 5))
    X_anomaly = np.random.normal(5, 1, size=(5, 5))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0]*100 + [1]*5)
    
    # Anomaly detection with Isolation Forest
    clf_if = IsolationForest(contamination=0.05, random_state=42)
    clf_if.fit(X)
    predictions = clf_if.predict(X)
    
    # Analyze feature importance of anomalous samples
    # Simply calculate deviation for each feature
    X_mean = X_normal.mean(axis=0)
    X_std = X_normal.std(axis=0)
    
    anomaly_idx = np.where(predictions == -1)[0][:3]  # First 3 anomalous samples
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, idx in enumerate(anomaly_idx):
        deviations = np.abs((X[idx] - X_mean) / X_std)
        axes[i].bar(range(5), deviations, color='steelblue', edgecolor='black')
        axes[i].axhline(y=2, color='red', linestyle='--', linewidth=2, label='2σ')
        axes[i].set_xlabel('Feature', fontsize=11)
        axes[i].set_ylabel('Standard Deviation', fontsize=11)
        axes[i].set_title(f'Anomaly Sample {idx} Deviations', fontsize=12, fontweight='bold')
        axes[i].set_xticks(range(5))
        axes[i].set_xticklabels([f'F{j}' for j in range(5)])
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Interpretability Example ===")
    for i, idx in enumerate(anomaly_idx):
        deviations = np.abs((X[idx] - X_mean) / X_std)
        max_dev_feature = deviations.argmax()
        print(f"Anomaly Sample {idx}: Feature {max_dev_feature} is most anomalous ({deviations[max_dev_feature]:.2f}σ)")
    

### Challenge Prioritization

Challenge | Impact | Difficulty | Priority  
---|---|---|---  
**Label Scarcity** | High | Medium | High  
**Concept Drift** | High | High | High  
**High Dimensionality** | Medium | Medium | Medium  
**Interpretability** | Medium | High | Medium  
  
* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Basics of Anomaly Detection**

     * Three types: Point, Contextual, and Collective anomalies
     * Applications to fraud detection, manufacturing, healthcare, and IT systems
     * Understanding business value
  2. **Task Classification**

     * Supervised, semi-supervised, and unsupervised learning
     * Novelty Detection vs Outlier Detection
     * Online vs Offline Detection
  3. **Evaluation Metrics**

     * Using Precision, Recall, and F1 appropriately
     * ROC-AUC vs PR-AUC
     * Handling class imbalance problems
  4. **Datasets and Visualization**

     * Validation with synthetic data
     * Characteristics of real-world data
     * Visualization with PCA, t-SNE, and anomaly scores
  5. **Challenges and Countermeasures**

     * Label Scarcity: Unsupervised and semi-supervised learning
     * High Dimensionality: Dimensionality reduction
     * Concept Drift: Online Learning
     * Interpretability: SHAP, Feature Importance

### Principles of Anomaly Detection

Principle | Description  
---|---  
**Leverage Domain Knowledge** | Reflect business knowledge in method selection and threshold setting  
**Appropriate Evaluation Metrics** | Prioritize PR-AUC and F1 for imbalanced data  
**Continuous Monitoring** | Retraining to handle Concept Drift  
**Emphasize Interpretability** | Explainable models necessary for production deployment  
**Cost Awareness** | Consider business costs of FP and FN  
  
### To the Next Chapter

In Chapter 2, we will learn about **Statistical Anomaly Detection** :

  * Z-score, Grubbs Test
  * Gaussian Mixture Models
  * Statistical Process Control
  * Bayesian Anomaly Detection
  * Applications to time series data

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the differences between Point Anomaly, Contextual Anomaly, and Collective Anomaly, and provide specific examples for each.

Sample Answer

**Answer** :

  1. **Point Anomaly**

     * Description: Individual data points that differ significantly from all other data
     * Example: Sudden credit card transaction of $100,000
  2. **Contextual Anomaly**

     * Description: Considered anomalous only in specific contexts (time, location, etc.)
     * Example: 35°C temperature is normal in summer but anomalous in winter. Access to an office building at 3 AM is anomalous
  3. **Collective Anomaly**

     * Description: Individual data points are normal but form an anomalous pattern as a collection
     * Example: Continuous abnormal waveforms in ECG, distributed DoS attack on web server

### Problem 2 (Difficulty: medium)

For the following scenario, select the appropriate anomaly detection task setting (supervised/semi-supervised/unsupervised) and explain your reasoning.

**Scenario** : You want to detect defective products from product images on a manufacturing line. There are abundant images of normal products but only a few images of defective products.

Sample Answer

**Answer** :

**Recommended Task** : **Semi-supervised Learning (Novelty Detection)**

**Reasoning** :

  1. **Label Situation**

     * Abundant images of normal products (labeled)
     * Only a few images of defective products (insufficient for supervised learning)
  2. **Task Nature**

     * Learn patterns of normal products and consider deviations as anomalies
     * Typical use case for Novelty Detection
  3. **Specific Methods**

     * One-Class SVM: Learn boundaries with normal data only
     * Autoencoder: Anomaly detection based on reconstruction error of normal images
     * Deep SVDD: Hypersphere representation of normal data using deep learning

**Why Supervised Learning is Inappropriate** :

  * Too few defective product samples (difficult to generalize with only a few)
  * Unknown types of defects (cannot detect defect patterns not included in training data)

**Why Unsupervised Learning is Inappropriate** :

  * Inefficient not to utilize available normal product labels
  * Semi-supervised learning provides higher accuracy

### Problem 3 (Difficulty: medium)

Explain why Accuracy alone is insufficient for evaluation in anomaly detection, and propose alternative metrics to use.

Sample Answer

**Answer** :

**Why Accuracy is Insufficient** :

In anomaly detection, there is a class imbalance problem where normal data overwhelmingly outnumber anomalies (95-99%).

**Specific Example** :

  * Data: 95% normal, 5% anomaly
  * Predictor A: Predicts everything as normal → Accuracy = 95% (not detecting a single anomaly)
  * Predictor B: Correctly detects all anomalies → Accuracy = 100%

Predictor A is useless yet receives a high evaluation of 95% accuracy.

**Recommended Evaluation Metrics** :

Metric | Recommendation Reason | Use Case  
---|---|---  
**PR-AUC** | Appropriate for imbalanced data, threshold-independent | Overall evaluation  
**F1 Score** | Balance of Precision/Recall | Single threshold evaluation  
**Recall** | Minimize missed anomalies | Medical, predictive maintenance  
**Precision** | Minimize false detections | Spam filter  
  
**Formulas** :

  * Precision = TP / (TP + FP): Proportion of true anomalies among predicted anomalies
  * Recall = TP / (TP + FN): Proportion of actual anomalies detected
  * F1 = 2 × (Precision × Recall) / (Precision + Recall)

### Problem 4 (Difficulty: hard)

Explain the impact of the curse of dimensionality on anomaly detection and provide three countermeasures. Show the relationship between number of dimensions and distance with Python code.

Sample Answer

**Answer** :

**Impact of Curse of Dimensionality** :

In high-dimensional spaces, distances between all data points become similar, causing distance-based anomaly detection methods (KNN, LOF, etc.) to become ineffective.

**Specific Problems** :

  1. Distance between nearest neighbor and farthest point converges
  2. Difference in distances between anomalous and normal data becomes smaller
  3. Euclidean distance loses meaning

**Countermeasures** :

  1. **Dimensionality Reduction**

     * PCA: Keep only important axes through principal component analysis
     * Autoencoder: Nonlinear dimensionality reduction
     * t-SNE/UMAP: Visualization and structure preservation
  2. **Feature Selection**

     * Mutual Information: Select features that contribute to anomaly detection
     * L1 regularization: Set weights of unnecessary features to zero
     * Domain knowledge: Feature selection by experts
  3. **Subspace Methods**

     * Subspace methods: Anomaly detection in multiple low-dimensional subspaces
     * Random Projection: Use multiple random low-dimensional projections

**Python Code** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Python Code:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    
    # Experiment on curse of dimensionality
    np.random.seed(42)
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]
    results = []
    
    for d in dimensions:
        # Generate random points from uniform distribution
        X = np.random.uniform(0, 1, size=(100, d))
    
        # Calculate pairwise distances
        distances = pdist(X, metric='euclidean')
    
        # Record statistics
        results.append({
            'dim': d,
            'mean': distances.mean(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max()
        })
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean distance and standard deviation
    dims = [r['dim'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    
    axes[0].plot(dims, means, marker='o', linewidth=2, markersize=8, label='Mean Distance')
    axes[0].fill_between(dims,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          alpha=0.3, label='±1σ')
    axes[0].set_xlabel('Number of Dimensions', fontsize=12)
    axes[0].set_ylabel('Euclidean Distance', fontsize=12)
    axes[0].set_title('Relationship Between Dimensions and Distance', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Ratio of minimum to maximum distance
    ratios = [r['min'] / r['max'] for r in results]
    axes[1].plot(dims, ratios, marker='s', linewidth=2, markersize=8, color='red')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', label='Perfect Match')
    axes[1].set_xlabel('Number of Dimensions', fontsize=12)
    axes[1].set_ylabel('Min Distance / Max Distance', fontsize=12)
    axes[1].set_title('Disappearance of Relative Distance Differences', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Curse of Dimensionality: Distance Statistics ===")
    for r in results:
        print(f"Dimensions {r['dim']:3d}: Mean={r['mean']:.3f}, "
              f"Std Dev={r['std']:.3f}, Min/Max Ratio={r['min']/r['max']:.3f}")
    
    print("\n→ As dimensions increase:")
    print("  1. Mean distance increases (scale effect)")
    print("  2. Relative distance differences shrink (min/max ratio approaches 1)")
    print("  3. All points appear equidistant (anomaly detection becomes difficult)")
    

**Conclusion** :

  * In high dimensions, distance differences become smaller, reducing anomaly detection accuracy
  * Preserve meaningful distances through dimensionality reduction or feature selection
  * Feature engineering utilizing domain knowledge is important

### Problem 5 (Difficulty: hard)

Explain the impact of Concept Drift on anomaly detection and demonstrate countermeasures using Online Learning. Include a simple implementation example with time series data.

Sample Answer

**Answer** :

**Impact of Concept Drift** :

Concept Drift refers to the phenomenon where the distribution of normal data changes over time. This causes models trained on past data to become unsuitable for current data.

**Specific Examples** :

  * E-commerce: Seasonal variations (purchasing patterns change between summer and winter)
  * Manufacturing: Normal vibration patterns change due to equipment aging
  * Network: Evolution of traffic patterns

**Problems** :

  1. Past models become outdated, increasing false positives (FP)
  2. New normal patterns are misclassified as anomalies
  3. Detection performance degrades over time

**Countermeasures with Online Learning** :

  1. **Sliding Window Approach**

     * Retrain model with only recent N samples
     * Discard old data and adapt to new patterns
  2. **Incremental Learning**

     * Update model incrementally with new data
     * Efficient without retraining on all data
  3. **Adaptive Thresholds**

     * Dynamically adjust anomaly detection thresholds
     * Update thresholds based on recent data distribution

**Implementation Example (Sliding Window)** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation Example (Sliding Window):
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    
    # Generate time series data (with concept drift)
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    
    # Normal data: mean changes over time (Concept Drift)
    mean_shift = time / 200  # Mean gradually increases
    X = np.random.normal(loc=mean_shift, scale=1.0, size=(n_samples, 5))
    
    # Add some anomalous data
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    X[anomaly_indices] += np.random.uniform(5, 10, size=(50, 5))
    
    # True labels
    y_true = np.zeros(n_samples)
    y_true[anomaly_indices] = 1
    
    # 1. Static model (train on initial data only)
    print("=== 1. Static Model (No Concept Drift Handling) ===")
    static_model = IsolationForest(contamination=0.05, random_state=42)
    static_model.fit(X[:200])  # Only initial 200 samples
    
    static_predictions = static_model.predict(X)
    static_predictions = (static_predictions == -1).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    static_precision = precision_score(y_true, static_predictions)
    static_recall = recall_score(y_true, static_predictions)
    static_f1 = f1_score(y_true, static_predictions)
    
    print(f"Precision: {static_precision:.3f}")
    print(f"Recall: {static_recall:.3f}")
    print(f"F1 Score: {static_f1:.3f}")
    
    # 2. Online Learning (Sliding Window)
    print("\n=== 2. Online Learning (Sliding Window, window=200) ===")
    window_size = 200
    online_predictions = np.zeros(n_samples)
    
    for i in range(window_size, n_samples):
        # Train model on recent window_size samples
        window_data = X[i-window_size:i]
        online_model = IsolationForest(contamination=0.05, random_state=42)
        online_model.fit(window_data)
    
        # Predict current sample
        pred = online_model.predict(X[i:i+1])
        online_predictions[i] = (pred == -1).astype(int)
    
    online_precision = precision_score(y_true[window_size:], online_predictions[window_size:])
    online_recall = recall_score(y_true[window_size:], online_predictions[window_size:])
    online_f1 = f1_score(y_true[window_size:], online_predictions[window_size:])
    
    print(f"Precision: {online_precision:.3f}")
    print(f"Recall: {online_recall:.3f}")
    print(f"F1 Score: {online_f1:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Change in data mean (Concept Drift)
    axes[0].plot(time, X.mean(axis=1), alpha=0.7, label='Data Mean')
    axes[0].scatter(anomaly_indices, X[anomaly_indices].mean(axis=1),
                    c='red', s=50, marker='X', label='Anomalous Data', zorder=5)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Mean Value', fontsize=12)
    axes[0].set_title('Concept Drift: Normal Data Distribution Changes Over Time',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Prediction comparison
    axes[1].scatter(time, static_predictions, alpha=0.5, label='Static Model', s=10)
    axes[1].scatter(time, online_predictions, alpha=0.5, label='Online Learning', s=10)
    axes[1].scatter(anomaly_indices, y_true[anomaly_indices],
                    c='red', marker='X', s=100, label='True Anomalies', zorder=5, edgecolors='black')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Anomaly Flag', fontsize=12)
    axes[1].set_title('Static Model vs Online Learning', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Performance Comparison ===")
    print(f"Static Model:      F1={static_f1:.3f}")
    print(f"Online Learning: F1={online_f1:.3f}")
    print(f"Improvement: {(online_f1 - static_f1):.3f}")
    

**Conclusion** :

  * In environments with Concept Drift, static models suffer performance degradation
  * Online learning with Sliding Window adapts to new patterns
  * Window size is a tradeoff between stability (large) and adaptation speed (small)

* * *

## References

  1. Chandola, V., Banerjee, A., & Kumar, V. (2009). _Anomaly detection: A survey_. ACM computing surveys (CSUR), 41(3), 1-58.
  2. Aggarwal, C. C. (2017). _Outlier analysis_ (2nd ed.). Springer.
  3. Goldstein, M., & Uchida, S. (2016). _A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data_. PloS one, 11(4), e0152173.
  4. Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021). _Deep learning for anomaly detection: A review_. ACM Computing Surveys (CSUR), 54(2), 1-38.
  5. Rousseeuw, P. J., & Hubert, M. (2011). _Robust statistics for outlier detection_. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 1(1), 73-79.
