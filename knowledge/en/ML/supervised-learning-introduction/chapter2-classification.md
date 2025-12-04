---
title: "Chapter 2: Classification Fundamentals"
chapter_title: "Chapter 2: Classification Fundamentals"
subtitle: Theory and Implementation of Category Prediction - From Logistic Regression to Decision Trees and SVM
reading_time: 25-30 min
difficulty: Beginner to Intermediate
code_examples: 12
exercises: 5
version: 1.0
created_at: 2025-10-20
---

## Learning Objectives

By the end of this chapter, you will be able to:

  * Understand the definition and applications of classification problems
  * Understand the theory and implement logistic regression
  * Explain the sigmoid function and probability interpretation
  * Understand the mechanism and implement decision trees
  * Apply k-NN and SVM
  * Evaluate models using confusion matrix, precision, recall, and F1 score
  * Understand and utilize ROC curves and AUC

* * *

## 2.1 What is Classification?

### Definition

**Classification** is a supervised learning task that predicts **discrete values (categories)** from input variables.

> "Learn a function $f: X \rightarrow y$ that predicts discrete class labels $y \in \\{1, 2, ..., K\\}$ from features $X$"

### Types of Classification

Type | Number of Classes | Examples  
---|---|---  
**Binary Classification** | 2 classes | Spam detection, disease diagnosis, customer churn prediction  
**Multi-class Classification** | 3+ classes | Handwritten digit recognition, image classification, sentiment analysis  
**Multi-label Classification** | Multiple labels | Tagging, gene function prediction  
  
### Real-World Applications
    
    
    ```mermaid
    graph LR
        A[Classification Applications] --> B[Healthcare: Disease Diagnosis]
        A --> C[Finance: Credit Scoring]
        A --> D[Marketing: Customer Segmentation]
        A --> E[Security: Fraud Detection]
        A --> F[Vision: Object Recognition]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 2.2 Logistic Regression

### Overview

**Logistic Regression** is a linear model used for binary classification. It applies the sigmoid function to linear regression to output probabilities.

### Sigmoid Function

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Properties:

  * Output range: $[0, 1]$ (interpretable as probability)
  * $z = 0$ yields $\sigma(z) = 0.5$
  * Smooth S-shaped curve

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Visualization
    z = np.linspace(-10, 10, 100)
    y = sigmoid(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, y, linewidth=2, label='σ(z)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold 0.5')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('z = w^T x', fontsize=12)
    plt.ylabel('σ(z)', fontsize=12)
    plt.title('Sigmoid Function', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    

### Model Definition

$$ P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}} $$

Prediction:

$$ \hat{y} = \begin{cases} 1 & \text{if } P(y=1 | \mathbf{x}) \geq 0.5 \\\ 0 & \text{otherwise} \end{cases} $$

### Loss Function: Cross-Entropy

$$ J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] $$

### Implementation Example
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("=== Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nWeights: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    # Decision boundary visualization
    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o',
                    edgecolors='k', s=80, label='Class 0')
        plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s',
                    edgecolors='k', s=80, label='Class 1')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title('Decision Boundary of Logistic Regression', fontsize=14)
        plt.legend()
        plt.show()
    
    plot_decision_boundary(model, X_test, y_test)
    

**Output** :
    
    
    === Logistic Regression ===
    Accuracy: 0.9550
    
    Weights: [2.14532851 1.87653214]
    Intercept: -0.2341
    

* * *

## 2.3 Decision Trees

### Overview

**Decision Trees** perform classification using a hierarchical structure of if-then-else rules. They recursively split data based on features.
    
    
    ```mermaid
    graph TD
        A[Feature 1 <= 0.5] -->|Yes| B[Feature 2 <= 1.2]
        A -->|No| C[Class 1]
        B -->|Yes| D[Class 0]
        B -->|No| E[Class 1]
    
        style A fill:#fff3e0
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#e3f2fd
        style E fill:#e8f5e9
    ```

### Splitting Criteria

**1\. Gini Impurity** :

$$ \text{Gini}(S) = 1 - \sum_{i=1}^{K} p_i^2 $$

  * $p_i$: Proportion of class $i$
  * Lower values indicate higher purity (biased toward one class)

**2\. Entropy** :

$$ \text{Entropy}(S) = -\sum_{i=1}^{K} p_i \log_2(p_i) $$

### Implementation Example
    
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    
    # Decision tree model
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Prediction
    y_pred_dt = dt_model.predict(X_test)
    
    print("=== Decision Tree ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    
    # Decision tree visualization
    plt.figure(figsize=(16, 10))
    plot_tree(dt_model, filled=True, feature_names=['Feature 1', 'Feature 2'],
              class_names=['Class 0', 'Class 1'], fontsize=10)
    plt.title('Decision Tree Structure', fontsize=16)
    plt.show()
    
    # Decision boundary
    plot_decision_boundary(dt_model, X_test, y_test)
    

**Output** :
    
    
    === Decision Tree ===
    Accuracy: 0.9450
    

### Feature Importance
    
    
    # Feature importance
    importances = dt_model.feature_importances_
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Feature 1', 'Feature 2'], importances, color=['#3498db', '#e74c3c'])
    plt.ylabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print(f"\nFeature 1 importance: {importances[0]:.4f}")
    print(f"Feature 2 importance: {importances[1]:.4f}")
    

* * *

## 2.4 k-Nearest Neighbors (k-NN)

### Overview

**k-NN** classifies by majority voting among the $k$ nearest training samples.

### Algorithm

  1. Calculate distances from test data $\mathbf{x}$ to all training data
  2. Select the $k$ nearest data points
  3. Predict the class with the most votes

### Distance Types

Distance | Formula  
---|---  
**Euclidean Distance** | $\sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$  
**Manhattan Distance** | $\sum_{i=1}^{n} |x_i - y_i|$  
**Minkowski Distance** | $\left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$  
  
### Implementation Example
    
    
    from sklearn.neighbors import KNeighborsClassifier
    
    # k-NN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    # Prediction
    y_pred_knn = knn_model.predict(X_test)
    
    print("=== k-NN (k=5) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    
    # Decision boundary
    plot_decision_boundary(knn_model, X_test, y_test)
    

**Output** :
    
    
    === k-NN (k=5) ===
    Accuracy: 0.9400
    

### Choosing k
    
    
    # Accuracy comparison for different k values
    k_range = range(1, 31)
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
    
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_scores, 'o-', label='Training Data', linewidth=2)
    plt.plot(k_range, test_scores, 's-', label='Test Data', linewidth=2)
    plt.xlabel('k (Number of Neighbors)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('k-NN: Relationship between k and Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_k = k_range[np.argmax(test_scores)]
    print(f"\nOptimal k: {best_k}")
    print(f"Best Accuracy: {max(test_scores):.4f}")
    

* * *

## 2.5 Support Vector Machine (SVM)

### Overview

**SVM** finds the decision boundary that maximizes the margin.
    
    
    ```mermaid
    graph LR
        A[SVM] --> B[Linear SVM]
        A --> C[Non-linear SVM Kernel Methods]
    
        B --> B1[Linearly Separable Data]
        C --> C1[RBF Kernel]
        C --> C2[Polynomial Kernel]
        C --> C3[Sigmoid Kernel]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

### Margin Maximization

$$ \text{maximize} \quad \frac{2}{||\mathbf{w}||} \quad \text{subject to} \quad y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 $$

### Kernel Trick

**RBF (Gaussian) Kernel** :

$$ K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\sigma^2}\right) $$

### Implementation Example
    
    
    from sklearn.svm import SVC
    
    # Linear SVM
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    
    # RBF SVM
    svm_rbf = SVC(kernel='rbf', gamma='auto')
    svm_rbf.fit(X_train, y_train)
    
    print("=== SVM (Linear Kernel) ===")
    print(f"Accuracy: {svm_linear.score(X_test, y_test):.4f}")
    
    print("\n=== SVM (RBF Kernel) ===")
    print(f"Accuracy: {svm_rbf.score(X_test, y_test):.4f}")
    
    # Decision boundary comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, model, title in zip(axes, [svm_linear, svm_rbf],
                                ['Linear SVM', 'RBF SVM']):
        h = 0.02
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                  c='blue', marker='o', edgecolors='k', s=80, label='Class 0')
        ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                  c='red', marker='s', edgecolors='k', s=80, label='Class 1')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title, fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === SVM (Linear Kernel) ===
    Accuracy: 0.9550
    
    === SVM (RBF Kernel) ===
    Accuracy: 0.9650
    

* * *

## 2.6 Evaluation of Classification Models

### Confusion Matrix

| Predicted: Positive | Predicted: Negative  
---|---|---  
**Actual: Positive** | TP (True Positive) | FN (False Negative)  
**Actual: Negative** | FP (False Positive) | TN (True Negative)  
  
### Evaluation Metrics

Metric | Formula | Meaning  
---|---|---  
**Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correct rate  
**Precision** | $\frac{TP}{TP + FP}$ | Accuracy of positive predictions  
**Recall** | $\frac{TP}{TP + FN}$ | Rate of capturing actual positives  
**F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall  
  
### Implementation Example
    
    
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    
    # Detailed evaluation report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                              target_names=['Class 0', 'Class 1']))
    

**Output** :
    
    
    === Classification Report ===
                  precision    recall  f1-score   support
    
         Class 0       0.96      0.95      0.95        99
         Class 1       0.95      0.96      0.96       101
    
        accuracy                           0.96       200
       macro avg       0.96      0.96      0.96       200
    weighted avg       0.96      0.96      0.96       200
    

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic) curve** shows the relationship between TPR (True Positive Rate) and FPR (False Positive Rate) as the threshold varies.

$$ \text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN} $$
    
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # ROC curve calculation
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    auc = roc_auc_score(y_test, y_proba[:, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"AUC: {auc:.4f}")
    

**Output** :
    
    
    AUC: 0.9876
    

> **AUC (Area Under the Curve)** : The area under the ROC curve. Values closer to 1 indicate a better model.

* * *

## 2.7 Chapter Summary

### What We Learned

  1. **Definition of Classification Problems**

     * Tasks predicting discrete values (categories)
     * Binary classification, multi-class classification, multi-label classification
  2. **Logistic Regression**

     * Probability output using sigmoid function
     * Cross-entropy loss
  3. **Decision Trees**

     * Hierarchical structure of if-then-else rules
     * Gini impurity, entropy
     * Feature importance
  4. **k-NN**

     * Majority voting among nearest neighbors
     * Importance of choosing k
  5. **SVM**

     * Margin maximization
     * Kernel trick
  6. **Evaluation Metrics**

     * Confusion matrix, accuracy, precision, recall, F1 score
     * ROC curve and AUC

### Next Chapter

In Chapter 3, we will learn about **Ensemble Methods** :

  * Principles of Bagging
  * Random Forest
  * Boosting (Gradient Boosting, XGBoost, LightGBM, CatBoost)

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

Explain situations where high accuracy may be inappropriate.

Solution

**Answer** :

In cases of **Imbalanced Data** , accuracy is inappropriate.

**Example** :

  * Cancer diagnosis data: 1% positive, 99% negative
  * Predicting all as "negative" gives 99% accuracy but is meaningless
  * Missing positive cases leads to serious consequences

**Appropriate metrics** :

  * Recall: Avoid missing positives
  * F1 Score: Balance between Precision and Recall
  * AUC: Threshold-independent evaluation

### Problem 2 (Difficulty: Medium)

Calculate accuracy, precision, recall, and F1 score from the following confusion matrix.

| Predicted: Positive | Predicted: Negative  
---|---|---  
Actual: Positive | 80 | 20  
Actual: Negative | 10 | 90  
Solution

**Answer** :
    
    
    TP = 80, FN = 20, FP = 10, TN = 90
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = (80 + 90) / (80 + 90 + 10 + 20)
             = 170 / 200 = 0.85 = 85%
    
    Precision = TP / (TP + FP)
              = 80 / (80 + 10)
              = 80 / 90 = 0.8889 = 88.89%
    
    Recall = TP / (TP + FN)
           = 80 / (80 + 20)
           = 80 / 100 = 0.80 = 80%
    
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
             = 2 * (0.8889 * 0.80) / (0.8889 + 0.80)
             = 2 * 0.7111 / 1.6889
             = 0.8421 = 84.21%
    

### Problem 3 (Difficulty: Medium)

Explain the problems when k is too small or too large when choosing the optimal k for k-NN.

Solution

**When k is too small (e.g., k=1)** :

  * **Overfitting** : Sensitive to noise
  * High accuracy on training data but low on test data
  * Complex, jagged decision boundary

**When k is too large (e.g., k=total data count)** :

  * **Excessive simplification** : Everything classified as the majority class
  * Decision boundary is too simple
  * Low accuracy on both training and test data

**Optimal k** :

  * Selected through cross-validation
  * Usually around $\sqrt{N}$ ($N$ is the number of data points)
  * Choose odd numbers (to avoid ties in binary classification)

### Problem 4 (Difficulty: Hard)

Explain the significance of using the kernel trick in SVM from a computational complexity perspective.

Solution

**Significance of the Kernel Trick** :

**Problem** : To classify non-linearly separable data, transformation to a higher-dimensional space is necessary.

**Direct approach** :

  * Explicitly transform features to higher dimensions: $\phi(\mathbf{x})$
  * Computational complexity: $O(d^2)$ or $O(d^3)$ ($d$ is the dimension)
  * Computation becomes infeasible for high dimensions

**Kernel trick** :

  * Directly compute the inner product $\langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ using the kernel function $K(\mathbf{x}, \mathbf{x}')$
  * No explicit high-dimensional transformation
  * Computational complexity: $O(d)$ (stays in original dimension)

**Example (RBF kernel)** :

  * Computes transformation to infinite dimensions in $O(d)$
  * $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$

**Conclusion** : The kernel trick enables efficient execution of high-dimensional computations in low-dimensional space.

### Problem 5 (Difficulty: Hard)

Implement logistic regression and perform binary classification on the iris dataset (setosa vs versicolor).

Solution
    
    
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load data
    iris = load_iris()
    X = iris.data[iris.target != 2]  # setosa (0) and versicolor (1) only
    y = iris.target[iris.target != 2]
    
    # Use only the first 2 features (for visualization)
    X = X[:, :2]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Prediction
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                              target_names=['setosa', 'versicolor']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['setosa', 'versicolor'],
                yticklabels=['setosa', 'versicolor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Decision boundary visualization
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train_scaled[y_train==0, 0], X_train_scaled[y_train==0, 1],
               c='blue', marker='o', edgecolors='k', s=80, label='setosa')
    plt.scatter(X_train_scaled[y_train==1, 0], X_train_scaled[y_train==1, 1],
               c='red', marker='s', edgecolors='k', s=80, label='versicolor')
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.title('Decision Boundary of Logistic Regression')
    plt.legend()
    plt.show()
    
    print(f"\nAccuracy: {model.score(X_test_scaled, y_test):.4f}")
    

**Output** :
    
    
    === Classification Report ===
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        10
      versicolor       1.00      1.00      1.00        10
    
        accuracy                           1.00        20
       macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20
    
    Accuracy: 1.0000
    

* * *

## References

  1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
  2. Murphy, K. P. (2012). _Machine Learning: A Probabilistic Perspective_. MIT Press.
  3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). _An Introduction to Statistical Learning_. Springer.
