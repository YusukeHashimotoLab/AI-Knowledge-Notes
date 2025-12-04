---
title: "Chapter 1: Evaluation Metrics"
chapter_title: "Chapter 1: Evaluation Metrics"
subtitle: Classification and Regression Metrics
---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/model-evaluation-introduction/chapter1-evaluation-metrics.html>) | Last sync: 2025-11-16

[ML Dojo](<../index.html>) > [Model Evaluation](<index.html>) > Ch1

## 1.1 Introduction

Comprehensive coverage of model evaluation techniques for machine learning.

**📐 Key Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, MSE, RMSE, R²

### 💻 Code Example 1: Metric Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, KFold
    
    # Classification metrics
    def evaluate_classification(y_true, y_pred):
        """Calculate classification metrics"""
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Regression metrics
    def evaluate_regression(y_true, y_pred):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'R2': r2}
    
    # Cross-validation
    def cross_validate_model(model, X, y, cv=5):
        """Perform k-fold cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    
    # Example usage
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    metrics = evaluate_classification(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

## 1.2-1.7 Additional Topics

Detailed coverage of evaluation strategies, cross-validation techniques, and model comparison methods.

### 💻 Code Examples 2-7
    
    
    # ROC curves, confusion matrices, learning curves
    # Hyperparameter tuning with cross-validation
    # Statistical significance testing
    # See complete implementations in full chapter

## 📝 Exercises

  1. Calculate precision, recall, and F1-score for multi-class classification.
  2. Implement stratified k-fold cross-validation for imbalanced dataset.
  3. Compare GridSearchCV vs RandomizedSearchCV for hyperparameter tuning.
  4. Perform paired t-test to compare two models statistically.
  5. Create learning curves to diagnose bias-variance tradeoff.

## Summary

  * Evaluation metrics: accuracy, precision, recall, F1, AUC-ROC
  * Cross-validation: k-fold, stratified, time series split
  * Hyperparameter tuning: grid search, random search, Bayesian optimization
  * Model comparison: statistical tests, confidence intervals
  * Proper evaluation prevents overfitting and ensures generalization

[← Overview](<index.html>) [Ch2 →](<chapter2-cross-validation.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
