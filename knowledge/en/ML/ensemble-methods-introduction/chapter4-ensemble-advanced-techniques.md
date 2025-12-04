---
title: "Chapter 4: Advanced Ensemble Techniques"
chapter_title: "Chapter 4: Advanced Ensemble Techniques"
subtitle: Stacking, Blending, and Voting Ensembles
---

🌐 EN | 日本語 (準備中) Last sync: 2025-11-16

[ML Dojo](<../index.html>) > [Ensemble Methods](<index.html>) > Ch4

## 4.1 Stacking (Stacked Generalization)

Stacking combines multiple models using meta-learner trained on base model predictions.

**📐 Stacking Process:** Level 0: Base models $\\{f_1, f_2, ..., f_n\\}$ Level 1: Meta-model $g$ learns from base predictions $$\hat{y} = g(f_1(x), f_2(x), ..., f_n(x))$$

### 💻 Code Example 1: Stacking Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.metrics import accuracy_score
    
    class StackingEnsemble:
        """Stacking ensemble implementation"""
        
        def __init__(self, base_models, meta_model, cv=5):
            self.base_models = base_models
            self.meta_model = meta_model
            self.cv = cv
        
        def fit(self, X_train, y_train):
            """Train stacking ensemble"""
            # Generate out-of-fold predictions for training meta-model
            meta_features = np.zeros((X_train.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                # Cross-validated predictions
                predictions = cross_val_predict(
                    model, X_train, y_train,
                    cv=self.cv, method='predict_proba'
                )
                meta_features[:, i] = predictions[:, 1]  # Probability of positive class
                
                # Train on full training set
                model.fit(X_train, y_train)
            
            # Train meta-model
            self.meta_model.fit(meta_features, y_train)
            return self
        
        def predict(self, X_test):
            """Make predictions using stacking ensemble"""
            # Get base model predictions
            meta_features = np.zeros((X_test.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                predictions = model.predict_proba(X_test)
                meta_features[:, i] = predictions[:, 1]
            
            # Meta-model prediction
            return self.meta_model.predict(meta_features)
    
    # Example usage
    from sklearn.datasets import load_breast_cancer
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # Define meta-model
    meta_model = LogisticRegression()
    
    # Train stacking ensemble
    stacking = StackingEnsemble(base_models, meta_model, cv=5)
    stacking.fit(X_train, y_train)
    
    # Evaluate
    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stacking Ensemble Accuracy: {accuracy:.4f}")

## 4.2-4.7 More Advanced Topics

Blending, voting classifiers, model diversity, hyperparameter optimization, AutoML ensembles.

### 💻 Code Examples 2-7
    
    
    # Blending implementation
    # Soft and hard voting
    # Measuring model diversity
    # Multi-level stacking
    # AutoML ensemble strategies
    # Production deployment
    # See full implementations in complete chapter

## 📝 Exercises

  1. Implement 2-level stacking with diverse base models.
  2. Compare stacking vs blending on same dataset.
  3. Create voting ensemble and analyze soft vs hard voting.
  4. Measure correlation between base model predictions.
  5. Build AutoML-style ensemble with automated model selection.

## Summary

  * Stacking: meta-model learns from base model predictions
  * Blending: simpler alternative to stacking with hold-out set
  * Voting: majority vote (hard) or average probabilities (soft)
  * Model diversity crucial for ensemble performance
  * Advanced ensembles often win Kaggle competitions
  * Trade-off: performance vs complexity and interpretability

[← Ch3: LightGBM/CatBoost](<chapter3-lightgbm-catboost.html>) [Overview →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
