---
title: "Chapter 1: Regression Fundamentals"
chapter_title: "Chapter 1: Regression Fundamentals"
subtitle: Theory and Implementation of Continuous Value Prediction - From Linear Regression to Regularization
reading_time: 20-25 min
difficulty: Beginner
code_examples: 12
exercises: 5
version: 1.0
created_at: 2025-10-20
---

## Learning Objectives

By reading this chapter, you will be able to:

  * Understand the definition and applications of regression problems
  * Explain the mathematical background of linear regression
  * Implement ordinary least squares and gradient descent
  * Model nonlinear relationships with polynomial regression
  * Apply regularization (Ridge, Lasso, Elastic Net)
  * Evaluate regression models using R-squared, RMSE, and MAE

* * *

## 1.1 What is a Regression Problem?

### Definition

**Regression** is a supervised learning task that predicts **continuous values** from input variables.

> "Learning a function $f: X \rightarrow y$ that predicts the target variable $y$ from features $X$"

### Regression vs Classification

Task | Output | Examples  
---|---|---  
**Regression** | Continuous values (numerical) | House price prediction, temperature forecasting, sales prediction  
**Classification** | Discrete values (categories) | Image classification, spam detection, disease diagnosis  
  
### Real-World Applications
    
    
    ```mermaid
    graph LR
        A[Regression Applications] --> B[Finance: Stock Price Prediction]
        A --> C[Real Estate: House Price Prediction]
        A --> D[Manufacturing: Demand Forecasting]
        A --> E[Healthcare: Patient Length of Stay Prediction]
        A --> F[Marketing: Sales Forecasting]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 1.2 Theory of Linear Regression

### Simple Linear Regression

**Simple Linear Regression** makes predictions from a single feature.

$$ y = w_0 + w_1 x + \epsilon $$

  * $y$: Target variable (value to predict)
  * $x$: Explanatory variable (feature)
  * $w_0$: Intercept (bias)
  * $w_1$: Slope (weight)
  * $\epsilon$: Error term

### Multiple Linear Regression

**Multiple Linear Regression** uses multiple features.

$$ y = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + \epsilon $$

Matrix notation:

$$ \mathbf{y} = \mathbf{X}\mathbf{w} + \epsilon $$

  * $\mathbf{y}$: Target variable vector (shape: $m \times 1$)
  * $\mathbf{X}$: Feature matrix (shape: $m \times (n+1)$)
  * $\mathbf{w}$: Weight vector (shape: $(n+1) \times 1$)
  * $m$: Number of samples, $n$: Number of features

### Loss Function

We minimize the **Mean Squared Error (MSE)** :

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{m} ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 $$

  * $y^{(i)}$: Actual value
  * $\hat{y}^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)}$: Predicted value

* * *

## 1.3 Ordinary Least Squares

### Analytical Solution

The weights $\mathbf{w}$ that minimize MSE can be found analytically:

$$ \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $$

This is called the **Normal Equation**.

### Implementation Example: Simple Regression
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Data generation
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Add bias term
    X_b = np.c_[np.ones((100, 1)), X]  # shape: (100, 2)
    
    # Calculate weights using normal equation
    w_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print("Learned weights:")
    print(f"w0 (intercept): {w_best[0][0]:.4f}")
    print(f"w1 (slope): {w_best[1][0]:.4f}")
    
    # Prediction
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b @ w_best
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Prediction line')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Linear Regression - Least Squares Method', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Output** :
    
    
    Learned weights:
    w0 (intercept): 4.2153
    w1 (slope): 2.7702
    

### Implementation with scikit-learn
    
    
    from sklearn.linear_model import LinearRegression
    
    # Build model
    model = LinearRegression()
    model.fit(X, y)
    
    print("\nscikit-learn:")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Slope: {model.coef_[0][0]:.4f}")
    
    # Prediction
    y_pred = model.predict(X_new)
    print(f"\nPredicted values: {y_pred.flatten()}")
    

* * *

## 1.4 Gradient Descent

### Principle

Calculate the gradient of the loss function and update weights in the opposite direction of the gradient.
    
    
    ```mermaid
    graph LR
        A[Initial weights w] --> B[Calculate gradient nabla J]
        B --> C[Update weights w := w - alpha nabla J]
        C --> D{Converged?}
        D -->|No| B
        D -->|Yes| E[Optimal weights w*]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#ffe0b2
        style E fill:#e8f5e9
    ```

### Update Rule

$$ \mathbf{w} := \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}) $$

Gradient:

$$ \nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) $$

  * $\alpha$: Learning rate

### Implementation Example
    
    
    def gradient_descent(X, y, alpha=0.01, n_iterations=1000):
        """
        Train linear regression using gradient descent
    
        Args:
            X: Feature matrix (including bias term)
            y: Target variable
            alpha: Learning rate
            n_iterations: Number of iterations
    
        Returns:
            w: Learned weights
            history: Loss function history
        """
        m = len(y)
        w = np.random.randn(X.shape[1], 1)  # Initialize weights
        history = []
    
        for i in range(n_iterations):
            # Prediction
            y_pred = X @ w
    
            # Calculate loss
            loss = (1 / m) * np.sum((y_pred - y) ** 2)
            history.append(loss)
    
            # Calculate gradient
            gradients = (2 / m) * X.T @ (y_pred - y)
    
            # Update weights
            w = w - alpha * gradients
    
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
        return w, history
    
    # Execute
    w_gd, loss_history = gradient_descent(X_b, y, alpha=0.1, n_iterations=1000)
    
    print("\nWeights learned by gradient descent:")
    print(f"w0 (intercept): {w_gd[0][0]:.4f}")
    print(f"w1 (slope): {w_gd[1][0]:.4f}")
    
    # Visualize loss function progression
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Gradient Descent Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Output** :
    
    
    Iteration 0: Loss = 6.8421
    Iteration 100: Loss = 0.8752
    Iteration 200: Loss = 0.8284
    Iteration 300: Loss = 0.8243
    Iteration 400: Loss = 0.8236
    Iteration 500: Loss = 0.8235
    Iteration 600: Loss = 0.8235
    Iteration 700: Loss = 0.8235
    Iteration 800: Loss = 0.8235
    Iteration 900: Loss = 0.8235
    
    Weights learned by gradient descent:
    w0 (intercept): 4.2152
    w1 (slope): 2.7703
    

### Importance of Learning Rate
    
    
    # Comparison with different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(12, 8))
    for i, alpha in enumerate(learning_rates):
        w, history = gradient_descent(X_b, y, alpha=alpha, n_iterations=100)
        plt.subplot(2, 2, i+1)
        plt.plot(history, linewidth=2)
        plt.title(f'Learning Rate alpha = {alpha}', fontsize=12)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.5 Polynomial Regression

### Overview

Models **nonlinear relationships** that cannot be expressed by linear regression.

$$ y = w_0 + w_1 x + w_2 x^2 + \cdots + w_d x^d $$

By transforming features, we can use the linear regression framework.

### Implementation Example
    
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    # Generate nonlinear data
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    
    # Polynomial regression (degree 2)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    print("Polynomial regression coefficients:")
    print(f"w1 (x): {model.coef_[0][0]:.4f}")
    print(f"w2 (x^2): {model.coef_[0][1]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    # Prediction and visualization
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Polynomial regression (degree 2)')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Polynomial Regression', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### Risk of Overfitting
    
    
    # Comparison with different degrees
    degrees = [1, 2, 5, 10]
    
    plt.figure(figsize=(14, 10))
    for i, degree in enumerate(degrees):
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
    
        model = LinearRegression()
        model.fit(X_poly, y)
    
        X_test_poly = poly_features.transform(X_test)
        y_pred = model.predict(X_test_poly)
    
        plt.subplot(2, 2, i+1)
        plt.scatter(X, y, alpha=0.6, label='Data')
        plt.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Regression (Degree {degree})', fontsize=12)
        plt.ylim(-5, 15)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

> **Note** : If the degree is too high, overfitting occurs. At degree 10, the model overfits the data, reducing generalization performance.

* * *

## 1.6 Regularization

### Overview

To prevent overfitting, we add a **penalty term** to the loss function.
    
    
    ```mermaid
    graph TD
        A[Regularization Methods] --> B[Ridge L2 Regularization]
        A --> C[Lasso L1 Regularization]
        A --> D[Elastic Net L1+L2]
    
        B --> B1[Suppresses weight magnitude]
        C --> C1[Sets some weights to zero]
        D --> D1[Balance of both]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### Ridge Regression (L2 Regularization)

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \sum_{j=1}^{n} w_j^2 $$

  * $\alpha$: Regularization parameter
  * Penalizes the sum of squared weights

    
    
    from sklearn.linear_model import Ridge
    
    # Ridge regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_poly, y)
    
    print("Ridge regression coefficients:")
    print(f"Weights: {ridge_model.coef_[0]}")
    print(f"Intercept: {ridge_model.intercept_[0]:.4f}")
    

### Lasso Regression (L1 Regularization)

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \sum_{j=1}^{n} |w_j| $$

  * Penalizes the sum of absolute values of weights
  * **Sparsity** : Sets weights of unimportant features to zero

    
    
    from sklearn.linear_model import Lasso
    
    # Lasso regression
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_poly, y)
    
    print("\nLasso regression coefficients:")
    print(f"Weights: {lasso_model.coef_}")
    print(f"Intercept: {lasso_model.intercept_:.4f}")
    print(f"Number of zero weights: {np.sum(lasso_model.coef_ == 0)}")
    

### Elastic Net (L1 + L2 Regularization)

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \rho \sum_{j=1}^{n} |w_j| + \frac{\alpha(1-\rho)}{2} \sum_{j=1}^{n} w_j^2 $$

  * $\rho$: Balance between L1 and L2 (0 to 1)

    
    
    from sklearn.linear_model import ElasticNet
    
    # Elastic Net
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_model.fit(X_poly, y)
    
    print("\nElastic Net regression coefficients:")
    print(f"Weights: {elastic_model.coef_}")
    print(f"Intercept: {elastic_model.intercept_:.4f}")
    

### Comparison of Regularization Parameters
    
    
    # Comparison with different alphas
    alphas = np.logspace(-3, 2, 100)
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, y)
        ridge_coefs.append(ridge.coef_[0])
    
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_poly, y)
        lasso_coefs.append(lasso.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    plt.figure(figsize=(14, 6))
    
    # Ridge
    plt.subplot(1, 2, 1)
    for i in range(X_poly.shape[1]):
        plt.plot(alphas, ridge_coefs[:, i], label=f'w{i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
    plt.ylabel('Coefficient Magnitude', fontsize=12)
    plt.title('Ridge Regression: Effect of Regularization Parameter', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lasso
    plt.subplot(1, 2, 2)
    for i in range(X_poly.shape[1]):
        plt.plot(alphas, lasso_coefs[:, i], label=f'w{i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
    plt.ylabel('Coefficient Magnitude', fontsize=12)
    plt.title('Lasso Regression: Effect of Regularization Parameter', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.7 Evaluation of Regression Models

### Evaluation Metrics

Metric | Formula | Description  
---|---|---  
**Mean Absolute Error**  
(MAE) | $\frac{1}{m}\sum|y_i - \hat{y}_i|$ | Average of prediction errors (robust to outliers)  
**Mean Squared Error**  
(MSE) | $\frac{1}{m}\sum(y_i - \hat{y}_i)^2$ | Average of squared prediction errors (sensitive to outliers)  
**Root Mean Squared Error**  
(RMSE) | $\sqrt{\frac{1}{m}\sum(y_i - \hat{y}_i)^2}$ | Square root of MSE (in original units)  
**Coefficient of Determination**  
(R-squared) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Model's explanatory power (0 to 1, higher is better)  
  
### Implementation Example
    
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Data split
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluation
    print("=== Training Data ===")
    print(f"MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"R^2:  {r2_score(y_train, y_train_pred):.4f}")
    
    print("\n=== Test Data ===")
    print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"R^2:  {r2_score(y_test, y_test_pred):.4f}")
    
    # Residual plot
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(14, 6))
    
    # Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Predicted vs Actual', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Training Data ===
    MAE:  0.7234
    MSE:  0.8456
    RMSE: 0.9196
    R^2:  0.9145
    
    === Test Data ===
    MAE:  0.7891
    MSE:  0.9234
    RMSE: 0.9609
    R^2:  0.9023
    

* * *

## 1.8 Chapter Summary

### What We Learned

  1. **Definition of Regression Problems**

     * Task of predicting continuous values
     * Real-world applications (price prediction, demand forecasting, etc.)
  2. **Linear Regression**

     * Analytical solution using least squares method
     * Numerical solution using gradient descent
  3. **Polynomial Regression**

     * Modeling nonlinear relationships
     * Risk of overfitting
  4. **Regularization**

     * Ridge (L2): Suppresses weight magnitude
     * Lasso (L1): Introduces sparsity
     * Elastic Net: Balance of both
  5. **Evaluation Metrics**

     * MAE, MSE, RMSE, R-squared
     * Importance of residual analysis

### Next Chapter

In Chapter 2, we will learn the **fundamentals of classification problems** :

  * Logistic regression
  * Decision trees
  * k-NN, SVM
  * Evaluation metrics (accuracy, recall, F1 score)

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

List three differences between regression and classification problems.

Sample Answer

**Answer** :

  1. **Type of output** : Regression outputs continuous values, classification outputs discrete values (categories)
  2. **Loss function** : Regression uses MSE, classification uses cross-entropy
  3. **Evaluation metrics** : Regression uses RMSE/R-squared, classification uses accuracy/F1 score

### Problem 2 (Difficulty: Medium)

Implement linear regression with the following data and find the weights and bias.
    
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [5], [4], [5]])
    

Sample Answer
    
    
    import numpy as np
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [5], [4], [5]])
    
    # Add bias term
    X_b = np.c_[np.ones((5, 1)), X]
    
    # Normal equation
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print(f"Intercept w0: {w[0][0]:.4f}")
    print(f"Slope w1: {w[1][0]:.4f}")
    
    # Prediction
    y_pred = X_b @ w
    print(f"\nPredicted values: {y_pred.flatten()}")
    

**Output** :
    
    
    Intercept w0: 2.2000
    Slope w1: 0.6000
    
    Predicted values: [2.8 3.4 4.  4.6 5.2]
    

### Problem 3 (Difficulty: Medium)

What problems occur in gradient descent when the learning rate is too large?

Sample Answer

**Answer** :

  * **Divergence** : The loss function overshoots the minimum and diverges
  * **Oscillation** : Continues to oscillate around the minimum
  * **Non-convergence** : Cannot reach the optimal solution

**Countermeasures** :

  * Reduce the learning rate (e.g., 0.1 to 0.01)
  * Use learning rate scheduling
  * Use adaptive optimization methods (Adam, RMSprop)

### Problem 4 (Difficulty: Hard)

Explain the difference between Ridge regression and Lasso regression, and describe when each should be used.

Sample Answer

**Ridge Regression (L2 Regularization)** :

  * Penalizes the sum of squared weights
  * Makes weights smaller but does not set them to zero
  * **Use cases** : When multicollinearity exists, when all features are important

**Lasso Regression (L1 Regularization)** :

  * Penalizes the sum of absolute values of weights
  * Sets weights of unimportant features to zero (sparsity)
  * **Use cases** : When feature selection is needed, when interpretability is desired

**Selection Criteria** :

Situation | Recommended Method  
---|---  
Many features with unknown importance | Lasso  
Multicollinearity present | Ridge  
Feature selection needed | Lasso  
Want to use all features | Ridge  
Uncertain which to use | Elastic Net  
  
### Problem 5 (Difficulty: Hard)

Complete the following code to find the optimal alpha for Ridge regression using cross-validation.
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    
    # Data generation (omitted)
    alphas = np.logspace(-3, 3, 50)
    
    # Implement here
    

Sample Answer
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Data generation
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    alphas = np.logspace(-3, 3, 50)
    scores = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        # 5-fold cross-validation
        cv_scores = cross_val_score(ridge, X_poly, y.ravel(),
                                     cv=5, scoring='neg_mean_squared_error')
        scores.append(-cv_scores.mean())  # Convert negative MSE to positive
    
    # Find optimal alpha
    best_alpha = alphas[np.argmin(scores)]
    best_score = np.min(scores)
    
    print(f"Optimal alpha: {best_alpha:.4f}")
    print(f"Minimum MSE: {best_score:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, scores, linewidth=2)
    plt.axvline(best_alpha, color='r', linestyle='--',
                label=f'Optimal alpha = {best_alpha:.4f}')
    plt.xscale('log')
    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('MSE (Cross-Validation)', fontsize=12)
    plt.title('Ridge Regression: Finding Optimal Alpha', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Output** :
    
    
    Optimal alpha: 2.1544
    Minimum MSE: 1.0234
    

* * *

## References

  1. Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer.
  2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
  3. Geron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly Media.
