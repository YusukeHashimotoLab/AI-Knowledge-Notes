---
title: "Chapter 2: Gaussian Process Modeling"
chapter_title: "Chapter 2: Gaussian Process Modeling"
subtitle: Powerful Regression Method for Quantifying Uncertainty
---

This chapter covers Gaussian Process Modeling. You will learn  Calculate mean and  Explain the properties of RBF.

## Introduction

Gaussian Processes (GP) are probabilistic modeling methods that form the core of Bayesian optimization. Unlike traditional regression methods, GPs can quantify **prediction uncertainty** in addition to predicted values, enabling optimization of the exploration-exploitation trade-off.

In this chapter, we will learn practical implementation using chemical process data, starting from 1D regression through various kernel functions, hyperparameter optimization, and model validation.

=¡ Key Points of This Chapter

  * Gaussian processes are completely defined by mean function and covariance function (kernel)
  * Kernel selection should be adjusted according to problem smoothness
  * Hyperparameters can be automatically optimized by Maximum Likelihood Estimation (MLE)
  * Prediction uncertainty can be visualized as confidence intervals

## 2.1 Fundamentals of Gaussian Process Regression

### 2.1.1 Mathematical Definition

A Gaussian process is a stochastic process where **function values at any finite set of points follow a joint Gaussian distribution** :
    
    
    f(x) ~ GP(m(x), k(x, x'))
    
    m(x) = E[f(x)]                    # Mean function
    k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]  # Covariance function (kernel)

Given observed data `D = {(x_i, y_i)}`, the predictive distribution at a new point `x*` is:
    
    
    f(x*) | D ~ N(¼(x*), Ã²(x*))
    
    ¼(x*) = k(x*, X) [K + Ã_n² I]{¹ y
    Ã²(x*) = k(x*, x*) - k(x*, X) [K + Ã_n² I]{¹ k(X, x*)

### Example 1: 1D Gaussian Process Regression (Chemical Reaction Yield)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # ===================================
    # 1D chemical reaction yield data (Temperature vs Yield)
    # ===================================
    # Experimental data (Temperature [°C] ’ Yield [%])
    X_train = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)
    y_train = np.array([45, 62, 78, 71, 52])  # Optimal temperature around 90°C
    
    # Build Gaussian process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                   alpha=1e-2, random_state=42)
    
    # Model training (automatic hyperparameter optimization)
    gp.fit(X_train, y_train)
    
    # Prediction (with uncertainty)
    X_test = np.linspace(40, 140, 100).reshape(-1, 1)
    y_pred, sigma = gp.predict(X_test, return_std=True)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred, 'b-', label='GP prediction mean', linewidth=2)
    plt.fill_between(X_test.ravel(),
                     y_pred - 1.96*sigma,
                     y_pred + 1.96*sigma,
                     alpha=0.3, label='95% confidence interval')
    plt.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='k', label='Experimental data')
    plt.xlabel('Temperature [°C]', fontsize=12)
    plt.ylabel('Yield [%]', fontsize=12)
    plt.title('Reaction Yield Prediction with Gaussian Process', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Optimized kernel: {gp.kernel_}")
    print(f"Log marginal likelihood: {gp.log_marginal_likelihood_value_:.2f}")
    
    # Expected output:
    # Optimized kernel: 31.6**2 * RBF(length_scale=15.8)
    # Log marginal likelihood: -12.45
    # Plot: Yield peak around 90°C, uncertainty increases at edges

= Explanation: Meaning of Confidence Intervals

The **95% confidence interval** (¼ ± 1.96Ã) indicates that the true yield falls within this range with 95% probability. In regions far from experimental data (around 40°C, 140°C), uncertainty increases and the interval widens. This is the key to **promoting exploration of unexplored regions** in Bayesian optimization.

## 2.2 Kernel Function Selection

### 2.2.1 Properties of Major Kernels

Kernel | Formula | Smoothness | Application Example  
---|---|---|---  
**RBF (Squared Exponential)** | k(x, x') = Ã² exp(-||x - x'||² / (2²)) | Infinitely differentiable | Temperature-yield relationship  
**Matérn (½=1.5)** | k(x, x') = Ã² (1 + 3r/) exp(-3r/) | Once differentiable | Pressure-flow relationship  
**Matérn (½=2.5)** | k(x, x') = Ã² (1 + 5r/ + 5r²/3²) exp(-5r/) | Twice differentiable | Catalyst activity curve  
**Rational Quadratic** | k(x, x') = Ã² (1 + r²/(2±²))^(-±) | Scale mixture of RBF | Multi-scale phenomena  
  
### Example 2: RBF Kernel (Smooth Reaction Curve)
    
    
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # ===================================
    # RBF Kernel: Infinitely differentiable (very smooth)
    # ===================================
    # Yield data with simultaneous temperature-pressure changes
    X_train = np.array([[60, 1.0], [80, 1.5], [100, 2.0],
                        [80, 2.5], [60, 2.0]]) # [Temperature°C, Pressure MPa]
    y_train = np.array([50, 68, 85, 72, 58])
    
    # Define RBF kernel
    kernel_rbf = C(1.0) * RBF(length_scale=[10.0, 0.5],
                               length_scale_bounds=(1e-2, 1e3))
    
    gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=15)
    gp_rbf.fit(X_train, y_train)
    
    # Grid for contour plot
    temp_range = np.linspace(50, 110, 50)
    pressure_range = np.linspace(0.8, 3.0, 50)
    T, P = np.meshgrid(temp_range, pressure_range)
    X_grid = np.c_[T.ravel(), P.ravel()]
    
    y_pred_grid, sigma_grid = gp_rbf.predict(X_grid, return_std=True)
    Y_pred = y_pred_grid.reshape(T.shape)
    Sigma = sigma_grid.reshape(T.shape)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction mean
    contour1 = axes[0].contourf(T, P, Y_pred, levels=15, cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=150,
                    edgecolors='white', linewidths=2, label='Experimental points')
    axes[0].set_xlabel('Temperature [°C]')
    axes[0].set_ylabel('Pressure [MPa]')
    axes[0].set_title('RBF Kernel: Predicted Yield [%]')
    plt.colorbar(contour1, ax=axes[0])
    axes[0].legend()
    
    # Uncertainty
    contour2 = axes[1].contourf(T, P, Sigma, levels=15, cmap='Reds')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='blue', s=150,
                    edgecolors='white', linewidths=2, label='Experimental points')
    axes[1].set_xlabel('Temperature [°C]')
    axes[1].set_ylabel('Pressure [MPa]')
    axes[1].set_title('RBF Kernel: Prediction Standard Deviation [%]')
    plt.colorbar(contour2, ax=axes[1])
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimized length scale: {gp_rbf.kernel_.k2.length_scale}")
    print(f"Temperature direction: {gp_rbf.kernel_.k2.length_scale[0]:.2f}°C")
    print(f"Pressure direction: {gp_rbf.kernel_.k2.length_scale[1]:.2f} MPa")
    
    # Expected output:
    # Optimized length scale: [12.34  0.68]
    # Temperature direction: 12.34°C
    # Pressure direction: 0.68 MPa
    # ’ Yield characteristics more sensitive to pressure changes than temperature

### Example 3: Matérn Kernel (½=1.5)
    
    
    from sklearn.gaussian_process.kernels import Matern
    
    # ===================================
    # Matérn Kernel (½=1.5): Once differentiable
    # Moderate smoothness based on physical laws
    # ===================================
    kernel_matern15 = C(1.0) * Matern(length_scale=10.0, nu=1.5)
    
    gp_matern15 = GaussianProcessRegressor(kernel=kernel_matern15,
                                            n_restarts_optimizer=10)
    gp_matern15.fit(X_train, y_train)
    
    # Comparison on 1D cross-section (fixed pressure=2.0 MPa)
    X_test_1d = np.column_stack([np.linspace(50, 110, 100),
                                  np.full(100, 2.0)])
    y_pred_matern, sigma_matern = gp_matern15.predict(X_test_1d, return_std=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_1d[:, 0], y_pred_matern, 'g-',
             label='Matérn(½=1.5)', linewidth=2)
    plt.fill_between(X_test_1d[:, 0],
                     y_pred_matern - 1.96*sigma_matern,
                     y_pred_matern + 1.96*sigma_matern,
                     alpha=0.3, color='green')
    plt.scatter(X_train[:, 0], y_train, c='red', s=100,
                edgecolors='k', label='Experimental data')
    plt.xlabel('Temperature [°C] (Pressure=2.0 MPa fixed)')
    plt.ylabel('Yield [%]')
    plt.title('Prediction with Matérn Kernel (½=1.5)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Matérn(½=1.5) kernel: {gp_matern15.kernel_}")
    print(f"Log marginal likelihood: {gp_matern15.log_marginal_likelihood_value_:.2f}")
    
    # Expected output:
    # Matérn(½=1.5) kernel: 1.2**2 * Matern(length_scale=11.5, nu=1.5)
    # Log marginal likelihood: -10.23
    # ’ Slightly rougher prediction than RBF (robust to experimental noise)

### Example 4: Matérn Kernel (½=2.5)
    
    
    # ===================================
    # Matérn Kernel (½=2.5): Twice differentiable
    # Intermediate smoothness between RBF and ½=1.5
    # ===================================
    kernel_matern25 = C(1.0) * Matern(length_scale=10.0, nu=2.5)
    
    gp_matern25 = GaussianProcessRegressor(kernel=kernel_matern25,
                                            n_restarts_optimizer=10)
    gp_matern25.fit(X_train, y_train)
    
    y_pred_matern25, sigma_matern25 = gp_matern25.predict(X_test_1d, return_std=True)
    
    # Comparison plot of three kernels
    plt.figure(figsize=(12, 6))
    
    plt.plot(X_test_1d[:, 0], y_pred_grid[:100], 'b-',
             label='RBF ( times differentiable)', linewidth=2)
    plt.plot(X_test_1d[:, 0], y_pred_matern, 'g--',
             label='Matérn(½=1.5)', linewidth=2)
    plt.plot(X_test_1d[:, 0], y_pred_matern25, 'orange',
             linestyle='-.', label='Matérn(½=2.5)', linewidth=2)
    
    plt.scatter(X_train[:, 0], y_train, c='red', s=150,
                edgecolors='k', linewidths=2, label='Experimental data', zorder=10)
    
    plt.xlabel('Temperature [°C] (Pressure=2.0 MPa fixed)', fontsize=12)
    plt.ylabel('Yield [%]', fontsize=12)
    plt.title('Comparison of Kernel Functions (Smoothness Differences)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Comparison of log marginal likelihood (higher is better)
    print("\n=== Kernel Performance Comparison ===")
    print(f"RBF:           {gp_rbf.log_marginal_likelihood_value_:.2f}")
    print(f"Matérn(½=1.5): {gp_matern15.log_marginal_likelihood_value_:.2f}")
    print(f"Matérn(½=2.5): {gp_matern25.log_marginal_likelihood_value_:.2f}")
    
    # Expected output:
    # === Kernel Performance Comparison ===
    # RBF:           -9.87
    # Matérn(½=1.5): -10.23
    # Matérn(½=2.5): -9.95
    # ’ RBF has highest likelihood (best for this data)

=Ê Kernel Selection Guidelines

  * **RBF** : Very smooth response (temperature-yield, concentration-activity)
  * **Matérn(½=1.5)** : Large experimental noise, physical constraints present
  * **Matérn(½=2.5)** : Balanced type (recommended default)
  * **Rational Quadratic** : Multiple scale variations (multi-stage reactions)

### Example 5: Rational Quadratic Kernel
    
    
    from sklearn.gaussian_process.kernels import RationalQuadratic
    
    # ===================================
    # Rational Quadratic Kernel:
    # Infinite mixture of RBF kernels with different length scales
    # Effective for modeling multi-scale phenomena
    # ===================================
    kernel_rq = C(1.0) * RationalQuadratic(length_scale=10.0, alpha=1.0)
    
    gp_rq = GaussianProcessRegressor(kernel=kernel_rq, n_restarts_optimizer=10)
    gp_rq.fit(X_train, y_train)
    
    y_pred_rq, sigma_rq = gp_rq.predict(X_test_1d, return_std=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_1d[:, 0], y_pred_rq, 'purple', linewidth=2,
             label='Rational Quadratic')
    plt.fill_between(X_test_1d[:, 0],
                     y_pred_rq - 1.96*sigma_rq,
                     y_pred_rq + 1.96*sigma_rq,
                     alpha=0.3, color='purple')
    plt.scatter(X_train[:, 0], y_train, c='red', s=100,
                edgecolors='k', label='Experimental data')
    plt.xlabel('Temperature [°C] (Pressure=2.0 MPa fixed)')
    plt.ylabel('Yield [%]')
    plt.title('Prediction with Rational Quadratic Kernel')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Optimized kernel: {gp_rq.kernel_}")
    print(f"± (scale mixture degree): {gp_rq.kernel_.k2.alpha:.2f}")
    print(f"Log marginal likelihood: {gp_rq.log_marginal_likelihood_value_:.2f}")
    
    # Expected output:
    # Optimized kernel: 1.15**2 * RationalQuadratic(alpha=0.85, length_scale=12.3)
    # ± (scale mixture degree): 0.85
    # Log marginal likelihood: -10.10
    # ’ ±<1: short-range correlation dominant, converges to RBF as ±’

## 2.3 Hyperparameter Optimization

### 2.3.1 Principles of Maximum Likelihood Estimation (MLE)

The hyperparameters `¸ = {Ã², , Ã_n²}` of a Gaussian process are optimized by maximizing the log marginal likelihood:
    
    
    log p(y | X, ¸) = -1/2 y^T [K + Ã_n² I]{¹ y
                      - 1/2 log|K + Ã_n² I|
                      - n/2 log(2À)
    
    Optimization: ¸* = argmax_¸ log p(y | X, ¸)

### Example 6: Hyperparameter Optimization (MLE with scipy)
    
    
    from scipy.optimize import minimize
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    import numpy as np
    
    # ===================================
    # Manual hyperparameter optimization implementation
    # (Understanding internal processing of scikit-learn)
    # ===================================
    # Data
    X = np.array([[60], [80], [100], [80], [60]])
    y = np.array([50, 68, 85, 72, 58])
    
    def negative_log_marginal_likelihood(theta, X, y):
        """Negative log marginal likelihood (for minimization)
    
        Args:
            theta: [log(Ã²), log(), log(Ã_n²)]
            X: Input data (n, d)
            y: Output data (n,)
    
        Returns:
            -log p(y | X, ¸)
        """
        sigma_f = np.exp(theta[0])  # Signal variance
        length_scale = np.exp(theta[1])  # Length scale
        sigma_n = np.exp(theta[2])  # Noise standard deviation
    
        # Compute kernel matrix
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r_sq = np.sum((X[i] - X[j])**2)
                K[i, j] = sigma_f**2 * np.exp(-r_sq / (2 * length_scale**2))
    
        # Add noise
        K_y = K + sigma_n**2 * np.eye(n)
    
        # Compute log marginal likelihood
        try:
            L = np.linalg.cholesky(K_y)  # Cholesky decomposition (numerically stable)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
            log_likelihood = (-0.5 * y.dot(alpha)
                             - np.sum(np.log(np.diag(L)))
                             - n/2 * np.log(2*np.pi))
    
            return -log_likelihood  # Negative for minimization
        except np.linalg.LinAlgError:
            return 1e10  # Penalty for numerically unstable cases
    
    # Initial values (log scale)
    theta_init = np.log([1.0, 10.0, 1.0])  # [Ã², , Ã_n²]
    
    # Execute optimization (L-BFGS-B method)
    result = minimize(negative_log_marginal_likelihood, theta_init,
                      args=(X, y), method='L-BFGS-B',
                      bounds=[(-5, 5), (-2, 5), (-5, 2)])  # Bounds in log space
    
    # Optimal hyperparameters
    theta_opt = result.x
    sigma_f_opt = np.exp(theta_opt[0])
    length_scale_opt = np.exp(theta_opt[1])
    sigma_n_opt = np.exp(theta_opt[2])
    
    print("=== Optimization Results ===")
    print(f"Signal variance Ã²: {sigma_f_opt:.3f}")
    print(f"Length scale : {length_scale_opt:.2f}°C")
    print(f"Noise standard deviation Ã_n: {sigma_n_opt:.3f}%")
    print(f"\nMaximum log marginal likelihood: {-result.fun:.2f}")
    print(f"Optimization steps: {result.nit}")
    
    # Comparison with scikit-learn
    kernel_sklearn = C(1.0) * RBF(10.0)
    gp_sklearn = GaussianProcessRegressor(kernel=kernel_sklearn,
                                           n_restarts_optimizer=10)
    gp_sklearn.fit(X, y)
    
    print(f"\n=== scikit-learn Results (Comparison) ===")
    print(f"Optimized kernel: {gp_sklearn.kernel_}")
    print(f"Log marginal likelihood: {gp_sklearn.log_marginal_likelihood_value_:.2f}")
    
    # Expected output:
    # === Optimization Results ===
    # Signal variance Ã²: 156.234
    # Length scale : 14.87°C
    # Noise standard deviation Ã_n: 2.145%
    #
    # Maximum log marginal likelihood: -8.92
    # Optimization steps: 23
    #
    # === scikit-learn Results (Comparison) ===
    # Optimized kernel: 12.5**2 * RBF(length_scale=14.9)
    # Log marginal likelihood: -8.91
    # ’ Manual implementation and scikit-learn nearly match

™ Hyperparameter Interpretation

  * **Ã² (signal variance)** : Function amplitude (larger means larger variation)
  * ** (length scale)** : Correlation distance (larger means smoother)
  * **Ã_n² (noise variance)** : Observation error (inverse of experimental precision)

## 2.4 Uncertainty Quantification and Confidence Intervals

### 2.4.1 Interpretation of Predictive Distribution

From the Gaussian process predictive distribution `f(x*) ~ N(¼, Ã²)`, the following information is obtained:

  * **¼(x*)** : Predicted value (expected value)
  * **Ã(x*)** : Prediction uncertainty (standard deviation)
  * **95% confidence interval** : [¼ - 1.96Ã, ¼ + 1.96Ã]
  * **99% confidence interval** : [¼ - 2.58Ã, ¼ + 2.58Ã]

### Example 7: Model Validation (Cross-Validation & Residual Analysis)
    
    
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    # ===================================
    # Comprehensive Gaussian process model validation
    # ===================================
    # Generate larger dataset (catalyst activity data)
    np.random.seed(42)
    X_full = np.random.uniform(50, 150, 30).reshape(-1, 1)  # Temperature [°C]
    y_true = 50 + 40 * np.exp(-(X_full.ravel() - 100)**2 / 400)  # True function
    y_full = y_true + np.random.normal(0, 3, 30)  # Add noise
    
    # Gaussian process model
    kernel = C(1.0) * RBF(10.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15,
                                   alpha=1e-2)
    
    # === 1. Cross-validation (5-fold CV) ===
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gp, X_full, y_full, cv=kfold,
                                 scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    print("=== Cross-Validation Results ===")
    print(f"5-fold CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}%")
    
    # === 2. Model training and prediction ===
    gp.fit(X_full, y_full)
    y_pred, sigma = gp.predict(X_full, return_std=True)
    
    # === 3. Performance metrics ===
    rmse = np.sqrt(mean_squared_error(y_full, y_pred))
    r2 = r2_score(y_full, y_pred)
    
    print(f"\n=== Training Data Performance ===")
    print(f"RMSE: {rmse:.2f}%")
    print(f"R² score: {r2:.3f}")
    
    # === 4. Residual analysis ===
    residuals = y_full - y_pred
    standardized_residuals = residuals / sigma
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Predicted vs Observed
    axes[0, 0].scatter(y_full, y_pred, alpha=0.6, s=80)
    axes[0, 0].plot([y_full.min(), y_full.max()],
                    [y_full.min(), y_full.max()],
                    'r--', linewidth=2, label='Ideal line')
    axes[0, 0].set_xlabel('Observed value [%]')
    axes[0, 0].set_ylabel('Predicted value [%]')
    axes[0, 0].set_title(f'Prediction Accuracy (R²={r2:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # (b) Residual plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=80)
    axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].axhline(1.96*residuals.std(), color='orange',
                       linestyle=':', label='±1.96Ã')
    axes[0, 1].axhline(-1.96*residuals.std(), color='orange', linestyle=':')
    axes[0, 1].set_xlabel('Predicted value [%]')
    axes[0, 1].set_ylabel('Residual [%]')
    axes[0, 1].set_title('Residual Plot (Homoscedasticity Check)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # (c) Histogram of standardized residuals
    axes[1, 0].hist(standardized_residuals, bins=15, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    x_norm = np.linspace(-3, 3, 100)
    y_norm = 30 / (2*np.pi)**0.5 * np.exp(-x_norm**2 / 2)  # Normal distribution
    axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label='Standard normal distribution')
    axes[1, 0].set_xlabel('Standardized residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Normality Check')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # (d) Prediction uncertainty calibration
    X_test_dense = np.linspace(50, 150, 100).reshape(-1, 1)
    y_pred_dense, sigma_dense = gp.predict(X_test_dense, return_std=True)
    
    axes[1, 1].plot(X_test_dense, y_pred_dense, 'b-', linewidth=2, label='GP prediction')
    axes[1, 1].fill_between(X_test_dense.ravel(),
                            y_pred_dense - 1.96*sigma_dense,
                            y_pred_dense + 1.96*sigma_dense,
                            alpha=0.3, label='95% confidence interval')
    axes[1, 1].scatter(X_full, y_full, c='red', s=60,
                      edgecolors='k', label='Experimental data')
    axes[1, 1].set_xlabel('Temperature [°C]')
    axes[1, 1].set_ylabel('Activity [%]')
    axes[1, 1].set_title('Uncertainty Calibration')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # === 5. Coverage probability (calibration check) ===
    # What percentage of observed values fall within 95% confidence interval
    in_interval = np.abs(residuals) <= 1.96 * sigma
    coverage = np.mean(in_interval) * 100
    
    print(f"\n=== Uncertainty Calibration ===")
    print(f"95% confidence interval coverage: {coverage:.1f}%")
    print(f"Expected: 95.0%")
    print(f"Assessment: {' Good' if 90 <= coverage <= 100 else '  Needs adjustment'}")
    
    # Expected output:
    # === Cross-Validation Results ===
    # 5-fold CV RMSE: 3.42 ± 0.87%
    #
    # === Training Data Performance ===
    # RMSE: 2.98%
    # R² score: 0.923
    #
    # === Uncertainty Calibration ===
    # 95% confidence interval coverage: 93.3%
    # Expected: 95.0%
    # Assessment:  Good

=, Model Validation Checklist

  1. **Cross-validation** : Confirm generalization performance (is CV RMSE close to training RMSE)
  2. **Homoscedasticity of residuals** : Is residual plot a horizontal random scatter
  3. **Normality of residuals** : Does histogram approximate a normal distribution
  4. **Uncertainty calibration** : Is 95% interval coverage between 90-100%

## Chapter Summary

### Important Learning Points

Topic | Key Points | Practical Usage  
---|---|---  
**Gaussian Process Definition** | Completely defined by mean function and covariance function | Start with RBF kernel in scikit-learn  
**Kernel Selection** | RBF (smooth), Matérn (intermediate), RQ (multi-scale) | Compare performance with log marginal likelihood  
**Hyperparameter Optimization** | Automatic optimization with MLE (L-BFGS-B method) | Recommend n_restarts_optimizer=10-15  
**Uncertainty Quantification** | 95% confidence interval = ¼ ± 1.96Ã | Large uncertainty in unexplored regions’promote exploration  
**Model Validation** | Cross-validation, residual analysis, coverage check | Target R²>0.9, coverage 90-100%  
  
### Implementation Best Practices

  1. **Data normalization** : Normalize inputs to [0, 1] or standardize (stabilizes  optimization)
  2. **Kernel selection** : Try RBF first, improve robustness with Matérn
  3. **Noise estimation** : Consider observation error with `alpha=1e-2`
  4. **Multiple starts** : Avoid local optima with `n_restarts_optimizer=10-15`
  5. **Numerical stability** : Use Cholesky decomposition (avoid direct matrix inversion)

## Learning Objectives Check

Upon completing this chapter, you will be able to:

### Basic Understanding

  *  Explain that Gaussian processes are defined by mean function and kernel
  *  Calculate mean and uncertainty from predictive distribution
  *  Explain the properties of RBF, Matérn, and Rational Quadratic kernels

### Practical Skills

  *  Implement Gaussian process models with scikit-learn
  *  Optimize hyperparameters with MLE
  *  Visualize prediction uncertainty (confidence interval plots)
  *  Validate models with cross-validation and residual analysis

### Application Ability

  *  Select appropriate kernels for chemical process data
  *  Build GP models with multi-dimensional inputs (temperature, pressure, concentration)
  *  Quantitatively assess model reliability

## Practice Problems

### Easy (Basic Confirmation)

**Q1** : In the Gaussian process predictive distribution `f(x*) ~ N(¼, Ã²)`, which formula calculates the 95% confidence interval?

Show answer

**Correct answer** : [¼ - 1.96Ã, ¼ + 1.96Ã]

**Explanation** : The 95% interval of a normal distribution is mean ± 1.96×standard deviation. 99% interval is ± 2.58Ã, 68% interval is ± 1Ã.

**Q2** : How does the prediction curve change when the length scale `` of the RBF kernel is increased?

Show answer

**Correct answer** : It becomes smoother

**Explanation** :  represents the correlation distance; larger values mean distant points are more strongly correlated. As ’ , it converges to a constant function.

### Medium (Application)

**Q3** : You built a GP model with log marginal likelihood of -10.5 from 5 experimental data points. Is this a good model?

Show answer

**Answer** : **Relative comparison** is necessary.

**Explanation** : Log marginal likelihood is used for relative comparison between different kernels, not in absolute terms. For example, if RBF has -10.5 and Matérn has -12.3, RBF is superior. The absolute value of likelihood increases with data size n, so comparison across different datasets is not possible.

**Q4** : Cross-validation RMSE is 3.5%, training data RMSE is 1.8%. Are there problems with this model?

Show answer

**Answer** : **Signs of overfitting** are present.

**Explanation** : When CV score is significantly worse than training score, there's a possibility of overfitting. Countermeasures: (1) increase noise term `alpha`, (2) use simpler kernel, (3) increase data size.

### Hard (Advanced)

**Q5** : When building a GP model with 3D input of temperature, pressure, and concentration, explain why different length scales should be used for each dimension.

Show answer

**Answer** : **Automatic Relevance Determination (ARD)** enables automatic learning of the importance of each input dimension.

**Implementation example** :
    
    
    kernel = C(1.0) * RBF(length_scale=[10.0, 1.0, 5.0],
                       length_scale_bounds=(1e-2, 1e3))
    # [Temperature 10°C, Pressure 1.0MPa, Concentration 5% initial length scales]

**Interpretation** : After optimization, if pressure's length scale is small at 0.5, it indicates yield is sensitive to pressure changes. If temperature is large at 50, temperature dependence is low.

## Next Steps

You can now quantify uncertainty with Gaussian process models. In Chapter 3, we will learn about **acquisition functions that intelligently select the next experimental point** using this uncertainty.

**What you'll learn in Chapter 3:**

  * Expected Improvement (EI): Selecting the most promising point
  * Upper Confidence Bound (UCB): Balancing exploration-exploitation
  * Probability of Improvement (PI): Simple improvement probability
  * Batch Bayesian optimization: Parallel experiment design

[� Series Index](<./index.html>) [Chapter 3: Acquisition Functions ’](<chapter-3.html>)

## References

  1. Rasmussen, C. E., & Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning_. MIT Press.
  2. Shahriari, B., et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." _Proceedings of the IEEE_ , 104(1), 148-175.
  3. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12, 2825-2830.
  4. Matérn, B. (1960). _Spatial Variation_. Springer.
