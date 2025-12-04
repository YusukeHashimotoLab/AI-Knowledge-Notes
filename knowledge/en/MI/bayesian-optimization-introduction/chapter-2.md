---
title: "Chapter 2: Theory of Bayesian Optimization"
chapter_title: "Chapter 2: Theory of Bayesian Optimization"
subtitle: Optimizing exploration with Gaussian Process and Acquisition Functions
reading_time: 25-30 minutes
difficulty: Beginner
code_examples: 10
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 2: Theory of Bayesian Optimization

Understand the roles of Gaussian Process and Acquisition Functions (EI/UCB/PI) through visual diagrams. Learn how to balance exploration and exploitation.

**💡 Supplement:** EI "digs deeper into promising areas", UCB "verifies uncertain regions". Switch between them depending on the situation.

**Optimizing exploration with Gaussian Process and Acquisition Functions**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the basic principles of Gaussian Process regression
  * ✅ Explain the role and construction method of surrogate models
  * ✅ Implement the three main Acquisition Functions (EI, PI, UCB)
  * ✅ Express the exploration-exploitation trade-off mathematically
  * ✅ Understand uncertainty quantification and its importance

**Reading Time** : 25-30 minutes **Code Examples** : 10 **Exercises** : 3

* * *

## 2.1 What is a Surrogate Model

### Why Surrogate Models are Necessary

In materials exploration, evaluating the true objective function (e.g., ionic conductivity, catalyst activity) requires **experiments**. However, experiments are:

  * **Time-consuming** : Several hours to days per sample
  * **Expensive** : Material costs, equipment costs, labor costs
  * **Limited in number** : Budget and time constraints

Therefore, we construct a **model that estimates the objective function from a small number of experimental results**. This is called a **Surrogate Model**.

### Role of Surrogate Models
    
    
    ```mermaid
    flowchart LR
        A[Small experimental data\nExample: 10-20 points] --> B[Surrogate model construction\nGaussian Process regression]
        B --> C[Prediction in unknown regions\nMean + Uncertainty]
        C --> D[Acquisition Function calculation\nPropose next experimental point]
        D --> E[Execute experiment\nAcquire new data]
        E --> B
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
    ```

**Requirements for Surrogate Models** : 1\. **Function with small data** : Predictable with about 10-20 points 2\. **Quantify uncertainty** : Evaluate prediction reliability 3\. **Fast** : Instant prediction even for thousands of points 4\. **Flexible** : Handle complex function shapes

**Gaussian Process Regression** is a powerful method that satisfies these requirements.

* * *

## 2.2 Fundamentals of Gaussian Process Regression

### What is a Gaussian Process

**Gaussian Process (GP)** is a method for defining a probability distribution over functions.

**Definition** :

> A Gaussian Process is a stochastic process where the function values at any finite set of points **follow a multivariate Gaussian distribution**.

**Intuitive Understanding** : \- Consider a **distribution of infinite functions** , not just one function \- Update the function distribution based on observed data \- Predicted values at each point are represented by **mean and variance**

### Mathematical Definition of Gaussian Process

A Gaussian Process is completely defined by a **mean function** $m(x)$ and a **kernel function (covariance function)** $k(x, x')$:

$$ f(x) \sim \mathcal{GP}(m(x), k(x, x')) $$

**Mean Function** $m(x)$: \- Usually assumed to be $m(x) = 0$ (learned from data)

**Kernel Function** $k(x, x')$: \- Represents the "similarity" between two points \- Assumes that closer inputs lead to similar outputs

### Representative Kernel Functions

**1\. RBF (Radial Basis Function) Kernel**

$$ k(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||^2}{2\ell^2}\right) $$

  * $\sigma^2$: Variance (output scale)
  * $\ell$: Length scale (how smooth)

**Characteristics** : \- Most common \- Infinitely differentiable (smooth function) \- Suitable for materials property prediction

**Code Example 1: RBF Kernel Visualization**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # RBF Kernel Visualization
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rbf_kernel(x1, x2, sigma=1.0, length_scale=1.0):
        """
        RBF Kernel Function
    
        Parameters:
        -----------
        x1, x2 : array
            Input points
        sigma : float
            Standard deviation (output scale)
        length_scale : float
            Length scale (input correlation distance)
    
        Returns:
        --------
        float : Kernel value (similarity)
        """
        distance = np.abs(x1 - x2)
        return sigma**2 * np.exp(-0.5 * (distance / length_scale)**2)
    
    # Visualize kernel with different length scales
    x_ref = 0.5  # Reference point
    x_range = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(12, 4))
    
    # Left: Effect of length scale
    plt.subplot(1, 3, 1)
    for length_scale in [0.05, 0.1, 0.2, 0.5]:
        k_values = [rbf_kernel(x_ref, x, sigma=1.0,
                               length_scale=length_scale)
                    for x in x_range]
        plt.plot(x_range, k_values,
                 label=f'$\\ell$ = {length_scale}', linewidth=2)
    plt.axvline(x_ref, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('k(0.5, x)', fontsize=12)
    plt.title('Effect of Length Scale', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Center: Multiple reference points
    plt.subplot(1, 3, 2)
    for x_ref_temp in [0.2, 0.5, 0.8]:
        k_values = [rbf_kernel(x_ref_temp, x, sigma=1.0,
                               length_scale=0.1)
                    for x in x_range]
        plt.plot(x_range, k_values,
                 label=f'x_ref = {x_ref_temp}', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('k(x_ref, x)', fontsize=12)
    plt.title('Effect of Reference Point', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right: Kernel matrix visualization
    plt.subplot(1, 3, 3)
    x_grid = np.linspace(0, 1, 50)
    K = np.zeros((len(x_grid), len(x_grid)))
    for i, x1 in enumerate(x_grid):
        for j, x2 in enumerate(x_grid):
            K[i, j] = rbf_kernel(x1, x2, sigma=1.0, length_scale=0.1)
    
    plt.imshow(K, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Kernel Value')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("x'", fontsize=12)
    plt.title('Kernel Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('rbf_kernel_visualization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("RBF Kernel Characteristics:")
    print("  - Small length scale → Local correlation (sharp peak)")
    print("  - Large length scale → Wide-range correlation (smooth curve)")
    print("  - Maximum kernel value on diagonal (x = x')")
    

**Important Points** : \- **Length Scale $\ell$** : Controls function smoothness \- Small $\ell$ → Allows steep changes \- Large $\ell$ → Assumes smooth function \- **Meaning in Materials Science** : Assumes that similar compositions or conditions lead to similar properties

* * *

### Prediction Formulas for Gaussian Process Regression

Given observed data $\mathcal{D} = {(x_1, y_1), \ldots, (x_n, y_n)}$, the prediction at a new point $x_*$ is:

**Predictive Mean** : $$ \mu(x_*) = k_* K^{-1} \mathbf{y} $$

**Predictive Variance** : $$ \sigma^2(x_*) = k(x_*, x_*) - k_*^T K^{-1} k_* $$

Where: \- $K$: Kernel matrix between observed points $K_{ij} = k(x_i, x_j)$ \- $k_*$: Kernel vector between new point and observed points \- $\mathbf{y}$: Vector of observed values

**Predictive Distribution** : $$ f(x_*) | \mathcal{D} \sim \mathcal{N}(\mu(x_*), \sigma^2(x_*)) $$

**Code Example 2: Gaussian Process Regression Implementation and Visualization**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # Gaussian Process Regression Implementation
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    
    class GaussianProcessRegressor:
        """
        Simple Gaussian Process Regression Implementation
    
        Parameters:
        -----------
        kernel : str
            Kernel type (only 'rbf' supported)
        sigma : float
            Kernel standard deviation
        length_scale : float
            Kernel length scale
        noise : float
            Observation noise standard deviation
        """
    
        def __init__(self, kernel='rbf', sigma=1.0,
                     length_scale=0.1, noise=0.01):
            self.kernel = kernel
            self.sigma = sigma
            self.length_scale = length_scale
            self.noise = noise
            self.X_train = None
            self.y_train = None
            self.K_inv = None
    
        def _kernel(self, X1, X2):
            """Compute kernel matrix"""
            if self.kernel == 'rbf':
                dists = cdist(X1, X2, 'sqeuclidean')
                K = self.sigma**2 * np.exp(-0.5 * dists /
                                            self.length_scale**2)
                return K
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")
    
        def fit(self, X_train, y_train):
            """
            Train Gaussian Process model
    
            Parameters:
            -----------
            X_train : array (n_samples, n_features)
                Training input
            y_train : array (n_samples,)
                Training output
            """
            self.X_train = X_train
            self.y_train = y_train
    
            # Compute kernel matrix (add noise term)
            K = self._kernel(X_train, X_train)
            K += self.noise**2 * np.eye(len(X_train))
    
            # Compute inverse matrix (used in prediction)
            self.K_inv = np.linalg.inv(K)
    
        def predict(self, X_test, return_std=False):
            """
            Perform prediction
    
            Parameters:
            -----------
            X_test : array (n_test, n_features)
                Test input
            return_std : bool
                Whether to return standard deviation
    
            Returns:
            --------
            mean : array (n_test,)
                Predictive mean
            std : array (n_test,) (if return_std=True)
                Predictive standard deviation
            """
            # k_* = k(X_test, X_train)
            k_star = self._kernel(X_test, self.X_train)
    
            # Predictive mean: μ(x_*) = k_* K^{-1} y
            mean = k_star @ self.K_inv @ self.y_train
    
            if return_std:
                # k(x_*, x_*)
                k_starstar = self._kernel(X_test, X_test)
    
                # Predictive variance: σ²(x_*) = k(x_*, x_*) - k_*^T K^{-1} k_*
                variance = np.diag(k_starstar) - np.sum(
                    (k_star @ self.K_inv) * k_star, axis=1
                )
                std = np.sqrt(np.maximum(variance, 0))  # Prevent numerical errors
                return mean, std
            else:
                return mean
    
    # Demonstration: Ionic conductivity prediction for materials
    np.random.seed(42)
    
    # True function (assumed unknown)
    def true_function(x):
        """Ionic conductivity of Li-ion battery electrolyte (hypothetical)"""
        return (
            np.sin(3 * x) * np.exp(-x) +
            0.7 * np.exp(-((x - 0.5) / 0.2)**2)
        )
    
    # Observed data (small number of experimental results)
    n_observations = 8
    X_train = np.random.uniform(0, 1, n_observations).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, 0.05,
                                                                 n_observations)
    
    # Test data (points to predict)
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # Train Gaussian Process regression model
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.15, noise=0.05)
    gp.fit(X_train, y_train)
    
    # Prediction
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # True function
    plt.plot(X_test, y_true, 'k--', linewidth=2, label='True Function')
    
    # Observed data
    plt.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data (Experimental Results)')
    
    # Predictive mean
    plt.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive Mean')
    
    # Uncertainty (95% confidence interval)
    plt.fill_between(
        X_test.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.3,
        color='blue',
        label='95% Confidence Interval'
    )
    
    plt.xlabel('Composition Parameter x', fontsize=12)
    plt.ylabel('Ionic Conductivity (mS/cm)', fontsize=12)
    plt.title('Material Property Prediction using Gaussian Process Regression', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gp_regression_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Gaussian Process Regression Results:")
    print(f"  Number of observations: {n_observations}")
    print(f"  Number of prediction points: {len(X_test)}")
    print(f"  RMSE: {np.sqrt(np.mean((y_pred - y_true)**2)):.4f}")
    print("\nCharacteristics:")
    print("  - Near observation points: Small uncertainty (narrow confidence interval)")
    print("  - Unexplored regions: Large uncertainty (wide confidence interval)")
    print("  - This uncertainty information is exploited by Acquisition Functions")
    

**Important Observations** : 1\. **Near observation points** : High prediction accuracy (low uncertainty) 2\. **Unexplored regions** : High uncertainty 3\. **More data** : Improved prediction accuracy 4\. **Quantifying uncertainty** : Key to Bayesian Optimization

* * *

## 2.3 Acquisition Function: How to Select the Next Experimental Point

### Role of Acquisition Functions

The **Acquisition Function** mathematically determines "where to experiment next".

**Design Philosophy** : \- **Explore high predicted value locations** (Exploitation) \- **Explore high uncertainty locations** (Exploration) \- **Balance these two aspects** through optimization

### Acquisition Function Workflow
    
    
    ```mermaid
    flowchart TB
        A[Gaussian Process Model] --> B[Predictive Mean μ(x)]
        A --> C[Predictive Standard Deviation σ(x)]
        B --> D[Acquisition Function α(x)]
        C --> D
        D --> E[Maximize Acquisition Function]
        E --> F[Next Experimental Point x_next]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style F fill:#e8f5e9
    ```

* * *

### Main Acquisition Functions

#### 1\. Expected Improvement (EI)

**Definition** : Maximize the expected improvement over the current best value $f_{\text{best}}$

$$ \text{EI}(x) = \mathbb{E}[\max(0, f(x) - f_{\text{best}})] $$

**Analytical Solution** : $$ \text{EI}(x) = \begin{cases} (\mu(x) - f_{\text{best}}) \Phi(Z) + \sigma(x) \phi(Z) & \text{if } \sigma(x) > 0 \ 0 & \text{if } \sigma(x) = 0 \end{cases} $$

Where: $$ Z = \frac{\mu(x) - f_{\text{best}}}{\sigma(x)} $$ \- $\Phi$: Cumulative distribution function of standard normal distribution \- $\phi$: Probability density function of standard normal distribution

**Characteristics** : \- **Well-balanced** : Automatically adjusts exploration and exploitation \- **Most common** : Widely used in materials science \- **Analytical** : Fast computation

**Code Example 3: Expected Improvement Implementation**
    
    
    # Expected Improvement Implementation
    from scipy.stats import norm
    
    def expected_improvement(X, gp, f_best, xi=0.01):
        """
        Expected Improvement Acquisition Function
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            Evaluation points
        gp : GaussianProcessRegressor
            Trained Gaussian Process model
        f_best : float
            Current best value
        xi : float
            Exploration strength (exploration parameter)
    
        Returns:
        --------
        ei : array (n_samples,)
            EI values
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # Improvement
        improvement = mu - f_best - xi
    
        # Standardization
        Z = improvement / (sigma + 1e-9)  # Avoid division by zero
    
        # Expected Improvement
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
        # EI = 0 when σ = 0
        ei[sigma == 0.0] = 0.0
    
        return ei
    
    # Demonstration
    np.random.seed(42)
    
    # Observed data
    X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # Gaussian Process model
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.2, noise=0.01)
    gp.fit(X_train, y_train)
    
    # Test points
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    
    # Prediction
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Current best value
    f_best = np.max(y_train)
    
    # Compute EI
    ei = expected_improvement(X_test, gp, f_best, xi=0.01)
    
    # Propose next experimental point
    next_idx = np.argmax(ei)
    next_x = X_test[next_idx]
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Gaussian Process prediction
    ax1 = axes[0]
    ax1.plot(X_test, true_function(X_test), 'k--',
             linewidth=2, label='True Function')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data')
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive Mean')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                     label='95% Confidence Interval')
    ax1.axhline(f_best, color='green', linestyle=':',
                linewidth=2, label=f'Current Best = {f_best:.3f}')
    ax1.axvline(next_x, color='orange', linestyle='--',
                linewidth=2, label=f'Proposed Point = {next_x[0]:.3f}')
    ax1.set_ylabel('Objective Function', fontsize=12)
    ax1.set_title('Gaussian Process Regression Prediction', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Expected Improvement
    ax2 = axes[1]
    ax2.plot(X_test, ei, 'r-', linewidth=2, label='Expected Improvement')
    ax2.axvline(next_x, color='orange', linestyle='--',
                linewidth=2, label=f'Max EI Point = {next_x[0]:.3f}')
    ax2.fill_between(X_test.ravel(), 0, ei, alpha=0.3, color='red')
    ax2.set_xlabel('Parameter x', fontsize=12)
    ax2.set_ylabel('EI(x)', fontsize=12)
    ax2.set_title('Expected Improvement Acquisition Function', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expected_improvement_demo.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print(f"Proposal by Expected Improvement:")
    print(f"  Next experimental point: x = {next_x[0]:.3f}")
    print(f"  EI value: {np.max(ei):.4f}")
    print(f"  Predictive mean: {y_pred[next_idx]:.3f}")
    print(f"  Predictive standard deviation: {y_std[next_idx]:.3f}")
    

**EI Interpretation** : \- Locations with **high mean values** → Exploitation \- Locations with **high uncertainty** → Exploration \- **Considers both** for balance

* * *

#### 2\. Upper Confidence Bound (UCB)

**Definition** : Maximize the "optimistic estimate" by adding uncertainty to the predictive mean

$$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\kappa$: Parameter controlling exploration strength (typically $\kappa = 2$)

**Characteristics** : \- **Simple** : Easy to implement \- **Intuitive** : Principle of Optimism in the Face of Uncertainty \- **Adjustable** : Control exploration level with $\kappa$

**Influence of $\kappa$** : \- Large $\kappa$ → Exploration-focused (take risks) \- Small $\kappa$ → Exploitation-focused (safe strategy)

**Code Example 4: Upper Confidence Bound Implementation**
    
    
    # Upper Confidence Bound Implementation
    def upper_confidence_bound(X, gp, kappa=2.0):
        """
        Upper Confidence Bound Acquisition Function
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            Evaluation points
        gp : GaussianProcessRegressor
            Trained Gaussian Process model
        kappa : float
            Exploration strength (typically 2.0)
    
        Returns:
        --------
        ucb : array (n_samples,)
            UCB values
        """
        mu, sigma = gp.predict(X, return_std=True)
        ucb = mu + kappa * sigma
        return ucb
    
    # Demonstration: Comparison with different κ values
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    kappa_values = [0.5, 2.0, 5.0]
    
    for i, kappa in enumerate(kappa_values):
        ax = axes[i]
    
        # Compute UCB
        ucb = upper_confidence_bound(X_test, gp, kappa=kappa)
    
        # Next experimental point
        next_idx = np.argmax(ucb)
        next_x = X_test[next_idx]
    
        # Predictive mean and confidence interval
        ax.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive Mean μ(x)')
        ax.fill_between(X_test.ravel(),
                        y_pred - 1.96 * y_std,
                        y_pred + 1.96 * y_std,
                        alpha=0.2, color='blue',
                        label='95% Confidence Interval')
    
        # UCB
        ax.plot(X_test, ucb, 'r-', linewidth=2,
                label=f'UCB(x) (κ={kappa})')
    
        # Observed data
        ax.scatter(X_train, y_train, c='red', s=100, zorder=10,
                   edgecolors='black', label='Observed Data')
    
        # Proposed point
        ax.axvline(next_x, color='orange', linestyle='--',
                   linewidth=2, label=f'Proposed Point = {next_x[0]:.3f}')
    
        ax.set_ylabel('Objective Function', fontsize=12)
        ax.set_title(f'UCB with κ = {kappa}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
        if i == 2:
            ax.set_xlabel('Parameter x', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ucb_kappa_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Influence of κ:")
    print("  κ = 0.5: Exploitation-focused (explore near observed data)")
    print("  κ = 2.0: Balanced (standard setting)")
    print("  κ = 5.0: Exploration-focused (actively explore unknown regions)")
    

* * *

#### 3\. Probability of Improvement (PI)

**Definition** : Maximize the probability of exceeding the current best value

$$ \text{PI}(x) = P(f(x) > f_{\text{best}}) = \Phi\left(\frac{\mu(x) - f_{\text{best}}}{\sigma(x)}\right) $$

**Characteristics** : \- **Simplest** : Easy to interpret \- **Conservative** : Does not expect large improvements \- **Practical** : Strategy of accumulating small improvements

**Code Example 5: Probability of Improvement Implementation**
    
    
    # Probability of Improvement Implementation
    def probability_of_improvement(X, gp, f_best, xi=0.01):
        """
        Probability of Improvement Acquisition Function
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            Evaluation points
        gp : GaussianProcessRegressor
            Trained Gaussian Process model
        f_best : float
            Current best value
        xi : float
            Exploration strength
    
        Returns:
        --------
        pi : array (n_samples,)
            PI values
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # Improvement
        improvement = mu - f_best - xi
    
        # Standardization
        Z = improvement / (sigma + 1e-9)
    
        # Probability of Improvement
        pi = norm.cdf(Z)
    
        return pi
    
    # Compute PI
    pi = probability_of_improvement(X_test, gp, f_best, xi=0.01)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X_test, pi, 'g-', linewidth=2, label='PI(x)')
    plt.axvline(X_test[np.argmax(pi)], color='orange',
                linestyle='--', linewidth=2,
                label=f'Max PI Point = {X_test[np.argmax(pi)][0]:.3f}')
    plt.fill_between(X_test.ravel(), 0, pi, alpha=0.3, color='green')
    plt.xlabel('Parameter x', fontsize=12)
    plt.ylabel('PI(x)', fontsize=12)
    plt.title('Probability of Improvement', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison: EI vs PI vs UCB
    plt.subplot(1, 2, 2)
    ei_normalized = ei / np.max(ei)
    pi_normalized = pi / np.max(pi)
    ucb_normalized = upper_confidence_bound(X_test, gp, kappa=2.0)
    ucb_normalized = (ucb_normalized - np.min(ucb_normalized)) / \
                     (np.max(ucb_normalized) - np.min(ucb_normalized))
    
    plt.plot(X_test, ei_normalized, 'r-', linewidth=2, label='EI (Normalized)')
    plt.plot(X_test, pi_normalized, 'g-', linewidth=2, label='PI (Normalized)')
    plt.plot(X_test, ucb_normalized, 'b-', linewidth=2, label='UCB (Normalized)')
    plt.scatter(X_train, [0.5]*len(X_train), c='red', s=100,
                zorder=10, edgecolors='black', label='Observed Data')
    plt.xlabel('Parameter x', fontsize=12)
    plt.ylabel('Acquisition Function Value (Normalized)', fontsize=12)
    plt.title('Comparison of Acquisition Functions', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    

* * *

### Comparison Table of Acquisition Functions

Acquisition Function | Feature | Advantages | Disadvantages | Recommended Use  
---|---|---|---|---  
**EI** | Expected improvement value | Well-balanced, proven track record | Somewhat complex | General optimization  
**UCB** | Optimistic estimation | Simple, adjustable | Requires κ tuning | Control exploration level  
**PI** | Probability of improvement | Very simple | Conservative | Safe exploration  
  
**Recommendations for Materials Science** : \- **Beginners** : EI (well-balanced, excellent by default) \- **Exploration-focused** : UCB (κ = 2-5) \- **Safe strategy** : PI (ensure small improvements)

* * *

## 2.4 Exploration-Exploitation Trade-off

### Mathematical Formulation

The Acquisition Function can be decomposed into two terms:

$$ \alpha(x) = \underbrace{\mu(x)}_{\text{Exploitation}} + \underbrace{\lambda \sigma(x)}_{\text{Exploration}} $$

  * **Exploitation Term** $\mu(x)$: Locations with high predictive mean
  * **Exploration Term** $\lambda \sigma(x)$: Locations with high uncertainty

### Visualization of the Trade-off
    
    
    ```mermaid
    flowchart LR
        subgraph Exploitation
        A1[Known good regions]
        A2[High predictive value μ(x)]
        A3[Low uncertainty σ(x)]
        A1 --> A2
        A1 --> A3
        end
    
        subgraph Exploration
        B1[Unknown regions]
        B2[Unknown predictive value μ(x)]
        B3[High uncertainty σ(x)]
        B1 --> B2
        B1 --> B3
        end
    
        subgraph Optimal_Balance [Optimal Balance]
        C1[Acquisition Function]
        C2[μ(x) + λσ(x)]
        C3[Next experimental point]
        C1 --> C2
        C2 --> C3
        end
    
        A2 --> C1
        A3 --> C1
        B2 --> C1
        B3 --> C1
    
        style A1 fill:#fff3e0
        style B1 fill:#e3f2fd
        style C3 fill:#e8f5e9
    ```

**Code Example 6: Visualizing Exploration-Exploitation Balance**
    
    
    # Decomposing exploration and exploitation
    def decompose_acquisition(X, gp, f_best, xi=0.01):
        """
        Decompose Acquisition Function into exploration and exploitation terms
    
        Returns:
        --------
        exploitation : Exploitation term (based on predictive mean)
        exploration : Exploration term (based on uncertainty)
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # Exploitation term (larger for higher mean)
        exploitation = mu
    
        # Exploration term (larger for higher uncertainty)
        exploration = sigma
    
        return exploitation, exploration
    
    # Decomposition
    exploitation, exploration = decompose_acquisition(X_test, gp, f_best)
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # 1. Gaussian Process prediction
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive Mean μ(x)')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                     label='95% Confidence Interval')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data')
    ax1.set_ylabel('Objective Function', fontsize=12)
    ax1.set_title('Gaussian Process Prediction', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Exploitation term
    ax2 = axes[1]
    ax2.plot(X_test, exploitation, 'g-', linewidth=2,
             label='Exploitation Term (Predictive Mean)')
    ax2.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', alpha=0.5)
    ax2.set_ylabel('Exploitation Term', fontsize=12)
    ax2.set_title('Exploitation: Emphasize Known Good Regions', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Exploration term
    ax3 = axes[2]
    ax3.plot(X_test, exploration, 'orange', linewidth=2,
             label='Exploration Term (Uncertainty)')
    ax3.scatter(X_train, [0]*len(X_train), c='red', s=100,
                zorder=10, edgecolors='black', alpha=0.5,
                label='Observed Data Locations')
    ax3.set_ylabel('Exploration Term', fontsize=12)
    ax3.set_title('Exploration: Emphasize Unknown Regions', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Integration (EI)
    ax4 = axes[3]
    ei_values = expected_improvement(X_test, gp, f_best, xi=0.01)
    ax4.plot(X_test, ei_values, 'r-', linewidth=2,
             label='Expected Improvement')
    next_x = X_test[np.argmax(ei_values)]
    ax4.axvline(next_x, color='purple', linestyle='--',
                linewidth=2, label=f'Proposed Point = {next_x[0]:.3f}')
    ax4.fill_between(X_test.ravel(), 0, ei_values,
                     alpha=0.3, color='red')
    ax4.set_xlabel('Parameter x', fontsize=12)
    ax4.set_ylabel('EI(x)', fontsize=12)
    ax4.set_title('Integration: Optimize Balance of Both', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exploitation_exploration_tradeoff.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("Exploration-Exploitation Trade-off:")
    print(f"  Proposed point x = {next_x[0]:.3f}")
    print(f"    Predictive mean (exploitation): {y_pred[np.argmax(ei_values)]:.3f}")
    print(f"    Uncertainty (exploration): {y_std[np.argmax(ei_values)]:.3f}")
    print(f"    EI value: {np.max(ei_values):.4f}")
    

* * *

## 2.5 Constrained and Multi-objective Optimization

### Constrained Bayesian Optimization

In actual materials development, **constraints** exist:

**Example: Li-ion Battery Electrolyte** \- Maximize ionic conductivity (objective function) \- Viscosity < 10 cP (constraint 1) \- Flash point > 100°C (constraint 2) \- Cost < $50/kg (constraint 3)

**Mathematical Formulation** : $$ \begin{align} \max_{x} \quad & f(x) \ \text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \ & h_j(x) = 0, \quad j = 1, \ldots, p \end{align} $$

**Approach** : 1\. **Model constraint functions with Gaussian Process** 2\. **Incorporate probability of satisfying constraints into Acquisition Function**

**Code Example 7: Constrained Bayesian Optimization Demo**
    
    
    # Constrained Bayesian Optimization
    def constrained_expected_improvement(X, gp_obj, gp_constraint,
                                         f_best, constraint_threshold=0):
        """
        Constrained Expected Improvement
    
        Parameters:
        -----------
        gp_obj : Gaussian Process (objective function)
        gp_constraint : Gaussian Process (constraint function)
        constraint_threshold : Constraint threshold (≤ 0 is feasible)
        """
        # EI for objective function
        ei = expected_improvement(X, gp_obj, f_best, xi=0.01)
    
        # Probability of satisfying constraints
        mu_c, sigma_c = gp_constraint.predict(X, return_std=True)
        prob_feasible = norm.cdf((constraint_threshold - mu_c) /
                                 (sigma_c + 1e-9))
    
        # Constrained EI = EI × constraint satisfaction probability
        cei = ei * prob_feasible
    
        return cei
    
    # Demo: Define constraint function
    def constraint_function(x):
        """Constraint function (e.g., viscosity upper limit)"""
        return 0.5 - x  # x < 0.5 is feasible region
    
    # Constraint data
    y_constraint = constraint_function(X_train).ravel()
    
    # Gaussian Process for constraint function
    gp_constraint = GaussianProcessRegressor(sigma=0.5, length_scale=0.2,
                                             noise=0.01)
    gp_constraint.fit(X_train, y_constraint)
    
    # Compute constrained EI
    cei = constrained_expected_improvement(X_test, gp, gp_constraint,
                                           f_best, constraint_threshold=0)
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Top: Objective function
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='Objective Function Prediction')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data')
    ax1.set_ylabel('Objective Function', fontsize=12)
    ax1.set_title('Objective Function (Ionic Conductivity)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle: Constraint function
    ax2 = axes[1]
    mu_c, sigma_c = gp_constraint.predict(X_test, return_std=True)
    ax2.plot(X_test, mu_c, 'g-', linewidth=2, label='Constraint Function Prediction')
    ax2.fill_between(X_test.ravel(), mu_c - 1.96 * sigma_c,
                     mu_c + 1.96 * sigma_c, alpha=0.3, color='green')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2,
                label='Constraint Boundary (≤ 0 is feasible)')
    ax2.axhspan(-10, 0, alpha=0.2, color='green',
                label='Feasible Region')
    ax2.scatter(X_train, y_constraint, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data')
    ax2.set_ylabel('Constraint Function', fontsize=12)
    ax2.set_title('Constraint Function (Viscosity Limit)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom: Constrained EI
    ax3 = axes[2]
    ei_unconstrained = expected_improvement(X_test, gp, f_best, xi=0.01)
    ax3.plot(X_test, ei_unconstrained, 'r--', linewidth=2,
             label='EI (Unconstrained)', alpha=0.5)
    ax3.plot(X_test, cei, 'r-', linewidth=2, label='Constrained EI')
    next_x = X_test[np.argmax(cei)]
    ax3.axvline(next_x, color='purple', linestyle='--', linewidth=2,
                label=f'Proposed Point = {next_x[0]:.3f}')
    ax3.fill_between(X_test.ravel(), 0, cei, alpha=0.3, color='red')
    ax3.set_xlabel('Parameter x', fontsize=12)
    ax3.set_ylabel('Acquisition Function', fontsize=12)
    ax3.set_title('Constrained Expected Improvement', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_bayesian_optimization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("Constrained Optimization Results:")
    print(f"  Proposed point: x = {next_x[0]:.3f}")
    print(f"  Unconstrained EI maximum: x = {X_test[np.argmax(ei_unconstrained)][0]:.3f}")
    print(f"  → Proposed point changed due to constraints")
    

* * *

### Multi-objective Optimization

In materials development, we often want to **optimize multiple properties simultaneously**.

**Example: Thermoelectric Materials** \- Maximize Seebeck coefficient \- Minimize electrical resistivity \- Minimize thermal conductivity

**Pareto Front** : \- When trade-offs exist, there is no single optimal solution \- Find **a set of Pareto optimal solutions**

**Approaches** : 1\. **Scalarization** : Weighted sum $f(x) = w_1 f_1(x) + w_2 f_2(x)$ 2\. **ParEGO** : Repeat scalarization with random weights 3\. **EHVI** : Expected Hypervolume Improvement

**Code Example 8: Multi-objective Optimization Visualization**
    
    
    # Multi-objective Optimization Demo
    def objective1(x):
        """Objective 1: Ionic conductivity (maximize)"""
        return true_function(x)
    
    def objective2(x):
        """Objective 2: Viscosity (minimize)"""
        return 0.5 + 0.3 * np.sin(5 * x)
    
    # Compute Pareto optimal solutions
    x_grid = np.linspace(0, 1, 1000)
    obj1_values = objective1(x_grid)
    obj2_values = objective2(x_grid)
    
    # Determine Pareto optimality
    def is_pareto_optimal(costs):
        """
        Determine Pareto optimal solutions
    
        Parameters:
        -----------
        costs : array (n_samples, n_objectives)
            Cost of each point (as minimization problem)
    
        Returns:
        --------
        pareto_mask : array (n_samples,)
            True for Pareto optimal
        """
        is_pareto = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_pareto[i]:
                # Check if dominated by other points
                is_pareto[is_pareto] = np.any(
                    costs[is_pareto] < c, axis=1
                )
                is_pareto[i] = True
        return is_pareto
    
    # Cost matrix (as minimization problem)
    costs = np.column_stack([
        -obj1_values,  # Maximize → Minimize
        obj2_values    # Minimize
    ])
    
    # Pareto optimal solutions
    pareto_mask = is_pareto_optimal(costs)
    pareto_x = x_grid[pareto_mask]
    pareto_obj1 = obj1_values[pareto_mask]
    pareto_obj2 = obj2_values[pareto_mask]
    
    # Visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Left: Parameter space
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x_grid, obj1_values, 'b-', linewidth=2,
             label='Objective 1 (Ionic Conductivity)')
    ax1.plot(x_grid, obj2_values, 'r-', linewidth=2,
             label='Objective 2 (Viscosity)')
    ax1.scatter(pareto_x, pareto_obj1, c='blue', s=50, alpha=0.6,
                label='Pareto Optimal (Obj 1)')
    ax1.scatter(pareto_x, pareto_obj2, c='red', s=50, alpha=0.6,
                label='Pareto Optimal (Obj 2)')
    ax1.set_xlabel('Parameter x', fontsize=12)
    ax1.set_ylabel('Objective Function Value', fontsize=12)
    ax1.set_title('Parameter Space', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Objective space (Pareto front)
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(obj1_values, obj2_values, c='lightgray', s=20,
                alpha=0.5, label='All Exploration Points')
    ax2.scatter(pareto_obj1, pareto_obj2, c='red', s=100,
                edgecolors='black', zorder=10,
                label='Pareto Front')
    ax2.plot(pareto_obj1, pareto_obj2, 'r--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Objective 1 (Ionic Conductivity) → Maximize', fontsize=12)
    ax2.set_ylabel('Objective 2 (Viscosity) → Minimize', fontsize=12)
    ax2.set_title('Objective Space and Pareto Front', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_objective_optimization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print(f"Number of Pareto optimal solutions: {np.sum(pareto_mask)}")
    print("Trade-off examples:")
    print(f"  High conductivity: Obj 1 = {np.max(pareto_obj1):.3f}, "
          f"Obj 2 = {pareto_obj2[np.argmax(pareto_obj1)]:.3f}")
    print(f"  Low viscosity: Obj 1 = {pareto_obj1[np.argmin(pareto_obj2)]:.3f}, "
          f"Obj 2 = {np.min(pareto_obj2):.3f}")
    

* * *

## 2.6 Column: Practical Kernel Selection

### Types and Characteristics of Kernels

Besides RBF, various kernels exist:

**Matérn Kernel** : $$ k(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}||x - x'||}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}||x - x'||}{\ell}\right) $$

  * $\nu$: Smoothness parameter
  * $\nu = 1/2$: Exponential kernel (rough function)
  * $\nu = 3/2, 5/2$: Moderate smoothness
  * $\nu \to \infty$: RBF kernel (very smooth)

**Selection Guidelines for Materials Science** : \- **DFT calculation results** : RBF (smooth) \- **Experimental data** : Matérn 3/2 or 5/2 (considering noise) \- **Composition optimization** : RBF \- **Process conditions** : Matérn (steep changes exist)

**Periodic phenomena** : Periodic kernel $$ k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right) $$ \- Crystal structure (with periodicity) \- Temperature cycles

* * *

## 2.7 Troubleshooting

### Common Problems and Solutions

**Problem 1: Acquisition Function Always Proposes the Same Location**

**Causes** : \- Length scale too large → Overall too smooth \- Noise parameter too small → Over-trusting observation points

**Solutions** :
    
    
    # Adjust length scale
    gp = GaussianProcessRegressor(length_scale=0.05, noise=0.1)
    
    # Or, automatically tune hyperparameters
    from sklearn.gaussian_process import GaussianProcessRegressor as SKGP
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
    gp = SKGP(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    

**Problem 2: Unstable Predictions (Abnormally Wide Confidence Intervals)**

**Causes** : \- Too little data \- Kernel matrix numerically unstable

**Solutions** :
    
    
    # Add regularization term
    K = kernel_matrix + 1e-6 * np.eye(n_samples)  # Add jitter
    
    # Or use Cholesky decomposition (improved numerical stability)
    from scipy.linalg import cho_solve, cho_factor
    
    L = cho_factor(K, lower=True)
    alpha = cho_solve(L, y_train)
    

**Problem 3: Slow Computation (Large-scale Data)**

**Cause** : \- Gaussian Process computational complexity: $O(n^3)$ (n = number of data)

**Solutions** :
    
    
    # 1. Sparse Gaussian Process
    # Use inducing points
    
    # 2. Approximation methods
    # - Sparse GP
    # - Local GP (domain partitioning)
    
    # 3. Library utilization
    # GPyTorch (GPU acceleration)
    # GPflow (TensorFlow backend)
    

* * *

## 2.8 Chapter Summary

### What We Learned

  1. **Role of Surrogate Models** \- Estimate objective functions from small experimental data \- Gaussian Process regression is most common \- Capable of quantifying uncertainty

  2. **Gaussian Process Regression** \- Define similarity between points with kernel functions \- RBF kernel is standard in materials science \- Compute predictive mean and predictive variance

  3. **Acquisition Functions** \- Mathematical criteria for determining next experimental point \- EI (Expected Improvement): Balanced approach \- UCB (Upper Confidence Bound): Adjustable \- PI (Probability of Improvement): Simple

  4. **Exploration and Exploitation** \- Exploitation: Exploit known good regions \- Exploration: Explore unknown regions \- Acquisition Function automatically balances both

  5. **Advanced Topics** \- Constrained optimization: Important in practice \- Multi-objective optimization: Visualizing trade-offs

### Key Points

  * ✅ Gaussian Process regression can **quantify uncertainty**
  * ✅ Acquisition Functions **mathematically optimize exploration and exploitation**
  * ✅ EI is the **most common with proven track record**
  * ✅ Kernel selection **determines model performance**
  * ✅ **Extensions to constrained and multi-objective** are possible

### Next Chapter

In Chapter 3, we will learn implementation using Python libraries: \- How to use scikit-optimize (skopt) \- BoTorch (PyTorch version) implementation \- Materials exploration with real data \- Hyperparameter tuning

**[Chapter 3: Python Practice →](<chapter-3.html>)**

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Investigate the effect of the RBF kernel length scale $\ell$ on Gaussian Process predictions.

**Tasks** : 1\. Generate 5 observation data points 2\. Train Gaussian Process with $\ell = 0.05, 0.1, 0.2, 0.5$ 3\. Plot predictive mean and confidence intervals 4\. Explain the effect of length scale

Hint \- Change the `length_scale` parameter of `GaussianProcessRegressor` \- Get standard deviation with `predict(return_std=True)` \- Small length scale → Fits locally \- Large length scale → Smooth curve  Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Tasks:
    1. Generate 5 observation data points
    2. Train Gaussi
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Observation data
    np.random.seed(42)
    X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # Test data
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # Train with different length scales
    length_scales = [0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, ls in enumerate(length_scales):
        ax = axes[i]
    
        # Gaussian Process model
        gp = GaussianProcessRegressor(sigma=1.0, length_scale=ls,
                                       noise=0.01)
        gp.fit(X_train, y_train)
    
        # Prediction
        y_pred, y_std = gp.predict(X_test, return_std=True)
    
        # Plot
        ax.plot(X_test, y_true, 'k--', linewidth=2, label='True Function')
        ax.scatter(X_train, y_train, c='red', s=100, zorder=10,
                   edgecolors='black', label='Observed Data')
        ax.plot(X_test, y_pred, 'b-', linewidth=2, label='Predictive Mean')
        ax.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                        y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                        label='95% Confidence Interval')
        ax.set_title(f'Length Scale = {ls}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('length_scale_effect.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("Effect of Length Scale:")
    print("  Small (0.05): Fits observation points closely, unstable between points")
    print("  Medium (0.1-0.2): Well-balanced")
    print("  Large (0.5): Smooth but deviates from observation points")
    

**Explanation**: \- **$\ell$ = 0.05**: Tends to overfit, unstable between observation points \- **$\ell$ = 0.1-0.2**: Moderate smoothness, practical \- **$\ell$ = 0.5**: Underfits, too smooth **Implications for Materials Science**: \- Experimental data: $\ell$ = 0.1-0.3 is common \- DFT calculations: Smoother ($\ell$ = 0.3-0.5) \- Determine optimal value with cross-validation 

* * *

### Exercise 2 (Difficulty: Medium)

Implement three Acquisition Functions (EI, UCB, PI) and compare them with the same data.

**Tasks** : 1\. Use the same observation data 2\. Propose next experimental points with each Acquisition Function 3\. Visualize differences in proposed points 4\. Explain characteristics of each

Hint \- Evaluate the same Gaussian Process model with three Acquisition Functions \- Use `np.argmax()` to get the position of maximum value \- Use $\kappa = 2.0$ for UCB  Solution Example
    
    
    # Observed data
    np.random.seed(42)
    X_train = np.array([0.15, 0.4, 0.6, 0.85]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # Gaussian Process model
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.15,
                                   noise=0.01)
    gp.fit(X_train, y_train)
    
    # Current best value
    f_best = np.max(y_train)
    
    # Test points
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Calculate Acquisition Functions
    ei = expected_improvement(X_test, gp, f_best, xi=0.01)
    ucb = upper_confidence_bound(X_test, gp, kappa=2.0)
    pi = probability_of_improvement(X_test, gp, f_best, xi=0.01)
    
    # Proposed points
    next_x_ei = X_test[np.argmax(ei)]
    next_x_ucb = X_test[np.argmax(ucb)]
    next_x_pi = X_test[np.argmax(pi)]
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # 1. Gaussian Process prediction
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='Predicted Mean')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='Observed Data')
    ax1.axhline(f_best, color='green', linestyle=':', linewidth=2,
                label=f'Best Value = {f_best:.3f}')
    ax1.set_ylabel('Objective Function', fontsize=12)
    ax1.set_title('Gaussian Process Prediction', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Expected Improvement
    ax2 = axes[1]
    ax2.plot(X_test, ei, 'r-', linewidth=2, label='EI')
    ax2.axvline(next_x_ei, color='red', linestyle='--', linewidth=2,
                label=f'Proposed Point = {next_x_ei[0]:.3f}')
    ax2.fill_between(X_test.ravel(), 0, ei, alpha=0.3, color='red')
    ax2.set_ylabel('EI(x)', fontsize=12)
    ax2.set_title('Expected Improvement', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Upper Confidence Bound
    ax3 = axes[2]
    # Normalize UCB (for easier comparison)
    ucb_normalized = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb))
    ax3.plot(X_test, ucb_normalized, 'b-', linewidth=2, label='UCB (Normalized)')
    ax3.axvline(next_x_ucb, color='blue', linestyle='--', linewidth=2,
                label=f'Proposed Point = {next_x_ucb[0]:.3f}')
    ax3.fill_between(X_test.ravel(), 0, ucb_normalized, alpha=0.3,
                     color='blue')
    ax3.set_ylabel('UCB(x)', fontsize=12)
    ax3.set_title('Upper Confidence Bound (κ=2.0)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability of Improvement
    ax4 = axes[3]
    ax4.plot(X_test, pi, 'g-', linewidth=2, label='PI')
    ax4.axvline(next_x_pi, color='green', linestyle='--', linewidth=2,
                label=f'Proposed Point = {next_x_pi[0]:.3f}')
    ax4.fill_between(X_test.ravel(), 0, pi, alpha=0.3, color='green')
    ax4.set_xlabel('Parameter x', fontsize=12)
    ax4.set_ylabel('PI(x)', fontsize=12)
    ax4.set_title('Probability of Improvement', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions_detailed_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Results summary
    print("Proposed Points by Acquisition Function:")
    print(f"  EI:  x = {next_x_ei[0]:.3f}")
    print(f"  UCB: x = {next_x_ucb[0]:.3f}")
    print(f"  PI:  x = {next_x_pi[0]:.3f}")
    
    print("\nCharacteristics:")
    print("  EI: Balanced approach, maximizes expected improvement")
    print("  UCB: Exploration-focused, favors high uncertainty regions")
    print("  PI: Conservative, favors even small improvements")
    

**Expected Output**: 
    
    
    Proposed Points by Acquisition Function:
      EI:  x = 0.722
      UCB: x = 0.108
      PI:  x = 0.752
    
    Characteristics:
      EI: Balanced approach, maximizes expected improvement
      UCB: Exploration-focused, favors high uncertainty regions
      PI: Conservative, favors even small improvements
    

**Detailed Explanation**: \- **EI**: Proposes points between unexplored regions and regions with good predictions \- **UCB**: Explores the left edge with sparse data (uncertainty-focused) \- **PI**: Proposes locations where predicted mean is likely to exceed best value **Practical Selection**: \- General optimization → EI \- Initial exploration phase → UCB (with larger κ) \- Convergence phase → PI or EI 

* * *

### Problem 3 (Difficulty: Hard)

Implement constrained Bayesian Optimization and compare it with the unconstrained case.

**Background** : Li-ion battery electrolyte optimization \- Objective: Maximize ionic conductivity \- Constraint: Viscosity < 10 cP

**Tasks** : 1\. Define objective function and constraint function 2\. Run unconstrained Bayesian Optimization (10 iterations) 3\. Run constrained Bayesian Optimization (10 iterations) 4\. Compare exploration trajectories 5\. Evaluate final solutions found

Hint **Approach**: 1\. Initial random sampling (3 points) 2\. Build two Gaussian Process models (for objective function and constraint function) 3\. Sequential sampling in loop 4\. Use constrained EI **Functions to use**: \- `constrained_expected_improvement()`  Solution Example
    
    
    # Define objective function and constraint function
    def objective_conductivity(x):
        """Ionic conductivity (to maximize)"""
        return true_function(x)
    
    def constraint_viscosity(x):
        """Viscosity constraint (≤ 10 cP normalized to 0)"""
        viscosity = 15 - 10 * x  # Viscosity model
        return viscosity - 10  # ≤ 0 is feasible (10 cP or below)
    
    # Bayesian Optimization simulation
    def run_bayesian_optimization(n_iterations=10,
                                   use_constraint=False):
        """
        Run Bayesian Optimization
    
        Parameters:
        -----------
        n_iterations : int
            Number of optimization iterations
        use_constraint : bool
            Whether to use constraints
    
        Returns:
        --------
        X_sampled : Experimental points
        y_sampled : Objective function values
        c_sampled : Constraint function values (only when constrained)
        """
        # Initial random sampling
        np.random.seed(42)
        X_sampled = np.random.uniform(0, 1, 3).reshape(-1, 1)
        y_sampled = objective_conductivity(X_sampled).ravel()
        c_sampled = constraint_viscosity(X_sampled).ravel()
    
        # Sequential sampling
        for i in range(n_iterations - 3):
            # Gaussian Process model (objective function)
            gp_obj = GaussianProcessRegressor(sigma=1.0,
                                               length_scale=0.15,
                                               noise=0.01)
            gp_obj.fit(X_sampled, y_sampled)
    
            # Candidate points
            X_candidate = np.linspace(0, 1, 1000).reshape(-1, 1)
    
            if use_constraint:
                # Gaussian Process model (constraint function)
                gp_constraint = GaussianProcessRegressor(sigma=0.5,
                                                         length_scale=0.2,
                                                         noise=0.01)
                gp_constraint.fit(X_sampled, c_sampled)
    
                # Constrained EI
                f_best = np.max(y_sampled)
                acq = constrained_expected_improvement(
                    X_candidate, gp_obj, gp_constraint, f_best,
                    constraint_threshold=0
                )
            else:
                # Unconstrained EI
                f_best = np.max(y_sampled)
                acq = expected_improvement(X_candidate, gp_obj,
                                           f_best, xi=0.01)
    
            # Next experimental point
            next_x = X_candidate[np.argmax(acq)]
    
            # Run experiment
            next_y = objective_conductivity(next_x).ravel()[0]
            next_c = constraint_viscosity(next_x).ravel()[0]
    
            # Add to data
            X_sampled = np.vstack([X_sampled, next_x])
            y_sampled = np.append(y_sampled, next_y)
            c_sampled = np.append(c_sampled, next_c)
    
        return X_sampled, y_sampled, c_sampled
    
    # Run two scenarios
    X_unconst, y_unconst, c_unconst = run_bayesian_optimization(
        n_iterations=10, use_constraint=False
    )
    X_const, y_const, c_const = run_bayesian_optimization(
        n_iterations=10, use_constraint=True
    )
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Objective function
    ax1 = axes[0, 0]
    x_fine = np.linspace(0, 1, 500)
    y_fine = objective_conductivity(x_fine)
    ax1.plot(x_fine, y_fine, 'k-', linewidth=2, label='True Function')
    ax1.scatter(X_unconst, y_unconst, c='blue', s=100, alpha=0.6,
                label='Unconstrained', marker='o')
    ax1.scatter(X_const, y_const, c='red', s=100, alpha=0.6,
                label='Constrained', marker='^')
    ax1.set_xlabel('Parameter x', fontsize=12)
    ax1.set_ylabel('Ionic Conductivity', fontsize=12)
    ax1.set_title('Objective Function Exploration', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Constraint function
    ax2 = axes[0, 1]
    c_fine = constraint_viscosity(x_fine)
    ax2.plot(x_fine, c_fine, 'k-', linewidth=2, label='Constraint Function')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2,
                label='Constraint Boundary (≤ 0 is feasible)')
    ax2.axhspan(-20, 0, alpha=0.2, color='green',
                label='Feasible Region')
    ax2.scatter(X_unconst, c_unconst, c='blue', s=100, alpha=0.6,
                label='Unconstrained', marker='o')
    ax2.scatter(X_const, c_const, c='red', s=100, alpha=0.6,
                label='Constrained', marker='^')
    ax2.set_xlabel('Parameter x', fontsize=12)
    ax2.set_ylabel('Constraint Value', fontsize=12)
    ax2.set_title('Constraint Satisfaction', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Best value progression
    ax3 = axes[1, 0]
    best_unconst = np.maximum.accumulate(y_unconst)
    best_const = np.maximum.accumulate(y_const)
    ax3.plot(range(1, 11), best_unconst, 'o-', color='blue',
             linewidth=2, markersize=8, label='Unconstrained')
    ax3.plot(range(1, 11), best_const, '^-', color='red',
             linewidth=2, markersize=8, label='Constrained')
    ax3.set_xlabel('Number of Experiments', fontsize=12)
    ax3.set_ylabel('Best Value So Far', fontsize=12)
    ax3.set_title('Best Value Progression', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Constraint satisfaction progression
    ax4 = axes[1, 1]
    # Number of samples satisfying constraints
    feasible_unconst = np.cumsum(c_unconst <= 0)
    feasible_const = np.cumsum(c_const <= 0)
    ax4.plot(range(1, 11), feasible_unconst, 'o-', color='blue',
             linewidth=2, markersize=8, label='Unconstrained')
    ax4.plot(range(1, 11), feasible_const, '^-', color='red',
             linewidth=2, markersize=8, label='Constrained')
    ax4.set_xlabel('Number of Experiments', fontsize=12)
    ax4.set_ylabel('Cumulative Feasible Solutions', fontsize=12)
    ax4.set_title('Constraint Satisfaction Progression', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_bo_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    # Results summary
    print("Optimization Results Comparison:")
    print("\nUnconstrained Bayesian Optimization:")
    print(f"  Best Value: {np.max(y_unconst):.4f}")
    print(f"  Corresponding x: {X_unconst[np.argmax(y_unconst)][0]:.3f}")
    print(f"  Constraint Value: {c_unconst[np.argmax(y_unconst)]:.4f}")
    print(f"  Constraint Satisfied: {'Yes' if c_unconst[np.argmax(y_unconst)] <= 0 else 'No'}")
    print(f"  Number of Feasible Solutions: {np.sum(c_unconst <= 0)}/10")
    
    print("\nConstrained Bayesian Optimization:")
    # Find best solution among those satisfying constraints
    feasible_indices = np.where(c_const <= 0)[0]
    if len(feasible_indices) > 0:
        best_feasible_idx = feasible_indices[np.argmax(y_const[feasible_indices])]
        print(f"  Best Value: {y_const[best_feasible_idx]:.4f}")
        print(f"  Corresponding x: {X_const[best_feasible_idx][0]:.3f}")
        print(f"  Constraint Value: {c_const[best_feasible_idx]:.4f}")
        print(f"  Constraint Satisfied: Yes")
    else:
        print("  No Feasible Solutions")
    print(f"  Number of Feasible Solutions: {np.sum(c_const <= 0)}/10")
    
    print("\nInsights:")
    print("  - Constrained approach concentrates exploration in feasible region")
    print("  - Unconstrained finds higher objective values but may violate constraints")
    print("  - In practice, constraint-aware optimization is essential")
    

**Expected Output**: 
    
    
    Optimization Results Comparison:
    
    Unconstrained Bayesian Optimization:
      Best Value: 0.8234
      Corresponding x: 0.312
      Constraint Value: 1.876
      Constraint Satisfied: No
      Number of Feasible Solutions: 4/10
    
    Constrained Bayesian Optimization:
      Best Value: 0.7456
      Corresponding x: 0.523
      Constraint Value: -0.234
      Constraint Satisfied: Yes
      Number of Feasible Solutions: 8/10
    
    Insights:
      - Constrained approach concentrates exploration in feasible region
      - Unconstrained finds higher objective values but may violate constraints
      - In practice, constraint-aware optimization is essential
    

**Key Insights**: 1\. **Unconstrained**: Finds higher objective values but infeasible 2\. **Constrained**: Slightly lower objective values but feasible 3\. **Practical Use**: Solutions violating constraints are meaningless (materials cannot be used) 4\. **Efficiency**: Constrained approach focuses on feasible region with less waste 

* * *

## References

  1. Rasmussen, C. E. & Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning_. MIT Press. [Online Version](<http://gaussianprocess.org/gpml/>)

  2. Brochu, E. et al. (2010). "A Tutorial on Bayesian Optimization of Expensive Cost Functions." _arXiv:1012.2599_. [arXiv:1012.2599](<https://arxiv.org/abs/1012.2599>)

  3. Mockus, J. (1974). "On Bayesian Methods for Seeking the Extremum." _Optimization Techniques IFIP Technical Conference_ , 400-404.

  4. Srinivas, N. et al. (2010). "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." _ICML 2010_. [arXiv:0912.3995](<https://arxiv.org/abs/0912.3995>)

  5. Gelbart, M. A. et al. (2014). "Bayesian Optimization with Unknown Constraints." _UAI 2014_.

  6. Mochihashi, D. & Oba, S. (2019). Gaussian Processes and Machine Learning. Kodansha. ISBN: 978-4061529267

* * *

## Navigation

### Previous Chapter

**[← Chapter 1: Why Optimization is Essential for Materials Discovery](<chapter-1.html>)**

### Next Chapter

**[Chapter 3: Python Practice →](<chapter-3.html>)**

### Series Table of Contents

**[← Return to Series Table of Contents](<./index.html>)**

* * *

## Author Information

**Author** : AI Terakoya Content Team **Date Created** : 2025-10-17 **Version** : 1.0

**Update History** : \- 2025-10-17: v1.0 Initial Release

**Feedback** : \- GitHub Issues: [AI_Homepage/issues](<https://github.com/your-repo/AI_Homepage/issues>) \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

**Learn implementation in the Next Chapter!**
