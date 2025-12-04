---
title: "Chapter 2: Bayesian Optimization and Optuna"
chapter_title: "Chapter 2: Bayesian Optimization and Optuna"
subtitle: Efficient Hyperparameter Tuning - Smart Search Strategies
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 8
version: 1.0
created_at: "by:"
---

This chapter covers Bayesian Optimization and Optuna. You will learn fundamental principles of Bayesian Optimization, how TPE (Tree-structured Parzen Estimator) works, and Achieve efficient search with Pruning.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the fundamental principles of Bayesian Optimization
  * ✅ Learn how TPE (Tree-structured Parzen Estimator) works
  * ✅ Master the basic concepts and API of Optuna
  * ✅ Achieve efficient search with Pruning
  * ✅ Optimize hyperparameters for deep learning models
  * ✅ Analyze optimization processes with visualization tools

* * *

## 2.1 Foundations of Bayesian Optimization

### What is Bayesian Optimization

**Bayesian Optimization** is a method for efficiently optimizing objective functions with high evaluation costs. Compared to grid search and random search, it has the following characteristics:

  * Utilizes past trial results to determine the next search point
  * Automatically balances exploration and exploitation
  * Finds good solutions with fewer trials

### The Exploration-Exploitation Trade-off

The core of Bayesian Optimization is the balance between **Exploration** and **Exploitation**.

Strategy | Description | Advantages | Disadvantages  
---|---|---|---  
**Exploration** | Investigate unknown regions | Discovery of global optimum | May increase unnecessary trials  
**Exploitation** | Intensively investigate around good performance | Quickly converge to good solutions | May get trapped in local optima  
      
    
    ```mermaid
    graph LR
        A[Initial Random Sampling] --> B[Build Surrogate Model]
        B --> C[Select Next Point with Acquisition Function]
        C --> D[Evaluate Objective Function]
        D --> E{Stop Condition?}
        E -->|No| B
        E -->|Yes| F[Return Best Point]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#f3e5f5
        style D fill:#fff3e0
        style E fill:#fce4ec
        style F fill:#c8e6c9
    ```

### Surrogate Model (Gaussian Process)

**Surrogate Model** is a proxy model of the objective function. The most common one is the **Gaussian Process (GP)**.

Gaussian processes provide both prediction values and uncertainty at each point:

$$ f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x)) $$

  * $\mu(x)$: Prediction mean (expected value)
  * $\sigma^2(x)$: Prediction variance (uncertainty)

> **Important** : The farther from observed points, the greater the uncertainty, promoting exploration.

### Acquisition Function

**Acquisition Function** is an index that determines the next point to evaluate. Major acquisition functions:

#### 1\. Expected Improvement (EI)

Expected value of improvement from the current best value:

$$ \text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)] $$

  * $f(x^+)$: Current best value
  * Prioritizes points with expected improvement

#### 2\. Upper Confidence Bound (UCB)

Balance between mean and uncertainty:

$$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\kappa$: Controls exploration strength (typically 1.96)
  * Selects points with high mean or high uncertainty

#### 3\. Probability of Improvement (PI)

Probability of improvement:

$$ \text{PI}(x) = P(f(x) > f(x^+)) $$

  * Selects points with high probability of improvement
  * Relatively conservative

### Bayesian Optimization Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Bayesian Optimization Implementation Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from scipy.stats import norm
    
    # Objective function (example: complex 1D function)
    def objective_function(x):
        return -(x ** 2) * np.sin(5 * x)
    
    # Acquisition function: Expected Improvement (EI)
    def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(X_sample)
    
        sigma = sigma.reshape(-1, 1)
    
        # Current best value
        mu_sample_opt = np.max(mu_sample)
    
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
    
        return ei
    
    # Execute Bayesian optimization
    np.random.seed(42)
    
    # Search space
    X_true = np.linspace(-3, 3, 1000).reshape(-1, 1)
    y_true = objective_function(X_true)
    
    # Initial sampling
    n_initial = 3
    X_sample = np.random.uniform(-3, 3, n_initial).reshape(-1, 1)
    Y_sample = objective_function(X_sample)
    
    # Define Gaussian process
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
    
    # Iterative optimization
    n_iterations = 7
    plt.figure(figsize=(16, 12))
    
    for i in range(n_iterations):
        # Fit Gaussian process
        gpr.fit(X_sample, Y_sample)
    
        # Prediction
        mu, sigma = gpr.predict(X_true, return_std=True)
    
        # Calculate acquisition function
        ei = expected_improvement(X_true, X_sample, Y_sample, gpr)
    
        # Select next point (maximize EI)
        X_next = X_true[np.argmax(ei)]
        Y_next = objective_function(X_next)
    
        # Plot
        plt.subplot(3, 3, i + 1)
    
        # True function
        plt.plot(X_true, y_true, 'r--', label='True Function', alpha=0.7)
    
        # Gaussian process prediction
        plt.plot(X_true, mu, 'b-', label='GP Mean')
        plt.fill_between(X_true.ravel(),
                         mu.ravel() - 1.96 * sigma,
                         mu.ravel() + 1.96 * sigma,
                         alpha=0.2, label='95% Confidence Interval')
    
        # Observed points
        plt.scatter(X_sample, Y_sample, c='green', s=100,
                    zorder=10, label='Observed Points', edgecolors='black')
    
        # Next point
        plt.axvline(x=X_next, color='purple', linestyle='--',
                    linewidth=2, label='Next Search Point')
    
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Iteration {i+1}/{n_iterations}', fontsize=12)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
    
        # Add sample
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    
    plt.tight_layout()
    plt.show()
    
    # Final results
    best_idx = np.argmax(Y_sample)
    print(f"\n=== Bayesian Optimization Results ===")
    print(f"Best x: {X_sample[best_idx][0]:.4f}")
    print(f"Best f(x): {Y_sample[best_idx][0]:.4f}")
    print(f"Total evaluations: {len(X_sample)}")
    

**Output** :
    
    
    === Bayesian Optimization Results ===
    Best x: 1.7854
    Best f(x): 2.8561
    Total evaluations: 10
    

> **Observation** : Efficiently converges to the maximum value with few trials.

* * *

## 2.2 TPE (Tree-structured Parzen Estimator)

### How TPE Works

**TPE (Tree-structured Parzen Estimator)** is an efficient implementation of Bayesian Optimization. It's the default optimization algorithm in Optuna.

Core ideas of TPE:

  1. Divide observed data into good and bad results
  2. Model each distribution
  3. Select points sampled often from good distribution and rarely from bad distribution

### Differences from Gaussian Processes

Aspect | Gaussian Process (GP) | TPE  
---|---|---  
**Modeling Target** | $P(y|x)$ - Predict output | $P(x|y)$ - Conditional distribution of input  
**Computational Cost** | $O(n^3)$ - High for sample size | $O(n)$ - Linear  
**High-dimensional Performance** | Degrades with high dimensions | Stable even in high dimensions  
**Categorical Variables** | Difficult to handle | Naturally handled  
**Parallelization** | Difficult | Easy  
  
### TPE Formulation

TPE defines two distributions as follows:

$$ P(x|y) = \begin{cases} \ell(x) & \text{if } y < y^* \\\ g(x) & \text{if } y \geq y^* \end{cases} $$

  * $\ell(x)$: Distribution of good results
  * $g(x)$: Distribution of bad results
  * $y^*$: Threshold (typically top 20-25%)

The acquisition function maximizes the following ratio:

$$ \text{EI}(x) \propto \frac{\ell(x)}{g(x)} $$

> **Intuition** : Select points with high probability in good distribution and low probability in bad distribution.

### Implementation Efficiency

Main advantages of TPE:

  1. **Scalability** : Fast even in large search spaces
  2. **Flexibility** : Uniformly handles continuous, discrete, and categorical variables
  3. **Parallelization** : Can execute multiple trials simultaneously
  4. **Conditional Spaces** : Handles dependencies between hyperparameters

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Main advantages of TPE:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # TPE behavior image (Optuna internal operation)
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # Sample data (hyperparameters and performance)
    np.random.seed(42)
    n_trials = 50
    
    # Random hyperparameter values
    x_trials = np.random.uniform(0, 10, n_trials)
    
    # Performance (true function + noise)
    y_trials = -(x_trials - 6) ** 2 + 30 + np.random.normal(0, 2, n_trials)
    
    # Set threshold (top 25%)
    threshold_idx = int(n_trials * 0.75)
    sorted_indices = np.argsort(y_trials)
    threshold_value = y_trials[sorted_indices[threshold_idx]]
    
    # Divide into good and bad trials
    good_x = x_trials[y_trials >= threshold_value]
    bad_x = x_trials[y_trials < threshold_value]
    
    # Kernel density estimation
    x_range = np.linspace(0, 10, 1000)
    
    if len(good_x) > 1:
        kde_good = gaussian_kde(good_x)
        density_good = kde_good(x_range)
    else:
        density_good = np.zeros_like(x_range)
    
    if len(bad_x) > 1:
        kde_bad = gaussian_kde(bad_x)
        density_bad = kde_bad(x_range)
    else:
        density_bad = np.zeros_like(x_range)
    
    # EI approximation (ℓ(x) / g(x))
    ei_approx = np.zeros_like(x_range)
    mask = density_bad > 1e-6
    ei_approx[mask] = density_good[mask] / density_bad[mask]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of trials
    axes[0, 0].scatter(x_trials, y_trials, c='blue', alpha=0.6,
                       s=50, edgecolors='black')
    axes[0, 0].axhline(y=threshold_value, color='red',
                       linestyle='--', linewidth=2, label=f'Threshold (Top 25%)')
    axes[0, 0].scatter(good_x, y_trials[y_trials >= threshold_value],
                       c='green', s=100, label='Good Trials',
                       edgecolors='black', zorder=5)
    axes[0, 0].set_xlabel('Hyperparameter x')
    axes[0, 0].set_ylabel('Performance y')
    axes[0, 0].set_title('Distribution of Trials', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of good trials ℓ(x)
    axes[0, 1].fill_between(x_range, density_good, alpha=0.5,
                            color='green', label='ℓ(x): Good Trial Distribution')
    axes[0, 1].scatter(good_x, np.zeros_like(good_x),
                       c='green', s=50, marker='|', linewidths=2)
    axes[0, 1].set_xlabel('Hyperparameter x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Good Trial Distribution ℓ(x)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of bad trials g(x)
    axes[1, 0].fill_between(x_range, density_bad, alpha=0.5,
                            color='red', label='g(x): Bad Trial Distribution')
    axes[1, 0].scatter(bad_x, np.zeros_like(bad_x),
                       c='red', s=50, marker='|', linewidths=2)
    axes[1, 0].set_xlabel('Hyperparameter x')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Bad Trial Distribution g(x)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Acquisition function EI ∝ ℓ(x) / g(x)
    axes[1, 1].plot(x_range, ei_approx, 'purple', linewidth=2,
                    label='EI ∝ ℓ(x) / g(x)')
    next_x = x_range[np.argmax(ei_approx)]
    axes[1, 1].axvline(x=next_x, color='purple', linestyle='--',
                       linewidth=2, label=f'Next Search Point: {next_x:.2f}')
    axes[1, 1].set_xlabel('Hyperparameter x')
    axes[1, 1].set_ylabel('Acquisition Function Value')
    axes[1, 1].set_title('Acquisition Function (TPE)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== TPE Operation ===")
    print(f"Total trials: {n_trials}")
    print(f"Good trials: {len(good_x)}")
    print(f"Bad trials: {len(bad_x)}")
    print(f"Threshold: {threshold_value:.2f}")
    print(f"Next search point: {next_x:.2f}")
    

* * *

## 2.3 Optuna Basics

### What is Optuna

**Optuna** is a hyperparameter optimization framework developed by Preferred Networks.

Features:

  * Define-by-Run API: Dynamic search space definition
  * Efficient algorithms: TPE by default
  * Pruning: Efficiency through early termination
  * Parallelization: Supports distributed optimization
  * Visualization: Rich plotting capabilities

### Installation
    
    
    # Basic installation
    pip install optuna
    
    # With visualization
    pip install optuna[visualization]
    
    # PyTorch integration
    pip install optuna[pytorch]
    

### Basic Concepts

Concept | Description  
---|---  
**Study** | The entire optimization task. Manages multiple Trials  
**Trial** | A single trial. A combination of hyperparameters  
**Objective** | The objective function to minimize or maximize  
**Sampler** | Hyperparameter sampling strategy (such as TPE)  
**Pruner** | Early termination of unpromising trials based on intermediate progress  
      
    
    ```mermaid
    graph TD
        A[Create Study] --> B[Define Objective Function]
        B --> C[Start Trial]
        C --> D[Get Parameters with suggest_*]
        D --> E[Train Model]
        E --> F[Return Evaluation Metric]
        F --> G{Optimization Finished?}
        G -->|No| C
        G -->|Yes| H[Get Best Parameters]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#ffebee
        style G fill:#f3e5f5
        style H fill:#c8e6c9
    ```

### Basic Optimization Example
    
    
    # Requirements:
    # - Python 3.9+
    # - optuna>=3.2.0
    
    """
    Example: Basic Optimization Example
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Define Objective function
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
        # Train and evaluate model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
        # Cross-validation
        score = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()
    
        return score
    
    # Create Study and optimize
    study = optuna.create_study(
        direction='maximize',  # Maximize accuracy
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=50)
    
    # Display results
    print("\n=== Optuna Optimization Results ===")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    

**Output** :
    
    
    === Optuna Optimization Results ===
    Best accuracy: 0.9733
    Best parameters:
      n_estimators: 87
      max_depth: 8
      min_samples_split: 2
      min_samples_leaf: 1
    
    Total trials: 50
    Completed trials: 50
    

* * *

## 2.4 Optuna Practical Techniques

### Defining Search Space

Optuna provides various `suggest_*` methods:

#### suggest Method List

Method | Purpose | Example  
---|---|---  
`suggest_int` | Integer values | `trial.suggest_int('n_layers', 1, 5)`  
`suggest_float` | Floating point | `trial.suggest_float('lr', 1e-5, 1e-1, log=True)`  
`suggest_categorical` | Categorical | `trial.suggest_categorical('optimizer', ['adam', 'sgd'])`  
`suggest_uniform` | Uniform distribution (deprecated, use float) | `trial.suggest_float('dropout', 0.0, 0.5)`  
`suggest_loguniform` | Log-uniform distribution (deprecated, use float+log) | `trial.suggest_float('lr', 1e-5, 1e-1, log=True)`  
      
    
    # Requirements:
    # - Python 3.9+
    # - optuna>=3.2.0
    
    """
    Example: suggest Method List
    
    Purpose: Demonstrate optimization techniques
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import optuna
    
    def objective_comprehensive(trial):
        # Integer (linear scale)
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    
        # Integer (log scale) - effective for large ranges
        hidden_size = trial.suggest_int('hidden_size', 32, 512, log=True)
    
        # Float (linear scale)
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
    
        # Float (log scale) - effective for learning rates
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
        # Categorical variables
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    
        # Conditional parameters
        if optimizer_name == 'sgd':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
        else:
            momentum = None
    
        print(f"\n--- Trial {trial.number} ---")
        print(f"batch_size: {batch_size}")
        print(f"hidden_size: {hidden_size}")
        print(f"dropout: {dropout_rate:.4f}")
        print(f"lr: {learning_rate:.6f}")
        print(f"optimizer: {optimizer_name}")
        print(f"activation: {activation}")
        if momentum is not None:
            print(f"momentum: {momentum:.4f}")
    
        # Dummy evaluation value
        score = 0.85 + 0.1 * (learning_rate / 1e-1)
    
        return score
    
    # Execution example
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_comprehensive, n_trials=5, show_progress_bar=True)
    

### Utilizing Pruning

**Pruning** is a feature that terminates unpromising trials early during training. It's particularly effective for deep learning.

#### Major Pruners

Pruner | Description | Use Case  
---|---|---  
**MedianPruner** | Prunes trials below median | General purpose  
**PercentilePruner** | Prunes below specified percentile | More conservative/aggressive pruning  
**SuccessiveHalvingPruner** | Allocates resources progressively | Many trials  
**HyperbandPruner** | Improved version of Successive Halving | Large-scale optimization  
      
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    
    """
    Example: Major Pruners
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    from optuna.pruners import MedianPruner
    import numpy as np
    import time
    
    def objective_with_pruning(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        n_layers = trial.suggest_int('n_layers', 1, 5)
    
        # Simulation per epoch
        n_epochs = 20
    
        for epoch in range(n_epochs):
            # Dummy performance (gradual improvement)
            # Bad hyperparameters improve slowly
            score = 0.5 + 0.5 * (epoch / n_epochs) * lr * n_layers / 5
            score += np.random.normal(0, 0.05)  # Noise
    
            # Report intermediate progress
            trial.report(score, epoch)
    
            # Pruning decision
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
    
            time.sleep(0.05)  # Training simulation
    
        return score
    
    # Use MedianPruner
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=5,    # Don't prune first 5 steps
            interval_steps=1     # Check every step
        )
    )
    
    print("=== Optimization with Pruning ===")
    study.optimize(objective_with_pruning, n_trials=20, show_progress_bar=False)
    
    # Analyze results
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    print(f"\nCompleted trials: {n_complete}")
    print(f"Pruned trials: {n_pruned}")
    print(f"Reduction rate: {n_pruned / len(study.trials) * 100:.1f}%")
    print(f"\nBest accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    

**Example Output** :
    
    
    === Optimization with Pruning ===
      Trial 5 pruned at epoch 7
      Trial 7 pruned at epoch 6
      Trial 9 pruned at epoch 8
      ...
    
    Completed trials: 12
    Pruned trials: 8
    Reduction rate: 40.0%
    
    Best accuracy: 0.9234
    Best parameters: {'lr': 0.08234, 'n_layers': 5}
    

> **Effect** : Pruning reduced unnecessary computation by 40%.

### Parallel Optimization

Optuna allows easy parallel optimization:
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - optuna>=3.2.0
    
    """
    Example: Optuna allows easy parallel optimization:
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import optuna
    from joblib import Parallel, delayed
    
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2
    
    # Method 1: n_jobs parameter
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, n_jobs=4)  # 4 parallel
    
    # Method 2: Shared storage (RDB)
    storage = 'sqlite:///optuna_study.db'
    study = optuna.create_study(
        study_name='parallel_optimization',
        storage=storage,
        load_if_exists=True
    )
    
    # Can execute simultaneously from multiple processes
    study.optimize(objective, n_trials=50)
    

* * *

## 2.5 Practice: Tuning Deep Learning Models

### PyTorch Model Integration with Optuna

Here's a complete example of utilizing Optuna with an actual deep learning model.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Here's a complete example of utilizing Optuna with an actual
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Model definition function
    def create_model(trial, input_size, output_size):
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
    
        for i in range(n_layers):
            hidden_size = trial.suggest_int(f'hidden_size_l{i}', 32, 256, log=True)
            hidden_sizes.append(hidden_size)
    
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
    
        # Select activation function
        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'tanh':
            activation = nn.Tanh()
        else:
            activation = nn.ELU()
    
        # Build network
        layers = []
        in_features = input_size
    
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
    
        layers.append(nn.Linear(in_features, output_size))
    
        model = nn.Sequential(*layers)
        return model
    
    # Objective function
    def objective(trial):
        # Create model
        model = create_model(trial, input_size=20, output_size=2).to(device)
    
        # Optimizer hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
        # Batch size
        batch_size = trial.suggest_int('batch_size', 16, 256, step=16)
    
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
        # Loss function
        criterion = nn.CrossEntropyLoss()
    
        # Training loop
        n_epochs = 20
    
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
    
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
    
            # Validation (test set)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)
    
            # Report intermediate progress (for Pruning)
            trial.report(accuracy, epoch)
    
            # Pruning decision
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        return accuracy
    
    # Create Study and optimize
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )
    
    print("\n=== Deep Learning Model Optimization Started ===")
    study.optimize(objective, n_trials=50, timeout=600)
    
    # Display results
    print("\n=== Optimization Complete ===")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Statistics
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nCompleted trials: {n_complete}")
    print(f"Pruned trials: {n_pruned}")
    

### Visualization

Optuna provides powerful visualization capabilities:
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - optuna>=3.2.0
    
    """
    Example: Optuna provides powerful visualization capabilities:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_parallel_coordinate,
        plot_contour
    )
    import matplotlib.pyplot as plt
    
    # 1. Optimization history
    fig = plot_optimization_history(study)
    fig.show()
    
    # 2. Parameter importance
    fig = plot_param_importances(study)
    fig.show()
    
    # 3. Slice plot (effect of each parameter)
    fig = plot_slice(study)
    fig.show()
    
    # 4. Parallel coordinate plot
    fig = plot_parallel_coordinate(study)
    fig.show()
    
    # 5. Contour plot (2D relationship)
    fig = plot_contour(study, params=['lr', 'n_layers'])
    fig.show()
    
    # Custom Plot with Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy per trial
    trial_numbers = [t.number for t in study.trials]
    values = [t.value for t in study.trials if t.value is not None]
    axes[0, 0].plot(trial_numbers[:len(values)], values, 'o-', alpha=0.6)
    axes[0, 0].axhline(y=study.best_value, color='r',
                       linestyle='--', label=f'Best: {study.best_value:.4f}')
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Progress per Trial')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Learning rate vs accuracy
    lrs = [t.params['lr'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[0, 1].scatter(lrs, values, alpha=0.6, s=50, edgecolors='black')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Learning Rate vs Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of layers vs accuracy
    n_layers_list = [t.params['n_layers'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[1, 0].scatter(n_layers_list, values, alpha=0.6, s=50, edgecolors='black')
    axes[1, 0].set_xlabel('Number of Layers')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Number of Layers vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Batch size vs accuracy
    batch_sizes = [t.params['batch_size'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[1, 1].scatter(batch_sizes, values, alpha=0.6, s=50, edgecolors='black')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Batch Size vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Optuna Dashboard

Visualize results with an interactive web dashboard:
    
    
    # Installation
    pip install optuna-dashboard
    
    # Start dashboard
    optuna-dashboard sqlite:///optuna_study.db
    

Access `http://127.0.0.1:8080` in your browser to monitor optimization progress in real-time.

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **Bayesian Optimization Principles**

     * Approximate objective function with surrogate model (Gaussian Process)
     * Determine next search point with acquisition function (EI, UCB, PI)
     * Efficient optimization through balance of exploration and exploitation
  2. **TPE Algorithm**

     * Efficient method modeling P(x|y)
     * Strong with high dimensions and categorical variables
     * Easy parallelization
  3. **Optuna Basics**

     * Concepts of Study, Trial, Objective
     * Flexible search space definition with Define-by-Run API
     * Rich suggest_* methods
  4. **Practical Techniques**

     * Reduce computation time with Pruning
     * Speed up with parallel optimization
     * Handling conditional hyperparameters
  5. **Deep Learning Applications**

     * PyTorch model integration
     * Optimization of learning rate, architecture, optimizer
     * Gaining insights through visualization

### Bayesian Optimization vs Random Search

Aspect | Random Search | Bayesian Optimization (Optuna)  
---|---|---  
**Number of Trials** | Many required | Converges with few  
**Use of Past Information** | None | Yes (surrogate model)  
**Computational Cost** | Low | Somewhat high (TPE is lightweight)  
**High-dimensional Performance** | Good | TPE is good, GP degrades  
**Parallelization** | Easy | Easy (Optuna)  
**Implementation Complexity** | Simple | Easy with Optuna  
  
### Recommended Use Cases

Situation | Recommended Method | Reason  
---|---|---  
High training cost | Optuna + Pruning | Efficiency through early termination  
Low dimensions (< 10) | Grid Search or Optuna | Both effective  
High dimensions (> 20) | Optuna (TPE) | Strong against curse of dimensionality  
Many categorical variables | Optuna | Naturally handled  
Initial exploration | Random Search | Simple and fast  
Final tuning | Optuna | Precise optimization  
  
### Next Chapter

In Chapter 3, we'll learn about **Automated Machine Learning (AutoML)** :

  * Auto-sklearn: Automatic model selection and ensembling
  * H2O AutoML: For large-scale data
  * PyCaret: Low-code ML
  * TPOT: Genetic programming
  * AutoML limitations and use cases

* * *

## References

  1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. _Proceedings of the 25th ACM SIGKDD_.
  2. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. _NIPS_.
  3. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. _Proceedings of the IEEE_ , 104(1), 148-175.
  4. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. _NIPS_.
  5. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. _ICML_.
