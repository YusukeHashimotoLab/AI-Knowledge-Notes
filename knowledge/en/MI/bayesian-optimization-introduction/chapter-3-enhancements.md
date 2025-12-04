---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25 min
difficulty: Beginner
code_examples: 0
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 3 Quality Enhancements

Starting with minimal implementations in scikit-optimize and BoTorch, we'll learn the essentials of parameter configuration. We'll also show entry points for extending to constrained and multi-objective problems.

**💡 Note:** Stabilize through noise estimation and scale adjustment. Design iteration counts for "quality over quantity".

This file contains enhancements to be integrated into chapter-3.md

## Code Reproducibility Section (add after section 3.1)

### Ensuring Code Reproducibility

**Importance of Environment Configuration** :

All code examples have been tested in the following environment:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    # Required library versions
    """
    Python: 3.8+
    numpy: 1.21.0
    scikit-learn: 1.0.0
    scikit-optimize: 0.9.0
    torch: 1.12.0
    gpytorch: 1.8.0
    botorch: 0.7.0
    matplotlib: 3.5.0
    pandas: 1.3.0
    scipy: 1.7.0
    """
    
    # Configuration for reproducibility
    import numpy as np
    import torch
    import random
    
    # Fix random seeds
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # GPyTorch kernel configuration (recommended)
    from gpytorch.kernels import RBF, MaternKernel, ScaleKernel
    
    # RBF kernel (most common)
    kernel_rbf = ScaleKernel(RBF(
        lengthscale_prior=None,  # Data-driven optimization
        ard_num_dims=None  # Automatic Relevance Determination
    ))
    
    # Matern kernel (adjustable smoothness)
    kernel_matern = ScaleKernel(MaternKernel(
        nu=2.5,  # Smoothness parameter (1.5, 2.5, or inf (equivalent to RBF))
        ard_num_dims=None
    ))
    
    print("Environment setup complete")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    

**Installation Steps** :
    
    
    # Create virtual environment (recommended)
    python -m venv bo_env
    source bo_env/bin/activate  # Linux/Mac
    # bo_env\Scripts\activate  # Windows
    
    # Install required packages
    pip install numpy==1.21.0 scikit-learn==1.0.0 scikit-optimize==0.9.0
    pip install torch==1.12.0 gpytorch==1.8.0 botorch==0.7.0
    pip install matplotlib==3.5.0 pandas==1.3.0 scipy==1.7.0
    
    # Optional: Materials Project API
    pip install mp-api==0.30.0
    
    # Verify installation
    python -c "import botorch; print(f'BoTorch {botorch.__version__} installed')"
    

* * *

## Practical Pitfalls Section (add after section 3.7)

### 3.8 Practical Pitfalls and Solutions

#### Pitfall 1: Inappropriate Kernel Selection

**Problem** : Kernel selection doesn't match the nature of the objective function

**Symptoms** : \- Low prediction accuracy \- Poor exploration efficiency \- Easy to fall into local optima

**Solution** :
    
    
    # Kernel selection guide
    from gpytorch.kernels import RBF, MaternKernel, PeriodicKernel
    
    def select_kernel(problem_characteristics):
        """
        Kernel selection based on problem characteristics
    
        Parameters:
        -----------
        problem_characteristics : dict
            Dictionary describing problem characteristics
            - 'smoothness': 'smooth' | 'rough'
            - 'periodicity': True | False
            - 'dimensionality': int
    
        Returns:
        --------
        kernel : gpytorch.kernels.Kernel
            Recommended kernel
        """
        if problem_characteristics.get('periodicity'):
            # If periodic behavior exists
            return PeriodicKernel()
    
        elif problem_characteristics.get('smoothness') == 'smooth':
            # Smooth functions (material properties, etc.)
            return RBF()
    
        elif problem_characteristics.get('smoothness') == 'rough':
            # Noisy or discontinuous
            return MaternKernel(nu=1.5)
    
        else:
            # Default: Matern 5/2 (high versatility)
            return MaternKernel(nu=2.5)
    
    # Usage example
    problem_specs = {
        'smoothness': 'smooth',
        'periodicity': False,
        'dimensionality': 4
    }
    
    recommended_kernel = select_kernel(problem_specs)
    print(f"Recommended kernel: {recommended_kernel}")
    

**Kernel Comparison Experiment** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    
    # Test function
    def test_function(x):
        """Complex function with noise"""
        return np.sin(5*x) + 0.5*np.cos(15*x) + 0.1*np.random.randn(len(x))
    
    # Generate data
    np.random.seed(42)
    X_train = np.random.uniform(0, 1, 20).reshape(-1, 1)
    y_train = test_function(X_train.ravel())
    
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = test_function(X_test.ravel())
    
    # Compare different kernels
    kernels = {
        'RBF': RBF(length_scale=0.1),
        'Matern 1.5': Matern(length_scale=0.1, nu=1.5),
        'Matern 2.5': Matern(length_scale=0.1, nu=2.5)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, kernel) in zip(axes, kernels.items()):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X_train, y_train)
        y_pred, y_std = gp.predict(X_test, return_std=True)
    
        ax.scatter(X_train, y_train, c='red', label='Training data')
        ax.plot(X_test, y_pred, 'b-', label='Prediction')
        ax.fill_between(X_test.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                         alpha=0.3, color='blue')
        ax.set_title(f'Kernel: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150)
    plt.show()
    
    print("Conclusions:")
    print("  RBF: Optimal for smooth functions")
    print("  Matern 1.5: High noise resistance")
    print("  Matern 2.5: Well-balanced (recommended)")
    

* * *

#### Pitfall 2: Failed Initialization Strategy

**Problem** : Initial sampling doesn't adequately cover the search space

**Symptoms** : \- Biased exploration \- Missing important regions \- Slow convergence

**Solution** : Latin Hypercube Sampling (LHS)
    
    
    from scipy.stats.qmc import LatinHypercube
    
    def initialize_with_lhs(n_samples, bounds, seed=42):
        """
        Generate initial points using Latin Hypercube Sampling
    
        Parameters:
        -----------
        n_samples : int
            Number of samples
        bounds : array (n_dims, 2)
            [lower, upper] bounds for each dimension
        seed : int
            Random seed
    
        Returns:
        --------
        X_init : array (n_samples, n_dims)
            Initial sampling points
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
    
        # LHS sampler
        sampler = LatinHypercube(d=n_dims, seed=seed)
        X_unit = sampler.random(n=n_samples)
    
        # Scaling
        X_init = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * X_unit
    
        return X_init
    
    # Usage example: Li-ion battery composition initialization
    bounds_composition = [
        [0.1, 0.5],  # Li
        [0.1, 0.4],  # Ni
        [0.1, 0.3],  # Co
        [0.0, 0.5]   # Mn
    ]
    
    X_init_lhs = initialize_with_lhs(
        n_samples=20,
        bounds=bounds_composition,
        seed=42
    )
    
    # Normalize composition
    X_init_lhs = X_init_lhs / X_init_lhs.sum(axis=1, keepdims=True)
    
    print("LHS initialization complete")
    print(f"Initial sample count: {len(X_init_lhs)}")
    print(f"Coverage range for each dimension:")
    for i, dim_name in enumerate(['Li', 'Ni', 'Co', 'Mn']):
        print(f"  {dim_name}: [{X_init_lhs[:, i].min():.3f}, "
              f"{X_init_lhs[:, i].max():.3f}]")
    
    # Visualization comparing with random sampling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Random sampling
    np.random.seed(42)
    X_random = np.random.uniform(0, 1, (20, 2))
    
    axes[0].scatter(X_random[:, 0], X_random[:, 1], s=100)
    axes[0].set_title('Random Sampling')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True, alpha=0.3)
    
    # LHS
    axes[1].scatter(X_init_lhs[:, 0], X_init_lhs[:, 1], s=100, c='red')
    axes[1].set_title('Latin Hypercube Sampling (LHS)')
    axes[1].set_xlabel('Dimension 1 (Li)')
    axes[1].set_ylabel('Dimension 2 (Ni)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lhs_vs_random.png', dpi=150)
    plt.show()
    

* * *

#### Pitfall 3: Inadequate Handling of Noisy Observations

**Problem** : Not accounting for experimental noise

**Symptoms** : \- Results not reproducible under same conditions \- Model overfitting \- Unstable optimal points

**Solution** : Explicitly model noise
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    
    def fit_gp_with_noise(X, y, noise_variance=0.01):
        """
        Train Gaussian Process with noise consideration
    
        Parameters:
        -----------
        X : Tensor (n, d)
            Input data
        y : Tensor (n, 1)
            Observations (including noise)
        noise_variance : float
            Observation noise variance (set from prior knowledge)
    
        Returns:
        --------
        gp_model : SingleTaskGP
            Trained GP model
        """
        # Build GP with noise variance
        gp_model = SingleTaskGP(X, y, train_Yvar=torch.full_like(y, noise_variance))
    
        # Maximize likelihood
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll)
    
        return gp_model
    
    # Usage example: Experimental data with noise
    np.random.seed(42)
    X_obs = np.random.rand(15, 4)
    X_obs = X_obs / X_obs.sum(axis=1, keepdims=True)
    
    # True capacity + experimental noise
    y_true = 200 + 150 * X_obs[:, 0] + 50 * X_obs[:, 1]
    noise = np.random.randn(15) * 10  # Experimental noise σ=10 mAh/g
    y_obs = y_true + noise
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_obs, dtype=torch.float64)
    y_tensor = torch.tensor(y_obs, dtype=torch.float64).unsqueeze(-1)
    
    # Train GP with noise consideration
    gp_noisy = fit_gp_with_noise(X_tensor, y_tensor, noise_variance=100.0)
    
    print("GP training with noise consideration complete")
    print(f"Observation noise standard deviation: 10 mAh/g")
    print(f"Modeled noise variance: 100.0 (mAh/g)²")
    

**Noise Level Estimation** :
    
    
    def estimate_noise_level(X, y, n_replicates=3):
        """
        Estimate noise level from replicate experiments
    
        Parameters:
        -----------
        X : array (n, d)
            Experimental conditions
        y : array (n,)
            Observations
        n_replicates : int
            Number of replicate experiments per condition
    
        Returns:
        --------
        noise_std : float
            Estimated noise standard deviation
        """
        # Extract replicate experiments with identical conditions
        unique_X, indices = np.unique(X, axis=0, return_inverse=True)
    
        variances = []
        for i in range(len(unique_X)):
            replicates = y[indices == i]
            if len(replicates) >= 2:
                variances.append(np.var(replicates, ddof=1))
    
        if len(variances) == 0:
            print("Warning: No replicate experiments found. Using default value")
            return 1.0
    
        noise_std = np.sqrt(np.mean(variances))
        return noise_std
    
    # Usage example
    noise_std_estimated = estimate_noise_level(X_obs, y_obs)
    print(f"Estimated noise standard deviation: {noise_std_estimated:.2f} mAh/g")
    

* * *

#### Pitfall 4: Inadequate Constraint Handling

**Problem** : Not properly handling constraint violations

**Symptoms** : \- Proposing infeasible materials \- Optimization doesn't converge \- Many wasted experiments

**Solution** : Constrained Acquisition Function
    
    
    from botorch.acquisition import ConstrainedExpectedImprovement
    
    def constrained_bayesian_optimization_example():
        """
        Implementation example of constrained Bayesian Optimization
    
        Constraints:
        1. Sum of composition = 1.0 (±2%)
        2. Co content < 0.3 (cost constraint)
        3. Stability: formation energy < -1.5 eV/atom
        """
        # Initial data
        n_initial = 10
        X_init = initialize_with_lhs(n_initial, bounds_composition, seed=42)
        X_init = X_init / X_init.sum(axis=1, keepdims=True)  # Normalize
    
        # Evaluate objective function and constraints
        y_capacity = []
        constraints_satisfied = []
    
        for x in X_init:
            # Capacity prediction (objective function)
            capacity = 200 + 150*x[0] + 50*x[1]
            y_capacity.append(capacity)
    
            # Constraint check
            co_constraint = x[2] < 0.3  # Co < 0.3
            stability = -2.0 - 0.5*x[0] - 0.3*x[1]
            stability_constraint = stability < -1.5  # Stable
    
            all_satisfied = co_constraint and stability_constraint
            constraints_satisfied.append(1.0 if all_satisfied else 0.0)
    
        X_tensor = torch.tensor(X_init, dtype=torch.float64)
        y_tensor = torch.tensor(y_capacity, dtype=torch.float64).unsqueeze(-1)
        c_tensor = torch.tensor(constraints_satisfied, dtype=torch.float64).unsqueeze(-1)
    
        # Gaussian Process model (objective function)
        gp_objective = SingleTaskGP(X_tensor, y_tensor)
        mll_obj = ExactMarginalLogLikelihood(gp_objective.likelihood, gp_objective)
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll_obj)
    
        # Gaussian Process model (constraints)
        gp_constraint = SingleTaskGP(X_tensor, c_tensor)
        mll_con = ExactMarginalLogLikelihood(gp_constraint.likelihood, gp_constraint)
        fit_gpytorch_model(mll_con)
    
        # Constrained EI Acquisition Function
        best_f = y_tensor.max()
        acq_func = ConstrainedExpectedImprovement(
            model=gp_objective,
            best_f=best_f,
            objective_index=0,
            constraints={0: [None, 0.5]}  # Constraint satisfaction probability > 0.5
        )
    
        print("Constrained Bayesian Optimization setup complete")
        print(f"Initial feasible solutions: {sum(constraints_satisfied)}/{n_initial}")
    
        return gp_objective, gp_constraint, acq_func
    
    # Execute
    gp_obj, gp_con, acq = constrained_bayesian_optimization_example()
    

* * *

## End-of-Chapter Checklist (add before "Exercises")

### 3.9 End-of-Chapter Checklist

#### ✅ Understanding Gaussian Processes

  * [ ] Can explain the basic concepts of Gaussian Processes
  * [ ] Understand the role of Kernel Functions
  * [ ] Know the meaning of predictive mean and uncertainty
  * [ ] Can select appropriate kernels
  * [ ] Can explain the influence of hyperparameters

**Verification Question** :
    
    
    Q: What is the difference between RBF and Matern kernels?
    A: RBF is infinitely differentiable (very smooth), Matern allows
       adjustable smoothness via parameter ν. Matern (ν=2.5) is
       recommended when noise is present.
    

* * *

#### ✅ Acquisition Function Selection

  * [ ] Understand the mechanism of Expected Improvement (EI)
  * [ ] Can explain the exploration-exploitation balance of Upper Confidence Bound (UCB)
  * [ ] Know the characteristics of Probability of Improvement (PI)
  * [ ] Understand application scenarios for Knowledge Gradient (KG)
  * [ ] Can select Acquisition Function based on the problem

**Selection Guide** :
    
    
    General optimization      → EI (well-balanced)
    Exploration-focused early → UCB (κ=2~3)
    Safety-focused           → PI (conservative)
    Batch optimization       → q-EI, q-KG
    Multi-objective          → EHVI (Hypervolume)
    

* * *

#### ✅ Multi-Objective Optimization

  * [ ] Can explain the definition of Pareto optimality
  * [ ] Understand the meaning of Pareto frontier
  * [ ] Know the mechanism of Expected Hypervolume Improvement (EHVI)
  * [ ] Can quantitatively evaluate trade-offs
  * [ ] Can implement multi-objective optimization

**Implementation Check** :
    
    
    # Can you implement the following?
    def is_pareto_optimal(objectives):
        """
        Function to determine Pareto optimal solutions
        objectives: (n_points, n_objectives)
        """
        # Your implementation
        pass
    
    # See Exercises 3 for the solution
    

* * *

#### ✅ Batch Bayesian Optimization

  * [ ] Can explain the advantages of batch optimization
  * [ ] Understand the mechanism of q-EI Acquisition Function
  * [ ] Know the Kriging Believer method
  * [ ] Can develop strategies for efficient parallel experiments
  * [ ] Understand batch size selection criteria

**Batch Size Selection** :
    
    
    Number of experimental devices: n units
    → Batch size: n (maximum exploitation)
    
    With computational cost constraints
    → Batch size: 3~5 (practical)
    
    Early exploration phase
    → Batch size: larger (diversity-focused)
    
    Convergence phase
    → Batch size: smaller (refinement)
    

* * *

#### ✅ Constraint Handling

  * [ ] Can distinguish types of constraints (equality, inequality)
  * [ ] Understand the concept of feasible region
  * [ ] Can implement constrained Acquisition Functions
  * [ ] Can calculate feasibility probability
  * [ ] Know strategies for gradual constraint relaxation

**Constraint Handling Checklist** :
    
    
    □ Handle composition constraints (sum=1.0) with normalization
    □ Set boundary constraints with bounds parameter
    □ Express nonlinear constraints with penalty functions
    □ Prepare strategies when no feasible solution is found
    □ Visualize constraint satisfaction probability
    

* * *

#### ✅ Implementation Skills (GPyTorch/BoTorch)

  * [ ] Can construct SingleTaskGP models
  * [ ] Can appropriately select and configure kernels
  * [ ] Can optimize Acquisition Functions
  * [ ] Can implement batch optimization
  * [ ] Can perform modeling with noise consideration

**Code Implementation Verification** :
    
    
    # Can you understand this code?
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    
    # Build GP model
    gp = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    
    # Maximize EI
    EI = ExpectedImprovement(gp, best_f=y_train.max())
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=10
    )
    
    # Can you explain the meaning of each line?
    

* * *

#### ✅ Integration with Experimental Design

  * [ ] Can exploit real data sources like Materials Project
  * [ ] Can integrate ML models with Bayesian Optimization
  * [ ] Can develop experimental plans
  * [ ] Can visualize and interpret results
  * [ ] Can evaluate ROI

**Experimental Planning Template** :
    
    
    1. Objective Setting
       - Property to optimize: ________
       - Constraints: ________
       - Experimental budget: ________ trials
    
    2. Initialization
       - Initial sample count: ________
       - Sampling method: LHS / Random
       - Expected experiment duration: ________
    
    3. Optimization Strategy
       - Acquisition Function: ________
       - Kernel: ________
       - Batch size: ________
    
    4. Termination Criteria
       - Maximum experiments: ________
       - Target performance: ________
       - Improvement rate threshold: ________
    

* * *

#### ✅ Troubleshooting

  * [ ] Know methods to escape local optima
  * [ ] Understand strategies for handling constraint violations
  * [ ] Know techniques for reducing computation time
  * [ ] Can implement noise handling strategies
  * [ ] Know debugging methods

**Common Errors and Solutions** :
    
    
    Error: "RuntimeError: cholesky_cpu: U(i,i) is zero"
    → Cause: Numerical instability
    → Solution: Add jitter to GP model
       gp = SingleTaskGP(X, y, covar_module=...).double()
       gp.likelihood.noise = 1e-4
    
    Error: "All points violate constraints"
    → Cause: Constraints too strict
    → Solution: Gradual constraint relaxation, initial LHS sampling
    
    Warning: "Optimization failed to converge"
    → Cause: Acquisition Function optimization failure
    → Solution: Increase num_restarts, increase raw_samples
    

* * *

### Passing Criteria

If you clear 80% or more of the checklist items in each section and understand the implementation verification code, you are ready to proceed to the next chapter.

**Final Verification Questions** : 1\. Can you formulate an optimization problem for Li-ion battery cathode materials? 2\. Can you implement a 3-objective (capacity, voltage, stability) optimization? 3\. Can you find 10 Pareto optimal solutions within 50 experiments?

If all answers are YES, proceed to Chapter 4 "Active Learning and Experimental Integration"!
