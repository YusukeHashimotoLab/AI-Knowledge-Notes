---
title: "Chapter 3: Acquisition Functions"
chapter_title: "Chapter 3: Acquisition Functions"
subtitle: Strategies for Balancing Exploration and Exploitation
---

This chapter covers Acquisition Functions. You will learn essential concepts and techniques.

## 3.1 What are Acquisition Functions?

Acquisition functions are crucial components in Bayesian optimization that determine which candidate point should be evaluated next. They take the mean and uncertainty predicted by the Gaussian process as input and quantify "which point should be tested next." 

#### =¡ Role of Acquisition Functions

  * **Exploration** : Investigate regions with high uncertainty
  * **Exploitation** : Examine areas around the current best value in detail
  * **Balance Control** : Regulate the ratio of exploration to exploitation based on problem and progress

In this chapter, we will implement seven representative acquisition functions and understand their characteristics and application scenarios. 

## 3.2 Expected Improvement (EI)

The most widely used acquisition function. It maximizes the expected value of improvement from the current best value. 

### Example 1: Implementation of Expected Improvement
    
    
    # Implementation of Expected Improvement (EI)
    import numpy as np
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    import matplotlib.pyplot as plt
    
    # 1D test function (process temperature optimization)
    def process_yield(x):
        """Simulation of reaction yield (response to temperature)"""
        return -(x - 2.5)**2 + 5 + 0.3 * np.sin(5 * x)
    
    # Expected Improvement calculation
    def expected_improvement(X, gp, y_best, xi=0.01):
        """EI acquisition function
    
        Args:
            X: Evaluation points
            gp: Trained Gaussian process
            y_best: Current best value
            xi: Improvement threshold (smaller values emphasize exploitation)
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
    
        # Calculate improvement
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
    
        return ei
    
    # Initial sampling
    np.random.seed(42)
    X_init = np.random.uniform(0, 5, 3).reshape(-1, 1)
    y_init = process_yield(X_init)
    
    # Build Gaussian process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_init, y_init)
    
    # Evaluate EI
    X_test = np.linspace(0, 5, 200).reshape(-1, 1)
    y_best = y_init.max()
    ei_values = expected_improvement(X_test, gp, y_best)
    
    # Next candidate point
    next_x = X_test[np.argmax(ei_values)]
    
    print(f"Current best value: {y_best:.3f}")
    print(f"Next experimental candidate: {next_x[0]:.3f}")
    print(f"EI value: {ei_values.max():.4f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Upper panel: GP prediction
    mu, sigma = gp.predict(X_test, return_std=True)
    ax1.plot(X_test, process_yield(X_test), 'r--', label='True function', alpha=0.5)
    ax1.plot(X_test, mu, 'b-', label='GP mean')
    ax1.fill_between(X_test.ravel(), mu - 1.96*sigma, mu + 1.96*sigma,
                     alpha=0.2, label='95% CI')
    ax1.scatter(X_init, y_init, c='red', s=100, zorder=10, label='Observations')
    ax1.axvline(next_x, color='green', linestyle=':', label='Next sample')
    ax1.set_ylabel('Yield')
    ax1.legend()
    ax1.set_title('Gaussian Process Prediction')
    
    # Lower panel: EI values
    ax2.plot(X_test, ei_values, 'g-', label='Expected Improvement')
    ax2.axvline(next_x, color='green', linestyle=':', label='Maximum EI')
    ax2.set_xlabel('Temperature (x)')
    ax2.set_ylabel('EI')
    ax2.legend()
    ax2.set_title('Expected Improvement Acquisition Function')
    
    plt.tight_layout()
    plt.savefig('ei_acquisition.png', dpi=150, bbox_inches='tight')
    print("Saved: ei_acquisition.png")
    

####  Characteristics of EI

  * Well-balanced exploration and exploitation (recommended default)
  * Adjustable via parameter `xi` (typically 0.01-0.1)
  * Less prone to local optima

## 3.3 Probability of Improvement (PI)

An acquisition function that maximizes the probability of improving the current best value. A more conservative strategy than EI. 

### Example 2: Implementation of Probability of Improvement
    
    
    # Implementation of Probability of Improvement (PI)
    def probability_of_improvement(X, gp, y_best, xi=0.01):
        """PI acquisition function
    
        Args:
            X: Evaluation points
            gp: Trained Gaussian process
            y_best: Current best value
            xi: Improvement threshold
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
    
        with np.errstate(divide='warn'):
            Z = (mu - y_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
    
        return pi
    
    # Comparison of PI and EI
    pi_values = probability_of_improvement(X_test, gp, y_best)
    next_x_pi = X_test[np.argmax(pi_values)]
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X_test, ei_values/ei_values.max(), 'g-', label='EI (normalized)', linewidth=2)
    ax.plot(X_test, pi_values, 'b-', label='PI', linewidth=2)
    ax.axvline(next_x, color='green', linestyle=':', alpha=0.7, label=f'EI max: {next_x[0]:.2f}')
    ax.axvline(next_x_pi, color='blue', linestyle=':', alpha=0.7, label=f'PI max: {next_x_pi[0]:.2f}')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Acquisition Value')
    ax.legend()
    ax.set_title('EI vs PI Acquisition Functions')
    ax.grid(alpha=0.3)
    plt.savefig('ei_vs_pi.png', dpi=150, bbox_inches='tight')
    
    print(f"EI recommended point: {next_x[0]:.3f}")
    print(f"PI recommended point: {next_x_pi[0]:.3f}")
    print(f"Difference: {abs(next_x[0] - next_x_pi[0]):.3f}")
    

####   PI Cautions

PI only considers the "probability" of improvement, not the "magnitude" of improvement. Therefore, it may select points with high probability even for minimal improvements. EI is more practical in most cases.

## 3.4 Upper Confidence Bound (UCB)

Maximizes the weighted sum of predicted mean and uncertainty. Allows explicit control of exploration degree through parameters. 

### Example 3: UCB Implementation and Exploration Parameter Effects
    
    
    # Implementation of Upper Confidence Bound (UCB)
    def upper_confidence_bound(X, gp, kappa=2.0):
        """UCB acquisition function
    
        Args:
            X: Evaluation points
            gp: Trained Gaussian process
            kappa: Exploration parameter (larger values emphasize exploration)
        """
        mu, sigma = gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    # Comparison with different kappa values
    kappas = [0.5, 1.0, 2.0, 5.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, kappa in enumerate(kappas):
        ax = axes[i//2, i%2]
    
        ucb_values = upper_confidence_bound(X_test, gp, kappa=kappa)
        next_x_ucb = X_test[np.argmax(ucb_values)]
    
        # GP prediction
        mu, sigma = gp.predict(X_test, return_std=True)
    
        ax.plot(X_test, process_yield(X_test), 'r--', alpha=0.3, label='True')
        ax.plot(X_test, mu, 'b-', alpha=0.5, label='GP mean')
        ax.fill_between(X_test.ravel(), mu - sigma, mu + sigma, alpha=0.1)
        ax.plot(X_test, ucb_values, 'g-', linewidth=2, label='UCB')
        ax.scatter(X_init, y_init, c='red', s=80, zorder=10)
        ax.axvline(next_x_ucb, color='green', linestyle=':', linewidth=2)
    
        ax.set_title(f'º = {kappa} (Next: {next_x_ucb[0]:.2f})')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ucb_kappa_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\nUCB exploration parameter effects:")
    for kappa in kappas:
        ucb = upper_confidence_bound(X_test, gp, kappa=kappa)
        next_x = X_test[np.argmax(ucb)]
        print(f"º={kappa:.1f}: Next experiment point = {next_x[0]:.3f}")
    

#### =¡ UCB Parameter Selection Guide

  * `kappa = 0.5-1.0`: Exploitation-focused (precise search around best value)
  * `kappa = 2.0-3.0`: Balanced (recommended default)
  * `kappa = 5.0-10.0`: Exploration-focused (investigation of unknown regions)

## 3.5 Thompson Sampling

Bayesian approach. Samples from the posterior distribution of the Gaussian process and uses its maximum as the next candidate. 

### Example 4: Thompson Sampling Implementation
    
    
    # Implementation of Thompson Sampling
    def thompson_sampling(X, gp, n_samples=10, random_state=None):
        """Thompson Sampling acquisition function
    
        Args:
            X: Evaluation points
            gp: Trained Gaussian process
            n_samples: Number of samples
            random_state: Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)
    
        # Sample functions from GP
        y_samples = gp.sample_y(X, n_samples=n_samples)
    
        # Position of maximum value for each sample
        max_indices = np.argmax(y_samples, axis=0)
    
        # Frequency-based score (which point is selected most)
        score = np.zeros(len(X))
        for idx in max_indices:
            score[idx] += 1
    
        return score, y_samples
    
    # Execute Thompson Sampling
    ts_score, samples = thompson_sampling(X_test, gp, n_samples=50, random_state=42)
    next_x_ts = X_test[np.argmax(ts_score)]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Sampled functions
    mu, sigma = gp.predict(X_test, return_std=True)
    ax1.plot(X_test, process_yield(X_test), 'r--', alpha=0.5, linewidth=2, label='True')
    ax1.plot(X_test, mu, 'b-', linewidth=2, label='GP mean')
    for i in range(min(10, samples.shape[1])):
        ax1.plot(X_test, samples[:, i], 'gray', alpha=0.3, linewidth=0.5)
    ax1.scatter(X_init, y_init, c='red', s=100, zorder=10, label='Observations')
    ax1.axvline(next_x_ts, color='green', linestyle=':', linewidth=2, label='TS selection')
    ax1.set_ylabel('Yield')
    ax1.legend()
    ax1.set_title(f'Thompson Sampling (50 samples)')
    ax1.grid(alpha=0.3)
    
    # Selection frequency
    ax2.bar(X_test.ravel(), ts_score, width=0.03, color='green', alpha=0.7)
    ax2.axvline(next_x_ts, color='green', linestyle=':', linewidth=2)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Selection Frequency')
    ax2.set_title('Thompson Sampling Scores')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thompson_sampling.png', dpi=150, bbox_inches='tight')
    
    print(f"Thompson Sampling recommended point: {next_x_ts[0]:.3f}")
    print(f"Selection frequency: {ts_score.max()}/50")
    

####  Advantages of Thompson Sampling

  * Theoretically optimal exploration strategy (Bayesian regret minimization)
  * Proven track record in multi-armed bandit problems
  * Randomness helps avoid local optima

## 3.6 Knowledge Gradient (KG)

A lookahead acquisition function that predicts the amount of improvement after taking the next sample. 

### Example 5: Knowledge Gradient Implementation
    
    
    # Simplified implementation of Knowledge Gradient
    def knowledge_gradient(X, X_candidate, gp, n_samples=100):
        """Knowledge Gradient acquisition function
    
        Args:
            X: Existing observation points
            X_candidate: Candidate points
            gp: Trained Gaussian process
            n_samples: Number of Monte Carlo samples
        """
        # Current best value
        y_pred_current = gp.predict(X_candidate, return_std=False)
        current_best = y_pred_current.max()
    
        kg_values = np.zeros(len(X))
    
        for i, x_new in enumerate(X):
            # Expected improvement when adding new point
            # Monte Carlo approximation
            improvements = []
    
            for _ in range(n_samples):
                # Sample at new point
                y_new_sample = gp.sample_y(x_new.reshape(1, -1), n_samples=1)[0, 0]
    
                # Update GP (hypothetically)
                X_temp = np.vstack([gp.X_train_, x_new.reshape(1, -1)])
                y_temp = np.hstack([gp.y_train_, y_new_sample])
    
                gp_temp = GaussianProcessRegressor(kernel=gp.kernel_)
                gp_temp.fit(X_temp, y_temp)
    
                # Best prediction after update
                y_pred_new = gp_temp.predict(X_candidate, return_std=False)
                new_best = y_pred_new.max()
    
                improvements.append(max(0, new_best - current_best))
    
            kg_values[i] = np.mean(improvements)
    
        return kg_values
    
    # KG calculation (coarse grid due to high computational cost)
    X_coarse = np.linspace(0, 5, 30).reshape(-1, 1)
    X_candidate = np.linspace(0, 5, 50).reshape(-1, 1)
    
    print("Calculating Knowledge Gradient (may take several minutes)...")
    kg_values = knowledge_gradient(X_coarse, X_candidate, gp, n_samples=20)
    next_x_kg = X_coarse[np.argmax(kg_values)]
    
    # Comparison with other acquisition functions
    ei_coarse = expected_improvement(X_coarse, gp, y_best)
    ucb_coarse = upper_confidence_bound(X_coarse, gp, kappa=2.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X_coarse, ei_coarse/ei_coarse.max(), 'g-', label='EI (norm)', linewidth=2)
    ax.plot(X_coarse, ucb_coarse/ucb_coarse.max(), 'b-', label='UCB (norm)', linewidth=2)
    ax.plot(X_coarse, kg_values/kg_values.max(), 'r-', label='KG (norm)', linewidth=2)
    ax.axvline(next_x_kg, color='red', linestyle=':', label=f'KG max: {next_x_kg[0]:.2f}')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Normalized Acquisition Value')
    ax.legend()
    ax.set_title('Knowledge Gradient vs Other Acquisition Functions')
    ax.grid(alpha=0.3)
    plt.savefig('knowledge_gradient.png', dpi=150, bbox_inches='tight')
    
    print(f"\nKnowledge Gradient recommended point: {next_x_kg[0]:.3f}")
    print(f"KG value: {kg_values.max():.4f}")
    

####   KG Computational Cost

Knowledge Gradient has high computational cost because it retrains the GP for each candidate point. In practice, use evaluation on coarse grids or analytical approximations (KG*). 

## 3.7 Entropy Search

Selects points that most reduce the uncertainty (entropy) about the location of the optimal solution. 

### Example 6: Conceptual Implementation of Entropy Search
    
    
    # Conceptual implementation of Entropy Search
    from scipy.stats import entropy
    
    def entropy_search_approx(X, gp, X_candidate, n_samples=100):
        """Approximate implementation of Entropy Search
    
        Args:
            X: Evaluation points
            gp: Trained Gaussian process
            X_candidate: Optimal solution candidate region
            n_samples: Number of samples
        """
        # Prior distribution of optimal solution location (based on predictions at candidate points)
        mu_candidate, sigma_candidate = gp.predict(X_candidate, return_std=True)
    
        # Normalize to probability distribution
        prob_optimum = np.exp(mu_candidate / sigma_candidate.max())
        prob_optimum /= prob_optimum.sum()
    
        # Current entropy
        current_entropy = entropy(prob_optimum)
    
        es_values = np.zeros(len(X))
    
        for i, x_new in enumerate(X):
            # Prediction at new point
            mu_new, sigma_new = gp.predict(x_new.reshape(1, -1), return_std=True)
    
            # Approximate entropy reduction via sampling
            entropy_reductions = []
    
            for _ in range(n_samples):
                # Sample observation value at new point
                y_new = np.random.normal(mu_new[0], sigma_new[0])
    
                # Approximate optimal solution distribution after GP update
                # Simplified: impact of new point observation on candidate probabilities
                updated_prob = prob_optimum.copy()
    
                # Impact when new point is better than candidates
                for j, x_cand in enumerate(X_candidate):
                    if y_new > mu_candidate[j]:
                        updated_prob[j] *= 0.5  # Simplified weighting
    
                updated_prob /= updated_prob.sum() + 1e-10
                new_entropy = entropy(updated_prob)
    
                entropy_reductions.append(current_entropy - new_entropy)
    
            es_values[i] = np.mean(entropy_reductions)
    
        return es_values
    
    # Execute Entropy Search
    es_values = entropy_search_approx(X_coarse, gp, X_candidate, n_samples=50)
    next_x_es = X_coarse[np.argmax(es_values)]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Probability distribution of optimal solution
    mu_cand, sigma_cand = gp.predict(X_candidate, return_std=True)
    prob_opt = np.exp(mu_cand / sigma_cand.max())
    prob_opt /= prob_opt.sum()
    
    ax1.plot(X_candidate, prob_opt, 'b-', linewidth=2, label='P(x is optimum)')
    ax1.fill_between(X_candidate.ravel(), 0, prob_opt, alpha=0.3)
    ax1.scatter(X_init, [0.01]*len(X_init), c='red', s=100, zorder=10, label='Observations')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.set_title('Estimated Distribution of Optimal Point')
    ax1.grid(alpha=0.3)
    
    # Entropy Search values
    ax2.plot(X_coarse, es_values, 'r-', linewidth=2, label='Entropy Reduction')
    ax2.axvline(next_x_es, color='red', linestyle=':', linewidth=2, label=f'ES max: {next_x_es[0]:.2f}')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Expected Entropy Reduction')
    ax2.legend()
    ax2.set_title('Entropy Search Acquisition Function')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropy_search.png', dpi=150, bbox_inches='tight')
    
    print(f"Entropy Search recommended point: {next_x_es[0]:.3f}")
    print(f"Expected entropy reduction: {es_values.max():.4f}")
    

#### =¡ Characteristics of Entropy Search

  * Rigorous formulation based on information theory
  * Efficient for optimal solution identification (suited for identification problems)
  * Complex implementation (approximation methods commonly used)

## 3.8 Comparative Experiments of Acquisition Functions

We compare the six acquisition functions learned so far in a real process optimization problem. 

### Example 7: Comprehensive Comparison of Acquisition Functions
    
    
    # Performance comparison experiment of acquisition functions
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # Test function: Complex process response
    def complex_process(x):
        """Multimodal process response (2D temperature and pressure)"""
        x1, x2 = x[:, 0], x[:, 1]
        return (-(x1 - 3)**2 - (x2 - 2)**2 + 10 +
                2 * np.sin(3*x1) * np.cos(2*x2) +
                np.random.normal(0, 0.1, len(x1)))
    
    # Bayesian optimization with each acquisition function
    def run_bo_with_acquisition(acquisition_func, n_iterations=20):
        """Run BO with specified acquisition function"""
        # Initial sampling
        np.random.seed(42)
        X_init = np.random.uniform([0, 0], [5, 4], (5, 2))
        y_init = complex_process(X_init)
    
        X_all = X_init.copy()
        y_all = y_init.copy()
        best_values = [y_all.max()]
    
        for iteration in range(n_iterations):
            # Build GP model
            kernel = C(1.0) * RBF([1.0, 1.0])
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                          normalize_y=True, random_state=42)
            gp.fit(X_all, y_all)
    
            # Generate candidate points
            X_candidates = np.random.uniform([0, 0], [5, 4], (1000, 2))
    
            # Select next point with acquisition function
            if acquisition_func == 'EI':
                acq_values = expected_improvement(X_candidates, gp, y_all.max())
            elif acquisition_func == 'PI':
                acq_values = probability_of_improvement(X_candidates, gp, y_all.max())
            elif acquisition_func == 'UCB':
                acq_values = upper_confidence_bound(X_candidates, gp, kappa=2.0)
            elif acquisition_func == 'Random':
                acq_values = np.random.rand(len(X_candidates))
    
            next_x = X_candidates[np.argmax(acq_values)]
            next_y = complex_process(next_x.reshape(1, -1))
    
            # Add data
            X_all = np.vstack([X_all, next_x])
            y_all = np.hstack([y_all, next_y])
            best_values.append(y_all.max())
    
        return best_values
    
    # Execute with each acquisition function
    acquisition_functions = ['EI', 'PI', 'UCB', 'Random']
    results = {}
    
    print("Running optimization with each acquisition function...")
    for acq_func in acquisition_functions:
        print(f"  {acq_func}...")
        results[acq_func] = run_bo_with_acquisition(acq_func, n_iterations=20)
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence curves
    for acq_func, best_vals in results.items():
        ax1.plot(best_vals, marker='o', label=acq_func, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Value Found')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Final performance comparison
    final_values = [vals[-1] for vals in results.values()]
    colors = ['green', 'blue', 'red', 'gray']
    ax2.bar(acquisition_functions, final_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Best Value')
    ax2.set_title('Final Performance Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Display values
    for i, (func, val) in enumerate(zip(acquisition_functions, final_values)):
        ax2.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('acquisition_comparison.png', dpi=150, bbox_inches='tight')
    
    # Statistical summary
    print("\n=== Acquisition Function Comparison Results ===")
    print(f"{'Function':<10} {'Final Value':<10} {'Improvement':<10} {'20-iter Convergence Rate'}")
    print("-" * 50)
    for acq_func, best_vals in results.items():
        improvement = best_vals[-1] - best_vals[0]
        convergence = (best_vals[-1] - best_vals[0]) / (max(final_values) - best_vals[0]) * 100
        print(f"{acq_func:<10} {best_vals[-1]:<10.3f} {improvement:<10.3f} {convergence:>6.1f}%")
    
    print("\nRecommended acquisition function:")
    best_acq = max(results.keys(), key=lambda k: results[k][-1])
    print(f"  Best performance: {best_acq}")
    print(f"  Final value: {results[best_acq][-1]:.3f}")
    

####  Insights from Comparison Experiments

  * **EI** : Best balance, recommended for most cases
  * **UCB** : High flexibility with parameter tuning
  * **PI** : Conservative, prone to local optima
  * **Random** : Baseline (for confirming BO superiority)

## 3.9 Practical Guide for Acquisition Function Selection

Acquisition Function | Application Scenario | Advantages | Disadvantages  
---|---|---|---  
**EI** | Default recommendation, multi-purpose | Well-balanced, theoretical foundation | Sensitive to noise  
**PI** | Conservative optimization | Simple implementation, easy interpretation | Tends toward exploitation  
**UCB** | Need to adjust exploration degree | Parameter-controllable | Difficult parameter selection  
**Thompson Sampling** | Parallel experiments, multi-armed bandits | Theoretical optimality, diversity | Sampling cost  
**KG** | Expensive experiments, few samples | Lookahead optimal, efficient | High computational cost  
**Entropy Search** | Optimal solution identification objective | Information-theoretic rigor | Complex implementation  
  
#### =¡ Recommended Practical Strategy

  1. **Start with EI** : Good performance in most cases
  2. **Insufficient exploration** : UCB (increase kappa) or Thompson Sampling
  3. **Slow convergence** : PI (exploitation-focused) or EI (decrease xi)
  4. **Parallel experiments** : Thompson Sampling or batch UCB
  5. **Extremely costly experiments** : Knowledge Gradient

## Summary

In this chapter, we learned about seven acquisition functions, the heart of Bayesian optimization, with implementation examples.

### Key Points

  * **Exploration and Exploitation** : All acquisition functions control this tradeoff
  * **EI is fundamental** : Start with Expected Improvement when uncertain
  * **UCB flexibility** : Explicitly adjust exploration degree with parameters
  * **Thompson Sampling theoretical optimality** : Bayesian optimal, also effective for parallelization
  * **Context-dependent selection** : Changing acquisition functions based on problem nature and progress is also effective

### Preview of Next Chapter

In Chapter 4, we will cover **multi-objective optimization** (balancing quality and efficiency, etc.) and **constrained optimization** (optimization within safe regions) that frequently appear in real processes. We will also learn advanced strategies combining multiple acquisition functions. 

[� Chapter 2: Gaussian Processes](<chapter-2.html>) [Back to Contents](<index.html>) [Chapter 4: Multi-objective and Constrained Optimization ’](<chapter-4.html>)

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
