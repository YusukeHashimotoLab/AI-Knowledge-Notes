---
title: "Chapter 4: Multi-Objective and Constrained Optimization"
chapter_title: "Chapter 4: Multi-Objective and Constrained Optimization"
subtitle: Addressing Complex Real-World Problems
---

This chapter covers Multi. You will learn essential concepts and techniques.

## 4.1 Real-World Complexity

In actual process optimization, simply maximizing a single objective function is insufficient. We need to simultaneously satisfy multiple requirements, including trade-offs between quality and efficiency, safety constraints, and physical constraints. 

#### =¡ Typical Requirements in Real Processes

  * **Multi-objective** : Increase yield ‘ AND reduce energy “ AND reduce cost “
  * **Constraints** : Temperature d 150°C, Pressure d 5 bar, pH 6-8
  * **Feasibility** : Only physically possible conditions
  * **Safety** : Outside explosion limits, toxic substance concentration limits

In this chapter, we will learn extended Bayesian optimization methods that address these complex requirements. 

## 4.2 Pareto Front Exploration

In multi-objective optimization, there is no single solution that is best in all objectives. Instead, we seek a set of Pareto optimal solutions (Pareto front). 

### Example 1: Multi-Objective Bayesian Optimization (Yield vs Energy)
    
    
    # Multi-objective Bayesian Optimization: Pareto Front Exploration
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # Two-objective process function (yield and energy consumption)
    def process_objectives(X):
        """
        Args:
            X: [temperature, pressure] (N, 2)
        Returns:
            yield: Yield (to be maximized)
            energy: Energy consumption (to be minimized)
        """
        temp, pres = X[:, 0], X[:, 1]
    
        # Yield model (function of temperature and pressure)
        yield_rate = (
            50 + 20 * np.sin(temp/20) +
            15 * np.cos(pres/10) -
            0.1 * (temp - 100)**2 -
            0.05 * (pres - 50)**2 +
            np.random.normal(0, 1, len(temp))
        )
    
        # Energy consumption (proportional to temperature and pressure)
        energy = (
            0.5 * temp + 0.3 * pres +
            0.01 * temp * pres +
            np.random.normal(0, 2, len(temp))
        )
    
        return yield_rate, energy
    
    # Pareto dominance check
    def is_pareto_efficient(costs):
        """Pareto optimality check (converted to minimization problem)
    
        Args:
            costs: (N, M) objective function values (minimization direction)
        Returns:
            is_efficient: (N,) bool array
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Points worse than i in all objectives are inefficient
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )
                is_efficient[i] = True
        return is_efficient
    
    # Execute multi-objective Bayesian optimization
    np.random.seed(42)
    n_iterations = 30
    
    # Initial sampling
    X = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1, Y2 = process_objectives(X)
    
    # Recording arrays
    all_X = X.copy()
    all_Y1 = Y1.copy()
    all_Y2 = Y2.copy()
    
    for iteration in range(n_iterations):
        # Build GP model for each objective
        kernel = C(1.0) * RBF([10.0, 10.0])
    
        gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp1.fit(all_X, all_Y1)
    
        gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp2.fit(all_X, all_Y2)
    
        # Generate candidate points
        X_candidates = np.random.uniform([50, 20], [150, 80], (500, 2))
    
        # Prediction at each candidate
        mu1, sigma1 = gp1.predict(X_candidates, return_std=True)
        mu2, sigma2 = gp2.predict(X_candidates, return_std=True)
    
        # Scalarization: weighted sum with uncertainty
        # Maximize yield = minimize -yield
        # Change weight per iteration to explore diverse Pareto solutions
        weight = iteration / n_iterations  # 0’1
        scalarized = (
            -(1 - weight) * (mu1 + 2 * sigma1) +  # Yield (maximize)
            weight * (mu2 - 2 * sigma2)            # Energy (minimize)
        )
    
        # Next experiment point
        next_idx = np.argmin(scalarized)
        next_x = X_candidates[next_idx]
    
        # Execute experiment
        next_y1, next_y2 = process_objectives(next_x.reshape(1, -1))
    
        # Add data
        all_X = np.vstack([all_X, next_x])
        all_Y1 = np.hstack([all_Y1, next_y1])
        all_Y2 = np.hstack([all_Y2, next_y2])
    
    # Extract Pareto front
    costs = np.column_stack([-all_Y1, all_Y2])  # Convert yield to negative (minimization)
    pareto_mask = is_pareto_efficient(costs)
    pareto_X = all_X[pareto_mask]
    pareto_Y1 = all_Y1[pareto_mask]
    pareto_Y2 = all_Y2[pareto_mask]
    
    # Sort (for visualization)
    sorted_idx = np.argsort(pareto_Y1)
    pareto_Y1_sorted = pareto_Y1[sorted_idx]
    pareto_Y2_sorted = pareto_Y2[sorted_idx]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Objective function space
    ax1.scatter(all_Y1, all_Y2, c='lightgray', alpha=0.5, s=30, label='All evaluations')
    ax1.scatter(pareto_Y1, pareto_Y2, c='red', s=100, marker='*',
                label=f'Pareto front ({len(pareto_Y1)} points)', zorder=10)
    ax1.plot(pareto_Y1_sorted, pareto_Y2_sorted, 'r--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Yield (%)', fontsize=12)
    ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax1.set_title('Pareto Front in Objective Space', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Decision variable space
    scatter = ax2.scatter(pareto_X[:, 0], pareto_X[:, 1], c=pareto_Y1,
                          s=150, cmap='RdYlGn', marker='*', edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Pressure (bar)', fontsize=12)
    ax2.set_title('Pareto Solutions in Decision Space', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Yield (%)', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=150, bbox_inches='tight')
    
    print(f"Discovered Pareto optimal solutions: {len(pareto_Y1)} points")
    print(f"\nRepresentative trade-off solutions:")
    print(f"  High yield priority: Yield={pareto_Y1.max():.1f}%, Energy={pareto_Y2[np.argmax(pareto_Y1)]:.1f}kWh")
    print(f"  Low energy priority: Yield={pareto_Y1[np.argmin(pareto_Y2)]:.1f}%, Energy={pareto_Y2.min():.1f}kWh")
    

####  Advantages of Pareto Optimization

  * Visualize trade-offs between multiple objectives
  * Decision makers can select solutions according to preferences
  * Can handle cases where objective importance is unknown in advance

## 4.3 Expected Hypervolume Improvement (EHVI)

An acquisition function specialized for multi-objective optimization. It maximizes the improvement in hypervolume occupied by the Pareto front. 

### Example 2: EHVI Acquisition Function Implementation
    
    
    # Expected Hypervolume Improvement (EHVI)
    from scipy.stats import norm
    
    def compute_hypervolume_2d(pareto_front, ref_point):
        """Calculate 2D hypervolume (simplified version)
    
        Args:
            pareto_front: (N, 2) Pareto optimal solutions (minimization problem)
            ref_point: (2,) reference point
        """
        # Sort
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
    
        hv = 0.0
        prev_x = ref_point[0]
    
        for point in sorted_front:
            width = prev_x - point[0]
            height = ref_point[1] - point[1]
            if width > 0 and height > 0:
                hv += width * height
            prev_x = point[0]
    
        return hv
    
    def expected_hypervolume_improvement(X, gp1, gp2, pareto_front, ref_point, n_samples=100):
        """EHVI acquisition function (Monte Carlo approximation)
    
        Args:
            X: Evaluation points
            gp1, gp2: GP models for each objective
            pareto_front: Current Pareto front
            ref_point: Reference point
            n_samples: Number of samples
        """
        mu1, sigma1 = gp1.predict(X, return_std=True)
        mu2, sigma2 = gp2.predict(X, return_std=True)
    
        # Current hypervolume
        current_hv = compute_hypervolume_2d(pareto_front, ref_point)
    
        ehvi = np.zeros(len(X))
    
        for i in range(len(X)):
            hv_improvements = []
    
            for _ in range(n_samples):
                # Sample objective function values at new point
                y1_new = np.random.normal(mu1[i], sigma1[i])
                y2_new = np.random.normal(mu2[i], sigma2[i])
    
                # Pareto front with new point added
                new_front_candidates = np.vstack([
                    pareto_front,
                    [-y1_new, y2_new]  # Convert yield to negative
                ])
    
                # New Pareto front
                new_pareto_mask = is_pareto_efficient(new_front_candidates)
                new_pareto_front = new_front_candidates[new_pareto_mask]
    
                # Hypervolume improvement
                new_hv = compute_hypervolume_2d(new_pareto_front, ref_point)
                hv_improvements.append(max(0, new_hv - current_hv))
    
            ehvi[i] = np.mean(hv_improvements)
    
        return ehvi
    
    # Multi-objective BO using EHVI
    print("Executing EHVI optimization...")
    
    # Initial setup
    X_init = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1_init, Y2_init = process_objectives(X_init)
    
    X_all = X_init.copy()
    Y1_all = Y1_init.copy()
    Y2_all = Y2_init.copy()
    
    n_iter = 20
    
    for iteration in range(n_iter):
        # Build GP models
        gp1 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp1.fit(X_all, Y1_all)
    
        gp2 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp2.fit(X_all, Y2_all)
    
        # Current Pareto front
        costs = np.column_stack([-Y1_all, Y2_all])
        pareto_mask = is_pareto_efficient(costs)
        current_pareto = costs[pareto_mask]
    
        # Reference point (slightly worse than worst values)
        ref_point = np.array([
            costs[:, 0].max() + 10,
            costs[:, 1].max() + 10
        ])
    
        # Candidate points
        X_candidates = np.random.uniform([50, 20], [150, 80], (100, 2))
    
        # EHVI calculation
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}/{n_iter}")
    
        ehvi = expected_hypervolume_improvement(
            X_candidates, gp1, gp2, current_pareto, ref_point, n_samples=50
        )
    
        # Next experiment point
        next_x = X_candidates[np.argmax(ehvi)]
        next_y1, next_y2 = process_objectives(next_x.reshape(1, -1))
    
        X_all = np.vstack([X_all, next_x])
        Y1_all = np.hstack([Y1_all, next_y1])
        Y2_all = np.hstack([Y2_all, next_y2])
    
    # Final Pareto front
    costs_final = np.column_stack([-Y1_all, Y2_all])
    pareto_mask_final = is_pareto_efficient(costs_final)
    
    print(f"\nEHVI optimization results:")
    print(f"  Number of evaluations: {len(X_all)}")
    print(f"  Pareto optimal solutions: {pareto_mask_final.sum()} points")
    print(f"  Hypervolume: {compute_hypervolume_2d(costs_final[pareto_mask_final], ref_point):.2f}")
    

#### =¡ EHVI Characteristics

  * Considers improvement of entire Pareto front (not just a single point)
  * Theoretically extensible to 3 or more objectives
  * Computational cost is high (approximation methods recommended)

## 4.4 Constrained Optimization

In real processes, physical and safety constraints exist. We will learn methods to optimize while avoiding evaluation at constraint-violating points. 

### Example 3: Constraint Handling with Gaussian Processes
    
    
    # Constrained Bayesian Optimization
    def constrained_process(X):
        """Constrained process
    
        Objective: Maximize reaction rate
        Constraints: Temperature < 130°C and Pressure < 60 bar (safety)
        """
        temp, pres = X[:, 0], X[:, 1]
    
        # Objective function (reaction rate)
        rate = (
            10 + 0.5*temp + 0.3*pres -
            0.002*temp**2 - 0.001*pres**2 +
            0.01*temp*pres +
            np.random.normal(0, 0.5, len(temp))
        )
    
        # Constraint functions (negative means violation)
        # g1: 130 - temp >= 0
        # g2: 60 - pres >= 0
        constraint1 = 130 - temp
        constraint2 = 60 - pres
    
        return rate, constraint1, constraint2
    
    def probability_of_feasibility(X, gp_constraints):
        """Calculate probability of feasibility
    
        Args:
            X: Evaluation points
            gp_constraints: [GP(c1), GP(c2), ...]
        Returns:
            prob: Probability of satisfying all constraints
        """
        prob = np.ones(len(X))
    
        for gp_c in gp_constraints:
            mu_c, sigma_c = gp_c.predict(X, return_std=True)
            # Probability of c >= 0
            prob *= norm.cdf(mu_c / (sigma_c + 1e-6))
    
        return prob
    
    def constrained_ei(X, gp_obj, gp_constraints, y_best, xi=0.01):
        """Constrained Expected Improvement
    
        Args:
            X: Evaluation points
            gp_obj: GP for objective function
            gp_constraints: List of GPs for constraints
            y_best: Best value in feasible region
        """
        # Standard EI
        mu, sigma = gp_obj.predict(X, return_std=True)
        imp = mu - y_best - xi
        Z = imp / (sigma + 1e-6)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    
        # Weight by feasibility probability
        pof = probability_of_feasibility(X, gp_constraints)
    
        return ei * pof
    
    # Execute constrained BO
    np.random.seed(42)
    
    # Initial sampling (including constraint violations)
    X = np.random.uniform([80, 30], [150, 80], (15, 2))
    rate, c1, c2 = constrained_process(X)
    
    X_all = X.copy()
    rate_all = rate.copy()
    c1_all = c1.copy()
    c2_all = c2.copy()
    
    n_iter = 25
    
    for iteration in range(n_iter):
        # Build GPs (objective and constraints)
        gp_obj = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_obj.fit(X_all, rate_all)
    
        gp_c1 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c1.fit(X_all, c1_all)
    
        gp_c2 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c2.fit(X_all, c2_all)
    
        # Best value in feasible region
        feasible_mask = (c1_all >= 0) & (c2_all >= 0)
        if feasible_mask.any():
            y_best = rate_all[feasible_mask].max()
        else:
            y_best = rate_all.min()  # No feasible solution yet
    
        # Candidate points
        X_cand = np.random.uniform([80, 30], [150, 80], (500, 2))
    
        # Constrained EI
        cei = constrained_ei(X_cand, gp_obj, [gp_c1, gp_c2], y_best)
    
        # Next experiment point
        next_x = X_cand[np.argmax(cei)]
        next_rate, next_c1, next_c2 = constrained_process(next_x.reshape(1, -1))
    
        X_all = np.vstack([X_all, next_x])
        rate_all = np.hstack([rate_all, next_rate])
        c1_all = np.hstack([c1_all, next_c1])
        c2_all = np.hstack([c2_all, next_c2])
    
    # Results analysis
    final_feasible = (c1_all >= 0) & (c2_all >= 0)
    feasible_X = X_all[final_feasible]
    feasible_rate = rate_all[final_feasible]
    
    best_idx = np.argmax(feasible_rate)
    best_X = feasible_X[best_idx]
    best_rate = feasible_rate[best_idx]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Search history
    colors = ['red' if not f else 'green' for f in final_feasible]
    ax1.scatter(X_all[:, 0], X_all[:, 1], c=colors, alpha=0.6, s=50)
    ax1.scatter(best_X[0], best_X[1], c='gold', s=300, marker='*',
                edgecolors='black', linewidth=2, label='Best feasible', zorder=10)
    ax1.axvline(130, color='red', linestyle='--', alpha=0.7, label='Temp constraint')
    ax1.axhline(60, color='blue', linestyle='--', alpha=0.7, label='Pressure constraint')
    ax1.fill_between([80, 130], 30, 60, alpha=0.1, color='green', label='Feasible region')
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Pressure (bar)', fontsize=12)
    ax1.set_title('Constrained Optimization History', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Convergence curve
    feasible_best = []
    for i in range(len(rate_all)):
        mask = final_feasible[:i+1]
        if mask.any():
            feasible_best.append(rate_all[:i+1][mask].max())
        else:
            feasible_best.append(np.nan)
    
    ax2.plot(feasible_best, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best Feasible Reaction Rate', fontsize=12)
    ax2.set_title('Convergence in Feasible Region', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_optimization.png', dpi=150, bbox_inches='tight')
    
    print(f"\nConstrained optimization results:")
    print(f"  Total evaluations: {len(X_all)}")
    print(f"  Feasible points: {final_feasible.sum()} ({final_feasible.sum()/len(X_all)*100:.1f}%)")
    print(f"  Best feasible solution:")
    print(f"    Temperature = {best_X[0]:.1f}°C, Pressure = {best_X[1]:.1f} bar")
    print(f"    Reaction rate = {best_rate:.2f}")
    

####  Constraint Handling Strategies

  * **Soft constraints** : Weight by feasibility probability (above example)
  * **Hard constraints** : Exclude constraint-violating points from candidates
  * **Penalty method** : Add penalty term to objective function

## 4.5 Probability of Feasibility

A method that explicitly models the probability of satisfying constraints. 

### Example 4: Probability of Feasibility Implementation
    
    
    # Detailed implementation of Probability of Feasibility (PoF)
    def visualize_feasibility_probability(gp_constraints, X_grid):
        """Visualize feasibility probability"""
        pof = probability_of_feasibility(X_grid, gp_constraints)
        return pof
    
    # Generate grid
    temp_range = np.linspace(80, 150, 100)
    pres_range = np.linspace(30, 80, 100)
    Temp, Pres = np.meshgrid(temp_range, pres_range)
    X_grid = np.column_stack([Temp.ravel(), Pres.ravel()])
    
    # PoF with final model
    gp_c1_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_c1_final.fit(X_all, c1_all)
    
    gp_c2_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_c2_final.fit(X_all, c2_all)
    
    pof_grid = visualize_feasibility_probability([gp_c1_final, gp_c2_final], X_grid)
    PoF = pof_grid.reshape(Temp.shape)
    
    # Objective function prediction
    gp_obj_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_obj_final.fit(X_all, rate_all)
    rate_pred = gp_obj_final.predict(X_grid).reshape(Temp.shape)
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # PoF
    im1 = ax1.contourf(Temp, Pres, PoF, levels=20, cmap='RdYlGn')
    ax1.contour(Temp, Pres, PoF, levels=[0.9, 0.95, 0.99], colors='black',
                linewidths=1.5, linestyles='--')
    ax1.scatter(X_all[:, 0], X_all[:, 1], c='blue', s=30, alpha=0.6, edgecolors='white')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Probability of Feasibility')
    plt.colorbar(im1, ax=ax1, label='PoF')
    
    # Predicted reaction rate
    im2 = ax2.contourf(Temp, Pres, rate_pred, levels=20, cmap='viridis')
    ax2.contour(Temp, Pres, PoF, levels=[0.5], colors='red', linewidths=2, linestyles='-')
    ax2.scatter(best_X[0], best_X[1], c='gold', s=300, marker='*', edgecolors='black', linewidth=2)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Predicted Reaction Rate')
    plt.colorbar(im2, ax=ax2, label='Rate')
    
    # Constrained EI
    cei_grid = constrained_ei(X_grid, gp_obj_final, [gp_c1_final, gp_c2_final],
                              rate_all[final_feasible].max())
    CEI = cei_grid.reshape(Temp.shape)
    
    im3 = ax3.contourf(Temp, Pres, CEI, levels=20, cmap='plasma')
    ax3.scatter(X_all[:, 0], X_all[:, 1], c='white', s=20, alpha=0.8)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Pressure (bar)')
    ax3.set_title('Constrained EI Acquisition')
    plt.colorbar(im3, ax=ax3, label='CEI')
    
    plt.tight_layout()
    plt.savefig('pof_landscape.png', dpi=150, bbox_inches='tight')
    
    print("\nFeasibility probability statistics:")
    print(f"  PoF > 0.5: {(PoF > 0.5).sum() / PoF.size * 100:.1f}%")
    print(f"  PoF > 0.9: {(PoF > 0.9).sum() / PoF.size * 100:.1f}%")
    print(f"  PoF > 0.99: {(PoF > 0.99).sum() / PoF.size * 100:.1f}%")
    

####   Interpretation of PoF

PoF = 0.9 means "90% probability of satisfying constraints". For safety-critical applications, requiring PoF > 0.95 or 0.99 is recommended. 

## 4.6 Safe Bayesian Optimization

Safe BO algorithm that avoids evaluation at unknown constraint-violating points. 

### Example 5: Safe Bayesian Optimization Implementation
    
    
    # Safe Bayesian Optimization
    def safe_ucb(X, gp_obj, gp_constraints, beta_obj=2.0, beta_safe=3.0):
        """Safe UCB acquisition function
    
        Args:
            X: Candidate points
            gp_obj: Objective function GP
            gp_constraints: Constraint GPs
            beta_obj: Exploration parameter for objective
            beta_safe: Confidence parameter for safety
        """
        # UCB for objective function
        mu_obj, sigma_obj = gp_obj.predict(X, return_std=True)
        ucb_obj = mu_obj + beta_obj * sigma_obj
    
        # Safety probability (conservative: mu - beta*sigma >= 0)
        safety_prob = np.ones(len(X))
        for gp_c in gp_constraints:
            mu_c, sigma_c = gp_c.predict(X, return_std=True)
            # Safe if lower confidence bound is positive
            lcb_c = mu_c - beta_safe * sigma_c
            safety_prob *= (lcb_c >= 0).astype(float)
    
        # Select only safe points
        ucb_obj[safety_prob == 0] = -np.inf
    
        return ucb_obj, safety_prob
    
    # Execute Safe BO
    print("\nExecuting Safe Bayesian Optimization...")
    
    # Initial points (starting from known safe region)
    X_safe = np.array([
        [100, 40],
        [105, 45],
        [110, 50],
        [95, 35],
        [90, 40]
    ])
    rate_safe, c1_safe, c2_safe = constrained_process(X_safe)
    
    X_all_safe = X_safe.copy()
    rate_all_safe = rate_safe.copy()
    c1_all_safe = c1_safe.copy()
    c2_all_safe = c2_safe.copy()
    
    safety_violations = 0
    
    for iteration in range(20):
        # Build GPs
        gp_obj_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_obj_s.fit(X_all_safe, rate_all_safe)
    
        gp_c1_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c1_s.fit(X_all_safe, c1_all_safe)
    
        gp_c2_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c2_s.fit(X_all_safe, c2_all_safe)
    
        # Candidate points
        X_cand = np.random.uniform([80, 30], [150, 80], (500, 2))
    
        # Safe UCB
        safe_ucb_values, safety = safe_ucb(X_cand, gp_obj_s, [gp_c1_s, gp_c2_s],
                                            beta_obj=2.0, beta_safe=3.0)
    
        if np.all(np.isinf(safe_ucb_values)):
            print(f"  Warning: Iteration {iteration} - No safe candidate points")
            # Select safest point (emergency measure)
            pof = probability_of_feasibility(X_cand, [gp_c1_s, gp_c2_s])
            next_x = X_cand[np.argmax(pof)]
        else:
            next_x = X_cand[np.argmax(safe_ucb_values)]
    
        # Experiment
        next_rate, next_c1, next_c2 = constrained_process(next_x.reshape(1, -1))
    
        # Safety check
        if next_c1[0] < 0 or next_c2[0] < 0:
            safety_violations += 1
            print(f"    Constraint violation occurred (Iteration {iteration})")
    
        X_all_safe = np.vstack([X_all_safe, next_x])
        rate_all_safe = np.hstack([rate_all_safe, next_rate])
        c1_all_safe = np.hstack([c1_all_safe, next_c1])
        c2_all_safe = np.hstack([c2_all_safe, next_c2])
    
    print(f"\nSafe BO results:")
    print(f"  Total evaluations: {len(X_all_safe)}")
    print(f"  Constraint violations: {safety_violations} times ({safety_violations/len(X_all_safe)*100:.1f}%)")
    
    # Compare with standard BO
    print(f"\nStandard BO (reference):")
    print(f"  Constraint violations: {(~final_feasible).sum()} times ({(~final_feasible).sum()/len(X_all)*100:.1f}%)")
    
    print(f"\nSafety improvement by Safe BO: {((~final_feasible).sum() - safety_violations)} violations reduced")
    

####  Advantages of Safe BO

  * Avoids experiments in dangerous regions with high probability
  * Applicable to safety-critical processes
  * Conservative exploration (adjustable via ² values)

## 4.7 Batch Bayesian Optimization

Batch optimization that proposes multiple points simultaneously to utilize parallel experimental equipment. 

### Example 6: Batch Acquisition Function
    
    
    # Batch Bayesian Optimization
    def batch_ucb_selection(X_candidates, gp, batch_size=5, kappa=2.0, diversity_weight=0.1):
        """Batch UCB selection (sequential greedy method)
    
        Args:
            X_candidates: Candidate points
            gp: GP model
            batch_size: Batch size
            kappa: UCB exploration parameter
            diversity_weight: Diversity weight
        """
        selected_batch = []
        remaining_candidates = X_candidates.copy()
    
        for i in range(batch_size):
            # UCB calculation
            mu, sigma = gp.predict(remaining_candidates, return_std=True)
            ucb = mu + kappa * sigma
    
            # Diversity penalty (favor points far from already selected)
            if len(selected_batch) > 0:
                selected_array = np.array(selected_batch)
                for selected_x in selected_array:
                    distances = np.linalg.norm(remaining_candidates - selected_x, axis=1)
                    diversity_bonus = diversity_weight * distances.min() / distances
                    ucb += diversity_bonus
    
            # Select best point
            best_idx = np.argmax(ucb)
            selected_batch.append(remaining_candidates[best_idx])
    
            # Remove selected point
            remaining_candidates = np.delete(remaining_candidates, best_idx, axis=0)
    
        return np.array(selected_batch)
    
    # Execute batch BO
    print("\nExecuting Batch Bayesian Optimization...")
    
    batch_size = 5
    n_batches = 8
    
    # Initial sampling
    X_batch = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1_batch, Y2_batch = process_objectives(X_batch)
    
    for batch_idx in range(n_batches):
        # Build GP (simplified with only objective 1)
        gp_batch = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_batch.fit(X_batch, Y1_batch)
    
        # Batch selection
        X_cand = np.random.uniform([50, 20], [150, 80], (200, 2))
        next_batch = batch_ucb_selection(X_cand, gp_batch, batch_size=batch_size,
                                         kappa=2.0, diversity_weight=0.5)
    
        # Parallel experiments (simulation)
        next_y1, next_y2 = process_objectives(next_batch)
    
        # Add data
        X_batch = np.vstack([X_batch, next_batch])
        Y1_batch = np.hstack([Y1_batch, next_y1])
        Y2_batch = np.hstack([Y2_batch, next_y2])
    
        print(f"  Batch {batch_idx+1}/{n_batches}: Best yield = {Y1_batch.max():.2f}")
    
    # Results visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by batch
    colors = np.repeat(np.arange(n_batches + 1), [10] + [batch_size]*n_batches)
    scatter = ax1.scatter(X_batch[:, 0], X_batch[:, 1], c=colors, cmap='tab10',
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Batch Optimization Trajectory')
    plt.colorbar(scatter, ax=ax1, label='Batch Number', ticks=range(n_batches+1))
    ax1.grid(alpha=0.3)
    
    # Diversity within batch (last batch)
    last_batch = X_batch[-batch_size:]
    distances = []
    for i in range(len(last_batch)):
        for j in range(i+1, len(last_batch)):
            dist = np.linalg.norm(last_batch[i] - last_batch[j])
            distances.append(dist)
    
    ax2.hist(distances, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Diversity in Last Batch ({batch_size} points)')
    ax2.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances):.2f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_optimization.png', dpi=150, bbox_inches='tight')
    
    print(f"\nBatch optimization results:")
    print(f"  Total evaluations: {len(X_batch)}")
    print(f"  Number of batches: {n_batches}")
    print(f"  Speedup by parallel experiments: {batch_size}x (theoretical)")
    print(f"  Best yield: {Y1_batch.max():.2f}")
    

#### =¡ Batch Selection Strategies

  * **Sequential greedy method** : Select one point at a time, add diversity penalty
  * **Parallel EI** : Kriging Believer approach
  * **Thompson Sampling** : Naturally ensures diversity
  * **Diversity-focused** : Combined with clustering

## 4.8 High-Dimensional Optimization

Strategies for cases with many parameters (10 dimensions or more). 

### Example 7: Addressing High-Dimensional Problems
    
    
    # High-dimensional Bayesian Optimization (dimensionality reduction approach)
    from sklearn.decomposition import PCA
    
    def high_dimensional_process(X):
        """10-dimensional process function
    
        Args:
            X: (N, 10) 10 parameters
        Returns:
            y: Objective function value
        """
        # In reality, optimal solution exists in low-dimensional subspace (common case)
        # Effective dimensions: x0, x1, x5 only
        effective_dims = X[:, [0, 1, 5]]
    
        y = (
            -np.sum((effective_dims - 0.5)**2, axis=1) +
            0.1 * np.sum(X, axis=1) +  # Small influence from other dimensions
            np.random.normal(0, 0.01, len(X))
        )
        return y
    
    # High-dimensional BO strategy 1: Random embedding
    print("\nExecuting high-dimensional BO...")
    
    n_dim = 10
    n_effective_dim = 3  # Estimated effective dimensions
    
    # Initial sampling
    n_init = 20  # More for high dimensions
    X_high = np.random.uniform(0, 1, (n_init, n_dim))
    y_high = high_dimensional_process(X_high)
    
    X_all_high = X_high.copy()
    y_all_high = y_high.copy()
    
    # Dimensionality reduction with PCA
    pca = PCA(n_components=n_effective_dim)
    X_reduced = pca.fit_transform(X_all_high)
    
    best_values_high = [y_all_high.max()]
    
    for iteration in range(30):
        # Build GP in low-dimensional space
        gp_high = GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0]*n_effective_dim),
            normalize_y=True,
            n_restarts_optimizer=5
        )
        gp_high.fit(X_reduced, y_all_high)
    
        # Generate candidates in low-dimensional space
        X_cand_reduced = np.random.uniform(
            X_reduced.min(axis=0),
            X_reduced.max(axis=0),
            (500, n_effective_dim)
        )
    
        # Select with UCB
        mu, sigma = gp_high.predict(X_cand_reduced, return_std=True)
        ucb = mu + 2.0 * sigma
        next_x_reduced = X_cand_reduced[np.argmax(ucb)]
    
        # Inverse transform to original space
        next_x_high = pca.inverse_transform(next_x_reduced.reshape(1, -1))
    
        # Range check and clip
        next_x_high = np.clip(next_x_high, 0, 1)
    
        # Experiment
        next_y_high = high_dimensional_process(next_x_high)
    
        # Add data
        X_all_high = np.vstack([X_all_high, next_x_high])
        y_all_high = np.hstack([y_all_high, next_y_high])
    
        # Update PCA
        X_reduced = pca.fit_transform(X_all_high)
    
        best_values_high.append(y_all_high.max())
    
    # Comparison: Random search
    X_random = np.random.uniform(0, 1, (len(y_all_high), n_dim))
    y_random = high_dimensional_process(X_random)
    best_random = [y_random[:i+1].max() for i in range(len(y_random))]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convergence comparison
    ax1.plot(best_values_high, 'g-', linewidth=2, marker='o', markersize=4,
             label='BO with PCA (3D)')
    ax1.plot(best_random, 'gray', linewidth=2, marker='s', markersize=4,
             alpha=0.6, label='Random Search')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Value Found')
    ax1.set_title('High-Dimensional Optimization (10D)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PCA explained variance
    pca_final = PCA(n_components=n_dim)
    pca_final.fit(X_all_high)
    explained_var = pca_final.explained_variance_ratio_
    
    ax2.bar(range(1, n_dim+1), explained_var, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.plot(range(1, n_dim+1), np.cumsum(explained_var), 'ro-', linewidth=2, markersize=6,
             label='Cumulative')
    ax2.axhline(0.95, color='green', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Analysis of Search Space')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('high_dimensional_bo.png', dpi=150, bbox_inches='tight')
    
    print(f"\nHigh-dimensional BO results:")
    print(f"  Dimensions: {n_dim}")
    print(f"  Effective dimensions (estimated): {n_effective_dim}")
    print(f"  Best value (BO+PCA): {y_all_high.max():.4f}")
    print(f"  Best value (Random): {y_random.max():.4f}")
    print(f"  Improvement rate: {(y_all_high.max() - y_random.max())/abs(y_random.max())*100:.1f}%")
    print(f"\nExplained variance of top 3 PCA components: {explained_var[:3].sum()*100:.1f}%")
    

####   Challenges in High-Dimensional BO

As dimensions increase, GP computational cost increases rapidly (O(n³)), and sample efficiency decreases. Countermeasures: (1) Dimensionality reduction (PCA, UMAP), (2) Random Embedding, (3) Additive GP, (4) Sparse approximation 

Dimensions | Recommended Method | Initial Samples | Considerations  
---|---|---|---  
1-3D | Standard BO (EI, UCB) | 5-10 | No issues  
4-7D | Standard BO + good kernel | 10-20 | Increased computational cost  
8-15D | Dimensionality reduction + BO | 20-50 | Estimating effective dimensions is important  
16D+ | Random Embedding/TuRBO | 50-100 | Scalability required  
  
## Summary

In this chapter, we learned advanced methods for addressing complex real-world optimization problems.

### Key Points

  * **Multi-objective optimization** : Pareto front exploration, EHVI for simultaneous achievement of multiple objectives
  * **Constrained optimization** : Safe experimental design with PoF and CEI
  * **Safe BO** : Optimization while avoiding dangerous regions
  * **Batch optimization** : Speedup with parallel experiments (diversity preservation is key)
  * **High-dimensional handling** : Practical application with dimensionality reduction and Random Embedding

### Preview of Next Chapter

In Chapter 5, we will integrate the techniques learned so far and provide detailed explanations of applications to **actual industrial processes**. We will develop practical skills through 7 case studies including reactor optimization, catalyst design, and quality control. 

[� Chapter 3: Acquisition Functions](<chapter-3.html>) [Back to Table of Contents](<index.html>) [Chapter 5: Industrial Applications ’](<chapter-5.html>)

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
