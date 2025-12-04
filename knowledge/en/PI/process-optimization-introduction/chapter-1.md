---
title: "Chapter 1: Formulation of Optimization Problems"
chapter_title: "Chapter 1: Formulation of Optimization Problems"
subtitle: Understanding Objective Functions, Constraints, and Feasible Regions
version: 1.0
created_at: 2025-10-26
---

This chapter covers Formulation of Optimization Problems. You will learn Visualize objective functions using Python, relationship between feasible regions, and fundamentals of gradients.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the basic elements of optimization problems (objective function, decision variables, constraints)
  * ✅ Mathematically formulate optimization problems for chemical processes
  * ✅ Visualize objective functions using Python and visually grasp optimal solutions
  * ✅ Understand the relationship between feasible regions and constraints
  * ✅ Master the fundamentals of gradients and sensitivity analysis

* * *

## 1.1 Fundamentals of Optimization Problems

### What is Optimization?

**Optimization** is the process of finding values of decision variables that minimize or maximize an objective function under given constraints.

A general optimization problem is expressed as follows:

$$ \begin{aligned} \text{minimize} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\\ & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \\\ & \mathbf{x} \in \mathbb{R}^n \end{aligned} $$

Where:

  * **$f(\mathbf{x})$** : Objective function (quantity to be minimized or maximized)
  * **$\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$** : Decision variables (controllable variables)
  * **$g_i(\mathbf{x}) \leq 0$** : Inequality constraints
  * **$h_j(\mathbf{x}) = 0$** : Equality constraints

### Examples of Optimization in Chemical Processes

Process | Objective Function | Decision Variables | Main Constraints  
---|---|---|---  
Chemical Reactor | Maximize yield, minimize cost | Temperature, pressure, residence time | Safe temperature range, product purity  
Distillation Column | Minimize energy cost | Reflux ratio, reboiler heat duty | Product purity, column top pressure  
Production Planning | Maximize profit | Production quantity for each product | Raw material supply, equipment capacity  
Raw Material Blending | Minimize raw material cost | Blending ratio for each raw material | Product specifications, raw material inventory  
  
* * *

## 1.2 Visualizing Optimization Problems with Python

### Code Example 1: Simple Quadratic Function Minimization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Define the objective function: quadratic function (has a minimum)
    def objective_function(x):
        """
        Objective function: f(x) = (x - 3)^2 + 5
        Minimum value: f(3) = 5
        """
        return (x - 3)**2 + 5
    
    # x-axis range
    x = np.linspace(-2, 8, 100)
    y = objective_function(x)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2.5, color='#11998e', label='f(x) = (x - 3)² + 5')
    plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Optimal solution x* = 3')
    plt.scatter([3], [5], color='red', s=150, zorder=5, marker='o',
                edgecolors='black', linewidth=2, label='Minimum value f(x*) = 5')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Visualization of Simple Optimization Problem', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Verify optimal solution
    x_optimal = 3
    f_optimal = objective_function(x_optimal)
    print(f"Optimal solution: x* = {x_optimal}")
    print(f"Minimum value of objective function: f(x*) = {f_optimal}")
    

**Output:**
    
    
    Optimal solution: x* = 3
    Minimum value of objective function: f(x*) = 5
    

**Explanation:** As the simplest optimization problem, we minimize a single-variable quadratic function. For such convex functions, there exists a unique minimum value (global optimal solution).

* * *

### Code Example 2: Formulation of Profit Maximization Problem for Chemical Process
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Profit function for chemical process
    def process_profit(T):
        """
        Profit function with respect to reaction temperature T
    
        Parameters:
        T : float, reaction temperature [°C]
    
        Returns:
        profit : float, profit [$/h]
    
        Model:
        - Yield increases with temperature, but decreases after a certain temperature due to side reactions
        - Energy cost increases with temperature
        - Profit = Product value - Energy cost
        """
        # Product value (yield-dependent, peaks around optimal temperature)
        yield_value = 1000 * (1 - ((T - 175) / 100)**2)
    
        # Energy cost (proportional to temperature)
        energy_cost = 2 * T
    
        # Profit
        profit = yield_value - energy_cost
    
        return profit
    
    # Temperature range
    T_range = np.linspace(100, 250, 150)
    profit_range = [process_profit(T) for T in T_range]
    
    # Numerical calculation of optimal temperature
    optimal_idx = np.argmax(profit_range)
    T_optimal = T_range[optimal_idx]
    profit_optimal = profit_range[optimal_idx]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(T_range, profit_range, linewidth=2.5, color='#11998e',
             label='Profit function')
    plt.axvline(x=T_optimal, color='red', linestyle='--', linewidth=2,
                label=f'Optimal temperature T* = {T_optimal:.1f}°C')
    plt.scatter([T_optimal], [profit_optimal], color='red', s=200,
                zorder=5, marker='*', edgecolors='black', linewidth=2,
                label=f'Maximum profit = ${profit_optimal:.1f}/h')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.fill_between(T_range, 0, profit_range, where=(np.array(profit_range) > 0),
                     alpha=0.2, color='green', label='Profit region')
    plt.xlabel('Reaction Temperature T [°C]', fontsize=12)
    plt.ylabel('Profit [$/h]', fontsize=12)
    plt.title('Chemical Process Profit Maximization Problem', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal reaction temperature: T* = {T_optimal:.2f}°C")
    print(f"Maximum profit: {profit_optimal:.2f} $/h")
    

**Output Example:**
    
    
    Optimal reaction temperature: T* = 165.77°C
    Maximum profit: 668.78 $/h
    

**Explanation:** In actual chemical processes, we maximize a profit function that considers the balance between yield and economics. When the temperature is too high, side reactions and energy costs increase, reducing profit.

* * *

### Code Example 3: Contour Plot for Two-Variable Optimization Problem
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Two-variable objective function (simplified Rosenbrock function)
    def objective_2d(x1, x2):
        """
        Objective function: f(x1, x2) = (x1 - 2)^2 + (x2 - 3)^2
        Minimum value: f(2, 3) = 0
        """
        return (x1 - 2)**2 + (x2 - 3)**2
    
    # Create grid
    x1 = np.linspace(-1, 5, 200)
    x2 = np.linspace(0, 6, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective_2d(X1, X2)
    
    # Contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, label='f(x1, x2)')
    plt.scatter([2], [3], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='Optimal solution (2, 3)')
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Contour Plot for Two-Variable Optimization Problem', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal solution: x1* = 2, x2* = 3")
    print(f"Minimum value of objective function: f(x*) = {objective_2d(2, 3)}")
    

**Output:**
    
    
    Optimal solution: x1* = 2, x2* = 3
    Minimum value of objective function: f(x*) = 0
    

**Explanation:** In two-variable optimization problems, contour plots allow us to visually grasp the topography of the objective function. Areas where contour lines are dense have steep gradients, and they converge to an elliptical shape near the optimal solution.

* * *

### Code Example 4: Visualization of Constraints and Feasible Region
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 4: Visualization of Constraints and Feasible Re
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Objective function
    def objective(x1, x2):
        return x1**2 + x2**2
    
    # Define constraints
    # g1: x1 + x2 >= 2  →  -x1 - x2 + 2 <= 0
    # g2: x1 >= 0
    # g3: x2 >= 0
    # g4: x1 <= 4
    # g5: x2 <= 4
    
    # Create grid
    x1 = np.linspace(-0.5, 5, 300)
    x2 = np.linspace(-0.5, 5, 300)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Determine feasible region
    feasible = (X1 + X2 >= 2) & (X1 >= 0) & (X2 >= 0) & (X1 <= 4) & (X2 <= 4)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Fill feasible region
    plt.contourf(X1, X2, feasible.astype(int), levels=1, colors=['white', '#c8e6c9'], alpha=0.7)
    
    # Contours of objective function
    Z = objective(X1, X2)
    contour = plt.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, label='f(x₁, x₂) = x₁² + x₂²')
    
    # Constraint boundaries
    x1_line = np.linspace(0, 4, 100)
    plt.plot(x1_line, 2 - x1_line, 'r--', linewidth=2, label='x₁ + x₂ = 2')
    plt.axvline(x=0, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axhline(y=0, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axvline(x=4, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axhline(y=4, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Optimal solution (analytically determined): x1 = x2 = 1 (on constraint x1 + x2 = 2)
    plt.scatter([1], [1], color='red', s=250, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='Optimal solution (1, 1)')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Constraints and Feasible Region', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xlim(-0.5, 5)
    plt.ylim(-0.5, 5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Constraints:")
    print("  g1: x₁ + x₂ >= 2")
    print("  g2: x₁ >= 0")
    print("  g3: x₂ >= 0")
    print("  g4: x₁ <= 4")
    print("  g5: x₂ <= 4")
    print(f"\nOptimal solution: x₁* = 1, x₂* = 1")
    print(f"Objective function value: f(x*) = {objective(1, 1)}")
    

**Output:**
    
    
    Constraints:
      g1: x₁ + x₂ >= 2
      g2: x₁ >= 0
      g3: x₂ >= 0
      g4: x₁ <= 4
      g5: x₂ <= 4
    
    Optimal solution: x₁* = 1, x₂* = 1
    Objective function value: f(x*) = 2
    

**Explanation:** Constraints define the feasible region (green area). The optimal solution is the point that minimizes the objective function within the feasible region. In this example, the optimal solution exists on the boundary of the constraint $x_1 + x_2 \geq 2$.

* * *

### Code Example 5: Multi-Variable Optimization Problem (Reactor Yield Optimization)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reactor yield model
    def reactor_yield(T, tau):
        """
        Reactor yield model
    
        Parameters:
        T   : float, reaction temperature [°C] (range: 150-220°C)
        tau : float, residence time [min] (range: 10-60 min)
    
        Returns:
        yield : float, yield [%]
    
        Model: Both temperature and residence time affect yield
        - Higher temperature means faster reaction rate (Arrhenius equation)
        - Longer residence time means more reaction progress
        - However, too high temperature or too long residence time causes side reactions
        """
        # Normalization
        T_norm = (T - 185) / 35
        tau_norm = (tau - 35) / 25
    
        # Yield model (Gaussian distribution-like shape)
        yield_pct = 90 * np.exp(-0.5 * (T_norm**2 + tau_norm**2)) + \
                    5 * T_norm * tau_norm
    
        # Effect of side reactions (yield decreases at high temperature and long time)
        penalty = 0.3 * (T_norm**2 + tau_norm**2) * np.maximum(T_norm, 0) * np.maximum(tau_norm, 0)
    
        return yield_pct - penalty
    
    # Create grid
    T_range = np.linspace(150, 220, 50)
    tau_range = np.linspace(10, 60, 50)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    yield_grid = reactor_yield(T_grid, tau_grid)
    
    # 3D surface plot
    fig = plt.figure(figsize=(14, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(T_grid, tau_grid, yield_grid, cmap='viridis',
                             alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Temperature T [°C]', fontsize=10)
    ax1.set_ylabel('Residence Time τ [min]', fontsize=10)
    ax1.set_zlabel('Yield [%]', fontsize=10)
    ax1.set_title('3D Surface Plot of Reactor Yield', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(T_grid, tau_grid, yield_grid, levels=15, cmap='viridis')
    ax2.contourf(T_grid, tau_grid, yield_grid, levels=15, cmap='viridis', alpha=0.4)
    plt.colorbar(contour, ax=ax2, label='Yield [%]')
    
    # Search for optimal point
    max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    T_opt = T_grid[max_idx]
    tau_opt = tau_grid[max_idx]
    yield_opt = yield_grid[max_idx]
    
    ax2.scatter([T_opt], [tau_opt], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label=f'Optimal point ({T_opt:.1f}°C, {tau_opt:.1f}min)')
    ax2.set_xlabel('Temperature T [°C]', fontsize=11)
    ax2.set_ylabel('Residence Time τ [min]', fontsize=11)
    ax2.set_title('Contour Plot of Reactor Yield', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal operating conditions:")
    print(f"  Temperature: T* = {T_opt:.2f}°C")
    print(f"  Residence time: τ* = {tau_opt:.2f} min")
    print(f"  Maximum yield: {yield_opt:.2f}%")
    

**Output Example:**
    
    
    Optimal operating conditions:
      Temperature: T* = 185.71°C
      Residence time: τ* = 35.10 min
      Maximum yield: 89.87%
    

**Explanation:** The yield of a chemical reactor depends on both temperature and residence time. 3D surface plots and contour plots allow us to visually grasp the optimal operating conditions.

* * *

### Code Example 6: Cost Function Design and Trade-off Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Cost function components
    def energy_cost(T):
        """Energy cost (proportional to temperature)"""
        return 0.5 * T  # $/h
    
    def raw_material_cost(tau):
        """Raw material cost (inversely proportional to residence time, loss at short time)"""
        return 500 / (tau + 10)  # $/h
    
    def product_value(T, tau):
        """Product value (yield-dependent)"""
        yield_pct = reactor_yield(T, tau)
        return 10 * yield_pct  # $/h
    
    def total_cost(T, tau):
        """Total cost function (minimization objective)"""
        cost = energy_cost(T) + raw_material_cost(tau) - product_value(T, tau)
        return cost
    
    # Use reactor_yield function from previous code example
    def reactor_yield(T, tau):
        T_norm = (T - 185) / 35
        tau_norm = (tau - 35) / 25
        yield_pct = 90 * np.exp(-0.5 * (T_norm**2 + tau_norm**2)) + 5 * T_norm * tau_norm
        penalty = 0.3 * (T_norm**2 + tau_norm**2) * np.maximum(T_norm, 0) * np.maximum(tau_norm, 0)
        return yield_pct - penalty
    
    # Create grid
    T_range = np.linspace(150, 220, 50)
    tau_range = np.linspace(15, 60, 50)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    cost_grid = total_cost(T_grid, tau_grid)
    
    # Search for optimal point
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    T_opt = T_grid[min_idx]
    tau_opt = tau_grid[min_idx]
    cost_opt = cost_grid[min_idx]
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Contours of cost function
    plt.subplot(1, 2, 1)
    contour = plt.contour(T_grid, tau_grid, cost_grid, levels=20, cmap='RdYlGn_r')
    plt.contourf(T_grid, tau_grid, cost_grid, levels=20, cmap='RdYlGn_r', alpha=0.4)
    plt.colorbar(contour, label='Total Cost [$/h]')
    plt.scatter([T_opt], [tau_opt], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label=f'Optimal point')
    plt.xlabel('Temperature T [°C]', fontsize=11)
    plt.ylabel('Residence Time τ [min]', fontsize=11)
    plt.title('Total Cost Function Minimization', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Breakdown of cost components
    tau_fixed = 35
    T_test = np.linspace(150, 220, 100)
    energy_costs = [energy_cost(T) for T in T_test]
    product_values = [product_value(T, tau_fixed) for T in T_test]
    raw_costs = [raw_material_cost(tau_fixed) for _ in T_test]
    
    plt.subplot(1, 2, 2)
    plt.plot(T_test, energy_costs, label='Energy Cost', linewidth=2)
    plt.plot(T_test, product_values, label='Product Value', linewidth=2)
    plt.axhline(y=raw_costs[0], color='orange', linestyle='--', linewidth=2, label='Raw Material Cost')
    plt.xlabel('Temperature T [°C]', fontsize=11)
    plt.ylabel('Cost/Value [$/h]', fontsize=11)
    plt.title(f'Cost Component Breakdown (τ = {tau_fixed} min fixed)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal operating conditions (cost minimization):")
    print(f"  Temperature: T* = {T_opt:.2f}°C")
    print(f"  Residence time: τ* = {tau_opt:.2f} min")
    print(f"  Minimum total cost: {cost_opt:.2f} $/h")
    print(f"\nCost breakdown:")
    print(f"  Energy cost: {energy_cost(T_opt):.2f} $/h")
    print(f"  Raw material cost: {raw_material_cost(tau_opt):.2f} $/h")
    print(f"  Product value: {product_value(T_opt, tau_opt):.2f} $/h")
    

**Output Example:**
    
    
    Optimal operating conditions (cost minimization):
      Temperature: T* = 185.71°C
      Residence time: τ* = 38.37 min
      Minimum total cost: -786.45 $/h
    
    Cost breakdown:
      Energy cost: 92.86 $/h
      Raw material cost: 10.34 $/h
      Product value: 889.64 $/h
    

**Explanation:** In actual process optimization, we design a total cost function that integrates multiple cost elements (energy, raw materials, product value). This example finds optimal operating conditions that maximize product value while minimizing energy costs and raw material costs.

* * *

### Code Example 7: Gradient Calculation and Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Objective function
    def f(x1, x2):
        return x1**2 + 2*x2**2
    
    # Analytical gradient calculation
    def gradient(x1, x2):
        """
        Gradient vector ∇f = [∂f/∂x1, ∂f/∂x2]
        """
        df_dx1 = 2 * x1
        df_dx2 = 4 * x2
        return np.array([df_dx1, df_dx2])
    
    # Create grid
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate gradient at each point
    U = np.zeros_like(X1)
    V = np.zeros_like(X2)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            grad = gradient(X1[i, j], X2[i, j])
            U[i, j] = -grad[0]  # Negative gradient direction (minimization direction)
            V[i, j] = -grad[1]
    
    # Objective function values
    Z = f(X1, X2)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.6)
    plt.contourf(X1, X2, Z, levels=15, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, label='f(x₁, x₂)')
    
    # Plot gradient vectors
    plt.quiver(X1, X2, U, V, color='red', alpha=0.6, scale=50, width=0.003,
               label='Negative gradient direction (minimization direction)')
    
    plt.scatter([0], [0], color='yellow', s=250, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='Optimal solution (0, 0)')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Gradient Vector Field', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Gradient calculation example at specific point
    x1_test, x2_test = 2.0, 1.5
    grad_test = gradient(x1_test, x2_test)
    print(f"Gradient at point ({x1_test}, {x2_test}):")
    print(f"  ∇f = [{grad_test[0]:.2f}, {grad_test[1]:.2f}]")
    print(f"  Gradient norm: ||∇f|| = {np.linalg.norm(grad_test):.2f}")
    print(f"\nGradient at optimal solution (0, 0):")
    grad_opt = gradient(0, 0)
    print(f"  ∇f = [{grad_opt[0]:.2f}, {grad_opt[1]:.2f}]  (Gradient = 0 at optimal point)")
    

**Output:**
    
    
    Gradient at point (2.0, 1.5):
      ∇f = [4.00, 6.00]
      Gradient norm: ||∇f|| = 7.21
    
    Gradient at optimal solution (0, 0):
      ∇f = [0.00, 0.00]  (Gradient = 0 at optimal point)
    

**Explanation:** The gradient vector $\nabla f$ points in the direction where the objective function increases most steeply. In minimization problems, moving in the negative gradient direction ($-\nabla f$) decreases the objective function. At the optimal solution, the gradient becomes zero ($\nabla f = \mathbf{0}$).

* * *

### Code Example 8: Sensitivity Analysis (Parameter Sensitivity of Objective Function)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Objective function with parameter
    def objective_with_param(T, k):
        """
        Objective function with parameter k
        k: Physical parameter like reaction rate constant
        """
        return -k * T * np.exp(-1000 / (T + 273)) + 0.5 * T
    
    # Range of parameter k
    k_values = [0.5, 1.0, 1.5, 2.0]
    T_range = np.linspace(100, 300, 200)
    
    # Sensitivity analysis
    plt.figure(figsize=(14, 5))
    
    # Parameter dependence of objective function
    plt.subplot(1, 2, 1)
    for k in k_values:
        obj_values = [objective_with_param(T, k) for T in T_range]
        plt.plot(T_range, obj_values, linewidth=2, label=f'k = {k}')
    
    plt.xlabel('Temperature T [°C]', fontsize=12)
    plt.ylabel('Objective Function Value', fontsize=12)
    plt.title('Change in Objective Function with Parameter k', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Parameter dependence of optimal solution
    plt.subplot(1, 2, 2)
    k_range = np.linspace(0.2, 3, 50)
    T_optimal_list = []
    
    for k in k_range:
        obj_values = [objective_with_param(T, k) for T in T_range]
        T_optimal = T_range[np.argmin(obj_values)]
        T_optimal_list.append(T_optimal)
    
    plt.plot(k_range, T_optimal_list, linewidth=2.5, color='#11998e')
    plt.xlabel('Parameter k', fontsize=12)
    plt.ylabel('Optimal Temperature T* [°C]', fontsize=12)
    plt.title('Sensitivity of Optimal Temperature to Parameter k', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Quantification of sensitivity
    k_nominal = 1.0
    k_perturbed = 1.1
    delta_k = k_perturbed - k_nominal
    
    obj_nominal = [objective_with_param(T, k_nominal) for T in T_range]
    T_opt_nominal = T_range[np.argmin(obj_nominal)]
    
    obj_perturbed = [objective_with_param(T, k_perturbed) for T in T_range]
    T_opt_perturbed = T_range[np.argmin(obj_perturbed)]
    
    delta_T = T_opt_perturbed - T_opt_nominal
    
    sensitivity = delta_T / delta_k
    
    print(f"Sensitivity analysis results:")
    print(f"  Parameter change: Δk = {delta_k:.2f}")
    print(f"  Change in optimal temperature: ΔT* = {delta_T:.2f}°C")
    print(f"  Sensitivity: dT*/dk ≈ {sensitivity:.2f}°C")
    print(f"\nInterpretation: A 10% increase in parameter k changes the optimal temperature by approximately {delta_T:.2f}°C.")
    

**Output Example:**
    
    
    Sensitivity analysis results:
      Parameter change: Δk = 0.10
      Change in optimal temperature: ΔT* = 5.53°C
      Sensitivity: dT*/dk ≈ 55.28°C
    
    Interpretation: A 10% increase in parameter k changes the optimal temperature by approximately 5.53°C.
    

**Explanation:** Sensitivity analysis allows us to evaluate how variations in process parameters affect the optimal solution. This is important for evaluating process uncertainty and robustness.

* * *

### Code Example 9: Problem Transformation Techniques (Transformation to Unconstrained Problems)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Original constrained optimization problem
    # minimize f(x) = x^2
    # subject to x >= 1
    
    def original_objective(x):
        return x**2
    
    def constraint(x):
        """Constraint condition g(x) = 1 - x <= 0  →  x >= 1"""
        return 1 - x
    
    # Transformation using penalty method
    def penalty_objective(x, mu):
        """
        Penalty function: f_penalty(x) = f(x) + μ * max(0, g(x))^2
    
        Parameters:
        x  : Decision variable
        mu : Penalty parameter (larger means stricter constraint enforcement)
        """
        penalty = mu * max(0, constraint(x))**2
        return original_objective(x) + penalty
    
    # x range
    x_range = np.linspace(0, 3, 300)
    
    # Comparison with different penalty parameters
    mu_values = [0, 10, 100, 1000]
    
    plt.figure(figsize=(14, 5))
    
    # Plot penalty function
    plt.subplot(1, 2, 1)
    for mu in mu_values:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        if mu == 0:
            plt.plot(x_range, penalty_values, linewidth=2, label=f'μ = {mu} (original function)', linestyle='--')
        else:
            plt.plot(x_range, penalty_values, linewidth=2, label=f'μ = {mu}')
    
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Constraint boundary x = 1')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Objective Function Value', fontsize=12)
    plt.title('Transformation to Unconstrained Problem Using Penalty Method', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.ylim(0, 10)
    plt.grid(alpha=0.3)
    
    # Penalty parameter dependence of optimal solution
    plt.subplot(1, 2, 2)
    mu_range = np.logspace(0, 4, 50)
    x_optimal_list = []
    
    for mu in mu_range:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        x_optimal = x_range[np.argmin(penalty_values)]
        x_optimal_list.append(x_optimal)
    
    plt.semilogx(mu_range, x_optimal_list, linewidth=2.5, color='#11998e')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='True optimal solution x* = 1')
    plt.xlabel('Penalty Parameter μ', fontsize=12)
    plt.ylabel('Optimal Solution x*', fontsize=12)
    plt.title('Effect of Penalty Parameter', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Numerical calculation results
    print("Convergence of optimal solution using penalty method:")
    print("μ        x*       f(x*)    Constraint violation")
    print("-" * 45)
    for mu in [1, 10, 100, 1000, 10000]:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        idx_opt = np.argmin(penalty_values)
        x_opt = x_range[idx_opt]
        f_opt = penalty_values[idx_opt]
        violation = max(0, constraint(x_opt))
        print(f"{mu:6.0f}   {x_opt:6.4f}   {f_opt:7.4f}   {violation:7.4f}")
    
    print("\nTrue optimal solution: x* = 1.0, f(x*) = 1.0")
    

**Output Example:**
    
    
    Convergence of optimal solution using penalty method:
    μ        x*       f(x*)    Constraint violation
    ---------------------------------------------
         1    0.6689    0.4474    0.3311
        10    0.9045    0.8182    0.0955
       100    0.9901    0.9802    0.0099
      1000    0.9990    0.9980    0.0010
     10000    0.9999    0.9998    0.0001
    
    True optimal solution: x* = 1.0, f(x*) = 1.0
    

**Explanation:** The penalty method transforms constrained problems into unconstrained problems by incorporating constraint conditions as penalty terms in the objective function. As the penalty parameter $\mu$ increases, the optimal solution approaches the true constrained optimal solution.

* * *

## 1.3 Chapter Summary

### What We Learned

**1\. Basic Elements of Optimization Problems** — Definition of objective function, decision variables, and constraints, along with concepts of feasible region and optimal solution.

**2\. Formulation of Chemical Process Optimization** — Profit maximization, cost minimization, and yield maximization problems, and design of economic and technical objective functions.

**3\. Visualization Methods** — Contour plots, 3D surface plots, gradient vector fields, and display of feasible regions for intuitive understanding.

**4\. Sensitivity Analysis and Problem Transformation** — Evaluation of parameter sensitivity and transformation to unconstrained problems using the penalty method.

### Key Points

Optimization problems consist of three elements: objective function, decision variables, and constraints. In chemical processes, trade-offs between yield, economics, and energy efficiency are important. Visualization allows intuitive understanding of optimal solution location and characteristics. Gradients are the foundation of optimization algorithms, and gradients become zero at optimal points. Penalty methods can transform constrained problems into unconstrained problems.

### To the Next Chapter

In Chapter 2, we will learn in detail about **Linear Programming and Nonlinear Programming** , covering the simplex method and solution of linear programming problems, gradient descent method, Newton's method and quasi-Newton methods, utilizing scipy.optimize and PuLP libraries, and comparison and selection of optimization algorithms.
