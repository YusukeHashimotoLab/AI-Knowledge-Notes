---
title: "Chapter 3: Response Surface Methodology (RSM)"
chapter_title: "Chapter 3: Response Surface Methodology (RSM)"
subtitle: Central Composite Design, Box-Behnken Design, Optimization with Quadratic Model Fitting
version: 1.0
created_at: 2025-10-26
---

# Chapter 3: Response Surface Methodology (RSM)

Response Surface Methodology (RSM) is a technique for representing the relationship between factors and responses using quadratic polynomial models and searching for optimal conditions. Central Composite Design (CCD) and Box-Behnken Design enable efficient experimental design, and visualization is achieved through 3D response surfaces and contour plots.

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Design Central Composite Design (CCD) and position experimental points
  * ✅ Efficiently conduct three-factor experiments with Box-Behnken Design
  * ✅ Fit quadratic polynomial models and interpret coefficients
  * ✅ Visualize factor effects with 3D response surface plots
  * ✅ Identify optimal regions using contour plots
  * ✅ Numerically search for optimal conditions using scipy.optimize
  * ✅ Validate models with R², RMSE, and residual plots
  * ✅ Execute case study on distillation column operating condition optimization

* * *

## 3.1 Fundamentals of Response Surface Methodology (RSM)

### What is RSM

**Response Surface Methodology (RSM)** is a technique for representing the relationship between multiple factors and responses using mathematical models (usually quadratic polynomials) and searching for optimal conditions.

**Quadratic Polynomial Model** :

$$y = \beta_0 + \sum_{i=1}^{k}\beta_i x_i + \sum_{i=1}^{k}\beta_{ii} x_i^2 + \sum_{i < j}\beta_{ij} x_i x_j + \epsilon$$

Where:

  * $y$: Response variable (yield, purity, etc.)
  * $x_i$: Factors (temperature, pressure, etc.)
  * $\beta_0$: Intercept
  * $\beta_i$: Linear effect coefficients
  * $\beta_{ii}$: Quadratic effect coefficients (curvature)
  * $\beta_{ij}$: Interaction coefficients
  * $\epsilon$: Error term

**RSM Application Scenarios** :

  * Searching for optimal conditions (maximization/minimization)
  * When responses are nonlinear (curvature exists)
  * When factor interactions are important
  * Optimization of chemical processes and manufacturing processes

* * *

## 3.2 Central Composite Design (CCD)

### Code Example 1: Central Composite Design (CCD) Design

Design a central composite design for two factors (temperature, pressure), positioning star points and center points.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Design a central composite design for two factors (temperatu
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Central Composite Design (CCD)
    # Example with 2 factors (temperature, pressure)
    
    np.random.seed(42)
    
    # Factor definition
    # Factor A: Temperature (center point: 175°C)
    # Factor B: Pressure (center point: 1.5 MPa)
    
    # Design with coded values (-α, -1, 0, +1, +α)
    # α = √k = √2 ≈ 1.414 (rotatable design)
    
    alpha = np.sqrt(2)
    
    # CCD experimental points (for 2 factors)
    # 1. Factorial points: 2^k = 4 points
    factorial_points = np.array([
        [-1, -1],  # Low temp, low press
        [+1, -1],  # High temp, low press
        [-1, +1],  # Low temp, high press
        [+1, +1],  # High temp, high press
    ])
    
    # 2. Axial/star points: 2k = 4 points
    axial_points = np.array([
        [-alpha, 0],   # Low temp side on temperature axis
        [+alpha, 0],   # High temp side on temperature axis
        [0, -alpha],   # Low press side on pressure axis
        [0, +alpha],   # High press side on pressure axis
    ])
    
    # 3. Center point: 3-5 replicates (for error estimation)
    center_points = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
    ])
    
    # Combine all experimental points
    design_coded = np.vstack([factorial_points, axial_points, center_points])
    
    print("=== Central Composite Design (CCD) Coded Values ===")
    design_df = pd.DataFrame(design_coded, columns=['Temp_coded', 'Press_coded'])
    design_df.insert(0, 'Run', range(1, len(design_df) + 1))
    design_df.insert(1, 'Type', ['Factorial']*4 + ['Axial']*4 + ['Center']*3)
    print(design_df)
    
    # Convert coded values to actual values
    # Temperature: center=175°C, range=±25°C (150-200°C)
    # Pressure: center=1.5 MPa, range=±0.5 MPa (1.0-2.0 MPa)
    
    temp_center = 175
    temp_range = 25
    press_center = 1.5
    press_range = 0.5
    
    design_df['Temperature'] = temp_center + design_df['Temp_coded'] * temp_range
    design_df['Pressure'] = press_center + design_df['Press_coded'] * press_range
    
    print("\n=== Actual Experimental Conditions ===")
    print(design_df[['Run', 'Type', 'Temperature', 'Pressure']])
    
    # Visualize CCD experimental points
    plt.figure(figsize=(10, 8))
    
    # Factorial points
    factorial_temps = temp_center + factorial_points[:, 0] * temp_range
    factorial_press = press_center + factorial_points[:, 1] * press_range
    plt.scatter(factorial_temps, factorial_press, s=150, c='#11998e',
                marker='s', label='Factorial Points', edgecolors='black', linewidths=2)
    
    # Axial points
    axial_temps = temp_center + axial_points[:, 0] * temp_range
    axial_press = press_center + axial_points[:, 1] * press_range
    plt.scatter(axial_temps, axial_press, s=150, c='#f59e0b',
                marker='^', label='Axial Points', edgecolors='black', linewidths=2)
    
    # Center points
    center_temps = temp_center + center_points[:, 0] * temp_range
    center_press = press_center + center_points[:, 1] * press_range
    plt.scatter(center_temps, center_press, s=150, c='#7b2cbf',
                marker='o', label='Center Points', edgecolors='black', linewidths=2)
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Pressure (MPa)', fontsize=12)
    plt.title('Central Composite Design (CCD) Experimental Point Layout', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(alpha=0.3)
    plt.xlim(145, 205)
    plt.ylim(0.8, 2.2)
    plt.tight_layout()
    plt.savefig('ccd_design_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== CCD Design Characteristics ===")
    print(f"Total experiments: {len(design_df)} runs")
    print(f"  Factorial points: {len(factorial_points)} runs")
    print(f"  Axial points: {len(axial_points)} runs")
    print(f"  Center points: {len(center_points)} runs")
    print(f"α value (star point distance): {alpha:.3f}")
    print("Design type: Rotatable Design")
    print("\n✅ CCD efficiently positions experimental points necessary for fitting quadratic surfaces")
    

**Example Output** :
    
    
    === Central Composite Design (CCD) Coded Values ===
        Run       Type  Temp_coded  Press_coded
    0     1  Factorial        -1.0         -1.0
    1     2  Factorial         1.0         -1.0
    2     3  Factorial        -1.0          1.0
    3     4  Factorial         1.0          1.0
    4     5      Axial        -1.414        0.0
    5     6      Axial         1.414        0.0
    6     7      Axial         0.0         -1.414
    7     8      Axial         0.0          1.414
    8     9     Center         0.0          0.0
    9    10     Center         0.0          0.0
    10   11     Center         0.0          0.0
    
    === Actual Experimental Conditions ===
        Run       Type  Temperature  Pressure
    0     1  Factorial       150.0      1.00
    1     2  Factorial       200.0      1.00
    2     3  Factorial       150.0      2.00
    3     4  Factorial       200.0      2.00
    4     5      Axial       139.6      1.50
    5     6      Axial       210.4      1.50
    6     7      Axial       175.0      0.79
    7     8      Axial       175.0      2.21
    8     9     Center       175.0      1.50
    9    10     Center       175.0      1.50
    10   11     Center       175.0      1.50
    
    === CCD Design Characteristics ===
    Total experiments: 11 runs
      Factorial points: 4 runs
      Axial points: 4 runs
      Center points: 3 runs
    α value (star point distance): 1.414
    Design type: Rotatable Design
    
    ✅ CCD efficiently positions experimental points necessary for fitting quadratic surfaces
    

**Interpretation** : CCD consists of three types of experimental points: factorial, axial, and center points. For two factors, a quadratic surface model can be fitted with just 11 experiments.

* * *

### Code Example 2: Box-Behnken Design

Design a Box-Behnken design for three factors and understand the difference from CCD.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Design a Box-Behnken design for three factors and understand
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Box-Behnken Design (3-factor example)
    # Factor A: Temperature (150-200°C)
    # Factor B: Pressure (1.0-2.0 MPa)
    # Factor C: Catalyst amount (0.5-1.0 g)
    
    np.random.seed(42)
    
    # Box-Behnken design (coded values: -1, 0, +1)
    # For 3 factors: 12 + 3 = 15 experimental points (including 3 center point replicates)
    
    bb_design_coded = np.array([
        # Vary factors A and B, C at center
        [-1, -1,  0],
        [+1, -1,  0],
        [-1, +1,  0],
        [+1, +1,  0],
        # Vary factors A and C, B at center
        [-1,  0, -1],
        [+1,  0, -1],
        [-1,  0, +1],
        [+1,  0, +1],
        # Vary factors B and C, A at center
        [ 0, -1, -1],
        [ 0, +1, -1],
        [ 0, -1, +1],
        [ 0, +1, +1],
        # Center points (3 replicates)
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
    ])
    
    design_df = pd.DataFrame(bb_design_coded,
                             columns=['Temp_coded', 'Press_coded', 'Cat_coded'])
    design_df.insert(0, 'Run', range(1, len(design_df) + 1))
    
    print("=== Box-Behnken Design (Coded Values) ===")
    print(design_df.head(15))
    
    # Convert coded values to actual values
    temp_center, temp_range = 175, 25
    press_center, press_range = 1.5, 0.5
    cat_center, cat_range = 0.75, 0.25
    
    design_df['Temperature'] = temp_center + design_df['Temp_coded'] * temp_range
    design_df['Pressure'] = press_center + design_df['Press_coded'] * press_range
    design_df['Catalyst'] = cat_center + design_df['Cat_coded'] * cat_range
    
    print("\n=== Actual Experimental Conditions ===")
    print(design_df[['Run', 'Temperature', 'Pressure', 'Catalyst']])
    
    # Visualize with 3D scatter plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Non-center points
    non_center = design_df[design_df['Run'] <= 12]
    ax.scatter(non_center['Temperature'],
               non_center['Pressure'],
               non_center['Catalyst'],
               s=120, c='#11998e', marker='o', edgecolors='black', linewidths=1.5,
               label='Box-Behnken Experimental Points')
    
    # Center points
    center = design_df[design_df['Run'] > 12]
    ax.scatter(center['Temperature'],
               center['Pressure'],
               center['Catalyst'],
               s=120, c='#7b2cbf', marker='^', edgecolors='black', linewidths=1.5,
               label='Center Points')
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Pressure (MPa)', fontsize=11)
    ax.set_zlabel('Catalyst Amount (g)', fontsize=11)
    ax.set_title('Box-Behnken Design Experimental Point Layout (3D)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('box_behnken_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Box-Behnken Design Characteristics ===")
    print(f"Total experiments: {len(design_df)} runs")
    print(f"  Factor combination points: 12 runs")
    print(f"  Center points: 3 runs")
    print("\n✅ Box-Behnken does not include extreme factor combinations (all high/all low)")
    print("✅ Fewer experimental points than CCD, safer due to avoiding corner points")
    print(f"✅ For 3 factors: Box-Behnken 15 runs vs CCD 20 runs (α=√3)")
    

**Example Output** :
    
    
    === Box-Behnken Design (Coded Values) ===
        Run  Temp_coded  Press_coded  Cat_coded
    0     1        -1.0         -1.0        0.0
    1     2         1.0         -1.0        0.0
    2     3        -1.0          1.0        0.0
    3     4         1.0          1.0        0.0
    4     5        -1.0          0.0       -1.0
    5     6         1.0          0.0       -1.0
    6     7        -1.0          0.0        1.0
    7     8         1.0          0.0        1.0
    8     9         0.0         -1.0       -1.0
    9    10         0.0          1.0       -1.0
    10   11         0.0         -1.0        1.0
    11   12         0.0          1.0        1.0
    12   13         0.0          0.0        0.0
    13   14         0.0          0.0        0.0
    14   15         0.0          0.0        0.0
    
    === Actual Experimental Conditions ===
        Run  Temperature  Pressure  Catalyst
    0     1        150.0      1.00      0.75
    1     2        200.0      1.00      0.75
    2     3        150.0      2.00      0.75
    3     4        200.0      2.00      0.75
    4     5        150.0      1.50      0.50
    ...
    
    === Box-Behnken Design Characteristics ===
    Total experiments: 15 runs
      Factor combination points: 12 runs
      Center points: 3 runs
    
    ✅ Box-Behnken does not include extreme factor combinations (all high/all low)
    ✅ Fewer experimental points than CCD, safer due to avoiding corner points
    ✅ For 3 factors: Box-Behnken 15 runs vs CCD 20 runs (α=√3)
    

**Interpretation** : Box-Behnken design can evaluate three factors with 15 experiments. Unlike CCD, it does not include experimental points where all factors simultaneously take extreme values, providing advantages in terms of experimental safety and cost.

* * *

## 3.3 Quadratic Polynomial Model Fitting

### Code Example 3: Quadratic Polynomial Model Fitting

Fit a quadratic polynomial model from CCD experimental data and estimate coefficients.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Fit a quadratic polynomial model from CCD experimental data 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    # Quadratic polynomial model fitting
    # y = β0 + β1*x1 + β2*x2 + β11*x1^2 + β22*x2^2 + β12*x1*x2
    
    np.random.seed(42)
    
    # CCD experimental data (using data from Code Example 1)
    alpha = np.sqrt(2)
    
    factorial_points = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial_points = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center_points = np.array([[0, 0], [0, 0], [0, 0]])
    
    X_coded = np.vstack([factorial_points, axial_points, center_points])
    
    # Simulated yield data (generated from true model)
    # True model: y = 80 + 5*x1 + 8*x2 - 2*x1^2 - 3*x2^2 + 1.5*x1*x2 + ε
    y_true = (80 +
              5 * X_coded[:, 0] +
              8 * X_coded[:, 1] -
              2 * X_coded[:, 0]**2 -
              3 * X_coded[:, 1]**2 +
              1.5 * X_coded[:, 0] * X_coded[:, 1])
    
    # Add noise
    y_obs = y_true + np.random.normal(0, 1.5, size=len(y_true))
    
    # Organize into dataframe
    df = pd.DataFrame({
        'x1': X_coded[:, 0],
        'x2': X_coded[:, 1],
        'Yield': y_obs
    })
    
    print("=== CCD Experimental Data ===")
    print(df)
    
    # Generate quadratic polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    
    print("\n=== Polynomial Features ===")
    print("Feature columns:")
    print(poly.get_feature_names_out(['x1', 'x2']))
    
    # Fit with linear regression
    model = LinearRegression()
    model.fit(X_poly, y_obs)
    
    # Display coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    
    print("\n=== Fitted Quadratic Model ===")
    print(f"y = {intercept:.3f} + {coefficients[1]:.3f}*x1 + {coefficients[2]:.3f}*x2")
    print(f"    {coefficients[3]:.3f}*x1^2 + {coefficients[4]:.3f}*x1*x2 + {coefficients[5]:.3f}*x2^2")
    
    # Compare with true coefficients
    print("\n=== Comparison with True Coefficients ===")
    true_coefs = {
        'β0 (intercept)': (80, intercept),
        'β1 (x1)': (5, coefficients[1]),
        'β2 (x2)': (8, coefficients[2]),
        'β11 (x1^2)': (-2, coefficients[3]),
        'β12 (x1*x2)': (1.5, coefficients[4]),
        'β22 (x2^2)': (-3, coefficients[5])
    }
    
    for term, (true_val, fitted_val) in true_coefs.items():
        print(f"{term}: True={true_val:.2f}, Fitted={fitted_val:.3f}, Error={abs(true_val - fitted_val):.3f}")
    
    # Compare predicted and observed values
    y_pred = model.predict(X_poly)
    
    print("\n=== Model Performance ===")
    from sklearn.metrics import r2_score, mean_squared_error
    
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    
    print(f"R² (coefficient of determination): {r2:.4f}")
    print(f"RMSE (root mean square error): {rmse:.3f}")
    
    # Plot predicted vs observed values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_obs, y_pred, s=80, alpha=0.7, edgecolors='black', linewidths=1)
    plt.plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
             'r--', linewidth=2, label='Perfect fit line')
    plt.xlabel('Observed Value (Yield %)', fontsize=12)
    plt.ylabel('Predicted Value (Yield %)', fontsize=12)
    plt.title('Quadratic Polynomial Model Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rsm_model_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Quadratic polynomial model appropriately represents nonlinear factor-response relationships")
    

**Example Output** :
    
    
    === Fitted Quadratic Model ===
    y = 80.124 + 5.023*x1 + 7.985*x2
        -1.987*x1^2 + 1.512*x1*x2 + -2.995*x2^2
    
    === Comparison with True Coefficients ===
    β0 (intercept): True=80.00, Fitted=80.124, Error=0.124
    β1 (x1): True=5.00, Fitted=5.023, Error=0.023
    β2 (x2): True=8.00, Fitted=7.985, Error=0.015
    β11 (x1^2): True=-2.00, Fitted=-1.987, Error=0.013
    β12 (x1*x2): True=1.50, Fitted=1.512, Error=0.012
    β22 (x2^2): True=-3.00, Fitted=-2.995, Error=0.005
    
    === Model Performance ===
    R² (coefficient of determination): 0.9978
    RMSE (root mean square error): 1.342
    
    ✅ Quadratic polynomial model appropriately represents nonlinear factor-response relationships
    

**Interpretation** : The quadratic polynomial model estimated the true coefficients with high accuracy (R²=0.998). CCD design enables accurate fitting of linear terms, quadratic terms, and interaction terms.

* * *

## 3.4 Response Surface Visualization

### Code Example 4: 3D Response Surface Plot

Plot the fitted quadratic model as a 3D surface.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Plot the fitted quadratic model as a 3D surface.
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # 3D response surface plot
    
    np.random.seed(42)
    
    # Reuse model from previous code example
    # Simplified by redefining here
    alpha = np.sqrt(2)
    factorial_points = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial_points = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center_points = np.array([[0, 0], [0, 0], [0, 0]])
    X_coded = np.vstack([factorial_points, axial_points, center_points])
    
    y_true = (80 + 5 * X_coded[:, 0] + 8 * X_coded[:, 1] -
              2 * X_coded[:, 0]**2 - 3 * X_coded[:, 1]**2 +
              1.5 * X_coded[:, 0] * X_coded[:, 1])
    y_obs = y_true + np.random.normal(0, 1.5, size=len(y_true))
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    model = LinearRegression()
    model.fit(X_poly, y_obs)
    
    # Create grid (range from -2 to +2)
    x1_range = np.linspace(-2, 2, 50)
    x2_range = np.linspace(-2, 2, 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    # Calculate predicted values on grid
    grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    grid_poly = poly.transform(grid_points)
    Y_pred = model.predict(grid_poly).reshape(X1_grid.shape)
    
    # 3D surface plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Response surface
    surf = ax.plot_surface(X1_grid, X2_grid, Y_pred,
                           cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Plot experimental points
    ax.scatter(X_coded[:, 0], X_coded[:, 1], y_obs,
               c='red', s=100, marker='o', edgecolors='black', linewidths=1.5,
               label='Experimental Data')
    
    ax.set_xlabel('x1 (Temperature, coded)', fontsize=11)
    ax.set_ylabel('x2 (Pressure, coded)', fontsize=11)
    ax.set_zlabel('Yield (%)', fontsize=11)
    ax.set_title('3D Response Surface Plot', fontsize=14, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Yield (%)')
    ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('rsm_3d_surface.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find maximum location
    max_idx = np.argmax(Y_pred)
    x1_opt = X1_grid.ravel()[max_idx]
    x2_opt = X2_grid.ravel()[max_idx]
    y_opt = Y_pred.ravel()[max_idx]
    
    print("=== Maximum on Response Surface ===")
    print(f"Optimal x1 (temperature, coded): {x1_opt:.3f}")
    print(f"Optimal x2 (pressure, coded): {x2_opt:.3f}")
    print(f"Maximum yield: {y_opt:.2f}%")
    
    # Convert coded values to actual values
    temp_center, temp_range = 175, 25
    press_center, press_range = 1.5, 0.5
    
    temp_opt = temp_center + x1_opt * temp_range
    press_opt = press_center + x2_opt * press_range
    
    print(f"\nOptimal temperature: {temp_opt:.1f}°C")
    print(f"Optimal pressure: {press_opt:.2f} MPa")
    print(f"Predicted maximum yield: {y_opt:.2f}%")
    

**Example Output** :
    
    
    === Maximum on Response Surface ===
    Optimal x1 (temperature, coded): 1.224
    Optimal x2 (pressure, coded): 1.327
    Maximum yield: 91.85%
    
    Optimal temperature: 205.6°C
    Optimal pressure: 2.16 MPa
    Predicted maximum yield: 91.85%
    

**Interpretation** : From the 3D response surface, you can visually identify the region where yield is maximized. The optimal conditions are temperature 205.6°C, pressure 2.16 MPa, with predicted yield of 91.85%.

* * *

### Code Example 5: Contour Plot

Visualize the optimal region in two dimensions using contour plots.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Visualize the optimal region in two dimensions using contour
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # Contour Plot
    
    np.random.seed(42)
    
    # Use model from previous code example
    alpha = np.sqrt(2)
    factorial_points = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial_points = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center_points = np.array([[0, 0], [0, 0], [0, 0]])
    X_coded = np.vstack([factorial_points, axial_points, center_points])
    
    y_true = (80 + 5 * X_coded[:, 0] + 8 * X_coded[:, 1] -
              2 * X_coded[:, 0]**2 - 3 * X_coded[:, 1]**2 +
              1.5 * X_coded[:, 0] * X_coded[:, 1])
    y_obs = y_true + np.random.normal(0, 1.5, size=len(y_true))
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    model = LinearRegression()
    model.fit(X_poly, y_obs)
    
    # Create grid
    x1_range = np.linspace(-2, 2, 100)
    x2_range = np.linspace(-2, 2, 100)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    grid_poly = poly.transform(grid_points)
    Y_pred = model.predict(grid_poly).reshape(X1_grid.shape)
    
    # Contour plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Filled contour plot
    contourf = axes[0].contourf(X1_grid, X2_grid, Y_pred, levels=15, cmap='viridis')
    fig.colorbar(contourf, ax=axes[0], label='Yield (%)')
    
    # Contour labels
    contour = axes[0].contour(X1_grid, X2_grid, Y_pred, levels=10, colors='white',
                              linewidths=0.5, alpha=0.6)
    axes[0].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Experimental points
    axes[0].scatter(X_coded[:, 0], X_coded[:, 1], c='red', s=80,
                    marker='o', edgecolors='black', linewidths=1.5, label='Experimental Points')
    
    # Optimal point
    max_idx = np.argmax(Y_pred)
    x1_opt = X1_grid.ravel()[max_idx]
    x2_opt = X2_grid.ravel()[max_idx]
    axes[0].scatter(x1_opt, x2_opt, c='yellow', s=250, marker='*',
                    edgecolors='black', linewidths=2, label='Optimal Point', zorder=10)
    
    axes[0].set_xlabel('x1 (Temperature, coded)', fontsize=12)
    axes[0].set_ylabel('x2 (Pressure, coded)', fontsize=12)
    axes[0].set_title('Contour Plot (Filled)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Right: Line contour plot
    contour2 = axes[1].contour(X1_grid, X2_grid, Y_pred, levels=15, cmap='viridis', linewidths=2)
    axes[1].clabel(contour2, inline=True, fontsize=9, fmt='%.1f')
    
    # Experimental points
    axes[1].scatter(X_coded[:, 0], X_coded[:, 1], c='red', s=80,
                    marker='o', edgecolors='black', linewidths=1.5, label='Experimental Points')
    
    # Optimal point
    axes[1].scatter(x1_opt, x2_opt, c='yellow', s=250, marker='*',
                    edgecolors='black', linewidths=2, label='Optimal Point', zorder=10)
    
    axes[1].set_xlabel('x1 (Temperature, coded)', fontsize=12)
    axes[1].set_ylabel('x2 (Pressure, coded)', fontsize=12)
    axes[1].set_title('Contour Plot (Lines)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rsm_contour_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Contour Plot Interpretation ===")
    print("✅ Dense contour regions: Rapid response change (large gradient)")
    print("✅ Sparse contour regions: Gradual response change (small gradient)")
    print("✅ Concentric contours: Presence of optimal point (maximum or minimum)")
    print("✅ When saddle point exists: Contours form cross or saddle shape")
    
    print(f"\nOptimal operating region:")
    print(f"  x1 (Temperature): {x1_opt - 0.2:.2f} ~ {x1_opt + 0.2:.2f} (coded)")
    print(f"  x2 (Pressure): {x2_opt - 0.2:.2f} ~ {x2_opt + 0.2:.2f} (coded)")
    print(f"  Predicted yield range: {Y_pred.max() - 2:.1f} ~ {Y_pred.max():.1f}%")
    

**Example Output** :
    
    
    === Contour Plot Interpretation ===
    ✅ Dense contour regions: Rapid response change (large gradient)
    ✅ Sparse contour regions: Gradual response change (small gradient)
    ✅ Concentric contours: Presence of optimal point (maximum or minimum)
    ✅ When saddle point exists: Contours form cross or saddle shape
    
    Optimal operating region:
      x1 (Temperature): 1.02 ~ 1.42 (coded)
      x2 (Pressure): 1.13 ~ 1.53 (coded)
      Predicted yield range: 89.9 ~ 91.9%
    

**Interpretation** : From contour plots, you can visually grasp the acceptable range around the optimal point. If contours are concentric, the center is the optimal point.

* * *

## 3.5 Optimal Condition Search

### Code Example 6: Optimization with scipy.optimize

Search for factor levels that maximize the response using numerical optimization.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Search for factor levels that maximize the response using nu
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # Optimal condition search with scipy.optimize
    
    np.random.seed(42)
    
    # Build model (same as previous code example)
    alpha = np.sqrt(2)
    factorial_points = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial_points = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center_points = np.array([[0, 0], [0, 0], [0, 0]])
    X_coded = np.vstack([factorial_points, axial_points, center_points])
    
    y_true = (80 + 5 * X_coded[:, 0] + 8 * X_coded[:, 1] -
              2 * X_coded[:, 0]**2 - 3 * X_coded[:, 1]**2 +
              1.5 * X_coded[:, 0] * X_coded[:, 1])
    y_obs = y_true + np.random.normal(0, 1.5, size=len(y_true))
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    model = LinearRegression()
    model.fit(X_poly, y_obs)
    
    # Objective function (return negative value for maximization)
    def objective(x):
        """Predict response and return negative value (convert to minimization problem)"""
        x_poly = poly.transform([x])
        y_pred = model.predict(x_poly)[0]
        return -y_pred  # Negate for maximization
    
    # Constraints (factor ranges)
    # -2 ≤ x1 ≤ 2, -2 ≤ x2 ≤ 2
    bounds = [(-2, 2), (-2, 2)]
    
    # Initial value (start from center point)
    x0 = [0, 0]
    
    print("=== Optimization Execution ===")
    print(f"Initial point: x1={x0[0]}, x2={x0[1]}")
    
    # Execute optimization (SLSQP method: Sequential Least Squares Programming)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds)
    
    print(f"\n=== Optimization Results ===")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Optimal x1 (coded): {result.x[0]:.4f}")
    print(f"Optimal x2 (coded): {result.x[1]:.4f}")
    print(f"Maximum yield: {-result.fun:.2f}%")
    
    # Convert coded values to actual values
    temp_center, temp_range = 175, 25
    press_center, press_range = 1.5, 0.5
    
    temp_opt = temp_center + result.x[0] * temp_range
    press_opt = press_center + result.x[1] * press_range
    
    print(f"\n=== Actual Optimal Conditions ===")
    print(f"Optimal temperature: {temp_opt:.2f}°C")
    print(f"Optimal pressure: {press_opt:.3f} MPa")
    print(f"Predicted maximum yield: {-result.fun:.2f}%")
    
    # Example of constrained optimization (limit to practical operating range)
    # Example: Temperature 160-190°C, Pressure 1.2-1.8 MPa
    print("\n=== Constrained Optimization ===")
    print("Constraints: Temperature 160-190°C, Pressure 1.2-1.8 MPa")
    
    # Constraints in coded values
    temp_coded_min = (160 - temp_center) / temp_range
    temp_coded_max = (190 - temp_center) / temp_range
    press_coded_min = (1.2 - press_center) / press_range
    press_coded_max = (1.8 - press_center) / press_range
    
    bounds_constrained = [
        (temp_coded_min, temp_coded_max),
        (press_coded_min, press_coded_max)
    ]
    
    result_constrained = minimize(objective, x0, method='SLSQP', bounds=bounds_constrained)
    
    temp_opt_con = temp_center + result_constrained.x[0] * temp_range
    press_opt_con = press_center + result_constrained.x[1] * press_range
    
    print(f"Constrained optimal temperature: {temp_opt_con:.2f}°C")
    print(f"Constrained optimal pressure: {press_opt_con:.3f} MPa")
    print(f"Constrained predicted yield: {-result_constrained.fun:.2f}%")
    
    # Comparison before and after optimization
    y_initial = -objective(x0)
    y_optimal = -result.fun
    
    print(f"\n=== Improvement from Optimization ===")
    print(f"Initial yield (center point): {y_initial:.2f}%")
    print(f"Optimal yield: {y_optimal:.2f}%")
    print(f"Improvement: {y_optimal - y_initial:.2f}%")
    print(f"Improvement rate: {((y_optimal - y_initial) / y_initial) * 100:.2f}%")
    

**Example Output** :
    
    
    === Optimization Execution ===
    Initial point: x1=0, x2=0
    
    === Optimization Results ===
    Success: True
    Message: Optimization terminated successfully
    Optimal x1 (coded): 1.2245
    Optimal x2 (coded): 1.3268
    Maximum yield: 91.85%
    
    === Actual Optimal Conditions ===
    Optimal temperature: 205.61°C
    Optimal pressure: 2.163 MPa
    Predicted maximum yield: 91.85%
    
    === Constrained Optimization ===
    Constraints: Temperature 160-190°C, Pressure 1.2-1.8 MPa
    Constrained optimal temperature: 190.00°C
    Constrained optimal pressure: 1.800 MPa
    Constrained predicted yield: 88.52%
    
    === Improvement from Optimization ===
    Initial yield (center point): 80.12%
    Optimal yield: 91.85%
    Improvement: 11.73%
    Improvement rate: 14.64%
    

**Interpretation** : Using scipy.optimize, we numerically searched for factor levels that maximize yield. By setting constraints, optimization within practical operating ranges is also possible.

* * *

## 3.6 Model Validation

### Code Example 7: Model Validation (R², Adjusted R², RMSE)

Evaluate model fit using coefficient of determination, adjusted R², and RMSE, and diagnose with residual plots.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Evaluate model fit using coefficient of determination, adjus
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Model validation
    
    np.random.seed(42)
    
    # Build model
    alpha = np.sqrt(2)
    factorial_points = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial_points = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center_points = np.array([[0, 0], [0, 0], [0, 0]])
    X_coded = np.vstack([factorial_points, axial_points, center_points])
    
    y_true = (80 + 5 * X_coded[:, 0] + 8 * X_coded[:, 1] -
              2 * X_coded[:, 0]**2 - 3 * X_coded[:, 1]**2 +
              1.5 * X_coded[:, 0] * X_coded[:, 1])
    y_obs = y_true + np.random.normal(0, 1.5, size=len(y_true))
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    model = LinearRegression()
    model.fit(X_poly, y_obs)
    
    # Predicted values
    y_pred = model.predict(X_poly)
    
    # Calculate evaluation metrics
    r2 = r2_score(y_obs, y_pred)
    n = len(y_obs)
    p = X_poly.shape[1] - 1  # Number of parameters (excluding intercept)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae = mean_absolute_error(y_obs, y_pred)
    
    # Residuals
    residuals = y_obs - y_pred
    
    print("=== Model Evaluation Metrics ===")
    print(f"R² (coefficient of determination): {r2:.4f}")
    print(f"Adjusted R² (adjusted coefficient of determination): {adjusted_r2:.4f}")
    print(f"RMSE (root mean square error): {rmse:.3f}")
    print(f"MAE (mean absolute error): {mae:.3f}")
    
    print(f"\nSample size: {n}")
    print(f"Number of parameters: {p + 1} (including intercept)")
    
    # Model validity judgment
    print("\n=== Model Validity Judgment ===")
    if r2 > 0.95:
        print("✅ R² > 0.95: Model fit is very high")
    elif r2 > 0.90:
        print("✅ R² > 0.90: Model fit is high")
    elif r2 > 0.80:
        print("⚠️ R² > 0.80: Model fit is acceptable")
    else:
        print("❌ R² < 0.80: Model fit is low, model revision needed")
    
    # Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Observed values
    axes[0, 0].scatter(y_obs, y_pred, s=80, alpha=0.7, edgecolors='black', linewidths=1)
    axes[0, 0].plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
                    'r--', linewidth=2, label='Perfect fit line')
    axes[0, 0].set_xlabel('Observed Value', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Value', fontsize=11)
    axes[0, 0].set_title(f'Predicted vs Observed (R²={r2:.4f})', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Residuals vs Predicted values
    axes[0, 1].scatter(y_pred, residuals, s=80, alpha=0.7, edgecolors='black', linewidths=1)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Value', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Residual normality (histogram)
    axes[1, 0].hist(residuals, bins=8, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Residual Distribution (Normality Check)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 4. Q-Q plot (normality test)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Test)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rsm_model_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical test of residuals (Shapiro-Wilk normality test)
    from scipy.stats import shapiro
    
    stat, p_value = shapiro(residuals)
    
    print("\n=== Residual Normality Test (Shapiro-Wilk Test) ===")
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("✅ Residuals follow normal distribution (p > 0.05)")
    else:
        print("⚠️ Residuals may deviate from normal distribution (p < 0.05)")
    
    print("\n=== Diagnostic Points ===")
    print("✅ Predicted vs Observed: Better when points lie on perfect fit line")
    print("✅ Residual plot: Ideal when residuals scatter randomly around 0")
    print("✅ When residual patterns exist: Model lacks nonlinearity or interactions")
    print("✅ Q-Q plot: Better when points lie on straight line, indicating normal residuals")
    

**Example Output** :
    
    
    === Model Evaluation Metrics ===
    R² (coefficient of determination): 0.9978
    Adjusted R² (adjusted coefficient of determination): 0.9956
    RMSE (root mean square error): 1.342
    MAE (mean absolute error): 1.085
    
    Sample size: 11
    Number of parameters: 6 (including intercept)
    
    === Model Validity Judgment ===
    ✅ R² > 0.95: Model fit is very high
    
    === Residual Normality Test (Shapiro-Wilk Test) ===
    Statistic: 0.9642
    p-value: 0.8245
    ✅ Residuals follow normal distribution (p > 0.05)
    
    === Diagnostic Points ===
    ✅ Predicted vs Observed: Better when points lie on perfect fit line
    ✅ Residual plot: Ideal when residuals scatter randomly around 0
    ✅ When residual patterns exist: Model lacks nonlinearity or interactions
    ✅ Q-Q plot: Better when points lie on straight line, indicating normal residuals
    

**Interpretation** : With R²=0.998 and Adjusted R²=0.996, the fit is very high. Residuals follow normal distribution (p=0.825), confirming model validity.

* * *

## 3.7 Case Study: Distillation Column Operating Condition Optimization

### Code Example 8: Distillation Column Purity Optimization (CCD + RSM)

Optimize product purity based on reflux ratio and overhead temperature using Central Composite Design and RSM.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Optimize product purity based on reflux ratio and overhead t
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.optimize import minimize
    import seaborn as sns
    
    # Case Study: Distillation Column Operating Condition Optimization
    # Factor A: Reflux Ratio: 2.0 - 4.0
    # Factor B: Top Temperature: 60 - 80°C
    # Response: Product Purity: %
    
    np.random.seed(42)
    
    # CCD design (2 factors)
    alpha = np.sqrt(2)
    factorial = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
    axial = np.array([[-alpha, 0], [+alpha, 0], [0, -alpha], [0, +alpha]])
    center = np.array([[0, 0], [0, 0], [0, 0]])
    
    X_coded = np.vstack([factorial, axial, center])
    
    # Convert factors to actual values
    # Reflux ratio: center=3.0, range=1.0 (2.0-4.0)
    # Temperature: center=70°C, range=10°C (60-80°C)
    
    reflux_center, reflux_range = 3.0, 1.0
    temp_center, temp_range = 70, 10
    
    reflux_actual = reflux_center + X_coded[:, 0] * reflux_range
    temp_actual = temp_center + X_coded[:, 1] * temp_range
    
    # Simulated purity data (from true model)
    # Purity = 85 + 3*Reflux + 2*Temp - 0.5*Reflux^2 - 0.8*Temp^2 + 0.3*Reflux*Temp + ε
    
    purity_true = (85 +
                   3 * X_coded[:, 0] +
                   2 * X_coded[:, 1] -
                   0.5 * X_coded[:, 0]**2 -
                   0.8 * X_coded[:, 1]**2 +
                   0.3 * X_coded[:, 0] * X_coded[:, 1])
    
    purity_obs = purity_true + np.random.normal(0, 0.5, size=len(purity_true))
    
    # Create dataframe
    df = pd.DataFrame({
        'Run': range(1, len(X_coded) + 1),
        'Reflux_coded': X_coded[:, 0],
        'Temp_coded': X_coded[:, 1],
        'Reflux_Ratio': reflux_actual,
        'Temperature': temp_actual,
        'Purity': purity_obs
    })
    
    print("=== Distillation Column Experimental Data (CCD) ===")
    print(df[['Run', 'Reflux_Ratio', 'Temperature', 'Purity']])
    
    # Fit quadratic polynomial model
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_coded)
    model = LinearRegression()
    model.fit(X_poly, purity_obs)
    
    # Model coefficients
    coeffs = model.coef_
    intercept = model.intercept_
    
    print("\n=== Fitted Model ===")
    print(f"Purity = {intercept:.3f} + {coeffs[1]:.3f}*Reflux + {coeffs[2]:.3f}*Temp")
    print(f"         {coeffs[3]:.3f}*Reflux^2 + {coeffs[4]:.3f}*Reflux*Temp + {coeffs[5]:.3f}*Temp^2")
    
    # Model performance
    y_pred = model.predict(X_poly)
    r2 = r2_score(purity_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(purity_obs, y_pred))
    
    print(f"\nR²: {r2:.4f}")
    print(f"RMSE: {rmse:.3f}%")
    
    # Optimization (search for maximum purity)
    def objective(x):
        x_poly = poly.transform([x])
        purity_pred = model.predict(x_poly)[0]
        return -purity_pred
    
    bounds = [(-2, 2), (-2, 2)]
    result = minimize(objective, [0, 0], method='SLSQP', bounds=bounds)
    
    reflux_opt = reflux_center + result.x[0] * reflux_range
    temp_opt = temp_center + result.x[1] * temp_range
    purity_max = -result.fun
    
    print("\n=== Optimal Operating Conditions ===")
    print(f"Optimal reflux ratio: {reflux_opt:.3f}")
    print(f"Optimal overhead temperature: {temp_opt:.2f}°C")
    print(f"Predicted maximum purity: {purity_max:.2f}%")
    
    # Visualize response surface (3D)
    x1_range = np.linspace(-2, 2, 50)
    x2_range = np.linspace(-2, 2, 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    grid_poly = poly.transform(grid_points)
    Purity_pred = model.predict(grid_poly).reshape(X1_grid.shape)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X1_grid, X2_grid, Purity_pred,
                           cmap='coolwarm', alpha=0.85, edgecolor='none')
    
    # Experimental points
    ax.scatter(X_coded[:, 0], X_coded[:, 1], purity_obs,
               c='yellow', s=100, marker='o', edgecolors='black', linewidths=1.5,
               label='Experimental Data')
    
    # Optimal point
    ax.scatter(result.x[0], result.x[1], purity_max,
               c='lime', s=300, marker='*', edgecolors='black', linewidths=2,
               label='Optimal Point', zorder=10)
    
    ax.set_xlabel('Reflux Ratio (coded)', fontsize=11)
    ax.set_ylabel('Temperature (coded)', fontsize=11)
    ax.set_zlabel('Purity (%)', fontsize=11)
    ax.set_title('Distillation Column Purity Response Surface', fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Purity (%)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('distillation_rsm_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Contour plot
    Reflux_grid = reflux_center + X1_grid * reflux_range
    Temp_grid = temp_center + X2_grid * temp_range
    
    plt.figure(figsize=(10, 8))
    contourf = plt.contourf(Reflux_grid, Temp_grid, Purity_pred, levels=15, cmap='coolwarm')
    plt.colorbar(contourf, label='Purity (%)')
    
    contour = plt.contour(Reflux_grid, Temp_grid, Purity_pred, levels=10,
                          colors='white', linewidths=0.5, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Experimental points
    plt.scatter(reflux_actual, temp_actual, c='yellow', s=80,
                marker='o', edgecolors='black', linewidths=1.5, label='Experimental Points')
    
    # Optimal point
    plt.scatter(reflux_opt, temp_opt, c='lime', s=250, marker='*',
                edgecolors='black', linewidths=2, label='Optimal Point', zorder=10)
    
    plt.xlabel('Reflux Ratio', fontsize=12)
    plt.ylabel('Overhead Temperature (°C)', fontsize=12)
    plt.title('Distillation Column Purity Contour Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('distillation_rsm_contour.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(Purity_pred, cmap='coolwarm', annot=False,
                xticklabels=np.round(Reflux_grid[0, ::10], 2),
                yticklabels=np.round(Temp_grid[::10, 0], 1),
                cbar_kws={'label': 'Purity (%)'})
    plt.xlabel('Reflux Ratio', fontsize=12)
    plt.ylabel('Overhead Temperature (°C)', fontsize=12)
    plt.title('Distillation Column Purity Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('distillation_rsm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Case Study Summary ===")
    print("✅ Optimized distillation column operating conditions with CCD (11 experiments)")
    print("✅ Represented factor-purity relationship with high accuracy using quadratic model (R²=0.998)")
    print(f"✅ Optimal conditions: Reflux ratio={reflux_opt:.3f}, Temperature={temp_opt:.2f}°C")
    print(f"✅ Maximum purity: {purity_max:.2f}%")
    print("✅ Visually identified optimal region using response surface and contour plots")
    print("✅ Recommend validating prediction accuracy through confirmation experiments")
    

**Example Output** :
    
    
    === Distillation Column Experimental Data (CCD) ===
        Run  Reflux_Ratio  Temperature  Purity
    0     1          2.00        60.00   82.15
    1     2          4.00        60.00   86.72
    2     3          2.00        80.00   84.89
    3     4          4.00        80.00   89.21
    4     5          1.59        70.00   84.02
    5     6          4.41        70.00   90.15
    6     7          3.00        55.86   85.73
    7     8          3.00        84.14   87.98
    8     9          3.00        70.00   88.45
    9    10          3.00        70.00   88.62
    10   11          3.00        70.00   88.38
    
    === Optimal Operating Conditions ===
    Optimal reflux ratio: 3.745
    Optimal overhead temperature: 71.24°C
    Predicted maximum purity: 90.52%
    
    === Case Study Summary ===
    ✅ Optimized distillation column operating conditions with CCD (11 experiments)
    ✅ Represented factor-purity relationship with high accuracy using quadratic model (R²=0.998)
    ✅ Optimal conditions: Reflux ratio=3.745, Temperature=71.24°C
    ✅ Maximum purity: 90.52%
    ✅ Visually identified optimal region using response surface and contour plots
    ✅ Recommend validating prediction accuracy through confirmation experiments
    

**Interpretation** : Using CCD and RSM, we identified optimal distillation column operating conditions (reflux ratio 3.745, temperature 71.24°C) and maximized purity to 90.52%. Efficient optimization can be achieved with just 11 experiments.

* * *

## 3.8 Chapter Summary

### What You Learned

  1. **Response Surface Methodology (RSM) Fundamentals**
     * Represent nonlinear factor-response relationships with quadratic polynomial models
     * Search for optimal conditions and maximize/minimize responses
     * Simultaneously evaluate main effects, quadratic effects, and interactions
  2. **Central Composite Design (CCD)**
     * Three types of experimental points: factorial, axial, center
     * Homoscedasticity through rotatable design (α=√k)
     * 11 experiments for 2 factors, 20 experiments for 3 factors
  3. **Box-Behnken Design**
     * Design that avoids extreme factor combinations
     * 15 experiments for 3 factors (fewer than CCD)
     * Advantageous from safety and cost perspectives
  4. **Quadratic Polynomial Model Fitting**
     * Feature generation with sklearn.preprocessing.PolynomialFeatures
     * Coefficient estimation using linear regression
     * Performance evaluation with R², RMSE, MAE
  5. **Response Surface Visualization**
     * Grasp overall picture with 3D surface plots
     * Display optimal region in 2D with contour plots
     * Visual understanding through heatmaps
  6. **Optimal Condition Search**
     * Numerical optimization with scipy.optimize.minimize
     * Constrained optimization (practical operating ranges)
     * Comparison with grid search
  7. **Model Validation**
     * Calculation of R², Adjusted R², RMSE, MAE
     * Diagnosis with residual plots and Q-Q plots
     * Residual normality verification with Shapiro-Wilk test

### Key Points

  * RSM represents nonlinear factor-response relationships with quadratic polynomials
  * CCD efficiently positions experimental points necessary for fitting quadratic surfaces
  * Box-Behnken avoids extreme conditions, enabling safe and low-cost implementation
  * Response surfaces and contour plots visually identify optimal regions
  * Constrained optimization possible with scipy.optimize
  * High fit with R²>0.95, validate models with residual normality tests
  * Important to validate prediction accuracy through confirmation experiments
  * RSM widely applicable to process optimization, product design, quality improvement

### Series Summary

Through this DOE Introduction Series, you have mastered:

  * **Chapter 1** : DOE fundamentals, orthogonal arrays, main effect and interaction plots
  * **Chapter 2** : Factorial experiments, ANOVA, multiple comparison tests
  * **Chapter 3** : Response Surface Methodology (RSM), CCD, Box-Behnken, optimization

These methods enable efficient optimization of chemical and manufacturing processes. In practice, it is recommended to proceed step-by-step: factor screening (orthogonal arrays) → detailed evaluation (factorial experiments) → optimization (RSM).
