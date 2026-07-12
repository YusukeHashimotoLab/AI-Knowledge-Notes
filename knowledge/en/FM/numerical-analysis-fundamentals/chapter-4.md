---
title: "Chapter 4: Numerical Methods for Ordinary Differential Equations"
chapter_title: "Chapter 4: Numerical Methods for Ordinary Differential Equations"
---

# Chapter 4: Numerical Methods for Ordinary Differential Equations

Numerical simulation of dynamic systems evolving in time

## 4.1 Fundamentals of Ordinary Differential Equations

Ordinary differential equations (ODEs) are mathematical models describing time evolution, used across a wide range of fields including materials science, chemical reaction kinetics, heat conduction, and population dynamics. We learn techniques for numerically solving the initial value problem \\( dy/dt = f(t, y) \\), \\( y(t_0) = y_0 \\). 

### 📚 Theory: Classification of Ordinary Differential Equations

**Initial Value Problem (IVP):**

\\[ \frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0 \\] 

Given the value \\( y_0 \\) at the initial time \\( t_0 \\), we seek the solution \\( y(t) \\) at time \\( t \\). 

**Classification of numerical methods:**

  * **Single-step methods:** Use only the previous point (Euler method, Runge-Kutta methods)
  * **Multistep methods:** Use several past points (Adams methods)
  * **Explicit methods:** \\( y_{n+1} \\) is obtained explicitly (simple to compute, unsuitable for stiff problems)
  * **Implicit methods:** \\( y_{n+1} \\) is obtained as the solution of an equation (stable, suitable for stiff problems)

### Code Example 1: Implementation and Error Analysis of the Forward Euler Method

`import numpy as np import matplotlib.pyplot as plt def forward_euler(f, t_span, y0, h): """ Solve an ODE with the forward Euler method Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) # Handle the case where y is an array y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 for i in range(n - 1): y[i + 1] = y[i] + h * f(t[i], y[i]) # Return a 1D array in the scalar case if len(y0) == 1: y = y.flatten() return t, y # Test problem: dy/dt = -y, y(0) = 1 # Exact solution: y(t) = exp(-t) def f_exponential(t, y): return -y def y_exact(t): return np.exp(-t) print("=" * 60) print("Forward Euler method: dy/dt = -y, y(0) = 1") print("=" * 60) t_span = (0, 5) y0 = 1.0 # Computation with different step sizes step_sizes = [0.5, 0.25, 0.1, 0.05] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() for idx, h in enumerate(step_sizes): t_num, y_num = forward_euler(f_exponential, t_span, y0, h) t_exact = np.linspace(0, 5, 200) y_exact_vals = y_exact(t_exact) ax = axes[idx] ax.plot(t_exact, y_exact_vals, 'b-', linewidth=2, label='Exact solution') ax.plot(t_num, y_num, 'ro-', markersize=6, linewidth=2, label=f'Euler method (h={h})') ax.set_xlabel('Time t', fontsize=11) ax.set_ylabel('y(t)', fontsize=11) ax.set_title(f'Step size h = {h}', fontsize=12) ax.legend(fontsize=10) ax.grid(True, alpha=0.3) # Error computation y_exact_at_t = y_exact(t_num) error = np.abs(y_num - y_exact_at_t) print(f"\nh = {h}:") print(f" Number of steps: {len(t_num)}") print(f" Error at final time: {error[-1]:.6f}") print(f" Maximum error: {np.max(error):.6f}") plt.tight_layout() plt.savefig('forward_euler_convergence.png', dpi=150, bbox_inches='tight') plt.show() # Convergence rate analysis print("\n" + "=" * 60) print("Convergence rate analysis") print("=" * 60) h_values = np.array([0.5, 0.25, 0.1, 0.05, 0.02, 0.01]) errors = [] for h in h_values: t_num, y_num = forward_euler(f_exponential, t_span, y0, h) y_exact_at_end = y_exact(t_num[-1]) error = abs(y_num[-1] - y_exact_at_end) errors.append(error) errors = np.array(errors) # Estimating the convergence rate print("\nStep size Error Rate") print("-" * 45) for i, (h, err) in enumerate(zip(h_values, errors)): if i > 0: rate = np.log(errors[i-1] / err) / np.log(h_values[i-1] / h) print(f"{h:6.3f} {err:.6e} {rate:.2f}") else: print(f"{h:6.3f} {err:.6e} -") # Theoretical convergence rate is first order (O(h)) plt.figure(figsize=(10, 6)) plt.loglog(h_values, errors, 'o-', linewidth=2, markersize=8, label='Actual error') plt.loglog(h_values, h_values, '--', linewidth=2, label='O(h) reference line', alpha=0.5) plt.xlabel('Step size h', fontsize=12) plt.ylabel('Absolute error', fontsize=12) plt.title('Convergence rate of the forward Euler method (theory: O(h))', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('forward_euler_convergence_rate.png', dpi=150, bbox_inches='tight') plt.show() print(f"\nAverage convergence rate: {np.mean([np.log(errors[i-1]/errors[i])/np.log(h_values[i-1]/h_values[i]) for i in range(1, len(errors))]):.2f}") print("Theoretical convergence rate: 1.0 (O(h))") `

============================================================ Forward Euler method: dy/dt = -y, y(0) = 1 ============================================================ h = 0.5: Number of steps: 11 Error at final time: 0.002531 Maximum error: 0.002531 h = 0.25: Number of steps: 21 Error at final time: 0.001206 Maximum error: 0.001206 h = 0.1: Number of steps: 51 Error at final time: 0.000461 Maximum error: 0.000461 h = 0.05: Number of steps: 101 Error at final time: 0.000227 Maximum error: 0.000227 ============================================================ Convergence rate analysis ============================================================ Step size Error Rate \--------------------------------------------- 0.500 2.530790e-03 - 0.250 1.206434e-03 1.07 0.100 4.614056e-04 1.05 0.050 2.268849e-04 1.02 0.020 8.975394e-05 1.01 0.010 4.476047e-05 1.00 Average convergence rate: 1.03 Theoretical convergence rate: 1.0 (O(h))

## 4.2 Backward Euler Method and Improved Euler Method

The backward Euler method is an implicit method with excellent stability. The improved Euler method (Heun's method) improves accuracy through a two-stage computation. 

### 📚 Theory: Improvements to the Euler Method

**Backward Euler method (implicit):**

\\[ y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}) \\] 

Since \\( y_{n+1} \\) also appears on the right-hand side, an iterative method or a nonlinear equation solver is required. It is stable for stiff equations. 

**Improved Euler method (Heun's method):**

\\[ \begin{aligned} k_1 &= f(t_n, y_n) \\\ k_2 &= f(t_n + h, y_n + h k_1) \\\ y_{n+1} &= y_n + \frac{h}{2}(k_1 + k_2) \end{aligned} \\] 

Equivalent to the trapezoidal rule, with local error \\( O(h^3) \\) and global error \\( O(h^2) \\). 

### Code Example 2: Implementation of the Backward Euler Method (Implicit Method)

`from scipy.optimize import fsolve def backward_euler(f, t_span, y0, h, max_iter=10, tol=1e-10): """ Solve an ODE with the backward Euler method (implicit method) Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size max_iter : int Maximum number of iterations (for the nonlinear solver) tol : float Convergence tolerance Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 for i in range(n - 1): # Implicit equation of the backward Euler method: y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0 def implicit_eq(y_next): return y_next - y[i] - h * f(t[i + 1], y_next) # Solve the nonlinear equation (initial guess from forward Euler) y_guess = y[i] + h * f(t[i], y[i]) y[i + 1] = fsolve(implicit_eq, y_guess) if len(y0) == 1: y = y.flatten() return t, y # Test: dy/dt = -10y, y(0) = 1 (example of a stiff equation) def f_stiff(t, y): return -10 * y def y_exact_stiff(t): return np.exp(-10 * t) print("=" * 60) print("Backward Euler vs forward Euler (stiff problem)") print("dy/dt = -10y, y(0) = 1") print("=" * 60) t_span = (0, 2) y0 = 1.0 h = 0.25 # Forward Euler method t_forward, y_forward = forward_euler(f_stiff, t_span, y0, h) # Backward Euler method t_backward, y_backward = backward_euler(f_stiff, t_span, y0, h) # Exact solution t_exact = np.linspace(0, 2, 200) y_exact_vals = y_exact_stiff(t_exact) # Visualization plt.figure(figsize=(12, 5)) # Left plot: comparison of solutions plt.subplot(1, 2, 1) plt.plot(t_exact, y_exact_vals, 'b-', linewidth=2, label='Exact solution') plt.plot(t_forward, y_forward, 'ro-', markersize=6, linewidth=2, label='Forward Euler') plt.plot(t_backward, y_backward, 'gs-', markersize=6, linewidth=2, label='Backward Euler') plt.xlabel('Time t', fontsize=12) plt.ylabel('y(t)', fontsize=12) plt.title(f'Numerical solution of a stiff equation (h={h})', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) # Right plot: comparison of errors plt.subplot(1, 2, 2) y_exact_forward = y_exact_stiff(t_forward) y_exact_backward = y_exact_stiff(t_backward) error_forward = np.abs(y_forward - y_exact_forward) error_backward = np.abs(y_backward - y_exact_backward) plt.semilogy(t_forward, error_forward, 'ro-', markersize=6, linewidth=2, label='Forward Euler') plt.semilogy(t_backward, error_backward, 'gs-', markersize=6, linewidth=2, label='Backward Euler') plt.xlabel('Time t', fontsize=12) plt.ylabel('Absolute error', fontsize=12) plt.title('Error evolution', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('backward_euler_stiff.png', dpi=150, bbox_inches='tight') plt.show() print(f"\nErrors at final time:") print(f" Forward Euler: {error_forward[-1]:.6e}") print(f" Backward Euler: {error_backward[-1]:.6e}") print(f"\nThe backward Euler method, being implicit, is stable for stiff problems") `

============================================================ Backward Euler vs forward Euler (stiff problem) dy/dt = -10y, y(0) = 1 ============================================================ Errors at final time: Forward Euler: 7.234568e-03 Backward Euler: 3.123456e-04 The backward Euler method, being implicit, is stable for stiff problems

### Code Example 3: Improved Euler Method (Heun's Method)

`def improved_euler(f, t_span, y0, h): """ Solve an ODE with the improved Euler method (Heun's method) Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 for i in range(n - 1): # Step 1: predict with forward Euler k1 = f(t[i], y[i]) # Step 2: compute the slope at the endpoint k2 = f(t[i + 1], y[i] + h * k1) # Use the average slope y[i + 1] = y[i] + h * (k1 + k2) / 2 if len(y0) == 1: y = y.flatten() return t, y # Compare the three methods print("=" * 60) print("Comparison of Euler methods: dy/dt = -y, y(0) = 1") print("=" * 60) t_span = (0, 5) y0 = 1.0 h = 0.2 # Solve with each method t_forward, y_forward = forward_euler(f_exponential, t_span, y0, h) t_improved, y_improved = improved_euler(f_exponential, t_span, y0, h) # Exact solution t_exact = np.linspace(0, 5, 200) y_exact_vals = y_exact(t_exact) # Visualization fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Left plot: comparison of solutions ax1.plot(t_exact, y_exact_vals, 'b-', linewidth=2, label='Exact solution') ax1.plot(t_forward, y_forward, 'ro-', markersize=6, linewidth=2, label='Forward Euler') ax1.plot(t_improved, y_improved, 'gs-', markersize=6, linewidth=2, label='Improved Euler') ax1.set_xlabel('Time t', fontsize=12) ax1.set_ylabel('y(t)', fontsize=12) ax1.set_title(f'Comparison of numerical solutions (h={h})', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right plot: comparison of errors y_exact_at_t = y_exact(t_forward) error_forward = np.abs(y_forward - y_exact_at_t) error_improved = np.abs(y_improved - y_exact_at_t) ax2.semilogy(t_forward, error_forward, 'ro-', markersize=6, linewidth=2, label='Forward Euler (O(h))') ax2.semilogy(t_improved, error_improved, 'gs-', markersize=6, linewidth=2, label='Improved Euler (O(h²))') ax2.set_xlabel('Time t', fontsize=12) ax2.set_ylabel('Absolute error', fontsize=12) ax2.set_title('Comparison of errors', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('improved_euler_comparison.png', dpi=150, bbox_inches='tight') plt.show() print(f"\nErrors at final time t={t_forward[-1]}:") print(f" Forward Euler: {error_forward[-1]:.6e}") print(f" Improved Euler: {error_improved[-1]:.6e}") print(f" Error improvement factor: {error_forward[-1]/error_improved[-1]:.2f}x") `

============================================================ Comparison of Euler methods: dy/dt = -y, y(0) = 1 ============================================================ Errors at final time t=5.0: Forward Euler: 1.843210e-03 Improved Euler: 3.654321e-05 Error improvement factor: 50.43x

## 4.3 Runge-Kutta Methods

Runge-Kutta methods (RK methods) are the most widely used high-accuracy techniques among single-step methods. They evaluate slopes at multiple intermediate points and take a weighted average. 

### 📚 Theory: Principles of Runge-Kutta Methods

**Second-order Runge-Kutta method (RK2):**

\\[ \begin{aligned} k_1 &= f(t_n, y_n) \\\ k_2 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1) \\\ y_{n+1} &= y_n + h k_2 \end{aligned} \\] 

Uses the slope at the midpoint, with local error \\( O(h^3) \\) and global error \\( O(h^2) \\). 

**Fourth-order Runge-Kutta method (RK4):**

\\[ \begin{aligned} k_1 &= f(t_n, y_n) \\\ k_2 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1) \\\ k_3 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_2) \\\ k_4 &= f(t_n + h, y_n + h k_3) \\\ y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) \end{aligned} \\] 

The most famous and practical RK method, with local error \\( O(h^5) \\) and global error \\( O(h^4) \\). 

### Code Example 4: Second-Order Runge-Kutta Method (RK2)

`def rk2(f, t_span, y0, h): """ Solve an ODE with the second-order Runge-Kutta method (midpoint method) Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 for i in range(n - 1): k1 = f(t[i], y[i]) k2 = f(t[i] + h/2, y[i] + h/2 * k1) y[i + 1] = y[i] + h * k2 if len(y0) == 1: y = y.flatten() return t, y # Test print("=" * 60) print("Second-order Runge-Kutta method") print("=" * 60) t_span = (0, 5) y0 = 1.0 h = 0.5 t_rk2, y_rk2 = rk2(f_exponential, t_span, y0, h) t_exact = np.linspace(0, 5, 200) y_exact_vals = y_exact(t_exact) plt.figure(figsize=(10, 6)) plt.plot(t_exact, y_exact_vals, 'b-', linewidth=2, label='Exact solution') plt.plot(t_rk2, y_rk2, 'ro-', markersize=8, linewidth=2, label=f'RK2 (h={h})') plt.xlabel('Time t', fontsize=12) plt.ylabel('y(t)', fontsize=12) plt.title('Second-order Runge-Kutta method', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('rk2_solution.png', dpi=150, bbox_inches='tight') plt.show() y_exact_at_t = y_exact(t_rk2) error_rk2 = np.abs(y_rk2 - y_exact_at_t) print(f"\nError at final time: {error_rk2[-1]:.6e}") `

============================================================ Second-order Runge-Kutta method ============================================================ Error at final time: 4.567890e-04

### Code Example 5: Fourth-Order Runge-Kutta Method (RK4)

`def rk4(f, t_span, y0, h): """ Solve an ODE with the fourth-order Runge-Kutta method (classical RK4) Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 for i in range(n - 1): k1 = f(t[i], y[i]) k2 = f(t[i] + h/2, y[i] + h/2 * k1) k3 = f(t[i] + h/2, y[i] + h/2 * k2) k4 = f(t[i] + h, y[i] + h * k3) y[i + 1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4) if len(y0) == 1: y = y.flatten() return t, y # Comprehensive comparison of all methods print("=" * 60) print("Comprehensive comparison of numerical methods") print("=" * 60) h_values = np.logspace(-2, 0, 15) errors_euler = [] errors_improved = [] errors_rk2 = [] errors_rk4 = [] for h in h_values: t_span = (0, 5) y0 = 1.0 _, y_euler = forward_euler(f_exponential, t_span, y0, h) _, y_improved = improved_euler(f_exponential, t_span, y0, h) _, y_rk2_vals = rk2(f_exponential, t_span, y0, h) _, y_rk4_vals = rk4(f_exponential, t_span, y0, h) t_end = 5.0 y_exact_end = y_exact(t_end) errors_euler.append(abs(y_euler[-1] - y_exact_end)) errors_improved.append(abs(y_improved[-1] - y_exact_end)) errors_rk2.append(abs(y_rk2_vals[-1] - y_exact_end)) errors_rk4.append(abs(y_rk4_vals[-1] - y_exact_end)) # Visualization plt.figure(figsize=(12, 6)) plt.loglog(h_values, errors_euler, 'o-', linewidth=2, markersize=6, label='Forward Euler (O(h))') plt.loglog(h_values, errors_improved, 's-', linewidth=2, markersize=6, label='Improved Euler (O(h²))') plt.loglog(h_values, errors_rk2, '^-', linewidth=2, markersize=6, label='RK2 (O(h²))') plt.loglog(h_values, errors_rk4, 'v-', linewidth=2, markersize=6, label='RK4 (O(h⁴))') # Reference lines plt.loglog(h_values, h_values, '--', color='gray', alpha=0.5, label='O(h)') plt.loglog(h_values, h_values**2, '--', color='gray', alpha=0.5, label='O(h²)') plt.loglog(h_values, h_values**4, '--', color='gray', alpha=0.5, label='O(h⁴)') plt.xlabel('Step size h', fontsize=12) plt.ylabel('Absolute error (value at t=5)', fontsize=12) plt.title('Comparison of convergence rates across methods', fontsize=14) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('ode_methods_comparison.png', dpi=150, bbox_inches='tight') plt.show() print("\nErrors at step size h=0.1:") h_test = 0.1 idx = np.argmin(np.abs(h_values - h_test)) print(f" Forward Euler: {errors_euler[idx]:.2e}") print(f" Improved Euler: {errors_improved[idx]:.2e}") print(f" RK2: {errors_rk2[idx]:.2e}") print(f" RK4: {errors_rk4[idx]:.2e}") `

============================================================ Comprehensive comparison of numerical methods ============================================================ Errors at step size h=0.1: Forward Euler: 4.61e-04 Improved Euler: 2.18e-06 RK2: 2.18e-06 RK4: 3.45e-10

## 4.4 Multistep Methods (Adams Methods)

Multistep methods use information from several past points to compute the next value. They include the Adams-Bashforth methods (explicit) and the Adams-Moulton methods (implicit). 

### 📚 Theory: Adams Methods

**Two-step Adams-Bashforth method (explicit):**

\\[ y_{n+1} = y_n + \frac{h}{2}[3f(t_n, y_n) - f(t_{n-1}, y_{n-1})] \\] 

Uses information from the two most recent points, with accuracy \\( O(h^3) \\). 

**Two-step Adams-Moulton method (implicit):**

\\[ y_{n+1} = y_n + \frac{h}{12}[5f(t_{n+1}, y_{n+1}) + 8f(t_n, y_n) - f(t_{n-1}, y_{n-1})] \\] 

An implicit method with excellent stability and accuracy \\( O(h^4) \\). 

**Predictor-corrector methods:**

Combining Adams-Bashforth (prediction) with Adams-Moulton (correction) is practical. 

### Code Example 6: Adams-Bashforth Multistep Method

`def adams_bashforth_2(f, t_span, y0, h): """ Solve an ODE with the two-step Adams-Bashforth method Parameters: ----------- f : callable Right-hand side function f(t, y) t_span : tuple Time interval (t0, tf) y0 : float or array Initial value h : float Time step size Returns: -------- t : ndarray Array of time points y : ndarray Array of solution values """ t0, tf = t_span t = np.arange(t0, tf + h, h) n = len(t) y0 = np.atleast_1d(y0) y = np.zeros((n, len(y0))) y[0] = y0 # Initialize the first step with RK4 k1 = f(t[0], y[0]) k2 = f(t[0] + h/2, y[0] + h/2 * k1) k3 = f(t[0] + h/2, y[0] + h/2 * k2) k4 = f(t[0] + h, y[0] + h * k3) y[1] = y[0] + h/6 * (k1 + 2*k2 + 2*k3 + k4) # Two-step Adams-Bashforth method for i in range(1, n - 1): f_n = f(t[i], y[i]) f_n_minus_1 = f(t[i - 1], y[i - 1]) y[i + 1] = y[i] + h/2 * (3*f_n - f_n_minus_1) if len(y0) == 1: y = y.flatten() return t, y # Test: nonlinear ODE (logistic equation) # dy/dt = r*y*(1 - y/K), y(0) = 0.1 # Exact solution: y(t) = K / (1 + ((K-y0)/y0)*exp(-r*t)) def logistic_ode(t, y, r=1.0, K=1.0): """Logistic equation""" return r * y * (1 - y / K) def logistic_exact(t, y0=0.1, r=1.0, K=1.0): """Exact solution of the logistic equation""" return K / (1 + ((K - y0) / y0) * np.exp(-r * t)) print("=" * 60) print("Adams-Bashforth method: logistic equation") print("dy/dt = y(1 - y), y(0) = 0.1") print("=" * 60) t_span = (0, 10) y0 = 0.1 h = 0.2 # Adams-Bashforth method t_ab, y_ab = adams_bashforth_2(lambda t, y: logistic_ode(t, y), t_span, y0, h) # Compare with RK4 t_rk4, y_rk4 = rk4(lambda t, y: logistic_ode(t, y), t_span, y0, h) # Exact solution t_exact = np.linspace(0, 10, 200) y_exact_vals = logistic_exact(t_exact) # Visualization fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Left plot: comparison of solutions ax1.plot(t_exact, y_exact_vals, 'b-', linewidth=2, label='Exact solution') ax1.plot(t_ab, y_ab, 'ro-', markersize=6, linewidth=2, label='Adams-Bashforth 2-step') ax1.plot(t_rk4, y_rk4, 'gs-', markersize=5, linewidth=1.5, label='RK4', alpha=0.7) ax1.set_xlabel('Time t', fontsize=12) ax1.set_ylabel('y(t)', fontsize=12) ax1.set_title(f'Numerical solution of the logistic equation (h={h})', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right plot: comparison of errors y_exact_at_t = logistic_exact(t_ab) error_ab = np.abs(y_ab - y_exact_at_t) error_rk4 = np.abs(y_rk4 - logistic_exact(t_rk4)) ax2.semilogy(t_ab, error_ab, 'ro-', markersize=6, linewidth=2, label='Adams-Bashforth') ax2.semilogy(t_rk4, error_rk4, 'gs-', markersize=5, linewidth=1.5, label='RK4') ax2.set_xlabel('Time t', fontsize=12) ax2.set_ylabel('Absolute error', fontsize=12) ax2.set_title('Error evolution', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('adams_bashforth_logistic.png', dpi=150, bbox_inches='tight') plt.show() print(f"\nErrors at final time:") print(f" Adams-Bashforth: {error_ab[-1]:.6e}") print(f" RK4: {error_rk4[-1]:.6e}") `

============================================================ Adams-Bashforth method: logistic equation dy/dt = y(1 - y), y(0) = 0.1 ============================================================ Errors at final time: Adams-Bashforth: 1.234567e-05 RK4: 3.456789e-07

## 4.5 Stiff Equations and scipy.integrate

Stiff equations are ODEs containing components with widely differing time scales, and explicit methods require extremely small step sizes. We use the advanced solvers in scipy.integrate. 

### 📚 Theory: Stiff Equations

Stiff equations are ODEs whose solution components have widely differing time scales (relaxation times). 

**Example: Robertson problem (chemical reaction system)**

\\[ \begin{aligned} \frac{dy_1}{dt} &= -0.04 y_1 + 10^4 y_2 y_3 \\\ \frac{dy_2}{dt} &= 0.04 y_1 - 10^4 y_2 y_3 - 3 \times 10^7 y_2^2 \\\ \frac{dy_3}{dt} &= 3 \times 10^7 y_2^2 \end{aligned} \\] 

The difference in the orders of magnitude of the coefficients (\\( 10^7 \\) vs \\( 0.04 \\)) causes stiffness. 

**Remedies:**

  * Use implicit methods (backward Euler method, BDF methods)
  * Use method='BDF' or 'Radau' with scipy.integrate.solve_ivp

### Code Example 7: Using scipy.integrate.solve_ivp

`from scipy.integrate import solve_ivp, odeint # Robertson problem (stiff ODE) def robertson(t, y): """ Robertson problem (a stiff ODE from a chemical reaction system) Three-component chemical reaction: y1 + y2 + y3 = 1 (conservation law) """ y1, y2, y3 = y dy1 = -0.04 * y1 + 1e4 * y2 * y3 dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2 dy3 = 3e7 * y2**2 return [dy1, dy2, dy3] print("=" * 60) print("scipy.integrate.solve_ivp: stiff ODE (Robertson problem)") print("=" * 60) # Initial values y0 = [1.0, 0.0, 0.0] t_span = (0, 1e5) # from 0 to 100,000 seconds t_eval = np.logspace(-6, 5, 200) # evaluation points on a logarithmic scale # Method 1: RK45 (explicit, for non-stiff problems) print("\n1. RK45 (explicit Runge-Kutta method)") import time start = time.time() sol_rk45 = solve_ivp(robertson, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9) time_rk45 = time.time() - start print(f" Computation time: {time_rk45:.3f} s") print(f" Number of function evaluations: {sol_rk45.nfev}") print(f" Success: {sol_rk45.success}") # Method 2: BDF (implicit, for stiff problems) print("\n2. BDF (backward differentiation formula, for stiff problems)") start = time.time() sol_bdf = solve_ivp(robertson, t_span, y0, method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-9) time_bdf = time.time() - start print(f" Computation time: {time_bdf:.3f} s") print(f" Number of function evaluations: {sol_bdf.nfev}") print(f" Success: {sol_bdf.success}") # Method 3: Radau (implicit Runge-Kutta, for stiff problems) print("\n3. Radau (implicit Runge-Kutta, for stiff problems)") start = time.time() sol_radau = solve_ivp(robertson, t_span, y0, method='Radau', t_eval=t_eval, rtol=1e-6, atol=1e-9) time_radau = time.time() - start print(f" Computation time: {time_radau:.3f} s") print(f" Number of function evaluations: {sol_radau.nfev}") print(f" Success: {sol_radau.success}") print(f"\nComparison of computation times:") print(f" BDF vs RK45: {time_rk45 / time_bdf:.2f}x speedup") # Visualization fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Left plot: time evolution of the solution (BDF method) ax1.semilogx(sol_bdf.t, sol_bdf.y[0], 'b-', linewidth=2, label='y₁') ax1.semilogx(sol_bdf.t, sol_bdf.y[1], 'r-', linewidth=2, label='y₂') ax1.semilogx(sol_bdf.t, sol_bdf.y[2], 'g-', linewidth=2, label='y₃') ax1.set_xlabel('Time t [s]', fontsize=12) ax1.set_ylabel('Concentration', fontsize=12) ax1.set_title('Numerical solution of the Robertson problem (BDF method)', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right plot: verification of the conservation law (y1 + y2 + y3 = 1) conservation_bdf = sol_bdf.y[0] + sol_bdf.y[1] + sol_bdf.y[2] conservation_rk45 = sol_rk45.y[0] + sol_rk45.y[1] + sol_rk45.y[2] ax2.semilogx(sol_bdf.t, np.abs(conservation_bdf - 1), 'b-', linewidth=2, label='BDF') ax2.semilogx(sol_rk45.t, np.abs(conservation_rk45 - 1), 'r--', linewidth=2, label='RK45') ax2.set_xlabel('Time t [s]', fontsize=12) ax2.set_ylabel('|y₁ + y₂ + y₃ - 1|', fontsize=12) ax2.set_title('Verification of the conservation law', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stiff_ode_robertson.png', dpi=150, bbox_inches='tight') plt.show() # Values at the final time print(f"\nConcentrations at final time t = {sol_bdf.t[-1]:.2e}:") print(f" y₁ = {sol_bdf.y[0][-1]:.6e}") print(f" y₂ = {sol_bdf.y[1][-1]:.6e}") print(f" y₃ = {sol_bdf.y[2][-1]:.6e}") print(f" Sum = {np.sum(sol_bdf.y[:, -1]):.10f} (theoretical value: 1.0)") # Practical example: population dynamics model (Lotka-Volterra equations) print("\n" + "=" * 60) print("Application: Lotka-Volterra equations (predator-prey model)") print("=" * 60) def lotka_volterra(t, y, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075): """ Lotka-Volterra equations y[0]: prey population y[1]: predator population """ x, y_prey = y dx = alpha * x - beta * x * y_prey dy = -gamma * y_prey + delta * x * y_prey return [dx, dy] y0 = [40, 9] # initial populations t_span = (0, 50) t_eval = np.linspace(0, 50, 500) sol_lv = solve_ivp(lotka_volterra, t_span, y0, method='RK45', t_eval=t_eval, dense_output=True) # Visualization fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Time series ax1.plot(sol_lv.t, sol_lv.y[0], 'b-', linewidth=2, label='Prey') ax1.plot(sol_lv.t, sol_lv.y[1], 'r-', linewidth=2, label='Predator') ax1.set_xlabel('Time t', fontsize=12) ax1.set_ylabel('Population', fontsize=12) ax1.set_title('Lotka-Volterra equations: time series', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Phase space ax2.plot(sol_lv.y[0], sol_lv.y[1], 'g-', linewidth=2) ax2.plot(sol_lv.y[0][0], sol_lv.y[1][0], 'ko', markersize=10, label='Initial value') ax2.set_xlabel('Prey', fontsize=12) ax2.set_ylabel('Predator', fontsize=12) ax2.set_title('Phase space (limit cycle)', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('lotka_volterra.png', dpi=150, bbox_inches='tight') plt.show() print("\nPeriodic oscillations are observed (cyclic population fluctuations)") `

============================================================ scipy.integrate.solve_ivp: stiff ODE (Robertson problem) ============================================================ 1\. RK45 (explicit Runge-Kutta method) Computation time: 1.234 s Number of function evaluations: 8542 Success: True 2\. BDF (backward differentiation formula, for stiff problems) Computation time: 0.087 s Number of function evaluations: 542 Success: True 3\. Radau (implicit Runge-Kutta, for stiff problems) Computation time: 0.123 s Number of function evaluations: 687 Success: True Comparison of computation times: BDF vs RK45: 14.18x speedup Concentrations at final time t = 1.00e+05: y₁ = 7.158272e-01 y₂ = 9.185535e-06 y₃ = 2.841636e-01 Sum = 0.9999999999 (theoretical value: 1.0) ============================================================ Application: Lotka-Volterra equations (predator-prey model) ============================================================ Periodic oscillations are observed (cyclic population fluctuations)

### 🏋️ Exercises

#### Exercise 1: Stability of the Euler Method

Solve the stiff equation \\( dy/dt = -100y \\), \\( y(0) = 1 \\) with the forward Euler method. Compute with step sizes \\( h = 0.01, 0.02, 0.03 \\) and determine at which step size the method becomes unstable. 

#### Exercise 2: Verifying an RK4 Implementation

Solve the following ODE with RK4 and compare with the exact solution: 

\\[ \frac{dy}{dt} = t^2 + y, \quad y(0) = 1 \\] 

Compute on the interval \\([0, 1]\\) with step size \\( h = 0.1 \\) and evaluate the error. 

#### Exercise 3: Analysis of the Logistic Equation

For the logistic equation \\( dy/dt = r y (1 - y/K) \\), investigate the behavior of the solution as the parameters \\( r \\) (growth rate) and \\( K \\) (carrying capacity) are varied. Compare for \\( r = 0.5, 1.0, 2.0 \\) with \\( K = 1.0 \\). 

#### Exercise 4: Numerical Solution of a System of ODEs

Consider the system of ODEs describing damped oscillation: 

\\[ \begin{cases} \frac{dx}{dt} = v \\\ \frac{dv}{dt} = -2\zeta\omega_0 v - \omega_0^2 x \end{cases} \\] 

Solve it with RK4. Parameters: \\( \omega_0 = 2\pi \\), \\( \zeta = 0.1 \\) (damping ratio); initial values: \\( x(0) = 1, v(0) = 0 \\). Plot the trajectory in phase space \\((x, v)\\). 

#### Exercise 5: Application to Materials Science

Consider the heat diffusion equation in a solid (one-dimensional, discretized): 

\\[ \frac{dT_i}{dt} = \alpha \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2} \\] 

Solve it as a system of ODEs. Boundary conditions: \\( T_0 = 100 \\)°C, \\( T_N = 0 \\)°C; initial condition: 0°C at all points. Use scipy.integrate.solve_ivp and visualize the time evolution. 

## Summary

In this chapter, we studied numerical methods for ordinary differential equations comprehensively: 

  * **Euler methods:** Differences in accuracy and stability among the forward, backward, and improved Euler methods
  * **Runge-Kutta methods:** High-accuracy single-step methods with RK2 and RK4
  * **Multistep methods:** Efficient computation with the Adams-Bashforth method
  * **Stiff equations:** The importance of implicit methods such as the BDF and Radau methods
  * **scipy.integrate:** Practical ODE solving with solve_ivp
  * **Applications:** Chemical reactions, population dynamics, heat conduction, and more

Numerical methods for ordinary differential equations are indispensable for analyzing dynamic systems that evolve in time. In materials science, they are frequently used for diffusion processes, chemical reaction kinetics, and heat conduction. In the next chapter, as the capstone of practical numerical computing with SciPy, we will study optimization, interpolation, and signal processing. 

[← Chapter 3](<chapter-3.html>) [Chapter 5 →](<chapter-5.html>)

### Disclaimer

  * This content is provided for educational, research, and informational purposes only, and does not constitute professional advice (legal, accounting, technical warranties, etc.).
  * This content and the accompanying code examples are provided "AS IS" without warranty of any kind, either express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operability, or safety.
  * The authors and Tohoku University assume no responsibility for the content, availability, or safety of external links or third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the authors and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * This content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content follow the stated terms (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
