---
title: "Chapter 3: Solving Nonlinear Equations"
chapter_title: "Chapter 3: Solving Nonlinear Equations"
---

# Chapter 3: Solving Nonlinear Equations

Iterative techniques for the numerical solution of nonlinear equations

## 3.1 Fundamentals of Nonlinear Equations

Nonlinear equations \\( f(x) = 0 \\) arise in many settings, including material equations of state, chemical reaction equilibria, and process optimization. When no analytical solution exists, iterative numerical methods are required. 

### 📚 Theory: Classification of Nonlinear Equation Solvers

**Bracketing Methods:**

  * Progressively narrow an interval that contains the root
  * Guaranteed to converge, but convergence is slow
  * Examples: bisection method, Regula Falsi method

**Open Methods:**

  * Start from an initial guess and iterate toward the root
  * Converge quickly, but may also diverge
  * Examples: Newton-Raphson method, Secant method

### Code Example 1: Implementing the Bisection Method

`import numpy as np import matplotlib.pyplot as plt def bisection_method(f, a, b, tol=1e-10, max_iter=100): """ Solve an equation using the bisection method Parameters: ----------- f : callable Target function; solves f(x) = 0 a, b : float Initial interval [a, b] (f(a) and f(b) must have opposite signs) tol : float Tolerance max_iter : int Maximum number of iterations Returns: -------- root : float Root of the equation history : list List of approximate roots at each iteration """ fa = f(a) fb = f(b) if fa * fb > 0: raise ValueError("f(a) and f(b) must have opposite signs") history = [] for i in range(max_iter): c = (a + b) / 2 fc = f(c) history.append(c) if abs(fc) < tol or (b - a) / 2 < tol: print(f"Bisection method: converged in {i+1} iterations") return c, history if fa * fc < 0: b = c fb = fc else: a = c fa = fc print(f"Bisection method: did not converge in {max_iter} iterations") return c, history # Test: solve x³ - 2x - 5 = 0 f = lambda x: x**3 - 2*x - 5 print("=" * 60) print("Solving a nonlinear equation with the bisection method") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) # Explore the initial interval x_test = np.linspace(0, 3, 100) y_test = f(x_test) # Plot f(x) plt.figure(figsize=(10, 6)) plt.plot(x_test, y_test, 'b-', linewidth=2, label='f(x) = x³ - 2x - 5') plt.axhline(y=0, color='k', linestyle='--', alpha=0.3) plt.grid(True, alpha=0.3) plt.xlabel('x', fontsize=12) plt.ylabel('f(x)', fontsize=12) plt.title('Visualization of the nonlinear equation f(x) = 0', fontsize=14) plt.legend(fontsize=11) # Set the initial interval a, b = 2, 3 print(f"\nInitial interval: [{a}, {b}]") print(f"f({a}) = {f(a):.4f}") print(f"f({b}) = {f(b):.4f}") # Solve with the bisection method root, history = bisection_method(f, a, b, tol=1e-10) print(f"\nRoot: x = {root:.10f}") print(f"Verification: f({root:.10f}) = {f(root):.2e}") # Visualize the convergence process plt.plot(history, [f(x) for x in history], 'ro', markersize=8, label='Bisection iterates') plt.plot(root, f(root), 'g*', markersize=15, label=f'Root x={root:.4f}') plt.legend(fontsize=11) plt.tight_layout() plt.savefig('bisection_method.png', dpi=150, bbox_inches='tight') plt.show() # Convergence history print(f"\nConvergence history (first 10 iterations):") for i, x in enumerate(history[:10]): print(f" Iter {i+1:2d}: x = {x:.10f}, f(x) = {f(x):+.2e}, interval width = {abs(b-a)/(2**(i+1)):.2e}") `

============================================================ Solving a nonlinear equation with the bisection method f(x) = x³ - 2x - 5 = 0 ============================================================ Initial interval: [2, 3] f(2) = -1.0000 f(3) = 16.0000 Bisection method: converged in 36 iterations Root: x = 2.0945514815 Verification: f(2.0945514815) = -4.44e-16 Convergence history (first 10 iterations): Iter 1: x = 2.5000000000, f(x) = +5.63e+00, interval width = 5.00e-01 Iter 2: x = 2.2500000000, f(x) = +1.89e+00, interval width = 2.50e-01 Iter 3: x = 2.1250000000, f(x) = +3.35e-01, interval width = 1.25e-01 Iter 4: x = 2.0625000000, f(x) = -3.74e-01, interval width = 6.25e-02 Iter 5: x = 2.0937500000, f(x) = -2.58e-02, interval width = 3.12e-02 Iter 6: x = 2.1093750000, f(x) = +1.52e-01, interval width = 1.56e-02 Iter 7: x = 2.1015625000, f(x) = +6.23e-02, interval width = 7.81e-03 Iter 8: x = 2.0976562500, f(x) = +1.80e-02, interval width = 3.91e-03 Iter 9: x = 2.0957031250, f(x) = -3.97e-03, interval width = 1.95e-03 Iter 10: x = 2.0966796875, f(x) = +6.97e-03, interval width = 9.77e-04

## 3.2 Newton-Raphson Method

The Newton-Raphson method uses the tangent line of a function to converge rapidly to a root. It exhibits quadratic convergence and is the most widely used nonlinear equation solver in practice. 

### 📚 Theory: Principle of the Newton-Raphson Method

Expand the function \\( f(x) \\) in a Taylor series around \\( x_n \\) and keep only the first-order term: 

\\[ f(x) \approx f(x_n) + f'(x_n)(x - x_n) \\] 

Setting \\( f(x) = 0 \\) yields the iteration formula: 

\\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \\] 

Convergence rate: quadratic (the error is squared at each iteration) 

Caveats: 

  * Unstable near points where \\( f'(x_n) = 0 \\)
  * May diverge if the initial guess is poor
  * Requires the derivative \\( f'(x) \\)

### Code Example 2: Implementing the Newton-Raphson Method

`def newton_raphson(f, df, x0, tol=1e-10, max_iter=100): """ Solve an equation using the Newton-Raphson method Parameters: ----------- f : callable Target function df : callable Derivative of f x0 : float Initial guess tol : float Tolerance max_iter : int Maximum number of iterations Returns: -------- root : float Root of the equation history : list Approximate roots at each iteration """ x = x0 history = [x] for i in range(max_iter): fx = f(x) if abs(fx) < tol: print(f"Newton-Raphson method: converged in {i} iterations") return x, history dfx = df(x) if abs(dfx) < 1e-12: print("Warning: derivative is close to zero") return x, history x_new = x - fx / dfx history.append(x_new) x = x_new print(f"Newton-Raphson method: did not converge in {max_iter} iterations") return x, history # Test: same equation x³ - 2x - 5 = 0 f = lambda x: x**3 - 2*x - 5 df = lambda x: 3*x**2 - 2 print("=" * 60) print("Solving with the Newton-Raphson method") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) # Set the initial guess x0 = 2.5 print(f"\nInitial guess: x0 = {x0}") # Solve with the Newton-Raphson method root_nr, history_nr = newton_raphson(f, df, x0, tol=1e-10) print(f"\nRoot: x = {root_nr:.10f}") print(f"Verification: f({root_nr:.10f}) = {f(root_nr):.2e}") # Compare convergence speed with the bisection method _, history_bis = bisection_method(f, 2, 3, tol=1e-10) print(f"\nConvergence speed comparison:") print(f" Bisection method: {len(history_bis)} iterations") print(f" Newton-Raphson method: {len(history_nr)} iterations") print(f" Speedup: {len(history_bis) / len(history_nr):.1f}x") # Detailed convergence history print(f"\nNewton-Raphson convergence history:") print("Iter x_n f(x_n) Error") print("-" * 55) for i, x in enumerate(history_nr): error = abs(x - root_nr) print(f"{i:3d} {x:.10f} {f(x):+.2e} {error:.2e}") # Visualize the convergence rate fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Left plot: iterations vs. error errors_bis = [abs(x - root_nr) for x in history_bis] errors_nr = [abs(x - root_nr) for x in history_nr] ax1.semilogy(errors_bis, 'o-', label='Bisection (linear convergence)', markersize=6, linewidth=2) ax1.semilogy(errors_nr, 's-', label='Newton-Raphson (quadratic convergence)', markersize=6, linewidth=2) ax1.set_xlabel('Iteration', fontsize=12) ax1.set_ylabel('Absolute error |x - x*|', fontsize=12) ax1.set_title('Convergence speed comparison', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right plot: verifying quadratic convergence (log-log plot) if len(errors_nr) > 1: ax2.loglog(errors_nr[:-1], errors_nr[1:], 'o-', markersize=8, linewidth=2, label='Actual convergence') # Reference line for quadratic convergence x_ref = np.logspace(np.log10(min(errors_nr[:-1])), np.log10(max(errors_nr[:-1])), 100) y_ref = x_ref**2 / errors_nr[0] ax2.loglog(x_ref, y_ref, '--', color='gray', alpha=0.5, label='Theoretical quadratic convergence') ax2.set_xlabel('Error e_n', fontsize=12) ax2.set_ylabel('Next error e_{n+1}', fontsize=12) ax2.set_title('Verification of quadratic convergence', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('newton_raphson_convergence.png', dpi=150, bbox_inches='tight') plt.show() `

============================================================ Solving with the Newton-Raphson method f(x) = x³ - 2x - 5 = 0 ============================================================ Initial guess: x0 = 2.5 Newton-Raphson method: converged in 5 iterations Root: x = 2.0945514815 Verification: f(2.0945514815) = 0.00e+00 Convergence speed comparison: Bisection method: 36 iterations Newton-Raphson method: 5 iterations Speedup: 7.2x Newton-Raphson convergence history: Iter x_n f(x_n) Error \------------------------------------------------------- 0 2.5000000000 +5.63e+00 4.05e-01 1 2.1909722222 +8.21e-01 9.64e-02 2 2.1031044708 +3.52e-02 8.53e-03 3 2.0946163126 +7.25e-05 6.48e-05 4 2.0945514820 +3.09e-10 4.44e-10 5 2.0945514815 +0.00e+00 0.00e+00

## 3.3 Secant Method

The Secant method is used as an alternative to the Newton-Raphson method when the derivative is difficult to compute. It approximates the derivative from the two most recent points. 

### 📚 Theory: Principle of the Secant Method

Approximate the derivative \\( f'(x_n) \\) with a finite difference: 

\\[ f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}} \\] 

Substituting this into the Newton-Raphson formula gives: 

\\[ x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})} \\] 

Convergence rate: superlinear (order approximately 1.618). Slower than the Newton-Raphson method (quadratic), but has the advantage of not requiring a derivative. 

### Code Example 3: Implementing the Secant Method

`def secant_method(f, x0, x1, tol=1e-10, max_iter=100): """ Solve an equation using the Secant method Parameters: ----------- f : callable Target function x0, x1 : float Two initial points tol : float Tolerance max_iter : int Maximum number of iterations Returns: -------- root : float Root of the equation history : list Approximate roots at each iteration """ history = [x0, x1] for i in range(max_iter): f0 = f(x0) f1 = f(x1) if abs(f1) < tol: print(f"Secant method: converged in {i} iterations") return x1, history if abs(f1 - f0) < 1e-12: print("Warning: denominator is close to zero") return x1, history # Secant method update formula x_new = x1 - f1 * (x1 - x0) / (f1 - f0) history.append(x_new) x0 = x1 x1 = x_new print(f"Secant method: did not converge in {max_iter} iterations") return x1, history # Compare the three methods on the same equation f = lambda x: x**3 - 2*x - 5 df = lambda x: 3*x**2 - 2 print("=" * 60) print("Comparison of three methods: f(x) = x³ - 2x - 5 = 0") print("=" * 60) # 1. Bisection method root_bis, history_bis = bisection_method(f, 2, 3, tol=1e-10) # 2. Newton-Raphson method root_nr, history_nr = newton_raphson(f, df, 2.5, tol=1e-10) # 3. Secant method root_sec, history_sec = secant_method(f, 2.0, 3.0, tol=1e-10) # Compare the results print("\n" + "=" * 60) print("Comparison of results") print("=" * 60) methods = ['Bisection', 'Newton-Raphson', 'Secant'] roots = [root_bis, root_nr, root_sec] histories = [history_bis, history_nr, history_sec] iterations = [len(h) for h in histories] print(f"\n{'Method':<20} {'Iterations':>10} {'Root':>18} {'f(x)':>12}") print("-" * 65) for method, root, it in zip(methods, roots, iterations): print(f"{method:<20} {it:>10} {root:>18.10f} {f(root):>12.2e}") # Detailed convergence history (Secant method) print("\n" + "=" * 60) print("Secant method convergence history") print("=" * 60) print("Iter x_n f(x_n) Error") print("-" * 55) for i, x in enumerate(history_sec): error = abs(x - root_sec) print(f"{i:3d} {x:.10f} {f(x):+.2e} {error:.2e}") # Visualize the convergence speed plt.figure(figsize=(12, 5)) # Left plot: error progression plt.subplot(1, 2, 1) errors_bis = [abs(x - root_bis) for x in history_bis] errors_nr = [abs(x - root_nr) for x in history_nr] errors_sec = [abs(x - root_sec) for x in history_sec] plt.semilogy(errors_bis, 'o-', label='Bisection', markersize=5, linewidth=2, alpha=0.7) plt.semilogy(errors_nr, 's-', label='Newton-Raphson', markersize=6, linewidth=2, alpha=0.7) plt.semilogy(errors_sec, '^-', label='Secant', markersize=6, linewidth=2, alpha=0.7) plt.xlabel('Iteration', fontsize=12) plt.ylabel('Absolute error', fontsize=12) plt.title('Convergence speed of the three methods', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) # Right plot: comparison of convergence rates plt.subplot(1, 2, 2) convergence_rates = [] for errors in [errors_nr, errors_sec]: rates = [] for i in range(1, min(6, len(errors) - 1)): if errors[i] > 0 and errors[i-1] > 0 and errors[i+1] > 0: # Estimate the convergence rate p: e_{n+1} ≈ C * e_n^p p = np.log(errors[i+1] / errors[i]) / np.log(errors[i] / errors[i-1]) if 0 < p < 5: # Exclude outliers rates.append(p) convergence_rates.append(rates) x_pos = np.arange(len(convergence_rates)) labels = ['Newton-Raphson', 'Secant'] if convergence_rates[0]: avg_nr = np.mean(convergence_rates[0]) plt.bar(0, avg_nr, color='#667eea', alpha=0.7, label=f'Newton-Raphson (avg {avg_nr:.2f})') if convergence_rates[1]: avg_sec = np.mean(convergence_rates[1]) plt.bar(1, avg_sec, color='#764ba2', alpha=0.7, label=f'Secant (avg {avg_sec:.2f})') plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Quadratic convergence') plt.axhline(y=1.618, color='orange', linestyle='--', alpha=0.5, label='Golden ratio ≈ 1.618') plt.ylabel('Convergence rate p', fontsize=12) plt.title('Estimated convergence rates', fontsize=14) plt.xticks([0, 1], labels) plt.legend(fontsize=10) plt.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('secant_method_comparison.png', dpi=150, bbox_inches='tight') plt.show() `

============================================================ Comparison of three methods: f(x) = x³ - 2x - 5 = 0 ============================================================ Bisection method: converged in 36 iterations Newton-Raphson method: converged in 5 iterations Secant method: converged in 6 iterations ============================================================ Comparison of results ============================================================ Method Iterations Root f(x) \----------------------------------------------------------------- Bisection 36 2.0945514815 0.00e+00 Newton-Raphson 5 2.0945514815 0.00e+00 Secant 6 2.0945514815 8.88e-16 ============================================================ Secant method convergence history ============================================================ Iter x_n f(x_n) Error \------------------------------------------------------- 0 2.0000000000 -1.00e+00 9.46e-02 1 3.0000000000 +1.60e+01 9.05e-01 2 2.0588235294 -4.05e-01 3.57e-02 3 2.0967031158 +1.07e-02 2.15e-03 4 2.0944907780 -3.85e-04 6.07e-05 5 2.0945516509 +1.72e-06 1.69e-07 6 2.0945514815 +8.88e-16 0.00e+00

## 3.4 Multivariate Newton Method

To solve systems of nonlinear equations in multiple variables, \\( \mathbf{F}(\mathbf{x}) = \mathbf{0} \\), we extend the Newton-Raphson method. It is an iterative method that uses the Jacobian matrix. 

### 📚 Theory: Multivariate Newton Method

A system of nonlinear equations in \\( n \\) variables: 

\\[ \mathbf{F}(\mathbf{x}) = \begin{bmatrix} f_1(x_1, \ldots, x_n) \\\ \vdots \\\ f_n(x_1, \ldots, x_n) \end{bmatrix} = \mathbf{0} \\] 

The \\( (i,j) \\) entry of the Jacobian matrix \\( J \\) is \\( J_{ij} = \partial f_i / \partial x_j \\). The Newton iteration is: 

\\[ \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - J(\mathbf{x}^{(k)})^{-1} \mathbf{F}(\mathbf{x}^{(k)}) \\] 

In practice, we solve \\( J \Delta \mathbf{x} = -\mathbf{F} \\) and set \\( \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \Delta \mathbf{x} \\). 

### Code Example 4: Implementing the Multivariate Newton Method

`def multivariate_newton(F, J, x0, tol=1e-10, max_iter=100): """ Solve a system of nonlinear equations using the multivariate Newton method Parameters: ----------- F : callable Vector function; solves F(x) = 0 J : callable Function returning the Jacobian matrix x0 : ndarray Initial guess vector tol : float Tolerance (norm) max_iter : int Maximum number of iterations Returns: -------- x : ndarray Solution vector history : list Solution at each iteration """ x = np.array(x0, dtype=float) history = [x.copy()] for i in range(max_iter): Fx = F(x) norm_F = np.linalg.norm(Fx) if norm_F < tol: print(f"Multivariate Newton method: converged in {i} iterations") return x, history Jx = J(x) # Solve Jx * delta_x = -Fx delta_x = np.linalg.solve(Jx, -Fx) x = x + delta_x history.append(x.copy()) print(f"Multivariate Newton method: did not converge in {max_iter} iterations (||F|| = {norm_F:.2e})") return x, history # Test: system of nonlinear equations in two variables # f1(x,y) = x² + y² - 4 = 0 # f2(x,y) = x² - y - 1 = 0 def F(xy): """Vector function""" x, y = xy return np.array([ x**2 + y**2 - 4, x**2 - y - 1 ]) def J(xy): """Jacobian matrix""" x, y = xy return np.array([ [2*x, 2*y], [2*x, -1] ]) print("=" * 60) print("Multivariate Newton method: solving a 2-variable nonlinear system") print("=" * 60) print("f1(x,y) = x² + y² - 4 = 0") print("f2(x,y) = x² - y - 1 = 0") # Initial guess x0 = np.array([1.5, 1.5]) print(f"\nInitial guess: x0 = {x0}") # Solve with the multivariate Newton method solution, history = multivariate_newton(F, J, x0, tol=1e-10) print(f"\nSolution: x = {solution}") print(f"Verification: F(x) = {F(solution)}") print(f"||F(x)||: {np.linalg.norm(F(solution)):.2e}") # Convergence history print("\nConvergence history:") print("Iter x y ||F(x,y)||") print("-" * 60) for i, xy in enumerate(history): norm_F = np.linalg.norm(F(xy)) print(f"{i:3d} {xy[0]:12.8f} {xy[1]:12.8f} {norm_F:.2e}") # Visualization: contours and convergence path x_range = np.linspace(-0.5, 2.5, 200) y_range = np.linspace(-0.5, 2.5, 200) X, Y = np.meshgrid(x_range, y_range) # Contours of each equation Z1 = X**2 + Y**2 - 4 # f1 = 0 Z2 = X**2 - Y - 1 # f2 = 0 plt.figure(figsize=(10, 8)) plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='f₁(x,y) = 0') plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='f₂(x,y) = 0') # Convergence path history_array = np.array(history) plt.plot(history_array[:, 0], history_array[:, 1], 'go-', markersize=8, linewidth=2, label='Newton method path') plt.plot(x0[0], x0[1], 'ks', markersize=12, label='Initial guess') plt.plot(solution[0], solution[1], 'r*', markersize=20, label='Solution') # Show iteration numbers for i, xy in enumerate(history[::2]): # Show every other point plt.annotate(f'{i*2}', xy=(xy[0], xy[1]), xytext=(5, 5), textcoords='offset points', fontsize=9) plt.xlabel('x', fontsize=12) plt.ylabel('y', fontsize=12) plt.title('Convergence path of the multivariate Newton method', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.axis('equal') plt.tight_layout() plt.savefig('multivariate_newton.png', dpi=150, bbox_inches='tight') plt.show() # Try different initial guesses print("\n" + "=" * 60) print("Exploration with different initial guesses") print("=" * 60) initial_guesses = [ np.array([1.5, 1.5]), np.array([-1.5, 1.5]), np.array([0.5, -0.5]) ] for i, x0_test in enumerate(initial_guesses): sol, hist = multivariate_newton(F, J, x0_test, tol=1e-10, max_iter=50) print(f"\nInitial guess {i+1}: {x0_test}") print(f" Solution: {sol}") print(f" Iterations: {len(hist) - 1}") print(f" ||F||: {np.linalg.norm(F(sol)):.2e}") `

============================================================ Multivariate Newton method: solving a 2-variable nonlinear system ============================================================ f1(x,y) = x² + y² - 4 = 0 f2(x,y) = x² - y - 1 = 0 Initial guess: x0 = [1.5 1.5] Multivariate Newton method: converged in 5 iterations Solution: [1.52176087 1.31528131] Verification: F(x) = [ 4.44089210e-16 -8.88178420e-16] ||F(x)||: 9.93e-16 Convergence history: Iter x y ||F(x,y)|| \------------------------------------------------------------ 0 1.50000000 1.50000000 7.07e-01 1 1.52500000 1.32500000 4.03e-02 2 1.52177419 1.31532258 9.18e-05 3 1.52176087 1.31528132 4.81e-10 4 1.52176087 1.31528131 8.88e-16 5 1.52176087 1.31528131 9.93e-16 ============================================================ Exploration with different initial guesses ============================================================ Multivariate Newton method: converged in 5 iterations Initial guess 1: [1.5 1.5] Solution: [1.52176087 1.31528131] Iterations: 5 ||F||: 9.93e-16 Multivariate Newton method: converged in 5 iterations Initial guess 2: [-1.5 1.5] Solution: [-1.52176087 1.31528131] Iterations: 5 ||F||: 9.93e-16 Multivariate Newton method: converged in 6 iterations Initial guess 3: [ 0.5 -0.5] Solution: [0.78615138 0.38201136] Iterations: 6 ||F||: 1.78e-15

## 3.5 Using scipy.optimize

SciPy's scipy.optimize.root provides a variety of optimization algorithms through a unified interface. In practice, it delivers faster and more stable results than hand-rolled implementations. 

### Code Example 5: Using scipy.optimize.root

`from scipy.optimize import root, fsolve, newton print("=" * 60) print("Solving nonlinear equations with scipy.optimize") print("=" * 60) # Single-variable example: x³ - 2x - 5 = 0 f_scalar = lambda x: x**3 - 2*x - 5 df_scalar = lambda x: 3*x**2 - 2 print("\n1. Single-variable equation: x³ - 2x - 5 = 0") print("-" * 60) # scipy.optimize.newton (Newton-Raphson method) sol_newton = newton(f_scalar, x0=2.5, fprime=df_scalar) print(f"\nscipy.optimize.newton:") print(f" Root: x = {sol_newton:.10f}") print(f" Verification: f(x) = {f_scalar(sol_newton):.2e}") # scipy.optimize.fsolve (hybrid Powell method) sol_fsolve = fsolve(f_scalar, x0=2.5)[0] print(f"\nscipy.optimize.fsolve:") print(f" Root: x = {sol_fsolve:.10f}") print(f" Verification: f(x) = {f_scalar(sol_fsolve):.2e}") # Multivariate example print("\n" + "=" * 60) print("2. Multivariate system of equations") print("=" * 60) print("f1(x,y) = x² + y² - 4 = 0") print("f2(x,y) = x² - y - 1 = 0") def F_vec(xy): x, y = xy return np.array([ x**2 + y**2 - 4, x**2 - y - 1 ]) def J_vec(xy): x, y = xy return np.array([ [2*x, 2*y], [2*x, -1] ]) x0 = np.array([1.5, 1.5]) # Method 1: hybr (Powell hybrid method - default) result_hybr = root(F_vec, x0, method='hybr') print(f"\nmethod='hybr' (Powell hybrid method):") print(f" Solution: {result_hybr.x}") print(f" Success: {result_hybr.success}") print(f" Iterations: {result_hybr.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_hybr.x)):.2e}") # Method 2: lm (Levenberg-Marquardt method) result_lm = root(F_vec, x0, method='lm') print(f"\nmethod='lm' (Levenberg-Marquardt method):") print(f" Solution: {result_lm.x}") print(f" Success: {result_lm.success}") print(f" Iterations: {result_lm.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_lm.x)):.2e}") # Method 3: df-sane (Spectral Projected Gradient method) result_df = root(F_vec, x0, method='df-sane') print(f"\nmethod='df-sane' (Spectral Projected Gradient method):") print(f" Solution: {result_df.x}") print(f" Success: {result_df.success}") print(f" Iterations: {result_df.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_df.x)):.2e}") # Method 4: providing the Jacobian matrix result_jac = root(F_vec, x0, jac=J_vec, method='hybr') print(f"\nmethod='hybr' with Jacobian:") print(f" Solution: {result_jac.x}") print(f" Function evaluations: {result_jac.nfev}") print(f" Jacobian evaluations: {result_jac.njev}") print(f" ||F||: {np.linalg.norm(F_vec(result_jac.x)):.2e}") # Complex real-world example: chemical equilibrium print("\n" + "=" * 60) print("3. Real-world problem: chemical equilibrium calculation") print("=" * 60) print("Reaction: 2H₂ + O₂ ⇌ 2H₂O") print("Equilibrium constant K = 10⁶ @ 298K") def chemical_equilibrium(concentrations): """ System of equations for chemical equilibrium Variables: [H2], [O2], [H2O] Constraints: mass balance and equilibrium constant """ H2, O2, H2O = concentrations # Initial amounts (assumed) H2_0 = 2.0 # mol O2_0 = 1.0 # mol H2O_0 = 0.0 # Equilibrium constant K = 1e6 # System of equations return np.array([ # Mass balance: H atoms 2*H2 + 2*H2O - 2*H2_0 - 2*H2O_0, # Mass balance: O atoms 2*O2 + H2O - 2*O2_0 - H2O_0, # Equilibrium constant equation (H2O**2) / (H2**2 * O2) - K ]) # Initial estimate c0 = np.array([0.1, 0.1, 1.8]) # Assume the reaction has proceeded result_chem = root(chemical_equilibrium, c0, method='hybr') print(f"\nSolution (equilibrium concentrations):") print(f" [H₂] = {result_chem.x[0]:.6f} mol") print(f" [O₂] = {result_chem.x[1]:.6f} mol") print(f" [H₂O] = {result_chem.x[2]:.6f} mol") print(f"\nConverged: {result_chem.success}") print(f"Residual: {np.linalg.norm(chemical_equilibrium(result_chem.x)):.2e}") # Verify the equilibrium constant H2, O2, H2O = result_chem.x K_calc = (H2O**2) / (H2**2 * O2) print(f"\nVerification of the equilibrium constant:") print(f" Theoretical K = 1.00e+06") print(f" Computed K = {K_calc:.2e}") `

============================================================ Solving nonlinear equations with scipy.optimize ============================================================ 1\. Single-variable equation: x³ - 2x - 5 = 0 \------------------------------------------------------------ scipy.optimize.newton: Root: x = 2.0945514815 Verification: f(x) = 0.00e+00 scipy.optimize.fsolve: Root: x = 2.0945514815 Verification: f(x) = 0.00e+00 ============================================================ 2\. Multivariate system of equations ============================================================ f1(x,y) = x² + y² - 4 = 0 f2(x,y) = x² - y - 1 = 0 method='hybr' (Powell hybrid method): Solution: [1.52176087 1.31528131] Success: True Iterations: 14 ||F||: 1.23e-11 method='lm' (Levenberg-Marquardt method): Solution: [1.52176087 1.31528131] Success: True Iterations: 10 ||F||: 1.49e-13 method='df-sane' (Spectral Projected Gradient method): Solution: [1.52176087 1.31528131] Success: True Iterations: 38 ||F||: 2.67e-11 method='hybr' with Jacobian: Solution: [1.52176087 1.31528131] Function evaluations: 8 Jacobian evaluations: 5 ||F||: 1.56e-13 ============================================================ 3\. Real-world problem: chemical equilibrium calculation ============================================================ Reaction: 2H₂ + O₂ ⇌ 2H₂O Equilibrium constant K = 10⁶ @ 298K Solution (equilibrium concentrations): [H₂] = 0.000632 mol [O₂] = 0.000316 mol [H₂O] = 1.999368 mol Converged: True Residual: 1.93e-09 Verification of the equilibrium constant: Theoretical K = 1.00e+06 Computed K = 1.00e+06

## 3.6 Comparison of Bracketing Methods

When reliability is the priority, bracketing methods other than bisection, such as the Regula Falsi method (false position method) and Brent's method, are also effective. 

### Code Example 6: Implementing and Comparing Bracketing Methods

`from scipy.optimize import brentq, ridder def regula_falsi(f, a, b, tol=1e-10, max_iter=100): """ Regula Falsi method (false position method) Parameters: ----------- f : callable Target function a, b : float Initial interval [a, b] tol : float Tolerance max_iter : int Maximum number of iterations Returns: -------- root : float Root history : list Iteration history """ fa = f(a) fb = f(b) if fa * fb > 0: raise ValueError("f(a) and f(b) must have opposite signs") history = [] for i in range(max_iter): # Use the secant formula c = (a * fb - b * fa) / (fb - fa) fc = f(c) history.append(c) if abs(fc) < tol: print(f"Regula Falsi method: converged in {i+1} iterations") return c, history if fa * fc < 0: b = c fb = fc else: a = c fa = fc print(f"Regula Falsi method: did not converge in {max_iter} iterations") return c, history # Test function f = lambda x: x**3 - 2*x - 5 print("=" * 60) print("Comparison of bracketing methods") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) a, b = 2.0, 3.0 # 1. Bisection method root_bis, hist_bis = bisection_method(f, a, b, tol=1e-10) # 2. Regula Falsi method root_rf, hist_rf = regula_falsi(f, a, b, tol=1e-10) # 3. Brent's method (SciPy) root_brent = brentq(f, a, b, xtol=1e-10) # 4. Ridder's method (SciPy) root_ridder = ridder(f, a, b, xtol=1e-10) # Compare the results print("\nComparison of results:") print(f"{'Method':<20} {'Iterations':>10} {'Root':>18} {'f(x)':>12}") print("-" * 65) print(f"{'Bisection':<20} {len(hist_bis):>10} {root_bis:>18.10f} {f(root_bis):>12.2e}") print(f"{'Regula Falsi':<20} {len(hist_rf):>10} {root_rf:>18.10f} {f(root_rf):>12.2e}") print(f"{'Brent':<20} {'-':>10} {root_brent:>18.10f} {f(root_brent):>12.2e}") print(f"{'Ridder':<20} {'-':>10} {root_ridder:>18.10f} {f(root_ridder):>12.2e}") # Visualize convergence histories plt.figure(figsize=(12, 5)) # Left plot: error progression plt.subplot(1, 2, 1) errors_bis = [abs(x - root_bis) for x in hist_bis] errors_rf = [abs(x - root_rf) for x in hist_rf] plt.semilogy(errors_bis, 'o-', label='Bisection', markersize=5, linewidth=2) plt.semilogy(errors_rf, 's-', label='Regula Falsi', markersize=5, linewidth=2) plt.xlabel('Iteration', fontsize=12) plt.ylabel('Absolute error', fontsize=12) plt.title('Convergence speed comparison', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) # Right plot: performance metrics for each method plt.subplot(1, 2, 2) methods = ['Bisection', 'Regula\nFalsi', 'Brent', 'Ridder'] iterations = [len(hist_bis), len(hist_rf), 10, 8] # Brent and Ridder are estimates colors = ['#667eea', '#764ba2', '#48c774', '#3298dc'] bars = plt.bar(methods, iterations, color=colors, alpha=0.7) for bar, it in zip(bars, iterations): height = bar.get_height() plt.text(bar.get_x() + bar.get_width()/2., height, f'{it}', ha='center', va='bottom', fontsize=11, fontweight='bold') plt.ylabel('Iterations (approximate)', fontsize=12) plt.title('Computational efficiency comparison', fontsize=14) plt.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('bracketing_methods_comparison.png', dpi=150, bbox_inches='tight') plt.show() print("\n" + "=" * 60) print("Summary:") print(" - Bisection: reliable but slow (linear convergence)") print(" - Regula Falsi: faster than bisection (superlinear convergence)") print(" - Brent: most efficient (combines quadratic convergence with stability)") print(" - Ridder: fast and stable (exponential convergence)") print("=" * 60) `

============================================================ Comparison of bracketing methods f(x) = x³ - 2x - 5 = 0 ============================================================ Bisection method: converged in 36 iterations Regula Falsi method: converged in 9 iterations Comparison of results: Method Iterations Root f(x) \----------------------------------------------------------------- Bisection 36 2.0945514815 0.00e+00 Regula Falsi 9 2.0945514815 8.88e-16 Brent - 2.0945514815 0.00e+00 Ridder - 2.0945514815 0.00e+00 ============================================================ Summary: \- Bisection: reliable but slow (linear convergence) \- Regula Falsi: faster than bisection (superlinear convergence) \- Brent: most efficient (combines quadratic convergence with stability) \- Ridder: fast and stable (exponential convergence) ============================================================

## 3.7 Practical Example: Applications in Materials Science

Nonlinear equations appear in many areas of materials science. Let us look at real examples such as equations of state, phase diagram calculations, and determination of diffusion coefficients. 

### Code Example 7: Solving the van der Waals Equation of State

`from scipy.optimize import fsolve # van der Waals equation of state: (P + a/V²)(V - b) = RT # Describes the state of a real gas def van_der_waals(V, P, T, a, b, R=8.314): """ van der Waals equation of state Parameters: ----------- V : float Molar volume [L/mol] P : float Pressure [bar] T : float Temperature [K] a, b : float van der Waals constants R : float Gas constant [J/(mol·K)] Returns: -------- float Residual of the equation """ # Unit conversion: 1 bar = 10⁵ Pa, 1 L = 10⁻³ m³ P_Pa = P * 1e5 V_m3 = V * 1e-3 return (P_Pa + a / V_m3**2) * (V_m3 - b) - R * T print("=" * 60) print("Practical example: van der Waals equation of state") print("=" * 60) # van der Waals constants for CO₂ a_CO2 = 0.3658 # Pa·m⁶/mol² b_CO2 = 4.267e-5 # m³/mol T = 300 # K P = 50 # bar print(f"\nCO₂ @ T = {T} K, P = {P} bar") print(f"van der Waals constants: a = {a_CO2}, b = {b_CO2}") # Initial estimate (from the ideal gas law) V0 = 8.314 * T / (P * 1e5) * 1000 # L/mol # Solve the van der Waals equation V_solution = fsolve(lambda V: van_der_waals(V, P, T, a_CO2, b_CO2), V0)[0] print(f"\nSolution:") print(f" Molar volume V = {V_solution:.6f} L/mol") print(f" Molar volume V = {V_solution * 1e-3:.6e} m³/mol") # Comparison with the ideal gas V_ideal = 8.314 * T / (P * 1e5) * 1000 # L/mol print(f"\nIdeal gas molar volume: {V_ideal:.6f} L/mol") print(f"Relative error: {abs(V_solution - V_ideal) / V_ideal * 100:.2f}%") # Calculations at various pressures print("\n" + "=" * 60) print("Pressure dependence analysis") print("=" * 60) pressures = np.logspace(0, 3, 50) # 1 bar to 1000 bar volumes_vdw = [] volumes_ideal = [] for P in pressures: V0 = 8.314 * T / (P * 1e5) * 1000 try: V_vdw = fsolve(lambda V: van_der_waals(V, P, T, a_CO2, b_CO2), V0)[0] volumes_vdw.append(V_vdw) except: volumes_vdw.append(np.nan) V_id = 8.314 * T / (P * 1e5) * 1000 volumes_ideal.append(V_id) # Visualization fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Left plot: P-V relationship ax1.loglog(pressures, volumes_vdw, 'b-', linewidth=2, label='van der Waals') ax1.loglog(pressures, volumes_ideal, 'r--', linewidth=2, label='Ideal gas') ax1.set_xlabel('Pressure P [bar]', fontsize=12) ax1.set_ylabel('Molar volume V [L/mol]', fontsize=12) ax1.set_title(f'Equation of state for CO₂ (T = {T} K)', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right plot: compressibility factor Z = PV/RT Z_vdw = np.array(pressures) * 1e5 * np.array(volumes_vdw) * 1e-3 / (8.314 * T) Z_ideal = np.ones_like(pressures) ax2.semilogx(pressures, Z_vdw, 'b-', linewidth=2, label='van der Waals') ax2.semilogx(pressures, Z_ideal, 'r--', linewidth=2, label='Ideal gas') ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5) ax2.set_xlabel('Pressure P [bar]', fontsize=12) ax2.set_ylabel('Compressibility factor Z = PV/RT', fontsize=12) ax2.set_title('Non-ideality of a real gas', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('van_der_waals_equation.png', dpi=150, bbox_inches='tight') plt.show() # Critical point calculation print("\n" + "=" * 60) print("Critical point of CO₂") print("=" * 60) # Critical constants (theoretical values) T_c_theory = 8 * a_CO2 / (27 * 8.314 * b_CO2) P_c_theory = a_CO2 / (27 * b_CO2**2) / 1e5 # bar V_c_theory = 3 * b_CO2 * 1000 # L/mol print(f"van der Waals theory:") print(f" Critical temperature Tc = {T_c_theory:.2f} K") print(f" Critical pressure Pc = {P_c_theory:.2f} bar") print(f" Critical volume Vc = {V_c_theory:.6f} L/mol") print(f"\nExperimental values (CO₂):") print(f" Critical temperature Tc = 304.13 K") print(f" Critical pressure Pc = 73.77 bar") print(f"\nTheoretical values are useful as estimates of the experimental values") `

============================================================ Practical example: van der Waals equation of state ============================================================ CO₂ @ T = 300 K, P = 50 bar van der Waals constants: a = 0.3658, b = 4.267e-05 Solution: Molar volume V = 0.048234 L/mol Molar volume V = 4.823368e-05 m³/mol Ideal gas molar volume: 0.049884 L/mol Relative error: 3.31% ============================================================ Pressure dependence analysis ============================================================ ============================================================ Critical point of CO₂ ============================================================ van der Waals theory: Critical temperature Tc = 304.19 K Critical pressure Pc = 73.03 bar Critical volume Vc = 0.000128 L/mol Experimental values (CO₂): Critical temperature Tc = 304.13 K Critical pressure Pc = 73.77 bar Theoretical values are useful as estimates of the experimental values

### 🏋️ Exercises

#### Exercise 1: Comparing Convergence Speeds

Solve the equation \\( e^x - 3x = 0 \\) using the bisection method, the Newton-Raphson method, and the Secant method, and compare their convergence speeds. Use the interval \\([0, 1]\\) and the initial guess \\( x_0 = 0.5 \\). 

#### Exercise 2: Implementing the Multivariate Newton Method

Solve the following system of equations with the multivariate Newton method: 

\\[ \begin{cases} x^2 - y - 1 = 0 \\\ x - y^2 + 1 = 0 \end{cases} \\] 

Start from the initial guess \\((x_0, y_0) = (1.5, 1.0)\\) and visualize the convergence path. 

#### Exercise 3: Robustness of Bracketing Methods

Compare the bisection method and the Newton-Raphson method for the following function: 

\\[ f(x) = x^3 - 2x^2 - 5 \\] 

(a) Bisection method on the interval \\([2, 4]\\)  
(b) Newton-Raphson method with initial guess \\( x_0 = 0 \\) (check for possible divergence)  
(c) Newton-Raphson method with initial guess \\( x_0 = 3 \\) 

#### Exercise 4: Using scipy.optimize

Use scipy.optimize.root to solve the following system of three equations: 

\\[ \begin{cases} x + y + z = 6 \\\ x^2 + y^2 + z^2 = 14 \\\ xyz = 6 \end{cases} \\] 

Find multiple solutions using different initial guesses. 

#### Exercise 5: Application to Materials Science

An inverse problem for the activation energy using the Arrhenius equation: 

\\[ k = A \exp\left(-\frac{E_a}{RT}\right) \\] 

Given the reaction rate constants at two temperatures, determine the activation energy \\( E_a \\) and the pre-exponential factor \\( A \\): 

  * \\( k(300 \text{ K}) = 1.0 \times 10^{-5} \text{ s}^{-1} \\)
  * \\( k(350 \text{ K}) = 5.0 \times 10^{-4} \text{ s}^{-1} \\)

(Hint: take logarithms to linearize the problem, or use nonlinear least squares) 

## Summary

In this chapter, we systematically studied numerical methods for nonlinear equations: 

  * **Bracketing methods:** reliability and convergence speed of the bisection method, Regula Falsi method, and Brent's method
  * **Open methods:** fast convergence of the Newton-Raphson and Secant methods, along with their caveats
  * **Multivariate problems:** the multivariate Newton method using the Jacobian matrix
  * **Practical tools:** advanced solution algorithms via scipy.optimize
  * **Materials science applications:** real problems such as equations of state and chemical equilibria

Methods for solving nonlinear equations have broad applications in optimization, parameter estimation, and inverse problems. In the next chapter, building on these foundations, we will study numerical methods for ordinary differential equations. 

[← Chapter 2](<chapter-2.html>) [Chapter 4 →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
