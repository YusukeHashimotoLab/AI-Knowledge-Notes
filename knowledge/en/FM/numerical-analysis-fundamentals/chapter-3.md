---
title: "Chapter 3: Nonlinear equation Solution Methods"
chapter_title: "Chapter 3: Nonlinear equation Solution Methods"
---

# Chapter 3: Nonlinear equation Solution Methods

This chapter covers Nonlinear equation Solution Methods. You will learn essential concepts and techniques.

Numerical techniques for solving nonlinear equations using iterative methods

## 3.1 Fundamentals of Nonlinear equations

Nonlinear equation \\( f(x) = 0 \\) appears in various contexts such as equation of state for materials, chemical reaction equilibrium, and process optimization. When analytical solutions are not available, numerical methods using iterative approaches are required. 

### 📚 Theory: Nonlinear equation classification and 

**Bracketing Methods:**

  * Progressively narrow the interval containing the solution
  * Guaranteed convergence but slow convergence rate
  * Examples: Bisection method, Regula Falsi method

**Open Methods:**

  * Start from initial value and converge to solution
  * Fast convergence but may diverge
  * Examples: Newton-Raphson method, Secant method

### Code Example 1: Implementation of Bisection Method
    
    
    import numpy as np import matplotlib.pyplot as plt def bisection_method(f, a, b, tol=1e-10, max_iter=100): """ Solving equations using bisection method Parameters: ----------- f : callable Target function f(x) = 0 Solve a, b : float Initial interval [a, b] (f(a)and f(b) must have opposite signs) tol : float Tolerance max_iter : int Maximum iterations Returns: -------- root : float Solution of the equation history : list List of approximate solutions at each iteration """ fa = f(a) fb = f(b) if fa * fb > 0: raise Valueerror("f(a) and f(b) must have opposite signs") history = [] for i in range(max_iter): c = (a + b) / 2 fc = f(c) history.append(c) if abs(fc) < tol or (b - a) / 2 < tol: print(f"Bisection method: {i+1}converged in iterations") return c, history if fa * fc < 0: b = c fb = fc else: a = c fa = fc print(f"Bisection method: {max_iter}did not converge in iterations") return c, history # Test: x³ - 2x - 5 = 0 Solve f = lambda x: x**3 - 2*x - 5 print("=" * 60) print("bisection method Nonlinear equation solution") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) # Initial interval x_test = np.linspace(0, 3, 100) y_test = f(x_test) # f(x)plot of plt.figure(figsize=(10, 6)) plt.plot(x_test, y_test, 'b-', linewidth=2, label='f(x) = x³ - 2x - 5') plt.axhline(y=0, color='k', linestyle='--', alpha=0.3) plt.grid(True, alpha=0.3) plt.xlabel('x', fontsize=12) plt.ylabel('f(x)', fontsize=12) plt.title('Nonlinear equation f(x) = 0 Visualization of', fontsize=14) plt.legend(fontsize=11) # Initial interval a, b = 2, 3 print(f"\nInitial interval: [{a}, {b}]") print(f"f({a}) = {f(a):.4f}") print(f"f({b}) = {f(b):.4f}") # bisection method solve root, history = bisection_method(f, a, b, tol=1e-10) print(f"\nSolution: x = {root:.10f}") print(f"Verification: f({root:.10f}) = {f(root):.2e}") # Visualize convergence process plt.plot(history, [f(x) for x in history], 'ro', markersize=8, label='iteration points of bisection method') plt.plot(root, f(root), 'g*', markersize=15, label=f' x={root:.4f}') plt.legend(fontsize=11) plt.tight_layout() plt.savefig('bisection_method.png', dpi=150, bbox_inches='tight') plt.show() # convergence history print(f"\nConvergence History (First 10 iterations):") for i, x in enumerate(history[:10]): print(f" iteration{i+1:2d}: x = {x:.10f}, f(x) = {f(x):+.2e}, interval width = {abs(b-a)/(2**(i+1)):.2e}") 

============================================================ bisection method Nonlinear equation solution f(x) = x³ - 2x - 5 = 0 ============================================================ Initial interval: [2, 3] f(2) = -1.0000 f(3) = 16.0000 Bisection method: 36converged in iterations Solution: x = 2.0945514815 Verification: f(2.0945514815) = -4.44e-16 Convergence History (First 10 iterations): iteration 1: x = 2.5000000000, f(x) = +5.63e+00, interval width = 5.00e-01 iteration 2: x = 2.2500000000, f(x) = +1.89e+00, interval width = 2.50e-01 iteration 3: x = 2.1250000000, f(x) = +3.35e-01, interval width = 1.25e-01 iteration 4: x = 2.0625000000, f(x) = -3.74e-01, interval width = 6.25e-02 iteration 5: x = 2.0937500000, f(x) = -2.58e-02, interval width = 3.12e-02 iteration 6: x = 2.1093750000, f(x) = +1.52e-01, interval width = 1.56e-02 iteration 7: x = 2.1015625000, f(x) = +6.23e-02, interval width = 7.81e-03 iteration 8: x = 2.0976562500, f(x) = +1.80e-02, interval width = 3.91e-03 iteration 9: x = 2.0957031250, f(x) = -3.97e-03, interval width = 1.95e-03 iteration10: x = 2.0966796875, f(x) = +6.97e-03, interval width = 9.77e-04

## 3.2 Newton-Raphson Method

Newton-Raphson 、function using fast convergence do method 。2convergencehas、widely usedNonlinear equation Solution Methods 。 

### 📚 Theory: Principle of Newton-Raphson Method

function \\( f(x) \\) \\( x_n \\) Taylor expansion around, approximated to first order: 

\\[ f(x) \approx f(x_n) + f'(x_n)(x - x_n) \\] 

\\( f(x) = 0 \\) and and 、 iterative formula : 

\\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \\] 

convergence: 2convergence（error iteration and 2） 

Notes: 

  * \\( f'(x_n) = 0 \\) is unstable near
  * May diverge with poor initial values
  * function \\( f'(x) \\) is required

### Code Example 2: Implementation of Newton-Raphson Method
    
    
    def newton_raphson(f, df, x0, tol=1e-10, max_iter=100): """ Solving equations using Newton-Raphson method Parameters: ----------- f : callable Target function df : callable f function x0 : float initial value tol : float Tolerance max_iter : int Maximum iterations Returns: -------- root : float Solution of the equation history : list iteration approximate solution """ x = x0 history = [x] for i in range(max_iter): fx = f(x) if abs(fx) < tol: print(f"Newton-Raphson method: {i}converged in iterations") return x, history dfx = df(x) if abs(dfx) < 1e-12: print("Warning: function ") return x, history x_new = x - fx / dfx history.append(x_new) x = x_new print(f"Newton-Raphson method: {max_iter}did not converge in iterations") return x, history # Test: Same equation x³ - 2x - 5 = 0 f = lambda x: x**3 - 2*x - 5 df = lambda x: 3*x**2 - 2 print("=" * 60) print("Solution using Newton-Raphson Method") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) # initial value x0 = 2.5 print(f"\ninitial value: x0 = {x0}") # Solve using Newton-Raphson method root_nr, history_nr = newton_raphson(f, df, x0, tol=1e-10) print(f"\nSolution: x = {root_nr:.10f}") print(f"Verification: f({root_nr:.10f}) = {f(root_nr):.2e}") # bisection method and convergence _, history_bis = bisection_method(f, 2, 3, tol=1e-10) print(f"\nConvergence Speed Comparison:") print(f" Bisection method: {len(history_bis)} iterations") print(f" Newton-Raphson method: {len(history_nr)} iterations") print(f" Speedup: {len(history_bis) / len(history_nr):.1f}times") # convergence history print(f"\nConvergence History of Newton-Raphson Method:") print("iteration x_n f(x_n) error") print("-" * 55) for i, x in enumerate(history_nr): error = abs(x - root_nr) print(f"{i:3d} {x:.10f} {f(x):+.2e} {error:.2e}") # convergenceVisualization of fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # : iterationiterations and error errors_bis = [abs(x - root_nr) for x in history_bis] errors_nr = [abs(x - root_nr) for x in history_nr] ax1.semilogy(errors_bis, 'o-', label='Bisection (Linear Convergence)', markersize=6, linewidth=2) ax1.semilogy(errors_nr, 's-', label='Newton-Raphson (Quadratic Convergence)', markersize=6, linewidth=2) ax1.set_xlabel('iterationiterations', fontsize=12) ax1.set_ylabel('error |x - x*|', fontsize=12) ax1.set_title('Comparison of Convergence Speed', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # Right: Verification of Quadratic Convergence (log-log plot) if len(errors_nr) > 1: ax2.loglog(errors_nr[:-1], errors_nr[1:], 'o-', markersize=8, linewidth=2, label='Actual convergence') # 2convergence x_ref = np.logspace(np.log10(min(errors_nr[:-1])), np.log10(max(errors_nr[:-1])), 100) y_ref = x_ref**2 / errors_nr[0] ax2.loglog(x_ref, y_ref, '--', color='gray', alpha=0.5, label='Theoretical quadratic convergence') ax2.set_xlabel('error e_n', fontsize=12) ax2.set_ylabel(' error e_{n+1}', fontsize=12) ax2.set_title('Verification of Quadratic Convergence', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('newton_raphson_convergence.png', dpi=150, bbox_inches='tight') plt.show() 

============================================================ Solution using Newton-Raphson Method f(x) = x³ - 2x - 5 = 0 ============================================================ initial value: x0 = 2.5 Newton-Raphson method: 5converged in iterations Solution: x = 2.0945514815 Verification: f(2.0945514815) = 0.00e+00 Convergence Speed Comparison: Bisection method: 36 iterations Newton-Raphson method: 5 iterations Speedup: 7.2times Convergence History of Newton-Raphson Method: iteration x_n f(x_n) error ------------------------------------------------------- 0 2.5000000000 +5.63e+00 4.05e-01 1 2.1909722222 +8.21e-01 9.64e-02 2 2.1031044708 +3.52e-02 8.53e-03 3 2.0946163126 +7.25e-05 6.48e-05 4 2.0945514820 +3.09e-10 4.44e-10 5 2.0945514815 +0.00e+00 0.00e+00

## 3.3 Secant Method

Secant 、function calculation Newton-Raphson and 。past2 from function 。 

### 📚 Theory: Principle of Secant Method

function \\( f'(x_n) \\) : 

\\[ f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}} \\] 

Newton-Raphson do and : 

\\[ x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})} \\] 

convergence: convergence（1.618）。Newton-Raphson（2） than slow 、functionnot required and 。 

### Code Example 3: Implementation of Secant Method
    
    
    def secant_method(f, x0, x1, tol=1e-10, max_iter=100): """ Secant equation solution Parameters: ----------- f : callable Target function x0, x1 : float initial value2 tol : float Tolerance max_iter : int Maximum iterations Returns: -------- root : float Solution of the equation history : list iteration approximate solution """ history = [x0, x1] for i in range(max_iter): f0 = f(x0) f1 = f(x1) if abs(f1) < tol: print(f"Secant: {i}converged in iterations") return x1, history if abs(f1 - f0) < 1e-12: print("Warning: ") return x1, history # Secant x_new = x1 - f1 * (x1 - x0) / (f1 - f0) history.append(x_new) x0 = x1 x1 = x_new print(f"Secant: {max_iter}did not converge in iterations") return x1, history # Same equation 3 method f = lambda x: x**3 - 2*x - 5 df = lambda x: 3*x**2 - 2 print("=" * 60) print("3 method : f(x) = x³ - 2x - 5 = 0") print("=" * 60) # 1. bisection method root_bis, history_bis = bisection_method(f, 2, 3, tol=1e-10) # 2. Newton-Raphson root_nr, history_nr = newton_raphson(f, df, 2.5, tol=1e-10) # 3. Secant root_sec, history_sec = secant_method(f, 2.0, 3.0, tol=1e-10) # print("\n" + "=" * 60) print(" ") print("=" * 60) methods = ['bisection method', 'Newton-Raphson', 'Secant'] roots = [root_bis, root_nr, root_sec] histories = [history_bis, history_nr, history_sec] iterations = [len(h) for h in histories] print(f"\n{'method':<20} {'iterationiterations':>10} {'':>18} {'f(x)':>12}") print("-" * 65) for method, root, it in zip(methods, roots, iterations): print(f"{method:<20} {it:>10} {root:>18.10f} {f(root):>12.2e}") # convergence history （Secant） print("\n" + "=" * 60) print("Secant convergence history") print("=" * 60) print("iteration x_n f(x_n) error") print("-" * 55) for i, x in enumerate(history_sec): error = abs(x - root_sec) print(f"{i:3d} {x:.10f} {f(x):+.2e} {error:.2e}") # convergenceVisualization of plt.figure(figsize=(12, 5)) # : error plt.subplot(1, 2, 1) errors_bis = [abs(x - root_bis) for x in history_bis] errors_nr = [abs(x - root_nr) for x in history_nr] errors_sec = [abs(x - root_sec) for x in history_sec] plt.semilogy(errors_bis, 'o-', label='bisection method', markersize=5, linewidth=2, alpha=0.7) plt.semilogy(errors_nr, 's-', label='Newton-Raphson', markersize=6, linewidth=2, alpha=0.7) plt.semilogy(errors_sec, '^-', label='Secant', markersize=6, linewidth=2, alpha=0.7) plt.xlabel('iterationiterations', fontsize=12) plt.ylabel('error', fontsize=12) plt.title('3 method convergence', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) # : convergence plt.subplot(1, 2, 2) convergence_rates = [] for errors in [errors_nr, errors_sec]: rates = [] for i in range(1, min(6, len(errors) - 1)): if errors[i] > 0 and errors[i-1] > 0 and errors[i+1] > 0: # convergence p : e_{n+1} ≈ C * e_n^p p = np.log(errors[i+1] / errors[i]) / np.log(errors[i] / errors[i-1]) if 0 < p < 5: # rates.append(p) convergence_rates.append(rates) x_pos = np.arange(len(convergence_rates)) labels = ['Newton-Raphson', 'Secant'] if convergence_rates[0]: avg_nr = np.mean(convergence_rates[0]) plt.bar(0, avg_nr, color='#667eea', alpha=0.7, label=f'Newton-Raphson ( {avg_nr:.2f})') if convergence_rates[1]: avg_sec = np.mean(convergence_rates[1]) plt.bar(1, avg_sec, color='#764ba2', alpha=0.7, label=f'Secant ( {avg_sec:.2f})') plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='2convergence') plt.axhline(y=1.618, color='orange', linestyle='--', alpha=0.5, label=' ≈ 1.618') plt.ylabel('convergence p', fontsize=12) plt.title('convergence ', fontsize=14) plt.xticks([0, 1], labels) plt.legend(fontsize=10) plt.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('secant_method_comparison.png', dpi=150, bbox_inches='tight') plt.show() 

============================================================ 3 method : f(x) = x³ - 2x - 5 = 0 ============================================================ Bisection method: 36converged in iterations Newton-Raphson method: 5converged in iterations Secant: 6converged in iterations ============================================================ ============================================================ method iterationiterations f(x) ----------------------------------------------------------------- bisection method 36 2.0945514815 0.00e+00 Newton-Raphson 5 2.0945514815 0.00e+00 Secant 6 2.0945514815 8.88e-16 ============================================================ Secant convergence history ============================================================ iteration x_n f(x_n) error ------------------------------------------------------- 0 2.0000000000 -1.00e+00 9.46e-02 1 3.0000000000 +1.60e+01 9.05e-01 2 2.0588235294 -4.05e-01 3.57e-02 3 2.0967031158 +1.07e-02 2.15e-03 4 2.0944907780 -3.85e-04 6.07e-05 5 2.0945516509 +1.72e-06 1.69e-07 6 2.0945514815 +8.88e-16 0.00e+00

## 3.4 Multivariate Newton Method

equation \\( \mathbf{F}(\mathbf{x}) = \mathbf{0} \\) Solve 、Newton-Raphson 。Jacobiiteration 。 

### 📚 Theory: Multivariate Newton Method

\\( n \\) equation: 

\\[ \mathbf{F}(\mathbf{x}) = \begin{bmatrix} f_1(x_1, \ldots, x_n) \\\ \vdots \\\ f_n(x_1, \ldots, x_n) \end{bmatrix} = \mathbf{0} \\] 

Jacobi \\( J \\) \\( (i,j) \\) \\( J_{ij} = \partial f_i / \partial x_j \\)。Newton iterative formula: 

\\[ \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - J(\mathbf{x}^{(k)})^{-1} \mathbf{F}(\mathbf{x}^{(k)}) \\] 

implementation 、\\( J \Delta \mathbf{x} = -\mathbf{F} \\) \\( \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \Delta \mathbf{x} \\) and 。 

### 4: Multivariate Newton Method implementation
    
    
    def multivariate_newton(F, J, x0, tol=1e-10, max_iter=100): """ Multivariate Newton Method Nonlinear equation solution Parameters: ----------- F : callable function F(x) = 0 Solve J : callable Jacobifunction x0 : ndarray initial value tol : float Tolerance（） max_iter : int Maximum iterations Returns: -------- x : ndarray history : list iteration """ x = np.array(x0, dtype=float) history = [x.copy()] for i in range(max_iter): Fx = F(x) norm_F = np.linalg.norm(Fx) if norm_F < tol: print(f"Multivariate Newton Method: {i}converged in iterations") return x, history Jx = J(x) # Jx * delta_x = -Fx Solve delta_x = np.linalg.solve(Jx, -Fx) x = x + delta_x history.append(x.copy()) print(f"Multivariate Newton Method: {max_iter}did not converge in iterations（||F|| = {norm_F:.2e}）") return x, history # Test: 2 equation # f1(x,y) = x² + y² - 4 = 0 # f2(x,y) = x² - y - 1 = 0 def F(xy): """function""" x, y = xy return np.array([ x**2 + y**2 - 4, x**2 - y - 1 ]) def J(xy): """Jacobi""" x, y = xy return np.array([ [2*x, 2*y], [2*x, -1] ]) print("=" * 60) print("Multivariate Newton Method: 2equation solution") print("=" * 60) print("f1(x,y) = x² + y² - 4 = 0") print("f2(x,y) = x² - y - 1 = 0") # initial value x0 = np.array([1.5, 1.5]) print(f"\ninitial value: x0 = {x0}") # Multivariate Newton Method solve solution, history = multivariate_newton(F, J, x0, tol=1e-10) print(f"\nSolution: x = {solution}") print(f"Verification: F(x) = {F(solution)}") print(f"||F(x)||: {np.linalg.norm(F(solution)):.2e}") # convergence history print("\nconvergence history:") print("iteration x y ||F(x,y)||") print("-" * 60) for i, xy in enumerate(history): norm_F = np.linalg.norm(F(xy)) print(f"{i:3d} {xy[0]:12.8f} {xy[1]:12.8f} {norm_F:.2e}") # : and convergence x_range = np.linspace(-0.5, 2.5, 200) y_range = np.linspace(-0.5, 2.5, 200) X, Y = np.meshgrid(x_range, y_range) # equation Z1 = X**2 + Y**2 - 4 # f1 = 0 Z2 = X**2 - Y - 1 # f2 = 0 plt.figure(figsize=(10, 8)) plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='f₁(x,y) = 0') plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='f₂(x,y) = 0') # convergence history_array = np.array(history) plt.plot(history_array[:, 0], history_array[:, 1], 'go-', markersize=8, linewidth=2, label='Newton ') plt.plot(x0[0], x0[1], 'ks', markersize=12, label='initial value') plt.plot(solution[0], solution[1], 'r*', markersize=20, label='') # iteration for i, xy in enumerate(history[::2]): # 2 and plt.annotate(f'{i*2}', xy=(xy[0], xy[1]), xytext=(5, 5), textcoords='offset points', fontsize=9) plt.xlabel('x', fontsize=12) plt.ylabel('y', fontsize=12) plt.title('Multivariate Newton Method convergence', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.axis('equal') plt.tight_layout() plt.savefig('multivariate_newton.png', dpi=150, bbox_inches='tight') plt.show() # initial value print("\n" + "=" * 60) print("initial value ") print("=" * 60) initial_guesses = [ np.array([1.5, 1.5]), np.array([-1.5, 1.5]), np.array([0.5, -0.5]) ] for i, x0_test in enumerate(initial_guesses): sol, hist = multivariate_newton(F, J, x0_test, tol=1e-10, max_iter=50) print(f"\ninitial value {i+1}: {x0_test}") print(f" Solution: {sol}") print(f" iterationiterations: {len(hist) - 1}") print(f" ||F||: {np.linalg.norm(F(sol)):.2e}") 

============================================================ Multivariate Newton Method: 2equation solution ============================================================ f1(x,y) = x² + y² - 4 = 0 f2(x,y) = x² - y - 1 = 0 initial value: x0 = [1.5 1.5] Multivariate Newton Method: 5converged in iterations Solution: [1.52176087 1.31528131] Verification: F(x) = [ 4.44089210e-16 -8.88178420e-16] ||F(x)||: 9.93e-16 convergence history: iteration x y ||F(x,y)|| ------------------------------------------------------------ 0 1.50000000 1.50000000 7.07e-01 1 1.52500000 1.32500000 4.03e-02 2 1.52177419 1.31532258 9.18e-05 3 1.52176087 1.31528132 4.81e-10 4 1.52176087 1.31528131 8.88e-16 5 1.52176087 1.31528131 9.93e-16 ============================================================ initial value ============================================================ Multivariate Newton Method: 5converged in iterations initial value 1: [1.5 1.5] Solution: [1.52176087 1.31528131] iterationiterations: 5 ||F||: 9.93e-16 Multivariate Newton Method: 5converged in iterations initial value 2: [-1.5 1.5] Solution: [-1.52176087 1.31528131] iterationiterations: 5 ||F||: 9.93e-16 Multivariate Newton Method: 6converged in iterations initial value 3: [ 0.5 -0.5] Solution: [0.78615138 0.38201136] iterationiterations: 6 ||F||: 1.78e-15

## 3.5 Utilizing scipy.optimize

SciPy scipy.optimize.root 、varioussuitablealgorithm 。 implementation than fast stable 。 

### Code Example 5: Using scipy.optimize.root
    
    
    from scipy.optimize import root, fsolve, newton print("=" * 60) print("scipy.optimize Nonlinear equation solution") print("=" * 60) # 1 : x³ - 2x - 5 = 0 f_scalar = lambda x: x**3 - 2*x - 5 df_scalar = lambda x: 3*x**2 - 2 print("\n1. 1equation: x³ - 2x - 5 = 0") print("-" * 60) # scipy.optimize.newton (Newton-Raphson) sol_newton = newton(f_scalar, x0=2.5, fprime=df_scalar) print(f"\nscipy.optimize.newton:") print(f" Solution: x = {sol_newton:.10f}") print(f" Verification: f(x) = {f_scalar(sol_newton):.2e}") # scipy.optimize.fsolve (hybrid Powell) sol_fsolve = fsolve(f_scalar, x0=2.5)[0] print(f"\nscipy.optimize.fsolve:") print(f" Solution: x = {sol_fsolve:.10f}") print(f" Verification: f(x) = {f_scalar(sol_fsolve):.2e}") # print("\n" + "=" * 60) print("2. equation") print("=" * 60) print("f1(x,y) = x² + y² - 4 = 0") print("f2(x,y) = x² - y - 1 = 0") def F_vec(xy): x, y = xy return np.array([ x**2 + y**2 - 4, x**2 - y - 1 ]) def J_vec(xy): x, y = xy return np.array([ [2*x, 2*y], [2*x, -1] ]) x0 = np.array([1.5, 1.5]) # method1: hybr (Powell hybrid - ) result_hybr = root(F_vec, x0, method='hybr') print(f"\nmethod='hybr' (Powell hybrid):") print(f" Solution: {result_hybr.x}") print(f" : {result_hybr.success}") print(f" iterationiterations: {result_hybr.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_hybr.x)):.2e}") # method2: lm (Levenberg-Marquardt) result_lm = root(F_vec, x0, method='lm') print(f"\nmethod='lm' (Levenberg-Marquardt):") print(f" Solution: {result_lm.x}") print(f" : {result_lm.success}") print(f" iterationiterations: {result_lm.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_lm.x)):.2e}") # method3: df-sane (Spectral Projected Gradient) result_df = root(F_vec, x0, method='df-sane') print(f"\nmethod='df-sane' (Spectral Projected Gradient):") print(f" Solution: {result_df.x}") print(f" : {result_df.success}") print(f" iterationiterations: {result_df.nfev}") print(f" ||F||: {np.linalg.norm(F_vec(result_df.x)):.2e}") # method4: Jacobi result_jac = root(F_vec, x0, jac=J_vec, method='hybr') print(f"\nmethod='hybr' with Jacobian:") print(f" Solution: {result_jac.x}") print(f" functioniterations: {result_jac.nfev}") print(f" Jacobiiterations: {result_jac.njev}") print(f" ||F||: {np.linalg.norm(F_vec(result_jac.x)):.2e}") # problem: print("\n" + "=" * 60) print("3. problem: calculation") print("=" * 60) print(": 2H₂ + O₂ ⇌ 2H₂O") print(" K = 10⁶ @ 298K") def chemical_equilibrium(concentrations): """ equation : [H2], [O2], [H2O] : and """ H2, O2, H2O = concentrations # （） H2_0 = 2.0 # mol O2_0 = 1.0 # mol H2O_0 = 0.0 # K = 1e6 # equation return np.array([ # : H 2*H2 + 2*H2O - 2*H2_0 - 2*H2O_0, # : O 2*O2 + H2O - 2*O2_0 - H2O_0, # (H2O**2) / (H2**2 * O2) - K ]) # c0 = np.array([0.1, 0.1, 1.8]) # result_chem = root(chemical_equilibrium, c0, method='hybr') print(f"\n（）:") print(f" [H₂] = {result_chem.x[0]:.6f} mol") print(f" [O₂] = {result_chem.x[1]:.6f} mol") print(f" [H₂O] = {result_chem.x[2]:.6f} mol") print(f"\nconvergence: {result_chem.success}") print(f": {np.linalg.norm(chemical_equilibrium(result_chem.x)):.2e}") # H2, O2, H2O = result_chem.x K_calc = (H2O**2) / (H2**2 * O2) print(f"\n Verification:") print(f" K = 1.00e+06") print(f" calculation K = {K_calc:.2e}") 

============================================================ scipy.optimize Nonlinear equation solution ============================================================ 1. 1equation: x³ - 2x - 5 = 0 ------------------------------------------------------------ scipy.optimize.newton: Solution: x = 2.0945514815 Verification: f(x) = 0.00e+00 scipy.optimize.fsolve: Solution: x = 2.0945514815 Verification: f(x) = 0.00e+00 ============================================================ 2. equation ============================================================ f1(x,y) = x² + y² - 4 = 0 f2(x,y) = x² - y - 1 = 0 method='hybr' (Powell hybrid): Solution: [1.52176087 1.31528131] : True iterationiterations: 14 ||F||: 1.23e-11 method='lm' (Levenberg-Marquardt): Solution: [1.52176087 1.31528131] : True iterationiterations: 10 ||F||: 1.49e-13 method='df-sane' (Spectral Projected Gradient): Solution: [1.52176087 1.31528131] : True iterationiterations: 38 ||F||: 2.67e-11 method='hybr' with Jacobian: Solution: [1.52176087 1.31528131] functioniterations: 8 Jacobiiterations: 5 ||F||: 1.56e-13 ============================================================ 3. problem: calculation ============================================================ : 2H₂ + O₂ ⇌ 2H₂O K = 10⁶ @ 298K （）: [H₂] = 0.000632 mol [O₂] = 0.000316 mol [H₂O] = 1.999368 mol convergence: True : 1.93e-09 Verification: K = 1.00e+06 calculation K = 1.00e+06

## 3.6 Comparison of Bracketing Methods

do 、bisection method Regula Falsi（）Brent etc. 。 

### Code Example 6: Implementation and Comparison of Bracketing Methods
    
    
    from scipy.optimize import brentq, ridder def regula_falsi(f, a, b, tol=1e-10, max_iter=100): """ Regula Falsi（） Parameters: ----------- f : callable Target function a, b : float Initial interval [a, b] tol : float Tolerance max_iter : int Maximum iterations Returns: -------- root : float history : list iteration """ fa = f(a) fb = f(b) if fa * fb > 0: raise Valueerror("f(a) and f(b) must have opposite signs") history = [] for i in range(max_iter): # use c = (a * fb - b * fa) / (fb - fa) fc = f(c) history.append(c) if abs(fc) < tol: print(f"Regula Falsi: {i+1}converged in iterations") return c, history if fa * fc < 0: b = c fb = fc else: a = c fa = fc print(f"Regula Falsi: {max_iter}did not converge in iterations") return c, history # function f = lambda x: x**3 - 2*x - 5 print("=" * 60) print(" ") print("f(x) = x³ - 2x - 5 = 0") print("=" * 60) a, b = 2.0, 3.0 # 1. bisection method root_bis, hist_bis = bisection_method(f, a, b, tol=1e-10) # 2. Regula Falsi root_rf, hist_rf = regula_falsi(f, a, b, tol=1e-10) # 3. Brent (SciPy) root_brent = brentq(f, a, b, xtol=1e-10) # 4. Ridder (SciPy) root_ridder = ridder(f, a, b, xtol=1e-10) # print("\n :") print(f"{'method':<20} {'iterationiterations':>10} {'':>18} {'f(x)':>12}") print("-" * 65) print(f"{'bisection method':<20} {len(hist_bis):>10} {root_bis:>18.10f} {f(root_bis):>12.2e}") print(f"{'Regula Falsi':<20} {len(hist_rf):>10} {root_rf:>18.10f} {f(root_rf):>12.2e}") print(f"{'Brent':<20} {'-':>10} {root_brent:>18.10f} {f(root_brent):>12.2e}") print(f"{'Ridder':<20} {'-':>10} {root_ridder:>18.10f} {f(root_ridder):>12.2e}") # convergence historyVisualization of plt.figure(figsize=(12, 5)) # : error plt.subplot(1, 2, 1) errors_bis = [abs(x - root_bis) for x in hist_bis] errors_rf = [abs(x - root_rf) for x in hist_rf] plt.semilogy(errors_bis, 'o-', label='bisection method', markersize=5, linewidth=2) plt.semilogy(errors_rf, 's-', label='Regula Falsi', markersize=5, linewidth=2) plt.xlabel('iterationiterations', fontsize=12) plt.ylabel('error', fontsize=12) plt.title('Comparison of Convergence Speed', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) # : method plt.subplot(1, 2, 2) methods = ['bisection method', 'Regula\nFalsi', 'Brent', 'Ridder'] iterations = [len(hist_bis), len(hist_rf), 10, 8] # Brent and Ridder colors = ['#667eea', '#764ba2', '#48c774', '#3298dc'] bars = plt.bar(methods, iterations, color=colors, alpha=0.7) for bar, it in zip(bars, iterations): height = bar.get_height() plt.text(bar.get_x() + bar.get_width()/2., height, f'{it}', ha='center', va='bottom', fontsize=11, fontweight='bold') plt.ylabel('iterationiterations（）', fontsize=12) plt.title('calculation ', fontsize=14) plt.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('bracketing_methods_comparison.png', dpi=150, bbox_inches='tight') plt.show() print("\n" + "=" * 60) print("Summary:") print(" - Bisection method: slow（convergence）") print(" - Regula Falsi: bisection method than fast（convergence）") print(" - Brent: （2convergence and stable）") print(" - Ridder: fast stable（convergence）") print("=" * 60) 

============================================================ f(x) = x³ - 2x - 5 = 0 ============================================================ Bisection method: 36converged in iterations Regula Falsi: 9converged in iterations : method iterationiterations f(x) ----------------------------------------------------------------- bisection method 36 2.0945514815 0.00e+00 Regula Falsi 9 2.0945514815 8.88e-16 Brent - 2.0945514815 0.00e+00 Ridder - 2.0945514815 0.00e+00 ============================================================ Summary: - Bisection method: slow（convergence） - Regula Falsi: bisection method than fast（convergence） - Brent: （2convergence and stable） - Ridder: fast stable（convergence） ============================================================

## 3.7 Practical Example: Application to Materials Science

Nonlinear equation materials science various 。equation、calculation、 etc. 。 

### Code Example 7: Solving van der Waals equation of State
    
    
    from scipy.optimize import fsolve # van der Waalsequation: (P + a/V²)(V - b) = RT # def van_der_waals(V, P, T, a, b, R=8.314): """ van der Waalsequation Parameters: ----------- V : float [L/mol] P : float [bar] T : float [K] a, b : float van der Waals R : float [J/(mol·K)] Returns: -------- float equation """ # : 1 bar = 10⁵ Pa, 1 L = 10⁻³ m³ P_Pa = P * 1e5 V_m3 = V * 1e-3 return (P_Pa + a / V_m3**2) * (V_m3 - b) - R * T print("=" * 60) print(": van der Waalsequation") print("=" * 60) # CO₂ van der Waals a_CO2 = 0.3658 # Pa·m⁶/mol² b_CO2 = 4.267e-5 # m³/mol T = 300 # K P = 50 # bar print(f"\nCO₂ @ T = {T} K, P = {P} bar") print(f"van der Waals: a = {a_CO2}, b = {b_CO2}") # （ from ） V0 = 8.314 * T / (P * 1e5) * 1000 # L/mol # van der WaalsequationSolve V_solution = fsolve(lambda V: van_der_waals(V, P, T, a_CO2, b_CO2), V0)[0] print(f"\nSolution:") print(f" V = {V_solution:.6f} L/mol") print(f" V = {V_solution * 1e-3:.6e} m³/mol") # and V_ideal = 8.314 * T / (P * 1e5) * 1000 # L/mol print(f"\n : {V_ideal:.6f} L/mol") print(f"error: {abs(V_solution - V_ideal) / V_ideal * 100:.2f}%") # various calculation print("\n" + "=" * 60) print(" analysis") print("=" * 60) pressures = np.logspace(0, 3, 50) # 1 bar to 1000 bar volumes_vdw = [] volumes_ideal = [] for P in pressures: V0 = 8.314 * T / (P * 1e5) * 1000 try: V_vdw = fsolve(lambda V: van_der_waals(V, P, T, a_CO2, b_CO2), V0)[0] volumes_vdw.append(V_vdw) except: volumes_vdw.append(np.nan) V_id = 8.314 * T / (P * 1e5) * 1000 volumes_ideal.append(V_id) # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # : P-V ax1.loglog(pressures, volumes_vdw, 'b-', linewidth=2, label='van der Waals') ax1.loglog(pressures, volumes_ideal, 'r--', linewidth=2, label='') ax1.set_xlabel(' P [bar]', fontsize=12) ax1.set_ylabel(' V [L/mol]', fontsize=12) ax1.set_title(f'CO₂ equation (T = {T} K)', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) # : Z = PV/RT Z_vdw = np.array(pressures) * 1e5 * np.array(volumes_vdw) * 1e-3 / (8.314 * T) Z_ideal = np.ones_like(pressures) ax2.semilogx(pressures, Z_vdw, 'b-', linewidth=2, label='van der Waals') ax2.semilogx(pressures, Z_ideal, 'r--', linewidth=2, label='') ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5) ax2.set_xlabel(' P [bar]', fontsize=12) ax2.set_ylabel(' Z = PV/RT', fontsize=12) ax2.set_title(' ', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('van_der_waals_equation.png', dpi=150, bbox_inches='tight') plt.show() # calculation print("\n" + "=" * 60) print("CO₂ ") print("=" * 60) # （） T_c_theory = 8 * a_CO2 / (27 * 8.314 * b_CO2) P_c_theory = a_CO2 / (27 * b_CO2**2) / 1e5 # bar V_c_theory = 3 * b_CO2 * 1000 # L/mol print(f"van der Waals:") print(f" Tc = {T_c_theory:.2f} K") print(f" Pc = {P_c_theory:.2f} bar") print(f" Vc = {V_c_theory:.6f} L/mol") print(f"\n (CO₂):") print(f" Tc = 304.13 K") print(f" Pc = 73.77 bar") print(f"\n and ") 

============================================================ : van der Waalsequation ============================================================ CO₂ @ T = 300 K, P = 50 bar van der Waals: a = 0.3658, b = 4.267e-05 Solution: V = 0.048234 L/mol V = 4.823368e-05 m³/mol : 0.049884 L/mol error: 3.31% ============================================================ analysis ============================================================ ============================================================ CO₂ ============================================================ van der Waals: Tc = 304.19 K Pc = 73.03 bar Vc = 0.000128 L/mol (CO₂): Tc = 304.13 K Pc = 73.77 bar and 

### 🏋️ Exercises

#### 1: Comparison of Convergence Speed

equation \\( e^x - 3x = 0 \\) bisection method、Newton-Raphson、Secant 、convergence。interval \\([0, 1]\\)、initial value \\( x_0 = 0.5 \\) use。 

#### 2: Multivariate Newton Method implementation

equationMultivariate Newton Method : 

\\[ \begin{cases} x^2 - y - 1 = 0 \\\ x - y^2 + 1 = 0 \end{cases} \\] 

initial value \\((x_0, y_0) = (1.5, 1.0)\\) from 、convergence。 

#### Exercise 3: Robustness of Bracketing Methods

function 、bisection method and Newton-Raphson: 

\\[ f(x) = x^3 - 2x^2 - 5 \\] 

(a) interval \\([2, 4]\\) bisection in  
(b) initial value \\( x_0 = 0 \\) Newton-Raphson（ do ）  
(c) initial value \\( x_0 = 3 \\) Newton-Raphson 

#### Exercise 4: Utilizing scipy.optimize

scipy.optimize.root、 3equation: 

\\[ \begin{cases} x + y + z = 6 \\\ x^2 + y^2 + z^2 = 14 \\\ xyz = 6 \end{cases} \\] 

initial value multiple 。 

#### Exercise 5: Application to Materials Science

Arrhenius problem: 

\\[ k = A \exp\left(-\frac{E_a}{RT}\right) \\] 

When reaction rate constants at two temperatures are known, find activation energy \\( E_a \\) and frequency factor \\( A \\) : 

  * \\( k(300 \text{ K}) = 1.0 \times 10^{-5} \text{ s}^{-1} \\)
  * \\( k(350 \text{ K}) = 5.0 \times 10^{-4} \text{ s}^{-1} \\)

(: do 、) 

## Summary

、Nonlinear equation : 

  * **Bracketing methods:** Reliability and convergence speed of bisection, Regula Falsi, and Brent methods
  * **Open methods:** Fast convergence and precautions for Newton-Raphson and Secant methods
  * **Multivariate problems:** JacobiMultivariate Newton Method
  * **Practical tools:** Advanced solution algorithms using scipy.optimize
  * **Materials science applications:** Real problems such as equations of state and chemical equilibrium

Nonlinear equation Solution Methods 、suitable、parameter estimation、problem etc.wideapplication 。 、 、equation 。 

[← Chapter 2](<chapter-2.html>)[Chapter 4 →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
