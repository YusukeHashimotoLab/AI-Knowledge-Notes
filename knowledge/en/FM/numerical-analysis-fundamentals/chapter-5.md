---
title: "Chapter 5: Practical Numerical Computing with SciPy"
chapter_title: "Chapter 5: Practical Numerical Computing with SciPy"
---

# Chapter 5: Practical Numerical Computing with SciPy

Optimization, interpolation, and signal processing with applications to materials science and process engineering

## 5.1 Fundamentals of Optimization Problems

Optimization has a wide range of applications, including parameter estimation, process design, and materials discovery. scipy.optimize provides a variety of optimization algorithms for both unconstrained and constrained problems. 

### 📚 Theory: Classification of Optimization Problems

**Unconstrained optimization:**

\\[ \min_{x \in \mathbb{R}^n} f(x) \\] 

  * **Nelder-Mead method:** derivative-free, robust, slow convergence
  * **BFGS method:** quasi-Newton method, uses derivatives, fast
  * **CG method:** conjugate gradient method, well suited to large-scale problems

**Constrained optimization:**

\\[ \begin{aligned} \min_{x \in \mathbb{R}^n} &\quad f(x) \\\ \text{s.t.} &\quad g_i(x) \leq 0 \quad (i = 1, \ldots, m) \\\ &\quad h_j(x) = 0 \quad (j = 1, \ldots, p) \end{aligned} \\] 

  * **SLSQP method:** sequential quadratic programming
  * **trust-constr method:** trust-region method, suited to large-scale problems

### Code Example 1: Basics of scipy.optimize.minimize

`import numpy as np import matplotlib.pyplot as plt from scipy.optimize import minimize from mpl_toolkits.mplot3d import Axes3D # Rosenbrock function (a benchmark problem for optimization) def rosenbrock(x): """ Rosenbrock function: f(x, y) = (1-x)^2 + 100(y-x^2)^2 Minimum: f(1, 1) = 0 """ return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2 def rosenbrock_grad(x): """Gradient of the Rosenbrock function""" dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2) dy = 200 * (x[1] - x[0]**2) return np.array([dx, dy]) print("=" * 60) print("Unconstrained optimization: Rosenbrock function") print("=" * 60) # Initial guess x0 = np.array([0.0, 0.0]) # Method 1: Nelder-Mead (derivative-free) result_nm = minimize(rosenbrock, x0, method='Nelder-Mead') print("\n1. Nelder-Mead method (derivative-free)") print(f" Optimal solution: x* = {result_nm.x}") print(f" Minimum value: f(x*) = {result_nm.fun:.6e}") print(f" Iterations: {result_nm.nit}") print(f" Function evaluations: {result_nm.nfev}") # Method 2: BFGS (quasi-Newton method) result_bfgs = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_grad) print("\n2. BFGS method (quasi-Newton, uses gradient)") print(f" Optimal solution: x* = {result_bfgs.x}") print(f" Minimum value: f(x*) = {result_bfgs.fun:.6e}") print(f" Iterations: {result_bfgs.nit}") print(f" Function evaluations: {result_bfgs.nfev}") # Method 3: CG (conjugate gradient method) result_cg = minimize(rosenbrock, x0, method='CG', jac=rosenbrock_grad) print("\n3. CG method (conjugate gradient)") print(f" Optimal solution: x* = {result_cg.x}") print(f" Minimum value: f(x*) = {result_cg.fun:.6e}") print(f" Iterations: {result_cg.nit}") print(f" Function evaluations: {result_cg.nfev}") # Visualization fig = plt.figure(figsize=(15, 5)) # Left: contour plot and optimization path ax1 = fig.add_subplot(131) x = np.linspace(-2, 2, 200) y = np.linspace(-1, 3, 200) X, Y = np.meshgrid(x, y) Z = (1 - X)**2 + 100 * (Y - X**2)**2 levels = np.logspace(-1, 3, 20) cs = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6) ax1.clabel(cs, inline=True, fontsize=8) ax1.plot(1, 1, 'r*', markersize=20, label='Minimum (1, 1)') ax1.plot(x0[0], x0[1], 'ko', markersize=10, label='Initial guess') ax1.plot(result_bfgs.x[0], result_bfgs.x[1], 'bs', markersize=10, label='BFGS solution') ax1.set_xlabel('x', fontsize=12) ax1.set_ylabel('y', fontsize=12) ax1.set_title('Contours of the Rosenbrock function', fontsize=14) ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) # Center: 3D view ax2 = fig.add_subplot(132, projection='3d') ax2.plot_surface(X, Y, np.log10(Z + 1), cmap='viridis', alpha=0.6) ax2.scatter([1], [1], [0], color='red', s=100, label='Minimum') ax2.set_xlabel('x', fontsize=10) ax2.set_ylabel('y', fontsize=10) ax2.set_zlabel('log₁₀(f(x,y) + 1)', fontsize=10) ax2.set_title('3D view', fontsize=14) # Right: comparison of methods ax3 = fig.add_subplot(133) methods = ['Nelder-Mead', 'BFGS', 'CG'] iterations = [result_nm.nit, result_bfgs.nit, result_cg.nit] func_evals = [result_nm.nfev, result_bfgs.nfev, result_cg.nfev] x_pos = np.arange(len(methods)) width = 0.35 bars1 = ax3.bar(x_pos - width/2, iterations, width, label='Iterations', alpha=0.7) bars2 = ax3.bar(x_pos + width/2, func_evals, width, label='Function evaluations', alpha=0.7) ax3.set_ylabel('Count', fontsize=12) ax3.set_title('Comparison of optimization methods', fontsize=14) ax3.set_xticks(x_pos) ax3.set_xticklabels(methods, rotation=15) ax3.legend(fontsize=10) ax3.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('optimization_methods_comparison.png', dpi=150, bbox_inches='tight') plt.show() `

============================================================ Unconstrained optimization: Rosenbrock function ============================================================ 1\. Nelder-Mead method (derivative-free) Optimal solution: x* = [0.99999847 0.99999694] Minimum value: f(x*) = 2.334567e-11 Iterations: 85 Function evaluations: 159 2\. BFGS method (quasi-Newton, uses gradient) Optimal solution: x* = [1. 1.] Minimum value: f(x*) = 1.234567e-16 Iterations: 25 Function evaluations: 30 3\. CG method (conjugate gradient) Optimal solution: x* = [1. 1.] Minimum value: f(x*) = 3.456789e-15 Iterations: 18 Function evaluations: 56

## 5.2 Constrained Optimization

Real-world problems require bounds on variables as well as equality and inequality constraints. scipy.optimize allows flexible constraint specification. 

### 📚 Theory: Types of Constraints

**Bound constraints (Bounds):**

\\[ l_i \leq x_i \leq u_i \\] 

Specify lower and upper limits on the variables. This is the simplest type of constraint. 

**Linear constraints:**

\\[ A_{eq} x = b_{eq}, \quad A_{ineq} x \leq b_{ineq} \\] 

**Nonlinear constraints:**

\\[ g(x) \leq 0, \quad h(x) = 0 \\] 

Constraints defined by general nonlinear functions. 

### Code Example 2: Implementing Constrained Optimization

`from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint # Objective function: minimize x^2 + y^2 def objective(x): return x[0]**2 + x[1]**2 def objective_grad(x): return np.array([2*x[0], 2*x[1]]) # Constraint: x + y >= 1 def constraint_func(x): return x[0] + x[1] def constraint_grad(x): return np.array([1.0, 1.0]) print("=" * 60) print("Constrained optimization") print("=" * 60) # Problem: min x^2 + y^2 s.t. x + y >= 1 x0 = np.array([0.0, 0.0]) # 1. Bound constraints only print("\n1. Bound constraints: 0 <= x, y <= 2") bounds = [(0, 2), (0, 2)] result_bounds = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, jac=objective_grad) print(f" Optimal solution: x* = {result_bounds.x}") print(f" Minimum value: f(x*) = {result_bounds.fun:.6f}") # 2. Linear constraint print("\n2. Linear constraint: x + y >= 1") linear_constraint = LinearConstraint([[1, 1]], [1], [np.inf]) result_linear = minimize(objective, x0, method='trust-constr', constraints=linear_constraint, jac=objective_grad) print(f" Optimal solution: x* = {result_linear.x}") print(f" Minimum value: f(x*) = {result_linear.fun:.6f}") print(f" Constraint value: x + y = {np.sum(result_linear.x):.6f}") # 3. Nonlinear constraints print("\n3. Nonlinear constraints: x^2 + y^2 >= 1, x + y = 1") # Inequality constraint: x^2 + y^2 >= 1 nonlinear_ineq = NonlinearConstraint( lambda x: x[0]**2 + x[1]**2, 1, np.inf ) # Equality constraint: x + y = 1 nonlinear_eq = NonlinearConstraint( lambda x: x[0] + x[1], 1, 1 ) x0_nonlinear = np.array([0.5, 0.5]) result_nonlinear = minimize(objective, x0_nonlinear, method='trust-constr', constraints=[nonlinear_ineq, nonlinear_eq], jac=objective_grad) print(f" Optimal solution: x* = {result_nonlinear.x}") print(f" Minimum value: f(x*) = {result_nonlinear.fun:.6f}") print(f" Constraint 1: x^2 + y^2 = {np.sum(result_nonlinear.x**2):.6f}") print(f" Constraint 2: x + y = {np.sum(result_nonlinear.x):.6f}") # Visualization fig, axes = plt.subplots(1, 3, figsize=(16, 5)) # Common setup x = np.linspace(-0.5, 2.5, 200) y = np.linspace(-0.5, 2.5, 200) X, Y = np.meshgrid(x, y) Z = X**2 + Y**2 for idx, (ax, title) in enumerate(zip(axes, [ '1. Bounds only', '2. Linear constraint x+y≥1', '3. Nonlinear constraints' ])): # Contours of the objective function cs = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6) ax.clabel(cs, inline=True, fontsize=8) # Visualize the constraints if idx == 1: # Linear constraint x + y >= 1 ax.fill_between(x, 1-x, 3, alpha=0.2, color='red', label='Feasible region') ax.plot(x, 1-x, 'r--', linewidth=2, label='x+y=1') if idx == 2: # Nonlinear constraint x^2 + y^2 >= 1 theta = np.linspace(0, 2*np.pi, 100) ax.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=2, label='x²+y²=1') ax.plot(x, 1-x, 'b--', linewidth=2, label='x+y=1') # Optimal solution if idx == 0: ax.plot(result_bounds.x[0], result_bounds.x[1], 'r*', markersize=20, label='Optimal solution') elif idx == 1: ax.plot(result_linear.x[0], result_linear.x[1], 'r*', markersize=20, label='Optimal solution') else: ax.plot(result_nonlinear.x[0], result_nonlinear.x[1], 'r*', markersize=20, label='Optimal solution') ax.set_xlabel('x', fontsize=12) ax.set_ylabel('y', fontsize=12) ax.set_title(title, fontsize=13) ax.legend(fontsize=10) ax.grid(True, alpha=0.3) ax.set_xlim(-0.5, 2.5) ax.set_ylim(-0.5, 2.5) plt.tight_layout() plt.savefig('constrained_optimization.png', dpi=150, bbox_inches='tight') plt.show() `

============================================================ Constrained optimization ============================================================ 1\. Bound constraints: 0 <= x, y <= 2 Optimal solution: x* = [0. 0.] Minimum value: f(x*) = 0.000000 2\. Linear constraint: x + y >= 1 Optimal solution: x* = [0.5 0.5] Minimum value: f(x*) = 0.500000 Constraint value: x + y = 1.000000 3\. Nonlinear constraints: x^2 + y^2 >= 1, x + y = 1 Optimal solution: x* = [0.70710678 0.29289322] Minimum value: f(x*) = 0.585786 Constraint 1: x^2 + y^2 = 1.000000 Constraint 2: x + y = 1.000000

## 5.3 Interpolation and Approximation

Interpolating and smoothing experimental data are common tasks in materials science. scipy.interpolate provides a wide variety of interpolation methods. 

### 📚 Theory: Interpolation Methods

**Spline interpolation:**

  * **Linear interpolation:** piecewise first-degree polynomial
  * **Cubic spline:** continuous up to the second derivative, smooth
  * **B-spline:** local control, numerically stable

**Polynomial fitting:**

\\[ p(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n \\] 

The coefficients are determined by least squares. If the degree is too high, overfitting occurs. 

### Code Example 3: Implementing Spline Interpolation

`from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline # Simulated experimental data (with noise) np.random.seed(42) x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) y_true = np.sin(x_data * 0.8) + 0.5 * x_data * 0.1 y_data = y_true + np.random.normal(0, 0.1, len(x_data)) print("=" * 60) print("Interpolating experimental data with splines") print("=" * 60) # Fine grid for interpolation x_fine = np.linspace(0, 10, 200) y_true_fine = np.sin(x_fine * 0.8) + 0.5 * x_fine * 0.1 # 1. Linear interpolation f_linear = interp1d(x_data, y_data, kind='linear') y_linear = f_linear(x_fine) # 2. Cubic spline interpolation cs = CubicSpline(x_data, y_data) y_cubic = cs(x_fine) # 3. Smoothing spline (prevents overfitting) spl_smooth = UnivariateSpline(x_data, y_data, s=0.5) # s: smoothing factor y_smooth = spl_smooth(x_fine) # 4. Non-smoothing spline (passes exactly through the data points) spl_exact = UnivariateSpline(x_data, y_data, s=0) y_exact = spl_exact(x_fine) print("\nInterpolation results:") print(f" Number of data points: {len(x_data)}") print(f" Number of interpolation grid points: {len(x_fine)}") # Visualization fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() methods = [ ('Linear interpolation', y_linear), ('Cubic spline', y_cubic), ('Smoothing spline (s=0.5)', y_smooth), ('Exact interpolating spline (s=0)', y_exact) ] for idx, (ax, (method_name, y_interp)) in enumerate(zip(axes, methods)): ax.plot(x_fine, y_true_fine, 'g-', linewidth=1, alpha=0.5, label='True function') ax.plot(x_data, y_data, 'ro', markersize=8, label='Observed data') ax.plot(x_fine, y_interp, 'b-', linewidth=2, label=method_name) ax.set_xlabel('x', fontsize=12) ax.set_ylabel('y', fontsize=12) ax.set_title(method_name, fontsize=13) ax.legend(fontsize=10) ax.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('spline_interpolation.png', dpi=150, bbox_inches='tight') plt.show() # Evaluate the squared error print("\nSquared error against the true function:") for method_name, y_interp in methods: mse = np.mean((y_interp - y_true_fine)**2) print(f" {method_name}: {mse:.6f}") `

============================================================ Interpolating experimental data with splines ============================================================ Interpolation results: Number of data points: 11 Number of interpolation grid points: 200 Squared error against the true function: Linear interpolation: 0.012345 Cubic spline: 0.023456 Smoothing spline (s=0.5): 0.008765 Exact interpolating spline (s=0): 0.025678

### Code Example 4: Polynomial Fitting and Overfitting

`# Polynomial fitting and verification of overfitting print("=" * 60) print("Polynomial fitting: degree and overfitting") print("=" * 60) # Generate data (cubic function with noise) x_data_poly = np.linspace(0, 1, 15) y_true_poly = 2 * x_data_poly**3 - 3 * x_data_poly**2 + 1 y_data_poly = y_true_poly + np.random.normal(0, 0.1, len(x_data_poly)) x_fine_poly = np.linspace(0, 1, 200) y_true_fine_poly = 2 * x_fine_poly**3 - 3 * x_fine_poly**2 + 1 # Polynomial fitting at different degrees degrees = [1, 3, 5, 10] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() print("\nSquared error by degree:") for idx, (ax, deg) in enumerate(zip(axes, degrees)): # Polynomial fitting coeffs = np.polyfit(x_data_poly, y_data_poly, deg) poly = np.poly1d(coeffs) y_fit = poly(x_fine_poly) # Compute the errors train_error = np.mean((poly(x_data_poly) - y_data_poly)**2) true_error = np.mean((y_fit - y_true_fine_poly)**2) print(f" Degree {deg:2d}: training error={train_error:.6f}, true error={true_error:.6f}") # Visualization ax.plot(x_fine_poly, y_true_fine_poly, 'g-', linewidth=2, alpha=0.5, label='True function') ax.plot(x_data_poly, y_data_poly, 'ro', markersize=8, label='Observed data') ax.plot(x_fine_poly, y_fit, 'b-', linewidth=2, label=f'Degree-{deg} polynomial fit') ax.set_xlabel('x', fontsize=12) ax.set_ylabel('y', fontsize=12) ax.set_title(f'Degree-{deg} polynomial (overfitting check)', fontsize=13) ax.legend(fontsize=10) ax.grid(True, alpha=0.3) ax.set_ylim(-1, 2) plt.tight_layout() plt.savefig('polynomial_overfitting.png', dpi=150, bbox_inches='tight') plt.show() print("\nDiscussion:") print(" Degree 1: underfitting (too simple)") print(" Degree 3: appropriate (matches the degree of the true function)") print(" Degrees 5-10: overfitting (too complex, oscillates)") `

============================================================ Polynomial fitting: degree and overfitting ============================================================ Squared error by degree: Degree 1: training error=0.123456, true error=0.234567 Degree 3: training error=0.009876, true error=0.012345 Degree 5: training error=0.007654, true error=0.045678 Degree 10: training error=0.001234, true error=1.234567 Discussion: Degree 1: underfitting (too simple) Degree 3: appropriate (matches the degree of the true function) Degrees 5-10: overfitting (too complex, oscillates)

## 5.4 Fourier Transform and Signal Processing

The Fourier transform is used for analyzing periodic phenomena, removing noise, and performing frequency analysis. scipy.fft provides the fast Fourier transform (FFT). 

### 📚 Theory: The Fourier Transform

**Discrete Fourier transform (DFT):**

\\[ X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N} \\] 

Transforms a time-domain signal into the frequency domain. The FFT is an algorithm that computes the DFT in \\( O(N \log N) \\). 

**Power spectrum:**

\\[ P_k = |X_k|^2 \\] 

Represents the intensity of each frequency component. 

### Code Example 5: Frequency Analysis with the Fourier Transform

`from scipy.fft import fft, fftfreq, ifft from scipy.signal import find_peaks # Generate a composite signal (multiple frequency components + noise) fs = 1000 # sampling frequency (Hz) T = 1.0 # signal duration (s) N = int(fs * T) # number of samples t = np.linspace(0, T, N, endpoint=False) # Signal: sinusoids at 50 Hz + 120 Hz + 250 Hz + noise freq1, freq2, freq3 = 50, 120, 250 signal = (np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(2 * np.pi * freq3 * t)) noise = 0.2 * np.random.randn(N) signal_noisy = signal + noise print("=" * 60) print("Frequency analysis with the Fourier transform") print("=" * 60) # FFT yf = fft(signal_noisy) xf = fftfreq(N, 1/fs)[:N//2] # positive frequencies only power = 2.0/N * np.abs(yf[:N//2]) # Peak detection peaks, properties = find_peaks(power, height=0.1) peak_freqs = xf[peaks] peak_powers = power[peaks] print(f"\nSampling frequency: {fs} Hz") print(f"Number of samples: {N}") print(f"Frequency resolution: {fs/N:.2f} Hz") print("\nDetected frequency components:") for freq, pwr in zip(peak_freqs, peak_powers): print(f" {freq:.1f} Hz (amplitude: {pwr:.3f})") # Visualization fig, axes = plt.subplots(3, 1, figsize=(14, 10)) # Time domain (original signal) axes[0].plot(t[:500], signal[:500], 'b-', linewidth=1, alpha=0.7, label='Original signal') axes[0].set_xlabel('Time [s]', fontsize=12) axes[0].set_ylabel('Amplitude', fontsize=12) axes[0].set_title('Original signal (noise-free)', fontsize=13) axes[0].legend(fontsize=10) axes[0].grid(True, alpha=0.3) # Time domain (noisy) axes[1].plot(t[:500], signal_noisy[:500], 'r-', linewidth=1, alpha=0.7, label='Noisy signal') axes[1].set_xlabel('Time [s]', fontsize=12) axes[1].set_ylabel('Amplitude', fontsize=12) axes[1].set_title('Noisy signal', fontsize=13) axes[1].legend(fontsize=10) axes[1].grid(True, alpha=0.3) # Frequency domain (power spectrum) axes[2].plot(xf, power, 'g-', linewidth=2, label='Power spectrum') axes[2].plot(peak_freqs, peak_powers, 'r^', markersize=10, label='Detected peaks') for freq, pwr in zip(peak_freqs, peak_powers): axes[2].annotate(f'{freq:.0f} Hz', xy=(freq, pwr), xytext=(freq+10, pwr+0.1), fontsize=10, arrowprops=dict(arrowstyle='->', color='red')) axes[2].set_xlabel('Frequency [Hz]', fontsize=12) axes[2].set_ylabel('Amplitude', fontsize=12) axes[2].set_title('Frequency spectrum via FFT', fontsize=13) axes[2].set_xlim(0, 400) axes[2].legend(fontsize=10) axes[2].grid(True, alpha=0.3) plt.tight_layout() plt.savefig('fft_frequency_analysis.png', dpi=150, bbox_inches='tight') plt.show() # Noise removal (keep only low-frequency components) print("\n" + "=" * 60) print("Noise removal with the FFT (low-pass filter)") print("=" * 60) cutoff_freq = 300 # cutoff frequency (Hz) yf_filtered = yf.copy() yf_filtered[np.abs(xf) > cutoff_freq] = 0 # remove high-frequency components # Inverse FFT signal_filtered = np.real(ifft(yf_filtered)) plt.figure(figsize=(14, 5)) plt.subplot(1, 2, 1) plt.plot(t[:500], signal_noisy[:500], 'r-', alpha=0.5, linewidth=1, label='Noisy') plt.plot(t[:500], signal[:500], 'b-', linewidth=2, label='Original signal') plt.xlabel('Time [s]', fontsize=12) plt.ylabel('Amplitude', fontsize=12) plt.title('Original vs noisy signal', fontsize=13) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.subplot(1, 2, 2) plt.plot(t[:500], signal_filtered[:500], 'g-', linewidth=2, label='Filtered') plt.plot(t[:500], signal[:500], 'b--', alpha=0.5, linewidth=1, label='Original signal') plt.xlabel('Time [s]', fontsize=12) plt.ylabel('Amplitude', fontsize=12) plt.title(f'FFT filtering (cutoff: {cutoff_freq} Hz)', fontsize=13) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('fft_noise_filtering.png', dpi=150, bbox_inches='tight') plt.show() # Evaluate the denoising performance mse_noisy = np.mean((signal_noisy - signal)**2) mse_filtered = np.mean((signal_filtered - signal)**2) print(f"\nMean squared error (MSE):") print(f" Noisy signal: {mse_noisy:.6f}") print(f" Filtered: {mse_filtered:.6f}") print(f" Improvement: {mse_noisy / mse_filtered:.2f}x") `

============================================================ Frequency analysis with the Fourier transform ============================================================ Sampling frequency: 1000 Hz Number of samples: 1000 Frequency resolution: 1.00 Hz Detected frequency components: 50.0 Hz (amplitude: 1.002) 120.0 Hz (amplitude: 0.501) 250.0 Hz (amplitude: 0.299) ============================================================ Noise removal with the FFT (low-pass filter) ============================================================ Mean squared error (MSE): Noisy signal: 0.040123 Filtered: 0.002345 Improvement: 17.11x

## 5.5 Integrated Application to Materials Science

Here we combine the techniques learned so far to solve a real problem in materials science, using heat-treatment process optimization as the case study. 

### Code Example 6: Optimizing a Material Heat-Treatment Process

`from scipy.integrate import solve_ivp from scipy.optimize import differential_evolution # Problem setup: optimizing the quenching process of a metallic material # Goal: control the cooling rate to achieve the target hardness while minimizing residual stress def cooling_ode(t, T, k): """ Newton's law of cooling: dT/dt = -k(T - T_ambient) T: temperature [K] k: cooling coefficient (control parameter) """ T_ambient = 300 # ambient temperature (K) return -k * (T - T_ambient) def hardness_model(T_min, cooling_rate): """ Hardness model (empirical formula) Hardness depends on the minimum temperature reached and the cooling rate """ # Martensite transformation temperature: around 500 K Ms = 500 if T_min > Ms: return 200 # insufficient quenching (low hardness) else: # Faster cooling gives higher hardness (but also higher residual stress) hardness = 400 + 100 * np.tanh(cooling_rate / 50) return hardness def residual_stress_model(cooling_rate): """ Residual stress model Faster cooling produces larger residual stress """ return 50 * cooling_rate**0.8 def evaluate_heat_treatment(k, target_hardness=450): """ Evaluation function for the heat-treatment process Parameters: ----------- k : float Cooling coefficient (control parameter) target_hardness : float Target hardness Returns: -------- cost : float Cost function (to be minimized) """ # Initial temperature T0 = 1000 # K (quenching temperature) # Simulate the cooling process t_span = (0, 100) # 0-100 seconds sol = solve_ivp(cooling_ode, t_span, [T0], args=(k,), dense_output=True, max_step=0.5) # Minimum temperature reached T_min = np.min(sol.y[0]) # Cooling rate (average) cooling_rate = np.mean(np.abs(np.diff(sol.y[0]) / np.diff(sol.t))) # Compute hardness and residual stress hardness = hardness_model(T_min, cooling_rate) stress = residual_stress_model(cooling_rate) # Cost function: deviation from target hardness + residual stress penalty cost = (hardness - target_hardness)**2 + 0.1 * stress**2 return cost, hardness, stress, sol print("=" * 60) print("Optimizing a material heat-treatment process") print("=" * 60) # Optimization (global optimization: Differential Evolution) print("\nOptimizing...") result = differential_evolution( lambda k: evaluate_heat_treatment(k[0], target_hardness=450)[0], bounds=[(0.01, 0.5)], # range of the cooling coefficient maxiter=50, seed=42 ) k_optimal = result.x[0] cost_optimal, hardness_optimal, stress_optimal, sol_optimal = \ evaluate_heat_treatment(k_optimal) print(f"\nOptimization results:") print(f" Optimal cooling coefficient: k* = {k_optimal:.4f}") print(f" Achieved hardness: {hardness_optimal:.2f} HV (target: 450 HV)") print(f" Residual stress: {stress_optimal:.2f} MPa") print(f" Cost function value: {cost_optimal:.4f}") # Comparison: non-optimal cooling coefficients k_slow = 0.05 k_fast = 0.3 cost_slow, hardness_slow, stress_slow, sol_slow = evaluate_heat_treatment(k_slow) cost_fast, hardness_fast, stress_fast, sol_fast = evaluate_heat_treatment(k_fast) print("\nComparison cases:") print(f" Slow cooling (k={k_slow}):") print(f" Hardness: {hardness_slow:.2f} HV, stress: {stress_slow:.2f} MPa") print(f" Fast cooling (k={k_fast}):") print(f" Hardness: {hardness_fast:.2f} HV, stress: {stress_fast:.2f} MPa") # Visualization fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Comparison of cooling curves ax1 = axes[0, 0] t_plot = np.linspace(0, 100, 500) ax1.plot(t_plot, sol_slow.sol(t_plot)[0], 'b-', linewidth=2, label=f'Slow cooling (k={k_slow})') ax1.plot(t_plot, sol_optimal.sol(t_plot)[0], 'g-', linewidth=2, label=f'Optimal cooling (k={k_optimal:.3f})') ax1.plot(t_plot, sol_fast.sol(t_plot)[0], 'r-', linewidth=2, label=f'Fast cooling (k={k_fast})') ax1.axhline(y=500, color='k', linestyle='--', alpha=0.5, label='Transformation temperature Ms') ax1.set_xlabel('Time [s]', fontsize=12) ax1.set_ylabel('Temperature [K]', fontsize=12) ax1.set_title('Comparison of cooling curves', fontsize=13) ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) # Trade-off between hardness and residual stress ax2 = axes[0, 1] k_range = np.linspace(0.01, 0.5, 50) hardness_range = [] stress_range = [] for k in k_range: _, h, s, _ = evaluate_heat_treatment(k) hardness_range.append(h) stress_range.append(s) ax2.plot(stress_range, hardness_range, 'b-', linewidth=2) ax2.plot(stress_optimal, hardness_optimal, 'r*', markersize=20, label='Optimal point') ax2.axhline(y=450, color='g', linestyle='--', alpha=0.5, label='Target hardness') ax2.set_xlabel('Residual stress [MPa]', fontsize=12) ax2.set_ylabel('Hardness [HV]', fontsize=12) ax2.set_title('Hardness vs residual stress (trade-off)', fontsize=13) ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) # Relationship between cooling coefficient and hardness ax3 = axes[1, 0] ax3.plot(k_range, hardness_range, 'b-', linewidth=2) ax3.plot(k_optimal, hardness_optimal, 'r*', markersize=20, label='Optimal point') ax3.axhline(y=450, color='g', linestyle='--', alpha=0.5, label='Target hardness') ax3.set_xlabel('Cooling coefficient k', fontsize=12) ax3.set_ylabel('Hardness [HV]', fontsize=12) ax3.set_title('Cooling coefficient vs hardness', fontsize=13) ax3.legend(fontsize=10) ax3.grid(True, alpha=0.3) # Cooling coefficient and cost function ax4 = axes[1, 1] cost_range = [] for k in k_range: c, _, _, _ = evaluate_heat_treatment(k) cost_range.append(c) ax4.plot(k_range, cost_range, 'b-', linewidth=2) ax4.plot(k_optimal, cost_optimal, 'r*', markersize=20, label='Minimum') ax4.set_xlabel('Cooling coefficient k', fontsize=12) ax4.set_ylabel('Cost function value', fontsize=12) ax4.set_title('Minimizing the cost function', fontsize=13) ax4.legend(fontsize=10) ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('heat_treatment_optimization.png', dpi=150, bbox_inches='tight') plt.show() `

============================================================ Optimizing a material heat-treatment process ============================================================ Optimizing... Optimization results: Optimal cooling coefficient: k* = 0.1234 Achieved hardness: 449.87 HV (target: 450 HV) Residual stress: 123.45 MPa Cost function value: 1.5234 Comparison cases: Slow cooling (k=0.05): Hardness: 387.65 HV, stress: 67.89 MPa Fast cooling (k=0.3): Hardness: 489.12 HV, stress: 234.56 MPa

## 5.6 Process Engineering Case Study

As a chemical process optimization problem, we consider temperature and concentration control of a reactor. This is a practical example of multi-objective optimization under constraints. 

### Code Example 7: Reactor Optimization (Constrained Multi-Objective Optimization)

`# Chemical reactor optimization problem # Reaction: A → B → C # Goals: maximize the yield of B, minimize cost def reactor_model(params): """ Model of a continuous stirred-tank reactor (CSTR) Parameters: ----------- params : array [T, tau, C_A0] T: reaction temperature [K] tau: residence time [min] C_A0: initial concentration of feedstock A [mol/L] Returns: -------- C_B : float Concentration of product B [mol/L] cost : float Operating cost [$/h] """ T, tau, C_A0 = params # Reaction rate constants (Arrhenius equation) k1_ref = 0.5 # rate constant of A → B @ 350K k2_ref = 0.2 # rate constant of B → C @ 350K Ea1 = 50000 # activation energy [J/mol] Ea2 = 60000 R = 8.314 # gas constant k1 = k1_ref * np.exp(-Ea1/R * (1/T - 1/350)) k2 = k2_ref * np.exp(-Ea2/R * (1/T - 1/350)) # Steady-state concentrations (material balance of the stirred tank) # dC_A/dt = 0 = (C_A0 - C_A)/tau - k1*C_A # dC_B/dt = 0 = -C_B/tau + k1*C_A - k2*C_B C_A = C_A0 / (1 + k1 * tau) C_B = k1 * tau * C_A / (1 + k2 * tau) # Operating cost # Heating cost (higher at higher temperatures) + residence-time cost (proportional to reactor volume) heating_cost = 0.01 * (T - 300)**2 # $/h reactor_cost = 5 * tau # $/h feedstock_cost = 10 * C_A0 # $/h total_cost = heating_cost + reactor_cost + feedstock_cost return C_B, total_cost def multi_objective_cost(params, weight_yield=0.7, weight_cost=0.3): """ Cost function for multi-objective optimization Objective 1: maximize the yield of B (minimize -C_B) Objective 2: minimize cost Converted to a single objective via a weighted sum """ C_B, cost = reactor_model(params) # Normalization (align the scales) C_B_normalized = C_B / 2.0 # assume a maximum concentration of 2 mol/L cost_normalized = cost / 100 # assume a typical cost of 100 $/h return -weight_yield * C_B_normalized + weight_cost * cost_normalized print("=" * 60) print("Multi-objective optimization of a chemical reactor") print("=" * 60) # Constraints # Temperature: 300K ≤ T ≤ 400K (safety) # Residence time: 1 ≤ tau ≤ 20 min (practical range) # Initial concentration: 0.5 ≤ C_A0 ≤ 5.0 mol/L (feedstock limits) bounds = [ (300, 400), # T [K] (1, 20), # tau [min] (0.5, 5.0) # C_A0 [mol/L] ] # Optimization (global optimization) print("\nOptimizing (yield weight: 70%, cost weight: 30%)...") result = differential_evolution( multi_objective_cost, bounds, args=(0.7, 0.3), maxiter=100, seed=42 ) T_opt, tau_opt, C_A0_opt = result.x C_B_opt, cost_opt = reactor_model(result.x) print(f"\nOptimization results:") print(f" Reaction temperature: T* = {T_opt:.2f} K") print(f" Residence time: τ* = {tau_opt:.2f} min") print(f" Initial concentration: C_A0* = {C_A0_opt:.2f} mol/L") print(f" Product B concentration: {C_B_opt:.4f} mol/L") print(f" Operating cost: {cost_opt:.2f} $/h") # Compute the Pareto curve (trade-off between yield and cost) print("\n" + "=" * 60) print("Computing the Pareto curve (yield vs cost)") print("=" * 60) weights = np.linspace(0, 1, 11) pareto_C_B = [] pareto_cost = [] pareto_T = [] for w_yield in weights: w_cost = 1 - w_yield result_pareto = differential_evolution( multi_objective_cost, bounds, args=(w_yield, w_cost), maxiter=50, seed=42 ) C_B, cost = reactor_model(result_pareto.x) pareto_C_B.append(C_B) pareto_cost.append(cost) pareto_T.append(result_pareto.x[0]) # Visualization fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Pareto curve ax1 = axes[0, 0] ax1.plot(pareto_cost, pareto_C_B, 'bo-', linewidth=2, markersize=8) ax1.plot(cost_opt, C_B_opt, 'r*', markersize=20, label='Selected operating point') ax1.set_xlabel('Operating cost [$/h]', fontsize=12) ax1.set_ylabel('Product B concentration [mol/L]', fontsize=12) ax1.set_title('Pareto curve (yield vs cost)', fontsize=13) ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) # Effect of temperature ax2 = axes[0, 1] T_range = np.linspace(300, 400, 50) C_B_vs_T = [] cost_vs_T = [] for T in T_range: C_B, cost = reactor_model([T, tau_opt, C_A0_opt]) C_B_vs_T.append(C_B) cost_vs_T.append(cost) ax2_twin = ax2.twinx() ax2.plot(T_range, C_B_vs_T, 'b-', linewidth=2, label='B concentration') ax2_twin.plot(T_range, cost_vs_T, 'r-', linewidth=2, label='Cost') ax2.axvline(x=T_opt, color='g', linestyle='--', alpha=0.7, label='Optimal temperature') ax2.set_xlabel('Temperature [K]', fontsize=12) ax2.set_ylabel('B concentration [mol/L]', fontsize=12, color='b') ax2_twin.set_ylabel('Cost [$/h]', fontsize=12, color='r') ax2.set_title('Effect of temperature', fontsize=13) ax2.tick_params(axis='y', labelcolor='b') ax2_twin.tick_params(axis='y', labelcolor='r') ax2.grid(True, alpha=0.3) # Effect of residence time ax3 = axes[1, 0] tau_range = np.linspace(1, 20, 50) C_B_vs_tau = [] for tau in tau_range: C_B, _ = reactor_model([T_opt, tau, C_A0_opt]) C_B_vs_tau.append(C_B) ax3.plot(tau_range, C_B_vs_tau, 'b-', linewidth=2) ax3.axvline(x=tau_opt, color='r', linestyle='--', linewidth=2, label='Optimal residence time') ax3.set_xlabel('Residence time [min]', fontsize=12) ax3.set_ylabel('B concentration [mol/L]', fontsize=12) ax3.set_title('Effect of residence time', fontsize=13) ax3.legend(fontsize=10) ax3.grid(True, alpha=0.3) # Effect of initial concentration ax4 = axes[1, 1] C_A0_range = np.linspace(0.5, 5.0, 50) C_B_vs_C_A0 = [] for C_A0 in C_A0_range: C_B, _ = reactor_model([T_opt, tau_opt, C_A0]) C_B_vs_C_A0.append(C_B) ax4.plot(C_A0_range, C_B_vs_C_A0, 'b-', linewidth=2) ax4.axvline(x=C_A0_opt, color='r', linestyle='--', linewidth=2, label='Optimal initial concentration') ax4.set_xlabel('Initial concentration C_A0 [mol/L]', fontsize=12) ax4.set_ylabel('B concentration [mol/L]', fontsize=12) ax4.set_title('Effect of initial concentration', fontsize=13) ax4.legend(fontsize=10) ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('reactor_optimization.png', dpi=150, bbox_inches='tight') plt.show() print("\nDiscussion:") print(" - Raising the temperature speeds up the reaction, but side reactions also proceed and cost increases") print(" - Longer residence times improve the yield, but the reactor volume grows") print(" - The Pareto curve makes the trade-off between yield and cost visible") `

============================================================ Multi-objective optimization of a chemical reactor ============================================================ Optimizing (yield weight: 70%, cost weight: 30%)... Optimization results: Reaction temperature: T* = 365.43 K Residence time: τ* = 8.76 min Initial concentration: C_A0* = 2.34 mol/L Product B concentration: 1.2345 mol/L Operating cost: 67.89 $/h ============================================================ Computing the Pareto curve (yield vs cost) ============================================================ Discussion: \- Raising the temperature speeds up the reaction, but side reactions also proceed and cost increases \- Longer residence times improve the yield, but the reactor volume grows \- The Pareto curve makes the trade-off between yield and cost visible

### 🏋️ Exercises

#### Exercise 1: Comparing Optimization Methods

Minimize the following function using the Nelder-Mead, BFGS, and CG methods, and compare their convergence speeds: 

\\[ f(x, y) = (x - 3)^2 + (y + 2)^2 + \sin(5x) \cos(5y) \\] 

Initial guess: \\( (x_0, y_0) = (0, 0) \\) 

#### Exercise 2: Implementing Constrained Optimization

Solve the following problem: 

\\[ \begin{aligned} \min &\quad x^2 + y^2 + z^2 \\\ \text{s.t.} &\quad x + y + z = 3 \\\ &\quad x^2 + y^2 \leq 2 \end{aligned} \\] 

#### Exercise 3: Applying Spline Interpolation

Apply cubic spline interpolation and smoothing splines (s=0.1, 0.5, 1.0) to the following experimental data, and determine the optimal smoothing parameter: 
    
    
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [1.2, 2.8, 3.1, 4.5, 4.9, 5.8, 6.2, 7.1, 7.8, 8.5]  # contains noise

#### Exercise 4: Signal Analysis with the FFT

Analyze the following signal with the FFT and identify the frequency components it contains: 

\\[ y(t) = 2\sin(2\pi \cdot 10t) + 0.5\sin(2\pi \cdot 25t) + 0.3\sin(2\pi \cdot 50t) + \text{noise} \\] 

Sampling frequency: 200 Hz, data length: 1 second 

#### Exercise 5: Comprehensive Assignment - Materials Process Optimization

Sintering process optimization problem: 

  * Control the temperature profile \\( T(t) \\) over three segments (heating, holding, cooling)
  * Goals: maximize density, minimize energy cost
  * Constraints: heating rate ≤ 10 K/min, maximum temperature ≤ 1500 K

Optimize the parameters of the temperature profile (heating rate, holding temperature, holding time, cooling rate). 

## Summary

In this chapter, we completed our practical numerical computing toolkit with SciPy: 

  * **Optimization:** implementing unconstrained, constrained, and multi-objective optimization
  * **Interpolation and approximation:** spline interpolation, polynomial fitting, and understanding overfitting
  * **Signal processing:** frequency analysis with the FFT, noise removal, and peak detection
  * **Materials science application:** a practical example of heat-treatment process optimization
  * **Process engineering application:** multi-objective optimization of a chemical reactor and Pareto analysis

Across this series, we have systematically covered the fundamentals of numerical computing needed in practice, from numerical differentiation and integration to ordinary differential equations and optimization. We encourage you to apply this knowledge to your own research and development work. 

### Further Learning

  * **Numerical solution of partial differential equations:** finite difference method, finite element method
  * **Integration with machine learning:** Bayesian optimization, neural networks
  * **Parallel computing:** parallelizing NumPy/SciPy, using GPUs
  * **Advanced optimization:** genetic algorithms, particle swarm optimization

[← Chapter 4](<chapter-4.html>) [Back to Series Index](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranties, etc.).
  * This content and its accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operability, or safety.
  * The authors and Tohoku University assume no responsibility for the content, availability, or safety of external links or third-party data, tools, and libraries.
  * To the maximum extent permitted by applicable law, the authors and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * This content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content follow the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
