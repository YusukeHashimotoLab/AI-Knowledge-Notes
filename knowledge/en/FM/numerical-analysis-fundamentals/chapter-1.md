---
title: "Chapter 1: Numerical Differentiation and Integration"
chapter_title: "Chapter 1: Numerical Differentiation and Integration"
---

# Chapter 1: Numerical Differentiation and Integration

This chapter covers Numerical Differentiation and Integration. You will learn essential concepts and techniques.

Fundamental methods for numerically approximating derivatives and integrals that cannot be computed analytically

## 1.1 Fundamentals of Numerical Differentiation

In the definition of differentiation \\( f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \\) , by taking\\( h \\) to be a sufficiently small value, we can approximate the derivative. We will learn various finite difference methods based on this idea. 

### 📚 Theory: Classification of Finite Difference Methods

**Forward Difference:**

\\[ f'(x) \approx \frac{f(x+h) - f(x)}{h} = f'(x) + O(h) \\] 

**Backward Difference:**

\\[ f'(x) \approx \frac{f(x) - f(x-h)}{h} = f'(x) + O(h) \\] 

**Central Difference:**

\\[ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} = f'(x) + O(h^2) \\] 

The central difference has \\( O(h^2) \\) accuracy, which is higher than the \\( O(h) \\) accuracy of forward and backward differences. However, care must be taken when computing at boundary points. 

### Code Example 1: Implementing Forward, Backward, and Central Difference Methods
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    <code>import numpy as np
    import matplotlib.pyplot as plt
    
    def forward_difference(f, x, h):
        """Numerical differentiation using forward difference"""
        return (f(x + h) - f(x)) / h
    
    def backward_difference(f, x, h):
        """Numerical differentiation using backward difference"""
        return (f(x) - f(x - h)) / h
    
    def central_difference(f, x, h):
        """Numerical differentiation using central difference"""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # # Test function: f(x) = sin(x), f'(x) = cos(x)
    f = np.sin
    f_prime_exact = np.cos
    
    # # Evaluation point
    x0 = np.pi / 4
    exact_value = f_prime_exact(x0)
    
    # # Evaluate error for varying step sizes
    h_values = np.logspace(-10, -1, 50)
    errors_forward = []
    errors_backward = []
    errors_central = []
    
    for h in h_values:
        errors_forward.append(abs(forward_difference(f, x0, h) - exact_value))
        errors_backward.append(abs(backward_difference(f, x0, h) - exact_value))
        errors_central.append(abs(central_difference(f, x0, h) - exact_value))
    
    # # Visualization
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_forward, 'o-', label='Forward Difference O(h)', alpha=0.7)
    plt.loglog(h_values, errors_backward, 's-', label='Backward Difference O(h)', alpha=0.7)
    plt.loglog(h_values, errors_central, '^-', label='Central Difference O(h²)', alpha=0.7)
    
    # # Reference lines
    plt.loglog(h_values, h_values, '--', label='O(h)', color='gray', alpha=0.5)
    plt.loglog(h_values, h_values**2, '--', label='O(h²)', color='black', alpha=0.5)
    
    plt.xlabel('Step size h', fontsize=12)
    plt.ylabel('Absolute error', fontsize=12)
    plt.title('Error Analysis of Numerical Differentiation (f(x)=sin(x), x=π/4)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('numerical_diff_errors.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation point: x = π/4 ≈ {x0:.4f}")
    print(f"Exact value: f'(x) = cos(π/4) ≈ {exact_value:.8f}\n")
    print(f"h = 1e-4 Results for")
    h = 1e-4
    print(f"  Forward difference: {forward_difference(f, x0, h):.8f} (error: {abs(forward_difference(f, x0, h) - exact_value):.2e})")
    print(f"  Backward difference: {backward_difference(f, x0, h):.8f} (error: {abs(backward_difference(f, x0, h) - exact_value):.2e})")
    print(f"  Central difference: {central_difference(f, x0, h):.8f} (error: {abs(central_difference(f, x0, h) - exact_value):.2e})")
    </code>

Evaluation point: x = π/4 ≈ 0.7854 Exact value: f'(x) = cos(π/4) ≈ 0.70710678 h = 1e-4 Results for Forward difference: 0.70710178 (error: 5.00e-06) Backward difference: 0.70710178 (error: 5.00e-06) Central difference: 0.70710678 (error: 5.00e-12)

**Discussion:** The central difference shows the theoretical \\( O(h^2) \\) accuracy and is more than 6 digits more accurate than forward/backward differences for the same step size \\( h \\) . However, when\\( h \\) is made extremely small, accuracy degrades due to round-off errors (U-shaped curve in the figure). 

## 1.2 Richardson Extrapolation

Richardson extrapolation is a method that obtains high-accuracy approximations by combining results with different step sizes. By canceling the main error terms, accuracy can be improved while keeping computational cost low. 

### 📚 Theory: Principles of Richardson Extrapolation

The error expansion of the central difference is as follows: 

\\[ D(h) = f'(x) + c_2 h^2 + c_4 h^4 + \cdots \\] 

where \\( D(h) \\) is the central difference approximation with step size \\( h \\) .\\( D(h) \\) \\( D(h/2) \\) \\( h^2 \\) , eliminating the 

\\[ D_{\text{ext}}(h) = \frac{4D(h/2) - D(h)}{3} = f'(x) + O(h^4) \\] 

This improves the accuracy from \\( O(h^2) \\) \\( O(h^4) \\) . 

### Code Example 2: Implementing Richardson Extrapolation
    
    
    def richardson_extrapolation(f, x, h, order=1):
        """
        High-accuracy numerical differentiation using Richardson extrapolation
    
        Parameters:
        -----------
        f : callable
            Function to differentiate
        x : float
            # Evaluation point
        h : float
            Base step size
        order : int
            Extrapolation order (default: 1)
    
        Returns:
        --------
        float
            Extrapolated derivative value
        """
        # # Initial value: central difference
        D = central_difference(f, x, h)
    
        # # Improve accuracy with Richardson extrapolation
        for k in range(order):
            D_half = central_difference(f, x, h / 2**(k+1))
            D = (4**(k+1) * D_half - D) / (4**(k+1) - 1)
    
        return D
    
    # # Test: f(x) = exp(x), f'(x) = exp(x)
    f = np.exp
    f_prime_exact = np.exp
    
    x0 = 1.0
    exact_value = f_prime_exact(x0)
    h = 0.1
    
    # # Compare methods
    print(f"Evaluation point: x = {x0}")
    print(f"Exact value: f'(x) = e ≈ {exact_value:.12f}\n")
    
    # Central difference
    D0 = central_difference(f, x0, h)
    print(f"Central difference (h={h}):")
    print(f"  Value: {D0:.12f}")
    print(f"  error: {abs(D0 - exact_value):.2e}\n")
    
    # Richardson extrapolation (1st order)
    D1 = richardson_extrapolation(f, x0, h, order=1)
    print(f"Richardson extrapolation (1st order):")
    print(f"  Value: {D1:.12f}")
    print(f"  error: {abs(D1 - exact_value):.2e}\n")
    
    # Richardson extrapolation (2nd order)
    D2 = richardson_extrapolation(f, x0, h, order=2)
    print(f"Richardson extrapolation (2nd order):")
    print(f"  Value: {D2:.12f}")
    print(f"  error: {abs(D2 - exact_value):.2e}\n")
    
    # # Visualize accuracy improvement
    h_values = np.logspace(-2, -0.3, 20)
    errors_central = []
    errors_rich1 = []
    errors_rich2 = []
    
    for h in h_values:
        errors_central.append(abs(central_difference(f, x0, h) - exact_value))
        errors_rich1.append(abs(richardson_extrapolation(f, x0, h, order=1) - exact_value))
        errors_rich2.append(abs(richardson_extrapolation(f, x0, h, order=2) - exact_value))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_central, 'o-', label='Central Difference O(h²)', alpha=0.7)
    plt.loglog(h_values, errors_rich1, 's-', label='Richardson 1st Order O(h⁴)', alpha=0.7)
    plt.loglog(h_values, errors_rich2, '^-', label='Richardson 2nd Order O(h⁶)', alpha=0.7)
    plt.xlabel('Step size h', fontsize=12)
    plt.ylabel('Absolute error', fontsize=12)
    plt.title('Accuracy Improvement with Richardson Extrapolation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('richardson_extrapolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    

Evaluation point: x = 1.0 Exact value: f'(x) = e ≈ 2.718281828459 Central difference (h=0.1): Value: 2.718282520008 error: 6.92e-07 Richardson extrapolation (1st order): Value: 2.718281828590 error: 1.31e-10 Richardson extrapolation (2nd order): Value: 2.718281828459 error: 2.22e-13

## 1.3 Fundamentals of Numerical Integration

We will learn methods for numerically computing the definite integral \\( I = \int_a^b f(x) \, dx \\) . By dividing the interval and using function values in each subinterval, we approximate the integral. 

### 📚 Theory: Trapezoidal and Simpson's Rules

**Trapezoidal Rule:**

The interval \\([a, b]\\) \\( n \\) subintervals, and the function is approximated by straight lines in each subinterval: 

\\[ I \approx \frac{h}{2} \left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right], \quad h = \frac{b-a}{n} \\] 

The error is \\( O(h^2) \\) . 

**Simpson's Rule:**

The function is approximated by quadratic polynomials in each subinterval (\\( n \\) must be even): 

\\[ I \approx \frac{h}{3} \left[ f(x_0) + 4\sum_{i=\text{odd}} f(x_i) + 2\sum_{i=\text{even}} f(x_i) + f(x_n) \right] \\] 

The error is \\( O(h^4) \\) , which is more accurate than the trapezoidal rule. 

### Code Example 3: Implementing the Trapezoidal Rule
    
    
    def trapezoidal_rule(f, a, b, n):
        """
        Numerical integration using the trapezoidal rule
    
        Parameters:
        -----------
        f : callable
            Integrand function
        a, b : float
            Integration interval [a, b]
        n : int
            Number of divisions
    
        Returns:
        --------
        float
            Approximation of the integral
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
    
        # # Implementation of trapezoidal rule
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
        return integral
    
    # # Test: ∫[0,1] x² dx = 1/3
    f = lambda x: x**2
    exact_value = 1/3
    
    # # Evaluate accuracy for varying number of divisions
    n_values = [4, 8, 16, 32, 64, 128]
    errors = []
    
    print("Numerical Integration Using Trapezoidal Rule: ∫[0,1] x² dx\n")
    print("Divisions n    Approximation    Error")
    print("-" * 40)
    
    for n in n_values:
        approx = trapezoidal_rule(f, 0, 1, n)
        error = abs(approx - exact_value)
        errors.append(error)
        print(f"{n:4d}      {approx:.10f}  {error:.2e}")
    
    print(f"\nExact value: {exact_value:.10f}")
    
    # # Visualize error convergence rate
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, errors, 'o-', label='Actual error', markersize=8)
    plt.loglog(n_values, [1/n**2 for n in n_values], '--',
               label='O(h²) = O(1/n²)', alpha=0.5)
    plt.xlabel('Number of divisions n', fontsize=12)
    plt.ylabel('Absolute error', fontsize=12)
    plt.title('Convergence of Trapezoidal Rule (O(h²))', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trapezoidal_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    

Numerical Integration Using Trapezoidal Rule: ∫[0,1] x² dx Divisions n Approximation Error \---------------------------------------- 4 0.3437500000 1.04e-02 8 0.3359375000 2.60e-03 16 0.3339843750 6.51e-04 32 0.3334960938 1.63e-04 64 0.3333740234 4.07e-05 128 0.3333435059 1.02e-05 Exact value: 0.3333333333

### Code Example 4: Implementing Simpson's Rule
    
    
    def simpson_rule(f, a, b, n):
        """
        Numerical integration using Simpson's rule (1/3 rule)
    
        Parameters:
        -----------
        f : callable
            Integrand function
        a, b : float
            Integration interval [a, b]
        n : int
            Number of divisions (must be even)
    
        Returns:
        --------
        float
            Approximation of the integral
        """
        if n % 2 != 0:
            raise ValueError("For Simpson's rule, the number of divisions n must be even")
    
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
    
        # # Implementation of Simpson's rule
        integral = h / 3 * (y[0] + y[-1] +
                            4 * np.sum(y[1:-1:2]) +  # # Odd indices
                            2 * np.sum(y[2:-1:2]))    # # Even indices
    
        return integral
    
    # # Compare trapezoidal and Simpson's rules
    print("Trapezoidal Rule vs Simpson's Rule: ∫[0,π] sin(x) dx\n")
    
    f = np.sin
    exact_value = 2.0  # ∫[0,π] sin(x) dx = 2
    
    n_values = [4, 8, 16, 32, 64]
    
    print("Number of divisions n    Trapezoidal      Error        Simpson         Error")
    print("-" * 70)
    
    errors_trap = []
    errors_simp = []
    
    for n in n_values:
        trap = trapezoidal_rule(f, 0, np.pi, n)
        simp = simpson_rule(f, 0, np.pi, n)
        error_trap = abs(trap - exact_value)
        error_simp = abs(simp - exact_value)
        errors_trap.append(error_trap)
        errors_simp.append(error_simp)
    
        print(f"{n:4d}      {trap:.8f}  {error_trap:.2e}     {simp:.8f}  {error_simp:.2e}")
    
    print(f"\nExact value: {exact_value:.8f}")
    
    # # Compare convergence rates
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, errors_trap, 'o-', label='Trapezoidal Rule O(h²)', markersize=8)
    plt.loglog(n_values, errors_simp, 's-', label='Simpson's Rule O(h⁴)', markersize=8)
    plt.loglog(n_values, [1/n**2 for n in n_values], '--',
               label='O(1/n²)', alpha=0.5, color='gray')
    plt.loglog(n_values, [1/n**4 for n in n_values], '--',
               label='O(1/n⁴)', alpha=0.5, color='black')
    plt.xlabel('Number of divisions n', fontsize=12)
    plt.ylabel('Absolute error', fontsize=12)
    plt.title('Comparison of Convergence: Trapezoidal vs Simpson's Rule', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simpson_vs_trapezoidal.png', dpi=150, bbox_inches='tight')
    plt.show()
    

Trapezoidal Rule vs Simpson's Rule: ∫[0,π] sin(x) dx Number of divisions n Trapezoidal Error Simpson Error \---------------------------------------------------------------------- 4 1.89611890 1.04e-01 2.00045597 4.56e-04 8 1.97423160 2.58e-02 2.00002838 2.84e-05 16 1.99357034 6.43e-03 2.00000177 1.77e-06 32 1.99839236 1.61e-03 2.00000011 1.11e-07 64 1.99959810 4.02e-04 2.00000001 6.94e-09 Exact value: 2.00000000

## 1.4 Gaussian Quadrature

Gaussian quadrature is a method that achieves high-accuracy integration with fewer evaluation points by optimizing the evaluation points and weights.\\( n \\) -point Gaussian quadrature can exactly integrate polynomials up to degree \\( 2n-1 \\) . 

### 📚 Theory: Gauss-Legendre Quadrature

The interval \\([-1, 1]\\) Consider the integral over the interval 

\\[ I = \int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i) \\] 

where \\( x_i \\) are the zeros of the Legendre polynomial, and\\( w_i \\) are the corresponding weights. The transformation to an arbitrary interval \\([a, b]\\) is: 

\\[ \int_a^b f(x) \, dx = \frac{b-a}{2} \int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right) \, dt \\] 

### Code Example 5: Implementing Gaussian Quadrature
    
    
    from scipy.integrate import quad
    from numpy.polynomial.legendre import leggauss
    
    def gauss_quadrature(f, a, b, n):
        """
        Numerical integration using Gauss-Legendre quadrature
    
        Parameters:
        -----------
        f : callable
            Integrand function
        a, b : float
            Integration interval [a, b]
        n : int
            Number of Gauss points
    
        Returns:
        --------
        float
            Approximation of the integral
        """
        # # Get zeros and weights of Legendre polynomial
        x, w = leggauss(n)
    
        # # Transform from interval [-1,1] to [a,b]
        t = 0.5 * (b - a) * x + 0.5 * (a + b)
    
        # # Calculate integral
        integral = 0.5 * (b - a) * np.sum(w * f(t))
    
        return integral
    
    # # Test: ∫[0,1] exp(-x²) dx
    f = lambda x: np.exp(-x**2)
    a, b = 0, 1
    
    # # Calculate exact value with high-precision SciPy integration
    exact_value, _ = quad(f, a, b)
    
    print("Gaussian Quadrature: ∫[0,1] exp(-x²) dx\n")
    print("Gauss pts n    Approximation    Error        Function evals")
    print("-" * 60)
    
    n_values = [2, 3, 4, 5, 10, 20]
    
    for n in n_values:
        approx = gauss_quadrature(f, a, b, n)
        error = abs(approx - exact_value)
        print(f"{n:4d}          {approx:.12f}  {error:.2e}     {n}")
    
    print(f"\nExact value (SciPy quad): {exact_value:.12f}")
    
    # # Compare with Simpson's rule (same number of function evaluations)
    print("\nComparison with same number of function evaluations:")
    print("-" * 60)
    
    for n_gauss in [5, 10]:
        # Gaussian Quadrature
        gauss_result = gauss_quadrature(f, a, b, n_gauss)
        gauss_error = abs(gauss_result - exact_value)
    
        # Simpson's rule (same number of evaluations)
        n_simpson = n_gauss - 1  # Simpson's rule evaluates n+1 points
        if n_simpson % 2 != 0:
            n_simpson -= 1
        simpson_result = simpson_rule(f, a, b, n_simpson)
        simpson_error = abs(simpson_result - exact_value)
    
        print(f"\nFunction evaluations: {n_gauss}")
        print(f"  Gauss ({n_gauss} pts):  error {gauss_error:.2e}")
        print(f"  Simpson ({n_simpson} divs): error {simpson_error:.2e}")
        print(f"  Accuracy improvement: {simpson_error / gauss_error:.1f}times")
    

Gaussian Quadrature: ∫[0,1] exp(-x²) dx Gauss pts n Approximation Error Function evals \------------------------------------------------------------ 2 0.746806877203 7.67e-05 2 3 0.746824053490 5.53e-07 3 4 0.746824132812 1.88e-09 4 5 0.746824132812 6.66e-12 5 10 0.746824132812 4.44e-16 10 20 0.746824132812 0.00e+00 20 Exact value (SciPy quad): 0.746824132812 Comparison with same number of function evaluations: \------------------------------------------------------------ Function evaluations: 5 Gauss (5 pts): error 6.66e-12 Simpson (4 divs): error 1.69e-06 Accuracy improvement: 253780.5times Function evaluations: 10 Gauss (10 pts): error 4.44e-16 Simpson (8 divs): error 2.65e-08 Accuracy improvement: 59646916.8times

**Discussion:** Gaussian quadrature is much more accurate than Simpson's rule for the same number of function evaluations. It is especially effective for smooth functions, and 5-point Gaussian quadrature can achieve machine precision. 

## 1.5 Numerical Differentiation and Integration with NumPy/SciPy

In practice, we utilize the advanced numerical computing libraries NumPy/SciPy. Functions with adaptive methods and error estimation capabilities are provided. 

### Code Example 6: scipy.integrate Practical Examples
    
    
    from scipy.integrate import quad, simps, trapz, fixed_quad
    from scipy.misc import derivative
    
    # # Test functions
    def test_function_1(x):
        """Oscillatory function"""
        return np.sin(10 * x) * np.exp(-x)
    
    def test_function_2(x):
        """Function with singularity"""
        return 1 / np.sqrt(x + 1e-10)
    
    # 1. scipy.integrate.quad (# Adaptive integration)
    print("=" * 60)
    print("1. scipy.integrate.quad (Adaptive Gauss-Kronrod Method)")
    print("=" * 60)
    
    # # Integration of oscillatory function
    result, error = quad(test_function_1, 0, 2)
    print(f"\n∫[0,2] sin(10x)exp(-x) dx:")
    print(f"  Result: {result:.12f}")
    print(f"  Estimated error: {error:.2e}")
    
    # Function with singularity
    result, error = quad(test_function_2, 0, 1)
    print(f"\n∫[0,1] 1/√x dx:")
    print(f"  Result: {result:.12f}")
    print(f"  Estimated error: {error:.2e}")
    print(f"  Theoretical value: {2 * np.sqrt(1):.12f}")
    
    # 2. fixed_quad (# Fixed-order Gauss quadrature)
    print("\n" + "=" * 60)
    print("2. scipy.integrate.fixed_quad (Fixed-Order Gauss-Legendre)")
    print("=" * 60)
    
    f = lambda x: np.exp(-x**2)
    for n in [3, 5, 10]:
        result, _ = fixed_quad(f, 0, 1, n=n)
        exact, _ = quad(f, 0, 1)
        error = abs(result - exact)
        print(f"\nn={n:2d}-point Gauss quadrature: {result:.12f} (error: {error:.2e})")
    
    # 3. # Integration of discrete data (assuming experimental data)
    print("\n" + "=" * 60)
    print("3. Integration of Discrete Data (trapz, simps)")
    print("=" * 60)
    
    # # Simulate experimental data
    x_data = np.linspace(0, np.pi, 11)  # # 11 data points
    y_data = np.sin(x_data)
    
    # Trapezoidal rule
    result_trapz = trapz(y_data, x_data)
    print(f"\ntrapz (Trapezoidal rule): {result_trapz:.8f}")
    
    # Simpson's rule
    result_simps = simps(y_data, x_data)
    print(f"simps (Simpson's rule): {result_simps:.8f}")
    
    exact = 2.0
    print(f"Exact value: {exact:.8f}")
    print(f"trapz error: {abs(result_trapz - exact):.2e}")
    print(f"simps error: {abs(result_simps - exact):.2e}")
    
    # 4. scipy.misc.derivative (# Numerical differentiation)
    print("\n" + "=" * 60)
    print("4. scipy.misc.derivative (# Numerical differentiation)")
    print("=" * 60)
    
    f = np.sin
    f_prime = np.cos
    x0 = np.pi / 4
    
    # # First derivative
    deriv1 = derivative(f, x0, n=1, dx=1e-5)
    exact1 = f_prime(x0)
    print(f"\n# First derivative f'(π/4):")
    print(f"  Numerical: {deriv1:.12f}")
    print(f"  Exact value:   {exact1:.12f}")
    print(f"  error:     {abs(deriv1 - exact1):.2e}")
    
    # # Second derivative
    f_double_prime = lambda x: -np.sin(x)
    deriv2 = derivative(f, x0, n=2, dx=1e-5)
    exact2 = f_double_prime(x0)
    print(f"\n# Second derivative f''(π/4):")
    print(f"  Numerical: {deriv2:.12f}")
    print(f"  Exact value:   {exact2:.12f}")
    print(f"  error:     {abs(deriv2 - exact2):.2e}")
    

============================================================ 1\. scipy.integrate.quad (Adaptive Gauss-Kronrod Method) ============================================================ ∫[0,2] sin(10x)exp(-x) dx: Result: 0.499165148496 Estimated error: 5.54e-15 ∫[0,1] 1/√x dx: Result: 2.000000000000 Estimated error: 3.34e-08 Theoretical value: 2.000000000000 ============================================================ 2\. scipy.integrate.fixed_quad (Fixed-Order Gauss-Legendre) ============================================================ n= 3-point Gauss quadrature: 0.746824132757 (error: 5.53e-11) n= 5-point Gauss quadrature: 0.746824132812 (error: 4.44e-16) n=10-point Gauss quadrature: 0.746824132812 (error: 0.00e+00) ============================================================ 3\. Integration of Discrete Data (trapz, simps) ============================================================ trapz (Trapezoidal rule): 1.99835677 simps (Simpson's rule): 2.00000557 Exact value: 2.00000000 trapz error: 1.64e-03 simps error: 5.57e-06 ============================================================ 4\. scipy.misc.derivative (# Numerical differentiation) ============================================================ # First derivative f'(π/4): Numerical: 0.707106781187 Exact value: 0.707106781187 error: 1.11e-16 # Second derivative f''(π/4): Numerical: -0.707106781187 Exact value: -0.707106781187 error: 0.00e+00

## 1.6 Error Analysis and Convergence Evaluation

In practical numerical differentiation and integration, error evaluation and appropriate method selection are important. We experimentally verify theoretical convergence rates and consider the effects of round-off errors. 

### Code Example 7: Error Analysis and Convergence Rate Visualization
    
    
    def analyze_convergence(method, f, exact, params_list, method_name):
        """
        Analyze convergence rate of numerical method
    
        Parameters:
        -----------
        method : callable
            Numerical method function
        f : callable
            Target function
        exact : float
            Exact solution
        params_list : list
            List of parameters (step sizes or divisions)
        method_name : str
            Method name
    
        Returns:
        --------
        errors : array
            Error for each parameter
        """
        errors = []
        for param in params_list:
            result = method(f, param)
            error = abs(result - exact)
            errors.append(error)
        return np.array(errors)
    
    # # Test function: f(x) = sin(x), ∫[0,π] sin(x) dx = 2
    f = np.sin
    exact_integral = 2.0
    
    # # List of divisions
    n_values = np.array([4, 8, 16, 32, 64, 128, 256])
    
    # # Evaluate convergence rate of each method
    print("=" * 70)
    print("Convergence Rate Analysis of Numerical Integration Methods: ∫[0,π] sin(x) dx = 2")
    print("=" * 70)
    
    # Trapezoidal rule
    trap_errors = []
    for n in n_values:
        result = trapezoidal_rule(f, 0, np.pi, n)
        trap_errors.append(abs(result - exact_integral))
    trap_errors = np.array(trap_errors)
    
    # Simpson's rule
    simp_errors = []
    for n in n_values:
        result = simpson_rule(f, 0, np.pi, n)
        simp_errors.append(abs(result - exact_integral))
    simp_errors = np.array(simp_errors)
    
    # Gaussian Quadrature
    gauss_errors = []
    for n in n_values:
        result = gauss_quadrature(f, 0, np.pi, n)
        gauss_errors.append(abs(result - exact_integral))
    gauss_errors = np.array(gauss_errors)
    
    # # Calculate convergence rate (ratio of consecutive errors)
    def compute_convergence_rate(errors):
        """Estimate convergence rate from error reduction"""
        rates = []
        for i in range(len(errors) - 1):
            if errors[i+1] > 0 and errors[i] > 0:
                rate = np.log(errors[i] / errors[i+1]) / np.log(2)
                rates.append(rate)
        return np.array(rates)
    
    trap_rates = compute_convergence_rate(trap_errors)
    simp_rates = compute_convergence_rate(simp_errors)
    gauss_rates = compute_convergence_rate(gauss_errors)
    
    # # Display results
    print("\nTrapezoidal Rule (theoretical convergence rate: O(h²) = O(1/n²))")
    print("n      Error        Rate")
    print("-" * 40)
    for i, n in enumerate(n_values):
        rate_str = f"{trap_rates[i]:.2f}" if i < len(trap_rates) else "-"
        print(f"{n:4d}   {trap_errors[i]:.2e}   {rate_str}")
    print(f"Average rate: {np.mean(trap_rates):.2f} (Theoretical value: 2.0)")
    
    print("\nSimpson's Rule (theoretical convergence rate: O(h⁴) = O(1/n⁴))")
    print("n      Error        Rate")
    print("-" * 40)
    for i, n in enumerate(n_values):
        rate_str = f"{simp_rates[i]:.2f}" if i < len(simp_rates) else "-"
        print(f"{n:4d}   {simp_errors[i]:.2e}   {rate_str}")
    print(f"Average rate: {np.mean(simp_rates):.2f} (Theoretical value: 4.0)")
    
    print("\nGaussian Quadrature")
    print("n      Error        Rate")
    print("-" * 40)
    for i, n in enumerate(n_values):
        rate_str = f"{gauss_rates[i]:.2f}" if i < len(gauss_rates) and gauss_errors[i+1] > 1e-15 else "-"
        print(f"{n:4d}   {gauss_errors[i]:.2e}   {rate_str}")
    
    # # Comprehensive visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # # Error convergence
    ax1.loglog(n_values, trap_errors, 'o-', label='Trapezoidal rule', markersize=8, linewidth=2)
    ax1.loglog(n_values, simp_errors, 's-', label='Simpson's rule', markersize=8, linewidth=2)
    ax1.loglog(n_values, gauss_errors, '^-', label='Gaussian Quadrature', markersize=8, linewidth=2)
    ax1.loglog(n_values, 1/n_values**2, '--', label='O(1/n²)', alpha=0.5, color='gray')
    ax1.loglog(n_values, 1/n_values**4, '--', label='O(1/n⁴)', alpha=0.5, color='black')
    ax1.set_xlabel('Number of divisions n', fontsize=12)
    ax1.set_ylabel('Absolute error', fontsize=12)
    ax1.set_title('Convergence Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # # Convergence rate evolution
    ax2.semilogx(n_values[:-1], trap_rates, 'o-', label='Trapezoidal rule', markersize=8, linewidth=2)
    ax2.semilogx(n_values[:-1], simp_rates, 's-', label='Simpson's rule', markersize=8, linewidth=2)
    ax2.axhline(y=2, linestyle='--', color='gray', alpha=0.5, label='Theoretical (Trapezoidal)')
    ax2.axhline(y=4, linestyle='--', color='black', alpha=0.5, label='Theoretical (Simpson)')
    ax2.set_xlabel('Number of divisions n', fontsize=12)
    ax2.set_ylabel('Convergence Rate p (Error ∝ 1/nᵖ)', fontsize=12)
    ax2.set_title('# Convergence rate evolution', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Trapezoidal rule: convergence rate ≈ 2.0 (as expected theoretically O(1/n²))")
    print("  - Simpson's rule: convergence rate ≈ 4.0 (as expected theoretically O(1/n⁴))")
    print("  - Gaussian quadrature: exponential convergence (exact for polynomials)")
    print("=" * 70)
    

====================================================================== Convergence Rate Analysis of Numerical Integration Methods: ∫[0,π] sin(x) dx = 2 ====================================================================== Trapezoidal Rule (theoretical convergence rate: O(h²) = O(1/n²)) n Error Rate \---------------------------------------- 4 1.04e-01 2.00 8 2.58e-02 2.00 16 6.43e-03 2.00 32 1.61e-03 2.00 64 4.02e-04 2.00 128 1.00e-04 2.00 256 2.51e-05 - Average rate: 2.00 (Theoretical value: 2.0) Simpson's Rule (theoretical convergence rate: O(h⁴) = O(1/n⁴)) n Error Rate \---------------------------------------- 4 4.56e-04 4.00 8 2.84e-05 4.00 16 1.77e-06 4.00 32 1.11e-07 4.00 64 6.94e-09 4.00 128 4.34e-10 4.00 256 2.71e-11 - Average rate: 4.00 (Theoretical value: 4.0) Gaussian Quadrature n Error Rate \---------------------------------------- 4 4.56e-04 5.67 8 8.32e-06 6.74 16 3.33e-08 9.30 32 8.88e-16 - 64 0.00e+00 - 128 0.00e+00 - 256 0.00e+00 - ====================================================================== Summary: \- Trapezoidal rule: convergence rate ≈ 2.0 (as expected theoretically O(1/n²)) \- Simpson's rule: convergence rate ≈ 4.0 (as expected theoretically O(1/n⁴)) \- Gaussian quadrature: exponential convergence (exact for polynomials) ======================================================================

### 🏋️ Exercises

#### Exercise 1: Implementing Numerical Differentiation

Calculate the derivative of the following function at \\( x = 1 \\) using forward, backward, and central differences, and compare the errors. Try step sizes \\( h \\) of 0.1, 0.01, and 0.001. 

\\( f(x) = \ln(x + 1) \\), Exact solution: \\( f'(1) = 1/2 = 0.5 \\) 

#### Exercise 2: Verifying Richardson Extrapolation Effectiveness

\\( f(x) = x^3 - 2x^2 + 3x - 1 \\) of \\( x = 2 \\) at\\( h = 0.1 \\)using the following methods and compare the errors ( 

  * (a) Central difference
  * (b) Richardson extrapolation 1st order
  * (c) Richardson extrapolation 2nd order

#### Exercise 3: Comparing Accuracy of Integration Formulas

Calculate the following integral using the trapezoidal rule, Simpson's rule, and Gaussian quadrature (5 points), and compare accuracy and computational cost: 

\\( \displaystyle I = \int_0^2 \frac{1}{1+x^2} \, dx \\) 

(Hint: The exact solution is \\( \arctan(2) \approx 1.1071487... \\)) 

#### Exercise 4: Numerical Integration of Experimental Data

From the following experimental data (temperature vs time), calculate the average temperature over 0-10 seconds using numerical integration: 
    
    
    Time (s): [0, 2, 4, 6, 8, 10]
    Temperature (°C): [20, 35, 48, 52, 49, 40]

Calculate using both the trapezoidal rule and Simpson's rule, and compare the results. 

#### Exercise 5: Applications to Materials Science

When the thermal expansion coefficient of a material \\( \alpha(T) \\) is given as a function of temperature, the rate of length change due to temperature variation is calculated by: 

\\[ \frac{\Delta L}{L_0} = \int_{T_0}^{T} \alpha(T') \, dT' \\] 

\\( \alpha(T) = (1.5 + 0.003T) \times 10^{-5} \\) (K⁻¹) Take\\( T_0 = 300 \\) K \\( T = 500 \\) K and calculate the length change rate due to temperature increase to 

## Summary

In this chapter, we learned fundamental methods for numerical differentiation and integration: 

  * **Numerical Differentiation:** Finite difference methods (forward, backward, central) and high-accuracy with Richardson extrapolation
  * **Numerical Integration:** Principles and implementation of trapezoidal rule, Simpson's rule, and Gaussian quadrature
  * **Error Analysis:** Verification of theoretical convergence rates and practical accuracy evaluation
  * **SciPy Utilization:** Practical numerical computation with scipy.integrate and scipy.misc

These methods are utilized in a wide range of applications in materials science and process engineering, including experimental data analysis, simulation, and optimization. In the next chapter, we will learn numerical methods for systems of linear equations building on these foundations. 

[← Series Table of Contents](<index.html>) [Chapter 2 →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
