---
title: "Chapter 3: Multivariable Calculus"
chapter_title: "Chapter 3: Multivariable Calculus"
subtitle: Multivariable Calculus
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/BEElvcgY5Uk?start=1684"
    title="Calculus & Vector Analysis Ch.3: Multivariable Calculus"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/calculus-vector-analysis/chapter-3.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../index.html>) > [FM Dojo](<../index.html>) > [Introduction to Calculus and Vector Analysis](<index.html>) > Chapter 3 

## 3.1 Partial Derivatives and Total Differentials

**📐 Definition: Partial Derivative**  
The partial derivative of multivariable function f(x, y) with respect to x is: $$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$ Differentiate with respect to one variable while keeping other variables fixed. 

### 💻 Code Example 1: Numerical Calculation of Partial Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """Two-variable function: f(x,y) = x² + xy + y²"""
    return x**2 + x*y + y**2

def partial_x(f, x, y, h=1e-5):
    """Numerical calculation of ∂f/∂x"""
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def partial_y(f, x, y, h=1e-5):
    """Numerical calculation of ∂f/∂y"""
    return (f(x, y+h) - f(x, y-h)) / (2*h)

# Partial derivatives at point (1, 2)
x0, y0 = 1, 2
df_dx = partial_x(f, x0, y0)
df_dy = partial_y(f, x0, y0)

print(f"Partial derivatives at point ({x0}, {y0}):")
print(f"∂f/∂x = {df_dx:.6f} (analytical solution: {2*x0 + y0})")
print(f"∂f/∂y = {df_dy:.6f} (analytical solution: {x0 + 2*y0})")

# 3D visualization
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
ax.scatter([x0], [y0], [f(x0, y0)], color='red', s=100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('f(x,y) = x² + xy + y²')
plt.show()
```

### 💻 Code Example 2: Partial Derivatives Using SymPy

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# Partial derivatives of various multivariable functions
f1 = x**2 + y**2
f2 = x*sp.exp(y)
f3 = sp.sin(x*y)
f4 = x**2 * y + y**2 * z + z**2 * x

functions = [f1, f2, f3, f4]

print("Examples of partial derivatives:")
for f in functions:
    print(f"\nf = {f}")
    # sorted() keeps the output order reproducible;
    # free_symbols is a set, so its iteration order is not deterministic
    for var in sorted(f.free_symbols, key=str):
        print(f"  ∂f/∂{var} = {sp.diff(f, var)}")
```

### 📐 Total Differential and Chain Rule

A partial derivative measures the change of f along one axis only. When **every** variable changes at once, the first-order change of f is the **total differential**: $$df = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy$$ This is the linear approximation of Δf = f(x+Δx, y+Δy) − f(x, y); the neglected part is of second order in (Δx, Δy).

If the variables themselves depend on a parameter t, i.e. x = x(t) and y = y(t), dividing the total differential by dt gives the **chain rule** for multivariable functions: $$\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$ This is the rule that lets you differentiate along a path through a scalar field — exactly what line integrals (Chapter 5) and backpropagation in neural networks both rely on.

```python
import numpy as np

def f(x, y):
    """f(x, y) = x² + xy + y²"""
    return x**2 + x*y + y**2

# --- Total differential: df = (∂f/∂x)dx + (∂f/∂y)dy ---
x0, y0 = 1.0, 2.0
fx, fy = 2*x0 + y0, x0 + 2*y0       # partial derivatives at (1, 2): 4.0 and 5.0
dx, dy = 0.01, -0.02

df_linear = fx*dx + fy*dy                      # first-order (linear) estimate
df_actual = f(x0 + dx, y0 + dy) - f(x0, y0)    # true change

print(f"df (total differential) = {df_linear:.6f}")
print(f"Δf (actual change)      = {df_actual:.6f}")
print(f"difference (2nd order)  = {abs(df_actual - df_linear):.2e}")

# --- Chain rule: x = cos(t), y = sin(t) ---
def df_dt_chain(t):
    x, y = np.cos(t), np.sin(t)
    return (2*x + y)*(-np.sin(t)) + (x + 2*y)*np.cos(t)

t0, h = 0.7, 1e-6
df_dt_numeric = (f(np.cos(t0+h), np.sin(t0+h)) - f(np.cos(t0-h), np.sin(t0-h))) / (2*h)

print(f"\ndf/dt by chain rule     = {df_dt_chain(t0):.6f}")
print(f"df/dt by central diff   = {df_dt_numeric:.6f}")
```

**Output:**

```
df (total differential) = -0.060000
Δf (actual change)      = -0.059700
difference (2nd order)  = 3.00e-04

df/dt by chain rule     = 0.169967
df/dt by central diff   = 0.169967
```

## 3.2 Gradient and Gradient Descent

**📐 Definition: Gradient**  
The gradient of scalar field f(x, y, z) is a vector with partial derivatives as components: $$\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right)$$ The gradient vector points in the direction of steepest increase of the function. 

### 💻 Code Example 3: Implementation of Gradient Descent Method

```python
def gradient(f, x, y, h=1e-5):
    """Numerical calculation of gradient vector"""
    grad_x = (f(x+h, y) - f(x-h, y)) / (2*h)
    grad_y = (f(x, y+h) - f(x, y-h)) / (2*h)
    return np.array([grad_x, grad_y])

def gradient_descent(f, x0, y0, learning_rate=0.1, n_iter=50):
    """Minimum search using gradient descent"""
    path = [(x0, y0)]
    x, y = x0, y0

    for i in range(n_iter):
        grad = gradient(f, x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path.append((x, y))

    return np.array(path)

# Search for minimum of f(x,y) = x² + y²
f_simple = lambda x, y: x**2 + y**2
path = gradient_descent(f_simple, x0=3, y0=2, learning_rate=0.2, n_iter=20)

print("Minimum search using gradient descent:")
print(f"Starting point: ({path[0,0]:.2f}, {path[0,1]:.2f})")
print(f"Ending point: ({path[-1,0]:.6f}, {path[-1,1]:.6f})")
print(f"Minimum value: f = {f_simple(path[-1,0], path[-1,1]):.6f}")

# Visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f_simple(X, Y)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(label='f(x,y)')
plt.plot(path[:,0], path[:,1], 'ro-', linewidth=2, markersize=6)
plt.scatter([path[0,0]], [path[0,1]], color='green', s=150, marker='*', label='Starting point')
plt.scatter([path[-1,0]], [path[-1,1]], color='red', s=150, marker='*', label='Ending point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Using Gradient Descent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## 3.3 Lagrange Multiplier Method

**📐 Theorem: Lagrange Multiplier Method**  
The problem of optimizing f(x, y) under constraint g(x, y) = 0 is reduced to finding stationary points of the Lagrange function L(x, y, λ) = f(x, y) - λg(x, y). 

### 💻 Code Example 4: Constrained Optimization Problem

```python
from scipy.optimize import minimize

# Problem: Minimize x² + y², constraint x + y = 1
def objective(X):
    x, y = X
    return x**2 + y**2

def constraint(X):
    x, y = X
    return x + y - 1  # x + y = 1

# Specify constraint in dictionary format
cons = {'type': 'eq', 'fun': constraint}

# Execute optimization
x0 = [0, 0]  # Initial value
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print("Results of constrained optimization:")
print(f"Optimal solution: x = {result.x[0]:.6f}, y = {result.x[1]:.6f}")
print(f"Minimum value: f = {result.fun:.6f}")
print(f"Constraint verification: x + y = {result.x[0] + result.x[1]:.6f}")

# Comparison with analytical solution (analytical solution for this problem is x = y = 0.5)
print(f"\nAnalytical solution: x = y = 0.5, f = 0.5")
print(f"Error: {abs(result.x[0] - 0.5):.2e}")
```

## 3.4 Double Integrals and Polar Coordinates

### 💻 Code Example 5: Numerical Calculation of Double Integrals

```python
from scipy import integrate

# ∫∫_D xy dxdy, D: 0 ≤ x ≤ 1, 0 ≤ y ≤ 2
def integrand(y, x):
    """Integrand f(x,y) = xy"""
    return x * y

# dblquad: double integral
result, error = integrate.dblquad(integrand,
                                   0, 1,      # x range
                                   0, 2)      # y range

print("Double integral calculation:")
print(f"∫₀¹ ∫₀² xy dy dx = {result:.6f}")
print(f"Analytical solution: [x²/2]₀¹ · [y²/2]₀² = 0.5 · 2 = 1.0")
print(f"Estimated error: {error:.2e}")
```

### 💻 Code Example 6: Double Integral Using Polar Coordinate Transformation

```python
# ∫∫ (x² + y²) dxdy over disk x² + y² ≤ 1
# Polar coordinate transformation: x = r cos θ, y = r sin θ, dxdy = r dr dθ

def integrand_polar(theta, r):
    """Integrand in polar coordinates: r² · r = r³"""
    return r**3

# Integration range: 0 ≤ r ≤ 1, 0 ≤ θ ≤ 2π
result, error = integrate.dblquad(integrand_polar,
                                   0, 1,           # r range
                                   0, 2*np.pi)     # θ range

print("\nDouble integral using polar coordinate transformation:")
print(f"∫∫_D (x² + y²) dxdy = {result:.6f}")
print(f"Analytical solution: ∫₀^2π dθ ∫₀¹ r³ dr = 2π · [r⁴/4]₀¹ = π/2 ≈ {np.pi/2:.6f}")
print(f"Error: {abs(result - np.pi/2):.2e}")
```

**Output:**

```
Double integral using polar coordinate transformation:
∫∫_D (x² + y²) dxdy = 1.570796
Analytical solution: ∫₀^2π dθ ∫₀¹ r³ dr = 2π · [r⁴/4]₀¹ = π/2 ≈ 1.570796
Error: 2.22e-16
```

### 💻 Code Example 7: Application to Materials Science (Dose Distribution Integration)

```python
# Impurity dose distribution on a circular wafer.
# Note the units: the implanted dose is an AREAL density [atoms/cm²],
# so integrating it over the wafer area [cm²] gives a pure atom count.
# (A volumetric concentration [atoms/cm³] integrated over an area would
#  leave units of atoms/cm, not atoms.)
def dose(r, theta):
    """
    Areal dose distribution D(r,θ) [atoms/cm²]
    Dose decreases with distance from center
    """
    r_max = 5.0  # Wafer radius [cm]
    D0 = 1e15    # Center dose [atoms/cm²]
    return D0 * np.exp(-r**2 / r_max**2)

# Calculate total impurity atoms in entire wafer
R = 5.0  # Wafer radius [cm]

def integrand_dose(theta, r):
    return dose(r, theta) * r  # Multiply by Jacobian r (dA = r dr dθ)

total_atoms, error = integrate.dblquad(integrand_dose,
                                        0, R,
                                        0, 2*np.pi)

print("\nApplication to materials science:")
print(f"Wafer radius: {R} cm")
print(f"Center dose: {1e15:.2e} atoms/cm²")
print(f"Total impurity atoms: {total_atoms:.4e} atoms")
print(f"Average dose: {total_atoms / (np.pi * R**2):.4e} atoms/cm²")

# Visualization of concentration distribution
r_plot = np.linspace(0, R, 100)
theta_plot = np.linspace(0, 2*np.pi, 100)
R_grid, Theta_grid = np.meshgrid(r_plot, theta_plot)
C_grid = dose(R_grid, Theta_grid)

# Convert polar to Cartesian coordinates
X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)

plt.figure(figsize=(10, 8))
plt.contourf(X_grid, Y_grid, C_grid, levels=20, cmap='hot')
plt.colorbar(label='Dose (atoms/cm²)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('Impurity Dose Distribution on Wafer')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

**Output:**

```
Application to materials science:
Wafer radius: 5.0 cm
Center dose: 1.00e+15 atoms/cm²
Total impurity atoms: 4.9647e+16 atoms
Average dose: 6.3212e+14 atoms/cm²
```

The closed form confirms the number: 2πD₀·(r_max²/2)·(1 − e⁻¹) = 4.9647e+16 atoms, and the average dose is D₀(1 − e⁻¹) = 6.3212e+14 atoms/cm².

## Summary

  * Partial derivatives represent the rate of change of multivariable functions with respect to each variable
  * The gradient vector indicates the direction of steepest increase and is utilized in optimization
  * Constrained optimization problems can be solved using Lagrange multipliers
  * Multiple integrals calculate accumulated quantities of multivariable functions, often simplified using polar coordinate transformations
  * In materials science, they are applied to integration calculations of concentration distributions, temperature distributions, etc.

[← Chapter 2: Fundamentals of Integration](<chapter-2.html>) [Chapter 4: Vector Fields →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
