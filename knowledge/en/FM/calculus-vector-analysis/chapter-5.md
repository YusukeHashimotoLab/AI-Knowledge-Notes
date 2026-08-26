---
title: "Chapter 5: Line Integrals, Surface Integrals, and Integral Theorems"
chapter_title: "Chapter 5: Line Integrals, Surface Integrals, and Integral Theorems"
subtitle: Line Integrals, Surface Integrals, and Integral Theorems
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/BEElvcgY5Uk?start=3408"
    title="Calculus & Vector Analysis Ch.5: Line Integrals, Surface Integrals, and Integral Theorems"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/calculus-vector-analysis/chapter-5.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../index.html>) > [FM Dojo](<../index.html>) > [Introduction to Calculus and Vector Analysis](<index.html>) > Chapter 5 

## 5.1 Fundamentals of Line Integrals

**📐 Definition: Line Integral**  
Line integral of scalar field f over curve C: $$\int_C f \, ds = \int_a^b f(\mathbf{r}(t)) \|\mathbf{r}'(t)\| \, dt$$ Line integral of vector field F (work): $$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt$$ 

### 💻 Code Example 1: Line Integral of Scalar Field

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Parametric representation of curve: r(t) = (cos t, sin t), 0 ≤ t ≤ π
def curve(t):
    x = np.cos(t)
    y = np.sin(t)
    return x, y

# Scalar field: f(x,y) = x² + y²
def scalar_field(x, y):
    return x**2 + y**2

# Integrand for line integral
def integrand(t):
    x, y = curve(t)
    f_val = scalar_field(x, y)
    # Magnitude of velocity vector ||r'(t)||
    dx_dt = -np.sin(t)
    dy_dt = np.cos(t)
    speed = np.sqrt(dx_dt**2 + dy_dt**2)
    return f_val * speed

# Calculate line integral
result, error = integrate.quad(integrand, 0, np.pi)
print(f"Line integral of scalar field:")
print(f"∫_C f ds = {result:.6f} (estimated error: {error:.2e})")

# Visualization
t = np.linspace(0, np.pi, 100)
x, y = curve(t)

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'r-', linewidth=3, label='Curve C')
plt.arrow(0, 1, 0.1, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
plt.scatter([x[0]], [y[0]], color='green', s=150, marker='o', label='Start point', zorder=5)
plt.scatter([x[-1]], [y[-1]], color='blue', s=150, marker='s', label='End point', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Path of Line Integration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

### 💻 Code Example 2: Line Integral of Vector Field (Work)

```python
# Vector field: F(x,y) = (y, -x)
def vector_field(x, y):
    return y, -x

# Line integral ∫_C F·dr
def integrand_vector(t):
    x, y = curve(t)
    Fx, Fy = vector_field(x, y)
    dx_dt = -np.sin(t)
    dy_dt = np.cos(t)
    return Fx * dx_dt + Fy * dy_dt

result_work, error = integrate.quad(integrand_vector, 0, np.pi)
print(f"\nLine integral of vector field (work):")
print(f"∫_C F·dr = {result_work:.6f}")

# Visualization of vector field
x_grid = np.linspace(-1.5, 1.5, 15)
y_grid = np.linspace(-1.5, 1.5, 15)
X, Y = np.meshgrid(x_grid, y_grid)
Fx, Fy = vector_field(X, Y)

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, Fx, Fy, alpha=0.6)
plt.plot(x, y, 'r-', linewidth=3, label='Path C')
plt.scatter([x[0]], [y[0]], color='green', s=150, marker='o', zorder=5)
plt.scatter([x[-1]], [y[-1]], color='blue', s=150, marker='s', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field F=(y,-x) and Integration Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## 5.2 Green's Theorem

**📐 Theorem: Green's Theorem**  
For boundary C (counterclockwise) of region D: $$\oint_C (P \, dx + Q \, dy) = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$ Line integrals can be converted to double integrals. 

### 💻 Code Example 3: Verification of Green's Theorem

```python
# Vector field F = (P, Q) = (x, y)
# Region D: unit disk x² + y² ≤ 1

# Left side: line integral ∮_C (x dx + y dy)
def curve_circle(t):
    return np.cos(t), np.sin(t)

def line_integral(t):
    x, y = curve_circle(t)
    dx_dt, dy_dt = -np.sin(t), np.cos(t)
    P, Q = x, y
    return P * dx_dt + Q * dy_dt

left_side, _ = integrate.quad(line_integral, 0, 2*np.pi)

# Right side: double integral ∬_D (∂Q/∂x - ∂P/∂y) dA, computed numerically
# in polar coordinates (dA = r dr dθ) rather than hardcoded.
def double_integral_over_disk(curl_z):
    """∬_D curl_z(x, y) dA over the unit disk, via polar coordinates."""
    def integrand(theta, r):
        return curl_z(r*np.cos(theta), r*np.sin(theta)) * r
    value, _ = integrate.dblquad(integrand, 0, 1, 0, 2*np.pi)
    return value

# F = (P, Q) = (x, y): ∂Q/∂x = ∂y/∂x = 0 and ∂P/∂y = ∂x/∂y = 0,
# so the integrand ∂Q/∂x - ∂P/∂y is identically 0.
right_side = double_integral_over_disk(lambda x, y: 0.0)

print("Verification of Green's theorem:")
print(f"F = (x, y), D = unit disk")
print(f"Left side (line integral): ∮_C F·dr = {left_side:.6f}")
print(f"Right side (double integral): ∬_D (∂Q/∂x - ∂P/∂y) dA = {right_side:.6f}")
print(f"Difference: {abs(left_side - right_side):.2e}")

# More interesting example: F = (-y, x)
def line_integral2(t):
    x, y = curve_circle(t)
    dx_dt, dy_dt = -np.sin(t), np.cos(t)
    P, Q = -y, x
    return P * dx_dt + Q * dy_dt

left_side2, _ = integrate.quad(line_integral2, 0, 2*np.pi)

# Right side: with P = -y and Q = x, ∂Q/∂x = 1 and ∂P/∂y = -1,
# so ∂Q/∂x - ∂P/∂y = 1 - (-1) = 2, and ∬_D 2 dA = 2 × (area of disk) = 2π.
right_side2 = double_integral_over_disk(lambda x, y: 2.0)

print(f"\nF = (-y, x), D = unit disk")
print(f"Left side (line integral): {left_side2:.6f}")
print(f"Right side (double integral): {right_side2:.6f}")
print(f"Error: {abs(left_side2 - right_side2):.2e}")
```

**Output:**

```
Verification of Green's theorem:
F = (x, y), D = unit disk
Left side (line integral): ∮_C F·dr = 0.000000
Right side (double integral): ∬_D (∂Q/∂x - ∂P/∂y) dA = 0.000000
Difference: 0.00e+00

F = (-y, x), D = unit disk
Left side (line integral): 6.283185
Right side (double integral): 6.283185
Error: 8.88e-16
```

Both sides are now genuinely computed — the line integral by `quad` along the circle, the area integral by `dblquad` over the disk — so the agreement is evidence, not an assertion.

## 5.3 Fundamentals of Surface Integrals

**📐 Definition: Surface Integral**  
Surface integral of scalar field f over surface S: $$\iint_S f \, dS$$ Surface integral of vector field F (flux): $$\iint_S \mathbf{F} \cdot \mathbf{n} \, dS$$ where n is the unit normal vector of the surface. 

### 💻 Code Example 4: Surface Integral on Sphere

```python
from scipy import integrate

# Parametric representation of sphere: r(θ,φ) = (sin θ cos φ, sin θ sin φ, cos θ)
# 0 ≤ θ ≤ π, 0 ≤ φ ≤ 2π

def sphere_parametrization(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

# Scalar field: f(x,y,z) = z
def scalar_field_3d(x, y, z):
    return z

# Integrand for surface integral (including Jacobian)
def surface_integrand(theta, phi):
    x, y, z = sphere_parametrization(theta, phi)
    f_val = scalar_field_3d(x, y, z)
    # Jacobian of sphere: r² sin θ (radius r=1)
    jacobian = np.sin(theta)
    return f_val * jacobian

# Calculate surface integral
result, error = integrate.dblquad(surface_integrand,
                                   0, 2*np.pi,    # φ range
                                   0, np.pi)      # θ range

print("Surface integral on sphere:")
print(f"∬_S z dS = {result:.6f} (analytical solution: 0 by symmetry)")
print(f"Estimated error: {error:.2e}")
```

## 5.4 Gauss's Divergence Theorem

**📐 Theorem: Gauss's Divergence Theorem**  
For closed surface S enclosing region V, with n the **outward** unit normal: $$\oiint_S \mathbf{F} \cdot \mathbf{n} \, dS = \iiint_V (\nabla \cdot \mathbf{F}) \, dV$$ Surface integrals can be converted to volume integrals. The outward orientation is part of the theorem: choosing the inward normal flips the sign of the left-hand side. 

### 💻 Code Example 5: Verification of Gauss's Divergence Theorem

```python
# Vector field F = (x, y, z)
# Region V: unit ball x² + y² + z² ≤ 1, boundary S: unit sphere

# Right side: volume integral ∭_V div F dV, evaluated in closed form.
# div F = ∂x/∂x + ∂y/∂y + ∂z/∂z = 3
# ∭_V 3 dV = 3 × (volume of the ball) = 3 × (4π/3) = 4π
right_side_gauss = 3 * (4/3) * np.pi

# Left side: surface integral ∬_S F·n dS, computed NUMERICALLY.
# Parametrize S by r(θ,φ) = (sinθ cosφ, sinθ sinφ, cosθ) with dS = sinθ dθ dφ.
# The OUTWARD unit normal on the unit sphere is n = r itself, so F·n = x² + y² + z².
def flux_integrand(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    F_dot_n = x**2 + y**2 + z**2     # = 1 on the unit sphere
    return F_dot_n * np.sin(theta)   # times the surface element

left_side_gauss, quad_err = integrate.dblquad(flux_integrand,
                                               0, 2*np.pi,   # φ range (outer)
                                               0, np.pi)     # θ range (inner)

print("Verification of Gauss's divergence theorem:")
print(f"F = (x, y, z), V = unit ball")
print(f"Left side (surface integral, dblquad):  ∬_S F·n dS = {left_side_gauss:.6f}")
print(f"Right side (volume integral, analytic): ∭_V div F dV = {right_side_gauss:.6f}")
print(f"Difference: {abs(left_side_gauss - right_side_gauss):.2e} "
      f"(estimated quadrature error {quad_err:.2e})")
```

**Output:**

```
Verification of Gauss's divergence theorem:
F = (x, y, z), V = unit ball
Left side (surface integral, dblquad):  ∬_S F·n dS = 12.566371
Right side (volume integral, analytic): ∭_V div F dV = 12.566371
Difference: 0.00e+00 (estimated quadrature error 1.40e-13)
```

12.566371 = 4π, so the two sides agree to machine precision. Note that this is a real test: the left side was integrated over the surface, not assigned the answer.

## 5.5 Stokes' Theorem

**📐 Theorem: Stokes' Theorem**  
For boundary C of surface S, with C and the unit normal n oriented by the **right-hand rule** (curl the fingers of your right hand along C and the thumb points along n): $$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS$$ Line integrals can be converted to surface integrals. This is the 3D version of Green's theorem; reversing either the traversal of C or the choice of n flips the sign. 

### 💻 Code Example 6: Application of Stokes' Theorem

```python
# Vector field F = (-y, x, 0)
# Surface S: hemisphere z = √(1-x²-y²), boundary C: unit circle x²+y²=1 (on z=0 plane)

# Left side: line integral ∮_C F·dr
def stokes_line_integral(t):
    x, y, z = np.cos(t), np.sin(t), 0
    dx_dt, dy_dt, dz_dt = -np.sin(t), np.cos(t), 0
    Fx, Fy, Fz = -y, x, 0
    return Fx*dx_dt + Fy*dy_dt + Fz*dz_dt

left_stokes, _ = integrate.quad(stokes_line_integral, 0, 2*np.pi)

# Right side: curl F = (0, 0, 2)
# ∬_S (0,0,2)·n dS = 2 × (projected area of hemisphere) = 2π

right_stokes = 2 * np.pi

print("\nVerification of Stokes' theorem:")
print(f"F = (-y, x, 0), S = hemisphere")
print(f"Left side (line integral): ∮_C F·dr = {left_stokes:.6f}")
print(f"Right side (surface integral): ∬_S (curl F)·n dS = {right_stokes:.6f}")
print(f"Error: {abs(left_stokes - right_stokes):.2e}")
```

## 5.6 Application to Materials Science: Diffusion Flux and Conservation Laws

**🔬 Application Example:** In atomic diffusion in materials, conservation law is derived from Gauss's divergence theorem: $$\frac{\partial C}{\partial t} = -\nabla \cdot \mathbf{J} + S$$ where C is concentration, J is diffusion flux, and S is generation term. 

### 💻 Code Example 7: Numerical Simulation of Diffusion Equation

```python
# 1D diffusion equation: ∂C/∂t = D ∂²C/∂x²
# Initial condition: Gaussian distribution
# Boundary condition: zero concentration at both ends

def diffusion_1d(C0, D, dx, dt, n_steps, record_every=10):
    """Numerical solution of 1D diffusion equation (explicit Euler method).

    Returns (times, snapshots) so every stored profile carries the simulated
    time it actually belongs to - never label a curve with an index times dt.
    """
    C = C0.copy()
    times = [0.0]
    snapshots = [C.copy()]

    for step in range(1, n_steps + 1):
        C_new = C.copy()
        for i in range(1, len(C)-1):
            # Laplacian: (C[i+1] - 2C[i] + C[i-1]) / dx²
            laplacian = (C[i+1] - 2*C[i] + C[i-1]) / dx**2
            C_new[i] = C[i] + D * dt * laplacian

        C = C_new
        # After `step` steps the simulated time is exactly step*dt.
        # Always record the last state so no steps are computed and thrown away.
        if step % record_every == 0 or step == n_steps:
            times.append(step * dt)
            snapshots.append(C.copy())

    return times, snapshots

# Parameter settings
nx = 100
L = 10.0
x = np.linspace(0, L, nx)
dx = x[1] - x[0]

# Initial concentration distribution (Gaussian distribution)
C0 = np.exp(-(x - L/2)**2 / 0.5)
C0[0] = C0[-1] = 0  # Boundary condition

# Diffusion coefficient and time step
D = 0.1
dt = 0.01
n_steps = 100

# Explicit Euler is only stable while D·dt/dx² ≤ 0.5 - check before running
stability = D * dt / dx**2
print(f"Stability number D·dt/dx² = {stability:.3f} (must be ≤ 0.5 for explicit Euler)")

# Execute simulation
times, C_history = diffusion_1d(C0, D, dx, dt, n_steps)

# Visualization - the label comes from the recorded time, not from the loop index
plt.figure(figsize=(12, 6))
for t, C in zip(times[::2], C_history[::2]):
    plt.plot(x, C, label=f't = {t:.2f}', alpha=0.7)

plt.xlabel('Position x')
plt.ylabel('Concentration C')
plt.title('Time Evolution of 1D Diffusion Equation')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

print("\nDiffusion simulation:")
print(f"Initial peak concentration: {C0.max():.4f}")
print(f"Final peak concentration (t = {times[-1]:.2f}): {C_history[-1].max():.4f}")
print(f"Concentration decrease due to diffusion: {(1 - C_history[-1].max()/C0.max())*100:.1f}%")
```

**Output:**

```
Stability number D·dt/dx² = 0.098 (must be ≤ 0.5 for explicit Euler)

Diffusion simulation:
Initial peak concentration: 0.9949
Final peak concentration (t = 1.00): 0.7436
Concentration decrease due to diffusion: 25.3%
```

**📝 Cross-check:** the initial profile exp(−(x − L/2)²/0.5) is itself a diffused Gaussian with 4Dt₀ = 0.5, i.e. t₀ = 1.25. The analytic peak decay is √(t₀/(t₀+t)), which predicts a 25.5% decrease at t = 1.00 — within 0.2 points of the simulated 25.3%, the residual being the explicit-Euler discretisation error.

## Summary

  * Line integrals calculate accumulated values along a path for scalar fields and work for vector fields
  * Surface integrals calculate accumulated values on surfaces and flux for vector fields
  * Green's theorem converts line integrals to double integrals (2D)
  * Gauss's divergence theorem converts surface integrals to volume integrals (3D)
  * Stokes' theorem converts line integrals to surface integrals (3D version of Green's theorem)
  * In materials science, they are applied to calculations of diffusion flux, conservation laws, etc.

[← Chapter 4: Vector Fields](<chapter-4.html>) [Back to Series Top](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
