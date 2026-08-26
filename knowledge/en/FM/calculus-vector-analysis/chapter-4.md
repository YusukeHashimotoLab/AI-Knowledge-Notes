---
title: "Chapter 4: Vector Fields and Differential Operators"
chapter_title: "Chapter 4: Vector Fields and Differential Operators"
subtitle: Vector Fields and Differential Operators
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/BEElvcgY5Uk?start=2579"
    title="Calculus & Vector Analysis Ch.4: Vector Fields and Differential Operators"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/calculus-vector-analysis/chapter-4.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../index.html>) > [FM Dojo](<../index.html>) > [Introduction to Calculus and Vector Analysis](<index.html>) > Chapter 4 

## 4.1 Definition and Visualization of Vector Fields

**📐 Definition: Vector Field**  
A function that associates one vector to each point in space is called a vector field: $$\mathbf{F}(\mathbf{r}) = (F_x(x,y,z), F_y(x,y,z), F_z(x,y,z))$$ Examples: velocity field of fluid, electric field, magnetic field, etc. 

### 💻 Code Example 1: Visualization of 2D Vector Field

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D vector field definition: F(x,y) = (-y, x) (rotational field)
def vector_field(x, y):
    """Rotating vector field"""
    Fx = -y
    Fy = x
    return Fx, Fy

# Create grid
x = np.linspace(-3, 3, 15)
y = np.linspace(-3, 3, 15)
X, Y = np.meshgrid(x, y)
Fx, Fy = vector_field(X, Y)

# Visualize vector field with quiver plot
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, Fx, Fy, np.sqrt(Fx**2 + Fy**2), cmap='viridis')
plt.colorbar(label='Vector magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rotational Vector Field F = (-y, x)')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

# Draw streamlines
x_fine = np.linspace(-3, 3, 100)
y_fine = np.linspace(-3, 3, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Fx_fine, Fy_fine = vector_field(X_fine, Y_fine)

plt.figure(figsize=(10, 8))
plt.streamplot(X_fine, Y_fine, Fx_fine, Fy_fine, density=1.5, color='blue', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamlines of Vector Field')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

## 4.2 Gradient (gradient, grad)

**📐 Definition: Gradient**  
The gradient of scalar field φ is: $$\nabla \phi = \text{grad}\,\phi = \left(\frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z}\right)$$ The gradient vector points in the direction where φ increases most rapidly. 

### 💻 Code Example 2: Calculation and Visualization of Gradient Vector Field

```python
def scalar_field(x, y):
    """Scalar field: φ(x,y) = x² + y²"""
    return x**2 + y**2

def gradient_field(x, y):
    """Gradient: ∇φ = (2x, 2y)"""
    grad_x = 2*x
    grad_y = 2*y
    return grad_x, grad_y

# Visualization
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
phi = scalar_field(X, Y)
grad_x, grad_y = gradient_field(X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Contour lines of scalar field
contour = ax1.contourf(X, Y, phi, levels=20, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='φ(x,y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Scalar Field φ = x² + y²')
ax1.axis('equal')

# Right plot: Gradient vector field
ax2.contour(X, Y, phi, levels=10, colors='gray', alpha=0.3)
ax2.quiver(X, Y, grad_x, grad_y, color='red')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Gradient Vector Field ∇φ = (2x, 2y)')
ax2.axis('equal')
plt.tight_layout()
plt.show()
```

## 4.3 Divergence (divergence, div)

**📐 Definition: Divergence**  
The divergence of vector field F is: $$\text{div}\,\mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$$ Divergence represents the strength of the "source" of the vector field at that point. 

### 💻 Code Example 3: Calculation of Divergence

```python
def divergence_numerical(Fx, Fy, x, y, h=1e-5):
    """Numerical calculation of divergence: div F = ∂Fx/∂x + ∂Fy/∂y"""
    dFx_dx = (Fx(x+h, y) - Fx(x-h, y)) / (2*h)
    dFy_dy = (Fy(x, y+h) - Fy(x, y-h)) / (2*h)
    return dFx_dx + dFy_dy

# Example 1: Vector field with positive divergence (diverging field)
def Fx_diverging(x, y):
    return x

def Fy_diverging(x, y):
    return y

# Example 2: Vector field with zero divergence (rotational field)
def Fx_rotating(x, y):
    return -y

def Fy_rotating(x, y):
    return x

# Calculate divergence
x0, y0 = 1, 1
div_diverging = divergence_numerical(Fx_diverging, Fy_diverging, x0, y0)
div_rotating = divergence_numerical(Fx_rotating, Fy_rotating, x0, y0)

print(f"Divergence of diverging field F=(x,y): div F = {div_diverging:.6f} (analytical solution: 2)")
print(f"Divergence of rotational field F=(-y,x): div F = {div_rotating:.6f} (analytical solution: 0)")
```

## 4.4 Curl (rotation, curl)

**📐 Definition: Curl**  
The curl of 3D vector field F is: $$\text{rot}\,\mathbf{F} = \nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\\ F_x & F_y & F_z \end{vmatrix}$$ 

### 💻 Code Example 4: Curl in 2D (Scalar)

```python
# In 2D, curl is a scalar value: rot F = ∂Fy/∂x - ∂Fx/∂y

def curl_2d(Fx, Fy, x, y, h=1e-5):
    """Numerical calculation of curl in 2D"""
    dFy_dx = (Fy(x+h, y) - Fy(x-h, y)) / (2*h)
    dFx_dy = (Fx(x, y+h) - Fx(x, y-h)) / (2*h)
    return dFy_dx - dFx_dy

# Calculate curl of rotational field
curl_rotating = curl_2d(Fx_rotating, Fy_rotating, x0, y0)
print(f"\nCurl of rotational field F=(-y,x): rot F = {curl_rotating:.6f} (analytical solution: 2)")

# Calculate curl of diverging field
curl_diverging = curl_2d(Fx_diverging, Fy_diverging, x0, y0)
print(f"Curl of diverging field F=(x,y): rot F = {curl_diverging:.6f} (analytical solution: 0)")
```

## 4.5 Laplacian (Laplacian, Δ)

**📐 Definition: Laplacian**  
The Laplacian of scalar field φ is: $$\Delta \phi = \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}$$ Appears in many physical laws such as heat equation and wave equation. 

### 💻 Code Example 5: Calculation and Application of Laplacian

```python
def laplacian_2d(phi, x, y, h=1e-4):
    """Numerical calculation of 2D Laplacian"""
    lap = (phi(x+h, y) + phi(x-h, y) + phi(x, y+h) + phi(x, y-h) - 4*phi(x, y)) / h**2
    return lap

# Test function: φ(x,y) = x² + y²
phi = lambda x, y: x**2 + y**2

lap = laplacian_2d(phi, 1, 1)
print(f"\nLaplacian of φ = x² + y²: Δφ = {lap:.6f} (analytical solution: 4)")

# Solution of Laplace equation Δφ = 0 (harmonic function)
phi_harmonic = lambda x, y: x**2 - y**2
lap_harmonic = laplacian_2d(phi_harmonic, 1, 1)
print(f"Laplacian of φ = x² - y²: Δφ = {lap_harmonic:.6f} (analytical solution: 0)")
```

## 4.6 Conservative Fields and Potential Functions

**📐 Theorem: Condition for Conservative Field**  
On a **simply connected** domain (a region with no holes, such as all of ℝ³ or a ball), the condition for vector field F to be conservative (scalar potential φ exists) is: $$\text{rot}\,\mathbf{F} = \mathbf{0}$$ In this case, F can be expressed as F = grad φ. The simple-connectedness hypothesis is essential: rot F = 0 is always necessary, but on a domain with a hole it is not sufficient — F = (−y/(x²+y²), x/(x²+y²)) on ℝ²∖{0} has rot F = 0 everywhere on its domain, yet ∮F·dr = 2π around any loop enclosing the origin, so no single-valued potential exists. 

### 💻 Code Example 6: Determination of Conservative Field and Calculation of Potential Function

```python
import sympy as sp

x, y = sp.symbols('x y')

# Vector field F = (2xy, x² + 2y)
Fx_sym = 2*x*y
Fy_sym = x**2 + 2*y

# Calculate curl
curl_z = sp.diff(Fy_sym, x) - sp.diff(Fx_sym, y)
print("Determination of conservative field for vector field F = (2xy, x² + 2y):")
print(f"rot F = ∂Fy/∂x - ∂Fx/∂y = {curl_z}")

if curl_z == 0:
    print("→ Since rot F = 0, this is a conservative field\n")

    # Calculate potential function
    # Find φ such that ∂φ/∂x = Fx, ∂φ/∂y = Fy
    phi = sp.integrate(Fx_sym, x)  # Integrate with respect to x
    print(f"∫ Fx dx = {phi} + g(y)")

    # Determine function g(y) of y
    dPhi_dy = sp.diff(phi, y)
    g_prime = Fy_sym - dPhi_dy
    g = sp.integrate(g_prime, y)
    phi_final = phi + g

    print(f"Potential function: φ = {phi_final}")

    # Verification: the residuals must simplify to exactly 0.
    # Never print a "✓" that the code did not actually check.
    residual_x = sp.simplify(sp.diff(phi_final, x) - Fx_sym)
    residual_y = sp.simplify(sp.diff(phi_final, y) - Fy_sym)
    print(f"\nVerification (the residuals must simplify to exactly 0):")
    print(f"∂φ/∂x - Fx = {residual_x}  → {'OK' if residual_x == 0 else 'MISMATCH'}")
    print(f"∂φ/∂y - Fy = {residual_y}  → {'OK' if residual_y == 0 else 'MISMATCH'}")
```

**Output:**

```
Determination of conservative field for vector field F = (2xy, x² + 2y):
rot F = ∂Fy/∂x - ∂Fx/∂y = 0
→ Since rot F = 0, this is a conservative field

∫ Fx dx = x**2*y + g(y)
Potential function: φ = x**2*y + y**2

Verification (the residuals must simplify to exactly 0):
∂φ/∂x - Fx = 0  → OK
∂φ/∂y - Fy = 0  → OK
```

### 💻 Code Example 7: Application to Materials Science (Diffusion Flux)

```python
# Fick's first law: J = -D ∇C (diffusion flux)
# Diffusion flux occurs due to concentration gradient

def concentration(x, y):
    """Concentration distribution C(x,y)"""
    return np.exp(-(x**2 + y**2))

# Calculate gradient (concentration gradient).
# 21 points (an ODD count) puts a grid node exactly at the origin,
# which is where we want to read off the divergence below.
x = np.linspace(-2, 2, 21)
y = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(x, y)

# Calculate gradient by numerical differentiation
h = x[1] - x[0]
C = concentration(X, Y)
dC_dx = np.gradient(C, h, axis=1)
dC_dy = np.gradient(C, h, axis=0)

# Diffusion flux: J = -D ∇C
D = 1.0  # Diffusion coefficient
Jx = -D * dC_dx
Jy = -D * dC_dy

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Concentration distribution
contour = ax1.contourf(X, Y, C, levels=15, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Concentration C')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Concentration Distribution C(x,y) = exp(-(x²+y²))')
ax1.axis('equal')

# Right plot: Diffusion flux vector
ax2.contour(X, Y, C, levels=10, colors='gray', alpha=0.3)
ax2.quiver(X, Y, Jx, Jy, color='red', alpha=0.7)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Diffusion Flux J = -D∇C')
ax2.axis('equal')
plt.tight_layout()
plt.show()

# Divergence div J (net inflow/outflow)
div_J = np.gradient(Jx, h, axis=1) + np.gradient(Jy, h, axis=0)

# Compare against the exact value: div J = -D ∇²C, and for C = exp(-(x²+y²))
# we have ∇²C = (4(x²+y²) - 4)·C.
i_c, j_c = len(y)//2, len(x)//2
exact_div_J = -D * (4*(X[i_c, j_c]**2 + Y[i_c, j_c]**2) - 4) * C[i_c, j_c]

print(f"\nGrid point used: (x, y) = ({X[i_c, j_c]:.3f}, {Y[i_c, j_c]:.3f}), h = {h:.3f}")
print(f"Divergence at center: div J = {div_J[i_c, j_c]:.6f}")
print(f"Exact value -D∇²C:    div J = {exact_div_J:.6f}")
print(f"Discretisation error:         {abs(div_J[i_c, j_c] - exact_div_J):.2e}")
print("(negative → inflow, positive → outflow)")
```

**Output:**

```
Grid point used: (x, y) = (0.000, 0.000), h = 0.200
Divergence at center: div J = 3.696405
Exact value -D∇²C:    div J = 4.000000
Discretisation error:         3.04e-01
```

**📝 Note:** The 7.6% gap is not a bug — applying `np.gradient` twice builds a wide 4h-spaced second-difference stencil, whose truncation error at h = 0.2 is exactly this size. Halving h roughly quarters the error. Always print an exact or reference value next to a numerical one; otherwise there is no way to tell a discretisation error from a coding error.

## Summary

  * Vector fields associate vectors to each point in space, representing fluid velocity, electric fields, etc.
  * Gradient (grad) is a vector field pointing in the direction of steepest increase of a scalar field
  * Divergence (div) represents the "source" of a vector field, curl represents the "vortex"
  * Laplacian is an important operator describing diffusion and wave phenomena
  * In conservative fields, curl is zero and a potential function exists

[← Chapter 3: Multivariable Functions](<chapter-3.html>) [Chapter 5: Line & Surface Integrals →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
