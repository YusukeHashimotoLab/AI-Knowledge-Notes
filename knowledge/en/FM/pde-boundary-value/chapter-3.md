---
title: "Chapter 3: Laplace Equation and Potential Theory"
chapter_title: "Chapter 3: Laplace Equation and Potential Theory"
subtitle: Laplace Equation and Potential Theory
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/pde-boundary-value/chapter-3.html>) | Last sync: 2025-11-16

[FM Dojo](<../index.html>) > [Partial Differential Equations and Boundary Value Problems](<index.html>) > Chapter 3 

## 🎯 Learning Objectives

  * Understand the fundamentals of the Laplace equation and potential theory
  * Learn the properties of harmonic functions and the maximum principle
  * Solve the Laplace equation in polar and cylindrical coordinates
  * Master solution methods for boundary value problems using Green's functions
  * Understand the Poisson equation and handling of charge distributions and heat sources
  * Implement numerical methods using iterative methods (Jacobi, Gauss-Seidel, SOR)
  * Understand applications to materials science (electrostatic field analysis, steady-state heat conduction)

## 📖 What is the Laplace Equation?

### Definition of the Laplace Equation

The **Laplace equation** is an elliptic partial differential equation of the following form:

\\[ \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} = 0 \\]

Solutions to the Laplace equation are called **harmonic functions**.

The **Poisson equation** is the case where the right-hand side is non-zero:

\\[ \nabla^2 u = f(x,y,z) \\]

where \\(f\\) represents heat sources or charge density.

### Physical Significance

  * **Electrostatic potential** : Electric potential \\(V\\) in a region without charges \\(\nabla^2 V = 0\\)
  * **Steady-state heat conduction** : Temperature distribution \\(T\\) in a region without heat sources \\(\nabla^2 T = 0\\)
  * **Fluid potential** : Velocity potential \\(\phi\\) for incompressible, inviscid flow \\(\nabla^2 \phi = 0\\)
  * **Gravitational potential** : Gravitational potential in a region without mass

### Properties of Harmonic Functions

**Maximum Principle** : Harmonic functions do not have extrema in the interior; maximum and minimum values are attained on the boundary.

**Mean Value Theorem** : The value of a harmonic function at a point \\((x_0, y_0)\\) equals the average value on a circle centered at that point:

\\[ u(x_0, y_0) = \frac{1}{2\pi} \int_0^{2\pi} u(x_0 + r\cos\theta, y_0 + r\sin\theta) d\theta \\]

**Uniqueness** : Under Dirichlet boundary conditions, the solution to the Laplace equation is unique.

## 💻 Example 3.1: Verification of Harmonic Functions and Maximum Principle

Python implementation: Verification of harmonic function properties
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 💻 Example 3.1: Verification of Harmonic Functions and Maximu
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Example of a harmonic function: u(x,y) = x^2 - y^2 (real part of z^2)
    def harmonic_function(x, y):
     return x**2 - y**2
    
    # Create 2D grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    U = harmonic_function(X, Y)
    
    # Calculate Laplacian (numerical differentiation)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Second partial derivatives
    d2u_dx2 = np.zeros_like(U)
    d2u_dy2 = np.zeros_like(U)
    
    for i in range(1, len(x)-1):
     for j in range(1, len(y)-1):
     d2u_dx2[j, i] = (U[j, i+1] - 2*U[j, i] + U[j, i-1]) / dx**2
     d2u_dy2[j, i] = (U[j+1, i] - 2*U[j, i] + U[j-1, i]) / dy**2
    
    laplacian = d2u_dx2 + d2u_dy2
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Harmonic function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, U, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title('Harmonic function: u = x² - y²')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, U, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour plot')
    ax2.axis('equal')
    
    # Laplacian
    ax3 = fig.add_subplot(133)
    laplacian_plot = ax3.imshow(laplacian[1:-1, 1:-1], extent=[-2, 2, -2, 2],
     origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1)
    plt.colorbar(laplacian_plot, ax=ax3, label='∇²u')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Laplacian (max: {np.max(np.abs(laplacian[1:-1,1:-1])):.2e})')
    
    plt.tight_layout()
    plt.savefig('laplace_harmonic_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Verification of maximum principle
    print("=== Verification of Maximum Principle ===")
    print(f"Maximum in interior: {np.max(U[10:-10, 10:-10]):.4f}")
    print(f"Maximum on boundary: {np.max([np.max(U[0,:]), np.max(U[-1,:]), np.max(U[:,0]), np.max(U[:,-1])]):.4f}")
    print(f"Minimum in interior: {np.min(U[10:-10, 10:-10]):.4f}")
    print(f"Minimum on boundary: {np.min([np.min(U[0,:]), np.min(U[-1,:]), np.min(U[:,0]), np.min(U[:,-1])]):.4f}")
    

**Output explanation** :

  * \\(u = x^2 - y^2\\) is a harmonic function (\\(\nabla^2 u = 2 - 2 = 0\\))
  * Verification that the Laplacian is zero within numerical error
  * By the maximum principle, extrema exist on the boundary

## 📚 Summary

  * The **Laplace equation** describes steady-state physical phenomena and has important properties as a harmonic function, including the maximum principle
  * Through **separation of variables in polar and cylindrical coordinates** , analytical solutions can be obtained for circular and spherical domains
  * Using **Green's functions** , solutions to boundary value problems can be constructed from the response to point sources
  * The **Poisson equation** handles problems involving heat sources and charge distributions, and is widely applied in materials science
  * **Iterative methods** (Jacobi, Gauss-Seidel, SOR) can be used to find numerical solutions, with SOR being the fastest
  * Practical applications to materials science are possible, such as steady-state heat conduction in complex geometries

### 💡 Exercise Problems

  1. **Verification of harmonic functions** : Verify by calculating the Laplacian that \\(u(x,y) = xy\\) is not a harmonic function.
  2. **Solution in polar coordinates** : Find the solution to the Laplace equation on a disk of radius \\(a\\) satisfying the boundary condition \\(u(a,\theta) = \cos(2\theta)\\), and visualize it.
  3. **Application of Green's function** : Using the Green's function for a rectangular domain, find the temperature distribution for a heat source \\(f(x,y) = \delta(x-0.5, y-0.5)\\).
  4. **Comparison of convergence** : Vary the relaxation parameter \\(\omega\\) for the SOR method from 1.0 to 2.0, and find the optimal \\(\omega\\).
  5. **Complex geometry** : Solve the Laplace equation in a rectangular domain with a circular hole, and visualize the temperature distribution around the hole.

[← Chapter 2: Heat Equation and Diffusion](<chapter-2.html>) [Chapter 4: Variational Methods and Optimization →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
