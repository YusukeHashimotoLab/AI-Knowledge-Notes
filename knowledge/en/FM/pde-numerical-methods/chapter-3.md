---
title: "Chapter 3: Fundamentals of Finite Element Method"
chapter_title: "Chapter 3: Fundamentals of Finite Element Method"
subtitle: Weak form, shape functions, and stiffness matrices for 1D FEM
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/pde-numerical-methods/chapter-3.html>) | Last sync: 2026-07-11

[Fundamentals of Mathematics & Physics Dojo](<../index.html>) > [Numerical Methods for PDEs](<index.html>) > Chapter 3 

## Learning Objectives

In the finite difference method (Chapter 1) and the Crank-Nicolson method (Chapter 2), we approximated derivatives by difference quotients on grid points. In this chapter we study a fundamentally different idea: the **finite element method (FEM)**. The finite element method rewrites the equation in a "weak form," partitions the domain into small elements, and approximates the solution with piecewise functions. This idea adapts flexibly to complex geometries and non-uniform meshes, and it has become a standard technique across structural mechanics, heat transfer, and electromagnetic field analysis. 

By the end of this chapter, you will be able to:

  * Explain how the finite element method differs in spirit from the finite difference method
  * Derive the weak form of the 1D Poisson equation yourself using integration by parts
  * Build the element stiffness matrix from linear shape functions and assemble it into the global matrix
  * Understand the difference between how Dirichlet and Neumann boundary conditions are handled
  * Implement a 1D finite element solver from scratch using only NumPy, compare it with the exact solution, and confirm the order of convergence

Estimated reading time: 30-35 minutes / Difficulty: Intermediate / Code examples: 4 / Exercises: 3

## 3.1 What Is the Finite Element Method?

The finite difference method replaced the derivatives appearing in a differential equation directly with differences of values at grid points. It is intuitive to implement and very powerful on regular grids. On the other hand, for problems with complex boundary shapes, or where locally finer resolution is required, the constraint of a regular grid becomes a weakness. 

The finite element method (FEM) partitions the target domain into small pieces called **elements** (line segments in 1D; triangles or quadrilaterals in 2D), takes the values at the **nodes** at the ends of each element as unknowns, and expresses the solution as a superposition of piecewise **basis functions**. Its distinctive feature is that the unknowns are determined so as to satisfy not the differential equation itself, but the **weak form** obtained by relaxing it into an integral form. 

### Basic terminology of the finite element method

  * **Element** : a small subinterval into which the domain is partitioned. In 1D, a line segment bounded by adjacent nodes.
  * **Node** : an endpoint of an element. The value of the solution here is the direct unknown.
  * **Shape function** : a basis function that interpolates the solution within each element. In this chapter we use piecewise linear functions.
  * **Stiffness matrix** : the coefficient matrix arising from the left-hand side of the weak form. The name derives from structural mechanics.

The differences from the finite difference method can be organized as follows. 

Aspect| Finite difference method (FDM)| Finite element method (FEM)  
---|---|---  
Object approximated| Derivatives at each point| Weak form (integral equation)  
Representation of solution| Grid-point values| Linear combination of shape functions  
Mesh| Regular grid by default| Flexible with irregular meshes  
Complex geometry| Difficult to handle| Well suited  
Boundary conditions| Substituted directly| Dirichlet enforced, Neumann enters naturally  
  
**Subject of this chapter:** To keep the exposition clear, this chapter focuses on the 1D Poisson equation \\( -u''(x) = f(x) \\), \\( x \in (0,1) \\), with boundary conditions \\( u(0)=u(1)=0 \\). Even in low dimension, the entire skeleton of the finite element method-weak form, shape functions, assembly, and boundary condition handling-appears here. 

## 3.2 Weak Form and the Variational Principle

The differential equation we start from is called the **strong form**. 

### Strong form

\\[ -u''(x) = f(x), \quad x \in (0,1), \qquad u(0) = u(1) = 0. \\] 

To obtain the weak form, we introduce a **test function** \\( v(x) \\). We take \\( v \\) to be a sufficiently smooth function satisfying \\( v(0)=v(1)=0 \\) at the boundary. Multiplying both sides of the strong form by \\( v \\) and integrating over \\( (0,1) \\) gives: 

\\[ -\int_0^1 u''(x)\,v(x)\,dx = \int_0^1 f(x)\,v(x)\,dx. \\] 

We apply **integration by parts** to the left-hand side. 

\\[ -\int_0^1 u''\,v\,dx = -\bigl[\,u'\,v\,\bigr]_0^1 + \int_0^1 u'(x)\,v'(x)\,dx. \\] 

Here the boundary term \\( [u'v]_0^1 \\) vanishes because the test function satisfies \\( v(0)=v(1)=0 \\). We therefore obtain the following weak form. 

### Weak form (variational form)

For every test function \\( v \\) with \\( v(0)=v(1)=0 \\),

\\[ \int_0^1 u'(x)\,v'(x)\,dx = \int_0^1 f(x)\,v(x)\,dx. \\] 

Whereas the strong form required \\( u \\) to be twice differentiable, in the weak form only the first derivatives of \\( u \\) and \\( v \\) appear. The fact that the required smoothness is "weakened" is the origin of the name weak form. Thanks to this relaxation, even piecewise linear functions such as polylines can be used as candidate solutions. 

### The Galerkin method

The method that approximates the unknown function as \\( u(x) \approx u_h(x) = \sum_{j} u_j\,\phi_j(x) \\) using a finite number of basis functions \\( \phi_j \\), and uses the same basis \\( \phi_i \\) for the test functions, is called the Galerkin method. Substituting into the weak form yields a system of linear equations for the nodal values \\( u_j \\): \\( \sum_j \left(\int_0^1 \phi_i' \phi_j'\,dx\right) u_j = \int_0^1 f\,\phi_i\,dx \\). The coefficient matrix on the left is the stiffness matrix \\( K \\), and the right-hand side is the load vector \\( F \\). 

## 3.3 Shape Functions and the Element Stiffness Matrix

Partition the interval \\( [0,1] \\) into \\( n_e \\) elements, with nodes \\( 0 = x_0 < x_1 < \dots < x_{n_e} = 1 \\). As the basis function \\( \phi_j \\) associated with each node \\( x_j \\), we use the triangular **hat function** that is "1 at \\( x_j \\) and 0 at the neighboring nodes." This is the piecewise linear shape function. 

In the implementation it is convenient to map each element to a reference element \\( \xi \in [-1, 1] \\). The linear shape functions on the reference element are the following two: 

\\[ N_1(\xi) = \frac{1-\xi}{2}, \qquad N_2(\xi) = \frac{1+\xi}{2}. \\] 

\\( N_1 \\) is 1 at the left end \\( \xi=-1 \\) and 0 at the right end, while \\( N_2 \\) is the opposite, and \\( N_1+N_2=1 \\) (partition of unity) always holds. On an element of width \\( h \\), the derivatives of the shape functions are \\( \dfrac{d N_1}{dx} = -\dfrac{1}{h} \\) and \\( \dfrac{d N_2}{dx} = +\dfrac{1}{h} \\). 

Taking just one element's worth of the left-hand side of the weak form gives the **element stiffness matrix**. 

### Element stiffness matrix

\\[ K^{e}_{ab} = \int_{x_L}^{x_R} \frac{dN_a}{dx}\,\frac{dN_b}{dx}\,dx = \frac{1}{h} \begin{pmatrix} 1 & -1 \\\ -1 & 1 \end{pmatrix}. \\] 

Since the derivatives are constant within the element, the integral amounts simply to multiplying by the width \\( h \\).

Let us confirm the shape functions and the element stiffness matrix in code. Everything below has been executed, and the actual output is shown. 

Code Example 1: Shape functions and the element stiffness matrix

`import numpy as np def shape_functions(xi): """Return the linear shape functions N1, N2 on the reference element [-1, 1]""" N1 = (1.0 - xi) / 2.0 N2 = (1.0 + xi) / 2.0 return N1, N2 def element_stiffness(h): """Return the element stiffness matrix for an element of width h""" return (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]]) print("=== Shape function values (check partition of unity N1+N2=1) ===") for xi in [-1.0, -0.5, 0.0, 0.5, 1.0]: N1, N2 = shape_functions(xi) print(f"xi={xi:+.1f}: N1={N1:.3f}, N2={N2:.3f}, sum={N1+N2:.3f}") print() print("=== Element stiffness matrix (h=0.25) ===") print(element_stiffness(0.25))`

=== Shape function values (check partition of unity N1+N2=1) === xi=-1.0: N1=1.000, N2=0.000, sum=1.000 xi=-0.5: N1=0.750, N2=0.250, sum=1.000 xi=+0.0: N1=0.500, N2=0.500, sum=1.000 xi=+0.5: N1=0.250, N2=0.750, sum=1.000 xi=+1.0: N1=0.000, N2=1.000, sum=1.000 === Element stiffness matrix (h=0.25) === [[ 4. -4.] [-4. 4.]]

At every evaluation point \\( N_1+N_2=1 \\) holds, confirming that the shape functions correctly partition unity. When \\( h=0.25 \\) we have \\( 1/h = 4 \\), so the entries of the element stiffness matrix being \\( \pm 4 \\) is also consistent with the theory. 

## 3.4 Handling Boundary Conditions and Solving

The operation of adding up each element's element stiffness matrix at the positions of the shared nodes to build the **global stiffness matrix** \\( K \\) is called **assembly**. The contribution of element \\( e \\) (which has nodes \\( e \\) and \\( e+1 \\)) is added into the corresponding rows and columns of the global matrix. Because adjacent elements share nodes, the diagonal entries of interior nodes accumulate contributions from two elements. 

Code Example 2: Assembly of the global stiffness matrix

`import numpy as np def assemble_global_stiffness(n_elem): """Assemble the global stiffness matrix on a uniform mesh of [0,1]""" n_nodes = n_elem + 1 h = 1.0 / n_elem K = np.zeros((n_nodes, n_nodes)) ke = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]]) for e in range(n_elem): # loop over elements for a in range(2): for b in range(2): K[e + a, e + b] += ke[a, b] # add to the corresponding position return K K = assemble_global_stiffness(4) print("=== Global stiffness matrix (n_elem=4, h=0.25, before BC) ===") print(K) # Dirichlet BC: remove the end degrees of freedom 0 and n n_nodes = K.shape[0] interior = np.arange(1, n_nodes - 1) print() print("=== Reduced system of interior DOFs only (after BC) ===") print(K[np.ix_(interior, interior)])`

=== Global stiffness matrix (n_elem=4, h=0.25, before BC) === [[ 4. -4. 0. 0. 0.] [-4. 8. -4. 0. 0.] [ 0. -4. 8. -4. 0.] [ 0. 0. -4. 8. -4.] [ 0. 0. 0. -4. 4.]] === Reduced system of interior DOFs only (after BC) === [[ 8. -4. 0.] [-4. 8. -4.] [ 0. -4. 8.]]

The diagonal entries of the interior nodes become \\( 8 = 2/h \\), which shows that two elements' worth of contributions overlap. The matrix is a symmetric tridiagonal matrix, with a structure very similar to the Laplacian matrix that appeared in the finite difference method. 

### Two kinds of boundary conditions

  * **Dirichlet condition** : specifies the value itself at the boundary (here \\( u(0)=u(1)=0 \\)). It is an **essential boundary condition** enforced on both the candidate solution and the test functions, and it is imposed by removing the corresponding degrees of freedom from the system (or by substituting their values). 
  * **Neumann condition** : specifies the derivative (flux) \\( u'(x) \\) at the boundary. Because it is incorporated into the load vector naturally through the boundary term \\( [u'v]_0^1 \\) that appeared in the integration by parts of the weak form, it is called a **natural boundary condition**. For the homogeneous Neumann condition \\( u'=0 \\), the boundary term vanishes and no special treatment is needed. 

Once the Dirichlet condition gives us a reduced system of interior degrees of freedom only, \\( K_{\text{int}}\,u_{\text{int}} = F_{\text{int}} \\), all that remains is to solve it. Since the system is symmetric positive definite, it can be solved stably with `numpy.linalg.solve`. 

## 3.5 Hands-on with Python: Implementing 1D FEM

We now combine the element stiffness matrix, assembly, and boundary condition handling covered so far into one, and implement a finite element solver for the 1D Poisson equation from scratch using only NumPy. The load vector \\( F_i = \int f\,\phi_i\,dx \\) is integrated numerically per element using **2-point Gauss quadrature**. The 2-point Gauss rule integrates cubic polynomials exactly, which is sufficient for the load computation of piecewise linear elements. 

For verification we use the **method of manufactured solutions**. Fixing the exact solution as \\( u(x)=\sin(\pi x) \\), we have \\( -u''(x) = \pi^2 \sin(\pi x) \\), so \\( f(x)=\pi^2\sin(\pi x) \\). We use this \\( f \\) as input, compute the numerical solution, and compare it with the known exact solution. 

Code Example 3: FEM solver for the 1D Poisson equation and comparison with the exact solution

`import numpy as np def solve_poisson_fem(n_elem, f_func): """Solve -u''(x) = f(x) on [0,1], u(0)=u(1)=0 with linear FEM""" n_nodes = n_elem + 1 nodes = np.linspace(0.0, 1.0, n_nodes) K = np.zeros((n_nodes, n_nodes)) F = np.zeros(n_nodes) # 2-point Gauss quadrature on the reference element [-1, 1] gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]) gauss_wts = np.array([1.0, 1.0]) for e in range(n_elem): x_left, x_right = nodes[e], nodes[e + 1] h = x_right - x_left # element stiffness matrix ke = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]]) # element load vector (Gauss quadrature) fe = np.zeros(2) for xi, w in zip(gauss_pts, gauss_wts): N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0]) x_phys = x_left + (xi + 1.0) / 2.0 * h # map to physical coordinate fe += w * (h / 2.0) * f_func(x_phys) * N # assembly for a in range(2): F[e + a] += fe[a] for b in range(2): K[e + a, e + b] += ke[a, b] # Dirichlet BC: remove the end degrees of freedom interior = np.arange(1, n_nodes - 1) K_int = K[np.ix_(interior, interior)] F_int = F[interior] u = np.zeros(n_nodes) u[interior] = np.linalg.solve(K_int, F_int) # solve the reduced system return nodes, u # Manufactured solution: u(x) = sin(pi x) -> f(x) = pi^2 sin(pi x) f = lambda x: np.pi**2 * np.sin(np.pi * x) u_exact = lambda x: np.sin(np.pi * x) nodes, u = solve_poisson_fem(8, f) print("nodes :", np.round(nodes, 4)) print("u_FEM :", np.round(u, 6)) print("u_exact:", np.round(u_exact(nodes), 6)) rms = np.sqrt(np.mean((u - u_exact(nodes))**2)) mx = np.max(np.abs(u - u_exact(nodes))) print("RMS nodal error (n_elem=8): {:.3e}".format(rms)) print("max nodal error (n_elem=8): {:.3e}".format(mx))`

nodes : [0. 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1. ] u_FEM : [0. 0.38269 0.707119 0.923895 1.000017 0.923895 0.707119 0.38269 0\. ] u_exact: [0. 0.382683 0.707107 0.92388 1. 0.92388 0.707107 0.382683 0\. ] RMS nodal error (n_elem=8): 1.110e-05 max nodal error (n_elem=8): 1.665e-05

Even with only 8 elements, the nodal values agree with the exact solution to the fourth decimal place. The reason the nodal error is extremely small, on the order of \\( 10^{-5} \\), is a property called **superconvergence** : for the 1D Poisson problem, the finite element solution with linear elements coincides with the exact solution at the nodes. Inside the elements, between nodes, the error is larger than this. 

Let us examine how the overall error, including inside the elements, decreases as the number of elements increases. We measure the difference from the exact solution in the \\( L^2 \\) norm \\( \|u_h - u\|_{L^2} = \left(\int_0^1 (u_h-u)^2\,dx\right)^{1/2} \\), and estimate the order of convergence from how many times the error changes each time the element width \\( h \\) is halved. 

Code Example 4: Verifying convergence (error evaluation for varying element counts)

`import numpy as np def l2_error(nodes, u, u_exact_func): """Compute the L2 error by 3-point Gauss quadrature on each element""" gp = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]) gw = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0]) err2 = 0.0 for e in range(len(nodes) - 1): xl, xr = nodes[e], nodes[e + 1] h = xr - xl for xi, w in zip(gp, gw): N = np.array([(1.0 - xi)/2.0, (1.0 + xi)/2.0]) x_phys = xl + (xi + 1.0)/2.0 * h uh = N[0]*u[e] + N[1]*u[e + 1] # interpolated value within the element err2 += w * (h/2.0) * (uh - u_exact_func(x_phys))**2 return np.sqrt(err2) print("{:>8} {:>10} {:>14} {:>8}".format("n_elem", "h", "L2_error", "rate")) prev_err, prev_h = None, None for n_elem in [4, 8, 16, 32, 64]: nodes, u = solve_poisson_fem(n_elem, f) h = 1.0 / n_elem e = l2_error(nodes, u, u_exact) if prev_err is None: rate = " - " else: rate = "{:.2f}".format(np.log(e/prev_err) / np.log(h/prev_h)) print("{:>8d} {:>10.5f} {:>14.4e} {:>8}".format(n_elem, h, e, rate)) prev_err, prev_h = e, h`

n_elem h L2_error rate 4 0.25000 3.9127e-02 - 8 0.12500 9.9108e-03 1.98 16 0.06250 2.4859e-03 2.00 32 0.03125 6.2198e-04 2.00 64 0.01562 1.5553e-04 2.00

Each time the element width \\( h \\) is halved, the \\( L^2 \\) error becomes roughly \\( 1/4 \\), and the order of convergence converges to 2. This agrees with the theoretical value for piecewise linear elements, \\( \|u_h - u\|_{L^2} = O(h^2) \\), providing strong evidence that the implementation is correct. 

**Implementation takeaway:** The pattern of "build the local matrix -> add it to the global one" inside the element loop remains unchanged when extending to 2D, 3D, or higher-order elements. Viewing a finite element implementation as a repetition of this local-to-global assembly makes it much easier to follow. 

## Exercises

### Exercise 3.1: Deriving the weak form

For the differential equation \\( -u''(x) + u(x) = f(x) \\), \\( x \in (0,1) \\), \\( u(0)=u(1)=0 \\), derive the weak form using a test function \\( v \\) via integration by parts. Compared with the \\( -u''=f \\) of this chapter, explain what term is added to the stiffness matrix. 

### Exercise 3.2: Implementing a Neumann boundary condition

Modify the solver of Code Example 3 to change the right-end boundary condition to \\( u'(1)=g \\) (a Neumann condition). Show how the boundary term \\( [u'v]_0^1 \\) of the weak form contributes to the load vector, and state why the end degree of freedom need not be removed when \\( g=0 \\). 

### Exercise 3.3: Verification with a different manufactured solution

If the exact solution is \\( u(x) = x(1-x) \\), what is the corresponding \\( f(x) \\)? Input that \\( f \\) into Code Examples 3 and 4, compute the numerical solution, and evaluate the \\( L^2 \\) error. Considering the relationship between linear elements and quadratic polynomials, discuss why the error becomes as small as machine precision regardless of the number of elements (hint: pay attention also to the quadrature order of the load). 

## Checking the Learning Objectives

Let us confirm the level of achievement against the learning objectives of this chapter.

  * You can explain, in contrast to the finite difference method, that the finite element method is based on the weak form and represents the solution with elements and shape functions
  * You can derive the weak form of the 1D Poisson equation by integration by parts and elimination of the boundary term
  * You can derive the element stiffness matrix \\( \frac{1}{h}\begin{pmatrix}1 & -1 \\\ -1 & 1\end{pmatrix} \\) from linear shape functions and assemble it into the global matrix
  * You can explain the difference in handling between Dirichlet (essential) and Neumann (natural) conditions
  * You can implement an FEM solver with only NumPy and numerically confirm the order of convergence \\( O(h^2) \\)

## Summary

  * The finite element method relaxes the differential equation into a weak form, partitions the domain into elements, and approximates the solution with piecewise shape functions.
  * The weak form is obtained from integration by parts; because the required smoothness drops to first derivatives, polyline-shaped approximations become possible.
  * From piecewise linear shape functions, for width \\( h \\) one obtains the element stiffness matrix \\( \frac{1}{h}\begin{pmatrix}1 & -1 \\\ -1 & 1\end{pmatrix} \\).
  * Assembly, which sums element stiffness matrices at node positions, builds a tridiagonal global stiffness matrix.
  * Dirichlet conditions are enforced by removing degrees of freedom, while Neumann conditions are incorporated naturally into the load vector.
  * When verified with the method of manufactured solutions, the \\( L^2 \\) error of linear elements converges as \\( O(h^2) \\), exactly as theory predicts.

## Next Steps

In this chapter we learned the finite element method, which solves partial differential equations through integration (the weak form) rather than relying on differences at grid points. In Chapter 4, "Spectral Methods and Monte Carlo Methods," we change perspective yet again, treating **spectral methods** that expand the solution in global basis functions (such as trigonometric functions) and **Monte Carlo methods** that estimate partial differential equations and integrals using random numbers. By placing the finite difference method (Chapters 1-2), the finite element method (Chapter 3), and spectral methods (Chapter 4) side by side, the breadth of design philosophies in numerical methods should come into view. 

## References

  1. Fumio Kikuchi, _Introduction to the Finite Element Method (revised edition)_ , Science-sha, 1999 (in Japanese).
  2. O. C. Zienkiewicz, R. L. Taylor, J. Z. Zhu, _The Finite Element Method: Its Basis and Fundamentals_ , 7th ed., Butterworth-Heinemann, 2013.
  3. C. Johnson, _Numerical Solution of Partial Differential Equations by the Finite Element Method_ , Dover, 2009.
  4. Nobuyoshi Tosaka and Kazue Onishi, _Numerical Solution of Partial Differential Equations (2nd edition)_ , University of Tokyo Press, 2003 (in Japanese).

[← Chapter 2](<chapter-2.html>) [Series Index](<index.html>) [Chapter 4 →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
