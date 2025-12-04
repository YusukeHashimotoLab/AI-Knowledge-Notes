---
title: "Chapter 5: Numerical Methods and Finite Element Method"
chapter_title: "Chapter 5: Numerical Methods and Finite Element Method"
subtitle: Numerical Methods and Finite Element Method
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/pde-boundary-value/chapter-5.html>) | Last sync: 2025-11-16

[FM Dojo](<../index.html>) > [Partial Differential Equations and Boundary Value Problems](<index.html>) > Chapter 5 

## 🎯 Learning Objectives

  * Understand the fundamentals of finite difference method (FDM) and various schemes
  * Master the theory and implementation of finite element method (FEM)
  * Learn the characteristics of time integration schemes (explicit and implicit methods)
  * Understand the theoretical foundations of stability analysis and convergence
  * Learn mesh generation and element selection
  * Understand extensions to 2D and 3D problems
  * Master efficient handling of sparse matrices
  * Implement applications to process simulations (heat treatment, reaction-diffusion)

## 📖 Fundamentals of Numerical Methods

### Classification of Numerical Methods

**Finite Difference Method (FDM)** :

  * Replaces derivatives with difference approximations
  * Easy to implement on structured grids
  * Difficult to apply to complex geometries

**Finite Element Method (FEM)** :

  * Uses weak formulation based on variational principles
  * Handles complex geometries with unstructured grids
  * High-accuracy interpolation within elements

**Finite Volume Method (FVM)** :

  * Handles conservation laws in integral form
  * Widely used in fluid dynamics
  * Strictly conserves mass and energy

### Stability and Convergence

**Stability** : Condition that numerical errors do not diverge during time evolution

**CFL Condition (Courant-Friedrichs-Lewy)** : Stability condition for wave equations

\\[ C = c \frac{\Delta t}{\Delta x} \leq C_{\text{max}} \\]

**Convergence** : Property of approaching the true solution as mesh width \\(\Delta x \to 0\\)

**Consistency** : Property that the difference scheme converges to the differential equation

**Lax Equivalence Theorem** : Consistency + Stability ⇒ Convergence

## Summary

  * The **finite difference method** is easy to implement but difficult to apply to complex geometries. Understanding the characteristics of FTCS, BTCS, and Crank-Nicolson schemes is important
  * The **finite element method** is based on variational principles and handles complex geometries with unstructured grids. Linear triangular elements are fundamental
  * **Stability and convergence** are fundamental concepts that ensure the reliability of numerical methods. The CFL condition and Lax equivalence theorem are important
  * **Adaptive mesh refinement** enables efficient high-accuracy solutions
  * **Time-dependent problems** are time-integrated using explicit or implicit methods after semi-discretization
  * **Nonlinear problems** are solved using iterative methods such as Newton-Raphson method
  * Efficient handling of **sparse matrices** is key to large-scale problems
  * Rich practical applications to materials science such as process simulations (quenching, thermal stress)

[← Chapter 4: Calculus of Variations and Optimization](<chapter-4.html>) [Series TOP](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
