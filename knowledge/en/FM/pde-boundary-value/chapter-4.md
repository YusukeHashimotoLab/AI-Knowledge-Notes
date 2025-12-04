---
title: "Chapter 4: Calculus of Variations and Optimization"
chapter_title: "Chapter 4: Calculus of Variations and Optimization"
subtitle: Calculus of Variations and Optimization
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/pde-boundary-value/chapter-4.html>) | Last sync: 2025-11-16

[FM Dojo](<../index.html>) > [Partial Differential Equations and Boundary Value Problems](<index.html>) > Chapter 4 

## 🎯 Learning Objectives

  * Understand the fundamental concepts of calculus of variations and functionals
  * Learn the derivation and applications of the Euler-Lagrange equation
  * Solve the brachistochrone curve problem
  * Understand geodesics and shortest path problems
  * Learn the principle of least action and its applications in physics
  * Handle isoperimetric problems and constrained extremal problems
  * Implement the fundamentals of finite element method and Galerkin method
  * Understand applications to materials science (elastic deformation, shape optimization)

## 📖 What is Calculus of Variations?

### Functionals and Variations

**Functional** is a mapping that takes a function as input and produces a real number as output:

\\[ J[y] = \int_{x_1}^{x_2} F(x, y(x), y'(x)) dx \\]

**Variational problem** : Find a function \\(y(x)\\) that extremizes the functional \\(J[y]\\)

**Variation** \\(\delta y\\): Infinitesimal change of the function \\(y(x)\\)

\\[ y(x) \to y(x) + \epsilon \eta(x), \quad \eta(x_1) = \eta(x_2) = 0 \\]

The condition that the first variation of the functional vanishes gives the extremal condition:

\\[ \delta J = \frac{d}{d\epsilon}J[y + \epsilon\eta]\bigg|_{\epsilon=0} = 0 \\]

### Euler-Lagrange Equation

A function \\(y(x)\\) that extremizes the functional \\(J[y] = \int F(x, y, y') dx\\) satisfies the following differential equation:

\\[ \frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0 \\]

This is called the **Euler-Lagrange equation**.

### Physical Significance

  * **Principle of least action** : The motion of a physical system occurs along a path that extremizes the action integral
  * **Energy minimization** : Equilibrium states minimize the energy functional
  * **Elastic deformation** : Deformation of elastic bodies minimizes strain energy
  * **Shape optimization** : Optimize performance functionals in structural design

## Summary

  * **Calculus of variations** is a method for finding functions that extremize functionals, with the Euler-Lagrange equation as its foundation
  * The **brachistochrone curve** (curve of fastest descent) is a cycloid and represents a classical application of calculus of variations
  * **Geodesics** are shortest paths on surfaces; on a sphere, they are great circles
  * The **principle of least action** is a fundamental principle of physics and forms the basis of Lagrangian mechanics
  * In the **isoperimetric problem** , the figure that maximizes area for a given perimeter is a circle
  * The **Galerkin method** is a powerful technique for solving partial differential equations in weak form
  * The **finite element method** is based on variational principles and is widely applied to elastic body deformation and shape optimization
  * In materials science, energy minimization principles and shape optimization are of practical importance

[← Chapter 3: Laplace Equation and Potential Theory](<chapter-3.html>) [Chapter 5: Numerical Methods and Finite Element Method →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
