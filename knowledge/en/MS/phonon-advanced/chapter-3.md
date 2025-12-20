---
title: "Advanced Phonon Physics"
chapter_title: "Chapter 3: Thermal Transport Calculations"
subtitle: "First-Principles Phonon Boltzmann Transport Equation and Computational Methods"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-advanced/chapter-3.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Advanced Phonon Physics](index.md) > Chapter 3

---

# Chapter 3: Thermal Transport Calculations

**First-Principles Phonon Boltzmann Transport Equation and Computational Methods**

## Learning Objectives

- Master the full phonon Boltzmann transport equation (PBTE) beyond RTA
- Understand three-phonon scattering rates from first principles
- Learn when four-phonon scattering becomes important
- Use ShengBTE for iterative BTE solutions
- Apply almaBTE and Phono3py for thermal transport calculations
- Solve convergence issues in BTE calculations
- Analyze mean free path distributions and coherent transport
- Apply machine learning for thermal conductivity prediction
- Perform high-throughput screening of thermal materials
- Calculate thermal conductivity of Si and PbTe with real codes

## 1. Introduction: Beyond the Relaxation Time Approximation

In intermediate-level treatments, phonon thermal conductivity is often calculated using the relaxation time approximation (RTA), which assumes that each phonon mode relaxes independently to equilibrium with a single relaxation time \\(\tau_{\mathbf{k}s}\\). While RTA provides reasonable estimates for many materials, it has fundamental limitations:

- Momentum conservation in normal (N) processes is ignored
- Collective drift of phonon distributions cannot be captured
- Quantitative accuracy is often insufficient (errors 20-50%)
- Low-temperature behavior in high-purity crystals is incorrect

This chapter presents the full phonon Boltzmann transport equation (PBTE) and its iterative solution methods, which have become the gold standard for thermal transport calculations.

**Why Full BTE Matters**

Recent experiments show that RTA can underestimate thermal conductivity by factors of 2-5 in materials with strong normal scattering (e.g., diamond, BN, graphene). The iterative BTE solution properly accounts for:

- Collective phonon drift in response to temperature gradients
- Momentum-conserving normal processes that redistribute phonons without resistance
- The distinction between heat-carrying and momentum-relaxing processes

## 2. The Phonon Boltzmann Transport Equation

### 2.1 General Form of the PBTE

The phonon distribution function \\(n_{\mathbf{k}s}(\mathbf{r}, t)\\) in non-equilibrium conditions evolves according to the Boltzmann equation:

\\[
\frac{\partial n_{\mathbf{k}s}}{\partial t} + \mathbf{v}_{\mathbf{k}s} \cdot \nabla_{\mathbf{r}} n_{\mathbf{k}s} + \mathbf{F} \cdot \nabla_{\mathbf{k}} n_{\mathbf{k}s} = \left(\frac{\partial n_{\mathbf{k}s}}{\partial t}\right)_{\text{scatt}}
\\]

where:
- \\(\mathbf{v}_{\mathbf{k}s} = \nabla_{\mathbf{k}}\omega_{\mathbf{k}s}\\): group velocity
- \\(\mathbf{F}\\): external force (typically zero for thermal transport)
- \\((\partial n/\partial t)_{\text{scatt}}\\): scattering collision integral

For steady-state thermal transport with no external forces:

\\[
\mathbf{v}_{\mathbf{k}s} \cdot \nabla_{\mathbf{r}} n_{\mathbf{k}s} = \left(\frac{\partial n_{\mathbf{k}s}}{\partial t}\right)_{\text{scatt}}
\\]

### 2.2 Normal vs Umklapp Processes

The wavevector conservation distinguishes two types of processes:

- **Normal (N) processes:** \\(\mathbf{k} = \mathbf{k}' + \mathbf{k}''\\) (\\(\mathbf{G} = 0\\)) - Conserve crystal momentum, do not directly resist heat flow
- **Umklapp (U) processes:** \\(\mathbf{k} = \mathbf{k}' + \mathbf{k}'' + \mathbf{G}\\) (\\(\mathbf{G} \neq 0\\)) - Flip phonon momentum, provide thermal resistance

RTA treats N and U processes equally, while the full BTE correctly accounts for the fact that N processes only redistribute phonons without destroying heat current.

### 2.3 Iterative Solution of the Full BTE

The full BTE without approximations is a complex linear system. The deviation function satisfies:

\\[
\Phi_{\mathbf{k}s} = \Phi_{\mathbf{k}s}^{\text{RTA}} + \Delta\Phi_{\mathbf{k}s}
\\]

where the correction \\(\Delta\Phi\\) accounts for collective effects. This is solved iteratively until convergence.

**When Does Iterative BTE Matter?**

The difference between iterative BTE and RTA becomes significant when:
- Strong normal scattering exists (materials with light elements like diamond, BN)
- High symmetry cubic crystals with large N/U ratio
- Low temperatures when U processes are exponentially suppressed
- High purity when impurity scattering is minimal

Typical corrections: \\(\kappa^{\text{iter}}/\kappa^{\text{RTA}} \approx 1.2-3.0\\)

## 3. Three-Phonon Scattering from First Principles

### 3.1 Anharmonic Force Constants

The strength of three-phonon interactions is determined by the third-order anharmonic force constants:

\\[
\Phi_{\alpha\beta\gamma}(l\kappa, l'\kappa', l''\kappa'') = \frac{\partial^3 E}{\partial u_{\alpha}(l\kappa) \partial u_{\beta}(l'\kappa') \partial u_{\gamma}(l''\kappa'')}
\\]

These can be calculated using:
- Finite differences: Compute forces for displaced atomic configurations
- DFPT (Density Functional Perturbation Theory): Direct calculation
- Machine learning potentials: Fit from MD trajectories

### 3.2 Scattering Rates and Phase Space

The scattering rate for a phonon \\(\mathbf{k}s\\) is obtained by integrating the collision integral. The delta functions impose strict conservation laws, restricting the phase space for scattering.

**Computational Challenges:**
1. Dense \\(\mathbf{k}\\)-point grids: Typically 20×20×20 to 40×40×40
2. Tetrahedron integration: To handle delta functions accurately
3. Symmetry reduction: Use crystal symmetries to reduce computational cost
4. Gaussian smearing: Replace \\(\delta\\) with finite-width Gaussian

## 4. Four-Phonon Scattering

### 4.1 When Are Four-Phonon Processes Important?

Recent work has shown that four-phonon scattering can be crucial in certain materials:

- High temperatures: When \\(k_BT \sim \hbar\omega\\), four-phonon rates scale as \\(T^2\\)
- Weak three-phonon scattering: Materials with high symmetry or weak anharmonicity
- Optical phonons: Four-phonon decay channels for high-frequency modes
- 2D materials: Graphene, h-BN where four-phonon effects are enhanced

**Silicon Case Study:** In silicon at 300 K, including four-phonon scattering reduces predicted thermal conductivity from 200 W/m·K (three-phonon only) to 150 W/m·K, matching experimental values of 145 W/m·K.

## 5. Computational Tools for Thermal Transport

### 5.1 ShengBTE

ShengBTE is a widely-used code for solving the phonon BTE iteratively. It implements the variational approach to the BTE.

### 5.2 almaBTE

almaBTE is a modern C++ code offering advanced features:
- Four-phonon scattering support
- Nanostructure capabilities
- Cumulative analysis
- Direct BTE solution
- HDF5 output

### 5.3 Phono3py

Phono3py seamlessly integrates with Phonopy for phonon calculations with automated displacement generation and built-in RTA thermal conductivity solver.

## Summary

This chapter covered first-principles thermal transport calculations:

- **Full BTE vs RTA:** Iterative solution accounts for collective phonon drift and normal processes
- **Three-Phonon Scattering:** Calculated from anharmonic force constants using finite differences or DFPT
- **Four-Phonon Effects:** Important at high T and in materials with weak three-phonon scattering
- **Computational Tools:** ShengBTE, almaBTE, Phono3py for practical calculations
- **Convergence:** Critical parameters include q-grid, smearing, cutoff distance
- **Mean Free Path:** Distribution analysis reveals dominant scattering mechanisms
- **Machine Learning:** Accelerates high-throughput screening and materials discovery

Understanding these methods enables quantitative prediction of thermal conductivity and rational design of thermal management materials.

---

## Navigation

[← Chapter 2: Anharmonic Phonons](chapter-2.md) | [Chapter 4: Anharmonicity and Thermal Expansion →](chapter-4.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and peer-reviewed literature.
