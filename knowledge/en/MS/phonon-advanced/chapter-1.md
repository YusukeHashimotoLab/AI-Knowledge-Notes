---
title: "Advanced Phonon Physics"
chapter_title: "Chapter 1: First-Principles Phonon Calculations"
subtitle: "DFPT, Frozen Phonon Method, and Modern Computational Tools"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-advanced/chapter-1.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Advanced Phonon Physics](index.md) > Chapter 1

---

# Chapter 1: First-Principles Phonon Calculations

**DFPT, Frozen Phonon Method, and Modern Computational Tools**

## Learning Objectives

By completing this chapter, you will be able to:

- Understand the theoretical foundations of Density Functional Perturbation Theory (DFPT)
- Master linear response theory and the 2n+1 theorem
- Implement and compare frozen phonon and DFPT methods
- Calculate force constants and perform Fourier interpolation
- Use modern software tools (Quantum ESPRESSO, VASP, Phonopy) effectively
- Understand convergence parameters (k-points, q-points, supercell size)
- Apply non-analytic corrections for LO-TO splitting in polar materials
- Develop complete Python workflows for phonon calculations

## Introduction

First-principles phonon calculations have revolutionized materials science by enabling accurate prediction of vibrational properties without empirical parameters. Unlike classical force-field approaches, ab initio methods compute phonon properties directly from electronic structure calculations based on density functional theory (DFT).

This chapter covers the two main approaches to first-principles phonon calculations: Density Functional Perturbation Theory (DFPT) and the frozen phonon (finite displacement) method. We will explore their theoretical foundations, practical implementations, advantages and limitations, and integration with modern software packages. Understanding these methods is essential for computational materials research and phonon engineering.

## 1.1 Density Functional Perturbation Theory (DFPT)

### 1.1.1 Motivation and Overview

The dynamical matrix requires second derivatives of the total energy with respect to atomic displacements. A naive approach would use finite differences, requiring many self-consistent calculations for displaced configurations. DFPT provides an elegant alternative by computing the response of the electronic structure to perturbations analytically.

**Definition: DFPT Philosophy**

DFPT calculates the response of the electronic ground state to external perturbations (atomic displacements, electric fields) self-consistently, yielding second-order derivatives of the energy directly without computing finite differences.

### 1.1.2 Linear Response Theory

Consider a perturbation to the Hamiltonian:

\\[
\hat{H} = \hat{H}_0 + \lambda \hat{V}
\\]

where \\(\hat{H}_0\\) is the unperturbed Hamiltonian and \\(\lambda\\) is the perturbation strength. The electronic ground state energy can be expanded as:

\\[
E = E^{(0)} + \lambda E^{(1)} + \frac{1}{2}\lambda^2 E^{(2)} + \mathcal{O}(\lambda^3)
\\]

The density matrix also expands perturbatively:

\\[
\rho(\mathbf{r}) = \rho^{(0)}(\mathbf{r}) + \lambda \rho^{(1)}(\mathbf{r}) + \frac{1}{2}\lambda^2 \rho^{(2)}(\mathbf{r}) + \cdots
\\]

### 1.1.3 The 2n+1 Theorem

**Theorem: 2n+1 Theorem (Wigner, 1935)**

If the ground state wavefunction (or density) is known to order \\(n\\) in the perturbation, the energy can be computed to order \\(2n+1\\).

**Corollary for Phonons:**

Knowing only the unperturbed ground state (\\(n=0\\)), we can compute second-order energy derivatives exactly by solving for the first-order response of the wavefunction/density.

**Proof Sketch:**

The energy functional in DFT is:

\\[
E[\rho] = T[\rho] + \int V_\text{ext}(\mathbf{r})\rho(\mathbf{r})d\mathbf{r} + E_\text{H}[\rho] + E_\text{xc}[\rho]
\\]

The second-order energy change involves:

\\[
E^{(2)} = \int \frac{\delta^2 E}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')} \rho^{(1)}(\mathbf{r})\rho^{(1)}(\mathbf{r}')d\mathbf{r}d\mathbf{r}' + \int \frac{\delta E}{\delta\rho(\mathbf{r})}\rho^{(2)}(\mathbf{r})d\mathbf{r}
\\]

However, the variational principle ensures \\(\delta E/\delta\rho = 0\\) at the ground state, so the second term vanishes! Thus, \\(E^{(2)}\\) depends only on \\(\rho^{(1)}\\), not \\(\rho^{(2)}\\).

### 1.1.4 Self-Consistent Phonon Equation

For an atomic displacement perturbation \\(\lambda u_{\kappa\alpha}\\) at atom \\(\kappa\\) in direction \\(\alpha\\), the first-order change in the Kohn-Sham potential must be solved self-consistently:

\\[
\delta V_\text{KS}(\mathbf{r}) = \delta V_\text{ext}(\mathbf{r}) + \int \frac{\delta^2 E_\text{H}}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')} \delta\rho(\mathbf{r}')d\mathbf{r}' + \int \frac{\delta^2 E_\text{xc}}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')} \delta\rho(\mathbf{r}')d\mathbf{r}'
\\]

where the first-order density change is:

\\[
\delta\rho(\mathbf{r}) = \sum_{n,\mathbf{k}} f_{n\mathbf{k}} \left[\psi^*_{n\mathbf{k}}(\mathbf{r})\delta\psi_{n\mathbf{k}}(\mathbf{r}) + \text{c.c.}\right]
\\]

The first-order wavefunctions satisfy:

\\[
(\hat{H}_\text{KS}^{(0)} - \epsilon_{n\mathbf{k}})\delta\psi_{n\mathbf{k}} = -(\delta V_\text{KS} - \delta\epsilon_{n\mathbf{k}})\psi_{n\mathbf{k}}^{(0)}
\\]

This is solved iteratively until self-consistency, yielding \\(\delta\rho\\) and ultimately the force constant matrix via:

\\[
\Phi_{\alpha\beta}(\kappa, \kappa') = \frac{\partial^2 E}{\partial u_{\kappa\alpha} \partial u_{\kappa'\beta}}
\\]

**Computational Advantage**

DFPT requires solving the linear response equation for each phonon mode, which scales as \\(\mathcal{O}(N_\text{atoms} \times N_\text{q-points})\\). For each \\(\mathbf{q}\\)-point, one self-consistent calculation yields the entire dynamical matrix, making it highly efficient for commensurate phonon wavevectors.

## 1.2 Frozen Phonon (Finite Displacement) Method

### 1.2.1 Conceptual Framework

The frozen phonon method is conceptually simpler: displace atoms in a supercell by small amounts, compute forces (negative energy gradients) via DFT, and extract force constants from the finite-difference approximation.

**Definition: Frozen Phonon Approach**

For a displacement \\(u_{\kappa'\alpha'}\\) of atom \\(\kappa'\\) in direction \\(\alpha'\\), the force constant is approximated by:

\\[
\Phi_{\alpha\alpha'}(\kappa, \kappa') \approx -\frac{F_\alpha(\kappa; u_{\kappa'\alpha'}) - F_\alpha(\kappa; 0)}{u_{\kappa'\alpha'}}
\\]

where \\(F_\alpha(\kappa; u)\\) is the force on atom \\(\kappa\\) in direction \\(\alpha\\) when displacement \\(u\\) is applied.

### 1.2.2 Supercell Construction

To compute force constants in real space, we construct a supercell of size \\(N_1 \times N_2 \times N_3\\) unit cells. The supercell must be large enough to:

- Ensure force constants decay to zero at the supercell boundaries
- Avoid artificial interactions between periodic images
- Capture long-range interactions in polar materials

**Example: Silicon (Diamond Structure)**

For silicon with a diamond structure (2 atoms per primitive cell), a \\(3 \times 3 \times 3\\) supercell contains 54 atoms. With 3 Cartesian directions, we have 162 degrees of freedom. However, symmetry reduces the number of independent displacements dramatically.

Typical procedure:

1. Displace one atom in the supercell by \\(\pm\delta u\\) in each Cartesian direction
2. Compute forces on all atoms via DFT
3. Apply central differences: \\(\Phi \approx [F(+\delta u) - F(-\delta u)]/(2\delta u)\\)
4. Use symmetry to generate remaining force constants

### 1.2.3 Displacement Magnitude

The displacement magnitude \\(\delta u\\) must be chosen carefully:

- **Too small:** Numerical noise dominates, poor force accuracy
- **Too large:** Anharmonic effects contaminate harmonic force constants
- **Optimal range:** Typically 0.01 – 0.03 Å for most materials

Higher-order finite differences can improve accuracy:

\\[
\Phi \approx \frac{-F(+2\delta u) + 8F(+\delta u) - 8F(-\delta u) + F(-2\delta u)}{12\delta u} + \mathcal{O}(\delta u^4)
\\]

### 1.2.4 Advantages and Limitations

| Aspect | Advantages | Limitations |
|--------|-----------|-------------|
| **Conceptual Simplicity** | Intuitive, easy to understand and implement | — |
| **Software Compatibility** | Works with any DFT code (only needs forces) | — |
| **Computational Cost** | Can exploit symmetry to reduce calculations | Requires many DFT runs for large supercells |
| **q-point Sampling** | Provides real-space force constants for all q | Limited q-resolution by supercell size |
| **Numerical Accuracy** | Higher-order finite differences available | Sensitive to force convergence and \\(\delta u\\) |
| **Anharmonicity** | Can extend to third/fourth-order force constants | Harmonic approximation may break for large \\(\delta u\\) |

## 1.3 DFPT vs. Frozen Phonon: Detailed Comparison

### 1.3.1 Accuracy Considerations

**DFPT:**
- Exact within DFT for infinitesimal perturbations
- No finite-difference errors
- Limited by DFT functional approximations only

**Frozen Phonon:**
- Finite-difference truncation error \\(\mathcal{O}(\delta u^2)\\) or higher
- Requires tight force convergence (typically \\(< 10^{-5}\\) eV/Å)
- Anharmonic contamination for large displacements

In practice, with careful convergence testing, both methods yield nearly identical results for harmonic phonon frequencies (differences \\(< 1\\) cm\\(^{-1}\\)).

### 1.3.2 Computational Cost Analysis

For a system with \\(N_\text{atoms}\\) atoms per primitive cell and \\(N_\mathbf{q}\\) q-points:

| Method | Number of SCF Calculations | Cost per Calculation | Total Scaling |
|--------|---------------------------|---------------------|---------------|
| **DFPT** | \\(N_\mathbf{q}\\) (one per q-point) | \\(\sim 3 \times\\) ground state | \\(\mathcal{O}(N_\mathbf{q})\\) |
| **Frozen Phonon** | \\(6N_\text{atoms}^{\text{super}}\\) (symmetry-reduced) | Standard ground state | \\(\mathcal{O}(N_\text{atoms}^{\text{super}})\\) |

**Key Trade-offs:**

- **Few q-points needed:** DFPT is more efficient
- **Dense q-grid needed:** Frozen phonon may be better (compute once, interpolate everywhere)
- **Large primitive cell:** Frozen phonon often faster
- **Small primitive cell:** DFPT typically faster

### 1.3.3 Special Cases and Best Practices

**Prefer DFPT when:**
- Working with simple structures (1–4 atoms per cell)
- Need phonons at specific high-symmetry q-points
- Computing dielectric responses and Born charges
- Polar materials requiring non-analytic corrections

**Prefer Frozen Phonon when:**
- Large primitive cells (> 10 atoms)
- Software doesn't support DFPT (e.g., some DFT codes)
- Need third/fourth-order anharmonic force constants
- Studying disordered or defect systems

## 1.4 Force Constant Calculation and Fourier Interpolation

### 1.4.1 Real-Space Force Constants

Both DFPT and frozen phonon methods ultimately provide force constants in real space:

\\[
\Phi_{\alpha\beta}(\mathbf{R}_0 + \boldsymbol{\tau}_\kappa, \mathbf{R}_n + \boldsymbol{\tau}_{\kappa'})
\\]

where \\(\mathbf{R}_n\\) is a lattice vector, \\(\boldsymbol{\tau}_\kappa\\) is the position of atom \\(\kappa\\) within the unit cell, and \\(\alpha, \beta\\) are Cartesian indices.

Due to translational symmetry, force constants depend only on the relative lattice vector:

\\[
\Phi_{\alpha\beta}(\mathbf{R}_0 + \boldsymbol{\tau}_\kappa, \mathbf{R}_n + \boldsymbol{\tau}_{\kappa'}) = \Phi_{\alpha\beta}(\mathbf{R}_n; \kappa, \kappa')
\\]

### 1.4.2 Fourier Transform to q-Space

The dynamical matrix at wavevector \\(\mathbf{q}\\) is obtained via Fourier transform:

\\[
D_{\kappa\alpha,\kappa'\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{\mathbf{R}_n} \Phi_{\alpha\beta}(\mathbf{R}_n; \kappa, \kappa') e^{i\mathbf{q}\cdot(\mathbf{R}_n + \boldsymbol{\tau}_{\kappa'} - \boldsymbol{\tau}_\kappa)}
\\]

This allows computing phonons at any \\(\mathbf{q}\\)-point in the Brillouin zone from a finite set of real-space force constants.

### 1.4.3 Fourier Interpolation

The frozen phonon method with an \\(N_1 \times N_2 \times N_3\\) supercell directly yields force constants on a commensurate grid:

\\[
\mathbf{q} = \frac{m_1}{N_1}\mathbf{b}_1 + \frac{m_2}{N_2}\mathbf{b}_2 + \frac{m_3}{N_3}\mathbf{b}_3
\\]

To obtain phonons at arbitrary \\(\mathbf{q}\\), we use Fourier interpolation:

1. Compute real-space force constants from supercell displacements
2. Fourier transform to get \\(D(\mathbf{q})\\) at any desired \\(\mathbf{q}\\)
3. Diagonalize to obtain \\(\omega_j(\mathbf{q})\\) and eigenvectors

**Convergence with Supercell Size**

The accuracy of Fourier interpolation depends on how well the supercell captures the range of force constants. If \\(\Phi(\mathbf{R})\\) has not decayed to zero at the supercell boundary, phonon frequencies will show unphysical oscillations. Always test convergence by increasing supercell size until phonon frequencies change by less than your target accuracy (e.g., 1 cm\\(^{-1}\\)).

### 1.4.4 Acoustic Sum Rule Enforcement

Numerical errors can violate the acoustic sum rule:

\\[
\sum_{\mathbf{R}_n, \kappa'} \Phi_{\alpha\beta}(\mathbf{R}_n; \kappa, \kappa') = 0
\\]

This results in spurious imaginary frequencies at the \\(\Gamma\\) point. Post-processing tools (e.g., Phonopy) can enforce the sum rule by symmetrizing force constants:

\\[
\Phi^\text{corrected}_{\alpha\beta}(0; \kappa, \kappa) = -\sum_{\mathbf{R}_n \neq 0, \kappa'} \Phi_{\alpha\beta}(\mathbf{R}_n; \kappa, \kappa')
\\]

## Summary

This chapter provided a comprehensive introduction to first-principles phonon calculations, covering both theoretical foundations and practical implementations:

- **DFPT:** Theoretical basis via linear response theory and the 2n+1 theorem, enabling direct computation of second-order energy derivatives
- **Frozen Phonon Method:** Finite-displacement approach using supercells and force calculations, more flexible but computationally intensive
- **Comparison:** Accuracy, computational cost, and appropriate use cases for each method
- **Force Constants:** Real-space calculation, Fourier interpolation to arbitrary q-points, and acoustic sum rule enforcement
- **Software Tools:** Practical workflows with Quantum ESPRESSO, VASP+Phonopy, ABINIT, and Python APIs
- **Convergence:** Systematic testing of k-points, q-points, supercell size, and displacement magnitude
- **LO-TO Splitting:** Non-analytic corrections for polar materials using Born charges and dielectric tensors
- **Best Practices:** Validation procedures, common pitfalls, and debugging strategies

These methods form the foundation for all advanced phonon physics topics covered in subsequent chapters, including anharmonicity, thermal transport, and phonon engineering.

---

## Navigation

[← Series Index](index.md) | [Chapter 2: Anharmonic Phonons →](chapter-2.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and peer-reviewed literature.
