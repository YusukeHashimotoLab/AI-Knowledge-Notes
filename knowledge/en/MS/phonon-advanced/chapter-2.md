---
title: "Advanced Phonon Physics"
chapter_title: "Chapter 2: Anharmonic Phonons and Phase Transitions"
subtitle: "Self-Consistent Phonon Theory, Soft Modes, and Structural Instabilities"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-advanced/chapter-2.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Advanced Phonon Physics](index.md) > Chapter 2

---

# Chapter 2: Anharmonic Phonons and Phase Transitions

**Self-Consistent Phonon Theory, Soft Modes, and Structural Instabilities**

## Learning Objectives

By completing this chapter, you will be able to:

- Understand limitations of the harmonic approximation at finite temperature
- Derive and implement self-consistent phonon (SCP) theory
- Apply modern anharmonic methods (SCAILD, SSCHA) to real materials
- Calculate temperature-dependent phonon frequencies
- Analyze soft modes and predict structural phase transitions
- Distinguish between displacive and order-disorder transitions
- Use ALAMODE, TDEP, and SSCHA software for anharmonic calculations

## 1. Introduction: Beyond the Harmonic Approximation

The harmonic approximation, where atomic vibrations are treated as independent harmonic oscillators, provides an excellent starting point for understanding lattice dynamics. However, at finite temperatures and for many important physical phenomena, this approximation breaks down.

### 1.1 Failures of the Harmonic Approximation

The harmonic approximation fails to describe:

- **Thermal expansion:** Harmonic crystals have zero thermal expansion
- **Phonon lifetimes:** Harmonic phonons do not decay
- **Temperature-dependent frequencies:** Harmonic phonons have constant frequencies
- **Structural phase transitions:** Cannot capture soft modes and instabilities
- **High-temperature behavior:** Underestimates entropy and heat capacity

**Physical Origin of Anharmonicity**

Anharmonic effects arise from the true potential energy surface deviating from a perfect quadratic form. The Taylor expansion of the potential includes cubic, quartic, and higher-order terms:

\\[
V = V_0 + \sum_i \frac{\partial V}{\partial u_i}u_i + \frac{1}{2}\sum_{ij}\frac{\partial^2 V}{\partial u_i \partial u_j}u_i u_j + \frac{1}{6}\sum_{ijk}\frac{\partial^3 V}{\partial u_i \partial u_j \partial u_k}u_i u_j u_k + \cdots
\\]

where \\(u_i\\) represents atomic displacements. The cubic and quartic terms are responsible for anharmonic phenomena.

### 1.2 Temperature Effects on Phonons

At finite temperature, several phenomena emerge that require anharmonic treatment. Temperature increase leads to larger atomic displacements, causing atoms to explore anharmonic regions of the potential energy surface. This results in:

- Phonon frequency shifts
- Phonon-phonon interactions
- Thermal expansion
- Soft mode condensation and structural phase transitions
- Phonon linewidths affecting thermal conductivity

The key insight is that thermal motion causes atoms to explore anharmonic regions of the potential energy surface, leading to temperature-dependent effective force constants.

## 2. Anharmonic Effects at Finite Temperature

### 2.1 Quasi-Harmonic Approximation (QHA)

The simplest extension beyond the harmonic approximation is the quasi-harmonic approximation, which assumes:

- Phonon frequencies depend on volume: \\(\omega_{\mathbf{q}\nu}(V)\\)
- At each volume, the system is harmonic
- Temperature dependence enters through thermal expansion

**QHA Free Energy**

\\[
F(V, T) = U_0(V) + F_{\text{vib}}(V, T)
\\]

\\[
F_{\text{vib}}(V, T) = k_B T \sum_{\mathbf{q}\nu} \ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}\nu}(V)}{2k_B T}\right)\right]
\\]

The equilibrium volume at temperature \\(T\\) minimizes \\(F(V, T)\\):

\\[
\left.\frac{\partial F}{\partial V}\right|_{V_{\text{eq}}(T)} = 0
\\]

### 2.2 Intrinsic Anharmonicity

The QHA captures volume-dependent phonon frequencies but misses intrinsic anharmonic effects that occur even at constant volume. These include phonon-phonon scattering, temperature-dependent frequencies at constant volume, soft mode hardening/softening, and contributions to negative thermal expansion.

### 2.3 Phonon-Phonon Interactions

Cubic anharmonicity leads to three-phonon processes where one phonon can decay into two phonons or two phonons can merge into one.

**Three-Phonon Scattering Rate**

\\[
\Gamma_{\mathbf{q}\nu} = \frac{2\pi}{\hbar} \sum_{\mathbf{q}'\nu'\nu''} |V_3(\mathbf{q}\nu, \mathbf{q}'\nu', \mathbf{q}''\nu'')|^2 \times \left[(n_{\nu'}+n_{\nu''}+1)\delta(\omega-\omega'-\omega'') + (n_{\nu'}-n_{\nu''})\delta(\omega-\omega'+\omega'')\right]
\\]

where \\(V_3\\) is the cubic anharmonic coupling and \\(n_{\nu} = 1/(e^{\hbar\omega/k_B T}-1)\\) is the Bose-Einstein distribution.

The phonon linewidth is \\(\Gamma_{\mathbf{q}\nu}\\), and the phonon lifetime is \\(\tau_{\mathbf{q}\nu} = 1/\Gamma_{\mathbf{q}\nu}\\). These quantities are crucial for thermal conductivity.

## Summary

This chapter explored anharmonic phonon physics and structural phase transitions:

- **Harmonic Limitations:** Zero thermal expansion, infinite phonon lifetimes, missing phase transitions
- **QHA:** Volume-dependent phonons capture thermal expansion but miss intrinsic anharmonicity
- **Self-Consistent Phonon Theory:** Renormalized force constants from thermal averaging
- **Modern Methods:** SCAILD, SSCHA, TDEP for first-principles anharmonic calculations
- **Soft Modes:** Phonon frequencies approaching zero signal structural instabilities
- **Phase Transitions:** Landau theory with order parameter, second-order vs first-order transitions
- **Perovskites:** BaTiO₃ as prototypical ferroelectric, SrTiO₃ as quantum paraelectric
- **Negative Thermal Expansion:** ZrW₂O₈ and the role of negative Grüneisen parameters

Understanding anharmonic effects is essential for predicting thermal properties, phase stability, and designing materials with targeted thermal behavior.

---

## Navigation

[← Chapter 1: First-Principles Calculations](chapter-1.md) | [Chapter 3: Thermal Transport →](chapter-3.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and peer-reviewed literature.
