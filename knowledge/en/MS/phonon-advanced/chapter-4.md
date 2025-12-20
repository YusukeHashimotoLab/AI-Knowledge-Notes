---
title: "Advanced Phonon Physics"
chapter_title: "Chapter 4: Anharmonicity and Thermal Expansion"
subtitle: "Grüneisen Parameters, Thermal Expansion, and Experimental Techniques"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-advanced/chapter-4.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Advanced Phonon Physics](index.md) > Chapter 4

---

# Chapter 4: Anharmonicity and Thermal Expansion

**Grüneisen Parameters, Thermal Expansion, and Experimental Techniques**

## Learning Objectives

- Understand the connection between anharmonicity and thermal expansion
- Calculate and interpret Grüneisen parameters
- Apply quasi-harmonic approximation (QHA) to predict thermal properties
- Analyze negative thermal expansion (NTE) materials
- Master experimental techniques for measuring phonon properties
- Implement computational methods for thermal expansion calculations
- Understand isotope effects on phonon properties
- Apply temperature-dependent effective potential methods

## 1. Introduction

Thermal expansion is one of the most fundamental consequences of anharmonic lattice vibrations. While the harmonic approximation provides an excellent framework for understanding phonon dispersion at zero temperature, it predicts zero thermal expansion—clearly contradicting experimental observations. This chapter explores how anharmonic effects, quantified through Grüneisen parameters, lead to thermal expansion and other temperature-dependent phenomena.

## 2. Theoretical Framework

### 2.1 Harmonic vs Anharmonic Potentials

The key to understanding thermal expansion lies in the asymmetry of the interatomic potential. A purely harmonic potential \\(V(r) \propto (r-r_0)^2\\) is symmetric about the equilibrium position \\(r_0\\), leading to zero net displacement when thermally populated. Real potentials, however, are anharmonic with asymmetric shapes that cause the time-averaged position to shift with temperature.

### 2.2 Grüneisen Parameters

The mode Grüneisen parameter \\(\gamma_{\mathbf{q}\nu}\\) quantifies how phonon frequency changes with volume:

\\[
\gamma_{\mathbf{q}\nu} = -\frac{V}{\omega_{\mathbf{q}\nu}}\frac{\partial \omega_{\mathbf{q}\nu}}{\partial V}
\\]

Physical interpretation:
- \\(\gamma > 0\\): Frequency decreases with volume expansion (most common)
- \\(\gamma < 0\\): Frequency increases with volume expansion (can lead to NTE)
- \\(\gamma \approx 0\\): Frequency insensitive to volume

The bulk Grüneisen parameter, averaged over all modes, determines thermal expansion:

\\[
\gamma = \frac{\sum_{\mathbf{q}\nu} \gamma_{\mathbf{q}\nu} C_{\mathbf{q}\nu}}{\sum_{\mathbf{q}\nu} C_{\mathbf{q}\nu}}
\\]

where \\(C_{\mathbf{q}\nu}\\) is the mode-specific heat capacity.

### 2.3 Thermal Expansion Coefficient

The volumetric thermal expansion coefficient is:

\\[
\alpha_V = \frac{1}{V}\frac{dV}{dT} = \frac{\gamma C_V}{BV}
\\]

where \\(B\\) is the bulk modulus and \\(C_V\\) is the heat capacity at constant volume.

## 3. Quasi-Harmonic Approximation (QHA)

### 3.1 QHA Theory

The quasi-harmonic approximation assumes that at each volume \\(V\\), the system can be described by harmonic phonons with volume-dependent frequencies \\(\omega_{\mathbf{q}\nu}(V)\\).

The Helmholtz free energy is:

\\[
F(V,T) = U_0(V) + F_{\text{vib}}(V,T)
\\]

where the vibrational free energy is:

\\[
F_{\text{vib}}(V,T) = k_BT\sum_{\mathbf{q}\nu}\ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}\nu}(V)}{2k_BT}\right)\right]
\\]

The equilibrium volume at temperature \\(T\\) minimizes the free energy:

\\[
\left.\frac{\partial F}{\partial V}\right|_{V_{\text{eq}}(T)} = 0
\\]

### 3.2 QHA Workflow

The computational procedure for QHA calculations:

1. Calculate total energy \\(E(V)\\) for multiple volumes
2. For each volume, compute phonon dispersion \\(\omega_{\mathbf{q}\nu}(V)\\)
3. Calculate \\(F(V,T)\\) for desired temperatures
4. Find \\(V_{\text{eq}}(T)\\) by minimizing \\(F(V,T)\\)
5. Compute thermal expansion: \\(\alpha(T) = \frac{1}{V}\frac{dV}{dT}\\)
6. Calculate other properties: \\(C_V(T)\\), \\(C_P(T)\\), \\(B(T)\\)

### 3.3 Limitations of QHA

The QHA fails when:
- Strong intrinsic anharmonicity exists (cannot be captured by volume dependence alone)
- Near structural phase transitions (soft modes require beyond-QHA methods)
- At very high temperatures (classical limit breakdown)
- For materials with strong phonon-phonon interactions

## 4. Negative Thermal Expansion (NTE)

### 4.1 Mechanisms of NTE

Negative thermal expansion occurs when \\(\alpha_V < 0\\), meaning the material contracts upon heating. This counterintuitive behavior arises from:

- Transverse acoustic modes with negative Grüneisen parameters
- Low-frequency librational or rotational modes
- Framework structures with geometric constraints

### 4.2 Famous NTE Materials

**ZrW₂O₈:** Exhibits isotropic NTE over a wide temperature range (0.3-1050 K) with \\(\alpha_V \approx -9 \times 10^{-6}\\) K⁻¹.

**ScF₃:** Cubic structure with strong NTE due to rigid unit modes.

**MOFs (Metal-Organic Frameworks):** Some show giant NTE due to flexible framework dynamics.

## 5. Experimental Techniques

### 5.1 Inelastic Neutron Scattering (INS)

INS is the gold standard for measuring phonon dispersion across the entire Brillouin zone. Energy and momentum conservation give:

\\[
\hbar\omega = E_i - E_f
\\]
\\[
\mathbf{q} = \mathbf{k}_i - \mathbf{k}_f \pm \mathbf{G}
\\]

### 5.2 Raman and Infrared Spectroscopy

Optical spectroscopies probe zone-center phonons (\\(\mathbf{q} \approx 0\\)):

- **Raman:** Detects phonons coupling to polarizability (selection rules depend on symmetry)
- **IR:** Detects phonons coupling to dipole moment (only IR-active modes)

Temperature-dependent Raman spectroscopy can track soft modes approaching phase transitions.

### 5.3 X-ray and Neutron Diffraction

Diffraction techniques measure:
- Lattice parameters as a function of temperature (thermal expansion)
- Atomic displacement parameters (Debye-Waller factors)
- Structural phase transitions

## Summary

This chapter explored the fundamental connection between anharmonicity and thermal properties:

- **Grüneisen Parameters:** Quantify volume dependence of phonon frequencies
- **QHA:** Practical approximation for thermal expansion and free energy
- **NTE Materials:** Negative Grüneisen modes lead to material contraction on heating
- **Experimental Methods:** INS, Raman, IR, and diffraction probe phonons
- **Computational Approaches:** QHA workflow and beyond-QHA methods
- **Applications:** Thermal management, precision engineering, phase diagram prediction

Understanding these concepts enables prediction and control of thermal properties for technological applications.

---

## Navigation

[← Chapter 3: Thermal Transport](chapter-3.md) | [Chapter 5: Phonon Engineering →](chapter-5.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and peer-reviewed literature.
