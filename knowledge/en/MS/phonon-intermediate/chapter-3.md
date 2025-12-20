---
title: "Chapter 3: Phonon Scattering"
chapter_title: "Phonon Scattering"
subtitle: "Thermal Conductivity and Phonon Transport"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-intermediate/chapter-3.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Intermediate Phonon Physics](index.md) > Chapter 3

# Chapter 3: Phonon Scattering

**Thermal Conductivity and Phonon Transport**

⏱️ 35-45 min | 💻 6 Code Examples | 📊 Intermediate-Advanced

## Learning Objectives

- Understand phonon heat current and thermal conductivity tensor
- Derive thermal conductivity using the Boltzmann transport equation (BTE)
- Master the relaxation time approximation (RTA) and its validity
- Learn scattering mechanisms: phonon-phonon, boundary, impurity, isotope
- Apply Matthiessen's rule and the Callaway model
- Analyze mean free path distributions and ballistic transport
- Calculate thermal conductivity with realistic scattering models

## 3.1 Phonon Heat Current

### Heat Current Density

The heat current density carried by phonons is:

\\[
\mathbf{J}_q = \sum_{\mathbf{k},s} \hbar\omega_{\mathbf{k}s} \, \mathbf{v}_{\mathbf{k}s} \, n_{\mathbf{k}s}
\\]

where \\(\mathbf{v}_{\mathbf{k}s} = \nabla_{\mathbf{k}}\omega_{\mathbf{k}s}\\) is the group velocity and \\(n_{\mathbf{k}s}\\) is the phonon occupation number.

### Fourier's Law

For small temperature gradients:

\\[
\mathbf{J}_q = -\boldsymbol{\kappa} \cdot \nabla T
\\]

For isotropic materials: \\(\mathbf{J}_q = -\kappa \nabla T\\)

## 3.2 Boltzmann Transport Equation

The phonon distribution evolves according to:

\\[
\frac{\partial n_{\mathbf{k}s}}{\partial t} + \mathbf{v}_{\mathbf{k}s} \cdot \nabla_{\mathbf{r}} n_{\mathbf{k}s} = \left(\frac{\partial n_{\mathbf{k}s}}{\partial t}\right)_{\text{scatt}}
\\]

### Relaxation Time Approximation

\\[
\left(\frac{\partial n_{\mathbf{k}s}}{\partial t}\right)_{\text{scatt}} = -\frac{n_{\mathbf{k}s} - n_{\mathbf{k}s}^0}{\tau_{\mathbf{k}s}}
\\]

where \\(\tau_{\mathbf{k}s}\\) is the relaxation time representing the average time between scattering events.

### Thermal Conductivity Formula

Solving the BTE in the RTA gives:

\\[
\kappa = \frac{1}{V} \sum_{\mathbf{k},s} C_{\mathbf{k}s} \, v_{\mathbf{k}s}^2 \, \tau_{\mathbf{k}s}
\\]

where \\(C_{\mathbf{k}s}\\) is the mode-specific heat capacity.

## 3.3 Kinetic Theory Formula

For a simple Debye model:

\\[
\kappa = \frac{1}{3} C_V v_s^2 \bar{\tau} = \frac{1}{3} C_V v_s \ell
\\]

where \\(\ell = v_s \bar{\tau}\\) is the mean free path.

## 3.4 Scattering Mechanisms

### Matthiessen's Rule

\\[
\frac{1}{\tau_{\text{total}}} = \sum_i \frac{1}{\tau_i} = \frac{1}{\tau_U} + \frac{1}{\tau_N} + \frac{1}{\tau_B} + \frac{1}{\tau_I} + \frac{1}{\tau_{\text{iso}}}
\\]

### Umklapp Scattering

Three-phonon umklapp processes:

\\[
\frac{1}{\tau_U} = B\omega^2 T e^{-\Theta_D/bT}
\\]

At high temperature (\\(T \gg \Theta_D\\)): \\(\tau_U^{-1} \propto \omega^2 T\\)

### Boundary Scattering

\\[
\tau_B = \frac{L}{v}, \quad \ell_B = L
\\]

where \\(L\\) is the characteristic sample size.

### Impurity and Isotope Scattering

Rayleigh scattering (\\(\omega^4\\) dependence):

\\[
\frac{1}{\tau_I} = A\omega^4
\\]

Isotope scattering parameter:

\\[
\Gamma_{\text{iso}} = \sum_i c_i \left(1 - \frac{M_i}{\bar{M}}\right)^2
\\]

## 3.5 The Callaway Model

Separates resistive and non-resistive scattering:

\\[
\kappa = \kappa_1 + \kappa_2
\\]

where:
- \\(\kappa_1\\): RTA-like contribution with combined relaxation time
- \\(\kappa_2\\): Normal process correction (momentum-conserving scattering)

## 3.6 Mean Free Path Distribution

The accumulation function shows cumulative thermal conductivity:

\\[
\kappa_{\text{acc}}(\Lambda) = \int_0^{\Lambda} \frac{d\kappa}{d\ell} d\ell
\\]

Typical findings:
- 50% of heat carried by phonons with \\(\ell < 100\\) nm at room temperature
- Some phonons have \\(\ell > 10\\) μm
- Nanostructuring blocks long-MFP phonons

## 3.7 Transport Regimes

**Diffusive regime** (\\(L \gg \ell\\)):
- Many scattering events
- Fourier's law valid
- \\(\kappa\\) independent of \\(L\\)

**Ballistic regime** (\\(L \ll \ell\\)):
- Phonons cross sample without scattering
- \\(\kappa_{\text{eff}} \propto L\\)

## 3.8 Minimum Thermal Conductivity

Cahill-Pohl formula for amorphous limit:

\\[
\kappa_{\min} \approx 0.4 \, k_B n^{2/3} \sum_i v_i
\\]

## Summary

- **Phonon heat current**: \\(\mathbf{J}_q = \sum_{\mathbf{k}s} \hbar\omega_{\mathbf{k}s} \mathbf{v}_{\mathbf{k}s} n_{\mathbf{k}s}\\)
- **BTE** governs phonon distribution under gradients and scattering
- **RTA** simplifies collision term: \\(-(n-n^0)/\tau\\)
- **Kinetic formula**: \\(\kappa = \frac{1}{3}C_V v \ell\\)
- **Scattering mechanisms**: Umklapp (\\(\omega^2 T\\)), boundary (\\(v/L\\)), impurity/isotope (\\(\omega^4\\))
- **Matthiessen's rule**: \\(1/\tau_{\text{tot}} = \sum_i 1/\tau_i\\)
- **Callaway model** separates normal and resistive scattering
- **Nanostructuring** reduces \\(\kappa\\) via boundary scattering

## Exercises

1. For \\(C_V = 2 \times 10^6\\) J/m³·K, \\(v_s = 5000\\) m/s, \\(\tau = 10^{-11}\\) s, calculate \\(\kappa\\) and mean free path.

2. Calculate isotope scattering parameter for natural Si (92.2% ²⁸Si, 4.7% ²⁹Si, 3.1% ³⁰Si).

3. Estimate thermal conductivity reduction for a 100 nm thin film with bulk \\(\ell = 1\\) μm.

4. Combine scattering times using Matthiessen's rule: \\(\tau_U = 5 \times 10^{-12}\\) s, \\(\tau_I = 2 \times 10^{-11}\\) s, \\(\tau_B = 1 \times 10^{-10}\\) s.

5. Write Python code to calculate thermal conductivity vs. temperature including all scattering mechanisms.

---

**Navigation**

[← Chapter 2: Anharmonic Effects](chapter-2.md) | [Chapter 4: Electron-Phonon Coupling →](chapter-4.md)

---

**Disclaimer**

This educational content was created for the Hashimoto Lab knowledge base. While care has been taken to ensure accuracy, readers should verify critical information with primary sources and consult original research papers.

**Author**: MS Knowledge Hub Content Team
**Version**: 1.0 | **Last Updated**: 2025-12-19
**License**: Creative Commons BY 4.0
