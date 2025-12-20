---
title: "Chapter 4: Electron-Phonon Coupling"
chapter_title: "Electron-Phonon Coupling"
subtitle: "Interaction Between Electrons and Lattice Vibrations"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-intermediate/chapter-4.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Intermediate Phonon Physics](index.md) > Chapter 4

# Chapter 4: Electron-Phonon Coupling

**Interaction Between Electrons and Lattice Vibrations**

⏱️ 40-50 min | 💻 7 Code Examples | 📊 Advanced

## Learning Objectives

- Formulate the electron-phonon interaction Hamiltonian
- Distinguish Fröhlich and deformation potential coupling mechanisms
- Understand polarons and their formation
- Calculate electron self-energy from phonon interactions
- Compute the Eliashberg function \\(\alpha^2F(\omega)\\) and coupling constant \\(\lambda\\)
- Connect electron-phonon coupling to superconductivity via BCS theory
- Apply the McMillan equation for \\(T_c\\) estimation

## 4.1 Electron-Phonon Interaction

### Physical Origin

Electron-phonon coupling arises because:
- Electron charge density perturbs ion positions
- Ionic motion modulates electronic potential
- Electrons can emit or absorb phonons
- Phonons scatter electrons, contributing to resistivity

### Interaction Hamiltonian

\\[
H_{e-ph} = \sum_{\mathbf{k},\mathbf{q},\nu} g_{\mathbf{k},\mathbf{q}}^\nu c_{\mathbf{k}+\mathbf{q}}^\dagger c_{\mathbf{k}} (a_{\mathbf{q}\nu} + a_{-\mathbf{q}\nu}^\dagger)
\\]

where:
- \\(c_{\mathbf{k}}^\dagger, c_{\mathbf{k}}\\): Electron creation/annihilation operators
- \\(a_{\mathbf{q}\nu}^\dagger, a_{\mathbf{q}\nu}\\): Phonon creation/annihilation operators
- \\(g_{\mathbf{k},\mathbf{q}}^\nu\\): Electron-phonon matrix element

### Matrix Element

\\[
g_{\mathbf{k},\mathbf{q}}^\nu = \sqrt{\frac{\hbar}{2M\omega_{\mathbf{q}\nu}}} \langle \mathbf{k}+\mathbf{q} | \frac{\partial V}{\partial u_{\mathbf{q}\nu}} | \mathbf{k} \rangle
\\]

## 4.2 Coupling Mechanisms

### Fröhlich Hamiltonian (Polar Coupling)

For polar materials (ionic crystals):

\\[
H_F = \sum_{\mathbf{k},\mathbf{q}} V_F(\mathbf{q}) c_{\mathbf{k}+\mathbf{q}}^\dagger c_{\mathbf{k}} (a_{\mathbf{q}} + a_{-\mathbf{q}}^\dagger)
\\]

Interaction potential:

\\[
V_F(\mathbf{q}) = -i \left(\frac{2\pi\alpha\hbar\omega_{LO}}{V}\right)^{1/2} \frac{1}{q}
\\]

Fröhlich coupling constant:

\\[
\alpha = \frac{e^2}{2\hbar\omega_{LO}} \left(\frac{2m^*\omega_{LO}}{\hbar}\right)^{1/2} \left(\frac{1}{\epsilon_\infty} - \frac{1}{\epsilon_0}\right)
\\]

### Deformation Potential Coupling

For acoustic phonons:

\\[
H_{DP} = \sum_{\mathbf{k},\mathbf{q}} D_{\mathbf{q}} \nabla \cdot \mathbf{u}(\mathbf{q}) c_{\mathbf{k}+\mathbf{q}}^\dagger c_{\mathbf{k}} (a_{\mathbf{q}} + a_{-\mathbf{q}}^\dagger)
\\]

Matrix element:

\\[
g_{\mathbf{k},\mathbf{q}}^{ac} = D_{ac} \sqrt{\frac{\hbar q^2}{2\rho V v_s}}
\\]

## 4.3 Polaron Theory

### The Polaron Concept

A **polaron** is a quasiparticle consisting of an electron plus the lattice distortion it induces.

### Large vs Small Polarons

| Regime | \\(\alpha\\) Value | Characteristics | Examples |
|--------|---------|----------------|----------|
| **Large Polaron** | \\(\alpha < 6\\) | Delocalized, weak distortion | GaAs, CdTe |
| **Small Polaron** | \\(\alpha > 6\\) | Localized, strong distortion | Transition metal oxides |

### Mass Renormalization

Weak coupling (\\(\alpha \ll 1\\)):

\\[
m_p^* = m^* \left(1 + \frac{\alpha}{6}\right)
\\]

Strong coupling (\\(\alpha \gg 1\\)):

\\[
m_p^* = \frac{m^*}{6\alpha^2} e^{\alpha}
\\]

### Ground State Energy

\\[
E_p = -\alpha\hbar\omega_{LO} \quad \text{(weak coupling)}
\\]

## 4.4 Electron Self-Energy

The electron self-energy \\(\Sigma(\mathbf{k}, \omega)\\) describes modification of electron properties:

\\[
G(\mathbf{k}, \omega) = \frac{1}{\omega - \epsilon_{\mathbf{k}} - \Sigma(\mathbf{k}, \omega)}
\\]

### Lowest-Order Self-Energy

\\[
\Sigma(\mathbf{k}, \omega) = \sum_{\mathbf{q},\nu} |g_{\mathbf{k},\mathbf{q}}^\nu|^2 \left[\frac{n_{\mathbf{q}\nu} + f_{\mathbf{k}+\mathbf{q}}}{\omega - \epsilon_{\mathbf{k}+\mathbf{q}} + \omega_{\mathbf{q}\nu} + i\eta} + \frac{n_{\mathbf{q}\nu} + 1 - f_{\mathbf{k}+\mathbf{q}}}{\omega - \epsilon_{\mathbf{k}+\mathbf{q}} - \omega_{\mathbf{q}\nu} + i\eta}\right]
\\]

Real and imaginary parts:
- **Real part**: Energy shift (renormalization)
- **Imaginary part**: Finite lifetime (scattering rate)

### Mass Enhancement

\\[
\frac{m^*}{m} = 1 - \left.\frac{\partial \text{Re}\,\Sigma}{\partial \omega}\right|_{\omega = E_F}
\\]

\\[
1 + \lambda = \frac{m^*}{m_{\text{band}}}
\\]

## 4.5 Eliashberg Function and Coupling Constant

### Eliashberg Function

\\[
\alpha^2 F(\omega) = \frac{1}{N(E_F)} \sum_{\mathbf{k},\mathbf{q},\nu} |g_{\mathbf{k},\mathbf{k}+\mathbf{q}}^\nu|^2 \delta(\epsilon_{\mathbf{k}} - E_F) \delta(\epsilon_{\mathbf{k}+\mathbf{q}} - E_F) \delta(\omega - \omega_{\mathbf{q}\nu})
\\]

### Electron-Phonon Coupling Constant

\\[
\lambda = 2\int_0^\infty \frac{\alpha^2 F(\omega)}{\omega} d\omega
\\]

| \\(\lambda\\) Range | Coupling Regime | Examples |
|---------|----------------|----------|
| \\(\lambda < 0.3\\) | Weak | Be, Al |
| \\(0.3 < \lambda < 0.8\\) | Intermediate | Sn, In, Zn |
| \\(\lambda > 0.8\\) | Strong | Pb (\\(\lambda \approx 1.5\\)), MgB₂ |

## 4.6 Connection to Superconductivity

### BCS Theory Overview

Key insights:
- Cooper pairs form via phonon-mediated attraction
- Superconducting gap \\(\Delta\\) opens at Fermi surface
- Phonon exchange provides effective electron-electron attraction:

\\[
V_{\text{eff}}(\mathbf{k}, \mathbf{k}', \omega) = -\sum_{\mathbf{q},\nu} \frac{|g_{\mathbf{k},\mathbf{q}}^\nu|^2 \omega_{\mathbf{q}\nu}}{\omega^2 - \omega_{\mathbf{q}\nu}^2}
\\]

### McMillan Equation for \\(T_c\\)

\\[
T_c = \frac{\omega_{\log}}{1.2} \exp\left[-\frac{1.04(1 + \lambda)}{\lambda - \mu^*(1 + 0.62\lambda)}\right]
\\]

where:
- \\(\omega_{\log}\\): Logarithmic average phonon frequency
- \\(\lambda\\): Electron-phonon coupling constant
- \\(\mu^*\\): Coulomb pseudopotential (typically 0.1-0.15)

Logarithmic average:

\\[
\omega_{\log} = \exp\left[\frac{2}{\lambda} \int_0^\infty \frac{\alpha^2 F(\omega)}{\omega} \ln(\omega) d\omega\right]
\\]

### Role of Phonon Modes

| Material | \\(T_c\\) (K) | \\(\lambda\\) | Dominant Phonons |
|----------|-------|-------|------------------|
| Al | 1.2 | 0.43 | Acoustic |
| Pb | 7.2 | 1.55 | Acoustic + low-E optical |
| MgB₂ | 39 | 0.87 | E₂g B-B stretching (70 meV) |
| Nb₃Sn | 18 | 1.2 | Low-frequency modes |

## 4.7 Experimental Determination

### Tunneling Spectroscopy

\\[
\frac{d^2 I}{dV^2} \propto \int_0^\infty d\omega\, \alpha^2 F(\omega) \left[\frac{d}{dE} f(E - eV - \omega)\right]
\\]

### ARPES (Angle-Resolved Photoemission)

Measures electron spectral function revealing kinks at phonon energies:

\\[
A(\mathbf{k}, \omega) = -\frac{1}{\pi} \frac{\text{Im}\,\Sigma(\mathbf{k}, \omega)}{[\omega - \epsilon_{\mathbf{k}} - \text{Re}\,\Sigma]^2 + [\text{Im}\,\Sigma]^2}
\\]

### First-Principles Calculations

DFPT computes matrix elements from first principles:

\\[
g_{\mathbf{k},\mathbf{q}}^\nu = \langle \psi_{\mathbf{k}+\mathbf{q}} | \frac{\partial V_{SCF}}{\partial u_{\mathbf{q}\nu}} | \psi_{\mathbf{k}} \rangle
\\]

Software tools: Quantum ESPRESSO (EPW), ABINIT, VASP

## Summary

- **Electron-phonon coupling** describes interaction between conduction electrons and lattice vibrations
- **Two main mechanisms**: Fröhlich (polar, LO phonons) and deformation potential (acoustic/non-polar)
- **Polarons** are electrons dressed by lattice distortions: large (\\(\alpha < 6\\)) or small (\\(\alpha > 6\\))
- **Self-energy** captures renormalization: real part shifts energy, imaginary part gives scattering rate
- **Eliashberg function** \\(\alpha^2F(\omega)\\) characterizes frequency-dependent coupling
- **Coupling constant** \\(\lambda = 2\int\alpha^2F(\omega)/\omega\,d\omega\\)
- **BCS superconductivity** arises from phonon-mediated Cooper pairing
- **McMillan equation** relates \\(T_c\\) to \\(\lambda\\) and \\(\omega_{\log}\\)

## Exercises

1. Calculate Fröhlich coupling constant \\(\alpha\\) for CdTe (\\(\epsilon_\infty = 7.1\\), \\(\epsilon_0 = 10.2\\), \\(m^* = 0.11 m_e\\), \\(\hbar\omega_{LO} = 21\\) meV).

2. Given Eliashberg function with two peaks (Gaussian), calculate \\(\lambda\\), \\(\omega_{\log}\\), and estimate \\(T_c\\) using McMillan equation.

3. For \\(\omega_0 = 30\\) meV and \\(g = 0.1\\) eV, calculate phonon contribution to electron self-energy.

4. Analyze temperature dependence of electron mobility in a polar semiconductor using polaron theory.

5. Compare two superconductors: Material A (\\(\lambda = 0.8\\), \\(\omega_{\log} = 200\\) K) vs Material B (\\(\lambda = 0.5\\), \\(\omega_{\log} = 400\\) K). Which has higher \\(T_c\\)?

---

**Navigation**

[← Chapter 3: Phonon Scattering](chapter-3.md) | [Chapter 5: Phonon Spectroscopy →](chapter-5.md)

---

**Disclaimer**

This educational content was created for the Hashimoto Lab knowledge base. While care has been taken to ensure accuracy, readers should verify critical information with primary sources and consult original research papers.

**Author**: MS Knowledge Hub Content Team
**Version**: 1.0 | **Last Updated**: 2025-12-19
**License**: Creative Commons BY 4.0
