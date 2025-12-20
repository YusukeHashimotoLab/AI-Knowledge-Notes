---
title: "Chapter 5: Phonon Spectroscopy"
chapter_title: "Phonon Spectroscopy"
subtitle: "Experimental Techniques for Measuring Phonon Dispersion and Dynamics"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-intermediate/chapter-5.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Intermediate Phonon Physics](index.md) > Chapter 5

# Chapter 5: Phonon Spectroscopy

**Experimental Techniques for Measuring Phonon Dispersion and Dynamics**

⏱️ 35-45 min | 💻 6 Code Examples | 📊 Intermediate

## Learning Objectives

- Understand quantum theory of inelastic neutron scattering
- Explain operation of triple-axis and time-of-flight spectrometers
- Apply selection rules for Raman and infrared spectroscopy
- Understand polariton dispersion in polar materials
- Describe EELS, IXS, and time-resolved pump-probe methods
- Compare complementary capabilities of different techniques

## 5.1 Inelastic Neutron Scattering (INS)

### Why Neutrons?

- Wavelength: \\(\lambda \approx 1-4\\) Å (comparable to interatomic spacing)
- Energy: \\(E \approx 5-100\\) meV (comparable to phonon energies)
- Direct momentum and energy conservation measurement
- Penetrating (bulk probe, not surface-sensitive)

### Scattering Theory

Energy and momentum conservation:

\\[
\mathbf{Q} = \mathbf{k}_i - \mathbf{k}_f, \quad \hbar\omega = E_i - E_f = \frac{\hbar^2}{2m_n}(k_i^2 - k_f^2)
\\]

For phonon creation:

\\[
\mathbf{Q} = \mathbf{q} + \mathbf{G}, \quad \hbar\omega = \hbar\omega_{\mathbf{q},s}
\\]

### Differential Cross-Section

One-phonon scattering:

\\[
\frac{d^2\sigma}{d\Omega dE_f} = \frac{k_f}{k_i} \frac{(2\pi)^3}{v_0} \sum_{\mathbf{G}} \sum_{s} |F(\mathbf{Q})|^2 \frac{(\hbar Q)^2}{2M\omega_{\mathbf{q},s}} (n_{\mathbf{q},s} + \frac{1}{2} \pm \frac{1}{2}) \delta(\omega - \omega_{\mathbf{q},s})
\\]

### Coherent vs Incoherent Scattering

| Type | Origin | Information | Examples |
|------|--------|-------------|----------|
| **Coherent** | Interference between nuclei | Phonon dispersion \\(\omega(\mathbf{q})\\) | V, Ti, Si |
| **Incoherent** | Random nuclear spin/isotopes | Phonon DOS \\(g(\omega)\\) | H, Natural Ni |

### Instrumentation

**Triple-Axis Spectrometer (TAS)**:
- Monochromator selects \\(\mathbf{k}_i\\)
- Sample oriented for \\(\mathbf{Q}\\)
- Analyzer selects \\(\mathbf{k}_f\\)
- Point-by-point mapping of \\(S(\mathbf{Q}, \omega)\\)
- High resolution (~0.1 meV), low count rate

**Time-of-Flight (TOF)**:
- Energy determined by travel time: \\(E = m_n L^2/(2t^2)\\)
- Simultaneous broad energy range
- Higher count rate, lower resolution
- Good for powder samples

## 5.2 Raman Scattering

### Quantum Theory

Inelastic photon scattering by phonons:

\\[
\hbar\omega_s = \hbar\omega_i \mp \hbar\omega_{\mathbf{q}}
\\]

Intensity proportional to Raman tensor:

\\[
I_{\text{Raman}} \propto \left|\mathbf{e}_s \cdot \frac{\partial \alpha}{\partial u} \cdot \mathbf{e}_i\right|^2 (n_{\mathbf{q}} + 1)
\\]

### Selection Rules

Raman-active if polarizability changes:

\\[
\frac{\partial \alpha_{ij}}{\partial Q_{\mathbf{q},s}} \neq 0
\\]

- **Centrosymmetric crystals**: Raman and IR mutually exclusive
- **Wavevector restriction**: \\(\mathbf{q} \approx 0\\) (zone-center only)

### First-Order vs Second-Order

| Order | Process | Information |
|-------|---------|-------------|
| **First-order** | One phonon | Zone-center frequencies, symmetry |
| **Second-order** | Two phonons | Phonon DOS, anharmonicity |

Second-order processes:
- **Overtones**: \\(\mathbf{q}_1 = \mathbf{q}_2\\)
- **Combinations**: \\(\mathbf{q}_1 + \mathbf{q}_2 = 0\\)
- Much weaker (~1% intensity)

## 5.3 Infrared Absorption

### Selection Rules

IR-active if dipole moment changes:

\\[
\frac{\partial \mathbf{P}}{\partial Q_{\mathbf{q},s}} \neq 0
\\]

Only **polar modes** (net dipole oscillation) are IR-active.

### Polariton Dispersion

In polar crystals, IR phonons couple to electromagnetic waves forming **phonon-polaritons**.

**Lyddane-Sachs-Teller relation**:

\\[
\frac{\omega_{LO}^2}{\omega_{TO}^2} = \frac{\epsilon_0}{\epsilon_\infty}
\\]

**Polariton branches**:
- Upper branch: photon-like at low \\(q\\), asymptotes to \\(\omega_{LO}\\)
- Lower branch: starts from \\(\omega_{TO}\\), becomes photon-like
- **Stop band** between \\(\omega_{TO}\\) and \\(\omega_{LO}\\): no propagating modes

### Reflectivity

\\[
R(\omega) = \left|\frac{\sqrt{\epsilon(\omega)} - 1}{\sqrt{\epsilon(\omega)} + 1}\right|^2
\\]

Maximum reflectivity in stop band (\\(\epsilon < 0\\)).

## 5.4 Complementary Techniques

### Electron Energy Loss Spectroscopy (EELS)

- **Probe**: High-energy electrons (50-300 keV)
- **Spatial resolution**: Atomic scale (STEM-EELS)
- **Energy resolution**: 10-100 meV (worse than optical)
- **Applications**: Nanostructures, interfaces, local modes

### Inelastic X-ray Scattering (IXS)

- **Probe**: High-energy X-rays (~10-20 keV)
- **Momentum resolution**: High (full Brillouin zone)
- **Energy resolution**: ~1.5 meV
- **Advantages**: Light elements, high pressure, small samples
- **Disadvantages**: Weaker signal (synchrotron required)

### Time-Resolved Pump-Probe

- **Probe**: Ultrafast laser pulses (femtosecond)
- **Time resolution**: ~10 fs
- **Observables**: Coherent phonon oscillations, phonon lifetimes
- **Applications**: Ultrafast dynamics, phase transitions, phonon-electron coupling

## 5.5 Comparison of Techniques

| Technique | Energy Resolution | \\(q\\)-range | Information | Best For |
|-----------|------------------|----------|-------------|----------|
| **INS (TAS)** | ~0.1 meV | Full BZ | Phonon dispersion | Complete dispersion |
| **INS (TOF)** | ~1 meV | Broad | Phonon DOS | Powder samples |
| **Raman** | ~1 cm⁻¹ | \\(q \approx 0\\) | Zone-center modes | Fast, micron resolution |
| **Infrared** | ~1 cm⁻¹ | \\(q \approx 0\\) | IR-active modes, polaritons | Polar materials |
| **EELS** | 10-100 meV | Moderate | Spatial mapping | Nanostructures |
| **IXS** | ~1.5 meV | Full BZ | Light elements | High pressure |
| **Pump-Probe** | ~10 fs (time) | \\(q \approx 0\\) | Phonon dynamics | Ultrafast processes |

### Complementary Strategy

1. **INS/IXS**: Complete phonon dispersion across Brillouin zone
2. **Raman**: Zone-center optical modes, symmetry
3. **IR**: Polar modes, dielectric constants
4. **Pump-probe**: Phonon lifetimes, decay channels
5. **EELS**: Local modes in heterostructures

## Summary

- **INS** is the gold standard for phonon dispersion with complete \\(\mathbf{q}\\)-space access
- **Coherent vs incoherent** scattering probe dispersion vs DOS
- **Triple-axis** provides high resolution point-by-point, **TOF** gives broad energy range simultaneously
- **Raman** probes zone-center modes via polarizability changes (\\(\partial\alpha/\partial u \neq 0\\))
- **First-order Raman** gives frequencies, **second-order** reveals DOS
- **IR** measures polar modes via dipole changes (\\(\partial P/\partial u \neq 0\\))
- **Polaritons** form from IR phonon-photon coupling with stop band \\([\omega_{TO}, \omega_{LO}]\\)
- **LST relation** connects LO/TO splitting to dielectric constants
- **EELS** offers atomic resolution, **IXS** works for light elements and high pressure
- **Pump-probe** accesses ultrafast phonon dynamics with femtosecond resolution
- **Combining techniques** provides comprehensive phonon characterization

## Exercises

1. A thermal neutron (\\(\lambda_i = 2.5\\) Å) creates a 25 meV phonon at \\(q = 0.4\pi/a\\). Calculate incident energy \\(E_i\\), final energy \\(E_f\\), and verify momentum conservation.

2. For D₄h symmetry, determine Raman-active representations using character tables. Which polarization configurations show A₁g mode?

3. GaP has \\(\omega_{TO} = 367\\) cm⁻¹, \\(\epsilon_\infty = 9.1\\), \\(\epsilon_0 = 11.1\\). Calculate \\(\omega_{LO}\\) using LST relation and stop band width.

4. Natural vanadium: \\(b_{coh} = -0.38\\) fm, \\(b_{inc} = 5.08\\) fm. Calculate coherent and incoherent cross-sections. Which is better for phonon dispersion measurements?

5. Recommend optimal technique(s) for: (a) 1 μm semiconductor nanowire zone-center modes, (b) Complete acoustic dispersion at 50 GPa, (c) Phonon lifetimes near superconducting \\(T_c\\).

---

**Navigation**

[← Chapter 4: Electron-Phonon Coupling](chapter-4.md) | [Back to Series Index](index.md)

---

**Disclaimer**

This educational content was created for the Hashimoto Lab knowledge base. While care has been taken to ensure accuracy, readers should verify critical information with primary sources and consult original research papers.

**Author**: MS Knowledge Hub Content Team
**Version**: 1.0 | **Last Updated**: 2025-12-19
**License**: Creative Commons BY 4.0
