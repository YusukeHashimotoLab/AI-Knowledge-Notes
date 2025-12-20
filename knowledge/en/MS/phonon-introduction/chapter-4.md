---
title: "Chapter 4: Thermal Properties of Solids"
chapter_title: "Chapter 4: Thermal Properties of Solids"
subtitle: "Heat Capacity, Thermal Expansion, and the Grüneisen Parameter"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-introduction/chapter-4.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Introduction to Phonons](index.md) > Chapter 4

---

# Chapter 4: Thermal Properties of Solids

**Heat Capacity, Thermal Expansion, and the Grüneisen Parameter**

📚 Introduction to Phonons | ⏱️ 2.5 hours | 📖 Chapter 4 of 5

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain how phonons contribute to the heat capacity of solids
- Understand the classical Dulong-Petit law and its limitations
- Derive and apply the Einstein and Debye models of heat capacity
- Explain the T³ law for heat capacity at low temperatures
- Connect the Grüneisen parameter to thermal expansion
- Understand the role of anharmonicity in thermal properties
- Calculate heat capacity using Python for different models

---

## 1. Introduction: Phonons and Thermal Properties

In previous chapters, we learned that phonons are quantized lattice vibrations. But why should we care about these quantum mechanical objects? The answer lies in their profound influence on the macroscopic thermal properties of materials. Every time you heat a metal, observe thermal expansion, or measure thermal conductivity, you're witnessing the collective behavior of trillions of phonons.

This chapter bridges the gap between microscopic phonon physics and macroscopic thermal phenomena. We'll see how the distribution of phonon modes—characterized by the density of states we studied in Chapter 3—directly determines properties like heat capacity and thermal expansion.

> **Key Insight**: The fundamental connection is simple: **thermal energy in solids is stored primarily in phonons**. When you add heat to a crystal, you're creating phonons. When the crystal expands with temperature, it's because anharmonic phonon interactions change the equilibrium lattice spacing.

## 2. Classical Theory: The Dulong-Petit Law

### 2.1 Historical Context

In 1819, French scientists Pierre Louis Dulong and Alexis Thérèse Petit made an empirical observation: at high temperatures, the molar heat capacity of many solid elements is approximately constant, around 25 J/(mol·K) or 3R, where R is the gas constant (8.314 J/(mol·K)).

### 2.2 Classical Derivation

From classical statistical mechanics, we can derive this result. Consider N atoms in a solid, each able to vibrate in three dimensions. Each degree of freedom has average energy k_B T/2 (kinetic) + k_B T/2 (potential) = k_B T by the equipartition theorem.

Total energy for N atoms with 3N degrees of freedom:

\\[U = 3Nk_BT\\]

The heat capacity at constant volume is:

\\[C_V = \left(\frac{\partial U}{\partial T}\right)_V = 3Nk_B = 3R \approx 25 \text{ J/(mol·K)}\\]

where we used N = N_A (Avogadro's number) for one mole.

**Dulong-Petit Law**: For a solid at high temperature: **C_V = 3R ≈ 25 J/(mol·K)**

This is remarkably accurate for many elements at room temperature and above.

### 2.3 Limitations of the Classical Theory

However, the Dulong-Petit law fails dramatically in several important cases:

- **Low Temperatures:** Heat capacity drops to zero as T → 0, not remaining at 3R
- **Light Elements:** Diamond, beryllium, and silicon have C_V ≪ 3R even at room temperature
- **Temperature Dependence:** The classical theory predicts no temperature dependence, but experiments show C_V(T) varies strongly

These failures pointed to a fundamental problem: **classical physics cannot explain the thermal properties of solids**. A quantum theory was needed.

## 3. Einstein Model of Heat Capacity

### 3.1 Einstein's Quantum Hypothesis (1907)

Albert Einstein made a revolutionary proposal: treat each atom as an independent quantum harmonic oscillator with a single characteristic frequency ω_E. Unlike classical oscillators that can have any energy, quantum oscillators have discrete energy levels:

\\[E_n = \hbar\omega_E\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, 3, \ldots\\]

The key quantum feature: oscillators at low temperature tend to stay in low energy states.

### 3.2 Derivation of Einstein Heat Capacity

Using Bose-Einstein statistics, the average number of phonons at temperature T is:

\\[\langle n \rangle = \frac{1}{e^{\hbar\omega_E/k_BT} - 1}\\]

The average energy per oscillator (excluding zero-point energy):

\\[\langle E \rangle = \hbar\omega_E \langle n \rangle = \frac{\hbar\omega_E}{e^{\hbar\omega_E/k_BT} - 1}\\]

For N atoms (3N oscillators), total energy:

\\[U = 3N\frac{\hbar\omega_E}{e^{\hbar\omega_E/k_BT} - 1}\\]

Heat capacity:

\\[C_V^{\text{Einstein}} = \frac{\partial U}{\partial T} = 3Nk_B\left(\frac{\hbar\omega_E}{k_BT}\right)^2 \frac{e^{\hbar\omega_E/k_BT}}{\left(e^{\hbar\omega_E/k_BT} - 1\right)^2}\\]

Defining the **Einstein temperature** Θ_E = ℏω_E/k_B:

\\[C_V^{\text{Einstein}} = 3R\left(\frac{\Theta_E}{T}\right)^2 \frac{e^{\Theta_E/T}}{\left(e^{\Theta_E/T} - 1\right)^2}\\]

### 3.3 Behavior of Einstein Model

**High Temperature Limit (T ≫ Θ_E):**

\\[C_V^{\text{Einstein}} \approx 3R\\]

✓ Recovers the classical Dulong-Petit law!

**Low Temperature Limit (T ≪ Θ_E):**

\\[C_V^{\text{Einstein}} \approx 3R\left(\frac{\Theta_E}{T}\right)^2 e^{-\Theta_E/T}\\]

✓ Goes to zero as T → 0 (satisfies third law of thermodynamics)
✗ But decreases exponentially, not as T³ as observed experimentally

## 4. Debye Model of Heat Capacity

### 4.1 Debye's Improvement (1912)

Peter Debye realized that Einstein's assumption of a single frequency was too simplistic. Real solids have a continuous spectrum of phonon frequencies, as we learned in Chapter 3. Debye proposed using the phonon density of states g(ω) in a more realistic model.

### 4.2 Debye Approximation

Debye made a crucial simplification: approximate the phonon DOS with:

\\[g(\omega) = \begin{cases}
\frac{9N}{\omega_D^3}\omega^2 & 0 \leq \omega \leq \omega_D \\
0 & \omega > \omega_D
\end{cases}\\]

where ω_D is the **Debye cutoff frequency**, chosen so that:

\\[\int_0^{\omega_D} g(\omega) d\omega = 3N\\]

(total of 3N modes for N atoms)

**Physical Meaning**: This approximation assumes all modes are acoustic-like with ω ∝ k up to a cutoff. It's exactly correct for the Debye model from Chapter 3, and a reasonable approximation for many real materials.

### 4.3 Debye Heat Capacity Formula

Total internal energy:

\\[U = \int_0^{\omega_D} g(\omega) \hbar\omega \langle n(\omega) \rangle d\omega = \int_0^{\omega_D} \frac{9N\omega^2}{\omega_D^3} \frac{\hbar\omega}{e^{\hbar\omega/k_BT} - 1} d\omega\\]

Defining the **Debye temperature** Θ_D = ℏω_D/k_B and x = ℏω/(k_B T):

\\[U = 9Nk_BT\left(\frac{T}{\Theta_D}\right)^3 \int_0^{\Theta_D/T} \frac{x^3}{e^x - 1} dx\\]

The heat capacity is:

\\[C_V^{\text{Debye}} = 9R\left(\frac{T}{\Theta_D}\right)^3 \int_0^{\Theta_D/T} \frac{x^4 e^x}{(e^x - 1)^2} dx\\]

### 4.4 Low and High Temperature Limits

**High Temperature (T ≫ Θ_D):**

\\[C_V^{\text{Debye}} \to 3R\\]

✓ Recovers Dulong-Petit law

**Low Temperature (T ≪ Θ_D):**

When T ≪ Θ_D, the upper limit of integration becomes ∞, and the integral equals 4π⁴/15:

\\[C_V^{\text{Debye}} = \frac{12\pi^4}{5}R\left(\frac{T}{\Theta_D}\right)^3 = \frac{234R}{\Theta_D^3}T^3\\]

> **Debye T³ Law**: At low temperatures: **C_V ∝ T³**
>
> This matches experimental observations beautifully and is one of the great triumphs of quantum theory in solid-state physics!

### 4.5 Physical Origin of T³ Law

Why does heat capacity scale as T³ at low temperature? Two factors:

1. **Density of states:** g(ω) ∝ ω² for acoustic phonons (3D Debye model)
2. **Thermal occupation:** Only phonons with ℏω ≲ k_B T are thermally excited. The number of such modes scales as (k_B T)³

Result: C_V ∝ (fraction of excited modes) × (energy per mode) ∝ T³ × T = T³

## 7. Thermal Expansion

### 7.1 The Origin of Thermal Expansion

Why do materials expand when heated? Surprisingly, **the harmonic approximation predicts zero thermal expansion!** In a purely harmonic potential, the average atomic position doesn't change with temperature—atoms just vibrate more vigorously about the same equilibrium point.

Thermal expansion arises from **anharmonicity** in the interatomic potential. Real potentials are asymmetric: it's easier to pull atoms apart than to push them together.

### 7.2 Anharmonic Potential

A realistic interatomic potential can be expanded as:

\\[U(r) = U_0 + \frac{1}{2}k(r - r_0)^2 + g(r - r_0)^3 + \ldots\\]

where:

- k is the harmonic force constant (k > 0)
- g is the cubic anharmonic term (typically g < 0)
- r₀ is the equilibrium separation at T = 0

The negative cubic term (g < 0) means the potential is softer (easier to stretch) at larger separations. As atoms vibrate with larger amplitude at higher T, they sample this asymmetric potential, leading to an increase in the average bond length.

### 7.3 Linear Thermal Expansion Coefficient

The linear thermal expansion coefficient is defined as:

\\[\alpha_L = \frac{1}{L}\frac{dL}{dT}\\]

where L is a linear dimension of the crystal.

For a 3D crystal, the volumetric thermal expansion coefficient is:

\\[\alpha_V = \frac{1}{V}\frac{dV}{dT} \approx 3\alpha_L\\]

Thermal expansion is directly related to phonon anharmonicity and can be quantified using the Grüneisen parameter.

## 8. The Grüneisen Parameter

### 8.1 Definition

The Grüneisen parameter γ quantifies how phonon frequencies change when the crystal volume changes. For a phonon mode with frequency ω:

\\[\gamma_i = -\frac{d\ln\omega_i}{d\ln V} = -\frac{V}{\omega_i}\frac{\partial\omega_i}{\partial V}\\]

The mode Grüneisen parameter tells us: if we compress the crystal (reduce V), how much does the phonon frequency increase?

### 8.2 Physical Interpretation

**Positive γ (typical):** Compression increases phonon frequencies (atoms vibrate faster when pushed closer together). Most materials have γ ≈ 1–3.

**Negative γ (rare):** Some modes in certain materials have frequencies that decrease upon compression. This can lead to negative thermal expansion!

### 8.3 Grüneisen Relation

The Grüneisen parameter connects thermal expansion to heat capacity through the fundamental relation:

\\[\alpha_V = \frac{\gamma C_V}{B_T V}\\]

where:

- α_V is the volumetric thermal expansion coefficient
- C_V is the heat capacity at constant volume
- B_T is the isothermal bulk modulus
- V is the volume

> **Key Insight**: The Grüneisen relation shows that thermal expansion is proportional to heat capacity. Both are determined by phonons, but thermal expansion requires anharmonicity while heat capacity does not.

## 10. Summary

**Key Takeaways**

- **Classical Dulong-Petit law** (C_V = 3R) works at high T but fails at low T and for light elements
- **Einstein model** introduces quantum mechanics with a single frequency, captures T → 0 behavior but with exponential decay
- **Debye model** uses realistic phonon DOS, predicts correct T³ law at low temperatures
- **Debye T³ law**: C_V ∝ T³ at T ≪ Θ_D is a signature of acoustic phonons in 3D
- **Thermal expansion** requires anharmonicity—harmonic crystals don't expand
- **Grüneisen parameter** γ connects phonon frequencies to volume changes and relates thermal expansion to heat capacity
- **Anharmonicity** is essential for thermal expansion, thermal conductivity, and phonon lifetimes

This chapter completes the bridge from microscopic phonon theory (Chapters 1-3) to macroscopic thermal properties. In Chapter 5, we'll explore how these concepts apply to real materials and how experimental techniques measure phonon properties.

---

**Navigation**

[← Chapter 3: Phonon Density of States](chapter-3.md) | [Table of Contents](index.md) | [Chapter 5: Phonons in Real Materials →](chapter-5.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and textbooks.
