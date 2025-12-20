---
title: "Chapter 2: Anharmonic Effects"
chapter_title: "Anharmonic Effects"
subtitle: "Beyond the Harmonic Approximation in Lattice Dynamics"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-intermediate/chapter-2.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Intermediate Phonon Physics](index.md) > Chapter 2

# Chapter 2: Anharmonic Effects

**Beyond the Harmonic Approximation in Lattice Dynamics**

⏱️ 35-45 min | 💻 7 Code Examples | 📊 Intermediate-Advanced

## Learning Objectives

- Understand the limitations of the harmonic approximation and when anharmonic effects become important
- Derive the anharmonic expansion of the crystal potential to cubic and quartic order
- Explain phonon-phonon interactions and three-phonon scattering processes
- Calculate phonon lifetimes using Fermi's golden rule and phase space considerations
- Understand thermal expansion as a consequence of anharmonicity
- Apply perturbation theory to compute anharmonic corrections to phonon frequencies
- Implement computational methods for anharmonic phonon calculations
- Connect anharmonicity to thermal conductivity and heat transport

## 2.1 Limitations of the Harmonic Approximation

### When Harmonic Theory Breaks Down

The harmonic approximation assumes small amplitude vibrations and a purely quadratic potential. However, real materials exhibit deviations that become significant when:

1. **Temperature increases**: Larger thermal amplitudes explore anharmonic regions of potential
2. **Soft phonon modes**: Near structural phase transitions, some modes have very low frequencies
3. **Light atoms**: H, He, and other light elements have large zero-point motion
4. **High pressure**: Atomic interactions become strongly repulsive at small separations

> **Observable Consequences of Anharmonicity**
>
> - **Thermal expansion**: Crystals expand with temperature (cannot occur in harmonic theory)
> - **Finite phonon lifetimes**: Phonons decay into other phonons
> - **Temperature-dependent phonon frequencies**: Modes shift and broaden with temperature
> - **Thermal conductivity**: Heat transport via phonon-phonon scattering
> - **Phonon-phonon coupling**: Energy transfer between different vibrational modes

### Physical Origin

In the harmonic approximation, atoms oscillate independently in parabolic potential wells. Anharmonicity arises from:

- **Asymmetric bonds**: Easier to stretch than compress (or vice versa)
- **Multi-body interactions**: Coordination-dependent forces
- **Electronic rearrangement**: Charge distribution changes with geometry

## 2.2 Anharmonic Expansion of the Potential

### Taylor Expansion to Higher Orders

Expanding the potential energy \\(U\\) beyond second order in atomic displacements \\(u\\):

\\[
U = U_0 + \underbrace{\sum_i \frac{\partial U}{\partial u_i} u_i}_{\text{Linear (= 0 at equilibrium)}} + \underbrace{\frac{1}{2} \sum_{ij} \frac{\partial^2 U}{\partial u_i \partial u_j} u_i u_j}_{\text{Harmonic}} + \underbrace{\frac{1}{6} \sum_{ijk} \frac{\partial^3 U}{\partial u_i \partial u_j \partial u_k} u_i u_j u_k}_{\text{Cubic anharmonicity}} + \underbrace{\frac{1}{24} \sum_{ijkl} \frac{\partial^4 U}{\partial u_i \partial u_j \partial u_k \partial u_l} u_i u_j u_k u_l}_{\text{Quartic anharmonicity}} + \cdots
\\]

**Notation:**

- \\(\Phi_{ij} = \partial^2 U / \partial u_i \partial u_j\\): Harmonic force constants
- \\(\Phi_{ijk} = \partial^3 U / \partial u_i \partial u_j \partial u_k\\): Cubic anharmonic coefficients
- \\(\Phi_{ijkl} = \partial^4 U / \partial u_i \partial u_j \partial u_k \partial u_l\\): Quartic anharmonic coefficients

### Three-Phonon and Four-Phonon Interactions

When expressed in terms of phonon creation/annihilation operators, anharmonic terms lead to:

**Cubic term** (\\(\Phi_{ijk}\\)): Three-phonon processes

\\[
H^{(3)} = \frac{1}{6} \sum_{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3} \sum_{s_1, s_2, s_3} V^{(3)}(\mathbf{q}_1 s_1, \mathbf{q}_2 s_2, \mathbf{q}_3 s_3) \times (a_{\mathbf{q}_1 s_1} + a_{-\mathbf{q}_1 s_1}^\dagger)(a_{\mathbf{q}_2 s_2} + a_{-\mathbf{q}_2 s_2}^\dagger)(a_{\mathbf{q}_3 s_3} + a_{-\mathbf{q}_3 s_3}^\dagger) \delta_{\mathbf{q}_1 + \mathbf{q}_2 + \mathbf{q}_3, \mathbf{G}}
\\]

**Quartic term** (\\(\Phi_{ijkl}\\)): Four-phonon processes

\\[
H^{(4)} = \frac{1}{24} \sum_{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3, \mathbf{q}_4} \sum_{s_1, s_2, s_3, s_4} V^{(4)}(\mathbf{q}_1 s_1, \mathbf{q}_2 s_2, \mathbf{q}_3 s_3, \mathbf{q}_4 s_4) \times \prod_{i=1}^{4} (a_{\mathbf{q}_i s_i} + a_{-\mathbf{q}_i s_i}^\dagger) \delta_{\sum \mathbf{q}_i, \mathbf{G}}
\\]

where \\(\mathbf{G}\\) is a reciprocal lattice vector (momentum conservation modulo \\(\mathbf{G}\\)).

## 2.3 Phonon-Phonon Scattering

### Three-Phonon Processes

The dominant anharmonic effect at moderate temperatures comes from **three-phonon scattering**, where one phonon decays into two or two phonons merge into one.

**Process types:**

1. **Decay (splitting)**: \\(\mathbf{q} \to \mathbf{q}' + \mathbf{q}''\\)
   - One phonon splits into two phonons

2. **Coalescence (merging)**: \\(\mathbf{q}' + \mathbf{q}'' \to \mathbf{q}\\)
   - Two phonons combine into one

### Selection Rules

**Energy conservation:**

\\[
\omega_{\mathbf{q}s} = \omega_{\mathbf{q}'s'} + \omega_{\mathbf{q}''s''} \quad \text{(decay)}
\\]

\\[
\omega_{\mathbf{q}s} = \omega_{\mathbf{q}'s'} - \omega_{\mathbf{q}''s''} \quad \text{(coalescence)}
\\]

**Momentum conservation (crystal momentum):**

\\[
\mathbf{q} = \mathbf{q}' + \mathbf{q}'' + \mathbf{G}
\\]

where \\(\mathbf{G}\\) is a reciprocal lattice vector.

### Normal vs Umklapp Processes

- **Normal (N) processes**: \\(\mathbf{G} = 0\\), crystal momentum is strictly conserved
  - Do not directly contribute to thermal resistance
  - Redistribute phonon momentum without dissipation

- **Umklapp (U) processes**: \\(\mathbf{G} \neq 0\\), scattering across Brillouin zone boundary
  - Primary source of thermal resistance in pure crystals
  - Reverse phonon momentum, leading to energy dissipation

**Temperature dependence**: Umklapp processes require phonons with \\(\mathbf{q} \sim \mathbf{G}\\), which are thermally activated. Thus, U-processes become more important at higher temperatures.

## 2.4 Phonon Lifetimes and Linewidths

### Fermi's Golden Rule

The **phonon lifetime** \\(\tau_{\mathbf{q}s}\\) due to three-phonon scattering is calculated using Fermi's golden rule:

\\[
\frac{1}{\tau_{\mathbf{q}s}} = \frac{2\pi}{\hbar} \sum_{\mathbf{q}', \mathbf{q}''} \sum_{s', s''} |V^{(3)}|^2 \left[ (n_{\mathbf{q}'s'} + 1)(n_{\mathbf{q}''s''} + 1) \delta(\omega_{\mathbf{q}s} - \omega_{\mathbf{q}'s'} - \omega_{\mathbf{q}''s''}) + 2n_{\mathbf{q}'s'}(n_{\mathbf{q}''s''} + 1) \delta(\omega_{\mathbf{q}s} + \omega_{\mathbf{q}'s'} - \omega_{\mathbf{q}''s''}) \right] \delta_{\mathbf{q} \pm \mathbf{q}' \pm \mathbf{q}'', \mathbf{G}}
\\]

where \\(n_{\mathbf{q}s}\\) is the Bose-Einstein occupation number:

\\[
n_{\mathbf{q}s} = \frac{1}{e^{\hbar\omega_{\mathbf{q}s}/k_B T} - 1}
\\]

### Spectral Linewidth

The phonon spectral function has a Lorentzian lineshape with width determined by the lifetime:

\\[
A(\mathbf{q}, \omega) = \frac{1}{\pi} \frac{\Gamma_{\mathbf{q}s}}{(\omega - \omega_{\mathbf{q}s})^2 + \Gamma_{\mathbf{q}s}^2}
\\]

where \\(\Gamma_{\mathbf{q}s} = \hbar/(2\tau_{\mathbf{q}s})\\) is the half-width at half-maximum (HWHM).

### Temperature Dependence of Linewidth

At low temperatures (\\(k_B T \ll \hbar\omega\\)):

\\[
\Gamma(T) \approx \Gamma_0 + A T^4
\\]

At high temperatures (\\(k_B T \gg \hbar\omega\\)):

\\[
\Gamma(T) \approx B T
\\]

## 2.5 Thermal Expansion

### Gruneisen Parameter

Thermal expansion is a direct consequence of anharmonicity. In the harmonic approximation, the equilibrium lattice constant is independent of temperature.

The **Gruneisen parameter** quantifies the volume dependence of phonon frequencies:

\\[
\gamma_{\mathbf{q}s} = -\frac{V}{\omega_{\mathbf{q}s}} \frac{\partial \omega_{\mathbf{q}s}}{\partial V}
\\]

**Mode-averaged Gruneisen parameter:**

\\[
\gamma = \frac{\sum_{\mathbf{q}s} \gamma_{\mathbf{q}s} C_{\mathbf{q}s}}{\sum_{\mathbf{q}s} C_{\mathbf{q}s}}
\\]

where \\(C_{\mathbf{q}s}\\) is the mode-specific heat capacity.

### Connection to Thermal Expansion Coefficient

The volumetric thermal expansion coefficient \\(\alpha_V\\) is related to the Gruneisen parameter:

\\[
\alpha_V = \frac{1}{V} \frac{\partial V}{\partial T} = \frac{\gamma C_V}{B V}
\\]

where \\(B\\) is the bulk modulus and \\(C_V\\) is the heat capacity at constant volume.

> **Physical Interpretation**
>
> - \\(\gamma > 0\\): Most materials expand upon heating (phonon frequencies decrease with volume)
> - \\(\gamma < 0\\): Negative thermal expansion (rare, e.g., some ceramics like ZrW₂O₈)
> - Larger \\(\gamma\\): Stronger anharmonicity and larger thermal expansion

## 2.6 Perturbation Theory for Anharmonic Corrections

### Self-Energy of Phonons

The phonon self-energy \\(\Pi(\mathbf{q}, \omega)\\) describes the modification of phonon properties due to anharmonic interactions:

\\[
\omega_{\mathbf{q}s}^{\text{renorm}} = \omega_{\mathbf{q}s}^{\text{harm}} + \text{Re}\, \Pi(\mathbf{q}, \omega_{\mathbf{q}s})
\\]

\\[
\Gamma_{\mathbf{q}s} = -\text{Im}\, \Pi(\mathbf{q}, \omega_{\mathbf{q}s})
\\]

### Lowest-Order Self-Energy (Bubble Diagram)

In lowest-order perturbation theory, the self-energy from three-phonon interactions is:

\\[
\Pi_{\mathbf{q}s}(\omega) = \sum_{\mathbf{q}', s', s''} |V^{(3)}_{\mathbf{q}s, \mathbf{q}'s', \mathbf{q}''s''}|^2 \left[ \frac{n_{\mathbf{q}'s'} - n_{\mathbf{q}''s''}}{\omega - \omega_{\mathbf{q}'s'} + \omega_{\mathbf{q}''s''} + i\eta} + \frac{n_{\mathbf{q}'s'} + n_{\mathbf{q}''s''} + 1}{\omega + \omega_{\mathbf{q}'s'} + \omega_{\mathbf{q}''s''} + i\eta} \right]
\\]

## 2.7 Computational Methods for Anharmonicity

### Direct Calculation of Anharmonic Force Constants

Modern DFT-based approaches calculate \\(\Phi_{ijk}\\) and \\(\Phi_{ijkl}\\) using:

1. **Finite differences of forces**: Displace atoms and compute energy/force derivatives numerically
2. **DFPT for anharmonicity**: Compute third-order derivatives directly (available in some codes)

### Python Example: Simple Model for Temperature-Dependent Phonon Frequency

```python
import numpy as np
import matplotlib.pyplot as plt

def phonon_self_energy_simple(omega_0, T, gamma_anh, omega_max):
    """
    Simple model for temperature-dependent phonon self-energy.

    Parameters:
    -----------
    omega_0 : float
        Harmonic phonon frequency (THz)
    T : float or ndarray
        Temperature (K)
    gamma_anh : float
        Anharmonic coupling strength
    omega_max : float
        Maximum phonon frequency (THz)

    Returns:
    --------
    omega_T : float or ndarray
        Temperature-dependent frequency
    Gamma_T : float or ndarray
        Phonon linewidth (HWHM)
    """
    k_B = 0.0862  # meV/K
    k_B_THz = 0.0208  # THz/K

    # Average thermal phonon occupation
    x = omega_0 / (k_B_THz * T + 1e-6)
    n_avg = 1.0 / (np.exp(x) - 1.0 + 1e-10)

    # Frequency shift (real part of self-energy)
    delta_omega = -gamma_anh * omega_0 * (n_avg + 0.5)

    # Linewidth (imaginary part, proportional to T at high T)
    Gamma_T = gamma_anh * omega_0 * n_avg * (n_avg + 1)

    omega_T = omega_0 + delta_omega

    return omega_T, Gamma_T

# Parameters
omega_0 = 10.0  # THz (harmonic frequency)
gamma_anh = 0.02  # anharmonic coupling strength
omega_max = 20.0  # THz

T_range = np.linspace(0, 1000, 200)  # K

# Calculate temperature-dependent properties
omega_T, Gamma_T = phonon_self_energy_simple(omega_0, T_range, gamma_anh, omega_max)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Frequency shift
ax1.plot(T_range, omega_T, 'b-', linewidth=2)
ax1.axhline(omega_0, color='gray', linestyle='--', label=f'ω₀ = {omega_0} THz')
ax1.set_xlabel('Temperature (K)', fontsize=12)
ax1.set_ylabel('Phonon Frequency (THz)', fontsize=12)
ax1.set_title('Temperature-Dependent Phonon Frequency Shift', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Linewidth
ax2.plot(T_range, Gamma_T * 1000, 'r-', linewidth=2)  # Convert to GHz
ax2.set_xlabel('Temperature (K)', fontsize=12)
ax2.set_ylabel('Phonon Linewidth Γ (GHz)', fontsize=12)
ax2.set_title('Temperature-Dependent Phonon Linewidth', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Frequency at 300 K: {phonon_self_energy_simple(omega_0, 300, gamma_anh, omega_max)[0]:.3f} THz")
print(f"Linewidth at 300 K: {phonon_self_energy_simple(omega_0, 300, gamma_anh, omega_max)[1] * 1000:.3f} GHz")
```

## 2.8 Connection to Thermal Conductivity

### Phonon Boltzmann Transport Equation

Thermal conductivity is determined by solving the **phonon Boltzmann transport equation**:

\\[
\kappa = \frac{1}{3V} \sum_{\mathbf{q}s} C_{\mathbf{q}s} v_{\mathbf{q}s}^2 \tau_{\mathbf{q}s}
\\]

where:
- \\(C_{\mathbf{q}s}\\): Mode-specific heat capacity
- \\(v_{\mathbf{q}s}\\): Group velocity
- \\(\tau_{\mathbf{q}s}\\): Phonon lifetime (from anharmonic scattering)

### Temperature Dependence

At **low temperatures** (\\(T \ll \Theta_D\\)):
- Limited Umklapp processes
- Boundary scattering dominates
- \\(\kappa \propto T^3\\)

At **high temperatures** (\\(T \gg \Theta_D\\)):
- Strong Umklapp scattering
- \\(\kappa \propto 1/T\\)

## Summary

- The **harmonic approximation** fails to explain thermal expansion, finite phonon lifetimes, and temperature-dependent properties
- **Anharmonic terms** (cubic \\(\Phi_{ijk}\\), quartic \\(\Phi_{ijkl}\\)) in the potential lead to phonon-phonon interactions
- **Three-phonon scattering** is the dominant anharmonic effect, governed by energy and momentum conservation
- **Normal (N) processes** conserve crystal momentum; **Umklapp (U) processes** scatter across Brillouin zone boundaries and cause thermal resistance
- **Phonon lifetimes** are calculated using Fermi's golden rule and determine spectral linewidths
- **Thermal expansion** arises from anharmonicity, quantified by the Gruneisen parameter
- **Perturbation theory** provides phonon self-energy corrections to frequencies and linewidths
- Modern **DFT-based methods** enable first-principles calculation of anharmonic force constants

## Exercises

**Exercise 1**: Derive the lowest-order expression for the phonon lifetime due to three-phonon decay processes using Fermi's golden rule. Include both decay and coalescence channels.

**Exercise 2**: For a material with Gruneisen parameter \\(\gamma = 2.0\\), bulk modulus \\(B = 100\\) GPa, and molar heat capacity \\(C_V = 25\\) J/(mol·K), calculate the volumetric thermal expansion coefficient at room temperature.

**Exercise 3**: Explain why Umklapp processes are essential for thermal resistance while normal processes are not. Sketch the phonon momentum states before and after an N-process and a U-process.

**Exercise 4**: Write a Python program to simulate phonon-phonon scattering events in a 1D chain and calculate the average phonon lifetime as a function of temperature.

**Exercise 5 (Advanced)**: The thermal conductivity of diamond is extremely high (~2000 W/m·K at room temperature). Explain this using concepts from anharmonicity, phonon lifetimes, and the phonon Boltzmann transport equation.

---

**Navigation**

[← Chapter 1: Lattice Dynamics Theory](chapter-1.md) | [Chapter 3: Phonon Scattering →](chapter-3.md)

---

**Disclaimer**

This educational content was created for the Hashimoto Lab knowledge base. While care has been taken to ensure accuracy, readers should verify critical information with primary sources and consult original research papers.

**Author**: MS Knowledge Hub Content Team
**Version**: 1.0 | **Last Updated**: 2025-12-19
**License**: Creative Commons BY 4.0
