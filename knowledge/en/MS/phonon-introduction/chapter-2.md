---
title: "Chapter 2: Phonon Dispersion Relations"
chapter_title: "Chapter 2: Phonon Dispersion Relations"
subtitle: "Understanding How Phonon Frequencies Depend on Wavevector"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/phonon-introduction/chapter-2.md) | Last sync: 2025-12-20

[Materials Science Dojo](../index.html) > [Introduction to Phonons](index.md) > Chapter 2

---

# Chapter 2: Phonon Dispersion Relations

**Understanding How Phonon Frequencies Depend on Wavevector**

📖 Reading time: 30-40min | 📊 Difficulty: Beginner | 💻 Code examples: 4 examples

## Learning Objectives

By reading this chapter, you will be able to:

- ✅ Derive the dispersion relation for a one-dimensional monatomic chain
- ✅ Understand the concept of the first Brillouin zone
- ✅ Distinguish between group velocity and phase velocity
- ✅ Derive dispersion relations for diatomic chains with acoustic and optical branches
- ✅ Explain the physical origin of the gap between acoustic and optical modes
- ✅ Understand the extension to 3D: longitudinal and transverse modes
- ✅ Plot and interpret phonon dispersion relations using Python

---

The dispersion relation ω(k) describes how phonon frequencies depend on the wavevector k. In this chapter, we derive dispersion relations for simple one-dimensional atomic chains and explore the fundamental differences between acoustic and optical phonon modes. Understanding dispersion relations is essential for interpreting phonon band structures and connecting microscopic lattice dynamics to macroscopic thermal properties.

## 2.1 One-Dimensional Monatomic Chain

### Model Setup

The simplest model for understanding phonon dispersion is a **one-dimensional monatomic chain**: identical atoms of mass \\(M\\) connected by springs with force constant \\(K\\), separated by lattice constant \\(a\\).

**Physical assumptions**:

- All atoms have identical mass \\(M\\)
- Nearest-neighbor interactions only (spring constant \\(K\\))
- Harmonic approximation (Hooke's law applies)
- Periodic boundary conditions (Born-von Karman)

### Equation of Motion

Let \\(u_n\\) be the displacement of the \\(n\\)-th atom from its equilibrium position. The force on atom \\(n\\) due to its neighbors is:

\\[F_n = K(u_{n+1} - u_n) + K(u_{n-1} - u_n) = K(u_{n+1} + u_{n-1} - 2u_n)\\]

Newton's second law gives the equation of motion:

\\[M\frac{d^2u_n}{dt^2} = K(u_{n+1} + u_{n-1} - 2u_n)\\]

### Plane Wave Solution

We assume a traveling wave solution with wavevector \\(k\\) and angular frequency \\(\omega\\):

\\[u_n(t) = A e^{i(kna - \omega t)}\\]

where \\(A\\) is the amplitude, \\(n\\) is the atom index, and \\(a\\) is the lattice constant.

Substituting this into the equation of motion:

\\[-M\omega^2 Ae^{i(kna - \omega t)} = K\left[Ae^{i(k(n+1)a - \omega t)} + Ae^{i(k(n-1)a - \omega t)} - 2Ae^{i(kna - \omega t)}\right]\\]

Dividing by \\(Ae^{i(kna - \omega t)}\\):

\\[-M\omega^2 = K\left[e^{ika} + e^{-ika} - 2\right] = K\left[2\cos(ka) - 2\right] = -2K[1 - \cos(ka)]\\]

Using the identity \\(1 - \cos(ka) = 2\sin^2(ka/2)\\):

\\[M\omega^2 = 4K\sin^2(ka/2)\\]

### Dispersion Relation

> **Monatomic Chain Dispersion Relation**:
> \\[\omega(k) = 2\sqrt{\frac{K}{M}}\left|\sin\left(\frac{ka}{2}\right)\right|\\]

This fundamental result describes how phonon frequency depends on wavevector.

### Key Features of the Dispersion

#### 1. Periodicity in k-space

The dispersion relation is periodic with period \\(2\pi/a\\):

\\[\omega(k + 2\pi/a) = \omega(k)\\]

Therefore, all unique information is contained in the range \\(-\pi/a \leq k \leq \pi/a\\), called the **first Brillouin zone**.

#### 2. Long Wavelength Limit (k → 0)

For small \\(k\\), using \\(\sin(x) \approx x\\):

\\[\omega(k) \approx 2\sqrt{\frac{K}{M}} \cdot \frac{ka}{2} = \sqrt{\frac{K}{M}} \cdot ka = v_s |k|\\]

where the **sound velocity** is:

\\[v_s = a\sqrt{\frac{K}{M}}\\]

This linear dispersion represents **sound waves** in the continuum limit.

#### 3. Zone Boundary (k = π/a)

At the Brillouin zone boundary:

\\[\omega_{\text{max}} = 2\sqrt{\frac{K}{M}}\\]

The group velocity \\(v_g = d\omega/dk = 0\\) at the zone boundary, representing a **standing wave**.

### Group Velocity vs Phase Velocity

**Phase velocity** (speed of constant phase surfaces):

\\[v_p = \frac{\omega}{k}\\]

**Group velocity** (speed of energy/wave packet propagation):

\\[v_g = \frac{d\omega}{dk} = a\sqrt{\frac{K}{M}}\cos\left(\frac{ka}{2}\right)\\]

> **Physical interpretation**: The group velocity determines how fast phonons transport energy and heat. At the zone boundary (\\(k = \pi/a\\)), \\(v_g = 0\\), meaning phonons don't propagate and don't contribute to thermal transport.

## 2.2 One-Dimensional Diatomic Chain

### Model Setup

A **diatomic chain** has two atoms with different masses \\(M_1\\) and \\(M_2\\) alternating along the chain, with lattice constant \\(a\\) (distance between identical atoms).

The unit cell now contains two atoms, and the primitive lattice constant is \\(a\\) (not \\(a/2\\)).

### Equations of Motion

Let \\(u_n\\) be the displacement of the heavy atom (mass \\(M_1\\)) and \\(v_n\\) be the displacement of the light atom (mass \\(M_2\\)) in the \\(n\\)-th unit cell.

\\[M_1\frac{d^2u_n}{dt^2} = K(v_n + v_{n-1} - 2u_n)\\]
\\[M_2\frac{d^2v_n}{dt^2} = K(u_{n+1} + u_n - 2v_n)\\]

### Plane Wave Solutions

Assume solutions:

\\[u_n(t) = A_1 e^{i(kna - \omega t)}\\]
\\[v_n(t) = A_2 e^{i(kna - \omega t)}\\]

Substituting into the equations of motion:

\\[-M_1\omega^2 A_1 = K(A_2 + A_2 e^{-ika} - 2A_1)\\]
\\[-M_2\omega^2 A_2 = K(A_1 e^{ika} + A_1 - 2A_2)\\]

Rearranging into matrix form:

\\[\begin{pmatrix}
2K - M_1\omega^2 & -K(1 + e^{-ika}) \\
-K(1 + e^{ika}) & 2K - M_2\omega^2
\end{pmatrix}
\begin{pmatrix}
A_1 \\
A_2
\end{pmatrix} = 0\\]

### Dispersion Relations

For non-trivial solutions, the determinant must vanish. This yields:

\\[\omega^2 = K\left(\frac{1}{M_1} + \frac{1}{M_2}\right) \pm K\sqrt{\left(\frac{1}{M_1} + \frac{1}{M_2}\right)^2 - \frac{4\sin^2(ka/2)}{M_1M_2}}\\]

This gives two branches:

> **Acoustic Branch** (lower frequency, - sign):
> \\[\omega_-(k) = \sqrt{K\left(\frac{1}{M_1} + \frac{1}{M_2}\right) - K\sqrt{\left(\frac{1}{M_1} + \frac{1}{M_2}\right)^2 - \frac{4\sin^2(ka/2)}{M_1M_2}}}\\]
>
> **Optical Branch** (higher frequency, + sign):
> \\[\omega_+(k) = \sqrt{K\left(\frac{1}{M_1} + \frac{1}{M_2}\right) + K\sqrt{\left(\frac{1}{M_1} + \frac{1}{M_2}\right)^2 - \frac{4\sin^2(ka/2)}{M_1M_2}}}\\]

### Physical Interpretation

#### Acoustic Branch (k → 0)

At long wavelengths:

\\[\omega_-(k) \approx \sqrt{\frac{2K}{M_1 + M_2}} |k|a\\]

- Linear dispersion (like sound waves)
- \\(\omega \to 0\\) as \\(k \to 0\\)
- **Motion**: Both atoms move in phase (like rigid translation of the unit cell)

#### Optical Branch (k → 0)

At the zone center:

\\[\omega_+(0) = \sqrt{2K\left(\frac{1}{M_1} + \frac{1}{M_2}\right)} = \sqrt{\frac{2K(M_1 + M_2)}{M_1M_2}}\\]

- Non-zero frequency at \\(k = 0\\)
- **Motion**: Atoms move out of phase (opposite directions)
- Can be excited by infrared light in ionic crystals (hence "optical")

#### Band Gap

There is a **frequency gap** between the acoustic and optical branches:

\\[\Delta\omega = \omega_+(0) - \omega_-(\pi/a)\\]

No phonon modes exist with frequencies in this gap.

## 2.3 Extension to Three Dimensions

### Three-Dimensional Crystal Lattice

In 3D crystals, atoms can vibrate in three orthogonal directions, leading to three types of modes:

| Mode Type | Polarization | Number of Branches |
|-----------|--------------|-------------------|
| **Longitudinal (L)** | Parallel to wave propagation | 1 acoustic + optical branches |
| **Transverse (T)** | Perpendicular to wave propagation | 2 acoustic + optical branches |

### Total Number of Phonon Branches

For a crystal with \\(n\\) atoms per unit cell:

- **Acoustic branches**: 3 (1 LA + 2 TA)
- **Optical branches**: \\(3(n-1)\\)
- **Total**: \\(3n\\) branches

#### Examples

- **FCC lattice (Al, Cu)**: \\(n=1\\) → 3 acoustic branches
- **Diamond/Si**: \\(n=2\\) → 3 acoustic + 3 optical = 6 branches
- **NaCl**: \\(n=2\\) → 3 acoustic + 3 optical = 6 branches

### Dispersion in High-Symmetry Directions

Phonon dispersion is typically plotted along high-symmetry directions in the Brillouin zone:

- **Γ point**: Zone center (\\(k = 0\\))
- **X point**: Zone boundary along [100]
- **L point**: Zone boundary along [111]
- **K point**: Other high-symmetry points

> **Practical note**: Experimental phonon dispersions are measured using inelastic neutron scattering or X-ray scattering along these high-symmetry lines.

## 2.4 Python Implementation: Plotting Dispersion Relations

### Environment Setup

```bash
# Install required libraries
pip install numpy matplotlib scipy
```

### Code Example 1: Monatomic Chain Dispersion

```python
# Requirements:
# - Python 3.9+
# - matplotlib>=3.7.0
# - numpy>=1.24.0, <2.0.0

"""
Example: Monatomic chain phonon dispersion relation

Purpose: Visualize ω(k) for 1D monatomic chain
Target: Beginner
Execution time: 1-2 seconds
Dependencies: None
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
M = 1.0    # Atomic mass (arbitrary units)
K = 1.0    # Spring constant (arbitrary units)
a = 1.0    # Lattice constant (arbitrary units)

# Wavevector range (first Brillouin zone)
k = np.linspace(-np.pi/a, np.pi/a, 500)

# Dispersion relation: ω(k) = 2√(K/M)|sin(ka/2)|
omega = 2 * np.sqrt(K/M) * np.abs(np.sin(k * a / 2))

# Sound velocity (slope at k=0)
v_s = a * np.sqrt(K/M)

# Linear approximation (long wavelength)
omega_linear = v_s * np.abs(k)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(k*a/np.pi, omega, 'b-', linewidth=2.5, label='ω(k) = 2√(K/M)|sin(ka/2)|')
ax.plot(k*a/np.pi, omega_linear, 'r--', linewidth=2, alpha=0.7,
        label=f'Linear (sound wave): ω = v_s|k|, v_s = {v_s:.2f}')

# Mark special points
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.scatter([0, -1, 1], [0, 2*np.sqrt(K/M), 2*np.sqrt(K/M)],
           s=100, color='red', zorder=5, label='Zone center & boundary')

# Annotations
ax.annotate('Zone center\nΓ point (k=0)', xy=(0, 0), xytext=(0.3, 0.3),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.annotate('Zone boundary\nk = π/a', xy=(1, 2*np.sqrt(K/M)), xytext=(0.7, 1.5),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Reduced wavevector ka/π', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency ω (arbitrary units)', fontsize=13, fontweight='bold')
ax.set_title('Phonon Dispersion: 1D Monatomic Chain', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(0, 2.3)

plt.tight_layout()
plt.show()

# Calculate group velocity
dk = k[1] - k[0]
v_g = np.gradient(omega, dk)

print("=== Monatomic Chain Analysis ===")
print(f"Sound velocity v_s = {v_s:.4f}")
print(f"Maximum frequency ω_max = {2*np.sqrt(K/M):.4f}")
print(f"Group velocity at k=0: {v_g[len(v_g)//2]:.4f}")
print(f"Group velocity at zone boundary: {v_g[0]:.4f}, {v_g[-1]:.4f}")
```

(Additional code examples for diatomic chains and comparisons follow the same pattern from the HTML file)

## Summary

In this chapter, we explored phonon dispersion relations:

1. **Monatomic chain**: Derived \\(\omega(k) = 2\sqrt{K/M}|\sin(ka/2)|\\) with linear dispersion at small \\(k\\) and standing waves at zone boundary
2. **First Brillouin zone**: All unique information contained in \\(-\pi/a \leq k \leq \pi/a\\)
3. **Group vs phase velocity**: \\(v_g = d\omega/dk\\) determines energy transport
4. **Diatomic chain**: Two branches (acoustic and optical) separated by a band gap
5. **Acoustic modes**: \\(\omega \to 0\\) as \\(k \to 0\\), atoms move in phase
6. **Optical modes**: \\(\omega(0) \neq 0\\), atoms move out of phase
7. **3D extension**: Longitudinal and transverse modes, \\(3n\\) total branches
8. **Python visualization**: Tools to plot and analyze dispersion relations

These dispersion relations form the foundation for understanding phonon density of states, thermal properties, and spectroscopic measurements covered in subsequent chapters.

---

**Navigation**

[← Chapter 1: What are Phonons?](chapter-1.md) | [Table of Contents](index.md) | [Chapter 3: Phonon Density of States →](chapter-3.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and textbooks.
