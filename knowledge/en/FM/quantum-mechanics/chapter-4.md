---
title: "Chapter 4: Perturbation Theory and Scattering Theory"
chapter_title: "Chapter 4: Perturbation Theory and Scattering Theory"
subtitle: Perturbation Theory and Scattering
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-mechanics/chapter-4.html>) | Last sync: 2025-11-16

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Mechanics](<index.html>) > Chapter 4 

## 4.1 Time-Independent Perturbation Theory

### 📚 Non-Degenerate Perturbation Theory

Hamiltonian: \\(H = H_0 + \lambda V\\)

**First-Order Energy Correction** :

\\[ E_n^{(1)} = \langle n^{(0)} | V | n^{(0)} \rangle \\]

**Second-Order Energy Correction** :

\\[ E_n^{(2)} = \sum_{m \neq n} \frac{|\langle m^{(0)} | V | n^{(0)} \rangle|^2}{E_n^{(0)} - E_m^{(0)}} \\]

**First-Order Wavefunction Correction** :

\\[ |n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m^{(0)} | V | n^{(0)} \rangle}{E_n^{(0)} - E_m^{(0)}} |m^{(0)}\rangle \\]

### Application Example: Stark Effect (Linear Stark Effect in Hydrogen Atom)

Hydrogen atom in electric field: \\(V = eEz\\)

The first-order energy correction is zero (parity conservation). Polarization appears at second order.

## 4.2 Time-Dependent Perturbation Theory

### 📚 Fermi's Golden Rule

Transition probability due to time-dependent perturbation \\(V(t) = Ve^{-i\omega t}\\):

\\[ w_{i \to f} = \frac{2\pi}{\hbar} |\langle f | V | i \rangle|^2 \rho(E_f) \\]

where \\(\rho(E_f)\\) is the density of final states.

**Selection Rules** : Dipole transitions

\\[ \Delta l = \pm 1, \quad \Delta m = 0, \pm 1 \\]

### Application: Light Absorption and Emission

This describes the process of photon absorption and emission by atoms. It is related to Einstein's A and B coefficients.

## 4.3 Scattering Theory

### 📚 Scattering Cross Section

**Differential Scattering Cross Section** :

\\[ \frac{d\sigma}{d\Omega} = |f(\theta)|^2 \\]

where \\(f(\theta)\\) is the **scattering amplitude**.

**Born Approximation** (weak potential):

\\[ f(\theta) = -\frac{m}{2\pi\hbar^2} \int e^{i\mathbf{q}\cdot\mathbf{r}} V(r) d^3r \\]

where \\(\mathbf{q} = \mathbf{k}_i - \mathbf{k}_f\\) is the momentum transfer.

### Partial Wave Expansion

Scattering in a central force field:

\\[ f(\theta) = \frac{1}{k} \sum_{l=0}^\infty (2l+1) e^{i\delta_l} \sin\delta_l P_l(\cos\theta) \\]

where \\(\delta_l\\) is the **phase shift**.

### 🎯 Exercise Problems

  1. **Harmonic Oscillator Perturbation** : Calculate the energy correction up to second order for a harmonic oscillator subjected to perturbation \\(V = \alpha x^4\\).
  2. **Zeeman Effect** : Calculate the energy splitting of a hydrogen atom in a magnetic field.
  3. **Born Approximation** : Find the scattering cross section for the Yukawa potential \\(V(r) = \frac{g^2 e^{-\mu r}}{r}\\).
  4. **Resonance Scattering** : Determine the resonance condition where the phase shift becomes \\(\delta_0 = \pi/2\\) for s-wave scattering (l=0).

## Summary

In this chapter, we have learned perturbation theory and scattering theory. **Time-independent perturbation theory** provides systematic methods for calculating energy corrections, with applications to the Stark and Zeeman effects. **Time-dependent perturbation theory** yields Fermi's golden rule, selection rules, and descriptions of light absorption processes. **Scattering theory** introduces the scattering cross section, Born approximation for weak potentials, partial wave expansion, and the concept of phase shifts.

In the next chapter, we will study **Relativistic Quantum Mechanics**.

[← Chapter 3: Angular Momentum](<chapter-3.html>) [Chapter 5: Relativistic Quantum Mechanics →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
