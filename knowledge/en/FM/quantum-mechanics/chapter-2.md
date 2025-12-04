---
title: "Chapter 2: 3D Quantum Systems and Symmetry"
chapter_title: "Chapter 2: 3D Quantum Systems and Symmetry"
subtitle: 3D Quantum Systems and Symmetry
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-mechanics/chapter-2.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Introduction to Quantum Mechanics](<index.html>) > Chapter 2 

## 2.1 Quantum Theory of the Hydrogen Atom

### 📚 Schrödinger Equation for the Hydrogen Atom

Electron in Coulomb potential \\(V(r) = -\frac{e^2}{4\pi\epsilon_0 r}\\):

\\[ -\frac{\hbar^2}{2m}\nabla^2\psi - \frac{e^2}{4\pi\epsilon_0 r}\psi = E\psi \\]

**Separation of variables in spherical coordinates** : \\(\psi(r,\theta,\phi) = R(r)Y_l^m(\theta,\phi)\\)

**Radial equation** :

\\[ -\frac{\hbar^2}{2m}\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) + \left[\frac{\hbar^2 l(l+1)}{2mr^2} - \frac{e^2}{4\pi\epsilon_0 r}\right]R = ER \\]

**Energy eigenvalues** :

\\[ E_n = -\frac{me^4}{32\pi^2\epsilon_0^2\hbar^2 n^2} = -\frac{13.6\text{ eV}}{n^2}, \quad n = 1, 2, 3, \ldots \\]

## Summary

In this chapter, we studied 3D quantum systems. The **hydrogen atom** provides exact solutions for the Coulomb potential, characterized by quantum numbers (n, l, m). We examined **radial wave functions** described by Laguerre polynomials and their probability distributions. The **spherical harmonics** describe the angular dependence and shapes of atomic orbitals. Finally, the **two-particle problem** demonstrates the separation of center-of-mass motion, the concept of reduced mass, and isotope effects.

In the next chapter, we will study angular momentum theory and spin.

[← Chapter 1: Foundations of Wave Mechanics](<chapter-1.html>) [Chapter 3: Angular Momentum →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
