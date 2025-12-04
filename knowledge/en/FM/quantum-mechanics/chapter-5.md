---
title: "Chapter 5: Relativistic Quantum Mechanics"
chapter_title: "Chapter 5: Relativistic Quantum Mechanics"
subtitle: Relativistic Quantum Mechanics
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-mechanics/chapter-5.html>) | Last sync: 2025-11-16

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Mechanics](<index.html>) > Chapter 5 

## 5.1 Klein-Gordon Equation

### 📚 Relativistic Energy-Momentum Relation

Einstein's relation: \\(E^2 = (pc)^2 + (mc^2)^2\\)

Quantization: \\(E \to i\hbar\frac{\partial}{\partial t}\\), \\(\mathbf{p} \to -i\hbar\nabla\\)

**Klein-Gordon Equation** :

\\[ \left(\frac{1}{c^2}\frac{\partial^2}{\partial t^2} - \nabla^2 + \frac{m^2c^2}{\hbar^2}\right)\psi = 0 \\]

**Problems** : The Klein-Gordon equation has important limitations. The probability density can become negative, violating the probabilistic interpretation of quantum mechanics. Additionally, the equation is not first-order in the time derivative, unlike the Schrodinger equation. These issues indicate that the Klein-Gordon equation describes spin-0 bosons (scalar fields) rather than fundamental particles like electrons.

## 5.2 Dirac Equation

### 📚 Derivation of the Dirac Equation

We seek a relativistic equation with a first-order time derivative.

**Dirac Equation** :

\\[ i\hbar\frac{\partial \psi}{\partial t} = \left(c\boldsymbol{\alpha}\cdot\hat{\mathbf{p}} + \beta mc^2\right)\psi \\]

**Dirac Matrices** (standard representation):

\\[ \alpha_i = \begin{pmatrix} 0 & \sigma_i \\\ \sigma_i & 0 \end{pmatrix}, \quad \beta = \begin{pmatrix} I & 0 \\\ 0 & -I \end{pmatrix} \\]

Here, \\(\psi\\) is a four-component **spinor**.

### Free Particle Solutions

Plane wave solutions:

\\[ \psi = u(\mathbf{p}) e^{-i(Et - \mathbf{p}\cdot\mathbf{r})/\hbar} \\]

**Positive Energy Solutions** : \\(E = +\sqrt{(pc)^2 + (mc^2)^2}\\) (particle)

**Negative Energy Solutions** : \\(E = -\sqrt{(pc)^2 + (mc^2)^2}\\) (antiparticle)

## 5.3 Antiparticles and Quantum Electrodynamics

### 📚 Dirac Sea and Hole Theory

We postulate the **Dirac sea** , in which all negative energy states are occupied.

A hole in the negative energy state = **particle with positive charge and positive energy** = antiparticle

Antiparticle of the electron (e⁻) = positron (e⁺)

**Pair Creation and Annihilation** :

\\[ \gamma \to e^- + e^+ \quad (E_\gamma > 2m_ec^2) \\]

\\[ e^- + e^+ \to 2\gamma \\]

### Development to Quantum Electrodynamics (QED)

The Dirac equation is fully understood within the framework of **quantum field theory**. **Feynman Diagrams** provide a visual representation of particle interactions, making calculations systematic and intuitive. **Renormalization Theory** addresses the removal of divergent infinities that arise in perturbation calculations. The **Lamb Shift** represents a fine structure correction due to vacuum polarization, and the **Anomalous Magnetic Moment** demonstrates remarkable agreement with experiment through QED corrections, achieving precision exceeding 10 significant digits.

## 5.4 Relativistic Corrections and Fine Structure

### 📚 Fine Structure of the Hydrogen Atom

Non-relativistic energy: \\(E_n = -\frac{13.6 \text{ eV}}{n^2}\\)

**Fine Structure Splitting** (relativistic correction):

\\[ \Delta E_{fs} \sim \alpha^2 E_n \\]

Here, \\(\alpha \approx 1/137\\) is the **fine-structure constant**.

**Spin-Orbit Interaction** :

\\[ H_{SO} = \frac{1}{2m^2c^2}\frac{1}{r}\frac{dV}{dr}\mathbf{L}\cdot\mathbf{S} \\]

It is diagonalized in the eigenstates of the total angular momentum \\(\mathbf{J} = \mathbf{L} + \mathbf{S}\\).

### 🎯 Exercise Problems

  1. **Klein-Gordon Equation** : Find the plane wave solution for a free particle and derive the dispersion relation.
  2. **Properties of Dirac Matrices** : Verify that \\(\\{\alpha_i, \alpha_j\\} = 2\delta_{ij}\\) and \\(\\{\alpha_i, \beta\\} = 0\\).
  3. **Spinor** : Construct the Dirac spinor for a particle at rest and apply the spin projection operator.
  4. **Fine Structure** : Calculate the fine structure splitting of the n=2 level of the hydrogen atom.

## Summary

In this chapter, we learned about relativistic quantum mechanics. The **Klein-Gordon equation** describes scalar fields and spin-0 bosons. The **Dirac equation** governs spin-1/2 fermions through four-component spinors. The concept of **antiparticles** emerges from the physical interpretation of negative energy solutions, leading to pair creation and annihilation processes. **QED** provides the complete quantum field theory framework with Feynman diagrams and renormalization theory. Finally, **fine structure** arises from relativistic corrections and spin-orbit interaction.

This completes the quantum mechanics series. To learn more, proceed to quantum field theory and particle physics.

[← Chapter 4: Perturbation Theory](<chapter-4.html>) [Return to Table of Contents](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
