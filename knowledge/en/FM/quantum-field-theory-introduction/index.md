---
title: ⚛️ Introduction to Quantum Field Theory
chapter_title: ⚛️ Introduction to Quantum Field Theory
subtitle: Introduction to Quantum Field Theory for Materials Science
---

[AI Terakoya Top](<../index.html>)›[Fundamentals Mathematics](<../../index.html>)›[Quantum Field Theory](<../../FM/quantum-field-theory-introduction/index.html>)

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/index.html>) | Last sync: 2025-11-16

[← Back to Fundamentals Mathematics](<../index.html>)

## 🎯 Series Overview

Quantum Field Theory (QFT) is the foundational theoretical framework for particle physics and many-body systems. This series covers field quantization basics from canonical quantization and path integrals, free field theory (scalar fields, Dirac fields, electromagnetic fields), interacting field theory, Feynman diagram techniques, and renormalization theory through a combination of theory and numerical simulations (Python/NumPy/SymPy). This is an advanced course designed for applications in condensed matter physics, solid state physics, and many-body problems in materials science. 

### Learning Path
    
    
    ```mermaid
    flowchart LR
        A[Chapter 1Relativistic QM]
        B[Chapter 2Canonical Quantization]
        C[Chapter 3Feynman Diagrams]
        D[Chapter 4Renormalization]
        E[Chapter 5QFT Applications]
        A --> B --> C --> D --> E
        style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style B fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style C fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style D fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style E fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    ```

### 📋 Learning Objectives

  * Understand and implement field quantization through canonical quantization and path integral formalism
  * Derive Green functions and propagators for free scalar fields, Dirac fields, and electromagnetic fields
  * Calculate scattering amplitudes and S-matrix for interacting fields
  * Systematically execute perturbative calculations using Feynman diagram techniques
  * Understand the basics of renormalization theory and UV divergence treatment, and apply them to many-body problems in materials science

### 📖 Prerequisites

Thorough understanding of quantum mechanics (Schrödinger equation, second quantization), analytical mechanics (Lagrangian formalism, Hamiltonian formalism, Noether's theorem), special relativity (Lorentz transformation, Minkowski spacetime), and complex analysis (residue theorem) is required. Experience with scientific computing in Python (NumPy, SciPy, SymPy) is recommended. 

Chapter 1

Field Quantization and Canonical Formalism

Learn quantization from classical field theory, canonical quantization of Klein-Gordon and Dirac fields, equal-time commutation and anticommutation relations, and construction of Fock space. Implement description of multi-particle states using creation and annihilation operators and normal ordering concept, and understand applications to excited states in materials (phonons, magnons). 

Canonical Quantization Klein-Gordon Equation Dirac Equation Fock Space Creation Annihilation Operators

💻 8 Code Examples ⏱️ 40-50 minutes

[Read Chapter 1 →](<chapter-1.html>)

Chapter 2

Free Field Theory and Propagators

Learn Fourier expansion and mode analysis of free scalar fields, Dirac fields, and electromagnetic fields, and derivation of Feynman propagators and Green functions. Implement causality and iε prescription, Wick rotation and path integrals, Källén-Lehmann representation, and understand applications to lattice vibrations and electron-phonon coupling. 

Propagators Green Functions Wick Rotation Path Integrals Mode Expansion

💻 8 Code Examples ⏱️ 40-50 minutes

[Read Chapter 2 →](<chapter-2.html>)

Chapter 3

Interacting Fields and S-Matrix Theory

Learn interaction picture and Dyson series, S-matrix definition and causality, Gell-Mann-Low theorem and LSZ reduction formula. Implement calculation of scattering amplitudes and differential cross sections, T-products and time-ordered products, Wick's theorem, and understand applications to electron-electron interactions and Coulomb scattering. 

S-Matrix Dyson Series LSZ Formula Wick's Theorem Scattering Amplitudes

💻 8 Code Examples ⏱️ 40-50 minutes

[Read Chapter 3 →](<chapter-3.html>)

Chapter 4

Feynman Diagram Techniques

Learn Feynman diagram rules and topology, connected and disconnected diagrams, correspondence between vertices and propagators, and regularization of loop integrals. Implement φ⁴ theory, one-loop calculations in quantum electrodynamics (QED), vacuum polarization and vertex corrections, and understand applications to screening effects in solids and RPA approximation. 

Feynman Diagrams Vertex Corrections Loop Integrals Regularization Vacuum Polarization

💻 8 Code Examples ⏱️ 40-50 minutes

[Read Chapter 4 →](<chapter-4.html>)

Chapter 5

Renormalization Theory and Effective Theory

Learn the origin and classification of UV divergences, dimensional regularization and minimal subtraction, renormalization group equations and β functions, and construction of effective field theory. Implement Wilson's renormalization group, critical phenomena and phase transitions, and Landau-Ginzburg theory, and understand applications to phase transitions and critical exponents in materials. 

Renormalization Dimensional Regularization Renormalization Group Effective Theory Critical Phenomena

💻 8 Code Examples ⏱️ 40-50 minutes

[Read Chapter 5 →](<chapter-5.html>)

## 📚 Recommended Learning Paths

### Pattern 1: Beginner - Theory and Practice Balanced (5-7 days)

  * Day 1: Chapter 1 (Fundamentals)
  * Day 2: Chapter 2 (Core Concepts)
  * Day 3: Chapter 3 (Advanced Theory)
  * Day 4: Chapter 4 (Applications)
  * Day 5: Chapter 5 (Python Practice) + Review

### Pattern 2: Intermediate - Fast Track (3 days)

  * Day 1: Chapters 1-2 (Fundamentals and Core Concepts)
  * Day 2: Chapters 3-4 (Advanced Theory and Applications)
  * Day 3: Chapter 5 (Practice) + All Exercises

### Pattern 3: Topic-Focused - Computational Skills (1 day)

  * Focus: Code examples from all chapters
  * Execute all Python implementations
  * Modify parameters and analyze results
  * Light theory review as needed

## 🎯 Overall Learning Outcomes

Upon completing this series, you will achieve:

### Knowledge Level

  * ✅ Understand fundamental theoretical concepts and mathematical formulations
  * ✅ Explain relationships between key equations and physical phenomena
  * ✅ Interpret results in context of real-world applications
  * ✅ Connect concepts across chapters systematically

### Practical Skills

  * ✅ Implement algorithms from scratch using Python
  * ✅ Utilize NumPy, SciPy, and Matplotlib effectively
  * ✅ Visualize complex data and results
  * ✅ Debug and optimize numerical code

### Application Ability

  * ✅ Apply theoretical concepts to practical problems
  * ✅ Design computational experiments
  * ✅ Analyze and interpret simulation results
  * ✅ Extend learned methods to new domains

## 🛠️ Technologies and Tools Used

### Main Libraries

  * **numpy**
  * **scipy**
  * **matplotlib**
  * **sympy**
  * **qutip**

### Development Environment

  * **Python** : 3.8 or higher
  * **Jupyter Notebook** : Interactive development and visualization
  * **IDE** : VSCode, PyCharm, or similar

### Recommended Tools

  * Google Colab (cloud-based, no setup required)
  * Anaconda Distribution (complete environment)
  * Git (version control for exercises)

## 🚀 Next Steps

### Deep Dive Learning

For more advanced study in this field:

  * Advanced QFT
  * Gauge Theory
  * String Theory

### Related Series

Expand your knowledge with related topics:

  * Quantum Mechanics
  * Particle Physics

### Practical Projects

Apply your skills to hands-on projects:

  * Scattering amplitude calculation
  * Feynman diagram generator
  * Field simulation

### ⚠️ Disclaimer

  * This content is provided for educational and informational purposes only and does not constitute professional advice.
  * All content and code examples are provided "AS IS" without warranty of any kind, either express or implied, including but not limited to warranties of accuracy, reliability, completeness, or fitness for a particular purpose.
  * The use of external links, data, tools, and libraries is at your own discretion and risk. The authors and contributors are not responsible for their availability, functionality, or suitability.
  * In no event shall the content creator or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising from the use of this content, to the maximum extent permitted by law.
  * Accuracy of information is not guaranteed. Content may contain errors or become outdated.
  * Content is licensed under Creative Commons BY 4.0 unless otherwise specified. Please refer to the license for usage terms.
