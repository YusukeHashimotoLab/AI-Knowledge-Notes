---
title: Introduction to Materials Properties Series
chapter_title: Introduction to Materials Properties Series
subtitle: From Solid State Electronic Theory to First-Principles Calculations - A Path to Computational Materials Science
difficulty: Intermediate to Advanced
code_examples: 50
version: 1.0
created_at: 2025-10-28
---

## Series Overview

This series is an introductory course for theoretically understanding the physical properties of materials (electrical, magnetic, optical, and thermal properties) and predicting them using first-principles calculations. You will learn from the fundamentals of solid state electronic theory to practical materials property calculations using DFT (Density Functional Theory), all while working with Python.

### Learning Flow
    
    
    ```mermaid
    graph LR
        A[Chapter 1Solid State Theory Basics] --> B[Chapter 2Crystal Field Theory]
        B --> C[Chapter 3DFT Introduction]
        C --> D[Chapter 4Electrical & Magnetic Properties]
        D --> E[Chapter 5Optical & Thermal Properties]
        E --> F[Chapter 6Practical Workflows]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style F fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## Series Structure

Chapter 1

Fundamentals of Solid State Electronic Theory

Learn the free electron model, Fermi energy, and basics of band structure. Understand the differences between metals, semiconductors, and insulators from their electronic states, and visualize band diagrams using Python. 

⏱️ 30-35 minutes 💻 9 code examples 📊 Intermediate

[Start Learning →](<chapter-1.html>)

Chapter 2

Crystal Field Theory and Electronic States

Learn about d-orbital splitting in transition metal compounds. Understand crystal field splitting, the Jahn-Teller effect, and ligand field theory, and calculate crystal field energies using Python. 

⏱️ 25-30 minutes 💻 8 code examples 📊 Intermediate

[Start Learning →](<chapter-2.html>)

Chapter 3

Introduction to First-Principles Calculations (DFT Basics)

Learn the fundamentals of Density Functional Theory (DFT). Understand the Hohenberg-Kohn theorems, Kohn-Sham equations, and exchange-correlation functionals (LDA, GGA, hybrid), and practice using ASE and Pymatgen. 

⏱️ 35-40 minutes 💻 12 code examples 📊 Advanced

[Start Learning →](<chapter-3.html>)

Chapter 4

Electrical and Magnetic Properties

Learn about electrical conduction (Drude model), the Hall effect, magnetism (ferromagnetism and antiferromagnetism), and spin-orbit interactions. Calculate and visualize magnetic properties using Python. 

⏱️ 30-35 minutes 💻 9 code examples 📊 Intermediate to Advanced

[Start Learning →](<chapter-4.html>)

Chapter 5

Optical and Thermal Properties

Learn about light absorption, band gaps, refractive indices, phonons, thermal conductivity, and thermoelectric properties. Calculate and visualize optical spectra and phonon DOS using Python. 

⏱️ 30-35 minutes 💻 9 code examples 📊 Intermediate to Advanced

[Start Learning →](<chapter-5.html>)

Chapter 6

Practical: Materials Property Calculation Workflows

Practice the complete workflow from structure optimization to DFT calculation to property analysis using Si, GaN, Fe, and BaTiO₃ as examples. Learn practical best practices from convergence testing to post-processing. 

⏱️ 40-45 minutes 💻 10 code examples 📊 Advanced

[Start Learning →](<chapter-6.html>)

## Learning Objectives

By completing this series, you will acquire the following skills and knowledge:

### Basic Understanding

  * ✅ Understand and explain concepts of the free electron model, band structure, and density of states
  * ✅ Understand d-orbital splitting by crystal field theory and explain the colors of transition metal compounds
  * ✅ Understand the basic principles of DFT (Hohenberg-Kohn, Kohn-Sham)
  * ✅ Explain the physical mechanisms of electrical conduction, magnetism, and optical properties

### Practical Skills

  * ✅ Plot band structures, DOS, and phonon DOS using Python
  * ✅ Manipulate crystal structures and execute DFT calculations using ASE and Pymatgen
  * ✅ Create VASP input files (INCAR, POSCAR, KPOINTS)
  * ✅ Extract and analyze property values from DFT calculation results

### Applied Capabilities

  * ✅ Predict properties of new materials using first-principles calculations
  * ✅ Properly perform convergence tests (k-mesh, cutoff energy)
  * ✅ Validate calculation results by comparing with experimental data
  * ✅ Apply computational materials science in Materials Informatics research

## Recommended Learning Patterns

### Pattern 1: For Beginners - Sequential Learning (7 days)

  * Day 1: Chapter 1 (Solid State Theory Basics)
  * Day 2: Chapter 2 (Crystal Field Theory)
  * Day 3: Chapter 3 (DFT Basics) - Important chapter, study thoroughly
  * Day 4: Chapter 3 continued + exercises
  * Day 5: Chapter 4 (Electrical & Magnetic Properties)
  * Day 6: Chapter 5 (Optical & Thermal Properties)
  * Day 7: Chapter 6 (Practical Workflows) + comprehensive review

### Pattern 2: For Intermediate Learners - Intensive Learning (3 days)

  * Day 1: Chapters 1-2 (Solid Physics Basics)
  * Day 2: Chapter 3 (DFT Introduction) - intensive study
  * Day 3: Chapters 4-6 (Property Calculations and Practice)

### Pattern 3: Practice-Focused - DFT Calculations Central (1 day)

  * Chapter 3: Thoroughly study DFT basics (2 hours)
  * Chapter 6: Focus on practical workflows (2-3 hours)
  * Chapters 1,2,4,5: Reference as needed

## Prerequisites

Subject | Required Level | Description  
---|---|---  
**Quantum Mechanics** | Undergraduate level | Basics of wave functions, Schrödinger equation, operators  
**Solid State Physics** | Introductory level | Basics of crystal structure, reciprocal lattice, Brillouin zone  
**Materials Science** | Introductory level | Completion of Materials Science Introduction series recommended  
**Python** | Intermediate level | Experience using NumPy, Matplotlib, classes and functions  
**Linux** | Basic level | Terminal operations, basics of shell scripting  
  
## Python Libraries and Tools Used

Major libraries and tools used in this series:

### Essential Libraries

  * **NumPy** : Numerical computation, matrix operations
  * **Matplotlib** : 2D graph creation (band diagrams, DOS)
  * **SciPy** : Scientific computing, numerical integration, optimization
  * **Pandas** : Data processing, table management

### Computational Materials Science Libraries

  * **ASE (Atomic Simulation Environment)** : Atomic structure manipulation, calculation setup
  * **Pymatgen** : Crystal structure analysis, DFT I/O, Materials Project integration
  * **GPAW** : Python implementation of DFT code (for demo purposes, optional)

### DFT Calculation Software (explanation only, execution optional)

  * **VASP** : Commercial first-principles calculation code (input examples provided)
  * **Quantum ESPRESSO** : Open-source DFT code

## FAQ - Frequently Asked Questions

### Q1: Is it okay if my knowledge of quantum mechanics is insufficient?

While understanding basic quantum mechanics (wave functions, eigenvalue problems) is desirable, necessary concepts will be explained as needed. However, knowledge of quantum mechanics will deepen your understanding, especially for Chapter 3 on DFT.

### Q2: Do I need to actually execute DFT calculations?

It's not essential for understanding the theory, but hands-on practice will significantly deepen your understanding. Chapter 6 teaches how to set up calculations using ASE and Pymatgen. If you don't have commercial software like VASP, you can still learn up to creating input files.

### Q3: How can this series be applied to Materials Informatics?

Predicting material properties is crucial in MI. First-principles calculations are the foundation for building property databases. The property calculation workflows learned in this series directly connect to "descriptor design" and "high-throughput calculations" in MI.

### Q4: Can I learn without VASP?

Yes, you can. VASP input examples are provided, but code execution is optional. You can acquire many practical skills without VASP, such as structure manipulation using ASE and Pymatgen, and plotting band diagrams and DOS.

### Q5: How much time will it take?

Approximately 180-220 minutes for all 6 chapters. Chapter 3 (DFT Basics) and Chapter 6 (Practice) particularly take time. If you're executing code while learning, you'll need additional time. Please proceed at your own pace.

## Key Learning Points

  * **Understanding Mathematics and Physics** : Emphasis on physically interpreting the meaning of mathematical equations
  * **Interpreting Calculation Results** : Understanding what can be learned from band diagrams and DOS, and their physical meanings
  * **Practical Workflows** : Acquiring calculation procedures that can be immediately used in research
  * **Troubleshooting** : Learning how to handle non-convergence and abnormal results

## Next Steps

After completing this series, we recommend the following advanced learning:

  * Introduction to Machine Learning Potentials (ML-IP)
  * High-Throughput Materials Calculations
  * Time-Dependent DFT (Precise Optical Property Calculations)
  * Topological Materials Property Calculations
  * Materials Informatics Practical Series
