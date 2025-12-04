---
title: Introduction to Computational Materials Science Series v1.0
chapter_title: Introduction to Computational Materials Science Series v1.0
subtitle: From Quantum Mechanics to Molecular Dynamics - A Complete Guide to Theory and Practice
reading_time: 20-30 minutes
difficulty: Intermediate to Advanced
code_examples: 5
version: 1.0
created_at: 2025-10-17
---

# Introduction to Computational Materials Science Series v1.0

**From Quantum Mechanics to Molecular Dynamics - A Complete Guide to Theory and Practice**

## Series Overview

This series is a comprehensive 5-chapter educational content designed to systematically teach you from the fundamental theories of computational materials science to practical simulation techniques. It covers essential computational methods for modern materials research, including quantum mechanics-based first-principles calculations, molecular dynamics simulations, and phonon calculations.

**Key Features:**

  * ✅ **Integration of Theory and Practice** : From physics and chemistry theory to actual code implementation
  * ✅ **Systematic Structure** : Progressive learning from Quantum Mechanics → DFT → MD → Phonons → Practical Applications
  * ✅ **Executable Code** : 30-35 practical Python/shell script examples
  * ✅ **Comprehensive Coverage of Major Tools** : Hands-on mastery of ASE, GPAW, LAMMPS, Phonopy, and more

**Total Learning Time** : 115-140 minutes (including code execution and exercises)

\---

## How to Use This Series

### Recommended Learning Path
    
    
    ```mermaid
    flowchart TD
        A[Chapter 1: Quantum Mechanics and Solid State Physics] --> B[Chapter 2: Density Functional Theory DFT]
        B --> C[Chapter 3: Molecular Dynamics MD]
        C --> D[Chapter 4: Phonon Calculations]
        D --> E[Chapter 5: Integration of First-Principles and ML]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
    ```

**For those with physics/chemistry background (undergraduate 3rd-4th year, graduate students):**

  * Chapter 1 (Foundation Review) → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5
  * Time Required: 115-140 minutes

**For experienced DFT/MD users (strengthening practical skills):**

  * Chapter 2 (Section 2.3 onwards) → Chapter 4 → Chapter 5
  * Time Required: 60-80 minutes

**For machine learning researchers (applying to materials science):**

  * Chapter 1 → Chapter 2 (Sections 2.1-2.2) → Chapter 5 (intensive study)
  * Time Required: 50-70 minutes

\---

## Chapter Details

### [Chapter 1: Fundamentals of Quantum Mechanics and Solid State Physics](<chapter-1.html>)

**Level** : Intermediate

**Reading Time** : 25-30 minutes

**Code Examples** : 6-7

#### Learning Content

  1. **Basic Principles of Quantum Mechanics**

\- Schrödinger Equation

\- Wave Functions and Hamiltonian

\- Born-Oppenheimer Approximation

  1. **Electronic Structure of Atoms and Molecules**

\- Solution for Hydrogen Atom

\- Multi-electron System Problems

\- Importance of Electron Correlation

  1. **Quantum Mechanics of Solids**

\- Periodic Boundary Conditions

\- Bloch's Theorem

\- Band Structure Fundamentals

  1. **Practical Exercises**

\- Solving Schrödinger Equation for Hydrogen Atom with Python

\- Energy Level Calculations for Quantum Wells

\- Simple Band Structure Visualization

#### Learning Objectives

  * ✅ Explain the physical meaning of the Schrödinger equation
  * ✅ Understand the importance of the Born-Oppenheimer approximation
  * ✅ Understand the concept of band structure in solids
  * ✅ Execute basic quantum mechanics calculations with Python

**[Read Chapter 1 →](<chapter-1.html>)**

\---

### [Chapter 2: Introduction to Density Functional Theory (DFT)](<chapter-2.html>)

**Level** : Intermediate to Advanced

**Reading Time** : 30-35 minutes

**Code Examples** : 7-8

#### Learning Content

  1. **Fundamental Theory of DFT**

\- Hohenberg-Kohn Theorems

\- Kohn-Sham Equations

\- Exchange-Correlation Functionals (LDA, GGA)

  1. **Practical Aspects of DFT Calculations**

\- Basis Functions (Plane Waves, Atomic Orbitals)

\- k-point Sampling

\- Convergence Criteria

  1. **Practical Application with ASE + GPAW**

\- Environment Setup

\- Structure Optimization

\- Band Gap Calculation

\- Density of States (DOS) Calculation

  1. **Limitations of DFT and Remedies**

\- Band Gap Problem

\- van der Waals Interactions

\- Strongly Correlated Systems

#### Learning Objectives

  * ✅ Explain the basic principles of DFT
  * ✅ Understand the meaning of Kohn-Sham equations
  * ✅ Execute DFT calculations using ASE and GPAW
  * ✅ Understand DFT limitations and make appropriate method selections

**[Read Chapter 2 →](<chapter-2.html>)**

\---

### [Chapter 3: Molecular Dynamics (MD) Simulation](<chapter-3.html>)

**Level** : Intermediate

**Reading Time** : 25-30 minutes

**Code Examples** : 6-7

#### Learning Content

  1. **Fundamental Theory of MD**

\- Newton's Equations of Motion

\- Force Fields and Potentials

\- Time Integration Algorithms (Verlet, Leap-frog)

  1. **Statistical Ensembles**

\- NVE (Microcanonical)

\- NVT (Canonical) - Nosé-Hoover Thermostat

\- NPT (Isothermal-Isobaric) - Parrinello-Rahman

  1. **Practical Application with LAMMPS**

\- Creating Input Files

\- MD Simulation of Water Molecules

\- Diffusion Coefficient Calculation

\- Radial Distribution Function (RDF) Analysis

  1. **Ab Initio MD (AIMD)**

\- DFT + MD Combination

\- Car-Parrinello MD

\- Born-Oppenheimer MD

#### Learning Objectives

  * ✅ Explain the basic principles of MD simulations
  * ✅ Understand the differences between statistical ensembles
  * ✅ Execute basic MD simulations with LAMMPS
  * ✅ Understand the characteristics and applications of Ab Initio MD

**[Read Chapter 3 →](<chapter-3.html>)**

\---

### [Chapter 4: Phonon Calculations and Thermodynamic Properties](<chapter-4.html>)

**Level** : Intermediate to Advanced

**Reading Time** : 20-25 minutes

**Code Examples** : 5-6

#### Learning Content

  1. **Theory of Lattice Vibrations**

\- Harmonic Approximation

\- Dynamical Matrix

\- Phonon Dispersion Relations

  1. **Phonon Calculations with Phonopy**

\- Finite Displacement Method

\- Force Constant Calculation

\- Phonon Band Structure

  1. **Calculation of Thermodynamic Properties**

\- Free Energy

\- Specific Heat

\- Thermal Expansion Coefficient

\- Debye Temperature

  1. **Practical Project**

\- Complete Phonon Calculation for Si

\- Temperature Dependence Analysis

\- Thermal Conductivity Estimation

#### Learning Objectives

  * ✅ Explain the relationship between lattice vibrations and phonons
  * ✅ Execute phonon calculations with Phonopy
  * ✅ Understand the principles of thermodynamic property calculations
  * ✅ Predict material properties from phonon calculation results

**[Read Chapter 4 →](<chapter-4.html>)**

\---

### [Chapter 5: Integration of First-Principles Calculations and Machine Learning](<chapter-5.html>)

**Level** : Advanced

**Reading Time** : 15-20 minutes

**Code Examples** : 5-6

#### Learning Content

  1. **Machine Learning Potentials (MLP)**

\- Gaussian Approximation Potential (GAP)

\- Neural Network Potential (NNP)

\- Moment Tensor Potential (MTP)

  1. **Data Generation Strategies**

\- Active Learning

\- Data Extraction from DFT Calculations

\- Efficient Sampling

  1. **Practical: Training NNP**

\- Building NNP with ASE and AMP

\- Preparing Training Data

\- Evaluating Potentials

  1. **Future Perspectives**

\- Universal Machine Learning Potentials

\- Foundation Models for Materials

\- Applications to Autonomous Experiments

#### Learning Objectives

  * ✅ Understand the concept of machine learning potentials
  * ✅ Train MLP from DFT data
  * ✅ Understand the importance of Active Learning
  * ✅ Keep up with the latest ML + computational materials science trends

**[Read Chapter 5 →](<chapter-5.html>)**

\---

## Overall Learning Outcomes

Upon completing this series, you will acquire the following skills and knowledge:

### Theoretical Knowledge (Understanding)

  * ✅ Understand the foundations of quantum mechanics and the Schrödinger equation
  * ✅ Explain DFT principles (Hohenberg-Kohn theorems, Kohn-Sham equations)
  * ✅ Understand the statistical mechanical foundation of MD simulations
  * ✅ Explain the relationship between phonon calculations and thermodynamic properties

### Practical Skills (Doing)

  * ✅ Execute DFT calculations with ASE/GPAW
  * ✅ Execute MD simulations with LAMMPS
  * ✅ Execute phonon calculations with Phonopy
  * ✅ Analyze and visualize computational results with Python
  * ✅ Train machine learning potentials

### Application Ability (Applying)

  * ✅ Calculate electronic structure of materials and predict properties
  * ✅ Evaluate thermodynamic properties from theoretical calculations
  * ✅ Select appropriate computational methods (DFT vs MD vs MLP)
  * ✅ Develop research strategies combining calculations and experiments

\---

## Recommended Learning Patterns

### Pattern 1: Complete Mastery from Basics (For Graduate Students)

**Target Audience** : Graduate students learning computational materials science for the first time

**Duration** : 2-3 weeks

**Approach** :
    
    
    Week 1:

  * Day 1-2: Chapter 1 (Review quantum mechanics fundamentals)
  * Day 3-4: Chapter 2 (DFT theory, first half)
  * Day 5-7: Chapter 2 (DFT practice, ASE/GPAW exercises)

Week 2:

  * Day 1-3: Chapter 3 (MD theory and practice)
  * Day 4-5: Chapter 4 (Phonon calculations)
  * Day 6-7: Review exercises from each chapter

Week 3:

  * Day 1-4: Chapter 5 (ML + computational materials science)
  * Day 5-7: Independent project (practical calculations on materials of interest)

**Deliverables** :

  * DFT calculation results for 3+ types of materials
  * MD simulation report
  * Phonon calculation project

### Pattern 2: Rapid Practical Skills Acquisition (For Postdocs/Researchers)

**Target Audience** : Researchers who know theory but lack implementation experience

**Duration** : 1 week

**Approach** :
    
    
    Day 1: Chapter 2 (ASE/GPAW environment setup and basic calculations)

Day 2-3: Chapter 2 (Practical projects: Band structure, DOS)

Day 4: Chapter 3 (LAMMPS practice)

Day 5: Chapter 4 (Phonopy practice)

Day 6-7: Chapter 5 (MLP training and Active Learning)

**Deliverables** :

  * Code and documentation published on GitHub
  * Computational workflow automation scripts

### Pattern 3: Targeted Learning (Mastering Specific Methods)

**Target Audience** : Researchers wanting to master specific computational methods

**Duration** : Flexible

**Selection Examples** :

  * **Want to learn DFT calculations** → Chapter 1 (Section 1.3) + Chapter 2 (Complete)
  * **Want to learn MD simulations** → Chapter 3 (Complete) + Chapter 5 (Section 5.3)
  * **Want to learn phonon calculations** → Chapter 1 (Section 1.3) + Chapter 4 (Complete)
  * **Want to learn machine learning potentials** → Chapter 5 (Complete) + Chapter 2 (Section 2.3)

\---

## FAQ (Frequently Asked Questions)

### Q1: What prerequisite knowledge is needed to learn this series?

**A** : The following knowledge is assumed:

  * **Physics** : University 1st-2nd year level quantum mechanics, statistical mechanics, solid state physics
  * **Chemistry** : University 1st-2nd year level physical chemistry, inorganic chemistry
  * **Mathematics** : Linear algebra, differential equations, Fourier transforms
  * **Programming** : Python basics (variables, functions, lists, NumPy/Matplotlib)

If uncertain, check the "Prerequisite Knowledge Checklist" in each chapter.

### Q2: Do I need to actually run the code?

**A** : **Strongly recommended**. Computational materials science is a practical skill. Not just theory, but actually running code and seeing results leads to deep understanding. If environment setup is difficult, you can start with Google Colab (free).

### Q3: Can I learn without commercial software (VASP, etc.)?

**A** : **Yes, absolutely**. This series primarily uses open-source tools (ASE, GPAW, LAMMPS, Phonopy). These are freely available and have sufficient functionality for both academic research and industrial applications. While VASP and Quantum ESPRESSO are mentioned, you can learn without them.

### Q4: How much computational resources are needed?

**A** :

  * **Minimum Requirements** : Laptop (8GB+ RAM, 4-core CPU) can run basic exercises
  * **Recommended Environment** : Desktop PC (16GB+ RAM, 8-core CPU) for comfortable execution
  * **Large-scale Calculations** : Use university supercomputers, cloud HPC (AWS, Google Cloud)

Basic exercises are sufficient with a laptop.

### Q5: How long does it take to master?

**A** : Depends on study time and goals:

  * **Understanding theory** : 1-2 weeks (reading all 5 chapters)
  * **Basic computational skills** : 2-4 weeks (executing all code examples)
  * **Research-level skills** : 2-3 months (including independent projects)
  * **Expert level** : 1-2 years (continuous practice and paper writing)

### Q6: What's the difference between DFT and MD? When should I use which?

**A** :

**DFT (Density Functional Theory)** :

  * Quantum mechanical calculation of electronic states
  * Band structure, band gap, magnetism, etc.
  * Computational cost: High (up to several hundred atoms)
  * Use cases: Electronic properties, chemical reactions, structure optimization

**MD (Molecular Dynamics)** :

  * Classical mechanics simulation of atomic motion
  * Diffusion, phase transitions, interface phenomena, etc.
  * Computational cost: Medium to high (up to millions of atoms possible)
  * Use cases: Thermodynamics, dynamic processes, large-scale systems

**How to Choose** :

  * Electronic structure is important → DFT
  * Time evolution or large-scale systems → MD
  * Both accuracies needed → Ab Initio MD (explained in Chapter 3)

### Q7: Can I write papers just from this series?

**A** : This series focuses on **building foundations and acquiring practical skills**. Paper writing additionally requires:

  1. Deep dive into specific research themes (3-6 months)
  2. Thorough survey of prior research
  3. Creation of original research outcomes
  4. Mastery of paper writing techniques

After completing this series, we recommend advancing your research while consulting with your advisor or collaborators.

\---

## Next Steps

### Recommended Actions After Series Completion

**Immediate (Within 1-2 weeks):**

  1. ✅ Upload computational scripts and notebooks to GitHub
  2. ✅ Present learned content at lab seminar
  3. ✅ Practice calculations on materials related to your research theme

**Short-term (1-3 months):**

  1. ✅ Read 10 specialized papers in depth (_Physical Review B_ , _Computational Materials Science_)
  2. ✅ Contribute to open-source projects (ASE, Phonopy, etc.)
  3. ✅ Conference presentation (aim for poster presentation)
  4. ✅ Develop computational workflow automation scripts

**Medium-term (3-6 months):**

  1. ✅ Compile calculation results from independent research project into paper
  2. ✅ Learn advanced computational methods (GW approximation, DMFT, QMC, etc.)
  3. ✅ Oral presentation at domestic conference
  4. ✅ Apply for supercomputer usage proposals

**Long-term (1+ years):**

  1. ✅ Present at international conferences (APS March Meeting, MRS)
  2. ✅ Submit and publish peer-reviewed papers
  3. ✅ Establish as doctoral dissertation theme
  4. ✅ Teach computational materials science to next generation of students

\---

## Learning Resources

### Recommended Textbooks

**DFT and First-Principles Calculations** :

  1. Richard M. Martin, "Electronic Structure: Basic Theory and Practical Methods"
  2. Shinji Tsuneyuki, "Computational Physics" (Iwanami Shoten, in Japanese)
  3. David S. Sholl & Janice A. Steckel, "Density Functional Theory: A Practical Introduction"

**Molecular Dynamics** :

  1. Daan Frenkel & Berend Smit, "Understanding Molecular Simulation"
  2. J. M. Haile, "Molecular Dynamics Simulation"
  3. Susumu Okazaki, "Molecular Simulation" (Iwanami Shoten, in Japanese)

**Solid State Physics and Quantum Mechanics** :

  1. Neil W. Ashcroft & N. David Mermin, "Solid State Physics"
  2. Tatsumi Kurosawa, "Solid State Physics" (Shokabo, in Japanese)
  3. J. J. Sakurai, "Modern Quantum Mechanics"

### Online Resources

**Official Documentation** :

  * [ASE Documentation](<https://wiki.fysik.dtu.dk/ase/>)
  * [GPAW Documentation](<https://wiki.fysik.dtu.dk/gpaw/>)
  * [LAMMPS Documentation](<https://docs.lammps.org/>)
  * [Phonopy Documentation](<https://phonopy.github.io/phonopy/>)

**Tutorials and Courses** :

  * Materials Project Tutorials
  * ASE Workshop Materials
  * LAMMPS Tutorials by Sandia National Lab

**Community** :

  * [Materials Project Forum](<https://matsci.org/>)
  * [LAMMPS Mailing List](<https://www.lammps.org/mail.html>)
  * [ASE Users Mailing List](<https://listserv.fysik.dtu.dk/mailman/listinfo/ase-users>)

\---

## Feedback and Support

### About This Series

This series was created as part of the MI Knowledge Hub project under Dr. Yusuke Hashimoto at Tohoku University.

**Creation Date** : October 17, 2025

**Version** : 1.0

### We Welcome Your Feedback

To improve this series, we welcome your feedback:

  * **Typos, errors, technical mistakes** : Report via GitHub repository Issues
  * **Improvement suggestions** : New topics, code examples you'd like to see, etc.
  * **Questions** : Difficult sections, areas needing additional explanation
  * **Success stories** : Projects using what you learned from this series

**Contact** : yusuke.hashimoto.b8@tohoku.ac.jp

\---

## License and Terms of Use

This series is published under the **CC BY 4.0** (Creative Commons Attribution 4.0 International) license.

**Permitted:**

  * ✅ Free viewing and downloading
  * ✅ Educational use (classes, study groups, etc.)
  * ✅ Modifications and derivative works (translations, summaries, etc.)

**Conditions:**

  * 📌 Attribution to the author is required
  * 📌 If modified, indicate the modifications
  * 📌 For commercial use, contact in advance

Details: [CC BY 4.0 License Full Text](<https://creativecommons.org/licenses/by/4.0/>)

\---

## Let's Get Started!

Are you ready? Start with Chapter 1 and begin your journey into the world of computational materials science!

**[Chapter 1: Fundamentals of Quantum Mechanics and Solid State Physics →](<chapter-1.html>)**

\---

**Update History**

  * **2025-10-17** : v1.0 Initial release

\---

**Your journey into computational materials science begins here!**
