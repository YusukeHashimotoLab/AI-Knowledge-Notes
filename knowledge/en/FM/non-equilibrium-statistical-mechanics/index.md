---
title: ⚡ Non-Equilibrium Statistical Mechanics
chapter_title: ⚡ Non-Equilibrium Statistical Mechanics
subtitle: Non-Equilibrium Statistical Mechanics for Materials Processes
---

[AI Terakoya Top](<../index.html>)›[Fundamental Mathematics](<../../index.html>)›[Non Equilibrium Statistical Mechanics](<../../FM/non-equilibrium-statistical-mechanics/index.html>)

🌐 EN | [🇯🇵 JP](<../../../jp/FM/non-equilibrium-statistical-mechanics/index.html>) | Last sync: 2025-11-16

[← Back to Fundamental Mathematics Dojo](<../index.html>)

## 🎯 Series Overview

Non-equilibrium statistical mechanics is a theory that microscopically describes diffusion, reaction, and relaxation phenomena in materials processes. In this series, we will learn the Boltzmann equation and H-theorem, Master equation, Langevin equation, Fokker-Planck equation, linear response theory and fluctuation-dissipation theorem, and implement applications to chemical reactions and diffusion processes using Python. 

### Learning Path
    
    
    ```mermaid
    flowchart LR
        A[Chapter 1Boltzmann Equation]
        B[Chapter 2Master Equation]
        C[Chapter 3Langevin Equation]
        D[Chapter 4Fokker-Planck]
        E[Chapter 5Linear Response]
        A --> B --> C --> D --> E
        style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style B fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style C fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style D fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style E fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    ```

### 📋 Learning Objectives

  * Understand the Boltzmann equation and H-theorem, and calculate gas relaxation processes
  * Describe stochastic processes using the Master equation
  * Understand the Langevin equation and Fokker-Planck equation, and simulate Brownian motion
  * Understand linear response theory and the fluctuation-dissipation theorem
  * Implement the dynamics of chemical reactions and diffusion processes in Python

### 📖 Prerequisites

Basic knowledge of statistical mechanics and probability theory is required. It is desirable to understand the basic usage of Python.

Chapter 1

Boltzmann Equation and H-Theorem

Derive the Boltzmann equation that describes the time evolution of distribution functions, and understand the entropy increase law through the H-theorem. Learn the treatment of collision terms and the relaxation time approximation, and numerically simulate the relaxation process of gas molecules using Python. 

Boltzmann Equation H-Theorem Collision Term Relaxation Time Approximation Entropy Increase Distribution Function

💻 7 Code Examples ⏱️ 18-22 min

[Read Chapter 1 →](<chapter-1.html>)

Chapter 2

Master Equation and Stochastic Processes

Derive the Master equation, which forms the basis of probabilistic descriptions, and understand the concepts of transition probability and detailed balance. Implement basic stochastic processes such as random walks and birth-death processes in Python and analyze their statistical properties. 

Master Equation Transition Probability Detailed Balance Random Walk Birth-Death Process Markov Process

💻 7 Code Examples ⏱️ 18-22 min

[Read Chapter 2 →](<chapter-2.html>)

Chapter 3

Langevin Equation and Brownian Motion

Derive the Langevin equation describing the motion of particles in a heat bath, and understand the corresponding Fokker-Planck equation. Implement numerical solutions using the Euler-Maruyama method and verify the statistical properties of Brownian motion (mean square displacement, diffusion coefficient) in Python. 

Langevin Equation Fokker-Planck Equation Brownian Motion Euler-Maruyama Method Diffusion Coefficient Mean Square Displacement

💻 7 Code Examples ⏱️ 18-22 min

[Read Chapter 3 →](<chapter-3.html>)

Chapter 4

Linear Response Theory and Fluctuation-Dissipation Theorem

Learn linear response theory that describes the response of systems to external fields. Understand the Green-Kubo formula and Onsager reciprocity, and derive the fluctuation-dissipation theorem. Implement methods for calculating transport coefficients in Python. 

Linear Response Theory Green-Kubo Formula Fluctuation-Dissipation Theorem Onsager Reciprocity Transport Coefficients Correlation Function

💻 7 Code Examples ⏱️ 18-22 min

[Read Chapter 4 →](<chapter-4.html>)

Chapter 5

Dynamics of Chemical Reactions and Diffusion Processes

Learn applications of non-equilibrium statistical mechanics to materials processes. Understand the theoretical framework of chemical reaction kinetics, solutions to diffusion equations, crystal growth dynamics, and phase separation kinetics, and implement practical materials process simulations in Python. 

Chemical Reaction Kinetics Diffusion Equation Crystal Growth Phase Separation Kinetics Cahn-Hilliard Equation Materials Processes

💻 7 Code Examples ⏱️ 18-22 min

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
  * **stochastic**

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

  * Quantum Transport
  * Kinetic Theory
  * Fluctuation Theorems

### Related Series

Expand your knowledge with related topics:

  * Classical Statistical Mechanics
  * Probability and Stochastic Processes

### Practical Projects

Apply your skills to hands-on projects:

  * Brownian motion simulation
  * Stochastic differential equations
  * Transport coefficient calculation

### ⚠️ Disclaimer

  * This content is provided for educational and informational purposes only and does not constitute professional advice.
  * All content and code examples are provided "AS IS" without warranty of any kind, either express or implied, including but not limited to warranties of accuracy, reliability, completeness, or fitness for a particular purpose.
  * The use of external links, data, tools, and libraries is at your own discretion and risk. The authors and contributors are not responsible for their availability, functionality, or suitability.
  * In no event shall the content creator or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising from the use of this content, to the maximum extent permitted by law.
  * Accuracy of information is not guaranteed. Content may contain errors or become outdated.
  * Content is licensed under Creative Commons BY 4.0 unless otherwise specified. Please refer to the license for usage terms.
