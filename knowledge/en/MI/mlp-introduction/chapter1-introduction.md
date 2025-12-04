---
title: "Chapter 1: Why Do We Need Machine Learning Potentials (MLPs)?"
chapter_title: "Chapter 1: Why Do We Need Machine Learning Potentials (MLPs)?"
---

# Chapter 1: Why Do We Need Machine Learning Potentials (MLPs)?

Understand the value of MLPs in achieving both quantum accuracy and computational speed by positioning them relative to DFT and classical MD. Quickly grasp which challenges they can address.

**💡 Supplement:** "DFT accuracy, approximated through shortcuts for speed." The ground truth is DFT, but practical calculations run with MLP—this is the complementary division of labor.

## Learning Objectives

By reading this chapter, you will master the following:  
\- Understand the historical evolution of molecular simulation (from the 1950s to present)  
\- Explain the limitations and challenges of empirical force fields and first-principles calculations (DFT)  
\- Understand the technical and societal background that necessitates MLPs  
\- Learn about the power of MLPs through concrete examples of catalytic reaction simulations

* * *

## 1.1 History of Molecular Simulation: 70 Years of Evolution

To understand the properties of matter and design new materials and drugs, scientists have computationally calculated **how molecules and atoms move**. This technology is called **molecular simulation**.

### 1950s: Birth of Molecular Simulation

**Origins of Molecular Dynamics**

In 1957, Bern and Alder[1] performed the first computer simulation of liquid argon behavior. This marked the beginning of Molecular Dynamics (MD).

  * **Computational target** : 32 argon atoms
  * **Computer used** : UNIVAC (state-of-the-art at the time)
  * **Computation time** : Several hours
  * **Potential** : Simple Lennard-Jones potential

    
    
    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    

This simple equation approximated the interaction energy between two atoms.

**Limitations of the Lennard-Jones Potential**

  * Cannot describe chemical bond formation/breaking
  * Does not consider electronic states
  * Parameters (ε, σ) must be determined experimentally for each element

Nevertheless, for simple systems like noble gases, it reproduced experimental results surprisingly well.

### 1970s: Applications to Biomolecules

**Beginning of Protein Simulation**

In 1977, McCammon et al.[2] performed the first MD simulation of a protein (bovine pancreatic trypsin inhibitor, 58 amino acid residues).

  * **Computational target** : Approximately 500 atoms
  * **Simulation time** : 9.2 picoseconds (9.2 × 10⁻¹² seconds)
  * **Actual computation time** : Several days
  * **Potential** : **Empirical force fields** such as AMBER and CHARMM

**Emergence of Empirical Force Fields**

More complex force fields were developed to handle proteins and organic molecules:

  * **AMBER** (Assisted Model Building with Energy Refinement, 1981)[3]
  * **CHARMM** (Chemistry at HARvard Macromolecular Mechanics, 1983)[4]
  * **GROMOS** (GROningen MOlecular Simulation, 1987)

These force fields include terms such as:
    
    
    E_total = E_bond + E_angle + E_dihedral + E_non-bonded
    
    E_bond = Σ k_b(r - r_0)²  (Bond stretching)
    E_angle = Σ k_θ(θ - θ_0)²  (Angle bending)
    E_dihedral = Σ V_n[1 + cos(nφ - γ)]  (Dihedral rotation)
    E_non-bonded = Σ [4ε((σ/r)¹² - (σ/r)⁶) + q_i q_j/(4πε_0 r)]  (Electrostatic interactions)
    

**Problem** : These parameters (k_b, r_0, ε, σ, etc.) must be **determined from experimental data or quantum chemical calculations** , requiring hundreds of parameters.

### 1980-1990s: Practical Implementation of First-Principles Calculations

**Rise of Density Functional Theory (DFT)**

DFT (Density Functional Theory), proposed by Hohenberg and Kohn in 1964 and Kohn and Sham in 1965[5,6], became practical in the 1980s onwards due to improvements in computational power.

**Revolutionary aspects of DFT** :  
\- Calculate molecular and solid properties without parameters  
\- First-principles calculations based on quantum mechanics  
\- Accurately describe chemical bond formation and breaking

**Computational burden** :  
\- Computational complexity: O(N³) (N: number of electrons)  
\- One step for a 100-atom system: Minutes to hours  
\- MD (hundreds of thousands of steps) is virtually impossible

**1998 Nobel Prize in Chemistry**

Walter Kohn and John Pople received the Nobel Prize in Chemistry for "development of computational methods in quantum chemistry." DFT became a standard tool in chemistry and materials science.

### 2000s: Ab Initio Molecular Dynamics (AIMD)

**Car-Parrinello Molecular Dynamics (1985)**

Car and Parrinello[7] developed **ab initio MD (AIMD)** by combining DFT and MD.

  * Execute DFT calculation at each MD step
  * Can simulate chemical reactions
  * **Extremely computationally expensive**

**Typical AIMD constraints (2000s)** :  
\- System size: Around 10² atoms  
\- Time scale: Picoseconds (10⁻¹² seconds)  
\- Computational resources: Supercomputers

**Problem** : Many important phenomena such as catalytic reactions, protein folding, and material fracture occur on **nanosecond to microsecond (10⁻⁹ to 10⁻⁶ seconds)** scales. AIMD cannot reach these timescales.

### 2010s: Emergence of Machine Learning Potentials (MLPs)

**Behler-Parrinello Neural Network Potential (2007)**

Jörg Behler and Michele Parrinello[8] proposed a method to **learn DFT-accuracy potentials using neural networks**.

**Revolutionary idea** :  
1\. Calculate energies for thousands to tens of thousands of atomic configurations with DFT  
2\. Neural network learns the "potential energy surface" from this data  
3\. Execute MD with the learned potential → **μs-scale simulation with DFT accuracy**

**Advantages of MLPs** :  
\- Accuracy: DFT level (can describe chemical reactions)  
\- Speed: Comparable to empirical force fields (10⁴-10⁶ times faster)  
\- Generality: Applicable to any system with available data

* * *

## 1.2 Limitations of Conventional Methods: Empirical Force Fields vs DFT

Molecular simulation broadly had two approaches. Each has serious limitations.

### Limitation 1: Empirical Force Fields - Lack of Generality

**Problems with Empirical Force Fields**

  1. **Cannot describe chemical reactions**  
\- Cannot describe bond formation/breaking  
\- Catalytic reactions, enzymatic reactions, material fracture cannot be computed

  2. **Parameters are not transferable**  
\- Parameters determined for one system cannot be used for another  
\- Example: Water force field parameters lose accuracy when applied to ice

  3. **Cannot apply to new materials**  
\- Cannot be used for elements or structures without parameters  
\- Not useful when designing new materials

**Specific example: CO₂ reduction reaction on copper catalyst surface**

Catalytic reactions that convert carbon dioxide (CO₂) into useful chemicals are key to addressing climate change.

With conventional empirical force fields:
    
    
    CO₂ + * → CO₂*  (Adsorption)
    CO₂* + H⁺ + e⁻ → ?  (Reaction initiation)
    

This "?" part (C-O bond breaking and new bond formation) cannot be described. This is because empirical force fields have **fixed bond topology**.

### Limitation 2: DFT - Computational Cost Barrier

**Computational Complexity of DFT**

Typical DFT calculation (plane-wave basis, PBE functional) computational complexity:  
\- Scaling: O(N³) (N: number of electrons ≈ number of atoms)  
\- 100 atoms: About 1 hour/step (on a supercomputer)  
\- MD requires 10⁵-10⁷ steps

**Specific numerical example**

System size | Number of atoms | DFT computation time (1 step) | Time needed for MD (10⁶ steps)  
---|---|---|---  
Small molecule | 10 | 1 minute | About 2 years  
Medium scale | 100 | 1 hour | About 11,000 years  
Large scale | 1,000 | 1 day | About 2.7 million years  
  
※ Estimated values on a typical supercomputer node (64 cores)

**Inaccessible time and spatial scales**

What DFT can actually reach:  
\- Number of atoms: Around 10²  
\- Time: Picoseconds (10⁻¹² seconds)

However, scales where important phenomena occur:  
\- Catalytic reactions: Nanoseconds to microseconds (10⁻⁹ to 10⁻⁶ seconds)  
\- Protein motion: Microseconds to milliseconds (10⁻⁶ to 10⁻³ seconds)  
\- Material fracture: Nanoseconds to microseconds  
\- Crystal growth: Microseconds and beyond

**The gap is more than 10⁶ times** (one million times).

### Limitation 3: The "Unfortunate" Trade-off Between Both

Conventional molecular simulation faced a **dilemma between accuracy and computational cost**.

**Illustration of the trade-off**
    
    
    ```mermaid
    flowchart LR
        A[Empirical force fieldsAMBER, CHARMM] --> B{Accuracy vs Speed}
        B -->|Fast10⁶ atoms, μs| C[No generalityChemical reactions ×]
    
        D[DFTAb initio MD] --> B
        B -->|High accuracyChemical reactions ○| E[Extremely slow10² atoms, ps]
    
        F[MLPNeural networks] --> G{Both achieved}
        G -->|Fast10⁴-10⁶ atoms| H[DFT accuracyChemical reactions ○]
    
        style C fill:#ffcccc
        style E fill:#ffcccc
        style H fill:#ccffcc
    ```

**Dilemma faced by researchers**

Scenario 1: Catalyst material design  
\- Use empirical force fields → Fast but cannot describe chemical reactions → **Unusable**  
\- Use DFT → High accuracy but can only calculate 10 catalyst atoms → **Impractical**

Scenario 2: Elucidating drug binding mechanisms  
\- Use empirical force fields → Fast but cannot describe bond formation/breaking → **Insufficient**  
\- Use DFT → Cannot calculate entire protein (thousands of atoms) → **Not applicable**

**Conclusion** : Conventional methods could not address **scientifically most important problems** (large-scale, long-time simulations including chemical reactions).

* * *

## 1.3 Case Study: Difficulty of Catalytic Reaction Simulation

As a specific example, let's consider the **CO₂ reduction reaction on copper (Cu) catalyst surface**. This is an important reaction that converts greenhouse gas CO₂ into useful chemicals (such as ethanol)[9].

### Reaction Overview

**Electrochemical CO₂ reduction reaction**
    
    
    CO₂ + 6H⁺ + 6e⁻ → CH₃OH + H₂O  (Methanol production)
    CO₂ + 12H⁺ + 12e⁻ → C₂H₅OH + 3H₂O  (Ethanol production)
    

**Complexity of the reaction mechanism** :  
1\. CO₂ adsorbs on copper surface  
2\. Hydrogen atoms (H _) are generated on the surface  
3\. CO₂_ is reduced stepwise (_COOH →_ CO → _CHO → ...)  
4\. Two CO_ molecules couple to form C₂ chemicals (C-C bond formation)  
5\. Final product desorbs

This process goes through **more than 10 intermediates** , each requiring overcoming reaction barriers.

### Difficulties with Conventional Methods

**Approach 1: Empirical Force Field (ReaxFF)**

ReaxFF[10] was developed as an empirical force field capable of describing chemical reactions.

**Attempt** :  
\- Model Cu catalyst surface with 100 water molecules and CO₂ molecules  
\- 1 nanosecond MD simulation  
\- Computation time: Several days (on GPU)

**Result** :  
\- CO₂ adsorption was observed  
\- However, **reduction reaction does not occur**  
\- Reason: ReaxFF parameters not optimized for this specific reaction system  
\- Determining new parameters requires large amounts of DFT data

**Issues** :  
\- Parameter fitting takes months to years  
\- Even after fitting, prediction accuracy is insufficient  
\- Different catalysts (Ag, Au, etc.) require parameter refitting

**Approach 2: DFT (ab initio MD)**

**Attempt** :  
\- Cu(111) surface slab (96 Cu atoms)  
\- Place CO₂, H₂O, CO, COOH intermediates on surface  
\- Explore reaction pathway with ab initio MD  
\- Computational resources used: 1,000 nodes on the "Fugaku" supercomputer

**Result** :  
\- 10 picosecond simulation takes **1 week**  
\- Reaction is not observed on this timescale  
\- Reason: Typical timescale of CO₂ reduction reaction is **nanoseconds to microseconds**  
\- **10⁶ times (one million times) insufficient**

**Computation time estimate** :  
\- Time needed for 1 microsecond simulation: About **100,000 weeks = 2000 years**

**Issues** :  
\- Cannot reach realistic timescales  
\- Cannot observe entire reaction mechanism  
\- Statistical sampling (many reaction events) impossible

### Solution with MLP (2018 onwards)

**Research example: MLP-MD of Cu catalyst CO₂ reduction using SchNet and DimeNet**

Cheng et al. (2020, Nature Communications)[11] achieved the following using MLP:

**Procedure** :  
1\. DFT data collection: Calculate energies and forces for about 5,000 atomic configurations (Cu surface + CO₂ + H₂O + intermediates)  
\- Computation time: About 1 week on supercomputer

  2. MLP training: Train SchNet model[12] (graph neural network)  
\- Training time: A few hours on 1 GPU  
\- Energy prediction accuracy: Mean absolute error (MAE) < 1 meV/atom (DFT accuracy)

  3. MLP-MD simulation: Molecular dynamics simulation with trained MLP  
\- System size: 200 Cu atoms + 50 water molecules + CO₂  
\- Simulation time: 1 microsecond  
\- Actual computation time: **1 day** on 1 GPU

**Achievements** :  
\- Observed reaction pathway: CO₂ → _COOH →_ CO → _CHO → CH₃OH  
\- Statistical sampling of reaction barriers  
\- Elucidated C-C bond formation mechanism  
\- __Reached timescales impossible with conventional DFT_ *

### Comparison Table: Conventional Methods vs MLP

Metric | Empirical force field  
(ReaxFF) | DFT  
(ab initio MD) | MLP  
(SchNet) | Improvement ratio  
---|---|---|---|---  
**Accuracy** | Low-Medium  
(Requires parameter tuning) | High  
(First-principles) | High  
(DFT accuracy) | DFT-level  
**Computational speed** | Fast  
10⁻⁶ sec/step | Extremely slow  
1-10 hours/step | Fast  
10⁻³ sec/step | **10⁶× faster than DFT**  
**Accessible timescale** | Nanoseconds-Microseconds | Picoseconds | Nanoseconds-Microseconds | **10⁶× longer**  
**System size** | 10⁴-10⁶ atoms | 10² atoms | 10³-10⁴ atoms | **10-100× larger**  
**Chemical reaction description** | Limited | Accurate | Accurate | DFT-level  
**Generality** | Low (system-specific tuning) | High | High (data-dependent) | DFT-level  
**Data preparation time** | Months (parameters) | None | 1 week (DFT calculation) | Essentially zero  
  
**Conclusion** : MLP achieved the "best of both worlds" by combining **DFT accuracy with empirical force field speed**.

* * *

## 1.4 Conventional Methods vs MLP: Workflow Comparison

As seen in the catalytic reaction simulation example, conventional methods had serious constraints. Let's compare entire research workflows.

### Workflow Comparison Diagram
    
    
    ```mermaid
    flowchart TD
        subgraph "Conventional method: Empirical force fields"
            A1[Research start] -->|1-2 months| A2[Survey existing force fields]
            A2 -->|1 week| A3{Existing force fieldsufficient accuracy?}
            A3 -->|Yes 10%| A4[MD simulation]
            A3 -->|No 90%| A5[Force field parameterrefitting]
            A5 -->|3-12 months| A6[Large-scale DFT calculations]
            A6 -->|1-2 months| A7[Parameter optimization]
            A7 -->|1 week| A8{Accuracy validation}
            A8 -->|No 30%| A5
            A8 -->|Yes 70%| A4
            A4 -->|1 week-1 month| A9[Result analysis]
    
            style A5 fill:#ffcccc
            style A6 fill:#ffcccc
            style A7 fill:#ffcccc
        end
    
        subgraph "Conventional method: DFT"
            B1[Research start] -->|1 week| B2[Model system construction50-100 atoms]
            B2 -->|Few hours| B3[DFT calculation test]
            B3 -->|1-2 weeks| B4[10 ps AIMD]
            B4 -->|1 day| B5{Reaction observed?}
            B5 -->|No 95%| B6[Longer time needed?]
            B6 -->|Impossible| B7[Manual reaction pathway searchNEB method, etc.]
            B7 -->|2-4 weeks| B8[Reaction barrier calculation]
            B5 -->|Yes 5%| B8
            B8 -->|1 week| B9[Result analysis]
    
            style B6 fill:#ffcccc
            style B7 fill:#ffcccc
        end
    
        subgraph "MLP method"
            C1[Research start] -->|1 week| C2[DFT data collection3,000-10,000 configs]
            C2 -->|1-3 days| C3[MLP model trainingGPU]
            C3 -->|Few hours| C4{Accuracy validation}
            C4 -->|No 20%| C5[Additional data collection]
            C5 -->|1-2 days| C2
            C4 -->|Yes 80%| C6[MLP-MD simulation1 μs]
            C6 -->|1-7 days| C7[Reaction observation & statistical analysis]
            C7 -->|1 day| C8{Goal achieved?}
            C8 -->|No 30%| C9[Active LearningAdd data]
            C9 -->|1-2 days| C3
            C8 -->|Yes 70%| C10[Paper writing]
    
            style C6 fill:#ccffcc
            style C7 fill:#ccffcc
            style C10 fill:#ccffcc
        end
    ```

### Quantitative Comparison

Metric | Empirical force field | DFT | MLP | MLP improvement  
---|---|---|---|---  
**Preparation period** | 3-12 months  
(Parameter tuning) | 1-2 weeks  
(Model construction) | 1-2 weeks  
(Data collection & training) | Same as DFT  
**One project duration** | 6-18 months | 3-6 months  
(With constraints) | 1-2 months | **3-9× faster**  
**Accessible phenomena** | Large-scale, long-time  
(Low accuracy) | Small-scale, short-time  
(High accuracy) | Large-scale, long-time  
(High accuracy) | **Both achieved**  
**Application to new systems** | Difficult  
(Requires retuning) | Easy | Easy  
(Retraining only) | DFT-level  
**Success rate** | 30-50%  
(System-dependent) | 80%  
(Within constraints) | 70-80% | High  
  
### Timeline comparison example: Cu catalyst CO₂ reduction research

**With empirical force field (ReaxFF)** :  
\- Existing parameter survey: 1 month  
\- Parameters found inadequate  
\- DFT reference data generation: 2 months  
\- Parameter fitting: 2-3 months  
\- Validation shows insufficient accuracy  
\- Refitting: 1-2 months  
\- MD simulation: 2 weeks  
\- Analysis: 2 weeks  
\- **Total: About 9 months**

**With DFT (AIMD)** :  
\- Model construction: 1 week  
\- 10 ps AIMD: 1 week  
\- Reaction not observed  
\- NEB method reaction pathway search: 3 weeks  
\- Multiple pathway calculations: 2 weeks  
\- Analysis: 1 week  
\- **Total: About 8 weeks**  
\- **Problem: Dynamic behavior, statistical sampling not possible**

**With MLP (SchNet/DimeNet)** :  
\- DFT data collection: 1 week (parallel calculation)  
\- MLP training: 1 day  
\- Accuracy validation: Half day  
\- 1 μs MLP-MD simulation: 3 days  
\- Reaction observation & statistical analysis: 3 days  
\- Accuracy improvement with Active Learning: 2 days  
\- Paper figure creation: 2 days  
\- **Total: About 2.5 weeks**  
\- **3× faster than DFT, 15× faster than empirical force field**

* * *

## 1.5 Column: A Computational Chemist's Day (2000 vs 2025)

Let's see how research practices have changed through specific stories.

### 2000: Struggles during the DFT Golden Era

**Professor Yamada (42 years old, national university) - One week**

**Monday**  
\- 9:00 - Arrive at lab. Check results of DFT calculation (CO adsorption on Cu surface) submitted last Friday.  
\- 10:00 - Notice it terminated with error. Convergence criteria too strict.  
\- 11:00 - Adjust parameters and resubmit. Expected to take 3 days this time.  
\- Afternoon - Read papers. Student guidance.

**Tuesday-Thursday**  
\- Waiting for calculation to complete.  
\- During this time, prepare other calculations (different project) and write papers.  
\- "Want to calculate larger systems, but computation time is too long..."

**Friday**  
\- 9:00 - Calculation completed. Finally got results.  
\- 10:00 - Obtained energy for one CO adsorption configuration.  
\- 11:00 - Want to calculate another adsorption site. Will wait another 3 days...  
\- Afternoon - Student looks anxious: "We only calculated one configuration this week. At this pace, won't finish doctoral dissertation."

**One week's achievements** : One atomic configuration DFT calculation

**One month's achievements** : About 5-10 configurations

**One year's achievements** : About 50-100 configurations, 1-2 papers

**Worries** :  
\- "Want to understand reaction mechanism, but dynamics simulation is computationally impossible"  
\- "Want to calculate larger catalyst clusters (100+ atoms), but won't finish in a week"  
\- "Want statistical sampling, but don't have computational resources"

### 2025: Efficiency in the MLP Era

**Associate Professor Sato (38 years old, same university) - One week**

**Monday**  
\- 9:00 - Arrive at lab. Check MLP-MD (1 microsecond simulation) results executed on GPU cluster over the weekend.  
\- 9:30 - **Confirm 3 reactions occurred!**  
\- Observed pathway: CO₂ → _COOH →_ CO → *CHO → CH₃OH  
\- Check trajectory with Visualizer  
\- 10:00 - Automatic reaction pathway analysis with Python script. Automatically detect transition states.  
\- 11:00 - Research meeting. Discuss reaction mechanism with students.  
\- "The timing of this C-O bond breaking seems important"  
\- "Then let's do DFT calculation of that configuration to examine electronic structure in detail"

**Monday afternoon**  
\- 14:00 - Detailed electronic structure analysis of reaction intermediates with DFT.  
\- Automatically extract "interesting configurations" with MLP  
\- DFT calculation of only those configurations → Complete in a few hours  
\- 16:00 - Add new configurations to DFT database for Active Learning.  
\- 17:00 - Retrain MLP model (30 minutes on 1 GPU). Accuracy improved.

**Tuesday**  
\- 9:00 - Execute additional 1 microsecond simulation with improved MLP (auto-execute overnight).  
\- Morning - Create paper figures. Reaction pathway diagrams, energy profiles, snapshots.  
\- Afternoon - Zoom meeting with collaborators. Suggest to experimental group: "These intermediates seem important".

**Wednesday-Friday**  
\- Additional simulation result analysis  
\- Paper writing  
\- Prepare another project (screening new catalyst materials)  
\- Read papers on new MLP methods (E(3) equivariant graph neural networks)

**Friday afternoon**  
\- 15:00 - Review this week's achievements  
\- **Completed 3 microseconds of MLP-MD simulation**  
\- Elucidated reaction mechanism  
\- Completed paper draft  
\- 16:00 - "Let's try the same reaction with different catalysts (Au, Ag) next week. With MLP, can compare in one week"

**One week's achievements** : 3 microseconds simulation, reaction mechanism elucidation, paper draft completed

**One month's achievements** : Multiple catalyst system comparison, 2-3 paper drafts

**One year's achievements** : 10-15 papers, multiple parallel projects

**Joy** :  
\- "Thanks to MLP, can see phenomena on timescales I couldn't imagine"  
\- "Calculations are fast, so easy to try different things. Can immediately test new ideas"  
\- "Students' thesis topics increased. Everyone enjoys research"

### Points of Change

Item | 2000 (DFT) | 2025 (MLP) | Change  
---|---|---|---  
**Daily calculation volume** | 1 configuration (DFT) | 1 μs simulation  
(10⁶ steps) | **10⁶× more**  
**Weekly achievements** | 1-2 configurations | Reaction mechanism elucidation | Qualitative change  
**Annual papers** | 1-2 papers | 10-15 papers | **5-10× more**  
**Calculation waiting time** | 3-7 days/calculation | Few hours to 1 day | **Stress greatly reduced**  
**Ease of trial and error** | Difficult | Easy | Research quality improved  
**Student satisfaction** | Low  
(Calculations slow) | High  
(Results come quickly) | Motivation improved  
  
**Important point** : MLP **liberated researchers' creativity**. Time spent waiting for calculations decreased, allowing focus on scientific insights and new ideas.

* * *

## 1.6 Why "Now" for MLP: Three Tailwinds

The MLP concept has existed since the 2000s, but practical implementation began in earnest **after 2015**. Why "now"?

### Tailwind 1: Dramatic Progress in Machine Learning Technology

**Deep Learning Revolution (2012)**

AlexNet[13] dominated the ImageNet image recognition competition, launching the deep learning boom.

**Spillover to chemistry and materials science** :  
\- **2015** : Schütt et al. proposed SchNet[12] (graph neural network)  
\- Represent molecules as "graphs"  
\- Learn atomic interactions  
\- Guarantee rotation and translation invariance

  * **2018** : Klicpera et al. proposed DimeNet[14]
  * Also considers bond angles
  * Further improved accuracy

  * **2021** : Batzner et al. proposed NequIP[15], Batatia et al. proposed MACE[16]

  * Implement E(3) equivariance (rotation covariance)
  * Incorporate physical laws into machine learning
  * Training data efficiency greatly improved (same accuracy with 1/10 the data)

**Proliferation of PyTorch and TensorFlow**

  * Frameworks enabling researchers to implement neural networks themselves
  * Easy GPU computation usage
  * Open source, accessible to anyone

### Tailwind 2: Democratization of GPU Computing

**GPU Performance Improvements**

Year | GPU | Performance (TFLOPS) | Price  
---|---|---|---  
2010 | NVIDIA GTX 480 | 1.3 | $500  
2015 | NVIDIA GTX 980 Ti | 5.6 | $650  
2020 | NVIDIA RTX 3090 | 35.6 | $1,500  
2024 | NVIDIA H100 | 989 | $30,000  
(Research use)  
  
**Proliferation of cloud GPUs**

  * Google Colab: Use GPU for free (research/education)
  * AWS, Google Cloud, Azure: Rent GPUs by the hour
  * Access latest GPUs for a few to tens of dollars per hour

**Result** : Anyone, anywhere can conduct large-scale machine learning training and MLP-MD simulation.

### Tailwind 3: Open Data and Open Source Culture

**Large-scale DFT databases**

Database | Data volume | Application  
---|---|---  
**Materials Project** | 140,000+ | Crystalline materials  
**QM9** | 134,000 | Small molecules  
**ANI-1x/2x** | 5 million | Organic molecules  
**OC20/OC22** | 1 million+ | Catalytic reactions  
  
These data are **freely downloadable**. Anyone can train MLPs.

**Open-source MLP software**

Software | Developer | Features  
---|---|---  
**SchNetPack** | TU Berlin | SchNet implementation  
**DimeNet** | TU Munich | Considers angular information  
**NequIP** | Harvard | E(3) equivariant  
**MACE** | Cambridge | High efficiency  
**DeePMD-kit** | Peking University | For large-scale systems  
  
All are published on **GitHub** and can be used and modified by anyone.

**Culture of publishing code and data with papers**

  * After 2020, top journals (Nature, Science, PRL, etc.) emphasize **reproducibility**
  * Encourage/mandate code and data publication upon paper submission
  * Result: Research acceleration, reduced duplication

### Tailwind 4: Rising Social Urgency

**Climate Change and Energy Issues**

  * **2015 Paris Agreement** : Limit global warming to within 2°C
  * Material development for catalysts (CO₂ reduction, hydrogen production), batteries, solar cells urgently needed
  * Conventional methods too slow for development speed

**Accelerating Drug Development**

  * COVID-19 pandemic (2020) made rapid drug development a global issue
  * MLP beginning to be used for protein-drug interaction simulation

**Global research investment**

  * **USA** : NSF, DOE invest tens of millions of dollars annually in MLP research
  * **Europe** : Multiple machine learning × materials science projects in Horizon Europe
  * **China** : Promoting AI × materials science as national strategy
  * **Japan** : JST, NEDO supporting materials informatics research

**Conclusion** : MLP is needed and realizable now precisely because technical maturity, democratization of computational resources, open science, and social necessity are **simultaneously fulfilled**.

* * *

## 1.7 Introduction to Major MLP Methods (Overview)

We'll learn in detail from Chapter 2 onwards, but here's a brief introduction to representative MLP methods.

### 1\. Behler-Parrinello Neural Network Potential (2007)[8]

**Features** :  
\- First practical MLP method  
\- Describe local environment of each atom with "Symmetry Functions"  
\- Predict energy for each atom with neural network  
\- Total energy = Σ(atomic energy)

**Advantages** : Simple and easy to understand  
**Disadvantages** : Manual design of Symmetry Functions, requires much training data

**Representative applications** : Water, silicon, organic molecules

### 2\. Graph Neural Networks (2017 onwards)

**SchNet (2017)[12]**  
\- Represent molecules as graphs (atoms=nodes, bonds=edges)  
\- Learn graphs with continuous-filter convolution  
\- Message passing according to distance

**DimeNet (2020)[14]**  
\- Also considers bond angle information  
\- Directional message passing  
\- Higher accuracy than SchNet

**Advantages** : No manual feature design needed, end-to-end learning  
**Disadvantages** : Requires relatively much training data

**Representative applications** : Organic molecules, catalytic reactions, drug design

### 3\. Equivariant Neural Networks (2021 onwards)

**NequIP (2021)[15]**  
\- Implements E(3) equivariance (covariant to rotations)  
\- Propagate messages as tensor fields  
\- High training data efficiency

**MACE (2022)[16]**  
\- Message-passing + Atomic Cluster Expansion  
\- Efficiently learn high-order many-body interactions  
\- Currently highest accuracy

**Advantages** : Extremely high data efficiency (high accuracy with thousands of configurations), incorporates physical laws  
**Disadvantages** : Implementation somewhat complex

**Representative applications** : Large-scale material simulation, complex chemical reactions

### 4\. Comparison Table

Method | Year | Data efficiency | Accuracy | Speed | Implementation difficulty  
---|---|---|---|---|---  
Behler-Parrinello | 2007 | Low | Medium | High | Medium  
SchNet | 2017 | Medium | High | High | Low  
DimeNet | 2020 | Medium | High | Medium | Medium  
NequIP | 2021 | **High** | **High** | Medium | High  
MACE | 2022 | **Highest** | **Highest** | Medium | High  
  
**Future learning** :  
\- Chapter 2: Mathematical foundations of these methods  
\- Chapter 3: SchNet implementation and hands-on  
\- Chapter 4: NequIP/MACE details and application examples

* * *

## 1.8 Chapter Summary

### What We Learned

  1. **History of molecular simulation**  
\- 1950s: Simple Lennard-Jones potential for noble gases  
\- 1970s: Empirical force fields for proteins and organic molecules  
\- 1990s: Practical implementation of DFT (first-principles calculations)  
\- 2000s: ab initio MD (AIMD) but extremely slow  
\- 2007: Proposal of Behler-Parrinello MLP  
\- 2015 onwards: Rapid development of deep learning and MLP

  2. **Limitations of conventional methods**  
\- **Empirical force fields** : Cannot describe chemical reactions, no parameter generality  
\- **DFT (AIMD)** : Extremely slow (limited to 10² atoms, picoseconds)  
\- **Trade-off** : Dilemma between accuracy vs speed

  3. **MLP revolution**  
\- Achieves **DFT accuracy** while maintaining **empirical force field speed**  
\- 10⁴-10⁶× speedup  
\- Large-scale, long-time simulations including chemical reactions now possible

  4. **Specific catalytic reaction example (Cu surface CO₂ reduction)**  
\- Empirical force field: Months for parameter adjustment, insufficient accuracy  
\- DFT: 10 ps limit, reaction observation impossible (needs 1 μs but takes 2000 years)  
\- MLP: 1 μs simulation completed in 1 day, reaction mechanism elucidated

  5. **Why "now" for MLP**  
\- Machine learning evolution (SchNet, NequIP, MACE, etc.)  
\- GPU computing democratization (free on Colab, affordable on cloud)  
\- Open data and open source culture  
\- Social urgency (climate change, energy, drug development)

### Key Points

  * MLP combines **DFT accuracy** with **empirical force field speed**
  * Simultaneously achieves **large-scale systems** and **long timescales** including chemical reactions
  * **Observation of dynamic reaction mechanisms** previously impossible now possible
  * Dramatically reduces researchers' **calculation waiting time** , liberating creativity

### To the Next Chapter

Chapter 2 will explore the **mathematical foundations** of MLP in detail:  
\- What is a potential energy surface  
\- Energy learning with neural networks  
\- Importance of symmetry and equivariance  
\- Evolution from Behler-Parrinello to the latest MACE

Additionally, we'll practice simple MLP training using Python.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

In the history of molecular simulation, create a table summarizing how "accuracy," "speed," and "generality" differ among the three methods: empirical force fields, DFT, and MLP.

Hint Compare from three perspectives: accuracy (can describe chemical reactions), speed (computation time per step), and generality (ease of application to new systems).  Sample Answer | Method | Accuracy | Speed | Generality | |------|------|------|--------| | **Empirical force fields**  
(AMBER, CHARMM, ReaxFF) | **Low-Medium**  
\- Chemical reactions: Only ReaxFF possible (requires tuning)  
\- No quantum effects | **Fast**  
\- 10⁻⁶ sec/step  
\- 10⁶ atoms, μs scale possible | **Low**  
\- Parameter tuning needed per system  
\- Difficult to apply to new materials | | **DFT**  
(ab initio MD) | **High**  
\- Can describe chemical reactions  
\- Accurately describes quantum effects | **Extremely slow**  
\- 1-10 hours/step  
\- 10² atoms, ps scale limit | **High**  
\- No parameters needed  
\- Applicable to any system | | **MLP**  
(SchNet, NequIP, MACE) | **High**  
\- DFT accuracy  
\- Can describe chemical reactions | **Fast**  
\- 10⁻³ sec/step  
\- 10³-10⁴ atoms, μs scale possible | **High**  
\- Applicable with training data  
\- Data collection takes 1-2 weeks | **Conclusion**: MLP achieved characteristics previously impossible: combining **DFT accuracy with empirical force field speed**. 

### Problem 2 (Difficulty: medium)

Explain why 1 microsecond ab initio MD simulation with DFT is "practically impossible," estimating computation time. Assume DFT calculation for one step takes 1 hour for a 100-atom system.

Hint How many steps are needed for 1 microsecond MD? (Typical timestep is 1 femtosecond = 10⁻¹⁵ seconds)  Sample Answer **Calculation**: 1\. **Required number of steps**: \- 1 microsecond = 10⁻⁶ seconds \- Timestep = 1 femtosecond = 10⁻¹⁵ seconds \- Required steps = 10⁻⁶ ÷ 10⁻¹⁵ = **10⁹ steps** (1 billion steps) 2\. **Computation time**: \- 1 step = 1 hour \- Total computation time = 10⁹ steps × 1 hour = **10⁹ hours** \- = 10⁹ ÷ 24 ÷ 365 = **About 114,000 years** 3\. **Even with parallelization**: \- Even with 1000-node parallelization on best supercomputers \- 114,000 years ÷ 1000 = **About 114 years** **Conclusion**: \- 1 microsecond simulation with DFT is **practically impossible with current computers** \- This is known as the "timescale gap" problem, a serious constraint \- MLP bridges this gap by achieving 10⁴-10⁶× speedup while maintaining DFT accuracy 

### Problem 3 (Difficulty: hard)

For catalytic reaction simulation (Cu surface CO₂ reduction), explain how total research project duration differs between conventional methods (empirical force field, DFT) and MLP, assuming specific workflows.

Hint Consider preparation period, simulation execution time, analysis time, and response time when difficulties arise for each method.  Sample Answer **With empirical force field (ReaxFF)**: 1\. **Preparation period (3-6 months)**: \- Existing parameter survey: 2 weeks \- Found parameters inadequate for Cu-C-O-H system \- DFT reference data generation: 2 months (200-500 configurations) \- Parameter fitting: 2-3 months \- Validation shows insufficient accuracy \- Refitting: 1-2 months 2\. **Simulation (1-2 weeks)**: \- 1 nanosecond MD: 1 week (using GPU) \- Multiple conditions (temperature, composition): Additional 1 week 3\. **Analysis (2 weeks)**: \- Reaction pathway identification \- Energy analysis 4\. **Problem occurrence (2-4 months)**: \- Predicted reaction pathway incorrect when verified with DFT \- Parameter readjustment: 2-4 months **Total: 6-12 months** **Problem**: Large accuracy uncertainty \--- **With DFT (ab initio MD)**: 1\. **Preparation period (1-2 weeks)**: \- Model system construction: 100-atom Cu(111) slab \- Convergence test: Several days 2\. **Simulation (2-4 weeks)**: \- 10 ps AIMD: 1 week (supercomputer) \- Reaction not observed \- Try multiple initial configurations: Additional 2 weeks 3\. **Alternative approach (4-8 weeks)**: \- Manually estimate reaction pathway \- Transition state search with NEB (Nudged Elastic Band) method \- Each pathway calculation: 2-3 weeks \- Multiple pathway comparison: 2-3 weeks 4\. **Analysis (1-2 weeks)**: \- Energy profile creation \- Electronic structure analysis **Total: 2-3 months** **Problem**: Cannot observe dynamic behavior, statistical sampling impossible \--- **With MLP (SchNet/NequIP)**: 1\. **Preparation period (1-2 weeks)**: \- DFT data collection: 5,000-10,000 configurations (parallel calculation 5-7 days) \- Active Sampling (automatically extract important configurations): 2 days \- MLP training: 1 day on 1 GPU \- Accuracy validation: Half day 2\. **Simulation (3-7 days)**: \- 1 microsecond MLP-MD: 3 days (1 GPU) \- Multiple conditions (temperature, composition): Additional 3 days (parallel execution) 3\. **Analysis (3-5 days)**: \- Automatic reaction event detection: 1 day \- Reaction pathway & statistical analysis: 2 days \- Detailed DFT analysis of important configurations: 1-2 days 4\. **Improvement cycle (2-3 days, if needed)**: \- Active Learning additional data collection: 1-2 days \- Model retraining: Half day **Total: 2.5-4 weeks** **Advantages**: Dynamic reaction observation possible, statistically significant results, flexible trial and error \--- **Comparison table**: | Method | Project duration | Success certainty | Information obtained | |------|----------------|------------|------------| | Empirical force field | 6-12 months | Low-Medium (Accuracy uncertain) | Dynamic behavior (but low accuracy) | | DFT | 2-3 months | High (Within constraints) | Static reaction pathways only | | MLP | 2.5-4 weeks | High | Dynamic behavior + statistics + high accuracy | **Conclusion**: \- MLP obtains **most comprehensive information** in the **shortest time** \- **3-5× faster** than DFT, **3-10× faster** than empirical force field \- Additionally achieves both accuracy and dynamic behavior 

* * *

## 1.10 Data Licensing and Reproducibility

To ensure reproducibility of research outcomes and case studies introduced in this chapter, we explicitly document related data and tool information.

### 1.10.1 Datasets Mentioned in This Chapter

Dataset | Description | License | Application  
---|---|---|---  
**MD17** | Molecular dynamics trajectories of small molecules (10 types) | CC0 1.0 (Public Domain) | Standard MLP learning benchmark  
**OC20** | Catalytic adsorption systems (1.3M configurations, including Cu catalyst CO₂ reduction) | CC BY 4.0 | Catalysis research case studies  
**QM9** | Small organic molecules (134k molecules, 13 types of properties) | CC0 1.0 (Public Domain) | Standard chemical property prediction data  
  
**Notes** :  
\- **Case study data** : Cu catalyst CO₂ reduction reaction example based on OC20 dataset  
\- **Historical method comparison** : Empirical force field (AMBER, CHARMM) and DFT calculation accuracy references literature values  
\- **Timeline** : This chapter's chronology (1950s-2025) based on publication years of original papers for each method

### 1.10.2 Comparison Data for Molecular Simulation Methods

**Basis for computational cost comparison** :

Method | Computation speed (per atom) | Typical system size | Timescale  
---|---|---|---  
**Empirical force field** | ~10⁻⁶ sec/step | 10⁴-10⁶ atoms | Microseconds-milliseconds  
**DFT** | ~1-10 sec/step | 10²-10³ atoms | Picoseconds  
**MLP** | ~10⁻³-10⁻² sec/step | 10³-10⁴ atoms | Nanoseconds-microseconds  
  
**Basis for accuracy comparison** :

  * **Empirical force field** : MAE ~10-50 kcal/mol (large errors in molecular energy, reaction barriers)
  * **DFT** : MAE ~1-5 kcal/mol (used as reference method)
  * **MLP** : MAE ~0.1-1 kcal/mol (when trained on DFT data)

### 1.10.3 Sources for Historical Timeline

Era | Event | Source (Reference number)  
---|---|---  
1957 | First molecular dynamics simulation | [1] Alder & Wainwright  
1977 | First all-atom MD of protein | [2] McCammon et al.  
1964-1965 | Establishment of DFT theory (Nobel Prize) | [5][6] Hohenberg, Kohn, Sham  
1985 | Car-Parrinello method (DFT + MD) | [7] Car & Parrinello  
2007 | Behler-Parrinello MLP (neural network potential) | [8] Behler & Parrinello  
2017 | SchNet (GNN-based MLP) | [12] Schütt et al.  
2022 | NequIP, MACE (equivariant MLP) | [15][16] Batzner et al., Batatia et al.  
  
* * *

## 1.11 Practical Considerations: What to Know Before Learning MLP

### 1.11.1 Common Misconceptions and Pitfalls

**Misconception 1: MLP is not omnipotent - Understanding application limits**

**Problem** :  
Misconception that "with MLP, can calculate any chemical system with high accuracy"

**Reality** :  
MLP exhibits high accuracy only within the range of training data.

**Specific example** :

  * **Training data** : Cu surface CO₂ adsorption system at room temperature (300K)
  * **Predictions accurate** : 250-350K, low-coverage CO₂ adsorption
  * **Predictions inaccurate** : High temperature (>600K), Cu surface melting phenomena, CO₂ dissociation reaction intermediates

**Countermeasures** :

  * Check training data range beforehand (temperature, pressure, chemical composition)
  * If extrapolation needed, collect additional data with Active Learning
  * Utilize prediction uncertainty estimation (ensemble, dropout)

**Prevention** :  
Understanding "descriptors" and "training data distribution" learned in Chapter 2 is important

**Misconception 2: DFT is not obsolete - Complementary relationship**

**Problem** :  
Misconception that "DFT no longer needed since MLP emerged"

**Reality** :  
MLP is trained on DFT data, so DFT accuracy is MLP's upper limit.

Situation | Optimal method | Reason  
---|---|---  
Initial exploration of new materials | DFT | No training data  
Initial reaction pathway screening | DFT | Ensure activation energy reliability  
Long-time MD (>100 ps) | MLP | Statistical sampling needed  
Large-scale systems (>1000 atoms) | MLP | DFT calculation impossible  
High-accuracy single-point calculation | DFT (or CCSD(T)) | MLP cannot exceed training data accuracy  
  
**Recommended workflow** :  
DFT (initial exploration) → MLP (training) → MLP-MD (large-scale simulation) → DFT (validation)

**Misconception 3: Understanding learning curve - Reality of time investment**

**Problem** :  
Expectation that "can use MLP immediately"

**Reality** :

Skill level | Time needed | What can be achieved  
---|---|---  
Beginner (no DFT experience) | 6-12 months | Model training on existing datasets  
Intermediate (DFT experience) | 3-6 months | Build MLP for own system + execute MD  
Advanced (computational chemistry + ML experience) | 1-3 months | Active Learning + custom architecture  
  
**Learning roadmap** :

  1. **Phase 1 (1-2 months)** : Python, PyTorch, basic quantum chemistry
  2. **Phase 2 (1-2 months)** : DFT calculation execution (VASP, Quantum ESPRESSO, etc.)
  3. **Phase 3 (1-2 months)** : Existing dataset training with SchNetPack
  4. **Phase 4 (2-4 months)** : Application to own research topic

**Time-saving tips** :

  * Leverage existing pretrained models (OC20, ANI-1x) with transfer learning
  * Utilize community forums (Materials Project Discourse, PyTorch Forum)
  * Thoroughly complete this series' exercises

**Pitfall 4: Computational resource estimation errors**

**Problem** :  
Misconception that "MLP faster than DFT, so laptop sufficient"

**Reality** :

**MLP training phase** (initial only):

  * **Required resources** : GPU (NVIDIA RTX 3090 or better recommended), 32GB+ memory
  * **Training time** : 6-24 hours for 100k configurations (architecture-dependent)
  * **Storage** : Training data 10-100GB

**MLP inference phase** (MD execution):

  * **Required resources** : 4-8 core CPU, or GPU (recommended)
  * **Execution time** : 1000-atom system, 1 ns = 1-10 hours (with GPU)

**Minimum configuration guidelines** :

  * **For training** : Laptop not suitable, GPU server or cloud (Google Colab Pro, AWS)
  * **For inference** : Desktop PC possible (GPU recommended, CPU-only executable but slow)

**Cost estimate example** :

  * **Cloud GPU** : AWS p3.2xlarge (V100 GPU) = $3/hour × 20 hours = $60/model
  * **Own GPU** : RTX 4090 = ¥300,000 (initial investment, amortizable across multiple projects)

**Pitfall 5: Reality of paper reproduction**

**Problem** :  
Expectation that "can directly apply Nature paper MLP to own system"

**Reality** :

  * Paper models optimized for specific systems, often cannot transfer directly
  * Even when code is published, environment setup frequently problematic
  * Hyperparameter readjustment essential (even if described in paper, requires changes for own system)

**Reproducibility checklist** :

  * □ Is training data published? (Confirm DOI, URL)
  * □ Is code published on GitHub? (Check star count, last update)
  * □ Are dependency library versions specified?
  * □ Does paper's Supporting Information include all hyperparameters?
  * □ Do authors provide Docker images or Conda environment files?

**Time-saving hint** :  
Criteria for selecting highly reproducible papers:

  * Nature/Science series → not necessarily easy to reproduce
  * NeurIPS/ICML series (ML conferences) → tend to have well-documented code and data
  * Paper's GitHub repository has 100+ stars → community-proven

* * *

## 1.12 Chapter-End Checklist: MLP Introduction Quality Assurance

After reading this chapter, check the following items. If all can be checked, you're ready to proceed to Chapter 2 "MLP Fundamentals."

### 1.12.1 Conceptual Understanding (Understanding)

**Historical background understanding** :

  * □ Can explain three eras of molecular simulation (1950s-1980s empirical force fields, 1980s-2010s DFT, 2010s-present MLP)
  * □ Can explain three limitations of empirical force fields (reactivity, accuracy, parameterization)
  * □ Can explain three limitations of DFT (computational cost, system size, timescale)
  * □ Can explain in own words the problem MLP solves (accuracy vs cost trade-off)

**MLP positioning** :

  * □ Understand MLP-DFT relationship (complementary, MLP trained on DFT data)
  * □ Can distinguish problems MLP suits (large-scale systems, long-time MD) from unsuitable problems (new materials, initial exploration)
  * □ Can explain meaning of "quantum accuracy, classical speed"
  * □ Understand MLP application limits (within training data range)

**Case study understanding** :

  * □ Can explain Cu catalyst CO₂ reduction reaction workflow comparison (empirical force field vs DFT vs MLP)
  * □ Can list three reasons why MLP produces results in "2.5-4 weeks"
  * □ Can explain where DFT-MLP computation time difference (~1000×) comes from

**Technology trend understanding** :

  * □ Can explain three tailwinds for "why MLP now" (hardware, data, algorithms)
  * □ Can briefly explain differences between Behler-Parrinello (2007) and NequIP/MACE (2022)
  * □ Understand social background in materials exploration (carbon neutrality, energy crisis)

### 1.12.2 Practical Skills (Doing)

**Research planning** :

  * □ Can judge whether MLP applicable to own research topic (criteria: system size, timescale, required accuracy)
  * □ Can estimate MLP project time (learning curve, computational resources, data preparation)
  * □ Can estimate required computational resources (GPU, storage, cloud vs own)
  * □ Can design standard workflow: DFT → MLP → MLP-MD

**Literature survey skills** :

  * □ Can cite 3+ MLP application examples in own research field (catalysis, batteries, drug discovery, etc.)
  * □ Can evaluate paper reproducibility (data publication, code publication, environment information availability)
  * □ Can research differences among MLP tools on GitHub (SchNetPack, NequIP, MACE)

**Communication** :

  * □ Can explain "what MLP is" to lab members in 5 minutes
  * □ Can logically explain "reasons for using MLP" to supervisor or PI (cost, time, accuracy perspectives)
  * □ Can tangibly describe differences between "computational chemist's day in 2000" and "2025"

### 1.12.3 Application Ability (Applying)

**Problem setting evaluation** :

  * □ When new research topic arises, can judge which is optimal: "empirical force field vs DFT vs MLP"
  * □ Can detect cases where MLP likely to fail (extrapolation, insufficient training data) beforehand
  * □ Can quantitatively evaluate computational cost vs project duration trade-off

**Research strategy construction** :

  * □ Can compare research plans for own research topic "without using MLP" and "using MLP"
  * □ Can judge Active Learning necessity (when training data insufficient)
  * □ Can consider pretrained model transfer learning (OC20, ANI-1x, etc.)

**Preparation for next chapter** :

  * □ Can list content to learn in Chapter 2 (descriptors, architecture, training methods)
  * □ Can clarify what to achieve in Chapter 3 hands-on (SchNetPack implementation)
  * □ Can set 3-month goal for how to utilize MLP in own project ultimately

* * *

## References

  1. Alder, B. J., & Wainwright, T. E. (1957). "Phase transition for a hard sphere system." _The Journal of Chemical Physics_ , 27(5), 1208-1209.  
DOI: [10.1063/1.1743957](<https://doi.org/10.1063/1.1743957>)

  2. McCammon, J. A., Gelin, B. R., & Karplus, M. (1977). "Dynamics of folded proteins." _Nature_ , 267(5612), 585-590.  
DOI: [10.1038/267585a0](<https://doi.org/10.1038/267585a0>)

  3. Cornell, W. D., et al. (1995). "A second generation force field for the simulation of proteins, nucleic acids, and organic molecules." _Journal of the American Chemical Society_ , 117(19), 5179-5197.  
DOI: [10.1021/ja00124a002](<https://doi.org/10.1021/ja00124a002>)

  4. Brooks, B. R., et al. (1983). "CHARMM: A program for macromolecular energy, minimization, and dynamics calculations." _Journal of Computational Chemistry_ , 4(2), 187-217.  
DOI: [10.1002/jcc.540040211](<https://doi.org/10.1002/jcc.540040211>)

  5. Hohenberg, P., & Kohn, W. (1964). "Inhomogeneous electron gas." _Physical Review_ , 136(3B), B864.  
DOI: [10.1103/PhysRev.136.B864](<https://doi.org/10.1103/PhysRev.136.B864>)

  6. Kohn, W., & Sham, L. J. (1965). "Self-consistent equations including exchange and correlation effects." _Physical Review_ , 140(4A), A1133.  
DOI: [10.1103/PhysRev.140.A1133](<https://doi.org/10.1103/PhysRev.140.A1133>)

  7. Car, R., & Parrinello, M. (1985). "Unified approach for molecular dynamics and density-functional theory." _Physical Review Letters_ , 55(22), 2471.  
DOI: [10.1103/PhysRevLett.55.2471](<https://doi.org/10.1103/PhysRevLett.55.2471>)

  8. Behler, J., & Parrinello, M. (2007). "Generalized neural-network representation of high-dimensional potential-energy surfaces." _Physical Review Letters_ , 98(14), 146401.  
DOI: [10.1103/PhysRevLett.98.146401](<https://doi.org/10.1103/PhysRevLett.98.146401>)

  9. Nitopi, S., et al. (2019). "Progress and perspectives of electrochemical CO2 reduction on copper in aqueous electrolyte." _Chemical Reviews_ , 119(12), 7610-7672.  
DOI: [10.1021/acs.chemrev.8b00705](<https://doi.org/10.1021/acs.chemrev.8b00705>)

  10. van Duin, A. C., et al. (2001). "ReaxFF: a reactive force field for hydrocarbons." _The Journal of Physical Chemistry A_ , 105(41), 9396-9409.  
DOI: [10.1021/jp004368u](<https://doi.org/10.1021/jp004368u>)

  11. Cheng, T., et al. (2020). "Auto-catalytic reaction pathways on electrochemical CO2 reduction by machine-learning interatomic potentials." _Nature Communications_ , 11(1), 5713.  
DOI: [10.1038/s41467-020-19497-z](<https://doi.org/10.1038/s41467-020-19497-z>)

  12. Schütt, K. T., et al. (2017). "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." _Advances in Neural Information Processing Systems_ , 30.  
arXiv: [1706.08566](<https://arxiv.org/abs/1706.08566>)

  13. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "Imagenet classification with deep convolutional neural networks." _Advances in Neural Information Processing Systems_ , 25.  
DOI: [10.1145/3065386](<https://doi.org/10.1145/3065386>)

  14. Klicpera, J., et al. (2020). "Directional message passing for molecular graphs." _International Conference on Learning Representations (ICLR)_.  
arXiv: [2003.03123](<https://arxiv.org/abs/2003.03123>)

  15. Batzner, S., et al. (2022). "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." _Nature Communications_ , 13(1), 2453.  
DOI: [10.1038/s41467-022-29939-5](<https://doi.org/10.1038/s41467-022-29939-5>)

  16. Batatia, I., et al. (2022). "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." _Advances in Neural Information Processing Systems_ , 35.  
arXiv: [2206.07697](<https://arxiv.org/abs/2206.07697>)

* * *

## Author Information

**Created by** : MI Knowledge Hub Content Team  
**Creation date** : 2025-10-17  
**Version** : 1.1 (Chapter 1 quality improvement)  
**Series** : MLP Introduction Series

**Update history** :  
\- 2025-10-19: v1.1 Quality improvement revision  
\- Added data licensing and reproducibility section (MD17, OC20, QM9 dataset information)  
\- Added basis data for computational cost and accuracy comparison (empirical force field vs DFT vs MLP)  
\- Specified sources for historical timeline (major papers from 1957-2022)  
\- Added practical considerations section (5 misconceptions and pitfalls, detailed countermeasures)  
\- Added chapter-end checklist (16 conceptual understanding items, 10 practical skill items, 9 application ability items)  
\- 2025-10-17: v1.0 Chapter 1 initial release  
\- History of molecular simulation (1950s-present)  
\- Detailed limitations of conventional methods (empirical force fields, DFT) from three perspectives  
\- Cu catalyst CO₂ reduction reaction case study (detailed workflow comparison)  
\- "Computational chemist's day" column (2000 vs 2025)  
\- "Why MLP now" section (3 tailwinds + social background)  
\- Overview of major MLP methods (Behler-Parrinello, SchNet, DimeNet, NequIP, MACE)  
\- 3 exercises (easy, medium, hard)  
\- 16 carefully selected references (important papers)

**License** : Creative Commons BY-NC-SA 4.0

[← Back to Series Index](<index.html>)
