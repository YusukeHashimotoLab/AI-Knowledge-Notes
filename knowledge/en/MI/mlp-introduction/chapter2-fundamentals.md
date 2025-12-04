---
title: "Chapter 2: MLP Fundamentals - Concepts, Methods, Ecosystem"
chapter_title: "Chapter 2: MLP Fundamentals - Concepts, Methods, Ecosystem"
---

# Chapter 2: MLP Fundamentals - Concepts, Methods, Ecosystem

We systematize the concepts of learning potential energy surfaces and the key aspects of GNNs that satisfy equivariance. We also cover the key considerations for training data design.

**💡 Supplement:** Equivariance is a property meaning "the meaning remains the same even if the coordinate system changes." This ensures generalization with data containing diverse structures.

## Learning Objectives

By reading this chapter, you will master the following:  
\- Understand the precise definition of MLP and its relationship with related fields (quantum chemistry, machine learning, molecular dynamics)  
\- Explain 15 frequently encountered technical terms in MLP research  
\- Understand the 5-step MLP workflow (data collection, descriptor design, model training, validation, simulation)  
\- Compare the characteristics and application scenarios of 3 types of descriptors (symmetry functions, SOAP, graphs)

* * *

## 2.1 What is MLP: Precise Definition

In Chapter 1, we learned that MLP is a revolutionary technology that "combines DFT accuracy with the speed of empirical force fields." Here, we define it more precisely.

### Definition

**Machine Learning Potential (MLP)** is:

> A method that trains machine learning models using datasets of atomic configurations, energies, and forces obtained from quantum mechanical calculations (mainly DFT) to rapidly and accurately predict the potential energy and forces for arbitrary atomic configurations.

**Expressed mathematically** :
    
    
    Training data: D = {(R₁, E₁, F₁), (R₂, E₂, F₂), ..., (Rₙ, Eₙ, Fₙ)}
    
    R: Atomic configuration (3N-dimensional vector, N=number of atoms)
    E: Energy (scalar)
    F: Forces (3N-dimensional vector)
    
    Objective: Learn function f_MLP
      E_pred = f_MLP(R)
      F_pred = -∇_R f_MLP(R)
    
    Constraints:
      - |E_pred - E_DFT| < several meV (millielectronvolts)
      - |F_pred - F_DFT| < tens of meV/Å
      - Computational time: 10⁴-10⁶ times faster than DFT
    

### Three Essential Elements of MLP

**1\. Data-Driven**  
\- Requires large amounts of data (thousands to tens of thousands of configurations) from DFT calculations  
\- Data quality and quantity determine model performance  
\- No need for "manual parameter tuning" like traditional empirical force fields

**2\. High-Dimensional Function Approximation**  
\- The Potential Energy Surface (PES) is ultra-high-dimensional  
\- Example: 100-atom system → 300-dimensional space  
\- Neural networks efficiently learn this complex function

**3\. Physical Constraints Incorporation**  
\- Energy conservation  
\- Translational and rotational invariance (energy remains unchanged when the entire system is moved)  
\- Atomic permutation symmetry (energy remains unchanged when atoms of the same element are swapped)  
\- Force-energy consistency (F = -∇E)

### Position in Related Fields
    
    
    ```mermaid
    flowchart TD
        QC[Quantum Chemistry<br>Quantum Chemistry] -->|Reference data generation| MLP[Machine Learning Potential<br>MLP]
        ML[Machine Learning<br>Machine Learning] -->|Models & algorithms| MLP
        MD[Molecular Dynamics<br>Molecular Dynamics] -->|Simulation methods| MLP
    
        MLP --> APP1[Materials Design<br>Materials Design]
        MLP --> APP2[Catalysis<br>Catalysis]
        MLP --> APP3[Drug Design<br>Drug Design]
        MLP --> APP4[Surface Science<br>Surface Science]
    
        style MLP fill:#a8d5ff
        style APP1 fill:#d4f1d4
        style APP2 fill:#d4f1d4
        style APP3 fill:#d4f1d4
        style APP4 fill:#d4f1d4
    ```

**Contributions from Quantum Chemistry** :  
\- DFT, quantum chemistry methods (CCSD(T), etc.)  
\- Concept of potential energy surfaces  
\- Chemical insights (bonding, reaction mechanisms)

**Contributions from Machine Learning** :  
\- Neural Networks (NN), Graph Neural Networks (GNN)  
\- Optimization algorithms (Adam, SGD, etc.)  
\- Regularization, overfitting countermeasures

**Contributions from Molecular Dynamics** :  
\- MD integration methods (Verlet method, etc.)  
\- Ensemble theory (NVT, NPT)  
\- Statistical mechanics analysis methods

* * *

## 2.2 MLP Glossary: 15 Important Concepts

We briefly explain technical terms frequently encountered in MLP research.

Term | English | Description  
---|---|---  
**Potential Energy Surface** | Potential Energy Surface (PES) | An ultra-high-dimensional function representing the relationship between atomic configuration and energy. Corresponds to a "topographic map" of chemical reactions.  
**Descriptor** | Descriptor | Numerical vector representation of atomic configuration features. Includes symmetry functions, SOAP, graph representations, etc.  
**Symmetry Function** | Symmetry Function | Descriptor proposed by Behler-Parrinello. Describes radial and angular distributions around atoms.  
**Message Passing** | Message Passing | Operation in graph neural networks where information propagates between adjacent atoms.  
**Equivariance** | Equivariance | Property where outputs transform correspondingly to input transformations (such as rotation). E(3) equivariance is crucial.  
**Invariance** | Invariance | Property where outputs remain unchanged under input transformations (such as rotation). Energy is rotationally invariant.  
**Cutoff Radius** | Cutoff Radius | Maximum distance for considering interatomic interactions. Typically 5-10Å. Balance between computational cost and accuracy.  
**MAE** | Mean Absolute Error | Mean absolute error. Metric for evaluating MLP accuracy. Units: meV/atom for energy, meV/Å for forces.  
**Active Learning** | Active Learning | Method where the model automatically selects uncertain configurations and adds DFT calculations to efficiently expand the dataset.  
**Data Efficiency** | Data Efficiency | Ability to achieve high accuracy with minimal training data. Latest methods (MACE) require only thousands of configurations.  
**Generalization** | Generalization | Ability to accurately predict configurations not in training data. Avoiding overfitting is essential.  
**Ensemble** | Ensemble | Method of training multiple independent MLP models to evaluate prediction averages and uncertainties.  
**Transfer Learning** | Transfer Learning | Method of applying models trained on one system to related systems. Reduces computational cost.  
**Many-Body Interaction** | Many-Body Interaction | Interactions involving three or more atoms. Essential for describing chemical bonding.  
**E(3) Equivariance** | E(3) Equivariance | Equivariance with respect to the 3D Euclidean group (translation, rotation, inversion). Core technology of NequIP and MACE.  
  
### Top 5 Most Important Terms

**Terms beginners should understand first** :

  1. **Potential Energy Surface (PES)** : What MLP learns
  2. **Descriptor** : Numerical representation of atomic configurations
  3. **Invariance and Equivariance** : Mathematical properties for satisfying physical laws
  4. **MAE** : Quantitative evaluation of model accuracy
  5. **Active Learning** : Efficient data collection strategy

* * *

## 2.3 Input to MLP: Types of Atomic Configuration Data

MLP training requires diverse atomic configurations. What types of data are used?

### Major Input Data Types

**1\. Equilibrium Structures and Vicinity**  
\- **Description** : Most stable structure (energy minimum) and its neighborhood  
\- **Use** : Properties of stable structures, vibrational spectra  
\- **Generation** : DFT structure optimization + small displacements  
\- **Data Volume** : Hundreds to thousands of configurations  
\- **Examples** : Crystal structures, optimized molecular structures

**2\. Molecular Dynamics Trajectories**  
\- **Description** : Time-series configurations from ab initio MD (AIMD)  
\- **Use** : Dynamic behavior, high-temperature properties  
\- **Generation** : Short-time (tens of ps) AIMD at various temperatures  
\- **Data Volume** : Thousands to tens of thousands of configurations  
\- **Examples** : Liquids, melting, diffusion processes

**3\. Reaction Pathways**  
\- **Description** : Paths from reactants to products including transition states  
\- **Generation** : Nudged Elastic Band (NEB) method, String method  
\- **Use** : Catalytic reactions, chemical reaction mechanisms  
\- **Data Volume** : Hundreds to thousands of configurations (including multiple paths)  
\- **Examples** : CO₂ reduction reactions, hydrogen generation reactions

**4\. Random Sampling**  
\- **Description** : Random exploration of configuration space  
\- **Generation** : Adding large displacements to existing structures, Monte Carlo sampling  
\- **Use** : Improving generalization performance, covering unknown regions  
\- **Data Volume** : Thousands to tens of thousands of configurations  
\- **Caution** : May include high-energy regions causing unstable calculations

**5\. Defects and Interface Structures**  
\- **Description** : Crystal defects, surfaces, grain boundaries, nanoparticles  
\- **Use** : Material fracture, catalytic active sites  
\- **Generation** : Systematically introduce defects and perform DFT calculations  
\- **Data Volume** : Hundreds to thousands of configurations  
\- **Examples** : Vacancies, dislocations, surface adsorption sites

### Example Data Type Combinations

**Typical dataset composition (Cu catalyst CO₂ reduction reaction)** :

Data Type | Configurations | Percentage | Purpose  
---|---|---|---  
Equilibrium structures (Cu surface + adsorbates) | 500 | 10% | Basic structures  
AIMD (300K, 500K, 700K) | 3,000 | 60% | Thermal fluctuations  
Reaction pathways (5 paths) | 500 | 10% | Reaction mechanisms  
Random sampling | 500 | 10% | Generalization  
Surface defects | 500 | 10% | Real catalyst heterogeneity  
**Total** | **5,000** | **100%** |   
  
* * *

## 2.4 MLP Ecosystem: Understanding the Big Picture

MLP does not function in isolation but operates within an ecosystem comprising data generation, training, simulation, and analysis.
    
    
    ```mermaid
    flowchart TB
        subgraph "Phase 1: Data Generation"
            DFT[DFT Calculation<br>VASP, Quantum ESPRESSO] --> DATA[Dataset<br>R, E, F]
        end
    
        subgraph "Phase 2: Model Training"
            DATA --> DESC[Descriptor Generation<br>Symmetry Func, SOAP, Graph]
            DESC --> TRAIN[NN Training<br>PyTorch, JAX]
            TRAIN --> VALID{Accuracy Validation<br>MAE < threshold?}
            VALID -->|No| AL[Active Learning<br>Select configurations to add]
            AL --> DFT
            VALID -->|Yes| MODEL[Trained MLP]
        end
    
        subgraph "Phase 3: Simulation"
            MODEL --> MD[MLP-MD<br>LAMMPS, ASE]
            MD --> TRAJ[Trajectory<br>Configuration/energy time series]
        end
    
        subgraph "Phase 4: Analysis"
            TRAJ --> ANA1[Structure Analysis<br>RDF, coordination number]
            TRAJ --> ANA2[Dynamic Properties<br>Diffusion coefficient, reaction rate]
            TRAJ --> ANA3[Thermodynamics<br>Free energy]
            ANA1 --> INSIGHT[Scientific Insights]
            ANA2 --> INSIGHT
            ANA3 --> INSIGHT
        end
    
        INSIGHT -.->;New hypothesis| DFT
    
        style MODEL fill:#ffeb99
        style TRAJ fill:#d4f1d4
        style INSIGHT fill:#ffb3b3
    ```

### Details of Each Ecosystem Phase

**Phase 1: Data Generation (DFT Calculations)**  
\- **Tools** : VASP, Quantum ESPRESSO, CP2K, GPAW  
\- **Computational Time** : Several days to weeks on supercomputers  
\- **Output** : Atomic configurations, energies, forces, stress tensors

**Phase 2: Model Training**  
\- **Tools** : SchNetPack, NequIP, MACE, DeePMD-kit  
\- **Computational Resources** : 1 to several GPUs  
\- **Training Time** : Several hours to days  
\- **Output** : Trained MLP model (.pth file, etc.)

**Phase 3: Simulation**  
\- **Tools** : LAMMPS, ASE, i-PI  
\- **Computational Resources** : 1 to tens of GPUs  
\- **Simulation Time** : Nanoseconds to microseconds  
\- **Output** : Trajectory files (.xyz, .lammpstrj)

**Phase 4: Analysis**  
\- **Tools** : Python (NumPy, MDAnalysis, MDTraj), OVITO, VMD  
\- **Analysis Content** :  
\- Structure: Radial Distribution Function (RDF), coordination number, cluster analysis  
\- Dynamics: Diffusion coefficient, reaction rate constants, residence time  
\- Thermodynamics: Free energy, entropy

* * *

## 2.5 MLP Workflow: 5 Steps

Research projects using MLP proceed through the following 5 steps.

### Step 1: Data Collection

**Objective** : Generate high-quality DFT data needed for MLP training

**Specific Tasks** :  
1\. **System Definition** : Determine target chemical system, size, composition  
2\. **Sampling Strategy** : Plan what configurations to calculate  
\- Equilibrium structures and vicinity  
\- AIMD (multiple temperatures)  
\- Reaction pathways  
\- Random sampling  
3\. **DFT Calculation Settings** :  
\- Functional (PBE, HSE06, etc.)  
\- Basis functions (plane waves, localized orbitals)  
\- Cutoff energy, k-point mesh  
4\. **Parallel Calculation Execution** : Calculate thousands of configurations on supercomputers

**Keys to Success** :  
\- **Diversity** : Include various configurations (monotonous data degrades generalization)  
\- **Balance** : Balance between low- and high-energy regions  
\- **Quality Control** : Check SCF convergence, force convergence

**Typical Cost** : 5,000 configurations → 3-7 days on supercomputer

### Step 2: Descriptor Design

**Objective** : Convert atomic configurations into numerical vectors suitable for machine learning

**Major Descriptor Types** (detailed in 2.6):  
\- **Symmetry Functions** : Manual design, Behler-Parrinello  
\- **SOAP (Smooth Overlap of Atomic Positions)** : Mathematically refined representation  
\- **Graph Neural Networks (GNN)** : Automatic learning, SchNet, DimeNet

**Descriptor Selection Criteria** :  
\- **Data Efficiency** : Can high accuracy be achieved with little data?  
\- **Computational Cost** : Computational time during inference (prediction)  
\- **Physical Interpretability** : Can chemical insights be obtained?

**Hyperparameters** :  
\- Cutoff radius (5-10Å)  
\- Number of radial basis functions (10-50)  
\- Angular resolution

### Step 3: Model Training

**Objective** : Optimize neural network to learn PES

**Training Process** :  
1\. **Data Split** : Training (80%), validation (10%), test (10%)  
2\. **Loss Function Definition** :  
```  
Loss = w_E × MSE(E_pred, E_true) + w_F × MSE(F_pred, F_true) w_E: Energy weight (typically 1)  
w_F: Force weight (typically 100-1000, for unit conversion)  
```  
3\. **Optimization** : Adam, SGD, etc. for thousands to tens of thousands of epochs  
4\. **Regularization** : L2 regularization, dropout to prevent overfitting

**Hyperparameter Tuning** :  
\- Learning rate: 10⁻³ to 10⁻⁵  
\- Batch size: 32-256  
\- Network depth: 3-6 layers  
\- Hidden layer size: 64-512 nodes

**Computational Resources** : Several hours to 2 days on 1 GPU

### Step 4: Validation

**Objective** : Evaluate whether model has sufficient accuracy and generalization performance

**Quantitative Metrics** :  
| Metric | Target Value | Description |  
|------|--------|------|  
| Energy MAE | < 1-5 meV/atom | Mean absolute error (test set) |  
| Force MAE | < 50-150 meV/Å | Force error on atoms |  
| Stress MAE | < 0.1 GPa | For solid materials |  
| R² (coefficient of determination) | > 0.99 | Energy correlation |

**Qualitative Validation** :  
\- **Extrapolation Test** : Check accuracy on configurations outside training data range  
\- **Physical Quantity Reproduction** : Do lattice constants, elastic constants, vibrational spectra match DFT?  
\- **Short-Time MD Test** : Run 10-100 ps MD, check energy conservation

**If Failed** :  
\- Add data (Active Learning)  
\- Hyperparameter tuning  
\- Change to more powerful model (SchNet → NequIP)

### Step 5: Production Simulation

**Objective** : Execute scientifically meaningful simulations with trained MLP

**Typical MLP-MD Simulation Settings** :
    
    
    System size: 10³-10⁴ atoms
    Temperature: 300-1000 K (depending on purpose)
    Pressure: 1 atm or constant volume
    Time step: 0.5-1.0 fs
    Total simulation time: 1-100 ns
    Ensemble: NVT (canonical), NPT (isothermal-isobaric)
    

**Execution Time Estimates** :  
\- 1,000 atoms, 1 ns simulation → 1-3 days on 1 GPU  
\- Further speedup possible with parallelization

**Cautions** :  
\- **Energy Drift** : Monitor for monotonic energy increase/decrease in long simulations  
\- **Unlearned Regions** : Accuracy may degrade when encountering configurations not in training data  
\- **Ensemble Uncertainty** : Evaluate prediction variance with multiple independent MLP models

* * *

## 2.6 Types of Descriptors: Numericalizing Atomic Configurations

MLP performance heavily depends on descriptor design. Here we compare three major approaches.

### 1\. Symmetry Functions

**Proposed by** : Behler & Parrinello (2007)

**Basic Idea** :  
\- Represent the local environment of each atom i using functions of radial (distance) and angular information  
\- Designed to satisfy rotational and translational invariance

**Radial Symmetry Functions** :
    
    
    G_i^rad = Σ_j exp(-η(r_ij - R_s)²) × f_c(r_ij)
    
    r_ij: distance between atoms i and j
    η: parameter determining Gaussian width
    R_s: center distance (use multiple values)
    f_c: cutoff function (smoothly decays influence of distant atoms)
    

**Angular Symmetry Functions** :
    
    
    G_i^ang = 2^(1-ζ) Σ_(j,k≠i) (1 + λcosθ_ijk)^ζ ×
              exp(-η(r_ij² + r_ik² + r_jk²)) ×
              f_c(r_ij) × f_c(r_ik) × f_c(r_jk)
    
    θ_ijk: angle formed by atoms j-i-k
    ζ, λ: parameters controlling angular resolution
    

**Advantages** :  
\- Physically interpretable (corresponds to radial and angular distributions)  
\- Rotational and translational invariance guaranteed  
\- Relatively simple implementation

**Disadvantages** :  
\- **Manual design** : Need to manually select parameters like η, R_s, ζ, λ  
\- **High-dimensional** : Requires 50-100 dimensional descriptors (increased computational cost)  
\- **Low data efficiency** : May need tens of thousands of configurations

**Application Examples** : Water, silicon, metal surfaces

### 2\. SOAP (Smooth Overlap of Atomic Positions)

**Proposed by** : Bartók et al. (2013)

**Basic Idea** :  
\- Approximate electron density around atoms with "Gaussian density"  
\- Calculate "overlap" of these density distributions as descriptor  
\- Mathematically rigorous rotational invariance

**Mathematical Definition (Simplified)** :
    
    
    ρ_i(r) = Σ_j exp(-α|r - r_j|²)  (Gaussian density around atom i)
    
    SOAP_i = integral[ρ_i(r) × ρ_i(r') × kernel(r, r')]
    
    kernel: radial/angular basis functions
    

In practice, efficiently calculated using **spherical harmonics expansion**.

**Advantages** :  
\- Mathematically refined representation  
\- Fewer parameters than symmetry functions  
\- Good compatibility with kernel methods (Gaussian Process Regression)

**Disadvantages** :  
\- Somewhat higher computational cost  
\- More complex combination with neural networks than SchNet

**Application Examples** : Crystalline materials, nanoclusters, complex alloys

### 3\. Graph Neural Networks

**Representative Methods** : SchNet (2017), DimeNet (2020), PaiNN (2021)

**Basic Idea** :  
\- Represent molecules as **graphs**  
\- Nodes (vertices) = atoms  
\- Edges = interatomic interactions  
\- **Message passing** : Propagate information between adjacent atoms  
\- No manual descriptor design, **neural network learns automatically**

**SchNet Architecture (Conceptual)** :
    
    
    Initial state:
      Atom i feature vector h_i^(0) = Embedding(Z_i)  (Z_i: atomic number)
    
    Message passing (repeat L layers):
      for l = 1 to L:
        m_ij = NN_filter(r_ij) × h_j^(l-1)  (distance-dependent filter)
        h_i^(l) = h_i^(l-1) + Σ_j m_ij      (message aggregation)
        h_i^(l) = NN_update(h_i^(l))        (nonlinear transformation)
    
    Energy prediction:
      E_i = NN_output(h_i^(L))              (per-atom energy)
      E_total = Σ_i E_i
    

**Advantages** :  
\- **End-to-end learning** : No descriptor design needed  
\- **Flexibility** : Applicable to various systems  
\- **Data efficiency** : High accuracy with less data than symmetry functions

**Disadvantages** :  
\- Black box nature (difficult to interpret)  
\- Requires computational resources for training

**Evolution: DimeNet (adding angular information)** :
    
    
    DimeNet = SchNet + explicit consideration of bond angles θ_ijk
    
    Embed angular information in messages:
      m_ij = NN(r_ij, {θ_ijk}_k)
    

**Latest: E(3) Equivariant GNN (NequIP, MACE)** :  
\- Implement **equivariance** rather than rotational **invariance**  
\- Propagate vector/tensor fields as messages  
\- Dramatically improved data efficiency (thousands of configurations sufficient)

**Application Examples** : Organic molecules, catalytic reactions, complex biomolecules

### Descriptor Comparison Table

Item | Symmetry Functions | SOAP | GNN (SchNet family) | E(3) Equivariant GNN  
---|---|---|---|---  
**Design Approach** | Manual | Mathematical formulation | Automatic learning | Automatic learning + physics  
**Invariance** | Rotation & translation | Rotation & translation | Rotation & translation | E(3) equivariant  
**Dimensionality** | 50-100 | 30-50 | Learnable | Learnable  
**Data Efficiency** | Low | Medium | Medium | **High**  
**Accuracy** | Medium | High | High | **Highest**  
**Computational Cost** | Low | Medium | Medium | Medium-High  
**Implementation Difficulty** | Low | Medium | Medium | High  
**Interpretability** | High | Medium | Low | Low  
  
**Selection Guidelines** :  
\- **Beginners, small systems** : Symmetry functions (easy to understand)  
\- **Complex crystals, alloys** : SOAP (combine with kernel methods)  
\- **Organic molecules, catalysis** : GNN (SchNet, DimeNet)  
\- **Cutting-edge, data scarcity** : E(3) equivariant GNN (NequIP, MACE)

* * *

## 2.7 Comparison of Major MLP Architectures

We compare representative MLP methods using the descriptors learned so far.

Method | Year | Descriptor | Features | Data Efficiency | Accuracy | Implementation  
---|---|---|---|---|---|---  
**Behler-Parrinello NN** | 2007 | Symmetry functions | Per-atom NN, simple | Low (tens of thousands) | Medium | n2p2, AMP  
**GAP (SOAP + GP)** | 2010 | SOAP | Gaussian process regression, uncertainty quantification | Medium (thousands) | High | QUIP  
**ANI** | 2017 | Symmetry functions | Organic molecule specialized, large dataset | Medium | High | TorchANI  
**SchNet** | 2017 | GNN (automatic) | Continuous-filter convolution, end-to-end | Medium (5k-10k) | High | SchNetPack  
**DimeNet** | 2020 | GNN (angular) | Directional message passing | Medium (5k-10k) | High | PyG  
**NequIP** | 2021 | E(3) equivariant GNN | Tensor field message passing | **High (thousands)** | **Highest** | NequIP  
**MACE** | 2022 | E(3) equivariant + ACE | Higher-order many-body terms, highest data efficiency | **Highest (thousands)** | **Highest** | MACE  
  
### Evolution Timeline
    
    
    ```mermaid
    timeline
        title Evolution of MLP Methods (2007-2024)
        2007 : Behler-Parrinello NN
             : Symmetry functions + feedforward NN
        2010 : GAP (SOAP + GP)
             : Mathematically rigorous descriptors
        2017 : SchNet, ANI
             : Graph NN, end-to-end learning
        2020 : DimeNet, PaiNN
             : Angular information, directional messages
        2021 : NequIP
             : E(3) equivariance implementation
        2022 : MACE
             : Higher-order many-body + data efficiency optimization
        2024 : Large-scale pre-trained models
             : ORB, GNoME (trained on hundreds of millions of configurations)
    ```

### Accuracy vs Data Efficiency Tradeoff

**Typical performance (guidelines for 100-atom molecular systems)** :

Method | Training Data | Energy MAE | Force MAE | Training Time (1 GPU)  
---|---|---|---|---  
Behler-Parrinello | 30,000 | 3-5 meV/atom | 80-120 meV/Å | 6-12 hours  
GAP | 10,000 | 1-2 meV/atom | 40-60 meV/Å | 12-24 hours (CPU)  
SchNet | 8,000 | 1-3 meV/atom | 50-80 meV/Å | 4-8 hours  
DimeNet | 8,000 | 0.8-2 meV/atom | 40-60 meV/Å | 8-16 hours  
NequIP | 4,000 | 0.5-1 meV/atom | 30-50 meV/Å | 12-24 hours  
MACE | 3,000 | **0.3-0.8 meV/atom** | **20-40 meV/Å** | 16-32 hours  
  
**Key Observations** :  
\- Data efficiency improved **10-fold** (30,000 → 3,000 configurations)  
\- Accuracy also improved **10-fold** (5 meV → 0.5 meV/atom)  
\- Training time remains similar (several hours to 1 day)

* * *

## 2.8 Column: Efficient Data Collection with Active Learning

Standard MLP training prepares large amounts of DFT data in advance. However, **Active Learning** can achieve high accuracy with minimal necessary data.

### Active Learning Workflow
    
    
    ```mermaid
    flowchart LR
        A[Initial Data<br>Hundreds of configs] --> B[MLP Training<br>v1.0]
        B --> C[Exploratory MD<br>Run with v1.0]
        C --> D{Uncertainty<br>Evaluation}
        D -->|Discover high-uncertainty configs| E[DFT Calculation<br>Addition]
        E --> F[Dataset<br>Update]
        F --> G[MLP Retraining<br>v1.1]
        G --> H{Accuracy<br>Sufficient?}
        H -->|No| C
        H -->|Yes| I[Final MLP<br>Production use]
    
        style D fill:#ffffcc
        style H fill:#ccffcc
    ```

### Uncertainty Evaluation Methods

**Ensemble Method** :
    
    
    # 5 independently trained MLP models
    models = [MLP_1, MLP_2, MLP_3, MLP_4, MLP_5]
    
    # Predict energy for a configuration R
    energies = [model.predict(R) for model in models]
    
    # Mean and standard deviation
    E_mean = mean(energies)
    E_std = std(energies)  # Uncertainty metric
    
    # Add DFT calculation if above threshold
    if E_std > threshold:
        E_DFT = run_DFT(R)
        add_to_dataset(R, E_DFT)
    

**Advantages** :  
\- Data collection efficiency improves **3-5 times**  
\- Automatically discovers important configurations (transition states, defects, etc.)  
\- Objective sampling not relying on human intuition

**Success Examples** :  
\- Si phase transition: Initial 500 configs → Active Learning +1,500 configs → Total 2,000 configs achieves DFT accuracy (normally requires 10,000)  
\- Cu catalyst CO₂ reduction: Automatically discovered reaction intermediates, reduced DFT cost by 60%

* * *

## 2.9 Chapter Summary

### What We Learned

  1. **Precise Definition of MLP**  
\- Data-driven high-dimensional function approximation  
\- Importance of physical constraints (invariance, equivariance, force-energy consistency)  
\- Fusion technology of quantum chemistry, machine learning, and molecular dynamics

  2. **15 Important Terms**  
\- PES, descriptor, symmetry functions, message passing, equivariance, invariance  
\- Cutoff radius, MAE, Active Learning, data efficiency, generalization  
\- Ensemble, transfer learning, many-body interactions, E(3) equivariance

  3. **MLP Input Data Types**  
\- Equilibrium structures, MD trajectories, reaction pathways, random sampling, defect structures  
\- Dataset composition balance (low/high energy regions, diversity)

  4. **MLP Ecosystem**  
\- 4 phases: Data generation → Model training → Simulation → Analysis  
\- Representative tools for each phase (VASP, SchNetPack, LAMMPS, MDAnalysis)

  5. **5-Step Workflow**  
\- Step 1: Data collection (DFT calculations, sampling strategy)  
\- Step 2: Descriptor design (symmetry functions, SOAP, GNN)  
\- Step 3: Model training (loss function, optimization, hyperparameters)  
\- Step 4: Accuracy validation (MAE, extrapolation tests, physical quantity reproduction)  
\- Step 5: Production simulation (MLP-MD, long timescales)

  6. **3 Types of Descriptors**  
\- **Symmetry functions** : Manual design, high physical interpretability, low data efficiency  
\- **SOAP** : Mathematical rigor, good compatibility with kernel methods, medium data efficiency  
\- **GNN** : Automatic learning, end-to-end, high data efficiency (especially E(3) equivariant types)

  7. **Major MLP Architectures**  
\- 2007 Behler-Parrinello → 2022 MACE: 10x data efficiency improvement, 10x accuracy improvement  
\- Latest methods (NequIP, MACE) achieve DFT accuracy with thousands of configurations

### Key Points

  * MLP performance heavily depends on **descriptor selection** and **data quality/quantity**
  * Latest methods with **E(3) equivariance** (NequIP, MACE) are best in data efficiency and accuracy
  * **Active Learning** can reduce DFT calculation costs by 50-70%
  * MLP is not a standalone technology but part of an **ecosystem** combining DFT, MD, and machine learning

### To the Next Chapter

In Chapter 3, we will experience actual **MLP training using SchNet** :  
\- Implementation with Python code  
\- Training on small dataset (MD17)  
\- Accuracy evaluation and hyperparameter tuning  
\- Troubleshooting

Furthermore, in Chapter 4, we will learn **advanced techniques of NequIP/MACE** and **real research applications**.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Create a comparison table for the three descriptors (symmetry functions, SOAP, graph neural networks) from the perspectives of "need for manual design," "data efficiency," and "physical interpretability."

Hint Recall the characteristics of each descriptor: \- Symmetry functions: Need to manually select parameters (η, R_s, ζ, λ) \- SOAP: Mathematically defined but some parameter tuning needed \- GNN: Neural network learns automatically  Sample Solution | Descriptor | Need for Manual Design | Data Efficiency | Physical Interpretability | |--------|----------------|-----------|------------| | **Symmetry Functions** | **High**  
\- Manually select η, R_s, ζ, λ  
\- Optimization needed per system | **Low**  
\- Need tens of thousands of configs  
\- High-dimensional (50-100D) | **High**  
\- Corresponds to radial/angular distributions  
\- Chemically interpretable | | **SOAP** | **Medium**  
\- Minimal parameters like number of basis functions  
\- Mathematically pre-defined | **Medium**  
\- Thousands to 10k configs  
\- 30-50D | **Medium**  
\- Interpreted as electron density overlap  
\- Somewhat abstract | | **GNN (SchNet family)** | **Low**  
\- End-to-end learning  
\- Descriptors auto-generated | **Medium-High**  
\- 5k-10k configs  
\- Learnable dimensions | **Low**  
\- Black box  
\- Difficult to visualize | | **E(3) Equivariant GNN  
(NequIP, MACE)** | **Low**  
\- Fully automatic learning  
\- Automatically incorporates physical laws | **High**  
\- Thousands of configs sufficient  
\- Highest data efficiency | **Low**  
\- Tensor field propagation  
\- Requires advanced mathematical interpretation | **Conclusion**: \- Beginners/interpretability emphasis → Symmetry functions \- Balanced approach → SOAP \- Best performance/data scarcity → E(3) equivariant GNN \- Versatility/easy implementation → GNN (SchNet) 

### Problem 2 (Difficulty: medium)

You are starting research on methanol oxidation reaction (CH₃OH → HCHO + H₂) on copper catalyst surfaces. Create a concrete work plan following the 5-step MLP workflow (data collection, descriptor design, training, validation, simulation). In particular, explain what types of atomic configurations should be prepared in Step 1 (data collection) and approximately how many, with reasoning.

Hint Consider the following: \- Catalytic reactions include not only stable structures but also transition states \- Temperature effects (thermal fluctuations) are important \- Cu surface structure (terraces, steps, defects) \- Diversity of adsorbates (CH₃OH, CH₃O, CH₂O, CHO, H, etc.)  Sample Solution **Step 1: Data Collection (DFT Calculations)** | Data Type | Configurations | Reason | |------------|--------|------| | **Equilibrium Structures** | 300 | Basic Cu(111) surface structures, stable adsorption sites for each adsorbate (CH₃OH, CH₃O, CH₂O, CHO, H, OH) at top, bridge, hollow sites | | **AIMD (300K, 500K)** | 2,000 | Thermal fluctuations at experimental temperature (300K) and high temperature (500K). Dynamic behavior including molecular rotation and surface diffusion | | **Reaction Pathways** | 800 | 4 major reaction paths (CH₃OH → CH₃O, CH₃O → CH₂O, CH₂O → CHO, CHO → CO + H) calculated by NEB method. 20 points × 4 paths × 10 conditions | | **Surface Defects** | 400 | Step edges, kinks, vacancies (actual catalysts are not perfect surfaces) | | **High-Energy Configurations** | 500 | Random sampling for generalization. Molecular dissociation states, multi-adsorption states | | **Total** | **4,000** | | **DFT Calculation Settings**: \- Functional: PBE + D3 (dispersion correction, important for methanol adsorption) \- Cutoff: 500 eV \- k-points: 4×4×1 (surface slab) \- Slab size: 4×4 Cu(111) surface (64 Cu atoms) + 15Å vacuum layer \- Computational time: ~5 days on supercomputer (parallelized) **Step 2: Descriptor Design** \- **Choice**: SchNet (easy implementation, proven for chemical reactions) \- Cutoff radius: 6Å (more than twice Cu-Cu nearest neighbor distance) \- Reason: Reactions involve many-body interactions (Cu-C-O-H). GNN learns automatically. **Step 3: Model Training** \- Framework: SchNetPack (PyTorch) \- Loss function: w_E=1, w_F=100 (emphasize force learning) \- Train/validation/test: 70%/15%/15% (2,800/600/600 configs) \- Training time: ~8 hours on 1 GPU (100 epochs) **Step 4: Accuracy Validation** \- **Target**: Energy MAE < 2 meV/atom, Force MAE < 60 meV/Å \- **Extrapolation Tests**: \- Check accuracy at adsorption sites not in training data (4-coordinate Cu sites) \- Check accuracy degradation at high temperature (700K) AIMD \- **Physical Quantity Reproduction**: \- CH₃OH adsorption energy (experimental value: ~-0.4 eV) \- Reaction barriers (compare with experiments) \- **If Failed**: Add uncertain configurations with Active Learning (+500-1,000 configs) **Step 5: Production Simulation** \- **System**: 8×8 Cu(111) surface (256 Cu atoms) + 10 CH₃OH molecules \- **Conditions**: 500K, atmospheric pressure equivalent \- **Time**: 10 ns (reaching experimental reaction timescale) \- **Expected Observations**: \- CH₃OH dehydrogenation reaction events (10-50 times) \- Statistical evaluation of reaction rate constants \- Identification of rate-limiting step \- **Computational Time**: ~3 days on 1 GPU **Expected Outcomes**: \- Elucidate methanol oxidation reaction mechanism \- Identify rate-limiting step \- Guidance for catalyst design (Cu alloying, etc.) 

### Problem 3 (Difficulty: hard)

Quantitatively compare how costs and time for the entire Cu catalyst CO₂ reduction reaction project (from data collection to simulation) change between standard MLP training without Active Learning and with Active Learning. Consider both supercomputer time and GPU time.

Hint \- Standard training: Calculate large amount of data (e.g., 15,000 configs) at once initially \- Active Learning: Start with small data (e.g., 500 configs), add through 3-4 iterations (each +500-1,000 configs) \- DFT calculation: 1 node-hour per configuration (1 hour on 1 supercomputer node) \- MLP training: 8 hours on 1 GPU per iteration \- Each iteration requires MLP-MD exploration (1 day on 1 GPU)  Sample Solution ### Scenario A: Standard MLP Training (without Active Learning) **Phase 1: Bulk Data Collection** 1\. Initial sampling plan: 2 weeks (human work) 2\. DFT calculation: 15,000 configs × 1 node-hour = **15,000 node-hours** \- Parallelization (100 nodes): 150 hours = **~6 days** 3\. Data preprocessing: 1 day **Phase 2: MLP Training** 1\. SchNet training: 12 hours on 1 GPU 2\. Accuracy validation: half a day 3\. Risk of insufficient accuracy: 30% probability requires retraining \- If failed: Add data (+5,000 configs, 2 days) + retrain (12 hours) **Phase 3: Simulation** 1\. 1 μs MLP-MD: 3 days on 1 GPU **Total (Success)**: \- Supercomputer: 15,000 node-hours \- GPU: 3.5 days \- Wall time: ~2 weeks (parallel execution) **Total (Failure, 30% probability)**: \- Supercomputer: 20,000 node-hours \- GPU: 4 days \- Wall time: ~3 weeks **Expected Value**: \- Supercomputer: 0.7×15,000 + 0.3×20,000 = **16,500 node-hours** \- GPU: 0.7×3.5 + 0.3×4 = **3.65 days** \- Wall time: ~**2.5 weeks** \--- ### Scenario B: With Active Learning **Phase 1: Initial Small Data Collection** 1\. Initial sampling: 3 days 2\. DFT calculation: 500 configs × 1 node-hour = **500 node-hours** (half day) 3\. Data preprocessing: half day **Phase 2: Iterative Cycles (3 iterations)** **Iteration 1**: 1\. MLP training v1.0: 4 hours on 1 GPU (faster with less data) 2\. Exploratory MLP-MD: 1 day on 1 GPU 3\. Uncertainty evaluation: half day 4\. Additional DFT: 800 configs × 1 node-hour = **800 node-hours** (1 day) **Iteration 2**: 1\. MLP training v1.1 (total 1,300 configs): 6 hours on 1 GPU 2\. Exploratory MLP-MD: 1 day on 1 GPU 3\. Uncertainty evaluation: half day 4\. Additional DFT: 600 configs × 1 node-hour = **600 node-hours** (1 day) **Iteration 3**: 1\. MLP training v1.2 (total 1,900 configs): 8 hours on 1 GPU 2\. Exploratory MLP-MD: 1 day on 1 GPU 3\. Accuracy validation: Pass (MAE < threshold) **Phase 3: Production Simulation** 1\. Final MLP training v2.0 (total 2,000 configs): 10 hours on 1 GPU 2\. 1 μs MLP-MD: 3 days on 1 GPU **Total**: \- Supercomputer: 500 + 800 + 600 = **1,900 node-hours** \- GPU: 0.17 + 1 + 0.25 + 1 + 0.33 + 1 + 0.42 + 3 = **7.17 days** \- Wall time: ~**2 weeks** (less parallelization) \--- ### Comparison Table | Item | Standard Training | Active Learning | Reduction | |------|---------|----------------|--------| | **Supercomputer Time** | 16,500 node-hours | 1,900 node-hours | **88% reduction** | | **GPU Time** | 3.65 days | 7.17 days | -96% (increase) | | **Wall Time** | 2.5 weeks | 2 weeks | 20% reduction | | **Total Data** | 15,000 configs | 2,000 configs | 87% reduction | | **Success Certainty** | 70% | 95% | Higher | \--- ### Cost Conversion (Assumption: Supercomputer 1 node-hour = $1, GPU 1 hour = $1) | Item | Standard Training | Active Learning | Difference | |------|---------|----------------|------| | Supercomputer cost | $16,500 | $1,900 | **-$14,600** | | GPU cost | $88 | $172 | +$84 | | **Total** | **$16,588** | **$2,072** | **-$14,516 (87% reduction)** | \--- ### Conclusion **Active Learning Advantages**: 1\. **88% reduction in supercomputer time** → Largest cost savings 2\. **7.5x improvement in data collection efficiency** (15,000 → 2,000 configs) 3\. **Improved success certainty** (70% → 95%) \- Reason: Automatically discovers important configurations (transition states, defects) 4\. **87% reduction in total cost** **Active Learning Disadvantages**: 1\. **GPU time approximately doubled** → But GPU is cheaper than supercomputer 2\. **Human intervention required** → Result verification at each iteration 3\. **Wall time approximately the same** → Lower parallelization degree **Recommendation**: \- **When supercomputer resources are limited**: Active Learning essential \- **Large projects**: Create initial model with Active Learning, then large-scale production simulations \- **Exploratory research**: Efficiently explore configuration space with Active Learning **Real Research Examples**: \- Nature Materials (2023) paper achieved **65% reduction in DFT costs** with Active Learning \- Phys. Rev. Lett. (2022) automatically discovered reaction intermediates, **shortening research period by 4 months** 

* * *

## 2.10 Data Licenses and Reproducibility

To ensure research reproducibility, we specify dataset licenses and code environment details used.

### 2.10.1 Major Dataset Licenses

Dataset | Description | License | Access  
---|---|---|---  
**MD17** | Molecular dynamics trajectories for small molecules (10 molecule types, 100k configs each) | CC0 1.0 (Public Domain) | [sgdml.org](<http://sgdml.org/#datasets>)  
**OC20** | Catalyst adsorption systems (1.3 million configs, 55 elements) | CC BY 4.0 | [opencatalystproject.org](<https://opencatalystproject.org>)  
**QM9** | Small organic molecules (134k molecules, 13 property types) | CC0 1.0 (Public Domain) | [quantum-machine.org](<http://quantum-machine.org/datasets/>)  
**ANI-1x** | Organic molecules (5 million configs, CCSD(T) accuracy) | CC BY 4.0 | [GitHub](<https://github.com/isayev/ANI1x_datasets>)  
**Materials Project** | Inorganic crystal structures (1.5 million structures, DFT-calculated) | CC BY 4.0 | [materialsproject.org](<https://materialsproject.org>)  
  
**Notes** :  
\- **Commercial Use** : CC BY 4.0 license allows commercial use with proper attribution  
\- **Paper Citation** : Always cite original papers when using datasets  
\- **Redistribution** : Redistribution allowed if license conditions met (MD17, QM9 are public domain)

### 2.10.2 Environment Information for Code Reproducibility

**Version Management for Major MLP Tools** :

Tool | Recommended Version | Installation Method | Notes  
---|---|---|---  
**SchNetPack** | 2.0.3 | `pip install schnetpack==2.0.3` | Major API changes from v2.0  
**NequIP** | 0.5.6 | `pip install nequip==0.5.6` | Check compatibility with e3nn 0.5.1  
**MACE** | 0.3.4 | `pip install mace-torch==0.3.4` | Requires PyTorch 2.0+  
**DeePMD-kit** | 2.2.7 | `pip install deepmd-kit==2.2.7` | Recommended TensorFlow 2.12  
**ASE** | 3.22.1 | `pip install ase==3.22.1` | Essential for MD calculations  
  
**DFT Calculation Parameter Recording Example** :
    
    
    # DFT settings recording for reproducibility (VASP example)
    system:
      name: "Cu(111) + CO2 adsorption"
      composition: "Cu64C1O2"
    
    dft_parameters:
      code: "VASP 6.3.2"
      functional: "PBE"
      dispersion: "D3(BJ)"
      encut: 500  # eV
      kpoints: "4x4x1 (Gamma-centered)"
      smearing: "Methfessel-Paxton, sigma=0.1 eV"
      convergence:
        electronic: 1.0e-6  # eV
        ionic: 1.0e-3  # eV/Angstrom
    
    structure:
      slab: "Cu(111) 4-layer, 4x4 supercell"
      vacuum: 15  # Angstrom
      fixed_layers: 2  # bottom 2 layers
    

### 2.10.3 Energy Unit Conversion Table

Conversion table for major energy units used in MLP:

Unit | eV | kcal/mol | Hartree (a.u.) | kJ/mol  
---|---|---|---|---  
**1 eV** | 1.0 | 23.061 | 0.036749 | 96.485  
**1 kcal/mol** | 0.043364 | 1.0 | 0.0015936 | 4.184  
**1 Hartree** | 27.211 | 627.51 | 1.0 | 2625.5  
**1 kJ/mol** | 0.010364 | 0.23901 | 0.00038088 | 1.0  
  
**Force Unit Conversion** :

Unit | eV/Å | kcal/(mol·Å) | Hartree/Bohr  
---|---|---|---  
**1 eV/Å** | 1.0 | 23.061 | 0.019447  
**1 kcal/(mol·Å)** | 0.043364 | 1.0 | 0.00084340  
**1 Hartree/Bohr** | 51.422 | 1185.8 | 1.0  
  
**Temperature and Energy Relationship** :
    
    
    k_B T (300 K) = 0.02585 eV = 0.596 kcal/mol
    
    Recommended accuracy targets:
    - Energy MAE < k_B T @ 300K  (< 0.026 eV = 0.6 kcal/mol)
    - Force MAE < 3 × k_B T / 1Å  (< 0.078 eV/Å = 1.8 kcal/mol/Å)
    

* * *

## 2.11 Practical Cautions: Common Failure Patterns

We summarize frequently encountered problems in MLP research and their solutions.

### 2.11.1 Dataset Design Failures

**Failure Example 1: Excessive Bias Toward Low-Energy Regions**
    
    
    # Problematic dataset composition
    Equilibrium structures: 8,000 configs (80%)
    AIMD trajectories: 1,500 configs (15%)
    Reaction pathways: 500 configs (5%)
    
    Problem: Insufficient data in high-energy regions (transition states, dissociation states)
    Result: Accuracy collapse in reaction pathway simulations
    
    Solution:
    1. Balance adjustment (equilibrium:AIMD:reaction = 40%:40%:20%)
    2. Add high-energy configurations with Active Learning
    3. Collect rare events with Enhanced Sampling (Metadynamics, etc.)
    

**Failure Example 2: Insufficient Temperature Range**
    
    
    Training data: 300K AIMD only
    
    Problem: Extrapolation errors when using at 500K MD
    Solution:
    - Cover wide temperature range during training (200K-600K)
    - Sample 1,000+ configs at each temperature
    - Always perform temperature extrapolation tests
    

### 2.11.2 Model Training Failures

**Failure Example 3: Cutoff Radius Selection Mistakes**

System | Inappropriate Cutoff | Recommended Cutoff | Reason  
---|---|---|---  
Metal surfaces | 3Å | 5-6Å | Metallic bonding is long-range  
Hydrogen bonding systems | 4Å | 6-8Å | Need to consider second solvation shell  
Ionic liquids | 5Å | 8-10Å | Electrostatic interactions are long-range  
Small organic molecules | 4Å (OK) | 4-5Å | Adjust according to molecule size  
  
**Diagnostic Method** :
    
    
    # Estimate cutoff from Radial Distribution Function (RDF)
    import numpy as np
    from ase.geometry import get_distances
    
    def estimate_cutoff(atoms, element='Cu'):
        """Estimate cutoff as third peak position in RDF"""
        positions = atoms.get_positions()
        distances = get_distances(positions, cell=atoms.cell, pbc=True)[1]
    
        # Calculate histogram
        rdf, bins = np.histogram(distances.flatten(), bins=200, range=(0, 12))
    
        # Find third peak position
        peaks = find_peaks(rdf, height=rdf.max()*0.3)[0]
        if len(peaks) >= 3:
            cutoff_estimate = bins[peaks[2]]
            print(f"Recommended cutoff: {cutoff_estimate:.2f} Å")
    
        return cutoff_estimate
    

**Failure Example 4: Loss Function Weight Setting Mistakes**
    
    
    # Bad setting example
    loss = 1.0 * MSE(E_pred, E_true) + 1.0 * MSE(F_pred, F_true)
    
    Problem: Unit mismatch
    - Energy: eV (absolute value 10¹-10³)
    - Force: eV/Å (absolute value 10⁻¹-10¹)
    → Energy learning dominates, force accuracy sacrificed
    
    Correct setting:
    loss = 1.0 * MSE(E_pred, E_true) + 100.0 * MSE(F_pred, F_true)
    # Or normalize both
    loss = MSE(E_pred/E_std, E_true/E_std) + MSE(F_pred/F_std, F_true/F_std)
    

### 2.11.3 Simulation Execution Failures

**Failure Example 5: Ignoring Energy Drift**
    
    
    # Bad example: Run long MD without checking drift
    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(10000000)  # 10 ns
    
    Problem: System heats/cools due to energy drift
    Diagnosis:
    total_energy = [a.get_total_energy() for a in traj]
    drift_per_step = (total_energy[-1] - total_energy[0]) / len(total_energy)
    
    if abs(drift_per_step) > 0.001:  # eV/step
        print("Warning: Significant energy drift detected")
        print("Solution 1: Halve timestep (1.0 fs → 0.5 fs)")
        print("Solution 2: Improve model accuracy (achieve force MAE < 0.05 eV/Å)")
        print("Solution 3: Use Langevin dynamics with thermostat coupling")
    

**Failure Example 6: Extrapolation to Unlearned Regions**
    
    
    Training data: 300K AIMD (energy range: E₀ ± 0.5 eV)
    Simulation: 1000K MD
    
    Problem: Reaching new configuration space at high temperature → Prediction accuracy collapse
    
    Preventive Measures:
    1. Ensemble uncertainty evaluation
       - Warn if standard deviation among multiple models exceeds threshold
    2. Gradual heating
       - Gradually increase temperature: 300K → 500K → 700K → 1000K
       - Perform Active Learning at each temperature
    3. Always perform extrapolation tests
       - Check accuracy on higher-energy configurations than training data
    

### 2.11.4 Unit Conversion Failures

**Failure Example 7: Hartree vs eV Confusion**
    
    
    # Bad example: Not checking DFT software default units
    energy_dft = read_dft_output()  # Hartree units
    
    # Direct input to SchNetPack
    dataset.add_sample(energy=energy_dft)  # Expects eV but actually Hartree
    
    Error example:
    - Training data energy overestimated by 27x
    - MAE appears small but actually a unit problem
    
    Correct method:
    HARTREE_TO_EV = 27.211386245988  # CODATA 2018
    energy_ev = energy_dft * HARTREE_TO_EV
    dataset.add_sample(energy=energy_ev)
    
    # Or handle units explicitly
    from ase import units
    energy_ev = energy_dft * units.Hartree
    

**Checklist: Unit Consistency Verification**

  1. [ ] Check DFT software output units (VASP: eV, Gaussian: Hartree, ORCA: Hartree)
  2. [ ] Check MLP library expected units (SchNetPack: eV, TorchANI: Hartree)
  3. [ ] Define conversion factors as constants (avoid magic numbers)
  4. [ ] Sanity check: Is energy in physically reasonable range (-10³ ~ 10³ eV)?
  5. [ ] Also verify force units simultaneously (eV/Å vs Hartree/Bohr)

* * *

## 2.12 End-of-Chapter Checklist: MLP Fundamentals Quality Assurance

Check whether you understand and can practice the content of this chapter.

### 2.12.1 Conceptual Understanding

**MLP Definition and Positioning**

  * [ ] Can explain the three essential elements of MLP (data-driven, high-dimensional approximation, physical constraints)
  * [ ] Can list three or more differences between MLP and conventional methods (empirical force fields, DFT)
  * [ ] Can explain contributions from quantum chemistry, machine learning, and molecular dynamics to MLP
  * [ ] Understand the concept of Potential Energy Surface (PES) and can illustrate it

**Understanding MLP Terminology**

  * [ ] Can accurately explain 10 or more of the 15 important terms
  * [ ] Can explain the difference between Invariance and Equivariance with concrete examples
  * [ ] Understand why E(3) equivariance is important
  * [ ] Know MAE target values (energy: <1-5 meV/atom, force: <50-150 meV/Å)

**Data and Workflow**

  * [ ] Can distinguish the uses of 5 input data types (equilibrium structures, AIMD, reaction pathways, random, defects)
  * [ ] Can explain the MLP workflow 5 steps in order
  * [ ] Can estimate typical computational time and required resources for each step
  * [ ] Understand the advantages and workflow of Active Learning

**Descriptors and Architectures**

  * [ ] Can compare characteristics of three types of descriptors: symmetry functions, SOAP, GNN
  * [ ] Understand differences in data efficiency (required configurations) for each descriptor
  * [ ] Can explain evolution and features of SchNet, DimeNet, NequIP, MACE
  * [ ] Can select appropriate descriptor/architecture for own research subject

### 2.12.2 Practical Skills

**Data Preparation**

  * [ ] Can select DFT calculation settings (functional, cutoff, k-points) needed for own system
  * [ ] Can plan balanced dataset composition (low/high energy regions)
  * [ ] Can design 5,000-configuration dataset composition for typical Cu catalyst system (see Exercise 2)
  * [ ] Can estimate DFT calculation time (configurations × time/configuration)

**Model Selection and Hyperparameters**

  * [ ] Can select cutoff radius based on system characteristics (metal: 5-6Å, hydrogen bonding systems: 6-8Å)
  * [ ] Can appropriately set loss function weights (w_E, w_F)
  * [ ] Can perform train/validation/test split (typically 80%/10%/10%)
  * [ ] Can adjust learning rate, batch size, epoch count

**Accuracy Evaluation**

  * [ ] Can calculate MAE on test set and compare with target values
  * [ ] Can create prediction vs true value correlation plots and calculate R²
  * [ ] Can design and conduct extrapolation tests (outside training data range)
  * [ ] Can verify reproducibility of physical quantities (lattice constants, vibrational spectra)

**Troubleshooting**

  * [ ] Can diagnose causes of insufficient accuracy (data shortage, hyperparameters, model selection)
  * [ ] Can implement Active Learning workflow (ensemble uncertainty evaluation)
  * [ ] Can accurately perform energy unit conversions (eV ↔ kcal/mol ↔ Hartree)
  * [ ] Can estimate appropriate cutoff radius from RDF

### 2.12.3 Application Ability

**Project Design**

  * [ ] Can make plans to apply MLP to new chemical system (own research subject)
  * [ ] Can estimate total project computational cost (DFT time, GPU time, personnel costs)
  * [ ] Can quantitatively compare costs between Active Learning vs standard training (see Exercise 3)
  * [ ] Can optimize dataset composition balance (equilibrium:AIMD:reaction pathways)

**Literature Understanding and Critical Evaluation**

  * [ ] Can understand method sections of MLP papers (descriptor, training data, accuracy)
  * [ ] Can judge whether reported MAE values are reasonable (compare with system size, complexity)
  * [ ] Can point out dataset biases (temperature range, energy distribution)
  * [ ] Can fairly compare performance of different architectures (SchNet vs NequIP)

**Research Strategy**

  * [ ] Can formulate data collection strategy with limited computational resources
  * [ ] Understand MLP limitations (extrapolation performance, transfer learning difficulties) and can take countermeasures
  * [ ] Can explain MLP's utility and limitations to experimental researchers
  * [ ] Aware of trends in next-generation methods (Foundation Models, Universal MLP)

**Self-Assessment of Achievement**

Category | Check Count | Evaluation  
---|---|---  
Conceptual Understanding | 14-16 / 16 | Excellent: Proceed to next chapter  
| 10-13 / 16 | Good: Some review recommended  
| < 10 / 16 | Needs Review: Re-read this chapter  
Practical Skills | 14-16 / 16 | Excellent: Can start practical projects  
| 10-13 / 16 | Good: Practice recommended with exercises  
| < 10 / 16 | Needs Review: Strengthen practical skills in Chapter 3  
Application Ability | 10-12 / 12 | Excellent: Can design research projects  
| 7-9 / 12 | Good: Case studies (Chapter 4) recommended  
| < 7 / 12 | Needs Review: Focus on exercises and case studies  
  
* * *

## References

  1. Behler, J., & Parrinello, M. (2007). "Generalized neural-network representation of high-dimensional potential-energy surfaces." _Physical Review Letters_ , 98(14), 146401.  
DOI: [10.1103/PhysRevLett.98.146401](<https://doi.org/10.1103/PhysRevLett.98.146401>)

  2. Bartók, A. P., et al. (2013). "On representing chemical environments." _Physical Review B_ , 87(18), 184115.  
DOI: [10.1103/PhysRevB.87.184115](<https://doi.org/10.1103/PhysRevB.87.184115>)

  3. Schütt, K. T., et al. (2017). "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." _Advances in Neural Information Processing Systems_ , 30.  
arXiv: [1706.08566](<https://arxiv.org/abs/1706.08566>)

  4. Klicpera, J., et al. (2020). "Directional message passing for molecular graphs." _International Conference on Learning Representations (ICLR)_.  
arXiv: [2003.03123](<https://arxiv.org/abs/2003.03123>)

  5. Batzner, S., et al. (2022). "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." _Nature Communications_ , 13(1), 2453.  
DOI: [10.1038/s41467-022-29939-5](<https://doi.org/10.1038/s41467-022-29939-5>)

  6. Batatia, I., et al. (2022). "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." _Advances in Neural Information Processing Systems_ , 35.  
arXiv: [2206.07697](<https://arxiv.org/abs/2206.07697>)

  7. Smith, J. S., et al. (2017). "ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost." _Chemical Science_ , 8(4), 3192-3203.  
DOI: [10.1039/C6SC05720A](<https://doi.org/10.1039/C6SC05720A>)

  8. Zhang, L., et al. (2018). "End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems." _Advances in Neural Information Processing Systems_ , 31.  
arXiv: [1805.09003](<https://arxiv.org/abs/1805.09003>)

  9. Schütt, K. T., et al. (2019). "Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions." _Nature Communications_ , 10(1), 5024.  
DOI: [10.1038/s41467-019-12875-2](<https://doi.org/10.1038/s41467-019-12875-2>)

  10. Musaelian, A., et al. (2023). "Learning local equivariant representations for large-scale atomistic dynamics." _Nature Communications_ , 14(1), 579.  
DOI: [10.1038/s41467-023-36329-y](<https://doi.org/10.1038/s41467-023-36329-y>)

* * *

## Author Information

**Author** : MI Knowledge Hub Content Team  
**Created** : 2025-10-17  
**Version** : 1.0 (Chapter 2 initial version)  
**Series** : MLP Introduction Series

**Update History** :  
\- 2025-10-19: v1.1 Quality improvement revision  
\- Added data license and reproducibility sections (MD17, OC20, QM9, ANI-1x, Materials Project)  
\- Added code reproducibility information (specified SchNetPack 2.0.3, NequIP 0.5.6, MACE 0.3.4 versions)  
\- Added energy/force unit conversion tables (eV, kcal/mol, Hartree, kJ/mol)  
\- Added practical cautions section (7 failure examples and solutions)  
\- Added end-of-chapter checklist (16 conceptual understanding items, 16 practical skill items, 12 application ability items)  
\- Added DFT calculation parameter recording example (VASP settings YAML format)  
\- 2025-10-17: v1.0 Initial Chapter 2 version  
\- Precise definition of MLP and positioning among related fields  
\- 15 important term glossary (concise table format)  
\- 5 major input data types  
\- MLP ecosystem diagram (Mermaid)  
\- 5-step workflow (detailed for each step)  
\- Descriptor types (symmetry functions, SOAP, GNN) and comparison table  
\- Comparison of major MLP architectures (7 methods)  
\- Active Learning column  
\- 4 learning objectives, 3 exercises (easy, medium, hard)  
\- 10 references

**License** : Creative Commons BY-NC-SA 4.0

[← Back to Series Table of Contents](<index.html>)
