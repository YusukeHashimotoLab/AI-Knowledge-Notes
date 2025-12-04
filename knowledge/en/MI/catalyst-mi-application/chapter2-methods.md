---
title: Materials Informatics Methods for Catalyst Design
chapter_title: Materials Informatics Methods for Catalyst Design
subtitle: From Descriptor Design to Bayesian Optimization
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 2
exercises: 7
version: 1.0
created_at: October
---

# Chapter 2: Materials Informatics Methods for Catalyst Design

This chapter covers Materials Informatics Methods for Catalyst Design. You will learn Catalysis-Hub: https://www.catalysis-hub.org/, Materials Project: https://materialsproject.org/, and ASE Documentation: https://wiki.fysik.dtu.dk/ase/.

## Learning Objectives

  1. **Descriptor Design** : Understand the 4 types of catalyst descriptors and how to use them appropriately
  2. **Prediction Models** : Explain the construction procedure for activity and selectivity prediction models
  3. **Bayesian Optimization** : Understand principles and application methods
  4. **DFT Integration** : Comprehend integration techniques for first-principles calculations and machine learning
  5. **Databases** : Understand characteristics and appropriate use of major catalyst databases

* * *

## 2.1 Catalyst Descriptors

### 2.1.1 Role of Descriptors

Descriptors are numerical representations of catalyst properties. They are used as inputs for machine learning models.

**Requirements for Good Descriptors** : \- ✅ Clear physical meaning \- ✅ Easy to compute \- ✅ Correlated with activity \- ✅ General applicability (applicable to different reaction systems)

### 2.1.2 Classification of Descriptors

**1\. Electronic Descriptors**

Descriptor | Definition | Relationship to Catalytic Activity  
---|---|---  
**d-Orbital Occupancy** | Number of electrons in d-orbitals of transition metals | Determines adsorption energy  
**d-band Center** | Center of gravity of d-orbital energy levels | Higher values lead to stronger adsorption  
**Work Function** | Energy required to remove electrons from surface | Affects electron transfer reactions  
**Bader Charge** | Localized atomic charge | Correlates with redox activity  
  
**d-band Theory (Nørskov)** :
    
    
    Adsorption Energy ∝ d-band Center Position
    
    d-band close to Fermi level
      → Increased occupation of antibonding orbitals
      → Strong adsorption
      → High activity (but slow desorption)
    
    Optimal d-band Center: Intermediate value (Sabatier principle)
    

**2\. Geometrical Descriptors**

Descriptor | Description | Example  
---|---|---  
**Coordination Number (CN)** | Number of neighboring atoms | Lower CN (edge, corner) is more active  
**Atomic Radius** | Size of metal atom | Correlates with lattice strain  
**Surface Area (BET)** | Specific surface area of catalyst | Larger area increases active sites  
**Pore Diameter** | Pore size of zeolites | Determines shape selectivity  
**Crystal Facet** | (111), (100), (110), etc. | Different active site densities  
  
**3\. Compositional Descriptors**

Descriptor | Definition | Application Example  
---|---|---  
**Elemental Composition** | Molar fraction of each element | Composition optimization  
**Electronegativity** | Strength of electron attraction | Redox activity  
**Ionic Radius** | Atomic size in ionic state | Interaction with support  
**Melting Point** | Melting point of metal | Indicator of thermal stability  
  
**4\. Reaction Descriptors**

Descriptor | Definition | Usage  
---|---|---  
**Adsorption Energy** | Energy when molecules adsorb on surface | Direct indicator of activity  
**Activation Energy** | Reaction barrier | Prediction of reaction rate  
**Transition State Energy** | Stability of transition state | Identification of rate-determining step  
  
* * *

## 2.2 Sabatier Principle and d-band Theory

### 2.2.1 Sabatier Principle

**Definition** : The optimal catalyst interacts with reaction intermediates with "just the right strength."
    
    
    Adsorption too weak:
      → Reactants don't stay on surface
      → Low activity
    
    Adsorption too strong:
      → Products can't desorb from surface
      → Low activity
    
    Optimal adsorption strength:
      → Peak of Volcano Plot
    

**Volcano Plot** :
    
    
    Activity (TOF)
        |
        |        *
        |      /   \
        |     /     \
        |    /       \
        |   /         \
        |  /           \
        |_________________ Adsorption Energy
       Weak  Optimal  Strong
    

### 2.2.2 Scaling Relations

Many adsorption energies have linear relationships:
    
    
    E(OH*) = 0.5 * E(O*) + 0.25 eV
    
    E(CHO*) = E(CO*) + 0.8 eV
    
    ⇒ One descriptor (e.g., E(O*)) can predict multiple adsorbates
    ⇒ Dimensionality reduction of descriptors
    

* * *

## 2.3 Activity and Selectivity Prediction Models

### 2.3.1 Regression Models (Activity Prediction)

**Objective** : Predict catalyst activity (TOF, conversion)

**Workflow** :
    
    
    1. Data Collection
       - Experimental data: Activity measurements
       - DFT data: Adsorption energies
    
    2. Descriptor Calculation
       - Electronic: d-band center
       - Geometrical: Coordination number
       - Compositional: Elemental composition
    
    3. Model Training
       - Random Forest
       - Gradient Boosting
       - Neural Network
    
    4. Performance Evaluation
       - R², RMSE, MAE
       - Cross-validation
    
    5. Prediction
       - Activity prediction for unknown catalysts
    

**Recommended Models** :

Model | Advantages | Disadvantages | Recommended Data Size  
---|---|---|---  
**Random Forest** | Interpretable, stable | Weak extrapolation | 100+  
**XGBoost** | High accuracy, fast | Many hyperparameters | 200+  
**Neural Network** | Learns complex relationships | Prone to overfitting | 500+  
**Gaussian Process** | Uncertainty quantification | Doesn't scale | <500  
  
### 2.3.2 Classification Models (Active Catalyst Screening)

**Objective** : Classify active vs. inactive catalysts

**Class Definition** :
    
    
    Active catalyst: TOF > Threshold (e.g., 1 s⁻¹)
    Inactive catalyst: TOF ≤ Threshold
    

**Evaluation Metrics** : \- **Precision** : Fraction of predicted active catalysts that are actually active \- **Recall** : Fraction of actual active catalysts correctly predicted \- **F1 Score** : Harmonic mean of Precision and Recall \- **ROC-AUC** : Overall evaluation of classification performance

* * *

## 2.4 Catalyst Discovery via Bayesian Optimization

### 2.4.1 Principles of Bayesian Optimization

**Objective** : Discover optimal catalyst with minimum number of experiments

**Components** : 1\. **Surrogate Model** : Gaussian Process 2\. **Acquisition Function** : Select next candidate to test

**Algorithm** :
    
    
    1. Initial experiments (10-20 samples)
       → Obtain composition and activity data
    
    2. Train surrogate model with Gaussian Process
       → Predict activity for unknown compositions (mean + uncertainty)
    
    3. Select next experiment using acquisition function
       - EI (Expected Improvement)
       - UCB (Upper Confidence Bound)
       - PI (Probability of Improvement)
    
    4. Conduct experiment with selected composition
    
    5. Update data and return to step 2
    
    6. Iterate until convergence criteria met
    

### 2.4.2 Comparison of Acquisition Functions

Acquisition Function | Formula | Characteristics | Recommended Scenario  
---|---|---|---  
**EI** | E[max(f(x) - f(x⁺), 0)] | Balanced | General purpose  
**UCB** | μ(x) + β·σ(x) | Exploration-focused | Broad exploration  
**PI** | P(f(x) > f(x⁺)) | Exploitation-focused | Local optimization  
  
**Parameter Tuning** : \- β (UCB exploration degree): Large initially (3.0), smaller later (1.0) \- ξ (EI trade-off): Typically 0.01-0.1

### 2.4.3 Multi-Objective Bayesian Optimization

**Objective** : Simultaneously optimize activity and selectivity

**Pareto Front** :
    
    
    Selectivity
        |
    100%|      * (ideal)
        |    *   *
        |  *       *
        | *         *
        |*___________*___ Activity (TOF)
       0%           High
    
    Pareto Front: Boundary where improving one objective worsens the other
    

**Methods** : \- ParEGO (Pareto Efficient Global Optimization) \- NSGA-II (Non-dominated Sorting Genetic Algorithm II) \- EHVI (Expected Hypervolume Improvement)

* * *

## 2.5 Integration with DFT Calculations

### 2.5.1 What is DFT (Density Functional Theory)?

**Objective** : Calculate electronic structure at atomic level based on quantum mechanics

**Calculable Properties** : \- Adsorption energy \- Activation energy (transition state) \- Electron density distribution \- Band structure

**Computational Cost** : \- 1 structure: Several hours to days (depends on CPU cores) \- Transition state search: Days to weeks

### 2.5.2 Multi-Fidelity Optimization

**Strategy** : Combine inexpensive low-fidelity and high-fidelity calculations
    
    
    Low-Fidelity:
    - Empirical models (bond-order force fields)
    - Small k-point mesh
    - Low cutoff energy
    - Cost: 1 minute/structure
    
    High-Fidelity:
    - Converged DFT calculation
    - Dense k-point mesh
    - High cutoff energy
    - Cost: 10 hours/structure
    
    Multi-Fidelity:
    1. Screen 10,000 structures with Low-Fidelity (~7 days)
    2. Calculate top 100 structures with High-Fidelity (~42 days)
    3. Train ML with both datasets
    4. Prediction accuracy: Equivalent to High-Fidelity alone
    5. Total cost: ~1/10
    

### 2.5.3 Transfer Learning

**Idea** : Transfer knowledge from existing reaction systems to new reaction systems

**Example** :
    
    
    Source Task: CO oxidation (large dataset available)
    Target Task: NO reduction (limited data)
    
    Procedure:
    1. Train DNN on Source Task
    2. Transfer learning on Target Task
       - Lower layers (general features): Fixed
       - Upper layers (task-specific): Retrain
    3. Required data: 1/5 to 1/10
    

* * *

## 2.6 Major Databases and Tools

### 2.6.1 Catalyst Databases

**1\. Catalysis-Hub.org** \- **Content** : 20,000+ catalyst reaction energies \- **Data** : DFT calculation results (adsorption energies, transition states) \- **Format** : JSON API, Python API \- **URL** : https://www.catalysis-hub.org/

**2\. Materials Project** \- **Content** : 140,000+ inorganic materials \- **Data** : Crystal structures, band gaps, formation energies \- **API** : Python (pymatgen) \- **URL** : https://materialsproject.org/

**3\. NIST Kinetics Database** \- **Content** : Chemical reaction rate constants \- **Data** : Arrhenius parameters (A, Ea) \- **Format** : Web search \- **URL** : https://kinetics.nist.gov/

### 2.6.2 Computational Tools

**1\. ASE (Atomic Simulation Environment)** \- **Language** : Python \- **Functions** : \- Structure optimization \- Vibrational analysis \- NEB (transition state search) \- Integration with various calculation engines (VASP, Quantum ESPRESSO) \- **Installation** : `conda install -c conda-forge ase`

**2\. Pymatgen** \- **Functions** : \- Read/write crystal structures \- Symmetry analysis \- Phase diagram calculation \- **Integration with Materials Project** \- **Installation** : `pip install pymatgen`

**3\. matminer** \- **Functions** : \- Automatic calculation of descriptors (200+ types) \- Data retrieval from databases \- Feature engineering \- **Installation** : `pip install matminer`

* * *

## 2.7 Catalyst MI Workflow

### Integrated Workflow
    
    
    ```mermaid
    flowchart TD
        A[Set Target Reaction] --> B[Initial Data Collection]
        B --> C[Descriptor Calculation]
        C --> D[ML Model Training]
        D --> E[Bayesian Optimization]
        E --> F[Candidate Catalyst Selection]
        F --> G{DFT Validation}
        G -->|Low Activity| E
        G -->|High Activity| H[Experimental Validation]
        H --> I{Goal Achieved?}
        I -->|No| C
        I -->|Yes| J[Optimal Catalyst]
    ```

### Implementation Example (Pseudocode)
    
    
    # Step 1: Data Collection
    data = load_catalysis_hub_data(reaction='CO_oxidation')
    
    # Step 2: Descriptor Calculation
    descriptors = calculate_descriptors(data['structures'])
    
    # Step 3: ML Model Training
    X_train, X_test, y_train, y_test = train_test_split(descriptors, data['activity'])
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Step 4: Bayesian Optimization
    optimizer = BayesianOptimization(model, acquisition='EI')
    for i in range(50):
        next_candidate = optimizer.suggest()
        dft_energy = run_dft(next_candidate)  # DFT calculation
        optimizer.update(next_candidate, dft_energy)
    
    # Step 5: Optimal Catalyst
    best_catalyst = optimizer.get_best()
    

* * *

## Summary

In this chapter, we learned Materials Informatics methods specialized for catalyst design:

### What We Learned

  1. **Descriptors** : 4 types - electronic, geometrical, compositional, and reaction descriptors
  2. **Sabatier Principle** : Optimal adsorption strength, volcano plots
  3. **Prediction Models** : When to use Random Forest, XGBoost, Neural Networks
  4. **Bayesian Optimization** : Efficient exploration, acquisition functions (EI, UCB, PI)
  5. **DFT Integration** : Multi-Fidelity, Transfer Learning
  6. **Databases** : Catalysis-Hub, Materials Project, ASE

### Next Steps

In Chapter 3, we'll implement catalyst MI in Python: \- Structure manipulation with ASE \- Building activity prediction models \- Composition exploration via Bayesian optimization \- Integration with DFT calculations \- 30 executable code examples

**[Proceed to Chapter 3 →](<./chapter3-hands-on.html>)**

* * *

## Exercises

### Basic Level

**Problem 1** : Explain why adsorption becomes stronger when the d-band center is close to the Fermi level in d-band theory.

**Problem 2** : Classify the following descriptors into electronic, geometrical, compositional, and reaction descriptors: \- Coordination number \- Adsorption energy \- Electronegativity \- Work function

**Problem 3** : Explain the Sabatier principle in three cases: "adsorption too weak," "too strong," and "optimal."

### Intermediate Level

**Problem 4** : Compare the three acquisition functions (EI, UCB, PI) in Bayesian optimization and explain in which situations each should be used.

**Problem 5** : Explain why Multi-Fidelity Optimization can reduce computational costs, including the characteristics of Low-Fidelity and High-Fidelity approaches.

### Advanced Level

**Problem 6** : In designing CO2 reduction catalysts, you need to simultaneously optimize activity and selectivity. Propose an exploration strategy using multi-objective Bayesian optimization. Include: \- Definition of objective functions \- Concept of Pareto front \- Specific acquisition functions

**Problem 7** : Design a strategy for using Transfer Learning to design catalysts for a new reaction system with limited data (e.g., ammonia decomposition). Explain what to choose as the Source Task and why.

* * *

## References

### Important Papers

  1. **Nørskov, J. K., et al.** (2011). "Towards the computational design of solid catalysts." _Nature Chemistry_ , 3, 273-278.

  2. **Ulissi, Z. W., et al.** (2017). "To address surface reaction network complexity using scaling relations machine learning and DFT calculations." _Nature Communications_ , 8, 14621.

  3. **Wertheim, M. K., et al.** (2020). "Bayesian optimization for catalysis." _ACS Catalysis_ , 10(20), 12186-12200.

### Databases & Tools

  * **Catalysis-Hub** : https://www.catalysis-hub.org/
  * **Materials Project** : https://materialsproject.org/
  * **ASE Documentation** : https://wiki.fysik.dtu.dk/ase/
  * **matminer** : https://hackingmaterials.lbl.gov/matminer/

* * *

**Last Updated** : October 19, 2025 **Version** : 1.0
