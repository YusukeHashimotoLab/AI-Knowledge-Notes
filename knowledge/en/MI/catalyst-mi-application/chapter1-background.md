---
title: Catalysis Chemistry and the Role of Materials Informatics
chapter_title: Catalysis Chemistry and the Role of Materials Informatics
subtitle: Fundamentals of Catalyst Science and AI-Driven Development
reading_time: 25-30 min
difficulty: Beginner
code_examples: 0
exercises: 7
version: 1.0
created_at: 2025-10-17
---

# Chapter 1: Catalysis Chemistry and the Role of Materials Informatics

This chapter covers Catalysis Chemistry and the Role of Materials Informatics. You will learn Catalysis-Hub.org: https://www.catalysis-hub.org/ and Materials Project: https://materialsproject.org/.

## Learning Objectives

By completing this chapter, you will be able to:

  1. **Fundamentals of Catalysts** : Explain the definition, role, and key metrics of catalysts
  2. **Catalyst Classification** : Compare characteristics of homogeneous and heterogeneous catalysts
  3. **Problem Recognition** : Identify limitations of traditional catalyst development with concrete examples
  4. **MI Value** : Understand the value that Materials Informatics brings to catalyst development
  5. **Industry Trends** : Grasp the market size and latest trends in the catalyst industry

* * *

## 1.1 Fundamentals of Catalysts

### 1.1.1 What is a Catalyst

**Definition** : A catalyst is a substance that changes the rate of a chemical reaction while remaining chemically unchanged itself before and after the reaction.

**Role of Catalysts** : \- Reduction of activation energy (Ea) \- Enhancement of reaction rate (thousands to millions of times) \- Control of reaction selectivity
    
    
    Energy Diagram:
    
    Energy
        |
        |     Non-catalyzed reaction: Ea (high)
        |       /\
        |      /  \
        |     /    \
        | Catalyzed reaction: Ea' (low)
        |   /\  /\
        |  /  \/  \
        | /        \
        |_______________ Reaction Coordinate
       Reactants → Products
    
    ⇒ Catalyst reduces Ea' < Ea (activation energy reduction)
    ⇒ Reaction rate k = A * exp(-Ea / RT) significantly improves
    

**Important Properties** : 1\. **Activity** : Amount of product per unit time 2\. **Selectivity** : Proportion of target product 3\. **Stability** : Catalyst lifetime, resistance to degradation

### 1.1.2 Catalyst Performance Metrics

**Turnover Number (TON)** : \- Definition: Number of substrate molecules converted per catalyst molecule (or site) \- Formula: `TON = (moles of product) / (moles of catalyst)` \- Example: TON = 10,000 → one catalyst molecule converts 10,000 substrate molecules

**Turnover Frequency (TOF)** : \- Definition: TON per unit time \- Formula: `TOF = TON / time` \- Units: s⁻¹ or h⁻¹ \- Example: TOF = 100 s⁻¹ → one catalyst molecule converts 100 molecules per second

**Faradaic Efficiency (FE)** (for electrochemical catalysts): \- Definition: Proportion of input charge used for target product \- Formula: `FE = (charge to target product) / (total charge) × 100%` \- Example: For CO2 reduction, FE(CO) = 80% → 80% of charge contributes to CO production

* * *

## 1.2 Classification of Catalysts

### 1.2.1 Homogeneous vs Heterogeneous Catalysts

Item | Homogeneous Catalysts | Heterogeneous Catalysts  
---|---|---  
**Phase** | Same phase as reactants (mainly liquid) | Different phase from reactants (mainly solid)  
**Examples** | Transition metal complexes, acids/bases | Supported metal catalysts, zeolites  
**Active Sites** | Well-defined (molecular level) | Poorly defined (surface sites)  
**Selectivity** | High (stereoselective possible) | Medium to high  
**Recovery** | Difficult (separation challenging) | Easy (filtration/centrifugation)  
**Industrial Applications** | Pharmaceuticals, fine chemicals | Petroleum refining, chemical manufacturing  
**Main Reactions** | Hydrogenation, coupling | Catalytic cracking, reforming, oxidation  
  
**Example of Homogeneous Catalysts** :
    
    
    Wilkinson's Catalyst
    [RhCl(PPh3)3]
    
    Application: Hydrogenation of olefins
    Reaction: R-CH=CH2 + H2 → R-CH2-CH3
    Features: High selectivity, mild conditions (room temperature, atmospheric pressure)
    

**Example of Heterogeneous Catalysts** :
    
    
    Haber-Bosch Catalyst
    Fe3O4 / Al2O3 / K2O
    
    Application: Ammonia synthesis
    Reaction: N2 + 3H2 → 2NH3
    Conditions: 400-500°C, 200-300 atm
    Industrial scale: 170 million tons per year (worldwide)
    

### 1.2.2 Types of Representative Heterogeneous Catalysts

**1\. Metal Catalysts** \- **Examples** : Pt, Pd, Ni, Ru \- **Reactions** : Hydrogenation, dehydrogenation, oxidation \- **Features** : High activity but expensive (precious metals)

**2\. Metal Oxide Catalysts** \- **Examples** : TiO2, CeO2, V2O5 \- **Reactions** : Selective oxidation, photocatalysis \- **Features** : Inexpensive, thermally stable

**3\. Zeolite Catalysts** \- **Examples** : ZSM-5, Y-type zeolite \- **Reactions** : Catalytic cracking, isomerization \- **Features** : Shape selectivity due to pore structure

**4\. Supported Metal Catalysts** \- **Examples** : Pt/Al2O3, Pd/C \- **Structure** : Metal nanoparticles + support \- **Advantages** : Improved metal dispersion, increased surface area

**5\. Single-Atom Catalysts (SAC)** \- **Examples** : Pt1/CeO2 (Pt atoms dispersed as single atoms) \- **Advantages** : Minimized metal usage, high activity \- **Challenges** : Difficult synthesis and stabilization

* * *

## 1.3 Current Status and Challenges in Catalyst Development

### 1.3.1 Traditional Catalyst Development Process

**Typical Development Flow** :
    
    
    1. Literature survey & hypothesis formulation (weeks to months)
       ↓
    2. Catalyst preparation (days to weeks)
       - Selection of metal precursors
       - Choice of support
       - Investigation of preparation methods (impregnation, precipitation, ion exchange, etc.)
       ↓
    3. Characterization (days)
       - XRD, TEM, BET, XPS measurements
       ↓
    4. Activity evaluation (days to weeks)
       - Optimization of reaction conditions
       - Measurement of conversion and selectivity
       ↓
    5. Result analysis & next experiment planning (days)
       ↓
    6. Return to Step 2 (iteration)
    
    Total duration: Several years to over 10 years
    

**Problems** : \- Trial-and-error approach (Edison-type research) \- Enormous search space (combinations of composition and preparation conditions) \- High experimental costs (equipment, time, labor) \- Unclear reaction mechanisms

### 1.3.2 Challenge 1: Enormous Compositional Space

**Example: Exploration of Ternary Metal Catalysts**
    
    
    A-B-C system catalyst (e.g., Pt-Pd-Rh)
    
    Composition ratio: A:B:C = x:y:z (where x + y + z = 100%)
    Step size: Exploration in 10% increments
    
    Number of combinations:
    - Binary system (A-B): 11 combinations
    - Ternary system (A-B-C): 66 combinations
    - Quaternary system (A-B-C-D): 286 combinations
    
    Furthermore:
    - Types of supports: Al2O3, SiO2, CeO2, TiO2, ... (10 types)
    - Preparation methods: Impregnation, co-precipitation, hydrothermal synthesis, ... (5 types)
    
    Total combinations = 66 × 10 × 5 = 3,300 combinations
    
    ⇒ At 1 week per experiment, 3,300 weeks = approximately 63 years!
    

**Actual Chemical Space** : \- Periodic table: About 80 metallic elements \- Binary alloys: About 3,200 combinations \- Ternary alloys: About 82,000 combinations \- Quaternary and beyond: Millions to hundreds of millions of combinations

⇒ Exhaustive exploration is virtually impossible

### 1.3.3 Challenge 2: Complexity of Reaction Mechanisms

**Reaction Steps in Heterogeneous Catalysis** :
    
    
    Surface Reaction Mechanism (Example: CO oxidation)
    
    1. Adsorption of reactants
       CO(g) + * → CO*
       O2(g) + 2* → 2O*
    
    2. Surface diffusion
       CO* + O* → CO-O* (transition state)
    
    3. Surface reaction
       CO-O* → CO2* + *
    
    4. Desorption
       CO2* → CO2(g) + *
    
    Rate constants for each step: k1, k2, k3, k4
    
    Overall reaction rate:
    r = f(k1, k2, k3, k4, number of adsorption sites, coverage, ...)
    
    ⇒ Extremely complex!
    

**Problems** : \- Unknown rate-limiting step (which step is slow) \- Difficult in situ measurements (high temperature and pressure conditions) \- Theoretical calculations (DFT) also require enormous computational cost

### 1.3.4 Challenge 3: Difficulty in Predicting Catalyst Degradation

**Causes of Catalyst Degradation** : 1\. **Sintering** : Agglomeration and coarsening of metal particles 2\. **Poisoning** : Adsorption of impurities (S, Cl, CO) 3\. **Coking** : Blockage due to carbon deposition 4\. **Phase transformation** : Changes in crystal structure 5\. **Support collapse** : Thermal and chemical degradation

**Difficulty in Prediction** : \- Degradation is a long-term process (months to years) \- Multiple factors proceed simultaneously \- Accelerated tests deviate from practical conditions

* * *

## 1.4 Catalyst Development Challenges Solved by MI

### 1.4.1 Prediction of Activity and Selectivity

**Prediction by Machine Learning Models** :
    
    
    Inputs (descriptors):
    - Electronic descriptors: d-orbital occupancy, work function, band center
    - Geometric descriptors: Atomic radius, coordination number, surface area
    - Compositional descriptors: Element composition, electronegativity
    
    Machine learning models:
    - Random Forest
    - Gradient Boosting
    - Neural Network
    - Graph Neural Network
    
    Outputs:
    - Activity (TOF, conversion rate)
    - Selectivity (yield of target product)
    - Stability (lifetime prediction)
    

**Success Examples** : \- **CO oxidation catalyst** : Pt-Pd-Rh composition optimization \- Traditional: 100 sample experiments → 6 months \- MI: 20 sample experiments + predictive model → 1 month \- Result: 1.5x improvement in activity

### 1.4.2 Efficient Exploration by Bayesian Optimization

**Active Learning Cycle** :
    
    
    ```mermaid
    flowchart LR
        A["Initial Data10-20 samples"] --> B["Machine LearningModel Training"]
        B --> C["PredictionActivity of unknown compositions"]
        C --> D["Bayesian OptimizationSelection of next experiment"]
        D --> E["ExperimentActivity measurement"]
        E --> F{Goal achieved?}
        F -->|No| B
        F -->|Yes| G["Optimal catalyst discovered"]
    ```

**Advantages** : \- Reduction of experimental count to 1/5 to 1/10 \- Avoidance of local optima \- Exploration considering uncertainty

**Real Example** : \- **Hydrogen Evolution Reaction (HER) catalyst** \- Search space: Pt-Ni-Co-Fe quaternary system \- Traditional method: 286 samples required \- Bayesian optimization: Optimal composition found with 45 samples \- Duration reduction: Approximately 85% reduction

### 1.4.3 Integration with DFT Calculations

**Multi-Fidelity Approach** :
    
    
    High-Fidelity (high accuracy, high cost):
    - DFT calculations (First-Principles)
    - Computation time: Hours to days / 1 structure
    - Accuracy: High (±0.1 eV)
    
    Low-Fidelity (low accuracy, low cost):
    - Empirical models
    - Machine learning predictions
    - Computation time: Seconds / 1 structure
    - Accuracy: Medium (±0.3 eV)
    
    Integration strategy:
    1. Wide screening with Low-Fidelity (10,000 structures)
    2. Precise calculation of promising candidates (top 100) with High-Fidelity
    3. ML model training with both datasets
    4. Experimental validation of optimal structures
    
    ⇒ Computational cost reduced to 1/100 while maintaining high accuracy
    

### 1.4.4 Automated Analysis of Reaction Mechanisms

**AI-Assisted Transition State Exploration** :

Traditional DFT calculations: \- Researchers estimate transition states \- Explore using NEB (Nudged Elastic Band) method \- Days to weeks per pathway

AI-assisted approach: \- Machine learning predicts transition states \- Automatic pathway exploration \- Hours per pathway

Achievements: \- Transition state exploration time: 1/10 to 1/100 \- Discovery of overlooked pathways \- Comprehensive understanding of reaction mechanisms

* * *

## 1.5 Impact on the Catalyst Industry

### 1.5.1 Market Size and Growth

**Global Catalyst Market** : \- 2024: Approximately $40 billion \- 2030 forecast: Approximately $60 billion \- Compound Annual Growth Rate (CAGR): Approximately 6-7%

**Market Share by Sector** : 1\. **Petroleum refining** : 35% (catalytic cracking, reforming) 2\. **Chemical manufacturing** : 30% (ethylene, propylene) 3\. **Environmental catalysts** : 20% (automotive exhaust, deNOx) 4\. **Energy conversion** : 15% (fuel cells, hydrogen production)

**By Region** : \- Asia-Pacific: 40% (growth in China and India) \- North America: 30% \- Europe: 25% \- Others: 5%

### 1.5.2 Contribution to Carbon Neutrality

**Catalyst Technologies Contributing to CO2 Reduction** :

**1\. Green Hydrogen Production** \- Technology: Water electrolysis (OER/HER catalysts) \- Goal: H2 production using renewable energy \- Challenge: Development of high-efficiency, low-cost catalysts \- Market: Predicted at $5B in 2030

**2\. CO2 Reduction and Utilization (CCU)** \- Technology: CO2 → CO, CH4, C2H4 \- Catalysts: Cu, Ag, Au-based \- Goal: Faradaic efficiency > 80% \- Impact: Potential to reduce 1 million tons of CO2 annually

**3\. Ammonia Synthesis (Green Ammonia)** \- Traditional: Haber-Bosch (fossil fuel-derived H2) \- New technology: Electrolytic H2 + low-temperature catalysts \- Reduction: 90% reduction in CO2 emissions \- Applications: Fertilizers, fuel

**4\. Biomass Conversion** \- Feedstock: Cellulose, lignin \- Products: Chemicals, fuels \- Catalysts: Pt/C, Ru/C, zeolites \- Effect: Reduction in fossil resource dependency

### 1.5.3 Major Companies and Research Institutions

**Catalyst Manufacturers (Global Top)** : 1\. **BASF** (Germany): Market share approximately 20% 2\. **Johnson Matthey** (UK): Precious metal catalysts 3\. **Clariant** (Switzerland): Chemical process catalysts 4\. **W.R. Grace** (USA): Petroleum refining catalysts 5\. **Umicore** (Belgium): Automotive catalysts

**Japanese Companies** : \- **JGC Catalysts and Chemicals** : Petrochemical catalysts \- **N.E. Chemcat** : Fine chemical catalysts \- **Tanaka Kikinzoku Kogyo** : Precious metal catalysts

**Research Institutions** : \- **SLAC National Accelerator Laboratory** (USA) \- **Max Planck Institute for Chemical Energy Conversion** (Germany) \- **National Institute of Advanced Industrial Science and Technology (AIST)** (Japan) \- **Catalysis Society** (international organization)

### 1.5.4 Latest Trends in AI Catalyst Development

**Major Projects** :

**1\. Catalysis-Hub.org** (USA SUNCAT) \- Database: Over 20,000 catalytic reaction energies \- Publication of DFT calculation results \- Sharing of machine learning models

**2\. NREL Catalyst Discovery** (USA) \- High-throughput experimentation + ML \- Optimization of water electrolysis catalysts \- Provision of public datasets

**3\. BASF Digital Catalyst Design** (Germany) \- Internal AI catalyst development platform \- Integration of experimental data + DFT + ML \- Achievement of 50% reduction in development period

**4\. Toyota Fuel Cell Catalyst** (Japan) \- Reduction of Pt usage (single-atom catalysts) \- Composition optimization by ML \- Target for commercialization in 2025

**Academic Trends** : \- Number of papers ("AI" + "Catalyst"): 2020: 500 articles → 2024: Over 2,000 articles \- Top journals: _Nature Catalysis_ , _ACS Catalysis_ , _JACS_ \- International conferences: NAM (North American Catalysis Society Meeting), ICAT

* * *

## 1.6 History of Catalyst MI

### 1.6.1 Development History

**1960-1980s: Theoretical Foundations** \- Rediscovery of Sabatier Principle (1913) \- d-band theory (Hammer & Nørskov, 1995) \- Development of computational chemistry (DFT)

**1990-2000s: High-Throughput Experimentation** \- Combinatorial catalysis chemistry \- Parallel reaction devices (96-well plates) \- Automated analysis systems

**2010s: Introduction of Machine Learning** \- Descriptor-based ML models \- Application of Bayesian optimization \- DFT + ML integration

**2020s: AI-Driven Development** \- Graph Neural Network (GNN) \- Transfer Learning \- Autonomous Experimentation \- Large Language Models (ChatGPT for Chemistry)

### 1.6.2 Important Milestones

**2015: Nørskov Group (Stanford)** \- Paper: "Scaling Properties of Adsorption Energies for Hydrogen-Containing Molecules" \- Discovery of scaling relations for adsorption energies \- Impact: Descriptor reduction, improved prediction accuracy

**2017: Google Brain + SLAC** \- Paper: "Machine Learning in Catalysis" \- ML training with 300,000 DFT calculation data \- New catalyst prediction accuracy: R² = 0.92

**2020: Toyota** \- Practical application of single-atom catalyst (Pt1/CeO2) \- ML-assisted design \- 90% reduction in Pt usage

**2023: Autonomous Lab** \- IBM, University of Liverpool \- Fully automated robot experiments + AI \- 24-hour unmanned operation, 100 experiments per day

* * *

## Summary

In this chapter, we learned from the fundamentals of catalysts to industry trends and the role of MI:

### What We Learned

  1. **Catalyst Basics** : \- Reaction acceleration by reducing activation energy \- Key metrics: Activity, selectivity, stability \- Homogeneous vs heterogeneous catalysts

  2. **Challenges in Catalyst Development** : \- Enormous compositional space (thousands of combinations for ternary systems) \- Complexity of reaction mechanisms \- Difficulty in predicting degradation

  3. **Contribution of MI** : \- Prediction of activity and selectivity (ML) \- Efficient exploration by Bayesian optimization \- DFT + ML integration \- 50-85% reduction in development period

  4. **Industrial Impact** : \- Market size $40B → $60B (2030) \- Key to achieving carbon neutrality \- Active investment by major companies

### Next Steps

In Chapter 2, we will learn in detail about MI methods specialized for catalyst design: \- Design of catalyst descriptors \- Construction of activity prediction models \- Implementation of Bayesian optimization \- Integration methods with DFT calculations

**[Proceed to Chapter 2 →](<./chapter2-methods.html>)**

* * *

## Exercises

### Basic Level

**Problem 1** : Explain the three important metrics of catalysts (activity, selectivity, stability) in one sentence each.

**Problem 2** : Classify the following catalysts as homogeneous or heterogeneous, and give one advantage of each: \- Wilkinson's catalyst [RhCl(PPh3)3] \- Pt/Al2O3 \- H2SO4 \- Zeolite

**Problem 3** : How many combinations are there when exploring a ternary catalyst (A-B-C) in 10% increments? Show the calculation process.

### Intermediate Level

**Problem 4** : Describe the surface reaction mechanism for CO oxidation (CO + 1/2 O2 → CO2) in the following four steps: 1\. Adsorption 2\. Surface diffusion 3\. Surface reaction 4\. Desorption

**Problem 5** : Explain why Bayesian optimization is more efficient than traditional exhaustive search methods, using the keywords "uncertainty" and "acquisition function".

### Advanced Level

**Problem 6** : List three catalyst technologies that contribute to achieving carbon neutrality, and explain the following for each: \- Reaction equation \- Catalyst used (existing or under development) \- Estimation of CO2 reduction effect

**Problem 7** : Among the five causes of catalyst degradation (sintering, poisoning, coking, phase transformation, support collapse), select two that are easy to predict by machine learning and one that is difficult to predict, and explain the reasons.

* * *

## References

### Textbooks

  1. **Nørskov, J. K., et al.** (2011). _Fundamental Concepts in Heterogeneous Catalysis_. Wiley.

  2. **Bell, A. T., & Gates, B. C.** (2021). _Catalysis Science & Technology: Impact of Artificial Intelligence_. RSC Catalysis Series.

  3. **van Santen, R. A., et al.** (2008). _Catalysis: An Integrated Textbook for Students_. Elsevier.

### Important Papers

  1. **Nørskov, J. K., et al.** (2011). "Towards the computational design of solid catalysts." _Nature Chemistry_ , 3(5), 273-278.

  2. **Hammer, B., & Nørskov, J. K.** (1995). "Why gold is the noblest of all the metals." _Nature_ , 376, 238-240.

  3. **Ulissi, Z. W., et al.** (2017). "To address surface reaction network complexity using scaling relations machine learning and DFT calculations." _Nature Communications_ , 8, 14621.

### Online Resources

  * **Catalysis-Hub.org** : https://www.catalysis-hub.org/
  * **Materials Project** : https://materialsproject.org/
  * **ASE (Atomic Simulation Environment)** : https://wiki.fysik.dtu.dk/ase/

* * *

**Last Updated** : October 19, 2025 **Version** : 1.0
