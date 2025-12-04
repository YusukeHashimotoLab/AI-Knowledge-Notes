---
title: "Chapter 2: Entropy and the Second Law"
chapter_title: "Chapter 2: Entropy and the Second Law"
subtitle: Fundamental Thermodynamic Principles Governing Irreversibility and Spontaneous Change
reading_time: 35-40 minutes
difficulty: Intermediate
code_examples: 5
version: 1.0
created_at: 2025-10-27
---

This chapter covers Entropy and the Second Law. You will learn  Explain the physical meaning of entropy,  Understand the difference between reversible, and  Explain the principles of the Carnot cycle.

Entropy is known as a "measure of disorder," but its essence is as the thermodynamic quantity that determines the direction of spontaneous change. The Second Law states that entropy always increases in an isolated system, explaining the irreversibility of nature. In materials science, entropy governs phase stability, mixing behavior in alloys, and defect formation. 

## Learning Objectives

By reading this chapter, you will master the following:

### Basic Understanding

  *  Understand multiple expressions of the Second Law of Thermodynamics (Clausius, Kelvin-Planck, entropy statements)
  *  Explain the physical meaning of entropy and its statistical mechanical interpretation (Boltzmann's relation)
  *  Understand the difference between reversible and irreversible processes, and the concept of entropy generation
  *  Explain the principles of the Carnot cycle and the limits of thermal efficiency

### Practical Skills

  *  Calculate entropy changes in reversible and irreversible processes
  *  Calculate configurational entropy (mixing entropy) of materials
  *  Calculate defect formation entropy and equilibrium defect concentration
  *  Visualize the relationship between statistical entropy and macrostates using Python

### Application

  *  Understand the role of mixing entropy in alloys and predict its influence on phase stability
  *  Explain the role of entropy in order-disorder transitions
  *  Evaluate the limits of thermal efficiency in materials processing

* * *

## 2.1 Formulation of the Second Law of Thermodynamics

### Spontaneous Change and Entropy

The First Law (conservation of energy) states that energy is conserved, but tells us nothing about the **direction of change**. For example:

  * When a hot object contacts a cold object, heat spontaneously flows from hot to cold
  * When gas is introduced into a container, it spontaneously diffuses throughout the entire volume
  * When two types of metals are mixed, they spontaneously mix together (under certain conditions)

The reverse processes (natural heat flow from cold to hot, gas spontaneously gathering on one side of a container) do not occur naturally, despite not violating energy conservation. This **direction of spontaneous change** is governed by the Second Law of Thermodynamics.

### Classical Statements of the Second Law

#### Clausius Statement

> Heat cannot spontaneously flow from a colder body to a hotter body. It is impossible to transfer heat from a colder to a hotter body without leaving any other change. 

#### Kelvin-Planck Statement

> It is impossible to construct a heat engine that absorbs heat from a single reservoir, converts it entirely to work, and leaves no other change (perpetual motion machine of the second kind). 

These two statements are equivalent and both describe the irreversible nature of heat energy.

### Entropy Statement

Clausius introduced a state function called **entropy (S)** to mathematically express the Second Law.

#### Definition of Entropy Change

In a **reversible process** , when the system absorbs an infinitesimal heat quantity $\delta Q_{\text{rev}}$ at temperature $T$, the entropy change is:

$$dS = \frac{\delta Q_{\text{rev}}}{T}$$

For a finite process:

$$\Delta S = \int \frac{\delta Q_{\text{rev}}}{T}$$

#### Clausius Inequality (Entropy Statement of the Second Law)

For **any process** (including reversible and irreversible):

$$dS \geq \frac{\delta Q}{T}$$

The equality holds for reversible processes, and the inequality for irreversible processes.

For an **isolated system** (no heat or matter exchange with surroundings):

$$\Delta S_{\text{isolated}} \geq 0$$

The equality signifies a reversible process (equilibrium state), and the inequality an irreversible process (spontaneous change).

#### =¡ Meaning of the Second Law

**The entropy of an isolated system never decreases**. This is a fundamental law of nature that determines the direction of time (from past to future). When a system reaches equilibrium, entropy reaches its maximum value, and no further spontaneous change occurs.

* * *

## 2.2 Statistical Mechanical Interpretation of Entropy

### Boltzmann's Relation

The physical meaning of entropy is revealed through statistical mechanics. Ludwig Boltzmann derived the relationship between entropy and the **number of microstates**.

#### Boltzmann Entropy

$$S = k_B \ln W$$

Where:

  * $S$: Entropy (J/K)
  * $k_B = 1.38 \times 10^{-23}$ J/K: Boltzmann constant
  * $W$: The **number of microstates** that realize the **macrostate**

### Macrostates and Microstates

**Macrostate** : A state characterized by macroscopically measurable quantities such as temperature, pressure, volume, and composition

**Microstate** : A state characterized by microscopic configurations such as the position and momentum of each particle. Many different microstates can exist for the same macrostate.

#### Example: Configurational Entropy of Coins

Consider tossing 10 coins and observing the outcomes of heads (H) and tails (T).

**Macrostate** : Number of heads (0-10)

**Microstate** : Specific configuration of each coin (e.g., HHTTHTHHTT)

**Macrostate with 5 heads and 5 tails** :

$$W = \binom{10}{5} = \frac{10!}{5! \cdot 5!} = 252 \text{ ways}$$

$$S = k_B \ln 252 \approx 5.53 k_B$$

**Macrostate with all heads** :

$$W = 1 \text{ way (only HHHHHHHHHH)}$$

$$S = k_B \ln 1 = 0$$

**Conclusion** : A state with equal numbers of heads and tails (high entropy) is overwhelmingly more likely to occur than a state where all are aligned (low entropy).

### Entropy and "Disorder"

The higher the entropy, the more microstates the system can access, meaning it is more **disordered**. Conversely, low entropy states are **ordered**.

**Examples in Materials Science** :

  * **Perfect crystal (0 K)** : Atoms arranged regularly ’ minimum entropy (Third Law: $S = 0$)
  * **Liquid** : Random atomic arrangement ’ higher entropy
  * **Gas** : Atoms move freely ’ highest entropy
  * **Random solid solution** : Different atoms mixed randomly ’ large mixing entropy

* * *

## 2.3 Calculation of Entropy Changes

### Entropy Change in Reversible Processes

**Isothermal process ($T = \text{const}$)** :

$$\Delta S = \frac{Q_{\text{rev}}}{T}$$

**Heating/cooling process** :

When changing a system with heat capacity $C$ from temperature $T_1$ to $T_2$:

$$\Delta S = \int_{T_1}^{T_2} \frac{C}{T} dT$$

If $C$ is temperature-independent:

$$\Delta S = C \ln\frac{T_2}{T_1}$$

**Isothermal expansion of ideal gas** :

$$\Delta S = nR \ln\frac{V_2}{V_1}$$

### Irreversible Processes and Entropy Generation

In irreversible processes, the entropy of the total system (system plus surroundings, forming an isolated system) increases.

**Entropy Production $\Delta S_{\text{gen}}$** :

$$\Delta S_{\text{total}} = \Delta S_{\text{system}} + \Delta S_{\text{surroundings}} = \Delta S_{\text{gen}} \geq 0$$

#### Example 2.1: Entropy Generation by Heat Conduction

**Problem** : A hot object (500 K, heat capacity 10 kJ/K) contacts a cold object (300 K, heat capacity 10 kJ/K), reaching thermal equilibrium. Find the total entropy change of the system.

View Solution

**Solution** :

**Step 1: Find the final temperature**

From energy conservation:

$$C_H (T_H - T_f) = C_C (T_f - T_C)$$

$$10 (500 - T_f) = 10 (T_f - 300)$$

$$T_f = 400 \text{ K}$$

**Step 2: Entropy change of each object**

Hot object: $\Delta S_H = C_H \ln\frac{T_f}{T_H} = 10 \ln\frac{400}{500} = 10 \times (-0.223) = -2.23$ kJ/K

Cold object: $\Delta S_C = C_C \ln\frac{T_f}{T_C} = 10 \ln\frac{400}{300} = 10 \times 0.288 = 2.88$ kJ/K

**Step 3: Total entropy change**

$$\Delta S_{\text{total}} = \Delta S_H + \Delta S_C = -2.23 + 2.88 = 0.65 \text{ kJ/K} > 0$$

**Conclusion** : The total entropy increased due to irreversible heat conduction. This means that the "quality" of the system has degraded, despite energy being conserved ($\Delta U = 0$).

### Entropy Change in Phase Transitions

During phase transitions (melting, vaporization, etc.), latent heat $L$ is absorbed or released at temperature $T$.

**Entropy change of fusion** :

$$\Delta S_{\text{fus}} = \frac{L_{\text{fus}}}{T_m}$$

**Entropy change of vaporization** :

$$\Delta S_{\text{vap}} = \frac{L_{\text{vap}}}{T_b}$$

Substance | Fusion Entropy (J/(mol·K)) | Vaporization Entropy (J/(mol·K))  
---|---|---  
Water (H‚O) | 22.0 | 109  
Iron (Fe) | 7.6 | 115  
Aluminum (Al) | 10.7 | 293  
Copper (Cu) | 9.6 | 305  
  
**Trouton's Rule** : For many liquids, the vaporization entropy is approximately 85-90 J/(mol·K) (except for polar molecules).

* * *

## 2.4 Entropy in Materials Science

### Configurational Entropy (Mixing Entropy)

In alloys and compounds, when different atoms are arranged randomly, the diversity of arrangements creates **configurational entropy**.

#### Ideal Mixing Entropy

When randomly placing $N_A$ A atoms and $N_B$ B atoms ($N_A + N_B = N$) on $N$ lattice sites, the number of microstates is:

$$W = \frac{N!}{N_A! \cdot N_B!}$$

Using Stirling's approximation ($\ln N! \approx N \ln N - N$):

$$S_{\text{config}} = k_B \ln W = -Nk_B (x_A \ln x_A + x_B \ln x_B)$$

Per mole ($Nk_B = R$):

$$\Delta S_{\text{mix}} = -R(x_A \ln x_A + x_B \ln x_B)$$

Where $x_A = N_A/N$, $x_B = N_B/N$ are mole fractions.

**Extension to multi-component systems** :

$$\Delta S_{\text{mix}} = -R \sum_{i} x_i \ln x_i$$

#### Example 2.2: Mixing Entropy of Cu-Ni Alloy

**Problem** : Find the mixing entropy of a random solid solution with 50 at% Cu and 50 at% Ni (per mole).

View Solution

**Solution** :

$$x_{\text{Cu}} = 0.5, \quad x_{\text{Ni}} = 0.5$$

$$\Delta S_{\text{mix}} = -R(x_{\text{Cu}} \ln x_{\text{Cu}} + x_{\text{Ni}} \ln x_{\text{Ni}})$$

$$= -8.314 \times (0.5 \ln 0.5 + 0.5 \ln 0.5)$$

$$= -8.314 \times (2 \times 0.5 \times (-0.693))$$

$$= 5.76 \text{ J/(mol·K)}$$

**Discussion** : This positive entropy change lowers the Gibbs energy upon mixing ($\Delta G = \Delta H - T\Delta S$), promoting alloy formation. At room temperature (298 K), there is an energy gain of $-T\Delta S_{\text{mix}} = -1717$ J/mol.

### Defect Formation Entropy

The formation of **point defects (vacancies, interstitials)** in crystals involves not only energy (enthalpy) but also configurational entropy.

**Equilibrium vacancy concentration** :

When creating $n$ vacancies on $N$ lattice sites, the number of configurations is:

$$W = \frac{N!}{n!(N-n)!}$$

Minimizing the Gibbs energy change for vacancy formation:

$$\frac{n}{N} = \exp\left(-\frac{\Delta G_f}{k_B T}\right) = \exp\left(\frac{\Delta S_f}{k_B}\right) \exp\left(-\frac{\Delta H_f}{k_B T}\right)$$

Where $\Delta H_f$ is the vacancy formation enthalpy (about 1 eV), and $\Delta S_f$ is the vacancy formation entropy (about $2k_B$).

Material | Vacancy Formation Enthalpy (eV) | Equilibrium Vacancy Concentration at Melting Point  
---|---|---  
Aluminum | 0.68 | $10^{-4}$  
Copper | 1.0 | $2 \times 10^{-4}$  
Iron | 1.4 | $2 \times 10^{-4}$  
Nickel | 1.6 | $10^{-4}$  
  
### Order-Disorder Transitions

In alloys, a transition can occur where atoms are arranged regularly at low temperature (**ordered phase**) and randomly at high temperature (**disordered phase**).

**Gibbs energy of ordering** :

$$\Delta G_{\text{order}} = \Delta H_{\text{order}} - T\Delta S_{\text{order}}$$

  * $\Delta H_{\text{order}} < 0$: Ordering reduces bond energy (stabilization)
  * $\Delta S_{\text{order}} < 0$: Ordering reduces configurational entropy (destabilization)

**Critical temperature $T_c$** : Temperature at which the order-disorder transition occurs

$$T_c \approx \frac{\Delta H_{\text{order}}}{\Delta S_{\text{order}}}$$

#### =¡ Implications for Materials Design

At high temperatures, the entropy term ($-T\Delta S$) dominates, and **disordered phases** (random solid solutions, liquid phases) become stable. At low temperatures, the enthalpy term ($\Delta H$) dominates, and **ordered phases** (ordered phases, intermetallic compounds) become stable. This temperature dependence determines the shape of phase diagrams.

* * *

## 2.5 Carnot Cycle and Thermal Efficiency

### Principle of the Carnot Cycle

The **Carnot cycle** is an **ideal reversible heat engine** operating between two heat reservoirs (high temperature $T_H$, low temperature $T_C$).
    
    
    ```mermaid
    flowchart TD
        A[1. Isothermal ExpansionT=T_H, Heat Absorption Q_H] --> B[2. Adiabatic ExpansionT_H ’ T_C]
        B --> C[3. Isothermal CompressionT=T_C, Heat Rejection Q_C]
        C --> D[4. Adiabatic CompressionT_C ’ T_H]
        D --> A
    
        style A fill:#ff9999,stroke:#cc0000,stroke-width:2px
        style B fill:#ffcc99,stroke:#ff9900,stroke-width:2px
        style C fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style D fill:#cc99ff,stroke:#9900cc,stroke-width:2px
    ```

### Carnot Efficiency

The **thermal efficiency** of the Carnot cycle is:

#### Carnot Efficiency

$$\eta_{\text{Carnot}} = \frac{W_{\text{net}}}{Q_H} = 1 - \frac{Q_C}{Q_H} = 1 - \frac{T_C}{T_H}$$

Where:

  * $W_{\text{net}}$: Net work ($= Q_H - Q_C$)
  * $Q_H$: Heat absorbed from the high-temperature reservoir
  * $Q_C$: Heat rejected to the low-temperature reservoir

**Important conclusions** :

  * The Carnot efficiency is the **upper limit of efficiency for any heat engine** operating between two given temperatures
  * To achieve 100% efficiency ($\eta = 1$), $T_C = 0$ K (absolute zero) is required ’ impossible to realize
  * Actual heat engines (internal combustion engines, thermal power plants) are always lower than Carnot efficiency due to irreversibility

#### Example 2.3: Maximum Efficiency of a Thermal Power Plant

**Problem** : Find the theoretical maximum efficiency of a thermal power plant with steam temperature 600°C (873 K) and cooling water temperature 25°C (298 K).

View Solution

**Solution** :

$$\eta_{\text{max}} = 1 - \frac{T_C}{T_H} = 1 - \frac{298}{873} = 1 - 0.341 = 0.659 = 65.9\%$$

**Actual efficiency** : Real thermal power plants have efficiencies around 40%. The difference from Carnot efficiency (25.9%) is lost due to the following irreversibilities:

  * Irreversibility of the combustion process
  * Temperature differences in heat transfer
  * Friction losses in turbines and piping

**Material constraints** : While increasing steam temperature improves efficiency, material heat resistance becomes the limiting factor. The latest ultra-supercritical pressure generation uses Ni-base superalloys to raise steam temperature above 700°C, achieving 45% efficiency.

### Application to Materials Processing

The concept of Carnot efficiency provides guidance for evaluating energy efficiency in materials processing:

  * **Steel production** : From the temperature difference between the blast furnace (1500°C) and the environment (25°C), the theoretical maximum efficiency is about 83%
  * **Heat treatment furnaces** : Waste heat recovery is effective for improving heating/cooling cycle efficiency
  * **Thermoelectric conversion materials** : High Seebeck coefficient and low thermal conductivity are required to approach Carnot efficiency

* * *

## 2.6 Entropy Calculations with Python

Below, we implement the concepts learned in this chapter using Python to deepen visual understanding. The code examples are executable and you can experiment by changing parameters.

**Note** : These codes are simplified models for educational purposes. Actual materials design requires high-accuracy calculations combining CALPHAD databases and DFT calculations.

* * *

## Chapter Summary

### What We Learned

  1. **Second Law of Thermodynamics**
     * Clausius statement, Kelvin-Planck statement, entropy statement
     * The entropy of an isolated system never decreases ($\Delta S \geq 0$)
     * Fundamental law determining the direction of spontaneous change
  2. **Statistical Mechanical Interpretation of Entropy**
     * Boltzmann equation: $S = k_B \ln W$ (relationship with number of microstates)
     * Entropy is a "measure of disorder"
     * The diversity of configurations realizing a macrostate creates entropy
  3. **Entropy Calculations**
     * Reversible process: $dS = \delta Q_{\text{rev}} / T$
     * Irreversible process: entropy generation $\Delta S_{\text{gen}} > 0$
     * Phase transitions: $\Delta S = L / T$
  4. **Entropy in Materials Science**
     * Configurational entropy (mixing entropy): $\Delta S_{\text{mix}} = -R\sum x_i \ln x_i$
     * Defect formation entropy and equilibrium defect concentration
     * Role of entropy in order-disorder transitions
  5. **Carnot Cycle**
     * Cycle of an ideal reversible heat engine
     * Carnot efficiency: $\eta = 1 - T_C / T_H$ (theoretical upper limit)
     * Material heat resistance determines the efficiency limit

### Important Points

  * Entropy is the most important thermodynamic quantity determining the direction of spontaneous change
  * At high temperatures, the entropy term dominates, and disordered phases (liquid, disordered solid solutions) become stable
  * Mixing entropy increases alloy solubility and significantly contributes to phase stability
  * The existence of defects (vacancies) is entropically favorable, and perfect crystals are metastable
  * Carnot efficiency provides the theoretical upper limit for heat engines and serves as a guide for materials development

### To the Next Chapter

In Chapter 3, we will learn the **fundamentals of phase equilibrium and phase diagrams** :

  * Gibbs energy and chemical potential
  * Conditions for phase equilibrium (chemical potential equality)
  * Gibbs phase rule and its applications
  * One-component phase diagrams (water, allotropic transformations of iron)
  * Understanding phase separation using the common tangent method
  * How to read materials state diagrams
