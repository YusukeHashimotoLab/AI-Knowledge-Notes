---
title: "Chapter 3: Fundamentals of Phase Equilibria and Phase Diagrams"
chapter_title: "Chapter 3: Fundamentals of Phase Equilibria and Phase Diagrams"
subtitle: Phase Equilibria and Phase Diagrams
difficulty: Intermediate
code_examples: 8
version: 1.0
created_at: "by:"
---

This chapter covers the fundamentals of Fundamentals of Phase Equilibria and Phase Diagrams, which 1. what is a phase?. You will learn essential concepts and techniques.

## Learning Objectives

By studying this chapter, you will acquire the following skills:

  * Understand the definition and types of phases and identify phases in materials
  * Explain the relationship between equilibrium conditions and chemical potential equilibrium
  * Apply the Gibbs phase rule (F = C - P + 2) to calculate the degrees of freedom in a system
  * Read and interpret unary phase diagrams (pressure-temperature diagrams)
  * Calculate phase boundaries using the Clapeyron and Clausius-Clapeyron equations
  * Understand the classification of phase transitions (first-order, second-order transitions) and provide examples
  * Calculate phase fractions using the lever rule
  * Draw phase diagrams using Python and predict phase transitions in real materials

## 1\. What is a Phase?

### 1.1 Definition of Phase

In materials science, a **phase** refers to a physically and chemically homogeneous region. Phases are distinguished from other phases by distinct interfaces.

#### Definition and Characteristics of Phase

A **phase** is a state of matter with the following characteristics:

  * **Uniform composition** : Chemical composition is the same at all locations within the phase
  * **Uniform physical properties** : Density, refractive index, crystal structure, etc., are constant
  * **Distinct interface** : A clear boundary exists between different phases
  * **Physically separable** : Can be separated from other phases in principle

### 1.2 Types of Phases

Various phases exist in materials:

Type of Phase | Description | Examples  
---|---|---  
**Gas phase** | Gaseous state. Large intermolecular distances, free motion | H‚O vapor, Ar atmosphere  
**Liquid phase** | Liquid state. Molecules closely packed but with fluidity | Liquid water, molten metal (Fe liquid)  
**Solid phase** | Solid state. Atoms arranged regularly or irregularly | Ice (H‚O solid), Fe crystal  
**Crystalline phase** | Solid phase with periodically arranged atoms | ±-Fe (BCC), ³-Fe (FCC)  
**Amorphous phase** | Solid phase without long-range order | Glass (SiO‚ amorphous), metallic glass  
  
#### Example: Phases of Pure Iron (Fe)

Pure iron exhibits different phases with different crystal structures depending on temperature:

  * **±-Fe (ferrite)** : Room temperature ~ 912°C, body-centered cubic (BCC) structure
  * **³-Fe (austenite)** : 912°C ~ 1394°C, face-centered cubic (FCC) structure
  * **´-Fe** : 1394°C ~ 1538°C (melting point), body-centered cubic (BCC) structure
  * **Liquid Fe** : Above 1538°C, atoms in irregular flow

These are called **allotropes** , phases of the same element with different crystal structures.

### 1.3 Difference Between Phase and Microstructure

#### Note: Phase and Microstructure are Different Concepts

  * **Phase** : Thermodynamically defined homogeneous region (± phase, ² phase, etc.)
  * **Microstructure** : Spatial arrangement and shape of phases (grain size, lamellar, spherical, etc.)

Example: Pearlite microstructure is a **microstructure** in which **two phases** , ±-Fe (ferrite) and FeƒC (cementite), are arranged in a lamellar pattern.

## 2\. Equilibrium State and Equilibrium Conditions

### 2.1 Definition of Equilibrium State

As learned in the previous chapter, under constant temperature and pressure, a system reaches equilibrium when **Gibbs energy (G) is minimized**. When multiple phases coexist, the equilibrium conditions are expressed more specifically.

#### Equilibrium Conditions in Multiphase Systems

For phases ±, ², ³ to coexist in equilibrium, the following conditions must be satisfied:

**1\. Thermal equilibrium** : Temperature is equal in all phases

$$ T^\alpha = T^\beta = T^\gamma $$ 

**2\. Mechanical equilibrium** : Pressure is equal in all phases (when interfacial tension is negligible)

$$ P^\alpha = P^\beta = P^\gamma $$ 

**3\. Chemical potential equilibrium** : Chemical potential of each component is equal in all phases

$$ \mu_i^\alpha = \mu_i^\beta = \mu_i^\gamma \quad \text{(for component $i$)} $$ 

### 2.2 Physical Meaning of Chemical Potential Equilibrium

The chemical potential equilibrium condition $\mu_i^\alpha = \mu_i^\beta$ means that "the driving force for component $i$ to transfer from ± phase to ² phase is zero."

#### Understanding Through Water Evaporation Equilibrium

Consider water in a cup repeatedly evaporating and condensing, eventually reaching a state where liquid and gas phases coexist:

  * **Non-equilibrium state** : $\mu_{\text{H}_2\text{O}}^{\text{liquid}} > \mu_{\text{H}_2\text{O}}^{\text{gas}}$ ’ Evaporation predominates
  * **Equilibrium state** : $\mu_{\text{H}_2\text{O}}^{\text{liquid}} = \mu_{\text{H}_2\text{O}}^{\text{gas}}$ ’ Evaporation and condensation are balanced

The gas pressure at this equilibrium state is the **saturated vapor pressure**.

### 2.3 Flow for Determining Equilibrium Conditions
    
    
    ```mermaid
    flowchart TD
        A[Initial state: Arbitrary temperature and pressure] --> B{Are T and P equalin all phases?}
        B -->|No| C[Thermal and mechanical equilibrationMake T and P uniform]
        C --> B
        B -->|Yes| D{For each component i,is ¼_i equal in all phases?}
        D -->|No| E[Mass transferHigh ¼ phase ’ Low ¼ phase]
        E --> D
        D -->|Yes| F[Chemical equilibrium achieved]
        F --> G[System Gibbs energyreaches minimum value]
        G --> H[Equilibrium state]
    
        style A fill:#fce7f3
        style F fill:#d1fae5
        style H fill:#dbeafe
    ```

## 3\. Gibbs Phase Rule

### 3.1 Derivation and Meaning of the Phase Rule

The Gibbs phase rule is an important relation that determines the **degrees of freedom** in a system at equilibrium.

#### Gibbs Phase Rule

$$ F = C - P + 2 $$ 

  * $F$: **Degrees of freedom** (number of intensive variables that can be varied independently)
  * $C$: **Number of components** (number of independent chemical components)
  * $P$: **Number of phases** (number of coexisting phases)
  * $2$: Two intensive variables, temperature and pressure

**Meaning of degrees of freedom $F$** : The number of variables that can be varied independently while maintaining equilibrium. If $F = 0$, invariant system (temperature, pressure, and composition all fixed); if $F = 1$, univariant system (e.g., pressure is determined when temperature is set).

### 3.2 Application Examples of the Phase Rule

=İ Code Example 1: Verification of Phase Rule in Various Systems Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 3.2 Application Examples of the Phase Rule
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import pandas as pd
    
    # Gibbs phase rule: F = C - P + 2
    
    # Degrees of freedom calculation for various systems
    systems = [
        {
            "System": "Pure water (single phase)",
            "Components C": 1,
            "Phases P": 1,
            "Degrees of freedom F": 1 - 1 + 2,
            "Example": "Liquid water only ’ T, P can be varied independently"
        },
        {
            "System": "Boiling water (two-phase)",
            "Components C": 1,
            "Phases P": 2,
            "Degrees of freedom F": 1 - 2 + 2,
            "Example": "Liquid + gas ’ P (vapor pressure) is determined when T is set"
        },
        {
            "System": "Water triple point",
            "Components C": 1,
            "Phases P": 3,
            "Degrees of freedom F": 1 - 3 + 2,
            "Example": "Solid + liquid + gas ’ T, P both fixed (0.01°C, 611 Pa)"
        },
        {
            "System": "Fe-C alloy (single phase)",
            "Components C": 2,
            "Phases P": 1,
            "Degrees of freedom F": 2 - 1 + 2,
            "Example": "³-Fe (austenite) only ’ T, P, composition x can be varied independently"
        },
        {
            "System": "Fe-C alloy (two-phase)",
            "Components C": 2,
            "Phases P": 2,
            "Degrees of freedom F": 2 - 2 + 2,
            "Example": "±-Fe + FeƒC ’ Composition of each phase is determined when T, P are set"
        },
        {
            "System": "Fe-C eutectic point",
            "Components C": 2,
            "Phases P": 3,
            "Degrees of freedom F": 2 - 3 + 2,
            "Example": "Liquid + ±-Fe + FeƒC ’ Others are determined when T or P is set"
        }
    ]
    
    df = pd.DataFrame(systems)
    
    print("=" * 80)
    print("Application Examples of Gibbs Phase Rule: F = C - P + 2")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Interpretation of degrees of freedom
    print("\nInterpretation of Degrees of Freedom")
    print("F = 0: Invariant system")
    print("       ’ All intensive variables are fixed (e.g., triple point)")
    print("\nF = 1: Univariant system")
    print("       ’ Other variables are determined when one variable is set (e.g., boiling curve)")
    print("\nF = 2: Bivariant system")
    print("       ’ Two variables can be varied independently (e.g., single-phase region)")
    print("\nF = 3: Trivariant system")
    print("       ’ Three variables can be varied independently (e.g., single phase in binary system)")
    

#### Note: Phase Rule Applies Only to Equilibrium States

The Gibbs phase rule is valid only when the system is in **thermodynamic equilibrium**. It cannot be applied in the following cases:

  * **Non-equilibrium states** : Metastable phases obtained by rapid cooling (martensite, etc.)
  * **Kinetic constraints** : States where equilibrium has not been reached due to slow reactions
  * **Interface effects** : Cases where interfacial energy dominates, such as nanoparticles

## 4\. Phase Diagrams of Unary Systems

### 4.1 Pressure-Temperature (P-T) Phase Diagram

Phase diagrams of unary systems (pure substances) are represented on a plot with **pressure (P) and temperature (T)** as axes. This diagram shows the regions where each phase is stable and the phase boundaries.

=İ Code Example 2: Pressure-Temperature Phase Diagram of Water (H‚O) Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Phase diagram data for water (simplified model)
    # The actual phase diagram is more complex, but simplified for educational purposes
    
    # Temperature range [°C]
    T_solid_liquid = np.array([-100, 0, 0.01])  # Melting curve
    T_liquid_vapor = np.linspace(0.01, 374, 100)  # Vaporization curve
    T_solid_vapor = np.linspace(-100, 0.01, 50)  # Sublimation curve
    
    # Vapor pressure by Clausius-Clapeyron equation (simplified)
    def vapor_pressure_liquid(T_celsius):
        """Vapor pressure at liquid-gas boundary (approximate equation)"""
        T = T_celsius + 273.15  # K
        # Simplified version of Antoine equation
        P = 0.611 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))  # kPa
        return P
    
    def vapor_pressure_solid(T_celsius):
        """Vapor pressure at solid-gas boundary (sublimation pressure)"""
        T = T_celsius + 273.15  # K
        # Sublimation pressure is lower than liquid phase (simplified)
        P = 0.611 * np.exp(21.87 * T_celsius / (T_celsius + 265.5))  # kPa
        return P
    
    # Pressure calculation
    P_solid_liquid = np.array([101.325, 101.325, 0.611])  # Melting curve (nearly vertical)
    P_liquid_vapor = vapor_pressure_liquid(T_liquid_vapor)
    P_solid_vapor = vapor_pressure_solid(T_solid_vapor)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Phase boundaries
    ax.plot(T_solid_liquid, P_solid_liquid, 'b-', linewidth=2.5, label='Solid-liquid boundary (melting curve)')
    ax.plot(T_liquid_vapor, P_liquid_vapor, 'r-', linewidth=2.5, label='Liquid-gas boundary (vaporization curve)')
    ax.plot(T_solid_vapor, P_solid_vapor, 'g-', linewidth=2.5, label='Solid-gas boundary (sublimation curve)')
    
    # Triple point
    T_triple = 0.01  # °C
    P_triple = 0.611  # kPa
    ax.plot(T_triple, P_triple, 'ko', markersize=12, label=f'Triple point ({T_triple}°C, {P_triple} kPa)', zorder=10)
    
    # Critical point
    T_critical = 374  # °C
    P_critical = 22064  # kPa
    ax.plot(T_critical, P_critical, 'rs', markersize=12, label=f'Critical point ({T_critical}°C, {P_critical/1000:.1f} MPa)', zorder=10)
    
    # Phase region labels
    ax.text(-50, 50000, 'Solid\n(Ice)', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(100, 50000, 'Liquid\n(Water)', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    ax.text(100, 0.05, 'Gas\n(Steam)', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Atmospheric pressure line
    ax.axhline(101.325, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Atmospheric pressure (101.325 kPa)')
    ax.text(-80, 101.325 * 1.2, '1 atm', fontsize=10, color='gray')
    
    ax.set_xlabel('Temperature [°C]', fontsize=13)
    ax.set_ylabel('Pressure [kPa]', fontsize=13)
    ax.set_title('Pressure-Temperature Phase Diagram of Water (H‚O)', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim([0.01, 100000])
    ax.set_xlim([-100, 400])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('water_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Important points in the water phase diagram:")
    print(f"ûTriple point: {T_triple}°C, {P_triple} kPa (three-phase coexistence, F=0)")
    print(f"ûCritical point: {T_critical}°C, {P_critical/1000:.1f} MPa (distinction between liquid and gas disappears)")
    print(f"ûBoiling point at atmospheric pressure: 100°C (intersection of liquid-gas boundary and atmospheric pressure line)")
    print(f"ûMelting point at atmospheric pressure: 0°C (intersection of solid-liquid boundary and atmospheric pressure line)")
    

### 4.2 How to Read Phase Diagrams

#### Basic Rules for Reading Phase Diagrams

  * **Regions** : Regions where a single phase is stable (solid region, liquid region, gas region)
  * **Boundaries** : Lines where two phases coexist (two-phase equilibrium lines), $F = 1$
  * **Triple point** : Point where three phases coexist, $F = 0$ (both temperature and pressure fixed)
  * **Critical point** : Point where the distinction between liquid and gas phases disappears (supercritical fluid)

#### Practical Example: Ice Polymorphs Under High Pressure

The water phase diagram actually contains **more than 15 ice polymorphs** (Ice I, II, III, ..., XV). Phases other than ordinary ice (Ice Ih) are stable only under high pressure:

  * **Ice Ih** : Ordinary ice at atmospheric pressure (hexagonal)
  * **Ice III** : Stable at approximately 300 MPa, around -20°C
  * **Ice VII** : Above approximately 2 GPa, high density (possible existence in Earth's deep interior)

These are important research subjects in planetary science and materials physics.

## 5\. Clapeyron Equation and Clausius-Clapeyron Equation

### 5.1 Clapeyron Equation

An important relation that expresses the slope of the phase boundary ($dP/dT$) in terms of the enthalpy and entropy of phase transition.

#### Clapeyron Equation

The slope of the transition boundary from phase ± to phase ² is:

$$ \frac{dP}{dT} = \frac{\Delta S_{\text{trans}}}{\Delta V_{\text{trans}}} = \frac{\Delta H_{\text{trans}}}{T \Delta V_{\text{trans}}} $$ 

  * $\Delta H_{\text{trans}}$: Transition enthalpy (heat of fusion, heat of vaporization, etc.) [J/mol]
  * $\Delta V_{\text{trans}}$: Molar volume change accompanying transition [m³/mol]
  * $\Delta S_{\text{trans}}$: Transition entropy change [J/(mol·K)]
  * $T$: Transition temperature [K]

=İ Code Example 3: Calculation of Melting Curve Slope Using Clapeyron Equation Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: $$ \frac{dP}{dT} = \frac{\Delta S_{\text{trans}}}{\Delta V_{
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Data for water melting
    T_m = 273.15  # Melting point [K]
    Delta_H_fusion = 6010  # Heat of fusion [J/mol]
    V_ice = 19.65e-6  # Molar volume of ice [m^3/mol]
    V_water = 18.02e-6  # Molar volume of water [m^3/mol]
    Delta_V = V_water - V_ice  # Molar volume change (negative value!)
    
    # Slope of melting curve by Clapeyron equation
    dP_dT = Delta_H_fusion / (T_m * Delta_V)  # [Pa/K]
    
    print("=" * 60)
    print("Calculation of Water Melting Curve Slope Using Clapeyron Equation")
    print("=" * 60)
    print(f"Melting point (0°C): {T_m} K")
    print(f"Heat of fusion: {Delta_H_fusion} J/mol = {Delta_H_fusion/1000:.2f} kJ/mol")
    print(f"Molar volume of ice: {V_ice * 1e6:.3f} cm³/mol")
    print(f"Molar volume of water: {V_water * 1e6:.3f} cm³/mol")
    print(f"Molar volume change ”V: {Delta_V * 1e6:.3f} cm³/mol (negative value!)")
    print(f"\nMelting curve slope dP/dT: {dP_dT:.2e} Pa/K")
    print(f"                          = {dP_dT / 1e6:.2f} MPa/K")
    print(f"                          = {dP_dT / 101325:.2f} atm/K")
    print("=" * 60)
    
    print("\nInterpretation")
    print("ûdP/dT < 0 (negative slope): Melting point decreases with increasing pressure")
    print("ûThis abnormal behavior is due to ice having a larger volume than water")
    print("ûThe slipperiness of ice skating is explained (partially) by this pressure melting")
    print("ûMost substances have dP/dT > 0 (melting point increases with pressure)")
    
    # Drawing the melting curve
    P_0 = 101325  # Atmospheric pressure [Pa]
    T_range = np.linspace(270, 276, 100)  # [K]
    P_range = P_0 + dP_dT * (T_range - T_m)  # [Pa]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Melting curve
    ax1.plot((T_range - 273.15), P_range / 1e6, linewidth=2.5, color='#3b82f6')
    ax1.axhline(P_0 / 1e6, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Atmospheric pressure')
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0°C')
    ax1.plot(0, P_0 / 1e6, 'ro', markersize=10, label='Melting point (atmospheric pressure)', zorder=5)
    ax1.set_xlabel('Temperature [°C]', fontsize=12)
    ax1.set_ylabel('Pressure [MPa]', fontsize=12)
    ax1.set_title('Solid-Liquid Boundary of Water (Melting Curve)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3, 3])
    
    # Right plot: Melting point change due to pressure
    pressures = np.linspace(0, 100, 100)  # [MPa]
    T_melt = T_m + pressures * 1e6 / dP_dT  # [K]
    ax2.plot(pressures, T_melt - 273.15, linewidth=2.5, color='#f093fb')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Pressure [MPa]', fontsize=12)
    ax2.set_ylabel('Melting Point [°C]', fontsize=12)
    ax2.set_title('Change in Melting Point Due to Pressure', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Annotation: Pressure effect
    ax2.text(50, -0.2, f'At 100 MPa, melting point\ndrops by approximately {(T_melt[-1] - T_m):.2f} K',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#fce7f3', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('clapeyron_equation_water.png', dpi=150, bbox_inches='tight')
    plt.show()
    

### 5.2 Clausius-Clapeyron Equation

At the liquid-gas or solid-gas boundary, the volume of the gas phase is overwhelmingly larger than the liquid or solid phase ($V_{\text{gas}} \gg V_{\text{liquid/solid}}$), so the Clapeyron equation can be simplified.

#### Clausius-Clapeyron Equation

For the liquid-gas boundary (vaporization curve):

$$ \frac{d \ln P}{dT} = \frac{\Delta H_{\text{vap}}}{RT^2} $$ 

Or, for pressure change from temperature change $T_1 \to T_2$:

$$ \ln \frac{P_2}{P_1} = -\frac{\Delta H_{\text{vap}}}{R} \left( \frac{1}{T_2} - \frac{1}{T_1} \right) $$ 

  * $\Delta H_{\text{vap}}$: Vaporization enthalpy [J/mol]
  * $R = 8.314$ J/(mol·K): Gas constant
  * $P$: Vapor pressure [Pa]

=İ Code Example 4: Vapor Pressure Curve Using Clausius-Clapeyron Equation Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: $$ \ln \frac{P_2}{P_1} = -\frac{\Delta H_{\text{vap}}}{R} \l
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Gas constant
    R = 8.314  # J/(mol·K)
    
    # Heat of vaporization data for various substances
    substances = {
        'H‚O': {'Delta_H_vap': 40660, 'T_boil': 373.15, 'P_boil': 101325, 'color': '#3b82f6'},
        'Ethanol': {'Delta_H_vap': 38560, 'T_boil': 351.45, 'P_boil': 101325, 'color': '#10b981'},
        'Acetone': {'Delta_H_vap': 29100, 'T_boil': 329.15, 'P_boil': 101325, 'color': '#f093fb'},
        'Benzene': {'Delta_H_vap': 30720, 'T_boil': 353.25, 'P_boil': 101325, 'color': '#f59e0b'}
    }
    
    # Temperature range [K]
    T = np.linspace(250, 400, 200)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Temperature dependence of vapor pressure (logarithmic scale)
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        P_boil = data['P_boil']
        color = data['color']
    
        # Clausius-Clapeyron equation
        ln_P_P0 = -(Delta_H / R) * (1/T - 1/T_boil)
        P = P_boil * np.exp(ln_P_P0)  # [Pa]
    
        ax1.plot(T - 273.15, P / 1000, linewidth=2.5, label=name, color=color)
    
        # Mark boiling point
        ax1.plot(T_boil - 273.15, P_boil / 1000, 'o', markersize=8, color=color)
    
    ax1.axhline(101.325, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Atmospheric pressure')
    ax1.set_xlabel('Temperature [°C]', fontsize=12)
    ax1.set_ylabel('Vapor Pressure [kPa]', fontsize=12)
    ax1.set_title('Vapor Pressure Curves of Various Substances', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim([0.1, 1000])
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right plot: ln(P) vs 1/T (verification of linear relationship)
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        P_boil = data['P_boil']
        color = data['color']
    
        T_inv = 1 / T  # [1/K]
        ln_P_P0 = -(Delta_H / R) * (1/T - 1/T_boil)
        ln_P = np.log(P_boil) + ln_P_P0
    
        ax2.plot(T_inv * 1000, ln_P, linewidth=2.5, label=name, color=color)
    
        # Annotation of slope (first substance only)
        if name == 'H‚O':
            slope = -Delta_H / R
            ax2.text(3.2, 10, f'Slope = -”H_vap/R\n= {slope:.0f} K',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#fce7f3', alpha=0.8))
    
    ax2.set_xlabel('1000/T [1000/K]', fontsize=12)
    ax2.set_ylabel('ln(P) [ln(Pa)]', fontsize=12)
    ax2.set_title('Clausius-Clapeyron Plot', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # Make temperature increase direction to the right
    
    plt.tight_layout()
    plt.savefig('clausius_clapeyron_equation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=" * 70)
    print("Application of Clausius-Clapeyron Equation")
    print("=" * 70)
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        print(f"\n{name}")
        print(f"  Boiling point (1 atm): {T_boil - 273.15:.2f} °C")
        print(f"  Heat of vaporization: {Delta_H/1000:.1f} kJ/mol")
    
        # Calculate vapor pressure at 80°C
        T_target = 273.15 + 80  # K
        ln_ratio = -(Delta_H / R) * (1/T_target - 1/T_boil)
        P_target = 101325 * np.exp(ln_ratio)
        print(f"  Vapor pressure at 80°C: {P_target/1000:.2f} kPa")
    
    print("\n" + "=" * 70)
    print("Practical example: Vacuum distillation")
    print("Substances with low vapor pressure (high boiling point) can be distilled at low temperatures by reducing pressure.")
    print("Example: When water is distilled at 50 mmHg (approximately 6.7 kPa), the boiling point drops to approximately 33°C.")
    

## 6\. Classification of Phase Transitions

### 6.1 Ehrenfest Classification

Phase transitions are classified by the continuity of derivatives of Gibbs energy (Ehrenfest classification).

Type of Phase Transition | Definition | Characteristics | Examples  
---|---|---|---  
**First-order phase transition** | $G$ is continuous, but  
$\frac{\partial G}{\partial T}$ is discontinuous | ûLatent heat present  
ûVolume change present  
ûTwo-phase coexistence | Melting, boiling, sublimation,  
allotropic transformation (Fe: ±’³)  
**Second-order phase transition** | $G$, $\frac{\partial G}{\partial T}$ are continuous, but  
$\frac{\partial^2 G}{\partial T^2}$ is discontinuous | ûNo latent heat  
ûNo volume change  
ûDiscontinuity in heat capacity | Paramagnetic-ferromagnetic transition (Fe: 770°C),  
superconducting transition  
  
#### How to Distinguish First-Order and Second-Order Phase Transitions

  * **Presence of latent heat** : In first-order phase transitions, heat is absorbed/released at the transition temperature (heat of fusion, heat of vaporization, etc.). In second-order phase transitions, latent heat is zero.
  * **Volume change** : In first-order phase transitions, density differs between phases (volume decreases from ice to water). In second-order phase transitions, it is continuous.
  * **Entropy change** : In first-order phase transitions, $\Delta S = \Delta H / T$ is discontinuous. In second-order phase transitions, it is continuous.

=İ Code Example 5: Gibbs Energy for First-Order and Second-Order Phase Transitions Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: How to Distinguish First-Order and Second-Order Phase Transi
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Temperature range
    T = np.linspace(200, 400, 500)  # [K]
    T_transition = 300  # Transition temperature [K]
    
    # Model for first-order phase transition (assuming melting)
    Delta_H_fusion = 5000  # J/mol
    Delta_S_fusion = Delta_H_fusion / T_transition  # J/(mol·K)
    
    G_solid_1st = 1000 * (T - 200)  # Gibbs energy of solid phase (simplified)
    G_liquid_1st = 1000 * (T - 200) - Delta_S_fusion * (T - T_transition)  # Liquid phase
    
    # Selection of stable phase (lower Gibbs energy)
    G_stable_1st = np.minimum(G_solid_1st, G_liquid_1st)
    
    # Model for second-order phase transition (assuming paramagnetic-ferromagnetic transition)
    # Gibbs energy is continuous, but second derivative (heat capacity) is discontinuous
    G_paramagnetic = 1000 * (T - 200) + 0.5 * (T - T_transition)**2
    G_ferromagnetic = 1000 * (T - 200) + 2.0 * (T - T_transition)**2
    
    # Selection of stable phase
    G_stable_2nd = np.where(T < T_transition, G_ferromagnetic, G_paramagnetic)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ===== First-order phase transition =====
    # Upper left: Gibbs energy
    axes[0, 0].plot(T, G_solid_1st / 1000, 'b--', linewidth=2, label='Solid phase', alpha=0.7)
    axes[0, 0].plot(T, G_liquid_1st / 1000, 'r--', linewidth=2, label='Liquid phase', alpha=0.7)
    axes[0, 0].plot(T, G_stable_1st / 1000, 'k-', linewidth=3, label='Stable phase')
    axes[0, 0].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Temperature [K]', fontsize=11)
    axes[0, 0].set_ylabel('Gibbs Energy [kJ/mol]', fontsize=11)
    axes[0, 0].set_title('First-Order Phase Transition: Gibbs Energy', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Upper middle: Entropy (first derivative)
    S_solid = -np.gradient(G_solid_1st, T)
    S_liquid = -np.gradient(G_liquid_1st, T)
    S_stable = -np.gradient(G_stable_1st, T)
    axes[0, 1].plot(T, S_solid, 'b--', linewidth=2, alpha=0.7)
    axes[0, 1].plot(T, S_liquid, 'r--', linewidth=2, alpha=0.7)
    axes[0, 1].plot(T, S_stable, 'k-', linewidth=3)
    axes[0, 1].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Temperature [K]', fontsize=11)
    axes[0, 1].set_ylabel('Entropy [J/(mol·K)]', fontsize=11)
    axes[0, 1].set_title('First-Order Phase Transition: Entropy (Discontinuous)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(T_transition + 10, np.mean(S_stable), '”S = ”H/T\n(Latent heat present)',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Upper right: Heat capacity (second derivative)
    C_p_solid = T * np.gradient(S_solid, T)
    C_p_liquid = T * np.gradient(S_liquid, T)
    C_p_stable = T * np.gradient(S_stable, T)
    axes[0, 2].plot(T, C_p_stable, 'k-', linewidth=3)
    axes[0, 2].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('Temperature [K]', fontsize=11)
    axes[0, 2].set_ylabel('Heat Capacity C_p [J/(mol·K)]', fontsize=11)
    axes[0, 2].set_title('First-Order Phase Transition: Heat Capacity (Divergence)', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-500, 500])
    
    # ===== Second-order phase transition =====
    # Lower left: Gibbs energy
    axes[1, 0].plot(T, G_paramagnetic / 1000, 'b--', linewidth=2, label='Paramagnetic phase', alpha=0.7)
    axes[1, 0].plot(T, G_ferromagnetic / 1000, 'r--', linewidth=2, label='Ferromagnetic phase', alpha=0.7)
    axes[1, 0].plot(T, G_stable_2nd / 1000, 'k-', linewidth=3, label='Stable phase')
    axes[1, 0].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Temperature [K]', fontsize=11)
    axes[1, 0].set_ylabel('Gibbs Energy [kJ/mol]', fontsize=11)
    axes[1, 0].set_title('Second-Order Phase Transition: Gibbs Energy (Continuous)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Lower middle: Entropy (first derivative, continuous)
    S_para = -np.gradient(G_paramagnetic, T)
    S_ferro = -np.gradient(G_ferromagnetic, T)
    S_stable_2nd = -np.gradient(G_stable_2nd, T)
    axes[1, 1].plot(T, S_stable_2nd, 'k-', linewidth=3)
    axes[1, 1].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Temperature [K]', fontsize=11)
    axes[1, 1].set_ylabel('Entropy [J/(mol·K)]', fontsize=11)
    axes[1, 1].set_title('Second-Order Phase Transition: Entropy (Continuous)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(T_transition + 10, np.mean(S_stable_2nd), '”S = 0\n(No latent heat)',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Lower right: Heat capacity (second derivative, discontinuous)
    C_p_stable_2nd = T * np.gradient(S_stable_2nd, T)
    axes[1, 2].plot(T, C_p_stable_2nd, 'k-', linewidth=3)
    axes[1, 2].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('Temperature [K]', fontsize=11)
    axes[1, 2].set_ylabel('Heat Capacity C_p [J/(mol·K)]', fontsize=11)
    axes[1, 2].set_title('Second-Order Phase Transition: Heat Capacity (Discontinuous)', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_transition_classification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Summary of phase transition classification:")
    print("\nFirst-Order Phase Transition")
    print("ûG: Continuous, S = -G/T: Discontinuous (latent heat present)")
    print("ûExamples: Melting (solid ’ liquid), vaporization (liquid ’ gas)")
    print("ûMeasurement: Peak appears in DSC (differential scanning calorimetry)")
    print("\nSecond-Order Phase Transition")
    print("ûG: Continuous, S: Continuous, C_p = T S/T: Discontinuous")
    print("ûExamples: Ferromagnetic-paramagnetic transition (Curie temperature of Fe: 770°C)")
    print("ûMeasurement: Step in heat capacity appears in DSC (not a peak)")
    

## 7\. Lever Rule

### 7.1 Principle of the Lever Rule

In a two-phase region, when the average composition of the system is given, the quantity ratio (phase fraction) of each phase is determined by the **lever rule**. This is mathematically the same as the "principle of the lever."

#### Lever Rule Formula

In a binary system, when the composition of ± phase is $x_\alpha$, the composition of ² phase is $x_\beta$, and the average composition of the system is $x_{\text{avg}}$ where $x_\alpha < x_{\text{avg}} < x_\beta$:

$$ \frac{n_\beta}{n_\alpha} = \frac{x_{\text{avg}} - x_\alpha}{x_\beta - x_{\text{avg}}} $$ 

Or, as mole fractions (mass fractions) of each phase:

$$ f_\alpha = \frac{x_\beta - x_{\text{avg}}}{x_\beta - x_\alpha}, \quad f_\beta = \frac{x_{\text{avg}} - x_\alpha}{x_\beta - x_\alpha} $$ 

Here, $f_\alpha + f_\beta = 1$.

#### Geometric Meaning of the Lever Rule

Imagine a lever. The fulcrum is at the average composition $x_{\text{avg}}$, the left end is at $x_\alpha$, and the right end is at $x_\beta$.

  * **Length of left arm (± phase side)** : $x_{\text{avg}} - x_\alpha$
  * **Length of right arm (² phase side)** : $x_\beta - x_{\text{avg}}$

From the lever balance condition: $n_\alpha \times (x_{\text{avg}} - x_\alpha) = n_\beta \times (x_\beta - x_{\text{avg}})$

Rearranging this gives the lever rule formula above.

=İ Code Example 6: Calculation and Visualization of Lever Rule Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Two-phase region settings
    x_alpha = 0.2  # Composition of ± phase
    x_beta = 0.8   # Composition of ² phase
    
    # Average composition range (within two-phase region)
    x_avg_range = np.linspace(x_alpha, x_beta, 100)
    
    # Phase fraction calculation by lever rule
    def lever_rule(x_avg, x_alpha, x_beta):
        """Calculate fraction of each phase using lever rule"""
        f_alpha = (x_beta - x_avg) / (x_beta - x_alpha)
        f_beta = (x_avg - x_alpha) / (x_beta - x_alpha)
        return f_alpha, f_beta
    
    f_alpha_range = []
    f_beta_range = []
    for x_avg in x_avg_range:
        f_a, f_b = lever_rule(x_avg, x_alpha, x_beta)
        f_alpha_range.append(f_a)
        f_beta_range.append(f_b)
    
    f_alpha_range = np.array(f_alpha_range)
    f_beta_range = np.array(f_beta_range)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Change in phase fraction
    ax1.plot(x_avg_range, f_alpha_range, linewidth=3, color='#3b82f6', label='± phase fraction $f_\\alpha$')
    ax1.plot(x_avg_range, f_beta_range, linewidth=3, color='#f5576c', label='² phase fraction $f_\\beta$')
    ax1.fill_between(x_avg_range, 0, f_alpha_range, alpha=0.3, color='#3b82f6')
    ax1.fill_between(x_avg_range, f_alpha_range, 1, alpha=0.3, color='#f5576c')
    
    # Boundary lines
    ax1.axvline(x_alpha, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(x_beta, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(x_alpha, 1.05, f'$x_\\alpha={x_alpha}$', ha='center', fontsize=10)
    ax1.text(x_beta, 1.05, f'$x_\\beta={x_beta}$', ha='center', fontsize=10)
    
    # Mark specific example
    x_example = 0.5
    f_a_ex, f_b_ex = lever_rule(x_example, x_alpha, x_beta)
    ax1.plot(x_example, f_a_ex, 'o', markersize=12, color='#10b981', zorder=5)
    ax1.plot(x_example, f_b_ex, 'o', markersize=12, color='#10b981', zorder=5)
    ax1.text(x_example + 0.05, f_a_ex, f'$x_{{avg}}={x_example}$\n$f_\\alpha={f_a_ex:.2f}$\n$f_\\beta={f_b_ex:.2f}$',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='#d1fae5', alpha=0.9))
    
    ax1.set_xlabel('Average Composition $x_{avg}$', fontsize=12)
    ax1.set_ylabel('Phase Fraction [-]', fontsize=12)
    ax1.set_title('Calculation of Phase Fraction Using Lever Rule', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='center left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_alpha - 0.05, x_beta + 0.05])
    ax1.set_ylim([0, 1.1])
    
    # Right plot: Illustration of lever principle
    ax2.axis('off')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Drawing the lever
    lever_y = 0.6
    ax2.plot([x_alpha, x_beta], [lever_y, lever_y], 'k-', linewidth=4)
    ax2.plot([x_alpha], [lever_y], 'o', markersize=15, color='#3b82f6', label='± phase')
    ax2.plot([x_beta], [lever_y], 'o', markersize=15, color='#f5576c', label='² phase')
    ax2.plot([x_example], [lever_y], '^', markersize=15, color='#10b981', label='Fulcrum (average composition)')
    
    # Display distances
    ax2.plot([x_alpha, x_example], [lever_y - 0.1, lever_y - 0.1], 'b-', linewidth=2)
    ax2.text((x_alpha + x_example) / 2, lever_y - 0.15, f'${x_example - x_alpha:.1f}$',
             ha='center', fontsize=11, color='blue')
    ax2.plot([x_example, x_beta], [lever_y - 0.1, lever_y - 0.1], 'r-', linewidth=2)
    ax2.text((x_example + x_beta) / 2, lever_y - 0.15, f'${x_beta - x_example:.1f}$',
             ha='center', fontsize=11, color='red')
    
    # Force arrows (corresponding to phase amounts)
    arrow_y = 0.75
    ax2.arrow(x_alpha, arrow_y, 0, -0.08, head_width=0.03, head_length=0.03,
              fc='#3b82f6', ec='#3b82f6', linewidth=2)
    ax2.text(x_alpha, arrow_y + 0.05, f'$n_\\alpha$ or $f_\\alpha={f_a_ex:.2f}$',
             ha='center', fontsize=10, color='#3b82f6')
    
    ax2.arrow(x_beta, arrow_y, 0, -0.08, head_width=0.03, head_length=0.03,
              fc='#f5576c', ec='#f5576c', linewidth=2)
    ax2.text(x_beta, arrow_y + 0.05, f'$n_\\beta$ or $f_\\beta={f_b_ex:.2f}$',
             ha='center', fontsize=10, color='#f5576c')
    
    # Explanation text
    ax2.text(0.5, 0.35, 'Lever balance condition:\n$n_\\alpha \\times$ (right arm) $= n_\\beta \\times$ (left arm)',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.9))
    ax2.text(0.5, 0.2, f'$\\frac{{n_\\beta}}{{n_\\alpha}} = \\frac{{{x_example - x_alpha:.1f}}}{{{x_beta - x_example:.1f}}} = {f_b_ex / f_a_ex:.2f}$',
             ha='center', fontsize=12)
    
    ax2.set_title('Geometric Meaning of Lever Rule', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('lever_rule.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Specific calculation examples
    print("=" * 70)
    print("Lever Rule Calculation Examples")
    print("=" * 70)
    print(f"± phase composition: x_± = {x_alpha}")
    print(f"² phase composition: x_² = {x_beta}")
    print(f"\nCase 1Average composition x_avg = {x_example}")
    print(f"  ± phase fraction: f_± = (x_² - x_avg) / (x_² - x_±) = {f_a_ex:.3f} ({f_a_ex*100:.1f}%)")
    print(f"  ² phase fraction: f_² = (x_avg - x_±) / (x_² - x_±) = {f_b_ex:.3f} ({f_b_ex*100:.1f}%)")
    print(f"  Verification: f_± + f_² = {f_a_ex + f_b_ex:.3f} ")
    
    x_example2 = 0.3
    f_a_ex2, f_b_ex2 = lever_rule(x_example2, x_alpha, x_beta)
    print(f"\nCase 2Average composition x_avg = {x_example2} (closer to ± phase)")
    print(f"  ± phase fraction: f_± = {f_a_ex2:.3f} ({f_a_ex2*100:.1f}%)")
    print(f"  ² phase fraction: f_² = {f_b_ex2:.3f} ({f_b_ex2*100:.1f}%)")
    print(f"  ’ More ± phase (because average composition is closer to ± phase)")
    
    x_example3 = 0.7
    f_a_ex3, f_b_ex3 = lever_rule(x_example3, x_alpha, x_beta)
    print(f"\nCase 3Average composition x_avg = {x_example3} (closer to ² phase)")
    print(f"  ± phase fraction: f_± = {f_a_ex3:.3f} ({f_a_ex3*100:.1f}%)")
    print(f"  ² phase fraction: f_² = {f_b_ex3:.3f} ({f_b_ex3*100:.1f}%)")
    print(f"  ’ More ² phase (because average composition is closer to ² phase)")
    print("=" * 70)
    

### 7.2 Practical Example of Lever Rule

#### Carbon Content and Microstructure in Steel

Consider the microstructure of Fe-C alloy (steel) at 727°C (eutectoid temperature):

  * **±-Fe (ferrite)** : C concentration 0.02 wt%
  * **FeƒC (cementite)** : C concentration 6.7 wt%
  * **Eutectoid steel** : C concentration 0.76 wt%

Calculate the mass ratio of ferrite to cementite in eutectoid steel using the lever rule:

$f_{\text{Fe}_3\text{C}} = \frac{0.76 - 0.02}{6.7 - 0.02} = \frac{0.74}{6.68} \approx 0.11$ (11%)

$f_{\alpha\text{-Fe}} = 1 - 0.11 = 0.89$ (89%)

In other words, eutectoid steel consists of approximately 89% ferrite and 11% cementite.

## 8\. Calculation of Phase Transformation Enthalpy

=İ Code Example 7: Determination and Visualization of Triple Point Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 8. Calculation of Phase Transformation Enthalpy
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    # Water phase transition data
    data = {
        'Melting': {
            'Delta_H': 6010,  # J/mol
            'T_0': 273.15,    # K (0°C)
            'P_0': 101325,    # Pa
            'Delta_V': (18.02e-6 - 19.65e-6),  # m^3/mol (liquid - solid)
        },
        'Vaporization': {
            'Delta_H': 40660,  # J/mol
            'T_0': 373.15,     # K (100°C)
            'P_0': 101325,     # Pa
        },
        'Sublimation': {
            'Delta_H': 51000,  # J/mol (close to sum of melting + vaporization)
            'T_0': 273.15,     # K
            'P_0': 611,        # Pa (estimated)
        }
    }
    
    R = 8.314  # J/(mol·K)
    
    # Equations for each boundary line
    def melting_curve(T):
        """Solid-liquid boundary (Clapeyron)"""
        dP_dT = data['Melting']['Delta_H'] / (data['Melting']['T_0'] * data['Melting']['Delta_V'])
        P = data['Melting']['P_0'] + dP_dT * (T - data['Melting']['T_0'])
        return P
    
    def vaporization_curve(T):
        """Liquid-gas boundary (Clausius-Clapeyron)"""
        Delta_H = data['Vaporization']['Delta_H']
        T_0 = data['Vaporization']['T_0']
        P_0 = data['Vaporization']['P_0']
        ln_ratio = -(Delta_H / R) * (1/T - 1/T_0)
        P = P_0 * np.exp(ln_ratio)
        return P
    
    def sublimation_curve(T):
        """Solid-gas boundary (Clausius-Clapeyron)"""
        Delta_H = data['Sublimation']['Delta_H']
        T_0 = data['Sublimation']['T_0']
        P_0 = data['Sublimation']['P_0']
        ln_ratio = -(Delta_H / R) * (1/T - 1/T_0)
        P = P_0 * np.exp(ln_ratio)
        return P
    
    # Determination of triple point (intersection of melting curve and sublimation curve)
    def triple_point_equation(T):
        """Condition at triple point: melting curve pressure = sublimation curve pressure"""
        return melting_curve(T) - sublimation_curve(T)
    
    T_triple = fsolve(triple_point_equation, 273.15)[0]
    P_triple = sublimation_curve(T_triple)
    
    print("=" * 60)
    print("Triple Point Calculation Results")
    print("=" * 60)
    print(f"Temperature: {T_triple:.2f} K = {T_triple - 273.15:.2f} °C")
    print(f"Pressure: {P_triple:.2f} Pa = {P_triple / 1000:.3f} kPa")
    print("\nExperimental value (literature value):")
    print("Temperature: 273.16 K = 0.01 °C")
    print("Pressure: 611.657 Pa = 0.612 kPa")
    print("\n’ Calculated value agrees well with experimental value")
    print("=" * 60)
    
    # Temperature range
    T_range = np.linspace(250, 400, 300)
    P_melt = melting_curve(T_range)
    P_vap = vaporization_curve(T_range)
    P_sub = sublimation_curve(T_range)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Phase boundaries
    T_melt_range = np.linspace(260, 280, 100)
    T_vap_range = np.linspace(273, 374, 100)
    T_sub_range = np.linspace(250, 273.16, 100)
    
    ax.plot(T_melt_range - 273.15, melting_curve(T_melt_range) / 1000,
            'b-', linewidth=3, label='Solid-liquid boundary')
    ax.plot(T_vap_range - 273.15, vaporization_curve(T_vap_range) / 1000,
            'r-', linewidth=3, label='Liquid-gas boundary')
    ax.plot(T_sub_range - 273.15, sublimation_curve(T_sub_range) / 1000,
            'g-', linewidth=3, label='Solid-gas boundary')
    
    # Triple point
    ax.plot(T_triple - 273.15, P_triple / 1000, 'ko', markersize=14, zorder=10,
            label=f'Triple point\n({T_triple - 273.15:.2f}°C, {P_triple / 1000:.3f} kPa)')
    
    # Phase rule display
    ax.text(-15, 50, 'F = 2\n(Single-phase region)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(50, 50, 'F = 1\n(Two-phase coexistence line)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(T_triple - 273.15 - 5, P_triple / 1000 + 0.3, 'F = 0\n(Three-phase coexistence point)',
            fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('Temperature [°C]', fontsize=13)
    ax.set_ylabel('Pressure [kPa]', fontsize=13)
    ax.set_title('Water Phase Diagram and Triple Point (Calculation Results)', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim([0.1, 200])
    ax.set_xlim([-25, 105])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('triple_point_calculation.png', dpi=150, bbox_inches='tight')
    plt.show()
    

## 9\. Calculation of Phase Transition Temperatures in Real Materials

=İ Code Example 8: Calculation of Allotropic Transformation Temperatures for Fe and Ti Copy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 9. Calculation of Phase Transition Temperatures in Real Mate
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Allotropic transformation data for pure metals
    metals = {
        'Fe (±’³)': {
            'T_trans': 912 + 273.15,  # K
            'Delta_H': 900,           # J/mol
            'phase_low': '±-Fe (BCC)',
            'phase_high': '³-Fe (FCC)',
            'color': '#f5576c'
        },
        'Fe (³’´)': {
            'T_trans': 1394 + 273.15,  # K
            'Delta_H': 840,            # J/mol
            'phase_low': '³-Fe (FCC)',
            'phase_high': '´-Fe (BCC)',
            'color': '#f093fb'
        },
        'Ti (±’²)': {
            'T_trans': 882 + 273.15,   # K
            'Delta_H': 4000,           # J/mol
            'phase_low': '±-Ti (HCP)',
            'phase_high': '²-Ti (BCC)',
            'color': '#3b82f6'
        },
        'Co (µ’±)': {
            'T_trans': 422 + 273.15,   # K
            'Delta_H': 450,            # J/mol
            'phase_low': 'µ-Co (HCP)',
            'phase_high': '±-Co (FCC)',
            'color': '#10b981'
        }
    }
    
    R = 8.314  # J/(mol·K)
    
    # Pressure dependence of phase transition temperature calculation (Clapeyron approximation)
    # Simplification: ”V estimated (typical value of 0.1 cm³/mol = 1e-7 m³/mol)
    Delta_V_typical = 1e-7  # m³/mol
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Comparison of allotropic transformation temperatures
    metal_names = []
    T_trans_list = []
    Delta_H_list = []
    colors = []
    
    for name, data in metals.items():
        metal_names.append(name)
        T_trans_list.append(data['T_trans'] - 273.15)  # °C
        Delta_H_list.append(data['Delta_H'] / 1000)    # kJ/mol
        colors.append(data['color'])
    
    x_pos = np.arange(len(metal_names))
    ax1.bar(x_pos, T_trans_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metal_names, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('Transformation Temperature [°C]', fontsize=12)
    ax1.set_title('Allotropic Transformation Temperatures of Pure Metals', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Display values on each bar
    for i, (T, DH) in enumerate(zip(T_trans_list, Delta_H_list)):
        ax1.text(i, T + 30, f'{T:.0f}°C\n”H={DH:.1f} kJ/mol',
                 ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right plot: Relationship between transformation enthalpy and temperature
    ax2.scatter(T_trans_list, Delta_H_list, s=200, c=colors, alpha=0.7,
                edgecolor='black', linewidth=2)
    for name, T, DH, color in zip(metal_names, T_trans_list, Delta_H_list, colors):
        ax2.annotate(name, (T, DH), xytext=(10, 10), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax2.set_xlabel('Transformation Temperature [°C]', fontsize=12)
    ax2.set_ylabel('Transformation Enthalpy ”H [kJ/mol]', fontsize=12)
    ax2.set_title('Relationship Between Transformation Temperature and Enthalpy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metal_allotropic_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Display detailed data
    print("=" * 80)
    print("Allotropic Transformation Data for Pure Metals")
    print("=" * 80)
    for name, data in metals.items():
        print(f"\n{name}")
        print(f"  Transformation temperature: {data['T_trans'] - 273.15:.0f}°C ({data['T_trans']:.2f} K)")
        print(f"  Transformation enthalpy: {data['Delta_H']/1000:.2f} kJ/mol")
        print(f"  Low-temperature phase: {data['phase_low']}")
        print(f"  High-temperature phase: {data['phase_high']}")
    
        # Calculation of transformation entropy
        Delta_S = data['Delta_H'] / data['T_trans']  # J/(mol·K)
        print(f"  Transformation entropy: ”S = ”H/T = {Delta_S:.2f} J/(mol·K)")
    
    print("\n" + "=" * 80)
    print("Significance of Allotropic Transformations")
    print("=" * 80)
    print("ûFe ±’³ transformation: Basis of heat treatment in steel through quenching and tempering")
    print("ûTi ±’² transformation: Improved plastic workability at high temperature (² processing)")
    print("ûCo µ’± transformation: Application as magnetic material")
    print("ûTransformation temperature can be controlled by alloying element addition (basis of phase diagram engineering)")
    
    # Simple estimation of pressure effect
    print("\n" + "=" * 80)
    print("Change in Transformation Temperature Due to Pressure (Estimation)")
    print("=" * 80)
    for name, data in metals.items():
        T_trans = data['T_trans']
        Delta_H = data['Delta_H']
        # Clapeyron: dT/dP = T ”V / ”H
        dT_dP = T_trans * Delta_V_typical / Delta_H  # K/Pa
        dT_dP_MPa = dT_dP * 1e6  # K/MPa
    
        print(f"{name}: dT/dP H {dT_dP_MPa:.3f} K/MPa")
        print(f"  ’ Change of approximately {dT_dP_MPa * 100:.1f} K at 100 MPa")
    

#### Note: Pressure Dependence of Actual Phase Transition Temperatures

The above calculation is simplified for educational purposes. In reality:

  * **Precise measurement of ”V** is required (determined by X-ray diffraction, etc.)
  * **Pressure dependence** : ”H and ”V also change with pressure
  * **High-pressure phases** : New phases appear above several GPa (e.g., Fe-µ phase)

Precise calculations use CALPHAD databases or first-principles calculations.

## Summary

#### Key Points Learned in This Chapter

  * **Definition of phase** : A region with uniform composition and properties, separated by clear interfaces
  * **Equilibrium conditions** : Temperature, pressure, and chemical potential are equal in all phases
  * **Gibbs phase rule** : $F = C - P + 2$, determines degrees of freedom in the system
  * **Unary phase diagram** : P-T diagram showing stable regions of solid, liquid, and gas phases
  * **Clapeyron equation** : Slope of phase boundary $dP/dT = \Delta H / (T \Delta V)$
  * **Clausius-Clapeyron equation** : Simplified equation for liquid-gas and solid-gas boundaries
  * **Classification of phase transitions** : First-order transition (with latent heat), second-order transition (without latent heat)
  * **Lever rule** : Calculate quantity ratio of each phase in two-phase region
  * **Application to real materials** : Phase transition temperatures and pressure dependence for water, Fe, Ti, etc.

In the next chapter, we will proceed to binary phase diagrams and learn about phase equilibria in more complex alloy systems. We will cover phase diagram interpretation including eutectic reactions, peritectic reactions, and solid solutions, which are directly relevant to practical materials design.

### =İ Exercises

#### Exercise 1: Application of Gibbs Phase Rule

**Problem** : Calculate the degrees of freedom $F$ for the following systems.

  * (a) State where only liquid phase of pure aluminum exists
  * (b) State where solid and liquid phases of pure aluminum coexist (during melting)
  * (c) Single-phase (± phase) region of Cu-Zn alloy (brass)
  * (d) Two-phase (± phase + ² phase) region of Cu-Zn alloy

**Hint** : Use $F = C - P + 2$. Under constant pressure (atmospheric), $F = C - P + 1$.

#### Exercise 2: Application of Clapeyron Equation

**Problem** : For the ±’³ transformation of pure iron, calculate the pressure dependence of transformation temperature $dT/dP$ using the following data.

  * Transformation temperature: $T_{\text{trans}} = 912°\text{C} = 1185$ K
  * Transformation enthalpy: $\Delta H = 900$ J/mol
  * Molar volume change: $\Delta V = V_\gamma - V_\alpha = 0.05 \times 10^{-6}$ m³/mol

**Hint** : Use $dP/dT = \Delta H / (T \Delta V)$ and take the reciprocal to obtain $dT/dP$.

#### Exercise 3: Lever Rule Calculation

**Problem** : Consider the eutectoid reaction in Fe-C alloy (carbon steel) at 727°C. Calculate the mass ratio of ±-Fe (ferrite) to FeƒC (cementite) under the following conditions.

  * Carbon concentration in ±-Fe: 0.02 wt% C
  * Carbon concentration in FeƒC: 6.7 wt% C
  * Overall carbon concentration in alloy: 0.4 wt% C (hypoeutectoid steel)

**Hint** : Use the lever rule $f_{\text{Fe}_3\text{C}} = (x_{\text{avg}} - x_\alpha) / (x_{\text{Fe}_3\text{C}} - x_\alpha)$.

#### Exercise 4: Vapor Pressure Curve Calculation (Hard)

**Problem** : Calculate the vapor pressure of ethanol at 20°C and 60°C using the following data.

  * Boiling point (1 atm): $T_{\text{boil}} = 78.3°\text{C}$
  * Vaporization enthalpy: $\Delta H_{\text{vap}} = 38560$ J/mol

**Hint** : Use the Clausius-Clapeyron equation $\ln(P_2 / P_1) = -(\Delta H_{\text{vap}} / R)(1/T_2 - 1/T_1)$. $P_1 = 101325$ Pa (1 atm), $T_1 = 78.3 + 273.15$ K.

## References

  1. D.R. Gaskell, D.E. Laughlin, "Introduction to the Thermodynamics of Materials", 6th Edition, CRC Press, 2017
  2. D.A. Porter, K.E. Easterling, M.Y. Sherif, "Phase Transformations in Metals and Alloys", 3rd Edition, CRC Press, 2009
  3. P. Atkins, J. de Paula, "Atkins' Physical Chemistry", 11th Edition, Oxford University Press, 2018
  4. H.L. Lukas, S.G. Fries, B. Sundman, "Computational Thermodynamics: The CALPHAD Method", Cambridge University Press, 2007
  5. J.W. Christian, "The Theory of Transformations in Metals and Alloys", 3rd Edition, Pergamon Press, 2002

[� Previous Chapter: Gibbs Energy and Chemical Potential](<chapter-2.html>) [Next Chapter: Binary Phase Diagrams and Phase Equilibria ’](<chapter-4.html>)
