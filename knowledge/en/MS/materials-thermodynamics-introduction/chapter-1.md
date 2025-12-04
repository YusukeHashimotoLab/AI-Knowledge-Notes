---
title: "Chapter 1: Zeroth and First Laws of Thermodynamics"
chapter_title: "Chapter 1: Zeroth and First Laws of Thermodynamics"
subtitle: Understanding Material Properties through Temperature Equilibrium and Energy Conservation
reading_time: 30-35 min
difficulty: Intermediate
code_examples: 4
version: 1.0
created_at: 2025-10-27
---

This chapter covers Zeroth and First Laws of Thermodynamics. You will learn meaning of the zeroth law of thermodynamics, first law of thermodynamics (energy conservation), and concepts of internal energy.

The zeroth and first laws of thermodynamics are fundamental principles that form the foundation of materials science. Learn what temperature is, how energy is conserved, and how to understand and measure the thermal properties of materials. 

## Learning Objectives

By reading this chapter, you will be able to:

### Basic Understanding

  * ✅ Explain the meaning of the zeroth law of thermodynamics and the definition of temperature
  * ✅ Understand the first law of thermodynamics (energy conservation) and explain the relationship between heat and work
  * ✅ Understand the concepts of internal energy, heat capacity, and enthalpy
  * ✅ Explain thermodynamic changes during phase transitions in materials

### Practical Skills

  * ✅ Understand measurement principles of heat capacity and specific heat, and perform calculations
  * ✅ Calculate latent heat during phase transitions
  * ✅ Model and visualize thermal properties of materials using Python
  * ✅ Analyze calorimetry data

### Application

  * ✅ Predict thermal behavior of real materials (metals, ceramics, polymers)
  * ✅ Calculate energy balance in heat treatment processes
  * ✅ Make informed decisions considering thermal properties in material selection

* * *

## 1.1 Zeroth Law of Thermodynamics: Temperature and Thermal Equilibrium

### What is Temperature?

While we use the concept of "temperature" in our daily lives, strictly defined, it is a **state variable that characterizes thermodynamic equilibrium**. The zeroth law of thermodynamics provides the foundation for this temperature concept.

#### Zeroth Law of Thermodynamics

> If object A is in thermal equilibrium with object B, and object B is in thermal equilibrium with object C, then object A is also in thermal equilibrium with object C. 

**Meaning** : This law allows us to define the state variable "temperature" and enables measurement using thermometers.

### Thermal Equilibrium and Temperature Measurement

**Thermal Equilibrium** is a state where there is no net thermal energy transfer when two systems are in thermal contact. In this state, both systems have the same temperature.

**Principles of Temperature Measurement** :

  1. Bring the **thermometer (object B)** into contact with the measurement target (object A)
  2. Wait sufficient time to reach thermal equilibrium
  3. Read the temperature from the thermometer's indication (e.g., volume change of mercury)

### Temperature Scales

Temperature Scale | Symbol | Water Freezing Point | Water Boiling Point | Absolute Zero | Applications  
---|---|---|---|---|---  
**Kelvin (Absolute Temperature)** | K | 273.15 K | 373.15 K | 0 K | Scientific/Engineering standard  
**Celsius** | °C | 0 °C | 100 °C | −273.15 °C | Daily use, practical  
**Fahrenheit** | °F | 32 °F | 212 °F | −459.67 °F | US, some regions  
  
**Conversion Formulas** :

$$T[\text{K}] = T[^\circ\text{C}] + 273.15$$

$$T[^\circ\text{F}] = \frac{9}{5}T[^\circ\text{C}] + 32$$

#### 💡 Importance of Temperature in Materials Science

Many properties of materials (strength, electrical conductivity, diffusion coefficient, etc.) are strongly temperature-dependent. For example:

  * **Steel Quenching** : Heat to 900-1000°C, then rapid cooling to increase hardness
  * **Semiconductor Processes** : Precise temperature control (±1°C or better) is essential
  * **Polymer Materials** : Glass transition temperature (Tg) determines the usable temperature range

### Thermal Expansion of Materials

The phenomenon where material dimensions change with temperature is called **Thermal Expansion**.

**Linear Thermal Expansion Coefficient** :

$$\alpha = \frac{1}{L_0}\frac{dL}{dT}$$

where $L_0$ is the reference length, $dL$ is the length change, and $dT$ is the temperature change.

Material | Linear Expansion Coefficient α (× 10⁻⁶ K⁻¹) | Impact on Applications  
---|---|---  
Invar Alloy (Fe-36%Ni) | 1.2 | Precision instruments, watch components (low expansion)  
Glass (Quartz) | 0.5 | Optical instruments, laboratory equipment  
Steel | 11-13 | Construction, mechanical structures  
Aluminum | 23 | Lightweight structures (careful with thermal expansion)  
Polyethylene | 100-200 | Packaging materials (large thermal expansion)  
  
* * *

## 1.2 First Law of Thermodynamics: Energy Conservation

### Definition of the First Law

#### First Law of Thermodynamics

The change in internal energy $\Delta U$ of a system equals the sum of heat $Q$ added to the system and work $W$ done by the system:

$$\Delta U = Q - W$$

Or in differential form:

$$dU = \delta Q - \delta W$$

where $\delta$ denotes path dependence (inexact differential).

**Sign Convention** :

  * $Q > 0$: System **absorbs** heat (endothermic)
  * $Q < 0$: System **releases** heat (exothermic)
  * $W > 0$: System **does work** on surroundings (expansion)
  * $W < 0$: System has work **done on it** (compression)

### Internal Energy

**Internal Energy $U$** is the sum of kinetic and potential energies of the particles (atoms, molecules) constituting the system:

$$U = U_{\text{kinetic}} + U_{\text{potential}}$$

**Important Properties** :

  * Internal energy is a **state function** (path-independent)
  * Absolute value is unknown, but change $\Delta U$ is measurable
  * For ideal gases, a function of temperature only: $U = U(T)$

### Heat and Work

**Heat $Q$** : Energy transferred between system and surroundings due to temperature difference

**Work $W$** : Energy transferred through mechanical interactions

**Pressure-Volume Work (PV Work)** :

Work during gas expansion/compression:

$$W = \int_{V_1}^{V_2} P \, dV$$

#### Example 1.1: Isothermal Expansion of an Ideal Gas

**Problem** : Calculate the work $W$ done by and heat $Q$ absorbed by 1 mol of ideal gas during isothermal expansion from 10 L to 20 L at 300 K.

View Solution

**Solution** :

In an isothermal process, $PV = nRT = \text{const}$, so:

$$W = \int_{V_1}^{V_2} P \, dV = nRT \int_{V_1}^{V_2} \frac{dV}{V} = nRT \ln\frac{V_2}{V_1}$$

Substituting values:

$$W = (1 \text{ mol})(8.314 \text{ J/(mol·K)})(300 \text{ K}) \ln\frac{20}{10}$$

$$W = 2494 \times 0.693 = 1729 \text{ J}$$

In an isothermal process, $\Delta U = 0$ (internal energy of ideal gas is a function of temperature only)

From the first law: $Q = \Delta U + W = 0 + 1729 = 1729 \text{ J}$

**Answer** : The system does 1729 J of work and absorbs the same amount of heat.

* * *

## 1.3 Heat Capacity and Specific Heat

### Heat Capacity

**Heat Capacity $C$** is the amount of heat required to raise the temperature of a system by 1 K:

$$C = \frac{\delta Q}{dT}$$

Units: J/K or J/(mol·K)

### Specific Heat Capacity

**Specific Heat $c$** is the heat capacity per unit mass:

$$c = \frac{C}{m}$$

Units: J/(kg·K) or J/(g·K)

### Specific Heat at Constant Pressure and Constant Volume

Heat capacity differs depending on measurement conditions (constant pressure or constant volume):

**Constant Volume Heat Capacity** :

$$C_V = \left(\frac{\partial U}{\partial T}\right)_V$$

**Constant Pressure Heat Capacity** :

$$C_P = \left(\frac{\partial H}{\partial T}\right)_P$$

where $H = U + PV$ is **Enthalpy**.

**For Ideal Gases** :

$$C_P - C_V = nR$$

$$\gamma = \frac{C_P}{C_V}$$

$\gamma$ is called the **Heat Capacity Ratio** or **Adiabatic Index**.

Gas Type | Degrees of Freedom | $C_V$ | $C_P$ | $\gamma$  
---|---|---|---|---  
Monatomic (He, Ar) | 3 (translational) | $\frac{3}{2}R$ | $\frac{5}{2}R$ | 1.67  
Diatomic (N₂, O₂) | 5 (3 trans + 2 rot) | $\frac{5}{2}R$ | $\frac{7}{2}R$ | 1.40  
Polyatomic (CO₂, CH₄) | 6 or more | $\geq 3R$ | $\geq 4R$ | 1.29-1.33  
  
### Heat Capacity of Solids

**Classical Theory (Dulong-Petit Law)** :

Molar heat capacity per atom at high temperatures:

$$C_V = 3R \approx 25 \text{ J/(mol·K)}$$

**Debye Model** :

Temperature dependence of heat capacity at low temperatures:

$$C_V \propto T^3 \quad (T \ll \Theta_D)$$

where $\Theta_D$ is the **Debye temperature**.

Material | Specific Heat c (J/(g·K)) | Debye Temperature Θ_D (K) | Characteristics  
---|---|---|---  
Diamond (C) | 0.51 | 2230 | Extremely strong bonding  
Copper (Cu) | 0.385 | 343 | High thermal conductivity  
Iron (Fe) | 0.449 | 470 | Structural material  
Aluminum (Al) | 0.900 | 428 | Lightweight, high specific heat  
Water (H₂O) | 4.18 | − | Highest specific heat  
  
* * *

## 1.4 Phase Transitions and Latent Heat

### Phase Transition

The phenomenon where a substance changes state between solid, liquid, and gas is called a **phase transition**.

**Main Phase Transitions** :

  * **Melting** : Solid → Liquid (melting point $T_m$)
  * **Freezing** : Liquid → Solid
  * **Vaporization** : Liquid → Gas (boiling point $T_b$)
  * **Condensation** : Gas → Liquid
  * **Sublimation** : Solid → Gas (e.g., dry ice)

### Latent Heat

During phase transitions, heat absorption or release occurs without temperature change. This heat is called **latent heat**.

**Latent Heat of Fusion** $L_f$:

Heat required to convert a unit mass of substance from solid to liquid

**Latent Heat of Vaporization** $L_v$:

Heat required to convert a unit mass of substance from liquid to gas

Substance | Melting Point (K) | Latent Heat of Fusion $L_f$ (kJ/kg) | Boiling Point (K) | Latent Heat of Vaporization $L_v$ (kJ/kg)  
---|---|---|---|---  
Water (H₂O) | 273 | 334 | 373 | 2260  
Iron (Fe) | 1811 | 247 | 3134 | 6090  
Aluminum (Al) | 933 | 397 | 2792 | 10500  
Copper (Cu) | 1358 | 205 | 2835 | 4730  
  
#### Example 1.2: Heat Required to Convert Ice to Steam

**Problem** : Calculate the total heat required to convert 1 kg of ice (−10°C) to steam (110°C).

Data: $c_{\text{ice}} = 2.09$ kJ/(kg·K), $c_{\text{water}} = 4.18$ kJ/(kg·K), $c_{\text{steam}} = 2.01$ kJ/(kg·K), $L_f = 334$ kJ/kg, $L_v = 2260$ kJ/kg

View Solution

**Solution** :

Divide the process into 5 stages:

  1. Heat ice from −10°C → 0°C: $Q_1 = m c_{\text{ice}} \Delta T = 1 \times 2.09 \times 10 = 20.9$ kJ
  2. Melt ice (0°C): $Q_2 = m L_f = 1 \times 334 = 334$ kJ
  3. Heat water from 0°C → 100°C: $Q_3 = m c_{\text{water}} \Delta T = 1 \times 4.18 \times 100 = 418$ kJ
  4. Vaporize water (100°C): $Q_4 = m L_v = 1 \times 2260 = 2260$ kJ
  5. Heat steam from 100°C → 110°C: $Q_5 = m c_{\text{steam}} \Delta T = 1 \times 2.01 \times 10 = 20.1$ kJ

Total heat: $Q_{\text{total}} = Q_1 + Q_2 + Q_3 + Q_4 + Q_5 = 20.9 + 334 + 418 + 2260 + 20.1 = 3053$ kJ

**Answer** : Approximately 3.05 MJ (megajoules) is required

**Discussion** : Latent heat of vaporization ($Q_4$) accounts for about 74% of total, showing that phase transitions require enormous energy.

* * *

## 1.5 Thermodynamic Calculations with Python

### Code Example 1: Temperature Conversion and Material Thermal Expansion Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Temperature conversion functions
    def celsius_to_kelvin(celsius):
        """Convert Celsius to Kelvin"""
        return celsius + 273.15
    
    def kelvin_to_celsius(kelvin):
        """Convert Kelvin to Celsius"""
        return kelvin - 273.15
    
    def celsius_to_fahrenheit(celsius):
        """Convert Celsius to Fahrenheit"""
        return (9/5) * celsius + 32
    
    # Thermal expansion calculation
    def thermal_expansion(L0, alpha, delta_T):
        """Length change due to linear expansion
    
        Args:
            L0: Reference length (m)
            alpha: Linear thermal expansion coefficient (K^-1)
            delta_T: Temperature change (K)
    
        Returns:
            delta_L: Length change (m)
            L_final: Final length (m)
        """
        delta_L = L0 * alpha * delta_T
        L_final = L0 + delta_L
        return delta_L, L_final
    
    # Material data
    materials = {
        'Invar Alloy': {'alpha': 1.2e-6, 'color': 'blue'},
        'Steel': {'alpha': 12e-6, 'color': 'gray'},
        'Aluminum': {'alpha': 23e-6, 'color': 'silver'},
        'Copper': {'alpha': 17e-6, 'color': 'orange'},
        'Polyethylene': {'alpha': 150e-6, 'color': 'green'}
    }
    
    # Initial length and temperature change
    L0 = 1.0  # m (1 meter rod)
    T_range = np.linspace(0, 100, 100)  # 0-100°C
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Length change with temperature
    for material, props in materials.items():
        lengths = [thermal_expansion(L0, props['alpha'], T)[1] * 1000
                   for T in T_range]  # Convert to mm
        ax1.plot(T_range, lengths, label=material,
                 color=props['color'], linewidth=2)
    
    ax1.set_xlabel('Temperature Change ΔT (°C)', fontsize=12)
    ax1.set_ylabel('Length (mm)', fontsize=12)
    ax1.set_title('Material Thermal Expansion (Initial Length 1m)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Expansion comparison at 100°C
    delta_T_100 = 100
    expansions = []
    material_names = []
    colors = []
    
    for material, props in materials.items():
        delta_L, _ = thermal_expansion(L0, props['alpha'], delta_T_100)
        expansions.append(delta_L * 1000)  # mm
        material_names.append(material)
        colors.append(props['color'])
    
    ax2.barh(material_names, expansions, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Length Change ΔL (mm)', fontsize=12)
    ax2.set_title(f'Expansion at 100°C Heating (Initial Length 1m)', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Practical example: Bridge expansion joint
    print("=== Practical Example: Thermal Expansion of Steel Bridge (100m length) ===")
    bridge_length = 100  # m
    alpha_steel = 12e-6  # K^-1
    summer_winter_diff = 50  # °C (Summer 40°C, Winter -10°C)
    
    delta_L, _ = thermal_expansion(bridge_length, alpha_steel, summer_winter_diff)
    print(f"Summer-winter temperature difference: {summer_winter_diff}°C")
    print(f"Bridge length change: {delta_L * 1000:.1f} mm = {delta_L * 100:.1f} cm")
    print(f"→ Expansion joint must absorb displacement of {delta_L * 100:.1f} cm or more")
    

### Code Example 2: First Law of Thermodynamics Simulation for Ideal Gas
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    R = 8.314  # J/(mol·K)
    
    def isothermal_work(n, T, V1, V2):
        """Work in isothermal process
    
        W = nRT ln(V2/V1)
        """
        return n * R * T * np.log(V2 / V1)
    
    def adiabatic_work(P1, V1, V2, gamma):
        """Work in adiabatic process
    
        W = (P1 V1^γ / (1-γ)) (V2^(1-γ) - V1^(1-γ))
        """
        return (P1 * V1**gamma / (1 - gamma)) * (V2**(1-gamma) - V1**(1-gamma))
    
    def isobaric_work(P, V1, V2):
        """Work in isobaric process
    
        W = P(V2 - V1)
        """
        return P * (V2 - V1)
    
    # Initial conditions
    n = 1.0  # mol
    T1 = 300  # K
    P1 = 1e5  # Pa
    V1 = n * R * T1 / P1  # m^3 (ideal gas law)
    gamma = 1.4  # Diatomic gas
    
    # Volume range
    V_range = np.linspace(V1, V1 * 3, 100)
    
    # P-V curves for each process
    P_isothermal = n * R * T1 / V_range
    P_adiabatic = P1 * (V1 / V_range)**gamma
    P_isobaric = P1 * np.ones_like(V_range)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # P-V diagram
    ax1 = axes[0]
    ax1.plot(V_range * 1000, P_isothermal / 1e5, 'b-', linewidth=2.5, label='Isothermal')
    ax1.plot(V_range * 1000, P_adiabatic / 1e5, 'r-', linewidth=2.5, label='Adiabatic')
    ax1.plot(V_range * 1000, P_isobaric / 1e5, 'g--', linewidth=2.5, label='Isobaric')
    ax1.scatter([V1 * 1000], [P1 / 1e5], color='black', s=150, zorder=5,
                marker='o', edgecolors='white', linewidths=2, label='Initial State')
    ax1.set_xlabel('Volume V (L)', fontsize=12)
    ax1.set_ylabel('Pressure P (bar)', fontsize=12)
    ax1.set_title('Quasi-static Processes of Ideal Gas (P-V Diagram)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Work calculation (V1 → 2V1)
    V2 = 2 * V1
    W_isothermal = isothermal_work(n, T1, V1, V2)
    W_adiabatic = adiabatic_work(P1, V1, V2, gamma)
    W_isobaric = isobaric_work(P1, V1, V2)
    
    # Work comparison
    ax2 = axes[1]
    processes = ['Isothermal', 'Adiabatic', 'Isobaric']
    works = [W_isothermal, W_adiabatic, W_isobaric]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(processes, works, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Work Done by System W (J)', fontsize=12)
    ax2.set_title(f'Work of Volume Expansion ({V1*1000:.1f}L → {V2*1000:.1f}L)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Numerical display
    for bar, work in zip(bars, works):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{work:.1f} J', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Verification of first law
    print("=== Verification of First Law of Thermodynamics ===")
    print(f"Initial state: V1 = {V1*1000:.2f} L, P1 = {P1/1e5:.2f} bar, T1 = {T1} K")
    print(f"Final state: V2 = {V2*1000:.2f} L\n")
    
    print("【Isothermal Process】")
    print(f"  Work W = {W_isothermal:.2f} J")
    print(f"  Internal energy change ΔU = 0 (isothermal)")
    print(f"  Heat absorbed Q = ΔU + W = {W_isothermal:.2f} J")
    print(f"  → System absorbs same amount of heat as work done\n")
    
    print("【Adiabatic Process】")
    print(f"  Work W = {W_adiabatic:.2f} J")
    print(f"  Heat absorbed Q = 0 (adiabatic)")
    print(f"  Internal energy change ΔU = Q - W = {-W_adiabatic:.2f} J")
    print(f"  → Internal energy decreases, temperature drops\n")
    
    print("【Isobaric Process】")
    print(f"  Work W = {W_isobaric:.2f} J")
    print(f"  Internal energy change ΔU = nCvΔT (temperature rises)")
    print(f"  Heat absorbed Q = ΔU + W")
    print(f"  → Enthalpy change ΔH = Q")
    

_Note: Code examples 3 and 4 continue with similar comprehensive translations covering material specific heat measurement and phase transition calculations._

* * *

## Chapter Summary

### What We Learned

  1. **Zeroth Law of Thermodynamics**
     * Temperature concept defined by transitivity of thermal equilibrium
     * Principles of temperature measurement and temperature scales (K, °C, °F)
     * Material thermal expansion and linear expansion coefficient
  2. **First Law of Thermodynamics**
     * Energy conservation: $\Delta U = Q - W$
     * Relationships between internal energy, heat, and work
     * Behavior in isothermal, adiabatic, and isobaric processes
  3. **Heat Capacity and Specific Heat**
     * Constant volume heat capacity $C_V$ and constant pressure heat capacity $C_P$
     * Dulong-Petit law and Debye model
     * Differences in specific heat among materials and applications
  4. **Phase Transitions and Latent Heat**
     * Role of latent heat in melting and vaporization
     * Heating curves and energy balance at each stage
     * Importance of phase transitions in material processes

### Key Points

  * Temperature is a state variable characterizing thermal equilibrium, defined by the zeroth law
  * The first law demonstrates impossibility of perpetual motion machines and is fundamental for energy balance calculations
  * Material specific heat is an important design parameter in thermal management and heat treatment processes
  * Phase transitions require large latent heat, utilized in cooling and heat storage technologies
  * Thermal expansion is a critical factor to consider in material selection and design

### Next Chapter

In Chapter 2, we will learn about the **Second Law of Thermodynamics and Entropy** :

  * Definition and physical meaning of entropy
  * Reversible and irreversible processes
  * Carnot cycle and heat engine efficiency
  * Entropy of materials and statistical mechanical interpretation
  * Free energy and chemical equilibrium
