---
title: "Chapter 1: Fundamentals of Supercritical Fluids"
chapter_title: "Chapter 1: Fundamentals of Supercritical Fluids"
subtitle: "Understanding Critical Points, Phase Diagrams, and Unique Properties"
---

🌐 EN | [🇯🇵 JP](../../../jp/MS/supercritical-fluids-introduction/chapter-1.md) | Last sync: 2025-12-25

[Materials Science Dojo](../index.html) > [Introduction to Supercritical Fluids](index.md) > Chapter 1

---

# Chapter 1: Fundamentals of Supercritical Fluids

**Understanding Critical Points, Phase Diagrams, and Unique Properties**

📚 Beginner Level | ⏱️ Approximately 60 minutes | 🎯 Critical Point・Phase Diagrams・Properties

## Learning Objectives

### Basic Understanding
- Define what a supercritical fluid is and identify the critical point on a phase diagram
- Explain the physical significance of critical temperature and critical pressure
- Describe how supercritical fluids differ from ordinary gases and liquids
- Understand why there is no distinct phase boundary above the critical point

### Practical Skills
- Read and interpret P-T phase diagrams for common substances
- Calculate reduced properties (reduced temperature, pressure, and density)
- Use critical constants to estimate supercritical fluid behavior
- Compare property values across gas, liquid, and supercritical fluid states

### Application
- Recognize situations where supercritical fluids offer advantages over conventional solvents
- Understand how pressure and temperature control can tune SCF properties
- Identify suitable supercritical fluids for specific applications based on critical constants

## 1.1 What is a Supercritical Fluid?

A **supercritical fluid (SCF)** is a state of matter that exists when a substance is heated and pressurized above its **critical temperature** (\\(T_c\\)) and **critical pressure** (\\(P_c\\)). In this state, the distinction between liquid and gas phases disappears, and the substance exhibits unique properties that combine characteristics of both liquids and gases.

**Formal Definition**

A substance is in the supercritical fluid state when:

\\[
T > T_c \quad \text{and} \quad P > P_c
\\]

where \\(T_c\\) is the critical temperature and \\(P_c\\) is the critical pressure.

### 1.1.1 The Critical Point on a Phase Diagram

The **critical point** is the terminus of the liquid-vapor coexistence curve. On a pressure-temperature (P-T) phase diagram, it represents the highest temperature and pressure at which distinct liquid and gas phases can coexist.

**Key Features of the Critical Point**

- **Coordinates**: (\\(T_c\\), \\(P_c\\), \\(\rho_c\\)) where \\(\rho_c\\) is the critical density
- **Phase boundary terminus**: The liquid-gas phase boundary ends at this point
- **Continuous transition**: Above \\(T_c\\), increasing pressure produces a continuous density change without a phase transition
- **No surface tension**: The liquid-gas interface vanishes, so surface tension becomes zero
- **Critical opalescence**: Near the critical point, fluids scatter light strongly due to large density fluctuations

### 1.1.2 Historical Discovery

The existence of the critical point was first observed by **Charles Cagniard de la Tour** in 1822, who noticed that liquids heated in sealed containers eventually became indistinguishable from their vapor. However, the theoretical understanding came later:

- **1869**: Thomas Andrews conducted systematic studies on CO₂, defining the critical temperature
- **1873**: Johannes van der Waals developed an equation of state that predicted the critical point
- **Late 1800s**: Recognition that supercritical fluids had unique solvent properties
- **1970s-present**: Industrial development of SCF technology for extraction, reaction media, and materials processing

Today, supercritical fluids are essential in industries ranging from coffee decaffeination to pharmaceutical manufacturing and carbon capture.

## 1.2 The Critical Point

### 1.2.1 Critical Constants

Every pure substance has characteristic critical constants that define the critical point:

**Critical Constants**

- **Critical Temperature (\\(T_c\\))**: The temperature above which gas cannot be liquefied regardless of pressure
- **Critical Pressure (\\(P_c\\))**: The minimum pressure required to liquefy a gas at \\(T_c\\)
- **Critical Density (\\(\rho_c\\))**: The density of the substance at the critical point
- **Critical Volume (\\(V_c\\))**: The molar volume at the critical point (\\(V_c = M/\rho_c\\) where \\(M\\) is molar mass)

### 1.2.2 Critical Constants for Common Substances

The following table lists critical constants for substances commonly used as supercritical fluids:

| Substance | Formula | \\(T_c\\) (°C) | \\(T_c\\) (K) | \\(P_c\\) (MPa) | \\(P_c\\) (bar) | \\(\rho_c\\) (g/cm³) |
|-----------|---------|---------|---------|----------|----------|-------------|
| **Carbon Dioxide** | CO₂ | 31.0 | 304.1 | 7.38 | 73.8 | 0.468 |
| **Water** | H₂O | 374.0 | 647.1 | 22.06 | 220.6 | 0.322 |
| **Nitrogen** | N₂ | -147.0 | 126.2 | 3.40 | 34.0 | 0.311 |
| **Ethanol** | C₂H₅OH | 241.0 | 514.0 | 6.14 | 61.4 | 0.276 |
| **Propane** | C₃H₈ | 96.7 | 369.8 | 4.25 | 42.5 | 0.217 |
| **Ammonia** | NH₃ | 132.3 | 405.5 | 11.35 | 113.5 | 0.235 |
| **Methanol** | CH₃OH | 239.5 | 512.6 | 8.09 | 80.9 | 0.272 |
| **Xenon** | Xe | 16.6 | 289.7 | 5.84 | 58.4 | 1.105 |

**Why CO₂ is Popular**

Supercritical CO₂ (sc-CO₂) is the most widely used supercritical fluid because:

- **Mild critical conditions**: \\(T_c = 31°C\\) is just above room temperature, requiring minimal heating
- **Moderate pressure**: \\(P_c = 73.8\\) bar is achievable with standard high-pressure equipment
- **Safety**: Non-toxic, non-flammable, and environmentally benign
- **Availability**: Abundant and inexpensive
- **Tunability**: Properties can be easily adjusted with pressure and temperature
- **GRAS status**: Generally Recognized As Safe by FDA for food applications

### 1.2.3 Molecular Interpretation of the Critical Point

At the molecular level, the critical point represents a unique balance of intermolecular forces and thermal energy:

**Below the Critical Temperature (\\(T < T_c\\))**

- Intermolecular attractive forces can overcome thermal motion
- Distinct liquid phase (high density, strong interactions) and gas phase (low density, weak interactions) exist
- Phase transition occurs with latent heat and discontinuous density change

**At the Critical Temperature (\\(T = T_c\\))**

- Thermal energy and intermolecular forces are exactly balanced
- Density fluctuations become very large (critical fluctuations)
- Surface tension approaches zero
- Compressibility diverges (isothermal compressibility \\(\kappa_T \to \infty\\))

**Above the Critical Temperature (\\(T > T_c\\))**

- Thermal energy dominates, preventing condensation
- Only a single fluid phase exists regardless of pressure
- Density changes continuously with pressure (no phase boundary)
- Properties can be tuned from gas-like to liquid-like by adjusting pressure

### 1.2.4 Critical Exponents and Universality

Near the critical point, many physical properties exhibit power-law behavior characterized by **critical exponents**. For example:

\\[
C_p \sim |T - T_c|^{-\alpha}, \quad \kappa_T \sim |T - T_c|^{-\gamma}, \quad \rho_L - \rho_G \sim |T - T_c|^{\beta}
\\]

where \\(\alpha\\), \\(\gamma\\), and \\(\beta\\) are critical exponents. Remarkably, these exponents are **universal** - they have the same values for all substances in the same universality class, independent of molecular details. This is a profound result from statistical mechanics and the theory of phase transitions.

## 1.3 Unique Properties of Supercritical Fluids

Supercritical fluids possess a remarkable combination of properties that make them useful for industrial and scientific applications.

### 1.3.1 Density: Liquid-like and Tunable

**Density Range**

\\[
\rho_{\text{gas}} \sim 0.001 \text{ g/cm}^3 \quad < \quad \rho_{\text{SCF}} \sim 0.1-0.9 \text{ g/cm}^3 \quad < \quad \rho_{\text{liquid}} \sim 1 \text{ g/cm}^3
\\]

Supercritical fluids have densities intermediate between gases and liquids, typically 0.2-0.9 g/cm³ depending on pressure and temperature. This is close to liquid densities, which gives SCFs strong solvating power.

**Tunability**

The density of a supercritical fluid is highly sensitive to pressure and temperature changes, especially near the critical point. This allows "dialing in" desired properties:

- **Increasing pressure** → higher density → stronger solvation
- **Increasing temperature** → lower density (at constant P) → weaker solvation
- **Small changes** in P or T near \\(T_c\\) can cause large density changes

### 1.3.2 Viscosity: Gas-like and Low

**Viscosity Range**

\\[
\eta_{\text{SCF}} \sim 10^{-4} \text{ Pa·s} \quad \approx \quad \eta_{\text{gas}} \sim 10^{-5} \text{ Pa·s} \quad \ll \quad \eta_{\text{liquid}} \sim 10^{-3} \text{ Pa·s}
\\]

Supercritical fluids have viscosities 10-100 times lower than liquids, comparable to gases. This results in:

- **Low flow resistance**: SCFs flow easily through porous materials
- **Rapid mass transfer**: Fast transport of solutes through the fluid
- **Efficient extraction**: Quick penetration into solid matrices

### 1.3.3 Diffusivity: Between Liquid and Gas

**Diffusion Coefficient Range**

\\[
D_{\text{gas}} \sim 10^{-5} \text{ m}^2/\text{s} \quad > \quad D_{\text{SCF}} \sim 10^{-7} \text{ m}^2/\text{s} \quad > \quad D_{\text{liquid}} \sim 10^{-9} \text{ m}^2/\text{s}
\\]

Diffusion in supercritical fluids is 10-100 times faster than in liquids, though slower than in gases. This enables:

- **Fast equilibration**: Rapid approach to equilibrium in extraction processes
- **Efficient chromatography**: High-speed supercritical fluid chromatography (SFC)
- **Enhanced reaction kinetics**: Faster diffusion-limited reactions

### 1.3.4 Surface Tension: Zero

A defining characteristic of supercritical fluids is the **absence of a liquid-gas interface**, which means:

\\[
\gamma = 0 \quad \text{(surface tension)}
\\]

**Consequences**

- **Complete wetting**: SCFs penetrate into pores and crevices that liquids cannot reach
- **No capillary effects**: No meniscus formation in porous materials
- **Uniform processing**: Ideal for aerogel synthesis and polymer impregnation
- **No bubble formation**: Enables supercritical drying without damaging delicate structures

### 1.3.5 Property Comparison Table

| Property | Gas (at 1 bar, 25°C) | Supercritical Fluid (near \\(T_c\\), \\(P_c\\)) | Liquid (at 1 bar, 25°C) |
|----------|---------|-----------------|---------|
| **Density** (g/cm³) | 0.0006-0.002 | 0.2-0.9 | 0.6-1.6 |
| **Viscosity** (Pa·s) | 10⁻⁵ | 10⁻⁵-10⁻⁴ | 10⁻⁴-10⁻³ |
| **Diffusivity** (m²/s) | 10⁻⁵ | 10⁻⁸-10⁻⁷ | 10⁻⁹-10⁻¹⁰ |
| **Surface Tension** (N/m) | 0 | 0 | 0.01-0.08 |
| **Solvating Power** | Very Low | Medium-High (Tunable) | High |
| **Compressibility** | High | High (esp. near \\(T_c\\)) | Low |

## 1.4 Solvent Properties of SCFs

### 1.4.1 Solvating Power and Dielectric Constant

The solvating power of a supercritical fluid depends on its density and molecular properties. Two key parameters describe solvent strength:

**Dielectric Constant (\\(\epsilon\\))**

The dielectric constant measures the ability to stabilize charged species and polar molecules. For supercritical CO₂:

- At \\(P_c\\), \\(T_c\\): \\(\epsilon \approx 1.5\\) (low polarity)
- At high density (200 bar, 40°C): \\(\epsilon \approx 1.6\\)
- For comparison, liquid hexane: \\(\epsilon \approx 2.0\\); water: \\(\epsilon \approx 80\\)

Supercritical CO₂ is a **non-polar solvent**, suitable for dissolving non-polar and weakly polar compounds.

**Hildebrand Solubility Parameter (\\(\delta\\))**

The Hildebrand parameter characterizes solvent strength based on cohesive energy density:

\\[
\delta = \sqrt{\frac{\Delta H_{\text{vap}} - RT}{V_m}}
\\]

For supercritical CO₂, \\(\delta\\) ranges from 5-15 MPa^{1/2} depending on density (liquid hexane \\(\approx 14.9\\); liquid acetone \\(\approx 20\\)). Higher density → higher \\(\delta\\) → stronger solvation.

### 1.4.2 Density-Dependent Solubility

Solubility in supercritical fluids is **highly density-dependent**. As density increases (by raising pressure or lowering temperature), solvating power increases dramatically.

**Empirical Solubility Relationship**

\\[
\ln y_2 = a + b\rho + c T
\\]

where \\(y_2\\) is the solute mole fraction, \\(\rho\\) is SCF density, and \\(a\\), \\(b\\), \\(c\\) are substance-specific constants. This shows that solubility increases exponentially with density.

**Pressure Effect Near the Critical Point**

Just above \\(T_c\\), a small pressure increase can increase density (and solubility) by orders of magnitude. This is the basis for pressure-tuned extraction and fractionation.

### 1.4.3 Example: Caffeine Extraction with Supercritical CO₂

One of the most successful commercial applications of supercritical fluids is **decaffeination of coffee** using sc-CO₂:

**Process Conditions**

- Temperature: 40-80°C (above \\(T_c = 31°C\\))
- Pressure: 150-300 bar (above \\(P_c = 73.8\\) bar)
- CO₂ density: ~0.6-0.8 g/cm³

**Why It Works**

1. **Selective solubility**: Caffeine is soluble in sc-CO₂, while flavor compounds (larger, more polar) remain in the coffee beans
2. **No residue**: CO₂ evaporates completely upon depressurization, leaving no solvent residue
3. **Recyclable**: CO₂ can be recycled by depressurization (caffeine precipitates) and recompression
4. **Safe and green**: No organic solvents (historically dichloromethane or ethyl acetate were used)

This process is now standard in the coffee industry, producing "naturally decaffeinated" coffee.

## 1.5 Python Code Examples

### Code Example 1: P-T Phase Diagram with Critical Point

```python
import numpy as np
import matplotlib.pyplot as plt

# Critical constants for CO2
Tc = 304.1  # K
Pc = 7.38   # MPa

# Generate phase boundary using Clausius-Clapeyron approximation
# (simplified for visualization; real data would be more accurate)
T_range = np.linspace(220, Tc, 100)
# Vapor pressure curve (simplified)
P_vap = Pc * np.exp(-(Tc/T_range - 1) * 5)

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot vapor pressure curve
ax.plot(T_range, P_vap, 'b-', linewidth=2, label='Liquid-Gas Boundary')

# Mark critical point
ax.plot(Tc, Pc, 'ro', markersize=12, label=f'Critical Point ({Tc} K, {Pc} MPa)')

# Shade regions
ax.fill_between(T_range, 0, P_vap, alpha=0.3, color='lightblue', label='Gas Region')
ax.fill_between(T_range, P_vap, 15, alpha=0.3, color='lightcoral', label='Liquid Region')
ax.fill_between([Tc, 400], 0, 15, alpha=0.3, color='lightyellow', label='Supercritical Region')

# Add annotations
ax.annotate('Supercritical\nFluid', xy=(350, 10), fontsize=14, ha='center', weight='bold')
ax.annotate('Liquid', xy=(260, 12), fontsize=12, ha='center')
ax.annotate('Gas', xy=(260, 2), fontsize=12, ha='center')

# Arrow pointing to critical point
ax.annotate('', xy=(Tc, Pc), xytext=(340, 5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.text(345, 4.5, 'Critical Point', fontsize=11, color='red')

# Formatting
ax.set_xlabel('Temperature (K)', fontsize=13)
ax.set_ylabel('Pressure (MPa)', fontsize=13)
ax.set_title('P-T Phase Diagram for CO₂', fontsize=15, weight='bold')
ax.set_xlim(220, 400)
ax.set_ylim(0, 15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

print("=== Critical Point Data for CO₂ ===")
print(f"Critical Temperature: {Tc} K ({Tc - 273.15:.1f}°C)")
print(f"Critical Pressure: {Pc} MPa ({Pc * 10:.1f} bar)")
print(f"Critical Density: 0.468 g/cm³")
```

**Learning Points**

- The liquid-gas coexistence curve terminates at the critical point
- Above \\(T_c\\) and \\(P_c\\), only supercritical fluid exists
- The phase diagram clearly shows the region where SCF properties can be tuned

### Code Example 2: Reduced Properties

Reduced properties normalize thermodynamic variables by critical constants, revealing universal behavior:

\\[
T_r = \frac{T}{T_c}, \quad P_r = \frac{P}{P_c}, \quad \rho_r = \frac{\rho}{\rho_c}
\\]

```python
import numpy as np
import matplotlib.pyplot as plt

# Critical constants for various substances
substances = {
    'CO₂': {'Tc': 304.1, 'Pc': 7.38, 'rhoc': 0.468},
    'H₂O': {'Tc': 647.1, 'Pc': 22.06, 'rhoc': 0.322},
    'N₂': {'Tc': 126.2, 'Pc': 3.40, 'rhoc': 0.311},
    'Ethanol': {'Tc': 514.0, 'Pc': 6.14, 'rhoc': 0.276}
}

def calculate_reduced_props(T, P, rho, Tc, Pc, rhoc):
    """Calculate reduced temperature, pressure, and density."""
    Tr = T / Tc
    Pr = P / Pc
    rhor = rho / rhoc
    return Tr, Pr, rhor

# Example: Typical supercritical conditions for each substance
print("=== Reduced Properties at Typical SCF Conditions ===\n")

for name, constants in substances.items():
    # Typical SCF: 1.1*Tc, 1.5*Pc, 0.8*rhoc
    T = 1.1 * constants['Tc']
    P = 1.5 * constants['Pc']
    rho = 0.8 * constants['rhoc']

    Tr, Pr, rhor = calculate_reduced_props(T, P, rho,
                                           constants['Tc'],
                                           constants['Pc'],
                                           constants['rhoc'])

    print(f"{name}:")
    print(f"  T = {T:.1f} K  →  Tr = {Tr:.2f}")
    print(f"  P = {P:.2f} MPa  →  Pr = {Pr:.2f}")
    print(f"  ρ = {rho:.3f} g/cm³  →  ρr = {rhor:.2f}")
    print()

# Visualize reduced property space
fig, ax = plt.subplots(figsize=(10, 7))

# Define grid for reduced properties
Tr_range = np.linspace(0.8, 1.5, 50)
Pr_range = np.linspace(0.5, 3.0, 50)
Tr_grid, Pr_grid = np.meshgrid(Tr_range, Pr_range)

# Shade supercritical region (Tr > 1, Pr > 1)
ax.fill_between([1, 1.5], 1, 3, alpha=0.3, color='gold', label='Supercritical Region')

# Plot critical point
ax.plot(1, 1, 'ro', markersize=15, label='Critical Point (Tr=1, Pr=1)')

# Plot example operating points
colors = ['blue', 'green', 'purple', 'orange']
for (name, constants), color in zip(substances.items(), colors):
    Tr = 1.1
    Pr = 1.5
    ax.plot(Tr, Pr, 'o', markersize=10, color=color, label=f'{name} (typical SCF)')

# Formatting
ax.set_xlabel('Reduced Temperature (Tr = T/Tc)', fontsize=13)
ax.set_ylabel('Reduced Pressure (Pr = P/Pc)', fontsize=13)
ax.set_title('Reduced Property Space (Universal Representation)', fontsize=15, weight='bold')
ax.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlim(0.8, 1.5)
ax.set_ylim(0.5, 3.0)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
```

**Learning Points**

- Reduced properties normalize different substances to a universal scale
- All substances have \\(T_r = P_r = 1\\) at their critical point
- Corresponding states principle: substances at same \\((T_r, P_r)\\) have similar behavior

### Code Example 3: Density vs Pressure at Different Temperatures

This code demonstrates the extreme density sensitivity near the critical temperature.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified equation of state for CO2 (van der Waals approximation)
# In practice, use NIST data or Peng-Robinson equation

def co2_density_simple(P, T):
    """
    Simplified density calculation for CO2.
    P in MPa, T in K, returns density in g/cm³

    This is a simplified model for demonstration.
    Real applications should use NIST REFPROP or similar.
    """
    Tc = 304.1  # K
    Pc = 7.38   # MPa
    rhoc = 0.468  # g/cm³

    # Reduced properties
    Tr = T / Tc
    Pr = P / Pc

    # Simplified density correlation (empirical fit near critical point)
    if Tr < 1.0:
        # Below Tc: not valid for SCF
        return None
    else:
        # Above Tc: simplified correlation
        rho = rhoc * (0.1 + 0.9 * Pr / (1 + (Tr - 1) * 2))
        return min(rho, 1.0)  # Cap at reasonable value

# Temperature series
temperatures = [310, 320, 340, 380]  # K (all above Tc = 304.1 K)
pressures = np.linspace(7.5, 30, 100)  # MPa

# Create plot
fig, ax = plt.subplots(figsize=(11, 7))

for T in temperatures:
    densities = [co2_density_simple(P, T) for P in pressures]
    densities = [d if d is not None else np.nan for d in densities]

    label = f'T = {T} K (Tr = {T/304.1:.2f})'
    ax.plot(pressures, densities, linewidth=2, label=label)

# Mark critical point
ax.plot(7.38, 0.468, 'ro', markersize=12, label='Critical Point', zorder=10)

# Formatting
ax.set_xlabel('Pressure (MPa)', fontsize=13)
ax.set_ylabel('Density (g/cm³)', fontsize=13)
ax.set_title('CO₂ Density vs Pressure at Different Temperatures', fontsize=15, weight='bold')
ax.axvline(x=7.38, color='gray', linestyle='--', alpha=0.5, label='Pc')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlim(7, 30)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

print("=== Key Observations ===")
print("1. Near Tc (310 K), density changes rapidly with small pressure changes")
print("2. Far from Tc (380 K), density increases more gradually")
print("3. High pressure can achieve liquid-like densities even at high temperature")
print("4. This tunability is the key advantage of supercritical fluids")
```

**Learning Points**

- Density is highly sensitive to pressure near \\(T_c\\)
- At higher temperatures, larger pressure changes are needed to increase density
- Compressibility \\(\kappa_T = -\frac{1}{V}(\frac{\partial V}{\partial P})_T\\) is very large near \\(T_c\\)

### Code Example 4: Property Comparison Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Property data
states = ['Gas\n(1 bar, 25°C)', 'Supercritical\n(near Tc, Pc)', 'Liquid\n(1 bar, 25°C)']

# Density (g/cm³)
density = [0.001, 0.5, 1.0]

# Viscosity (relative scale: gas = 1)
viscosity_rel = [1, 5, 100]

# Diffusivity (relative scale: gas = 1)
diffusivity_rel = [1, 0.01, 0.0001]

# Surface tension (relative: liquid = 1)
surface_tension_rel = [0, 0, 1]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Density
axes[0, 0].bar(states, density, color=['lightblue', 'gold', 'lightcoral'])
axes[0, 0].set_ylabel('Density (g/cm³)', fontsize=12)
axes[0, 0].set_title('Density Comparison', fontsize=13, weight='bold')
axes[0, 0].set_ylim(0, 1.2)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Viscosity
axes[0, 1].bar(states, viscosity_rel, color=['lightblue', 'gold', 'lightcoral'])
axes[0, 1].set_ylabel('Relative Viscosity (gas = 1)', fontsize=12)
axes[0, 1].set_title('Viscosity Comparison', fontsize=13, weight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Diffusivity
axes[1, 0].bar(states, diffusivity_rel, color=['lightblue', 'gold', 'lightcoral'])
axes[1, 0].set_ylabel('Relative Diffusivity (gas = 1)', fontsize=12)
axes[1, 0].set_title('Diffusivity Comparison', fontsize=13, weight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Surface Tension
axes[1, 1].bar(states, surface_tension_rel, color=['lightblue', 'gold', 'lightcoral'])
axes[1, 1].set_ylabel('Relative Surface Tension (liquid = 1)', fontsize=12)
axes[1, 1].set_title('Surface Tension Comparison', fontsize=13, weight='bold')
axes[1, 1].set_ylim(0, 1.2)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Property Comparison: Gas vs Supercritical Fluid vs Liquid',
             fontsize=16, weight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Print summary table
print("=== Property Comparison Summary ===\n")
print(f"{'Property':<25} {'Gas':<15} {'SCF':<15} {'Liquid':<15}")
print("-" * 70)
print(f"{'Density (g/cm³)':<25} {density[0]:<15.3f} {density[1]:<15.2f} {density[2]:<15.2f}")
print(f"{'Viscosity (10⁻⁵ Pa·s)':<25} {'1':<15} {'5-10':<15} {'100-1000':<15}")
print(f"{'Diffusivity (10⁻⁷ m²/s)':<25} {'100':<15} {'1':<15} {'0.001-0.01':<15}")
print(f"{'Surface Tension (N/m)':<25} {'0':<15} {'0':<15} {'0.01-0.08':<15}")
print("\nKey Insight: SCF combines liquid-like density with gas-like transport properties")
```

**Learning Points**

- SCF has intermediate density (closer to liquid)
- SCF has gas-like viscosity and diffusivity
- Zero surface tension enables unique penetration and wetting
- This combination is ideal for extraction and reaction applications

### Code Example 5: Hildebrand Solubility Parameter Estimation

```python
import numpy as np
import matplotlib.pyplot as plt

def hildebrand_parameter(density, Tc, Pc):
    """
    Estimate Hildebrand solubility parameter for supercritical CO2.

    Parameters:
    - density: in g/cm³
    - Tc: critical temperature in K
    - Pc: critical pressure in MPa

    Returns:
    - delta: solubility parameter in MPa^0.5
    """
    # Empirical correlation for CO2
    # delta ≈ constant * density^0.5 * (Tc/T)^0.5
    # Simplified model for demonstration
    delta = 15.0 * (density / 0.468)  # Scale by critical density
    return delta

# Density range for supercritical CO2
densities = np.linspace(0.1, 0.9, 100)  # g/cm³

# Calculate solubility parameter
deltas = [hildebrand_parameter(rho, 304.1, 7.38) for rho in densities]

# Reference values for common solvents
reference_solvents = {
    'n-Hexane': 14.9,
    'Toluene': 18.2,
    'Chloroform': 19.0,
    'Acetone': 20.0,
    'Ethanol': 26.5,
    'Water': 47.8
}

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot SCF range
ax.plot(densities, deltas, linewidth=3, color='gold',
        label='Supercritical CO₂ (tunable)')

# Add reference lines
for solvent, delta in reference_solvents.items():
    ax.axhline(y=delta, linestyle='--', alpha=0.5)
    ax.text(0.95, delta + 0.5, solvent, fontsize=10, ha='right')

# Formatting
ax.set_xlabel('Density (g/cm³)', fontsize=13)
ax.set_ylabel('Hildebrand Solubility Parameter δ (MPa^0.5)', fontsize=13)
ax.set_title('Tunability of sc-CO₂ Solvent Strength', fontsize=15, weight='bold')
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

# Shade regions
ax.fill_between([0.1, 0.9], 0, 20, alpha=0.1, color='blue', label='Non-polar region')
ax.fill_between([0.1, 0.9], 20, 50, alpha=0.1, color='red', label='Polar region')

plt.tight_layout()
plt.show()

print("=== Solubility Parameter Tuning ===")
print("\nBy adjusting pressure (and thus density), sc-CO₂ can match:")
print("  - Low density (0.2 g/cm³): δ ≈ 6 MPa^0.5 → dissolves very non-polar compounds")
print("  - Medium density (0.5 g/cm³): δ ≈ 16 MPa^0.5 → matches hexane/toluene")
print("  - High density (0.8 g/cm³): δ ≈ 26 MPa^0.5 → matches ethanol")
print("\nNote: CO₂ remains non-polar; co-solvents (e.g., ethanol) are used for polar solutes")
```

**Learning Points**

- Solubility parameter increases with density
- sc-CO₂ can match non-polar to moderately polar solvents
- For highly polar solutes, polar co-solvents (modifiers) are added
- "Like dissolves like" principle: match \\(\delta\\) values for best solubility

### Code Example 6: Interactive Phase Region Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Critical constants for CO2
Tc = 304.1  # K
Pc = 7.38   # MPa

# Create figure with slider
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)

# Initial temperature
T_init = 320  # K

# Pressure range
P_range = np.linspace(0, 15, 300)

def classify_state(P, T):
    """Classify the state of CO2 given P and T."""
    if T < Tc and P < Pc:
        # Could be gas or liquid depending on vapor pressure
        # Simplified: assume gas below Pc at T < Tc
        return 'Gas'
    elif T < Tc and P >= Pc:
        return 'Liquid'
    elif T >= Tc and P < Pc:
        return 'Supercritical-like Gas'
    else:  # T >= Tc and P >= Pc
        return 'Supercritical Fluid'

def plot_phase(T_value):
    """Plot phase regions for given temperature."""
    ax.clear()

    states = [classify_state(P, T_value) for P in P_range]

    # Color map for states
    color_map = {
        'Gas': 'lightblue',
        'Liquid': 'lightcoral',
        'Supercritical-like Gas': 'lightyellow',
        'Supercritical Fluid': 'gold'
    }

    # Plot colored regions
    for i in range(len(P_range) - 1):
        color = color_map[states[i]]
        ax.fill_between([T_value - 50, T_value + 50], P_range[i], P_range[i+1],
                        color=color, alpha=0.6)

    # Draw critical point
    ax.plot(Tc, Pc, 'ro', markersize=15, label='Critical Point')

    # Draw current point
    P_current = 10  # Example pressure
    state_current = classify_state(P_current, T_value)
    ax.plot(T_value, P_current, 'bs', markersize=12,
            label=f'Current: {state_current}')

    # Reference lines
    ax.axhline(y=Pc, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=Tc, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('Pressure (MPa)', fontsize=13)
    ax.set_title(f'CO₂ Phase at T = {T_value:.1f} K', fontsize=15, weight='bold')
    ax.set_xlim(250, 400)
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    # Annotate regions
    if T_value < Tc:
        ax.text(T_value, 3, 'Gas', fontsize=14, ha='center', weight='bold')
        ax.text(T_value, 12, 'Liquid', fontsize=14, ha='center', weight='bold')
    else:
        ax.text(T_value, 3, 'Gas-like', fontsize=12, ha='center', style='italic')
        ax.text(T_value, 12, 'Supercritical\nFluid', fontsize=14, ha='center', weight='bold')

# Initial plot
plot_phase(T_init)

# Add slider for temperature
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
temp_slider = Slider(ax_slider, 'Temperature (K)', 250, 400, valinit=T_init, valstep=1)

def update(val):
    T_new = temp_slider.val
    plot_phase(T_new)
    fig.canvas.draw_idle()

temp_slider.on_changed(update)

plt.show()

print("=== Interactive Phase Classification ===")
print("Use the slider to change temperature and observe phase transitions.")
print(f"\nCritical Point: Tc = {Tc} K, Pc = {Pc} MPa")
print("\nPhase Rules:")
print("  - T < Tc, P < Pc: Gas (below vapor pressure)")
print("  - T < Tc, P > Pc: Liquid")
print("  - T > Tc, P < Pc: Dense gas (no distinct phase)")
print("  - T > Tc, P > Pc: Supercritical Fluid")
```

**Learning Points**

- State classification depends on both temperature and pressure
- Above \\(T_c\\), there is no distinct gas-liquid transition
- Interactive visualization helps understand the phase space
- Real systems would use van der Waals or Peng-Robinson equations for accurate boundaries

## 1.6 Summary

**Key Concepts Covered**

- **Supercritical fluids** exist above the critical temperature (\\(T_c\\)) and critical pressure (\\(P_c\\))
- The **critical point** is the terminus of the liquid-gas coexistence curve on a phase diagram
- **Critical constants** (\\(T_c\\), \\(P_c\\), \\(\rho_c\\)) vary widely among substances; CO₂ has mild conditions making it popular
- **No phase boundary** exists above \\(T_c\\); density changes continuously with pressure
- **Unique properties** of SCFs combine liquid-like density with gas-like transport:
  - Density: 0.2-0.9 g/cm³ (tunable by pressure)
  - Viscosity: gas-like (low, ~10⁻⁴ Pa·s)
  - Diffusivity: intermediate (~10⁻⁷ m²/s)
  - Surface tension: zero (complete wetting)
- **Solvating power** is density-dependent and can be tuned by adjusting pressure and temperature
- **Hildebrand solubility parameter** increases with density; modifiers enhance polarity
- **Applications** exploit tunability, zero surface tension, and environmentally benign nature (especially CO₂)

**Important Equations**

Supercritical condition:
\\[
T > T_c \quad \text{and} \quad P > P_c
\\]

Reduced properties:
\\[
T_r = \frac{T}{T_c}, \quad P_r = \frac{P}{P_c}, \quad \rho_r = \frac{\rho}{\rho_c}
\\]

Hildebrand solubility parameter:
\\[
\delta = \sqrt{\frac{\Delta H_{\text{vap}} - RT}{V_m}}
\\]

## 1.7 Exercises

### Conceptual Questions

1. **Phase Diagram Interpretation**
   Explain why there is no distinct liquid-gas phase boundary above the critical temperature, even at very high pressures.

2. **Property Comparison**
   Why do supercritical fluids have gas-like viscosity but liquid-like density? What molecular factors determine this combination?

3. **Tunability**
   Describe how you would adjust pressure and temperature to maximize the dissolving power of supercritical CO₂ for a non-polar solute. What are the practical limits?

4. **Critical Opalescence**
   What causes critical opalescence near the critical point? Why does the fluid become opaque?

5. **Zero Surface Tension**
   What are the practical consequences of zero surface tension in supercritical fluid extraction and materials processing?

### Quantitative Problems

1. **Critical Constants**
   For water (\\(T_c = 374°C\\), \\(P_c = 22.06\\) MPa):
   - (a) Convert \\(T_c\\) to Kelvin and \\(P_c\\) to bar
   - (b) Calculate reduced properties at 400°C and 25 MPa
   - (c) Is this in the supercritical region?

2. **Pressure-Density Relationship**
   Using the simplified relationship \\(\rho = \rho_c (0.1 + 0.9 P_r / (1 + 2(T_r - 1)))\\) for sc-CO₂:
   - (a) Calculate density at 40°C and 100 bar
   - (b) What pressure is needed to achieve \\(\rho = 0.7\\) g/cm³ at 40°C?
   - (c) Compare densities at 50°C and 80°C at the same pressure (150 bar)

3. **Solubility Parameter**
   If the Hildebrand parameter for sc-CO₂ is \\(\delta = 15 \times (\rho/\rho_c)\\) MPa^{0.5}:
   - (a) Calculate \\(\delta\\) at \\(\rho = 0.3\\), \\(0.5\\), and \\(0.8\\) g/cm³
   - (b) Which common solvent (hexane δ=14.9, acetone δ=20, ethanol δ=26.5) does each density best match?
   - (c) Estimate the pressure needed to match ethanol's solubility parameter at 40°C

4. **Energy Considerations**
   Estimate the energy required to compress 1 kg of CO₂ from 1 bar (gas) to 150 bar (supercritical) at 40°C. Assume isothermal ideal gas behavior (simplified):
   - Work \\(W = nRT \ln(P_2/P_1)\\)
   - Compare this to the energy needed to heat water by 10°C

### Computational Exercises

1. **Phase Diagram Exploration**
   Modify Code Example 1 to:
   - (a) Add the solid-liquid and solid-gas boundaries (use approximate data)
   - (b) Plot the triple point
   - (c) Create an animation showing a heating path at constant pressure crossing into the supercritical region

2. **Property Tunability**
   Extend Code Example 3 to:
   - (a) Create a contour plot of density as a function of both P and T
   - (b) Overlay solubility parameter contours on the same plot
   - (c) Identify the optimal (P, T) operating window for a target density range

3. **Multi-Component Comparison**
   Write a function that:
   - (a) Takes critical constants as input and plots P-T diagrams for CO₂, N₂, and ethanol on the same axes
   - (b) Compares their supercritical regions
   - (c) Calculates and plots the reduced property space (\\(T_r\\) vs \\(P_r\\)) to show universality

4. **Solvent Selection Tool**
   Create an interactive tool that:
   - (a) Accepts a solute's Hildebrand parameter as input
   - (b) Suggests the required sc-CO₂ density to match
   - (c) Calculates the (P, T) conditions needed
   - (d) Recommends whether a co-solvent modifier is needed

---

**Navigation**

[← Back to Series Index](index.md) | [Next: Chapter 2 - Thermodynamics of Supercritical Fluids →](chapter-2.md)

---

## Disclaimer

This educational content was generated with AI assistance for the Hashimoto Lab knowledge base. While efforts have been made to ensure accuracy, readers should verify critical information with primary sources and established textbooks such as:

- McHugh, M. A., & Krukonis, V. J. (1994). *Supercritical Fluid Extraction: Principles and Practice* (2nd ed.). Butterworth-Heinemann.
- Brunner, G. (2005). *Supercritical Fluids as Solvents and Reaction Media*. Elsevier.
- Jessop, P. G., & Leitner, W. (1999). *Chemical Synthesis Using Supercritical Fluids*. Wiley-VCH.

---

© 2025 Hashimoto Lab, Tohoku University. All rights reserved.
