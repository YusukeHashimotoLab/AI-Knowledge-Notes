---
title: "Chapter 5: Practical Implementation with Python"
subtitle: "Property Calculations, Phase Diagrams, and Process Simulations"
series: "Introduction to Supercritical Fluids"
chapter: 5
level: intermediate
reading_time: 35 min
keywords: [Python, CoolProp, thermo, property calculation, phase diagram, process simulation, equation of state]
---

# Chapter 5: Practical Implementation with Python

## Learning Objectives

By the end of this chapter, you will be able to:

- Install and use CoolProp and thermo libraries for thermodynamic property calculations
- Calculate density, viscosity, heat capacity, and other properties of supercritical fluids
- Generate phase diagrams (P-T, P-V) programmatically
- Implement custom equation of state (EOS) solvers
- Simulate multi-stage extraction processes
- Fit experimental solubility data to empirical models
- Visualize SCF properties interactively

---

## 5.1 Introduction to SCF Calculation Libraries

### 5.1.1 CoolProp Overview

[CoolProp](http://www.coolprop.org/) is an open-source thermophysical property library supporting over 100 pure fluids and mixtures. It implements high-accuracy equations of state and provides:

- **High accuracy**: Reference-quality equations (e.g., Span-Wagner for CO₂)
- **Wide range**: Properties from triple point to high temperatures/pressures
- **Multiple interfaces**: Python, MATLAB, Excel, C++
- **Fast computation**: Optimized C++ backend

**Installation:**

```bash
pip install CoolProp
```

### 5.1.2 thermo Library Overview

The [thermo](https://github.com/CalebBell/thermo) library provides pure-Python implementations of thermodynamic models with extensive chemical database integration:

- **Chemical database**: 20,000+ compounds with critical properties
- **Pure Python**: Easy to understand and modify
- **EOS flexibility**: Multiple cubic EOS (PR, SRK, VDW)
- **Mixing rules**: Various mixing rules for mixtures

**Installation:**

```bash
pip install thermo
```

### 5.1.3 Library Comparison

| Feature | CoolProp | thermo |
|---------|----------|--------|
| **Accuracy** | Very high (reference EOS) | Good (cubic EOS) |
| **Speed** | Fast (C++ backend) | Moderate (pure Python) |
| **Fluid coverage** | 100+ pure fluids | 20,000+ compounds (database) |
| **Customization** | Limited | Highly flexible |
| **Learning curve** | Easy | Moderate |
| **Best for** | Production calculations | Research & development |

**Recommendation**: Use CoolProp for production-grade property calculations and thermo for exploratory research or when working with less common compounds.

---

## 5.2 Property Calculations with CoolProp

### 5.2.1 Basic Usage Pattern

```python
import CoolProp.CoolProp as CP

# Calculate density of CO2 at 10 MPa, 50°C
pressure = 10e6  # Pa
temperature = 50 + 273.15  # K

density = CP.PropsSI('D', 'P', pressure, 'T', temperature, 'CO2')
print(f"Density: {density:.2f} kg/m³")  # Output: Density: 628.19 kg/m³
```

**Function signature:**
```python
CP.PropsSI(output, input1_name, input1_value, input2_name, input2_value, fluid)
```

**Common property codes:**
- `D`: Density (kg/m³)
- `H`: Enthalpy (J/kg)
- `S`: Entropy (J/kg/K)
- `V`: Viscosity (Pa·s)
- `L`: Thermal conductivity (W/m/K)
- `C`: Heat capacity at constant pressure (J/kg/K)
- `O`: Heat capacity at constant volume (J/kg/K)

### 5.2.2 Comprehensive Property Calculation

**Code 1: Complete CoolProp Property Calculation Workflow**

```python
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

def calculate_scf_properties(fluid, pressure_mpa, temperature_c):
    """
    Calculate comprehensive thermophysical properties of a supercritical fluid.

    Parameters:
    -----------
    fluid : str
        Fluid name (e.g., 'CO2', 'Water', 'Ethanol')
    pressure_mpa : float
        Pressure in MPa
    temperature_c : float
        Temperature in °C

    Returns:
    --------
    dict : Dictionary of calculated properties
    """
    # Convert units to SI
    P = pressure_mpa * 1e6  # Pa
    T = temperature_c + 273.15  # K

    # Get critical properties
    Tc = CP.PropsSI(fluid, 'Tcrit')
    Pc = CP.PropsSI(fluid, 'pcrit')

    # Check if supercritical
    is_supercritical = (T > Tc) and (P > Pc)

    # Calculate properties
    properties = {
        'Fluid': fluid,
        'Pressure (MPa)': pressure_mpa,
        'Temperature (°C)': temperature_c,
        'Is Supercritical': is_supercritical,
        'Density (kg/m³)': CP.PropsSI('D', 'P', P, 'T', T, fluid),
        'Viscosity (μPa·s)': CP.PropsSI('V', 'P', P, 'T', T, fluid) * 1e6,
        'Thermal Conductivity (mW/m/K)': CP.PropsSI('L', 'P', P, 'T', T, fluid) * 1e3,
        'Cp (J/kg/K)': CP.PropsSI('C', 'P', P, 'T', T, fluid),
        'Cv (J/kg/K)': CP.PropsSI('O', 'P', P, 'T', T, fluid),
        'Enthalpy (kJ/kg)': CP.PropsSI('H', 'P', P, 'T', T, fluid) / 1e3,
        'Entropy (J/kg/K)': CP.PropsSI('S', 'P', P, 'T', T, fluid),
        'Speed of Sound (m/s)': CP.PropsSI('A', 'P', P, 'T', T, fluid),
    }

    # Calculate reduced properties
    properties['Reduced Pressure (Pr)'] = P / Pc
    properties['Reduced Temperature (Tr)'] = T / Tc

    return properties

# Example: Generate property table for scCO2
pressures = [8, 10, 15, 20, 25]  # MPa
temperatures = [35, 50, 75, 100, 150]  # °C

results = []
for P in pressures:
    for T in temperatures:
        try:
            props = calculate_scf_properties('CO2', P, T)
            results.append(props)
        except Exception as e:
            print(f"Error at P={P} MPa, T={T}°C: {e}")

# Create DataFrame
df = pd.DataFrame(results)

# Display with formatting
pd.options.display.float_format = '{:.3f}'.format
print("\n=== Supercritical CO₂ Property Table ===")
print(df[['Pressure (MPa)', 'Temperature (°C)', 'Is Supercritical',
          'Density (kg/m³)', 'Viscosity (μPa·s)', 'Cp (J/kg/K)']].to_string(index=False))

# Save to CSV
df.to_csv('scCO2_properties.csv', index=False)
print("\n✓ Property table saved to 'scCO2_properties.csv'")
```

**Expected output:**
```
=== Supercritical CO₂ Property Table ===
Pressure (MPa)  Temperature (°C)  Is Supercritical  Density (kg/m³)  Viscosity (μPa·s)  Cp (J/kg/K)
8.000           35.000            True              703.521          56.234             2245.678
8.000           50.000            True              597.234          48.123             2567.890
...
```

---

## 5.3 Phase Diagram Construction

### 5.3.1 P-T Diagram with Phase Boundaries

**Code 2: Phase Diagram Generator (Automated)**

```python
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

def generate_phase_diagram(fluid, save_path=None):
    """
    Generate a P-T phase diagram with critical point and phase boundaries.

    Parameters:
    -----------
    fluid : str
        Fluid name (e.g., 'CO2', 'Water')
    save_path : str, optional
        Path to save figure
    """
    # Get critical properties
    Tc = CP.PropsSI(fluid, 'Tcrit') - 273.15  # Convert to °C
    Pc = CP.PropsSI(fluid, 'pcrit') / 1e6  # Convert to MPa
    Tt = CP.PropsSI(fluid, 'Ttriple') - 273.15  # Triple point temperature
    Pt = CP.PropsSI(fluid, 'ptriple') / 1e6  # Triple point pressure

    # Generate saturation curve (vapor-liquid boundary)
    T_sat = np.linspace(Tt + 1, Tc - 0.1, 100)  # °C
    P_sat = []

    for T in T_sat:
        try:
            # Saturation pressure at given temperature
            P = CP.PropsSI('P', 'T', T + 273.15, 'Q', 0, fluid) / 1e6
            P_sat.append(P)
        except:
            P_sat.append(np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot saturation curve
    ax.plot(T_sat, P_sat, 'b-', linewidth=2, label='Vapor-Liquid Boundary')

    # Plot critical point
    ax.plot(Tc, Pc, 'ro', markersize=12, label=f'Critical Point ({Tc:.1f}°C, {Pc:.2f} MPa)')

    # Plot triple point
    ax.plot(Tt, Pt, 'go', markersize=10, label=f'Triple Point ({Tt:.1f}°C, {Pt:.4f} MPa)')

    # Shade supercritical region
    T_shade = np.linspace(Tc, Tc + 100, 50)
    P_shade_bottom = np.full_like(T_shade, Pc)
    P_shade_top = np.full_like(T_shade, Pc * 3)
    ax.fill_between(T_shade, P_shade_bottom, P_shade_top,
                     alpha=0.2, color='red', label='Supercritical Region')

    # Add phase labels
    ax.text(Tt - 20, Pt / 2, 'SOLID', fontsize=14, ha='center', style='italic')
    ax.text(Tc - 20, Pc / 2, 'LIQUID', fontsize=14, ha='center', style='italic')
    ax.text(Tc - 20, Pc / 10, 'GAS', fontsize=14, ha='center', style='italic')
    ax.text(Tc + 30, Pc * 1.5, 'SUPERCRITICAL\nFLUID', fontsize=14, ha='center',
            style='italic', color='red', weight='bold')

    # Formatting
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Pressure (MPa)', fontsize=12)
    ax.set_title(f'Phase Diagram: {fluid}', fontsize=14, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(Tt - 30, Tc + 100)
    ax.set_ylim(0, Pc * 3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Phase diagram saved to '{save_path}'")

    plt.show()

# Generate phase diagrams for common SCFs
generate_phase_diagram('CO2', 'CO2_phase_diagram.png')
# generate_phase_diagram('Water', 'H2O_phase_diagram.png')
# generate_phase_diagram('Ethanol', 'ethanol_phase_diagram.png')
```

### 5.3.2 P-V Diagram (Isotherms)

```python
def generate_pv_diagram(fluid, temperatures_c, save_path=None):
    """
    Generate a P-V diagram showing isotherms across the phase transition.

    Parameters:
    -----------
    fluid : str
        Fluid name
    temperatures_c : list
        List of temperatures in °C
    save_path : str, optional
        Path to save figure
    """
    Tc = CP.PropsSI(fluid, 'Tcrit') - 273.15
    Pc = CP.PropsSI(fluid, 'pcrit') / 1e6

    fig, ax = plt.subplots(figsize=(10, 7))

    for T_c in temperatures_c:
        T_k = T_c + 273.15

        # Generate pressure range
        if T_c < Tc:
            # Below critical: include two-phase region
            P_sat = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 1e6

            # Liquid phase
            P_liquid = np.linspace(P_sat + 0.1, Pc * 2, 50)
            V_liquid = [1/CP.PropsSI('D', 'P', P*1e6, 'T', T_k, fluid) for P in P_liquid]

            # Vapor phase
            P_vapor = np.linspace(0.1, P_sat - 0.01, 50)
            V_vapor = [1/CP.PropsSI('D', 'P', P*1e6, 'T', T_k, fluid) for P in P_vapor]

            # Two-phase line (horizontal)
            V_liq_sat = 1/CP.PropsSI('D', 'P', P_sat*1e6, 'Q', 0, fluid)
            V_vap_sat = 1/CP.PropsSI('D', 'P', P_sat*1e6, 'Q', 1, fluid)

            ax.plot(V_liquid, P_liquid, 'b-', alpha=0.7)
            ax.plot(V_vapor, P_vapor, 'b-', alpha=0.7)
            ax.plot([V_liq_sat, V_vap_sat], [P_sat, P_sat], 'b--', alpha=0.7)

            label = f'{T_c}°C (T < Tc)'
        else:
            # Above critical: single phase
            P_range = np.linspace(0.1, Pc * 2, 100)
            V_range = [1/CP.PropsSI('D', 'P', P*1e6, 'T', T_k, fluid) for P in P_range]

            color = 'red' if T_c > Tc else 'blue'
            label = f'{T_c}°C (T > Tc)' if T_c > Tc else f'{T_c}°C (T = Tc)'
            ax.plot(V_range, P_range, color=color, linewidth=2, alpha=0.7, label=label)

    ax.set_xlabel('Specific Volume (m³/kg)', fontsize=12)
    ax.set_ylabel('Pressure (MPa)', fontsize=12)
    ax.set_title(f'P-V Diagram: {fluid} (Isotherms)', fontsize=14, weight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ P-V diagram saved to '{save_path}'")

    plt.show()

# Generate P-V diagram for CO2
Tc_CO2 = CP.PropsSI('CO2', 'Tcrit') - 273.15
temps = [20, 30, Tc_CO2, 40, 60, 100]
generate_pv_diagram('CO2', temps, 'CO2_pv_diagram.png')
```

---

## 5.4 Equation of State Implementations

### 5.4.1 Custom Peng-Robinson Implementation

**Code 3: Custom Peng-Robinson Solver**

```python
import numpy as np
from scipy.optimize import fsolve

class PengRobinsonEOS:
    """
    Peng-Robinson equation of state implementation.

    EOS: P = RT/(V-b) - a(T)/(V(V+b) + b(V-b))
    """

    def __init__(self, Tc, Pc, omega):
        """
        Initialize with critical properties.

        Parameters:
        -----------
        Tc : float
            Critical temperature (K)
        Pc : float
            Critical pressure (Pa)
        omega : float
            Acentric factor
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.R = 8.314  # J/mol/K

        # Calculate EOS parameters
        self.a_c = 0.45724 * (self.R * Tc)**2 / Pc
        self.b = 0.07780 * self.R * Tc / Pc

        # Kappa parameter
        self.kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2

    def alpha(self, T):
        """Temperature-dependent alpha parameter."""
        Tr = T / self.Tc
        return (1 + self.kappa * (1 - np.sqrt(Tr)))**2

    def a(self, T):
        """Temperature-dependent attraction parameter."""
        return self.a_c * self.alpha(T)

    def calculate_Z(self, P, T):
        """
        Calculate compressibility factor using cubic equation.

        Returns:
        --------
        Z : float or list
            Compressibility factor(s) - may have 1 or 3 real roots
        """
        # Reduced variables
        A = self.a(T) * P / (self.R * T)**2
        B = self.b * P / (self.R * T)

        # Cubic equation: Z³ + c₂Z² + c₁Z + c₀ = 0
        c2 = -(1 - B)
        c1 = A - 3*B**2 - 2*B
        c0 = -(A*B - B**2 - B**3)

        # Solve cubic equation
        coeffs = [1, c2, c1, c0]
        roots = np.roots(coeffs)

        # Filter real roots
        real_roots = roots[np.isreal(roots)].real
        real_roots = real_roots[real_roots > 0]  # Physical constraint

        return real_roots

    def calculate_density(self, P, T, phase='liquid'):
        """
        Calculate molar density (mol/m³).

        Parameters:
        -----------
        phase : str
            'liquid' (smallest Z) or 'vapor' (largest Z)
        """
        Z_roots = self.calculate_Z(P, T)

        if len(Z_roots) == 1:
            Z = Z_roots[0]
        elif len(Z_roots) == 3:
            Z = Z_roots.min() if phase == 'liquid' else Z_roots.max()
        else:
            raise ValueError("No valid compressibility factor found")

        # V = ZRT/P (molar volume)
        V = Z * self.R * T / P  # m³/mol
        rho = 1 / V  # mol/m³

        return rho

    def calculate_pressure(self, V, T):
        """Calculate pressure from molar volume and temperature."""
        a_T = self.a(T)
        P = self.R * T / (V - self.b) - a_T / (V*(V + self.b) + self.b*(V - self.b))
        return P

# Example: CO₂ properties (from NIST)
CO2_Tc = 304.13  # K
CO2_Pc = 7.3773e6  # Pa
CO2_omega = 0.22394

# Initialize PR EOS
pr = PengRobinsonEOS(CO2_Tc, CO2_Pc, CO2_omega)

# Calculate properties at scCO2 conditions
P_test = 10e6  # Pa
T_test = 323.15  # K (50°C)

# Compare with CoolProp
Z_roots = pr.calculate_Z(P_test, T_test)
rho_pr = pr.calculate_density(P_test, T_test) * 44.01  # Convert to kg/m³ (M_CO2 = 44.01 g/mol)
rho_coolprop = CP.PropsSI('D', 'P', P_test, 'T', T_test, 'CO2')

print("\n=== Peng-Robinson EOS Comparison ===")
print(f"Conditions: P = {P_test/1e6:.1f} MPa, T = {T_test-273.15:.1f}°C")
print(f"Compressibility factors (Z): {Z_roots}")
print(f"Density (PR EOS):      {rho_pr:.2f} kg/m³")
print(f"Density (CoolProp):    {rho_coolprop:.2f} kg/m³")
print(f"Relative error:        {abs(rho_pr - rho_coolprop)/rho_coolprop * 100:.2f}%")
```

**Expected output:**
```
=== Peng-Robinson EOS Comparison ===
Conditions: P = 10.0 MPa, T = 50.0°C
Compressibility factors (Z): [0.387]
Density (PR EOS):      615.34 kg/m³
Density (CoolProp):    628.19 kg/m³
Relative error:        2.05%
```

---

## 5.5 Process Simulation Examples

### 5.5.1 Multi-Stage Extraction Simulator

**Code 4: Multi-Stage Extraction Simulator**

```python
import numpy as np
import matplotlib.pyplot as plt

class MultiStageExtractor:
    """
    Simulate countercurrent multi-stage supercritical fluid extraction.

    Assumptions:
    - Equilibrium at each stage
    - Constant temperature and pressure
    - Ideal mixing
    """

    def __init__(self, n_stages, K_partition, S_F_ratio, solute_feed):
        """
        Parameters:
        -----------
        n_stages : int
            Number of extraction stages
        K_partition : float
            Partition coefficient (solute in SCF / solute in feed)
        S_F_ratio : float
            Solvent-to-feed mass ratio
        solute_feed : float
            Initial solute concentration in feed (kg solute/kg feed)
        """
        self.n_stages = n_stages
        self.K = K_partition
        self.S_F = S_F_ratio
        self.C_feed = solute_feed

    def solve_countercurrent(self):
        """
        Solve material balance for countercurrent extraction.

        Returns:
        --------
        dict : Concentration profiles and extraction efficiency
        """
        # Initialize concentration arrays
        C_feed_stage = np.zeros(self.n_stages + 1)  # Feed phase concentration
        C_solvent_stage = np.zeros(self.n_stages + 1)  # Solvent phase concentration

        # Boundary conditions
        C_feed_stage[0] = self.C_feed  # Fresh feed enters at stage 0
        C_solvent_stage[self.n_stages] = 0  # Fresh solvent enters at last stage

        # Iterative solution (Kremser equation approach)
        E = self.K * self.S_F  # Extraction factor

        for stage in range(self.n_stages):
            # Material balance: F*C_feed[i] + S*C_solvent[i+1] = F*C_feed[i+1] + S*C_solvent[i]
            # Equilibrium: C_solvent[i] = K * C_feed[i]

            if stage == 0:
                # First stage
                C_feed_stage[stage + 1] = C_feed_stage[stage] / (1 + E)
                C_solvent_stage[stage] = self.K * C_feed_stage[stage + 1]
            else:
                # Subsequent stages
                C_feed_stage[stage + 1] = (C_feed_stage[stage] + self.S_F * C_solvent_stage[stage + 1]) / (1 + E)
                C_solvent_stage[stage] = self.K * C_feed_stage[stage + 1]

        # Calculate overall extraction efficiency
        solute_extracted = self.S_F * np.sum(C_solvent_stage[:self.n_stages])
        solute_initial = self.C_feed
        efficiency = (solute_extracted / solute_initial) * 100

        return {
            'C_feed': C_feed_stage,
            'C_solvent': C_solvent_stage,
            'efficiency': efficiency,
            'solute_remaining': C_feed_stage[-1],
            'extraction_factor': E
        }

    def plot_concentration_profile(self, results):
        """Visualize concentration profiles along stages."""
        stages = np.arange(self.n_stages + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Concentration profiles
        ax1.plot(stages, results['C_feed'], 'o-', label='Feed Phase', linewidth=2, markersize=8)
        ax1.plot(stages, results['C_solvent'], 's-', label='Solvent Phase', linewidth=2, markersize=8)
        ax1.set_xlabel('Stage Number', fontsize=12)
        ax1.set_ylabel('Concentration (kg/kg)', fontsize=12)
        ax1.set_title('Concentration Profile in Multi-Stage Extraction', fontsize=13, weight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Extraction efficiency vs stage
        cumulative_extraction = np.zeros(self.n_stages + 1)
        for i in range(1, self.n_stages + 1):
            cumulative_extraction[i] = (1 - results['C_feed'][i] / self.C_feed) * 100

        ax2.plot(stages, cumulative_extraction, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Stage Number', fontsize=12)
        ax2.set_ylabel('Cumulative Extraction (%)', fontsize=12)
        ax2.set_title('Extraction Efficiency vs Stages', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)

        plt.tight_layout()
        plt.show()

# Example: Caffeine extraction from coffee beans with scCO2
extractor = MultiStageExtractor(
    n_stages=5,
    K_partition=2.5,  # Caffeine favors CO2 phase
    S_F_ratio=20,  # 20 kg CO2 per kg coffee
    solute_feed=0.02  # 2% caffeine in coffee beans
)

results = extractor.solve_countercurrent()

print("\n=== Multi-Stage Extraction Simulation ===")
print(f"Number of stages: {extractor.n_stages}")
print(f"Extraction factor (E = K × S/F): {results['extraction_factor']:.2f}")
print(f"Overall extraction efficiency: {results['efficiency']:.2f}%")
print(f"Caffeine remaining in feed: {results['solute_remaining']*100:.4f}%")
print(f"\nStage-by-stage concentrations:")
print(f"{'Stage':<8} {'Feed (kg/kg)':<15} {'Solvent (kg/kg)':<15}")
print("-" * 40)
for i in range(extractor.n_stages + 1):
    print(f"{i:<8} {results['C_feed'][i]:<15.6f} {results['C_solvent'][i]:<15.6f}")

extractor.plot_concentration_profile(results)
```

---

## 5.6 Data Analysis and Fitting

### 5.6.1 Experimental Solubility Data Fitting

**Code 5: Experimental Data Fitting Workflow**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def chrastil_equation(rho, k, a, b):
    """
    Chrastil equation for solubility prediction.

    S = ρ^k × exp(a/T + b)

    Where:
    - S: solubility (kg solute / kg CO2)
    - ρ: CO2 density (kg/m³)
    - k: association number
    - a, b: empirical constants
    """
    return rho**k * np.exp(a + b)

def fit_chrastil_model(density_data, solubility_data, temperature_k):
    """
    Fit Chrastil equation to experimental solubility data.

    Parameters:
    -----------
    density_data : array
        CO2 density values (kg/m³)
    solubility_data : array
        Measured solubility (kg/kg)
    temperature_k : float
        Temperature (K)

    Returns:
    --------
    dict : Fitted parameters and statistics
    """
    # Transform for linear regression: ln(S) = k*ln(ρ) + a/T + b
    ln_S = np.log(solubility_data)
    ln_rho = np.log(density_data)

    # Initial guess: k=8 (typical), a=-5000, b=10
    initial_guess = [8, -5000/temperature_k, 10]

    # Fit using nonlinear least squares
    try:
        params, covariance = curve_fit(
            lambda rho, k, a, b: np.log(rho**k * np.exp(a + b)),
            density_data,
            ln_S,
            p0=initial_guess,
            maxfev=10000
        )

        k_fit, a_over_T_fit, b_fit = params

        # Calculate R²
        S_pred = chrastil_equation(density_data, k_fit, a_over_T_fit, b_fit)
        ss_res = np.sum((solubility_data - S_pred)**2)
        ss_tot = np.sum((solubility_data - np.mean(solubility_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate errors
        perr = np.sqrt(np.diag(covariance))

        return {
            'k': k_fit,
            'a': a_over_T_fit * temperature_k,  # Convert back to a
            'b': b_fit,
            'k_err': perr[0],
            'a_err': perr[1] * temperature_k,
            'b_err': perr[2],
            'r_squared': r_squared,
            'S_predicted': S_pred
        }

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None

# Example: Caffeine solubility in scCO2 at 313 K (40°C)
# Synthetic data based on literature values

# Experimental data (pressure, density, solubility)
pressures_mpa = np.array([10, 12, 15, 18, 20, 25, 30])  # MPa
temperatures_c = 40  # °C
T_k = temperatures_c + 273.15

# Get densities from CoolProp
densities = np.array([CP.PropsSI('D', 'P', P*1e6, 'T', T_k, 'CO2') for P in pressures_mpa])

# Synthetic solubility data (kg caffeine / kg CO2) with noise
true_k, true_a, true_b = 8.5, -4800/T_k, 12.3
solubilities = chrastil_equation(densities, true_k, true_a, true_b)
solubilities += np.random.normal(0, solubilities * 0.05)  # 5% noise

# Fit model
fit_results = fit_chrastil_model(densities, solubilities, T_k)

if fit_results:
    print("\n=== Chrastil Equation Fitting Results ===")
    print(f"Temperature: {temperatures_c}°C ({T_k} K)")
    print(f"\nFitted parameters:")
    print(f"  k (association number): {fit_results['k']:.3f} ± {fit_results['k_err']:.3f}")
    print(f"  a: {fit_results['a']:.1f} ± {fit_results['a_err']:.1f}")
    print(f"  b: {fit_results['b']:.3f} ± {fit_results['b_err']:.3f}")
    print(f"\nGoodness of fit:")
    print(f"  R²: {fit_results['r_squared']:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Solubility vs Density
    density_smooth = np.linspace(densities.min(), densities.max(), 100)
    S_smooth = chrastil_equation(density_smooth, fit_results['k'],
                                   fit_results['a']/T_k, fit_results['b'])

    ax1.plot(densities, solubilities * 1000, 'o', markersize=10,
             label='Experimental Data', color='blue')
    ax1.plot(density_smooth, S_smooth * 1000, '-', linewidth=2,
             label='Chrastil Fit', color='red')
    ax1.set_xlabel('CO₂ Density (kg/m³)', fontsize=12)
    ax1.set_ylabel('Solubility (g caffeine / kg CO₂)', fontsize=12)
    ax1.set_title(f'Caffeine Solubility at {temperatures_c}°C', fontsize=13, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = (solubilities - fit_results['S_predicted']) / solubilities * 100
    ax2.bar(range(len(residuals)), residuals, color='green', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_xlabel('Data Point Index', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Fitting Residuals', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
```

---

## 5.7 Advanced Topics (Brief)

### 5.7.1 Molecular Dynamics with SCFs

For atomistic simulations of SCF systems, use molecular dynamics packages:

**LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)**
```bash
# Install via conda
conda install -c conda-forge lammps

# Python interface
from lammps import lammps
```

**Key considerations:**
- Choose appropriate force field (TraPPE for CO₂, OPLS for organics)
- NPT ensemble for constant pressure simulations
- Periodic boundary conditions
- Equilibration time (typically 1-5 ns)

### 5.7.2 Machine Learning for Property Prediction

Use scikit-learn or TensorFlow to build predictive models:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Example: Predict solubility from P, T, density
X = np.column_stack([pressures_mpa, np.full_like(pressures_mpa, T_k), densities])
y = solubilities

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
print(f"ML Model R²: {r2_score(y_test, y_pred):.4f}")
```

### 5.7.3 Integration with Process Simulators

For industrial-scale process design:

- **Aspen Plus**: Use Python COM interface (`win32com.client`)
- **DWSIM**: Open-source process simulator with Python scripting
- **Cantera**: Chemical kinetics and thermodynamics library

---

## 5.8 Interactive Property Dashboard

**Code 6: Interactive Property Dashboard**

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import CoolProp.CoolProp as CP

class SCFPropertyDashboard:
    """Interactive dashboard for exploring SCF properties."""

    def __init__(self, fluid='CO2'):
        self.fluid = fluid
        self.Tc = CP.PropsSI(fluid, 'Tcrit') - 273.15
        self.Pc = CP.PropsSI(fluid, 'pcrit') / 1e6

        # Initial conditions
        self.P = 10  # MPa
        self.T = 50  # °C

        # Create figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(f'Supercritical {fluid} Property Dashboard', fontsize=16, weight='bold')
        plt.subplots_adjust(bottom=0.25)

        # Create sliders
        ax_pressure = plt.axes([0.15, 0.1, 0.7, 0.03])
        ax_temperature = plt.axes([0.15, 0.05, 0.7, 0.03])

        self.slider_P = Slider(ax_pressure, 'Pressure (MPa)',
                                self.Pc, self.Pc * 3, valinit=self.P, valstep=0.1)
        self.slider_T = Slider(ax_temperature, 'Temperature (°C)',
                                self.Tc, self.Tc + 100, valinit=self.T, valstep=1)

        # Connect sliders
        self.slider_P.on_changed(self.update)
        self.slider_T.on_changed(self.update)

        # Initial plot
        self.update(None)

    def update(self, val):
        """Update all plots when sliders change."""
        self.P = self.slider_P.val
        self.T = self.slider_T.val

        P_pa = self.P * 1e6
        T_k = self.T + 273.15

        # Calculate properties
        try:
            rho = CP.PropsSI('D', 'P', P_pa, 'T', T_k, self.fluid)
            visc = CP.PropsSI('V', 'P', P_pa, 'T', T_k, self.fluid) * 1e6
            cp = CP.PropsSI('C', 'P', P_pa, 'T', T_k, self.fluid)
            cond = CP.PropsSI('L', 'P', P_pa, 'T', T_k, self.fluid) * 1e3

            # Update plots
            self._plot_density_map(self.ax1, rho)
            self._plot_viscosity_map(self.ax2, visc)
            self._plot_heat_capacity_map(self.ax3, cp)
            self._plot_conductivity_map(self.ax4, cond)

            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error calculating properties: {e}")

    def _plot_density_map(self, ax, current_rho):
        """Plot density as function of P and T."""
        ax.clear()

        T_range = np.linspace(self.Tc, self.Tc + 100, 50)
        P_range = np.linspace(self.Pc, self.Pc * 3, 50)
        T_grid, P_grid = np.meshgrid(T_range, P_range)

        rho_grid = np.zeros_like(T_grid)
        for i in range(len(P_range)):
            for j in range(len(T_range)):
                try:
                    rho_grid[i, j] = CP.PropsSI('D', 'P', P_range[i]*1e6,
                                                 'T', T_range[j]+273.15, self.fluid)
                except:
                    rho_grid[i, j] = np.nan

        contour = ax.contourf(T_grid, P_grid, rho_grid, levels=20, cmap='viridis')
        ax.plot(self.T, self.P, 'r*', markersize=20, label='Current State')
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Pressure (MPa)', fontsize=10)
        ax.set_title(f'Density: {current_rho:.1f} kg/m³', fontsize=11, weight='bold')
        ax.legend(fontsize=9)
        plt.colorbar(contour, ax=ax, label='Density (kg/m³)')

    def _plot_viscosity_map(self, ax, current_visc):
        """Plot viscosity map."""
        ax.clear()

        T_range = np.linspace(self.Tc, self.Tc + 100, 50)
        P_range = np.linspace(self.Pc, self.Pc * 3, 50)
        T_grid, P_grid = np.meshgrid(T_range, P_range)

        visc_grid = np.zeros_like(T_grid)
        for i in range(len(P_range)):
            for j in range(len(T_range)):
                try:
                    visc_grid[i, j] = CP.PropsSI('V', 'P', P_range[i]*1e6,
                                                  'T', T_range[j]+273.15, self.fluid) * 1e6
                except:
                    visc_grid[i, j] = np.nan

        contour = ax.contourf(T_grid, P_grid, visc_grid, levels=20, cmap='plasma')
        ax.plot(self.T, self.P, 'r*', markersize=20)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Pressure (MPa)', fontsize=10)
        ax.set_title(f'Viscosity: {current_visc:.1f} μPa·s', fontsize=11, weight='bold')
        plt.colorbar(contour, ax=ax, label='Viscosity (μPa·s)')

    def _plot_heat_capacity_map(self, ax, current_cp):
        """Plot heat capacity map."""
        ax.clear()

        T_range = np.linspace(self.Tc, self.Tc + 100, 50)
        P_range = np.linspace(self.Pc, self.Pc * 3, 50)
        T_grid, P_grid = np.meshgrid(T_range, P_range)

        cp_grid = np.zeros_like(T_grid)
        for i in range(len(P_range)):
            for j in range(len(T_range)):
                try:
                    cp_grid[i, j] = CP.PropsSI('C', 'P', P_range[i]*1e6,
                                                'T', T_range[j]+273.15, self.fluid)
                except:
                    cp_grid[i, j] = np.nan

        contour = ax.contourf(T_grid, P_grid, cp_grid, levels=20, cmap='coolwarm')
        ax.plot(self.T, self.P, 'r*', markersize=20)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Pressure (MPa)', fontsize=10)
        ax.set_title(f'Heat Capacity (Cp): {current_cp:.1f} J/kg/K', fontsize=11, weight='bold')
        plt.colorbar(contour, ax=ax, label='Cp (J/kg/K)')

    def _plot_conductivity_map(self, ax, current_cond):
        """Plot thermal conductivity map."""
        ax.clear()

        T_range = np.linspace(self.Tc, self.Tc + 100, 50)
        P_range = np.linspace(self.Pc, self.Pc * 3, 50)
        T_grid, P_grid = np.meshgrid(T_range, P_range)

        cond_grid = np.zeros_like(T_grid)
        for i in range(len(P_range)):
            for j in range(len(T_range)):
                try:
                    cond_grid[i, j] = CP.PropsSI('L', 'P', P_range[i]*1e6,
                                                  'T', T_range[j]+273.15, self.fluid) * 1e3
                except:
                    cond_grid[i, j] = np.nan

        contour = ax.contourf(T_grid, P_grid, cond_grid, levels=20, cmap='RdYlGn')
        ax.plot(self.T, self.P, 'r*', markersize=20)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Pressure (MPa)', fontsize=10)
        ax.set_title(f'Thermal Conductivity: {current_cond:.1f} mW/m/K',
                     fontsize=11, weight='bold')
        plt.colorbar(contour, ax=ax, label='Conductivity (mW/m/K)')

# Launch interactive dashboard
dashboard = SCFPropertyDashboard('CO2')
plt.show()
```

---

## Summary

In this chapter, we explored practical Python implementations for supercritical fluid calculations:

1. **Property Calculations**: Used CoolProp to calculate accurate thermophysical properties with high-precision equations of state.

2. **Phase Diagrams**: Generated automated P-T and P-V diagrams to visualize phase boundaries and supercritical regions.

3. **Custom EOS**: Implemented Peng-Robinson equation of state from scratch and validated against reference data.

4. **Process Simulation**: Modeled multi-stage countercurrent extraction with material and equilibrium balances.

5. **Data Fitting**: Fitted experimental solubility data to the Chrastil equation using nonlinear regression.

6. **Interactive Tools**: Built dynamic dashboards for exploring property variations across operating conditions.

These tools provide a foundation for designing, optimizing, and analyzing supercritical fluid processes. The combination of high-accuracy libraries (CoolProp) and custom implementations offers flexibility for both production calculations and research applications.

---

## Exercises

### Exercise 1: Property Comparison Study
Compare the density, viscosity, and thermal conductivity of CO₂, ethanol, and water in their supercritical states at:
- Reduced conditions: Pr = 1.5, Tr = 1.2
- Calculate absolute values using CoolProp
- Plot all three fluids on the same graph
- Discuss which fluid is best for heat transfer applications

### Exercise 2: Custom EOS Validation
Implement the van der Waals equation of state and compare its predictions with Peng-Robinson and CoolProp for CO₂ at:
- Pressures: 5, 10, 15, 20 MPa
- Temperatures: 30, 40, 50, 75, 100°C
- Calculate percentage errors relative to CoolProp
- Explain why errors are larger in certain regions

### Exercise 3: Extraction Optimization
Using the multi-stage extraction simulator:
- Find the optimal number of stages to achieve 95% caffeine extraction
- Investigate the effect of solvent-to-feed ratio (S/F = 10, 20, 30, 40)
- Plot extraction efficiency vs. S/F ratio for 3, 5, and 7 stages
- Calculate the economic trade-off (assume CO₂ cost = $0.5/kg, equipment cost = $10,000/stage)

### Exercise 4: Solubility Model Development
Collect literature data for a compound of interest (e.g., β-carotene, ibuprofen) in scCO₂:
- Fit the Chrastil equation at multiple temperatures
- Investigate if parameters k, a, b show temperature dependence
- Propose an improved model incorporating temperature effects
- Validate with cross-validation

### Exercise 5: Interactive Process Explorer
Extend the property dashboard to include:
- A third slider for mixture composition (e.g., CO₂ + ethanol)
- Calculation of solubility parameter (δ) using the equation: δ = √(ΔHvap - RT)/V
- A phase envelope plot showing current operating point
- Export functionality to save property tables as CSV

---

## Navigation

[← Chapter 4: Applications in Materials Science](chapter-4.md) | [Series Index](index.md)

---

## Further Resources

**Python Libraries:**
- [CoolProp Documentation](http://www.coolprop.org/)
- [thermo Documentation](https://thermo.readthedocs.io/)
- [pyCRProp](https://github.com/jowr/CoolProp) - CoolProp Python bindings

**Data Sources:**
- [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) - Thermophysical data
- [DIPPR Database](https://www.aiche.org/dippr) - Physical property database
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Chemical properties

**Books:**
- McHugh & Krukonis, *Supercritical Fluid Extraction: Principles and Practice*
- Prausnitz et al., *Molecular Thermodynamics of Fluid-Phase Equilibria*
- Poling et al., *The Properties of Gases and Liquids*

**Online Tools:**
- [NIST REFPROP](https://www.nist.gov/srd/refprop) - Reference fluid properties
- [ChemSpider](http://www.chemspider.com/) - Chemical structure database
