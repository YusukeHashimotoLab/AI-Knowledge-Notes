---
title: "Chapter 7: Practical Python for Supercritical Fluids"
chapter_title: "Chapter 7: Practical Python for Supercritical Fluids"
subtitle: Property Calculations, Phase Diagrams, Custom EOS Solvers, and Process Simulation
reading_time: 35-40 minutes
difficulty: Advanced
code_examples: 10
---

Chapter 6 gave you the equations; this chapter turns them into working code. We start with CoolProp, the reference-quality property library that removes the need to hand-code an equation of state, then build phase diagrams programmatically, implement van der Waals and Peng-Robinson solvers and benchmark them against reference data, and finish with process models for multi-stage extraction, RESS particle formation, and solubility data fitting. Every numbered example is self-contained, and the printed outputs were produced by actually running the code.

## Learning Objectives

After completing this chapter, you will be able to:

  * Install and use CoolProp and `thermo` for thermophysical property calculations, and choose sensibly between them
  * Retrieve density, viscosity, thermal conductivity, heat capacity, enthalpy and speed of sound for any supported fluid at arbitrary $T$ and $P$
  * Generate $P$-$T$ and $P$-$V$ phase diagrams programmatically, including saturation curves and the supercritical region
  * Implement van der Waals and Peng-Robinson solvers from scratch and quantify their error against a reference EOS
  * Simulate a multi-stage supercritical extraction cascade and interpret the concentration profiles
  * Model an scCO₂ extraction run end to end: solubility, yield, and CO₂ consumption per kilogram of product
  * Estimate RESS particle size from nozzle conditions
  * Fit experimental solubility data to the Chrastil equation and report honest parameter uncertainties

* * *

## 7.1 Supercritical Fluid Calculation Libraries

### CoolProp

[CoolProp](<http://www.coolprop.org/>) is an open-source thermophysical property library covering more than 120 pure fluids plus mixtures. It implements reference-quality Helmholtz-energy equations of state — Span-Wagner for CO₂, IAPWS-95 for water — so its output is close to experimental accuracy rather than a cubic-EOS approximation.

  * **High accuracy** : reference equations, typically better than 0.1% in density
  * **Wide range** : triple point to high temperature and pressure
  * **Multiple interfaces** : Python, MATLAB, Excel, C++
  * **Fast** : optimized C++ backend

Installation
    
    
    pip install CoolProp

### The thermo Library

[thermo](<https://github.com/CalebBell/thermo>) is a pure-Python chemical-engineering property library with a very large compound database and a flexible model layer.

  * **Chemical database** : 20,000+ compounds with critical properties
  * **Pure Python** : easy to read, modify and extend
  * **EOS flexibility** : multiple cubic equations (PR, SRK, VDW) and mixing rules
  * **Group contribution** : estimation methods when experimental data are missing

Installation
    
    
    pip install thermo

### Choosing Between Them

Feature | CoolProp | thermo  
---|---|---  
Accuracy | Very high (reference EOS) | Good (cubic EOS and estimation methods)  
Speed | Fast (C++ backend) | Moderate (pure Python)  
Fluid coverage | 120+ pure fluids | 20,000+ compounds (database)  
Mixtures | Supported (HEOS backend) | Strong support, flexible mixing rules  
Customization | Limited | Highly flexible  
Learning curve | Gentle | Moderate  
Best for | Production property calculations | Research, uncommon compounds, model development  
  
**Recommendation** : use CoolProp for the solvent (CO₂, water, ethanol), where reference equations exist, and `thermo` for the solute, which is usually an organic compound with no reference EOS. The examples in this chapter use CoolProp, because in supercritical processes it is the solvent properties that dominate the design.

* * *

## 7.2 Property Calculations with CoolProp

### The Basic Call Pattern

Almost everything goes through a single function, `PropsSI`, which works entirely in SI units:

PropsSI: the one function you need
    
    
    from CoolProp.CoolProp import PropsSI
    
    # Signature: PropsSI(output, input1_name, input1_value,
    #                            input2_name, input2_value, fluid)
    
    # Density of CO2 at 10 MPa and 50 degC
    rho = PropsSI('D', 'P', 10e6, 'T', 50 + 273.15, 'CO2')
    print(f"Density: {rho:.2f} kg/m³")   # Density: 384.33 kg/m³

The state is fixed by any two independent properties, so `('P', 'T')`, `('T', 'D')`, `('P', 'H')` and `('T', 'Q')` (vapour quality, for saturation states) are all valid input pairs.

Code | Property | SI unit  
---|---|---  
`T`| Temperature| K  
`P`| Pressure| Pa  
`D`| Density| kg/m³  
`H`| Specific enthalpy| J/kg  
`S`| Specific entropy| J/(kg·K)  
`C`| Isobaric heat capacity $c_p$| J/(kg·K)  
`O`| Isochoric heat capacity $c_v$| J/(kg·K)  
`V`| Dynamic viscosity| Pa·s  
`L`| Thermal conductivity| W/(m·K)  
`A`| Speed of sound| m/s  
`Q`| Vapour quality| -  
  
**Everything is SI, and that trips people up.** Pressures are pascals, not bar or MPa; temperatures are kelvin, not °C; viscosity is Pa·s, not cP. A factor-of-10⁶ error in pressure is the single most common CoolProp mistake. Convert once, at the boundary of your code, and keep SI internally.

### Example 1: A Complete Property Table

The first thing you usually want is a property table over the intended operating window. This example builds one, flags which states are actually supercritical, adds reduced coordinates, and writes a CSV.

Code Example 1: Comprehensive scCO₂ Property Table
    
    
    import CoolProp.CoolProp as CP
    import pandas as pd
    
    def calculate_scf_properties(fluid, pressure_mpa, temperature_c):
        """
        Calculate comprehensive thermophysical properties of a supercritical fluid.
    
        Parameters:
        -----------
        fluid : str
            Fluid name (e.g. 'CO2', 'Water', 'Ethanol')
        pressure_mpa : float
            Pressure in MPa
        temperature_c : float
            Temperature in °C
    
        Returns:
        --------
        dict : Dictionary of calculated properties
        """
        # Convert units to SI
        P = pressure_mpa * 1e6      # Pa
        T = temperature_c + 273.15  # K
    
        # Critical properties
        Tc = CP.PropsSI('Tcrit', fluid)
        Pc = CP.PropsSI('pcrit', fluid)
    
        # A fluid is supercritical only if BOTH are exceeded
        is_supercritical = (T > Tc) and (P > Pc)
    
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
    
        # Reduced properties
        properties['Reduced Pressure (Pr)'] = P / Pc
        properties['Reduced Temperature (Tr)'] = T / Tc
    
        return properties
    
    # Generate a property table for scCO2 over a realistic operating window
    pressures = [8, 10, 15, 20, 25]       # MPa
    temperatures = [35, 50, 75, 100, 150]  # °C
    
    results = []
    for P in pressures:
        for T in temperatures:
            try:
                results.append(calculate_scf_properties('CO2', P, T))
            except Exception as e:
                print(f"Error at P={P} MPa, T={T}°C: {e}")
    
    df = pd.DataFrame(results)
    
    pd.options.display.float_format = '{:.2f}'.format
    print("=== Supercritical CO2 Property Table ===")
    print(df[['Pressure (MPa)', 'Temperature (°C)', 'Is Supercritical',
              'Density (kg/m³)', 'Viscosity (μPa·s)',
              'Cp (J/kg/K)']].to_string(index=False))
    
    df.to_csv('scCO2_properties.csv', index=False)
    print("\nProperty table saved to 'scCO2_properties.csv'")

=== Supercritical CO2 Property Table === Pressure (MPa) Temperature (°C) Is Supercritical Density (kg/m³) Viscosity (μPa·s) Cp (J/kg/K) 8 35 True 419.09 29.16 29593.72 8 50 True 219.18 20.29 2512.52 8 75 True 166.54 20.05 1573.36 8 100 True 141.27 20.68 1335.83 8 150 True 112.98 22.42 1180.29 10 35 True 712.81 57.99 3988.64 10 50 True 384.33 27.79 5807.71 10 75 True 233.43 22.09 2009.83 10 100 True 188.56 21.92 1521.75 10 150 True 145.56 23.15 1253.35 15 35 True 815.06 74.49 2534.09 15 50 True 699.75 56.77 3049.49 15 75 True 463.33 34.24 3165.55 15 100 True 332.35 27.52 2103.98 15 150 True 233.93 25.76 1455.21 20 35 True 865.72 84.72 2201.52 20 50 True 784.29 69.45 2371.44 20 75 True 626.23 48.92 2620.93 20 100 True 480.53 36.71 2340.43 20 150 True 327.10 29.62 1639.76 25 35 True 901.23 92.97 2039.98 25 50 True 834.19 78.50 2120.56 25 75 True 711.61 59.38 2247.71 25 100 True 588.45 46.11 2199.85 25 150 True 415.50 34.53 1749.36 Property table saved to 'scCO2_properties.csv'

#### Three things this table tells you

  * **The density cliff.** At 10 MPa, cooling from 50 °C to 35 °C nearly doubles the density, from 384 to 713 kg/m³. That single number is the commercial basis of supercritical extraction: dissolve at high density, precipitate at low density, and use pressure or temperature as the switch.
  * **The $c_p$ divergence is real.** At 8 MPa and 35 °C — barely above the critical point at 7.38 MPa, 31.0 °C — $c_p$ reaches 29 594 J/(kg·K), twenty-five times its high-temperature value. This is the divergence derived in Chapter 6, and it is why heaters near the critical point respond so sluggishly.
  * **Gas-like viscosity persists.** Even at 901 kg/m³ (25 MPa, 35 °C) the viscosity is 93 μPa·s, about a tenth that of liquid water. Liquid-like density with gas-like viscosity is exactly the combination that makes supercritical mass transfer fast.

### Example 2: Critical and Triple Point Data

Before any calculation, check where the critical point actually is. CoolProp exposes fixed-point data through the same function with only two arguments.

Code Example 2: Fixed-Point Data for Common SCF Solvents
    
    
    from CoolProp.CoolProp import PropsSI
    
    def print_fixed_points(fluid):
        """Print critical and triple point data for a fluid."""
        T_crit = PropsSI('Tcrit', fluid)      # K
        P_crit = PropsSI('pcrit', fluid)      # Pa
        rho_crit = PropsSI('rhocrit', fluid)  # kg/m³
        T_triple = PropsSI('Ttriple', fluid)  # K
        P_triple = PropsSI('ptriple', fluid)  # Pa
        M = PropsSI('molar_mass', fluid)      # kg/mol
    
        print(f"{fluid}:")
        print(f"  Molar mass:     {M * 1000:.2f} g/mol")
        print(f"  Critical point: {T_crit - 273.15:7.2f} °C, "
              f"{P_crit / 1e6:6.3f} MPa, {rho_crit:6.1f} kg/m³")
        print(f"  Triple point:   {T_triple - 273.15:7.2f} °C, "
              f"{P_triple:.3e} Pa")
    
        # Critical compressibility factor
        R = 8.314462618  # J/(mol·K)
        Zc = P_crit * M / (rho_crit * R * T_crit)
        print(f"  Zc = {Zc:.4f}")
    
    for fluid in ['CO2', 'Water', 'Ethanol']:
        print_fixed_points(fluid)
        print()

CO2: Molar mass: 44.01 g/mol Critical point: 30.98 °C, 7.377 MPa, 467.6 kg/m³ Triple point: -56.56 °C, 5.180e+05 Pa Zc = 0.2746 Water: Molar mass: 18.02 g/mol Critical point: 373.95 °C, 22.064 MPa, 322.0 kg/m³ Triple point: 0.01 °C, 6.117e+02 Pa Zc = 0.2294 Ethanol: Molar mass: 46.07 g/mol Critical point: 241.56 °C, 6.268 MPa, 273.2 kg/m³ Triple point: -114.05 °C, 7.354e-04 Pa Zc = 0.2470

Two observations worth internalizing. First, the computed $Z_c$ values (0.229-0.275) sit well below the van der Waals prediction of 0.375 derived in Chapter 6 — a three-line numerical confirmation of that equation's central weakness. Second, CO₂'s triple-point pressure is 0.518 MPa, above atmospheric, which is precisely why solid CO₂ sublimes rather than melts at ambient pressure.

* * *

## 7.3 Phase Diagram Construction

### Example 3: Automated P-T Phase Diagram

A reusable phase-diagram generator: it looks up the fixed points, traces the saturation curve, shades the supercritical region, and labels the phases.

Code Example 3: P-T Phase Diagram Generator
    
    
    import CoolProp.CoolProp as CP
    import matplotlib.pyplot as plt
    import numpy as np
    
    def generate_phase_diagram(fluid, save_path=None):
        """
        Generate a P-T phase diagram with critical point and phase boundaries.
    
        Parameters:
        -----------
        fluid : str
            Fluid name (e.g. 'CO2', 'Water')
        save_path : str, optional
            Path to save the figure
        """
        # Fixed points, converted to °C and MPa
        Tc = CP.PropsSI('Tcrit', fluid) - 273.15
        Pc = CP.PropsSI('pcrit', fluid) / 1e6
        Tt = CP.PropsSI('Ttriple', fluid) - 273.15
        Pt = CP.PropsSI('ptriple', fluid) / 1e6
    
        # Saturation curve (vapour-liquid boundary)
        T_sat = np.linspace(Tt + 1, Tc - 0.1, 100)  # °C
        P_sat = []
    
        for T in T_sat:
            try:
                P_sat.append(CP.PropsSI('P', 'T', T + 273.15, 'Q', 0, fluid) / 1e6)
            except ValueError:
                P_sat.append(np.nan)
    
        fig, ax = plt.subplots(figsize=(10, 7))
    
        ax.plot(T_sat, P_sat, 'b-', linewidth=2, label='Vapour-liquid boundary')
        ax.plot(Tc, Pc, 'ro', markersize=12,
                label=f'Critical point ({Tc:.1f} °C, {Pc:.2f} MPa)')
        ax.plot(Tt, Pt, 'go', markersize=10,
                label=f'Triple point ({Tt:.1f} °C, {Pt:.4f} MPa)')
    
        # Shade the supercritical region
        T_shade = np.linspace(Tc, Tc + 100, 50)
        ax.fill_between(T_shade, np.full_like(T_shade, Pc),
                        np.full_like(T_shade, Pc * 3),
                        alpha=0.2, color='red', label='Supercritical region')
    
        # Phase labels
        ax.text(Tt - 20, Pt / 2, 'SOLID', fontsize=14, ha='center', style='italic')
        ax.text(Tc - 20, Pc / 2, 'LIQUID', fontsize=14, ha='center', style='italic')
        ax.text(Tc - 20, Pc / 10, 'GAS', fontsize=14, ha='center', style='italic')
        ax.text(Tc + 30, Pc * 1.5, 'SUPERCRITICAL\nFLUID', fontsize=14, ha='center',
                style='italic', color='red', weight='bold')
    
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
            print(f"Phase diagram saved to '{save_path}'")
    
        plt.show()
    
    # Generate phase diagrams for common SCF solvents
    generate_phase_diagram('CO2', 'CO2_phase_diagram.png')
    # generate_phase_diagram('Water', 'H2O_phase_diagram.png')
    # generate_phase_diagram('Ethanol', 'ethanol_phase_diagram.png')

**What this diagram does not show.** CoolProp gives you the vapour-liquid boundary directly, but not the sublimation and melting curves — those require solid-phase data that reference Helmholtz equations do not carry. For a complete CO₂ phase diagram the solid boundaries must be added from a separate correlation. The melting curve of CO₂ is famously steep, of order $dP/dT \sim 10^7$ Pa/K near the triple point, so it appears almost vertical on any linear pressure axis.

### Example 4: P-V Isotherms Across the Phase Transition

The $P$-$V$ diagram is where the two-phase region becomes visible: below $T_c$ the isotherm is interrupted by a horizontal tie line joining saturated liquid to saturated vapour; above $T_c$ it is continuous.

Code Example 4: P-V Isotherms for CO₂
    
    
    import CoolProp.CoolProp as CP
    import matplotlib.pyplot as plt
    import numpy as np
    
    def generate_pv_diagram(fluid, temperatures_c, save_path=None):
        """
        Generate a P-V diagram showing isotherms across the phase transition.
    
        Parameters:
        -----------
        fluid : str
            Fluid name
        temperatures_c : list
            Temperatures in °C
        save_path : str, optional
            Path to save the figure
        """
        Tc = CP.PropsSI('Tcrit', fluid) - 273.15
        Pc = CP.PropsSI('pcrit', fluid) / 1e6
    
        fig, ax = plt.subplots(figsize=(10, 7))
    
        for T_c in temperatures_c:
            T_k = T_c + 273.15
    
            if T_c < Tc - 0.5:
                # Below critical: the isotherm crosses the two-phase region
                P_sat = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 1e6
    
                # Compressed liquid branch
                P_liquid = np.linspace(P_sat + 0.1, Pc * 2, 50)
                V_liquid = [1 / CP.PropsSI('D', 'P', P * 1e6, 'T', T_k, fluid)
                            for P in P_liquid]
    
                # Superheated vapour branch
                P_vapor = np.linspace(0.1, P_sat - 0.01, 50)
                V_vapor = [1 / CP.PropsSI('D', 'P', P * 1e6, 'T', T_k, fluid)
                           for P in P_vapor]
    
                # Saturated end points of the tie line
                V_liq_sat = 1 / CP.PropsSI('D', 'T', T_k, 'Q', 0, fluid)
                V_vap_sat = 1 / CP.PropsSI('D', 'T', T_k, 'Q', 1, fluid)
    
                ax.plot(V_liquid, P_liquid, 'b-', alpha=0.7)
                ax.plot(V_vapor, P_vapor, 'b-', alpha=0.7,
                        label=f'{T_c:.0f} °C (T below Tc)')
                ax.plot([V_liq_sat, V_vap_sat], [P_sat, P_sat], 'b--', alpha=0.7)
            else:
                # At or above critical: a single continuous branch
                P_range = np.linspace(0.1, Pc * 2, 100)
                V_range = [1 / CP.PropsSI('D', 'P', P * 1e6, 'T', T_k, fluid)
                           for P in P_range]
    
                label = (f'{T_c:.1f} °C (T = Tc)' if abs(T_c - Tc) < 0.5
                         else f'{T_c:.0f} °C (T above Tc)')
                ax.plot(V_range, P_range, 'r-', linewidth=2, alpha=0.8, label=label)
    
        ax.set_xlabel('Specific Volume (m³/kg)', fontsize=12)
        ax.set_ylabel('Pressure (MPa)', fontsize=12)
        ax.set_title(f'P-V Diagram: {fluid} (Isotherms)', fontsize=14, weight='bold')
        ax.set_xscale('log')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"P-V diagram saved to '{save_path}'")
    
        plt.show()
    
    Tc_CO2 = CP.PropsSI('Tcrit', 'CO2') - 273.15
    generate_pv_diagram('CO2', [20, 30, Tc_CO2, 40, 60, 100], 'CO2_pv_diagram.png')

**A practical variant.** Replacing specific volume with density on the abscissa gives the $P$-$\rho$ diagram, which is usually the more useful plot for extraction work, because solubility correlates with density (Chapter 6, Chrastil) rather than with volume. The change is a one-line edit: plot `CP.PropsSI('D', ...)` directly instead of its reciprocal, and drop the logarithmic scale.

* * *

## 7.4 Implementing Equations of State

Chapter 6 derived the van der Waals and Peng-Robinson equations. Implementing them yourself is worth the effort for two reasons: you need custom EOS code whenever a compound is missing from the reference libraries, and benchmarking your implementation against CoolProp is the fastest way to develop a feel for where cubic equations fail.

### Example 5: van der Waals Solver Benchmarked Against CoolProp

Code Example 5: van der Waals EOS Class
    
    
    import numpy as np
    from CoolProp.CoolProp import PropsSI
    
    class VanDerWaalsEOS:
        """van der Waals equation of state."""
    
        def __init__(self, Tc, Pc):
            """
            Parameters
            ----------
            Tc : float
                Critical temperature (K)
            Pc : float
                Critical pressure (Pa)
            """
            self.Tc = Tc
            self.Pc = Pc
            self.R = 8.314  # J/(mol·K)
    
            # van der Waals parameters from the critical constants
            self.a = 27 * (self.R * Tc)**2 / (64 * Pc)  # Pa·m⁶/mol²
            self.b = self.R * Tc / (8 * Pc)             # m³/mol
    
        def pressure(self, V, T):
            """
            Pressure from molar volume and temperature.
    
            Parameters
            ----------
            V : float
                Molar volume (m³/mol)
            T : float
                Temperature (K)
    
            Returns
            -------
            float
                Pressure (Pa)
            """
            return self.R * T / (V - self.b) - self.a / V**2
    
        def molar_volume(self, P, T):
            """
            Molar volume: real positive roots of the cubic form.
    
            V³ - (b + RT/P)V² + (a/P)V - ab/P = 0
    
            Returns
            -------
            array
                Real positive roots (m³/mol), sorted ascending
            """
            coeffs = [
                1,
                -(self.b + self.R * T / P),
                self.a / P,
                -self.a * self.b / P
            ]
    
            roots = np.roots(coeffs)
            real_positive = roots[(np.isreal(roots)) & (roots.real > 0)].real
    
            return np.sort(real_positive)
    
    # Test with CO2
    Tc_co2 = PropsSI('Tcrit', 'CO2')
    Pc_co2 = PropsSI('pcrit', 'CO2')
    M_co2 = PropsSI('molar_mass', 'CO2')  # kg/mol
    
    vdw = VanDerWaalsEOS(Tc_co2, Pc_co2)
    
    # Supercritical conditions
    T = 313.15   # 40 °C
    P = 100e5    # 100 bar
    
    V_roots = vdw.molar_volume(P, T)
    
    # Reference value: molar volume = molar mass / mass density
    V_coolprop = M_co2 / PropsSI('D', 'T', T, 'P', P, 'CO2')
    
    print(f"van der Waals at T = {T - 273.15:.1f} °C, P = {P / 1e5:.0f} bar")
    print(f"  Real positive roots: {np.round(V_roots * 1e6, 2)} cm³/mol")
    print(f"  Selected (largest):  {V_roots[-1] * 1e6:.2f} cm³/mol")
    print(f"  CoolProp reference:  {V_coolprop * 1e6:.2f} cm³/mol")
    print(f"  Relative error:      "
          f"{abs(V_roots[-1] - V_coolprop) / V_coolprop * 100:.1f}%")

van der Waals at T = 40.0 °C, P = 100 bar Real positive roots: [90.44] cm³/mol Selected (largest): 90.44 cm³/mol CoolProp reference: 70.01 cm³/mol Relative error: 29.2%

A 29% error in molar volume at a very ordinary extraction condition. Note the unit handling in the reference line: CoolProp's `'D'` is a _mass_ density in kg/m³, so converting to molar volume requires dividing the molar mass by it. Mixing molar and mass bases is the most common source of thousand-fold errors in EOS validation code.

### Example 6: Peng-Robinson Solver Benchmarked Against CoolProp

Code Example 6: Peng-Robinson EOS Class and Validation
    
    
    import numpy as np
    import CoolProp.CoolProp as CP
    
    class PengRobinsonEOS:
        """
        Peng-Robinson equation of state.
    
        P = RT/(V-b) - a(T)/(V(V+b) + b(V-b))
        """
    
        def __init__(self, Tc, Pc, omega):
            """
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
            self.R = 8.314  # J/(mol·K)
    
            self.a_c = 0.45724 * (self.R * Tc)**2 / Pc
            self.b = 0.07780 * self.R * Tc / Pc
    
            # Kappa: standard correlation, with the extended form for
            # heavy molecules (omega > 0.49)
            if omega <= 0.49:
                self.kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
            else:
                self.kappa = (0.379642 + 1.48503*omega
                              - 0.164423*omega**2 + 0.016666*omega**3)
    
        def alpha(self, T):
            """Temperature-dependent alpha parameter."""
            Tr = T / self.Tc
            return (1 + self.kappa * (1 - np.sqrt(Tr)))**2
    
        def a(self, T):
            """Temperature-dependent attraction parameter."""
            return self.a_c * self.alpha(T)
    
        def calculate_Z(self, P, T):
            """
            Compressibility factor(s) from the cubic form.
    
            Returns:
            --------
            array : real positive roots (1 or 3 of them), sorted ascending
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # Z^3 + c2 Z^2 + c1 Z + c0 = 0
            c2 = -(1 - B)
            c1 = A - 3*B**2 - 2*B
            c0 = -(A*B - B**2 - B**3)
    
            roots = np.roots([1, c2, c1, c0])
    
            real_roots = roots[np.isreal(roots)].real
            return np.sort(real_roots[real_roots > 0])
    
        def calculate_density(self, P, T, phase='vapor'):
            """
            Molar density (mol/m³).
    
            Parameters:
            -----------
            phase : str
                'liquid' (smallest root) or 'vapor' (largest root)
            """
            Z_roots = self.calculate_Z(P, T)
    
            if len(Z_roots) == 0:
                raise ValueError("No valid compressibility factor found")
    
            Z = Z_roots[0] if phase == 'liquid' else Z_roots[-1]
    
            V = Z * self.R * T / P  # m³/mol
            return 1 / V            # mol/m³
    
        def calculate_pressure(self, V, T):
            """Pressure from molar volume and temperature."""
            a_T = self.a(T)
            return (self.R * T / (V - self.b)
                    - a_T / (V*(V + self.b) + self.b*(V - self.b)))
    
    # CO2 properties (NIST values)
    CO2_Tc = 304.13     # K
    CO2_Pc = 7.3773e6   # Pa
    CO2_omega = 0.22394
    M_CO2 = 44.01e-3    # kg/mol
    
    pr = PengRobinsonEOS(CO2_Tc, CO2_Pc, CO2_omega)
    
    print("=== Peng-Robinson vs CoolProp reference EOS (CO2) ===")
    print(f"{'T (degC)':>9} {'P (MPa)':>9} {'PR (kg/m3)':>12} "
          f"{'CoolProp':>10} {'Error':>8}")
    
    for T_c, P_mpa in [(35, 10), (50, 10), (50, 20), (75, 15), (100, 25)]:
        T = T_c + 273.15
        P = P_mpa * 1e6
    
        # molar density (mol/m3) * molar mass (kg/mol) = mass density (kg/m3)
        rho_pr = pr.calculate_density(P, T) * M_CO2
        rho_ref = CP.PropsSI('D', 'P', P, 'T', T, 'CO2')
        err = (rho_pr - rho_ref) / rho_ref * 100
    
        print(f"{T_c:>9} {P_mpa:>9} {rho_pr:>12.1f} {rho_ref:>10.1f} {err:>7.1f}%")

=== Peng-Robinson vs CoolProp reference EOS (CO2) === T (degC) P (MPa) PR (kg/m3) CoolProp Error 35 10 651.6 712.8 -8.6% 50 10 375.3 384.3 -2.3% 50 20 762.9 784.3 -2.7% 75 15 440.8 463.3 -4.9% 100 25 565.1 588.5 -4.0%

**Peng-Robinson earns its reputation.** Errors of 2-9% against a reference equation, versus 29% for van der Waals at a comparable state — and the worst case (−8.6%) is at 35 °C, only 4 K above $T_c$, exactly where cubic equations are known to struggle. Away from the immediate critical region the error settles at 2-5%. For a three-parameter equation that costs one cubic root, that is remarkable value; it is also why you should still reach for CoolProp when the number matters.

* * *

## 7.5 Process Simulation

### Example 7: Multi-Stage Extraction Cascade

A staged extraction model with an equilibrium partition coefficient at each stage. The key dimensionless group is the **extraction factor** $E = K \cdot (S/F)$, the product of the partition coefficient and the solvent-to-feed ratio.

Code Example 7: Multi-Stage Extraction Simulator
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class MultiStageExtractor:
        """
        Simulate multi-stage supercritical fluid extraction.
    
        Assumptions:
        - Equilibrium is reached at each stage
        - Constant temperature and pressure
        - Ideal mixing
        - Fresh solvent is supplied to each stage (a staged-batch cascade).
          A true countercurrent contactor requires iterating the stage
          balances to convergence; this explicit form is the standard
          first estimate.
        """
    
        def __init__(self, n_stages, K_partition, S_F_ratio, solute_feed):
            """
            Parameters:
            -----------
            n_stages : int
                Number of extraction stages
            K_partition : float
                Partition coefficient (mass fraction in SCF / mass fraction in feed)
            S_F_ratio : float
                Solvent-to-feed mass ratio
            solute_feed : float
                Initial solute concentration in the feed (kg solute / kg feed)
            """
            self.n_stages = n_stages
            self.K = K_partition
            self.S_F = S_F_ratio
            self.C_feed = solute_feed
    
        def solve_cascade(self):
            """
            Solve the stage-by-stage material balance.
    
            Returns:
            --------
            dict : concentration profiles and extraction efficiency
            """
            C_feed_stage = np.zeros(self.n_stages + 1)     # feed-phase concentration
            C_solvent_stage = np.zeros(self.n_stages + 1)  # solvent-phase concentration
    
            # Boundary conditions
            C_feed_stage[0] = self.C_feed  # fresh feed enters at stage 0
            C_solvent_stage[self.n_stages] = 0  # clean solvent
    
            E = self.K * self.S_F  # extraction factor
    
            for stage in range(self.n_stages):
                # Material balance with equilibrium C_solvent = K * C_feed
                C_feed_stage[stage + 1] = (
                    (C_feed_stage[stage] + self.S_F * C_solvent_stage[stage + 1])
                    / (1 + E)
                )
                C_solvent_stage[stage] = self.K * C_feed_stage[stage + 1]
    
            # Overall recovery, from the closed material balance
            solute_extracted = self.S_F * np.sum(C_solvent_stage[:self.n_stages])
            efficiency = solute_extracted / self.C_feed * 100
    
            return {
                'C_feed': C_feed_stage,
                'C_solvent': C_solvent_stage,
                'efficiency': efficiency,
                'solute_remaining': C_feed_stage[-1],
                'extraction_factor': E
            }
    
        def plot_concentration_profile(self, results):
            """Visualize the concentration profiles along the cascade."""
            stages = np.arange(self.n_stages + 1)
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
            ax1.plot(stages, results['C_feed'], 'o-', label='Feed phase',
                     linewidth=2, markersize=8)
            ax1.plot(stages, results['C_solvent'], 's-', label='Solvent phase',
                     linewidth=2, markersize=8)
            ax1.set_xlabel('Stage number', fontsize=12)
            ax1.set_ylabel('Concentration (kg/kg)', fontsize=12)
            ax1.set_title('Concentration profile in multi-stage extraction',
                          fontsize=13, weight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
    
            cumulative = np.zeros(self.n_stages + 1)
            for i in range(1, self.n_stages + 1):
                cumulative[i] = (1 - results['C_feed'][i] / self.C_feed) * 100
    
            ax2.plot(stages, cumulative, 'o-', color='green',
                     linewidth=2, markersize=8)
            ax2.set_xlabel('Stage number', fontsize=12)
            ax2.set_ylabel('Cumulative extraction (%)', fontsize=12)
            ax2.set_title('Extraction efficiency vs number of stages',
                          fontsize=13, weight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 105)
    
            plt.tight_layout()
            plt.show()
    
    # Caffeine extraction from coffee beans with scCO2.
    # K = 0.05: caffeine is far more concentrated in the wet bean matrix
    # than in the CO2 phase, which is why a large solvent ratio is needed.
    extractor = MultiStageExtractor(
        n_stages=5,
        K_partition=0.05,
        S_F_ratio=20,     # 20 kg CO2 per kg coffee
        solute_feed=0.02  # 2 wt% caffeine in the beans
    )
    
    results = extractor.solve_cascade()
    
    print("=== Multi-Stage Extraction Simulation ===")
    print(f"Number of stages: {extractor.n_stages}")
    print(f"Extraction factor (E = K x S/F): {results['extraction_factor']:.2f}")
    print(f"Overall extraction efficiency: {results['efficiency']:.2f}%")
    print(f"Caffeine remaining in feed: {results['solute_remaining']*100:.4f} wt%")
    print("\nStage-by-stage concentrations:")
    print(f"{'Stage':<8} {'Feed (kg/kg)':<15} {'Solvent (kg/kg)':<15}")
    print("-" * 40)
    for i in range(extractor.n_stages + 1):
        print(f"{i:<8} {results['C_feed'][i]:<15.6f} "
              f"{results['C_solvent'][i]:<15.6f}")
    
    extractor.plot_concentration_profile(results)

=== Multi-Stage Extraction Simulation === Number of stages: 5 Extraction factor (E = K x S/F): 1.00 Overall extraction efficiency: 96.88% Caffeine remaining in feed: 0.0625 wt% Stage-by-stage concentrations: Stage Feed (kg/kg) Solvent (kg/kg) \---------------------------------------- 0 0.020000 0.000500 1 0.010000 0.000250 2 0.005000 0.000125 3 0.002500 0.000063 

**Why $E$ is the number that matters.** With $E = 1$ each stage removes exactly half of the remaining solute, so the feed concentration halves stage by stage and the recovery after $n$ stages is $1 - (1+E)^{-n}$ — 96.9% here. Doubling the solvent ratio to $S/F = 40$ gives $E = 2$ and 99.6% in the same five stages; halving it to $S/F = 10$ gives $E = 0.5$ and only 86.8%. Stages and solvent ratio are interchangeable to a degree, and the economic optimum is where the cost of one more vessel equals the cost of the extra CO₂ recycle duty.

### Example 8: Extraction Yield and CO₂ Consumption

A process-level model: Chrastil solubility sets the thermodynamic ceiling, a first-order rate constant sets how close you get to it in the available time, and the result is expressed as the metric that decides plant economics — kilograms of CO₂ per kilogram of product.

Code Example 8: scCO₂ Extraction Process Model
    
    
    import numpy as np
    from CoolProp.CoolProp import PropsSI
    
    class SCF_ExtractionModel:
        """Simplified supercritical fluid extraction model."""
    
        def __init__(self, solute_mw, chrastil_params):
            """
            Parameters
            ----------
            solute_mw : float
                Solute molar mass (g/mol)
            chrastil_params : dict
                Chrastil parameters {'k': ..., 'a': ..., 'b': ...}
                for S = (rho/1000)**k * exp(a/T + b) with S in kg/kg
            """
            self.solute_mw = solute_mw
            self.k = chrastil_params['k']
            self.a = chrastil_params['a']
            self.b = chrastil_params['b']
    
        def solubility_chrastil(self, T, rho):
            """
            Chrastil solubility.
    
            S = (rho/1000)**k * exp(a/T + b)
    
            Parameters
            ----------
            T : float
                Temperature (K)
            rho : float
                CO2 density (kg/m³)
    
            Returns
            -------
            float
                Solubility (kg solute / kg CO2)
            """
            return (rho / 1000)**self.k * np.exp(self.a / T + self.b)
    
        def extraction_yield(self, T_celsius, P_bar, flow_rate_co2, time_hours,
                             solute_mass_initial, k_rate=0.5):
            """
            Simulate the extraction yield.
    
            Parameters
            ----------
            T_celsius : float
                Temperature (°C)
            P_bar : float
                Pressure (bar)
            flow_rate_co2 : float
                CO2 mass flow rate (kg/h)
            time_hours : float
                Extraction time (h)
            solute_mass_initial : float
                Initial solute charge (kg)
            k_rate : float
                First-order extraction rate constant (1/h), fitted experimentally
    
            Returns
            -------
            dict
                Extraction results
            """
            T = T_celsius + 273.15
            P = P_bar * 1e5
    
            # CO2 density from the reference EOS
            rho = PropsSI('D', 'T', T, 'P', P, 'CO2')
    
            # Equilibrium solubility
            S = self.solubility_chrastil(T, rho)
    
            # Total CO2 consumed
            total_co2 = flow_rate_co2 * time_hours  # kg
    
            # Thermodynamic ceiling: all CO2 leaves saturated
            max_extracted = S * total_co2  # kg
    
            # Kinetic limitation: E(t) = E_max * (1 - exp(-k t)),
            # capped by the solute actually present
            actual_extracted = min(
                max_extracted * (1 - np.exp(-k_rate * time_hours)),
                solute_mass_initial
            )
    
            yield_percent = actual_extracted / solute_mass_initial * 100
    
            return {
                'CO2 density [kg/m3]': rho,
                'Solubility [g/kg-CO2]': S * 1000,
                'Total CO2 used [kg]': total_co2,
                'Extracted [kg]': actual_extracted,
                'Yield [%]': yield_percent,
                'CO2 intensity [kg-CO2/kg-product]': (
                    total_co2 / actual_extracted if actual_extracted > 0 else np.inf
                )
            }
    
    # Caffeine. NOTE: k and a are typical literature magnitudes; b has been
    # calibrated so that S(50 °C, 200 bar) is about 1 g per kg CO2, the order
    # of magnitude reported for caffeine in scCO2. These are illustrative
    # parameters, not a citable parameter set - refit to your own data.
    caffeine_params = {'k': 8.0, 'a': -5000, 'b': 10.51}
    extractor = SCF_ExtractionModel(solute_mw=194.19,
                                    chrastil_params=caffeine_params)
    
    conditions = {
        'T_celsius': 50,
        'P_bar': 200,
        'flow_rate_co2': 10,       # kg/h
        'time_hours': 3,
        'solute_mass_initial': 0.5  # kg
    }
    
    result = extractor.extraction_yield(**conditions)
    
    print("=== scCO2 Extraction Simulation ===")
    for key, value in result.items():
        print(f"  {key}: {value:.2f}")

=== scCO2 Extraction Simulation === CO2 density [kg/m3]: 784.29 Solubility [g/kg-CO2]: 1.00 Total CO2 used [kg]: 30.00 Extracted [kg]: 0.02 Yield [%]: 4.67 CO2 intensity [kg-CO2/kg-product]: 1285.58

**Read the CO₂ intensity, then look at your flowsheet again.** 1286 kg of CO₂ per kilogram of caffeine extracted, and only 4.7% of the charge recovered in three hours. Both numbers follow directly from a solubility of 1 g/kg: dilute solutions demand enormous solvent throughput. This is exactly why every industrial supercritical plant runs a closed CO₂ loop with recompression rather than once-through solvent, and why the compressor, not the extractor vessel, usually dominates both capital and operating cost.

The Chrastil parameters here are calibrated for illustration, not cited from a paper — see the comment in the code. Fit your own, as in Example 10.

### Example 9: RESS Particle Size Estimation

In RESS (Rapid Expansion of Supercritical Solutions) a saturated supercritical solution is expanded through a fine nozzle; the solvent power collapses in microseconds and the solute nucleates as fine particles. A Weber-number scaling gives an order-of-magnitude size estimate.

Code Example 9: RESS Particle Size from Nozzle Conditions
    
    
    import numpy as np
    from CoolProp.CoolProp import PropsSI
    
    def ress_particle_size_model(T_celsius, P_bar, nozzle_diameter_mm,
                                 expansion_ratio, sigma=0.01):
        """
        Order-of-magnitude particle size for RESS, from a Weber-number scaling.
    
        Parameters
        ----------
        T_celsius : float
            Upstream temperature (°C)
        P_bar : float
            Upstream pressure (bar)
        nozzle_diameter_mm : float
            Nozzle diameter (mm)
        expansion_ratio : float
            P_upstream / P_downstream
        sigma : float
            Effective surface tension (N/m). Solute dependent; 0.01 N/m
            is a typical order of magnitude.
    
        Returns
        -------
        dict
            Predicted nozzle exit velocity, Weber number and particle size
        """
        T = T_celsius + 273.15
        P_upstream = P_bar * 1e5
    
        # Upstream properties
        rho_upstream = PropsSI('D', 'T', T, 'P', P_upstream, 'CO2')
        a = PropsSI('A', 'T', T, 'P', P_upstream, 'CO2')  # speed of sound
    
        # Heat capacity ratio
        gamma = (PropsSI('C', 'T', T, 'P', P_upstream, 'CO2')
                 / PropsSI('O', 'T', T, 'P', P_upstream, 'CO2'))
    
        # Nozzle exit velocity, isentropic expansion approximation:
        # v = sqrt(2 dh) with dh ~ a^2 ln(P1/P2) / gamma
        v_exit = np.sqrt(2 * a**2 * np.log(expansion_ratio) / gamma)
    
        # Weber number We = rho v^2 d / sigma
        d_nozzle = nozzle_diameter_mm * 1e-3  # m
        We = rho_upstream * v_exit**2 * d_nozzle / sigma
    
        # Empirical scaling: d_particle ~ d_nozzle / sqrt(We)
        d_particle = d_nozzle / np.sqrt(We)
    
        return {
            'CO2 density [kg/m3]': rho_upstream,
            'Speed of sound [m/s]': a,
            'Heat capacity ratio [-]': gamma,
            'Exit velocity [m/s]': v_exit,
            'Weber number [-]': We,
            'Predicted particle size [nm]': d_particle * 1e9,
        }
    
    ress_result = ress_particle_size_model(
        T_celsius=60,
        P_bar=150,
        nozzle_diameter_mm=0.1,
        expansion_ratio=150
    )
    
    print("=== RESS Particle Size Prediction ===")
    for key, value in ress_result.items():
        print(f"  {key}: {value:.4g}")

=== RESS Particle Size Prediction === CO2 density [kg/m3]: 604.1 Speed of sound [m/s]: 309.2 Heat capacity ratio [-]: 3.582 Exit velocity [m/s]: 517.2 Weber number [-]: 1.616e+06 Predicted particle size [nm]: 78.67

**Treat this as a scaling law, not a prediction.** Two things drive the answer: the exit velocity (517 m/s here — the expansion is supersonic, which is why RESS quenches so fast) and the assumed surface tension. Because $d_p \propto \sigma^{1/2}$, an order-of-magnitude uncertainty in $\sigma$ moves the predicted size by a factor of three. The model has no nucleation kinetics, no coagulation and no solute properties beyond $\sigma$, so read the answer as "tens to hundreds of nanometres", which is the right regime for RESS, and calibrate against your own measurements before using it for design.

Note also the heat capacity ratio of 3.58 — far from the 1.3 of an ideal diatomic gas, because $c_p$ is still strongly elevated at 60 °C and 150 bar. Substituting an ideal-gas $\gamma$ here would give a 60% error in exit velocity.

* * *

## 7.6 Data Analysis and Model Fitting

### Example 10: Fitting the Chrastil Equation to Experimental Data

The realistic workflow: measured solubilities at several temperatures and pressures, CoolProp for the densities, and `curve_fit` for the three Chrastil parameters — with their uncertainties, which matter more than the point estimates.

Code Example 10: Chrastil Regression with Parameter Uncertainties
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from CoolProp.CoolProp import PropsSI
    
    # Experimental data: naphthalene solubility in scCO2 (illustrative values)
    experimental_data = {
        'T': [308.15, 308.15, 308.15, 318.15, 318.15, 318.15],  # K
        'P': [90e5, 120e5, 150e5, 90e5, 120e5, 150e5],          # Pa
        'S': [2.1e-3, 3.5e-3, 4.8e-3, 1.8e-3, 3.0e-3, 4.2e-3]   # kg/kg
    }
    
    def chrastil_model(X, k, a, b):
        """
        Chrastil equation: S = (rho/1000)**k * exp(a/T + b)
    
        Parameters
        ----------
        X : tuple of arrays
            (T, P) - temperature (K) and pressure (Pa)
        k, a, b : float
            Fitting parameters
    
        Returns
        -------
        array
            Solubility (kg/kg)
        """
        T, P = X
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        rho = np.array([PropsSI('D', 'T', t, 'P', p, 'CO2') for t, p in zip(T, P)])
        return (rho / 1000)**k * np.exp(a / T + b)
    
    T_data = np.array(experimental_data['T'])
    P_data = np.array(experimental_data['P'])
    S_data = np.array(experimental_data['S'])
    
    # CO2 densities at the experimental states
    rho_data = np.array([PropsSI('D', 'T', t, 'P', p, 'CO2')
                         for t, p in zip(T_data, P_data)])
    print("CO2 densities at the experimental states (kg/m3):")
    print(np.round(rho_data, 1))
    
    # Nonlinear least squares
    params, covariance = curve_fit(
        chrastil_model, (T_data, P_data), S_data,
        p0=[8.0, -5000, 15], maxfev=10000
    )
    
    k_fit, a_fit, b_fit = params
    errors = np.sqrt(np.diag(covariance))
    
    print("\nChrastil fit:")
    print(f"  k = {k_fit:.3f} +/- {errors[0]:.3f}")
    print(f"  a = {a_fit:.1f} +/- {errors[1]:.1f} K")
    print(f"  b = {b_fit:.3f} +/- {errors[2]:.3f}")
    
    # Model validation
    S_predicted = chrastil_model((T_data, P_data), *params)
    rmse = np.sqrt(np.mean((S_data - S_predicted)**2))
    r_squared = (1 - np.sum((S_data - S_predicted)**2)
                 / np.sum((S_data - np.mean(S_data))**2))
    
    print("\nModel accuracy:")
    print(f"  RMSE: {rmse*1000:.4f} g/kg")
    print(f"  R²:   {r_squared:.4f}")
    
    # Parity plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(S_data * 1000, S_predicted * 1000, s=100, alpha=0.7,
               edgecolors='black')
    limit = max(S_data) * 1000 * 1.1
    ax.plot([0, limit], [0, limit], 'r--', label='Parity')
    ax.set_xlabel('Experimental (g/kg-CO₂)', fontsize=12)
    ax.set_ylabel('Predicted (g/kg-CO₂)', fontsize=12)
    ax.set_title(f'Chrastil fit (R² = {r_squared:.3f})', fontsize=14,
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chrastil_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()

CO2 densities at the experimental states (kg/m3): [662.1 767.1 815.1 337.5 657.7 742. ] Chrastil fit: k = 1.753 +/- 0.813 a = -1650.0 +/- 2032.2 K b = 0.218 +/- 6.614 Model accuracy: RMSE: 0.5459 g/kg R²: 0.7391

#### This is what an honest fit looks like

$R^2 = 0.74$ and $k = 1.75 \pm 0.81$ — a 46% relative uncertainty on the association number, and $a$ whose error bar is larger than its value. Three lessons follow, and they generalize to almost every solubility correlation you will meet in the literature:

  * **Six points cannot constrain three parameters.** The covariance matrix says so plainly. Published parameter sets quoted without uncertainties should be treated with suspicion.
  * **The near-critical point is doing the damage.** At 318.15 K and 90 bar the CO₂ density is only 338 kg/m³, less than half the 662 kg/m³ at 308.15 K and the same pressure. A single $k$ cannot describe both the dense and the near-critical regime, which is exactly where Chrastil's single-association-number assumption breaks down.
  * **Design your experiment for the fit.** Sampling at least four pressures across three temperatures, and staying above $T_r \approx 1.05$ where density is less violently pressure-sensitive, typically brings $R^2$ above 0.98 and the uncertainty on $k$ below 10%.

* * *

## 7.7 Beyond Property Calculations

### Mixtures and Co-Solvents

Adding a polar entrainer such as ethanol is the standard way to extend scCO₂ to polar solutes (Chapter 2). CoolProp handles mixtures through its Helmholtz (HEOS) backend with an explicit composition string:

Mixture density with the HEOS backend
    
    
    from CoolProp.CoolProp import PropsSI
    
    T, P = 323.15, 150e5  # 50 degC, 150 bar
    
    rho_pure = PropsSI('D', 'T', T, 'P', P, 'CO2')
    print(f"Pure CO2:              {rho_pure:.1f} kg/m³")
    
    for partner in ['Nitrogen', 'Methane', 'Ethane', 'Propane', 'Ethanol']:
        mixture = f'HEOS::CO2[0.9]&{partner}[0.1]'
        try:
            rho = PropsSI('D', 'T', T, 'P', P, mixture)
            print(f"CO2 + 10 mol% {partner:<9}: {rho:.1f} kg/m³")
        except ValueError as e:
            print(f"CO2 + 10 mol% {partner:<9}: unavailable "
                  f"({str(e).split(';')[-1].strip()[:48]}...)")

Pure CO2: 699.8 kg/m³ CO2 + 10 mol% Nitrogen : 516.8 kg/m³ CO2 + 10 mol% Methane : 546.0 kg/m³ CO2 + 10 mol% Ethane : 617.7 kg/m³ CO2 + 10 mol% Propane : 656.7 kg/m³ CO2 + 10 mol% Ethanol : unavailable (error: Could not match the binary pair [124-38-9...)

**The most useful pair is the one that is missing.** HEOS mixture calculations need fitted binary interaction parameters, and CoolProp simply refuses when it has none — as it does for CO₂ + ethanol, the single most important entrainer system in supercritical extraction. This is not a bug but an honest admission that no reference-quality mixture model exists for that pair.

When you hit this wall, use `thermo` with a cubic EOS and an explicit $k_{ij}$ (Chapter 6 gives $k_{ij} \approx 0.10$ for CO₂-ethanol), accept cubic-EOS accuracy, and validate against whatever binary VLE data you can find. Note also how strongly the light gases dilute the mixture: 10 mol% nitrogen drops the density from 700 to 517 kg/m³, which is why dissolved air or incomplete purging silently destroys extraction yield.

### Molecular Dynamics

For molecular-level questions — local density enhancement around a solute, cluster lifetimes, diffusion mechanisms — molecular dynamics is the tool of choice.

  * **LAMMPS** : general-purpose; the TraPPE force field is the standard choice for CO₂
  * **GROMACS** : biomolecular systems, supercritical drying of biological samples
  * **NAMD** : large systems

Typical MD tooling
    
    
    conda install -c conda-forge lammps   # simulation engine
    pip install MDAnalysis                # trajectory analysis

Practical guidance: use TraPPE for CO₂ and OPLS for organic solutes, run the NPT ensemble for constant-pressure states, apply periodic boundary conditions, and allow 1-5 ns of equilibration — near the critical point, correlation times are long and under-equilibrated runs are the most common source of wrong answers.

### Machine Learning for Property Prediction

When no EOS covers your solute, a data-driven surrogate can interpolate a measured dataset. The pattern is standard supervised regression on $(T, P, \rho, \text{molecular descriptors}) \rightarrow S$:

Surrogate model sketch (scikit-learn)
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Features: temperature, pressure, density (add molecular descriptors
    # such as molar mass, logP or polar surface area for multi-solute models)
    X = np.column_stack([T_data, P_data, rho_data])
    y = S_data
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Surrogate R²: {r2_score(y_test, y_pred):.4f}")

**A surrogate cannot extrapolate.** Tree ensembles predict a constant outside the training envelope, so a random forest fitted to 10-20 MPa data will happily return a plausible-looking number at 40 MPa that carries no physics at all. Useful libraries for descriptor generation are `RDKit`, `DeepChem` and `ChemML`; useful discipline is to always report the training domain alongside the model, and to prefer a physically-grounded correlation such as Chrastil whenever the data are sparse.

### Process Simulators

For plant-scale flowsheets:

  * **Aspen Plus / HYSYS** : drive from Python over COM (`pywin32`); the standard choice for industrial supercritical extraction and sCO₂ power cycle design
  * **DWSIM** : open-source, with a built-in Python scripting interface
  * **Cantera** : chemical kinetics and thermodynamics, useful for supercritical water oxidation modelling

The property models in these packages are usually cubic EOS by default. Given the errors quantified in Example 6, check whether your simulator can be pointed at a reference EOS (REFPROP or CoolProp) for the solvent before trusting a near-critical flowsheet.

* * *

## Summary

### Key Takeaways

**1\. Property calculations**

  * CoolProp's `PropsSI` gives reference-quality properties from two state variables; everything is SI.
  * A property table over the operating window exposes the density cliff and the $c_p$ divergence immediately.

**2\. Phase diagrams**

  * $P$-$T$ diagrams come from the saturation curve plus the fixed points; solid boundaries need separate data.
  * $P$-$V$ isotherms make the two-phase tie line visible; $P$-$\rho$ is more useful for extraction work.

**3\. Custom equations of state**

  * van der Waals: 29% density error at ordinary extraction conditions.
  * Peng-Robinson: 2-9%, worst in the immediate near-critical region.
  * Always benchmark a new EOS implementation against a reference EOS before using it.

**4\. Process models**

  * The extraction factor $E = K \cdot (S/F)$ controls staged recovery: $1 - (1+E)^{-n}$.
  * Low solubility translates directly into high CO₂ intensity, which is why plants recycle the solvent.
  * RESS particle size follows a Weber-number scaling; treat it as an order of magnitude.

**5\. Data fitting**

  * Report parameter uncertainties, not just point estimates.
  * Sparse data plus a near-critical point produces poorly constrained Chrastil parameters.
  * Data-driven surrogates interpolate; they never extrapolate.

These tools cover the calculations that a supercritical fluid project actually needs: property lookup, phase behaviour, custom thermodynamics, process estimation and data analysis. The combination of a reference library for the solvent and your own transparent code for everything else gives both accuracy and the ability to see why a number came out the way it did.

* * *

**Exercises**

#### Exercise 1: Property Comparison Across Solvents

Compare the density, viscosity and thermal conductivity of CO₂, ethanol and water in their supercritical states at matched reduced conditions ($P_r = 1.5$, $T_r = 1.2$). Convert to absolute conditions with CoolProp, plot all three fluids together, and argue which is the best heat-transfer fluid and why.

#### Exercise 2: Custom EOS Validation

Extend Example 6 into a full error map: compute the van der Waals, Peng-Robinson and CoolProp densities of CO₂ at 5, 10, 15 and 20 MPa and at 30, 40, 50, 75 and 100 °C. Plot the percentage error against reduced temperature and explain why the errors peak where they do.

#### Exercise 3: Extraction Optimization

Using Examples 7 and 8, minimize the CO₂ intensity (kg CO₂ per kg caffeine) subject to a recovery of at least 90%, with 1 kg of caffeine charged, a CO₂ flow of 5-20 kg/h, 40-80 °C and 100-300 bar. Use `scipy.optimize.minimize` and comment on which constraint binds at the optimum.

#### Exercise 4: Statistical Analysis of Solubility Data

Fit the Chrastil equation to the following representative ibuprofen data and report the parameters with 95% confidence intervals, $R^2$, and a residual plot. Then repeat the fit excluding the 60 °C / 100 bar point and discuss how much a single near-critical measurement moves the answer.

T (°C)| P (bar)| S (mg/kg)  
---|---|---  
40| 100| 1.2  
40| 150| 2.8  
40| 200| 4.5  
60| 100| 0.9  
60| 150| 2.1  
60| 200| 3.8  
  
#### Exercise 5: Phase Diagram Customization

Generate the $P$-$T$ phase diagram of ethanol with Example 3, then add the 1 atm isobar, mark the normal boiling point where it crosses the saturation curve, and shade the supercritical region. Why is supercritical ethanol used far less than scCO₂ despite its useful polarity?

#### Exercise 6: Mixed-Solvent Density

Compute the density of a CO₂ + ethanol mixture (10 mol% ethanol) at 50 °C and 150 bar with the HEOS backend, then repeat across 0-20 mol% ethanol. Plot density against composition and compare with the ideal (mole-fraction-weighted) mixing prediction. How large is the excess volume?

* * *

## References and Further Resources

### Python Libraries

  * [CoolProp documentation](<http://www.coolprop.org/>) \- reference thermophysical properties
  * [thermo documentation](<https://thermo.readthedocs.io/>) \- chemical engineering property models
  * [SciPy](<https://docs.scipy.org/doc/scipy/>) \- optimization and curve fitting

### Data Sources

  * [NIST Chemistry WebBook](<https://webbook.nist.gov/chemistry/>) \- thermophysical data
  * [NIST REFPROP](<https://www.nist.gov/srd/refprop>) \- reference fluid properties
  * [DIPPR database](<https://www.aiche.org/dippr>) \- physical property database
  * [PubChem](<https://pubchem.ncbi.nlm.nih.gov/>) \- chemical properties and descriptors

### Books

  * McHugh & Krukonis, _Supercritical Fluid Extraction: Principles and Practice_
  * Prausnitz, Lichtenthaler & de Azevedo, _Molecular Thermodynamics of Fluid-Phase Equilibria_
  * Poling, Prausnitz & O'Connell, _The Properties of Gases and Liquids_

* * *
