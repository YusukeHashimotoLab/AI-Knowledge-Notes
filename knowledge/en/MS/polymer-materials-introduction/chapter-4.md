---
title: "Chapter 4: Functional Polymers"
chapter_title: "Chapter 4: Functional Polymers"
---

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>):Chapter 4

🌐 EN | [🇯🇵 JP](<../../../jp/MS/polymer-materials-introduction/chapter-4.html>) | Last sync: 2025-11-16

  * [Table of Contents](<index.html>)
  * [� Chapter 3](<chapter-3.html>)
  * [Chapter 4](<chapter-4.html>)
  * [Chapter 5 ’](<index.html>)

This chapter covers Functional Polymers. You will learn essential concepts and techniques.

### Learning Objectives

**Beginner:**

  * Understand the basic mechanisms of conductive polymers (À-conjugated systems and doping)
  * Explain the requirements and representative examples of biocompatible polymers
  * Understand the principles of stimuli-responsive polymers (temperature and pH response)

**Intermediate:**

  * Calculate the relationship between conductivity and doping level
  * Predict LCST (Lower Critical Solution Temperature) using Flory-Huggins theory
  * Analyze drug release rates using kinetics models

**Advanced:**

  * Calculate bandgap and predict optical absorption spectra
  * Analyze ionic conductivity using Arrhenius equation
  * Model biodegradation rates and predict degradation time

## 4.1 Conductive Polymers

**Conductive Polymers** are organic materials with conjugated À-electron systems that exhibit electrical conductivity upon doping. Representative examples include **Polyaniline (PANI)** , **PEDOT:PSS** , and **Polypyrrole (PPy)**. 
    
    
    ```mermaid
    flowchart TD
                        A[Conductive Polymers] --> B[À-Conjugated System]
                        B --> C[HOMO-LUMO Bandgap]
                        A --> D[Doping]
                        D --> E[Oxidative Doping p-type]
                        D --> F[Reductive Doping n-type]
                        E --> G[Polaron Formation]
                        G --> H[Charge Carrier Generation]
                        H --> I[Electrical ConductivityÃ = 1-1000 S/cm]
                        I --> J[Applications: Transparent ElectrodesOrganic LED, Solar Cells]
    ```

### 4.1.1 Conductivity and Doping

Conductivity Ã depends on the doping level (oxidation/reduction degree). Below, we perform a doping simulation for polyaniline. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Conductive polymer doping simulation
    def simulate_conductivity_doping(polymer='Polyaniline'):
        """
        Simulate the relationship between doping level and conductivity
    
        Parameters:
        - polymer: Polymer name ('Polyaniline', 'PEDOT', 'Polypyrrole')
    
        Returns:
        - doping_levels: Doping level (%)
        - conductivities: Conductivity (S/cm)
        """
        # Doping level range (0-50%)
        doping_levels = np.linspace(0, 50, 100)
    
        # Conductivity model (empirical formula)
        # Ã = Ã_max * (x / x_opt)^2 * exp(-((x - x_opt) / w)^2)
        # x: doping level, x_opt: optimal doping, w: width parameter
    
        polymer_params = {
            'Polyaniline': {'sigma_max': 200, 'x_opt': 25, 'w': 15},
            'PEDOT': {'sigma_max': 1000, 'x_opt': 30, 'w': 20},
            'Polypyrrole': {'sigma_max': 100, 'x_opt': 20, 'w': 12}
        }
    
        params = polymer_params.get(polymer, polymer_params['Polyaniline'])
    
        # Conductivity calculation (S/cm)
        x_opt = params['x_opt']
        w = params['w']
        sigma_max = params['sigma_max']
    
        conductivities = sigma_max * ((doping_levels / x_opt) ** 2) * \
                         np.exp(-((doping_levels - x_opt) / w) ** 2)
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Conductivity vs Doping Level
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            x_opt_p = poly_params['x_opt']
            w_p = poly_params['w']
            sigma_max_p = poly_params['sigma_max']
            sigma_p = sigma_max_p * ((doping_levels / x_opt_p) ** 2) * \
                      np.exp(-((doping_levels - x_opt_p) / w_p) ** 2)
            plt.plot(doping_levels, sigma_p, linewidth=2, label=poly_name)
    
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Conductivity Ã (S/cm)', fontsize=12)
        plt.title('Conductivity vs Doping Level', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # Subplot 2: Carrier Density
        plt.subplot(1, 3, 2)
        # Carrier density n  doping level
        carrier_density = doping_levels * 1e20  # cm^-3 (hypothetical value)
        mobility = conductivities / (1.6e-19 * carrier_density + 1e-10)  # cm²/V·s
    
        plt.plot(doping_levels, carrier_density / 1e20, 'b-', linewidth=2)
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Carrier Density n (×10²p cm{³)', fontsize=12)
        plt.title(f'{polymer}: Carrier Density', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # Subplot 3: Mobility
        plt.subplot(1, 3, 3)
        plt.plot(doping_levels, mobility, 'r-', linewidth=2)
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Mobility ¼ (cm²/V·s)', fontsize=12)
        plt.title(f'{polymer}: Charge Carrier Mobility', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        plt.tight_layout()
        plt.savefig('conductivity_doping.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print(f"=== {polymer} Doping Analysis ===")
        print(f"Maximum Conductivity: {sigma_max} S/cm")
        print(f"Optimal Doping Level: {x_opt}%")
    
        # Values at specific doping levels
        for doping in [10, 25, 40]:
            idx = np.argmin(np.abs(doping_levels - doping))
            print(f"\nDoping Level {doping}%:")
            print(f"  Conductivity: {conductivities[idx]:.2f} S/cm")
            print(f"  Carrier Density: {carrier_density[idx]:.2e} cm{³")
    
        return doping_levels, conductivities
    
    # Execute
    simulate_conductivity_doping('Polyaniline')
    

### 4.1.2 Bandgap Calculation

The bandgap Eg of conjugated polymers can be determined from optical absorption spectra. Below, we simulate the relationship between HOMO-LUMO gap and optical absorption. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Bandgap and optical absorption spectrum
    def calculate_bandgap_absorption(bandgap_eV=2.5):
        """
        Calculate optical absorption spectrum from bandgap
    
        Parameters:
        - bandgap_eV: Bandgap energy (eV)
    
        Returns:
        - wavelengths: Wavelength (nm)
        - absorbance: Absorbance
        """
        # Wavelength range (nm)
        wavelengths = np.linspace(300, 800, 500)
    
        # Energy conversion E(eV) = 1240 / »(nm)
        photon_energies = 1240 / wavelengths
    
        # Absorption spectrum (simplified: step function + Gaussian broadening)
        def absorption_profile(E, Eg, width=0.3):
            """Absorption spectrum (Gaussian broadening)"""
            if E < Eg:
                return 0
            else:
                return np.exp(-((E - Eg) / width) ** 2)
    
        absorbance = np.array([absorption_profile(E, bandgap_eV) for E in photon_energies])
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Absorption spectrum
        plt.subplot(1, 3, 1)
        plt.plot(wavelengths, absorbance, 'b-', linewidth=2)
        # Bandgap corresponding wavelength
        lambda_g = 1240 / bandgap_eV
        plt.axvline(lambda_g, color='red', linestyle='--', linewidth=1.5,
                    label=f'»g = {lambda_g:.0f} nm (Eg = {bandgap_eV} eV)')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Absorbance (a.u.)', fontsize=12)
        plt.title('UV-Vis Absorption Spectrum', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Color-coded visible light region
        plt.fill_betweenx([0, max(absorbance)], 380, 450, color='violet', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 450, 495, color='blue', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 495, 570, color='green', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 570, 590, color='yellow', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 590, 620, color='orange', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 620, 750, color='red', alpha=0.2)
    
        # Subplot 2: Relationship between bandgap and color
        plt.subplot(1, 3, 2)
        bandgaps = np.linspace(1.5, 3.5, 50)
        lambda_gaps = 1240 / bandgaps
        colors_perceived = []
        for lam in lambda_gaps:
            if lam < 400:
                colors_perceived.append('UV')
            elif lam < 450:
                colors_perceived.append('Violet')
            elif lam < 495:
                colors_perceived.append('Blue')
            elif lam < 570:
                colors_perceived.append('Green')
            elif lam < 590:
                colors_perceived.append('Yellow')
            elif lam < 620:
                colors_perceived.append('Orange')
            elif lam < 750:
                colors_perceived.append('Red')
            else:
                colors_perceived.append('IR')
    
        plt.scatter(bandgaps, lambda_gaps, c=lambda_gaps, cmap='rainbow', s=50, edgecolors='black')
        plt.xlabel('Bandgap Eg (eV)', fontsize=12)
        plt.ylabel('Absorption Edge »g (nm)', fontsize=12)
        plt.title('Bandgap vs Absorption Wavelength', fontsize=14, fontweight='bold')
        plt.colorbar(label='Wavelength (nm)')
        plt.grid(alpha=0.3)
        plt.axhline(lambda_g, color='red', linestyle='--', alpha=0.7)
    
        # Subplot 3: Comparison of multiple bandgaps
        plt.subplot(1, 3, 3)
        bandgaps_examples = [1.8, 2.5, 3.2]
        for Eg in bandgaps_examples:
            photon_E = 1240 / wavelengths
            abs_spec = np.array([absorption_profile(E, Eg, 0.3) for E in photon_E])
            plt.plot(wavelengths, abs_spec, linewidth=2, label=f'Eg = {Eg} eV')
    
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Absorbance (a.u.)', fontsize=12)
        plt.title('Effect of Bandgap on Absorption', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(300, 800)
    
        plt.tight_layout()
        plt.savefig('bandgap_absorption.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== Bandgap Analysis ===")
        print(f"Bandgap: {bandgap_eV} eV")
        print(f"Absorption Edge Wavelength: {lambda_g:.1f} nm")
    
        if lambda_g < 400:
            color_range = "Ultraviolet (UV)"
        elif lambda_g < 750:
            color_range = "Visible Light"
        else:
            color_range = "Infrared (IR)"
    
        print(f"Absorption Region: {color_range}")
        print(f"\nTypical Eg for conductive polymers: 1.5-3.0 eV")
    
        return wavelengths, absorbance
    
    # Example execution: Eg = 2.5 eV (equivalent to PEDOT:PSS)
    calculate_bandgap_absorption(bandgap_eV=2.5)
    

## 4.2 Biocompatible Polymers

**Biocompatible Polymers** are materials that do not cause toxicity or immune response when in contact with biological tissues. Representative examples include **PEG (Polyethylene Glycol)** , **Polylactic Acid (PLA)** , and **PLGA (Poly(lactic-co-glycolic acid))**. 

### 4.2.1 Drug Release Kinetics

Drug release from biodegradable polymers is determined by the competition between diffusion and degradation. Below, we analyze release behavior using the Korsmeyer-Peppas model. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Drug release kinetics
    def simulate_drug_release_kinetics(model='Korsmeyer-Peppas'):
        """
        Simulate drug release from biodegradable polymers
    
        Parameters:
        - model: Release model ('Korsmeyer-Peppas', 'Higuchi', 'First-order')
    
        Returns:
        - time: Time (hours)
        - release_fraction: Cumulative release fraction (%)
        """
        # Time range (hours)
        time = np.linspace(0, 48, 500)
    
        # Korsmeyer-Peppas model: Mt/M = k * t^n
        # n: diffusion exponent (n=0.5: Fickian diffusion, n=1.0: Case II, 0.5<n<1: """first-order="" """higuchi="" """korsmeyer-peppas="" #="" ')="" 'anomalous="" 'case="" 'fickian="" 'first-order':="" 'higuchi':="" (%="" (%)',="" (1="" (degradation-limited)="" (dm="" (hours)',="" (n="0.5)':" (porous="" (t="" )="" *="" **="" -="" 0.5,="" 0.7,="" 1)="" 1.0]="" 100="" 100%="" 100)="" 100,="" 105)="" 1:="" 2)="" 24="" 24))]:.1f}%")="" 2:="" 3)="" 3,="" 3:="" 48)="" 5))="" 50%="" 50))]="" <="" =="" analysis='==")' anomalous="" at="" bbox_inches="tight" cap="" code="" comparison="" comparison',="" def="" diffusion="" diffusion)="" dm="" dpi="300," drug="" dt="time[1]" dt)="" effect="" execute="" exp(-k*t))="" exponent="" first-order="" first_order(t,="" first_order(time,="" fontsize="12)" fontweight="bold" for="" higuchi="" higuchi(t,="" higuchi(time,="" hour)',="" hours")="" hours:="" ii="" in="" k="0.1," kinetics="" kinetics:="" korsmeyer_peppas(t,="" korsmeyer_peppas(time,="" label="model_name)" linewidth="2," m="k" matrix)="" model="" model"""="" model:="" model_name,="" models="" models_data="{" models_data.items():="" mt="" mt_minf="k" multiple="" n="0.5):" n',="" n)="" n_values="[0.3," n_values:="" none="" np.exp(-k="" np.minimum(mt_minf="" np.minimum(mt_minf,="" np.sqrt(t)="" of="" output="" plt.figure(figsize="(14," plt.grid(alpha="0.3)" plt.legend()="" plt.plot(time,="" plt.savefig('drug_release_kinetics.png',="" plt.show()="" plt.subplot(1,="" plt.tight_layout()="" plt.title('drug="" plt.title('effect="" plt.xlabel('time="" plt.xlim(0,="" plt.ylabel('cumulative="" plt.ylabel('release="" plt.ylim(0,="" print("="==" print(f"="" print(f"\n{model_name}:")="" rate="" release="" release,="" release_data="" release_data,="" release_rate="np.gradient(release_data," release_rate,="" results="" return="" simulate_drug_release_kinetics()="" sqrt(t)="" subplot="" t))="" t_50="time[np.argmin(np.abs(release_data" time="" time',="" time,="" time:="" time[0]="" visualization="" vs="" {release_data[np.argmin(np.abs(time="" }=""></n<1:>

### 4.2.2 Biodegradation Rate Analysis

Biodegradable polymers such as polylactic acid (PLA) degrade by hydrolysis. Molecular weight reduction can be modeled as a first-order reaction. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Biodegradation rate simulation
    def simulate_biodegradation(polymer='PLA', temperature=310):
        """
        Simulate biodegradation rate of biodegradable polymers
    
        Parameters:
        - polymer: Polymer name ('PLA', 'PLGA', 'PCL')
        - temperature: Temperature (K)
    
        Returns:
        - time: Time (days)
        - molecular_weight: Molecular weight (relative value)
        """
        # Degradation parameters (Arrhenius equation)
        # k = k0 * exp(-Ea/RT)
        polymer_params = {
            'PLA': {'k0': 1e10, 'Ea': 80000},  # J/mol
            'PLGA': {'k0': 1e11, 'Ea': 75000},
            'PCL': {'k0': 1e9, 'Ea': 85000}
        }
    
        params = polymer_params.get(polymer, polymer_params['PLA'])
        k0 = params['k0']
        Ea = params['Ea']
        R = 8.314  # J/mol·K
    
        # Rate constant (1/day)
        k = k0 * np.exp(-Ea / (R * temperature)) * 86400  # seconds to days conversion
    
        # Time range (days)
        time = np.linspace(0, 365, 500)
    
        # First-order degradation: Mw(t) = Mw0 * exp(-k*t)
        Mw0 = 100000  # Initial molecular weight (g/mol)
        molecular_weight = Mw0 * np.exp(-k * time)
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Molecular weight reduction
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            k_poly = poly_params['k0'] * np.exp(-poly_params['Ea'] / (R * temperature)) * 86400
            Mw_t = Mw0 * np.exp(-k_poly * time)
            plt.plot(time, Mw_t / 1000, linewidth=2, label=poly_name)
    
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Molecular Weight (kDa)', fontsize=12)
        plt.title(f'Biodegradation at {temperature}K ({temperature-273.15:.0f}°C)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # Subplot 2: Temperature dependence
        plt.subplot(1, 3, 2)
        temperatures = [298, 310, 323]  # K (25, 37, 50°C)
        for T in temperatures:
            k_T = k0 * np.exp(-Ea / (R * T)) * 86400
            Mw_T = Mw0 * np.exp(-k_T * time)
            plt.plot(time, Mw_T / 1000, linewidth=2, label=f'{T-273.15:.0f}°C')
    
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Molecular Weight (kDa)', fontsize=12)
        plt.title(f'{polymer}: Temperature Dependence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # Subplot 3: Degradation percentage (mass remaining)
        plt.subplot(1, 3, 3)
        # Degradation rate = (Mw0 - Mw(t)) / Mw0 * 100
        degradation_percent = (1 - molecular_weight / Mw0) * 100
        plt.plot(time, degradation_percent, 'b-', linewidth=2)
        plt.axhline(50, color='red', linestyle='--', linewidth=1.5, label='50% Degradation')
        # 50% degradation time
        t_50 = -np.log(0.5) / k
        plt.axvline(t_50, color='green', linestyle='--', linewidth=1.5,
                    label=f't…€ = {t_50:.0f} days')
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Degradation (%)', fontsize=12)
        plt.title(f'{polymer}: Degradation Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('biodegradation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print(f"=== {polymer} Biodegradation Analysis ({temperature}K = {temperature-273.15:.0f}°C) ===")
        print(f"Initial Molecular Weight Mw0: {Mw0/1000:.0f} kDa")
        print(f"Activation Energy Ea: {Ea/1000:.0f} kJ/mol")
        print(f"Degradation Rate Constant k: {k:.2e} 1/day")
        print(f"50% Degradation Time t…€: {t_50:.0f} days")
        print(f"90% Degradation Time t‰€: {-np.log(0.1)/k:.0f} days")
    
        return time, molecular_weight
    
    # Example execution: PLA, 37°C (body temperature)
    simulate_biodegradation('PLA', temperature=310)
    

## 4.3 Stimuli-Responsive Polymers

**Stimuli-Responsive Polymers** change their structure or properties in response to external stimuli such as temperature, pH, and light. A representative example is **PNIPAM (Poly(N-isopropylacrylamide))** , which exhibits **LCST (Lower Critical Solution Temperature)**. 

### 4.3.1 LCST Calculation (Flory-Huggins Theory)

LCST is explained by the temperature dependence of the Flory-Huggins interaction parameter Ç: 

\\[ \chi = A + \frac{B}{T} \\] 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # LCST calculation (Flory-Huggins theory)
    def calculate_lcst_flory_huggins(polymer='PNIPAM'):
        """
        Calculate LCST phase diagram based on Flory-Huggins theory
    
        Parameters:
        - polymer: Polymer name ('PNIPAM', 'PEO')
    
        Returns:
        - temperatures: Temperature (K)
        - volume_fractions: Volume fraction
        """
        # Flory-Huggins parameter (temperature dependence)
        # Ç(T) = A + B/T
        polymer_params = {
            'PNIPAM': {'A': -12.0, 'B': 4300},  # K
            'PEO': {'A': -15.0, 'B': 5000}
        }
    
        params = polymer_params.get(polymer, polymer_params['PNIPAM'])
        A = params['A']
        B = params['B']
    
        # Volume fraction range
        phi = np.linspace(0.01, 0.99, 100)
    
        # Degree of polymerization (polymer/solvent)
        N = 1000  # Polymer degree of polymerization
    
        # Spinodal curve (second derivative = 0)
        # d²”Gmix/dÆ² = 0 ’ Ç_spinodal = 0.5 * (1/(N*Æ) + 1/(1-Æ))
        chi_spinodal = 0.5 * (1 / (N * phi) + 1 / (1 - phi))
    
        # Temperature calculation (from Ç = A + B/T, T = B / (Ç - A))
        temperatures_spinodal = B / (chi_spinodal - A)
    
        # Critical point (Æ = 1/(N+1) H 1/N)
        phi_critical = 1 / np.sqrt(N + 1)
        chi_critical = 0.5 * (1 + 1/np.sqrt(N))**2
        T_critical = B / (chi_critical - A)
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Phase diagram
        plt.subplot(1, 3, 1)
        plt.plot(phi * 100, temperatures_spinodal - 273.15, 'b-', linewidth=2, label='Spinodal Curve (LCST)')
        plt.scatter([phi_critical * 100], [T_critical - 273.15], s=200, c='red',
                    edgecolors='black', linewidths=2, zorder=5, label=f'Critical Point ({T_critical-273.15:.1f}°C)')
        plt.fill_between(phi * 100, temperatures_spinodal - 273.15, 100, alpha=0.3, color='red',
                         label='Two-Phase Region')
        plt.fill_between(phi * 100, 0, temperatures_spinodal - 273.15, alpha=0.3, color='green',
                         label='Single-Phase Region')
        plt.xlabel('Polymer Volume Fraction Æ (%)', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.title(f'{polymer} LCST Phase Diagram', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 100)
    
        # Subplot 2: Temperature dependence of Ç parameter
        plt.subplot(1, 3, 2)
        T_range = np.linspace(273, 373, 100)  # K
        chi_T = A + B / T_range
        plt.plot(T_range - 273.15, chi_T, 'purple', linewidth=2)
        plt.axhline(chi_critical, color='red', linestyle='--', linewidth=1.5,
                    label=f'Ç_crit = {chi_critical:.3f}')
        plt.axvline(T_critical - 273.15, color='green', linestyle='--', linewidth=1.5,
                    label=f'LCST = {T_critical-273.15:.1f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Flory-Huggins Parameter Ç', fontsize=12)
        plt.title('Temperature Dependence of Ç', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Turbidity change simulation
        plt.subplot(1, 3, 3)
        phi_sample = 0.05  # 5% solution
        temperatures_exp = np.linspace(10, 60, 100)  # °C
        chi_exp = A + B / (temperatures_exp + 273.15)
        # Phase separation determination (phase separates when Ç > Ç_spinodal)
        chi_spinodal_at_phi = 0.5 * (1 / (N * phi_sample) + 1 / (1 - phi_sample))
        turbidity = np.where(chi_exp > chi_spinodal_at_phi, 1, 0)
    
        plt.plot(temperatures_exp, turbidity, 'b-', linewidth=3)
        plt.fill_between(temperatures_exp, turbidity, alpha=0.3, color='blue')
        plt.axvline(T_critical - 273.15, color='red', linestyle='--', linewidth=1.5,
                    label=f'LCST = {T_critical-273.15:.1f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Turbidity (Phase Separation)', fontsize=12)
        plt.title(f'{polymer} (Æ = {phi_sample*100}%): Turbidity Change', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(-0.1, 1.2)
    
        plt.tight_layout()
        plt.savefig('lcst_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print(f"=== {polymer} LCST Analysis (Flory-Huggins Theory) ===")
        print(f"Flory-Huggins Parameter: Ç = {A} + {B}/T")
        print(f"Degree of Polymerization N: {N}")
        print(f"Critical Volume Fraction Æ_c: {phi_critical:.4f}")
        print(f"Critical Ç Value: {chi_critical:.4f}")
        print(f"LCST: {T_critical - 273.15:.1f}°C")
    
        return temperatures_spinodal, phi
    
    # Execute
    calculate_lcst_flory_huggins('PNIPAM')
    

### 4.3.2 pH-Responsive Ionization Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # pH-responsive polymer ionization calculation
    def calculate_ph_responsive_ionization(pKa=5.5):
        """
        Calculate ionization degree of pH-responsive polymer using Henderson-Hasselbalch equation
    
        Parameters:
        - pKa: Acid dissociation constant
    
        Returns:
        - pH_values: pH values
        - ionization_degrees: Ionization degree
        """
        # pH range
        pH_values = np.linspace(2, 10, 200)
    
        # Henderson-Hasselbalch equation
        # ± = 1 / (1 + 10^(pKa - pH)) (for weak acidic groups)
        ionization_degree = 1 / (1 + 10**(pKa - pH_values))
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Ionization degree vs pH
        plt.subplot(1, 3, 1)
        pKa_values = [4.5, 5.5, 6.5]
        for pKa_val in pKa_values:
            alpha = 1 / (1 + 10**(pKa_val - pH_values))
            plt.plot(pH_values, alpha * 100, linewidth=2, label=f'pKa = {pKa_val}')
    
        plt.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('Ionization Degree (%)', fontsize=12)
        plt.title('pH-Responsive Ionization', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Swelling ratio (proportional to ionization degree)
        plt.subplot(1, 3, 2)
        # Swelling ratio Q  ±² (Donnan effect)
        swelling_ratio = 1 + 10 * ionization_degree**2
        plt.plot(pH_values, swelling_ratio, 'purple', linewidth=2)
        plt.axvline(pKa, color='red', linestyle='--', linewidth=1.5, label=f'pKa = {pKa}')
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('Swelling Ratio Q/Q€', fontsize=12)
        plt.title('pH-Induced Swelling', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Titration curve
        plt.subplot(1, 3, 3)
        # Titration curve (NaOH addition and pH)
        # Simplified: weak acid titration
        V_NaOH = np.linspace(0, 50, 200)  # mL
        # pH = pKa + log((V_NaOH) / (V_eq - V_NaOH)) (V_eq: equivalence point)
        V_eq = 25  # mL
        pH_titration = []
        for V in V_NaOH:
            if V < V_eq:
                if V > 0:
                    pH_val = pKa + np.log10(V / (V_eq - V))
                else:
                    pH_val = 3  # Initial pH (assumed)
            elif V == V_eq:
                pH_val = 7  # Equivalence point (weak acid-strong base)
            else:
                pH_val = 7 + np.log10((V - V_eq) / V_eq)
            pH_titration.append(pH_val)
    
        plt.plot(V_NaOH, pH_titration, 'g-', linewidth=2)
        plt.axvline(V_eq, color='red', linestyle='--', linewidth=1.5, label=f'Equivalence Point ({V_eq} mL)')
        plt.axhline(pKa, color='blue', linestyle='--', linewidth=1.5, label=f'pKa = {pKa}')
        plt.xlabel('Volume of NaOH (mL)', fontsize=12)
        plt.ylabel('pH', fontsize=12)
        plt.title('Titration Curve of pH-Responsive Polymer', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(2, 12)
    
        plt.tight_layout()
        plt.savefig('ph_responsive_ionization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== pH-Responsive Polymer Analysis ===")
        print(f"pKa: {pKa}")
        print(f"Ionization Degree at pH = pKa: 50%")
        print(f"\nIonization Degree by pH:")
        for pH_target in [3, 5, 7, 9]:
            idx = np.argmin(np.abs(pH_values - pH_target))
            print(f"  pH {pH_target}: {ionization_degree[idx]*100:.1f}%")
    
        return pH_values, ionization_degree
    
    # Execute
    calculate_ph_responsive_ionization(pKa=5.5)
    

## 4.4 Polymer Electrolytes

**Polymer Electrolytes** are polymer materials that exhibit ionic conductivity and are applied in lithium-ion batteries and fuel cells. A representative example is **Nafion** (proton conducting membrane). 

### 4.4.1 Arrhenius Analysis of Ionic Conductivity
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Arrhenius analysis of ionic conductivity
    def analyze_ionic_conductivity(polymer='Nafion'):
        """
        Analyze ionic conductivity of polymer electrolytes using Arrhenius equation
    
        Parameters:
        - polymer: Polymer name ('Nafion', 'PEO-LiTFSI')
    
        Returns:
        - temperatures: Temperature (K)
        - conductivities: Ionic conductivity (S/cm)
        """
        # Arrhenius equation: Ã = Ã0 * exp(-Ea / RT)
        polymer_params = {
            'Nafion': {'sigma0': 1e4, 'Ea': 15000},  # S/cm, J/mol
            'PEO-LiTFSI': {'sigma0': 1e6, 'Ea': 50000}
        }
    
        params = polymer_params.get(polymer, polymer_params['Nafion'])
        sigma0 = params['sigma0']
        Ea = params['Ea']
        R = 8.314  # J/mol·K
    
        # Temperature range (K)
        temperatures = np.linspace(273, 373, 100)
    
        # Ionic conductivity (S/cm)
        conductivities = sigma0 * np.exp(-Ea / (R * temperatures))
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Arrhenius plot
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            sigma0_p = poly_params['sigma0']
            Ea_p = poly_params['Ea']
            sigma_p = sigma0_p * np.exp(-Ea_p / (R * temperatures))
            plt.plot(1000 / temperatures, np.log10(sigma_p), linewidth=2, label=poly_name)
    
        plt.xlabel('1000/T (K{¹)', fontsize=12)
        plt.ylabel('log(Ã) [S/cm]', fontsize=12)
        plt.title('Arrhenius Plot of Ionic Conductivity', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Conductivity vs temperature
        plt.subplot(1, 3, 2)
        plt.plot(temperatures - 273.15, conductivities, 'b-', linewidth=2, label=polymer)
        plt.axhline(1e-4, color='red', linestyle='--', linewidth=1.5,
                    label='Target (10{t S/cm)')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Ionic Conductivity Ã (S/cm)', fontsize=12)
        plt.title(f'{polymer}: Conductivity vs Temperature', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Activation energy comparison
        plt.subplot(1, 3, 3)
        poly_names = list(polymer_params.keys())
        Ea_values = [polymer_params[p]['Ea'] / 1000 for p in poly_names]  # kJ/mol
    
        bars = plt.bar(poly_names, Ea_values, color=['#4A90E2', '#E74C3C'],
                       edgecolor='black', linewidth=2)
        plt.ylabel('Activation Energy Ea (kJ/mol)', fontsize=12)
        plt.title('Comparison of Activation Energies', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
    
        for bar, val in zip(bars, Ea_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f} kJ/mol', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('ionic_conductivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print(f"=== {polymer} Ionic Conductivity Analysis ===")
        print(f"Arrhenius Parameters: Ã0 = {sigma0:.2e} S/cm, Ea = {Ea/1000:.1f} kJ/mol")
    
        for T_target in [298, 323, 353]:  # 25, 50, 80°C
            idx = np.argmin(np.abs(temperatures - T_target))
            print(f"\nTemperature {T_target}K ({T_target-273.15:.0f}°C):")
            print(f"  Ionic Conductivity: {conductivities[idx]:.2e} S/cm")
    
        return temperatures, conductivities
    
    # Execute
    analyze_ionic_conductivity('Nafion')
    

## Exercises

#### Exercise 1: Conductivity Calculation (Easy)

Calculate the conductivity using the simple formula Ã = Ã_max × (x/x_opt) when doping level is 30%, maximum conductivity is 500 S/cm, and optimal doping is 25%.

Show Answer
    
    
    sigma_max = 500
    x = 30
    x_opt = 25
    sigma = sigma_max * (x / x_opt)
    print(f"Conductivity: {sigma} S/cm")
    # Output: 600 S/cm (over-doping)

#### Exercise 2: Bandgap Calculation (Easy)

Calculate the bandgap (eV) of a conductive polymer with an absorption edge wavelength of 550 nm.

Show Answer
    
    
    lambda_nm = 550
    Eg = 1240 / lambda_nm
    print(f"Bandgap: {Eg:.2f} eV")
    # Output: 2.25 eV

#### Exercise 3: Drug Release Time (Easy)

Calculate the 50% release time using the Korsmeyer-Peppas model (k=0.1, n=0.5).

Show Answer
    
    
    k = 0.1
    n = 0.5
    Mt_Minf = 0.5
    t_50 = (Mt_Minf / k)**(1/n)
    print(f"50% Release Time: {t_50:.1f} hours")
    # Output: 25.0 hours

#### Exercise 4: LCST Prediction (Medium)

Calculate the LCST when Ç = -12 + 4300/T and critical Ç = 0.502.

Show Answer
    
    
    A = -12
    B = 4300
    chi_critical = 0.502
    T_lcst = B / (chi_critical - A)
    print(f"LCST: {T_lcst:.1f} K = {T_lcst - 273.15:.1f}°C")
    # Output: LCST: 344.0 K = 70.8°C

#### Exercise 5: pH-Responsive Ionization (Medium)

Calculate the ionization degree when a polymer with pKa = 5.5 is immersed in a solution at pH 7.0.

Show Answer
    
    
    pKa = 5.5
    pH = 7.0
    alpha = 1 / (1 + 10**(pKa - pH))
    print(f"Ionization Degree: {alpha*100:.1f}%")
    # Output: 96.9%

#### Exercise 6: Biodegradation Half-life (Medium)

Calculate the molecular weight half-life when the rate constant k = 0.005 1/day.

Show Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculate the molecular weight half-life when the rate const
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    k = 0.005
    t_half = np.log(2) / k
    print(f"Half-life: {t_half:.0f} days")
    # Output: 139 days

#### Exercise 7: Ionic Conductivity Calculation (Medium)

Calculate the ionic conductivity when Ã0 = 1×10t S/cm, Ea = 15 kJ/mol, T = 80°C (353 K) (R = 8.314 J/mol·K).

Show Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculate the ionic conductivity when Ã0 = 1×10t S/cm, Ea = 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    sigma0 = 1e4
    Ea = 15000
    R = 8.314
    T = 353
    sigma = sigma0 * np.exp(-Ea / (R * T))
    print(f"Ionic Conductivity: {sigma:.2e} S/cm")
    # Output: Approximately 0.01 S/cm

#### Exercise 8: Optical Absorption Spectrum Prediction (Hard)

For a conjugated polymer with Eg = 2.0 eV, predict the absorption edge wavelength and main absorption color. Also estimate the color of transmitted light.

Show Answer
    
    
    Eg = 2.0
    lambda_edge = 1240 / Eg
    print(f"Absorption Edge Wavelength: {lambda_edge:.0f} nm")
    print("Absorption Color: Red to green (wavelengths below 620nm)")
    print("Transmitted Light Color: Red (red is transmitted as complementary color)")
    # Output: 620 nm, absorbs up to red region, appears red

#### Exercise 9: Drug Release Control (Hard)

Optimize parameter k in the Korsmeyer-Peppas model (n=0.6) to achieve 80% release in 24 hours.

Show Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Optimize parameter k in the Korsmeyer-Peppas model (n=0.6) t
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    t_target = 24
    Mt_Minf_target = 0.8
    n = 0.6
    k = Mt_Minf_target / (t_target**n)
    print(f"Optimal k: {k:.4f}")
    print(f"Verification: Mt/M = {k * (24**n):.2f}")
    # Output: k H 0.0927, 80% release at 24 hours

#### Exercise 10: Multifunctional Polymer Design (Hard)

Design a polymer that combines conductivity (Ã > 1 S/cm) and biocompatibility. Assuming a PEDOT-PEG copolymer, propose the optimal composition.

Show Answer

**Design Strategy:**

  * PEDOT content: 70-80% (ensure conductivity)
  * PEG content: 20-30% (provide biocompatibility and hydrophilicity)
  * Expected properties: Ã = 1-10 S/cm, good cell adhesion

    
    
    # Optimal composition simulation
    PEDOT_ratio = 0.75
    PEG_ratio = 0.25
    sigma_max_PEDOT = 1000  # S/cm
    sigma_estimated = sigma_max_PEDOT * PEDOT_ratio * 0.1  # Considering dilution effect
    print(f"PEDOT: {PEDOT_ratio*100}%, PEG: {PEG_ratio*100}%")
    print(f"Predicted Conductivity: {sigma_estimated:.1f} S/cm")
    print("Biocompatibility: PEG improves cell adhesion")
    # Output: Ã H 7.5 S/cm, good biocompatibility

## References

  1. Skotheim, T. A., & Reynolds, J. R. (Eds.). (2007). _Handbook of Conducting Polymers_ (3rd ed.). CRC Press. pp. 1-85.
  2. Ratner, B. D., et al. (2013). _Biomaterials Science: An Introduction to Materials in Medicine_ (3rd ed.). Academic Press. pp. 120-195.
  3. Stuart, M. A. C., et al. (2010). Emerging applications of stimuli-responsive polymer materials. _Nature Materials_ , 9, 101-113.
  4. Dobrynin, A. V., & Rubinstein, M. (2005). Theory of polyelectrolytes in solutions and at surfaces. _Progress in Polymer Science_ , 30, 1049-1118.
  5. Mauritz, K. A., & Moore, R. B. (2004). State of understanding of Nafion. _Chemical Reviews_ , 104(10), 4535-4585.
  6. Siepmann, J., & Peppas, N. A. (2001). Modeling of drug release from delivery systems. _Advanced Drug Delivery Reviews_ , 48, 139-157.

### Connection to Next Chapter

In Chapter 5, we will integrate all the knowledge learned in this series to build practical workflows with Python. From polymer structure generation with RDKit, Tg prediction using machine learning, MD simulation data analysis, to database integration with PolyInfo and others, you will acquire skills that are immediately applicable in practical work. 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
