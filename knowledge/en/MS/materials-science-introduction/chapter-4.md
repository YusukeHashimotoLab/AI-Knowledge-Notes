---
title: "Chapter 4: Relationship Between Material Properties and Structure"
chapter_title: "Chapter 4: Relationship Between Material Properties and Structure"
subtitle: Understanding Mechanical, Electrical, Thermal, and Optical Properties
reading_time: 35-40 minutes
difficulty: Intermediate
code_examples: 6
version: 1.0
created_at: 2025-10-25
---

The properties of a material are determined by its atomic structure, crystal structure, and the types of chemical bonding. In this chapter, we study mechanical properties (strength, hardness, ductility), electrical properties (conductivity, semiconductivity), thermal properties (heat conduction, thermal expansion), and optical properties (transparency, color), and use Python to calculate and visualize material characteristics. 

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand stress-strain curves and evaluate the mechanical properties of materials
  * ✅ Understand hardness testing methods (Vickers, Brinell, Rockwell) and convert between them
  * ✅ Explain the relationship between band structure and electrical conduction
  * ✅ Understand thermal properties (thermal conductivity, linear expansion coefficient, specific heat)
  * ✅ Understand the basics of optical properties (transparency, color, refractive index)
  * ✅ Calculate, plot, and compare material properties with Python

* * *

## 4.1 Mechanical Properties (Strength, Hardness, Ductility)

### Stress and Strain

**Stress** is the force acting per unit area of a material:

$$\sigma = \frac{F}{A}$$

where:

  * $\sigma$: stress (Pa = N/m² or MPa)
  * $F$: load (N)
  * $A$: cross-sectional area (m²)

**Strain** is the relative amount of deformation of a material:

$$\varepsilon = \frac{\Delta L}{L_0}$$

where:

  * $\varepsilon$: strain (dimensionless)
  * $\Delta L$: elongation (m)
  * $L_0$: original length (m)

### Stress-Strain Curve

When a material is loaded in tension, a stress-strain curve is obtained. This curve reveals the mechanical properties of the material.

**Key regions** :

  1. **Elastic Region** : Linear region following Hooke's law. The material returns to its original shape when the load is removed.
  2. **Yield Point** : The point of transition from elastic to plastic behavior, characterized by the yield strength.
  3. **Plastic Region** : Region where permanent deformation occurs.
  4. **Ultimate Tensile Strength (UTS)** : The maximum stress the material can withstand.
  5. **Fracture** : The point at which the material breaks.

**Young's Modulus (E)** : the slope of the elastic region

$$E = \frac{\sigma}{\varepsilon}$$

The unit is GPa (gigapascal). The larger the Young's modulus, the stiffer the material and the harder it is to deform.

Material | Young's Modulus (GPa) | Yield Strength (MPa) | Tensile Strength (MPa) | Ductility  
---|---|---|---|---  
**Steel** | 200 | 250-400 | 400-550 | High  
**Aluminum (Al)** | 69 | 35-100 | 90-150 | High  
**Copper (Cu)** | 130 | 70 | 220 | High  
**Titanium (Ti)** | 116 | 140-500 | 240-550 | Medium  
**Glass (SiO₂)** | 70 | - | 50-100 | Brittle  
**Ceramics (Al₂O₃)** | 380 | - | 300-400 | Brittle  
  
### Ductility and Brittleness

**Ductile materials** : undergo large plastic deformation (metals)

  * Elongate significantly before fracture
  * High elongation at break (typically > 5%)
  * Examples: copper, aluminum, steel

**Brittle materials** : undergo little plastic deformation (ceramics, glass)

  * Fracture with almost no elongation
  * Low elongation at break (typically < 5%)
  * Examples: glass, ceramics, cast iron

### Code Example 1: Creating and Plotting Stress-Strain Curves (Multi-Material Comparison)

We create stress-strain curves for steel, aluminum, and glass, and visualize the differences between materials.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Stress-strain curve simulation
    def stress_strain_curve(material='steel'):
        """
        Simulate the stress-strain curve of a material
    
        Parameters:
        material: one of 'steel', 'aluminum', 'glass'
    
        Returns:
        strain: strain array
        stress: stress array (MPa)
        """
        if material == 'steel':
            # Steel (ductile material)
            E = 200e3  # Young's modulus (MPa)
            yield_stress = 250  # Yield strength (MPa)
            yield_strain = yield_stress / E  # Yield strain
            uts = 400  # Tensile strength (MPa)
            fracture_strain = 0.25  # Fracture strain (25% elongation)
    
            # Elastic region (from 0 to the yield point)
            strain_elastic = np.linspace(0, yield_strain, 100)
            stress_elastic = E * strain_elastic
    
            # Plastic region (from yield point to fracture) - includes work hardening
            strain_plastic = np.linspace(yield_strain, fracture_strain, 300)
            # Work hardening: stress increases, but at a decreasing rate
            stress_plastic = yield_stress + (uts - yield_stress) * \
                            (1 - np.exp(-10 * (strain_plastic - yield_strain)))
    
            # Softening after necking
            strain_necking = np.linspace(fracture_strain, fracture_strain + 0.05, 50)
            stress_necking = uts * np.exp(-20 * (strain_necking - fracture_strain))
    
            strain = np.concatenate([strain_elastic, strain_plastic, strain_necking])
            stress = np.concatenate([stress_elastic, stress_plastic, stress_necking])
    
            properties = {
                'E': E,
                'yield_stress': yield_stress,
                'UTS': uts,
                'fracture_strain': fracture_strain + 0.05,
                'type': 'Ductile material'
            }
    
        elif material == 'aluminum':
            # Aluminum (ductile material, softer than steel)
            E = 69e3  # Young's modulus (MPa)
            yield_stress = 35  # Yield strength (MPa)
            yield_strain = yield_stress / E
            uts = 90  # Tensile strength (MPa)
            fracture_strain = 0.18  # Fracture strain (18% elongation)
    
            strain_elastic = np.linspace(0, yield_strain, 100)
            stress_elastic = E * strain_elastic
    
            strain_plastic = np.linspace(yield_strain, fracture_strain, 300)
            stress_plastic = yield_stress + (uts - yield_stress) * \
                            (1 - np.exp(-8 * (strain_plastic - yield_strain)))
    
            strain_necking = np.linspace(fracture_strain, fracture_strain + 0.04, 50)
            stress_necking = uts * np.exp(-15 * (strain_necking - fracture_strain))
    
            strain = np.concatenate([strain_elastic, strain_plastic, strain_necking])
            stress = np.concatenate([stress_elastic, stress_plastic, stress_necking])
    
            properties = {
                'E': E,
                'yield_stress': yield_stress,
                'UTS': uts,
                'fracture_strain': fracture_strain + 0.04,
                'type': 'Ductile material'
            }
    
        elif material == 'glass':
            # Glass (brittle material)
            E = 70e3  # Young's modulus (MPa)
            fracture_stress = 70  # Fracture stress (MPa)
            fracture_strain = fracture_stress / E  # Fracture strain (about 0.1%)
    
            # Elastic region only (linear up to fracture)
            strain = np.linspace(0, fracture_strain, 200)
            stress = E * strain
    
            properties = {
                'E': E,
                'yield_stress': None,  # No yielding
                'UTS': fracture_stress,
                'fracture_strain': fracture_strain,
                'type': 'Brittle material'
            }
    
        else:
            raise ValueError("material must be one of 'steel', 'aluminum', 'glass'")
    
        return strain, stress, properties
    
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    materials = ['steel', 'aluminum', 'glass']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    labels = ['Steel', 'Aluminum', 'Glass']
    
    for material, color, label in zip(materials, colors, labels):
        # Compute the stress-strain curve
        strain, stress, props = stress_strain_curve(material)
    
        # Convert strain to percent
        strain_percent = strain * 100
    
        # Plot
        ax.plot(strain_percent, stress, linewidth=2.5, color=color, label=label)
    
        # Mark the yield point (ductile materials only)
        if props['yield_stress'] is not None:
            yield_strain = props['yield_stress'] / props['E']
            ax.plot(yield_strain * 100, props['yield_stress'],
                   'o', markersize=10, color=color,
                   markeredgecolor='black', markeredgewidth=1.5)
    
        # Mark the tensile strength
        if material != 'glass':
            # Ductile materials: find the UTS point
            uts_idx = np.argmax(stress)
            ax.plot(strain_percent[uts_idx], stress[uts_idx],
                   's', markersize=10, color=color,
                   markeredgecolor='black', markeredgewidth=1.5)
    
    # Axis labels and title
    ax.set_xlabel('Strain (% = $\\varepsilon$ × 100)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Stress (MPa = $\\sigma$)', fontsize=13, fontweight='bold')
    ax.set_title('Comparison of Stress-Strain Curves (Ductile vs Brittle Materials)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 450)
    
    # Add annotations
    ax.annotate('Yield Point', xy=(0.125, 250), xytext=(3, 350),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))
    
    ax.annotate('Tensile Strength\n(UTS)', xy=(12, 400), xytext=(15, 320),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))
    
    ax.annotate('Brittle fracture\n(almost no elongation)', xy=(0.1, 70), xytext=(2, 150),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.6))
    
    plt.tight_layout()
    plt.show()
    
    # Display material properties
    print("="*70)
    print("Comparison of Mechanical Properties of Materials")
    print("="*70)
    
    for material, label in zip(materials, labels):
        _, _, props = stress_strain_curve(material)
        print(f"\n[{label}]")
        print(f"Material type: {props['type']}")
        print(f"Young's modulus E = {props['E']/1e3:.1f} GPa")
        if props['yield_stress'] is not None:
            print(f"Yield strength σ_y = {props['yield_stress']:.1f} MPa")
        print(f"Tensile strength UTS = {props['UTS']:.1f} MPa")
        print(f"Fracture strain ε_f = {props['fracture_strain']*100:.2f} %")
    
    print("\n" + "="*70)
    print("What the stress-strain curve tells us:")
    print("- Slope of elastic region → Young's modulus (material stiffness)")
    print("- Yield point → stress at which plastic deformation begins (basis for design strength)")
    print("- Tensile strength → maximum stress the material can withstand")
    print("- Fracture strain → measure of ductility (larger means more ductile)")
    print("- Area under the curve → energy absorbed until fracture (toughness)")
    

**Explanation** : The stress-strain curve is the most important graph describing a material's mechanical properties. Ductile materials (steel, aluminum) deform plastically after yielding and elongate significantly before fracturing. Brittle materials (glass) have no yield point and fracture with almost no elongation.

* * *

### Hardness

**Hardness** is the resistance of a material's surface to indentation, serving as an index of how hard the material is.

**Major hardness testing methods** :

  1. **Vickers Hardness (HV)**
     * Uses a diamond square-pyramid indenter
     * Load range: 1 gf to 120 kgf
     * Applicable to virtually all materials
     * $HV = 1.854 \times \frac{F}{d^2}$ (F: load [kgf], d: indentation diagonal length [mm])
  2. **Brinell Hardness (HB)**
     * Uses a carbide ball (tungsten carbide ball)
     * Large loads (500 to 3000 kgf)
     * Applied to materials with coarse microstructures such as castings
     * $HB = \frac{2F}{\pi D(D - \sqrt{D^2 - d^2})}$
  3. **Rockwell Hardness (HR)**
     * Measured from the depth of indenter penetration
     * Enables rapid measurement
     * Many scales (HRA, HRB, HRC, etc.)
     * HRC: used for hardened steels

Material | Vickers Hardness (HV) | Brinell Hardness (HB) | Rockwell Hardness (HRC)  
---|---|---|---  
**Mild steel** | 120-140 | 120-140 | -  
**Hardened steel** | 600-800 | - | 55-65  
**Stainless steel** | 150-200 | 150-200 | 20-30  
**Aluminum** | 20-30 | 20-30 | -  
**Copper** | 40-60 | 40-60 | -  
**Cemented carbide** | 1400-1800 | - | -  
**Diamond** | 10000 | - | -  
  
### Code Example 2: Mechanical Property Calculator (Young's Modulus, Yield Strength, Tensile Strength)

A tool that calculates mechanical properties from tensile test data.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class MechanicalPropertyCalculator:
        """
        Class for calculating mechanical properties from tensile test data
        """
    
        def __init__(self, force_data, length_data, original_length, cross_section_area):
            """
            Parameters:
            force_data: array of load data (N)
            length_data: array of length data (mm)
            original_length: initial gauge length (mm)
            cross_section_area: cross-sectional area (mm²)
            """
            self.force = np.array(force_data)  # N
            self.length = np.array(length_data)  # mm
            self.L0 = original_length  # mm
            self.A0 = cross_section_area  # mm²
    
            # Compute stress and strain
            self.stress = self.force / self.A0  # MPa (N/mm² = MPa)
            self.strain = (self.length - self.L0) / self.L0  # dimensionless
    
        def calculate_youngs_modulus(self, elastic_range=(0, 0.002)):
            """
            Calculate Young's modulus (slope of the elastic region)
    
            Parameters:
            elastic_range: strain range of the elastic region (tuple)
    
            Returns:
            E: Young's modulus (MPa)
            """
            # Extract data in the elastic region
            mask = (self.strain >= elastic_range[0]) & (self.strain <= elastic_range[1])
            strain_elastic = self.strain[mask]
            stress_elastic = self.stress[mask]
    
            # Linear fit (least squares)
            # Find the slope E of stress = E * strain
            E = np.polyfit(strain_elastic, stress_elastic, 1)[0]
    
            return E
    
        def calculate_yield_strength(self, offset=0.002):
            """
            Calculate the 0.2% proof stress (yield strength)
    
            Parameters:
            offset: offset strain (default 0.2% = 0.002)
    
            Returns:
            yield_strength: yield strength (MPa)
            """
            # Compute Young's modulus
            E = self.calculate_youngs_modulus()
    
            # Create the offset line (shifted by the strain offset)
            offset_stress = E * (self.strain - offset)
    
            # Find the intersection of the stress-strain curve and the offset line
            # The point where the difference is smallest is taken as the yield point
            diff = np.abs(self.stress - offset_stress)
            yield_idx = np.argmin(diff[self.strain > offset])
    
            # Adjust the index within the range beyond the offset
            yield_idx = np.where(self.strain > offset)[0][yield_idx]
            yield_strength = self.stress[yield_idx]
    
            return yield_strength
    
        def calculate_ultimate_tensile_strength(self):
            """
            Calculate the tensile strength (maximum stress)
    
            Returns:
            UTS: tensile strength (MPa)
            """
            UTS = np.max(self.stress)
            return UTS
    
        def calculate_elongation(self):
            """
            Calculate the elongation at break
    
            Returns:
            elongation: elongation at break (%)
            """
            max_strain = np.max(self.strain)
            elongation = max_strain * 100  # expressed in percent
            return elongation
    
        def plot_results(self):
            """
            Plot the stress-strain curve and calculated results
            """
            fig, ax = plt.subplots(figsize=(12, 7))
    
            # Plot the stress-strain curve
            ax.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='Experimental data')
    
            # Young's modulus line (elastic region)
            E = self.calculate_youngs_modulus()
            elastic_strain = np.linspace(0, 0.002, 50)
            elastic_stress = E * elastic_strain
            ax.plot(elastic_strain * 100, elastic_stress, 'r--', linewidth=2,
                   label=f"Young's modulus E = {E/1e3:.1f} GPa")
    
            # Mark the yield strength
            yield_strength = self.calculate_yield_strength()
            yield_idx = np.argmin(np.abs(self.stress - yield_strength))
            ax.plot(self.strain[yield_idx] * 100, yield_strength, 'go',
                   markersize=12, label=f'Yield strength = {yield_strength:.1f} MPa',
                   markeredgecolor='black', markeredgewidth=1.5)
    
            # Mark the tensile strength
            UTS = self.calculate_ultimate_tensile_strength()
            uts_idx = np.argmax(self.stress)
            ax.plot(self.strain[uts_idx] * 100, UTS, 'rs',
                   markersize=12, label=f'Tensile strength UTS = {UTS:.1f} MPa',
                   markeredgecolor='black', markeredgewidth=1.5)
    
            # Axis labels and title
            ax.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
            ax.set_title('Tensile Test Results and Calculated Mechanical Properties', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def print_summary(self):
            """
            Display a summary of the calculated results
            """
            E = self.calculate_youngs_modulus()
            yield_strength = self.calculate_yield_strength()
            UTS = self.calculate_ultimate_tensile_strength()
            elongation = self.calculate_elongation()
    
            print("="*70)
            print("Calculated Mechanical Properties")
            print("="*70)
            print(f"\nSpecimen information:")
            print(f"  Initial gauge length L₀ = {self.L0:.2f} mm")
            print(f"  Cross-sectional area A₀ = {self.A0:.2f} mm²")
    
            print(f"\nCalculated mechanical properties:")
            print(f"  Young's modulus E = {E/1e3:.2f} GPa ({E:.0f} MPa)")
            print(f"  Yield strength σ_y = {yield_strength:.2f} MPa")
            print(f"  Tensile strength UTS = {UTS:.2f} MPa")
            print(f"  Elongation at break = {elongation:.2f} %")
    
            # Estimate the material class
            if elongation > 15:
                material_type = "Ductile material (copper, aluminum, etc.)"
            elif elongation > 5:
                material_type = "Moderately ductile material (steel, etc.)"
            else:
                material_type = "Brittle material (glass, ceramics, etc.)"
    
            print(f"\nEstimated material type: {material_type}")
    
    
    # Simulate actual tensile test data (example for steel)
    # In practice, use data obtained from experiments
    np.random.seed(42)
    
    # Simulation parameters
    L0 = 50.0  # Initial gauge length (mm)
    A0 = 78.5  # Cross-sectional area (mm², circular cross-section of 10 mm diameter)
    E_actual = 200e3  # Actual Young's modulus (MPa = 200 GPa)
    yield_stress_actual = 250  # Actual yield strength (MPa)
    
    # Strain data (from 0% to fracture)
    strain_data = np.concatenate([
        np.linspace(0, 0.002, 50),  # Elastic region (0 to 0.2%)
        np.linspace(0.002, 0.20, 200)  # Plastic region (0.2% to 20%)
    ])
    
    # Compute stress data (stress-strain relationship)
    stress_data = np.zeros_like(strain_data)
    for i, strain in enumerate(strain_data):
        if strain <= 0.00125:  # Elastic region
            stress_data[i] = E_actual * strain
        else:  # Plastic region (includes work hardening)
            yield_strain = yield_stress_actual / E_actual
            stress_data[i] = yield_stress_actual + \
                            (400 - yield_stress_actual) * (1 - np.exp(-8 * (strain - yield_strain)))
    
    # Add noise (to mimic real measurements)
    stress_data += np.random.normal(0, 2, len(stress_data))
    
    # Compute load data (stress = load / cross-sectional area)
    force_data = stress_data * A0  # N
    
    # Compute length data (strain = (L - L0) / L0)
    length_data = L0 * (1 + strain_data)  # mm
    
    # Initialize the calculator
    calc = MechanicalPropertyCalculator(force_data, length_data, L0, A0)
    
    # Display the results
    calc.print_summary()
    calc.plot_results()
    
    print("\n" + "="*70)
    print("Why mechanical properties matter:")
    print("- Design: use below the yield strength (with a safety factor)")
    print("- Material selection: balance of strength and ductility")
    print("- Quality control: verify material quality with tensile tests")
    

**Explanation** : From tensile test data, we can calculate Young's modulus, yield strength, tensile strength, and elongation at break. These values are essential information for material design and selection.

### Code Example 3: Hardness Conversion Tool (Vickers ↔ Brinell ↔ Rockwell)

A tool that converts between different hardness scales.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class HardnessConverter:
        """
        Class for converting between different hardness scales
        Uses approximate formulas (errors vary by material)
        """
    
        @staticmethod
        def vickers_to_brinell(HV):
            """
            Convert Vickers hardness to Brinell hardness
    
            Parameters:
            HV: Vickers hardness
    
            Returns:
            HB: Brinell hardness
            """
            # Empirical formula (usable in the range where HV ≈ HB)
            # Strictly, the relationship is about HV = 1.05 * HB
            HB = HV / 1.05
            return HB
    
        @staticmethod
        def vickers_to_rockwell_c(HV):
            """
            Convert Vickers hardness to Rockwell C hardness
    
            Parameters:
            HV: Vickers hardness
    
            Returns:
            HRC: Rockwell C hardness
            """
            # Empirical formula (approximation for steels)
            # Valid in the range HV > 200
            if HV < 200:
                return None  # Out of applicable range
    
            # Approximate formula: HRC = a * log(HV) + b
            # Empirical formula derived from actual data
            HRC = 68.5 - 1000 / HV
    
            # Limit HRC range (20-70)
            HRC = np.clip(HRC, 20, 70)
    
            return HRC
    
        @staticmethod
        def brinell_to_vickers(HB):
            """
            Convert Brinell hardness to Vickers hardness
    
            Parameters:
            HB: Brinell hardness
    
            Returns:
            HV: Vickers hardness
            """
            HV = HB * 1.05
            return HV
    
        @staticmethod
        def rockwell_c_to_vickers(HRC):
            """
            Convert Rockwell C hardness to Vickers hardness
    
            Parameters:
            HRC: Rockwell C hardness
    
            Returns:
            HV: Vickers hardness
            """
            # Inverse calculation (approximate)
            HV = 1000 / (68.5 - HRC)
            return HV
    
        @staticmethod
        def estimate_tensile_strength(HV):
            """
            Estimate tensile strength from Vickers hardness
    
            Parameters:
            HV: Vickers hardness
    
            Returns:
            UTS: estimated tensile strength (MPa)
            """
            # Empirical formula (steels): UTS ≈ 3.3 * HV
            UTS = 3.3 * HV
            return UTS
    
    
    # Example usage of the conversion tool
    converter = HardnessConverter()
    
    print("="*70)
    print("Hardness Conversion Tool")
    print("="*70)
    
    # Test data (several materials)
    materials = [
        {'name': 'Mild steel', 'HV': 130},
        {'name': 'Stainless steel', 'HV': 180},
        {'name': 'Hardened steel (low-temp tempered)', 'HV': 600},
        {'name': 'Hardened steel (high-temp tempered)', 'HV': 400},
        {'name': 'Tool steel', 'HV': 750},
    ]
    
    print("\nHardness conversion table:")
    print("-" * 70)
    print(f"{'Material':<36} {'HV':>8} {'HB':>8} {'HRC':>8} {'Est. UTS(MPa)':>15}")
    print("-" * 70)
    
    for mat in materials:
        HV = mat['HV']
        HB = converter.vickers_to_brinell(HV)
        HRC = converter.vickers_to_rockwell_c(HV)
        UTS = converter.estimate_tensile_strength(HV)
    
        if HRC is not None:
            print(f"{mat['name']:<36} {HV:>8.0f} {HB:>8.0f} {HRC:>8.1f} {UTS:>15.0f}")
        else:
            print(f"{mat['name']:<36} {HV:>8.0f} {HB:>8.0f} {'N/A':>8} {UTS:>15.0f}")
    
    # Plot the relationship between hardness and tensile strength
    HV_range = np.linspace(100, 800, 100)
    UTS_range = converter.estimate_tensile_strength(HV_range)
    HB_range = converter.vickers_to_brinell(HV_range)
    HRC_range = np.array([converter.vickers_to_rockwell_c(hv) for hv in HV_range])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Conversion relationships between hardness scales
    ax1 = axes[0]
    ax1.plot(HV_range, HV_range, 'b-', linewidth=2, label='HV')
    ax1.plot(HV_range, HB_range, 'r--', linewidth=2, label='HB')
    ax1.set_xlabel('Vickers hardness HV', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hardness value', fontsize=12, fontweight='bold')
    ax1.set_title('Hardness Scale Conversion (HV ↔ HB)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Relationship between hardness and tensile strength
    ax2 = axes[1]
    ax2.plot(HV_range, UTS_range, 'g-', linewidth=2.5)
    ax2.set_xlabel('Vickers hardness HV', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Estimated tensile strength (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('Estimating Tensile Strength from Hardness\n(Empirical formula: UTS ≈ 3.3 × HV)',
                 fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Plot the data points
    for mat in materials:
        HV = mat['HV']
        UTS = converter.estimate_tensile_strength(HV)
        ax2.plot(HV, UTS, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print a comparison of hardness testing methods
    print("\n" + "="*70)
    print("Comparison of Hardness Testing Methods")
    print("="*70)
    print("\n[Vickers Hardness (HV)]")
    print("- Indenter: diamond square pyramid (136° face angle)")
    print("- Load: 1 gf to 120 kgf (notation by test force: HV0.1, HV10, etc.)")
    print("- Features: applicable to all materials, wide measurement range")
    print("- Uses: thin sheets, surface-treated layers, small parts")
    
    print("\n[Brinell Hardness (HB)]")
    print("- Indenter: carbide ball (diameter 2.5 mm, 5 mm, 10 mm)")
    print("- Load: 500 to 3000 kgf")
    print("- Features: large indentation, measurable even on coarse microstructures")
    print("- Uses: castings, large parts, coarse-grained materials")
    
    print("\n[Rockwell Hardness (HRC)]")
    print("- Indenter: diamond cone (HRC), steel ball (HRB)")
    print("- Load: 60 kgf (HRA), 100 kgf (HRB), 150 kgf (HRC)")
    print("- Features: measured by penetration depth, rapid measurement possible")
    print("- Uses: hardened steels (HRC), mild steel and non-ferrous metals (HRB)")
    
    print("\n" + "="*70)
    print("Relationship between hardness and tensile strength:")
    print("- Rule of thumb: UTS (MPa) ≈ 3.3 × HV (for steels)")
    print("- Strength can be estimated simply from hardness (non-destructive)")
    print("- Caution: the coefficient varies by material type (about 3.0 to 3.5)")
    

**Explanation** : There are several hardness testing methods, chosen according to the measurement purpose and material. Vickers hardness is the most versatile, Brinell hardness suits large parts, and Rockwell hardness suits rapid measurement. Tensile strength can also be estimated from hardness.

* * *

## 4.2 Electrical Properties (Conductivity, Semiconductivity, Insulation)

### Electrical Conductivity and Resistivity

**Electrical Conductivity (σ)** expresses how easily current flows:

$$\sigma = \frac{1}{\rho}$$

where $\rho$ is the **resistivity** (unit: Ω·m).

**Classification of materials** :

  * **Conductor** : $\rho < 10^{-5}$ Ω·m (metals)
  * **Semiconductor** : $10^{-5} < \rho < 10^{7}$ Ω·m (Si, Ge)
  * **Insulator** : $\rho > 10^{7}$ Ω·m (glass, ceramics, polymers)

Material | Resistivity (Ω·m, 20°C) | Classification  
---|---|---  
**Silver (Ag)** | 1.59 × 10⁻⁸ | Conductor  
**Copper (Cu)** | 1.68 × 10⁻⁸ | Conductor  
**Gold (Au)** | 2.44 × 10⁻⁸ | Conductor  
**Aluminum (Al)** | 2.82 × 10⁻⁸ | Conductor  
**Germanium (Ge)** | 4.6 × 10⁻¹ | Semiconductor  
**Silicon (Si)** | 6.4 × 10² | Semiconductor  
**Glass (SiO₂)** | 10¹⁰ - 10¹⁴ | Insulator  
**Polyethylene** | 10¹⁶ | Insulator  
  
### Band Structure and Electrical Conduction

**Band theory** describes the electronic states of a material in terms of energy bands:

  * **Valence Band** : band filled with electrons
  * **Conduction Band** : empty band (where electrons can move freely)
  * **Band Gap (Eg)** : energy difference between the valence band and the conduction band

**Material classification by band gap** :

  * **Metal (Conductor)** : Eg = 0 (valence and conduction bands overlap)
  * **Semiconductor** : 0 < Eg < 3 eV (excitation possible at room temperature)
  * **Insulator** : Eg > 3 eV (excitation difficult)

Semiconductor Material | Band Gap Eg (eV, 300K) | Applications  
---|---|---  
**Si** | 1.12 | Integrated circuits, solar cells  
**Ge** | 0.66 | Infrared detectors  
**GaAs** | 1.42 | High-speed devices, LEDs  
**GaN** | 3.44 | Blue LEDs, power devices  
**InP** | 1.35 | Optical communication devices  
**SiC** | 3.26 | High-temperature, high-voltage devices  
  
### Code Example 4: Visualizing the Relationship Between Band Gap and Electrical Conductivity

We visualize the relationship between band gap and electrical conductivity.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    k_B = 8.617e-5  # Boltzmann constant (eV/K)
    
    def intrinsic_carrier_concentration(Eg, T=300):
        """
        Calculate the intrinsic carrier concentration
    
        Parameters:
        Eg: band gap (eV)
        T: temperature (K)
    
        Returns:
        n_i: intrinsic carrier concentration (cm⁻³)
        """
        # Simplified approximation
        # n_i ∝ exp(-Eg / 2k_B T)
        # Strictly, n_i = sqrt(N_c * N_v) * exp(-Eg / 2k_B T)
        # Here we compute relative values
        n_i = 1e19 * np.exp(-Eg / (2 * k_B * T))
        return n_i
    
    def electrical_conductivity(n, mu=1000):
        """
        Calculate the electrical conductivity
    
        Parameters:
        n: carrier concentration (cm⁻³)
        mu: mobility (cm²/V·s)
    
        Returns:
        sigma: electrical conductivity (S/cm)
        """
        q = 1.602e-19  # Elementary charge (C)
        sigma = q * n * mu  # S/cm
        return sigma
    
    # Band gap range (0 to 5 eV)
    Eg_range = np.linspace(0.1, 5, 100)
    
    # Compute carrier concentration and conductivity for each band gap
    n_i_range = np.array([intrinsic_carrier_concentration(Eg) for Eg in Eg_range])
    sigma_range = electrical_conductivity(n_i_range)
    
    # Compute resistivity
    rho_range = 1 / sigma_range  # Ω·cm
    
    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Band gap vs carrier concentration
    ax1 = axes[0]
    ax1.semilogy(Eg_range, n_i_range, 'b-', linewidth=2.5)
    ax1.set_xlabel('Band gap $E_g$ (eV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Intrinsic carrier concentration $n_i$ (cm⁻³)', fontsize=12, fontweight='bold')
    ax1.set_title('Band Gap vs Carrier Concentration', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, which='both')
    
    # Plot materials
    materials_bandgap = [
        ('Ge', 0.66), ('Si', 1.12), ('GaAs', 1.42),
        ('InP', 1.35), ('SiC', 3.26), ('GaN', 3.44)
    ]
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        ax1.plot(Eg, n_i, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax1.annotate(name, xy=(Eg, n_i), xytext=(Eg+0.1, n_i*2),
                    fontsize=9, ha='left')
    
    # Band gap vs resistivity
    ax2 = axes[1]
    ax2.semilogy(Eg_range, rho_range, 'g-', linewidth=2.5)
    ax2.set_xlabel('Band gap $E_g$ (eV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resistivity $\\rho$ (Ω·cm)', fontsize=12, fontweight='bold')
    ax2.set_title('Band Gap vs Resistivity', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, which='both')
    
    # Shade the material classification regions
    ax2.axhspan(1e-8, 1e-5, alpha=0.2, color='blue', label='Conductor region')
    ax2.axhspan(1e-5, 1e7, alpha=0.2, color='yellow', label='Semiconductor region')
    ax2.axhspan(1e7, 1e20, alpha=0.2, color='red', label='Insulator region')
    ax2.legend(fontsize=10, loc='upper left')
    
    # Plot materials
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        sigma = electrical_conductivity(n_i)
        rho = 1 / sigma
        ax2.plot(Eg, rho, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    plt.show()
    
    # Display material properties
    print("="*70)
    print("Band Gaps and Electrical Properties of Semiconductor Materials")
    print("="*70)
    
    print(f"\n{'Material':<10} {'Eg(eV)':>10} {'n_i(cm⁻³)':>15} {'ρ(Ω·cm)':>15} {'Class':<15}")
    print("-" * 70)
    
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        sigma = electrical_conductivity(n_i)
        rho = 1 / sigma
    
        if rho < 1e-5:
            classification = "Conductor"
        elif rho < 1e7:
            classification = "Semiconductor"
        else:
            classification = "Insulator"
    
        print(f"{name:<10} {Eg:>10.2f} {n_i:>15.2e} {rho:>15.2e} {classification:<15}")
    
    print("\n" + "="*70)
    print("Why the band gap matters:")
    print("- Small Eg → high carrier concentration → high electrical conductivity")
    print("- Large Eg → low carrier concentration → high insulation")
    print("- Si (Eg=1.12eV): the most important semiconductor material (moderate conductivity at room temperature)")
    print("- GaN (Eg=3.44eV): wide-band-gap semiconductor (high-temperature, high-voltage operation)")
    

**Explanation** : The smaller the band gap, the higher the carrier concentration at room temperature and the greater the electrical conductivity (the lower the resistivity). The band gap of a semiconductor is a key parameter that determines the material's applications.

### Code Example 5: Plotting the Temperature Dependence of Resistivity (Metal vs Semiconductor)

We visualize how the temperature dependence of resistivity is opposite for metals and semiconductors.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def resistivity_metal(T, rho_0=1.68e-8, alpha=0.0039):
        """
        Resistivity of a metal (temperature dependence)
    
        Parameters:
        T: temperature (K)
        rho_0: resistivity at the reference temperature (273 K) (Ω·m)
        alpha: temperature coefficient (1/K)
    
        Returns:
        rho: resistivity (Ω·m)
        """
        T_0 = 273  # Reference temperature (K)
        rho = rho_0 * (1 + alpha * (T - T_0))
        return rho
    
    def resistivity_semiconductor(T, Eg=1.12, rho_room=640):
        """
        Resistivity of a semiconductor (temperature dependence)
    
        Parameters:
        T: temperature (K)
        Eg: band gap (eV)
        rho_room: resistivity at room temperature (300 K) (Ω·m)
    
        Returns:
        rho: resistivity (Ω·m)
        """
        k_B = 8.617e-5  # Boltzmann constant (eV/K)
        T_room = 300  # Room temperature (K)
    
        # The resistivity of an intrinsic semiconductor is proportional to exp(Eg / 2k_B T)
        rho = rho_room * np.exp(Eg / (2 * k_B) * (1/T - 1/T_room))
        return rho
    
    # Temperature range (200 K to 500 K)
    T_range = np.linspace(200, 500, 100)
    
    # Resistivity of a metal (copper)
    rho_metal = resistivity_metal(T_range, rho_0=1.68e-8, alpha=0.0039)
    
    # Resistivity of a semiconductor (silicon)
    rho_si = resistivity_semiconductor(T_range, Eg=1.12, rho_room=640)
    
    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    ax1 = axes[0]
    ax1.plot(T_range - 273, rho_metal * 1e8, 'b-', linewidth=2.5, label='Metal (Cu)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(T_range - 273, rho_si, 'r--', linewidth=2.5, label='Semiconductor (Si)')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Resistivity (metal, 10⁻⁸ Ω·m)', fontsize=11, fontweight='bold', color='b')
    ax1_twin.set_ylabel('Resistivity (semiconductor, Ω·m)', fontsize=11, fontweight='bold', color='r')
    ax1.set_title('Temperature Dependence of Resistivity (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(alpha=0.3)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    # Logarithmic scale
    ax2 = axes[1]
    ax2.semilogy(T_range - 273, rho_metal, 'b-', linewidth=2.5, label='Metal (Cu)')
    ax2.semilogy(T_range - 273, rho_si, 'r--', linewidth=2.5, label='Semiconductor (Si)')
    ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resistivity (Ω·m)', fontsize=12, fontweight='bold')
    ax2.set_title('Temperature Dependence of Resistivity (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display temperature coefficients
    print("="*70)
    print("Temperature Dependence of Resistivity")
    print("="*70)
    
    # Compute values at specific temperatures
    temps_celsius = [0, 25, 100, 200]
    temps_kelvin = [t + 273 for t in temps_celsius]
    
    print("\n[Resistivity of a metal (copper)]")
    print(f"{'Temp(°C)':>10} {'Resistivity(10⁻⁸ Ω·m)':>25} {'Change(%)':>15}")
    print("-" * 70)
    rho_ref = resistivity_metal(273)
    for T_c, T_k in zip(temps_celsius, temps_kelvin):
        rho = resistivity_metal(T_k)
        change = ((rho - rho_ref) / rho_ref) * 100
        print(f"{T_c:>10} {rho*1e8:>25.4f} {change:>15.2f}")
    
    print("\n[Resistivity of a semiconductor (silicon)]")
    print(f"{'Temp(°C)':>10} {'Resistivity(Ω·m)':>25} {'Change(%)':>15}")
    print("-" * 70)
    rho_ref = resistivity_semiconductor(300)
    for T_c, T_k in zip(temps_celsius, temps_kelvin):
        rho = resistivity_semiconductor(T_k)
        change = ((rho - rho_ref) / rho_ref) * 100
        print(f"{T_c:>10} {rho:>25.2e} {change:>15.2f}")
    
    print("\n" + "="*70)
    print("Differences in temperature dependence:")
    print("\n[Metals]")
    print("- Temperature rise → resistivity increases (positive temperature coefficient)")
    print("- Reason: lattice vibrations grow, increasing electron scattering")
    print("- Temperature coefficient α ≈ +0.4% / K (for copper)")
    print("- Applications: resistance thermometers (e.g., platinum resistance thermometers)")
    
    print("\n[Semiconductors]")
    print("- Temperature rise → resistivity decreases (negative temperature coefficient)")
    print("- Reason: thermal excitation increases the carrier concentration")
    print("- The temperature coefficient is negative and large (on the order of -several %/K)")
    print("- Applications: thermistors (temperature sensors)")
    

**Explanation** : In metals, resistivity increases with rising temperature, whereas in semiconductors it decreases. This reflects the different mechanisms of electrical conduction: in metals, scattering by lattice vibrations dominates, while in semiconductors, the increase in carrier concentration by thermal excitation dominates.

* * *

## 4.3 Thermal Properties (Heat Conduction, Thermal Expansion)

### Thermal Conductivity

**Thermal Conductivity (κ)** expresses how easily heat is transferred:

$$q = -\kappa \nabla T$$

where $q$ is the heat flux (W/m²) and $\nabla T$ is the temperature gradient (K/m).

**Classification of materials** :

  * **Metals** : κ = 50-400 W/(m·K) (high thermal conductivity)
  * **Ceramics** : κ = 1-50 W/(m·K)
  * **Polymers** : κ = 0.1-0.5 W/(m·K) (low thermal conductivity)

### Coefficient of Linear Thermal Expansion

The **Coefficient of Thermal Expansion (CTE, α)** is the rate of change in length with respect to temperature:

$$\alpha = \frac{1}{L} \frac{dL}{dT}$$

The unit is 1/K or ppm/K (10⁻⁶/K).

### Specific Heat

The **Specific Heat Capacity (c)** is the amount of heat required to raise the temperature of a unit mass of a substance by 1 K:

$$Q = mc\Delta T$$

The unit is J/(kg·K).

### Code Example 6: Comparing Thermal Properties (Thermal Conductivity, Linear Expansion Coefficient, Specific Heat)

We compare the thermal properties of representative materials.
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Database of thermal properties of materials
    materials_thermal = {
        # Metals
        'Copper (Cu)': {
            'thermal_conductivity': 401,  # W/(m·K)
            'thermal_expansion': 16.5,    # ppm/K (10⁻⁶/K)
            'specific_heat': 385,         # J/(kg·K)
            'density': 8960,              # kg/m³
            'category': 'Metal'
        },
        'Aluminum (Al)': {
            'thermal_conductivity': 237,
            'thermal_expansion': 23.1,
            'specific_heat': 897,
            'density': 2700,
            'category': 'Metal'
        },
        'Iron (Fe)': {
            'thermal_conductivity': 80,
            'thermal_expansion': 11.8,
            'specific_heat': 449,
            'density': 7874,
            'category': 'Metal'
        },
        'Stainless steel (SUS304)': {
            'thermal_conductivity': 16,
            'thermal_expansion': 17.3,
            'specific_heat': 500,
            'density': 8000,
            'category': 'Metal'
        },
        # Ceramics
        'Alumina (Al₂O₃)': {
            'thermal_conductivity': 30,
            'thermal_expansion': 8.1,
            'specific_heat': 775,
            'density': 3950,
            'category': 'Ceramic'
        },
        'Silicon nitride (Si₃N₄)': {
            'thermal_conductivity': 28,
            'thermal_expansion': 3.2,
            'specific_heat': 680,
            'density': 3200,
            'category': 'Ceramic'
        },
        'Glass (SiO₂)': {
            'thermal_conductivity': 1.4,
            'thermal_expansion': 0.55,
            'specific_heat': 750,
            'density': 2200,
            'category': 'Ceramic'
        },
        # Polymers
        'Polyethylene (PE)': {
            'thermal_conductivity': 0.42,
            'thermal_expansion': 100,
            'specific_heat': 2300,
            'density': 950,
            'category': 'Polymer'
        },
        'Polystyrene (PS)': {
            'thermal_conductivity': 0.13,
            'thermal_expansion': 70,
            'specific_heat': 1300,
            'density': 1050,
            'category': 'Polymer'
        }
    }
    
    # Organize the data
    materials = list(materials_thermal.keys())
    thermal_cond = [materials_thermal[m]['thermal_conductivity'] for m in materials]
    thermal_exp = [materials_thermal[m]['thermal_expansion'] for m in materials]
    specific_heat = [materials_thermal[m]['specific_heat'] for m in materials]
    categories = [materials_thermal[m]['category'] for m in materials]
    
    # Color-code by category
    color_map = {'Metal': '#1f77b4', 'Ceramic': '#ff7f0e', 'Polymer': '#2ca02c'}
    colors = [color_map[cat] for cat in categories]
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Comparison of thermal conductivity
    ax1 = axes[0, 0]
    y_pos = np.arange(len(materials))
    bars1 = ax1.barh(y_pos, thermal_cond, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(materials, fontsize=9)
    ax1.set_xlabel('Thermal conductivity κ (W/(m·K))', fontsize=11, fontweight='bold')
    ax1.set_title('Comparison of Thermal Conductivity', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xscale('log')
    
    # Display values at the ends of the bars
    for i, (bar, val) in enumerate(zip(bars1, thermal_cond)):
        ax1.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=8)
    
    # Comparison of linear expansion coefficient
    ax2 = axes[0, 1]
    bars2 = ax2.barh(y_pos, thermal_exp, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(materials, fontsize=9)
    ax2.set_xlabel('Linear expansion coefficient α (ppm/K = 10⁻⁶/K)', fontsize=11, fontweight='bold')
    ax2.set_title('Comparison of Linear Expansion Coefficient', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xscale('log')
    
    for i, (bar, val) in enumerate(zip(bars2, thermal_exp)):
        ax2.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=8)
    
    # Comparison of specific heat
    ax3 = axes[1, 0]
    bars3 = ax3.barh(y_pos, specific_heat, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(materials, fontsize=9)
    ax3.set_xlabel('Specific heat c (J/(kg·K))', fontsize=11, fontweight='bold')
    ax3.set_title('Comparison of Specific Heat', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars3, specific_heat)):
        ax3.text(val + 50, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)
    
    # Scatter plot of thermal conductivity vs linear expansion coefficient
    ax4 = axes[1, 1]
    for cat in ['Metal', 'Ceramic', 'Polymer']:
        indices = [i for i, c in enumerate(categories) if c == cat]
        tc = [thermal_cond[i] for i in indices]
        te = [thermal_exp[i] for i in indices]
        ax4.scatter(tc, te, s=150, c=color_map[cat], label=cat,
                   edgecolors='black', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Thermal conductivity κ (W/(m·K))', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Linear expansion coefficient α (ppm/K)', fontsize=11, fontweight='bold')
    ax4.set_title('Thermal Conductivity vs Linear Expansion Coefficient', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, which='both')
    
    # Plot the material names
    for i, mat in enumerate(materials):
        ax4.annotate(mat.split(' (')[0],
                    xy=(thermal_cond[i], thermal_exp[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate thermal diffusivity
    print("="*70)
    print("Comparison of Thermal Properties of Materials")
    print("="*70)
    
    print(f"\n{'Material':<28} {'κ(W/m·K)':>12} {'α(ppm/K)':>12} {'c(J/kg·K)':>12} {'a(mm²/s)':>12}")
    print("-" * 70)
    
    for mat in materials:
        props = materials_thermal[mat]
        kappa = props['thermal_conductivity']
        alpha_exp = props['thermal_expansion']
        c = props['specific_heat']
        rho = props['density']
    
        # Compute thermal diffusivity a = κ / (ρ * c)
        thermal_diffusivity = kappa / (rho * c) * 1e6  # mm²/s
    
        print(f"{mat:<28} {kappa:>12.2f} {alpha_exp:>12.2f} {c:>12.0f} {thermal_diffusivity:>12.3f}")
    
    print("\n" + "="*70)
    print("Meaning and applications of thermal properties:")
    print("="*70)
    
    print("\n[Thermal conductivity κ]")
    print("- High → transfers heat quickly → heat sinks, heat dissipation materials")
    print("- Low → good thermal insulation → insulation and heat-retention materials")
    print("- Order: metals > ceramics > polymers")
    print("- Applications: copper (heat dissipation), stainless steel (insulation)")
    
    print("\n[Linear expansion coefficient α]")
    print("- High → expands/contracts greatly with temperature changes → thermal stress arises easily")
    print("- Low → high dimensional stability → suitable for precision instruments")
    print("- Order: polymers > metals > ceramics")
    print("- Applications: glass (low CTE), managing thermal stress in dissimilar-material joints")
    
    print("\n[Specific heat c]")
    print("- High → temperature changes slowly → heat storage materials")
    print("- Low → temperature changes quickly → fast thermal response")
    print("- Order: polymers > ceramics > metals (per unit mass)")
    print("- Applications: water (high specific heat, coolant), metals (low specific heat, cookware)")
    
    print("\n[Thermal diffusivity a = κ/(ρc)]")
    print("- The speed at which heat diffuses through a material")
    print("- High → the whole body quickly reaches a uniform temperature")
    print("- Highest in metals (copper, aluminum)")
    

**Explanation** : The thermal properties of materials are important for thermal management design. Metals have high thermal conductivity and suit heat dissipation, whereas polymers have low thermal conductivity and suit insulation. The linear expansion coefficient matters when considering thermal stress in joints between dissimilar materials.

* * *

## 4.4 Optical Properties (Transparency, Color)

### Transparency and Opacity

**Transparent** : visible light passes through with almost no absorption

  * Examples: glass, transparent polymers (PMMA, PC)
  * Condition: band gap > energy of visible light (about 1.8 to 3.1 eV)

**Translucent** : light passes through while being scattered

  * Examples: frosted glass, thin paper

**Opaque** : light is absorbed or reflected

  * Examples: metals, black materials
  * Metals: reflection by free electrons

### Color and Absorption Spectra

The **color** of a material arises when specific wavelengths of visible light are absorbed and the remainder is reflected or transmitted.

**Wavelength ranges of visible light** :

  * Violet: 380-450 nm
  * Blue: 450-495 nm
  * Green: 495-570 nm
  * Yellow: 570-590 nm
  * Orange: 590-620 nm
  * Red: 620-750 nm

**Complementary color relationship** : when a color is absorbed, its complementary color is seen

  * Absorb blue → appears orange
  * Absorb red → appears blue-green

### Refractive Index

The **Refractive Index (n)** is the ratio of the speed of light in vacuum to that in the material:

$$n = \frac{c}{v}$$

where $c$ is the speed of light in vacuum and $v$ is the speed of light in the material.

Material | Refractive Index (589 nm, D line) | Transparency  
---|---|---  
**Vacuum** | 1.0000 | -  
**Air** | 1.0003 | -  
**Water** | 1.333 | Transparent  
**Fused silica (SiO₂)** | 1.458 | Transparent  
**Soda-lime glass** | 1.52 | Transparent  
**PMMA (acrylic)** | 1.49 | Transparent  
**Polycarbonate (PC)** | 1.586 | Transparent  
**Diamond** | 2.417 | Transparent  
  
**Applications of optical properties** :

  * **Lenses** : high-refractive-index materials (optical glass, polymers)
  * **Optical fibers** : low-loss transparent materials (fused silica)
  * **Anti-reflection coatings** : exploit thin-film interference
  * **Colored materials** : absorption spectrum control by pigments and dyes
  * **Solar cells** : visible-light-absorbing materials (Si, GaAs, etc.)

> **Summary** : The properties of materials (mechanical, electrical, thermal, and optical) are deeply related to their atomic and crystal structures. To choose the right material for an application, it is essential to understand and compare these properties quantitatively.

* * *

## 4.5 Chapter Summary

### What We Learned

  1. **Mechanical properties**
     * Stress-strain curve: Young's modulus, yield strength, tensile strength, elongation at break
     * Differences between ductile and brittle materials
     * Hardness testing methods (Vickers, Brinell, Rockwell) and conversions
  2. **Electrical properties**
     * Classification into conductors, semiconductors, and insulators (by resistivity)
     * Relationship between band gap and electrical conductivity
     * Different temperature dependence of metals and semiconductors (positive vs negative temperature coefficient)
  3. **Thermal properties**
     * Thermal conductivity: metals > ceramics > polymers
     * Linear expansion coefficient: polymers > metals > ceramics
     * Meaning of specific heat and thermal diffusivity
  4. **Optical properties**
     * Condition for transparency (band gap > visible light energy)
     * Relationship between color and absorption spectra
     * Refractive index and optical applications

### Key Points

  * Material properties originate from **structure (atomic arrangement, crystal structure, chemical bonding)**
  * Mechanical properties are the most fundamental criteria for material selection
  * Electrical properties are explained by band structure
  * Thermal properties are essential for thermal management design
  * Python enables quantitative calculation and comparison of material properties

### To the Next Chapter

In Chapter 5, we will study **crystal structure visualization with Python** :

  * Introduction to pymatgen (a crystal structure library)
  * Reading CIF files and analyzing structures
  * Using the Materials Project database
  * Structural analysis of representative materials (Si, Fe, Al₂O₃)
  * An integrated workflow (structure → analysis → visualization → property prediction)
