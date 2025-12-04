---
title: "Chapter 4: Relationship between Material Properties and Structure"
chapter_title: "Chapter 4: Relationship between Material Properties and Structure"
subtitle: Understanding Mechanical, Electrical, Thermal, and Optical Properties
reading_time: 35-40 minutes
difficulty: Intermediate
code_examples: 6
---

Material properties are determined by atomic structure, crystal structure, and types of chemical bonding. In this chapter, we will learn about mechanical properties (strength, hardness, ductility), electrical properties (conductivity, semiconductivity), thermal properties (thermal conductivity, thermal expansion), and optical properties (transparency, color), and compute and visualize material characteristics using Python. 

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand stress-strain curves and evaluate mechanical properties of materials
  * ✅ Understand hardness measurement methods (Vickers, Brinell, Rockwell) and perform conversions
  * ✅ Explain the relationship between band structure and electrical conduction
  * ✅ Understand thermal properties (thermal conductivity, linear expansion coefficient, specific heat)
  * ✅ Understand the basics of optical properties (transparency, color, refractive index)
  * ✅ Compute, plot, and compare material properties using Python

* * *

## 4.1 Mechanical Properties (Strength, Hardness, Ductility)

### Stress and Strain

**Stress** is the force acting per unit area of material:

$$\sigma = \frac{F}{A}$$

where,

  * $\sigma$: stress (Pa = N/m² or MPa)
  * $F$: load (N)
  * $A$: cross-sectional area (m²)

**Strain** is the relative deformation of the material:

$$\varepsilon = \frac{\Delta L}{L_0}$$

where,

  * $\varepsilon$: strain (dimensionless)
  * $\Delta L$: elongation (m)
  * $L_0$: original length (m)

### Stress-Strain Curve

Materialwhen subjected to tensile load、Stress-Strain Curve 得 れ。this曲線fromMaterialthe mechanical properties can be determined。

**Main Regions** :

  1. **Elastic Region** : Linear region following Hooke's law. Returns to original shape when load is removed.
  2. **Yield Point** : Point of transition from elastic to plastic deformation. Evaluated by yield strength.
  3. **Plastic Region** : Region where permanent deformation occurs.
  4. **Ultimate Tensile Strength (UTS)** : Maximum stress the material can withstand.
  5. **Fracture** : Point where the material breaks.

**Young's Modulus (E)** : Slope of the elastic region

$$E = \frac{\sigma}{\varepsilon}$$

The unit is GPa (gigapascal). The larger the Young's modulus, the stiffer the material and the more resistant to deformation.

Material | Young's Modulus (GPa) | Yield strength (MPa) | Tensile strength (MPa) | Ductility  
---|---|---|---|---  
**Steel** | 200 | 250-400 | 400-550 | High  
**Aluminum (Al)** | 69 | 35-100 | 90-150 | High  
**Copper (Cu)** | 130 | 70 | 220 | High  
**Titanium (Ti)** | 116 | 140-500 | 240-550 | Medium  
**Glass (SiO₂)** | 70 | - | 50-100 | Brittle  
**Ceramics (Al₂O₃)** | 380 | - | 300-400 | Brittle  
  
### Ductility and Brittle

**DuctilityMaterial（Ductile）** ：plastic deformation large（金属）

  * Elongates significantly before fracture
  * elongation at fracture率 Highい（usually > 5%）
  * Examples: copper, aluminum, steel

**BrittleMaterial（Brittle）** ：plastic deformation small（ceramics、ガラス）

  * Fractures with little elongation
  * Low elongation at fracture (typically < 5%)
  * Examples: glass, ceramics, cast iron

### codeexample1: Stress-Strain Curve 作成 and plot（複数Materialcomparison）

鋼、Aluminum、ガラス Stress-Strain Curve 作成し、Material difference visualizationdo。
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 鋼、Aluminum、ガラス Stress-Strain Curve 作成し、Material difference v
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt # Stress-Strain Curve simulation def stress_strain_curve(material='steel'): """ Material Stress-Strain Curve シミュレート Parameters: material: 'steel', 'aluminum', 'glass'one of Returns: strain: strain array stress: stress array (MPa) """ if material == 'steel': # 鋼（DuctilityMaterial） E = 200e3 # Young's modulus (MPa) yield_stress = 250 # Yield strength (MPa) yield_strain = yield_stress / E # Yield strain uts = 400 # Tensile strength (MPa) fracture_strain = 0.25 # Fracture strain (25% elongation) # Elastic region (from 0 to yield point) strain_elastic = np.linspace(0, yield_strain, 100) stress_elastic = E * strain_elastic # Plastic region (yield to fracture) - considering work hardening strain_plastic = np.linspace(yield_strain, fracture_strain, 300) # Work hardening: stress increases but rate decreases stress_plastic = yield_stress + (uts - yield_stress) * \ (1 - np.exp(-10 * (strain_plastic - yield_strain))) # Softening after necking strain_necking = np.linspace(fracture_strain, fracture_strain + 0.05, 50) stress_necking = uts * np.exp(-20 * (strain_necking - fracture_strain)) strain = np.concatenate([strain_elastic, strain_plastic, strain_necking]) stress = np.concatenate([stress_elastic, stress_plastic, stress_necking]) properties = { 'E': E, 'yield_stress': yield_stress, 'UTS': uts, 'fracture_strain': fracture_strain + 0.05, 'type': 'DuctilityMaterial' } elif material == 'aluminum': # Aluminum（DuctilityMaterial、鋼 from soft） E = 69e3 # Young's modulus (MPa) yield_stress = 35 # Yield strength (MPa) yield_strain = yield_stress / E uts = 90 # Tensile strength (MPa) fracture_strain = 0.18 # 破断ひずみ（18%伸び） strain_elastic = np.linspace(0, yield_strain, 100) stress_elastic = E * strain_elastic strain_plastic = np.linspace(yield_strain, fracture_strain, 300) stress_plastic = yield_stress + (uts - yield_stress) * \ (1 - np.exp(-8 * (strain_plastic - yield_strain))) strain_necking = np.linspace(fracture_strain, fracture_strain + 0.04, 50) stress_necking = uts * np.exp(-15 * (strain_necking - fracture_strain)) strain = np.concatenate([strain_elastic, strain_plastic, strain_necking]) stress = np.concatenate([stress_elastic, stress_plastic, stress_necking]) properties = { 'E': E, 'yield_stress': yield_stress, 'UTS': uts, 'fracture_strain': fracture_strain + 0.04, 'type': 'DuctilityMaterial' } elif material == 'glass': # ガラス（BrittleMaterial） E = 70e3 # Young's modulus (MPa) fracture_stress = 70 # Fracture stress (MPa) fracture_strain = fracture_stress / E # Fracture strain (approx. 0.1%) # Elastic region only (linear until fracture) strain = np.linspace(0, fracture_strain, 200) stress = E * strain properties = { 'E': E, 'yield_stress': None, # No yield 'UTS': fracture_stress, 'fracture_strain': fracture_strain, 'type': 'BrittleMaterial' } else: raise ValueError("material 'steel', 'aluminum', 'glass' one of") return strain, stress, properties # Create plot fig, ax = plt.subplots(figsize=(12, 7)) materials = ['steel', 'aluminum', 'glass'] colors = ['#1f77b4', '#ff7f0e', '#d62728'] labels = ['Steel', 'Aluminum', 'Glass'] for material, color, label in zip(materials, colors, labels): # Stress-Strain Curve calculation strain, stress, props = stress_strain_curve(material) # convert strain to percentage display strain_percent = strain * 100 # plot ax.plot(strain_percent, stress, linewidth=2.5, color=color, label=label) # 降伏点 マーク（DuctilityMaterial み） if props['yield_stress'] is not None: yield_strain = props['yield_stress'] / props['E'] ax.plot(yield_strain * 100, props['yield_stress'], 'o', markersize=10, color=color, markeredgecolor='black', markeredgewidth=1.5) # Mark tensile strength if material!= 'glass': # DuctilityMaterial：UTS点 見つける uts_idx = np.argmax(stress) ax.plot(strain_percent[uts_idx], stress[uts_idx], 's', markersize=10, color=color, markeredgecolor='black', markeredgewidth=1.5) # Axis labels and title ax.set_xlabel('Strain (% = $\\varepsilon$ × 100)', fontsize=13, fontweight='bold') ax.set_ylabel('Stress (MPa = $\\sigma$)', fontsize=13, fontweight='bold') ax.set_title('Stress-Strain Curve comparison（DuctilityMaterial vs BrittleMaterial）', fontsize=14, fontweight='bold', pad=15) ax.legend(fontsize=11, loc='upper left') ax.grid(alpha=0.3, linestyle='--') ax.set_xlim(0, 30) ax.set_ylim(0, 450) # Add annotations ax.annotate('Yield Point', xy=(0.125, 250), xytext=(3, 350), arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=10, ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6)) ax.annotate('Tensile Strength\n(UTS)', xy=(12, 400), xytext=(15, 320), arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=10, ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6)) ax.annotate('Brittle破壊\n(ほぼ伸びなし)', xy=(0.1, 70), xytext=(2, 150), arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=10, ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.6)) plt.tight_layout() plt.show() # Material特性 display print("="*70) print("Material 機械的propertycomparison") print("="*70) for material, label in zip(materials, labels): _, _, props = stress_strain_curve(material) print(f"\n【{label}】") print(f"Materialtype: {props['type']}") print(f"Young's modulus E = {props['E']/1e3:.1f} GPa") if props['yield_stress'] is not None: print(f"Yield strength σ_y = {props['yield_stress']:.1f} MPa") print(f"Tensile strength UTS = {props['UTS']:.1f} MPa") print(f"Fracture strain ε_f = {props['fracture_strain']*100:.2f} %") print("\n" + "="*70) print("Stress-Strain Curvefrom分 るthing:") print("- Elastic region 傾き → ヤング率（Material 硬さ）") print("- Yield point → Stress at which plastic deformation begins (design strength criterion)") print("- tensile strength → Material maximum stress that can be withstood") print("- 破断ひずみ → Ductility 指標（largeほどDuctilityMaterial）") print("- Area under curve → Energy absorbed until fracture (toughness)") 

**解説** : Stress-Strain Curve Materialis the most important graph representing mechanical properties。DuctilityMaterial（鋼、Aluminum） 降伏後 plastic deformationし、elongates significantly before fracture。BrittleMaterial（ガラス） 降伏点 なく、Fractures with little elongationdo。

* * *

### Hardness

**硬度** 、Materialthe resistance when an indenter is pressed into the surface、Material 硬さ show指標 す。

**Main Hardness Measurement Methods** :

  1. **Vickers Hardness (HV)**
     * Uses diamond square pyramid indenter
     * Load range: 1gf to 120kgf
     * あ ゆるMaterial applicablepossible
     * $HV = 1.854 \times \frac{F}{d^2}$ (F: load [kgf], d: diagonal length of indentation [mm])
  2. **Brinell Hardness (HB)**
     * Uses carbide ball (tungsten carbide ball)
     * Large load (500 to 3000kgf)
     * 鋳物etc粗大組織 Material applicable
     * $HB = \frac{2F}{\pi D(D - \sqrt{D^2 - d^2})}$
  3. **Rockwell Hardness (HR)**
     * Measured from indentation depth
     * Rapid measurement possible
     * Multiple scales (HRA, HRB, HRC, etc.)
     * HRC: Used for hardened steel

Material | Vickers Hardness (HV) | Brinell Hardness (HB) | Rockwell Hardness (HRC)  
---|---|---|---  
**Mild steel** | 120-140 | 120-140 | -  
**Hardened steel** | 600-800 | - | 55-65  
**Stainless steel** | 150-200 | 150-200 | 20-30  
**Aluminum** | 20-30 | 20-30 | -  
**Copper** | 40-60 | 40-60 | -  
**Cemented carbide** | 1400-1800 | - | -  
**Diamond** | 10000 | - | -  
  
### Code Example 2: Mechanical Properties Calculator (Young's Modulus, Yield Strength, Tensile Strength)

Tool for calculating mechanical properties from tensile test data.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Tool for calculating mechanical properties from tensile test
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt class MechanicalPropertyCalculator: """ Class for calculating mechanical properties from tensile test data """ def __init__(self, force_data, length_data, original_length, cross_section_area): """ Parameters: force_data: Array of load data (N) length_data: Array of length data (mm) original_length: Initial gauge length (mm) cross_section_area: Cross-sectional area (mm²) """ self.force = np.array(force_data) # N self.length = np.array(length_data) # mm self.L0 = original_length # mm self.A0 = cross_section_area # mm² # Stress and Strain calculation self.stress = self.force / self.A0 # MPa（N/mm² = MPa） self.strain = (self.length - self.L0) / self.L0 # dimensionless def calculate_youngs_modulus(self, elastic_range=(0, 0.002)): """ Calculate Young's modulus (slope of elastic region) Parameters: elastic_range: Strain range of elastic region (tuple) Returns: E: Young's modulus (MPa) """ # Extract elastic region data mask = (self.strain >= elastic_range[0]) & (self.strain <= elastic_range[1]) strain_elastic = self.strain[mask] stress_elastic = self.stress[mask] # Linear fitting (least squares method) # Find slope E of stress = E * strain E = np.polyfit(strain_elastic, stress_elastic, 1)[0] return E def calculate_yield_strength(self, offset=0.002): """ Calculate 0.2% proof stress (yield strength) Parameters: offset: Offset strain (default 0.2% = 0.002) Returns: yield_strength: Yield strength (MPa) """ # Calculate Young's modulus E = self.calculate_youngs_modulus() # Create offset line (parallel shift by offset strain) offset_stress = E * (self.strain - offset) # Stress-Strain Curvefind the intersection with offset line # Point with minimum difference is yield point diff = np.abs(self.stress - offset_stress) yield_idx = np.argmin(diff[self.strain > offset]) # Adjust index for range after offset yield_idx = np.where(self.strain > offset)[0][yield_idx] yield_strength = self.stress[yield_idx] return yield_strength def calculate_ultimate_tensile_strength(self): """ Calculate tensile strength (maximum stress) Returns: UTS: Tensile strength (MPa) """ UTS = np.max(self.stress) return UTS def calculate_elongation(self): """ Calculate elongation at fracture Returns: elongation: Elongation at fracture (%) """ max_strain = np.max(self.strain) elongation = max_strain * 100 # Percent display return elongation def plot_results(self): """ Stress-Strain Curve and calculation結果 plot """ fig, ax = plt.subplots(figsize=(12, 7)) # Stress-Strain Curve plot ax.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='Experimental data') # Young's modulus line (elastic region) E = self.calculate_youngs_modulus() elastic_strain = np.linspace(0, 0.002, 50) elastic_stress = E * elastic_strain ax.plot(elastic_strain * 100, elastic_stress, 'r--', linewidth=2, label=f'Young's modulus E = {E/1e3:.1f} GPa') # Mark yield strength yield_strength = self.calculate_yield_strength() yield_idx = np.argmin(np.abs(self.stress - yield_strength)) ax.plot(self.strain[yield_idx] * 100, yield_strength, 'go', markersize=12, label=f'yield strength = {yield_strength:.1f} MPa', markeredgecolor='black', markeredgewidth=1.5) # Mark tensile strength UTS = self.calculate_ultimate_tensile_strength() uts_idx = np.argmax(self.stress) ax.plot(self.strain[uts_idx] * 100, UTS, 'rs', markersize=12, label=f'Tensile strength UTS = {UTS:.1f} MPa', markeredgecolor='black', markeredgewidth=1.5) # Axis labels and title ax.set_xlabel('Strain (%)', fontsize=13, fontweight='bold') ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold') ax.set_title('Tensile Test Results and Mechanical Properties Calculation', fontsize=14, fontweight='bold') ax.legend(fontsize=11, loc='lower right') ax.grid(alpha=0.3) plt.tight_layout() plt.show() def print_summary(self): """ Display summary of calculation results """ E = self.calculate_youngs_modulus() yield_strength = self.calculate_yield_strength() UTS = self.calculate_ultimate_tensile_strength() elongation = self.calculate_elongation() print("="*70) print("Calculation Results of Mechanical Properties") print("="*70) print(f"\nSpecimen information:") print(f" Initial gauge length L₀ = {self.L0:.2f} mm") print(f" Cross-sectional area A₀ = {self.A0:.2f} mm²") print(f"\nCalculated mechanical properties:") print(f" Young's modulus E = {E/1e3:.2f} GPa ({E:.0f} MPa)") print(f" Yield strength σ_y = {yield_strength:.2f} MPa") print(f" Tensile strength UTS = {UTS:.2f} MPa") print(f" Elongation at fracture = {elongation:.2f} %") # MaterialClassification 推定 if elongation > 15: material_type = "DuctilityMaterial（Copper、Aluminumetc）" elif elongation > 5: material_type = "Medium程度 DuctilityMaterial（鋼etc）" else: material_type = "BrittleMaterial（ガラス、ceramicsetc）" print(f"\n推定されるMaterialtype: {material_type}") # Simulate actual tensile test data (steel example) # In practice, use data obtained from experiments np.random.seed(42) # Simulation parameters L0 = 50.0 # Initial gauge length (mm) A0 = 78.5 # Cross-sectional area (mm², circular section with 10mm diameter) E_actual = 200e3 # Actual Young's modulus (MPa = 200 GPa) yield_stress_actual = 250 # 実際 Yield strength (MPa) # Strain data (from 0% to fracture) strain_data = np.concatenate([np.linspace(0, 0.002, 50), # Elastic region (0 to 0.2%) np.linspace(0.002, 0.20, 200) # Plastic region (0.2% to 20%) ]) # Calculate stress data (stress-strain relationship) stress_data = np.zeros_like(strain_data) for i, strain in enumerate(strain_data): if strain <= 0.00125: # Elastic region stress_data[i] = E_actual * strain else: # Plastic region (considering work hardening) yield_strain = yield_stress_actual / E_actual stress_data[i] = yield_stress_actual + \ (400 - yield_stress_actual) * (1 - np.exp(-8 * (strain - yield_strain))) # Add noise (simulate actual measurement) stress_data += np.random.normal(0, 2, len(stress_data)) # Calculate load data (stress = load / cross-sectional area) force_data = stress_data * A0 # N # Calculate length data (strain = (L - L0) / L0) length_data = L0 * (1 + strain_data) # mm # Initialize calculator calc = MechanicalPropertyCalculator(force_data, length_data, L0, A0) # Display results calc.print_summary() calc.plot_results() print("\n" + "="*70) print("Importance of mechanical properties:") print("- Design: Use below yield strength (considering safety factor)") print("- Materialselection: 強度 and Ductility バランス") print("- 品質管理: 引張試験 Material品質 確認") 

**解説** : tensile test datafrom、ヤング率、yield strength、tensile strength、Calculate elongation at fracture き。these 値 Materialis essential information for design and selection。

### Code Example 3: Hardness Conversion Tool (Vickers ↔ Brinell ↔ Rockwell)

Tool for converting between different hardness scales.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Tool for converting between different hardness scales.
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt class HardnessConverter: """ Class for converting between different hardness scales 近似Equation use（Material by 誤差あ ） """ @staticmethod def vickers_to_brinell(HV): """ Convert Vickers hardness to Brinell hardness Parameters: HV: Vickers hardness Returns: HB: Brinell hardness """ # Empirical formula (applicable in HV ≈ HB range) # Strictly HV ≈ 1.05 * HB relationship HB = HV / 1.05 return HB @staticmethod def vickers_to_rockwell_c(HV): """ Vickers hardnessfromRockwell C hardness to 換算 Parameters: HV: Vickers hardness Returns: HRC: Rockwell C hardness """ # Empirical formula (approximation for steel) # Valid for HV > 200 range if HV < 200: return None # Outside applicable range # Approximation: HRC = a * log(HV) + b # Empirical formula derived from actual data HRC = 68.5 - 1000 / HV # Range limit for HRC (20-70) HRC = np.clip(HRC, 20, 70) return HRC @staticmethod def brinell_to_vickers(HB): """ Brinell hardnessfromVickers hardness to 換算 Parameters: HB: Brinell hardness Returns: HV: Vickers hardness """ HV = HB * 1.05 return HV @staticmethod def rockwell_c_to_vickers(HRC): """ Rockwell C hardnessfromVickers hardness to 換算 Parameters: HRC: Rockwell C hardness Returns: HV: Vickers hardness """ # Inverse calculation (approximation) HV = 1000 / (68.5 - HRC) return HV @staticmethod def estimate_tensile_strength(HV): """ Vickers hardnessfromtensile strength 推定 Parameters: HV: Vickers hardness Returns: UTS: 推定Tensile strength (MPa) """ # Empirical formula (steel): UTS ≈ 3.3 * HV UTS = 3.3 * HV return UTS # Usage example of conversion tool converter = HardnessConverter() print("="*70) print("Hardness Conversion Tool") print("="*70) # testdata（いくつ Material） materials = [{'name': 'Mild steel', 'HV': 130}, {'name': 'Stainless steel', 'HV': 180}, {'name': 'Hardened steel（Low温焼戻し）', 'HV': 600}, {'name': 'Hardened steel（High温焼戻し）', 'HV': 400}, {'name': 'Tool steel', 'HV': 750}, ] print("\nHardness conversion table:") print("-" * 70) print(f"{'Material':<20} {'HV':>8} {'HB':>8} {'HRC':>8} {'Estimated UTS (MPa)':>15}") print("-" * 70) for mat in materials: HV = mat['HV'] HB = converter.vickers_to_brinell(HV) HRC = converter.vickers_to_rockwell_c(HV) UTS = converter.estimate_tensile_strength(HV) if HRC is not None: print(f"{mat['name']:<20} {HV:>8.0f} {HB:>8.0f} {HRC:>8.1f} {UTS:>15.0f}") else: print(f"{mat['name']:<20} {HV:>8.0f} {HB:>8.0f} {'N/A':>8} {UTS:>15.0f}") # Plot relationship between hardness and tensile strength HV_range = np.linspace(100, 800, 100) UTS_range = converter.estimate_tensile_strength(HV_range) HB_range = converter.vickers_to_brinell(HV_range) HRC_range = np.array([converter.vickers_to_rockwell_c(hv) for hv in HV_range]) fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Conversion relationship between hardness scales ax1 = axes[0] ax1.plot(HV_range, HV_range, 'b-', linewidth=2, label='HV') ax1.plot(HV_range, HB_range, 'r--', linewidth=2, label='HB') ax1.set_xlabel('Vickers hardness HV', fontsize=12, fontweight='bold') ax1.set_ylabel('Hardness Value', fontsize=12, fontweight='bold') ax1.set_title('Hardness Scale Conversion (HV ↔ HB)', fontsize=13, fontweight='bold') ax1.legend(fontsize=11) ax1.grid(alpha=0.3) # Relationship between hardness and tensile strength ax2 = axes[1] ax2.plot(HV_range, UTS_range, 'g-', linewidth=2.5) ax2.set_xlabel('Vickers hardness HV', fontsize=12, fontweight='bold') ax2.set_ylabel('Estimated Tensile Strength (MPa)', fontsize=12, fontweight='bold') ax2.set_title('Estimate Tensile Strength from Hardness\n(Empirical Formula: UTS ≈ 3.3 × HV)', fontsize=13, fontweight='bold') ax2.grid(alpha=0.3) # Plot data points for mat in materials: HV = mat['HV'] UTS = converter.estimate_tensile_strength(HV) ax2.plot(HV, UTS, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5) plt.tight_layout() plt.show() # Output comparison table of hardness measurement methods print("\n" + "="*70) print("Comparison of Hardness Measurement Methods") print("="*70) print("\n【Vickers Hardness (HV)】") print("- 圧子: Diamond四角錐（対面角136°）") print("- Load: 1gf to 120kgf (notation varies with test force: HV0.1, HV10, etc.)") print("- feature: あ ゆるMaterial applicablepossible、measurementrange 広い") print("- Applications: Thin sheets, surface treatment layers, small parts") print("\n【Brinell Hardness (HB)】") print("- Indenter: Carbide ball (diameter 2.5mm, 5mm, 10mm)") print("- Load: 500 to 3000kgf") print("- Features: Large indentation, can measure coarse microstructures") print("- Applications: 鋳物、大型部品、粗大組織Material") print("\n【Rockwell Hardness (HRC)】") print("- 圧子: Diamond円錐（HRC）、鋼球（HRB）") print("- Load: 60kgf (HRA), 100kgf (HRB), 150kgf (HRC)") print("- Features: Measured by indentation depth, rapid measurement possible") print("- Applications: 鋼 焼入れ材（HRC）、Mild steel・非鉄金属（HRB）") print("\n" + "="*70) print("Relationship between hardness and tensile strength:") print("- Empirical rule: UTS (MPa) ≈ 3.3 × HV (for steel)") print("- Strength can be roughly estimated from hardness (non-destructive)") print("- 注意: Material種類bycoefficient 異become（3.0〜3.5程度）") 

**解説** : 硬度measurement法 複数あ 、measurement目的 and Material used depending on。Vickers hardness 最 汎用性 Highく、Brinell hardness 大型部品、Rockwell hardness is suitable for rapid measurement。It is also possible to estimate tensile strength from hardness。

* * *

## 4.2 Electrical Properties (Conductivity, Semiconductivity, Insulation)

### Electrical Conductivity and Resistivity

**Electrical Conductivity (σ)** represents how easily current flows:

$$\sigma = \frac{1}{\rho}$$

where,$\rho$ **resistivity（Resistivity）** （単位: Ω·m） す。

**Material Classification** ：

  * **Conductor** : $\rho < 10^{-5}$ Ω·m (metals)
  * **Semiconductor** : $10^{-5} < \rho < 10^{7}$ Ω·m (Si, Ge)
  * **Insulator（Insulator）** : $\rho > 10^{7}$ Ω·m（ガラス、ceramics、High分子）

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

**バンド理論** 、Materialrepresents the electronic states in energy bands：

  * **Valence Band** : Band filled with electrons
  * **Conduction Band** : Empty band (electrons can move freely)
  * **Band Gap (Eg)** : Energy difference between valence band and conduction band

**MaterialClassification and band gap** ：

  * **Metal (Conductor)** : Eg = 0 (valence band and conduction band overlap)
  * **Semiconductor（Semiconductor）** : 0 < Eg < 3 eV（room temperature 励起possible）
  * **Insulator（Insulator）** : Eg > 3 eV（励起困難）

SemiconductorMaterial | Band Gap Eg (eV, 300K) | Applications  
---|---|---  
**Si** | 1.12 | Integrated circuits, solar cells  
**Ge** | 0.66 | Infrared detectors  
**GaAs** | 1.42 | High速device、LED  
**GaN** | 3.44 | Blue LEDs, power devices  
**InP** | 1.35 | Optical communication devices  
**SiC** | 3.26 | High温・Highvoltage device  
  
### Code Example 4: Visualization of Relationship between Band Gap and Electrical Conductivity

Visualize the relationship between band gap and electrical conductivity.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Visualize the relationship between band gap and electrical c
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt # Physical constants k_B = 8.617e-5 # Boltzmann constant (eV/K) def intrinsic_carrier_concentration(Eg, T=300): """ Calculate intrinsic carrier concentration Parameters: Eg: Band gap (eV) T: Temperature (K) Returns: n_i: Intrinsic carrier concentration (cm⁻³) """ # Simplified approximation formula # n_i ∝ exp(-Eg / 2k_B T) # Actually n_i = sqrt(N_c * N_v) * exp(-Eg / 2k_B T) # Here we calculate relative values n_i = 1e19 * np.exp(-Eg / (2 * k_B * T)) return n_i def electrical_conductivity(n, mu=1000): """ Calculate electrical conductivity Parameters: n: Carrier concentration (cm⁻³) mu: Mobility (cm²/V·s) Returns: sigma: Electrical conductivity (S/cm) """ q = 1.602e-19 # Elementary charge (C) sigma = q * n * mu # S/cm return sigma # Band gap range (0 to 5 eV) Eg_range = np.linspace(0.1, 5, 100) # carrier concentration for each band gap andCalculate electrical conductivity n_i_range = np.array([intrinsic_carrier_concentration(Eg) for Eg in Eg_range]) sigma_range = electrical_conductivity(n_i_range) # Calculate resistivity rho_range = 1 / sigma_range # Ω·cm # Create plot fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Band gap vs carrier concentration ax1 = axes[0] ax1.semilogy(Eg_range, n_i_range, 'b-', linewidth=2.5) ax1.set_xlabel('Band Gap $E_g$ (eV)', fontsize=12, fontweight='bold') ax1.set_ylabel('Intrinsic Carrier Concentration $n_i$ (cm⁻³)', fontsize=12, fontweight='bold') ax1.set_title('Relationship between Band Gap and Carrier Concentration', fontsize=13, fontweight='bold') ax1.grid(alpha=0.3, which='both') # Material plot materials_bandgap = [('Ge', 0.66), ('Si', 1.12), ('GaAs', 1.42), ('InP', 1.35), ('SiC', 3.26), ('GaN', 3.44) ] for name, Eg in materials_bandgap: n_i = intrinsic_carrier_concentration(Eg) ax1.plot(Eg, n_i, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5) ax1.annotate(name, xy=(Eg, n_i), xytext=(Eg+0.1, n_i*2), fontsize=9, ha='left') # Band gap vs resistivity ax2 = axes[1] ax2.semilogy(Eg_range, rho_range, 'g-', linewidth=2.5) ax2.set_xlabel('Band Gap $E_g$ (eV)', fontsize=12, fontweight='bold') ax2.set_ylabel('Resistivity $\\rho$ (Ω·cm)', fontsize=12, fontweight='bold') ax2.set_title('Relationship between Band Gap and Resistivity', fontsize=13, fontweight='bold') ax2.grid(alpha=0.3, which='both') # MaterialClassification region Color分け ax2.axhspan(1e-8, 1e-5, alpha=0.2, color='blue', label='Conductorregion') ax2.axhspan(1e-5, 1e7, alpha=0.2, color='yellow', label='Semiconductorregion') ax2.axhspan(1e7, 1e20, alpha=0.2, color='red', label='Insulatorregion') ax2.legend(fontsize=10, loc='upper left') # Material plot for name, Eg in materials_bandgap: n_i = intrinsic_carrier_concentration(Eg) sigma = electrical_conductivity(n_i) rho = 1 / sigma ax2.plot(Eg, rho, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5) plt.tight_layout() plt.show() # Material特性 display print("="*70) print("SemiconductorMaterialband gap and electrical properties of") print("="*70) print(f"\n{'Material':<10} {'Eg(eV)':>10} {'n_i(cm⁻³)':>15} {'ρ(Ω·cm)':>15} {'Classification':<10}") print("-" * 70) for name, Eg in materials_bandgap: n_i = intrinsic_carrier_concentration(Eg) sigma = electrical_conductivity(n_i) rho = 1 / sigma if rho < 1e-5: classification = "Conductor" elif rho < 1e7: classification = "Semiconductor" else: classification = "Insulator" print(f"{name:<10} {Eg:>10.2f} {n_i:>15.2e} {rho:>15.2e} {classification:<10}") print("\n" + "="*70) print("Importance of band gap:") print("- Eg small → キャリア濃度 Highい → 電気伝導度 Highい") print("- Eg large → キャリア濃度 Lowい → 絶縁性 Highい") print("- Si（Eg=1.12eV）: 最 importantなSemiconductorMaterial（room temperature moderate conductivity）") print("- GaN（Eg=3.44eV）: wide band gap semiConductor（High温・High電圧動作）") 

**解説** : the smaller the band gap、room temperature キャリア濃度 Highくな 、electrical conductivity increases（resistivity 小さくな ）。Semiconductor band gap Material Applicationsis an important parameter that determines。

### codeexample5: temperature dependence plot of resistivity（金属 vs Semiconductor）

金属 and Semiconductor 、visualize that the temperature dependence of resistivity is opposite。
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 金属 and Semiconductor 、visualize that the temperature depende
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def resistivity_metal(T, rho_0=1.68e-8, alpha=0.0039): """ Resistivity of metals (temperature dependence) Parameters: T: Temperature (K) rho_0: Resistivity at reference temperature (273K) (Ω·m) alpha: Temperature coefficient (1/K) Returns: rho: Resistivity (Ω·m) """ T_0 = 273 # 基準Temperature (K) rho = rho_0 * (1 + alpha * (T - T_0)) return rho def resistivity_semiconductor(T, Eg=1.12, rho_room=640): """ Semiconductor resistivity（temperature依存性） Parameters: T: Temperature (K) Eg: Band gap (eV) rho_room: Room temperature (300K) Resistivity (Ω·m) Returns: rho: Resistivity (Ω·m) """ k_B = 8.617e-5 # Boltzmann constant (eV/K) T_room = 300 # room temperature (K) # 真性Semiconductor resistivity exp(Eg / 2k_B T) 比example rho = rho_room * np.exp(Eg / (2 * k_B) * (1/T - 1/T_room)) return rho # temperature range（200K 〜 500K） T_range = np.linspace(200, 500, 100) # Metal (Copper) resistivity rho_metal = resistivity_metal(T_range, rho_0=1.68e-8, alpha=0.0039) # Semiconductor (Silicon) resistivity rho_si = resistivity_semiconductor(T_range, Eg=1.12, rho_room=640) # Create plot fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Normal scale ax1 = axes[0] ax1.plot(T_range - 273, rho_metal * 1e8, 'b-', linewidth=2.5, label='Metal (Cu)') ax1_twin = ax1.twinx() ax1_twin.plot(T_range - 273, rho_si, 'r--', linewidth=2.5, label='Semiconductor (Si)') ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold') ax1.set_ylabel('Resistivity (Metal, 10⁻⁸ Ω·m)', fontsize=11, fontweight='bold', color='b') ax1_twin.set_ylabel('Resistivity (Semiconductor, Ω·m)', fontsize=11, fontweight='bold', color='r') ax1.set_title('Temperature Dependence of Resistivity (Normal Scale)', fontsize=13, fontweight='bold') ax1.tick_params(axis='y', labelcolor='b') ax1_twin.tick_params(axis='y', labelcolor='r') ax1.grid(alpha=0.3) # Legend lines1, labels1 = ax1.get_legend_handles_labels() lines2, labels2 = ax1_twin.get_legend_handles_labels() ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left') # Logarithmic scale ax2 = axes[1] ax2.semilogy(T_range - 273, rho_metal, 'b-', linewidth=2.5, label='Metal (Cu)') ax2.semilogy(T_range - 273, rho_si, 'r--', linewidth=2.5, label='Semiconductor (Si)') ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold') ax2.set_ylabel('Resistivity (Ω·m)', fontsize=12, fontweight='bold') ax2.set_title('Temperature Dependence of Resistivity (Logarithmic Scale)', fontsize=13, fontweight='bold') ax2.legend(fontsize=11) ax2.grid(alpha=0.3, which='both') plt.tight_layout() plt.show() # calculation of temperature coefficient and display print("="*70) print("resistivity temperature依存性") print("="*70) # specific temperature 値 calculation temps_celsius = [0, 25, 100, 200] temps_kelvin = [t + 273 for t in temps_celsius] print("\n[Resistivity of Metal (Copper)]") print(f"{'Temperature (°C)':>10} {'Resistivity (10⁻⁸ Ω·m)':>25} {'Change Rate (%)':>15}") print("-" * 70) rho_ref = resistivity_metal(273) for T_c, T_k in zip(temps_celsius, temps_kelvin): rho = resistivity_metal(T_k) change = ((rho - rho_ref) / rho_ref) * 100 print(f"{T_c:>10} {rho*1e8:>25.4f} {change:>15.2f}") print("\n[Resistivity of Semiconductor (Silicon)]") print(f"{'Temperature (°C)':>10} {'Resistivity (Ω·m)':>25} {'Change Rate (%)':>15}") print("-" * 70) rho_ref = resistivity_semiconductor(300) for T_c, T_k in zip(temps_celsius, temps_kelvin): rho = resistivity_semiconductor(T_k) change = ((rho - rho_ref) / rho_ref) * 100 print(f"{T_c:>10} {rho:>25.2e} {change:>15.2f}") print("\n" + "="*70) print("Difference in temperature dependence:") print("\n[Metals]") print("- temperature increase → resistivity increase（positive temperature coefficient）") print("- 理由: 格子振動 増大し、電子散乱 増える") print("- temperaturecoefficient α ≈ +0.4% / K（Copper case）") print("- application: 測温抵抗体（platinum resistance thermometeretc）") print("\n[Semiconductors]") print("- temperature increase → resistivity decrease（negative temperature coefficient）") print("- 理由: carrier concentration increases due to thermal excitation") print("- temperaturecoefficient 負 large（-数%/K）") print("- application: サーミスタ（temperatureセンサ）") 

**解説** : metals increase resistivity with temperature increase、Semiconductorresistivity decreases with temperature increase。This is due to the difference in the mechanism of electrical conduction。in metals, scattering by lattice vibration is dominant、Semiconductorthe increase in carrier concentration due to thermal excitation is dominant。

* * *

## 4.3 Thermal Properties (Thermal Conduction, Thermal Expansion)

### Thermal Conductivity

**Thermal Conductivity（Thermal Conductivity, κ）** 、represents how easily heat is transferred：

$$q = -\kappa \nabla T$$

where,$q$ 熱流束（W/m²）、$\nabla T$ temperature勾配（K/m） す。

**MaterialClassification** ：

  * **金属** : κ = 50-400 W/(m·K)（HighいThermal Conductivity）
  * **ceramics** : κ = 1-50 W/(m·K)
  * **High分子** : κ = 0.1-0.5 W/(m·K)（LowいThermal Conductivity）

### 線膨張coefficient

**線膨張coefficient（Coefficient of Thermal Expansion, CTE, α）** 、is the rate of change in length with respect to temperature change：

$$\alpha = \frac{1}{L} \frac{dL}{dT}$$

単位 1/K also ppm/K（10⁻⁶/K） す。

### Specific Heat

**Specific Heat（Specific Heat Capacity, c）** 、単位質量 物質 1Kis the amount of heat required to raise the temperature：

$$Q = mc\Delta T$$

単位 J/(kg·K) す。

### codeexample6: 熱的property comparison（Thermal Conductivity、線膨張coefficient、Specific Heat）

代Table 的なMaterial 熱的property comparisondo。
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 代Table 的なMaterial 熱的property comparisondo。
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt # Material 熱的propertydatabase materials_thermal = { # 金属 'Copper (Cu)': { 'thermal_conductivity': 401, # W/(m·K) 'thermal_expansion': 16.5, # ppm/K (10⁻⁶/K) 'specific_heat': 385, # J/(kg·K) 'density': 8960, # kg/m³ 'category': '金属' }, 'Aluminum (Al)': { 'thermal_conductivity': 237, 'thermal_expansion': 23.1, 'specific_heat': 897, 'density': 2700, 'category': '金属' }, '鉄（Fe）': { 'thermal_conductivity': 80, 'thermal_expansion': 11.8, 'specific_heat': 449, 'density': 7874, 'category': '金属' }, 'Stainless steel（SUS304）': { 'thermal_conductivity': 16, 'thermal_expansion': 17.3, 'specific_heat': 500, 'density': 8000, 'category': '金属' }, # ceramics 'アルミナ（Al₂O₃）': { 'thermal_conductivity': 30, 'thermal_expansion': 8.1, 'specific_heat': 775, 'density': 3950, 'category': 'ceramics' }, '窒化ケイ素（Si₃N₄）': { 'thermal_conductivity': 28, 'thermal_expansion': 3.2, 'specific_heat': 680, 'density': 3200, 'category': 'ceramics' }, 'Glass (SiO₂)': { 'thermal_conductivity': 1.4, 'thermal_expansion': 0.55, 'specific_heat': 750, 'density': 2200, 'category': 'ceramics' }, # High分子 'Polyethylene（PE）': { 'thermal_conductivity': 0.42, 'thermal_expansion': 100, 'specific_heat': 2300, 'density': 950, 'category': 'High分子' }, 'polystyrene（PS）': { 'thermal_conductivity': 0.13, 'thermal_expansion': 70, 'specific_heat': 1300, 'density': 1050, 'category': 'High分子' } } # data 整理 materials = list(materials_thermal.keys()) thermal_cond = [materials_thermal[m]['thermal_conductivity'] for m in materials] thermal_exp = [materials_thermal[m]['thermal_expansion'] for m in materials] specific_heat = [materials_thermal[m]['specific_heat'] for m in materials] categories = [materials_thermal[m]['category'] for m in materials] # カテゴリご and Color分け color_map = {'金属': '#1f77b4', 'ceramics': '#ff7f0e', 'High分子': '#2ca02c'} colors = [color_map[cat] for cat in categories] # Create plot fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Thermal Conductivity comparison ax1 = axes[0, 0] y_pos = np.arange(len(materials)) bars1 = ax1.barh(y_pos, thermal_cond, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7) ax1.set_yticks(y_pos) ax1.set_yticklabels(materials, fontsize=9) ax1.set_xlabel('Thermal Conductivity κ (W/(m·K))', fontsize=11, fontweight='bold') ax1.set_title('Thermal Conductivity comparison', fontsize=12, fontweight='bold') ax1.grid(axis='x', alpha=0.3) ax1.set_xscale('log') # 値 バー 端 display for i, (bar, val) in enumerate(zip(bars1, thermal_cond)): ax1.text(val * 1.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8) # Comparison of linear expansion coefficient ax2 = axes[0, 1] bars2 = ax2.barh(y_pos, thermal_exp, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7) ax2.set_yticks(y_pos) ax2.set_yticklabels(materials, fontsize=9) ax2.set_xlabel('線膨張coefficient α (ppm/K = 10⁻⁶/K)', fontsize=11, fontweight='bold') ax2.set_title('線膨張coefficient comparison', fontsize=12, fontweight='bold') ax2.grid(axis='x', alpha=0.3) ax2.set_xscale('log') for i, (bar, val) in enumerate(zip(bars2, thermal_exp)): ax2.text(val * 1.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8) # Specific Heat comparison ax3 = axes[1, 0] bars3 = ax3.barh(y_pos, specific_heat, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7) ax3.set_yticks(y_pos) ax3.set_yticklabels(materials, fontsize=9) ax3.set_xlabel('Specific Heat c (J/(kg·K))', fontsize=11, fontweight='bold') ax3.set_title('Specific Heat comparison', fontsize=12, fontweight='bold') ax3.grid(axis='x', alpha=0.3) for i, (bar, val) in enumerate(zip(bars3, specific_heat)): ax3.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=8) # Thermal Conductivity vs 線膨張coefficient 散布Figure ax4 = axes[1, 1] for cat in ['金属', 'ceramics', 'High分子']: indices = [i for i, c in enumerate(categories) if c == cat] tc = [thermal_cond[i] for i in indices] te = [thermal_exp[i] for i in indices] ax4.scatter(tc, te, s=150, c=color_map[cat], label=cat, edgecolors='black', linewidth=1.5, alpha=0.7) ax4.set_xlabel('Thermal Conductivity κ (W/(m·K))', fontsize=11, fontweight='bold') ax4.set_ylabel('線膨張coefficient α (ppm/K)', fontsize=11, fontweight='bold') ax4.set_title('Thermal Conductivity vs 線膨張coefficient', fontsize=12, fontweight='bold') ax4.set_xscale('log') ax4.set_yscale('log') ax4.legend(fontsize=10) ax4.grid(alpha=0.3, which='both') # Material名 plot for i, mat in enumerate(materials): ax4.annotate(mat.split('（')[0], xy=(thermal_cond[i], thermal_exp[i]), xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.8) plt.tight_layout() plt.show() # 熱拡散率 calculation print("="*70) print("Material 熱的propertycomparison") print("="*70) print(f"\n{'Material':<20} {'κ(W/m·K)':>12} {'α(ppm/K)':>12} {'c(J/kg·K)':>12} {'a(mm²/s)':>12}") print("-" * 70) for mat in materials: props = materials_thermal[mat] kappa = props['thermal_conductivity'] alpha_exp = props['thermal_expansion'] c = props['specific_heat'] rho = props['density'] # 熱拡散率 a = κ / (ρ * c) calculation thermal_diffusivity = kappa / (rho * c) * 1e6 # mm²/s print(f"{mat:<20} {kappa:>12.2f} {alpha_exp:>12.2f} {c:>12.0f} {thermal_diffusivity:>12.3f}") print("\n" + "="*70) print("Meaning and applications of thermal properties:") print("="*70) print("\n【Thermal Conductivity κ】") print("- Highい → 熱 quickly伝える → ヒートシンク、放熱Material") print("- Lowい → 断熱性 Highい → 断熱材、保温材") print("- 金属 > ceramics > High分子 順") print("- application: Copper（放熱）、ステンレス（断熱）") print("\n【線膨張coefficient α】") print("- Highい → temperaturechange 大きく伸縮 → 熱応力 発生し and すい") print("- Lowい → 寸法安定性 Highい → 精密機器 適do") print("- High分子 > 金属 > ceramics 順") print("- application: ガラス（LowCTE）、thermal stress management during dissimilar material joining") print("\n【Specific Heat c】") print("- Highい → temperaturechangeし くい → 蓄熱Material") print("- Lowい → quicklytemperature 変わる → 熱応答 fast") print("- High分子 > ceramics > 金属 順（質量あた ）") print("- application: 水（HighSpecific Heat、冷却材）、金属（LowSpecific Heat、調理器具）") print("\n【熱拡散率 a = κ/(ρc)】") print("- 熱 MaterialMedium 拡散do速さ") print("- Highい → the whole becomes uniformly heated quickly") print("- 金属 最 Highい（Copper、Aluminum）") 

**解説** : Material 熱的property 、熱管理design important す。金属 Thermal Conductivity Highく放熱 適し、High分子 Thermal Conductivity Lowく断熱 適do。Linear expansion coefficient is important when considering thermal stress during dissimilar material bonding。

* * *

## 4.4 Optical Properties (Transparency, Color)

### Transparency and 不Transparency

**透明（Transparent）** : visible light is transmitted with almost no absorption

  * example: ガラス、透明High分子（PMMA, PC）
  * 条件: band gap > 可視光 energy（約1.8〜3.1 eV）

**半透明（Translucent）** : 光 散乱されな transmission

  * example: す ガラス、薄い紙

**不透明（Opaque）** : light is absorbed or reflected

  * example: 金属、黒ColorMaterial
  * 金属: 自由電子 るreflection

### Color and absorptionスペクトル

Material **Color** 、absorbs specific wavelengths of visible light、残 reflection・transmissiondothing 生じ。

**可視光 波長range** :

  * 紫: 380-450 nm
  * 青: 450-495 nm
  * 緑: 495-570 nm
  * 黄: 570-590 nm
  * 橙: 590-620 nm
  * 赤: 620-750 nm

**補Color relationship** : existColor absorptiondo and 、that補Color 見える

  * 青 absorption → 橙Color 見える
  * 赤 absorption → 青緑Color 見える

### Refractive Index

**Refractive Index（Refractive Index, n）** 、光 物質Medium 進む速さ 比 す：

$$n = \frac{c}{v}$$

where,$c$ 真空Medium 光速、$v$ 物質Medium 光速 す。

Material | Refractive Index（589nm, D線） | Transparency  
---|---|---  
**真空** | 1.0000 | -  
**Air** | 1.0003 | -  
**水** | 1.333 | 透明  
**石英Glass (SiO₂)** | 1.458 | 透明  
**soda-lime glass** | 1.52 | 透明  
**PMMA（アクリル）** | 1.49 | 透明  
**polycarbonate（PC）** | 1.586 | 透明  
**Diamond** | 2.417 | 透明  
  
**Applications of Optical Properties** :

  * **レンズ** : HighRefractive IndexMaterial（光学ガラス、High分子）
  * **光ファイバ** : Low損失透明Material（石英ガラス）
  * **reflectionpreventive coating** : 薄膜干渉 利用
  * **着ColorMaterial** : 顔料・control of absorption spectrum by dyes
  * **Solar cells** : 可視光absorptionMaterial（Si, GaAs etc）

> **ま and め** : Material property（機械的・電気的・熱的・光学的） 、deeply related to atomic structure and crystal structure。Applications 応じて適切なMaterial selectiondo 、understand these properties quantitatively、comparisondothing important す。

* * *

## 4.5 本Chapter Summary

### 学んだthing

  1. **Mechanical Properties**
     * Stress-Strain Curve: ヤング率、yield strength、tensile strength、elongation at fracture
     * DuctilityMaterial vs BrittleMaterial difference
     * 硬度measurement法（Vickers, Brinell, Rockwell） and 換算
  2. **Electrical Properties**
     * Conductor・Semiconductor・Insulator Classification（resistivity る）
     * Relationship between band gap and electrical conductivity
     * 金属 and Semiconductor temperature依存性 difference（正 vs negative temperature coefficient）
  3. **Thermal Properties**
     * Thermal Conductivity: 金属 > ceramics > High分子
     * 線膨張coefficient: High分子 > 金属 > ceramics
     * Specific Heat and 熱拡散率 意味
  4. **Optical Properties**
     * Transparency 条件（band gap > visible light energy）
     * Color and absorptionスペクトル relationship
     * Refractive Index and 光学application

### importantなpoint

  * Material property **structure（原子配列・結晶structure・化学結合）** 起因do
  * 機械的property Materialselection most basic indicator
  * Electrical properties are explained by band structure
  * Thermal properties are essential for thermal management design
  * Python Material特性 定量的 calculation・comparisoncan

### Next Chapter

Chapter 5 、**Python 学ぶ結晶structurevisualization** 学び：

  * pymatgen入門（結晶structurelibrary）
  * CIFfile loading and structure analysis
  * Materials Projectdatabase 活用
  * 代Table 的Material（Si, Fe, Al₂O₃） structure解析
  * comprehensive workflow（structure→解析→visualization→特性予測）
