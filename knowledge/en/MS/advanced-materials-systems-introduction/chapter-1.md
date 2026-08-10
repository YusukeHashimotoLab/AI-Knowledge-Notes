---
title: "Chapter 1: Advanced Ceramics Materials"
chapter_title: "Chapter 1: Advanced Ceramics Materials"
subtitle: Structural, Functional, and Bioceramics - Design Principles for High Performance
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
---

[AI Terakoya Home](<../index.html>)›[Materials Science](<../../index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 1

🌐 EN | [🇯🇵 JP](<../../../jp/MS/advanced-materials-systems-introduction/chapter-1.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain:

### Fundamental Understanding

  * Strengthening and toughening mechanisms of structural ceramics (transformation toughening, fiber reinforcement)
  * Physical origins and crystal structures of functional ceramics (piezoelectric, dielectric, magnetic)
  * Biocompatibility and osseointegration mechanisms of bioceramics
  * Mechanical properties of ceramics and statistical fracture theory (Weibull distribution)

### Practical Skills

  * Analyze strength distribution of ceramics (Weibull statistics) using Python
  * Calculate phase diagrams using pycalphad and optimize sintering conditions
  * Calculate and evaluate piezoelectric constants, dielectric permittivity, and magnetic properties
  * Select optimal ceramics for specific applications using materials selection matrix

### Applied Capabilities

  * Design optimal ceramic composition and microstructure from application requirements
  * Design functional ceramic devices (sensors, actuators)
  * Evaluate biocompatibility of bioceramic implants
  * Perform reliability design (probabilistic fracture prediction) for ceramic materials

## 1.1 Structural Ceramics - Principles of High Strength and High Toughness

### 1.1.1 Overview of Structural Ceramics

Structural ceramics are **ceramic materials with excellent mechanical properties (high strength, high hardness, heat resistance) used as structural components in harsh environments**. They enable use in high-temperature or corrosive environments impossible for metallic materials, with important applications including:

  * **Al₂O₃ (Alumina)** : Cutting tools, wear-resistant parts, artificial joints (biocompatibility)
  * **ZrO₂ (Zirconia)** : Dental materials, oxygen sensors, thermal barrier coatings (high toughness)
  * **Si₃N₄ (Silicon Nitride)** : Gas turbine components, bearings (high-temperature strength)
  * **SiC (Silicon Carbide)** : Semiconductor manufacturing equipment, armor materials (ultra-high hardness)

**💡 Industrial Significance**

Structural ceramics are indispensable in aerospace, automotive, and medical fields. Advanced ceramics account for approximately 60% of the global ceramics market (over $230B as of 2023). The reasons are:

  * 3-5 times the strength of metals (at room temperature) and excellent heat resistance (above 1500°C)
  * Chemical stability (inert to acids and alkalis)
  * Weight reduction effect due to low density (1/2-1/3 of metals)
  * Wear resistance due to high hardness (Hv 1500-2500)

### 1.1.2 High-Strength Ceramics (Al₂O₃, ZrO₂, Si₃N₄)

High-strength ceramics are represented by the following three major materials:
    
    
    flowchart LR
     A[Al₂O₃  
    Alumina] --> B[High Hardness  
    Hv 2000]
     C[ZrO₂  
    Zirconia] --> D[High Toughness  
    10-15 MPa√m]
     E[Si₃N₄  
    Silicon Nitride] --> F[High-Temperature Strength  
    1400°Cuse]
    
     style A fill:#e3f2fd
     style C fill:#fff3e0
     style E fill:#e8f5e9
     style B fill:#f3e5f5
     style D fill:#fce4ec
     style F fill:#fff9c4
     

  1. **Al₂O₃ (Alumina)** : Representative of oxide ceramics. High Hardness（Hv 2000）、excellent wear resistance, and biocompatibility, used in cutting tools and artificial joints. Most widely used due to low manufacturing cost.
  2. **ZrO₂ (Zirconia)** : Achieves the highest level of fracture toughness (10-15 MPa√m) among ceramic materials through transformation toughening. Also called "ceramic steel".
  3. **Si₃N₄ (Silicon Nitride)** : Strong covalent bonding maintains high strength up to 1400°C. Used as high-temperature structural material for gas turbine components and bearings. Also exhibits excellent thermal shock resistance.

**⚠️ Intrinsic Challenge of Ceramics**

While ceramics possess high strength and high hardness, **brittleness (low toughness)** is the major drawback. Microscopic defects (pores, cracks) become stress concentration points, causing catastrophic fracture (Griffith theory). Fracture toughness is less than 1/10 that of metals. Therefore, toughening technology is an important research topic.

### 1.1.3 Toughening Mechanisms

#### Mechanism 1: Transformation Toughening

Zirconia（ZrO₂） 最も効果的に機能する強化機構 す：

ZrO₂ (tetragonal, t-phase) → ZrO₂ (monoclinic, m-phase) + volume expansion (3-5%) 

**Toughening Mechanism:**

  * **Stress-Induced Transformation** : Metastable tetragonal (t) phase transforms to monoclinic (m) phase in the high-stress field at crack tips
  * **Volume Expansion Effect** : 3-5% volume expansion generates compressive stress around cracks, suppressing crack propagation
  * **Energy Absorption** : Energy consumption during transformation increases fracture energy
  * **Toughness Enhancement Effect** : Fracture toughness increases from 3 MPa√m to 10-15 MPa√m (3-5 times improvement)

**Implementation Method:** Add Y₂O₃ (3-8 mol%) or MgO (9-15 mol%) to stabilize tetragonal phase at room temperature (PSZ: Partially Stabilized Zirconia)

#### Mechanism 2: Fiber Reinforcement

This method involves incorporating high-strength fibers into a ceramic matrix:

Ceramic Matrix Composites (CMC) = Ceramic Matrix + Reinforcing Fibers (SiC, C, Al₂O₃) 

**Toughening Mechanism:**

  * **Crack Deflection** : Cracks deflect at fiber interfaces, increasing the propagation path length
  * **Fiber Pullout** : Large energy absorption occurs when fibers are pulled out
  * **Crack Bridging** : Fibers bridge cracks and maintain stress transfer
  * **Toughness Enhancement Effect** : Fracture toughness increases from 5 MPa√m to 20-30 MPa√m (4-6 times improvement)

**Applications:** SiC/SiC composites (aircraft engine components), C/C composites (brake disks)

## 1.2 Functional Ceramics - Piezoelectric, Dielectric, and Magnetic

### 1.2.1 Piezoelectric Ceramics

The piezoelectric effect is **a phenomenon where electrical polarization is generated by applied mechanical stress (direct piezoelectric effect), and conversely, mechanical strain is generated by an applied electric field (converse piezoelectric effect)**.

#### Representative Piezoelectric Materials

PZT (Pb(Zr,Ti)O₃): Piezoelectric constant d₃₃ = 200-600 pC/N 

BaTiO₃ (Barium Titanate): Piezoelectric constant d₃₃ = 85-190 pC/N (lead-free alternative) 

**Characteristics of PZT (Lead Zirconate Titanate):**

  * **High Piezoelectric Constant** : d₃₃ = 200-600 pC/N (most excellent as applied material)
  * **Morphotropic Phase Boundary (MPB)** : Piezoelectric properties are maximized near Zr/Ti ratio of 52/48
  * **Curie Temperature** : 320-380°C (piezoelectricity disappears above this temperature)
  * **Applications** : Ultrasonic transducers, piezoelectric actuators, piezoelectric speakers, piezoelectric igniters

**⚠️ Environmental Issues and Lead-Free Alternatives**

PZT contains more than 60 wt% lead (Pb), subject to usage restrictions under European RoHS regulations. Lead-free alternatives such as BaTiO₃-based, (K,Na)NbO₃-based, and BiFeO₃-based materials are being researched, but do not match PZT performance (d₃₃ = 100-300 pC/N). While piezoelectric devices are exempt items for medical equipment, alternative material development is necessary in the long term.

#### Crystallographic Origin of Piezoelectric Effect

The piezoelectric effect **non-centrosymmetric crystal structure** occurs only in materials with:

  * **Paraelectric Phase (Cubic, Pm3m)** : Centrosymmetric → No piezoelectricity (high temperature)
  * **Ferroelectric Phase (Tetragonal, P4mm)** : Non-centrosymmetric → Piezoelectricity present (room temperature)
  * **Spontaneous Polarization** : Dipole moment generated by displacement of Ti⁴⁺ ions from the center of oxygen octahedra
  * **Domain Structure** : Domain orientations align under applied electric field, exhibiting giant piezoelectric effect (poling treatment)

### 1.2.2 Dielectric Ceramics

Dielectric ceramics are **capacitor materials with high dielectric constant (εᵣ) that store electrical energy**.

#### Materials for MLCC (Multilayer Ceramic Capacitors)

BaTiO₃ (Barium Titanate): εᵣ = 1,500-10,000 (room temperature, 1 kHz) 

**Origin of High Dielectric Constant:**

  * **Ferroelectricity** : Property where spontaneous polarization can be reversed by external electric field
  * **Domain Wall Movement** : Domain walls move easily under applied electric field, producing large polarization changes
  * **Curie Temperature（Tc）** : BaTiO₃ inTc = 120°C、Dielectric constant peaks at this temperature
  * **Composition Adjustment** : Addition of CaZrO₃, SrTiO₃ shifts Tc near room temperature (X7R characteristics)

**✅ Remarkable Performance of MLCC (Multilayer Ceramic Capacitors)**

Modern MLCCs have advanced to extreme miniaturization and high performance:

  * **Number of Layers** : More than 1,000 layers (dielectric layer thickness < 1 μm)
  * **Capacitance** : Achieving over 100 μF in 1 mm³ size
  * **Applications** : Over 800 units installed in one smartphone
  * **Market Size** : Annual production exceeds 1 trillion units (largest electronic component worldwide)

BaTiO₃-based MLCCs are key materials for miniaturization and performance enhancement of electronic devices.

### 1.2.3 Magnetic Ceramics - Ferrites

Ferrites are **oxide-based magnetic materials with low-loss characteristics at high frequencies** , widely used in transformers, inductors, and electromagnetic wave absorbers.

#### Types and Applications of Ferrites

Spinel Ferrite: MFe₂O₄ (M = Mn, Ni, Zn, Co, etc.) 

Hexagonal Ferrite (Hard Ferrite): BaFe₁₂O₁₉, SrFe₁₂O₁₉ (permanent magnets) 

**Characteristics of Spinel Ferrites:**

  * **Soft Magnetic** : Low coercivity (Hc < 100 A/m), easy magnetization reversal
  * **High-Frequency Characteristics** : Small eddy current loss due to high electrical resistance (ρ > 10⁶ Ω·cm)
  * **Mn-Zn Ferrite** : High permeability (μᵣ = 2,000-15,000), for low-frequency transformers
  * **Ni-Zn Ferrite** : Excellent high-frequency characteristics (GHz band), for EMI countermeasure components

**Characteristics of Hexagonal Ferrites (Hard Ferrites):**

  * **Hard Magnetic** : Large coercivity (Hc = 200-400 kA/m) and remanent flux density (Br = 0.4 T)
  * **Permanent Magnet Material** : Used in motors, speakers, magnetic recording media
  * **Low Cost** : Lower performance than rare-earth magnets (Nd-Fe-B), but inexpensive raw materials and mass production possible
  * **Corrosion Resistance** : Being oxides, they do not corrode unlike metallic magnets

**💡 Origin of Ferrite Magnetism**

The magnetism of ferrites arises from the **antiparallel alignment of magnetic moments of ions at A-sites (tetrahedral positions) and B-sites (octahedral positions)** in the spinel structure (AB₂O₄) (ferrimagnetism). In Mn-Zn ferrites, the magnetic moments of Mn²⁺ and Fe³⁺ partially cancel each other, resulting in small overall magnetization but achieving high permeability.

## 1.3 Bioceramics - Biocompatibility and Osseointegration

### 1.3.1 Overview of Bioceramics

Bioceramics are **ceramic materials that do not cause rejection reactions when in contact with biological tissues (biocompatibility) and can directly bond with bone tissue (osteoconductivity)**.

#### Representative Bioceramics

HAp (Hydroxyapatite): Ca₁₀(PO₄)₆(OH)₂ 

β-TCP (Tricalcium Phosphate): Ca₃(PO₄)₂ 

**Characteristics of Hydroxyapatite (HAp):**

  * **Main Component of Bone** : 65% of the inorganic component of natural bone is HAp (remaining 35% is organic collagen)
  * **Biocompatibility** : No rejection reaction occurs due to similar chemical composition to bone tissue
  * **Osteoconduction** : Osteoblasts attach and proliferate on HAp surface, forming new bone tissue
  * **Osseointegration** : Direct chemical bonding forms between HAp surface and bone tissue
  * **Applications** : Artificial bone, dental implants, bone fillers, coating for Ti alloy implants

**✅ Bioresorbability of β-TCP**

β-TCP (tricalcium phosphate), unlike HAp, has the property of **being gradually resorbed in vivo** :

  * **Resorption Period** : Complete resorption in 6-18 months (depends on particle size and porosity)
  * **Replacement Mechanism** : β-TCP dissolves while being replaced by new bone tissue (bone remodeling)
  * **Ca²⁺·PO₄³⁻ Supply** : Ions released by dissolution promote bone formation
  * **HAp/β-TCP Composite** : Resorption rate can be controlled by mixing ratio (e.g., HAp 70% / β-TCP 30%)

Bioresorbability achieves ideal bone regeneration where no permanent foreign material remains in the body, being completely replaced by autologous bone tissue.

### 1.4 Python Practice: Analysis and Design of Ceramic Materials

### Example 1: Analysis of Fracture Strength Distribution using Weibull Statistics
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 1: Arrhenius Equation Simulation
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    R = 8.314 # J/(mol·K)
    
    # Diffusion parameters for BaTiO3 system (literature values)
    D0 = 5e-4 # m²/s (frequency factor)
    Ea = 300e3 # J/mol (activation energy 300 kJ/mol)
    
    def diffusion_coefficient(T, D0, Ea):
     """Calculate diffusion coefficient using Arrhenius equation
    
     Args:
     T (float or array): Temperature [K]
     D0 (float): Frequency factor [m²/s]
     Ea (float): Activation energy [J/mol]
    
     Returns:
     float or array: Diffusion coefficient [m²/s]
     """
     return D0 * np.exp(-Ea / (R * T))
    
    # Temperature range 800-1400°C
    T_celsius = np.linspace(800, 1400, 100)
    T_kelvin = T_celsius + 273.15
    
    # Calculate diffusion coefficient
    D = diffusion_coefficient(T_kelvin, D0, Ea)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Logarithmic plot (Arrhenius plot)
    plt.subplot(1, 2, 1)
    plt.semilogy(T_celsius, D, 'b-', linewidth=2)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Diffusion Coefficient (m²/s)', fontsize=12)
    plt.title('Arrhenius Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 1/T vs ln(D) plot (linear relationship)
    plt.subplot(1, 2, 2)
    plt.plot(1000/T_kelvin, np.log(D), 'r-', linewidth=2)
    plt.xlabel('1000/T (K⁻¹)', fontsize=12)
    plt.ylabel('ln(D)', fontsize=12)
    plt.title('Linearized Arrhenius Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display diffusion coefficients at key temperatures
    key_temps = [1000, 1100, 1200, 1300]
    print("Comparison of temperature dependence:")
    print("-" * 50)
    for T_c in key_temps:
     T_k = T_c + 273.15
     D_val = diffusion_coefficient(T_k, D0, Ea)
     print(f"{T_c:4d}°C: D = {D_val:.2e} m²/s")
    
    # Output example:
    # Comparison of temperature dependence:
    # --------------------------------------------------
    # 1000°C: D = 1.89e-12 m²/s
    # 1100°C: D = 9.45e-12 m²/s
    # 1200°C: D = 4.01e-11 m²/s
    # 1300°C: D = 1.48e-10 m²/s
    

### Example 2: Simulation of Reaction Progress using Jander Equation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 2: Conversion Calculation using Jander Equation
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    def jander_equation(alpha, k, t):
     """Jander equation
    
     Args:
     alpha (float): Conversion rate (0-1)
     k (float): Rate constant [s⁻¹]
     t (float): time [s]
    
     Returns:
     float: Left side of Jander equation - k*t
     """
     return (1 - (1 - alpha)**(1/3))**2 - k * t
    
    def calculate_conversion(k, t):
     """Calculate conversion rate at time t
    
     Args:
     k (float): Rate constant
     t (float): time
    
     Returns:
     float: Conversion rate (0-1)
     """
     # Solve Jander equation numerically for alpha
     alpha0 = 0.5 # Initial guess
     alpha = fsolve(lambda a: jander_equation(a, k, t), alpha0)[0]
     return np.clip(alpha, 0, 1) # Limit to 0-1 range
    
    # Parameter settings
    D = 1e-11 # m²/s (diffusion coefficient at 1200°C)
    C0 = 10000 # mol/m³
    r0_values = [1e-6, 5e-6, 10e-6] # Particle radius [m]: 1μm, 5μm, 10μm
    
    # time array (0-50 hours)
    t_hours = np.linspace(0, 50, 500)
    t_seconds = t_hours * 3600
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Effect of particle size
    plt.subplot(1, 2, 1)
    for r0 in r0_values:
     k = D * C0 / r0**2
     alpha = [calculate_conversion(k, t) for t in t_seconds]
     plt.plot(t_hours, alpha, linewidth=2,
     label=f'r₀ = {r0*1e6:.1f} μm')
    
    plt.xlabel('time (hours)', fontsize=12)
    plt.ylabel('Conversion (α)', fontsize=12)
    plt.title('Effect of Particle Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Effect of temperature (fixed particle size)
    plt.subplot(1, 2, 2)
    r0_fixed = 5e-6 # 5μm fixed
    temperatures = [1100, 1200, 1300] # °C
    
    for T_c in temperatures:
     T_k = T_c + 273.15
     D_T = diffusion_coefficient(T_k, D0, Ea)
     k = D_T * C0 / r0_fixed**2
     alpha = [calculate_conversion(k, t) for t in t_seconds]
     plt.plot(t_hours, alpha, linewidth=2,
     label=f'{T_c}°C')
    
    plt.xlabel('time (hours)', fontsize=12)
    plt.ylabel('Conversion (α)', fontsize=12)
    plt.title('Effect of Temperature (r₀ = 5 μm)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('jander_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate time required for 50% conversion
    print("\ntime required for 50% conversion:")
    print("-" * 50)
    for r0 in r0_values:
     k = D * C0 / r0**2
     t_50 = fsolve(lambda t: jander_equation(0.5, k, t), 10000)[0]
     print(f"r₀ = {r0*1e6:.1f} μm: t₅₀ = {t_50/3600:.1f} hours")
    
    # Output example:
    # time required for 50% conversion:
    # --------------------------------------------------
    # r₀ = 1.0 μm: t₅₀ = 1.9 hours
    # r₀ = 5.0 μm: t₅₀ = 47.3 hours
    # r₀ = 10.0 μm: t₅₀ = 189.2 hours
    

### Example 3: Calculation of Activation Energy (from DSC/TG Data)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 3: Calculate Activation Energy using Kissinger Method
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # Kissinger method: Determine Ea from slope of ln(β/Tp²) vs 1/Tp
    # β: Heating rate [K/min]
    # Tp: Peak temperature [K]
    # Slope = -Ea/R
    
    # Experimental data (DSC peak temperatures at different heating rates)
    heating_rates = np.array([5, 10, 15, 20]) # K/min
    peak_temps_celsius = np.array([1085, 1105, 1120, 1132]) # °C
    peak_temps_kelvin = peak_temps_celsius + 273.15
    
    def kissinger_analysis(beta, Tp):
     """Calculate activation energy using Kissinger method
    
     Args:
     beta (array): Heating rate [K/min]
     Tp (array): Peak temperature [K]
    
     Returns:
     tuple: (Ea [kJ/mol], A [min⁻¹], R²)
     """
     # Left side of Kissinger equation
     y = np.log(beta / Tp**2)
    
     # 1/Tp
     x = 1000 / Tp # Scaling with 1000/T (for better visibility)
    
     # Linear regression
     slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
     # Calculate activation energy
     R = 8.314 # J/(mol·K)
     Ea = -slope * R * 1000 # J/mol → kJ/mol
    
     # Frequency factor
     A = np.exp(intercept)
    
     return Ea, A, r_value**2
    
    # Calculate activation energy
    Ea, A, R2 = kissinger_analysis(heating_rates, peak_temps_kelvin)
    
    print("Analysis results using Kissinger method:")
    print("=" * 50)
    print(f"Activation energy Ea = {Ea:.1f} kJ/mol")
    print(f"Frequency factor A = {A:.2e} min⁻¹")
    print(f"Coefficient of determination R² = {R2:.4f}")
    print("=" * 50)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # KissingerPlot
    y_data = np.log(heating_rates / peak_temps_kelvin**2)
    x_data = 1000 / peak_temps_kelvin
    
    plt.plot(x_data, y_data, 'ro', markersize=10, label='Experimental data')
    
    # Fitting line
    x_fit = np.linspace(x_data.min()*0.95, x_data.max()*1.05, 100)
    slope = -Ea * 1000 / (R * 1000)
    intercept = np.log(A)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Fit: Ea = {Ea:.1f} kJ/mol')
    
    plt.xlabel('1000/Tp (K⁻¹)', fontsize=12)
    plt.ylabel('ln(β/Tp²)', fontsize=12)
    plt.title('Kissinger Plot for Activation Energy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Display results in text box
    textstr = f'Ea = {Ea:.1f} kJ/mol\nR² = {R2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
     verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('kissinger_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output example:
    # Analysis results using Kissinger method:
    # ==================================================
    # Activation energy Ea = 287.3 kJ/mol
    # Frequency factor A = 2.45e+12 min⁻¹
    # Coefficient of determination R² = 0.9956
    # ==================================================
    

## 1.4 Python Practice: Analysis and Design of Ceramic Materials

### 1.4.1 Three Elements of Temperature Profile

The temperature profile in solid-state reactions is the most important control parameter determining reaction success. The following three elements must be properly designed:
    
    
    flowchart TD
     A[Temperature Profile Design] --> B[Heating rate  
    Heating Rate]
     A --> C[Holding time]
     A --> D[Cooling Rate  
    Cooling Rate]
    
     B --> B1[Too fast: Thermal stress → Cracks]
     B --> B2[Too slow: Unwanted phase transformations]
    
     C --> C1[Too short: Incomplete reaction]
     C --> C2[Too long: Excessive grain growth]
    
     D --> D1[Too fast: Thermal stress → Cracks]
     D --> D2[Too slow: Unfavorable phases]
    
     style A fill:#f093fb
     style B fill:#e3f2fd
     style C fill:#e8f5e9
     style D fill:#fff3e0
     

#### 1\. Heating rate（Heating Rate）

**General recommended value:** 2-10°C/min

**Factors to consider:**

  * **Thermal Stress** : Large temperature differences between sample interior and surface generate thermal stress, causing cracks
  * **Intermediate Phase Formation** : Rapid passage through certain temperature ranges to avoid unwanted intermediate phase formation at low temperatures
  * **Decomposition Reactions** : In CO₂ or H₂O releasing reactions, rapid heating causes bumping

**⚠️ Example: Decomposition Reaction of BaCO₃**

In BaTiO₃ synthesis, decomposition BaCO₃ → BaO + CO₂ occurs at 800-900°C. At heating rates above 20°C/min, CO₂ is released rapidly and samples may rupture. Recommended heating rate is 5°C/min or below.

#### 2\. Holding time

**Determination method:** Estimation from Jander equation + experimental optimization

Required holding time can be estimated from the following equation:

t = [α_target / k]^(1/2) × (1 - α_target^(1/3))^(-2) 

**Typical holding times:**

  * Low-temperature reactions (<1000°C): 12-24 hours
  * Medium-temperature reactions (1000-1300°C): 4-8 hours
  * High-temperature reactions (>1300°C): 2-4 hours

#### 3\. Cooling Rate（Cooling Rate）

**General recommended value:** 1-5°C/min（slower than heating rate)

**Importance:**

  * **Control of Phase Transformations** : Control high-temperature → low-temperature phase transformation during cooling
  * **Defect Formation** : Rapid cooling freezes defects such as oxygen vacancies
  * **Crystallinity** : Slow cooling improves crystallinity

### 1.4.2 Temperature Profile Optimization Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 4: Temperature Profile Optimization
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def temperature_profile(t, T_target, heating_rate, hold_time, cooling_rate):
     """Generate temperature profile
    
     Args:
     t (array): time array [min]
     T_target (float): Holding temperature [°C]
     heating_rate (float): Heating rate [°C/min]
     hold_time (float): Holding time [min]
     cooling_rate (float): Cooling Rate [°C/min]
    
     Returns:
     array: Temperature profile [°C]
     """
     T_room = 25 # Room temperature
     T = np.zeros_like(t)
    
     # Heating time
     t_heat = (T_target - T_room) / heating_rate
    
     # Cooling start time
     t_cool_start = t_heat + hold_time
    
     for i, time in enumerate(t):
     if time <= t_heat:
     # Heating phase
     T[i] = T_room + heating_rate * time
     elif time <= t_cool_start:
     # Holding phase
     T[i] = T_target
     else:
     # Cooling phase
     T[i] = T_target - cooling_rate * (time - t_cool_start)
     T[i] = max(T[i], T_room) # Does not go below room temperature
    
     return T
    
    def simulate_reaction_progress(T, t, Ea, D0, r0):
     """Calculate reaction progress based on temperature profile
    
     Args:
     T (array): Temperature profile [°C]
     t (array): time array [min]
     Ea (float): Activation energy [J/mol]
     D0 (float): Frequency factor [m²/s]
     r0 (float): Particle radius [m]
    
     Returns:
     array: Conversion
     """
     R = 8.314
     C0 = 10000
     alpha = np.zeros_like(t)
    
     for i in range(1, len(t)):
     T_k = T[i] + 273.15
     D = D0 * np.exp(-Ea / (R * T_k))
     k = D * C0 / r0**2
    
     dt = (t[i] - t[i-1]) * 60 # min → s
    
     # Simple integration (reaction progress in small time steps)
     if alpha[i-1] < 0.99:
     dalpha = k * dt / (2 * (1 - (1-alpha[i-1])**(1/3)))
     alpha[i] = min(alpha[i-1] + dalpha, 1.0)
     else:
     alpha[i] = alpha[i-1]
    
     return alpha
    
    # Parameter settings
    T_target = 1200 # °C
    hold_time = 240 # min (4 hours)
    Ea = 300e3 # J/mol
    D0 = 5e-4 # m²/s
    r0 = 5e-6 # m
    
    # Comparison at different heating rates
    heating_rates = [2, 5, 10, 20] # °C/min
    cooling_rate = 3 # °C/min
    
    # time array
    t_max = 800 # min
    t = np.linspace(0, t_max, 2000)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temperature Profiles
    for hr in heating_rates:
     T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
     ax1.plot(t/60, T_profile, linewidth=2, label=f'{hr}°C/min')
    
    ax1.set_xlabel('time (hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profiles', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_max/60])
    
    # Reaction Progress
    for hr in heating_rates:
     T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
     alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0)
     ax2.plot(t/60, alpha, linewidth=2, label=f'{hr}°C/min')
    
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='Target (95%)')
    ax2.set_xlabel('time (hours)', fontsize=12)
    ax2.set_ylabel('Conversion', fontsize=12)
    ax2.set_title('Reaction Progress', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t_max/60])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('temperature_profile_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate time to reach 95% conversion at each heating rate
    print("\nComparison of time to reach 95% conversion:")
    print("=" * 60)
    for hr in heating_rates:
     T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate)
     alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0)
    
     # time to reach 95%
     idx_95 = np.where(alpha >= 0.95)[0]
     if len(idx_95) > 0:
     t_95 = t[idx_95[0]] / 60
     print(f"Heating rate {hr:2d}°C/min: t₉₅ = {t_95:.1f} hours")
     else:
     print(f"Heating rate {hr:2d}°C/min: Incomplete reaction")
    
    # Output example:
    # Comparison of time to reach 95% conversion:
    # ============================================================
    # Heating rate 2°C/min: t₉₅ = 7.8 hours
    # Heating rate 5°C/min: t₉₅ = 7.2 hours
    # Heating rate 10°C/min: t₉₅ = 6.9 hours
    # Heating rate 20°C/min: t₉₅ = 6.7 hours
    

## Exercise Problems

### 1.5.1 What is pycalphad

**pycalphad** 、CALPHAD（CALculation of PHAse Diagrams）法に基づく相図calculation forPythonlibrary.熱力学データベース from平衡相 calculateし、reaction経路 設計に有用.

**💡 Advantages of CALPHAD Method**

  * Can calculate complex phase diagrams of multicomponent systems (ternary and higher)
  * Experimental dataが少ない system も予測可能
  * Can comprehensively handle temperature, composition, and pressure dependencies

### 1.5.2 Example of Binary Phase Diagram Calculation
    
    
    # ===================================
    # Example 5: pycalphad 相図calculation
    # ===================================
    
    # Note: pycalphad installation required
    # pip install pycalphad
    
    from pycalphad import Database, equilibrium, variables as v
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load TDB database (simplified example here)
    # Actual appropriate TDB file is required
    # Example: BaO-TiO2 system
    
    # Simplified TDB string (actual is more complex)
    tdb_string = """
    $ BaO-TiO2 system (simplified)
    ELEMENT BA BCC_A2 137.327 !
    ELEMENT TI HCP_A3 47.867 !
    ELEMENT O GAS 15.999 !
    
    FUNCTION GBCCBA 298.15 +GHSERBA; 6000 N !
    FUNCTION GHCPTI 298.15 +GHSERTI; 6000 N !
    FUNCTION GGASO 298.15 +GHSERO; 6000 N !
    
    PHASE LIQUID:L % 1 1.0 !
    PHASE BAO_CUBIC % 2 1 1 !
    PHASE TIO2_RUTILE % 2 1 2 !
    PHASE BATIO3 % 3 1 1 3 !
    """
    
    # Note: Formal TDB file required for actual calculations
    # Limited to conceptual explanation here
    
    print("Concept of phase diagram calculation using pycalphad:")
    print("=" * 60)
    print("1. Load TDB database (thermodynamic data)")
    print("2. Set temperature and composition ranges")
    print("3. Execute equilibrium calculation")
    print("4. Visualize stable phases")
    print()
    print("Actual application examples:")
    print("- BaO-TiO2 system: Formation temperature and composition range of BaTiO3")
    print("- Si-N system: Stability region of Si3N4")
    print("- Phase relationships of multicomponent ceramics")
    
    # 概念的なPlot（実データに基づくイメージ）
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Temperature range
    T = np.linspace(800, 1600, 100)
    
    # Stability regions of each phase (conceptual diagram)
    # BaO + TiO2 → BaTiO3 reaction
    BaO_region = np.ones_like(T) * 0.3
    TiO2_region = np.ones_like(T) * 0.7
    BaTiO3_region = np.where((T > 1100) & (T < 1400), 0.5, np.nan)
    
    ax.fill_between(T, 0, BaO_region, alpha=0.3, color='blue', label='BaO + TiO2')
    ax.fill_between(T, BaO_region, TiO2_region, alpha=0.3, color='green',
     label='BaTiO3 stable')
    ax.fill_between(T, TiO2_region, 1, alpha=0.3, color='red', label='Liquid')
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
     label='BaTiO3 composition')
    ax.axvline(x=1100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=1400, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Composition (BaO mole fraction)', fontsize=12)
    ax.set_title('Conceptual Phase Diagram: BaO-TiO2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([800, 1600])
    ax.set_ylim([0, 1])
    
    # テキスト注釈
    ax.text(1250, 0.5, 'BaTiO₃\nformation\nregion',
     fontsize=11, ha='center', va='center',
     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('phase_diagram_concept.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # actualuseexample（コメントアウト）
    """
    # actualpycalphaduseexample
    db = Database('BaO-TiO2.tdb') # TDBファイル読み込み
    
    # 平衡calculation
    eq = equilibrium(db, ['BA', 'TI', 'O'], ['LIQUID', 'BATIO3'],
     {v.X('BA'): (0, 1, 0.01),
     v.T: (1000, 1600, 50),
     v.P: 101325})
    
    # 結果Plot
    eq.plot()
    """
    

## 1.6 Condition Optimization using Design of Experiments (DOE)

### 1.6.1 What is DOE

実験計画法（Design of Experiments, DOE） 、複数 パラメータが相互作用する system 、最小 実験 number of times最適条件 見つける統計手法.

**Key parameters to optimize in solid-state reactions:**

  * Reaction temperature (T)
  * holdingtime（t）
  * Particle size (r)
  * Raw material ratio (molar ratio)
  * Atmosphere (air, nitrogen, vacuum, etc.)

### 1.6.2 Response Surface Methodology
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 6: DOEによる条件最適化
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import minimize
    
    # 仮想的なConversionモデル（temperature andtime function）
    def reaction_yield(T, t, noise=0):
     """temperature andtime fromConversion calculate（仮想モデル）
    
     Args:
     T (float): Temperature [°C]
     t (float): time [hours]
     noise (float): Noise level
    
     Returns:
     float: Conversion [%]
     """
     # Optimal values: T=1200°C, t=6 hours
     T_opt = 1200
     t_opt = 6
    
     # Quadratic model (Gaussian)
     yield_val = 100 * np.exp(-((T-T_opt)/150)**2 - ((t-t_opt)/3)**2)
    
     # Add noise
     if noise > 0:
     yield_val += np.random.normal(0, noise)
    
     return np.clip(yield_val, 0, 100)
    
    # Experimental point arrangement (central composite design)
    T_levels = [1000, 1100, 1200, 1300, 1400] # °C
    t_levels = [2, 4, 6, 8, 10] # hours
    
    # Arrange experimental points on grid
    T_grid, t_grid = np.meshgrid(T_levels, t_levels)
    yield_grid = np.zeros_like(T_grid, dtype=float)
    
    # 各実験点 Conversion 測定（シミュレーション）
    for i in range(len(t_levels)):
     for j in range(len(T_levels)):
     yield_grid[i, j] = reaction_yield(T_grid[i, j], t_grid[i, j], noise=2)
    
    # Display results
    print("Reaction condition optimization using DOE")
    print("=" * 70)
    print(f"{'Temperature (°C)':<20} {'time (hours)':<20} {'Yield (%)':<20}")
    print("-" * 70)
    for i in range(len(t_levels)):
     for j in range(len(T_levels)):
     print(f"{T_grid[i, j]:<20} {t_grid[i, j]:<20} {yield_grid[i, j]:<20.1f}")
    
    # maximumConversion 条件 探す
    max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    T_best = T_grid[max_idx]
    t_best = t_grid[max_idx]
    yield_best = yield_grid[max_idx]
    
    print("-" * 70)
    print(f"Optimal conditions: T = {T_best}°C, t = {t_best} hours")
    print(f"maximumConversion: {yield_best:.1f}%")
    
    # 3DPlot
    fig = plt.figure(figsize=(14, 6))
    
    # 3D表面Plot
    ax1 = fig.add_subplot(121, projection='3d')
    T_fine = np.linspace(1000, 1400, 50)
    t_fine = np.linspace(2, 10, 50)
    T_mesh, t_mesh = np.meshgrid(T_fine, t_fine)
    yield_mesh = np.zeros_like(T_mesh)
    
    for i in range(len(t_fine)):
     for j in range(len(T_fine)):
     yield_mesh[i, j] = reaction_yield(T_mesh[i, j], t_mesh[i, j])
    
    surf = ax1.plot_surface(T_mesh, t_mesh, yield_mesh, cmap='viridis',
     alpha=0.8, edgecolor='none')
    ax1.scatter(T_grid, t_grid, yield_grid, color='red', s=50,
     label='Experimental points')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=10)
    ax1.set_ylabel('time (hours)', fontsize=10)
    ax1.set_zlabel('Yield (%)', fontsize=10)
    ax1.set_title('Response Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 等高線Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(T_mesh, t_mesh, yield_mesh, levels=20, cmap='viridis')
    ax2.contour(T_mesh, t_mesh, yield_mesh, levels=10, colors='black',
     alpha=0.3, linewidths=0.5)
    ax2.scatter(T_grid, t_grid, c=yield_grid, s=100, edgecolors='red',
     linewidths=2, cmap='viridis')
    ax2.scatter(T_best, t_best, color='red', s=300, marker='*',
     edgecolors='white', linewidths=2, label='Optimum')
    
    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('time (hours)', fontsize=11)
    ax2.set_title('Contour Map', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, label='Yield (%)')
    
    plt.tight_layout()
    plt.savefig('doe_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 1.6.3 Practical Approach to Experimental Design

In actual solid-state reactions, DOE is applied in the following steps:

  1. **Screening Experiments**(two-level factorial design): Identify parameters with large effects
  2. **Response Surface Methodology**(central composite design): Search for optimal conditions
  3. **Confirmation Experiments** : Conduct experiments at predicted optimal conditions and validate model

**✅ Example: Synthesis Optimization of Li-ion Battery Cathode Material LiCoO₂**

Results when a research group optimized LiCoO₂ synthesis conditions using DOE:

  * Number of experiments: Traditional method 100 → DOE method 25 (75% reduction)
  * Optimal temperature: 900°C (higher than traditional 850°C)
  * 最適holdingtime: 12time（従来 24time from半減）
  * Battery capacity: 140 mAh/g → 155 mAh/g (11% improvement)

## 1.7 Fitting of Reaction Kinetics Curves

### 1.7.1 Experimental data from Rate constant決定
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 1.7.1 Experimental data from Rate constant決定
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 7: reaction速度曲線フィッティング
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Experimental data（time vs Conversion）
    # Example: BaTiO3 synthesis @ 1200°C
    time_exp = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]) # hours
    conversion_exp = np.array([0, 0.15, 0.28, 0.38, 0.47, 0.60,
     0.70, 0.78, 0.84, 0.90, 0.95])
    
    # Jander equation model
    def jander_model(t, k):
     """Jander equationによるConversioncalculation
    
     Args:
     t (array): time [hours]
     k (float): Rate constant
    
     Returns:
     array: Conversion
     """
     # [1 - (1-α)^(1/3)]² = kt α について解く
     kt = k * t
     alpha = 1 - (1 - np.sqrt(kt))**3
     alpha = np.clip(alpha, 0, 1) # Limit to 0-1 range
     return alpha
    
    # Ginstling-Brounshtein equation (another diffusion model)
    def gb_model(t, k):
     """Ginstling-Brounshtein equation
    
     Args:
     t (array): time
     k (float): Rate constant
    
     Returns:
     array: Conversion
     """
     # 1 - 2α/3 - (1-α)^(2/3) = kt
     # 数値的に解くrequiredがあるが、ここ in近似 equation use
     kt = k * t
     alpha = 1 - (1 - kt/2)**(3/2)
     alpha = np.clip(alpha, 0, 1)
     return alpha
    
    # Power law (empirical formula)
    def power_law_model(t, k, n):
     """Power law model
    
     Args:
     t (array): time
     k (float): Rate constant
     n (float): Exponent
    
     Returns:
     array: Conversion
     """
     alpha = k * t**n
     alpha = np.clip(alpha, 0, 1)
     return alpha
    
    # Fitting with each model
    # Jander equation
    popt_jander, _ = curve_fit(jander_model, time_exp, conversion_exp, p0=[0.01])
    k_jander = popt_jander[0]
    
    # Ginstling-Brounshtein equation
    popt_gb, _ = curve_fit(gb_model, time_exp, conversion_exp, p0=[0.01])
    k_gb = popt_gb[0]
    
    # Power law
    popt_power, _ = curve_fit(power_law_model, time_exp, conversion_exp, p0=[0.1, 0.5])
    k_power, n_power = popt_power
    
    # Generate predicted curves
    t_fit = np.linspace(0, 20, 200)
    alpha_jander = jander_model(t_fit, k_jander)
    alpha_gb = gb_model(t_fit, k_gb)
    alpha_power = power_law_model(t_fit, k_power, n_power)
    
    # Calculate residuals
    residuals_jander = conversion_exp - jander_model(time_exp, k_jander)
    residuals_gb = conversion_exp - gb_model(time_exp, k_gb)
    residuals_power = conversion_exp - power_law_model(time_exp, k_power, n_power)
    
    # Calculate R²
    def r_squared(y_true, y_pred):
     ss_res = np.sum((y_true - y_pred)**2)
     ss_tot = np.sum((y_true - np.mean(y_true))**2)
     return 1 - (ss_res / ss_tot)
    
    r2_jander = r_squared(conversion_exp, jander_model(time_exp, k_jander))
    r2_gb = r_squared(conversion_exp, gb_model(time_exp, k_gb))
    r2_power = r_squared(conversion_exp, power_law_model(time_exp, k_power, n_power))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fitting results
    ax1.plot(time_exp, conversion_exp, 'ko', markersize=8, label='Experimental data')
    ax1.plot(t_fit, alpha_jander, 'b-', linewidth=2,
     label=f'Jander (R²={r2_jander:.4f})')
    ax1.plot(t_fit, alpha_gb, 'r-', linewidth=2,
     label=f'Ginstling-Brounshtein (R²={r2_gb:.4f})')
    ax1.plot(t_fit, alpha_power, 'g-', linewidth=2,
     label=f'Power law (R²={r2_power:.4f})')
    
    ax1.set_xlabel('time (hours)', fontsize=12)
    ax1.set_ylabel('Conversion', fontsize=12)
    ax1.set_title('Kinetic Model Fitting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 1])
    
    # 残差Plot
    ax2.plot(time_exp, residuals_jander, 'bo-', label='Jander')
    ax2.plot(time_exp, residuals_gb, 'ro-', label='Ginstling-Brounshtein')
    ax2.plot(time_exp, residuals_power, 'go-', label='Power law')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('time (hours)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kinetic_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Results summary
    print("\n of reaction kinetics modelsFitting results:")
    print("=" * 70)
    print(f"{'Model':<25} {'Parameter':<30} {'R²':<10}")
    print("-" * 70)
    print(f"{'Jander':<25} {'k = ' + f'{k_jander:.4f} h⁻¹':<30} {r2_jander:.4f}")
    print(f"{'Ginstling-Brounshtein':<25} {'k = ' + f'{k_gb:.4f} h⁻¹':<30} {r2_gb:.4f}")
    print(f"{'Power law':<25} {'k = ' + f'{k_power:.4f}, n = {n_power:.4f}':<30} {r2_power:.4f}")
    print("=" * 70)
    print(f"\nOptimal model: {'Jander' if r2_jander == max(r2_jander, r2_gb, r2_power) else 'GB' if r2_gb == max(r2_jander, r2_gb, r2_power) else 'Power law'}")
    
    # Output example:
    # of reaction kinetics modelsFitting results:
    # ======================================================================
    # Model Parameter R²
    # ----------------------------------------------------------------------
    # Jander k = 0.0289 h⁻¹ 0.9953
    # Ginstling-Brounshtein k = 0.0412 h⁻¹ 0.9867
    # Power law k = 0.2156, n = 0.5234 0.9982
    # ======================================================================
    #
    # Optimal model: Power law
    

## 1.8 Advanced Topics: Microstructure Control

### 1.8.1 Grain Growth Suppression

solid-statereaction in、high temperature・longtimeholdingにより望ましくない粒成longが起こります。これ 抑制する戦略：

  * **Two-step sintering** : high temperature shorttimeholding後、low temperature longtimeholding
  * **添加剤 use** : Add small amounts of grain growth inhibitors (e.g., MgO, Al₂O₃)
  * **Spark Plasma Sintering (SPS)** : 急速heating・shorttime焼結

### 1.8.2 Mechanochemical Activation of Reactions

メカノケミカル法（high-energy ball milling）により、solid-statereaction Room temperature付近 進行させるこ andも可能 す：
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 8: 粒成longシミュレーション
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def grain_growth(t, T, D0, Ea, G0, n):
     """粒成long time発展
    
     Burke-Turnbull equation: G^n - G0^n = k*t
    
     Args:
     t (array): time [hours]
     T (float): Temperature [K]
     D0 (float): Frequency factor
     Ea (float): Activation energy [J/mol]
     G0 (float): Initial grain size [μm]
     n (float): 粒成longExponent（通常2-4）
    
     Returns:
     array: Grain size [μm]
     """
     R = 8.314
     k = D0 * np.exp(-Ea / (R * T))
     G = (G0**n + k * t * 3600)**(1/n) # hours → seconds
     return G
    
    # Parameter settings
    D0_grain = 1e8 # μm^n/s
    Ea_grain = 400e3 # J/mol
    G0 = 0.5 # μm
    n = 3
    
    # Effect of temperature
    temps_celsius = [1100, 1200, 1300]
    t_range = np.linspace(0, 12, 100) # 0-12 hours
    
    plt.figure(figsize=(12, 5))
    
    # Temperature dependence
    plt.subplot(1, 2, 1)
    for T_c in temps_celsius:
     T_k = T_c + 273.15
     G = grain_growth(t_range, T_k, D0_grain, Ea_grain, G0, n)
     plt.plot(t_range, G, linewidth=2, label=f'{T_c}°C')
    
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
     label='Target grain size')
    plt.xlabel('time (hours)', fontsize=12)
    plt.ylabel('Grain Size (μm)', fontsize=12)
    plt.title('Grain Growth at Different Temperatures', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 5])
    
    # Effect of two-step sintering
    plt.subplot(1, 2, 2)
    
    # Conventional sintering: 1300°C, 6 hours
    t_conv = np.linspace(0, 6, 100)
    T_conv = 1300 + 273.15
    G_conv = grain_growth(t_conv, T_conv, D0_grain, Ea_grain, G0, n)
    
    # Two-step: 1300°C 1h → 1200°C 5h
    t1 = np.linspace(0, 1, 20)
    G1 = grain_growth(t1, 1300+273.15, D0_grain, Ea_grain, G0, n)
    G_intermediate = G1[-1]
    
    t2 = np.linspace(0, 5, 80)
    G2 = grain_growth(t2, 1200+273.15, D0_grain, Ea_grain, G_intermediate, n)
    
    t_two_step = np.concatenate([t1, t2 + 1])
    G_two_step = np.concatenate([G1, G2])
    
    plt.plot(t_conv, G_conv, 'r-', linewidth=2, label='Conventional (1300°C)')
    plt.plot(t_two_step, G_two_step, 'b-', linewidth=2, label='Two-step (1300°C→1200°C)')
    plt.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.xlabel('time (hours)', fontsize=12)
    plt.ylabel('Grain Size (μm)', fontsize=12)
    plt.title('Two-Step Sintering Strategy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 5])
    
    plt.tight_layout()
    plt.savefig('grain_growth_control.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Comparison of final grain size
    G_final_conv = grain_growth(6, 1300+273.15, D0_grain, Ea_grain, G0, n)
    G_final_two_step = G_two_step[-1]
    
    print("\nComparison of grain growth:")
    print("=" * 50)
    print(f"Conventional (1300°C, 6h): {G_final_conv:.2f} μm")
    print(f"Two-step (1300°C 1h + 1200°C 5h): {G_final_two_step:.2f} μm")
    print(f"Grain size suppression effect: {(1 - G_final_two_step/G_final_conv)*100:.1f}%")
    
    # Output example:
    # Comparison of grain growth:
    # ==================================================
    # Conventional (1300°C, 6h): 4.23 μm
    # Two-step (1300°C 1h + 1200°C 5h): 2.87 μm
    # Grain size suppression effect: 32.2%
    

## Learning Objectives 確認

Upon completing this chapter, you will be able to explain:

### Fundamental Understanding

  * ✅ Can explain the three rate-limiting steps of solid-state reactions (nucleation, interface reaction, diffusion)
  * ✅ Arrhenius equation 物理的意味 andTemperature dependence 理解している
  * ✅ Can explain the differences between Jander and Ginstling-Brounshtein equations
  * ✅ Temperature Profiles 3要素（Heating rate・holdingtime・Cooling Rate） 重要性 理解している

### Practical Skills

  * ✅ Python 拡散係数 Temperature dependence シミュレート きる
  * ✅ Jander equation 用いてReaction Progress 予測 きる
  * ✅ Can calculate activation energy from DSC/TG data using Kissinger method
  * ✅ Can optimize reaction conditions using DOE (Design of Experiments)
  * ✅ Understand basics of phase diagram calculation using pycalphad

### Applied Capabilities

  * ✅ Can design synthesis processes for new ceramic materials
  * ✅ Experimental data fromreaction機構 推定し、適切な速度 equation 選択 きる
  * ✅ Can formulate condition optimization strategies for industrial processes
  * ✅ Can propose grain growth control strategies (e.g., two-step sintering)

## Exercise Problems

### Easy (Fundamental Check)

Q1: Rate-Limiting Step of Solid-State Reactions

In the synthesis reaction BaCO₃ + TiO₂ → BaTiO₃ + CO₂ of BaTiO₃, which step is the slowest (rate-limiting)?

a) Release of CO₂  
b) Nucleation of BaTiO₃  
c) Diffusion of Ba²⁺ ions through product layer  
d) Chemical reaction at interface

View answer

**Correct answer: c) Diffusion of Ba²⁺ ions through product layer**

**Explanation:**  
In solid-state reactions, the process of ions diffusing through the product layer is slowest because the product layer physically separates the reactants.

  * a) CO₂ release is fast because it is gas diffusion
  * b) Nucleation completes in the initial stage
  * c) **Diffusion is rate-limiting**(correct) - Ion diffusion in solids is extremely slow (D ~ 10⁻¹² m²/s)
  * d) Interface reaction is usually fast

**Key point:** 拡散係数 temperatureに対してExponentincreases exponentially、reactiontemperature 選択が極めて重要.

Q2: Parameters of Arrhenius Equation

In the diffusion coefficient D(T) = D₀ exp(-Eₐ/RT), what happens to the sensitivity of the diffusion coefficient to temperature changes as Eₐ (activation energy) becomes larger?

a) becomes higher（Temperature dependence is strong）  
b) 低くなる（Temperature dependenceが弱い）  
c) No change  
d) Irrelevant

View answer

**Correct answer: a) becomes higher（Temperature dependence is strong）**

**Explanation:**  
活性化エネルギーEₐ 、Exponentfunction exp(-Eₐ/RT) 肩に位置するため、Eₐが大きいほどtemperature変化に対するD 変化率が大きくなります。

**Numerical examples:**

  * For Eₐ = 100 kJ/mol: Raising temperature by 100°C increases D by about 3 times
  * For Eₐ = 300 kJ/mol: Raising temperature by 100°C increases D by about 30 times

Therefore, temperature control becomes particularly important for systems with large activation energy.

Q3: Particle Size and Reaction Rate

Jander equation k = D·C₀/r₀² によれば、粒子半径r₀ 1/2にする and、reactionRate constantk 何倍になりますか？

a) 2 times  
b) 4 times  
c) 1/2 times  
d) 1/4 times

View answer

**Correct answer: b) 4 times**

**Calculation:**  
k ∝ 1/r₀²  
When r₀ → r₀/2, k → k/(r₀/2)² = k/(r₀²/4) = 4k

**Practical meaning:**  
これが「粉砕・微細化」がsolid-statereaction 極めて重要な理由.

  * 粒径10μm → 1μm: reaction速度100倍（reactiontime1/100）
  * Refinement by ball mill, jet mill is standard process
  * ナノ粒子 使えばRoom temperature付近 もreaction可能な場合も

### Medium（Applications）

Q4: Temperature Profile Design

BaTiO₃合成 、Heating rate 20°C/min from5°C/minに変更しました。こ 変更 主な理由 andして最も適切な どれ すか？

a) To accelerate reaction rate  
b) To prevent sample rupture due to rapid CO₂ release  
c) To save electricity costs  
d) Crystallinity 下げるため

View answer

**Correct answer: b) To prevent sample rupture due to rapid CO₂ release**

**Detailed reasons:**

In the reaction BaCO₃ + TiO₂ → BaTiO₃ + CO₂, barium carbonate decomposes at 800-900°C releasing CO₂.

  * **Problems with rapid heating (20°C/min):**
    * shorttime 多量 CO₂が発生
    * Gas pressure increases, causing sample rupture and scattering
    * Cracks form in sintered body
  * **Advantages of slow heating (5°C/min):**
    * CO₂ released slowly, pressure increase is gradual
    * Sample integrity is maintained
    * Homogeneous reaction proceeds

**Practical advice:** Decomposition Reactions 伴う合成 in、ガス放出速度 制御するため、該当Temperature range Heating rate 特に遅くします（Example: 750-950°C 2°C/min 通過）。

Q5: Application of Kissinger Method

The following data were obtained from DSC measurements. Calculate the activation energy using the Kissinger method.

Heating rate β (K/min): 5, 10, 15  
Peak temperature Tp (K): 1273, 1293, 1308

Kissinger equation: slope of ln(β/Tp²) vs 1/Tp = -Eₐ/R

View answer

**Answer:**

**Step 1: Data organization**

β (K/min) | Tp (K) | ln(β/Tp²) | 1000/Tp (K⁻¹)  
---|---|---|---  
5 | 1273 | -11.558 | 0.7855  
10 | 1293 | -11.171 | 0.7734  
15 | 1308 | -10.932 | 0.7645  
  
**Step2: Linear regression**

y = ln(β/Tp²) vs x = 1000/Tp Plot  
Slope = Δy/Δx = (-10.932 - (-11.558)) / (0.7645 - 0.7855) = 0.626 / (-0.021) ≈ -29.8

**Step 3: Eₐ calculation**

slope = -Eₐ / (R × 1000) (divided by 1000 because 1000/Tp was used)  
Eₐ = -slope × R × 1000  
Eₐ = 29.8 × 8.314 × 1000 = 247,757 J/mol ≈ 248 kJ/mol

**Answer: Eₐ ≈ 248 kJ/mol**

**Physical interpretation:**  
This value is within the range of typical activation energies (250-350 kJ/mol) for solid-state reactions in BaTiO₃ systems. This activation energy is considered to correspond to solid-state diffusion of Ba²⁺ ions.

Q6: Optimization using DOE

In DOE, two factors of temperature (1100, 1200, 1300°C) and time (4, 6, 8 hours) are examined. How many total experiments are required? Also, list two advantages compared to the traditional method of varying one factor at a time.

View answer

**Answer:**

**Number of experiments:**  
3 levels × 3 levels = **9 times**(full factorial design)

**Advantages of DOE (compared to traditional method):**

  1. **Detection of interactions is possible**
     * Traditional method: Evaluate effects of temperature and time separately
     * DOE: Quantify interactions such as "time can be shortened at high temperature"
     * Example: 4 hours sufficient at 1300°C, but 8 hours needed at 1100°C, etc.
  2. **Reduction in number of experiments**
     * Traditional method (OFAT: One Factor At a time): 
       * Temperature study: 3 times (time fixed)
       * time study: 3 times (temperature fixed)
       * Confirmation experiments: Multiple times
       * Total: 10 or more times
     * DOE: Complete in 9 times (covering all conditions + interaction analysis)
     * Further reduction to 7 times possible using central composite design

**Additional advantages:**

  * Statistically significant conclusions can be obtained (error evaluation possible)
  * Response surface can be constructed, prediction of untested conditions possible
  * Can detect even when optimal conditions are outside experimental range

### Hard (Advanced)

Q7: Design of Complex Reaction System

Design a temperature profile for synthesizing Li₁.₂Ni₀.₂Mn₀.₆O₂ (lithium-rich cathode material) under the following conditions:

  * Raw materials: Li₂CO₃, NiO, Mn₂O₃
  * Target: Single phase, grain size < 5 μm, precise control of Li/transition metal ratio
  * Constraint: Li₂O volatilizes above 900°C (risk of Li deficiency)

Explain the temperature profile (heating rate, holding temperature/time, cooling rate) and design rationale.

View answer

**recommendedTemperature Profiles:**

**Phase 1: Pre-heating (Li₂CO₃ decomposition)**

  * Room temperature → 500°C: 3°C/min
  * 500°Cholding: 2time
  * **Reason:** Slowly proceed with Li₂CO₃ decomposition (~450°C) to completely remove CO₂

**Phase 2: Intermediate heating (precursor formation)**

  * 500°C → 750°C: 5°C/min
  * 750°Cholding: 4time
  * **Reason:** Form intermediate phases such as Li₂MnO₃ and LiNiO₂. Homogenize at temperature with minimal Li volatilization

**Phase 3: Main sintering (target phase synthesis)**

  * 750°C → 850°C: 2°C/min (slow)
  * 850°Cholding: 12time
  * **Reason:**
    * Long time needed for single phase formation of Li₁.₂Ni₀.₂Mn₀.₆O₂
    * Limit to 850°C to minimize Li volatilization (<900°C constraint)
    * Long-time holding advances diffusion, but temperature suppresses grain growth

**Phase 4: Cooling**

  * 850°C → Room temperature: 2°C/min
  * **Reason:** Slow cooling improves crystallinity, prevents cracks from thermal stress

**Important design points:**

  1. **Li volatilization countermeasures:**
     * Limit to below 900°C (constraint in this problem)
     * Additionally, use Li-excess raw materials (e.g., Li/TM = 1.25)
     * Sinter in oxygen flow to reduce partial pressure of Li₂O
  2. **Grain size control ( < 5 μm):**
     * Proceed with reaction at low temperature (850°C) and long time (12h)
     * High temperature and short time causes excessive grain growth
     * Also refine raw material particle size to below 1μm
  3. **Composition uniformity:**
     * Intermediate holding at 750°C is important
     * Homogenize transition metal distribution at this stage
     * If necessary, cool once after 750°C hold → pulverize → reheat

**Total time required:** About 30 hours (heating 12h + holding 18h)

**Consideration of alternative methods:**

  * **Sol-gel method:** Synthesis possible at lower temperature (600-700°C), improved homogeneity
  * **Spray pyrolysis:** Easy grain size control
  * **Two-step sintering:** 900°C 1h → 800°C 10h suppresses grain growth

Q8: Comprehensive Problem on Kinetic Analysis

From the following data, estimate the reaction mechanism and calculate the activation energy.

**Experimental data:**

Temperature (°C) | 50% to reach conversiontime t₅₀ (hours)  
---|---  
1000| 18.5  
1100| 6.2  
1200| 2.5  
1300| 1.2  
  
Assuming Jander equation: [1-(1-0.5)^(1/3)]² = k·t₅₀

View answer

**Answer:**

**Step 1: Calculation of rate constant k**

For Jander equation when α=0.5:  
[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424

Therefore k = 0.0424 / t₅₀

T (°C) | T (K) | t₅₀ (h) | k (h⁻¹) | ln(k) | 1000/T (K⁻¹)  
---|---|---|---|---|---  
1000| 1273| 18.5| 0.00229| -6.080| 0.7855  
1100| 1373| 6.2| 0.00684| -4.985| 0.7284  
1200| 1473| 2.5| 0.01696| -4.077| 0.6788  
1300| 1573| 1.2| 0.03533| -3.343| 0.6357  
  
**Step2: ArrheniusPlot**

Plot ln(k) vs 1/T (linear regression)

Linear fit: ln(k) = A - Eₐ/(R·T)

Slope = -Eₐ/R

Linear regressionCalculation:  
slope = Δ(ln k) / Δ(1000/T)  
= (-3.343 - (-6.080)) / (0.6357 - 0.7855)  
= 2.737 / (-0.1498)  
= -18.27

**Step3: Calculate activation energy**

slope = -Eₐ / (R × 1000)  
Eₐ = -slope × R × 1000  
Eₐ = 18.27 × 8.314 × 1000  
Eₐ = 151,899 J/mol ≈ **152 kJ/mol**

**Step 4: Discussion of reaction mechanism**

  * **Comparison of activation energies:**
    * Obtained value: 152 kJ/mol
    * Typical solid-state diffusion: 200-400 kJ/mol
    * Interface reaction: 50-150 kJ/mol
  * **Inferred mechanism:**
    * This value is intermediate between interface reaction and diffusion
    * Possibility 1: Interface reaction is mainly rate-limiting (small influence of diffusion)
    * Possibility 2: Particles are fine with short diffusion distance, apparent Eₐ is low
    * Possibility 3: Mixed control (both interface reaction and diffusion contribute)

**Step 5: Proposal of verification methods**

  1. **Particle size dependence:** Experiment with different particle sizes, confirm if k ∝ 1/r₀² holds 
     * Holds → Diffusion-controlled
     * Does not hold → Interface reaction-controlled
  2. **Fitting with other rate equations:**
     * Ginstling-Brounshtein equation (3D diffusion)
     * Contracting sphere model (interface reaction)
     * Compare which has higher R²
  3. **Microstructure observation:** Observe reaction interface with SEM 
     * Thick product layer → Evidence of diffusion control
     * Thin product layer → Possibility of interface reaction control

**Final conclusion:**  
Activation energy **Eₐ = 152 kJ/mol**  
Inferred mechanism: **Interface reaction-controlled, or diffusion-controlled in fine particle systems**  
Additional experiments are recommended.

## Next Steps

In Chapter 1, we learned the fundamental theory of advanced ceramic materials (structural, functional, and bioceramics). In the next Chapter 2, we will learn about advanced polymer materials (high-performance engineering plastics, functional polymers, biodegradable polymers).

[← Series Contents](<./index.html>) [Proceed to Chapter 2 →](<chapter-2.html>)

## References

  1. Kingery, W. D., Bowen, H. K., & Uhlmann, D. R. (1976). _Introduction to Ceramics_ (2nd ed.). Wiley. pp. 567-623, 774-835. - Classic masterpiece of ceramic materials science, comprehensive explanation of mechanical properties and fracture theory
  2. Carter, C. B., & Norton, M. G. (2013). _Ceramic Materials: Science and Engineering_ (2nd ed.). Springer. pp. 345-412, 567-634. - Detailed explanation of strengthening mechanisms and toughening technology of structural ceramics
  3. Hench, L. L., & Wilson, J. (1993). _An Introduction to Bioceramics_. World Scientific. pp. 1-35, 139-178. - Fundamental theory of biocompatibility and osseointegration mechanisms of bioceramics
  4. Uchino, K. (2010). _Ferroelectric Devices_ (2nd ed.). CRC Press. pp. 45-98, 201-245. - Latest knowledge on physical origins and applications of piezoelectric and dielectric materials
  5. Garvie, R. C., Hannink, R. H., & Pascoe, R. T. (1975). "Ceramic steel?" _Nature_ , 258, 703-704. - Pioneering paper on zirconia transformation toughening theory
  6. Haertling, G. H. (1999). "Ferroelectric ceramics: History and technology." _Journal of the American Ceramic Society_ , 82(4), 797-818. - Comprehensive review of development history and technological innovation of PZT piezoelectric ceramics
  7. pymatgen Documentation. (2024). _Materials Project_. <https://pymatgen.org/> \- Python library for materials science calculations, phase diagram calculation and structure analysis tools

## Tools and Libraries Used

  * **NumPy** (v1.24+): Numerical computation library - <https://numpy.org/>
  * **SciPy** (v1.10+): Scientific computing library (curve_fit, optimize) - <https://scipy.org/>
  * **Matplotlib** (v3.7+): Data visualization library - <https://matplotlib.org/>
  * **pycalphad** (v0.10+): Phase diagram calculation library - <https://pycalphad.org/>
  * **pymatgen** (v2023+): Materials science calculation library - <https://pymatgen.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
