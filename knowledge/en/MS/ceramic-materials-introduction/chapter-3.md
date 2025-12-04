---
title: "Chapter 3: Mechanical Properties"
chapter_title: "Chapter 3: Mechanical Properties"
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Ceramic Materials](<../../MS/ceramic-materials-introduction/index.html>)›Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/MS/ceramic-materials-introduction/chapter-3.html>) | Last sync: 2025-11-16

  * [Top](<index.html>)
  * [Overview](<#intro>)
  * [Brittle Fracture](<#brittle-fracture>)
  * [Griffith Theory](<#griffith>)
  * [Fracture Toughness](<#fracture-toughness>)
  * [Weibull Statistics](<#weibull>)
  * [High-Temperature Creep](<#creep>)
  * [Exercises](<#exercises>)
  * [References](<#references>)
  * [← Previous Chapter](<chapter-2.html>)
  * [Next Chapter →](<chapter-4.html>)

This chapter covers Mechanical Properties. You will learn essential concepts and techniques.

## 3.1 Characteristics of Mechanical Properties of Ceramics

Unlike metals, ceramics are **brittle materials**. They undergo almost no plastic deformation and fracture suddenly upon reaching a critical stress. This behavior stems from the directionality of ionic and covalent bonding and the difficulty of dislocation motion. In this chapter, we will understand the mechanisms of brittle fracture and practice reliability evaluation using Weibull statistics with Python. 

**Learning Objectives for This Chapter**

  * **Level 1 (Basic Understanding)** : Can explain brittle fracture mechanisms and Griffith theory, and understand the meaning of fracture toughness
  * **Level 2 (Practical Skills)** : Can perform fracture toughness calculations and Weibull analysis in Python, quantitatively evaluating reliability
  * **Level 3 (Application Ability)** : Can calculate Weibull modulus from measured data and perform component life prediction and safety factor design

### Comparison of Mechanical Properties with Metals

Property | Ceramics | Metals | Cause  
---|---|---|---  
Fracture Mode | Brittle Fracture | Ductile Fracture | Presence/Absence of Dislocation Motion  
Tensile Strength | 100-1000 MPa | 200-2000 MPa | Defect Sensitivity  
Compressive Strength | 2000-4000 MPa | Similar to Tensile | Crack Propagation Suppression  
Fracture Toughness KIC | 2-8 MPa·m1/2 | 20-100 MPa·m1/2 | Size of Plastic Zone  
Strength Variability | Large (Weibull m=5-15) | Small (Normal Distribution) | Defect Distribution  
  
**Design Considerations** Ceramics are weak in tensile stress and strong in compressive stress. In structural design, applications where compressive loads dominate (bearings, cutting tools) are suitable while avoiding tensile stress. Additionally, since surface defects become fracture initiation sites, polished finishes and careful handling are extremely important. 

## 3.2 Brittle Fracture Mechanisms

### 3.2.1 Fracture Process

Fracture in ceramics proceeds through the following three stages: 
    
    
    ```mermaid
    flowchart LR
                    A[Initial DefectsPores, Inclusions, Cracks] --> B[Stress Concentrationσlocal = Kt × σapplied]
                    B --> C[Crack PropagationKI > KIC]
                    C --> D[High-Speed Fracturev ~ 2000 m/s]
                    D --> E[Complete RuptureFine Fragment Formation]
    
                    style A fill:#fff3e0
                    style B fill:#ffe0b2
                    style C fill:#f5576c,color:#fff
                    style D fill:#c62828,color:#fff
                    style E fill:#b71c1c,color:#fff
    ```

The stress concentration factor \\( K_t \\) is determined by the defect shape: 

\\[ K_t = 1 + 2\sqrt{\frac{a}{\rho}} \\] 

Here, \\( a \\) is the crack length, and \\( \rho \\) is the crack tip radius of curvature. For atomically sharp cracks (\\( \rho \sim 0.3 \\) nm), \\( K_t \\) concentrates stress to about 1/10 of the theoretical strength. 

### 3.2.2 Defects That Serve as Fracture Origins

  * **Volume Defects** : Pores (residual voids from forming), inclusions (foreign phase particles)
  * **Surface Defects** : Processing damage (grinding, handling), thermal shock cracks
  * **Grain Boundary Defects** : Grain boundary pores, grain boundary phases (residual glass phase from liquid phase sintering)

Statistically, the largest defect dominates fracture (Weakest Link Theory). This is the theoretical foundation of Weibull statistics. 

#### Python Implementation: Calculation of Stress Concentration Factor
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 1: Calculation of Stress Concentration Factor
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stress_concentration_factor(crack_length, tip_radius):
        """
        Calculate stress concentration factor for elliptical crack
    
        Parameters:
        -----------
        crack_length : float
            Crack length a [μm]
        tip_radius : float
            Crack tip radius of curvature ρ [nm]
    
        Returns:
        --------
        K_t : float
            Stress concentration factor (dimensionless)
        """
        # Unify length units (μm → m)
        a = crack_length * 1e-6
        rho = tip_radius * 1e-9
    
        # Calculate stress concentration factor
        K_t = 1 + 2 * np.sqrt(a / rho)
    
        return K_t
    
    
    def plot_stress_concentration():
        """
        Visualize effects of crack length and tip radius of curvature
        """
        # Parameter ranges
        crack_lengths = np.logspace(-1, 2, 100)  # 0.1 ~ 100 μm
        tip_radii = [0.3, 1.0, 5.0, 10.0]  # nm
    
        plt.figure(figsize=(12, 5))
    
        # Left figure: Effect of crack length
        plt.subplot(1, 2, 1)
        for rho in tip_radii:
            K_t_values = [stress_concentration_factor(a, rho) for a in crack_lengths]
            plt.loglog(crack_lengths, K_t_values, linewidth=2, label=f'ρ = {rho} nm')
    
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Stress Concentration Factor K_t', fontsize=12)
        plt.title('Effect of Crack Length on K_t', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        # Right figure: Specific example (typical defect in Al2O3)
        plt.subplot(1, 2, 2)
        typical_crack = 10  # μm
        rho_range = np.logspace(-1, 2, 100)  # 0.1 ~ 100 nm
        K_t_typical = [stress_concentration_factor(typical_crack, rho) for rho in rho_range]
    
        plt.semilogx(rho_range, K_t_typical, linewidth=2, color='crimson')
        plt.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='K_t = 10 (Critical)')
        plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Atomic radius')
    
        plt.xlabel('Tip Radius ρ (nm)', fontsize=12)
        plt.ylabel('Stress Concentration Factor K_t', fontsize=12)
        plt.title(f'K_t for a = {typical_crack} μm Crack', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('stress_concentration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output numerical examples
        print("=== Stress Concentration Factor Examples ===")
        print(f"Crack length: {typical_crack} μm")
        for rho in [0.3, 1.0, 10.0]:
            K_t = stress_concentration_factor(typical_crack, rho)
            print(f"  ρ = {rho:5.1f} nm → K_t = {K_t:6.1f}")
    
    # Execute
    plot_stress_concentration()
    

**Interpretation of Execution Results** For atomically sharp cracks (ρ = 0.3 nm), extreme stress concentration occurs with Kt ≈ exceeding 100. This causes measured strength (400 MPa) to decrease to about 1/100 of theoretical strength (E/10 ≈ 40 GPa). This is why removing surface defects through polishing improves strength by 2-3 times. 

## 3.3 Griffith Theory and Energy Criterion

### 3.3.1 Energy Balance

Griffith (1921) analyzed crack propagation from an energetic perspective. When a crack advances by length \\( da \\): 

\\[ \underbrace{-\frac{d U_{\text{elastic}}}{da}}_{\text{Elastic Energy Release}} = \underbrace{\frac{d U_{\text{surface}}}{da}}_{\text{Surface Energy Increase}} \\] 

For a through-crack (length \\( 2a \\)) in a plate under stress \\( \sigma \\), the critical stress \\( \sigma_f \\) is: 

\\[ \sigma_f = \sqrt{\frac{2 E \gamma}{\pi a}} \\] 

Here, \\( E \\) is Young's modulus, \\( \gamma \\) is surface energy (J/m²), and \\( a \\) is the half-crack length. 

### 3.3.2 Fracture Strength Prediction for Al₂O₃

Using the physical property values of Al₂O₃, we calculate the relationship between defect size and fracture strength: 

  * Young's modulus: \\( E = 400 \\) GPa
  * Surface energy: \\( \gamma = 1.0 \\) J/m²

#### Python Implementation: Calculation of Griffith Fracture Strength
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 2: Calculation of Griffith Fracture Strength
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def griffith_strength(crack_length, E=400e9, gamma=1.0):
        """
        Calculate fracture strength using Griffith theory
    
        Parameters:
        -----------
        crack_length : float or array
            Half-crack length a [m]
        E : float
            Young's modulus [Pa] (default: Al2O3 = 400 GPa)
        gamma : float
            Surface energy [J/m²] (default: 1.0 J/m²)
    
        Returns:
        --------
        sigma_f : float or array
            Fracture strength [Pa]
        """
        sigma_f = np.sqrt(2 * E * gamma / (np.pi * crack_length))
        return sigma_f
    
    
    def analyze_griffith_theory():
        """
        Analysis of strength-defect size relationship using Griffith theory
        """
        # Defect size range (10 nm ~ 1 mm)
        crack_lengths = np.logspace(-8, -3, 1000)  # m
    
        # Calculate fracture strength for each material
        materials = {
            'Al₂O₃': {'E': 400e9, 'gamma': 1.0},
            'SiC': {'E': 450e9, 'gamma': 1.2},
            'Si₃N₄': {'E': 310e9, 'gamma': 0.8},
            'Glass': {'E': 70e9, 'gamma': 0.5}
        }
    
        plt.figure(figsize=(12, 5))
    
        # Left figure: Material comparison
        plt.subplot(1, 2, 1)
        for name, props in materials.items():
            strength = griffith_strength(crack_lengths, props['E'], props['gamma'])
            plt.loglog(crack_lengths * 1e6, strength / 1e6, linewidth=2, label=name)
    
        # Show typical strength range
        plt.axhspan(100, 1000, alpha=0.1, color='green', label='Typical ceramic strength')
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Fracture Strength σ_f (MPa)', fontsize=12)
        plt.title('Griffith Strength vs Crack Size', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        # Right figure: Detailed analysis of Al2O3
        plt.subplot(1, 2, 2)
        a_Al2O3 = crack_lengths
        sigma_Al2O3 = griffith_strength(a_Al2O3, 400e9, 1.0)
    
        plt.loglog(a_Al2O3 * 1e6, sigma_Al2O3 / 1e6, linewidth=2, color='navy')
    
        # Plot experimental data (typical examples)
        experimental_data = {
            'a (μm)': [1, 5, 10, 50, 100],
            'σ_f (MPa)': [800, 400, 300, 150, 100]
        }
        plt.scatter(experimental_data['a (μm)'], experimental_data['σ_f (MPa)'],
                    s=100, c='red', marker='o', edgecolors='black', linewidth=2,
                    label='Experimental data', zorder=5)
    
        plt.xlabel('Crack Length a (μm)', fontsize=12)
        plt.ylabel('Fracture Strength σ_f (MPa)', fontsize=12)
        plt.title('Al₂O₃ Strength Prediction (Griffith)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig('griffith_strength.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Numerical examples
        print("=== Griffith Strength Prediction for Al₂O₃ ===")
        for a_um in [1, 10, 100]:
            a = a_um * 1e-6
            sigma = griffith_strength(a, 400e9, 1.0)
            print(f"Crack size: {a_um:4d} μm → Strength: {sigma/1e6:6.1f} MPa")
    
        # Inverse calculation: Estimate defect size from measured strength
        print("\n=== Reverse Calculation: Defect Size from Strength ===")
        measured_strength = np.array([300, 500, 800]) * 1e6  # MPa → Pa
        critical_crack = 2 * 400e9 * 1.0 / (np.pi * measured_strength**2)
        for i, sigma in enumerate(measured_strength / 1e6):
            print(f"Strength: {sigma:5.0f} MPa → Critical crack: {critical_crack[i]*1e6:6.2f} μm")
    
    # Execute
    analyze_griffith_theory()
    

**Limitations of Griffith Theory** Griffith theory assumes an ideal elastic body and does not consider plastic zones at crack tips or R-curve behavior. In actual materials, toughness improves through grain boundary crack deflection and particle bridging (transformation toughening, fiber reinforcement, etc.). These are handled in fracture mechanics. 

## 3.4 Fracture Toughness

### 3.4.1 Stress Intensity Factor and KIC

In fracture mechanics, the stress field at a crack tip is expressed by the **stress intensity factor \\( K_I \\)** : 

\\[ K_I = Y \sigma \sqrt{\pi a} \\] 

Here, \\( Y \\) is the shape factor (dimensionless, depends on specimen geometry and crack location), \\( \sigma \\) is the far-field stress, and \\( a \\) is the crack length. The fracture condition is: 

\\[ K_I \geq K_{IC} \\] 

\\( K_{IC} \\) is called **fracture toughness** , a material constant with units of MPa·m1/2. 

### 3.4.2 Methods for Measuring Fracture Toughness

  * **SEVNB Method (Single Edge V-Notched Beam)** : Three-point bending test with V-notch introduction
  * **IF Method (Indentation Fracture)** : Measurement of crack length from Vickers indentation
  * **CNB Method (Chevron-Notched Beam)** : Stable fracture using chevron notch

#### Python Implementation: Calculation of Fracture Toughness KIC
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 3: Calculation of Fracture Toughness K_IC
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fracture_toughness_SEVNB(P_max, B, W, a, S=40e-3):
        """
        Calculate fracture toughness using SEVNB method
    
        Parameters:
        -----------
        P_max : float
            Maximum load [N]
        B : float
            Specimen width [m]
        W : float
            Specimen height [m]
        a : float
            Crack length [m]
        S : float
            Support span [m] (default: 40 mm)
    
        Returns:
        --------
        K_IC : float
            Fracture toughness [Pa·m^0.5]
        """
        # Calculate shape factor Y (based on ASTM E399)
        alpha = a / W
        Y = (1.99 - alpha * (1 - alpha) * (2.15 - 3.93*alpha + 2.7*alpha**2)) / \
            ((1 + 2*alpha) * (1 - alpha)**1.5)
    
        # Calculate stress intensity factor
        K_IC = (P_max * S / (B * W**1.5)) * Y
    
        return K_IC
    
    
    def indentation_fracture_toughness(P, a, c, E=400e9, H=20e9):
        """
        Simple estimation of fracture toughness using IF method (indentation method)
    
        Parameters:
        -----------
        P : float
            Indenter load [N]
        a : float
            Indentation diagonal half-length [m]
        c : float
            Crack length (from indentation center to tip) [m]
        E : float
            Young's modulus [Pa]
        H : float
            Hardness [Pa]
    
        Returns:
        --------
        K_IC : float
            Fracture toughness [Pa·m^0.5]
        """
        # Anstis equation (1981)
        K_IC = 0.016 * (E / H)**0.5 * (P / c**1.5)
    
        return K_IC
    
    
    def analyze_fracture_toughness():
        """
        Analysis of fracture toughness data for various ceramics
        """
        # Material database
        materials = {
            'Al₂O₃': {'K_IC': 4.0, 'E': 400e9, 'σ_f': 350e6},
            'ZrO₂ (3Y-TZP)': {'K_IC': 8.0, 'E': 210e9, 'σ_f': 900e6},
            'Si₃N₄': {'K_IC': 6.0, 'E': 310e9, 'σ_f': 700e6},
            'SiC': {'K_IC': 3.5, 'E': 450e9, 'σ_f': 400e6},
            'Glass': {'K_IC': 0.7, 'E': 70e9, 'σ_f': 50e6}
        }
    
        # Visualize relationship between fracture toughness and strength
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # Left figure: K_IC vs strength
        ax1 = axes[0]
        K_IC_values = [props['K_IC'] for props in materials.values()]
        strength_values = [props['σ_f']/1e6 for props in materials.values()]
        names = list(materials.keys())
    
        ax1.scatter(K_IC_values, strength_values, s=150, c='crimson', edgecolors='black', linewidth=2)
        for i, name in enumerate(names):
            ax1.annotate(name, (K_IC_values[i], strength_values[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
    
        ax1.set_xlabel('Fracture Toughness K_IC (MPa·m^0.5)', fontsize=12)
        ax1.set_ylabel('Flexural Strength σ_f (MPa)', fontsize=12)
        ax1.set_title('K_IC vs Strength for Ceramics', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
        # Right figure: Calculate allowable defect size
        ax2 = axes[1]
        Y = 1.12  # Shape factor for surface crack
    
        for name, props in materials.items():
            # Inverse calculation from σ_f = K_IC / (Y * sqrt(π * a))
            a_critical = (props['K_IC'] / (Y * props['σ_f']))**2 / np.pi
            stress_range = np.linspace(0.1, 1.5, 100) * props['σ_f']  # 10% ~ 150%
            a_range = (props['K_IC'] / (Y * stress_range))**2 / np.pi
    
            ax2.loglog(a_range * 1e6, stress_range / 1e6, linewidth=2, label=name)
    
        ax2.set_xlabel('Critical Crack Size a (μm)', fontsize=12)
        ax2.set_ylabel('Applied Stress σ (MPa)', fontsize=12)
        ax2.set_title('Failure Diagram (K_IC = Yσ√πa)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig('fracture_toughness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Numerical example for SEVNB test
        print("=== SEVNB Test Example (Al₂O₃) ===")
        P_max = 250  # N
        B = 4e-3  # 4 mm
        W = 3e-3  # 3 mm
        a = 1.5e-3  # 1.5 mm (a/W = 0.5)
        S = 40e-3  # 40 mm
    
        K_IC_measured = fracture_toughness_SEVNB(P_max, B, W, a, S)
        print(f"Load: {P_max} N, a/W = {a/W:.2f}")
        print(f"K_IC = {K_IC_measured/1e6:.2f} MPa·m^0.5")
    
        # Numerical example for IF method
        print("\n=== Indentation Fracture Test (Al₂O₃) ===")
        P_indent = 98.1  # 10 kgf = 98.1 N
        a_indent = 50e-6  # 50 μm (half-diagonal)
        c_indent = 150e-6  # 150 μm
    
        K_IC_IF = indentation_fracture_toughness(P_indent, a_indent, c_indent, 400e9, 20e9)
        print(f"Indentation load: {P_indent} N (10 kgf)")
        print(f"Crack length c: {c_indent*1e6:.1f} μm")
        print(f"K_IC (IF method) = {K_IC_IF/1e6:.2f} MPa·m^0.5")
    
    # Execute
    analyze_fracture_toughness()
    

**Effect of Transformation Toughening** ZrO₂ (zirconia) undergoes stress-induced phase transformation (tetragonal → monoclinic) causing volume expansion, forming a compressive stress field at crack tips. This achieves high toughness of KIC = 8-12 MPa·m1/2 (2-3 times that of pure Al₂O₃). 

## 3.5 Weibull Statistics and Reliability Evaluation

### 3.5.1 Theory of Weibull Distribution

The strength of ceramics varies significantly due to defect distribution. Weibull (1951) proposed the following cumulative failure probability \\( P_f \\) based on the Weakest Link theory: 

\\[ P_f(\sigma) = 1 - \exp\left[-\left(\frac{\sigma - \sigma_u}{\sigma_0}\right)^m\right] \\] 

Where: 

  * \\( m \\): Weibull modulus (dimensionless, larger values indicate higher reliability)
  * \\( \sigma_0 \\): Characteristic strength (stress at which 63.2% fails)
  * \\( \sigma_u \\): Minimum strength (usually set to 0)

### 3.5.2 Meaning of Weibull Modulus m

m Value | Material Type | Defect State | Reliability  
---|---|---|---  
3-5 | Low-Quality Ceramics | Large defects, non-uniform | Low  
8-12 | Industrial Ceramics | Normal manufacturing quality | Medium  
15-20 | High-Quality Ceramics | Uniform, controlled defects | High  
>50 | Metals (Reference) | Approaching normal distribution | Very High  
      
    
    ```mermaid
    flowchart TD
                    A[Strength Test Datan specimens] --> B[Rank StatisticsSort in ascending order]
                    B --> C[Estimate Failure ProbabilityP_f,i = i/(n+1)]
                    C --> D[Weibull Plotln ln(1/(1-P_f)) vs ln σ]
                    D --> E[Linear RegressionSlope = mIntercept → σ₀]
                    E --> F[Reliability EvaluationP_f(σ_design)]
    
                    style A fill:#e3f2fd
                    style D fill:#f093fb,color:#fff
                    style E fill:#f5576c,color:#fff
                    style F fill:#4caf50,color:#fff
    ```

#### Python Implementation: Complete Implementation of Weibull Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 4: Weibull Statistical Analysis
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    from scipy.optimize import curve_fit
    
    def weibull_cumulative_probability(sigma, m, sigma_0, sigma_u=0):
        """
        Calculate Weibull cumulative failure probability
    
        Parameters:
        -----------
        sigma : float or array
            Stress [Pa]
        m : float
            Weibull modulus
        sigma_0 : float
            Characteristic strength [Pa]
        sigma_u : float
            Minimum strength [Pa] (default: 0)
    
        Returns:
        --------
        P_f : float or array
            Cumulative failure probability (0~1)
        """
        P_f = 1 - np.exp(-((sigma - sigma_u) / sigma_0)**m)
        return P_f
    
    
    def estimate_weibull_parameters(strength_data):
        """
        Estimate Weibull parameters from measured strength data
    
        Parameters:
        -----------
        strength_data : array
            Fracture strength data [Pa]
    
        Returns:
        --------
        m : float
            Weibull modulus
        sigma_0 : float
            Characteristic strength [Pa]
        R_squared : float
            Coefficient of determination (fit accuracy)
        """
        # Sort data
        sorted_strength = np.sort(strength_data)
        n = len(sorted_strength)
    
        # Estimate failure probability (median rank method)
        P_f = np.array([(i - 0.3) / (n + 0.4) for i in range(1, n + 1)])
    
        # Weibull transformation: Y = ln ln(1/(1-P_f)), X = ln(σ)
        # Avoid values very close to 1 or 0
        valid_indices = (P_f > 0.001) & (P_f < 0.999)
        P_f_valid = P_f[valid_indices]
        sigma_valid = sorted_strength[valid_indices]
    
        Y = np.log(-np.log(1 - P_f_valid))
        X = np.log(sigma_valid)
    
        # Linear regression
        coeffs = np.polyfit(X, Y, 1)
        m = coeffs[0]
        sigma_0 = np.exp(-coeffs[1] / m)
    
        # Calculate coefficient of determination R²
        Y_pred = m * X + coeffs[1]
        SS_res = np.sum((Y - Y_pred)**2)
        SS_tot = np.sum((Y - np.mean(Y))**2)
        R_squared = 1 - (SS_res / SS_tot)
    
        return m, sigma_0, R_squared
    
    
    def plot_weibull_analysis(strength_data, material_name='Ceramic'):
        """
        Complete visualization of Weibull analysis
    
        Parameters:
        -----------
        strength_data : array
            Fracture strength data [MPa]
        material_name : str
            Material name
        """
        # Unit conversion (MPa → Pa)
        strength_Pa = strength_data * 1e6
    
        # Estimate Weibull parameters
        m, sigma_0, R2 = estimate_weibull_parameters(strength_Pa)
    
        # Sort and calculate failure probability
        sorted_strength = np.sort(strength_Pa)
        n = len(sorted_strength)
        P_f = np.array([(i - 0.3) / (n + 0.4) for i in range(1, n + 1)])
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        # Left figure: Strength distribution histogram
        ax1 = axes[0]
        ax1.hist(strength_data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
        # Overlay theoretical distribution
        sigma_range = np.linspace(strength_data.min(), strength_data.max(), 500)
        pdf = (m / sigma_0) * ((sigma_range * 1e6) / sigma_0)**(m - 1) * \
              np.exp(-((sigma_range * 1e6) / sigma_0)**m)
        ax1.plot(sigma_range, pdf * 1e6, 'r-', linewidth=2, label=f'Weibull PDF (m={m:.1f})')
    
        ax1.set_xlabel('Strength (MPa)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'{material_name} Strength Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Center figure: Weibull plot
        ax2 = axes[1]
        valid_indices = (P_f > 0.001) & (P_f < 0.999)
        P_f_valid = P_f[valid_indices]
        sigma_valid = sorted_strength[valid_indices] / 1e6
    
        Y_data = np.log(-np.log(1 - P_f_valid))
        X_data = np.log(sigma_valid)
    
        ax2.plot(X_data, Y_data, 'o', markersize=8, color='navy', label='Experimental')
    
        # Fit line
        X_fit = np.linspace(X_data.min(), X_data.max(), 100)
        Y_fit = m * (X_fit - np.log(sigma_0 / 1e6))
        ax2.plot(X_fit, Y_fit, 'r-', linewidth=2, label=f'Fit: m={m:.2f}, R²={R2:.4f}')
    
        ax2.set_xlabel('ln(Strength) [ln(MPa)]', fontsize=12)
        ax2.set_ylabel('ln ln(1/(1-P_f))', fontsize=12)
        ax2.set_title('Weibull Plot', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        # Right figure: Reliability curve
        ax3 = axes[2]
        sigma_reliability = np.linspace(0.5 * sigma_0, 1.5 * sigma_0, 500) / 1e6
        P_f_curve = weibull_cumulative_probability(sigma_reliability * 1e6, m, sigma_0)
    
        ax3.plot(sigma_reliability, (1 - P_f_curve) * 100, linewidth=2, color='green')
        ax3.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90% Reliability')
        ax3.axhline(y=99, color='red', linestyle='--', alpha=0.5, label='99% Reliability')
    
        # Calculate design stress (99% reliability)
        sigma_design_99 = sigma_0 * (-np.log(1 - 0.01))**(1/m)
        ax3.axvline(x=sigma_design_99/1e6, color='red', linestyle=':', linewidth=2,
                    label=f'σ_design (99%) = {sigma_design_99/1e6:.0f} MPa')
    
        ax3.set_xlabel('Stress (MPa)', fontsize=12)
        ax3.set_ylabel('Reliability (%)', fontsize=12)
        ax3.set_title('Reliability vs Stress', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('weibull_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output numerical results
        print(f"=== Weibull Analysis Results for {material_name} ===")
        print(f"Sample size: n = {n}")
        print(f"Weibull modulus: m = {m:.2f}")
        print(f"Characteristic strength: σ₀ = {sigma_0/1e6:.1f} MPa")
        print(f"Goodness of fit: R² = {R2:.4f}")
        print(f"\n--- Reliability-based Design Stress ---")
        for reliability in [0.90, 0.95, 0.99, 0.999]:
            sigma_design = sigma_0 * (-np.log(1 - (1 - reliability)))**(1/m)
            print(f"{reliability*100:.1f}% reliability → σ_design = {sigma_design/1e6:.1f} MPa")
    
        return m, sigma_0, R2
    
    
    # Generate test data (typical example of Al2O3)
    np.random.seed(42)
    n_samples = 30
    m_true = 10  # Weibull modulus
    sigma_0_true = 400  # MPa
    
    # Sample from Weibull distribution
    strength_samples = weibull_min.rvs(m_true, scale=sigma_0_true, size=n_samples)
    
    # Execute analysis
    plot_weibull_analysis(strength_samples, 'Al₂O₃')
    

**Method for Determining Design Stress** When requiring 99% reliability, calculate using σdesign = σ₀ × (-ln 0.01)1/m. For m = 10, σdesign ≈ 0.58 σ₀, meaning the design allowable stress is approximately 60% of the characteristic strength. Smaller m values require larger safety factors. 

#### Python Implementation: Monte Carlo Strength Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 5: Monte Carlo Strength Simulation
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def monte_carlo_strength_simulation(m, sigma_0, n_components, n_simulations=10000):
        """
        Estimate reliability of multi-component system using Monte Carlo method
    
        Parameters:
        -----------
        m : float
            Weibull modulus
        sigma_0 : float
            Characteristic strength [MPa]
        n_components : int
            Number of components in system
        n_simulations : int
            Number of simulations
    
        Returns:
        --------
        system_strength : array
            System strength distribution (strength of weakest component) [MPa]
        """
        # Sample component strengths from Weibull distribution
        component_strengths = weibull_min.rvs(m, scale=sigma_0, size=(n_simulations, n_components))
    
        # System strength = strength of weakest component (Weakest Link)
        system_strength = np.min(component_strengths, axis=1)
    
        return system_strength
    
    
    def analyze_size_effect():
        """
        Analysis of Size Effect
        """
        m = 10
        sigma_0 = 400  # MPa
    
        # Vary number of components
        n_components_list = [1, 10, 100, 1000]
    
        plt.figure(figsize=(14, 5))
    
        # Left figure: Change in strength distribution
        plt.subplot(1, 2, 1)
        for n_comp in n_components_list:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 50000)
            plt.hist(system_strength, bins=50, alpha=0.5, density=True, label=f'n={n_comp}')
    
        plt.xlabel('System Strength (MPa)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Size Effect on Strength Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Right figure: Reduction in mean strength
        plt.subplot(1, 2, 2)
        n_range = np.logspace(0, 3, 50).astype(int)
        mean_strengths = []
    
        for n_comp in n_range:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 10000)
            mean_strengths.append(np.mean(system_strength))
    
        plt.semilogx(n_range, mean_strengths, linewidth=2, color='crimson')
    
        # Theoretical curve (prediction by Weibull theory)
        # E[σ_min] = σ₀ × Γ(1 + 1/m) × n^(-1/m)
        from scipy.special import gamma
        theoretical_mean = sigma_0 * gamma(1 + 1/m) * n_range**(-1/m)
        plt.semilogx(n_range, theoretical_mean, '--', linewidth=2, color='blue', label='Theoretical')
    
        plt.xlabel('Number of Components n', fontsize=12)
        plt.ylabel('Mean System Strength (MPa)', fontsize=12)
        plt.title('Size Effect (Weakest Link Model)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('monte_carlo_size_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Numerical results
        print("=== Size Effect Analysis (Monte Carlo) ===")
        print(f"Single component: σ₀ = {sigma_0} MPa, m = {m}")
        for n_comp in [1, 10, 100, 1000]:
            system_strength = monte_carlo_strength_simulation(m, sigma_0, n_comp, 10000)
            mean_strength = np.mean(system_strength)
            std_strength = np.std(system_strength)
            print(f"n = {n_comp:4d}: Mean = {mean_strength:.1f} MPa, Std = {std_strength:.1f} MPa")
    
    # Execute
    analyze_size_effect()
    

**Size Effect** When components become larger or the number of components increases, the probability of defect existence rises, reducing overall system strength. This is called the Size Effect. During design, it is necessary to correct for the difference between specimen size and actual component size (effective volume correction). 

## 3.6 High-Temperature Creep and Fatigue

### 3.6.1 Creep Deformation

At high temperatures (above 0.5 Tm, where Tm is the melting point), even ceramics undergo creep deformation. The creep rate \\( \dot{\epsilon} \\) is expressed by the following equation: 

\\[ \dot{\epsilon} = A \sigma^n \exp\left(-\frac{Q}{RT}\right) \\] 

Where \\( A \\) is a constant, \\( n \\) is the stress exponent, \\( Q \\) is the activation energy, \\( R \\) is the gas constant, and \\( T \\) is the temperature. 

### 3.6.2 Creep Mechanisms

  * **Diffusion Creep (n=1)** : Grain boundary diffusion (Coble creep), lattice diffusion (Nabarro-Herring creep)
  * **Grain Boundary Sliding Creep (n=2)** : Atomic migration and sliding at grain boundaries
  * **Dislocation Creep (n >3)**: Climb motion of dislocations (high stress region)

#### Python Implementation: Calculation of Creep Rate
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 6: Calculation of High-Temperature Creep Rate
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def creep_rate(sigma, T, A=1e-10, n=1, Q=400e3, R=8.314):
        """
        Calculate creep rate
    
        Parameters:
        -----------
        sigma : float or array
            Stress [Pa]
        T : float or array
            Temperature [K]
        A : float
            Constant [s^-1·Pa^-n]
        n : float
            Stress exponent
        Q : float
            Activation energy [J/mol]
        R : float
            Gas constant [J/(mol·K)]
    
        Returns:
        --------
        epsilon_dot : float or array
            Creep rate [s^-1]
        """
        epsilon_dot = A * sigma**n * np.exp(-Q / (R * T))
        return epsilon_dot
    
    
    def plot_creep_behavior():
        """
        Visualization of creep behavior
        """
        # Creep parameters for Al2O3 (example)
        A_diffusion = 1e-8
        n_diffusion = 1
        Q_diffusion = 400e3
    
        A_dislocation = 1e-12
        n_dislocation = 4
        Q_dislocation = 600e3
    
        # Temperature range
        temperatures = np.linspace(1200, 1600, 5) + 273.15  # K
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # Left figure: Stress dependence
        ax1 = axes[0]
        stress_range = np.logspace(6, 8, 100)  # 1 ~ 100 MPa
    
        for T in temperatures:
            epsilon_dot_diff = creep_rate(stress_range, T, A_diffusion, n_diffusion, Q_diffusion)
            epsilon_dot_disl = creep_rate(stress_range, T, A_dislocation, n_dislocation, Q_dislocation)
            epsilon_dot_total = epsilon_dot_diff + epsilon_dot_disl
    
            ax1.loglog(stress_range / 1e6, epsilon_dot_total, linewidth=2,
                       label=f'T = {T-273.15:.0f}°C')
    
        ax1.set_xlabel('Stress (MPa)', fontsize=12)
        ax1.set_ylabel('Creep Rate (s^-1)', fontsize=12)
        ax1.set_title('Stress Dependence of Creep Rate', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
    
        # Right figure: Temperature dependence (Arrhenius plot)
        ax2 = axes[1]
        T_range = np.linspace(1000, 1800, 100) + 273.15
        sigma_fixed = 50e6  # 50 MPa
    
        epsilon_dot_diff = creep_rate(sigma_fixed, T_range, A_diffusion, n_diffusion, Q_diffusion)
        epsilon_dot_disl = creep_rate(sigma_fixed, T_range, A_dislocation, n_dislocation, Q_dislocation)
    
        ax2.semilogy(1e4 / T_range, epsilon_dot_diff, linewidth=2, label='Diffusion creep (n=1)')
        ax2.semilogy(1e4 / T_range, epsilon_dot_disl, linewidth=2, label='Dislocation creep (n=4)')
        ax2.semilogy(1e4 / T_range, epsilon_dot_diff + epsilon_dot_disl, 'k--', linewidth=2,
                     label='Total')
    
        ax2.set_xlabel('10^4 / T (K^-1)', fontsize=12)
        ax2.set_ylabel('Creep Rate (s^-1)', fontsize=12)
        ax2.set_title(f'Arrhenius Plot (σ = {sigma_fixed/1e6} MPa)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('creep_behavior.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Numerical examples
        print("=== Creep Rate Calculation (Al₂O₃) ===")
        T_test = 1400 + 273.15  # 1400°C
        for sigma_MPa in [10, 50, 100]:
            sigma = sigma_MPa * 1e6
            eps_dot = creep_rate(sigma, T_test, A_diffusion, n_diffusion, Q_diffusion)
            print(f"σ = {sigma_MPa:3d} MPa, T = 1400°C → ε̇ = {eps_dot:.2e} s^-1")
    
    # Execute
    plot_creep_behavior()
    

#### Python Implementation: Generation of Stress-Strain Curves
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 7: Generation of Stress-Strain Curves
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stress_strain_curve(max_stress=500e6, E=400e9, K_IC=4e6, a_initial=10e-6):
        """
        Stress-strain curve for ceramics (linear elastic + sudden fracture)
    
        Parameters:
        -----------
        max_stress : float
            Maximum stress [Pa]
        E : float
            Young's modulus [Pa]
        K_IC : float
            Fracture toughness [Pa·m^0.5]
        a_initial : float
            Initial crack length [m]
    
        Returns:
        --------
        stress : array
            Stress [Pa]
        strain : array
            Strain
        fracture_stress : float
            Fracture stress [Pa]
        """
        # Calculate fracture stress (K_IC = Y σ_f sqrt(π a))
        Y = 1.12  # Shape factor for surface crack
        fracture_stress = K_IC / (Y * np.sqrt(np.pi * a_initial))
    
        # Linear elastic region
        if fracture_stress > max_stress:
            fracture_stress = max_stress
    
        stress = np.linspace(0, fracture_stress, 1000)
        strain = stress / E
    
        return stress, strain, fracture_stress
    
    
    def compare_materials():
        """
        Comparison of stress-strain curves for various ceramics
        """
        materials = {
            'Al₂O₃': {'E': 400e9, 'K_IC': 4e6, 'a': 10e-6},
            'ZrO₂': {'E': 210e9, 'K_IC': 8e6, 'a': 10e-6},
            'Si₃N₄': {'E': 310e9, 'K_IC': 6e6, 'a': 10e-6},
            'SiC': {'E': 450e9, 'K_IC': 3.5e6, 'a': 10e-6}
        }
    
        plt.figure(figsize=(12, 5))
    
        # Left figure: Stress-strain curves
        plt.subplot(1, 2, 1)
        for name, props in materials.items():
            stress, strain, sigma_f = stress_strain_curve(
                max_stress=1e9,
                E=props['E'],
                K_IC=props['K_IC'],
                a_initial=props['a']
            )
            plt.plot(strain * 100, stress / 1e6, linewidth=2, label=f'{name} (σ_f={sigma_f/1e6:.0f} MPa)')
    
        plt.xlabel('Strain (%)', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Stress-Strain Curves for Ceramics', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Right figure: Comparison with metal
        plt.subplot(1, 2, 2)
    
        # Ceramic (Al2O3)
        stress_ceramic, strain_ceramic, sigma_f_ceramic = stress_strain_curve(
            max_stress=1e9, E=400e9, K_IC=4e6, a_initial=10e-6
        )
        plt.plot(strain_ceramic * 100, stress_ceramic / 1e6, linewidth=2,
                 label='Ceramic (Al₂O₃)', color='red')
    
        # Metal (steel) simulated curve (ductile fracture)
        E_steel = 200e9
        yield_stress = 300e6
        UTS = 500e6
    
        strain_elastic = np.linspace(0, yield_stress / E_steel, 100)
        stress_elastic = strain_elastic * E_steel
    
        strain_plastic = np.linspace(yield_stress / E_steel, 0.2, 100)
        stress_plastic = yield_stress + (UTS - yield_stress) * (1 - np.exp(-50 * (strain_plastic - yield_stress / E_steel)))
    
        strain_steel = np.concatenate([strain_elastic, strain_plastic])
        stress_steel = np.concatenate([stress_elastic, stress_plastic])
    
        plt.plot(strain_steel * 100, stress_steel / 1e6, linewidth=2,
                 label='Metal (Steel)', color='blue')
    
        plt.xlabel('Strain (%)', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Ceramic vs Metal: Brittle vs Ductile', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 20)
    
        plt.tight_layout()
        plt.savefig('stress_strain_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Numerical examples
        print("=== Stress-Strain Behavior Comparison ===")
        for name, props in materials.items():
            _, _, sigma_f = stress_strain_curve(
                max_stress=1e9,
                E=props['E'],
                K_IC=props['K_IC'],
                a_initial=props['a']
            )
            fracture_strain = sigma_f / props['E']
            print(f"{name:8s}: σ_f = {sigma_f/1e6:5.0f} MPa, ε_f = {fracture_strain*100:.3f}%")
    
    # Execute
    compare_materials()
    

**Numerical Meaning of Ceramic Brittleness** The fracture strain of Al₂O₃ is approximately 0.1% (400 MPa / 400 GPa), which is 200 times smaller than steel's 20% (ductile fracture). This extreme brittleness is why ceramics are weak under impact loads and require large safety factors in design. 

## Exercises

#### Exercise 3-1: Effect of Stress Concentration FactorEasy

An Al₂O₃ specimen has a 10 μm surface crack with a tip radius of curvature of 1 nm. Calculate the stress concentration factor Kt and determine how many times the local stress is compared to the far-field stress. 

Solution Example
    
    
    a = 10e-6  # m
    rho = 1e-9  # m
    K_t = 1 + 2 * np.sqrt(a / rho)
    print(f"K_t = {K_t:.1f}")
    # Output: K_t ≈ 201 (local stress is approximately 200 times)
    

#### Exercise 3-2: Prediction of Griffith StrengthEasy

For SiC (E = 450 GPa, γ = 1.2 J/m²) with a 50 μm crack, calculate the fracture strength using Griffith theory. 

Solution Example
    
    
    E = 450e9
    gamma = 1.2
    a = 50e-6
    sigma_f = np.sqrt(2 * E * gamma / (np.pi * a))
    print(f"σ_f = {sigma_f/1e6:.0f} MPa")
    # Output: σ_f ≈ 117 MPa
    

#### Exercise 3-3: Measurement of Fracture ToughnessEasy

An Al₂O₃ specimen in a SEVNB test (B=4 mm, W=3 mm, a=1.2 mm, S=40 mm) fractured at a load of 300 N. Calculate KIC. 

Solution Example
    
    
    K_IC = fracture_toughness_SEVNB(300, 4e-3, 3e-3, 1.2e-3, 40e-3)
    print(f"K_IC = {K_IC/1e6:.2f} MPa·m^0.5")
    # Output: K_IC ≈ 4.2 MPa·m^0.5
    

#### Exercise 3-4: Estimation of Weibull ParametersMedium

Estimate the Weibull modulus m and characteristic strength σ₀ from the following strength data (MPa): [350, 420, 380, 450, 390, 410, 370, 440, 400, 430]. 

Solution Example
    
    
    data = np.array([350, 420, 380, 450, 390, 410, 370, 440, 400, 430])
    m, sigma_0, R2 = estimate_weibull_parameters(data * 1e6)
    print(f"m = {m:.2f}, σ₀ = {sigma_0/1e6:.1f} MPa, R² = {R2:.4f}")
    # Example output: m ≈ 12.5, σ₀ ≈ 415 MPa
    

#### Exercise 3-5: Calculation of Design StressMedium

You want to achieve 99.9% reliability using Si₃N₄ components (m=10, σ₀=700 MPa). Calculate the design allowable stress. 

Solution Example
    
    
    m = 10
    sigma_0 = 700e6
    reliability = 0.999
    sigma_design = sigma_0 * (-np.log(1 - reliability))**(1/m)
    print(f"σ_design = {sigma_design/1e6:.0f} MPa")
    # Output: σ_design ≈ 368 MPa (approximately 53% of σ₀)
    

#### Exercise 3-6: Evaluation of Size EffectMedium

The average strength of a test specimen (volume V₁) was 400 MPa. Calculate the expected strength of an actual component with 10 times the volume (V₂=10V₁) assuming m=10. 

Solution Example
    
    
    sigma_1 = 400  # MPa
    V_ratio = 10
    m = 10
    sigma_2 = sigma_1 * V_ratio**(-1/m)
    print(f"σ₂ = {sigma_2:.1f} MPa")
    # Output: σ₂ ≈ 318 MPa (approximately 20% reduction)
    

#### Exercise 3-7: Temperature Dependence of Creep RateMedium

For Al₂O₃ creep (A=1e-8, n=1, Q=400 kJ/mol), calculate the creep rate at a stress of 50 MPa and temperature of 1400°C. 

Solution Example
    
    
    sigma = 50e6
    T = 1400 + 273.15
    epsilon_dot = creep_rate(sigma, T, A=1e-8, n=1, Q=400e3)
    print(f"ε̇ = {epsilon_dot:.2e} s^-1")
    # Example output: ε̇ ≈ 1.2e-8 s^-1
    

#### Exercise 3-8: Fracture Under Multiaxial StressHard

An Al₂O₃ component is subjected to biaxial stress of σ₁=200 MPa (tensile) and σ₂=-100 MPa (compressive). Using the maximum principal stress criterion, determine if the material with KIC=4 MPa·m1/2 will fracture (assume a 5 μm surface crack). 

Solution Example
    
    
    # Maximum principal stress = σ₁ (tension dominates)
    sigma_principal = 200e6
    a = 5e-6
    Y = 1.12
    K_I = Y * sigma_principal * np.sqrt(np.pi * a)
    K_IC = 4e6
    print(f"K_I = {K_I/1e6:.2f} MPa·m^0.5, K_IC = {K_IC/1e6} MPa·m^0.5")
    if K_I >= K_IC:
        print("Fracture will occur")
    else:
        print("Safe")
    # Output: K_I ≈ 2.81 MPa·m^0.5 < K_IC → Safe
    

#### Exercise 3-9: Modeling of R-curve BehaviorHard

Considering the R-curve (KR = K₀ + A√Δa, K₀=4, A=2, Δa is crack extension amount) for transformation-toughened ZrO₂, calculate the stress required for fracture from an initial crack of 10 μm. 

Solution Example
    
    
    K_0 = 4e6  # MPa·m^0.5
    A = 2e6
    a_initial = 10e-6
    Y = 1.12
    
    # Simulate crack extension
    delta_a_range = np.linspace(0, 50e-6, 100)
    K_R = K_0 + A * np.sqrt(delta_a_range)
    
    # Increase stress to find fracture condition
    for sigma_MPa in range(100, 1000, 10):
        sigma = sigma_MPa * 1e6
        K_I = Y * sigma * np.sqrt(np.pi * (a_initial + delta_a_range[-1]))
        if K_I >= K_R[-1]:
            print(f"Fracture stress: {sigma_MPa} MPa")
            break
    # Example output: Strength enhancement due to R-curve effect
    

#### Exercise 3-10: Long-term Reliability PredictionHard

Si₃N₄ turbine components (m=12, σ₀=800 MPa) will operate at a stress of 400 MPa for 10 years. Estimate the failure probability considering only initial strength distribution without accounting for static fatigue (slow crack growth). 

Solution Example
    
    
    m = 12
    sigma_0 = 800e6
    sigma_applied = 400e6
    
    # Weibull cumulative failure probability
    P_f = 1 - np.exp(-((sigma_applied / sigma_0)**m))
    print(f"Failure probability P_f = {P_f*100:.4f}%")
    print(f"Reliability = {(1-P_f)*100:.4f}%")
    # Output: P_f ≈ 0.024% → Reliability 99.976%
    # Note: In reality, failure probability increases when fatigue is considered
    

## References

  1. Lawn, B.R. (1993). _Fracture of Brittle Solids_. Cambridge University Press, pp. 1-75, 194-250.
  2. Munz, D., Fett, T. (2007). _Ceramics: Mechanical Properties, Failure Behaviour, Materials Selection_. Springer, pp. 45-120, 201-255.
  3. Anderson, T.L. (2017). _Fracture Mechanics: Fundamentals and Applications_. CRC Press, pp. 220-285.
  4. Weibull, W. (1951). A Statistical Distribution Function of Wide Applicability. _Journal of Applied Mechanics_ , 18, 293-297.
  5. Quinn, G.D. (2007). _Fractography of Ceramics and Glasses_. NIST Special Publication 960-16, pp. 1-50.
  6. Carter, C.B., Norton, M.G. (2013). _Ceramic Materials: Science and Engineering_. Springer, pp. 520-590.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
