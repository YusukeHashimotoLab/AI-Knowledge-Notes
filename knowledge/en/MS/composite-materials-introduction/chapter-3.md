---
title: "Chapter 3: Particle and Laminate Composites"
chapter_title: "Chapter 3: Particle and Laminate Composites"
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Composite Materials](<../../MS/composite-materials-introduction/index.html>)›Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/MS/composite-materials-introduction/chapter-3.html>) | Last sync: 2025-11-16

### Composite Materials Introduction

  * [Table of Contents](<index.html>)
  * [Chapter 1: Fundamentals of Composite Materials](<chapter-1.html>)
  * [Chapter 2: Fiber-Reinforced Composites](<chapter-2.html>)
  * [Chapter 3: Particle and Laminate Composites](<chapter-3.html>)
  * [Chapter 4: Evaluation of Composite Materials](<chapter-4.html>)
  * [Chapter 5: Python Practice](<chapter-5.html>)

#### Materials Science Series

  * [Polymer Materials Introduction](<../polymer-materials-introduction/index.html>)
  * [Thin Film and Nanomaterials Introduction](<../thin-film-nano-introduction/index.html>)
  * [Composite Materials Introduction](<index.html>)

# Chapter 3: Particle and Laminate Composites

This chapter covers Particle and Laminate Composites. You will learn essential concepts and techniques.

### Learning Objectives

  * **Foundation Level:** Understand types and strengthening mechanisms of particle-reinforced composites and perform basic property predictions
  * **Application Level:** Apply Orowan mechanism to optimize particle size and volume fraction
  * **Advanced Level:** Comprehensively evaluate MMC/CMC design parameters and select materials for specific applications

## 3.1 Fundamentals of Particle-Reinforced Composites

### 3.1.1 Classification of Particle Reinforcement

Particle-reinforced composites are materials with particle-shaped reinforcements dispersed in a matrix. They are classified according to strengthening mechanisms: 

Classification | Particle Size | Strengthening Mechanism | Representative Examples  
---|---|---|---  
Dispersion Strengthening | 10-100 nm | Dislocation bypass (Orowan) | ODS alloys, precipitation-strengthened steels  
Particle Strengthening | 1-100 μm | Load transfer, thermal expansion mismatch | SiC/Al, WC/Co  
Filler | 1-100 μm | Cost reduction, dimensional stability | Calcium carbonate/resin  
      
    
    ```mermaid
    flowchart TD
                                A[Particle-Reinforced Composites] --> B[Metal Matrix MMC]
                                A --> C[Ceramic Matrix CMC]
                                A --> D[Polymer Matrix PMC]
    
                                B --> E[SiC/AlAutomotive pistons]
                                B --> F[Al2O3/AlSliding components]
                                B --> G[B4C/AlArmor materials]
    
                                C --> H[SiC/SiCHeat-resistant components]
                                C --> I[Al2O3/ZrO2Cutting tools]
    
                                D --> J[Carbon black/rubberTires]
                                D --> K[Glass beads/resinElectronic substrates]
    
                                style A fill:#e1f5ff
                                style E fill:#ffe1e1
                                style F fill:#ffe1e1
                                style G fill:#ffe1e1
                                style H fill:#c8e6c9
                                style I fill:#c8e6c9
                                style J fill:#fff9c4
                                style K fill:#fff9c4
    ```

### 3.1.2 MMC (Metal Matrix Composites)

Metal matrix composites are characterized by light weight, high strength, and high heat resistance. 

Matrix | Reinforcement | Manufacturing Method | Application  
---|---|---|---  
Al alloy | SiC particles (15-20 vol%) | Powder metallurgy, melt stirring | Automotive engine components  
Al alloy | Al₂O₃ particles | Spray forming | Brake discs  
Ti alloy | TiB fibers | Reactive synthesis | Aircraft structural materials  
Cu alloy | Graphite particles | Powder metallurgy | Electrical contacts, bearings  
  
### 3.1.3 CMC (Ceramic Matrix Composites)

Ceramic matrix composites are materials where fibers or particles are incorporated into a ceramic matrix to improve brittleness. 

Material System | Operating Temperature | Characteristics | Application  
---|---|---|---  
SiC/SiC | ~1400°C | High toughness, oxidation resistance | Jet engine nozzles  
C/SiC | ~1600°C | Lightweight, high heat resistance | Aircraft brakes  
Al₂O₃/Al₂O₃ | ~1200°C | High hardness, wear resistance | Cutting tools  
  
## 3.2 Mechanical Models for Particle Strengthening

### 3.2.1 Elastic Modulus Prediction

For spherical particle reinforcement, the Hashin-Shtrikman upper and lower bound models are commonly used. For isotropic materials: 

#### Bulk Modulus

$$K_c = K_m + \frac{V_p}{(K_p - K_m)^{-1} + 3(1-V_p)/(3K_m + 4G_m)}$$ 

#### Shear Modulus

$$G_c = G_m + \frac{V_p}{(G_p - G_m)^{-1} + 6(K_m + 2G_m)(1-V_p)/(5G_m(3K_m + 4G_m))}$$ 

Young's modulus and Poisson's ratio are calculated from:

$$E_c = \frac{9K_c G_c}{3K_c + G_c}, \quad \nu_c = \frac{3K_c - 2G_c}{2(3K_c + G_c)}$$ 

#### Example 3.1: Elastic Modulus Calculation of SiC/Al Composite
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p):
        """
        Composite material elastic modulus calculation using Hashin-Shtrikman model
    
        Parameters:
        -----------
        K_m, G_m : float
            Bulk and shear modulus of matrix [GPa]
        K_p, G_p : float
            Bulk and shear modulus of particle [GPa]
        V_p : float or array
            Particle volume fraction
    
        Returns:
        --------
        E_c, nu_c : float or array
            Young's modulus and Poisson's ratio of composite
        """
        # Bulk modulus
        K_c = K_m + V_p / (1/(K_p - K_m) + 3*(1 - V_p)/(3*K_m + 4*G_m))
    
        # Shear modulus
        G_c = G_m + V_p / (1/(G_p - G_m) + 6*(K_m + 2*G_m)*(1 - V_p)/(5*G_m*(3*K_m + 4*G_m)))
    
        # Young's modulus and Poisson's ratio
        E_c = 9 * K_c * G_c / (3 * K_c + G_c)
        nu_c = (3 * K_c - 2 * G_c) / (2 * (3 * K_c + G_c))
    
        return E_c, nu_c
    
    def E_nu_to_K_G(E, nu):
        """Convert Young's modulus and Poisson's ratio to bulk and shear modulus"""
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        return K, G
    
    # Al alloy matrix properties
    E_m = 70.0   # GPa
    nu_m = 0.33
    K_m, G_m = E_nu_to_K_G(E_m, nu_m)
    
    # SiC particle properties
    E_p = 450.0  # GPa
    nu_p = 0.17
    K_p, G_p = E_nu_to_K_G(E_p, nu_p)
    
    # Volume fraction range
    V_p_range = np.linspace(0, 0.5, 100)
    
    # Elastic modulus calculation
    E_c, nu_c = hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p_range)
    
    # Comparison with rule of mixtures (upper and lower bounds)
    E_voigt = E_m * (1 - V_p_range) + E_p * V_p_range  # Upper bound
    E_reuss = 1 / ((1 - V_p_range)/E_m + V_p_range/E_p)  # Lower bound
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Young's modulus
    ax1.plot(V_p_range, E_c, 'b-', linewidth=2, label='Hashin-Shtrikman')
    ax1.plot(V_p_range, E_voigt, 'r--', linewidth=1.5, label='Voigt (upper bound)')
    ax1.plot(V_p_range, E_reuss, 'g--', linewidth=1.5, label='Reuss (lower bound)')
    ax1.fill_between(V_p_range, E_reuss, E_voigt, alpha=0.2, color='gray',
                      label='Rule of mixtures range')
    ax1.set_xlabel('SiC Volume Fraction')
    ax1.set_ylabel('Young\'s Modulus [GPa]')
    ax1.set_title('Young\'s Modulus of SiC/Al Composite')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Poisson's ratio
    ax2.plot(V_p_range, nu_c, 'b-', linewidth=2, label='Hashin-Shtrikman')
    ax2.axhline(y=nu_m, color='r', linestyle='--', label=f'Al matrix ({nu_m:.2f})')
    ax2.axhline(y=nu_p, color='g', linestyle='--', label=f'SiC particle ({nu_p:.2f})')
    ax2.set_xlabel('SiC Volume Fraction')
    ax2.set_ylabel('Poisson\'s Ratio')
    ax2.set_title('Poisson\'s Ratio of SiC/Al Composite')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('particle_composite_modulus.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Values at practical particle fractions
    V_p_practical = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    E_c_practical, nu_c_practical = hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p_practical)
    
    print("Elastic Properties of SiC/Al Composite:")
    print("="*60)
    print(f"{'V_p':>6} {'E_c [GPa]':>12} {'Increase[%]':>12} {'Poisson\'s Ratio':>12}")
    print("-"*60)
    for vp, ec, nuc in zip(V_p_practical, E_c_practical, nu_c_practical):
        increase = (ec / E_m - 1) * 100
        print(f"{vp:6.2f} {ec:12.1f} {increase:12.1f} {nuc:12.3f}")

### 3.2.2 Strength Prediction

The strength of particle-reinforced composites is determined by the combined effects of the following factors: 

  * **Load transfer effect:** Particles bear the load
  * **Dislocation strengthening:** Increased dislocation density around particles
  * **Orowan mechanism:** Dislocations bypass particles
  * **Thermal expansion mismatch:** Residual stress from cooling

#### Example 3.2: Yield Strength Prediction of Particle-Reinforced Composites
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def particle_strengthening(sigma_m, V_p, d_p, b, G_m):
        """
        Yield strength prediction for particle-reinforced composites
    
        Parameters:
        -----------
        sigma_m : float
            Matrix yield strength [MPa]
        V_p : float
            Particle volume fraction
        d_p : float
            Particle diameter [μm]
        b : float
            Burgers vector [nm]
        G_m : float
            Matrix shear modulus [GPa]
    
        Returns:
        --------
        sigma_c : float
            Composite yield strength [MPa]
        """
        # Load transfer term (simplified model)
        sigma_load = sigma_m * (1 + 0.5 * V_p)
    
        # Orowan strengthening term
        # Estimation of interparticle spacing
        lambda_p = d_p * (np.sqrt(np.pi / (4 * V_p)) - 1)  # [μm]
    
        # Orowan stress [MPa]
        G_m_MPa = G_m * 1000  # GPa → MPa
        b_m = b * 1e-9  # nm → m
        lambda_p_m = lambda_p * 1e-6  # μm → m
    
        sigma_orowan = 0.4 * G_m_MPa * b_m / lambda_p_m / 1e6  # MPa
    
        # Total strength (simplified addition)
        sigma_c = sigma_load + sigma_orowan
    
        return sigma_c
    
    # Al alloy matrix
    sigma_m = 100  # MPa (annealed)
    b = 0.286      # nm (Al Burgers vector)
    G_m = 26       # GPa
    
    # Effect of SiC particle size
    d_p_range = np.logspace(-1, 1.5, 50)  # 0.1-30 μm
    V_p_values = [0.10, 0.15, 0.20, 0.25]
    
    plt.figure(figsize=(10, 6))
    
    for V_p in V_p_values:
        sigma_c = []
        for d_p in d_p_range:
            s_c = particle_strengthening(sigma_m, V_p, d_p, b, G_m)
            sigma_c.append(s_c)
    
        plt.plot(d_p_range, sigma_c, linewidth=2, label=f'V_p = {V_p:.2f}')
    
    plt.xscale('log')
    plt.xlabel('Particle Diameter [μm]')
    plt.ylabel('Composite Yield Strength [MPa]')
    plt.title('Relationship between Particle Size and Yield Strength (SiC/Al)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('particle_size_strengthening.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Examination of optimal particle size
    V_p_opt = 0.20
    d_p_test = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    
    print("Particle Size and Strengthening Effect:")
    print("="*60)
    print(f"{'Particle Diameter [μm]':>15} {'Yield Strength [MPa]':>18} {'Strengthening [%]':>15}")
    print("-"*60)
    
    for d_p in d_p_test:
        sigma_c = particle_strengthening(sigma_m, V_p_opt, d_p, b, G_m)
        strengthening = (sigma_c / sigma_m - 1) * 100
        print(f"{d_p:15.1f} {sigma_c:18.1f} {strengthening:15.1f}")

## 3.3 Orowan Mechanism

### 3.3.1 Dislocation-Particle Interactions

The Orowan mechanism is a strengthening mechanism that occurs when dislocations cannot cut through particles, but instead **bypass** between particles. 

$$\Delta\sigma_{\text{Orowan}} = \frac{0.4Gb}{\lambda}$$ 

where \\(G\\): shear modulus, \\(b\\): Burgers vector, \\(\lambda\\): interparticle spacing 

Interparticle spacing can be estimated from particle size and volume fraction:

$$\lambda \approx d_p \left(\sqrt{\frac{\pi}{4V_p}} - 1\right)$$ 
    
    
    ```mermaid
    flowchart TD
                                A[Dislocation Movement] --> B{Interaction with Particles}
                                B --> C[Particle Cutting PossibleWeak Interface]
                                B --> D[Particle Cutting ImpossibleHard Particles]
    
                                C --> E[Dislocation cuts particleSmall strengthening effect]
                                D --> F[Orowan Bypass]
    
                                F --> G[Dislocation loop remains around particle]
                                G --> H[Hinders subsequent dislocation movement]
                                H --> I[Strength increase]
    
                                style A fill:#e1f5ff
                                style F fill:#ffe1e1
                                style I fill:#c8e6c9
    ```

### 3.3.2 Design of Optimal Particle Size and Fraction

To maximize Orowan strengthening, the interparticle spacing must be minimized. However, the following trade-offs exist: 

  * **Small particle size → Large strengthening effect** (λ decreases) but particles tend to agglomerate
  * **Large volume fraction → Large strengthening effect** (λ decreases) but ductility decreases
  * **Optimization of particle size/fraction** is important

#### Example 3.3: Optimal Design of Orowan Strengthening
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def orowan_stress(d_p, V_p, G_m, b):
        """
        Orowan stress calculation
    
        Parameters:
        -----------
        d_p : float
            Particle diameter [μm]
        V_p : float
            Particle volume fraction
        G_m : float
            Matrix shear modulus [GPa]
        b : float
            Burgers vector [nm]
    
        Returns:
        --------
        sigma_orowan : float
            Orowan stress [MPa]
        """
        # Interparticle spacing [m]
        lambda_p = d_p * 1e-6 * (np.sqrt(np.pi / (4 * V_p)) - 1)
    
        # Calculate only when interparticle spacing is positive
        if lambda_p <= 0:
            return 0
    
        # Orowan stress [MPa]
        G_m_Pa = G_m * 1e9  # GPa → Pa
        b_m = b * 1e-9      # nm → m
    
        sigma_orowan = 0.4 * G_m_Pa * b_m / lambda_p / 1e6  # MPa
    
        return sigma_orowan
    
    def ductility_reduction_factor(V_p):
        """
        Ductility reduction factor estimation (empirical model)
    
        Ductility decreases as V_p increases
        """
        return np.exp(-3 * V_p)
    
    # Al alloy parameters
    G_m = 26  # GPa
    b = 0.286  # nm
    
    # Parameter ranges
    d_p_range = np.logspace(-1, 1.2, 40)  # 0.1-16 μm
    V_p_range = np.linspace(0.05, 0.40, 40)
    
    # Create mesh grid
    D_p, V_p_grid = np.meshgrid(d_p_range, V_p_range)
    
    # Calculate Orowan stress
    sigma_orowan_grid = np.zeros_like(D_p)
    performance_index = np.zeros_like(D_p)
    
    for i in range(len(V_p_range)):
        for j in range(len(d_p_range)):
            sigma_o = orowan_stress(D_p[i,j], V_p_grid[i,j], G_m, b)
            sigma_orowan_grid[i,j] = sigma_o
    
            # Performance index: strength × ductility factor
            ductility = ductility_reduction_factor(V_p_grid[i,j])
            performance_index[i,j] = sigma_o * ductility
    
    # 3D plot
    fig = plt.figure(figsize=(16, 6))
    
    # Orowan stress
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(np.log10(D_p), V_p_grid, sigma_orowan_grid,
                              cmap='viridis', alpha=0.8)
    ax1.set_xlabel('log₁₀(Particle Diameter) [μm]')
    ax1.set_ylabel('Volume Fraction')
    ax1.set_zlabel('Orowan Stress [MPa]')
    ax1.set_title('Orowan Strengthening Effect')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Performance index
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(np.log10(D_p), V_p_grid, performance_index,
                              cmap='plasma', alpha=0.8)
    ax2.set_xlabel('log₁₀(Particle Diameter) [μm]')
    ax2.set_ylabel('Volume Fraction')
    ax2.set_zlabel('Performance Index [Strength×Ductility]')
    ax2.set_title('Overall Performance Index')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Contour plot
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(np.log10(D_p), V_p_grid, performance_index,
                            levels=20, cmap='plasma')
    ax3.set_xlabel('log₁₀(Particle Diameter) [μm]')
    ax3.set_ylabel('Volume Fraction')
    ax3.set_title('Performance Index Contour')
    fig.colorbar(contour, ax=ax3)
    
    # Find optimal point
    max_idx = np.unravel_index(np.argmax(performance_index), performance_index.shape)
    d_p_opt = D_p[max_idx]
    V_p_opt = V_p_grid[max_idx]
    sigma_opt = sigma_orowan_grid[max_idx]
    
    ax3.plot(np.log10(d_p_opt), V_p_opt, 'r*', markersize=15,
             label=f'Optimal: d_p={d_p_opt:.2f} μm, V_p={V_p_opt:.2f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('orowan_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Optimal Design of Orowan Strengthening:")
    print("="*60)
    print(f"Optimal particle diameter: {d_p_opt:.2f} μm")
    print(f"Optimal volume fraction: {V_p_opt:.2f}")
    print(f"Orowan stress: {sigma_opt:.1f} MPa")
    print(f"Interparticle spacing: {d_p_opt * (np.sqrt(np.pi/(4*V_p_opt)) - 1):.2f} μm")

## 3.4 Laminate Composites

### 3.4.1 Types and Characteristics of Laminates

By laminating different materials in layers, composite materials can be designed to utilize the properties of each layer. 

Laminate System | Composition | Characteristics | Application  
---|---|---|---  
Metal Laminates | Al/Ti, Cu/Al | Thermal conductivity, weight reduction | Heat exchangers, electronic devices  
Clad Steel | Stainless steel/carbon steel | Corrosion resistance + strength | Chemical plants  
Functionally Graded Materials | Ceramic → metal | Thermal stress mitigation | Thermal barrier coatings  
Electromagnetic Shielding Materials | Cu/resin/Cu | EMI shielding | Electronic substrates  
  
### 3.4.2 Thermal Stress in Laminates

When materials with different thermal expansion coefficients are laminated, stress develops at the interface due to temperature changes. 

$$\sigma_{\text{thermal}} = \frac{E_1 E_2 (\alpha_1 - \alpha_2) \Delta T}{E_1 t_2 + E_2 t_1}$$ 

where \\(E_i\\): elastic modulus of each layer, \\(\alpha_i\\): thermal expansion coefficient, \\(t_i\\): layer thickness, \\(\Delta T\\): temperature change 

#### Example 3.4: Thermal Stress Analysis of Laminates
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_stress_bilayer(E1, E2, alpha1, alpha2, t1, t2, delta_T):
        """
        Thermal stress calculation for bilayer laminate
    
        Parameters:
        -----------
        E1, E2 : float
            Young's modulus of each layer [GPa]
        alpha1, alpha2 : float
            Thermal expansion coefficient of each layer [/°C]
        t1, t2 : float
            Thickness of each layer [mm]
        delta_T : float
            Temperature change [°C]
    
        Returns:
        --------
        sigma1, sigma2 : float
            Thermal stress in each layer [MPa]
        """
        # Thermal stress (simplified model)
        E1_GPa = E1 * 1000  # GPa → MPa
        E2_GPa = E2 * 1000
    
        sigma_thermal = (E1_GPa * E2_GPa * (alpha1 - alpha2) * delta_T /
                         (E1_GPa * t2 + E2_GPa * t1))
    
        # Layer 1 is compression, layer 2 is tension (when α1 > α2)
        sigma1 = -sigma_thermal * t2 / t1
        sigma2 = sigma_thermal
    
        return sigma1, sigma2
    
    # Al/Ti laminate
    E_Al = 70   # GPa
    E_Ti = 110  # GPa
    alpha_Al = 23e-6  # /°C
    alpha_Ti = 9e-6   # /°C
    
    # Varying layer thickness ratio
    t_total = 10  # mm (total thickness)
    t1_ratio = np.linspace(0.1, 0.9, 50)
    t1 = t1_ratio * t_total
    t2 = (1 - t1_ratio) * t_total
    
    delta_T = -155  # °C (180°C → 25°C)
    
    sigma_Al = []
    sigma_Ti = []
    
    for t1_val, t2_val in zip(t1, t2):
        s_Al, s_Ti = thermal_stress_bilayer(E_Al, E_Ti, alpha_Al, alpha_Ti,
                                             t1_val, t2_val, delta_T)
        sigma_Al.append(s_Al)
        sigma_Ti.append(s_Ti)
    
    sigma_Al = np.array(sigma_Al)
    sigma_Ti = np.array(sigma_Ti)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Thermal stress
    ax1.plot(t1_ratio, sigma_Al, 'b-', linewidth=2, label='Al layer stress')
    ax1.plot(t1_ratio, sigma_Ti, 'r-', linewidth=2, label='Ti layer stress')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Al Layer Thickness Ratio (t_Al / t_total)')
    ax1.set_ylabel('Thermal Stress [MPa]')
    ax1.set_title(f'Thermal Stress in Al/Ti Laminate (ΔT = {delta_T}°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Comparison with yield strength
    sigma_y_Al = 100  # MPa (annealed Al)
    sigma_y_Ti = 350  # MPa (pure Ti)
    
    # Safety factor
    SF_Al = np.abs(sigma_y_Al / sigma_Al)
    SF_Ti = np.abs(sigma_y_Ti / sigma_Ti)
    SF_min = np.minimum(SF_Al, SF_Ti)
    
    ax2.plot(t1_ratio, SF_Al, 'b-', linewidth=2, label='Al layer safety factor')
    ax2.plot(t1_ratio, SF_Ti, 'r-', linewidth=2, label='Ti layer safety factor')
    ax2.plot(t1_ratio, SF_min, 'k--', linewidth=2, label='Minimum safety factor')
    ax2.axhline(y=1.0, color='g', linestyle=':', linewidth=1.5, label='Safety limit')
    ax2.set_xlabel('Al Layer Thickness Ratio (t_Al / t_total)')
    ax2.set_ylabel('Safety Factor')
    ax2.set_title('Safety Factor of Each Layer')
    ax2.set_ylim([0, 10])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('laminate_thermal_stress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Optimal thickness ratio (maximum safety factor)
    opt_idx = np.argmax(SF_min)
    t1_opt_ratio = t1_ratio[opt_idx]
    
    print("Thermal Stress Analysis of Al/Ti Laminate:")
    print("="*60)
    print(f"Temperature change: {delta_T}°C")
    print(f"Optimal Al layer thickness ratio: {t1_opt_ratio:.2f}")
    print(f"Minimum safety factor: {SF_min[opt_idx]:.2f}")
    print(f"\nStresses at thickness ratio {t1_opt_ratio:.2f}:")
    print(f"  Al layer stress: {sigma_Al[opt_idx]:.1f} MPa")
    print(f"  Ti layer stress: {sigma_Ti[opt_idx]:.1f} MPa")

### 3.4.3 Functionally Graded Materials (FGM)

Materials with continuously varying composition to mitigate thermal stress. Representative example: Graded material from ZrO₂ (ceramic) → Ni (metal) 

#### Example 3.5: Composition Distribution Design of FGM
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fgm_property_profile(z, n, prop_ceramic, prop_metal):
        """
        FGM property distribution based on power law
    
        Parameters:
        -----------
        z : array
            Thickness direction coordinate (0: ceramic side, 1: metal side)
        n : float
            Gradient index (n=1: linear, n>1: concentrated on ceramic side)
        prop_ceramic, prop_metal : float
            Property values of ceramic and metal
    
        Returns:
        --------
        prop : array
            Property value at position z
        """
        V_metal = z**n
        prop = prop_ceramic * (1 - V_metal) + prop_metal * V_metal
        return prop
    
    # ZrO2/Ni FGM
    E_ZrO2 = 200   # GPa
    E_Ni = 210     # GPa
    alpha_ZrO2 = 10e-6   # /°C
    alpha_Ni = 13e-6     # /°C
    
    # Thickness direction coordinate
    z = np.linspace(0, 1, 100)
    
    # Effect of gradient index
    n_values = [0.5, 1.0, 2.0, 5.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for n in n_values:
        E_profile = fgm_property_profile(z, n, E_ZrO2, E_Ni)
        alpha_profile = fgm_property_profile(z, n, alpha_ZrO2, alpha_Ni)
    
        ax1.plot(z, E_profile, linewidth=2, label=f'n = {n}')
        ax2.plot(z, alpha_profile * 1e6, linewidth=2, label=f'n = {n}')
    
    ax1.set_xlabel('Thickness Direction Coordinate z (0: ZrO₂, 1: Ni)')
    ax1.set_ylabel('Young\'s Modulus [GPa]')
    ax1.set_title('Young\'s Modulus Distribution in FGM')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Thickness Direction Coordinate z (0: ZrO₂, 1: Ni)')
    ax2.set_ylabel('Thermal Expansion Coefficient [×10⁻⁶ /°C]')
    ax2.set_title('Thermal Expansion Coefficient Distribution in FGM')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('fgm_property_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Property Distribution of Functionally Graded Material (FGM):")
    print("="*60)
    print(f"{'Gradient Index n':>12} {'Center E [GPa]':>18} {'Center α [10⁻⁶/°C]':>25}")
    print("-"*60)
    
    for n in n_values:
        E_mid = fgm_property_profile(0.5, n, E_ZrO2, E_Ni)
        alpha_mid = fgm_property_profile(0.5, n, alpha_ZrO2, alpha_Ni)
        print(f"{n:12.1f} {E_mid:18.1f} {alpha_mid*1e6:25.2f}")

## 3.5 Summary

In this chapter, we learned about particle-reinforced and laminate composites:

  * Classification of particle-reinforced composites (dispersion strengthening, particle strengthening, fillers)
  * Types and manufacturing methods of MMC/CMC
  * Elastic modulus prediction using Hashin-Shtrikman model
  * Orowan mechanism and optimal particle design
  * Thermal stress in laminates and functionally graded materials (FGM)

In the next chapter, we will learn about mechanical evaluation methods for composite materials (tensile testing, bending testing, impact testing) and nondestructive inspection (ultrasonic, X-ray CT, thermography). 

## Exercises

### Foundation Level

#### Problem 3.1: Hashin-Shtrikman Model

Calculate the Young's modulus of an Al alloy (E=70 GPa, ν=0.33) composite containing 20 vol% Al₂O₃ particles (E=380 GPa, ν=0.23) using the Hashin-Shtrikman model. 

#### Problem 3.2: Interparticle Spacing Calculation

Calculate the average interparticle spacing in a composite containing 15 vol% SiC particles with a diameter of 2 μm. 

#### Problem 3.3: Thermal Stress Calculation

Calculate the thermal stress when a bilayer laminate of Cu (E=120 GPa, α=17×10⁻⁶ /°C) and Al (E=70 GPa, α=23×10⁻⁶ /°C) (each layer 1 mm thick) experiences a 100°C temperature decrease. 

### Application Level

#### Problem 3.4: Optimization of Orowan Strengthening

For an Al alloy (G=26 GPa, b=0.286 nm), determine the optimal combination of SiC particle size and volume fraction to achieve a target yield strength of 200 MPa. (Matrix yield strength: 100 MPa) 

#### Problem 3.5: MMC Design

Design a SiC/Al composite for automotive engine pistons. Required properties: Young's modulus ≥ 100 GPa, density ≤ 2.9 g/cm³ 

#### Problem 3.6: Laminate Optimization

Optimize the layer thickness ratio of an Al/Ti laminate (total thickness 5 mm) to maximize the minimum safety factor for a 200°C temperature change. 

#### Problem 3.7: Programming Exercise

Create a property prediction program for particle-reinforced composites: 

  * Calculate elastic modulus using Hashin-Shtrikman model
  * Calculate strength using Orowan model
  * Create contour plots for particle size and volume fraction

### Advanced Level

#### Problem 3.8: Multi-Objective Optimization

For a SiC/Al composite, simultaneously optimize: 

  * Objective 1: Maximize specific strength (strength/density)
  * Objective 2: Minimize cost
  * Constraint: Young's modulus ≥ 90 GPa

Plot the Pareto optimal solutions.

#### Problem 3.9: Thermal Stress Analysis of FGM

For a ZrO₂/Ni functionally graded material (thickness 10 mm), use finite element method to calculate temperature distribution and thermal stress distribution. (Surface temperature: ZrO₂ side 1200°C, Ni side 400°C) 

#### Problem 3.10: Nano-Particle Dispersion Strengthening

Analyze the strengthening mechanism of ODS alloys with nano-sized Al₂O₃ particles (diameter 10-100 nm). Determine the critical size at which the mechanism transitions from Orowan mechanism to dislocation cutting mechanism when particle size becomes less than 10 nm. 

## References

  1. Chawla, N. and Chawla, K. K., "Metal Matrix Composites", 2nd ed., Springer, 2013, pp. 89-156, 234-278
  2. Clyne, T. W. and Withers, P. J., "An Introduction to Metal Matrix Composites", Cambridge University Press, 1993, pp. 67-112
  3. Kainer, K. U., "Metal Matrix Composites: Custom-made Materials for Automotive and Aerospace Engineering", Wiley-VCH, 2006, pp. 45-89
  4. Courtney, T. H., "Mechanical Behavior of Materials", 2nd ed., Waveland Press, 2005, pp. 389-445
  5. Hashin, Z. and Shtrikman, S., "A Variational Approach to the Theory of the Elastic Behaviour of Multiphase Materials", Journal of the Mechanics and Physics of Solids, Vol. 11, 1963, pp. 127-140
  6. Koizumi, M., "FGM Activities in Japan", Composites Part B, Vol. 28, 1997, pp. 1-4
  7. Suresh, S. and Mortensen, A., "Fundamentals of Functionally Graded Materials", IOM Communications, 1998, pp. 23-67, 134-189
  8. Naebe, M. and Shirvanimoghaddam, K., "Functionally Graded Materials: A Review of Fabrication and Properties", Applied Materials Today, Vol. 5, 2016, pp. 223-245

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
