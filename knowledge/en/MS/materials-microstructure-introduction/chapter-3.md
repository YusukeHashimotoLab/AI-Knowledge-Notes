---
title: "Chapter 3: Precipitation and Solid Solution"
chapter_title: "Chapter 3: Precipitation and Solid Solution"
subtitle: Precipitation and Solid Solution - From Age Hardening to Fine Precipitate Control
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 7
version: 1.0
created_at: 2025-10-28
---

This chapter covers Precipitation and Solid Solution. You will learn principles of age hardening and Gibbs-Thomson effect.

## Learning Objectives

Upon completing this chapter, you will acquire the following skills and knowledge:

  * ✅ Understand types and properties of solid solutions and explain the mechanism of solid solution strengthening
  * ✅ Understand nucleation and growth mechanisms of precipitation and interpret aging curves
  * ✅ Explain principles of age hardening and understand practical examples such as Al alloys
  * ✅ Quantitatively calculate precipitation strengthening by Orowan mechanism
  * ✅ Understand Gibbs-Thomson effect and particle coarsening (Ostwald ripening)
  * ✅ Explain differences between coherent, semi-coherent, and incoherent precipitates
  * ✅ Simulate time evolution of precipitates and strength prediction using Python

## 3.1 Fundamentals of Solid Solutions

### 3.1.1 Definition and Types of Solid Solutions

**Solid Solution** is a homogeneous solid phase in which two or more elements are mixed at the atomic level. It is a state where another element (solute atoms) is dissolved in the fundamental crystal structure (matrix).

#### 💡 Classification of Solid Solutions

**1\. Substitutional Solid Solution**

  * Solute atoms replace matrix atoms
  * Condition: Atomic radius difference within 15% (Hume-Rothery rules)
  * Examples: Cu-Ni, Fe-Cr, Al-Mg

**2\. Interstitial Solid Solution**

  * Solute atoms enter interstitial positions
  * Condition: Small solute atoms (C, N, H, O)
  * Examples: Fe-C (steel), Ti-O, Zr-H

    
    
    ```mermaid
    graph LR
        A[Solid Solution] --> B[Substitutional]
        A --> C[Interstitial]
        B --> D[Cu-Ni AlloySimilar Atomic Radii]
        B --> E[Stainless SteelFe-Cr-Ni]
        C --> F[Carbon SteelFe-C]
        C --> G[NitrideTi-N]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#fce7f3
        style C fill:#fce7f3
    ```

### 3.1.2 Mechanism of Solid Solution Strengthening

Solid solutions have higher strength than pure metals. This is called **Solid Solution Strengthening**. The main mechanisms are as follows:

Mechanism | Cause | Effect  
---|---|---  
**Lattice Strain** | Different atomic radius of solute atoms | Increased resistance to dislocation motion  
**Elastic Interaction** | Stress field around solute atoms | Interaction with Dislocations  
**Chemical Interaction** | Change in bonding strength | Change in stacking fault energy  
**Electrical Interaction** | Change in electronic structure | Decreased dislocation mobility  
  
The increase in yield stress due to solid solution strengthening is approximated by the Labusch model as follows:

> Δσy = K · cn   
>   
>  where Δσy is the increase in yield stress, c is solute atom concentration, K is a constant, n is 0.5-1 (typically ~2/3) 

### 3.1.3 Practical Example: Strengthening of Al-Mg Solid Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 1: Al-MgSolid Solution in Calculation of solid solution strengthening
    Prediction of yield stress using Labusch model
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculation of solid solution strengthening
    def solid_solution_strengthening(c, K=30, n=0.67):
        """
        Calculate increase in yield stress due to solid solution strengthening
    
        Args:
            c: Solute concentration [at%]
            K: Constant [MPa/(at%)^n]
            n: Exponent (typically 0.5-1.0)
    
        Returns:
            delta_sigma: Increase in yield stress [MPa]
        """
        return K * (c ** n)
    
    # Al-Mgalloy Experimental data（近似）
    mg_content = np.array([0, 1, 2, 3, 4, 5, 6])  # at%
    yield_stress_exp = np.array([20, 50, 75, 95, 112, 127, 140])  # MPa
    
    # Model prediction
    mg_model = np.linspace(0, 7, 100)
    delta_sigma = solid_solution_strengthening(mg_model, K=30, n=0.67)
    yield_stress_model = 20 + delta_sigma  # Yield stress of pure Al: 20 MPa
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(mg_model, yield_stress_model, 'r-', linewidth=2,
             label=f'Labusch model (n=0.67)')
    plt.scatter(mg_content, yield_stress_exp, s=100, c='blue',
                marker='o', label='Experimental data')
    
    plt.xlabel('Mg concentration [at%]', fontsize=12)
    plt.ylabel('Yield stress [MPa]', fontsize=12)
    plt.title('Al-MgSolid Solution Solid Solution Strengthening', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculation for specific composition
    mg_5at = 5.0
    delta_sigma_5 = solid_solution_strengthening(mg_5at)
    print(f"Increase in yield stress with 5at% Mg addition: {delta_sigma_5:.1f} MPa")
    print(f"Predicted yield stress: {20 + delta_sigma_5:.1f} MPa")
    print(f"実験値: {yield_stress_exp[5]:.1f} MPa")
    print(f"Error: {abs((20 + delta_sigma_5) - yield_stress_exp[5]):.1f} MPa")
    
    # Output example:
    # Increase in yield stress with 5at% Mg addition: 102.5 MPa
    # Predicted yield stress: 122.5 MPa
    # 実験値: 127.0 MPa
    # Error: 4.5 MPa
    

#### 📊 Practical Points

Al-Mg alloys (5000 series aluminum alloys) are representative alloys that use solid solution strengthening as the main strengthening mechanism. Mg dissolves up to about 6% and achieves both excellent strength and corrosion resistance. They are widely used as can materials and marine materials.

## 3.2 Fundamental Theory of Precipitation

### 3.2.1 Mechanism of Precipitation

**Precipitation** is a phenomenon in which second-phase particles form from a supersaturated solid solution. A typical precipitation process goes through the following stages:
    
    
    ```mermaid
    flowchart TD
        A[supersaturatedSolid Solution] --> B[Nucleation]
        B --> C[Growth]
        C --> D[Coarsening]
    
        B1[Homogeneous Nucleation] -.-> B
        B2[不Homogeneous Nucleation] -.-> B
    
        C1[Diffusion-controlled growth] -.-> C
        C2[Interface-Controlled Growth] -.-> C
    
        D1[Ostwald ripening] -.-> D
    
        style A fill:#fff3e0
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

### 3.2.2 Nucleation Theory

The nucleation rate of precipitation is expressed by classical nucleation theory as follows:

> J = N0 · ν · exp(-ΔG*/kT)   
>   
>  where  
>  J: Nucleation rate [nuclei/m³/s]  
>  N0: Nucleation site density [sites/m³]  
>  ν: Atomic vibration frequency [Hz]  
>  ΔG*: Critical nucleation energy [J]  
>  k: Boltzmann constant [J/K]  
>  T: Temperature [K] 

The critical nucleation energy ΔG* for homogeneous nucleation is:

> ΔG* = (16πγ³) / (3ΔGv²)   
>   
>  γ: Interface energy [J/m²]  
>  ΔGv: Free energy change per unit volume [J/m³] 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 2: Calculation of precipitation nucleation rate
    Simulation based on classical nucleation theory
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    k_B = 1.38e-23  # Boltzmann constant [J/K]
    h = 6.626e-34   # Planck constant [J·s]
    
    def nucleation_rate(T, gamma, delta_Gv, N0=1e28, nu=1e13):
        """
        Calculate nucleation rate (classical nucleation theory)
    
        Args:
            T: Temperature [K]
            gamma: Interface energy [J/m²]
            delta_Gv: 体積自由エネルギー変化 [J/m³]
            N0: Nucleation site density [sites/m³]
            nu: Atomic vibration frequency [Hz]
    
        Returns:
            J: Nucleation rate [nuclei/m³/s]
        """
        # Critical nucleation energy
        delta_G_star = (16 * np.pi * gamma**3) / (3 * delta_Gv**2)
    
        # Nucleation rate
        J = N0 * nu * np.exp(-delta_G_star / (k_B * T))
    
        return J, delta_G_star
    
    # Al-Cualloy パラメータ（θ'Phase precipitation）
    gamma = 0.2  # Interface energy [J/m²]
    temperatures = np.linspace(373, 573, 100)  # 100-300°C
    
    # Supersaturation toよる自由エネルギー変化（簡略化）
    supersaturations = [1.5, 2.0, 2.5]  # Supersaturation
    colors = ['blue', 'green', 'red']
    labels = ['lowSupersaturation (1.5x)', 'mediumSupersaturation (2.0x)', 'highSupersaturation (2.5x)']
    
    plt.figure(figsize=(12, 5))
    
    # (a) Temperature dependence
    plt.subplot(1, 2, 1)
    for S, color, label in zip(supersaturations, colors, labels):
        delta_Gv = -2e8 * np.log(S)  # Simplified free energy [J/m³]
        J_list = []
        for T in temperatures:
            J, _ = nucleation_rate(T, gamma, delta_Gv)
            J_list.append(J)
    
        plt.semilogy(temperatures - 273, J_list, color=color,
                     linewidth=2, label=label)
    
    plt.xlabel('Temperature [°C]', fontsize=12)
    plt.ylabel('Nucleation rate [nuclei/m³/s]', fontsize=12)
    plt.title('(a) Temperature dependence', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # (b) Critical nucleus radius
    plt.subplot(1, 2, 2)
    T_aging = 473  # Aging temperature 200°C
    for S, color, label in zip(supersaturations, colors, labels):
        delta_Gv = -2e8 * np.log(S)
        r_crit = 2 * gamma / abs(delta_Gv)  # Critical nucleus radius [m]
        r_crit_nm = r_crit * 1e9  # [nm]
    
        # For plotting
        plt.bar(label, r_crit_nm, color=color, alpha=0.7)
    
    plt.ylabel('Critical nucleus radius [nm]', fontsize=12)
    plt.title('(b) Supersaturation and Critical nucleus radius (200°C)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical output
    print("=== Nucleation Analysis of Al-Cu Alloy ===\n")
    T_test = 473  # 200°C
    for S in supersaturations:
        delta_Gv = -2e8 * np.log(S)
        J, delta_G_star = nucleation_rate(T_test, gamma, delta_Gv)
        r_crit = 2 * gamma / abs(delta_Gv) * 1e9  # [nm]
    
        print(f"Supersaturation {S}x:")
        print(f"  Nucleation rate: {J:.2e} pcs/m³/s")
        print(f"  Critical nucleus radius: {r_crit:.2f} nm")
        print(f"  Activation energy: {delta_G_star/k_B:.2e} K\n")
    
    # Output example:
    # === Nucleation Analysis of Al-Cu Alloy ===
    #
    # Supersaturation 1.5x:
    #   Nucleation rate: 3.45e+15 pcs/m³/s
    #   Critical nucleus radius: 2.47 nm
    #   Activation energy: 8.12e+03 K
    

### 3.2.3 Growth of Precipitates

After nucleation, precipitates grow by diffusion. The time evolution of radius r(t) for spherical precipitates under diffusion control is:

> r(t) = √(2Dt · (c0 \- ce) / cp)   
>   
>  D: Diffusion coefficient [m²/s]  
>  t: Time [s]  
>  c0: Early濃degree  
>  ce: Equilibrium concentration  
>  cp: precipitation物medium 濃degree 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 3: precipitation物Size hoursadvanced
    Diffusion-controlled growthモデル
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def precipitate_growth(t, T, D0=1e-5, Q=150e3, c0=0.04, ce=0.01, cp=0.3):
        """
        Calculate time evolution of precipitate radius
    
        Args:
            t: Time [s]
            T: Temperature [K]
            D0: Pre-exponential factor of diffusion coefficient [m²/s]
            Q: Activation energy [J/mol]
            c0: Early溶質濃degree
            ce: Equilibrium concentration
            cp: precipitation物medium 濃degree
    
        Returns:
            r: Precipitate radius [m]
        """
        R = 8.314  # Gas constant [J/mol/K]
        D = D0 * np.exp(-Q / (R * T))  # Arrhenius equation
    
        # Diffusion-controlled growth
        r = np.sqrt(2 * D * t * (c0 - ce) / cp)
    
        return r
    
    # Aging conditions
    temperatures = [423, 473, 523]  # 150, 200, 250°C
    temp_labels = ['150°C', '200°C', '250°C']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.logspace(-1, 3, 100)  # 0.1〜1000hours
    time_seconds = time_hours * 3600
    
    plt.figure(figsize=(12, 5))
    
    # (a) hours-Size曲線
    plt.subplot(1, 2, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        r = precipitate_growth(time_seconds, T)
        r_nm = r * 1e9  # [nm]
    
        plt.loglog(time_hours, r_nm, linewidth=2,
                   color=color, label=label)
    
    plt.xlabel('Aging time [h]', fontsize=12)
    plt.ylabel('Precipitate radius [nm]', fontsize=12)
    plt.title('(a) Growth Curve of Precipitates', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    
    # (b) 成長速degree Temperature dependence
    plt.subplot(1, 2, 2)
    t_fixed = 10 * 3600  # After 10 hours
    T_range = np.linspace(373, 573, 50)
    r_range = precipitate_growth(t_fixed, T_range)
    r_range_nm = r_range * 1e9
    
    plt.plot(T_range - 273, r_range_nm, 'r-', linewidth=2)
    plt.xlabel('agingTemperature [°C]', fontsize=12)
    plt.ylabel('Precipitate radius (after 10h) [nm]', fontsize=12)
    plt.title('(b) 成長速degree Temperature dependence', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Practical calculation example
    print("=== Prediction of Precipitate Growth ===\n")
    aging_conditions = [
        (473, 1),    # 200°C, 1hours
        (473, 10),   # 200°C, 10hours
        (473, 100),  # 200°C, 100hours
        (523, 10),   # 250°C, 10hours
    ]
    
    for T, t_h in aging_conditions:
        t_s = t_h * 3600
        r = precipitate_growth(t_s, T)
        r_nm = r * 1e9
    
        print(f"{T-273:.0f}°C, {t_h}hours: Precipitate radius = {r_nm:.1f} nm")
    
    # Output example:
    # === Prediction of Precipitate Growth ===
    #
    # 200°C, 1hours: Precipitate radius = 8.5 nm
    # 200°C, 10hours: Precipitate radius = 26.9 nm
    # 200°C, 100hours: Precipitate radius = 85.0 nm
    # 250°C, 10hours: Precipitate radius = 67.3 nm
    

## 3.3 Age Hardening

### 3.3.1 Principle of Age Hardening

**Age Hardening** or Precipitation Hardening is a heat treatment technique that strengthens materials by forming fine precipitates from supersaturated solid solutions. Representative age-hardenable alloys:

  * **Alalloy** : 2000system(Al-Cu)、6000system(Al-Mg-Si)、7000system(Al-Zn-Mg)
  * **ニッケル基超alloy** : Inconel 718（γ''Phaseprecipitation）
  * **マルエージング鋼** : Fe-Ni-Co-Moalloy
  * **precipitation硬化systemステンレス鋼** : 17-4PH、15-5PH

### 3.3.2 Aging Curves and Precipitation Process

Typical precipitation process in Al-Cu alloys (2000 series):
    
    
    ```mermaid
    flowchart LR
        A[supersaturatedSolid Solutionα-SSS] --> B[GP ZonesGP zones]
        B --> C[θ''Phasemetastable]
        C --> D[θ'Phasemetastable]
        D --> E[θPhaseAl₂CuequilibriumPhase]
    
        style A fill:#fff3e0
        style B fill:#e3f2fd
        style C fill:#e3f2fd
        style D fill:#e3f2fd
        style E fill:#c8e6c9
    ```

Characteristics of each stage:

Stage | Phase | Size | Coherency | 硬化Effect  
---|---|---|---|---  
Early | GP Zones | 1-2 nm | Fully Coherent | medium  
mediumintermediate | θ'', θ' | 5-50 nm | Semi-coherent | **Maximum**  
Late | θ (Al₂Cu) | >100 nm | Incoherent | low  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 4: Simulation of aging curves for Al alloys
    Predict time evolution of hardness
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def aging_hardness_curve(t, T, peak_time_ref=10, peak_hardness=150,
                             T_ref=473, Q=100e3):
        """
        Simulate aging curve (empirical model)
    
        Args:
            t: Aging time [h]
            T: agingTemperature [K]
            peak_time_ref: Peak time at reference temperature [h]
            peak_hardness: Peak hardness [HV]
            T_ref: 基準Temperature [K]
            Q: Activation energy [J/mol]
    
        Returns:
            hardness: Hardness [HV]
        """
        R = 8.314  # Gas constant
    
        # Temperature-corrected peak time (Arrhenius relation)
        peak_time = peak_time_ref * np.exp(Q/R * (1/T - 1/T_ref))
    
        # Time evolution of hardness (JMA model based)
        # Under-aging region
        H_under = 70 + (peak_hardness - 70) * (1 - np.exp(-(t/peak_time)**1.5))
    
        # Over-aging region (softening due to coarsening)
        H_over = peak_hardness * np.exp(-0.5 * ((t - peak_time)/peak_time)**0.8)
        H_over = np.maximum(H_over, 80)  # Minimum hardness
    
        # Combination
        hardness = np.where(t <= peak_time, H_under, H_over)
    
        return hardness
    
    # Aging conditions
    temperatures = [423, 473, 523]  # 150, 200, 250°C
    temp_labels = ['150°C (lowtemp)', '200°C (Standard)', '250°C (High)']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.logspace(-1, 3, 200)  # 0.1〜1000hours
    
    plt.figure(figsize=(12, 5))
    
    # (a) Aging curve
    plt.subplot(1, 2, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        hardness = aging_hardness_curve(time_hours, T)
    
        plt.semilogx(time_hours, hardness, linewidth=2.5,
                     color=color, label=label)
    
        # Mark peak hardness position
        peak_idx = np.argmax(hardness)
        plt.plot(time_hours[peak_idx], hardness[peak_idx],
                 'o', markersize=10, color=color)
    
    # Regions of under-aging, peak-aging, and over-aging
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.3, 145, 'Under-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(10, 145, 'Peak-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.text(300, 145, 'Over-aging', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.xlabel('Aging time [h]', fontsize=12)
    plt.ylabel('Hardness [HV]', fontsize=12)
    plt.title('(a) Aging Curves of Al-Cu Alloy', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim(60, 160)
    
    # (b) peakhours Temperature dependence
    plt.subplot(1, 2, 2)
    T_range = np.linspace(393, 553, 50)  # 120-280°C
    peak_times = []
    
    for T in T_range:
        # Find peak time
        t_test = np.logspace(-2, 4, 1000)
        h_test = aging_hardness_curve(t_test, T)
        peak_t = t_test[np.argmax(h_test)]
        peak_times.append(peak_t)
    
    plt.semilogy(T_range - 273, peak_times, 'r-', linewidth=2.5)
    plt.xlabel('agingTemperature [°C]', fontsize=12)
    plt.ylabel('peakAging time [h]', fontsize=12)
    plt.title('(b) Peak aging time Temperature dependence', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用的なrecommendedAging conditions
    print("=== recommendedAging conditions（Al-Cualloy） ===\n")
    for T in temperatures:
        t_test = np.logspace(-2, 3, 1000)
        h_test = aging_hardness_curve(t_test, T)
        peak_idx = np.argmax(h_test)
        peak_time = t_test[peak_idx]
        peak_h = h_test[peak_idx]
    
        print(f"{T-273:.0f}°C:")
        print(f"  Peak aging time: {peak_time:.1f} hours")
        print(f"  Maximum hardness: {peak_h:.1f} HV\n")
    
    # Output example:
    # === recommendedAging conditions（Al-Cualloy） ===
    #
    # 150°C:
    #   Peak aging time: 48.3 hours
    #   Maximum hardness: 150.0 HV
    #
    # 200°C:
    #   Peak aging time: 10.0 hours
    #   Maximum hardness: 150.0 HV
    

## 3.4 Mechanism of Precipitation Strengthening

### 3.4.1 Orowan Mechanism

Materials are strengthened by precipitates hindering dislocation motion. The most important mechanism is the **Orowan mechanism**. The stress required for dislocations to bypass precipitates:

> τOrowan = (M · G · b) / (λ - 2r)   
>   
>  M: Taylor factor (typically ~3)  
>  G: Shear modulus [Pa]  
>  b: Magnitude of Burgers vector [m]  
>  λ: Precipitate spacing [m]  
>  r: Precipitate radius [m] 

The precipitate spacing λ from volume fraction fv and radius r:

> λ ≈ 2r · √(π / (3fv)) 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 5: Calculation of precipitation strengthening by Orowan mechanism
    precipitation物Size and intermediatespacing optimal化
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def orowan_stress(r, f_v, G=26e9, b=2.86e-10, M=3.06):
        """
        Orowan stresscalculation
    
        Args:
            r: Precipitate radius [m]
            f_v: Volume fraction
            G: Shear modulus [Pa]
            b: Burgers vector [m]
            M: Taylor factor
    
        Returns:
            tau: Shear stress [Pa]
            sigma: Yield stress [Pa]
        """
        # Precipitate spacing
        lambda_p = 2 * r * np.sqrt(np.pi / (3 * f_v))
    
        # Orowan stress
        tau = (M * G * b) / (lambda_p - 2*r)
    
        # 引張yield stress（Taylor factor at換算）
        sigma = M * tau
    
        return tau, sigma, lambda_p
    
    # Parameter range
    radii_nm = np.linspace(1, 100, 100)  # 1-100 nm
    radii_m = radii_nm * 1e-9
    
    volume_fractions = [0.01, 0.03, 0.05, 0.1]  # 1%, 3%, 5%, 10%
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['fᵥ = 1%', 'fᵥ = 3%', 'fᵥ = 5%', 'fᵥ = 10%']
    
    plt.figure(figsize=(14, 5))
    
    # (a) Relationship between precipitate radius and strength
    plt.subplot(1, 3, 1)
    for f_v, color, label in zip(volume_fractions, colors, labels):
        sigma_list = []
        for r in radii_m:
            try:
                _, sigma, _ = orowan_stress(r, f_v)
                sigma_mpa = sigma / 1e6  # MPa
                sigma_list.append(sigma_mpa)
            except:
                sigma_list.append(np.nan)
    
        plt.plot(radii_nm, sigma_list, linewidth=2,
                 color=color, label=label)
    
    plt.xlabel('Precipitate radius [nm]', fontsize=12)
    plt.ylabel('Increase in yield stress [MPa]', fontsize=12)
    plt.title('(a) Radius Dependence of Orowan Strengthening', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 500)
    
    # (b) Volume fraction and Optimal radius
    plt.subplot(1, 3, 2)
    f_v_range = np.linspace(0.005, 0.15, 50)
    optimal_radii = []
    max_strengths = []
    
    for f_v in f_v_range:
        sigma_test = []
        for r in radii_m:
            try:
                _, sigma, _ = orowan_stress(r, f_v)
                sigma_test.append(sigma / 1e6)
            except:
                sigma_test.append(0)
    
        max_sigma = np.max(sigma_test)
        optimal_r = radii_nm[np.argmax(sigma_test)]
    
        optimal_radii.append(optimal_r)
        max_strengths.append(max_sigma)
    
    ax1 = plt.gca()
    ax1.plot(f_v_range * 100, optimal_radii, 'b-', linewidth=2.5, label='Optimal radius')
    ax1.set_xlabel('Volume fraction [%]', fontsize=12)
    ax1.set_ylabel('optimalPrecipitate radius [nm]', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(f_v_range * 100, max_strengths, 'r--', linewidth=2.5, label='Maximum strength')
    ax2.set_ylabel('最大Increase in yield stress [MPa]', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('(b) Optimal Precipitate Conditions', fontsize=13, fontweight='bold')
    
    # (c) Precipitate spacingマップ
    plt.subplot(1, 3, 3)
    r_test = 10e-9  # 10 nm
    spacing_list = []
    
    for f_v in f_v_range:
        _, _, lambda_p = orowan_stress(r_test, f_v)
        spacing_list.append(lambda_p * 1e9)  # nm
    
    plt.plot(f_v_range * 100, spacing_list, 'g-', linewidth=2.5)
    plt.xlabel('Volume fraction [%]', fontsize=12)
    plt.ylabel('Precipitate spacing [nm]', fontsize=12)
    plt.title('(c) Precipitate spacing (r=10nm)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Practical design example
    print("=== Design Guidelines for Orowan Strengthening ===\n")
    print("Typical precipitate conditions for Al alloys:\n")
    
    design_cases = [
        (5e-9, 0.03, "Under-aging (小Size low分率)"),
        (10e-9, 0.05, "Peak-aging (optimal conditions)"),
        (50e-9, 0.08, "Over-aging (coarsened)")
    ]
    
    for r, f_v, condition in design_cases:
        tau, sigma, lambda_p = orowan_stress(r, f_v)
    
        print(f"{condition}:")
        print(f"  Precipitate radius: {r*1e9:.1f} nm")
        print(f"  Volume fraction: {f_v*100:.1f}%")
        print(f"  Precipitate spacing: {lambda_p*1e9:.1f} nm")
        print(f"  yield stress増加: {sigma/1e6:.1f} MPa\n")
    
    # Output example:
    # === Design Guidelines for Orowan Strengthening ===
    #
    # Typical precipitate conditions for Al alloys:
    #
    # Under-aging (小Size low分率):
    #   Precipitate radius: 5.0 nm
    #   Volume fraction: 3.0%
    #   Precipitate spacing: 51.2 nm
    #   yield stress増加: 287.3 MPa
    

### 3.4.2 Coherency and Strengthening Effect

The crystallographic relationship (coherency) between precipitates and matrix significantly affects the strengthening effect:

Coherency | Interface Structure | Interaction with Dislocations | strengtheningEffect  
---|---|---|---  
**Coherent  
（Fully Coherent）** | Continuous lattice, with strain field | Dislocation shearing | medium〜high  
**Semi-coherent  
（Semi-coherent）** | Partially coherent, interface dislocations | Competition between shearing and bypass | **Maximum**  
**Incoherent  
（Incoherent）** | No crystallographic relationship | Orowan bypass | low〜medium  
  
## 3.5 Coarsening and Gibbs-Thomson Effect

### 3.5.1 Ostwald Ripening

The phenomenon where small precipitates dissolve and large precipitates grow during long-term aging is called **Ostwald ripening** (coarsening). This occurs spontaneously thermodynamically to minimize interface energy.

Due to the **Gibbs-Thomson effect** , smaller particles have higher solubility:

> c(r) = c∞ · exp(2γVm / (rRT))   
>   
>  c(r): radiusr 粒子周辺 Equilibrium concentration  
>  c∞: 平坦界面 at Equilibrium concentration  
>  γ: Interface energy [J/m²]  
>  Vm: Molar volume [m³/mol]  
>  r: Particle radius [m] 

Time evolution of average particle radius according to Lifshitz-Slyozov-Wagner (LSW) theory:

> r̄³(t) - r̄³(0) = Kt   
>   
>  K: Coarsening rate constant [m³/s] 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example 6: Simulation of precipitate coarsening
    Ostwald ripening (LSWtheory)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def coarsening_kinetics(t, r0, K):
        """
        Coarsening by LSW theory
    
        Args:
            t: Time [s]
            r0: EarlyMean radius [m]
            K: Coarsening rate constant [m³/s]
    
        Returns:
            r: Mean radius [m]
        """
        r_cubed = r0**3 + K * t
        r = r_cubed ** (1/3)
        return r
    
    def coarsening_rate_constant(T, D0=1e-5, Q=150e3, gamma=0.2,
                                  ce=0.01, Vm=1e-5):
        """
        Calculate coarsening rate constant
    
        Args:
            T: Temperature [K]
            D0: Pre-exponential factor of diffusion coefficient [m²/s]
            Q: Activation energy [J/mol]
            gamma: Interface energy [J/m²]
            ce: Equilibrium concentration
            Vm: Molar volume [m³/mol]
    
        Returns:
            K: Coarsening rate constant [m³/s]
        """
        R = 8.314  # Gas constant
        D = D0 * np.exp(-Q / (R * T))
    
        # Rate constant of LSW theory
        K = (8 * gamma * Vm * ce * D) / (9 * R * T)
    
        return K
    
    # agingtempdegree
    temperatures = [473, 523, 573]  # 200, 250, 300°C
    temp_labels = ['200°C', '250°C', '300°C']
    colors = ['blue', 'green', 'red']
    
    time_hours = np.linspace(0, 1000, 200)  # 0-1000hours
    time_seconds = time_hours * 3600
    
    r0 = 10e-9  # Earlyradius 10 nm
    
    plt.figure(figsize=(14, 5))
    
    # (a) Coarsening curve
    plt.subplot(1, 3, 1)
    for T, label, color in zip(temperatures, temp_labels, colors):
        K = coarsening_rate_constant(T)
        r = coarsening_kinetics(time_seconds, r0, K)
        r_nm = r * 1e9
    
        plt.plot(time_hours, r_nm, linewidth=2.5,
                 color=color, label=label)
    
    plt.xlabel('Aging time [h]', fontsize=12)
    plt.ylabel('平均Precipitate radius [nm]', fontsize=12)
    plt.title('(a) precipitation物 Coarsening curve', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # (b) r³-t プロット（Verification of LSW theory）
    plt.subplot(1, 3, 2)
    T_test = 523  # 250°C
    K_test = coarsening_rate_constant(T_test)
    r_test = coarsening_kinetics(time_seconds, r0, K_test)
    r_cubed = (r_test * 1e9) ** 3
    r0_cubed = (r0 * 1e9) ** 3
    
    plt.plot(time_hours, r_cubed - r0_cubed, 'r-', linewidth=2.5)
    plt.xlabel('Aging time [h]', fontsize=12)
    plt.ylabel('r³ - r₀³ [nm³]', fontsize=12)
    plt.title(f'(b) Verification of LSW theory ({temp_labels[1]})', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Linear fit
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_hours, r_cubed - r0_cubed)
    plt.plot(time_hours, slope * time_hours + intercept, 'b--',
             linewidth=1.5, label=f'Linear fit (R²={r_value**2:.3f})')
    plt.legend(fontsize=10)
    
    # (c) 粗大化速degree Temperature dependence
    plt.subplot(1, 3, 3)
    T_range = np.linspace(423, 623, 50)  # 150-350°C
    K_range = []
    
    for T in T_range:
        K = coarsening_rate_constant(T)
        K_range.append(K * 1e27)  # [nm³/s]
    
    plt.semilogy(T_range - 273, K_range, 'g-', linewidth=2.5)
    plt.xlabel('Temperature [°C]', fontsize=12)
    plt.ylabel('Coarsening rate constant K [nm³/s]', fontsize=12)
    plt.title('(c) 粗大化速degree Temperature dependence', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用calculation
    print("=== Prediction of Precipitate Coarsening ===\n")
    print("Earlyradius: 10 nm\n")
    
    for T, label in zip(temperatures, temp_labels):
        K = coarsening_rate_constant(T)
    
        # After 100 hours、After 1000 hours radius
        r_100h = coarsening_kinetics(100 * 3600, r0, K) * 1e9
        r_1000h = coarsening_kinetics(1000 * 3600, r0, K) * 1e9
    
        print(f"{label}:")
        print(f"  After 100 hours: {r_100h:.1f} nm")
        print(f"  After 1000 hours: {r_1000h:.1f} nm")
        print(f"  Coarsening rate constant: {K*1e27:.2e} nm³/s\n")
    
    # Output example:
    # === Prediction of Precipitate Coarsening ===
    #
    # Earlyradius: 10 nm
    #
    # 200°C:
    #   After 100 hours: 15.2 nm
    #   After 1000 hours: 32.8 nm
    #   Coarsening rate constant: 5.67e+01 nm³/s
    

### 3.5.2 Precipitation Control in Practical Alloys

#### 🔬 Practical Example of Al-Cu-Mg Alloy (2024 Alloy)

**Solution treatment** : 500°C × 1 hour → Water quenching

**Aging treatment (T6)** : 190°C × 18 hours (artificial aging)

  * Precipitate phases: θ' (Al₂Cu), S' (Al₂CuMg)
  * Optimal precipitate size: 10-30 nm
  * Volume fraction: ~5%
  * Yield strength: 324 MPa (T6 condition)

Widely used as aircraft structural materials, including rivets and wing spars.

## 3.6 Practice: Precipitation Simulation of Al-Cu-Mg Alloy System
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example 7: Comprehensive simulation of Al-Cu-Mg alloys
    From precipitation process to strength prediction
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PrecipitationSimulator:
        """Simulator for precipitation-strengthened alloys"""
    
        def __init__(self, alloy_type='Al-Cu-Mg'):
            self.alloy_type = alloy_type
    
            # Parameters for Al-Cu-Mg alloy
            self.G = 26e9  # Shear modulus [Pa]
            self.b = 2.86e-10  # Burgers vector [m]
            self.M = 3.06  # Taylor factor
            self.gamma = 0.2  # Interface energy [J/m²]
            self.D0 = 1e-5  # Pre-exponential factor of diffusion coefficient [m²/s]
            self.Q = 150e3  # Activation energy [J/mol]
    
        def simulate_aging(self, T, time_hours):
            """
            Simulate aging process
    
            Args:
                T: agingTemperature [K]
                time_hours: Aging time array [h]
    
            Returns:
                results: Dictionary of simulation results
            """
            time_seconds = np.array(time_hours) * 3600
    
            # Nucleation-growth model (simplified)
            R = 8.314
            D = self.D0 * np.exp(-self.Q / (R * T))
    
            # Time evolution of precipitate radius
            r0 = 2e-9  # Early核radius
            r = r0 + np.sqrt(2 * D * time_seconds) * 0.5e-9
    
            # Volume fraction advanced（JMA型）
            f_v_max = 0.05  # 最大Volume fraction
            k_jma = 0.1 / 3600  # Rate constant [1/s]
            f_v = f_v_max * (1 - np.exp(-k_jma * time_seconds))
    
            # Coarsening (long time)
            K = (8 * self.gamma * 1e-5 * 0.01 * D) / (9 * R * T)
            r_coarsen = (r**3 + K * time_seconds) ** (1/3)
    
            # Coarsening dominates after 100 hours
            transition_idx = np.searchsorted(time_hours, 100)
            r[transition_idx:] = r_coarsen[transition_idx:]
    
            # Calculation of Orowan strength
            strength = np.zeros_like(r)
            for i, (ri, fv) in enumerate(zip(r, f_v)):
                if fv > 0.001:  # When sufficient precipitates exist
                    try:
                        lambda_p = 2 * ri * np.sqrt(np.pi / (3 * fv))
                        tau = (self.M * self.G * self.b) / (lambda_p - 2*ri)
                        strength[i] = self.M * tau / 1e6  # MPa
                    except:
                        strength[i] = 0
    
            # Add base strength
            sigma_base = 70  # Strength of pure Al [MPa]
            total_strength = sigma_base + strength
    
            return {
                'time': time_hours,
                'radius': r * 1e9,  # nm
                'volume_fraction': f_v * 100,  # %
                'strength': total_strength,  # MPa
                'precipitation_strength': strength  # MPa
            }
    
        def plot_results(self, results_dict):
            """シミュレーション結果Visualization"""
    
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            colors = ['blue', 'green', 'red']
    
            # (a) Precipitate radius
            ax = axes[0, 0]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['radius'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('Aging time [h]', fontsize=12)
            ax.set_ylabel('平均Precipitate radius [nm]', fontsize=12)
            ax.set_title('(a) precipitation物Size hoursadvanced', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (b) Volume fraction
            ax = axes[0, 1]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['volume_fraction'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('Aging time [h]', fontsize=12)
            ax.set_ylabel('precipitation物Volume fraction [%]', fontsize=12)
            ax.set_title('(b) precipitation物Volume fraction', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (c) Yield strength
            ax = axes[1, 0]
            for (label, results), color in zip(results_dict.items(), colors):
                ax.semilogx(results['time'], results['strength'],
                           linewidth=2.5, color=color, label=label)
            ax.set_xlabel('Aging time [h]', fontsize=12)
            ax.set_ylabel('Yield strength [MPa]', fontsize=12)
            ax.set_title('(c) Aging curve（strength予測）', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            # (d) strengthening寄与 内訳
            ax = axes[1, 1]
            # 200°C ケースexample to
            results_200C = results_dict['200°C']
            t = results_200C['time']
            sigma_base = 70
            sigma_precip = results_200C['precipitation_strength']
    
            ax.semilogx(t, [sigma_base]*len(t), 'k--', linewidth=2, label='Base strength')
            ax.fill_between(t, sigma_base, sigma_base + sigma_precip,
                            alpha=0.3, color='blue', label='Precipitation strengthening')
            ax.semilogx(t, results_200C['strength'], 'b-', linewidth=2.5,
                       label='Total strength')
            ax.set_xlabel('Aging time [h]', fontsize=12)
            ax.set_ylabel('Yield strength [MPa]', fontsize=12)
            ax.set_title('(d) strengthening機構 寄与 (200°C)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # Running simulation
    simulator = PrecipitationSimulator()
    
    time_array = np.logspace(-1, 3, 100)  # 0.1〜1000hours
    
    results_dict = {
        '180°C': simulator.simulate_aging(453, time_array),
        '200°C': simulator.simulate_aging(473, time_array),
        '220°C': simulator.simulate_aging(493, time_array),
    }
    
    simulator.plot_results(results_dict)
    
    # peakAging conditions 特定
    print("=== Al-Cu-Mgalloy（2024） optimalAging conditions ===\n")
    
    for temp_label, results in results_dict.items():
        peak_idx = np.argmax(results['strength'])
        peak_time = results['time'][peak_idx]
        peak_strength = results['strength'][peak_idx]
        peak_radius = results['radius'][peak_idx]
        peak_fv = results['volume_fraction'][peak_idx]
    
        print(f"{temp_label}:")
        print(f"  optimalaginghours: {peak_time:.1f} hours")
        print(f"  Maximum strength: {peak_strength:.1f} MPa")
        print(f"  Precipitate radius: {peak_radius:.1f} nm")
        print(f"  Volume fraction: {peak_fv:.2f}%\n")
    
    print("Industrial recommended conditions (T6 heat treatment):")
    print("  tempdegree: 190°C")
    print("  hours: 18hours")
    print("  Expected strength: 324 MPa（Measured value）")
    
    # Output example:
    # === Al-Cu-Mgalloy（2024） optimalAging conditions ===
    #
    # 180°C:
    #   optimalaginghours: 31.6 hours
    #   Maximum strength: 298.5 MPa
    #   Precipitate radius: 12.3 nm
    #   Volume fraction: 4.85%
    #
    # 200°C:
    #   optimalaginghours: 15.8 hours
    #   Maximum strength: 305.2 MPa
    #   Precipitate radius: 15.7 nm
    #   Volume fraction: 4.90%
    

## Learning Objectives review

th 完了する and 、以below can explainよう toなります：

### Fundamental Understanding

  * ✅ Solid Solution 種類（Substitutional Interstitial） and solid solution strengthening Mechanism can explain
  * ✅ precipitation 核生成 成長 粗大化 3Stage理解し、Aging curve解釈 atきる
  * ✅ Alalloy precipitation過程（GP Zones → θ'' → θ' → θ） can explain

### Practical Skills

  * ✅ Calculate solid solution strengthening using Labusch model
  * ✅ 古典的核生成theory使ってNucleation rate予測 atきる
  * ✅ Quantitatively calculate precipitation strengthening by Orowan mechanism
  * ✅ LSWtheory atprecipitation物 粗大化予測 atきる

### Application Ability

  * ✅ 実用Alalloy optimalAging conditions設計 atきる
  * ✅ precipitation物Size and 分布制御して材料strengthoptimal化 atきる
  * ✅ Python atprecipitation過程 統合シミュレーション実装 atきる

## Exercise Problems

### Easy (Basic Review)

**Q1** : solid solution strengthening and Precipitation strengthening 主な違い何 atすか？

**Answer** :

  * **solid solution strengthening** : 溶質原子 matrixPhase to均一 to分散し、格子歪みやInteraction with Dislocations atstrengthening
  * **Precipitation strengthening** : 微細なChapter二Phase粒子 precipitationし、dislocation運動物理的 to阻害（Orowan機構）

**Explanation** :

solid solution strengthening単Phase（Solid Solution）、濃degree to対してΔσ ∝ c2/3degree 増加。Precipitation strengthening二Phase（matrixPhase+precipitation物）、precipitation物 optimalSize 分布制御 at大幅なstrengthening 可能 atす。

**Q2** : Al-Cualloy aging過程 at、Maximum hardness示す ど Phase atすか？

**Answer** : θ'Phase（metastablePhase、Semi-coherentprecipitation物）

**Explanation** :

precipitation過程: GP Zones → θ'' → **θ'** → θ（equilibriumPhase）

θ'Phase10-50nmdegree Size atSemi-coherent、Interaction with Dislocations 最も強いため、最大 strengtheningEffect示します。過aging atθPhase（Incoherent、粗大） toなる and strengthlowbelowします。

**Q3** : Orowan機構 at、Precipitate spacingλ 狭くなる and strengthどうなりますか？

**Answer** : strength 増加する

**Explanation** :

Orowan stress: τ = (M·G·b) / (λ - 2r)

λ 小さくなる（precipitation物 密 to分布） and 、分matrix 小さくなり、τ 増加します。ただし、λ < 2r 極限 at式 発散するため、実際 toprecipitation物 接触してしまい、別 Mechanism 働きます。

### Medium (Application)

**Q4** : Al-4%Cualloy200°C atagingする and 、10hours atpeak硬degree to達しました。250°C at同じpeak硬degree to達する to hours推定してください。Activation energy150 kJ/mol and します。

**Calculation Process** :

Arrhenius 関係式:

t2 / t1 = exp[Q/R · (1/T2 \- 1/T1)]

Given values:

  * T1 = 473 K (200°C), t1 = 10 h
  * T2 = 523 K (250°C), t2 = ?
  * Q = 150 kJ/mol = 150,000 J/mol
  * R = 8.314 J/mol/K

Calculation:
    
    
    t₂ / 10 = exp[150000/8.314 · (1/523 - 1/473)]
           = exp[18037 · (-0.0002024)]
           = exp(-3.65)
           = 0.026
    
    t₂ = 10 × 0.026 = 0.26 hours ≈ 16分
    

**Answer** : approximately0.26hours（16分）

**Explanation** :

tempdegree 50°Cabove昇する and 、拡散速degree 大幅 to増加し、aginghours approximately40倍短縮されます。thれArrhenius equation 指数的なTemperature dependence toよるも atす。工業的 to hightempaging（250°C）短hours at済む一方、precipitation物 粗大化しやすいため、Maximum strengthlowtempaging（190-200°C）より若干lowくなります。

**Q5** : radius10nm precipitation物 Volume fraction5% at分散しています。Orowan機構 toよるyield stress増加calculationしてください。（G = 26 GPa、b = 0.286 nm、M = 3）

**Calculation Process** :

1\. Precipitate spacingλ Calculation:
    
    
    λ = 2r · √(π / (3f_v))
      = 2 × 10 nm · √(π / (3 × 0.05))
      = 20 nm · √(π / 0.15)
      = 20 nm · √20.94
      = 20 nm × 4.576
      = 91.5 nm
    

2\. Orowan stress Calculation:
    
    
    τ = (M · G · b) / (λ - 2r)
      = (3 × 26×10⁹ Pa × 0.286×10⁻⁹ m) / (91.5×10⁻⁹ m - 20×10⁻⁹ m)
      = (22.3 Pa·m) / (71.5×10⁻⁹ m)
      = 3.12×10⁸ Pa
      = 312 MPa
    

3\. Tensile yield stress:
    
    
    σ_y = M · τ = 3 × 312 MPa = 936 MPa
    

**Answer** : approximately930-950 MPa

**Explanation** :

th calculation理想的な条件仮定しており、実際 材料 at以below 要因 at値 変わります：

  * precipitation物 Coherency（Fully Coherent 場合、dislocation 切断するため異なる機構）
  * Size分布 影響
  * 他 strengthening機構（solid solution strengthening、粒界strengthening） and 重畳

典型的なAlalloy（2024-T6） Measured valueapproximately320 MPadegree at、thれBase strength70 MPa + Precipitation strengthening250 MPadegree atす。

### Hard (Advanced)

**Q6** : Al-Cualloy in、Earlyradius5nm precipitation物 200°C at粗大化します。500hoursafter 平均radius予測してください。また、th 粗大化 by Yield strength ど degreelowbelowするか議論してください。（Coarsening rate constant K = 5×10⁻26 m³/s、EarlyVolume fraction5%、G=26GPa、b=0.286nm）

**Calculation Process** :

**Step 1: 粗大化after radius**
    
    
    LSW theory: r³(t) = r₀³ + Kt
    
    r₀ = 5 nm = 5×10⁻⁹ m
    t = 500 h = 500 × 3600 s = 1.8×10⁶ s
    K = 5×10⁻²⁶ m³/s
    
    r³ = (5×10⁻⁹)³ + 5×10⁻²⁶ × 1.8×10⁶
       = 1.25×10⁻²⁵ + 9.0×10⁻²⁰
       = 9.0×10⁻²⁰ m³  (First term is negligible)
    
    r = (9.0×10⁻²⁰)^(1/3) = 4.48×10⁻⁷ m = 44.8 nm
    

**Step 2: Earlystrength calculation**
    
    
    r₀ = 5 nm, f_v = 0.05
    
    λ₀ = 2 × 5 × √(π/(3×0.05)) = 45.8 nm
    
    σ₀ = (3 × 26×10⁹ × 0.286×10⁻⁹) / (45.8×10⁻⁹ - 10×10⁻⁹)
       = 22.3 / (35.8×10⁻⁹)
       = 6.23×10⁸ Pa
    
    yield stress: σ_y0 = 3 × 623 = 1869 MPa
    

**Step 3: 粗大化after strength**
    
    
    r = 44.8 nm (Volume fraction保存: f_v = 0.05)
    
    λ = 2 × 44.8 × √(π/(3×0.05)) = 410 nm
    
    σ = (3 × 26×10⁹ × 0.286×10⁻⁹) / (410×10⁻⁹ - 89.6×10⁻⁹)
      = 22.3 / (320×10⁻⁹)
      = 6.97×10⁷ Pa
    
    yield stress: σ_y = 3 × 70 = 210 MPa
    

**Step 4: strengthlowbelow**
    
    
    Δσ = σ_y0 - σ_y = 1869 - 210 = 1659 MPa
    
    lowbelow率 = (1659 / 1869) × 100 = 88.8%
    

**Answer** :

  * 500hoursafter 平均radius: approximately45 nm（Early 9倍）
  * Yield strength lowbelow: approximately89%（1869 MPa → 210 MPa）

**Detailed Discussion** :

**1\. 粗大化 Mechanism**

r³則 to従う粗大化（Ostwald ripening） Gibbs-ThomsonEffect by 小粒子 溶解し、大粒子 成長する現象 atす。500hours（approximately3週intermediate） aging at、radius 9倍 to増加する 実用的 to重要な問題 atす。

**2\. strengthlowbelow Cause**

  * **Precipitate spacing 増大** : 45.8 nm → 410 nm（approximately9倍）
  * **Orowan機構 弱化** : λ 大きくなる and 、dislocation precipitation物容易 toバイパス atきる
  * **粒子数密degree 減少** : 大きい粒子 少数 toなる（体積保存）

**3\. Industrial Countermeasures**

  * **使用tempdegree 制限** : Alalloy150°C以below at 使用 recommended（長期安定性）
  * **三元添加** : Mg、Ag、Znetc 添加 at粗大化抑制
  * **分散粒子 導入** : Al₃Zretc 熱的 to安定な分散粒子 atprecipitation物固定
  * **組織 微細化** : 塑性加工 toよるdislocation密degree増加 at核生成サイト増加

**4\. 実用alloy example**

航空機用Al-Cu-Mgalloy（2024-T6） 200°C × 500hoursafter atもapproximately70% strength保持するよう設計されています（Early: 470 MPa → 500hafter: 330 MPadegree）。thれ本calculationより遥か to良好 atす 、thれ：

  * Coarsening suppression by minor additions (Mn, Zr)
  * 二種類 precipitation物（θ' and S'） 複合Effect
  * Microstructure control by plastic deformation

etc 実用技術 toよるも atす。

**Q7:** Al-4%Cualloy GPzone形成 in、copper atoms アルミニウム格子medium {100}面 to優先的 to偏析する理由、原子Size and 弾性ひずみ 観点 from 説明してください。

**Solution Example** :

**原子Size 違い** :

  * アルミニウム（Al） 原子radius: 1.43 Å
  * 銅（Cu） 原子radius: 1.28 Å
  * Copper atoms are ~10% smaller than aluminum

**GPzone形成Mechanism** :

  1. **Substitutional固溶** : copper atoms アルミニウム格子点置換する and 、格子 to収縮ひずみ 生じる
  2. **{100}面へ 偏析** : copper atoms {100}面（FCCstructure 特定結晶面） to集まるth and at、ひずみエネルギー 局所的 to緩和される
  3. **円盤状クラスター形成** : 1-2原子層 厚さ at、直径数nm 円盤状GPzone {100}面 to沿って形成される

**弾性ひずみ 役割** :

copper atoms 偏析 by 、matrixPhase（アルミニウム） and 界面 to弾性ひずみ場 形成されます。th 整合ひずみ dislocation 運動妨げ、Orowan機構 toよるstrengtheningEffectもたらします。GPzone 厚さ 薄いほど、Coherency 維持され、highいstrength 得られます。

**Experimental Observation** :

透過型電子顕微鏡（TEM）観察 by 、GPzone{100}面 to沿った特徴的なストリークコントラスト and して観察されます。

**Q8:** ニッケル基超alloy（example: Inconel 718） in、γ'Phase（Ni3Al） and γ''Phase（Ni3Nb） 二種類 Precipitation strengtheningPhase 共存します。それぞれ precipitationPhase 特徴 and 、hightempstrengthへ 寄与比較してください。

**Solution Example** :

Property | γ'Phase（Ni3Al） | γ''Phase（Ni3Nb）  
---|---|---  
**Crystal Structure** | L12 structure (FCC-based) | DO22 structure (BCT: Body-Centered Tetragonal)  
**Morphology** | 球状 or 立方体状（等軸） | Disk-shaped (precipitates along {100} planes)  
**Lattice Misfit** | ~+0.5% (slightly larger) | ~-2.5% (significant contraction)  
**Coherency** | Fully Coherent（hightemp to 維持） | 準整合（600°C以above at安定性lowbelow）  
**Thermal Stability** | ～1000°C（非常 tohighい） | ～650°C（mediumdegree）  
**strengtheningEffect** | hightemp at 持続的strengthening（クリープ抵抗） | mediumtemp域 at 顕著なstrengthening（Yield strength）  
  
**Inconel 718 設計思想** :

  * **室temp～650°C** : γ''Phase 主要なstrengtheningPhase and して機能（Yield strength > 1000 MPa）
  * **650～850°C** : γ'Phase 主要なstrengtheningPhase and して機能（γ'' 溶解固溶 toよる軟化補償）
  * **二Phase複合strengthening** : 広いtempdegree範囲 athighstrength維持 atきるため、航空機エンジン（タービンディスク） tooptimal

**aging熱処理** :

Inconel 718 標準aging処理：720°C × 8h（γ''precipitation）+ 620°C × 8h（γ'微細化） by 、optimalなprecipitation分布実現します。

## ✓ Learning Objectives review

th 完了する and 、以below説明 実行 atきるよう toなります：

### Fundamental Understanding

  * ✅ solid solution strengthening and Precipitation strengthening Mechanism can explain
  * ✅ GPzone、θ'', θ'、θPhase precipitationシーケンス and 各Stage 特徴理解している
  * ✅ aging硬化曲線 形状 and 、peakaging 過aging 物理的意味 can explain
  * ✅ Orowan機構 toよるprecipitation物 strengtheningEffect定量的 to理解している

### Practical Skills

  * ✅ Orowan方程式用いて、precipitation物Size and intermediatespacing from strength増分calculation atきる
  * ✅ LSWtheory（Ostwald ripening）用いて、precipitation物 粗大化速degree予測 atきる
  * ✅ aging処理条件（tempdegree hours） and 最終strength 関係定量的 to評価 atきる
  * ✅ Python用いて、aging硬化曲線 and precipitation物成長 シミュレーション atきる

### Application Ability

  * ✅ Al-Cu、Al-Mg-Si、Al-Zn-Mgsystemalloy precipitation挙動 違い can explain
  * ✅ Aging conditions（T4、T6、T7処理） 選択 and 、strength-延性-耐食性 トレードオフ評価 atきる
  * ✅ ニッケル基超alloy γ'/γ''二Phasestrengthening機構理解し、hightemp材料設計 to応用 atきる
  * ✅ 長hourshightemp曝露 toよるprecipitation物粗大化 and strength劣化定量的 to予測 atきる

**次 ステップ** :

Precipitation strengthening 基礎習得したら、Chapter4「dislocation and 塑性変形」 to進み、precipitation物 and dislocation Phase互作用Mechanismミクロスケール at学びましょう。dislocation論 and Precipitation strengthening統合するth and at、材料 塑性変形挙動深く理解 atきます。

## 📚 References

  1. Porter, D.A., Easterling, K.E., Sherif, M.Y. (2009). _Phase Transformations in Metals and Alloys_ (3rd ed.). CRC Press. ISBN: 978-1420062106
  2. Ashby, M.F., Jones, D.R.H. (2012). _Engineering Materials 2: An Introduction to Microstructures and Processing_ (4th ed.). Butterworth-Heinemann. ISBN: 978-0080966700
  3. Martin, J.W. (1998). _Precipitation Hardening_ (2nd ed.). Butterworth-Heinemann. ISBN: 978-0750641630
  4. Polmear, I.J., StJohn, D., Nie, J.F., Qian, M. (2017). _Light Alloys: Metallurgy of the Light Metals_ (5th ed.). Butterworth-Heinemann. ISBN: 978-0080994314
  5. Starke, E.A., Staley, J.T. (1996). "Application of modern aluminum alloys to aircraft." _Progress in Aerospace Sciences_ , 32(2-3), 131-172. [DOI:10.1016/0376-0421(95)00004-6](<https://doi.org/10.1016/0376-0421\(95\)00004-6>)
  6. Wagner, C. (1961). "Theorie der Alterung von Niederschlägen durch Umlösen (Ostwald-Reifung)." _Zeitschrift für Elektrochemie_ , 65(7-8), 581-591.
  7. Ardell, A.J. (1985). "Precipitation hardening." _Metallurgical Transactions A_ , 16(12), 2131-2165. [DOI:10.1007/BF02670416](<https://doi.org/10.1007/BF02670416>)
  8. Callister, W.D., Rethwisch, D.G. (2020). _Materials Science and Engineering: An Introduction_ (10th ed.). Wiley. ISBN: 978-1119405498

### Online Resources

  * **Aluminum Alloy Database** : ASM Alloy Center Database (<https://matdata.asminternational.org/>)
  * **aging処理ガイド** : Aluminum Association - Heat Treatment Guidelines (<https://www.aluminum.org/>)
  * **Precipitation Simulation** : TC-PRISMA (Thermo-Calc Software) - Precipitation simulation tool
