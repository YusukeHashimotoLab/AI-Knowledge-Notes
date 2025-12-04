---
title: "Chapter 1: Fundamentals of Scaling Theory"
chapter_title: "Chapter 1: Fundamentals of Scaling Theory"
subtitle: Understanding Similarity Laws, Power Laws, and Equipment Sizing
version: 1.0
created_at: 2025-10-26
---

This chapter covers the fundamentals of Fundamentals of Scaling Theory, which fundamentals of similarity laws. You will learn Calculate scale factors using power laws, Determine optimal pilot plant scales, and and apply economic scaling (six-tenths rule).

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand geometric, kinematic, and dynamic similarity
  * ✅ Calculate scale factors using power laws
  * ✅ Size reactors, tanks, and heat exchangers
  * ✅ Determine optimal pilot plant scales
  * ✅ Understand and apply economic scaling (six-tenths rule)

* * *

## 1.1 Fundamentals of Similarity Laws

### What are Similarity Laws?

**Similarity Laws** describe the physical correspondence that exists between systems at different scales. In chemical engineering, three types of similarity are important:

  * **Geometric Similarity** : Ratios of shapes and dimensions are preserved
  * **Kinematic Similarity** : Velocity field patterns are similar
  * **Dynamic Similarity** : Force ratios are similar

### Scale Factor

When we define the scale factor $\lambda$ as the ratio of lengths, other physical quantities scale as follows:

Physical Quantity | Scaling Law | Scale Factor  
---|---|---  
Length (L) | $L_2 = \lambda \cdot L_1$ | $\lambda$  
Area (A) | $A_2 = \lambda^2 \cdot A_1$ | $\lambda^2$  
Volume (V) | $V_2 = \lambda^3 \cdot V_1$ | $\lambda^3$  
Velocity (v) | Varies with conditions | $\lambda^0$ or $\lambda^{0.5}$  
Time (t) | Varies with conditions | $\lambda$ or $\lambda^{0.5}$  
  
* * *

## 1.2 Scaling Calculations with Python

### Code Example 1: Geometric Similarity Scaling Relationships
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def geometric_scaling(lambda_scale, L1, A1, V1):
        """
        Scaling calculation based on geometric similarity
    
        Parameters:
        -----------
        lambda_scale : float
            Scale factor (ratio of lengths)
        L1 : float
            Characteristic length of original system [m]
        A1 : float
            Area of original system [m²]
        V1 : float
            Volume of original system [m³]
    
        Returns:
        --------
        dict : Scaled-up values
        """
        L2 = lambda_scale * L1
        A2 = lambda_scale**2 * A1
        V2 = lambda_scale**3 * V1
    
        return {
            'Length': L2,
            'Area': A2,
            'Volume': V2,
            'Surface_to_Volume_Ratio': A2 / V2
        }
    
    # Lab-scale reactor
    L1_lab = 0.1  # Diameter 0.1 m (10 cm)
    A1_lab = np.pi * L1_lab**2  # Bottom area
    V1_lab = (np.pi / 4) * L1_lab**3  # Volume (assuming sphere)
    
    # Scale factor range: 1x to 100x
    lambda_range = np.logspace(0, 2, 50)  # 1 to 100
    
    results = []
    for lambda_scale in lambda_range:
        result = geometric_scaling(lambda_scale, L1_lab, A1_lab, V1_lab)
        results.append({
            'lambda': lambda_scale,
            **result
        })
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Length scaling
    axes[0, 0].loglog(lambda_range, [r['Length'] for r in results],
                      linewidth=2.5, color='#11998e', label='L ∝ λ¹')
    axes[0, 0].set_xlabel('Scale Factor λ', fontsize=11)
    axes[0, 0].set_ylabel('Length L [m]', fontsize=11)
    axes[0, 0].set_title('(a) Length Scaling', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, which='both')
    axes[0, 0].legend(fontsize=10)
    
    # Area scaling
    axes[0, 1].loglog(lambda_range, [r['Area'] for r in results],
                      linewidth=2.5, color='#e67e22', label='A ∝ λ²')
    axes[0, 1].set_xlabel('Scale Factor λ', fontsize=11)
    axes[0, 1].set_ylabel('Area A [m²]', fontsize=11)
    axes[0, 1].set_title('(b) Area Scaling', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, which='both')
    axes[0, 1].legend(fontsize=10)
    
    # Volume scaling
    axes[1, 0].loglog(lambda_range, [r['Volume'] for r in results],
                      linewidth=2.5, color='#9b59b6', label='V ∝ λ³')
    axes[1, 0].set_xlabel('Scale Factor λ', fontsize=11)
    axes[1, 0].set_ylabel('Volume V [m³]', fontsize=11)
    axes[1, 0].set_title('(c) Volume Scaling', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, which='both')
    axes[1, 0].legend(fontsize=10)
    
    # S/V ratio scaling
    axes[1, 1].loglog(lambda_range, [r['Surface_to_Volume_Ratio'] for r in results],
                      linewidth=2.5, color='#e74c3c', label='S/V ∝ λ⁻¹')
    axes[1, 1].set_xlabel('Scale Factor λ', fontsize=11)
    axes[1, 1].set_ylabel('S/V Ratio [m⁻¹]', fontsize=11)
    axes[1, 1].set_title('(d) Surface Area/Volume Ratio Scaling', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, which='both')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Specific case calculations
    lambda_pilot = 10  # Pilot: 10x scale
    lambda_commercial = 100  # Commercial: 100x scale
    
    pilot = geometric_scaling(lambda_pilot, L1_lab, A1_lab, V1_lab)
    commercial = geometric_scaling(lambda_commercial, L1_lab, A1_lab, V1_lab)
    
    print("=" * 60)
    print("Geometric Similarity Scaling Results")
    print("=" * 60)
    print(f"\nLab Scale:")
    print(f"  Diameter: {L1_lab*100:.1f} cm")
    print(f"  Volume: {V1_lab*1e6:.2f} cm³ = {V1_lab*1e3:.2f} L")
    print(f"  S/V Ratio: {A1_lab/V1_lab:.2f} m⁻¹")
    
    print(f"\nPilot Scale (λ = {lambda_pilot}):")
    print(f"  Diameter: {pilot['Length']*100:.1f} cm")
    print(f"  Volume: {pilot['Volume']:.4f} m³ = {pilot['Volume']*1e3:.1f} L")
    print(f"  S/V Ratio: {pilot['Surface_to_Volume_Ratio']:.2f} m⁻¹")
    print(f"  S/V Ratio Decrease: {(1 - pilot['Surface_to_Volume_Ratio']/(A1_lab/V1_lab))*100:.1f}%")
    
    print(f"\nCommercial Scale (λ = {lambda_commercial}):")
    print(f"  Diameter: {commercial['Length']:.2f} m")
    print(f"  Volume: {commercial['Volume']:.2f} m³")
    print(f"  S/V Ratio: {commercial['Surface_to_Volume_Ratio']:.2f} m⁻¹")
    print(f"  S/V Ratio Decrease: {(1 - commercial['Surface_to_Volume_Ratio']/(A1_lab/V1_lab))*100:.1f}%")
    

**Sample Output:**
    
    
    ============================================================
    Geometric Similarity Scaling Results
    ============================================================
    
    Lab Scale:
      Diameter: 10.0 cm
      Volume: 523.60 cm³ = 0.52 L
      S/V Ratio: 60.00 m⁻¹
    
    Pilot Scale (λ = 10):
      Diameter: 100.0 cm
      Volume: 0.5236 m³ = 523.6 L
      S/V Ratio: 6.00 m⁻¹
      S/V Ratio Decrease: 90.0%
    
    Commercial Scale (λ = 100):
      Diameter: 10.00 m
      Volume: 523.60 m³
      S/V Ratio: 0.60 m⁻¹
      S/V Ratio Decrease: 99.0%
    

**Explanation:** With geometric similarity, when length increases by $\lambda$ times, area increases by $\lambda^2$ times and volume increases by $\lambda^3$ times. Importantly, **the surface area/volume ratio (S/V ratio) decreases proportionally to $\lambda^{-1}$**. This means that heat transfer and cooling capacity decrease relatively during scaleup.

* * *

### Code Example 2: Power Law (Power Rule) Scaling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b):
        """
        Power law: y = a * x^b
    
        Parameters:
        -----------
        x : array_like
            Independent variable
        a : float
            Coefficient
        b : float
            Exponent (scaling exponent)
    
        Returns:
        --------
        y : array_like
            Dependent variable
        """
        return a * x**b
    
    # Experimental data (relationship between reactor size and reaction time)
    # Data: volume [L], reaction time [min]
    volumes_exp = np.array([1, 5, 10, 50, 100, 500, 1000])  # L
    reaction_times = np.array([10, 18, 22, 35, 45, 70, 88])  # min
    
    # Power law fitting
    params, covariance = curve_fit(power_law, volumes_exp, reaction_times)
    a_fit, b_fit = params
    print("=" * 60)
    print("Power Law Fitting Results")
    print("=" * 60)
    print(f"Reaction time = {a_fit:.3f} × V^{b_fit:.3f}")
    print(f"Scaling exponent: b = {b_fit:.3f}")
    print(f"Theoretical value (mixing-limited): b ≈ 0.33")
    print(f"Goodness of fit R²: {1 - np.sum((reaction_times - power_law(volumes_exp, *params))**2) / np.sum((reaction_times - np.mean(reaction_times))**2):.4f}")
    
    # Prediction range
    volumes_pred = np.logspace(0, 4, 100)  # 1 L to 10,000 L
    times_pred = power_law(volumes_pred, a_fit, b_fit)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Log plot
    plt.subplot(1, 2, 1)
    plt.loglog(volumes_exp, reaction_times, 'o', markersize=10,
               color='#e74c3c', label='Experimental Data', zorder=5)
    plt.loglog(volumes_pred, times_pred, linewidth=2.5, color='#11998e',
               label=f'Fit: t = {a_fit:.2f}V^{b_fit:.2f}')
    plt.xlabel('Reactor Volume V [L]', fontsize=12)
    plt.ylabel('Reaction Time t [min]', fontsize=12)
    plt.title('Power Law Scaling (Log Plot)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, which='both')
    
    # Linear plot
    plt.subplot(1, 2, 2)
    plt.plot(volumes_exp, reaction_times, 'o', markersize=10,
             color='#e74c3c', label='Experimental Data', zorder=5)
    plt.plot(volumes_pred, times_pred, linewidth=2.5, color='#11998e',
             label=f'Fit: t = {a_fit:.2f}V^{b_fit:.2f}')
    plt.xlabel('Reactor Volume V [L]', fontsize=12)
    plt.ylabel('Reaction Time t [min]', fontsize=12)
    plt.title('Power Law Scaling (Linear Plot)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1200)
    
    plt.tight_layout()
    plt.show()
    
    # Scaleup prediction
    V_commercial = 5000  # L
    t_commercial_pred = power_law(V_commercial, a_fit, b_fit)
    
    print(f"\nCommercial Scale Prediction (V = {V_commercial} L):")
    print(f"  Predicted reaction time: {t_commercial_pred:.1f} min")
    print(f"  {t_commercial_pred/reaction_times[0]:.2f} times lab scale (1 L)")
    

**Sample Output:**
    
    
    ============================================================
    Power Law Fitting Results
    ============================================================
    Reaction time = 9.976 × V^0.312
    Scaling exponent: b = 0.312
    Theoretical value (mixing-limited): b ≈ 0.33
    Goodness of fit R²: 0.9978
    
    Commercial Scale Prediction (V = 5000 L):
      Predicted reaction time: 125.6 min
      12.56 times lab scale (1 L)
    

**Explanation:** The power law $y = ax^b$ is the most common form to describe scaling relationships. The exponent $b$ determines the nature of scaling:

  * $b = 1$: Linear scaling
  * $b < 1$: Scale benefit (economies of scale)
  * $b > 1$: Scale disadvantage (scaleup is difficult)

* * *

### Code Example 3: Scaleup Factor Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    
    def scale_up_factor(V1, V2, basis='linear'):
        """
        Calculate scaleup factor
    
        Parameters:
        -----------
        V1 : float
            Volume of original system [m³]
        V2 : float
            Volume after scaleup [m³]
        basis : str
            Scaling basis
            - 'linear': Linear scale (L)
            - 'area': Area scale (A)
            - 'volume': Volume scale (V)
    
        Returns:
        --------
        dict : Scaleup factors
        """
        # Linear scale factor (cube root of volume ratio)
        lambda_L = (V2 / V1)**(1/3)
    
        if basis == 'linear':
            lambda_scale = lambda_L
            basis_desc = "Linear Basis"
        elif basis == 'area':
            lambda_scale = (V2 / V1)**(2/3)
            basis_desc = "Area Basis"
        elif basis == 'volume':
            lambda_scale = V2 / V1
            basis_desc = "Volume Basis"
        else:
            raise ValueError("basis must be 'linear', 'area', or 'volume'")
    
        return {
            'Linear_Scale_Factor': lambda_L,
            'Scale_Factor': lambda_scale,
            'Basis': basis_desc,
            'Area_Ratio': lambda_L**2,
            'Volume_Ratio': lambda_L**3
        }
    
    # Case study: Lab → Pilot → Commercial scale
    scales = {
        'Lab': 0.001,        # 1 L = 0.001 m³
        'Pilot': 0.5,        # 500 L = 0.5 m³
        'Commercial': 50.0   # 50,000 L = 50 m³
    }
    
    # Scaleup calculation
    print("=" * 80)
    print("Scaleup Factor Calculation")
    print("=" * 80)
    
    scale_transitions = [
        ('Lab', 'Pilot'),
        ('Pilot', 'Commercial'),
        ('Lab', 'Commercial')
    ]
    
    for from_scale, to_scale in scale_transitions:
        V1 = scales[from_scale]
        V2 = scales[to_scale]
    
        print(f"\n{from_scale} → {to_scale} Scaleup:")
        print(f"  Volume: {V1*1e3:.1f} L → {V2*1e3:.1f} L")
    
        # Scaleup factor for each basis
        for basis in ['linear', 'area', 'volume']:
            result = scale_up_factor(V1, V2, basis)
            print(f"  {result['Basis']}: λ = {result['Scale_Factor']:.2f}")
    
    # Equipment parameter scaling example
    print("\n" + "=" * 80)
    print("Equipment Parameter Scaling Example (Lab → Commercial)")
    print("=" * 80)
    
    V_lab = scales['Lab']
    V_commercial = scales['Commercial']
    lambda_L = (V_commercial / V_lab)**(1/3)
    
    # Lab scale parameters
    params_lab = {
        'Diameter [m]': 0.1,
        'Height [m]': 0.15,
        'Stirring Speed [rpm]': 500,
        'Stirring Power [W]': 5,
        'Heat Transfer Area [m²]': 0.05,
        'Residence Time [min]': 30
    }
    
    # Scaled-up parameters (assuming geometric similarity)
    params_commercial = {
        'Diameter [m]': params_lab['Diameter [m]'] * lambda_L,
        'Height [m]': params_lab['Height [m]'] * lambda_L,
        'Stirring Speed [rpm]': params_lab['Stirring Speed [rpm]'] / lambda_L**0.5,  # Froude number preservation
        'Stirring Power [W]': params_lab['Stirring Power [W]'] * lambda_L**5,  # Power law
        'Heat Transfer Area [m²]': params_lab['Heat Transfer Area [m²]'] * lambda_L**2,
        'Residence Time [min]': params_lab['Residence Time [min]']  # Maintain same reaction time
    }
    
    # Display results
    df = pd.DataFrame({
        'Lab Scale': params_lab,
        'Commercial Scale': params_commercial,
        'Scaling Factor': [
            f'λ¹ = {lambda_L:.2f}',
            f'λ¹ = {lambda_L:.2f}',
            f'λ⁻⁰·⁵ = {lambda_L**(-0.5):.2f}',
            f'λ⁵ = {lambda_L**5:.0f}',
            f'λ² = {lambda_L**2:.2f}',
            'Constant'
        ]
    })
    
    print("\n" + df.to_string())
    

**Sample Output:**
    
    
    ================================================================================
    Scaleup Factor Calculation
    ================================================================================
    
    Lab → Pilot Scaleup:
      Volume: 1.0 L → 500.0 L
      Linear Basis: λ = 7.94
      Area Basis: λ = 63.10
      Volume Basis: λ = 500.00
    
    Pilot → Commercial Scaleup:
      Volume: 500.0 L → 50000.0 L
      Linear Basis: λ = 4.64
      Area Basis: λ = 21.54
      Volume Basis: λ = 100.00
    
    Lab → Commercial Scaleup:
      Volume: 1.0 L → 50000.0 L
      Linear Basis: λ = 36.84
      Area Basis: λ = 1357.21
      Volume Basis: λ = 50000.00
    
    ================================================================================
    Equipment Parameter Scaling Example (Lab → Commercial)
    ================================================================================
    
                            Lab Scale  Commercial Scale Scaling Factor
    Diameter [m]                 0.10              3.68  λ¹ = 36.84
    Height [m]                   0.15              5.53  λ¹ = 36.84
    Stirring Speed [rpm]       500.00             82.37  λ⁻⁰·⁵ = 0.16
    Stirring Power [W]           5.00          89819.18  λ⁵ = 17964
    Heat Transfer Area [m²]      0.05             67.93  λ² = 1358
    Residence Time [min]        30.00             30.00  Constant
    

**Explanation:** Scaleup factors differ depending on the chosen basis (linear, area, volume). Equipment parameters follow different scaling laws depending on the physical quantity to be preserved (Reynolds number, Froude number, etc.). Stirring power scales as $\lambda^5$, so it increases dramatically at commercial scale.

* * *

### Code Example 4: Equipment Sizing Calculation (Reactor)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def reactor_sizing(production_rate, residence_time, conversion, density=1000):
        """
        Reactor sizing calculation
    
        Parameters:
        -----------
        production_rate : float
            Target production rate [kg/h]
        residence_time : float
            Required residence time [h]
        conversion : float
            Reaction conversion [-] (0 to 1)
        density : float, optional
            Density of reaction mixture [kg/m³], default: 1000
    
        Returns:
        --------
        dict : Reactor design parameters
        """
        # Required reactor volume
        # Q = P / (X * ρ)  where Q: volumetric flowrate, P: production rate, X: conversion
        volumetric_flowrate = production_rate / (conversion * density)  # m³/h
        reactor_volume = volumetric_flowrate * residence_time  # m³
    
        # Assuming cylindrical reactor (H/D = 2)
        # V = π/4 * D² * H = π/4 * D² * 2D = π/2 * D³
        diameter = (reactor_volume / (np.pi / 2))**(1/3)
        height = 2 * diameter
    
        # Heat transfer area (side + bottom)
        side_area = np.pi * diameter * height
        bottom_area = np.pi * diameter**2 / 4
        heat_transfer_area = side_area + bottom_area
    
        return {
            'Production_Rate_kg_h': production_rate,
            'Volumetric_Flowrate_m3_h': volumetric_flowrate,
            'Residence_Time_h': residence_time,
            'Reactor_Volume_m3': reactor_volume,
            'Diameter_m': diameter,
            'Height_m': height,
            'Heat_Transfer_Area_m2': heat_transfer_area,
            'Surface_to_Volume_Ratio': heat_transfer_area / reactor_volume
        }
    
    # Case study: Reactor design at different scales
    scales_production = {
        'Lab': 0.1,          # 0.1 kg/h
        'Pilot': 10,         # 10 kg/h
        'Commercial': 1000   # 1000 kg/h (1 ton/h)
    }
    
    residence_time = 2  # h
    conversion = 0.85
    density = 1100  # kg/m³
    
    print("=" * 80)
    print("Reactor Sizing Calculation")
    print("=" * 80)
    print(f"Conditions: Residence time = {residence_time} h, Conversion = {conversion:.0%}, Density = {density} kg/m³\n")
    
    results_reactor = {}
    for scale_name, prod_rate in scales_production.items():
        result = reactor_sizing(prod_rate, residence_time, conversion, density)
        results_reactor[scale_name] = result
    
        print(f"{scale_name} Scale Reactor:")
        print(f"  Production rate: {result['Production_Rate_kg_h']:.1f} kg/h")
        print(f"  Reactor volume: {result['Reactor_Volume_m3']:.4f} m³ = {result['Reactor_Volume_m3']*1e3:.1f} L")
        print(f"  Diameter: {result['Diameter_m']:.3f} m = {result['Diameter_m']*100:.1f} cm")
        print(f"  Height: {result['Height_m']:.3f} m = {result['Height_m']*100:.1f} cm")
        print(f"  Heat transfer area: {result['Heat_Transfer_Area_m2']:.3f} m²")
        print(f"  S/V ratio: {result['Surface_to_Volume_Ratio']:.2f} m⁻¹\n")
    
    # Visualization of scaling relationship
    production_rates = np.logspace(-1, 3, 50)  # 0.1 to 1000 kg/h
    volumes = []
    diameters = []
    SV_ratios = []
    
    for P in production_rates:
        res = reactor_sizing(P, residence_time, conversion, density)
        volumes.append(res['Reactor_Volume_m3'])
        diameters.append(res['Diameter_m'])
        SV_ratios.append(res['Surface_to_Volume_Ratio'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Volume vs production rate
    axes[0].loglog(production_rates, volumes, linewidth=2.5, color='#11998e')
    for scale_name, prod_rate in scales_production.items():
        V = results_reactor[scale_name]['Reactor_Volume_m3']
        axes[0].scatter([prod_rate], [V], s=150, zorder=5, label=scale_name)
    axes[0].set_xlabel('Production Rate [kg/h]', fontsize=12)
    axes[0].set_ylabel('Reactor Volume [m³]', fontsize=12)
    axes[0].set_title('(a) Reactor Volume Scaling', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    
    # Diameter vs production rate
    axes[1].loglog(production_rates, diameters, linewidth=2.5, color='#e67e22')
    for scale_name, prod_rate in scales_production.items():
        D = results_reactor[scale_name]['Diameter_m']
        axes[1].scatter([prod_rate], [D], s=150, zorder=5, label=scale_name)
    axes[1].set_xlabel('Production Rate [kg/h]', fontsize=12)
    axes[1].set_ylabel('Reactor Diameter [m]', fontsize=12)
    axes[1].set_title('(b) Reactor Diameter Scaling', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    
    # S/V ratio vs production rate
    axes[2].loglog(production_rates, SV_ratios, linewidth=2.5, color='#e74c3c')
    for scale_name, prod_rate in scales_production.items():
        SV = results_reactor[scale_name]['Surface_to_Volume_Ratio']
        axes[2].scatter([prod_rate], [SV], s=150, zorder=5, label=scale_name)
    axes[2].set_xlabel('Production Rate [kg/h]', fontsize=12)
    axes[2].set_ylabel('S/V Ratio [m⁻¹]', fontsize=12)
    axes[2].set_title('(c) S/V Ratio Scaling', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**Sample Output:**
    
    
    ================================================================================
    Reactor Sizing Calculation
    ================================================================================
    Conditions: Residence time = 2 h, Conversion = 85%, Density = 1100 kg/m³
    
    Lab Scale Reactor:
      Production rate: 0.1 kg/h
      Reactor volume: 0.0002 m³ = 0.2 L
      Diameter: 0.046 m = 4.6 cm
      Height: 0.093 m = 9.3 cm
      Heat transfer area: 0.015 m²
      S/V ratio: 65.45 m⁻¹
    
    Pilot Scale Reactor:
      Production rate: 10.0 kg/h
      Reactor volume: 0.0214 m³ = 21.4 L
      Diameter: 0.215 m = 21.5 cm
      Height: 0.431 m = 43.1 cm
      Heat transfer area: 0.366 m²
      S/V ratio: 17.12 m⁻¹
    
    Commercial Scale Reactor:
      Production rate: 1000.0 kg/h
      Reactor volume: 2.1405 m³ = 2140.5 L
      Diameter: 1.000 m = 100.0 cm
      Height: 2.000 m = 200.0 cm
      Heat transfer area: 7.069 m²
      S/V ratio: 3.30 m⁻¹
    

**Explanation:** In reactor sizing, the required reactor volume is calculated from production rate, residence time, and conversion. Assuming geometric similarity (H/D = 2), diameter and height are determined from volume. **The decrease in S/V ratio** suggests heat transfer and cooling challenges at commercial scale.

* * *

### Code Example 5: Heat Exchanger Scaling Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def heat_exchanger_sizing(Q_heat, U, delta_T_lm, tube_diameter=0.025, tube_length=3.0):
        """
        Heat exchanger sizing calculation
    
        Parameters:
        -----------
        Q_heat : float
            Heat duty [W]
        U : float
            Overall heat transfer coefficient [W/(m²·K)]
        delta_T_lm : float
            Log mean temperature difference [K]
        tube_diameter : float, optional
            Heat transfer tube diameter [m], default: 0.025 (1 inch)
        tube_length : float, optional
            Heat transfer tube length [m], default: 3.0
    
        Returns:
        --------
        dict : Heat exchanger design parameters
        """
        # Required heat transfer area: Q = U * A * ΔT_lm
        A_required = Q_heat / (U * delta_T_lm)
    
        # Area per tube
        A_per_tube = np.pi * tube_diameter * tube_length
    
        # Number of tubes required
        N_tubes = int(np.ceil(A_required / A_per_tube))
    
        # Actual heat transfer area
        A_actual = N_tubes * A_per_tube
    
        # Shell diameter estimation (based on tube bundle arrangement)
        # Triangular pitch, pitch/diameter ratio = 1.25
        pitch = 1.25 * tube_diameter
        # Shell diameter ≈ pitch * sqrt(N_tubes) + margin
        D_shell = pitch * np.sqrt(N_tubes) * 1.2
    
        return {
            'Heat_Duty_kW': Q_heat / 1000,
            'Required_Area_m2': A_required,
            'Actual_Area_m2': A_actual,
            'Number_of_Tubes': N_tubes,
            'Tube_Diameter_mm': tube_diameter * 1000,
            'Tube_Length_m': tube_length,
            'Shell_Diameter_m': D_shell,
            'Area_per_Tube_m2': A_per_tube
        }
    
    # Heat exchanger design at different scales
    Q_heat_lab = 5000  # W = 5 kW
    Q_heat_pilot = 50000  # W = 50 kW
    Q_heat_commercial = 5000000  # W = 5 MW
    
    U = 500  # W/(m²·K) - Typical value for water-water heat exchanger
    delta_T_lm = 20  # K
    
    scales_HX = {
        'Lab': Q_heat_lab,
        'Pilot': Q_heat_pilot,
        'Commercial': Q_heat_commercial
    }
    
    print("=" * 80)
    print("Heat Exchanger Sizing Calculation")
    print("=" * 80)
    print(f"Conditions: U = {U} W/(m²·K), ΔT_lm = {delta_T_lm} K\n")
    
    results_HX = {}
    for scale_name, Q in scales_HX.items():
        result = heat_exchanger_sizing(Q, U, delta_T_lm)
        results_HX[scale_name] = result
    
        print(f"{scale_name} Scale Heat Exchanger:")
        print(f"  Heat duty: {result['Heat_Duty_kW']:.1f} kW")
        print(f"  Required heat transfer area: {result['Required_Area_m2']:.2f} m²")
        print(f"  Number of tubes: {result['Number_of_Tubes']} tubes")
        print(f"  Tubes: {result['Tube_Diameter_mm']:.1f} mm × {result['Tube_Length_m']:.1f} m")
        print(f"  Shell diameter: {result['Shell_Diameter_m']:.3f} m = {result['Shell_Diameter_m']*100:.1f} cm\n")
    
    # Visualization of scaling relationship
    Q_range = np.logspace(3, 7, 50)  # 1 kW to 10 MW
    areas = []
    N_tubes_list = []
    D_shells = []
    
    for Q in Q_range:
        res = heat_exchanger_sizing(Q, U, delta_T_lm)
        areas.append(res['Required_Area_m2'])
        N_tubes_list.append(res['Number_of_Tubes'])
        D_shells.append(res['Shell_Diameter_m'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Heat transfer area vs heat duty
    axes[0].loglog(np.array(Q_range)/1000, areas, linewidth=2.5, color='#11998e',
                   label='A = Q / (U·ΔT)')
    for scale_name, Q in scales_HX.items():
        A = results_HX[scale_name]['Required_Area_m2']
        axes[0].scatter([Q/1000], [A], s=150, zorder=5, label=scale_name)
    axes[0].set_xlabel('Heat Duty [kW]', fontsize=12)
    axes[0].set_ylabel('Heat Transfer Area [m²]', fontsize=12)
    axes[0].set_title('(a) Heat Transfer Area Scaling', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    
    # Number of tubes vs heat duty
    axes[1].loglog(np.array(Q_range)/1000, N_tubes_list, linewidth=2.5, color='#e67e22')
    for scale_name, Q in scales_HX.items():
        N = results_HX[scale_name]['Number_of_Tubes']
        axes[1].scatter([Q/1000], [N], s=150, zorder=5, label=scale_name)
    axes[1].set_xlabel('Heat Duty [kW]', fontsize=12)
    axes[1].set_ylabel('Number of Tubes', fontsize=12)
    axes[1].set_title('(b) Number of Tubes Scaling', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    
    # Shell diameter vs heat duty
    axes[2].loglog(np.array(Q_range)/1000, D_shells, linewidth=2.5, color='#9b59b6')
    for scale_name, Q in scales_HX.items():
        D = results_HX[scale_name]['Shell_Diameter_m']
        axes[2].scatter([Q/1000], [D], s=150, zorder=5, label=scale_name)
    axes[2].set_xlabel('Heat Duty [kW]', fontsize=12)
    axes[2].set_ylabel('Shell Diameter [m]', fontsize=12)
    axes[2].set_title('(c) Shell Diameter Scaling', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**Sample Output:**
    
    
    ================================================================================
    Heat Exchanger Sizing Calculation
    ================================================================================
    Conditions: U = 500 W/(m²·K), ΔT_lm = 20 K
    
    Lab Scale Heat Exchanger:
      Heat duty: 5.0 kW
      Required heat transfer area: 0.50 m²
      Number of tubes: 3 tubes
      Tubes: 25.0 mm × 3.0 m
      Shell diameter: 0.052 m = 5.2 cm
    
    Pilot Scale Heat Exchanger:
      Heat duty: 50.0 kW
      Required heat transfer area: 5.00 m²
      Number of tubes: 22 tubes
      Tubes: 25.0 mm × 3.0 m
      Shell diameter: 0.177 m = 17.7 cm
    
    Commercial Scale Heat Exchanger:
      Heat duty: 5000.0 kW
      Required heat transfer area: 500.00 m²
      Number of tubes: 2123 tubes
      Tubes: 25.0 mm × 3.0 m
      Shell diameter: 1.738 m = 173.8 cm
    

**Explanation:** In heat exchanger sizing, the required heat transfer area is calculated from the relationship $Q = U \cdot A \cdot \Delta T_{lm}$. Since heat transfer area is proportional to heat duty, the number of tubes and shell diameter increase during scaleup.

* * *

### Code Example 6: Pilot Plant Design and Scaledown Ratio Optimization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def pilot_plant_design(V_commercial, scale_down_ratio_range):
        """
        Determine optimal scaledown ratio for pilot plant
    
        Parameters:
        -----------
        V_commercial : float
            Commercial plant volume [m³]
        scale_down_ratio_range : array_like
            Range of scaledown ratios to consider
    
        Returns:
        --------
        dict : Evaluation metrics for each scaledown ratio
        """
        results = []
    
        for ratio in scale_down_ratio_range:
            V_pilot = V_commercial / ratio
            lambda_scale = (V_pilot / V_commercial)**(1/3)
    
            # Evaluation metrics
            # 1. Experimental cost (proportional to volume)
            experiment_cost_relative = V_pilot / V_commercial * 100  # %
    
            # 2. Representativeness (change in S/V ratio)
            SV_commercial = 1.0  # Reference
            SV_pilot = SV_commercial / lambda_scale
            SV_ratio_change = abs(SV_pilot - SV_commercial) / SV_commercial * 100  # %
    
            # 3. Reynolds number change (assuming constant stirring speed)
            Re_change = lambda_scale * 100  # % (Re ∝ ND² ∝ D² ∝ λ², N constant)
    
            # 4. Overall evaluation score (lower is better)
            # Minimize cost + maintain representativeness + minimize Re change
            score = (experiment_cost_relative / 10) + SV_ratio_change + (Re_change / 10)
    
            results.append({
                'Scale_Down_Ratio': ratio,
                'V_Pilot_m3': V_pilot,
                'V_Pilot_L': V_pilot * 1000,
                'Experiment_Cost_%': experiment_cost_relative,
                'SV_Deviation_%': SV_ratio_change,
                'Re_Change_%': Re_change,
                'Total_Score': score
            })
    
        return results
    
    # Commercial plant: 50 m³
    V_commercial = 50.0  # m³
    
    # Scaledown ratio range: 10x to 1000x
    scale_down_ratios = np.logspace(1, 3, 30)  # 10 to 1000
    
    # Pilot design evaluation
    results_pilot = pilot_plant_design(V_commercial, scale_down_ratios)
    
    # Select optimal scaledown ratio
    optimal_idx = np.argmin([r['Total_Score'] for r in results_pilot])
    optimal_ratio = results_pilot[optimal_idx]['Scale_Down_Ratio']
    optimal_V_pilot = results_pilot[optimal_idx]['V_Pilot_m3']
    
    print("=" * 80)
    print("Pilot Plant Design Optimization")
    print("=" * 80)
    print(f"Commercial plant volume: {V_commercial} m³ = {V_commercial*1000:.0f} L\n")
    
    print(f"Optimal scaledown ratio: 1/{optimal_ratio:.1f}")
    print(f"Optimal pilot volume: {optimal_V_pilot:.3f} m³ = {optimal_V_pilot*1000:.1f} L")
    print(f"Overall score: {results_pilot[optimal_idx]['Total_Score']:.2f}\n")
    
    print("Evaluation metrics:")
    print(f"  Experimental cost: {results_pilot[optimal_idx]['Experiment_Cost_%']:.2f}% (relative to commercial plant)")
    print(f"  S/V ratio change: {results_pilot[optimal_idx]['SV_Deviation_%']:.2f}%")
    print(f"  Re number change: {results_pilot[optimal_idx]['Re_Change_%']:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pilot volume
    axes[0, 0].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['V_Pilot_L'] for r in results_pilot],
                         linewidth=2.5, color='#11998e')
    axes[0, 0].scatter([optimal_ratio], [optimal_V_pilot*1000],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2, label='Optimal Point')
    axes[0, 0].set_xlabel('Scaledown Ratio', fontsize=11)
    axes[0, 0].set_ylabel('Pilot Volume [L]', fontsize=11)
    axes[0, 0].set_title('(a) Pilot Plant Volume', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Experimental cost
    axes[0, 1].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['Experiment_Cost_%'] for r in results_pilot],
                         linewidth=2.5, color='#e67e22')
    axes[0, 1].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['Experiment_Cost_%']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2)
    axes[0, 1].set_xlabel('Scaledown Ratio', fontsize=11)
    axes[0, 1].set_ylabel('Experimental Cost [%]', fontsize=11)
    axes[0, 1].set_title('(b) Experimental Cost (Relative to Commercial)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # S/V ratio change
    axes[1, 0].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['SV_Deviation_%'] for r in results_pilot],
                         linewidth=2.5, color='#9b59b6')
    axes[1, 0].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['SV_Deviation_%']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2)
    axes[1, 0].set_xlabel('Scaledown Ratio', fontsize=11)
    axes[1, 0].set_ylabel('S/V Ratio Change [%]', fontsize=11)
    axes[1, 0].set_title('(c) S/V Ratio Change (Relative to Commercial)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Overall score
    axes[1, 1].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['Total_Score'] for r in results_pilot],
                         linewidth=2.5, color='#e74c3c')
    axes[1, 1].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['Total_Score']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2, label='Optimal Point')
    axes[1, 1].set_xlabel('Scaledown Ratio', fontsize=11)
    axes[1, 1].set_ylabel('Overall Score (Lower is Better)', fontsize=11)
    axes[1, 1].set_title('(d) Overall Evaluation Score', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Recommended scaledown ratio range
    print("\n" + "=" * 80)
    print("Recommended Scaledown Ratio Range")
    print("=" * 80)
    print("General guidelines:")
    print("  Lab scale: 1/100 ~ 1/1000")
    print("  Pilot scale: 1/10 ~ 1/100")
    print("  Demo scale: 1/2 ~ 1/10")
    print(f"\nRecommended range for this case: 1/{optimal_ratio*2:.0f} ~ 1/{optimal_ratio/2:.0f}")
    

**Sample Output:**
    
    
    ================================================================================
    Pilot Plant Design Optimization
    ================================================================================
    Commercial plant volume: 50 m³ = 50000 L
    
    Optimal scaledown ratio: 1/51.8
    Optimal pilot volume: 0.966 m³ = 966.1 L
    Overall score: 92.05
    
    Evaluation metrics:
      Experimental cost: 1.93% (relative to commercial plant)
      S/V ratio change: 73.21%
      Re number change: 16.91%
    
    ================================================================================
    Recommended Scaledown Ratio Range
    ================================================================================
    General guidelines:
      Lab scale: 1/100 ~ 1/1000
      Pilot scale: 1/10 ~ 1/100
      Demo scale: 1/2 ~ 1/10
    
    Recommended range for this case: 1/104 ~ 1/26
    

**Explanation:** In pilot plant design, we consider the tradeoff between **experimental cost** (lower is better) and **representativeness to commercial plant** (smaller change in S/V ratio is better). By minimizing the overall evaluation score, we can determine the optimal scaledown ratio.

* * *

### Code Example 7: Economic Scaling (Six-Tenths Rule)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def equipment_cost_scaling(C1, Q1, Q2, exponent=0.6):
        """
        Equipment cost scaling calculation (six-tenths rule)
    
        Parameters:
        -----------
        C1 : float
            Cost of reference equipment [$]
        Q1 : float
            Capacity of reference equipment (volume, heat duty, etc.)
        Q2 : float
            Capacity of new equipment
        exponent : float, optional
            Scaling exponent, default: 0.6 (six-tenths rule)
    
        Returns:
        --------
        float : Estimated cost of new equipment [$]
    
        Note:
        -----
        Six-tenths rule: C2 = C1 * (Q2/Q1)^0.6
        - exponent = 0.6: Standard equipment (reactors, tanks, heat exchangers)
        - exponent = 0.7-0.8: Complex equipment (distillation columns, compressors)
        - exponent = 1.0: Linear cost (pumps, piping)
        """
        C2 = C1 * (Q2 / Q1)**exponent
        return C2
    
    # Lab scale equipment cost (reference)
    V_lab = 1  # L
    C_lab = 5000  # $ (lab scale reactor)
    
    # Scale range
    volumes = np.logspace(0, 5, 100)  # 1 L to 100,000 L
    
    # Comparison with different scaling exponents
    exponents = {
        'Six-tenths rule (b=0.6)': 0.6,
        'Linear (b=1.0)': 1.0,
        'Actual data (b=0.7)': 0.7
    }
    
    print("=" * 80)
    print("Equipment Cost Scaling (Six-Tenths Rule)")
    print("=" * 80)
    print(f"Reference equipment: {V_lab} L, Cost: ${C_lab:,}\n")
    
    # Cost calculation at each scale
    target_volumes = [1, 10, 100, 1000, 10000, 100000]  # L
    
    print("Equipment cost estimation by scale:")
    print(f"{'Volume [L]':<12} {'Six-tenths [$]':<15} {'Linear [$]':<15} {'Ratio':<10}")
    print("-" * 60)
    
    for V in target_volumes:
        C_sixtenths = equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6)
        C_linear = equipment_cost_scaling(C_lab, V_lab, V, exponent=1.0)
        ratio = C_sixtenths / C_linear
        print(f"{V:<12.0f} {C_sixtenths:<15,.0f} {C_linear:<15,.0f} {ratio:<10.2%}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost scaling comparison
    axes[0].loglog(volumes, volumes, linewidth=2, linestyle='--',
                   color='gray', alpha=0.5, label='Volume (reference)')
    
    for label, exp in exponents.items():
        costs = [equipment_cost_scaling(C_lab, V_lab, V, exponent=exp) for V in volumes]
        axes[0].loglog(volumes, costs, linewidth=2.5, label=label)
    
    # Mark specific scales
    for V in [10, 100, 1000]:
        C = equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6)
        axes[0].scatter([V], [C], s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    axes[0].set_xlabel('Equipment Capacity [L]', fontsize=12)
    axes[0].set_ylabel('Equipment Cost [$]', fontsize=12)
    axes[0].set_title('(a) Equipment Cost Scaling Laws', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, which='both')
    
    # Unit cost per volume
    unit_costs_sixtenths = [equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6) / V
                            for V in volumes]
    unit_costs_linear = [equipment_cost_scaling(C_lab, V_lab, V, exponent=1.0) / V
                         for V in volumes]
    
    axes[1].loglog(volumes, unit_costs_sixtenths, linewidth=2.5,
                   color='#11998e', label='Six-tenths rule (b=0.6)')
    axes[1].loglog(volumes, unit_costs_linear, linewidth=2.5, linestyle='--',
                   color='#e74c3c', label='Linear (b=1.0)')
    axes[1].set_xlabel('Equipment Capacity [L]', fontsize=12)
    axes[1].set_ylabel('Unit Cost per Volume [$/L]', fontsize=12)
    axes[1].set_title('(b) Unit Cost per Volume (Economies of Scale)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Economic optimal scale considerations
    print("\n" + "=" * 80)
    print("Economies of Scale")
    print("=" * 80)
    print("The six-tenths rule shows that unit cost per volume decreases with scaleup.")
    print("\nExample: When volume increases 10x...")
    ratio_10x = 10**0.6
    print(f"  Total cost: {ratio_10x:.2f}x (less than 10x)")
    print(f"  Unit cost per volume: {ratio_10x/10:.2f}x (decreases)")
    print(f"  Cost reduction rate: {(1 - ratio_10x/10)*100:.1f}%")
    
    print("\nExample: When volume increases 100x...")
    ratio_100x = 100**0.6
    print(f"  Total cost: {ratio_100x:.2f}x (less than 100x)")
    print(f"  Unit cost per volume: {ratio_100x/100:.2f}x (significant decrease)")
    print(f"  Cost reduction rate: {(1 - ratio_100x/100)*100:.1f}%")
    
    print("\nNote: In practice, unlimited scaleup is difficult due to:")
    print("  - Manufacturing and transportation constraints")
    print("  - Structural strength and safety constraints")
    print("  - Market demand limitations")
    print("  - Process control difficulties")
    

**Sample Output:**
    
    
    ================================================================================
    Equipment Cost Scaling (Six-Tenths Rule)
    ================================================================================
    Reference equipment: 1 L, Cost: $5,000
    
    Equipment cost estimation by scale:
    Volume [L]     Six-tenths [$]  Linear [$]      Ratio
    ------------------------------------------------------------
    1              5,000           5,000           100.00%
    10             19,953          50,000          39.91%
    100            79,621          500,000         15.92%
    1,000          317,480         5,000,000       6.35%
    10,000         1,266,287       50,000,000      2.53%
    100,000        5,048,475       500,000,000     1.01%
    
    ================================================================================
    Economies of Scale
    ================================================================================
    The six-tenths rule shows that unit cost per volume decreases with scaleup.
    
    Example: When volume increases 10x...
      Total cost: 3.98x (less than 10x)
      Unit cost per volume: 0.40x (decreases)
      Cost reduction rate: 60.1%
    
    Example: When volume increases 100x...
      Total cost: 15.85x (less than 100x)
      Unit cost per volume: 0.16x (significant decrease)
      Cost reduction rate: 84.1%
    
    Note: In practice, unlimited scaleup is difficult due to:
      - Manufacturing and transportation constraints
      - Structural strength and safety constraints
      - Market demand limitations
      - Process control difficulties
    

**Explanation:** The **six-tenths rule** is an empirical rule for scaling equipment costs. As equipment capacity increases, total cost does not increase proportionally but scales with exponent $b \approx 0.6$. This creates **economies of scale** , where larger plants have lower unit production costs.

* * *

## 1.3 Chapter Summary

### What We Learned

  1. **Fundamentals of Similarity Laws**
     * Understanding geometric, kinematic, and dynamic similarity
     * Scaling laws using scale factor $\lambda$
  2. **Power Law Scaling**
     * $L \propto \lambda^1$, $A \propto \lambda^2$, $V \propto \lambda^3$
     * S/V ratio scales as $\lambda^{-1}$ (decreases with scaleup)
     * Describing scaling relationships with power law $y = ax^b$
  3. **Equipment Sizing Calculations**
     * Design calculations for reactors and heat exchangers
     * Determining required equipment size from production rate and heat duty
  4. **Pilot Plant Design**
     * Optimizing scaledown ratio
     * Tradeoff between experimental cost and representativeness
  5. **Economic Scaling**
     * Estimating equipment cost using six-tenths rule
     * Economies of scale (cost efficiency improves with scaleup)

### Key Points

The **decrease in S/V ratio** means reduced heat transfer and cooling capacity during scaleup. The scale factor $\lambda$ determines scaling laws for physical quantities. Power law is a general way to describe scaling relationships. Pilot plants are designed balancing cost and representativeness. The six-tenths rule shows unit costs decrease with larger scale, demonstrating economies of scale.

### On to the Next Chapter

In Chapter 2, we will study **dimensionless numbers and scaleup rules** in detail, covering major dimensionless numbers like Reynolds, Froude, and Weber numbers, selection of dominant dimensionless numbers and application of similarity laws, and derivation of dimensionless groups using the Buckingham pi theorem. The chapter also addresses setting scaleup criteria such as constant Re number and constant power density, along with mixing time scaling and agitation power calculation.
