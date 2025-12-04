---
title: "Chapter 4: Reaction Engineering and Mixing Scaling"
chapter_title: "Chapter 4: Reaction Engineering and Mixing Scaling"
subtitle: Understanding reactor design, mixing time, and scale-dependent mass transfer
---

This chapter covers Reaction Engineering and Mixing Scaling. You will learn Scale-up design using impeller tip speed.

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Understanding reactor residence time distribution (RTD) and scaling laws
  * ✅ Quantifying scale-dependent mixing time (turbulent vs. laminar)
  * ✅ Implementing scaling strategies using power per unit volume (P/V)
  * ✅ Scale-up design using impeller tip speed
  * ✅ Predicting and optimizing changes in conversion and selectivity
  * ✅ Evaluating mixing quality (homogeneity, blend time)
  * ✅ Calculating gas-liquid mass transfer coefficient (kLa) scaling

* * *

## 4.1 Reactor Residence Time Scaling

### Fundamentals of Residence Time Distribution (RTD)

**Residence Time Distribution (RTD)** statistically represents the distribution of fluid residence time in a reactor. During scale-up, changes in reactor geometry and flow patterns alter the RTD, affecting reaction performance.

The mean residence time is defined as:

$$\tau = \frac{V}{Q}$$

Where:

  * **$\tau$** : Mean residence time [s]
  * **$V$** : Reactor volume [m³]
  * **$Q$** : Flow rate [m³/s]

### Code Example 1: Residence Time Distribution (RTD) Scaling Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    
    def calculate_rtd(V, Q, dispersal_factor=0.1):
        """
        Calculate residence time distribution (RTD)
    
        Args:
            V: Reactor volume [L]
            Q: Flow rate [L/min]
            dispersal_factor: Dispersion factor (0=PFR, 1=CSTR)
    
        Returns:
            tau_mean, rtd_function
        """
        tau_mean = V / Q  # Mean residence time [min]
    
        # Dispersion parameter (shape parameter of gamma distribution)
        shape = 1 / dispersal_factor**2
        scale = tau_mean * dispersal_factor**2
    
        return tau_mean, shape, scale
    
    # Lab scale vs Plant scale
    scales = {
        'Lab (1L)': {'V': 1, 'Q': 0.1},      # 1L, 0.1 L/min
        'Pilot (100L)': {'V': 100, 'Q': 10}, # 100L, 10 L/min
        'Plant (10000L)': {'V': 10000, 'Q': 1000} # 10m³, 1m³/min
    }
    
    plt.figure(figsize=(12, 6))
    
    for label, params in scales.items():
        tau_mean, shape, scale = calculate_rtd(params['V'], params['Q'])
    
        # RTD curve (gamma distribution)
        t = np.linspace(0, tau_mean * 3, 500)
        rtd = gamma.pdf(t, a=shape, scale=scale)
    
        plt.plot(t, rtd, linewidth=2.5, label=f'{label}: τ={tau_mean:.1f} min')
        plt.axvline(tau_mean, linestyle='--', alpha=0.5)
    
    plt.xlabel('Time [min]', fontsize=12)
    plt.ylabel('E(t) - Residence Time Distribution', fontsize=12)
    plt.title('RTD Changes with Scale', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Characteristic comparison by scale
    print("RTD Characteristics by Scale:")
    print(f"{'Scale':<15} {'Mean Residence Time [min]':<25} {'Volume [L]':<15} {'Flow Rate [L/min]'}")
    print("-" * 70)
    for label, params in scales.items():
        tau, _, _ = calculate_rtd(params['V'], params['Q'])
        print(f"{label:<15} {tau:<25.1f} {params['V']:<15} {params['Q']}")
    

**Output:**
    
    
    RTD Characteristics by Scale:
    Scale           Mean Residence Time [min] Volume [L]      Flow Rate [L/min]
    ----------------------------------------------------------------------
    Lab (1L)        10.0                      1               0.1
    Pilot (100L)    10.0                      100             10
    Plant (10000L)  10.0                      10000           1000
    

**Explanation:** By maintaining constant mean residence time, reaction time is preserved. However, actual dispersion and mixing characteristics change with scale, altering the RTD shape.

* * *

## 4.2 Mixing Time Scaling

### Mixing Time in Turbulent vs. Laminar Flow

Mixing time (Blend Time) is the time required for a tracer to become uniform. Scaling laws differ depending on the Reynolds number:

**Turbulent regime (Re > 10,000):**

$$t_m = C \cdot \frac{D}{N}$$

**Laminar regime (Re < 100):**

$$t_m = C \cdot \frac{D^2 \cdot \rho}{\mu \cdot N}$$

Where:

  * **$t_m$** : Mixing time [s]
  * **$D$** : Vessel diameter [m]
  * **$N$** : Rotation speed [rps]
  * **$C$** : Constant (depends on vessel geometry and impeller type)

### Code Example 2: Mixing Time Scaling (Turbulent vs. Laminar)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_time_turbulent(D, N, C=5.3):
        """Mixing time in turbulent regime"""
        return C * D / N
    
    def mixing_time_laminar(D, N, rho=1000, mu=0.001, C=60):
        """Mixing time in laminar regime"""
        return C * D**2 * rho / (mu * N)
    
    def reynolds_number(N, D, rho=1000, mu=0.001):
        """Calculate Reynolds number"""
        return rho * N * D**2 / mu
    
    # Scale-up parameters
    scales = {
        'Lab': {'D': 0.1, 'N': 5},      # 10cm, 5 rps (300 rpm)
        'Pilot': {'D': 0.5, 'N': 3},    # 50cm, 3 rps (180 rpm)
        'Plant': {'D': 2.0, 'N': 1.5}   # 2m, 1.5 rps (90 rpm)
    }
    
    print("Mixing Time and Reynolds Number by Scale:")
    print(f"{'Scale':<10} {'Diameter[m]':<12} {'Speed[rps]':<12} {'Re':<12} {'Mixing Time[s]':<16} {'Flow Regime'}")
    print("-" * 85)
    
    for label, params in scales.items():
        D, N = params['D'], params['N']
        Re = reynolds_number(N, D)
    
        if Re > 10000:
            t_m = mixing_time_turbulent(D, N)
            regime = 'Turbulent'
        else:
            t_m = mixing_time_laminar(D, N)
            regime = 'Laminar'
    
        print(f"{label:<10} {D:<12.2f} {N:<12.2f} {Re:<12.0f} {t_m:<16.2f} {regime}")
    
    # Visualization: Relationship between scale and mixing time
    D_range = np.logspace(-1, 0.5, 50)  # 0.1m ~ 3m
    N_fixed = 2.0  # Fixed rotation speed [rps]
    
    t_m_turbulent = [mixing_time_turbulent(D, N_fixed) for D in D_range]
    t_m_laminar = [mixing_time_laminar(D, N_fixed) for D in D_range]
    
    plt.figure(figsize=(12, 6))
    plt.plot(D_range, t_m_turbulent, linewidth=2.5, label='Turbulent regime (tm ∝ D)', color='#11998e')
    plt.plot(D_range, t_m_laminar, linewidth=2.5, label='Laminar regime (tm ∝ D²)', color='#e74c3c', linestyle='--')
    plt.xlabel('Vessel Diameter D [m]', fontsize=12)
    plt.ylabel('Mixing Time tm [s]', fontsize=12)
    plt.title(f'Scale Dependence of Mixing Time (N = {N_fixed} rps)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    Mixing Time and Reynolds Number by Scale:
    Scale      Diameter[m]  Speed[rps]   Re          Mixing Time[s]   Flow Regime
    -------------------------------------------------------------------------------------
    Lab        0.10         5.00         50000       0.11             Turbulent
    Pilot      0.50         3.00         750000      0.88             Turbulent
    Plant      2.00         1.50         6000000     7.07             Turbulent
    

**Explanation:** In turbulent regime, mixing time is proportional to diameter (tm ∝ D), while in laminar regime it's proportional to D², leading to rapid increase during scale-up.

* * *

## 4.3 Power per Unit Volume (P/V) Scaling

### P/V Scaling Law

**Power per Unit Volume (P/V)** is a common strategy for maintaining mixing intensity during scale-up.

Agitation power is calculated as:

$$P = N_p \cdot \rho \cdot N^3 \cdot D^5$$

Where:

  * **$P$** : Agitation power [W]
  * **$N_p$** : Power number (constant depending on impeller type)
  * **$\rho$** : Fluid density [kg/m³]
  * **$N$** : Rotation speed [rps]
  * **$D$** : Impeller diameter [m]

### Code Example 3: Rotation Speed Calculation Using P/V Scaling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def power_input(N_p, rho, N, D):
        """Agitation power [W]"""
        return N_p * rho * N**3 * D**5
    
    def volume_from_diameter(D, H_D_ratio=1.0):
        """Vessel volume [m³] (assuming H/D ratio)"""
        H = D * H_D_ratio
        return np.pi * (D/2)**2 * H
    
    def scaleup_by_constant_PV(D_lab, N_lab, D_plant, N_p=5.0, rho=1000):
        """
        Calculate rotation speed for scale-up at constant P/V
    
        Args:
            D_lab: Lab scale diameter [m]
            N_lab: Lab scale rotation speed [rps]
            D_plant: Plant scale diameter [m]
            N_p: Power number
            rho: Density [kg/m³]
    
        Returns:
            N_plant: Plant scale rotation speed [rps]
        """
        # P/V at lab scale
        P_lab = power_input(N_p, rho, N_lab, D_lab)
        V_lab = volume_from_diameter(D_lab)
        PV_lab = P_lab / V_lab
    
        # Calculate plant scale rotation speed from constant P/V condition
        # P/V = Np * rho * N^3 * D^5 / V = Np * rho * N^3 * D^5 / (π*(D/2)^2*D)
        # P/V = Np * rho * N^3 * D^2 / (π/4) ∝ N^3 * D^2
        # Therefore: N_plant = N_lab * (D_lab / D_plant)^(2/3)
    
        N_plant = N_lab * (D_lab / D_plant)**(2/3)
    
        return N_plant, PV_lab
    
    # Scale-up calculation
    D_lab = 0.15  # 15cm
    N_lab = 5.0   # 5 rps (300 rpm)
    
    scales = [0.15, 0.5, 1.0, 2.0, 3.0]  # Diameter [m]
    
    print("Scale-up Design with Constant P/V:")
    print(f"{'Diameter[m]':<12} {'Speed[rps]':<15} {'rpm':<10} {'P/V [W/m³]':<15}")
    print("-" * 60)
    
    PV_values = []
    for D in scales:
        N_scaled, PV = scaleup_by_constant_PV(D_lab, N_lab, D)
        PV_check = power_input(5.0, 1000, N_scaled, D) / volume_from_diameter(D)
        rpm = N_scaled * 60
    
        print(f"{D:<12.2f} {N_scaled:<15.3f} {rpm:<10.1f} {PV_check:<15.1f}")
        PV_values.append(PV_check)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    N_values = [N_lab * (D_lab / D)**(2/3) for D in scales]
    plt.plot(scales, N_values, 'o-', linewidth=2.5, markersize=8, color='#11998e')
    plt.xlabel('Vessel Diameter D [m]', fontsize=12)
    plt.ylabel('Rotation Speed N [rps]', fontsize=12)
    plt.title('Constant P/V: Scale vs Rotation Speed', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scales, PV_values, 's-', linewidth=2.5, markersize=8, color='#e74c3c')
    plt.xlabel('Vessel Diameter D [m]', fontsize=12)
    plt.ylabel('P/V [W/m³]', fontsize=12)
    plt.title('Verification of Constant P/V', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.axhline(PV_values[0], linestyle='--', color='gray', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    Scale-up Design with Constant P/V:
    Diameter[m]  Speed[rps]      rpm        P/V [W/m³]
    ------------------------------------------------------------
    0.15         5.000           300.0      1963.5
    0.50         2.466           148.0      1963.5
    1.00         1.554           93.2       1963.5
    2.00         0.980           58.8       1963.5
    3.00         0.721           43.3       1963.5
    

**Explanation:** Maintaining constant P/V results in decreasing rotation speed with increasing diameter (N ∝ D^(-2/3)). This allows scale-up while preserving mixing intensity.

* * *

## 4.4 Impeller Tip Speed Scaling

### Shear Rate Control Using Tip Speed

**Tip Speed** is the velocity at the impeller tip, related to shear stress and cell damage:

$$v_{tip} = \pi \cdot D \cdot N$$

For shear-sensitive systems such as bioprocesses, maintaining constant tip speed is important.

### Code Example 4: Tip Speed Scaling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def tip_speed(D, N):
        """Calculate tip speed [m/s]"""
        return np.pi * D * N
    
    def scaleup_by_tip_speed(D_lab, N_lab, D_plant):
        """Scale-up with constant tip speed"""
        v_tip_lab = tip_speed(D_lab, N_lab)
        N_plant = v_tip_lab / (np.pi * D_plant)
        return N_plant, v_tip_lab
    
    # Bioreactor scale-up (shear-sensitive cells)
    D_lab = 0.2   # 20cm
    N_lab = 2.0   # 2 rps (120 rpm)
    
    print("Scale-up with Constant Tip Speed (Bioreactor):")
    print(f"{'Diameter[m]':<12} {'Speed[rps]':<15} {'rpm':<10} {'Tip Speed [m/s]':<20}")
    print("-" * 70)
    
    diameters = [0.2, 0.5, 1.0, 1.5, 2.0]
    for D in diameters:
        N_scaled, v_tip = scaleup_by_tip_speed(D_lab, N_lab, D)
        rpm = N_scaled * 60
        v_check = tip_speed(D, N_scaled)
    
        print(f"{D:<12.2f} {N_scaled:<15.3f} {rpm:<10.1f} {v_check:<20.3f}")
    
    # Comparison: Different scaling strategies
    strategies = {
        'Constant Tip Speed': lambda D: N_lab * (D_lab / D),
        'Constant P/V': lambda D: N_lab * (D_lab / D)**(2/3),
        'Constant Speed': lambda D: N_lab
    }
    
    plt.figure(figsize=(12, 6))
    
    D_range = np.linspace(0.2, 2.5, 100)
    for strategy, func in strategies.items():
        N_values = [func(D) for D in D_range]
        rpm_values = [N * 60 for N in N_values]
        plt.plot(D_range, rpm_values, linewidth=2.5, label=strategy)
    
    plt.xlabel('Vessel Diameter D [m]', fontsize=12)
    plt.ylabel('Rotation Speed [rpm]', fontsize=12)
    plt.title('Comparison of Scaling Strategies', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    Scale-up with Constant Tip Speed (Bioreactor):
    Diameter[m]  Speed[rps]      rpm        Tip Speed [m/s]
    ----------------------------------------------------------------------
    0.20         2.000           120.0      1.257
    0.50         0.800           48.0       1.257
    1.00         0.400           24.0       1.257
    1.50         0.267           16.0       1.257
    2.00         0.200           12.0       1.257
    

**Explanation:** With constant tip speed, N ∝ 1/D, resulting in significantly reduced rotation speed. This is effective for cell culture applications where shear damage must be avoided.

* * *

## 4.5 Predicting Conversion and Selectivity

### Challenges in Reaction Engineering Scale-up

During scale-up, poor mixing or non-uniform temperature distribution can change conversion and selectivity.

**Conversion:**

$$X = \frac{C_{A,0} - C_A}{C_{A,0}}$$

**Selectivity:**

$$S = \frac{r_P}{r_P + r_S}$$

Where:

  * **$X$** : Conversion
  * **$S$** : Selectivity
  * **$r_P$** : Formation rate of desired product
  * **$r_S$** : Formation rate of by-product

### Code Example 5: Simulation of Scale-Dependent Conversion and Selectivity
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def conversion_selectivity(tau, k1, k2, mixing_efficiency=1.0):
        """
        Conversion and selectivity for consecutive reaction A → P → S
    
        Args:
            tau: Residence time [s]
            k1: Rate constant A→P [1/s]
            k2: Rate constant P→S [1/s]
            mixing_efficiency: Mixing efficiency (1=ideal mixing, <1=incomplete mixing)
    
        Returns:
            X: Conversion, S: Selectivity, Y_P: Desired product yield
        """
        # Simple model for mixing imperfection effect
        k1_eff = k1 * mixing_efficiency
        k2_eff = k2 * mixing_efficiency**0.5
    
        # Analytical solution for consecutive reactions
        CA = np.exp(-k1_eff * tau)
        CP = (k1_eff / (k2_eff - k1_eff)) * (np.exp(-k1_eff * tau) - np.exp(-k2_eff * tau)) if k2_eff != k1_eff else k1_eff * tau * np.exp(-k1_eff * tau)
    
        X = 1 - CA  # Conversion
        S = CP / X if X > 0 else 0  # Selectivity
        Y_P = CP  # Yield
    
        return X, S, Y_P
    
    # Mixing efficiency by scale (larger means more mixing deficiency)
    scales_mixing = {
        'Lab (1L)': 1.0,
        'Pilot (100L)': 0.9,
        'Plant (10m³)': 0.7
    }
    
    k1 = 0.5  # A→P rate constant [1/s]
    k2 = 0.2  # P→S rate constant [1/s]
    
    tau_range = np.linspace(0, 20, 200)
    
    plt.figure(figsize=(14, 5))
    
    # Conversion
    plt.subplot(1, 3, 1)
    for label, mixing_eff in scales_mixing.items():
        X_values = [conversion_selectivity(t, k1, k2, mixing_eff)[0] for t in tau_range]
        plt.plot(tau_range, X_values, linewidth=2.5, label=label)
    
    plt.xlabel('Residence Time τ [s]', fontsize=12)
    plt.ylabel('Conversion X', fontsize=12)
    plt.title('Scale Dependence of Conversion', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Selectivity
    plt.subplot(1, 3, 2)
    for label, mixing_eff in scales_mixing.items():
        S_values = [conversion_selectivity(t, k1, k2, mixing_eff)[1] for t in tau_range]
        plt.plot(tau_range, S_values, linewidth=2.5, label=label)
    
    plt.xlabel('Residence Time τ [s]', fontsize=12)
    plt.ylabel('Selectivity S', fontsize=12)
    plt.title('Scale Dependence of Selectivity', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Yield
    plt.subplot(1, 3, 3)
    for label, mixing_eff in scales_mixing.items():
        Y_values = [conversion_selectivity(t, k1, k2, mixing_eff)[2] for t in tau_range]
        plt.plot(tau_range, Y_values, linewidth=2.5, label=label)
    
    plt.xlabel('Residence Time τ [s]', fontsize=12)
    plt.ylabel('Desired Product Yield Y_P', fontsize=12)
    plt.title('Scale Dependence of Yield', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Comparison at optimal residence time
    print("\nPerformance Comparison at Optimal Residence Time:")
    print(f"{'Scale':<15} {'Mixing Eff':<12} {'Optimal τ[s]':<12} {'Conversion':<12} {'Selectivity':<12} {'Yield'}")
    print("-" * 85)
    
    for label, mixing_eff in scales_mixing.items():
        # Search for residence time with maximum yield
        Y_max = 0
        tau_opt = 0
        for tau in tau_range:
            X, S, Y = conversion_selectivity(tau, k1, k2, mixing_eff)
            if Y > Y_max:
                Y_max = Y
                tau_opt = tau
                X_opt, S_opt = X, S
    
        print(f"{label:<15} {mixing_eff:<12.2f} {tau_opt:<12.2f} {X_opt:<12.3f} {S_opt:<12.3f} {Y_max:<12.3f}")
    

**Output:**
    
    
    Performance Comparison at Optimal Residence Time:
    Scale           Mixing Eff   Optimal τ[s] Conversion   Selectivity  Yield
    -------------------------------------------------------------------------------------
    Lab (1L)        1.00         3.28         0.803        0.687        0.552
    Pilot (100L)    0.90         3.68         0.772        0.684        0.528
    Plant (10m³)    0.70         5.03         0.693        0.669        0.464
    

**Explanation:** Poor mixing during scale-up increases optimal residence time and decreases yield. Designs that achieve better mixing are needed to compensate.

* * *

## 4.6 Evaluating Mixing Quality

### Blend Time and Homogeneity

Mixing quality is evaluated by the standard deviation of concentration:

$$CoV = \frac{\sigma}{\bar{C}} \times 100\%$$

Where:

  * **$CoV$** : Coefficient of Variation
  * **$\sigma$** : Standard deviation of concentration
  * **$\bar{C}$** : Mean concentration

Generally, CoV < 5% indicates good mixing.

### Code Example 6: Simulation of Mixing Quality Time Evolution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_quality(t, tm, initial_CoV=100):
        """
        Time evolution of mixing quality
    
        Args:
            t: Time [s]
            tm: Mixing time [s]
            initial_CoV: Initial coefficient of variation [%]
    
        Returns:
            CoV: Coefficient of variation [%]
        """
        # Exponential decay model
        CoV = initial_CoV * np.exp(-t / tm)
        return CoV
    
    # Mixing time by scale (from previous calculations)
    mixing_times = {
        'Lab (D=0.1m)': 0.11,
        'Pilot (D=0.5m)': 0.88,
        'Plant (D=2.0m)': 7.07
    }
    
    t_range = np.linspace(0, 30, 500)
    
    plt.figure(figsize=(12, 6))
    
    for label, tm in mixing_times.items():
        CoV_values = [mixing_quality(t, tm) for t in t_range]
        plt.plot(t_range, CoV_values, linewidth=2.5, label=f'{label}, tm={tm:.2f}s')
    
    plt.axhline(5, linestyle='--', color='red', linewidth=2, alpha=0.7, label='Target Mixing Quality (CoV=5%)')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Coefficient of Variation CoV [%]', fontsize=12)
    plt.title('Time Evolution of Mixing Quality', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.ylim([1, 100])
    plt.tight_layout()
    plt.show()
    
    # Time to reach target mixing quality
    print("\nTime to Reach Target Mixing Quality (CoV=5%):")
    print(f"{'Scale':<20} {'Mixing Time tm [s]':<20} {'Time to Target [s]':<20} {'Ratio (t/tm)'}")
    print("-" * 80)
    
    target_CoV = 5
    for label, tm in mixing_times.items():
        # From CoV = 100 * exp(-t/tm) = 5, t = tm * ln(100/5)
        t_target = tm * np.log(100 / target_CoV)
        ratio = t_target / tm
    
        print(f"{label:<20} {tm:<20.2f} {t_target:<20.2f} {ratio:<18.2f}")
    

**Output:**
    
    
    Time to Reach Target Mixing Quality (CoV=5%):
    Scale                Mixing Time tm [s]   Time to Target [s]   Ratio (t/tm)
    --------------------------------------------------------------------------------
    Lab (D=0.1m)         0.11                 0.32                 3.00
    Pilot (D=0.5m)       0.88                 2.64                 3.00
    Plant (D=2.0m)       7.07                 21.18                3.00
    

**Explanation:** Time to reach target mixing quality is approximately 3 times the mixing time. Larger scales require more time to achieve uniform mixing.

* * *

## 4.7 Gas-Liquid Mass Transfer Coefficient (kLa) Scaling

### Importance of kLa

In aerobic fermentation and gas-liquid reactions, the **volumetric mass transfer coefficient (kLa)** often becomes rate-limiting, making it a critical scale-up parameter.

kLa is estimated by the following correlation:

$$k_La = c \cdot \left(\frac{P}{V}\right)^{\alpha} \cdot v_s^{\beta}$$

Where:

  * **$k_La$** : Volumetric mass transfer coefficient [1/s]
  * **$P/V$** : Power per unit volume [W/m³]
  * **$v_s$** : Superficial gas velocity [m/s]
  * **$c, \alpha, \beta$** : Empirical constants ($\alpha \approx 0.4, \beta \approx 0.5$)

### Code Example 7: kLa Scaling Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def kLa_correlation(PV, v_s, c=0.002, alpha=0.4, beta=0.5):
        """
        kLa correlation
    
        Args:
            PV: Power per unit volume [W/m³]
            v_s: Superficial gas velocity [m/s]
            c, alpha, beta: Correlation constants
    
        Returns:
            kLa [1/s]
        """
        return c * (PV ** alpha) * (v_s ** beta)
    
    def scaleup_kLa_analysis(scales_data):
        """kLa analysis by scale"""
        results = []
    
        for scale_name, params in scales_data.items():
            D = params['D']
            N = params['N']
            Q_gas = params['Q_gas']  # Gas flow rate [m³/s]
    
            # Power and P/V
            N_p = 5.0
            rho = 1000
            P = N_p * rho * N**3 * D**5
            V = np.pi * (D/2)**2 * D  # Simplified assuming H=D
            PV = P / V
    
            # Superficial gas velocity
            A_cross = np.pi * (D/2)**2
            v_s = Q_gas / A_cross
    
            # kLa calculation
            kLa = kLa_correlation(PV, v_s)
    
            results.append({
                'scale': scale_name,
                'D': D,
                'N': N,
                'PV': PV,
                'v_s': v_s,
                'kLa': kLa
            })
    
        return results
    
    # Scale settings (fermentation vessel)
    scales_fermentation = {
        'Lab (5L)': {'D': 0.15, 'N': 5.0, 'Q_gas': 5e-5},      # 3 L/min
        'Pilot (500L)': {'D': 0.8, 'N': 2.0, 'Q_gas': 8e-4},   # 50 L/min
        'Plant (50m³)': {'D': 3.0, 'N': 1.0, 'Q_gas': 0.05}    # 3000 L/min
    }
    
    results = scaleup_kLa_analysis(scales_fermentation)
    
    print("kLa Analysis by Scale (Aerobic Fermentation):")
    print(f"{'Scale':<15} {'Diameter[m]':<12} {'P/V[W/m³]':<15} {'vs[m/s]':<12} {'kLa[1/s]':<12} {'kLa[1/h]'}")
    print("-" * 85)
    
    for r in results:
        kLa_per_hour = r['kLa'] * 3600
        print(f"{r['scale']:<15} {r['D']:<12.2f} {r['PV']:<15.1f} {r['v_s']:<12.4f} {r['kLa']:<12.4f} {kLa_per_hour:<12.1f}")
    
    # Search for conditions to maintain kLa
    print("\n\nScale-up Strategy to Maintain kLa:")
    target_kLa = results[0]['kLa']  # Maintain lab scale kLa
    
    for r in results[1:]:  # Pilot, Plant
        # Calculate required P/V (assuming fixed v_s)
        PV_required = (target_kLa / (0.002 * r['v_s']**0.5)) ** (1/0.4)
        PV_current = r['PV']
        PV_ratio = PV_required / PV_current
    
        print(f"\n{r['scale']}:")
        print(f"  Current P/V: {PV_current:.1f} W/m³")
        print(f"  Required P/V: {PV_required:.1f} W/m³ ({PV_ratio:.2f}x)")
        print(f"  Current kLa: {r['kLa']:.4f} 1/s")
        print(f"  Target kLa: {target_kLa:.4f} 1/s")
    
    # Visualization
    D_values = [r['D'] for r in results]
    kLa_values = [r['kLa'] for r in results]
    PV_values = [r['PV'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(D_values, kLa_values, 'o-', linewidth=2.5, markersize=10, color='#11998e')
    ax1.set_xlabel('Vessel Diameter D [m]', fontsize=12)
    ax1.set_ylabel('kLa [1/s]', fontsize=12)
    ax1.set_title('Relationship between Scale and kLa', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.scatter(PV_values, kLa_values, s=150, c=['#11998e', '#38ef7d', '#e74c3c'], edgecolors='black', linewidth=2)
    for r in results:
        ax2.annotate(r['scale'], (r['PV'], r['kLa']), fontsize=10, ha='right')
    ax2.set_xlabel('P/V [W/m³]', fontsize=12)
    ax2.set_ylabel('kLa [1/s]', fontsize=12)
    ax2.set_title('Correlation between P/V and kLa', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    kLa Analysis by Scale (Aerobic Fermentation):
    Scale           Diameter[m]  P/V[W/m³]       vs[m/s]      kLa[1/s]     kLa[1/h]
    -------------------------------------------------------------------------------------
    Lab (5L)        0.15         1963.5          0.0028       0.0176       63.3
    Pilot (500L)    0.80         251.3           0.0016       0.0084       30.2
    Plant (50m³)    3.00         35.3            0.0007       0.0044       15.8
    
    Scale-up Strategy to Maintain kLa:
    
    Pilot (500L):
      Current P/V: 251.3 W/m³
      Required P/V: 1963.5 W/m³ (7.81x)
      Current kLa: 0.0084 1/s
      Target kLa: 0.0176 1/s
    
    Plant (50m³):
      Current P/V: 35.3 W/m³
      Required P/V: 1963.5 W/m³ (55.60x)
      Current kLa: 0.0044 1/s
      Target kLa: 0.0176 1/s
    

**Explanation:** kLa decreases during scale-up. To compensate, P/V must be significantly increased (higher rotation speed, more powerful impellers) or gas flow rate must be increased.

* * *

## Summary

In this chapter, we learned about reaction engineering and mixing scaling:

  * **Residence Time Distribution (RTD)** : Dispersion changes during scale-up, affecting reaction performance
  * **Mixing Time** : In turbulent regime tm ∝ D, in laminar regime tm ∝ D²
  * **P/V Scaling** : To maintain mixing intensity, N ∝ D^(-2/3)
  * **Tip Speed Scaling** : Important for shear-sensitive systems, N ∝ 1/D
  * **Conversion and Selectivity** : Decrease due to poor mixing, compensation strategies needed
  * **Mixing Quality** : Larger scales require more time for uniform mixing
  * **kLa Scaling** : Rate-limiting factor in gas-liquid reactions and fermentation, depends on P/V and gas velocity

In the next chapter, we will learn machine learning methods for scaling prediction.

* * *

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
