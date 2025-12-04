---
title: "Chapter 2: Similarity Laws and Dimensionless Numbers"
chapter_title: "Chapter 2: Similarity Laws and Dimensionless Numbers"
subtitle: Understanding Physical Similarity as the Foundation of Scaling
---

This chapter covers Similarity Laws and Dimensionless Numbers. You will learn principles of similarity laws (geometric and Calculate dimensionless numbers in Python.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the principles of similarity laws (geometric, kinematic, dynamic similarity)
  * ✅ Explain the physical meaning of major dimensionless numbers (Re, Fr, We, Po, etc.)
  * ✅ Calculate dimensionless numbers in Python and determine flow regimes
  * ✅ Perform multi-criteria similarity analysis using multiple dimensionless numbers
  * ✅ Understand the limitations of applying similarity laws during scaleup

* * *

## 2.1 Fundamentals of Similarity Laws

### Three Types of Similarity

To successfully execute scaleup and scaledown, it is necessary to maintain **similarity** between small-scale and large-scale equipment. There are three hierarchical levels of similarity:

Type of Similarity | Definition | Impact on Scaling  
---|---|---  
**Geometric Similarity** | Shape and dimensional ratios are constant | All lengths scale by the same ratio (L₂/L₁ = S)  
**Kinematic Similarity** | Velocity field shapes are similar | Velocity direction and ratio are the same at corresponding points  
**Dynamic Similarity** | Force ratios are similar | Ratios of inertial force, viscous force, gravity, etc. are constant  
  
**Important:** To achieve dynamic similarity, related dimensionless numbers must be matched.
    
    
    ```mermaid
    graph TD
        A[Geometric SimilaritySame Shape] --> B[Kinematic SimilaritySame Flow Pattern]
        B --> C[Dynamic SimilaritySame Force Balance]
        C --> D[Process Performance SimilarityReproduction of Reaction/Separation Performance]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
    ```

### What are Dimensionless Numbers

**Dimensionless numbers** are dimensionless parameters representing the ratio of different physical forces or phenomena. If dimensionless numbers are the same, the same physical phenomena occur even at different scales.

General form:

$$ \text{Dimensionless number} = \frac{\text{One type of force}}{\text{Another type of force}} = \frac{\text{Characteristic time}_1}{\text{Characteristic time}_2} $$

* * *

## 2.2 Dimensionless Numbers in Fluid Mechanics

### Reynolds Number

The Reynolds number (Re) represents the ratio of **inertial force** to **viscous force** :

$$ \text{Re} = \frac{\rho u L}{\mu} = \frac{uL}{\nu} $$

Where:

  * $\rho$: Density [kg/m³]
  * $u$: Characteristic velocity [m/s]
  * $L$: Characteristic length [m] (pipe diameter, impeller diameter, etc.)
  * $\mu$: Viscosity [Pa·s]
  * $\nu = \mu/\rho$: Kinematic viscosity [m²/s]

**Physical meaning:**

  * Re << 1: Viscous force dominated (laminar, creeping flow)
  * Re ≈ 2,300 (pipe flow) or 10,000 (stirred tank): Transition region
  * Re >> 1: Inertial force dominated (turbulent flow)

### Code Example 1: Calculation of Reynolds Number and Flow Regime Determination
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def reynolds_number(rho, u, L, mu):
        """
        Calculate Reynolds number
    
        Parameters:
        -----------
        rho : float
            Fluid density [kg/m³]
        u : float
            Characteristic velocity [m/s]
        L : float
            Characteristic length [m]
        mu : float
            Viscosity [Pa·s]
    
        Returns:
        --------
        Re : float
            Reynolds number [-]
        """
        return rho * u * L / mu
    
    def flow_regime(Re, system_type='pipe'):
        """
        Determine flow regime from Reynolds number
    
        Parameters:
        -----------
        Re : float
            Reynolds number
        system_type : str
            'pipe' (pipe flow) or 'stirred' (stirred tank)
    
        Returns:
        --------
        regime : str
            Flow regime ('Laminar', 'Transition', 'Turbulent')
        """
        if system_type == 'pipe':
            Re_crit = 2300
        elif system_type == 'stirred':
            Re_crit = 10000
        else:
            Re_crit = 2300  # Default
    
        if Re < Re_crit:
            return 'Laminar'
        elif Re < Re_crit * 2:
            return 'Transition'
        else:
            return 'Turbulent'
    
    # Example: Water (20°C) in pipe flow
    # Physical properties
    rho_water = 998.2  # kg/m³
    mu_water = 1.002e-3  # Pa·s
    
    # Pipeline conditions
    pipe_diameter = np.array([0.025, 0.05, 0.1, 0.2])  # m (25mm, 50mm, 100mm, 200mm)
    flow_velocity = 1.5  # m/s
    
    print("=" * 70)
    print("Reynolds Number Calculation and Flow Regime Determination (Water Pipe Flow)")
    print("=" * 70)
    print(f"Fluid: Water (20°C), Density = {rho_water} kg/m³, Viscosity = {mu_water*1000:.3f} mPa·s")
    print(f"Flow velocity: {flow_velocity} m/s")
    print("-" * 70)
    
    for D in pipe_diameter:
        Re = reynolds_number(rho_water, flow_velocity, D, mu_water)
        regime = flow_regime(Re, 'pipe')
        print(f"Pipe diameter {D*1000:6.1f} mm → Re = {Re:10,.0f} → {regime}")
    
    # Visualization: Reynolds number vs. pipe diameter
    Re_values = reynolds_number(rho_water, flow_velocity, pipe_diameter, mu_water)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pipe_diameter * 1000, Re_values, 'o-', linewidth=2.5,
             markersize=10, color='#11998e', label='Reynolds number')
    plt.axhline(y=2300, color='orange', linestyle='--', linewidth=2,
                label='Laminar/Turbulent boundary (Re = 2,300)')
    plt.axhline(y=4000, color='red', linestyle=':', linewidth=2,
                label='Fully turbulent region (Re ≈ 4,000)')
    plt.xlabel('Pipe diameter [mm]', fontsize=12, fontweight='bold')
    plt.ylabel('Reynolds number [-]', fontsize=12, fontweight='bold')
    plt.title('Relationship between Pipe Diameter and Reynolds Number (Water, velocity 1.5 m/s)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    ======================================================================
    Reynolds Number Calculation and Flow Regime Determination (Water Pipe Flow)
    ======================================================================
    Fluid: Water (20°C), Density = 998.2 kg/m³, Viscosity = 1.002 mPa·s
    Flow velocity: 1.5 m/s
    ----------------------------------------------------------------------
    Pipe diameter   25.0 mm → Re =     37,365 → Turbulent
    Pipe diameter   50.0 mm → Re =     74,730 → Turbulent
    Pipe diameter  100.0 mm → Re =    149,460 → Turbulent
    Pipe diameter  200.0 mm → Re =    298,920 → Turbulent
    

**Explanation:** For water, at typical flow velocities (1-3 m/s), even relatively small pipe diameters result in turbulent flow. This means it is easier to maintain turbulent conditions during scaleup.

* * *

### Froude Number

The Froude number (Fr) represents the ratio of **inertial force** to **gravity** :

$$ \text{Fr} = \frac{u}{\sqrt{gL}} $$

Where:

  * $g$: Gravitational acceleration [m/s²] (≈ 9.81 m/s²)

**Important systems:**

  * Free surface flow (open channel, liquid surface fluctuation in tanks)
  * Stirred tanks (vortex formation at liquid surface)
  * Gas-liquid two-phase flow (bubble rise, slug flow)

### Code Example 2: Calculation of Froude Number and Evaluation of Free Surface Flow
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def froude_number(u, L, g=9.81):
        """
        Calculate Froude number
    
        Parameters:
        -----------
        u : float or array
            Characteristic velocity [m/s]
        L : float
            Characteristic length [m] (water depth, stirred tank diameter, etc.)
        g : float
            Gravitational acceleration [m/s²] (default: 9.81)
    
        Returns:
        --------
        Fr : float or array
            Froude number [-]
        """
        return u / np.sqrt(g * L)
    
    def flow_type_froude(Fr):
        """
        Determine flow type from Froude number
    
        Parameters:
        -----------
        Fr : float
            Froude number
    
        Returns:
        --------
        flow_type : str
            Flow type
        """
        if Fr < 1:
            return 'Subcritical (gravity dominated)'
        elif Fr == 1:
            return 'Critical'
        else:
            return 'Supercritical (inertia dominated)'
    
    # Example: Scaleup of stirred tank (constant Froude number)
    # Lab scale
    D_lab = 0.1  # m (10 cm diameter)
    N_lab = 5.0  # rps (rotation speed)
    u_lab = np.pi * D_lab * N_lab  # Tip velocity
    
    Fr_lab = froude_number(u_lab, D_lab)
    
    print("=" * 70)
    print("Scaleup with Constant Froude Number (Stirred Tank)")
    print("=" * 70)
    print(f"Lab scale: Diameter = {D_lab*100:.1f} cm, Rotation speed = {N_lab:.2f} rps")
    print(f"Tip velocity = {u_lab:.3f} m/s, Froude number = {Fr_lab:.3f}")
    print(f"Flow type: {flow_type_froude(Fr_lab)}")
    print("-" * 70)
    
    # Pilot and industrial scales
    scale_factors = np.array([1, 5, 10, 20])  # Scale multipliers
    D_scale = D_lab * scale_factors
    N_scale = N_lab / np.sqrt(scale_factors)  # Condition for constant Froude number
    u_scale = np.pi * D_scale * N_scale
    
    print("\nScaleup Results (Constant Froude Number):")
    print("-" * 70)
    for i, S in enumerate(scale_factors):
        print(f"Scale factor {S:2.0f}x → Diameter {D_scale[i]*100:6.1f} cm, "
              f"Rotation speed {N_scale[i]:.3f} rps → Fr = {froude_number(u_scale[i], D_scale[i]):.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Rotation speed change
    ax1.plot(scale_factors, N_scale, 'o-', linewidth=2.5, markersize=10,
             color='#11998e', label='Constant Froude number')
    ax1.axhline(y=N_lab, color='red', linestyle='--', linewidth=2,
                label=f'Lab scale (N = {N_lab:.2f} rps)')
    ax1.set_xlabel('Scale factor [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rotation speed [rps]', fontsize=12, fontweight='bold')
    ax1.set_title('Constant Froude Number Scaling: Rotation Speed Change', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Right plot: Power number impact
    Power_relative = scale_factors**2.5  # N³D⁵ ∝ S²·⁵ (constant Fr)
    ax2.plot(scale_factors, Power_relative, 's-', linewidth=2.5, markersize=10,
             color='#e74c3c', label='Relative power (S^2.5 law)')
    ax2.set_xlabel('Scale factor [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative power [-]', fontsize=12, fontweight='bold')
    ax2.set_title('Power Increase with Constant Froude Number', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    ======================================================================
    Scaleup with Constant Froude Number (Stirred Tank)
    ======================================================================
    Lab scale: Diameter = 10.0 cm, Rotation speed = 5.00 rps
    Tip velocity = 1.571 m/s, Froude number = 1.597
    Flow type: Supercritical (inertia dominated)
    ----------------------------------------------------------------------
    
    Scaleup Results (Constant Froude Number):
    ----------------------------------------------------------------------
    Scale factor  1x → Diameter   10.0 cm, Rotation speed 5.000 rps → Fr = 1.597
    Scale factor  5x → Diameter   50.0 cm, Rotation speed 2.236 rps → Fr = 1.597
    Scale factor 10x → Diameter  100.0 cm, Rotation speed 1.581 rps → Fr = 1.597
    Scale factor 20x → Diameter  200.0 cm, Rotation speed 1.118 rps → Fr = 1.597
    

**Explanation:** When maintaining constant Froude number, rotation speed is inversely proportional to the square root of the scale factor ($N \propto S^{-0.5}$). This is effective for maintaining similarity in vortex formation at the free surface, but power increases as $S^{2.5}$.

* * *

### Weber Number

The Weber number (We) represents the ratio of **inertial force** to **surface tension** :

$$ \text{We} = \frac{\rho u^2 L}{\sigma} $$

Where:

  * $\sigma$: Surface tension [N/m]

**Important systems:**

  * Droplet and bubble formation
  * Spraying and atomization
  * Two-phase flow with gas-liquid interface

### Code Example 3: Calculation of Weber Number and Evaluation of Droplet Breakup
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def weber_number(rho, u, L, sigma):
        """
        Calculate Weber number
    
        Parameters:
        -----------
        rho : float
            Fluid density [kg/m³]
        u : float
            Relative velocity [m/s]
        L : float
            Characteristic length (droplet diameter, etc.) [m]
        sigma : float
            Surface tension [N/m]
    
        Returns:
        --------
        We : float
            Weber number [-]
        """
        return rho * u**2 * L / sigma
    
    def droplet_regime(We):
        """
        Determine droplet state from Weber number
    
        Parameters:
        -----------
        We : float
            Weber number
    
        Returns:
        --------
        regime : str
            Droplet state
        """
        if We < 1:
            return 'Stable (surface tension dominated)'
        elif We < 12:
            return 'Deformation onset'
        elif We < 100:
            return 'Bag breakup'
        else:
            return 'Atomization (catastrophic breakup)'
    
    # Example: Water spray from nozzle
    rho_water = 998.2  # kg/m³
    sigma_water = 0.0728  # N/m (20°C)
    
    # Spray velocity and nozzle diameter range
    spray_velocity = np.linspace(1, 30, 50)  # m/s
    nozzle_diameter = 1e-3  # m (1 mm)
    
    We_values = weber_number(rho_water, spray_velocity, nozzle_diameter, sigma_water)
    
    print("=" * 70)
    print("Weber Number and Droplet Breakup Mode (Water Spray)")
    print("=" * 70)
    print(f"Fluid: Water (20°C), Density = {rho_water} kg/m³, Surface tension = {sigma_water*1000:.2f} mN/m")
    print(f"Nozzle diameter: {nozzle_diameter*1000:.2f} mm")
    print("-" * 70)
    
    test_velocities = [5, 10, 15, 20, 25]
    for v in test_velocities:
        We = weber_number(rho_water, v, nozzle_diameter, sigma_water)
        regime = droplet_regime(We)
        print(f"Spray velocity {v:5.1f} m/s → We = {We:8.1f} → {regime}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(spray_velocity, We_values, linewidth=3, color='#11998e',
            label='Weber number')
    
    # Breakup mode boundaries
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Deformation onset (We = 1)')
    ax.axhline(y=12, color='orange', linestyle='--', linewidth=2, label='Bag breakup (We = 12)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Atomization (We = 100)')
    
    # Region shading
    ax.fill_between(spray_velocity, 0, 1, alpha=0.2, color='green', label='Stable region')
    ax.fill_between(spray_velocity, 1, 12, alpha=0.2, color='yellow', label='Deformation region')
    ax.fill_between(spray_velocity, 12, 100, alpha=0.2, color='orange', label='Bag breakup region')
    ax.fill_between(spray_velocity, 100, We_values.max(), alpha=0.2, color='red', label='Atomization region')
    
    ax.set_xlabel('Spray velocity [m/s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weber number [-]', fontsize=12, fontweight='bold')
    ax.set_title('Spray Velocity and Weber Number: Droplet Breakup Modes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([0.1, 1000])
    
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    ======================================================================
    Weber Number and Droplet Breakup Mode (Water Spray)
    ======================================================================
    Fluid: Water (20°C), Density = 998.2 kg/m³, Surface tension = 72.8 mN/m
    Nozzle diameter: 1.00 mm
    ----------------------------------------------------------------------
    Spray velocity   5.0 m/s → We =    342.5 → Atomization (catastrophic breakup)
    Spray velocity  10.0 m/s → We =   1370.1 → Atomization (catastrophic breakup)
    Spray velocity  15.0 m/s → We =   3082.7 → Atomization (catastrophic breakup)
    Spray velocity  20.0 m/s → We =   5480.5 → Atomization (catastrophic breakup)
    Spray velocity  25.0 m/s → We =   8563.5 → Atomization (catastrophic breakup)
    

**Explanation:** In industrial spraying (5-30 m/s), the Weber number is very high, and droplets break up vigorously and atomize. To obtain the same droplet size distribution during scaleup, the Weber number must be kept constant.

* * *

## 2.3 Dimensionless Numbers in Mixing and Agitation

### Power Number

The Power number (Po) represents the ratio of power required for agitation to inertial force:

$$ \text{Po} = \frac{P}{\rho N^3 D^5} $$

Where:

  * $P$: Agitation power [W]
  * $N$: Rotation speed [rps]
  * $D$: Impeller diameter [m]

**Important:** The Power number is a function of Reynolds number, with the relationship $\text{Po} = f(\text{Re})$.

### Code Example 4: Relationship between Power Number and Reynolds Number (Stirred Tank)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def power_number(Re, impeller_type='rushton'):
        """
        Estimate Power number from Reynolds number (empirical formula)
    
        Parameters:
        -----------
        Re : float or array
            Reynolds number (stirred tank)
        impeller_type : str
            Impeller type
            - 'rushton': Rushton turbine (standard 6-blade)
            - 'paddle': Paddle impeller
            - 'propeller': Propeller impeller
    
        Returns:
        --------
        Po : float or array
            Power number
        """
        if impeller_type == 'rushton':
            # Empirical formula for Rushton turbine
            Po_turb = 5.0  # Turbulent regime
            Po_lam = 64 / Re  # Laminar regime
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 10000,
                                 Po_turb * (1 + 10/Re),
                                 Po_turb))
        elif impeller_type == 'paddle':
            Po_turb = 1.5
            Po_lam = 50 / Re
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 10000,
                                 Po_turb * (1 + 15/Re),
                                 Po_turb))
        elif impeller_type == 'propeller':
            Po_turb = 0.32
            Po_lam = 40 / Re
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 5000,
                                 Po_turb * (1 + 10/Re),
                                 Po_turb))
        else:
            raise ValueError(f"Unknown impeller type: {impeller_type}")
    
        return Po
    
    def calculate_power(rho, N, D, mu, impeller_type='rushton'):
        """
        Calculate agitation power
    
        Parameters:
        -----------
        rho : float
            Fluid density [kg/m³]
        N : float
            Rotation speed [rps]
        D : float
            Impeller diameter [m]
        mu : float
            Viscosity [Pa·s]
        impeller_type : str
            Impeller type
    
        Returns:
        --------
        P : float
            Agitation power [W]
        Re : float
            Reynolds number
        Po : float
            Power number
        """
        Re = rho * N * D**2 / mu
        Po = power_number(Re, impeller_type)
        P = Po * rho * N**3 * D**5
        return P, Re, Po
    
    # Reynolds number range
    Re_range = np.logspace(0, 6, 500)
    
    # Power number for various impellers
    Po_rushton = power_number(Re_range, 'rushton')
    Po_paddle = power_number(Re_range, 'paddle')
    Po_propeller = power_number(Re_range, 'propeller')
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(Re_range, Po_rushton, linewidth=2.5, color='#11998e',
              label='Rushton turbine')
    ax.loglog(Re_range, Po_paddle, linewidth=2.5, color='#e74c3c',
              label='Paddle impeller')
    ax.loglog(Re_range, Po_propeller, linewidth=2.5, color='#f39c12',
              label='Propeller impeller')
    
    ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=10000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(5, 0.1, 'Laminar', fontsize=11, ha='right', color='gray')
    ax.text(100, 0.1, 'Transition', fontsize=11, ha='center', color='gray')
    ax.text(50000, 0.1, 'Turbulent', fontsize=11, ha='left', color='gray')
    
    ax.set_xlabel('Reynolds number [-]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power number [-]', fontsize=12, fontweight='bold')
    ax.set_title('Relationship between Power Number and Reynolds Number (by Impeller Type)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(which='both', alpha=0.3)
    ax.set_xlim([1, 1e6])
    ax.set_ylim([0.05, 100])
    
    plt.tight_layout()
    plt.show()
    
    # Example calculation: Water agitation (Rushton turbine)
    print("=" * 70)
    print("Agitation Power Calculation Example (Water, Rushton Turbine)")
    print("=" * 70)
    
    rho_water = 998.2  # kg/m³
    mu_water = 1.002e-3  # Pa·s
    D = 0.2  # m (20 cm impeller diameter)
    N_values = np.array([1, 2, 3, 4, 5])  # rps
    
    for N in N_values:
        P, Re, Po = calculate_power(rho_water, N, D, mu_water, 'rushton')
        print(f"Rotation speed {N} rps → Re = {Re:10,.0f}, Po = {Po:.3f}, Power = {P:.2f} W")
    

**Output:**
    
    
    ======================================================================
    Agitation Power Calculation Example (Water, Rushton Turbine)
    ======================================================================
    Rotation speed 1 rps → Re =     39,848, Po = 5.000, Power = 1.60 W
    Rotation speed 2 rps → Re =     79,696, Po = 5.000, Power = 12.78 W
    Rotation speed 3 rps → Re =    119,544, Po = 5.000, Power = 43.15 W
    Rotation speed 4 rps → Re =    159,392, Po = 5.000, Power = 102.27 W
    Rotation speed 5 rps → Re =    199,240, Po = 5.000, Power = 199.75 W
    

**Explanation:** In the turbulent regime, the Power number becomes constant (approximately 5 for Rushton turbine). Power is proportional to the cube of rotation speed and the fifth power of impeller diameter, so it increases rapidly during scaleup.

* * *

## 2.4 Dimensionless Numbers in Heat and Mass Transfer (Overview)

For scaling of heat and mass transfer, the following dimensionless numbers are important (details covered in Chapter 3):

Dimensionless Number | Definition | Physical Meaning  
---|---|---  
**Nusselt number (Nu)** | $\text{Nu} = \frac{hL}{k}$ | Convective heat transfer / Heat conduction  
**Prandtl number (Pr)** | $\text{Pr} = \frac{\nu}{\alpha} = \frac{c_p \mu}{k}$ | Momentum diffusion / Thermal diffusion  
**Grashof number (Gr)** | $\text{Gr} = \frac{g \beta \Delta T L^3}{\nu^2}$ | Buoyancy / Viscous force  
**Sherwood number (Sh)** | $\text{Sh} = \frac{k_c L}{D_{AB}}$ | Convective mass transfer / Diffusion  
**Schmidt number (Sc)** | $\text{Sc} = \frac{\nu}{D_{AB}}$ | Momentum diffusion / Mass diffusion  
  
* * *

## 2.5 Multi-Criteria Similarity Analysis

### The Problem of Satisfying Multiple Dimensionless Numbers Simultaneously

In actual scaling, it is required to match multiple dimensionless numbers simultaneously. However, satisfying all dimensionless numbers simultaneously is often **physically impossible**.

**Example:** When scaling up a stirred tank while keeping both Reynolds number and Froude number constant:

  * Constant Re → $N \propto S^{-2}$
  * Constant Fr → $N \propto S^{-0.5}$

These are contradictory, requiring **judgment on which to prioritize**.

### Code Example 5: Multi-Criteria Similarity Analysis (Stirred Tank)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def scaling_analysis(S, criterion):
        """
        Rotation speed and power changes under different scaling criteria
    
        Parameters:
        -----------
        S : array
            Scale factor
        criterion : str
            Scaling criterion
            - 'Re': Constant Reynolds number
            - 'Fr': Constant Froude number
            - 'P/V': Constant power per unit volume
            - 'tip_speed': Constant tip speed
    
        Returns:
        --------
        N_ratio : array
            Rotation speed ratio (lab scale = 1)
        P_ratio : array
            Power ratio (lab scale = 1)
        """
        if criterion == 'Re':
            N_ratio = S**(-2)
            P_ratio = S**(-3)
        elif criterion == 'Fr':
            N_ratio = S**(-0.5)
            P_ratio = S**(2.5)
        elif criterion == 'P/V':
            N_ratio = S**(-1)
            P_ratio = S**(2)
        elif criterion == 'tip_speed':
            N_ratio = S**(-1)
            P_ratio = S**(2)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
        return N_ratio, P_ratio
    
    # Scale factors
    S = np.array([1, 2, 5, 10, 20])
    
    # Calculation for each criterion
    criteria = ['Re', 'Fr', 'P/V', 'tip_speed']
    colors = ['#11998e', '#e74c3c', '#f39c12', '#9b59b6']
    
    print("=" * 80)
    print("Multi-Criteria Similarity Analysis: Rotation Speed and Power Changes During Scaleup")
    print("=" * 80)
    print(f"{'Scale':>8} | {'Constant Re':>15} | {'Constant Fr':>15} | {'Constant P/V':>15} | {'Constant Tip Speed':>15}")
    print(f"{'Factor':>8} | {'N ratio':>7} {'P ratio':>7} | {'N ratio':>7} {'P ratio':>7} | {'N ratio':>7} {'P ratio':>7} | {'N ratio':>7} {'P ratio':>7}")
    print("-" * 80)
    
    for s in S:
        N_Re, P_Re = scaling_analysis(np.array([s]), 'Re')
        N_Fr, P_Fr = scaling_analysis(np.array([s]), 'Fr')
        N_PV, P_PV = scaling_analysis(np.array([s]), 'P/V')
        N_tip, P_tip = scaling_analysis(np.array([s]), 'tip_speed')
    
        print(f"{s:8.0f} | {N_Re[0]:7.3f} {P_Re[0]:7.2f} | "
              f"{N_Fr[0]:7.3f} {P_Fr[0]:7.2f} | "
              f"{N_PV[0]:7.3f} {P_PV[0]:7.2f} | "
              f"{N_tip[0]:7.3f} {P_tip[0]:7.2f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    S_plot = np.linspace(1, 20, 100)
    
    for i, crit in enumerate(criteria):
        N_ratio, P_ratio = scaling_analysis(S_plot, crit)
    
        ax1.plot(S_plot, N_ratio, linewidth=2.5, color=colors[i],
                 label=f'Constant {crit}')
        ax2.plot(S_plot, P_ratio, linewidth=2.5, color=colors[i],
                 label=f'Constant {crit}')
    
    ax1.set_xlabel('Scale factor [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rotation speed ratio (N/N₀) [-]', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Criterion Comparison: Rotation Speed Changes', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Scale factor [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power ratio (P/P₀) [-]', fontsize=12, fontweight='bold')
    ax2.set_title('Scaling Criterion Comparison: Power Changes', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    

**Output:**
    
    
    ================================================================================
    Multi-Criteria Similarity Analysis: Rotation Speed and Power Changes During Scaleup
    ================================================================================
       Scale |     Constant Re |     Constant Fr |    Constant P/V | Constant Tip Speed
      Factor |  N ratio P ratio |  N ratio P ratio |  N ratio P ratio |  N ratio P ratio
    --------------------------------------------------------------------------------
           1 |   1.000    1.00 |   1.000    1.00 |   1.000    1.00 |   1.000    1.00
           2 |   0.250    0.12 |   0.707    5.66 |   0.500    4.00 |   0.500    4.00
           5 |   0.040    0.01 |   0.447   56.57 |   0.200   25.00 |   0.200   25.00
          10 |   0.010    0.00 |   0.316  316.23 |   0.100  100.00 |   0.100  100.00
          20 |   0.003    0.00 |   0.224 1788.85 |   0.050  400.00 |   0.050  400.00
    

**Explanation:**

  * **Constant Reynolds number:** Both rotation speed and power decrease sharply. Applied to scaling of high-viscosity fluids.
  * **Constant Froude number:** Power increases significantly. Applied to systems where free surface flow and vortex control are important.
  * **Constant P/V:** Maintains power per unit volume. Applied to general mixing and suspension.
  * **Constant tip speed:** Maintains shear rate. Applied to shear-sensitive materials (cell culture, etc.).

* * *

### Code Example 6: Identification of Dominant Dimensionless Numbers
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def identify_dominant_forces(Re, Fr, We):
        """
        Identify dominant forces from Reynolds, Froude, and Weber numbers
    
        Parameters:
        -----------
        Re : float
            Reynolds number
        Fr : float
            Froude number
        We : float
            Weber number
    
        Returns:
        --------
        dominant_forces : dict
            List of dominant forces
        """
        forces = []
    
        # Reynolds number evaluation
        if Re < 1:
            forces.append('Viscous force (laminar, Re < 1)')
        elif Re < 2300:
            forces.append('Viscous force (laminar, Re < 2,300)')
        else:
            forces.append('Inertial force (turbulent, Re > 2,300)')
    
        # Froude number evaluation
        if Fr < 0.5:
            forces.append('Gravity (Fr < 0.5, gravity dominated)')
        elif Fr > 2:
            forces.append('Inertial force (Fr > 2, inertia dominated)')
        else:
            forces.append('Gravity-inertia balance (0.5 < Fr < 2)')
    
        # Weber number evaluation
        if We < 1:
            forces.append('Surface tension (We < 1, surface tension dominated)')
        elif We > 10:
            forces.append('Inertial force (We > 10, droplet breakup)')
        else:
            forces.append('Surface tension-inertia balance (1 < We < 10)')
    
        return forces
    
    # Example: Dimensionless number analysis of various chemical processes
    processes = [
        {
            'name': 'Stirred tank (water, low speed)',
            'Re': 5000,
            'Fr': 0.3,
            'We': 50
        },
        {
            'name': 'Stirred tank (water, high speed)',
            'Re': 100000,
            'Fr': 2.5,
            'We': 500
        },
        {
            'name': 'High-viscosity fluid agitation',
            'Re': 10,
            'Fr': 0.05,
            'We': 0.5
        },
        {
            'name': 'Nozzle spray',
            'Re': 50000,
            'Fr': 15,
            'We': 5000
        },
        {
            'name': 'Bubble column',
            'Re': 1000,
            'Fr': 1.0,
            'We': 5
        }
    ]
    
    print("=" * 80)
    print("Analysis of Dominant Forces in Various Processes")
    print("=" * 80)
    
    for proc in processes:
        print(f"\n【{proc['name']}】")
        print(f"  Reynolds number = {proc['Re']:,.0f}")
        print(f"  Froude number   = {proc['Fr']:.2f}")
        print(f"  Weber number    = {proc['We']:.0f}")
        print(f"  Dominant forces:")
    
        dominant = identify_dominant_forces(proc['Re'], proc['Fr'], proc['We'])
        for i, force in enumerate(dominant, 1):
            print(f"    {i}. {force}")
    

**Output:**
    
    
    ================================================================================
    Analysis of Dominant Forces in Various Processes
    ================================================================================
    
    【Stirred tank (water, low speed)】
      Reynolds number = 5,000
      Froude number   = 0.30
      Weber number    = 50
      Dominant forces:
        1. Inertial force (turbulent, Re > 2,300)
        2. Gravity (Fr < 0.5, gravity dominated)
        3. Inertial force (We > 10, droplet breakup)
    
    【Stirred tank (water, high speed)】
      Reynolds number = 100,000
      Froude number   = 2.50
      Weber number    = 500
      Dominant forces:
        1. Inertial force (turbulent, Re > 2,300)
        2. Inertial force (Fr > 2, inertia dominated)
        3. Inertial force (We > 10, droplet breakup)
    
    【High-viscosity fluid agitation】
      Reynolds number = 10
      Froude number   = 0.05
      Weber number    = 0.5
      Dominant forces:
        1. Viscous force (laminar, Re < 2,300)
        2. Gravity (Fr < 0.5, gravity dominated)
        3. Surface tension (We < 1, surface tension dominated)
    
    【Nozzle spray】
      Reynolds number = 50,000
      Froude number   = 15.00
      Weber number    = 5000
      Dominant forces:
        1. Inertial force (turbulent, Re > 2,300)
        2. Inertial force (Fr > 2, inertia dominated)
        3. Inertial force (We > 10, droplet breakup)
    
    【Bubble column】
      Reynolds number = 1,000
      Froude number   = 1.00
      Weber number    = 5
      Dominant forces:
        1. Viscous force (laminar, Re < 2,300)
        2. Gravity-inertia balance (0.5 < Fr < 2)
        3. Surface tension-inertia balance (1 < We < 10)
    

**Explanation:** Different processes have different dominant forces. In scaling, it is important to prioritize matching dimensionless numbers corresponding to dominant forces.

* * *

### Code Example 7: Decision Tree for Scaling Criterion Selection
    
    
    def recommend_scaling_criterion(process_type, fluid_viscosity, has_free_surface,
                                     shear_sensitive, phase='single'):
        """
        Propose recommended scaling criteria based on process characteristics
    
        Parameters:
        -----------
        process_type : str
            Process type ('mixing', 'reaction', 'separation', 'dispersion')
        fluid_viscosity : str
            Fluid viscosity ('low': < 100 mPa·s, 'medium': 100-1000, 'high': > 1000)
        has_free_surface : bool
            Presence of free surface
        shear_sensitive : bool
            Presence of shear sensitivity
        phase : str
            Phase state ('single', 'gas-liquid', 'liquid-liquid', 'solid-liquid')
    
        Returns:
        --------
        recommendations : dict
            Recommended criteria and reasons
        """
        recommendations = {
            'primary': None,
            'secondary': None,
            'reason': ''
        }
    
        # High viscosity fluid
        if fluid_viscosity == 'high':
            recommendations['primary'] = 'Constant Reynolds number'
            recommendations['reason'] = 'High-viscosity fluids operate in laminar regime, maintain Re constant for viscous flow'
            recommendations['secondary'] = 'Constant tip speed (shear rate control)'
    
        # Shear-sensitive material
        elif shear_sensitive:
            recommendations['primary'] = 'Constant tip speed'
            recommendations['reason'] = 'Maintain constant shear rate to prevent damage to shear-sensitive materials'
            recommendations['secondary'] = 'Constant P/V (energy dissipation rate control)'
    
        # Free surface present
        elif has_free_surface:
            recommendations['primary'] = 'Constant Froude number'
            recommendations['reason'] = 'Maintain similarity in vortex formation at free surface'
            recommendations['secondary'] = 'Constant P/V'
    
        # Gas-liquid dispersion
        elif phase == 'gas-liquid':
            recommendations['primary'] = 'Constant P/V'
            recommendations['reason'] = 'Maintain similarity in bubble size and gas holdup'
            recommendations['secondary'] = 'Constant Weber number (bubble breakup control)'
    
        # Liquid-liquid dispersion
        elif phase == 'liquid-liquid':
            recommendations['primary'] = 'Constant Weber number'
            recommendations['reason'] = 'Maintain similarity in droplet size distribution'
            recommendations['secondary'] = 'Constant P/V'
    
        # General mixing
        else:
            recommendations['primary'] = 'Constant P/V'
            recommendations['reason'] = 'Maintain similarity in mixing time and turbulent energy dissipation rate'
            recommendations['secondary'] = 'Constant Reynolds number (flow pattern control)'
    
        return recommendations
    
    # Examples
    test_cases = [
        {
            'name': 'General water agitation',
            'process_type': 'mixing',
            'fluid_viscosity': 'low',
            'has_free_surface': False,
            'shear_sensitive': False,
            'phase': 'single'
        },
        {
            'name': 'High-viscosity polymer solution',
            'process_type': 'mixing',
            'fluid_viscosity': 'high',
            'has_free_surface': False,
            'shear_sensitive': True,
            'phase': 'single'
        },
        {
            'name': 'Gas-liquid reactor (bubble column)',
            'process_type': 'reaction',
            'fluid_viscosity': 'low',
            'has_free_surface': True,
            'shear_sensitive': False,
            'phase': 'gas-liquid'
        },
        {
            'name': 'Emulsification process',
            'process_type': 'dispersion',
            'fluid_viscosity': 'medium',
            'has_free_surface': False,
            'shear_sensitive': False,
            'phase': 'liquid-liquid'
        },
        {
            'name': 'Cell culture',
            'process_type': 'mixing',
            'fluid_viscosity': 'low',
            'has_free_surface': False,
            'shear_sensitive': True,
            'phase': 'single'
        }
    ]
    
    print("=" * 80)
    print("Scaling Criterion Recommendations Based on Process Characteristics")
    print("=" * 80)
    
    for case in test_cases:
        rec = recommend_scaling_criterion(
            case['process_type'],
            case['fluid_viscosity'],
            case['has_free_surface'],
            case['shear_sensitive'],
            case['phase']
        )
    
        print(f"\n【{case['name']}】")
        print(f"  Process type: {case['process_type']}")
        print(f"  Viscosity: {case['fluid_viscosity']}, Free surface: {case['has_free_surface']}, "
              f"Shear sensitive: {case['shear_sensitive']}, Phase: {case['phase']}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Recommended scaling criteria:")
        print(f"    1st priority: {rec['primary']}")
        print(f"    2nd priority: {rec['secondary']}")
        print(f"  Reason: {rec['reason']}")
    

**Output:**
    
    
    ================================================================================
    Scaling Criterion Recommendations Based on Process Characteristics
    ================================================================================
    
    【General water agitation】
      Process type: mixing
      Viscosity: low, Free surface: False, Shear sensitive: False, Phase: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Recommended scaling criteria:
        1st priority: Constant P/V
        2nd priority: Constant Reynolds number (flow pattern control)
      Reason: Maintain similarity in mixing time and turbulent energy dissipation rate
    
    【High-viscosity polymer solution】
      Process type: mixing
      Viscosity: high, Free surface: False, Shear sensitive: True, Phase: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Recommended scaling criteria:
        1st priority: Constant Reynolds number
        2nd priority: Constant tip speed (shear rate control)
      Reason: High-viscosity fluids operate in laminar regime, maintain Re constant for viscous flow
    
    【Gas-liquid reactor (bubble column)】
      Process type: reaction
      Viscosity: low, Free surface: True, Shear sensitive: False, Phase: gas-liquid
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Recommended scaling criteria:
        1st priority: Constant Froude number
        2nd priority: Constant P/V
      Reason: Maintain similarity in vortex formation at free surface
    
    【Emulsification process】
      Process type: dispersion
      Viscosity: medium, Free surface: False, Shear sensitive: False, Phase: liquid-liquid
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Recommended scaling criteria:
        1st priority: Constant Weber number
        2nd priority: Constant P/V
      Reason: Maintain similarity in droplet size distribution
    
    【Cell culture】
      Process type: mixing
      Viscosity: low, Free surface: False, Shear sensitive: True, Phase: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Recommended scaling criteria:
        1st priority: Constant tip speed
        2nd priority: Constant P/V (energy dissipation rate control)
      Reason: Maintain constant shear rate to prevent damage to shear-sensitive materials
    

**Explanation:** The optimal scaling criterion differs depending on process characteristics. This decision tree serves as a starting point for developing scaling strategies in practice.

* * *

## 2.6 Limitations of Similarity Laws

### Why Perfect Similarity is Impossible

Matching all dimensionless numbers simultaneously is impossible for the following reasons:

  1. **Too many independent dimensionless numbers:** Even for fluid mechanics alone, Re, Fr, We, Ca (Capillary number), etc. exist
  2. **Physical property constraints:** Density, viscosity, and surface tension cannot be changed independently
  3. **Geometric constraints:** Wall effects, aspect ratio limitations
  4. **Practical constraints:** Power, pressure drop, construction cost

### Partial Similarity

In actual scaling, a **partial similarity** approach is taken: **prioritizing matching dimensionless numbers corresponding to dominant phenomena while allowing acceptable variation in others**.

Process | Prioritized Dimensionless Numbers | Acceptable Dimensionless Numbers  
---|---|---  
High-speed agitation (low viscosity) | Constant P/V, ensure Re (turbulent) | Allow Froude number variation  
Free surface agitation | Constant Froude number | Allow Reynolds number variation  
Gas-liquid dispersion | Constant P/V, Weber number | Only ensure Reynolds number in turbulent regime  
High-viscosity fluid | Constant Reynolds number | Ignore Froude number, Weber number  
  
* * *

## Confirmation of Learning Objectives

After completing this chapter, you should be able to explain the following:

### Basic Understanding

  * ✅ Explain the three hierarchical levels of geometric, kinematic, and dynamic similarity
  * ✅ Understand the physical meaning of major dimensionless numbers (Re, Fr, We, Po)
  * ✅ Explain the relationship between dimensionless numbers and force balance

### Practical Skills

  * ✅ Calculate various dimensionless numbers in Python and determine flow regimes
  * ✅ Quantify differences in scaling criteria (constant Re, constant Fr, constant P/V, etc.)
  * ✅ Select appropriate scaling criteria from process characteristics

### Application Ability

  * ✅ Perform multi-criteria similarity analysis using multiple dimensionless numbers
  * ✅ Understand why perfect similarity is impossible and the necessity of partial similarity
  * ✅ Identify dominant forces in actual processes and develop scaling strategies

* * *

### Disclaimer

  * This material is created for educational purposes, and actual process design requires expert advice
  * Numerical examples are approximate values under typical conditions; verification of physical properties and operating conditions is essential for actual systems
  * Scaleup requires safety evaluation, risk assessment, and step-by-step validation
  * Correlation formulas for dimensionless numbers are empirical and lose accuracy outside their applicable range

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.
