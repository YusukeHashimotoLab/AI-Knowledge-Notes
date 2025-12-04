---
title: "Chapter 3: Surface Treatment Technologies"
chapter_title: "Chapter 3: Surface Treatment Technologies"
subtitle: Electroplating, Anodizing, Surface Modification, Coating Technologies
difficulty: Intermediate
version: 1.0
---

This chapter covers Surface Treatment Technologies. You will learn Calculate plating thickness using Faraday's law, relationship between voltage, and coating technology selection criteria.

## Learning Objectives

Upon completing this chapter, you will be able to:

  * ✅ Calculate plating thickness using Faraday's law and optimize current density
  * ✅ Understand the relationship between voltage and film thickness in anodization processes and design anodizing treatments
  * ✅ Model ion implantation concentration profiles using Gaussian distributions
  * ✅ Understand coating technology selection criteria and choose appropriate methods
  * ✅ Evaluate the relationship between particle velocity/temperature and adhesion in thermal spray processes
  * ✅ Optimize surface treatment process parameters and perform troubleshooting

## 3.1 Electroplating

### 3.1.1 Faraday's Law and Electrochemical Fundamentals

Electroplating is a process of reducing and depositing metal ions on the cathode (workpiece) surface through electrolysis. Plating rate and film thickness follow Faraday's law.

**Faraday's First Law** : Deposited metal mass is proportional to the amount of charge passed

$$ m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta $$ 

Where:

  * $m$: Deposited mass [g]
  * $M$: Atomic weight of metal [g/mol]
  * $I$: Current [A]
  * $t$: Plating time [s]
  * $n$: Number of electrons (e.g., 2 for Cu²⁺)
  * $F$: Faraday constant (96485 C/mol)
  * $\eta$: Current efficiency (typically 0.85-0.98)

Plating thickness $d$ [μm] is calculated from deposited mass and density:

$$ d = \frac{m}{\rho \cdot A} \times 10^4 $$ 

$\rho$: Metal density [g/cm³], $A$: Plating area [cm²]

**Effect of Current Density** :

  * **Low current density** (0.5-2 A/dm²): Smooth, dense film, slow deposition
  * **High current density** (5-20 A/dm²): Rough film, dendritic growth, fast deposition

**Throwing Power** :

For complex-shaped parts, current density distribution becomes non-uniform, causing variations in film thickness. Throwing power is improved by bath composition, additives, and agitation.

#### Code Example 3.1: Plating Thickness Calculation Using Faraday's Law
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_plating_thickness(current_A, time_s, area_cm2,
                                     metal='Cu', efficiency=0.95):
        """
        Plating thickness calculation using Faraday's law
    
        Parameters:
        -----------
        current_A : float
            Current [A]
        time_s : float
            Plating time [s]
        area_cm2 : float
            Plating area [cm²]
        metal : str
            Metal type ('Cu', 'Ni', 'Cr', 'Au', 'Ag')
        efficiency : float
            Current efficiency (0-1)
    
        Returns:
        --------
        thickness_um : float
            Plating thickness [μm]
        """
        # Metal property database
        metal_data = {
            'Cu': {'M': 63.55, 'n': 2, 'rho': 8.96},   # Copper
            'Ni': {'M': 58.69, 'n': 2, 'rho': 8.91},   # Nickel
            'Cr': {'M': 52.00, 'n': 3, 'rho': 7.19},   # Chromium
            'Au': {'M': 196.97, 'n': 1, 'rho': 19.32}, # Gold
            'Ag': {'M': 107.87, 'n': 1, 'rho': 10.49}  # Silver
        }
    
        F = 96485  # Faraday constant [C/mol]
    
        props = metal_data[metal]
        M = props['M']
        n = props['n']
        rho = props['rho']
    
        # Deposited mass [g]
        mass_g = (M * current_A * time_s * efficiency) / (n * F)
    
        # Plating thickness [μm]
        thickness_um = (mass_g / (rho * area_cm2)) * 1e4
    
        return thickness_um
    
    # Example: Copper plating
    current = 2.0      # 2A
    time_hour = 1.0   # 1 hour
    time_s = time_hour * 3600
    area = 100.0       # 100cm²
    
    thickness = calculate_plating_thickness(current, time_s, area,
                                             metal='Cu', efficiency=0.95)
    
    print(f"=== Copper Plating Process Calculation ===")
    print(f"Current: {current} A")
    print(f"Current density: {current/area*100:.2f} A/dm²")
    print(f"Plating time: {time_hour} hour")
    print(f"Plating area: {area} cm²")
    print(f"Current efficiency: 95%")
    print(f"➡ Plating thickness: {thickness:.2f} μm")
    
    # Plot: Plating time vs thickness
    time_range = np.linspace(0, 2, 100) * 3600  # 0-2 hour
    thicknesses = [calculate_plating_thickness(current, t, area, 'Cu', 0.95)
                   for t in time_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_range/3600, thicknesses, linewidth=2, color='#f5576c')
    plt.xlabel('Plating Time [hour]', fontsize=12)
    plt.ylabel('Plating Thickness [μm]', fontsize=12)
    plt.title('Copper Plating: Relationship Between Time and Thickness', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Effect of current density
    current_densities = np.linspace(0.5, 10, 50)  # 0.5-10 A/dm²
    area_dm2 = area / 100  # cm² → dm²
    time_fixed = 3600  # 1 hour
    
    thicknesses_cd = []
    for cd in current_densities:
        I = cd * area_dm2
        thick = calculate_plating_thickness(I, time_fixed, area, 'Cu', 0.95)
        thicknesses_cd.append(thick)
    
    plt.figure(figsize=(10, 6))
    plt.plot(current_densities, thicknesses_cd, linewidth=2, color='#f093fb')
    plt.axvspan(0.5, 2, alpha=0.2, color='green', label='Low current density (smooth)')
    plt.axvspan(5, 10, alpha=0.2, color='red', label='High current density (rough)')
    plt.xlabel('Current Density [A/dm²]', fontsize=12)
    plt.ylabel('Plating Thickness [μm]', fontsize=12)
    plt.title('Relationship Between Current Density and Plating Thickness (1 hour)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.1.2 Plating Bath and Additives

The composition of the plating bath has a decisive influence on film quality.

Bath Component | Role | Typical Concentration  
---|---|---  
Metal salt (e.g., CuSO₄) | Metal ion supply | 200-250 g/L  
Conducting salt (e.g., H₂SO₄) | Conductivity enhancement | 50-80 g/L  
Brightener | Smoothing, gloss imparting | Several ppm to several hundred ppm  
Leveling agent | Unevenness flattening | Several ppm to tens of ppm  
Surfactant | Hydrogen gas release promotion | Several ppm  
  
#### Code Example 3.2: Current Distribution Simulation (2D Electrode)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import laplace
    
    def simulate_current_distribution_2d(width=50, height=50,
                                          anode_position='top',
                                          cathode_position='bottom',
                                          iterations=500):
        """
        Current density distribution simulation for 2D electrode configuration
        Solve Laplace's equation using finite difference method
    
        Parameters:
        -----------
        width, height : int
            Calculation grid size
        anode_position : str
            Anode position ('top', 'bottom', 'left', 'right')
        cathode_position : str
            Cathode position ('top', 'bottom', 'left', 'right')
        iterations : int
            Number of iterations
        """
        # Initialize potential distribution
        phi = np.zeros((height, width))
    
        # Set boundary conditions
        if anode_position == 'top':
            phi[0, :] = 1.0  # Anode potential
        elif anode_position == 'bottom':
            phi[-1, :] = 1.0
        elif anode_position == 'left':
            phi[:, 0] = 1.0
        elif anode_position == 'right':
            phi[:, -1] = 1.0
    
        if cathode_position == 'top':
            phi[0, :] = 0.0  # Cathode potential
        elif cathode_position == 'bottom':
            phi[-1, :] = 0.0
        elif cathode_position == 'left':
            phi[:, 0] = 0.0
        elif cathode_position == 'right':
            phi[:, -1] = 0.0
    
        # Solve Laplace's equation iteratively (∇²φ = 0)
        for _ in range(iterations):
            phi_new = phi.copy()
            phi_new[1:-1, 1:-1] = (phi[:-2, 1:-1] + phi[2:, 1:-1] +
                                   phi[1:-1, :-2] + phi[1:-1, 2:]) / 4.0
    
            # Reapply boundary conditions
            if anode_position == 'top':
                phi_new[0, :] = 1.0
            elif anode_position == 'bottom':
                phi_new[-1, :] = 1.0
    
            if cathode_position == 'bottom':
                phi_new[-1, :] = 0.0
            elif cathode_position == 'top':
                phi_new[0, :] = 0.0
    
            phi = phi_new
    
        # Current density = -∇φ (proportional to potential gradient)
        grad_y, grad_x = np.gradient(phi)
        current_density = np.sqrt(grad_x**2 + grad_y**2)
    
        return phi, current_density
    
    # Example: Top anode, bottom cathode
    phi, j = simulate_current_distribution_2d(width=50, height=50,
                                               anode_position='top',
                                               cathode_position='bottom')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Potential distribution
    im1 = axes[0].imshow(phi, cmap='viridis', origin='lower')
    axes[0].set_title('Potential Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[0], label='Potential [V]')
    
    # Current density distribution
    im2 = axes[1].imshow(j, cmap='hot', origin='lower')
    axes[1].set_title('Current Density Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1], label='Current Density [a.u.]')
    
    # Current density distribution on cathode surface
    cathode_j = j[-1, :]  # Bottom edge (cathode)
    axes[2].plot(cathode_j, linewidth=2, color='#f5576c')
    axes[2].set_title('Current Density on Cathode Surface', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X Position')
    axes[2].set_ylabel('Current Density [a.u.]')
    axes[2].grid(True, alpha=0.3)
    
    # Uniformity evaluation
    uniformity = (1 - (cathode_j.std() / cathode_j.mean())) * 100
    axes[2].text(0.5, 0.95, f'Uniformity: {uniformity:.1f}%',
                 transform=axes[2].transAxes,
                 ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Current density uniformity: {uniformity:.2f}%")
    print(f"Max/Min current density ratio: {cathode_j.max()/cathode_j.min():.2f}")
    

## 3.2 Anodizing

### 3.2.1 Principles of Aluminum Anodizing

Anodizing is a process of electrochemically oxidizing a metal surface to form an oxide film. Aluminum anodizing (anodization) is a representative example.

**Anodizing Process** :

  1. Immerse aluminum as anode and platinum as cathode in an electrolyte (sulfuric acid, oxalic acid, etc.)
  2. Apply DC voltage to grow Al₂O₃ film on Al surface
  3. Film has porous structure (barrier layer + porous layer)

    
    
    ```mermaid
    flowchart TB
        subgraph "Anodizing Cell"
            A[Aluminum Anode]
            B[ElectrolyteSulfuric/Oxalic Acid]
            C[Platinum Cathode]
            D[DC Power Supply]
        end
    
        D -->|Voltage Applied| A
        D --> C
        A -->|Al³⁺| B
        B -->|O²⁻| A
        A -->|Al₂O₃ Formation| E[Oxide Film]
    
        E --> F[Barrier LayerDense, Thin]
        E --> G[Porous LayerPorous, Thick]
    
        G --> H[Sealing TreatmentHot Water/Steam]
        H --> I[Final FilmImproved Corrosion Resistance]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
    ```

**Relationship Between Film Thickness and Voltage** :

For sulfuric acid baths, the barrier layer thickness is approximately proportional to the applied voltage (empirical rule):

$$ d_{\text{barrier}} \approx 1.4 \, [\text{nm/V}] \times V $$ 

The total film thickness (barrier layer + porous layer) depends on plating time and current density.

#### Code Example 3.3: Anodizing Film Thickness vs Voltage Relationship
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def anodization_thickness(voltage, material='Al',
                              electrolyte='H2SO4', time_min=30):
        """
        Anodizing film thickness calculation
    
        Parameters:
        -----------
        voltage : float or array
            Applied voltage [V]
        material : str
            Material ('Al', 'Ti')
        electrolyte : str
            Electrolyte ('H2SO4', 'H2C2O4')
        time_min : float
            Treatment time [min]
    
        Returns:
        --------
        barrier_thickness : float
            Barrier layer thickness [nm]
        total_thickness : float
            Total thickness [μm]
        """
        # Constants for material and electrolyte
        if material == 'Al':
            if electrolyte == 'H2SO4':
                k_barrier = 1.4  # nm/V (sulfuric acid bath)
                k_porous = 0.3   # μm/min at 1.5 A/dm²
            elif electrolyte == 'H2C2O4':
                k_barrier = 1.0  # nm/V (oxalic acid bath)
                k_porous = 0.5   # μm/min
        elif material == 'Ti':
            k_barrier = 2.5  # nm/V (TiO₂)
            k_porous = 0.2   # μm/min
    
        # Barrier layer thickness [nm]
        barrier_thickness = k_barrier * voltage
    
        # Porous layer thickness [μm] (simple model)
        porous_thickness = k_porous * time_min
    
        # Total thickness [μm]
        total_thickness = (barrier_thickness / 1000) + porous_thickness
    
        return barrier_thickness, total_thickness
    # Scan voltage range
    voltages = np.linspace(10, 100, 100)
    barrier_thicknesses = []
    total_thicknesses = []
    
    for V in voltages:
        d_barrier, d_total = anodization_thickness(V, 'Al', 'H2SO4', 30)
        barrier_thicknesses.append(d_barrier)
        total_thicknesses.append(d_total)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Barrier layer thickness vs Voltage
    axes[0].plot(voltages, barrier_thicknesses, linewidth=2,
                 color='#f5576c', label='Barrier layer')
    axes[0].set_xlabel('Applied Voltage [V]', fontsize=12)
    axes[0].set_ylabel('Barrier layer thickness [nm]', fontsize=12)
    axes[0].set_title('Barrier layer thicknessvs Voltage (Al/Sulfuric acid bath)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Total thickness vs Voltage
    axes[1].plot(voltages, total_thicknesses, linewidth=2,
                 color='#f093fb', label='Total thickness (30min)')
    axes[1].set_xlabel('Applied Voltage [V]', fontsize=12)
    axes[1].set_ylabel('Total thickness [μm]', fontsize=12)
    axes[1].set_title('Total thicknessvs Voltage (Al/Sulfuric acid bath)',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Design example：50nm Barrier layeris required
    target_barrier = 50  # nm
    required_voltage = target_barrier / 1.4
    print(f"=== Anodizing Process Design ===")
    print(f"targetBarrier layer thickness: {target_barrier} nm")
    print(f"➡ Required voltage: {required_voltage:.1f} V")
    
    # Effect of time
    times = np.linspace(10, 60, 50)  # 10-60min
    total_thicknesses_time = []
    for t in times:
        _, d_total = anodization_thickness(50, 'Al', 'H2SO4', t)
        total_thicknesses_time.append(d_total)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_thicknesses_time, linewidth=2, color='#f5576c')
    plt.xlabel('Treatment Time [min]', fontsize=12)
    plt.ylabel('Total thickness [μm]', fontsize=12)
    plt.title('Anodizingfilm thicknessvs Treatment Time (50V, Sulfuric acid bath)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.2.2 Sealing Treatment (Sealing)

Post-treatment to close the pores in the porous layer and improve corrosion resistance。

  * **Hot water sealing** ：95-100°C pure water with 30-60min、Al(OH)₃ closes the pores
  * **Steam sealing** ：110°C steam with 10-30min
  * **Cold sealing** ：Nickel salt solution with room temperature (energy-saving)

## 3.3 Surface Modification Technologies

### 3.3.1 Ion Implantation (Ion Implantation)

Ion Implantation is a technique that bombards the material surface with high-energy ions to modify chemical composition and crystal structure。It is used for doping in semiconductor manufacturing and surface hardening of metals。

**Ion Implantationprocess** ：

  1. Ion generation at ion source (e.g., N⁺, B⁺, P⁺)
  2. Acceleration to10-200 keVin acceleration field
  3. Selection of desired ions by mass analyzer
  4. Irradiation of sample in vacuum chamber

**Concentration Profile (LSStheory)** ：

The concentration distribution after ion implantation is approximated by a Gaussian distribution：

$$ C(x) = \frac{\Phi}{\sqrt{2\pi} \Delta R_p} \exp\left(-\frac{(x - R_p)^2}{2 \Delta R_p^2}\right) $$ 

  * $C(x)$: Depth $x$ at depth [atoms/cm³]
  * $\Phi$: Dose (total ions/area) [ions/cm²]
  * $R_p$: Range (peak depth) [nm]
  * $\Delta R_p$: Range straggling (standard deviation) [nm]

#### Code Example 3.4: Ion Implantation Concentration Profile (Gaussian LSS Theory)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    
    def ion_implantation_profile(energy_keV, dose_cm2, ion='N',
                                  substrate='Si', depth_range=None):
        """
        Ion Implantation Concentration Profile Calculation (Gaussian Approximation)
    
        Parameters:
        -----------
        energy_keV : float
            Ion energy [keV]
        dose_cm2 : float
            Dose [ions/cm²]
        ion : str
            Ion species ('N', 'B', 'P', 'As')
        substrate : str
            Substrate material ('Si', 'Fe', 'Ti')
        depth_range : array
            Depth range [nm] (auto-set if None)
    
        Returns:
        --------
        depth : array
            Depth [nm]
        concentration : array
            Concentration [atoms/cm³]
        """
        # Simplified LSS theory parameters (empirical formulas)
        # In practice, use simulation tools like SRIM/TRIM
    
        # Ion mass
        ion_masses = {'N': 14, 'B': 11, 'P': 31, 'As': 75}
        M_ion = ion_masses[ion]
    
        # Substrate density and atomic weight
        substrate_data = {
            'Si': {'rho': 2.33, 'M': 28},
            'Fe': {'rho': 7.87, 'M': 56},
            'Ti': {'rho': 4.51, 'M': 48}
        }
        rho_sub = substrate_data[substrate]['rho']
        M_sub = substrate_data[substrate]['M']
    
        # Range Rp [nm] (simplified formula)
        Rp = 10 * energy_keV**0.7 * (M_sub / M_ion)**0.5
    
        # Range straggling ΔRp [nm]
        delta_Rp = 0.3 * Rp
    
        if depth_range is None:
            depth_range = np.linspace(0, 3 * Rp, 500)
    
        # Gaussian concentration distribution
        concentration = (dose_cm2 / (np.sqrt(2 * np.pi) * delta_Rp)) * \
                        np.exp(-(depth_range - Rp)**2 / (2 * delta_Rp**2))
    
        return depth_range, concentration, Rp, delta_Rp
    
    # Example: Nitrogen ion implantation into silicon
    energy = 50  # keV
    dose = 1e16  # ions/cm²
    
    depth, conc, Rp, delta_Rp = ion_implantation_profile(
        energy, dose, ion='N', substrate='Si'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(depth, conc, linewidth=2, color='#f5576c', label=f'{energy} keV, {dose:.0e} ions/cm²')
    plt.axvline(Rp, color='gray', linestyle='--', alpha=0.7, label=f'Rp = {Rp:.1f} nm')
    plt.axvspan(Rp - delta_Rp, Rp + delta_Rp, alpha=0.2, color='orange',
                label=f'ΔRp = {delta_Rp:.1f} nm')
    plt.xlabel('Depth [nm]', fontsize=12)
    plt.ylabel('Concentration [atoms/cm³]', fontsize=12)
    plt.title('Ion ImplantationConcentration Profile (N⁺ → Si)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Energy dependence
    energies = [30, 50, 100, 150]  # keV
    plt.figure(figsize=(10, 6))
    for E in energies:
        d, c, rp, drp = ion_implantation_profile(E, dose, 'N', 'Si')
        plt.plot(d, c, linewidth=2, label=f'{E} keV (Rp={rp:.1f} nm)')
    
    plt.xlabel('Depth [nm]', fontsize=12)
    plt.ylabel('Concentration [atoms/cm³]', fontsize=12)
    plt.title('Ion Implantation Energy and Concentration Profile', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"=== Ion Implantation Parameters ===")
    print(f"Ion species: N⁺")
    print(f"Substrate: Si")
    print(f"Energy: {energy} keV")
    print(f"Dose: {dose:.0e} ions/cm²")
    print(f"➡ range Rp: {Rp:.2f} nm")
    print(f"➡ Range straggling ΔRp: {delta_Rp:.2f} nm")
    print(f"➡ Peak concentration: {conc.max():.2e} atoms/cm³")
    

### 3.3.2 Plasma Treatment (Plasma Treatment)

Plasma breaks and modifies surface chemical bonds to improve wettability, adhesion, and biocompatibility。

  * **Oxygen plasma** ：Surface hydrophilization, organic matter removal
  * **Argon plasma** ：Surface cleaning, activation
  * **Nitrogen plasma** ：Surface nitriding, hardness improvement

### 3.3.3 Laser Surface Melting (Laser Surface Melting)

High-power laser rapidly heats, melts, and cools the surface to form fine grains or amorphous layers. Hardness and wear resistance are improved。

## 3.4 Coating Technologies

### 3.4.1 Thermal Spray (Thermal Spray)

Thermal Spray is a process that forms a coating layer by impacting molten or semi-molten particles at high velocity onto the substrate。

**Classification of Thermal Spray Methods** ：

  * **Flame spray** ：Particle melting with acetylene/oxygen flame, inexpensive, medium adhesion
  * **Plasma spray** ：High-temperature plasma (10,000°Chigh quality, ceramics possible
  * **High-Velocity Oxy-Fuel spray (HVOF)** ：Supersonic flame (Mach 2-3), high adhesion, high density
  * **Cold spray** ：Supersonic acceleration of particles in solid phase, low oxidation, metals and composites

**Important Parameters** ：

  * **Particle velocity** ：100-1200 m/s (varies by method)
  * **Particle temperature** ：Near melting point-3000°C
  * **Adhesion strength** ：Mechanical interlocking + metallic bonding + diffusion bonding

#### Code Example 3.5: Coating Adhesion Strength Prediction (Mechanical and Thermal Properties)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def predict_coating_adhesion(particle_velocity_ms,
                                  particle_temp_C,
                                  coating_material='WC-Co',
                                  substrate_material='Steel'):
        """
        Coating Adhesion Strength Prediction (Simplified Model)
    
        Parameters:
        -----------
        particle_velocity_ms : float
            Particle velocity [m/s]
        particle_temp_C : float
            Particle temperature [°C]
        coating_material : str
            Coating material
        substrate_material : str
            Substrate material
    
        Returns:
        --------
        adhesion_MPa : float
            Predicted adhesion strength [MPa]
        """
        # Material property database
        material_data = {
            'WC-Co': {'T_melt': 2870, 'rho': 14.5, 'E': 600},
            'Al2O3': {'T_melt': 2072, 'rho': 3.95, 'E': 380},
            'Ni': {'T_melt': 1455, 'rho': 8.9, 'E': 200},
            'Steel': {'T_melt': 1500, 'rho': 7.85, 'E': 210}
        }
    
        coating_props = material_data[coating_material]
        substrate_props = material_data[substrate_material]
    
        # Simplified adhesion strength model (empirical formula)
        # adhesion ∝ v^a * (T/Tm)^b
    
        # Velocity contribution (kinetic energy → plastic deformation)
        v_factor = (particle_velocity_ms / 500)**1.5  # Normalized
    
        # Temperature contribution (diffusion bonding promotion)
        T_ratio = particle_temp_C / coating_props['T_melt']
        T_factor = T_ratio**0.8
    
        # Young's modulus compatibility (large difference is disadvantageous)
        E_ratio = min(coating_props['E'], substrate_props['E']) / \
                  max(coating_props['E'], substrate_props['E'])
        E_factor = E_ratio**0.5
    
        # Base adhesion strength (material-dependent)
        base_adhesion = 30  # MPa
    
        # Total adhesion strength [MPa]
        adhesion_MPa = base_adhesion * v_factor * T_factor * E_factor
    
        return adhesion_MPa
    
    # Parameter scan: Effect of particle velocity
    velocities = np.linspace(100, 1000, 50)  # m/s
    temp_fixed = 2000  # °C
    
    adhesions_wc = []
    adhesions_al2o3 = []
    
    for v in velocities:
        adh_wc = predict_coating_adhesion(v, temp_fixed, 'WC-Co', 'Steel')
        adh_al2o3 = predict_coating_adhesion(v, temp_fixed, 'Al2O3', 'Steel')
        adhesions_wc.append(adh_wc)
        adhesions_al2o3.append(adh_al2o3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, adhesions_wc, linewidth=2,
             color='#f5576c', label='WC-Co coating')
    plt.plot(velocities, adhesions_al2o3, linewidth=2,
             color='#f093fb', label='Al₂O₃ coating')
    plt.xlabel('Particle velocity [m/s]', fontsize=12)
    plt.ylabel('Predicted adhesion strength [MPa]', fontsize=12)
    plt.title('Thermal Spray: Particle Velocity and Coating Adhesion Strength', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Parameter scan: Effect of particle temperature
    temps = np.linspace(1000, 2800, 50)  # °C
    vel_fixed = 600  # m/s
    
    adhesions_temp = []
    for T in temps:
        adh = predict_coating_adhesion(vel_fixed, T, 'WC-Co', 'Steel')
        adhesions_temp.append(adh)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temps, adhesions_temp, linewidth=2, color='#f5576c')
    plt.xlabel('Particle temperature [°C]', fontsize=12)
    plt.ylabel('Predicted adhesion strength [MPa]', fontsize=12)
    plt.title('Thermal Spray: Particle Temperature and Coating Adhesion Strength (WC-Co)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Optimization example
    v_opt = 800  # m/s
    T_opt = 2500  # °C
    adh_opt = predict_coating_adhesion(v_opt, T_opt, 'WC-Co', 'Steel')
    
    print(f"=== Thermal Spray Process Optimization ===")
    print(f"Coating material: WC-Co")
    print(f"Substrate material: Steel")
    print(f"Optimal particle velocity: {v_opt} m/s")
    print(f"Optimal particle temperature: {T_opt} °C")
    print(f"➡ Predicted adhesion strength: {adh_opt:.2f} MPa")
    

### 3.4.2 PVD/CVD Basics

**PVD (Physical Vapor Deposition)** ：Thin film formation by physical evaporation and sputtering (details in Chapter 5)

**CVD (Chemical Vapor Deposition)** ：Thin film formation by chemical reactions (details in Chapter 5)

In the context of surface treatment, these are used for hard coatings such as TiN (titanium nitride), CrN (chromium nitride), and DLC (diamond-like carbon)。

### 3.4.3 Sol-Gel Coating (Sol-Gel Coating)

Sol-gel method is a technique to form oxide thin films by gelation and sintering from liquid phase。

  * **Advantages** ：Low-temperature process, large-area compatible, porous films possible, easy composition control
  * **Applications** ：Anti-reflection films, corrosion-resistant films, catalyst supports, optical films

#### Code Example 3.6: Thermal Spray Particle Temperature and Velocity Modeling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_spray_particle_dynamics(particle_diameter_um,
                                          material='WC-Co',
                                          spray_method='HVOF',
                                          distance_mm=150):
        """
        Thermal Spray Particle Temperature and Velocity Change Model During Flight
    
        Parameters:
        -----------
        particle_diameter_um : float
            Particle diameter [μm]
        material : str
            Particle material
        spray_method : str
            Spray method ('Flame', 'Plasma', 'HVOF')
        distance_mm : float
            Spray distance [mm]
    
        Returns:
        --------
        velocity : array
            Velocity [m/s]
        temperature : array
            Temperature [K]
        distance : array
            Distance [mm]
        """
        # Material properties
        material_props = {
            'WC-Co': {'rho': 14500, 'Cp': 200, 'T_melt': 2870 + 273},
            'Al2O3': {'rho': 3950, 'Cp': 880, 'T_melt': 2072 + 273},
            'Ni': {'rho': 8900, 'Cp': 444, 'T_melt': 1455 + 273}
        }
        props = material_props[material]
    
        # Initial conditions for each spray method
        initial_conditions = {
            'Flame': {'v0': 100, 'T0': 2500 + 273},
            'Plasma': {'v0': 300, 'T0': 10000 + 273},
            'HVOF': {'v0': 800, 'T0': 2800 + 273}
        }
        ic = initial_conditions[spray_method]
    
        # Distance range
        distance = np.linspace(0, distance_mm, 500)
    
        # Simple drag model (velocity decay)
        drag_coeff = 0.44  # Spherical particle
        air_rho = 1.2  # kg/m³
        particle_mass = (4/3) * np.pi * (particle_diameter_um/2 * 1e-6)**3 * props['rho']
        particle_area = np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # Velocity decay constant
        k_v = (0.5 * drag_coeff * air_rho * particle_area) / particle_mass
        velocity = ic['v0'] * np.exp(-k_v * distance * 1e-3)
    
        # Temperature decay (convective cooling)
        h = 100  # Heat transfer coefficient [W/m²K]
        T_air = 300  # Air temperature [K]
        surface_area = 4 * np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # Temperature decay constant
        k_T = (h * surface_area) / (particle_mass * props['Cp'])
        temperature = T_air + (ic['T0'] - T_air) * np.exp(-k_T * distance * 1e-3 / velocity[0])
    
        return velocity, temperature - 273, distance  # Convert temperature to °C
    
    # Example: HVOF spray with WC-Co particles
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Velocity profile
    axes[0].plot(d, v, linewidth=2, color='#f5576c')
    axes[0].set_xlabel('Spray distance [mm]', fontsize=12)
    axes[0].set_ylabel('Particle velocity [m/s]', fontsize=12)
    axes[0].set_title('Thermal Spray Particle Velocity Profile (HVOF, WC-Co, 40μm)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature profile
    axes[1].plot(d, T, linewidth=2, color='#f093fb')
    axes[1].axhline(2870, color='red', linestyle='--', alpha=0.7, label='WC-Comelting point')
    axes[1].set_xlabel('Spray distance [mm]', fontsize=12)
    axes[1].set_ylabel('Particle temperature [°C]', fontsize=12)
    axes[1].set_title('Thermal Spray Particle Temperature Profile', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Particle state upon substrate arrival
    v_impact = v[-1]
    T_impact = T[-1]
    print(f"=== Particle State Upon Substrate Arrival ===")
    print(f"Spray distance: {d[-1]:.1f} mm")
    print(f"Impact velocity: {v_impact:.1f} m/s")
    print(f"Arrival temperature: {T_impact:.1f} °C")
    print(f"Melting state: {'Molten' if T_impact > 2870 else 'Solid phase'}")
    
    # Comparison of multiple particle sizes
    diameters = [20, 40, 60, 80]  # μm
    plt.figure(figsize=(10, 6))
    for dia in diameters:
        v_d, T_d, d_d = thermal_spray_particle_dynamics(dia, 'WC-Co', 'HVOF', 150)
        plt.plot(d_d, v_d, linewidth=2, label=f'{dia} μm')
    
    plt.xlabel('Spray distance [mm]', fontsize=12)
    plt.ylabel('Particle velocity [m/s]', fontsize=12)
    plt.title('Velocity Profile Differences by Particle Size (HVOF, WC-Co)',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.5 Surface Treatment Technology Selection

### 3.5.1 Required Properties and Technology Correspondence

Required Property | Suitable Technology | Characteristics  
---|---|---  
Corrosion resistance | Plating (Ni, Cr)、Anodizing | Chemical barrier layer formation  
Wear resistance | Thermal Spray (WC-Co)、PVD (TiN, CrN) | High hardness layer formation  
Decorative (appearance) | Plating (Au, Ag, Ni-Cr)、Anodizing | Gloss, coloration  
Conductivity | Plating (Cu, Ag, Au) | Low resistance contact  
Biocompatibility | Plasma Treatment、Anodizing (Ti) | Surface hydrophilization, oxide layer  
Thermal insulation | Thermal Spray (ceramics) | Low thermal conductivity  
Surface hardening | Ion Implantation (N⁺), laser treatment | No substrate deformation  
  
### 3.5.2 Technology Selection Flowchart
    
    
    ```mermaid
    flowchart TD
        A[Surface treatment requirements] --> B{Main required properties?}
    
        B -->|Corrosion resistance| C{Film thickness requirements}
        C -->|Thin film1-10μm| D[Anodizing]
        C -->|thick films10-100μm| E[PlatingNi/Cr]
    
        B -->|Wear resistance| F{Operating temperature}
        F -->|Room temperature-300°C| G[PVD/CVDTiN, CrN]
        F -->|300°C or higher| H[Thermal SprayWC-Co]
    
        B -->|Decorative properties| I{Conductivity required?}
        I -->|Required| J[PlatingAu/Ag]
        I -->|Not required| K[Anodizingcoloring]
    
        B -->|Conductivity| L[PlatingCu/Ag/Au]
    
        B -->|Biocompatibility| M[Plasma Treatmentor TiAnodizing]
    
        B -->|Surface hardening| N{Substrate heating OK?}
        N -->|NG| O[Ion Implantation]
        N -->|OK| P[Laser treatmentor Thermal Spray]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style E fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style G fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style H fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style J fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style K fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style L fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style M fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style O fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style P fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
    ```

#### Code Example 3.7: Surface Treatment Process Comprehensive Workflow (Parameter Optimization)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    class SurfaceTreatmentOptimizer:
        """
        Surface treatment process parameter optimization class
        """
        def __init__(self, treatment_type='electroplating'):
            self.treatment_type = treatment_type
    
        def objective_function(self, params, targets):
            """
            Objective function: Minimize error from target properties
    
            Parameters:
            -----------
            params : array
                Process parameters (varies by method)
            targets : dict
                Target property value
    
            Returns:
            --------
            error : float
                Error (smaller is better)
            """
            if self.treatment_type == 'electroplating':
                # Parameters: [Current density A/dm², Plating time h, efficiency]
                current_density, time_h, efficiency = params
                area_dm2 = 1.0  # Normalized
    
                # PlatingThicknessCalculation
                current_A = current_density * area_dm2
                thickness = calculate_plating_thickness(
                    current_A, time_h * 3600, area_dm2 * 100, 'Cu', efficiency
                )
    
                # ErrorCalculation
                error_thickness = (thickness - targets['thickness'])**2
    
                # Constraint penalty (film quality deteriorates if current density is too high)
                penalty = 0
                if current_density > 5.0:
                    penalty += 100 * (current_density - 5.0)**2
                if current_density < 0.5:
                    penalty += 100 * (0.5 - current_density)**2
    
                return error_thickness + penalty
    
            elif self.treatment_type == 'anodizing':
                # Parameters: [Voltage V, Time min]
                voltage, time_min = params
    
                # film thicknessCalculation
                _, thickness = anodization_thickness(voltage, 'Al', 'H2SO4', time_min)
    
                error_thickness = (thickness - targets['thickness'])**2
    
                # Constraint penalty
                penalty = 0
                if voltage > 100:
                    penalty += 100 * (voltage - 100)**2
    
                return error_thickness + penalty
    
            else:
                return 0
    
        def optimize(self, targets, initial_guess):
            """
            Optimization execution
            """
            result = minimize(
                lambda p: self.objective_function(p, targets),
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )
    
            return result
    
    # Example 1: Plating process optimization
    print("=== Electroplating Process Optimization ===")
    optimizer_plating = SurfaceTreatmentOptimizer('electroplating')
    
    targets_plating = {
        'thickness': 20.0  # target20μm
    }
    
    initial_guess_plating = [2.0, 1.0, 0.95]  # [Current density, time, efficiency]
    
    result_plating = optimizer_plating.optimize(targets_plating, initial_guess_plating)
    
    print(f"targetPlatingThickness: {targets_plating['thickness']} μm")
    print(f"Optimal parameters:")
    print(f"  Current density: {result_plating.x[0]:.2f} A/dm²")
    print(f"  Plating time: {result_plating.x[1]:.2f} hour")
    print(f"  Current efficiency: {result_plating.x[2]:.3f}")
    
    # Achieved film thickness
    achieved_thickness = calculate_plating_thickness(
        result_plating.x[0], result_plating.x[1] * 3600, 100, 'Cu', result_plating.x[2]
    )
    print(f"➡ Achieved thickness: {achieved_thickness:.2f} μm")
    print(f"  Error: {abs(achieved_thickness - targets_plating['thickness']):.2f} μm")
    
    # Example 2: Anodizing process optimization
    print("\n=== Anodizing Process Optimization ===")
    optimizer_anodizing = SurfaceTreatmentOptimizer('anodizing')
    
    targets_anodizing = {
        'thickness': 15.0  # target15μm
    }
    
    initial_guess_anodizing = [50.0, 30.0]  # [Voltage V, hour min]
    
    result_anodizing = optimizer_anodizing.optimize(targets_anodizing, initial_guess_anodizing)
    
    print(f"targetfilm thickness: {targets_anodizing['thickness']} μm")
    print(f"Optimal parameters:")
    print(f"  Voltage: {result_anodizing.x[0]:.1f} V")
    print(f"  Treatment Time: {result_anodizing.x[1]:.1f} min")
    
    # Achieved film thickness
    _, achieved_thickness_anodizing = anodization_thickness(
        result_anodizing.x[0], 'Al', 'H2SO4', result_anodizing.x[1]
    )
    print(f"➡ Achieved thickness: {achieved_thickness_anodizing:.2f} μm")
    print(f"  Error: {abs(achieved_thickness_anodizing - targets_anodizing['thickness']):.2f} μm")
    
    # Parameter sensitivity analysis (plating)
    current_densities_scan = np.linspace(0.5, 5.0, 30)
    times_scan = np.linspace(0.5, 2.5, 30)
    
    CD, T = np.meshgrid(current_densities_scan, times_scan)
    Thickness = np.zeros_like(CD)
    
    for i in range(len(times_scan)):
        for j in range(len(current_densities_scan)):
            cd = CD[i, j]
            t = T[i, j]
            thick = calculate_plating_thickness(cd, t * 3600, 100, 'Cu', 0.95)
            Thickness[i, j] = thick
    
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(CD, T, Thickness, levels=20, cmap='viridis')
    plt.colorbar(contour, label='PlatingThickness [μm]')
    plt.contour(CD, T, Thickness, levels=[20], colors='red', linewidths=2)
    plt.scatter([result_plating.x[0]], [result_plating.x[1]],
                color='red', s=200, marker='*', edgecolors='white', linewidths=2,
                label='Optimal point')
    plt.xlabel('Current density [A/dm²]', fontsize=12)
    plt.ylabel('Plating time [hour]', fontsize=12)
    plt.title('Plating Process Parameter Map (target 20μm)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.6 Practice Exercises

#### Exercise3.1 (Easy): PlatingThicknessCalculation

In a copper plating process, calculate the plating thickness under the conditions of 2A current, 1 hour plating time, 100cm² plating area, and 95% current efficiency.

Show Solution

**Calculation Procedure** :

  1. Faraday's law: $m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta$
  2. Copper parameters: M = 63.55 g/mol, n = 2, ρ = 8.96 g/cm³
  3. $m = \frac{63.55 \times 2.0 \times 3600}{2 \times 96485} \times 0.95 = 2.25$ g
  4. $d = \frac{2.25}{8.96 \times 100} \times 10^4 = 25.1$ μm

**Answer** : PlatingThickness = 25.1 μm
    
    
    thickness = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"PlatingThickness: {thickness:.2f} μm")  # 25.11 μm
    

#### Exercise 3.2 (Easy): Anodizing Voltage Determination

In aluminum anodizing, we want to form a 50nm barrier layer. When using a sulfuric acid bath, find the required applied voltage (empirical rule: 1.4 nm/V).

Show Solution

**Calculation** :

$V = \frac{d_{\text{barrier}}}{k} = \frac{50}{1.4} = 35.7$ V

**Answer** : Required voltage = 35.7 V (in practice36-40V)

#### Exercise3.3 (Easy): Surface Treatment Technology Selection

We want to provide corrosion resistance and wear resistance to aircraft engine parts (titanium alloy). The temperature reaches 300-600°C. Select an appropriate surface treatment technology and explain the reason.

Show Solution

**Recommended technology** : Thermal spray (plasma spray or HVOF) with ceramic coating (Al₂O₃ or YSZ)

**Reason** :

  * Plating and anodizing are unsuitable in high-temperature environments (300-600°C)
  * ceramic coatingare resistant to high-temperature oxidation
  * Thermal Spray thick films (100-500μm)can formWear resistanceare excellent for
  * HVOFmethod has high adhesion and is suitable for high-speed rotating parts

#### Exercise3.4 (Medium): Improving Throwing Power

In plating of complex-shaped parts, the plating thickness is non-uniform with 25μm on convex areas and 15μm in concave areas. Propose three methods to improve uniformity and explain the effects of each.

Show Solution

**Improvement Methods** :

  1. **Current density reduction**
     * Effect: Homogenization of potential distribution, transition to diffusion-controlled regime
     * Implementation: 2 A/dm² → 0.8 A/dm²reduction, compensate with extended plating time
  2. **Add leveling agent**
     * Effect: Selectively suppress deposition rate on convex areas, preferential deposition in concave areas
     * Implementation: Add thiourea etc. at several ppm
  3. **Enhanced bath agitation**
     * Effect: Homogenization of metal ion diffusion layer thickness
     * Implementation: Aeration, sample rotation, pump circulation

**Expected effect** : Thickness ratio 25:15 → 22:18 range (uniformity 60% → 82%)

#### Exercise 3.5 (Medium): Ion Implantation Dose Calculation

Implant nitrogen ions into a silicon substrate to achieve a peak concentration of 5×10²⁰ atoms/cm³ at a depth of 50nm from the silicon substrate surface. For an energy of 50 keV (Rp = 80 nm, ΔRp = 24 nm), calculate the required dose.

Show Solution

**Calculation Procedure** :

Gaussian distribution at peak concentration (x = Rp):

$$C_{\text{peak}} = \frac{\Phi}{\sqrt{2\pi} \Delta R_p}$$

In the problem, x = 50 nm ≠ Rp = 80 nm, so:

$$C(50) = \frac{\Phi}{\sqrt{2\pi} \cdot 24} \exp\left(-\frac{(50 - 80)^2}{2 \times 24^2}\right)$$

$$5 \times 10^{20} = \frac{\Phi}{\sqrt{2\pi} \cdot 24 \times 10^{-7}} \times 0.557$$

$$\Phi = \frac{5 \times 10^{20} \times \sqrt{2\pi} \times 24 \times 10^{-7}}{0.557} = 1.7 \times 10^{16} \text{ ions/cm}^2$$

**Answer** : Dose = 1.7×10¹⁶ ions/cm²

#### Exercise 3.6 (Medium): Thermal Spray Process Parameter Selection

Apply WC-Co coating by HVOF spray. Under the conditions of particle size 40μm and spray distance 150mm, verify that the particle velocity and temperature upon substrate arrival are 600 m/s or higher and 2500°C or higher. Using code example 3.6, verify if these conditions are met, and if not, propose improvement measures.

Show Solution

**Verification** :
    
    
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    print(f"Impact velocity: {v[-1]:.1f} m/s")  # approximately 650 m/s ✓
    print(f"Arrival temperature: {T[-1]:.1f} °C")   # approximately 2400 °C ✗
    

**Judgment** : Velocity is satisfied, but temperature is insufficient (2400°C < 2500°C)

**Improvement Measures** :

  1. **Spray distance reduction** : 150 mm → 120 mm to reduce temperature loss
  2. **Reduce particle size** : 40 μm → 30 μmreduces cooling rate (heat capacity/surface area ratio↑)
  3. **Initial temperature increase** : Adjust fuel/oxygen ratio, enhance preheating

**Final recommendation** : Spray distance 120 mm + particle size 35 μm → arrival temperature approximately 2550°C (target achieved)

#### Exercise3.7 (Hard): Multi-layer Coating Design

We want to provide both wear resistance and corrosion resistance to automotive engine parts (steel). Design a multi-layer coating under the following conditions:

  * Innermost layer: Adhesion layer (thin film)
  * Middle layer: Wear-resistant layer (thick film)
  * Outermost layer: Corrosion-resistant layer (medium film)

Select the material, thickness, and method for each layer, and explain the design rationale.

Show Solution

**Multi-layer Coating Design** :

layer | Material | Thickness | Method | Reason  
---|---|---|---|---  
Adhesion layer | Ni | 5μm | Electroplating | Good adhesion to steel, stress relaxation  
Wear-resistant layer | WC-Co | 150μm | HVOF spray | High hardness (HV1200)、Wear resistance  
Corrosion-resistant layer | Cr₃C₂-NiCr | 50μm | HVOF spray | Oxidation resistance, high-temperature corrosion resistance  
  
**Process Sequence** :

  1. Steel substrate pretreatment (degreasing, sandblasting, Ra = 3-5μm)
  2. Electroplating with NiAdhesion layer (Current density2 A/dm², 1hour)
  3. HVOF spray WC-Co layer (particle size 30μm, spray distance 120mm, velocity 800m/s)
  4. HVOF spray Cr₃C₂-NiCr layer (particle size 40μm, spray distance 150mm)
  5. Post-treatment (polishing, sealing as needed)

**Expected Performance** :

  * Wear resistance: Coefficient of friction0.3、wear rate < 10⁻⁶ mm³/Nm
  * Corrosion resistance: Salt spray test1000hour
  * Adhesion strength: > 50 MPa

#### Exercise3.8 (Hard): Process Troubleshooting

The following defects occurred in the copper plating process. Propose the causes and countermeasures for each defect:

  * **Defect A** : Many small protrusions (nodules) appeared on the plating surface
  * **DefectB** : PlatingThickness target20μmagainst target12μmonly achieves
  * **Defect C** : Peeling occurred in the adhesion test (tape test) after plating

Show Solution

**Defect A: Nodules (Surface Protrusions)**

**Candidate Causes** :

  * Impurities and particles in the plating bath (dust, other metal ions)
  * Insufficient bath filtration
  * Dendritic growth due to excessive current density

**Countermeasures** :

  1. Plating bath filtration (5μm cartridge filter, 24-hour circulation)
  2. Activated carbon treatment of anode (impurity removal)
  3. Current density reduction (5 A/dm² → 2 A/dm²)
  4. Strengthen sample pretreatment (degreasing → pickling → pure water rinse)

**DefectB: Insufficient Film Thickness**

**Candidate Causes** :

  * Decrease in current efficiency (due to side reactions)
  * Insufficient metal ion concentration
  * Actual current value lower than set value

**Verification** :
    
    
    # Theoretical thickness (efficiency 95%)
    d_theoretical = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"theoryfilm thickness: {d_theoretical:.1f} μm")  # 25.1 μm
    
    # Current efficiency back-calculated from actual 12μm
    actual_efficiency = 12 / d_theoretical * 0.95
    print(f"Actual current efficiency: {actual_efficiency:.1%}")  # approximately 45% (significant decrease)
    

**Countermeasures** :

  1. Bath composition analysis (CuSO₄Concentration、H₂SO₄Concentration)→ replenish if insufficient
  2. Check ammeter calibration
  3. Check bath temperature (current efficiency decreases at low temperature) → maintain at 25±2°C
  4. Check balance of anode and cathode area (1:1-2:1is ideal)

**Defect C: Adhesion defect**

**Candidate Causes** :

  * Contamination on substrate surface (oils, oxide film)
  * Insufficient pretreatment
  * Stress due to thermal expansion coefficient mismatch with substrate

**Countermeasures** :

  1. Review pretreatment process 
     * Degreasing: Alkaline degreasing (60°C, 10min) + ultrasonic cleaning
     * Pickling: 10% H₂SO₄ (room temperature, 1min) for oxide film removal
     * Activation: 5% HCl (room temperature, 30 seconds) immediate pretreatment
  2. Strike plating (thin Ni or Cu layer) to improve adhesion
  3. Post-plating baking (150°C, 1 hour) to remove hydrogen embrittlement and improve adhesion

**Verification method** :

  * Adhesion test: JIS H8504 (cross-cut → tape test)
  * Tensile test: ASTM B571 (tensile adhesion strength > 20 MPa target)

## 3.7 Learning Confirmation Checklist

### Basic Understanding (5 items)

  * □ Can calculate plating thickness using Faraday's law
  * □ Can explain the difference between barrier layer and porous layer in anodizing
  * □ Ion ImplantationUnderstand the relationship between range and dose
  * □ Understand the classification of coating technologies (plating, thermal spray, PVD/CVD)
  * □ Can explain the effects of thermal spray particle velocity and temperature on adhesion

### Practical Skills (5 items)

  * □ Can design plating conditions considering current density and current efficiency
  * □ Can calculate the relationship between anodizing voltage and film thickness
  * □ Ion ImplantationCan simulate profiles in Python
  * □ Can utilize surface treatment technology selection flowchart
  * □ PlatingDefect (nodules、Insufficient Film Thickness、Adhesion defect Can estimate causes of plating defects

### Applied Skills (5 items)

  * □ Can propose methods to improve uniformity for complex-shaped parts
  * □ Can design multi-layer coating and select material, thickness, and method for each layer
  * □ Can optimize thermal spray process parameters (particle size, spray distance)
  * □ Can select surface treatment technology according to required properties (corrosion resistance, wear resistance, conductivity, etc.)
  * □ Can troubleshoot process anomalies

## 3.8 References

  1. Kanani, N. (2004). _Electroplating: Basic Principles, Processes and Practice_. Elsevier, **pp. 56-89** (Faraday's law and electrochemical fundamentals).
  2. Wernick, S., Pinner, R., Sheasby, P.G. (1987). _The Surface Treatment and Finishing of Aluminum and Its Alloys_ (5th ed.). ASM International, **pp. 234-267** (Anodizingprocess and film structure).
  3. Davis, J.R. (Ed.) (2004). _Handbook of Thermal Spray Technology_. ASM International, **pp. 123-156** (Thermal Sprayprocesses and coating properties).
  4. Pawlowski, L. (2008). _The Science and Engineering of Thermal Spray Coatings_ (2nd ed.). Wiley, **pp. 189-223** (HVOFspray and particle dynamics).
  5. Townsend, P.D., Chandler, P.J., Zhang, L. (1994). _Optical Effects of Ion Implantation_. Cambridge University Press, **pp. 45-78** (Ion Implantationtheory and LSS model).
  6. Inagaki, M., Toyoda, M., Soneda, Y., Morishita, T. (2014). "Nitrogen-doped carbon materials." _Carbon_ , 132, 104-140, **pp. 115-128** , DOI: 10.1016/j.carbon.2014.01.027 (Plasma nitriding process).
  7. Fauchais, P.L., Heberlein, J.V.R., Boulos, M.I. (2014). _Thermal Spray Fundamentals: From Powder to Part_. Springer, **pp. 567-612** (Thermal Sprayfundamentals and applications).
  8. Schlesinger, M., Paunovic, M. (Eds.) (2010). _Modern Electroplating_ (5th ed.). Wiley, **pp. 209-248** (Latest plating technology and troubleshooting).

## Summary

In this chapter, we learned from the fundamentals to practice of material surface treatment technologies. In electroplating, we studied film thickness calculation using Faraday's law and current density optimization; in anodizing, the formation mechanisms of barrier layer and porous layer; in ion implantation, concentration profile modeling; and in thermal spray, the relationship between particle dynamics and adhesion strength.

Surface treatment technology is an important process technology that provides surface functions (corrosion resistance, wear resistance, conductivity, decorativeness, etc.) without changing the internal properties of materials. Appropriate technology selection and parameter optimization can significantly improve product performance and lifespan.

In the next chapter, we will learn about powder sintering processes. We will acquire the fundamentals and practice of powder metallurgy, including sintering mechanisms, densification models, hot pressing, and SPS.
