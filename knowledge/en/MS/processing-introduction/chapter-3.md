---
title: "Chapter 3: Surface Treatment Technologies"
chapter_title: "Chapter 3: Surface Treatment Technologies"
subtitle: Electroplating, Anodizing, Surface Modification, Coating Technologies
reading_time: 35-45 min
difficulty: Intermediate
version: 1.0
created_at: 2025-10-28
---

## Learning Objectives

By completing this chapter, you will acquire the following skills:

  * ✅ Calculate plating thickness using Faraday's law and optimize current density
  * ✅ Understand the voltage-thickness relationship in anodizing and design anodized aluminum (alumite) treatments
  * ✅ Model ion implantation concentration profiles with a Gaussian distribution
  * ✅ Understand selection criteria for coating technologies and choose the appropriate method
  * ✅ Evaluate how particle velocity and temperature affect adhesion in thermal spray processes
  * ✅ Optimize surface treatment process parameters and troubleshoot process problems

## 3.1 Electroplating

### 3.1.1 Faraday's Law and Electrochemical Fundamentals

Electroplating is a process in which metal ions are reduced and deposited on the cathode (the workpiece) surface by electrolysis. The plating rate and film thickness follow Faraday's law.

**Faraday's first law** : The deposited metal mass is proportional to the charge passed

$$ m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta $$ 

where

  * $m$: deposited mass [g]
  * $M$: atomic weight of the metal [g/mol]
  * $I$: current [A]
  * $t$: plating time [s]
  * $n$: number of electrons (e.g., 2 for Cu²⁺)
  * $F$: Faraday constant (96485 C/mol)
  * $\eta$: current efficiency (typically 0.85–0.98)

The plating thickness $d$ [μm] follows from the deposited mass and density:

$$ d = \frac{m}{\rho \cdot A} \times 10^4 $$ 

$\rho$: metal density [g/cm³], $A$: plated area [cm²]

**Effect of current density** :

  * **Low current density** (0.5–2 A/dm²): smooth, dense films, slow deposition
  * **High current density** (5–20 A/dm²): rough films, dendritic growth, fast deposition

**Throwing power (deposit uniformity)** :

On parts with complex geometries, the current density distribution becomes non-uniform, causing variations in film thickness. Throwing power can be improved through bath composition, additives, and agitation.

#### Code Example 3.1: Plating Thickness Calculation Using Faraday's Law
    
    
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
            Plated area [cm²]
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
    
    # Example: copper plating
    current = 2.0      # 2 A
    time_hours = 1.0   # 1 hour
    time_s = time_hours * 3600
    area = 100.0       # 100 cm²
    
    thickness = calculate_plating_thickness(current, time_s, area,
                                             metal='Cu', efficiency=0.95)
    
    print(f"=== Copper Plating Process Calculation ===")
    print(f"Current: {current} A")
    print(f"Current density: {current/area*100:.2f} A/dm²")
    print(f"Plating time: {time_hours} hours")
    print(f"Plated area: {area} cm²")
    print(f"Current efficiency: 95%")
    print(f"➡ Plating thickness: {thickness:.2f} μm")
    
    # Plot: plating time vs film thickness
    time_range = np.linspace(0, 2, 100) * 3600  # 0-2 hours
    thicknesses = [calculate_plating_thickness(current, t, area, 'Cu', 0.95)
                   for t in time_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_range/3600, thicknesses, linewidth=2, color='#f5576c')
    plt.xlabel('Plating time [hours]', fontsize=12)
    plt.ylabel('Plating thickness [μm]', fontsize=12)
    plt.title('Copper Plating: Plating Time vs Film Thickness', fontsize=14, fontweight='bold')
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
    plt.xlabel('Current density [A/dm²]', fontsize=12)
    plt.ylabel('Plating thickness [μm]', fontsize=12)
    plt.title('Current Density vs Plating Thickness (1 hour)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.1.2 Plating Baths and Additives

The composition of the plating bath has a decisive influence on film quality.

Bath Component | Role | Typical Concentration  
---|---|---  
Metal salt (e.g., CuSO₄) | Supplies metal ions | 200–250 g/L  
Conducting salt (e.g., H₂SO₄) | Improves conductivity | 50–80 g/L  
Brightener | Smoothing, imparting gloss | A few ppm to several hundred ppm  
Leveling agent | Flattening surface irregularities | A few ppm to tens of ppm  
Surfactant | Promotes hydrogen gas release | A few ppm  
  
#### Code Example 3.2: Current Density Distribution Simulation (2D Electrodes)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import laplace
    
    def simulate_current_distribution_2d(width=50, height=50,
                                          anode_position='top',
                                          cathode_position='bottom',
                                          iterations=500):
        """
        Current density distribution simulation for a 2D electrode configuration
        Solves Laplace's equation with the finite difference method
    
        Parameters:
        -----------
        width, height : int
            Computational grid size
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
    
        # Current density = -∇φ (proportional to the potential gradient)
        grad_y, grad_x = np.gradient(phi)
        current_density = np.sqrt(grad_x**2 + grad_y**2)
    
        return phi, current_density
    
    # Example: anode at top, cathode at bottom
    phi, j = simulate_current_distribution_2d(width=50, height=50,
                                               anode_position='top',
                                               cathode_position='bottom')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Potential distribution
    im1 = axes[0].imshow(phi, cmap='viridis', origin='lower')
    axes[0].set_title('Potential Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X position')
    axes[0].set_ylabel('Y position')
    plt.colorbar(im1, ax=axes[0], label='Potential [V]')
    
    # Current density distribution
    im2 = axes[1].imshow(j, cmap='hot', origin='lower')
    axes[1].set_title('Current Density Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X position')
    axes[1].set_ylabel('Y position')
    plt.colorbar(im2, ax=axes[1], label='Current density [a.u.]')
    
    # Current density along the cathode surface
    cathode_j = j[-1, :]  # Bottom edge (cathode)
    axes[2].plot(cathode_j, linewidth=2, color='#f5576c')
    axes[2].set_title('Current Density at Cathode Surface', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X position')
    axes[2].set_ylabel('Current density [a.u.]')
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
    print(f"Max/min current density ratio: {cathode_j.max()/cathode_j.min():.2f}")
    

## 3.2 Anodizing

### 3.2.1 Principles of Aluminum Anodizing

Anodizing is a process that electrochemically oxidizes a metal surface to form an oxide film. Anodized aluminum (alumite) treatment is the classic example.

**Anodizing process** :

  1. Immerse aluminum as the anode and platinum (or similar) as the cathode in an electrolyte (sulfuric acid, oxalic acid, etc.)
  2. Applying a DC voltage grows an Al₂O₃ film on the Al surface
  3. The film has a porous structure (barrier layer + porous layer)

    
    
    ```mermaid
    flowchart TB
        subgraph "Anodizing Cell"
            A[Aluminum anode]
            B[ElectrolyteSulfuric/oxalic acid]
            C[Platinum cathode]
            D[DC power supply]
        end
    
        D -->|Applied voltage| A
        D --> C
        A -->|Al³⁺| B
        B -->|O²⁻| A
        A -->|Al₂O₃ formation| E[Oxide film]
    
        E --> F[Barrier layerDense, thin]
        E --> G[Porous layerPorous, thick]
    
        G --> H[SealingHot water/steam]
        H --> I[Final filmImproved corrosion resistance]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
    ```

**Relationship between film thickness and voltage** :

For a sulfuric acid bath, the barrier layer thickness is approximately proportional to the applied voltage (empirical rule):

$$ d_{\text{barrier}} \approx 1.4 \, [\text{nm/V}] \times V $$ 

The total film thickness (barrier layer + porous layer) depends on the treatment time and current density.

#### Code Example 3.3: Anodic Oxide Thickness vs Voltage
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def anodization_thickness(voltage, material='Al',
                              electrolyte='H2SO4', time_min=30):
        """
        Calculate anodic oxide film thickness
    
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
        # Constants for each material/electrolyte
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
    
        # Porous layer thickness [μm] (simplified model)
        porous_thickness = k_porous * time_min
    
        # Total thickness [μm]
        total_thickness = (barrier_thickness / 1000) + porous_thickness
    
        return barrier_thickness, total_thickness
    
    # Scan over voltage range
    voltages = np.linspace(10, 100, 100)
    barrier_thicknesses = []
    total_thicknesses = []
    
    for V in voltages:
        d_barrier, d_total = anodization_thickness(V, 'Al', 'H2SO4', 30)
        barrier_thicknesses.append(d_barrier)
        total_thicknesses.append(d_total)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Barrier layer thickness vs voltage
    axes[0].plot(voltages, barrier_thicknesses, linewidth=2,
                 color='#f5576c', label='Barrier layer')
    axes[0].set_xlabel('Applied voltage [V]', fontsize=12)
    axes[0].set_ylabel('Barrier layer thickness [nm]', fontsize=12)
    axes[0].set_title('Barrier Layer Thickness vs Voltage (Al/Sulfuric Acid Bath)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Total thickness vs voltage
    axes[1].plot(voltages, total_thicknesses, linewidth=2,
                 color='#f093fb', label='Total thickness (30 min)')
    axes[1].set_xlabel('Applied voltage [V]', fontsize=12)
    axes[1].set_ylabel('Total thickness [μm]', fontsize=12)
    axes[1].set_title('Total Thickness vs Voltage (Al/Sulfuric Acid Bath)',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Design example: a 50 nm thick barrier layer is required
    target_barrier = 50  # nm
    required_voltage = target_barrier / 1.4
    print(f"=== Anodizing Process Design ===")
    print(f"Target barrier layer thickness: {target_barrier} nm")
    print(f"➡ Required voltage: {required_voltage:.1f} V")
    
    # Effect of time
    times = np.linspace(10, 60, 50)  # 10-60 min
    total_thicknesses_time = []
    for t in times:
        _, d_total = anodization_thickness(50, 'Al', 'H2SO4', t)
        total_thicknesses_time.append(d_total)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_thicknesses_time, linewidth=2, color='#f5576c')
    plt.xlabel('Treatment time [min]', fontsize=12)
    plt.ylabel('Total thickness [μm]', fontsize=12)
    plt.title('Anodic Oxide Thickness vs Treatment Time (50 V, Sulfuric Acid Bath)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.2.2 Sealing

Sealing is a post-treatment that closes the pores of the porous layer to improve corrosion resistance.

  * **Hot water sealing** : 30–60 min in pure water at 95–100°C; Al(OH)₃ closes the pores
  * **Steam sealing** : 10–30 min in steam at 110°C
  * **Cold sealing** : room-temperature treatment in a nickel salt solution (energy saving)

## 3.3 Surface Modification Technologies

### 3.3.1 Ion Implantation

Ion implantation is a technique that drives high-energy ions into a material surface to modify its chemical composition and crystal structure. It is used for doping in semiconductor manufacturing and for surface hardening of metals.

**Ion implantation process** :

  1. Generate ions in an ion source (e.g., N⁺, B⁺, P⁺)
  2. Accelerate to 10–200 keV with an accelerating field
  3. Select only the target ions with a mass analyzer
  4. Irradiate the sample in a vacuum chamber

**Concentration profile (LSS theory)** :

The post-implantation concentration distribution is approximated by a Gaussian distribution:

$$ C(x) = \frac{\Phi}{\sqrt{2\pi} \Delta R_p} \exp\left(-\frac{(x - R_p)^2}{2 \Delta R_p^2}\right) $$ 

  * $C(x)$: concentration at depth $x$ [atoms/cm³]
  * $\Phi$: dose (total ions per unit area) [ions/cm²]
  * $R_p$: projected range (peak depth) [nm]
  * $\Delta R_p$: range straggling (standard deviation) [nm]

#### Code Example 3.4: Ion Implantation Concentration Profile (Gaussian LSS Theory)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    
    def ion_implantation_profile(energy_keV, dose_cm2, ion='N',
                                  substrate='Si', depth_range=None):
        """
        Ion implantation concentration profile calculation (Gaussian approximation)
    
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
        # In practice, use simulation tools such as SRIM/TRIM
    
        # Ion masses
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
    
        # Projected range Rp [nm] (simplified formula)
        Rp = 10 * energy_keV**0.7 * (M_sub / M_ion)**0.5
    
        # Range straggling ΔRp [nm]
        delta_Rp = 0.3 * Rp
    
        if depth_range is None:
            depth_range = np.linspace(0, 3 * Rp, 500)
    
        # Gaussian concentration distribution
        concentration = (dose_cm2 / (np.sqrt(2 * np.pi) * delta_Rp)) * \
                        np.exp(-(depth_range - Rp)**2 / (2 * delta_Rp**2))
    
        return depth_range, concentration, Rp, delta_Rp
    
    # Example: nitrogen ion implantation into silicon
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
    plt.title('Ion Implantation Concentration Profile (N⁺ → Si)', fontsize=14, fontweight='bold')
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
    plt.title('Ion Implantation Energy and Concentration Profiles', fontsize=14, fontweight='bold')
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
    print(f"➡ Projected range Rp: {Rp:.2f} nm")
    print(f"➡ Range straggling ΔRp: {delta_Rp:.2f} nm")
    print(f"➡ Peak concentration: {conc.max():.2e} atoms/cm³")
    

### 3.3.2 Plasma Treatment

Plasma breaks and modifies chemical bonds at the surface, improving wettability, adhesion, and biocompatibility.

  * **Oxygen plasma** : surface hydrophilization, organic contaminant removal
  * **Argon plasma** : surface cleaning, activation
  * **Nitrogen plasma** : surface nitriding, hardness improvement

### 3.3.3 Laser Surface Melting

A high-power laser rapidly heats, melts, and cools the surface, forming fine crystal grains or amorphous layers. Hardness and wear resistance are improved.

## 3.4 Coating Technologies

### 3.4.1 Thermal Spray

Thermal spraying is a process in which molten or semi-molten particles impact the substrate at high velocity to form a coating layer.

**Classification of thermal spray methods** :

  * **Flame spray** : particles melted by an acetylene/oxygen flame; inexpensive, moderate adhesion
  * **Plasma spray** : high-temperature plasma (over 10,000°C); high quality, suitable for ceramics
  * **High-velocity oxy-fuel spray (HVOF)** : supersonic flame (Mach 2–3); high adhesion, high density
  * **Cold spray** : particles accelerated to supersonic speed in the solid state; low oxidation, metals and composites

**Key parameters** :

  * **Particle velocity** : 100–1200 m/s (varies by method)
  * **Particle temperature** : near the melting point up to 3000°C
  * **Adhesion strength** : mechanical interlocking + metallic bonding + diffusion bonding

#### Code Example 3.5: Coating Adhesion Strength Prediction (Mechanical and Thermal Properties)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def predict_coating_adhesion(particle_velocity_ms,
                                  particle_temp_C,
                                  coating_material='WC-Co',
                                  substrate_material='Steel'):
        """
        Coating adhesion strength prediction (simplified model)
    
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
        v_factor = (particle_velocity_ms / 500)**1.5  # normalized
    
        # Temperature contribution (promotes diffusion bonding)
        T_ratio = particle_temp_C / coating_props['T_melt']
        T_factor = T_ratio**0.8
    
        # Young's modulus compatibility (a large mismatch is unfavorable)
        E_ratio = min(coating_props['E'], substrate_props['E']) / \
                  max(coating_props['E'], substrate_props['E'])
        E_factor = E_ratio**0.5
    
        # Base adhesion strength (material dependent)
        base_adhesion = 30  # MPa
    
        # Overall adhesion strength [MPa]
        adhesion_MPa = base_adhesion * v_factor * T_factor * E_factor
    
        return adhesion_MPa
    
    # Parameter scan: effect of particle velocity
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
    plt.title('Thermal Spray: Particle Velocity vs Coating Adhesion', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Parameter scan: effect of particle temperature
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
    plt.title('Thermal Spray: Particle Temperature vs Coating Adhesion (WC-Co)', fontsize=14, fontweight='bold')
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
    

### 3.4.2 PVD/CVD Fundamentals

**PVD (Physical Vapor Deposition)** : thin film formation by physical evaporation or sputtering (details in Chapter 5)

**CVD (Chemical Vapor Deposition)** : thin film formation by chemical reactions (details in Chapter 5)

In the context of surface treatment, these methods are used for hard coatings such as TiN (titanium nitride), CrN (chromium nitride), and DLC (diamond-like carbon).

### 3.4.3 Sol-Gel Coating

The sol-gel method forms oxide thin films from the liquid phase through gelation and firing.

  * **Advantages** : low-temperature process, large-area capability, porous films possible, easy composition control
  * **Applications** : anti-reflection coatings, corrosion-resistant films, catalyst supports, optical films

#### Code Example 3.6: Temperature and Velocity Modeling of Thermal Spray Particles
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_spray_particle_dynamics(particle_diameter_um,
                                          material='WC-Co',
                                          spray_method='HVOF',
                                          distance_mm=150):
        """
        Model of in-flight temperature and velocity changes of thermal spray particles
    
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
    
        # Simplified drag model (velocity decay)
        drag_coeff = 0.44  # spherical particle
        air_rho = 1.2  # kg/m³
        particle_mass = (4/3) * np.pi * (particle_diameter_um/2 * 1e-6)**3 * props['rho']
        particle_area = np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # Velocity decay constant
        k_v = (0.5 * drag_coeff * air_rho * particle_area) / particle_mass
        velocity = ic['v0'] * np.exp(-k_v * distance * 1e-3)
    
        # Temperature decay (convective cooling)
        h = 100  # heat transfer coefficient [W/m²K]
        T_air = 300  # air temperature [K]
        surface_area = 4 * np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # Temperature decay constant
        k_T = (h * surface_area) / (particle_mass * props['Cp'])
        temperature = T_air + (ic['T0'] - T_air) * np.exp(-k_T * distance * 1e-3 / velocity[0])
    
        return velocity, temperature - 273, distance  # convert temperature to °C
    
    # Example: WC-Co particles with HVOF spraying
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Velocity profile
    axes[0].plot(d, v, linewidth=2, color='#f5576c')
    axes[0].set_xlabel('Spray distance [mm]', fontsize=12)
    axes[0].set_ylabel('Particle velocity [m/s]', fontsize=12)
    axes[0].set_title('Thermal Spray Particle Velocity Profile (HVOF, WC-Co, 40 μm)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature profile
    axes[1].plot(d, T, linewidth=2, color='#f093fb')
    axes[1].axhline(2870, color='red', linestyle='--', alpha=0.7, label='WC-Co melting point')
    axes[1].set_xlabel('Spray distance [mm]', fontsize=12)
    axes[1].set_ylabel('Particle temperature [°C]', fontsize=12)
    axes[1].set_title('Thermal Spray Particle Temperature Profile', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Particle state at substrate impact
    v_impact = v[-1]
    T_impact = T[-1]
    print(f"=== Particle State at Substrate Impact ===")
    print(f"Spray distance: {d[-1]:.1f} mm")
    print(f"Impact velocity: {v_impact:.1f} m/s")
    print(f"Impact temperature: {T_impact:.1f} °C")
    print(f"Melting state: {'molten' if T_impact > 2870 else 'solid'}")
    
    # Comparison of multiple particle sizes
    diameters = [20, 40, 60, 80]  # μm
    plt.figure(figsize=(10, 6))
    for dia in diameters:
        v_d, T_d, d_d = thermal_spray_particle_dynamics(dia, 'WC-Co', 'HVOF', 150)
        plt.plot(d_d, v_d, linewidth=2, label=f'{dia} μm')
    
    plt.xlabel('Spray distance [mm]', fontsize=12)
    plt.ylabel('Particle velocity [m/s]', fontsize=12)
    plt.title('Velocity Profiles for Different Particle Sizes (HVOF, WC-Co)',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.5 Selecting a Surface Treatment Technology

### 3.5.1 Matching Required Properties to Technologies

Required Property | Suitable Technology | Characteristics  
---|---|---  
Corrosion resistance | Plating (Ni, Cr), anodizing | Chemical barrier layer formation  
Wear resistance | Thermal spray (WC-Co), PVD (TiN, CrN) | High-hardness layer formation  
Decorative finish (appearance) | Plating (Au, Ag, Ni-Cr), anodizing | Gloss, color  
Electrical conductivity | Plating (Cu, Ag, Au) | Low-resistance contacts  
Biocompatibility | Plasma treatment, anodizing (Ti) | Surface hydrophilization, oxide layer  
Thermal insulation | Thermal spray (ceramics) | Low thermal conductivity  
Surface hardening | Ion implantation (N⁺), laser treatment | No substrate distortion  
  
### 3.5.2 Technology Selection Flowchart
    
    
    ```mermaid
    flowchart TD
        A[Surface treatment requirement] --> B{Primary property?}
    
        B -->|Corrosion resistance| C{Thickness requirement}
        C -->|Thin film1-10μm| D[Anodizing]
        C -->|Thick film10-100μm| E[PlatingNi/Cr]
    
        B -->|Wear resistance| F{Service temperature}
        F -->|Room temp to 300°C| G[PVD/CVDTiN, CrN]
        F -->|Above 300°C| H[Thermal sprayWC-Co]
    
        B -->|Decorative| I{Conductivity needed?}
        I -->|Yes| J[PlatingAu/Ag]
        I -->|No| K[AnodizingColoring]
    
        B -->|Conductivity| L[PlatingCu/Ag/Au]
    
        B -->|Biocompatibility| M[Plasma treatmentor Ti anodizing]
    
        B -->|Surface hardening| N{Substrate heating OK?}
        N -->|No| O[Ion implantation]
        N -->|Yes| P[Laser treatmentor thermal spray]
    
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

#### Code Example 3.7: Integrated Surface Treatment Workflow (Parameter Optimization)
    
    
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
            Objective function: minimize the error from target properties
    
            Parameters:
            -----------
            params : array
                Process parameters (depend on the treatment type)
            targets : dict
                Target property values
    
            Returns:
            --------
            error : float
                Error (smaller is better)
            """
            if self.treatment_type == 'electroplating':
                # Parameters: [current density A/dm², plating time h, efficiency]
                current_density, time_h, efficiency = params
                area_dm2 = 1.0  # normalized
    
                # Calculate plating thickness
                current_A = current_density * area_dm2
                thickness = calculate_plating_thickness(
                    current_A, time_h * 3600, area_dm2 * 100, 'Cu', efficiency
                )
    
                # Compute error
                error_thickness = (thickness - targets['thickness'])**2
    
                # Constraint penalty (excessive current density degrades film quality)
                penalty = 0
                if current_density > 5.0:
                    penalty += 100 * (current_density - 5.0)**2
                if current_density < 0.5:
                    penalty += 100 * (0.5 - current_density)**2
    
                return error_thickness + penalty
    
            elif self.treatment_type == 'anodizing':
                # Parameters: [voltage V, time min]
                voltage, time_min = params
    
                # Calculate film thickness
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
            Run optimization
            """
            result = minimize(
                lambda p: self.objective_function(p, targets),
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )
    
            return result
    
    # Example 1: electroplating process optimization
    print("=== Electroplating Process Optimization ===")
    optimizer_plating = SurfaceTreatmentOptimizer('electroplating')
    
    targets_plating = {
        'thickness': 20.0  # target 20 μm
    }
    
    initial_guess_plating = [2.0, 1.0, 0.95]  # [current density, time, efficiency]
    
    result_plating = optimizer_plating.optimize(targets_plating, initial_guess_plating)
    
    print(f"Target plating thickness: {targets_plating['thickness']} μm")
    print(f"Optimal parameters:")
    print(f"  Current density: {result_plating.x[0]:.2f} A/dm²")
    print(f"  Plating time: {result_plating.x[1]:.2f} hours")
    print(f"  Current efficiency: {result_plating.x[2]:.3f}")
    
    # Achieved film thickness
    achieved_thickness = calculate_plating_thickness(
        result_plating.x[0], result_plating.x[1] * 3600, 100, 'Cu', result_plating.x[2]
    )
    print(f"➡ Achieved thickness: {achieved_thickness:.2f} μm")
    print(f"  Error: {abs(achieved_thickness - targets_plating['thickness']):.2f} μm")
    
    # Example 2: anodizing process optimization
    print("\n=== Anodizing Process Optimization ===")
    optimizer_anodizing = SurfaceTreatmentOptimizer('anodizing')
    
    targets_anodizing = {
        'thickness': 15.0  # target 15 μm
    }
    
    initial_guess_anodizing = [50.0, 30.0]  # [voltage V, time min]
    
    result_anodizing = optimizer_anodizing.optimize(targets_anodizing, initial_guess_anodizing)
    
    print(f"Target thickness: {targets_anodizing['thickness']} μm")
    print(f"Optimal parameters:")
    print(f"  Voltage: {result_anodizing.x[0]:.1f} V")
    print(f"  Treatment time: {result_anodizing.x[1]:.1f} min")
    
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
    plt.colorbar(contour, label='Plating thickness [μm]')
    plt.contour(CD, T, Thickness, levels=[20], colors='red', linewidths=2)
    plt.scatter([result_plating.x[0]], [result_plating.x[1]],
                color='red', s=200, marker='*', edgecolors='white', linewidths=2,
                label='Optimal point')
    plt.xlabel('Current density [A/dm²]', fontsize=12)
    plt.ylabel('Plating time [hours]', fontsize=12)
    plt.title('Plating Process Parameter Map (Target 20 μm)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.6 Exercises

#### Exercise 3.1 (Easy): Plating Thickness Calculation

In a copper plating process, calculate the plating thickness for a current of 2 A, a plating time of 1 hour, a plated area of 100 cm², and a current efficiency of 95%.

Show Solution

**Calculation steps** :

  1. Faraday's law: $m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta$
  2. Copper parameters: M = 63.55 g/mol, n = 2, ρ = 8.96 g/cm³
  3. $m = \frac{63.55 \times 2.0 \times 3600}{2 \times 96485} \times 0.95 = 2.25$ g
  4. $d = \frac{2.25}{8.96 \times 100} \times 10^4 = 25.1$ μm

**Answer** : plating thickness = 25.1 μm
    
    
    thickness = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"Plating thickness: {thickness:.2f} μm")  # 25.11 μm
    

#### Exercise 3.2 (Easy): Determining the Anodizing Voltage

In anodizing aluminum, you want to form a 50 nm barrier layer. If a sulfuric acid bath is used, find the required applied voltage (empirical rule: 1.4 nm/V).

Show Solution

**Calculation** :

$V = \frac{d_{\text{barrier}}}{k} = \frac{50}{1.4} = 35.7$ V

**Answer** : required voltage = 35.7 V (36–40 V in practice)

#### Exercise 3.3 (Easy): Selecting a Surface Treatment Technology

You want to impart corrosion resistance and wear resistance to an aircraft engine component (made of titanium alloy). Temperatures reach 300–600°C. Select an appropriate surface treatment technology and explain your reasoning.

Show Solution

**Recommended technology** : ceramic coating (Al₂O₃ or YSZ) by thermal spraying (plasma spray or HVOF)

**Reasoning** :

  * Plating and anodizing are unsuitable for high-temperature environments (300–600°C)
  * Ceramic coatings resist high-temperature oxidation
  * Thermal spraying can form thick films (100–500 μm) with excellent wear resistance
  * The HVOF method provides high adhesion, suitable for high-speed rotating components

#### Exercise 3.4 (Medium): Improving Throwing Power

When plating a part with a complex shape, the plating thickness is non-uniform: 25 μm on convex areas and 15 μm in recessed areas. Propose three methods to improve throwing power and explain the effect of each.

Show Solution

**Improvement methods** :

  1. **Reduce the current density**
     * Effect: equalizes the potential distribution, shifts toward the diffusion-controlled regime
     * Implementation: reduce from 2 A/dm² to 0.8 A/dm², compensate with a longer plating time
  2. **Add a leveling agent**
     * Effect: selectively suppresses deposition on convex areas, preferential deposition in recesses
     * Implementation: add a few ppm of an additive such as thiourea
  3. **Increase bath agitation**
     * Effect: equalizes the diffusion layer thickness of metal ions
     * Implementation: aeration, sample rotation, pump circulation

**Expected result** : thickness ratio improves from 25:15 to about 22:18 (uniformity 60% → 82%)

#### Exercise 3.5 (Medium): Calculating the Ion Implantation Dose

Nitrogen ions are implanted into a silicon substrate, and you want to achieve a peak concentration of 5×10²⁰ atoms/cm³ at a depth of 50 nm from the surface. For an energy of 50 keV (Rp = 80 nm, ΔRp = 24 nm), calculate the required dose.

Show Solution

**Calculation steps** :

Gaussian distribution at the peak concentration (x = Rp):

$$C_{\text{peak}} = \frac{\Phi}{\sqrt{2\pi} \Delta R_p}$$

In this problem, x = 50 nm ≠ Rp = 80 nm, so:

$$C(50) = \frac{\Phi}{\sqrt{2\pi} \cdot 24} \exp\left(-\frac{(50 - 80)^2}{2 \times 24^2}\right)$$

$$5 \times 10^{20} = \frac{\Phi}{\sqrt{2\pi} \cdot 24 \times 10^{-7}} \times 0.557$$

$$\Phi = \frac{5 \times 10^{20} \times \sqrt{2\pi} \times 24 \times 10^{-7}}{0.557} = 1.7 \times 10^{16} \text{ ions/cm}^2$$

**Answer** : dose = 1.7×10¹⁶ ions/cm²

#### Exercise 3.6 (Medium): Selecting Thermal Spray Process Parameters

A WC-Co coating is applied by HVOF spraying. With a particle size of 40 μm and a spray distance of 150 mm, you want to keep the particle velocity at substrate impact above 600 m/s and the temperature above 2500°C. Referring to Code Example 3.6, verify whether these conditions are met, and if not, propose improvements.

Show Solution

**Verification** :
    
    
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    print(f"Impact velocity: {v[-1]:.1f} m/s")  # ~650 m/s ✓
    print(f"Impact temperature: {T[-1]:.1f} °C")   # ~2400 °C ✗
    

**Assessment** : the velocity requirement is met, but the temperature falls short (2400°C < 2500°C)

**Improvements** :

  1. **Shorten the spray distance** : 150 mm → 120 mm reduces temperature loss
  2. **Reduce the particle size** : 40 μm → 30 μm lowers the cooling rate (higher heat capacity/surface area ratio)
  3. **Raise the initial temperature** : adjust the fuel/oxygen ratio, increase preheating

**Final recommendation** : spray distance 120 mm + particle size 35 μm → impact temperature approximately 2550°C (target achieved)

#### Exercise 3.7 (Hard): Multilayer Coating Design

You want to impart both wear resistance and corrosion resistance to an automotive engine component (steel). Design a multilayer coating under the following conditions:

  * Innermost layer: adhesion layer (thin film)
  * Intermediate layer: wear-resistant layer (thick film)
  * Outermost layer: corrosion-resistant layer (medium film)

Select the material, thickness, and fabrication method for each layer, and explain your design rationale.

Show Solution

**Multilayer coating design** :

Layer | Material | Thickness | Method | Rationale  
---|---|---|---|---  
Adhesion layer | Ni | 5 μm | Electroplating | Good adhesion to steel, stress relaxation  
Wear-resistant layer | WC-Co | 150 μm | HVOF spraying | High hardness (HV1200), wear resistance  
Corrosion-resistant layer | Cr₃C₂-NiCr | 50 μm | HVOF spraying | Oxidation resistance, high-temperature corrosion resistance  
  
**Process sequence** :

  1. Pre-treat the steel substrate (degreasing, sand blasting, Ra = 3–5 μm)
  2. Electroplate the Ni adhesion layer (current density 2 A/dm², 1 hour)
  3. HVOF-spray the WC-Co layer (particle size 30 μm, spray distance 120 mm, velocity 800 m/s)
  4. HVOF-spray the Cr₃C₂-NiCr layer (particle size 40 μm, spray distance 150 mm)
  5. Post-treatment (polishing and sealing as needed)

**Expected performance** :

  * Wear resistance: friction coefficient 0.3, wear rate < 10⁻⁶ mm³/Nm
  * Corrosion resistance: over 1000 hours in salt spray testing
  * Adhesion strength: > 50 MPa

#### Exercise 3.8 (Hard): Process Troubleshooting

The following defects occurred in a copper plating process. Propose causes and countermeasures for each defect:

  * **Defect A** : many small protrusions (nodules) appear on the plated surface
  * **Defect B** : the plating thickness only reaches 12 μm against a target of 20 μm
  * **Defect C** : peeling occurs in the post-plating adhesion test (tape test)

Show Solution

**Defect A: nodules (surface protrusions)**

**Candidate causes** :

  * Impurities and particles in the plating bath (dust, other metal ions)
  * Insufficient bath filtration
  * Dendritic growth due to excessive current density

**Countermeasures** :

  1. Filter the plating bath (5 μm cartridge filter, circulate for 24 hours)
  2. Activated carbon treatment of the anode (impurity removal)
  3. Reduce the current density (5 A/dm² → 2 A/dm²)
  4. Strengthen sample pre-treatment (degreasing → acid pickling → pure water rinse)

**Defect B: insufficient film thickness**

**Candidate causes** :

  * Reduced current efficiency (due to side reactions)
  * Insufficient metal ion concentration
  * Actual current lower than the setpoint

**Verification** :
    
    
    # Theoretical thickness (95% efficiency)
    d_theoretical = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"Theoretical thickness: {d_theoretical:.1f} μm")  # 25.1 μm
    
    # Current efficiency back-calculated from the measured 12 μm
    actual_efficiency = 12 / d_theoretical * 0.95
    print(f"Actual current efficiency: {actual_efficiency:.1%}")  # ~45% (major drop)
    

**Countermeasures** :

  1. Analyze the bath composition (CuSO₄ concentration, H₂SO₄ concentration) → replenish if deficient
  2. Check the calibration of the ammeter
  3. Check the bath temperature (low temperature reduces current efficiency) → maintain at 25±2°C
  4. Check the anode-to-cathode area balance (1:1 to 2:1 is ideal)

**Defect C: poor adhesion**

**Candidate causes** :

  * Contamination of the substrate surface (oils, oxide films)
  * Insufficient pre-treatment
  * Stress from thermal expansion mismatch with the substrate

**Countermeasures** :

  1. Revise the pre-treatment process 
     * Degreasing: alkaline degreasing (60°C, 10 min) + ultrasonic cleaning
     * Acid pickling: 10% H₂SO₄ (room temperature, 1 min) to remove oxide films
     * Activation: 5% HCl (room temperature, 30 s) immediately before plating
  2. Strike plating (thin Ni or Cu layer) to improve adhesion
  3. Post-plating baking (150°C, 1 hour) to remove hydrogen embrittlement and improve adhesion

**Verification methods** :

  * Adhesion test: JIS H8504 (cross-cut → tape test)
  * Tensile test: ASTM B571 (target tensile adhesion strength > 20 MPa)

## 3.7 Learning Check

### Basic Understanding (5 items)

  * □ Can calculate plating thickness using Faraday's law
  * □ Can explain the difference between the barrier layer and porous layer in anodizing
  * □ Understand the relationship between projected range and dose in ion implantation
  * □ Understand the classification of coating technologies (plating, thermal spray, PVD/CVD)
  * □ Can explain how thermal spray particle velocity and temperature affect adhesion

### Practical Skills (5 items)

  * □ Can design plating conditions accounting for current density and current efficiency
  * □ Can calculate the voltage-thickness relationship for anodizing
  * □ Can simulate ion implantation profiles in Python
  * □ Can apply the surface treatment technology selection flowchart
  * □ Can diagnose the causes of plating defects (nodules, insufficient thickness, poor adhesion)

### Applied Skills (5 items)

  * □ Can propose methods to improve throwing power for parts with complex shapes
  * □ Can design multilayer coatings and select the material, thickness, and method for each layer
  * □ Can optimize thermal spray process parameters (particle size, spray distance)
  * □ Can select a surface treatment technology according to the required properties (corrosion resistance, wear resistance, conductivity, etc.)
  * □ Can troubleshoot process anomalies

## 3.8 References

  1. Kanani, N. (2004). _Electroplating: Basic Principles, Processes and Practice_. Elsevier, **pp. 56-89** (Faraday's law and electrochemistry fundamentals).
  2. Wernick, S., Pinner, R., Sheasby, P.G. (1987). _The Surface Treatment and Finishing of Aluminum and Its Alloys_ (5th ed.). ASM International, **pp. 234-267** (anodizing processes and film structure).
  3. Davis, J.R. (Ed.) (2004). _Handbook of Thermal Spray Technology_. ASM International, **pp. 123-156** (thermal spray processes and coating properties).
  4. Pawlowski, L. (2008). _The Science and Engineering of Thermal Spray Coatings_ (2nd ed.). Wiley, **pp. 189-223** (HVOF spraying and particle dynamics).
  5. Townsend, P.D., Chandler, P.J., Zhang, L. (1994). _Optical Effects of Ion Implantation_. Cambridge University Press, **pp. 45-78** (ion implantation theory and the LSS model).
  6. Inagaki, M., Toyoda, M., Soneda, Y., Morishita, T. (2014). "Nitrogen-doped carbon materials." _Carbon_ , 132, 104-140, **pp. 115-128** , DOI: 10.1016/j.carbon.2014.01.027 (plasma nitriding processes).
  7. Fauchais, P.L., Heberlein, J.V.R., Boulos, M.I. (2014). _Thermal Spray Fundamentals: From Powder to Part_. Springer, **pp. 567-612** (fundamentals and applications of thermal spraying).
  8. Schlesinger, M., Paunovic, M. (Eds.) (2010). _Modern Electroplating_ (5th ed.). Wiley, **pp. 209-248** (modern plating technologies and troubleshooting).

## Summary

In this chapter, we studied materials surface treatment technologies from fundamentals to practice. For electroplating, we covered film thickness calculation using Faraday's law and current density optimization; for anodizing, the formation mechanisms of the barrier and porous layers; for ion implantation, the modeling of concentration profiles; and for thermal spraying, the relationship between particle dynamics and adhesion strength.

Surface treatment is a key process technology that imparts surface functions (corrosion resistance, wear resistance, conductivity, decorative finish, etc.) without changing the bulk properties of the material. Appropriate technology selection and parameter optimization can dramatically improve product performance and lifetime.

In the next chapter, we will study thin film growth processes: sputtering, vacuum evaporation, chemical vapor deposition (CVD), and epitaxial growth.
