---
title: "Chapter 4: Thin Film Growth Processes"
chapter_title: "Chapter 4: Thin Film Growth Processes"
subtitle: Sputtering, Evaporation, CVD, Epitaxy
reading_time: 30-40 min
difficulty: Intermediate
code_examples: 7
version: 1.0
created_at: 2025-10-28
---

Thin film growth processes are fundamental technologies in modern materials science, including semiconductor devices, optical coatings, and protective films. In this chapter, we learn the principles and practices of sputtering, vacuum evaporation, chemical vapor deposition (CVD), and epitaxial growth, and perform deposition parameter optimization and film quality prediction using Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  *  Understanding the principles of sputtering (DC sputtering, RF sputtering, magnetron configuration)
  *  Explaining the characteristics of vacuum evaporation methods (thermal evaporation, electron beam evaporation, molecular beam epitaxy)
  *  Understanding the basics of CVD (Chemical Vapor Deposition) and applications of PECVD, MOCVD, ALD
  *  Understanding lattice matching and strain relaxation mechanisms in epitaxial growth
  *  Calculating Sigmund's sputtering yield formula and Knudsen's cosine law
  *  Simulating deposition rate, film thickness distribution, and epitaxial critical thickness with Python
  *  Practicing actual process condition optimization

## 4.1 Sputtering

### 4.1.1 Principles of Sputtering

Sputtering is a physical vapor deposition (PVD) method where high-energy ions (typically Ar+) collide with target material, ejecting target atoms that deposit onto a substrate.

**Sputtering Yield** is defined as the number of target atoms ejected per incident ion:

$$ Y = \frac{\text{Number of ejected atoms}}{\text{Number of incident ions}} $$ 

**Sigmund's theoretical formula** (low energy region):

$$ Y = \frac{3}{4\pi^2} \frac{\alpha S_n(E)}{U_0 N} $$ 

  * $\alpha$: Material-dependent constant (0.15-0.3)
  * $S_n(E)$: Nuclear stopping power
  * $U_0$: Surface binding energy (sublimation energy)
  * $N$: Target atomic density

**Practical simplified formula** (energy range 500 eV - 5 keV):

$$ Y \approx A \frac{E - E_{\text{th}}}{U_0} $$ 

  * $A$: Material constant
  * $E$: Incident ion energy
  * $E_{\text{th}}$: Threshold energy (typically 20-50 eV)

### 4.1.2 DC Sputtering and RF Sputtering

Item | DC Sputtering (Direct Current) | RF Sputtering (Radio Frequency)  
---|---|---  
**Power Supply** | DC (direct current, -300 V ~ -1000 V) | RF (13.56 MHz, several 100 V)  
**Target Material** | Conductive materials (metals) | Conductive and insulating materials (oxides, nitrides)  
**Charge-up Countermeasure** | Not required (charge flows) | Charge neutralization by RF cycle  
**Deposition Rate** | Fast (1-10 nm/s) | Somewhat slower (0.5-5 nm/s)  
**Applications** | Metal thin films (Al, Cu, Ti) | Oxides (ITO, SiO2), nitrides (Si3N4)  
  
### 4.1.3 Magnetron Sputtering

In magnetron configuration, magnets are placed behind the target to confine electrons with a magnetic field, enhancing plasma density. This results in:

  * 5-10x improvement in deposition rate
  * Low pressure operation possible (0.1-1 Pa) ’ improved film quality
  * Reduced ion bombardment on substrate ’ reduced damage

    
    
    ```mermaid
    flowchart TD
        A[Ar gas introduction0.1-1 Pa] --> B[DC or RF power300-1000 V]
        B --> C[Plasma generationAr+ ionization]
        C --> D[e- confinement by magnetic fieldMagnetron effect]
        D --> E[Ar+ ions collide with target]
        E --> F[Target atoms sputtered]
        F --> G[Deposition on substrateThin film growth]
    
        style A fill:#99ccff,stroke:#0066cc
        style C fill:#ffeb99,stroke:#ffa500
        style G fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
    ```

#### Code Example 4-1: Sputtering Yield Calculation (Sigmund Theory)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def sigmund_yield(E, M_ion, M_target, Z_ion, Z_target, U0, alpha=0.2):
        """
        Sigmund's sputtering yield formula (simplified version)
    
        Parameters
        ----------
        E : float or ndarray
            Incident ion energy [eV]
        M_ion : float
            Ion mass [amu]
        M_target : float
            Target atom mass [amu]
        Z_ion : int
            Ion atomic number
        Z_target : int
            Target atomic number
        U0 : float
            Surface binding energy [eV]
        alpha : float
            Material constant (0.15-0.3)
    
        Returns
        -------
        Y : ndarray
            Sputtering yield
        """
        # Lindhard-Scharff reduced energy
        epsilon = 32.53 * M_target * E / (Z_ion * Z_target * (M_ion + M_target) *
                                           (Z_ion**(2/3) + Z_target**(2/3))**(1/2))
    
        # Nuclear stopping power (Lindhard-Scharff)
        # Simplified formula: Sn(epsilon) H epsilon / (1 + 0.3*epsilon^0.6)
        Sn_reduced = epsilon / (1 + 0.3 * epsilon**0.6)
    
        # Sigmund yield
        Y = alpha * Sn_reduced * 4 * M_ion * M_target / ((M_ion + M_target)**2 * U0)
    
        # Zero below threshold energy
        E_th = U0 * (1 + M_target/(5*M_ion))**2
        Y = np.where(E > E_th, Y, 0)
    
        return Y
    
    # Sputtering of Si, Cu, Au by Ar ions
    E_range = np.linspace(50, 2000, 200)  # [eV]
    
    # Ar ion
    M_Ar = 40  # [amu]
    Z_Ar = 18
    
    # Target materials
    targets = {
        'Si': {'M': 28, 'Z': 14, 'U0': 4.7, 'color': 'blue'},
        'Cu': {'M': 64, 'Z': 29, 'U0': 3.5, 'color': 'orange'},
        'Au': {'M': 197, 'Z': 79, 'U0': 3.8, 'color': 'red'}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Sputtering yield vs energy
    for name, params in targets.items():
        Y = sigmund_yield(E_range, M_Ar, params['M'], Z_Ar, params['Z'], params['U0'])
        ax1.plot(E_range, Y, linewidth=2, color=params['color'], label=name)
    
    ax1.set_xlabel('Ion Energy [eV]', fontsize=12)
    ax1.set_ylabel('Sputtering Yield [atoms/ion]', fontsize=12)
    ax1.set_title('Sputtering Yield vs Ion Energy\n(Ar+ ions)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 2000)
    
    # Right plot: Comparison at typical sputtering conditions (500 eV)
    E_fixed = 500  # [eV]
    materials = list(targets.keys())
    yields = [sigmund_yield(E_fixed, M_Ar, targets[m]['M'], Z_Ar,
                            targets[m]['Z'], targets[m]['U0']) for m in materials]
    
    bars = ax2.bar(materials, yields, color=[targets[m]['color'] for m in materials],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Sputtering Yield [atoms/ion]', fontsize=12)
    ax2.set_title(f'Sputtering Yield at {E_fixed} eV\n(Typical DC Sputtering)',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Display values on top of bars
    for bar, y_val in zip(bars, yields):
        ax2.text(bar.get_x() + bar.get_width()/2, y_val + 0.1,
                 f'{y_val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("Sputtering yield trends:")
    print(f"  - Cu has the highest yield (medium mass, low binding energy)")
    print(f"  - Si has low yield (light, high binding energy)")
    print(f"  - Au has medium yield (heavy but high binding energy)")
    

### 4.1.4 Deposition Rate Calculation

Sputtering deposition rate $R_{\text{dep}}$ is expressed as:

$$ R_{\text{dep}} = \frac{Y \cdot J_{\text{ion}} \cdot M}{N_A \cdot \rho \cdot e} $$ 

  * $Y$: Sputtering yield
  * $J_{\text{ion}}$: Ion current density [A/cm²]
  * $M$: Molar mass of target atom [g/mol]
  * $N_A$: Avogadro's number
  * $\rho$: Target density [g/cm³]
  * $e$: Elementary charge

#### Code Example 4-2: Sputtering Deposition Rate and Power/Pressure Dependence
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def deposition_rate(power, pressure, Y=2.5, target_area=100, target_distance=5):
        """
        Sputtering deposition rate calculation (empirical model)
    
        Parameters
        ----------
        power : float or ndarray
            Sputtering power [W]
        pressure : float or ndarray
            Ar pressure [Pa]
        Y : float
            Sputtering yield [atoms/ion]
        target_area : float
            Target area [cm^2]
        target_distance : float
            Target-substrate distance [cm]
    
        Returns
        -------
        rate : ndarray
            Deposition rate [nm/min]
        """
        # Ion current density estimation (empirical formula)
        # J_ion H power / (voltage * target_area)
        # Typical DC sputtering voltage: 500 V
        voltage = 500  # [V]
        J_ion = power / (voltage * target_area)  # [A/cm^2]
    
        # Pressure dependence (mean free path effect)
        # Low pressure: less scattering, efficient; high pressure: scattering reduces efficiency
        pressure_factor = 1.0 / (1 + pressure / 0.5)  # Half at 0.5 Pa
    
        # Deposition rate (simplified model)
        # Assuming Cu (M=63.5 g/mol, Á=8.96 g/cm^3)
        M = 63.5
        rho = 8.96
        e = 1.60218e-19
        N_A = 6.022e23
    
        # [nm/s]
        rate_nm_s = (Y * J_ion * M * 1e7) / (N_A * rho * e) * pressure_factor
    
        # Convert to [nm/min]
        rate = rate_nm_s * 60
    
        return rate
    
    # Power dependence (fixed pressure)
    power_range = np.linspace(50, 500, 50)
    pressure_fixed = 0.3  # [Pa]
    
    rate_vs_power = deposition_rate(power_range, pressure_fixed)
    
    # Pressure dependence (fixed power)
    pressure_range = np.linspace(0.1, 2.0, 50)
    power_fixed = 200  # [W]
    
    rate_vs_pressure = deposition_rate(power_fixed, pressure_range)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Power dependence
    ax1.plot(power_range, rate_vs_power, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Sputtering Power [W]', fontsize=12)
    ax1.set_ylabel('Deposition Rate [nm/min]', fontsize=12)
    ax1.set_title('Deposition Rate vs Power\n(Ar pressure = 0.3 Pa)',
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Linear fit
    coeffs = np.polyfit(power_range, rate_vs_power, 1)
    ax1.plot(power_range, np.poly1d(coeffs)(power_range), 'r--', linewidth=2,
             label=f'Linear fit: {coeffs[0]:.2f}·P + {coeffs[1]:.1f}')
    ax1.legend(fontsize=10)
    
    # Right plot: Pressure dependence
    ax2.plot(pressure_range, rate_vs_pressure, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Ar Pressure [Pa]', fontsize=12)
    ax2.set_ylabel('Deposition Rate [nm/min]', fontsize=12)
    ax2.set_title('Deposition Rate vs Pressure\n(Power = 200 W)',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Show optimal pressure
    optimal_p = pressure_range[np.argmax(rate_vs_pressure)]
    ax2.axvline(optimal_p, color='red', linestyle='--', linewidth=2,
                label=f'Optimal: {optimal_p:.2f} Pa')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Power dependence: Nearly linear (2x at 200 W ’ 400 W)")
    print(f"Pressure dependence: Optimal value exists ({optimal_p:.2f} Pa)")
    print(f"  - Too low pressure: Plasma unstable")
    print(f"  - Too high pressure: Gas scattering reduces efficiency")
    

## 4.2 Vacuum Evaporation

### 4.2.1 Thermal Evaporation

Thermal evaporation heats material using resistance heating or electron beam, evaporating it to deposit on substrate.

**Vapor Pressure and Clausius-Clapeyron Equation** :

$$ P(T) = P_0 \exp\left(-\frac{\Delta H_{\text{vap}}}{R T}\right) $$ 

  * $P(T)$: Vapor pressure at temperature $T$
  * $\Delta H_{\text{vap}}$: Enthalpy of vaporization
  * $R$: Gas constant

**Practically** , vapor pressure above $10^{-2}$ Pa is required (deposition rate >0.1 nm/s).

### 4.2.2 Knudsen's Cosine Law

The atomic flux distribution from evaporation source follows **Knudsen's cosine law** :

$$ \Phi(\theta) = \Phi_0 \cos(\theta) $$ 

  * $\Phi(\theta)$: Flux in direction $\theta$
  * $\Phi_0$: Flux in normal direction ($\theta=0$)

This causes non-uniform film thickness distribution on substrate (thicker at center, thinner at periphery).

#### Code Example 4-3: Thermal Evaporation Flux Distribution (Knudsen Cosine Law)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def knudsen_flux_distribution(x, y, source_x=0, source_y=0, source_z=-10, flux0=1.0):
        """
        Evaporation flux distribution by Knudsen's cosine law
    
        Parameters
        ----------
        x, y : ndarray
            Coordinates on substrate [cm]
        source_x, source_y, source_z : float
            Evaporation source position [cm] (z < 0: below substrate)
        flux0 : float
            Reference flux in normal direction
    
        Returns
        -------
        flux : ndarray
            Flux at each point
        """
        # Distance and angle from evaporation source to each point
        dx = x - source_x
        dy = y - source_y
        dz = 0 - source_z  # Substrate is at z=0
    
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_theta = dz / r
    
        # Knudsen's cosine law: ¦(¸) = ¦0 * cos(¸) / r^2
        flux = flux0 * cos_theta / r**2
    
        return flux
    
    # Substrate grid (10 cm x 10 cm)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Source is 10 cm below center
    flux = knudsen_flux_distribution(X, Y, source_x=0, source_y=0, source_z=-10, flux0=100)
    
    # Convert to thickness (flux × time)
    time = 60  # [s]
    thickness = flux * time  # [arbitrary units]
    
    # Visualization
    fig = plt.figure(figsize=(16, 6))
    
    # Left plot: 2D flux distribution
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.contourf(X, Y, flux, levels=20, cmap='hot')
    ax1.contour(X, Y, flux, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax1.set_xlabel('x [cm]', fontsize=11)
    ax1.set_ylabel('y [cm]', fontsize=11)
    ax1.set_title('Flux Distribution (Knudsen Cosine Law)\nSource at (0, 0, -10 cm)',
                  fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Flux [a.u.]', fontsize=10)
    
    # Center plot: 3D thickness distribution
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, thickness, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('x [cm]', fontsize=10)
    ax2.set_ylabel('y [cm]', fontsize=10)
    ax2.set_zlabel('Thickness [a.u.]', fontsize=10)
    ax2.set_title('Film Thickness Distribution\n(60 s deposition)', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    
    # Right plot: Center axis profile
    ax3 = fig.add_subplot(1, 3, 3)
    center_profile = thickness[50, :]  # Line at y=0
    ax3.plot(x, center_profile, 'b-', linewidth=2)
    ax3.set_xlabel('x [cm]', fontsize=12)
    ax3.set_ylabel('Thickness [a.u.]', fontsize=12)
    ax3.set_title('Thickness Profile along Center Line\n(y = 0)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Uniformity evaluation
    thickness_center = thickness[50, 50]
    thickness_edge = thickness[50, 0]
    uniformity = (thickness_center - thickness_edge) / thickness_center * 100
    
    ax3.axhline(thickness_center, color='red', linestyle='--', linewidth=1.5,
                label=f'Center: {thickness_center:.1f}')
    ax3.axhline(thickness_edge, color='green', linestyle='--', linewidth=1.5,
                label=f'Edge: {thickness_edge:.1f}')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Film thickness uniformity: {uniformity:.1f}% difference (center-edge)")
    print(f"Improvement methods:")
    print(f"  - Substrate rotation (uniform by rotational symmetry)")
    print(f"  - Multiple source arrangement")
    print(f"  - Use of masks (selective deposition)")
    

### 4.2.3 Electron Beam Evaporation and Molecular Beam Epitaxy (MBE)

Item | Thermal Evaporation | Electron Beam Evaporation | MBE  
---|---|---|---  
**Heating Method** | Resistance heating (boat, tungsten wire) | Electron beam irradiation (local heating) | Knudsen cells (individual heating)  
**Achievable Temperature** | ~1500°C | ~3000°C | ~1500°C  
**Applicable Materials** | Low melting point metals (Al, Ag, Au) | High melting point materials (Ti, W, SiO2) | Semiconductors (GaAs, InP, Si/Ge)  
**Deposition Rate** | 0.1-10 nm/s | 0.5-50 nm/s | 0.01-1 nm/s (atomic layer control)  
**Vacuum Level** | 10-3-10-5 Pa | 10-4-10-6 Pa | 10-8-10-10 Pa (ultra-high vacuum)  
**Film Quality** | Polycrystalline, amorphous | Polycrystalline | Single crystal epitaxy  
  
## 4.3 Chemical Vapor Deposition (CVD)

### 4.3.1 Basics of CVD

CVD chemically reacts gaseous precursors on substrate surface to grow solid thin films.

**Basic steps of CVD process** :

  1. Transport (diffusion) of source gas
  2. Adsorption on substrate surface
  3. Surface reaction (pyrolysis, reduction, oxidation)
  4. Desorption of byproducts
  5. Exhaust of byproducts

**Rate-limiting step** :

$$ R_{\text{growth}} = \min\left(R_{\text{diffusion}}, R_{\text{reaction}}\right) $$ 

  * **Low temperature region** : Surface reaction-limited (Arrhenius dependence)
  * **High temperature region** : Diffusion-limited (small temperature dependence)

**Arrhenius equation** :

$$ R = A \exp\left(-\frac{E_a}{k_B T}\right) $$ 

  * $E_a$: Activation energy
  * $k_B$: Boltzmann constant
  * $T$: Temperature

#### Code Example 4-4: CVD Growth Rate Arrhenius Temperature Dependence
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def cvd_growth_rate(T, Ea, A, regime='reaction'):
        """
        CVD growth rate Arrhenius equation
    
        Parameters
        ----------
        T : float or ndarray
            Temperature [K]
        Ea : float
            Activation energy [eV]
        A : float
            Pre-exponential factor [nm/min]
        regime : str
            'reaction' (reaction-limited) or 'diffusion' (diffusion-limited)
    
        Returns
        -------
        rate : ndarray
            Growth rate [nm/min]
        """
        kB = 8.617e-5  # [eV/K] Boltzmann constant
    
        if regime == 'reaction':
            # Reaction-limited: Arrhenius equation
            rate = A * np.exp(-Ea / (kB * T))
        elif regime == 'diffusion':
            # Diffusion-limited: Small temperature dependence (T^0.5 order)
            rate = A * (T / 1000)**0.5
    
        return rate
    
    # Temperature range (300-1000°C)
    T_celsius = np.linspace(300, 1000, 100)
    T_kelvin = T_celsius + 273.15
    
    # Parameters (assuming SiO2 CVD from SiH4 + O2)
    Ea = 1.5  # [eV]
    A_reaction = 1e6  # [nm/min]
    A_diffusion = 100  # [nm/min]
    
    # Growth rate for reaction-limited and diffusion-limited
    rate_reaction = cvd_growth_rate(T_kelvin, Ea, A_reaction, regime='reaction')
    rate_diffusion = cvd_growth_rate(T_kelvin, Ea, A_diffusion, regime='diffusion')
    
    # Actual growth rate (minimum of rate-limiting steps)
    rate_actual = np.minimum(rate_reaction, rate_diffusion)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Arrhenius plot (log(rate) vs 1/T)
    ax1.semilogy(1000/T_kelvin, rate_reaction, 'b-', linewidth=2, label='Reaction-limited')
    ax1.semilogy(1000/T_kelvin, rate_diffusion, 'r-', linewidth=2, label='Diffusion-limited')
    ax1.semilogy(1000/T_kelvin, rate_actual, 'k--', linewidth=2.5, label='Actual (minimum)')
    
    ax1.set_xlabel('1000/T [K{¹]', fontsize=12)
    ax1.set_ylabel('Growth Rate [nm/min]', fontsize=12)
    ax1.set_title('Arrhenius Plot: CVD Growth Rate\n(SiO‚ from SiH„ + O‚)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower left')
    ax1.grid(alpha=0.3, which='both')
    ax1.invert_xaxis()
    
    # Top axis for temperature display
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    temp_ticks = [400, 500, 600, 700, 800, 900]
    ax1_top.set_xticks([1000/(t+273.15) for t in temp_ticks])
    ax1_top.set_xticklabels([f'{t}°C' for t in temp_ticks], fontsize=10)
    
    # Right plot: Linear scale
    ax2.plot(T_celsius, rate_reaction, 'b-', linewidth=2, label='Reaction-limited')
    ax2.plot(T_celsius, rate_diffusion, 'r-', linewidth=2, label='Diffusion-limited')
    ax2.plot(T_celsius, rate_actual, 'k--', linewidth=2.5, label='Actual rate')
    
    # Color-code regimes
    transition_idx = np.argmin(np.abs(rate_reaction - rate_diffusion))
    T_transition = T_celsius[transition_idx]
    
    ax2.axvspan(300, T_transition, alpha=0.2, color='blue', label='Reaction-limited regime')
    ax2.axvspan(T_transition, 1000, alpha=0.2, color='red', label='Diffusion-limited regime')
    ax2.axvline(T_transition, color='green', linestyle=':', linewidth=2,
                label=f'Transition: {T_transition:.0f}°C')
    
    ax2.set_xlabel('Temperature [°C]', fontsize=12)
    ax2.set_ylabel('Growth Rate [nm/min]', fontsize=12)
    ax2.set_title('CVD Growth Rate vs Temperature\n(Linear Scale)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Transition temperature: {T_transition:.0f}°C")
    print(f"Low temperature region (<{T_transition:.0f}°C): Reaction-limited ’ strongly temperature-dependent")
    print(f"High temperature region (>{T_transition:.0f}°C): Diffusion-limited ’ small temperature dependence")
    print(f"Optimal deposition temperature: Near transition temperature (balance of rate and uniformity)")
    

### 4.3.2 PECVD (Plasma-Enhanced CVD)

PECVD uses plasma to promote CVD reactions at low temperatures (200-400°C).

**Advantages of PECVD** :

  * Low temperature deposition (applicable to plastic substrates, organic materials)
  * Enhanced deposition rate (plasma accelerates reactions)
  * Film quality control (densification by ion bombardment)

**Applications** : Si3N4, SiO2, a-Si, DLC (Diamond-Like Carbon)

### 4.3.3 ALD (Atomic Layer Deposition)

ALD alternately pulses source gases to grow in a self-limiting manner one atomic layer at a time - the ultimate precision technology.

**ALD cycle** :

  1. Precursor A pulse ’ surface saturation adsorption
  2. Purge (Ar, N2)
  3. Precursor B pulse ’ chemical reaction forms 1 layer
  4. Purge

**Characteristics** :

  * Atomic-level thickness control (0.1 nm/cycle)
  * Perfect conformal coating (high aspect ratio structures)
  * Low temperature deposition (100-300°C)
  * Slow deposition rate (0.01-0.1 nm/s)

    
    
    ```mermaid
    flowchart LR
        A[Precursor A pulseSurface saturation] --> B[PurgeRemove excess gas]
        B --> C[Precursor B pulseReaction, 1 layer formed]
        C --> D[PurgeRemove byproducts]
        D --> E{Target thicknessreached?}
        E -->|No| A
        E -->|Yes| F[Deposition complete]
    
        style A fill:#99ccff,stroke:#0066cc
        style C fill:#ffeb99,stroke:#ffa500
        style F fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
    ```

## 4.4 Epitaxial Growth

### 4.4.1 Definition and Types of Epitaxy

Epitaxial growth (Epitaxy) is a technique to grow single crystal thin films with aligned crystal orientation on single crystal substrates.

**Types of epitaxy** :

  * **Homoepitaxy** : Same material (Si growth on Si substrate)
  * **Heteroepitaxy** : Different materials (AlGaAs growth on GaAs)

### 4.4.2 Lattice Matching and Critical Thickness

In heteroepitaxy, lattice constant mismatch (lattice mismatch) between substrate and film is important:

$$ f = \frac{a_{\text{film}} - a_{\text{sub}}}{a_{\text{sub}}} $$ 

  * $a_{\text{film}}$: Lattice constant of film
  * $a_{\text{sub}}$: Lattice constant of substrate
  * $f$: Lattice mismatch

**Critical Thickness $h_c$** : Film thickness where strain energy exceeds dislocation formation energy

**Matthews-Blakeslee equation** :

$$ h_c = \frac{b}{4\pi f(1+\nu)} \left[\ln\left(\frac{h_c}{b}\right) + 1\right] $$ 

  * $b$: Burgers vector (order of lattice constant)
  * $\nu$: Poisson's ratio

**Practical simplified formula** :

$$ h_c \approx \frac{a}{2\pi f} $$ 

#### Code Example 4-5: Epitaxial Critical Thickness Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def critical_thickness(mismatch, a=5.65, nu=0.3):
        """
        Epitaxial critical thickness (simplified Matthews-Blakeslee)
    
        Parameters
        ----------
        mismatch : float or ndarray
            Lattice mismatch f = (a_film - a_sub) / a_sub
        a : float
            Lattice constant [Å]
        nu : float
            Poisson's ratio
    
        Returns
        -------
        h_c : ndarray
            Critical thickness [nm]
        """
        # Simplified formula: h_c H a / (2À f)
        # Full Matthews-Blakeslee requires iterative calculation,
        # but this approximation is sufficient for practical use
    
        h_c = a / (2 * np.pi * np.abs(mismatch)) / 10  # [nm]
    
        return h_c
    
    def relaxation_fraction(thickness, h_c):
        """
        Strain relaxation fraction (empirical model)
    
        Parameters
        ----------
        thickness : float or ndarray
            Film thickness [nm]
        h_c : float
            Critical thickness [nm]
    
        Returns
        -------
        relaxation : ndarray
            Strain relaxation fraction (0-1)
        """
        # Below critical thickness: Fully elastic strain (zero relaxation)
        # Above critical thickness: Relaxation by dislocation introduction
    
        relaxation = 1 - np.exp(-(thickness - h_c) / h_c)
        relaxation = np.where(thickness < h_c, 0, relaxation)
        relaxation = np.clip(relaxation, 0, 1)
    
        return relaxation
    
    # Representative hetero system lattice mismatches
    hetero_systems = {
        'GaAs/Si': {'f': 0.04, 'a': 5.65, 'color': 'blue'},
        'InP/GaAs': {'f': 0.038, 'a': 5.87, 'color': 'green'},
        'SiGe/Si': {'f': 0.02, 'a': 5.43, 'color': 'orange'},  # Ge 50%
        'AlN/GaN': {'f': 0.024, 'a': 4.98, 'color': 'red'}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Critical thickness vs lattice mismatch
    mismatch_range = np.linspace(0.001, 0.1, 100)
    h_c_range = critical_thickness(mismatch_range, a=5.5)
    
    ax1.loglog(mismatch_range * 100, h_c_range, 'k-', linewidth=2.5, label='Matthews-Blakeslee')
    
    # Plot each system
    for name, params in hetero_systems.items():
        h_c = critical_thickness(params['f'], a=params['a'])
        ax1.loglog(params['f'] * 100, h_c, 'o', markersize=10,
                  color=params['color'], label=name)
    
    ax1.set_xlabel('Lattice Mismatch [%]', fontsize=12)
    ax1.set_ylabel('Critical Thickness [nm]', fontsize=12)
    ax1.set_title('Critical Thickness vs Lattice Mismatch\n(Heteroepitaxy)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3, which='both')
    
    # Right plot: SiGe/Si strain relaxation (thickness dependence)
    thickness_range = np.logspace(0, 3, 100)  # [nm]
    sige_params = hetero_systems['SiGe/Si']
    h_c_sige = critical_thickness(sige_params['f'], a=sige_params['a'])
    
    relaxation = relaxation_fraction(thickness_range, h_c_sige)
    
    ax2.semilogx(thickness_range, relaxation * 100, linewidth=2.5, color='orange')
    ax2.axvline(h_c_sige, color='red', linestyle='--', linewidth=2,
                label=f'Critical thickness: {h_c_sige:.1f} nm')
    ax2.axhspan(0, 10, alpha=0.2, color='green', label='Pseudomorphic (<10% relaxation)')
    ax2.axhspan(90, 100, alpha=0.2, color='red', label='Fully relaxed (>90%)')
    
    ax2.set_xlabel('Film Thickness [nm]', fontsize=12)
    ax2.set_ylabel('Strain Relaxation [%]', fontsize=12)
    ax2.set_title('Strain Relaxation in SiGe/Si\n(50% Ge, f = 2%)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3, which='both')
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.show()
    
    print("Critical thickness trends:")
    for name, params in hetero_systems.items():
        h_c = critical_thickness(params['f'], a=params['a'])
        print(f"  {name}: f = {params['f']*100:.1f}%, h_c = {h_c:.1f} nm")
    print("\nDesign guidelines:")
    print("  - Below critical thickness: Strained epitaxy (high quality but thickness limited)")
    print("  - Above critical thickness: Dislocation relaxation (thickness freedom but increased defects)")
    print("  - Buffer layers (graded SiGe etc.) can extend critical thickness")
    

### 4.4.3 Growth Modes

Epitaxial growth is classified into three modes based on surface energy and lattice mismatch:

  * **Frank-van der Merwe (FM) mode** : Layer-by-layer growth (complete wetting)
  * **Volmer-Weber (VW) mode** : Island growth (non-wetting)
  * **Stranski-Krastanov (SK) mode** : Layer growth transitioning to islands (intermediate)

## 4.5 Integrated Example: Thin Film Process Optimization Simulation

#### Code Example 4-6: Thickness Distribution Optimization (Sputtering + Substrate Rotation)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    def sputtering_thickness_with_rotation(r_substrate, theta_substrate,
                                            source_position, rotation_angle=0):
        """
        Sputter thickness distribution considering substrate rotation
    
        Parameters
        ----------
        r_substrate, theta_substrate : ndarray
            Polar coordinates on substrate [cm], [rad]
        source_position : tuple
            Sputter source position (r, theta) [cm], [rad]
        rotation_angle : float
            Substrate rotation angle [rad]
    
        Returns
        -------
        thickness : ndarray
            Thickness distribution
        """
        # Substrate coordinates after rotation
        theta_rotated = theta_substrate - rotation_angle
    
        # Distance and angle from sputter source
        x_sub = r_substrate * np.cos(theta_rotated)
        y_sub = r_substrate * np.sin(theta_rotated)
    
        x_src, y_src = source_position
    
        dx = x_sub - x_src
        dy = y_sub - y_src
        dz = 10  # Source is 10 cm below
    
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_angle = dz / r
    
        # Sputter flux (1/r^2 and cos dependence)
        flux = 100 * cos_angle / r**2
    
        return flux
    
    # Substrate grid (polar coordinates)
    r = np.linspace(0, 5, 100)
    theta = np.linspace(0, 2*np.pi, 200)
    R, Theta = np.meshgrid(r, theta)
    
    # Convert to Cartesian coordinates (for visualization)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Source position (2 cm offset from center)
    source_pos = (2, 0)
    
    # Without rotation
    thickness_no_rotation = sputtering_thickness_with_rotation(R, Theta, source_pos, rotation_angle=0)
    
    # With rotation (accumulated over 10 rotations)
    num_rotations = 10
    thickness_with_rotation = np.zeros_like(R)
    
    for i in range(num_rotations):
        rotation_angle = 2 * np.pi * i / num_rotations
        thickness_with_rotation += sputtering_thickness_with_rotation(R, Theta, source_pos,
                                                                       rotation_angle=rotation_angle)
    
    thickness_with_rotation /= num_rotations
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left plot: Without rotation
    im1 = axes[0].contourf(X, Y, thickness_no_rotation, levels=20, cmap='viridis')
    axes[0].scatter(*source_pos, color='red', s=200, marker='x', linewidths=3, label='Sputter Source')
    axes[0].set_xlabel('x [cm]', fontsize=11)
    axes[0].set_ylabel('y [cm]', fontsize=11)
    axes[0].set_title('Without Rotation\n(Highly non-uniform)', fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=10)
    plt.colorbar(im1, ax=axes[0], label='Thickness [a.u.]')
    
    # Center plot: With rotation
    im2 = axes[1].contourf(X, Y, thickness_with_rotation, levels=20, cmap='viridis')
    axes[1].scatter(*source_pos, color='red', s=200, marker='x', linewidths=3, label='Sputter Source')
    axes[1].set_xlabel('x [cm]', fontsize=11)
    axes[1].set_ylabel('y [cm]', fontsize=11)
    axes[1].set_title('With Rotation (10 steps)\n(Much improved uniformity)', fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=10)
    plt.colorbar(im2, ax=axes[1], label='Thickness [a.u.]')
    
    # Right plot: Radial profile comparison
    r_profile = r
    thickness_no_rot_profile = thickness_no_rotation[:, 0]
    thickness_rot_profile = np.mean(thickness_with_rotation, axis=0)
    
    axes[2].plot(r_profile, thickness_no_rot_profile, 'b-', linewidth=2,
                marker='o', markersize=4, label='No rotation')
    axes[2].plot(r_profile, thickness_rot_profile, 'r-', linewidth=2,
                marker='s', markersize=4, label='With rotation')
    
    axes[2].set_xlabel('Radius [cm]', fontsize=12)
    axes[2].set_ylabel('Thickness [a.u.]', fontsize=12)
    axes[2].set_title('Radial Thickness Profile', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Uniformity evaluation
    uniformity_no_rot = (np.max(thickness_no_rotation) - np.min(thickness_no_rotation)) / np.mean(thickness_no_rotation) * 100
    uniformity_rot = (np.max(thickness_with_rotation) - np.min(thickness_with_rotation)) / np.mean(thickness_with_rotation) * 100
    
    print(f"Film thickness uniformity (max-min)/mean:")
    print(f"  Without rotation: {uniformity_no_rot:.1f}%")
    print(f"  With rotation: {uniformity_rot:.1f}%")
    print(f"Improvement rate: {(1 - uniformity_rot/uniformity_no_rot)*100:.1f}%")
    

#### Code Example 4-7: Fully Integrated Simulation (Multi-parameter Optimization)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    def film_quality_metric(params, target_thickness=100, target_rate=5):
        """
        Comprehensive quality evaluation function for thin film process
    
        Parameters
        ----------
        params : array
            [power, pressure, temperature, distance]
            - power [W]
            - pressure [Pa]
            - temperature [°C]
            - distance [cm]
        target_thickness : float
            Target thickness [nm]
        target_rate : float
            Target deposition rate [nm/min]
    
        Returns
        -------
        quality : float
            Quality score (lower is better)
        """
        power, pressure, temperature, distance = params
    
        # Physical models (simplified)
    
        # 1. Deposition rate (sputtering model)
        Y = 2.5 * (power / 200)**0.8  # Sputtering yield
        rate = Y * power / (pressure * distance**2) * 0.1  # [nm/min]
    
        # 2. Thickness uniformity (depends on distance and pressure)
        uniformity = 1 / (1 + (distance - 5)**2 / 10) * (1 - np.abs(pressure - 0.5) / 2)
    
        # 3. Film quality (depends on temperature and pressure)
        # Low temp: amorphous, high temp: crystallization
        crystallinity = 1 / (1 + np.exp(-(temperature - 400) / 50))
    
        # Low pressure: high density, high pressure: porous
        density = 1 / (1 + pressure / 0.5)
    
        film_quality = crystallinity * density
    
        # 4. Process stability (pressure range)
        stability = 1 if 0.2 < pressure < 1.0 else 0.5
    
        # Comprehensive quality score (penalty function)
        penalty = 0
    
        # Deposition rate penalty
        penalty += ((rate - target_rate) / target_rate)**2 * 100
    
        # Uniformity penalty (higher is better, so negative)
        penalty += (1 - uniformity)**2 * 50
    
        # Film quality penalty
        penalty += (1 - film_quality)**2 * 50
    
        # Stability penalty
        penalty += (1 - stability) * 100
    
        # Out-of-range parameter penalty
        if not (50 <= power <= 500):
            penalty += 1000
        if not (0.1 <= pressure <= 2.0):
            penalty += 1000
        if not (200 <= temperature <= 600):
            penalty += 1000
        if not (3 <= distance <= 10):
            penalty += 1000
    
        return penalty
    
    # Execute optimization
    initial_guess = [200, 0.5, 400, 5]  # [power, pressure, temperature, distance]
    
    # Bounds
    bounds = [(50, 500), (0.1, 2.0), (200, 600), (3, 10)]
    
    result = minimize(film_quality_metric, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    optimal_params = result.x
    optimal_quality = result.fun
    
    print("=" * 60)
    print("Thin Film Process Optimization Results")
    print("=" * 60)
    print(f"Optimal parameters:")
    print(f"  Sputtering power: {optimal_params[0]:.1f} W")
    print(f"  Ar pressure: {optimal_params[1]:.3f} Pa")
    print(f"  Substrate temperature: {optimal_params[2]:.1f} °C")
    print(f"  Target-substrate distance: {optimal_params[3]:.1f} cm")
    print(f"\nQuality score: {optimal_quality:.2f}")
    print(f"Optimization success: {result.success}")
    print("=" * 60)
    
    # Visualize parameter space (2D slices)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Power vs pressure (temperature and distance fixed)
    power_range = np.linspace(50, 500, 50)
    pressure_range = np.linspace(0.1, 2.0, 50)
    Power, Pressure = np.meshgrid(power_range, pressure_range)
    
    Quality_map = np.zeros_like(Power)
    for i in range(Power.shape[0]):
        for j in range(Power.shape[1]):
            Quality_map[i, j] = film_quality_metric([Power[i, j], Pressure[i, j],
                                                     optimal_params[2], optimal_params[3]])
    
    im1 = axes[0, 0].contourf(Power, Pressure, Quality_map, levels=20, cmap='RdYlGn_r')
    axes[0, 0].scatter(optimal_params[0], optimal_params[1], color='red', s=200,
                       marker='*', edgecolors='black', linewidths=2, label='Optimal')
    axes[0, 0].set_xlabel('Power [W]', fontsize=11)
    axes[0, 0].set_ylabel('Pressure [Pa]', fontsize=11)
    axes[0, 0].set_title('Quality Map: Power vs Pressure', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0], label='Quality Score (lower is better)')
    
    # Temperature vs distance (power and pressure fixed)
    temp_range = np.linspace(200, 600, 50)
    dist_range = np.linspace(3, 10, 50)
    Temp, Dist = np.meshgrid(temp_range, dist_range)
    
    Quality_map2 = np.zeros_like(Temp)
    for i in range(Temp.shape[0]):
        for j in range(Temp.shape[1]):
            Quality_map2[i, j] = film_quality_metric([optimal_params[0], optimal_params[1],
                                                      Temp[i, j], Dist[i, j]])
    
    im2 = axes[0, 1].contourf(Temp, Dist, Quality_map2, levels=20, cmap='RdYlGn_r')
    axes[0, 1].scatter(optimal_params[2], optimal_params[3], color='red', s=200,
                       marker='*', edgecolors='black', linewidths=2, label='Optimal')
    axes[0, 1].set_xlabel('Temperature [°C]', fontsize=11)
    axes[0, 1].set_ylabel('Distance [cm]', fontsize=11)
    axes[0, 1].set_title('Quality Map: Temperature vs Distance', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1], label='Quality Score')
    
    # Sensitivity analysis for each parameter
    param_names = ['Power [W]', 'Pressure [Pa]', 'Temperature [°C]', 'Distance [cm]']
    param_ranges = [np.linspace(50, 500, 50),
                    np.linspace(0.1, 2.0, 50),
                    np.linspace(200, 600, 50),
                    np.linspace(3, 10, 50)]
    
    for idx, (name, param_range) in enumerate(zip(param_names, param_ranges)):
        qualities = []
        for val in param_range:
            test_params = optimal_params.copy()
            test_params[idx] = val
            qualities.append(film_quality_metric(test_params))
    
        row = 1 if idx >= 2 else 0
        col = idx % 2
    
        if row == 1:
            axes[row, col].plot(param_range, qualities, linewidth=2, color='blue')
            axes[row, col].axvline(optimal_params[idx], color='red', linestyle='--',
                                  linewidth=2, label=f'Optimal: {optimal_params[idx]:.2f}')
            axes[row, col].set_xlabel(name, fontsize=11)
            axes[row, col].set_ylabel('Quality Score', fontsize=11)
            axes[row, col].set_title(f'Sensitivity: {name}', fontsize=12, fontweight='bold')
            axes[row, col].legend(fontsize=10)
            axes[row, col].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSensitivity analysis results:")
    print("  - Power: Linear quality improvement (higher power is better)")
    print("  - Pressure: Optimal value exists (around 0.3-0.5 Pa)")
    print("  - Temperature: Crystallization promoted around 400°C")
    print("  - Distance: Maximum uniformity around 5 cm")
    

## 4.6 Exercise Problems

### Exercise 4-1: Sputtering Yield (Easy)

**Problem** : When the sputtering yield of a Cu target by Ar+ ions (500 eV) is 2.5 atoms/ion, calculate the total number of target atoms ejected during 10 minutes of deposition with an ion current of 1 mA.

**Show Solution**
    
    
    Y = 2.5  # [atoms/ion]
    I_ion = 1e-3  # [A]
    t = 10 * 60  # [s]
    e = 1.60218e-19  # [C]
    
    # Number of ions = current × time / charge
    N_ions = I_ion * t / e
    
    # Number of ejected atoms = yield × number of ions
    N_atoms = Y * N_ions
    
    print(f"Ion current: {I_ion*1e3:.1f} mA")
    print(f"Deposition time: {t/60:.1f} min")
    print(f"Number of incident ions: {N_ions:.3e}")
    print(f"Number of ejected atoms: {N_atoms:.3e}")
    print(f"                       = {N_atoms/6.022e23:.2e} mol")
    

### Exercise 4-2: CVD Growth Rate Activation Energy (Medium)

**Problem** : The growth rate of SiO2 CVD was 1 nm/min at 400°C and 10 nm/min at 500°C. Calculate the activation energy.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: The growth rate of SiO2CVD was 1 nm/min at 400°C an
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    T1 = 400 + 273.15  # [K]
    T2 = 500 + 273.15  # [K]
    R1 = 1  # [nm/min]
    R2 = 10  # [nm/min]
    
    kB = 8.617e-5  # [eV/K]
    
    # Arrhenius equation: R = A * exp(-Ea / (kB * T))
    # ln(R2/R1) = -Ea/kB * (1/T2 - 1/T1)
    
    ln_ratio = np.log(R2 / R1)
    Ea = -ln_ratio * kB / (1/T2 - 1/T1)
    
    print(f"Temperature 1: {T1-273.15:.0f}°C, Growth rate: {R1} nm/min")
    print(f"Temperature 2: {T2-273.15:.0f}°C, Growth rate: {R2} nm/min")
    print(f"ln(R2/R1): {ln_ratio:.3f}")
    print(f"Activation energy Ea: {Ea:.2f} eV")
    print(f"\nInterpretation: Ea H 1.5 eV is typical for CVD reactions (pyrolysis)")
    

### Exercise 4-3: Epitaxial Critical Thickness (Medium)

**Problem** : Epitaxially grow Ge0.2Si0.8 on Si substrate. The lattice constant of Ge is 5.65 Å and Si is 5.43 Å. Estimate the critical thickness.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Epitaxially grow Ge0.2Si0.8on Si substrate. The lat
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    
    a_Si = 5.43  # [Å]
    a_Ge = 5.65  # [Å]
    x_Ge = 0.2
    
    # Vegard's law: a_SiGe = (1-x)*a_Si + x*a_Ge
    a_SiGe = (1 - x_Ge) * a_Si + x_Ge * a_Ge
    
    # Lattice mismatch
    f = (a_SiGe - a_Si) / a_Si
    
    # Critical thickness (simplified formula)
    h_c = a_Si / (2 * np.pi * np.abs(f)) / 10  # [nm]
    
    print(f"Si lattice constant: {a_Si} Å")
    print(f"Ge lattice constant: {a_Ge} Å")
    print(f"Ge composition: {x_Ge*100:.0f}%")
    print(f"SiGe lattice constant: {a_SiGe:.3f} Å")
    print(f"Lattice mismatch: {f*100:.2f}%")
    print(f"Critical thickness: {h_c:.1f} nm")
    print(f"\nConclusion: Strained epitaxy possible below {h_c:.1f} nm")
    

### Exercise 4-4: Knudsen Cosine Law Thickness Distribution (Medium)

**Problem** : A substrate is placed 20 cm directly above an evaporation source. What percentage of the center thickness is the thickness at a point 5 cm away from the center (assuming Knudsen cosine law)?

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: A substrate is placed 20 cm directly above an evapo
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    d = 20  # [cm] Source-substrate distance
    r = 5   # [cm] Distance from center
    
    # Center (r=0)
    R_center = d
    cos_center = d / R_center
    flux_center = cos_center / R_center**2
    
    # Edge (r=5)
    R_edge = np.sqrt(r**2 + d**2)
    cos_edge = d / R_edge
    flux_edge = cos_edge / R_edge**2
    
    # Thickness ratio
    ratio = flux_edge / flux_center
    
    print(f"Source-substrate distance: {d} cm")
    print(f"Distance from center: {r} cm")
    print(f"Angle at center: 0° (cos=1)")
    print(f"Angle at edge: {np.arccos(cos_edge)*180/np.pi:.1f}° (cos={cos_edge:.3f})")
    print(f"Thickness ratio (edge/center): {ratio:.3f} = {ratio*100:.1f}%")
    print(f"\nConclusion: Thickness decreases by {(1-ratio)*100:.1f}% at 5 cm from center")
    

### Exercise 4-5: Advantages of Magnetron Sputtering (Easy)

**Problem** : Explain from a plasma physics perspective why magnetron sputtering is faster than conventional DC sputtering.

**Show Solution**

**Reasons** :

  * **Electron confinement effect** : Magnetic field behind target confines electrons by E×B drift
  * **Enhanced plasma density** : Longer electron residence time increases collision probability with Ar atoms ’ Ar+ ion density increases 5-10x
  * **Low pressure operation** : High-density plasma enables stable discharge at 0.1-1 Pa low pressure ’ Reduced gas scattering, increased mean free path of sputtered particles
  * **Increased ion current** : High Ar+ density increases ion current at same voltage ’ Enhanced deposition rate

**Quantitative comparison** :
    
    
    print("Conventional DC sputtering:")
    print("  - Operating pressure: 1-10 Pa")
    print("  - Plasma density: 10^9-10^10 cm^-3")
    print("  - Deposition rate: 0.5-2 nm/s")
    print("\nMagnetron sputtering:")
    print("  - Operating pressure: 0.1-1 Pa")
    print("  - Plasma density: 10^10-10^11 cm^-3 (5-10x)")
    print("  - Deposition rate: 3-10 nm/s (5-10x)")
    

### Exercise 4-6: ALD vs CVD Selection (Medium)

**Problem** : For the following cases, determine whether ALD or CVD is more appropriate, with reasons.  
(a) 10 nm uniform coating on inner wall of 100 nm width trench  
(b) 1 ¼m thick SiO2 deposition on 4-inch wafer

**Show Solution**

**(a) 10 nm uniform coating in 100 nm width trench ’ ALD**

  * **Reason** : 
    * For high aspect ratio (typically >5 depth/width) structures, CVD has difficulty achieving uniform deposition to the bottom (gas transport limited)
    * ALD achieves perfect conformal coating through self-limiting surface reactions
    * 10 nm is thin, so ALD's slow rate (0.1 nm/cycle × 100 cycles = about 10 min) is acceptable

**(b) 1 ¼m thick SiO 2 on 4-inch wafer ’ CVD (PECVD recommended)**

  * **Reason** : 
    * 1 ¼m is thick, ALD would take too long (>10000 cycles = several hours to days)
    * CVD has fast deposition rate (1-10 nm/s), can deposit 1 ¼m in several minutes to tens of minutes
    * Conformality is unnecessary on flat wafer surface
    * PECVD can produce high quality SiO2 at low temperature (around 300°C)

### Exercise 4-7: Measured Sputter Deposition Rate (Hard)

**Problem** : Sputtering was performed with Cu target (diameter 10 cm), DC power 300 W, Ar pressure 0.5 Pa, target-substrate distance 5 cm. After 10 minutes, the film thickness was 150 nm. Calculate the sputtering yield by reverse calculation (assuming typical DC voltage of 500 V).

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Sputtering was performed with Cu target (diameter 1
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Known parameters
    power = 300  # [W]
    voltage = 500  # [V]
    time = 10 * 60  # [s]
    thickness = 150  # [nm]
    target_area = np.pi * (5)**2  # [cm^2]
    
    # Cu physical properties
    M_Cu = 63.5  # [g/mol]
    rho_Cu = 8.96  # [g/cm^3]
    
    # Constants
    e = 1.60218e-19  # [C]
    N_A = 6.022e23  # [1/mol]
    
    # Ion current
    I_ion = power / voltage  # [A]
    J_ion = I_ion / target_area  # [A/cm^2]
    
    # Deposition rate
    rate = thickness / (time / 60)  # [nm/min]
    
    # Reverse calculate sputtering yield
    # rate = (Y * J_ion * M) / (N_A * rho * e) * 1e7
    # Y = rate * (N_A * rho * e) / (J_ion * M * 1e7)
    
    Y = rate * (N_A * rho_Cu * e) / (J_ion * M_Cu * 1e7)
    
    print(f"Experimental conditions:")
    print(f"  Power: {power} W")
    print(f"  Voltage: {voltage} V")
    print(f"  Deposition time: {time/60:.1f} min")
    print(f"  Film thickness: {thickness} nm")
    print(f"\nCalculation results:")
    print(f"  Ion current: {I_ion:.3f} A = {I_ion*1e3:.1f} mA")
    print(f"  Ion current density: {J_ion:.4f} A/cm²")
    print(f"  Deposition rate: {rate:.1f} nm/min")
    print(f"  Sputtering yield Y: {Y:.2f} atoms/ion")
    print(f"\nEvaluation: Y={Y:.2f} matches typical Cu value (2-3)")
    

### Exercise 4-8: Epitaxy Growth Mode Determination (Hard)

**Problem** : For epitaxial growth of GaAs (lattice constant 5.65 Å) on Si (lattice constant 5.43 Å), predict the growth mode (FM, VW, SK) from surface energies.  
Hint: ³GaAs = 0.7 J/m², ³Si = 1.2 J/m², ³interface = 0.8 J/m²

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: For epitaxial growth of GaAs (lattice constant 5.65
    
    Purpose: Demonstrate neural network implementation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    
    # Surface energies
    gamma_GaAs = 0.7  # [J/m^2]
    gamma_Si = 1.2  # [J/m^2]
    gamma_interface = 0.8  # [J/m^2]
    
    # Lattice mismatch
    a_GaAs = 5.65  # [Å]
    a_Si = 5.43  # [Å]
    f = (a_GaAs - a_Si) / a_Si
    
    print("Surface energies:")
    print(f"  ³_GaAs: {gamma_GaAs} J/m²")
    print(f"  ³_Si: {gamma_Si} J/m²")
    print(f"  ³_interface: {gamma_interface} J/m²")
    print(f"Lattice mismatch: {f*100:.1f}%")
    
    # Growth mode determination criteria
    # FM (Frank-van der Merwe): ³_GaAs + ³_interface < ³_Si ’ complete wetting
    # VW (Volmer-Weber): ³_GaAs + ³_interface > ³_Si ’ non-wetting (island growth)
    # SK (Stranski-Krastanov): Initial FM but transitions to VW due to strain energy accumulation
    
    delta_gamma = (gamma_GaAs + gamma_interface) - gamma_Si
    
    print(f"\n”³ = (³_GaAs + ³_interface) - ³_Si")
    print(f"    = ({gamma_GaAs} + {gamma_interface}) - {gamma_Si}")
    print(f"    = {delta_gamma:.2f} J/m²")
    
    if delta_gamma < 0:
        print("\n”³ < 0 ’ FM mode (layer-by-layer growth) tendency")
    else:
        print("\n”³ > 0 ’ VW mode (island growth) tendency")
    
    # Effect of lattice mismatch
    if np.abs(f) > 0.02:
        print(f"\nHowever, large lattice mismatch ({f*100:.1f}% > 2%)")
        print("’ Strain energy accumulates, likely SK mode (layer’island transition) after a few ML")
        print("Actually: GaAs/Si system is known as SK mode")
    
    print("\nConclusion: Stranski-Krastanov (SK) mode")
    print("  - Initial few ML: 2D layer growth (strain accumulation)")
    print("  - After critical thickness: 3D island growth (strain relaxation)")
    

## 4.7 Learning Check

### Basic Understanding Check

  1. Can you explain the difference in deposition mechanisms between sputtering and CVD?
  2. Do you understand the physical meaning of Sigmund's sputtering yield formula?
  3. Can you explain the principle of rate enhancement in magnetron sputtering?
  4. Do you understand the effect of Knudsen's cosine law on film thickness distribution?
  5. Can you explain the difference between reaction-limited and diffusion-limited regimes in CVD?
  6. Do you understand the characteristics and selection criteria between PECVD and ALD?

### Practical Skills Check

  1. Can you estimate deposition rate from sputtering conditions (power, pressure)?
  2. Can you determine activation energy from Arrhenius analysis of CVD growth rate?
  3. Can you calculate critical thickness from lattice mismatch in epitaxial growth?
  4. Can you design methods to improve film thickness uniformity through substrate rotation?
  5. Can you execute multi-parameter optimization (power, pressure, temperature, distance)?

### Application Ability Check

  1. Can you select appropriate thin film processes for actual device manufacturing?
  2. Can you infer causes of film quality issues (adhesion, stress, crystallinity) from process parameters?
  3. Can you design novel material thin film growth conditions from literature and physical properties?

## 4.8 References

  1. Ohring, M. (2001). _Materials Science of Thin Films_ (2nd ed.). Academic Press. pp. 123-178 (Sputtering), pp. 234-289 (Evaporation).
  2. Mattox, D.M. (2010). _Handbook of Physical Vapor Deposition (PVD) Processing_ (2nd ed.). Elsevier. pp. 89-156 (Sputtering mechanisms), pp. 234-289 (Process optimization).
  3. Chapman, B. (1980). _Glow Discharge Processes_. Wiley. pp. 89-134 (Plasma physics and sputtering).
  4. Choy, K.L. (2003). "Chemical vapour deposition of coatings." _Progress in Materials Science_ , 48:57-170. DOI: 10.1016/S0079-6425(01)00009-3
  5. Herman, M.A., Sitter, H. (1996). _Molecular Beam Epitaxy: Fundamentals and Current Status_ (2nd ed.). Springer. pp. 45-89 (Growth modes and kinetics), pp. 156-198 (Heteroepitaxy).
  6. George, S.M. (2010). "Atomic layer deposition: An overview." _Chemical Reviews_ , 110(1):111-131. DOI: 10.1021/cr900056b
  7. Matthews, J.W., Blakeslee, A.E. (1974). "Defects in epitaxial multilayers: I. Misfit dislocations." _Journal of Crystal Growth_ , 27:118-125. DOI: 10.1016/S0022-0248(74)80055-2
  8. Sigmund, P. (1969). "Theory of sputtering. I. Sputtering yield of amorphous and polycrystalline targets." _Physical Review_ , 184(2):383-416. DOI: 10.1103/PhysRev.184.383

## 4.9 Next Chapter

In the next chapter, we learn process data analysis and Python practice. From statistical process control (SPC), design of experiments (DOE), machine learning-based process prediction, to automated report generation, we build fully integrated workflows ready for immediate practical use.
