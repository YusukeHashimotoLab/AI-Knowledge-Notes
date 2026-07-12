---
title: "Chapter 4: Electrical and Magnetic Properties"
chapter_title: "Chapter 4: Electrical and Magnetic Properties"
subtitle: The Physics of Conduction Phenomena and Magnetism
difficulty: Intermediate to Advanced
---

## What You Will Learn in This Chapter

### Learning Objectives (3 Levels)

#### Basic Level

  * Explain the mechanism of electrical conduction using the Drude model
  * Understand the relationship between the Hall effect, carrier density, and mobility
  * Explain the differences between ferromagnetism, antiferromagnetism, and paramagnetism

#### Intermediate Level

  * Calculate electrical conductivity from band structure
  * Understand and compute the relationship between magnetic moment and spin density
  * Explain how spin-orbit coupling affects magnetism

#### Advanced Level

  * Quantitatively predict electrical and magnetic properties from DFT calculation results
  * Understand the basic mechanism of superconductivity (BCS theory)
  * Compare experimental data with DFT calculations and assess the validity of functionals

## Classical Theory of Electrical Conduction: The Drude Model

### Free Electron Approximation

This is the simplest model, which treats the valence electrons in a metal as "freely moving particles". Electrons move freely through the lattice of atomic nuclei, and scattering occurs due to lattice vibrations (phonons) and impurities.

### Basic Equation of the Drude Model

The equation of motion for an electron in an electric field $\mathbf{E}$:

$$ m^* \frac{d\mathbf{v}}{dt} = -e\mathbf{E} - \frac{m^*\mathbf{v}}{\tau} $$ 

  * $m^*$: effective mass of the electron
  * $\mathbf{v}$: drift velocity
  * $\tau$: relaxation time (mean time between collisions)
  * $-e$: electron charge

In the steady state ($d\mathbf{v}/dt = 0$):

$$ \mathbf{v} = -\frac{e\tau}{m^*}\mathbf{E} $$ 

### Electrical Conductivity

The current density $\mathbf{J}$ is:

$$ \mathbf{J} = -ne\mathbf{v} = \frac{ne^2\tau}{m^*}\mathbf{E} = \sigma \mathbf{E} $$ 

Therefore, the electrical conductivity is:

$$ \sigma = \frac{ne^2\tau}{m^*} $$ 

  * $n$: carrier density [m⁻³]
  * $e$: electron charge ($1.602 \times 10^{-19}$ C)

Relationship with the **mobility $\mu$** :

$$ \mu = \frac{e\tau}{m^*}, \quad \sigma = ne\mu $$ 

#### Typical Values (Room Temperature)

Material | Electrical Conductivity [S/m] | Carrier Density [m⁻³] | Mobility [cm²/Vs]  
---|---|---|---  
Cu (copper) | 5.96 × 10⁷ | 8.5 × 10²⁸ | 43  
Si (n-type) | 10³ - 10⁵ | 10²¹ - 10²³ | 1400  
GaAs (n-type) | 10³ - 10⁶ | 10²¹ - 10²³ | 8500  
  
### Simulating the Drude Model
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    e = 1.602e-19  # Electron charge [C]
    m_e = 9.109e-31  # Electron mass [kg]
    
    def calculate_conductivity(n, tau, m_star=1.0):
        """
        Calculate electrical conductivity
    
        Parameters:
        -----------
        n : float
            Carrier density [m^-3]
        tau : float
            Relaxation time [s]
        m_star : float
            Effective mass (in units of electron mass)
    
        Returns:
        --------
        sigma : float
            Electrical conductivity [S/m]
        mu : float
            Mobility [cm^2/Vs]
        """
        m_eff = m_star * m_e
        sigma = n * e**2 * tau / m_eff  # Conductivity [S/m]
        mu = e * tau / m_eff * 1e4  # Mobility [cm^2/Vs]
        return sigma, mu
    
    # Typical metal (Cu)
    n_Cu = 8.5e28  # [m^-3]
    tau_Cu = 2.7e-14  # [s]
    sigma_Cu, mu_Cu = calculate_conductivity(n_Cu, tau_Cu, m_star=1.0)
    
    print("=== Electrical Properties of Copper (Cu) ===")
    print(f"Carrier density: {n_Cu:.2e} m^-3")
    print(f"Relaxation time: {tau_Cu:.2e} s")
    print(f"Electrical conductivity: {sigma_Cu:.2e} S/m")
    print(f"Mobility: {mu_Cu:.1f} cm^2/Vs")
    
    # Temperature dependence of mobility in a semiconductor (n-type Si)
    temperatures = np.linspace(100, 500, 50)  # [K]
    # Temperature dependence of mobility (simplified model: μ ∝ T^-3/2)
    mu_Si_ref = 1400  # [cm^2/Vs] at 300K
    T_ref = 300
    mu_Si = mu_Si_ref * (temperatures / T_ref)**(-1.5)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, mu_Si, linewidth=2, color='#f093fb')
    plt.axhline(y=1400, color='red', linestyle='--', label='Room-temperature value (300K)')
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Mobility [cm²/Vs]', fontsize=12)
    plt.title('Temperature Dependence of Mobility in n-type Si', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mobility_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Hall Effect and Carrier Measurement

### Principle of the Hall Effect

When a magnetic field is applied perpendicular to a current-carrying conductor, the Lorentz force deflects the charges, producing a transverse potential difference (Hall voltage).
    
    
    ```mermaid
    graph LR
        A[Current Jx] --> B[Magnetic field Bz]
        B --> C[Lorentz forceF = -e v × B]
        C --> D[Charge separation]
        D --> E[Hall voltage VH]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#d4edda,stroke:#28a745,stroke-width:2px
    ```

### Hall Coefficient

The Hall field $E_y$ is:

$$ E_y = R_H J_x B_z $$ 

The Hall coefficient $R_H$ is:

$$ R_H = \frac{1}{ne} $$ 

  * For holes: $R_H > 0$
  * For electrons: $R_H < 0$

**Measuring carrier density** :

$$ n = \frac{1}{|R_H| e} $$ 

**Measuring mobility** :

$$ \mu = |R_H| \sigma $$ 

### Simulating a Hall Effect Measurement
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hall_effect_simulation(n, mu, B_range, thickness=1e-3):
        """
        Simulate the Hall effect
    
        Parameters:
        -----------
        n : float
            Carrier density [m^-3]
        mu : float
            Mobility [m^2/Vs]
        B_range : array
            Magnetic field range [T]
        thickness : float
            Sample thickness [m]
        """
        e = 1.602e-19
    
        # Hall coefficient
        R_H = 1 / (n * e)  # [m^3/C]
    
        # Assume constant current density
        J = 1e6  # [A/m^2]
    
        # Hall voltage
        V_H = R_H * J * B_range * thickness  # [V]
    
        # Hall resistance
        R_Hall = V_H / (J * thickness**2)  # [Ω]
    
        return V_H, R_Hall, R_H
    
    # Example: n-type Si semiconductor
    n_Si = 1e22  # [m^-3]
    mu_Si = 0.14  # [m^2/Vs] = 1400 cm^2/Vs
    
    B_range = np.linspace(-2, 2, 100)  # [T]
    V_H, R_Hall, R_H = hall_effect_simulation(n_Si, mu_Si, B_range)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hall voltage vs magnetic field
    ax1.plot(B_range, V_H * 1e3, linewidth=2, color='#f093fb')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Magnetic field [T]', fontsize=12)
    ax1.set_ylabel('Hall voltage [mV]', fontsize=12)
    ax1.set_title('Magnetic Field Dependence of Hall Voltage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Hall resistance vs magnetic field
    ax2.plot(B_range, R_Hall, linewidth=2, color='#f5576c')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Magnetic field [T]', fontsize=12)
    ax2.set_ylabel('Hall resistance [Ω]', fontsize=12)
    ax2.set_title('Magnetic Field Dependence of Hall Resistance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hall_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Hall Effect Measurement Results ===")
    print(f"Carrier density: {n_Si:.2e} m^-3")
    print(f"Hall coefficient: {R_H:.2e} m^3/C")
    print(f"Sign of Hall coefficient: {'negative (electrons)' if R_H < 0 else 'positive (holes)'}")
    print(f"Hall voltage at 1T field: {V_H[np.argmin(np.abs(B_range - 1.0))] * 1e3:.3f} mV")
    

## Fundamentals of Magnetism

### Origin of the Magnetic Moment

The magnetic moment of atoms and molecules has two contributions:

  1. **Orbital magnetic moment** : arising from the orbital motion of the electron $$\mathbf{\mu}_L = -\frac{e}{2m_e}\mathbf{L}$$ 
  2. **Spin magnetic moment** : arising from the intrinsic angular momentum (spin) of the electron $$\mathbf{\mu}_S = -g_s \frac{e}{2m_e}\mathbf{S}$$ 

Here, $g_s \approx 2$ is the g-factor. Using the Bohr magneton $\mu_B$:

$$ \mu_B = \frac{e\hbar}{2m_e} = 9.274 \times 10^{-24} \, \text{J/T} $$ 

### Magnetization and Magnetic Susceptibility

The magnetization $M$ is the magnetic moment per unit volume:

$$ \mathbf{M} = \frac{1}{V}\sum_i \mathbf{\mu}_i $$ 

The magnetic susceptibility $\chi$ is:

$$ \mathbf{M} = \chi \mathbf{H} $$ 

  * $\chi > 0$: paramagnetism (magnetization along the field direction)
  * $\chi < 0$: diamagnetism (magnetization opposite to the field)

### Classification of Magnetism

Magnetism | Susceptibility $\chi$ | Characteristics | Typical Examples  
---|---|---|---  
**Diamagnetism** | $\chi < 0$ (small) | Repelled by external fields, no temperature dependence | Cu, Au, Si  
**Paramagnetism** | $\chi > 0$ (small) | Weakly magnetized along the field, Curie law ($\chi \propto 1/T$) | Al, Pt, O₂  
**Ferromagnetism** | $\chi \gg 1$ | Spontaneous magnetization, ordered below the Curie temperature $T_C$ | Fe, Co, Ni  
**Antiferromagnetism** | $\chi > 0$ (small) | Neighboring spins antiparallel, ordered below the Néel temperature $T_N$ | MnO, Cr  
**Ferrimagnetism** | $\chi > 0$ (large) | Antiparallel but unequal magnitudes → net magnetization | Fe₃O₄ (magnetite)  
  
### Mean-Field Theory of Ferromagnetism (Weiss Theory)

In ferromagnets, an "exchange interaction" acts to align neighboring spins parallel to each other. Weiss introduced an "effective field" $H_{\text{eff}}$ felt by each spin:

$$ H_{\text{eff}} = H + \lambda M $$ 

$\lambda$ is the Weiss constant (molecular field constant). Solving the self-consistent equation yields the Curie temperature $T_C$:

$$ T_C = \frac{C\lambda}{N_A k_B} $$ 

Spontaneous magnetization arises for $T < T_C$.

## Predicting Magnetism with DFT Calculations

### Spin-Polarized DFT Calculations

In DFT calculations of magnetic materials, spin-up (↑) and spin-down (↓) electrons are treated separately (spin-polarized calculation).

The electron density is decomposed into spin components:

$$ n(\mathbf{r}) = n_\uparrow(\mathbf{r}) + n_\downarrow(\mathbf{r}) $$ 

**Spin density** (magnetization density):

$$ m(\mathbf{r}) = n_\uparrow(\mathbf{r}) - n_\downarrow(\mathbf{r}) $$ 

**Magnetic moment** :

$$ \mu = \mu_B \int m(\mathbf{r}) d\mathbf{r} $$ 

### Setting Up Spin-Polarized Calculations in VASP
    
    
    # INCAR file to set up a spin-polarized calculation in VASP
    
    def create_magnetic_incar(system_name='Fe', initial_magmom=2.0):
        """
        Generate a VASP INCAR file for magnetic materials
    
        Parameters:
        -----------
        system_name : str
            System name
        initial_magmom : float
            Initial magnetic moment [μB/atom]
        """
    
        incar_content = f"""SYSTEM = {system_name} magnetic calculation
    
    # Electronic structure
    ENCUT = 400
    PREC = Accurate
    LREAL = Auto
    
    # Exchange-correlation
    GGA = PE
    
    # SCF convergence
    EDIFF = 1E-6
    NELM = 100
    
    # Smearing (for metals)
    ISMEAR = 1          # Methfessel-Paxton
    SIGMA = 0.2
    
    # Spin-polarized calculation
    ISPIN = 2           # Enable spin polarization
    MAGMOM = {initial_magmom}  # Initial magnetic moment [μB]
    
    # Output of magnetic moments
    LORBIT = 11         # Atom- and orbital-projected magnetic moments
    
    # Parallelization
    NCORE = 4
    """
        return incar_content
    
    # Calculation setup for ferromagnetic Fe (BCC)
    incar_fe = create_magnetic_incar('Fe BCC', initial_magmom=2.2)
    print("=== INCAR for Ferromagnetic Fe Calculation ===")
    print(incar_fe)
    
    # Calculation setup for antiferromagnetic MnO (rocksalt)
    # Set alternating initial spins on the Mn atoms
    incar_mno = """SYSTEM = MnO antiferromagnetic
    
    ENCUT = 450
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-6
    ISMEAR = 0
    SIGMA = 0.05
    
    # Spin-polarized calculation
    ISPIN = 2
    MAGMOM = 4.0 -4.0 4.0 -4.0 0 0 0 0  # 4 Mn (alternating) + 4 O (nonmagnetic)
    
    LORBIT = 11
    NCORE = 4
    """
    
    print("\n=== INCAR for Antiferromagnetic MnO Calculation ===")
    print(incar_mno)
    

### Visualizing the Spin Density
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Generate dummy spin density data (in practice, read from VASP output)
    def generate_spin_density_data():
        """
        Generate a mock spin density around an Fe atom
        """
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
    
        # Approximate the spin density with a Gaussian distribution
        spin_density = 2.2 * np.exp(-(X**2 + Y**2) / 2)
    
        return X, Y, spin_density
    
    X, Y, spin_density = generate_spin_density_data()
    
    # 2D plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contour plot
    contour = ax1.contourf(X, Y, spin_density, levels=20, cmap='RdBu_r')
    ax1.contour(X, Y, spin_density, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    fig.colorbar(contour, ax=ax1, label='Spin density [μB/Å³]')
    ax1.set_xlabel('x [Å]', fontsize=12)
    ax1.set_ylabel('y [Å]', fontsize=12)
    ax1.set_title('Spin Density around an Fe Atom (2D)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # 3D surface
    from matplotlib import cm
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, spin_density, cmap=cm.coolwarm, alpha=0.8)
    ax2.set_xlabel('x [Å]', fontsize=10)
    ax2.set_ylabel('y [Å]', fontsize=10)
    ax2.set_zlabel('Spin density [μB/Å³]', fontsize=10)
    ax2.set_title('Spin Density around an Fe Atom (3D)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spin_density.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate the magnetic moment (numerical integration)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    total_moment = np.sum(spin_density) * dx * dy
    
    print(f"\n=== Magnetic Moment Calculation Results ===")
    print(f"Integrated magnetic moment: {total_moment:.2f} μB")
    print(f"(Actual Fe: about 2.2 μB)")
    

## Spin-Orbit Coupling (SOC)

### Origin of SOC

This is the effect in which the electron spin $\mathbf{S}$ interacts with the orbital angular momentum $\mathbf{L}$. It arises from relativistic effects:

$$ H_{\text{SOC}} = \lambda \mathbf{L} \cdot \mathbf{S} $$ 

$\lambda$ is the spin-orbit coupling constant, which increases rapidly with atomic number $Z$ ($\lambda \propto Z^4$).

### Physical Effects of SOC

  * **Magnetic anisotropy** : the energy depends on the direction of magnetization
  * **Magnetic circular dichroism** (MCD): difference in absorption of circularly polarized light
  * **Rashba effect** : spin splitting in systems with broken inversion symmetry
  * **Topological insulators** : SOC-induced band inversion

### Setting Up SOC Calculations in VASP
    
    
    # VASP calculation setup including spin-orbit coupling
    
    def create_soc_incar(system_name='Pt', include_soc=True):
        """
        Generate an INCAR file for SOC calculations
    
        Parameters:
        -----------
        system_name : str
            System name
        include_soc : bool
            Whether to enable SOC
        """
    
        incar_content = f"""SYSTEM = {system_name} with SOC
    
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-7        # High accuracy required for SOC calculations
    ISMEAR = 1
    SIGMA = 0.2
    
    # Spin polarization + SOC
    ISPIN = 2
    """
    
        if include_soc:
            incar_content += """LSORBIT = .TRUE.    # Enable spin-orbit coupling
    LNONCOLLINEAR = .TRUE.  # Noncollinear magnetism (spin directions are free)
    GGA_COMPAT = .FALSE.    # Recommended for SOC calculations
    """
    
        incar_content += """
    LORBIT = 11
    NCORE = 4
    """
        return incar_content
    
    # Pt (heavy element, SOC important)
    incar_pt_soc = create_soc_incar('Pt bulk', include_soc=True)
    print("=== INCAR for Pt + SOC Calculation ===")
    print(incar_pt_soc)
    
    # For comparison without SOC
    incar_pt_no_soc = create_soc_incar('Pt bulk', include_soc=False)
    print("\n=== INCAR for Pt (without SOC) Calculation ===")
    print(incar_pt_no_soc)
    

### SOC-Induced Band Splitting
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Band structure simulation with and without SOC
    def simulate_soc_band_splitting():
        """
        Visualize SOC-induced band splitting with a mock model
        """
        k = np.linspace(-np.pi, np.pi, 200)
    
        # Without SOC (degenerate)
        E_no_soc = np.cos(k) + 0.5 * np.cos(2*k)
    
        # With SOC (split)
        lambda_soc = 0.3  # SOC strength
        E_soc_up = E_no_soc + lambda_soc * np.abs(np.sin(k))
        E_soc_down = E_no_soc - lambda_soc * np.abs(np.sin(k))
    
        return k, E_no_soc, E_soc_up, E_soc_down
    
    k, E_no_soc, E_soc_up, E_soc_down = simulate_soc_band_splitting()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Without SOC
    ax1.plot(k/np.pi, E_no_soc, linewidth=2, color='blue', label='Degenerate band')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('k [π/a]', fontsize=12)
    ax1.set_ylabel('Energy [eV]', fontsize=12)
    ax1.set_title('Without SOC', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # With SOC
    ax2.plot(k/np.pi, E_soc_up, linewidth=2, color='red', label='Spin up')
    ax2.plot(k/np.pi, E_soc_down, linewidth=2, color='blue', label='Spin down')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('k [π/a]', fontsize=12)
    ax2.set_ylabel('Energy [eV]', fontsize=12)
    ax2.set_title('With SOC', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soc_band_splitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Splitting energy at k=π/2
    idx = len(k) // 4
    splitting = E_soc_up[idx] - E_soc_down[idx]
    print(f"\n=== SOC-Induced Band Splitting ===")
    print(f"Splitting at k=π/2: {splitting:.3f} eV")
    

## Fundamentals of Superconductivity

### The Superconducting Phenomenon

Below a critical temperature $T_c$, the electrical resistance drops to zero. It was discovered in Hg by Kamerlingh Onnes in 1911.

### BCS Theory (1957)

The microscopic theory by Bardeen, Cooper, and Schrieffer. Electrons acquire an attractive interaction mediated by phonons (lattice vibrations) and form "Cooper pairs".

#### Mechanism of Cooper Pair Formation

  1. Electron A distorts the lattice (attracting positive charge)
  2. The distorted lattice attracts electron B
  3. An effective attraction acts between electrons A and B (phonon-mediated)
  4. An electron pair with opposite spins and opposite momenta forms ($\mathbf{k}\uparrow, -\mathbf{k}\downarrow$)

**Superconducting gap** :

$$ \Delta(T) = \Delta_0 \tanh\left(1.74\sqrt{\frac{T_c - T}{T}}\right) $$ 

Gap at $T=0$ K:

$$ \Delta_0 \approx 1.76 k_B T_c $$ 

### Representative Superconductors

Material | $T_c$ [K] | Type | Notes  
---|---|---|---  
Hg (mercury) | 4.15 | Type I | First superconductor ever discovered  
Nb₃Sn | 18.3 | Type II | A15 structure, used in magnets  
YBa₂Cu₃O₇ (YBCO) | 92 | High-Tc | Cuprate, above liquid nitrogen temperature  
MgB₂ | 39 | Type II | Simple structure, explainable by BCS theory  
H₃S (high pressure) | 203 | High-Tc | 150 GPa, record-high $T_c$  
  
### Temperature Dependence of the Superconducting Gap
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def superconducting_gap(T, Tc):
        """
        Temperature dependence of the superconducting gap from BCS theory
    
        Parameters:
        -----------
        T : array
            Temperature [K]
        Tc : float
            Critical temperature [K]
    
        Returns:
        --------
        Delta : array
            Superconducting gap [meV]
        """
        k_B = 8.617e-5  # Boltzmann constant [eV/K]
    
        # Approximate formula from BCS theory
        Delta_0 = 1.76 * k_B * Tc * 1000  # [meV]
    
        Delta = np.zeros_like(T)
        mask = T < Tc
        Delta[mask] = Delta_0 * np.tanh(1.74 * np.sqrt((Tc - T[mask]) / T[mask]))
    
        return Delta
    
    # T_c of various superconductors
    materials = {
        'Al': 1.2,
        'Nb': 9.2,
        'MgB₂': 39,
        'YBCO': 92
    }
    
    T = np.linspace(0.1, 100, 500)
    
    plt.figure(figsize=(10, 6))
    
    for name, Tc in materials.items():
        Delta = superconducting_gap(T, Tc)
        plt.plot(T, Delta, linewidth=2, label=f'{name} ($T_c$={Tc}K)')
    
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Superconducting gap Δ(T) [meV]', fontsize=12)
    plt.title('Temperature Dependence of the Superconducting Gap', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('superconducting_gap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Verify Δ_0 / k_B T_c (1.76 in BCS theory)
    for name, Tc in materials.items():
        k_B = 8.617e-5
        Delta_0 = 1.76 * k_B * Tc * 1000
        ratio = Delta_0 / (k_B * Tc * 1000)
        print(f"{name}: Δ₀/(kB·Tc) = {ratio:.2f}")
    

## Summary

### What You Learned in This Chapter

#### Electrical Properties

  * Understanding electrical conduction with the Drude model: $\sigma = ne^2\tau/m^*$
  * The Hall effect allows measurement of carrier density and mobility
  * In semiconductors, mobility decreases with rising temperature ($\mu \propto T^{-3/2}$)

#### Magnetic Properties

  * Magnetism is classified into diamagnetism, paramagnetism, ferromagnetism, antiferromagnetism, and ferrimagnetism
  * Spin-polarized DFT calculations (ISPIN=2) can predict magnetic moments
  * Spin-orbit coupling (SOC) is important for heavy elements and is the origin of magnetic anisotropy

#### Superconductivity

  * BCS theory: electrons form Cooper pairs, leading to zero resistance
  * Superconducting gap: $\Delta_0 \approx 1.76 k_B T_c$
  * High-temperature superconductors (YBCO: 92K) operate above liquid nitrogen temperature

#### Preparing for the Next Chapter

  * In Chapter 5, we will study optical and thermal properties
  * We will calculate light absorption, band gaps, phonons, and thermal conduction

## Exercises

#### Exercise 1: Applying the Drude Model (Difficulty: ★☆☆)

**Problem** : Calculate the electrical conductivity and mobility from the following data.

  * Material: n-type Si semiconductor
  * Carrier density: $n = 1.0 \times 10^{22}$ m⁻³
  * Relaxation time: $\tau = 0.1$ ps = $1.0 \times 10^{-13}$ s
  * Effective mass: $m^* = 0.26 m_e$

**Hints** :

  * $\sigma = ne^2\tau/m^*$
  * $\mu = e\tau/m^*$
  * $e = 1.602 \times 10^{-19}$ C, $m_e = 9.109 \times 10^{-31}$ kg

**Answer** : $\sigma \approx 1.08 \times 10^4$ S/m, $\mu \approx 674$ cm²/Vs

#### Exercise 2: Identifying Carriers with the Hall Effect (Difficulty: ★★☆)

**Problem** : A 1 T magnetic field was applied to a semiconductor and a Hall measurement was performed, yielding a Hall coefficient $R_H = +5.0 \times 10^{-4}$ m³/C.

  1. Are the carriers electrons or holes?
  2. Calculate the carrier density
  3. Given an electrical conductivity of $\sigma = 100$ S/m, calculate the mobility

**Answers** :

  1. Since $R_H > 0$, the carriers are holes (p-type semiconductor)
  2. $n = 1/(R_H \cdot e) = 1.25 \times 10^{22}$ m⁻³
  3. $\mu = R_H \cdot \sigma = 0.05$ m²/Vs = 500 cm²/Vs

#### Exercise 3: Calculating the Magnetic Moment (Difficulty: ★★☆)

**Problem** : A spin-polarized DFT calculation of an Fe atom (BCC structure, a=2.87 Å) gave the following results:

  * Number of spin-up electrons: 8.1
  * Number of spin-down electrons: 5.9

Calculate the magnetic moment and compare it with the experimental value (2.2 μB).

**Hint** : $\mu = (N_\uparrow - N_\downarrow) \mu_B$

**Answer** : $\mu = (8.1 - 5.9) \mu_B = 2.2 \mu_B$ (in agreement with experiment)

#### Exercise 4: Calculating the Superconducting Gap (Difficulty: ★★☆)

**Problem** : The critical temperature of Nb (niobium) is $T_c = 9.2$ K.

  1. Calculate the superconducting gap $\Delta_0$ at $T = 0$ K
  2. Calculate the superconducting gap $\Delta(5K)$ at $T = 5$ K

**Hints** :

  * $\Delta_0 = 1.76 k_B T_c$
  * $\Delta(T) = \Delta_0 \tanh(1.74\sqrt{(T_c - T)/T})$
  * $k_B = 8.617 \times 10^{-5}$ eV/K

**Answers** :

  1. $\Delta_0 = 1.76 \times 8.617 \times 10^{-5} \times 9.2 = 1.40$ meV
  2. $\Delta(5K) = 1.40 \times \tanh(1.74\sqrt{(9.2-5)/5}) = 1.40 \times 0.87 = 1.22$ meV

#### Exercise 5: Preparing a VASP Spin-Polarized Calculation (Difficulty: ★★★)

**Problem** : Prepare a spin-polarized DFT calculation of antiferromagnetic MnO (rocksalt structure, a=4.43 Å).

  1. Create the MnO structure with ASE (2×2×2 supercell)
  2. Set alternating initial magnetic moments on the Mn atoms (±5.0 μB)
  3. Create the INCAR file (ISPIN=2, MAGMOM settings)
  4. Create the KPOINTS file (6×6×6 mesh)

**Evaluation criteria** :

  * Is MAGMOM set only for the Mn atoms?
  * Are the Mn spins arranged in an alternating pattern?
  * Are the initial magnetic moments of the O atoms set to 0?

#### Exercise 6: Temperature Dependence of Magnetic Susceptibility (Difficulty: ★★★)

**Problem** : The magnetic susceptibility of a paramagnetic material follows the Curie law:

$$ \chi = \frac{C}{T} $$ 

where $C$ is the Curie constant. Determine the Curie constant from the following data:

Temperature [K] | Susceptibility $\chi$ [10⁻⁶]  
---|---  
100| 8.5  
200| 4.2  
300| 2.8  
400| 2.1  
  
**Hint** : The slope of the line in a plot of $\chi$ versus $1/T$ gives the Curie constant

**Example answer** : $C \approx 8.5 \times 10^{-4}$ K (linear fit)

## References

  1. Ashcroft, N. W., & Mermin, N. D. (1976). "Solid State Physics". Harcourt College Publishers.
  2. Kittel, C. (2004). "Introduction to Solid State Physics" (8th ed.). Wiley.
  3. Blundell, S. (2001). "Magnetism in Condensed Matter". Oxford University Press.
  4. Tinkham, M. (2004). "Introduction to Superconductivity" (2nd ed.). Dover Publications.
  5. Bardeen, J., Cooper, L. N., & Schrieffer, J. R. (1957). "Theory of Superconductivity". Physical Review, 108, 1175.
  6. VASP manual: Magnetism and SOC - https://www.vasp.at/wiki/index.php/Magnetism

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
