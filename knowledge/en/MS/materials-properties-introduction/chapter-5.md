---
title: "Chapter 5: Optical and Thermal Properties"
chapter_title: "Chapter 5: Optical and Thermal Properties"
subtitle: Light Absorption, Bandgap, Phonons and Thermal Conduction
difficulty: Intermediate to Advanced
---

This chapter covers Optical and Thermal Properties. You will learn essential concepts and techniques.

## What You Will Learn in This Chapter

### Learning Objectives (3 Levels)

#### Basic Level

  * Explain the mechanism of light absorption and its relationship with bandgap
  * Understand the basic concept of phonons (lattice vibrations)
  * Explain the classical theory of thermal conduction (Debye model)

#### Intermediate Level

  * Calculate optical transitions and absorption spectra from band structures
  * Determine phonon DOS (density of states) from DFT calculations
  * Evaluate thermal conductivity and thermoelectric properties

#### Advanced Level

  * Predict dielectric functions and refractive indices through first-principles calculations
  * Analyze phonon dispersion relations and evaluate material stability
  * Understand strategies to maximize the thermoelectric figure of merit ZT

## Fundamentals of Optical Properties

### Light-Matter Interaction

When light is incident on a material, the following processes occur:

  * **Reflection** : Light bounces off the surface
  * **Absorption** : Electrons absorb photons and become excited
  * **Transmission** : Light passes through the material

Optical properties are described by the **complex dielectric function** $\varepsilon(\omega)$:

$$ \varepsilon(\omega) = \varepsilon_1(\omega) + i\varepsilon_2(\omega) $$ 

  * $\varepsilon_1$: Real part (related to refractive index)
  * $\varepsilon_2$: Imaginary part (related to absorption)

### Complex Refractive Index

The complex refractive index $\tilde{n}$ is related to the dielectric function as follows:

$$ \tilde{n}(\omega) = n(\omega) + i\kappa(\omega) $$ $$ \tilde{n}^2 = \varepsilon(\omega) $$ 

  * $n$: Refractive index
  * $\kappa$: Extinction coefficient (absorption coefficient)

Relationship between real and imaginary parts:

$$ n^2 - \kappa^2 = \varepsilon_1 $$ $$ 2n\kappa = \varepsilon_2 $$ 

### Optical Absorption Coefficient

The optical absorption coefficient $\alpha(\omega)$ is:

$$ \alpha(\omega) = \frac{2\omega\kappa(\omega)}{c} $$ 

Lambert-Beer's law: Attenuation of light intensity in the material

$$ I(x) = I_0 e^{-\alpha x} $$ 

### Bandgap and Light Absorption

In semiconductors and insulators, when photon energy $\hbar\omega$ exceeds the bandgap $E_g$, light absorption begins through interband transitions:

$$ \hbar\omega \geq E_g \quad \Rightarrow \quad \text{Light absorption} $$ 

#### Direct and Indirect Transitions

  * **Direct transition** (GaAs, CdSe, etc.): Valence band maximum and conduction band minimum at the same k-point → Strong light absorption
  * **Indirect transition** (Si, Ge, etc.): Different k-points → Phonon assistance required, weak light absorption

Material | Bandgap [eV] | Transition Type | Absorption Wavelength [nm]  
---|---|---|---  
Si| 1.12| Indirect| ~1107  
GaAs| 1.42| Direct| ~873  
GaN| 3.4| Direct| ~365 (UV)  
CdTe| 1.5| Direct| ~827  
  
### Simulation of Optical Absorption Spectra
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    h = 4.135667e-15  # Planck constant [eV·s]
    c = 3e8  # Speed of light [m/s]
    
    def direct_transition_absorption(energy, E_g, A=1e5):
        """
        Optical absorption coefficient for direct transition
    
        α(E) = A * sqrt(E - E_g)  for E > E_g
    
        Parameters:
        -----------
        energy : array
            Photon energy [eV]
        E_g : float
            Bandgap [eV]
        A : float
            Proportionality constant [cm^-1 eV^-1/2]
        """
        alpha = np.zeros_like(energy)
        mask = energy > E_g
        alpha[mask] = A * np.sqrt(energy[mask] - E_g)
        return alpha
    
    def indirect_transition_absorption(energy, E_g, A=1e4):
        """
        Optical absorption coefficient for indirect transition
    
        α(E) = A * (E - E_g)^2  for E > E_g
        """
        alpha = np.zeros_like(energy)
        mask = energy > E_g
        alpha[mask] = A * (energy[mask] - E_g)**2
        return alpha
    
    # Energy range
    energy = np.linspace(0, 4, 500)  # [eV]
    
    # GaAs (direct transition)
    E_g_GaAs = 1.42
    alpha_GaAs = direct_transition_absorption(energy, E_g_GaAs)
    
    # Si (indirect transition)
    E_g_Si = 1.12
    alpha_Si = indirect_transition_absorption(energy, E_g_Si)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absorption coefficient vs Energy
    ax1.plot(energy, alpha_GaAs, linewidth=2, color='#f093fb', label='GaAs (direct)')
    ax1.plot(energy, alpha_Si, linewidth=2, color='#3498db', label='Si (indirect)')
    ax1.axvline(x=E_g_GaAs, color='#f093fb', linestyle='--', alpha=0.5)
    ax1.axvline(x=E_g_Si, color='#3498db', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Photon Energy [eV]', fontsize=12)
    ax1.set_ylabel('Absorption Coefficient α [cm⁻¹]', fontsize=12)
    ax1.set_title('Light Absorption by Interband Transitions', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e2, 1e6)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wavelength conversion
    wavelength = 1240 / energy  # λ[nm] = 1240 / E[eV]
    ax2.plot(wavelength, alpha_GaAs, linewidth=2, color='#f093fb', label='GaAs')
    ax2.plot(wavelength, alpha_Si, linewidth=2, color='#3498db', label='Si')
    ax2.set_xlabel('Wavelength [nm]', fontsize=12)
    ax2.set_ylabel('Absorption Coefficient α [cm⁻¹]', fontsize=12)
    ax2.set_title('Wavelength Dependence', fontsize=14, fontweight='bold')
    ax2.set_xlim(300, 1200)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optical_absorption.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Bandgap and Absorption Wavelength ===")
    print(f"GaAs: E_g = {E_g_GaAs} eV, λ_edge = {1240/E_g_GaAs:.0f} nm")
    print(f"Si:   E_g = {E_g_Si} eV, λ_edge = {1240/E_g_Si:.0f} nm")
    

## Phonons (Lattice Vibrations)

### What are Phonons?

Atoms in a crystal vibrate around their equilibrium positions. These collective vibrations, when quantized, are called **phonons**.

A crystal with $N$ atoms has $3N$ vibrational modes:

  * **Acoustic phonons** (3 modes): Propagate as sound waves at long wavelengths (LA, TA1, TA2)
  * **Optical phonons** ($3N-3$ modes): Atoms vibrate in anti-phase, infrared active

### Phonon Dispersion Relations

The relationship between phonon angular frequency $\omega$ and wave vector $\mathbf{q}$ is called the **dispersion relation** $\omega(\mathbf{q})$.

Dispersion relation for a 1D monatomic chain:

$$ \omega(q) = \sqrt{\frac{4K}{M}} \left|\sin\left(\frac{qa}{2}\right)\right| $$ 

  * $K$: Spring constant (interatomic force constant)
  * $M$: Atomic mass
  * $a$: Lattice constant

### Phonon Density of States (DOS)

The phonon DOS $g(\omega)$ represents the number of phonon states at angular frequency $\omega$:

$$ g(\omega) = \frac{1}{(2\pi)^3} \int \delta(\omega - \omega_s(\mathbf{q})) d\mathbf{q} $$ 

where $s$ is the band index (acoustic/optical mode).

### Simulation of Phonon Dispersion Relations
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def phonon_dispersion_1D_monoatomic(q, K=10, M=1, a=1):
        """
        Phonon dispersion for 1D monatomic chain
    
        Parameters:
        -----------
        q : array
            Wave vector [rad/m]
        K : float
            Spring constant [N/m]
        M : float
            Atomic mass [kg]
        a : float
            Lattice constant [m]
        """
        omega = np.sqrt(4 * K / M) * np.abs(np.sin(q * a / 2))
        return omega
    
    def phonon_dispersion_1D_diatomic(q, K=10, M1=1, M2=2, a=1):
        """
        Phonon dispersion for 1D diatomic chain (optical and acoustic branches)
    
        Returns:
        --------
        omega_acoustic : array
            Acoustic phonons
        omega_optical : array
            Optical phonons
        """
        # Simplified dispersion relations
        omega_max = np.sqrt(2 * K * (1/M1 + 1/M2))
        omega_min = 0
    
        # Acoustic branch
        omega_acoustic = omega_max / 2 * np.abs(np.sin(q * a / 2))
    
        # Optical branch
        omega_optical = omega_max * np.sqrt(1 - 0.5 * (np.sin(q * a / 2))**2)
    
        return omega_acoustic, omega_optical
    
    # Wave vector range (first Brillouin zone)
    q = np.linspace(-np.pi, np.pi, 200)
    
    # Monatomic chain
    omega_mono = phonon_dispersion_1D_monoatomic(q)
    
    # Diatomic chain
    omega_ac, omega_op = phonon_dispersion_1D_diatomic(q)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Monatomic chain
    ax1.plot(q/np.pi, omega_mono, linewidth=2, color='#f093fb')
    ax1.set_xlabel('Wave Vector q [π/a]', fontsize=12)
    ax1.set_ylabel('Angular Frequency ω [rad/s]', fontsize=12)
    ax1.set_title('Phonon Dispersion for 1D Monatomic Chain', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    
    # Diatomic chain
    ax2.plot(q/np.pi, omega_ac, linewidth=2, color='#3498db', label='Acoustic Branch')
    ax2.plot(q/np.pi, omega_op, linewidth=2, color='#f5576c', label='Optical Branch')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Wave Vector q [π/a]', fontsize=12)
    ax2.set_ylabel('Angular Frequency ω [rad/s]', fontsize=12)
    ax2.set_title('Phonon Dispersion for 1D Diatomic Chain', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('phonon_dispersion.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Phonon Calculations Using DFT

### DFPT (Density Functional Perturbation Theory)

To calculate phonons, we need to determine the energy change (force constants) when atoms are displaced. DFPT (Density Functional Perturbation Theory) enables efficient calculation.

**Force constant matrix** :

$$ \Phi_{\alpha\beta}(ll') = \frac{\partial^2 E}{\partial u_\alpha(l) \partial u_\beta(l')} $$ 

  * $u_\alpha(l)$: Displacement of atom $l$ in the $\alpha$ direction

**Phonon frequencies** :

$$ \omega^2(\mathbf{q}) = \text{eigenvalues of } D(\mathbf{q}) $$ 

The dynamical matrix $D(\mathbf{q})$ is the Fourier transform of the force constant matrix.

### Setting Up Phonon Calculations in VASP
    
    
    # INCAR file for phonon calculations in VASP
    
    def create_phonon_incar(system_name='Si'):
        """
        Generate INCAR file for phonon calculations
    
        VASP has limited DFPT support. External tools like Phonopy are recommended.
        """
        incar_content = f"""SYSTEM = {system_name} phonon calculation
    
    # Electronic structure
    ENCUT = 500         # High precision required
    PREC = Accurate
    EDIFF = 1E-8        # Very high precision convergence
    
    # SCF
    ISMEAR = 0
    SIGMA = 0.01        # Small smearing
    
    # Force calculation
    IBRION = -1         # Single-point calculation (Phonopy manages displacements)
    NSW = 0
    
    # High precision forces
    ADDGRID = .TRUE.    # High precision grid
    LREAL = .FALSE.     # Turn off real-space projection (prioritize accuracy)
    
    NCORE = 4
    """
        return incar_content
    
    incar_phonon = create_phonon_incar('Si')
    print("=== Phonon Calculation INCAR (with Phonopy) ===")
    print(incar_phonon)
    
    # Phonopy workflow
    print("\n=== Phonopy Workflow ===")
    workflow = """
    1. Create supercell:
       phonopy -d --dim="2 2 2"
    
    2. Run VASP for displaced structures:
       Run VASP for each POSCAR-XXX
    
    3. Calculate force constants:
       phonopy -f vasprun-001.xml vasprun-002.xml ...
    
    4. Calculate phonon bands and DOS:
       phonopy --dim="2 2 2" -p band.conf
       phonopy --dim="2 2 2" -p mesh.conf
    
    5. Calculate thermal properties (heat capacity, entropy):
       phonopy --dim="2 2 2" -t -p mesh.conf
    """
    print(workflow)
    

### Visualization of Phonon DOS
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate dummy phonon DOS data (in practice, read from Phonopy output)
    def generate_phonon_dos():
        """
        Simulated phonon DOS for Si
        """
        freq = np.linspace(0, 600, 300)  # [cm^-1]
    
        # Acoustic phonons (low frequency)
        dos_acoustic = 2 * np.exp(-(freq - 150)**2 / (2 * 50**2))
    
        # Optical phonons (high frequency)
        dos_optical = 1.5 * np.exp(-(freq - 520)**2 / (2 * 30**2))
    
        dos_total = dos_acoustic + dos_optical
    
        return freq, dos_total, dos_acoustic, dos_optical
    
    freq, dos_total, dos_acoustic, dos_optical = generate_phonon_dos()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phonon DOS
    ax1.plot(freq, dos_total, linewidth=2, color='black', label='Total')
    ax1.fill_between(freq, 0, dos_acoustic, alpha=0.5, color='#3498db', label='Acoustic')
    ax1.fill_between(freq, dos_acoustic, dos_total, alpha=0.5, color='#f5576c', label='Optical')
    ax1.set_xlabel('Frequency [cm⁻¹]', fontsize=12)
    ax1.set_ylabel('Phonon DOS [states/cm⁻¹]', fontsize=12)
    ax1.set_title('Si Phonon Density of States', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Frequency → Energy conversion
    energy_meV = freq * 0.124  # 1 cm^-1 ≈ 0.124 meV
    ax2.plot(energy_meV, dos_total, linewidth=2, color='black')
    ax2.fill_between(energy_meV, 0, dos_acoustic, alpha=0.5, color='#3498db')
    ax2.fill_between(energy_meV, dos_acoustic, dos_total, alpha=0.5, color='#f5576c')
    ax2.set_xlabel('Energy [meV]', fontsize=12)
    ax2.set_ylabel('Phonon DOS', fontsize=12)
    ax2.set_title('Energy Conversion', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phonon_dos.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Thermal Properties

### Heat Capacity

From the phonon DOS, we can calculate the heat capacity at constant volume $C_V$ (Einstein/Debye models):

$$ C_V = k_B \int g(\omega) \left(\frac{\hbar\omega}{k_B T}\right)^2 \frac{e^{\hbar\omega/k_B T}}{(e^{\hbar\omega/k_B T} - 1)^2} d\omega $$ 

**Dulong-Petit law** (high temperature limit):

$$ C_V \to 3Nk_B $$ 

### Thermal Conduction

Thermal conductivity $\kappa$ is determined by phonon heat transport:

$$ \kappa = \frac{1}{3} C_V v_s^2 \tau $$ 

  * $C_V$: Heat capacity
  * $v_s$: Sound velocity (phonon group velocity)
  * $\tau$: Phonon relaxation time (scattering time)

#### Representative Thermal Conductivity Values (Room Temperature)

  * Diamond: 2,200 W/(m·K) (highest class)
  * Cu (copper): 400 W/(m·K)
  * Si: 150 W/(m·K)
  * Stainless steel: 15 W/(m·K)
  * Air: 0.026 W/(m·K)

### Simulation of Thermal Conductivity
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_conductivity_kinetic(C_V, v_s, tau):
        """
        Thermal conductivity from kinetic theory
    
        κ = (1/3) C_V v_s^2 τ
    
        Parameters:
        -----------
        C_V : float
            Volumetric heat capacity [J/(m^3·K)]
        v_s : float
            Sound velocity [m/s]
        tau : float
            Relaxation time [s]
        """
        kappa = (1/3) * C_V * v_s**2 * tau
        return kappa
    
    # Parameters for various materials
    materials = {
        'Si': {'C_V': 1.66e6, 'v_s': 8433, 'tau': 4e-11},
        'Ge': {'C_V': 1.70e6, 'v_s': 5400, 'tau': 3e-11},
        'GaAs': {'C_V': 1.50e6, 'v_s': 5150, 'tau': 2e-11},
    }
    
    print("=== Thermal Conductivity Calculation ===")
    for name, params in materials.items():
        kappa = thermal_conductivity_kinetic(**params)
        print(f"{name}: κ = {kappa:.1f} W/(m·K)")
    
    # Temperature dependence (phonon scattering)
    temperatures = np.linspace(100, 600, 100)  # [K]
    
    # High temperature: τ ∝ 1/T (Umklapp scattering)
    tau_ref = 4e-11  # 300K
    T_ref = 300
    tau_T = tau_ref * (temperatures / T_ref)**(-1)
    
    # Heat capacity (Debye model approximation)
    C_V_300 = 1.66e6
    C_V_T = C_V_300 * (temperatures / T_ref)**3 / (np.exp(temperatures / T_ref) - 1)
    
    # Thermal conductivity
    v_s = 8433
    kappa_T = (1/3) * C_V_T * v_s**2 * tau_T
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, kappa_T, linewidth=2, color='#f093fb')
    plt.axvline(x=300, color='red', linestyle='--', label='Room Temperature')
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Thermal Conductivity κ [W/(m·K)]', fontsize=12)
    plt.title('Temperature Dependence of Si Thermal Conductivity', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('thermal_conductivity_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Thermoelectric Materials

### Seebeck Effect

The phenomenon where a temperature difference $\Delta T$ generates a potential difference $\Delta V$:

$$ \Delta V = S \Delta T $$ 

$S$ is the **Seebeck coefficient** (thermopower).

### Thermoelectric Figure of Merit ZT

The performance of thermoelectric materials is evaluated by the **dimensionless figure of merit ZT** :

$$ ZT = \frac{S^2 \sigma T}{\kappa} $$ 

  * $S$: Seebeck coefficient [V/K]
  * $\sigma$: Electrical conductivity [S/m]
  * $\kappa$: Thermal conductivity [W/(m·K)]
  * $T$: Absolute temperature [K]

**Requirements for high-performance thermoelectric materials** :

  * High Seebeck coefficient ($S > 200$ μV/K)
  * High electrical conductivity (metal-like)
  * Low thermal conductivity (glass-like)
  * → "Phonon Glass Electron Crystal" (PGEC)

Material | ZT (Room Temp) | ZT (Optimal Temp) | Application  
---|---|---|---  
Bi₂Te₃| 0.8-1.0| 1.0 (300K)| Cooling devices  
PbTe| 0.5| 1.5 (700K)| Power generation  
Half-Heusler (TiNiSn)| 0.5| 1.0 (800K)| High-temp power  
SnSe (single crystal)| 0.5| 2.6 (923K)| Under research  
  
### Simulation of ZT Optimization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermoelectric_ZT(S, sigma, kappa, T):
        """
        Calculate thermoelectric figure of merit ZT
    
        Parameters:
        -----------
        S : float
            Seebeck coefficient [V/K]
        sigma : float
            Electrical conductivity [S/m]
        kappa : float
            Thermal conductivity [W/(m·K)]
        T : float
            Temperature [K]
        """
        ZT = (S**2 * sigma * T) / kappa
        return ZT
    
    # Variation of S, σ, κ with carrier density (simplified model)
    carrier_density = np.logspace(18, 22, 100)  # [m^-3]
    
    # Seebeck coefficient (decreases with increasing carrier density)
    S = 300e-6 / (carrier_density / 1e20)**0.5  # [V/K]
    
    # Electrical conductivity (proportional to carrier density)
    sigma = 1e3 * (carrier_density / 1e20)  # [S/m]
    
    # Thermal conductivity (electronic and phonon contributions)
    kappa_lattice = 1.5  # Lattice thermal conductivity [W/(m·K)]
    kappa_electronic = 2.44e-8 * sigma * 300  # Wiedemann-Franz law
    kappa = kappa_lattice + kappa_electronic
    
    # ZT calculation
    T = 300  # [K]
    ZT = thermoelectric_ZT(S, sigma, kappa, T)
    
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Seebeck coefficient
    ax1.plot(carrier_density, S * 1e6, linewidth=2, color='#f093fb')
    ax1.set_xscale('log')
    ax1.set_xlabel('Carrier Density [m⁻³]', fontsize=12)
    ax1.set_ylabel('Seebeck Coefficient [μV/K]', fontsize=12)
    ax1.set_title('Seebeck Coefficient', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Electrical conductivity
    ax2.plot(carrier_density, sigma, linewidth=2, color='#3498db')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Carrier Density [m⁻³]', fontsize=12)
    ax2.set_ylabel('Electrical Conductivity [S/m]', fontsize=12)
    ax2.set_title('Electrical Conductivity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Thermal conductivity
    ax3.plot(carrier_density, kappa, linewidth=2, color='#f5576c', label='Total')
    ax3.axhline(y=kappa_lattice, color='green', linestyle='--', label='Lattice')
    ax3.set_xscale('log')
    ax3.set_xlabel('Carrier Density [m⁻³]', fontsize=12)
    ax3.set_ylabel('Thermal Conductivity [W/(m·K)]', fontsize=12)
    ax3.set_title('Thermal Conductivity', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ZT
    ax4.plot(carrier_density, ZT, linewidth=2, color='black')
    optimal_idx = np.argmax(ZT)
    ax4.plot(carrier_density[optimal_idx], ZT[optimal_idx], 'ro', markersize=10, label=f'Max ZT = {ZT[optimal_idx]:.2f}')
    ax4.set_xscale('log')
    ax4.set_xlabel('Carrier Density [m⁻³]', fontsize=12)
    ax4.set_ylabel('ZT', fontsize=12)
    ax4.set_title('Thermoelectric Figure of Merit ZT', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermoelectric_ZT_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Optimal ZT ===")
    print(f"Optimal carrier density: {carrier_density[optimal_idx]:.2e} m^-3")
    print(f"Maximum ZT: {ZT[optimal_idx]:.2f}")
    print(f"Corresponding Seebeck coefficient: {S[optimal_idx]*1e6:.1f} μV/K")
    print(f"Corresponding electrical conductivity: {sigma[optimal_idx]:.1f} S/m")
    

## Prediction of Optical and Thermal Properties Using DFT Calculations

### Calculation of Dielectric Functions
    
    
    # INCAR file for calculating optical properties (dielectric function) in VASP
    
    def create_optics_incar(system_name='Si', nbands=None):
        """
        Generate INCAR file for optical calculations
    
        Parameters:
        -----------
        system_name : str
            System name
        nbands : int
            Number of bands (recommend approximately 2x default)
        """
        nbands_str = f"NBANDS = {nbands}" if nbands else "# NBANDS = Recommend 2x automatic setting"
    
        incar_content = f"""SYSTEM = {system_name} optical properties
    
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-6
    ISMEAR = 0
    SIGMA = 0.05
    
    # Optical calculations
    LOPTICS = .TRUE.    # Enable dielectric function calculation
    {nbands_str}
    
    # High-density k-points required
    # Recommend 16×16×16 or higher in KPOINTS
    
    NCORE = 4
    """
        return incar_content
    
    incar_optics = create_optics_incar('Si', nbands=48)
    print("=== Optical Calculation INCAR ===")
    print(incar_optics)
    
    # Calculated physical quantities
    print("\n=== Calculated Optical Properties ===")
    print("- Complex dielectric function ε(ω) = ε₁(ω) + iε₂(ω)")
    print("- Absorption coefficient α(ω)")
    print("- Refractive index n(ω)")
    print("- Reflectivity R(ω)")
    print("\nOutput file: vasprun.xml ( tag)")
    

### Calculation of Thermal Properties from Phonons
    
    
    # Calculate thermal properties using Phonopy
    
    phonopy_workflow = """
    === Thermal Properties Calculation Workflow with Phonopy ===
    
    1. Phonon calculation (as described previously):
       phonopy --dim="2 2 2" -c POSCAR -f vasprun-*.xml
    
    2. Create mesh configuration file (mesh.conf):
       DIM = 2 2 2
       MP = 16 16 16
       TPROP = .TRUE.
       TMIN = 0
       TMAX = 1000
       TSTEP = 10
    
    3. Calculate thermal properties:
       phonopy -t mesh.conf --dim="2 2 2"
    
    4. Output files:
       - thermal_properties.yaml
         * Heat capacity at constant volume C_V(T)
         * Entropy S(T)
         * Free energy F(T)
    
    5. Plot:
       phonopy -t -p mesh.conf
    """
    
    print(phonopy_workflow)
    
    # Temperature dependence of thermal properties (approximation using Debye model)
    def debye_heat_capacity(T, T_D, N=1):
        """
        Heat capacity using Debye model
    
        Parameters:
        -----------
        T : array
            Temperature [K]
        T_D : float
            Debye temperature [K]
        N : int
            Number of atoms
        """
        k_B = 1.380649e-23  # Boltzmann constant [J/K]
        x = T_D / T
    
        def debye_function(x):
            # Numerical integration required (simplified)
            return 3 * (x / np.sinh(x/2))**2
    
        C_V = 3 * N * k_B * debye_function(x)
        return C_V
    
    # Si Debye temperature: 645 K
    T = np.linspace(10, 600, 100)
    C_V = debye_heat_capacity(T, T_D=645, N=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T, C_V / 1.380649e-23, linewidth=2, color='#f093fb')
    plt.axhline(y=3, color='red', linestyle='--', label='Dulong-Petit Law (high-temp limit)')
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Heat Capacity C_V / k_B', fontsize=12)
    plt.title('Temperature Dependence of Si Heat Capacity (Debye Model)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_capacity_debye.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Summary

### What We Learned in This Chapter

#### Optical Properties

  * Light absorption is described by the complex dielectric function $\varepsilon(\omega)$
  * Strong absorption occurs at photon energies above the bandgap
  * Direct transitions (GaAs) show stronger light absorption than indirect transitions (Si)
  * DFT calculations can predict dielectric functions, refractive indices, and absorption coefficients

#### Phonons and Thermal Properties

  * Phonons are quantized excitations of lattice vibrations
  * Classified into acoustic and optical phonons
  * Heat capacity and entropy can be calculated from phonon DOS
  * Thermal conductivity is determined by phonon transport: $\kappa = (1/3) C_V v_s^2 \tau$

#### Thermoelectric Conversion

  * Evaluated by figure of merit $ZT = S^2\sigma T / \kappa$
  * High ZT requires high S, high σ, and low κ (PGEC concept)
  * Carrier density optimization is important

#### Preparation for Next Chapter

  * In Chapter 6, we will learn practical workflows integrating everything learned so far
  * Using Si, GaN, Fe, and BaTiO₃ as examples, we will execute the full process from structure optimization → DFT → property analysis

## Exercises

#### Exercise 1: Light Absorption Calculation (Easy)

**Problem** : Calculate the absorption edge wavelength for GaN (bandgap 3.4 eV).

**Hint** : $\lambda = hc / E = 1240 / E[eV]$ [nm]

**Answer** : $\lambda = 1240 / 3.4 = 365$ nm (ultraviolet region)

#### Exercise 2: Interpretation of Phonon Dispersion (Medium)

**Problem** : For the phonon dispersion of a 1D monatomic chain $\omega(q) = \omega_{\max} |\sin(qa/2)|$:

  1. What is the frequency at $q=0$ (long wavelength limit)? What is its physical meaning?
  2. What is the frequency at $q=\pi/a$ (Brillouin zone boundary)?
  3. Calculate the group velocity $v_g = d\omega/dq$ at $q=0$

**Answer** :

  1. $\omega(0) = 0$ (translational motion, propagates as sound wave)
  2. $\omega(\pi/a) = \omega_{\max}$ (neighboring atoms in anti-phase)
  3. $v_g = d\omega/dq|_{q=0} = (\omega_{\max}a/2) \cos(0) = \omega_{\max}a/2$ (sound velocity)

#### Exercise 3: Estimation of Thermal Conductivity (Medium)

**Problem** : Estimate the thermal conductivity of Si from the following data.

  * Volumetric heat capacity: $C_V = 1.66 \times 10^6$ J/(m³·K)
  * Sound velocity: $v_s = 8433$ m/s
  * Phonon relaxation time: $\tau = 4 \times 10^{-11}$ s

**Hint** : $\kappa = (1/3) C_V v_s^2 \tau$

**Answer** : $\kappa = (1/3) \times 1.66 \times 10^6 \times 8433^2 \times 4 \times 10^{-11} = 157$ W/(m·K)

#### Exercise 4: ZT Optimization (Hard)

**Problem** : To maximize the ZT of a thermoelectric material, explain how the following parameters should be changed.

  1. Double the Seebeck coefficient $S$
  2. Double the electrical conductivity $\sigma$
  3. Halve the thermal conductivity $\kappa$

**Recommended Answer** :

  * Doubling $S$ → ZT increases 4-fold ($ZT \propto S^2$)
  * Doubling $\sigma$ → ZT increases 2-fold ($ZT \propto \sigma$)
  * Halving $\kappa$ → ZT increases 2-fold ($ZT \propto 1/\kappa$)
  * Conclusion: Improving Seebeck coefficient is most effective. However, since S, σ, and κ are interdependent, independent control is difficult. Carrier density optimization is realistic.

#### Exercise 5: Preparation for Phonopy Calculation (Hard)

**Problem** : Prepare to execute phonon calculations for Si crystal using Phonopy.

  1. Write the command to create a 2×2×2 supercell
  2. Predict the number of displaced structures (without symmetry considerations)
  3. Set INCAR parameters for VASP calculations on each displaced structure

**Example Answer** :

  1. `phonopy -d --dim="2 2 2"`
  2. Si (diamond structure): 2 atoms/unit cell × 8 unit cells = 16 atoms. Each atom displaced in 3 directions → Number of displaced structures reduced by Phonopy using symmetry (typically 1-2)
  3. INCAR: IBRION=-1, EDIFF=1E-8, ADDGRID=.TRUE., LREAL=.FALSE. (high-precision force calculation)

#### Exercise 6: Practical Problem (Hard)

**Problem** : Design a DFT calculation to predict the optical properties of GaAs.

  1. Select an appropriate functional for bandgap calculation (with reasoning)
  2. Specify recommended k-point mesh and NBANDS when calculating with LOPTICS=.TRUE.
  3. Explain how to extract absorption spectra from calculation results

**Recommended Answer** :

  1. HSE06 functional (GGA-PBE underestimates bandgap)
  2. k-points: 16×16×16 (Gamma-centered), NBANDS: 2x default (include high-energy conduction band states)
  3. Read <dielectricfunction> from vasprun.xml, calculate absorption coefficient α(ω) from ε₂(ω): $\alpha = 2\omega\kappa/c$, where κ is derived from ε₂

## References

  1. Fox, M. (2010). "Optical Properties of Solids" (2nd ed.). Oxford University Press.
  2. Dove, M. T. (1993). "Introduction to Lattice Dynamics". Cambridge University Press.
  3. Togo, A., & Tanaka, I. (2015). "First principles phonon calculations in materials science". Scripta Materialia, 108, 1-5.
  4. Snyder, G. J., & Toberer, E. S. (2008). "Complex thermoelectric materials". Nature Materials, 7, 105-114.
  5. Ashcroft & Mermin (1976). "Solid State Physics". Chapters on phonons and thermal properties.
  6. Phonopy documentation: https://phonopy.github.io/phonopy/

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
