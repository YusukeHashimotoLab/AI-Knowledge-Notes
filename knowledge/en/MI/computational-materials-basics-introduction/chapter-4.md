---
title: "Chapter 4: Phonon Calculations and Thermodynamic Properties"
chapter_title: "Chapter 4: Phonon Calculations and Thermodynamic Properties"
subtitle: From Lattice Vibrations to Thermal Transport
reading_time: 20-25 min
difficulty: Intermediate to Advanced
code_examples: 6
exercises: 0
version: 1.0
created_at: "by:"
---

# Chapter 4: Phonon Calculations and Thermodynamic Properties

Grasp the concepts of lattice vibrations and density of states, and step into the world of property calculations including thermal transport.

**💡 Note:** Phonons are like an "atomic orchestra". Their frequency distribution reveals the characteristics of heat and vibrations.

## Learning Objectives

By reading this chapter, you will master: \- Understanding the theoretical foundations of lattice vibrations (phonons) \- Explaining the concepts of harmonic approximation and dynamical matrix \- Executing actual phonon calculations with Phonopy \- Interpreting phonon band structures and density of states \- Calculating thermodynamic properties (heat capacity, free energy, entropy)

* * *

## 4.1 Theory of Lattice Vibrations

### Harmonic Oscillator Model

Atoms in a crystal vibrate around their equilibrium positions $\mathbf{R}_I^0$. Let the displacement be $\mathbf{u}_I$:

$$ \mathbf{R}_I = \mathbf{R}_I^0 + \mathbf{u}_I $$

Taylor expansion of potential energy (harmonic approximation):

$$ U = U_0 + \sum_I \frac{\partial U}{\partial \mathbf{u}_I}\Bigg|_0 \mathbf{u}_I + \frac{1}{2}\sum_{I,J} \mathbf{u}_I \cdot \frac{\partial^2 U}{\partial \mathbf{u}_I \partial \mathbf{u}_J}\Bigg|_0 \cdot \mathbf{u}_J + O(\mathbf{u}^3) $$

At equilibrium, the first-order term vanishes:

$$ U \approx U_0 + \frac{1}{2}\sum_{I,J} \mathbf{u}_I \cdot \mathbf{\Phi}_{IJ} \cdot \mathbf{u}_J $$

$\mathbf{\Phi}_{IJ}$ is the **force constant matrix** :

$$ \Phi_{IJ}^{\alpha\beta} = \frac{\partial^2 U}{\partial u_I^\alpha \partial u_J^\beta}\Bigg|_0 $$

### Equations of Motion

Equation of motion for atom $I$:

$$ M_I \frac{d^2 u_I^\alpha}{dt^2} = -\sum_{J,\beta} \Phi_{IJ}^{\alpha\beta} u_J^\beta $$

Assume plane wave solutions:

$$ u_I^\alpha = \frac{1}{\sqrt{M_I}} e_I^\alpha(\mathbf{k}) e^{i(\mathbf{k}\cdot\mathbf{R}_I - \omega t)} $$

Introducing the **dynamical matrix** :

$$ D_{IJ}^{\alpha\beta}(\mathbf{k}) = \frac{1}{\sqrt{M_I M_J}} \Phi_{IJ}^{\alpha\beta} e^{i\mathbf{k}\cdot(\mathbf{R}_I - \mathbf{R}_J)} $$

### Eigenvalue Problem

$$ \sum_{J,\beta} D_{IJ}^{\alpha\beta}(\mathbf{k}) e_J^\beta(\mathbf{k}) = \omega^2(\mathbf{k}) e_I^\alpha(\mathbf{k}) $$

  * $\omega(\mathbf{k})$: Phonon angular frequency (phonon dispersion)
  * $\mathbf{e}(\mathbf{k})$: Phonon eigenvector (polarization vector)

**Number of phonon bands** : $3N_{\text{atom}}$ branches (for $N_{\text{atom}}$ atoms per unit cell)

* * *

## 4.2 Classification of Phonons

### Acoustic Modes

  * $\omega(\mathbf{k}) \to 0$ as $\mathbf{k} \to 0$
  * Zero frequency at Γ point ($\mathbf{k}=0$)
  * 3 branches (1 longitudinal LA, 2 transverse TA)
  * Physical meaning: Translational motion

### Optical Modes

  * $\omega(\mathbf{k}) \neq 0$ at $\mathbf{k} = 0$
  * Finite frequency at Γ point
  * $3(N_{\text{atom}} - 1)$ branches
  * Physical meaning: Relative motion within unit cell

### Longitudinal vs Transverse Waves

  * **Longitudinal (L)** : Vibration direction parallel to wave propagation
  * **Transverse (T)** : Vibration direction perpendicular to wave propagation

**Monatomic crystals (Si, Cu, etc.)** : \- Acoustic modes: 3 branches (1 LA + 2 TA) \- Optical modes: None

**Diatomic crystals (NaCl, GaAs, etc.)** : \- Acoustic modes: 3 branches \- Optical modes: 3 branches (1 LO + 2 TO)

* * *

## 4.3 Practical Application with Phonopy

### Installation
    
    
    pip install phonopy
    # Or with conda
    conda install -c conda-forge phonopy
    

### Example 1: Phonon Calculation for Si (using GPAW)

**Step 1: Ground State Calculation**
    
    
    from ase.build import bulk
    from gpaw import GPAW, PW
    
    # Create Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # DFT calculation
    calc = GPAW(mode=PW(400),
                xc='PBE',
                kpts=(8, 8, 8),
                txt='si_gs.txt')
    
    si.calc = calc
    si.get_potential_energy()
    calc.write('si_groundstate.gpw')
    

**Step 2: Create Supercell for Phonopy**
    
    
    # Create phonopy_disp.conf
    cat > phonopy_disp.conf <<EOF
    DIM = 2 2 2
    ATOM_NAME = Si
    EOF
    
    # Generate displacement structures with Phonopy
    phonopy -d --dim="2 2 2" --gpaw
    

This generates `supercell-XXX.py` files.

**Step 3: Force Calculations**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Step 3: Force Calculations
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # Run supercell-001.py (for each displaced structure)
    from gpaw import GPAW
    
    calc = GPAW('si_groundstate.gpw', txt=None)
    atoms = calc.get_atoms()
    forces = atoms.get_forces()
    
    # Write to FORCE_SETS file (format for Phonopy)
    import numpy as np
    np.savetxt('forces_001.dat', forces)
    

**Repeat for all displaced structures** (typically automated with scripts)

**Step 4: Phonon Calculation**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Step 4: Phonon Calculation
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from phonopy import Phonopy
    from phonopy.interface.gpaw import read_gpaw
    import matplotlib.pyplot as plt
    
    # Create Phonopy object
    unitcell, calc_forces = read_gpaw('si_groundstate.gpw')
    phonon = Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    
    # Set force constant matrix (from FORCE_SETS)
    phonon.set_displacement_dataset(dataset)
    phonon.produce_force_constants()
    
    # Calculate band structure
    path = [[[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625],
             [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L"]
    qpoints, connections = phonon.get_band_structure_plot_data(path)
    
    phonon.plot_band_structure(path, labels=labels)
    plt.ylabel('Frequency (THz)')
    plt.savefig('si_phonon_band.png', dpi=150)
    plt.show()
    
    # Density of states (DOS)
    phonon.set_mesh([20, 20, 20])
    phonon.set_total_DOS()
    dos_freq, dos_val = phonon.get_total_DOS()
    
    plt.figure(figsize=(8, 6))
    plt.plot(dos_freq, dos_val, linewidth=2)
    plt.xlabel('Frequency (THz)', fontsize=12)
    plt.ylabel('DOS (states/THz)', fontsize=12)
    plt.title('Si Phonon DOS', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('si_phonon_dos.png', dpi=150)
    plt.show()
    

* * *

### Example 2: Fully Automated Script
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from ase.build import bulk
    from gpaw import GPAW, PW
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    import numpy as np
    
    def calculate_phonon(symbol='Si', a=5.43, dim=(2,2,2)):
        """
        Fully automated phonon calculation
        """
        # 1. Ground state calculation
        print("Step 1: Ground state calculation...")
        atoms = bulk(symbol, 'diamond', a=a)
        calc = GPAW(mode=PW(400), xc='PBE', kpts=(8,8,8), txt=None)
        atoms.calc = calc
        atoms.get_potential_energy()
    
        # 2. Phonopy setup
        print("Step 2: Generate displacements...")
        cell = PhonopyAtoms(symbols=[symbol]*len(atoms),
                           cell=atoms.cell,
                           scaled_positions=atoms.get_scaled_positions())
    
        phonon = Phonopy(cell, np.diag(dim))
        phonon.generate_displacements(distance=0.01)
    
        # 3. Force calculations
        print(f"Step 3: Calculate forces for {len(phonon.supercells_with_displacements)} supercells...")
        set_of_forces = []
    
        for i, scell in enumerate(phonon.supercells_with_displacements):
            supercell = bulk(symbol, 'diamond', a=a).repeat(dim)
            supercell.positions = scell.positions
            supercell.calc = calc
            forces = supercell.get_forces()
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / len(forces)  # Drift correction
            set_of_forces.append(forces)
    
        # 4. Phonon calculation
        print("Step 4: Phonon calculation...")
        phonon.produce_force_constants(forces=set_of_forces)
    
        # Band structure
        path = [[[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625],
                 [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
        labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L"]
    
        phonon.auto_band_structure(plot=True, labels=labels, filename=f'{symbol}_band.png')
    
        # DOS
        phonon.auto_total_dos(plot=True, filename=f'{symbol}_dos.png')
    
        print("Done!")
        return phonon
    
    # Execute
    si_phonon = calculate_phonon('Si', a=5.43, dim=(2,2,2))
    

* * *

## 4.4 Calculation of Thermodynamic Properties

### Free Energy

**Helmholtz free energy** (NVT):

$$ F(T) = U_0 + k_B T \sum_{\mathbf{q},j} \ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}j}}{2k_B T}\right)\right] $$

  * $U_0$: Zero-point energy
  * $\omega_{\mathbf{q}j}$: Phonon frequency at wavevector $\mathbf{q}$, band $j$

### Internal Energy

$$ U(T) = U_0 + \sum_{\mathbf{q},j} \hbar\omega_{\mathbf{q}j} \left[\frac{1}{2} + n_B(\omega_{\mathbf{q}j}, T)\right] $$

$n_B(\omega, T)$ is the Bose-Einstein distribution function:

$$ n_B(\omega, T) = \frac{1}{e^{\hbar\omega/(k_B T)} - 1} $$

### Entropy

$$ S(T) = -\left(\frac{\partial F}{\partial T}\right)_V = k_B \sum_{\mathbf{q},j} \left[\frac{\hbar\omega_{\mathbf{q}j}}{k_B T} n_B(\omega_{\mathbf{q}j}, T) - \ln(1 - e^{-\hbar\omega_{\mathbf{q}j}/(k_B T)})\right] $$

### Heat Capacity (Constant Volume)

$$ C_V(T) = \left(\frac{\partial U}{\partial T}\right)_V = k_B \sum_{\mathbf{q},j} \left(\frac{\hbar\omega_{\mathbf{q}j}}{k_B T}\right)^2 \frac{e^{\hbar\omega_{\mathbf{q}j}/(k_B T)}}{(e^{\hbar\omega_{\mathbf{q}j}/(k_B T)} - 1)^2} $$

### Implementation in Phonopy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation in Phonopy
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Temperature range
    temperatures = np.arange(0, 1000, 10)
    
    # Calculate thermodynamic properties
    si_phonon.set_mesh([20, 20, 20])
    si_phonon.set_thermal_properties(t_step=10, t_max=1000, t_min=0)
    
    tp_dict = si_phonon.get_thermal_properties_dict()
    temps = tp_dict['temperatures']
    free_energy = tp_dict['free_energy']  # kJ/mol
    entropy = tp_dict['entropy']  # J/K/mol
    heat_capacity = tp_dict['heat_capacity']  # J/K/mol
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Free energy
    axes[0,0].plot(temps, free_energy, linewidth=2)
    axes[0,0].set_xlabel('Temperature (K)', fontsize=12)
    axes[0,0].set_ylabel('Free Energy (kJ/mol)', fontsize=12)
    axes[0,0].set_title('Helmholtz Free Energy', fontsize=14)
    axes[0,0].grid(alpha=0.3)
    
    # Entropy
    axes[0,1].plot(temps, entropy, linewidth=2, color='orange')
    axes[0,1].set_xlabel('Temperature (K)', fontsize=12)
    axes[0,1].set_ylabel('Entropy (J/K/mol)', fontsize=12)
    axes[0,1].set_title('Entropy', fontsize=14)
    axes[0,1].grid(alpha=0.3)
    
    # Heat capacity
    axes[1,0].plot(temps, heat_capacity, linewidth=2, color='green')
    axes[1,0].axhline(3*8.314, color='red', linestyle='--', label='Dulong-Petit (3R)')
    axes[1,0].set_xlabel('Temperature (K)', fontsize=12)
    axes[1,0].set_ylabel('Heat Capacity (J/K/mol)', fontsize=12)
    axes[1,0].set_title('Heat Capacity at Constant Volume', fontsize=14)
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # Internal energy (calculated from F = U - TS)
    internal_energy = free_energy + temps * entropy / 1000  # kJ/mol
    axes[1,1].plot(temps, internal_energy, linewidth=2, color='purple')
    axes[1,1].set_xlabel('Temperature (K)', fontsize=12)
    axes[1,1].set_ylabel('Internal Energy (kJ/mol)', fontsize=12)
    axes[1,1].set_title('Internal Energy', fontsize=14)
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('si_thermodynamics.png', dpi=150)
    plt.show()
    
    # Calculation of Debye temperature
    # Temperature where heat capacity reaches 63% of 3R (Dulong-Petit limit)
    R = 8.314  # J/K/mol
    target_cv = 0.63 * 3 * R
    idx = np.argmin(np.abs(heat_capacity - target_cv))
    theta_D = temps[idx]
    print(f"Debye temperature: {theta_D:.1f} K")
    print(f"Experimental value (Si): 645 K")
    

* * *

## 4.5 Thermal Expansion Coefficient

### Quasi-harmonic Approximation (QHA)

Perform phonon calculations at multiple volumes:

$$ \alpha(T) = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P = -\frac{1}{V}\frac{(\partial^2 F/\partial T \partial V)}{(\partial^2 F/\partial V^2)} $$

### Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate for different lattice constants
    lattice_constants = np.linspace(5.35, 5.51, 5)  # Å
    phonons = []
    
    for a in lattice_constants:
        print(f"Calculating phonon for a = {a:.3f} Å...")
        phonon = calculate_phonon('Si', a=a, dim=(2,2,2))
        phonons.append(phonon)
    
    # Calculate free energy for each temperature and volume
    temps = np.arange(0, 1000, 50)
    volumes = (lattice_constants / 5.43)**3  # Normalized volume
    free_energies = np.zeros((len(temps), len(volumes)))
    
    for i, phonon in enumerate(phonons):
        phonon.set_mesh([20, 20, 20])
        phonon.set_thermal_properties(t_step=50, t_max=1000, t_min=0)
        tp = phonon.get_thermal_properties_dict()
        free_energies[:, i] = tp['free_energy']
    
    # Find equilibrium volume for each temperature (F minimum)
    V_eq = np.zeros(len(temps))
    for i, T in enumerate(temps):
        # Quadratic polynomial fit
        coeffs = np.polyfit(volumes, free_energies[i], 2)
        V_eq[i] = -coeffs[1] / (2 * coeffs[0])  # Extremum
    
    # Thermal expansion coefficient
    alpha = np.gradient(V_eq, temps) / V_eq
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(temps, alpha * 1e6, linewidth=2)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Thermal expansion coefficient (10$^{-6}$ K$^{-1}$)', fontsize=12)
    plt.title('Si Thermal Expansion (QHA)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('si_thermal_expansion.png', dpi=150)
    plt.show()
    
    print(f"Thermal expansion coefficient at room temperature (300K): {alpha[6]*1e6:.2f} × 10⁻⁶ K⁻¹")
    print(f"Experimental value: 2.6 × 10⁻⁶ K⁻¹")
    

* * *

## 4.6 Applications of Phonon Calculations

### 1\. Lattice Thermal Conductivity

**Estimation from Boltzmann transport equation** (simplified version):

$$ \kappa = \frac{1}{3} \sum_{\mathbf{q},j} C_{\mathbf{q}j} v_{\mathbf{q}j}^2 \tau_{\mathbf{q}j} $$

  * $C_{\mathbf{q}j}$: Heat capacity per mode
  * $v_{\mathbf{q}j}$: Group velocity ($\partial\omega/\partial\mathbf{q}$)
  * $\tau_{\mathbf{q}j}$: Relaxation time (phonon scattering)

**Group velocity calculation in Phonopy** :
    
    
    # Calculate group velocity
    si_phonon.set_group_velocity()
    group_velocities = si_phonon.get_group_velocity()
    
    print("Group velocities at Gamma point:")
    print(group_velocities[0])  # [band, cartesian_direction]
    

### 2\. Superconducting Critical Temperature (Tc)

**McMillan formula** (simplified):

$$ T_c = \frac{\omega_{\text{log}}}{1.2} \exp\left[-\frac{1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\right] $$

  * $\omega_{\text{log}}$: Logarithmic average phonon frequency
  * $\lambda$: Electron-phonon coupling constant
  * $\mu^*$: Coulomb pseudopotential

### 3\. Detection of Phase Transitions

**Imaginary phonon modes** indicate structural instability:
    
    
    # Check for negative frequencies (imaginary modes)
    frequencies = si_phonon.get_frequencies(q=[0, 0, 0])
    if np.any(frequencies < -1e-3):
        print("Warning: Imaginary phonon modes detected!")
        print("Structure may be unstable.")
    

* * *

## 4.7 Chapter Summary

### What We Learned

  1. **Theory of Lattice Vibrations** \- Harmonic approximation \- Dynamical matrix \- Phonon dispersion relations

  2. **Classification of Phonons** \- Acoustic modes vs optical modes \- Longitudinal vs transverse waves

  3. **Practical Application with Phonopy** \- Phonon band structures \- Phonon density of states \- Fully automated scripts

  4. **Thermodynamic Properties** \- Free energy \- Internal energy \- Entropy \- Heat capacity \- Debye temperature

  5. **Applications** \- Thermal expansion coefficient (quasi-harmonic approximation) \- Thermal conductivity \- Phase transition detection

### Key Points

  * Phonons are quantized lattice vibrations
  * Harmonic approximation provides sufficient accuracy (in most cases)
  * Thermodynamic properties can be calculated from phonons
  * Good agreement with experiments (Si: Debye temperature, thermal expansion coefficient)

### To the Next Chapter

In Chapter 5, we will learn cutting-edge methods integrating DFT calculations with machine learning.

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

Explain the differences between acoustic and optical modes, including their physical meanings.

Sample Answer **Acoustic modes**: **Number**: 3 branches (1 LA + 2 TA) **Characteristics**: Zero frequency at Γ point ($\mathbf{k}=0$), $\omega \to 0$ as $\mathbf{k} \to 0$ **Physical meaning**: Translational motion of entire crystal, corresponds to sound waves \- All atoms vibrate in the same direction and phase \- Becomes elastic waves (sound waves) in the long-wavelength limit **Optical modes**: **Number**: $3(N_{\text{atom}} - 1)$ branches (for $N_{\text{atom}}$ atoms per unit cell) **Characteristics**: Finite frequency at Γ point ($\omega(\mathbf{k}=0) \neq 0$) **Physical meaning**: Relative motion of atoms within unit cell, couples with light (infrared) \- Different atoms vibrate in opposite phases \- In ionic crystals, creates electric dipole moment → infrared absorption **Example (NaCl crystal)**: \- Acoustic mode: Na and Cl move in the same direction \- Optical mode: Na and Cl move in opposite directions → dipole → light absorption 

### Problem 2 (Difficulty: Medium)

Explain the physical meaning of the Debye temperature $\theta_D$ and its relationship to heat capacity.

Sample Answer **Definition of Debye temperature $\theta_D$**: In the Debye model, all phonon modes are approximated by a single Debye frequency $\omega_D$: $$ \theta_D = \frac{\hbar\omega_D}{k_B} $$ **Physical meaning**: 1\. **Boundary temperature for quantum effects**: \- $T \ll \theta_D$: Quantum effects dominate (low temperature) \- $T \gg \theta_D$: Classical behavior (high temperature) 2\. **Indicator of phonon excitation**: \- $\theta_D$ corresponds to the maximum phonon frequency \- $T < \theta_D$: High-energy phonons are not excited \- $T > \theta_D$: All phonon modes are excited **Relationship to heat capacity**: **Low temperature ($T \ll \theta_D$)**: Debye $T^3$ law $$ C_V \propto T^3 $$ Quantum effects of phonons are significant. **High temperature ($T \gg \theta_D$)**: Dulong-Petit law $$ C_V \to 3Nk_B = 3R $$ All degrees of freedom are excited, classical limit. **Typical values**: \- Si: $\theta_D \approx 645$ K \- Diamond: $\theta_D \approx 2230$ K (hard lattice → high frequency) \- Pb: $\theta_D \approx 105$ K (soft lattice → low frequency) **Practical meaning**: \- High $\theta_D$ → Quantum effects important even at room temperature \- Low $\theta_D$ → Classical behavior at room temperature 

### Problem 3 (Difficulty: Hard)

Explain why multiple volume calculations are necessary when computing thermal expansion coefficient using the quasi-harmonic approximation (QHA).

Sample Answer **Limitations of harmonic approximation**: In the harmonic approximation, the volume dependence of free energy $F(V, T)$ is: $$ F_{\text{harm}}(V, T) = U_0(V) + k_B T \sum_{\mathbf{q},j} \ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}j}(V)}{2k_B T}\right)\right] $$ where $\omega_{\mathbf{q}j}(V)$ depends on volume $V$. However, the harmonic approximation assumes **constant lattice constant**, leading to these problems: 1\. **Cannot describe thermal expansion**: $(∂V/∂T)_P = 0$ 2\. **Only constant volume heat capacity**: $C_P = C_V$ (actually $C_P > C_V$) **Introduction of Quasi-harmonic Approximation (QHA)**: QHA applies **harmonic approximation independently at different volumes**: **Procedure**: 1\. Perform phonon calculations at multiple volumes $V_1, V_2, \ldots, V_n$ 2\. Calculate free energy $F(V_i, T)$ at each volume 3\. Find equilibrium volume $V_{\text{eq}}(T)$ that minimizes free energy at each temperature $T$: $$\left(\frac{\partial F}{\partial V}\right)_T = 0$$ 4\. Calculate thermal expansion coefficient: $$\alpha(T) = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P$$ **Why multiple volumes are needed**: \- To know the volume dependence $F(V, T)$ \- Curvature of free energy $(\partial^2 F/\partial V^2)$ is required \- Minimum position $V_{\text{eq}}(T)$ is temperature-dependent **Assumptions of QHA**: \- Considers volume dependence of phonon frequencies \- However, anharmonic interactions between phonons are ignored \- Harmonic approximation applied independently at each volume **Limitations of QHA**: \- Inaccurate at high temperatures (near melting point) \- Difficult to describe strong anharmonicity (e.g., negative thermal expansion materials) \- Full anharmonic calculations (TDEP, SSCHA, etc.) are necessary **Computational cost**: \- Phonon calculations at 5-10 volumes → 5-10 times the cost \- However, benefit of obtaining thermal expansion coefficient 

* * *

## Data License and Citation

### Software Used

  1. **Phonopy - Phonon calculation tool** (BSD 3-Clause) \- Phonon calculation software \- URL: https://phonopy.github.io/phonopy/ \- Citation: Togo, A., & Tanaka, I. (2015). *Scr. Mater.*, 108, 1-5.

  2. **GPAW** (GPL v3) + **ASE** (LGPL v2.1+) \- DFT force calculation backend \- Same licenses as Chapter 2

### Calculation Parameters

  * **Displacement amplitude** : 0.01 Å (standard for harmonic approximation)
  * **Supercell size** : 2×2×2 or larger (convergence testing required)
  * **q-point mesh** : 20×20×20 or larger (for DOS and enthalpy calculations)

* * *

## Code Reproducibility Checklist

### Environment Setup
    
    
    conda create -n phonon python=3.11
    conda activate phonon
    conda install -c conda-forge phonopy gpaw ase matplotlib
    pip install spglib  # Symmetry analysis (recommended)
    

### Calculation Time Estimates

System | Supercell | Displacements | DFT time/displacement | Total time  
---|---|---|---|---  
Si (2 atoms) | 2×2×2 | 3-6 | 10 min | ~1 hour  
NaCl (2 atoms) | 2×2×2 | 6-12 | 15 min | ~3 hours  
GaAs (2 atoms) | 3×3×3 | 6-12 | 30 min | ~6 hours  
  
### Troubleshooting

**Problem** : Imaginary phonons (negative frequency²) **Solutions** :
    
    
    # 1. Insufficient structure optimization
    # → Set force threshold to 0.01 eV/Å or below
    # 2. Supercell too small
    # → Increase from 2×2×2 to 3×3×3
    # 3. Symmetry breaking
    # → Check symmetry with spglib
    

* * *

## Practical Pitfalls and Solutions

### 1\. Insufficient Supercell Size
    
    
    # ❌ Insufficient: 1×1×1 (does not converge)
    phonon = Phonopy(unitcell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # ✅ Recommended: 2×2×2 or larger (convergence test required)
    for dim in [(2,2,2), (3,3,3), (4,4,4)]:
        phonon = Phonopy(unitcell, np.diag(dim))
        # Check phonon frequency convergence
    

### 2\. Incomplete Structure Optimization
    
    
    # ❌ Insufficient: Coarse optimization → imaginary phonons
    opt.run(fmax=0.1)  # Forces too large
    
    # ✅ Recommended: High-precision optimization
    opt.run(fmax=0.01)  # 0.01 eV/Å or below
    

### 3\. Insufficient q-point Mesh (DOS Calculation)
    
    
    # ❌ Coarse: 10×10×10 (misses features)
    phonon.set_mesh([10, 10, 10])
    
    # ✅ Recommended: 20×20×20 or larger
    phonon.set_mesh([20, 20, 20])
    

### 4\. Unit Conversion Errors
    
    
    # Phonopy internal units: THz
    # Conversions:
    # - cm⁻¹ = THz × 33.356
    # - meV = THz × 4.136
    # - K (temperature) = THz × 47.992
    

* * *

## Quality Assurance Checklist

### Validity of Phonon Calculation

  * [ ] Acoustic modes at Γ point (k=0) are 0 Hz (error < 0.1 THz)
  * [ ] 3 acoustic modes (translational degrees of freedom)
  * [ ] All frequencies are real (no imaginary modes)
  * [ ] Bands degenerate at high-symmetry points
  * [ ] DOS integrates to 3N (number of degrees of freedom)

### Validity of Thermodynamic Properties

  * [ ] Heat capacity → 3R in high-temperature limit (Dulong-Petit)
  * [ ] Debye temperature within ±20% of literature values
  * [ ] Thermal expansion coefficient is positive (normal materials)
  * [ ] Entropy increases monotonically with temperature

* * *

## References

  1. Dove, M. T. (1993). *Introduction to Lattice Dynamics*. Cambridge University Press.

  2. Togo, A., & Tanaka, I. (2015). "First principles phonon calculations in materials science." *Scripta Materialia*, 108, 1-5. DOI: [10.1016/j.scriptamat.2015.07.021](<https://doi.org/10.1016/j.scriptamat.2015.07.021>)

  3. Phonopy Documentation: https://phonopy.github.io/phonopy/

  4. Shulumba, N., et al. (2017). "Temperature-dependent elastic properties of Ti$_x$Zr$_{1-x}$N alloys." _Applied Physics Letters_ , 111, 061901.

* * *

## Author Information

**Created by** : MI Knowledge Hub Content Team **Created** : 2025-10-17 **Version** : 1.0 **Series** : Computational Materials Science Basics v1.0

**License** : Creative Commons BY-NC-SA 4.0
