---
title: "Chapter 3: Molecular Dynamics (MD) Simulation"
chapter_title: "Chapter 3: Molecular Dynamics (MD) Simulation"
subtitle: Calculation of Atomic Motion and Thermodynamic Properties
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 7
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Molecular Dynamics (MD) Simulation

Understand the relationship between equations of motion and potentials, and grasp the basics of temperature and pressure control.

**💡 Note:** Temperature is an indicator of motion intensity, pressure is an indicator of packing density. Differences in control methods reflect differences in "how to create the environment."

## Learning Objectives

By completing this chapter, you will be able to: \- Understand the basic principles of MD simulation (Newton's equations of motion, time integration) \- Understand the concepts of force fields and potentials \- Explain the differences between statistical ensembles (NVE, NVT, NPT) \- Execute basic MD simulations using LAMMPS \- Understand the differences between Ab Initio MD (AIMD) and Classical MD

* * *

## 3.1 Basic Principles of Molecular Dynamics

### Newton's Equations of Motion

The core of MD simulation is **Newton's equations of motion** from classical mechanics:

$$ m_i \frac{d^2 \mathbf{r}_i}{dt^2} = \mathbf{F}_i = -\nabla_i U(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N) $$

  * $m_i$: Mass of atom $i$
  * $\mathbf{r}_i$: Position of atom $i$
  * $\mathbf{F}_i$: Force acting on atom $i$
  * $U(\mathbf{r}_1, \ldots, \mathbf{r}_N)$: Potential energy

### MD Simulation Procedure
    
    
    ```mermaid
    flowchart TD
        A[Set initial positions and velocities] --> B[Calculate forces: F = -∇U]
        B --> C[Update positions and velocities: Time integration]
        C --> D[Calculate physical quantities: T, P, E, etc.]
        D --> E{Time finished?}
        E -->|No| B
        E -->|Yes| F[Data analysis and visualization]
    
        style A fill:#e3f2fd
        style F fill:#c8e6c9
    ```

**Time step** : Typically $\Delta t = 0.5$-$2$ fs (femtosecond, $10^{-15}$ seconds)

* * *

## 3.2 Time Integration Algorithms

### Verlet Method (Most Basic)

Taylor expansion of position:

$$ \mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 + O(\Delta t^3) $$

$$ \mathbf{r}(t - \Delta t) = \mathbf{r}(t) - \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 + O(\Delta t^3) $$

Adding these two equations:

$$ \mathbf{r}(t + \Delta t) = 2\mathbf{r}(t) - \mathbf{r}(t - \Delta t) + \mathbf{a}(t)\Delta t^2 $$

**Features** : \- ✅ Simple, memory efficient \- ✅ Preserves time-reversal symmetry \- ❌ Velocity does not appear explicitly

### Velocity Verlet Method (Most Commonly Used)

$$ \mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 $$

$$ \mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2}[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)]\Delta t $$

**Features** : \- ✅ Updates position and velocity simultaneously \- ✅ Good energy conservation \- ✅ Most widely used

### Leap-frog Method

$$ \mathbf{v}(t + \frac{\Delta t}{2}) = \mathbf{v}(t - \frac{\Delta t}{2}) + \mathbf{a}(t)\Delta t $$

$$ \mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \frac{\Delta t}{2})\Delta t $$

Position and velocity are staggered by half a time step, moving like "stepping stones."

### Python Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lennard_jones(r, epsilon=1.0, sigma=1.0):
        """Lennard-Jones potential"""
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    
    def lj_force(r, epsilon=1.0, sigma=1.0):
        """Force from Lennard-Jones potential"""
        return 24 * epsilon * (2*(sigma/r)**13 - (sigma/r)**7) / r
    
    def velocity_verlet_md(N_steps=1000, dt=0.001):
        """
        1D MD simulation using Velocity Verlet method
        Two Lennard-Jones particles
        """
        # Initial conditions
        r = np.array([0.0, 2.0])  # Position
        v = np.array([0.5, -0.5])  # Velocity
        m = np.array([1.0, 1.0])  # Mass
    
        # History arrays
        r_history = np.zeros((N_steps, 2))
        v_history = np.zeros((N_steps, 2))
        E_history = np.zeros(N_steps)
    
        for step in range(N_steps):
            # Calculate force
            r12 = r[1] - r[0]
            F = lj_force(abs(r12))
            a = np.array([-F/m[0], F/m[1]])  # Acceleration
    
            # Update position
            r = r + v * dt + 0.5 * a * dt**2
    
            # Calculate new force
            r12_new = r[1] - r[0]
            F_new = lj_force(abs(r12_new))
            a_new = np.array([-F_new/m[0], F_new/m[1]])
    
            # Update velocity
            v = v + 0.5 * (a + a_new) * dt
    
            # Energy calculation
            KE = 0.5 * np.sum(m * v**2)
            PE = lennard_jones(abs(r12_new))
            E_total = KE + PE
    
            # Store
            r_history[step] = r
            v_history[step] = v
            E_history[step] = E_total
    
        return r_history, v_history, E_history
    
    # Run simulation
    r_hist, v_hist, E_hist = velocity_verlet_md(N_steps=5000, dt=0.001)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time = np.arange(len(r_hist)) * 0.001
    
    # Position
    axes[0,0].plot(time, r_hist[:, 0], label='Particle 1')
    axes[0,0].plot(time, r_hist[:, 1], label='Particle 2')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Position')
    axes[0,0].set_title('Particle Positions')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Velocity
    axes[0,1].plot(time, v_hist[:, 0], label='Particle 1')
    axes[0,1].plot(time, v_hist[:, 1], label='Particle 2')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Velocity')
    axes[0,1].set_title('Particle Velocities')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Energy conservation
    axes[1,0].plot(time, E_hist)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Total Energy')
    axes[1,0].set_title('Energy Conservation (NVE)')
    axes[1,0].grid(alpha=0.3)
    
    # Phase space trajectory
    axes[1,1].plot(r_hist[:, 0] - r_hist[:, 1], v_hist[:, 0], alpha=0.5)
    axes[1,1].set_xlabel('Relative Position')
    axes[1,1].set_ylabel('Velocity 1')
    axes[1,1].set_title('Phase Space Trajectory')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('md_verlet.png', dpi=150)
    plt.show()
    
    print(f"Energy fluctuation: {np.std(E_hist):.6f} (ideally close to 0)")
    

**Execution result** : Energy fluctuation < $10^{-6}$ (good energy conservation)

* * *

## 3.3 Force Fields and Potentials

### Lennard-Jones Potential

The most basic two-body potential:

$$ U_{\text{LJ}}(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right] $$

  * $\epsilon$: Depth of potential (binding energy)
  * $\sigma$: Equilibrium distance scale
  * $r^{-12}$: Short-range repulsion (Pauli repulsion)
  * $r^{-6}$: Long-range attraction (van der Waals force)

**Applications** : Noble gases (Ar, Ne), coarse-grained models

### Many-body Potentials

**Embedded Atom Method (EAM)** \- For metals:

$$ U_{\text{EAM}} = \sum_i F_i(\rho_i) + \frac{1}{2}\sum_{i \neq j} \phi_{ij}(r_{ij}) $$

  * $F_i(\rho_i)$: Embedding energy (function of electron density $\rho_i$)
  * $\phi_{ij}(r_{ij})$: Pair potential

**Tersoff/Brenner** \- For covalent systems (C, Si, Ge):

$$ U_{\text{Tersoff}} = \sum_i \sum_{j>i} [f_R(r_{ij}) - b_{ij} f_A(r_{ij})] $$

$b_{ij}$ is the bond order, dependent on the surrounding environment.

### Water Force Field

**TIP3P** (3-point charge model): \- O atom has negative charge $-0.834e$ \- Each H atom has positive charge $+0.417e$ \- Lennard-Jones potential (O-O interactions only)

**Features** : \- ✅ Computationally fast \- ✅ Reproduces liquid water density and diffusion coefficient \- ❌ Ice structure is inaccurate

* * *

## 3.4 Statistical Ensembles

### NVE (Microcanonical) Ensemble

**Conditions** : Number of particles $N$, volume $V$, energy $E$ are constant (isolated system)

**Implementation** : Time integration only (no thermostat)

**Applications** : \- Energy conservation tests \- Fundamental theoretical research

### NVT (Canonical) Ensemble

**Conditions** : Number of particles $N$, volume $V$, temperature $T$ are constant (in contact with heat bath)

**Nosé-Hoover Thermostat** :

$$ \frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i $$

$$ \frac{d\mathbf{v}_i}{dt} = \frac{\mathbf{F}_i}{m_i} - \zeta \mathbf{v}_i $$

$$ \frac{d\zeta}{dt} = \frac{1}{Q}\left(\sum_i m_i v_i^2 - 3NkT\right) $$

  * $\zeta$: Thermostat variable (acts like a friction coefficient)
  * $Q$: "Mass" of thermostat (controls relaxation time)

**Applications** : \- Calculation of thermodynamic quantities in equilibrium states \- Study of phase transitions

### NPT (Isothermal-Isobaric) Ensemble

**Conditions** : Number of particles $N$, pressure $P$, temperature $T$ are constant (heat bath + pressure bath)

**Parrinello-Rahman Method** : Cell shape and size can vary

**Applications** : \- Direct comparison with experimental conditions \- Calculation of lattice constants \- Phase transitions (solid-liquid, etc.)

### Comparison Table

Ensemble | Conserved | Fluctuates | Applications  
---|---|---|---  
NVE | $N, V, E$ | $T, P$ | Isolated system, validation  
NVT | $N, V, T$ | $E, P$ | Thermal equilibrium  
NPT | $N, P, T$ | $E, V$ | Experimental conditions  
  
* * *

## 3.5 Practice with LAMMPS

### What is LAMMPS?

**LAMMPS** (Large-scale Atomic/Molecular Massively Parallel Simulator): \- Developed by Sandia National Laboratory \- Open source, free \- Optimized for parallel computing (capable of billions of atoms scale)

### Example 1: Ar Gas Equilibration (NVT)
    
    
    # LAMMPS input file: ar_nvt.in
    
    # Initial setup
    units lj                    # Lennard-Jones unit system
    atom_style atomic
    dimension 3
    boundary p p p              # Periodic boundary conditions
    
    # Create system
    region box block 0 10 0 10 0 10
    create_box 1 box
    create_atoms 1 random 100 12345 box
    
    # Set mass
    mass 1 1.0
    
    # Potential
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0 2.5  # epsilon, sigma, cutoff
    
    # Initial velocity (corresponding to temperature 1.0)
    velocity all create 1.0 87287 dist gaussian
    
    # NVT setup (Nosé-Hoover)
    fix 1 all nvt temp 1.0 1.0 0.1
    
    # Time step
    timestep 0.005
    
    # Thermodynamic output
    thermo 100
    thermo_style custom step temp press pe ke etotal vol density
    
    # Save trajectory
    dump 1 all custom 1000 ar_nvt.lammpstrj id type x y z vx vy vz
    
    # Run
    run 10000
    
    # Finish
    write_data ar_nvt_final.data
    

**Execution** :
    
    
    lammps -in ar_nvt.in
    

### Example 2: Controlling LAMMPS from Python
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 2: Controlling LAMMPS from Python
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from lammps import lammps
    import numpy as np
    import matplotlib.pyplot as plt
    
    # LAMMPS instance
    lmp = lammps()
    
    # Execute input file
    lmp.file('ar_nvt.in')
    
    # Extract thermodynamic data
    temps = lmp.extract_compute("thermo_temp", 0, 0)
    press = lmp.extract_compute("thermo_press", 0, 0)
    
    print(f"Equilibrium temperature: {temps:.3f} (target: 1.0)")
    print(f"Equilibrium pressure: {press:.3f}")
    
    # Calculate radial distribution function (RDF)
    lmp.command("compute myRDF all rdf 100")
    lmp.command("fix 2 all ave/time 100 1 100 c_myRDF[*] file ar_rdf.dat mode vector")
    lmp.command("run 5000")
    
    # Load and plot RDF
    rdf_data = np.loadtxt('ar_rdf.dat')
    r = rdf_data[:, 1]
    g_r = rdf_data[:, 2]
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, g_r, linewidth=2)
    plt.xlabel('r (σ)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Radial Distribution Function (Ar gas)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('ar_rdf.png', dpi=150)
    plt.show()
    
    lmp.close()
    

* * *

### Example 3: MD Simulation of Water Molecules (TIP3P)
    
    
    # water_nvt.in
    
    units real                  # Real unit system (Å, fs, kcal/mol)
    atom_style full
    dimension 3
    boundary p p p
    
    # Read data file (216 water molecules)
    read_data water_box.data
    
    # TIP3P water model
    pair_style lj/cut/coul/long 10.0
    pair_coeff 1 1 0.102 3.188   # O-O
    pair_coeff * 2 0.0 0.0       # H atoms have no LJ
    kspace_style pppm 1e-4       # Long-range Coulomb (Ewald sum)
    
    # Bonds and angles
    bond_style harmonic
    bond_coeff 1 450.0 0.9572    # O-H bond
    angle_style harmonic
    angle_coeff 1 55.0 104.52    # H-O-H angle
    
    # SHAKE constraint (fix O-H bonds)
    fix shake all shake 0.0001 20 0 b 1 a 1
    
    # NVT (300K)
    fix 1 all nvt temp 300.0 300.0 100.0
    
    # Time step
    timestep 1.0  # 1 fs
    
    # Output
    thermo 100
    dump 1 all custom 1000 water.lammpstrj id mol type x y z
    
    # Run (100 ps)
    run 100000
    

**Calculation of Diffusion Coefficient** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculation of Diffusion Coefficient:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from MDAnalysis import Universe
    import matplotlib.pyplot as plt
    
    # Load trajectory
    u = Universe('water_box.data', 'water.lammpstrj', format='LAMMPSDUMP')
    
    # Select oxygen atoms only
    oxygens = u.select_atoms('type 1')
    
    # Calculate mean square displacement (MSD)
    n_frames = len(u.trajectory)
    msd = np.zeros(n_frames)
    
    for i, ts in enumerate(u.trajectory):
        if i == 0:
            r0 = oxygens.positions.copy()
        dr = oxygens.positions - r0
        msd[i] = np.mean(np.sum(dr**2, axis=1))
    
    # Time axis (fs)
    time = np.arange(n_frames) * 1.0
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(time/1000, msd, linewidth=2)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('MSD (Å²)', fontsize=12)
    plt.title('Mean Square Displacement (H₂O)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('water_msd.png', dpi=150)
    plt.show()
    
    # Calculate diffusion coefficient (Einstein relation)
    # MSD = 6Dt → D = slope / 6
    slope = np.polyfit(time[len(time)//2:], msd[len(time)//2:], 1)[0]
    D = slope / 6 / 1000  # Å²/ps → 10⁻⁵ cm²/s
    print(f"Diffusion coefficient: {D:.2f} × 10⁻⁵ cm²/s")
    print(f"Experimental value: 2.30 × 10⁻⁵ cm²/s (300K)")
    

* * *

## 3.6 Ab Initio MD (AIMD)

### Classical MD vs Ab Initio MD

Item | Classical MD | Ab Initio MD  
---|---|---  
**Force Calculation** | Empirical potential (force field) | DFT calculation (first-principles)  
**Accuracy** | Depends on force field | Quantum mechanically accurate  
**Computational Cost** | Low ($\sim$1 ns/day) | Extremely high ($\sim$10 ps/day)  
**System Size** | Millions of atoms | Hundreds of atoms  
**Applications** | Large systems, long times | Chemical reactions, electronic states  
  
### Born-Oppenheimer MD (BOMD)

Execute DFT calculation at each time step:
    
    
    1. Give atomic configuration R(t)
    2. Calculate ground state energy E(R(t)) by DFT
    3. Calculate force F = -∇E
    4. Calculate R(t+Δt) by Newton's equation
    5. Return to 1
    

### AIMD Implementation Example with GPAW
    
    
    from ase import Atoms
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase import units
    from gpaw import GPAW, PW
    
    # Create water molecule
    h2o = Atoms('H2O',
                positions=[[0.00, 0.00, 0.00],
                           [0.96, 0.00, 0.00],
                           [0.24, 0.93, 0.00]])
    h2o.center(vacuum=5.0)
    
    # DFT calculator (instead of force field)
    calc = GPAW(mode=PW(400),
                xc='PBE',
                txt='h2o_aimd.txt')
    
    h2o.calc = calc
    
    # Initial velocity (300K)
    MaxwellBoltzmannDistribution(h2o, temperature_K=300)
    
    # Velocity Verlet MD
    dyn = VelocityVerlet(h2o, timestep=1.0*units.fs,
                         trajectory='h2o_aimd.traj')
    
    # Run for 10 ps (actually takes very long time)
    def print_energy(a=h2o):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        print(f"Time: {dyn.get_time()/units.fs:.1f} fs, "
              f"Epot: {epot:.3f} eV, "
              f"Ekin: {ekin:.3f} eV, "
              f"Etot: {epot+ekin:.3f} eV")
    
    dyn.attach(print_energy, interval=10)
    dyn.run(100)  # 100 steps = 100 fs
    

**Applications** : \- Study of chemical reactions \- Mechanisms of phase transitions \- Novel materials without existing force fields

* * *

## 3.7 Chapter Summary

### What We Learned

  1. **Basic Principles of MD** \- Newton's equations of motion \- Time integration algorithms (Verlet, Velocity Verlet, Leap-frog) \- Energy conservation law

  2. **Force Fields and Potentials** \- Lennard-Jones (noble gases) \- EAM (metals) \- Tersoff/Brenner (covalent systems) \- TIP3P (water)

  3. **Statistical Ensembles** \- NVE (microcanonical) \- NVT (canonical, Nosé-Hoover thermostat) \- NPT (isothermal-isobaric, Parrinello-Rahman)

  4. **Practice with LAMMPS** \- Creating input files \- Equilibration simulations \- Radial distribution function, diffusion coefficient

  5. **Ab Initio MD** \- Differences from Classical MD \- Born-Oppenheimer MD \- Implementation with GPAW

### Key Points

  * MD is based on classical mechanics
  * Time step is about 1 fs
  * Statistical ensembles reproduce experimental conditions
  * Force field choice determines accuracy
  * AIMD has high accuracy but high computational cost

### To the Next Chapter

In Chapter 4, we will learn about lattice vibrations (phonons) and calculation of thermodynamic properties.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the differences between the Velocity Verlet method and the Leap-frog method.

Sample Answer **Velocity Verlet Method**: \- Updates position $\mathbf{r}$ and velocity $\mathbf{v}$ at the same time $t$ \- Velocity uses average of current and next time accelerations \- Thermodynamic quantities (temperature, kinetic energy) can be directly calculated **Leap-frog Method**: \- Position $\mathbf{r}(t)$ and velocity $\mathbf{v}(t+\Delta t/2)$ are staggered by half a time step \- Updates alternately like "stepping stones" \- Velocity interpolation needed to calculate thermodynamic quantities **Both methods**: \- Equivalent energy conservation \- Second-order accuracy ($O(\Delta t^2)$) \- Possess time-reversal symmetry 

### Problem 2 (Difficulty: medium)

Explain when to use NVT ensemble versus NPT ensemble, with specific examples.

Sample Answer **NVT Ensemble (N, V, T constant)**: **Use cases**: \- Thermodynamic properties of solids (specific heat, thermal expansion coefficient) \- Structure factor, radial distribution function of liquids \- Surface/interface simulations (fixed volume) \- When studying behavior at specific density **Specific examples**: \- Atomic vibrations in Si crystal at 300K \- Structural fluctuations of proteins in aqueous solution \- Transport properties of lithium-ion battery electrolytes **NPT Ensemble (N, P, T constant)**: **Use cases**: \- Direct comparison with experimental conditions (1 atm, room temperature, etc.) \- Phase transitions (solid-liquid, liquid-gas) \- Calculation of lattice constants \- Temperature and pressure dependence of density **Specific examples**: \- Density calculation of liquid water at 1 atm, 300K \- Structural changes of materials under high pressure \- Calculation of thermal expansion coefficient \- Ice melting simulation **Decision criteria**: \- Experiments at constant pressure → NPT \- Theoretical research with fixed density → NVT \- Study of phase transitions → NPT (observe volume changes) 

### Problem 3 (Difficulty: hard)

Estimate the computational cost difference between Classical MD and Ab Initio MD for a 100-atom system. Assume that DFT calculation takes 1 second per MD step.

Sample Answer **Assumptions**: \- System: 100 atoms \- Time step: $\Delta t = 1$ fs \- DFT calculation: 1 second per step **Classical MD**: Force calculation: Force field (analytical formula) \- Lennard-Jones case: $O(N^2)$ or $O(N)$ (using cutoff) \- Calculation time per step: $\sim 10^{-3}$ seconds (100 atoms) **Simulation time**: \- 1 ns (nanosecond) = $10^6$ fs = $10^6$ steps \- Total calculation time: $10^6 \times 10^{-3}$ seconds = 1000 seconds ≈ **17 minutes** **Ab Initio MD (AIMD)**: Force calculation: DFT (SCF calculation) \- Calculation time per step: 1 second (assumption) **Simulation time**: \- 10 ps (picosecond) = $10^4$ fs = $10^4$ steps \- Total calculation time: $10^4 \times 1$ seconds = 10000 seconds ≈ **2.8 hours** **Comparison**: | Item | Classical MD | AIMD | |-----|-------------|-----| | Calculation time per step | 0.001 sec | 1 sec | | Reachable time | 1 ns (17 min) | 10 ps (2.8 hours) | | **Time scale ratio** | **100x longer** | - | | **Computational cost ratio** | 1 | **1000x** | **Conclusion**: \- AIMD is about 1000x slower per step \- For the same calculation time, Classical MD can simulate 100x longer times \- AIMD is applicable to chemical reactions (ps-ns scale) \- Diffusion processes (ns scale and beyond) require Classical MD **Practical strategy**: 1\. Short time simulation with AIMD (10-100 ps) 2\. Train force field (Machine Learning Potential) from AIMD results 3\. Long time simulation with Classical MD using MLP (ns-μs) → We will learn about MLP in detail in Chapter 5! 

* * *

## Data License and Citation

### Software and Force Fields Used

  1. **LAMMPS - Molecular Dynamics Simulator** (GPL v2) \- Large-scale parallel MD calculation software \- URL: https://www.lammps.org/ \- Citation: Thompson, A. P., et al. (2022). _Comp. Phys. Comm._ , 271, 108171.

  2. **ASE Molecular Dynamics Module** (LGPL v2.1+) \- Python MD library \- URL: https://wiki.fysik.dtu.dk/ase/ase/md.html

  3. **Force Field Database** \- **TIP3P water model** : Jorgensen, W. L., et al. (1983). _J. Chem. Phys._ , 79, 926. \- **EAM metal potential** : OpenKIM Repository (CDDL License)

     * URL: https://openkim.org/
     * **Tersoff Si potential** : Tersoff, J. (1988). _Phys. Rev. B_ , 38, 9902.

### Sources of Standard Parameters

  * **Time step** : 1-2 fs (standard literature value)
  * **Nosé-Hoover thermostat** : Nosé, S. (1984). _J. Chem. Phys._ , 81, 511.
  * **Parrinello-Rahman barostat** : Parrinello, M., & Rahman, A. (1981). _J. Appl. Phys._ , 52, 7182.

* * *

## Code Reproducibility Checklist

### Environment Setup
    
    
    # LAMMPS installation (Anaconda recommended)
    conda create -n md-sim python=3.11
    conda activate md-sim
    conda install -c conda-forge lammps
    conda install numpy matplotlib MDAnalysis
    
    # Version check
    lmp -v  # LAMMPS 23Jun2022 or later
    python -c "import lammps; print(lammps.__version__)"
    

### Hardware Requirements

System Size | Memory | CPU Time (10 ps) | Recommended Cores  
---|---|---|---  
100 atoms (Ar) | ~100 MB | ~1 min | 1 core  
1,000 atoms (water) | ~500 MB | ~10 min | 2-4 cores  
10,000 atoms (protein) | ~2 GB | ~2 hours | 8-16 cores  
100,000 atoms (nanoparticle) | ~10 GB | ~20 hours | 32-64 cores  
  
### Calculation Time Estimation
    
    
    # Simple estimation formula
    N_atoms = 1000
    N_steps = 100000
    timestep = 1.0  # fs
    time_per_step = N_atoms * 1e-5  # seconds (rough estimate)
    total_time = N_steps * time_per_step / 60  # minutes
    print(f"Estimated calculation time: {total_time:.1f} minutes")
    

### Troubleshooting

**Problem** : `ERROR: Unknown pair style` **Solution** :
    
    
    # Install additional LAMMPS packages
    conda install -c conda-forge lammps-packages
    

**Problem** : Energy divergence (NaN) **Solution** :
    
    
    # Reduce time step
    timestep 0.5  # 1.0 → 0.5 fs
    
    # Or check for overlapping initial configuration
    minimize 1.0e-4 1.0e-6 1000 10000
    

**Problem** : Temperature does not converge to target value **Solution** :
    
    
    # Adjust damping parameter of thermostat
    fix 1 all nvt temp 300.0 300.0 100.0  # (100.0 = 100*timestep)
    # Change 100x → 10x (stronger coupling)
    fix 1 all nvt temp 300.0 300.0 10.0
    

* * *

## Practical Pitfalls and Countermeasures

### 1\. Incorrect Time Step Selection

**Pitfall** : Time step too large causing energy divergence
    
    
    # ❌ Too large: 2 fs for system containing hydrogen atoms
    timestep 2.0  # Cannot capture O-H vibration (~10 fs period)
    
    # ✅ Recommended values:
    # - Heavy atoms only: 2 fs
    # - Including hydrogen: 0.5-1 fs
    # - Chemical reactions: 0.25 fs
    timestep 0.5
    

**Validation method** :
    
    
    # Energy conservation check (NVE)
    fix 1 all nve
    run 10000
    variable E_drift equal abs((etotal[10000]-etotal[1])/etotal[1])
    if "${E_drift} > 0.001" then "print 'WARNING: Energy drift > 0.1%'"
    

### 2\. Insufficient Equilibration Time

**Pitfall** : Starting production calculation without equilibration
    
    
    # ❌ Insufficient: Start statistics immediately
    fix 1 all nvt temp 300.0 300.0 100.0
    run 1000  # 1 ps (insufficient)
    # Start statistics here → non-equilibrium state
    
    # ✅ Recommended: Sufficient equilibration
    # 1. Energy relaxation
    minimize 1.0e-4 1.0e-6 1000 10000
    
    # 2. Temperature equilibration (10-50 ps)
    fix 1 all nvt temp 300.0 300.0 100.0
    run 50000  # 50 ps
    
    # 3. Start statistics collection
    reset_timestep 0
    run 100000  # 100 ps
    

**Equilibration criteria** : \- Temperature fluctuation < ±5% \- Energy fluctuation < ±1% \- Pressure fluctuation < ±10%

### 3\. Misuse of Periodic Boundary Conditions

**Pitfall** : Molecules cut across periodic boundaries
    
    
    # ❌ Wrong: Cell size too small
    region box block 0 5 0 5 0 5  # 5 Å × 5 Å × 5 Å
    # Water molecule (~3 Å) crosses boundaries
    
    # ✅ Correct: More than 3 times molecular size
    region box block 0 15 0 15 0 15  # 15 Å × 15 Å × 15 Å
    

**Validation** :
    
    
    # Relationship between cutoff distance and cell size
    # Cell size > 2 × cutoff distance
    pair_style lj/cut 10.0  # Cutoff 10 Å
    # → Cell size > 20 Å required
    

### 4\. Confusion of Unit Systems

**Pitfall** : Mixing LAMMPS unit systems
    
    
    # ❌ Confusion: Mixing real and metal
    units real  # Energy: kcal/mol
    pair_coeff 1 1 1.0 3.4  # LJ-epsilon = 1.0 kcal/mol
    
    # Changing to metal midway (dangerous!)
    units metal  # Energy: eV
    
    # ✅ Correct: Unified in one unit system
    units real
    # Set all parameters in real units
    

**Major unit systems** : | Unit System | Energy | Distance | Time | Applications | |-------|-----------|-----|------|-----| | real | kcal/mol | Å | fs | Biomolecules | | metal | eV | Å | ps | Metals, materials | | si | J | m | s | Education | | lj | ε | σ | τ | Theory |

### 5\. Incorrect Force Field Parameters

**Pitfall** : Wrong units or values for force field parameters
    
    
    # ❌ Wrong: Inaccurate TIP3P water parameters
    pair_coeff 1 1 0.102 3.188  # O-O (wrong: mixing kcal/mol and Å)
    
    # ✅ Correct: Accurate literature values
    # TIP3P (Jorgensen 1983, real units):
    pair_coeff 1 1 0.1521 3.1507  # ε=0.1521 kcal/mol, σ=3.1507 Å
    bond_coeff 1 450.0 0.9572     # k=450 kcal/mol/Å², r0=0.9572 Å
    angle_coeff 1 55.0 104.52     # k=55 kcal/mol/rad², θ0=104.52°
    

* * *

## Quality Assurance Checklist

### MD Simulation Validity Verification

#### Energy Conservation (NVE)

  * [ ] Total energy drift < 0.1% / ns
  * [ ] Kinetic and potential energy interconvert
  * [ ] Temperature fluctuations follow Maxwell distribution

#### Temperature Control (NVT)

  * [ ] Average temperature within ±2% of target
  * [ ] Standard deviation of temperature matches theoretical value (√(2T²/3N))
  * [ ] Temperature time series is stationary (no trend)

#### Pressure Control (NPT)

  * [ ] Average pressure within ±10% of target
  * [ ] Volume has reached equilibrium (fluctuation < 1%)
  * [ ] Density within ±5% of experimental value

#### Structural Validity

  * [ ] Radial distribution function (RDF) agrees with experiment/literature
  * [ ] Coordination number is chemically reasonable
  * [ ] Bond length and angle distributions within expected range

#### Dynamic Properties

  * [ ] Mean square displacement (MSD) is linear in time (diffusive regime)
  * [ ] Diffusion coefficient within ±50% of experimental value
  * [ ] Velocity autocorrelation function decays properly

### Numerical Calculation Sanity

  * [ ] No atoms flying out of system
  * [ ] No abnormal energy values (NaN, Inf)
  * [ ] Force magnitude is reasonable (< 10 eV/Å)
  * [ ] Minimum interatomic distance is reasonable (> 0.8 Å)

### MD-Specific Quality Checks

#### Statistical Accuracy

  * [ ] Sampling time is at least 10 times correlation time
  * [ ] Error evaluation by block averaging method
  * [ ] Reproducibility confirmed with multiple independent runs

#### System Size Effects

  * [ ] Cell size is at least twice the cutoff
  * [ ] System size dependence checked (if possible)
  * [ ] Finite size effects are negligible

* * *

## References

  1. Frenkel, D., & Smit, B. (2001). _Understanding Molecular Simulation: From Algorithms to Applications_ (2nd ed.). Academic Press.

  2. Haile, J. M. (1992). _Molecular Dynamics Simulation: Elementary Methods_. Wiley-Interscience.

  3. Plimpton, S. (1995). "Fast Parallel Algorithms for Short-Range Molecular Dynamics." _Journal of Computational Physics_ , 117(1), 1-19. DOI: [10.1006/jcph.1995.1039](<https://doi.org/10.1006/jcph.1995.1039>)

  4. LAMMPS Documentation: https://docs.lammps.org/

  5. ASE-MD Documentation: https://wiki.fysik.dtu.dk/ase/ase/md.html

* * *

## Author Information

**Author** : MI Knowledge Hub Content Team **Created** : 2025-10-17 **Version** : 1.0 **Series** : Introduction to Computational Materials Science v1.0

**License** : Creative Commons BY-NC-SA 4.0
