---
title: "Chapter 3: Introduction to First-Principles Calculations (DFT Fundamentals)"
chapter_title: "Chapter 3: Introduction to First-Principles Calculations (DFT Fundamentals)"
subtitle: Solving Electronic Structure of Materials with Density Functional Theory
difficulty: Advanced
---

This chapter covers the fundamentals of Introduction to First, which principles calculations (dft fundamentals). You will learn essential concepts and techniques.

## What You Will Learn in This Chapter

### Learning Objectives (3 Levels)

#### Basic Level

  * Explain the basic principles of DFT (Hohenberg-Kohn theorem, Kohn-Sham equations)
  * Understand the differences between exchange-correlation functionals (LDA, GGA, hybrid)
  * Explain the basic workflow of DFT calculations

#### Intermediate Level

  * Create crystal structures and prepare DFT calculation inputs using ASE/Pymatgen
  * Understand the roles of VASP input files (INCAR, POSCAR, KPOINTS, POTCAR)
  * Perform convergence tests for k-point mesh and cutoff energy

#### Advanced Level

  * Select appropriate functionals based on material properties
  * Understand the differences between pseudopotentials and PAW method and use them appropriately
  * Build DFT calculation workflows for practical research

## What Are First-Principles Calculations?

First-principles calculations are computational methods that predict material properties theoretically by starting from the fundamental principles of quantum mechanics, without using empirical parameters. They are powerful tools for materials development because they can predict properties of experimentally unknown materials.

### Difficulties of the Many-Body Schrödinger Equation

Ideally, we should solve the full wave function $\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N, \mathbf{R}_1, \ldots, \mathbf{R}_M)$ for a system consisting of $N$ electrons and $M$ nuclei:

$$ \hat{H}\Psi = E\Psi $$ 

However, solving the many-body Schrödinger equation exactly is impossible from a computational standpoint. For example, a system with 100 electrons requires solving a partial differential equation in $3N = 300$ dimensions.

### Born-Oppenheimer Approximation

Since nuclear masses are about 1,000 times heavier than electrons, we can assume that electrons instantly follow nuclear motion (adiabatic approximation). This allows us to separate electronic and nuclear motion:

$$ \Psi(\mathbf{r}_i, \mathbf{R}_\alpha) = \psi(\mathbf{r}_i; \mathbf{R}_\alpha) \chi(\mathbf{R}_\alpha) $$ 

We can solve the electronic Schrödinger equation by fixing nuclear positions $\\{\mathbf{R}_\alpha\\}$.

## Fundamentals of Density Functional Theory (DFT)

### Hohenberg-Kohn Theorems (1964)

The foundation of DFT is based on two theorems by Hohenberg and Kohn:

#### First Theorem: Uniqueness Theorem

The external potential $V_{\text{ext}}(\mathbf{r})$ (Coulomb potential from nuclei) is uniquely determined by the electron density $n(\mathbf{r})$ (up to a constant).

**Physical meaning** : If we know the electron density, all physical quantities of the system are determined.

#### Second Theorem: Variational Principle

The ground state energy is a functional of the electron density $E[n]$, which takes its minimum value at the true ground state density:

$$ E_0 = \min_{n} E[n] $$ 

**Physical meaning** : Energy can be minimized using electron density as a variational parameter.

This theorem allows us to handle the electron density $n(\mathbf{r})$ (3-dimensional) instead of the many-body wave function $\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)$ ($3N$-dimensional). This represents a dramatic reduction in computational cost.

### Kohn-Sham Equations (1965)

The Hohenberg-Kohn theorem is an existence theorem, but it doesn't show how to actually calculate $E[n]$. Kohn and Sham proposed the idea of mapping the interacting many-electron system to a "non-interacting system" with the same density.

The electron density $n(\mathbf{r})$ is expressed using $N$ Kohn-Sham orbitals $\\{\phi_i(\mathbf{r})\\}$:

$$ n(\mathbf{r}) = \sum_{i=1}^N |\phi_i(\mathbf{r})|^2 $$ 

Each orbital satisfies the Kohn-Sham equation:

$$ \left[ -\frac{\hbar^2}{2m}\nabla^2 + V_{\text{eff}}(\mathbf{r}) \right] \phi_i(\mathbf{r}) = \varepsilon_i \phi_i(\mathbf{r}) $$ 

Here, the effective potential $V_{\text{eff}}$ is:

$$ V_{\text{eff}}(\mathbf{r}) = V_{\text{ext}}(\mathbf{r}) + V_{\text{H}}(\mathbf{r}) + V_{\text{xc}}(\mathbf{r}) $$ 

  * **$V_{\text{ext}}$** : External potential (Coulomb potential from nuclei)
  * **$V_{\text{H}}$** : Hartree potential (classical Coulomb interaction between electrons)
  * **$V_{\text{xc}}$** : Exchange-correlation potential (includes all quantum mechanical effects)

### Exchange-Correlation Energy: The Heart of DFT

The exchange-correlation energy $E_{\text{xc}}[n]$ is the only approximation in DFT. It includes exchange and correlation effects:

  * **Exchange energy** : Reduction of repulsion between electrons with parallel spins due to Pauli exclusion principle
  * **Correlation energy** : Correlation of motion due to electron-electron interactions

The exchange-correlation potential is obtained by functional differentiation of $E_{\text{xc}}[n]$:

$$ V_{\text{xc}}(\mathbf{r}) = \frac{\delta E_{\text{xc}}[n]}{\delta n(\mathbf{r})} $$ 

## Types of Exchange-Correlation Functionals

### LDA (Local Density Approximation)

The simplest approximation. It assumes that the exchange-correlation energy density at each point $\mathbf{r}$ equals that of a uniform electron gas with density $n(\mathbf{r})$:

$$ E_{\text{xc}}^{\text{LDA}}[n] = \int n(\mathbf{r}) \varepsilon_{\text{xc}}(n(\mathbf{r})) d\mathbf{r} $$ 

**Characteristics** :

  * Low computational cost
  * Good accuracy for systems with slowly varying density (metals)
  * Tends to underestimate band gaps
  * Cannot describe weak interactions (van der Waals forces)

### GGA (Generalized Gradient Approximation)

Considers not only the density $n(\mathbf{r})$ but also its gradient $\nabla n(\mathbf{r})$:

$$ E_{\text{xc}}^{\text{GGA}}[n] = \int n(\mathbf{r}) \varepsilon_{\text{xc}}(n(\mathbf{r}), |\nabla n(\mathbf{r})|) d\mathbf{r} $$ 

**Representative GGA functionals** :

  * **PBE** (Perdew-Burke-Ernzerhof): Most widely used for solid-state calculations
  * **PW91** : Predecessor of PBE
  * **BLYP** : Commonly used for molecular systems

**Characteristics** :

  * Improved bond lengths and binding energies compared to LDA
  * High accuracy for molecular and surface systems
  * Computational cost similar to LDA
  * Band gap underestimation still present

### Hybrid Functionals

Methods that mix a portion of Hartree-Fock exchange (exact exchange). Improves band gap accuracy:

$$ E_{\text{xc}}^{\text{hybrid}} = aE_{\text{x}}^{\text{HF}} + (1-a)E_{\text{x}}^{\text{DFT}} + E_{\text{c}}^{\text{DFT}} $$ 

**Representative hybrid functionals** :

  * **PBE0** : Mixes 25% HF exchange ($a=0.25$)
  * **HSE06** : Uses HF exchange only at short range (reduces computational cost)
  * **B3LYP** : Widely used for molecular systems

**Characteristics** :

  * Accurately predicts band gaps
  * Effective for semiconductors and insulators
  * High computational cost (5-10 times that of GGA)

    
    
    ```mermaid
    graph TD
        A[Exchange-Correlation Functionals] --> B[LDA]
        A --> C[GGA]
        A --> D[Hybrid]
        A --> E[Meta-GGA]
    
        B --> B1[Fastest/Low accuracy]
        C --> C1[Standard/High accuracy]
        D --> D1[High accuracy/High cost]
        E --> E1[Research/Highest accuracy]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#d4edda,stroke:#28a745,stroke-width:2px
        style D fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    ```

## Pseudopotentials and PAW Method

### Difficulties of All-Electron Calculations

Core electrons of atoms (1s, 2s, 2p, etc.):

  * Contribute little to chemical bonding
  * Have rapidly oscillating wave functions near the nucleus
  * Require many plane wave basis functions to describe (high computational cost)

### Pseudopotentials

A method that replaces core electron effects with "pseudopotentials" and explicitly treats only valence electrons.

#### Requirements for Pseudopotentials

  * Match all-electron wave functions outside cutoff radius $r_c$
  * Have same scattering properties as all-electron system
  * Norm conservation: charge is conserved

**Types** :

  * **Norm-conserving** : High accuracy but challenges with transferability
  * **Ultrasoft** : Good transferability but complex
  * **PAW method** : Current standard

### PAW (Projector Augmented Wave) Method

The PAW method introduces a transformation that rigorously relates all-electron and pseudo wave functions:

$$ |\psi\rangle = |\tilde{\psi}\rangle + \sum_i (|\phi_i\rangle - |\tilde{\phi}_i\rangle) \langle\tilde{p}_i|\tilde{\psi}\rangle $$ 

  * $|\tilde{\psi}\rangle$: Pseudo wave function (used in calculation)
  * $|\psi\rangle$: All-electron wave function (used for physical quantities)

**Advantages of PAW** :

  * Accuracy close to all-electron calculations
  * Computational cost comparable to pseudopotentials
  * Standard method in VASP

## k-Point Sampling

### Bloch's Theorem and Periodic Boundary Conditions

Crystals have periodic structure, so Bloch's theorem holds:

$$ \psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r}) $$ 

Here, $u_{n\mathbf{k}}(\mathbf{r})$ is a lattice-periodic function. Wave functions are labeled by wave vector $\mathbf{k}$.

### Brillouin Zone and k-Point Mesh

We need to solve Kohn-Sham equations at $\mathbf{k}$-points within the first Brillouin zone. In practice, we sample with a finite number of k-points.

**Monkhorst-Pack mesh** :

Divide Brillouin zone at equal intervals:

$$ \mathbf{k} = \frac{n_1}{N_1}\mathbf{b}_1 + \frac{n_2}{N_2}\mathbf{b}_2 + \frac{n_3}{N_3}\mathbf{b}_3 $$ 

Specify mesh density as $N_1 \times N_2 \times N_3$.

#### Importance of k-Point Convergence Tests

If the k-point mesh is too coarse, energies and properties won't converge. Metals especially require denser meshes (3-4 times denser than semiconductors). Guidelines:

  * Semiconductors: 4×4×4 to 8×8×8
  * Metals: 12×12×12 to 16×16×16
  * Surfaces/2D systems: coarser in k_z direction (e.g., 8×8×1)

## Plane Wave Basis and Cutoff Energy

### Plane Wave Expansion

For periodic systems, Kohn-Sham orbitals can be expanded in plane waves:

$$ \phi_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}} $$ 

$\mathbf{G}$ are reciprocal lattice vectors. In principle this is an infinite sum, but in practice we truncate to finite terms.

### Cutoff Energy $E_{\text{cut}}$

The kinetic energy corresponding to wave number $|\mathbf{k}+\mathbf{G}|$:

$$ E = \frac{\hbar^2}{2m}|\mathbf{k}+\mathbf{G}|^2 $$ 

We use only plane waves with energy below $E_{\text{cut}}$:

$$ \frac{\hbar^2}{2m}|\mathbf{k}+\mathbf{G}|^2 < E_{\text{cut}} $$ 

**Typical values** :

  * PAW: 400-600 eV (element-dependent)
  * Ultrasoft PP: 30-50 Ry (approximately 400-680 eV)

#### Cutoff Energy Convergence Test

Test until energy converges with increasing cutoff. Goals:

  * Total energy: < 1 meV/atom change
  * Lattice constant: < 0.01 Å change

## Learning DFT with Python: Introduction to ASE/Pymatgen

### ASE (Atomic Simulation Environment) Basics

ASE is a Python library that provides unified handling of atomic structure manipulation and DFT calculation setup.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: ASE is a Python library that provides unified handling of at
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from ase import Atoms
    from ase.build import bulk, surface, molecule
    from ase.visualize import view
    import matplotlib.pyplot as plt
    
    # 1. Creating crystal structures
    # Si (diamond structure)
    si = bulk('Si', 'diamond', a=5.43)
    print("Si crystal structure:")
    print(si)
    print(f"Lattice constant: {si.cell[0,0]:.3f} Å")
    print(f"Number of atoms: {len(si)}")
    
    # 2. Creating various crystal structures
    structures = {
        'Al (FCC)': bulk('Al', 'fcc', a=4.05),
        'Fe (BCC)': bulk('Fe', 'bcc', a=2.87),
        'Cu (FCC)': bulk('Cu', 'fcc', a=3.61),
        'GaAs (zincblende)': bulk('GaAs', 'zincblende', a=5.65)
    }
    
    for name, struct in structures.items():
        print(f"\n{name}:")
        print(f"  Lattice constant: {struct.cell[0,0]:.3f} Å")
        print(f"  Number of atoms: {len(struct)}")
        print(f"  Chemical formula: {struct.get_chemical_formula()}")
    
    # 3. Creating surface structures
    # Si(111) surface
    si_surface = surface('Si', (1,1,1), layers=4, vacuum=10.0)
    print(f"\nSi(111) surface:")
    print(f"  Number of atoms: {len(si_surface)}")
    print(f"  Cell size: {si_surface.cell.lengths()}")
    
    # 4. Creating molecular structures
    h2o = molecule('H2O')
    print(f"\nH2O molecule:")
    print(f"  Number of atoms: {len(h2o)}")
    for i, atom in enumerate(h2o):
        print(f"  {atom.symbol}: {atom.position}")
    

### Obtaining Detailed Crystal Structure Information
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Obtaining Detailed Crystal Structure Information
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from ase.build import bulk
    import numpy as np
    
    # Detailed analysis of Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # Cell information
    print("Cell matrix:")
    print(si.cell)
    print(f"\nCell volume: {si.get_volume():.3f} Å³")
    
    # Atomic positions (fractional coordinates)
    print("\nFractional coordinates of atoms:")
    scaled_pos = si.get_scaled_positions()
    for i, pos in enumerate(scaled_pos):
        print(f"  Atom {i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # Atomic positions (Cartesian coordinates)
    print("\nCartesian coordinates of atoms [Å]:")
    for i, pos in enumerate(si.positions):
        print(f"  Atom {i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # Calculate interatomic distances
    from ase.neighborlist import NeighborList, natural_cutoffs
    
    cutoffs = natural_cutoffs(si)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(si)
    
    print("\nNearest-neighbor distances:")
    for i in range(len(si)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            dist = si.get_distance(i, j, mic=True)
            print(f"  Atom {i} - Atom {j}: {dist:.4f} Å")
        break  # Display only first atom
    

### Manipulating Crystal Structures with Pymatgen
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Manipulating Crystal Structures with Pymatgen
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # 1. Create crystal structure from lattice constants
    # Si (diamond structure)
    lattice = Lattice.cubic(5.43)
    si_struct = Structure(
        lattice,
        ["Si", "Si"],
        [[0.00, 0.00, 0.00],
         [0.25, 0.25, 0.25]]
    )
    
    print("Si crystal structure (Pymatgen):")
    print(si_struct)
    
    # 2. Symmetry analysis
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    analyzer = SpacegroupAnalyzer(si_struct)
    print(f"\nSpace group: {analyzer.get_space_group_symbol()}")
    print(f"Space group number: {analyzer.get_space_group_number()}")
    print(f"Point group: {analyzer.get_point_group_symbol()}")
    
    # 3. Convert to primitive cell
    primitive = analyzer.get_primitive_standard_structure()
    print(f"\nNumber of atoms in primitive cell: {len(primitive)}")
    print(f"Number of atoms in conventional cell: {len(si_struct)}")
    
    # 4. Compound material: GaAs
    gaas_struct = Structure(
        Lattice.cubic(5.65),
        ["Ga", "As"],
        [[0.00, 0.00, 0.00],
         [0.25, 0.25, 0.25]]
    )
    print("\nGaAs crystal structure:")
    print(gaas_struct)
    print(f"Space group: {SpacegroupAnalyzer(gaas_struct).get_space_group_symbol()}")
    

### Converting Between ASE and Pymatgen
    
    
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    
    # ASE → Pymatgen
    si_ase = bulk('Si', 'diamond', a=5.43)
    adaptor = AseAtomsAdaptor()
    si_pmg = adaptor.get_structure(si_ase)
    
    print("Conversion from ASE to Pymatgen:")
    print(si_pmg)
    
    # Pymatgen → ASE
    si_ase_back = adaptor.get_atoms(si_pmg)
    print("\nConversion from Pymatgen to ASE:")
    print(si_ase_back)
    
    # Leverage advantages of both
    # ASE: Convenient for calculation setup
    # Pymatgen: Powerful for symmetry analysis, Materials Project integration
    

## Creating VASP Input Files

### VASP's Four Input Files

VASP requires four main input files:

File name | Contents | How to create  
---|---|---  
**INCAR** | Calculation parameters (functional, k-points, convergence criteria, etc.) | Manual or Pymatgen  
**POSCAR** | Atomic structure (lattice constants, atomic positions) | Auto-generated from ASE/Pymatgen  
**KPOINTS** | k-point mesh settings | Manual or Pymatgen  
**POTCAR** | Pseudopotentials (PAW) | Copy from VASP distribution  
  
### POSCAR (Atomic Structure File)
    
    
    from ase.build import bulk
    from ase.io import write
    
    # Create POSCAR file for Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # Expand to 2×2×2 supercell
    si_supercell = si.repeat((2, 2, 2))
    
    # Write to POSCAR file
    write('POSCAR', si_supercell, format='vasp')
    
    print("Created POSCAR file")
    print(f"Number of atoms: {len(si_supercell)}")
    
    # Display contents of POSCAR file
    with open('POSCAR', 'r') as f:
        print("\nContents of POSCAR:")
        print(f.read())
    

Format of generated POSCAR file:
    
    
    Si16
    1.0
       10.8600000000    0.0000000000    0.0000000000
        0.0000000000   10.8600000000    0.0000000000
        0.0000000000    0.0000000000   10.8600000000
    Si
    16
    Direct
      0.0000000000  0.0000000000  0.0000000000
      0.1250000000  0.1250000000  0.1250000000
      ...
    

### Creating INCAR Files
    
    
    # Generate INCAR file template
    
    def create_incar(calculation_type='scf', system_name='Si',
                     functional='PBE', encut=400, ismear=0, sigma=0.05):
        """
        Generate VASP INCAR file
    
        Parameters:
        -----------
        calculation_type : str
            'scf' (single-point), 'relax' (structure optimization), 'band' (band structure)
        functional : str
            'PBE', 'LDA', 'PBE0', 'HSE06'
        encut : float
            Cutoff energy [eV]
        ismear : int
            Smearing method (0: Gaussian, 1: Methfessel-Paxton, -5: tetrahedron)
        sigma : float
            Smearing width [eV]
        """
    
        incar_content = f"""SYSTEM = {system_name}
    
    # Electronic structure
    ENCUT = {encut}         # Cutoff energy [eV]
    PREC = Accurate         # Precision (Normal, Accurate, High)
    LREAL = Auto            # Real-space projection (Auto recommended)
    
    # Exchange-correlation
    GGA = PE                # PBE functional (remove for LDA)
    
    # SCF convergence
    EDIFF = 1E-6            # Electronic convergence criterion [eV]
    NELM = 100              # Maximum SCF iterations
    
    # Smearing (different settings for metals/semiconductors)
    ISMEAR = {ismear}       # Smearing method
    SIGMA = {sigma}         # Smearing width [eV]
    
    # Parallelization
    NCORE = 4               # Core parallelization (system-dependent)
    """
    
        # For structure relaxation
        if calculation_type == 'relax':
            incar_content += """
    # Structure relaxation
    IBRION = 2              # Ion relaxation algorithm (2: CG, 1: RMM-DIIS)
    NSW = 100               # Maximum ionic steps
    ISIF = 3                # Optimize cell and ionic positions
    EDIFFG = -0.01          # Force convergence criterion [eV/Å]
    """
    
        # For band calculation
        elif calculation_type == 'band':
            incar_content += """
    # Band structure calculation
    ICHARG = 11             # Read charge density
    LORBIT = 11             # Calculate projected DOS
    """
    
        return incar_content
    
    # Generate INCAR for SCF calculation
    incar_scf = create_incar(calculation_type='scf', system_name='Si bulk')
    with open('INCAR', 'w') as f:
        f.write(incar_scf)
    
    print("Created INCAR file:")
    print(incar_scf)
    

### Creating KPOINTS Files
    
    
    # Generate KPOINTS file
    
    def create_kpoints(kpts=(8, 8, 8), shift=(0, 0, 0), mode='Monkhorst-Pack'):
        """
        Generate VASP KPOINTS file
    
        Parameters:
        -----------
        kpts : tuple
            k-point mesh (nx, ny, nz)
        shift : tuple
            Mesh shift (usually (0,0,0))
        mode : str
            'Monkhorst-Pack' or 'Gamma'
        """
    
        kpoints_content = f"""Automatic mesh
    0
    {mode}
    {kpts[0]} {kpts[1]} {kpts[2]}
    {shift[0]} {shift[1]} {shift[2]}
    """
        return kpoints_content
    
    # 8×8×8 Monkhorst-Pack mesh
    kpoints_dense = create_kpoints(kpts=(8, 8, 8))
    with open('KPOINTS', 'w') as f:
        f.write(kpoints_dense)
    
    print("Created KPOINTS file:")
    print(kpoints_dense)
    
    # Gamma-centered mesh (recommended for metals)
    kpoints_gamma = create_kpoints(kpts=(12, 12, 12), mode='Gamma')
    print("\nGamma-centered mesh:")
    print(kpoints_gamma)
    

### Batch Creation of Input Files Using Pymatgen
    
    
    from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet
    
    # 1. Create Si structure with Pymatgen
    si_struct = Structure.from_file('POSCAR')  # Or create directly
    
    # 2. Generate complete set of input files with Materials Project standard settings
    # (Standard settings widely used in research community)
    mp_set = MPRelaxSet(si_struct)
    
    # 3. Write files
    mp_set.write_input('vasp_calc/')  # Create all files in vasp_calc/ directory
    
    print("Created complete set of VASP input files in vasp_calc/")
    print("Included files: INCAR, POSCAR, KPOINTS, POTCAR (copy needed)")
    
    # 4. Customize individually
    custom_incar = Incar({
        'SYSTEM': 'Si bulk',
        'ENCUT': 520,
        'ISMEAR': 0,
        'SIGMA': 0.05,
        'EDIFF': 1e-6,
        'PREC': 'Accurate'
    })
    
    custom_kpoints = Kpoints.gamma_automatic(kpts=(10, 10, 10))
    
    # Write
    custom_incar.write_file('vasp_calc/INCAR')
    custom_kpoints.write_file('vasp_calc/KPOINTS')
    

## Convergence Tests for k-Points and Cutoff Energy

### Convergence Test Strategy
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Convergence Test Strategy
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convergence test simulation
    # (Simulating actual VASP calculation results)
    
    # 1. k-point convergence test
    k_values = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    # Converging total energy (dummy data)
    energy_k = -5.4 + 0.1 * np.exp(-k_values/4) + np.random.normal(0, 0.001, len(k_values))
    
    # 2. Cutoff energy convergence test
    encut_values = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600])
    # Converging total energy (dummy data)
    energy_encut = -5.4 - 0.05 * np.exp(-(encut_values-200)/100) + np.random.normal(0, 0.001, len(encut_values))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # k-point convergence
    ax1.plot(k_values, energy_k, 'o-', markersize=8, linewidth=2, color='#f093fb')
    ax1.axhline(y=energy_k[-1], color='red', linestyle='--', label='Converged value')
    ax1.fill_between(k_values, energy_k[-1]-0.001, energy_k[-1]+0.001,
                      alpha=0.2, color='red', label='±1 meV range')
    ax1.set_xlabel('k-point mesh (k×k×k)', fontsize=12)
    ax1.set_ylabel('Total energy [eV/atom]', fontsize=12)
    ax1.set_title('k-point Convergence Test', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cutoff energy convergence
    ax2.plot(encut_values, energy_encut, 's-', markersize=8, linewidth=2, color='#f5576c')
    ax2.axhline(y=energy_encut[-1], color='red', linestyle='--', label='Converged value')
    ax2.fill_between(encut_values, energy_encut[-1]-0.001, energy_encut[-1]+0.001,
                      alpha=0.2, color='red', label='±1 meV range')
    ax2.set_xlabel('Cutoff energy [eV]', fontsize=12)
    ax2.set_ylabel('Total energy [eV/atom]', fontsize=12)
    ax2.set_title('Cutoff Energy Convergence Test', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_tests.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Convergence determination
    print("=== Convergence Test Results ===")
    print(f"\nk-point convergence:")
    for i in range(1, len(k_values)):
        diff = abs(energy_k[i] - energy_k[i-1]) * 1000  # meV
        status = "✓" if diff < 1.0 else "✗"
        print(f"  k={k_values[i]:2d}: ΔE = {diff:.3f} meV {status}")
    
    print(f"\nCutoff energy convergence:")
    for i in range(1, len(encut_values)):
        diff = abs(energy_encut[i] - energy_encut[i-1]) * 1000  # meV
        status = "✓" if diff < 1.0 else "✗"
        print(f"  ENCUT={encut_values[i]:3d} eV: ΔE = {diff:.3f} meV {status}")
    

### Practical Convergence Test Script
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    
    import os
    import numpy as np
    from ase.build import bulk
    from ase.io import write
    
    def run_convergence_test(structure, test_type='kpoints',
                              k_range=None, encut_range=None):
        """
        Prepare directory structure and files for convergence tests
    
        Parameters:
        -----------
        structure : ase.Atoms
            Crystal structure to test
        test_type : str
            'kpoints' or 'encut'
        k_range : list
            List of k-point meshes to test (e.g., [4, 6, 8, 10, 12])
        encut_range : list
            List of cutoff energies to test [eV]
        """
    
        if test_type == 'kpoints' and k_range is not None:
            print("=== k-point Convergence Test Preparation ===")
            for k in k_range:
                dirname = f'ktest_{k}x{k}x{k}'
                os.makedirs(dirname, exist_ok=True)
    
                # Create POSCAR
                write(f'{dirname}/POSCAR', structure, format='vasp')
    
                # Create KPOINTS
                with open(f'{dirname}/KPOINTS', 'w') as f:
                    f.write(f"""Automatic mesh
    0
    Monkhorst-Pack
    {k} {k} {k}
    0 0 0
    """)
    
                # Create INCAR (common settings)
                with open(f'{dirname}/INCAR', 'w') as f:
                    f.write("""SYSTEM = k-point convergence test
    ENCUT = 400
    ISMEAR = 0
    SIGMA = 0.05
    EDIFF = 1E-6
    PREC = Accurate
    """)
    
                print(f"  {dirname}/ created")
    
        elif test_type == 'encut' and encut_range is not None:
            print("=== Cutoff Energy Convergence Test Preparation ===")
            for encut in encut_range:
                dirname = f'encut_{encut}'
                os.makedirs(dirname, exist_ok=True)
    
                write(f'{dirname}/POSCAR', structure, format='vasp')
    
                with open(f'{dirname}/KPOINTS', 'w') as f:
                    f.write("""Automatic mesh
    0
    Monkhorst-Pack
    8 8 8
    0 0 0
    """)
    
                with open(f'{dirname}/INCAR', 'w') as f:
                    f.write(f"""SYSTEM = ENCUT convergence test
    ENCUT = {encut}
    ISMEAR = 0
    SIGMA = 0.05
    EDIFF = 1E-6
    PREC = Accurate
    """)
    
                print(f"  {dirname}/ created")
    
    # Example usage
    si = bulk('Si', 'diamond', a=5.43)
    
    # Create directories for k-point convergence test
    run_convergence_test(si, test_type='kpoints',
                          k_range=[4, 6, 8, 10, 12, 14])
    
    # Create directories for cutoff convergence test
    run_convergence_test(si, test_type='encut',
                          encut_range=[300, 350, 400, 450, 500, 550, 600])
    
    print("\nAll test directories created")
    print("Next step: Run VASP in each directory")
    

## DFT Calculation Workflow
    
    
    ```mermaid
    flowchart TD
        A[Prepare crystal structure] --> B[Create input files]
        B --> C{Convergence testsk-points・ENCUT}
        C -->|Not converged| B
        C -->|Converged| D[SCF calculation]
        D --> E[Structure optimization]
        E --> F{Structure changesmall?}
        F -->|No| E
        F -->|Yes| G[Property calculations]
        G --> H[Band structure]
        G --> I[Density of states]
        G --> J[Charge density]
        H --> K[Results analysis]
        I --> K
        J --> K
        K --> L[Papers/Reports]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#d4edda,stroke:#28a745,stroke-width:2px
        style G fill:#fff3cd,stroke:#ffc107,stroke-width:2px
        style L fill:#d4edda,stroke:#28a745,stroke-width:2px
    ```

### Standard DFT Calculation Procedure
    
    
    # Standard DFT calculation workflow (pseudocode)
    
    def dft_workflow(material_name, structure):
        """
        Standard DFT calculation workflow
        """
    
        print(f"=== DFT Calculation Workflow for {material_name} ===\n")
    
        # Step 1: Convergence tests
        print("Step 1: Convergence Tests")
        print("  1-1. k-point mesh convergence test")
        print("       → Optimal k-points: 8×8×8 determined")
        print("  1-2. Cutoff energy convergence test")
        print("       → Optimal ENCUT: 450 eV determined")
    
        # Step 2: Structure optimization
        print("\nStep 2: Structure Optimization (IBRION=2, ISIF=3)")
        print("  - Initial structure: experimental values")
        print("  - After optimization:")
        print("    Lattice constant: 5.43 → 5.47 Å (+0.7%, PBE overestimation)")
        print("    Atomic positions: no change (high symmetry)")
    
        # Step 3: Static calculation (high accuracy)
        print("\nStep 3: Static Calculation (single-point, NSW=0)")
        print("  - High-accuracy SCF with optimized structure")
        print("  - Save charge density and wave functions")
    
        # Step 4: Band structure calculation
        print("\nStep 4: Band Structure Calculation (ICHARG=11)")
        print("  - High-symmetry path: Γ-X-W-K-Γ-L")
        print("  - Indirect band gap: 0.65 eV (underestimates experimental 1.12 eV)")
    
        # Step 5: Density of states calculation
        print("\nStep 5: Density of States Calculation (LORBIT=11)")
        print("  - Dense k-point mesh: 16×16×16")
        print("  - Projected DOS (atom/orbital resolved)")
    
        # Step 6: Results analysis
        print("\nStep 6: Results Analysis")
        print("  - Plot band diagram")
        print("  - Plot DOS")
        print("  - Compare with experimental values")
    
        return {
            'optimal_k': (8, 8, 8),
            'optimal_encut': 450,
            'lattice_constant': 5.47,
            'band_gap': 0.65
        }
    
    # Example execution
    si = bulk('Si', 'diamond', a=5.43)
    results = dft_workflow('Silicon', si)
    
    print(f"\n=== Calculation Results Summary ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
    

## Summary

### What We Learned in This Chapter

#### Theoretical Understanding

  * DFT is a theory that solves the many-body Schrödinger equation using electron density
  * Hohenberg-Kohn theorem: electron density determines all physical quantities
  * Kohn-Sham equations: makes calculation possible through mapping to non-interacting system
  * Exchange-correlation functionals (LDA, GGA, hybrid) are the only approximation

#### Practical Skills

  * Can create and manipulate crystal structures using ASE/Pymatgen
  * Can create VASP input files (INCAR, POSCAR, KPOINTS)
  * Can perform convergence tests for k-point mesh and cutoff energy
  * Understood standard DFT calculation workflow

#### Preparation for Next Chapter

  * In Chapter 4, we will calculate electrical and magnetic properties from DFT results
  * We will learn calculation methods for specific properties like electrical conductivity, Hall effect, and magnetization

## Exercises

#### Exercise 1: Basic Theory (Easy)

**Problem** : Determine whether the following statements are true or false, and correct any errors.

  1. In DFT, the electron density $n(\mathbf{r})$ is a 3-dimensional function, making it easier to handle than the many-body wave function.
  2. The Kohn-Sham equation exactly solves the interacting many-electron system.
  3. GGA functionals always provide higher accuracy than LDA because they consider density gradients.
  4. The PAW method is a type of pseudopotential method that achieves accuracy close to all-electron calculations.

**Key points for answer** :

  1. True. Dramatic reduction from $3N$ dimensions → 3 dimensions.
  2. False. Kohn-Sham equations are a "mapping to non-interacting system". Exchange-correlation functional $E_{\text{xc}}$ is an approximation.
  3. False. GGA is often superior, but LDA also works well for systems with nearly uniform density (simple metals).
  4. True. PAW has a transformation that restores the all-electron wave function.

#### Exercise 2: Functional Selection (Medium)

**Problem** : Select the optimal exchange-correlation functional for the following systems and explain your reasoning.

  1. Band gap calculation for Si bulk (semiconductor)
  2. Lattice constant optimization for Cu metal
  3. Electronic structure calculation for TiO₂ (titanium oxide)
  4. Binding energy calculation for organic molecules

**Recommended answers** :

  1. HSE06 (hybrid): GGA underestimates band gaps by about 50%.
  2. PBE (GGA): Metals have smooth electron density, GGA provides sufficient accuracy with low computational cost.
  3. PBE+U or HSE06: Need to account for strong correlation effects in Ti d-orbitals.
  4. B3LYP (hybrid): Extensively validated for molecular systems. PBE also acceptable.

#### Exercise 3: Python Coding (Medium)

**Problem** : Write code to create the following crystal structures with ASE and save them in POSCAR format.

  1. GaN (wurtzite structure, a=3.19 Å, c=5.19 Å)
  2. Fe (BCC structure, a=2.87 Å)
  3. Al₂O₃ (corundum structure, download from Materials Project)

**Hints** :

  * GaN is wurtzite structure
  * Fe is BCC (body-centered cubic)
  * Use `pymatgen.ext.matproj` to access Materials Project

#### Exercise 4: Convergence Test Analysis (Medium)

**Problem** : Analyze the following k-point convergence test results and recommend the optimal k-point mesh.

k-point mesh | Total energy [eV/atom] | Calculation time [min]  
---|---|---  
4×4×4| -5.3421| 2  
6×6×6| -5.3998| 5  
8×8×8| -5.4125| 12  
10×10×10| -5.4138| 25  
12×12×12| -5.4141| 45  
14×14×14| -5.4142| 75  
  
**Discussion points** :

  * Use 1 meV/atom (0.001 eV/atom) as convergence criterion
  * Consider balance between computational cost and accuracy
  * Judgment changes depending on whether metal or semiconductor

**Recommended answer** :

Recommend 8×8×8 to 10×10×10. Reasoning:

  * 8×8×8: 1.27 meV/atom improvement from 6×6×6 (near convergence)
  * 10×10×10: 0.13 meV/atom improvement from 8×8×8 (nearly converged)
  * 12×12×12 and above: no improvement commensurate with computational cost
  * Recommend 8×8×8 for semiconductors, 10×10×10 for metals

#### Exercise 5: INCAR Settings Optimization (Hard)

**Problem** : Set optimal INCAR parameters for the following calculation purposes.

  1. Lattice constant optimization for metallic Al
  2. Band gap calculation for semiconductor GaAs (HSE06)
  3. Magnetic moment calculation for ferromagnetic Fe

**Recommended answers** :

1\. Lattice constant optimization for metallic Al:
    
    
    SYSTEM = Al lattice optimization
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    ISMEAR = 1          # Methfessel-Paxton (optimal for metals)
    SIGMA = 0.2         # Larger smearing
    IBRION = 2          # CG method
    ISIF = 3            # Optimize cell shape too
    EDIFF = 1E-6
    EDIFFG = -0.01
    NSW = 50
    

2\. Band gap calculation for semiconductor GaAs (HSE06):
    
    
    SYSTEM = GaAs band gap (HSE06)
    ENCUT = 450
    PREC = Accurate
    LHFCALC = .TRUE.    # Enable hybrid functional
    HFSCREEN = 0.2      # HSE06 screening parameter
    ISMEAR = 0          # Gaussian (semiconductors)
    SIGMA = 0.05        # Smaller smearing
    EDIFF = 1E-7        # High accuracy convergence
    ALGO = Damped       # Recommended for HSE06
    TIME = 0.4
    

3\. Magnetic moment calculation for ferromagnetic Fe:
    
    
    SYSTEM = Fe magnetic moment
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    ISMEAR = 1
    SIGMA = 0.2
    ISPIN = 2           # Spin-polarized calculation
    MAGMOM = 2.0        # Initial magnetic moment (Fe: approximately 2.2 μB)
    LORBIT = 11         # Magnetic moment projection
    EDIFF = 1E-6
    

#### Exercise 6: Practical Challenge (Hard)

**Problem** : Prepare for actual DFT calculations following these steps.

  1. Create diamond structure Si crystal (2×2×2 supercell) using ASE
  2. Generate VASP input files (INCAR, POSCAR, KPOINTS)
  3. Create directory structure for k-point convergence tests (k=4,6,8,10,12)
  4. Verify that configuration files in each directory are correct and ready for calculation

**Evaluation criteria** :

  * Is POSCAR file in correct VASP format
  * Are INCAR settings appropriate for semiconductor Si (ISMEAR=0, etc.)
  * Is k-point mesh correctly set in each directory
  * Is directory structure organized and traceable

## References

  1. Hohenberg, P., & Kohn, W. (1964). "Inhomogeneous Electron Gas". Physical Review, 136(3B), B864.
  2. Kohn, W., & Sham, L. J. (1965). "Self-Consistent Equations Including Exchange and Correlation Effects". Physical Review, 140(4A), A1133.
  3. Perdew, J. P., Burke, K., & Ernzerhof, M. (1996). "Generalized Gradient Approximation Made Simple". Physical Review Letters, 77, 3865.
  4. Blöchl, P. E. (1994). "Projector augmented-wave method". Physical Review B, 50, 17953.
  5. Sholl, D., & Steckel, J. A. (2011). "Density Functional Theory: A Practical Introduction". Wiley.
  6. ASE documentation: https://wiki.fysik.dtu.dk/ase/
  7. Pymatgen documentation: https://pymatgen.org/
  8. VASP manual: https://www.vasp.at/wiki/index.php/The_VASP_Manual

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
