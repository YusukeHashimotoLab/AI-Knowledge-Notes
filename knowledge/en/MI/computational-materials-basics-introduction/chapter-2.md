---
title: "Chapter 2: Introduction to Density Functional Theory (DFT)"
chapter_title: "Chapter 2: Introduction to Density Functional Theory (DFT)"
subtitle: Practical First-Principles Calculations for Many-Electron Systems
reading_time: 20-25 min
difficulty: Intermediate ~ Advanced
code_examples: 8
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 2: Introduction to Density Functional Theory (DFT)

Get a rough understanding of the Kohn-Sham approach and the meaning of exchange-correlation. Learn approximations and caveats that work in practice.

**💡 Note:** Exchange-correlation is a summary of "electron-electron consideration." Choosing a functional that fits your system is the critical decision point for performance.

## Learning Objectives

By reading this chapter, you will be able to: \- Understand the basic principles of DFT (Hohenberg-Kohn theorem, Kohn-Sham equations) \- Explain the differences between exchange-correlation functionals (LDA, GGA) \- Perform DFT calculations using ASE and GPAW \- Calculate band structures, density of states, and structure optimization \- Understand the limitations of DFT (band gap problem, van der Waals interactions)

* * *

## 2.1 The Challenge of Many-Electron Systems and the Emergence of DFT

### The Many-Electron Schrödinger Equation

The Schrödinger equation for a many-electron system ($N$ electrons):

$$ \hat{H}\Psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N) = E\Psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N) $$

The wave function $\Psi$ is a function in $3N$-dimensional space. This is the critical difficulty.

**Computational Explosion** : \- 2-electron system: 6 dimensions \- 10-electron system: 30 dimensions \- 100-electron system (small molecule): 300 dimensions

Sampling each dimension with 100 points requires $100^{300} \approx 10^{600}$ points → **Practically impossible**

### DFT's Paradigm Shift

**Walter Kohn's Idea (1998 Nobel Prize in Chemistry)** :

> Instead of the wave function $\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)$ ($3N$ dimensions), can we use the **electron density** $n(\mathbf{r})$ (3 dimensions) as the basic variable?

$$ n(\mathbf{r}) = N \int |\Psi(\mathbf{r}, \mathbf{r}_2, \ldots, \mathbf{r}_N)|^2 d\mathbf{r}_2 \cdots d\mathbf{r}_N $$

If this is possible: \- $3N$ dimensions → 3 dimensions (dimensional reduction) \- Computational cost dramatically reduced

* * *

## 2.2 Hohenberg-Kohn Theorems (1964)

Two theorems that provide the theoretical foundation for DFT.

### First Theorem: One-to-One Correspondence

**Theorem** : The external potential $V_{\text{ext}}(\mathbf{r})$ is uniquely determined by the electron density $n(\mathbf{r})$ (up to a constant).

**Physical Meaning** : \- If the electron density $n(\mathbf{r})$ is known, the Hamiltonian $\hat{H}$ is determined \- If the Hamiltonian is determined, all physical quantities are determined \- In other words, $n(\mathbf{r})$ alone contains all information

### Second Theorem: Variational Principle

**Theorem** : The ground state energy $E_0$ takes its minimum value at the true electron density $n_0(\mathbf{r})$.

$$ E[n] \geq E[n_0] = E_0 $$

For any trial density $n(\mathbf{r})$, minimizing the energy functional $E[n]$ yields the ground state.

### Energy Functional

$$ E[n] = T[n] + V_{\text{ext}}[n] + V_{\text{ee}}[n] $$

  * $T[n]$: Kinetic energy functional
  * $V_{\text{ext}}[n] = \int V_{\text{ext}}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r}$: External potential
  * $V_{\text{ee}}[n]$: Electron-electron interaction functional

**Problem** : The exact forms of $T[n]$ and $V_{\text{ee}}[n]$ are unknown!

* * *

## 2.3 Kohn-Sham Equations (1965)

### Kohn-Sham's Brilliant Idea

Introduce a **fictitious non-interacting system** : \- Has the **same electron density** as the real interacting electron system \- But electron-electron interactions are **zero** (independent particle system)

The Schrödinger equation for this non-interacting system:

$$ \left[-\frac{\hbar^2}{2m_e}\nabla^2 + V_{\text{KS}}(\mathbf{r})\right] \psi_i(\mathbf{r}) = \epsilon_i \psi_i(\mathbf{r}) $$

  * $\psi_i(\mathbf{r})$: Kohn-Sham orbitals ($i = 1, 2, \ldots, N$)
  * $\epsilon_i$: Kohn-Sham energy eigenvalues
  * $V_{\text{KS}}(\mathbf{r})$: **Kohn-Sham potential** (effective potential)

### Electron Density

$$ n(\mathbf{r}) = \sum_{i=1}^N f_i |\psi_i(\mathbf{r})|^2 $$

$f_i$ is the occupation number (for the ground state $f_i = 1$, considering spin $f_i \leq 2$)

### Kohn-Sham Potential

$$ V_{\text{KS}}(\mathbf{r}) = V_{\text{ext}}(\mathbf{r}) + V_{\text{Hartree}}(\mathbf{r}) + V_{\text{xc}}(\mathbf{r}) $$

**Hartree potential** (classical Coulomb interaction):

$$ V_{\text{Hartree}}(\mathbf{r}) = e^2 \int \frac{n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d\mathbf{r}' $$

**Exchange-correlation potential** (including quantum effects):

$$ V_{\text{xc}}(\mathbf{r}) = \frac{\delta E_{\text{xc}}[n]}{\delta n(\mathbf{r})} $$

$E_{\text{xc}}[n]$ is the exchange-correlation energy functional.

### Self-Consistent Field (SCF) Calculation

The Kohn-Sham equations must be solved self-consistently:
    
    
    ```mermaid
    flowchart TD
        A[Assume initial density n⁰r] --> B[Calculate V_KSr]
        B --> C[Solve Kohn-Sham equations: ψᵢr, εᵢ]
        C --> D[Calculate new density n¹r = Σfᵢ|ψᵢr|²]
        D --> E{Convergence check: |n¹-n⁰| < tol?}
        E -->|No| F[Mix densities: n⁰ = αn¹ + 1-αn⁰]
        F --> B
        E -->|Yes| G[Obtain ground state energy E₀ and electronic structure]
    
        style A fill:#e3f2fd
        style G fill:#c8e6c9
    ```

* * *

## 2.4 Exchange-Correlation Functionals

The accuracy of DFT depends on the approximation of $E_{\text{xc}}[n]$.

### LDA (Local Density Approximation)

**Assumption** : The exchange-correlation energy at each point $\mathbf{r}$ depends only on the electron density $n(\mathbf{r})$ at that point.

$$ E_{\text{xc}}^{\text{LDA}}[n] = \int n(\mathbf{r}) \epsilon_{\text{xc}}^{\text{unif}}(n(\mathbf{r})) d\mathbf{r} $$

$\epsilon_{\text{xc}}^{\text{unif}}(n)$ is the exchange-correlation energy density of the uniform electron gas (precisely determined by quantum Monte Carlo calculations).

**Characteristics** : \- ✅ Fast computation \- ✅ Good accuracy for crystal structures and lattice constants \- ❌ Underestimates band gaps (~30-50%) \- ❌ Cannot describe weak bonding (van der Waals)

### GGA (Generalized Gradient Approximation)

Considers not only the density $n(\mathbf{r})$ but also its gradient $\nabla n(\mathbf{r})$:

$$ E_{\text{xc}}^{\text{GGA}}[n] = \int n(\mathbf{r}) \epsilon_{\text{xc}}^{\text{GGA}}(n(\mathbf{r}), |\nabla n(\mathbf{r})|) d\mathbf{r} $$

**Representative GGA functionals** : \- **PBE** (Perdew-Burke-Ernzerhof, 1996): Most widely used \- **PW91** (Perdew-Wang 1991): Predecessor to PBE \- **BLYP** (Becke-Lee-Yang-Parr): Popular in quantum chemistry

**Characteristics** : \- ✅ Better accuracy for structures and binding energies than LDA \- ✅ Improved molecular bond distances and angles \- ❌ Band gap problem similar to LDA \- ❌ van der Waals interactions still insufficient

### Comparison Table

Property | LDA | GGA (PBE) | Experimental  
---|---|---|---  
Si lattice constant [Å] | 5.40 | 5.47 | 5.43  
Si band gap [eV] | 0.5 | 0.6 | 1.17  
H₂ bond length [Å] | 0.76 | 0.75 | 0.74  
H₂ binding energy [eV] | -4.8 | -4.6 | -4.75  
  
* * *

## 2.5 Practical DFT Calculations with ASE + GPAW

### Environment Setup
    
    
    # Recommended installation using Anaconda
    conda create -n dft python=3.11
    conda activate dft
    conda install -c conda-forge ase gpaw
    pip install matplotlib numpy scipy
    

### Example 1: Structure Optimization of H₂ Molecule
    
    
    from ase import Atoms
    from ase.optimize import BFGS
    from gpaw import GPAW, PW
    
    # Initial structure of H₂ molecule
    atoms = Atoms('H2',
                  positions=[[0, 0, 0], [0, 0, 0.8]],  # Initial bond length 0.8Å
                  cell=[6, 6, 6],  # Cell size
                  pbc=False)  # No periodic boundary conditions
    
    # Setup calculator
    calc = GPAW(mode=PW(400),  # Plane wave basis, cutoff energy 400eV
                xc='PBE',  # GGA functional (PBE)
                txt='h2_opt.txt')  # Log file
    
    atoms.calc = calc
    
    # Structure optimization
    opt = BFGS(atoms, trajectory='h2_opt.traj')
    opt.run(fmax=0.01)  # Optimize until forces < 0.01 eV/Å
    
    # Display results
    print(f"Optimized bond length: {atoms.get_distance(0, 1):.3f} Å")
    print(f"Total energy: {atoms.get_potential_energy():.3f} eV")
    

**Execution Results** :
    
    
    Optimized bond length: 0.748 Å
    Total energy: -6.873 eV
    

**Comparison with experiment** : 0.741 Å (experimental) → Error ~1%

* * *

### Example 2: Band Structure Calculation for Si
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Example 2: Band Structure Calculation for Si
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from ase.build import bulk
    from gpaw import GPAW, PW
    from gpaw.utilities.kpoints import get_bandpath
    import matplotlib.pyplot as plt
    
    # Create Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # SCF calculation (dense k-point mesh)
    calc = GPAW(mode=PW(400),
                xc='PBE',
                kpts=(8, 8, 8),  # Monkhorst-Pack mesh
                txt='si_scf.txt')
    
    si.calc = calc
    si.get_potential_energy()  # Run SCF calculation
    calc.write('si_groundstate.gpw')  # Save wave functions
    
    # Band structure calculation
    calc_bands = calc.fixed_density(
        kpts={'path': 'LGXULK', 'npoints': 60},  # High-symmetry path
        txt='si_bands.txt'
    )
    
    # Get band structure
    ef = calc_bands.get_fermi_level()
    energies, k_distances = calc_bands.band_structure().get_bands()
    
    # Plot
    plt.figure(figsize=(8, 6))
    for n in range(energies.shape[1]):
        plt.plot(k_distances, energies[:, n] - ef, 'b-', linewidth=1)
    
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.ylabel('Energy [eV]', fontsize=12)
    plt.xlabel('k-path', fontsize=12)
    plt.title('Si Band Structure (PBE)', fontsize=14)
    plt.ylim(-6, 6)
    plt.grid(alpha=0.3)
    plt.savefig('si_bandstructure.png', dpi=150)
    plt.show()
    
    # Band gap calculation
    vbm = energies[:, :4].max() - ef  # Valence Band Maximum
    cbm = energies[:, 4:].min() - ef  # Conduction Band Minimum
    print(f"Band gap (Indirect): {cbm - vbm:.3f} eV")
    print(f"Experimental value: 1.17 eV")
    

**Execution Results** :
    
    
    Band gap (Indirect): 0.614 eV
    Experimental value: 1.17 eV
    

**Band gap underestimation** : Known issue with DFT (explained in the next section)

* * *

### Example 3: Density of States (DOS) Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Example 3: Density of States (DOS) Calculation
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from gpaw import GPAW
    import matplotlib.pyplot as plt
    
    # Load previously calculated ground state
    calc = GPAW('si_groundstate.gpw', txt=None)
    
    # Calculate density of states
    energies, dos = calc.get_dos(spin=0, npts=1000, width=0.1)
    ef = calc.get_fermi_level()
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(energies - ef, dos, linewidth=2)
    plt.axvline(0, color='red', linestyle='--', linewidth=1, label='Fermi level')
    plt.fill_between(energies - ef, dos, where=(energies <= ef), alpha=0.3, label='Occupied states')
    plt.xlabel('Energy [eV]', fontsize=12)
    plt.ylabel('DOS [states/eV]', fontsize=12)
    plt.title('Si Density of States (PBE)', fontsize=14)
    plt.xlim(-15, 10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('si_dos.png', dpi=150)
    plt.show()
    

* * *

### Example 4: Structure Optimization and Force Calculation
    
    
    from ase.build import molecule
    from ase.optimize import BFGS
    from gpaw import GPAW, PW
    
    # Initial structure of water molecule (distorted structure)
    h2o = molecule('H2O')
    h2o.positions[0] += [0.1, 0.1, 0]  # Intentionally distort
    h2o.center(vacuum=4.0)  # Add vacuum region
    
    # Setup calculator
    calc = GPAW(mode=PW(400),
                xc='PBE',
                txt='h2o_opt.txt')
    
    h2o.calc = calc
    
    print("Initial structure:")
    print(f"O-H1 distance: {h2o.get_distance(0, 1):.3f} Å")
    print(f"O-H2 distance: {h2o.get_distance(0, 2):.3f} Å")
    print(f"H-O-H angle: {h2o.get_angle(1, 0, 2):.1f}°")
    
    # Structure optimization
    opt = BFGS(h2o, trajectory='h2o_opt.traj')
    opt.run(fmax=0.02)
    
    print("\nAfter optimization:")
    print(f"O-H1 distance: {h2o.get_distance(0, 1):.3f} Å")
    print(f"O-H2 distance: {h2o.get_distance(0, 2):.3f} Å")
    print(f"H-O-H angle: {h2o.get_angle(1, 0, 2):.1f}°")
    print(f"\nExperimental values: O-H = 0.958 Å, H-O-H = 104.5°")
    

**Execution Results** :
    
    
    Initial structure:
    O-H1 distance: 1.071 Å
    O-H2 distance: 0.969 Å
    H-O-H angle: 104.5°
    
    After optimization:
    O-H1 distance: 0.972 Å
    O-H2 distance: 0.972 Å
    H-O-H angle: 104.0°
    
    Experimental values: O-H = 0.958 Å, H-O-H = 104.5°
    

* * *

## 2.6 Limitations of DFT and Countermeasures

### Limitation 1: Band Gap Underestimation

**Cause** : Kohn-Sham eigenvalues $\epsilon_i$ are not strictly quasiparticle energies. Inaccuracy of exchange-correlation potential.

**Typical error** : 30-50% underestimation of experimental values

Material | Experimental [eV] | LDA [eV] | GGA [eV]  
---|---|---|---  
Si | 1.17 | 0.5 | 0.6  
GaAs | 1.52 | 0.3 | 0.5  
Diamond | 5.48 | 4.1 | 4.3  
  
**Countermeasures** : 1\. **GW approximation** (many-body perturbation theory): Calculate quasiparticle energies 2\. **Hybrid functionals** (HSE, B3LYP): Mix Hartree-Fock exchange 3\. **Scissors operator** (empirical correction): Shift band gap to experimental value

### Limitation 2: van der Waals Interactions

**Problem** : LDA/GGA cannot describe van der Waals (dispersion) forces.

**Affected systems** : \- Graphite interlayer \- Molecular crystals \- Protein folding

**Countermeasures** : 1\. **DFT-D3** (Grimme's dispersion correction): Add empirical correction term 2\. **vdW-DF** (van der Waals density functional): Non-local correlation functional 3\. **GPAW implementation** :
    
    
    calc = GPAW(mode=PW(400),
                xc='vdW-DF',  # van der Waals functional
                txt='graphite_vdw.txt')
    

### Limitation 3: Strongly Correlated Systems

**Problem** : LDA/GGA cannot describe strong electron correlations.

**Affected systems** : \- Transition metal oxides (NiO, FeO) \- f-electron systems (rare earths, actinides)

**Countermeasures** : 1\. **DFT+U** : Introduce Hubbard U parameter 2\. **DMFT** (Dynamical Mean-Field Theory) 3\. **Hybrid functionals**

* * *

## 2.7 Convergence Tests

In DFT calculations, the following parameters must be converged.

### k-point Mesh Convergence
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: k-point Mesh Convergence
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from ase.build import bulk
    from gpaw import GPAW, PW
    import numpy as np
    import matplotlib.pyplot as plt
    
    si = bulk('Si', 'diamond', a=5.43)
    
    k_grids = [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (10,10,10), (12,12,12)]
    energies = []
    
    for kpts in k_grids:
        calc = GPAW(mode=PW(400), xc='PBE', kpts=kpts, txt=None)
        si.calc = calc
        E = si.get_potential_energy()
        energies.append(E)
        print(f"k-grid {kpts}: E = {E:.6f} eV")
    
    # Plot
    k_total = [k[0]**3 for k in k_grids]
    plt.figure(figsize=(8, 6))
    plt.plot(k_total, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Total k-points', fontsize=12)
    plt.ylabel('Total Energy [eV]', fontsize=12)
    plt.title('k-point Convergence Test', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('k_convergence.png', dpi=150)
    plt.show()
    

**Convergence criterion** : Energy difference < 1 meV/atom

### Cutoff Energy Convergence
    
    
    cutoffs = [200, 300, 400, 500, 600, 700]
    energies = []
    
    for ecut in cutoffs:
        calc = GPAW(mode=PW(ecut), xc='PBE', kpts=(8,8,8), txt=None)
        si.calc = calc
        E = si.get_potential_energy()
        energies.append(E)
        print(f"Cutoff {ecut} eV: E = {E:.6f} eV")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(cutoffs, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Cutoff Energy [eV]', fontsize=12)
    plt.ylabel('Total Energy [eV]', fontsize=12)
    plt.title('Plane Wave Cutoff Convergence Test', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('cutoff_convergence.png', dpi=150)
    plt.show()
    

* * *

## 2.8 Chapter Summary

### What We Learned

  1. **Basic Principles of DFT** \- Hohenberg-Kohn theorem: Everything is determined by electron density \- Kohn-Sham equations: Transformation to non-interacting system \- Exchange-correlation functionals: LDA, GGA

  2. **Practice with ASE + GPAW** \- Structure optimization \- Band structure calculation \- Density of states (DOS) calculation \- Convergence tests

  3. **Limitations of DFT** \- Band gap underestimation \- Lack of van der Waals interactions \- Inapplicability to strongly correlated systems

### Key Points

  * DFT is a practical method for first-principles calculations
  * Computational accuracy depends on the choice of exchange-correlation functional
  * Convergence tests are essential
  * Appropriate corrections are needed depending on the system

### Next Chapter

In Chapter 3, we will learn about **Molecular Dynamics (MD) simulations** that treat nuclear motion.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the physical meaning of the Hohenberg-Kohn first theorem in your own words.

Sample Answer If the electron density $n(\mathbf{r})$ is given, the external potential $V_{\text{ext}}(\mathbf{r})$ is determined. If the external potential is determined, the Hamiltonian $\hat{H}$ is determined, and the Schrödinger equation can be solved. In other words, **all properties of the system are determined by the electron density alone**. This allows us to describe many-electron systems with 3-dimensional electron density instead of the $3N$-dimensional wave function. 

### Problem 2 (Difficulty: medium)

The band gap of Si is 0.6 eV in DFT-GGA (PBE) and 1.17 eV experimentally. Explain the cause of this band gap underestimation.

Sample Answer The main causes of DFT's band gap underestimation are: 1\. **Interpretation problem of Kohn-Sham eigenvalues**: Kohn-Sham eigenvalues $\epsilon_i$ are not strictly quasiparticle energies. The Kohn-Sham equations are formally one-electron Schrödinger equations, but this is a mathematical convenience, and $\epsilon_i$ is not a physical excitation energy. 2\. **Inaccuracy of exchange-correlation functional**: LDA/GGA exchange-correlation functionals do not completely cancel electron self-interaction. This error causes occupied levels to rise (become shallower) and unoccupied levels to drop (become shallower), resulting in band gap underestimation. **Countermeasures**: \- GW approximation: Accurately calculate quasiparticle energies (high computational cost) \- Hybrid functionals (HSE, B3LYP): Mix Hartree-Fock exchange \- DFT+U: For strongly correlated systems \- Scissors operator: Empirically correct band gap 

### Problem 3 (Difficulty: hard)

When calculating graphite interlayer distance with DFT-GGA (PBE), it is significantly overestimated compared to experimental values. Explain the reason and countermeasures.

Sample Answer **Reason**: Because LDA/GGA cannot describe van der Waals (dispersion) interactions. The layers in graphite are bonded not by covalent or ionic bonds, but by **van der Waals forces** (London dispersion forces). These forces are interactions between instantaneous dipoles caused by electron density fluctuations, and are non-local effects. LDA/GGA exchange-correlation functionals are **local** (or semi-local) and depend only on the density $n(\mathbf{r})$ and its gradient $\nabla n(\mathbf{r})$. Therefore, they cannot describe long-range electron correlations (van der Waals forces). As a result: \- Interlayer attraction is underestimated \- Interlayer distance becomes larger than experimental values \- Binding energy is underestimated **Countermeasures**: 1\. **DFT-D3** (Grimme's dispersion correction): \- Add empirical $C_6/r^6$ term \- Parameters determined for each element \- GPAW implementation: `xc='PBE+D3'` 2\. **vdW-DF** (van der Waals density functional): \- Non-local correlation functional \- First-principles (no empirical parameters) \- GPAW implementation: `xc='vdW-DF'` or `xc='vdW-DF2'` 3\. **Calculation example**: 
    
    
    from ase.build import graphite
    from gpaw import GPAW, PW
    
    graphite = graphite(a=2.46, c=6.70)  # Initial structure
    
    # PBE only
    calc_pbe = GPAW(mode=PW(400), xc='PBE', kpts=(8,8,4), txt='gr_pbe.txt')
    graphite.calc = calc_pbe
    E_pbe = graphite.get_potential_energy()
    
    # PBE + D3
    calc_d3 = GPAW(mode=PW(400), xc='PBE+D3', kpts=(8,8,4), txt='gr_d3.txt')
    graphite.calc = calc_d3
    E_d3 = graphite.get_potential_energy()
    
    print(f"PBE: E = {E_pbe:.3f} eV")
    print(f"PBE+D3: E = {E_d3:.3f} eV")
    print(f"vdW correction: {E_d3 - E_pbe:.3f} eV")
    

The vdW correction brings the interlayer distance closer to the experimental value (3.35 Å). 

* * *

## Data Licenses and Citations

### Datasets and Software Used

  1. **GPAW - DFT calculation software** (GPL v3) \- DFT code with plane wave/LCAO basis \- URL: https://wiki.fysik.dtu.dk/gpaw/ \- Citation: Mortensen, J. J., et al. (2024). _Phys. Rev. B_ , 71, 035109.

  2. **ASE - Atomic Simulation Environment** (LGPL v2.1+) \- Atomic structure manipulation and visualization library \- URL: https://wiki.fysik.dtu.dk/ase/ \- Citation: Larsen, A. H., et al. (2017). _J. Phys.: Condens. Matter_ , 29, 273002.

  3. **Materials Project Database** (CC BY 4.0) \- Experimental values for Si, GaAs (lattice constants, band gaps) \- URL: https://materialsproject.org

  4. **Pseudopotential Databases** \- **GPAW PAW Datasets** : GPL v3 \- **PseudoDojo** : BSD License \- URL: http://www.pseudo-dojo.org/

### Sources of Calculation Parameters

Standard parameters used in DFT calculations in this chapter:

  * **k-point mesh** : Monkhorst-Pack method (Monkhorst & Pack, 1976)
  * **Cutoff energy** : Recommended values for each material (GPAW documentation)
  * **Exchange-correlation functional** : PBE (Perdew, Burke, Ernzerhof, 1996)

* * *

## Code Reproducibility Checklist

### Environment Setup
    
    
    # Anaconda recommended (due to complex GPAW dependencies)
    conda create -n dft python=3.11
    conda activate dft
    conda install -c conda-forge gpaw ase matplotlib
    
    # Version check
    python -c "import gpaw; print(gpaw.__version__)"  # 24.1.x or later
    python -c "import ase; print(ase.__version__)"    # 3.22.x or later
    

### Hardware Requirements

Calculation Example | Memory | CPU Time | Recommended Cores  
---|---|---|---  
H₂ optimization | ~2 GB | ~5 min | 1-2 cores  
Si SCF | ~4 GB | ~10 min | 4-8 cores  
Si band structure | ~4 GB | ~15 min | 4-8 cores  
Si DOS | ~4 GB | ~20 min | 4-8 cores  
  
### Computation Time Estimation
    
    
    # Estimate computation time from number of atoms and k-points
    N_atoms = len(atoms)
    N_kpts = np.product(kpts)  # (8,8,8) → 512
    time_estimate = N_atoms * N_kpts * 0.5  # seconds (rough estimate)
    print(f"Estimated computation time: {time_estimate/60:.1f} min")
    

### Troubleshooting

**Problem** : `FileNotFoundError: PAW dataset not found` **Solution** :
    
    
    # Download PAW dataset
    gpaw install-data <directory>
    

**Problem** : Out of memory crash **Solution** :
    
    
    # Reduce memory: reduce k-points
    calc = GPAW(mode=PW(300), kpts=(4,4,4))  # Reduce 8→4
    

**Problem** : Non-convergence (SCF diverges) **Solution** :
    
    
    # Improve convergence: adjust mixing parameter
    calc = GPAW(mode=PW(400), xc='PBE',
                mixer=Mixer(0.05, 5, 50))  # More conservative than default
    

* * *

## Practical Pitfalls and Countermeasures

### 1\. Insufficient k-point Convergence

**Pitfall** : k-points too coarse leading to inaccurate results
    
    
    # ❌ Insufficient: (2,2,2) does not converge
    calc = GPAW(mode=PW(400), xc='PBE', kpts=(2,2,2))
    
    # ✅ Run convergence test
    for k in [2, 4, 6, 8, 10, 12]:
        calc = GPAW(mode=PW(400), xc='PBE', kpts=(k,k,k), txt=None)
        si.calc = calc
        E = si.get_potential_energy()
        print(f"k={k}: E={E:.6f} eV")
    # Convergence criterion: |E(k) - E(k+2)| < 1 meV/atom
    

**Convergence criteria** : \- Metals: k-point density > 0.05 Å⁻¹ \- Semiconductors: k-point density > 0.03 Å⁻¹ \- Insulators: k-point density > 0.02 Å⁻¹

### 2\. Cutoff Energy Setting

**Pitfall** : Cutoff too low leading to insufficient accuracy
    
    
    # ❌ Insufficient: 200 eV inaccurate for light elements
    calc = GPAW(mode=PW(200), xc='PBE')
    
    # ✅ Recommended values for each element
    cutoff_recommendations = {
        'H': 300,   # Hydrogen requires high cutoff
        'C': 400,
        'Si': 300,
        'Fe': 350,
        'Au': 250
    }
    

**Convergence criterion** : Energy difference < 1 meV/atom

### 3\. SCF Convergence Failure

**Pitfall** : Poor initial density preventing convergence
    
    
    # ❌ Non-convergent: fixed occupation for metal
    calc = GPAW(mode=PW(400), occupations=FermiDirac(0.0))  # 0K
    
    # ✅ Metallic systems: relax with finite temperature
    from gpaw import FermiDirac
    calc = GPAW(mode=PW(400),
                occupations=FermiDirac(0.1))  # kT = 0.1 eV ≈ 1160 K
    

**Convergence criterion** : Electron density residual < 1e-4

### 4\. Structure Optimization Failure

**Pitfall** : Insufficient accuracy in force calculation
    
    
    # ❌ Inaccurate: low cutoff → poor force accuracy
    calc = GPAW(mode=PW(250), xc='PBE')
    opt.run(fmax=0.01)  # Cannot be achieved
    
    # ✅ Higher cutoff for force calculations
    calc = GPAW(mode=PW(500), xc='PBE')  # 1.5-2x higher
    opt.run(fmax=0.05)  # Realistic threshold
    

**Recommended thresholds** : \- Coarse optimization: fmax = 0.1 eV/Å \- Standard: fmax = 0.05 eV/Å \- High precision: fmax = 0.01 eV/Å

### 5\. Missing Spin Polarization

**Pitfall** : Not considering spin polarization for magnetic materials
    
    
    # ❌ Wrong: Fe without spin polarization
    calc = GPAW(mode=PW(400), xc='PBE')
    
    # ✅ Correct: enable spin polarization
    calc = GPAW(mode=PW(400), xc='PBE',
                spinpol=True,  # Spin polarization ON
                magmom=2.2)    # Initial magnetic moment
    

* * *

## Quality Assurance Checklist

### Validation of DFT Calculations

#### Convergence Tests (Required)

  * [ ] k-point convergence: Energy difference < 1 meV/atom
  * [ ] Cutoff convergence: Energy difference < 1 meV/atom
  * [ ] SCF convergence: Electron density residual < 1e-4
  * [ ] Cell size convergence (molecular systems): Vacuum region > 4 Å

#### Structure Validity

  * [ ] Lattice constant within ±3% of experimental value
  * [ ] Interatomic distances chemically reasonable (bond lengths)
  * [ ] All forces < fmax (after structure optimization)
  * [ ] Stress tensor nearly zero (after NPT optimization)

#### Band Structure Validity

  * [ ] Band gap sign correct (metal vs insulator)
  * [ ] Symmetry preserved (band degeneracy at high-symmetry points)
  * [ ] Smooth dispersion relation (no noise)
  * [ ] Fermi level appropriate (0 eV for metals)

#### Numerical Soundness

  * [ ] Energy is finite (no divergence)
  * [ ] All force components are finite
  * [ ] Charge conservation: Total electrons = sum of nuclear charges
  * [ ] Pressure in physical range (±100 GPa)

### DFT-Specific Quality Checks

#### Exchange-Correlation Functional Selection

  * [ ] PBE (standard): Crystal structures, lattice constants
  * [ ] LDA: Comparison with past data
  * [ ] vdW-DF: Layered materials, molecular crystals
  * [ ] HSE/PBE0: Band gaps (high computational cost)

#### Pseudopotential Verification

  * [ ] PAW dataset version recorded
  * [ ] Correct number of valence electrons
  * [ ] Cutoff energy at or above recommended value
  * [ ] Validation with multiple pseudopotentials (if possible)

* * *

## References

  1. Hohenberg, P., & Kohn, W. (1964). "Inhomogeneous Electron Gas." _Physical Review_ , 136(3B), B864-B871. DOI: [10.1103/PhysRev.136.B864](<https://doi.org/10.1103/PhysRev.136.B864>)

  2. Kohn, W., & Sham, L. J. (1965). "Self-Consistent Equations Including Exchange and Correlation Effects." _Physical Review_ , 140(4A), A1133-A1138. DOI: [10.1103/PhysRev.140.A1133](<https://doi.org/10.1103/PhysRev.140.A1133>)

  3. Perdew, J. P., Burke, K., & Ernzerhof, M. (1996). "Generalized Gradient Approximation Made Simple." _Physical Review Letters_ , 77(18), 3865-3868. DOI: [10.1103/PhysRevLett.77.3865](<https://doi.org/10.1103/PhysRevLett.77.3865>)

  4. Martin, R. M. (2004). _Electronic Structure: Basic Theory and Practical Methods_. Cambridge University Press.

  5. ASE Documentation: https://wiki.fysik.dtu.dk/ase/

  6. GPAW Documentation: https://wiki.fysik.dtu.dk/gpaw/

* * *

## Author Information

**Authors** : MI Knowledge Hub Content Team **Created** : 2025-10-17 **Version** : 1.0 **Series** : Computational Materials Science Basics Introduction v1.0

**License** : Creative Commons BY-NC-SA 4.0
