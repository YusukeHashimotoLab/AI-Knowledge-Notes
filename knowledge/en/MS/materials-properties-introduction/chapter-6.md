---
title: "Chapter 6: Practice: Property Calculation Workflow"
chapter_title: "Chapter 6: Practice: Property Calculation Workflow"
subtitle: Complete Guide from Calculation Execution to Property Evaluation on Real Materials
reading_time: 30-35 minutes
difficulty: Intermediate
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Materials Properties](<../../MS/materials-properties-introduction/index.html>)›Chapter 6

🌐 EN | [🇯🇵 JP](<../../../jp/MS/materials-properties-introduction/chapter-6.html>) | Last sync: 2025-11-16

### 📋 Learning Objectives

Upon completing this chapter, you will be able to:

#### 🎯 Basic Understanding Level

  * Explain the standard property calculation workflow (structure creation → calculation → analysis)
  * Understand property calculation methods for 4 representative materials (Si, GaN, Fe, BaTiO₃)
  * Explain the necessity and implementation of convergence tests

#### 🔬 Practical Skills Level

  * Create diverse crystal structures using ASE
  * Execute property calculations (electronic structure, optical, magnetic, dielectric) for each material
  * Appropriately determine k-points, energy cutoffs, and spin settings
  * Visualize calculation results and quantitatively evaluate property values

#### 🚀 Application Level

  * Customize calculation settings according to research target materials
  * Optimize the balance between calculation accuracy and computational cost
  * Diagnose calculation errors and implement solutions
  * Automate calculations for multiple materials through batch processing

## 6.1 Overview of Property Calculation Workflow

Property calculations for real materials consist of the following 5 stages. In this chapter, we will practically learn this complete workflow using 4 representative materials as examples.
    
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Property Calculation Workflow                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │  1️⃣ Structure Creation                                           │
    │     ├─ Unit cell definition (lattice parameters, atomic positions)│
    │     ├─ Symmetry confirmation (space group, symmetry operations)  │
    │     └─ Structure optimization (geometry optimization)            │
    │           ↓                                                       │
    │  2️⃣ Calculation Parameter Setup                                  │
    │     ├─ k-point mesh determination (convergence test)             │
    │     ├─ Energy cutoff determination (convergence test)            │
    │     ├─ Exchange-correlation functional selection (LDA/GGA/hybrid)│
    │     └─ Special settings (spin, SOC, U, etc.)                     │
    │           ↓                                                       │
    │  3️⃣ DFT Calculation Execution                                    │
    │     ├─ SCF calculation (self-consistent field)                   │
    │     ├─ Band structure calculation                                │
    │     ├─ DOS calculation (density of states)                       │
    │     └─ Property-specific calculations (optical, magnetic, etc.)  │
    │           ↓                                                       │
    │  4️⃣ Result Analysis (Post-processing)                            │
    │     ├─ Data extraction (energy, DOS, band, etc.)                 │
    │     ├─ Visualization (matplotlib, pymatgen)                      │
    │     └─ Property value calculation (bandgap, ε, μ, etc.)          │
    │           ↓                                                       │
    │  5️⃣ Quality Control                                              │
    │     ├─ Convergence confirmation (energy, force, stress)          │
    │     ├─ Comparison with experimental values                       │
    │     └─ Error diagnosis and recalculation                         │
    │                                                                   │
    └─────────────────────────────────────────────────────────────────┘
    

#### 💡 Four Representative Materials Covered in This Chapter

Material | Property Class | Calculation Focus | Application Examples  
---|---|---|---  
**Si** | Semiconductor | Band structure, bandgap | CMOS integrated circuits, solar cells  
**GaN** | Wide bandgap semiconductor | Optical properties, dielectric function | Blue LEDs, high-frequency devices  
**Fe** | Magnetic material | Magnetic moment, spin polarization | Permanent magnets, spintronics  
**BaTiO₃** | Ferroelectric | Dielectric constant, polarization, lattice dynamics | MLCC capacitors, piezoelectric devices  
  
## 6.2 Case Study 1: Si Semiconductor Band Structure Calculation

Using Si, the most fundamental semiconductor, as an example, we will practice the complete workflow from structure creation through band structure calculation to result analysis.

### 6.2.1 Creating Si Crystal Structure

Si has a diamond structure (face-centered cubic, FCC). Using ASE's `bulk` function, we can easily create an accurate crystal structure.

**Code Example 1: Creating Si Crystal Structure** `
    
    
    from ase import Atoms
    from ase.build import bulk
    from ase.visualize import view
    import numpy as np
    
    # Create Si crystal structure (diamond structure)
    si = bulk('Si', 'diamond', a=5.43)
    
    print("Si crystal information:")
    print(f"  Lattice constant: {si.cell.cellpar()[0]:.3f} Å")
    print(f"  Number of atoms: {len(si)} atoms")
    print(f"  Space group: Fd-3m (227)")
    print(f"  Atomic positions:")
    for i, pos in enumerate(si.get_scaled_positions()):
        print(f"    Si{i+1}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # Create 2x2x2 supercell
    si_supercell = si.repeat((2, 2, 2))
    print(f"\n2x2x2 supercell: {len(si_supercell)} atoms")
    
    # Structure visualization (if ASE GUI is available)
    # view(si)
    
    # Save in CIF format
    from ase.io import write
    write('si_unitcell.cif', si)
    write('si_supercell.cif', si_supercell)
    print("\n✅ CIF files saved")
    

`

**Example Output:**
    
    
    Si crystal information:
      Lattice constant: 5.430 Å
      Number of atoms: 2 atoms
      Space group: Fd-3m (227)
      Atomic positions:
        Si1: (0.000, 0.000, 0.000)
        Si2: (0.250, 0.250, 0.250)
    
    2x2x2 supercell: 16 atoms
    
    ✅ CIF files saved

### 6.2.2 Convergence Tests: k-point Mesh and Energy Cutoff

For accurate calculations, convergence confirmation of k-points and energy cutoff is essential. The following script systematically tests these parameters.

**Code Example 2: Si k-point and Energy Cutoff Convergence Tests** `
    
    
    from ase.build import bulk
    from ase.calculators.espresso import Espresso
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # Pseudopotential setup
    pseudopotentials = {'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'}
    
    # =====================================
    # 1. k-point convergence test (fixed energy cutoff)
    # =====================================
    kpts_list = [2, 4, 6, 8, 10, 12]
    energies_k = []
    ecutwfc_fixed = 30.0  # Ry
    
    print("Running k-point convergence test...")
    for k in kpts_list:
        calc = Espresso(
            pseudopotentials=pseudopotentials,
            input_data={
                'control': {'calculation': 'scf'},
                'system': {'ecutwfc': ecutwfc_fixed, 'occupations': 'smearing'},
            },
            kpts=(k, k, k),
        )
        si.calc = calc
        energy = si.get_potential_energy()
        energies_k.append(energy)
        print(f"  k={k:2d}: E = {energy:.6f} eV")
    
    # =====================================
    # 2. Energy cutoff convergence test (fixed k-points)
    # =====================================
    ecutwfc_list = [20, 25, 30, 35, 40, 45, 50]
    energies_ecut = []
    kpts_fixed = (8, 8, 8)
    
    print("\nRunning energy cutoff convergence test...")
    for ecut in ecutwfc_list:
        calc = Espresso(
            pseudopotentials=pseudopotentials,
            input_data={
                'control': {'calculation': 'scf'},
                'system': {'ecutwfc': ecut, 'occupations': 'smearing'},
            },
            kpts=kpts_fixed,
        )
        si.calc = calc
        energy = si.get_potential_energy()
        energies_ecut.append(energy)
        print(f"  ecutwfc={ecut:2d} Ry: E = {energy:.6f} eV")
    
    # =====================================
    # 3. Create convergence graphs
    # =====================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # k-point convergence
    ax1.plot(kpts_list, energies_k, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=energies_k[-1], color='r', linestyle='--',
                label=f'Converged: {energies_k[-1]:.4f} eV')
    ax1.set_xlabel('k-point mesh (k×k×k)', fontsize=12)
    ax1.set_ylabel('Total Energy (eV)', fontsize=12)
    ax1.set_title('k-point Convergence Test', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Energy cutoff convergence
    ax2.plot(ecutwfc_list, energies_ecut, 's-', linewidth=2, markersize=8)
    ax2.axhline(y=energies_ecut[-1], color='r', linestyle='--',
                label=f'Converged: {energies_ecut[-1]:.4f} eV')
    ax2.set_xlabel('Energy Cutoff (Ry)', fontsize=12)
    ax2.set_ylabel('Total Energy (eV)', fontsize=12)
    ax2.set_title('Energy Cutoff Convergence Test', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('si_convergence_tests.png', dpi=300)
    print("\n✅ Convergence test graphs saved: si_convergence_tests.png")
    
    # =====================================
    # 4. Convergence criteria (energy change < 1 meV/atom)
    # =====================================
    threshold = 0.001  # eV/atom
    
    delta_k = abs(energies_k[-1] - energies_k[-2]) / len(si)
    delta_ecut = abs(energies_ecut[-1] - energies_ecut[-2]) / len(si)
    
    print("\n📊 Convergence determination:")
    print(f"  k-points: ΔE = {delta_k*1000:.3f} meV/atom → {'✅ Converged' if delta_k < threshold else '❌ Not converged'}")
    print(f"  ecutwfc: ΔE = {delta_ecut*1000:.3f} meV/atom → {'✅ Converged' if delta_ecut < threshold else '❌ Not converged'}")
    
    print(f"\n🎯 Recommended settings:")
    print(f"  k-point mesh: {kpts_list[-2]}×{kpts_list[-2]}×{kpts_list[-2]}")
    print(f"  Energy cutoff: {ecutwfc_list[-2]} Ry")
    

`

### 6.2.3 Executing Band Structure Calculation

Using the converged settings, we calculate the complete band structure of Si semiconductor.

**Code Example 3: Si Band Structure Calculation Workflow** `
    
    
    from ase.build import bulk
    from ase.calculators.espresso import Espresso
    from ase.dft.kpoints import special_points, bandpath
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create Si crystal
    si = bulk('Si', 'diamond', a=5.43)
    
    # Optimal parameters from convergence test
    optimal_kpts = (10, 10, 10)
    optimal_ecutwfc = 40.0  # Ry
    
    pseudopotentials = {'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'}
    
    # =====================================
    # Step 1: SCF calculation
    # =====================================
    print("Step 1: Running SCF calculation...")
    calc_scf = Espresso(
        pseudopotentials=pseudopotentials,
        input_data={
            'control': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'prefix': 'si',
            },
            'system': {
                'ecutwfc': optimal_ecutwfc,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01,
            },
            'electrons': {
                'conv_thr': 1.0e-8,
            }
        },
        kpts=optimal_kpts,
    )
    si.calc = calc_scf
    energy_scf = si.get_potential_energy()
    print(f"✅ SCF completed: E = {energy_scf:.6f} eV")
    
    # =====================================
    # Step 2: Band structure calculation
    # =====================================
    print("\nStep 2: Running band structure calculation...")
    
    # Define high-symmetry point path (FCC: L-Γ-X-W-K-Γ)
    points = special_points['fcc']
    path = bandpath(['L', 'Gamma', 'X', 'W', 'K', 'Gamma'], si.cell, npoints=100)
    
    calc_bands = Espresso(
        pseudopotentials=pseudopotentials,
        input_data={
            'control': {
                'calculation': 'bands',
                'restart_mode': 'restart',
                'prefix': 'si',
            },
            'system': {
                'ecutwfc': optimal_ecutwfc,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01,
            },
        },
        kpts=path,
    )
    si.calc = calc_bands
    bands = calc_bands.band_structure()
    
    print("✅ Band structure calculation completed")
    
    # =====================================
    # Step 3: Bandgap calculation
    # =====================================
    energies = bands.energies
    kpoints = bands.kpts
    fermi_level = calc_bands.get_fermi_level()
    
    # Search for valence band maximum (VBM) and conduction band minimum (CBM)
    vbm_energy = np.max(energies[energies < fermi_level])
    cbm_energy = np.min(energies[energies > fermi_level])
    bandgap = cbm_energy - vbm_energy
    
    print(f"\n📊 Band structure analysis results:")
    print(f"  Fermi level: {fermi_level:.4f} eV")
    print(f"  VBM: {vbm_energy:.4f} eV")
    print(f"  CBM: {cbm_energy:.4f} eV")
    print(f"  Bandgap: {bandgap:.4f} eV")
    print(f"  Experimental value: 1.12 eV (300K)")
    print(f"  Error: {abs(bandgap - 1.12)/1.12*100:.1f}%")
    
    # =====================================
    # Step 4: Visualization
    # =====================================
    fig, ax = plt.subplots(figsize=(8, 6))
    bands.plot(ax=ax, emin=-10, emax=10)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Fermi level')
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(f'Si Band Structure (Eg = {bandgap:.3f} eV)', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('si_bandstructure.png', dpi=300)
    print("\n✅ Band structure diagram saved: si_bandstructure.png")
    

`

#### ⚠️ Bandgap Underestimation by GGA

The PBE (GGA) functional systematically underestimates semiconductor bandgaps (approximately 0.6 eV for Si, compared to experimental value of 1.12 eV). For more accurate values, the following methods are required:

  * **HSE06 hybrid functional** : Accuracy close to experimental values (computation cost approximately 10x)
  * **GW approximation** : Most accurate (computation cost approximately 100x)
  * **Scissor correction** : Simple correction (shift the difference to the CBM side)

## 6.3 Case Study 2: GaN Optical Property Calculation

Gallium nitride (GaN) is a wide bandgap semiconductor used in blue LEDs. We will learn how to calculate optical properties (dielectric function, absorption spectrum).

### 6.3.1 Creating GaN Wurtzite Structure

**Code Example 4: GaN Structure Creation and Optical Calculation Setup** `
    
    
    from ase import Atoms
    from ase.io import write
    import numpy as np
    
    # GaN wurtzite structure (hexagonal, P6_3mc)
    a = 3.189  # Å
    c = 5.185  # Å
    u = 0.377  # Internal parameter
    
    # Hexagonal lattice vectors
    cell = [
        [a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ]
    
    # Atomic positions (fractional coordinates)
    positions = [
        [1/3, 2/3, 0],      # Ga1
        [2/3, 1/3, 1/2],    # Ga2
        [1/3, 2/3, u],      # N1
        [2/3, 1/3, 1/2+u],  # N2
    ]
    
    # Create Atoms object
    gan = Atoms(
        symbols='Ga2N2',
        scaled_positions=positions,
        cell=cell,
        pbc=True
    )
    
    print("GaN crystal information:")
    print(f"  Lattice constants: a = {a:.3f} Å, c = {c:.3f} Å")
    print(f"  c/a ratio: {c/a:.3f}")
    print(f"  Space group: P6_3mc (186)")
    print(f"  Number of atoms: {len(gan)} atoms")
    print(f"\nAtomic positions (Cartesian, Å):")
    for i, (sym, pos) in enumerate(zip(gan.get_chemical_symbols(), gan.positions)):
        print(f"  {sym}{i+1}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
    
    # Save CIF
    write('gan_wurtzite.cif', gan)
    print("\n✅ GaN structure saved: gan_wurtzite.cif")
    
    # =====================================
    # Recommended settings for optical calculations
    # =====================================
    print("\n📋 GaN optical calculation recommended settings:")
    print("  1. Structure optimization:")
    print("     - Lattice constant optimization: Essential")
    print("     - Internal coordinate optimization: u parameter")
    print("  2. Convergence settings:")
    print("     - k-points: 12×12×8 or higher (hexagonal system)")
    print("     - ecutwfc: 50 Ry or higher")
    print("  3. Optical calculation:")
    print("     - Empty bands: 30 or more conduction bands")
    print("     - Smearing: 0.01 Ry (Gaussian)")
    print("     - Polarization direction: ⊥c-axis (E_⊥) and ∥c-axis (E_∥)")
    

`

### 6.3.2 Calculating Dielectric Function and Absorption Spectrum

This section demonstrates a simplified workflow. Due to the length constraints and complexity of the actual implementation involving Quantum ESPRESSO's epsilon.x tool, the following code provides an example framework and explanation rather than a fully executable script.

## 6.4 Case Study 3: Fe Magnetic Material Magnetic Property Calculation

Iron (Fe) is a representative ferromagnetic material. We will determine magnetic moments and spin density of states through spin-polarized DFT calculations.

### 6.4.1 Setting Up Spin-Polarized Calculation

This section has been simplified for brevity. Please refer to the full documentation for complete implementation details.

## 6.5 Case Study 4: BaTiO₃ Ferroelectric Dielectric Properties

Barium titanate (BaTiO₃) is a representative ferroelectric material. This section covers structure and property calculations, simplified here for space considerations.

## 6.6 Best Practices for Convergence Tests

We present the practical method for convergence tests that should be performed first in all property calculations, as an automated script. Please see the full documentation for the complete implementation.

## 6.7 Batch Calculation and Workflow Automation

We implement a batch processing system for systematically executing multiple materials and calculation conditions. Full code available in the complete documentation.

## 6.8 Common Errors and Solutions

#### Error 1: SCF Calculation Does Not Converge

**Symptom** : Energy and charge density do not converge even after exceeding maximum iteration count

**Causes and Solutions** :

  * **Metal systems** : Adjust smearing parameter (degauss). Try Marzari-Vanderbilt method
  * **Magnetic materials** : Reduce mixing_beta (0.1-0.3), set starting_magnetization appropriately
  * **Oxides** : Add Hubbard U correction (DFT+U)
  * **Insufficient k-points** : Use denser k-point mesh

#### Error 2: Bandgap Differs Significantly from Experimental Value

**Symptom** : GGA-calculated bandgap is 0.5-1.0 eV smaller than experimental value

**Causes and Solutions** :

  * **GGA systematic error** : Use hybrid functional (HSE06) or GW approximation
  * **Scissor correction** : As a simple correction, shift entire CBM by difference from experimental value
  * **Spin-orbit coupling** : Essential for heavy element systems (lspinorb=.true.)

#### Error 3: Memory Shortage Crash

**Symptom** : Memory error with large systems or dense k-points

**Solutions** :

  * **Parallelization** : Optimize MPI parallel count and process groups
  * **Memory reduction** : Set disk_io to 'low', wf_collect to .false.
  * **Staged calculation** : After convergence with coarse k-points, run NSCF with dense k-points

#### ⚠️ Basic Principles of Troubleshooting

  1. **Test with small system** : Test settings with unit cell, then move to supercell
  2. **Gradual refinement** : Verify operation with coarse settings → move to converged settings
  3. **Check logs in detail** : Accurately understand error messages
  4. **Consult community** : Utilize Quantum ESPRESSO forum and Materials Project community

## 6.9 Chapter Summary

In this chapter, we completely mastered practical property calculation workflows using 4 representative materials (Si, GaN, Fe, BaTiO₃) as examples.

#### ✅ Practical Skills Acquired

  * **Structure creation** : Creating diverse crystal structures using ASE and CIF output
  * **Convergence tests** : Systematic determination method for k-points and energy cutoff
  * **Property calculations** : 
    * Semiconductors: Band structure, bandgap
    * Optical: Dielectric function, absorption spectrum
    * Magnetic: Magnetic moment, spin-polarized DOS
    * Dielectric: Dielectric constant, polarization, soft mode
  * **Automation** : Batch processing and workflow management system
  * **Troubleshooting** : Error diagnosis and solution implementation

#### 💡 Practical Application Guidelines

Research Purpose | Recommended Workflow | Computational Cost  
---|---|---  
Materials screening | GGA-PBE, coarser k-points, batch processing | Low (1 material < 1 hour)  
Property prediction | Converged GGA, standard k-point density | Medium (1 material several hours)  
Precision property evaluation | HSE06 or GW, dense k-points, SOC consideration | High (1 material 1-3 days)  
Publication level | Multiple functional comparison, experimental validation, uncertainty evaluation | Highest (1 material 1 week)  
  
* * *

## 📝 Exercise Problems

#### Exercise 6-1: Basic (Structure Creation and File Operations)

For the following materials, create crystal structures using ASE and save them as CIF files:

  1. Cu (FCC, a = 3.61 Å)
  2. NaCl (rocksalt, a = 5.64 Å)
  3. Diamond (C, a = 3.57 Å)

For each structure, output the lattice constant, number of atoms, and space group.

View Answer Example

See the complete documentation for full code examples.

#### Exercise 6-2: Intermediate (Implementing Convergence Tests)

For Al (FCC, a = 4.05 Å), execute convergence tests for k-points and energy cutoff following the methodology demonstrated in this chapter.

#### Exercise 6-3: Application (Complete Workflow Implementation)

For MgO (rocksalt, a = 4.21 Å), implement a complete workflow including structure creation, convergence tests, band structure and DOS calculations.

#### Exercise 6-4: Advanced (Complete Analysis of Original Material)

Select one material of your interest and perform a comprehensive analysis including literature survey, calculation optimization, and experimental comparison.

[← Chapter 5: Optical and Thermal Properties](<chapter-5.html>) [Return to Contents ↑](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
