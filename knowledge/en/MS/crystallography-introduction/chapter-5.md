---
title: "Chapter 5: Practical Crystallographic Computing with pymatgen"
chapter_title: "Chapter 5: Practical Crystallographic Computing with pymatgen"
subtitle: From Materials Project Integration to Practical Workflows
reading_time: 30-36 minutes
difficulty: Intermediate
code_examples: 8
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/crystallography-introduction/chapter-5.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../index.html>) > [MS Dojo](<../index.html>) > [Introduction to Crystallography](<index.html>) > Chapter 5 

## Learning Objectives

By completing this final chapter, you will acquire the following practical skills:

  * Understand basic usage of the **pymatgen library** and manipulate Structure objects
  * Read and write **CIF files**
  * Retrieve crystal structures from databases using the **Materials Project API**
  * Perform **crystal structure transformations** (supercells, slab models)
  * Conduct **symmetry analysis** (space group identification, Wyckoff positions)
  * Perform **crystal structure visualization** (2D/3D)
  * Execute **practical workflows** (data acquisition → analysis → visualization)
  * Establish a **solid foundation** for Materials Informatics research

## 1\. Overview of pymatgen and Environment Setup

### 1.1 What is pymatgen?

**pymatgen (Python Materials Genomics)** is a comprehensive Python library for materials science. Developed by the MIT Materials Project research group, it is widely used in crystal structure analysis, first-principles calculations, and Materials Informatics research. 

#### Key Features of pymatgen

  * **Crystal structure representation and manipulation** : Managing atomic arrangements through Structure objects
  * **CIF file reading and writing** : Integration with crystal structure databases
  * **Symmetry analysis** : Automatic space group recognition, Wyckoff position calculation
  * **Materials Project integration** : Access to a database of over 140,000 materials
  * **Structure transformations** : Supercells, surface models, doping
  * **Physical property calculations** : XRD patterns, density of states, band structures
  * **First-principles calculation integration** : Input/output for VASP, Quantum ESPRESSO, etc.

### 1.2 Installation and Setup

#### Installation Procedure
    
    
    # Execute in terminal/command prompt
    # Basic installation
    pip install pymatgen
    
    # For Materials Project API integration (recommended)
    pip install mp-api
    
    # Install visualization libraries together
    pip install matplotlib numpy scipy plotly
    
    # Verify installation
    python -c "import pymatgen; print(pymatgen.__version__)"
    # Example output: 2024.10.3
    

#### Obtaining a Materials Project API Key

To access the Materials Project database, you need a free API key:

  1. Visit the [Materials Project official site](<https://next-gen.materialsproject.org/>)
  2. Create a free account via "Sign Up" in the upper right
  3. After logging in, navigate to the "API" section
  4. Click "Generate API Key" to create a new API key
  5. Copy the displayed API key (e.g., `abc123xyz...`)
  6. Set it as an environment variable or use in your script

**Security Note** : Your API key is confidential information. Do not commit it to GitHub or other repositories.

## 2\. Basic Operations with Structure Objects

### 2.1 Creating Crystal Structures and Retrieving Information

#### Code Example 1: Basic Structure Object Operations

Create the diamond structure of silicon (Si) and retrieve basic information:
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # Create silicon with diamond structure
    # Define lattice vectors (cubic, a = 5.43 Å)
    a = 5.43
    lattice = Lattice.cubic(a)
    
    # Atom species and fractional coordinates
    species = ['Si'] * 8
    coords = [
        [0.00, 0.00, 0.00],  # FCC lattice points
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],  # Tetrahedral interstices
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ]
    
    # Create Structure object
    si_structure = Structure(lattice, species, coords)
    
    print("=== Silicon (Si) Crystal Structure Information ===\n")
    
    # Retrieve basic information
    print(f"Composition: {si_structure.composition}")
    print(f"Formula: {si_structure.formula}")
    print(f"Unit cell volume: {si_structure.volume:.4f} Å³")
    print(f"Density: {si_structure.density:.4f} g/cm³")
    print(f"Number of atoms: {len(si_structure)}")
    
    # Retrieve lattice parameters
    print(f"\nLattice parameters:")
    print(f"  a = {si_structure.lattice.a:.4f} Å")
    print(f"  b = {si_structure.lattice.b:.4f} Å")
    print(f"  c = {si_structure.lattice.c:.4f} Å")
    print(f"  α = {si_structure.lattice.alpha:.2f}°")
    print(f"  β = {si_structure.lattice.beta:.2f}°")
    print(f"  γ = {si_structure.lattice.gamma:.2f}°")
    
    # Information for each atom
    print(f"\nAtomic coordinates:")
    print(f"{'Atom':<8} {'Fractional coords (x, y, z)':<30} {'Cartesian coords (Å)':<25}")
    print("-" * 65)
    
    for i, site in enumerate(si_structure):
        species = site.species_string
        frac_coords = site.frac_coords
        cart_coords = site.coords
    
        print(f"{species:<8} ({frac_coords[0]:6.3f}, {frac_coords[1]:6.3f}, {frac_coords[2]:6.3f})    "
              f"({cart_coords[0]:6.3f}, {cart_coords[1]:6.3f}, {cart_coords[2]:6.3f})")
    
    # Nearest neighbor distances
    print(f"\nNearest neighbor distances:")
    neighbors = si_structure.get_neighbors(si_structure[0], 3.0)
    for neighbor, distance in neighbors:
        print(f"  {distance:.4f} Å")
    

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
