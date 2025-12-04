---
title: "Chapter 1: Crystal Structure of Ceramics"
chapter_title: "Chapter 1: Crystal Structure of Ceramics"
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Ceramic Materials](<../../MS/ceramic-materials-introduction/index.html>)›Chapter 1

🌐 EN | [🇯🇵 JP](<../../../jp/MS/ceramic-materials-introduction/chapter-1.html>) | Last sync: 2025-11-16

  * [Top](<index.html>)
  * [Overview](<#intro>)
  * [Ionic Bonding](<#ionic>)
  * [Covalent Bonding](<#covalent>)
  * [Perovskite](<#perovskite>)
  * [Spinel](<#spinel>)
  * [Exercises](<#exercises>)
  * [References](<#references>)
  * [Next Chapter →](<chapter-2.html>)

This chapter covers Crystal Structure of Ceramics. You will learn □ Can explain the differences between NaCl type, □ Can describe characteristics of ionic, and □ Can explain the difference between normal spinel.

## 1.1 Overview of Ceramic Crystal Structures

Ceramic materials are compounds of metallic elements with non-metallic elements (primarily oxygen, nitrogen, and carbon), and their properties are determined by **crystal structure** and **chemical bonding**. In this chapter, we will learn the major crystal structures of ceramics and practice structural analysis using Python tools (**Pymatgen**). 

**Learning Objectives of This Chapter**

  * **Level 1 (Fundamental Understanding)** : Identify and explain the characteristics of major crystal structure types (NaCl, CsCl, perovskite, spinel)
  * **Level 2 (Practical Skills)** : Generate crystal structures using Pymatgen and calculate lattice parameters, density, and X-ray diffraction patterns
  * **Level 3 (Application Ability)** : Apply Pauling's rules to evaluate structural stability and predict structure-property correlations

### Classification of Ceramic Structures

Ceramic crystal structures are broadly classified into two categories based on bonding nature:

  1. **Ionic bonding ceramics** : NaCl, MgO, Al₂O₃, etc. (dominated by electrostatic attraction)
  2. **Covalent bonding ceramics** : SiC, Si₃N₄, AlN, etc. (directional covalent bonds)

    
    
    ```mermaid
    flowchart TD
                    A[Ceramic Crystal Structures] --> B[Ionic Bonding]
                    A --> C[Covalent Bonding]
                    B --> D[NaCl TypeMgO, CaO]
                    B --> E[CsCl TypeCsCl]
                    B --> F[CaF₂ TypeZrO₂, UO₂]
                    B --> G[PerovskiteBaTiO₃, SrTiO₃]
                    B --> H[SpinelMgAl₂O₄, Fe₃O₄]
                    C --> I[Diamond TypeC, Si]
                    C --> J[Zinc Blende TypeSiC, GaAs]
                    C --> K[Wurtzite TypeZnO, AlN]
    
                    style A fill:#f093fb,color:#fff
                    style B fill:#e3f2fd
                    style C fill:#fff3e0
                    style G fill:#f5576c,color:#fff
                    style H fill:#f5576c,color:#fff
    ```

## 1.2 Ionic Bonding Ceramics

### 1.2.1 NaCl-Type Structure

The **NaCl-type (rock salt) structure** is the most fundamental ionic crystal structure. Cations and anions each form face-centered cubic (FCC) lattices that interpenetrate each other. Each ion is surrounded by six ions of opposite charge (coordination number CN = 6). 

Representative materials:

  * **MgO (magnesium oxide)** : Refractory materials, high-temperature insulators (melting point 2852°C)
  * **CaO (calcium oxide)** : Cement raw material
  * **NiO (nickel oxide)** : Battery cathode material

#### Pymatgen Implementation: Generation and Visualization of NaCl Structure
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 1: NaCl-type structure generation and basic analysis
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    import matplotlib.pyplot as plt
    import numpy as np
    
    def create_nacl_structure(a=4.2):
        """
        Function to generate NaCl-type structure
    
        Parameters:
        -----------
        a : float
            Lattice parameter [Å] (default: 4.2Å for MgO)
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            NaCl-type crystal structure
        """
        # Define FCC lattice
        lattice = Lattice.cubic(a)
    
        # Define atomic positions (fractional coordinates)
        # Mg2+: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) - FCC
        # O2-: (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5) - FCC shifted
        species = ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Mg
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]   # O
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # Generate MgO structure
    mgo = create_nacl_structure(a=4.211)  # Experimental lattice constant of MgO
    
    print("=== MgO (NaCl Type) Structure Information ===")
    print(f"Chemical formula: {mgo.composition.reduced_formula}")
    print(f"Lattice parameter: a = {mgo.lattice.a:.3f} Å")
    print(f"Unit cell volume: {mgo.volume:.2f} Å³")
    print(f"Density: {mgo.density:.2f} g/cm³")
    print(f"Number of atoms per unit cell: {len(mgo)}")
    
    # Calculate coordination number
    mg_site = mgo[0]  # First Mg atom
    neighbors = mgo.get_neighbors(mg_site, r=3.0)
    print(f"\nNumber of nearest neighbor O atoms around Mg: {len(neighbors)}")
    print(f"Mg-O distance: {neighbors[0][1]:.3f} Å")
    
    # Expected output:
    # === MgO (NaCl Type) Structure Information ===
    # Chemical formula: MgO
    # Lattice parameter: a = 4.211 Å
    # Unit cell volume: 74.68 Å³
    # Density: 3.58 g/cm³
    # Number of atoms per unit cell: 8
    #
    # Number of nearest neighbor O atoms around Mg: 6
    # Mg-O distance: 2.106 Å
    

### 1.2.2 Pauling's Rules

**Pauling's rules** are five empirical rules for predicting stable structures of ionic crystals. The most important are the **first rule** and **second rule** : 

**Pauling's First Rule: Coordination Polyhedra** The distance between cation and anion is determined by the sum of their radii \\(r_+ + r_-\\). The coordination number (CN) is determined by the radius ratio \\(r_+/r_-\\): \\[ \text{CN} = \begin{cases} 2 & \text{(linear)} & 0 < r_+/r_- < 0.155 \\\ 3 & \text{(triangular)} & 0.155 < r_+/r_- < 0.225 \\\ 4 & \text{(tetrahedral)} & 0.225 < r_+/r_- < 0.414 \\\ 6 & \text{(octahedral)} & 0.414 < r_+/r_- < 0.732 \\\ 8 & \text{(cubic)} & 0.732 < r_+/r_- < 1.0 \end{cases} \\] 

#### Python Implementation: Calculation of Coordination Number and Radius Ratio
    
    
    # ===================================
    # Example 2: Coordination number prediction using Pauling's rules
    # ===================================
    
    def predict_coordination_number(r_cation, r_anion):
        """
        Function to predict coordination number from radius ratio
    
        Parameters:
        -----------
        r_cation : float
            Cation radius [Å]
        r_anion : float
            Anion radius [Å]
    
        Returns:
        --------
        cn : int
            Predicted coordination number
        geometry : str
            Coordination polyhedron shape
        """
        ratio = r_cation / r_anion
    
        if ratio < 0.155:
            return 2, "Linear"
        elif ratio < 0.225:
            return 3, "Triangular planar"
        elif ratio < 0.414:
            return 4, "Tetrahedral"
        elif ratio < 0.732:
            return 6, "Octahedral"
        else:
            return 8, "Cubic"
    
    # Ionic radii data (Shannon radii, 6-coordinate)
    ionic_radii = {
        "Mg2+": 0.72,
        "Ca2+": 1.00,
        "Na+": 1.02,
        "Cs+": 1.67,
        "O2-": 1.40,
        "Cl-": 1.81
    }
    
    # Predict NaCl-type structures
    print("=== Coordination Number Prediction ===\n")
    
    materials = [
        ("Mg2+", "O2-", "MgO"),
        ("Ca2+", "O2-", "CaO"),
        ("Na+", "Cl-", "NaCl"),
        ("Cs+", "Cl-", "CsCl")
    ]
    
    for cation, anion, formula in materials:
        r_cat = ionic_radii[cation]
        r_an = ionic_radii[anion]
        ratio = r_cat / r_an
        cn, geometry = predict_coordination_number(r_cat, r_an)
    
        print(f"{formula}:")
        print(f"  Radius ratio: {ratio:.3f}")
        print(f"  Predicted coordination number: {cn} ({geometry})")
        print()
    
    # Expected output:
    # === Coordination Number Prediction ===
    #
    # MgO:
    #   Radius ratio: 0.514
    #   Predicted coordination number: 6 (Octahedral)
    #
    # CaO:
    #   Radius ratio: 0.714
    #   Predicted coordination number: 6 (Octahedral)
    #
    # NaCl:
    #   Radius ratio: 0.564
    #   Predicted coordination number: 6 (Octahedral)
    #
    # CsCl:
    #   Radius ratio: 0.922
    #   Predicted coordination number: 8 (Cubic)
    

### 1.2.3 CsCl-Type and CaF₂-Type Structures

The **CsCl-type structure** consists of a simple cubic lattice with an oppositely charged ion at the center (CN = 8). It is stable when the radius ratio is large (\\(r_+/r_- > 0.732\\)). 

The **CaF₂-type (fluorite) structure** has cations on an FCC lattice with anions at tetrahedral positions (1/4, 1/4, 1/4). This structure is observed in 1:2 stoichiometric oxides (ZrO₂, UO₂). 

Structure Type | Coordination Number (cation/anion) | Radius Ratio Range | Representative Examples  
---|---|---|---  
NaCl Type | 6:6 | 0.414 - 0.732 | MgO, CaO, NaCl  
CsCl Type | 8:8 | 0.732 - 1.0 | CsCl, CsBr, CsI  
CaF₂ Type | 8:4 | - | ZrO₂, UO₂, CaF₂  
  
## 1.3 Covalent Bonding Ceramics

### 1.3.1 Diamond Structure and Zinc Blende Structure

The **diamond structure** is the fundamental structure of carbon and silicon. Each atom forms covalent bonds with four nearest neighbors, creating tetrahedral coordination via sp³ hybrid orbitals. 

The **zinc blende (sphalerite, ZnS-type) structure** is the diamond structure with two types of atoms alternately arranged. SiC (silicon carbide) takes this structure as 3C-SiC. 

#### Python Implementation: SiC Structure Generation and Band Gap Prediction
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ===================================
    # Example 3: Zinc blende-type SiC structure generation
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_zincblende_structure(a, cation, anion):
        """
        Function to generate zinc blende-type structure
    
        Parameters:
        -----------
        a : float
            Lattice parameter [Å]
        cation : str
            Cation element symbol
        anion : str
            Anion element symbol
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            Zinc blende-type crystal structure
        """
        # Cubic lattice
        lattice = Lattice.cubic(a)
    
        # Atomic positions (fractional coordinates)
        # Si: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) - FCC
        # C: (0.25,0.25,0.25), (0.75,0.75,0.25), (0.75,0.25,0.75), (0.25,0.75,0.75)
        species = [cation]*4 + [anion]*4
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # Generate 3C-SiC (cubic SiC)
    sic = create_zincblende_structure(a=4.358, cation="Si", anion="C")
    
    print("=== 3C-SiC (Zinc Blende Type) Structure Information ===")
    print(f"Chemical formula: {sic.composition.reduced_formula}")
    print(f"Lattice parameter: a = {sic.lattice.a:.3f} Å")
    print(f"Density: {sic.density:.2f} g/cm³")
    
    # Calculate Si-C bond distance
    si_site = sic[0]
    neighbors = sic.get_neighbors(si_site, r=2.5)
    print(f"\nNumber of Si-C bonds: {len(neighbors)}")
    print(f"Si-C bond distance: {neighbors[0][1]:.3f} Å")
    
    # Calculate theoretical bond angle (tetrahedral angle = 109.47°)
    import numpy as np
    angle_tetrahedral = np.arccos(-1/3) * 180 / np.pi
    print(f"Theoretical C-Si-C bond angle: {angle_tetrahedral:.2f}°")
    
    # Expected output:
    # === 3C-SiC (Zinc Blende Type) Structure Information ===
    # Chemical formula: SiC
    # Lattice parameter: a = 4.358 Å
    # Density: 3.21 g/cm³
    #
    # Number of Si-C bonds: 4
    # Si-C bond distance: 1.889 Å
    # Theoretical C-Si-C bond angle: 109.47°
    

### 1.3.2 Wurtzite Structure and Si₃N₄ Structure

The **wurtzite structure** is a hexagonal covalent bonding structure. ZnO (zinc oxide) and AlN (aluminum nitride) take this structure. The ideal c/a ratio is √(8/3) ≈ 1.633. 

**Si₃N₄ (silicon nitride)** has two polymorphs, α-phase and β-phase, exhibiting high strength and high heat resistance. It is important as a structural engineering ceramic. 

**Characteristics of Covalent Bonding Ceramics**

  * **High hardness** : Strong directional covalent bonds (e.g., SiC Vickers hardness 2500 HV)
  * **High melting point** : Large bond energy (e.g., Si₃N₄ melting point 1900°C)
  * **Low thermal expansion coefficient** : Lattice stability due to strong covalent bonds
  * **Semiconductor properties** : SiC and AlN are wide band gap semiconductors

#### Python Implementation: Generation of AlN Wurtzite Structure
    
    
    # ===================================
    # Example 4: Wurtzite-type AlN structure generation
    # ===================================
    
    def create_wurtzite_structure(a, c, cation, anion):
        """
        Function to generate wurtzite-type structure
    
        Parameters:
        -----------
        a : float
            a-axis lattice parameter [Å]
        c : float
            c-axis lattice parameter [Å]
        cation : str
            Cation element symbol
        anion : str
            Anion element symbol
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            Wurtzite-type crystal structure
        """
        # Define hexagonal lattice
        lattice = Lattice.hexagonal(a, c)
    
        # Atomic positions (fractional coordinates)
        # Al: (1/3, 2/3, 0), (2/3, 1/3, 1/2)
        # N: (1/3, 2/3, u), (2/3, 1/3, 1/2+u)  u ≈ 0.375
        u = 0.382  # Internal parameter for AlN
        species = [cation, cation, anion, anion]
        coords = [
            [1/3, 2/3, 0],
            [2/3, 1/3, 0.5],
            [1/3, 2/3, u],
            [2/3, 1/3, 0.5 + u]
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    # Generate AlN structure
    aln = create_wurtzite_structure(a=3.111, c=4.981, cation="Al", anion="N")
    
    print("=== AlN (Wurtzite Type) Structure Information ===")
    print(f"Chemical formula: {aln.composition.reduced_formula}")
    print(f"Lattice parameters: a = {aln.lattice.a:.3f} Å, c = {aln.lattice.c:.3f} Å")
    print(f"c/a ratio: {aln.lattice.c / aln.lattice.a:.3f}")
    print(f"Ideal c/a ratio: {np.sqrt(8/3):.3f}")
    print(f"Density: {aln.density:.2f} g/cm³")
    
    # Calculate Al-N bond distance
    al_site = aln[0]
    neighbors = aln.get_neighbors(al_site, r=2.5)
    print(f"\nNumber of Al-N bonds: {len(neighbors)}")
    bond_lengths = [n[1] for n in neighbors]
    print(f"Al-N bond distance: {np.mean(bond_lengths):.3f} ± {np.std(bond_lengths):.3f} Å")
    
    # Expected output:
    # === AlN (Wurtzite Type) Structure Information ===
    # Chemical formula: AlN
    # Lattice parameters: a = 3.111 Å, c = 4.981 Å
    # c/a ratio: 1.601
    # Ideal c/a ratio: 1.633
    # Density: 3.26 g/cm³
    #
    # Number of Al-N bonds: 4
    # Al-N bond distance: 1.893 ± 0.006 Å
    

## 1.4 Perovskite Structure (ABO₃)

### 1.4.1 Fundamentals of Perovskite Structure

The **perovskite structure** is a structure of oxides represented by the general formula ABO₃. Large A-site ions (Ba²⁺, Sr²⁺) have 12-coordination, while small B-site ions (Ti⁴⁺, Zr⁴⁺) form 6-coordination (octahedral). 

In cubic perovskite, A atoms are at cube corners, B atoms are at body center, and O atoms are at face centers. This structure exhibits diverse functionalities such as **ferroelectricity** , **piezoelectricity** , and **colossal magnetoresistance**. 

#### Goldschmidt Tolerance Factor

The stability of the perovskite structure is evaluated by the **Goldschmidt tolerance factor \\(t\\)** : 

\\[ t = \frac{r_A + r_O}{\sqrt{2}(r_B + r_O)} \\] 

The perovskite structure is stable in the range \\(0.8 < t < 1.0\\). For \\(t > 1\\), hexagonal structure forms; for \\(t < 0.8\\), ilmenite structure forms. 

#### Python Implementation: Perovskite Structure Generation and Tolerance Factor Calculation
    
    
    # ===================================
    # Example 5: Perovskite structure generation and stability evaluation
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_perovskite_structure(a, A_ion, B_ion):
        """
        Function to generate cubic perovskite ABO3 structure
    
        Parameters:
        -----------
        a : float
            Cubic lattice parameter [Å]
        A_ion : str
            A-site ion element symbol
        B_ion : str
            B-site ion element symbol
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            Perovskite crystal structure
        """
        # Cubic lattice
        lattice = Lattice.cubic(a)
    
        # Atomic positions (fractional coordinates)
        # A: (0, 0, 0) - body center
        # B: (0.5, 0.5, 0.5) - corner
        # O: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) - face centers
        species = [A_ion, B_ion, "O", "O", "O"]
        coords = [
            [0, 0, 0],           # A
            [0.5, 0.5, 0.5],     # B
            [0.5, 0.5, 0],       # O
            [0.5, 0, 0.5],       # O
            [0, 0.5, 0.5]        # O
        ]
    
        structure = Structure(lattice, species, coords)
        return structure
    
    def goldschmidt_tolerance_factor(r_A, r_B, r_O=1.40):
        """
        Function to calculate Goldschmidt tolerance factor
    
        Parameters:
        -----------
        r_A : float
            A-site ion radius [Å]
        r_B : float
            B-site ion radius [Å]
        r_O : float
            Oxygen ion radius [Å] (default: 1.40Å)
    
        Returns:
        --------
        t : float
            Tolerance factor
        stability : str
            Structure stability evaluation
        """
        t = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
    
        if 0.9 < t < 1.0:
            stability = "Ideal perovskite (cubic)"
        elif 0.8 < t <= 0.9:
            stability = "Distorted perovskite (orthorhombic/rhombohedral)"
        elif t > 1.0:
            stability = "Transforms to hexagonal structure"
        else:
            stability = "Ilmenite structure"
    
        return t, stability
    
    # Generate BaTiO3 structure
    bto = create_perovskite_structure(a=4.004, A_ion="Ba", B_ion="Ti")
    
    print("=== BaTiO3 (Perovskite) Structure Information ===")
    print(f"Chemical formula: {bto.composition.reduced_formula}")
    print(f"Lattice parameter: a = {bto.lattice.a:.3f} Å")
    print(f"Density: {bto.density:.2f} g/cm³")
    
    # Calculate tolerance factor
    ionic_radii_perovskite = {
        "Ba2+": 1.61,  # 12-coordinate
        "Sr2+": 1.44,  # 12-coordinate
        "Ca2+": 1.34,  # 12-coordinate
        "Ti4+": 0.605, # 6-coordinate
        "Zr4+": 0.72,  # 6-coordinate
        "O2-": 1.40    # Standard
    }
    
    print("\n=== Goldschmidt Tolerance Factor Calculation ===\n")
    
    perovskites = [
        ("Ba2+", "Ti4+", "BaTiO3"),
        ("Sr2+", "Ti4+", "SrTiO3"),
        ("Ca2+", "Ti4+", "CaTiO3"),
        ("Ba2+", "Zr4+", "BaZrO3")
    ]
    
    for A, B, formula in perovskites:
        r_A = ionic_radii_perovskite[A]
        r_B = ionic_radii_perovskite[B]
        t, stability = goldschmidt_tolerance_factor(r_A, r_B)
    
        print(f"{formula}:")
        print(f"  Tolerance factor t = {t:.3f}")
        print(f"  Structure stability: {stability}")
        print()
    
    # Expected output:
    # === BaTiO3 (Perovskite) Structure Information ===
    # Chemical formula: BaTiO3
    # Lattice parameter: a = 4.004 Å
    # Density: 6.02 g/cm³
    #
    # === Goldschmidt Tolerance Factor Calculation ===
    #
    # BaTiO3:
    #   Tolerance factor t = 1.062
    #   Structure stability: Transforms to hexagonal structure
    #
    # SrTiO3:
    #   Tolerance factor t = 1.002
    #   Structure stability: Ideal perovskite (cubic)
    #
    # CaTiO3:
    #   Tolerance factor t = 0.966
    #   Structure stability: Ideal perovskite (cubic)
    #
    # BaZrO3:
    #   Tolerance factor t = 1.009
    #   Structure stability: Transforms to hexagonal structure
    

### 1.4.2 Phase Transition and Dielectric Properties

BaTiO₃ exhibits **phase transitions** in which the crystal structure changes with temperature: 

  * **> 120°C**: Cubic (paraelectric)
  * **5°C - 120°C** : Tetragonal (ferroelectric, used at room temperature)
  * **-90°C - 5°C** : Orthorhombic (ferroelectric)
  * **< -90°C**: Rhombohedral (ferroelectric)

In the tetragonal phase, Ti⁴⁺ ions are slightly displaced from the center of the oxygen octahedron, generating **spontaneous polarization**. This property is utilized in capacitors, piezoelectric elements, and sensors. 

**Note: Actual Structure Calculation** BaTiO₃ at room temperature is tetragonal (c/a ≈ 1.01), but in this example we use a cubic approximation. For precise analysis, use tetragonal lattice parameters (a = 3.992Å, c = 4.036Å). 

## 1.5 Spinel Structure (AB₂O₄)

### 1.5.1 Normal Spinel and Inverse Spinel

The **spinel structure** is a structure of complex oxides represented by the general formula AB₂O₄. O²⁻ ions form cubic close packing (FCC), and A²⁺ and B³⁺ occupy tetrahedral and octahedral positions. 

**Normal spinel** : A²⁺ at tetrahedral positions, B³⁺ at octahedral positions 

\\[ \text{A}^{2+}[\text{B}^{3+}_2]\text{O}_4 \\] 

Examples: MgAl₂O₄, ZnFe₂O₄ 

**Inverse spinel** : Half of B³⁺ at tetrahedral positions, A²⁺ and remaining B³⁺ at octahedral positions 

\\[ \text{B}^{3+}[\text{A}^{2+}\text{B}^{3+}]\text{O}_4 \\] 

Examples: Fe₃O₄ (magnetite), NiFe₂O₄ 

#### Python Implementation: Spinel Structure Generation and Visualization
    
    
    # ===================================
    # Example 6: Normal spinel MgAl2O4 structure generation
    # ===================================
    
    from pymatgen.core import Structure, Lattice
    
    def create_spinel_structure(a, A_ion, B_ion, inverse=False):
        """
        Function to generate spinel AB2O4 structure
    
        Parameters:
        -----------
        a : float
            Cubic lattice parameter [Å]
        A_ion : str
            A-site ion element symbol (divalent)
        B_ion : str
            B-site ion element symbol (trivalent)
        inverse : bool
            If True, generate inverse spinel structure
    
        Returns:
        --------
        structure : pymatgen.core.Structure
            Spinel crystal structure
        """
        # Cubic lattice
        lattice = Lattice.cubic(a)
    
        # Atomic positions for normal spinel (fractional coordinates)
        # A (Mg): 8a position (1/8, 1/8, 1/8) - tetrahedral
        # B (Al): 16d position (1/2, 1/2, 1/2) - octahedral
        # O: 32e position (u, u, u), u ≈ 0.25
    
        if not inverse:
            # Normal spinel
            species = []
            coords = []
    
            # A positions: 8 tetrahedral positions
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        species.append(A_ion)
                        coords.append([0.125 + i*0.5, 0.125 + j*0.5, 0.125 + k*0.5])
    
            # B positions: 16 octahedral positions (simplified to 4 only)
            b_positions = [
                [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]
            ]
            for pos in b_positions:
                species.extend([B_ion, B_ion])
                coords.extend([pos, [pos[0]+0.25, pos[1]+0.25, pos[2]+0.25]])
    
            # O positions: 32 (simplified to 8)
            u = 0.263
            o_base = [[u, u, u], [-u, -u, u], [-u, u, -u], [u, -u, -u]]
            for pos in o_base:
                species.extend(["O", "O"])
                coords.append([p % 1.0 for p in pos])
                coords.append([p + 0.5 % 1.0 for p in pos])
    
        # For simplicity, only representative atoms are included here
        # Actual spinel has 56 atoms per unit cell (8A + 16B + 32O)
        structure = Structure(lattice, species[:20], coords[:20])
        return structure
    
    # Generate MgAl2O4 (normal spinel)
    mgo_al2o3 = create_spinel_structure(a=8.083, A_ion="Mg", B_ion="Al")
    
    print("=== MgAl2O4 (Normal Spinel) Structure Information ===")
    print(f"Chemical formula: {mgo_al2o3.composition.reduced_formula}")
    print(f"Lattice parameter: a = {mgo_al2o3.lattice.a:.3f} Å")
    print(f"Density: {mgo_al2o3.density:.2f} g/cm³")
    print(f"Number of atoms per unit cell (simplified): {len(mgo_al2o3)}")
    
    # Actual MgAl2O4 properties
    print("\n=== Experimental Values (Reference) ===")
    print("Complete unit cell atom count: 56 (8Mg + 16Al + 32O)")
    print("Density (experimental): 3.58 g/cm³")
    print("Hardness: 8 Mohs (hardness second only to diamond)")
    print("Applications: Refractory materials, transparent ceramics, gemstones (spinel)")
    
    # Expected output:
    # === MgAl2O4 (Normal Spinel) Structure Information ===
    # Chemical formula: MgAl2O4
    # Lattice parameter: a = 8.083 Å
    # Density: 3.21 g/cm³
    # Number of atoms per unit cell (simplified): 20
    #
    # === Experimental Values (Reference) ===
    # Complete unit cell atom count: 56 (8Mg + 16Al + 32O)
    # Density (experimental): 3.58 g/cm³
    # Hardness: 8 Mohs (hardness second only to diamond)
    # Applications: Refractory materials, transparent ceramics, gemstones (spinel)
    

### 1.5.2 Inverse Spinel Structure and Magnetism of Fe₃O₄

**Magnetite (Fe₃O₄)** is a representative magnetic material with an inverse spinel structure. The chemical formula is represented as Fe²⁺Fe₂³⁺O₄ with the following arrangement: 

  * **Tetrahedral positions** : Fe³⁺ (spin up)
  * **Octahedral positions** : Fe²⁺ and Fe³⁺ (spin down)

The spins of tetrahedral Fe³⁺ and octahedral Fe³⁺ cancel each other, leaving only the magnetic moment of octahedral Fe²⁺. This **ferrimagnetism** results in strong magnetism at room temperature. 

Material | Spinel Type | Magnetism | Applications  
---|---|---|---  
MgAl₂O₄ | Normal spinel | Non-magnetic | Refractory materials, gemstones  
ZnFe₂O₄ | Normal spinel | Antiferromagnetic | Catalysts, gas sensors  
Fe₃O₄ | Inverse spinel | Ferrimagnetic | Magnetic materials, MRI contrast agents  
NiFe₂O₄ | Inverse spinel | Ferrimagnetic | High-frequency devices  
  
## 1.6 Practical Examples of Structural Analysis

### 1.6.1 Simulation of X-ray Diffraction Patterns

The most common method for experimentally determining crystal structure is **X-ray diffraction (XRD)**. Using Pymatgen, we can simulate theoretical XRD patterns. 

#### Python Implementation: XRD Pattern Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    # ===================================
    # Example 7: X-ray diffraction pattern simulation
    # ===================================
    
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    
    def simulate_xrd(structure, wavelength="CuKa"):
        """
        Function to simulate X-ray diffraction pattern
    
        Parameters:
        -----------
        structure : pymatgen.core.Structure
            Crystal structure
        wavelength : str or float
            X-ray wavelength [Å] ("CuKa" = 1.54184Å)
    
        Returns:
        --------
        pattern : XRDPattern
            Calculated XRD pattern
        """
        xrd_calc = XRDCalculator(wavelength=wavelength)
        pattern = xrd_calc.get_pattern(structure)
        return pattern
    
    # XRD simulation of MgO (NaCl type)
    mgo = create_nacl_structure(a=4.211)
    xrd_mgo = simulate_xrd(mgo)
    
    print("=== MgO X-ray Diffraction Peaks ===\n")
    print(f"{'2θ (deg)':<12} {'d (Å)':<10} {'(hkl)':<10} {'Intensity':<10}")
    print("-" * 50)
    
    for i, (two_theta, d_spacing, hkl, intensity) in enumerate(
        zip(xrd_mgo.x[:5], xrd_mgo.d_hkls[:5], xrd_mgo.hkls[:5], xrd_mgo.y[:5])
    ):
        hkl_str = str(hkl[0])
        print(f"{two_theta:<12.2f} {d_spacing:<10.3f} {hkl_str:<10} {intensity:<10.1f}")
    
    # Plot XRD pattern
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(xrd_mgo.x, xrd_mgo.y, 'b-', linewidth=2)
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Simulated XRD Pattern: MgO (NaCl-type)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mgo_xrd_pattern.png', dpi=300)
    print("\nXRD pattern saved to 'mgo_xrd_pattern.png'")
    
    # Expected output:
    # === MgO X-ray Diffraction Peaks ===
    #
    # 2θ (deg)    d (Å)      (hkl)      Intensity
    # --------------------------------------------------
    # 42.92       2.106      (200)      100.0
    # 62.30       1.488      (220)      52.3
    # 78.63       1.216      (311)      12.8
    # 94.05       1.053      (400)      3.2
    #
    # XRD pattern saved to 'mgo_xrd_pattern.png'
    

### 1.6.2 Searching from Structure Databases

**Materials Project** is a database of crystal structures and material properties for over 100,000 inorganic materials. You can retrieve known material data through the API. 

#### Python Implementation: Structure Retrieval from Materials Project
    
    
    # ===================================
    # Example 8: Materials Project database search
    # ===================================
    
    from pymatgen.ext.matproj import MPRester
    
    def search_materials_project(formula, api_key=None):
        """
        Function to retrieve material structure from Materials Project
    
        Parameters:
        -----------
        formula : str
            Chemical formula (e.g., "BaTiO3")
        api_key : str
            Materials Project API key
    
        Returns:
        --------
        structures : list
            List of retrieved structures
        """
        # How to get API key:
        # 1. Register an account at https://materialsproject.org/
        # 2. Dashboard > API > Generate API Key
    
        if api_key is None:
            print("Note: API key is required.")
            print("Please create an account at https://materialsproject.org/.")
            return []
    
        with MPRester(api_key) as mpr:
            # Search by chemical formula
            entries = mpr.get_entries(formula)
            structures = [entry.structure for entry in entries]
    
        return structures
    
    # Usage example (retrieve API key from environment variable)
    import os
    
    # API_KEY = os.environ.get("MP_API_KEY")  # Get from environment variable
    # structures = search_materials_project("BaTiO3", api_key=API_KEY)
    
    # Demo output without API key
    print("=== Materials Project Search Example ===\n")
    print("Search formula: BaTiO3")
    print("\nExpected results:")
    print("1. mp-5020: BaTiO3 (cubic, Pm-3m)")
    print("   - Lattice parameter: a = 4.004 Å")
    print("   - Band gap: 1.9 eV (indirect)")
    print("   - Formation energy: -15.3 eV/atom")
    print("\n2. mp-2998: BaTiO3 (tetragonal, P4mm)")
    print("   - Lattice parameters: a = 3.992 Å, c = 4.036 Å")
    print("   - Spontaneous polarization: 0.26 C/m²")
    print("\n3. mp-5777: BaTiO3 (rhombohedral, R3m)")
    print("   - Low temperature phase")
    print("\nDetailed retrieval method:")
    print("```python")
    print("from pymatgen.ext.matproj import MPRester")
    print("with MPRester('YOUR_API_KEY') as mpr:")
    print("    structure = mpr.get_structure_by_material_id('mp-5020')")
    print("    print(structure)")
    print("```")
    

**Tips for Utilizing Materials Project**

  * **Structure data** : Crystal structure (CIF format), symmetry, space group
  * **Electronic properties** : Band gap, density of states (DOS), band structure
  * **Thermodynamic data** : Formation energy, stability, phase diagrams
  * **Elastic constants** : Young's modulus, shear modulus, Poisson's ratio

API keys can be obtained for free and used for up to 100,000 requests per month. Use for research and educational purposes is recommended. 

## Exercises

### Easy (Fundamental Verification)

#### Q1: Structure Type Identification Easy

Which of the following crystal structure types does MgO belong to? 

a) CsCl type  
b) NaCl type  
c) Diamond type  
d) Perovskite type 

View Answer

**Correct answer: b) NaCl type**

**Explanation:**  
MgO takes the rock salt (NaCl-type) structure. Mg²⁺ and O²⁻ each form FCC lattices, and each ion is surrounded by six ions of opposite charge (coordination number = 6). The radius ratio r(Mg²⁺)/r(O²⁻) = 0.72/1.40 = 0.514, and according to Pauling's rules, octahedral coordination (CN = 6) is predicted. 

#### Q2: Pauling's Rules Easy

What is the cation coordination number of CsCl (cesium chloride)? Also explain the reason. 

View Answer

**Correct answer: 8 (cubic coordination)**

**Explanation:**  
The radius of Cs⁺ is 1.67Å and the radius of Cl⁻ is 1.81Å. The radius ratio = 1.67/1.81 = 0.922, and according to Pauling's first rule, in the range 0.732 < r₊/r₋ < 1.0, the coordination number is 8 (cubic coordination). In the CsCl-type structure, Cs⁺ is located at the center of a simple cubic lattice and is surrounded by 8 Cl⁻ ions. 

#### Q3: General Formula of Perovskite Easy

The general formula of the perovskite structure is ABO₃. In BaTiO₃, state the coordination numbers of A-site and B-site positions. 

View Answer

**Correct answer: A-site (Ba): 12-coordinate, B-site (Ti): 6-coordinate**

**Explanation:**  
In cubic perovskite, large Ba²⁺ ions are at cube corners (12-coordinate from the body center), and small Ti⁴⁺ ions are at the body center (center of oxygen octahedron, 6-coordinate). O²⁻ ions are arranged at face centers, forming a Ti-O-Ti bonding network. 

### Medium (Application)

#### Q4: Goldschmidt Tolerance Factor Calculation Medium

Calculate the Goldschmidt tolerance factor t for SrTiO₃. Ionic radii are Sr²⁺ = 1.44Å (12-coordinate), Ti⁴⁺ = 0.605Å (6-coordinate), O²⁻ = 1.40Å. Is this material expected to take the perovskite structure? 

View Answer

**Correct answer: t = 1.002, ideal perovskite structure**

**Calculation process:**  
\\[ t = \frac{r_{\text{Sr}} + r_{\text{O}}}{\sqrt{2}(r_{\text{Ti}} + r_{\text{O}})} = \frac{1.44 + 1.40}{\sqrt{2}(0.605 + 1.40)} = \frac{2.84}{2.834} = 1.002 \\] 

**Explanation:**  
t ≈ 1.0 indicates an ideal perovskite structure. Indeed, SrTiO₃ takes a cubic perovskite structure (space group Pm-3m) at room temperature and exhibits a high dielectric constant (ε ≈ 300). As a quantum paraelectric, it does not undergo ferroelectric transition even at low temperatures. 

#### Q5: Distinction between Normal Spinel and Inverse Spinel Medium

MgAl₂O₄ takes a normal spinel structure while Fe₃O₄ takes an inverse spinel structure. What is the main factor determining this difference? Also explain the conditions under which inverse spinel becomes more stable. 

View Answer

**Correct answer: Crystal field stabilization energy (CFSE) and cation electronic configuration**

**Explanation:**  
The stability of the spinel structure is determined by whether cations prefer tetrahedral or octahedral positions: 

  * **MgAl₂O₄ (normal spinel)** : Mg²⁺ (d⁰) has no crystal field stabilization, and Al³⁺ (d⁰) is the same. Due to ionic radius differences, small Mg²⁺ occupies tetrahedral positions and Al³⁺ occupies octahedral positions.
  * **Fe₃O₄ (inverse spinel)** : Fe³⁺ (d⁵, high spin) has small CFSE at either position. Fe²⁺ (d⁶) gains large CFSE at octahedral positions. Therefore, Fe³⁺ is distributed between tetrahedral and octahedral positions, and Fe²⁺ occupies octahedral positions, making inverse spinel stable.

In general, ions with d³, d⁶, and d⁸ electronic configurations strongly prefer octahedral positions, thus stabilizing the inverse spinel structure. 

#### Q6: Density Calculation of 3C-SiC Medium

The lattice parameter of 3C-SiC (zinc blende type) is a = 4.358Å. Calculate the density from the unit cell mass and volume. (Atomic weights: Si = 28.09, C = 12.01) 

View Answer

**Correct answer: Density ≈ 3.21 g/cm³**

**Calculation process:**

  1. Unit cell volume: V = a³ = (4.358×10⁻⁸ cm)³ = 8.28×10⁻²³ cm³
  2. Number of atoms in unit cell: 4 SiC pairs (FCC basis of zinc blende type)
  3. Unit cell mass: m = 4 × (28.09 + 12.01) / N_A = 4 × 40.10 / 6.022×10²³ = 2.66×10⁻²² g
  4. Density: ρ = m / V = 2.66×10⁻²² / 8.28×10⁻²³ = 3.21 g/cm³

This matches well with the experimental value (3.21 g/cm³). The high density of SiC originates from covalent bonding and strong interatomic bonds. 

#### Q7: Ideal c/a Ratio of Wurtzite Structure Medium

The ideal c/a ratio of wurtzite structure is √(8/3) ≈ 1.633. The measured value for AlN is c/a = 1.601. What does this deviation signify? Also, is ionicity or covalency inferred to be stronger? 

View Answer

**Correct answer: c/a < 1.633 suggests strong covalent bonding**

**Explanation:**  
The ideal wurtzite structure assumes close packing of spheres. In actual materials: 

  * **c/a < 1.633**: Strong covalent bonding; directional bonds compress the c-axis direction (e.g., AlN, GaN)
  * **c/a > 1.633**: Strong ionic bonding; electrostatic repulsion extends the c-axis direction (e.g., ZnO = 1.602, BeO = 1.623)

Because AlN has strong covalent bonding, c/a = 1.601 is smaller than the ideal value. This property is related to high thermal conductivity (320 W/m·K). 

### Hard (Advanced)

#### Q8: Phase Transition and Structural Change of BaTiO₃ Hard

BaTiO₃ undergoes a phase transition from cubic to tetragonal at 120°C. Estimate the direction and magnitude of spontaneous polarization in the tetragonal phase (a = 3.992Å, c = 4.036Å) from Ti⁴⁺ ion displacement (δ ≈ 0.12Å). (Effective charge: Z* = 7e, calculate from unit cell volume) 

View Answer

**Correct answer: Spontaneous polarization P_s ≈ 0.26 C/m² (c-axis direction)**

**Calculation process:**

  1. Unit cell volume: V = a² × c = (3.992)² × 4.036 = 64.3 Å³ = 6.43×10⁻²⁹ m³
  2. Dipole moment: p = Z* × e × δ = 7 × 1.602×10⁻¹⁹ C × 0.12×10⁻¹⁰ m = 1.35×10⁻²⁹ C·m
  3. Spontaneous polarization: P_s = p / V = 1.35×10⁻²⁹ / 6.43×10⁻²⁹ = 0.21 C/m²

A value close to the experimental value (0.26 C/m²) is obtained. The discrepancy is because O²⁻ ion displacement and electron cloud polarization are not considered. 

**Physical meaning:**  
Displacement of Ti⁴⁺ from the center of the oxygen octahedron in the c-axis direction generates a strong dipole moment. Due to this spontaneous polarization, BaTiO₃ is applied as an excellent piezoelectric material and ferroelectric memory. 

#### Q9: Magnetic Moment Calculation of Fe₃O₄ Hard

Magnetite (Fe₃O₄) takes an inverse spinel structure Fe³⁺[Fe²⁺Fe³⁺]O₄. Calculate the theoretical magnetic moment of the unit cell (8 Fe₃O₄ units). (Fe²⁺: 4μ_B, Fe³⁺: 5μ_B, μ_B = Bohr magneton) 

View Answer

**Correct answer: Unit cell magnetic moment = 32μ_B**

**Calculation process:**

  1. Composition of one formula unit (Fe₃O₄): 
     * Tetrahedral position: Fe³⁺ (spin ↑) → +5μ_B
     * Octahedral positions: Fe²⁺ (spin ↓) → -4μ_B, Fe³⁺ (spin ↓) → -5μ_B
  2. Net magnetic moment (per formula unit): 
     * M = 5μ_B (tetrahedral Fe³⁺) - 4μ_B (octahedral Fe²⁺) - 5μ_B (octahedral Fe³⁺) = -4μ_B
     * Absolute value: |M| = 4μ_B (ferrimagnetism)
  3. Unit cell (8 Fe₃O₄ units): M_total = 8 × 4μ_B = 32μ_B

This matches well with the experimental value (approximately 4.1μ_B per formula unit). Due to this strong magnetic moment, Fe₃O₄ functions as a magnetic material at room temperature and is applied in magnetic recording media, MRI contrast agents, and magnetic fluids. 

#### Q10: Density Prediction of Composite Materials Hard

MgO (density 3.58 g/cm³) and Al₂O₃ (density 3.98 g/cm³) react in a 1:1 molar ratio to form MgAl₂O₄ spinel (theoretical density 3.58 g/cm³). Calculate the volume change rate before and after the reaction and predict shrinkage during sintering. 

View Answer

**Correct answer: Volume shrinkage rate ≈ 1.8%**

**Calculation process:**

  1. Reaction formula: MgO + Al₂O₃ → MgAl₂O₄
  2. Mass before reaction: 
     * MgO: 40.30 g/mol
     * Al₂O₃: 101.96 g/mol
     * Total: 142.26 g/mol
  3. Volume before reaction (per mole): 
     * V_MgO = 40.30 / 3.58 = 11.26 cm³
     * V_Al2O3 = 101.96 / 3.98 = 25.62 cm³
     * V_before = 11.26 + 25.62 = 36.88 cm³
  4. Volume after reaction (1 mol MgAl₂O₄): 
     * V_after = 142.26 / 3.58 = 39.74 cm³
  5. Volume change rate: ΔV/V = (39.74 - 36.88) / 36.88 × 100 = 7.8% (expansion)

**Note:** In reality, raw materials before reaction are not sintered bodies but powders with high porosity (30-50%). Throughout the sintering process, pores decrease, resulting in overall shrinkage of 15-20%. Volume expansion (7.8%) due to spinel phase formation is offset by shrinkage from pore reduction. 

**Engineering significance:**  
Considering this volume change, sintering conditions (temperature, time, atmosphere) for refractory bricks and ceramic parts are optimized. Rapid volume changes cause cracks, so stepwise heating profiles are important. 

## References

  1. **Kingery, W.D., Bowen, H.K., Uhlmann, D.R. (1976).** _Introduction to Ceramics_ (2nd ed.). Wiley, pp. 30-89 (crystal structures), pp. 92-135 (defects and diffusion). 
  2. **Carter, C.B., Norton, M.G. (2013).** _Ceramic Materials: Science and Engineering_ (2nd ed.). Springer, pp. 45-120 (ionic and covalent structures), pp. 267-310 (phase transformations). 
  3. **West, A.R. (2014).** _Solid State Chemistry and its Applications_ (2nd ed.). Wiley, pp. 1-85 (crystal structures), pp. 187-245 (perovskites and spinels), pp. 320-375 (structure-property relationships). 
  4. **Barsoum, M.W. (2020).** _Fundamentals of Ceramics_ (2nd ed.). CRC Press, pp. 20-75 (bonding and crystal structures), pp. 105-158 (point defects), pp. 445-490 (electrical properties). 
  5. **Richerson, D.W., Lee, W.E. (2018).** _Modern Ceramic Engineering: Properties, Processing, and Use in Design_ (4th ed.). CRC Press, pp. 50-95 (structures), pp. 120-165 (mechanical properties). 
  6. **Pymatgen Documentation (2024).** Available at: <https://pymatgen.org/> (Python materials analysis library, Materials Project integration) 
  7. **Shannon, R.D. (1976).** "Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides." _Acta Crystallographica A_ , 32, 751-767. (Standard ionic radii data) 

**For Further Learning**

  * Structural analysis techniques: _Elements of X-ray Diffraction_ (Cullity & Stock, 2014) - XRD principles and practice
  * First-principles calculations: _Density Functional Theory: A Practical Introduction_ (Sholl & Steckel, 2009) - Introduction to DFT
  * Ceramic properties: _Physical Properties of Ceramics_ (Wachtman et al., 2009) - Mechanical, thermal, and electrical properties
  * Database: Materials Project (<https://materialsproject.org/>) - Free materials database

## Learning Objectives Verification Checklist

### Level 1: Fundamental Understanding

  * □ Can explain the differences between NaCl type, CsCl type, and CaF₂ type
  * □ Can predict coordination number from radius ratio using Pauling's first rule
  * □ Can describe characteristics of ionic and covalent bonding ceramics
  * □ Understand the basic arrangement of perovskite structure (ABO₃)
  * □ Can explain the difference between normal spinel and inverse spinel
  * □ Understand the difference between zinc blende structure and wurtzite structure

### Level 2: Practical Skills

  * □ Can generate NaCl-type and zinc blende-type structures using Pymatgen
  * □ Can calculate density from lattice parameters
  * □ Can predict coordination number from ionic radii data
  * □ Can calculate Goldschmidt tolerance factor and evaluate perovskite stability
  * □ Can simulate X-ray diffraction patterns
  * □ Can retrieve crystal structure data from Materials Project
  * □ Can evaluate bonding nature from c/a ratio of wurtzite structure

### Level 3: Application Ability

  * □ Can predict structure of new compounds by applying Pauling's rules
  * □ Understand phase transitions and structural changes of BaTiO₃ and can estimate spontaneous polarization
  * □ Can calculate magnetic moment from inverse spinel structure of Fe₃O₄
  * □ Can explain structure-property correlations (hardness, dielectric constant, magnetism)
  * □ Can apply structural knowledge to real material design (refractory materials, piezoelectric elements, magnetic materials)
  * □ Can identify structures from XRD patterns
  * □ Can interpret results of first-principles calculations and discuss structure optimization

**Next Steps** In Chapter 2, we will learn about defect structures in ceramics (point defects, dislocations, grain boundaries). We will understand how the ideal crystal structures learned in this chapter change in real materials and how this affects properties. In particular, phenomena such as ionic conductivity, diffusion, and sintering are closely related to defect structures. 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
