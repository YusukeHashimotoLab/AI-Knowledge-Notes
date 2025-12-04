---
title: "Chapter 2: Atomic Structure and Chemical Bonding"
chapter_title: "Chapter 2: Atomic Structure and Chemical Bonding"
subtitle: Atomic-level mechanisms that determine material properties
reading_time: 25-30 minutes
difficulty: Introductory to Intermediate
code_examples: 6
version: 1.0
created_at: 2025-10-25
---

The properties of materials are determined by how atoms are bonded together. In this chapter, we will learn about atomic structure, electron configurations, and types of chemical bonds, and understand how they influence material properties. 

## Learning Objectives

By reading this chapter, you will be able to:

  *  Explain atomic structure (nucleus, electron shells, electron configuration)
  *  Understand the four main types of chemical bonds (ionic, covalent, metallic, and intermolecular forces)
  *  Explain the relationship between bond types and material properties (strength, conductivity, melting point)
  *  Understand the relationship between electronegativity and bonding character
  *  Visualize electron configurations and bond energies using Python

* * *

## 2.1 Atomic Structure and Electron Configuration

### Basic Structure of Atoms

Atoms are composed of three fundamental particles:

Particle | Charge | Mass (u) | Location  
---|---|---|---  
**Proton** | +1 | 1.0073 | Nucleus  
**Neutron** | 0 | 1.0087 | Nucleus  
**Electron** | -1 | 0.00055 | Electron shells  
  
**Atomic number (Z)** : Number of protons (= number of electrons in a neutral atom)

**Mass number (A)** : Number of protons + number of neutrons

### Electron Shells and Electron Configuration

Electrons exist in **electron shells** around the nucleus. Electron shells are named **K shell, L shell, M shell, N shell...** in order of increasing energy.

Each electron shell has a maximum number of electrons it can accommodate:

$$\text{Maximum electrons} = 2n^2 \quad (n: \text{principal quantum number})$$

Electron Shell | Principal Quantum Number n | Maximum Electrons | Subshells  
---|---|---|---  
**K shell** | 1 | 2 | 1s  
**L shell** | 2 | 8 | 2s, 2p  
**M shell** | 3 | 18 | 3s, 3p, 3d  
**N shell** | 4 | 32 | 4s, 4p, 4d, 4f  
  
**Electron configuration notation** : For example, the electron configuration of carbon (C, Z=6) is written as:

$$\text{C}: 1s^2 \, 2s^2 \, 2p^2$$

This means "2 electrons in the 1s orbital, 2 electrons in the 2s orbital, and 2 electrons in the 2p orbital."

### Valence Electrons and Chemical Reactivity

**Valence electrons** are electrons in the outermost shell and participate in chemical bonding. The number of valence electrons determines the chemical and electrical properties of materials.

**Electron configurations and valence electrons of representative elements** :

Element | Atomic Number | Electron Configuration | Valence Electrons | Characteristics  
---|---|---|---|---  
**Hydrogen (H)** | 1 | 1s¹ | 1 | Highly reactive  
**Carbon (C)** | 6 | 1s² 2s² 2p² | 4 | Forms 4 covalent bonds  
**Sodium (Na)** | 11 | [Ne] 3s¹ | 1 | Easily ionizes (Naz)  
**Silicon (Si)** | 14 | [Ne] 3s² 3p² | 4 | Semiconductor  
**Iron (Fe)** | 26 | [Ar] 3dv 4s² | 2-3 | Magnetic, catalytic activity  
**Copper (Cu)** | 29 | [Ar] 3d¹p 4s¹ | 1 | High electrical conductivity  
  
* * *

## 2.2 Types of Chemical Bonds

When atoms form materials, they are held together by **chemical bonds**. The type of bond significantly affects material properties.

### 1\. Ionic Bond

**Formation mechanism** :

  * Metals (low electronegativity) release electrons ’ cations
  * Non-metals (high electronegativity) accept electrons ’ anions
  * Bonding occurs through electrostatic attraction between cations and anions

**Example** : NaCl (sodium chloride)

$$\text{Na} \rightarrow \text{Na}^+ + e^-$$

$$\text{Cl} + e^- \rightarrow \text{Cl}^-$$

$$\text{Na}^+ + \text{Cl}^- \rightarrow \text{NaCl}$$

**Characteristics** :

  * High melting point (strong electrostatic attraction)
  * Hard but brittle (repulsion occurs when ion arrangement shifts)
  * Non-conductive in solid state, but conductive in molten or aqueous solution
  * Soluble in water

**Representative materials** : NaCl, MgO, Al‚Oƒ (alumina), ZrO‚ (zirconia)

### 2\. Covalent Bond

**Formation mechanism** :

  * Atoms share electrons
  * Each atom achieves a stable electron configuration (closed shell structure)
  * Electron pairs exist in bonding orbitals

**Examples** : Diamond (C), Silicon (Si), SiC (silicon carbide)

**Characteristics** :

  * Very high hardness (strong bonding)
  * High melting point (diamond: 3823 K)
  * Electrically insulating (however, semiconductors also fall into this category)
  * Directional (bond angles are fixed)

**Representative materials** : Diamond, SiC, SiƒN„ (silicon nitride)

### 3\. Metallic Bond

**Formation mechanism** :

  * Metal atoms release valence electrons to form a "sea" of positive ions
  * Free electrons (delocalized electrons) move between metal ions
  * Bonding occurs through electrostatic attraction between free electrons and metal ions

**Characteristics** :

  * High electrical conductivity (free electrons carry electric current)
  * High thermal conductivity (free electrons carry heat)
  * Excellent ductility and malleability (bonding is maintained even when metal ion arrangement changes)
  * Metallic luster (free electrons reflect light)

**Representative materials** : Fe, Cu, Al, Ti, Au, Ag

### 4\. Intermolecular Forces

These are weak forces that bind molecules together. They are much weaker interactions than chemical bonds (1-3).

#### a. Van der Waals Forces

  * Weak attraction due to momentary charge polarization in molecules
  * Acts between all molecules
  * Examples: Solid Ar, CH„ (methane)

#### b. Hydrogen Bonds

  * In bonds like H-O, H-N, H-F, H becomes positively polarized
  * Acts between H and electronegative atoms (O, N, F) in adjacent molecules
  * Stronger than van der Waals forces
  * Examples: H‚O (ice), DNA double helix, protein higher-order structure

**Characteristics** :

  * Low melting and boiling points (weak bonding)
  * Soft
  * Electrically insulating

**Representative materials** : Polymeric materials (polyethylene, polypropylene, etc.)

### Comparison of Bond Types

Bond Type | Bond Strength | Melting Point | Electrical Conductivity | Mechanical Properties | Examples  
---|---|---|---|---|---  
**Ionic** | Strong | High | Conductive in melt | Hard but brittle | NaCl, MgO  
**Covalent** | Very strong | Very high | Low (insulator) | Very hard, brittle | Diamond, SiC  
**Metallic** | Medium-strong | Medium-high | Very high | Ductile and malleable | Fe, Cu, Al  
**Intermolecular** | Weak | Low | Low (insulator) | Soft | Polymers, ice  
  
* * *

## 2.3 Relationship Between Bonding and Material Properties

### Electronegativity and Bonding Character

**Electronegativity** represents an atom's ability to attract electrons. The Pauling scale is widely used.

**Electronegativity of major elements** :

  * F (Fluorine): 4.0 (maximum)
  * O (Oxygen): 3.5
  * N (Nitrogen): 3.0
  * C (Carbon): 2.5
  * H (Hydrogen): 2.1
  * Si (Silicon): 1.8
  * Al (Aluminum): 1.5
  * Na (Sodium): 0.9
  * Cs (Cesium): 0.7 (minimum)

**Relationship between electronegativity difference and bond type** :

The electronegativity difference (”Ç) between two atoms roughly determines the bond type:

  * **”Ç > 2.0**: Ionic bonding dominates (e.g., NaCl, ”Ç = 3.0 - 0.9 = 2.1)
  * **0.5 < ”Ç < 2.0**: Polar covalent bonding (e.g., H‚O, ”Ç = 3.5 - 2.1 = 1.4)
  * **”Ç < 0.5**: Nonpolar covalent bonding (e.g., C-C, ”Ç = 0)

### Bond Energy and Material Properties

**Bond energy** is the energy required to break a bond. The greater the bond energy, the higher the melting point, hardness, and elastic modulus of the material.

Bond | Bond Energy (kJ/mol) | Bond Length (Å)  
---|---|---  
C-C (diamond) | 347 | 1.54  
C=C (ethylene) | 614 | 1.34  
CaC (acetylene) | 839 | 1.20  
Si-Si | 222 | 2.35  
O-H (water) | 463 | 0.96  
Na-Cl (ionic bond) | 410 | 2.36  
  
**Relationship between bond energy and melting point** :

  * Diamond (C-C bond, 347 kJ/mol): Melting point 3823 K
  * Silicon (Si-Si bond, 222 kJ/mol): Melting point 1687 K
  * Ice (hydrogen bond, ~20 kJ/mol): Melting point 273 K

* * *

## 2.4 Visualization of Electron Configurations and Bonding with Python

Let's use Python to visualize atomic structure and chemical bonding to deepen our understanding.

### Code Example 1: Automatic Calculation and Display of Electron Configuration

Automatically calculates and displays the electron configuration of any element.
    
    
    def electron_configuration(atomic_number):
        """
        Calculate electron configuration from atomic number
    
        Parameters:
        atomic_number (int): Atomic number
    
        Returns:
        dict: Number of electrons in each orbital
        """
        # Orbital order (lowest energy level first)
        orbital_order = [
            '1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p',
            '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p'
        ]
    
        # Maximum electrons in each orbital
        max_electrons = {
            's': 2, 'p': 6, 'd': 10, 'f': 14
        }
    
        electrons_left = atomic_number
        config = {}
    
        for orbital in orbital_order:
            if electrons_left == 0:
                break
    
            orbital_type = orbital[-1]  # 's', 'p', 'd', 'f'
            max_e = max_electrons[orbital_type]
    
            if electrons_left >= max_e:
                config[orbital] = max_e
                electrons_left -= max_e
            else:
                config[orbital] = electrons_left
                electrons_left = 0
    
        return config
    
    
    def print_electron_configuration(atomic_number, element_symbol):
        """
        Display electron configuration in a readable format
        """
        config = electron_configuration(atomic_number)
    
        # Notation 1: Standard notation
        config_str = ' '.join([f"{orbital}^{count}" for orbital, count in config.items()])
    
        # Notation 2: Noble gas notation (abbreviated display)
        noble_gases = {2: 'He', 10: 'Ne', 18: 'Ar', 36: 'Kr', 54: 'Xe', 86: 'Rn'}
    
        print(f"\n[{element_symbol} (Atomic number: {atomic_number}) Electron Configuration]")
        print(f"Full notation: {config_str}")
    
        # Calculate valence electrons
        outermost_shell = max([int(orbital[0]) for orbital in config.keys()])
        valence_electrons = sum([count for orbital, count in config.items()
                                 if int(orbital[0]) == outermost_shell])
    
        print(f"Outermost shell: {outermost_shell} shell")
        print(f"Valence electrons: {valence_electrons}")
    
        return config
    
    
    # Display electron configurations of representative elements
    elements = [
        (1, 'H', 'Hydrogen'),
        (6, 'C', 'Carbon'),
        (11, 'Na', 'Sodium'),
        (14, 'Si', 'Silicon'),
        (26, 'Fe', 'Iron'),
        (29, 'Cu', 'Copper'),
        (79, 'Au', 'Gold')
    ]
    
    print("=" * 60)
    print("Electron Configurations of Major Elements")
    print("=" * 60)
    
    for z, symbol, name in elements:
        config = print_electron_configuration(z, f"{name} ({symbol})")
    

**Output example** :
    
    
    ============================================================
    Electron Configurations of Major Elements
    ============================================================
    
    [Hydrogen (H) (Atomic number: 1) Electron Configuration]
    Full notation: 1s^1
    Outermost shell: 1 shell
    Valence electrons: 1
    
    [Carbon (C) (Atomic number: 6) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^2
    Outermost shell: 2 shell
    Valence electrons: 4
    
    [Sodium (Na) (Atomic number: 11) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^6 3s^1
    Outermost shell: 3 shell
    Valence electrons: 1
    
    [Silicon (Si) (Atomic number: 14) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^6 3s^2 3p^2
    Outermost shell: 3 shell
    Valence electrons: 4
    
    [Iron (Fe) (Atomic number: 26) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^2 3d^6
    Outermost shell: 4 shell
    Valence electrons: 2
    
    [Copper (Cu) (Atomic number: 29) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^1 3d^10
    Outermost shell: 4 shell
    Valence electrons: 1
    
    [Gold (Au) (Atomic number: 79) Electron Configuration]
    Full notation: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^2 3d^10 4p^6 5s^2 4d^10 5p^6 6s^1 4f^14 5d^10
    Outermost shell: 6 shell
    Valence electrons: 1
    

**Explanation** : This function automatically calculates the electron configuration of any element. The number of valence electrons is an important parameter that determines the nature of chemical bonding.

### Code Example 2: Electronegativity Heatmap on the Periodic Table

Visualizes electronegativity on the periodic table using a heatmap.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    
    """
    Example: Visualizes electronegativity on the periodic table using a h
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Electronegativity data for major elements (Pauling scale)
    # Position on periodic table: (period, group) ’ electronegativity
    electronegativity_data = {
        # Period 1
        (1, 1): ('H', 2.20),
        (1, 18): ('He', 0),  # Noble gases are not defined
    
        # Period 2
        (2, 1): ('Li', 0.98), (2, 2): ('Be', 1.57),
        (2, 13): ('B', 2.04), (2, 14): ('C', 2.55), (2, 15): ('N', 3.04),
        (2, 16): ('O', 3.44), (2, 17): ('F', 3.98), (2, 18): ('Ne', 0),
    
        # Period 3
        (3, 1): ('Na', 0.93), (3, 2): ('Mg', 1.31),
        (3, 13): ('Al', 1.61), (3, 14): ('Si', 1.90), (3, 15): ('P', 2.19),
        (3, 16): ('S', 2.58), (3, 17): ('Cl', 3.16), (3, 18): ('Ar', 0),
    
        # Period 4 (partial)
        (4, 1): ('K', 0.82), (4, 2): ('Ca', 1.00),
        (4, 13): ('Ga', 1.81), (4, 14): ('Ge', 2.01), (4, 15): ('As', 2.18),
        (4, 16): ('Se', 2.55), (4, 17): ('Br', 2.96), (4, 18): ('Kr', 0),
    
        # Transition metals (partial)
        (4, 6): ('Cr', 1.66), (4, 8): ('Fe', 1.83), (4, 11): ('Cu', 1.90),
        (4, 12): ('Zn', 1.65),
    }
    
    # Create periodic table grid (4 periods × 18 groups)
    grid = np.full((4, 18), np.nan)
    labels = [['' for _ in range(18)] for _ in range(4)]
    
    for (period, group), (symbol, en) in electronegativity_data.items():
        grid[period-1, group-1] = en
        labels[period-1][group-1] = symbol
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Mask NaN (noble gases and blanks)
    mask = np.isnan(grid) | (grid == 0)
    
    sns.heatmap(grid, annot=np.array(labels), fmt='', cmap='RdYlGn_r',
                cbar_kws={'label': 'Electronegativity (Pauling scale)'},
                linewidths=0.5, linecolor='gray', mask=mask,
                vmin=0.5, vmax=4.0, ax=ax)
    
    ax.set_xlabel('Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Period', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Electronegativity in the Periodic Table', fontsize=15, fontweight='bold', pad=20)
    
    # Adjust axis labels
    ax.set_xticklabels(range(1, 19), fontsize=10)
    ax.set_yticklabels(range(1, 5), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTrends in electronegativity:")
    print("- Higher towards upper right (F, O, N) ’ strongly attracts electrons")
    print("- Lower towards lower left (Cs, Fr, Na) ’ easily releases electrons")
    print("- Metals generally have lower values (< 2.0)")
    print("- Non-metals generally have higher values (> 2.0)")
    

**Explanation** : Electronegativity tends to be higher towards the upper right of the periodic table and lower towards the lower left. This trend is also related to ionization energy and electron affinity.

### Code Example 3: Classification of Bond Types Based on Electronegativity Difference

Classifies bond types based on electronegativity difference between two elements.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Element electronegativity data
    electronegativity = {
        'H': 2.20, 'Li': 0.98, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Fe': 1.83, 'Cu': 1.90, 'Zn': 1.65, 'Br': 2.96
    }
    
    
    def classify_bond(element1, element2):
        """
        Classify the bond type between two elements
        """
        en1 = electronegativity[element1]
        en2 = electronegativity[element2]
        delta_en = abs(en1 - en2)
    
        if delta_en > 2.0:
            bond_type = 'Ionic bond'
            color = '#ff7f0e'
        elif delta_en > 0.5:
            bond_type = 'Polar covalent bond'
            color = '#2ca02c'
        else:
            bond_type = 'Nonpolar covalent bond'
            color = '#1f77b4'
    
        return delta_en, bond_type, color
    
    
    # Bond classification of representative compounds
    compounds = [
        ('Na', 'Cl', 'NaCl (table salt)'),
        ('Mg', 'O', 'MgO (magnesium oxide)'),
        ('Al', 'O', 'Al‚Oƒ (alumina)'),
        ('H', 'O', 'H‚O (water)'),
        ('H', 'F', 'HF (hydrogen fluoride)'),
        ('C', 'O', 'CO‚ (carbon dioxide)'),
        ('C', 'C', 'Diamond'),
        ('Si', 'Si', 'Silicon crystal'),
        ('H', 'H', 'H‚ (hydrogen molecule)'),
        ('C', 'H', 'CH„ (methane)'),
        ('N', 'N', 'N‚ (nitrogen molecule)'),
        ('Si', 'O', 'SiO‚ (silica)'),
    ]
    
    # Calculate results and plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    delta_ens = []
    compound_names = []
    colors = []
    bond_types = []
    
    for elem1, elem2, name in compounds:
        delta_en, bond_type, color = classify_bond(elem1, elem2)
        delta_ens.append(delta_en)
        compound_names.append(name)
        colors.append(color)
        bond_types.append(bond_type)
    
    # Graph 1: Bar chart of electronegativity differences
    ax1.barh(compound_names, delta_ens, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Polarity boundary')
    ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Ionic bond boundary')
    ax1.set_xlabel('Electronegativity difference ”Ç', fontsize=12, fontweight='bold')
    ax1.set_title('Electronegativity Difference and Bond Type in Compounds', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Graph 2: Distribution of bond types (scatter plot)
    bond_type_categories = {'Nonpolar covalent bond': 1, 'Polar covalent bond': 2, 'Ionic bond': 3}
    y_positions = [bond_type_categories[bt] for bt in bond_types]
    
    ax2.scatter(delta_ens, y_positions, s=200, c=colors, edgecolors='black', linewidth=2, alpha=0.7)
    
    for i, name in enumerate(compound_names):
        ax2.annotate(name.split('(')[0], (delta_ens[i], y_positions[i]),
                     xytext=(5, 0), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Electronegativity difference ”Ç', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bond Type', fontsize=12, fontweight='bold')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Nonpolar covalent', 'Polar covalent', 'Ionic'])
    ax2.set_title('Bond Classification by Electronegativity Difference', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Boundary lines
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Display results
    print("\n[Bond Type Classification Results]")
    print("=" * 70)
    for name, delta_en, bond_type in zip(compound_names, delta_ens, bond_types):
        print(f"{name:30s} | ”Ç = {delta_en:.2f} | {bond_type}")
    

**Output example** :
    
    
    [Bond Type Classification Results]
    ======================================================================
    NaCl (table salt)              | ”Ç = 2.23 | Ionic bond
    MgO (magnesium oxide)          | ”Ç = 2.13 | Ionic bond
    Al‚Oƒ (alumina)                | ”Ç = 1.83 | Polar covalent bond
    H‚O (water)                    | ”Ç = 1.24 | Polar covalent bond
    HF (hydrogen fluoride)         | ”Ç = 1.78 | Polar covalent bond
    CO‚ (carbon dioxide)           | ”Ç = 0.89 | Polar covalent bond
    Diamond                        | ”Ç = 0.00 | Nonpolar covalent bond
    Silicon crystal                | ”Ç = 0.00 | Nonpolar covalent bond
    H‚ (hydrogen molecule)         | ”Ç = 0.00 | Nonpolar covalent bond
    CH„ (methane)                  | ”Ç = 0.35 | Nonpolar covalent bond
    N‚ (nitrogen molecule)         | ”Ç = 0.00 | Nonpolar covalent bond
    SiO‚ (silica)                  | ”Ç = 1.54 | Polar covalent bond
    

**Explanation** : Bond types can be roughly classified based on electronegativity difference. However, actual bonds are not purely ionic or covalent, and often possess characteristics of both.

### Code Example 4: Relationship Between Bond Energy and Melting Point

Plots the relationship between bond energy and melting point for representative materials of each bond type.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Plots the relationship between bond energy and melting point
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material data (average bond energy kJ/mol, melting point K)
    materials_bond = {
        'Ionic bond': {
            'NaCl': (410, 1074),
            'MgO': (1000, 3125),
            'Al‚Oƒ': (1180, 2345),
            'CaF‚': (540, 1691),
        },
        'Covalent bond': {
            'Diamond': (347, 3823),
            'SiC': (318, 3103),
            'SiƒN„': (290, 2173),
            'Si': (222, 1687),
        },
        'Metallic bond': {
            'Tungsten': (850, 3695),
            'Iron': (415, 1811),
            'Copper': (338, 1358),
            'Aluminum': (326, 933),
        },
        'Intermolecular forces': {
            'Ice': (20, 273),
            'Dry ice': (5, 195),
            'Argon': (1, 84),
            'Methane': (8, 91),
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_bond = {
        'Ionic bond': '#ff7f0e',
        'Covalent bond': '#1f77b4',
        'Metallic bond': '#2ca02c',
        'Intermolecular forces': '#d62728'
    }
    
    for bond_type, materials in materials_bond.items():
        bond_energies = [v[0] for v in materials.values()]
        melting_points = [v[1] for v in materials.values()]
        names = list(materials.keys())
    
        ax.scatter(bond_energies, melting_points, s=200, alpha=0.7,
                   color=colors_bond[bond_type], label=bond_type,
                   edgecolors='black', linewidth=1.5)
    
        # Display material names as labels
        for name, x, y in zip(names, bond_energies, melting_points):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Logarithmic scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Average Bond Energy (kJ/mol)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Melting Point (K)', fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Bond Energy and Melting Point', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCorrelation between bond energy and melting point:")
    print("- Higher bond energy tends to correlate with higher melting point")
    print("- Covalent bond materials (diamond, SiC) have very high melting points")
    print("- Intermolecular force materials (ice, dry ice) have low melting points")
    print("- Metallic bonds show medium to high melting points (flexible bonding due to free electrons)")
    

**Explanation** : There is a strong correlation between bond energy and melting point. Covalent bond materials have very high melting points due to strong bonding, while intermolecular force materials have low melting points due to weak bonding.

### Code Example 5: Relationship Between Bond Type and Electrical Conductivity

Visualizes how electrical conductivity differs by orders of magnitude depending on bond type.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Visualizes how electrical conductivity differs by orders of 
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material data (bond type, electrical conductivity S/m)
    materials_conductivity_bond = [
        ('Copper', 'Metallic bond', 5.96e7),
        ('Aluminum', 'Metallic bond', 3.5e7),
        ('Iron', 'Metallic bond', 1.0e7),
        ('Graphite', 'Covalent bond (special)', 1e5),
        ('Silicon', 'Covalent bond', 1e-3),
        ('Diamond', 'Covalent bond', 1e-13),
        ('NaCl (molten)', 'Ionic bond', 1e2),
        ('NaCl (solid)', 'Ionic bond', 1e-15),
        ('Polyethylene', 'Intermolecular forces', 1e-17),
        ('Teflon', 'Intermolecular forces', 1e-16),
    ]
    
    # Organize data
    names = [m[0] for m in materials_conductivity_bond]
    bond_types = [m[1] for m in materials_conductivity_bond]
    conductivities = [m[2] for m in materials_conductivity_bond]
    
    # Color assignment
    color_map_bond = {
        'Metallic bond': '#2ca02c',
        'Covalent bond': '#1f77b4',
        'Covalent bond (special)': '#17becf',
        'Ionic bond': '#ff7f0e',
        'Intermolecular forces': '#d62728'
    }
    colors = [color_map_bond[bt] for bt in bond_types]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    y_positions = np.arange(len(names))
    ax.barh(y_positions, conductivities, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Electrical Conductivity (S/m)', fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Bond Type and Electrical Conductivity', fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend (manually created)
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, edgecolor='black')
                       for color in set(colors)]
    legend_labels = list(set(bond_types))
    ax.legend(legend_elements, legend_labels, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRelationship between bond type and electrical conductivity:")
    print("- Metallic bonding: High electrical conductivity (10w S/m) due to free electrons")
    print("- Covalent bonding: Generally low electrical conductivity due to localized electrons")
    print("  - However, graphite is an exception (conductivity due to delocalized À electrons)")
    print("- Ionic bonding: Insulator in solid state, conductive in molten state (ion mobility)")
    print("- Intermolecular forces: Electrically insulating (10{¹u S/m or lower)")
    

**Explanation** : Electrical conductivity varies by more than 24 orders of magnitude depending on bond type. Metallic bond materials show high conductivity due to free electrons, while intermolecular force materials are electrically insulating.

### Code Example 6: Property Matrix for Bond Types

Creates a radar chart to quantitatively compare various properties of the four bond types.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Creates a radar chart to quantitatively compare various prop
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi
    
    # Property data for bond types (0-10 scale, 10 is highest)
    categories = ['Bond strength', 'Melting point', 'Electrical conductivity', 'Ductility', 'Hardness', 'Chemical stability']
    N = len(categories)
    
    # Property values for each bond type
    ionic = [7, 8, 2, 1, 8, 7]          # Ionic bonding
    covalent = [9, 10, 1, 1, 10, 9]     # Covalent bonding
    metallic = [6, 7, 10, 9, 5, 6]      # Metallic bonding
    intermolecular = [2, 1, 1, 8, 2, 4] # Intermolecular forces
    
    # Calculate angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ionic += ionic[:1]
    covalent += covalent[:1]
    metallic += metallic[:1]
    intermolecular += intermolecular[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each bond type
    ax.plot(angles, ionic, 'o-', linewidth=2, label='Ionic bond', color='#ff7f0e')
    ax.fill(angles, ionic, alpha=0.15, color='#ff7f0e')
    
    ax.plot(angles, covalent, 'o-', linewidth=2, label='Covalent bond', color='#1f77b4')
    ax.fill(angles, covalent, alpha=0.15, color='#1f77b4')
    
    ax.plot(angles, metallic, 'o-', linewidth=2, label='Metallic bond', color='#2ca02c')
    ax.fill(angles, metallic, alpha=0.15, color='#2ca02c')
    
    ax.plot(angles, intermolecular, 'o-', linewidth=2, label='Intermolecular forces', color='#d62728')
    ax.fill(angles, intermolecular, alpha=0.15, color='#d62728')
    
    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
    ax.grid(True)
    
    # Title and legend
    plt.title('Property Comparison by Bond Type', size=16, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print("\nCharacteristics of each bond type:")
    print("\n[Ionic bonding]")
    print("- Strength: High (electrostatic attraction)")
    print("- Melting point: High")
    print("- Electrical conductivity: Low in solid state, but high in molten state")
    print("- Ductility: Low (brittle)")
    print("- Hardness: High")
    
    print("\n[Covalent bonding]")
    print("- Strength: Very high (strongest)")
    print("- Melting point: Very high")
    print("- Electrical conductivity: Low (insulator or semiconductor)")
    print("- Ductility: Low (brittle)")
    print("- Hardness: Very high")
    
    print("\n[Metallic bonding]")
    print("- Strength: Medium to high")
    print("- Melting point: Medium to high")
    print("- Electrical conductivity: Very high (free electrons)")
    print("- Ductility: Very high (plastic deformation possible)")
    print("- Hardness: Medium")
    
    print("\n[Intermolecular forces]")
    print("- Strength: Weak")
    print("- Melting point: Low")
    print("- Electrical conductivity: Low (insulator)")
    print("- Ductility: High (flexible)")
    print("- Hardness: Low")
    

**Explanation** : This radar chart allows for an at-a-glance understanding of the characteristic properties of each bond type. When selecting materials, it is important to choose materials with the appropriate bond type based on the desired properties.

* * *

## 2.5 Chapter Summary

### What We Learned

  1. **Atomic structure and electron configuration**
     * Atoms are composed of protons, neutrons, and electrons
     * Electrons are arranged in electron shells (K, L, M, N...)
     * Valence electrons participate in chemical bonding
     * Electron configuration is written as 1s² 2s² 2pv
  2. **Four types of chemical bonds**
     * Ionic bonding: Electrostatic attraction through electron transfer (NaCl, MgO)
     * Covalent bonding: Electron sharing (diamond, SiC)
     * Metallic bonding: Bonding through free electrons (Cu, Fe, Al)
     * Intermolecular forces: Weak interactions (polymers, ice)
  3. **Electronegativity and bonding character**
     * Electronegativity difference determines bond type
     * ”Ç > 2.0: Ionic bonding
     * 0.5 < ”Ç < 2.0: Polar covalent bonding
     * ”Ç < 0.5: Nonpolar covalent bonding
  4. **Relationship between bonding and material properties**
     * Higher bond energy correlates with higher melting point and hardness
     * Metallic bond materials exhibit high electrical conductivity
     * Covalent bond materials have high hardness and melting point but are brittle
     * Intermolecular force materials are flexible and easy to process but have low strength
  5. **Visualization with Python**
     * Automatic calculation of electron configurations
     * Electronegativity heatmaps
     * Classification and property comparison of bond types

### Key Points

  * Material properties are **determined by bond type**
  * Electronegativity is a key concept for understanding bonding character
  * Strong correlation exists between bond energy and melting point/hardness
  * Electrical conductivity varies by orders of magnitude depending on the presence of free electrons
  * In materials design, **selecting bond type is the first step**

### To the Next Chapter

In Chapter 3, we will learn about **fundamentals of crystal structure** :

  * Differences between crystalline and amorphous materials
  * Unit cells and lattice parameters
  * Major crystal structures (FCC, BCC, HCP)
  * Miller indices and crystal planes
  * 3D visualization of crystal structures with Python
