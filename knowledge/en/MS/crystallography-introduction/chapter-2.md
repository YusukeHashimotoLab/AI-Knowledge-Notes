---
title: "Chapter 2: Bravais Lattices and Space Groups"
chapter_title: "Chapter 2: Bravais Lattices and Space Groups"
subtitle: Understanding crystal symmetry and learning the fundamentals of 230 space groups
reading_time: 26-32 minutes
difficulty: Beginner to Intermediate
code_examples: 8
version: 1.0
created_at: "by:"
---

Building on the lattice concepts learned in Chapter 1, this chapter delves into the core of crystallography: the 14 Bravais lattices and 230 space groups. These constitute the fundamental framework for classifying all crystal structures and represent one of the most important concepts in materials science. You will also learn practical analysis techniques using the Python library "pymatgen." 

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Classify and identify the 14 Bravais lattices
  * ✅ Understand the difference between translational symmetry and point symmetry
  * ✅ Understand the basics of symmetry operations (rotation, reflection, inversion, rotoinversion)
  * ✅ Master the basic concepts of space groups and how to read international symbols
  * ✅ Retrieve and analyze space group information using pymatgen
  * ✅ Investigate space groups of real materials and understand symmetry operations

* * *

## 2.1 What are Bravais Lattices?

### Classification of Lattices

As we learned in Chapter 1, crystals have a periodic arrangement called a **lattice**. However, not all lattices have the same shape. In 1848, French physicist **Auguste Bravais** systematically classified lattices in three-dimensional space and proved that only **14 Bravais lattices** exist.

Bravais lattices are classified based on two criteria:

  1. **Crystal system** : Seven classifications based on lattice symmetry
  2. **Centering type** : Positions of lattice points within the unit cell

### Lattice Point Arrangement Patterns

There are four types of lattice point arrangements within unit cells:

Symbol | Name (Japanese) | Name (English) | Lattice Point Positions  
---|---|---|---  
**P** | Primitive lattice | Primitive | Corners only (8 locations)  
**I** | Body-centered lattice | Body-centered (Innenzentriert) | Corners + body center  
**F** | Face-centered lattice | Face-centered | Corners + center of each face (6 locations)  
**C/A/B** | Base-centered lattice | Base-centered | Corners + center of one pair of opposite faces  
  
**Important note** : These arrangements must satisfy the condition "cannot reduce the unit cell while preserving symmetry." For example, monoclinic systems do not have body-centered (I) or face-centered (F) types (because they can be reduced to primitive (P) lattices).

### Existence of 14 Bravais Lattices

Combining seven crystal systems with four centering types theoretically yields 28 lattices, but in reality only **14** exist independently. This is because some combinations can be converted to other lattices.

> **Mathematical Background** : The number 14 for Bravais lattices derives from the combination of point group symmetry and translational symmetry in three-dimensional Euclidean space. This classification is based on the theory of **crystallographic point groups** in group theory. 

* * *

## 2.2 Details of the 14 Bravais Lattices

Below, Bravais lattices are classified by the seven crystal systems. Characteristics and examples of real materials are shown for each lattice.
    
    
    ```mermaid
    graph TB
        A[14 Bravais Lattices] --> B[Triclinic1 type]
        A --> C[Monoclinic2 types]
        A --> D[Orthorhombic4 types]
        A --> E[Tetragonal2 types]
        A --> F[Hexagonal1 type]
        A --> G[Trigonal1 type]
        A --> H[Cubic3 types]
    
        B --> B1[P]
        C --> C1[P]
        C --> C2[C]
        D --> D1[P]
        D --> D2[C]
        D --> D3[I]
        D --> D4[F]
        E --> E1[P]
        E --> E2[I]
        F --> F1[P]
        G --> G1[P or R]
        H --> H1[Psimple cubic]
        H --> H2[IBCC]
        H --> H3[FFCC]
    
        style B fill:#fce7f3
        style C fill:#fce7f3
        style D fill:#fce7f3
        style E fill:#fce7f3
        style F fill:#fce7f3
        style G fill:#fce7f3
        style H fill:#fce7f3
    ```

### 1\. Triclinic System - 1 type

**Lattice parameters** : $a \neq b \neq c$, $\alpha \neq \beta \neq \gamma \neq 90°$ (all different)

  * **P (Primitive)** : Lattice with lowest symmetry
  * **Examples** : CuSO₄·5H₂O (copper sulfate pentahydrate), K₂Cr₂O₇ (potassium dichromate)

### 2\. Monoclinic System - 2 types

**Lattice parameters** : $a \neq b \neq c$, $\alpha = \gamma = 90° \neq \beta$

  * **P (Primitive)** : Corners only
  * **C (Base-centered)** : Adds lattice point at center of face perpendicular to c-axis (ab plane)
  * **Examples** : β-S (orthorhombic sulfur), gypsum (CaSO₄·2H₂O)

### 3\. Orthorhombic System - 4 types

**Lattice parameters** : $a \neq b \neq c$, $\alpha = \beta = \gamma = 90°$

  * **P (Primitive)** : Corners only
  * **C (Base-centered)** : Lattice point at center of ab plane
  * **I (Body-centered)** : Lattice point at body center
  * **F (Face-centered)** : Lattice points at centers of all faces
  * **Examples** : α-S (orthorhombic sulfur, P), BaSO₄ (barite, P), U (uranium, C)

### 4\. Tetragonal System - 2 types

**Lattice parameters** : $a = b \neq c$, $\alpha = \beta = \gamma = 90°$

  * **P (Primitive)** : Corners only
  * **I (Body-centered)** : Lattice point at body center
  * **Examples** : TiO₂ (rutile, P), In (indium, I), Sn (white tin, I)

### 5\. Hexagonal System - 1 type

**Lattice parameters** : $a = b \neq c$, $\alpha = \beta = 90°$, $\gamma = 120°$

  * **P (Primitive)** : Lattice points at hexagonal vertices
  * **Examples** : Graphite (C), Mg (magnesium), Zn (zinc), ice

**Note** : Hexagonal system crystals often have **Hexagonal Close-Packed (HCP)** structure. HCP is not a Bravais lattice itself but a **crystal structure** with atoms arranged on the lattice.

### 6\. Trigonal (Rhombohedral) System - 1 type

**Lattice parameters** : $a = b = c$, $\alpha = \beta = \gamma \neq 90°$

  * **P (or R)** : Rhombohedral lattice
  * **Examples** : Calcite (CaCO₃), quartz (α-SiO₂), Bi (bismuth)

**Relationship with hexagonal system** : Trigonal system crystals can also be described using hexagonal system settings. In crystallography, they are often represented using hexagonal axes.

### 7\. Cubic System - 3 types

**Lattice parameters** : $a = b = c$, $\alpha = \beta = \gamma = 90°$

  * **P (Simple cubic)** : Corners only (very few real materials)
  * **I (Body-Centered Cubic, BCC)** : Lattice point at body center
  * **F (Face-Centered Cubic, FCC)** : Lattice points at centers of all faces

**Examples** :

  * **P (simple cubic)** : α-Po (polonium), CsCl-type structure (though CsCl itself has 2-atom basis)
  * **I (BCC)** : Fe (α-iron, room temperature), Cr (chromium), W (tungsten), Na (sodium)
  * **F (FCC)** : Al (aluminum), Cu (copper), Au (gold), Ag (silver), Ni (nickel), Pb (lead)

### Summary Table of 14 Bravais Lattices

Crystal System | Lattice Parameter Conditions | Bravais Lattices | Total  
---|---|---|---  
Triclinic | $a \neq b \neq c$, $\alpha \neq \beta \neq \gamma$ | P | 1  
Monoclinic | $a \neq b \neq c$, $\alpha = \gamma = 90° \neq \beta$ | P, C | 2  
Orthorhombic | $a \neq b \neq c$, $\alpha = \beta = \gamma = 90°$ | P, C, I, F | 4  
Tetragonal | $a = b \neq c$, $\alpha = \beta = \gamma = 90°$ | P, I | 2  
Hexagonal | $a = b \neq c$, $\alpha = \beta = 90°, \gamma = 120°$ | P | 1  
Trigonal | $a = b = c$, $\alpha = \beta = \gamma \neq 90°$ | P (R) | 1  
Cubic | $a = b = c$, $\alpha = \beta = \gamma = 90°$ | P, I, F | 3  
**Total** | **14**  
  
* * *

## 2.3 Symmetry Operations

Crystal symmetry is described by **symmetry operations**. A symmetry operation is a transformation (rotation, reflection, etc.) that moves the crystal but leaves it indistinguishable from its original configuration.

Symmetry operations are classified into two broad categories:

  1. **Point symmetry operations** : At least one point remains fixed (does not move)
  2. **Translational symmetry operations** : All points move

### Point Symmetry Operations

#### 1\. Rotation

Operation that rotates by a specific angle around an axis. In crystallography, only **1-fold, 2-fold, 3-fold, 4-fold, and 6-fold rotation axes** are allowed (5-fold and 7-fold or higher are incompatible with translational symmetry).

Rotational Symmetry | Symbol | Rotation Angle | Repetition Count  
---|---|---|---  
1-fold rotation | 1 | 360° | 1 (identity operation)  
2-fold rotation | 2 | 180° | 2  
3-fold rotation | 3 | 120° | 3  
4-fold rotation | 4 | 90° | 4  
6-fold rotation | 6 | 60° | 6  
  
> **Crystallographic restriction** : Why don't 5-fold or 7-fold rotational symmetries exist? This is because it has been mathematically proven that only 1, 2, 3, 4, and 6-fold rotational symmetries are compatible with translational symmetry (lattice periodicity). 5-fold symmetry appears only in quasicrystals. 

#### 2\. Mirror Reflection

Operation having mirror symmetry with respect to a plane. Denoted by symbol **m**.

#### 3\. Inversion

Operation that inverts all coordinates with respect to a point (inversion center). Denoted by symbol **$\bar{1}$** or **i**.

Mathematical expression for inversion: $(x, y, z) \rightarrow (-x, -y, -z)$

#### 4\. Rotoinversion

Operation combining rotation and inversion. Denoted by symbols **$\bar{1}, \bar{2}, \bar{3}, \bar{4}, \bar{6}$**.

For example, $\bar{4}$ is the operation "rotate 90° then invert."

### Translational Symmetry Operations

Translational symmetry operations combine rotation or reflection with translation (parallel displacement).

#### 1\. Glide Reflection

Operation combining reflection and translation. Denoted by symbols **a, b, c, n, d**.

Example: **a-glide** is reflection with respect to a plane followed by translation in the a-axis direction by $a/2$.

#### 2\. Screw Axis

Operation combining rotation and translation. Denoted by symbols **2₁, 3₁, 4₁, 6₁** , etc.

Example: **2₁** is 180° rotation followed by translation in the rotation axis direction by half the lattice vector (read as "21-screw").

### Hermann-Mauguin Notation

Combinations of symmetry operations are represented by **Hermann-Mauguin notation (international symbols)**. This is the standard method for describing point groups and space groups of crystals.

Examples:

  * **2/m** : 2-fold rotation axis and perpendicular mirror plane
  * **4mm** : 4-fold rotation axis and two mirror planes
  * **$\bar{3}$m** : 3-fold rotoinversion axis and mirror plane

* * *

## 2.4 Space Group Concepts

### What are Space Groups?

A **space group** is the set of all symmetry operations (point symmetry operations + translational symmetry operations) of a crystal. In 1891, Russian crystallographer **Evgraf Fedorov** and German mathematician **Arthur Schönflies** independently proved that only **230 space groups** exist in three-dimensional space.

This is a remarkable result: all crystal structures (infinitely diverse) can be classified into only 230 symmetry patterns.

### Relationship Between Point Groups and Space Groups

To understand space groups, we must first understand **point groups**.

  * **Point group** : Set of point symmetry operations only (no translation) → **32 types** exist
  * **Space group** : Set of point symmetry operations + translational symmetry operations → **230 types** exist

Each space group corresponds to one point group. The 230 space groups are constructed based on the 32 point groups.

### Space Group Numbers and International Symbols

The 230 space groups are assigned **numbers** from 1 to 230 and **international symbols** using Hermann-Mauguin notation.

**Examples of representative space groups** :

Number | International Symbol | Crystal System | Examples  
---|---|---|---  
1 | P1 | Triclinic | Lowest symmetry  
2 | P$\bar{1}$ | Triclinic | Has inversion center  
63 | Cmcm | Orthorhombic | Base-centered + mirror + c-glide + mirror  
139 | I4/mmm | Tetragonal | Body-centered + 4-fold rotation + mirror  
194 | P6₃/mmc | Hexagonal | Mg, Zn, Ti (HCP structure)  
225 | Fm$\bar{3}$m | Cubic | Al, Cu, Au, Ag (FCC structure)  
229 | Im$\bar{3}$m | Cubic | Fe, Cr, W (BCC structure)  
  
### Reading International Symbols

Space group international symbols are structured as follows:

**Symbol structure** : `[Lattice type][Symmetry element 1][Symmetry element 2][Symmetry element 3]...`

Example: **Fm$\bar{3}$m** (Space group 225)

  * **F** : Face-centered lattice
  * **m** : Mirror plane
  * **$\bar{3}$** : 3-fold rotoinversion axis
  * **m** : Another mirror plane

This space group corresponds to FCC structure metals such as copper (Cu) and aluminum (Al).

### General Positions and Special Positions

Space groups have **general positions** and **special positions (Wyckoff positions)** where atoms can be placed.

  * **General position** : Symmetry operations generate multiple equivalent positions
  * **Special position** : On symmetry elements, so fewer equivalent positions are generated

For example, the general position in Fm$\bar{3}$m (FCC) multiplies 48-fold, but special positions (e.g., (0, 0, 0)) remain 1-fold.

* * *

## 2.5 Analyzing Bravais Lattices and Space Groups with Python

From here, we will practically analyze Bravais lattices and space groups using the Python library **pymatgen (Python Materials Genomics)**.

### Installing pymatgen

Pymatgen is a powerful Python library for materials science. It provides many features including crystal structure manipulation, symmetry analysis, and Materials Project API integration.
    
    
    # Install pymatgen
    pip install pymatgen
    
    # Optionally, install these as well
    pip install matplotlib numpy pandas plotly

**Notes** :

  * Pymatgen has many dependencies, so installation may take time
  * Latest versions (2024 onward) have API changes
  * Using Materials Project API requires a free API key (described later)

### Code Example 1: Information Table for 14 Bravais Lattices

First, create a table organizing basic information about the 14 Bravais lattices.
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: First, create a table organizing basic information about the
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    
    # Data for 14 Bravais lattices
    bravais_lattices = [
        # Triclinic
        {'Crystal System': 'Triclinic', 'Bravais Symbol': 'aP', 'Lattice Type': 'P (Primitive)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'Lowest', 'Example': 'CuSO₄·5H₂O'},
    
        # Monoclinic
        {'Crystal System': 'Monoclinic', 'Bravais Symbol': 'mP', 'Lattice Type': 'P (Primitive)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'Low', 'Example': 'Gypsum'},
        {'Crystal System': 'Monoclinic', 'Bravais Symbol': 'mC', 'Lattice Type': 'C (Base-centered)',
         'Lattice Points/Unit Cell': 2, 'Symmetry': 'Low', 'Example': 'β-S'},
    
        # Orthorhombic
        {'Crystal System': 'Orthorhombic', 'Bravais Symbol': 'oP', 'Lattice Type': 'P (Primitive)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'Medium', 'Example': 'α-S'},
        {'Crystal System': 'Orthorhombic', 'Bravais Symbol': 'oC', 'Lattice Type': 'C (Base-centered)',
         'Lattice Points/Unit Cell': 2, 'Symmetry': 'Medium', 'Example': 'U'},
        {'Crystal System': 'Orthorhombic', 'Bravais Symbol': 'oI', 'Lattice Type': 'I (Body-centered)',
         'Lattice Points/Unit Cell': 2, 'Symmetry': 'Medium', 'Example': 'TiO₂ (brookite)'},
        {'Crystal System': 'Orthorhombic', 'Bravais Symbol': 'oF', 'Lattice Type': 'F (Face-centered)',
         'Lattice Points/Unit Cell': 4, 'Symmetry': 'Medium', 'Example': 'NaNO₃'},
    
        # Tetragonal
        {'Crystal System': 'Tetragonal', 'Bravais Symbol': 'tP', 'Lattice Type': 'P (Primitive)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'High', 'Example': 'TiO₂ (rutile)'},
        {'Crystal System': 'Tetragonal', 'Bravais Symbol': 'tI', 'Lattice Type': 'I (Body-centered)',
         'Lattice Points/Unit Cell': 2, 'Symmetry': 'High', 'Example': 'In, Sn (white)'},
    
        # Hexagonal
        {'Crystal System': 'Hexagonal', 'Bravais Symbol': 'hP', 'Lattice Type': 'P (Primitive)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'High', 'Example': 'Mg, Zn, Graphite'},
    
        # Trigonal
        {'Crystal System': 'Trigonal', 'Bravais Symbol': 'hR', 'Lattice Type': 'R (Rhombohedral)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'High', 'Example': 'Calcite, Bi'},
    
        # Cubic
        {'Crystal System': 'Cubic', 'Bravais Symbol': 'cP', 'Lattice Type': 'P (Simple cubic)',
         'Lattice Points/Unit Cell': 1, 'Symmetry': 'Highest', 'Example': 'α-Po'},
        {'Crystal System': 'Cubic', 'Bravais Symbol': 'cI', 'Lattice Type': 'I (BCC)',
         'Lattice Points/Unit Cell': 2, 'Symmetry': 'Highest', 'Example': 'Fe, Cr, W'},
        {'Crystal System': 'Cubic', 'Bravais Symbol': 'cF', 'Lattice Type': 'F (FCC)',
         'Lattice Points/Unit Cell': 4, 'Symmetry': 'Highest', 'Example': 'Al, Cu, Au, Ag'},
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(bravais_lattices)
    
    print("=" * 100)
    print("Classification of 14 Bravais Lattices")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # Aggregation by crystal system
    print("\n" + "=" * 100)
    print("Number of Bravais Lattices by Crystal System")
    print("=" * 100)
    print(df.groupby('Crystal System').size())
    
    print("\n【Explanation】")
    print("- Bravais Symbol: Also called Pearson symbol, represented by crystal system initial + lattice type")
    print("- Lattice points: Number of lattice points in unit cell (P=1, C/I=2, F=4)")
    print("- Higher symmetry means stronger constraints on lattice parameters")
    print("- Cubic systems (cP, cI, cF) have highest symmetry and are important in materials science")

**Example output** :
    
    
    ====================================================================================================
    Classification of 14 Bravais Lattices
    ====================================================================================================
    Crystal System Bravais Symbol           Lattice Type  Lattice Points/Unit Cell Symmetry                Example
         Triclinic             aP     P (Primitive)                         1   Lowest       CuSO₄·5H₂O
        Monoclinic             mP     P (Primitive)                         1      Low             Gypsum
        Monoclinic             mC  C (Base-centered)                         2      Low               β-S
      Orthorhombic             oP     P (Primitive)                         1   Medium               α-S
      Orthorhombic             oC  C (Base-centered)                         2   Medium                 U
      Orthorhombic             oI  I (Body-centered)                         2   Medium  TiO₂ (brookite)
      Orthorhombic             oF  F (Face-centered)                         4   Medium            NaNO₃
        Tetragonal             tP     P (Primitive)                         1     High   TiO₂ (rutile)
        Tetragonal             tI  I (Body-centered)                         2     High      In, Sn (white)
         Hexagonal             hP     P (Primitive)                         1     High   Mg, Zn, Graphite
          Trigonal             hR  R (Rhombohedral)                         1     High         Calcite, Bi
             Cubic             cP  P (Simple cubic)                         1  Highest             α-Po
             Cubic             cI         I (BCC)                         2  Highest          Fe, Cr, W
             Cubic             cF         F (FCC)                         4  Highest      Al, Cu, Au, Ag
    
    ====================================================================================================
    Number of Bravais Lattices by Crystal System
    ====================================================================================================
    Crystal System
    Triclinic       1
    Trigonal        1
    Hexagonal       1
    Monoclinic      2
    Tetragonal      2
    Orthorhombic    4
    Cubic           3
    dtype: int64

### Code Example 2: Visualization of Three Cubic Bravais Lattices

Visualize the three most important cubic Bravais lattices (P, I, F) in 3D.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - plotly>=5.14.0
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_cubic_lattice_points(lattice_type='P', a=1.0):
        """
        Generate lattice point coordinates for cubic lattices
    
        Parameters:
        -----------
        lattice_type : str
            'P' (Primitive), 'I' (Body-centered), 'F' (Face-centered)
        a : float
            Lattice constant
    
        Returns:
        --------
        np.ndarray : Lattice point coordinates (N, 3)
        """
        # Corners (8 locations)
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
            [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
        ])
    
        if lattice_type == 'P':
            return corners
    
        elif lattice_type == 'I':
            # Add body center
            body_center = np.array([[a/2, a/2, a/2]])
            return np.vstack([corners, body_center])
    
        elif lattice_type == 'F':
            # Add 6 face centers
            face_centers = np.array([
                [a/2, a/2, 0],   # bottom face
                [a/2, a/2, a],   # top face
                [a/2, 0, a/2],   # front face
                [a/2, a, a/2],   # back face
                [0, a/2, a/2],   # left face
                [a, a/2, a/2]    # right face
            ])
            return np.vstack([corners, face_centers])
    
        else:
            raise ValueError("lattice_type must be 'P', 'I', or 'F'")
    
    
    def plot_cubic_unit_cell(ax, lattice_type='P', a=1.0):
        """
        Plot cubic lattice unit cell
        """
        # Generate lattice points
        points = create_cubic_lattice_points(lattice_type, a)
    
        # Plot lattice points
        ax.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name='Lattice points'
        ))
    
        # Draw unit cell edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]
    
        corners = create_cubic_lattice_points('P', a)
    
        for edge in edges:
            ax.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
    
    
    # Create 3 subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('(a) Simple Cubic (P)', '(b) Body-Centered Cubic (I)', '(c) Face-Centered Cubic (F)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )
    
    # Plot each lattice type
    for i, lattice_type in enumerate(['P', 'I', 'F'], start=1):
        points = create_cubic_lattice_points(lattice_type, a=1.0)
    
        # Lattice points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=6, color='red'),
            showlegend=False
        ), row=1, col=i)
    
        # Unit cell edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        corners = create_cubic_lattice_points('P', a=1.0)
    
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=1, col=i)
    
    # Layout settings
    fig.update_layout(
        title_text="Three Bravais Lattices of Cubic System",
        height=500,
        showlegend=False
    )
    
    # Axis settings
    for i in range(1, 4):
        fig.update_scenes(
            xaxis=dict(range=[0, 1], title='x'),
            yaxis=dict(range=[0, 1], title='y'),
            zaxis=dict(range=[0, 1], title='z'),
            aspectmode='cube',
            row=1, col=i
        )
    
    fig.show()
    
    # Compare lattice point counts
    print("\n【Three Bravais Lattices of Cubic System】")
    print("=" * 60)
    for lattice_type, name, example in [('P', 'Simple Cubic', 'α-Po'),
                                          ('I', 'BCC', 'Fe, Cr, W'),
                                          ('F', 'FCC', 'Al, Cu, Au, Ag')]:
        points = create_cubic_lattice_points(lattice_type)
        print(f"{name:20s} | Lattice points: {len(points):2d} | Example: {example}")
    print("=" * 60)
    print("Note: Corner lattice points are shared by 8 unit cells, counted as 1/8 each")
    print("    P: 8×(1/8) = 1, I: 8×(1/8) + 1 = 2, F: 8×(1/8) + 6×(1/2) = 4")

### Code Example 3: Comparison of Four Orthorhombic Bravais Lattices

Visualize the differences among the four orthorhombic Bravais lattices (P, C, I, F).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - plotly>=5.14.0
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_orthorhombic_lattice_points(lattice_type='P', a=1.0, b=1.2, c=0.8):
        """
        Generate lattice point coordinates for orthorhombic lattices
    
        Parameters:
        -----------
        lattice_type : str
            'P', 'C', 'I', 'F'
        a, b, c : float
            Lattice constants (a ≠ b ≠ c)
        """
        # Corners (8 locations)
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]
        ])
    
        if lattice_type == 'P':
            return corners
    
        elif lattice_type == 'C':
            # C-centered: center of ab plane (z=0 and z=c, 2 locations)
            base_centers = np.array([
                [a/2, b/2, 0],
                [a/2, b/2, c]
            ])
            return np.vstack([corners, base_centers])
    
        elif lattice_type == 'I':
            # Body-centered: body center
            body_center = np.array([[a/2, b/2, c/2]])
            return np.vstack([corners, body_center])
    
        elif lattice_type == 'F':
            # Face-centered: centers of 6 faces
            face_centers = np.array([
                [a/2, b/2, 0],   # z=0 face
                [a/2, b/2, c],   # z=c face
                [a/2, 0, c/2],   # y=0 face
                [a/2, b, c/2],   # y=b face
                [0, b/2, c/2],   # x=0 face
                [a, b/2, c/2]    # x=a face
            ])
            return np.vstack([corners, face_centers])
    
        else:
            raise ValueError("lattice_type must be 'P', 'C', 'I', or 'F'")
    
    
    # Create 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('(a) Primitive (P)', '(b) C-centered (C)',
                        '(c) Body-centered (I)', '(d) Face-centered (F)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    a, b, c = 1.0, 1.3, 0.7
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    lattice_types = ['P', 'C', 'I', 'F']
    
    for (row, col), lattice_type in zip(positions, lattice_types):
        points = create_orthorhombic_lattice_points(lattice_type, a, b, c)
    
        # Lattice points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            showlegend=False
        ), row=row, col=col)
    
        # Unit cell edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        corners = create_orthorhombic_lattice_points('P', a, b, c)
    
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=row, col=col)
    
    # Layout settings
    fig.update_layout(
        title_text="Four Bravais Lattices of Orthorhombic System (a ≠ b ≠ c, α = β = γ = 90°)",
        height=800,
        showlegend=False
    )
    
    fig.show()
    
    # Compare lattice point counts
    print("\n【Four Bravais Lattices of Orthorhombic System】")
    print("=" * 70)
    print(f"{'Lattice Type':15s} | {'Points':8s} | {'Effective':12s} | Example")
    print("=" * 70)
    for lt, name, eff, example in [
        ('P', 'Primitive', 1, 'α-S'),
        ('C', 'C-centered', 2, 'U'),
        ('I', 'Body-centered', 2, 'TiO₂ (brookite)'),
        ('F', 'Face-centered', 4, 'NaNO₃')
    ]:
        points = create_orthorhombic_lattice_points(lt, a, b, c)
        print(f"{name:15s} | {len(points):8d} | {eff:12d} | {example}")
    print("=" * 70)

### Code Example 4: Retrieving Space Group Information with pymatgen

Now, let's perform practical space group analysis using pymatgen. First, learn how to retrieve basic space group information.
    
    
    from pymatgen.symmetry.groups import SpaceGroup
    
    # Retrieve information for representative space groups
    space_groups = [
        1,    # P1 (triclinic, lowest symmetry)
        2,    # P-1 (with inversion center)
        194,  # P6₃/mmc (HCP structure)
        225,  # Fm-3m (FCC structure)
        229   # Im-3m (BCC structure)
    ]
    
    print("=" * 100)
    print("Detailed Information on Representative Space Groups")
    print("=" * 100)
    
    for sg_num in space_groups:
        sg = SpaceGroup.from_int_number(sg_num)
    
        print(f"\n【Space Group {sg_num}】")
        print(f"  International symbol:     {sg.symbol}")
        print(f"  Crystal system:       {sg.crystal_system}")
        print(f"  Point group:         {sg.point_group}")
        print(f"  Number of symmetry operations:   {len(sg.symmetry_ops)}")
        print(f"  Centrosymmetric: {sg.is_centrosymmetric} (presence of inversion center)")
    
        # Display some symmetry operations (first 5)
        print(f"  Symmetry operations (first 5):")
        for i, op in enumerate(sg.symmetry_ops[:5], 1):
            print(f"    {i}. {op}")
    
        if len(sg.symmetry_ops) > 5:
            print(f"    ... plus {len(sg.symmetry_ops) - 5} more")
    
    print("\n" + "=" * 100)
    print("【Explanation】")
    print("- Number of symmetry operations varies by space group (No. 1: 1, No. 225: 192)")
    print("- Centrosymmetric indicates presence of inversion center")
    print("- Symmetry operations are represented by rotation matrices and translation vectors")
    print("=" * 100)

**Example output** :
    
    
    ====================================================================================================
    Detailed Information on Representative Space Groups
    ====================================================================================================
    
    【Space Group 1】
      International symbol:     P1
      Crystal system:       triclinic
      Point group:         1
      Number of symmetry operations:   1
      Centrosymmetric: False (presence of inversion center)
      Symmetry operations (first 5):
        1. Rot:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    tau
    [0. 0. 0.]
    
    【Space Group 225】
      International symbol:     Fm-3m
      Crystal system:       cubic
      Point group:         m-3m
      Number of symmetry operations:   192
      Centrosymmetric: True (presence of inversion center)
      Symmetry operations (first 5):
        1. Rot:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    tau
    [0. 0. 0.]
        2. Rot:
    [[-1.  0.  0.]
     [ 0. -1.  0.]
     [ 0.  0.  1.]]
    tau
    [0. 0. 0.]
        ... plus 187 more

### Code Example 5: List of Symmetry Operations for a Specific Space Group

Examine symmetry operations of space group 225 (Fm-3m, FCC structure) in detail.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Examine symmetry operations of space group 225 (Fm-3m, FCC s
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from pymatgen.symmetry.groups import SpaceGroup
    import numpy as np
    
    # Space group 225 (Fm-3m, FCC structure)
    sg = SpaceGroup.from_int_number(225)
    
    print("=" * 100)
    print(f"List of Symmetry Operations for Space Group {sg.int_number}: {sg.symbol}")
    print("=" * 100)
    
    print(f"Crystal system: {sg.crystal_system}")
    print(f"Point group: {sg.point_group}")
    print(f"Total number of symmetry operations: {len(sg.symmetry_ops)}")
    print(f"Inversion center: {'Present' if sg.is_centrosymmetric else 'Absent'}")
    
    print("\n【Classification of Symmetry Operations】")
    
    # Classify symmetry operations
    pure_translations = []
    rotations = []
    reflections = []
    inversions = []
    other_ops = []
    
    for i, op in enumerate(sg.symmetry_ops, 1):
        rotation_matrix = op.rotation_matrix
        translation_vector = op.translation_vector
    
        # Identity operation
        if np.allclose(rotation_matrix, np.eye(3)) and np.allclose(translation_vector, 0):
            continue  # Skip (identity operation)
    
        # Pure translation
        if np.allclose(rotation_matrix, np.eye(3)):
            pure_translations.append((i, op))
    
        # Inversion (-I)
        elif np.allclose(rotation_matrix, -np.eye(3)):
            inversions.append((i, op))
    
        # Other operations
        else:
            # Determine rotation/reflection from determinant
            det = np.linalg.det(rotation_matrix)
            if np.isclose(det, 1):
                rotations.append((i, op))
            elif np.isclose(det, -1):
                reflections.append((i, op))
            else:
                other_ops.append((i, op))
    
    print(f"  Identity operation:        1")
    print(f"  Pure translations:      {len(pure_translations)}")
    print(f"  Rotation operations:        {len(rotations)}")
    print(f"  Reflection operations:        {len(reflections)}")
    print(f"  Inversion operations:        {len(inversions)}")
    print(f"  Other:          {len(other_ops)}")
    print(f"  Total:            {1 + len(pure_translations) + len(rotations) + len(reflections) + len(inversions) + len(other_ops)}")
    
    # Display details of pure translations
    print("\n【Pure Translation Operations (first 10)】")
    for i, (num, op) in enumerate(pure_translations[:10], 1):
        print(f"  {num:3d}. Translation vector: {op.translation_vector}")
    
    # Display details of rotation operations (first 5)
    print("\n【Rotation Operations (first 5)】")
    for i, (num, op) in enumerate(rotations[:5], 1):
        print(f"  {num:3d}. Rotation matrix:")
        print(f"       {op.rotation_matrix}")
        print(f"       Translation: {op.translation_vector}")
    
    print("\n" + "=" * 100)
    print("【Explanation】")
    print("- Fm-3m (FCC) has the highest symmetry among cubic systems")
    print("- 192 symmetry operations = 48 point symmetry operations × 4 translations (FCC lattice characteristic)")
    print("- This high symmetry makes physical properties (elastic moduli, optical properties, etc.) isotropic")
    print("=" * 100)

### Code Example 6: Space Group Determination from Crystal Structure

Show an example where pymatgen automatically determines the space group from a known crystal structure.
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # Example 1: FCC structure aluminum (Al)
    # Lattice constant: a = 4.05 Å
    lattice_al = Lattice.cubic(4.05)
    
    # FCC structure: atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    al_structure = Structure(
        lattice_al,
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    
    # Space group analysis
    sga_al = SpacegroupAnalyzer(al_structure)
    
    print("=" * 100)
    print("【Example 1: Crystal Structure Analysis of Aluminum (Al)】")
    print("=" * 100)
    print(f"Lattice constant: a = 4.05 Å (cubic)")
    print(f"Number of atoms: {len(al_structure)}")
    print(f"\nSpace group number:       {sga_al.get_space_group_number()}")
    print(f"Space group symbol:       {sga_al.get_space_group_symbol()}")
    print(f"Point group:             {sga_al.get_point_group_symbol()}")
    print(f"Crystal system:           {sga_al.get_crystal_system()}")
    print(f"Lattice type:       {sga_al.get_lattice_type()}")
    
    # Example 2: BCC structure iron (Fe)
    lattice_fe = Lattice.cubic(2.87)
    
    # BCC structure: atoms at (0,0,0) and (0.5,0.5,0.5)
    fe_structure = Structure(
        lattice_fe,
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    sga_fe = SpacegroupAnalyzer(fe_structure)
    
    print("\n" + "=" * 100)
    print("【Example 2: Crystal Structure Analysis of Iron (Fe)】")
    print("=" * 100)
    print(f"Lattice constant: a = 2.87 Å (cubic)")
    print(f"Number of atoms: {len(fe_structure)}")
    print(f"\nSpace group number:       {sga_fe.get_space_group_number()}")
    print(f"Space group symbol:       {sga_fe.get_space_group_symbol()}")
    print(f"Point group:             {sga_fe.get_point_group_symbol()}")
    print(f"Crystal system:           {sga_fe.get_crystal_system()}")
    print(f"Lattice type:       {sga_fe.get_lattice_type()}")
    
    # Example 3: HCP structure magnesium (Mg)
    # HCP: a = 3.21 Å, c = 5.21 Å, c/a = 1.624
    lattice_mg = Lattice.hexagonal(a=3.21, c=5.21)
    
    # HCP structure: atoms at (0,0,0), (1/3, 2/3, 1/2)
    mg_structure = Structure(
        lattice_mg,
        ["Mg", "Mg"],
        [[0, 0, 0], [1/3, 2/3, 0.5]]
    )
    
    sga_mg = SpacegroupAnalyzer(mg_structure)
    
    print("\n" + "=" * 100)
    print("【Example 3: Crystal Structure Analysis of Magnesium (Mg)】")
    print("=" * 100)
    print(f"Lattice constants: a = 3.21 Å, c = 5.21 Å (hexagonal)")
    print(f"c/a ratio: {5.21/3.21:.3f}")
    print(f"Number of atoms: {len(mg_structure)}")
    print(f"\nSpace group number:       {sga_mg.get_space_group_number()}")
    print(f"Space group symbol:       {sga_mg.get_space_group_symbol()}")
    print(f"Point group:             {sga_mg.get_point_group_symbol()}")
    print(f"Crystal system:           {sga_mg.get_crystal_system()}")
    print(f"Lattice type:       {sga_mg.get_lattice_type()}")
    
    print("\n" + "=" * 100)
    print("【Explanation】")
    print("- Al (FCC): Space group 225 Fm-3m")
    print("- Fe (BCC): Space group 229 Im-3m")
    print("- Mg (HCP): Space group 194 P6₃/mmc")
    print("- pymatgen can automatically determine space groups from atomic coordinates")
    print("=" * 100)

### Code Example 7: Calculating Equivalent Positions

Calculate equivalent positions generated from a single atomic position by symmetry operations of a space group.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculate equivalent positions generated from a single atomi
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from pymatgen.symmetry.groups import SpaceGroup
    import numpy as np
    
    # Space group 225 (Fm-3m, FCC structure)
    sg = SpaceGroup.from_int_number(225)
    
    print("=" * 100)
    print(f"Generation of Equivalent Positions in Space Group {sg.int_number}: {sg.symbol}")
    print("=" * 100)
    
    # General position example: (x, y, z) = (0.3, 0.2, 0.1)
    original_position = np.array([0.3, 0.2, 0.1])
    
    print(f"\nOriginal atomic position: ({original_position[0]:.2f}, {original_position[1]:.2f}, {original_position[2]:.2f})")
    print(f"\nEquivalent positions generated by symmetry operations:")
    
    # Set to store equivalent positions (for duplicate removal)
    equivalent_positions = set()
    
    for i, op in enumerate(sg.symmetry_ops, 1):
        # Apply symmetry operation
        new_position = op.operate(original_position)
    
        # Apply periodic boundary conditions (0 ≤ coord < 1)
        new_position = new_position % 1.0
    
        # Convert to tuple and add to set (for duplicate checking)
        pos_tuple = tuple(np.round(new_position, 6))
        equivalent_positions.add(pos_tuple)
    
    # Display results
    print(f"\nTotal number of equivalent positions: {len(equivalent_positions)}\n")
    
    for i, pos in enumerate(sorted(equivalent_positions), 1):
        print(f"  {i:3d}. ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})")
    
    # Special position example: (0, 0, 0) - high symmetry point
    print("\n" + "=" * 100)
    print("【Special Position Example: (0, 0, 0)】")
    print("=" * 100)
    
    special_position = np.array([0.0, 0.0, 0.0])
    special_positions = set()
    
    for op in sg.symmetry_ops:
        new_position = op.operate(special_position)
        new_position = new_position % 1.0
        pos_tuple = tuple(np.round(new_position, 6))
        special_positions.add(pos_tuple)
    
    print(f"Total number of equivalent positions: {len(special_positions)}")
    print("→ Special positions (high symmetry points) generate fewer equivalent positions than general positions")
    
    print("\n" + "=" * 100)
    print("【Explanation】")
    print("- General position (x, y, z): many equivalent positions generated by symmetry operations")
    print("- Special position (on symmetry elements): fewer equivalent positions (e.g., (0,0,0) only 1)")
    print("- General position in Fm-3m multiplies 192-fold (192 symmetry operations)")
    print("- In crystal structure description, using special positions requires fewer atoms")
    print("=" * 100)

### Code Example 8: Retrieving Crystal Examples from Space Group Number (Conceptual)

Finally, introduce how to search for real materials with specific space groups using Materials Project API integration (requires API key).
    
    
    """
    Example of space group search using Materials Project API
    
    Note: This code requires a Materials Project API key.
    You can obtain an API key by registering for free at https://next-gen.materialsproject.org/
    
    Installation:
    pip install mp-api
    """
    
    # The following is conceptual code (requires API key)
    
    from mp_api.client import MPRester
    
    def search_materials_by_space_group(space_group_number, api_key=None, max_results=5):
        """
        Search for materials with a specific space group
    
        Parameters:
        -----------
        space_group_number : int
            Space group number (1-230)
        api_key : str
            Materials Project API key
        max_results : int
            Maximum number of results to retrieve
    
        Returns:
        --------
        list : Search results (list of material ID, formula, space group symbol)
        """
        if api_key is None:
            print("Error: API key required")
            print("How to obtain: Register for free at https://next-gen.materialsproject.org/")
            return []
    
        with MPRester(api_key) as mpr:
            # Search by space group
            docs = mpr.materials.summary.search(
                spacegroup_number=space_group_number,
                fields=["material_id", "formula_pretty", "symmetry"],
                num_chunks=1
            )
    
            results = []
            for i, doc in enumerate(docs[:max_results]):
                results.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'space_group': doc.symmetry.symbol,
                    'crystal_system': doc.symmetry.crystal_system
                })
    
            return results
    
    
    # Usage example (if you have API key)
    # API_KEY = "your_api_key_here"
    # results = search_materials_by_space_group(225, api_key=API_KEY, max_results=10)
    
    # Demo output without API key
    print("=" * 100)
    print("【Representative Materials with Space Group 225 (Fm-3m, FCC)】")
    print("=" * 100)
    
    # Known material examples (manual list)
    fcc_materials = [
        {'formula': 'Al', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Aluminum'},
        {'formula': 'Cu', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Copper'},
        {'formula': 'Au', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Gold'},
        {'formula': 'Ag', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Silver'},
        {'formula': 'Ni', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Nickel'},
        {'formula': 'Pt', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Platinum'},
        {'formula': 'Pb', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Lead'},
        {'formula': 'NaCl', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Sodium chloride'},
        {'formula': 'MgO', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'Magnesium oxide'},
    ]
    
    for i, mat in enumerate(fcc_materials, 1):
        print(f"{i:2d}. {mat['formula']:10s} | {mat['space_group']:10s} | {mat['description']}")
    
    print("\n" + "=" * 100)
    print("【Representative Materials with Space Group 229 (Im-3m, BCC)】")
    print("=" * 100)
    
    bcc_materials = [
        {'formula': 'Fe', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Iron (α-phase, room temp)'},
        {'formula': 'Cr', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Chromium'},
        {'formula': 'W', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Tungsten'},
        {'formula': 'Mo', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Molybdenum'},
        {'formula': 'V', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Vanadium'},
        {'formula': 'Na', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Sodium'},
        {'formula': 'K', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'Potassium'},
    ]
    
    for i, mat in enumerate(bcc_materials, 1):
        print(f"{i:2d}. {mat['formula']:10s} | {mat['space_group']:10s} | {mat['description']}")
    
    print("\n" + "=" * 100)
    print("【Explanation】")
    print("- Using Materials Project API (https://next-gen.materialsproject.org/),")
    print("  you can search for real materials by space group number (free API key registration required)")
    print("- The above examples are manually listed representative materials")
    print("- FCC (No. 225) and BCC (No. 229) are the most common structures in metallic materials")
    print("=" * 100)

* * *

## 2.6 Practice Problems

**Exercise 1: Classification of 14 Bravais Lattices**

**Problem** : Crystals with the following lattice parameters are classified into which crystal system and which Bravais lattice?

  1. $a = b = c = 5.0$ Å, $\alpha = \beta = \gamma = 90°$, lattice points only at corners
  2. $a = b = 3.2$ Å, $c = 5.1$ Å, $\alpha = \beta = 90°$, $\gamma = 120°$
  3. $a = 4.0$ Å, $b = 5.0$ Å, $c = 6.0$ Å, $\alpha = \beta = \gamma = 90°$, lattice point at body center
  4. $a = b = c = 4.05$ Å, $\alpha = \beta = \gamma = 90°$, lattice points at centers of all faces

**Answers** :

  1. **Cubic system, P (simple cubic)** \- all equal, 90°, corners only
  2. **Hexagonal system, P** \- $a = b \neq c$, $\gamma = 120°$
  3. **Orthorhombic system, I (body-centered)** \- $a \neq b \neq c$, all 90°, body center
  4. **Cubic system, F (FCC)** \- all equal, 90°, face-centered

**Exercise 2: Packing Fraction Comparison of Three Cubic Bravais Lattices**

**Problem** : For the three cubic Bravais lattices (P, I, F), calculate the packing fraction when atoms are treated as rigid spheres. Let lattice constant be $a$ and atomic radius be $r$.

**Hint** :

  * Packing fraction = (Volume of atoms) / (Volume of unit cell)
  * Volume of atom = $\frac{4}{3}\pi r^3$
  * Volume of unit cell = $a^3$
  * P: Atoms only at corners, nearest-neighbor distance = $a$
  * I (BCC): Atoms at corners + body center, nearest-neighbor distance = $\frac{\sqrt{3}}{2}a$
  * F (FCC): Atoms at corners + face centers, nearest-neighbor distance = $\frac{a}{\sqrt{2}}$

**Answers** :

**1\. Simple Cubic (P)** :

  * Atoms per unit cell: $8 \times \frac{1}{8} = 1$
  * Nearest-neighbor distance: $2r = a$ → $r = \frac{a}{2}$
  * Packing fraction: $\frac{1 \times \frac{4}{3}\pi r^3}{a^3} = \frac{\frac{4}{3}\pi (\frac{a}{2})^3}{a^3} = \frac{\pi}{6} \approx 0.524$ (52.4%)

**2\. BCC (I)** :

  * Atoms per unit cell: $8 \times \frac{1}{8} + 1 = 2$
  * Nearest-neighbor distance: $2r = \frac{\sqrt{3}}{2}a$ → $r = \frac{\sqrt{3}}{4}a$
  * Packing fraction: $\frac{2 \times \frac{4}{3}\pi r^3}{a^3} = \frac{2 \times \frac{4}{3}\pi (\frac{\sqrt{3}}{4}a)^3}{a^3} = \frac{\sqrt{3}\pi}{8} \approx 0.680$ (68.0%)

**3\. FCC (F)** :

  * Atoms per unit cell: $8 \times \frac{1}{8} + 6 \times \frac{1}{2} = 4$
  * Nearest-neighbor distance: $2r = \frac{a}{\sqrt{2}}$ → $r = \frac{a}{2\sqrt{2}}$
  * Packing fraction: $\frac{4 \times \frac{4}{3}\pi r^3}{a^3} = \frac{4 \times \frac{4}{3}\pi (\frac{a}{2\sqrt{2}})^3}{a^3} = \frac{\pi}{3\sqrt{2}} \approx 0.740$ (74.0%)

**Conclusion** : FCC (74.0%) > BCC (68.0%) > Simple Cubic (52.4%)

**Exercise 3: Identifying Types of Symmetry Operations**

**Problem** : Classify the following symmetry operations by type:

  1. 90° rotation around z-axis
  2. Reflection with respect to xy plane
  3. Inversion with respect to origin: $(x, y, z) \rightarrow (-x, -y, -z)$
  4. 180° rotation followed by translation in rotation axis direction by half lattice vector
  5. Reflection followed by translation parallel to reflection plane by half lattice vector

**Answers** :

  1. **4-fold rotation** \- Symbol: 4
  2. **Mirror reflection** \- Symbol: m
  3. **Inversion** \- Symbol: $\bar{1}$ or i
  4. **Screw axis** \- Symbol: 2₁ (2-fold screw)
  5. **Glide reflection** \- Symbol: a, b, c, n, d (varies by direction)

**Exercise 4: Reading Hermann-Mauguin Symbols from Space Group Numbers**

**Problem** : Explain the meaning of the following space group symbols.

  1. P6₃/mmc (Space group 194)
  2. Fm$\bar{3}$m (Space group 225)
  3. Im$\bar{3}$m (Space group 229)

**Answers** :

**1\. P6₃/mmc** :

  * **P** : Primitive lattice
  * **6₃** : 6-fold screw axis
  * **/** : Perpendicular symmetry element
  * **m** : Mirror plane
  * **m** : Another mirror plane
  * **c** : c-glide (glide reflection in c-axis direction)
  * Hexagonal system, HCP structure (Mg, Zn, Ti, etc.)

**2\. Fm$\bar{3}$m** :

  * **F** : Face-centered lattice
  * **m** : Mirror plane
  * **$\bar{3}$** : 3-fold rotoinversion axis
  * **m** : Another mirror plane
  * Cubic system, FCC structure (Al, Cu, Au, Ag, etc.)

**3\. Im$\bar{3}$m** :

  * **I** : Body-centered lattice
  * **m** : Mirror plane
  * **$\bar{3}$** : 3-fold rotoinversion axis
  * **m** : Another mirror plane
  * Cubic system, BCC structure (Fe, Cr, W, etc.)

**Exercise 5: Investigating Space Group of Specific Crystal with pymatgen**

**Problem** : Use pymatgen to investigate the space group of diamond (C) crystal structure. Diamond has the following characteristics:

  * Lattice constant: $a = 3.567$ Å (cubic)
  * Atomic coordinates: (0, 0, 0), (0.25, 0.25, 0.25), (0.5, 0.5, 0), (0.75, 0.75, 0.25), (0.5, 0, 0.5), (0.75, 0.25, 0.75), (0, 0.5, 0.5), (0.25, 0.75, 0.75)

**Answer Code** :
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # Diamond structure
    lattice_diamond = Lattice.cubic(3.567)
    
    diamond_structure = Structure(
        lattice_diamond,
        ["C"] * 8,
        [
            [0, 0, 0], [0.25, 0.25, 0.25],
            [0.5, 0.5, 0], [0.75, 0.75, 0.25],
            [0.5, 0, 0.5], [0.75, 0.25, 0.75],
            [0, 0.5, 0.5], [0.25, 0.75, 0.75]
        ]
    )
    
    sga = SpacegroupAnalyzer(diamond_structure)
    
    print("【Diamond (C) Crystal Structure Analysis】")
    print(f"Space group number: {sga.get_space_group_number()}")
    print(f"Space group symbol: {sga.get_space_group_symbol()}")
    print(f"Point group: {sga.get_point_group_symbol()}")
    print(f"Crystal system: {sga.get_crystal_system()}")
    print(f"Lattice type: {sga.get_lattice_type()}")

**Expected Output** :
    
    
    【Diamond (C) Crystal Structure Analysis】
    Space group number: 227
    Space group symbol: Fd-3m
    Point group: m-3m
    Crystal system: cubic
    Lattice type: face_centered

**Explanation** :

  * Diamond is space group 227 Fd-3m (face-centered diamond cubic)
  * Structure with 2-atom basis on FCC lattice (diamond cubic structure)
  * Si and Ge also have this structure

* * *

## 2.7 Chapter Summary

### What We Learned

  1. **14 Bravais Lattices**
     * Combination of 7 crystal systems and 4 centering types (P, C, I, F)
     * Three cubic types (P, I, F) are most important in materials science
     * Hexagonal (HCP) and cubic systems (FCC, BCC) are main structures in metallic materials
  2. **Symmetry Operations**
     * Point symmetry operations: rotation (1, 2, 3, 4, 6), mirror (m), inversion ($\bar{1}$), rotoinversion ($\bar{2}, \bar{3}, \bar{4}, \bar{6}$)
     * Translational symmetry operations: screw, glide
     * Crystallographic restriction: 5-fold and 7-fold or higher rotational symmetries do not exist
  3. **Space Groups**
     * 230 space groups classify all crystal structures
     * Space group = point group (32 types) + translational symmetry operations
     * Represented by Hermann-Mauguin notation (international symbols)
     * General positions and special positions (Wyckoff positions)
  4. **Practical Use with pymatgen**
     * Retrieving and analyzing space group information
     * Automatic space group determination from crystal structure
     * Generating equivalent positions through symmetry operations
     * Integration with Materials Project API (material search)

### Important Points

  * Bravais lattices describe the "framework" of crystals
  * Space groups describe the "complete symmetry" of crystals
  * Same Bravais lattice can have different space groups due to atomic arrangement (basis)
  * FCC (No. 225), BCC (No. 229), HCP (No. 194) are the most important space groups
  * Pymatgen is a powerful tool for crystallographic analysis

### To the Next Chapter

In Chapter 3, we will learn about **Miller Indices and Reciprocal Lattice** :

  * Notation of crystal planes and directions using Miller indices
  * Calculation of interplanar spacing
  * Reciprocal lattice concept and application to X-ray diffraction
  * Ewald sphere and Bragg condition
  * Calculation of reciprocal lattice vectors using pymatgen
