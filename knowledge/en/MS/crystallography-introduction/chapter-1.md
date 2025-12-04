---
title: "Chapter 1: Fundamentals of Crystallography and Lattice Concepts"
chapter_title: "Chapter 1: Fundamentals of Crystallography and Lattice Concepts"
subtitle: Understanding the Regularity and Beauty of Atomic Arrangements
reading_time: 26-32 minutes
difficulty: Introductory
code_examples: 8
version: 1.0
created_at: "by:"
---

Crystallography is the discipline that studies the regularity of atomic arrangements in materials. In this chapter, we will learn the differences between crystalline and amorphous materials, the concepts of lattice points and unit cells, and the characteristics of the seven crystal systems. Let's step into the beautiful world of crystallography while visualizing lattice structures with Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the differences between crystalline and amorphous materials
  * ✅ Master the concepts of lattice points, unit cells, and lattice parameters
  * ✅ Understand the characteristics of the seven crystal systems
  * ✅ Understand the meaning of lattice parameters (a, b, c, α, β, γ)
  * ✅ Visualize basic lattice structures using Python

* * *

## 1.1 What is Crystallography?

### Definition and Importance of Crystallography

**Crystallography** is the scientific discipline that studies the atomic arrangement, symmetry, and structure of crystals. It serves as foundational knowledge in many fields including materials science, chemistry, physics, mineralogy, and biology.

> **Crystallography** is the discipline that studies the internal structure, symmetry, and physical/chemical properties of crystalline materials. 

**Why Crystallography is Important** : Crystallography is essential for **understanding material properties** since atomic arrangements determine mechanical, electrical, and optical properties. It provides the foundation for **materials design** , enabling the creation of materials with desired properties. It also serves as the basis for **X-ray diffraction analysis** , the primary technique for experimentally determining crystal structures, and drives **new materials development** across various fields including semiconductors, catalysts, and pharmaceuticals.

### Differences Between Crystalline and Amorphous Materials

There are two broad patterns of atomic arrangements in materials:

#### 1\. Crystalline (Crystal)

**Definition** : A material in which atoms or molecules are **arranged regularly**

**Characteristics** :

  * **Long-range Order** : The regularity of atomic arrangement extends over long distances
  * **Periodicity** : The same pattern repeats
  * **Anisotropy** : Properties differ depending on direction (e.g., cleavage, refractive index)
  * **Sharp Melting Point** : Transitions from solid to liquid at a constant temperature
  * **Sharp X-ray Diffraction Peaks** : Strong diffraction at specific angles due to regular structure

**Representative Examples** :

  * **Sodium Chloride (NaCl)** : Table salt, cubic crystal system
  * **Silicon (Si)** : Semiconductor, diamond structure
  * **Iron (Fe)** : Structural material, body-centered cubic lattice
  * **Quartz (SiO₂)** : Hexagonal crystal system

#### 2\. Amorphous

**Definition** : A material in which atoms or molecules are **arranged irregularly**

**Characteristics** :

  * **Short-range Order Only** : Regular at short distances but disordered at long distances
  * **No Periodicity** : No repeating pattern exists
  * **Isotropy** : Properties are the same in all directions
  * **Glass Transition Temperature** : Gradually softens (no clear melting point)
  * **Broad X-ray Diffraction Pattern** : No sharp peaks appear

**Representative Examples** :

  * **Glass (SiO₂ glass)** : Window glass, optical fiber
  * **Amorphous Silicon (a-Si)** : Solar cells, thin-film transistors
  * **Polymer Glass** : Polystyrene, PMMA
  * **Metallic Glass** : Alloys produced by rapid solidification

#### Comparison Table of Crystalline and Amorphous Materials

Property | Crystalline | Amorphous  
---|---|---  
**Atomic Arrangement** | Regular (long-range order) | Irregular (short-range order only)  
**Periodicity** | Present | Absent  
**Symmetry** | High (point group, space group) | Low (isotropic)  
**Anisotropy** | Present (direction dependent) | Absent (isotropic)  
**Melting** | Sharp melting point | Glass transition (gradual softening)  
**X-ray Diffraction** | Sharp peaks (Bragg reflection) | Broad halo  
**Density** | High (high packing efficiency) | Somewhat lower (more voids)  
**Thermodynamic Stability** | Stable (minimum free energy) | Metastable (can crystallize)  
  
### Concepts of Periodicity and Symmetry

There are two important concepts for understanding crystal structures:

#### 1\. Periodicity

**Definition** : The repetition of the same pattern at regular intervals

In crystals, atomic arrangements repeat periodically in three-dimensional space. The minimum repeating unit is called the **unit cell**.

**Mathematical Expression** :

When an atom exists at position $\vec{r}$, the same atomic arrangement also exists at the following positions:

$$\vec{r}' = \vec{r} + n_1\vec{a} + n_2\vec{b} + n_3\vec{c}$$

where $\vec{a}, \vec{b}, \vec{c}$ are lattice vectors and $n_1, n_2, n_3$ are integers.

#### 2\. Symmetry

**Definition** : The property of being indistinguishable from the original state after performing a certain operation

**Types of Symmetry Operations** :

  * **Translational Symmetry (Translation)** : Parallel displacement in a certain direction
  * **Rotational Symmetry (Rotation)** : Rotation around an axis (1-fold, 2-fold, 3-fold, 4-fold, 6-fold symmetry)
  * **Mirror Symmetry (Reflection)** : Inversion with respect to a mirror plane
  * **Inversion Symmetry (Inversion)** : Inversion with respect to a point

**Important Constraint** : In crystals, rotational symmetries compatible with periodicity are **only 1-fold, 2-fold, 3-fold, 4-fold, and 6-fold** (5-fold symmetry does not exist).

* * *

## 1.2 Lattice Points and Unit Cells

### Lattice Point

**Definition** : A reference point of a periodically repeating structure

A lattice point is not the atom itself, but **an abstract point indicating positions with identical atomic arrangements**. The environment surrounding all lattice points is identical.

**Important Properties** : All lattice points are equivalent, meaning the atomic arrangement as seen from any lattice point is the same. However, the arrangement of lattice points alone does not reveal the types or positions of atoms.

### Unit Cell

**Definition** : The minimum repeating unit that represents the crystal structure

By repeatedly arranging the unit cell in three-dimensional space, an infinitely extending crystal structure can be constructed.

**Ways to Choose a Unit Cell** : The **primitive cell** is the minimum unit with lattice points only at vertices, containing 1 lattice point. The **conventional cell** is a unit that more easily expresses symmetry and may contain multiple lattice points.

### Lattice Parameters

The shape of a unit cell is completely described by **six lattice parameters** :

#### Length Parameters (3)

The three length parameters are **$a$** (length of the unit cell in the x-direction, unit: Å = 10⁻¹⁰ m), **$b$** (length in the y-direction), and **$c$** (length in the z-direction).

#### Angle Parameters (3)

The three angle parameters are **$\alpha$** (angle between $b$ and $c$, unit: °), **$\beta$** (angle between $c$ and $a$), and **$\gamma$** (angle between $a$ and $b$).

**Unit Cell Volume** :

From the lattice parameters, the volume $V$ of the unit cell can be calculated:

$$V = abc\sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma}$$

In special cases, simpler formulas apply: for **cubic** systems $V = a^3$, for **tetragonal** systems $V = a^2c$, and for **orthorhombic** systems $V = abc$.

* * *

## 1.3 The Seven Crystal Systems

All crystal structures are classified into **seven crystal systems** based on symmetry. This is determined by the relationships among lattice parameters and symmetry.
    
    
    ```mermaid
    graph TD
        A[Seven Crystal Systems] --> B[Triclinic]
        A --> C[Monoclinic]
        A --> D[Orthorhombic]
        A --> E[Tetragonal]
        A --> F[Hexagonal]
        A --> G[Trigonal]
        A --> H[Cubic]
    
        B --> B1[Lowest Symmetrya≠b≠c, α≠β≠γ≠90°]
        C --> C1[One 2-fold Axisa≠b≠c, α=γ=90°≠β]
        D --> D1[Three Perpendicular 2-fold Axesa≠b≠c, α=β=γ=90°]
        E --> E1[One 4-fold Axisa=b≠c, α=β=γ=90°]
        F --> F1[One 6-fold Axisa=b≠c, α=β=90°, γ=120°]
        G --> G1[One 3-fold Axisa=b=c, α=β=γ≠90°]
        H --> H1[Highest Symmetrya=b=c, α=β=γ=90°]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style C fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style D fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style E fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style F fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style G fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style H fill:#fce7f3,stroke:#f093fb,stroke-width:1px
    ```

### 1\. Triclinic

**Lattice Parameter Relationships** :

$$a \neq b \neq c, \quad \alpha \neq \beta \neq \gamma \neq 90°$$

**Symmetry** : Lowest (translation and inversion only)

**Representative Examples** :

  * Feldspar (Albite, NaAlSi₃O₈)
  * Copper sulfate pentahydrate (CuSO₄·5H₂O)

### 2\. Monoclinic

**Lattice Parameter Relationships** :

$$a \neq b \neq c, \quad \alpha = \gamma = 90° \neq \beta$$

**Symmetry** : One 2-fold rotation axis or mirror plane

**Representative Examples** :

  * Gypsum (CaSO₄·2H₂O)
  * Monoclinic sulfur (S)

### 3\. Orthorhombic

**Lattice Parameter Relationships** :

$$a \neq b \neq c, \quad \alpha = \beta = \gamma = 90°$$

**Symmetry** : Three mutually perpendicular 2-fold rotation axes

**Representative Examples** :

  * Orthorhombic sulfur (α-S)
  * Topaz (Al₂SiO₄(F,OH)₂)
  * Barite (BaSO₄)

### 4\. Tetragonal

**Lattice Parameter Relationships** :

$$a = b \neq c, \quad \alpha = \beta = \gamma = 90°$$

**Symmetry** : One 4-fold rotation axis (c-axis)

**Representative Examples** :

  * Rutile titanium dioxide (TiO₂)
  * Zircon (ZrSiO₄)
  * Tin (β-Sn, white tin)

### 5\. Hexagonal

**Lattice Parameter Relationships** :

$$a = b \neq c, \quad \alpha = \beta = 90°, \quad \gamma = 120°$$

**Symmetry** : One 6-fold rotation axis (c-axis)

**Representative Examples** :

  * Quartz (α-SiO₂)
  * Beryllium (Be)
  * Magnesium (Mg)
  * Zinc (Zn)
  * Graphite (C)

### 6\. Trigonal / Rhombohedral

**Lattice Parameter Relationships** :

$$a = b = c, \quad \alpha = \beta = \gamma \neq 90°$$

**Symmetry** : One 3-fold rotation axis

**Representative Examples** :

  * Calcite (CaCO₃)
  * Corundum (α-Al₂O₃)
  * Mercury (Hg, low-temperature phase)

**Note** : The trigonal crystal system is sometimes described with hexagonal lattices (hexagonal lattice + 3-fold rotation axis).

### 7\. Cubic

**Lattice Parameter Relationships** :

$$a = b = c, \quad \alpha = \beta = \gamma = 90°$$

**Symmetry** : Highest (four 3-fold rotation axes)

**Representative Examples** :

  * Sodium chloride (NaCl)
  * Diamond (C)
  * Silicon (Si)
  * Iron (α-Fe, ferrite)
  * Copper (Cu)
  * Gold (Au)

### Comparison Table of the Seven Crystal Systems

Crystal System | Axis Length Relationship | Angle Relationship | Key Symmetry Element | Representative Examples  
---|---|---|---|---  
**Triclinic** | $a \neq b \neq c$ | $\alpha \neq \beta \neq \gamma \neq 90°$ | Inversion center only | Feldspar, CuSO₄·5H₂O  
**Monoclinic** | $a \neq b \neq c$ | $\alpha = \gamma = 90° \neq \beta$ | One 2-fold axis | Gypsum, monoclinic sulfur  
**Orthorhombic** | $a \neq b \neq c$ | $\alpha = \beta = \gamma = 90°$ | Three perpendicular 2-fold axes | Orthorhombic sulfur, BaSO₄  
**Tetragonal** | $a = b \neq c$ | $\alpha = \beta = \gamma = 90°$ | One 4-fold axis | TiO₂, β-Sn  
**Hexagonal** | $a = b \neq c$ | $\alpha = \beta = 90°, \gamma = 120°$ | One 6-fold axis | Quartz, Mg, Zn  
**Trigonal** | $a = b = c$ | $\alpha = \beta = \gamma \neq 90°$ | One 3-fold axis | CaCO₃, α-Al₂O₃  
**Cubic** | $a = b = c$ | $\alpha = \beta = \gamma = 90°$ | Four 3-fold axes | NaCl, Si, Fe, Cu  
  
* * *

## 1.4 Visualization of Lattices with Python

Now, let's use Python to visualize lattice structures and understand crystallography concepts visually.

### Environment Setup

Install the required libraries:
    
    
    # Install required libraries
    pip install numpy matplotlib pandas plotly
    

### Code Example 1: Creating a Table of Lattice Parameters for the Seven Crystal Systems

First, let's create a table summarizing the relationships of lattice parameters for the seven crystal systems.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: First, let's create a table summarizing the relationships of
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    # Define lattice parameter relationships for the seven crystal systems
    crystal_systems_data = {
        'Crystal System': ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
                           'Hexagonal', 'Trigonal', 'Cubic'],
        'English Name': ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
                         'Hexagonal', 'Trigonal', 'Cubic'],
        'Axis Length': ['a≠b≠c', 'a≠b≠c', 'a≠b≠c', 'a=b≠c', 'a=b≠c', 'a=b=c', 'a=b=c'],
        'α': ['≠90°', '90°', '90°', '90°', '90°', '≠90°', '90°'],
        'β': ['≠90°', '≠90°', '90°', '90°', '90°', '=α', '90°'],
        'γ': ['≠90°', '90°', '90°', '90°', '120°', '=α', '90°'],
        'Key Symmetry': ['Inversion only', 'One 2-fold axis', 'Three 2-fold axes', 'One 4-fold axis',
                         'One 6-fold axis', 'One 3-fold axis', 'Four 3-fold axes'],
        'Example': ['Feldspar', 'Gypsum', 'Orthorhombic S', 'TiO₂', 'Mg', 'CaCO₃', 'NaCl']
    }
    
    # Create DataFrame
    df_crystal_systems = pd.DataFrame(crystal_systems_data)
    
    # Display
    print("=" * 100)
    print("Summary of Lattice Parameters for the Seven Crystal Systems")
    print("=" * 100)
    print(df_crystal_systems.to_string(index=False))
    print("=" * 100)
    
    # Specific examples of lattice parameters
    print("\nSpecific Examples of Lattice Parameters (Representative Materials):")
    print("-" * 80)
    
    examples = [
        {'Material': 'NaCl (Sodium Chloride)', 'System': 'Cubic',
         'a': 5.64, 'b': 5.64, 'c': 5.64, 'α': 90, 'β': 90, 'γ': 90},
        {'Material': 'Si (Silicon)', 'System': 'Cubic',
         'a': 5.43, 'b': 5.43, 'c': 5.43, 'α': 90, 'β': 90, 'γ': 90},
        {'Material': 'TiO₂ (Rutile)', 'System': 'Tetragonal',
         'a': 4.59, 'b': 4.59, 'c': 2.96, 'α': 90, 'β': 90, 'γ': 90},
        {'Material': 'Mg (Magnesium)', 'System': 'Hexagonal',
         'a': 3.21, 'b': 3.21, 'c': 5.21, 'α': 90, 'β': 90, 'γ': 120},
        {'Material': 'α-Fe (Ferrite)', 'System': 'Cubic',
         'a': 2.87, 'b': 2.87, 'c': 2.87, 'α': 90, 'β': 90, 'γ': 90},
    ]
    
    for ex in examples:
        print(f"{ex['Material']:30s} ({ex['System']})")
        print(f"  a={ex['a']:.2f}Å, b={ex['b']:.2f}Å, c={ex['c']:.2f}Å")
        print(f"  α={ex['α']}°, β={ex['β']}°, γ={ex['γ']}°")
        print()
    

**Output Example** :
    
    
    ====================================================================================================
    Summary of Lattice Parameters for the Seven Crystal Systems
    ====================================================================================================
    Crystal System  English Name  Axis Length  α    β    γ    Key Symmetry        Example
    Triclinic       Triclinic     a≠b≠c        ≠90° ≠90° ≠90° Inversion only      Feldspar
    Monoclinic      Monoclinic    a≠b≠c        90°  ≠90° 90°  One 2-fold axis     Gypsum
    Orthorhombic    Orthorhombic  a≠b≠c        90°  90°  90°  Three 2-fold axes   Orthorhombic S
    Tetragonal      Tetragonal    a=b≠c        90°  90°  90°  One 4-fold axis     TiO₂
    Hexagonal       Hexagonal     a=b≠c        90°  90°  120° One 6-fold axis     Mg
    Trigonal        Trigonal      a=b=c        ≠90° =α   =α   One 3-fold axis     CaCO₃
    Cubic           Cubic         a=b=c        90°  90°  90°  Four 3-fold axes    NaCl
    ====================================================================================================
    

**Explanation** : This table allows you to understand the lattice parameter relationships of the seven crystal systems at a glance. The higher the symmetry, the more constraints exist (e.g., cubic has a=b=c), resulting in fewer parameters.

### Code Example 2: Generation and Visualization of a 2D Lattice (Square Lattice)

First, let's generate a two-dimensional square lattice and visualize the lattice points and unit cell.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: First, let's generate a two-dimensional square lattice and v
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 2D square lattice parameters
    a = 1.0  # Lattice constant (arbitrary units)
    n_cells = 5  # Number of unit cells to display (x, y direction)
    
    # Generate lattice points
    x_points = []
    y_points = []
    
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            x_points.append(i * a)
            y_points.append(j * a)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot lattice points
    ax.scatter(x_points, y_points, s=150, c='#f093fb',
               edgecolors='black', linewidths=2, zorder=3, label='Lattice Points')
    
    # Draw unit cell edges
    for i in range(n_cells):
        for j in range(n_cells):
            # Draw 4 edges of unit cell
            # Bottom edge
            ax.plot([i*a, (i+1)*a], [j*a, j*a], 'b-', linewidth=1.5, alpha=0.6)
            # Right edge
            ax.plot([(i+1)*a, (i+1)*a], [j*a, (j+1)*a], 'b-', linewidth=1.5, alpha=0.6)
            # Top edge
            ax.plot([(i+1)*a, i*a], [(j+1)*a, (j+1)*a], 'b-', linewidth=1.5, alpha=0.6)
            # Left edge
            ax.plot([i*a, i*a], [(j+1)*a, j*a], 'b-', linewidth=1.5, alpha=0.6)
    
    # Highlight the first unit cell
    ax.plot([0, a], [0, 0], 'r-', linewidth=3, label='Unit Cell')
    ax.plot([a, a], [0, a], 'r-', linewidth=3)
    ax.plot([a, 0], [a, a], 'r-', linewidth=3)
    ax.plot([0, 0], [a, 0], 'r-', linewidth=3)
    
    # Display lattice vectors as arrows
    ax.annotate('', xy=(a, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    ax.annotate('', xy=(0, a), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    ax.text(a/2, -0.3, r'$\vec{a}$', fontsize=14, color='green', fontweight='bold')
    ax.text(-0.3, a/2, r'$\vec{b}$', fontsize=14, color='green', fontweight='bold')
    
    # Axis settings
    ax.set_xlim(-0.5, n_cells * a + 0.5)
    ax.set_ylim(-0.5, n_cells * a + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('Visualization of a 2D Square Lattice', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Total number of lattice points:", len(x_points))
    print(f"Lattice constant: a = {a}")
    print(f"Unit cell area: {a**2:.2f}")
    print(f"Number of unit cells in displayed range: {n_cells * n_cells}")
    

**Explanation** : This plot visually demonstrates that lattice points are periodically arranged, and the entire lattice is constructed by repeating the unit cell (red box). The lattice vectors $\vec{a}$ and $\vec{b}$ define the basic structure of the lattice.

### Code Example 3: 3D Unit Cell Visualization (Cubic)

Let's visualize a three-dimensional cubic unit cell using Plotly.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - plotly>=5.14.0
    
    """
    Example: Let's visualize a three-dimensional cubic unit cell using Pl
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import plotly.graph_objects as go
    
    # Cubic unit cell parameters
    a = 1.0  # Lattice constant
    
    # Vertex coordinates of unit cell (8 points)
    vertices = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a, a, 0],
        [0, a, 0],
        [0, 0, a],
        [a, 0, a],
        [a, a, a],
        [0, a, a]
    ])
    
    # Edges of unit cell (12 edges)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Create plot
    fig = go.Figure()
    
    # Draw edges
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            line=dict(color='blue', width=6),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw vertices (lattice points)
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=10, color='#f093fb',
                    line=dict(color='black', width=2)),
        name='Lattice Points',
        hovertemplate='Coordinate: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))
    
    # Axis labels
    fig.update_layout(
        title='3D Visualization of a Cubic Unit Cell',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='cube',
            xaxis=dict(range=[-0.2, a+0.2]),
            yaxis=dict(range=[-0.2, a+0.2]),
            zaxis=dict(range=[-0.2, a+0.2])
        ),
        width=700,
        height=700
    )
    
    fig.show()
    
    print("Information about the cubic unit cell:")
    print(f"Lattice constants: a = b = c = {a} Å")
    print(f"Angles: α = β = γ = 90°")
    print(f"Unit cell volume: V = a³ = {a**3:.2f} Å³")
    print(f"Number of lattice points (vertices only): 8 points (equivalent to 1 lattice point: 8 × 1/8 = 1)")
    

**Explanation** : This 3D visualization intuitively shows that the cubic unit cell has a cube shape. The 8 lattice points at the vertices are shared with adjacent unit cells, so only 1 lattice point is actually contained in one unit cell (8 × 1/8 = 1).

### Code Example 4: Calculating Unit Cell Volume from Lattice Parameters

Let's create a function to calculate the unit cell volume from lattice parameters.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def calc_unit_cell_volume(a, b, c, alpha, beta, gamma):
        """
        Calculate unit cell volume from lattice parameters
    
        Parameters:
        -----------
        a, b, c : float
            Axis lengths (Å)
        alpha, beta, gamma : float
            Angles (degrees)
    
        Returns:
        --------
        volume : float
            Unit cell volume (Å³)
        """
        # Convert angles to radians
        alpha_rad = np.deg2rad(alpha)
        beta_rad = np.deg2rad(beta)
        gamma_rad = np.deg2rad(gamma)
    
        # Volume calculation formula
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
    
        volume = a * b * c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )
    
        return volume
    
    # Calculate volume for representative examples of each crystal system
    examples = [
        {'Name': 'NaCl', 'System': 'Cubic', 'a': 5.64, 'b': 5.64, 'c': 5.64,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'Si', 'System': 'Cubic', 'a': 5.43, 'b': 5.43, 'c': 5.43,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'TiO₂', 'System': 'Tetragonal', 'a': 4.59, 'b': 4.59, 'c': 2.96,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'Mg', 'System': 'Hexagonal', 'a': 3.21, 'b': 3.21, 'c': 5.21,
         'alpha': 90, 'beta': 90, 'gamma': 120},
        {'Name': 'CaCO₃', 'System': 'Trigonal', 'a': 4.99, 'b': 4.99, 'c': 4.99,
         'alpha': 101.9, 'beta': 101.9, 'gamma': 101.9},
    ]
    
    print("=" * 70)
    print("Calculating Unit Cell Volume from Lattice Parameters")
    print("=" * 70)
    
    for ex in examples:
        volume = calc_unit_cell_volume(
            ex['a'], ex['b'], ex['c'], ex['alpha'], ex['beta'], ex['gamma']
        )
    
        print(f"\n{ex['Name']} ({ex['System']})")
        print(f"  Lattice parameters: a={ex['a']:.2f}Å, b={ex['b']:.2f}Å, c={ex['c']:.2f}Å")
        print(f"  Angles: α={ex['alpha']:.1f}°, β={ex['beta']:.1f}°, γ={ex['gamma']:.1f}°")
        print(f"  Unit cell volume: V = {volume:.2f} Å³")
    
    print("\n" + "=" * 70)
    
    # Comparison with simplified formula for cubic system
    print("\nFor cubic (NaCl), simplified formula V = a³ can also be used:")
    a_nacl = 5.64
    v_simple = a_nacl ** 3
    v_general = calc_unit_cell_volume(a_nacl, a_nacl, a_nacl, 90, 90, 90)
    print(f"  Simplified formula: V = {a_nacl}³ = {v_simple:.2f} Å³")
    print(f"  General formula: V = {v_general:.2f} Å³")
    print(f"  Difference: {abs(v_simple - v_general):.6f} Å³ (matches within error margin)")
    

**Output Example** :
    
    
    ======================================================================
    Calculating Unit Cell Volume from Lattice Parameters
    ======================================================================
    
    NaCl (Cubic)
      Lattice parameters: a=5.64Å, b=5.64Å, c=5.64Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      Unit cell volume: V = 179.41 Å³
    
    Si (Cubic)
      Lattice parameters: a=5.43Å, b=5.43Å, c=5.43Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      Unit cell volume: V = 160.10 Å³
    
    TiO₂ (Tetragonal)
      Lattice parameters: a=4.59Å, b=4.59Å, c=2.96Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      Unit cell volume: V = 62.35 Å³
    
    Mg (Hexagonal)
      Lattice parameters: a=3.21Å, b=3.21Å, c=5.21Å
      Angles: α=90.0°, β=90.0°, γ=120.0°
      Unit cell volume: V = 46.49 Å³
    

**Explanation** : This function allows you to calculate the unit cell volume for any crystal system using the general formula. While simplified formulas ($V = a^3$ or $V = a^2c$) can be used for cubic and tetragonal systems, the general formula is applicable to all crystal systems.

### Code Example 5: Visualizing Unit Cells of Different Crystal Systems (Cubic vs Tetragonal vs Orthorhombic)

Let's visualize unit cells of three different crystal systems side by side to understand their differences.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - plotly>=5.14.0
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_unit_cell_vertices(a, b, c):
        """Generate vertex coordinates of unit cell"""
        return np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]
        ])
    
    def add_unit_cell_to_fig(fig, a, b, c, row, col, title):
        """Add unit cell to subplot"""
        vertices = create_unit_cell_vertices(a, b, c)
    
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
    
        # Draw edges
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                mode='lines', line=dict(color='blue', width=4),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=col)
    
        # Draw vertices
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            mode='markers', marker=dict(size=6, color='#f093fb'),
            showlegend=False, hoverinfo='skip'
        ), row=row, col=col)
    
        # Subplot title and axis settings
        scene_name = f'scene{(row-1)*3 + col}' if row > 1 or col > 1 else 'scene'
        fig.update_layout({
            scene_name: dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                aspectmode='cube',
                xaxis=dict(range=[0, max(a, b, c)]),
                yaxis=dict(range=[0, max(a, b, c)]),
                zaxis=dict(range=[0, max(a, b, c)])
            )
        })
    
    # Create subplots (1 row, 3 columns)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Cubic (a=b=c)', 'Tetragonal (a=b≠c)', 'Orthorhombic (a≠b≠c)')
    )
    
    # Cubic (a=b=c)
    add_unit_cell_to_fig(fig, a=2.0, b=2.0, c=2.0, row=1, col=1, title='Cubic')
    
    # Tetragonal (a=b≠c)
    add_unit_cell_to_fig(fig, a=2.0, b=2.0, c=3.0, row=1, col=2, title='Tetragonal')
    
    # Orthorhombic (a≠b≠c)
    add_unit_cell_to_fig(fig, a=2.0, b=2.5, c=3.5, row=1, col=3, title='Orthorhombic')
    
    # Adjust layout
    fig.update_layout(
        title_text="Comparison of Unit Cells for Three Crystal Systems",
        height=500,
        width=1200,
        showlegend=False
    )
    
    fig.show()
    
    # Output information for each crystal system
    print("Comparison of Three Crystal Systems:")
    print("-" * 60)
    print("Cubic:       a = b = c = 2.0 Å, α = β = γ = 90°")
    print("             Volume V = a³ = 8.0 Å³")
    print("-" * 60)
    print("Tetragonal:  a = b = 2.0 Å, c = 3.0 Å, α = β = γ = 90°")
    print("             Volume V = a²c = 12.0 Å³")
    print("-" * 60)
    print("Orthorhombic: a = 2.0 Å, b = 2.5 Å, c = 3.5 Å, α = β = γ = 90°")
    print("              Volume V = abc = 17.5 Å³")
    print("-" * 60)
    

**Explanation** : This side-by-side comparison visually demonstrates that symmetry decreases in the order cubic → tetragonal → orthorhombic, with shape constraints becoming less restrictive. Cubic has the highest symmetry, while orthorhombic has three different axis lengths.

### Code Example 6: Program to Generate Lattice Points (3D Lattice)

Let's generate lattice points in three-dimensional space and visualize a supercell (multiple unit cells).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - plotly>=5.14.0
    
    import numpy as np
    import plotly.graph_objects as go
    
    def generate_3d_lattice_points(a, b, c, nx, ny, nz):
        """
        Generate 3D lattice points
    
        Parameters:
        -----------
        a, b, c : float
            Lattice constants (Å)
        nx, ny, nz : int
            Number of unit cells in each direction
    
        Returns:
        --------
        points : ndarray
            Array of lattice point coordinates (N, 3)
        """
        points = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i * a
                    y = j * b
                    z = k * c
                    points.append([x, y, z])
        return np.array(points)
    
    # Generate cubic supercell (3×3×3)
    a = 1.0  # Lattice constant
    nx, ny, nz = 3, 3, 3  # Number of unit cells in each direction
    
    lattice_points = generate_3d_lattice_points(a, a, a, nx, ny, nz)
    
    # Create plot
    fig = go.Figure()
    
    # Draw lattice points
    fig.add_trace(go.Scatter3d(
        x=lattice_points[:, 0],
        y=lattice_points[:, 1],
        z=lattice_points[:, 2],
        mode='markers',
        marker=dict(size=5, color='#f093fb',
                    line=dict(color='black', width=1)),
        name='Lattice Points',
        hovertemplate='Coordinate: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>'
    ))
    
    # Draw edges of unit cell (first unit cell only)
    unit_cell_edges = [
        [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
        [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
        [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
        [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
        [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
        [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
    ]
    
    for edge in unit_cell_edges:
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Layout settings
    fig.update_layout(
        title=f'Generation of 3D Lattice Points ({nx}×{ny}×{nz} Supercell)',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='cube'
        ),
        width=700,
        height=700
    )
    
    fig.show()
    
    # Output statistics
    print(f"Supercell Information:")
    print(f"  Lattice constants: a = b = c = {a} Å")
    print(f"  Number of unit cells: {nx} × {ny} × {nz} = {nx*ny*nz} cells")
    print(f"  Total lattice points: {len(lattice_points)} points")
    print(f"  Lattice point density: {len(lattice_points)/(nx*a*ny*a*nz*a):.2f} points/Å³")
    print(f"  Total volume: {nx*a} × {ny*a} × {nz*a} = {nx*ny*nz*a**3:.2f} Å³")
    

**Explanation** : This program allows you to generate supercells of arbitrary size (larger structures combining multiple unit cells). You can visually understand how the periodic arrangement of lattice points repeats in three-dimensional space.

### Code Example 7: Supercell Generation (2×2×2 Expansion)

Let's generate a supercell by expanding the unit cell by a factor of 2×2×2.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def expand_unit_cell_to_supercell(unit_cell_atoms, lattice_vectors, n1, n2, n3):
        """
        Expand unit cell to supercell
    
        Parameters:
        -----------
        unit_cell_atoms : ndarray
            Atomic coordinates in unit cell (fractional coordinates) (N, 3)
        lattice_vectors : ndarray
            Lattice vectors (3, 3) [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
        n1, n2, n3 : int
            Expansion factor in each direction
    
        Returns:
        --------
        supercell_atoms : ndarray
            Atomic coordinates in supercell (Cartesian coordinates) (M, 3)
        """
        supercell_atoms = []
    
        for atom_frac in unit_cell_atoms:
            for i in range(n1):
                for j in range(n2):
                    for k in range(n3):
                        # Add translation to fractional coordinates
                        frac_coord = atom_frac + np.array([i/n1, j/n2, k/n3])
    
                        # Convert to Cartesian coordinates
                        cart_coord = np.dot(frac_coord, lattice_vectors)
                        supercell_atoms.append(cart_coord)
    
        return np.array(supercell_atoms)
    
    # Example: Simple cubic lattice unit cell (lattice points only)
    unit_cell_atoms = np.array([
        [0.0, 0.0, 0.0]  # Atom only at origin (fractional coordinates)
    ])
    
    # Lattice vectors (cubic, a=2.0 Å)
    a = 2.0
    lattice_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])
    
    # Generate 2×2×2 supercell
    supercell = expand_unit_cell_to_supercell(unit_cell_atoms, lattice_vectors, 2, 2, 2)
    
    print("2×2×2 Supercell Generation Results:")
    print("=" * 60)
    print(f"Number of atoms in unit cell: {len(unit_cell_atoms)}")
    print(f"Number of atoms in supercell: {len(supercell)} (= {len(unit_cell_atoms)} × 2³)")
    print("=" * 60)
    print("\nAtomic Coordinates in Supercell (Cartesian Coordinates, Å):")
    for i, coord in enumerate(supercell, 1):
        print(f"  Atom{i:2d}: ({coord[0]:5.2f}, {coord[1]:5.2f}, {coord[2]:5.2f})")
    
    # Calculate supercell size
    supercell_size = 2 * a
    supercell_volume = supercell_size ** 3
    print(f"\nSupercell Size: {supercell_size} × {supercell_size} × {supercell_size} Å")
    print(f"Supercell Volume: {supercell_volume:.2f} Å³")
    print(f"Atomic Density: {len(supercell)/supercell_volume:.4f} atoms/Å³")
    

**Output Example** :
    
    
    2×2×2 Supercell Generation Results:
    ============================================================
    Number of atoms in unit cell: 1
    Number of atoms in supercell: 8 (= 1 × 2³)
    ============================================================
    
    Atomic Coordinates in Supercell (Cartesian Coordinates, Å):
      Atom 1: ( 0.00,  0.00,  0.00)
      Atom 2: ( 0.00,  0.00,  2.00)
      Atom 3: ( 0.00,  2.00,  0.00)
      Atom 4: ( 0.00,  2.00,  2.00)
      Atom 5: ( 2.00,  0.00,  0.00)
      Atom 6: ( 2.00,  0.00,  2.00)
      Atom 7: ( 2.00,  2.00,  0.00)
      Atom 8: ( 2.00,  2.00,  2.00)
    
    Supercell Size: 4.0 × 4.0 × 4.0 Å
    Supercell Volume: 64.00 Å³
    Atomic Density: 0.1250 atoms/Å³
    

**Explanation** : Supercells are frequently used in computational chemistry and simulations. By repeating the unit cell, you can model large-scale structures such as surfaces, interfaces, and defects.

### Code Example 8: Crystal System Determination Program (Determination from Lattice Parameters)

Let's create a program that determines which crystal system a material belongs to based on input lattice parameters.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def determine_crystal_system(a, b, c, alpha, beta, gamma, tolerance=0.01):
        """
        Determine crystal system from lattice parameters
    
        Parameters:
        -----------
        a, b, c : float
            Axis lengths (Å)
        alpha, beta, gamma : float
            Angles (degrees)
        tolerance : float
            Tolerance for determination (relative error)
    
        Returns:
        --------
        system : str
            Name of crystal system
        """
        # Angle tolerance (degrees)
        angle_tol = 1.0
    
        # Length comparison (relative error)
        def is_equal(x, y):
            return abs(x - y) / max(x, y) < tolerance
    
        # Angle comparison
        def angle_is(angle, target):
            return abs(angle - target) < angle_tol
    
        # Cubic: a=b=c, α=β=γ=90°
        if is_equal(a, b) and is_equal(b, c) and \
           angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return 'Cubic'
    
        # Tetragonal: a=b≠c, α=β=γ=90°
        elif is_equal(a, b) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return 'Tetragonal'
    
        # Orthorhombic: a≠b≠c, α=β=γ=90°
        elif not is_equal(a, b) and not is_equal(b, c) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return 'Orthorhombic'
    
        # Hexagonal: a=b≠c, α=β=90°, γ=120°
        elif is_equal(a, b) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 120):
            return 'Hexagonal'
    
        # Trigonal: a=b=c, α=β=γ≠90°
        elif is_equal(a, b) and is_equal(b, c) and \
             is_equal(alpha, beta) and is_equal(beta, gamma) and not angle_is(alpha, 90):
            return 'Trigonal / Rhombohedral'
    
        # Monoclinic: a≠b≠c, α=γ=90°≠β
        elif not is_equal(a, b) and not is_equal(b, c) and not is_equal(a, c) and \
             angle_is(alpha, 90) and not angle_is(beta, 90) and angle_is(gamma, 90):
            return 'Monoclinic'
    
        # Triclinic: a≠b≠c, α≠β≠γ≠90° (most general)
        else:
            return 'Triclinic'
    
    # Test cases
    test_cases = [
        {'Name': 'NaCl', 'a': 5.64, 'b': 5.64, 'c': 5.64, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'Si', 'a': 5.43, 'b': 5.43, 'c': 5.43, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'TiO₂', 'a': 4.59, 'b': 4.59, 'c': 2.96, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'Name': 'Mg', 'a': 3.21, 'b': 3.21, 'c': 5.21, 'alpha': 90, 'beta': 90, 'gamma': 120},
        {'Name': 'CaCO₃', 'a': 4.99, 'b': 4.99, 'c': 4.99, 'alpha': 101.9, 'beta': 101.9, 'gamma': 101.9},
        {'Name': 'Gypsum', 'a': 5.68, 'b': 15.18, 'c': 6.29, 'alpha': 90, 'beta': 113.8, 'gamma': 90},
        {'Name': 'Sulfur', 'a': 10.47, 'b': 12.87, 'c': 24.49, 'alpha': 90, 'beta': 90, 'gamma': 90},
    ]
    
    print("=" * 80)
    print("Crystal System Determination Program")
    print("=" * 80)
    
    for case in test_cases:
        system = determine_crystal_system(
            case['a'], case['b'], case['c'],
            case['alpha'], case['beta'], case['gamma']
        )
    
        print(f"\n{case['Name']:10s}")
        print(f"  Lattice parameters: a={case['a']:.2f}Å, b={case['b']:.2f}Å, c={case['c']:.2f}Å")
        print(f"  Angles: α={case['alpha']:.1f}°, β={case['beta']:.1f}°, γ={case['gamma']:.1f}°")
        print(f"  → Determination result: {system}")
    
    print("\n" + "=" * 80)
    

**Output Example** :
    
    
    ================================================================================
    Crystal System Determination Program
    ================================================================================
    
    NaCl
      Lattice parameters: a=5.64Å, b=5.64Å, c=5.64Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      → Determination result: Cubic
    
    Si
      Lattice parameters: a=5.43Å, b=5.43Å, c=5.43Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      → Determination result: Cubic
    
    TiO₂
      Lattice parameters: a=4.59Å, b=4.59Å, c=2.96Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      → Determination result: Tetragonal
    
    Mg
      Lattice parameters: a=3.21Å, b=3.21Å, c=5.21Å
      Angles: α=90.0°, β=90.0°, γ=120.0°
      → Determination result: Hexagonal
    
    CaCO₃
      Lattice parameters: a=4.99Å, b=4.99Å, c=4.99Å
      Angles: α=101.9°, β=101.9°, γ=101.9°
      → Determination result: Trigonal / Rhombohedral
    
    Gypsum
      Lattice parameters: a=5.68Å, b=15.18Å, c=6.29Å
      Angles: α=90.0°, β=113.8°, γ=90.0°
      → Determination result: Monoclinic
    
    Sulfur
      Lattice parameters: a=10.47Å, b=12.87Å, c=24.49Å
      Angles: α=90.0°, β=90.0°, γ=90.0°
      → Determination result: Orthorhombic
    
    ================================================================================
    

**Explanation** : This program allows automatic determination of the crystal system from lattice parameters. By inputting lattice parameters obtained from experimental data, classification of crystal structures is possible. By adjusting the tolerance, it can also accommodate measurement errors.

* * *

## 1.5 Exercises

Let's work on the following exercises to confirm what you've learned.

**Exercise 1: Calculating Cubic Unit Cell Volume**

**Problem** : The lattice constant of iron (α-Fe) is a = 2.87 Å. Calculate the unit cell volume.

**Solution Example** :
    
    
    # Calculate cubic unit cell volume
    a_Fe = 2.87  # Å
    V_Fe = a_Fe ** 3
    print(f"Unit cell volume of iron (α-Fe): V = {V_Fe:.2f} Å³")
    # Output: Unit cell volume of iron (α-Fe): V = 23.64 Å³
    

**Exercise 2: Determining Crystal System from Lattice Parameters**

**Problem** : Determine the crystal system of a material with the following lattice parameters.

  * a = 4.5 Å, b = 4.5 Å, c = 6.0 Å
  * α = 90°, β = 90°, γ = 90°

**Solution Example** :
    
    
    # Use the function from Code Example 8
    system = determine_crystal_system(4.5, 4.5, 6.0, 90, 90, 90)
    print(f"Determination result: {system}")
    # Output: Determination result: Tetragonal
    

**Reason** : Since a = b ≠ c and α = β = γ = 90°, it is a tetragonal system.

**Exercise 3: Calculating 2D Lattice Point Coordinates**

**Problem** : For a two-dimensional square lattice (a = 3.0 Å), list all lattice point coordinates in a 3×3 range.

**Solution Example** :
    
    
    a = 3.0
    points = []
    for i in range(4):  # 0, 1, 2, 3
        for j in range(4):
            points.append((i * a, j * a))
    
    print("Lattice point coordinates:")
    for i, (x, y) in enumerate(points, 1):
        print(f"  Point{i:2d}: ({x:.1f}, {y:.1f})")
    print(f"\nTotal lattice points: {len(points)}")
    # Total lattice points: 16
    

**Exercise 4: Calculating Supercell Size**

**Problem** : For a cubic unit cell (a = 5.0 Å) expanded by a factor of 4×4×4, calculate the supercell volume and the total number of atoms in the supercell if the unit cell contains 1 atom.

**Solution Example** :
    
    
    a = 5.0  # Å
    n = 4    # Expansion factor
    
    # Supercell size
    supercell_size = n * a
    supercell_volume = supercell_size ** 3
    
    # Total atoms
    atoms_per_unit_cell = 1
    total_atoms = atoms_per_unit_cell * n ** 3
    
    print(f"Supercell size: {supercell_size} Å")
    print(f"Supercell volume: {supercell_volume} Å³")
    print(f"Total atoms: {total_atoms} atoms")
    # Output:
    # Supercell size: 20.0 Å
    # Supercell volume: 8000.0 Å³
    # Total atoms: 64 atoms
    

**Exercise 5: Comparing Symmetry of Crystal Systems**

**Problem** : Compare the symmetries of cubic, tetragonal, and orthorhombic systems, and state the number of rotational symmetry axes for each.

**Answer** :

  * **Cubic** : 
    * 4-fold axes: 3 (x, y, z axes)
    * 3-fold axes: 4 (body diagonals)
    * 2-fold axes: 6 (face diagonals)
    * Highest symmetry
  * **Tetragonal** : 
    * 4-fold axis: 1 (c-axis)
    * 2-fold axes: 4 (a, b axes and diagonals)
    * Moderate symmetry
  * **Orthorhombic** : 
    * 2-fold axes: 3 (a, b, c axes)
    * Lowest symmetry (among the three)

**Conclusion** : Symmetry is highest in the order: Cubic > Tetragonal > Orthorhombic.

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Basic Concepts of Crystallography**
     * Crystals are materials in which atoms are regularly arranged (long-range order, periodicity)
     * Amorphous materials are those in which atoms are irregularly arranged (short-range order only)
     * Periodicity and symmetry are fundamental to crystal structures
  2. **Lattice Points and Unit Cells**
     * Lattice points are reference points of periodically repeating structures
     * The unit cell is the minimum repeating unit of the crystal structure
     * Lattice parameters: six parameters consisting of a, b, c (axis lengths) and α, β, γ (angles)
  3. **Seven Crystal Systems**
     * Triclinic, Monoclinic, Orthorhombic, Tetragonal, Hexagonal, Trigonal, Cubic
     * Classified by relationships among lattice parameters and symmetry
     * Cubic has the highest symmetry, triclinic the lowest
  4. **Visualization with Python**
     * Generation and visualization of lattice points
     * 3D display of unit cells
     * Construction of supercells
     * Automatic determination of crystal systems from lattice parameters

### Important Points

  * Crystal structures are characterized by **periodicity and symmetry**
  * By repeating the unit cell, infinitely extending crystal structures can be constructed
  * Lattice parameters are fundamental parameters that quantitatively describe crystal structures
  * In crystals, **only 1-, 2-, 3-, 4-, and 6-fold symmetries** are allowed (5-fold symmetry is not possible)
  * Higher symmetry means more constraints among lattice parameters (lower degrees of freedom)

### Next Chapter

In Chapter 2, we will learn about **Bravais Lattices and Space Groups** :

  * The 14 Bravais lattices (primitive, body-centered, face-centered, base-centered)
  * Space groups and symmetry operations (rotation, inversion, reflection, screw, glide)
  * Complete description methods for crystal structures
  * Visualization of Bravais lattices with Python
  * Representative crystal structures (fcc, bcc, hcp, diamond structure)
