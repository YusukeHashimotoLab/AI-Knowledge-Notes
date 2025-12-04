---
title: "Chapter 3: Fundamentals of Crystal Structure"
chapter_title: "Chapter 3: Fundamentals of Crystal Structure"
subtitle: Relationship Between Regular Atomic Arrangement and Material Properties
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 7
version: 1.0
created_at: 2025-10-25
---

Material properties greatly depend on how atoms are arranged. In this chapter, we will learn the fundamental concepts of crystal structure, major crystal systems, Miller indices, and visualize crystal structures using Python. 

## Learning Objectives

By reading this chapter, you will be able to:

  *  Understand the difference between crystalline and amorphous materials and distinguish them from X-ray diffraction patterns
  *  Understand the concepts of unit cell and lattice parameters
  *  Explain the characteristics and representative materials of major crystal structures (FCC, BCC, HCP)
  *  Represent crystal planes and directions using Miller indices
  *  Calculate packing fraction and coordination number
  *  Visualize 3D crystal structures using Python

* * *

## 3.1 Crystalline and Amorphous Materials

### What is Crystal Structure?

**Crystals** are solids in which atoms are arranged regularly and periodically in three-dimensional space. On the other hand, **amorphous** materials are solids without long-range order in atomic arrangement.

Property | Crystalline | Amorphous  
---|---|---  
**Atomic Arrangement** | Long-range order (periodic) | Short-range order only (random)  
**Melting Point** | Sharp melting point | Glass transition temperature (gradual softening)  
**X-ray Diffraction** | Sharp peaks (Bragg reflection) | Broad halo  
**Anisotropy** | Properties vary with direction | Isotropic (same in all directions)  
**Examples** | Metals, Si, NaCl, Diamond | Glass, Polymers, Amorphous Si  
  
### Evaluation of Crystallinity by X-ray Diffraction

**X-ray Diffraction (XRD)** is the most important technique for analyzing crystal structure. When X-rays are incident on a crystal, **Bragg reflection** occurs at specific angles due to the atomic arrangement.

**Bragg's Law** :

$$n\lambda = 2d\sin\theta$$

where,

  * $n$: Reflection order (integer)
  * $\lambda$: X-ray wavelength
  * $d$: Interplanar spacing
  * $\theta$: Incident angle (Bragg angle)

Crystalline materials show strong diffraction peaks at specific angles ($2\theta$), while amorphous materials show broad halo patterns.

### Code Example 1: Simulation of X-ray Diffraction Patterns for Crystalline and Amorphous Materials

Visualize the difference between X-ray diffraction patterns of crystalline and amorphous materials.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # X-ray diffraction pattern simulation
    def simulate_crystalline_xrd():
        """
        Simulate XRD pattern for crystalline material
        Sharp Bragg peaks appear at specific angles
        """
        # 2¸ range (degrees)
        two_theta = np.linspace(10, 80, 1000)
        intensity = np.zeros_like(two_theta)
    
        # Bragg reflection peaks from major crystal planes
        # Diffraction from (111), (200), (220), (311) planes
        peaks = [28.4, 33.0, 47.5, 56.1]  # 2¸ positions (e.g., FCC structure)
        peak_intensities = [100, 50, 80, 40]  # Relative intensities
        peak_width = 0.3  # Peak width (degrees)
    
        # Add Gaussian peaks
        for peak_pos, peak_int in zip(peaks, peak_intensities):
            intensity += peak_int * np.exp(-((two_theta - peak_pos) / peak_width)**2)
    
        # Background noise
        intensity += np.random.normal(2, 0.5, len(two_theta))
    
        return two_theta, intensity
    
    
    def simulate_amorphous_xrd():
        """
        Simulate XRD pattern for amorphous material
        Broad halo pattern
        """
        two_theta = np.linspace(10, 80, 1000)
    
        # Broad halo (due to short-range order)
        halo_center = 25  # Halo center position
        halo_width = 15   # Halo width
        intensity = 30 * np.exp(-((two_theta - halo_center) / halo_width)**2)
    
        # Additional broad peak
        intensity += 15 * np.exp(-((two_theta - 45) / 20)**2)
    
        # Noise
        intensity += np.random.normal(2, 0.5, len(two_theta))
    
        return two_theta, intensity
    
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # XRD pattern of crystalline material
    two_theta_cryst, intensity_cryst = simulate_crystalline_xrd()
    ax1.plot(two_theta_cryst, intensity_cryst, linewidth=1.5, color='#1f77b4')
    ax1.set_xlabel('2¸ (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax1.set_title('XRD Pattern of Crystalline Material\n(Sharp Bragg Peaks)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 110)
    
    # Annotate peak positions
    peaks_labels = ['(111)', '(200)', '(220)', '(311)']
    peaks_pos = [28.4, 33.0, 47.5, 56.1]
    for label, pos in zip(peaks_labels, peaks_pos):
        ax1.annotate(label, xy=(pos, 105), ha='center', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # XRD pattern of amorphous material
    two_theta_amor, intensity_amor = simulate_amorphous_xrd()
    ax2.plot(two_theta_amor, intensity_amor, linewidth=1.5, color='#ff7f0e')
    ax2.set_xlabel('2¸ (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax2.set_title('XRD Pattern of Amorphous Material\n(Broad Halo)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 35)
    
    plt.tight_layout()
    plt.show()
    
    print("Interpretation of XRD Patterns:")
    print("\n[Crystalline Materials]")
    print("- Sharp peaks appear at clear angles")
    print("- Lattice parameters and crystal structure determined from peak positions")
    print("- Information about atomic arrangement obtained from peak intensities")
    print("- Examples: Metals, Ceramics, Silicon single crystals")
    
    print("\n[Amorphous Materials]")
    print("- Broad halo pattern")
    print("- Short-range order exists, but no long-range order")
    print("- Average interatomic distance estimated from halo position")
    print("- Examples: Glass, Amorphous silicon, Some polymers")
    

**Explanation** : X-ray diffraction patterns clearly indicate the presence or absence of crystallinity. Crystalline materials show sharp peaks at specific angles, which allows identification of the crystal structure. Amorphous materials show broad patterns, indicating the absence of long-range order.

* * *

## 3.2 Unit Cell and Lattice Parameters

### Unit Cell

The **unit cell** is the smallest repeating unit of a crystal structure. The entire crystal can be constructed by arranging unit cells in three-dimensional space.

**Lattice Parameters** :

  * $a, b, c$: Edge lengths of the unit cell (in Å)
  * $\alpha, \beta, \gamma$: Angles between edges (in degrees or radians)

### Seven Crystal Systems

Based on the shape of the unit cell, crystals are classified into seven crystal systems:

Crystal System | Lattice Parameter Relationship | Angle Relationship | Examples  
---|---|---|---  
**Cubic** | a = b = c | ± = ² = ³ = 90° | NaCl, Cu, Fe, Si  
**Tetragonal** | a = b ` c | ± = ² = ³ = 90° | TiO‚, SnO‚  
**Orthorhombic** | a ` b ` c | ± = ² = ³ = 90° | ±-S, BaSO„  
**Hexagonal** | a = b ` c | ± = ² = 90°, ³ = 120° | Mg, Zn, Graphite  
**Rhombohedral** | a = b = c | ± = ² = ³ ` 90° | Quartz, CaCOƒ  
**Monoclinic** | a ` b ` c | ± = ³ = 90° ` ² | ²-S, CaSO„·2H‚O  
**Triclinic** | a ` b ` c | ± ` ² ` ³ ` 90° | CuSO„·5H‚O  
  
The most important in materials science is the **cubic system**. Many metals belong to the cubic system.

### Miller Indices

**Miller indices** are a notation method for representing planes and directions in crystals.

**Notation for crystal planes** : $(hkl)$

**Notation for crystal directions** : $[uvw]$

**Method for determining Miller indices** (for crystal planes):

  1. Find the coordinates where the plane intersects the a, b, c axes (intercepts)
  2. Take the reciprocals of each intercept
  3. Convert to integer ratios
  4. Express as $(hkl)$

**Examples** :

  * $(100)$ plane: Plane perpendicular to the a-axis
  * $(110)$ plane: Plane inclined at 45° to both a and b axes
  * $(111)$ plane: Plane equally inclined to the a, b, and c axes

### Calculation of Interplanar Spacing

For cubic systems, the interplanar spacing $d_{hkl}$ of the $(hkl)$ plane is calculated by the following equation:

$$d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}$$

where $a$ is the lattice parameter.

### Code Example 2: Calculator for Unit Cell Volume and Interplanar Spacing

Calculate the unit cell volume and interplanar spacing for planes specified by Miller indices from lattice parameters.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    class CrystalCalculator:
        """
        Class for calculating various parameters of crystal structure
        """
    
        def __init__(self, a, b=None, c=None, alpha=90, beta=90, gamma=90):
            """
            Set lattice parameters
    
            Parameters:
            a, b, c: Lattice parameters (Å)
            alpha, beta, gamma: Angles (degrees)
            """
            self.a = a
            self.b = b if b is not None else a
            self.c = c if c is not None else a
            self.alpha = np.radians(alpha)
            self.beta = np.radians(beta)
            self.gamma = np.radians(gamma)
    
        def unit_cell_volume(self):
            """
            Calculate unit cell volume (Å³)
            """
            # General formula (applicable to all crystal systems)
            cos_alpha = np.cos(self.alpha)
            cos_beta = np.cos(self.beta)
            cos_gamma = np.cos(self.gamma)
    
            volume = self.a * self.b * self.c * np.sqrt(
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                + 2*cos_alpha*cos_beta*cos_gamma
            )
            return volume
    
        def d_spacing_cubic(self, h, k, l):
            """
            Calculate interplanar spacing for cubic systems (simplified version)
    
            Parameters:
            h, k, l: Miller indices
    
            Returns:
            d: Interplanar spacing (Å)
            """
            if h == 0 and k == 0 and l == 0:
                raise ValueError("Miller indices cannot all be zero")
    
            d = self.a / np.sqrt(h**2 + k**2 + l**2)
            return d
    
        def print_crystal_info(self, crystal_name):
            """
            Display crystal information in an organized format
            """
            print(f"\n{'='*60}")
            print(f"[Crystal Parameters of {crystal_name}]")
            print(f"{'='*60}")
            print(f"Lattice parameter a = {self.a:.4f} Å")
            if self.b != self.a or self.c != self.a:
                print(f"Lattice parameter b = {self.b:.4f} Å")
                print(f"Lattice parameter c = {self.c:.4f} Å")
    
            volume = self.unit_cell_volume()
            print(f"Unit cell volume V = {volume:.4f} Å³")
    
            # Calculate interplanar spacing for major crystal planes (cubic system)
            if self.a == self.b == self.c and \
               self.alpha == self.beta == self.gamma == np.radians(90):
                print(f"\nInterplanar spacing for major crystal planes:")
                planes = [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0)]
                for h, k, l in planes:
                    d = self.d_spacing_cubic(h, k, l)
                    print(f"  ({h}{k}{l}) plane: d = {d:.4f} Å")
    
    
    # Crystal parameters for representative materials
    print("Crystal Structure Calculations for Representative Materials")
    
    # Copper (FCC)
    cu = CrystalCalculator(a=3.615)  # Å
    cu.print_crystal_info("Copper (Cu, FCC)")
    
    # Iron (BCC)
    fe = CrystalCalculator(a=2.866)  # Å
    fe.print_crystal_info("Iron (Fe, BCC)")
    
    # Silicon (Diamond structure)
    si = CrystalCalculator(a=5.431)  # Å
    si.print_crystal_info("Silicon (Si)")
    
    # Alumina (Hexagonal system)
    al2o3 = CrystalCalculator(a=4.759, c=12.991, gamma=120)  # Å
    print(f"\n{'='*60}")
    print(f"[Crystal Parameters of Alumina (Al‚Oƒ, HCP)]")
    print(f"{'='*60}")
    print(f"Lattice parameter a = {al2o3.a:.4f} Å")
    print(f"Lattice parameter c = {al2o3.c:.4f} Å")
    print(f"c/a ratio = {al2o3.c/al2o3.a:.4f}")
    volume = al2o3.unit_cell_volume()
    print(f"Unit cell volume V = {volume:.4f} Å³")
    
    print("\n" + "="*60)
    print("Applications of interplanar spacing:")
    print("- Prediction of X-ray diffraction peak positions")
    print("- Calculation of atomic plane density")
    print("- Analysis of slip systems (plastic deformation mechanisms)")
    

**Sample Output** :
    
    
    Crystal Structure Calculations for Representative Materials
    ============================================================
    [Crystal Parameters of Copper (Cu, FCC)]
    ============================================================
    Lattice parameter a = 3.6150 Å
    Unit cell volume V = 47.2418 Å³
    
    Interplanar spacing for major crystal planes:
      (100) plane: d = 3.6150 Å
      (110) plane: d = 2.5557 Å
      (111) plane: d = 2.0871 Å
      (200) plane: d = 1.8075 Å
      (220) plane: d = 1.2779 Å

**Explanation** : Unit cell volume and interplanar spacing can be calculated from lattice parameters. Interplanar spacing is used for predicting X-ray diffraction peak positions and calculating atomic plane density.

* * *

## 3.3 Major Crystal Structures

Most metallic materials have one of the following three crystal structures:

### 1\. Face-Centered Cubic (FCC)

**Characteristics** :

  * Atoms located at the center of each face of the cube
  * Number of atoms per unit cell: 4 (8 corners × 1/8 + 6 faces × 1/2 = 4)
  * Coordination number: 12 (number of nearest atoms)
  * Atomic Packing Fraction (APF): 74%
  * Close-packed structure

**Representative Materials** :

  * Copper (Cu): a = 3.615 Å
  * Aluminum (Al): a = 4.049 Å
  * Gold (Au): a = 4.078 Å
  * Silver (Ag): a = 4.086 Å
  * Nickel (Ni): a = 3.524 Å

**Properties** : Excellent ductility (many slip systems), relatively soft

### 2\. Body-Centered Cubic (BCC)

**Characteristics** :

  * Atom located at the center of the cube
  * Number of atoms per unit cell: 2 (8 corners × 1/8 + 1 body center = 2)
  * Coordination number: 8
  * Atomic Packing Fraction (APF): 68%

**Representative Materials** :

  * Iron (Fe, ±-iron): a = 2.866 Å
  * Chromium (Cr): a = 2.885 Å
  * Tungsten (W): a = 3.165 Å
  * Molybdenum (Mo): a = 3.147 Å
  * Vanadium (V): a = 3.024 Å

**Properties** : High strength, prone to brittle fracture at low temperatures

### 3\. Hexagonal Close-Packed (HCP)

**Characteristics** :

  * Close-packed structure of hexagonal system
  * Number of atoms per unit cell: 6
  * Coordination number: 12
  * Atomic Packing Fraction (APF): 74% (same as FCC)
  * Ideal c/a ratio: 1.633

**Representative Materials** :

  * Magnesium (Mg): a = 3.209 Å, c/a = 1.624
  * Zinc (Zn): a = 2.665 Å, c/a = 1.856
  * Titanium (Ti): a = 2.951 Å, c/a = 1.588
  * Cobalt (Co): a = 2.507 Å, c/a = 1.623

**Properties** : Few slip systems, low ductility (strong anisotropy)

### Comparison Table of Crystal Structures

Item | FCC | BCC | HCP  
---|---|---|---  
**Atoms per Unit Cell** | 4 | 2 | 6  
**Coordination Number** | 12 | 8 | 12  
**Packing Fraction (APF)** | 74% | 68% | 74%  
**Number of Slip Systems** | 12 | 48 (limited at low temp) | 3 (few)  
**Ductility** | High | Medium to High | Low  
**Representative Metals** | Cu, Al, Au, Ag | Fe, Cr, W, Mo | Mg, Zn, Ti, Co  
  
### Code Example 3: Calculation of Atomic Packing Fraction (APF)

Calculate and compare the atomic packing fraction for FCC, BCC, and HCP structures.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def calculate_apf_fcc(a, r=None):
        """
        Calculate atomic packing fraction (APF) for FCC structure
    
        Parameters:
        a: Lattice parameter (Å)
        r: Atomic radius (Å). If None, calculated from lattice parameter
    
        Returns:
        apf: Packing fraction
        """
        # In FCC structure, atoms touch along face diagonal
        # Face diagonal = 4r = a2, so r = a2/4
        if r is None:
            r = a * np.sqrt(2) / 4
    
        # Number of atoms per unit cell
        n_atoms = 4
    
        # Volume of atoms
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # Volume of unit cell
        v_cell = a**3
    
        # Packing fraction
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    def calculate_apf_bcc(a, r=None):
        """
        Calculate atomic packing fraction (APF) for BCC structure
    
        Parameters:
        a: Lattice parameter (Å)
        r: Atomic radius (Å). If None, calculated from lattice parameter
    
        Returns:
        apf: Packing fraction
        """
        # In BCC structure, atoms touch along body diagonal
        # Body diagonal = 4r = a3, so r = a3/4
        if r is None:
            r = a * np.sqrt(3) / 4
    
        # Number of atoms per unit cell
        n_atoms = 2
    
        # Volume of atoms
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # Volume of unit cell
        v_cell = a**3
    
        # Packing fraction
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    def calculate_apf_hcp(a, c, r=None):
        """
        Calculate atomic packing fraction (APF) for HCP structure
    
        Parameters:
        a: Lattice parameter of basal plane (Å)
        c: Lattice parameter in c-axis direction (Å)
        r: Atomic radius (Å). If None, assumed to be a/2
    
        Returns:
        apf: Packing fraction
        """
        # In HCP structure, atoms touch within the basal plane
        # a = 2r
        if r is None:
            r = a / 2
    
        # Number of atoms per unit cell
        n_atoms = 6
    
        # Volume of atoms
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # Volume of unit cell (hexagonal)
        # V = (3/2) * a² * c
        v_cell = (np.sqrt(3) / 2) * a**2 * c
    
        # Packing fraction
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    # Calculate and compare packing fractions
    print("="*70)
    print("Atomic Packing Fraction (APF) Calculation for Major Crystal Structures")
    print("="*70)
    
    # FCC (Copper as example)
    a_fcc = 3.615  # Å
    apf_fcc, r_fcc = calculate_apf_fcc(a_fcc)
    print(f"\n[FCC Structure (e.g., Copper)]")
    print(f"Lattice parameter a = {a_fcc} Å")
    print(f"Atomic radius r = {r_fcc:.4f} Å")
    print(f"Number of atoms per unit cell = 4")
    print(f"Packing fraction APF = {apf_fcc:.4f} ({apf_fcc*100:.2f}%)")
    print(f"Theoretical value: 0.7405 (74.05%)")
    
    # BCC (Iron as example)
    a_bcc = 2.866  # Å
    apf_bcc, r_bcc = calculate_apf_bcc(a_bcc)
    print(f"\n[BCC Structure (e.g., Iron)]")
    print(f"Lattice parameter a = {a_bcc} Å")
    print(f"Atomic radius r = {r_bcc:.4f} Å")
    print(f"Number of atoms per unit cell = 2")
    print(f"Packing fraction APF = {apf_bcc:.4f} ({apf_bcc*100:.2f}%)")
    print(f"Theoretical value: 0.6802 (68.02%)")
    
    # HCP (Magnesium as example, ideal c/a ratio)
    a_hcp = 3.209  # Å
    c_hcp = a_hcp * np.sqrt(8/3)  # Ideal c/a = 1.633
    apf_hcp, r_hcp = calculate_apf_hcp(a_hcp, c_hcp)
    print(f"\n[HCP Structure (e.g., Magnesium)]")
    print(f"Lattice parameter a = {a_hcp} Å")
    print(f"Lattice parameter c = {c_hcp:.4f} Å (ideal value)")
    print(f"c/a ratio = {c_hcp/a_hcp:.4f}")
    print(f"Atomic radius r = {r_hcp:.4f} Å")
    print(f"Number of atoms per unit cell = 6")
    print(f"Packing fraction APF = {apf_hcp:.4f} ({apf_hcp*100:.2f}%)")
    print(f"Theoretical value: 0.7405 (74.05%)")
    
    # Comparison
    print("\n" + "="*70)
    print("Comparison of Packing Fractions:")
    print("="*70)
    print(f"FCC: {apf_fcc*100:.2f}% - Close-packed, excellent ductility")
    print(f"BCC: {apf_bcc*100:.2f}% - Somewhat sparse, high strength")
    print(f"HCP: {apf_hcp*100:.2f}% - Close-packed, low ductility")
    
    print("\nRelationship between packing fraction and material properties:")
    print("- High packing fraction ’ High density, tendency for good ductility")
    print("- Low packing fraction ’ Many voids, easier atomic movement")
    print("- High coordination number ’ More bonds, higher stability")
    

**Sample Output** :
    
    
    ======================================================================
    Atomic Packing Fraction (APF) Calculation for Major Crystal Structures
    ======================================================================
    
    [FCC Structure (e.g., Copper)]
    Lattice parameter a = 3.615 Å
    Atomic radius r = 1.2780 Å
    Number of atoms per unit cell = 4
    Packing fraction APF = 0.7405 (74.05%)
    Theoretical value: 0.7405 (74.05%)
    
    [BCC Structure (e.g., Iron)]
    Lattice parameter a = 2.866 Å
    Atomic radius r = 1.2410 Å
    Number of atoms per unit cell = 2
    Packing fraction APF = 0.6802 (68.02%)
    Theoretical value: 0.6802 (68.02%)

**Explanation** : Atomic Packing Fraction (APF) is the ratio of the volume occupied by atoms to the unit cell volume. FCC and HCP have 74% packing (close-packed structures), while BCC has 68% (somewhat sparse structure). Packing fraction affects material density and mechanical properties.

### Code Example 4: Visualization of Coordination Number and Nearest Neighbor Distance

Visualize the coordination number (number of nearest atoms) and interatomic distances for FCC and BCC structures.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def plot_coordination_fcc():
        """
        Visualize coordination number of FCC structure with 3D plot
        Central atom (face center) and its surrounding 12 nearest neighbors
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Lattice parameter (normalized: a=1)
        a = 1.0
    
        # Position of central atom (e.g., face center at (0.5, 0.5, 0))
        center = np.array([0.5, 0.5, 0])
    
        # Relative positions of nearest neighbors in FCC structure (displacement from central atom)
        # Coordination number 12: distances to 4 different types of face centers
        nearest_neighbors = [
            # 4 in the same plane
            np.array([0.5, 0, 0]), np.array([-0.5, 0, 0]),
            np.array([0, 0.5, 0]), np.array([0, -0.5, 0]),
            # 4 in the upper plane
            np.array([0.5, 0, 1]), np.array([-0.5, 0, 1]),
            np.array([0, 0.5, 1]), np.array([0, -0.5, 1]),
            # 4 in the lower plane
            np.array([0.5, 0, -1]), np.array([-0.5, 0, -1]),
            np.array([0, 0.5, -1]), np.array([0, -0.5, -1]),
        ]
    
        # Simplification: Calculate actual positions
        # Note: The above is for conceptual explanation. Actually calculate positions within unit cell accurately
    
        # More accurate nearest neighbor positions (centered at face center)
        # FCC: Nearest atoms from face center are corners and other face centers
        neighbors_accurate = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                neighbors_accurate.append(center + np.array([i*0.5, j*0.5, 0]))
                neighbors_accurate.append(center + np.array([i*0.5, 0, j*0.5]))
                neighbors_accurate.append(center + np.array([0, i*0.5, j*0.5]))
    
        # Remove duplicates
        neighbors_unique = []
        for n in neighbors_accurate:
            is_duplicate = False
            for nu in neighbors_unique:
                if np.allclose(n, nu):
                    is_duplicate = True
                    break
            if not is_duplicate:
                neighbors_unique.append(n)
    
        # Plot central atom
        ax.scatter(*center, s=500, c='red', marker='o',
                   edgecolors='black', linewidth=2, label='Central atom', alpha=0.8)
    
        # Plot nearest neighbors
        neighbors_array = np.array(neighbors_unique)
        ax.scatter(neighbors_array[:, 0], neighbors_array[:, 1], neighbors_array[:, 2],
                   s=200, c='blue', marker='o', edgecolors='black', linewidth=1.5,
                   label='Nearest neighbors', alpha=0.6)
    
        # Draw bonds
        for neighbor in neighbors_unique:
            ax.plot([center[0], neighbor[0]],
                    [center[1], neighbor[1]],
                    [center[2], neighbor[2]],
                    'k--', linewidth=1, alpha=0.3)
    
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.set_title('Coordination Number of FCC Structure (12)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_box_aspect([1,1,1])
    
        return fig
    
    
    def calculate_nearest_neighbor_distances():
        """
        Calculate nearest neighbor distances for FCC, BCC, HCP structures
        """
        print("="*70)
        print("Calculation of Nearest Neighbor Distances")
        print("="*70)
    
        # FCC
        a_fcc = 3.615  # Copper lattice parameter (Å)
        # FCC: Face diagonal = 4r = a2 ’ Nearest distance = 2r = a/2
        d_fcc = a_fcc / np.sqrt(2)
        print(f"\n[FCC (Copper)]")
        print(f"Lattice parameter a = {a_fcc} Å")
        print(f"Nearest neighbor distance = {d_fcc:.4f} Å")
        print(f"Coordination number = 12")
    
        # BCC
        a_bcc = 2.866  # Iron lattice parameter (Å)
        # BCC: Body diagonal = 4r = a3 ’ Nearest distance = 2r = a3/2
        d_bcc = a_bcc * np.sqrt(3) / 2
        print(f"\n[BCC (Iron)]")
        print(f"Lattice parameter a = {a_bcc} Å")
        print(f"Nearest neighbor distance = {d_bcc:.4f} Å")
        print(f"Coordination number = 8")
    
        # HCP
        a_hcp = 3.209  # Magnesium lattice parameter (Å)
        # HCP: Within basal plane = a = 2r ’ Nearest distance = a
        d_hcp = a_hcp
        print(f"\n[HCP (Magnesium)]")
        print(f"Lattice parameter a = {a_hcp} Å")
        print(f"Nearest neighbor distance = {d_hcp:.4f} Å")
        print(f"Coordination number = 12")
    
        # Comparison graphs
        structures = ['FCC\n(Cu)', 'BCC\n(Fe)', 'HCP\n(Mg)']
        distances = [d_fcc, d_bcc, d_hcp]
        coordination_numbers = [12, 8, 12]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Comparison of nearest neighbor distances
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax1.bar(structures, distances, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Nearest Neighbor Distance (Å)', fontsize=12, fontweight='bold')
        ax1.set_title('Nearest Neighbor Distance by Crystal Structure', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
        # Display values on top of bars
        for i, (s, d) in enumerate(zip(structures, distances)):
            ax1.text(i, d + 0.05, f'{d:.3f} Å', ha='center', fontsize=11, fontweight='bold')
    
        # Comparison of coordination numbers
        ax2.bar(structures, coordination_numbers, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Coordination Number', fontsize=12, fontweight='bold')
        ax2.set_title('Coordination Number by Crystal Structure', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 14)
        ax2.grid(axis='y', alpha=0.3)
    
        # Display values on top of bars
        for i, (s, cn) in enumerate(zip(structures, coordination_numbers)):
            ax2.text(i, cn + 0.3, f'{cn}', ha='center', fontsize=12, fontweight='bold')
    
        plt.tight_layout()
        plt.show()
    
        print("\n" + "="*70)
        print("Relationship between coordination number and material properties:")
        print("- High coordination number ’ More bonds ’ Stable, densely packed")
        print("- FCC, HCP: Coordination number 12 ’ Close-packed, excellent ductility")
        print("- BCC: Coordination number 8 ’ Somewhat sparse, high strength but brittle at low temperatures")
    
    
    # Execute
    calculate_nearest_neighbor_distances()
    # plot_coordination_fcc()  # Uncomment if 3D plot is needed
    plt.show()
    

**Explanation** : Coordination number is the number of atoms arranged closest around a given atom. The higher the coordination number, the more interatomic bonds exist, and the material is more stable and densely packed. FCC and HCP have coordination number 12 (close-packed), while BCC has coordination number 8 (somewhat sparse structure).

* * *

## 3.4 Crystal Structure and Material Properties

### Density Calculation

The theoretical density of a material can be calculated from its crystal structure:

$$\rho = \frac{n \cdot M}{V_{cell} \cdot N_A}$$

where,

  * $\rho$: Density (g/cm³)
  * $n$: Number of atoms per unit cell
  * $M$: Atomic weight (g/mol)
  * $V_{cell}$: Unit cell volume (cm³)
  * $N_A$: Avogadro's number (6.022×10²³ mol{¹)

### Slip Systems and Ductility

**Slip systems** are combinations of planes and directions along which atomic planes slide during plastic deformation. The more slip systems available, the better the ductility of the material.

**Slip systems for major crystal structures** :

  * **FCC** : {111}è110é ’ 4 planes × 3 directions = 12 slip systems ’ High ductility
  * **BCC** : {110}è111é, {112}è111é, {123}è111é ’ 48 slip systems (theoretically) ’ Medium ductility
  * **HCP** : {0001}è1120é ’ 1 plane × 3 directions = 3 slip systems ’ Low ductility

**Relationship between crystal structure and mechanical properties** :

  * FCC metals (Cu, Al, Au): Many slip systems ’ High ductility, good workability
  * BCC metals (Fe, Cr, W): High temperature dependence ’ Brittle at low temperatures, ductile at high temperatures
  * HCP metals (Mg, Zn, Ti): Few slip systems ’ Low ductility, difficult to process

### Code Example 5: Density Calculation Tool

Calculate the theoretical density of materials from crystal structure parameters.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    # Avogadro's number
    NA = 6.022e23  # mol^-1
    
    def calculate_density(n_atoms, atomic_mass, a, b=None, c=None,
                         alpha=90, beta=90, gamma=90):
        """
        Calculate theoretical density from crystal structure
    
        Parameters:
        n_atoms: Number of atoms per unit cell
        atomic_mass: Atomic weight (g/mol)
        a, b, c: Lattice parameters (Å)
        alpha, beta, gamma: Angles (degrees)
    
        Returns:
        density: Density (g/cm³)
        """
        # Set default values for lattice parameters
        if b is None:
            b = a
        if c is None:
            c = a
    
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
    
        # Calculate unit cell volume (Å³)
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
    
        V_cell = a * b * c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
            + 2*cos_alpha*cos_beta*cos_gamma
        )
    
        # Convert Å³ ’ cm³ (1 Å = 10^-8 cm)
        V_cell_cm3 = V_cell * 1e-24
    
        # Calculate density
        density = (n_atoms * atomic_mass) / (V_cell_cm3 * NA)
    
        return density
    
    
    # Calculate theoretical density for representative materials
    print("="*70)
    print("Theoretical Density Calculated from Crystal Structure")
    print("="*70)
    
    materials = [
        {
            'name': 'Copper (Cu, FCC)',
            'n_atoms': 4,
            'atomic_mass': 63.546,  # g/mol
            'a': 3.615,  # Å
            'structure': 'FCC'
        },
        {
            'name': 'Iron (Fe, BCC)',
            'n_atoms': 2,
            'atomic_mass': 55.845,
            'a': 2.866,
            'structure': 'BCC'
        },
        {
            'name': 'Aluminum (Al, FCC)',
            'n_atoms': 4,
            'atomic_mass': 26.982,
            'a': 4.049,
            'structure': 'FCC'
        },
        {
            'name': 'Tungsten (W, BCC)',
            'n_atoms': 2,
            'atomic_mass': 183.84,
            'a': 3.165,
            'structure': 'BCC'
        },
        {
            'name': 'Magnesium (Mg, HCP)',
            'n_atoms': 6,
            'atomic_mass': 24.305,
            'a': 3.209,
            'c': 5.211,
            'gamma': 120,
            'structure': 'HCP'
        },
        {
            'name': 'Gold (Au, FCC)',
            'n_atoms': 4,
            'atomic_mass': 196.967,
            'a': 4.078,
            'structure': 'FCC'
        }
    ]
    
    # Calculate and display density
    calculated_densities = []
    experimental_densities = [8.96, 7.87, 2.70, 19.25, 1.74, 19.32]  # g/cm³ (measured values)
    
    for i, mat in enumerate(materials):
        if 'c' in mat:
            density = calculate_density(mat['n_atoms'], mat['atomic_mass'],
                                       mat['a'], c=mat['c'], gamma=mat.get('gamma', 90))
        else:
            density = calculate_density(mat['n_atoms'], mat['atomic_mass'], mat['a'])
    
        calculated_densities.append(density)
    
        print(f"\n[{mat['name']}]")
        print(f"Crystal structure: {mat['structure']}")
        print(f"Number of atoms per unit cell: {mat['n_atoms']}")
        print(f"Atomic weight: {mat['atomic_mass']} g/mol")
        print(f"Lattice parameter a = {mat['a']} Å" + (f", c = {mat['c']} Å" if 'c' in mat else ""))
        print(f"Calculated density: {density:.3f} g/cm³")
        print(f"Experimental density: {experimental_densities[i]:.3f} g/cm³")
        print(f"Error: {abs(density - experimental_densities[i]) / experimental_densities[i] * 100:.2f}%")
    
    # Comparison graph
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(materials))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, calculated_densities, width, label='Calculated density',
                    color='#1f77b4', edgecolor='black', linewidth=1.5, alpha=0.7)
    rects2 = ax.bar(x + width/2, experimental_densities, width, label='Experimental density',
                    color='#ff7f0e', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax.set_ylabel('Density (g/cm³)', fontsize=13, fontweight='bold')
    ax.set_title('Comparison of Calculated Density from Crystal Structure and Experimental Values', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m['name'].split('(')[0].strip() for m in materials], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Display values on top of bars
    for rect1, rect2 in zip(rects1, rects2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.text(rect1.get_x() + rect1.get_width()/2., height1,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(rect2.get_x() + rect2.get_width()/2., height2,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("Applications of density calculation:")
    print("- Material identification (determine crystal structure from XRD pattern and density)")
    print("- Evaluation of defects and impurities (estimate vacancy concentration from difference between experimental and theoretical density)")
    print("- Lightweight design (optimize balance between density and strength)")
    

**Sample Output** :
    
    
    ======================================================================
    Theoretical Density Calculated from Crystal Structure
    ======================================================================
    
    [Copper (Cu, FCC)]
    Crystal structure: FCC
    Number of atoms per unit cell: 4
    Atomic weight: 63.546 g/mol
    Lattice parameter a = 3.615 Å
    Calculated density: 8.933 g/cm³
    Experimental density: 8.960 g/cm³
    Error: 0.30%
    
    [Iron (Fe, BCC)]
    Crystal structure: BCC
    Number of atoms per unit cell: 2
    Atomic weight: 55.845 g/mol
    Lattice parameter a = 2.866 Å
    Calculated density: 7.879 g/cm³
    Experimental density: 7.870 g/cm³
    Error: 0.11%

**Explanation** : Theoretical density can be calculated with high accuracy from crystal structure parameters. The difference between calculated and experimental values can be used to estimate vacancy concentration and other defects.

### Code Example 6: Visualization of Slip Systems

Visualize the major slip system {111}è110é of FCC structure in 2D.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, FancyArrowPatch
    
    def visualize_slip_systems_fcc():
        """
        Visualize slip systems of FCC structure in 2D projection
        {111}<110> slip system
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
        # Draw projection of (111) plane
        # Unit cell of FCC structure (simplified 2D representation)
    
        # Examples of slip planes and slip directions
        slip_systems = [
            {
                'plane': '(111)',
                'direction': '[110]',
                'description': 'Most active slip system'
            },
            {
                'plane': '(111)',
                'direction': '[110]',
                'description': 'Symmetrical slip system'
            },
            {
                'plane': '(111)',
                'direction': '[101]',
                'description': 'Additional slip system'
            }
        ]
    
        for idx, (ax, slip_sys) in enumerate(zip(axes, slip_systems)):
            # Atomic positions (simplified 2D array)
            atoms_x = [0, 1, 2, 0.5, 1.5, 1]
            atoms_y = [0, 0, 0, 0.866, 0.866, 1.732]
    
            # Plot atoms
            ax.scatter(atoms_x, atoms_y, s=300, c='lightblue',
                      edgecolors='black', linewidth=2, zorder=3)
    
            # Show slip plane (gray band)
            slip_plane = Polygon([(0, 0), (2, 0), (2.5, 0.866), (0.5, 0.866)],
                                alpha=0.2, facecolor='gray', edgecolor='black',
                                linewidth=1.5, linestyle='--', zorder=1)
            ax.add_patch(slip_plane)
    
            # Show slip direction with arrow
            arrow = FancyArrowPatch((0.5, 0.4), (1.8, 0.4),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='red', zorder=2)
            ax.add_patch(arrow)
    
            ax.set_xlim(-0.5, 3)
            ax.set_ylim(-0.5, 2.5)
            ax.set_aspect('equal')
            ax.set_title(f'{slip_sys["plane"]} plane\n{slip_sys["direction"]} direction\n({slip_sys["description"]})',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
    
            # Legend
            ax.text(0.1, 2.2, 'Ï Atoms', fontsize=9)
            ax.text(0.1, 2.0, '   Slip plane', fontsize=9, color='gray')
            ax.text(0.1, 1.8, '’ Slip direction', fontsize=9, color='red')
    
        plt.suptitle('Slip Systems of FCC Structure {111}è110é (Total 12 systems)',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
        # Explain relationship between slip systems and ductility
        print("="*70)
        print("Slip Systems and Material Ductility")
        print("="*70)
    
        print("\n[FCC Structure]")
        print("- Slip planes: {111} (4 planes)")
        print("- Slip directions: è110é (3 directions per plane)")
        print("- Total slip systems: 4 × 3 = 12")
        print("- Ductility: Very high (many slip systems)")
        print("- Examples: Cu, Al, Au, Ag ’ Easy plastic deformation")
    
        print("\n[BCC Structure]")
        print("- Slip planes: {110}, {112}, {123} (multiple)")
        print("- Slip directions: è111é")
        print("- Total slip systems: 48 (theoretically)")
        print("- Ductility: High temperature dependence (limited at low temperatures)")
        print("- Examples: Fe, Cr, W ’ Prone to brittle fracture at low temperatures")
    
        print("\n[HCP Structure]")
        print("- Slip planes: {0001} (1 plane, basal plane)")
        print("- Slip directions: è1120é (3 directions)")
        print("- Total slip systems: 1 × 3 = 3 (very few)")
        print("- Ductility: Low (limited slip systems)")
        print("- Examples: Mg, Zn, Ti ’ Difficult to process at room temperature")
    
        print("\n" + "="*70)
        print("Number of slip systems and workability:")
        print("- Many slip systems ’ Easy plastic deformation ’ High ductility, good workability")
        print("- Few slip systems ’ Difficult plastic deformation ’ Low ductility, prone to brittle fracture")
        print("- Material selection: FCC metals are advantageous for applications requiring processing")
    
    
    # Execute
    visualize_slip_systems_fcc()
    

**Explanation** : Slip systems are combinations of planes and directions along which atomic planes slide during plastic deformation. The more slip systems available, the more directions from which the material can receive forces and still plastically deform, resulting in excellent ductility. FCC has 12 slip systems (high ductility), while HCP has 3 slip systems (low ductility).

### Code Example 7: 3D Visualization of Crystal Structures (FCC, BCC, HCP)

Visualize the three major crystal structures in 3D to understand the differences in structure.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def plot_crystal_structure_3d(structure_type='FCC'):
        """
        Visualize crystal structure in 3D
    
        Parameters:
        structure_type: One of 'FCC', 'BCC', 'HCP'
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Lattice parameter (normalized: a=1)
        a = 1.0
    
        if structure_type == 'FCC':
            # Atomic positions for FCC structure
            positions = [
                # 8 corners (each 1/8)
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                # 6 face centers (each 1/2)
                [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
                [a/2, a/2, a], [a/2, a, a/2], [a, a/2, a/2]
            ]
            title = 'FCC (Face-Centered Cubic)'
            description = '4 atoms per unit cell\nCoordination number 12, Packing fraction 74%'
    
        elif structure_type == 'BCC':
            # Atomic positions for BCC structure
            positions = [
                # 8 corners (each 1/8)
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                # 1 body center
                [a/2, a/2, a/2]
            ]
            title = 'BCC (Body-Centered Cubic)'
            description = '2 atoms per unit cell\nCoordination number 8, Packing fraction 68%'
    
        elif structure_type == 'HCP':
            # Atomic positions for HCP structure (simplified representation)
            c_a_ratio = 1.633  # Ideal c/a ratio
            c = a * c_a_ratio
    
            # 3 corners of basal plane + 3 corners of top plane + 2 internal atoms
            positions = [
                # Lower basal plane (3 corners)
                [0, 0, 0], [a, 0, 0], [a/2, a*np.sqrt(3)/2, 0],
                # Upper basal plane
                [0, 0, c], [a, 0, c], [a/2, a*np.sqrt(3)/2, c],
                # Internal (2 atoms)
                [a/2, a/(2*np.sqrt(3)), c/2], [a/2, a*np.sqrt(3)/6, c/2]
            ]
            title = 'HCP (Hexagonal Close-Packed)'
            description = '6 atoms per unit cell\nCoordination number 12, Packing fraction 74%'
        else:
            raise ValueError("structure_type must be one of 'FCC', 'BCC', 'HCP'")
    
        # Convert atomic positions to array
        positions = np.array(positions)
    
        # Plot atoms
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=300, c='lightblue', edgecolors='darkblue', linewidth=2,
                  alpha=0.8, depthshade=True)
    
        # Draw unit cell frame
        # Edges of cube
        edges = [
            # Bottom face
            [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
            [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
            # Top face
            [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
            [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
            # Vertical edges
            [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
            [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
        ]
    
        if structure_type != 'HCP':
            for edge in edges:
                edge = np.array(edge)
                ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2],
                         'k-', linewidth=1.5, alpha=0.6)
        else:
            # For HCP, hexagonal prism frame
            # Simplified, omitted (implementation is complex)
            pass
    
        # Axis labels
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n{description}', fontsize=13, fontweight='bold', pad=20)
    
        # Adjust viewpoint
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
        # Turn off grid
        ax.grid(False)
    
        return fig, ax
    
    
    # Display three crystal structures side by side
    fig = plt.figure(figsize=(18, 6))
    
    for i, structure in enumerate(['FCC', 'BCC', 'HCP'], 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
    
        # Lattice parameter
        a = 1.0
    
        if structure == 'FCC':
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
                [a/2, a/2, a], [a/2, a, a/2], [a, a/2, a/2]
            ])
            title = 'FCC'
            info = 'Atoms: 4\nCoordination: 12\nAPF: 74%'
    
        elif structure == 'BCC':
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                [a/2, a/2, a/2]
            ])
            title = 'BCC'
            info = 'Atoms: 2\nCoordination: 8\nAPF: 68%'
    
        else:  # HCP
            c = a * 1.633
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a/2, a*np.sqrt(3)/2, 0],
                [0, 0, c], [a, 0, c], [a/2, a*np.sqrt(3)/2, c],
                [a/2, a/(2*np.sqrt(3)), c/2]
            ])
            title = 'HCP'
            info = 'Atoms: 6\nCoordination: 12\nAPF: 74%'
    
        # Plot atoms
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=250, c='lightblue', edgecolors='darkblue', linewidth=2,
                  alpha=0.8, depthshade=True)
    
        # Unit cell frame
        if structure != 'HCP':
            edges = [
                [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
                [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
                [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
                [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
                [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
                [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
            ]
            for edge in edges:
                edge = np.array(edge)
                ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2],
                         'k-', linewidth=1.5, alpha=0.5)
    
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'{title}\n{info}', fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
        ax.grid(False)
    
    plt.suptitle('3D Visualization of Major Crystal Structures', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("Significance of crystal structure visualization:")
    print("="*70)
    print("\n3D visualization helps understand:")
    print("- Spatial arrangement of atoms (where atoms are located)")
    print("- Packing density (densely packed or with voids)")
    print("- Symmetry (relationship between structural beauty and properties)")
    print("- Prediction of slip planes (which planes slip easily)")
    
    print("\nApplications in actual material development:")
    print("- Prediction of crystal structures for new materials")
    print("- Understanding of phase transformations (e.g., FCC’BCC)")
    print("- Prediction of material properties (estimate properties from structure)")
    

**Explanation** : 3D visualization clearly shows the differences in atomic arrangement among FCC, BCC, and HCP. FCC and BCC belong to the cubic system, while HCP belongs to the hexagonal system, with different packing densities (packing fractions) and coordination numbers. These differences significantly affect the mechanical properties of materials.

* * *

## 3.5 Chapter Summary

### What We Learned

  1. **Difference Between Crystalline and Amorphous Materials**
     * Crystalline: Long-range order, sharp melting point, sharp XRD peaks
     * Amorphous: Short-range order only, glass transition, broad XRD patterns
  2. **Unit Cell and Lattice Parameters**
     * Unit cell: Smallest repeating unit of crystal
     * Seven crystal systems: Cubic, tetragonal, orthorhombic, hexagonal, rhombohedral, monoclinic, triclinic
     * Miller indices: Notation for crystal planes with (hkl) and directions with [uvw]
  3. **Major Crystal Structures**
     * FCC: 4 atoms, coordination 12, APF 74%, high ductility (Cu, Al, Au)
     * BCC: 2 atoms, coordination 8, APF 68%, high strength (Fe, Cr, W)
     * HCP: 6 atoms, coordination 12, APF 74%, low ductility (Mg, Zn, Ti)
  4. **Crystal Structure and Material Properties**
     * Density can be calculated from crystal structure
     * Number of slip systems determines ductility
     * FCC: 12 slip systems ’ High ductility
     * HCP: 3 slip systems ’ Low ductility
  5. **Visualization with Python**
     * Simulation of XRD patterns
     * Calculation of packing fraction and coordination number
     * Density calculation tool
     * 3D visualization of crystal structures

### Key Points

  * Crystal structure is a **critical factor governing mechanical properties** of materials
  * Higher packing fraction generally means higher density and better ductility
  * Number of slip systems determines ductility (more systems = higher ductility)
  * XRD is the most important technique for crystal structure analysis
  * Miller indices are the standard method for representing crystal planes and directions

### To the Next Chapter

In Chapter 4, we will learn about the **relationship between material properties and structure** :

  * Mechanical properties (stress-strain curves, hardness)
  * Electrical properties (band structure, conductivity)
  * Thermal properties (thermal conductivity, thermal expansion)
  * Optical properties (absorption spectra, color)
  * Calculation and plotting of material properties using Python
