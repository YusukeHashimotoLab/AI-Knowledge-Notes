---
title: "Chapter 1: Fundamentals of X-ray Diffraction"
chapter_title: "Chapter 1: Fundamentals of X-ray Diffraction"
subtitle: From Bragg's Law to Structure Factors - Theoretical Foundations of Crystal Structure Analysis
reading_time: 25-30 min
difficulty: Intermediate
code_examples: 8
---

This chapter covers the fundamentals of Fundamentals of X, which ray diffraction. You will learn Infer crystal symmetry from systematic absences and structure factor calculations.

## Learning Objectives

Upon completing this chapter, you will be able to:

  * Understand Bragg's law and calculate diffraction conditions using formulas and Python
  * Visualize and explain the relationship between crystal lattice and reciprocal lattice
  * Infer crystal symmetry from systematic absences
  * Implement structure factor calculations and predict diffraction intensities
  * Visualize diffraction conditions using the Ewald sphere

## 1.1 Bragg's Law and Diffraction Conditions

X-ray diffraction (XRD) is the most fundamental technique for crystal structure analysis. When X-rays interact with the atomic arrangement in a crystal, diffraction occurs at specific angles, providing information about the crystal structure.

### 1.1.1 Derivation of Bragg's Law

Bragg's law was discovered in 1912 by William Lawrence Bragg and William Henry Bragg. Considering a crystal as a collection of atomic planes, constructive interference occurs when the path difference between X-rays reflected from adjacent atomic planes equals an integer multiple of the wavelength.

When X-rays of wavelength \\( \lambda \\) are incident at angle \\( \theta \\) on crystal planes with spacing \\( d \\), the diffraction condition is expressed as:

\\[ 2d\sin\theta = n\lambda \\] 

Here, \\( n \\) is the order of diffraction (integer), \\( d \\) is the plane spacing, \\( \theta \\) is the incident angle (Bragg angle), and \\( \lambda \\) is the X-ray wavelength.
    
    
    ```mermaid
    graph LR
        A[X-ray Incidence» = 1.54 Å] --> B[Crystal Planed = 3.5 Å]
        B --> C{Bragg Condition2dsin¸ = n»}
        C -->|Satisfied| D[Diffraction PeakObserved]
        C -->|Not Satisfied| E[No DiffractionNot Observed]
        style C fill:#fce7f3
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### 1.1.2 Python Implementation for Calculations
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def bragg_law(d, wavelength, n=1):
        """Calculate diffraction angle using Bragg's law
    
        Args:
            d (float): Plane spacing [Å]
            wavelength (float): X-ray wavelength [Å]
            n (int): Order of diffraction
    
        Returns:
            float: Bragg angle [degrees], or None if diffraction is impossible
        """
        # 2d*sin(theta) = n*lambda
        # sin(theta) = n*lambda / (2*d)
        sin_theta = (n * wavelength) / (2 * d)
    
        # Check for physically valid solution
        if sin_theta > 1.0:
            return None  # Diffraction impossible
    
        theta_rad = np.arcsin(sin_theta)
        theta_deg = np.degrees(theta_rad)
    
        return theta_deg
    
    
    # Example: Calculation using Cu K± radiation (1.54 Å)
    wavelength_CuKa = 1.54056  # Å
    
    # Calculate diffraction angles for different plane spacings
    d_spacings = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5]  # Å
    
    print("d-spacing [Å] | Bragg angle [deg] (n=1) | Bragg angle [deg] (n=2)")
    print("-" * 60)
    for d in d_spacings:
        theta_n1 = bragg_law(d, wavelength_CuKa, n=1)
        theta_n2 = bragg_law(d, wavelength_CuKa, n=2)
    
        if theta_n1 is not None:
            print(f"{d:6.2f}      | {theta_n1:15.2f}    | ", end="")
            if theta_n2 is not None:
                print(f"{theta_n2:15.2f}")
            else:
                print("   Impossible")
        else:
            print(f"{d:6.2f}      |    Impossible    |    Impossible")
    
    # Expected output:
    # d-spacing [Å] | Bragg angle [deg] (n=1) | Bragg angle [deg] (n=2)
    # ------------------------------------------------------------
    #   4.00      |           11.10    |            22.48
    #   3.50      |           12.69    |            25.77
    #   3.00      |           14.83    |            30.36
    #   2.50      |           17.93    |            37.32
    #   2.00      |           22.58    |            48.92
    #   1.50      |           30.96    |            75.55

### 1.1.3 Visualization of Diffraction Angles
    
    
    def plot_bragg_angles(wavelength, d_range=(1.0, 5.0), n_max=3):
        """Plot the dependence of Bragg angle on plane spacing
    
        Args:
            wavelength (float): X-ray wavelength [Å]
            d_range (tuple): Range of plane spacing [Å]
            n_max (int): Maximum order of diffraction
        """
        d_values = np.linspace(d_range[0], d_range[1], 200)
    
        plt.figure(figsize=(10, 6))
        colors = ['#f093fb', '#f5576c', '#3498db']
    
        for n in range(1, n_max + 1):
            theta_values = []
            valid_d_values = []
    
            for d in d_values:
                theta = bragg_law(d, wavelength, n)
                if theta is not None:
                    theta_values.append(theta)
                    valid_d_values.append(d)
    
            if theta_values:
                plt.plot(valid_d_values, theta_values,
                        label=f'n = {n}', linewidth=2, color=colors[n-1])
    
        plt.xlabel('Plane spacing d [Å]', fontsize=12)
        plt.ylabel('Bragg angle ¸ [degrees]', fontsize=12)
        plt.title(f'Bragg\'s Law: Relationship between Diffraction Angle and Plane Spacing (» = {wavelength:.2f} Å)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xlim(d_range)
        plt.ylim(0, 90)
        plt.tight_layout()
        plt.show()
    
    # For Cu K± radiation
    plot_bragg_angles(1.54056, d_range=(1.0, 5.0), n_max=3)

## 1.2 Crystal Lattice and Reciprocal Lattice

### 1.2.1 Fundamentals of Crystal Lattice

A crystal is a structure where basic units (unit cells) are periodically arranged in three-dimensional space. The unit cell is defined by three basis vectors \\( \mathbf{a}, \mathbf{b}, \mathbf{c} \\).

The position vector \\( \mathbf{R} \\) of any lattice point is expressed as:

\\[ \mathbf{R} = u\mathbf{a} + v\mathbf{b} + w\mathbf{c} \\] 

Here, \\( u, v, w \\) are integers.

### 1.2.2 Reciprocal Lattice Vectors

The reciprocal lattice is an extremely important concept for understanding diffraction phenomena in crystals. The reciprocal lattice vectors \\( \mathbf{a}^*, \mathbf{b}^*, \mathbf{c}^* \\) are defined as follows:

\\[ \mathbf{a}^* = \frac{\mathbf{b} \times \mathbf{c}}{V}, \quad \mathbf{b}^* = \frac{\mathbf{c} \times \mathbf{a}}{V}, \quad \mathbf{c}^* = \frac{\mathbf{a} \times \mathbf{b}}{V} \\] 

Here, \\( V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) \\) is the volume of the unit cell.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def reciprocal_lattice_vectors(a, b, c):
        """Calculate reciprocal lattice vectors from real lattice vectors
    
        Args:
            a, b, c (np.ndarray): Real lattice basis vectors [Å]
    
        Returns:
            tuple: (a*, b*, c*) Reciprocal lattice vectors [Å^-1]
        """
        # Unit cell volume
        V = np.dot(a, np.cross(b, c))
    
        # Reciprocal lattice vectors
        a_star = np.cross(b, c) / V
        b_star = np.cross(c, a) / V
        c_star = np.cross(a, b) / V
    
        return a_star, b_star, c_star
    
    
    def d_spacing_from_hkl(h, k, l, a_star, b_star, c_star):
        """Calculate plane spacing from Miller indices
    
        Args:
            h, k, l (int): Miller indices
            a_star, b_star, c_star (np.ndarray): Reciprocal lattice vectors [Å^-1]
    
        Returns:
            float: Plane spacing d [Å]
        """
        # Reciprocal lattice vector G = h*a* + k*b* + l*c*
        G = h * a_star + k * b_star + l * c_star
    
        # Plane spacing d = 1 / |G|
        d = 1.0 / np.linalg.norm(G)
    
        return d
    
    
    # Example: Cubic crystal system (a = b = c = 4.0 Å, 90 degree angles)
    a = np.array([4.0, 0.0, 0.0])
    b = np.array([0.0, 4.0, 0.0])
    c = np.array([0.0, 0.0, 4.0])
    
    a_star, b_star, c_star = reciprocal_lattice_vectors(a, b, c)
    
    print("Real lattice vectors:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}\n")
    
    print("Reciprocal lattice vectors:")
    print(f"a* = {a_star} [Å^-1]")
    print(f"b* = {b_star} [Å^-1]")
    print(f"c* = {c_star} [Å^-1]\n")
    
    # Calculate plane spacings for various Miller indices
    miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 2, 0)]
    
    print("Miller indices | Plane spacing d [Å]")
    print("-" * 30)
    for h, k, l in miller_indices:
        d = d_spacing_from_hkl(h, k, l, a_star, b_star, c_star)
        print(f" ({h} {k} {l})      | {d:12.4f}")
    
    # Expected output (cubic crystal):
    # (1 0 0)      |       4.0000
    # (1 1 0)      |       2.8284
    # (1 1 1)      |       2.3094
    # (2 0 0)      |       2.0000
    # (2 2 0)      |       1.4142

## 1.3 Systematic Absences and Symmetry

### 1.3.1 Systematic Absences

Systematic absences refer to the phenomenon where reflections with specific Miller indices disappear due to crystal symmetry (space group). This reflects the symmetry of atomic arrangements within the unit cell.

**Representative systematic absences:**

Lattice Type | Systematic Absence | Example  
---|---|---  
Body-centered cubic (BCC) | Reflections with \\( h + k + l = odd \\) are absent | (1,0,0), (1,1,0) are absent  
Face-centered cubic (FCC) | Reflections with mixed \\( h, k, l \\) (mixed parity) are absent | (1,0,0), (1,1,0) are absent  
C-centered orthorhombic (C) | Reflections with \\( h + k = odd \\) are absent | (1,0,0), (0,1,0) are absent  
  
### 1.3.2 Implementation of Systematic Absences
    
    
    def systematic_absences(h, k, l, lattice_type):
        """Determine systematic absences
    
        Args:
            h, k, l (int): Miller indices
            lattice_type (str): Lattice type ('P', 'I', 'F', 'C', 'A', 'B')
    
        Returns:
            bool: True = reflection observed, False = absent
        """
        if lattice_type == 'P':  # Primitive lattice
            return True
    
        elif lattice_type == 'I':  # Body-centered lattice
            return (h + k + l) % 2 == 0
    
        elif lattice_type == 'F':  # Face-centered lattice
            # h, k, l are all even or all odd
            parity = [h % 2, k % 2, l % 2]
            return len(set(parity)) == 1
    
        elif lattice_type == 'C':  # C-centered
            return (h + k) % 2 == 0
    
        elif lattice_type == 'A':  # A-centered
            return (k + l) % 2 == 0
    
        elif lattice_type == 'B':  # B-centered
            return (h + l) % 2 == 0
    
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    
    # Test: Confirm systematic absences for various lattice types
    lattice_types = ['P', 'I', 'F', 'C']
    miller_list = [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0), (2,2,2)]
    
    print("Miller indices | P (Primitive) | I (BCC) | F (FCC) | C (C-centered)")
    print("-" * 65)
    
    for hkl in miller_list:
        h, k, l = hkl
        results = [systematic_absences(h, k, l, lt) for lt in lattice_types]
        status = [' ' if r else ' -' for r in results]
        print(f" ({h} {k} {l})     | {status[0]}      | {status[1]}      | {status[2]}      | {status[3]}")
    
    # Expected output:
    #  (1 0 0)     |        |  -      |  -      |  -
    #  (1 1 0)     |        |        |  -      |  -
    #  (1 1 1)     |        |  -      |        |  
    #  (2 0 0)     |        |        |        |  
    #  (2 2 0)     |        |        |        |  
    #  (2 2 2)     |        |        |        |  

## 1.4 Structure Factor

### 1.4.1 Definition of Structure Factor

The structure factor \\( F_{hkl} \\) is a complex number that determines the diffraction intensity for specific Miller indices \\( (h, k, l) \\). It is the sum of the amplitudes and phases of scattered waves from all atoms in the unit cell.

\\[ F_{hkl} = \sum_{j=1}^{N} f_j \exp\left[2\pi i (hx_j + ky_j + lz_j)\right] \\] 

Here, \\( f_j \\) is the atomic scattering factor of atom \\( j \\), and \\( (x_j, y_j, z_j) \\) are the fractional coordinates of atom \\( j \\).

The diffraction intensity \\( I_{hkl} \\) is proportional to the square of the absolute value of the structure factor:

\\[ I_{hkl} \propto |F_{hkl}|^2 = F_{hkl} \cdot F_{hkl}^* \\] 

### 1.4.2 Implementation of Structure Factor Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def structure_factor(h, k, l, atoms, scattering_factors):
        """Calculate structure factor
    
        Args:
            h, k, l (int): Miller indices
            atoms (list): List of atomic positions [(x1, y1, z1), (x2, y2, z2), ...]
            scattering_factors (list): Scattering factor for each atom [f1, f2, ...]
    
        Returns:
            complex: Structure factor F_hkl
        """
        F_hkl = 0.0 + 0.0j  # Complex number
    
        for (x, y, z), f_j in zip(atoms, scattering_factors):
            # Phase factor exp[2Ài(hx + ky + lz)]
            phase = 2 * np.pi * (h * x + k * y + l * z)
            F_hkl += f_j * np.exp(1j * phase)
    
        return F_hkl
    
    
    def intensity_from_structure_factor(F_hkl):
        """Calculate diffraction intensity from structure factor
    
        Args:
            F_hkl (complex): Structure factor
    
        Returns:
            float: Diffraction intensity I  |F|^2
        """
        return np.abs(F_hkl) ** 2
    
    
    # Example: Simple cubic (SC) - 1 atom at (0, 0, 0)
    print("=== Simple Cubic (SC) ===")
    atoms_sc = [(0, 0, 0)]
    f_sc = [1.0]  # Normalized scattering factor
    
    for hkl in [(1,0,0), (1,1,0), (1,1,1), (2,0,0)]:
        h, k, l = hkl
        F = structure_factor(h, k, l, atoms_sc, f_sc)
        I = intensity_from_structure_factor(F)
        print(f"({h} {k} {l}): F = {F:.4f}, I = {I:.4f}")
    
    # Example: Body-centered cubic (BCC) - 2 atoms at (0,0,0), (1/2,1/2,1/2)
    print("\n=== Body-Centered Cubic (BCC) ===")
    atoms_bcc = [(0, 0, 0), (0.5, 0.5, 0.5)]
    f_bcc = [1.0, 1.0]
    
    for hkl in [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0)]:
        h, k, l = hkl
        F = structure_factor(h, k, l, atoms_bcc, f_bcc)
        I = intensity_from_structure_factor(F)
        print(f"({h} {k} {l}): F = {F.real:6.4f}{F.imag:+6.4f}i, I = {I:.4f}")
    
    # Expected output:
    # BCC: (1,0,0) and (1,1,0) have I=0 (systematic absence), (1,1,1) and (2,2,0) have I>0

### 1.4.3 Application to Real Crystals
    
    
    def calculate_xrd_pattern(a, lattice_type, atom_positions, scattering_factors,
                              wavelength=1.54056, two_theta_max=90):
        """Simulate XRD pattern
    
        Args:
            a (float): Lattice constant [Å]
            lattice_type (str): Lattice type ('P', 'I', 'F')
            atom_positions (list): Fractional coordinates of atoms
            scattering_factors (list): Atomic scattering factors
            wavelength (float): X-ray wavelength [Å]
            two_theta_max (float): Maximum 2¸ angle [degrees]
    
        Returns:
            tuple: (two_theta_list, intensity_list)
        """
        # Reciprocal lattice vectors (simplified for cubic)
        a_vec = np.array([a, 0, 0])
        b_vec = np.array([0, a, 0])
        c_vec = np.array([0, 0, a])
        a_star, b_star, c_star = reciprocal_lattice_vectors(a_vec, b_vec, c_vec)
    
        two_theta_list = []
        intensity_list = []
    
        # Scan Miller indices
        for h in range(0, 6):
            for k in range(0, 6):
                for l in range(0, 6):
                    if h == 0 and k == 0 and l == 0:
                        continue
    
                    # Check systematic absences
                    if not systematic_absences(h, k, l, lattice_type):
                        continue
    
                    # Calculate plane spacing
                    d = d_spacing_from_hkl(h, k, l, a_star, b_star, c_star)
    
                    # Calculate Bragg angle
                    theta = bragg_law(d, wavelength, n=1)
                    if theta is None or 2*theta > two_theta_max:
                        continue
    
                    # Calculate structure factor and intensity
                    F = structure_factor(h, k, l, atom_positions, scattering_factors)
                    I = intensity_from_structure_factor(F)
    
                    # Lorentz-polarization factor (simplified)
                    LP = (1 + np.cos(2*np.radians(theta))**2) / (np.sin(np.radians(theta))**2 * np.cos(np.radians(theta)))
                    I_corrected = I * LP
    
                    two_theta_list.append(2 * theta)
                    intensity_list.append(I_corrected)
    
        return np.array(two_theta_list), np.array(intensity_list)
    
    
    # Simulation: ±-Fe (BCC, a = 2.87 Å)
    a_Fe = 2.87
    atoms_Fe = [(0, 0, 0), (0.5, 0.5, 0.5)]
    f_Fe = [26.0, 26.0]  # Atomic number of Fe
    
    two_theta, intensity = calculate_xrd_pattern(
        a=a_Fe,
        lattice_type='I',
        atom_positions=atoms_Fe,
        scattering_factors=f_Fe,
        wavelength=1.54056,
        two_theta_max=120
    )
    
    # Display peaks
    print("=== ±-Fe (BCC) XRD Pattern Simulation ===")
    print("2¸ [deg]  | Relative intensity")
    print("-" * 30)
    
    # Normalize intensities
    intensity_normalized = 100 * intensity / np.max(intensity)
    
    # Sort by 2theta
    sorted_indices = np.argsort(two_theta)
    for idx in sorted_indices[:10]:  # Top 10 peaks
        print(f"{two_theta[idx]:7.2f}  | {intensity_normalized[idx]:7.1f}")
    
    # Expected output: Typical peak positions for ±-Fe
    # (110): ~44.7°
    # (200): ~65.0°
    # (211): ~82.3°

## 1.5 Ewald Sphere and Visualization of Diffraction Conditions

### 1.5.1 Concept of the Ewald Sphere

The Ewald sphere is a powerful tool for visualizing the geometric conditions of X-ray diffraction. In reciprocal space, a sphere of radius \\( 1/\lambda \\) is drawn with the endpoint of the incident X-ray vector \\( \mathbf{k}_0 \\) at the origin.

The diffraction condition is satisfied when the Ewald sphere passes through a reciprocal lattice point:

\\[ \mathbf{k} - \mathbf{k}_0 = \mathbf{G}_{hkl} \\] 

Here, \\( \mathbf{k} \\) is the wave vector of the diffracted X-ray, and \\( \mathbf{G}_{hkl} \\) is the reciprocal lattice vector.
    
    
    ```mermaid
    graph TD
        A[Incident X-raywave vector k0] --> B[Ewald Sphereradius = 1/»]
        B --> C{Reciprocal lattice pointon Ewald sphere?}
        C -->|Yes| D[Diffraction condition satisfiedDiffraction occurs]
        C -->|No| E[No diffraction]
        D --> F[Diffracted X-raywave vector k]
        style B fill:#fce7f3
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### 1.5.2 Plotting the Ewald Sphere
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.patches import FancyArrowPatch
    
    def plot_ewald_sphere_2d(wavelength, a, hkl_max=3):
        """Draw 2D cross-section of Ewald sphere
    
        Args:
            wavelength (float): X-ray wavelength [Å]
            a (float): Lattice constant [Å] (cubic)
            hkl_max (int): Maximum Miller index for reciprocal lattice points to plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Radius of Ewald sphere
        k = 1.0 / wavelength  # [Å^-1]
    
        # Center of Ewald sphere (endpoint of incident X-ray)
        center = np.array([k, 0])
    
        # Draw Ewald sphere
        circle = Circle(center, k, fill=False, edgecolor='#f093fb', linewidth=2, label='Ewald Sphere')
        ax.add_patch(circle)
    
        # Draw reciprocal lattice points (cubic, 2D cross-section)
        a_star = 1.0 / a  # [Å^-1]
    
        reciprocal_points = []
        for h in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                G_x = h * a_star
                G_z = l * a_star
                reciprocal_points.append((G_x, G_z, h, l))
    
                # Plot reciprocal lattice point
                ax.plot(G_x, G_z, 'o', color='#2c3e50', markersize=4)
    
        # Incident X-ray vector
        ax.arrow(0, 0, k, 0, head_width=0.02, head_length=0.03,
                 fc='#f5576c', ec='#f5576c', linewidth=2, label='Incident X-ray k0')
    
        # Highlight reciprocal lattice points that intersect with Ewald sphere
        for G_x, G_z, h, l in reciprocal_points:
            dist_from_center = np.sqrt((G_x - center[0])**2 + (G_z - center[1])**2)
            if abs(dist_from_center - k) < 0.02:  # On the sphere
                ax.plot(G_x, G_z, 'o', color='#e74c3c', markersize=10,
                       label=f'Diffraction possible: ({h},0,{l})' if h==1 and l==0 else '')
    
                # Diffracted X-ray vector
                ax.arrow(0, 0, G_x, G_z, head_width=0.01, head_length=0.02,
                        fc='#e74c3c', ec='#e74c3c', linewidth=1.5,
                        linestyle='--', alpha=0.7)
    
        ax.set_xlabel('Reciprocal lattice h direction [Å$^{-1}$]', fontsize=12)
        ax.set_ylabel('Reciprocal lattice l direction [Å$^{-1}$]', fontsize=12)
        ax.set_title(f'Ewald Sphere and Reciprocal Lattice (» = {wavelength:.2f} Å, a = {a:.2f} Å)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
        # Clean up legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
        plt.tight_layout()
        plt.show()
    
    # For Cu K± radiation, lattice constant a = 3.5 Å
    plot_ewald_sphere_2d(wavelength=1.54056, a=3.5, hkl_max=2)

## Review of Learning Objectives

Having completed this chapter, you are now able to explain and implement the following:

### Basic Understanding

  *  Physical meaning and derivation of Bragg's law \\( 2d\sin\theta = n\lambda \\)
  *  Relationship between crystal lattice and reciprocal lattice, and calculation of plane spacing
  *  Relationship between systematic absences and crystal symmetry

### Practical Skills

  *  Calculate diffraction angles in Python and visualize the relationship with plane spacing
  *  Derive plane spacing for Miller indices from reciprocal lattice vectors
  *  Calculate structure factors and predict diffraction intensities
  *  Simulate XRD patterns and compare with experimental data

### Application Capability

  *  Geometrically understand diffraction conditions using the Ewald sphere
  *  Infer crystal lattice type from systematic absences
  *  Simulate XRD patterns for real materials (±-Fe, etc.)

## Exercises

### Easy (Basic Confirmation)

**Q1** : Calculate the Bragg angle (n=1) for a crystal plane with spacing d = 2.5 Å using Cu K± radiation (» = 1.54 Å).

**Solution** :
    
    
    theta = bragg_law(d=2.5, wavelength=1.54, n=1)
    print(f"Bragg angle: {theta:.2f} degrees")
    # Output: Bragg angle: 17.93 degrees

**Explanation** :

From Bragg's law \\( 2d\sin\theta = n\lambda \\):

\\[ \sin\theta = \frac{n\lambda}{2d} = \frac{1 \times 1.54}{2 \times 2.5} = 0.308 \\]

\\[ \theta = \arcsin(0.308) = 17.93° \\]

**Q2** : In a body-centered cubic (BCC) lattice, is the (1,1,0) reflection observed? Verify using systematic absences.

**Solution** :
    
    
    is_allowed = systematic_absences(1, 1, 0, 'I')
    print(f"(1,1,0) reflection: {'Observed' if is_allowed else 'Absent'}")
    # Output: (1,1,0) reflection: Observed

**Explanation** :

BCC (body-centered lattice, lattice_type='I') systematic absence: Only \\( h + k + l = even \\) are observed.

For (1,1,0): \\( 1 + 1 + 0 = 2 \\) (even) ’ Observed 

For (1,0,0): \\( 1 + 0 + 0 = 1 \\) (odd) ’ Absent 

### Medium (Application)

**Q3** : Calculate the plane spacing for (2,2,0) and (1,1,1) planes in a cubic crystal (a=4.0Å) and compare which has larger spacing.

**Solution** :
    
    
    a = 4.0
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([0, a, 0])
    c_vec = np.array([0, 0, a])
    a_star, b_star, c_star = reciprocal_lattice_vectors(a_vec, b_vec, c_vec)
    
    d_220 = d_spacing_from_hkl(2, 2, 0, a_star, b_star, c_star)
    d_111 = d_spacing_from_hkl(1, 1, 1, a_star, b_star, c_star)
    
    print(f"d_220 = {d_220:.4f} Å")
    print(f"d_111 = {d_111:.4f} Å")
    print(f"Larger spacing: {'(1,1,1)' if d_111 > d_220 else '(2,2,0)'}")
    
    # Expected output:
    # d_220 = 1.4142 Å
    # d_111 = 2.3094 Å
    # Larger spacing: (1,1,1)

**Explanation** :

Cubic crystal plane spacing formula: \\( d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}} \\)

(2,2,0): \\( d = 4.0 / \sqrt{4+4+0} = 4.0 / 2.83 = 1.414 \\) Å

(1,1,1): \\( d = 4.0 / \sqrt{1+1+1} = 4.0 / 1.732 = 2.309 \\) Å

Conclusion: (1,1,1) has larger plane spacing (smaller Miller indices correspond to wider spacing)

**Q4** : Compare the intensity of the (1,0,0) reflection for simple cubic (SC) and face-centered cubic (FCC) lattices using structure factors.

**Solution** :
    
    
    # SC: 1 atom at (0,0,0)
    atoms_sc = [(0, 0, 0)]
    f_sc = [1.0]
    F_sc = structure_factor(1, 0, 0, atoms_sc, f_sc)
    I_sc = intensity_from_structure_factor(F_sc)
    
    # FCC: 4 atoms at (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
    atoms_fcc = [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
    f_fcc = [1.0, 1.0, 1.0, 1.0]
    F_fcc = structure_factor(1, 0, 0, atoms_fcc, f_fcc)
    I_fcc = intensity_from_structure_factor(F_fcc)
    
    print(f"SC: F_100 = {F_sc:.4f}, I = {I_sc:.4f}")
    print(f"FCC: F_100 = {F_fcc:.4f}, I = {I_fcc:.4f}")
    
    # Expected output:
    # SC: F_100 = 1.0000, I = 1.0000
    # FCC: F_100 = 0.0000, I = 0.0000 (absent)

**Explanation** :

FCC structure factor for (1,0,0):

\\[ F = 1 \cdot e^{0} + 1 \cdot e^{i\pi} + 1 \cdot e^{i\pi} + 1 \cdot e^{0} = 1 + (-1) + (-1) + 1 = 0 \\]

In FCC, mixed-index reflections ((1,0,0), etc.) are systematically absent.

### Hard (Advanced)

**Q5** : Simulate XRD patterns for ±-Fe (BCC, a=2.87Å) and ³-Fe (FCC, a=3.65Å), and compare the position (2¸) of the strongest peak. Use Cu K± radiation.

**Solution** :
    
    
    # ±-Fe (BCC)
    two_theta_bcc, intensity_bcc = calculate_xrd_pattern(
        a=2.87, lattice_type='I',
        atom_positions=[(0,0,0), (0.5,0.5,0.5)],
        scattering_factors=[26.0, 26.0]
    )
    max_idx_bcc = np.argmax(intensity_bcc)
    print(f"±-Fe (BCC) strongest peak: 2¸ = {two_theta_bcc[max_idx_bcc]:.2f}°")
    
    # ³-Fe (FCC)
    two_theta_fcc, intensity_fcc = calculate_xrd_pattern(
        a=3.65, lattice_type='F',
        atom_positions=[(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
        scattering_factors=[26.0]*4
    )
    max_idx_fcc = np.argmax(intensity_fcc)
    print(f"³-Fe (FCC) strongest peak: 2¸ = {two_theta_fcc[max_idx_fcc]:.2f}°")
    
    # Expected output:
    # ±-Fe (BCC) strongest peak: 2¸ H 44.7° (110 reflection)
    # ³-Fe (FCC) strongest peak: 2¸ H 43.6° (111 reflection)

**Explanation** :

±-Fe (BCC): Strongest peak is (110) reflection (h+k+l=2, minimum even number)

³-Fe (FCC): Strongest peak is (111) reflection (all odd, minimum)

This difference allows clear distinction between BCC and FCC by XRD. This is applied in phase transformation analysis of actual steel materials.

**Q6** : Using the Ewald sphere concept, explain why not all reciprocal lattice points diffract simultaneously with monochromatic X-rays.

**Solution** :

**Reason** :

  1. The radius of the Ewald sphere is fixed (= 1/»)
  2. The diffraction condition is "reciprocal lattice point is on the Ewald sphere"
  3. With monochromatic X-rays, the Ewald sphere is at a fixed position unless the sample is rotated
  4. Most reciprocal lattice points are inside or outside the sphere, not on the surface

**Experimental Solutions** :

  * **Powder XRD** : Randomly oriented microcrystals allow observation of diffraction from all orientations
  * **Single crystal rotation method** : Rotate the sample to sequentially pass reciprocal lattice points through the Ewald sphere
  * **White X-ray (Laue method)** : Various wavelengths ’ various Ewald sphere radii ’ simultaneous observation of many reflections

## Confirmation of Learning Objectives

Upon completing this chapter, you can explain and implement the following:

### Basic Understanding

  *  Derive Bragg's law (n» = 2d sin¸) and explain its physical meaning
  *  Understand the relationship between crystal lattice and reciprocal lattice, and calculate reciprocal lattice vectors
  *  Explain the physical mechanism of systematic absences (destructive interference)
  *  Understand the calculation method and physical meaning of structure factor F(hkl)

### Practical Skills

  *  Calculate d-spacing in Python and predict diffraction angles
  *  Apply systematic absences to any space group and determine allowed reflections
  *  Calculate structure factors from atomic coordinates and predict diffraction intensities
  *  Visualize diffraction conditions using the Ewald sphere concept

### Application Capability

  *  Infer crystal structure (BCC/FCC, etc.) from experimental XRD patterns
  *  Predict the effect of lattice constant changes on XRD patterns
  *  Use systematic absences to narrow down space group candidates
  *  Plan XRD analysis strategies for polycrystalline materials

## References

  1. Cullity, B. D., & Stock, S. R. (2001). _Elements of X-Ray Diffraction_ (3rd ed.). Prentice Hall. - Classic textbook on X-ray diffraction, comprehensive coverage from Bragg's law to crystal structure analysis
  2. Warren, B. E. (1990). _X-ray Diffraction_. Dover Publications. - Detailed explanation of the physical foundations of diffraction theory, excellent derivation of structure factors
  3. Pecharsky, V. K., & Zavalij, P. Y. (2009). _Fundamentals of Powder Diffraction and Structural Characterization of Materials_ (2nd ed.). Springer. - Practical textbook specialized in powder XRD
  4. International Tables for Crystallography, Volume A: Space-Group Symmetry (2016). International Union of Crystallography. - Definitive reference for space groups and systematic absences
  5. Giacovazzo, C., et al. (2011). _Fundamentals of Crystallography_ (3rd ed.). Oxford University Press. - Theoretical foundations of crystallography, clear explanation of reciprocal lattice concepts
  6. Hammond, C. (2015). _The Basics of Crystallography and Diffraction_ (4th ed.). Oxford University Press. - Textbook with excellent visual understanding of Ewald sphere construction
  7. Ladd, M., & Palmer, R. (2013). _Structure Determination by X-ray Crystallography_ (5th ed.). Springer. - Practical guide from structure factor calculation to crystal structure determination

## Next Steps

In Chapter 1, we learned the theoretical foundations of X-ray diffraction. Concepts such as Bragg's law, reciprocal lattice, systematic absences, and structure factors form the basis of all XRD analysis.

**Chapter 2** will apply these theories to actual XRD measurements. We will learn practical analysis techniques such as powder X-ray diffraction data acquisition, peak identification, background removal, and peak fitting.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
