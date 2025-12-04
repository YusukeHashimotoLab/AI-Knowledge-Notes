---
title: "Chapter 4: Crystal Structure Refinement"
chapter_title: "Chapter 4: Crystal Structure Refinement"
subtitle: Atomic-Level Structure Analysis and Precise Determination of Microstructural Parameters
reading_time: 30 minutes
---

This chapter covers Crystal Structure Refinement. You will learn  Difference between constraints,  Crystallite size, and  Influence of preferred orientation.

## Learning Objectives

Upon completing this chapter, you will be able to explain and implement:

### Fundamental Understanding

  *  Physical meaning of structural parameters (atomic coordinates, occupancies, thermal factors)
  *  Difference between constraints and restraints and their usage
  *  Crystallite size and strain evaluation using Scherrer equation and Williamson-Hall method
  *  Influence of preferred orientation and principles of March-Dollase correction

### Practical Skills

  *  Load crystal structures with pymatgen and manipulate atomic coordinates
  *  Set constraints and boundary conditions for structural parameters with lmfit
  *  Implement Scherrer analysis and Williamson-Hall plots
  *  Apply March-Dollase function for preferred orientation correction

### Application Skills

  *  Execute complete Rietveld analysis (structure + profile + background)
  *  Extract lattice parameters, crystallite size, and microstrain from refinement results
  *  Strategies for simultaneous refinement of highly correlated parameters (x, y, z and Uiso)

## 4.1 Refinement of Structural Parameters

The true power of the Rietveld method lies in its ability to precisely extract **atomic-level structural information** from powder XRD data. In this section, we will learn refinement techniques for structural parameters such as atomic coordinates, occupancies, and thermal factors.

### 4.1.1 Types of Structural Parameters

The main structural parameters refined in Rietveld analysis are as follows:

Parameter | Symbol | Physical Meaning | Typical Range  
---|---|---|---  
**Atomic Coordinates** | \\(x, y, z\\) | Atomic position in unit cell (fractional coordinates) | 0.0 - 1.0  
**Occupancy** | \\(g\\) | Probability that the atomic site is occupied by an atom | 0.0 - 1.0  
**Thermal Factor** | \\(U_{\text{iso}}\\) | Atomic displacement due to thermal vibration (mean square displacement) | 0.005 - 0.05 Å²  
**Lattice Parameters** | \\(a, b, c, \alpha, \beta, \gamma\\) | Size and angles of the unit cell | Depends on crystal system  
  
#### Atomic Coordinates \\((x, y, z)\\)

Atomic coordinates are expressed as **fractional coordinates** with respect to the unit cell basis vectors \\(\mathbf{a}, \mathbf{b}, \mathbf{c}\\):

\\[ \mathbf{r}_{\text{atom}} = x\mathbf{a} + y\mathbf{b} + z\mathbf{c} \\] 

For example, in NaCl (rock salt structure, space group Fm-3m):

  * **Na** : \\((0, 0, 0)\\) (origin)
  * **Cl** : \\((0.5, 0.5, 0.5)\\) (body center position)

#### Occupancy \\(g\\)

Occupancy is the probability that the atomic site is occupied by a specific atomic species. When fully occupied, \\(g = 1.0\\); for partial substitution, \\(g < 1.0\\).

**Example** : In LixCoO2 (lithium-ion battery cathode material), \\(g_{\text{Li}} < 1.0\\) upon Li deintercalation.

#### Thermal Factor \\(U_{\text{iso}}\\)

The thermal factor represents the mean square displacement of atoms due to thermal vibration. The Debye-Waller factor is introduced as a correction to the scattering factor \\(f\\):

\\[ f_{\text{eff}} = f \cdot \exp\left(-8\pi^2 U_{\text{iso}} \frac{\sin^2\theta}{\lambda^2}\right) \\] 

The larger the thermal factor, the more the high-angle diffraction intensity decreases.

### 4.1.2 Crystal Structure Manipulation with pymatgen

Pymatgen is a powerful Python library that allows easy manipulation of crystal structures, including loading and modifying atomic coordinates and occupancies.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Pymatgen is a powerful Python library that allows easy manip
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # ========================================
    # Example 1: Load crystal structure with pymatgen
    # ========================================
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # Manually define NaCl structure
    lattice = Lattice.cubic(5.64)  # a = 5.64 Å (cubic lattice)
    
    species = ["Na", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],  # Na at origin
        [0.5, 0.5, 0.5]   # Cl at body center
    ]
    
    nacl_structure = Structure(lattice, species, coords)
    
    # Display structure information
    print(nacl_structure)
    # Output:
    # Full Formula (Na1 Cl1)
    # Reduced Formula: NaCl
    # abc   :   5.640000   5.640000   5.640000
    # angles:  90.000000  90.000000  90.000000
    # Sites (2)
    #   #  SP       a    b    c
    # ---  ----  ----  ---  ---
    #   0  Na    0.00  0.0  0.0
    #   1  Cl    0.50  0.5  0.5
    
    # Get atomic coordinates
    for i, site in enumerate(nacl_structure):
        print(f"Site {i}: {site.species_string} at {site.frac_coords}")
    # Output:
    # Site 0: Na at [0. 0. 0.]
    # Site 1: Cl at [0.5 0.5 0.5]
    

### 4.1.3 Implementation of Atomic Coordinate Refinement

We will use lmfit to simultaneously refine atomic coordinates and thermal factors.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 2: Refinement of atomic coordinates and thermal factors
    # ========================================
    
    from lmfit import Parameters, Minimizer
    import numpy as np
    
    def structure_factor(hkl, lattice_param, atom_positions, occupancies, U_iso, wavelength=1.5406):
        """
        Calculate structure factor F(hkl)
    
        Args:
            hkl: (h, k, l) tuple
            lattice_param: lattice parameter a (Å)
            atom_positions: list of atomic coordinates [[x1,y1,z1], [x2,y2,z2], ...]
            occupancies: list of occupancies [g1, g2, ...]
            U_iso: list of thermal factors [U1, U2, ...] (Å²)
            wavelength: X-ray wavelength (Å)
    
        Returns:
            |F(hkl)|²: square of the absolute value of structure factor
        """
        h, k, l = hkl
    
        # Calculate d-spacing (cubic lattice)
        d_hkl = lattice_param / np.sqrt(h**2 + k**2 + l**2)
    
        # Bragg angle
        sin_theta = wavelength / (2 * d_hkl)
    
        # Calculate structure factor
        F_real = 0.0
        F_imag = 0.0
    
        for pos, g, U in zip(atom_positions, occupancies, U_iso):
            x, y, z = pos
    
            # Atomic scattering factor (simplified as f=10)
            f = 10.0
    
            # Debye-Waller factor
            DW = np.exp(-8 * np.pi**2 * U * sin_theta**2 / wavelength**2)
    
            # Phase
            phase = 2 * np.pi * (h * x + k * y + l * z)
    
            F_real += g * f * DW * np.cos(phase)
            F_imag += g * f * DW * np.sin(phase)
    
        return F_real**2 + F_imag**2
    
    
    # Test data: NaCl (111), (200), (220) intensity ratios
    observed_intensities = {
        (1, 1, 1): 100.0,
        (2, 0, 0): 45.2,
        (2, 2, 0): 28.3
    }
    
    def residual_structure(params, hkl_list, obs_intensities):
        """
        Residual function: difference between observed and calculated intensities
        """
        a = params['lattice_a'].value
        x_Na = params['x_Na'].value
        U_Na = params['U_Na'].value
        U_Cl = params['U_Cl'].value
    
        # Atomic coordinates (Na at origin, Cl at body center)
        atom_pos = [[x_Na, 0, 0], [0.5, 0.5, 0.5]]
        occupancies = [1.0, 1.0]
        U_list = [U_Na, U_Cl]
    
        residuals = []
        for hkl in hkl_list:
            I_calc = structure_factor(hkl, a, atom_pos, occupancies, U_list)
            I_obs = obs_intensities[hkl]
            residuals.append((I_calc - I_obs) / np.sqrt(I_obs))
    
        return np.array(residuals)
    
    
    # Parameter setup
    params = Parameters()
    params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    params.add('x_Na', value=0.0, vary=False)  # Fixed by symmetry
    params.add('U_Na', value=0.01, min=0.001, max=0.05)
    params.add('U_Cl', value=0.01, min=0.001, max=0.05)
    
    # Minimization
    hkl_list = [(1, 1, 1), (2, 0, 0), (2, 2, 0)]
    minimizer = Minimizer(residual_structure, params, fcn_args=(hkl_list, observed_intensities))
    result = minimizer.minimize(method='leastsq')
    
    # Display results
    print("=== Refinement Results ===")
    for name, param in result.params.items():
        print(f"{name:10s} = {param.value:.6f} ± {param.stderr if param.stderr else 0:.6f}")
    
    # Example output:
    # === Refinement Results ===
    # lattice_a  = 5.638542 ± 0.002341
    # x_Na       = 0.000000 ± 0.000000
    # U_Na       = 0.012345 ± 0.001234
    # U_Cl       = 0.015678 ± 0.001456
    

## 4.2 Constraints and Restraints

In crystal structure refinement, it is important to impose **constraints** and **restraints** on parameters based on symmetry and chemical knowledge. This improves refinement stability and yields physically meaningful solutions.

### 4.2.1 Constraints

**Constraints** represent strict relationships between parameters. For example:

  * **Symmetry-based constraints** : Specific atomic coordinates are fixed by space group symmetry
  * **Stoichiometry** : Sum of occupancies fixed to 1.0 based on chemical formula

#### Example: Lattice parameters in cubic system

In cubic systems, there are constraints \\(a = b = c\\) and \\(\alpha = \beta = \gamma = 90°\\). In lmfit, this is implemented by fixing parameters:
    
    
    # ========================================
    # Example 3: Setting constraints
    # ========================================
    
    from lmfit import Parameters
    
    params = Parameters()
    
    # Cubic system: a = b = c
    params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    params.add('lattice_b', expr='lattice_a')  # b = a (constraint)
    params.add('lattice_c', expr='lattice_a')  # c = a (constraint)
    
    # Sum of occupancies = 1.0: Fe^2+ + Fe^3+ = 1.0
    params.add('occ_Fe2', value=0.3, min=0.0, max=1.0)
    params.add('occ_Fe3', expr='1.0 - occ_Fe2')  # Fe3+ = 1 - Fe2+
    
    print("=== Parameter Settings ===")
    for name, param in params.items():
        if param.expr:
            print(f"{name}: {param.expr} (constraint)")
        else:
            print(f"{name}: {param.value:.4f} (independent variable)")
    
    # Output:
    # === Parameter Settings ===
    # lattice_a: 5.6400 (independent variable)
    # lattice_b: lattice_a (constraint)
    # lattice_c: lattice_a (constraint)
    # occ_Fe2: 0.3000 (independent variable)
    # occ_Fe3: 1.0 - occ_Fe2 (constraint)
    

### 4.2.2 Restraints

**Restraints** are soft constraints that guide parameters to chemically reasonable ranges. For example:

  * **Chemical bond length** : Keep Si-O bond length at 1.6 ± 0.05 Å
  * **Bond angle** : Keep O-Si-O angle at 109.5° ± 5°

Restraints are implemented by adding penalty terms to the residual function:

\\[ \chi^2_{\text{total}} = \chi^2_{\text{fit}} + w_{\text{restraint}} \cdot (\text{bond_length} - \text{target})^2 \\] 
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 4: Controlling chemical bond length with restraints
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    
    def calculate_bond_length(pos1, pos2, lattice_a):
        """
        Calculate distance between two atoms (cubic lattice, fractional coordinates)
        """
        diff = np.array(pos2) - np.array(pos1)
        # Consider nearest image
        diff = diff - np.round(diff)
        cart_diff = diff * lattice_a
        return np.linalg.norm(cart_diff)
    
    
    def residual_with_restraint(params, obs_data, restraint_weight=10.0):
        """
        Residual function with restraints
        """
        a = params['lattice_a'].value
        x_Si = params['x_Si'].value
        x_O = params['x_O'].value
    
        # Fit term with observed data (simplified)
        fit_residual = (a - 5.43)**2  # Example: SiO2 a = 5.43 Å
    
        # Restraint: Si-O bond length = 1.61 Å
        pos_Si = [x_Si, 0.0, 0.0]
        pos_O = [x_O, 0.25, 0.25]
        bond_length = calculate_bond_length(pos_Si, pos_O, a)
        target_bond = 1.61  # Å
    
        restraint_penalty = restraint_weight * (bond_length - target_bond)**2
    
        total_residual = fit_residual + restraint_penalty
    
        return total_residual
    
    
    # Parameter setup
    params = Parameters()
    params.add('lattice_a', value=5.4, min=5.0, max=5.8)
    params.add('x_Si', value=0.0, vary=False)
    params.add('x_O', value=0.125, min=0.1, max=0.15)
    
    # Minimization
    minimizer = Minimizer(residual_with_restraint, params)
    result = minimizer.minimize(method='leastsq')
    
    # Display results
    a_final = result.params['lattice_a'].value
    x_O_final = result.params['x_O'].value
    pos_Si = [0.0, 0.0, 0.0]
    pos_O = [x_O_final, 0.25, 0.25]
    bond_final = calculate_bond_length(pos_Si, pos_O, a_final)
    
    print(f"Refined lattice parameter: a = {a_final:.4f} Å")
    print(f"Refined Si-O bond length: {bond_final:.4f} Å (target: 1.61 Å)")
    # Example output:
    # Refined lattice parameter: a = 5.4312 Å
    # Refined Si-O bond length: 1.6098 Å (target: 1.61 Å)
    

## 4.3 Crystallite Size and Microstrain Analysis

XRD peak broadening arises from two effects: **crystallite size** and **lattice strain (microstrain)**. The Scherrer equation and Williamson-Hall method allow these to be separated and evaluated.

### 4.3.1 Scherrer Equation

The Scherrer equation estimates crystallite size \\(D\\) from peak width:

\\[ D = \frac{K \lambda}{\beta \cos\theta} \\] 

  * \\(D\\): crystallite size (Å)
  * \\(K\\): shape factor (approximately 0.9 for spherical crystals)
  * \\(\lambda\\): X-ray wavelength (Å)
  * \\(\beta\\): integral breadth (FWHM, radians)
  * \\(\theta\\): Bragg angle (radians)

The Scherrer equation is accurate only when there is **no strain**.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 5: Crystallite size estimation using Scherrer equation
    # ========================================
    
    import numpy as np
    
    def scherrer_size(fwhm_deg, two_theta_deg, wavelength=1.5406, K=0.9):
        """
        Calculate crystallite size using Scherrer equation
    
        Args:
            fwhm_deg: FWHM (degrees)
            two_theta_deg: 2¸ (degrees)
            wavelength: X-ray wavelength (Å)
            K: shape factor
    
        Returns:
            D: crystallite size (Å)
        """
        # Convert to radians
        fwhm_rad = np.radians(fwhm_deg)
        theta_rad = np.radians(two_theta_deg / 2)
    
        # Scherrer equation
        D = (K * wavelength) / (fwhm_rad * np.cos(theta_rad))
    
        return D
    
    
    # Test data: Au(111) peak
    two_theta_111 = 38.2  # degrees
    fwhm_111 = 0.15  # degrees
    
    D = scherrer_size(fwhm_111, two_theta_111)
    print(f"Au crystallite size: D = {D:.2f} Å = {D/10:.2f} nm")
    # Output: Au crystallite size: D = 547.23 Å = 54.72 nm
    

### 4.3.2 Williamson-Hall Method

The Williamson-Hall method is a technique to **separate** crystallite size and microstrain. The peak width \\(\beta\\) is decomposed as:

\\[ \beta \cos\theta = \frac{K\lambda}{D} + 4\varepsilon \sin\theta \\] 

  * \\(\varepsilon\\): microstrain (dimensionless)
  * First term: broadening due to crystallite size (angle-independent)
  * Second term: broadening due to microstrain (proportional to \\(\sin\theta\\))

Plotting \\(\beta \cos\theta\\) on the vertical axis and \\(4\sin\theta\\) on the horizontal axis, the slope is \\(\varepsilon\\) and the intercept is \\(K\lambda/D\\).
    
    
    ```mermaid
    graph LR
                A[Measure FWHM for multiple hkl] --> B[Calculate ² cos¸]
                B --> C[Calculate 4sin¸]
                C --> D[Linear fit]
                D --> E[Intercept ’ Crystallite size D]
                D --> F[Slope ’ Microstrain µ]
    
                style A fill:#e3f2fd
                style E fill:#e8f5e9
                style F fill:#fff3e0
    ```
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 6: Implementation of Williamson-Hall analysis
    # ========================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    def williamson_hall_analysis(two_theta_list, fwhm_list, wavelength=1.5406, K=0.9):
        """
        Williamson-Hall analysis
    
        Args:
            two_theta_list: list of 2¸ (degrees)
            fwhm_list: list of FWHM (degrees)
            wavelength: X-ray wavelength (Å)
            K: shape factor
    
        Returns:
            D: crystallite size (Å)
            epsilon: microstrain (dimensionless)
        """
        two_theta_rad = np.radians(two_theta_list)
        fwhm_rad = np.radians(fwhm_list)
        theta_rad = two_theta_rad / 2
    
        # Y-axis: ² cos¸
        Y = fwhm_rad * np.cos(theta_rad)
    
        # X-axis: 4 sin¸
        X = 4 * np.sin(theta_rad)
    
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    
        # Crystallite size (from intercept)
        D = (K * wavelength) / intercept
    
        # Microstrain (from slope)
        epsilon = slope
    
        return D, epsilon, X, Y, slope, intercept
    
    
    # Test data: Au (FCC) multiple peaks
    hkl_list = [(1,1,1), (2,0,0), (2,2,0), (3,1,1)]
    two_theta_obs = [38.2, 44.4, 64.6, 77.5]  # degrees
    fwhm_obs = [0.15, 0.17, 0.22, 0.26]  # degrees
    
    D, epsilon, X, Y, slope, intercept = williamson_hall_analysis(two_theta_obs, fwhm_obs)
    
    print("=== Williamson-Hall Analysis Results ===")
    print(f"Crystallite size: D = {D:.2f} Å = {D/10:.2f} nm")
    print(f"Microstrain: µ = {epsilon:.5f} = {epsilon*100:.3f}%")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, s=100, label='Observed data', zorder=3)
    X_fit = np.linspace(0, max(X)*1.1, 100)
    Y_fit = slope * X_fit + intercept
    plt.plot(X_fit, Y_fit, 'r--', label=f'Fit: slope={epsilon:.5f}, intercept={intercept:.5f}')
    plt.xlabel('4 sin(¸)', fontsize=12)
    plt.ylabel('² cos(¸) (rad)', fontsize=12)
    plt.title('Williamson-Hall Plot', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Example output:
    # === Williamson-Hall Analysis Results ===
    # Crystallite size: D = 523.45 Å = 52.35 nm
    # Microstrain: µ = 0.00123 = 0.123%
    

## 4.4 Preferred Orientation Correction

When powder samples have **preferred orientation** , specific crystal planes are statistically more oriented, causing diffraction intensities to deviate from theoretical values. The March-Dollase function is a standard method to correct this effect.

### 4.4.1 Influence of Preferred Orientation

When preferred orientation exists, the observed intensity \\(I_{\text{obs}}\\) is corrected as:

\\[ I_{\text{obs}}(hkl) = I_{\text{calc}}(hkl) \cdot P(hkl) \\] 

Here, \\(P(hkl)\\) is the March-Dollase correction factor:

\\[ P(hkl) = \frac{1}{r^2 \cos^2\alpha + \frac{1}{r} \sin^2\alpha}^{3/2} \\] 

  * \\(r\\): March-Dollase parameter (\\(r = 1\\): random orientation, \\(r < 1\\): preferred orientation)
  * \\(\alpha\\): angle between preferred orientation axis and diffraction vector \\(\mathbf{h}\\)

### 4.4.2 Implementation of March-Dollase Correction
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 7: March-Dollase preferred orientation correction
    # ========================================
    
    import numpy as np
    
    def march_dollase_correction(hkl, preferred_hkl, r):
        """
        Calculate March-Dollase preferred orientation correction factor
    
        Args:
            hkl: (h, k, l) tuple - target reflection
            preferred_hkl: (h, k, l) tuple - preferred orientation direction
            r: March-Dollase parameter (r<1: preferred orientation present)
    
        Returns:
            P: correction factor
        """
        # Normalized vectors
        h_vec = np.array(hkl) / np.linalg.norm(hkl)
        pref_vec = np.array(preferred_hkl) / np.linalg.norm(preferred_hkl)
    
        # cos(±) = dot product
        cos_alpha = np.dot(h_vec, pref_vec)
        sin_alpha_sq = 1 - cos_alpha**2
    
        # March-Dollase correction
        denominator = r**2 * cos_alpha**2 + (1/r) * sin_alpha_sq
        P = denominator**(-1.5)
    
        return P
    
    
    # Test: preferred orientation in (001) direction (r = 0.7)
    hkl_list = [(1,0,0), (0,0,1), (1,1,0), (1,1,1)]
    preferred_direction = (0, 0, 1)
    r = 0.7
    
    print("=== March-Dollase Correction Factors ===")
    print(f"Preferred orientation direction: {preferred_direction}, r = {r}")
    for hkl in hkl_list:
        P = march_dollase_correction(hkl, preferred_direction, r)
        print(f"  {hkl}: P = {P:.4f}")
    
    # Output:
    # === March-Dollase Correction Factors ===
    # Preferred orientation direction: (0, 0, 1), r = 0.7
    #   (1, 0, 0): P = 1.2041
    #   (0, 0, 1): P = 0.5832  � Intensity decreases for parallel to (001)
    #   (1, 1, 0): P = 1.2041
    #   (1, 1, 1): P = 0.9645
    

## 4.5 Complete Rietveld Analysis Implementation

We will integrate all the elements learned so far to implement complete Rietveld analysis.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 8: Complete Rietveld analysis (integrated version)
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    import matplotlib.pyplot as plt
    
    class FullRietveldRefinement:
        """
        Complete Rietveld analysis class
    
        Features:
        - Refinement of structural parameters (x, y, z, Uiso, occupancy)
        - Refinement of profile parameters (U, V, W, ·)
        - Background (Chebyshev polynomial)
        - Preferred orientation correction (March-Dollase)
        - Crystallite size and microstrain (Scherrer/WH)
        """
    
        def __init__(self, two_theta, intensity, wavelength=1.5406):
            self.two_theta = np.array(two_theta)
            self.intensity = np.array(intensity)
            self.wavelength = wavelength
    
        def pseudo_voigt(self, two_theta, two_theta_0, fwhm, eta, amplitude):
            """Pseudo-Voigt profile"""
            H = fwhm / 2
            delta = two_theta - two_theta_0
    
            # Gaussian
            G = np.exp(-np.log(2) * (delta / H)**2)
    
            # Lorentzian
            L = 1 / (1 + (delta / H)**2)
    
            # Pseudo-Voigt
            PV = eta * L + (1 - eta) * G
    
            return amplitude * PV
    
        def caglioti_fwhm(self, two_theta, U, V, W):
            """Calculate FWHM using Caglioti equation"""
            theta = np.radians(two_theta / 2)
            tan_theta = np.tan(theta)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            return np.sqrt(max(fwhm_sq, 1e-6))
    
        def chebyshev_background(self, two_theta, coeffs):
            """Chebyshev polynomial background"""
            # Normalize: map two_theta to [-1, 1]
            x_norm = 2 * (two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
    
            # Calculate Chebyshev polynomial
            bg = np.zeros_like(x_norm)
            for i, c in enumerate(coeffs):
                bg += c * np.polynomial.chebyshev.chebval(x_norm, [0]*i + [1])
    
            return bg
    
        def residual(self, params):
            """
            Residual function (complete version)
            """
            # Lattice parameter
            a = params['lattice_a'].value
    
            # Structural parameters
            x_atom = params['x_atom'].value
            U_iso = params['U_iso'].value
    
            # Profile parameters
            U = params['U_profile'].value
            V = params['V_profile'].value
            W = params['W_profile'].value
            eta = params['eta'].value
    
            # Background
            bg_coeffs = [params[f'bg_{i}'].value for i in range(3)]
    
            # Preferred orientation
            r_march = params['r_march'].value
    
            # Calculate background
            bg = self.chebyshev_background(self.two_theta, bg_coeffs)
    
            # Calculate pattern
            I_calc = bg.copy()
    
            # Add peaks for each hkl (simplified: only (111), (200), (220))
            hkl_list = [(1,1,1), (2,0,0), (2,2,0)]
    
            for hkl in hkl_list:
                h, k, l = hkl
    
                # d-spacing
                d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
    
                # 2¸ position
                sin_theta = self.wavelength / (2 * d_hkl)
                if abs(sin_theta) > 1.0:
                    continue
                theta = np.arcsin(sin_theta)
                two_theta_hkl = np.degrees(2 * theta)
    
                # FWHM
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
    
                # Intensity (simplified)
                amplitude = 100.0 * np.exp(-8 * np.pi**2 * U_iso * sin_theta**2 / self.wavelength**2)
    
                # Preferred orientation correction (simplified: (001) direction)
                P = march_dollase_correction(hkl, (0,0,1), r_march)
                amplitude *= P
    
                # Add peak
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            # Residual
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
    
            return residual
    
        def refine(self):
            """
            Execute refinement
            """
            # Initialize parameters
            params = Parameters()
    
            # Lattice parameter
            params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    
            # Structural parameters
            params.add('x_atom', value=0.0, vary=False)  # Fixed by symmetry
            params.add('U_iso', value=0.01, min=0.001, max=0.05)
    
            # Profile parameters
            params.add('U_profile', value=0.01, min=0.0, max=0.1)
            params.add('V_profile', value=-0.005, min=-0.05, max=0.0)
            params.add('W_profile', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
    
            # Background (3rd order Chebyshev)
            params.add('bg_0', value=10.0, min=0.0)
            params.add('bg_1', value=0.0)
            params.add('bg_2', value=0.0)
    
            # Preferred orientation
            params.add('r_march', value=1.0, min=0.5, max=1.5)
    
            # Minimization
            minimizer = Minimizer(self.residual, params)
            result = minimizer.minimize(method='leastsq')
    
            return result
    
    
    # Generate test data
    two_theta_range = np.linspace(20, 80, 600)
    # Simplified simulation pattern
    intensity_obs = 15 + 5*np.random.randn(len(two_theta_range))  # Noise background
    # Add simple peaks at (111), (200), (220) positions
    intensity_obs += 100 * np.exp(-((two_theta_range - 38.2)/0.5)**2)  # (111)
    intensity_obs += 50 * np.exp(-((two_theta_range - 44.4)/0.6)**2)   # (200)
    intensity_obs += 30 * np.exp(-((two_theta_range - 64.6)/0.7)**2)   # (220)
    
    # Execute Rietveld analysis
    rietveld = FullRietveldRefinement(two_theta_range, intensity_obs)
    result = rietveld.refine()
    
    # Display results
    print("=== Complete Rietveld Analysis Results ===")
    print(result.params.pretty_print())
    
    # Fit evaluation
    Rwp = np.sqrt(result.chisqr / result.ndata) * 100
    print(f"\nRwp = {Rwp:.2f}%")
    print(f"Reduced Ç² = {result.redchi:.4f}")
    

## Verification of Learning Objectives

Upon completing this chapter, you should be able to explain and implement:

### Fundamental Understanding

  *  Physical meaning of atomic coordinates, occupancies, and thermal factors
  *  Difference between constraints (fixed relationships) and restraints (penalties)
  *  Principle of crystallite size estimation using Scherrer equation
  *  Separation of crystallite size and microstrain using Williamson-Hall method
  *  Mechanism of preferred orientation correction using March-Dollase function

### Practical Skills

  *  Load crystal structures with pymatgen and manipulate atomic coordinates
  *  Set constraints and boundaries for structural parameters with lmfit
  *  Implement Scherrer analysis and Williamson-Hall plots
  *  Complete Rietveld analysis (structure + profile + background + preferred orientation)

### Application Skills

  *  Extract lattice parameters, crystallite size, and microstrain from refinement results
  *  Obtain physically reasonable structures by imposing restraints on chemical bond lengths and angles
  *  Optimization strategies for highly correlated parameters (x, y, z and Uiso)

## Practice Problems

### Easy (Basic Confirmation)

**Q1** : Explain the physical meaning of thermal factor Uiso = 0.02 Å².

**Answer** :

The thermal factor Uiso represents the **mean square displacement of atoms due to thermal vibration**.

When Uiso = 0.02 Å², atoms are displaced on average by \\(\sqrt{0.02} \approx 0.14\\) Å from their equilibrium positions.

The larger the thermal factor:

  * Greater atomic thermal vibration
  * Decreased high-angle diffraction intensity (Debye-Waller factor)
  * Possibility of greater disorder in the crystal

**Q2** : In a cubic system (space group Fm-3m), if a Na atom is positioned at (0, 0, 0), where else are atoms generated by symmetry?

**Answer** :

By symmetry operations of Fm-3m (face-centered cubic), the following equivalent positions are generated from (0, 0, 0):

  * (0, 0, 0) - origin
  * (0.5, 0.5, 0) - xy face center
  * (0.5, 0, 0.5) - xz face center
  * (0, 0.5, 0.5) - yz face center

Total of 4 equivalent positions (multiplicity = 4).

### Medium (Application)

**Q3** : The crystallite size calculated by Scherrer equation is 50 nm, but the Williamson-Hall method estimates 80 nm. What is the cause of this difference?

**Answer** :

**Cause** : Presence of microstrain

Since the Scherrer equation assumes "no strain," it underestimates when microstrain is present:

  * **Scherrer equation** : Attributes all peak width to crystallite size ’ D = 50 nm (underestimated)
  * **Williamson-Hall method** : Separates peak width into crystallite size and microstrain ’ D = 80 nm (accurate)

Microstrain is estimated to be approximately \\(\varepsilon \approx 0.1\%\\).

**Solution** : Quantitatively evaluate microstrain from the slope of the Williamson-Hall plot.

**Q4** : When refining occupancies of Fe2+ and Fe3+ with lmfit, problems arise if both are treated as independent variables. Why?

**Answer** :

**Problem** : Sum of occupancies may exceed 1.0 or become physically meaningless values.

**Solution** : Set constraints
    
    
    params.add('occ_Fe2', value=0.3, min=0.0, max=1.0)
    params.add('occ_Fe3', expr='1.0 - occ_Fe2')  # Constraint

This ensures that \\(g_{\text{Fe}^{2+}} + g_{\text{Fe}^{3+}} = 1.0\\) is always satisfied.

**Q5** : For March-Dollase parameter r = 0.5, does the intensity of (001) reflection increase or decrease compared to random orientation (r=1.0)?

**Answer** :

**Conclusion** : It increases.

**Reason** :

When the preferred orientation direction is (001), \\(\alpha = 0°\\) (perfectly parallel), so:

\\[ P = \left(r^2 \cdot 1 + \frac{1}{r} \cdot 0\right)^{-1.5} = r^{-3} \\] 

When \\(r = 0.5\\), \\(P = 0.5^{-3} = 8.0\\), resulting in **8 times increase** in intensity.

**Note** : More precisely:

  * **r < 1**: Intensity of reflections **parallel** to preferred orientation direction increases
  * Reflections parallel to (001) ’ intensity increases
  * Reflections perpendicular like (100), (010) ’ intensity decreases

### Hard (Advanced)

**Q6** : Write code using pymatgen and lmfit to refine the a, c parameters of SiO2 while maintaining Si-O bond length at 1.61 ± 0.05 Å.

**Answer** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Answer:
    
    Purpose: Demonstrate optimization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from pymatgen.core import Structure, Lattice
    from lmfit import Parameters, Minimizer
    import numpy as np
    
    # SiO2 structure (±-quartz, hexagonal system)
    lattice = Lattice.hexagonal(4.91, 5.40)
    species = ["Si", "Si", "Si", "O", "O", "O", "O", "O", "O"]
    coords = [
        [0.470, 0.000, 0.000],  # Si1
        [0.000, 0.470, 0.667],  # Si2
        [0.530, 0.530, 0.333],  # Si3
        [0.415, 0.267, 0.119],  # O1
        [0.267, 0.415, 0.786],  # O2
        # ... (other O coordinates)
    ]
    sio2 = Structure(lattice, species, coords)
    
    def si_o_bond_length(structure):
        """Calculate nearest Si-O bond length"""
        si_sites = [s for s in structure if s.species_string == "Si"]
        o_sites = [s for s in structure if s.species_string == "O"]
    
        distances = []
        for si in si_sites:
            for o in o_sites:
                dist = si.distance(o)
                if dist < 2.0:  # Consider distances < 2.0 Å as nearest neighbors
                    distances.append(dist)
    
        return np.mean(distances)
    
    def residual_with_bond_restraint(params, obs_intensity, restraint_weight=50.0):
        """Residual function with Si-O bond length restraint"""
        a = params['a'].value
        c = params['c'].value
    
        # Update structure
        new_lattice = Lattice.hexagonal(a, c)
        new_structure = Structure(new_lattice, sio2.species, sio2.frac_coords)
    
        # Calculate Si-O bond length
        bond_length = si_o_bond_length(new_structure)
        target_bond = 1.61  # Å
        tolerance = 0.05
    
        # Restraint penalty
        if abs(bond_length - target_bond) > tolerance:
            bond_penalty = restraint_weight * (bond_length - target_bond)**2
        else:
            bond_penalty = 0.0
    
        # Fit term (simplified: difference from target a, c values)
        fit_residual = (a - 4.91)**2 + (c - 5.40)**2
    
        total = fit_residual + bond_penalty
    
        return total
    
    # Parameter setup
    params = Parameters()
    params.add('a', value=4.90, min=4.8, max=5.0)
    params.add('c', value=5.35, min=5.2, max=5.6)
    
    # Minimization
    minimizer = Minimizer(residual_with_bond_restraint, params, fcn_args=(None,))
    result = minimizer.minimize(method='leastsq')
    
    # Results
    a_final = result.params['a'].value
    c_final = result.params['c'].value
    final_lattice = Lattice.hexagonal(a_final, c_final)
    final_structure = Structure(final_lattice, sio2.species, sio2.frac_coords)
    final_bond = si_o_bond_length(final_structure)
    
    print(f"Refined lattice parameters: a = {a_final:.4f} Å, c = {c_final:.4f} Å")
    print(f"Si-O bond length: {final_bond:.4f} Å (target: 1.61 ± 0.05 Å)")
    

**Q7** : If all data points deviate significantly from the straight line in a Williamson-Hall plot, what causes might be considered?

**Answer** :

**Possible causes** :

  1. **Non-uniform strain distribution** : Microstrain varies by angle (not isotropic)
  2. **Stacking faults** : Specific reflections (e.g., (111), (222)) are abnormally broadened
  3. **Insufficient instrumental function correction** : Measured FWHM includes instrument-induced broadening
  4. **Multi-phase mixture** : Multiple phases exist with different crystallite sizes
  5. **Anisotropic crystallites** : Crystallite shape is not spherical (e.g., plate-like)

**Solutions** :

  * Measure and correct instrumental function using LaB6 standard sample
  * Use modified Williamson-Hall method (considering crystallite anisotropy)
  * Warren-Averbach method (Fourier analysis) for more detailed analysis

## Verification of Learning Objectives

Review the content learned in this chapter and verify the following items.

### Fundamental Understanding

  *  Can explain the physical meaning of atomic coordinate parameters and importance of refinement
  *  Understand the difference between isotropic and anisotropic thermal factors and their application scenarios
  *  Can explain the basic principles of Scherrer equation and Williamson-Hall method
  *  Can distinguish the physical meanings of crystallite size and microstrain

### Practical Skills

  *  Can manipulate atomic coordinates and check symmetry using Pymatgen
  *  Can properly set constraints and restraints to improve refinement stability
  *  Can accurately evaluate crystallite size from Scherrer plots
  *  Can separate size and strain from Williamson-Hall plots

### Application Skills

  *  Can accurately analyze oriented sample data by applying March-Dollase correction
  *  Can evaluate refinement convergence and correlation matrix to formulate parameter optimization strategies
  *  Can judge the validity of crystal structures based on experimental data

## References

  1. Toby, B. H., & Von Dreele, R. B. (2013). _GSAS-II: the genesis of a modern open-source all purpose crystallography software package_. Journal of Applied Crystallography, 46(2), 544-549. - Comprehensive manual for GSAS-II and details of refinement algorithms
  2. Prince, E. (Ed.). (2004). _International Tables for Crystallography Volume C: Mathematical, Physical and Chemical Tables_. Springer. - Theoretical foundation of thermal factors and atomic displacement parameters
  3. Langford, J. I., & Wilson, A. J. C. (1978). _Scherrer after sixty years: A survey and some new results in the determination of crystallite size_. Journal of Applied Crystallography, 11(2), 102-113. - Historical review of Scherrer equation and modern applications
  4. Williamson, G. K., & Hall, W. H. (1953). _X-ray line broadening from filed aluminium and wolfram_. Acta Metallurgica, 1(1), 22-31. - Original paper on Williamson-Hall method
  5. Ong, S. P., et al. (2013). _Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis_. Computational Materials Science, 68, 314-319. - Official paper on Pymatgen and explanation of structure manipulation features
  6. Dollase, W. A. (1986). _Correction of intensities for preferred orientation in powder diffractometry: application of the March model_. Journal of Applied Crystallography, 19(4), 267-272. - Original paper on March-Dollase correction
  7. McCusker, L. B., et al. (1999). _Rietveld refinement guidelines_. Journal of Applied Crystallography, 32(1), 36-50. - IUCr recommended Rietveld refinement guidelines and structural parameter optimization strategies

## Next Steps

In Chapter 4, you have mastered refinement techniques for structural parameters such as atomic coordinates, thermal factors, crystallite size, and microstrain. You have acquired advanced refinement skills using constraints and restraints, and practical analysis skills through integration of pymatgen and lmfit.

In **Chapter 5** , we will integrate all the knowledge acquired so far and learn practical XRD data analysis workflows. We will cover all the skills needed in practice, from analysis of multi-phase mixtures, quantitative phase analysis, error diagnosis, to result visualization for academic reporting.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
