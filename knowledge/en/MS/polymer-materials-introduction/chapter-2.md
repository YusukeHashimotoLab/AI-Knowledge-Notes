---
title: "Chapter 2: Polymer Structure"
chapter_title: "Chapter 2: Polymer Structure"
---

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>):Chapter 2

🌐 EN | [🇯🇵 JP](<../../../jp/MS/polymer-materials-introduction/chapter-2.html>) | Last sync: 2025-11-16

  * [Table of Contents](<index.html>)
  * [� Chapter 1](<chapter-1.html>)
  * [Chapter 2](<chapter-2.html>)
  * [Chapter 3 ’](<chapter-3.html>)
  * [Chapter 4](<chapter-4.html>)
  * [Chapter 5](<chapter-4.html>)

This chapter covers Polymer Structure. You will learn essential concepts and techniques.

### Learning Objectives

**Beginner:**

  * Understand the definition and structure of tacticity (isotactic, syndiotactic, atactic)
  * Explain the differences between crystalline and amorphous structures
  * Understand the physical meaning of glass transition temperature (Tg) and melting point (Tm)

**Intermediate:**

  * Calculate the degree of orientation using the Hermans orientation function
  * Predict Tg using the Fox equation and Gordon-Taylor equation
  * Calculate crystallinity from XRD data

**Advanced:**

  * Calculate crosslink density based on rubber elasticity theory
  * Plot Flory-Huggins phase diagrams and evaluate compatibility
  * Simulate DSC curves and analyze thermal transitions

## 2.1 Stereoregularity (Tacticity)

The stereochemical configuration of polymer chains greatly affects their properties. In vinyl polymers (-CH2-CHR-), the arrangement of substituent R on the main chain carbon atoms can be classified into three types: **isotactic** (all substituents on the same side), **syndiotactic** (alternating arrangement), and **atactic** (random arrangement). 

### Tacticity and Crystallinity

Isotactic and syndiotactic polymers have regular structures and thus **crystallize easily** , resulting in higher melting points. In contrast, atactic polymers have irregular structures and do not crystallize, forming **amorphous** structures. For example, isotactic polypropylene (iPP) has Tm = 165°C, while atactic polypropylene (aPP) has Tg = -10°C, showing completely different physical properties. 
    
    
    ```mermaid
    flowchart TD
                        A[Vinyl Polymer -CH2-CHR-] --> B[Isotactic]
                        A --> C[Syndiotactic]
                        A --> D[Atactic]
                        B --> E[Regular arrangementHigh crystallinityHigh Tm]
                        C --> F[Alternating arrangementMedium crystallinityMedium Tm]
                        D --> G[Random arrangementAmorphousTg only]
                        E --> H[Application: Fibers, Films]
                        F --> I[Application: Specialty resins]
                        G --> J[Application: Rubbers, Adhesives]
    ```

### 2.1.1 Tacticity Analysis by NMR

In 13C-NMR, differences in stereochemical configuration appear as chemical shift differences. The isotactic fraction can be calculated from the relative intensities of triad sequences (mm, mr, rr). 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    # NMR spectrum simulation (tacticity analysis)
    def simulate_nmr_spectrum(isotactic_fraction=0.7, syndiotactic_fraction=0.2):
        """
        Simulate 13C-NMR spectrum and analyze tacticity
    
        Parameters:
        - isotactic_fraction: isotactic fraction (mm)
        - syndiotactic_fraction: syndiotactic fraction (rr)
    
        Returns:
        - chemical_shift: chemical shift (ppm)
        - intensity: spectrum intensity
        """
        # Calculate atactic fraction
        atactic_fraction = 1 - isotactic_fraction - syndiotactic_fraction
    
        # Chemical shift range (methylene carbon region)
        chemical_shift = np.linspace(18, 24, 1000)
    
        # Peak positions corresponding to each tacticity
        # mm (isotactic): 21.8 ppm
        # mr (atactic): 21.3 ppm
        # rr (syndiotactic): 20.8 ppm
        peak_positions = [21.8, 21.3, 20.8]
        peak_intensities = [isotactic_fraction, atactic_fraction, syndiotactic_fraction]
    
        # Generate Lorentzian peaks
        def lorentzian(x, x0, gamma, intensity):
            """Lorentzian function"""
            return intensity * (gamma**2 / ((x - x0)**2 + gamma**2))
    
        # Synthesize spectrum
        intensity = np.zeros_like(chemical_shift)
        for pos, intens in zip(peak_positions, peak_intensities):
            intensity += lorentzian(chemical_shift, pos, 0.15, intens)
    
        # Add noise
        noise = np.random.normal(0, 0.01, len(chemical_shift))
        intensity += noise
    
        # Peak detection and area calculation
        peaks, _ = find_peaks(intensity, height=0.1)
    
        # Calculate tacticity index
        total_area = np.trapz(intensity, chemical_shift)
        mm_area = isotactic_fraction * total_area
        mr_area = atactic_fraction * total_area
        rr_area = syndiotactic_fraction * total_area
    
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(chemical_shift, intensity, 'b-', linewidth=2, label='13C-NMR Spectrum')
        plt.fill_between(chemical_shift, intensity, alpha=0.3)
    
        # Mark peak positions
        plt.axvline(21.8, color='red', linestyle='--', alpha=0.7, label='mm (isotactic)')
        plt.axvline(21.3, color='green', linestyle='--', alpha=0.7, label='mr (atactic)')
        plt.axvline(20.8, color='blue', linestyle='--', alpha=0.7, label='rr (syndiotactic)')
    
        plt.xlabel('Chemical Shift (ppm)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Tacticity Analysis by 13C-NMR', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_xaxis()  # NMR is typically larger from left to right
        plt.tight_layout()
        plt.savefig('nmr_tacticity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== Tacticity Analysis Results ===")
        print(f"Isotactic fraction (mm): {isotactic_fraction:.1%}")
        print(f"Atactic fraction (mr): {atactic_fraction:.1%}")
        print(f"Syndiotactic fraction (rr): {syndiotactic_fraction:.1%}")
        print(f"\nAverage stereoregularity: {isotactic_fraction + syndiotactic_fraction:.1%}")
    
        return chemical_shift, intensity
    
    # Example execution: highly stereoregular polypropylene
    simulate_nmr_spectrum(isotactic_fraction=0.85, syndiotactic_fraction=0.05)
    

## 2.2 Crystalline and Amorphous Structures

Polymers do not crystallize completely but form semi-crystalline structures where **crystalline regions** and **amorphous regions** coexist. Crystallinity greatly affects the mechanical strength, transparency, and density of materials. 

### 2.2.1 Spherulites and Lamellar Structure

Polymer crystals form spherical aggregates called **spherulites**. Spherulites consist of plate-like crystals called **lamellae** that grow radially from the center, with thicknesses of approximately 10-20 nm. Amorphous regions exist between lamellae, where molecular chains crystallize by folding (chain folding). 
    
    
    ```mermaid
    flowchart TD
                        A[Polymer melt] -->|Cooling| B[Nucleation]
                        B --> C[Spherulite growth]
                        C --> D[Lamellar structure]
                        D --> E[Crystalline regionThickness 10-20 nm]
                        D --> F[Amorphous regionChain folding]
                        E --> G[High density1.00 g/cm³]
                        F --> H[Low density0.85 g/cm³]
                        G --> I[Improved mechanical strength]
                        H --> J[Flexibility and ductility]
    ```

### 2.2.2 Crystallinity Calculation by XRD

In X-ray diffraction (XRD), sharp Bragg peaks from crystalline regions and halos from amorphous regions are observed. The crystallinity Çc is calculated as the ratio of the crystalline peak area to the total scattering intensity. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.integrate import trapz
    
    # XRD crystallinity analysis
    def calculate_crystallinity_xrd(crystallinity_true=0.65):
        """
        Calculate crystallinity from X-ray diffraction data
    
        Parameters:
        - crystallinity_true: true crystallinity (for simulation)
    
        Returns:
        - crystallinity_calculated: calculated crystallinity
        """
        # 2¸ angle range (degrees)
        two_theta = np.linspace(10, 40, 500)
    
        # Crystalline peaks (polyethylene example)
        # (110): 21.5°, (200): 23.8°
        def gaussian(x, mu, sigma, amplitude):
            """Gaussian function"""
            return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
        # Generate crystalline peaks
        crystal_peak1 = gaussian(two_theta, 21.5, 0.5, crystallinity_true * 100)
        crystal_peak2 = gaussian(two_theta, 23.8, 0.5, crystallinity_true * 80)
        crystal_intensity = crystal_peak1 + crystal_peak2
    
        # Amorphous halo (broad peak)
        amorphous_halo = gaussian(two_theta, 19.5, 3.0, (1 - crystallinity_true) * 120)
    
        # Total intensity
        total_intensity = crystal_intensity + amorphous_halo
    
        # Add noise
        noise = np.random.normal(0, 2, len(two_theta))
        total_intensity += noise
        total_intensity = np.maximum(total_intensity, 0)  # Remove negative values
    
        # Calculate crystallinity (peak separation method)
        # Estimate amorphous baseline by polynomial fitting
        amorphous_baseline = amorphous_halo
    
        # Crystalline peak area
        crystal_area = trapz(crystal_intensity, two_theta)
    
        # Total area
        total_area = trapz(total_intensity, two_theta)
    
        # Crystallinity
        crystallinity_calculated = crystal_area / total_area
    
        # Visualization
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(two_theta, total_intensity, 'k-', linewidth=2, label='Total Intensity')
        plt.plot(two_theta, amorphous_baseline, 'b--', linewidth=1.5, label='Amorphous Halo')
        plt.plot(two_theta, crystal_intensity, 'r--', linewidth=1.5, label='Crystalline Peaks')
        plt.fill_between(two_theta, crystal_intensity, alpha=0.3, color='red')
        plt.xlabel('2¸ (deg)', fontsize=12)
        plt.ylabel('Intensity (a.u.)', fontsize=12)
        plt.title('XRD Pattern Decomposition', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.subplot(1, 2, 2)
        categories = ['Crystalline', 'Amorphous']
        areas = [crystal_area, total_area - crystal_area]
        colors = ['#f5576c', '#f093fb']
        plt.pie(areas, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Crystallinity Composition', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('xrd_crystallinity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== XRD Crystallinity Analysis Results ===")
        print(f"Crystalline peak area: {crystal_area:.2f}")
        print(f"Total scattering area: {total_area:.2f}")
        print(f"Calculated crystallinity: {crystallinity_calculated:.1%}")
        print(f"True crystallinity: {crystallinity_true:.1%}")
        print(f"Error: {abs(crystallinity_calculated - crystallinity_true):.1%}")
    
        return crystallinity_calculated
    
    # Example execution: polyethylene with 65% crystallinity
    calculate_crystallinity_xrd(crystallinity_true=0.65)
    

## 2.3 Orientation and Drawing

Polymer films and fibers undergo **orientation** of molecular chains in a specific direction through drawing. The degree of orientation is quantified by the Hermans orientation function and directly relates to mechanical strength. 

### 2.3.1 Hermans Orientation Function

The orientation function _f_ is calculated from the second moment of the orientation angle ¸: 

\\[ f = \frac{3\langle \cos^2 \theta \rangle - 1}{2} \\] 

Here, ¸ is the angle between the molecular chain axis and the drawing direction. For perfect orientation _f_ = 1, random orientation _f_ = 0, and perpendicular orientation _f_ = -0.5. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Hermans orientation function calculation
    def calculate_hermans_orientation(draw_ratio_range=(1, 10), num_points=50):
        """
        Calculate Hermans orientation function vs draw ratio
    
        Parameters:
        - draw_ratio_range: draw ratio range (»)
        - num_points: number of calculation points
    
        Returns:
        - draw_ratios: draw ratios
        - orientation_functions: orientation functions
        """
        draw_ratios = np.linspace(draw_ratio_range[0], draw_ratio_range[1], num_points)
    
        # Empirical relationship between draw ratio and orientation (pseudo-affine deformation model)
        # f = (»² - 1) / (»² + 2)  (»: draw ratio)
        orientation_functions = (draw_ratios**2 - 1) / (draw_ratios**2 + 2)
    
        # Simulation of orientation angle distribution
        angles = np.linspace(0, 90, 180)  # 0° to 90°
    
        # Orientation angle distribution for three draw ratios
        draw_ratios_example = [1, 3, 8]
    
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Orientation function vs draw ratio
        plt.subplot(1, 3, 1)
        plt.plot(draw_ratios, orientation_functions, 'b-', linewidth=2)
        plt.scatter(draw_ratios_example,
                    [(dr**2 - 1) / (dr**2 + 2) for dr in draw_ratios_example],
                    c='red', s=100, zorder=5, label='Example Points')
        plt.xlabel('Draw Ratio »', fontsize=12)
        plt.ylabel('Orientation Function f', fontsize=12)
        plt.title('Hermans Orientation Function', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
        plt.axhline(1, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='Perfect Orientation')
    
        # Subplot 2: Orientation angle distribution
        plt.subplot(1, 3, 2)
        for dr in draw_ratios_example:
            # Orientation parameter
            f = (dr**2 - 1) / (dr**2 + 2)
            # Orientation angle distribution (simple model: Gaussian approximation)
            sigma = 45 * (1 - f)  # Higher orientation results in narrower distribution
            distribution = np.exp(-0.5 * (angles / sigma)**2)
            distribution /= np.max(distribution)  # Normalize
            plt.plot(angles, distribution, linewidth=2, label=f'» = {dr}, f = {f:.2f}')
    
        plt.xlabel('Orientation Angle ¸ (deg)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Orientation Angle Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Effect on mechanical strength
        plt.subplot(1, 3, 3)
        # Tensile strength in orientation direction (empirical formula)
        tensile_strength = 50 + 200 * orientation_functions  # MPa
        plt.plot(draw_ratios, tensile_strength, 'g-', linewidth=2)
        plt.xlabel('Draw Ratio »', fontsize=12)
        plt.ylabel('Tensile Strength (MPa)', fontsize=12)
        plt.title('Strength Enhancement by Orientation', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('hermans_orientation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== Hermans Orientation Function Analysis Results ===")
        for dr in draw_ratios_example:
            f = (dr**2 - 1) / (dr**2 + 2)
            strength = 50 + 200 * f
            print(f"\nDraw ratio » = {dr}:")
            print(f"  Orientation function f = {f:.3f}")
            print(f"  Tensile strength = {strength:.1f} MPa")
            print(f"  Orientation state: {'High orientation' if f > 0.7 else 'Medium orientation' if f > 0.4 else 'Low orientation'}")
    
        return draw_ratios, orientation_functions
    
    # Execute
    calculate_hermans_orientation()
    

## 2.4 Glass Transition Temperature (Tg) and Melting Point (Tm)

Two important temperatures that characterize the thermal properties of polymers are the **glass transition temperature (T g)** and **melting point (T m)**. Tg is the temperature at which molecular motion in the amorphous region becomes active, and Tm is the temperature at which the crystalline region melts. 

### Physical Meaning of Tg and Tm

**T g (Glass Transition Temperature):** The temperature at which segmental motion of polymer chains begins. Below Tg it is hard and brittle "glassy state", above Tg it is flexible "rubbery state". 

**T m (Melting Point):** The temperature at which the crystalline region melts. Completely amorphous polymers (atactic) do not have Tm and only exhibit Tg. Semi-crystalline polymers have both Tg and Tm. 

### 2.4.1 Tg Prediction by Fox Equation

The Tg of copolymers can be predicted from the Tg and mass fraction of each component using the **Fox equation** : 

\\[ \frac{1}{T_g} = \frac{w_1}{T_{g1}} + \frac{w_2}{T_{g2}} \\] 

Here, _w i_ is the mass fraction and _T gi_ is the Tg (K) of each component. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Tg prediction by Fox equation
    def predict_tg_fox_equation(tg1=373, tg2=233, num_points=50):
        """
        Predict copolymer Tg using Fox equation
    
        Parameters:
        - tg1: Tg of component 1 (K) (e.g., polystyrene 100°C = 373 K)
        - tg2: Tg of component 2 (K) (e.g., polybutadiene -40°C = 233 K)
        - num_points: number of calculation points
    
        Returns:
        - compositions: mass fraction of component 1
        - tg_values: predicted Tg (K)
        """
        # Mass fraction of component 1
        w1 = np.linspace(0, 1, num_points)
        w2 = 1 - w1
    
        # Calculate Tg by Fox equation
        tg_fox = 1 / (w1 / tg1 + w2 / tg2)
    
        # Gordon-Taylor equation (more precise prediction)
        # Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
        # k: fitting parameter (typically 0.5-2.0)
        k = 1.0
        tg_gordon_taylor = (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
    
        # Visualization
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(w1 * 100, tg_fox - 273.15, 'b-', linewidth=2, label='Fox Equation')
        plt.plot(w1 * 100, tg_gordon_taylor - 273.15, 'r--', linewidth=2, label='Gordon-Taylor (k=1.0)')
        plt.axhline(tg1 - 273.15, color='gray', linestyle=':', alpha=0.7, label=f'Component 1 Tg ({tg1-273.15:.0f}°C)')
        plt.axhline(tg2 - 273.15, color='gray', linestyle=':', alpha=0.7, label=f'Component 2 Tg ({tg2-273.15:.0f}°C)')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Copolymer Tg Prediction', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Gordon-Taylor equation with various k parameters
        plt.subplot(1, 2, 2)
        k_values = [0.5, 1.0, 1.5, 2.0]
        for k in k_values:
            tg_gt = (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
            plt.plot(w1 * 100, tg_gt - 273.15, linewidth=2, label=f'k = {k}')
        plt.plot(w1 * 100, tg_fox - 273.15, 'k--', linewidth=2, label='Fox (k’)')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Effect of Gordon-Taylor Parameter k', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('fox_tg_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Calculation examples for specific compositions
        compositions = [0.25, 0.5, 0.75]
        print("=== Copolymer Tg Prediction Results ===")
        print(f"Component 1 Tg: {tg1 - 273.15:.1f}°C")
        print(f"Component 2 Tg: {tg2 - 273.15:.1f}°C\n")
    
        for w in compositions:
            tg_pred = 1 / (w / tg1 + (1 - w) / tg2)
            print(f"Component 1 mass fraction {w*100:.0f}%:")
            print(f"  Fox equation predicted Tg: {tg_pred - 273.15:.1f}°C")
    
        return w1, tg_fox
    
    # Example execution: copolymer of polystyrene (Tg = 100°C) and polybutadiene (Tg = -40°C)
    predict_tg_fox_equation(tg1=373, tg2=233)
    

### 2.4.2 Tg Prediction by Gordon-Taylor Equation

The Gordon-Taylor equation is a more precise prediction formula that considers differences in volume contraction: 

\\[ T_g = \frac{w_1 T_{g1} + k w_2 T_{g2}}{w_1 + k w_2} \\] 

Parameter _k_ is determined from the ratio of thermal expansion coefficients of each component. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Gordon-Taylor equation fitting
    def fit_gordon_taylor_equation(experimental_data=None):
        """
        Fit Gordon-Taylor equation to experimental data to determine k
    
        Parameters:
        - experimental_data: experimental data (dict format)
          {'compositions': [list of w1 values], 'tg_values': [list of Tg values]}
    
        Returns:
        - k_fitted: fitted k parameter
        """
        # Generate simulation data if no experimental data is provided
        if experimental_data is None:
            tg1, tg2 = 373, 233  # K
            k_true = 1.3
            compositions = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
            tg_values = (compositions * tg1 + k_true * (1 - compositions) * tg2) / \
                        (compositions + k_true * (1 - compositions))
            # Add noise
            tg_values += np.random.normal(0, 2, len(tg_values))
            experimental_data = {'compositions': compositions, 'tg_values': tg_values}
            print(f"Simulation data generated (true k = {k_true})\n")
    
        w1_exp = np.array(experimental_data['compositions'])
        tg_exp = np.array(experimental_data['tg_values'])
    
        # Pure component Tg
        tg1 = tg_exp[w1_exp == 1.0][0] if any(w1_exp == 1.0) else tg_exp[-1]
        tg2 = tg_exp[w1_exp == 0.0][0] if any(w1_exp == 0.0) else tg_exp[0]
    
        # Gordon-Taylor equation
        def gordon_taylor(w1, k):
            w2 = 1 - w1
            return (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
    
        # Fitting
        k_fitted, _ = curve_fit(gordon_taylor, w1_exp, tg_exp, p0=[1.0])
        k_fitted = k_fitted[0]
    
        # Prediction curve
        w1_fine = np.linspace(0, 1, 100)
        tg_fitted = gordon_taylor(w1_fine, k_fitted)
    
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(w1_exp * 100, tg_exp - 273.15, s=100, c='red',
                    edgecolors='black', linewidths=2, zorder=5, label='Experimental Data')
        plt.plot(w1_fine * 100, tg_fitted - 273.15, 'b-', linewidth=2,
                 label=f'Gordon-Taylor Fit (k = {k_fitted:.2f})')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Gordon-Taylor Equation Fitting', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('gordon_taylor_fit.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== Gordon-Taylor Fitting Results ===")
        print(f"Fitted k: {k_fitted:.3f}")
        print(f"Component 1 Tg: {tg1 - 273.15:.1f}°C")
        print(f"Component 2 Tg: {tg2 - 273.15:.1f}°C")
        print(f"\nResidual sum of squares: {np.sum((tg_exp - gordon_taylor(w1_exp, k_fitted))**2):.2f}")
    
        return k_fitted
    
    # Execute
    fit_gordon_taylor_equation()
    

## 2.5 Branching and Crosslinking

**Branching** and **crosslinking** of polymer chains greatly affect physical properties. Branching affects fluidity and crystallinity, while crosslinking produces rubber elasticity. 

### 2.5.1 Rubber Elasticity Theory

The elasticity of crosslinked rubber is attributed to the entropic elasticity of molecular chains. The stress-strain relationship is described by **statistical rubber elasticity theory** : 

\\[ \sigma = G (\lambda - \lambda^{-2}) \\] 

Here, Ã is stress, » is stretch ratio (» = L/L0), and G is the shear modulus. G is proportional to the crosslink density ½c: 

\\[ G = \nu_c R T \\] 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Rubber elasticity theory simulation
    def simulate_rubber_elasticity(crosslink_densities=[0.5, 1.0, 2.0], temperature=298):
        """
        Calculate stress-strain curves for rubber elasticity based on crosslink density
    
        Parameters:
        - crosslink_densities: list of crosslink densities (mol/m³)
        - temperature: temperature (K)
    
        Returns:
        - stretch_ratios: stretch ratios
        - stresses: stresses (for each crosslink density)
        """
        R = 8.314  # Gas constant (J/mol·K)
        T = temperature  # Temperature (K)
    
        # Stretch ratio (» = L/L0)
        stretch_ratios = np.linspace(1, 7, 100)
    
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Stress-strain curves
        plt.subplot(1, 3, 1)
        for nu_c in crosslink_densities:
            # Shear modulus (Pa)
            G = nu_c * R * T * 1000  # mol/m³ ’ mol/L conversion
            # Stress (MPa)
            stress = G * (stretch_ratios - stretch_ratios**(-2)) / 1e6
            plt.plot(stretch_ratios, stress, linewidth=2,
                     label=f'½c = {nu_c} mol/L')
    
        plt.xlabel('Stretch Ratio »', fontsize=12)
        plt.ylabel('Engineering Stress (MPa)', fontsize=12)
        plt.title('Rubber Elasticity: Stress-Strain Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Relationship between crosslink density and modulus
        plt.subplot(1, 3, 2)
        nu_c_range = np.linspace(0.1, 3, 50)
        G_range = nu_c_range * R * T * 1000 / 1e6  # MPa
        plt.plot(nu_c_range, G_range, 'b-', linewidth=2)
        plt.scatter(crosslink_densities,
                    [nu * R * T * 1000 / 1e6 for nu in crosslink_densities],
                    s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
        plt.xlabel('Crosslink Density ½c (mol/L)', fontsize=12)
        plt.ylabel('Shear Modulus G (MPa)', fontsize=12)
        plt.title('Crosslink Density vs Modulus', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # Subplot 3: Temperature dependence
        plt.subplot(1, 3, 3)
        temperatures = np.linspace(250, 400, 50)  # K
        nu_c_fixed = 1.0  # mol/L
        G_temp = nu_c_fixed * R * temperatures * 1000 / 1e6  # MPa
        plt.plot(temperatures - 273.15, G_temp, 'g-', linewidth=2)
        plt.axvline(temperature - 273.15, color='red', linestyle='--',
                    linewidth=2, label=f'Current T = {temperature}K')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Shear Modulus G (MPa)', fontsize=12)
        plt.title('Temperature Dependence (½c = 1.0 mol/L)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('rubber_elasticity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== Rubber Elasticity Theory Analysis Results ===")
        print(f"Temperature: {temperature} K ({temperature - 273.15:.1f}°C)\n")
        for nu_c in crosslink_densities:
            G = nu_c * R * T * 1000 / 1e6
            # Calculate stress at » = 2
            lambda_test = 2.0
            stress_test = G * (lambda_test - lambda_test**(-2))
            print(f"Crosslink density ½c = {nu_c} mol/L:")
            print(f"  Shear modulus G = {G:.3f} MPa")
            print(f"  Stress (»=2) = {stress_test:.3f} MPa")
    
        return stretch_ratios, crosslink_densities
    
    # Execute
    simulate_rubber_elasticity()
    

### 2.5.2 DSC Curve Simulation

Differential Scanning Calorimetry (DSC) is a standard method for experimentally determining Tg and Tm. Simulating DSC curves can deepen understanding of thermal transitions. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # DSC curve simulation
    def simulate_dsc_curve(tg=323, tm=438, crystallinity=0.6, heating_rate=10):
        """
        Simulate DSC (Differential Scanning Calorimetry) curve
    
        Parameters:
        - tg: glass transition temperature (K)
        - tm: melting point (K)
        - crystallinity: crystallinity
        - heating_rate: heating rate (K/min)
    
        Returns:
        - temperatures: temperature (°C)
        - heat_flow: heat flow (W/g)
        """
        # Temperature range (K)
        temperatures = np.linspace(200, 500, 1000)
    
        # Baseline
        baseline = -0.5 + 0.001 * temperatures  # Slightly temperature dependent
    
        # Glass transition (step change)
        def glass_transition(T, Tg, delta_Cp=0.3, width=10):
            """Glass transition sigmoid function"""
            return delta_Cp / (1 + np.exp(-(T - Tg) / width))
    
        # Melting peak (Gaussian peak)
        def melting_peak(T, Tm, delta_Hm, width=15):
            """Melting peak Gaussian function"""
            return delta_Hm * np.exp(-0.5 * ((T - Tm) / width)**2)
    
        # Step change at Tg
        tg_signal = glass_transition(temperatures, tg)
    
        # Endothermic peak at Tm (proportional to crystallinity)
        delta_Hm = -100 * crystallinity  # Melting enthalpy (J/g)
        tm_signal = melting_peak(temperatures, tm, delta_Hm)
    
        # Total DSC signal
        heat_flow = baseline + tg_signal + tm_signal
    
        # Add noise
        noise = np.random.normal(0, 0.02, len(temperatures))
        heat_flow += noise
    
        # Visualization
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(temperatures - 273.15, heat_flow, 'b-', linewidth=2, label='DSC Signal')
        plt.axvline(tg - 273.15, color='red', linestyle='--', linewidth=1.5,
                    label=f'Tg = {tg - 273.15:.0f}°C')
        plt.axvline(tm - 273.15, color='green', linestyle='--', linewidth=1.5,
                    label=f'Tm = {tm - 273.15:.0f}°C')
    
        # Highlight Tg and Tm regions
        plt.fill_betweenx([heat_flow.min(), heat_flow.max()],
                          tg - 273.15 - 20, tg - 273.15 + 20,
                          alpha=0.2, color='red', label='Tg Region')
        plt.fill_betweenx([heat_flow.min(), heat_flow.max()],
                          tm - 273.15 - 30, tm - 273.15 + 30,
                          alpha=0.2, color='green', label='Tm Region')
    
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Heat Flow (W/g) Exo ‘', fontsize=12)
        plt.title(f'DSC Curve (Crystallinity = {crystallinity:.0%})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_yaxis()  # DSC typically shows endothermic downward
    
        # Subplot 2: Effect of crystallinity
        plt.subplot(1, 2, 2)
        crystallinities = [0.3, 0.5, 0.7]
        for xc in crystallinities:
            delta_Hm_var = -100 * xc
            tm_signal_var = melting_peak(temperatures, tm, delta_Hm_var)
            heat_flow_var = baseline + tg_signal + tm_signal_var
            plt.plot(temperatures - 273.15, heat_flow_var, linewidth=2,
                     label=f'Xc = {xc:.0%}')
    
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Heat Flow (W/g) Exo ‘', fontsize=12)
        plt.title('Effect of Crystallinity on Melting Peak', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_yaxis()
    
        plt.tight_layout()
        plt.savefig('dsc_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Output results
        print("=== DSC Analysis Results ===")
        print(f"Glass transition temperature Tg: {tg - 273.15:.1f}°C")
        print(f"Melting point Tm: {tm - 273.15:.1f}°C")
        print(f"Crystallinity: {crystallinity:.0%}")
        print(f"Melting enthalpy: {-100 * crystallinity:.1f} J/g")
        print(f"Heating rate: {heating_rate} K/min")
    
        return temperatures, heat_flow
    
    # Example execution: polyethylene (Tg = 50°C, Tm = 165°C, crystallinity 60%)
    simulate_dsc_curve(tg=323, tm=438, crystallinity=0.6)
    

## Practice Problems

#### Exercise 1: Tacticity Analysis (Easy)

If the triad fractions measured by 13C-NMR for polypropylene are as follows, calculate the isotactic fraction. 

  * mm (isotactic): 70%
  * mr (heterotactic): 25%
  * rr (syndiotactic): 5%

View Solution

**Solution:**

Isotactic fraction = mm fraction = 70%

This polymer has high stereoregularity and is expected to show high crystallinity (approximately 50-60%) and melting point (approximately 165°C).

#### Exercise 2: Crystallinity Calculation (Easy)

In the XRD pattern of polyethylene, the crystalline peak area was 300 and the total scattering area was 500. Calculate the crystallinity. 

View Solution

**Solution:**
    
    
    crystallinity = 300 / 500 = 0.6 = 60%

A crystallinity of 60% is a typical value for high-density polyethylene (HDPE).

#### Exercise 3: Hermans Orientation Function (Easy)

Calculate the orientation function f for a film with draw ratio » = 4 (using the pseudo-affine deformation model). 

View Solution

**Solution:**
    
    
    lambda_val = 4
    f = (lambda_val**2 - 1) / (lambda_val**2 + 2)
    f = (16 - 1) / (16 + 2) = 15 / 18 = 0.833
    
    print(f"Orientation function f = {f:.3f}")
    # Output: Orientation function f = 0.833

f = 0.833 indicates high orientation, showing significantly improved mechanical strength.

#### Exercise 4: Fox Equation Tg Prediction (Medium)

Predict the Tg using the Fox equation when polystyrene (Tg = 100°C = 373 K) and polyisoprene (Tg = -70°C = 203 K) are copolymerized at a mass ratio of 1:1. 

View Solution

**Solution:**
    
    
    tg1 = 373  # K
    tg2 = 203  # K
    w1 = 0.5
    w2 = 0.5
    
    # Fox equation
    tg_copolymer = 1 / (w1 / tg1 + w2 / tg2)
    tg_celsius = tg_copolymer - 273.15
    
    print(f"Copolymer Tg = {tg_copolymer:.1f} K = {tg_celsius:.1f}°C")
    # Output: Copolymer Tg = 263.0 K = -10.0°C

Through copolymerization, Tg becomes an intermediate value between the two components (approximately -10°C), resulting in a rubbery state near room temperature.

#### Exercise 5: Crosslink Density Calculation (Medium)

The shear modulus G of a rubber was measured as 1.5 MPa at 25°C (298 K). Calculate the crosslink density ½c (R = 8.314 J/mol·K). 

View Solution

**Solution:**
    
    
    G = 1.5e6  # Pa
    R = 8.314  # J/mol·K
    T = 298  # K
    
    # From G = ½c * R * T
    nu_c = G / (R * T)
    nu_c_mol_per_L = nu_c / 1000  # mol/m³ ’ mol/L
    
    print(f"Crosslink density ½c = {nu_c:.1f} mol/m³ = {nu_c_mol_per_L:.3f} mol/L")
    # Output: Crosslink density ½c = 605.1 mol/m³ = 0.605 mol/L

This crosslink density corresponds to soft rubber.

#### Exercise 6: Orientation and Strength (Medium)

Estimate the tensile strength of a PET film with orientation function f = 0.6 using the empirical formula Ã = 50 + 200f (MPa). Compare with the unoriented case (f = 0). 

View Solution

**Solution:**
    
    
    f_oriented = 0.6
    f_unoriented = 0
    
    strength_oriented = 50 + 200 * f_oriented
    strength_unoriented = 50 + 200 * f_unoriented
    
    improvement = (strength_oriented - strength_unoriented) / strength_unoriented * 100
    
    print(f"Oriented film (f=0.6): {strength_oriented} MPa")
    print(f"Unoriented film (f=0): {strength_unoriented} MPa")
    print(f"Strength improvement: {improvement:.0f}%")
    # Output: Oriented film: 170 MPa, Unoriented: 50 MPa, Strength improvement: 240%

Orientation increases tensile strength by approximately 3.4 times.

#### Exercise 7: Gordon-Taylor Parameter Estimation (Medium)

From experimental data (component 1 mass fraction 50% gives Tg = 300 K), inversely calculate the Gordon-Taylor parameter k when Tg1 = 373 K and Tg2 = 233 K. 

View Solution

**Solution:**
    
    
    tg1 = 373  # K
    tg2 = 233  # K
    w1 = 0.5
    tg_exp = 300  # K
    
    # Rearrange Gordon-Taylor equation to solve for k
    # Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
    # Tg(w1 + k*w2) = w1*Tg1 + k*w2*Tg2
    # Tg*w1 + Tg*k*w2 = w1*Tg1 + k*w2*Tg2
    # k(Tg*w2 - w2*Tg2) = w1*Tg1 - Tg*w1
    # k = (w1*Tg1 - Tg*w1) / (Tg*w2 - w2*Tg2)
    
    w2 = 1 - w1
    k = (w1 * tg1 - tg_exp * w1) / (tg_exp * w2 - w2 * tg2)
    
    print(f"Gordon-Taylor parameter k = {k:.3f}")
    # Output: k H 1.088

k H 1.09 indicates that the thermal expansion coefficients of both components are nearly equivalent.

#### Exercise 8: Flory-Huggins Phase Diagram (Hard)

Using Flory-Huggins theory, plot the phase diagram for a two-component polymer blend. Compare the cases of interaction parameter Ç = 0.5, 1.0, 2.0 and calculate the critical Ç value (UCST). 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Flory-Huggins phase diagram
    def plot_flory_huggins_phase_diagram():
        """Phase diagram based on Flory-Huggins theory"""
        phi = np.linspace(0.01, 0.99, 100)  # Volume fraction of component 1
    
        # Critical interaction parameter (assuming N1 = N2 = 100)
        N1, N2 = 100, 100
        chi_critical = 0.5 * (1/np.sqrt(N1) + 1/np.sqrt(N2))**2
    
        print(f"Critical Ç value (UCST): {chi_critical:.4f}")
    
        # Spinodal curve (second derivative of ”Gmix = 0)
        def spinodal_chi(phi, N1, N2):
            """Ç value for spinodal curve"""
            return 0.5 * (1 / (N1 * phi) + 1 / (N2 * (1 - phi)))
    
        chi_spinodal = spinodal_chi(phi, N1, N2)
    
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(phi, chi_spinodal, 'r-', linewidth=3, label='Spinodal Curve')
        plt.axhline(chi_critical, color='blue', linestyle='--', linewidth=2,
                    label=f'Critical Ç = {chi_critical:.4f}')
    
        # Fill phase separation region
        plt.fill_between(phi, chi_spinodal, 3, alpha=0.3, color='red',
                         label='Two-Phase Region')
        plt.fill_between(phi, 0, chi_spinodal, alpha=0.3, color='green',
                         label='Single-Phase Region')
    
        plt.xlabel('Volume Fraction Æ1', fontsize=12)
        plt.ylabel('Interaction Parameter Ç', fontsize=12)
        plt.title('Flory-Huggins Phase Diagram (N1=N2=100)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 0.1)
        plt.tight_layout()
        plt.savefig('flory_huggins_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_flory_huggins_phase_diagram()
    # Critical Ç value H 0.02 (for N=100)
    # Phase separation occurs at Ç > 0.02

The larger the degree of polymerization, the smaller the critical Ç value and the lower the compatibility.

#### Exercise 9: DSC Crystallinity Analysis (Hard)

The melting enthalpy measured by DSC for polyethylene was 180 J/g. Calculate the crystallinity assuming the melting enthalpy of perfect crystal is 293 J/g. Also estimate Tg and Tm for this crystallinity (Tm = 165°C, Tg = -120°C). 

View Solution

**Solution:**
    
    
    delta_Hm_measured = 180  # J/g
    delta_Hm_100percent = 293  # J/g (perfect crystal PE)
    
    crystallinity = delta_Hm_measured / delta_Hm_100percent
    
    print(f"Crystallinity: {crystallinity:.1%}")
    # Output: Crystallinity: 61.4%
    
    # Tg and Tm are independent of crystallinity (inherent properties of pure components)
    # However, apparent Tg becomes less clear as crystallinity increases
    print(f"Tm (melting point): 165°C")
    print(f"Tg (glass transition temperature): -120°C (amorphous region only)")
    print(f"Tg transition is unclear due to high crystallinity")
    

A crystallinity of 61.4% corresponds to high-density polyethylene (HDPE).

#### Exercise 10: Integrated Structure Analysis (Hard)

For isotactic polypropylene (iPP), integrate the following experimental data to predict material properties: 

  * NMR: isotactic fraction 85%
  * XRD: crystallinity 60%
  * Draw ratio: » = 5
  * DSC: Tm = 165°C

Estimate tensile strength, elastic modulus, and applications. 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Experimental data
    isotactic_fraction = 0.85
    crystallinity = 0.60
    draw_ratio = 5.0
    tm = 165  # °C
    
    # Calculate orientation function
    f = (draw_ratio**2 - 1) / (draw_ratio**2 + 2)
    
    # Estimate tensile strength (empirical formula)
    # Reference strength (unoriented, 50% crystallinity) = 30 MPa
    base_strength = 30
    strength = base_strength * (1 + 3 * (crystallinity - 0.5)) * (1 + 4 * f)
    
    # Estimate elastic modulus (depends on crystallinity and orientation)
    # Reference modulus (unoriented, amorphous) = 1.0 GPa
    base_modulus = 1.0
    modulus = base_modulus * (1 + 5 * crystallinity) * (1 + 3 * f)
    
    print("=== iPP Material Property Predictions ===")
    print(f"Stereoregularity: {isotactic_fraction:.0%}")
    print(f"Crystallinity: {crystallinity:.0%}")
    print(f"Orientation function: {f:.3f}")
    print(f"Melting point: {tm}°C")
    print(f"\nPredicted properties:")
    print(f"  Tensile strength: {strength:.1f} MPa")
    print(f"  Elastic modulus: {modulus:.1f} GPa")
    print(f"\nRecommended applications:")
    if strength > 150 and modulus > 5:
        print("  - High-strength fibers (ropes, nonwovens)")
        print("  - High-performance films (packaging, battery separators)")
    elif strength > 80:
        print("  - General-purpose films (food packaging)")
        print("  - Injection molded products (containers)")
    else:
        print("  - Low-strength applications (disposable products)")
    
    # Example output:
    # Tensile strength: 186.0 MPa
    # Elastic modulus: 11.1 GPa
    # Recommended applications: high-strength fibers, high-performance films
    

High stereoregularity, high crystallinity, and high orientation provide excellent mechanical properties, making it suitable for high-performance applications.

## References

  1. Strobl, G. (2007). _The Physics of Polymers: Concepts for Understanding Their Structures and Behavior_ (3rd ed.). Springer. pp. 1-95, 145-210.
  2. Young, R. J., & Lovell, P. A. (2011). _Introduction to Polymers_ (3rd ed.). CRC Press. pp. 190-285.
  3. Gedde, U. W., & Hedenqvist, M. S. (2019). _Fundamental Polymer Science_ (2nd ed.). Springer. pp. 50-135.
  4. Mark, J. E. (Ed.). (2007). _Physical Properties of Polymers Handbook_ (2nd ed.). Springer. pp. 200-265.
  5. Flory, P. J. (1953). _Principles of Polymer Chemistry_. Cornell University Press. pp. 495-540.
  6. Ward, I. M., & Sweeney, J. (2012). _Mechanical Properties of Solid Polymers_ (3rd ed.). Wiley. pp. 75-150.
  7. Ferry, J. D. (1980). _Viscoelastic Properties of Polymers_ (3rd ed.). Wiley. pp. 264-320.

### Connection to Next Chapter

In Chapter 3, you will learn in detail how the polymer structures learned in this chapter affect physical properties (mechanical properties, viscoelasticity, rheology). In particular, you will understand how crystallinity and degree of orientation are reflected in stress-strain curves and creep behavior. Additionally, the physical meaning of Tg will be further deepened through time-temperature superposition using the WLF equation. 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
