---
title: "Chapter 3: Introduction to Transmission Electron Microscopy (TEM)"
chapter_title: "Chapter 3: Introduction to Transmission Electron Microscopy (TEM)"
subtitle: TEM Imaging Theory, Diffraction Analysis, and Fundamentals of High-Resolution Observation
reading_time: 25-35 minutes
difficulty: Intermediate
code_examples: 7
version: 1.0
created_at: "by:"
---

Transmission Electron Microscopy (TEM) is a powerful tool that uses electron beams transmitted through specimens to observe the internal structure of materials at the atomic level. In this chapter, we will learn TEM imaging theory, bright field/dark field imaging, Selected Area Electron Diffraction (SAED), High-Resolution TEM (HRTEM), and aberration correction techniques, and practice diffraction pattern analysis and FFT processing using Python. 

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Understand TEM imaging theory and contrast mechanisms (mass-thickness, diffraction contrast)
  * ✅ Comprehend the formation principles of bright field (BF) and dark field (DF) images and their applications
  * ✅ Index Selected Area Electron Diffraction (SAED) patterns
  * ✅ Understand the differences between lattice images and high-resolution TEM images and perform FFT analysis
  * ✅ Explain the role of Contrast Transfer Function (CTF) and aberration correction techniques
  * ✅ Implement Ewald sphere construction, SAED analysis, and FFT processing of HRTEM images in Python
  * ✅ Quantitatively evaluate the effects of defocus and spherical aberration on images

## 3.1 Fundamentals of TEM Imaging Theory

### 3.1.1 TEM Configuration

The Transmission Electron Microscope (TEM) has an optical system where the objective lens forms an image of the electron beam transmitted through the specimen, and the projection lens magnifies it.
    
    
    ```mermaid
    flowchart TD
        A[Electron Gun] --> B[Illumination System Lenses]
        B --> C[Specimen Stage]
        C --> D[Objective Lens]
        D --> E[Selected Area ApertureSelected Area Aperture]
        E --> F[Intermediate Lens]
        F --> G[Projection Lens]
        G --> H[Fluorescent Screen/Detector]
    
        D -.Diffraction Plane.-> I[Back Focal PlaneBFP]
        I -.-> F
        D -.Image Plane.-> J[Gaussian Image Plane]
        J -.-> F
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#ffeb99,stroke:#ffa500
        style J fill:#99ccff,stroke:#0066cc
    ```

**Important Concepts** :

  * **Back Focal Plane (BFP)** : The focal plane of the objective lens where the diffraction pattern is formed
  * **Gaussian Image Plane** : The plane where the real image of the specimen is formed
  * **Objective Aperture** : Placed at the BFP to select specific diffraction spots and control contrast
  * **Selected Area Aperture** : Placed at the image plane to obtain diffraction patterns from specific regions (SAED: Selected Area Electron Diffraction)

### 3.1.2 TEM Image Contrast Mechanisms

TEM image contrast is formed by three main mechanisms:

Contrast Type | Physical Origin | Applications  
---|---|---  
**Mass-Thickness Contrast** | Differences in electron scattering intensity due to specimen density and thickness | Observation of amorphous specimens, biological specimens, thickness variations  
**Diffraction Contrast** | Changes in crystal diffraction conditions (Bragg condition) | Observation of dislocations, twins, grain boundaries  
**Phase Contrast** | Interference due to phase differences between transmitted and diffracted waves | Lattice image observation in High-Resolution TEM (HRTEM)  
  
**Bright Field Image (BF)** : Formed by allowing only the transmitted beam (000 reflection) to pass through the objective aperture. Thick or strongly scattering regions appear dark.

**Dark Field Image (DF)** : Formed by allowing only a specific diffracted beam to pass through the objective aperture. Only crystal grains satisfying that reflection condition appear bright.

### 3.1.3 Contrast Transfer Function (CTF)

In high-resolution TEM, phase contrast is important. This phase contrast is described by the **Contrast Transfer Function (CTF)** :

$$ \text{CTF}(k) = A(k) \sin\left[\chi(k)\right] $$ 

Here, $A(k)$ is the aperture function, and $\chi(k)$ is the phase shift expressed as:

$$ \chi(k) = \frac{2\pi}{\lambda}\left(\frac{\lambda^2 k^2}{2}\Delta f + \frac{\lambda^4 k^4}{4}C_s\right) $$ 

  * $k$: Spatial frequency (in inverse Å)
  * $\lambda$: Electron wavelength (determined by accelerating voltage)
  * $\Delta f$: Defocus (focus deviation amount, negative values are underfocus)
  * $C_s$: Spherical aberration coefficient (lens aberration parameter)

**Physical Meaning** :

  * In spatial frequency regions where CTF is positive, phase contrast contributes to the image without inversion
  * In regions where CTF is negative, contrast is inverted (black-white reversal)
  * At frequencies where CTF becomes zero (zero crossing), that spatial frequency component does not contribute to the image
  * By setting appropriate defocus (Scherzer focus), a CTF with a constant sign over a wide spatial frequency range can be achieved

#### Code Example 3-1: Contrast Transfer Function (CTF) Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_ctf(k, voltage_kV, defocus_nm, Cs_mm, aperture_mrad=None):
        """
        Calculate Contrast Transfer Function (CTF)
    
        Parameters
        ----------
        k : array-like
            Spatial frequency [1/Å]
        voltage_kV : float
            Accelerating voltage [kV]
        defocus_nm : float
            Defocus [nm] (negative: underfocus)
        Cs_mm : float
            Spherical aberration coefficient [mm]
        aperture_mrad : float, optional
            Objective aperture semi-angle [mrad]. None for infinite (no aperture)
    
        Returns
        -------
        ctf : ndarray
            CTF values
        """
        # Electron wavelength calculation (with relativistic correction)
        m0 = 9.10938e-31  # Electron mass [kg]
        e = 1.60218e-19   # Charge [C]
        c = 2.99792e8     # Speed of light [m/s]
        h = 6.62607e-34   # Planck constant [J·s]
    
        E = voltage_kV * 1000 * e  # Energy [J]
        lambda_pm = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2))) * 1e12  # [pm]
        lambda_A = lambda_pm / 100  # [Å]
    
        # Convert parameters to appropriate units
        defocus_A = defocus_nm * 10  # [Å]
        Cs_A = Cs_mm * 1e7  # [Å]
    
        # Phase shift χ(k)
        chi = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_A
        )
    
        # Aperture function (Gaussian approximation)
        if aperture_mrad is not None:
            theta_max = aperture_mrad * 1e-3  # [rad]
            k_max = theta_max / lambda_A
            A = np.exp(-(k / k_max)**4)
        else:
            A = 1.0
    
        # CTF = A(k) * sin(χ(k))
        ctf = A * np.sin(chi)
    
        return ctf, lambda_A
    
    # Simulation settings
    voltage = 200  # [kV]
    Cs = 0.5  # [mm] (modern TEM)
    k = np.linspace(0, 10, 1000)  # Spatial frequency [1/Å]
    
    # CTF for different defocus values
    defocus_values = [-50, -70, -100]  # [nm]
    colors = ['blue', 'green', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Defocus dependence
    for df, color in zip(defocus_values, colors):
        ctf, wavelength = calculate_ctf(k, voltage, df, Cs)
        ax1.plot(k, ctf, color=color, linewidth=2, label=f'Δf = {df} nm')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
    ax1.set_ylabel('CTF', fontsize=12)
    ax1.set_title(f'CTF vs Defocus (200 kV, Cs = {Cs} mm)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1, 1)
    
    # Right panel: Spherical aberration coefficient dependence
    defocus = -70  # [nm]
    Cs_values = [0.5, 1.0, 2.0]  # [mm]
    colors2 = ['purple', 'orange', 'brown']
    
    for Cs_val, color in zip(Cs_values, colors2):
        ctf, wavelength = calculate_ctf(k, voltage, defocus, Cs_val)
        ax2.plot(k, ctf, color=color, linewidth=2, label=f'Cs = {Cs_val} mm')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
    ax2.set_ylabel('CTF', fontsize=12)
    ax2.set_title(f'CTF vs Cs (200 kV, Δf = {defocus} nm)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Scherzer focus (optimal defocus) calculation
    Cs_mm = 0.5
    lambda_A = 0.0251  # Wavelength at 200 kV
    Cs_A = Cs_mm * 1e7
    scherzer_defocus_A = -1.2 * np.sqrt(Cs_A * lambda_A)
    scherzer_defocus_nm = scherzer_defocus_A / 10
    
    print(f"Scherzer Focus: {scherzer_defocus_nm:.1f} nm")
    print(f"First zero crossing frequency: ~{1/2.5:.2f} Å")
    

**Output Interpretation** :

  * When defocus is too large, CTF oscillates at low spatial frequencies, making the image complex
  * Larger Cs leads to more vigorous CTF oscillation in the high spatial frequency region, reducing resolution
  * At Scherzer focus (~-70 nm), a positive CTF is obtained over a wide spatial frequency range

## 3.2 Bright Field and Dark Field Images

### 3.2.1 Bright Field (BF) Image Formation

Bright field images are formed by selecting only the transmitted beam (000 reflection) with the objective aperture. For thin and uniform specimens, mass-thickness contrast is dominant.

**Characteristics of Bright Field Images** :

  * Thick regions or regions containing heavy elements appear dark
  * High S/N ratio, suitable for low magnification observation
  * Effective for observing amorphous specimens and biological specimens
  * For crystalline specimens, crystal grains satisfying Bragg conditions appear dark (diffraction contrast)

### 3.2.2 Dark Field (DF) Image Formation

Dark field images are formed by selecting only a specific diffracted beam (e.g., 111 reflection) with the objective aperture.

**Advantages of Dark Field Images** :

  * Only particles with specific crystal orientations appear bright
  * Effective for detecting fine precipitates or second phases
  * Easy identification of crystal grains (grains with the same orientation have the same brightness)
  * Suitable for observing dislocations and stacking faults

**Centered Dark Field (CDF) Method** : A technique where the optical axis is tilted to bring the diffracted beam to the center. It reduces the effect of spherical aberration and provides higher resolution.

#### Code Example 3-2: Bright Field and Dark Field Image Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter, rotate
    
    def simulate_polycrystalline_sample(size=512, num_grains=50):
        """
        Simulate a polycrystalline specimen
    
        Parameters
        ----------
        size : int
            Image size [pixels]
        num_grains : int
            Number of crystal grains
    
        Returns
        -------
        grain_map : ndarray
            Grain map (grain ID for each pixel)
        orientations : ndarray
            Orientation of each grain [degrees]
        """
        # Generate grains using Voronoi tessellation
        from scipy.spatial import Voronoi
    
        # Random seed points
        np.random.seed(42)
        points = np.random.rand(num_grains, 2) * size
    
        # Coordinates of all pixels
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        pixels = np.stack([x.ravel(), y.ravel()], axis=1)
    
        # Find nearest seed point to create Voronoi regions
        from scipy.spatial.distance import cdist
        distances = cdist(pixels, points)
        grain_map = np.argmin(distances, axis=1).reshape(size, size)
    
        # Assign random orientation to each grain
        orientations = np.random.rand(num_grains) * 360
    
        return grain_map, orientations
    
    def calculate_diffraction_intensity(grain_map, orientations, hkl=(111,)):
        """
        Calculate intensity under specific diffraction conditions
    
        Parameters
        ----------
        grain_map : ndarray
            Grain map
        orientations : ndarray
            Orientation of each grain [degrees]
        hkl : tuple
            Miller indices (considering only orientation dependence for simplification)
    
        Returns
        -------
        intensity : ndarray
            Diffraction intensity map
        """
        # Bragg condition: Strong diffraction only in specific orientation range
        # Simplification: Assume strong diffraction at 30°±5°
        target_angle = 30.0
        tolerance = 5.0
    
        intensity = np.zeros_like(grain_map, dtype=float)
    
        for grain_id in range(len(orientations)):
            mask = (grain_map == grain_id)
            angle = orientations[grain_id]
    
            # Determine if diffraction condition is satisfied
            angle_diff = min(abs(angle - target_angle),
                            abs(angle - target_angle + 360),
                            abs(angle - target_angle - 360))
    
            if angle_diff < tolerance:
                # Bragg condition satisfied → Strong diffraction (dark in BF, bright in DF)
                intensity[mask] = 1.0
            else:
                intensity[mask] = 0.1  # Weak diffraction
    
        # Add noise
        intensity += np.random.normal(0, 0.05, intensity.shape)
    
        return intensity
    
    # Execute simulation
    size = 512
    grain_map, orientations = simulate_polycrystalline_sample(size, num_grains=30)
    
    # Bright field image: Strongly diffracting regions are dark
    diffraction_intensity = calculate_diffraction_intensity(grain_map, orientations)
    bf_image = 1.0 - diffraction_intensity * 0.7  # Transmitted intensity = 1 - diffraction intensity
    
    # Dark field image: Only grains satisfying specific diffraction conditions are bright
    df_image = calculate_diffraction_intensity(grain_map, orientations)
    
    # Add blur (make it more realistic)
    bf_image = gaussian_filter(bf_image, sigma=1.0)
    df_image = gaussian_filter(df_image, sigma=1.0)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grain map
    im0 = axes[0].imshow(grain_map, cmap='tab20', interpolation='nearest')
    axes[0].set_title('Crystal Grain Map\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Bright field image
    im1 = axes[1].imshow(bf_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Bright Field Image\n(Transmitted Beam: 000)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Dark field image
    im2 = axes[2].imshow(df_image, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Dark Field Image\n(Diffracted Beam: 111)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Bright field image: Crystal grains satisfying diffraction conditions appear dark")
    print("Dark field image: Only grains satisfying specific diffraction conditions (111 reflection) appear bright")
    

**Observation Points** :

  * In bright field images, crystal grains with strong diffraction appear dark
  * In dark field images, only crystal grains with specific orientations selectively appear bright
  * In actual TEM, different diffraction spots can be selected by changing the objective aperture position

## 3.3 Selected Area Electron Diffraction (SAED)

### 3.3.1 Principles of Electron Diffraction

Electron diffraction is a phenomenon where electron waves are diffracted by the periodic structure of crystals. Bragg's law applies:

$$ 2d_{hkl}\sin\theta = n\lambda $$ 

In TEM, the incident electron beam enters nearly perpendicular to the specimen, so $\theta$ is very small (typically less than 1°), and the small-angle approximation holds:

$$ \sin\theta \approx \theta \approx \tan\theta = \frac{R}{L} $$ 

Here, $R$ is the distance of the diffraction spot, and $L$ is the camera length. Therefore:

$$ d_{hkl} = \frac{\lambda L}{R} $$ 

**SAED (Selected Area Electron Diffraction)** : A technique to obtain diffraction patterns from specific regions (typically several hundred nm to several μm) using a selected area aperture.

### 3.3.2 Ewald Sphere Construction

Electron diffraction is understood as the intersection of the Ewald Sphere with reciprocal lattice points in reciprocal space.

  * **Ewald Sphere** : A sphere with radius $1/\lambda$ drawn along the incident beam direction
  * **Reciprocal Lattice Points** : Points in reciprocal space corresponding to the periodic structure of the crystal
  * **Diffraction Condition** : When the Ewald sphere passes through a reciprocal lattice point, the Bragg condition is satisfied and diffraction occurs

#### Code Example 3-3: Ewald Sphere Construction and Diffraction Pattern Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def generate_reciprocal_lattice_fcc(max_hkl=3):
        """
        Generate reciprocal lattice points for FCC crystal
    
        Parameters
        ----------
        max_hkl : int
            Maximum Miller index
    
        Returns
        -------
        points : list of tuples
            Reciprocal lattice points (h, k, l)
        """
        points = []
        for h in range(-max_hkl, max_hkl + 1):
            for k in range(-max_hkl, max_hkl + 1):
                for l in range(-max_hkl, max_hkl + 1):
                    # FCC extinction rule: h, k, l are all even or all odd
                    if (h % 2 == k % 2 == l % 2):
                        points.append((h, k, l))
        return points
    
    def plot_ewald_sphere(a_lattice=4.05, voltage_kV=200, zone_axis=[0, 0, 1]):
        """
        Plot Ewald sphere construction in 3D
    
        Parameters
        ----------
        a_lattice : float
            Lattice constant [Å]
        voltage_kV : float
            Accelerating voltage [kV]
        zone_axis : list
            Zone axis [uvw]
        """
        # Electron wavelength calculation
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        h = 6.62607e-34
        E = voltage_kV * 1000 * e
        lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        lambda_A = lambda_m * 1e10
    
        # Ewald sphere radius (inverse Å)
        k = 1 / lambda_A
    
        # Reciprocal lattice vectors (FCC)
        reciprocal_points = generate_reciprocal_lattice_fcc(max_hkl=2)
    
        # Reciprocal lattice constant
        a_star = 1 / a_lattice
    
        # 3D plot
        fig = plt.figure(figsize=(14, 6))
    
        # Left panel: 3D Ewald sphere construction
        ax1 = fig.add_subplot(121, projection='3d')
    
        # Plot reciprocal lattice points
        for (h, k, l) in reciprocal_points:
            x = h * a_star
            y = k * a_star
            z = l * a_star
            ax1.scatter(x, y, z, c='blue', s=30, alpha=0.6)
            if abs(l) <= 1:  # Near [001] zone axis
                ax1.text(x, y, z, f'  {h}{k}{l}', fontsize=8)
    
        # Ewald sphere (simplified: represented as circle)
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
    
        x_sphere = k * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = k * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = k * np.outer(np.ones(100), np.cos(phi)) - k
    
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='red')
    
        # Incident beam
        ax1.quiver(0, 0, -k, 0, 0, k*1.5, color='green', arrow_length_ratio=0.1, linewidth=2)
    
        ax1.set_xlabel('$k_x$ [1/Å]', fontsize=11)
        ax1.set_ylabel('$k_y$ [1/Å]', fontsize=11)
        ax1.set_zlabel('$k_z$ [1/Å]', fontsize=11)
        ax1.set_title('Ewald Sphere Construction\n(3D Reciprocal Space)', fontsize=13, fontweight='bold')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_zlim(-2, 2)
    
        # Right panel: [001] zone axis diffraction pattern (2D projection)
        ax2 = fig.add_subplot(122)
    
        for (h, k, l) in reciprocal_points:
            if l == 0:  # [001] zone axis
                x = h * a_star
                y = k * a_star
                ax2.scatter(x, y, c='blue', s=100, alpha=0.8)
                ax2.text(x, y + 0.1, f'{h}{k}{l}', fontsize=10, ha='center', fontweight='bold')
    
        ax2.scatter(0, 0, c='red', s=200, marker='o', label='Transmitted Beam (000)')
        ax2.set_xlabel('$k_x$ [1/Å]', fontsize=12)
        ax2.set_ylabel('$k_y$ [1/Å]', fontsize=12)
        ax2.set_title('SAED Pattern: [001] Zone Axis\n(Al FCC, 200 kV)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
    
        plt.tight_layout()
        plt.show()
    
        print(f"Electron wavelength: {lambda_A:.5f} Å")
        print(f"Ewald sphere radius: {k:.2f} 1/Å")
        print(f"Reciprocal lattice constant: {a_star:.3f} 1/Å")
    
    # Execute
    plot_ewald_sphere(a_lattice=4.05, voltage_kV=200, zone_axis=[0, 0, 1])
    

### 3.3.3 Indexing SAED Patterns

Procedure to identify crystal structure from SAED patterns:

  1. **Calibrate camera constant** : Determine $\lambda L$ with a known specimen (e.g., Au)
  2. **Measure diffraction spot spacing** : Distance $R$ from center (000) to each spot
  3. **Calculate interplanar spacing** : $d = \lambda L / R$
  4. **Identify Miller indices** : Compare calculated $d$ values with crystallographic data
  5. **Determine zone axis** : Determine zone axis $[uvw]$ from multiple diffraction spots

#### Code Example 3-4: SAED Pattern Indexing Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_d_spacing_fcc(a, h, k, l):
        """
        Calculate interplanar spacing for FCC crystal
    
        Parameters
        ----------
        a : float
            Lattice constant [Å]
        h, k, l : int
            Miller indices
    
        Returns
        -------
        d : float
            Interplanar spacing [Å]
        """
        d = a / np.sqrt(h**2 + k**2 + l**2)
        return d
    
    def index_saed_pattern(camera_length_mm=500, voltage_kV=200, crystal='Al'):
        """
        SAED pattern indexing simulation
    
        Parameters
        ----------
        camera_length_mm : float
            Camera length [mm]
        voltage_kV : float
            Accelerating voltage [kV]
        crystal : str
            Crystal type ('Al', 'Cu', 'Au')
        """
        # Electron wavelength calculation
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        h = 6.62607e-34
        E = voltage_kV * 1000 * e
        lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        lambda_pm = lambda_m * 1e12  # [pm]
    
        # Camera constant
        lambda_L = lambda_pm * camera_length_mm  # [pm·mm]
    
        # Lattice constant database
        lattice_constants = {
            'Al': 4.05,  # [Å]
            'Cu': 3.61,
            'Au': 4.08
        }
        a = lattice_constants[crystal]
    
        # FCC [001] zone axis allowed reflections
        reflections = [
            (0, 0, 0),  # Transmitted beam
            (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0),
            (2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0),
            (4, 0, 0), (0, 4, 0), (-4, 0, 0), (0, -4, 0)
        ]
    
        # Generate diffraction pattern
        fig, ax = plt.subplots(figsize=(10, 10))
    
        for (h, k, l) in reflections:
            if h == 0 and k == 0 and l == 0:
                # Transmitted beam
                ax.scatter(0, 0, c='red', s=300, marker='o', edgecolors='black', linewidths=2, zorder=10)
                ax.text(0, -5, '000', fontsize=12, ha='center', fontweight='bold', color='red')
                continue
    
            # Calculate interplanar spacing
            d = calculate_d_spacing_fcc(a, h, k, l)
    
            # Diffraction spot position (distance on screen) [mm]
            R_mm = lambda_L / (d * 100)  # Convert d [Å] → [pm]
    
            # Position in 2D pattern ([001] zone axis)
            x = h / np.sqrt(h**2 + k**2) * R_mm if h != 0 or k != 0 else 0
            y = k / np.sqrt(h**2 + k**2) * R_mm if h != 0 or k != 0 else 0
    
            # Simplification: Use h, k signs directly
            x = h * (lambda_L / (calculate_d_spacing_fcc(a, 2, 0, 0) * 100)) / 2
            y = k * (lambda_L / (calculate_d_spacing_fcc(a, 0, 2, 0) * 100)) / 2
    
            ax.scatter(x, y, c='blue', s=150, alpha=0.8, edgecolors='black', linewidths=1)
            ax.text(x, y - 2, f'{h}{k}{l}', fontsize=10, ha='center', fontweight='bold')
            ax.text(x, y + 2, f'd={d:.3f}Å', fontsize=8, ha='center', color='green')
    
        ax.set_xlabel('x [mm on screen]', fontsize=13)
        ax.set_ylabel('y [mm on screen]', fontsize=13)
        ax.set_title(f'Indexed SAED Pattern: {crystal} FCC [001]\n' +
                     f'(λL = {lambda_L:.2f} pm·mm, a = {a} Å)',
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
        # Scale bar
        ax.plot([25, 35], [-35, -35], 'k-', linewidth=3)
        ax.text(30, -38, '10 mm', fontsize=10, ha='center')
    
        plt.tight_layout()
        plt.show()
    
        # Output d-values of major reflections
        print(f"\n{crystal} FCC major reflections interplanar spacing:")
        print(f"  (200): d = {calculate_d_spacing_fcc(a, 2, 0, 0):.3f} Å")
        print(f"  (220): d = {calculate_d_spacing_fcc(a, 2, 2, 0):.3f} Å")
        print(f"  (400): d = {calculate_d_spacing_fcc(a, 4, 0, 0):.3f} Å")
    
    # Execute
    index_saed_pattern(camera_length_mm=500, voltage_kV=200, crystal='Al')
    

## 3.4 High-Resolution TEM (HRTEM)

### 3.4.1 Lattice Images and Structure Images

In High-Resolution TEM (HRTEM), interference between the transmitted beam and multiple diffracted beams forms lattice images.

**Differences between Lattice Images and HRTEM Images** :

  * **Lattice Fringes** : Formed by two-beam interference (transmitted beam + one diffracted beam). Specific lattice planes appear as fringe patterns
  * **HRTEM Images** : Formed by multi-beam interference (transmitted beam + many diffracted beams). Directly reflect atomic arrangements

**Cautions in Interpreting HRTEM Images** :

  * HRTEM images are **projections** of atomic arrangements, with thickness direction information superimposed
  * Due to defocus and Cs, the image may not correspond to atomic positions (black-white reversal)
  * Comparison with image simulation (multislice method, etc.) is essential

### 3.4.2 FFT Analysis of HRTEM Images

**Fast Fourier Transform (FFT)** of HRTEM images allows extraction of reciprocal lattice information. FFT patterns correspond to SAED patterns.

#### Code Example 3-5: HRTEM Image Generation and FFT Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift
    
    def generate_hrtem_image(size=512, lattice_a=4.05, zone_axis=[1, 1, 0], noise_level=0.1):
        """
        Simulate simplified HRTEM image (two-beam approximation)
    
        Parameters
        ----------
        size : int
            Image size [pixels]
        lattice_a : float
            Lattice constant [Å]
        zone_axis : list
            Zone axis [uvw]
        noise_level : float
            Noise level
    
        Returns
        -------
        image : ndarray
            HRTEM image
        """
        # Pixel size (Å/pixel)
        pixel_size = 0.1  # 0.1 Å/pixel → resolution equivalent to 0.2 Å
    
        # Coordinate grid
        x = np.arange(size) * pixel_size
        y = np.arange(size) * pixel_size
        X, Y = np.meshgrid(x, y)
    
        # For [110] zone axis, interference fringes from (111) and (-111) planes
        # Interplanar spacing (FCC)
        d_111 = lattice_a / np.sqrt(3)
    
        # Lattice fringe simulation
        k1 = 2 * np.pi / d_111
    
        # Superimpose lattice fringes in two directions
        fringe1 = np.cos(k1 * (X + Y) / np.sqrt(2))
        fringe2 = np.cos(k1 * (X - Y) / np.sqrt(2))
    
        # HRTEM image: simplified multi-beam interference model
        image = 0.5 + 0.3 * fringe1 + 0.3 * fringe2
    
        # Add noise
        image += np.random.normal(0, noise_level, image.shape)
    
        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
    
        return image, pixel_size
    
    def plot_hrtem_with_fft(image, pixel_size):
        """
        Plot HRTEM image and its FFT pattern
    
        Parameters
        ----------
        image : ndarray
            HRTEM image
        pixel_size : float
            Pixel size [Å/pixel]
        """
        # FFT calculation
        fft_image = fftshift(fft2(image))
        fft_magnitude = np.abs(fft_image)
        fft_magnitude_log = np.log(1 + fft_magnitude)  # Log scale
    
        # Frequency axis (1/Å)
        freq_x = np.fft.fftshift(np.fft.fftfreq(image.shape[1], d=pixel_size))
        freq_y = np.fft.fftshift(np.fft.fftfreq(image.shape[0], d=pixel_size))
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
        # HRTEM image (full)
        im0 = axes[0].imshow(image, cmap='gray', extent=[0, image.shape[1]*pixel_size,
                                                          0, image.shape[0]*pixel_size])
        axes[0].set_title('HRTEM Image (Full)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('x [Å]', fontsize=11)
        axes[0].set_ylabel('y [Å]', fontsize=11)
    
        # HRTEM image (zoomed)
        zoom_size = 50
        center = image.shape[0] // 2
        zoomed = image[center-zoom_size:center+zoom_size, center-zoom_size:center+zoom_size]
        extent_zoom = [0, zoom_size*2*pixel_size, 0, zoom_size*2*pixel_size]
    
        im1 = axes[1].imshow(zoomed, cmap='gray', extent=extent_zoom)
        axes[1].set_title('HRTEM Image (Zoomed)\nLattice Fringes Visible', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('x [Å]', fontsize=11)
        axes[1].set_ylabel('y [Å]', fontsize=11)
    
        # FFT pattern (log scale)
        im2 = axes[2].imshow(fft_magnitude_log, cmap='hot',
                            extent=[freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()])
        axes[2].set_title('FFT Pattern (Log Scale)\n(= Diffractogram)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('$k_x$ [1/Å]', fontsize=11)
        axes[2].set_ylabel('$k_y$ [1/Å]', fontsize=11)
        axes[2].set_xlim(-5, 5)
        axes[2].set_ylim(-5, 5)
    
        # Mark main peaks in FFT
        # Center (000)
        axes[2].scatter(0, 0, c='cyan', s=100, marker='o', edgecolors='white', linewidths=2)
    
        plt.tight_layout()
        plt.show()
    
    # Execute
    image, pixel_size = generate_hrtem_image(size=512, lattice_a=4.05, zone_axis=[1, 1, 0], noise_level=0.05)
    plot_hrtem_with_fft(image, pixel_size)
    
    print("FFT pattern interpretation:")
    print("  - Center (bright spot): Transmitted beam (000)")
    print("  - Symmetrical bright spots: Diffracted beams from lattice planes")
    print("  - Spot spacing ∝ 1/d (reciprocal of interplanar spacing)")
    

### 3.4.3 Aberration Correction Technology

In modern TEM, spherical aberration correctors (Cs-correctors) can bring the spherical aberration coefficient close to zero.

**Effects of Aberration Correction** :

  * **Resolution improvement** : 0.5 Å → 0.05 Å (atomic level)
  * **CTF improvement** : Achieve CTF with constant sign over wide spatial frequency range
  * **Reduced defocus dependence** : Scherzer focus constraint relaxed
  * **Simplified image interpretation** : Intuitive correspondence between atomic positions and images

#### Code Example 3-6: CTF Comparison Before and After Aberration Correction
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def ctf_comparison_corrected_vs_uncorrected():
        """
        CTF comparison before and after aberration correction
        """
        k = np.linspace(0, 15, 1000)  # Spatial frequency [1/Å]
    
        voltage = 200  # [kV]
        lambda_A = 0.0251  # Wavelength at 200 kV [Å]
        defocus_nm = -70  # [nm]
        defocus_A = defocus_nm * 10  # [Å]
    
        # Before aberration correction: Cs = 0.5 mm
        Cs_uncorrected_mm = 0.5
        Cs_uncorrected_A = Cs_uncorrected_mm * 1e7
    
        chi_uncorrected = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_uncorrected_A
        )
        ctf_uncorrected = np.sin(chi_uncorrected)
    
        # After aberration correction: Cs ≈ -0.01 mm (negative aberration for correction)
        Cs_corrected_mm = -0.01
        Cs_corrected_A = Cs_corrected_mm * 1e7
    
        chi_corrected = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_corrected_A
        )
        ctf_corrected = np.sin(chi_corrected)
    
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
        # Top panel: CTF comparison
        ax1.plot(k, ctf_uncorrected, 'b-', linewidth=2, label='Uncorrected (Cs = 0.5 mm)')
        ax1.plot(k, ctf_corrected, 'r-', linewidth=2, label='Corrected (Cs ≈ 0 mm)')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
        ax1.set_ylabel('CTF', fontsize=12)
        ax1.set_title('CTF: Aberration-Corrected vs Uncorrected TEM\n(200 kV, Δf = -70 nm)',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 15)
        ax1.set_ylim(-1, 1)
    
        # Annotation: Information transfer limit
        ax1.axvline(x=1/0.2, color='blue', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(1/0.2 + 0.3, 0.8, 'Uncorrected limit\n(~2 Å)', fontsize=10, color='blue')
    
        ax1.axvline(x=1/0.08, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(1/0.08 + 0.3, -0.8, 'Corrected limit\n(~0.8 Å)', fontsize=10, color='red')
    
        # Bottom panel: CTF phase shift χ(k)
        ax2.plot(k, chi_uncorrected, 'b-', linewidth=2, label='Uncorrected (Cs = 0.5 mm)')
        ax2.plot(k, chi_corrected, 'r-', linewidth=2, label='Corrected (Cs ≈ 0 mm)')
        ax2.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
        ax2.set_ylabel('Phase Shift χ(k) [rad]', fontsize=12)
        ax2.set_title('Phase Shift Function χ(k)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 15)
    
        plt.tight_layout()
        plt.show()
    
        # Resolution estimation
        # Frequency range up to first zero crossing is effective for information transfer
        k_zero_uncorrected = k[np.where(np.diff(np.sign(ctf_uncorrected)))[0][0]]
        k_zero_corrected = k[np.where(np.diff(np.sign(ctf_corrected)))[0][0]]
    
        resolution_uncorrected = 1 / k_zero_uncorrected
        resolution_corrected = 1 / k_zero_corrected
    
        print(f"Resolution before aberration correction: ~{resolution_uncorrected:.2f} Å")
        print(f"Resolution after aberration correction: ~{resolution_corrected:.2f} Å")
        print(f"Resolution improvement: {resolution_uncorrected / resolution_corrected:.1f}×")
    
    # Execute
    ctf_comparison_corrected_vs_uncorrected()
    

## 3.5 Exercises

### Exercise 3-1: CTF Optimization

**Problem** : Calculate the Scherzer focus for a 300 kV TEM (Cs = 1.0 mm) and determine the spatial frequency of the first zero crossing.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Calculate the Scherzer focus for a 300 kV TEM (Cs =
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Parameters
    voltage_kV = 300
    Cs_mm = 1.0
    
    # Electron wavelength calculation (relativistic correction)
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h = 6.62607e-34
    E = voltage_kV * 1000 * e
    lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_A = lambda_m * 1e10
    
    # Scherzer focus
    Cs_A = Cs_mm * 1e7
    scherzer_defocus_A = -1.2 * np.sqrt(Cs_A * lambda_A)
    scherzer_defocus_nm = scherzer_defocus_A / 10
    
    # First zero crossing spatial frequency (approximation)
    k_first_zero = 1.5 / (Cs_A**0.25 * lambda_A**0.75)
    resolution_A = 1 / k_first_zero
    
    print(f"Electron wavelength at 300 kV: {lambda_A:.5f} Å")
    print(f"Scherzer Focus: {scherzer_defocus_nm:.1f} nm")
    print(f"First zero crossing spatial frequency: {k_first_zero:.3f} 1/Å")
    print(f"Point resolution: {resolution_A:.3f} Å")
    

### Exercise 3-2: SAED Indexing

**Problem** : For a Cu FCC specimen (a = 3.61 Å) [011] zone axis SAED pattern, the 200 reflection spot was 15 mm away from the center. Determine the camera constant λL (accelerating voltage 200 kV).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: For a Cu FCC specimen (a = 3.61 Å) [011] zone axis 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Given data
    a = 3.61  # [Å]
    h, k, l = 2, 0, 0
    R_mm = 15  # [mm]
    
    # Interplanar spacing
    d_200 = a / np.sqrt(h**2 + k**2 + l**2)
    
    # Camera constant λL = R * d
    # Convert d to pm
    d_pm = d_200 * 100
    lambda_L = R_mm * d_pm
    
    print(f"Cu (200) plane interplanar spacing: {d_200:.3f} Å = {d_pm:.1f} pm")
    print(f"Camera constant λL: {lambda_L:.1f} pm·mm")
    
    # Verification: Theoretical value at 200 kV
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h_planck = 6.62607e-34
    E = 200 * 1000 * e
    lambda_m = h_planck / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_pm = lambda_m * 1e12
    
    # Assuming camera length 500 mm
    L_mm = lambda_L / lambda_pm
    print(f"Estimated camera length: {L_mm:.0f} mm")
    

### Exercise 3-3: FFT Analysis

**Problem** : In an FFT pattern of an HRTEM image, the nearest symmetrical spots from the center are 3.5 1/Å apart. Determine the interplanar spacing of the corresponding lattice planes.

**Show Answer**
    
    
    k = 3.5  # [1/Å] (FFT pattern spot position)
    
    # Interplanar spacing d = 1/k
    d = 1 / k
    
    print(f"FFT spot spatial frequency: {k} 1/Å")
    print(f"Corresponding interplanar spacing: {d:.3f} Å")
    print(f"This could correspond to, for example, Si (111) plane (d = 3.14 Å) or Al (111) plane (d = 2.34 Å)")
    

### Exercise 3-4: Dark Field Image Application

**Problem** : You want to observe Al2Cu precipitates (θ' phase) in an Al alloy using dark field imaging. Explain the strategy for selecting the appropriate diffraction spot.

**Show Answer**

**Strategy** :

  1. First, tilt the specimen to a low-index zone axis such as [001]_Al
  2. Obtain SAED pattern and identify diffraction spots from the Al matrix and θ' phase
  3. Select **diffraction spots unique to the θ' phase** (indices not present in Al)
  4. Examples: (002) or (100) reflections of the θ' phase
  5. Take dark field image with that spot → Only θ' precipitates will appear bright
  6. Using the Centered Dark Field (CDF) method will provide higher resolution images

### Exercise 3-5: Understanding Multi-Beam Interference

**Problem** : Explain the differences between two-beam interference (transmitted beam + one diffracted beam) and multi-beam interference (transmitted beam + multiple diffracted beams) from a CTF perspective.

**Show Answer**

**Answer** :

  * **Two-beam interference** : Only one spatial frequency component interferes. CTF is evaluated at that single point. Observed as lattice fringes, but details of atomic arrangement are not visible
  * **Multi-beam interference** : Many spatial frequency components interfere simultaneously. A wide range of CTF contributes to image formation. HRTEM images reflecting atomic arrangements are obtained
  * **Role of CTF** : Determines how much each spatial frequency component contributes to the image. With aberration correction achieving constant-sign CTF over a wide frequency range, more faithful atomic arrangement images can be obtained

### Exercise 3-6: Effects of Aberration Correction

**Problem** : Compare the point resolution of a 300 kV TEM before aberration correction (Cs = 1.0 mm) and after (Cs ≈ 0.001 mm).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Compare the point resolution of a 300 kV TEM before
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    voltage_kV = 300
    
    # Electron wavelength
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h = 6.62607e-34
    E = voltage_kV * 1000 * e
    lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_A = lambda_m * 1e10
    
    # Before aberration correction
    Cs_uncorrected_mm = 1.0
    Cs_uncorrected_A = Cs_uncorrected_mm * 1e7
    k_limit_uncorrected = 1.5 / (Cs_uncorrected_A**0.25 * lambda_A**0.75)
    resolution_uncorrected = 1 / k_limit_uncorrected
    
    # After aberration correction
    Cs_corrected_mm = 0.001
    Cs_corrected_A = Cs_corrected_mm * 1e7
    k_limit_corrected = 1.5 / (Cs_corrected_A**0.25 * lambda_A**0.75)
    resolution_corrected = 1 / k_limit_corrected
    
    print(f"Electron wavelength at 300 kV: {lambda_A:.5f} Å")
    print(f"\nBefore aberration correction (Cs = {Cs_uncorrected_mm} mm):")
    print(f"  Point resolution: {resolution_uncorrected:.3f} Å")
    print(f"\nAfter aberration correction (Cs = {Cs_corrected_mm} mm):")
    print(f"  Point resolution: {resolution_corrected:.3f} Å")
    print(f"\nResolution improvement: {resolution_uncorrected / resolution_corrected:.1f}×")
    

### Exercise 3-7: Practical HRTEM Analysis

**Problem** : Calculate the FFT of a given HRTEM image (512×512 pixels, pixel size 0.05 Å/pixel) and extract the interplanar spacings of three major lattice planes.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Calculate the FFT of a given HRTEM image (512×512 p
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift
    from scipy.ndimage import gaussian_filter
    
    # Generate dummy HRTEM image (replace with actual data)
    size = 512
    pixel_size = 0.05  # [Å/pixel]
    
    # Simulate crystal with three lattice planes
    x = np.arange(size) * pixel_size
    y = np.arange(size) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    # Lattice plane 1: d1 = 3.5 Å
    d1 = 3.5
    image = 0.5 + 0.2 * np.cos(2*np.pi * X / d1)
    
    # Lattice plane 2: d2 = 2.0 Å
    d2 = 2.0
    image += 0.15 * np.cos(2*np.pi * (X + Y) / (d2 * np.sqrt(2)))
    
    # Lattice plane 3: d3 = 1.5 Å
    d3 = 1.5
    image += 0.1 * np.cos(2*np.pi * Y / d3)
    
    # Add noise
    image += np.random.normal(0, 0.05, image.shape)
    image = gaussian_filter(image, sigma=0.5)
    
    # FFT calculation
    fft_image = fftshift(fft2(image))
    fft_magnitude = np.abs(fft_image)
    fft_magnitude_log = np.log(1 + fft_magnitude)
    
    # Frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    
    # FFT peak detection (simplified: manually extract three major peaks from center)
    center = size // 2
    # In actual analysis, use scipy.signal.find_peaks, etc.
    
    print("FFT analysis results (simulated data):")
    print(f"  Peak 1: k ≈ {1/d1:.3f} 1/Å → d ≈ {d1:.2f} Å")
    print(f"  Peak 2: k ≈ {1/d2:.3f} 1/Å → d ≈ {d2:.2f} Å")
    print(f"  Peak 3: k ≈ {1/d3:.3f} 1/Å → d ≈ {d3:.2f} Å")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('HRTEM Image', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(fft_magnitude_log, cmap='hot', extent=[freq.min(), freq.max(), freq.min(), freq.max()])
    ax2.set_title('FFT Pattern (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('$k_x$ [1/Å]', fontsize=11)
    ax2.set_ylabel('$k_y$ [1/Å]', fontsize=11)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()
    

### Exercise 3-8: Experimental Planning

**Problem** : Plan an experiment to analyze an unknown crystalline specimen using TEM. Explain in what order to acquire bright field images, dark field images, SAED, and HRTEM, and what to investigate with each.

**Show Answer**

**Experimental Plan** :

  1. **Low magnification bright field image** : Understand overall specimen morphology, thickness, and crystal grain size
  2. **SAED (wide area)** : Determine if polycrystalline or single crystal. For polycrystalline, obtain Debye rings
  3. **Specimen tilt and SAED** : Search for low-index zone axes ([001], [011], [111], etc.). Estimate lattice constant and crystal system from Debye ring analysis
  4. **SAED (selected area)** : Obtain SAED pattern from specific grains and index. Identify crystal structure
  5. **Dark field image** : Take dark field image with specific diffraction spot. Observe distribution of crystal grains, twins, and precipitates
  6. **HRTEM** : Obtain high-resolution images and precisely measure interplanar spacings through lattice imaging and FFT analysis. Observe defects (dislocations, stacking faults)
  7. **Data integration** : Integrate results from SAED, HRTEM, and dark field images to comprehensively analyze crystal structure, orientation, and defects

## 3.6 Learning Check

Confirm your understanding by answering the following questions:

  1. Can you explain the roles of the back focal plane (BFP) and image plane in TEM imaging?
  2. Can you describe the differences between bright field and dark field images and provide application examples for each?
  3. Do you understand the physical meaning of the Contrast Transfer Function (CTF)?
  4. Can you explain the purpose of Scherzer focus and how to calculate it?
  5. Can you explain the procedure for determining lattice constants from SAED patterns?
  6. Can you explain diffraction conditions using the Ewald sphere construction?
  7. Can you explain what can be determined from FFT analysis of HRTEM images?
  8. Do you understand the effects of aberration correction techniques and their impact on resolution?

## 3.7 References

  1. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - The definitive TEM textbook
  2. Kirkland, E. J. (2020). _Advanced Computing in Electron Microscopy_ (3rd ed.). Springer. - HRTEM image simulation and FFT analysis
  3. Pennycook, S. J., & Nellist, P. D. (Eds.). (2011). _Scanning Transmission Electron Microscopy: Imaging and Analysis_. Springer. - STEM techniques (detailed in next chapter)
  4. Spence, J. C. H. (2013). _High-Resolution Electron Microscopy_ (4th ed.). Oxford University Press. - Details of HRTEM theory
  5. Hawkes, P. W., & Spence, J. C. H. (Eds.). (2019). _Springer Handbook of Microscopy_. Springer. - Comprehensive handbook on electron microscopy techniques
  6. Reimer, L., & Kohl, H. (2008). _Transmission Electron Microscopy: Physics of Image Formation_ (5th ed.). Springer. - Details of TEM imaging theory
  7. Haider, M., et al. (1998). "Electron microscopy image enhanced." _Nature_ , 392, 768–769. - Breakthrough paper on aberration correction technology

## 3.8 Next Chapter

In the next chapter, we will learn the principles of Scanning Transmission Electron Microscopy (STEM), Z-contrast imaging (HAADF-STEM), Electron Energy Loss Spectroscopy (EELS), atomic resolution analysis, and fundamentals and applications of tomography. STEM is a powerful technique that can simultaneously detect various signals by scanning a convergent electron beam across the specimen.
