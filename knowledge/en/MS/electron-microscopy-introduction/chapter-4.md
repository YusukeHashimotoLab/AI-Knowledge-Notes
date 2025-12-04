---
title: "Chapter 4: STEM and Analytical Techniques"
chapter_title: "Chapter 4: STEM and Analytical Techniques"
subtitle: Z-contrast Imaging, EELS, Elemental Mapping, Atomic Resolution Analysis
reading_time: 25-35 minutes
difficulty: Intermediate to Advanced
code_examples: 7
version: 1.0
created_at: "by:"
---

Scanning Transmission Electron Microscopy (STEM) forms images by scanning a converged electron beam across a specimen and detecting transmitted and inelastically scattered electrons. In this chapter, we will learn the STEM principles, Z-contrast imaging (ADF/HAADF), annular bright field (ABF) imaging, electron energy loss spectroscopy (EELS), elemental mapping, atomic resolution analysis, and tomography fundamentals and applications, with practical quantitative analysis using Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand STEM imaging principles and differences from conventional TEM
  * ✅ Explain Z-contrast imaging (ADF/HAADF-STEM) formation mechanisms and atomic number dependence
  * ✅ Understand annular bright field (ABF) imaging principles and applications to light element observation
  * ✅ Master electron energy loss spectroscopy (EELS) principles and quantitative analysis methods
  * ✅ Perform spectral analysis (core-loss analysis, plasmon analysis)
  * ✅ Acquire and interpret STEM elemental mapping (EELS/EDS)
  * ✅ Understand electron tomography principles and 3D reconstruction fundamentals

## 4.1 STEM Principles and Detector Configuration

### 4.1.1 Differences Between STEM and TEM

STEM (Scanning Transmission Electron Microscopy) forms images by scanning a converged electron beam across a specimen and detecting transmitted and scattered electrons at each point.

Item | TEM (Transmission) | STEM (Scanning Transmission)  
---|---|---  
**Illumination** | Parallel beam (entire specimen) | Converged beam scanning (point-by-point)  
**Image Formation** | Lens-based imaging (image plane) | Detector signal scan synchronization (mapping)  
**Resolution** | Determined by objective lens aberrations | Determined by probe size  
**Detectors** | CCD camera, fluorescent screen | Annular detectors (ADF, HAADF, ABF), EELS, EDS  
**Simultaneous Signal Acquisition** | Difficult (image or diffraction) | Easy (multiple detectors in parallel)  
**Z-contrast** | Not directly achievable | Easily achieved with HAADF detector  
  
### 4.1.2 Types of STEM Detectors

In STEM, **annular detectors** that detect electrons by angle after specimen transmission play crucial roles.
    
    
    ```mermaid
    flowchart TD
        A[Converged Electron Beam] --> B[Specimen]
        B --> C{Scattering Angle}
        C -->|~0 mrad| D[BF DetectorBright Field]
        C -->|10-50 mrad| E[ABF DetectorAnnular BF]
        C -->|30-100 mrad| F[ADF DetectorAnnular DF]
        C -->|>50 mrad| G[HAADF DetectorHigh-Angle ADF]
        C -->|All angles| H[EELS Spectrometer]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#ffeb99,stroke:#ffa500
        style G fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style H fill:#99ccff,stroke:#0066cc
    ```

**Characteristics of Each Detector** :

  * **BF (Bright Field)** : Detects low-angle scattered electrons. Corresponds to conventional TEM bright field images. Primarily phase contrast
  * **ABF (Annular Bright Field)** : Annular bright field with central beam blocked. Can directly observe atomic positions of light elements (Li, H, O, etc.)
  * **ADF (Annular Dark Field)** : Detects mid-angle scattered electrons. Mass-thickness contrast
  * **HAADF (High-Angle ADF)** : Detects high-angle scattered electrons. Provides **Z-contrast** (atomic number-dependent contrast). Incoherent imaging
  * **EELS (Electron Energy Loss Spectroscopy)** : Spectrally analyzes energy-loss electrons. Element identification, chemical state, bandgap measurement

### 4.1.3 Probe Size and Current Trade-off

STEM resolution is determined by **probe size** , but making the probe smaller reduces current density and degrades S/N ratio.

$$ d_{\text{probe}} \approx 0.6 \frac{\lambda}{\alpha} $$ 

  * $d_{\text{probe}}$: Probe diameter
  * $\lambda$: Electron wavelength
  * $\alpha$: Convergence semi-angle (controlled by objective aperture)

**Practical Trade-offs** :

  * Atomic resolution (<1 Å): Convergence angle 20-30 mrad, probe current 50-200 pA
  * High S/N analysis (~2 Å): Convergence angle 10-15 mrad, probe current >500 pA

#### Code Example 4-1: STEM Detector Scattering Angle and Signal Intensity Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rutherford_scattering_cross_section(Z, theta, E_keV=200):
        """
        Rutherford scattering cross section (simplified model)
    
        Parameters
        ----------
        Z : int
            Atomic number
        theta : array-like
            Scattering angle [rad]
        E_keV : float
            Electron energy [keV]
    
        Returns
        -------
        sigma : ndarray
            Differential scattering cross section [arbitrary units]
        """
        # Simplification: High-angle scattering proportional to Z^2, low-angle is phase contrast
        # Rutherford scattering: dσ/dΩ ∝ Z^2 / (sin^4(θ/2))
    
        # Set minimum angle to avoid divergence in low-angle region
        theta = np.maximum(theta, 0.001)
    
        sigma = (Z**2) / (np.sin(theta / 2 + 1e-6)**4)
    
        return sigma
    
    def plot_stem_detector_signals():
        """
        Plot STEM detector angular dependence
        """
        # Scattering angle (mrad)
        theta_mrad = np.linspace(0.1, 200, 1000)
        theta_rad = theta_mrad * 1e-3
    
        # Scattering intensity for different atomic numbers
        elements = [('C', 6), ('Al', 13), ('Fe', 26), ('Au', 79)]
        colors = ['green', 'blue', 'orange', 'red']
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
        # Upper plot: Angular dependence of scattering intensity (log scale)
        for (elem, Z), color in zip(elements, colors):
            sigma = rutherford_scattering_cross_section(Z, theta_rad)
            ax1.plot(theta_mrad, sigma, color=color, linewidth=2, label=f'{elem} (Z={Z})')
    
        # Show detector ranges as bands
        ax1.axvspan(0, 10, alpha=0.2, color='cyan', label='BF (0-10 mrad)')
        ax1.axvspan(10, 50, alpha=0.2, color='lightgreen', label='ABF (10-50 mrad)')
        ax1.axvspan(50, 100, alpha=0.2, color='yellow', label='ADF (50-100 mrad)')
        ax1.axvspan(100, 200, alpha=0.2, color='pink', label='HAADF (>100 mrad)')
    
        ax1.set_xlabel('Scattering Angle [mrad]', fontsize=12)
        ax1.set_ylabel('Scattering Intensity [a.u.]', fontsize=12)
        ax1.set_title('STEM Detector Ranges and Scattering Intensity\n(Z-dependence)', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 200)
    
        # Lower plot: Integrated intensity at each detector (Z-dependence)
        Z_range = np.arange(1, 80)
        detector_ranges = {
            'BF': (0, 10),
            'ABF': (10, 50),
            'ADF': (50, 100),
            'HAADF': (100, 200)
        }
    
        for det_name, (theta_min, theta_max) in detector_ranges.items():
            intensities = []
            for Z in Z_range:
                theta_range = np.linspace(theta_min*1e-3, theta_max*1e-3, 100)
                sigma = rutherford_scattering_cross_section(Z, theta_range)
                # Integration (trapezoidal rule)
                intensity = np.trapz(sigma, theta_range)
                intensities.append(intensity)
    
            ax2.plot(Z_range, intensities, linewidth=2, marker='o', markersize=3, label=det_name)
    
        ax2.set_xlabel('Atomic Number Z', fontsize=12)
        ax2.set_ylabel('Integrated Signal Intensity [a.u.]', fontsize=12)
        ax2.set_title('Z-Dependence of STEM Detector Signals', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(1, 79)
    
        plt.tight_layout()
        plt.show()
    
        print("HAADF signal has near Z^2 dependence (Z-contrast)")
        print("Low-angle detectors (BF, ABF) also contain phase contrast components")
    
    # Execute
    plot_stem_detector_signals()
    

**Key Observations** :

  * HAADF signal detects high-angle scattering, providing strong signals from heavy elements (high Z)
  * Origin of Z-contrast is the $Z^2$ dependence of Rutherford scattering cross section
  * ABF detects mid-angle region where scattering intensity difference between light and heavy elements is relatively small, advantageous for light element observation

## 4.2 Z-contrast Imaging (HAADF-STEM)

### 4.2.1 Principles of Z-contrast Imaging

HAADF-STEM images detect high-angle scattered electrons (typically >50 mrad) with an annular detector. This scattering is primarily from **thermal diffuse scattering (TDS)** and inelastic scattering, with the following characteristics:

  * **Incoherent imaging** : No phase contrast, intuitive image interpretation
  * **Z^2 dependence** : Scattering intensity has near-square dependence on atomic number (Z-contrast)
  * **Thickness dependence** : Signal increases proportionally with specimen thickness
  * **Defocus independence** : Not affected by CTF (easy interpretation even without aberration correction)

HAADF signal intensity is approximated by:

$$ I_{\text{HAADF}} \propto Z^{1.7-2.0} \cdot t $$ 

Where $Z$ is the atomic number and $t$ is the specimen thickness.

### 4.2.2 Atomic Resolution Z-contrast Imaging

With aberration-corrected STEM reducing probe size below 1 Å, **atomic columns** can be directly observed.

**Application Examples** :

  * Atomic arrangement analysis at interfaces (semiconductor heterostructures, metal/ceramic junctions)
  * Dopant atom position identification (semiconductor devices)
  * Surface atomic structure of nanoparticles (catalysts)
  * Atomic-level analysis of grain boundaries

#### Code Example 4-2: Z-contrast Image Simulation (Atomic Column Intensity)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    def simulate_haadf_image(lattice_a=4.0, Z_matrix=13, Z_dopant=79, dopant_positions=None, size=256, pixel_size=0.1):
        """
        Simulate HAADF-STEM image (simplified model)
    
        Parameters
        ----------
        lattice_a : float
            Lattice constant [Å]
        Z_matrix : int
            Matrix atomic number
        Z_dopant : int
            Dopant atomic number
        dopant_positions : list of tuples
            Dopant positions [(x, y), ...] (in lattice units)
        size : int
            Image size [pixels]
        pixel_size : float
            Pixel size [Å/pixel]
    
        Returns
        -------
        image : ndarray
            HAADF image
        """
        image = np.zeros((size, size))
    
        # Generate lattice points (square lattice)
        grid_points_per_side = int(size * pixel_size / lattice_a)
    
        for i in range(grid_points_per_side):
            for j in range(grid_points_per_side):
                x_grid = i * lattice_a / pixel_size
                y_grid = j * lattice_a / pixel_size
    
                # Atomic column position (in pixels)
                x_px = int(x_grid)
                y_px = int(y_grid)
    
                if x_px < size and y_px < size:
                    # Determine atomic number (dopant or matrix)
                    Z = Z_matrix
                    if dopant_positions is not None:
                        for (dx, dy) in dopant_positions:
                            if abs(i - dx) < 0.5 and abs(j - dy) < 0.5:
                                Z = Z_dopant
                                break
    
                    # HAADF signal intensity ∝ Z^1.7
                    intensity = Z**1.7
    
                    # Represent atomic column with probe shape (Gaussian)
                    probe_sigma = 0.5  # [pixels] (probe size ~1 Å)
    
                    # Add Gaussian to image
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            px = x_px + dx
                            py = y_px + dy
                            if 0 <= px < size and 0 <= py < size:
                                r = np.sqrt(dx**2 + dy**2)
                                image[py, px] += intensity * np.exp(-r**2 / (2 * probe_sigma**2))
    
        # Add noise
        image += np.random.poisson(lam=10, size=image.shape)
    
        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
    
        return image
    
    # Run simulation
    # Al matrix (Z=13) with Au dopants (Z=79)
    dopant_positions = [(10, 10), (15, 15), (20, 12), (12, 20)]
    
    image = simulate_haadf_image(lattice_a=4.0, Z_matrix=13, Z_dopant=79,
                                  dopant_positions=dopant_positions, size=256, pixel_size=0.2)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # HAADF image (full view)
    im1 = ax1.imshow(image, cmap='gray', extent=[0, 256*0.2, 0, 256*0.2])
    ax1.set_title('HAADF-STEM Image\n(Al matrix + Au dopants)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x [Å]', fontsize=11)
    ax1.set_ylabel('y [Å]', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Intensity [a.u.]', fontsize=10)
    
    # Zoomed view (around dopant)
    zoom_center = (10, 10)
    zoom_size = 30
    zoomed = image[zoom_center[1]*5-zoom_size:zoom_center[1]*5+zoom_size,
                   zoom_center[0]*5-zoom_size:zoom_center[0]*5+zoom_size]
    
    im2 = ax2.imshow(zoomed, cmap='gray', extent=[0, zoom_size*2*0.2, 0, zoom_size*2*0.2])
    ax2.set_title('Zoomed View\n(Bright spot = Au dopant)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x [Å]', fontsize=11)
    ax2.set_ylabel('y [Å]', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Intensity [a.u.]', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Z-contrast: Au (Z=79) atomic columns appear brighter than Al (Z=13)")
    print("Intensity ratio: I(Au)/I(Al) ≈ (79/13)^1.7 ≈ 18x")
    

## 4.3 Annular Bright Field (ABF) Imaging and Light Element Observation

### 4.3.1 Principles of ABF Imaging

Annular Bright Field (ABF) imaging detects low to mid-angle scattered electrons (10-50 mrad) with an annular detector that blocks the central beam.

**ABF Characteristics** :

  * Light element (Li, H, O, N) atomic columns observed as **dark spots**
  * Can simultaneously observe light elements invisible in HAADF with heavy elements
  * Mixture of phase contrast and incoherent contrast
  * Relatively low defocus dependence

**Application Examples** :

  * Determining oxygen atom positions in oxides (perovskite, spinel structures)
  * Li distribution in lithium-ion battery materials
  * N atom observation in nitride semiconductors
  * H atom arrangement in hydrogen storage materials

#### Code Example 4-3: Simultaneous ABF and HAADF Image Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_abf_haadf_comparison():
        """
        Simultaneous observation simulation of ABF and HAADF images
        Assuming perovskite structure (heavy metal + oxygen)
        """
        size = 128
        pixel_size = 0.1  # [Å/pixel]
        lattice_a = 4.0  # [Å]
    
        image_haadf = np.zeros((size, size))
        image_abf = np.ones((size, size)) * 0.5  # ABF has intermediate background
    
        # Simplified perovskite structure model
        # Heavy metal (Sr, Ba, Pb etc. Z~80): Corner and body center
        # Oxygen (Z=8): face-centered
    
        grid_size = int(size * pixel_size / lattice_a)
    
        for i in range(grid_size):
            for j in range(grid_size):
                # Heavy metal site (corner)
                x_heavy = i * lattice_a / pixel_size
                y_heavy = j * lattice_a / pixel_size
    
                # Oxygen sites (face-centered)
                oxygen_sites = [
                    (x_heavy + lattice_a/(2*pixel_size), y_heavy),
                    (x_heavy, y_heavy + lattice_a/(2*pixel_size))
                ]
    
                # Plot heavy metal (HAADF: bright, ABF: dark)
                add_atom_column(image_haadf, x_heavy, y_heavy, Z=80, detector='HAADF')
                add_atom_column(image_abf, x_heavy, y_heavy, Z=80, detector='ABF')
    
                # Plot oxygen (HAADF: barely visible, ABF: dark spot)
                for (ox, oy) in oxygen_sites:
                    if ox < size and oy < size:
                        add_atom_column(image_haadf, ox, oy, Z=8, detector='HAADF')
                        add_atom_column(image_abf, ox, oy, Z=8, detector='ABF')
    
        # Add noise
        image_haadf += np.random.normal(0, 0.02, image_haadf.shape)
        image_abf += np.random.normal(0, 0.02, image_abf.shape)
    
        # Normalize
        image_haadf = np.clip(image_haadf, 0, 1)
        image_abf = np.clip(image_abf, 0, 1)
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
        # HAADF image
        im0 = axes[0].imshow(image_haadf, cmap='gray', extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[0].set_title('HAADF-STEM Image\n(Heavy atoms bright)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('x [Å]', fontsize=11)
        axes[0].set_ylabel('y [Å]', fontsize=11)
    
        # ABF image
        im1 = axes[1].imshow(image_abf, cmap='gray', extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[1].set_title('ABF-STEM Image\n(Both heavy and light atoms dark)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('x [Å]', fontsize=11)
        axes[1].set_ylabel('y [Å]', fontsize=11)
    
        # Overlay (color composite)
        # HAADF (red) + ABF inverted (green)
        overlay = np.zeros((size, size, 3))
        overlay[:, :, 0] = image_haadf  # Red: HAADF
        overlay[:, :, 1] = 1 - image_abf  # Green: ABF inverted
        overlay[:, :, 2] = 0
    
        im2 = axes[2].imshow(overlay, extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[2].set_title('Overlay: HAADF (Red) + ABF (Green)\n(Yellow = both present)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('x [Å]', fontsize=11)
        axes[2].set_ylabel('y [Å]', fontsize=11)
    
        plt.tight_layout()
        plt.show()
    
        print("HAADF: Only heavy elements appear bright")
        print("ABF: Both heavy and light elements (oxygen) observed as dark spots")
    
    def add_atom_column(image, x, y, Z, detector='HAADF'):
        """
        Add atomic column to image
    
        Parameters
        ----------
        image : ndarray
            Image array
        x, y : float
            Atom position [pixels]
        Z : int
            Atomic number
        detector : str
            'HAADF' or 'ABF'
        """
        size = image.shape[0]
        x_int = int(x)
        y_int = int(y)
    
        if detector == 'HAADF':
            # HAADF: Proportional to Z^1.7
            intensity = (Z / 80.0)**1.7 * 0.8
            sign = +1
        else:  # ABF
            # ABF: Darker at atom positions (negative contrast)
            intensity = (Z / 80.0)**0.5 * 0.3
            sign = -1
    
        probe_sigma = 0.5
    
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px = x_int + dx
                py = y_int + dy
                if 0 <= px < size and 0 <= py < size:
                    r = np.sqrt((px - x)**2 + (py - y)**2)
                    image[py, px] += sign * intensity * np.exp(-r**2 / (2 * probe_sigma**2))
    
    # Execute
    simulate_abf_haadf_comparison()
    

## 4.4 Electron Energy Loss Spectroscopy (EELS)

### 4.4.1 EELS Principles

Electron Energy Loss Spectroscopy (EELS) measures energy loss of electrons transmitted through a specimen, analyzing elemental composition, chemical bonding states, and bandgaps.

**EELS Spectrum Components** :

  * **Zero-loss peak** (0 eV): Elastically scattered electrons (no energy loss)
  * **Low-loss region** (0-50 eV): Plasmon excitation, interband transitions
  * **Core-loss region** (>50 eV): Inner shell electron excitation. Element-specific edges (K, L, M shells)

### 4.4.2 Core-loss Analysis and Element Quantification

Element concentration can be quantified from integrated intensity of core-loss edges:

$$ \frac{N_A}{N_B} = \frac{I_A(\Delta, \beta)}{I_B(\Delta, \beta)} \cdot \frac{\sigma_B(\Delta, \beta)}{\sigma_A(\Delta, \beta)} $$ 

  * $N_A, N_B$: Atomic number densities of elements A and B
  * $I_A, I_B$: Edge integrated intensities
  * $\sigma_A, \sigma_B$: Partial ionization cross sections (theoretical values or Hartree-Slater calculations)
  * $\Delta$: Integration window width, $\beta$: Collection semi-angle

**EELS Quantitative Analysis Procedure** :

  1. **Background removal** : Fit pre-edge region with power law ($AE^{-r}$)
  2. **Calculate integrated intensity** : Integrate over energy range from edge (typically 50-100 eV)
  3. **Cross section correction** : Automatic calculation in libraries like HyperSpy
  4. **Calculate concentration** : Compute element ratio using above equation

#### Code Example 4-4: EELS Spectrum Simulation and Quantitative Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def simulate_eels_spectrum(elements=['C', 'O', 'Fe'], concentrations=[0.3, 0.5, 0.2],
                                energy_range=(200, 800), noise_level=0.05):
        """
        Simulate EELS spectrum
    
        Parameters
        ----------
        elements : list
            Element list
        concentrations : list
            Relative concentration of each element
        energy_range : tuple
            Energy range [eV]
        noise_level : float
            Noise level
    
        Returns
        -------
        energy : ndarray
            Energy axis [eV]
        spectrum : ndarray
            Intensity
        """
        energy = np.linspace(energy_range[0], energy_range[1], 1000)
    
        # Typical core-loss edge energies (simplified)
        edge_energies = {
            'C': 284,   # C-K
            'N': 401,   # N-K
            'O': 532,   # O-K
            'Fe': 708,  # Fe-L3
            'Al': 1560, # Al-K
            'Si': 1839  # Si-K
        }
    
        # Background (power law)
        background = 1000 * (energy / energy[0])**(-3)
    
        spectrum = background.copy()
    
        # Add edges for each element
        for elem, conc in zip(elements, concentrations):
            if elem in edge_energies:
                edge_E = edge_energies[elem]
                if energy_range[0] < edge_E < energy_range[1]:
                    # Edge jump (step function + decay)
                    edge_mask = energy >= edge_E
                    edge_intensity = conc * 300 * np.exp(-(energy[edge_mask] - edge_E) / 100)
                    spectrum[edge_mask] += edge_intensity
    
        # Add noise (Poisson noise)
        spectrum += np.random.poisson(lam=noise_level*spectrum.mean(), size=spectrum.shape)
    
        return energy, spectrum
    
    def quantify_eels_edges(energy, spectrum, edges_dict):
        """
        Quantify elements from EELS spectrum
    
        Parameters
        ----------
        energy : ndarray
            Energy axis [eV]
        spectrum : ndarray
            Intensity
        edges_dict : dict
            {'Element': edge_energy}
    
        Returns
        -------
        results : dict
            {'Element': integrated_intensity}
        """
        results = {}
    
        for elem, edge_E in edges_dict.items():
            # Pre-edge region (background fit)
            pre_edge_mask = (energy >= edge_E - 50) & (energy < edge_E)
            if np.sum(pre_edge_mask) < 10:
                continue
    
            # Power law fit A * E^(-r)
            E_pre = energy[pre_edge_mask]
            I_pre = spectrum[pre_edge_mask]
    
            # Log transform for linear fit
            log_E = np.log(E_pre)
            log_I = np.log(I_pre + 1)  # Avoid division by zero
    
            coeffs = np.polyfit(log_E, log_I, 1)
            r = -coeffs[0]
            A = np.exp(coeffs[1])
    
            # Remove background
            background = A * energy**(-r)
            spectrum_bg_removed = spectrum - background
            spectrum_bg_removed = np.maximum(spectrum_bg_removed, 0)
    
            # Integrate post-edge region (50 eV window)
            post_edge_mask = (energy >= edge_E) & (energy < edge_E + 50)
            integrated_intensity = np.trapz(spectrum_bg_removed[post_edge_mask],
                                            energy[post_edge_mask])
    
            results[elem] = integrated_intensity
    
        return results
    
    # Run simulation
    elements = ['C', 'O', 'Fe']
    concentrations = [0.3, 0.5, 0.2]
    
    energy, spectrum = simulate_eels_spectrum(elements, concentrations,
                                              energy_range=(200, 800), noise_level=0.05)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Upper plot: Raw spectrum
    ax1.plot(energy, spectrum, 'b-', linewidth=1.5, label='Measured Spectrum')
    
    # Show edge positions
    edge_energies = {'C': 284, 'O': 532, 'Fe': 708}
    colors_edge = {'C': 'green', 'O': 'red', 'Fe': 'orange'}
    
    for elem, edge_E in edge_energies.items():
        if 200 < edge_E < 800:
            ax1.axvline(edge_E, color=colors_edge[elem], linestyle='--', linewidth=2, alpha=0.7,
                       label=f'{elem}-K edge ({edge_E} eV)')
    
    ax1.set_xlabel('Energy Loss [eV]', fontsize=12)
    ax1.set_ylabel('Intensity [counts]', fontsize=12)
    ax1.set_title('Simulated EELS Spectrum (C, O, Fe)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Lower plot: Quantification results
    quantified = quantify_eels_edges(energy, spectrum, edge_energies)
    
    elem_list = list(quantified.keys())
    intensities = list(quantified.values())
    
    # Normalize to relative concentration
    total = sum(intensities)
    relative_conc = [I / total for I in intensities]
    
    x_pos = np.arange(len(elem_list))
    bars = ax2.bar(x_pos, relative_conc, color=['green', 'red', 'orange'], alpha=0.7, edgecolor='black')
    
    # Plot true concentration (input values)
    true_conc = concentrations
    ax2.scatter(x_pos, true_conc, color='blue', s=150, marker='D', label='True Concentration', zorder=5)
    
    ax2.set_xlabel('Element', fontsize=12)
    ax2.set_ylabel('Relative Concentration', fontsize=12)
    ax2.set_title('EELS Quantification Results', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(elem_list, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuantification results:")
    for elem, conc in zip(elem_list, relative_conc):
        print(f"  {elem}: {conc:.2f}")
    

### 4.4.3 ELNES (Energy Loss Near Edge Structure)

Fine structure immediately after edges (Energy Loss Near Edge Structure, ELNES) reflects chemical bonding states and local atomic arrangements.

**ELNES Analysis Applications** :

  * Determining oxidation state (L3/L2 ratio of Fe2+ vs Fe3+)
  * Coordination number estimation (O 1s→2p transition shape)
  * Crystal field splitting observation (transition metal oxides)
  * Bandgap measurement (semiconductor materials)

## 4.5 STEM Elemental Mapping

### 4.5.1 Principles of EELS/EDS Mapping

By acquiring EELS or EDS spectra at each point while scanning the beam in STEM mode, **2D elemental distribution maps** can be created.

**EELS Mapping** :

  * **Advantages** : High spatial resolution (probe size limited), high sensitivity to light elements, simultaneous chemical state acquisition
  * **Disadvantages** : Requires thin specimens (<100 nm), large data volume, time-consuming measurement

**EDS Mapping** :

  * **Advantages** : Can measure thick specimens, simultaneous multi-element analysis, relatively fast measurement
  * **Disadvantages** : Somewhat lower spatial resolution (X-ray generation volume), difficult light element detection

#### Code Example 4-5: STEM Elemental Mapping Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    def simulate_stem_mapping(size=128, num_particles=10):
        """
        Simulate STEM elemental mapping (binary nanoparticles)
    
        Parameters
        ----------
        size : int
            Map size [pixels]
        num_particles : int
            Number of particles
    
        Returns
        -------
        map_Fe : ndarray
            Fe element map
        map_O : ndarray
            O element map
        haadf : ndarray
            HAADF image
        """
        # Fe3O4 nanoparticles dispersed on carbon substrate
        map_Fe = np.zeros((size, size))
        map_O = np.zeros((size, size))
        map_C = np.ones((size, size)) * 0.3  # Substrate C
    
        np.random.seed(42)
    
        for _ in range(num_particles):
            # Particle position and size
            x_center = np.random.randint(10, size - 10)
            y_center = np.random.randint(10, size - 10)
            radius = np.random.randint(5, 15)
    
            # Generate particle region
            y, x = np.ogrid[:size, :size]
            mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    
            # Fe3O4 stoichiometric ratio (Fe:O = 3:4)
            map_Fe[mask] = 0.75 + np.random.normal(0, 0.05, np.sum(mask))
            map_O[mask] = 1.0 + np.random.normal(0, 0.05, np.sum(mask))
            map_C[mask] = 0  # No C inside particles
    
        # Blur due to probe size
        map_Fe = gaussian_filter(map_Fe, sigma=1.0)
        map_O = gaussian_filter(map_O, sigma=1.0)
        map_C = gaussian_filter(map_C, sigma=1.0)
    
        # HAADF image (Z-contrast approximation)
        haadf = map_Fe * 26**1.7 + map_O * 8**1.7 + map_C * 6**1.7
        haadf = haadf / haadf.max()
    
        # Add noise
        map_Fe += np.random.normal(0, 0.02, map_Fe.shape)
        map_O += np.random.normal(0, 0.02, map_O.shape)
        map_C += np.random.normal(0, 0.02, map_C.shape)
        haadf += np.random.normal(0, 0.02, haadf.shape)
    
        # Clip
        map_Fe = np.clip(map_Fe, 0, 1)
        map_O = np.clip(map_O, 0, 1)
        map_C = np.clip(map_C, 0, 1)
        haadf = np.clip(haadf, 0, 1)
    
        return map_Fe, map_O, map_C, haadf
    
    # Run simulation
    map_Fe, map_O, map_C, haadf = simulate_stem_mapping(size=128, num_particles=8)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # HAADF image
    im0 = axes[0, 0].imshow(haadf, cmap='gray')
    axes[0, 0].set_title('HAADF-STEM Image', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Fe map
    im1 = axes[0, 1].imshow(map_Fe, cmap='Reds')
    axes[0, 1].set_title('Fe Elemental Map\n(EELS Fe-L edge)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='Fe Intensity')
    
    # O map
    im2 = axes[0, 2].imshow(map_O, cmap='Blues')
    axes[0, 2].set_title('O Elemental Map\n(EELS O-K edge)', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='O Intensity')
    
    # C map
    im3 = axes[1, 0].imshow(map_C, cmap='Greens')
    axes[1, 0].set_title('C Elemental Map\n(Substrate)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, label='C Intensity')
    
    # RGB composite (Fe=Red, O=Blue, C=Green)
    rgb_composite = np.stack([map_Fe, map_C, map_O], axis=2)
    rgb_composite = rgb_composite / rgb_composite.max()
    
    im4 = axes[1, 1].imshow(rgb_composite)
    axes[1, 1].set_title('RGB Composite\n(Fe:Red, O:Blue, C:Green)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Fe/O ratio map (stoichiometry)
    fe_o_ratio = np.divide(map_Fe, map_O + 1e-6)  # Avoid division by zero
    im5 = axes[1, 2].imshow(fe_o_ratio, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes[1, 2].set_title('Fe/O Ratio Map\n(Stoichiometry)', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')
    cbar5 = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    cbar5.set_label('Fe/O', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Fe3O4 nanoparticle mapping:")
    print("  - Distribution of Fe, O, C obtained simultaneously")
    print("  - Fe/O ratio map evaluates stoichiometric homogeneity")
    

## 4.6 Electron Tomography

### 4.6.1 Tomography Principles

Electron Tomography reconstructs three-dimensional structures by imaging specimens from various angles.

**Basic Procedure** :

  1. **Tilt series acquisition** : Acquire images at 1-2° increments while tilting specimen from approximately -70° to +70° (80-140 images)
  2. **Image alignment** : Register images at each tilt angle (using gold markers, etc.)
  3. **3D reconstruction** : Calculate 3D image using inverse Radon transform or iterative reconstruction based on projection theorem
  4. **Segmentation** : Extract specific structures (nanoparticles, pores, etc.) from 3D image
  5. **Quantitative analysis** : Calculate volume, surface area, shape, spatial distribution

**Projection Theorem** :

The Fourier transform of a projection $P_\theta(x', y')$ of a 3D object $f(x, y, z)$ from a certain direction corresponds to a cross-section of the 3D Fourier space in that direction.

#### Code Example 4-6: Electron Tomography Projection and Reconstruction Simulation (2D)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import radon, iradon
    from scipy.ndimage import rotate
    
    def create_test_object_2d(size=128):
        """
        Generate 2D test object (nanoparticles)
    
        Parameters
        ----------
        size : int
            Image size [pixels]
    
        Returns
        -------
        obj : ndarray
            Test object
        """
        obj = np.zeros((size, size))
    
        # Three circular particles
        particles = [
            (40, 40, 15),
            (80, 70, 20),
            (60, 90, 12)
        ]
    
        y, x = np.ogrid[:size, :size]
    
        for (cx, cy, r) in particles:
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            obj[mask] = 1.0
    
        return obj
    
    def simulate_tilt_series(obj, angles):
        """
        Simulate tilt series (Radon transform)
    
        Parameters
        ----------
        obj : ndarray
            2D object
        angles : array-like
            Projection angles [degrees]
    
        Returns
        -------
        sinogram : ndarray
            Sinogram (all projection data)
        """
        # Radon transform (projection)
        sinogram = radon(obj, theta=angles, circle=True)
    
        return sinogram
    
    # Generate test object
    obj_original = create_test_object_2d(size=128)
    
    # Acquire tilt series (-70° to +70°, 2° increments)
    angles = np.arange(-70, 71, 2)
    sinogram = simulate_tilt_series(obj_original, angles)
    
    # Add noise (realistic measurement)
    sinogram_noisy = sinogram + np.random.normal(0, 0.05*sinogram.max(), sinogram.shape)
    
    # 3D reconstruction (inverse Radon transform)
    reconstruction = iradon(sinogram_noisy, theta=angles, circle=True, filter_name='ramp')
    
    # Clip
    reconstruction = np.clip(reconstruction, 0, 1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original object
    im0 = axes[0, 0].imshow(obj_original, cmap='gray')
    axes[0, 0].set_title('Original Object\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Sinogram
    im1 = axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto',
                            extent=[angles.min(), angles.max(), 0, sinogram.shape[0]])
    axes[0, 1].set_title('Sinogram (Tilt Series)\nProjections at Different Angles', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Tilt Angle [degrees]', fontsize=11)
    axes[0, 1].set_ylabel('Projection Position', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Reconstructed image
    im2 = axes[1, 0].imshow(reconstruction, cmap='gray')
    axes[1, 0].set_title('Reconstructed Object\n(Filtered Back-Projection)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Error map
    error = np.abs(obj_original - reconstruction)
    im3 = axes[1, 1].imshow(error, cmap='hot')
    axes[1, 1].set_title('Reconstruction Error\n(|Original - Reconstructed|)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    cbar3.set_label('Error', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Quantitative evaluation
    mse = np.mean((obj_original - reconstruction)**2)
    print(f"Reconstruction mean squared error (MSE): {mse:.4f}")
    print(f"Projection angle range: {angles.min()}° to {angles.max()}°")
    print(f"Number of projections: {len(angles)}")
    

**Tomography Applications** :

  * 3D shape and surface structure analysis of catalyst nanoparticles
  * Pore size distribution and connectivity evaluation in porous materials
  * 3D microstructure observation of battery electrode materials
  * 3D reconstruction of biological cell organelles

## 4.7 Exercises

### Exercise 4-1: Z-contrast Intensity Ratio

**Problem** : Calculate the atomic column intensity ratio of Si (Z=14) and Ge (Z=32) in HAADF-STEM images (assuming Z^1.7 dependence).

**Show Answer**
    
    
    Z_Si = 14
    Z_Ge = 32
    
    I_ratio = (Z_Ge / Z_Si)**1.7
    
    print(f"Si atomic column intensity: I_Si ∝ {Z_Si}^1.7")
    print(f"Ge atomic column intensity: I_Ge ∝ {Z_Ge}^1.7")
    print(f"Intensity ratio I_Ge/I_Si: {I_ratio:.2f}")
    print("Ge appears approximately 6 times brighter than Si")
    

### Exercise 4-2: EELS Quantification

**Problem** : In an EELS spectrum of Al2O3, the Al-K integrated intensity is 1200 and O-K integrated intensity is 3000. Calculate the Al/O atomic number ratio when the partial ionization cross section ratio σ(O-K)/σ(Al-K) = 1.5.

**Show Answer**
    
    
    I_Al = 1200
    I_O = 3000
    sigma_ratio = 1.5  # σ(O) / σ(Al)
    
    # N_Al / N_O = (I_Al / I_O) * (σ_O / σ_Al)
    atom_ratio = (I_Al / I_O) * sigma_ratio
    
    print(f"Al integrated intensity: {I_Al}")
    print(f"O integrated intensity: {I_O}")
    print(f"Cross section ratio σ(O)/σ(Al): {sigma_ratio}")
    print(f"Atomic number ratio N_Al/N_O: {atom_ratio:.3f}")
    print(f"\nTheoretical value (Al2O3): 2/3 = 0.667")
    print(f"Measured value: {atom_ratio:.3f} → Good agreement")
    

### Exercise 4-3: ABF Image Applications

**Problem** : Explain why ABF imaging is effective for determining O atom positions in SrTiO3 perovskite structures.

**Show Answer**

**Reasons** :

  * In HAADF images, Sr (Z=38) and Ti (Z=22) appear bright, but O (Z=8) has weak scattering intensity and is difficult to detect
  * ABF imaging detects mid-angle region scattering, making the scattering intensity difference between heavy and light elements relatively small
  * O atomic columns are also clearly observed as dark spots
  * By simultaneously acquiring Sr and Ti positions (HAADF) and O positions (ABF), complete atomic arrangements can be determined

### Exercise 4-4: Plasmon Analysis

**Problem** : A plasmon peak was observed at 15 eV in an Al EELS spectrum. Calculate the free electron density (plasmon energy $E_p = \hbar\omega_p = \hbar\sqrt{ne^2/m_e\epsilon_0}$).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: A plasmon peak was observed at 15 eV in an Al EELS 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    E_p = 15  # [eV]
    e = 1.60218e-19  # [C]
    m_e = 9.10938e-31  # [kg]
    epsilon_0 = 8.85419e-12  # [F/m]
    hbar = 1.05457e-34  # [J·s]
    
    # E_p = hbar * omega_p
    omega_p = E_p * e / hbar  # [rad/s]
    
    # omega_p = sqrt(n * e^2 / (m_e * epsilon_0))
    n = (omega_p**2) * m_e * epsilon_0 / (e**2)
    
    print(f"Plasmon energy: {E_p} eV")
    print(f"Plasmon angular frequency: {omega_p:.3e} rad/s")
    print(f"Free electron density: {n:.3e} m^-3")
    print(f"              = {n/1e28:.2f} × 10^28 m^-3")
    
    # Al theoretical value (3 electrons/atom, lattice constant 4.05 Å)
    a = 4.05e-10  # [m]
    atoms_per_cell = 4  # FCC
    n_theory = (atoms_per_cell * 3) / a**3
    print(f"\nTheoretical value (Al FCC, 3 valence e/atom): {n_theory:.3e} m^-3")
    

### Exercise 4-5: Tomography Projection Number

**Problem** : Estimate the minimum number of projections needed to reconstruct a 50 nm diameter nanoparticle with 1 nm resolution using the Crowther criterion.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Estimate the minimum number of projections needed t
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    D = 50  # [nm] particle diameter
    resolution = 1  # [nm] target resolution
    
    # Crowther criterion: N >= π * D / resolution
    N_min = np.pi * D / resolution
    
    print(f"Particle diameter: {D} nm")
    print(f"Target resolution: {resolution} nm")
    print(f"Minimum projections by Crowther criterion: {N_min:.1f}")
    print(f"Practical recommendation: {int(np.ceil(N_min * 1.5))} projections or more")
    print(f"\nFor tilt range -70° to +70° (140°) with 2° increments: 70 projections")
    print(f"With 1° increments: 140 projections → Sufficient")
    

### Exercise 4-6: STEM Detector Optimization

**Problem** : What detector combination is optimal for simultaneously observing light elements (C, N, O) and heavy elements (Pt)? Answer with reasoning.

**Show Answer**

**Optimal Combination** :

  * **HAADF + ABF Simultaneous Acquisition**

**Reasoning** :

  * **HAADF** : Observes Pt (Z=78) positions with high contrast. Z^1.7 dependence makes light elements nearly invisible
  * **ABF** : Detects C, N, O as dark spots. Pt also observable
  * **Simultaneous acquisition** : STEM can acquire signals from multiple detectors in parallel during one scan, making it efficient
  * **Complementary information** : HAADF captures structure skeleton (heavy elements), ABF supplements light element positions

### Exercise 4-7: EELS Thickness Measurement

**Problem** : EELS zero-loss peak integrated intensity is 10000 and total spectrum integrated intensity is 15000. Estimate specimen thickness t when mean free path λ is 100 nm ($I_{\text{total}}/I_0 = \exp(t/\lambda)$).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: EELS zero-loss peak integrated intensity is 10000 a
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    I_zero_loss = 10000
    I_total = 15000
    lambda_mfp = 100  # [nm] mean free path
    
    # I_total / I_zero_loss = exp(t / lambda)
    ratio = I_total / I_zero_loss
    t = lambda_mfp * np.log(ratio)
    
    print(f"Zero-loss peak integrated: {I_zero_loss}")
    print(f"Total integrated intensity: {I_total}")
    print(f"Intensity ratio: {ratio:.3f}")
    print(f"Mean free path: {lambda_mfp} nm")
    print(f"Specimen thickness: {t:.1f} nm")
    
    if t < 50:
        print("→ Thin specimen (suitable for EELS quantification)")
    elif t < 100:
        print("→ Moderate thickness (multiple scattering effects present)")
    else:
        print("→ Thick specimen (EELS quantification difficult, needs thinning)")
    

### Exercise 4-8: Practical Exercise

**Problem** : Develop a STEM-EELS analysis plan for Fe-Cr-Ni stainless steel. Include signals to measure, expected challenges, and countermeasures.

**Show Answer**

**Analysis Plan** :

  1. **Specimen preparation** : 
     * FIB (Focused Ion Beam) thinning (target thickness <50 nm)
     * Low-energy ion polishing to remove surface damage
  2. **HAADF-STEM observation** : 
     * Morphology observation of grains, precipitates, interfaces
     * Qualitative evaluation of Cr, Fe concentration changes using Z-contrast
  3. **EELS mapping** : 
     * Map Fe-L2,3 (708 eV), Cr-L2,3 (575 eV), Ni-L2,3 (855 eV)
     * Dual EELS (simultaneous low-loss + core-loss) for thickness correction
  4. **Quantitative analysis** : 
     * Extract integrated intensity for each edge using HyperSpy
     * Quantify concentration using Hartree-Slater cross sections

**Expected Challenges and Countermeasures** :

  * **Challenge 1** : Fe, Cr edges close together (708 vs 575 eV) → **Countermeasure** : High energy resolution setting (<1 eV), peak separation
  * **Challenge 2** : Specimen thickness inhomogeneity → **Countermeasure** : Estimate thickness at each point from low-loss spectrum and correct
  * **Challenge 3** : Beam damage (especially at low acceleration voltage) → **Countermeasure** : Use liquid nitrogen cooling holder, minimize beam current

## 4.8 Learning Check

Answer the following questions to confirm your understanding:

  1. Can you explain the differences in image formation principles between STEM and TEM?
  2. Do you understand the physical origin of Z-contrast characteristics in HAADF-STEM images?
  3. Can you explain why ABF imaging is suitable for light element observation?
  4. Do you understand the EELS spectrum components (zero-loss, low-loss, core-loss)?
  5. Can you execute the EELS quantitative analysis procedure (background removal, integration, cross section correction)?
  6. Can you determine when to use EELS versus EDS for STEM elemental mapping?
  7. Do you understand the projection theorem and 3D reconstruction principles in electron tomography?

## 4.9 References

  1. Pennycook, S. J., & Nellist, P. D. (Eds.). (2011). _Scanning Transmission Electron Microscopy: Imaging and Analysis_. Springer. - Comprehensive STEM technology textbook
  2. Egerton, R. F. (2011). _Electron Energy-Loss Spectroscopy in the Electron Microscope_ (3rd ed.). Springer. - EELS analysis bible
  3. Findlay, S. D., et al. (2010). "Robust atomic resolution imaging of light elements using scanning transmission electron microscopy." _Applied Physics Letters_ , 95, 191913. - ABF imaging principle paper
  4. Muller, D. A. (2009). "Structure and bonding at the atomic scale by scanning transmission electron microscopy." _Nature Materials_ , 8, 263-270. - Atomic resolution STEM analysis
  5. de Jonge, N., & Ross, F. M. (2011). "Electron microscopy of specimens in liquid." _Nature Nanotechnology_ , 6, 695-704. - Liquid-phase STEM observation
  6. Midgley, P. A., & Dunin-Borkowski, R. E. (2009). "Electron tomography and holography in materials science." _Nature Materials_ , 8, 271-280. - Electron tomography applications
  7. Krivanek, O. L., et al. (2010). "Atom-by-atom structural and chemical analysis by annular dark-field electron microscopy." _Nature_ , 464, 571-574. - Single atom analysis STEM

## 4.10 Next Chapter

In the next chapter, we will practice integrated analysis of EDS, EELS, and EBSD data using Python. We will learn spectrum processing with the HyperSpy library, phase classification using machine learning, EBSD orientation analysis, and troubleshooting to build actual materials analysis workflows.
