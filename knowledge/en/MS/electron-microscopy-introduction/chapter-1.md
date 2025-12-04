---
title: "Chapter 1: Fundamentals of Electron Microscopy"
chapter_title: "Chapter 1: Fundamentals of Electron Microscopy"
subtitle: Electron-Matter Interactions, Resolution Theory, and Principles of Electron Microscopy
reading_time: 20-30 minutes
difficulty: Beginner to Intermediate
code_examples: 7
version: 1.0
created_at: "by:"
---

Electron microscopy is a powerful tool that can observe atomic and molecular-level structures, surpassing the resolution limits of optical microscopy. In this chapter, we will learn about electron-matter interactions, the relationship between electron wavelength and resolution, and the fundamental principles of electron microscopy, practicing calculations of electron wavelength, resolution simulation, and scattering cross-section computations using Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the wave-particle duality of electron waves and calculate de Broglie wavelength
  * ✅ Explain the definition of resolution by Rayleigh criterion and the principles of resolution improvement in electron microscopy
  * ✅ Understand the differences between elastic and inelastic scattering and the applications of each scattering signal
  * ✅ Quantitatively evaluate the effects of acceleration voltage on electron wavelength and resolution
  * ✅ Explain the roles of electron microscope components (electron gun, lenses, detectors)
  * ✅ Calculate scattering intensity, contrast transfer function, and signal intensity distribution using Python
  * ✅ Explain the essential differences between optical and electron microscopy from a wavelength perspective

## 1.1 Fundamentals of Electron Waves

### 1.1.1 Wave-Particle Duality

Electrons are quantum mechanical particles possessing both **wave properties** and **particle properties**. Louis de Broglie (1924) proposed that a particle with momentum $p$ has a wavelength $\lambda$:

$$ \lambda = \frac{h}{p} $$ 

where $h = 6.626 \times 10^{-34}$ J·s (Planck's constant), $p = mv$ (momentum: mass $m$ × velocity $v$).

**Relativistic correction** : When electrons are accelerated to high speeds, relativistic effects must be considered. The energy of an electron accelerated by voltage $V$ is:

$$ E = eV $$ 

where $e = 1.602 \times 10^{-19}$ C (elementary charge). The relativistic momentum is calculated by:

$$ p = \sqrt{\frac{2m_0 eV}{c^2}\left(1 + \frac{eV}{2m_0 c^2}\right)} $$ 

$m_0 = 9.109 \times 10^{-31}$ kg (electron rest mass), $c = 2.998 \times 10^8$ m/s (speed of light).

#### Code Example 1-1: Calculation of Electron Wavelength (with Relativistic Correction)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_electron_wavelength(voltage_kV):
        """
        Calculate electron wavelength from acceleration voltage (with relativistic correction)
    
        Parameters
        ----------
        voltage_kV : float or array-like
            Acceleration voltage [kV]
    
        Returns
        -------
        wavelength_pm : float or array-like
            Electron wavelength [pm]
        """
        # Physical constants
        h = 6.62607e-34    # Planck constant [J·s]
        m0 = 9.10938e-31   # Electron mass [kg]
        e = 1.60218e-19    # Elementary charge [C]
        c = 2.99792e8      # Speed of light [m/s]
    
        # Convert acceleration voltage to J
        V = voltage_kV * 1000  # [V]
        E = e * V              # Energy [J]
    
        # Relativistic momentum
        p = np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    
        # de Broglie wavelength
        wavelength_m = h / p
        wavelength_pm = wavelength_m * 1e12  # [pm]
    
        return wavelength_pm
    
    # Acceleration voltage range
    voltages = np.array([10, 50, 100, 200, 300, 500, 1000])  # [kV]
    wavelengths = calculate_electron_wavelength(voltages)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Wavelength vs acceleration voltage
    ax1.plot(voltages, wavelengths, 'o-', linewidth=2, markersize=8, color='#f093fb')
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Electron Wavelength [pm]', fontsize=12)
    ax1.set_title('Electron Wavelength vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Add visible light wavelength as reference line
    ax1.axhline(y=500000, color='orange', linestyle='--', linewidth=2, label='Visible Light (~500 nm)')
    ax1.legend(fontsize=10)
    
    # Right plot: Wavelength comparison at representative acceleration voltages
    selected_voltages = [100, 200, 300]
    selected_wavelengths = calculate_electron_wavelength(selected_voltages)
    
    ax2.bar([f'{v} kV' for v in selected_voltages], selected_wavelengths,
            color=['#f093fb', '#f5576c', '#d07be8'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Electron Wavelength [pm]', fontsize=12)
    ax2.set_title('Typical Operating Voltages', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (v, wl) in enumerate(zip(selected_voltages, selected_wavelengths)):
        ax2.text(i, wl + 0.1, f'{wl:.3f} pm', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Output specific numerical values
    print("Correspondence between acceleration voltage and electron wavelength:")
    for v, wl in zip(voltages, wavelengths):
        print(f"  {v:4.0f} kV → λ = {wl:.4f} pm = {wl/100:.4f} Å")
    print(f"\nWavelength ratio of 200 kV electron wave to visible light (550 nm): {550000 / wavelengths[3]:.0f}:1")
    

**Output interpretation** :

  * Wavelength of approximately 3.7 pm at 100 kV, 2.5 pm at 200 kV, 2.0 pm at 300 kV
  * **Approximately 200,000 times shorter wavelength** compared to visible light (~500 nm)
  * Higher acceleration voltage leads to shorter wavelength and improved resolution

### 1.1.2 Resolution and Rayleigh Criterion

The **resolution** of a microscope is defined as the minimum distance at which two points can be distinguished. According to the Rayleigh criterion:

$$ d = \frac{0.61 \lambda}{n \sin\alpha} $$ 

where:

  * $d$: resolution (minimum distinguishable distance)
  * $\lambda$: wavelength
  * $n \sin\alpha$: numerical aperture (NA)
  * $\alpha$: half-angle of objective lens

**Optical Microscopy vs Electron Microscopy** :

Parameter | Optical Microscopy | Electron Microscopy (200 kV)  
---|---|---  
Wavelength $\lambda$ | ~500 nm (visible light) | ~2.5 pm (electron wave)  
Resolution $d$ | ~200 nm (theoretical limit) | ~0.05 nm (aberration-corrected TEM)  
Magnification | ~2,000× | ~50,000,000×  
  
#### Code Example 1-2: Resolution Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def resolution_rayleigh(wavelength_nm, alpha_deg, n=1.0):
        """
        Calculate resolution by Rayleigh criterion
    
        Parameters
        ----------
        wavelength_nm : float
            Wavelength [nm]
        alpha_deg : float
            Objective lens half-angle [degrees]
        n : float
            Refractive index (1.0 in vacuum for electron microscopy)
    
        Returns
        -------
        d_nm : float
            Resolution [nm]
        """
        alpha_rad = np.deg2rad(alpha_deg)
        d_nm = 0.61 * wavelength_nm / (n * np.sin(alpha_rad))
        return d_nm
    
    # Comparison of optical and electron microscopy
    alpha_range = np.linspace(0.5, 30, 100)  # [degrees]
    
    # Optical microscopy (visible light)
    optical_wavelength = 550  # [nm]
    optical_resolution = resolution_rayleigh(optical_wavelength, alpha_range)
    
    # Electron microscopy (100 kV → 3.7 pm = 0.0037 nm)
    em_100kV_wavelength = 0.0037  # [nm]
    em_100kV_resolution = resolution_rayleigh(em_100kV_wavelength, alpha_range)
    
    # Electron microscopy (200 kV → 2.5 pm = 0.0025 nm)
    em_200kV_wavelength = 0.0025  # [nm]
    em_200kV_resolution = resolution_rayleigh(em_200kV_wavelength, alpha_range)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(alpha_range, optical_resolution, linewidth=2.5, label='Optical (550 nm)', color='orange')
    ax.plot(alpha_range, em_100kV_resolution, linewidth=2.5, label='EM 100 kV (3.7 pm)', color='#f093fb')
    ax.plot(alpha_range, em_200kV_resolution, linewidth=2.5, label='EM 200 kV (2.5 pm)', color='#f5576c')
    
    # Practical resolution limit of optical microscopy
    ax.axhline(y=200, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Optical Limit (~200 nm)')
    
    # Practical resolution of electron microscopy (effects of spherical aberration, etc.)
    ax.axhline(y=0.1, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Practical EM Limit (~0.1 nm)')
    
    ax.set_xlabel('Objective Lens Half-Angle α [degrees]', fontsize=12)
    ax.set_ylabel('Resolution d [nm]', fontsize=12)
    ax.set_title('Resolution vs Lens Aperture (Rayleigh Criterion)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e3)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate practical values
    alpha_typical = 10  # [degrees] (typical objective lens half-angle)
    print(f"Resolution at objective lens half-angle α = {alpha_typical}°:")
    print(f"  Optical microscopy: {resolution_rayleigh(optical_wavelength, alpha_typical):.1f} nm")
    print(f"  Electron microscopy (100 kV): {resolution_rayleigh(em_100kV_wavelength, alpha_typical):.4f} nm")
    print(f"  Electron microscopy (200 kV): {resolution_rayleigh(em_200kV_wavelength, alpha_typical):.4f} nm")
    

## 1.2 Electron-Matter Interactions

### 1.2.1 Types of Scattering

When an electron beam enters a specimen, various **scattering phenomena** occur due to interactions with atomic nuclei and electrons.
    
    
    ```mermaid
    flowchart TD
        A[Incident Electron Beam] --> B{Interactions in Specimen}
        B --> C[Elastic ScatteringElastic Scattering]
        B --> D[Inelastic ScatteringInelastic Scattering]
    
        C --> E[Backscattered ElectronsBSE]
        C --> F[Transmitted ElectronsTEM Signal]
    
        D --> G[Secondary ElectronsSE]
        D --> H[X-raysEDS/WDS]
        D --> I[Auger ElectronsAES]
        D --> J[Energy Loss ElectronsEELS]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style D fill:#ffeb99,stroke:#ffa500,stroke-width:2px
    ```

**Elastic Scattering** :

  * Electron energy loss is nearly zero ($\Delta E \approx 0$)
  * Scattering by Coulomb potential of atomic nuclei
  * Scattering angle depends on atomic number $Z$ (heavier elements scatter at larger angles)
  * **Applications** : TEM diffraction, HRTEM, HAADF-STEM (Z-contrast imaging)

**Inelastic Scattering** :

  * Electrons lose energy ($\Delta E > 0$)
  * Inner-shell electron excitation, plasmon excitation, phonon excitation, etc.
  * **Applications** : EDS (elemental analysis), EELS (electronic state analysis), SE (surface morphology observation)

### 1.2.2 Scattering Cross Section and Atomic Number Dependence

Scattering intensity is represented by the **scattering cross section** $\sigma$. In the Rutherford scattering approximation:

$$ \frac{d\sigma}{d\Omega} = \left(\frac{Ze^2}{4\pi\epsilon_0 E}\right)^2 \frac{1}{\sin^4(\theta/2)} $$ 

where $Z$ is the atomic number, $E$ is the electron energy, and $\theta$ is the scattering angle.

#### Code Example 1-3: Atomic Number Dependence of Scattering Cross Section
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rutherford_scattering_cross_section(Z, E_keV, theta_deg):
        """
        Calculate Rutherford scattering cross section
    
        Parameters
        ----------
        Z : int or array-like
            Atomic number
        E_keV : float
            Electron energy [keV]
        theta_deg : float
            Scattering angle [degrees]
    
        Returns
        -------
        cross_section : float or array-like
            Differential scattering cross section [nm²/sr]
        """
        # Constants
        e = 1.60218e-19      # [C]
        epsilon_0 = 8.85419e-12  # [F/m]
        a0 = 0.0529          # Bohr radius [nm]
    
        # Convert energy to J
        E_J = E_keV * 1000 * e
    
        # Convert scattering angle to radians
        theta_rad = np.deg2rad(theta_deg)
    
        # Rutherford scattering cross section (simplified formula)
        # dσ/dΩ ∝ Z² / (E² sin⁴(θ/2))
        Z = np.asarray(Z)
        cross_section = (Z / E_keV)**2 / (np.sin(theta_rad / 2)**4)
    
        # Normalization factor (arbitrary units)
        cross_section = cross_section * 1e-3
    
        return cross_section
    
    # Representative elements
    elements = {
        'C': 6, 'Al': 13, 'Si': 14, 'Ti': 22,
        'Fe': 26, 'Cu': 29, 'Mo': 42, 'W': 74, 'Pt': 78, 'Au': 79
    }
    
    E_keV = 200  # 200 kV
    theta_range = np.linspace(1, 30, 100)  # [degrees]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Angular dependence
    selected_elements = ['C', 'Si', 'Fe', 'Au']
    colors = ['#d3d3d3', '#f093fb', '#ffa500', '#ffd700']
    
    for elem, color in zip(selected_elements, colors):
        Z = elements[elem]
        cs = rutherford_scattering_cross_section(Z, E_keV, theta_range)
        ax1.plot(theta_range, cs, linewidth=2.5, label=f'{elem} (Z={Z})', color=color)
    
    ax1.set_xlabel('Scattering Angle θ [degrees]', fontsize=12)
    ax1.set_ylabel('d$\sigma$/d$\Omega$ [arb. units]', fontsize=12)
    ax1.set_title(f'Scattering Cross Section vs Angle\n(E = {E_keV} keV)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2, 1e4)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # Right plot: Atomic number dependence (fixed scattering angle)
    Z_values = np.array(list(elements.values()))
    element_names = list(elements.keys())
    theta_fixed = 10  # [degrees]
    
    cs_Z = rutherford_scattering_cross_section(Z_values, E_keV, theta_fixed)
    
    ax2.scatter(Z_values, cs_Z, s=150, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=1.5, zorder=3)
    ax2.plot(Z_values, cs_Z, 'k--', linewidth=1, alpha=0.5, zorder=1)
    
    for i, (Z, cs, name) in enumerate(zip(Z_values, cs_Z, element_names)):
        ax2.text(Z, cs * 1.2, name, fontsize=9, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Number Z', fontsize=12)
    ax2.set_ylabel('d$\sigma$/d$\Omega$ [arb. units]', fontsize=12)
    ax2.set_title(f'Scattering Cross Section vs Z\n(θ = {theta_fixed}°, E = {E_keV} keV)',
                  fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Atomic number dependence of scattering cross section (θ = {theta_fixed}°):")
    print(f"  C (Z=6):   {rutherford_scattering_cross_section(6, E_keV, theta_fixed):.3f}")
    print(f"  Si (Z=14): {rutherford_scattering_cross_section(14, E_keV, theta_fixed):.3f}")
    print(f"  Fe (Z=26): {rutherford_scattering_cross_section(26, E_keV, theta_fixed):.3f}")
    print(f"  Au (Z=79): {rutherford_scattering_cross_section(79, E_keV, theta_fixed):.3f}")
    print(f"\nAu scattering cross section is approximately {(79/6)**2:.0f} times that of C (Z² dependence)")
    

**Important observations** :

  * Scattering cross section is proportional to $Z^2$ → heavier elements scatter more strongly
  * Small-angle scattering dominates (proportional to inverse of $\sin^4(\theta/2)$)
  * HAADF-STEM (high-angle annular dark field) detects large-angle scattering to obtain Z-contrast images

## 1.3 Components of Electron Microscopy

### 1.3.1 Electron Gun

The electron gun is a device that generates a stable electron beam. Main types:

Type | Principle | Brightness | Energy Spread | Applications  
---|---|---|---|---  
**Tungsten Thermionic** | Emission by heating | Low (~10⁵ A/cm²·sr) | ~2-3 eV | General-purpose SEM, low cost  
**LaB₆ Thermionic** | LaB₆ filament heating | Medium (~10⁶ A/cm²·sr) | ~1-2 eV | General-purpose TEM, balanced  
**Field Emission Gun (FEG)** | Quantum tunneling by strong electric field | High (~10⁸ A/cm²·sr) | ~0.3-0.7 eV | High-resolution TEM/STEM, EELS  
  
**Advantages of Field Emission Gun (FEG)** :

  * High brightness → enables fine probes (below 0.1 nm)
  * Narrow energy spread → high energy resolution in EELS
  * High coherence → high resolution in HRTEM

### 1.3.2 Electron Lenses and Aberrations

Electron lenses focus the electron beam using magnetic fields (electromagnetic lenses) or electrostatic fields.

**Major aberrations** :

  * **Spherical Aberration ($C_s$)** : Electrons passing through lens edges deviate from focus
  * **Chromatic Aberration ($C_c$)** : Focus shift due to energy spread
  * **Astigmatism** : Image distortion due to lens asymmetry

Modern TEMs use **aberration correctors (Cs-correctors)** to correct spherical aberration to nearly zero, achieving resolutions below 0.05 nm.

#### Code Example 1-4: Basics of Contrast Transfer Function (CTF)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def ctf_simple(k, wavelength_A, defocus_nm, Cs_mm):
        """
        Simplified contrast transfer function (CTF)
    
        Parameters
        ----------
        k : array-like
            Spatial frequency [1/Å]
        wavelength_A : float
            Electron wavelength [Å]
        defocus_nm : float
            Defocus [nm] (negative: underfocus)
        Cs_mm : float
            Spherical aberration coefficient [mm]
    
        Returns
        -------
        ctf : array-like
            CTF values
        """
        # Unit conversion
        defocus_A = defocus_nm * 10
        Cs_A = Cs_mm * 1e7
    
        # Phase shift χ(k)
        chi = (np.pi * wavelength_A / 2) * k**2 * (defocus_A + 0.5 * wavelength_A**2 * k**2 * Cs_A)
    
        # CTF = sin(χ)
        ctf = np.sin(chi)
    
        return ctf
    
    # Parameters
    voltage_kV = 200
    wavelength_pm = 2.508  # Wavelength at 200 kV
    wavelength_A = wavelength_pm / 100
    
    k = np.linspace(0, 10, 500)  # Spatial frequency [1/Å]
    
    # CTF at different defocus and Cs values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Defocus dependence
    Cs = 1.0  # [mm]
    defocus_values = [-50, -100, -200]  # [nm]
    colors = ['#f093fb', '#f5576c', '#d07be8']
    
    for df, color in zip(defocus_values, colors):
        ctf = ctf_simple(k, wavelength_A, df, Cs)
        ax1.plot(k, ctf, linewidth=2, label=f'Δf = {df} nm', color=color)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Spatial Frequency k [1/Å]', fontsize=12)
    ax1.set_ylabel('CTF', fontsize=12)
    ax1.set_title(f'CTF vs Defocus\n(200 kV, Cs = {Cs} mm)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1, 1)
    
    # Right plot: Spherical aberration coefficient dependence
    defocus = -100  # [nm]
    Cs_values = [0.5, 1.0, 2.0]  # [mm]
    colors2 = ['#99ccff', '#ffa500', '#ff6347']
    
    for Cs_val, color in zip(Cs_values, colors2):
        ctf = ctf_simple(k, wavelength_A, defocus, Cs_val)
        ax2.plot(k, ctf, linewidth=2, label=f'Cs = {Cs_val} mm', color=color)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Spatial Frequency k [1/Å]', fontsize=12)
    ax2.set_ylabel('CTF', fontsize=12)
    ax2.set_title(f'CTF vs Cs\n(200 kV, Δf = {defocus} nm)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("CTF interpretation:")
    print("  - Positive CTF region: Spatial frequency components contribute to image with positive contrast")
    print("  - Negative CTF region: Contrast is inverted")
    print("  - Zero CTF: Frequency components do not contribute to image (information loss)")
    

### 1.3.3 Detectors

In electron microscopy, multiple detectors are used to detect various signals:

  * **Fluorescent screen** : Converts electron beam to visible light (real-time observation)
  * **CCD camera** : High sensitivity, digital image acquisition
  * **Direct Electron Detector (DED)** : Direct electron detection, fast and high sensitivity
  * **Energy Dispersive X-ray Spectroscopy (EDS/EDX)** : Elemental analysis
  * **Electron Energy Loss Spectrometer (EELS)** : Electronic state analysis

## 1.4 Signal Intensity and Statistics

### 1.4.1 Signal-to-Noise Ratio (S/N Ratio)

The quality of electron microscopy images is evaluated by the **signal-to-noise ratio (S/N)** :

$$ \text{S/N} = \frac{I_{\text{signal}}}{\sigma_{\text{noise}}} $$ 

Since the electron beam follows Poisson statistics, for $N$ detected electrons, the noise standard deviation is $\sqrt{N}$. Therefore:

$$ \text{S/N} = \sqrt{N} $$ 

**Strategies for improving S/N ratio** :

  * Increase electron dose (increase $N$) → S/N $\propto \sqrt{N}$
  * Average multiple images → S/N $\propto \sqrt{M}$ ($M$: number of images)
  * However, be careful of specimen damage (beam damage)

#### Code Example 1-5: Poisson Statistics and Signal-to-Noise Ratio
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_electron_image(size=256, signal_level=100, noise_factor=1.0):
        """
        Simulate electron microscopy image following Poisson statistics
    
        Parameters
        ----------
        size : int
            Image size [pixels]
        signal_level : float
            Average electron count [counts/pixel]
        noise_factor : float
            Noise scaling coefficient
    
        Returns
        -------
        image : ndarray
            Simulated image
        snr : float
            Signal-to-noise ratio
        """
        # Generate electron count following Poisson distribution
        image = np.random.poisson(signal_level, (size, size)).astype(float)
    
        # Additional Gaussian noise (readout noise, etc.)
        image += np.random.normal(0, noise_factor * np.sqrt(signal_level), (size, size))
    
        # Calculate S/N ratio
        signal = signal_level
        noise = np.sqrt(signal_level + noise_factor**2 * signal_level)
        snr = signal / noise
    
        return image, snr
    
    # Simulation at different electron doses
    dose_levels = [10, 50, 200, 1000]  # [electrons/pixel]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, dose in enumerate(dose_levels):
        image, snr = simulate_electron_image(size=128, signal_level=dose, noise_factor=0.5)
    
        # Display image
        axes[0, i].imshow(image, cmap='gray', vmin=0, vmax=1200)
        axes[0, i].set_title(f'Dose: {dose} e⁻/px\nS/N: {snr:.2f}', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
    
        # Histogram
        axes[1, i].hist(image.ravel(), bins=50, color='#f093fb', alpha=0.7, edgecolor='black')
        axes[1, i].set_xlabel('Intensity [counts]', fontsize=10)
        axes[1, i].set_ylabel('Frequency', fontsize=10)
        axes[1, i].set_title(f'Histogram (σ={np.std(image):.1f})', fontsize=10)
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare theoretical and measured S/N ratios
    print("Relationship between electron dose and S/N ratio:")
    for dose in dose_levels:
        _, snr = simulate_electron_image(size=128, signal_level=dose, noise_factor=0.5)
        theoretical_snr = np.sqrt(dose)
        print(f"  Dose {dose:4d} e⁻/px: Theoretical S/N={theoretical_snr:.2f}, Measured S/N={snr:.2f}")
    

## 1.5 Acceleration Voltage Selection and Specimen Damage

### 1.5.1 Effects of Acceleration Voltage

Acceleration voltage directly affects both electron microscope performance and specimen effects:

Acceleration Voltage | Wavelength | Resolution | Penetration | Specimen Damage  
---|---|---|---|---  
Low voltage (60-80 kV) | Long | Low | Low | Small  
Medium voltage (100-200 kV) | Medium | Medium | Medium | Medium  
High voltage (300-1000 kV) | Short | High | High | Large  
  
**Mechanisms of specimen damage** :

  * **Knock-on damage** : High-energy electrons knock out atoms (above threshold energy)
  * **Radiolysis** : Chemical bond breaking due to electron beam irradiation
  * **Thermal damage** : Specimen heating due to inelastic scattering

#### Code Example 1-6: Threshold Energy for Knock-on Damage
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def knock_on_threshold_voltage(element_mass, displacement_energy_eV=25):
        """
        Calculate threshold acceleration voltage for knock-on damage
    
        Parameters
        ----------
        element_mass : float
            Atomic mass of element [u]
        displacement_energy_eV : float
            Displacement energy of atom [eV] (typical: 25 eV)
    
        Returns
        -------
        V_threshold : float
            Threshold acceleration voltage [kV]
        """
        # Constants
        m_e = 9.10938e-31   # Electron mass [kg]
        u = 1.66054e-27     # Atomic mass unit [kg]
        e = 1.60218e-19     # Elementary charge [C]
        c = 2.99792e8       # Speed of light [m/s]
    
        # Atomic mass [kg]
        M = element_mass * u
    
        # Knock-on threshold energy (relativistic)
        # E_threshold ≈ (M + 2m_e) / (2M) * E_d * (1 + E_d / (2 M c²))
        E_d = displacement_energy_eV * e  # [J]
    
        # Simplified formula (non-relativistic approximation)
        E_threshold_J = (M / (2 * m_e)) * E_d
        V_threshold_kV = E_threshold_J / (e * 1000)
    
        return V_threshold_kV
    
    # Knock-on thresholds for representative elements
    elements = {
        'C': 12, 'Si': 28, 'Fe': 56, 'Cu': 64, 'Mo': 96, 'W': 184, 'Au': 197
    }
    
    element_names = list(elements.keys())
    masses = list(elements.values())
    thresholds = [knock_on_threshold_voltage(m) for m in masses]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Thresholds by element
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(elements)))
    bars = ax1.bar(element_names, thresholds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add typical acceleration voltages as reference lines
    ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Low Voltage TEM (80 kV)')
    ax1.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Standard TEM (200 kV)')
    ax1.axhline(y=300, color='red', linestyle='--', linewidth=2, label='High Voltage TEM (300 kV)')
    
    ax1.set_ylabel('Knock-on Threshold Voltage [kV]', fontsize=12)
    ax1.set_title('Knock-on Damage Threshold for Elements', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 350)
    
    # Numerical labels
    for bar, threshold in zip(bars, thresholds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 5, f'{threshold:.0f}',
                 ha='center', fontsize=9, fontweight='bold')
    
    # Right plot: Atomic mass vs threshold
    ax2.scatter(masses, thresholds, s=200, c=masses, cmap='plasma',
                edgecolors='black', linewidths=2, zorder=3)
    ax2.plot(masses, thresholds, 'k--', linewidth=1.5, alpha=0.5, zorder=1)
    
    for name, mass, threshold in zip(element_names, masses, thresholds):
        ax2.text(mass, threshold + 10, name, fontsize=10, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Mass [u]', fontsize=12)
    ax2.set_ylabel('Knock-on Threshold Voltage [kV]', fontsize=12)
    ax2.set_title('Threshold Voltage vs Atomic Mass', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Knock-on damage threshold voltages:")
    for name, mass, threshold in zip(element_names, masses, thresholds):
        print(f"  {name:2s} (M={mass:3.0f} u): {threshold:6.1f} kV")
    print("\nCarbon (graphene, etc.) can reduce damage by observing below 80 kV")
    

## 1.6 Exercise Problems

### Exercise 1-1: Calculating Electron Wavelength (Easy)

**Problem** : Calculate the electron wavelengths at acceleration voltages of 100 kV, 200 kV, and 300 kV, and find the wavelength ratio to visible light (550 nm).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Calculate the electron wavelengths at acceleration 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    def calc_wavelength(V_kV):
        h = 6.62607e-34
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        E = V_kV * 1000 * e
        p = np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        return (h / p) * 1e12  # [pm]
    
    voltages = [100, 200, 300]
    visible_light_nm = 550
    
    print("Comparison of electron wavelength with visible light:")
    for V in voltages:
        wl_pm = calc_wavelength(V)
        wl_nm = wl_pm / 1000
        ratio = visible_light_nm / wl_nm
        print(f"  {V} kV: λ = {wl_pm:.3f} pm = {wl_nm:.6f} nm, ratio = {ratio:.0f}:1")
    

### Exercise 1-2: Comparing Resolution (Medium)

**Problem** : Compare the resolution of optical microscopy (λ=550 nm, α=30°) and electron microscopy (200 kV, α=10°) using the Rayleigh criterion.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Compare the resolution of optical microscopy (λ=550
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    def resolution(wavelength_nm, alpha_deg):
        alpha_rad = np.deg2rad(alpha_deg)
        return 0.61 * wavelength_nm / np.sin(alpha_rad)
    
    # Optical microscopy
    optical_res = resolution(550, 30)
    
    # Electron microscopy (200 kV → 2.508 pm = 0.002508 nm)
    em_res = resolution(0.002508, 10)
    
    print(f"Optical microscopy resolution: {optical_res:.1f} nm")
    print(f"Electron microscopy resolution: {em_res:.5f} nm = {em_res * 10:.3f} Å")
    print(f"Resolution improvement: {optical_res / em_res:.0f}×")
    

### Exercise 1-3: Calculating Scattering Cross Section (Medium)

**Problem** : For a 200 kV electron beam, find the scattering cross section ratio at a scattering angle of 10° between Si (Z=14) and Au (Z=79).

**Show Answer**
    
    
    Z_Si = 14
    Z_Au = 79
    E_keV = 200
    theta = 10
    
    # Scattering cross section is proportional to Z²
    ratio = (Z_Au / Z_Si)**2
    
    print(f"Scattering cross section ratio between Si (Z={Z_Si}) and Au (Z={Z_Au}):")
    print(f"  σ_Au / σ_Si = ({Z_Au}/{Z_Si})² = {ratio:.1f}")
    print(f"  Au scattering cross section is approximately {ratio:.0f}× that of Si")
    

### Exercise 1-4: Improving Signal-to-Noise Ratio (Hard)

**Problem** : How much electron dose is needed to improve an electron microscopy image with a current S/N ratio of 10 to an S/N ratio of 100? Also calculate the S/N ratio improvement effect when averaging 10 images.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: How much electron dose is needed to improve an elec
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Since S/N = √N, increasing S/N ratio by 10× requires 100× dose
    current_snr = 10
    target_snr = 100
    dose_increase = (target_snr / current_snr)**2
    
    print(f"Dose increase needed to improve S/N ratio from {current_snr} to {target_snr}:")
    print(f"  Required dose increase: {dose_increase}×")
    
    # Effect of averaging 10 images
    num_images = 10
    snr_improvement = np.sqrt(num_images)
    new_snr = current_snr * snr_improvement
    
    print(f"\nWhen averaging 10 images:")
    print(f"  S/N ratio improvement factor: √{num_images} = {snr_improvement:.2f}")
    print(f"  New S/N ratio: {current_snr} × {snr_improvement:.2f} = {new_snr:.1f}")
    

### Exercise 1-5: Evaluating Knock-on Damage (Hard)

**Problem** : Evaluate whether knock-on damage occurs when observing graphene (carbon, M=12) with 200 kV TEM. Also propose the optimal acceleration voltage to avoid damage (displacement energy 25 eV).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Evaluate whether knock-on damage occurs when observ
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    def knock_on_threshold(M, E_d=25):
        m_e = 9.10938e-31
        u = 1.66054e-27
        e = 1.60218e-19
        M_kg = M * u
        E_d_J = E_d * e
        E_threshold = (M_kg / (2 * m_e)) * E_d_J
        return E_threshold / (e * 1000)  # [kV]
    
    M_C = 12
    threshold_kV = knock_on_threshold(M_C, E_d=25)
    operating_voltage = 200
    
    print(f"Knock-on threshold for graphene (carbon, M={M_C}):")
    print(f"  Threshold voltage: {threshold_kV:.1f} kV")
    print(f"  Operating voltage: {operating_voltage} kV")
    
    if operating_voltage > threshold_kV:
        print(f"  ⚠️ Observation at {operating_voltage} kV will cause knock-on damage")
        print(f"  Recommended voltage: {threshold_kV * 0.8:.0f} kV or below (20% safety margin)")
    else:
        print(f"  ✅ Observation at {operating_voltage} kV is safe")
    

### Exercise 1-6: CTF Zero Crossing (Hard)

**Problem** : Find the spatial frequency of the first CTF zero crossing for 200 kV TEM (Cs=1.0 mm, defocus -100 nm). This corresponds to real-space resolution.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Find the spatial frequency of the first CTF zero cr
    
    Purpose: Demonstrate optimization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.optimize import fsolve
    
    voltage_kV = 200
    Cs_mm = 1.0
    defocus_nm = -100
    
    # Electron wavelength
    wavelength_pm = 2.508
    wavelength_A = wavelength_pm / 100
    
    # Unit conversion
    Cs_A = Cs_mm * 1e7
    defocus_A = defocus_nm * 10
    
    # Find first non-trivial solution where CTF phase shift χ(k) = 0
    def chi(k):
        return (np.pi * wavelength_A / 2) * k**2 * (defocus_A + 0.5 * wavelength_A**2 * k**2 * Cs_A)
    
    # First zero crossing: sin(χ) = 0 → χ = π
    def equation(k):
        return chi(k) - np.pi
    
    k_zero = fsolve(equation, 2.0)[0]  # Initial estimate 2.0
    resolution_A = 1 / k_zero
    
    print(f"200 kV TEM (Cs={Cs_mm} mm, Δf={defocus_nm} nm):")
    print(f"  First zero crossing spatial frequency: {k_zero:.3f} 1/Å")
    print(f"  Corresponding real-space resolution: {resolution_A:.3f} Å = {resolution_A/10:.3f} nm")
    

### Exercise 1-7: Optimizing Acceleration Voltage (Hard)

**Problem** : When observing a 100 nm thick Si specimen, discuss which acceleration voltage (80 kV or 200 kV) provides better transmission images, considering mean free path.

**Show Answer**

**Answer points** :

  * Mean free path $\lambda_{\text{mfp}}$ increases with higher acceleration voltage
  * 80 kV: $\lambda_{\text{mfp}} \approx 50$ nm → multiple scattering occurs, high contrast but reduced resolution
  * 200 kV: $\lambda_{\text{mfp}} \approx 150$ nm → close to single scattering, high resolution
  * **Conclusion** : 200 kV is advantageous for high resolution, 80 kV for high contrast

### Exercise 1-8: Experimental Planning (Hard)

**Problem** : Plan an experimental approach to analyze unknown metal nanoparticles (10 nm diameter) by electron microscopy. Explain your choices of acceleration voltage, observation mode (TEM/SEM/STEM), and analytical methods (EDS/EELS).

**Show Answer**

**Experimental plan** :

  1. **SEM observation (low magnification)** : Check nanoparticle dispersion and morphology (acceleration voltage 5-10 kV)
  2. **TEM observation (bright field image)** : Determine particle size distribution and crystallinity (acceleration voltage 200 kV)
  3. **SAED (Selected Area Electron Diffraction)** : Identify crystal structure
  4. **HRTEM** : Measure lattice spacing and determine crystal orientation by lattice imaging
  5. **EDS analysis** : Quantify elemental composition (acceleration voltage 200 kV, observe on carbon support film)
  6. **EELS (optional)** : Check for surface oxide layers and valence states

**Rationale** :

  * 200 kV provides sufficient resolution and penetration for 10 nm nanoparticles
  * Comprehensive analysis of morphology by bright field, structure by SAED, atomic arrangement by HRTEM, and composition by EDS
  * EELS is effective for detecting oxide layers due to its high surface sensitivity

## 1.7 Learning Check

Check your understanding by answering the following questions:

  1. Can you derive de Broglie's wavelength equation and explain its relationship with acceleration voltage?
  2. Can you understand the definition of resolution by Rayleigh criterion and explain the differences between optical and electron microscopy?
  3. Can you explain the physical mechanisms of elastic and inelastic scattering and their detection signals?
  4. Do you understand the atomic number dependence of scattering cross section ($Z^2$)?
  5. Can you explain the types of electron guns (tungsten, LaB₆, FEG) and their characteristics?
  6. Do you understand how spherical and chromatic aberrations affect images?
  7. Can you explain the relationship between signal-to-noise ratio and Poisson statistics?
  8. Do you understand the threshold energy for knock-on damage and strategies for reducing specimen damage?

## 1.8 References

  1. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - Comprehensive electron microscopy textbook
  2. Reimer, L., & Kohl, H. (2008). _Transmission Electron Microscopy: Physics of Image Formation_ (5th ed.). Springer. - Detailed image formation theory
  3. Goldstein, J. I., et al. (2017). _Scanning Electron Microscopy and X-Ray Microanalysis_ (4th ed.). Springer. - Standard text for SEM and EDS analysis
  4. Egerton, R. F. (2011). _Electron Energy-Loss Spectroscopy in the Electron Microscope_ (3rd ed.). Springer. - EELS technique details
  5. Spence, J. C. H. (2013). _High-Resolution Electron Microscopy_ (4th ed.). Oxford University Press. - HRTEM theory
  6. Kirkland, E. J. (2020). _Advanced Computing in Electron Microscopy_ (3rd ed.). Springer. - Image simulation
  7. De Graef, M. (2003). _Introduction to Conventional Transmission Electron Microscopy_. Cambridge University Press. - TEM introductory text

## 1.9 Next Chapter

In the next chapter, we will learn about scanning electron microscopy (SEM) principles, secondary electron (SE) and backscattered electron (BSE) imaging, energy dispersive X-ray analysis (EDS), and image analysis basics. SEM is a versatile technique for surface morphology observation and elemental analysis at high spatial resolution.
