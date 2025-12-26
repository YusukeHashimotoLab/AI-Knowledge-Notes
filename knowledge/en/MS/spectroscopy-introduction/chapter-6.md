---
title: "Chapter 6: Python Practice - Spectroscopic Data Analysis Workflow"
chapter_title: "Chapter 6: Python Practice - Spectroscopic Data Analysis Workflow"
---

[AI Terakoya Top](<../../index.html>)>[Materials Science](<../index.html>)>[Spectroscopy Introduction](<index.html>)>Chapter 6

EN | [JP](<../../../jp/MS/spectroscopy-introduction/chapter-6.html>) | Last sync: 2025-12-26

# Chapter 6: Python Practice - Spectroscopic Data Analysis Workflow

**What you will learn in this chapter:** This chapter integrates all the spectroscopic analysis techniques learned in Chapters 1-5 (IR, Raman, UV-Vis, XPS) and builds practical Python data analysis workflows. You will develop a universal spectral data loader, automated peak detection algorithms, machine learning-based classification and regression, batch processing systems, and interactive visualization tools. All code is provided in reusable module format that can be readily applied to your own research data.

## Learning Objectives

  * Design universal spectral data loaders for multiple file formats (CSV, TXT, binary)
  * Implement automated peak detection and fitting algorithms using scipy
  * Apply machine learning for spectral classification and property prediction
  * Build batch processing pipelines for large spectral datasets
  * Apply depth profiling techniques to analyze layered structures
  * Implement XPS data analysis workflows in Python

## 5.1 The Photoelectric Effect and Binding Energy

### 5.1.1 Fundamental Principle

XPS is based on the photoelectric effect, first explained by Albert Einstein in 1905. When a material is irradiated with photons of sufficient energy, electrons are ejected from the atomic orbitals. The kinetic energy of these photoelectrons depends on the incident photon energy and the binding energy of the electron in its original orbital.

**Einstein's Photoelectric Equation:**

\\[ E_{\text{kinetic}} = h\nu - E_{\text{binding}} - \phi \\] 

where:

  * \\( E_{\text{kinetic}} \\) = kinetic energy of the emitted photoelectron (eV)
  * \\( h\nu \\) = photon energy of the incident X-ray (eV)
  * \\( E_{\text{binding}} \\) = binding energy of the electron (eV)
  * \\( \phi \\) = spectrometer work function (eV)

Rearranging to solve for binding energy:

\\[ E_{\text{binding}} = h\nu - E_{\text{kinetic}} - \phi \\] 

The binding energy is element-specific and also depends on the chemical environment (oxidation state, bonding partners). This makes XPS a powerful tool for both elemental identification and chemical state analysis.

### 5.1.2 Core Level Spectroscopy

XPS primarily probes core-level electrons (1s, 2s, 2p, 3s, 3p, 3d, etc.) because their binding energies are characteristic of each element. The naming convention for XPS peaks follows the notation: Element + Orbital (e.g., C 1s, O 1s, Fe 2p).

#### Key Characteristics of XPS

  * **Surface Sensitivity:** Information depth of 1-10 nm due to the short inelastic mean free path (IMFP) of photoelectrons
  * **Elemental Range:** Detects all elements except H and He (detection limit: 0.1-1 atomic %)
  * **Chemical Shift:** Binding energy shifts of 0.1-10 eV reveal oxidation states and bonding environments
  * **Quantitative Analysis:** Peak areas correlate with elemental concentrations
  * **Non-destructive:** Standard measurements do not damage most samples

    
    
    ```mermaid
    flowchart LR
        A[X-ray SourceAl K-alpha 1486.6 eV] --> B[Sample SurfaceCore Electrons]
        B --> C[Photoelectron EmissionE_kin = hv - E_B - phi]
        C --> D[Energy AnalyzerHemispherical]
        D --> E[DetectorChanneltron/MCP]
        E --> F[XPS SpectrumIntensity vs E_B]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fce4ec
        style D fill:#e8f5e9
        style E fill:#f3e5f5
        style F fill:#ffe0b2
    ```

#### Code Example 1: Binding Energy Calculation and Spectrum Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    """
    XPS Binding Energy Calculation and Spectrum Simulation
    
    Purpose: Demonstrate the relationship between kinetic and binding energy
             and simulate basic XPS spectra
    Target: Intermediate
    Execution time: ~2 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def kinetic_to_binding(E_kinetic, photon_energy, work_function=4.5):
        """
        Convert kinetic energy to binding energy.
    
        Parameters
        ----------
        E_kinetic : float or array
            Kinetic energy of photoelectrons (eV)
        photon_energy : float
            X-ray photon energy (eV)
        work_function : float
            Spectrometer work function (eV), default 4.5 eV
    
        Returns
        -------
        E_binding : float or array
            Binding energy (eV)
        """
        return photon_energy - E_kinetic - work_function
    
    def gaussian_peak(x, amplitude, center, sigma):
        """Generate a Gaussian peak."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def simulate_xps_spectrum(binding_energy_range, peaks, noise_level=0.02):
        """
        Simulate an XPS spectrum with multiple peaks.
    
        Parameters
        ----------
        binding_energy_range : array
            Binding energy axis (eV)
        peaks : list of dict
            Each dict contains 'center', 'amplitude', 'sigma', 'label'
        noise_level : float
            Relative noise level
    
        Returns
        -------
        spectrum : array
            Simulated intensity values
        """
        spectrum = np.zeros_like(binding_energy_range)
        for peak in peaks:
            spectrum += gaussian_peak(
                binding_energy_range,
                peak['amplitude'],
                peak['center'],
                peak['sigma']
            )
        # Add noise
        max_intensity = spectrum.max()
        noise = np.random.normal(0, noise_level * max_intensity, len(spectrum))
        spectrum += noise
        return np.maximum(spectrum, 0)  # Ensure non-negative
    
    # Example: Calculate binding energies
    print("=== XPS Binding Energy Calculations ===\n")
    
    # Common X-ray sources
    xray_sources = {
        'Al K-alpha': 1486.6,
        'Mg K-alpha': 1253.6
    }
    
    # Measured kinetic energies for different elements
    measured_kinetic = {
        'C 1s': 1202.1,
        'O 1s': 954.6,
        'Si 2p': 1387.3,
        'Fe 2p3/2': 779.6
    }
    
    print("Using Al K-alpha X-ray source (1486.6 eV):\n")
    for peak_name, E_kin in measured_kinetic.items():
        E_bind = kinetic_to_binding(E_kin, xray_sources['Al K-alpha'])
        print(f"  {peak_name}: E_kin = {E_kin:.1f} eV -> E_bind = {E_bind:.1f} eV")
    
    # Simulate a survey spectrum
    print("\n=== Simulating XPS Survey Spectrum ===")
    
    BE = np.linspace(0, 1200, 2400)  # Binding energy range
    
    # Define peaks for a SiO2 sample with carbon contamination
    survey_peaks = [
        {'center': 103.5, 'amplitude': 800, 'sigma': 2.0, 'label': 'Si 2p'},
        {'center': 154.0, 'amplitude': 200, 'sigma': 2.5, 'label': 'Si 2s'},
        {'center': 285.0, 'amplitude': 400, 'sigma': 1.5, 'label': 'C 1s'},
        {'center': 532.5, 'amplitude': 1200, 'sigma': 2.0, 'label': 'O 1s'},
        {'center': 978.0, 'amplitude': 100, 'sigma': 3.0, 'label': 'O KLL Auger'},
    ]
    
    spectrum = simulate_xps_spectrum(BE, survey_peaks, noise_level=0.03)
    
    # Plot survey spectrum
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(BE, spectrum, 'b-', linewidth=0.8)
    ax.fill_between(BE, spectrum, alpha=0.3)
    
    # Label peaks
    for peak in survey_peaks:
        idx = np.argmin(np.abs(BE - peak['center']))
        ax.annotate(peak['label'],
                    xy=(peak['center'], spectrum[idx]),
                    xytext=(peak['center'], spectrum[idx] + 100),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (counts)', fontsize=12)
    ax.set_title('Simulated XPS Survey Spectrum (SiO2 with Carbon Contamination)',
                 fontsize=14, fontweight='bold')
    ax.invert_xaxis()  # XPS convention: high BE on left
    ax.grid(alpha=0.3)
    ax.set_xlim(1200, 0)
    plt.tight_layout()
    plt.show()
    
    print("\nSpectrum simulation complete.")
    print(f"Detected elements: Si, O, C (adventitious carbon)")
    

## 5.2 XPS Instrumentation

### 5.2.1 X-ray Sources

Modern XPS instruments use characteristic X-rays from metal anodes. The two most common sources are aluminum K-alpha (1486.6 eV) and magnesium K-alpha (1253.6 eV). Monochromatic X-ray sources using crystal monochromators provide improved energy resolution and eliminate satellite peaks.

X-ray Source | Photon Energy (eV) | Natural Linewidth (eV) | Features  
---|---|---|---  
Mg K-alpha (standard) | 1253.6 | 0.70 | Lower energy, narrower linewidth  
Al K-alpha (standard) | 1486.6 | 0.85 | Higher energy, broader elemental coverage  
Al K-alpha (monochromatic) | 1486.6 | 0.25-0.30 | Best resolution, no satellites, reduced sample damage  
Ag L-alpha | 2984.3 | 2.6 | Higher kinetic energies, greater depth  
  
### 5.2.2 Energy Analyzer

The hemispherical analyzer (HSA) is the standard energy analyzer in modern XPS instruments. It consists of two concentric hemispheres with a potential difference that allows only electrons of specific kinetic energy (pass energy) to reach the detector.

#### Energy Resolution

The total energy resolution of an XPS measurement is determined by:

\\[ \Delta E_{\text{total}} = \sqrt{(\Delta E_{\text{X-ray}})^2 + (\Delta E_{\text{analyzer}})^2 + (\Delta E_{\text{natural}})^2} \\] 

where:

  * \\( \Delta E_{\text{X-ray}} \\) = X-ray source linewidth
  * \\( \Delta E_{\text{analyzer}} \\) = analyzer resolution (depends on pass energy and slit width)
  * \\( \Delta E_{\text{natural}} \\) = natural linewidth of the core level

#### Code Example 2: Analyzer Resolution and Pass Energy Effects
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    """
    XPS Analyzer Resolution Simulation
    
    Purpose: Demonstrate the effect of pass energy on spectral resolution
    Target: Intermediate
    Execution time: ~3 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_analyzer_resolution(pass_energy, slit_width_mm=0.4,
                                       analyzer_radius_mm=127):
        """
        Calculate hemispherical analyzer resolution.
    
        Parameters
        ----------
        pass_energy : float
            Pass energy (eV)
        slit_width_mm : float
            Entrance slit width (mm)
        analyzer_radius_mm : float
            Mean radius of hemispheres (mm)
    
        Returns
        -------
        resolution : float
            Energy resolution (eV)
        """
        # Simplified formula: Delta_E = E_pass * (w / 2R)
        resolution = pass_energy * (slit_width_mm / (2 * analyzer_radius_mm))
        return resolution
    
    def total_resolution(xray_linewidth, analyzer_res, natural_linewidth=0.3):
        """Calculate total energy resolution."""
        return np.sqrt(xray_linewidth**2 + analyzer_res**2 + natural_linewidth**2)
    
    def lorentzian_peak(x, amplitude, center, gamma):
        """Generate a Lorentzian peak (natural linewidth)."""
        return amplitude * (gamma**2) / ((x - center)**2 + gamma**2)
    
    def voigt_approximation(x, amplitude, center, sigma, gamma):
        """
        Voigt profile approximation using pseudo-Voigt.
        Voigt is the convolution of Gaussian and Lorentzian.
        """
        # Pseudo-Voigt approximation
        f_G = (2.0 / sigma) * np.sqrt(np.log(2) / np.pi) * \
              np.exp(-4 * np.log(2) * ((x - center) / sigma)**2)
        f_L = (2.0 / np.pi) * (gamma / ((x - center)**2 + gamma**2))
    
        # Mixing parameter (simplified)
        f = gamma / (sigma + gamma)
    
        return amplitude * (f * f_L + (1 - f) * f_G) * sigma
    
    # Calculate resolution at different pass energies
    print("=== XPS Analyzer Resolution ===\n")
    
    pass_energies = [10, 20, 50, 100, 200]
    xray_linewidth = 0.26  # Monochromatic Al K-alpha
    
    print("Pass Energy (eV) | Analyzer Res (eV) | Total Res (eV)")
    print("-" * 55)
    for PE in pass_energies:
        analyzer_res = calculate_analyzer_resolution(PE)
        total_res = total_resolution(xray_linewidth, analyzer_res)
        print(f"    {PE:3d}          |      {analyzer_res:.3f}         |    {total_res:.3f}")
    
    # Simulate C 1s spectrum at different pass energies
    print("\n=== Simulating C 1s Spectra at Different Pass Energies ===")
    
    BE = np.linspace(280, 295, 1500)
    
    # C 1s has multiple chemical states in a polymer sample
    # C-C at 285.0 eV, C-O at 286.5 eV, C=O at 288.0 eV
    peak_positions = [285.0, 286.5, 288.0]
    peak_amplitudes = [1000, 400, 200]
    peak_labels = ['C-C/C-H', 'C-O', 'C=O']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    pass_energy_demo = [10, 20, 50, 100]
    
    for idx, PE in enumerate(pass_energy_demo):
        ax = axes[idx]
    
        # Calculate resolution
        analyzer_res = calculate_analyzer_resolution(PE)
        total_res = total_resolution(xray_linewidth, analyzer_res)
    
        # Generate spectrum
        spectrum = np.zeros_like(BE)
        for pos, amp in zip(peak_positions, peak_amplitudes):
            # Use total resolution as sigma
            spectrum += voigt_approximation(BE, amp, pos,
                                            sigma=total_res, gamma=0.2)
    
        # Add noise (more noise at higher pass energy due to higher count rate)
        noise_scale = 0.02 + 0.01 * (PE / 100)
        noise = np.random.normal(0, noise_scale * spectrum.max(), len(spectrum))
        spectrum += noise
        spectrum = np.maximum(spectrum, 0)
    
        ax.plot(BE, spectrum, 'b-', linewidth=1.2)
        ax.fill_between(BE, spectrum, alpha=0.3)
        ax.set_xlabel('Binding Energy (eV)', fontsize=11)
        ax.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax.set_title(f'Pass Energy = {PE} eV (Resolution = {total_res:.2f} eV)',
                     fontsize=12, fontweight='bold')
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    
        # Mark peak positions
        for pos, label in zip(peak_positions, peak_labels):
            ax.axvline(pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('Effect of Pass Energy on C 1s Spectral Resolution',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\nKey observations:")
    print("- Lower pass energy = better resolution but lower count rate")
    print("- Higher pass energy = faster acquisition but peaks may overlap")
    print("- Typical survey: 100-200 eV pass energy")
    print("- High-resolution: 10-20 eV pass energy")
    

## 5.3 Chemical State Analysis and Peak Fitting

### 5.3.1 Chemical Shifts

The binding energy of core electrons is influenced by the chemical environment of the atom. When an atom loses electrons (oxidation), the remaining electrons experience stronger nuclear attraction, resulting in higher binding energy. Conversely, electron gain leads to lower binding energy.

#### Origin of Chemical Shifts

  * **Oxidation State Effect:** Higher oxidation states increase binding energy   
Example: Si 2p: Si0 (99.3 eV) < SiO (101.5 eV) < SiO2 (103.5 eV)
  * **Electronegativity Effect:** Bonding to more electronegative atoms increases BE   
Example: C 1s: C-C (285.0 eV) < C-O (286.5 eV) < C=O (288.0 eV) < O-C=O (289.5 eV)
  * **Coordination Number:** Changes in coordination affect electron density

Element | Peak | Binding Energy (eV) | Chemical State  
---|---|---|---  
C | 1s | 284.5-285.0 | C-C, C-H (aliphatic, aromatic)  
1s | 286.0-286.5 | C-O (ether, alcohol)  
1s | 287.5-288.5 | C=O (carbonyl, amide)  
1s | 289.0-290.0 | O-C=O (carboxyl, carbonate)  
Si | 2p3/2 | 99.0-99.5 | Si0 (elemental silicon)  
2p3/2 | 101.0-102.0 | Si2+ (SiO, suboxides)  
2p3/2 | 103.0-104.0 | Si4+ (SiO2)  
Fe | 2p3/2 | 706.5-707.5 | Fe0 (metallic iron)  
2p3/2 | 710.5-711.5 | Fe3+ (Fe2O3)  
  
### 5.3.2 Peak Fitting with Voigt Functions

XPS peaks have a shape that results from the convolution of Gaussian (instrumental broadening) and Lorentzian (natural lifetime broadening) components. The Voigt function accurately describes this lineshape and is essential for quantitative peak fitting.

**Voigt Profile:**

\\[ V(x; \sigma, \gamma) = \int_{-\infty}^{\infty} G(x'; \sigma) \cdot L(x - x'; \gamma) \, dx' \\] 

where G is the Gaussian and L is the Lorentzian component.

**Pseudo-Voigt Approximation:**

\\[ V(x) \approx \eta \cdot L(x; \gamma) + (1 - \eta) \cdot G(x; \sigma) \\] 

where \\( \eta \\) is the mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian).

#### Code Example 3: Voigt Function Peak Fitting
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - scipy>=1.10.0
    # - matplotlib>=3.7.0
    
    """
    XPS Peak Fitting with Voigt Functions
    
    Purpose: Demonstrate multi-peak fitting for chemical state analysis
    Target: Intermediate to Advanced
    Execution time: ~5 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile
    
    def voigt_peak(x, amplitude, center, sigma, gamma):
        """
        Voigt function for XPS peak fitting.
    
        Parameters
        ----------
        x : array
            Binding energy values
        amplitude : float
            Peak amplitude
        center : float
            Peak center position (eV)
        sigma : float
            Gaussian width parameter (eV)
        gamma : float
            Lorentzian width parameter (eV)
    
        Returns
        -------
        intensity : array
            Peak intensity values
        """
        # scipy's voigt_profile uses different parameterization
        # sigma_voigt = sigma / sqrt(2)
        # gamma_voigt = gamma
        z = (x - center) / (sigma * np.sqrt(2))
        return amplitude * voigt_profile(z, gamma / (sigma * np.sqrt(2)))
    
    def shirley_background(x, y, tol=1e-5, max_iter=50):
        """
        Calculate Shirley background.
    
        Parameters
        ----------
        x : array
            Binding energy
        y : array
            Intensity
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
    
        Returns
        -------
        background : array
            Shirley background
        """
        # Start and end points
        y_left = y[0]
        y_right = y[-1]
    
        # Initialize background
        bg = np.linspace(y_left, y_right, len(y))
    
        # Iterative calculation
        for iteration in range(max_iter):
            bg_old = bg.copy()
    
            # Calculate cumulative integral from right
            cumsum = np.cumsum((y - bg)[::-1])[::-1]
    
            # Scale background
            k = (y_left - y_right) / cumsum[0] if cumsum[0] != 0 else 0
            bg = y_right + k * cumsum
    
            # Check convergence
            if np.max(np.abs(bg - bg_old)) < tol:
                break
    
        return bg
    
    def multi_voigt(x, *params):
        """
        Sum of multiple Voigt peaks.
    
        Parameters are grouped as: (amplitude, center, sigma, gamma) for each peak
        """
        n_peaks = len(params) // 4
        result = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            amp = params[4*i]
            center = params[4*i + 1]
            sigma = params[4*i + 2]
            gamma = params[4*i + 3]
            result += voigt_peak(x, amp, center, sigma, gamma)
        return result
    
    # Generate synthetic C 1s spectrum with multiple chemical states
    print("=== XPS C 1s Peak Fitting Example ===\n")
    
    np.random.seed(42)
    BE = np.linspace(280, 295, 750)
    
    # True peak parameters: [amplitude, center, sigma, gamma]
    true_peaks = [
        [1000, 285.0, 0.6, 0.3],   # C-C/C-H
        [400, 286.5, 0.6, 0.3],    # C-O
        [200, 288.0, 0.7, 0.3],    # C=O
        [100, 289.5, 0.7, 0.3],    # O-C=O
    ]
    
    # Generate spectrum
    spectrum = np.zeros_like(BE)
    for params in true_peaks:
        spectrum += voigt_peak(BE, *params)
    
    # Add Shirley-like background
    bg_level = 50 + 30 * (1 - np.exp(-(BE - 280) / 10))
    spectrum += bg_level
    
    # Add noise
    noise = np.random.normal(0, 15, len(spectrum))
    measured_spectrum = spectrum + noise
    measured_spectrum = np.maximum(measured_spectrum, 0)
    
    # Estimate and subtract Shirley background
    background = shirley_background(BE, measured_spectrum)
    spectrum_no_bg = measured_spectrum - background
    
    # Perform peak fitting
    # Initial guesses
    p0 = [800, 285.0, 0.7, 0.3,
          300, 286.5, 0.7, 0.3,
          150, 288.0, 0.7, 0.3,
          80, 289.5, 0.7, 0.3]
    
    # Bounds
    lower_bounds = [0, 284, 0.3, 0.1,
                    0, 285.5, 0.3, 0.1,
                    0, 287, 0.3, 0.1,
                    0, 288.5, 0.3, 0.1]
    upper_bounds = [2000, 286, 1.5, 1.0,
                    1000, 287.5, 1.5, 1.0,
                    500, 289, 1.5, 1.0,
                    300, 291, 1.5, 1.0]
    
    try:
        popt, pcov = curve_fit(multi_voigt, BE, spectrum_no_bg, p0=p0,
                               bounds=(lower_bounds, upper_bounds), maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    
        # Calculate fitted peaks
        fitted_total = multi_voigt(BE, *popt)
        fitted_peaks = []
        for i in range(4):
            peak = voigt_peak(BE, popt[4*i], popt[4*i+1], popt[4*i+2], popt[4*i+3])
            fitted_peaks.append(peak)
    
        fitting_success = True
    except Exception as e:
        print(f"Fitting failed: {e}")
        fitting_success = False
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Raw spectrum with background
    axes[0, 0].plot(BE, measured_spectrum, 'b-', linewidth=1, label='Measured')
    axes[0, 0].plot(BE, background, 'r--', linewidth=2, label='Shirley Background')
    axes[0, 0].fill_between(BE, background, alpha=0.2, color='red')
    axes[0, 0].set_xlabel('Binding Energy (eV)', fontsize=11)
    axes[0, 0].set_ylabel('Intensity (counts)', fontsize=11)
    axes[0, 0].set_title('Raw Spectrum with Shirley Background', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].invert_xaxis()
    axes[0, 0].grid(alpha=0.3)
    
    # Background-subtracted spectrum with fit
    if fitting_success:
        axes[0, 1].plot(BE, spectrum_no_bg, 'ko', markersize=2, alpha=0.5, label='Data')
        axes[0, 1].plot(BE, fitted_total, 'r-', linewidth=2, label='Fitted Total')
    
        colors = ['blue', 'green', 'orange', 'purple']
        labels = ['C-C/C-H', 'C-O', 'C=O', 'O-C=O']
        for i, (peak, color, label) in enumerate(zip(fitted_peaks, colors, labels)):
            axes[0, 1].fill_between(BE, peak, alpha=0.4, color=color, label=label)
    
        axes[0, 1].set_xlabel('Binding Energy (eV)', fontsize=11)
        axes[0, 1].set_ylabel('Intensity (counts)', fontsize=11)
        axes[0, 1].set_title('Peak Fitting Results', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].invert_xaxis()
        axes[0, 1].grid(alpha=0.3)
    
        # Residuals
        residuals = spectrum_no_bg - fitted_total
        axes[1, 0].plot(BE, residuals, 'g-', linewidth=1)
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].fill_between(BE, residuals, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Binding Energy (eV)', fontsize=11)
        axes[1, 0].set_ylabel('Residual (counts)', fontsize=11)
        axes[1, 0].set_title('Fitting Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].invert_xaxis()
        axes[1, 0].grid(alpha=0.3)
    
        # Peak areas (quantification)
        peak_areas = []
        for i in range(4):
            area = np.trapz(fitted_peaks[i], BE)
            peak_areas.append(area)
    
        total_area = sum(peak_areas)
        percentages = [100 * area / total_area for area in peak_areas]
    
        axes[1, 1].bar(labels, percentages, color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_xlabel('Chemical State', fontsize=11)
        axes[1, 1].set_ylabel('Relative Percentage (%)', fontsize=11)
        axes[1, 1].set_title('Chemical State Distribution', fontsize=12, fontweight='bold')
        for i, (label, pct) in enumerate(zip(labels, percentages)):
            axes[1, 1].text(i, pct + 1, f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print fitting results
    if fitting_success:
        print("\nFitting Results:")
        print("-" * 70)
        print(f"{'Component':<12} {'Center (eV)':<15} {'FWHM (eV)':<12} {'Area':<12} {'%':<8}")
        print("-" * 70)
        for i, label in enumerate(labels):
            center = popt[4*i + 1]
            sigma = popt[4*i + 2]
            gamma = popt[4*i + 3]
            # FWHM of Voigt is approximately:
            fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2*gamma)**2 + (2.355*sigma)**2)
            print(f"{label:<12} {center:<15.2f} {fwhm:<12.2f} {peak_areas[i]:<12.0f} {percentages[i]:<8.1f}")
        print("-" * 70)
        print(f"{'Total':<12} {'':<15} {'':<12} {total_area:<12.0f} {'100.0':<8}")
    

## 5.4 Quantitative Analysis and Sensitivity Factors

### 5.4.1 Quantification Principle

The atomic concentration of an element can be calculated from the peak area, corrected by the relative sensitivity factor (RSF). The RSF accounts for differences in photoionization cross-sections, electron kinetic energies, and analyzer transmission functions.

**Atomic Concentration Calculation:**

\\[ C_i = \frac{I_i / S_i}{\sum_j (I_j / S_j)} \times 100\% \\] 

where:

  * \\( C_i \\) = atomic concentration of element i (at%)
  * \\( I_i \\) = peak area of element i
  * \\( S_i \\) = relative sensitivity factor of element i

### 5.4.2 Relative Sensitivity Factors

RSF values depend on the X-ray source, spectrometer geometry, and the specific core level being measured. The most commonly used RSF values are the Scofield cross-sections, often modified by empirical factors for specific instruments.

Element | Core Level | RSF (Al K-alpha) | RSF (Mg K-alpha)  
---|---|---|---  
C| 1s| 0.25| 0.25  
N| 1s| 0.42| 0.42  
O| 1s| 0.66| 0.66  
F| 1s| 1.00| 1.00  
Si| 2p| 0.27| 0.27  
S| 2p| 0.54| 0.54  
Ti| 2p| 1.80| 1.80  
Fe| 2p| 2.96| 2.96  
Cu| 2p| 4.20| 4.20  
Au| 4f| 4.95| 4.95  
  
#### Code Example 4: Quantitative Surface Composition Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0
    
    """
    XPS Quantitative Analysis
    
    Purpose: Calculate atomic concentrations from peak areas using RSF values
    Target: Intermediate
    Execution time: ~3 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    class XPSQuantification:
        """
        XPS quantitative analysis class.
        """
    
        # Default RSF values (Scofield cross-sections, Al K-alpha)
        DEFAULT_RSF = {
            'C 1s': 0.25,
            'N 1s': 0.42,
            'O 1s': 0.66,
            'F 1s': 1.00,
            'Na 1s': 1.68,
            'Mg 1s': 2.08,
            'Al 2p': 0.19,
            'Si 2p': 0.27,
            'P 2p': 0.39,
            'S 2p': 0.54,
            'Cl 2p': 0.73,
            'K 2p': 0.83,
            'Ca 2p': 1.00,
            'Ti 2p': 1.80,
            'Cr 2p': 2.40,
            'Mn 2p': 2.66,
            'Fe 2p': 2.96,
            'Co 2p': 3.59,
            'Ni 2p': 3.90,
            'Cu 2p': 4.20,
            'Zn 2p': 4.65,
            'Au 4f': 4.95,
            'Ag 3d': 5.98,
        }
    
        def __init__(self, custom_rsf=None):
            """
            Initialize with RSF values.
    
            Parameters
            ----------
            custom_rsf : dict, optional
                Custom RSF values to override defaults
            """
            self.rsf = self.DEFAULT_RSF.copy()
            if custom_rsf:
                self.rsf.update(custom_rsf)
    
        def calculate_atomic_percent(self, peak_areas):
            """
            Calculate atomic percentages from peak areas.
    
            Parameters
            ----------
            peak_areas : dict
                Dictionary of {peak_name: area}
    
            Returns
            -------
            atomic_percent : dict
                Dictionary of {element: atomic %}
            normalized_intensities : dict
                Dictionary of {element: I/S}
            """
            # Calculate I/S for each element
            normalized = {}
            for peak, area in peak_areas.items():
                if peak in self.rsf:
                    normalized[peak] = area / self.rsf[peak]
                else:
                    print(f"Warning: RSF not found for {peak}, skipping...")
    
            # Calculate total
            total = sum(normalized.values())
    
            # Calculate atomic percentages
            atomic_percent = {}
            for peak, norm_intensity in normalized.items():
                atomic_percent[peak] = (norm_intensity / total) * 100
    
            return atomic_percent, normalized
    
        def generate_report(self, peak_areas, sample_name="Sample"):
            """
            Generate a quantification report.
    
            Parameters
            ----------
            peak_areas : dict
                Dictionary of {peak_name: area}
            sample_name : str
                Sample identifier
    
            Returns
            -------
            df : pandas.DataFrame
                Report as DataFrame
            """
            atomic_percent, normalized = self.calculate_atomic_percent(peak_areas)
    
            data = []
            for peak in peak_areas.keys():
                if peak in atomic_percent:
                    element = peak.split()[0]
                    data.append({
                        'Peak': peak,
                        'Element': element,
                        'Raw Area': peak_areas[peak],
                        'RSF': self.rsf[peak],
                        'Normalized (I/S)': normalized[peak],
                        'Atomic %': atomic_percent[peak]
                    })
    
            df = pd.DataFrame(data)
            return df
    
    # Example: Analyze a Ti-based thin film sample
    print("=== XPS Quantitative Analysis ===\n")
    print("Sample: TiO2 thin film on Si substrate with carbon contamination\n")
    
    # Measured peak areas (arbitrary units)
    peak_areas = {
        'Ti 2p': 15000,
        'O 1s': 42000,
        'C 1s': 8000,
        'Si 2p': 3000
    }
    
    # Create quantification object
    quant = XPSQuantification()
    
    # Generate report
    report = quant.generate_report(peak_areas, "TiO2 film")
    print(report.to_string(index=False))
    print()
    
    # Calculate atomic percentages
    atomic_percent, _ = quant.calculate_atomic_percent(peak_areas)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    elements = [p.split()[0] for p in atomic_percent.keys()]
    percentages = list(atomic_percent.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(elements)))
    
    wedges, texts, autotexts = axes[0].pie(
        percentages,
        labels=elements,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(elements)
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    axes[0].set_title('Surface Composition (Atomic %)', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(elements, percentages, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Element', fontsize=12)
    axes[1].set_ylabel('Atomic Concentration (%)', fontsize=12)
    axes[1].set_title('Surface Composition Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                     f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Stoichiometry analysis
    print("\n=== Stoichiometry Analysis ===")
    Ti_conc = atomic_percent['Ti 2p']
    O_conc = atomic_percent['O 1s']
    ratio = O_conc / Ti_conc if Ti_conc > 0 else 0
    
    print(f"Ti concentration: {Ti_conc:.1f} at%")
    print(f"O concentration: {O_conc:.1f} at%")
    print(f"O/Ti ratio: {ratio:.2f}")
    print(f"Expected for TiO2: 2.00")
    
    # Excluding adventitious carbon
    print("\n=== Composition Excluding Carbon Contamination ===")
    Ti_only = peak_areas['Ti 2p']
    O_only = peak_areas['O 1s']
    Si_only = peak_areas['Si 2p']
    
    clean_areas = {'Ti 2p': Ti_only, 'O 1s': O_only, 'Si 2p': Si_only}
    clean_percent, _ = quant.calculate_atomic_percent(clean_areas)
    
    for peak, pct in clean_percent.items():
        print(f"  {peak}: {pct:.1f} at%")
    

## 5.5 Depth Profiling and Surface Sensitivity

### 5.5.1 Information Depth and IMFP

The surface sensitivity of XPS arises from the limited escape depth of photoelectrons. The inelastic mean free path (IMFP) determines how far an electron can travel through a material without losing energy through inelastic scattering.

**Information Depth:**

\\[ d = 3\lambda \cos\theta \\] 

where \\( \lambda \\) is the IMFP (typically 1-3 nm for most materials) and \\( \theta \\) is the emission angle from the surface normal.

**Signal Attenuation:**

\\[ I(d) = I_0 \exp\left(-\frac{d}{\lambda \cos\theta}\right) \\] 

95% of the signal comes from a depth of \\( 3\lambda \\) (typically 3-10 nm).

### 5.5.2 Angle-Resolved XPS (ARXPS)

By varying the detection angle, different sampling depths can be probed non-destructively. At grazing angles (large theta from normal), the sampling depth decreases, making the measurement more surface-sensitive.

### 5.5.3 Sputter Depth Profiling

For analyzing thicker layers or buried interfaces, ion sputtering can be combined with XPS measurements. Ar+ or cluster ion beams remove material layer by layer, while XPS spectra are acquired at each depth.

#### Code Example 5: Depth Profile Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - scipy>=1.10.0
    # - matplotlib>=3.7.0
    
    """
    XPS Depth Profile Analysis
    
    Purpose: Simulate and analyze sputter depth profiles
    Target: Intermediate to Advanced
    Execution time: ~4 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    from scipy.optimize import curve_fit
    
    def interface_profile(z, interface_position, interface_width):
        """
        Model concentration profile across an interface.
        Uses error function for smooth transition.
    
        Parameters
        ----------
        z : array
            Depth (nm)
        interface_position : float
            Center of interface (nm)
        interface_width : float
            Interface width/roughness (nm)
    
        Returns
        -------
        profile : array
            Normalized concentration (0 to 1)
        """
        return 0.5 * (1 + erf((interface_position - z) / (interface_width * np.sqrt(2))))
    
    def simulate_depth_profile(depth, layers, noise_level=0.02):
        """
        Simulate a multilayer depth profile.
    
        Parameters
        ----------
        depth : array
            Depth values (nm)
        layers : list of dict
            Each dict contains 'element', 'interfaces', 'concentrations'
        noise_level : float
            Relative noise level
    
        Returns
        -------
        profiles : dict
            {element: concentration_profile}
        """
        profiles = {}
    
        for layer in layers:
            element = layer['element']
            conc = np.zeros_like(depth)
    
            interfaces = layer['interfaces']
            concentrations = layer['concentrations']
    
            for i in range(len(interfaces)):
                pos = interfaces[i]['position']
                width = interfaces[i]['width']
                c_before = concentrations[i]
                c_after = concentrations[i + 1] if i + 1 < len(concentrations) else 0
    
                transition = interface_profile(depth, pos, width)
                conc += (c_before - c_after) * transition
    
            conc += concentrations[-1]
    
            # Add noise
            noise = np.random.normal(0, noise_level * max(concentrations), len(depth))
            profiles[element] = np.maximum(conc + noise, 0)
    
        return profiles
    
    def fit_interface(depth, concentration, initial_guess):
        """
        Fit interface position and width from concentration profile.
        """
        def model(z, interface_pos, interface_width, c_top, c_bottom):
            return c_bottom + (c_top - c_bottom) * interface_profile(z, interface_pos, interface_width)
    
        try:
            popt, pcov = curve_fit(model, depth, concentration, p0=initial_guess, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            return popt, perr
        except Exception as e:
            print(f"Fitting failed: {e}")
            return None, None
    
    # Simulate SiO2/Si structure with interface
    print("=== XPS Sputter Depth Profile Simulation ===\n")
    print("Sample: SiO2 (20 nm) / Si substrate with 2 nm interface roughness\n")
    
    np.random.seed(42)
    depth = np.linspace(0, 50, 200)  # 0-50 nm depth
    
    # Define layer structure
    # SiO2 layer: Si (~33 at%), O (~67 at%) for stoichiometric SiO2
    # Si substrate: Si (100 at%), O (0 at%)
    
    layers = [
        {
            'element': 'Si',
            'interfaces': [{'position': 20, 'width': 2}],
            'concentrations': [33, 100]  # SiO2 -> Si
        },
        {
            'element': 'O',
            'interfaces': [{'position': 20, 'width': 2}],
            'concentrations': [67, 0]  # SiO2 -> Si
        }
    ]
    
    # Simulate profiles
    profiles = simulate_depth_profile(depth, layers, noise_level=0.03)
    
    # Simulate discrete sputter measurements
    sputter_times = np.array([0, 2, 5, 8, 12, 16, 20, 25, 30, 40, 50])  # minutes
    sputter_rate = 1.0  # nm/min
    measured_depths = sputter_times * sputter_rate
    
    # Sample at measurement points
    measured_Si = np.interp(measured_depths, depth, profiles['Si']) + \
                  np.random.normal(0, 2, len(measured_depths))
    measured_O = np.interp(measured_depths, depth, profiles['O']) + \
                 np.random.normal(0, 2, len(measured_depths))
    
    # Fit interface
    initial_guess = [20, 2, 33, 100]  # [pos, width, c_top, c_bottom]
    popt_Si, perr_Si = fit_interface(measured_depths, measured_Si, initial_guess)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Theoretical profiles
    axes[0, 0].plot(depth, profiles['Si'], 'b-', linewidth=2, label='Si (theoretical)')
    axes[0, 0].plot(depth, profiles['O'], 'r-', linewidth=2, label='O (theoretical)')
    axes[0, 0].scatter(measured_depths, measured_Si, s=80, c='blue', marker='o',
                       edgecolors='black', linewidth=1.5, label='Si (measured)', zorder=5)
    axes[0, 0].scatter(measured_depths, measured_O, s=80, c='red', marker='s',
                       edgecolors='black', linewidth=1.5, label='O (measured)', zorder=5)
    axes[0, 0].axvline(20, color='green', linestyle='--', linewidth=2,
                       label='Interface (20 nm)')
    axes[0, 0].set_xlabel('Depth (nm)', fontsize=12)
    axes[0, 0].set_ylabel('Atomic Concentration (at%)', fontsize=12)
    axes[0, 0].set_title('Sputter Depth Profile: SiO2/Si', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='right')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(-5, 110)
    
    # Layer structure visualization
    ax_struct = axes[0, 1]
    ax_struct.fill_between([0, 20], 0, 100, alpha=0.4, color='red', label='SiO2 Layer')
    ax_struct.fill_between([20, 50], 0, 100, alpha=0.4, color='blue', label='Si Substrate')
    ax_struct.axvline(20, color='green', linestyle='-', linewidth=4, label='Interface')
    ax_struct.text(10, 50, 'SiO2\n(~20 nm)', fontsize=16, fontweight='bold',
                   ha='center', va='center')
    ax_struct.text(35, 50, 'Si\nSubstrate', fontsize=16, fontweight='bold',
                   ha='center', va='center')
    ax_struct.set_xlabel('Depth (nm)', fontsize=12)
    ax_struct.set_ylabel('Layer', fontsize=12)
    ax_struct.set_title('Sample Structure (Cross-section)', fontsize=14, fontweight='bold')
    ax_struct.set_yticks([])
    ax_struct.legend(loc='upper right')
    ax_struct.grid(alpha=0.3, axis='x')
    ax_struct.set_xlim(0, 50)
    
    # Interface fitting result
    if popt_Si is not None:
        fitted_pos = popt_Si[0]
        fitted_width = popt_Si[1]
    
        axes[1, 0].scatter(measured_depths, measured_Si, s=80, c='blue', marker='o',
                           edgecolors='black', linewidth=1.5, label='Measured Si', zorder=5)
    
        z_fit = np.linspace(0, 50, 500)
        fitted_profile = popt_Si[3] + (popt_Si[2] - popt_Si[3]) * \
                         interface_profile(z_fit, popt_Si[0], popt_Si[1])
        axes[1, 0].plot(z_fit, fitted_profile, 'r-', linewidth=2, label='Fitted Profile')
    
        axes[1, 0].axvline(fitted_pos, color='green', linestyle='--', linewidth=2)
        axes[1, 0].axvspan(fitted_pos - fitted_width, fitted_pos + fitted_width,
                           alpha=0.2, color='yellow', label=f'Interface Region')
    
        axes[1, 0].set_xlabel('Depth (nm)', fontsize=12)
        axes[1, 0].set_ylabel('Si Concentration (at%)', fontsize=12)
        axes[1, 0].set_title('Interface Fitting Result', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
        # Print fitting results
        print("Interface Fitting Results:")
        print(f"  Interface Position: {fitted_pos:.2f} +/- {perr_Si[0]:.2f} nm")
        print(f"  Interface Width: {fitted_width:.2f} +/- {perr_Si[1]:.2f} nm")
        print(f"  SiO2 Layer Si conc: {popt_Si[2]:.1f} at%")
        print(f"  Si Substrate conc: {popt_Si[3]:.1f} at%")
    
    # O/Si ratio vs depth
    O_Si_ratio = profiles['O'] / np.maximum(profiles['Si'], 0.1)
    axes[1, 1].plot(depth, O_Si_ratio, 'purple', linewidth=2)
    axes[1, 1].axhline(2.0, color='red', linestyle='--', linewidth=1.5,
                       label='Stoichiometric SiO2 (O/Si = 2)')
    axes[1, 1].axvline(20, color='green', linestyle='--', linewidth=2, label='Interface')
    axes[1, 1].set_xlabel('Depth (nm)', fontsize=12)
    axes[1, 1].set_ylabel('O/Si Atomic Ratio', fontsize=12)
    axes[1, 1].set_title('Stoichiometry vs Depth', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(0, 4)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Depth Profile Analysis Complete ===")
    

#### Code Example 6: Angle-Resolved XPS (ARXPS) Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    """
    Angle-Resolved XPS (ARXPS) Simulation
    
    Purpose: Demonstrate non-destructive depth analysis using variable take-off angles
    Target: Advanced
    Execution time: ~3 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def arxps_intensity(overlayer_thickness, imfp, takeoff_angle_deg):
        """
        Calculate XPS intensity from substrate through an overlayer.
    
        Parameters
        ----------
        overlayer_thickness : float
            Thickness of overlayer (nm)
        imfp : float
            Inelastic mean free path (nm)
        takeoff_angle_deg : float
            Take-off angle from surface (degrees)
    
        Returns
        -------
        attenuation : float
            Signal attenuation factor (0 to 1)
        """
        theta = np.radians(takeoff_angle_deg)
        return np.exp(-overlayer_thickness / (imfp * np.sin(theta)))
    
    def calculate_overlayer_thickness(I_ratio, imfp_overlayer, imfp_substrate,
                                       takeoff_angles_deg):
        """
        Calculate overlayer thickness from angle-dependent intensity ratios.
    
        Uses the Thickogram method for thin overlayers.
        """
        # Linearized analysis for thin films
        # ln(I_overlayer/I_substrate) vs 1/sin(theta)
        x = 1 / np.sin(np.radians(takeoff_angles_deg))
        y = np.log(I_ratio)
    
        # Linear fit
        slope, intercept = np.polyfit(x, y, 1)
    
        # Thickness from slope
        # For homogeneous overlayer: slope = d/lambda
        thickness = -slope * imfp_overlayer
    
        return thickness, slope, intercept
    
    # Simulate ARXPS measurement of thin oxide layer on metal
    print("=== Angle-Resolved XPS (ARXPS) Simulation ===\n")
    print("Sample: 2.5 nm TiO2 overlayer on Ti metal substrate\n")
    
    # Parameters
    overlayer_thickness = 2.5  # nm (TiO2 layer)
    imfp_oxide = 2.0  # nm (IMFP in TiO2)
    imfp_metal = 1.5  # nm (IMFP in Ti metal)
    
    # Take-off angles
    angles = np.array([15, 30, 45, 60, 75, 90])
    
    # Calculate intensities
    # Overlayer (oxide) signal
    I_oxide = 1 - arxps_intensity(overlayer_thickness, imfp_oxide, angles)
    # Substrate (metal) signal attenuated by overlayer
    I_metal = arxps_intensity(overlayer_thickness, imfp_oxide, angles)
    
    # Add noise
    np.random.seed(42)
    I_oxide_measured = I_oxide * (1 + np.random.normal(0, 0.03, len(angles)))
    I_metal_measured = I_metal * (1 + np.random.normal(0, 0.03, len(angles)))
    
    # Calculate intensity ratio
    I_ratio = I_oxide_measured / I_metal_measured
    
    # Calculate sampling depth at each angle
    sampling_depths = 3 * imfp_oxide * np.sin(np.radians(angles))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Angular dependence of intensities
    axes[0, 0].plot(angles, I_oxide_measured, 'bo-', linewidth=2, markersize=10,
                    label='Ti(oxide) - TiO2')
    axes[0, 0].plot(angles, I_metal_measured, 'rs-', linewidth=2, markersize=10,
                    label='Ti(metal) - substrate')
    axes[0, 0].set_xlabel('Take-off Angle (degrees)', fontsize=12)
    axes[0, 0].set_ylabel('Normalized Intensity', fontsize=12)
    axes[0, 0].set_title('Angular Dependence of XPS Signals', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # Sampling depth vs angle
    axes[0, 1].bar(angles, sampling_depths, width=8, color='green',
                   edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(overlayer_thickness, color='red', linestyle='--', linewidth=2,
                       label=f'Overlayer thickness ({overlayer_thickness} nm)')
    axes[0, 1].set_xlabel('Take-off Angle (degrees)', fontsize=12)
    axes[0, 1].set_ylabel('Sampling Depth (nm)', fontsize=12)
    axes[0, 1].set_title('Information Depth vs Take-off Angle', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Linearized analysis for thickness determination
    x_plot = 1 / np.sin(np.radians(angles))
    y_plot = np.log(I_ratio)
    
    # Linear fit
    slope, intercept = np.polyfit(x_plot, y_plot, 1)
    fit_line = slope * x_plot + intercept
    
    axes[1, 0].scatter(x_plot, y_plot, s=100, c='blue', marker='o',
                       edgecolors='black', linewidth=1.5, label='Measured data')
    axes[1, 0].plot(x_plot, fit_line, 'r-', linewidth=2,
                    label=f'Linear fit (slope = {slope:.3f})')
    axes[1, 0].set_xlabel('1/sin(theta)', fontsize=12)
    axes[1, 0].set_ylabel('ln(I_oxide/I_metal)', fontsize=12)
    axes[1, 0].set_title('Linearized Analysis for Thickness', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Calculate thickness
    calculated_thickness = -slope * imfp_oxide
    
    # Summary visualization
    summary_text = f"""
    ARXPS Analysis Results:
    ------------------------------
    True overlayer thickness: {overlayer_thickness:.2f} nm
    Calculated thickness: {calculated_thickness:.2f} nm
    Error: {abs(calculated_thickness - overlayer_thickness):.3f} nm
    
    Analysis Parameters:
    - IMFP (TiO2): {imfp_oxide} nm
    - IMFP (Ti metal): {imfp_metal} nm
    - Angles measured: {len(angles)} points
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('ARXPS Analysis Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("ARXPS Analysis Results:")
    print(f"  True overlayer thickness: {overlayer_thickness:.2f} nm")
    print(f"  Calculated thickness: {calculated_thickness:.2f} nm")
    print(f"  Error: {abs(calculated_thickness - overlayer_thickness):.3f} nm ({100*abs(calculated_thickness - overlayer_thickness)/overlayer_thickness:.1f}%)")
    

#### Code Example 7: Complete XPS Data Processing Pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - scipy>=1.10.0
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0
    
    """
    Complete XPS Data Processing Pipeline
    
    Purpose: Demonstrate end-to-end XPS data analysis workflow
    Target: Advanced
    Execution time: ~8 seconds
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter
    import pandas as pd
    
    class XPSSpectrum:
        """
        Class for XPS spectrum processing and analysis.
        """
    
        def __init__(self, binding_energy, intensity, name="Spectrum"):
            """
            Initialize XPS spectrum.
    
            Parameters
            ----------
            binding_energy : array
                Binding energy axis (eV)
            intensity : array
                Intensity counts
            name : str
                Spectrum identifier
            """
            self.BE = np.array(binding_energy)
            self.intensity = np.array(intensity)
            self.name = name
            self.background = None
            self.intensity_bg_subtracted = None
            self.fit_results = None
    
        def smooth(self, window_length=11, polyorder=3):
            """Apply Savitzky-Golay smoothing."""
            self.intensity = savgol_filter(self.intensity, window_length, polyorder)
            return self
    
        def calibrate(self, reference_peak_observed, reference_peak_true):
            """
            Apply charge correction.
    
            Parameters
            ----------
            reference_peak_observed : float
                Observed position of reference peak (eV)
            reference_peak_true : float
                True position of reference peak (eV)
            """
            shift = reference_peak_observed - reference_peak_true
            self.BE = self.BE - shift
            print(f"Charge correction applied: {shift:.2f} eV shift")
            return self
    
        def shirley_background(self, tol=1e-5, max_iter=50):
            """Calculate and store Shirley background."""
            y = self.intensity
            y_left = y[0]
            y_right = y[-1]
    
            bg = np.linspace(y_left, y_right, len(y))
    
            for _ in range(max_iter):
                bg_old = bg.copy()
                cumsum = np.cumsum((y - bg)[::-1])[::-1]
                k = (y_left - y_right) / cumsum[0] if cumsum[0] != 0 else 0
                bg = y_right + k * cumsum
                if np.max(np.abs(bg - bg_old)) < tol:
                    break
    
            self.background = bg
            self.intensity_bg_subtracted = self.intensity - bg
            return self
    
        def fit_peaks(self, peak_guesses, model='voigt'):
            """
            Fit multiple peaks to the spectrum.
    
            Parameters
            ----------
            peak_guesses : list of dict
                Each dict: {'center': float, 'amplitude': float, 'label': str}
            model : str
                Peak model ('gaussian', 'lorentzian', 'voigt')
            """
            from scipy.special import voigt_profile
    
            def voigt_peak(x, amplitude, center, sigma, gamma):
                z = (x - center) / (sigma * np.sqrt(2))
                return amplitude * voigt_profile(z, gamma / (sigma * np.sqrt(2)))
    
            def multi_voigt(x, *params):
                n_peaks = len(params) // 4
                result = np.zeros_like(x, dtype=float)
                for i in range(n_peaks):
                    result += voigt_peak(x, params[4*i], params[4*i+1],
                                        params[4*i+2], params[4*i+3])
                return result
    
            # Prepare initial parameters
            p0 = []
            labels = []
            for guess in peak_guesses:
                p0.extend([guess.get('amplitude', 500), guess['center'], 0.6, 0.3])
                labels.append(guess.get('label', f"Peak at {guess['center']:.1f}"))
    
            # Set bounds
            lower_bounds = []
            upper_bounds = []
            for guess in peak_guesses:
                lower_bounds.extend([0, guess['center'] - 2, 0.2, 0.05])
                upper_bounds.extend([10000, guess['center'] + 2, 2.0, 1.0])
    
            # Fit
            data = self.intensity_bg_subtracted if self.intensity_bg_subtracted is not None \
                   else self.intensity
    
            try:
                popt, pcov = curve_fit(multi_voigt, self.BE, data, p0=p0,
                                       bounds=(lower_bounds, upper_bounds), maxfev=10000)
    
                # Store results
                self.fit_results = {
                    'parameters': popt,
                    'covariance': pcov,
                    'labels': labels,
                    'n_peaks': len(peak_guesses),
                    'fitted_total': multi_voigt(self.BE, *popt)
                }
    
                # Calculate individual peaks and areas
                peaks = []
                areas = []
                for i in range(len(peak_guesses)):
                    peak = voigt_peak(self.BE, popt[4*i], popt[4*i+1],
                                     popt[4*i+2], popt[4*i+3])
                    peaks.append(peak)
                    areas.append(np.trapz(peak, self.BE))
    
                self.fit_results['individual_peaks'] = peaks
                self.fit_results['areas'] = areas
    
                print(f"Peak fitting successful: {len(peak_guesses)} peaks fitted")
    
            except Exception as e:
                print(f"Peak fitting failed: {e}")
                self.fit_results = None
    
            return self
    
        def get_quantification(self, rsf_values):
            """
            Calculate atomic percentages from fitted peak areas.
    
            Parameters
            ----------
            rsf_values : list
                RSF values for each peak (same order as peak_guesses)
            """
            if self.fit_results is None:
                print("No fit results available. Run fit_peaks first.")
                return None
    
            areas = self.fit_results['areas']
            normalized = [area / rsf for area, rsf in zip(areas, rsf_values)]
            total = sum(normalized)
            percentages = [100 * n / total for n in normalized]
    
            return dict(zip(self.fit_results['labels'], percentages))
    
        def plot(self, show_fit=True, show_background=True):
            """Generate comprehensive plot of the spectrum."""
            fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})
    
            # Main spectrum plot
            ax1 = axes[0]
            ax1.plot(self.BE, self.intensity, 'b-', linewidth=1, label='Raw Data')
    
            if show_background and self.background is not None:
                ax1.plot(self.BE, self.background, 'g--', linewidth=2,
                        label='Shirley Background')
                ax1.fill_between(self.BE, self.background, alpha=0.2, color='green')
    
            if show_fit and self.fit_results is not None:
                ax1.plot(self.BE, self.fit_results['fitted_total'] +
                        (self.background if self.background is not None else 0),
                        'r-', linewidth=2, label='Fitted Total')
    
                colors = plt.cm.Set1(np.linspace(0, 1, len(self.fit_results['individual_peaks'])))
                for i, (peak, label, color) in enumerate(zip(
                        self.fit_results['individual_peaks'],
                        self.fit_results['labels'],
                        colors)):
                    bg = self.background if self.background is not None else 0
                    ax1.fill_between(self.BE, bg, peak + bg, alpha=0.4,
                                    color=color, label=label)
    
            ax1.set_xlabel('Binding Energy (eV)', fontsize=12)
            ax1.set_ylabel('Intensity (counts)', fontsize=12)
            ax1.set_title(f'XPS Spectrum: {self.name}', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.invert_xaxis()
            ax1.grid(alpha=0.3)
    
            # Residuals plot
            ax2 = axes[1]
            if self.fit_results is not None and self.intensity_bg_subtracted is not None:
                residuals = self.intensity_bg_subtracted - self.fit_results['fitted_total']
                ax2.plot(self.BE, residuals, 'g-', linewidth=1)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2.fill_between(self.BE, residuals, alpha=0.3, color='green')
                ax2.set_ylabel('Residual', fontsize=11)
            else:
                ax2.text(0.5, 0.5, 'No fit available', ha='center', va='center',
                        transform=ax2.transAxes)
    
            ax2.set_xlabel('Binding Energy (eV)', fontsize=12)
            ax2.invert_xaxis()
            ax2.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
            return fig, axes
    
    # Example: Complete analysis workflow
    print("=== Complete XPS Data Processing Pipeline ===\n")
    
    # Generate synthetic C 1s spectrum
    np.random.seed(42)
    BE = np.linspace(280, 295, 750)
    
    # True peaks (polymer with multiple functional groups)
    true_spectrum = (
        800 * np.exp(-0.5 * ((BE - 285.0) / 0.7)**2) +  # C-C/C-H
        350 * np.exp(-0.5 * ((BE - 286.5) / 0.7)**2) +  # C-O
        180 * np.exp(-0.5 * ((BE - 288.0) / 0.75)**2) + # C=O
        90 * np.exp(-0.5 * ((BE - 289.5) / 0.75)**2)    # O-C=O
    )
    
    # Add Shirley-like background
    bg_level = 40 + 25 * (1 - np.exp(-(BE - 280) / 12))
    spectrum_with_bg = true_spectrum + bg_level
    
    # Add noise and simulate charging (+1.5 eV shift)
    charge_shift = 1.5
    BE_charged = BE + charge_shift
    noise = np.random.normal(0, 12, len(BE))
    measured_spectrum = spectrum_with_bg + noise
    
    # Create XPS spectrum object
    xps = XPSSpectrum(BE_charged, measured_spectrum, name="C 1s - Polymer Sample")
    
    # Processing pipeline
    print("1. Applying smoothing filter...")
    xps.smooth(window_length=7, polyorder=3)
    
    print("2. Applying charge correction (C-C reference at 285.0 eV)...")
    # Find maximum (should be near C-C peak)
    max_idx = np.argmax(xps.intensity)
    observed_cc = xps.BE[max_idx]
    xps.calibrate(observed_cc, 285.0)
    
    print("3. Calculating Shirley background...")
    xps.shirley_background()
    
    print("4. Fitting peaks with Voigt functions...")
    peak_guesses = [
        {'center': 285.0, 'amplitude': 700, 'label': 'C-C/C-H'},
        {'center': 286.5, 'amplitude': 300, 'label': 'C-O'},
        {'center': 288.0, 'amplitude': 150, 'label': 'C=O'},
        {'center': 289.5, 'amplitude': 75, 'label': 'O-C=O'},
    ]
    xps.fit_peaks(peak_guesses)
    
    # Generate report
    print("\n5. Generating quantification report...")
    if xps.fit_results is not None:
        n_peaks = xps.fit_results['n_peaks']
        params = xps.fit_results['parameters']
        areas = xps.fit_results['areas']
        labels = xps.fit_results['labels']
    
        total_area = sum(areas)
    
        print("\n" + "=" * 70)
        print(f"{'Peak Fitting Report':^70}")
        print("=" * 70)
        print(f"{'Component':<15} {'Center (eV)':<12} {'FWHM (eV)':<10} {'Area':<12} {'%':<8}")
        print("-" * 70)
    
        for i, label in enumerate(labels):
            center = params[4*i + 1]
            sigma = params[4*i + 2]
            gamma = params[4*i + 3]
            fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2*gamma)**2 + (2.355*sigma)**2)
            pct = 100 * areas[i] / total_area
            print(f"{label:<15} {center:<12.2f} {fwhm:<10.2f} {areas[i]:<12.0f} {pct:<8.1f}")
    
        print("-" * 70)
        print(f"{'Total':<15} {'':<12} {'':<10} {total_area:<12.0f} {'100.0':<8}")
        print("=" * 70)
    
    # Plot results
    print("\n6. Generating visualization...")
    xps.plot()
    
    print("\n=== Pipeline Complete ===")
    

## 5.6 Exercises

### Basic Exercises

#### Exercise 1: Binding Energy Calculation

An XPS measurement using Al K-alpha radiation (1486.6 eV) detected a photoelectron with kinetic energy of 1382.3 eV. The spectrometer work function is 4.5 eV. Calculate the binding energy and identify the element/orbital.

View Solution

**Solution:**

Using Einstein's equation:

\\[ E_{\text{binding}} = h\nu - E_{\text{kinetic}} - \phi = 1486.6 - 1382.3 - 4.5 = 99.8 \text{ eV} \\] 

This binding energy corresponds to **Si 2p** (metallic silicon at ~99.3 eV).
    
    
    # Verification
    h_nu = 1486.6  # eV (Al K-alpha)
    E_kin = 1382.3  # eV
    phi = 4.5  # eV
    E_bind = h_nu - E_kin - phi
    print(f"Binding Energy: {E_bind:.1f} eV")
    print("Element: Si 2p (metallic silicon)")
    

#### Exercise 2: Chemical State Identification

A Si 2p spectrum shows two peaks at 99.3 eV and 103.5 eV. Identify the chemical states and explain the origin of the binding energy difference.

View Solution

**Solution:**

  * **99.3 eV:** Si0 (elemental/metallic silicon)
  * **103.5 eV:** Si4+ (silicon dioxide, SiO2)

The 4.2 eV shift to higher binding energy occurs because:

  1. In SiO2, silicon has oxidation state +4, losing electron density
  2. The effective nuclear charge on remaining electrons increases
  3. Core electrons experience stronger binding to the nucleus

#### Exercise 3: Quantitative Analysis

An XPS measurement of a sample shows the following peak areas: Fe 2p = 18,500, O 1s = 32,000. Using RSF values of Fe 2p = 2.96 and O 1s = 0.66, calculate the atomic concentrations.

View Solution

**Solution:**

Normalized intensities:

\\[ \frac{I_{\text{Fe}}}{S_{\text{Fe}}} = \frac{18500}{2.96} = 6250 \\] \\[ \frac{I_{\text{O}}}{S_{\text{O}}} = \frac{32000}{0.66} = 48485 \\] 

Total = 6250 + 48485 = 54735

Atomic concentrations:

\\[ C_{\text{Fe}} = \frac{6250}{54735} \times 100 = 11.4\% \\] \\[ C_{\text{O}} = \frac{48485}{54735} \times 100 = 88.6\% \\] 
    
    
    I_Fe, RSF_Fe = 18500, 2.96
    I_O, RSF_O = 32000, 0.66
    
    norm_Fe = I_Fe / RSF_Fe
    norm_O = I_O / RSF_O
    total = norm_Fe + norm_O
    
    C_Fe = 100 * norm_Fe / total
    C_O = 100 * norm_O / total
    
    print(f"Fe: {C_Fe:.1f} at%")
    print(f"O: {C_O:.1f} at%")
    print(f"O/Fe ratio: {C_O/C_Fe:.1f}")
    # Expected for Fe2O3: O/Fe = 1.5
    

### Intermediate Exercises

#### Exercise 4: Peak Fitting Challenge

Write a Python function to fit a C 1s spectrum with 3 peaks (C-C at 285.0 eV, C-O at 286.5 eV, C=O at 288.0 eV) using Gaussian functions. Calculate the relative percentages of each chemical state.

View Solution
    
    
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    def gaussian(x, amp, center, sigma):
        return amp * np.exp(-0.5 * ((x - center) / sigma)**2)
    
    def triple_gaussian(x, a1, c1, s1, a2, c2, s2, a3, c3, s3):
        return (gaussian(x, a1, c1, s1) +
                gaussian(x, a2, c2, s2) +
                gaussian(x, a3, c3, s3))
    
    # Generate synthetic data
    np.random.seed(42)
    BE = np.linspace(282, 292, 500)
    true_spectrum = (gaussian(BE, 800, 285.0, 0.7) +
                     gaussian(BE, 350, 286.5, 0.75) +
                     gaussian(BE, 150, 288.0, 0.8))
    data = true_spectrum + np.random.normal(0, 15, len(BE))
    
    # Initial guesses and bounds
    p0 = [700, 285.0, 0.7, 300, 286.5, 0.7, 100, 288.0, 0.7]
    bounds = ([0, 284, 0.3]*3, [2000, 290, 1.5]*3)
    
    # Fit
    popt, _ = curve_fit(triple_gaussian, BE, data, p0=p0, bounds=bounds)
    
    # Calculate areas and percentages
    peaks = [gaussian(BE, popt[i*3], popt[i*3+1], popt[i*3+2]) for i in range(3)]
    areas = [np.trapz(p, BE) for p in peaks]
    total = sum(areas)
    percentages = [100*a/total for a in areas]
    
    labels = ['C-C/C-H', 'C-O', 'C=O']
    for label, pct in zip(labels, percentages):
        print(f"{label}: {pct:.1f}%")
    

#### Exercise 5: ARXPS Layer Thickness

ARXPS measurements of a thin oxide layer on metal show the following oxide/metal intensity ratios: 0.45 at 90 degrees, 0.82 at 45 degrees, and 2.1 at 20 degrees. Estimate the oxide layer thickness assuming IMFP = 2.5 nm.

View Solution
    
    
    import numpy as np
    
    # Data
    angles = np.array([90, 45, 20])
    ratios = np.array([0.45, 0.82, 2.1])
    imfp = 2.5  # nm
    
    # Linearized analysis: ln(ratio) vs 1/sin(theta)
    x = 1 / np.sin(np.radians(angles))
    y = np.log(ratios)
    
    # Linear fit
    slope, intercept = np.polyfit(x, y, 1)
    
    # Thickness from slope
    thickness = slope * imfp
    
    print(f"Fitted slope: {slope:.3f}")
    print(f"Calculated layer thickness: {thickness:.2f} nm")
    

**Result:** Layer thickness is approximately 3.2 nm

### Advanced Exercises

#### Exercise 6: Complete Analysis Pipeline

Develop a Python class that can:

  1. Load XPS data from a CSV file
  2. Apply Savitzky-Golay smoothing
  3. Calculate and subtract Shirley background
  4. Fit multiple Voigt peaks
  5. Generate a quantification report

View Solution Outline

See Code Example 7 for a complete implementation. Key components:

  * `__init__`: Initialize with BE and intensity arrays
  * `smooth()`: Apply Savitzky-Golay filter
  * `calibrate()`: Apply charge correction
  * `shirley_background()`: Iterative background calculation
  * `fit_peaks()`: Multi-peak Voigt fitting with bounds
  * `get_quantification()`: RSF-corrected atomic percentages
  * `plot()`: Visualization with residuals

#### Exercise 7: Depth Profile Deconvolution

A sputter depth profile shows O concentration decreasing from 67% at the surface to 0% at 15 nm depth, with an interface width of 3 nm. Write code to model this profile and extract the interface parameters from noisy data.

View Solution Outline

See Code Example 5 for the full implementation using error functions. Key steps:

  1. Model concentration profile with erf function
  2. Add realistic noise to simulated data
  3. Use curve_fit to extract interface position and width
  4. Compare fitted vs true parameters

## Chapter Summary

In this chapter, you learned:

  * The photoelectric effect and Einstein's equation relating kinetic and binding energies
  * XPS instrumentation including X-ray sources, hemispherical analyzers, and detectors
  * Chemical state analysis through binding energy shifts and peak deconvolution
  * Peak fitting with Voigt functions for accurate lineshape modeling
  * Quantitative analysis using relative sensitivity factors
  * Depth profiling through sputter profiling and angle-resolved XPS
  * Complete data processing pipelines in Python

## References

  1. Briggs, D., Seah, M.P. (1990). _Practical Surface Analysis, Volume 1: Auger and X-ray Photoelectron Spectroscopy_ (2nd ed.). Wiley. - Comprehensive guide to XPS principles and practice.
  2. Moulder, J.F., Stickle, W.F., Sobol, P.E., Bomben, K.D. (1992). _Handbook of X-ray Photoelectron Spectroscopy_. Physical Electronics. - Standard reference for XPS binding energies.
  3. Scofield, J.H. (1976). Hartree-Slater subshell photoionization cross-sections at 1254 and 1487 eV. _Journal of Electron Spectroscopy and Related Phenomena_ , 8(2), 129-137. - Source of RSF values.
  4. Shirley, D.A. (1972). High-resolution X-ray photoemission spectrum of the valence bands of gold. _Physical Review B_ , 5(12), 4709-4714. - Original Shirley background method.
  5. Powell, C.J., Jablonski, A. (2010). NIST Electron Inelastic-Mean-Free-Path Database. NIST. - IMFP values for depth analysis.
  6. Hufner, S. (2003). _Photoelectron Spectroscopy: Principles and Applications_ (3rd ed.). Springer. - Theoretical foundations of photoemission.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice.
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
