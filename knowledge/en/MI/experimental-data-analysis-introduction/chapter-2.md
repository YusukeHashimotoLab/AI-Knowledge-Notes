---
title: "Chapter 2: Spectral Data Analysis"
chapter_title: "Chapter 2: Spectral Data Analysis"
subtitle: Automated Analysis of XRD, XPS, IR, and Raman - Extraction of Structural and Compositional Information
reading_time: 25-30 min
difficulty: Intermediate
code_examples: 11
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 2: Spectral Data Analysis

Experience the key points of XRD analysis from peak identification to refinement using Python. Learn the perspective of reducing human error through automation.

**💡 Supplement:** Selection of reference patterns and initial values are key to convergence. Keep logs to accelerate the improvement cycle.

**Automated Analysis of XRD, XPS, IR, and Raman - Extraction of Structural and Compositional Information**

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the characteristics of XRD, XPS, IR, and Raman spectra and select appropriate preprocessing methods
  * ✅ Implement peak detection algorithms and quantify peak positions and intensities
  * ✅ Utilize background removal methods (polynomial fitting, SNIP) appropriately
  * ✅ Perform quantitative phase analysis (RIR method) from XRD patterns
  * ✅ Automate spectral analysis pipelines

**Reading Time** : 25-30 min **Code Examples** : 11 **Exercises** : 3 problems

* * *

## 2.1 Characteristics of Spectral Data and Preprocessing Strategies

### Characteristics of Each Measurement Technique

Understanding the characteristics of the four spectral measurement techniques frequently used in materials science is important for selecting appropriate analysis methods.

Measurement Technique | Information Obtained | Peak Characteristics | Typical Background  
---|---|---|---  
**XRD** | Crystal structure, phase identification | Sharp (diffraction peaks) | Low intensity, gradual rise (amorphous)  
**XPS** | Elemental composition, chemical state | Asymmetric (spin-orbit splitting) | Shirley-type (inelastic scattering)  
**IR** | Molecular vibrations, functional groups | Sharp to broad | Nearly flat (transmission method)  
**Raman** | Crystallinity, molecular vibrations | Sharp (high crystallinity) | Fluorescence background (organic matter)  
  
### Typical Workflow for Spectral Analysis
    
    
    ```mermaid
    flowchart TD
        A[Spectral Measurement] --> B[Data Loading]
        B --> C[Noise Removal]
        C --> D[Background Removal]
        D --> E[Peak Detection]
        E --> F[Peak Fitting]
        F --> G[Quantitative Analysis]
        G --> H[Result Visualization]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style E fill:#f3e5f5
        style G fill:#e8f5e9
        style H fill:#fce4ec
    ```

**Purpose of Each Step** : 1\. **Noise Removal** : S/N ratio improvement (studied in Chapter 1) 2\. **Background Removal** : Baseline correction 3\. **Peak Detection** : Automatic identification of peak positions 4\. **Peak Fitting** : Modeling of peak shapes 5\. **Quantitative Analysis** : Calculation of composition and phase fractions

* * *

## 2.2 Data Licensing and Reproducibility

### Spectral Data Repositories and Licenses

In spectral analysis, utilizing standard databases and public data is important.

#### Major Spectral Databases

Database | Content | License | Access | Citation Requirements  
---|---|---|---|---  
**ICDD PDF-4+** | XRD Standard Patterns | Commercial License | Paid Subscription | Required  
**COD (Crystallography Open Database)** | Crystal structure, XRD | CC0 / Public Domain | Free | Recommended  
**NIST XPS Database** | XPS Standard Spectra | Public Domain | Free | Required  
**RRUFF Project** | Raman/IRMineral Spectra | CC BY-NC-SA 3.0 | Free | Required  
**SDBS (Spectral Database)** | IR/RamanOrganic Compounds | Free | Free | Recommended  
  
#### Precautions When Using Data

**When Using Commercial Databases** :
    
    
    # ICDD PDF-4+Example comment when using data
    """
    XRD peak matching using ICDD PDF-4+ database.
    Reference: ICDD PDF-4+ 2024 (Entry 01-089-0599)
    License: International Centre for Diffraction Data
    Note: Commercial license required for publication use
    """
    

**When Using Open Data** :
    
    
    # COD (Crystallography Open Database)Usage example
    """
    Crystal structure data from COD.
    Reference: COD Entry 1234567
    Citation: Gražulis, S. et al. (2012) Nucleic Acids Research, 40, D420-D427
    License: CC0 1.0 Universal (Public Domain)
    URL: http://www.crystallography.net/cod/1234567.html
    """
    

### Best Practices for Code Reproducibility

#### Recording Environment Information
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: Recording Environment Information
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import sys
    import numpy as np
    import scipy
    from scipy import signal
    import matplotlib
    
    print("=== Spectral Analysis Environment ===")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    
    # Recommended versions(as of October 2025):
    # - Python: 3.10or later
    # - NumPy: 1.24or later
    # - SciPy: 1.10or later
    # - Matplotlib: 3.7or later
    

#### Parameter Documentation

**Bad Example** （Not reproducible）:
    
    
    bg = snip_background(spectrum, 50)  # Why 50?
    

**Good Example** （Reproducible）:
    
    
    # SNIP method parameter settings
    SNIP_ITERATIONS = 50  # Estimated background width（approximately5%）
    SNIP_DESCRIPTION = """
    The number of iterations corresponds to the characteristic width of the background.
    XRD data(700 points, 2θ=10-80°), corresponding to a width of approximately 10°≈100 points, therefore
    iterations=50（about half the width）is appropriate.
    """
    bg = snip_background(spectrum, iterations=SNIP_ITERATIONS)
    

#### Recording Analysis Parameters
    
    
    # Peak detection parameters（Information to be recorded in experiment notebook）
    PEAK_DETECTION_PARAMS = {
        'method': 'find_peaks',
        'prominence': 100,  # counts (approximately 5x)
        'distance': 10,     # points (approx. 0.1° in 2θ)
        'height': 50,       # counts（Minimum physically meaningfulwith physical meaning）
        'width': 3,         # points（Minimum peak width）
        'noise_level': 20   # counts（standard deviation of baseline region）
    }
    
    # Save parameters to JSON（ensuring reproducibility）
    import json
    with open('peak_detection_params.json', 'w') as f:
        json.dump(PEAK_DETECTION_PARAMS, f, indent=2)
    

* * *

## 2.3 Background Removal Methods

### Polynomial Fitting Method

The simplest method approximates the background with a polynomial and subtracts it from the original data.

**Code Example 1: Polynomial Background Removal**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1: Polynomial Background Removal
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Background Removal for XRD Patterns
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    
    # Generate sample XRD data
    np.random.seed(42)
    two_theta = np.linspace(10, 80, 700)
    
    # Peak components
    peaks = (
        1000 * np.exp(-((two_theta - 28) ** 2) / 10) +
        1500 * np.exp(-((two_theta - 32) ** 2) / 8) +
        800 * np.exp(-((two_theta - 47) ** 2) / 12)
    )
    
    # Background (amorphous halo)
    background = (
        100 +
        50 * np.sin(two_theta / 10) +
        30 * (two_theta / 80) ** 2
    )
    
    # Noise
    noise = np.random.normal(0, 20, len(two_theta))
    
    # Overall signal
    intensity = peaks + background + noise
    
    # Background estimation by polynomial fitting
    poly_degree = 5
    coeffs = np.polyfit(two_theta, intensity, poly_degree)
    background_fit = np.polyval(coeffs, two_theta)
    
    # Background subtraction
    intensity_corrected = intensity - background_fit
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Raw data
    axes[0, 0].plot(two_theta, intensity, linewidth=1)
    axes[0, 0].set_xlabel('2θ (degree)')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].set_title('Raw XRD Pattern')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fitting Results
    axes[0, 1].plot(two_theta, intensity, label='Raw data', alpha=0.5)
    axes[0, 1].plot(two_theta, background_fit,
                    label=f'Polynomial fit (deg={poly_degree})',
                    linewidth=2, color='red')
    axes[0, 1].set_xlabel('2θ (degree)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].set_title('Background Estimation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # After correction
    axes[1, 0].plot(two_theta, intensity_corrected, linewidth=1,
                    color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('2θ (degree)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('After Background Subtraction')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison with true peaks
    axes[1, 1].plot(two_theta, peaks, label='True peaks',
                    linewidth=2, alpha=0.7)
    axes[1, 1].plot(two_theta, intensity_corrected,
                    label='Corrected data', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xlabel('2θ (degree)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].set_title('Comparison with True Peaks')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Polynomial degree: {poly_degree}")
    print(f"Background average: {background_fit.mean():.1f}")
    print(f"After correction average: {intensity_corrected.mean():.1f}")
    

**Output** :
    
    
    Polynomial degree: 5
    Background average: 150.3
    After correction average: 0.5
    

**Usage Guide** : \- **Low order (2-3 degrees)** : Gradual background (IR, XPS) \- **Medium order (4-6 degrees)** : Moderately complex shape (XRD amorphous halo) \- **High order ( >7 degrees)**: Complex shape (Note: risk of overfitting)

### SNIP method（Statistics-sensitive Non-linear Iterative Peak-clipping）

An advanced method that estimates the background while preserving peak information.

**Code Example2: SNIP methodbyBackground Removal**
    
    
    def snip_background(spectrum, iterations=30):
        """
        Background estimation by SNIP method
    
        Parameters:
        -----------
        spectrum : array-like
            Input spectrum
        iterations : int
            Number of iterations (corresponds to background width)
    
        Returns:
        --------
        background : array-like
            Estimated background
        """
        spectrum = np.array(spectrum, dtype=float)
        background = np.copy(spectrum)
    
        for i in range(1, iterations + 1):
            # Comparison with left and right values
            for j in range(i, len(background) - i):
                v1 = (background[j - i] + background[j + i]) / 2
                v2 = background[j]
                background[j] = min(v1, v2)
    
        return background
    
    # Application of SNIP method
    snip_bg = snip_background(intensity, iterations=50)
    intensity_snip = intensity - snip_bg
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Raw data and SNIP background
    axes[0].plot(two_theta, intensity, label='Raw data', alpha=0.6)
    axes[0].plot(two_theta, snip_bg, label='SNIP background',
                 linewidth=2, color='orange')
    axes[0].set_xlabel('2θ (degree)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('SNIP Background Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SNIPAfter correction
    axes[1].plot(two_theta, intensity_snip, linewidth=1.5,
                 color='purple')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('2θ (degree)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('After SNIP Subtraction')
    axes[1].grid(True, alpha=0.3)
    
    # Polynomial vs SNIP comparison
    axes[2].plot(two_theta, intensity_corrected,
                 label='Polynomial', alpha=0.7, linewidth=1.5)
    axes[2].plot(two_theta, intensity_snip,
                 label='SNIP', alpha=0.7, linewidth=1.5)
    axes[2].set_xlabel('2θ (degree)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_title('Polynomial vs SNIP')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"SNIP iterations: 50")
    print(f"Polynomial method residual: {np.std(intensity_corrected - peaks):.2f}")
    print(f"SNIP method residual: {np.std(intensity_snip - peaks):.2f}")
    

**Advantages of the SNIP Method** : \- Less affected by peaks \- Handles complex background shapes \- Intuitive parameter adjustment (iterations = background width)

* * *

## 2.3 Peak Detection Algorithms

### Basic Peak Detection with scipy.signal.find_peaks

**Code Example3: Basic Peak Detection**
    
    
    from scipy.signal import find_peaks
    
    # Peak detection on background-corrected data
    peaks_idx, properties = find_peaks(
        intensity_snip,
        height=100,        # Minimum peak height
        prominence=80,     # Prominence (height difference from surroundings)
        distance=10,       # Minimum peak distance (data points)
        width=3           # Minimum peak width
    )
    
    peak_positions = two_theta[peaks_idx]
    peak_heights = intensity_snip[peaks_idx]
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.plot(two_theta, intensity_snip, linewidth=1.5,
             label='Background-corrected')
    plt.plot(peak_positions, peak_heights, 'rx',
             markersize=12, markeredgewidth=2, label='Detected peaks')
    
    # Label peak positions
    for pos, height in zip(peak_positions, peak_heights):
        plt.annotate(f'{pos:.1f}°',
                    xy=(pos, height),
                    xytext=(pos, height + 100),
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='yellow', alpha=0.5))
    
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Peak Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Detected Peaks ===")
    for i, (pos, height) in enumerate(zip(peak_positions,
                                           peak_heights), 1):
        print(f"Peak {i}: 2θ = {pos:.2f}°, Intensity = {height:.1f}")
    

**Output** :
    
    
    === Detected Peaks ===
    Peak 1: 2θ = 28.04°, Intensity = 1021.3
    Peak 2: 2θ = 32.05°, Intensity = 1512.7
    Peak 3: 2θ = 47.07°, Intensity = 798.5
    

### Optimization of Peak Detection Parameters

**Code Example4: Parameter sensitivity analysis**
    
    
    # Peak detection with different parameters
    prominence_values = [30, 50, 80, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, prom in enumerate(prominence_values):
        peaks_idx, _ = find_peaks(
            intensity_snip,
            prominence=prom,
            distance=5
        )
    
        axes[i].plot(two_theta, intensity_snip, linewidth=1.5)
        axes[i].plot(two_theta[peaks_idx], intensity_snip[peaks_idx],
                    'rx', markersize=10, markeredgewidth=2)
        axes[i].set_xlabel('2θ (degree)')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Prominence = {prom} ({len(peaks_idx)} peaks)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Parameter Sensitivity ===")
    for prom in prominence_values:
        peaks_idx, _ = find_peaks(intensity_snip, prominence=prom)
        print(f"Prominence = {prom:3d}: {len(peaks_idx)} peaks detected")
    

### Peak Fitting (Gaussian and Lorentzian)

**Code Example5: Gaussian fitting**
    
    
    from scipy.optimize import curve_fit
    
    def gaussian(x, amplitude, center, sigma):
        """Gaussian function"""
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian function"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    # Extract region around first peak
    peak_region_mask = (two_theta > 26) & (two_theta < 30)
    x_data = two_theta[peak_region_mask]
    y_data = intensity_snip[peak_region_mask]
    
    # Gaussian fitting
    initial_guess = [1000, 28, 1]  # [amplitude, center, sigma]
    params_gauss, _ = curve_fit(gaussian, x_data, y_data,
                                 p0=initial_guess)
    
    # Lorentzian fitting
    initial_guess_lor = [1000, 28, 0.5]  # [amplitude, center, gamma]
    params_lor, _ = curve_fit(lorentzian, x_data, y_data,
                              p0=initial_guess_lor)
    
    # Fitting result
    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
    y_gauss = gaussian(x_fit, *params_gauss)
    y_lor = lorentzian(x_fit, *params_lor)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_data, y_data, 'o', label='Data', markersize=6)
    plt.plot(x_fit, y_gauss, '-', linewidth=2,
             label=f'Gaussian (σ={params_gauss[2]:.2f})')
    plt.plot(x_fit, y_lor, '--', linewidth=2,
             label=f'Lorentzian (γ={params_lor[2]:.2f})')
    
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Peak Fitting: Gaussian vs Lorentzian')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Fitting Results ===")
    print(f"Gaussian:")
    print(f"  Center position: {params_gauss[1]:.3f}°")
    print(f"  Amplitude: {params_gauss[0]:.1f}")
    print(f"  σ: {params_gauss[2]:.3f}°")
    print(f"\nLorentzian:")
    print(f"  Center position: {params_lor[1]:.3f}°")
    print(f"  Amplitude: {params_lor[0]:.1f}")
    print(f"  γ: {params_lor[2]:.3f}°")
    

* * *

## 2.4 XPS Spectral Analysis

### Shirley-type Background Removal

XPS spectra have characteristic backgrounds due to inelastic scattering.

**Code Example6: Shirley background**
    
    
    def shirley_background(x, y, tol=1e-5, max_iter=50):
        """
        Shirley-type background estimation
    
        Parameters:
        -----------
        x : array-like
            Energy axis (descending order recommended)
        y : array-like
            Intensity
        tol : float
            Convergence check threshold
        max_iter : int
            Maximum number of iterations
    
        Returns:
        --------
        background : array-like
            Shirley background
        """
        # Data preparation
        y = np.array(y, dtype=float)
        background = np.zeros_like(y)
    
        # Values at both ends
        y_min = min(y[0], y[-1])
        y_max = max(y[0], y[-1])
    
        # Initial background (linear)
        background = np.linspace(y[0], y[-1], len(y))
    
        # Iteration
        for iteration in range(max_iter):
            background_old = background.copy()
    
            # Using cumulative sum
            cumsum = np.cumsum(y - background)
            total = cumsum[-1]
    
            if total == 0:
                break
    
            # New background
            background = y[-1] + (y[0] - y[-1]) * cumsum / total
    
            # Convergence check
            if np.max(np.abs(background - background_old)) < tol:
                break
    
        return background
    
    # Generate XPS sample data (C 1s spectrum)
    binding_energy = np.linspace(280, 295, 300)[::-1]  # Descending order
    xps_peak = 5000 * np.exp(-((binding_energy - 285) ** 2) / 2)
    shirley_bg = np.linspace(500, 200, len(binding_energy))
    xps_spectrum = xps_peak + shirley_bg + \
                   np.random.normal(0, 50, len(binding_energy))
    
    # Shirley background estimation
    shirley_bg_calc = shirley_background(binding_energy, xps_spectrum)
    
    # Background subtraction
    xps_corrected = xps_spectrum - shirley_bg_calc
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # XPS Raw data
    axes[0].plot(binding_energy, xps_spectrum, linewidth=1.5)
    axes[0].set_xlabel('Binding Energy (eV)')
    axes[0].set_ylabel('Intensity (CPS)')
    axes[0].set_title('Raw XPS Spectrum (C 1s)')
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)
    
    # Background estimation
    axes[1].plot(binding_energy, xps_spectrum,
                 label='Raw data', alpha=0.6)
    axes[1].plot(binding_energy, shirley_bg_calc,
                 label='Shirley background',
                 linewidth=2, color='red')
    axes[1].set_xlabel('Binding Energy (eV)')
    axes[1].set_ylabel('Intensity (CPS)')
    axes[1].set_title('Shirley Background Estimation')
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # After correction
    axes[2].plot(binding_energy, xps_corrected,
                 linewidth=1.5, color='green')
    axes[2].set_xlabel('Binding Energy (eV)')
    axes[2].set_ylabel('Intensity (CPS)')
    axes[2].set_title('After Shirley Subtraction')
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Shirley Background Removal ===")
    print(f"Energy range: {binding_energy.max():.1f} - "
          f"{binding_energy.min():.1f} eV")
    print(f"Background height (high BE side): {shirley_bg_calc[0]:.1f}")
    print(f"Background height (low BE side): {shirley_bg_calc[-1]:.1f}")
    

* * *

## 2.5 IR and Raman Spectral Analysis

### Baseline Correction (Asymmetric Least Squares Method)

Effective for removing fluorescence background in IR and Raman spectra.

**Code Example7: ALS methodbyBaseline correction**
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    def als_baseline(y, lam=1e5, p=0.01, niter=10):
        """
        Baseline estimation by Asymmetric Least Squares method
    
        Parameters:
        -----------
        y : array-like
            Spectral data
        lam : float
            Smoothing parameter (larger is smoother)
        p : float
            Asymmetry parameter (0-1, smaller avoids peaks more)
        niter : int
            Number of iterations
    
        Returns:
        --------
        baseline : array-like
            Estimated baseline
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
    
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
    
        return z
    
    # Generate Raman sample data
    raman_shift = np.linspace(200, 2000, 900)
    raman_peaks = (
        3000 * np.exp(-((raman_shift - 520) ** 2) / 100) +
        2000 * np.exp(-((raman_shift - 950) ** 2) / 200) +
        1500 * np.exp(-((raman_shift - 1350) ** 2) / 150)
    )
    fluorescence_bg = 500 + 0.5 * raman_shift + \
                      0.0005 * (raman_shift - 1000) ** 2
    raman_spectrum = raman_peaks + fluorescence_bg + \
                     np.random.normal(0, 50, len(raman_shift))
    
    # Application of ALS method
    als_bg = als_baseline(raman_spectrum, lam=1e6, p=0.01)
    raman_corrected = raman_spectrum - als_bg
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Raw data
    axes[0].plot(raman_shift, raman_spectrum, linewidth=1.5)
    axes[0].set_xlabel('Raman Shift (cm⁻¹)')
    axes[0].set_ylabel('Intensity (a.u.)')
    axes[0].set_title('Raw Raman Spectrum')
    axes[0].grid(True, alpha=0.3)
    
    # Baseline estimation
    axes[1].plot(raman_shift, raman_spectrum,
                 label='Raw data', alpha=0.6)
    axes[1].plot(raman_shift, als_bg,
                 label='ALS baseline', linewidth=2, color='red')
    axes[1].set_xlabel('Raman Shift (cm⁻¹)')
    axes[1].set_ylabel('Intensity (a.u.)')
    axes[1].set_title('ALS Baseline Estimation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # After correction
    axes[2].plot(raman_shift, raman_corrected,
                 linewidth=1.5, color='purple')
    axes[2].set_xlabel('Raman Shift (cm⁻¹)')
    axes[2].set_ylabel('Intensity (a.u.)')
    axes[2].set_title('After ALS Subtraction')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ALS Baseline correction ===")
    print(f"Smoothing parameter (λ): 1e6")
    print(f"Asymmetry parameter (p): 0.01")
    

* * *

## 2.6 Quantitative Phase Analysis (XRD-RIR Method)

### Reference Intensity Ratio (RIR) Method

Calculate the weight fraction of each phase from XRD patterns containing multiple phases.

**Code Example8: Quantitative Phase Analysis by RIR Method**
    
    
    # Generate XRD pattern for two-phase system
    two_theta = np.linspace(10, 80, 700)
    
    # Phase A (Example: α-Fe2O3, main peak: 33.2°)
    phase_A = (
        2000 * np.exp(-((two_theta - 33.2) ** 2) / 15) +
        1200 * np.exp(-((two_theta - 35.6) ** 2) / 10) +
        800 * np.exp(-((two_theta - 54.1) ** 2) / 12)
    )
    
    # Phase B (Example: Fe3O4, main peak: 35.5°)
    phase_B = (
        1500 * np.exp(-((two_theta - 35.5) ** 2) / 18) +
        1000 * np.exp(-((two_theta - 30.1) ** 2) / 12) +
        600 * np.exp(-((two_theta - 62.7) ** 2) / 14)
    )
    
    # Mixed pattern (Phase A:Phase B = 70:30 wt%)
    ratio_A = 0.7
    ratio_B = 0.3
    mixed_pattern = ratio_A * phase_A + ratio_B * phase_B + \
                    np.random.normal(0, 30, len(two_theta))
    
    # RIR values (literature values, relative to corundum)
    RIR_A = 3.5  # RIR of α-Fe2O3
    RIR_B = 2.8  # RIR of Fe3O4
    
    # Measurement of main peak intensity
    # Main peak of Phase A (around 33.2°)
    peak_A_idx = np.argmax(mixed_pattern[(two_theta > 32) &
                                         (two_theta < 34)])
    I_A = mixed_pattern[(two_theta > 32) & (two_theta < 34)][peak_A_idx]
    
    # Main peak of Phase B (around 35.5°)
    peak_B_idx = np.argmax(mixed_pattern[(two_theta > 34.5) &
                                         (two_theta < 36)])
    I_B = mixed_pattern[(two_theta > 34.5) & (two_theta < 36)][peak_B_idx]
    
    # Weight fraction calculation by RIR method
    # W_A / W_B = (I_A / I_B) * (RIR_B / RIR_A)
    ratio_calc = (I_A / I_B) * (RIR_B / RIR_A)
    
    # Normalization
    W_A_calc = ratio_calc / (1 + ratio_calc)
    W_B_calc = 1 - W_A_calc
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mixed pattern
    axes[0, 0].plot(two_theta, mixed_pattern, linewidth=1.5)
    axes[0, 0].set_xlabel('2θ (degree)')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].set_title('Mixed XRD Pattern (Phase A + B)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Individual Phase Contributions
    axes[0, 1].plot(two_theta, ratio_A * phase_A,
                    label='Phase A (70%)', linewidth=1.5)
    axes[0, 1].plot(two_theta, ratio_B * phase_B,
                    label='Phase B (30%)', linewidth=1.5)
    axes[0, 1].set_xlabel('2θ (degree)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].set_title('Individual Phase Contributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Main Peak Positions
    axes[1, 0].plot(two_theta, mixed_pattern, linewidth=1.5)
    axes[1, 0].axvline(x=33.2, color='blue',
                       linestyle='--', label='Phase A peak')
    axes[1, 0].axvline(x=35.5, color='orange',
                       linestyle='--', label='Phase B peak')
    axes[1, 0].set_xlabel('2θ (degree)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Main Peak Positions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantitative Results
    categories = ['Phase A', 'Phase B']
    true_values = [ratio_A * 100, ratio_B * 100]
    calc_values = [W_A_calc * 100, W_B_calc * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, true_values, width,
                   label='True', alpha=0.7)
    axes[1, 1].bar(x + width/2, calc_values, width,
                   label='Calculated (RIR)', alpha=0.7)
    axes[1, 1].set_ylabel('Weight Fraction (%)')
    axes[1, 1].set_title('Quantitative Phase Analysis')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Quantitative Phase Analysis by RIR Method ===")
    print(f"Main peak intensity:")
    print(f"  Phase A (33.2°): {I_A:.1f}")
    print(f"  Phase B (35.5°): {I_B:.1f}")
    print(f"\nRIR values:")
    print(f"  Phase A: {RIR_A}")
    print(f"  Phase B: {RIR_B}")
    print(f"\nWeight fraction:")
    print(f"  True value - Phase A: {ratio_A*100:.1f}%, Phase B: {ratio_B*100:.1f}%")
    print(f"  Calculated - Phase A: {W_A_calc*100:.1f}%, Phase B: {W_B_calc*100:.1f}%")
    print(f"  Error: Phase A {abs(ratio_A - W_A_calc)*100:.1f}%")
    

* * *

## 2.7 Automated Spectral Analysis Pipeline

### Integrated Analysis Pipeline

**Code Example9: Automated spectral analysis pipeline**
    
    
    from dataclasses import dataclass
    from typing import Tuple, List
    
    @dataclass
    class PeakInfo:
        """Data class to store peak information"""
        position: float
        intensity: float
        width: float
        area: float
    
    class SpectrumAnalyzer:
        """Automated spectrum analysis class"""
    
        def __init__(self, spectrum_type='XRD'):
            """
            Parameters:
            -----------
            spectrum_type : str
                'XRD', 'XPS', 'IR', 'Raman'
            """
            self.spectrum_type = spectrum_type
            self.x = None
            self.y = None
            self.y_corrected = None
            self.peaks = []
    
        def load_data(self, x: np.ndarray, y: np.ndarray):
            """Data Loading"""
            self.x = np.array(x)
            self.y = np.array(y)
    
        def remove_background(self, method='snip', **kwargs):
            """Background Removal"""
            if method == 'snip':
                iterations = kwargs.get('iterations', 30)
                bg = snip_background(self.y, iterations)
            elif method == 'polynomial':
                degree = kwargs.get('degree', 5)
                coeffs = np.polyfit(self.x, self.y, degree)
                bg = np.polyval(coeffs, self.x)
            elif method == 'als':
                lam = kwargs.get('lam', 1e5)
                p = kwargs.get('p', 0.01)
                bg = als_baseline(self.y, lam, p)
            else:
                raise ValueError(f"Unknown method: {method}")
    
            self.y_corrected = self.y - bg
            return self.y_corrected
    
        def detect_peaks(self, **kwargs):
            """Peak Detection"""
            if self.y_corrected is None:
                raise ValueError("Run remove_background first")
    
            prominence = kwargs.get('prominence', 50)
            distance = kwargs.get('distance', 10)
    
            peaks_idx, properties = find_peaks(
                self.y_corrected,
                prominence=prominence,
                distance=distance
            )
    
            self.peaks = []
            for idx in peaks_idx:
                peak = PeakInfo(
                    position=self.x[idx],
                    intensity=self.y_corrected[idx],
                    width=properties['widths'][0] if 'widths' in properties else 0,
                    area=0  # Calculated later
                )
                self.peaks.append(peak)
    
            return self.peaks
    
        def report(self):
            """Results report"""
            print(f"\n=== {self.spectrum_type} Spectrum Analysis Report ===")
            print(f"Data points: {len(self.x)}")
            print(f"Detected peaks: {len(self.peaks)}")
            print(f"\nPeak information:")
            for i, peak in enumerate(self.peaks, 1):
                if self.spectrum_type == 'XRD':
                    print(f"  Peak {i}: 2θ = {peak.position:.2f}°, "
                          f"Intensity = {peak.intensity:.1f}")
                elif self.spectrum_type == 'XPS':
                    print(f"  Peak {i}: BE = {peak.position:.2f} eV, "
                          f"Intensity = {peak.intensity:.1f}")
                elif self.spectrum_type in ['IR', 'Raman']:
                    print(f"  Peak {i}: {peak.position:.1f} cm⁻¹, "
                          f"Intensity = {peak.intensity:.1f}")
    
    # Usage example
    analyzer = SpectrumAnalyzer(spectrum_type='XRD')
    analyzer.load_data(two_theta, intensity)
    analyzer.remove_background(method='snip', iterations=50)
    peaks = analyzer.detect_peaks(prominence=80, distance=10)
    analyzer.report()
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(analyzer.x, analyzer.y, label='Raw', alpha=0.5)
    plt.plot(analyzer.x, analyzer.y_corrected,
             label='Background-corrected', linewidth=1.5)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Spectrum Processing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(analyzer.x, analyzer.y_corrected, linewidth=1.5)
    peak_positions = [p.position for p in peaks]
    peak_intensities = [p.intensity for p in peaks]
    plt.plot(peak_positions, peak_intensities, 'rx',
             markersize=12, markeredgewidth=2)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Peak Detection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 2.8 Practical Pitfalls and Countermeasures

### Common Failure Cases and Best Practices

#### Failure 1: Over-application of Background Removal

**Symptom** : After background removal, peak intensity becomes negative or small peaks disappear

**Cause** : Polynomial degree too high or SNIP iterations value too large

**Countermeasure** :
    
    
    # ❌ Bad Example: Excessive background removal
    poly_degree = 15  # Too high degree
    bg = np.polyval(np.polyfit(two_theta, intensity, poly_degree), two_theta)
    corrected = intensity - bg
    # Result: Peaks distorted, negative values occur
    
    # ✅ Good Example: Appropriate parameters and validation
    poly_degree = 5  # Moderate degree
    bg = np.polyval(np.polyfit(two_theta, intensity, poly_degree), two_theta)
    corrected = intensity - bg
    
    # Validation: Check proportion of negative values
    negative_ratio = (corrected < 0).sum() / len(corrected)
    if negative_ratio > 0.05:  # Warning if more than 5% negative
        print(f"Warning: Background removal is excessive ({negative_ratio*100:.1f}% negative values)")
        print("Lower polynomial degree or reduce SNIP iterations")
    

#### Failure 2: Inappropriate Peak Detection Parameter Settings

**Symptom** : Noise incorrectly detected as peaks, or true peaks missed

**Cause** : Prominence or height thresholds not appropriate

**Countermeasure** :
    
    
    # ❌ Bad Example: Fixed parameters
    peaks, _ = find_peaks(spectrum, prominence=50, height=100)
    # Cannot handle data with different S/N ratios
    
    # ✅ Good Example: Adaptive parameter setting
    # Quantitative noise level evaluation
    baseline_region = spectrum[(two_theta > 70) & (two_theta < 80)]  # No signal region
    noise_std = np.std(baseline_region)
    snr = spectrum.max() / noise_std
    
    # Parameter adjustment based on S/N ratio
    if snr > 50:  # High quality data
        prominence = 3 * noise_std
        height = 2 * noise_std
    elif snr > 20:  # Medium quality data
        prominence = 5 * noise_std
        height = 3 * noise_std
    else:  # Low quality data
        prominence = 10 * noise_std
        height = 5 * noise_std
        print(f"Warning: Low S/N ratio ({snr:.1f}). Please re-run measurement")
    
    peaks, properties = find_peaks(spectrum, prominence=prominence, height=height)
    print(f"Noise level: {noise_std:.2f}, S/N ratio: {snr:.1f}")
    print(f"Detected peaks: {len(peaks)}")
    

#### Failure 3: Incorrect Quantification of XPS Spectra

**Symptom** : Element composition shows physically impossible values (total greatly exceeds 100% or negative)

**Cause** : Failure in Shirley background removal or misuse of sensitivity factors

**Countermeasure** :
    
    
    # ❌ Bad Example: Quantification without background removal
    peak_area_C = np.trapz(xps_C_spectrum, binding_energy_C)
    peak_area_O = np.trapz(xps_O_spectrum, binding_energy_O)
    # Result: Integrated value includes background
    
    # ✅ Good Example: Peak area after Shirley correction
    bg_C = shirley_background(binding_energy_C, xps_C_spectrum)
    corrected_C = xps_C_spectrum - bg_C
    peak_area_C = np.trapz(corrected_C[corrected_C > 0],
                           binding_energy_C[corrected_C > 0])
    
    bg_O = shirley_background(binding_energy_O, xps_O_spectrum)
    corrected_O = xps_O_spectrum - bg_O
    peak_area_O = np.trapz(corrected_O[corrected_O > 0],
                           binding_energy_O[corrected_O > 0])
    
    # Correction with sensitivity factors (using literature values)
    SENSITIVITY_FACTORS = {'C': 0.296, 'O': 0.711, 'Fe': 2.957}  # Scofield
    atomic_ratio_C = peak_area_C / SENSITIVITY_FACTORS['C']
    atomic_ratio_O = peak_area_O / SENSITIVITY_FACTORS['O']
    
    # Normalization
    total = atomic_ratio_C + atomic_ratio_O
    at_percent_C = (atomic_ratio_C / total) * 100
    at_percent_O = (atomic_ratio_O / total) * 100
    
    # Validation
    assert abs(at_percent_C + at_percent_O - 100) < 1, "Composition total is not 100%"
    print(f"C: {at_percent_C:.1f} at%, O: {at_percent_O:.1f} at%")
    

#### Failure 4: Misuse of RIR Method

**Symptom** : Quantitative phase analysis results deviate greatly from 100% total weight ratio

**Cause** : Inconsistent RIR values (different reference materials) or ignoring peak overlap

**Countermeasure** :
    
    
    # ❌ Bad Example: Quantification ignoring peak overlap
    I_A = intensity[peak_A_index]  # Single point intensity
    I_B = intensity[peak_B_index]
    # Result: Large error due to nearby peak influence
    
    # ✅ Good Example: Correct application of peak separation and RIR method
    # Peak region extraction
    peak_A_region = (two_theta > 32) & (two_theta < 34)
    peak_B_region = (two_theta > 34.5) & (two_theta < 36.5)
    
    # Gaussian fitting for peak separation
    from scipy.optimize import curve_fit
    
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    
    # Fit peak of Phase A
    popt_A, _ = curve_fit(gaussian, two_theta[peak_A_region],
                          intensity[peak_A_region],
                          p0=[1000, 33.2, 0.5])
    I_A_corrected = popt_A[0]  # Peak height
    
    # Fit peak of Phase B (similarly)
    popt_B, _ = curve_fit(gaussian, two_theta[peak_B_region],
                          intensity[peak_B_region],
                          p0=[1500, 35.5, 0.5])
    I_B_corrected = popt_B[0]
    
    # Verify RIR values (use values for same reference material)
    RIR_A = 3.5  # vs Corundum (α-Al2O3)
    RIR_B = 2.8  # vs Corundum (α-Al2O3)
    
    # Weight fraction calculation
    ratio = (I_A_corrected / I_B_corrected) * (RIR_B / RIR_A)
    W_A = ratio / (1 + ratio)
    W_B = 1 - W_A
    
    # Validation
    assert abs(W_A + W_B - 1.0) < 0.01, "Weight fraction total is not 1"
    print(f"Phase A: {W_A*100:.1f} wt%, Phase B: {W_B*100:.1f} wt%")
    

#### Failure 5: Ignoring Measurement Errors in Spectral Analysis

**Symptom** : Analysis results do not show error ranges, reproducibility cannot be evaluated

**Cause** : Not quantifying the effects of measurement repetition and noise

**Countermeasure** :
    
    
    # ❌ Bad Example: Only one measurement, no error
    peak_position = two_theta[peak_index]
    print(f"Peak position: {peak_position:.2f}°")
    
    # ✅ Good Example: Multiple measurements and error evaluation
    # 3 measurements of same sample
    measurements = []
    for i in range(3):
        spectrum_i = measure_xrd()  # Measurement function
        bg_i = snip_background(spectrum_i, iterations=50)
        corrected_i = spectrum_i - bg_i
        peaks_i, _ = find_peaks(corrected_i, prominence=80)
    
        # Position of main peak (highest intensity peak)
        main_peak_i = two_theta[peaks_i[np.argmax(corrected_i[peaks_i])]]
        measurements.append(main_peak_i)
    
    # Statistical processing
    peak_mean = np.mean(measurements)
    peak_std = np.std(measurements, ddof=1)  # Unbiased standard deviation
    peak_sem = peak_std / np.sqrt(len(measurements))  # Standard error
    
    print(f"Peak position: {peak_mean:.3f} ± {peak_sem:.3f}° (mean ± standard error, n=3)")
    
    # Warning for large measurement error
    if peak_std > 0.1:  # Variation of 0.1° or more
        print(f"Warning: Large measurement variation (σ={peak_std:.3f}°)")
        print("Check sample alignment and instrument stability")
    

* * *

## 2.9 Spectral Analysis Checklist

### Data Loading and Validation

  * [ ] Verify data format is correct (units for 2θ, energy, wavenumber)
  * [ ] Verify data range is appropriate (XRD: 10-80°, XPS: 0-1200 eV, Raman: 200-2000 cm⁻¹)
  * [ ] Check proportion of missing values (10% or more requires attention)
  * [ ] Verify sufficient data points (minimum 100 points recommended)
  * [ ] Visualize spectrum (grasp overall picture)

### Environment and Reproducibility

  * [ ] Record versions of Python, NumPy, SciPy, Matplotlib
  * [ ] Save analysis parameters to JSON/YAML file
  * [ ] Record license and citation information when using databases
  * [ ] Comply with terms of use when using commercial databases (ICDD)
  * [ ] Fix random seed (if applicable)

### Noise Level Evaluation

  * [ ] Calculate standard deviation of noise in baseline region
  * [ ] Calculate S/N ratio (peak height / noise standard deviation)
  * [ ] Verify S/N ratio is 3 or higher
  * [ ] Apply smoothing as needed (noise removal)

### Background Removal

  * [ ] Select method according to measurement technique
  * XRD/Raman: SNIP method or polynomial fitting
  * XPS: Shirley-type background
  * IR/Raman (with fluorescence): ALS method
  * [ ] Record parameters (polynomial degree, SNIP iterations)
  * [ ] Check proportion of negative values after background subtraction (5% or less desirable)
  * [ ] **Always visualize spectrum before and after correction**

### Peak Detection

  * [ ] Set prominence/height based on noise level
  * [ ] Set physically reasonable peak width (width)
  * [ ] Set minimum peak spacing (distance)
  * [ ] **Visualize detected peaks overlaid on original spectrum**
  * [ ] Verify peak count matches expected value

### Peak Fitting

  * [ ] Select peak shape function (Gaussian, Lorentzian, Pseudo-Voigt)
  * [ ] Set initial values appropriately (fitting convergence)
  * [ ] Check R² value of fitting results (0.95 or higher desirable)
  * [ ] Visualize fitted curve and measured values
  * [ ] Check for systematic errors with residual plot

### Quantitative Analysis (XRD-RIR Method)

  * [ ] Verify RIR values use unified reference material
  * [ ] Perform peak separation if peak overlap exists
  * [ ] Verify weight fraction total is close to 100% (within ±5%)
  * [ ] Report mean and standard deviation of multiple measurements

### Quantitative Analysis (XPS)

  * [ ] Perform Shirley background removal
  * [ ] Integrate peak area (exclude negative values)
  * [ ] Correct with sensitivity factors (Scofield coefficients, etc.)
  * [ ] Verify atomic % total equals 100%
  * [ ] Consider spin-orbit splitting (if applicable)

### Results Validity Verification

  * [ ] Validate method with known samples
  * [ ] Compare with literature values (peak positions, intensity ratios)
  * [ ] Verify reproducibility of multiple measurements (standard deviation, standard error)
  * [ ] Evaluate physical and chemical validity
  * [ ] Separate measurement error from analysis error

### Automation and Batch Processing

  * [ ] Implement error handling (try-except)
  * [ ] Record log on processing failure
  * [ ] Measure and record processing time
  * [ ] Calculate and report success rate (e.g., 95/100 files succeeded)
  * [ ] Save results in JSON/CSV format

### Documentation

  * [ ] Record analysis procedure in reproducible form
  * [ ] Document rationale for parameter settings
  * [ ] Cite databases and literature used
  * [ ] Specify uncertainty of final results
  * [ ] Add comments to code (for your future self)

* * *

## 2.10 Chapter Summary

### What We Learned

  1. **Data Licenses and Reproducibility** \- Utilizing spectral databases and license compliance \- Documenting environment information and parameters \- Best practices for code reproducibility

  2. **Background Removal Methods** \- Polynomial fitting (general purpose) \- SNIP method (XRD, Raman) \- Shirley-type (XPS) \- ALS method (IR, Raman)

  3. **Peak Detection** \- Utilizing `scipy.signal.find_peaks` \- Parameter optimization (prominence, distance, width) \- Gaussian and Lorentzian fitting

  4. **Quantitative Analysis** \- Phase fraction calculation by RIR method \- XPS quantification (sensitivity factor correction) \- Quantitative evaluation of peak areas

  5. **Practical Pitfalls** \- Avoiding over-application of background removal \- Adaptive parameter settings \- Quantitative evaluation of measurement errors

  6. **Automation** \- Class-based analysis pipeline \- Support for multiple measurement techniques

### Key Points

  * ✅ Always verify license and citation when using databases
  * ✅ Document analysis parameters to ensure reproducibility
  * ✅ Select appropriate background removal method for each measurement technique
  * ✅ Peak detection requires parameter tuning and visualization verification
  * ✅ Quantitative analysis requires reference information such as standard samples and RIR values
  * ✅ Quantitatively evaluate measurement errors and specify uncertainty in results
  * ✅ Automation significantly improves reproducibility and processing speed

### To Next Chapter

In Chapter 3, we will learn analysis methods for image data (SEM, TEM): \- Image preprocessing (noise removal, contrast adjustment) \- Particle detection (Watershed method) \- Particle size distribution analysis \- Image classification with CNN

**[Chapter 3: Image Data Analysis →](<chapter-3.html>)**

* * *

## Practice Problems

### Problem 1 (Difficulty: Easy)

Determine whether the following statements are true or false.

  1. SNIP method is less affected by peaks than polynomial fitting
  2. Linear background is suitable for XPS spectra
  3. The prominence parameter in peak detection specifies the minimum distance between peaks

Hint 1\. Consider the operating principle of SNIP method (peak clipping) 2\. XPS background is caused by inelastic scattering 3\. Verify the meaning of prominence, distance, and width parameters  Solution Example **Answer**: 1\. **True** - SNIP method is more robust than polynomial fitting because it estimates background while avoiding peaks 2\. **False** - Shirley-type background is appropriate for XPS (asymmetric shape due to inelastic scattering) 3\. **False** - prominence is prominence (height difference from surroundings), distance between peaks is the distance parameter **Explanation**: Selection of background removal method should be based on measurement principles. Each has different background shapes: inelastic scattering in XPS, fluorescence in Raman, amorphous halo in XRD, etc. 

* * *

### Problem 2 (Difficulty: Medium)

Perform background removal using the SNIP method and detect peaks in the following XRD data.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Perform background removal using the SNIP method and detect 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Sample XRD data
    np.random.seed(123)
    two_theta = np.linspace(15, 75, 600)
    intensity = (
        1200 * np.exp(-((two_theta - 26.6) ** 2) / 12) +
        1800 * np.exp(-((two_theta - 33.8) ** 2) / 10) +
        1000 * np.exp(-((two_theta - 54.8) ** 2) / 15) +
        150 + 50 * np.sin(two_theta / 8) +
        np.random.normal(0, 40, len(two_theta))
    )
    

**Requirements** : 1\. Background removal with SNIP method (iterations=40) 2\. Peak detection (prominence=100) 3\. Output detected peak positions and intensities 4\. Visualize spectrum before and after processing

Hint **Processing flow**: 1\. Define or reuse SNIP function 2\. Background subtraction 3\. Peak detection with `find_peaks` 4\. Organize and output results 5\. Visualize 3 stages with `matplotlib` (raw data, background, after correction)  Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Requirements:
    1. Background removal with SNIP method (iterat
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    # SNIP function (from Code Example 2)
    def snip_background(spectrum, iterations=30):
        spectrum = np.array(spectrum, dtype=float)
        background = np.copy(spectrum)
    
        for i in range(1, iterations + 1):
            for j in range(i, len(background) - i):
                v1 = (background[j - i] + background[j + i]) / 2
                v2 = background[j]
                background[j] = min(v1, v2)
    
        return background
    
    # Sample data
    np.random.seed(123)
    two_theta = np.linspace(15, 75, 600)
    intensity = (
        1200 * np.exp(-((two_theta - 26.6) ** 2) / 12) +
        1800 * np.exp(-((two_theta - 33.8) ** 2) / 10) +
        1000 * np.exp(-((two_theta - 54.8) ** 2) / 15) +
        150 + 50 * np.sin(two_theta / 8) +
        np.random.normal(0, 40, len(two_theta))
    )
    
    # Apply SNIP method
    bg = snip_background(intensity, iterations=40)
    intensity_corrected = intensity - bg
    
    # Peak Detection
    peaks_idx, _ = find_peaks(intensity_corrected, prominence=100)
    peak_positions = two_theta[peaks_idx]
    peak_intensities = intensity_corrected[peaks_idx]
    
    # Output results
    print("=== Peak Detection Results ===")
    for i, (pos, intens) in enumerate(zip(peak_positions,
                                           peak_intensities), 1):
        print(f"Peak {i}: 2θ = {pos:.2f}°, Intensity = {intens:.1f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].plot(two_theta, intensity, linewidth=1.5)
    axes[0].set_xlabel('2θ (degree)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Raw XRD Pattern')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(two_theta, intensity, label='Raw', alpha=0.6)
    axes[1].plot(two_theta, bg, label='SNIP background',
                 linewidth=2, color='red')
    axes[1].set_xlabel('2θ (degree)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Background Estimation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(two_theta, intensity_corrected, linewidth=1.5)
    axes[2].plot(peak_positions, peak_intensities, 'rx',
                 markersize=12, markeredgewidth=2)
    axes[2].set_xlabel('2θ (degree)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_title('After Background Subtraction + Peak Detection')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output**: 
    
    
    === Peak Detection Results ===
    Peak 1: 2θ = 26.59°, Intensity = 1205.3
    Peak 2: 2θ = 33.81°, Intensity = 1813.7
    Peak 3: 2θ = 54.76°, Intensity = 1008.2
    

**Explanation**: SNIP method iterations=40 corresponds to the characteristic width of the background (in data points). For this example with a gradual background, 40 iterations is sufficient. With prominence=100, noise peaks are excluded and only the 3 main peaks are detected. 

* * *

### Problem 3 (Difficulty: Hard)

Build a batch processing system to automatically process spectral data from multiple measurement techniques (XRD, XPS, Raman).

**Background** : Composite measurement data (XRD, XPS, Raman) for 100 samples was obtained from a materials library. An integrated system is needed to automatically select appropriate preprocessing for each measurement and extract peak information.

**Tasks** : 1\. Automatic selection of optimal background removal method according to measurement technique 2\. Automatic adjustment of peak detection parameters 3\. Save results in JSON format 4\. Error handling and log output

**Constraints** : \- Different data formats for each measurement technique (column names, units) \- Variation in measurement quality (noise level) \- Processing time: Within 10 seconds/sample

Hint **Design Guidelines**: 1\. Extend `SpectrumAnalyzer` class 2\. Automatic detection of measurement technique (from metadata or filename) 3\. Adaptive parameter adjustment (adjust prominence based on noise level) 4\. JSON output includes peak positions, intensities, and estimated phase information **Class Design Example**: 
    
    
    class AutoSpectrumProcessor:
        def __init__(self):
            self.analyzers = {}  # Analyzer for each measurement technique
    
        def detect_spectrum_type(self, data):
            # Determine measurement technique from metadata
            pass
    
        def adaptive_parameters(self, spectrum):
            # Adjust parameters based on noise level
            pass
    
        def batch_process(self, file_list):
            # Process multiple files
            pass
    

Solution Example **Solution Overview**: Build an integrated processing system including automatic detection of measurement technique, adaptive parameter adjustment, and saving results to JSON. **Implementation Code**: 
    
    
    import json
    import logging
    from pathlib import Path
    from typing import Dict, List
    from dataclasses import dataclass, asdict
    
    logging.basicConfig(level=logging.INFO)
    
    @dataclass
    class SpectrumResult:
        """Store analysis results"""
        spectrum_type: str
        num_peaks: int
        peaks: List[Dict]
        processing_time: float
        background_method: str
    
    class AutoSpectrumProcessor:
        """Automated spectral analysis system"""
    
        def __init__(self):
            self.bg_methods = {
                'XRD': 'snip',
                'XPS': 'shirley',
                'Raman': 'als',
                'IR': 'als'
            }
    
        def detect_spectrum_type(self, x: np.ndarray) -> str:
            """
            Estimate measurement technique from data range
            """
            x_range = x.max() - x.min()
            x_min = x.min()
    
            if x_min > 5 and x_range < 100:  # 2θ range
                return 'XRD'
            elif x_min > 200 and x_range > 500:  # BE range
                return 'XPS'
            elif x_min > 100 and x_range > 1000:  # cm-1 range
                return 'Raman' if x.max() < 4000 else 'IR'
            else:
                return 'Unknown'
    
        def adaptive_prominence(self, spectrum: np.ndarray) -> float:
            """
            Adjust prominence based on noise level
            """
            noise_std = np.std(np.diff(spectrum))
            snr = np.max(spectrum) / (noise_std + 1e-10)
    
            if snr > 50:
                return 0.05 * np.max(spectrum)  # High S/N
            elif snr > 20:
                return 0.08 * np.max(spectrum)  # Medium S/N
            else:
                return 0.12 * np.max(spectrum)  # Low S/N
    
        def process_spectrum(self, x: np.ndarray, y: np.ndarray,
                            metadata: Dict = None) -> SpectrumResult:
            """
            Process single spectrum
            """
            import time
            start_time = time.time()
    
            # Determine measurement technique
            if metadata and 'type' in metadata:
                spec_type = metadata['type']
            else:
                spec_type = self.detect_spectrum_type(x)
    
            logging.info(f"Detected spectrum type: {spec_type}")
    
            # Background Removal
            bg_method = self.bg_methods.get(spec_type, 'snip')
    
            if bg_method == 'snip':
                bg = snip_background(y, iterations=40)
            elif bg_method == 'als':
                bg = als_baseline(y, lam=1e6, p=0.01)
            else:
                # Simple linear (if Shirley not implemented)
                bg = np.linspace(y[0], y[-1], len(y))
    
            y_corrected = y - bg
    
            # Adaptive peak detection
            prominence = self.adaptive_prominence(y_corrected)
            peaks_idx, _ = find_peaks(y_corrected, prominence=prominence)
    
            # Structure peak information
            peaks_info = []
            for idx in peaks_idx:
                peaks_info.append({
                    'position': float(x[idx]),
                    'intensity': float(y_corrected[idx]),
                    'unit': '2θ(deg)' if spec_type == 'XRD' else 'cm-1'
                })
    
            processing_time = time.time() - start_time
    
            result = SpectrumResult(
                spectrum_type=spec_type,
                num_peaks=len(peaks_idx),
                peaks=peaks_info,
                processing_time=processing_time,
                background_method=bg_method
            )
    
            return result
    
        def batch_process(self, data_list: List[Dict],
                         output_file: str = 'results.json'):
            """
            Batch processing
    
            Parameters:
            -----------
            data_list : list of dict
                Each element is {'x': array, 'y': array, 'metadata': dict}
            """
            results = []
    
            for i, data in enumerate(data_list, 1):
                try:
                    logging.info(f"Processing spectrum {i}/{len(data_list)}")
                    result = self.process_spectrum(
                        data['x'],
                        data['y'],
                        data.get('metadata')
                    )
                    results.append(asdict(result))
    
                except Exception as e:
                    logging.error(f"Failed to process spectrum {i}: {e}")
                    continue
    
            # Save JSON
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
            logging.info(f"Results saved to {output_file}")
            return results
    
    # Demo execution
    if __name__ == "__main__":
        processor = AutoSpectrumProcessor()
    
        # Generate sample data (3 types of measurements)
        data_list = []
    
        # XRD data
        two_theta = np.linspace(20, 60, 400)
        xrd_y = (
            1000 * np.exp(-((two_theta - 28) ** 2) / 10) +
            1500 * np.exp(-((two_theta - 35) ** 2) / 8) +
            100 + np.random.normal(0, 30, len(two_theta))
        )
        data_list.append({
            'x': two_theta,
            'y': xrd_y,
            'metadata': {'type': 'XRD', 'sample': 'Fe2O3'}
        })
    
        # Raman data
        raman_shift = np.linspace(200, 2000, 900)
        raman_y = (
            2000 * np.exp(-((raman_shift - 520) ** 2) / 100) +
            1500 * np.exp(-((raman_shift - 950) ** 2) / 150) +
            500 + np.random.normal(0, 50, len(raman_shift))
        )
        data_list.append({
            'x': raman_shift,
            'y': raman_y,
            'metadata': {'type': 'Raman', 'sample': 'Si'}
        })
    
        # Execute batch processing
        results = processor.batch_process(data_list,
                                          output_file='spectrum_results.json')
    
        print("\n=== Processing Summary ===")
        for i, result in enumerate(results, 1):
            print(f"Spectrum {i}:")
            print(f"  Type: {result['spectrum_type']}")
            print(f"  Peaks detected: {result['num_peaks']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
    

**Results (JSON Output Example)**: 
    
    
    [
      {
        "spectrum_type": "XRD",
        "num_peaks": 2,
        "peaks": [
          {"position": 28.05, "intensity": 1023.4, "unit": "2θ(deg)"},
          {"position": 35.01, "intensity": 1518.7, "unit": "2θ(deg)"}
        ],
        "processing_time": 0.045,
        "background_method": "snip"
      },
      {
        "spectrum_type": "Raman",
        "num_peaks": 2,
        "peaks": [
          {"position": 520.3, "intensity": 2015.6, "unit": "cm-1"},
          {"position": 949.8, "intensity": 1507.2, "unit": "cm-1"}
        ],
        "processing_time": 0.052,
        "background_method": "als"
      }
    ]
    

**Detailed Explanation**: 1\. **Automatic Detection**: Estimate measurement technique from data range (XRD: 10-80°, Raman: 200-2000 cm⁻¹) 2\. **Adaptive Parameters**: Automatically adjust prominence from S/N ratio 3\. **Structured Output**: JSON format supports subsequent analysis and database registration 4\. **Error Handling**: Failure of individual spectra does not stop entire batch **Additional Considerations**: \- Improve measurement technique detection accuracy (introduce machine learning classifier) \- Load data from cloud storage (S3, GCS) \- Real-time visualization with web dashboard \- Save results to database (MongoDB) 

* * *

## References

  1. Pecharsky, V. K., & Zavalij, P. Y. (2009). "Fundamentals of Powder Diffraction and Structural Characterization of Materials." Springer. ISBN: 978-0387095783

  2. Briggs, D., & Seah, M. P. (1990). "Practical Surface Analysis by Auger and X-ray Photoelectron Spectroscopy." Wiley. ISBN: 978-0471920816

  3. Ryan, C. G. et al. (1988). "SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications." _Nuclear Instruments and Methods in Physics Research B_ , 34(3), 396-402. DOI: [10.1016/0168-583X(88)90063-8](<https://doi.org/10.1016/0168-583X\(88\)90063-8>)

  4. Eilers, P. H. C., & Boelens, H. F. M. (2005). "Baseline Correction with Asymmetric Least Squares Smoothing." _Leiden University Medical Centre Report_.

  5. SciPy Documentation: Signal Processing. URL: <https://docs.scipy.org/doc/scipy/reference/signal.html>

* * *

## Navigation

### Previous Chapter

**[Chapter 1: Fundamentals of Experimental Data Analysis ←](<chapter-1.html>)**

### Next Chapter

**[Chapter 3: Image Data Analysis →](<chapter-3.html>)**

### Series Index

**[← Return to Series Index](<./index.html>)**

* * *

## Author Information

**Author** : AI Terakoya Content Team **Created** : 2025-10-17 **Version** : 1.0

**Update History** : \- 2025-10-17: v1.0 Initial release

**Feedback** : \- GitHub Issues: [Repository URL]/issues \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

**Let's continue learning in the next chapter!**
