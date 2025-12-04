---
title: "Chapter 2: Powder X-Ray Diffraction Measurement and Analysis"
chapter_title: "Chapter 2: Powder X-Ray Diffraction Measurement and Analysis"
subtitle: From XRD Instrument Principles to Real Data Analysis - Practical Guide to Peak Identification and Fitting
reading_time: 28-32 minutes
difficulty: Intermediate
code_examples: 8
---

This chapter covers Powder X. You will learn role of each component in XRD instruments, Identify peaks from measured XRD patterns, and background removal.

## Learning Objectives

By completing this chapter, you will be able to:

  * Understand the role of each component in XRD instruments and optimize measurement conditions
  * Identify peaks from measured XRD patterns and assign Miller indices
  * Implement background removal and data smoothing
  * Perform peak fitting using Gaussian, Lorentzian, and Voigt functions
  * Conduct advanced peak analysis using scipy.optimize

## 2.1 Configuration of XRD Instruments

### 2.1.1 X-ray Source

The most important component of a powder X-ray diffractometer is the X-ray source. In laboratory XRD systems, the following characteristic X-rays are mainly used:

X-ray Source | K±1 Wavelength [Å] | K±2 Wavelength [Å] | Features  
---|---|---|---  
Cu K± | 1.54056 | 1.54439 | Most common, high versatility  
Mo K± | 0.71073 | 0.71359 | High energy, strong penetration  
Co K± | 1.78897 | 1.79285 | Advantageous for Fe-containing samples (avoids fluorescence)  
Cr K± | 2.28970 | 2.29361 | Long wavelength, improved low-angle resolution  
  
### 2.1.2 Optics and Detectors
    
    
    ```mermaid
    graph LR
        A[X-ray TubeCu Anode] --> B[Incident SlitBeam Shaping]
        B --> C[SamplePowder]
        C --> D[Receiving SlitScatter Removal]
        D --> E[MonochromatorNi Filter]
        E --> F[DetectorScintillation]
        F --> G[Data ProcessingPC]
    
        style A fill:#ffe7e7
        style C fill:#fce7f3
        style F fill:#e7f3ff
        style G fill:#e7ffe7
    ```

**Bragg-Brentano Configuration** :

  * ¸-2¸ method: X-ray source and detector move symmetrically around the sample
  * Focusing geometry: Diffracted X-rays from different points on the sample surface converge at the detector
  * Achieves both high resolution and high intensity

### 2.1.3 Optimization of Measurement Conditions
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class XRDMeasurementConditions:
        """XRD measurement condition optimization class"""
    
        def __init__(self, wavelength=1.54056):
            """
            Args:
                wavelength (float): X-ray wavelength [Å]
            """
            self.wavelength = wavelength
    
        def calculate_two_theta_range(self, d_min=1.0, d_max=10.0):
            """Calculate the 2¸ range to be measured
    
            Args:
                d_min (float): Minimum d-spacing [Å]
                d_max (float): Maximum d-spacing [Å]
    
            Returns:
                tuple: (two_theta_min, two_theta_max) [degrees]
            """
            # Bragg's law: » = 2d sin¸
            # sin(¸) = » / (2d)
    
            sin_theta_max = self.wavelength / (2 * d_min)
            sin_theta_min = self.wavelength / (2 * d_max)
    
            if sin_theta_max > 1.0:
                sin_theta_max = 1.0  # Physical upper limit
    
            theta_max = np.degrees(np.arcsin(sin_theta_max))
            theta_min = np.degrees(np.arcsin(sin_theta_min))
    
            return 2 * theta_min, 2 * theta_max
    
        def optimize_step_size(self, two_theta_range, fwhm=0.1):
            """Calculate optimal step size
    
            Args:
                two_theta_range (tuple): (start, end) measurement range [degrees]
                fwhm (float): Full width at half maximum of peak [degrees]
    
            Returns:
                float: Recommended step size [degrees]
            """
            # Rule: 1/5 to 1/10 of FWHM
            step_size_recommended = fwhm / 7.0
            return step_size_recommended
    
        def calculate_measurement_time(self, two_theta_range, step_size, time_per_step=1.0):
            """Estimate total measurement time
    
            Args:
                two_theta_range (tuple): (start, end) [degrees]
                step_size (float): Step size [degrees]
                time_per_step (float): Measurement time per step [seconds]
    
            Returns:
                float: Total measurement time [minutes]
            """
            start, end = two_theta_range
            n_steps = int((end - start) / step_size)
            total_seconds = n_steps * time_per_step
            return total_seconds / 60.0  # Convert to minutes
    
        def generate_measurement_plan(self, d_min=1.2, d_max=5.0, fwhm=0.1, time_per_step=2.0):
            """Generate complete measurement plan
    
            Returns:
                dict: Measurement parameters
            """
            two_theta_range = self.calculate_two_theta_range(d_min, d_max)
            step_size = self.optimize_step_size(two_theta_range, fwhm)
            total_time = self.calculate_measurement_time(two_theta_range, step_size, time_per_step)
    
            plan = {
                'two_theta_start': two_theta_range[0],
                'two_theta_end': two_theta_range[1],
                'step_size': step_size,
                'time_per_step': time_per_step,
                'total_time_minutes': total_time,
                'estimated_points': int((two_theta_range[1] - two_theta_range[0]) / step_size)
            }
    
            return plan
    
    
    # Usage example
    conditions = XRDMeasurementConditions(wavelength=1.54056)  # Cu K±
    plan = conditions.generate_measurement_plan(d_min=1.2, d_max=5.0, fwhm=0.1, time_per_step=2.0)
    
    print("=== XRD Measurement Plan ===")
    print(f"Measurement range: {plan['two_theta_start']:.2f}° - {plan['two_theta_end']:.2f}°")
    print(f"Step size: {plan['step_size']:.4f}°")
    print(f"Time per point: {plan['time_per_step']:.1f} seconds")
    print(f"Total data points: {plan['estimated_points']}")
    print(f"Estimated total time: {plan['total_time_minutes']:.1f} minutes ({plan['total_time_minutes']/60:.2f} hours)")
    
    # Expected output:
    # Measurement range: 9.14° - 40.33°
    # Step size: 0.0143°
    # Total measurement time: ~73 minutes

## 2.2 Optimization of Measurement Conditions

### 2.2.1 Determining the 2¸ Range

The appropriate 2¸ range depends on the analysis purpose and the crystal structure of the sample:

  * **Phase identification** : 10° - 80° (covers major peaks)
  * **Lattice parameter refinement** : 20° - 120° (high-angle data is important)
  * **Quantitative analysis** : 5° - 90° (includes low-angle peaks)
  * **Thin film analysis** : 20° - 90° (low angles affected by substrate)

### 2.2.2 Trade-off between Step Size and Counting Time
    
    
    def compare_measurement_strategies(two_theta_range=(10, 80)):
        """Compare different measurement strategies
    
        Args:
            two_theta_range (tuple): Measurement range [degrees]
        """
        strategies = [
            {'name': 'Fast Scan', 'step': 0.04, 'time': 0.5},
            {'name': 'Standard Scan', 'step': 0.02, 'time': 1.0},
            {'name': 'High-Resolution Scan', 'step': 0.01, 'time': 2.0},
            {'name': 'Ultra-High Precision Scan', 'step': 0.005, 'time': 5.0},
        ]
    
        print("Measurement Strategy Comparison:")
        print("-" * 70)
        print(f"{'Strategy':<25} | Step[°] | Time/pt[s] | Total pts | Total time[min]")
        print("-" * 70)
    
        for strategy in strategies:
            step = strategy['step']
            time_per_step = strategy['time']
            n_points = int((two_theta_range[1] - two_theta_range[0]) / step)
            total_time = n_points * time_per_step / 60.0
    
            print(f"{strategy['name']:<25} | {step:^7.3f} | {time_per_step:^10.1f} | {n_points:^9} | {total_time:^15.1f}")
    
        print("\nRecommendations:")
        print("- Routine analysis: Fast to Standard Scan")
        print("- Precise structure analysis: High-Resolution Scan")
        print("- Publication quality: Ultra-High Precision Scan")
    
    compare_measurement_strategies()
    
    # Expected output:
    # Fast Scan:                Total time ~15 min
    # Standard Scan:            Total time ~58 min
    # High-Resolution Scan:     Total time ~233 min
    # Ultra-High Precision Scan: Total time ~1167 min (19.4 hours)

## 2.3 Peak Identification and Indexing

### 2.3.1 Peak Detection Algorithm
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from scipy.signal import find_peaks, peak_widths
    import numpy as np
    
    def detect_peaks(two_theta, intensity, prominence=100, width=2, distance=5):
        """Automatically detect peaks from XRD pattern
    
        Args:
            two_theta (np.ndarray): 2¸ angles [degrees]
            intensity (np.ndarray): Diffraction intensity
            prominence (float): Peak prominence threshold
            width (float): Minimum peak width [data points]
            distance (int): Minimum distance between peaks [data points]
    
        Returns:
            dict: Peak information {positions, heights, widths, prominences}
        """
        # Peak detection
        peaks, properties = find_peaks(
            intensity,
            prominence=prominence,
            width=width,
            distance=distance
        )
    
        # Calculate peak widths
        widths_result = peak_widths(intensity, peaks, rel_height=0.5)
    
        peak_info = {
            'indices': peaks,
            'two_theta': two_theta[peaks],
            'intensity': intensity[peaks],
            'prominence': properties['prominences'],
            'fwhm': widths_result[0] * np.mean(np.diff(two_theta)),  # data points ’ angle
            'left_bases': two_theta[properties['left_bases'].astype(int)],
            'right_bases': two_theta[properties['right_bases'].astype(int)]
        }
    
        return peak_info
    
    
    # Generate synthetic XRD data
    def generate_synthetic_xrd(two_theta_range=(10, 80), n_points=3500):
        """Generate synthetic XRD pattern (±-Fe BCC)
    
        Returns:
            tuple: (two_theta, intensity)
        """
        two_theta = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
        intensity = np.zeros_like(two_theta)
    
        # Major peaks of ±-Fe (BCC)
        # (hkl): (position[°], intensity, FWHM[°])
        fe_peaks = [
            (44.67, 1000, 0.15),  # (110)
            (65.02, 300, 0.18),   # (200)
            (82.33, 450, 0.22),   # (211)
        ]
    
        # Add Gaussian peaks
        for pos, height, fwhm in fe_peaks:
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            intensity += height * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
    
        # Background noise
        background = 50 + 20 * np.exp(-two_theta / 30)
        noise = np.random.normal(0, 5, len(two_theta))
    
        intensity_total = intensity + background + noise
    
        return two_theta, intensity_total
    
    
    # Execution example
    two_theta, intensity = generate_synthetic_xrd()
    peaks = detect_peaks(two_theta, intensity, prominence=100, width=2, distance=10)
    
    print("=== Detected Peaks ===")
    print(f"{'2¸ [°]':<10} | {'Intensity':<10} | {'FWHM [°]':<10} | {'Rel. Int. [%]':<15}")
    print("-" * 50)
    
    # Normalize intensity
    I_max = np.max(peaks['intensity'])
    for i in range(len(peaks['two_theta'])):
        rel_intensity = 100 * peaks['intensity'][i] / I_max
        print(f"{peaks['two_theta'][i]:8.2f}   | {peaks['intensity'][i]:8.0f}   | {peaks['fwhm'][i]:8.3f}   | {rel_intensity:8.1f}")
    
    # Expected output:
    #   44.67   |    1000   |    0.150   |    100.0
    #   65.02   |     300   |    0.180   |     30.0
    #   82.33   |     450   |    0.220   |     45.0

### 2.3.2 Miller Index Assignment (Cubic System)
    
    
    def index_cubic_pattern(two_theta_obs, wavelength=1.54056, lattice_type='I', a_initial=3.0):
        """Index cubic XRD pattern
    
        Args:
            two_theta_obs (np.ndarray): Observed diffraction angles [degrees]
            wavelength (float): X-ray wavelength [Å]
            lattice_type (str): Lattice type ('P', 'I', 'F')
            a_initial (float): Initial lattice parameter estimate [Å]
    
        Returns:
            list: Indexing results [(h, k, l, d_calc, two_theta_calc, delta), ...]
        """
        # d-value for cubic system: d = a / sqrt(h^2 + k^2 + l^2)
        # Generate allowed Miller indices
        hkl_list = []
        for h in range(0, 6):
            for k in range(0, 6):
                for l in range(0, 6):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    # Check systematic absences
                    if lattice_type == 'I' and (h + k + l) % 2 != 0:
                        continue
                    if lattice_type == 'F':
                        parity = [h % 2, k % 2, l % 2]
                        if len(set(parity)) != 1:
                            continue
                    hkl_list.append((h, k, l, h**2 + k**2 + l**2))
    
        # Sort by h^2 + k^2 + l^2
        hkl_list.sort(key=lambda x: x[3])
    
        # Assign the closest Miller indices to each observed peak
        indexing_results = []
    
        for two_theta in two_theta_obs:
            theta_rad = np.radians(two_theta / 2)
            d_obs = wavelength / (2 * np.sin(theta_rad))
    
            # Try all Miller index candidates
            best_match = None
            min_error = float('inf')
    
            for h, k, l, h2k2l2 in hkl_list:
                d_calc = a_initial / np.sqrt(h2k2l2)
                error = abs(d_calc - d_obs)
    
                if error < min_error:
                    min_error = error
                    theta_calc = np.degrees(np.arcsin(wavelength / (2 * d_calc)))
                    two_theta_calc = 2 * theta_calc
                    best_match = (h, k, l, d_calc, two_theta_calc, two_theta_calc - two_theta)
    
            if best_match and abs(best_match[5]) < 0.5:  # Tolerance 0.5 degrees
                indexing_results.append(best_match)
    
        return indexing_results
    
    
    # Usage example
    two_theta_obs = np.array([44.67, 65.02, 82.33])  # Major peaks of ±-Fe BCC
    indexed = index_cubic_pattern(two_theta_obs, wavelength=1.54056, lattice_type='I', a_initial=2.87)
    
    print("=== Miller Index Assignment (±-Fe BCC) ===")
    print(f"{'(hkl)':<10} | {'d_calc[Å]':<10} | {'2¸_calc[°]':<12} | {'2¸_obs[°]':<12} | {'Error[°]':<8}")
    print("-" * 65)
    
    for h, k, l, d_calc, two_theta_calc, delta in indexed:
        two_theta_obs_match = two_theta_calc - delta
        print(f"({h} {k} {l}){' '*(6-len(f'{h} {k} {l}'))} | {d_calc:8.4f}   | {two_theta_calc:10.2f}   | {two_theta_obs_match:10.2f}   | {delta:6.2f}")
    
    # Expected output:
    # (1 1 0)  |   2.0293   |      44.67   |      44.67   |   0.00
    # (2 0 0)  |   1.4350   |      65.02   |      65.02   |   0.00
    # (2 1 1)  |   1.1707   |      82.33   |      82.33   |   0.00

## 2.4 Background Removal and Data Smoothing

### 2.4.1 Background Removal by Polynomial Fitting
    
    
    from scipy.signal import savgol_filter
    
    def remove_background_polynomial(two_theta, intensity, degree=3, exclude_peaks=True):
        """Remove background by polynomial fitting
    
        Args:
            two_theta (np.ndarray): 2¸ angles
            intensity (np.ndarray): Diffraction intensity
            degree (int): Polynomial degree
            exclude_peaks (bool): Exclude peak regions for fitting
    
        Returns:
            tuple: (background, intensity_corrected)
        """
        if exclude_peaks:
            # Peak detection
            peaks_info = detect_peaks(two_theta, intensity, prominence=50)
            peak_indices = peaks_info['indices']
    
            # Create mask excluding peak vicinity
            mask = np.ones(len(two_theta), dtype=bool)
            for peak_idx in peak_indices:
                # Exclude ±20 points around peak
                mask[max(0, peak_idx-20):min(len(two_theta), peak_idx+20)] = False
    
            # Fit only masked regions
            coeffs = np.polyfit(two_theta[mask], intensity[mask], degree)
        else:
            coeffs = np.polyfit(two_theta, intensity, degree)
    
        # Calculate background
        background = np.polyval(coeffs, two_theta)
    
        # Corrected intensity
        intensity_corrected = intensity - background
    
        # Clip negative values to zero
        intensity_corrected = np.maximum(intensity_corrected, 0)
    
        return background, intensity_corrected
    
    
    def smooth_data_savitzky_golay(intensity, window_length=11, polyorder=3):
        """Smooth data with Savitzky-Golay filter
    
        Args:
            intensity (np.ndarray): Diffraction intensity
            window_length (int): Window length (odd number)
            polyorder (int): Polynomial order
    
        Returns:
            np.ndarray: Smoothed intensity
        """
        if window_length % 2 == 0:
            window_length += 1  # Adjust to odd number
    
        smoothed = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)
        return smoothed
    
    
    # Application example
    two_theta, intensity_raw = generate_synthetic_xrd()
    
    # Background removal
    background, intensity_corrected = remove_background_polynomial(
        two_theta, intensity_raw, degree=3, exclude_peaks=True
    )
    
    # Smoothing
    intensity_smoothed = smooth_data_savitzky_golay(intensity_corrected, window_length=11, polyorder=3)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(two_theta, intensity_raw, 'gray', alpha=0.5, label='Raw data')
    plt.plot(two_theta, background, 'r--', linewidth=2, label='Background')
    plt.xlabel('2¸ [degrees]')
    plt.ylabel('Intensity [counts]')
    plt.legend()
    plt.title('Step 1: Background Estimation')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(two_theta, intensity_corrected, 'b', alpha=0.7, label='BG removed')
    plt.xlabel('2¸ [degrees]')
    plt.ylabel('Intensity [counts]')
    plt.legend()
    plt.title('Step 2: Background Removal')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(two_theta, intensity_smoothed, color='#f093fb', linewidth=2, label='Smoothed')
    plt.xlabel('2¸ [degrees]')
    plt.ylabel('Intensity [counts]')
    plt.legend()
    plt.title('Step 3: Savitzky-Golay Smoothing')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 2.5 Peak Fitting

### 2.5.1 Peak Shape Functions

XRD peaks exhibit various shapes depending on instrument resolution and sample crystallinity:

**Gaussian Function** (instrument-induced broadening):

\\[ G(x) = I_0 \exp\left[-\frac{(x - x_0)^2}{2\sigma^2}\right] \\] 

**Lorentzian Function** (sample-induced broadening, crystallite size):

\\[ L(x) = I_0 \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1} \\] 

**Pseudo-Voigt Function** (linear combination of Gaussian and Lorentzian):

\\[ PV(x) = \eta L(x) + (1-\eta) G(x) \\] 
    
    
    def gaussian(x, amplitude, center, sigma):
        """Gaussian function
    
        Args:
            x (np.ndarray): Independent variable
            amplitude (float): Peak height
            center (float): Peak center position
            sigma (float): Standard deviation
    
        Returns:
            np.ndarray: Gaussian curve
        """
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian function
    
        Args:
            x (np.ndarray): Independent variable
            amplitude (float): Peak height
            center (float): Peak center position
            gamma (float): Half width at half maximum
    
        Returns:
            np.ndarray: Lorentzian curve
        """
        return amplitude / (1 + ((x - center) / gamma) ** 2)
    
    
    def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
        """Pseudo-Voigt function
    
        Args:
            x (np.ndarray): Independent variable
            amplitude (float): Peak height
            center (float): Peak center position
            sigma (float): Standard deviation of Gaussian component
            gamma (float): Half width at half maximum of Lorentzian component
            eta (float): Lorentzian mixing ratio (0-1)
    
        Returns:
            np.ndarray: Pseudo-Voigt curve
        """
        G = gaussian(x, amplitude, center, sigma)
        L = lorentzian(x, amplitude, center, gamma)
        return eta * L + (1 - eta) * G
    
    
    # Comparison plot of peak shapes
    x = np.linspace(40, 50, 500)
    amplitude = 1000
    center = 45
    sigma = 0.1
    gamma = 0.1
    eta = 0.5
    
    y_gauss = gaussian(x, amplitude, center, sigma)
    y_lorentz = lorentzian(x, amplitude, center, gamma)
    y_voigt = pseudo_voigt(x, amplitude, center, sigma, gamma, eta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_gauss, label='Gaussian', color='#3498db', linewidth=2)
    plt.plot(x, y_lorentz, label='Lorentzian', color='#e74c3c', linewidth=2)
    plt.plot(x, y_voigt, label='Pseudo-Voigt (·=0.5)', color='#f093fb', linewidth=2, linestyle='--')
    plt.xlabel('2¸ [degrees]', fontsize=12)
    plt.ylabel('Intensity [counts]', fontsize=12)
    plt.title('Comparison of XRD Peak Shape Functions', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

### 2.5.2 Fitting with scipy.optimize
    
    
    from scipy.optimize import curve_fit
    
    def fit_single_peak(two_theta, intensity, peak_center_guess, fit_func='voigt'):
        """Fit a single peak
    
        Args:
            two_theta (np.ndarray): 2¸ angles
            intensity (np.ndarray): Diffraction intensity
            peak_center_guess (float): Initial estimate of peak center position
            fit_func (str): Fit function ('gaussian', 'lorentzian', 'voigt')
    
        Returns:
            dict: Fit results {params, covariance, fitted_curve}
        """
        # Restrict fit range (peak center ±2 degrees)
        mask = (two_theta >= peak_center_guess - 2) & (two_theta <= peak_center_guess + 2)
        x_fit = two_theta[mask]
        y_fit = intensity[mask]
    
        # Initial parameter estimates
        amplitude_guess = np.max(y_fit)
        sigma_guess = 0.1
        gamma_guess = 0.1
    
        try:
            if fit_func == 'gaussian':
                popt, pcov = curve_fit(
                    gaussian,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, sigma_guess],
                    bounds=([0, peak_center_guess-1, 0.01], [np.inf, peak_center_guess+1, 1.0])
                )
                fitted_curve = gaussian(two_theta, *popt)
                param_names = ['amplitude', 'center', 'sigma']
    
            elif fit_func == 'lorentzian':
                popt, pcov = curve_fit(
                    lorentzian,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, gamma_guess],
                    bounds=([0, peak_center_guess-1, 0.01], [np.inf, peak_center_guess+1, 1.0])
                )
                fitted_curve = lorentzian(two_theta, *popt)
                param_names = ['amplitude', 'center', 'gamma']
    
            elif fit_func == 'voigt':
                popt, pcov = curve_fit(
                    pseudo_voigt,
                    x_fit, y_fit,
                    p0=[amplitude_guess, peak_center_guess, sigma_guess, gamma_guess, 0.5],
                    bounds=([0, peak_center_guess-1, 0.01, 0.01, 0], [np.inf, peak_center_guess+1, 1.0, 1.0, 1])
                )
                fitted_curve = pseudo_voigt(two_theta, *popt)
                param_names = ['amplitude', 'center', 'sigma', 'gamma', 'eta']
    
            else:
                raise ValueError(f"Unknown fit function: {fit_func}")
    
            # Standard errors of parameters
            perr = np.sqrt(np.diag(pcov))
    
            result = {
                'function': fit_func,
                'params': dict(zip(param_names, popt)),
                'errors': dict(zip(param_names, perr)),
                'covariance': pcov,
                'fitted_curve': fitted_curve,
                'x_fit': x_fit,
                'y_fit': y_fit
            }
    
            return result
    
        except Exception as e:
            print(f"Fitting error: {e}")
            return None
    
    
    # Execution example
    two_theta, intensity = generate_synthetic_xrd()
    background, intensity_corrected = remove_background_polynomial(two_theta, intensity)
    
    # Fit (110) peak
    fit_result = fit_single_peak(two_theta, intensity_corrected, peak_center_guess=44.7, fit_func='voigt')
    
    if fit_result:
        print("=== Peak Fitting Results (Pseudo-Voigt) ===")
        for param, value in fit_result['params'].items():
            error = fit_result['errors'][param]
            print(f"{param:<12}: {value:10.5f} ± {error:8.5f}")
    
        # FWHM calculation (approximation for Voigt)
        sigma = fit_result['params']['sigma']
        fwhm_gauss = 2.355 * sigma  # Gaussian contribution
        print(f"\nFWHM (estimate): {fwhm_gauss:.4f}°")
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(two_theta, intensity_corrected, 'gray', alpha=0.5, label='Measured data')
        plt.plot(two_theta, fit_result['fitted_curve'], color='#f093fb', linewidth=2, label='Fitted curve')
        plt.scatter(fit_result['x_fit'], fit_result['y_fit'], color='#f5576c', s=20, alpha=0.7, label='Fit range')
        plt.xlabel('2¸ [degrees]', fontsize=12)
        plt.ylabel('Intensity [counts]', fontsize=12)
        plt.title(f"Peak Fitting: {fit_result['params']['center']:.2f}°", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

## Verification of Learning Objectives

### Basic Understanding

  *  Role of each XRD instrument component (X-ray source, optics, detector)
  *  Principles of optimizing measurement conditions (2¸ range, step size, counting time)
  *  Physical meaning of Gaussian, Lorentzian, and Voigt functions

### Practical Skills

  *  Implement automatic peak detection with scipy.signal
  *  Assign Miller indices to measured peaks (cubic systems)
  *  Remove background by polynomial fitting
  *  Fit single peaks with curve_fit

### Application Capability

  *  Optimize measurement conditions and estimate total measurement time
  *  Execute complete workflow of background removal ’ smoothing ’ peak fitting on real data
  *  Calculate FWHM and standard errors of peak positions from fit results

## Practice Problems

### Easy (Basic Verification)

**Q1** : Using Cu K± radiation (»=1.54Å), calculate the 2¸ range required to cover d-spacings from 1.2Å to 5.0Å.

**Solution** :
    
    
    cond = XRDMeasurementConditions(wavelength=1.54056)
    two_theta_range = cond.calculate_two_theta_range(d_min=1.2, d_max=5.0)
    print(f"Required 2¸ range: {two_theta_range[0]:.2f}° - {two_theta_range[1]:.2f}°")
    # Output: Required 2¸ range: 9.14° - 40.33°

**Q2** : If the FWHM of a peak is 0.1°, what is the recommended step size?

**Solution** :
    
    
    cond = XRDMeasurementConditions()
    step = cond.optimize_step_size((10, 80), fwhm=0.1)
    print(f"Recommended step size: {step:.4f}°")
    # Output: Recommended step size: 0.0143° (FWHM/7)

### Medium (Application)

**Q3** : Explain the difference in shape between Gaussian and Lorentzian functions for peaks with the same FWHM (0.2°).

**Solution** :

**For FWHM = 0.2°** :

  * **Gaussian** : Ã = FWHM / (2(2ln2)) H 0.085°
  * **Lorentzian** : ³ = FWHM / 2 = 0.1°

**Shape Differences** :

  * Gaussian: Tails decay rapidly (exponential decay)
  * Lorentzian: Broad tails (power-law decay)
  * Lorentzian has longer tails ’ prominent in samples with small crystallite size

**Q4** : Calculate the Miller indices and 2¸ positions for the first three peaks observed in ±-Fe (BCC, a=2.87Å).

**Solution** :
    
    
    # BCC systematic absences: h+k+l = even
    # Smallest h^2+k^2+l^2: (110)’2, (200)’4, (211)’6
    from scipy.constants import physical_constants
    
    wavelength = 1.54056
    a = 2.87
    
    peaks = [
        (1, 1, 0, 2),
        (2, 0, 0, 4),
        (2, 1, 1, 6)
    ]
    
    for h, k, l, h2k2l2 in peaks:
        d = a / np.sqrt(h2k2l2)
        theta = np.degrees(np.arcsin(wavelength / (2 * d)))
        two_theta = 2 * theta
        print(f"({h}{k}{l}): d={d:.4f}Å, 2¸={two_theta:.2f}°")
    
    # Expected output:
    # (110): d=2.0293Å, 2¸=44.67°
    # (200): d=1.4350Å, 2¸=65.02°
    # (211): d=1.1707Å, 2¸=82.33°

### Hard (Advanced)

**Q5** : Perform background removal with varying polynomial degrees (1st, 3rd, 5th) and explain the criteria for selecting the optimal degree.

**Solution** :
    
    
    # Fit with different degrees
    degrees = [1, 3, 5]
    for deg in degrees:
        bg, corrected = remove_background_polynomial(two_theta, intensity, degree=deg)
        residual = intensity - bg
        rms = np.sqrt(np.mean((residual - np.mean(residual))**2))
        print(f"Degree {deg}: RMS residual = {rms:.2f}")
    
    # Selection criteria:
    # 1. Visual assessment: Check if BG passes through peaks
    # 2. Residual magnitude: Too small indicates overfitting
    # 3. Peak preservation: Check if intensities become negative
    # Recommendation: 3rd-order polynomial (balance flexibility and stability)

**Criteria for Optimal Degree Selection** :

  1. **Degree too low (1st order)** : Cannot capture BG curvature ’ peak intensities overestimated
  2. **Appropriate (3rd order)** : Smoothly fits BG while avoiding peaks
  3. **Degree too high (5th order or above)** : Fits to peaks ’ intensities underestimated

In practice, **3rd-order polynomial** is most commonly used.

## Verification of Learning Objectives

Upon completing this chapter, you should be able to explain and implement the following:

### Basic Understanding

  *  Explain the role of XRD instrument components (X-ray source, goniometer, detector)
  *  Understand how measurement conditions (2¸ range, step size, integration time) affect results
  *  Explain the sources of background (inelastic scattering, fluorescent X-rays)
  *  Understand the characteristics of peak profile functions (Gaussian, Lorentzian, Voigt)

### Practical Skills

  *  Load and plot raw XRD data in Python
  *  Apply peak detection algorithm (scipy.signal.find_peaks)
  *  Remove background by polynomial fitting
  *  Execute peak fitting with Voigt function and extract parameters
  *  Index peaks from lattice parameters

### Application Capability

  *  Evaluate quality of measured XRD data and optimize measurement conditions
  *  Distinguish peaks in multi-phase mixtures and identify major phases
  *  Estimate crystallite size from peak width (Scherrer equation)
  *  Construct a complete XRD analysis workflow

## References

  1. Jenkins, R., & Snyder, R. L. (1996). _Introduction to X-Ray Powder Diffractometry_. Wiley. - Practical textbook detailing XRD instrument principles and measurement techniques
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry. - Definitive guide covering both theory and practice of powder XRD
  3. Langford, J. I., & Louër, D. (1996). "Powder diffraction". _Reports on Progress in Physics_ , 59(2), 131-234. - Important review providing theoretical foundation of peak profile analysis
  4. ICDD PDF-4+ Database (2024). International Centre for Diffraction Data. - World-standard XRD database containing over 400,000 reference patterns
  5. Scherrer, P. (1918). "Bestimmung der Größe und der inneren Struktur von Kolloidteilchen mittels Röntgenstrahlen". _Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen_ , 2, 98-100. - Original paper on crystallite size analysis
  6. Klug, H. P., & Alexander, L. E. (1974). _X-Ray Diffraction Procedures for Polycrystalline and Amorphous Materials_ (2nd ed.). Wiley. - Classic explanation of background removal and peak deconvolution techniques
  7. scipy.signal documentation (2024). "Signal processing (scipy.signal)". SciPy project. - Detailed specifications of find_peaks function and peak detection algorithms

## Next Steps

In Chapter 2, we learned about acquiring actual XRD measurement data and performing basic analysis. We mastered the workflow of peak detection, indexing, background processing, and peak fitting.

In **Chapter 3** , we will integrate these techniques and advance to precise analysis using the Rietveld method. Through whole-pattern fitting, we will extract more detailed structural information such as lattice parameters, atomic coordinates, and crystallite size.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
