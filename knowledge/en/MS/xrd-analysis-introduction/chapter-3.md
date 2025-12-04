---
title: "Chapter 3: Introduction to Rietveld Analysis"
chapter_title: "Chapter 3: Introduction to Rietveld Analysis"
subtitle: Precise Crystal Structure Analysis through Whole Pattern Fitting
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 8
---

This chapter introduces the basics of Introduction to Rietveld Analysis. You will learn background modeling using Chebyshev polynomials and Calculate R-factors (Rwp.

## Learning Objectives

Upon completing this chapter, you will be able to:

  * Understand and implement the principles of the Rietveld method and least-squares minimization
  * Perform whole pattern fitting using Pseudo-Voigt profile functions
  * Implement background modeling using Chebyshev polynomials
  * Calculate R-factors (Rwp, RB, GOF) and evaluate fit quality
  * Execute practical Rietveld analysis using lmfit.Minimizer

## 3.1 Principles of the Rietveld Method

### 3.1.1 Whole Pattern Fitting

The Rietveld method is a whole pattern fitting technique for powder X-ray diffraction data developed by Hugo Rietveld in 1969. Unlike conventional methods that fit individual peaks independently, it **optimizes the entire measured pattern simultaneously**.

Minimize the sum of squared differences between observed intensity \\( y_i^{obs} \\) and calculated intensity \\( y_i^{calc} \\):

\\[ S = \sum_{i} w_i \left(y_i^{obs} - y_i^{calc}\right)^2 \\] 

where \\( w_i = 1/y_i^{obs} \\) is the statistical weight (based on counting statistics), and \\( i \\) is the index of measurement points.
    
    
    ```mermaid
    flowchart TD
        A[Initial Structure ModelLattice constants, atomic coordinates] --> B[Calculate Diffraction Patterny_calc]
        B --> C[Calculate difference fromobserved pattern y_obs]
        C --> D{Residual Ssmall enough?}
        D -->|No| E[Adjust ParametersLeast-squares method]
        E --> B
        D -->|Yes| F[Final Structure ModelOutput R-factors]
    
        style A fill:#fce7f3
        style D fill:#fff3e0
        style F fill:#e8f5e9
    ```

### 3.1.2 Formulation of Calculated Intensity

The calculated intensity at measurement point \\( i \\) is given by:

\\[ y_i^{calc} = y_{bi} + \sum_{K} s_K \sum_{hkl} m_{hkl} |F_{hkl}|^2 \Omega(2\theta_i - 2\theta_{hkl}) A_{hkl} \\] 

Meaning of each term:

  * \\( y_{bi} \\): Background intensity
  * \\( s_K \\): Scale factor of phase \\( K \\)
  * \\( m_{hkl} \\): Multiplicity
  * \\( |F_{hkl}|^2 \\): Square of structure factor
  * \\( \Omega \\): Peak profile function
  * \\( A_{hkl} \\): Lorentz-polarization factor, absorption correction, etc.

### 3.1.3 Implementation of Least-Squares Method
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import minimize
    
    class RietveldRefinement:
        """Basic implementation of Rietveld analysis"""
    
        def __init__(self, two_theta_obs, intensity_obs):
            """
            Args:
                two_theta_obs (np.ndarray): Observed 2theta angles
                intensity_obs (np.ndarray): Observed intensities
            """
            self.two_theta_obs = two_theta_obs
            self.intensity_obs = intensity_obs
            self.weights = 1.0 / np.sqrt(np.maximum(intensity_obs, 1.0))  # Statistical weights
    
        def calculate_pattern(self, params, two_theta, peak_positions, structure_factors):
            """Calculate diffraction pattern
    
            Args:
                params (dict): Refinement parameters
                two_theta (np.ndarray): 2theta angles
                peak_positions (list): Peak positions [(2theta_hkl, m_hkl), ...]
                structure_factors (list): Squares of structure factors [|F_hkl|^2, ...]
    
            Returns:
                np.ndarray: Calculated intensity
            """
            # Background (implemented with Chebyshev polynomials later)
            background = self._calculate_background(two_theta, params['bg_coeffs'])
    
            # Peak contributions
            intensity_peaks = np.zeros_like(two_theta)
    
            for (peak_2theta, multiplicity), F_hkl_sq in zip(peak_positions, structure_factors):
                # Profile function (Pseudo-Voigt)
                profile = self._pseudo_voigt_profile(
                    two_theta,
                    center=peak_2theta,
                    fwhm=params['fwhm'],
                    eta=params['eta']
                )
    
                # Scale factor × multiplicity × structure factor
                intensity_peaks += params['scale'] * multiplicity * F_hkl_sq * profile
    
            return background + intensity_peaks
    
        def _pseudo_voigt_profile(self, x, center, fwhm, eta):
            """Pseudo-Voigt profile function
    
            Args:
                x (np.ndarray): 2theta
                center (float): Peak center
                fwhm (float): Full width at half maximum
                eta (float): Lorentzian component fraction (0-1)
    
            Returns:
                np.ndarray: Profile shape
            """
            # Gaussian component
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
            # Lorentzian component
            gamma = fwhm / 2
            lorentz = gamma / (np.pi * (gamma**2 + (x - center)**2))
    
            # Pseudo-Voigt
            return eta * lorentz + (1 - eta) * gauss
    
        def _calculate_background(self, two_theta, coeffs):
            """Calculate background with Chebyshev polynomials"""
            # Normalized x: [-1, 1]
            x_norm = 2 * (two_theta - two_theta.min()) / (two_theta.max() - two_theta.min()) - 1
    
            # Calculate Chebyshev polynomials T_n(x)
            bg = np.zeros_like(two_theta)
            for n, c in enumerate(coeffs):
                bg += c * np.polynomial.chebyshev.chebval(x_norm, [0]*n + [1])
    
            return bg
    
        def residual(self, params, two_theta, peak_positions, structure_factors):
            """Residual function (objective function for minimization)
    
            Returns:
                np.ndarray: Weighted residuals
            """
            y_calc = self.calculate_pattern(params, two_theta, peak_positions, structure_factors)
            residual = (self.intensity_obs - y_calc) * self.weights
            return residual
    
        def chi_squared(self, residual):
            """Calculate chi^2"""
            return np.sum(residual**2)
    
    
    # Usage example: Simple Rietveld analysis
    def simple_rietveld_demo():
        """Demo of simple Rietveld analysis"""
        # Generate mock data
        two_theta = np.linspace(10, 80, 3500)
        true_params = {
            'scale': 1000,
            'fwhm': 0.15,
            'eta': 0.5,
            'bg_coeffs': [100, -20, 5]
        }
    
        # Peak positions and structure factors (alpha-Fe BCC)
        peak_positions = [(44.67, 12), (65.02, 6), (82.33, 24)]  # (2theta, multiplicity)
        structure_factors = [1.0, 0.8, 1.2]  # Normalized |F|^2
    
        # True calculated pattern
        rietveld = RietveldRefinement(two_theta, np.zeros_like(two_theta))
        y_true = rietveld.calculate_pattern(true_params, two_theta, peak_positions, structure_factors)
    
        # Add noise
        y_obs = y_true + np.random.normal(0, np.sqrt(y_true + 10), len(y_true))
    
        print("=== Rietveld Analysis Demo ===")
        print(f"Number of data points: {len(two_theta)}")
        print(f"Number of peaks: {len(peak_positions)}")
        print(f"Number of refinement parameters: {1 + 1 + 1 + len(true_params['bg_coeffs'])} (scale, FWHM, eta, BG coeffs×3)")
    
        return two_theta, y_obs, peak_positions, structure_factors
    
    two_theta, y_obs, peak_pos, F_sq = simple_rietveld_demo()

## 3.2 Profile Functions

### 3.2.1 Details of the Pseudo-Voigt Function

The Pseudo-Voigt function is the most commonly used profile function in Rietveld analysis:

\\[ \Omega(x) = \eta L(x) + (1-\eta) G(x) \\] 

The parameter \\( \eta \\) (0-1) represents the contribution of the Lorentzian component.

**Angular dependence of FWHM (Full Width at Half Maximum)** \- Caglioti equation:

\\[ \text{FWHM}^2 = U\tan^2\theta + V\tan\theta + W \\] 

where \\( U, V, W \\) are Caglioti parameters.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    def caglioti_fwhm(two_theta, U, V, W):
        """Calculate FWHM using Caglioti equation
    
        Args:
            two_theta (float or np.ndarray): 2theta angle [degrees]
            U, V, W (float): Caglioti parameters
    
        Returns:
            float or np.ndarray: FWHM [degrees]
        """
        theta_rad = np.radians(two_theta / 2)
        tan_theta = np.tan(theta_rad)
    
        fwhm_squared = U * tan_theta**2 + V * tan_theta + W
    
        # Avoid negative values
        fwhm_squared = np.maximum(fwhm_squared, 0.001)
    
        return np.sqrt(fwhm_squared)
    
    
    # Typical values of U, V, W parameters (Cu Kalpha, laboratory XRD)
    U = 0.01  # [degrees^2]
    V = -0.005  # [degrees^2]
    W = 0.005  # [degrees^2]
    
    # FWHM at various angles
    two_theta_range = np.linspace(10, 120, 100)
    fwhm_values = caglioti_fwhm(two_theta_range, U, V, W)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(two_theta_range, fwhm_values, color='#f093fb', linewidth=2)
    plt.xlabel('2theta [degrees]', fontsize=12)
    plt.ylabel('FWHM [degrees]', fontsize=12)
    plt.title('Caglioti Equation: Angular Dependence of FWHM', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== FWHM Calculation Examples ===")
    for angle in [20, 40, 60, 80, 100]:
        fwhm = caglioti_fwhm(angle, U, V, W)
        print(f"2theta = {angle:3d} degrees: FWHM = {fwhm:.4f} degrees")

### 3.2.2 Thompson-Cox-Hastings Pseudo-Voigt (TCH-pV)

For higher accuracy profile functions, there is TCH-pV. This treats the FWHM of Gaussian and Lorentzian components independently:
    
    
    def tch_pseudo_voigt(x, center, fwhm_G, fwhm_L):
        """Thompson-Cox-Hastings Pseudo-Voigt function
    
        Args:
            x (np.ndarray): 2theta
            center (float): Peak center
            fwhm_G (float): FWHM of Gaussian component
            fwhm_L (float): FWHM of Lorentzian component
    
        Returns:
            np.ndarray: Profile
        """
        # Effective FWHM
        fwhm_eff = (fwhm_G**5 + 2.69269 * fwhm_G**4 * fwhm_L +
                    2.42843 * fwhm_G**3 * fwhm_L**2 +
                    4.47163 * fwhm_G**2 * fwhm_L**3 +
                    0.07842 * fwhm_G * fwhm_L**4 + fwhm_L**5) ** 0.2
    
        # Mixing parameter eta
        eta = 1.36603 * (fwhm_L / fwhm_eff) - 0.47719 * (fwhm_L / fwhm_eff)**2 + \
              0.11116 * (fwhm_L / fwhm_eff)**3
    
        # Gaussian component
        sigma_G = fwhm_eff / (2 * np.sqrt(2 * np.log(2)))
        G = np.exp(-0.5 * ((x - center) / sigma_G)**2) / (sigma_G * np.sqrt(2 * np.pi))
    
        # Lorentzian component
        gamma_L = fwhm_eff / 2
        L = gamma_L / (np.pi * (gamma_L**2 + (x - center)**2))
    
        return eta * L + (1 - eta) * G
    
    
    # Comparison: Simple pV vs TCH-pV
    x = np.linspace(44, 46, 500)
    center = 45.0
    
    # Simple pV
    simple_pv = RietveldRefinement(x, np.zeros_like(x))._pseudo_voigt_profile(
        x, center, fwhm=0.2, eta=0.5
    )
    
    # TCH-pV
    tch_pv = tch_pseudo_voigt(x, center, fwhm_G=0.15, fwhm_L=0.1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, simple_pv, label='Simple Pseudo-Voigt', color='#3498db', linewidth=2)
    plt.plot(x, tch_pv, label='TCH Pseudo-Voigt', color='#f093fb', linewidth=2, linestyle='--')
    plt.xlabel('2theta [degrees]', fontsize=12)
    plt.ylabel('Profile Intensity (normalized)', fontsize=12)
    plt.title('Comparison of Profile Functions', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## 3.3 Background Model

### 3.3.1 Chebyshev Polynomials

Chebyshev polynomials are the standard choice for modeling backgrounds in Rietveld analysis. They are numerically stable and can represent complex shapes with few parameters:

\\[ y_{bg}(x) = \sum_{n=0}^{N} c_n T_n(x) \\] 

where \\( T_n(x) \\) is the \\( n \\)-th order Chebyshev polynomial, \\( x \in [-1, 1] \\) (normalized 2theta).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy.polynomial.chebyshev as cheb
    
    class BackgroundModel:
        """Background modeling class"""
    
        def __init__(self, two_theta_range):
            """
            Args:
                two_theta_range (tuple): (min, max) measurement range
            """
            self.two_theta_min = two_theta_range[0]
            self.two_theta_max = two_theta_range[1]
    
        def normalize_two_theta(self, two_theta):
            """Normalize 2theta to [-1, 1]
    
            Args:
                two_theta (np.ndarray): 2theta angles
    
            Returns:
                np.ndarray: Normalized values
            """
            return 2 * (two_theta - self.two_theta_min) / (self.two_theta_max - self.two_theta_min) - 1
    
        def chebyshev_background(self, two_theta, coefficients):
            """Calculate background with Chebyshev polynomials
    
            Args:
                two_theta (np.ndarray): 2theta angles
                coefficients (list): Chebyshev coefficients [c0, c1, c2, ...]
    
            Returns:
                np.ndarray: Background intensity
            """
            x_norm = self.normalize_two_theta(two_theta)
            return cheb.chebval(x_norm, coefficients)
    
        def fit_background(self, two_theta, intensity, degree=5, exclude_peaks=True):
            """Fit background
    
            Args:
                two_theta (np.ndarray): 2theta angles
                intensity (np.ndarray): Intensity
                degree (int): Degree of Chebyshev polynomial
                exclude_peaks (bool): Whether to exclude peak regions
    
            Returns:
                np.ndarray: Chebyshev coefficients
            """
            x_norm = self.normalize_two_theta(two_theta)
    
            if exclude_peaks:
                # Exclude high intensity regions (simplified version)
                threshold = np.percentile(intensity, 60)
                mask = intensity < threshold
                coeffs = cheb.chebfit(x_norm[mask], intensity[mask], degree)
            else:
                coeffs = cheb.chebfit(x_norm, intensity, degree)
    
            return coeffs
    
    
    # Usage example
    two_theta = np.linspace(10, 80, 3500)
    
    # Simulate complex background shape
    true_bg = 200 * np.exp(-two_theta / 40) + 50 + 10 * np.sin(two_theta / 10)
    
    # Add peaks
    peak1 = 1000 * np.exp(-0.5 * ((two_theta - 45) / 0.15)**2)
    peak2 = 500 * np.exp(-0.5 * ((two_theta - 65) / 0.18)**2)
    intensity = true_bg + peak1 + peak2 + np.random.normal(0, 5, len(two_theta))
    
    # Background fitting
    bg_model = BackgroundModel((10, 80))
    
    # Fit with different degrees
    degrees = [3, 5, 7, 10]
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(two_theta, intensity, 'gray', alpha=0.5, label='All data')
    plt.plot(two_theta, true_bg, 'k--', linewidth=2, label='True BG')
    
    for deg in degrees:
        coeffs = bg_model.fit_background(two_theta, intensity, degree=deg, exclude_peaks=True)
        bg_fitted = bg_model.chebyshev_background(two_theta, coeffs)
        plt.plot(two_theta, bg_fitted, linewidth=1.5, label=f'Chebyshev deg={deg}')
    
    plt.xlabel('2theta [degrees]')
    plt.ylabel('Intensity [counts]')
    plt.title('Chebyshev Background Fitting')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Residual plot
    best_deg = 5
    coeffs = bg_model.fit_background(two_theta, intensity, degree=best_deg, exclude_peaks=True)
    bg_fitted = bg_model.chebyshev_background(two_theta, coeffs)
    residual = intensity - bg_fitted
    
    plt.plot(two_theta, residual, color='#f093fb', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('2theta [degrees]')
    plt.ylabel('Residual [counts]')
    plt.title(f'Residual (Chebyshev degree={best_deg})')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"=== Chebyshev Coefficients (degree={best_deg}) ===")
    for i, c in enumerate(coeffs):
        print(f"c_{i} = {c:10.4f}")

## 3.4 R-factors and Goodness-of-Fit Evaluation

### 3.4.1 Definition of R-factors

The quality of Rietveld analysis is evaluated using various R-factors:

**Rwp (Weighted Profile R-factor)** \- Most important:

\\[ R_{wp} = \sqrt{\frac{\sum_i w_i (y_i^{obs} - y_i^{calc})^2}{\sum_i w_i (y_i^{obs})^2}} \times 100\% \\] 

**Rp (Profile R-factor)** \- Unweighted version:

\\[ R_p = \frac{\sum_i |y_i^{obs} - y_i^{calc}|}{\sum_i y_i^{obs}} \times 100\% \\] 

**RBragg (Bragg R-factor)** \- Based on integrated intensity:

\\[ R_B = \frac{\sum_{hkl} |I_{hkl}^{obs} - I_{hkl}^{calc}|}{\sum_{hkl} I_{hkl}^{obs}} \times 100\% \\] 

**GOF (Goodness of Fit)** \- Comparison with expected value:

\\[ \text{GOF} = \sqrt{\frac{\sum_i w_i (y_i^{obs} - y_i^{calc})^2}{N - P}} = \frac{R_{wp}}{R_{exp}} \\] 

where \\( N \\) is the number of data points, \\( P \\) is the number of parameters, and \\( R_{exp} \\) is the expected R-factor.

R-factor | Excellent | Good | Acceptable | Problematic  
---|---|---|---|---  
Rwp | < 5% | 5-10% | 10-15% | > 15%  
RBragg | < 3% | 3-7% | 7-12% | > 12%  
GOF | 1.0-1.3 | 1.3-2.0 | 2.0-3.0 | > 3.0 or < 1.0  
  
### 3.4.2 Implementation of R-factor Calculation
    
    
    class RFactorCalculator:
        """R-factor calculation class"""
    
        @staticmethod
        def calculate_rwp(y_obs, y_calc, weights=None):
            """Calculate Weighted Profile R-factor
    
            Args:
                y_obs (np.ndarray): Observed intensity
                y_calc (np.ndarray): Calculated intensity
                weights (np.ndarray): Weights (if None, use 1/y_obs)
    
            Returns:
                float: Rwp [%]
            """
            if weights is None:
                weights = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
    
            numerator = np.sum(weights * (y_obs - y_calc)**2)
            denominator = np.sum(weights * y_obs**2)
    
            return 100 * np.sqrt(numerator / denominator)
    
        @staticmethod
        def calculate_rp(y_obs, y_calc):
            """Calculate Profile R-factor
    
            Returns:
                float: Rp [%]
            """
            numerator = np.sum(np.abs(y_obs - y_calc))
            denominator = np.sum(y_obs)
    
            return 100 * numerator / denominator
    
        @staticmethod
        def calculate_rexp(y_obs, n_params, weights=None):
            """Calculate Expected R-factor
    
            Args:
                y_obs (np.ndarray): Observed intensity
                n_params (int): Number of parameters
                weights (np.ndarray): Weights
    
            Returns:
                float: Rexp [%]
            """
            if weights is None:
                weights = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
    
            N = len(y_obs)
            numerator = N - n_params
            denominator = np.sum(weights * y_obs**2)
    
            return 100 * np.sqrt(numerator / denominator)
    
        @staticmethod
        def calculate_gof(rwp, rexp):
            """Calculate Goodness of Fit
    
            Returns:
                float: GOF
            """
            return rwp / rexp
    
        @classmethod
        def calculate_all_r_factors(cls, y_obs, y_calc, n_params, weights=None):
            """Calculate all R-factors
    
            Returns:
                dict: Dictionary of R-factors
            """
            rwp = cls.calculate_rwp(y_obs, y_calc, weights)
            rp = cls.calculate_rp(y_obs, y_calc)
            rexp = cls.calculate_rexp(y_obs, n_params, weights)
            gof = cls.calculate_gof(rwp, rexp)
    
            return {
                'Rwp': rwp,
                'Rp': rp,
                'Rexp': rexp,
                'GOF': gof
            }
    
    
    # Usage example
    # Mock data: Good fit
    np.random.seed(42)
    y_obs = np.abs(np.random.normal(1000, 50, 1000))
    y_calc_good = y_obs + np.random.normal(0, 10, 1000)  # Small error
    
    # Mock data: Poor fit
    y_calc_bad = y_obs + np.random.normal(0, 100, 1000)  # Large error
    
    n_params = 15  # Number of parameters
    
    # Calculate R-factors
    r_good = RFactorCalculator.calculate_all_r_factors(y_obs, y_calc_good, n_params)
    r_bad = RFactorCalculator.calculate_all_r_factors(y_obs, y_calc_bad, n_params)
    
    print("=== R-factor Comparison ===")
    print("\nGood fit:")
    for key, value in r_good.items():
        print(f"  {key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
    print("\nPoor fit:")
    for key, value in r_bad.items():
        print(f"  {key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
    # Judgment
    print("\nJudgment:")
    if r_good['Rwp'] < 10 and 1.0 < r_good['GOF'] < 2.0:
        print("  Good: Rwp < 10%, 1.0 < GOF < 2.0 ")
    else:
        print("  Needs improvement")
    
    if r_bad['Rwp'] > 15 or r_bad['GOF'] > 3.0:
        print("  Poor: Rwp > 15% or GOF > 3.0 ")

## 3.5 Python Implementation Fundamentals (Using lmfit)

### 3.5.1 Introduction to lmfit.Minimizer

The lmfit library wraps scipy.optimize and makes it easy to handle parameter bounds, constraints, and error estimation:
    
    
    # Install lmfit (if needed)
    # !pip install lmfit
    
    from lmfit import Parameters, Minimizer, report_fit
    
    class LmfitRietveldRefinement:
        """Rietveld analysis using lmfit"""
    
        def __init__(self, two_theta_obs, intensity_obs):
            self.two_theta_obs = two_theta_obs
            self.intensity_obs = intensity_obs
            self.weights = 1.0 / np.sqrt(np.maximum(intensity_obs, 1.0))
    
        def setup_parameters(self):
            """Set initial values and bounds for parameters
    
            Returns:
                lmfit.Parameters: Parameter object
            """
            params = Parameters()
    
            # Scale factor
            params.add('scale', value=1000, min=0, max=1e6)
    
            # Caglioti parameters
            params.add('U', value=0.01, min=0, max=1.0)
            params.add('V', value=-0.005, min=-1.0, max=1.0)
            params.add('W', value=0.005, min=0, max=1.0)
    
            # Pseudo-Voigt mixing parameter
            params.add('eta', value=0.5, min=0, max=1)
    
            # Chebyshev background coefficients
            params.add('bg0', value=100, min=0)
            params.add('bg1', value=0)
            params.add('bg2', value=0)
            params.add('bg3', value=0)
    
            # Lattice constant (cubic example)
            params.add('a', value=2.87, min=2.8, max=3.0)
    
            return params
    
        def calculate_pattern_lmfit(self, params, two_theta, hkl_list):
            """Calculate diffraction pattern from lmfit parameter object
    
            Args:
                params (lmfit.Parameters): Parameters
                two_theta (np.ndarray): 2theta
                hkl_list (list): [(h, k, l, multiplicity, |F|^2), ...]
    
            Returns:
                np.ndarray: Calculated intensity
            """
            # Extract parameters
            scale = params['scale'].value
            U = params['U'].value
            V = params['V'].value
            W = params['W'].value
            eta = params['eta'].value
            a = params['a'].value
    
            bg_coeffs = [params[f'bg{i}'].value for i in range(4)]
    
            # Background
            bg_model = BackgroundModel((two_theta.min(), two_theta.max()))
            background = bg_model.chebyshev_background(two_theta, bg_coeffs)
    
            # Peak calculation
            intensity_peaks = np.zeros_like(two_theta)
            wavelength = 1.54056  # Cu Kalpha
    
            for h, k, l, mult, F_sq in hkl_list:
                # Calculate d-value (cubic)
                d = a / np.sqrt(h**2 + k**2 + l**2)
    
                # Bragg angle
                theta = np.degrees(np.arcsin(wavelength / (2 * d)))
                peak_2theta = 2 * theta
    
                # FWHM (Caglioti equation)
                fwhm = caglioti_fwhm(peak_2theta, U, V, W)
    
                # Pseudo-Voigt profile
                profile = self._pseudo_voigt_profile(two_theta, peak_2theta, fwhm, eta)
    
                # Scale × multiplicity × structure factor
                intensity_peaks += scale * mult * F_sq * profile
    
            return background + intensity_peaks
    
        def _pseudo_voigt_profile(self, x, center, fwhm, eta):
            """Pseudo-Voigt profile"""
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
            gamma = fwhm / 2
            lorentz = gamma / (np.pi * (gamma**2 + (x - center)**2))
    
            return eta * lorentz + (1 - eta) * gauss
    
        def residual_lmfit(self, params, two_theta, intensity_obs, weights, hkl_list):
            """Residual function for lmfit
    
            Returns:
                np.ndarray: Weighted residuals
            """
            y_calc = self.calculate_pattern_lmfit(params, two_theta, hkl_list)
            return (intensity_obs - y_calc) * weights
    
    
    # Example execution
    def run_lmfit_rietveld():
        """Execute Rietveld analysis with lmfit"""
        # Generate mock data (alpha-Fe BCC)
        two_theta = np.linspace(40, 90, 2500)
    
        hkl_list = [
            (1, 1, 0, 12, 1.0),
            (2, 0, 0, 6, 0.8),
            (2, 1, 1, 24, 1.2),
            (2, 2, 0, 12, 0.6),
        ]
    
        # Calculate with true parameters
        true_params = {
            'scale': 5000,
            'U': 0.01, 'V': -0.005, 'W': 0.005,
            'eta': 0.5,
            'bg0': 100, 'bg1': -10, 'bg2': 2, 'bg3': 0,
            'a': 2.87
        }
    
        rietveld = LmfitRietveldRefinement(two_theta, np.zeros_like(two_theta))
    
        # Generate true pattern
        params_true = rietveld.setup_parameters()
        for key, value in true_params.items():
            params_true[key].value = value
    
        y_true = rietveld.calculate_pattern_lmfit(params_true, two_theta, hkl_list)
        y_obs = y_true + np.random.normal(0, np.sqrt(y_true + 10), len(y_true))
    
        # Initial parameters (slightly offset from true values)
        params_init = rietveld.setup_parameters()
        params_init['scale'].value = 4500
        params_init['a'].value = 2.90
    
        # Execute minimization
        minimizer = Minimizer(
            rietveld.residual_lmfit,
            params_init,
            fcn_args=(two_theta, y_obs, rietveld.weights, hkl_list)
        )
    
        result = minimizer.minimize(method='leastsq')  # Levenberg-Marquardt method
    
        # Display results
        print("=== Rietveld Analysis Results (lmfit) ===\n")
        report_fit(result)
    
        # Calculate R-factors
        y_final = rietveld.calculate_pattern_lmfit(result.params, two_theta, hkl_list)
        r_factors = RFactorCalculator.calculate_all_r_factors(
            y_obs, y_final, result.nvarys, rietveld.weights
        )
    
        print("\n=== R-factors ===")
        for key, value in r_factors.items():
            print(f"{key:<6}: {value:6.2f}{'%' if key != 'GOF' else ''}")
    
        # Plot
        plt.figure(figsize=(12, 8))
    
        plt.subplot(2, 1, 1)
        plt.plot(two_theta, y_obs, 'o', markersize=2, color='gray', alpha=0.5, label='Observed data')
        plt.plot(two_theta, y_final, color='#f093fb', linewidth=2, label='Calculated pattern')
        plt.xlabel('2theta [degrees]')
        plt.ylabel('Intensity [counts]')
        plt.title('Rietveld Analysis: Fit Result')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.subplot(2, 1, 2)
        residual = y_obs - y_final
        plt.plot(two_theta, residual, color='#f5576c', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        plt.xlabel('2theta [degrees]')
        plt.ylabel('Residual [counts]')
        plt.title(f'Residual Plot (Rwp = {r_factors["Rwp"]:.2f}%)')
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        return result
    
    # Execute
    # result = run_lmfit_rietveld()  # Uncomment to run

## Confirmation of Learning Objectives

### Basic Understanding

  *  Principles of whole pattern fitting in the Rietveld method and least-squares minimization
  *  Physical meaning of Pseudo-Voigt function and Caglioti equation
  *  Advantages of background modeling using Chebyshev polynomials
  *  Definitions and evaluation criteria for R-factors (Rwp, Rp, RBragg, GOF)

### Practical Skills

  *  Implement Pseudo-Voigt profile function in NumPy
  *  Fit complex backgrounds with Chebyshev polynomials
  *  Calculate R-factors and quantitatively evaluate fit quality
  *  Execute practical Rietveld analysis with lmfit.Minimizer

### Application Skills

  *  Properly set initial values and bounds for parameters
  *  Extract lattice constants, FWHM, and scale factors from fit results
  *  Diagnose refinement issues from residual plots and R-factors

## Practice Problems

### Easy (Basic Confirmation)

**Q1** : If Rwp = 8.5% and Rexp = 5.2%, what is the GOF? Is this result good?

**Answer** :
    
    
    GOF = Rwp / Rexp = 8.5 / 5.2 = 1.63

**Judgment** : GOF = 1.63 is **in the range 1.3-2.0** ’ Good fit 

If GOF > 2.0, there are problems; if GOF < 1.0, there may be overfitting.

**Q2** : Using the Caglioti equation, calculate the FWHM at 2theta = 30 degrees (U=0.01, V=-0.005, W=0.005).

**Answer** :
    
    
    fwhm = caglioti_fwhm(two_theta=30, U=0.01, V=-0.005, W=0.005)
    print(f"FWHM at 2theta=30 degrees: {fwhm:.4f} degrees")
    # Output: FWHM at 2theta=30 degrees: 0.0844 degrees

### Medium (Application)

**Q3** : If you increase the degree of Chebyshev polynomials from 3 to 7, will Rwp necessarily improve? Explain why.

**Answer** :

**Conclusion** : It usually improves, but not necessarily physically correct.

**Reasons** :

  * **Rwp decreases** : More parameters improve fit to data
  * **However...**
    * Overfitting: May treat peak regions as background
    * GOF deterioration: As number of parameters P increases, Rexp becomes smaller, causing GOF = Rwp/Rexp to increase
    * Loss of physical meaning: Polynomials of 7th order or higher oscillate and generate non-physical background shapes

**Recommendation** : Chebyshev polynomials of degree 3-5 are practical. Verify with visual evaluation and GOF.

**Q4** : For the Pseudo-Voigt function, explain the difference in shape when eta = 0 vs eta = 1.

**Answer** :

  * **eta = 0** : Pure Gaussian ’ Tails decay rapidly, instrument-induced broadening
  * **eta = 1** : Pure Lorentzian ’ Wide tails, sample-induced (crystallite size, strain)

Actual XRD peaks are intermediate between the two (eta = 0.3-0.7), reflecting both instrument and sample effects.

### Hard (Advanced)

**Q5** : When refining lattice constant a with lmfit, you set initial value to 2.87 Angstroms and bounds to [2.8, 3.0]. After fitting, a = 2.999 Angstroms. How should you interpret this result?

**Answer** :

**Problem** : Parameter is stuck at boundary ’ Optimization may not have converged

**Solutions** :

  1. **Widen bounds** : Change to [2.7, 3.2] and re-run
  2. **Revise initial value** : Use value estimated from indexing
  3. **Check correlation with other parameters** : The U parameter and a may be strongly correlated
  4. **Verify systematic absences** : Re-verify that the assumed lattice type (BCC/FCC/SC) is correct
  5. **Check high-angle peaks** : Lattice constant is sensitive to high-angle data ’ Check data at 2theta > 80 degrees

If a = 2.99 Angstroms after re-refinement, it may be the true value. However, **verification with other methods (single crystal XRD, neutron diffraction) is desirable**.

## Confirmation of Learning Objectives

Upon completing this chapter, you should be able to explain and implement the following:

### Basic Understanding

  *  Explain the principles of the Rietveld method (whole pattern fitting by least-squares minimization)
  *  Understand the physical meaning of Pseudo-Voigt profile function parameters (FWHM, eta)
  *  Explain the advantages of background modeling using Chebyshev polynomials
  *  Understand the definitions and interpretation criteria for R-factors (Rwp, RB, GOF)

### Practical Skills

  *  Implement Rietveld fitting using Python's lmfit library
  *  Properly initialize and refine profile parameters
  *  Model background with polynomials and select optimal degree
  *  Calculate R-factors and quantitatively evaluate fitting quality
  *  Generate difference plots (Yobs - Ycalc) and visually evaluate fitting

### Application Skills

  *  Execute Rietveld analysis on actual measured XRD data
  *  Understand parameter correlations and formulate refinement strategies
  *  Troubleshoot when convergence fails
  *  Evaluate the validity of refinement results from multiple perspectives

## References

  1. Rietveld, H. M. (1969). "A profile refinement method for nuclear and magnetic structures". _Journal of Applied Crystallography_ , 2(2), 65-71. - Original paper on the Rietveld method, foundations of whole pattern fitting
  2. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press. - Definitive comprehensive explanation of theory and practice of the Rietveld method
  3. McCusker, L. B., et al. (1999). "Rietveld refinement guidelines". _Journal of Applied Crystallography_ , 32(1), 36-50. - Official guidelines for Rietveld analysis established by the International Union of Crystallography
  4. Toby, B. H. (2006). "R factors in Rietveld analysis: How good is good enough?". _Powder Diffraction_ , 21(1), 67-70. - Important paper clearly showing interpretation criteria for R-factors
  5. Thompson, P., Cox, D. E., & Hastings, J. B. (1987). "Rietveld refinement of Debye-Scherrer synchrotron X-ray data from Al‚Oƒ". _Journal of Applied Crystallography_ , 20(2), 79-83. - Paper on formulation of Pseudo-Voigt function
  6. Cheary, R. W., & Coelho, A. (1992). "A fundamental parameters approach to X-ray line-profile fitting". _Journal of Applied Crystallography_ , 25(2), 109-121. - Physical interpretation of profile parameters
  7. lmfit documentation (2024). "Non-Linear Least-Squares Minimization and Curve-Fitting for Python". - Practical guide for Python Rietveld implementation

## Next Steps

In Chapter 3, we learned the theory and basic implementation of the Rietveld method. You have mastered the foundations of precision analysis: whole pattern fitting, profile functions, background modeling, and evaluation using R-factors.

In **Chapter 4** , we will advance these techniques and proceed to refinement of more detailed structural parameters such as atomic coordinates, thermal factors, crystallite size, and microstrain. You will learn advanced refinement techniques using constraints and restraints.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
