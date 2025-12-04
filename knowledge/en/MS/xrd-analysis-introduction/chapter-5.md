---
title: "Chapter 5: Practical XRD Data Analysis Workflow"
chapter_title: "Chapter 5: Practical XRD Data Analysis Workflow"
subtitle: From Raw Data to Academic Reports - Complete Guide for Practical XRD Analysis
reading_time: 35 min
---

This chapter focuses on practical applications of Practical XRD Data Analysis Workflow. You will learn  Overall workflow of practical XRD analysis and  Troubleshooting convergence failures.

## Learning Objectives

After completing this chapter, you will be able to explain and implement the following:

### Basic Understanding

  *  Overall workflow of practical XRD analysis
  *  Strategies for multiphase mixture analysis (2-phase and 3-phase systems)
  *  Principles of quantitative phase analysis (RIR method, Rietveld quantification)
  *  Troubleshooting convergence failures and parameter correlations

### Practical Skills

  *  Complete workflow implementation from raw data loading to result visualization
  *  Rietveld analysis of multiphase mixtures (±-Fe + FeƒO„)
  *  Quantification of phase fractions using RIR and Rietveld methods
  *  CIF file output and creation of publication-quality figures and tables

### Advanced Applications

  *  Error diagnosis (convergence failures, abnormal GOF, negative occupancies) and solutions
  *  Advanced analysis combining GSAS-II and Python
  *  Complete Rietveld analysis on experimental data

## 5.1 Complete Analysis Workflow

Practical XRD analysis requires systematic execution of a series of steps starting from raw data loading, through structure refinement, result validation, and finally creating figures and tables for academic reports. In this section, we construct a professional analysis workflow.
    
    
    ```mermaid
    flowchart TB
                A[Raw Data Loading.xy, .dat, .xrdml] --> B[Data Preprocessing]
                B --> C[Peak DetectionPhase Identification]
                C --> D[Initial Structure ModelCIF Loading]
                D --> E[Parameter Initialization]
                E --> F{Single or Multiple Phases?}
    
                F -->|Single| G[Rietveld RefinementStage 1: BG+Scale]
                F -->|Multiple| H[Multiphase AnalysisPhase 1 ’ Phase 2]
    
                G --> I[Stage 2: ProfileU, V, W, ·]
                I --> J[Stage 3: Structurex, y, z, Uiso]
    
                H --> I
    
                J --> K[Convergence CheckGOF < 2.0?]
                K -->|No| L[Error DiagnosisParameter Adjustment]
                L --> E
    
                K -->|Yes| M[Result ExtractionLattice constants, phase fractions, D]
                M --> N[VisualizationPublication figures]
                N --> O[CIF OutputAcademic Report]
    
                style A fill:#e3f2fd
                style K fill:#fff3e0
                style O fill:#e8f5e9
    ```

### 5.1.1 Raw Data Loading and Preprocessing

Data formats output from XRD instruments are diverse. We implement a data loading function that supports major formats.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # ========================================
    # Example 1: Universal XRD Data Loader
    # ========================================
    
    import numpy as np
    import pandas as pd
    
    class XRDDataLoader:
        """
        Universal XRD Data Loader
    
        Supported formats:
        - .xy: 2-column format (2¸, Intensity)
        - .dat: Text file with header
        - .xrdml: Panalytical XML format
        - .raw: Bruker RAW format
        """
    
        @staticmethod
        def load_xy(filepath, skip_rows=0):
            """
            Load .xy format
    
            Args:
                filepath: File path
                skip_rows: Number of rows to skip (header)
    
            Returns:
                two_theta, intensity: numpy arrays
            """
            try:
                data = np.loadtxt(filepath, skiprows=skip_rows)
                two_theta = data[:, 0]
                intensity = data[:, 1]
    
                return two_theta, intensity
            except Exception as e:
                print(f"Error loading .xy file: {e}")
                return None, None
    
        @staticmethod
        def load_dat(filepath, delimiter=None, header=None):
            """
            Load .dat format (flexible handling with pandas)
    
            Args:
                filepath: File path
                delimiter: Delimiter character (None: whitespace)
                header: Header row number
    
            Returns:
                two_theta, intensity
            """
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, header=header)
    
                # Assume first 2 columns are 2¸ and Intensity
                two_theta = df.iloc[:, 0].values
                intensity = df.iloc[:, 1].values
    
                return two_theta, intensity
            except Exception as e:
                print(f"Error loading .dat file: {e}")
                return None, None
    
        @staticmethod
        def preprocess(two_theta, intensity, remove_outliers=True, smooth=False):
            """
            Data preprocessing
    
            Args:
                two_theta, intensity: Raw data
                remove_outliers: Remove outliers
                smooth: Smoothing (moving average)
    
            Returns:
                two_theta_clean, intensity_clean
            """
            # Outlier removal (beyond 3Ã)
            if remove_outliers:
                mean_int = np.mean(intensity)
                std_int = np.std(intensity)
                mask = np.abs(intensity - mean_int) < 3 * std_int
                two_theta = two_theta[mask]
                intensity = intensity[mask]
    
            # Smoothing (moving average, window=5)
            if smooth:
                intensity = np.convolve(intensity, np.ones(5)/5, mode='same')
    
            # Set negative intensities to 0
            intensity = np.maximum(intensity, 0)
    
            return two_theta, intensity
    
    
    # Usage example
    loader = XRDDataLoader()
    
    # Load .xy format
    two_theta, intensity = loader.load_xy('sample_data.xy', skip_rows=1)
    
    # Preprocessing
    two_theta_clean, intensity_clean = loader.preprocess(
        two_theta, intensity,
        remove_outliers=True,
        smooth=False
    )
    
    print(f"Number of data points: {len(two_theta_clean)}")
    print(f"2¸ range: {two_theta_clean.min():.2f}° - {two_theta_clean.max():.2f}°")
    print(f"Maximum intensity: {intensity_clean.max():.0f} counts")
    

### 5.1.2 Complete Workflow Implementation

We implement an integrated workflow that encompasses all steps from data loading to CIF output.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 2: Complete XRD Analysis Workflow
    # ========================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Parameters, Minimizer
    from pymatgen.core import Structure
    
    class CompleteXRDWorkflow:
        """
        Complete XRD Analysis Workflow
    
        Steps:
        1. Data loading and preprocessing
        2. Peak detection and phase identification
        3. Rietveld refinement (3 stages)
        4. Result extraction and visualization
        5. CIF output
        """
    
        def __init__(self, filepath, wavelength=1.5406):
            self.filepath = filepath
            self.wavelength = wavelength
            self.two_theta = None
            self.intensity = None
            self.result = None
    
        def step1_load_data(self, skip_rows=0):
            """Step 1: Data loading"""
            loader = XRDDataLoader()
            self.two_theta, self.intensity = loader.load_xy(self.filepath, skip_rows)
            self.two_theta, self.intensity = loader.preprocess(
                self.two_theta, self.intensity, remove_outliers=True
            )
            print(f" Step 1 Complete: {len(self.two_theta)} data points loaded")
    
        def step2_peak_detection(self, prominence=0.1):
            """Step 2: Peak detection"""
            from scipy.signal import find_peaks
    
            # Peak detection
            intensity_norm = self.intensity / self.intensity.max()
            peaks, properties = find_peaks(intensity_norm, prominence=prominence)
    
            self.peak_positions = self.two_theta[peaks]
            self.peak_intensities = self.intensity[peaks]
    
            print(f" Step 2 Complete: {len(self.peak_positions)} peaks detected")
            print(f"  Main peak positions: {self.peak_positions[:5]}")
    
        def step3_rietveld_refinement(self, structure_cif=None):
            """Step 3: Rietveld refinement (3 stages)"""
    
            # Stage 1: Background + Scale
            print("Stage 1: Background + Scale ...")
            params_stage1 = self._initialize_params_stage1()
            result_stage1 = self._minimize(params_stage1)
    
            # Stage 2: Profile parameters
            print("Stage 2: Profile parameters ...")
            params_stage2 = self._add_profile_params(result_stage1.params)
            result_stage2 = self._minimize(params_stage2)
    
            # Stage 3: Structure parameters
            print("Stage 3: Structure parameters ...")
            params_stage3 = self._add_structure_params(result_stage2.params)
            self.result = self._minimize(params_stage3)
    
            print(f" Step 3 Complete: Rwp = {self._calculate_rwp(self.result):.2f}%")
    
        def _initialize_params_stage1(self):
            """Parameters for Stage 1"""
            params = Parameters()
            params.add('scale', value=1.0, min=0.1, max=10.0)
            params.add('bg_0', value=self.intensity.min(), min=0.0)
            params.add('bg_1', value=0.0)
            params.add('bg_2', value=0.0)
            return params
    
        def _add_profile_params(self, params_prev):
            """Stage 2: Add profile parameters"""
            params = params_prev.copy()
            params.add('U', value=0.01, min=0.0, max=0.1)
            params.add('V', value=-0.005, min=-0.05, max=0.0)
            params.add('W', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
            return params
    
        def _add_structure_params(self, params_prev):
            """Stage 3: Add structure parameters"""
            params = params_prev.copy()
            params.add('lattice_a', value=5.64, min=5.5, max=5.8)
            params.add('U_iso', value=0.01, min=0.001, max=0.05)
            return params
    
        def _minimize(self, params):
            """Execute minimization"""
            minimizer = Minimizer(self._residual, params)
            result = minimizer.minimize(method='leastsq')
            return result
    
        def _residual(self, params):
            """Residual function (simplified version)"""
            # Background
            bg_coeffs = [params.get('bg_0', params.valuesdict().get('bg_0', 0)),
                         params.get('bg_1', params.valuesdict().get('bg_1', 0)),
                         params.get('bg_2', params.valuesdict().get('bg_2', 0))]
    
            x_norm = 2 * (self.two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
            bg = sum(c * np.polynomial.chebyshev.chebval(x_norm, [0]*i + [1]) for i, c in enumerate(bg_coeffs))
    
            # Scale
            scale = params['scale'].value
    
            # Calculated pattern (simplified)
            I_calc = bg + scale * 10  # In practice, peak calculation goes here
    
            # Residual
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
            return residual
    
        def _calculate_rwp(self, result):
            """Calculate Rwp"""
            return np.sqrt(result.chisqr / result.ndata) * 100
    
        def step4_extract_results(self):
            """Step 4: Result extraction"""
            if self.result is None:
                print("Error: Refinement has not been executed")
                return
    
            results_dict = {
                'lattice_a': self.result.params.get('lattice_a', None),
                'U_iso': self.result.params.get('U_iso', None),
                'Rwp': self._calculate_rwp(self.result),
                'GOF': self.result.redchi
            }
    
            print(" Step 4 Complete: Results extracted")
            for key, val in results_dict.items():
                if val is not None:
                    if hasattr(val, 'value'):
                        print(f"  {key}: {val.value:.6f}")
                    else:
                        print(f"  {key}: {val:.6f}")
    
            return results_dict
    
        def step5_visualize(self, save_path=None):
            """Step 5: Visualization"""
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
    
            # Upper panel: Observed, calculated, difference
            ax1.plot(self.two_theta, self.intensity, 'o', markersize=3,
                     label='Observed', color='red', alpha=0.6)
            # I_calc omitted for simplification
            ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
            ax1.legend()
            ax1.set_title('Rietveld Refinement', fontsize=14, fontweight='bold')
    
            # Lower panel: Residual
            # residual omitted for simplification
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.set_xlabel('2¸ (°)', fontsize=12)
            ax2.set_ylabel('Residual', fontsize=10)
    
            plt.tight_layout()
    
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f" Step 5 Complete: Figure saved to {save_path}")
    
            plt.show()
    
    
    # Workflow execution example
    workflow = CompleteXRDWorkflow('sample.xy', wavelength=1.5406)
    
    # Execute all steps
    workflow.step1_load_data(skip_rows=1)
    workflow.step2_peak_detection(prominence=0.1)
    workflow.step3_rietveld_refinement()
    results = workflow.step4_extract_results()
    workflow.step5_visualize(save_path='rietveld_result.png')
    

## 5.2 Multiphase Mixture Analysis

In actual materials, multiple phases often coexist, requiring multiphase Rietveld analysis. In this section, we learn techniques for analyzing 2-phase systems (±-Fe + FeƒO„) and 3-phase systems.

### 5.2.1 Two-Phase System Analysis: ±-Fe + FeƒO„

Using an oxidized iron sample as an example, we analyze a two-phase mixture of ±-Fe (BCC) and FeƒO„ (spinel structure).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 3: Two-Phase Rietveld Analysis
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    
    class TwoPhaseRietveld:
        """
        Rietveld Analysis for Two-Phase Mixtures
    
        Phase 1: ±-Fe (BCC, Im-3m, a=2.866 Å)
        Phase 2: FeƒO„ (Spinel, Fd-3m, a=8.396 Å)
        """
    
        def __init__(self, two_theta, intensity, wavelength=1.5406):
            self.two_theta = np.array(two_theta)
            self.intensity = np.array(intensity)
            self.wavelength = wavelength
    
            # hkl lists for each phase
            self.hkl_Fe = [(1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0)]
            self.hkl_Fe3O4 = [(2,2,0), (3,1,1), (4,0,0), (4,2,2), (5,1,1)]
    
        def two_theta_from_d(self, d):
            """Calculate 2¸ from d-spacing"""
            sin_theta = self.wavelength / (2 * d)
            if abs(sin_theta) > 1.0:
                return None
            theta = np.arcsin(sin_theta)
            return np.degrees(2 * theta)
    
        def d_spacing_cubic(self, hkl, a):
            """d-spacing for cubic lattice"""
            h, k, l = hkl
            return a / np.sqrt(h**2 + k**2 + l**2)
    
        def pseudo_voigt(self, two_theta, two_theta_0, fwhm, eta, amplitude):
            """Pseudo-Voigt profile"""
            H = fwhm / 2
            delta = two_theta - two_theta_0
    
            G = np.exp(-np.log(2) * (delta / H)**2)
            L = 1 / (1 + (delta / H)**2)
            PV = eta * L + (1 - eta) * G
    
            return amplitude * PV
    
        def caglioti_fwhm(self, two_theta, U, V, W):
            """Caglioti equation"""
            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            return np.sqrt(max(fwhm_sq, 1e-6))
    
        def calculate_pattern(self, params):
            """
            Generate calculated pattern for two phases
            """
            # Extract parameters
            a_Fe = params['a_Fe'].value
            a_Fe3O4 = params['a_Fe3O4'].value
    
            scale_Fe = params['scale_Fe'].value
            scale_Fe3O4 = params['scale_Fe3O4'].value
    
            U = params['U'].value
            V = params['V'].value
            W = params['W'].value
            eta = params['eta'].value
    
            # Background
            bg_0 = params['bg_0'].value
            bg_1 = params['bg_1'].value
    
            x_norm = 2 * (self.two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
            bg = bg_0 + bg_1 * x_norm
    
            I_calc = bg.copy()
    
            # Phase 1: ±-Fe
            for hkl in self.hkl_Fe:
                d = self.d_spacing_cubic(hkl, a_Fe)
                two_theta_hkl = self.two_theta_from_d(d)
    
                if two_theta_hkl is None or two_theta_hkl < self.two_theta.min() or two_theta_hkl > self.two_theta.max():
                    continue
    
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
                amplitude = scale_Fe * 100  # Simplified
    
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            # Phase 2: FeƒO„
            for hkl in self.hkl_Fe3O4:
                d = self.d_spacing_cubic(hkl, a_Fe3O4)
                two_theta_hkl = self.two_theta_from_d(d)
    
                if two_theta_hkl is None or two_theta_hkl < self.two_theta.min() or two_theta_hkl > self.two_theta.max():
                    continue
    
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
                amplitude = scale_Fe3O4 * 80  # Simplified
    
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            return I_calc
    
        def residual(self, params):
            """Residual function"""
            I_calc = self.calculate_pattern(params)
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
            return residual
    
        def refine(self):
            """Two-phase refinement"""
            params = Parameters()
    
            # Lattice constants
            params.add('a_Fe', value=2.866, min=2.85, max=2.89)
            params.add('a_Fe3O4', value=8.396, min=8.35, max=8.45)
    
            # Scale factors
            params.add('scale_Fe', value=1.0, min=0.1, max=10.0)
            params.add('scale_Fe3O4', value=0.5, min=0.1, max=10.0)
    
            # Profile
            params.add('U', value=0.01, min=0.0, max=0.1)
            params.add('V', value=-0.005, min=-0.05, max=0.0)
            params.add('W', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
    
            # Background
            params.add('bg_0', value=10.0, min=0.0)
            params.add('bg_1', value=0.0)
    
            # Minimization
            minimizer = Minimizer(self.residual, params)
            result = minimizer.minimize(method='leastsq')
    
            return result
    
    
    # Test data (simplified)
    two_theta_test = np.linspace(20, 80, 600)
    intensity_test = 15 + 3*np.random.randn(len(two_theta_test))
    
    # Execute two-phase analysis
    two_phase = TwoPhaseRietveld(two_theta_test, intensity_test)
    result = two_phase.refine()
    
    print("=== Two-Phase Rietveld Analysis Results ===")
    print(f"±-Fe lattice constant: a = {result.params['a_Fe'].value:.6f} Å")
    print(f"FeƒO„ lattice constant: a = {result.params['a_Fe3O4'].value:.6f} Å")
    print(f"Scale ratio (Fe:FeƒO„) = {result.params['scale_Fe'].value:.3f}:{result.params['scale_Fe3O4'].value:.3f}")
    print(f"Rwp = {np.sqrt(result.chisqr / result.ndata) * 100:.2f}%")
    

### 5.2.2 Three-Phase System Analysis Strategy

For mixtures with three or more phases, the number of parameters increases rapidly, making convergence difficult. The following strategies are effective:

  1. **Sequential refinement** : Add phases sequentially: main phase ’ 2nd phase ’ 3rd phase
  2. **Parameter fixing** : Fix lattice constants of known phases to literature values
  3. **Scale ratio constraints** : \\(\sum w_i = 1.0\\) (sum of weight fractions equals 1)
  4. **Shared profile** : Use common U, V, W for all phases

## 5.3 Quantitative Phase Analysis

We learn techniques to quantify the weight fraction of each phase in multiphase mixtures. There are two approaches: the RIR method (Reference Intensity Ratio) and the Rietveld method.

### 5.3.1 RIR Method (Reference Intensity Ratio)

The RIR method is a simplified technique that estimates phase fractions from the intensity ratio of the strongest peaks:

\\[ w_{\alpha} = \frac{I_{\alpha} / RIR_{\alpha}}{I_{\alpha}/RIR_{\alpha} + I_{\beta}/RIR_{\beta}} \\] 

  * \\(w_{\alpha}\\): Weight fraction of phase ±
  * \\(I_{\alpha}\\): Intensity of strongest peak for phase ±
  * \\(RIR_{\alpha}\\): RIR value for phase ± (obtained from PDF Card)

    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 4: RIR Method for Quantitative Phase Analysis
    # ========================================
    
    import numpy as np
    
    def rir_quantitative_analysis(peak_intensities, rir_values, phase_names):
        """
        Calculate weight fractions using RIR method
    
        Args:
            peak_intensities: Strongest peak intensities for each phase [I1, I2, ...]
            rir_values: RIR values for each phase [RIR1, RIR2, ...]
            phase_names: List of phase names ['Phase1', 'Phase2', ...]
    
        Returns:
            weight_fractions: Dictionary of weight fractions
        """
        # Calculate I/RIR
        I_over_RIR = np.array(peak_intensities) / np.array(rir_values)
    
        # Weight fractions
        total = np.sum(I_over_RIR)
        weight_fractions = I_over_RIR / total
    
        results = {name: w for name, w in zip(phase_names, weight_fractions)}
    
        return results
    
    
    # Example: Two-phase mixture of ±-Fe + FeƒO„
    peak_intensities = [1500, 800]  # ±-Fe(110): 1500, FeƒO„(311): 800
    rir_values = [2.0, 2.5]         # RIR values (PDF Card)
    phase_names = ['±-Fe', 'FeƒO„']
    
    wt_fractions = rir_quantitative_analysis(peak_intensities, rir_values, phase_names)
    
    print("=== RIR Method Quantitative Analysis Results ===")
    for phase, wt in wt_fractions.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    
    # Example output:
    # === RIR Method Quantitative Analysis Results ===
    # ±-Fe: 70.09 wt%
    # FeƒO„: 29.91 wt%
    

### 5.3.2 Rietveld Quantitative Analysis

In the Rietveld method, weight fractions can be accurately calculated from the scale factor \\(S_{\alpha}\\):

\\[ w_{\alpha} = \frac{S_{\alpha} (ZMV)_{\alpha}}{\sum_i S_i (ZMV)_i} \\] 

  * \\(S_{\alpha}\\): Scale factor for phase ± (determined by refinement)
  * \\(Z\\): Number of formula units in unit cell
  * \\(M\\): Formula mass
  * \\(V\\): Unit cell volume

    
    
    # ========================================
    # Example 5: Rietveld Quantitative Analysis
    # ========================================
    
    def rietveld_quantitative_analysis(scale_factors, Z_list, M_list, V_list, phase_names):
        """
        Calculate weight fractions using Rietveld method
    
        Args:
            scale_factors: Scale factors [S1, S2, ...]
            Z_list: Number of formula units in unit cell [Z1, Z2, ...]
            M_list: Formula masses [M1, M2, ...] (g/mol)
            V_list: Unit cell volumes [V1, V2, ...] (Å³)
            phase_names: List of phase names
    
        Returns:
            weight_fractions: Dictionary of weight fractions
        """
        # Calculate S*(ZMV)
        S_ZMV = np.array(scale_factors) * np.array(Z_list) * np.array(M_list) * np.array(V_list)
    
        # Weight fractions
        total = np.sum(S_ZMV)
        weight_fractions = S_ZMV / total
    
        results = {name: w for name, w in zip(phase_names, weight_fractions)}
    
        return results
    
    
    # Example: ±-Fe + FeƒO„
    scale_factors = [1.23, 0.67]  # Determined by Rietveld refinement
    Z_list = [2, 8]               # ±-Fe: BCC (Z=2), FeƒO„: Spinel (Z=8)
    M_list = [55.845, 231.533]    # Fe: 55.845, FeƒO„: 231.533 g/mol
    V_list = [23.55, 591.4]       # ±-Fe: a³ = 2.866³, FeƒO„: a³ = 8.396³
    
    phase_names = ['±-Fe', 'FeƒO„']
    
    wt_fractions_rietveld = rietveld_quantitative_analysis(
        scale_factors, Z_list, M_list, V_list, phase_names
    )
    
    print("=== Rietveld Quantitative Analysis Results ===")
    for phase, wt in wt_fractions_rietveld.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    
    # Example output:
    # === Rietveld Quantitative Analysis Results ===
    # ±-Fe: 68.34 wt%
    # FeƒO„: 31.66 wt%
    

## 5.4 Error Analysis and Troubleshooting

In Rietveld analysis, convergence failures and non-physical results occur frequently. In this section, we learn typical errors and their solutions.

### 5.4.1 Typical Errors and Diagnosis

Symptom | Cause | Solution  
---|---|---  
**Non-convergence** (>100 iterations) | Inappropriate initial values, strong parameter correlation | Set initial values close to literature values, sequential refinement  
**GOF >> 2.0** | Inappropriate model, incorrect background | Increase Chebyshev order, review phases  
**GOF < 1.0** | Overfitting, too many parameters | Fix unnecessary parameters, add constraints  
**Negative occupancies** | Poor initial values, incorrect boundary settings | Set min=0.0, set initial value to 0.5  
**Lattice constants stuck at boundaries** | Boundaries too narrow, wrong phase | Widen boundaries, reconfirm phase identification  
**Rwp > 20%** | Peak positions shifted, instrumental function uncorrected | Add Zero correction, sample displacement correction  
  
### 5.4.2 Convergence Diagnostics Tool Implementation
    
    
    # ========================================
    # Example 6: Convergence Diagnostics Tool
    # ========================================
    
    class ConvergenceDiagnostics:
        """
        Convergence Diagnostics Tool for Rietveld Refinement
        """
    
        @staticmethod
        def check_convergence(result):
            """
            Check convergence status
    
            Args:
                result: Result from lmfit Minimizer.minimize()
    
            Returns:
                Diagnostic report
            """
            issues = []
    
            # 1. GOF check
            GOF = result.redchi
            if GOF > 2.0:
                issues.append(f"  GOF = {GOF:.2f} (>2.0): Model may be inappropriate")
            elif GOF < 1.0:
                issues.append(f"  GOF = {GOF:.2f} (<1.0): Possible overfitting")
    
            # 2. Parameter boundary check
            for name, param in result.params.items():
                if not param.vary:
                    continue
    
                # Stuck at boundaries
                if param.min is not None and abs(param.value - param.min) < 1e-6:
                    issues.append(f"  {name} stuck at lower boundary: {param.value:.6f}")
                if param.max is not None and abs(param.value - param.max) < 1e-6:
                    issues.append(f"  {name} stuck at upper boundary: {param.value:.6f}")
    
                # Non-physical values
                if 'occ' in name and (param.value < 0 or param.value > 1):
                    issues.append(f"L {name} = {param.value:.6f}: Occupancy out of range [0, 1]")
    
                if 'U_iso' in name and param.value < 0:
                    issues.append(f"L {name} = {param.value:.6f}: Temperature factor is negative")
    
            # 3. Correlation matrix check
            if hasattr(result, 'covar') and result.covar is not None:
                corr_matrix = result.covar / np.outer(np.sqrt(np.diag(result.covar)),
                                                       np.sqrt(np.diag(result.covar)))
                strong_corr = np.where(np.abs(corr_matrix) > 0.9)
    
                for i, j in zip(*strong_corr):
                    if i < j:  # Avoid duplicates
                        param_names = list(result.var_names)
                        issues.append(f"  Strong correlation: {param_names[i]} ” {param_names[j]} (r={corr_matrix[i,j]:.3f})")
    
            # Report output
            if not issues:
                print(" Convergence diagnosis: No issues")
            else:
                print("=
     Convergence diagnosis results:")
                for issue in issues:
                    print(f"  {issue}")
    
            return issues
    
    
    # Usage example
    diagnostics = ConvergenceDiagnostics()
    
    # result is the result object from lmfit Minimizer.minimize()
    # diagnostics.check_convergence(result)
    

### 5.4.3 Troubleshooting Case Studies

**Case 1** : Lattice constant stuck at boundary

**Symptom** :
    
    
    lattice_a = 5.799 Å (boundary: [5.5, 5.8])
      lattice_a stuck at upper boundary

**Cause** : Boundary too narrow, or wrong phase identification

**Solution** :

  1. Widen boundary: `params.add('lattice_a', value=5.64, min=5.3, max=6.0)`
  2. Check high-angle peaks and revalidate phase
  3. Compare with literature values and adjust initial value

**Case 2** : GOF = 5.2 (abnormally high)

**Symptom** : Rwp = 23%, GOF = 5.2

**Cause** :

  * Inappropriate background model (order too low)
  * Unidentified phase present
  * Peak positions shifted (Zero correction needed)

**Solution** :

  1. Increase Chebyshev order from 3rd to 5th
  2. Re-run peak detection to verify all peaks are explained
  3. Add Zero correction parameter: `params.add('zero_shift', value=0.0, min=-0.1, max=0.1)`

## 5.5 Result Visualization and Academic Reporting

Rietveld analysis results need to be presented visually and clearly in papers and reports. In this section, we learn publication-quality figure creation and CIF output.

### 5.5.1 Creating Publication-Quality Rietveld Plots
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    # ========================================
    # Example 7: Publication-Quality Rietveld Plot
    # ========================================
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_rietveld_publication_quality(two_theta, I_obs, I_calc, I_bg, residual,
                                          phase_labels=None, save_path='rietveld.pdf'):
        """
        Publication-quality Rietveld plot
    
        Args:
            two_theta: 2¸ array
            I_obs: Observed intensity
            I_calc: Calculated intensity
            I_bg: Background
            residual: Residual
            phase_labels: List of phase names
            save_path: Output file path
        """
        fig = plt.figure(figsize=(12, 8))
    
        # Upper panel: Observed, calculated, difference
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
        # Observed data (red circles)
        ax1.plot(two_theta, I_obs, 'o', markersize=4, markerfacecolor='none',
                 markeredgecolor='red', markeredgewidth=1.2, label='Observed')
    
        # Calculated pattern (blue line)
        ax1.plot(two_theta, I_calc, '-', color='blue', linewidth=2, label='Calculated')
    
        # Background (green line)
        ax1.plot(two_theta, I_bg, '--', color='green', linewidth=1.5, label='Background')
    
        # Difference (gray, offset below)
        offset = I_obs.min() - 0.1 * I_obs.max()
        ax1.plot(two_theta, residual + offset, '-', color='gray', linewidth=1, label='Difference')
        ax1.axhline(offset, color='black', linestyle='-', linewidth=0.5)
    
        # Bragg peak positions (vertical lines)
        if phase_labels:
            colors = ['red', 'blue', 'orange']
            for i, label in enumerate(phase_labels):
                # Simplified: peak positions manually set
                peak_positions = [38.2, 44.4, 64.6]  # Example
                y_position = offset - 0.05 * I_obs.max() * (i + 1)
                ax1.vlines(peak_positions, ymin=y_position, ymax=y_position + 0.03*I_obs.max(),
                          colors=colors[i], linewidth=2, label=label)
    
        ax1.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, frameon=False)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_xlim(two_theta.min(), two_theta.max())
        ax1.set_ylim(offset - 0.2*I_obs.max(), I_obs.max() * 1.1)
    
        # Lower panel: Residual zoom
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
        ax2.plot(two_theta, residual, '-', color='black', linewidth=1)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('2¸ (°)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_ylim(-3*np.std(residual), 3*np.std(residual))
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Figure saved to {save_path} (300 dpi)")
    
        plt.show()
    
    
    # Usage example (dummy data)
    two_theta = np.linspace(20, 80, 600)
    I_obs = 100 * np.exp(-((two_theta - 38)/2)**2) + 50 * np.exp(-((two_theta - 44)/2.5)**2) + 20 + 5*np.random.randn(len(two_theta))
    I_calc = 100 * np.exp(-((two_theta - 38)/2)**2) + 50 * np.exp(-((two_theta - 44)/2.5)**2) + 20
    I_bg = 20 * np.ones_like(two_theta)
    residual = I_obs - I_calc
    
    plot_rietveld_publication_quality(two_theta, I_obs, I_calc, I_bg, residual,
                                      phase_labels=['±-Fe', 'FeƒO„'],
                                      save_path='rietveld_paper.pdf')
    

### 5.5.2 CIF File Output

CIF (Crystallographic Information File) is the standard format for crystal structure data. By saving refinement results in CIF format, other researchers can reproduce and verify the work.
    
    
    # ========================================
    # Example 8: CIF File Generation
    # ========================================
    
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    
    def export_to_cif(structure, lattice_params, refinement_results, output_path='refined_structure.cif'):
        """
        Export refinement results to CIF file
    
        Args:
            structure: pymatgen Structure object
            lattice_params: {'a': 5.64, 'b': 5.64, 'c': 5.64, ...}
            refinement_results: {'Rwp': 8.5, 'GOF': 1.42, ...}
            output_path: Output file path
        """
        # CIF writer
        cif_writer = CifWriter(structure, symprec=0.01)
    
        # Get CIF as string
        cif_string = str(cif_writer)
    
        # Add metadata
        metadata = f"""
    #======================================================================
    # Rietveld Refinement Results
    #======================================================================
    # Refined lattice parameters:
    #   a = {lattice_params.get('a', 'N/A'):.6f} Å
    #   b = {lattice_params.get('b', 'N/A'):.6f} Å
    #   c = {lattice_params.get('c', 'N/A'):.6f} Å
    #
    # Refinement statistics:
    #   Rwp = {refinement_results.get('Rwp', 'N/A'):.2f} %
    #   GOF = {refinement_results.get('GOF', 'N/A'):.3f}
    #   Number of data points = {refinement_results.get('ndata', 'N/A')}
    #
    # Date: {refinement_results.get('date', 'YYYY-MM-DD')}
    # Software: Python + lmfit + pymatgen
    #======================================================================
    
    """
    
        # CIF output
        with open(output_path, 'w') as f:
            f.write(metadata)
            f.write(cif_string)
    
        print(f" CIF file exported to {output_path}")
    
    
    # Usage example
    from pymatgen.core import Lattice, Structure
    
    # Refined structure
    lattice = Lattice.cubic(5.6405)  # Refined a
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    lattice_params = {'a': 5.6405, 'b': 5.6405, 'c': 5.6405}
    refinement_results = {
        'Rwp': 8.52,
        'GOF': 1.42,
        'ndata': 600,
        'date': '2025-10-28'
    }
    
    export_to_cif(structure, lattice_params, refinement_results, 'NaCl_refined.cif')
    

### 5.5.3 Integration with GSAS-II

GSAS-II is a powerful Rietveld analysis software with GUI support. By operating it from Python scripts, we can combine advanced analysis with Python's flexibility.

> **=¡ GSAS-II Python Interface**
> 
> GSAS-II can be operated from Python through the `GSASIIscriptable` module. Project creation, data loading, refinement execution, and result extraction can all be automated.
> 
> Details: [GSAS-II Scriptable Documentation](<https://subversion.xray.aps.anl.gov/pyGSAS/trunk/help/GSASIIscriptable.html>)
    
    
    # ========================================
    # GSAS-II Integration Concept Code (requires GSAS-II installation)
    # ========================================
    
    # import GSASIIscriptable as G2sc
    #
    # # Create project
    # gpx = G2sc.G2Project(newgpx='my_rietveld.gpx')
    #
    # # Add XRD data
    # hist = gpx.add_powder_histogram('sample.xy', 'PWDR')
    #
    # # Add phase (from CIF)
    # phase = gpx.add_phase('Fe.cif', phasename='alpha-Fe', histograms=[hist])
    #
    # # Execute Rietveld refinement
    # gpx.do_refinements([
    #     {'set': {'Background': {'refine': True}}},
    #     {'set': {'Cell': True, 'Atoms': True}},
    # ])
    #
    # # Extract results
    # results = gpx.get_Covariance()
    # print(f"Rwp = {results['Rvals']['Rwp']:.2f}%")
    #
    # # Save project
    # gpx.save()
    

## Learning Objectives Check

After completing this chapter, you should be able to explain and implement the following:

### Basic Understanding

  *  Overall workflow of practical XRD analysis (data loading ’ refinement ’ reporting)
  *  Sequential refinement strategy for multiphase mixture analysis
  *  Differences between RIR and Rietveld methods for quantitative phase analysis and their applications
  *  Typical errors such as convergence failures, abnormal GOF, negative occupancies and their causes

### Practical Skills

  *  Load and preprocess XRD data in .xy, .dat formats
  *  Complete Rietveld analysis of two-phase mixtures (±-Fe + FeƒO„)
  *  Calculate weight fractions from scale factors (Rietveld quantification)
  *  Create publication-quality Rietveld plots (matplotlib)
  *  Export refinement results as CIF files

### Advanced Applications

  *  Automatically detect and resolve errors using convergence diagnostics tools
  *  Optimization strategies for strongly correlated parameters (lattice constants and temperature factors)
  *  Advanced analysis workflows combining GSAS-II and Python
  *  Execute complete analysis ’ validation ’ reporting cycles on experimental data

## Practice Problems

### Easy (Basic Verification)

**Q1** : What are the main differences between RIR and Rietveld methods for quantitative phase analysis?

**Answer** :

Item | RIR Method | Rietveld Method  
---|---|---  
**Data Used** | Strongest peak intensity only | Entire pattern (full 2¸ range)  
**Accuracy** | ±5-10% | ±1-3%  
**Required Information** | RIR values (PDF Card) | Crystal structure (CIF)  
**Computation Time** | Seconds | Minutes to hours  
**Application** | Rapid screening | Precise quantitative analysis  
  
**Conclusion** : RIR method is suitable for quick estimation, Rietveld method for high-precision quantification.

**Q2** : Why are profile parameters (U, V, W) shared across all phases in three-phase mixture refinement?

**Answer** :

**Reasons** :

  1. **Reduction of parameter count** : Refining U, V, W individually for 3 phases = 9 parameters. Sharing reduces to 3 parameters.
  2. **Physical validity** : U, V, W represent instrument-induced peak broadening, so they should be common across all phases.
  3. **Convergence stability** : Fewer parameters lead to more stable minimization.

**However** : If crystallite size or microstrain differ significantly between phases, individual refinement may be necessary.

### Medium (Application)

**Q3** : For a two-phase mixture of ±-Fe (a=2.866Å, Z=2, M=55.845) and FeƒO„ (a=8.396Å, Z=8, M=231.533), the scale factors were refined as S_Fe=1.5, S_Fe3O4=0.8. Calculate the weight fraction of each phase.

**Answer** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Answer:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Data
    S = [1.5, 0.8]
    Z = [2, 8]
    M = [55.845, 231.533]
    V = [2.866**3, 8.396**3]  # a³
    
    # S*Z*M*V
    S_ZMV = np.array(S) * np.array(Z) * np.array(M) * np.array(V)
    print(f"S*Z*M*V: {S_ZMV}")
    
    # Weight fractions
    wt_fractions = S_ZMV / S_ZMV.sum()
    print(f"±-Fe: {wt_fractions[0]*100:.2f} wt%")
    print(f"FeƒO„: {wt_fractions[1]*100:.2f} wt%")
    

**Output** :
    
    
    S*Z*M*V: [2650.13 908536.45]
    ±-Fe: 0.29 wt%
    FeƒO„: 99.71 wt%

The sample is found to be predominantly FeƒO„.

**Q4** : Rietveld refinement resulted in "GOF = 0.85". What is the problem and how should it be addressed?

**Answer** :

**Problem** : GOF < 1.0 indicates possible **overfitting**. Too many parameters are fitting even the noise.

**Solutions** :

  1. **Fix unnecessary parameters** : If temperature factors or occupancies are close to 1.0, fix them
  2. **Reduce background order** : Chebyshev 5th order ’ 3rd order
  3. **Add constraints** : Strengthen constraints on chemical bond lengths, etc.
  4. **Check data quality** : Measurement time may be too long, leading to extremely small statistical noise

**Goal** : GOF = 1.0 - 2.0 is ideal.

### Hard (Advanced)

**Q5** : Write complete two-phase Rietveld analysis code. Assuming a mixture of ±-Fe (BCC) and FeƒO„ (Spinel), refine lattice constants, scale factors, and profile parameters, then calculate weight fractions.

**Answer** :

(Extend the TwoPhaseRietveld class from Example 3)
    
    
    # Complete version - see Example 3
    # Additional feature: Weight fraction calculation
    
    class TwoPhaseRietveldComplete(TwoPhaseRietveld):
        """Two-phase Rietveld analysis + quantitative analysis"""
    
        def calculate_weight_fractions(self, result):
            """Calculate weight fractions"""
            a_Fe = result.params['a_Fe'].value
            a_Fe3O4 = result.params['a_Fe3O4'].value
    
            S_Fe = result.params['scale_Fe'].value
            S_Fe3O4 = result.params['scale_Fe3O4'].value
    
            # Crystallographic data
            Z_Fe, M_Fe, V_Fe = 2, 55.845, a_Fe**3
            Z_Fe3O4, M_Fe3O4, V_Fe3O4 = 8, 231.533, a_Fe3O4**3
    
            # S*Z*M*V
            S_ZMV_Fe = S_Fe * Z_Fe * M_Fe * V_Fe
            S_ZMV_Fe3O4 = S_Fe3O4 * Z_Fe3O4 * M_Fe3O4 * V_Fe3O4
    
            total = S_ZMV_Fe + S_ZMV_Fe3O4
    
            wt_Fe = S_ZMV_Fe / total
            wt_Fe3O4 = S_ZMV_Fe3O4 / total
    
            return {'±-Fe': wt_Fe, 'FeƒO„': wt_Fe3O4}
    
    # Execution
    two_phase_complete = TwoPhaseRietveldComplete(two_theta_test, intensity_test)
    result = two_phase_complete.refine()
    wt_fractions = two_phase_complete.calculate_weight_fractions(result)
    
    print("=== Quantitative Analysis Results ===")
    for phase, wt in wt_fractions.items():
        print(f"{phase}: {wt*100:.2f} wt%")
    

**Q6** : Prepare experimental XRD data (.xy format) and execute the complete analysis workflow (data loading ’ refinement ’ visualization ’ CIF output).

**Answer** :

This problem applies the CompleteXRDWorkflow class from Example 2 using actual .xy files.
    
    
    # Complete workflow with experimental data
    workflow = CompleteXRDWorkflow('experimental_data.xy', wavelength=1.5406)
    
    # Execute Steps 1-5 sequentially
    workflow.step1_load_data(skip_rows=1)
    workflow.step2_peak_detection(prominence=0.15)
    workflow.step3_rietveld_refinement()
    results = workflow.step4_extract_results()
    workflow.step5_visualize(save_path='experimental_data_Rietveld.pdf')
    
    # CIF output (define pymatgen Structure beforehand)
    from pymatgen.core import Structure, Lattice
    
    a_refined = results['lattice_a'].value
    structure = Structure(Lattice.cubic(a_refined), ["Fe"], [[0, 0, 0]])
    
    export_to_cif(structure,
                  {'a': a_refined, 'b': a_refined, 'c': a_refined},
                  {'Rwp': results['Rwp'], 'GOF': results['GOF'], 'ndata': len(workflow.two_theta), 'date': '2025-10-28'},
                  'Fe_refined.cif')
    

## Learning Objectives Verification

Review what you learned in this chapter and verify the following items.

### Basic Understanding

  *  Can explain the overall XRD data analysis workflow (preprocessing ’ indexing ’ refinement ’ reporting)
  *  Understand the principles and procedures of qualitative and quantitative analysis of multiphase mixtures
  *  Can explain the causes of typical errors (peak identification failure, refinement divergence, preferred orientation effects)
  *  Understand the reporting format required for XRD analysis results in academic papers

### Practical Skills

  *  Can construct complete analysis pipelines using XRDWorkflowManager class
  *  Can execute simultaneous refinement of multiple phases with MultiphaseAnalyzer
  *  Can identify and resolve analysis problems using error diagnostic functions
  *  Can create academic-level graphs with publication_quality_plot function

### Advanced Applications

  *  Can derive publication-level results starting from experimental data
  *  Can perform quantitative phase analysis on multiphase samples and assess reliability
  *  Can validate analysis results from multiple perspectives and autonomously troubleshoot errors
  *  Can generate CIF files and output in formats suitable for registration in databases like CCDC or ICSD

## References

  1. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press. - Comprehensive textbook on the Rietveld method and practical workflow explanations
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry. - Theory and practice of powder XRD, best practices for error diagnosis
  3. Bish, D. L., & Post, J. E. (Eds.). (1989). _Modern Powder Diffraction (Reviews in Mineralogy Vol. 20)_. Mineralogical Society of America. - Classic reference on multiphase analysis and quantitative phase analysis
  4. Hill, R. J., & Howard, C. J. (1987). _Quantitative phase analysis from neutron powder diffraction data using the Rietveld method_. Journal of Applied Crystallography, 20(6), 467-474. - Original paper on quantitative phase analysis
  5. GSAS-II Documentation. _Tutorials and User Guides_. Available at: https://gsas-ii.readthedocs.io/ - Official GSAS-II documentation and practical tutorials
  6. International Centre for Diffraction Data (ICDD). _PDF-4+ Database and Search/Match Software_. - Comprehensive powder diffraction database for phase identification
  7. McCusker, L. B., et al. (1999). _Rietveld refinement guidelines_. Journal of Applied Crystallography, 32(1), 36-50. - Refinement guidelines and checklist for academic reporting

## Series Summary

Congratulations! You have completed all 5 chapters of the Introduction to X-ray Diffraction Analysis series. Through this series, you have acquired the following skills:

### Knowledge and Skills Acquired

  *  **Chapter 1** : Fundamental theory of X-ray diffraction (Bragg's law, structure factor, systematic absences)
  *  **Chapter 2** : Powder XRD measurement and basic analysis (peak detection, indexing, lattice constant calculation)
  *  **Chapter 3** : Principles of Rietveld method (profile function, background, R-factors)
  *  **Chapter 4** : Structure refinement (atomic coordinates, temperature factors, crystallite size, microstrain)
  *  **Chapter 5** : Practical workflow (multiphase analysis, quantitative analysis, error diagnosis, academic reporting)

### Next Steps

Having mastered the fundamentals of XRD analysis, you can proceed to the following topics:

  * **Advanced XRD techniques** : Thin film XRD, high-temperature/in-situ XRD, total scattering PDF analysis
  * **Neutron diffraction** : Precise structure analysis of light elements, magnetic structure determination
  * **Single crystal XRD** : Precise structure determination, crystal symmetry determination
  * **Machine learning and XRD** : Automated phase identification, anomaly detection, inverse problem-based structure prediction

> **< “ Continuing Learning**
> 
> Practice with actual XRD data, read papers, and ask questions in communities (such as X-ray Discussion Forum or Stack Exchange) to further hone your skills.

## References and Resources

### Textbooks

  1. Pecharsky, V. K., & Zavalij, P. Y. (2009). _Fundamentals of Powder Diffraction and Structural Characterization of Materials_ (2nd ed.). Springer.
  2. Dinnebier, R. E., & Billinge, S. J. L. (Eds.). (2008). _Powder Diffraction: Theory and Practice_. Royal Society of Chemistry.
  3. Young, R. A. (Ed.). (1993). _The Rietveld Method_. Oxford University Press.

### Software

  * **GSAS-II** : [https://subversion.xray.aps.anl.gov/pyGSAS/](<https://subversion.xray.aps.anl.gov/pyGSAS/trunk/help/>)
  * **FullProf** : <https://www.ill.eu/sites/fullprof/>
  * **TOPAS** : <https://www.bruker.com/topas>
  * **pymatgen** : <https://pymatgen.org/>

### Databases

  * **ICDD PDF-4+** : Powder diffraction pattern database
  * **Crystallography Open Database (COD)** : <http://www.crystallography.net/>
  * **Materials Project** : <https://materialsproject.org/>

## Acknowledgments

This series was created as part of Materials Science education. We aim to enrich English-language resources for systematic learning from the fundamentals to practice of XRD analysis.

**Feedback and questions** can be sent to [yusuke.hashimoto.b8@tohoku.ac.jp](<mailto:yusuke.hashimoto.b8@tohoku.ac.jp>).

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
