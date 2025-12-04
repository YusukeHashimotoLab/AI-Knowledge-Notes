---
title: "Chapter 3: UV-Vis Spectroscopy"
chapter_title: "Chapter 3: UV-Vis Spectroscopy"
---

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Spectroscopy](<../../MS/spectroscopy-introduction/index.html>):Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/MS/spectroscopy-introduction/chapter-3.html>) | Last sync: 2025-11-16

# Chapter 3: UV-Vis Spectroscopy

**What you will learn in this chapter:** Ultraviolet-visible (UV-Vis) spectroscopy is a spectroscopic method that observes electronic transitions in materials, and is an indispensable technique for optical property evaluation, bandgap measurement, and electronic state analysis of coordination compounds in materials science. In this chapter, you will systematically learn UV-Vis spectroscopy from fundamentals to practice, including the theoretical foundations of electronic transitions, practical applications of the Lambert-Beer law, bandgap determination using Tauc plot method, and interpretation of transition metal complexes using ligand field theory.

## 3.1 Electronic Transitions and Principles of UV-Vis Spectroscopy

### 3.1.1 Energy of Electronic Transitions

In UV-Vis spectroscopy, light absorption in the ultraviolet (10-400 nm) to visible (400-800 nm) region is measured. This wavelength range corresponds to molecular electronic transition energies (approximately 1.5-6 eV).

**Relationship between electronic transition energy and wavelength:**

\\[ E = h\nu = \frac{hc}{\lambda} \\] 

Where \\( h = 6.626 \times 10^{-34} \\) J·s (Planck's constant), \\( c = 3.0 \times 10^8 \\) m/s (speed of light), and \\( \lambda \\) is wavelength.

Conversion formula between wavelength (nm) and energy (eV):

\\[ E\,(\text{eV}) = \frac{1239.8}{\lambda\,(\text{nm})} \\] 

### 3.1.2 Types of Electronic Transitions

#### Major Electronic Transitions

  * **Ã ’ Ã* transition:** Transition from Ã orbital of single bond to antibonding Ã* orbital (far UV region, » < 200 nm)
  * **n ’ Ã* transition:** Transition from non-bonding electron pair to antibonding Ã* orbital (150-250 nm)
  * **À ’ À* transition:** Transition from À orbital of double bond to À* orbital (200-400 nm, main region of UV-Vis)
  * **n ’ À* transition:** Transition from non-bonding electron pair to À* orbital (250-350 nm, weak absorption)
  * **d ’ d transition:** Transition between d orbitals in transition metal complexes (visible region, explained by ligand field theory)
  * **Charge transfer transition (CT):** Charge transfer from metal to ligand (MLCT) or ligand to metal (LMCT) (strong absorption)

### 3.1.3 HOMO-LUMO Transition and Bandgap

In organic molecules and semiconductor materials, the lowest energy electronic transition is from the highest occupied molecular orbital (HOMO) to the lowest unoccupied molecular orbital (LUMO). This transition energy corresponds to the bandgap \\( E_g \\) of semiconductors.

**Relationship between bandgap and UV-Vis absorption edge:**

\\[ E_g = h\nu_{\text{onset}} = \frac{1239.8}{\lambda_{\text{onset}}\,(\text{nm})} \\] 

Where \\( \lambda_{\text{onset}} \\) is the absorption onset wavelength.
    
    
    ```mermaid
    flowchart TD
            A[Ground StateHOMO Electron Configuration] -->|Light Absorption h½| B[Excited StateLUMO Electron Configuration]
            B -->|Fluorescence| C[Ground StateEnergy Release]
            B -->|Non-radiative Decay| D[Ground StateThermal Energy]
    
            style A fill:#e3f2fd
            style B fill:#fff3e0
            style C fill:#e8f5e9
            style D fill:#fce4ec
        
    
        3.2 Theory and Applications of the Lambert-Beer Law
    3.2.1 Mathematical Expression of the Lambert-Beer Law
    The Lambert-Beer law is a fundamental law that describes the relationship between light absorption and concentration in solutions. We reconsider the equation introduced in Chapter 1 in the context of UV-Vis spectroscopy.
    
    Definition of absorbance:
            \[
            A = \log_{10}\left(\frac{I_0}{I}\right) = \epsilon c l
            \]
            Where \( A \) is absorbance, \( I_0 \) is incident light intensity, \( I \) is transmitted light intensity, \( \epsilon \) is molar absorptivity (L mol-1 cm-1), \( c \) is concentration (mol/L), and \( l \) is path length (cm).
    Relationship with transmittance:
            \[
            T = \frac{I}{I_0} = 10^{-A}
            \]
            \[
            A = -\log_{10} T = 2 - \log_{10}(\%T)
            \]
        
    3.2.2 Physical Meaning of Molar Absorptivity
    Molar absorptivity \( \epsilon \) is an intrinsic property value that represents the light absorption ability of a substance at a specific wavelength. A large \( \epsilon \) value (\( \epsilon > 10^4 \) L mol-1 cm-1) indicates an allowed transition, while a small \( \epsilon \) value (\( \epsilon < 10^3 \)) indicates a forbidden transition.
    
    
    
    Transition Type
    Molar Absorptivity µ (L mol-1 cm-1)
    Examples
    
    
    
    
    À ’ À* (conjugated system)
    10,000 - 100,000
    Benzene, Anthracene
    
    
    n ’ À*
    10 - 1,000
    Carbonyl compounds
    
    
    d ’ d (transition metal)
    1 - 100
    Cu2+, Ni2+ complexes
    
    
    Charge transfer (CT)
    1,000 - 50,000
    MnO4-, Fe-phenanthroline
    
    
    
    3.2.3 Quantitative Analysis Using Calibration Curves
    Using the linearity of the Lambert-Beer law, the concentration of unknown samples can be determined. A series of standard solutions of known concentrations are measured to create a calibration curve of absorbance \( A \) vs. concentration \( c \).
    Code Example 1: Creating Calibration Curves and Quantitative Analysis Using the Lambert-Beer Law
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    def create_calibration_curve(concentrations, absorbances):
        """
        Create calibration curve and return linear regression parameters
    
        Parameters:
        -----------
        concentrations : array-like
            Concentrations of standard solutions (mol/L)
        absorbances : array-like
            Absorbance at each concentration
    
        Returns:
        --------
        slope : float
            Slope of calibration curve (molar absorptivity × path length)
        intercept : float
            Intercept (should be zero)
        r_value : float
            Correlation coefficient
        """
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(concentrations, absorbances)
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(concentrations, absorbances, s=100, alpha=0.7, label='Measured Data')
    
        # Regression line
        conc_fit = np.linspace(0, max(concentrations)*1.1, 100)
        abs_fit = slope * conc_fit + intercept
        plt.plot(conc_fit, abs_fit, 'r--', linewidth=2,
                 label=f'Regression Line: A = {slope:.3f}c + {intercept:.4f}\nR² = {r_value**2:.4f}')
    
        plt.xlabel('Concentration (mol/L)', fontsize=12)
        plt.ylabel('Absorbance', fontsize=12)
        plt.title('Calibration Curve Using Lambert-Beer Law', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        return slope, intercept, r_value
    
    def determine_concentration(absorbance_sample, slope, intercept):
        """
        Determine concentration of unknown sample from calibration curve
    
        Parameters:
        -----------
        absorbance_sample : float
            Absorbance of unknown sample
        slope : float
            Slope of calibration curve
        intercept : float
            Intercept of calibration curve
    
        Returns:
        --------
        concentration : float
            Concentration of unknown sample (mol/L)
        """
        concentration = (absorbance_sample - intercept) / slope
        return concentration
    
    # Execution example: Quantitative analysis of methylene blue
    concentrations = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-5  # mol/L
    absorbances = np.array([0.12, 0.24, 0.48, 0.72, 0.96, 1.20])
    
    slope, intercept, r_value = create_calibration_curve(concentrations, absorbances)
    
    # Calculate molar absorptivity for path length of 1 cm
    epsilon = slope  # L mol^-1 cm^-1
    print(f"Molar absorptivity µ = {epsilon:.2e} L mol{¹ cm{¹")
    
    # Determine concentration of unknown sample
    A_unknown = 0.60
    c_unknown = determine_concentration(A_unknown, slope, intercept)
    print(f"Concentration of unknown sample: {c_unknown:.2e} mol/L")
    print(f"Correlation coefficient of calibration curve R² = {r_value**2:.4f}")
    
    3.3 Bandgap Measurement Using Tauc Plot Method
    3.3.1 Theoretical Background of Tauc's Law
    To precisely determine the bandgap of semiconductor and insulator materials, the analytical method proposed by Jan Tauc (1968) is widely used. Tauc's law describes the relationship between absorption coefficient \( \alpha \) and photon energy \( h\nu \).
    
    Tauc's law (direct transition):
            \[
            (\alpha h\nu)^2 = B(h\nu - E_g)
            \]
            Where \( B \) is a material constant and \( E_g \) is the bandgap.
    Tauc's law (indirect transition):
            \[
            (\alpha h\nu)^{1/2} = B(h\nu - E_g)
            \]
    
            Calculation of absorption coefficient:
            \[
            \alpha = \frac{2.303 \cdot A}{l}
            \]
            Where \( A \) is absorbance and \( l \) is sample thickness (cm).
    
    3.3.2 Procedure for Creating Tauc Plots
    
    Measure UV-Vis absorption spectrum and obtain wavelength \( \lambda \) and absorbance \( A \)
    Convert wavelength to photon energy \( h\nu = 1239.8/\lambda \) (eV)
    Calculate absorption coefficient from absorbance \( \alpha = 2.303 \cdot A/l \)
    For direct transitions: Plot \( (\alpha h\nu)^2 \) vs. \( h\nu \)
    For indirect transitions: Plot \( (\alpha h\nu)^{1/2} \) vs. \( h\nu \)
    Extrapolate the linear region of the absorption edge and determine bandgap \( E_g \) from the intersection with the horizontal axis
    
    Code Example 2: Bandgap Measurement Using Tauc Plot Method
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def tauc_plot_direct(wavelength, absorbance, thickness_cm, plot_range=(2.0, 4.0)):
        """
        Create Tauc plot for direct transition material and determine bandgap
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        absorbance : array
            Absorbance
        thickness_cm : float
            Sample thickness (cm)
        plot_range : tuple
            Energy range for fitting (eV)
    
        Returns:
        --------
        Eg : float
            Bandgap (eV)
        """
        # Convert wavelength to photon energy
        photon_energy = 1239.8 / wavelength  # eV
    
        # Calculate absorption coefficient
        alpha = 2.303 * absorbance / thickness_cm  # cm^-1
    
        # Tauc plot: (±h½)^2
        tauc_y = (alpha * photon_energy)**2
    
        # Select fitting range
        mask = (photon_energy >= plot_range[0]) & (photon_energy <= plot_range[1])
        E_fit = photon_energy[mask]
        tauc_fit = tauc_y[mask]
    
        # Linear fitting
        def linear(x, B, Eg):
            return B * (x - Eg)
    
        popt, pcov = curve_fit(linear, E_fit, tauc_fit, p0=[1e10, 3.0])
        B, Eg = popt
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(photon_energy, tauc_y, 'o-', label='Measured Data', alpha=0.7)
    
        # Fitting line
        E_extended = np.linspace(Eg - 0.5, plot_range[1], 100)
        tauc_extended = linear(E_extended, B, Eg)
        plt.plot(E_extended, tauc_extended, 'r--', linewidth=2,
                 label=f'Linear Fit\nEg = {Eg:.3f} eV')
    
        # Highlight bandgap position
        plt.axvline(Eg, color='green', linestyle=':', linewidth=2, label=f'Bandgap: {Eg:.3f} eV')
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
        plt.xlabel('Photon Energy (eV)', fontsize=12)
        plt.ylabel('(±h½)² (eV² cm{²)', fontsize=12)
        plt.title('Tauc Plot (Direct Transition)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(photon_energy.min(), photon_energy.max())
        plt.tight_layout()
        plt.show()
    
        print(f"Determined bandgap: Eg = {Eg:.3f} eV")
        print(f"Corresponding wavelength: » = {1239.8/Eg:.1f} nm")
    
        return Eg
    
    def tauc_plot_indirect(wavelength, absorbance, thickness_cm, plot_range=(1.5, 3.0)):
        """
        Create Tauc plot for indirect transition material and determine bandgap
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        absorbance : array
            Absorbance
        thickness_cm : float
            Sample thickness (cm)
        plot_range : tuple
            Energy range for fitting (eV)
    
        Returns:
        --------
        Eg : float
            Bandgap (eV)
        """
        # Convert wavelength to photon energy
        photon_energy = 1239.8 / wavelength  # eV
    
        # Calculate absorption coefficient
        alpha = 2.303 * absorbance / thickness_cm  # cm^-1
    
        # Tauc plot: (±h½)^(1/2)
        tauc_y = np.sqrt(alpha * photon_energy)
    
        # Select fitting range
        mask = (photon_energy >= plot_range[0]) & (photon_energy <= plot_range[1])
        E_fit = photon_energy[mask]
        tauc_fit = tauc_y[mask]
    
        # Linear fitting
        def linear(x, B, Eg):
            return B * (x - Eg)
    
        popt, pcov = curve_fit(linear, E_fit, tauc_fit, p0=[100, 2.0])
        B, Eg = popt
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(photon_energy, tauc_y, 'o-', label='Measured Data', alpha=0.7)
    
        # Fitting line
        E_extended = np.linspace(Eg - 0.3, plot_range[1], 100)
        tauc_extended = linear(E_extended, B, Eg)
        plt.plot(E_extended, tauc_extended, 'r--', linewidth=2,
                 label=f'Linear Fit\nEg = {Eg:.3f} eV')
    
        # Highlight bandgap position
        plt.axvline(Eg, color='green', linestyle=':', linewidth=2, label=f'Bandgap: {Eg:.3f} eV')
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
        plt.xlabel('Photon Energy (eV)', fontsize=12)
        plt.ylabel('(±h½)^(1/2) (eV^(1/2) cm^(-1/2))', fontsize=12)
        plt.title('Tauc Plot (Indirect Transition)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(photon_energy.min(), photon_energy.max())
        plt.tight_layout()
        plt.show()
    
        print(f"Determined bandgap: Eg = {Eg:.3f} eV")
        print(f"Corresponding wavelength: » = {1239.8/Eg:.1f} nm")
    
        return Eg
    
    # Execution example: TiO‚ nanoparticles (direct transition, Eg H 3.2 eV)
    wavelength_nm = np.linspace(300, 500, 200)
    # Simulation data (in practice obtained from spectrophotometer)
    Eg_true = 3.2  # eV
    alpha_true = 1e4 * np.maximum(0, (1239.8/wavelength_nm - Eg_true))**2
    absorbance_sim = alpha_true * 0.01 / 2.303  # Sample thickness 0.01 cm
    
    Eg_measured = tauc_plot_direct(wavelength_nm, absorbance_sim, thickness_cm=0.01)
    
    3.4 Ligand Field Theory and d-d Transitions
    3.4.1 Ligand Field Splitting Energy
    Transition metal complexes (Cu2+, Ni2+, Co2+, etc.) exhibit characteristic colors in the visible region. This is because d orbitals are split by the electrostatic field of ligands, and d-d transitions cause absorption of visible light.
    
    d-Orbital Splitting in Octahedral Ligand Field (Oh Symmetry)
    
    eg orbitals (high energy): dz², dx²-y² (directly oppose ligands, large repulsion)
    t2g orbitals (low energy): dxy, dxz, dyz (small repulsion with ligands)
    
    The splitting energy \( \Delta_o \) (10Dq) is directly related to the color of the complex:
            \[
            \Delta_o = h\nu = \frac{hc}{\lambda}
            \]
        
    
        flowchart TB
            A[Free Ion5 Degenerate d Orbitals] -->|Octahedral Ligand Field| B[e_g OrbitalsHigh Energy]
            A -->|Octahedral Ligand Field| C[t_2g OrbitalsLow Energy]
    
            B -.->|d-d TransitionLight Absorption| C
    
            D[d Electron ConfigurationGround State] -->|Visible Light Absorption| E[d Electron ConfigurationExcited State]
    
            style A fill:#e3f2fd
            style B fill:#ffebee
            style C fill:#e8f5e9
            style D fill:#fff3e0
            style E fill:#fce4ec
        
    
        3.4.2 Spectrochemical Series
    The magnitude of splitting energy \( \Delta_o \) varies depending on the type of ligand. This is called the spectrochemical series:
    
    Spectrochemical series (ligand strength order):
            \[
            \text{I}^- < \text{Br}^- < \text{Cl}^- < \text{F}^- < \text{OH}^- < \text{H}_2\text{O} < \text{NH}_3 < \text{en} < \text{NO}_2^- < \text{CN}^- < \text{CO}
            \]
            Weak field ligands (left side) ’ Small \( \Delta_o \), long wavelength absorption (red/yellow)
    Strong field ligands (right side) ’ Large \( \Delta_o \), short wavelength absorption (blue/purple)
    
    Code Example 3: Color Prediction of Transition Metal Complexes Using Ligand Field Theory
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    def predict_complex_color(delta_o_cm, d_electron_count, geometry='octahedral'):
        """
        Predict d-d transition wavelength and complex color from ligand field splitting energy
    
        Parameters:
        -----------
        delta_o_cm : float
            Ligand field splitting energy (cm^-1)
        d_electron_count : int
            Number of d electrons (1-10)
        geometry : str
            Coordination geometry ('octahedral' or 'tetrahedral')
    
        Returns:
        --------
        wavelength_nm : float
            Wavelength of d-d transition (nm)
        observed_color : str
            Observed complex color (complementary color)
        """
        # Convert to wavelength
        wavelength_nm = 1e7 / delta_o_cm  # nm
    
        # Color of absorbed light
        if wavelength_nm < 450:
            absorbed_color = 'Violet'
            observed_color = 'Yellow-green'
        elif wavelength_nm < 495:
            absorbed_color = 'Blue'
            observed_color = 'Yellow'
        elif wavelength_nm < 570:
            absorbed_color = 'Green'
            observed_color = 'Red-purple'
        elif wavelength_nm < 590:
            absorbed_color = 'Yellow'
            observed_color = 'Blue-purple'
        elif wavelength_nm < 620:
            absorbed_color = 'Orange'
            observed_color = 'Blue'
        elif wavelength_nm < 750:
            absorbed_color = 'Red'
            observed_color = 'Green'
        else:
            absorbed_color = 'Infrared'
            observed_color = 'Colorless (infrared absorption)'
    
        print(f"Ligand field splitting energy ”o = {delta_o_cm:.0f} cm{¹")
        print(f"d-d transition wavelength: » = {wavelength_nm:.1f} nm")
        print(f"Absorbed light: {absorbed_color} ({wavelength_nm:.1f} nm)")
        print(f"Observed complex color: {observed_color} (complementary color)")
        print(f"d electron count: d^{d_electron_count} ({geometry} coordination)")
    
        return wavelength_nm, observed_color
    
    def plot_spectrochemical_series():
        """
        Visualize spectrochemical series of representative transition metal complexes
        """
        # ”o data for representative complexes (cm^-1)
        complexes = [
            '[Ti(H2O)6]3+',
            '[V(H2O)6]3+',
            '[Cr(H2O)6]3+',
            '[Mn(H2O)6]2+',
            '[Fe(H2O)6]2+',
            '[Co(H2O)6]2+',
            '[Ni(H2O)6]2+',
            '[Cu(H2O)6]2+',
            '[Co(NH3)6]3+',
            '[Cr(CN)6]3-'
        ]
    
        delta_o_values = np.array([20300, 18900, 17400, 21000, 10400, 9300, 8500, 12600, 22900, 26600])
        wavelengths = 1e7 / delta_o_values  # nm
    
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Display absorption wavelength of each complex with colored bars
        colors_map = {
            (380, 450): ('#8B00FF', 'Yellow-green'),
            (450, 495): ('#0000FF', 'Yellow'),
            (495, 570): ('#00FF00', 'Red-purple'),
            (570, 590): ('#FFFF00', 'Blue-purple'),
            (590, 620): ('#FFA500', 'Blue'),
            (620, 750): ('#FF0000', 'Green')
        }
    
        for i, (name, wl) in enumerate(zip(complexes, wavelengths)):
            # Color corresponding to absorption wavelength
            color = '#808080'  # Default gray
            observed = 'Unknown'
            for (wl_min, wl_max), (c, obs) in colors_map.items():
                if wl_min <= wl < wl_max:
                    color = c
                    observed = obs
                    break
    
            ax.barh(i, wl, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.text(wl + 20, i, f'{wl:.1f} nm\nObserved: {observed}',
                    va='center', fontsize=9, fontweight='bold')
    
        ax.set_yticks(range(len(complexes)))
        ax.set_yticklabels(complexes, fontsize=11)
        ax.set_xlabel('d-d Transition Wavelength (nm)', fontsize=12)
        ax.set_title('Spectrochemical Series and d-d Transition Wavelengths of Transition Metal Complexes', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 800)
        ax.grid(axis='x', alpha=0.3)
    
        # Highlight visible light region
        ax.axvspan(380, 750, alpha=0.1, color='yellow', label='Visible Region')
        ax.legend()
    
        plt.tight_layout()
        plt.show()
    
    # Execution example 1: Color prediction of [Cu(H2O)6]2+ (d^9 electron configuration)
    delta_o_cu = 12600  # cm^-1
    wavelength, color = predict_complex_color(delta_o_cu, d_electron_count=9)
    
    print("\n" + "="*50)
    # Execution example 2: Color prediction of [Cr(NH3)6]3+ (d^3 electron configuration, strong field ligand)
    delta_o_cr = 21500  # cm^-1
    wavelength2, color2 = predict_complex_color(delta_o_cr, d_electron_count=3)
    
    # Plot spectrochemical series
    plot_spectrochemical_series()
    
    3.5 Charge Transfer Transition
    3.5.1 LMCT and MLCT Transitions
    Charge transfer transitions have larger molar absorptivity (\( \epsilon > 10^4 \)) than d-d transitions and exhibit strong colors.
    
    Classification of Charge Transfer Transitions
    
    LMCT (Ligand-to-Metal Charge Transfer): Transition where electrons move from ligand to metal ion. Examples: MnO4- (purple), CrO42- (yellow)
    MLCT (Metal-to-Ligand Charge Transfer): Transition where electrons move from metal ion to ligand. Examples: Fe(II)-phenanthroline complex (red), Ru(bpy)32+ (orange)
    
    
    Code Example 4: LMCT Transition Analysis of Permanganate Ion
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_lmct_spectrum(wavelength, lambda_max, epsilon_max, bandwidth):
        """
        Simulate UV-Vis spectrum of LMCT transition with Gaussian function
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        lambda_max : float
            Maximum absorption wavelength (nm)
        epsilon_max : float
            Maximum molar absorptivity (L mol^-1 cm^-1)
        bandwidth : float
            Absorption band width (full width at half maximum, nm)
    
        Returns:
        --------
        epsilon : array
            Molar absorptivity at each wavelength
        """
        sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
        epsilon = epsilon_max * np.exp(-((wavelength - lambda_max)**2) / (2 * sigma**2))
        return epsilon
    
    # Simulate LMCT transition spectrum of MnO4^-
    wavelength = np.linspace(400, 700, 300)
    
    # MnO4^- has strong absorption at 526 nm (green) ’ appears purple
    epsilon_mno4 = simulate_lmct_spectrum(wavelength, lambda_max=526, epsilon_max=2300, bandwidth=80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spectrum plot
    ax1.plot(wavelength, epsilon_mno4, linewidth=2, color='purple', label='MnO„{ LMCT Transition')
    ax1.axvline(526, color='green', linestyle='--', linewidth=1.5, label='Absorption Maximum (526 nm)')
    ax1.fill_between(wavelength, epsilon_mno4, alpha=0.3, color='purple')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Molar Absorptivity µ (L mol{¹ cm{¹)', fontsize=12)
    ax1.set_title('LMCT Transition Spectrum of MnO„{', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Visible spectrum and observed color
    visible_colors = [
        (380, 450, '#8B00FF'),
        (450, 495, '#0000FF'),
        (495, 570, '#00FF00'),
        (570, 590, '#FFFF00'),
        (590, 620, '#FFA500'),
        (620, 750, '#FF0000')
    ]
    
    for wl_min, wl_max, color in visible_colors:
        ax2.axvspan(wl_min, wl_max, color=color, alpha=0.7)
    
    ax2.axvline(526, color='black', linestyle='--', linewidth=2, label='MnO„{ Absorption (526 nm)')
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_title('Visible Spectrum and MnO„{ Absorption', fontsize=14, fontweight='bold')
    ax2.set_xlim(380, 750)
    ax2.set_yticks([])
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("MnO„{ ion:")
    print("- Absorption maximum: 526 nm (green)")
    print("- Observed color: Purple (complementary to green)")
    print("- Transition type: LMCT (ligand O²{ ’ Mnwz)")
    print("- Molar absorptivity: µ H 2300 L mol{¹ cm{¹ (strong absorption)")
    
    3.6 Solvent Effects and Solvent Shifts
    3.6.1 Absorption Wavelength Changes Due to Solvent Polarity
    Solvent polarity stabilizes or destabilizes the electronic states of solute molecules, causing absorption wavelength shifts.
    
    Classification of Solvent Shifts
    
    Red shift (bathochromic shift): Shift to longer wavelengths. Occurs when polar solvents stabilize excited states more.
    Blue shift (hypsochromic shift): Shift to shorter wavelengths. Occurs when polar solvents stabilize ground states more.
    
    
    Code Example 5: Simulation of Absorption Spectrum Shifts Due to Solvent Polarity
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_solvent_shift(wavelength, lambda_max_nonpolar, shift_per_polarity_unit):
        """
        Simulate absorption spectrum shifts due to solvent polarity
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        lambda_max_nonpolar : float
            Absorption maximum wavelength in nonpolar solvent (nm)
        shift_per_polarity_unit : float
            Shift amount per unit polarity (nm)
    
        Returns:
        --------
        spectra : dict
            Spectra in each solvent
        """
        solvents = {
            'Hexane': 0.0,
            'Ethanol': 5.2,
            'Methanol': 6.6,
            'Water': 9.0,
            'DMSO': 7.2
        }
    
        fig, ax = plt.subplots(figsize=(12, 7))
    
        colors = ['blue', 'green', 'orange', 'red', 'purple']
    
        for (solvent, polarity), color in zip(solvents.items(), colors):
            lambda_max = lambda_max_nonpolar + shift_per_polarity_unit * polarity
    
            # Gaussian absorption band
            sigma = 30
            absorbance = np.exp(-((wavelength - lambda_max)**2) / (2 * sigma**2))
    
            ax.plot(wavelength, absorbance, linewidth=2, label=f'{solvent} (»max = {lambda_max:.1f} nm)', color=color)
            ax.axvline(lambda_max, linestyle='--', linewidth=1, color=color, alpha=0.5)
    
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Normalized Absorbance', fontsize=12)
        ax.set_title('Red Shift of Absorption Spectrum Due to Solvent Polarity', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Solvatochromism of À ’ À* transition molecule
    wavelength = np.linspace(300, 500, 200)
    simulate_solvent_shift(wavelength, lambda_max_nonpolar=350, shift_per_polarity_unit=3.0)
    
    print("Absorption wavelength shift due to solvent polarity:")
    print("- Polar solvent ’ Excited state stabilization ’ Red shift (long wavelength shift)")
    print("- Nonpolar solvent ’ No shift")
    print("- À ’ À* transition (high polarity) tends to red shift")
    print("- n ’ À* transition (low polarity) may blue shift")
    
    3.7 Baseline Correction and Spectral Preprocessing
    3.7.1 Scattering Light Correction
    When measuring solid samples or suspensions, scattered light distorts the baseline. Appropriate baseline correction is necessary.
    Code Example 6: Baseline Correction and Scattering Light Removal
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from scipy.interpolate import UnivariateSpline
    
    def baseline_correction_polynomial(wavelength, absorbance, baseline_region, poly_order=2):
        """
        Baseline correction by polynomial fitting
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        absorbance : array
            Raw absorbance data
        baseline_region : tuple
            Wavelength range for baseline region (nm)
        poly_order : int
            Polynomial order
    
        Returns:
        --------
        corrected_absorbance : array
            Corrected absorbance
        """
        # Extract baseline region data
        mask = (wavelength >= baseline_region[0]) & (wavelength <= baseline_region[1])
        wl_base = wavelength[mask]
        abs_base = absorbance[mask]
    
        # Polynomial fitting
        poly_coef = np.polyfit(wl_base, abs_base, poly_order)
        baseline = np.polyval(poly_coef, wavelength)
    
        # Baseline subtraction
        corrected_absorbance = absorbance - baseline
    
        return corrected_absorbance, baseline
    
    def baseline_correction_spline(wavelength, absorbance, baseline_points):
        """
        Baseline correction by spline interpolation
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        absorbance : array
            Raw absorbance data
        baseline_points : list of tuples
            Baseline points [(wl1, abs1), (wl2, abs2), ...]
    
        Returns:
        --------
        corrected_absorbance : array
            Corrected absorbance
        """
        wl_base = np.array([p[0] for p in baseline_points])
        abs_base = np.array([p[1] for p in baseline_points])
    
        # Spline interpolation
        spline = UnivariateSpline(wl_base, abs_base, s=0, k=3)
        baseline = spline(wavelength)
    
        # Baseline subtraction
        corrected_absorbance = absorbance - baseline
    
        return corrected_absorbance, baseline
    
    # Simulation data: Spectrum including scattering light
    wavelength = np.linspace(300, 700, 400)
    
    # True absorption spectrum (Gaussian peak)
    true_abs = 0.8 * np.exp(-((wavelength - 450)**2) / (2 * 50**2))
    
    # Baseline due to scattering light (proportional to inverse power of wavelength)
    scattering_baseline = 0.3 * (wavelength / 300)**(-4)
    
    # Noise
    noise = np.random.normal(0, 0.01, len(wavelength))
    
    # Observed spectrum
    observed_abs = true_abs + scattering_baseline + noise
    
    # Baseline correction (polynomial)
    corrected_abs_poly, baseline_poly = baseline_correction_polynomial(
        wavelength, observed_abs, baseline_region=(600, 700), poly_order=3
    )
    
    # Baseline correction (spline)
    baseline_points = [(300, observed_abs[0]), (380, observed_abs[80]),
                       (600, observed_abs[300]), (700, observed_abs[-1])]
    corrected_abs_spline, baseline_spline = baseline_correction_spline(
        wavelength, observed_abs, baseline_points
    )
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original spectrum
    axes[0, 0].plot(wavelength, observed_abs, label='Observed Spectrum (with scattering)', color='blue')
    axes[0, 0].plot(wavelength, true_abs, '--', label='True Spectrum', color='red', linewidth=2)
    axes[0, 0].plot(wavelength, scattering_baseline, ':', label='Scattering Baseline', color='green', linewidth=2)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Absorbance')
    axes[0, 0].set_title('Observed Spectrum Including Scattering Light')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Polynomial baseline correction
    axes[0, 1].plot(wavelength, observed_abs, label='Observed Spectrum', color='blue', alpha=0.5)
    axes[0, 1].plot(wavelength, baseline_poly, '--', label='Polynomial Baseline', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Absorbance')
    axes[0, 1].set_title('Polynomial Fitting')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Spline baseline correction
    axes[1, 0].plot(wavelength, observed_abs, label='Observed Spectrum', color='blue', alpha=0.5)
    axes[1, 0].plot(wavelength, baseline_spline, '--', label='Spline Baseline', color='purple', linewidth=2)
    for wl, abs_val in baseline_points:
        axes[1, 0].plot(wl, abs_val, 'ro', markersize=8)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Absorbance')
    axes[1, 0].set_title('Spline Interpolation')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Correction result comparison
    axes[1, 1].plot(wavelength, true_abs, '--', label='True Spectrum', color='red', linewidth=2)
    axes[1, 1].plot(wavelength, corrected_abs_poly, label='Polynomial Correction', color='orange', alpha=0.7)
    axes[1, 1].plot(wavelength, corrected_abs_spline, label='Spline Correction', color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Absorbance')
    axes[1, 1].set_title('Baseline Correction Results')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Baseline correction evaluation:")
    poly_error = np.mean((corrected_abs_poly - true_abs)**2)
    spline_error = np.mean((corrected_abs_spline - true_abs)**2)
    print(f"Mean squared error of polynomial correction: {poly_error:.6f}")
    print(f"Mean squared error of spline correction: {spline_error:.6f}")
    
    3.8 Multi-wavelength Analysis and Multi-component Quantification
    3.8.1 Principle of Absorption Additivity
    In mixed solutions where multiple absorbing species coexist, the contribution of each component can be separated using the additivity of the Lambert-Beer law.
    
    Lambert-Beer law for multi-component systems:
            \[
            A(\lambda) = \sum_{i=1}^{n} \epsilon_i(\lambda) \cdot c_i \cdot l
            \]
            Matrix notation (\( m \) wavelengths, \( n \) components):
            \[
            \mathbf{A} = \mathbf{E} \mathbf{c} l
            \]
            Where \( \mathbf{A} \) is absorbance vector (\( m \times 1 \)), \( \mathbf{E} \) is molar absorptivity matrix (\( m \times n \)), and \( \mathbf{c} \) is concentration vector (\( n \times 1 \)).
    
    Code Example 7: Quantification of Two-component Mixtures by Multi-wavelength Analysis
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import nnls  # Non-Negative Least Squares
    
    def multiwavelength_analysis(wavelength, absorbance_mixture, epsilon_matrix, path_length=1.0):
        """
        Determine concentration of each component in mixed solution by multi-wavelength analysis
    
        Parameters:
        -----------
        wavelength : array
            Measurement wavelength (nm)
        absorbance_mixture : array
            Absorbance spectrum of mixed solution
        epsilon_matrix : 2D array
            Molar absorptivity spectra of each component (shape: n_wavelengths × n_components)
        path_length : float
            Path length (cm)
    
        Returns:
        --------
        concentrations : array
            Determined concentrations of each component (mol/L)
        """
        # Determine concentrations by non-negative least squares (prohibit negative concentrations)
        concentrations, residual = nnls(epsilon_matrix * path_length, absorbance_mixture)
    
        # Reconstructed spectrum
        absorbance_reconstructed = epsilon_matrix @ concentrations * path_length
    
        return concentrations, absorbance_reconstructed, residual
    
    # Simulation: Mixed solution of methylene blue (MB) and methyl orange (MO)
    wavelength = np.linspace(400, 700, 300)
    
    # Component 1: Methylene blue (»max = 664 nm)
    epsilon_MB = 8e4 * np.exp(-((wavelength - 664)**2) / (2 * 40**2))
    
    # Component 2: Methyl orange (»max = 464 nm)
    epsilon_MO = 2.7e4 * np.exp(-((wavelength - 464)**2) / (2 * 35**2))
    
    # Molar absorptivity matrix
    epsilon_matrix = np.column_stack([epsilon_MB, epsilon_MO])
    
    # True concentrations (mol/L)
    c_MB_true = 1.5e-5
    c_MO_true = 3.0e-5
    
    # Absorbance of mixed solution (path length 1 cm)
    absorbance_mixture = epsilon_MB * c_MB_true + epsilon_MO * c_MO_true
    absorbance_mixture += np.random.normal(0, 0.005, len(wavelength))  # Noise
    
    # Multi-wavelength analysis
    concentrations, absorbance_recon, residual = multiwavelength_analysis(
        wavelength, absorbance_mixture, epsilon_matrix, path_length=1.0
    )
    
    c_MB_calc, c_MO_calc = concentrations
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Molar absorptivity spectra of each component
    axes[0, 0].plot(wavelength, epsilon_MB, label='Methylene Blue (MB)', color='blue', linewidth=2)
    axes[0, 0].plot(wavelength, epsilon_MO, label='Methyl Orange (MO)', color='orange', linewidth=2)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Molar Absorptivity (L mol{¹ cm{¹)')
    axes[0, 0].set_title('Molar Absorptivity Spectra of Each Component')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Spectrum of mixed solution
    axes[0, 1].plot(wavelength, absorbance_mixture, 'o-', label='Observed Spectrum',
                    color='purple', alpha=0.6, markersize=2)
    axes[0, 1].plot(wavelength, absorbance_recon, '--', label='Reconstructed Spectrum',
                    color='red', linewidth=2)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Absorbance')
    axes[0, 1].set_title('Spectrum of Mixed Solution and Fitting')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Contribution of each component
    abs_MB_contrib = epsilon_MB * c_MB_calc
    abs_MO_contrib = epsilon_MO * c_MO_calc
    axes[1, 0].plot(wavelength, absorbance_mixture, label='Total Absorbance', color='black', linewidth=2)
    axes[1, 0].fill_between(wavelength, 0, abs_MB_contrib, alpha=0.5, color='blue', label='MB Contribution')
    axes[1, 0].fill_between(wavelength, abs_MB_contrib, abs_MB_contrib + abs_MO_contrib,
                            alpha=0.5, color='orange', label='MO Contribution')
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Absorbance')
    axes[1, 0].set_title('Absorbance Contribution of Each Component')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Concentration determination results
    components = ['Methylene Blue', 'Methyl Orange']
    concentrations_true = [c_MB_true, c_MO_true]
    concentrations_calc = [c_MB_calc, c_MO_calc]
    
    x = np.arange(len(components))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, np.array(concentrations_true)*1e5, width, label='True Concentration', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, np.array(concentrations_calc)*1e5, width, label='Calculated Concentration', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Component')
    axes[1, 1].set_ylabel('Concentration (×10{u mol/L)')
    axes[1, 1].set_title('Concentration Determination Results')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("Multi-wavelength analysis results:")
    print(f"Methylene Blue: True concentration = {c_MB_true:.2e} mol/L, Calculated concentration = {c_MB_calc:.2e} mol/L")
    print(f"Methyl Orange: True concentration = {c_MO_true:.2e} mol/L, Calculated concentration = {c_MO_calc:.2e} mol/L")
    print(f"Relative error (MB): {abs(c_MB_calc - c_MB_true)/c_MB_true * 100:.2f}%")
    print(f"Relative error (MO): {abs(c_MO_calc - c_MO_true)/c_MO_true * 100:.2f}%")
    print(f"Residual: {residual:.6f}")
    
    3.9 Time-resolved UV-Vis Spectroscopy
    3.9.1 Kinetic Analysis
    UV-Vis spectroscopy can track the progress of chemical reactions in real-time. Reaction rate constants can be determined from time-dependent changes in absorbance.
    
    Rate equation for first-order reaction:
            \[
            \frac{d[A]}{dt} = -k[A]
            \]
            Integrated form:
            \[
            [A]_t = [A]_0 e^{-kt}
            \]
            Expression in terms of absorbance (\( A_t = \epsilon [A]_t l \)):
            \[
            A_t = A_0 e^{-kt}
            \]
            \[
            \ln A_t = \ln A_0 - kt
            \]
        
    Code Example 8: Determination of First-order Reaction Rate Constant by Time-resolved UV-Vis Spectroscopy
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def first_order_kinetics(time, A0, k):
        """
        Absorbance time change for first-order reaction
    
        Parameters:
        -----------
        time : array
            Time (s)
        A0 : float
            Initial absorbance
        k : float
            Rate constant (s^-1)
    
        Returns:
        --------
        absorbance : array
            Absorbance at time
        """
        return A0 * np.exp(-k * time)
    
    def determine_rate_constant(time, absorbance):
        """
        Determine first-order reaction rate constant from time-resolved UV-Vis data
    
        Parameters:
        -----------
        time : array
            Time (s)
        absorbance : array
            Absorbance at each time
    
        Returns:
        --------
        k : float
            Rate constant (s^-1)
        half_life : float
            Half-life (s)
        """
        # Nonlinear fitting
        popt, pcov = curve_fit(first_order_kinetics, time, absorbance, p0=[absorbance[0], 0.01])
        A0_fit, k_fit = popt
    
        # Half-life
        half_life = np.log(2) / k_fit
    
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Absorbance vs time (exponential plot)
        ax1.plot(time, absorbance, 'o', label='Measured Data', markersize=8, alpha=0.7)
        time_fit = np.linspace(0, time.max(), 200)
        abs_fit = first_order_kinetics(time_fit, A0_fit, k_fit)
        ax1.plot(time_fit, abs_fit, 'r--', linewidth=2,
                 label=f'Fit: A = {A0_fit:.3f} exp(-{k_fit:.4f}t)\nk = {k_fit:.4f} s{¹\nt�/‚ = {half_life:.1f} s')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Absorbance', fontsize=12)
        ax1.set_title('Time Change of First-order Reaction', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
        # ln(A) vs time (linear plot)
        ln_abs = np.log(absorbance)
        ax2.plot(time, ln_abs, 'o', label='Measured Data', markersize=8, alpha=0.7)
        ln_abs_fit = np.log(A0_fit) - k_fit * time_fit
        ax2.plot(time_fit, ln_abs_fit, 'r--', linewidth=2,
                 label=f'Linear Fit\nSlope = -{k_fit:.4f} s{¹')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('ln(Absorbance)', fontsize=12)
        ax2.set_title('First-order Plot (logarithmic)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        return k_fit, half_life, A0_fit
    
    # Simulation: Base hydrolysis reaction of crystal violet
    time_data = np.linspace(0, 300, 30)  # 0-300 seconds, 30 measurements
    k_true = 0.012  # s^-1
    A0_true = 1.2
    absorbance_data = first_order_kinetics(time_data, A0_true, k_true)
    absorbance_data += np.random.normal(0, 0.02, len(time_data))  # Noise
    
    # Rate constant determination
    k_calc, t_half, A0_calc = determine_rate_constant(time_data, absorbance_data)
    
    print("Kinetic analysis results:")
    print(f"Rate constant k = {k_calc:.4f} s{¹ (true value: {k_true:.4f} s{¹)")
    print(f"Half-life t�/‚ = {t_half:.1f} s")
    print(f"Initial absorbance A€ = {A0_calc:.3f}")
    print(f"Relative error: {abs(k_calc - k_true)/k_true * 100:.2f}%")
    
    3.10 Exercise Problems
    
    Basic Problems (Easy)
    Problem 1: Wavelength and Energy Conversion
    A compound observed by UV-Vis spectroscopy has an absorption maximum wavelength of 450 nm. Calculate the photon energy corresponding to this absorption in eV units.
    
    View Answer
    
    Answer:
    Using the wavelength and energy conversion formula:
                    \[
                    E\,(\text{eV}) = \frac{1239.8}{\lambda\,(\text{nm})} = \frac{1239.8}{450} = 2.755\,\text{eV}
                    \]
                    Answer: 2.76 eV
    Python code:
    lambda_nm = 450
    E_eV = 1239.8 / lambda_nm
    print(f"Photon energy: {E_eV:.3f} eV")
    
    
    
    Problem 2: Concentration Calculation Using Lambert-Beer Law
    A compound solution with molar absorptivity \( \epsilon = 1.5 \times 10^4 \) L mol-1 cm-1 (path length 1 cm) has an absorbance of 0.75. Calculate the concentration (mol/L) of this solution.
    
    View Answer
    
    Answer:
    From Lambert-Beer law: \( A = \epsilon c l \),
                    \[
                    c = \frac{A}{\epsilon l} = \frac{0.75}{1.5 \times 10^4 \times 1} = 5.0 \times 10^{-5}\,\text{mol/L}
                    \]
                    Answer: 5.0 × 10-5 mol/L
    Python code:
    A = 0.75
    epsilon = 1.5e4  # L mol^-1 cm^-1
    l = 1.0  # cm
    c = A / (epsilon * l)
    print(f"Concentration: {c:.2e} mol/L")
    
    
    
    Problem 3: Conversion Between Transmittance and Absorbance
    A solution has a transmittance of 40%. Calculate the absorbance of this solution.
    
    View Answer
    
    Answer:
    Relationship between absorbance and transmittance: \( A = -\log_{10} T = 2 - \log_{10}(\%T) \)
                    \[
                    A = 2 - \log_{10}(40) = 2 - 1.602 = 0.398
                    \]
                    Answer: 0.398
    Python code:
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Python code:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    T_percent = 40
    A = 2 - np.log10(T_percent)
    print(f"Absorbance: {A:.3f}")
    
    
    
    
    
    Intermediate Problems (Medium)
    Problem 4: Bandgap Determination by Tauc Plot
    The following data were obtained from the UV-Vis spectrum of a semiconductor material. Create a Tauc plot (direct transition) and determine the bandgap. Sample thickness: 0.01 cm
    
    Wavelength (nm)400420440460480500
    Absorbance1.201.050.850.600.350.15
    
    
    View Answer
    
    Answer:
    1. Convert wavelength to photon energy: \( E = 1239.8 / \lambda \)
    2. Calculate absorption coefficient: \( \alpha = 2.303 \cdot A / l \)
    3. Tauc plot: Extrapolate linear region of \( (\alpha h\nu)^2 \) vs. \( h\nu \)
    Python code:
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Python code:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    wavelength = np.array([400, 420, 440, 460, 480, 500])
    absorbance = np.array([1.20, 1.05, 0.85, 0.60, 0.35, 0.15])
    thickness = 0.01  # cm
    
    # Photon energy
    E = 1239.8 / wavelength  # eV
    
    # Absorption coefficient
    alpha = 2.303 * absorbance / thickness  # cm^-1
    
    # Tauc plot
    tauc_y = (alpha * E)**2
    
    # Linear region fitting (E > 2.7 eV)
    mask = E > 2.7
    slope, intercept, r_value, _, _ = linregress(E[mask], tauc_y[mask])
    
    # Bandgap (horizontal axis intercept)
    Eg = -intercept / slope
    
    plt.figure(figsize=(10, 6))
    plt.plot(E, tauc_y, 'o', markersize=10, label='Measured Data')
    E_fit = np.linspace(Eg, E.max(), 100)
    tauc_fit = slope * E_fit + intercept
    plt.plot(E_fit, tauc_fit, 'r--', linewidth=2, label=f'Eg = {Eg:.3f} eV')
    plt.axvline(Eg, color='green', linestyle=':', linewidth=2)
    plt.xlabel('Photon Energy (eV)', fontsize=12)
    plt.ylabel('(±h½)² (eV² cm{²)', fontsize=12)
    plt.title('Tauc Plot', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"Bandgap Eg = {Eg:.3f} eV")
    print(f"Corresponding wavelength: » = {1239.8/Eg:.1f} nm")
    
    Answer: Eg H 2.6-2.7 eV (depends on actual fitting results)
    
    
    Problem 5: Color Prediction of Complex Using Ligand Field Theory
    The ligand field splitting energy of [Co(H2O)6]2+ is \( \Delta_o = 9300 \) cm-1. Predict the wavelength of light absorbed by this complex and the observed color of the complex.
    
    View Answer
    
    Answer:
    1. Calculate absorption wavelength:
                    \[
                    \lambda = \frac{1}{\Delta_o\,(\text{cm}^{-1})} \times 10^7\,(\text{nm}) = \frac{10^7}{9300} = 1075\,\text{nm}
                    \]
                    2. 1075 nm is in the near-infrared region (outside visible region). However, Co2+ (d7) has multiple d-d transitions, and actually has absorption in the visible region.
    3. [Co(H2O)6]2+ has strong absorption around 510 nm (green) and appears pink (complementary to green).
    Python code:
    delta_o_cm = 9300
    wavelength_nm = 1e7 / delta_o_cm
    print(f"”o corresponding wavelength: {wavelength_nm:.1f} nm (near-infrared)")
    print("Actual [Co(H2O)6]2+ complex:")
    print("- Main absorption: 510 nm (green)")
    print("- Observed color: Pink (complementary to green)")
    
    Answer: Complex color is pink
    
    
    Problem 6: Two-component Quantification by Multi-wavelength Analysis
    A mixed solution of methylene blue (MB, \( \epsilon_{664} = 8 \times 10^4 \) L mol-1 cm-1) and methyl orange (MO, \( \epsilon_{464} = 2.7 \times 10^4 \)) was measured in a 1 cm cell, yielding A664 = 0.40 and A464 = 0.54. Determine the concentration of each component. Assume that MO does not absorb at 664 nm and MB absorption at 464 nm is negligible.
    
    View Answer
    
    Answer:
    At 664 nm (only MB absorbs):
                    \[
                    c_{\text{MB}} = \frac{A_{664}}{\epsilon_{\text{MB},664} \cdot l} = \frac{0.40}{8 \times 10^4 \times 1} = 5.0 \times 10^{-6}\,\text{mol/L}
                    \]
    
                    At 464 nm (only MO absorbs):
                    \[
                    c_{\text{MO}} = \frac{A_{464}}{\epsilon_{\text{MO},464} \cdot l} = \frac{0.54}{2.7 \times 10^4 \times 1} = 2.0 \times 10^{-5}\,\text{mol/L}
                    \]
    
                    Python code:
    A_664 = 0.40
    A_464 = 0.54
    epsilon_MB_664 = 8e4  # L mol^-1 cm^-1
    epsilon_MO_464 = 2.7e4
    l = 1.0  # cm
    
    c_MB = A_664 / (epsilon_MB_664 * l)
    c_MO = A_464 / (epsilon_MO_464 * l)
    
    print(f"Methylene Blue concentration: {c_MB:.2e} mol/L")
    print(f"Methyl Orange concentration: {c_MO:.2e} mol/L")
    
    Answer: MB = 5.0 × 10-6 mol/L, MO = 2.0 × 10-5 mol/L
    
    
    
    
    Advanced Problems (Hard)
    Problem 7: Determination of Thermodynamic Parameters from Temperature-dependent UV-Vis Spectra
    The equilibrium constant \( K \) for the equilibrium system A Ì B was determined by UV-Vis spectroscopy at different temperatures. From the following data, create a van't Hoff plot and determine the enthalpy change \( \Delta H^\circ \) and entropy change \( \Delta S^\circ \) of the reaction.
    
    Temperature (K)298308318328338
    Equilibrium constant K0.500.801.201.752.40
    
    
    View Answer
    
    Answer:
    van't Hoff equation:
                    \[
                    \ln K = -\frac{\Delta H^\circ}{R} \cdot \frac{1}{T} + \frac{\Delta S^\circ}{R}
                    \]
                    Determine \( \Delta H^\circ \) from the slope and \( \Delta S^\circ \) from the intercept of the \( \ln K \) vs. \( 1/T \) plot.
    Python code:
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Python code:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    T = np.array([298, 308, 318, 328, 338])  # K
    K = np.array([0.50, 0.80, 1.20, 1.75, 2.40])
    
    # van't Hoff plot
    inv_T = 1 / T  # K^-1
    ln_K = np.log(K)
    
    # Linear regression
    slope, intercept, r_value, _, _ = linregress(inv_T, ln_K)
    
    # Thermodynamic parameters
    R = 8.314  # J mol^-1 K^-1
    Delta_H = -slope * R  # J/mol
    Delta_S = intercept * R  # J/(mol K)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(inv_T * 1000, ln_K, 'o', markersize=10, label='Measured Data')
    inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
    ln_K_fit = slope * inv_T_fit + intercept
    plt.plot(inv_T_fit * 1000, ln_K_fit, 'r--', linewidth=2,
             label=f'”H° = {Delta_H/1000:.2f} kJ/mol\n”S° = {Delta_S:.2f} J/(mol·K)\nR² = {r_value**2:.4f}')
    plt.xlabel('1000/T (K{¹)', fontsize=12)
    plt.ylabel('ln K', fontsize=12)
    plt.title("van't Hoff Plot", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"Enthalpy change: ”H° = {Delta_H/1000:.2f} kJ/mol")
    print(f"Entropy change: ”S° = {Delta_S:.2f} J/(mol·K)")
    print(f"Correlation coefficient: R² = {r_value**2:.4f}")
    
    Answer: ”H° H 35-40 kJ/mol, ”S° H 90-100 J/(mol·K) (depends on actual data)
    
    
    Problem 8: Advanced Baseline Correction for Spectra Including Scattering Light
    The UV-Vis spectrum of nanoparticle suspension has superimposed Rayleigh scattering (\( \propto \lambda^{-4} \)) and Mie scattering (\( \propto \lambda^{-n}, n < 4 \)). Create a Python program to extract the true absorption spectrum from the following spectrum and determine the bandgap.
    
    View Answer
    
    Answer:
    Baseline correction strategy:
    
    Fit scattering component from data at longer wavelengths than absorption edge
    Fit with form \( A_{\text{scattering}} = C \lambda^{-n} \)
    Subtract scattering component over entire wavelength range
    Determine bandgap by Tauc plot from corrected spectrum
    
    Python code:
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def scattering_baseline(wavelength, C, n):
        """Scattering baseline (power law)"""
        return C * wavelength**(-n)
    
    def advanced_baseline_correction(wavelength, absorbance, scattering_region):
        """
        Advanced baseline correction for spectra including scattering light
    
        Parameters:
        -----------
        wavelength : array
            Wavelength (nm)
        absorbance : array
            Observed absorbance
        scattering_region : tuple
            Scattering fitting region (nm)
    
        Returns:
        --------
        corrected_absorbance : array
            Corrected absorbance
        """
        # Scattering region data
        mask = (wavelength >= scattering_region[0]) & (wavelength <= scattering_region[1])
        wl_scatter = wavelength[mask]
        abs_scatter = absorbance[mask]
    
        # Power law fitting
        popt, _ = curve_fit(scattering_baseline, wl_scatter, abs_scatter, p0=[1e7, 4.0], maxfev=5000)
        C, n = popt
    
        # Calculate scattering component over entire wavelength range
        baseline = scattering_baseline(wavelength, C, n)
    
        # Baseline subtraction
        corrected = absorbance - baseline
        corrected = np.maximum(corrected, 0)  # Clip negative values to 0
    
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # Original spectrum
        axes[0, 0].plot(wavelength, absorbance, label='Observed Spectrum', color='blue')
        axes[0, 0].plot(wavelength, baseline, '--', label=f'Scattering Baseline\n(»^-{n:.2f})', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Wavelength (nm)')
        axes[0, 0].set_ylabel('Absorbance')
        axes[0, 0].set_title('Spectrum Including Scattering Light')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
        # Corrected spectrum
        axes[0, 1].plot(wavelength, corrected, label='Corrected Spectrum', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Absorbance')
        axes[0, 1].set_title('After Baseline Correction')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
        # Tauc plot
        E = 1239.8 / wavelength
        alpha = 2.303 * corrected / 0.01  # Assumption: sample thickness 0.01 cm
        tauc_y = (alpha * E)**2
    
        # Bandgap determination
        mask_tauc = (E > 2.5) & (E < 3.5)
        if np.sum(mask_tauc) > 5:
            from scipy.stats import linregress
            slope_t, intercept_t, _, _, _ = linregress(E[mask_tauc], tauc_y[mask_tauc])
            Eg = -intercept_t / slope_t if slope_t > 0 else np.nan
        else:
            Eg = np.nan
    
        axes[1, 0].plot(E, tauc_y, 'o-', label='Tauc Plot')
        if not np.isnan(Eg):
            E_fit = np.linspace(Eg, E[mask_tauc].max(), 100)
            tauc_fit = slope_t * E_fit + intercept_t
            axes[1, 0].plot(E_fit, tauc_fit, 'r--', linewidth=2, label=f'Eg = {Eg:.3f} eV')
            axes[1, 0].axvline(Eg, color='green', linestyle=':', linewidth=2)
        axes[1, 0].set_xlabel('Photon Energy (eV)')
        axes[1, 0].set_ylabel('(±h½)² (eV² cm{²)')
        axes[1, 0].set_title('Tauc Plot (After Correction)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
        # Scattering index evaluation
        axes[1, 1].text(0.5, 0.5, f'Scattering Analysis Results:\n\nScattering Index n = {n:.2f}\n\nRayleigh Scattering (n=4): Small particles\nMie Scattering (n<4): Large particles\n\nBandgap Eg = {Eg:.3f} eV',
                       transform=axes[1, 1].transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
    
        plt.tight_layout()
        plt.show()
    
        return corrected, baseline, Eg
    
    # Simulation data
    wavelength = np.linspace(300, 800, 500)
    # True absorption (TiO2, Eg=3.2 eV)
    Eg_true = 3.2
    alpha_true = 1e4 * np.maximum(0, (1239.8/wavelength - Eg_true))**2
    true_abs = alpha_true * 0.01 / 2.303
    
    # Scattering component (Rayleigh + Mie)
    scattering = 0.5 * (wavelength / 300)**(-3.5)
    
    # Observed spectrum
    observed = true_abs + scattering + np.random.normal(0, 0.01, len(wavelength))
    
    # Advanced baseline correction
    corrected, baseline, Eg_calc = advanced_baseline_correction(
        wavelength, observed, scattering_region=(600, 800)
    )
    
    print(f"Determined bandgap: Eg = {Eg_calc:.3f} eV")
    print(f"True bandgap: Eg = {Eg_true:.3f} eV")
    print(f"Error: {abs(Eg_calc - Eg_true):.3f} eV")
    
    Answer: Precise determination of Eg (error within 0.1 eV)
    
    
    Problem 9: Structure Prediction from UV-Vis Spectra Using Machine Learning
    Build a machine learning model that predicts molecular structure (conjugation length, functional group types) from UV-Vis spectra of compounds. Create a program that predicts the number of conjugated double bonds from absorption maximum wavelength using scikit-learn's random forest regression.
    
    View Answer
    
    Answer:
    Relationship between conjugation length and absorption wavelength (based on Woodward-Fieser rules):
                    \[
                    \lambda_{\max} = \lambda_{\text{base}} + \Delta \lambda \times n
                    \]
                    Where \( n \) is the number of conjugated double bonds.
    Python code:
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Python code:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # Training data (absorption wavelength data for conjugated polyenes)
    # n: number of conjugated double bonds, lambda_max: absorption maximum wavelength (nm)
    n_conjugated = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lambda_max = np.array([165, 217, 258, 290, 315, 334, 349, 364, 377, 390])
    
    # Features (can include absorption wavelength, molar absorptivity, absorption bandwidth, etc.)
    X = lambda_max.reshape(-1, 1)
    y = n_conjugated
    
    # Train-test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Prediction
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Evaluation
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Training data and model
    plt.subplot(1, 2, 1)
    lambda_range = np.linspace(150, 400, 200).reshape(-1, 1)
    n_predicted = rf_model.predict(lambda_range)
    plt.plot(lambda_range, n_predicted, 'r-', linewidth=2, label='RF Prediction Model')
    plt.scatter(X_train, y_train, s=100, alpha=0.7, label='Training Data', color='blue')
    plt.scatter(X_test, y_test, s=100, alpha=0.7, label='Test Data', color='green')
    plt.xlabel('Absorption Maximum Wavelength (nm)', fontsize=12)
    plt.ylabel('Number of Conjugated Double Bonds', fontsize=12)
    plt.title(f'Random Forest Regression\nR²(train) = {r2_train:.3f}, R²(test) = {r2_test:.3f}',
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Prediction accuracy
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, s=100, alpha=0.7, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2, label='Ideal Line')
    plt.xlabel('True Conjugated Bond Count', fontsize=12)
    plt.ylabel('Predicted Conjugated Bond Count', fontsize=12)
    plt.title(f'Prediction Accuracy\nMAE = {mae_test:.2f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Prediction for unknown samples
    unknown_lambda = np.array([[280], [350], [400]])
    predicted_n = rf_model.predict(unknown_lambda)
    print("Structure prediction for unknown samples:")
    for lam, n_pred in zip(unknown_lambda.flatten(), predicted_n):
        print(f"  »max = {lam:.0f} nm ’ Predicted conjugation: {n_pred:.1f}")
    
    Answer: High-accuracy prediction model construction with R² > 0.95
    
    
    
    
    Learning Objectives Check
    Self-evaluate the following items:
    Level 1: Basic Understanding
    
    Can explain the principles of UV-Vis spectroscopy and types of electronic transitions
    Can perform concentration calculations using the Lambert-Beer law
    Can convert between wavelength and energy
    Understand the relationship between absorbance and transmittance
    
    Level 2: Practical Skills
    
    Can determine bandgap by Tauc plot method
    Can perform quantitative analysis by calibration curve method
    Can predict color of transition metal complexes using ligand field theory
    Can perform baseline correction and spectral preprocessing
    Can perform multi-component quantification by multi-wavelength analysis
    
    Level 3: Application
    
    Can correct complex spectra including scattering light
    Can determine reaction rate constants by time-resolved UV-Vis spectroscopy
    Can perform spectral analysis considering solvent effects
    Can perform spectral data analysis using machine learning
    
    
    
    References
    
    Atkins, P., de Paula, J. (2010). Physical Chemistry (9th ed.). Oxford University Press, pp. 465-468 (Beer-Lambert law), pp. 495-502 (electronic transitions), pp. 510-518 (ligand field theory). - Detailed explanation of quantum mechanical foundations of UV-Vis spectroscopy, selection rules for electronic transitions, ligand field theory
    Figgis, B.N., Hitchman, M.A. (2000). Ligand Field Theory and Its Applications. Wiley-VCH, pp. 85-105 (d-orbital splitting), pp. 120-135 (spectrochemical series), pp. 140-150 (electronic spectra of complexes). - Systematic explanation of ligand field theory, d-d transitions and complex colors, theoretical background of spectrochemical series
    Tauc, J., Grigorovici, R., Vancu, A. (1966). Optical properties and electronic structure of amorphous germanium. Physica Status Solidi (b), 15(2), 627-637. DOI: 10.1002/pssb.19660150224 - Original paper of Tauc plot method, establishment of semiconductor bandgap determination method
    Perkampus, H.-H. (1992). UV-VIS Spectroscopy and Its Applications. Springer, pp. 1-18 (principles), pp. 32-48 (quantitative analysis), pp. 120-145 (solvent effects), pp. 165-180 (practical applications). - Practical applications of UV-Vis spectroscopy, quantitative analysis methods, examples of spectral interpretation
    Casida, M. E. (1995). Time-dependent density functional response theory for molecules. In Recent Advances in Density Functional Methods (Part I), pp. 155-192. World Scientific, Singapore. - Theoretical foundations of UV-Vis spectrum calculation by TDDFT method
    SciPy 1.11 documentation. scipy.optimize.curve_fit, scipy.optimize.nnls. https://docs.scipy.org/doc/scipy/reference/optimize.html - Nonlinear least squares fitting, non-negative least squares method, application to spectral fitting
    MakuBa, P., Pacia, M., Macyk, W. (2018). How to correctly determine the band gap energy of modified semiconductor photocatalysts based on UVVis spectra. Journal of Physical Chemistry Letters, 9(23), 6814-6817. DOI: 10.1021/acs.jpclett.8b02892 - Correct application of Tauc plot method, common errors and their avoidance
    scikit-learn 1.3 documentation. Ensemble methods (RandomForestRegressor). https://scikit-learn.org/stable/modules/ensemble.html#forest - Random forest regression, application to machine learning analysis of UV-Vis spectra
    ```

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
