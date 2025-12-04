---
title: "Chapter 2: Infrared and Raman Spectroscopy"
chapter_title: "Chapter 2: Infrared and Raman Spectroscopy"
subtitle: Molecular Structure and Chemical bonding via vibrational Spectroscopy
---

This chapter covers Infrared and Raman Spectroscopy. You will learn difference in selection rules for IR, Functional Group (C=O: 1700 cm⁻¹, and centrosymmetricsymmetric molecule Description.

## Introduction

Infrared spectroscopy (Infrared Spectroscopy, IR) and Raman Spectroscopy are complementary techniques that elucidate chemical bonds, functional groups, and crystal structures through molecular vibrational information. IR measures the absorption of infrared light, while Raman observes the frequency shift of scattered light. Since they follow different selection rules, vibrations that are IR-active may be Raman-inactive, and vice versa, providing complementary information.

**Choosing Between IR and Raman**  

  * **IR** : Detection of polar groups (C=O, O-H, N-H), identification of functional groups in organic compounds, applicable to solids, liquids, and gases
  * **Raman** : Detection of symmetric vibrations (C=C, S-S), aqueous samples, crystallinity evaluation (low-frequency region), non-destructive and non-contact measurement

## 1\. Fundamentals of Molecular vibrations

### 1.1 Harmonic Oscillator model

The vibration of a diatomic molecule can be approximated by a harmonic oscillator. The potential energy follows Hooke's law:

$$V(r) = \frac{1}{2}k(r - r_e)^2$$

where $k$ is the force constant (N/m) and $r_e$ is the equilibrium internuclear distance. The vibrational frequency $\nu$ is given by:

$$\nu = \frac{1}{2\pi}\sqrt{\frac{k}{\mu}}$$

$\mu = \frac{m_1 m_2}{m_1 + m_2}$ is the reduced mass. The vibrational energy levels are quantized:

$$E_v = h\nu \left(v + \frac{1}{2}\right), \quad v = 0, 1, 2, \ldots$$

In the harmonic oscillator approximation, the selection rule is $\Delta v = \pm 1$ (only fundamental vibrations are allowed). In real molecules, anharmonicity allows weak observation of $\Delta v = \pm 2, \pm 3, \ldots$ (overtones).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 1: Calculation of Energy Levels and vibrational Frequencies for Harmonic Oscillator</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    h = 6.62607015e-34 # J·s
    c = 2.99792458e8 # m/s
    u = 1.66053906660e-27 # kg (atomic mass unit)
    
    def vibrational_frequency(k, m1, m2):
     """
     Calculate vibrational frequency (Hz) and wavenumber (cm^-1) for diatomic molecule
    
     Parameters:
     -----------
     k : float
     Force constant (N/m)
     m1, m2 : float
     Atomic mass (amu)
    
     Returns:
     --------
     freq_Hz : float
     vibrational frequency (Hz)
     wavenumber : float
     wavenumber (cm^-1)
     """
     # Reduced mass
     mu = (m1 * m2) / (m1 + m2) * u # kg
    
     # vibrational frequency
     freq_Hz = (1 / (2 * np.pi)) * np.sqrt(k / mu)
    
     # Convert to wavenumber
     wavenumber = freq_Hz / (c * 100) # cm^-1
    
     return freq_Hz, wavenumber
    
    def energy_levels(v_max, freq_Hz):
     """
     Energy levels of harmonic oscillator
    
     Parameters:
     -----------
     v_max : int
     Maximum vibrational quantum number
     freq_Hz : float
     vibrational frequency (Hz)
    
     Returns:
     --------
     v : array
     vibrational quantum number
     E : array
     Energy (eV)
     """
     v = np.arange(0, v_max + 1)
     E_J = h * freq_Hz * (v + 0.5)
     E_eV = E_J / 1.602176634e-19
     return v, E_eV
    
    # Calculation for typical chemical bonds
    bonds = {
     'C-H': {'k': 500, 'm1': 12, 'm2': 1},
     'C=O': {'k': 1200, 'm1': 12, 'm2': 16},
     'C-C': {'k': 400, 'm1': 12, 'm2': 12},
     'O-H': {'k': 750, 'm1': 16, 'm2': 1}
    }
    
    print("=" * 70)
    print("Vibrational Frequencies of Typical Chemical Bonds")
    print("=" * 70)
    print(f"{'bond':<8} {'Force Constant (N/m)':<18} {'Frequency (Hz)':<18} {'Wavenumber (cm⁻¹)':<15}")
    print("-" * 70)
    
    for bond, params in bonds.items():
     freq_Hz, wavenumber = vibrational_frequency(params['k'], params['m1'], params['m2'])
     print(f"{bond:<8} {params['k']:<18} {freq_Hz:.3e} {wavenumber:<15.1f}")
    
    # Energy Levels of C=O stretching vibration
    freq_Hz_CO, wn_CO = vibrational_frequency(1200, 12, 16)
    v, E = energy_levels(5, freq_Hz_CO)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy level diagram
    ax1.hlines(E, 0, 1, colors='#f093fb', linewidths=3)
    for i, (vi, Ei) in enumerate(zip(v, E)):
     ax1.text(1.1, Ei, f'v={vi}', fontsize=11, va='center')
     if i < len(v) - 1:
     # transition arrow
     ax1.annotate('', xy=(0.5, E[i+1]), xytext=(0.5, E[i]),
     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.set_xlim(-0.2, 1.5)
    ax1.set_ylim(E[0] - 0.05, E[-1] + 0.1)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('Energy Levels of C=O stretching vibration', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(axis='y', alpha=0.3)
    
    # Isotope effect
    masses_C = np.array([12, 13, 14])
    wavenumbers = []
    for m_C in masses_C:
     _, wn = vibrational_frequency(1200, m_C, 16)
     wavenumbers.append(wn)
    
    ax2.bar([f'$^{{{int(m)}}}$C=O' for m in masses_C], wavenumbers,
     color=['#f093fb', '#f5576c', '#4ecdc4'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax2.set_title('Isotope Effect: C=O stretching vibration', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (m, wn) in enumerate(zip(masses_C, wavenumbers)):
     ax2.text(i, wn + 20, f'{wn:.0f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('vibrational_fundamentals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nWavenumber of C=O stretching vibration: {wn_CO:.1f} cm⁻¹")
    print(f"Zero-point energy of ground state (v=0): {E[0]:.4f} eV")
    print(f"v=0 → v=1 transition energy: {E[1] - E[0]:.4f} eV")
    

### 1.2 vibrational modes of Polyatomic Molecules

A molecule consisting of $N$ atoms has $3N$ degrees of freedom: 3 are translational, 3 are rotational (for non-linear molecules), and the remaining $3N - 6$ (or $3N - 5$ for linear molecules) are vibrational degrees of freedom.
    
    
    ```mermaid
    flowchart TD
     A[Total DOF: 3N] --> B[Translation: 3]
     A --> C[Rotation]
     A --> D[vibration]
    
     C --> C1[Non-linear: 3]
     C --> C2[Linear: 2]
    
     D --> D1[Non-linear: 3N-6]
     D --> D2[Linear: 3N-5]
    
     D1 --> E[H₂O: 3 modesCO₂: 4 modesBenzene: 30 modes]
    
     style A fill:#f093fb,color:#fff
     style D fill:#f5576c,color:#fff
     style E fill:#a8e6cf,color:#000
    ```

Vibrational modes are classified into **stretching vibrations** and **bending vibrations** :

  * **Stretching vibrations** : symmetric stretching(symmetric stretch, νₛ), asymmetric stretching(asymmetric stretch, νₐₛ)
  * **bendingvibration** : scissoringvibration(scissoring, δ), rockingvibration(rocking, ρ), waggingvibration(wagging, ω), twistingvibration(twisting, τ)

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 2: Simulation of Three vibrational modes of H₂O Molecule</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    def h2o_normal_modes():
     """
     Visualization of three vibrational modes of H2O molecule (symmetric stretching, asymmetric stretching, bending)
    
     Returns:
     --------
     fig : matplotlib figure
     vibrational mode diagram
     """
     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
     # Equilibrium positions (O atom at origin)
     O = np.array([0, 0])
     H1 = np.array([-0.76, 0.59]) # Left H
     H2 = np.array([0.76, 0.59]) # Right H
    
     modes = [
     {
     'name': 'Symmetric stretch (νₛ)',
     'freq': '3657 cm⁻¹',
     'displacements': {
     'O': np.array([0, 0]),
     'H1': np.array([-0.1, 0.08]),
     'H2': np.array([0.1, 0.08])
     }
     },
     {
     'name': 'bendingvibration (δ)',
     'freq': '1595 cm⁻¹',
     'displacements': {
     'O': np.array([0, 0]),
     'H1': np.array([-0.05, -0.1]),
     'H2': np.array([0.05, -0.1])
     }
     },
     {
     'name': 'Asymmetric stretch (νₐₛ)',
     'freq': '3756 cm⁻¹',
     'displacements': {
     'O': np.array([0, 0]),
     'H1': np.array([-0.1, 0.08]),
     'H2': np.array([0.1, -0.08])
     }
     }
     ]
    
     for ax, mode in zip(axes, modes):
     # Equilibrium position
     ax.plot(*O, 'ro', markersize=15, label='O')
     ax.plot(*H1, 'bo', markersize=10, label='H')
     ax.plot(*H2, 'bo', markersize=10)
     ax.plot([O[0], H1[0]], [O[1], H1[1]], 'k-', linewidth=2)
     ax.plot([O[0], H2[0]], [O[1], H2[1]], 'k-', linewidth=2)
    
     # Vibrational displacement (enlarged)
     scale = 2
     O_disp = O + scale * mode['displacements']['O']
     H1_disp = H1 + scale * mode['displacements']['H1']
     H2_disp = H2 + scale * mode['displacements']['H2']
    
     ax.plot(*O_disp, 'ro', markersize=15, alpha=0.3)
     ax.plot(*H1_disp, 'bo', markersize=10, alpha=0.3)
     ax.plot(*H2_disp, 'bo', markersize=10, alpha=0.3)
     ax.plot([O_disp[0], H1_disp[0]], [O_disp[1], H1_disp[1]], 'k--', linewidth=2, alpha=0.3)
     ax.plot([O_disp[0], H2_disp[0]], [O_disp[1], H2_disp[1]], 'k--', linewidth=2, alpha=0.3)
    
     # Displacement vector
     ax.arrow(*O, *mode['displacements']['O'], head_width=0.08, head_length=0.05, fc='red', ec='red')
     ax.arrow(*H1, *mode['displacements']['H1'], head_width=0.08, head_length=0.05, fc='blue', ec='blue')
     ax.arrow(*H2, *mode['displacements']['H2'], head_width=0.08, head_length=0.05, fc='blue', ec='blue')
    
     ax.set_xlim(-1.2, 1.2)
     ax.set_ylim(-0.5, 1)
     ax.set_aspect('equal')
     ax.set_title(f"{mode['name']}\n{mode['freq']}", fontsize=12, fontweight='bold')
     ax.axis('off')
    
     plt.tight_layout()
     plt.savefig('h2o_normal_modes.png', dpi=300, bbox_inches='tight')
     plt.show()
    
     return fig
    
    # Execute
    fig = h2o_normal_modes()
    
    # Characteristics of vibrational modes
    print("=" * 60)
    print("Three Fundamentalvibrational mode")
    print("=" * 60)
    print("1. Symmetric stretch (νₛ): 3657 cm⁻¹")
    print(" - Both O-H bonds stretch simultaneously")
    print(" - strong IR absorption (large dipole moment change)")
    print("")
    print("2. bendingvibration (δ): 1595 cm⁻¹")
    print(" - H-O-H angle changes")
    print(" - Lowest frequency (weaker force constant)")
    print("")
    print("3. Asymmetric stretch (νₐₛ): 3756 cm⁻¹")
    print(" - When one O-H stretches, the other contracts")
    print(" - strongest IR absorption")
    

## 2\. Infrared Spectroscopy (IR)

### 2.1 Selection Rules for IR Absorption

For a vibration to be IR active, the **dipole moment $\boldsymbol{\mu}$ must change** during the vibration:

$$\left(\frac{\partial \boldsymbol{\mu}}{\partial Q}\right)_0 \neq 0$$

where $Q$ is the normal coordinate of the vibration. The symmetric stretching vibration of symmetric molecules (such as CO₂) is IR-inactive, but asymmetric stretching and bending vibrations are IR-active.

### 2.2 Functional Groups and Characteristic Absorption

Functional groups in organic compounds exhibit characteristic absorptions in specific wavenumber regions. This allows molecular structure to be estimated from IR spectra.

Functional Group | vibrational mode | Wavenumber (cm⁻¹) | Intensity  
---|---|---|---  
O-H (Alcohol) | stretching | 3200-3600 | strong, broad  
N-H | stretching | 3300-3500 | medium  
C-H (Aliphatic) | stretching | 2850-2960 | strong  
C≡N | stretching | 2210-2260 | medium  
C=O (Carbonyl) | stretching | 1650-1750 | verystrong  
C=C | stretching | 1620-1680 | weak to medium  
C-O | stretching | 1000-1300 | strong  
Aromatic ring | Out-of-plane bending | 690-900 | strong  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 3: IR Spectrum Simulation (Ethanol)</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def lorentzian_peak(x, center, intensity, width):
     """Lorentzian line shape function"""
     return intensity * (width**2 / ((x - center)**2 + width**2))
    
    def simulate_ir_spectrum(peaks, x_range=(4000, 400)):
     """
     IR spectrum simulation
    
     Parameters:
     -----------
     peaks : list of dict
     Information for each peak [{'center': cm-1, 'intensity': 0-1, 'width': cm-1, 'label': str},...]
     x_range : tuple
     Wavenumber range (cm⁻¹)
    
     Returns:
     --------
     wavenumbers : array
     Wavenumber axis
     transmittance : array
     Transmittance (%)
     """
     wavenumbers = np.linspace(x_range[0], x_range[1], 2000)
     absorbance = np.zeros_like(wavenumbers)
    
     for peak in peaks:
     absorbance += lorentzian_peak(wavenumbers, peak['center'],
     peak['intensity'], peak['width'])
    
     transmittance = 100 * np.exp(-absorbance)
     return wavenumbers, transmittance, peaks
    
    # IR spectrum of ethanol (CH₃CH₂OH)
    ethanol_peaks = [
     {'center': 3350, 'intensity': 1.5, 'width': 100, 'label': 'O-Hstretching'},
     {'center': 2970, 'intensity': 0.8, 'width': 20, 'label': 'C-Hstretching(CH₃)'},
     {'center': 2930, 'intensity': 0.7, 'width': 20, 'label': 'C-Hstretching(CH₂)'},
     {'center': 1450, 'intensity': 0.4, 'width': 30, 'label': 'C-H bending'},
     {'center': 1380, 'intensity': 0.3, 'width': 20, 'label': 'C-H bending'},
     {'center': 1050, 'intensity': 1.0, 'width': 40, 'label': 'C-Ostretching'},
     {'center': 880, 'intensity': 0.5, 'width': 25, 'label': 'C-Cstretching'},
    ]
    
    wavenumbers, transmittance, peaks = simulate_ir_spectrum(ethanol_peaks)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(wavenumbers, transmittance, linewidth=1.5, color='#f093fb')
    ax.fill_between(wavenumbers, transmittance, 100, alpha=0.2, color='#f5576c')
    
    # Peak labels
    for peak in peaks:
     idx = np.argmin(np.abs(wavenumbers - peak['center']))
     y_pos = transmittance[idx]
     if y_pos < 50:
     ax.annotate(peak['label'], xy=(peak['center'], y_pos),
     xytext=(peak['center'], y_pos - 15),
     fontsize=9, ha='center', rotation=90,
     arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Transmittance (%)', fontsize=12)
    ax.set_title('IR Spectrum of Ethanol (CH₃CH₂OH) (Simulation)',
     fontsize=14, fontweight='bold')
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 105)
    ax.invert_xaxis() # IR spectra conventionally have high wavenumbers on the left
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ethanol_ir_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Peak assignment table
    print("=" * 70)
    print("IR Spectrum Assignment of Ethanol")
    print("=" * 70)
    print(f"{'Wavenumber (cm⁻¹)':<15} {'Intensity':<10} {'Assignment':<30}")
    print("-" * 70)
    for peak in sorted(peaks, key=lambda x: x['center'], reverse=True):
     intensity_str = 'strong' if peak['intensity'] > 1.0 else ('medium' if peak['intensity'] > 0.5 else 'weak')
     print(f"{peak['center']:<15} {intensity_str:<10} {peak['label']:<30}")
    

### 2.3 FTIR (Fourier Transform Infrared Spectroscopy)

modern IR spectrometers predominantly use FTIR (Fourier Transform Infrared Spectroscopy) based on the Michelson interferometer. The interferogram (time-domain signal) obtained from the interferometer is Fourier-transformed to obtain the frequency-domain spectrum.

**Advantages of FTIR**

  * **Fast measurement** : Simultaneous measurement of all wavenumbers (Fellgett's advantage)
  * **High sensitivity** : High light energy utilization efficiency (Jacquinot's advantage)
  * **High wavenumber accuracy** : Internal calibration using He-Ne laser
  * **Multiple averaging** : Improved S/N ratio

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 4: FTIR Interferogram and Fourier Transform</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    def generate_interferogram(wavenumbers_cm, intensities, mirror_displacement_max=0.05):
     """
     Generate interferogram from IR spectrum (simplified version)
    
     Parameters:
     -----------
     wavenumbers_cm : array
     Wavenumber (cm⁻¹)
     intensities : array
     intensity for each wavenumber
     mirror_displacement_max : float
     Maximum mirror displacement (cm)
    
     Returns:
     --------
     displacement : array
     Mirror displacement
     interferogram : array
     Interferogram
     """
     # Mirror displacement
     n_points = 2048
     displacement = np.linspace(-mirror_displacement_max, mirror_displacement_max, n_points)
    
     # Interferogram (sum of interference patterns for each wavenumber component)
     interferogram = np.zeros_like(displacement)
     for wn, intensity in zip(wavenumbers_cm, intensities):
     # Convert wavenumber from cm^-1 to wavelength in cm
     # cos(2π * wavenumber * displacement)
     interferogram += intensity * np.cos(2 * np.pi * wn * displacement)
    
     # Add DC component
     interferogram += np.sum(intensities)
    
     return displacement, interferogram
    
    def fourier_transform_spectrum(displacement, interferogram):
     """
     Fourier transform interferogram to restore spectrum
    
     Parameters:
     -----------
     displacement : array
     Mirror displacement(cm)
     interferogram : array
     Interferogram
    
     Returns:
     --------
     wavenumbers : array
     Wavenumber (cm⁻¹)
     spectrum : array
     Restored spectrum
     """
     # Fourier transform
     N = len(interferogram)
     spectrum_complex = fft(interferogram)
     spectrum = np.abs(spectrum_complex[:N//2])
    
     # Wavenumber axis(cm⁻¹)
     delta_x = displacement[1] - displacement[0]
     wavenumbers = fftfreq(N, delta_x)[:N//2]
    
     return wavenumbers, spectrum
    
    # Simulation: Three IR peaks
    true_wavenumbers = np.array([1000, 1700, 2900]) # cm^-1
    true_intensities = np.array([0.5, 1.0, 0.7])
    
    # Generate interferogram
    displacement, interferogram = generate_interferogram(true_wavenumbers, true_intensities)
    
    # Restore spectrum by Fourier transform
    wavenumbers_ft, spectrum_ft = fourier_transform_spectrum(displacement, interferogram)
    
    # visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Interferogram
    ax1.plot(displacement, interferogram, linewidth=1.5, color='#f093fb')
    ax1.set_xlabel('Mirror displacement (cm)', fontsize=12)
    ax1.set_ylabel('InterferogramIntensity', fontsize=12)
    ax1.set_title('FTIR Interferogram (Time Domain)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Path Difference')
    ax1.legend()
    
    # Spectrum after Fourier transform
    ax2.plot(wavenumbers_ft, spectrum_ft, linewidth=1.5, color='#f5576c')
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('IR Spectrum after Fourier Transform (Frequency Domain)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 4000)
    ax2.grid(True, alpha=0.3)
    
    # Mark true peak positions
    for wn, intensity in zip(true_wavenumbers, true_intensities):
     ax2.axvline(x=wn, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
     ax2.text(wn, max(spectrum_ft) * 0.9, f'{wn} cm⁻¹',
     rotation=90, va='bottom', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig('ftir_interferogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)
    print("Principles of FTIR Measurement")
    print("=" * 60)
    print("1. Move mirror in Michelson interferometer")
    print("2. Record interference pattern (interferogram)")
    print("3. Restore frequency-domain spectrum by Fourier transform")
    print("")
    print("True peak positions: ", true_wavenumbers, " cm⁻¹")
    print("Restored peaks: Confirmed in spectrum after Fourier transform")
    

## 3\. Raman Spectroscopy

### 3.1 Principles of Raman Scattering

Raman scattering is the phenomenon in which incident light (frequency $\nu_0$) is observed as scattered light shifted by the molecular vibrational energy (frequency $\nu_m$) due to light-molecule interaction:

  * **Rayleigh scattering** (elastic): $\nu_{\text{scatter}} = \nu_0$(majority, 106 timesstrong)
  * **Stokes Raman scattering** : $\nu_{\text{scatter}} = \nu_0 - \nu_m$ (molecule excited)
  * **Anti-Stokes Raman scattering** : $\nu_{\text{scatter}} = \nu_0 + \nu_m$ (molecule already in excited state returns to ground state, weak)

The selection rule for Raman scattering is that the **polarizability $\alpha$ must change** during the vibration:

$$\left(\frac{\partial \alpha}{\partial Q}\right)_0 \neq 0$$

Unlike IR, this selection rule often makes symmetric vibrations Raman-active and asymmetric vibrations Raman-inactive (complementarity rule).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 5: Raman Spectrum and Stokes/Anti-Stokes Ratio</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def boltzmann_population(E_vib_cm, T=300):
     """
     Occupancy of vibrational excited state by Boltzmann distribution
    
     Parameters:
     -----------
     E_vib_cm : float
     Vibrational energy (cm⁻¹)
     T : float
     Temperature (K)
    
     Returns:
     --------
     ratio : float
     Occupancy ratio of v=1 to v=0: n₁/n₀
     """
     k_B = 1.380649e-23 # J/K
     h = 6.62607015e-34 # J·s
     c = 2.99792458e8 # m/s
    
     E_J = h * c * E_vib_cm * 100 # cm^-1 to J
     ratio = np.exp(-E_J / (k_B * T))
     return ratio
    
    def raman_spectrum_simulation(laser_wavelength=532):
     """
     Raman spectrum (Stokes/Anti-Stokes) simulation
    
     Parameters:
     -----------
     laser_wavelength : float
     Excitation laser wavelength (nm)
    
     Returns:
     --------
     fig : matplotlib figure
     """
     # vibrational mode
     vibrations = [
     {'mode': 'C-Cstretching', 'shift': 1000, 'intensity': 0.8},
     {'mode': 'C=Ostretching', 'shift': 1700, 'intensity': 1.0},
     {'mode': 'C-Hstretching', 'shift': 2900, 'intensity': 0.6}
     ]
    
     # Laser frequency
     laser_freq = 1e7 / laser_wavelength # cm^-1
    
     # Raman shift axis (typically -3500 to +3500 cm^-1)
     raman_shift = np.linspace(-3500, 3500, 3000)
    
     # Initialize spectrum
     spectrum = np.zeros_like(raman_shift)
    
     # Peaks for each vibrational mode
     for vib in vibrations:
     shift = vib['shift']
     intensity = vib['intensity']
    
     # Stokes peak (positive shift)
     stokes = intensity * np.exp(-(raman_shift - shift)**2 / (2 * 30**2))
     spectrum += stokes
    
     # Anti-Stokes peak (negative shift)
     # BoltzmannfactorIntensity reduces
     boltzmann_ratio = boltzmann_population(shift, T=300)
     anti_stokes = intensity * boltzmann_ratio * np.exp(-(raman_shift + shift)**2 / (2 * 30**2))
     spectrum += anti_stokes
    
     # Rayleigh scattering(centrosymmetric, verystrong)
     rayleigh = 100 * np.exp(-(raman_shift)**2 / (2 * 20**2))
    
     # Plot
     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
     # Full spectrum (including Rayleigh)
     ax1.plot(raman_shift, spectrum + rayleigh, linewidth=1.5, color='#f093fb')
     ax1.fill_between(raman_shift, spectrum + rayleigh, alpha=0.3, color='#f5576c')
     ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Rayleigh scattering')
     ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
     ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
     ax1.set_title(f'Full Raman Spectrum (Laser: {laser_wavelength} nm)',
     fontsize=14, fontweight='bold')
     ax1.legend()
     ax1.grid(True, alpha=0.3)
     ax1.set_yscale('log')
     ax1.set_ylim(0.01, 150)
    
     # Enlarged Stokes region (practical measurement range)
     ax2.plot(raman_shift, spectrum, linewidth=1.5, color='#f5576c')
     ax2.fill_between(raman_shift, spectrum, alpha=0.3, color='#f093fb')
     ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
     ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
     ax2.set_title('Raman Spectrum (After Rayleigh Removal)', fontsize=14, fontweight='bold')
     ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
     ax2.grid(True, alpha=0.3)
    
     # Peak assignment
     for vib in vibrations:
     shift = vib['shift']
     ax2.text(shift, vib['intensity'] * 1.1, vib['mode'],
     ha='center', fontsize=10, rotation=45)
     ax2.text(-shift, vib['intensity'] * boltzmann_population(shift) * 1.1,
     vib['mode'] + '\n(Anti-Stokes)',
     ha='center', fontsize=9, rotation=45, alpha=0.7)
    
     plt.tight_layout()
     plt.savefig('raman_spectrum_stokes_antistokes.png', dpi=300, bbox_inches='tight')
     plt.show()
    
     # Temperature dependence of Boltzmann ratio
     print("=" * 70)
     print("Stokes/Anti-Stokesintensity ratiotemperature dependence")
     print("=" * 70)
     print(f"{'vibrational mode':<20} {'Raman shift (cm⁻¹)':<25} {'I(Anti-Stokes)/I(Stokes) at 300K':<30}")
     print("-" * 70)
     for vib in vibrations:
     ratio = boltzmann_population(vib['shift'], T=300)
     print(f"{vib['mode']:<20} {vib['shift']:<25} {ratio:<30.4f}")
    
     return fig
    
    # Execute
    fig = raman_spectrum_simulation(laser_wavelength=532)
    
    # Calculate temperature dependence
    temperatures = np.linspace(100, 800, 50)
    raman_shift_1000 = 1000 # cm^-1
    
    ratios = [boltzmann_population(raman_shift_1000, T) for T in temperatures]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, ratios, linewidth=2, color='#f093fb')
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('I(Anti-Stokes) / I(Stokes)', fontsize=12)
    plt.title('Ramanintensity ratiotemperature dependence(1000 cm⁻¹mode)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=boltzmann_population(1000, 300), color='red', linestyle='--',
     label='Room temperature (300 K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('raman_temperature_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 3.2 Complementarity of IR and Raman

centrosymmetricsymmetric molecule(e.g., CO₂, benzene), IR and Ramanselection rule ：

**Mutual Exclusion Rule**  
centrosymmetricsymmetric molecule, IR activevibration Ramaninactive, Raman activevibration IRinactive and 。This Symmetry selection rule。 

vibrational mode | Symmetry | IR active | Raman active  
---|---|---|---  
CO₂symmetric stretching | Σg⁺ | inactive | active  
CO₂asymmetric stretching | Σu⁺ | active | inactive  
CO₂bendingvibration | Πu | active | inactive  
  
### 3.3 Applications of Raman Spectroscopy

  * **crystallinityevaluation** : frequency region(<200 cm⁻¹)latticevibrational mode evaluation
  * **aqueous solutionsample** : water IRabsorption isstrong, Ramanless interference
  * **Non-contact and non-destructive measurement** : Micro-area measurement by focusing laser (Raman microscopy)
  * **surfaceenhancementRaman(SERS)** : metal nanoparticlesurface 106〜1014 timesenhancement

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 6: Raman Peak Fitting for Crystallinity Evaluation</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def crystallinity_analysis(raman_shift, intensity):
     """
     Evaluate crystallinity from Raman spectrum (polymer example)
    
     Parameters:
     -----------
     raman_shift : array
     Raman shift (cm⁻¹)
     intensity : array
     RamanIntensity
    
     Returns:
     --------
     crystallinity : float
     Crystallinity (%)
     fit_params : dict
     Fitting parameters
     """
     def two_peak_model(x, A1, c1, w1, A2, c2, w2):
     """Two-component model of crystalline and amorphous peaks"""
     peak1 = A1 * np.exp(-(x - c1)**2 / (2 * w1**2)) # Crystalline peak
     peak2 = A2 * np.exp(-(x - c2)**2 / (2 * w2**2)) # Amorphous peak
     return peak1 + peak2
    
     # Initial guess
     p0 = [100, 1095, 10, 80, 1080, 15]
    
     # Fitting
     popt, pcov = curve_fit(two_peak_model, raman_shift, intensity, p0=p0)
    
     # Individual peaks
     crystal_peak = popt[0] * np.exp(-(raman_shift - popt[1])**2 / (2 * popt[2]**2))
     amorphous_peak = popt[3] * np.exp(-(raman_shift - popt[4])**2 / (2 * popt[5]**2))
    
     # Crystallinity (peak area ratio)
     crystal_area = popt[0] * popt[2] * np.sqrt(2 * np.pi)
     amorphous_area = popt[3] * popt[5] * np.sqrt(2 * np.pi)
     crystallinity = crystal_area / (crystal_area + amorphous_area) * 100
    
     fit_params = {
     'crystal_center': popt[1],
     'crystal_width': popt[2],
     'amorphous_center': popt[4],
     'amorphous_width': popt[5],
     'crystal_peak': crystal_peak,
     'amorphous_peak': amorphous_peak,
     'fitted_curve': two_peak_model(raman_shift, *popt)
     }
    
     return crystallinity, fit_params
    
    # ー(crystallinitypolymerC-Cstretchingregion)
    raman_shift = np.linspace(1050, 1130, 300)
    crystal_peak_true = 70 * np.exp(-(raman_shift - 1095)**2 / (2 * 8**2))
    amorphous_peak_true = 50 * np.exp(-(raman_shift - 1080)**2 / (2 * 12**2))
    intensity = crystal_peak_true + amorphous_peak_true + np.random.normal(0, 2, len(raman_shift))
    
    # Crystallinity analysis
    crystallinity, fit_params = crystallinity_analysis(raman_shift, intensity)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Peak separation
    ax1.plot(raman_shift, intensity, 'k.', markersize=4, alpha=0.5, label='Experimental data')
    ax1.plot(raman_shift, fit_params['fitted_curve'], 'r-', linewidth=2, label='Fitting')
    ax1.plot(raman_shift, fit_params['crystal_peak'], 'b--', linewidth=2, label='Crystalline component')
    ax1.plot(raman_shift, fit_params['amorphous_peak'], 'g--', linewidth=2, label='Amorphous component')
    ax1.fill_between(raman_shift, fit_params['crystal_peak'], alpha=0.3, color='blue')
    ax1.fill_between(raman_shift, fit_params['amorphous_peak'], alpha=0.3, color='green')
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('Crystallinity Analysis by Peak Separation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Display crystallinity
    ax2.bar(['Crystalline component', 'Amorphous component'],
     [crystallinity, 100 - crystallinity],
     color=['#4ecdc4', '#ffe66d'], edgecolor='black', linewidth=2)
    ax2.set_ylabel('Fraction (%)', fontsize=12)
    ax2.set_title(f'Crystallinity: {crystallinity:.1f}%', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate(zip(['Crystalline component', 'Amorphous component'],
     [crystallinity, 100 - crystallinity])):
     ax2.text(i, value + 3, f'{value:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('raman_crystallinity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output results
    print("=" * 60)
    print("RamanCrystallinity analysisresults")
    print("=" * 60)
    print(f"Crystalline peakcentrosymmetric: {fit_params['crystal_center']:.1f} cm⁻¹")
    print(f"Crystalline peakwidth(FWHM): {2.355 * fit_params['crystal_width']:.1f} cm⁻¹")
    print(f"Amorphous peakcentrosymmetric: {fit_params['amorphous_center']:.1f} cm⁻¹")
    print(f"Amorphous peakwidth(FWHM): {2.355 * fit_params['amorphous_width']:.1f} cm⁻¹")
    print(f"\nCrystallinity: {crystallinity:.1f}%")
    print(f"Amorphous fraction: {100 - crystallinity:.1f}%")
    

## 4\. Group Theory and vibrational Selection Rules

### 4.1 Molecular Symmetry and Irreducible Representations

moleculevibrational mode, moleculeSymmetry(point group)。vibrational mode point groupirreducible representationcorresponds to, the Symmetryselection rule(IR active, Raman active) 。

**Examples of Major Point Groups and Irreducible Representations**  

  * **C 2v**(H₂O): A₁, A₂, B₁, B₂
  * **D 6h**(benzene): A1g, A2g, B1g, B2g, E1g, E2g, A1u, A2u, B1u, B2u, E1u, E2u
  * **T d**(CH₄): A₁, A₂, E, T₁, T₂

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 7: vibrational modes and Symmetry of H₂O Molecule</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def h2o_symmetry_analysis():
     """
     H₂Omolecule(C2vpoint group)vibrational mode and selection rule
    
     Returns:
     --------
     table : dict
     vibrational modeinformation
     """
     # H2Omolecule3fundamentalvibrational mode
     modes = {
     'mode1': {
     'name': 'symmetric stretching',
     'symmetry': 'A₁',
     'wavenumber': 3657,
     'IR_active': True,
     'Raman_active': True,
     'description': 'both O-H bonds stretching simultaneously, symmetric'
     },
     'mode2': {
     'name': 'bendingvibration',
     'symmetry': 'A₁',
     'wavenumber': 1595,
     'IR_active': True,
     'Raman_active': True,
     'description': 'H-O-H angle changes'
     },
     'mode3': {
     'name': 'asymmetric stretching',
     'symmetry': 'B₁',
     'wavenumber': 3756,
     'IR_active': True,
     'Raman_active': True,
     'description': 'When one O-H stretches, the other contracts'
     }
     }
    
     # Display in table format
     print("=" * 80)
     print("H₂Omolecule(C₂vpoint group)vibrational mode and selection rule")
     print("=" * 80)
     print(f"{'mode':<12} {'Symmetry':<10} {'Wavenumber (cm⁻¹)':<15} {'IR active':<10} {'Raman active':<12} {'Description':<30}")
     print("-" * 80)
    
     for mode_id, mode in modes.items():
     ir_str = '○' if mode['IR_active'] else '×'
     raman_str = '○' if mode['Raman_active'] else '×'
     print(f"{mode['name']:<12} {mode['symmetry']:<10} {mode['wavenumber']:<15} "
     f"{ir_str:<10} {raman_str:<12} {mode['description']:<30}")
    
     print("\n" + "=" * 80)
     print("Character Table of C₂v Point Group")
     print("=" * 80)
     print(" C₂v | E C₂ σv σv' | Basis functions")
     print("-" * 80)
     print(" A₁ | 1 1 1 1 | z, x², y², z²")
     print(" A₂ | 1 1 -1 -1 | Rz")
     print(" B₁ | 1 -1 1 -1 | x, Ry")
     print(" B₂ | 1 -1 -1 1 | y, Rx")
     print("\nSelection rules:")
     print(" IR active: μx, μy, μz(dipole moment) included in basis")
     print(" Raman active: αxx, αyy, αzz, αxy, αxz, αyz(polarizability tensor) included in basis")
     print("\nH₂Ocase, A₁ and B₁ bothIR and Raman active")
    
     # visualization：energy level diagram
     fig, ax = plt.subplots(figsize=(10, 8))
    
     # Ground state and excited states
     y_ground = 0
     modes_sorted = sorted(modes.items(), key=lambda x: x[1]['wavenumber'])
    
     colors = ['#f093fb', '#f5576c', '#4ecdc4']
    
     for i, (mode_id, mode) in enumerate(modes_sorted):
     y_excited = mode['wavenumber'] / 100 # Scaling
     ax.hlines(y_excited, i*0.5, i*0.5 + 0.4, colors=colors[i], linewidths=5,
     label=f"{mode['name']} ({mode['symmetry']})")
     ax.text(i*0.5 + 0.45, y_excited, f"{mode['wavenumber']} cm⁻¹",
     va='center', fontsize=10)
    
     # Transition arrows
     ax.annotate('', xy=(i*0.5 + 0.2, y_excited), xytext=(i*0.5 + 0.2, y_ground),
     arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
     ax.hlines(y_ground, -0.2, 1.5, colors='black', linewidths=3, label='Ground state (v=0)')
     ax.set_xlim(-0.3, 1.6)
     ax.set_ylim(-2, 40)
     ax.set_ylabel('Relative Energy (cm⁻¹ / 100)', fontsize=12)
     ax.set_title('H₂Omoleculevibrationexcitationーlevel', fontsize=14, fontweight='bold')
     ax.set_xticks([])
     ax.legend(loc='upper left', fontsize=10)
     ax.grid(axis='y', alpha=0.3)
    
     plt.tight_layout()
     plt.savefig('h2o_symmetry_modes.png', dpi=300, bbox_inches='tight')
     plt.show()
    
     return modes
    
    # Execute
    modes = h2o_symmetry_analysis()
    

### 4.2 Determination of Selection Rules

irreducible representation correspondingvibrational mode IR active Raman active, rule ：

  * **IR active** : vibrational modeirreducible representation, dipole momentcomponent(x, y, z)any of and Symmetry 
  * **Raman active** : vibrational modeirreducible representation, polarizability tensorcomponent(x², y², z², xy, xz, yz)any of and Symmetry 

## 5\. Practical Spectral Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
     <h4>Code Example 8: Integrated Analysis Workflow for IR and Raman</h4>
     <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    class vibrationalSpectroscopyAnalyzer:
     """Integrated analysis class for IR and Raman spectra"""
    
     def __init__(self):
     # Functional Groupdatabase(simplified version)
     self.functional_groups = {
     'O-H': {'IR': (3200, 3600), 'Raman': (3200, 3600), 'intensity_IR': 'strong'},
     'N-H': {'IR': (3300, 3500), 'Raman': (3300, 3500), 'intensity_IR': 'medium'},
     'C-H': {'IR': (2850, 3000), 'Raman': (2850, 3000), 'intensity_IR': 'strong'},
     'C=O': {'IR': (1650, 1750), 'Raman': (1650, 1750), 'intensity_IR': 'very strong'},
     'C=C': {'IR': (1620, 1680), 'Raman': (1620, 1680), 'intensity_Raman': 'strong'},
     'C-C': {'IR': (800, 1200), 'Raman': (800, 1200), 'intensity_Raman': 'medium'},
     }
    
     def identify_functional_groups(self, wavenumbers, intensity, threshold=0.3):
     """
     Functional Groupidentify
    
     Parameters:
     -----------
     wavenumbers : array
     Wavenumber (cm⁻¹)
     intensity : array
     Intensity
     threshold : float
     Peak detection threshold (relative to maximum)
    
     Returns:
     --------
     identified_groups : list
     Functional Group
     """
     # Peak detection
     peaks, properties = find_peaks(intensity, prominence=threshold * np.max(intensity))
    
     identified_groups = []
    
     for peak in peaks:
     peak_wn = wavenumbers[peak]
    
     # Functional Groupdatabasematch with
     for group, ranges in self.functional_groups.items():
     ir_range = ranges['IR']
     if ir_range[0] <= peak_wn <= ir_range[1]:
     identified_groups.append({
     'functional_group': group,
     'wavenumber': peak_wn,
     'intensity': intensity[peak]
     })
    
     return identified_groups
    
     def complementary_analysis(self, ir_spectrum, raman_spectrum):
     """
     Complementary analysis of IR and Raman
    
     Parameters:
     -----------
     ir_spectrum : dict
     {'wavenumbers': array, 'intensity': array}
     raman_spectrum : dict
     {'wavenumbers': array, 'intensity': array}
    
     Returns:
     --------
     analysis_result : dict
     Integrated analysis results
     """
     # IR Functional Group
     ir_groups = self.identify_functional_groups(ir_spectrum['wavenumbers'],
     ir_spectrum['intensity'])
    
     # Raman Functional Group
     raman_groups = self.identify_functional_groups(raman_spectrum['wavenumbers'],
     raman_spectrum['intensity'])
    
     # Integration
     all_groups = set([g['functional_group'] for g in ir_groups] +
     [g['functional_group'] for g in raman_groups])
    
     analysis_result = {
     'IR_only': [g for g in ir_groups if g['functional_group'] not in
     [rg['functional_group'] for rg in raman_groups]],
     'Raman_only': [g for g in raman_groups if g['functional_group'] not in
     [ig['functional_group'] for ig in ir_groups]],
     'Both': list(all_groups.intersection(set([g['functional_group'] for g in ir_groups]),
     set([g['functional_group'] for g in raman_groups])))
     }
    
     return analysis_result
    
    # Execute：(CH₃COCH₃)IR・RamanIntegrationanalysis
    analyzer = vibrationalSpectroscopyAnalyzer()
    
    # Synthetic IR spectrum
    wn_ir = np.linspace(4000, 400, 2000)
    ir_intensity = (
     1.5 * np.exp(-(wn_ir - 2970)**2 / (2 * 20**2)) + # C-Hstretching
     2.0 * np.exp(-(wn_ir - 1715)**2 / (2 * 30**2)) + # C=Ostretching
     0.5 * np.exp(-(wn_ir - 1360)**2 / (2 * 25**2)) + # C-H bending
     0.3 * np.random.random(len(wn_ir)) # Noise
    )
    
    # Synthetic Raman spectrum
    wn_raman = np.linspace(3500, 100, 2000)
    raman_intensity = (
     0.8 * np.exp(-(wn_raman - 2970)**2 / (2 * 20**2)) + # C-Hstretching
     1.2 * np.exp(-(wn_raman - 1715)**2 / (2 * 30**2)) + # C=Ostretching
     1.5 * np.exp(-(wn_raman - 900)**2 / (2 * 25**2)) + # C-Cstretching
     0.2 * np.random.random(len(wn_raman)) # Noise
    )
    
    # Integrationanalysis
    ir_spec = {'wavenumbers': wn_ir, 'intensity': ir_intensity}
    raman_spec = {'wavenumbers': wn_raman, 'intensity': raman_intensity}
    
    result = analyzer.complementary_analysis(ir_spec, raman_spec)
    
    # visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # IR spectrum
    ax1.plot(wn_ir, ir_intensity, linewidth=1.5, color='#f093fb', label='IR spectrum')
    ax1.fill_between(wn_ir, ir_intensity, alpha=0.3, color='#f5576c')
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Absorbance (a.u.)', fontsize=12)
    ax1.set_title('IR Spectrum of Acetone', fontsize=14, fontweight='bold')
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Raman spectrum
    ax2.plot(wn_raman, raman_intensity, linewidth=1.5, color='#4ecdc4', label='Raman spectrum')
    ax2.fill_between(wn_raman, raman_intensity, alpha=0.3, color='#ffe66d')
    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('Raman Spectrum of Acetone', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acetone_ir_raman_complementary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display analysis results
    print("=" * 70)
    print("IR and RamanIntegrated analysis results()")
    print("=" * 70)
    print("\nDetected by IR only:")
    for group in result['IR_only']:
     print(f" {group['functional_group']}: {group['wavenumber']:.0f} cm⁻¹")
    
    print("\nDetected by Raman only:")
    for group in result['Raman_only']:
     print(f" {group['functional_group']}: {group['wavenumber']:.0f} cm⁻¹")
    
    print("\nDetected by both:")
    for group in result['Both']:
     print(f" {group}")
    
    print("\nConclusion:")
    print(" - C=Ostretching: IR verystrong, Raman ")
    print(" - C-Hstretching: IR・Raman strong")
    print(" - C-Cstretching: Raman strong(IR)")
    print(" → By combining IR and Raman, comprehensive understanding of molecular structure is achieved")
    

## Exercise Problems

**Exercise Problems (Click to Expand)**

### Easy Level (Basic Calculations)

**1** : C-Obond( $k = 1000$ N/m)vibrationWavenumber (cm⁻¹) 。 12 amu, 16 amu。

View Solution

**Solution** :
    
    
    # Use function from Code Example 1
    freq_Hz, wavenumber = vibrational_frequency(k=1000, m1=12, m2=16)
    print(f"C-Ostretchingvibration: {wavenumber:.1f} cm⁻¹")
    # Output: Approximately 1270 cm⁻¹
    

**2** : H₂Omolecule(, 3)vibration 。

View Solution

**Solution** :

$$3N - 6 = 3 \times 3 - 6 = 3$$

H₂O 3vibrational mode(symmetric stretching, bendingvibration, asymmetric stretching) 。

**3** : IR spectrum 1715 cm⁻¹ strongー 。This Functional Group ？

View Solution

****: (C=O)stretchingvibration。1650-1750 cm⁻¹region C=O。

### medium Level (Practical Calculations)

**4** : Isotope effect, ¹²C=O and ¹³C=O vibrational frequency ( and )。

View Solution

**Solution** :
    
    
    _, wn_12C = vibrational_frequency(1200, 12, 16)
    _, wn_13C = vibrational_frequency(1200, 13, 16)
    
    ratio = wn_12C / wn_13C
    print(f"¹²C=O wavenumber: {wn_12C:.1f} cm⁻¹")
    print(f"¹³C=O wavenumber: {wn_13C:.1f} cm⁻¹")
    print(f"Ratio: {ratio:.4f}")
    # Output: Ratio ≈ 1.017 (approximately 1.7% shift)
    

**5** : Raman, (300 K) Stokes and Anti-Stokesintensity ratio 。vibrational mode 1500 cm⁻¹ and 。

View Solution

**Solution** :
    
    
    # Use boltzmann_population function from Code Example 5
    ratio = boltzmann_population(1500, T=300)
    print(f"I(Anti-Stokes) / I(Stokes) = {ratio:.4f}")
    # Output: Approximately 0.023 (Anti-Stokes is about 2.3% of Stokes)
    

**6** : CO₂molecule(, 3)vibration, vibrational modeSymmetry(Σg⁺, Σu⁺, Πu) and IR/Raman active 。

View Solution

**Solution** :

$$3N - 5 = 3 \times 3 - 5 = 4$$

mode| Symmetry| Wavenumber (cm⁻¹)| IR active| Raman active  
---|---|---|---|---  
symmetric stretching| Σg⁺| 1340| inactive| active  
asymmetric stretching| Σu⁺| 2349| active| inactive  
bendingvibration(2)| Πu| 667| active| inactive  
  
CO₂ centrosymmetricsymmetric molecule, 。

### Hard Level (Advanced Analysis)

**7** : IR spectrumー, FTIRInterferogram 。the, Fourier transform restoreThis and 。

View Solution(ー)
    
    
    # Use functions from Code Example 4
    true_wavenumbers = np.array([1000, 1500, 2000, 2900])
    true_intensities = np.array([0.6, 1.0, 0.8, 0.7])
    
    # Generate interferogram
    displacement, interferogram = generate_interferogram(true_wavenumbers, true_intensities,
     mirror_displacement_max=0.1)
    
    # Restore spectrum by Fourier transform
    wavenumbers_ft, spectrum_ft = fourier_transform_spectrum(displacement, interferogram)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(displacement, interferogram, linewidth=1.5, color='#f093fb')
    ax1.set_xlabel('Mirror displacement (cm)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('IR spectrumInterferogram', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wavenumbers_ft, spectrum_ft, linewidth=1.5, color='#f5576c')
    for wn in true_wavenumbers:
     ax2.axvline(x=wn, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('Fourier transform Restored spectrum', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 4000)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("True peak positions:", true_wavenumbers)
    print("restore: Fourier transform ")
    

**8** : benzene(D6hpoint group)vibrational mode(30), IR activemode and Raman activemode 。Symmetry 。

View Solution

**Solution** :

benzene centrosymmetricsymmetric molecule(D6h), 。30vibrational mode：

  * **IR active** : u(ungerade)Symmetrymode → 4E1umode
  * **Raman active** : g(gerade)Symmetrymode → 7A1g, E1g, E2gmode
  * mode inactive(A2g, A2u, B1u, B2u)

analysis, IR activemode and Raman activemode 。

**9** : Raman spectrumevaluation, Crystalline peak(1130 cm⁻¹) and Amorphous peak(1080 cm⁻¹)intensity ratio 2:1 。ー (ーwidth the 10 cm⁻¹, 15 cm⁻¹ and )。

View Solution

**Solution** :
    
    
    # Calculate peak area with Gaussian approximation
    I_crystal = 2.0 # intensity ratio
    I_amorphous = 1.0
    width_crystal = 10 # cm^-1
    width_amorphous = 15 # cm^-1
    
    # = Intensity × width × sqrt(2π)(Gaussian)
    area_crystal = I_crystal * width_crystal * np.sqrt(2 * np.pi)
    area_amorphous = I_amorphous * width_amorphous * np.sqrt(2 * np.pi)
    
    crystallinity = area_crystal / (area_crystal + area_amorphous) * 100
    print(f"Crystalline peak: {area_crystal:.1f}")
    print(f"Amorphous peak: {area_amorphous:.1f}")
    print(f"Crystallinity: {crystallinity:.1f}%")
    # Output: Approximately 51.3%
    

**10** : H₂OmoleculeC2vpoint group 3vibrational mode(A₁, A₁, B₁), IR and Raman active, Description。

View Solution

**Solution** :

From the character table of C2v point group:

  * **A₁Symmetry** : Basis functions z(dipole moment) and x², y², z²(polarizability tensor) → IR activeRaman active
  * **B₁Symmetry** : Basis functions x(dipole moment) and xz(polarizability tensor) → IR activeRaman active

, H₂O3vibrational mode(symmetric stretchingA₁, bendingA₁, asymmetric stretchingB₁) IR and Raman 。centrosymmetricsymmetric molecule, 。

## Learning Objectives Review

Review what you learned in this chapter and check the following items.

### Basic Understanding

  * ✅ vibration and vibrational frequency・ Description 
  * ✅ Understand the difference in selection rules for IR and Raman (dipole moment change vs. polarizability change)
  * ✅ Functional Group (C=O: 1700 cm⁻¹, O-H: 3400 cm⁻¹)
  * ✅ centrosymmetricsymmetric molecule Description 

### Practical Skills

  * ✅ vibrational frequency and Isotope effectevaluation 
  * ✅ IR spectrumFunctional Groupidentify 
  * ✅ Raman spectrumStokes/Anti-Stokesintensity ratio 
  * ✅ evaluationPeak separation 

### Application Skills

  * ✅ Can determine molecular structure by combining complementary information from IR and Raman
  * ✅ vibrational modeSymmetry and IR/Raman active 
  * ✅ FTIR(Interferogram and Fourier transform), 

## References

  1. Raman, C. V., Krishnan, K. S. (1928). A new type of secondary radiation. _Nature_ , 121(3048), 501-502. DOI: 10.1038/121501c0 - Historic original paper reporting the discovery of Raman scattering effect
  2. Nakamoto, K. (2008). _Infrared and Raman Spectra of Inorganic and Coordination Compounds_ (6th ed.). Wiley, pp. 25-31 (IR theory), pp. 78-95 (Raman theory), pp. 115-140 (group theory applications). - IR・Raman spectrum and Functional GroupAssignment
  3. Long, D. A. (2002). _The Raman Effect: A Unified Treatment of the Theory of Raman Scattering by Molecules_. Wiley, pp. 50-68 (classical theory), pp. 95-115 (quantum theory), pp. 145-160 (selection rules). - Quantum mechanical theory and selection rules of Raman scattering
  4. Wilson, E. B., Decius, J. C., Cross, P. C. (1980). _Molecular vibrations: The Theory of Infrared and Raman vibrational Spectra_. Dover Publications, pp. 25-42 (normal modes), pp. 65-85 (group theory), pp. 95-110 (selection rules). - moleculevibration, and selection rule
  5. Colthup, N. B., Daly, L. H., Wiberley, S. E. (1990). _Introduction to Infrared and Raman Spectroscopy_ (3rd ed.). Academic Press, pp. 100-125 (functional group frequencies), pp. 180-210 (spectral interpretation), pp. 220-240 (peak assignment). - and ーAssignment
  6. Savitzky, A., Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. _Analytical Chemistry_ , 36(8), 1627-1639. DOI: 10.1021/ac60214a047 - Original paper on Savitzky-Golay smoothing filter (used in code examples)
  7. SciPy 1.11 Signal Processing Documentation. _scipy.signal.find_peaks, scipy.signal.savgol_filter_. Available at: https://docs.scipy.org/doc/scipy/reference/signal.html - Spectral data processing using Python
  8. Smith, E., Dent, G. (2019). _modern Raman Spectroscopy: A Practical Approach_ (2nd ed.). Wiley, pp. 15-28 (instrumentation), pp. 45-65 (sampling techniques), pp. 72-80 (data processing). - Practical techniques of modern Raman spectroscopy
  9. Cotton, F. A. (1990). _Chemical Applications of Group Theory_ (3rd ed.). Wiley, pp. 250-275 (point groups), pp. 285-305 (vibrational modes), pp. 310-320 (selection rules). - and vibrationSymmetryanalysis

## Next Steps

2, ・, selection rule, Functional Group, Symmetryanalysis 。vibration, FTIR, Stokes/Anti-Stokes, evaluation, ーanalysis。

**Chapter 3** will cover UV-Vis (ultraviolet-visible) spectroscopy. We will cover everything about electronic state analysis of semiconductors and organic materials, including electronic transitions, applications of Lambert-Beer law, band gap measurement by Tauc plot, ligand field theory, and absorption spectrum analysis using Python.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
