---
title: "Chapter 4: Raman Spectroscopy"
chapter_title: "Chapter 4: Raman Spectroscopy"
version: 1.0
created_at: 2025-12-26
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/n6yPS5_-4bA"
    title="Spectroscopy Introduction Ch.4: Raman Spectroscopy"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Spectroscopy](<index.html>):Chapter 4

EN | [JP](<../../../jp/MS/spectroscopy-introduction/chapter-4.html>) | Last sync: 2025-12-26

# Chapter 4: Raman Spectroscopy

**What you will learn in this chapter:** Raman spectroscopy is a powerful vibrational spectroscopy technique based on inelastic light scattering. Unlike infrared spectroscopy, Raman probes changes in molecular polarizability, making it complementary to IR for complete vibrational analysis. In this chapter, you will learn the physics of Raman scattering (Stokes and anti-Stokes), selection rules based on polarizability changes, the complementary relationship between Raman and IR spectroscopy, Surface-Enhanced Raman Spectroscopy (SERS), and practical applications in carbon materials analysis including graphene and carbon nanotubes. You will also develop Python programming skills for Raman spectral analysis, D/G band fitting, and peak deconvolution.

### Learning Objectives

  * Understand the physical mechanism of Raman scattering and distinguish between Stokes, anti-Stokes, and Rayleigh scattering
  * Apply Raman selection rules based on polarizability changes
  * Explain the complementary relationship between Raman and IR spectroscopy
  * Describe the principles and applications of Surface-Enhanced Raman Spectroscopy (SERS)
  * Interpret Raman spectra of carbon materials (D, G, 2D bands)
  * Implement Python code for Raman spectral analysis and peak deconvolution

## 4.1 Fundamentals of Raman Scattering

### 4.1.1 The Raman Effect

Raman spectroscopy is based on the inelastic scattering of light, discovered by C.V. Raman in 1928. When monochromatic light interacts with a molecule, most photons are elastically scattered (Rayleigh scattering) with no change in energy. However, a small fraction (approximately 1 in 10 million photons) undergoes inelastic scattering, where energy is exchanged between the photon and molecular vibrations.

**Energy Conservation in Raman Scattering:**

\\[ E_{\text{scattered}} = E_{\text{incident}} \pm E_{\text{vibration}} \\] 

where the plus sign corresponds to anti-Stokes scattering and the minus sign to Stokes scattering.

**Raman Shift (in wavenumbers):**

\\[ \Delta \tilde{\nu} = \tilde{\nu}_{\text{incident}} - \tilde{\nu}_{\text{scattered}} = \frac{1}{\lambda_{\text{incident}}} - \frac{1}{\lambda_{\text{scattered}}} \\] 

The Raman shift is expressed in cm$^{-1}$ and is independent of the excitation wavelength, depending only on the vibrational energy of the molecule.

### 4.1.2 Stokes, Anti-Stokes, and Rayleigh Scattering

The three types of light scattering can be understood through energy level diagrams:
    
    
    ```mermaid
    graph TD
        subgraph "Rayleigh Scattering"
        A1[Ground State v=0] -->|Incident photon| B1[Virtual State]
        B1 -->|Scattered photon<br/>same energy| A1
        end
    
        subgraph "Stokes Raman"
        A2[Ground State v=0] -->|Incident photon| B2[Virtual State]
        B2 -->|Scattered photon<br/>lower energy| C2[Excited State v=1]
        end
    
        subgraph "Anti-Stokes Raman"
        A3[Excited State v=1] -->|Incident photon| B3[Virtual State]
        B3 -->|Scattered photon<br/>higher energy| C3[Ground State v=0]
        end
    ```

#### Comparison of Scattering Types

Property | Rayleigh | Stokes | Anti-Stokes  
---|---|---|---  
Energy change | None | Photon loses energy | Photon gains energy  
Wavelength shift | None | Longer wavelength (red shift) | Shorter wavelength (blue shift)  
Initial state | v = 0 | v = 0 | v = 1 (or higher)  
Final state | v = 0 | v = 1 | v = 0  
Intensity | Very strong | Weak | Very weak  
Temperature dependence | Minimal | Minimal | Strong (Boltzmann)  
  
#### Code Example 1: Simulating Stokes and Anti-Stokes Spectra
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def boltzmann_ratio(delta_E_cm, temperature):
        """
        Calculate the anti-Stokes to Stokes intensity ratio using Boltzmann distribution.
    
        Parameters:
        -----------
        delta_E_cm : float
            Vibrational energy in cm^-1
        temperature : float
            Temperature in Kelvin
    
        Returns:
        --------
        ratio : float
            Anti-Stokes / Stokes intensity ratio
        """
        # Physical constants
        h = 6.62607015e-34  # J*s
        c = 2.99792458e10   # cm/s
        k_B = 1.380649e-23  # J/K
    
        # Energy in Joules
        delta_E_J = h * c * delta_E_cm
    
        # Boltzmann ratio
        ratio = np.exp(-delta_E_J / (k_B * temperature))
        return ratio
    
    def lorentzian(x, center, width, amplitude):
        """Generate Lorentzian peak."""
        return amplitude * (width**2) / ((x - center)**2 + width**2)
    
    # Simulation parameters
    raman_shift = np.linspace(-1800, 1800, 1000)  # cm^-1
    temperature = 300  # Kelvin
    
    # Define vibrational modes (example: organic molecule)
    modes = [
        {'shift': 500, 'width': 15, 'amplitude': 0.3},
        {'shift': 1000, 'width': 20, 'amplitude': 0.5},
        {'shift': 1450, 'width': 25, 'amplitude': 1.0},
    ]
    
    # Generate Stokes and anti-Stokes spectra
    stokes_spectrum = np.zeros_like(raman_shift)
    anti_stokes_spectrum = np.zeros_like(raman_shift)
    
    for mode in modes:
        shift = mode['shift']
        width = mode['width']
        amp = mode['amplitude']
    
        # Stokes peaks (positive shift)
        stokes_spectrum += lorentzian(raman_shift, shift, width, amp)
    
        # Anti-Stokes peaks (negative shift, intensity reduced by Boltzmann factor)
        ratio = boltzmann_ratio(shift, temperature)
        anti_stokes_spectrum += lorentzian(raman_shift, -shift, width, amp * ratio)
    
    # Add Rayleigh scattering (central peak)
    rayleigh = lorentzian(raman_shift, 0, 5, 50)
    
    # Total spectrum
    total_spectrum = stokes_spectrum + anti_stokes_spectrum + rayleigh
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full spectrum
    ax1 = axes[0]
    ax1.fill_between(raman_shift[raman_shift < 0], anti_stokes_spectrum[raman_shift < 0],
                      alpha=0.5, color='blue', label='Anti-Stokes')
    ax1.fill_between(raman_shift[raman_shift > 0], stokes_spectrum[raman_shift > 0],
                      alpha=0.5, color='red', label='Stokes')
    ax1.plot(raman_shift, rayleigh, 'g-', linewidth=2, label='Rayleigh')
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('Raman Spectrum: Stokes, Anti-Stokes, and Rayleigh Scattering', fontsize=14)
    ax1.legend()
    ax1.set_xlim(-1800, 1800)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Temperature dependence of anti-Stokes/Stokes ratio
    ax2 = axes[1]
    temperatures = np.linspace(100, 600, 100)
    mode_shifts = [500, 1000, 1500]
    
    for shift in mode_shifts:
        ratios = [boltzmann_ratio(shift, T) for T in temperatures]
        ax2.plot(temperatures, ratios, linewidth=2, label=f'{shift} cm$^{{-1}}$')
    
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Anti-Stokes / Stokes Ratio', fontsize=12)
    ax2.set_title('Temperature Dependence of Anti-Stokes/Stokes Intensity Ratio', fontsize=14)
    ax2.legend(title='Vibrational Mode')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('raman_stokes_antistokes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print intensity ratios at room temperature
    print("Anti-Stokes/Stokes intensity ratios at 300 K:")
    print("-" * 40)
    for shift in mode_shifts:
        ratio = boltzmann_ratio(shift, 300)
        print(f"  {shift} cm^-1: {ratio:.4f} ({ratio*100:.2f}%)")
    

## 4.2 Raman Selection Rules

### 4.2.1 Polarizability and Raman Activity

The fundamental selection rule for Raman spectroscopy is based on the change in molecular polarizability during vibration. A vibration is Raman-active if the polarizability changes during the vibration:

**Raman Selection Rule:**

\\[ \left(\frac{\partial \alpha}{\partial Q}\right)_{Q=0} \neq 0 \\] 

where \\(\alpha\\) is the polarizability tensor and \\(Q\\) is the normal coordinate of vibration.

**Induced Dipole Moment:**

\\[ \vec{\mu}_{\text{induced}} = \alpha \cdot \vec{E} \\] 

The polarizability \\(\alpha\\) is a tensor that relates the induced dipole moment to the electric field of the incident light.

### 4.2.2 Comparison with IR Selection Rules

#### IR vs. Raman Selection Rules

Aspect | IR Spectroscopy | Raman Spectroscopy  
---|---|---  
Physical basis | Change in dipole moment | Change in polarizability  
Selection rule | \\(\left(\frac{\partial \mu}{\partial Q}\right) \neq 0\\) | \\(\left(\frac{\partial \alpha}{\partial Q}\right) \neq 0\\)  
Symmetric vibrations | Often IR-inactive | Often Raman-active  
Asymmetric vibrations | Often IR-active | Often Raman-inactive  
Centrosymmetric molecules | g modes: inactive | g modes: active  
Water interference | Strong | Weak  
  
For molecules with a center of symmetry (centrosymmetric), the mutual exclusion rule applies: vibrations that are Raman-active are IR-inactive and vice versa. This makes IR and Raman spectroscopy complementary techniques.

#### Code Example 2: Comparing IR and Raman Activity for CO2
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def gaussian(x, center, width, amplitude):
        """Generate Gaussian peak."""
        return amplitude * np.exp(-((x - center)**2) / (2 * width**2))
    
    # CO2 vibrational modes
    # Mode | Wavenumber | Symmetry | IR Active | Raman Active
    # --------------------------------------------------------
    # Symmetric stretch | 1388 cm^-1 | Sigma_g+ | No | Yes
    # Asymmetric stretch | 2349 cm^-1 | Sigma_u+ | Yes | No
    # Bending (x2) | 667 cm^-1 | Pi_u | Yes | No
    
    wavenumber = np.linspace(400, 2600, 1000)
    
    # IR spectrum (only asymmetric stretch and bending are active)
    ir_spectrum = np.zeros_like(wavenumber)
    ir_spectrum += gaussian(wavenumber, 667, 20, 0.6)   # Bending mode
    ir_spectrum += gaussian(wavenumber, 2349, 25, 1.0)  # Asymmetric stretch
    
    # Raman spectrum (only symmetric stretch is active)
    raman_spectrum = np.zeros_like(wavenumber)
    raman_spectrum += gaussian(wavenumber, 1388, 20, 1.0)  # Symmetric stretch
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # IR spectrum
    ax1 = axes[0]
    ax1.plot(wavenumber, ir_spectrum, 'b-', linewidth=2)
    ax1.fill_between(wavenumber, ir_spectrum, alpha=0.3, color='blue')
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.set_title('CO$_2$ IR Spectrum (Asymmetric Stretch and Bending Active)', fontsize=14)
    ax1.annotate('Bending\n667 cm$^{-1}$', xy=(667, 0.6), xytext=(500, 0.8),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax1.annotate('Asymmetric Stretch\n2349 cm$^{-1}$', xy=(2349, 1.0), xytext=(2100, 0.8),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Raman spectrum
    ax2 = axes[1]
    ax2.plot(wavenumber, raman_spectrum, 'r-', linewidth=2)
    ax2.fill_between(wavenumber, raman_spectrum, alpha=0.3, color='red')
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('CO$_2$ Raman Spectrum (Symmetric Stretch Active)', fontsize=14)
    ax2.annotate('Symmetric Stretch\n1388 cm$^{-1}$', xy=(1388, 1.0), xytext=(1600, 0.8),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Molecular vibrations diagram
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 4)
    
    # Draw CO2 molecule and vibration modes
    def draw_co2(ax, x, y, mode_name, arrows=None):
        # Atoms
        ax.plot(x, y, 'ko', markersize=20)  # Central C
        ax.plot(x-1, y, 'ro', markersize=25)  # Left O
        ax.plot(x+1, y, 'ro', markersize=25)  # Right O
        ax.text(x, y, 'C', ha='center', va='center', fontsize=10, color='white')
        ax.text(x-1, y, 'O', ha='center', va='center', fontsize=10, color='white')
        ax.text(x+1, y, 'O', ha='center', va='center', fontsize=10, color='white')
        ax.text(x, y-0.5, mode_name, ha='center', fontsize=10)
    
        # Draw arrows if provided
        if arrows:
            for arrow in arrows:
                ax.annotate('', xy=(x+arrow[0]+arrow[2], y+arrow[3]),
                           xytext=(x+arrow[0], y),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Symmetric stretch (Raman active)
    draw_co2(ax3, 2, 3, 'Symmetric Stretch\n(Raman Active)',
             [(-1, 0, -0.3, 0), (1, 0, 0.3, 0)])
    
    # Asymmetric stretch (IR active)
    draw_co2(ax3, 5, 3, 'Asymmetric Stretch\n(IR Active)',
             [(-1, 0, -0.3, 0), (1, 0, -0.3, 0)])
    
    # Bending (IR active)
    draw_co2(ax3, 8, 3, 'Bending\n(IR Active)',
             [(0, 0, 0, 0.3)])
    
    ax3.set_title('CO$_2$ Vibrational Modes and Selection Rules (Mutual Exclusion)', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(400, 2600)
        ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('ir_raman_comparison_co2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\nCO2 Vibrational Modes Summary:")
    print("=" * 70)
    print(f"{'Mode':<25} {'Wavenumber':>12} {'Symmetry':>12} {'IR':>8} {'Raman':>8}")
    print("-" * 70)
    print(f"{'Symmetric stretch':<25} {'1388 cm^-1':>12} {'Sigma_g+':>12} {'No':>8} {'Yes':>8}")
    print(f"{'Asymmetric stretch':<25} {'2349 cm^-1':>12} {'Sigma_u+':>12} {'Yes':>8} {'No':>8}")
    print(f"{'Bending (2x degenerate)':<25} {'667 cm^-1':>12} {'Pi_u':>12} {'Yes':>8} {'No':>8}")
    print("=" * 70)
    print("\nNote: CO2 is centrosymmetric - mutual exclusion rule applies")
    

## 4.3 IR and Raman Complementarity

### 4.3.1 Practical Advantages of Each Technique

#### When to Use Raman vs. IR

Application | Preferred Technique | Reason  
---|---|---  
Aqueous solutions | Raman | Water has weak Raman scattering  
Symmetric vibrations | Raman | Strong polarizability change  
C=C, C-C, S-S bonds | Raman | Highly polarizable bonds  
Polar functional groups | IR | Strong dipole moment change  
O-H, N-H, C=O groups | IR | Large dipole derivatives  
In situ measurements | Raman | Fiber optic probes available  
Low wavenumber region | Raman | No cutoff from optics  
  
#### Code Example 3: Combined IR-Raman Analysis of Polymers
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    # - scipy>=1.10.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def generate_polymer_spectrum(wavenumber, peaks, widths, intensities, noise_level=0.02):
        """
        Generate synthetic polymer spectrum with noise.
    
        Parameters:
        -----------
        wavenumber : array
            Wavenumber axis
        peaks : list
            Peak positions in cm^-1
        widths : list
            Peak widths (FWHM)
        intensities : list
            Peak intensities
        noise_level : float
            Standard deviation of Gaussian noise
    
        Returns:
        --------
        spectrum : array
            Generated spectrum with noise
        """
        spectrum = np.zeros_like(wavenumber)
        for peak, width, intensity in zip(peaks, widths, intensities):
            spectrum += intensity * np.exp(-((wavenumber - peak)**2) / (2 * (width/2.355)**2))
        spectrum += np.random.normal(0, noise_level, len(wavenumber))
        return np.maximum(spectrum, 0)
    
    # Wavenumber range
    wavenumber = np.linspace(400, 3500, 1000)
    
    # Polyethylene (PE) characteristic peaks
    # IR-active modes
    pe_ir_peaks = [720, 1465, 2850, 2920]  # CH2 rocking, CH2 scissor, CH2 sym stretch, CH2 asym stretch
    pe_ir_widths = [30, 25, 40, 40]
    pe_ir_intensities = [0.4, 0.6, 0.9, 1.0]
    
    # Raman-active modes
    pe_raman_peaks = [1130, 1295, 1440, 2850, 2885]  # C-C stretch, CH2 twist, CH2 scissor, CH2 sym, CH2 asym
    pe_raman_widths = [25, 20, 25, 35, 35]
    pe_raman_intensities = [0.7, 0.5, 0.4, 1.0, 0.8]
    
    # Generate spectra
    ir_spectrum = generate_polymer_spectrum(wavenumber, pe_ir_peaks, pe_ir_widths, pe_ir_intensities)
    raman_spectrum = generate_polymer_spectrum(wavenumber, pe_raman_peaks, pe_raman_widths, pe_raman_intensities)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # IR spectrum
    ax1 = axes[0]
    ax1.plot(wavenumber, ir_spectrum, 'b-', linewidth=1.5)
    ax1.fill_between(wavenumber, ir_spectrum, alpha=0.3, color='blue')
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.set_title('Polyethylene (PE) - IR Spectrum', fontsize=14, fontweight='bold')
    
    # Annotate IR peaks
    for peak, label in zip(pe_ir_peaks, ['CH$_2$ rocking', 'CH$_2$ scissor', 'CH$_2$ sym str.', 'CH$_2$ asym str.']):
        idx = np.argmin(np.abs(wavenumber - peak))
        ax1.annotate(f'{label}\n{peak} cm$^{{-1}}$',
                     xy=(peak, ir_spectrum[idx]),
                     xytext=(peak, ir_spectrum[idx] + 0.15),
                     ha='center', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='darkblue', lw=1))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.4)
    
    # Raman spectrum
    ax2 = axes[1]
    ax2.plot(wavenumber, raman_spectrum, 'r-', linewidth=1.5)
    ax2.fill_between(wavenumber, raman_spectrum, alpha=0.3, color='red')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Polyethylene (PE) - Raman Spectrum', fontsize=14, fontweight='bold')
    
    # Annotate Raman peaks
    for peak, label in zip(pe_raman_peaks, ['C-C str.', 'CH$_2$ twist', 'CH$_2$ scissor', 'CH$_2$ sym', 'CH$_2$ asym']):
        idx = np.argmin(np.abs(wavenumber - peak))
        ax2.annotate(f'{label}\n{peak} cm$^{{-1}}$',
                     xy=(peak, raman_spectrum[idx]),
                     xytext=(peak, raman_spectrum[idx] + 0.15),
                     ha='center', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=1))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.4)
    
    # Invert x-axis (spectroscopy convention)
    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlim(3500, 400)
    
    plt.tight_layout()
    plt.savefig('ir_raman_polymer.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print complementary information table
    print("\nPolyethylene (PE) Vibrational Analysis Summary:")
    print("=" * 80)
    print(f"{'Mode':<20} {'Wavenumber':>12} {'IR Intensity':>15} {'Raman Intensity':>15}")
    print("-" * 80)
    modes = [
        ('CH2 rocking', 720, 'Medium', 'Weak'),
        ('C-C stretch', 1130, 'Weak', 'Strong'),
        ('CH2 twist', 1295, 'Very weak', 'Medium'),
        ('CH2 scissor', 1440, 'Medium', 'Medium'),
        ('CH2 sym stretch', 2850, 'Strong', 'Very Strong'),
        ('CH2 asym stretch', 2920, 'Very Strong', 'Strong'),
    ]
    for mode, wn, ir_int, raman_int in modes:
        print(f"{mode:<20} {wn:>8} cm^-1 {ir_int:>15} {raman_int:>15}")
    print("=" * 80)
    

## 4.4 Surface-Enhanced Raman Spectroscopy (SERS)

### 4.4.1 Principles of SERS

Surface-Enhanced Raman Spectroscopy (SERS) dramatically enhances Raman signals (by factors of $10^4$ to $10^{10}$) for molecules adsorbed on or near nanostructured metal surfaces, particularly silver (Ag) and gold (Au). This enhancement arises from two mechanisms:

#### SERS Enhancement Mechanisms

  1. **Electromagnetic Enhancement (EM)** : Localized surface plasmon resonance (LSPR) creates intense electromagnetic fields near the metal nanostructure surface. This is the dominant mechanism, providing enhancement factors of $10^4$ to $10^8$.
  2. **Chemical Enhancement (CE)** : Charge transfer between the adsorbed molecule and metal surface modifies the molecular polarizability. This provides additional enhancement of $10^1$ to $10^2$.

**SERS Enhancement Factor:**

\\[ \text{EF} = \frac{I_{\text{SERS}} / N_{\text{surf}}}{I_{\text{normal}} / N_{\text{vol}}} \\] 

where \\(I_{\text{SERS}}\\) and \\(I_{\text{normal}}\\) are the SERS and normal Raman intensities, \\(N_{\text{surf}}\\) is the number of molecules on the SERS substrate, and \\(N_{\text{vol}}\\) is the number of molecules in the normal Raman scattering volume.

**Electromagnetic Enhancement Approximation:**

\\[ \text{EF}_{\text{EM}} \approx \left|\frac{E_{\text{local}}}{E_0}\right|^4 \\] 

The fourth power dependence arises because both the incident and scattered fields are enhanced.

#### Code Example 4: SERS Enhancement Factor Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def calculate_field_enhancement(wavelength, particle_radius, distance):
        """
        Simplified calculation of electric field enhancement near a spherical
        metal nanoparticle using the quasi-static approximation.
    
        Parameters:
        -----------
        wavelength : float
            Incident light wavelength (nm)
        particle_radius : float
            Nanoparticle radius (nm)
        distance : float
            Distance from particle surface (nm)
    
        Returns:
        --------
        enhancement : float
            |E/E0|^2 field enhancement factor
        """
        # Simplified model parameters for silver
        # Resonance wavelength ~400 nm for small Ag nanoparticles
        omega_p = 9.0  # Plasma frequency (eV)
        gamma = 0.05   # Damping (eV)
        epsilon_m = 1.0  # Medium dielectric constant (air)
    
        # Convert wavelength to energy
        energy = 1240.0 / wavelength  # eV
    
        # Drude model for silver dielectric function (simplified)
        epsilon_r = 1 - omega_p**2 / (energy**2 + gamma**2)
        epsilon_i = omega_p**2 * gamma / (energy * (energy**2 + gamma**2))
        epsilon = complex(epsilon_r, epsilon_i)
    
        # Polarizability (quasi-static)
        alpha = particle_radius**3 * (epsilon - epsilon_m) / (epsilon + 2*epsilon_m)
    
        # Field enhancement at distance r from center
        r = particle_radius + distance
    
        # Near-field enhancement (radial component, simplified)
        enhancement = 1 + 2 * np.abs(alpha) / (r**3)
    
        return enhancement
    
    # Wavelength dependence of enhancement
    wavelengths = np.linspace(350, 700, 200)
    particle_radius = 30  # nm
    distance = 1  # nm from surface
    
    enhancements = [calculate_field_enhancement(wl, particle_radius, distance)
                    for wl in wavelengths]
    
    # Distance dependence
    distances = np.linspace(0.5, 20, 50)
    enhancement_vs_distance = [calculate_field_enhancement(450, particle_radius, d)
                               for d in distances]
    
    # SERS enhancement (|E/E0|^4)
    sers_enhancement = np.array(enhancement_vs_distance)**2  # |E/E0|^4
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Wavelength dependence
    ax1 = axes[0, 0]
    ax1.plot(wavelengths, enhancements, 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('|E/E$_0$|$^2$', fontsize=12)
    ax1.set_title('Field Enhancement vs. Wavelength\n(Ag nanoparticle, r=30 nm, d=1 nm)', fontsize=12)
    ax1.axvline(x=450, color='r', linestyle='--', alpha=0.5, label='Maximum enhancement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distance dependence
    ax2 = axes[0, 1]
    ax2.semilogy(distances, sers_enhancement, 'r-', linewidth=2)
    ax2.set_xlabel('Distance from surface (nm)', fontsize=12)
    ax2.set_ylabel('SERS Enhancement Factor', fontsize=12)
    ax2.set_title('SERS Enhancement vs. Distance\n(|E/E$_0$|$^4$ decay)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Normal vs SERS spectrum comparison
    ax3 = axes[1, 0]
    wavenumber = np.linspace(400, 2000, 500)
    
    # Normal Raman spectrum (weak)
    normal_peaks = [600, 1000, 1350, 1580]
    normal_spectrum = np.zeros_like(wavenumber)
    for peak in normal_peaks:
        normal_spectrum += 0.01 * np.exp(-((wavenumber - peak)**2) / (2 * 20**2))
    normal_spectrum += np.random.normal(0, 0.002, len(wavenumber))
    
    # SERS spectrum (enhanced by 10^6)
    sers_spectrum = normal_spectrum * 1e6 + np.random.normal(0, 1000, len(wavenumber))
    
    ax3.plot(wavenumber, normal_spectrum * 1e5, 'b-', linewidth=1.5, label='Normal Raman (x10$^5$)')
    ax3.plot(wavenumber, sers_spectrum / 1e4, 'r-', linewidth=1.5, label='SERS (x10$^{-4}$)')
    ax3.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax3.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax3.set_title('Normal Raman vs. SERS Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Hot spot illustration
    ax4 = axes[1, 1]
    ax4.set_aspect('equal')
    
    # Draw two nanoparticles with hot spot
    theta = np.linspace(0, 2*np.pi, 100)
    x1, y1 = 0, 0
    x2, y2 = 2.2, 0
    
    circle1_x = x1 + np.cos(theta)
    circle1_y = y1 + np.sin(theta)
    circle2_x = x2 + np.cos(theta)
    circle2_y = y2 + np.sin(theta)
    
    ax4.fill(circle1_x, circle1_y, color='gold', alpha=0.8)
    ax4.fill(circle2_x, circle2_y, color='gold', alpha=0.8)
    ax4.plot(circle1_x, circle1_y, 'k-', linewidth=2)
    ax4.plot(circle2_x, circle2_y, 'k-', linewidth=2)
    
    # Hot spot region
    hotspot = plt.Circle((1.1, 0), 0.3, color='red', alpha=0.5, label='Hot spot')
    ax4.add_patch(hotspot)
    
    # Field lines (simplified)
    for angle in np.linspace(-0.3, 0.3, 5):
        ax4.annotate('', xy=(0.9, angle), xytext=(0.3, angle),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        ax4.annotate('', xy=(1.9, angle), xytext=(1.3, angle),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    ax4.set_xlim(-2, 4.5)
    ax4.set_ylim(-2, 2)
    ax4.set_title('SERS Hot Spot Between Nanoparticles', fontsize=12)
    ax4.text(1.1, -1.5, 'Enhancement up to 10$^{10}$\nin hot spot region', ha='center', fontsize=10)
    ax4.text(-0.5, 0, 'Au/Ag\nNP', ha='center', va='center', fontsize=10)
    ax4.text(2.7, 0, 'Au/Ag\nNP', ha='center', va='center', fontsize=10)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('sers_enhancement.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSERS Key Parameters:")
    print("=" * 50)
    print(f"Maximum field enhancement (|E/E0|^2): {max(enhancements):.1f}")
    print(f"SERS enhancement at 1 nm: {sers_enhancement[0]:.2e}")
    print(f"SERS enhancement at 5 nm: {sers_enhancement[np.argmin(np.abs(np.array(distances)-5))]:.2e}")
    print(f"SERS enhancement at 10 nm: {sers_enhancement[np.argmin(np.abs(np.array(distances)-10))]:.2e}")
    

## 4.5 Applications in Carbon Materials

### 4.5.1 Raman Spectroscopy of Carbon Materials

Raman spectroscopy is particularly powerful for characterizing carbon materials due to the high polarizability of carbon-carbon bonds. The technique can distinguish between different carbon allotropes and assess structural quality, defects, and layer number.

#### Key Raman Bands in Carbon Materials

Band | Position (cm$^{-1}$) | Origin | Information  
---|---|---|---  
D band | ~1350 | Breathing mode of sp$^2$ rings (requires defect for activation) | Defect density, disorder  
G band | ~1580 | E$_{2g}$ phonon at Brillouin zone center | sp$^2$ hybridization  
2D (G') band | ~2700 | Second-order overtone of D band | Layer number, stacking  
D' band | ~1620 | Intravalley defect-activated mode | Edge defects  
RBM (CNT) | 100-400 | Radial breathing mode | CNT diameter  
  
### 4.5.2 Graphene Characterization

Raman spectroscopy is the primary technique for identifying the number of graphene layers and assessing quality. The 2D band shape and the I$_{2D}$/I$_G$ ratio are particularly diagnostic:

  * **Single-layer graphene** : Sharp, symmetric 2D band; I$_{2D}$/I$_G$ > 2
  * **Bilayer graphene** : Broader 2D band with 4-peak structure; I$_{2D}$/I$_G$ ~ 1
  * **Multilayer graphene** : Asymmetric 2D band; I$_{2D}$/I$_G$ < 1
  * **Graphite** : Asymmetric 2D band with shoulder; I$_{2D}$/I$_G$ < 0.5

#### Code Example 5: Graphene Layer Identification from Raman Spectra
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    # - scipy>=1.10.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def lorentzian(x, center, width, amplitude):
        """Generate Lorentzian peak."""
        return amplitude * (width**2) / ((x - center)**2 + width**2)
    
    def generate_graphene_spectrum(wavenumber, n_layers, defect_level=0.05):
        """
        Generate synthetic Raman spectrum for graphene with different layer numbers.
    
        Parameters:
        -----------
        wavenumber : array
            Wavenumber axis (cm^-1)
        n_layers : int
            Number of graphene layers (1, 2, 3, or 'bulk' for graphite)
        defect_level : float
            Relative D band intensity (0 = pristine, 1 = highly defective)
    
        Returns:
        --------
        spectrum : array
            Simulated Raman spectrum
        """
        spectrum = np.zeros_like(wavenumber, dtype=float)
    
        # G band (always present, ~1580 cm^-1)
        g_center = 1582
        g_width = 15
        g_amp = 1.0
        spectrum += lorentzian(wavenumber, g_center, g_width, g_amp)
    
        # D band (defect-activated, ~1350 cm^-1)
        d_center = 1350
        d_width = 30
        d_amp = defect_level * g_amp
        spectrum += lorentzian(wavenumber, d_center, d_width, d_amp)
    
        # 2D band - shape depends on layer number
        if n_layers == 1:
            # Single layer: single sharp Lorentzian
            spectrum += lorentzian(wavenumber, 2680, 25, 2.5 * g_amp)
        elif n_layers == 2:
            # Bilayer: 4-component structure (simplified as broader peak)
            spectrum += lorentzian(wavenumber, 2660, 20, 0.4 * g_amp)
            spectrum += lorentzian(wavenumber, 2690, 20, 0.6 * g_amp)
            spectrum += lorentzian(wavenumber, 2710, 20, 0.4 * g_amp)
            spectrum += lorentzian(wavenumber, 2730, 20, 0.2 * g_amp)
        elif n_layers == 3:
            # Trilayer: broader asymmetric
            spectrum += lorentzian(wavenumber, 2670, 25, 0.3 * g_amp)
            spectrum += lorentzian(wavenumber, 2700, 30, 0.5 * g_amp)
            spectrum += lorentzian(wavenumber, 2730, 25, 0.2 * g_amp)
        else:  # Graphite (bulk)
            # Asymmetric with shoulder
            spectrum += lorentzian(wavenumber, 2680, 35, 0.3 * g_amp)
            spectrum += lorentzian(wavenumber, 2720, 25, 0.2 * g_amp)
    
        # Add small noise
        spectrum += np.random.normal(0, 0.02, len(wavenumber))
    
        return spectrum
    
    # Generate wavenumber axis
    wavenumber = np.linspace(1200, 2900, 1000)
    
    # Generate spectra for different layer numbers
    layers = [1, 2, 3, 'bulk']
    spectra = {}
    for n in layers:
        spectra[n] = generate_graphene_spectrum(wavenumber, n if isinstance(n, int) else 10, defect_level=0.1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'1': '#e74c3c', '2': '#3498db', '3': '#2ecc71', 'bulk': '#9b59b6'}
    labels = {'1': 'Monolayer', '2': 'Bilayer', '3': 'Trilayer', 'bulk': 'Graphite'}
    
    for ax, (n, spectrum) in zip(axes.flat, spectra.items()):
        key = str(n)
        ax.plot(wavenumber, spectrum, color=colors[key], linewidth=1.5)
        ax.fill_between(wavenumber, spectrum, alpha=0.3, color=colors[key])
        ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=11)
        ax.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax.set_title(f'{labels[key]} Graphene', fontsize=13, fontweight='bold')
    
        # Annotate bands
        ax.annotate('D', xy=(1350, spectrum[np.argmin(np.abs(wavenumber - 1350))]),
                    xytext=(1350, 0.4), ha='center', fontsize=12, fontweight='bold')
        ax.annotate('G', xy=(1582, spectrum[np.argmin(np.abs(wavenumber - 1582))]),
                    xytext=(1582, 1.2), ha='center', fontsize=12, fontweight='bold')
        ax.annotate('2D', xy=(2700, spectrum[np.argmin(np.abs(wavenumber - 2700))]),
                    xytext=(2700, max(spectrum)*1.1), ha='center', fontsize=12, fontweight='bold')
    
        ax.set_xlim(1200, 2900)
        ax.set_ylim(0, max(spectrum) * 1.3)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('graphene_layer_raman.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate I_2D/I_G ratios
    print("\nGraphene Layer Identification Metrics:")
    print("=" * 60)
    print(f"{'Layer':<15} {'I_2D/I_G':>15} {'2D FWHM':>15} {'Characteristic':>15}")
    print("-" * 60)
    
    for n, spectrum in spectra.items():
        g_idx = np.argmin(np.abs(wavenumber - 1582))
        d2_region = (wavenumber > 2600) & (wavenumber < 2800)
        i_g = spectrum[g_idx]
        i_2d = np.max(spectrum[d2_region])
        ratio = i_2d / i_g
    
        fwhm = "~30" if n == 1 else "~50" if n == 2 else "~60" if n == 3 else "~70"
        char = "Symmetric" if n == 1 else "4-peak" if n == 2 else "Asymmetric"
    
        print(f"{labels[str(n)]:<15} {ratio:>15.2f} {fwhm + ' cm^-1':>15} {char:>15}")
    

### 4.5.3 Carbon Nanotube Analysis

Raman spectroscopy provides rich information about carbon nanotubes, including diameter, chirality, and electronic character (metallic vs. semiconducting). The radial breathing mode (RBM) fixes the **diameter** through $\omega_{\text{RBM}} = A/d + B$. The **electronic character is not a function of diameter** : it is set by the chirality indices $(n, m)$, with the tube metallic (or quasi-metallic) when $(n - m)$ is divisible by 3 and semiconducting otherwise. Metallic and semiconducting tubes therefore coexist at essentially every diameter, and they are told apart by the G-band lineshape (a broad, downshifted Breit-Wigner-Fano G$^-$ marks a metallic tube) or by the resonance condition read from a Kataura plot.

#### Code Example 6: CNT Diameter Determination from RBM
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rbm_to_diameter(omega_rbm, A=248, B=0):
        """
        Calculate CNT diameter from RBM frequency.
    
        Parameters:
        -----------
        omega_rbm : float or array
            RBM frequency in cm^-1
        A : float
            Constant (typically 234-248 for bundled/isolated CNTs)
        B : float
            Environmental correction term
    
        Returns:
        --------
        diameter : float or array
            CNT diameter in nm
    
        The relation is: omega_RBM = A/d + B
        """
        return A / (omega_rbm - B)
    
    def diameter_to_rbm(diameter, A=248, B=0):
        """Convert diameter to RBM frequency."""
        return A / diameter + B
    
    def generate_cnt_raman(wavenumber, diameters, excitation_wavelength=532):
        """
        Generate synthetic CNT Raman spectrum.
    
        Parameters:
        -----------
        wavenumber : array
            Wavenumber axis
        diameters : list
            List of CNT diameters in nm
        excitation_wavelength : float
            Laser wavelength in nm
    
        Returns:
        --------
        spectrum : array
            Simulated Raman spectrum
        """
        spectrum = np.zeros_like(wavenumber, dtype=float)
    
        # RBM peaks
        for d in diameters:
            rbm_freq = diameter_to_rbm(d)
            # Intensity depends on resonance (simplified)
            intensity = 0.5 + 0.5 * np.random.random()
            spectrum += lorentzian(wavenumber, rbm_freq, 5, intensity)
    
        # D band
        spectrum += lorentzian(wavenumber, 1340, 25, 0.3)
    
        # G band (G+ and G-)
        spectrum += lorentzian(wavenumber, 1570, 15, 0.8)  # G- (metallic) or G+ (semiconducting)
        spectrum += lorentzian(wavenumber, 1590, 10, 1.0)  # G+
    
        # 2D band
        spectrum += lorentzian(wavenumber, 2680, 35, 0.6)
    
        return spectrum
    
    # Example CNT sample with different diameters
    wavenumber_full = np.linspace(100, 3000, 2000)
    wavenumber_rbm = np.linspace(100, 400, 500)
    
    # Sample containing CNTs with diameters 0.8, 1.0, 1.2, 1.4 nm
    diameters = [0.8, 1.0, 1.2, 1.4]
    cnt_spectrum = generate_cnt_raman(wavenumber_full, diameters)
    rbm_spectrum = generate_cnt_raman(wavenumber_rbm, diameters)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full spectrum
    ax1 = axes[0, 0]
    ax1.plot(wavenumber_full, cnt_spectrum, 'b-', linewidth=1.5)
    ax1.fill_between(wavenumber_full, cnt_spectrum, alpha=0.3)
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('CNT Raman Spectrum (Full Range)', fontsize=13, fontweight='bold')
    ax1.annotate('RBM', xy=(250, 0.6), fontsize=11, fontweight='bold')
    ax1.annotate('D', xy=(1340, 0.4), fontsize=11, fontweight='bold')
    ax1.annotate('G', xy=(1580, 1.1), fontsize=11, fontweight='bold')
    ax1.annotate('2D', xy=(2680, 0.7), fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # RBM region
    ax2 = axes[0, 1]
    ax2.plot(wavenumber_rbm, rbm_spectrum, 'r-', linewidth=1.5)
    ax2.fill_between(wavenumber_rbm, rbm_spectrum, alpha=0.3, color='red')
    ax2.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('RBM Region (Diameter Information)', fontsize=13, fontweight='bold')
    
    # Annotate each RBM peak with corresponding diameter
    for d in diameters:
        rbm_freq = diameter_to_rbm(d)
        idx = np.argmin(np.abs(wavenumber_rbm - rbm_freq))
        ax2.annotate(f'd = {d:.1f} nm\n({rbm_freq:.0f} cm$^{{-1}}$)',
                     xy=(rbm_freq, rbm_spectrum[idx]),
                     xytext=(rbm_freq, rbm_spectrum[idx] + 0.2),
                     ha='center', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='darkred'))
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Diameter-RBM relationship
    ax3 = axes[1, 0]
    diameters_range = np.linspace(0.5, 3.0, 100)
    rbm_range = diameter_to_rbm(diameters_range)
    
    ax3.plot(diameters_range, rbm_range, 'b-', linewidth=2)
    ax3.scatter([0.8, 1.0, 1.2, 1.4], [diameter_to_rbm(d) for d in [0.8, 1.0, 1.2, 1.4]],
               s=100, c='red', zorder=5, label='Sample CNTs')
    ax3.set_xlabel('CNT Diameter (nm)', fontsize=12)
    ax3.set_ylabel('RBM Frequency (cm$^{-1}$)', fontsize=12)
    ax3.set_title('Diameter-RBM Relationship: $\\omega_{RBM} = 248/d$', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, 3.0)
    ax3.set_ylim(50, 500)
    
    # G band analysis (metallic vs semiconducting)
    ax4 = axes[1, 1]
    wavenumber_g = np.linspace(1500, 1650, 300)
    
    # Semiconducting CNT G band
    g_semi = lorentzian(wavenumber_g, 1590, 10, 1.0) + lorentzian(wavenumber_g, 1570, 8, 0.3)
    # Metallic CNT G band (broader G- due to Kohn anomaly)
    g_metal = lorentzian(wavenumber_g, 1590, 10, 1.0) + lorentzian(wavenumber_g, 1550, 30, 0.6)
    
    ax4.plot(wavenumber_g, g_semi, 'b-', linewidth=2, label='Semiconducting CNT')
    ax4.plot(wavenumber_g, g_metal, 'r-', linewidth=2, label='Metallic CNT')
    ax4.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax4.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax4.set_title('G Band: Metallic vs. Semiconducting CNT', fontsize=13, fontweight='bold')
    ax4.annotate('G$^+$', xy=(1590, 1.0), xytext=(1600, 1.1), fontsize=11)
    ax4.annotate('G$^-$ (narrow)', xy=(1570, 0.35), xytext=(1530, 0.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax4.annotate('G$^-$ (broad, BWF)', xy=(1550, 0.6), xytext=(1510, 0.8), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('cnt_raman_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print diameter analysis
    print("\nCNT Diameter Analysis from RBM:")
    print("=" * 50)
    print(f"{'RBM (cm^-1)':<15} {'Diameter (nm)':<15} {'Electronic type':<40}")
    print("-" * 50)
    # IMPORTANT: the RBM frequency gives the DIAMETER only. Whether a tube is
    # metallic or semiconducting is set by its CHIRALITY (n, m):
    #     metallic (or quasi-metallic) when (n - m) mod 3 == 0,
    #     semiconducting otherwise.
    # Roughly one third of all (n, m) tubes are metallic at any diameter, so a
    # diameter threshold cannot classify electronic character. The assignment
    # requires the (n, m) index - from a Kataura plot / resonance profile - or
    # the G-band lineshape (metallic tubes show a broad, downshifted BWF G-).
    def metallic_from_chirality(n, m):
        """True if the (n, m) tube is metallic: (n - m) mod 3 == 0."""
        return (n - m) % 3 == 0
    
    for d in diameters:
        rbm = diameter_to_rbm(d)
        note = "metallic or semiconducting - needs (n,m)"
        print(f"{rbm:<15.0f} {d:<15.1f} {note:<40}")
    
    # Example: tubes of nearly the same diameter, opposite electronic character
    for (n, m) in [(10, 10), (11, 9), (10, 9)]:
        d_nm = 0.142 * np.sqrt(3 * (n**2 + n*m + m**2)) / np.pi  # a_CC = 0.142 nm
        kind = "metallic" if metallic_from_chirality(n, m) else "semiconducting"
        print(f"({n},{m}): d = {d_nm:.2f} nm -> {kind}")
    

## 4.6 D/G Band Analysis and Peak Fitting

### 4.6.1 D/G Ratio as Defect Indicator

The intensity ratio I$_D$/I$_G$ is widely used to quantify the defect density in carbon materials. For graphene and graphite, the relationship between I$_D$/I$_G$ and defect density follows the Tuinstra-Koenig relation:

**Tuinstra-Koenig Relation:**

\\[ \frac{I_D}{I_G} = \frac{C(\lambda_L)}{L_a} \\] 

where \\(L_a\\) is the in-plane crystallite size and \\(C(\lambda_L)\\) is a calibration constant that depends on the laser wavelength; the classical value is \\(C(514.5\ \text{nm}) \approx 4.4\\) nm. Note the **inverse** dependence: in this regime the D band comes from the crystallite edges, so \\(I_D/I_G\\) *grows* as the crystallites get smaller.

**Modified Relation for Nano-crystalline / Amorphous Carbon (Ferrari-Robertson regime):**

\\[ \frac{I_D}{I_G} = C'(\lambda_L) \cdot L_a^2 \\] 

This relation applies only when \\(L_a\\) is small (< 2 nm). Here the rings themselves are being destroyed, so the D band tracks the number of surviving aromatic rings rather than the crystallite perimeter: \\(I_D/I_G\\) now *decreases* as \\(L_a\\) shrinks - the opposite of the Tuinstra-Koenig trend.

Because the two regimes have opposite slopes, \\(I_D/I_G\\) passes through a maximum near \\(L_a \approx 2\\) nm and a single measured ratio is ambiguous on its own: the G-band position and width (and the presence of a resolvable D' band) are needed to decide which branch of the amorphization trajectory the sample is on.

#### Code Example 7: Complete D/G Band Fitting and Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0
    # - matplotlib>=3.7.0
    # - scipy>=1.10.0
    # - lmfit>=1.2.0 (optional, for advanced fitting)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter
    
    def voigt(x, center, sigma, gamma, amplitude):
        """
        Voigt profile (convolution of Gaussian and Lorentzian).
        Approximation using the Faddeeva function approach.
        """
        from scipy.special import wofz
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    def multi_peak_model(x, *params):
        """
        Multi-peak model for D and G band fitting.
    
        Parameters layout: [amp_D, center_D, sigma_D, gamma_D,
                            amp_G, center_G, sigma_G, gamma_G,
                            amp_D', center_D', sigma_D', gamma_D',
                            baseline_slope, baseline_intercept]
        """
        n_peaks = (len(params) - 2) // 4
        result = np.zeros_like(x, dtype=float)
    
        for i in range(n_peaks):
            idx = i * 4
            amp, center, sigma, gamma = params[idx:idx+4]
            result += voigt(x, center, sigma, gamma, amp)
    
        # Linear baseline
        result += params[-2] * x + params[-1]
    
        return result
    
    def fit_dg_bands(wavenumber, intensity, n_peaks=3):
        """
        Fit D, G, and D' bands in carbon Raman spectrum.
    
        Parameters:
        -----------
        wavenumber : array
            Wavenumber data
        intensity : array
            Intensity data
        n_peaks : int
            Number of peaks to fit (2=D+G, 3=D+G+D')
    
        Returns:
        --------
        popt : array
            Optimized parameters
        fitted : array
            Fitted curve
        components : dict
            Individual peak components
        """
        # Initial guesses
        if n_peaks == 2:
            # D and G bands only
            p0 = [
                intensity.max() * 0.3, 1350, 30, 20,  # D band
                intensity.max(), 1580, 20, 10,         # G band
                0, 0                                    # baseline
            ]
            bounds_low = [0, 1300, 5, 1, 0, 1550, 5, 1, -np.inf, -np.inf]
            bounds_high = [np.inf, 1400, 100, 50, np.inf, 1610, 50, 30, np.inf, np.inf]
        else:
            # D, G, and D' bands
            p0 = [
                intensity.max() * 0.3, 1350, 30, 20,  # D band
                intensity.max(), 1580, 20, 10,         # G band
                intensity.max() * 0.1, 1620, 15, 10,   # D' band
                0, 0                                    # baseline
            ]
            bounds_low = [0, 1300, 5, 1, 0, 1550, 5, 1, 0, 1600, 5, 1, -np.inf, -np.inf]
            bounds_high = [np.inf, 1400, 100, 50, np.inf, 1610, 50, 30, np.inf, 1660, 50, 30, np.inf, np.inf]
    
        try:
            popt, pcov = curve_fit(multi_peak_model, wavenumber, intensity,
                                   p0=p0, bounds=(bounds_low, bounds_high), maxfev=10000)
        except RuntimeError:
            print("Warning: Fitting did not converge, using initial guess")
            popt = p0
    
        fitted = multi_peak_model(wavenumber, *popt)
    
        # Extract individual components
        components = {}
        peak_names = ['D', 'G', "D'"][:n_peaks]
        for i, name in enumerate(peak_names):
            idx = i * 4
            amp, center, sigma, gamma = popt[idx:idx+4]
            components[name] = {
                'curve': voigt(wavenumber, center, sigma, gamma, amp),
                'amplitude': amp,
                'center': center,
                'sigma': sigma,
                'gamma': gamma,
                'fwhm': 2 * (0.5346 * gamma + np.sqrt(0.2166 * gamma**2 + sigma**2 * 2.355**2 / 4))
            }
    
        # Baseline
        components['baseline'] = popt[-2] * wavenumber + popt[-1]
    
        return popt, fitted, components
    
    # Generate synthetic carbon spectrum with noise
    np.random.seed(42)
    wavenumber = np.linspace(1100, 1800, 500)
    
    # True parameters for a moderately defective graphene sample
    true_D = voigt(wavenumber, 1350, 35, 25, 0.4)
    true_G = voigt(wavenumber, 1582, 18, 12, 1.0)
    true_Dp = voigt(wavenumber, 1620, 12, 8, 0.15)
    baseline = 0.0001 * wavenumber + 0.05
    noise = np.random.normal(0, 0.03, len(wavenumber))
    
    synthetic_spectrum = true_D + true_G + true_Dp + baseline + noise
    synthetic_spectrum = np.maximum(synthetic_spectrum, 0)
    
    # Perform fitting
    popt, fitted, components = fit_dg_bands(wavenumber, synthetic_spectrum, n_peaks=3)
    
    # Calculate I_D/I_G ratio
    id_ig_ratio = components['D']['amplitude'] / components['G']['amplitude']
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fitting result
    ax1 = axes[0, 0]
    ax1.plot(wavenumber, synthetic_spectrum, 'ko', markersize=2, alpha=0.5, label='Data')
    ax1.plot(wavenumber, fitted, 'r-', linewidth=2, label='Total Fit')
    ax1.plot(wavenumber, components['D']['curve'] + components['baseline'], 'b--',
             linewidth=1.5, label=f"D band ({components['D']['center']:.0f} cm$^{{-1}}$)")
    ax1.plot(wavenumber, components['G']['curve'] + components['baseline'], 'g--',
             linewidth=1.5, label=f"G band ({components['G']['center']:.0f} cm$^{{-1}}$)")
    ax1.plot(wavenumber, components["D'"]['curve'] + components['baseline'], 'm--',
             linewidth=1.5, label=f"D' band ({components[\"D'\"]['center']:.0f} cm$^{{-1}}$)")
    ax1.plot(wavenumber, components['baseline'], 'k--', linewidth=1, alpha=0.5, label='Baseline')
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('D/G Band Fitting with Voigt Profiles', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Residual
    ax2 = axes[0, 1]
    residual = synthetic_spectrum - fitted
    ax2.plot(wavenumber, residual, 'b-', linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(wavenumber, residual, alpha=0.3)
    ax2.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title(f'Fitting Residual (RMSE = {np.sqrt(np.mean(residual**2)):.4f})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Deconvoluted peaks
    ax3 = axes[1, 0]
    ax3.fill_between(wavenumber, components['D']['curve'], alpha=0.5, color='blue', label='D band')
    ax3.fill_between(wavenumber, components['G']['curve'], alpha=0.5, color='green', label='G band')
    ax3.fill_between(wavenumber, components["D'"]['curve'], alpha=0.5, color='magenta', label="D' band")
    ax3.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax3.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax3.set_title('Deconvoluted Peak Components', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create table
    table_data = [
        ['Parameter', 'D Band', 'G Band', "D' Band"],
        ['Center (cm$^{-1}$)', f"{components['D']['center']:.1f}",
         f"{components['G']['center']:.1f}", f"{components[\"D'\"]['center']:.1f}"],
        ['Amplitude', f"{components['D']['amplitude']:.3f}",
         f"{components['G']['amplitude']:.3f}", f"{components[\"D'\"]['amplitude']:.3f}"],
        ['FWHM (cm$^{-1}$)', f"{components['D']['fwhm']:.1f}",
         f"{components['G']['fwhm']:.1f}", f"{components[\"D'\"]['fwhm']:.1f}"],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#f093fb')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title(f'\nFitting Results Summary\n$I_D/I_G$ = {id_ig_ratio:.3f}',
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dg_band_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("\nD/G Band Fitting Results:")
    print("=" * 60)
    for name in ['D', 'G', "D'"]:
        comp = components[name]
        print(f"\n{name} Band:")
        print(f"  Center: {comp['center']:.1f} cm^-1")
        print(f"  Amplitude: {comp['amplitude']:.4f}")
        print(f"  FWHM: {comp['fwhm']:.1f} cm^-1")
    
    print("\n" + "=" * 60)
    print(f"I_D/I_G ratio: {id_ig_ratio:.3f}")
    print(f"I_D'/I_G ratio: {components[\"D'\"]['amplitude']/components['G']['amplitude']:.3f}")
    
    # Estimate crystallite size using the Tuinstra-Koenig relation.
    # The classical constant is tabulated for the 514.5 nm Ar+ line,
    # C(514.5 nm) ~ 4.4 nm, and is commonly reused at 532 nm.
    # It applies to the PEAK-HEIGHT ratio used above.
    C_TK = 4.4  # nm (514.5 nm; used here as an approximation for 532 nm)
    La = C_TK / id_ig_ratio
    print(f"\nEstimated crystallite size (L_a): {La:.1f} nm")
    print("(Tuinstra-Koenig, peak-height ratio, C ~ 4.4 nm at 514.5/532 nm)")
    

### 4.6.2 Crystallinity Assessment in Silicon

The same peak-shape logic applies outside carbon materials. Crystalline silicon has a single sharp first-order optical phonon at 520 cm$^{-1}$ with a FWHM of only ~3-4 cm$^{-1}$, whereas amorphous silicon loses long-range order, the $q \approx 0$ selection rule relaxes, and the band collapses into a broad hump centred near 480 cm$^{-1}$ that is tens of wavenumbers wide. A mixed-phase film shows both features at once, so the degree of crystallinity can be estimated by decomposing the 400-560 cm$^{-1}$ envelope into a crystalline and an amorphous component:

$$X_c \approx \frac{I_c}{I_c + I_a}$$

where $I_c$ and $I_a$ are the integrated intensities of the 520 cm$^{-1}$ and 480 cm$^{-1}$ components. This expression is a simplification: quantitative work fits a third intermediate component near 500-510 cm$^{-1}$ for grain boundaries and applies a Raman cross-section ratio $y = \sigma_a/\sigma_c$ (typically 0.7-0.9, and itself grain-size and excitation-wavelength dependent), so $X_c$ from the formula above should be treated as a calibrated relative index rather than an absolute crystalline volume fraction. Laser-induced heating also downshifts and broadens the 520 cm$^{-1}$ line, so the excitation power must be kept low enough to leave the peak position stable.

### 4.6.3 Biomedical Applications

Because water is a weak Raman scatterer, Raman spectroscopy can probe biological tissue and live cells directly, without labels or staining. Proteins and lipids give characteristic bands — the phenylalanine ring-breathing mode at 1003 cm$^{-1}$, Amide III near 1250 cm$^{-1}$, the CH$_2$ bending mode near 1445 cm$^{-1}$, and Amide I near 1660 cm$^{-1}$ — and shifts in their relative intensities distinguish diseased from healthy tissue, which is the basis of Raman-assisted cancer tissue diagnosis. Confocal Raman imaging maps these bands at sub-micron resolution for label-free imaging of cells, and time-resolved measurements can follow drug-cell interactions as the drug's own Raman signature changes inside the cell. These applications carry the same practical advantages discussed above: non-destructive measurement, minimal sample preparation, and compatibility with glass and aqueous environments.

## Exercises

#### Exercise 1: Stokes/Anti-Stokes Temperature Measurement (Basic)

The anti-Stokes to Stokes intensity ratio for a vibrational mode at 1000 cm$^{-1}$ is measured to be 0.15. Calculate the sample temperature.

View Solution
    
    
    import numpy as np
    
    # Physical constants
    h = 6.62607015e-34  # J*s
    c = 2.99792458e10   # cm/s
    k_B = 1.380649e-23  # J/K
    
    # Given data
    omega = 1000  # cm^-1
    ratio = 0.15  # I_anti-Stokes / I_Stokes
    
    # From Boltzmann distribution:
    # ratio = exp(-h*c*omega / (k_B * T))
    # ln(ratio) = -h*c*omega / (k_B * T)
    # T = -h*c*omega / (k_B * ln(ratio))
    
    delta_E = h * c * omega  # Energy in Joules
    T = -delta_E / (k_B * np.log(ratio))
    
    print(f"Sample temperature: {T:.0f} K ({T-273.15:.0f} C)")
    # Answer: approximately 758 K (485 C)
    

#### Exercise 2: Identifying Unknown Carbon Material (Intermediate)

A carbon sample shows the following Raman features: D band at 1348 cm$^{-1}$ with I$_D$/I$_G$ = 0.85, G band at 1582 cm$^{-1}$, and 2D band at 2695 cm$^{-1}$ with I$_{2D}$/I$_G$ = 0.45. The 2D band FWHM is approximately 65 cm$^{-1}$. Identify the material and estimate its quality.

View Solution

**Analysis:**

  * The I$_{2D}$/I$_G$ ratio of 0.45 is less than 2, indicating this is NOT single-layer graphene.
  * The 2D band FWHM of ~65 cm$^{-1}$ is broader than single-layer (~30 cm$^{-1}$) or bilayer (~50 cm$^{-1}$).
  * The significant D band (I$_D$/I$_G$ = 0.85) indicates substantial defects or edges.

**Conclusion:** This is most likely multilayer graphene (3+ layers) or graphite with significant defects/edges. The high I$_D$/I$_G$ ratio suggests either:

  1. Nano-crystalline graphite with small domain size
  2. Reduced graphene oxide (rGO) with residual defects
  3. Graphene with high edge density (e.g., graphene nanoribbons)

Using the Tuinstra-Koenig relation with the classical constant C(514.5 nm) ~ 4.4 nm (conventionally reused at 532 nm, peak-height ratio):

La = 4.4 / 0.85 = 5.2 nm

This small crystallite size confirms nano-crystalline or heavily defected graphitic carbon.

#### Exercise 3: CNT Diameter Distribution (Intermediate)

Write a Python function that takes RBM peak positions as input and returns the diameter distribution of a CNT sample. Apply it to a sample with RBM peaks at 165, 195, 235, and 285 cm$^{-1}$.

View Solution
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def analyze_cnt_diameters(rbm_peaks, A=248, B=0):
        """
        Calculate CNT diameter distribution from RBM peaks.
    
        Parameters:
        -----------
        rbm_peaks : list
            RBM peak positions in cm^-1
        A, B : float
            Parameters for omega = A/d + B
    
        Returns:
        --------
        diameters : list
            Calculated diameters in nm
        """
        diameters = [A / (omega - B) for omega in rbm_peaks]
    
        # Print results
        print("CNT Diameter Analysis:")
        print("=" * 40)
        print(f"{'RBM (cm^-1)':<15} {'Diameter (nm)':<15}")
        print("-" * 40)
        for rbm, d in zip(rbm_peaks, diameters):
            print(f"{rbm:<15.0f} {d:<15.2f}")
    
        # Statistics
        print("-" * 40)
        print(f"Mean diameter: {np.mean(diameters):.2f} nm")
        print(f"Diameter range: {min(diameters):.2f} - {max(diameters):.2f} nm")
    
        return diameters
    
    # Apply to sample
    rbm_peaks = [165, 195, 235, 285]
    diameters = analyze_cnt_diameters(rbm_peaks)
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(diameters)), diameters, color='steelblue', edgecolor='black')
    plt.xticks(range(len(diameters)), [f'{rbm} cm$^{{-1}}$' for rbm in rbm_peaks])
    plt.xlabel('RBM Peak Position')
    plt.ylabel('CNT Diameter (nm)')
    plt.title('CNT Diameter Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Results:
    # 165 cm^-1 -> 1.50 nm
    # 195 cm^-1 -> 1.27 nm
    # 235 cm^-1 -> 1.06 nm
    # 285 cm^-1 -> 0.87 nm
    

#### Exercise 4: SERS Enhancement Calculation (Advanced)

A molecule shows a normal Raman signal of 100 counts per second at 1 M concentration. When the same molecule is adsorbed on a SERS substrate, it gives $10^8$ counts per second at $10^{-6}$ M concentration. Calculate the analytical enhancement factor (AEF) and estimate the true enhancement factor assuming only 1% of molecules are in hot spots.

View Solution
    
    
    import numpy as np
    
    # Given data
    I_normal = 100  # counts/s
    C_normal = 1    # M
    I_SERS = 1e8    # counts/s
    C_SERS = 1e-6   # M
    
    # Analytical Enhancement Factor (AEF)
    # AEF = (I_SERS / C_SERS) / (I_normal / C_normal)
    AEF = (I_SERS / C_SERS) / (I_normal / C_normal)
    print(f"Analytical Enhancement Factor (AEF): {AEF:.2e}")
    
    # True Enhancement Factor considering hot spot fraction
    hot_spot_fraction = 0.01  # 1% of molecules in hot spots
    
    # If only 1% contribute, the actual enhancement per molecule is higher
    true_EF = AEF / hot_spot_fraction
    print(f"True Enhancement Factor (1% in hot spots): {true_EF:.2e}")
    
    # The electromagnetic enhancement scales as |E/E0|^4
    # So the field enhancement is:
    field_enhancement = true_EF ** 0.25
    print(f"Estimated field enhancement |E/E0|: {field_enhancement:.0f}")
    
    # Results:
    # AEF = 10^12
    # True EF = 10^14 (if 1% in hot spots)
    # Field enhancement |E/E0| ~ 3162
    

#### Exercise 5: Complete Raman Spectrum Analysis (Advanced)

Write a Python program that performs complete analysis of a carbon Raman spectrum, including: (1) baseline correction, (2) D/G band fitting with Voigt profiles, (3) calculation of I$_D$/I$_G$ and crystallite size, and (4) generation of a publication-quality figure.

View Solution
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter
    from scipy.special import wofz
    
    class CarbonRamanAnalyzer:
        """Complete Raman spectrum analyzer for carbon materials."""
    
        def __init__(self, wavenumber, intensity, laser_wavelength=532):
            self.wavenumber = np.array(wavenumber)
            self.intensity = np.array(intensity)
            self.laser_wavelength = laser_wavelength
            self.results = {}
    
        def baseline_correct(self, degree=3, regions=None):
            """Polynomial baseline correction."""
            if regions is None:
                # Default: use ends of spectrum for baseline
                mask = (self.wavenumber < 1200) | (self.wavenumber > 1750)
            else:
                mask = np.zeros_like(self.wavenumber, dtype=bool)
                for low, high in regions:
                    mask |= (self.wavenumber > low) & (self.wavenumber < high)
    
            coeffs = np.polyfit(self.wavenumber[mask], self.intensity[mask], degree)
            self.baseline = np.polyval(coeffs, self.wavenumber)
            self.intensity_corrected = self.intensity - self.baseline
            return self.intensity_corrected
    
        @staticmethod
        def voigt(x, center, sigma, gamma, amplitude):
            """Voigt profile."""
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
        def fit_bands(self):
            """Fit D, G, and D' bands."""
            def model(x, a_d, c_d, s_d, g_d, a_g, c_g, s_g, g_g, a_dp, c_dp, s_dp, g_dp):
                return (self.voigt(x, c_d, s_d, g_d, a_d) +
                        self.voigt(x, c_g, s_g, g_g, a_g) +
                        self.voigt(x, c_dp, s_dp, g_dp, a_dp))
    
            p0 = [0.3, 1350, 30, 20, 1.0, 1580, 15, 10, 0.1, 1620, 10, 8]
            bounds_low = [0, 1300, 5, 1, 0, 1550, 5, 1, 0, 1600, 5, 1]
            bounds_high = [2, 1400, 80, 50, 2, 1610, 50, 30, 1, 1660, 40, 25]
    
            popt, pcov = curve_fit(model, self.wavenumber, self.intensity_corrected,
                                   p0=p0, bounds=(bounds_low, bounds_high), maxfev=10000)
    
            self.results = {
                'D': {'amp': popt[0], 'center': popt[1], 'sigma': popt[2], 'gamma': popt[3]},
                'G': {'amp': popt[4], 'center': popt[5], 'sigma': popt[6], 'gamma': popt[7]},
                "D'": {'amp': popt[8], 'center': popt[9], 'sigma': popt[10], 'gamma': popt[11]},
            }
    
            for band in ['D', 'G', "D'"]:
                r = self.results[band]
                r['curve'] = self.voigt(self.wavenumber, r['center'], r['sigma'], r['gamma'], r['amp'])
                r['fwhm'] = 2 * (0.5346*r['gamma'] + np.sqrt(0.2166*r['gamma']**2 + (2.355*r['sigma'])**2/4))
    
            self.fitted = model(self.wavenumber, *popt)
            return self.results
    
        def calculate_metrics(self):
            """Calculate I_D/I_G ratio and crystallite size."""
            self.id_ig = self.results['D']['amp'] / self.results['G']['amp']
    
            # Tuinstra-Koenig relation, PEAK-HEIGHT convention:
            #     L_a = C(lambda_L) / (I_D/I_G),   C(514.5 nm) ~ 4.4 nm
            # The same constant is conventionally reused at 532 nm.
            C_TK = 4.4  # nm
            self.La = C_TK / self.id_ig
    
            # NOTE: the Cancado (2006) general-wavelength calibration
            #     L_a = 2.4e-10 * lambda_L**4 / (A_D/A_G)   [lambda_L in nm]
            # is a DIFFERENT convention - it is defined for integrated peak
            # AREAS, not heights - and returns 2.4e-10 * 532**4 = 19.2 nm
            # divided by the area ratio at 532 nm. The two calibrations are
            # not interchangeable; pick one and apply it consistently. This
            # class reports the Tuinstra-Koenig value only, because the fit
            # above returns peak heights.
    
            return {'I_D/I_G': self.id_ig, 'L_a (nm)': self.La}
    
        def plot_results(self, save_path=None):
            """Generate publication-quality figure."""
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
            # Original and baseline
            ax1 = axes[0, 0]
            ax1.plot(self.wavenumber, self.intensity, 'b-', lw=1.5, label='Original')
            ax1.plot(self.wavenumber, self.baseline, 'r--', lw=1.5, label='Baseline')
            ax1.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax1.set_ylabel('Intensity (a.u.)')
            ax1.set_title('(a) Raw Spectrum with Baseline')
            ax1.legend()
            ax1.invert_xaxis()
    
            # Fitting
            ax2 = axes[0, 1]
            ax2.plot(self.wavenumber, self.intensity_corrected, 'ko', ms=2, alpha=0.5, label='Data')
            ax2.plot(self.wavenumber, self.fitted, 'r-', lw=2, label='Fit')
            for name, color in [('D', 'blue'), ('G', 'green'), ("D'", 'magenta')]:
                ax2.plot(self.wavenumber, self.results[name]['curve'], '--', color=color,
                         lw=1.5, label=f"{name} band")
            ax2.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax2.set_ylabel('Intensity (a.u.)')
            ax2.set_title('(b) Peak Fitting')
            ax2.legend(fontsize=9)
            ax2.invert_xaxis()
    
            # Residual
            ax3 = axes[1, 0]
            residual = self.intensity_corrected - self.fitted
            ax3.plot(self.wavenumber, residual, 'b-', lw=1)
            ax3.axhline(0, color='r', ls='--')
            ax3.fill_between(self.wavenumber, residual, alpha=0.3)
            ax3.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax3.set_ylabel('Residual')
            ax3.set_title(f'(c) Residual (RMSE = {np.sqrt(np.mean(residual**2)):.4f})')
            ax3.invert_xaxis()
    
            # Results table
            ax4 = axes[1, 1]
            ax4.axis('off')
    
            table_data = [['Band', 'Center', 'FWHM', 'Amplitude']]
            for name in ['D', 'G', "D'"]:
                r = self.results[name]
                table_data.append([name, f"{r['center']:.1f}", f"{r['fwhm']:.1f}", f"{r['amp']:.3f}"])
    
            table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                              colWidths=[0.2, 0.25, 0.25, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)
    
            for i in range(4):
                table[(0, i)].set_facecolor('#f093fb')
                table[(0, i)].set_text_props(color='white', fontweight='bold')
    
            ax4.set_title(f'(d) Results\n$I_D/I_G$ = {self.id_ig:.3f}, $L_a$ = {self.La:.1f} nm',
                          fontsize=12, fontweight='bold')
    
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    # Usage example
    np.random.seed(42)
    wavenumber = np.linspace(1100, 1800, 500)
    intensity = (0.4 * CarbonRamanAnalyzer.voigt(wavenumber, 1350, 35, 25, 1) +
                 1.0 * CarbonRamanAnalyzer.voigt(wavenumber, 1582, 18, 12, 1) +
                 0.15 * CarbonRamanAnalyzer.voigt(wavenumber, 1620, 12, 8, 1) +
                 0.0001 * wavenumber + 0.1 +
                 np.random.normal(0, 0.03, len(wavenumber)))
    
    analyzer = CarbonRamanAnalyzer(wavenumber, intensity, laser_wavelength=532)
    analyzer.baseline_correct()
    analyzer.fit_bands()
    metrics = analyzer.calculate_metrics()
    analyzer.plot_results('carbon_raman_analysis.png')
    
    print("\nAnalysis Complete:")
    print(f"I_D/I_G = {metrics['I_D/I_G']:.3f}")
    print(f"Crystallite size L_a = {metrics['L_a (nm)']:.1f} nm")
    

## Chapter Summary

#### Key Takeaways

  * **Raman scattering** is an inelastic light scattering process where photons exchange energy with molecular vibrations. Stokes scattering (energy loss) is more intense than anti-Stokes scattering (energy gain) at room temperature.
  * **Selection rules** : A vibration is Raman-active if it causes a change in molecular polarizability. This is complementary to IR spectroscopy, which requires a change in dipole moment.
  * **IR-Raman complementarity** : Symmetric vibrations are typically Raman-active; asymmetric vibrations are typically IR-active. For centrosymmetric molecules, mutual exclusion applies.
  * **SERS** provides dramatic enhancement ($10^4$-$10^{10}$) through electromagnetic and chemical mechanisms, enabling single-molecule detection.
  * **Carbon materials** are excellent candidates for Raman analysis. The D/G ratio indicates defect density, the 2D band reveals layer number in graphene, and RBM frequency gives CNT diameter.
  * **Practical analysis** requires proper baseline correction, peak fitting (Voigt profiles recommended), and understanding of the physical meaning of spectral features.

## References

  1. Ferraro, J. R., Nakamoto, K., and Brown, C. W. (2003). _Introductory Raman Spectroscopy_ (2nd ed.). Academic Press. pp. 1-94. - Comprehensive introduction to Raman theory and instrumentation.
  2. Ferrari, A. C. and Basko, D. M. (2013). Raman spectroscopy as a versatile tool for studying the properties of graphene. _Nature Nanotechnology_ , 8(4), 235-246. DOI: 10.1038/nnano.2013.46 - Essential reference for graphene Raman analysis.
  3. Dresselhaus, M. S., Jorio, A., Hofmann, M., Dresselhaus, G., and Saito, R. (2010). Perspectives on carbon nanotubes and graphene Raman spectroscopy. _Nano Letters_ , 10(3), 751-758. DOI: 10.1021/nl904286r - CNT Raman fundamentals.
  4. Le Ru, E. C. and Etchegoin, P. G. (2009). _Principles of Surface-Enhanced Raman Spectroscopy_. Elsevier. pp. 185-264. - Definitive guide to SERS theory and applications.
  5. Tuinstra, F. and Koenig, J. L. (1970). Raman spectrum of graphite. _The Journal of Chemical Physics_ , 53(3), 1126-1130. DOI: 10.1063/1.1674108 - Original paper on D/G ratio and crystallite size.
  6. Cancado, L. G., et al. (2006). General equation for the determination of the crystallite size La of nanographite by Raman spectroscopy. _Applied Physics Letters_ , 88(16), 163106. DOI: 10.1063/1.2196057 - Refined Tuinstra-Koenig relation.
  7. Jorio, A., Saito, R., Dresselhaus, G., and Dresselhaus, M. S. (2011). _Raman Spectroscopy in Graphene Related Systems_. Wiley-VCH. - Comprehensive treatment of graphene and CNT Raman.
  8. Long, D. A. (2002). _The Raman Effect: A Unified Treatment of the Theory of Raman Scattering by Molecules_. John Wiley and Sons. pp. 85-152. - Rigorous quantum mechanical treatment of Raman scattering.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice.
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, or libraries.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any damages arising from the use of this content.
  * The content may be changed, updated, or discontinued without notice.
  * License: Creative Commons BY 4.0
