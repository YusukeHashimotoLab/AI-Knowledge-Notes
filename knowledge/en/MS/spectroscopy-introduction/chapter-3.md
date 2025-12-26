---
title: "Chapter 3: Infrared Spectroscopy"
chapter_title: "Chapter 3: Infrared Spectroscopy"
version: 1.0
---

[AI Terakoya Top](<../../index.html>)>[Materials Science](<../index.html>)>[Spectroscopy Introduction](<index.html>)>Chapter 3

EN | [JP](<../../../jp/MS/spectroscopy-introduction/chapter-3.html>) | Last sync: 2025-12-26

# Chapter 3: Infrared Spectroscopy

**What you will learn in this chapter:** Infrared (IR) spectroscopy is a powerful analytical technique that probes molecular vibrations to identify functional groups and characterize materials. In this chapter, you will learn the fundamental principles of molecular vibrations including stretching and bending modes, understand IR selection rules based on dipole moment changes, explore Fourier Transform Infrared (FTIR) instrumentation and interferometry, master functional group identification using characteristic absorption frequencies, and discover ATR-FTIR for surface analysis. Practical Python code examples will guide you through IR spectral analysis, peak identification, and baseline correction.

### Learning Objectives

  * Understand molecular vibrations and distinguish between stretching and bending modes
  * Apply IR selection rules based on dipole moment change requirements
  * Explain the principles of FTIR spectroscopy and Michelson interferometry
  * Identify functional groups from characteristic IR absorption frequencies
  * Understand ATR-FTIR principles and applications for surface analysis
  * Perform IR spectral analysis using Python including baseline correction and peak identification

## 3.1 Fundamentals of Molecular Vibrations

### 3.1.1 The Harmonic Oscillator Model

Molecular vibrations can be understood through the quantum mechanical harmonic oscillator model. Consider a diatomic molecule as two masses connected by a spring. The vibrational frequency depends on the force constant (bond strength) and the reduced mass of the atoms.

**Classical vibrational frequency:**

$$ \nu = \frac{1}{2\pi}\sqrt{\frac{k}{\mu}} $$ 

where $k$ is the force constant (N/m), and $\mu$ is the reduced mass:

$$ \mu = \frac{m_1 m_2}{m_1 + m_2} $$ 

**Quantum mechanical energy levels:**

$$ E_v = \left(v + \frac{1}{2}\right)h\nu, \quad v = 0, 1, 2, \ldots $$ 

The vibrational wavenumber in cm$^{-1}$ is:

$$ \tilde{\nu} = \frac{1}{2\pi c}\sqrt{\frac{k}{\mu}} $$ 
    
    
    # Code Example 1: Calculating Vibrational Frequencies for Diatomic Molecules
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    c = 2.998e10  # Speed of light in cm/s
    h = 6.626e-34  # Planck's constant in J*s
    amu_to_kg = 1.66054e-27  # Atomic mass unit to kg
    
    def calculate_vibrational_frequency(k, m1, m2):
        """
        Calculate vibrational frequency for a diatomic molecule.
    
        Parameters:
        -----------
        k : float
            Force constant in N/m
        m1, m2 : float
            Atomic masses in amu
    
        Returns:
        --------
        nu_hz : float
            Frequency in Hz
        nu_wavenumber : float
            Wavenumber in cm^-1
        """
        # Convert masses to kg
        m1_kg = m1 * amu_to_kg
        m2_kg = m2 * amu_to_kg
    
        # Calculate reduced mass
        mu = (m1_kg * m2_kg) / (m1_kg + m2_kg)
    
        # Calculate frequency
        nu_hz = (1 / (2 * np.pi)) * np.sqrt(k / mu)
        nu_wavenumber = nu_hz / c
    
        return nu_hz, nu_wavenumber
    
    # Example calculations for common diatomic molecules
    molecules = {
        'H-Cl': {'k': 480, 'm1': 1.008, 'm2': 35.45},
        'H-Br': {'k': 410, 'm1': 1.008, 'm2': 79.90},
        'H-I': {'k': 320, 'm1': 1.008, 'm2': 126.9},
        'C-O': {'k': 1860, 'm1': 12.01, 'm2': 16.00},
        'N-N': {'k': 2260, 'm1': 14.01, 'm2': 14.01},
    }
    
    print("Vibrational Frequencies of Diatomic Molecules")
    print("=" * 60)
    print(f"{'Molecule':<10} {'k (N/m)':<12} {'Frequency (Hz)':<18} {'Wavenumber (cm-1)'}")
    print("-" * 60)
    
    for mol, params in molecules.items():
        nu_hz, nu_cm = calculate_vibrational_frequency(
            params['k'], params['m1'], params['m2']
        )
        print(f"{mol:<10} {params['k']:<12.0f} {nu_hz:<18.3e} {nu_cm:.0f}")
    
    # Visualize the effect of reduced mass on vibrational frequency
    k_fixed = 500  # Fixed force constant
    masses = np.linspace(1, 50, 100)  # Mass of second atom (first is H = 1)
    
    wavenumbers = []
    for m2 in masses:
        _, nu_cm = calculate_vibrational_frequency(k_fixed, 1.008, m2)
        wavenumbers.append(nu_cm)
    
    plt.figure(figsize=(10, 6))
    plt.plot(masses, wavenumbers, 'b-', linewidth=2)
    plt.xlabel('Mass of Second Atom (amu)', fontsize=12)
    plt.ylabel('Vibrational Wavenumber (cm$^{-1}$)', fontsize=12)
    plt.title('Effect of Reduced Mass on Vibrational Frequency\n(H-X molecules, k = 500 N/m)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=3000, color='r', linestyle='--', alpha=0.5, label='Typical C-H stretch region')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vibrational_frequency_mass.png', dpi=150, bbox_inches='tight')
    plt.show()
    

### 3.1.2 Types of Molecular Vibrations

For a molecule with $N$ atoms, the total number of vibrational modes is $3N - 6$ for nonlinear molecules and $3N - 5$ for linear molecules. These vibrations are classified into two main categories:

#### Stretching Vibrations

  * **Symmetric stretching ($\nu_s$):** Both bonds stretch and compress in phase
  * **Asymmetric stretching ($\nu_{as}$):** One bond stretches while the other compresses

#### Bending Vibrations (Deformations)

  * **Scissoring ($\delta$):** In-plane bending where bond angles change symmetrically
  * **Rocking ($\rho$):** In-plane bending where atoms move in the same direction
  * **Wagging ($\omega$):** Out-of-plane bending where atoms move together perpendicular to the plane
  * **Twisting ($\tau$):** Out-of-plane bending where atoms move in opposite directions

    
    
    ```mermaid
    flowchart TB
        A[Molecular Vibrations3N-6 or 3N-5 modes] --> B[Stretching]
        A --> C[Bending]
    
        B --> D[SymmetricBoth bonds in phase]
        B --> E[AsymmetricOut of phase]
    
        C --> F[In-plane]
        C --> G[Out-of-plane]
    
        F --> H[Scissoring]
        F --> I[Rocking]
    
        G --> J[Wagging]
        G --> K[Twisting]
    
        style A fill:#f093fb,color:#fff
        style B fill:#f5576c,color:#fff
        style C fill:#f5576c,color:#fff
    ```
    
    
    # Code Example 2: Visualizing Vibrational Modes of a Water Molecule
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.animation import FuncAnimation
    
    def draw_water_molecule(ax, positions, title, arrows=None):
        """
        Draw a water molecule with optional displacement arrows.
    
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to draw on
        positions : list of tuples
            [(O_x, O_y), (H1_x, H1_y), (H2_x, H2_y)]
        title : str
            Title for the subplot
        arrows : list of tuples or None
            Displacement vectors for each atom
        """
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
        O_pos, H1_pos, H2_pos = positions
    
        # Draw bonds
        ax.plot([O_pos[0], H1_pos[0]], [O_pos[1], H1_pos[1]], 'k-', linewidth=3)
        ax.plot([O_pos[0], H2_pos[0]], [O_pos[1], H2_pos[1]], 'k-', linewidth=3)
    
        # Draw atoms
        ax.add_patch(Circle(O_pos, 0.25, color='red', zorder=5))
        ax.add_patch(Circle(H1_pos, 0.15, color='white', ec='blue', linewidth=2, zorder=5))
        ax.add_patch(Circle(H2_pos, 0.15, color='white', ec='blue', linewidth=2, zorder=5))
    
        # Add labels
        ax.text(O_pos[0], O_pos[1], 'O', ha='center', va='center', fontsize=10, color='white', zorder=6)
        ax.text(H1_pos[0], H1_pos[1], 'H', ha='center', va='center', fontsize=8, color='blue', zorder=6)
        ax.text(H2_pos[0], H2_pos[1], 'H', ha='center', va='center', fontsize=8, color='blue', zorder=6)
    
        # Draw displacement arrows
        if arrows:
            for pos, arrow in zip(positions, arrows):
                if np.linalg.norm(arrow) > 0.01:
                    ax.annotate('', xy=(pos[0] + arrow[0], pos[1] + arrow[1]),
                               xytext=pos,
                               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Create figure for vibrational modes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Equilibrium positions (water molecule geometry)
    bond_length = 0.96  # O-H bond length (scaled for visualization)
    angle = 104.5 * np.pi / 180  # H-O-H angle
    
    O_eq = (0, 0)
    H1_eq = (-bond_length * np.sin(angle/2), -bond_length * np.cos(angle/2))
    H2_eq = (bond_length * np.sin(angle/2), -bond_length * np.cos(angle/2))
    
    # Mode 1: Symmetric stretch (3657 cm-1)
    # Both H atoms move away from O simultaneously
    sym_arrows = [(0, 0.15), (-0.3, -0.25), (0.3, -0.25)]
    draw_water_molecule(axes[0], [O_eq, H1_eq, H2_eq],
                       'Symmetric Stretch\n$\\nu_1$ = 3657 cm$^{-1}$', sym_arrows)
    
    # Mode 2: Bending/Scissoring (1595 cm-1)
    # H atoms move toward each other, O moves up
    bend_arrows = [(0, 0.2), (0.25, 0), (-0.25, 0)]
    draw_water_molecule(axes[1], [O_eq, H1_eq, H2_eq],
                       'Bending (Scissoring)\n$\\nu_2$ = 1595 cm$^{-1}$', bend_arrows)
    
    # Mode 3: Asymmetric stretch (3756 cm-1)
    # One H moves away while other moves toward O
    asym_arrows = [(0.1, 0), (-0.3, -0.2), (0.15, 0.1)]
    draw_water_molecule(axes[2], [O_eq, H1_eq, H2_eq],
                       'Asymmetric Stretch\n$\\nu_3$ = 3756 cm$^{-1}$', asym_arrows)
    
    plt.tight_layout()
    plt.savefig('water_vibrational_modes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nWater Molecule Vibrational Modes Summary:")
    print("=" * 50)
    print("Atoms (N) = 3, Nonlinear molecule")
    print(f"Number of vibrational modes = 3N - 6 = {3*3 - 6} modes")
    print("\nMode    Type              Wavenumber (cm-1)")
    print("-" * 50)
    print("v1      Symmetric stretch     3657")
    print("v2      Bending (scissor)     1595")
    print("v3      Asymmetric stretch    3756")
    

## 3.2 IR Selection Rules: Dipole Moment Change

### 3.2.1 The Selection Rule for IR Absorption

Not all molecular vibrations are IR active. For a vibration to absorb infrared radiation, it must cause a change in the molecular dipole moment during the vibration. This is the fundamental IR selection rule.

**IR Selection Rule:**

$$ \left(\frac{\partial \mu}{\partial Q}\right)_{Q=0} \neq 0 $$ 

where $\mu$ is the dipole moment and $Q$ is the normal coordinate of the vibration.

A vibration is **IR active** if the dipole moment changes during the vibration.

A vibration is **IR inactive** if the dipole moment remains constant (e.g., symmetric stretches in homonuclear diatomic molecules).

#### Examples of IR Active and Inactive Vibrations

Molecule | Vibration | Dipole Change | IR Activity  
---|---|---|---  
H$_2$, N$_2$, O$_2$ | Symmetric stretch | No change | Inactive  
HCl, CO | Stretch | Changes | Active  
CO$_2$ | Symmetric stretch | No change | Inactive  
CO$_2$ | Asymmetric stretch | Changes | Active  
CO$_2$ | Bending | Changes | Active  
H$_2$O | All modes | Changes | Active  
      
    
    # Code Example 3: Demonstrating IR Selection Rules with CO2
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_dipole_moment(positions, charges):
        """
        Calculate dipole moment vector for a set of point charges.
    
        Parameters:
        -----------
        positions : array
            Atomic positions [[x1,y1], [x2,y2], ...]
        charges : array
            Partial charges for each atom
    
        Returns:
        --------
        dipole : array
            Dipole moment vector [mu_x, mu_y]
        """
        positions = np.array(positions)
        charges = np.array(charges)
        dipole = np.sum(positions.T * charges, axis=1)
        return dipole
    
    def simulate_co2_vibrations():
        """
        Simulate CO2 vibrational modes and dipole moment changes.
        """
        # CO2 equilibrium geometry (linear)
        # C at origin, O atoms at +/- 1.16 Angstrom
        bond_length = 1.16
    
        # Partial charges (simplified model)
        q_C = +0.4  # Carbon
        q_O = -0.2  # Oxygen (each)
    
        # Time points for one vibrational period
        t = np.linspace(0, 2*np.pi, 100)
        amplitude = 0.1  # Vibrational amplitude
    
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
        modes = [
            ('Symmetric Stretch', 'sym'),
            ('Asymmetric Stretch', 'asym'),
            ('Bending', 'bend')
        ]
    
        for i, (mode_name, mode_type) in enumerate(modes):
            dipoles_x = []
            dipoles_y = []
    
            for phase in t:
                if mode_type == 'sym':
                    # Symmetric stretch: both O atoms move symmetrically
                    O1_x = -bond_length - amplitude * np.sin(phase)
                    O2_x = bond_length + amplitude * np.sin(phase)
                    C_x = 0
                    O1_y = O2_y = C_y = 0
                elif mode_type == 'asym':
                    # Asymmetric stretch: O atoms move in opposite directions
                    O1_x = -bond_length - amplitude * np.sin(phase)
                    O2_x = bond_length - amplitude * np.sin(phase)
                    C_x = amplitude * 0.5 * np.sin(phase)
                    O1_y = O2_y = C_y = 0
                else:  # bend
                    # Bending: O atoms move perpendicular to bond axis
                    O1_x = -bond_length
                    O2_x = bond_length
                    C_x = 0
                    O1_y = -amplitude * np.sin(phase)
                    O2_y = -amplitude * np.sin(phase)
                    C_y = amplitude * 0.5 * np.sin(phase)
    
                positions = [[O1_x, O1_y], [C_x, C_y], [O2_x, O2_y]]
                charges = [q_O, q_C, q_O]
    
                dipole = calculate_dipole_moment(positions, charges)
                dipoles_x.append(dipole[0])
                dipoles_y.append(dipole[1])
    
            # Plot molecular motion (snapshot)
            ax1 = axes[i, 0]
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-0.5, 0.5)
            ax1.set_aspect('equal')
    
            # Draw equilibrium position
            ax1.plot([-bond_length, 0], [0, 0], 'k-', linewidth=3, alpha=0.3)
            ax1.plot([0, bond_length], [0, 0], 'k-', linewidth=3, alpha=0.3)
            ax1.scatter([-bond_length, bond_length], [0, 0], s=200, c='red', alpha=0.3, zorder=5)
            ax1.scatter([0], [0], s=150, c='gray', alpha=0.3, zorder=5)
    
            # Draw displaced position at t=pi/2
            if mode_type == 'sym':
                O1_disp = (-bond_length - amplitude, 0)
                O2_disp = (bond_length + amplitude, 0)
                C_disp = (0, 0)
            elif mode_type == 'asym':
                O1_disp = (-bond_length - amplitude, 0)
                O2_disp = (bond_length - amplitude, 0)
                C_disp = (amplitude * 0.5, 0)
            else:
                O1_disp = (-bond_length, -amplitude)
                O2_disp = (bond_length, -amplitude)
                C_disp = (0, amplitude * 0.5)
    
            ax1.plot([O1_disp[0], C_disp[0]], [O1_disp[1], C_disp[1]], 'b-', linewidth=3)
            ax1.plot([C_disp[0], O2_disp[0]], [C_disp[1], O2_disp[1]], 'b-', linewidth=3)
            ax1.scatter([O1_disp[0], O2_disp[0]], [O1_disp[1], O2_disp[1]], s=200, c='red', zorder=5)
            ax1.scatter([C_disp[0]], [C_disp[1]], s=150, c='gray', zorder=5)
    
            ax1.set_title(f'{mode_name}', fontsize=12, fontweight='bold')
            ax1.text(-bond_length, 0.3, 'O', ha='center', fontsize=10)
            ax1.text(0, 0.3, 'C', ha='center', fontsize=10)
            ax1.text(bond_length, 0.3, 'O', ha='center', fontsize=10)
            ax1.set_xlabel('x position')
            ax1.axis('off')
    
            # Plot dipole moment change
            ax2 = axes[i, 1]
            ax2.plot(t, dipoles_x, 'b-', linewidth=2, label='$\mu_x$')
            ax2.plot(t, dipoles_y, 'r-', linewidth=2, label='$\mu_y$')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Vibrational Phase', fontsize=10)
            ax2.set_ylabel('Dipole Moment (a.u.)', fontsize=10)
    
            # Determine IR activity
            max_change = max(np.ptp(dipoles_x), np.ptp(dipoles_y))
            ir_active = "IR ACTIVE" if max_change > 0.01 else "IR INACTIVE"
            color = 'green' if max_change > 0.01 else 'red'
    
            ax2.set_title(f'Dipole Moment Change - {ir_active}', fontsize=12,
                         fontweight='bold', color=color)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('co2_ir_selection_rules.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    simulate_co2_vibrations()
    
    print("\nCO2 Vibrational Modes and IR Activity:")
    print("=" * 60)
    print("Mode                  Wavenumber    Dipole Change    IR Active")
    print("-" * 60)
    print("Symmetric stretch     1388 cm-1     No change        NO")
    print("Asymmetric stretch    2349 cm-1     Changes (x)      YES")
    print("Bending (x2)          667 cm-1      Changes (y)      YES")
    

## 3.3 FTIR Principles and Instrumentation

### 3.3.1 The Michelson Interferometer

Fourier Transform Infrared (FTIR) spectroscopy uses a Michelson interferometer to measure all wavelengths simultaneously, providing significant advantages over dispersive instruments in terms of speed (Fellgett advantage) and sensitivity (Jacquinot advantage).
    
    
    ```mermaid
    flowchart LR
        A[IR Source] --> B[Beamsplitter]
        B --> C[Fixed Mirror]
        B --> D[Moving Mirror]
        C --> B
        D --> B
        B --> E[Sample]
        E --> F[Detector]
        F --> G[Computer FFT]
        G --> H[IR Spectrum]
    
        style A fill:#f093fb,color:#fff
        style B fill:#4ecdc4,color:#fff
        style G fill:#f5576c,color:#fff
        style H fill:#a8e6cf,color:#000
    ```

**Interferogram:** The detector records intensity as a function of mirror displacement $\delta$:

$$ I(\delta) = \int_0^{\infty} B(\tilde{\nu})[1 + \cos(2\pi\tilde{\nu}\delta)]d\tilde{\nu} $$ 

**Fourier Transform:** The spectrum is obtained by Fourier transformation:

$$ B(\tilde{\nu}) = \int_{-\infty}^{\infty} [I(\delta) - I(0)/2]\cos(2\pi\tilde{\nu}\delta)d\delta $$ 

where $B(\tilde{\nu})$ is the spectral intensity and $\tilde{\nu}$ is the wavenumber.
    
    
    # Code Example 4: Simulating FTIR Interferometry and Fourier Transform
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    def simulate_ftir_measurement():
        """
        Simulate an FTIR measurement demonstrating interferometry principles.
        """
        # Simulation parameters
        n_points = 4096
        max_displacement = 0.5  # cm (mirror travel)
    
        # Mirror displacement (optical path difference)
        delta = np.linspace(-max_displacement, max_displacement, n_points)
    
        # Simulate a sample with multiple absorption peaks
        # Peak positions in cm^-1 and their intensities
        peaks = [
            (1000, 0.3),   # C-O stretch
            (1650, 0.5),   # C=O stretch
            (2900, 0.8),   # C-H stretch
            (3300, 0.4),   # O-H stretch
        ]
    
        # Generate source spectrum (blackbody-like)
        wavenumber_range = np.linspace(400, 4000, 2000)
        source_spectrum = np.exp(-(wavenumber_range - 2000)**2 / (2 * 1500**2))
    
        # Apply absorption peaks (simulate transmission)
        transmission = np.ones_like(wavenumber_range)
        for peak_pos, intensity in peaks:
            peak_width = 50  # cm^-1
            absorption = intensity * np.exp(-(wavenumber_range - peak_pos)**2 / (2 * peak_width**2))
            transmission *= (1 - absorption)
    
        detected_spectrum = source_spectrum * transmission
    
        # Calculate interferogram from spectrum
        interferogram = np.zeros_like(delta)
        for i, d in enumerate(delta):
            # Sum contributions from all wavenumbers
            for j, wn in enumerate(wavenumber_range):
                interferogram[i] += detected_spectrum[j] * np.cos(2 * np.pi * wn * d)
    
        # Normalize interferogram
        interferogram = interferogram / interferogram.max()
    
        # Apply apodization (triangular window)
        apodization = 1 - np.abs(delta) / max_displacement
        interferogram_apodized = interferogram * apodization
    
        # Perform FFT to recover spectrum
        n_fft = len(delta)
        fft_result = np.abs(fft(interferogram_apodized))[:n_fft//2]
    
        # Calculate wavenumber axis from FFT
        sample_spacing = delta[1] - delta[0]
        wavenumbers_fft = fftfreq(n_fft, sample_spacing)[:n_fft//2]
    
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # Plot 1: Interferogram
        ax1 = axes[0, 0]
        ax1.plot(delta * 10, interferogram, 'b-', linewidth=0.5)
        ax1.set_xlabel('Mirror Displacement (mm)', fontsize=11)
        ax1.set_ylabel('Detector Signal (a.u.)', fontsize=11)
        ax1.set_title('Raw Interferogram', fontsize=12, fontweight='bold')
        ax1.set_xlim(-2, 2)
        ax1.grid(True, alpha=0.3)
    
        # Plot 2: Apodized interferogram
        ax2 = axes[0, 1]
        ax2.plot(delta * 10, interferogram_apodized, 'g-', linewidth=0.5)
        ax2.set_xlabel('Mirror Displacement (mm)', fontsize=11)
        ax2.set_ylabel('Detector Signal (a.u.)', fontsize=11)
        ax2.set_title('Apodized Interferogram (Triangular Window)', fontsize=12, fontweight='bold')
        ax2.set_xlim(-5, 5)
        ax2.grid(True, alpha=0.3)
    
        # Plot 3: Original spectrum
        ax3 = axes[1, 0]
        ax3.plot(wavenumber_range, detected_spectrum, 'b-', linewidth=1.5)
        for peak_pos, intensity in peaks:
            ax3.axvline(x=peak_pos, color='r', linestyle='--', alpha=0.5)
            ax3.text(peak_pos, 0.9, f'{peak_pos}', rotation=90, va='top', ha='right', fontsize=9)
        ax3.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
        ax3.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax3.set_title('Input Spectrum (with absorption peaks)', fontsize=12, fontweight='bold')
        ax3.set_xlim(4000, 400)  # Conventional IR axis direction
        ax3.grid(True, alpha=0.3)
    
        # Plot 4: FFT recovered spectrum
        ax4 = axes[1, 1]
        valid_range = (wavenumbers_fft > 400) & (wavenumbers_fft < 4000)
        ax4.plot(wavenumbers_fft[valid_range], fft_result[valid_range] / fft_result[valid_range].max(),
                 'b-', linewidth=1.5)
        ax4.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
        ax4.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax4.set_title('FFT Recovered Spectrum', fontsize=12, fontweight='bold')
        ax4.set_xlim(4000, 400)
        ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('ftir_interferometry.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        return delta, interferogram, wavenumber_range, detected_spectrum
    
    # Run simulation
    delta, interferogram, wavenumbers, spectrum = simulate_ftir_measurement()
    
    print("\nFTIR Advantages:")
    print("=" * 50)
    print("1. Fellgett (Multiplex) Advantage:")
    print("   - All wavelengths measured simultaneously")
    print("   - SNR improvement by factor of sqrt(N)")
    print("\n2. Jacquinot (Throughput) Advantage:")
    print("   - No slits needed, higher light throughput")
    print("   - Typical improvement: 10-200x")
    print("\n3. Connes Advantage:")
    print("   - Internal laser reference for precise wavenumber calibration")
    print("   - Wavenumber accuracy: +/- 0.01 cm-1")
    

### 3.3.2 FTIR Resolution and Spectral Quality

The spectral resolution of an FTIR spectrometer is determined by the maximum optical path difference (OPD) achieved by the moving mirror:

**Spectral Resolution:**

$$ \Delta\tilde{\nu} = \frac{1}{2 \times \text{OPD}_{\text{max}}} $$ 

For example, with OPD$_{\text{max}}$ = 0.5 cm, the resolution is 1 cm$^{-1}$.

## 3.4 Functional Group Identification

### 3.4.1 Characteristic Group Frequencies

IR spectroscopy is particularly valuable for identifying functional groups because certain bonds absorb at characteristic frequencies regardless of the rest of the molecule. The IR spectrum is typically divided into two regions:

#### Functional Group Region (4000-1500 cm-1)

Contains characteristic stretching vibrations of functional groups. Peaks in this region are relatively easy to assign.

#### Fingerprint Region (1500-400 cm-1)

Contains complex patterns of bending vibrations and C-C stretches. This region is unique for each compound and is used for identification by comparison with reference spectra.

Functional Group | Vibration Type | Wavenumber (cm-1) | Intensity  
---|---|---|---  
O-H (alcohol, free) | Stretch | 3650-3600 | Strong, sharp  
O-H (alcohol, H-bonded) | Stretch | 3500-3200 | Strong, broad  
N-H (amine) | Stretch | 3500-3300 | Medium  
C-H (alkane) | Stretch | 3000-2850 | Strong  
C-H (alkene) | Stretch | 3100-3000 | Medium  
C-H (aromatic) | Stretch | 3150-3000 | Medium  
C-H (aldehyde) | Stretch | 2850-2700 | Medium (two peaks)  
C=O (carbonyl) | Stretch | 1750-1650 | Strong  
C=C (alkene) | Stretch | 1680-1620 | Variable  
C=C (aromatic) | Stretch | 1600, 1500 | Medium  
C-O (ether, alcohol) | Stretch | 1300-1000 | Strong  
C-N (amine) | Stretch | 1350-1000 | Medium  
      
    
    # Code Example 5: Functional Group Identification Tool
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Database of characteristic IR absorptions
    IR_DATABASE = {
        'O-H stretch (free)': {'range': (3600, 3650), 'intensity': 'strong', 'shape': 'sharp'},
        'O-H stretch (H-bonded)': {'range': (3200, 3500), 'intensity': 'strong', 'shape': 'broad'},
        'N-H stretch (primary amine)': {'range': (3300, 3500), 'intensity': 'medium', 'shape': 'two peaks'},
        'N-H stretch (secondary amine)': {'range': (3300, 3400), 'intensity': 'medium', 'shape': 'one peak'},
        'C-H stretch (sp3)': {'range': (2850, 3000), 'intensity': 'strong', 'shape': 'multiple'},
        'C-H stretch (sp2)': {'range': (3000, 3100), 'intensity': 'medium', 'shape': 'sharp'},
        'C-H stretch (sp, alkyne)': {'range': (3300, 3320), 'intensity': 'strong', 'shape': 'sharp'},
        'C-H stretch (aldehyde)': {'range': (2700, 2850), 'intensity': 'medium', 'shape': 'two peaks'},
        'C=O stretch (aldehyde)': {'range': (1720, 1740), 'intensity': 'strong', 'shape': 'sharp'},
        'C=O stretch (ketone)': {'range': (1705, 1725), 'intensity': 'strong', 'shape': 'sharp'},
        'C=O stretch (carboxylic acid)': {'range': (1700, 1725), 'intensity': 'strong', 'shape': 'sharp'},
        'C=O stretch (ester)': {'range': (1735, 1750), 'intensity': 'strong', 'shape': 'sharp'},
        'C=O stretch (amide)': {'range': (1640, 1690), 'intensity': 'strong', 'shape': 'sharp'},
        'C=C stretch (alkene)': {'range': (1620, 1680), 'intensity': 'variable', 'shape': 'sharp'},
        'C=C stretch (aromatic)': {'range': (1450, 1600), 'intensity': 'medium', 'shape': 'multiple'},
        'C-C stretch (alkyne)': {'range': (2100, 2260), 'intensity': 'variable', 'shape': 'sharp'},
        'C-N stretch (amine)': {'range': (1000, 1350), 'intensity': 'medium', 'shape': 'broad'},
        'C-O stretch (alcohol)': {'range': (1000, 1260), 'intensity': 'strong', 'shape': 'broad'},
        'C-O stretch (ether)': {'range': (1000, 1150), 'intensity': 'strong', 'shape': 'broad'},
        'N-H bend (amine)': {'range': (1550, 1650), 'intensity': 'medium', 'shape': 'sharp'},
        'C-H bend (methyl)': {'range': (1375, 1380), 'intensity': 'medium', 'shape': 'sharp'},
        'C-H bend (methylene)': {'range': (1465, 1470), 'intensity': 'medium', 'shape': 'sharp'},
    }
    
    def identify_functional_groups(peak_positions, tolerance=50):
        """
        Identify possible functional groups from peak positions.
    
        Parameters:
        -----------
        peak_positions : list
            List of peak wavenumbers in cm^-1
        tolerance : float
            Tolerance for peak matching in cm^-1
    
        Returns:
        --------
        assignments : dict
            Dictionary mapping peaks to possible functional groups
        """
        assignments = {}
    
        for peak in peak_positions:
            assignments[peak] = []
            for group_name, group_info in IR_DATABASE.items():
                low, high = group_info['range']
                if low - tolerance <= peak <= high + tolerance:
                    assignments[peak].append({
                        'group': group_name,
                        'expected_range': group_info['range'],
                        'intensity': group_info['intensity'],
                        'shape': group_info['shape']
                    })
    
        return assignments
    
    def plot_ir_spectrum_with_assignments(wavenumbers, absorbance, peak_positions):
        """
        Plot IR spectrum with functional group assignments.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
    
        # Plot spectrum
        ax.plot(wavenumbers, absorbance, 'b-', linewidth=1.5)
        ax.set_xlim(4000, 400)
        ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        ax.set_title('IR Spectrum with Functional Group Assignments', fontsize=14, fontweight='bold')
    
        # Add region labels
        ax.axvspan(4000, 1500, alpha=0.1, color='blue', label='Functional Group Region')
        ax.axvspan(1500, 400, alpha=0.1, color='green', label='Fingerprint Region')
    
        # Get assignments and annotate peaks
        assignments = identify_functional_groups(peak_positions)
    
        colors = plt.cm.Set1(np.linspace(0, 1, len(peak_positions)))
    
        for i, (peak, groups) in enumerate(assignments.items()):
            # Find peak height for annotation
            idx = np.argmin(np.abs(wavenumbers - peak))
            height = absorbance[idx]
    
            # Mark peak
            ax.axvline(x=peak, color=colors[i], linestyle='--', alpha=0.5)
            ax.plot(peak, height, 'o', color=colors[i], markersize=8)
    
            # Add annotation
            if groups:
                label = groups[0]['group'].split('(')[0].strip()
                ax.annotate(f'{peak}\n{label}',
                           xy=(peak, height),
                           xytext=(peak, height + 0.1),
                           fontsize=8,
                           ha='center',
                           rotation=90,
                           va='bottom')
    
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ir_functional_groups.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        return assignments
    
    # Generate synthetic IR spectrum (simulating ethanol)
    wavenumbers = np.linspace(4000, 400, 2000)
    
    # Ethanol peaks
    peaks_data = [
        (3350, 0.6, 200),   # O-H stretch (broad)
        (2970, 0.5, 30),    # C-H stretch (asymmetric)
        (2920, 0.4, 30),    # C-H stretch
        (2880, 0.3, 30),    # C-H stretch (symmetric)
        (1450, 0.25, 20),   # C-H bend
        (1380, 0.2, 15),    # C-H bend (methyl)
        (1050, 0.5, 40),    # C-O stretch
        (880, 0.15, 25),    # C-C stretch
    ]
    
    # Generate spectrum
    absorbance = np.zeros_like(wavenumbers)
    for center, amplitude, width in peaks_data:
        absorbance += amplitude * np.exp(-(wavenumbers - center)**2 / (2 * width**2))
    
    # Add some noise
    absorbance += np.random.normal(0, 0.01, len(wavenumbers))
    
    # List of peak positions for analysis
    peak_positions = [p[0] for p in peaks_data]
    
    # Plot and analyze
    print("Functional Group Analysis for Synthetic Ethanol Spectrum")
    print("=" * 60)
    
    assignments = plot_ir_spectrum_with_assignments(wavenumbers, absorbance, peak_positions)
    
    print("\nDetailed Peak Assignments:")
    print("-" * 60)
    for peak, groups in assignments.items():
        print(f"\nPeak at {peak} cm-1:")
        if groups:
            for g in groups:
                print(f"  - {g['group']}")
                print(f"    Expected range: {g['expected_range'][0]}-{g['expected_range'][1]} cm-1")
                print(f"    Intensity: {g['intensity']}, Shape: {g['shape']}")
        else:
            print("  No standard assignment found")
    

## 3.5 ATR-FTIR for Surface Analysis

### 3.5.1 Principles of Attenuated Total Reflectance

Attenuated Total Reflectance (ATR) is a sampling technique that enables direct measurement of solid and liquid samples without extensive preparation. It relies on total internal reflection of the IR beam at the interface between a high-refractive-index crystal and the sample.

**Evanescent Wave Penetration Depth:**

$$ d_p = \frac{\lambda}{2\pi n_1 \sqrt{\sin^2\theta - (n_2/n_1)^2}} $$ 

where:

  * $d_p$ = penetration depth
  * $\lambda$ = wavelength of IR radiation
  * $n_1$ = refractive index of ATR crystal
  * $n_2$ = refractive index of sample
  * $\theta$ = angle of incidence

Typical penetration depths range from 0.5 to 5 micrometers, making ATR ideal for surface analysis.

#### Common ATR Crystal Materials

Crystal | Refractive Index | Spectral Range (cm-1) | Applications  
---|---|---|---  
Diamond | 2.4 | 4000-400 | Universal, hard samples  
Germanium (Ge) | 4.0 | 5500-600 | Dark/absorbing samples  
Zinc Selenide (ZnSe) | 2.4 | 20000-500 | General purpose  
Silicon | 3.4 | 8300-1500 | Semiconductor samples  
      
    
    # Code Example 6: ATR Penetration Depth Calculator
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_penetration_depth(wavelength_um, n1, n2, theta_deg):
        """
        Calculate evanescent wave penetration depth for ATR.
    
        Parameters:
        -----------
        wavelength_um : float or array
            Wavelength in micrometers
        n1 : float
            Refractive index of ATR crystal
        n2 : float
            Refractive index of sample
        theta_deg : float
            Angle of incidence in degrees
    
        Returns:
        --------
        dp : float or array
            Penetration depth in micrometers
        """
        theta = np.radians(theta_deg)
    
        # Check for total internal reflection condition
        critical_angle = np.arcsin(n2 / n1)
        if theta < critical_angle:
            raise ValueError(f"Angle must be greater than critical angle: {np.degrees(critical_angle):.1f} degrees")
    
        # Calculate penetration depth
        sin_term = np.sin(theta)**2 - (n2/n1)**2
        dp = wavelength_um / (2 * np.pi * n1 * np.sqrt(sin_term))
    
        return dp
    
    def plot_atr_penetration():
        """
        Visualize ATR penetration depth as a function of wavelength.
        """
        # ATR crystal properties
        crystals = {
            'Diamond': {'n': 2.4, 'color': 'blue'},
            'Germanium': {'n': 4.0, 'color': 'red'},
            'ZnSe': {'n': 2.4, 'color': 'green'},
            'Silicon': {'n': 3.4, 'color': 'purple'},
        }
    
        # Typical sample refractive index (organic material)
        n_sample = 1.5
    
        # Typical angle of incidence
        theta = 45  # degrees
    
        # Wavelength range (mid-IR: 2.5-25 um = 4000-400 cm-1)
        wavenumbers = np.linspace(4000, 400, 500)
        wavelengths_um = 10000 / wavenumbers  # Convert cm-1 to um
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Plot 1: Penetration depth vs wavenumber for different crystals
        for crystal_name, props in crystals.items():
            try:
                dp = calculate_penetration_depth(wavelengths_um, props['n'], n_sample, theta)
                ax1.plot(wavenumbers, dp, label=f"{crystal_name} (n={props['n']})",
                        color=props['color'], linewidth=2)
            except ValueError:
                continue
    
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
        ax1.set_ylabel('Penetration Depth ($\mu$m)', fontsize=12)
        ax1.set_title(f'ATR Penetration Depth vs Wavenumber\n(Sample n={n_sample}, $\\theta$={theta})',
                      fontsize=12, fontweight='bold')
        ax1.set_xlim(4000, 400)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Add reference regions
        ax1.axhspan(0, 1, alpha=0.1, color='yellow', label='Surface sensitive')
        ax1.axhspan(1, 3, alpha=0.1, color='orange')
    
        # Plot 2: Effect of angle of incidence (Diamond crystal)
        angles = [40, 45, 50, 55, 60]
        n_diamond = 2.4
    
        for angle in angles:
            try:
                dp = calculate_penetration_depth(wavelengths_um, n_diamond, n_sample, angle)
                ax2.plot(wavenumbers, dp, label=f'$\\theta$ = {angle}', linewidth=2)
            except ValueError:
                continue
    
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
        ax2.set_ylabel('Penetration Depth ($\mu$m)', fontsize=12)
        ax2.set_title(f'Effect of Incidence Angle on Penetration Depth\n(Diamond ATR, Sample n={n_sample})',
                      fontsize=12, fontweight='bold')
        ax2.set_xlim(4000, 400)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('atr_penetration_depth.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        # Print summary table
        print("\nATR Penetration Depth Summary (at 1000 cm-1)")
        print("=" * 60)
        print(f"{'Crystal':<15} {'n_crystal':<12} {'Depth (um)':<15}")
        print("-" * 60)
    
        wl_1000 = 10  # wavelength at 1000 cm-1 in um
        for crystal_name, props in crystals.items():
            try:
                dp = calculate_penetration_depth(wl_1000, props['n'], n_sample, theta)
                print(f"{crystal_name:<15} {props['n']:<12.1f} {dp:<15.2f}")
            except ValueError:
                print(f"{crystal_name:<15} {props['n']:<12.1f} {'N/A (below critical angle)'}")
    
    plot_atr_penetration()
    

## 3.6 IR Spectral Data Analysis with Python

### 3.6.1 Baseline Correction

Raw IR spectra often contain baseline distortions due to scattering, instrumental drift, or sample preparation artifacts. Proper baseline correction is essential for accurate quantitative analysis.
    
    
    # Code Example 7: Comprehensive IR Spectral Analysis Toolkit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline
    
    class IRSpectrumAnalyzer:
        """
        A comprehensive toolkit for IR spectral analysis.
        """
    
        def __init__(self, wavenumbers, absorbance):
            """
            Initialize with spectral data.
    
            Parameters:
            -----------
            wavenumbers : array
                Wavenumber values in cm^-1
            absorbance : array
                Absorbance values
            """
            self.wavenumbers = np.array(wavenumbers)
            self.absorbance = np.array(absorbance)
            self.baseline = None
            self.corrected = None
            self.peaks = None
    
        def smooth_spectrum(self, window_length=11, polyorder=3):
            """
            Apply Savitzky-Golay smoothing filter.
            """
            self.absorbance = savgol_filter(self.absorbance, window_length, polyorder)
            return self
    
        def rubber_band_baseline(self, n_points=50):
            """
            Perform rubber band baseline correction.
    
            This method finds the convex hull of the spectrum and uses it
            as the baseline.
            """
            from scipy.spatial import ConvexHull
    
            # Create points for convex hull
            points = np.column_stack([self.wavenumbers, self.absorbance])
    
            # Find convex hull
            hull = ConvexHull(points)
    
            # Extract lower boundary of hull
            hull_points = points[hull.vertices]
            hull_points = hull_points[hull_points[:, 0].argsort()]
    
            # Interpolate to get baseline
            baseline_interp = np.interp(self.wavenumbers,
                                         hull_points[:, 0],
                                         hull_points[:, 1])
    
            # Take minimum envelope
            self.baseline = np.minimum(baseline_interp, self.absorbance)
            self.corrected = self.absorbance - self.baseline
    
            return self
    
        def polynomial_baseline(self, degree=3, regions=None):
            """
            Perform polynomial baseline correction.
    
            Parameters:
            -----------
            degree : int
                Degree of polynomial
            regions : list of tuples
                Wavenumber regions to use for fitting [(low1, high1), (low2, high2), ...]
                If None, uses automatic region selection
            """
            if regions is None:
                # Automatically select baseline regions (avoid peaks)
                peaks, _ = find_peaks(self.absorbance, prominence=0.05 * np.ptp(self.absorbance))
                mask = np.ones(len(self.wavenumbers), dtype=bool)
                for peak in peaks:
                    width = 20  # Exclude 20 points around each peak
                    mask[max(0, peak-width):min(len(mask), peak+width)] = False
            else:
                mask = np.zeros(len(self.wavenumbers), dtype=bool)
                for low, high in regions:
                    mask |= (self.wavenumbers >= low) & (self.wavenumbers <= high)
    
            # Fit polynomial to baseline regions
            coeffs = np.polyfit(self.wavenumbers[mask], self.absorbance[mask], degree)
            self.baseline = np.polyval(coeffs, self.wavenumbers)
            self.corrected = self.absorbance - self.baseline
    
            return self
    
        def als_baseline(self, lam=1e6, p=0.01, n_iter=10):
            """
            Asymmetric Least Squares baseline correction.
    
            Parameters:
            -----------
            lam : float
                Smoothness parameter (larger = smoother)
            p : float
                Asymmetry parameter (0 < p < 1, smaller = baseline below signal)
            n_iter : int
                Number of iterations
            """
            from scipy.sparse import diags, csc_matrix
            from scipy.sparse.linalg import spsolve
    
            L = len(self.absorbance)
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = csc_matrix(D)
            w = np.ones(L)
    
            for _ in range(n_iter):
                W = diags(w, 0, shape=(L, L))
                Z = W + lam * D.dot(D.T)
                self.baseline = spsolve(Z, w * self.absorbance)
                w = p * (self.absorbance > self.baseline) + (1-p) * (self.absorbance <= self.baseline)
    
            self.corrected = self.absorbance - self.baseline
            return self
    
        def find_peaks(self, prominence=0.02, width=5):
            """
            Find peaks in the spectrum.
    
            Parameters:
            -----------
            prominence : float
                Minimum prominence of peaks (relative to max)
            width : int
                Minimum width of peaks in data points
    
            Returns:
            --------
            peaks_info : list of dict
                List of dictionaries containing peak information
            """
            spectrum = self.corrected if self.corrected is not None else self.absorbance
    
            # Find peaks
            peaks, properties = find_peaks(
                spectrum,
                prominence=prominence * np.ptp(spectrum),
                width=width
            )
    
            self.peaks = []
            for i, peak_idx in enumerate(peaks):
                peak_info = {
                    'index': peak_idx,
                    'wavenumber': self.wavenumbers[peak_idx],
                    'absorbance': spectrum[peak_idx],
                    'prominence': properties['prominences'][i],
                    'width': properties['widths'][i]
                }
                self.peaks.append(peak_info)
    
            return self.peaks
    
        def plot_analysis(self, title="IR Spectrum Analysis"):
            """
            Create comprehensive analysis plot.
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # Plot 1: Original spectrum
            ax1 = axes[0, 0]
            ax1.plot(self.wavenumbers, self.absorbance, 'b-', linewidth=1.5)
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
            ax1.set_ylabel('Absorbance', fontsize=11)
            ax1.set_title('Original Spectrum', fontsize=12, fontweight='bold')
            ax1.set_xlim(max(self.wavenumbers), min(self.wavenumbers))
            ax1.grid(True, alpha=0.3)
    
            # Plot 2: Baseline correction
            ax2 = axes[0, 1]
            ax2.plot(self.wavenumbers, self.absorbance, 'b-', linewidth=1.5, label='Original', alpha=0.7)
            if self.baseline is not None:
                ax2.plot(self.wavenumbers, self.baseline, 'r--', linewidth=2, label='Baseline')
            ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
            ax2.set_ylabel('Absorbance', fontsize=11)
            ax2.set_title('Baseline Detection', fontsize=12, fontweight='bold')
            ax2.set_xlim(max(self.wavenumbers), min(self.wavenumbers))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
            # Plot 3: Corrected spectrum with peaks
            ax3 = axes[1, 0]
            spectrum = self.corrected if self.corrected is not None else self.absorbance
            ax3.plot(self.wavenumbers, spectrum, 'b-', linewidth=1.5)
    
            if self.peaks:
                peak_wn = [p['wavenumber'] for p in self.peaks]
                peak_abs = [p['absorbance'] for p in self.peaks]
                ax3.scatter(peak_wn, peak_abs, c='red', s=50, zorder=5, label='Detected Peaks')
    
                for peak in self.peaks:
                    ax3.annotate(f"{peak['wavenumber']:.0f}",
                               xy=(peak['wavenumber'], peak['absorbance']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', fontsize=8, rotation=90)
    
            ax3.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
            ax3.set_ylabel('Absorbance', fontsize=11)
            ax3.set_title('Baseline-Corrected Spectrum with Peaks', fontsize=12, fontweight='bold')
            ax3.set_xlim(max(self.wavenumbers), min(self.wavenumbers))
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
            # Plot 4: Peak table
            ax4 = axes[1, 1]
            ax4.axis('off')
    
            if self.peaks:
                table_data = [['Peak #', 'Wavenumber\n(cm$^{-1}$)', 'Absorbance', 'Width']]
                for i, peak in enumerate(self.peaks[:10]):  # Limit to top 10 peaks
                    table_data.append([
                        str(i+1),
                        f"{peak['wavenumber']:.1f}",
                        f"{peak['absorbance']:.4f}",
                        f"{peak['width']:.1f}"
                    ])
    
                table = ax4.table(cellText=table_data[1:],
                                 colLabels=table_data[0],
                                 cellLoc='center',
                                 loc='center',
                                 colWidths=[0.15, 0.25, 0.25, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
    
                ax4.set_title('Peak Summary', fontsize=12, fontweight='bold', y=0.95)
    
            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig('ir_analysis_complete.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Demonstration with synthetic polymer spectrum
    np.random.seed(42)
    
    # Generate synthetic PMMA-like IR spectrum
    wavenumbers = np.linspace(4000, 400, 2000)
    
    # PMMA characteristic peaks
    pmma_peaks = [
        (2995, 0.35, 25),   # C-H stretch (CH3)
        (2950, 0.40, 25),   # C-H stretch
        (1730, 0.90, 35),   # C=O stretch (ester)
        (1485, 0.25, 20),   # C-H bend
        (1450, 0.30, 20),   # C-H bend
        (1385, 0.20, 15),   # C-H bend (CH3)
        (1240, 0.55, 40),   # C-O stretch
        (1190, 0.50, 35),   # C-O stretch
        (1145, 0.45, 30),   # C-O stretch
        (985, 0.20, 25),    # C-C stretch
        (840, 0.15, 20),    # CH2 rock
        (750, 0.18, 25),    # Skeletal
    ]
    
    # Generate spectrum with peaks
    absorbance = np.zeros_like(wavenumbers)
    for center, amplitude, width in pmma_peaks:
        absorbance += amplitude * np.exp(-(wavenumbers - center)**2 / (2 * width**2))
    
    # Add baseline (polynomial drift + scattering)
    baseline_true = 0.1 + 0.00002 * (wavenumbers - 2000)**2 - 0.0001 * (wavenumbers - 2000)
    absorbance += baseline_true
    
    # Add noise
    absorbance += np.random.normal(0, 0.01, len(wavenumbers))
    
    # Create analyzer and process
    analyzer = IRSpectrumAnalyzer(wavenumbers, absorbance)
    
    # Apply processing steps
    analyzer.smooth_spectrum(window_length=11, polyorder=3)
    analyzer.polynomial_baseline(degree=2)
    peaks = analyzer.find_peaks(prominence=0.03, width=3)
    
    # Plot results
    analyzer.plot_analysis("PMMA IR Spectrum Analysis")
    
    # Print peak assignments
    print("\nPeak Analysis Results:")
    print("=" * 70)
    print(f"{'Peak #':<8} {'Wavenumber':<15} {'Absorbance':<15} {'Possible Assignment'}")
    print("-" * 70)
    
    assignments = identify_functional_groups([p['wavenumber'] for p in peaks])
    for i, peak in enumerate(peaks):
        wn = peak['wavenumber']
        assignment = assignments.get(wn, [])
        if assignment:
            assign_text = assignment[0]['group']
        else:
            assign_text = "Unknown"
        print(f"{i+1:<8} {wn:<15.1f} {peak['absorbance']:<15.4f} {assign_text}")
    

## Exercises

#### Exercise 1: Vibrational Frequency Calculation

Calculate the fundamental vibrational frequency (in cm-1) for the C=O bond in formaldehyde (H2CO). The force constant for C=O is approximately 1200 N/m.

View Solution
    
    
    # Solution
    import numpy as np
    
    # Physical constants
    c = 2.998e10  # cm/s
    amu_to_kg = 1.66054e-27
    
    # Given data
    k = 1200  # N/m (force constant)
    m_C = 12.01  # amu
    m_O = 16.00  # amu
    
    # Calculate reduced mass
    mu = (m_C * m_O) / (m_C + m_O) * amu_to_kg
    print(f"Reduced mass: {mu:.4e} kg")
    
    # Calculate frequency
    nu_hz = (1 / (2 * np.pi)) * np.sqrt(k / mu)
    nu_cm = nu_hz / c
    print(f"Vibrational frequency: {nu_cm:.0f} cm^-1")
    
    # Compare with experimental value (~1746 cm^-1)
    print(f"Experimental value: ~1746 cm^-1")
    print(f"Difference: {abs(nu_cm - 1746):.0f} cm^-1")
    

#### Exercise 2: IR Selection Rules

For each of the following molecules, determine how many vibrational modes are IR active and explain why:

  * (a) N2 (nitrogen)
  * (b) HCl (hydrogen chloride)
  * (c) CO2 (carbon dioxide)
  * (d) H2O (water)

View Solution

**(a) N 2:** Linear, homonuclear diatomic. 3(2)-5 = 1 vibrational mode. IR INACTIVE because symmetric stretch produces no dipole change.

**(b) HCl:** Heteronuclear diatomic. 1 vibrational mode. IR ACTIVE because the stretch changes the dipole moment (asymmetric charge distribution).

**(c) CO 2:** Linear molecule with 3(3)-5 = 4 modes: 

  * Symmetric stretch: IR INACTIVE (no dipole change)
  * Asymmetric stretch: IR ACTIVE
  * Two degenerate bending modes: IR ACTIVE

Total: 3 IR active modes (counting degeneracy as 1)

**(d) H 2O:** Nonlinear molecule with 3(3)-6 = 3 modes: 

  * Symmetric stretch: IR ACTIVE (dipole changes)
  * Asymmetric stretch: IR ACTIVE
  * Bending: IR ACTIVE

Total: 3 IR active modes (all modes are active)

#### Exercise 3: ATR Penetration Depth

Calculate the penetration depth for a diamond ATR crystal (n = 2.4) at 1000 cm-1 with a 45 degree angle of incidence. The sample is an organic polymer with n = 1.5.

View Solution
    
    
    # Solution
    import numpy as np
    
    # Given parameters
    n1 = 2.4  # Diamond refractive index
    n2 = 1.5  # Sample refractive index
    theta = 45  # degrees
    wavenumber = 1000  # cm^-1
    
    # Convert to wavelength
    wavelength_um = 10000 / wavenumber  # 10 um
    
    # Convert angle to radians
    theta_rad = np.radians(theta)
    
    # Calculate penetration depth
    sin_term = np.sin(theta_rad)**2 - (n2/n1)**2
    dp = wavelength_um / (2 * np.pi * n1 * np.sqrt(sin_term))
    
    print(f"Penetration depth: {dp:.2f} um")
    print(f"This means the IR beam probes approximately the top {dp:.2f} um of the sample")
    

#### Exercise 4: Functional Group Identification

An unknown organic compound shows IR absorption peaks at the following wavenumbers: 3350 (broad), 2920, 2850, 1710, 1465, 1380, and 1250 cm-1. Based on these peaks, propose a possible structure or functional groups present.

View Solution

**Peak assignments:**

  * **3350 cm -1 (broad):** O-H stretch, hydrogen-bonded (carboxylic acid or alcohol)
  * **2920, 2850 cm -1:** C-H stretch (alkane, CH2 and CH3)
  * **1710 cm -1:** C=O stretch (carboxylic acid)
  * **1465 cm -1:** CH2 bending
  * **1380 cm -1:** CH3 bending
  * **1250 cm -1:** C-O stretch

**Conclusion:** The compound is likely a carboxylic acid (RCOOH). The broad O-H stretch combined with the C=O at 1710 cm-1 and C-O at 1250 cm-1 are characteristic of carboxylic acids. The alkane C-H stretches suggest an aliphatic chain. A possible structure could be a fatty acid like octanoic acid (CH3(CH2)6COOH).

#### Exercise 5: Baseline Correction Practice

Write Python code to implement a simple linear baseline correction between two user-specified wavenumber points. Apply it to correct a sloped baseline in a spectrum.

View Solution
    
    
    # Solution
    import numpy as np
    import matplotlib.pyplot as plt
    
    def linear_baseline_correction(wavenumbers, absorbance, point1_wn, point2_wn):
        """
        Perform linear baseline correction using two anchor points.
    
        Parameters:
        -----------
        wavenumbers : array
            Wavenumber values
        absorbance : array
            Absorbance values
        point1_wn, point2_wn : float
            Wavenumber positions for baseline anchor points
    
        Returns:
        --------
        corrected : array
            Baseline-corrected absorbance
        baseline : array
            Calculated baseline
        """
        # Find indices closest to specified wavenumbers
        idx1 = np.argmin(np.abs(wavenumbers - point1_wn))
        idx2 = np.argmin(np.abs(wavenumbers - point2_wn))
    
        # Get baseline anchor points
        x1, y1 = wavenumbers[idx1], absorbance[idx1]
        x2, y2 = wavenumbers[idx2], absorbance[idx2]
    
        # Calculate linear baseline
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        baseline = slope * wavenumbers + intercept
    
        # Subtract baseline
        corrected = absorbance - baseline
    
        return corrected, baseline
    
    # Example usage
    wavenumbers = np.linspace(4000, 400, 1000)
    # Create spectrum with sloped baseline
    true_peak = 0.5 * np.exp(-(wavenumbers - 1700)**2 / (2 * 50**2))
    sloped_baseline = 0.2 + 0.0001 * (wavenumbers - 2000)
    noisy_spectrum = true_peak + sloped_baseline + np.random.normal(0, 0.01, len(wavenumbers))
    
    # Apply correction
    corrected, baseline = linear_baseline_correction(wavenumbers, noisy_spectrum, 3800, 600)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(wavenumbers, noisy_spectrum, 'b-', label='Original')
    ax1.plot(wavenumbers, baseline, 'r--', label='Baseline')
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax1.set_ylabel('Absorbance')
    ax1.set_title('Before Correction')
    ax1.set_xlim(4000, 400)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wavenumbers, corrected, 'g-', label='Corrected')
    ax2.plot(wavenumbers, true_peak, 'k--', alpha=0.5, label='True Peak')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_ylabel('Absorbance')
    ax2.set_title('After Correction')
    ax2.set_xlim(4000, 400)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## Summary

#### Key Takeaways from Chapter 3

  * **Molecular Vibrations:** IR spectroscopy probes molecular vibrations. Stretching modes (symmetric and asymmetric) occur at higher frequencies than bending modes (scissoring, rocking, wagging, twisting).
  * **Selection Rules:** A vibration is IR active only if it causes a change in the molecular dipole moment. Homonuclear diatomics (H2, N2) are IR inactive.
  * **FTIR Advantages:** The Fellgett (multiplex), Jacquinot (throughput), and Connes (wavelength precision) advantages make FTIR superior to dispersive instruments.
  * **Functional Group Identification:** The functional group region (4000-1500 cm-1) contains characteristic absorptions that enable identification of chemical groups.
  * **ATR-FTIR:** Enables direct measurement of samples with minimal preparation; penetration depth is wavelength and crystal dependent (typically 0.5-5 um).
  * **Data Processing:** Baseline correction and peak identification are essential preprocessing steps for quantitative IR analysis.

## References

  1. Griffiths, P. R., de Haseth, J. A. (2007). _Fourier Transform Infrared Spectrometry_ (2nd ed.). Wiley-Interscience. ISBN: 978-0-471-19404-0
  2. Stuart, B. H. (2004). _Infrared Spectroscopy: Fundamentals and Applications_. Wiley. ISBN: 978-0-470-85427-3
  3. Silverstein, R. M., Webster, F. X., Kiemle, D. J. (2014). _Spectrometric Identification of Organic Compounds_ (8th ed.). Wiley. ISBN: 978-0-470-61637-6
  4. Smith, B. C. (2011). _Fundamentals of Fourier Transform Infrared Spectroscopy_ (2nd ed.). CRC Press. ISBN: 978-1-4200-6929-7
  5. Mirabella, F. M. (1998). _Modern Techniques in Applied Molecular Spectroscopy_. Wiley. ISBN: 978-0-471-12359-0
  6. Socrates, G. (2004). _Infrared and Raman Characteristic Group Frequencies: Tables and Charts_ (3rd ed.). Wiley. ISBN: 978-0-470-09307-8
  7. Larkin, P. (2017). _Infrared and Raman Spectroscopy: Principles and Spectral Interpretation_ (2nd ed.). Elsevier. ISBN: 978-0-12-804162-8

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice.
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
