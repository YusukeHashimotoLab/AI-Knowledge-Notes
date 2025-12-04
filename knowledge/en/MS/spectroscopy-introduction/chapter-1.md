---
title: "Chapter 1: Fundamentals of Spectroscopic Analysis"
chapter_title: "Chapter 1: Fundamentals of Spectroscopic Analysis"
subtitle: Understanding the Essence of Materials through Light-Matter Interactions
difficulty: Beginner
---

This chapter covers the fundamentals of Fundamentals of Spectroscopic Analysis, which introduction. You will learn  Understand the differences,  Can explain the physical meaning, and  Understand the concept of selection rules.

## Introduction

Spectroscopy is a powerful analytical technique that utilizes the interaction between light and matter to elucidate the electronic states, chemical bonding, structure, and composition of materials. In this chapter, we will learn the quantum mechanical principles that form the foundation of all spectroscopic methods and acquire theoretical fundamentals for interpreting experimental data, including the Beer-Lambert law, transition moments, and selection rules.

**Why is Spectroscopy Important?**  
Spectroscopy is non-destructive, applicable to trace samples, and can probe different properties of materials across diverse energy ranges (from X-rays to microwaves). It serves as a foundational technology in Materials Informatics for material exploration and design, including band gap measurements in semiconductors, functional group identification in organic molecules, and chemical state analysis of catalyst surfaces. 

## 1\. Fundamentals of Light-Matter Interactions

### 1.1 Properties of Electromagnetic Waves

Light exhibits wave-particle duality as an electromagnetic wave. As a wave, light is expressed by wavelength $\lambda$ (nm) or wavenumber $\tilde{\nu}$ (cm-1), while as a particle, photons possess the following energy:

$$E = h\nu = \frac{hc}{\lambda} = hc\tilde{\nu}$$

where $h = 6.626 \times 10^{-34}$ J·s (Planck's constant) and $c = 2.998 \times 10^8$ m/s (speed of light).

**Energy Regions and Corresponding Transitions**  

  * **X-ray (0.01-10 nm)** : Core electron excitation (XPS, XRF)
  * **UV-Vis (200-800 nm)** : Valence electron excitation, HOMO-LUMO transitions
  * **Infrared (2.5-25 ¼m)** : Molecular vibrations
  * **Microwave (0.1-10 cm)** : Molecular rotations

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 1: Planck Function and Energy Calculations</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Planck's constant and speed of light
    h = 6.62607015e-34  # J·s
    c = 2.99792458e8    # m/s
    eV = 1.602176634e-19  # J (1 eV)
    
    def wavelength_to_energy(wavelength_nm):
        """
        Calculate photon energy (eV) from wavelength (nm)
    
        Parameters:
        -----------
        wavelength_nm : float or array
            Wavelength (nanometer)
    
        Returns:
        --------
        energy_eV : float or array
            Photon energy (electron volt)
        """
        wavelength_m = wavelength_nm * 1e-9
        energy_J = h * c / wavelength_m
        energy_eV = energy_J / eV
        return energy_eV
    
    def wavenumber_to_energy(wavenumber_cm):
        """
        Calculate photon energy (eV) from wavenumber (cm^-1)
    
        Parameters:
        -----------
        wavenumber_cm : float or array
            Wavenumber (cm^-1)
    
        Returns:
        --------
        energy_eV : float or array
            Photon energy (eV)
        """
        energy_J = h * c * wavenumber_cm * 100  # cm^-1 to m^-1
        energy_eV = energy_J / eV
        return energy_eV
    
    # Energy calculation for visible light region (380-780 nm)
    wavelengths = np.linspace(380, 780, 100)
    energies = wavelength_to_energy(wavelengths)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, energies, linewidth=2, color='#f093fb')
    plt.fill_between(wavelengths, energies, alpha=0.3, color='#f5576c')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Photon Energy (eV)', fontsize=12)
    plt.title('Wavelength-Energy Relationship in Visible Region', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wavelength_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Specific examples
    print(f"Red light (650 nm) energy: {wavelength_to_energy(650):.3f} eV")
    print(f"Blue light (450 nm) energy: {wavelength_to_energy(450):.3f} eV")
    print(f"IR vibration (1000 cm^-1) energy: {wavenumber_to_energy(1000):.4f} eV")
    

### 1.2 Basic Processes: Absorption, Emission, and Scattering

Light-matter interactions are classified into three main processes:
    
    
    ```mermaid
    flowchart TD
                A[Incident Light] --> B{Interaction with Matter}
                B -->|Absorption| C[Transition to Excited StateE = E‚ - E�]
                B -->|Emission| D[Transition to Ground StateFluorescence/Phosphorescence]
                B -->|Scattering| E[Rayleigh ScatteringElastic]
                B -->|Scattering| F[Raman ScatteringInelastic]
    
                style A fill:#f093fb,color:#fff
                style C fill:#f5576c,color:#fff
                style D fill:#f5576c,color:#fff
                style E fill:#a8e6cf,color:#000
                style F fill:#a8e6cf,color:#000
    ```

  * **Absorption** : When the photon energy matches the energy difference between two energy levels $\Delta E = E_2 - E_1$ of the material, the photon is absorbed and the material transitions to an excited state.
  * **Emission** : Photons are emitted during the transition from an excited state to the ground state. This includes fluorescence, phosphorescence, and chemiluminescence.
  * **Scattering** : Classified into Rayleigh scattering (elastic, no energy change) and Raman scattering (inelastic, vibrational energy change).

### 1.3 Beer-Lambert Law

The Beer-Lambert law (also called Lambert-Beer law), the fundamental law of absorption spectroscopy, describes the relationship between absorbance $A$ and sample concentration $c$ and path length $l$:

$$A = \log_{10}\left(\frac{I_0}{I}\right) = \varepsilon c l$$

where $I_0$ is the incident light intensity, $I$ is the transmitted light intensity, and $\varepsilon$ is the molar absorption coefficient (L mol-1 cm-1). Transmittance $T$ is defined as $T = I/I_0$, with the relationship $A = -\log_{10}(T)$.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 2: Beer-Lambert Law Simulation</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    def beer_lambert(I0, epsilon, concentration, path_length):
        """
        Calculate transmitted light intensity using Beer-Lambert law
    
        Parameters:
        -----------
        I0 : float
            Incident light intensity
        epsilon : float
            Molar absorption coefficient (L mol^-1 cm^-1)
        concentration : float or array
            Concentration (mol/L)
        path_length : float
            Path length (cm)
    
        Returns:
        --------
        I : float or array
            Transmitted light intensity
        A : float or array
            Absorbance
        T : float or array
            Transmittance
        """
        A = epsilon * concentration * path_length
        T = 10**(-A)
        I = I0 * T
        return I, A, T
    
    # Parameter settings
    I0 = 1.0  # Incident light intensity (normalized)
    epsilon = 1000  # Molar absorption coefficient (typical organic dye)
    path_length = 1.0  # Path length 1 cm
    concentrations = np.linspace(0, 1e-4, 100)  # Concentration range (mol/L)
    
    # Calculation
    I, A, T = beer_lambert(I0, epsilon, concentrations, path_length)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absorbance vs concentration
    ax1.plot(concentrations * 1e6, A, linewidth=2, color='#f093fb', label='Absorbance')
    ax1.set_xlabel('Concentration (¼mol/L)', fontsize=12)
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.set_title('Beer-Lambert Law: Concentration Dependence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Transmittance vs concentration
    ax2.plot(concentrations * 1e6, T * 100, linewidth=2, color='#f5576c', label='Transmittance')
    ax2.set_xlabel('Concentration (¼mol/L)', fontsize=12)
    ax2.set_ylabel('Transmittance (%)', fontsize=12)
    ax2.set_title('Concentration Dependence of Transmittance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('beer_lambert.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Quantitative analysis example: Calculate concentration from absorbance
    measured_A = 0.5
    calculated_c = measured_A / (epsilon * path_length)
    print(f"Measured absorbance: {measured_A}")
    print(f"Calculated concentration: {calculated_c * 1e6:.2f} ¼mol/L")
    

## 2\. Quantum Mechanical Foundations

### 2.1 Electronic and Vibrational States

Molecular energy levels can be separated into three degrees of freedom: electronic states, vibrational states, and rotational states (Born-Oppenheimer approximation):

$$E_{\text{total}} = E_{\text{electronic}} + E_{\text{vibrational}} + E_{\text{rotational}}$$

Typical energy scales are as follows:

  * $E_{\text{electronic}} \sim 1-10$ eV (UV-Vis region)
  * $E_{\text{vibrational}} \sim 0.05-0.5$ eV (infrared region)
  * $E_{\text{rotational}} \sim 10^{-4}-10^{-3}$ eV (microwave region)

### 2.2 Transition Moments and Fermi's Golden Rule

The transition probability from state $\left|\psi_i\right\rangle$ to $\left|\psi_f\right\rangle$ due to light absorption is given by Fermi's golden rule:

$$W_{i \to f} = \frac{2\pi}{\hbar} \left| \left\langle \psi_f \right| \hat{H}_{\text{int}} \left| \psi_i \right\rangle \right|^2 \rho(E_f)$$

where $\hat{H}_{\text{int}}$ is the light-matter interaction Hamiltonian and $\rho(E_f)$ is the density of final states. In the electric dipole approximation, the transition dipole moment $\boldsymbol{\mu}_{fi}$ becomes important:

$$\boldsymbol{\mu}_{fi} = \left\langle \psi_f \right| \hat{\boldsymbol{\mu}} \left| \psi_i \right\rangle = \int \psi_f^* \hat{\boldsymbol{\mu}} \psi_i \, d\tau$$

When the transition dipole moment is non-zero ($\boldsymbol{\mu}_{fi} \neq 0$), the transition is called "allowed transition"; when zero, it is called "forbidden transition".
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 3: Transition Probability Calculation Using Fermi's Golden Rule</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    hbar = 1.054571817e-34  # J·s
    eV = 1.602176634e-19    # J
    
    def transition_probability(transition_dipole_moment, state_density, energy_eV):
        """
        Calculate transition probability using Fermi's golden rule (simplified version)
    
        Parameters:
        -----------
        transition_dipole_moment : float
            Transition dipole moment (Debye)
        state_density : float
            Density of states (eV^-1)
        energy_eV : float
            Transition energy (eV)
    
        Returns:
        --------
        W : float
            Transition probability (s^-1)
        """
        # Convert Debye to C·m (1 D H 3.336e-30 C·m)
        mu = transition_dipole_moment * 3.336e-30
    
        # Simplified transition probability (electric dipole approximation)
        # Actual calculation also considers electric field intensity
        W = (2 * np.pi / hbar) * (mu**2) * state_density * eV
        return W
    
    def franck_condon_factor(n_initial, n_final, displacement):
        """
        Approximate calculation of Franck-Condon factor (harmonic oscillator approximation)
    
        Parameters:
        -----------
        n_initial : int
            Initial vibrational quantum number
        n_final : int
            Final vibrational quantum number
        displacement : float
            Displacement of equilibrium position (dimensionless)
    
        Returns:
        --------
        fc_factor : float
            Franck-Condon factor
        """
        # Simplification: Gaussian approximation
        delta_n = n_final - n_initial
        fc_factor = np.exp(-displacement**2 / 2) * (displacement**delta_n / np.math.factorial(abs(delta_n)))
        return abs(fc_factor)**2
    
    # Relationship between transition dipole moment and transition probability
    dipole_moments = np.linspace(0.1, 5.0, 50)  # Debye
    state_density = 1e15  # eV^-1 (typical value for solids)
    energy = 2.0  # eV
    
    transition_probs = [transition_probability(mu, state_density, energy) for mu in dipole_moments]
    
    # Calculate Franck-Condon factors (v=0 ’ v' transitions)
    displacements = np.linspace(0, 3, 4)
    vibrational_levels = np.arange(0, 8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transition dipole moment and transition probability
    ax1.plot(dipole_moments, np.array(transition_probs) * 1e-15, linewidth=2, color='#f093fb')
    ax1.set_xlabel('Transition Dipole Moment (Debye)', fontsize=12)
    ax1.set_ylabel('Transition Probability (×10¹u s{¹)', fontsize=12)
    ax1.set_title('Transition Dipole Moment vs Transition Probability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Franck-Condon factors
    for d in displacements:
        fc_factors = [franck_condon_factor(0, v, d) for v in vibrational_levels]
        ax2.plot(vibrational_levels, fc_factors, marker='o', linewidth=2, label=f'”q = {d:.1f}')
    
    ax2.set_xlabel("Final Vibrational Quantum Number v'", fontsize=12)
    ax2.set_ylabel('Franck-Condon Factor', fontsize=12)
    ax2.set_title('Franck-Condon Factor Calculation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transition_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 2.3 Selection Rules

Selection rules are quantum mechanical rules that determine which transitions are allowed and which are forbidden. The main selection rules are shown below:

**Major Selection Rules**

  * **Electric dipole transitions (UV-Vis, IR)** : $\Delta l = \pm 1$ (orbital angular momentum quantum number), $\Delta S = 0$ (spin), $\Delta v = \pm 1$ (vibrational quantum number, harmonic oscillator approximation)
  * **Laporte rule (centrosymmetric molecules)** : Only $g \leftrightarrow u$ allowed ($g \leftrightarrow g$ or $u \leftrightarrow u$ forbidden)
  * **Spin selection rule** : Singlet-singlet and triplet-triplet transitions allowed, singlet-triplet transitions forbidden (though relaxed by spin-orbit coupling)
  * **Raman scattering** : Polarizability change $\partial \alpha / \partial Q \neq 0$ (complementary to IR absorption)

### 2.4 Franck-Condon Principle

The Franck-Condon principle states that electronic transitions are very fast (~10-15 s) compared to the timescale of vibrations (~10-13 s), so the nuclear positions hardly change during electronic transitions. This results in vibrational structure appearing in absorption and emission spectra.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 4: Absorption Spectrum Simulation Based on Franck-Condon Principle</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import factorial
    
    def harmonic_oscillator_wavefunction(x, n, omega=1.0, m=1.0, hbar=1.0):
        """
        Harmonic oscillator wavefunction
    
        Parameters:
        -----------
        x : array
            Coordinate
        n : int
            Quantum number
        omega : float
            Angular frequency
        m : float
            Mass
        hbar : float
            Reduced Planck's constant
    
        Returns:
        --------
        psi : array
            Wavefunction
        """
        alpha = np.sqrt(m * omega / hbar)
        norm = (alpha / np.pi)**0.25 / np.sqrt(2**n * factorial(n))
        hermite = np.polynomial.hermite.hermval(alpha * x, [0]*n + [1])
        psi = norm * hermite * np.exp(-alpha**2 * x**2 / 2)
        return psi
    
    def franck_condon_spectrum(displacement=1.5, n_levels=6):
        """
        Absorption spectrum simulation based on Franck-Condon principle
    
        Parameters:
        -----------
        displacement : float
            Displacement of equilibrium positions between excited and ground states
        n_levels : int
            Number of vibrational levels to consider
    
        Returns:
        --------
        energies : array
            Transition energies
        intensities : array
            Relative intensities
        """
        # Coordinate grid
        x = np.linspace(-6, 6, 1000)
    
        # Ground state v=0 wavefunction
        psi_ground = harmonic_oscillator_wavefunction(x, 0)
    
        energies = []
        intensities = []
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Potential energy curves
        V_ground = 0.5 * x**2
        V_excited = 0.5 * (x - displacement)**2 + 3.0  # Excited state (upward shift)
    
        ax1.plot(x, V_ground, 'b-', linewidth=2, label='Ground State')
        ax1.plot(x, V_excited, 'r-', linewidth=2, label='Excited State')
    
        # Transitions to each vibrational level
        for n in range(n_levels):
            # Excited state v=n wavefunction (equilibrium position shifted by displacement)
            psi_excited = harmonic_oscillator_wavefunction(x - displacement, n)
    
            # Franck-Condon integral (overlap integral)
            fc_integral = np.trapz(psi_ground * psi_excited, x)
            fc_factor = fc_integral**2
    
            # Transition energy (electronic transition + vibrational energy)
            E_transition = 3.0 + n * 0.2  # in eV
    
            energies.append(E_transition)
            intensities.append(fc_factor)
    
            # Draw vibrational levels on potential curves
            E_vib_ground = 0.5
            E_vib_excited = 3.0 + n * 0.2
            ax1.axhline(y=E_vib_excited, xmin=0.5, xmax=0.9, color='red', alpha=0.3, linewidth=1)
    
            # Transition arrows
            if n < 4:
                ax1.annotate('', xy=(displacement, E_vib_excited), xytext=(0, E_vib_ground),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.5, lw=1.5))
    
        ax1.set_xlabel('Nuclear Coordinate (a.u.)', fontsize=12)
        ax1.set_ylabel('Energy (eV)', fontsize=12)
        ax1.set_title('Franck-Condon Principle', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
    
        # Absorption spectrum
        energies = np.array(energies)
        intensities = np.array(intensities)
    
        # Gaussian broadening
        E_range = np.linspace(2.8, 4.5, 500)
        spectrum = np.zeros_like(E_range)
        broadening = 0.05  # eV
    
        for E, I in zip(energies, intensities):
            spectrum += I * np.exp(-(E_range - E)**2 / (2 * broadening**2))
    
        ax2.plot(E_range, spectrum, linewidth=2, color='#f093fb')
        ax2.fill_between(E_range, spectrum, alpha=0.3, color='#f5576c')
        ax2.set_xlabel('Photon Energy (eV)', fontsize=12)
        ax2.set_ylabel('Absorption Intensity (a.u.)', fontsize=12)
        ax2.set_title('Simulated Absorption Spectrum', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('franck_condon_spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return energies, intensities
    
    # Execute
    energies, intensities = franck_condon_spectrum(displacement=1.5)
    print("Transition energies (eV):", energies)
    print("Franck-Condon factors:", intensities)
    

## 3\. Classification of Spectroscopic Methods

### 3.1 Classification by Energy Region
    
    
    ```mermaid
    flowchart LR
                A[Electromagnetic Spectrum] --> B[X-ray0.01-10 nm]
                A --> C[UV-Vis200-800 nm]
                A --> D[Infrared2.5-25 ¼m]
                A --> E[Microwave0.1-10 cm]
    
                B --> B1[XPSChemical State]
                B --> B2[XRFElemental Analysis]
    
                C --> C1[UV-VisElectronic Transitions]
                C --> C2[PLEmission]
    
                D --> D1[FTIRVibrations]
                D --> D2[RamanVibrations]
    
                E --> E1[ESRMagnetic Resonance]
                E --> E2[NMRNuclear Spin]
    
                style A fill:#f093fb,color:#fff
                style B fill:#ff6b6b,color:#fff
                style C fill:#4ecdc4,color:#fff
                style D fill:#ffe66d,color:#000
                style E fill:#a8e6cf,color:#000
    ```

### 3.2 Classification by Measurement Principle

  * **Absorption Spectroscopy** : UV-Vis, FTIR, Atomic Absorption Spectroscopy (AAS)
  * **Emission Spectroscopy** : Fluorescence (PL), Phosphorescence, Chemiluminescence
  * **Scattering Spectroscopy** : Raman, Brillouin scattering
  * **Resonance Spectroscopy** : NMR, ESR

## 4\. How to Read Spectra

### 4.1 Conversion of Horizontal and Vertical Axes

Spectroscopic data are expressed in various unit systems:

  * **Horizontal axis** : Wavelength $\lambda$ (nm), wavenumber $\tilde{\nu}$ (cm-1), energy $E$ (eV), frequency $\nu$ (Hz)
  * **Vertical axis** : Transmittance $T$ (%), absorbance $A$, intensity $I$ (a.u.), molar absorption coefficient $\varepsilon$

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 5: Wavelength-Wavenumber-Energy Conversion Calculator</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    class SpectroscopyUnitConverter:
        """Spectroscopy unit conversion class"""
    
        def __init__(self):
            self.h = 6.62607015e-34  # J·s
            self.c = 2.99792458e8    # m/s
            self.eV = 1.602176634e-19  # J
    
        def wavelength_to_wavenumber(self, wavelength_nm):
            """Wavelength (nm) ’ Wavenumber (cm^-1)"""
            wavelength_cm = wavelength_nm * 1e-7
            return 1 / wavelength_cm
    
        def wavenumber_to_wavelength(self, wavenumber_cm):
            """Wavenumber (cm^-1) ’ Wavelength (nm)"""
            wavelength_cm = 1 / wavenumber_cm
            return wavelength_cm * 1e7
    
        def wavelength_to_energy_eV(self, wavelength_nm):
            """Wavelength (nm) ’ Energy (eV)"""
            wavelength_m = wavelength_nm * 1e-9
            energy_J = self.h * self.c / wavelength_m
            return energy_J / self.eV
    
        def energy_eV_to_wavelength(self, energy_eV):
            """Energy (eV) ’ Wavelength (nm)"""
            energy_J = energy_eV * self.eV
            wavelength_m = self.h * self.c / energy_J
            return wavelength_m * 1e9
    
        def wavelength_to_frequency(self, wavelength_nm):
            """Wavelength (nm) ’ Frequency (Hz)"""
            wavelength_m = wavelength_nm * 1e-9
            return self.c / wavelength_m
    
        def transmittance_to_absorbance(self, T):
            """Transmittance (%) ’ Absorbance"""
            T_fraction = T / 100
            return -np.log10(T_fraction)
    
        def absorbance_to_transmittance(self, A):
            """Absorbance ’ Transmittance (%)"""
            return 10**(-A) * 100
    
    # Instantiate converter
    converter = SpectroscopyUnitConverter()
    
    # Conversion table for UV-Vis region
    wavelengths_nm = np.array([200, 250, 300, 400, 500, 600, 700, 800])
    wavenumbers_cm = converter.wavelength_to_wavenumber(wavelengths_nm)
    energies_eV = converter.wavelength_to_energy_eV(wavelengths_nm)
    frequencies_THz = converter.wavelength_to_frequency(wavelengths_nm) / 1e12
    
    print("=" * 70)
    print("UV-Vis Region Unit Conversion Table")
    print("=" * 70)
    print(f"{'Wavelength (nm)':<12} {'Wavenumber (cm{¹)':<15} {'Energy (eV)':<15} {'Frequency (THz)':<12}")
    print("-" * 70)
    for wl, wn, E, f in zip(wavelengths_nm, wavenumbers_cm, energies_eV, frequencies_THz):
        print(f"{wl:<12.0f} {wn:<15.0f} {E:<15.2f} {f:<12.1f}")
    
    # Conversion table for IR region
    print("\n" + "=" * 70)
    print("Infrared Region Unit Conversion Table")
    print("=" * 70)
    wavenumbers_IR = np.array([4000, 3000, 2000, 1500, 1000, 500])
    wavelengths_IR = converter.wavenumber_to_wavelength(wavenumbers_IR)
    energies_IR_eV = converter.wavelength_to_energy_eV(wavelengths_IR)
    
    print(f"{'Wavenumber (cm{¹)':<15} {'Wavelength (¼m)':<15} {'Energy (eV)':<15}")
    print("-" * 70)
    for wn, wl, E in zip(wavenumbers_IR, wavelengths_IR / 1000, energies_IR_eV):
        print(f"{wn:<15.0f} {wl:<15.2f} {E:<15.4f}")
    
    # Relationship between transmittance and absorbance
    transmittances = np.linspace(1, 100, 100)
    absorbances = converter.transmittance_to_absorbance(transmittances)
    
    plt.figure(figsize=(10, 6))
    plt.plot(transmittances, absorbances, linewidth=2, color='#f093fb')
    plt.xlabel('Transmittance (%)', fontsize=12)
    plt.ylabel('Absorbance', fontsize=12)
    plt.title('Relationship between Transmittance and Absorbance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('transmittance_absorbance.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 4.2 Interpretation of Peak Position, Intensity, and Width

The following information can be obtained from spectral peaks:

  * **Peak position** : Energy level difference $\Delta E$, type of chemical bonding, band gap
  * **Peak intensity** : Transition probability, concentration, molar absorption coefficient
  * **Peak width (FWHM)** : Inhomogeneous broadening (crystallinity, defects), homogeneous broadening (lifetime)

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 6: Gaussian and Lorentzian Lineshape Fitting</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def gaussian(x, amplitude, center, width):
        """
        Gaussian lineshape function
    
        Parameters:
        -----------
        x : array
            Horizontal axis (wavelength, energy, etc.)
        amplitude : float
            Peak height
        center : float
            Peak center position
        width : float
            Standard deviation (FWHM = 2.355 * width)
    
        Returns:
        --------
        y : array
            Gaussian curve
        """
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
    
    def lorentzian(x, amplitude, center, width):
        """
        Lorentzian lineshape function
    
        Parameters:
        -----------
        x : array
            Horizontal axis
        amplitude : float
            Peak height
        center : float
            Peak center position
        width : float
            Half width at half maximum (HWHM)
    
        Returns:
        --------
        y : array
            Lorentzian curve
        """
        return amplitude * (width**2 / ((x - center)**2 + width**2))
    
    def voigt(x, amplitude, center, width_g, width_l):
        """
        Voigt lineshape function (convolution of Gaussian and Lorentzian)
        Simplified version: pseudo-Voigt
    
        Parameters:
        -----------
        x : array
            Horizontal axis
        amplitude : float
            Peak height
        center : float
            Peak center position
        width_g : float
            Gaussian component width
        width_l : float
            Lorentzian component width
    
        Returns:
        --------
        y : array
            Voigt curve
        """
        # pseudo-Voigt: Linear combination of Gaussian and Lorentzian
        eta = 0.5  # mixing parameter
        g = gaussian(x, amplitude, center, width_g)
        l = lorentzian(x, amplitude, center, width_l)
        return eta * l + (1 - eta) * g
    
    # Generate synthetic spectrum (with noise)
    x_data = np.linspace(400, 700, 300)  # Wavelength (nm)
    true_params = {
        'amplitude': 1.0,
        'center': 550,
        'width': 30
    }
    
    # Gaussian + noise
    y_data = gaussian(x_data, **true_params) + np.random.normal(0, 0.02, len(x_data))
    
    # Fitting
    initial_guess = [0.8, 540, 25]
    
    popt_gauss, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    popt_lorentz, _ = curve_fit(lorentzian, x_data, y_data, p0=initial_guess)
    
    # Visualization of results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fitting results
    ax1.scatter(x_data, y_data, s=10, alpha=0.5, color='gray', label='Experimental Data')
    ax1.plot(x_data, gaussian(x_data, *popt_gauss), 'r-', linewidth=2, label='Gaussian Fit')
    ax1.plot(x_data, lorentzian(x_data, *popt_lorentz), 'b--', linewidth=2, label='Lorentzian Fit')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Absorbance (a.u.)', fontsize=12)
    ax1.set_title('Peak Shape Fitting', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Comparison of lineshape functions
    x_comparison = np.linspace(-100, 100, 500)
    y_gauss = gaussian(x_comparison, 1.0, 0, 20)
    y_lorentz = lorentzian(x_comparison, 1.0, 0, 20)
    y_voigt = voigt(x_comparison, 1.0, 0, 20, 10)
    
    ax2.plot(x_comparison, y_gauss, 'r-', linewidth=2, label='Gaussian')
    ax2.plot(x_comparison, y_lorentz, 'b-', linewidth=2, label='Lorentzian')
    ax2.plot(x_comparison, y_voigt, 'g-', linewidth=2, label='Voigt (pseudo)')
    ax2.set_xlabel('Relative Position', fontsize=12)
    ax2.set_ylabel('Normalized Intensity', fontsize=12)
    ax2.set_title('Comparison of Lineshape Functions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 2)
    
    plt.tight_layout()
    plt.savefig('peak_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output fitting parameters
    print("Gaussian fit results:")
    print(f"  Amplitude: {popt_gauss[0]:.3f}")
    print(f"  Center: {popt_gauss[1]:.1f} nm")
    print(f"  FWHM: {2.355 * popt_gauss[2]:.1f} nm")
    
    print("\nLorentzian fit results:")
    print(f"  Amplitude: {popt_lorentz[0]:.3f}")
    print(f"  Center: {popt_lorentz[1]:.1f} nm")
    print(f"  FWHM: {2 * popt_lorentz[2]:.1f} nm")
    

### 4.3 Importance of Baseline Processing

Measured spectra contain baselines due to absorption by sample holders, scattering, instrumental drift, etc. Baseline correction is essential for accurate quantitative analysis.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 7: Baseline Correction (Polynomial Fitting)</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def polynomial_baseline(x, y, degree=2, exclude_peaks=True):
        """
        Baseline correction using polynomial fitting
    
        Parameters:
        -----------
        x : array
            Horizontal axis data
        y : array
            Vertical axis data (spectrum)
        degree : int
            Degree of polynomial
        exclude_peaks : bool
            Exclude peak regions for fitting
    
        Returns:
        --------
        baseline : array
            Estimated baseline
        corrected : array
            Baseline-corrected spectrum
        """
        if exclude_peaks:
            # Peak detection
            peaks, _ = find_peaks(y, prominence=0.1 * np.max(y))
    
            # Mask excluding peak regions
            mask = np.ones(len(y), dtype=bool)
            window = int(len(y) * 0.05)  # Exclude 5% around peaks
            for peak in peaks:
                start = max(0, peak - window)
                end = min(len(y), peak + window)
                mask[start:end] = False
    
            # Polynomial fitting in masked regions
            coeffs = np.polyfit(x[mask], y[mask], degree)
        else:
            coeffs = np.polyfit(x, y, degree)
    
        baseline = np.polyval(coeffs, x)
        corrected = y - baseline
    
        return baseline, corrected
    
    # Synthetic spectrum (with baseline)
    x = np.linspace(400, 700, 500)
    
    # True spectrum (two peaks)
    true_spectrum = gaussian(x, 0.8, 500, 25) + gaussian(x, 0.5, 600, 20)
    
    # Baseline (2nd order polynomial)
    baseline_true = 0.1 + 0.0005 * (x - 550) + 0.000001 * (x - 550)**2
    
    # Observed spectrum = true spectrum + baseline + noise
    observed_spectrum = true_spectrum + baseline_true + np.random.normal(0, 0.01, len(x))
    
    # Baseline correction
    baseline_estimated, corrected_spectrum = polynomial_baseline(x, observed_spectrum, degree=2, exclude_peaks=True)
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Observed spectrum
    ax1.plot(x, observed_spectrum, 'k-', linewidth=1.5, label='Observed Spectrum')
    ax1.plot(x, baseline_true, 'r--', linewidth=2, label='True Baseline')
    ax1.plot(x, baseline_estimated, 'b--', linewidth=2, label='Estimated Baseline')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('Baseline Correction: Observed Spectrum', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Corrected spectrum
    ax2.plot(x, corrected_spectrum, 'g-', linewidth=2, label='Corrected Spectrum')
    ax2.plot(x, true_spectrum, 'r--', linewidth=2, alpha=0.7, label='True Spectrum')
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('After Baseline Correction', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residual analysis
    residual = corrected_spectrum - true_spectrum
    ax3.plot(x, residual, 'purple', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.fill_between(x, residual, alpha=0.3, color='purple')
    ax3.set_xlabel('Wavelength (nm)', fontsize=12)
    ax3.set_ylabel('Residual (a.u.)', fontsize=12)
    ax3.set_title('Correction Residual', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_correction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistics
    print(f"RMS error before baseline correction: {np.sqrt(np.mean((observed_spectrum - true_spectrum)**2)):.4f}")
    print(f"RMS error after baseline correction: {np.sqrt(np.mean(residual**2)):.4f}")
    

## 5\. Analysis of Multi-Peak Spectra
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    
                <h4>Code Example 8: Automatic Multi-Peak Detection and Fitting</h4>
                <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    
    def multi_gaussian(x, *params):
        """
        Sum of multiple Gaussian functions
    
        Parameters:
        -----------
        x : array
            Horizontal axis
        params : tuple
            (amplitude1, center1, width1, amplitude2, center2, width2, ...)
    
        Returns:
        --------
        y : array
            Sum of multiple Gaussians
        """
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amplitude = params[i]
            center = params[i+1]
            width = params[i+2]
            y += gaussian(x, amplitude, center, width)
        return y
    
    def auto_peak_fitting(x, y, min_prominence=0.1):
        """
        Automatic peak detection and multi-Gaussian fitting
    
        Parameters:
        -----------
        x : array
            Horizontal axis data
        y : array
            Vertical axis data
        min_prominence : float
            Minimum prominence for peak detection (relative to maximum)
    
        Returns:
        --------
        params : array
            Fitting parameters
        fitted : array
            Fitted curve
        individual_peaks : list of arrays
            Individual peak components
        """
        # Baseline correction
        baseline, y_corrected = polynomial_baseline(x, y, degree=1, exclude_peaks=True)
    
        # Peak detection
        peaks, properties = find_peaks(y_corrected, prominence=min_prominence * np.max(y_corrected))
    
        print(f"Number of detected peaks: {len(peaks)}")
    
        # Initial guess
        initial_params = []
        for peak in peaks:
            amplitude = y_corrected[peak]
            center = x[peak]
            width = 20  # Initial width estimate
            initial_params.extend([amplitude, center, width])
    
        # Multi-Gaussian fitting
        try:
            popt, _ = curve_fit(multi_gaussian, x, y_corrected, p0=initial_params, maxfev=10000)
            fitted = multi_gaussian(x, *popt)
    
            # Individual peak components
            individual_peaks = []
            for i in range(0, len(popt), 3):
                peak_component = gaussian(x, popt[i], popt[i+1], popt[i+2])
                individual_peaks.append(peak_component)
    
            return popt, fitted, individual_peaks, baseline
    
        except RuntimeError:
            print("Fitting failed: did not converge")
            return None, None, None, baseline
    
    # Generate complex spectrum (4 peaks)
    x_data = np.linspace(400, 700, 600)
    true_components = [
        gaussian(x_data, 0.7, 450, 20),
        gaussian(x_data, 1.0, 520, 25),
        gaussian(x_data, 0.6, 580, 18),
        gaussian(x_data, 0.4, 650, 22)
    ]
    true_spectrum = sum(true_components)
    baseline = 0.05 + 0.0001 * x_data
    observed = true_spectrum + baseline + np.random.normal(0, 0.02, len(x_data))
    
    # Automatic fitting
    params, fitted, individual, baseline_est = auto_peak_fitting(x_data, observed, min_prominence=0.15)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Fitting results
    ax1.plot(x_data, observed, 'k.', markersize=2, alpha=0.5, label='Observed Data')
    if fitted is not None:
        ax1.plot(x_data, fitted + baseline_est, 'r-', linewidth=2, label='Fitted Curve')
        for i, peak in enumerate(individual):
            ax1.plot(x_data, peak + baseline_est, '--', linewidth=1.5, alpha=0.7, label=f'Peak {i+1}')
    ax1.plot(x_data, baseline_est, 'g--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('Automatic Fitting of Multi-Peak Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Separated peak components
    if individual is not None:
        for i, peak in enumerate(individual):
            ax2.plot(x_data, peak, linewidth=2, label=f'Peak {i+1}')
            ax2.fill_between(x_data, peak, alpha=0.3)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('Separated Peak Components', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_peak_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed fitting results
    if params is not None:
        print("\n" + "=" * 60)
        print("Fitting Results")
        print("=" * 60)
        for i in range(0, len(params), 3):
            peak_num = i // 3 + 1
            amplitude = params[i]
            center = params[i+1]
            width = params[i+2]
            fwhm = 2.355 * width
            area = amplitude * width * np.sqrt(2 * np.pi)
            print(f"\nPeak {peak_num}:")
            print(f"  Center position: {center:.1f} nm")
            print(f"  Amplitude: {amplitude:.3f}")
            print(f"  FWHM: {fwhm:.1f} nm")
            print(f"  Area: {area:.3f}")
    

## Exercises

**Exercises (click to expand)**

### Easy Level (Basic Calculations)

**Problem 1** : Calculate the photon energy in eV for light with a wavelength of 500 nm. Also, what color in the visible spectrum does this light correspond to?

View solution

**Solution** :
    
    
    h = 6.626e-34  # J·s
    c = 2.998e8    # m/s
    eV = 1.602e-19  # J
    
    wavelength = 500e-9  # m
    E = h * c / wavelength / eV
    print(f"Photon energy: {E:.2f} eV")
    # Output: Photon energy: 2.48 eV
    # 500 nm corresponds to green light
    

**Problem 2** : In the Beer-Lambert law, if the molar absorption coefficient $\varepsilon = 50000$ L mol-1 cm-1, path length $l = 1$ cm, and absorbance $A = 0.8$, find the sample concentration (mol/L).

View solution

**Solution** :
    
    
    epsilon = 50000  # L mol^-1 cm^-1
    l = 1.0  # cm
    A = 0.8
    
    c = A / (epsilon * l)
    print(f"Concentration: {c:.2e} mol/L = {c * 1e6:.2f} ¼mol/L")
    # Output: Concentration: 1.60e-05 mol/L = 16.00 ¼mol/L
    

**Problem 3** : In infrared spectroscopy, a peak is observed at wavenumber 1650 cm-1. Calculate the wavelength (¼m) and energy (eV) corresponding to this wavenumber.

View solution

**Solution** :
    
    
    wavenumber = 1650  # cm^-1
    
    # Wavenumber to wavelength
    wavelength_cm = 1 / wavenumber
    wavelength_um = wavelength_cm * 1e4
    print(f"Wavelength: {wavelength_um:.2f} ¼m")
    
    # Energy
    h = 6.626e-34
    c = 2.998e8
    eV = 1.602e-19
    energy_J = h * c * wavenumber * 100  # cm^-1 to m^-1
    energy_eV = energy_J / eV
    print(f"Energy: {energy_eV:.4f} eV")
    # Output: Wavelength: 6.06 ¼m, Energy: 0.2045 eV
    

### Medium Level (Practical Calculations)

**Problem 4** : The data below show measurements of concentration and absorbance for a solution. Create a calibration curve using the Beer-Lambert law and estimate the concentration of an unknown sample (absorbance 0.65).

Concentration (¼mol/L)| 10| 20| 30| 40| 50  
---|---|---|---|---|---  
Absorbance| 0.18| 0.35| 0.53| 0.71| 0.88  
View solution

**Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # Data
    concentrations = np.array([10, 20, 30, 40, 50])  # ¼mol/L
    absorbances = np.array([0.18, 0.35, 0.53, 0.71, 0.88])
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(concentrations, absorbances)
    
    print(f"Calibration curve: A = {slope:.4f} * C + {intercept:.4f}")
    print(f"Correlation coefficient R²: {r_value**2:.4f}")
    
    # Estimate concentration of unknown sample
    unknown_A = 0.65
    unknown_C = (unknown_A - intercept) / slope
    print(f"Unknown sample concentration: {unknown_C:.1f} ¼mol/L")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(concentrations, absorbances, s=100, color='#f093fb', label='Measured Data')
    plt.plot(concentrations, slope * concentrations + intercept, 'r-', linewidth=2, label=f'Calibration Curve (R²={r_value**2:.3f})')
    plt.scatter([unknown_C], [unknown_A], s=150, color='#f5576c', marker='*', label='Unknown Sample', zorder=5)
    plt.xlabel('Concentration (¼mol/L)', fontsize=12)
    plt.ylabel('Absorbance', fontsize=12)
    plt.title('Creating Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Output: Unknown sample concentration: 36.4 ¼mol/L
    

**Problem 5** : Suppose an absorption spectrum can be approximated by the following Gaussian function. Calculate the peak center, FWHM, and integrated intensity.

$$I(\lambda) = 0.8 \exp\left(-\frac{(\lambda - 520)^2}{2 \times 25^2}\right)$$

View solution

**Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.integrate import quad
    
    # Gaussian function parameters
    amplitude = 0.8
    center = 520  # nm
    sigma = 25  # nm
    
    # FWHM calculation
    fwhm = 2.355 * sigma
    print(f"Peak center: {center} nm")
    print(f"FWHM: {fwhm:.1f} nm")
    
    # Integrated intensity (analytical solution)
    integral_analytical = amplitude * sigma * np.sqrt(2 * np.pi)
    print(f"Integrated intensity (analytical): {integral_analytical:.2f}")
    
    # Verification by numerical integration
    def gaussian_func(x):
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    integral_numerical, error = quad(gaussian_func, 0, 1000)
    print(f"Integrated intensity (numerical): {integral_numerical:.2f}")
    
    # Output: Peak center: 520 nm, FWHM: 58.9 nm, Integrated intensity: 50.13
    

**Problem 6** : Based on the Franck-Condon principle, calculate transition intensities from the ground state (v=0) to different vibrational levels of the excited state (v'=0, 1, 2, 3). Assume the equilibrium position of the excited state is displaced by 1.2 in dimensionless coordinates from the ground state.

View solution

**Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def franck_condon_factor_harmonic(n_i, n_f, displacement):
        """
        Franck-Condon factor (harmonic oscillator approximation)
        Simplified formula for n_i = 0
        """
        from scipy.special import factorial
    
        if n_i == 0:
            S = displacement**2 / 2  # Huang-Rhys factor
            fc = np.exp(-S) * (S**n_f) / factorial(n_f)
        return fc
    
    displacement = 1.2
    vibrational_levels = [0, 1, 2, 3]
    
    print("Franck-Condon factors (v=0 ’ v' transitions):")
    print("=" * 40)
    for v_f in vibrational_levels:
        fc = franck_condon_factor_harmonic(0, v_f, displacement)
        print(f"v=0 ’ v'={v_f}: {fc:.4f}")
    
    # Normalized relative intensities
    fc_values = [franck_condon_factor_harmonic(0, v, displacement) for v in vibrational_levels]
    fc_normalized = np.array(fc_values) / np.max(fc_values)
    print("\nRelative intensities (normalized to maximum):")
    for v, intensity in zip(vibrational_levels, fc_normalized):
        print(f"v'={v}: {intensity:.2f}")
    

### Hard Level (Advanced Analysis)

**Problem 7** : Generate a synthetic spectrum consisting of two Gaussian peaks (centers: 500 nm, 550 nm; width: 20 nm; amplitude ratio 2:1) and add noise (standard deviation 0.05). Then perform two-component fitting using scipy.optimize.curve_fit to recover the original parameters.

View solution

**Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def two_gaussian(x, A1, c1, w1, A2, c2, w2):
        """Sum of two Gaussians"""
        g1 = A1 * np.exp(-(x - c1)**2 / (2 * w1**2))
        g2 = A2 * np.exp(-(x - c2)**2 / (2 * w2**2))
        return g1 + g2
    
    # True parameters
    true_params = [1.0, 500, 20, 0.5, 550, 20]
    
    # Synthetic spectrum
    x = np.linspace(400, 650, 300)
    y_true = two_gaussian(x, *true_params)
    y_noisy = y_true + np.random.normal(0, 0.05, len(x))
    
    # Fitting (initial guess)
    initial_guess = [0.8, 495, 18, 0.4, 545, 22]
    popt, pcov = curve_fit(two_gaussian, x, y_noisy, p0=initial_guess)
    
    # Visualization of results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_noisy, 'k.', markersize=3, alpha=0.5, label='Noisy Data')
    plt.plot(x, y_true, 'g--', linewidth=2, label='True Spectrum')
    plt.plot(x, two_gaussian(x, *popt), 'r-', linewidth=2, label='Fitted Result')
    
    # Individual components
    plt.plot(x, popt[0] * np.exp(-(x - popt[1])**2 / (2 * popt[2]**2)), 'b--', alpha=0.7, label='Peak 1')
    plt.plot(x, popt[3] * np.exp(-(x - popt[4])**2 / (2 * popt[5]**2)), 'm--', alpha=0.7, label='Peak 2')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Two-Component Gaussian Fitting', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Parameter comparison
    print("Parameter recovery results:")
    print("=" * 60)
    labels = ['Amplitude1', 'Center1', 'Width1', 'Amplitude2', 'Center2', 'Width2']
    for label, true, fitted in zip(labels, true_params, popt):
        error = abs(fitted - true) / true * 100
        print(f"{label:8s}: True={true:6.2f}, Fitted={fitted:6.2f}, Error={error:5.2f}%")
    

**Problem 8** : Based on selection rules, determine whether the following transitions are allowed or forbidden.

  * (a) 1s ’ 2p transition in hydrogen atom
  * (b) 1s ’ 2s transition in hydrogen atom
  * (c) À ’ À* transition in benzene (D6h symmetry)
  * (d) d-d transition in octahedral complex (Oh symmetry)

View solution

**Solution** :

(a) **Allowed** : $\Delta l = +1$ (s ’ p), satisfies selection rule for electric dipole transitions.

(b) **Forbidden** : $\Delta l = 0$ (s ’ s), electric dipole transitions require $\Delta l = \pm 1$.

(c) **Allowed** : Transition from À orbital (Àu) to À* orbital (Àg*) satisfies Laporte rule (g ” u).

(d) **Forbidden** (Laporte-forbidden): Both are d orbitals (g symmetry), so g ” g transition. However, weak absorption may be observed due to symmetry breaking by vibrations or ligand effects.

**Problem 9** : Given measured absorption spectrum data, perform the following analysis steps:

  1. Baseline correction (2nd order polynomial fitting)
  2. Automatic peak detection
  3. Gaussian fitting of each peak
  4. Calculate peak center, FWHM, and integrated intensity

View solution (complete analysis code)

Due to length constraints, this solution is abbreviated. Please refer to Code Example 8 for the complete implementation framework.

**Problem 10** : Implement the Voigt lineshape (convolution of Gaussian and Lorentzian) and plot and compare pure Gaussian, pure Lorentzian, and Voigt lineshapes. The Voigt lineshape is defined by the following integral:

$$V(x; \sigma, \gamma) = \int_{-\infty}^{\infty} G(x'; \sigma) L(x - x'; \gamma) \, dx'$$

View solution

Use scipy.special.wofz (Faddeeva function) for efficient Voigt profile calculation. See the peak fitting section for implementation details.

## Learning Objectives Check

Review what you learned in this chapter and verify the following items.

### Basic Understanding

  *  Can explain the relationship between photon energy and wavelength (Planck relation)
  *  Understand the differences and physical mechanisms of absorption, emission, and scattering
  *  Can explain the physical meaning and application conditions of Beer-Lambert law
  *  Understand the concept of selection rules and major selection rules ($\Delta l = \pm 1$, Laporte rule, etc.)

### Practical Skills

  *  Can interconvert wavelength, wavenumber, and energy
  *  Can perform concentration calculations and create calibration curves using Beer-Lambert law
  *  Can perform peak fitting (Gaussian/Lorentzian) on spectra
  *  Can appropriately perform baseline correction

### Applied Capabilities

  *  Can calculate transition probabilities from Fermi's golden rule
  *  Can interpret vibrational structure based on Franck-Condon principle
  *  Can separate and quantify multi-peak spectra

## References

  1. Atkins, P., de Paula, J. (2010). _Physical Chemistry_ (9th ed.). Oxford University Press, pp. 465-468 (Beer-Lambert law), pp. 485-490 (transition dipole moments), pp. 501-506 (selection rules). - Detailed explanation of quantum mechanical foundations of spectroscopy and transition moments
  2. Banwell, C. N., McCash, E. M. (1994). _Fundamentals of Molecular Spectroscopy_ (4th ed.). McGraw-Hill, pp. 8-15 (electromagnetic radiation), pp. 28-35 (Beer-Lambert law applications). - Basic principles of spectroscopic analysis and applications of Beer-Lambert law
  3. Hollas, J. M. (2004). _Modern Spectroscopy_ (4th ed.). Wiley, pp. 15-23 (selection rules), pp. 45-52 (Franck-Condon principle), pp. 78-85 (transition probabilities). - Comprehensive explanation of selection rules and Franck-Condon principle
  4. Beer, A. (1852). Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten. _Annalen der Physik und Chemie_ , 86, 78-88. DOI: 10.1002/andp.18521620505 - Original paper of Beer-Lambert law (historical reference)
  5. Shirley, D. A. (1972). High-resolution X-ray photoemission spectrum of the valence bands of gold. _Physical Review B_ , 5(12), 4709-4714. DOI: 10.1103/PhysRevB.5.4709 - Original paper of Shirley background correction algorithm
  6. NumPy 1.24 and SciPy 1.11 Documentation. _Signal Processing (scipy.signal) and Optimization (scipy.optimize)_. Available at: https://docs.scipy.org/doc/scipy/reference/signal.html - Practical methods for spectral data analysis with Python
  7. Turrell, G., Corset, J. (Eds.). (1996). _Raman Microscopy: Developments and Applications_. Academic Press, pp. 25-34 (classical scattering theory), pp. 58-67 (selection rules for Raman). - Details of scattering spectroscopy theory and selection rules
  8. Levine, I. N. (2013). _Quantum Chemistry_ (7th ed.). Pearson, pp. 580-595 (time-dependent perturbation theory), pp. 620-635 (transition dipole moments). - Transition dipole moments and quantum mechanical calculation methods

## Next Steps

In Chapter 1, we learned the foundations of spectroscopic analysis: light-matter interactions, Beer-Lambert law, quantum mechanical principles, and selection rules. We also acquired practical data processing skills with Python (unit conversion, peak fitting, baseline correction).

In **Chapter 2** , we will apply this foundational knowledge to learn the principles and practice of infrared and Raman spectroscopy. We will cover everything about vibrational spectroscopy frequently used in materials science, including functional group identification, crystallinity evaluation, complementary relationship between IR and Raman, and detailed selection rules through group theory.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
