---
title: "Chapter 2: Introduction to SEM"
chapter_title: "Chapter 2: Introduction to SEM"
subtitle: Principles of Scanning Electron Microscopy, Secondary Electron Images, Backscattered Electron Images, and EDS Analysis
reading_time: 25-35 minutes
difficulty: Beginner to Intermediate
code_examples: 7
version: 1.0
created_at: "by:"
---

A Scanning Electron Microscope (SEM) is an instrument that scans a sample surface with an electron beam and detects secondary electrons (SE) and backscattered electrons (BSE) to obtain high-resolution images. In this chapter, you will learn the operating principles of SEM, the formation mechanisms of SE and BSE images, elemental analysis using Energy Dispersive X-ray Spectroscopy (EDS), and the basics of image processing, while practicing signal simulation, quantitative analysis, and particle analysis using Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the scanning principle of SEM and signal detection mechanisms
  * ✅ Explain the generation mechanisms and characteristics of secondary electrons (SE) and backscattered electrons (BSE)
  * ✅ Understand how to use SE and BSE images appropriately, along with their respective advantages and disadvantages
  * ✅ Master the principles of EDS (Energy Dispersive X-ray Spectroscopy) and quantitative analysis methods
  * ✅ Evaluate the influence of acceleration voltage and working distance on spatial resolution and signal intensity
  * ✅ Implement electron yield, EDS quantification corrections, and particle size distribution analysis using Python
  * ✅ Understand the basics of optimizing SEM image acquisition conditions and image processing

## 2.1 Basic Principles of SEM

### 2.1.1 Configuration of Scanning Electron Microscopes

An SEM (Scanning Electron Microscope) scans the sample surface with a focused electron beam and forms images by detecting the generated signals synchronously.
    
    
    ```mermaid
    flowchart TD
        A[Electron GunElectron Gun] --> B[Condenser LensesCondenser Lenses]
        B --> C[Objective LensObjective Lens]
        C --> D[Scan CoilsScan Coils]
        D --> E[SampleSample]
    
        E --> F[Secondary ElectronsSE]
        E --> G[Backscattered ElectronsBSE]
        E --> H[Characteristic X-raysX-ray]
    
        F --> I[ET Detector]
        G --> J[BSE Detector]
        H --> K[EDS Detector]
    
        I --> L[Image DisplayDisplay]
        J --> L
        K --> M[Spectrum AnalysisAnalysis]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style L fill:#99ccff,stroke:#0066cc,stroke-width:2px
    ```

**Features of SEM** :

  * **Wide magnification range** : 10× to 1,000,000× (fills the gap between optical microscopes and TEM)
  * **Large depth of field** : Samples with uneven surfaces remain in focus over a wide range
  * **Easy sample preparation** : Bulk samples can be observed directly (only requires conductive coating)
  * **Multiple signal detection** : SE, BSE, and X-rays can be acquired simultaneously

### 2.1.2 Interaction Volume between Electron Beam and Sample

When an electron beam enters a sample, various signals are generated in a region called the **Interaction Volume**. This volume changes depending on the acceleration voltage and the atomic number of the sample.

**Depth of interaction volume (simplified formula)** :

$$ R_{\text{KO}} = \frac{0.0276 A E_0^{1.67}}{Z^{0.89} \rho} $$ 

Here, $R_{\text{KO}}$ is the Kanaya-Okayama range (μm), $A$ is the atomic weight, $E_0$ is the acceleration voltage (kV), $Z$ is the atomic number, and $\rho$ is the density (g/cm³).

#### Code Example 2-1: Calculation of Interaction Volume Depth
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def kanaya_okayama_range(Z, A, rho, E0_kV):
        """
        Calculate electron range using Kanaya-Okayama formula
    
        Parameters
        ----------
        Z : int or array-like
            Atomic number
        A : float or array-like
            Atomic weight [g/mol]
        rho : float or array-like
            Density [g/cm³]
        E0_kV : float or array-like
            Acceleration voltage [kV]
    
        Returns
        -------
        R_KO : float or array-like
            Electron range [μm]
        """
        R_KO = 0.0276 * A * (E0_kV ** 1.67) / ((Z ** 0.89) * rho)
        return R_KO
    
    # Representative materials
    materials = {
        'C': {'Z': 6, 'A': 12, 'rho': 2.26},
        'Al': {'Z': 13, 'A': 27, 'rho': 2.70},
        'Si': {'Z': 14, 'A': 28, 'rho': 2.33},
        'Fe': {'Z': 26, 'A': 56, 'rho': 7.87},
        'Cu': {'Z': 29, 'A': 64, 'rho': 8.96},
        'Au': {'Z': 79, 'A': 197, 'rho': 19.3}
    }
    
    # Acceleration voltage range
    E0_values = np.array([5, 10, 15, 20, 30])  # [kV]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Acceleration voltage dependence
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(materials)))
    for (name, props), color in zip(materials.items(), colors):
        ranges = kanaya_okayama_range(props['Z'], props['A'], props['rho'], E0_values)
        ax1.plot(E0_values, ranges, 'o-', linewidth=2.5, markersize=8,
                 label=f"{name} (Z={props['Z']})", color=color)
    
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Electron Range R$_{KO}$ [μm]', fontsize=12)
    ax1.set_title('Interaction Volume Depth vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 8)
    
    # Right plot: Atomic number dependence (fixed acceleration voltage)
    E0_fixed = 15  # [kV]
    Z_values = np.array([mat['Z'] for mat in materials.values()])
    A_values = np.array([mat['A'] for mat in materials.values()])
    rho_values = np.array([mat['rho'] for mat in materials.values()])
    mat_names = list(materials.keys())
    
    ranges_Z = kanaya_okayama_range(Z_values, A_values, rho_values, E0_fixed)
    
    ax2.scatter(Z_values, ranges_Z, s=200, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=2, zorder=3)
    ax2.plot(Z_values, ranges_Z, 'k--', linewidth=1.5, alpha=0.5, zorder=1)
    
    for Z, R, name in zip(Z_values, ranges_Z, mat_names):
        ax2.text(Z, R + 0.15, name, fontsize=10, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Number Z', fontsize=12)
    ax2.set_ylabel('Electron Range R$_{KO}$ [μm]', fontsize=12)
    ax2.set_title(f'Interaction Volume Depth vs Z\n(E$_0$ = {E0_fixed} kV)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 4)
    
    plt.tight_layout()
    plt.show()
    
    # Output specific values
    print(f"Electron range at 15 kV acceleration voltage:")
    for name, props in materials.items():
        R = kanaya_okayama_range(props['Z'], props['A'], props['rho'], 15)
        print(f"  {name:2s}: {R:.2f} μm")
    

**Key observations** :

  * Higher acceleration voltage results in longer electron range (signals generated from deeper regions)
  * Heavier elements (high Z) have shorter electron ranges (more surface-sensitive)
  * To improve spatial resolution, observe at low acceleration voltage

## 2.2 Secondary Electron Images (SE Images)

### 2.2.1 Secondary Electron Generation Mechanism

**Secondary Electrons (SE)** are low-energy electrons (<50 eV) emitted from the near-surface region (several nm) due to inelastic scattering with incident electrons.

**Characteristics of SE images** :

  * **Surface-sensitive** : Since the escape depth is only a few nm, surface morphology is observed with high contrast
  * **Edge effect** : Secondary electron yield increases at convex parts and edges of the sample, appearing brighter
  * **High resolution** : Probe size determines spatial resolution (5-10 nm)
  * **Charging** : In insulating samples, charge accumulation causes image distortion

### 2.2.2 Secondary Electron Yield

The secondary electron yield $\delta$ is the number of secondary electrons emitted per incident electron. $\delta$ depends on the acceleration voltage and sample tilt angle:

$$ \delta = \delta_{\max} \exp\left[-\left(\frac{E_0 - E_{\max}}{E_{\max}}\right)^2\right] \cdot \frac{1}{\cos\theta} $$ 

Here, $\delta_{\max}$ is the maximum secondary electron yield, $E_{\max}$ is the acceleration voltage that gives $\delta_{\max}$, and $\theta$ is the sample tilt angle.

#### Code Example 2-2: Simulation of Secondary Electron Yield
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def se_yield(E0_kV, theta_deg, delta_max=1.5, E_max_kV=0.5):
        """
        Calculate secondary electron yield
    
        Parameters
        ----------
        E0_kV : float or array-like
            Acceleration voltage [kV]
        theta_deg : float or array-like
            Sample tilt angle [degrees] (0°: normal incidence)
        delta_max : float
            Maximum secondary electron yield
        E_max_kV : float
            Acceleration voltage giving maximum yield [kV]
    
        Returns
        -------
        delta : float or array-like
            Secondary electron yield
        """
        theta_rad = np.deg2rad(theta_deg)
        delta = delta_max * np.exp(-((E0_kV - E_max_kV) / E_max_kV)**2) / np.cos(theta_rad)
        return delta
    
    # Acceleration voltage dependence
    E0_range = np.linspace(0.1, 30, 200)  # [kV]
    theta_values = [0, 30, 60, 75]  # [degrees]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Acceleration voltage dependence (different tilt angles)
    colors = ['#f093fb', '#f5576c', '#ffa500', '#ff6347']
    for theta, color in zip(theta_values, colors):
        delta = se_yield(E0_range, theta, delta_max=1.5, E_max_kV=0.5)
        ax1.plot(E0_range, delta, linewidth=2.5, label=f'θ = {theta}°', color=color)
    
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='δ = 1 (Charge Balance)')
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Secondary Electron Yield δ', fontsize=12)
    ax1.set_title('SE Yield vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 4)
    
    # Right plot: Tilt angle dependence (fixed acceleration voltage)
    E0_fixed = 5  # [kV]
    theta_range = np.linspace(0, 80, 100)  # [degrees]
    
    delta_theta = se_yield(E0_fixed, theta_range, delta_max=1.5, E_max_kV=0.5)
    
    ax2.plot(theta_range, delta_theta, linewidth=3, color='#f093fb')
    ax2.fill_between(theta_range, 0, delta_theta, alpha=0.3, color='#f093fb')
    
    ax2.set_xlabel('Sample Tilt Angle θ [degrees]', fontsize=12)
    ax2.set_ylabel('Secondary Electron Yield δ', fontsize=12)
    ax2.set_title(f'SE Yield vs Tilt Angle\n(E$_0$ = {E0_fixed} kV)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 10)
    
    # Add explanation of edge effect
    ax2.text(60, 7, 'Edge Effect:\nHigher yield at steep angles',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("Secondary electron yield characteristics:")
    print(f"  - Normal incidence (θ=0°), 5 kV: δ = {se_yield(5, 0):.2f}")
    print(f"  - Tilt 60°, 5 kV: δ = {se_yield(5, 60):.2f}")
    print("  - Convex parts and edges show increased yield → appear brighter (edge effect)")
    

## 2.3 Backscattered Electron Images (BSE Images)

### 2.3.1 Characteristics of Backscattered Electrons

**Backscattered Electrons (BSE)** are high-energy electrons (>50 eV) that are reflected back from the sample surface through elastic scattering with the sample.

**Characteristics of BSE images** :

  * **Compositional contrast** : Higher atomic number $Z$ results in higher BSE yield, appearing brighter
  * **Deeper information** : BSE are generated from the entire interaction volume (hundreds of nm to several μm)
  * **Topographic information** : BSE yield changes from tilted surfaces
  * **Spatial resolution** : Inferior to SE images (larger interaction volume)

### 2.3.2 Backscattered Electron Yield

BSE yield $\eta$ is approximately proportional to atomic number $Z$ (empirical formula):

$$ \eta \approx -0.0254 + 0.016 Z - 1.86 \times 10^{-4} Z^2 + 8.3 \times 10^{-7} Z^3 $$ 

#### Code Example 2-3: Atomic Number Dependence of Backscattered Electron Yield
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def bse_yield(Z):
        """
        Calculate backscattered electron yield (empirical formula)
    
        Parameters
        ----------
        Z : int or array-like
            Atomic number
    
        Returns
        -------
        eta : float or array-like
            BSE yield
        """
        Z = np.asarray(Z)
        eta = -0.0254 + 0.016*Z - 1.86e-4*Z**2 + 8.3e-7*Z**3
        return eta
    
    # BSE yield for each element
    elements = {
        'C': 6, 'Al': 13, 'Si': 14, 'Ti': 22, 'Fe': 26,
        'Cu': 29, 'Zn': 30, 'Mo': 42, 'Ag': 47, 'W': 74, 'Pt': 78, 'Au': 79
    }
    
    Z_values = np.array(list(elements.values()))
    element_names = list(elements.keys())
    eta_values = bse_yield(Z_values)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Atomic number vs BSE yield
    Z_range = np.linspace(1, 92, 200)
    eta_range = bse_yield(Z_range)
    
    ax1.plot(Z_range, eta_range, linewidth=3, color='#f093fb', label='Empirical Formula')
    ax1.scatter(Z_values, eta_values, s=150, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=1.5, zorder=3)
    
    # Label representative elements
    for Z, eta, name in zip(Z_values[::2], eta_values[::2], element_names[::2]):
        ax1.text(Z, eta + 0.02, name, fontsize=9, ha='center', fontweight='bold')
    
    ax1.set_xlabel('Atomic Number Z', fontsize=12)
    ax1.set_ylabel('Backscattered Electron Yield η', fontsize=12)
    ax1.set_title('BSE Yield vs Atomic Number', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 0.6)
    
    # Right plot: Simulation of compositional contrast
    # Simulate BSE image of Fe-Cu alloy
    size = 256
    image = np.zeros((size, size))
    
    # Fe region (Z=26)
    Z_Fe = 26
    eta_Fe = bse_yield(Z_Fe)
    image[:, :size//2] = eta_Fe
    
    # Cu region (Z=29)
    Z_Cu = 29
    eta_Cu = bse_yield(Z_Cu)
    image[:, size//2:] = eta_Cu
    
    # Add noise
    image += np.random.normal(0, 0.01, image.shape)
    
    im = ax2.imshow(image, cmap='gray', vmin=0.1, vmax=0.4)
    ax2.axvline(x=size//2, color='red', linestyle='--', linewidth=2)
    ax2.text(size//4, size*0.1, f'Fe (Z={Z_Fe})\nη={eta_Fe:.3f}',
             fontsize=11, ha='center', color='white', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(size*3//4, size*0.1, f'Cu (Z={Z_Cu})\nη={eta_Cu:.3f}',
             fontsize=11, ha='center', color='black', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.set_title('BSE Image Simulation: Fe-Cu Interface\n(Compositional Contrast)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('BSE Yield η', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("Examples of backscattered electron yield:")
    print(f"  C  (Z=6):  η = {bse_yield(6):.3f}")
    print(f"  Fe (Z=26): η = {bse_yield(26):.3f}")
    print(f"  Cu (Z=29): η = {bse_yield(29):.3f}")
    print(f"  Au (Z=79): η = {bse_yield(79):.3f}")
    print(f"\nAu brightness is approximately {bse_yield(79)/bse_yield(6):.1f}× that of C")
    

## 2.4 Energy Dispersive X-ray Spectroscopy (EDS)

### 2.4.1 Generation of Characteristic X-rays

When incident electrons excite inner-shell electrons, **characteristic X-rays** are emitted as outer-shell electrons transition. The energy of these X-rays is element-specific, enabling elemental identification and quantitative analysis.

**Main X-ray series** :

  * **K series** : Transitions to K shell (n=1). Kα (L→K), Kβ (M→K)
  * **L series** : Transitions to L shell (n=2). Lα, Lβ, Lγ
  * **M series** : Transitions to M shell (n=3). Important for heavy elements

**Moseley's law** : Characteristic X-ray energy is proportional to the square of atomic number $Z$:

$$ E_{\text{K}\alpha} \approx 10.2 (Z - 1)^2 \text{ eV} $$ 

### 2.4.2 EDS Quantitative Analysis and ZAF Correction

In EDS quantitative analysis, composition is determined from measured X-ray intensity ratios, but **ZAF correction** is necessary:

  * **Z correction (atomic number effect)** : Correction for backscattering and stopping power
  * **A correction (absorption correction)** : Correction for X-ray absorption within the sample
  * **F correction (fluorescence correction)** : Correction for secondary excitation by X-rays from other elements

The corrected mass concentration $C_i$ is:

$$ C_i = \frac{k_i \cdot \text{ZAF}_i}{\sum_j k_j \cdot \text{ZAF}_j} $$ 

Here, $k_i$ is the intensity ratio relative to the standard sample.

#### Code Example 2-4: Simulation of EDS Quantitative Analysis (ZAF Correction)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def zaf_correction(Z, A, F=1.0):
        """
        Simplified ZAF correction factor
    
        Parameters
        ----------
        Z : float
            Atomic number correction factor
        A : float
            Absorption correction factor
        F : float
            Fluorescence correction factor (typically close to 1)
    
        Returns
        -------
        ZAF : float
            Total correction factor
        """
        return Z * A * F
    
    def quantitative_eds_analysis(k_ratios, elements, Z_factors, A_factors):
        """
        EDS quantitative analysis (ZAF correction)
    
        Parameters
        ----------
        k_ratios : array-like
            Intensity ratio for each element (ratio to standard sample)
        elements : list
            List of element names
        Z_factors : array-like
            Atomic number correction factors
        A_factors : array-like
            Absorption correction factors
    
        Returns
        -------
        concentrations : dict
            Mass concentration of each element [wt%]
        """
        k_ratios = np.array(k_ratios)
        Z_factors = np.array(Z_factors)
        A_factors = np.array(A_factors)
    
        # ZAF correction
        zaf_factors = zaf_correction(Z_factors, A_factors)
        corrected_intensities = k_ratios * zaf_factors
    
        # Normalization
        total = np.sum(corrected_intensities)
        concentrations = {elem: (corr / total) * 100 for elem, corr in zip(elements, corrected_intensities)}
    
        return concentrations
    
    # Quantitative analysis example of Fe-Cr-Ni alloy
    elements = ['Fe', 'Cr', 'Ni']
    k_ratios = [0.70, 0.18, 0.12]  # Measured intensity ratios
    Z_factors = [1.00, 0.98, 1.03]  # Atomic number correction
    A_factors = [1.00, 0.95, 1.02]  # Absorption correction
    
    concentrations = quantitative_eds_analysis(k_ratios, elements, Z_factors, A_factors)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Before and after correction comparison
    uncorrected = {elem: (k / sum(k_ratios)) * 100 for elem, k in zip(elements, k_ratios)}
    
    x = np.arange(len(elements))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, list(uncorrected.values()), width,
                    label='Before ZAF Correction', color='#f093fb', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, list(concentrations.values()), width,
                    label='After ZAF Correction', color='#f5576c', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Concentration [wt%]', fontsize=12)
    ax1.set_title('EDS Quantitative Analysis: Fe-Cr-Ni Alloy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Numerical labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
                 ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
                 ha='center', fontsize=10, fontweight='bold')
    
    # Right plot: EDS spectrum simulation
    energy = np.linspace(0, 10, 1000)  # [keV]
    
    # Characteristic X-ray peaks for each element
    Fe_Ka = 6.40  # [keV]
    Cr_Ka = 5.41
    Ni_Ka = 7.47
    
    # Gaussian peaks
    def gaussian_peak(E, E0, sigma, amplitude):
        return amplitude * np.exp(-((E - E0) / sigma)**2)
    
    spectrum = np.zeros_like(energy)
    spectrum += gaussian_peak(energy, Fe_Ka, 0.15, concentrations['Fe'] * 10)
    spectrum += gaussian_peak(energy, Cr_Ka, 0.15, concentrations['Cr'] * 10)
    spectrum += gaussian_peak(energy, Ni_Ka, 0.15, concentrations['Ni'] * 10)
    
    # Background (bremsstrahlung)
    background = 500 * np.exp(-energy / 2)
    spectrum += background
    
    ax2.plot(energy, spectrum, linewidth=2, color='#2c3e50')
    ax2.fill_between(energy, 0, spectrum, alpha=0.3, color='#f093fb')
    
    # Indicate peak positions with arrows
    ax2.annotate('Fe Kα', xy=(Fe_Ka, gaussian_peak(Fe_Ka, Fe_Ka, 0.15, concentrations['Fe']*10)),
                 xytext=(Fe_Ka-1, 800), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('Cr Kα', xy=(Cr_Ka, gaussian_peak(Cr_Ka, Cr_Ka, 0.15, concentrations['Cr']*10)),
                 xytext=(Cr_Ka-1.5, 600), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('Ni Kα', xy=(Ni_Ka, gaussian_peak(Ni_Ka, Ni_Ka, 0.15, concentrations['Ni']*10)),
                 xytext=(Ni_Ka+0.5, 600), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax2.set_xlabel('Energy [keV]', fontsize=12)
    ax2.set_ylabel('Intensity [counts]', fontsize=12)
    ax2.set_title('Simulated EDS Spectrum', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1000)
    
    plt.tight_layout()
    plt.show()
    
    print("EDS quantitative analysis results (after ZAF correction):")
    for elem, conc in concentrations.items():
        print(f"  {elem}: {conc:.2f} wt%")
    

## 2.5 SEM Image Analysis

### 2.5.1 Particle Size Distribution Analysis

Quantitative analysis of particle size distribution from SEM images is important for materials evaluation. Image processing procedure:

  1. **Preprocessing** : Noise removal (Gaussian filter), contrast adjustment
  2. **Binarization** : Extract particle regions by thresholding
  3. **Labeling** : Assign unique IDs to each particle
  4. **Feature extraction** : Calculate area, perimeter, circularity, etc.
  5. **Statistical analysis** : Calculate size distribution, mean particle size, standard deviation

#### Code Example 2-5: Particle Size Distribution Analysis from SEM Images
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter, label
    from scipy import ndimage
    
    def generate_particle_image(size=512, num_particles=50):
        """
        Simulate SEM image with dispersed particles
    
        Parameters
        ----------
        size : int
            Image size [pixels]
        num_particles : int
            Number of particles
    
        Returns
        -------
        image : ndarray
            Simulated image
        """
        image = np.ones((size, size)) * 0.2  # Background
    
        np.random.seed(42)
        for _ in range(num_particles):
            # Random position and size
            x = np.random.randint(20, size-20)
            y = np.random.randint(20, size-20)
            radius = np.random.randint(5, 25)
    
            # Draw circular particle
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= radius**2
            image[mask] = 0.8 + np.random.normal(0, 0.05)
    
        # Add noise
        image += np.random.normal(0, 0.03, image.shape)
        image = np.clip(image, 0, 1)
    
        # Blur
        image = gaussian_filter(image, sigma=1.0)
    
        return image
    
    def analyze_particles(image, threshold=0.5, pixel_size_nm=10):
        """
        Analyze particle size distribution
    
        Parameters
        ----------
        image : ndarray
            Input image
        threshold : float
            Binarization threshold
        pixel_size_nm : float
            Pixel size [nm/pixel]
    
        Returns
        -------
        areas : list
            Area of each particle [nm²]
        diameters : list
            Equivalent diameter of each particle [nm]
        binary : ndarray
            Binary image
        labeled : ndarray
            Labeled image
        """
        # Binarization
        binary = image > threshold
    
        # Labeling
        labeled, num_features = label(binary)
    
        # Calculate area of each particle
        areas = []
        diameters = []
    
        for i in range(1, num_features + 1):
            area_pixels = np.sum(labeled == i)
            area_nm2 = area_pixels * (pixel_size_nm ** 2)
    
            # Equivalent diameter (diameter with same area as circle)
            diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)
    
            # Exclude particles that are too small
            if area_pixels > 10:
                areas.append(area_nm2)
                diameters.append(diameter_nm)
    
        return areas, diameters, binary, labeled
    
    # Run simulation
    image = generate_particle_image(size=512, num_particles=60)
    areas, diameters, binary, labeled = analyze_particles(image, threshold=0.5, pixel_size_nm=5)
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original SEM Image', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # Binary image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Binary Image\n(Threshold = 0.5)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    # Labeled image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(labeled, cmap='tab20')
    ax3.set_title(f'Labeled Image\n({len(diameters)} particles detected)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    
    # Histogram: particle diameter distribution
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(diameters, bins=20, color='#f093fb', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Particle Diameter [nm]', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Particle Size Distribution', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add statistical information
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    ax4.axvline(mean_diameter, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diameter:.1f} nm')
    ax4.axvline(mean_diameter - std_diameter, color='orange', linestyle=':', linewidth=1.5, label=f'Std: ±{std_diameter:.1f} nm')
    ax4.axvline(mean_diameter + std_diameter, color='orange', linestyle=':', linewidth=1.5)
    ax4.legend(fontsize=11)
    
    # Statistical table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_text = f"""
    Particle Analysis Results
    
    Number of particles: {len(diameters)}
    
    Diameter [nm]:
      Mean:   {mean_diameter:.2f}
      Median: {np.median(diameters):.2f}
      Std:    {std_diameter:.2f}
      Min:    {np.min(diameters):.2f}
      Max:    {np.max(diameters):.2f}
    
    Area [nm²]:
      Mean:   {np.mean(areas):.1f}
      Total:  {np.sum(areas):.1f}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Particle analysis results:")
    print(f"  Number of particles detected: {len(diameters)}")
    print(f"  Mean diameter: {mean_diameter:.2f} ± {std_diameter:.2f} nm")
    print(f"  Diameter range: {np.min(diameters):.2f} - {np.max(diameters):.2f} nm")
    

## 2.6 Exercise Problems

### Exercise 2-1: Calculation of Electron Range (Easy)

**Problem** : Calculate the electron range when observing a Si sample (Z=14, A=28, ρ=2.33 g/cm³) at 15 kV.

**Show Answer**
    
    
    Z = 14
    A = 28
    rho = 2.33
    E0 = 15
    
    R_KO = 0.0276 * A * (E0 ** 1.67) / ((Z ** 0.89) * rho)
    print(f"Electron range for Si (15 kV): {R_KO:.2f} μm")
    

### Exercise 2-2: Evaluation of Secondary Electron Yield (Medium)

**Problem** : Calculate the secondary electron yield when the sample is tilted 60° at an acceleration voltage of 5 kV (δ_max=1.5, E_max=0.5 kV). Compare with normal incidence.

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Calculate the secondary electron yield when the sam
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    E0 = 5
    delta_max = 1.5
    E_max = 0.5
    
    # Normal incidence (θ=0°)
    theta_0 = 0
    delta_0 = delta_max * np.exp(-((E0 - E_max) / E_max)**2) / np.cos(np.deg2rad(theta_0))
    
    # Tilt 60°
    theta_60 = 60
    delta_60 = delta_max * np.exp(-((E0 - E_max) / E_max)**2) / np.cos(np.deg2rad(theta_60))
    
    print(f"Normal incidence (θ=0°): δ = {delta_0:.3f}")
    print(f"Tilt 60°: δ = {delta_60:.3f}")
    print(f"Yield increase: {delta_60 / delta_0:.2f}×")
    

### Exercise 2-3: Quantification of BSE Compositional Contrast (Medium)

**Problem** : When observing the boundary between Ti (Z=22) and Ni (Z=28) in a BSE image, calculate the BSE yield of both and estimate the image contrast.

**Show Answer**
    
    
    Z_Ti = 22
    Z_Ni = 28
    
    eta_Ti = -0.0254 + 0.016*Z_Ti - 1.86e-4*Z_Ti**2 + 8.3e-7*Z_Ti**3
    eta_Ni = -0.0254 + 0.016*Z_Ni - 1.86e-4*Z_Ni**2 + 8.3e-7*Z_Ni**3
    
    contrast = (eta_Ni - eta_Ti) / eta_Ti * 100
    
    print(f"Ti (Z={Z_Ti}): η = {eta_Ti:.3f}")
    print(f"Ni (Z={Z_Ni}): η = {eta_Ni:.3f}")
    print(f"Contrast: {contrast:.1f}%")
    print(f"Ni region appears {contrast:.1f}% brighter than Ti region")
    

### Exercise 2-4: EDS Quantitative Analysis (Hard)

**Problem** : In EDS analysis of an Al-Si alloy, k(Al)=0.65 and k(Si)=0.35 were obtained. If the ZAF correction factors are Al: 1.02 and Si: 0.98, determine the composition.

**Show Answer**
    
    
    k_Al = 0.65
    k_Si = 0.35
    ZAF_Al = 1.02
    ZAF_Si = 0.98
    
    corrected_Al = k_Al * ZAF_Al
    corrected_Si = k_Si * ZAF_Si
    
    total = corrected_Al + corrected_Si
    
    C_Al = (corrected_Al / total) * 100
    C_Si = (corrected_Si / total) * 100
    
    print(f"Composition after ZAF correction:")
    print(f"  Al: {C_Al:.2f} wt%")
    print(f"  Si: {C_Si:.2f} wt%")
    

### Exercise 2-5: Particle Size Statistics (Hard)

**Problem** : Ten particles have diameters of [50, 55, 60, 52, 58, 62, 48, 54, 56, 61] nm. Calculate the mean particle size, standard deviation, and relative standard deviation (RSD).

**Show Answer**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Problem: Ten particles have diameters of [50, 55, 60, 52, 58
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    diameters = np.array([50, 55, 60, 52, 58, 62, 48, 54, 56, 61])
    
    mean_d = np.mean(diameters)
    std_d = np.std(diameters, ddof=1)  # Unbiased standard deviation
    rsd = (std_d / mean_d) * 100
    
    print(f"Mean particle size: {mean_d:.2f} nm")
    print(f"Standard deviation: {std_d:.2f} nm")
    print(f"Relative standard deviation (RSD): {rsd:.2f} %")
    

### Exercise 2-6: Optimization of Acceleration Voltage (Hard)

**Problem** : To maximize the compositional contrast between Ti and Fe, which is more appropriate: 5 kV or 20 kV? Calculate the difference in BSE yield and discuss.

**Show Answer**

**Key points for answer** :

  * BSE yield $\eta$ depends mainly on atomic number and is almost independent of acceleration voltage
  * Therefore, compositional contrast itself is similar at both 5 kV and 20 kV
  * However, lower acceleration voltage (5 kV) is more surface-sensitive and provides higher spatial resolution
  * **Conclusion** : 5 kV is advantageous for observing surface compositional distribution with high resolution

### Exercise 2-7: Evaluation of X-ray Spatial Resolution (Hard)

**Problem** : When observing an Al sample at 15 kV, estimate the depth of the generation region for Al Kα X-rays (1.49 keV). Assume approximately 70% of the electron range.

**Show Answer**
    
    
    Z_Al = 13
    A_Al = 27
    rho_Al = 2.70
    E0 = 15
    
    R_KO = 0.0276 * A_Al * (E0 ** 1.67) / ((Z_Al ** 0.89) * rho_Al)
    X_ray_depth = R_KO * 0.7
    
    print(f"Electron range for Al (15 kV): {R_KO:.2f} μm")
    print(f"Generation depth of Al Kα X-rays (estimated): {X_ray_depth:.2f} μm")
    print(f"EDS spatial resolution is approximately {X_ray_depth*1000:.0f} nm")
    

### Exercise 2-8: Experimental Planning (Hard)

**Problem** : Plan an experiment to observe and quantify Mg2Si precipitates (size tens of nm) in an Al alloy. Explain how to use SE images, BSE images, and EDS appropriately.

**Show Answer**

**Experimental plan** :

  1. **SE image (low magnification)** : Overall observation of sample surface, confirm precipitate distribution (5-10 kV)
  2. **BSE image (high magnification)** : Observe compositional contrast between Mg2Si (light elements) and Al matrix (10-15 kV). Mg2Si appears dark
  3. **SE image (high magnification)** : Measure precipitate morphology and particle size (5 kV, surface-sensitive)
  4. **EDS point analysis** : Acquire spectra on precipitates and matrix separately, quantify Mg/Si/Al ratio (15 kV)
  5. **EDS mapping** : Visualize elemental distribution of Mg, Si, Al (15 kV, long integration time)
  6. **Image analysis** : Quantify particle size distribution and number density from SE or BSE images

**Rationale** :

  * SE images are optimal for surface morphology observation, BSE images identify precipitates with compositional contrast
  * 15 kV is sufficient to excite Mg Kα (1.25 keV) and Si Kα (1.74 keV)
  * Low acceleration voltage (5 kV) provides high resolution but reduces X-ray excitation efficiency, making it unsuitable for quantification

## 2.7 Learning Check

Answer the following questions to check your understanding:

  1. Can you explain the scanning principle and image formation mechanism of SEM?
  2. Do you understand the differences in generation mechanisms between secondary electrons and backscattered electrons?
  3. Can you explain the characteristics and appropriate use of SE and BSE images?
  4. Do you understand the atomic number dependence and acceleration voltage dependence of electron range?
  5. Can you explain the generation mechanism of characteristic X-rays and Moseley's law?
  6. Do you understand the necessity of ZAF correction in EDS quantitative analysis?
  7. Can you explain the procedure for particle size distribution analysis from SEM images?
  8. Do you understand the influence of acceleration voltage and working distance selection on image quality?

## 2.8 References

  1. Goldstein, J. I., et al. (2017). _Scanning Electron Microscopy and X-Ray Microanalysis_ (4th ed.). Springer. - Comprehensive textbook on SEM and EDS analysis
  2. Reimer, L. (1998). _Scanning Electron Microscopy: Physics of Image Formation and Microanalysis_ (2nd ed.). Springer. - Detailed SEM imaging theory
  3. Newbury, D. E., & Ritchie, N. W. M. (2013). "Is Scanning Electron Microscopy/Energy Dispersive X-ray Spectrometry (SEM/EDS) Quantitative?" _Scanning_ , 35, 141-168. - Accuracy evaluation of EDS quantitative analysis
  4. Joy, D. C. (1995). _Monte Carlo Modeling for Electron Microscopy and Microanalysis_. Oxford University Press. - Electron beam simulation
  5. Echlin, P. (2009). _Handbook of Sample Preparation for Scanning Electron Microscopy and X-Ray Microanalysis_. Springer. - Sample preparation techniques
  6. JEOL Application Notes. "SEM Basics and Applications" - Manufacturer technical documentation
  7. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - Electron-specimen interactions (fundamental theory)

## 2.9 Next Chapter

In the next chapter, you will learn the principles of Transmission Electron Microscopy (TEM), imaging theory, bright-field and dark-field images, Selected Area Electron Diffraction (SAED), High-Resolution TEM (HRTEM), and aberration correction techniques. TEM is a powerful method for observing atomic-level structures using electron beams transmitted through samples.
