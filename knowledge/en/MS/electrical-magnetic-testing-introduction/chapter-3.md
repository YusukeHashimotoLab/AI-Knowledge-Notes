---
title: "Chapter 3: Magnetic Measurements"
chapter_title: "Chapter 3: Magnetic Measurements"
subtitle: VSM/SQUID Magnetometry, M-H Curve Analysis, Magnetic Anisotropy Evaluation, PPMS Integrated Measurements
reading_time: 50-60 minutes
difficulty: Intermediate to Advanced
code_examples: 7
---

Magnetic measurements are techniques for quantitatively evaluating the magnetic properties of materials (magnetization, magnetic moment, magnetic anisotropy). In this chapter, we will learn the principles of VSM (Vibrating Sample Magnetometer) and SQUID (Superconducting Quantum Interference Device), M-H curve analysis (saturation magnetization, coercivity, remanence), fitting with Curie-Weiss law and Langevin function, temperature-dependent measurements (FC/ZFC), and integrated measurement techniques using PPMS (Physical Property Measurement System), and perform practical magnetic data analysis with Python. 

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the basic principles of diamagnetism, paramagnetism, and ferromagnetism
  * ✅ Explain the measurement principles and characteristics of VSM and SQUID
  * ✅ Extract saturation magnetization, coercivity, and remanence from M-H curves
  * ✅ Fit data with the Curie-Weiss law (χ = C/(T - θ))
  * ✅ Analyze paramagnetism with the Langevin function
  * ✅ Calculate magnetic anisotropy energy
  * ✅ Build complete VSM/SQUID data processing workflows in Python

## 3.1 Fundamentals of Magnetism

### 3.1.1 Relationship Between Magnetization and Magnetic Field

**Magnetization** $M$ is the magnetic moment per unit volume:

$$ M = \frac{\sum \mu_i}{V} $$ 

where $\mu_i$ is the magnetic moment of individual atoms/molecules, and $V$ is the volume.

**Magnetic susceptibility** $\chi$ is the response of magnetization to magnetic field $H$:

$$ M = \chi H $$ 

**Types of magnetic fields** :

  * **Magnetic field $H$** (magnetic field strength): unit A/m or Oe (Oersted)
  * **Magnetic flux density $B$** : unit T (Tesla)
  * Relationship: $B = \mu_0(H + M)$ (SI units), $\mu_0 = 4\pi \times 10^{-7}$ H/m

### 3.1.2 Classification of Magnetism

Magnetism | Susceptibility $\chi$ | Temperature Dependence | Characteristics | Material Examples  
---|---|---|---|---  
**Diamagnetism** | $\chi < 0$  
(~$-10^{-5}$) | Almost none | Weak magnetization  
opposing external field | Cu, Au, H₂O  
Superconductors  
**Paramagnetism** | $\chi > 0$  
(~$10^{-5}$-$10^{-3}$) | $\chi \propto 1/T$  
(Curie's law) | Weak magnetization  
in field direction | Al, Pt, O₂  
Rare earth ions  
**Ferromagnetism** | $\chi \gg 1$  
(~$10^2$-$10^5$) | Large for T < T$_C$  
Paramagnetic for T > T$_C$ | Spontaneous  
magnetization  
Hysteresis | Fe, Co, Ni  
NdFeB, SmCo  
**Antiferromagnetism** | $\chi > 0$  
(small) | Peak at T = T$_N$ | Adjacent spins  
antiparallel | MnO, Cr, FeO  
**Ferrimagnetism** | $\chi \gg 1$ | Similar to  
ferromagnetism | Unequal antiparallel  
spin arrangement | Fe₃O₄ (Magnetite)  
Ferrites  
      
    
    ```mermaid
    flowchart TD
        A[External Field H] --> B{Material Magnetic Response}
        B --> C[DiamagneticM ↓ H]
        B --> D[ParamagneticM ↑ HM ∝ H/T]
        B --> E[FerromagneticSpontaneousmagnetizationM >> H]
        
        E --> F[Saturation M_s]
        E --> G[Coercivity H_c]
        E --> H[Remanence M_r]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## 3.2 VSM (Vibrating Sample Magnetometer)

### 3.2.1 VSM Measurement Principle

**VSM** (developed by Simon Foner in 1956) is a method that vibrates the sample and measures the voltage induced in detection coils by its magnetic moment.

**Measurement principle** :

  1. Place the sample in a uniform magnetic field and vibrate it vertically at frequency $f$ (typically 10-100 Hz)
  2. When the sample's magnetic moment $\mu$ vibrates, a time-varying magnetic flux $\Phi(t)$ is generated around the detection coils
  3. According to Faraday's law of electromagnetic induction, induced voltage $V(t)$ is generated in the coils: $$ V(t) = -\frac{d\Phi}{dt} \propto \mu \cdot f $$ 
  4. Lock-in detection extracts the component at vibration frequency $f$ to quantify the magnetic moment

**VSM characteristics** :

  * **Sensitivity** : $10^{-6}$ - $10^{-8}$ emu (electromagnetic unit)
  * **Magnetic field range** : 0 - 3 T (typical electromagnet)
  * **Temperature range** : 4 K - 1000 K (with liquid He cryostat)
  * **Measurement time** : 0.1 - 1 second per point (relatively fast)

#### Code Example 3-1: Curie-Weiss Law Fitting
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def curie_weiss(T, C, theta):
        """
        Curie-Weiss law: χ = C / (T - θ)
        
        Parameters
        ----------
        T : array-like
            Temperature [K]
        C : float
            Curie constant
        theta : float
            Curie-Weiss temperature (Weiss constant) [K]
        
        Returns
        -------
        chi : array-like
            Magnetic susceptibility
        """
        return C / (T - theta)
    
    # Generate simulation data (paramagnetic material)
    T_range = np.linspace(100, 400, 30)  # [K]
    C_true = 1.5  # Curie constant
    theta_true = -10  # Curie-Weiss temperature [K] (negative → antiferromagnetic interaction)
    
    chi_data = curie_weiss(T_range, C_true, theta_true)
    chi_data_noise = chi_data * (1 + 0.05 * np.random.randn(len(T_range)))  # 5% noise
    
    # Fitting
    model = Model(curie_weiss)
    params = model.make_params(C=1.0, theta=0)
    result = model.fit(chi_data_noise, params, T=T_range)
    
    print("Curie-Weiss Law Fitting Results:")
    print(result.fit_report())
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: χ vs T
    ax1.scatter(T_range, chi_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Measured data', color='#f093fb')
    ax1.plot(T_range, result.best_fit, linewidth=2.5, label=f'Fit: C={result.params["C"].value:.2f}, θ={result.params["theta"].value:.1f} K', color='#f5576c')
    ax1.set_xlabel('Temperature T [K]', fontsize=12)
    ax1.set_ylabel('Magnetic Susceptibility χ', fontsize=12)
    ax1.set_title('Curie-Weiss Law: χ vs T', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Right: 1/χ vs T (Curie-Weiss plot)
    ax2.scatter(T_range, 1/chi_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (1/χ)', color='#ffa500')
    T_fit_extended = np.linspace(0, 450, 100)
    chi_fit_extended = curie_weiss(T_fit_extended, result.params['C'].value, result.params['theta'].value)
    ax2.plot(T_fit_extended, 1/chi_fit_extended, linewidth=2.5, label='Linear fit (extended)', color='#f5576c')
    
    # Highlight θ
    theta_fit = result.params['theta'].value
    ax2.axvline(theta_fit, color='red', linestyle='--', linewidth=2, label=f'θ = {theta_fit:.1f} K')
    ax2.scatter([theta_fit], [0], s=200, c='red', edgecolors='black', linewidth=2, marker='X', zorder=5)
    
    ax2.set_xlabel('Temperature T [K]', fontsize=12)
    ax2.set_ylabel('1/χ', fontsize=12)
    ax2.set_title('Curie-Weiss Plot: 1/χ vs T', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 450)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPhysical Interpretation:")
    print(f"  Curie constant C = {result.params['C'].value:.2f}")
    print(f"  Curie-Weiss temperature θ = {result.params['theta'].value:.1f} K")
    if result.params['theta'].value < 0:
        print(f"  θ < 0 → Dominant antiferromagnetic interactions")
    elif result.params['theta'].value > 0:
        print(f"  θ > 0 → Dominant ferromagnetic interactions")
    else:
        print(f"  θ ≈ 0 → Ideal paramagnetism (no interactions)")
    

## 3.3 SQUID (Superconducting Quantum Interference Device)

### 3.3.1 SQUID Measurement Principle

**SQUID** (Superconducting Quantum Interference Device) is an ultra-high sensitivity magnetic sensor using superconducting rings and Josephson junctions.

**Measurement principle** :

  1. When magnetic flux $\Phi$ penetrates a superconducting ring, quantized superconducting current flows
  2. The critical current through the Josephson junction changes periodically with magnetic flux: $$ I_c(\Phi) = I_0 \left|\cos\left(\frac{\pi\Phi}{\Phi_0}\right)\right| $$ where $\Phi_0 = h/(2e) \approx 2.07 \times 10^{-15}$ Wb is the magnetic flux quantum 
  3. Minute changes in magnetic flux are detected with extremely high sensitivity as changes in critical current

**SQUID characteristics** :

  * **Sensitivity** : $10^{-10}$ - $10^{-12}$ emu (over 1000 times better than VSM)
  * **Magnetic field range** : 0 - 7 T (superconducting magnet)
  * **Temperature range** : 1.8 K - 400 K (with He gas)
  * **Measurement time** : 1 - 10 seconds per point (slower than VSM, but higher precision)
  * **Noise level** : $10^{-14}$ T/√Hz (world's highest sensitivity magnetic sensor)

#### Code Example 3-2: Langevin Function Fitting (Paramagnetism)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def langevin(H, M_s, T, mu):
        """
        Langevin function (classical paramagnetism)
        
        Parameters
        ----------
        H : array-like
            Magnetic field [A/m or Oe]
        M_s : float
            Saturation magnetization [emu/g or Am^2/kg]
        T : float
            Temperature [K]
        mu : float
            Magnetic moment per particle [Bohr magneton units]
        
        Returns
        -------
        M : array-like
            Magnetization [emu/g]
        """
        mu_B = 9.274e-24  # Bohr magneton [J/T]
        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]
        k_B = 1.38065e-23  # Boltzmann constant [J/K]
        
        # Convert field to SI units (Oe → T)
        H_T = H * 1e-4  # 1 Oe = 10^-4 T
        
        # Langevin parameter
        xi = mu * mu_B * H_T / (k_B * T)
        
        # Langevin function: L(ξ) = coth(ξ) - 1/ξ
        L = np.where(np.abs(xi) < 1e-3, xi/3, 1/np.tanh(xi) - 1/xi)  # Avoid divergence for small ξ
        
        M = M_s * L
        return M
    
    # Generate simulation data (superparamagnetic nanoparticles)
    H_range = np.linspace(-10000, 10000, 100)  # [Oe]
    M_s_true = 50  # Saturation magnetization [emu/g]
    T_true = 300  # Temperature [K]
    mu_true = 5000  # Particle magnetic moment [μ_B]
    
    M_data = langevin(H_range, M_s_true, T_true, mu_true)
    M_data_noise = M_data + 0.5 * np.random.randn(len(H_range))  # Add noise
    
    # Fitting
    model = Model(langevin)
    params = model.make_params(M_s=40, T=300, mu=3000)
    params['T'].vary = False  # Fix temperature
    
    result = model.fit(M_data_noise, params, H=H_range)
    
    print("Langevin Function Fitting Results:")
    print(result.fit_report())
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: M vs H
    ax1.scatter(H_range, M_data_noise, s=50, alpha=0.6, edgecolors='black', linewidths=1, label='Data (with noise)', color='#f093fb')
    ax1.plot(H_range, result.best_fit, linewidth=2.5, label=f'Fit: M_s={result.params["M_s"].value:.1f} emu/g, μ={result.params["mu"].value:.0f} μ_B', color='#f5576c')
    ax1.axhline(M_s_true, color='green', linestyle='--', linewidth=1.5, label=f'True M_s = {M_s_true} emu/g')
    ax1.axhline(-M_s_true, color='green', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax1.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax1.set_title('Langevin Function Fit (Paramagnetism)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Right: Temperature dependence (Langevin curves at different temperatures)
    T_list = [50, 100, 200, 300, 400]
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(T_list)))
    
    for T, color in zip(T_list, colors):
        M_T = langevin(H_range, result.params['M_s'].value, T, result.params['mu'].value)
        ax2.plot(H_range, M_T, linewidth=2.5, label=f'T = {T} K', color=color)
    
    ax2.set_xlabel('Magnetic Field H [Oe]', fontsize=12)
    ax2.set_ylabel('Magnetization M [emu/g]', fontsize=12)
    ax2.set_title('Temperature Dependence of Langevin Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPhysical Interpretation:")
    print(f"  Saturation magnetization M_s = {result.params['M_s'].value:.1f} emu/g")
    print(f"  Particle magnetic moment μ = {result.params['mu'].value:.0f} μ_B")
    print(f"  Lower temperature approaches saturation at lower field (reduced thermal fluctuation)")
    

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
