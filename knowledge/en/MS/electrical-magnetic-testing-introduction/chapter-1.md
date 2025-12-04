---
title: "Chapter 1: Fundamentals of Electrical Conductivity Measurement"
chapter_title: "Chapter 1: Fundamentals of Electrical Conductivity Measurement"
subtitle: From Drude Model to Four-Point Probe, van der Pauw Method, and Temperature-Dependent Analysis
reading_time: 40-50 min
code_examples: 7
version: 1.0
created_at: 2025-10-28
---

Electrical conductivity measurement is a fundamental method for quantitatively evaluating the electrical properties of materials. In this chapter, you will learn the theoretical foundations of electrical conduction through the Drude model, the differences between two-terminal and four-terminal measurements, the van der Pauw method for measuring arbitrarily shaped samples, information about carrier scattering mechanisms obtained from temperature dependence, and practical data analysis using Python. 

## Learning Objectives

By reading this chapter, you will master the following:

  *  Understand the Drude model and derive the electrical conductivity equation $\sigma = ne^2\tau/m$
  *  Explain the principles and differences between two-terminal and four-terminal measurements
  *  Understand the theory of the van der Pauw method and implement it in Python
  *  Evaluate the effects of contact resistance and perform appropriate corrections
  *  Analyze carrier scattering mechanisms from temperature-dependent data
  *  Understand the relationship between sheet resistance and bulk resistivity
  *  Perform fitting and uncertainty evaluation using Python

## 1.1 Drude Model of Electrical Conduction

### 1.1.1 Fundamentals of the Drude Model

The **Drude model** (proposed in 1900 by Paul Drude) is a classical theory explaining electrical conduction in metals and heavily doped semiconductors. In this model, electrons are treated as a "free electron gas" moving freely.

**Basic Assumptions** :

  * Electrons move freely until they **scatter** with atomic nuclei or other electrons
  * Scattering occurs with an average relaxation time $\tau$ (scattering probability $\propto 1/\tau$)
  * After scattering, the electron velocity is directed randomly (drift from electric field is reset)

**Derivation of Electrical Conductivity** :

When an electric field $\vec{E}$ is applied, electrons are accelerated:

$$ m\frac{d\vec{v}}{dt} = -e\vec{E} - \frac{m\vec{v}}{\tau} $$ 

Here, $m$ is the electron mass, $e$ is the elementary charge ($e > 0$), and $\tau$ is the scattering relaxation time. In steady state ($d\vec{v}/dt = 0$):

$$ \vec{v}_{\text{drift}} = -\frac{e\tau}{m}\vec{E} $$ 

The current density $\vec{j}$ is the product of carrier density $n$, charge $-e$, and drift velocity $\vec{v}_{\text{drift}}$:

$$ \vec{j} = -ne\vec{v}_{\text{drift}} = \frac{ne^2\tau}{m}\vec{E} $$ 

**Electrical conductivity** $\sigma$ and **resistivity** $\rho$ are:

$$ \sigma = \frac{ne^2\tau}{m}, \quad \rho = \frac{1}{\sigma} = \frac{m}{ne^2\tau} $$ 

**Mobility** $\mu$ is the drift velocity per unit electric field:

$$ \mu = \frac{|v_{\text{drift}}|}{E} = \frac{e\tau}{m} $$ 

Therefore, electrical conductivity can also be expressed as:

$$ \sigma = ne\mu $$ 

#### Code Example 1-1: Electrical Conductivity Calculation Using the Drude Model
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def drude_conductivity(n, tau, m=9.10938e-31):
        """
        Calculate electrical conductivity using the Drude model
    
        Parameters
        ----------
        n : float or array-like
            Carrier density [m^-3]
        tau : float or array-like
            Scattering relaxation time [s]
        m : float
            Effective mass [kg] (default: free electron mass)
    
        Returns
        -------
        sigma : float or array-like
            Electrical conductivity [S/m]
        """
        e = 1.60218e-19  # Elementary charge [C]
        sigma = n * e**2 * tau / m
        return sigma
    
    # Typical metal (copper) parameters
    n_Cu = 8.5e28  # Carrier density [m^-3] (free electron density of copper)
    tau_Cu = 2.5e-14  # Scattering relaxation time [s] (room temperature)
    
    sigma_Cu = drude_conductivity(n_Cu, tau_Cu)
    rho_Cu = 1 / sigma_Cu
    
    print(f"Copper electrical conductivity: {sigma_Cu:.3e} S/m")
    print(f"Copper resistivity: {rho_Cu:.3e} ©·m = {rho_Cu * 1e8:.2f} ¼©·cm")
    print(f"Experimental value (room temp): Á H 1.68 ¼©·cm")
    
    # Carrier density dependence
    n_range = np.logspace(26, 30, 100)  # [m^-3]
    tau_fixed = 1e-14  # [s]
    
    sigma_n = drude_conductivity(n_range, tau_fixed)
    
    # Scattering relaxation time dependence
    n_fixed = 1e28  # [m^-3]
    tau_range = np.logspace(-15, -12, 100)  # [s]
    
    sigma_tau = drude_conductivity(n_fixed, tau_range)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Carrier density dependence
    ax1.loglog(n_range, sigma_n, linewidth=2.5, color='#f093fb')
    ax1.scatter([n_Cu], [sigma_Cu], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='Cu (room temp)')
    ax1.set_xlabel('Carrier Density n [m$^{-3}$]', fontsize=12)
    ax1.set_ylabel('Conductivity Ã [S/m]', fontsize=12)
    ax1.set_title('Conductivity vs Carrier Density\n(Ä = 10 fs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # Right: Scattering relaxation time dependence
    ax2.loglog(tau_range * 1e15, sigma_tau, linewidth=2.5, color='#f093fb')
    ax2.scatter([tau_Cu * 1e15], [sigma_Cu], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='Cu (room temp)')
    ax2.set_xlabel('Scattering Relaxation Time Ä [fs]', fontsize=12)
    ax2.set_ylabel('Conductivity Ã [S/m]', fontsize=12)
    ax2.set_title('Conductivity vs Relaxation Time\n(n = 10$^{28}$ m$^{-3}$)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**Output Interpretation** :

  * Copper electrical conductivity: approximately $5.8 \times 10^7$ S/m (agrees well with experimental value)
  * Electrical conductivity is proportional to $n$ and $\tau$ ($\sigma \propto n\tau$)
  * In metals, $n$ is nearly constant, so temperature dependence is mainly due to $\tau$

### 1.1.2 Temperature Dependence and Scattering Mechanisms

The scattering relaxation time $\tau$ varies with temperature depending on the scattering mechanism.

Scattering Mechanism | Temperature Dependence | Dominant Temperature Range | Material Examples  
---|---|---|---  
**Phonon scattering** | $\rho \propto T$ (high temp) | Above room temperature | Pure metals  
**Impurity scattering** | $\rho = \rho_0$ (constant) | Low temperature | Alloys, doped semiconductors  
**Grain boundary scattering** | Weak temperature dependence | All temperature ranges | Polycrystalline materials  
**Electron-electron scattering** | $\rho \propto T^2$ | Very low temperature | Fermi liquids  
  
**Matthiessen's Rule** : When multiple scattering mechanisms work independently, the total resistivity is the sum of contributions from each mechanism:

$$ \rho(T) = \rho_0 + \rho_{\text{phonon}}(T) + \rho_{\text{other}}(T) $$ 

Here, $\rho_0$ is the residual resistivity (due to impurities and defects, independent of temperature), and $\rho_{\text{phonon}}(T)$ is the temperature-dependent term due to phonon scattering.

## 1.2 Two-Terminal and Four-Terminal Measurements

### 1.2.1 Problems with Two-Terminal Measurement

In two-terminal measurement, the terminals for passing current and measuring voltage are the same, so **contact resistance** and **wire resistance** are included in the measured value.
    
    
    ```mermaid
    flowchart LR
        A[Current Source] -->|I| B[Contact 1R_c1]
        B --> C[SampleR_sample]
        C --> D[Contact 2R_c2]
        D -->|I| E[Voltmeter]
        E --> A
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style E fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#ffeb99,stroke:#ffa500,stroke-width:2px
    ```

Measured voltage:

$$ V_{\text{measured}} = I(R_{\text{c1}} + R_{\text{sample}} + R_{\text{c2}}) $$ 

Contact resistance $R_c$ can be larger than the sample resistance $R_{\text{sample}}$ (especially for thin films and semiconductors), hindering accurate measurement.

### 1.2.2 Four-Terminal Measurement (Kelvin Measurement)

In **four-terminal measurement** , current and voltage terminals are separated, eliminating the effect of contact resistance.
    
    
    ```mermaid
    flowchart LR
        A[Current Source] -->|I| B[Current Contact 1]
        B --> C[SampleR_sample]
        C --> D[Current Contact 2]
        D -->|I| A
    
        E[High Input ImpedanceVoltmeter] -.->|V+| F[Voltage Contact 3]
        F --> C
        C --> G[Voltage Contact 4]
        G -.->|V-| E
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style E fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

Since the voltmeter has high input impedance (ideally infinite), almost no current flows through the voltage terminals. Therefore, the voltage drop due to contact resistance at the voltage terminals is zero, and only the voltage inside the sample is measured:

$$ V_{\text{sample}} = I \cdot R_{\text{sample}} $$ 

**Advantages of Four-Terminal Measurement** :

  * Not affected by contact resistance
  * Wire resistance effects can also be eliminated
  * High-precision measurement of low-resistance materials possible (¼© order measurable)

#### Code Example 1-2: Simulation of Two-Terminal vs Four-Terminal Measurement
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def two_terminal_measurement(R_sample, R_contact, I):
        """
        Simulate two-terminal measurement
        """
        V_measured = I * (2 * R_contact + R_sample)
        R_measured = V_measured / I
        return R_measured
    
    def four_terminal_measurement(R_sample, I):
        """
        Simulate four-terminal measurement
        """
        V_sample = I * R_sample
        R_measured = V_sample / I
        return R_measured
    
    # Parameters
    R_sample = 1.0  # Sample resistance [©]
    R_contact_range = np.linspace(0, 5, 100)  # Contact resistance [©]
    I = 0.1  # Current [A]
    
    # Calculate measured values
    R_2terminal = [two_terminal_measurement(R_sample, Rc, I) for Rc in R_contact_range]
    R_4terminal = [four_terminal_measurement(R_sample, I) for Rc in R_contact_range]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(R_contact_range, R_2terminal, linewidth=2.5, label='2-terminal (with contact resistance)', color='#ffa500')
    ax.plot(R_contact_range, R_4terminal, linewidth=2.5, label='4-terminal (no contact resistance)', color='#f093fb', linestyle='--')
    ax.axhline(y=R_sample, color='black', linestyle=':', linewidth=1.5, label=f'True sample resistance = {R_sample} ©')
    
    ax.set_xlabel('Contact Resistance R$_c$ [©]', fontsize=12)
    ax.set_ylabel('Measured Resistance [©]', fontsize=12)
    ax.set_title('2-Terminal vs 4-Terminal Measurement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Specific example
    R_contact_example = 2.0  # [©]
    R_2t = two_terminal_measurement(R_sample, R_contact_example, I)
    R_4t = four_terminal_measurement(R_sample, I)
    
    print(f"Sample resistance: {R_sample} ©")
    print(f"Contact resistance: {R_contact_example} © (each contact)")
    print(f"2-terminal measurement: {R_2t:.2f} © (error: {(R_2t - R_sample) / R_sample * 100:.1f}%)")
    print(f"4-terminal measurement: {R_4t:.2f} © (error: 0%)")
    

## 1.3 van der Pauw Method

### 1.3.1 van der Pauw Theorem

The **van der Pauw method** (proposed in 1958 by L.J. van der Pauw) is a powerful technique for measuring the sheet resistance of arbitrarily shaped thin film samples. It does not require the sample to have a specific shape (rectangular, circular, etc.) and can be measured with just four contacts.

**Conditions** :

  * The sample is planar with uniform thickness $t$
  * The sample has no holes or defects (simply connected)
  * Four contacts are placed at the edge of the sample
  * Contacts are sufficiently small

**van der Pauw Theorem** :

Place four contacts A, B, C, D around the sample and measure the following two resistances:

$$ R_{\text{AB,CD}} = \frac{V_{\text{CD}}}{I_{\text{AB}}} \quad \text{(current from AB, voltage measured at CD)} $$ $$ R_{\text{BC,DA}} = \frac{V_{\text{DA}}}{I_{\text{BC}}} \quad \text{(current from BC, voltage measured at DA)} $$ 

By the van der Pauw theorem, the sheet resistance $R_s$ satisfies the following equation:

$$ \exp\left(-\frac{\pi R_{\text{AB,CD}}}{R_s}\right) + \exp\left(-\frac{\pi R_{\text{BC,DA}}}{R_s}\right) = 1 $$ 

By solving this equation for $R_s$, the sheet resistance is obtained.

**Relationship between Sheet Resistance and Bulk Resistivity** :

$$ R_s = \frac{\rho}{t} $$ 

Here, $\rho$ is the bulk resistivity and $t$ is the sample thickness. The unit of $R_s$ is expressed as ©/¡ (ohm per square).

#### Code Example 1-3: Sheet Resistance Calculation Using the van der Pauw Method
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    def van_der_pauw_equation(Rs, R1, R2):
        """
        van der Pauw equation: exp(-À R1/Rs) + exp(-À R2/Rs) = 1
        """
        return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
    def calculate_sheet_resistance(R_AB_CD, R_BC_DA):
        """
        Calculate sheet resistance using the van der Pauw method
    
        Parameters
        ----------
        R_AB_CD : float
            Resistance R_AB,CD [©]
        R_BC_DA : float
            Resistance R_BC,DA [©]
    
        Returns
        -------
        R_s : float
            Sheet resistance [©/sq]
        """
        # Initial estimate: average resistance
        R_initial = (R_AB_CD + R_BC_DA) / 2 * np.pi / np.log(2)
    
        # Solve numerically
        R_s = fsolve(van_der_pauw_equation, R_initial, args=(R_AB_CD, R_BC_DA))[0]
    
        return R_s
    
    # Measurement example 1: Square sample (symmetric configuration)
    R1 = 100  # [©]
    R2 = 100  # [©]
    
    R_s1 = calculate_sheet_resistance(R1, R2)
    print("Example 1: Square sample (symmetric configuration)")
    print(f"  R_AB,CD = {R1:.1f} ©")
    print(f"  R_BC,DA = {R2:.1f} ©")
    print(f"  Sheet resistance R_s = {R_s1:.2f} ©/sq")
    print(f"  Approximate formula R_s H (À/ln2)(R1+R2)/2 = {np.pi / np.log(2) * (R1 + R2) / 2:.2f} ©/sq")
    
    # Measurement example 2: Asymmetric configuration
    R1 = 120  # [©]
    R2 = 80   # [©]
    
    R_s2 = calculate_sheet_resistance(R1, R2)
    print("\nExample 2: Asymmetric configuration")
    print(f"  R_AB,CD = {R1:.1f} ©")
    print(f"  R_BC,DA = {R2:.1f} ©")
    print(f"  Sheet resistance R_s = {R_s2:.2f} ©/sq")
    
    # Calculate bulk resistivity from thickness
    t = 100e-9  # Thickness 100 nm
    rho1 = R_s1 * t
    rho2 = R_s2 * t
    
    print(f"\nWith thickness t = {t * 1e9:.0f} nm:")
    print(f"  Example 1 bulk resistivity Á = {rho1:.3e} ©·m = {rho1 * 1e8:.2f} ¼©·cm")
    print(f"  Example 2 bulk resistivity Á = {rho2:.3e} ©·m = {rho2 * 1e8:.2f} ¼©·cm")
    
    # Visualization of van der Pauw equation
    R1_range = np.linspace(50, 150, 50)
    R2_range = np.linspace(50, 150, 50)
    R1_mesh, R2_mesh = np.meshgrid(R1_range, R2_range)
    
    R_s_mesh = np.zeros_like(R1_mesh)
    for i in range(R1_mesh.shape[0]):
        for j in range(R1_mesh.shape[1]):
            R_s_mesh[i, j] = calculate_sheet_resistance(R1_mesh[i, j], R2_mesh[i, j])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(R1_mesh, R2_mesh, R_s_mesh, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Sheet Resistance R$_s$ [©/sq]', fontsize=12)
    
    ax.scatter([R1], [R2], s=200, c='white', edgecolors='black', linewidth=2, marker='o', label='Example 2')
    ax.set_xlabel('R$_{AB,CD}$ [©]', fontsize=12)
    ax.set_ylabel('R$_{BC,DA}$ [©]', fontsize=12)
    ax.set_title('van der Pauw Method: Sheet Resistance Map', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.show()
    

**Output Interpretation** :

  * For symmetric configuration ($R_1 = R_2$), the simplified formula $R_s \approx \frac{\pi}{\ln 2}R_1$ is often used
  * Even for asymmetric configuration, accurate $R_s$ can be obtained by numerically solving the van der Pauw equation
  * Bulk resistivity is obtained by multiplying sheet resistance by thickness

## 1.4 Temperature Dependence Measurement and Fitting

### 1.4.1 Temperature Dependence of Metals (Bloch-Grüneisen Model)

For metals, resistivity is dominated by impurity scattering (temperature-independent $\rho_0$) at low temperatures and phonon scattering (proportional to $T$) at high temperatures:

$$ \rho(T) = \rho_0 + A T $$ 

Here, $A$ is a coefficient related to phonon scattering.

### 1.4.2 Temperature Dependence of Semiconductors (Arrhenius Plot)

In semiconductors, carrier density varies with temperature (thermal excitation):

$$ n(T) \propto \exp\left(-\frac{E_a}{k_B T}\right) $$ 

Resistivity is:

$$ \rho(T) = \rho_0 \exp\left(\frac{E_a}{k_B T}\right) $$ 

A plot of $\ln \rho$ vs $1/T$ (Arrhenius plot) yields the activation energy $E_a$.

#### Code Example 1-4: Fitting Temperature-Dependent Data
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1-4: Fitting Temperature-Dependent Data
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    # Metal model: Á(T) = Á€ + A*T
    def metal_resistivity(T, rho0, A):
        return rho0 + A * T
    
    # Semiconductor model: Á(T) = Á€ * exp(Ea / (kB * T))
    def semiconductor_resistivity(T, rho0, Ea):
        kB = 8.617333e-5  # Boltzmann constant [eV/K]
        return rho0 * np.exp(Ea / (kB * T))
    
    # Generate simulation data
    # Metal (copper)
    T_metal = np.linspace(50, 400, 30)  # [K]
    rho0_true = 0.5e-8  # [©·m]
    A_true = 5e-11  # [©·m/K]
    rho_metal = metal_resistivity(T_metal, rho0_true, A_true) * (1 + 0.02 * np.random.randn(len(T_metal)))  # 2% noise
    
    # Semiconductor (silicon)
    T_semi = np.linspace(300, 600, 30)  # [K]
    rho0_semi_true = 1e-5  # [©·m]
    Ea_true = 0.5  # [eV]
    rho_semi = semiconductor_resistivity(T_semi, rho0_semi_true, Ea_true) * (1 + 0.05 * np.random.randn(len(T_semi)))  # 5% noise
    
    # Fitting: metal
    metal_model = Model(metal_resistivity)
    metal_params = metal_model.make_params(rho0=1e-8, A=1e-11)
    metal_result = metal_model.fit(rho_metal, metal_params, T=T_metal)
    
    print("Metal fitting results:")
    print(metal_result.fit_report())
    
    # Fitting: semiconductor
    semi_model = Model(semiconductor_resistivity)
    semi_params = semi_model.make_params(rho0=1e-6, Ea=0.6)
    semi_result = semi_model.fit(rho_semi, semi_params, T=T_semi)
    
    print("\nSemiconductor fitting results:")
    print(semi_result.fit_report())
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metal: Á vs T
    axes[0, 0].scatter(T_metal, rho_metal * 1e8, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (with noise)', color='#f093fb')
    axes[0, 0].plot(T_metal, metal_result.best_fit * 1e8, linewidth=2.5, label='Fit: Á = Á€ + AT', color='#f5576c')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Resistivity Á [¼©·cm]', fontsize=12)
    axes[0, 0].set_title('Metal (Cu): Resistivity vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Metal: residuals
    residuals_metal = rho_metal - metal_result.best_fit
    axes[0, 1].scatter(T_metal, residuals_metal * 1e8, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, color='#99ccff')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Residuals [¼©·cm]', fontsize=12)
    axes[0, 1].set_title('Fit Residuals (Metal)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Semiconductor: Á vs T
    axes[1, 0].scatter(T_semi, rho_semi, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (with noise)', color='#ffa500')
    axes[1, 0].plot(T_semi, semi_result.best_fit, linewidth=2.5, label='Fit: Á = Á€exp(Ea/kT)', color='#ff6347')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Resistivity Á [©·m]', fontsize=12)
    axes[1, 0].set_title('Semiconductor (Si): Resistivity vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Semiconductor: Arrhenius plot
    axes[1, 1].scatter(1000 / T_semi, np.log(rho_semi), s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data', color='#ffa500')
    axes[1, 1].plot(1000 / T_semi, np.log(semi_result.best_fit), linewidth=2.5, label='Fit (Arrhenius)', color='#ff6347')
    axes[1, 1].set_xlabel('1000/T [K$^{-1}$]', fontsize=12)
    axes[1, 1].set_ylabel('ln(Á)', fontsize=12)
    axes[1, 1].set_title('Arrhenius Plot (Semiconductor)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 1.5 Contact Resistance Evaluation and Correction

### 1.5.1 Transfer Length Method (TLM)

The **TLM (Transfer Length Method)** is a technique for quantitatively evaluating contact resistance. Contact pairs are created at different spacings, and contact resistance is determined from the relationship between measured resistance and contact spacing.

The total measured resistance $R_{\text{total}}$ is:

$$ R_{\text{total}} = 2R_c + R_s \frac{L}{W} $$ 

Here, $R_c$ is the contact resistance, $R_s$ is the sheet resistance, $L$ is the contact spacing, and $W$ is the sample width. When $R_{\text{total}}$ is plotted against $L$, the intercept gives $2R_c$ and the slope gives $R_s/W$.

#### Code Example 1-5: Contact Resistance Evaluation Using TLM
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 1-5: Contact Resistance Evaluation Using TLM
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # Parameters
    R_s = 50  # Sheet resistance [©/sq]
    R_c = 10  # Contact resistance [©]
    W = 0.01  # Sample width [m] = 1 cm
    
    # Contact spacing
    L_values = np.array([0.001, 0.002, 0.003, 0.005, 0.010])  # [m]
    
    # Measured resistance (with noise)
    R_total = 2 * R_c + R_s * L_values / W
    R_total_noise = R_total * (1 + 0.03 * np.random.randn(len(L_values)))  # 3% noise
    
    # Linear fitting
    slope, intercept, r_value, p_value, std_err = linregress(L_values * 1000, R_total_noise)
    
    R_c_fit = intercept / 2
    R_s_fit = slope * W * 1000  # slope × W
    
    print("Contact resistance evaluation using TLM:")
    print(f"  True contact resistance R_c = {R_c:.2f} ©")
    print(f"  Fitted R_c = {R_c_fit:.2f} ©")
    print(f"  True sheet resistance R_s = {R_s:.2f} ©/sq")
    print(f"  Fitted R_s = {R_s_fit:.2f} ©/sq")
    print(f"  Coefficient of determination R² = {r_value**2:.4f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(L_values * 1000, R_total_noise, s=150, edgecolors='black', linewidths=2, label='Measured data', color='#f093fb', zorder=5)
    L_fit = np.linspace(0, np.max(L_values) * 1000, 100)
    R_fit = slope * L_fit + intercept
    ax.plot(L_fit, R_fit, linewidth=2.5, label=f'Linear fit: R = {slope:.2f}L + {intercept:.2f}', color='#f5576c', linestyle='--')
    
    # Highlight intercept
    ax.axhline(y=intercept, color='orange', linestyle=':', linewidth=2, label=f'Intercept = 2R$_c$ = {intercept:.2f} ©')
    ax.scatter([0], [intercept], s=200, c='orange', edgecolors='black', linewidth=2, marker='s', zorder=5)
    
    ax.set_xlabel('Contact Spacing L [mm]', fontsize=12)
    ax.set_ylabel('Total Resistance R$_{total}$ [©]', fontsize=12)
    ax.set_title('Transfer Length Method (TLM)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.show()
    

## 1.6 Exercises

### Exercise 1-1: Drude Model Calculation (Easy)

Easy **Problem** : For gold with carrier density $n = 5.9 \times 10^{28}$ m$^{-3}$ and scattering relaxation time $\tau = 3 \times 10^{-14}$ s, calculate the electrical conductivity and resistivity.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: EasyProblem: For gold with carrier density $n = 5.9 \times 1
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Constants
    e = 1.60218e-19  # [C]
    m = 9.10938e-31  # [kg]
    
    # Gold parameters
    n = 5.9e28  # [m^-3]
    tau = 3e-14  # [s]
    
    # Electrical conductivity
    sigma = n * e**2 * tau / m
    rho = 1 / sigma
    
    print(f"Electrical conductivity Ã = {sigma:.3e} S/m")
    print(f"Resistivity Á = {rho:.3e} ©·m = {rho * 1e8:.2f} ¼©·cm")
    print(f"Experimental value (room temp): Á H 2.44 ¼©·cm")
    

**Answer** : Ã H 4.54 × 10$^7$ S/m, Á H 2.20 × 10$^{-8}$ ©·m = 2.20 ¼©·cm (agrees well with experimental value)

### Exercise 1-2: Mobility Calculation (Easy)

Easy **Problem** : Calculate the mobility $\mu$ of an electron with scattering relaxation time $\tau = 1 \times 10^{-14}$ s.

**Show Solution**
    
    
    e = 1.60218e-19  # [C]
    m = 9.10938e-31  # [kg]
    tau = 1e-14  # [s]
    
    mu = e * tau / m
    print(f"Mobility ¼ = {mu:.3e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**Answer** : ¼ H 1.76 × 10$^{-3}$ m$^2$/(V·s) = 17.6 cm$^2$/(V·s)

### Exercise 1-3: van der Pauw Method Calculation (Medium)

Medium **Problem** : In van der Pauw measurement, $R_{\text{AB,CD}} = 85$ © and $R_{\text{BC,DA}} = 115$ © were obtained. Calculate the sheet resistance $R_s$ and determine the bulk resistivity $\rho$ for thickness $t = 50$ nm.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MediumProblem: In van der Pauw measurement, $R_{\text{AB,CD}
    
    Purpose: Demonstrate optimization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_eq(Rs, R1, R2):
        return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
    R1 = 85  # [©]
    R2 = 115  # [©]
    
    R_s = fsolve(van_der_pauw_eq, (R1 + R2) / 2 * np.pi / np.log(2), args=(R1, R2))[0]
    
    print(f"Sheet resistance R_s = {R_s:.2f} ©/sq")
    
    t = 50e-9  # [m]
    rho = R_s * t
    
    print(f"Thickness t = {t * 1e9:.0f} nm")
    print(f"Bulk resistivity Á = {rho:.3e} ©·m = {rho * 1e8:.2f} ¼©·cm")
    

**Answer** : R$_s$ H 136.8 ©/sq, Á H 6.84 × 10$^{-6}$ ©·m = 684 ¼©·cm

### Exercise 1-4: Temperature Dependence Fitting (Medium)

Medium **Problem** : A metal sample has resistivity of 0.8 ¼©·cm at T = 100 K and 2.0 ¼©·cm at 300 K. Fit with model $\rho(T) = \rho_0 + AT$ and determine $\rho_0$ and $A$.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MediumProblem: A metal sample has resistivity of 0.8 ¼©·cm a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    T1, rho1 = 100, 0.8e-8  # [K], [©·m]
    T2, rho2 = 300, 2.0e-8  # [K], [©·m]
    
    # Solve linear equations
    A = (rho2 - rho1) / (T2 - T1)
    rho0 = rho1 - A * T1
    
    print(f"Á€ = {rho0:.3e} ©·m = {rho0 * 1e8:.2f} ¼©·cm")
    print(f"A = {A:.3e} ©·m/K")
    
    # Verification
    rho_100 = rho0 + A * 100
    rho_300 = rho0 + A * 300
    print(f"\nVerification:")
    print(f"  Á(100 K) = {rho_100 * 1e8:.2f} ¼©·cm (given value: 0.80 ¼©·cm)")
    print(f"  Á(300 K) = {rho_300 * 1e8:.2f} ¼©·cm (given value: 2.00 ¼©·cm)")
    

**Answer** : Á$_0$ = 2.00 × 10$^{-9}$ ©·m = 0.20 ¼©·cm, A = 6.00 × 10$^{-11}$ ©·m/K

### Exercise 1-5: TLM Analysis (Medium)

Medium **Problem** : In TLM measurement, for contact spacing $L$ = 1, 2, 3, 4, 5 mm, total resistance $R_{\text{total}}$ = 25, 30, 35, 40, 45 © was obtained. With sample width $W = 1$ cm, determine contact resistance $R_c$ and sheet resistance $R_s$.

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MediumProblem: In TLM measurement, for contact spacing $L$ =
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.stats import linregress
    
    L = np.array([1, 2, 3, 4, 5])  # [mm]
    R_total = np.array([25, 30, 35, 40, 45])  # [©]
    W = 0.01  # [m]
    
    slope, intercept, r_value, _, _ = linregress(L, R_total)
    
    R_c = intercept / 2
    R_s = slope * W * 1000  # Convert W to mm
    
    print(f"Contact resistance R_c = {R_c:.2f} ©")
    print(f"Sheet resistance R_s = {R_s:.2f} ©/sq")
    print(f"Coefficient of determination R² = {r_value**2:.4f}")
    

**Answer** : R$_c$ = 10.0 ©, R$_s$ = 50.0 ©/sq, R$^2$ = 1.0000 (perfect linear relationship)

### Exercise 1-6: Semiconductor Activation Energy (Hard)

Hard **Problem** : A semiconductor sample has resistivity of 1.0 ©·m at T = 300 K and 0.1 ©·m at 400 K. Fit with model $\rho(T) = \rho_0 \exp(E_a / k_B T)$ and determine activation energy $E_a$ in eV ($k_B = 8.617 \times 10^{-5}$ eV/K).

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: HardProblem: A semiconductor sample has resistivity of 1.0 ©
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    T1, rho1 = 300, 1.0  # [K], [©·m]
    T2, rho2 = 400, 0.1  # [K], [©·m]
    kB = 8.617e-5  # [eV/K]
    
    # ln(Á) = ln(Á€) + Ea/(kB T)
    # ln(rho1) = ln(rho0) + Ea/(kB T1)
    # ln(rho2) = ln(rho0) + Ea/(kB T2)
    # ln(rho1) - ln(rho2) = Ea/(kB) * (1/T1 - 1/T2)
    
    Ea = kB * (np.log(rho1) - np.log(rho2)) / (1/T1 - 1/T2)
    
    print(f"Activation energy Ea = {Ea:.3f} eV")
    
    # Determine Á€
    rho0 = rho1 * np.exp(-Ea / (kB * T1))
    print(f"Á€ = {rho0:.3e} ©·m")
    
    # Verification
    rho_300 = rho0 * np.exp(Ea / (kB * 300))
    rho_400 = rho0 * np.exp(Ea / (kB * 400))
    print(f"\nVerification:")
    print(f"  Á(300 K) = {rho_300:.2f} ©·m (given value: 1.00 ©·m)")
    print(f"  Á(400 K) = {rho_400:.2f} ©·m (given value: 0.10 ©·m)")
    

**Answer** : E$_a$ H 0.661 eV, Á$_0$ H 3.94 × 10$^{-13}$ ©·m

### Exercise 1-7: Four-Terminal Measurement Error Evaluation (Hard)

Hard **Problem** : With sample resistance $R_{\text{sample}} = 0.5$ ©, contact resistance $R_c = 10$ ©, and current $I = 0.1$ A, calculate the relative error for two-terminal and four-terminal measurements. Also determine what maximum contact resistance allows the two-terminal measurement to have relative error within 5%.

**Show Solution**
    
    
    R_sample = 0.5  # [©]
    R_c = 10  # [©]
    
    # Two-terminal measurement
    R_2terminal = 2 * R_c + R_sample
    error_relative = (R_2terminal - R_sample) / R_sample * 100
    
    print(f"Sample resistance R_sample = {R_sample} ©")
    print(f"Contact resistance R_c = {R_c} ©")
    print(f"Two-terminal measurement: {R_2terminal} ©")
    print(f"Relative error: {error_relative:.1f}%")
    
    # Condition for relative error within 5%
    # (2*R_c + R_sample - R_sample) / R_sample <= 0.05
    # 2*R_c / R_sample <= 0.05
    # R_c <= 0.05 * R_sample / 2
    
    R_c_max = 0.05 * R_sample / 2
    print(f"\nCondition for relative error within 5%: R_c <= {R_c_max:.4f} © = {R_c_max * 1000:.2f} m©")
    

**Answer** : Relative error 4100%, R$_c$ d 0.0125 © = 12.5 m© (very stringent condition)

### Exercise 1-8: Experimental Design (Hard)

Hard **Problem** : Design an experimental plan to evaluate the electrical conduction properties of an unknown thin film material (thickness 200 nm). Explain measurement methods (two-terminal/four-terminal, van der Pauw), temperature range, and data analysis methods.

**Show Solution**

**Experimental Plan** :

  1. **Sample Preparation** : Place four contacts (diameter < 0.5 mm) at four corners (van der Pauw configuration)
  2. **Room Temperature Measurement** : 
     * Measure $R_{\text{AB,CD}}$ and $R_{\text{BC,DA}}$ using van der Pauw method
     * Calculate sheet resistance $R_s$, determine resistivity $\rho$ from thickness 200 nm
  3. **Temperature Dependence Measurement** : 
     * Temperature range: 77 K (liquid nitrogen temperature) to 400 K
     * Measurement interval: every 20-30 K, 20-30 points
     * Wait for thermal equilibrium at each temperature (10-15 min)
  4. **Data Analysis** : 
     * Create $\rho$ vs $T$ plot
     * Determine metallic behavior ($\rho \propto T$) or semiconducting behavior ($\rho \propto \exp(E_a / k_B T)$)
     * Fit with appropriate model (using lmfit)
     * Evaluate residual resistivity $\rho_0$ using Matthiessen's rule
  5. **Contact Resistance Evaluation (Optional)** : Measure multiple contact spacings using TLM method to quantify contact resistance contribution

**Expected Results** :

  * Metallic material: $\rho \approx 1-100$ ¼©·cm, positive temperature coefficient
  * Semiconductor material: $\rho \approx 0.1-1000$ ©·cm, negative temperature coefficient, activation energy 0.1-1 eV

## 1.7 Learning Verification

Check your understanding with the following checklist:

### Basic Understanding

  * Can explain the assumptions of the Drude model and the electrical conductivity equation $\sigma = ne^2\tau/m$
  * Understand the physical meaning of scattering relaxation time $\tau$
  * Can explain the principles and differences between two-terminal and four-terminal measurements
  * Understand the van der Pauw theorem equation and can calculate sheet resistance
  * Understand Matthiessen's rule

### Practical Skills

  * Can calculate electrical conductivity using the Drude model
  * Can implement Python code for the van der Pauw method
  * Can fit temperature-dependent data (using lmfit)
  * Can evaluate contact resistance using TLM method
  * Can evaluate uncertainty in measurement data

### Applied Ability

  * Can explain differences in temperature dependence between metals and semiconductors
  * Can estimate scattering mechanisms from experimental data
  * Can select appropriate measurement method (two-terminal/four-terminal, van der Pauw)
  * Can identify main sources of measurement error and propose countermeasures

## 1.8 References

  1. van der Pauw, L. J. (1958). _A method of measuring specific resistivity and Hall effect of discs of arbitrary shape_. Philips Research Reports, 13(1), 1-9. - Original paper on van der Pauw method
  2. Drude, P. (1900). _Zur Elektronentheorie der Metalle_. Annalen der Physik, 306(3), 566-613. - Original paper on Drude model
  3. Schroder, D. K. (2006). _Semiconductor Material and Device Characterization_ (3rd ed.). Wiley-Interscience. - Standard textbook on semiconductor measurement techniques
  4. Streetman, B. G., & Banerjee, S. K. (2015). _Solid State Electronic Devices_ (7th ed.). Pearson. - Textbook on electrical conduction theory
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Holt, Rinehart and Winston. - Details on Drude model and metal properties
  6. Cohen, M. H., et al. (1960). _Contact Resistance and Methods for Its Determination_. Solid-State Electronics, 1(2), 159-169. - Contact resistance measurement techniques
  7. Reeves, G. K., & Harrison, H. B. (1982). _Obtaining the specific contact resistance from transmission line model measurements_. IEEE Electron Device Letters, 3(5), 111-113. - Practical explanation of TLM method

## 1.9 Next Chapter

In the next chapter, you will learn the principles and practice of **Hall effect measurement**. The Hall effect is a powerful method that uses the deflection of charges in a magnetic field to determine carrier density and carrier type (electron/hole). Combined with the van der Pauw method, it enables complete characterization of the electrical properties of materials.
