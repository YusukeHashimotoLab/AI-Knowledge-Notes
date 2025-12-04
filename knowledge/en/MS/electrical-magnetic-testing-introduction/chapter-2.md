---
title: "Chapter 2: Hall Effect Measurement"
chapter_title: "Chapter 2: Hall Effect Measurement"
subtitle: From Lorentz Force to van der Pauw Hall Configuration, Multi-Carrier Analysis, and Temperature Dependence
reading_time: 45-55 minutes
difficulty: Intermediate
code_examples: 7
version: 1.0
created_at: "by:"
---

The Hall effect is a powerful technique that utilizes the deflection of charge carriers in a magnetic field to determine carrier density, carrier type (electron/hole), and mobility. In this chapter, you will learn the theory of the Hall effect based on the Lorentz force, the relationship between Hall coefficient and carrier density, van der Pauw Hall measurement configuration, multi-carrier system analysis, and the elucidation of carrier scattering mechanisms from temperature dependence, and perform practical Hall data analysis using Python. 

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Derive Hall effect equations from the Lorentz force and calculate Hall voltage
  * ✅ Understand the relationship between Hall coefficient $R_H = 1/(ne)$ and carrier density
  * ✅ Explain the van der Pauw Hall measurement configuration and measurement procedure
  * ✅ Calculate mobility $\mu = \sigma R_H$ and understand its physical meaning
  * ✅ Analyze multi-carrier systems (two-band model)
  * ✅ Analyze carrier scattering mechanisms from temperature dependence
  * ✅ Build a complete Hall data processing workflow in Python

## 2.1 Theory of Hall Effect

### 2.1.1 Lorentz Force and Hall Voltage

**Hall effect** (discovered in 1879 by Edwin Herbert Hall) is a phenomenon in which, when a perpendicular magnetic field is applied to a conductor carrying electric current, a voltage is generated in a direction perpendicular to both the current and the magnetic field.
    
    
    ```mermaid
    flowchart TD
        A[Current Ix-direction] --> B[Magnetic Field Bz-direction]
        B --> C[Lorentz ForceF = -e v × B]
        C --> D[Carrier Deflectiony-direction]
        D --> E[Hall Voltage V_HGenerated in y-direction]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style B fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#ff9999,stroke:#ff0000,stroke-width:2px
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

**Physical Mechanism** :

  1. When current $I$ flows through a conductor, carriers (electrons) move in the x-direction at drift velocity $v_x$
  2. When a magnetic field $B_z$ is applied in the z-direction, the Lorentz force $\vec{F} = q\vec{v} \times \vec{B}$ acts
  3. Electrons ($q = -e$) are deflected in the y-direction and accumulate on one side
  4. Charge accumulation generates Hall electric field $E_y$
  5. In steady state, Lorentz force and electric field force balance: $eE_y = ev_x B_z$

**Derivation of Hall Voltage** :

If the sample width is $w$, thickness $t$, and current $I$, the current density is:

$$ j_x = \frac{I}{wt} $$ 

If the carrier density is $n$, the relationship between current density and drift velocity is:

$$ j_x = nev_x \quad \Rightarrow \quad v_x = \frac{j_x}{ne} = \frac{I}{newt} $$ 

From force balance in steady state:

$$ E_y = v_x B_z = \frac{IB_z}{newt} $$ 

**Hall voltage** $V_H$ は, 試料幅 $w$ にわたる電場のintegral:

$$ V_H = E_y \cdot w = \frac{IB_z}{net} $$ 

**Hall coefficient** $R_H$ is defined as:

$$ R_H = \frac{E_y}{j_x B_z} = \frac{V_H t}{IB_z} = \frac{1}{ne} $$ 

Therefore, **carrier density** $n$ can be directly obtained from the Hall coefficient:

$$ n = \frac{1}{eR_H} $$ 

### 2.1.2 Determination of Carrier Type

The sign of the Hall coefficient can determine whether carriers are electrons or holes:

Carrier type | Hall Coefficient $R_H$ | Sign of Hall Voltage | Physical Interpretation  
---|---|---|---  
**Electron** (n-type) | $R_H < 0$ | Negative | Electrons carry negative charge  
**Hole** (p-type) | $R_H > 0$ | Positive | Holes carry positive charge  
  
#### Code Example 2-1: Calculation of Hall Coefficient and Carrier Density
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_hall_coefficient(V_H, I, B, t):
        """
        Calculate Hall coefficient
    
        Parameters
        ----------
        V_H : float
            Hall voltage [V]
        I : float
            Current [A]
        B : float
            Magnetic field [T]
        t : float
            Sample thickness [m]
    
        Returns
        -------
        R_H : float
            Hall coefficient [m^3/C]
        """
        R_H = V_H * t / (I * B)
        return R_H
    
    def calculate_carrier_density(R_H):
        """
        Calculate carrier density
    
        Parameters
        ----------
        R_H : float
            Hall coefficient [m^3/C]
    
        Returns
        -------
        n : float
            Carrier density [m^-3]
        carrier_type : str
            Carrier type（'electron' or 'hole'）
        """
        e = 1.60218e-19  # Elementary charge [C]
        n = 1 / (np.abs(R_H) * e)
        carrier_type = 'electron' if R_H < 0 else 'hole'
        return n, carrier_type
    
    # Measurement Example 1: n-type Silicon
    V_H1 = -2.5e-3  # Hall voltage [V]（Negative:electron）
    I1 = 1e-3  # Current [A]
    B1 = 0.5  # Magnetic field [T]
    t1 = 500e-9  # Thickness [m] = 500 nm
    
    R_H1 = calculate_hall_coefficient(V_H1, I1, B1, t1)
    n1, type1 = calculate_carrier_density(R_H1)
    
    print("Measurement example 1: n-type silicon")
    print(f"  Hall voltage: {V_H1 * 1e3:.2f} mV")
    print(f"  Hall coefficient: {R_H1:.3e} m³/C")
    print(f"  Carrier type: {type1}")
    print(f"  Carrier density: {n1:.3e} m⁻³ = {n1 / 1e6:.3e} cm⁻³")
    
    # Measurement Example 2: p-type Gallium Arsenide
    V_H2 = +3.8e-3  # Hall voltage [V]（Positive:hole）
    I2 = 1e-3  # Current [A]
    B2 = 0.5  # Magnetic field [T]
    t2 = 300e-9  # Thickness [m] = 300 nm
    
    R_H2 = calculate_hall_coefficient(V_H2, I2, B2, t2)
    n2, type2 = calculate_carrier_density(R_H2)
    
    print("\nMeasurement example 2: p-type gallium arsenide")
    print(f"  Hall voltage: {V_H2 * 1e3:.2f} mV")
    print(f"  Hall coefficient: {R_H2:.3e} m³/C")
    print(f"  Carrier type: {type2}")
    print(f"  Carrier density: {n2:.3e} m⁻³ = {n2 / 1e6:.3e} cm⁻³")
    
    # Visualization of carrier density dependence
    n_range = np.logspace(20, 28, 100)  # [m^-3]
    R_H_range = 1 / (n_range * 1.60218e-19)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: n vs R_H
    ax1.loglog(n_range / 1e6, np.abs(R_H_range), linewidth=2.5, color='#f093fb', label='|R_H| = 1/(ne)')
    ax1.scatter([n1 / 1e6], [np.abs(R_H1)], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='n-Si (example 1)')
    ax1.scatter([n2 / 1e6], [np.abs(R_H2)], s=150, c='#ffa500', edgecolors='black', linewidth=2, zorder=5, label='p-GaAs (example 2)')
    ax1.set_xlabel('Carrier Density n [cm$^{-3}$]', fontsize=12)
    ax1.set_ylabel('|Hall Coefficient R$_H$| [m$^3$/C]', fontsize=12)
    ax1.set_title('Hall Coefficient vs Carrier Density', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # Right: Hall voltage vs magnetic field
    B_range = np.linspace(0, 1, 100)  # [T]
    V_H_n = R_H1 * I1 * B_range / t1 * 1e3  # n-type [mV]
    V_H_p = R_H2 * I2 * B_range / t2 * 1e3  # p-type [mV]
    
    ax2.plot(B_range, V_H_n, linewidth=2.5, color='#f5576c', label='n-type (electron, R_H < 0)')
    ax2.plot(B_range, V_H_p, linewidth=2.5, color='#ffa500', label='p-type (hole, R_H > 0)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax2.set_ylabel('Hall Voltage V$_H$ [mV]', fontsize=12)
    ax2.set_title('Hall Voltage vs Magnetic Field', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 2.2 Determination of Mobility

### 2.2.1 Relationship between Mobility and Hall Coefficient

**Mobility** $\mu$ is obtained from electrical conductivity $\sigma$ and Hall coefficient $R_H$:

$$ \mu = \sigma R_H $$ 

This is directly derived from $\sigma = ne\mu$ and $R_H = 1/(ne)$.

**Physical Meaning** :

  * $\sigma$ is the ease of electrical conduction (depends on both carrier density $n$ and mobility $\mu$)
  * $R_H$ depends only on carrier density $n$
  * By combining both, mobility $\mu$ can be obtained separately

#### Code Example 2-2: Calculation of Mobility
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def calculate_mobility(sigma, R_H):
        """
        Calculate mobility
    
        Parameters
        ----------
        sigma : float
            Electrical conductivity [S/m]
        R_H : float
            Hall coefficient [m^3/C]
    
        Returns
        -------
        mu : float
            Mobility [m^2/(V·s)]
        """
        mu = sigma * np.abs(R_H)
        return mu
    
    # Measurement example: n-type Silicon (from previous example)
    R_H = -2.5e-3  # [m^3/C]
    sigma = 1e4  # Electrical conductivity [S/m](typical value)
    
    mu = calculate_mobility(sigma, R_H)
    
    e = 1.60218e-19  # [C]
    n = 1 / (np.abs(R_H) * e)
    
    print("Electrical properties of n-type Silicon:")
    print(f"  Electrical conductivity σ = {sigma:.2e} S/m")
    print(f"  Hall coefficient R_H = {R_H:.2e} m³/C")
    print(f"  Carrier density n = {n:.2e} m⁻³ = {n / 1e6:.2e} cm⁻³")
    print(f"  Mobility μ = {mu:.2e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    print(f"\nVerification: σ = neμ = {n * e * mu:.2e} S/m(match)")
    
    # Material comparison
    materials = {
        'Si (bulk, n-type)': {'sigma': 1e4, 'R_H': -2.5e-3},
        'GaAs (bulk, n-type)': {'sigma': 1e5, 'R_H': -5e-3},
        'InSb (n-type)': {'sigma': 1e6, 'R_H': -1e-2},
        'Graphene': {'sigma': 1e5, 'R_H': -1e-4}
    }
    
    print("\nMaterial comparison:")
    print(f"{'Material':<25} {'n [cm⁻³]':<15} {'μ [cm²/(V·s)]':<20}")
    print("-" * 60)
    
    for name, props in materials.items():
        n_mat = 1 / (np.abs(props['R_H']) * e)
        mu_mat = calculate_mobility(props['sigma'], props['R_H'])
        print(f"{name:<25} {n_mat / 1e6:.2e}      {mu_mat * 1e4:.1f}")
    

**Output Interpretation** :

  * Silicon mobility: ~1000 cm²/(V·s) (typical value)
  * GaAs is a high-mobility material (~5000 cm²/(V·s))
  * InSb has ultra-high mobility (~77,000 cm²/(V·s))
  * Graphene has extremely high mobility (~10,000-100,000 cm²/(V·s))

## 2.3 van der Pauw Hall Measurement Configuration

### 2.3.1 8-Contact van der Pauw Hall Configuration

**van der Pauw Hall measurement** is a standard technique that can measure the Hall effect on thin film samples of arbitrary shape. Combined with the sheet resistance measurement learned in Chapter 1, all of $\sigma$, $R_H$, $n$, and $\mu$ can be determined from a single sample.
    
    
    ```mermaid
    flowchart TD
        A[Sample Preparation8 contacts at 4 corners] --> B[Sheet Resistance MeasurementR_AB,CD, R_BC,DA]
        B --> C[Calculate Sheet Resistance R_svan der Pauw equation]
        C --> D[Hall MeasurementApply magnetic field B]
        D --> E[Measure Hall Voltage V_Hwith current I]
        E --> F[Calculate Hall Coefficient R_HR_H = V_H t / IB]
        F --> G[Carrier Density nn = 1/eR_H]
        F --> H[Mobility μμ = σR_H]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style F fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style G fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style H fill:#99ff99,stroke:#00cc00,stroke-width:2px
    ```

**Measurement Procedure** :

  1. **Sheet Resistance Measurement** (no magnetic field, B = 0): 
     * Current through contacts 1→2, voltage measurement between 3-4 → $R_{12,34}$
     * Current through contacts 2→3, voltage measurement between 4-1 → $R_{23,41}$
     * Calculate $R_s$ using van der Pauw equation
  2. **Hall Measurement** (apply magnetic field B, e.g., B = +0.5 T): 
     * Current $I$ through contacts 1→3, measure voltage $V_{24}^{+B}$ between 2-4
     * Reverse magnetic field (B = -0.5 T), measure $V_{24}^{-B}$ with same current
     * Hall voltage: $V_H = \frac{1}{2}(V_{24}^{+B} - V_{24}^{-B})$
     * Hall coefficient: $R_H = \frac{V_H t}{IB}$
  3. **Derivation of Electrical Properties** : 
     * Electrical conductivity: $\sigma = \frac{1}{R_s t}$
     * Carrier density: $n = \frac{1}{eR_H}$
     * Mobility: $\mu = \sigma R_H$

> **Important** : The reason for measuring with reversed magnetic field is to cancel offset voltage (due to thermoelectric effect and inhomogeneity). Hall voltage is an odd function with respect to magnetic field ($V_H(B) = -V_H(-B)$), but offset voltage is an even function, so taking the difference yields pure Hall voltage. 

#### Code Example 2-3: Simulation of van der Pauw Hall Measurement
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_sheet_resistance(R1, R2):
        """Calculate sheet resistance using van der Pauw equation"""
        def equation(Rs):
            return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
        R_initial = (R1 + R2) / 2 * np.pi / np.log(2)
        R_s = fsolve(equation, R_initial)[0]
        return R_s
    
    def complete_hall_analysis(R_AB_CD, R_BC_DA, V_24_pos_B, V_24_neg_B, I, B, t):
        """
        Complete van der Pauw Hall analysis
    
        Parameters
        ----------
        R_AB_CD, R_BC_DA : float
            van der Pauw resistance [Ω]
        V_24_pos_B, V_24_neg_B : float
            at positive and negative magnetic field forHall voltage [V]
        I : float
            Current [A]
        B : float
            Magnetic field magnitude [T]
        t : float
            Sample thickness [m]
    
        Returns
        -------
        results : dict
            Analysis results (R_s, sigma, R_H, n, mu)
        """
        e = 1.60218e-19  # [C]
    
        # 1. Sheet resistance
        R_s = van_der_pauw_sheet_resistance(R_AB_CD, R_BC_DA)
    
        # 2. Electrical conductivity
        sigma = 1 / (R_s * t)
    
        # 3. Hall voltage (offset removed)
        V_H = 0.5 * (V_24_pos_B - V_24_neg_B)
    
        # 4. Hall coefficient
        R_H = V_H * t / (I * B)
    
        # 5. Carrier density
        n = 1 / (np.abs(R_H) * e)
        carrier_type = 'electron' if R_H < 0 else 'hole'
    
        # 6. Mobility
        mu = sigma * np.abs(R_H)
    
        results = {
            'R_s': R_s,
            'sigma': sigma,
            'rho': 1 / sigma,
            'V_H': V_H,
            'R_H': R_H,
            'n': n,
            'carrier_type': carrier_type,
            'mu': mu
        }
    
        return results
    
    # Measurement data example: n-type silicon thin film
    R_AB_CD = 1000  # [Ω]
    R_BC_DA = 950   # [Ω]
    V_24_plus = -5.2e-3  # Voltage at +B [V]
    V_24_minus = +4.8e-3  # Voltage at -B [V]
    I = 100e-6  # Current [A] = 100 μA
    B = 0.5  # Magnetic field [T]
    t = 200e-9  # Thickness [m] = 200 nm
    
    results = complete_hall_analysis(R_AB_CD, R_BC_DA, V_24_plus, V_24_minus, I, B, t)
    
    print("van der Pauw Hall Measurement Analysis Results:")
    print("=" * 60)
    print(f"Measurement Conditions:")
    print(f"  R_AB,CD = {R_AB_CD:.1f} Ω")
    print(f"  R_BC,DA = {R_BC_DA:.1f} Ω")
    print(f"  V_24(+B) = {V_24_plus * 1e3:.2f} mV")
    print(f"  V_24(-B) = {V_24_minus * 1e3:.2f} mV")
    print(f"  Current I = {I * 1e6:.1f} μA")
    print(f"  Magnetic field B = ±{B:.2f} T")
    print(f"  Thickness t = {t * 1e9:.0f} nm")
    print("\nAnalysis Results:")
    print(f"  Sheet resistance R_s = {results['R_s']:.2f} Ω/sq")
    print(f"  Electrical conductivity σ = {results['sigma']:.2e} S/m")
    print(f"  Resistivity ρ = {results['rho']:.2e} Ω·m = {results['rho'] * 1e8:.2f} μΩ·cm")
    print(f"  Hall voltage V_H = {results['V_H'] * 1e3:.2f} mV")
    print(f"  Hall coefficient R_H = {results['R_H']:.2e} m³/C")
    print(f"  Carrier type: {results['carrier_type']}")
    print(f"  Carrier density n = {results['n']:.2e} m⁻³ = {results['n'] / 1e6:.2e} cm⁻³")
    print(f"  Mobility μ = {results['mu']:.2e} m²/(V·s) = {results['mu'] * 1e4:.1f} cm²/(V·s)")
    print("\nVerification:")
    print(f"  σ = neμ = {results['n'] * 1.60218e-19 * results['mu']:.2e} S/m")
    print(f"  (Matches calculated value σ = {results['sigma']:.2e} S/m)")
    

## 2.4 Multi-Carrier Analysis (Two-Band Model)

### 2.4.1 Theory of Two-Carrier Systems

In semiconductors, both electrons and holes may contribute to conduction (e.g., narrow bandgap semiconductors, InSb, HgCdTe, etc.). In this case, a simple one-band model is insufficient and a **two-band model** is required.

**Electrical Conductivity** (two-carrier):

$$ \sigma = n_e e \mu_e + n_h e \mu_h $$ 

**Hall Coefficient** (two-carrier):

$$ R_H = \frac{n_h \mu_h^2 - n_e \mu_e^2}{e(n_h \mu_h + n_e \mu_e)^2} $$ 

Here, $n_e$, $\mu_e$ are the electron carrier density and mobility, and $n_h$, $\mu_h$ are the hole carrier density and mobility.

**Physical Interpretation** :

  * $\mu_h \gg \mu_e$ case, holeがHall効果を支配（$R_H > 0$）
  * When $\mu_e \gg \mu_h$, electrons dominate the Hall effect ($R_H < 0$)
  * When mobilities are comparable, the sign of $R_H$ depends on the ratio of carrier densities $n_h/n_e$

#### Code Example 2-4: Fitting of Two-Band Model
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def two_band_conductivity(n_e, mu_e, n_h, mu_h):
        """Electrical conductivity of two-band model"""
        e = 1.60218e-19
        sigma = n_e * e * mu_e + n_h * e * mu_h
        return sigma
    
    def two_band_hall_coefficient(n_e, mu_e, n_h, mu_h):
        """Hall coefficient of two-band model"""
        e = 1.60218e-19
        numerator = n_h * mu_h**2 - n_e * mu_e**2
        denominator = (n_h * mu_h + n_e * mu_e)**2
        R_H = numerator / (e * denominator)
        return R_H
    
    # Simulation: InSb（electron・hole coexistence system)
    T_range = np.linspace(200, 400, 50)  # Temperature [K]
    
    # Temperature dependence (simplified)
    n_e = 1e22 * np.exp(-0.1 / (8.617e-5 * T_range))  # electrondensity [m^-3]
    n_h = 5e21 * np.exp(-0.08 / (8.617e-5 * T_range))  # holedensity [m^-3]
    mu_e = 7e4 * (300 / T_range)**1.5 * 1e-4  # electronMobility [m^2/(V·s)]
    mu_h = 1e3 * (300 / T_range)**2.5 * 1e-4  # holeMobility [m^2/(V·s)]
    
    sigma = two_band_conductivity(n_e, mu_e, n_h, mu_h)
    R_H = two_band_hall_coefficient(n_e, mu_e, n_h, mu_h)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Carrier density
    axes[0, 0].semilogy(T_range, n_e / 1e6, linewidth=2.5, label='Electron density n$_e$', color='#f5576c')
    axes[0, 0].semilogy(T_range, n_h / 1e6, linewidth=2.5, label='Hole density n$_h$', color='#ffa500')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Carrier Density [cm$^{-3}$]', fontsize=12)
    axes[0, 0].set_title('Carrier Densities (Two-Band Model)', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # Top right: Mobility
    axes[0, 1].loglog(T_range, mu_e * 1e4, linewidth=2.5, label='Electron mobility μ$_e$', color='#f5576c')
    axes[0, 1].loglog(T_range, mu_h * 1e4, linewidth=2.5, label='Hole mobility μ$_h$', color='#ffa500')
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Mobility [cm$^2$/(V·s)]', fontsize=12)
    axes[0, 1].set_title('Mobilities (Temperature Dependence)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3, which='both')
    
    # Bottom left: Electrical conductivity
    axes[1, 0].semilogy(T_range, sigma, linewidth=2.5, color='#f093fb')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Conductivity σ [S/m]', fontsize=12)
    axes[1, 0].set_title('Electrical Conductivity', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Bottom right: Hall coefficient
    axes[1, 1].plot(T_range, R_H, linewidth=2.5, color='#f093fb')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 1].set_ylabel('Hall Coefficient R$_H$ [m$^3$/C]', fontsize=12)
    axes[1, 1].set_title('Hall Coefficient (Sign Change)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detect sign reversal temperature of Hall coefficient
    sign_change_idx = np.where(np.diff(np.sign(R_H)))[0]
    if len(sign_change_idx) > 0:
        T_sign_change = T_range[sign_change_idx[0]]
        print(f"\nHall coefficient sign reversal temperature: {T_sign_change:.1f} K")
        print("  → Low temperature: R_H < 0（electrondominant）")
        print("  → High temperature: R_H > 0（holedominant）")
    

## 2.5 Temperature-Dependent Hall Measurement

### 2.5.1 Analysis of Carrier Scattering Mechanisms

Dominant carrier scattering mechanisms can be identified from temperature dependence of mobility:

Scattering Mechanism | Temperature Dependence of Mobility | Dominant Temperature Range | Material Examples  
---|---|---|---  
**Acoustic Phonon Scattering** | $\mu \propto T^{-3/2}$ | Above room temperature | Si, GaAs（High temperature）  
**Ionized Impurity Scattering** | $\mu \propto T^{3/2}$ | Low temperature (< 100 K) | Doped semiconductors  
**Optical Phonon Scattering** | Complex (temperature dependent) | High temperature | Polar semiconductors (GaAs)  
**Neutral Impurity Scattering** | $\mu \approx$ constant | Low temperature | Highly doped materials  
  
**Matthiessen's Rule** (mobility version):

$$ \frac{1}{\mu_{\text{total}}} = \frac{1}{\mu_{\text{phonon}}} + \frac{1}{\mu_{\text{impurity}}} + \frac{1}{\mu_{\text{other}}} $$ 

#### Code example2-5: Temperature-Dependent Hall Measurement analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    # Acoustic phonon scattering model
    def acoustic_phonon_mobility(T, mu0, T0=300):
        """μ ∝ T^(-3/2)"""
        return mu0 * (T0 / T)**(3/2)
    
    # Ionized impurity scattering model
    def ionized_impurity_mobility(T, mu1, T0=300):
        """μ ∝ T^(3/2)"""
        return mu1 * (T / T0)**(3/2)
    
    # Matthiessen's rule
    def combined_mobility(T, mu0, mu1, T0=300):
        """1/μ_total = 1/μ_phonon + 1/μ_impurity"""
        mu_phonon = acoustic_phonon_mobility(T, mu0, T0)
        mu_impurity = ionized_impurity_mobility(T, mu1, T0)
        mu_total = 1 / (1/mu_phonon + 1/mu_impurity)
        return mu_total
    
    # Generate simulation data
    T_range = np.linspace(50, 400, 30)  # [K]
    mu0_true = 8000  # Phonon scattering limited mobility (room temperature) [cm^2/(V·s)]
    mu1_true = 2000  # Impurity scattering limited mobility (room temperature) [cm^2/(V·s)]
    
    mu_data = combined_mobility(T_range, mu0_true, mu1_true)
    mu_data_noise = mu_data * (1 + 0.05 * np.random.randn(len(T_range)))  # 5% noise
    
    # Fitting
    model = Model(combined_mobility)
    params = model.make_params(mu0=5000, mu1=3000, T0=300)
    params['T0'].vary = False  # T0 is fixed
    
    result = model.fit(mu_data_noise, params, T=T_range)
    
    print("Temperature-Dependent Hall Measurement fitting results:")
    print(result.fit_report())
    
    # 各Scattering Mechanism contributionを計算
    mu_phonon_fit = acoustic_phonon_mobility(T_range, result.params['mu0'].value)
    mu_impurity_fit = ionized_impurity_mobility(T_range, result.params['mu1'].value)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Mobility vs temperature
    ax1.scatter(T_range, mu_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Measured data', color='#f093fb')
    ax1.plot(T_range, result.best_fit, linewidth=2.5, label='Fit (Matthiessen)', color='#f5576c')
    ax1.plot(T_range, mu_phonon_fit, linewidth=2, linestyle='--', label='Phonon scattering (T$^{-3/2}$)', color='#ffa500')
    ax1.plot(T_range, mu_impurity_fit, linewidth=2, linestyle=':', label='Impurity scattering (T$^{3/2}$)', color='#99ccff')
    ax1.set_xlabel('Temperature T [K]', fontsize=12)
    ax1.set_ylabel('Mobility μ [cm$^2$/(V·s)]', fontsize=12)
    ax1.set_title('Temperature-Dependent Hall Mobility', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Right: Scattering rate (1/μ) vs temperature
    ax2.scatter(T_range, 1/mu_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (1/μ)', color='#f093fb')
    ax2.plot(T_range, 1/mu_phonon_fit, linewidth=2.5, label='Phonon (1/μ$_{ph}$)', color='#ffa500')
    ax2.plot(T_range, 1/mu_impurity_fit, linewidth=2.5, label='Impurity (1/μ$_{imp}$)', color='#99ccff')
    ax2.plot(T_range, 1/result.best_fit, linewidth=2.5, label='Total (sum)', color='#f5576c', linestyle='--')
    ax2.set_xlabel('Temperature T [K]', fontsize=12)
    ax2.set_ylabel('Scattering Rate 1/μ [V·s/cm$^2$]', fontsize=12)
    ax2.set_title('Matthiessen Rule: Scattering Rate Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation of results
    print(f"\nScattering Mechanism analysis:")
    print(f"  Phonon scattering limited mobility (room temperature): {result.params['mu0'].value:.1f} cm²/(V·s)")
    print(f"  Impurity scattering limited mobility (room temperature): {result.params['mu1'].value:.1f} cm²/(V·s)")
    print(f"\nDominant scattering mechanisms:")
    print(f"  Low temperature (< 150 K): Impurity scattering（μ ∝ T^(3/2)）")
    print(f"  High temperature (> 250 K): Phonon scattering（μ ∝ T^(-3/2)）")
    

## 2.6 Complete Hall Data Processing Workflow

#### Code Example 2-6: Complete Analysis VH → n, μ
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    class HallDataProcessor:
        """Complete Hall data processing class"""
    
        def __init__(self, thickness):
            """
            Parameters
            ----------
            thickness : float
                Sample thickness [m]
            """
            self.t = thickness
            self.e = 1.60218e-19  # Elementary charge [C]
            self.data = {}
    
        def load_data(self, filename=None, mock_data=True):
            """
            Load measurement data
    
            Parameters
            ----------
            filename : str or None
                CSV filename（None caseはGenerate mock data）
            mock_data : bool
                Whether to generate mock data
            """
            if mock_data:
                # Generate mock data (temperature dependence)
                T = np.array([77, 100, 150, 200, 250, 300, 350, 400])  # [K]
                I = 100e-6  # Current [A]
                B = 0.5  # Magnetic field [T]
    
                # van der Pauw resistance (temperature dependent)
                R_AB_CD = 500 + 2.0 * T
                R_BC_DA = 480 + 1.8 * T
    
                # Hall voltage (temperature dependent, including carrier density change)
                V_pos = -3e-3 * (1 + 0.002 * (T - 300))
                V_neg = +2.9e-3 * (1 + 0.002 * (T - 300))
    
                self.data = pd.DataFrame({
                    'T': T,
                    'I': I,
                    'B': B,
                    'R_AB_CD': R_AB_CD,
                    'R_BC_DA': R_BC_DA,
                    'V_pos_B': V_pos,
                    'V_neg_B': V_neg
                })
            else:
                # Load actual data from CSV
                self.data = pd.read_csv(filename)
    
            return self.data
    
        def calculate_sheet_resistance(self):
            """Calculate sheet resistance"""
            from scipy.optimize import fsolve
    
            def vdp_eq(Rs, R1, R2):
                return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
            R_s_list = []
            for _, row in self.data.iterrows():
                R1, R2 = row['R_AB_CD'], row['R_BC_DA']
                R_initial = (R1 + R2) / 2 * np.pi / np.log(2)
                R_s = fsolve(vdp_eq, R_initial, args=(R1, R2))[0]
                R_s_list.append(R_s)
    
            self.data['R_s'] = R_s_list
            self.data['sigma'] = 1 / (np.array(R_s_list) * self.t)
            self.data['rho'] = 1 / self.data['sigma']
    
        def calculate_hall_properties(self):
            """Calculate Hall properties"""
            # Hall voltage (offset removed)
            self.data['V_H'] = 0.5 * (self.data['V_pos_B'] - self.data['V_neg_B'])
    
            # Hall coefficient
            self.data['R_H'] = (self.data['V_H'] * self.t) / (self.data['I'] * self.data['B'])
    
            # Carrier density
            self.data['n'] = 1 / (np.abs(self.data['R_H']) * self.e)
    
            # Mobility
            self.data['mu'] = self.data['sigma'] * np.abs(self.data['R_H'])
    
            # Carrier type
            self.data['carrier_type'] = ['electron' if rh < 0 else 'hole' for rh in self.data['R_H']]
    
        def plot_results(self):
            """Visualize results"""
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
            T = self.data['T']
    
            # Sheet resistance
            axes[0, 0].plot(T, self.data['R_s'], 'o-', linewidth=2.5, markersize=8, color='#f093fb')
            axes[0, 0].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 0].set_ylabel('Sheet Resistance [Ω/sq]', fontsize=11)
            axes[0, 0].set_title('Sheet Resistance', fontsize=12, fontweight='bold')
            axes[0, 0].grid(alpha=0.3)
    
            # Electrical conductivity
            axes[0, 1].semilogy(T, self.data['sigma'], 'o-', linewidth=2.5, markersize=8, color='#f5576c')
            axes[0, 1].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 1].set_ylabel('Conductivity [S/m]', fontsize=11)
            axes[0, 1].set_title('Electrical Conductivity', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
    
            # Hall voltage
            axes[0, 2].plot(T, self.data['V_H'] * 1e3, 'o-', linewidth=2.5, markersize=8, color='#ffa500')
            axes[0, 2].axhline(0, color='black', linestyle='--', linewidth=1.5)
            axes[0, 2].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 2].set_ylabel('Hall Voltage [mV]', fontsize=11)
            axes[0, 2].set_title('Hall Voltage', fontsize=12, fontweight='bold')
            axes[0, 2].grid(alpha=0.3)
    
            # Hall coefficient
            axes[1, 0].plot(T, self.data['R_H'], 'o-', linewidth=2.5, markersize=8, color='#99ccff')
            axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5)
            axes[1, 0].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 0].set_ylabel('Hall Coefficient [m³/C]', fontsize=11)
            axes[1, 0].set_title('Hall Coefficient', fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
    
            # Carrier density
            axes[1, 1].semilogy(T, self.data['n'] / 1e6, 'o-', linewidth=2.5, markersize=8, color='#99ff99')
            axes[1, 1].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 1].set_ylabel('Carrier Density [cm$^{-3}$]', fontsize=11)
            axes[1, 1].set_title('Carrier Density', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)
    
            # Mobility
            axes[1, 2].semilogy(T, self.data['mu'] * 1e4, 'o-', linewidth=2.5, markersize=8, color='#ff9999')
            axes[1, 2].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 2].set_ylabel('Mobility [cm$^2$/(V·s)]', fontsize=11)
            axes[1, 2].set_title('Hall Mobility', fontsize=12, fontweight='bold')
            axes[1, 2].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def save_results(self, filename='hall_results.csv'):
            """Save results to CSV"""
            self.data.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
    
    # Usage example
    processor = HallDataProcessor(thickness=200e-9)  # 200 nm
    processor.load_data(mock_data=True)
    processor.calculate_sheet_resistance()
    processor.calculate_hall_properties()
    
    print("Hall measurement data processing results:")
    print(processor.data[['T', 'R_s', 'sigma', 'n', 'mu']].to_string(index=False))
    
    processor.plot_results()
    processor.save_results('hall_analysis_output.csv')
    

#### Code Example 2-7: Uncertainty Analysis of Hall Measurement
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hall_measurement_uncertainty(V_H, delta_V_H, I, delta_I, B, delta_B, t, delta_t):
        """
        Uncertainty propagation for Hall measurement
    
        Parameters
        ----------
        V_H, delta_V_H : float
            Hall voltageand its uncertainty [V]
        I, delta_I : float
            currentand its uncertainty [A]
        B, delta_B : float
            magnetic fieldand its uncertainty [T]
        t, delta_t : float
            thicknessand its uncertainty [m]
    
        Returns
        -------
        R_H, delta_R_H : float
            Hall coefficientand its uncertainty
        n, delta_n : float
            Carrier densityand its uncertainty
        """
        e = 1.60218e-19
    
        # Hall coefficient: R_H = V_H * t / (I * B)
        R_H = V_H * t / (I * B)
    
        # 不確かさ伝播（partial derivatives）
        # δR_H/R_H = sqrt((δV_H/V_H)^2 + (δt/t)^2 + (δI/I)^2 + (δB/B)^2)
        rel_unc_V_H = delta_V_H / np.abs(V_H)
        rel_unc_t = delta_t / t
        rel_unc_I = delta_I / I
        rel_unc_B = delta_B / B
    
        rel_unc_R_H = np.sqrt(rel_unc_V_H**2 + rel_unc_t**2 + rel_unc_I**2 + rel_unc_B**2)
        delta_R_H = np.abs(R_H) * rel_unc_R_H
    
        # Carrier density: n = 1 / (e * R_H)
        n = 1 / (e * np.abs(R_H))
    
        # δn/n = δR_H/R_H
        delta_n = n * rel_unc_R_H
    
        return R_H, delta_R_H, n, delta_n, rel_unc_R_H
    
    # measurementexample
    V_H = -5.0e-3  # [V]
    delta_V_H = 0.1e-3  # voltage計 precision [V]
    I = 100e-6  # [A]
    delta_I = 0.5e-6  # current source precision [A]
    B = 0.5  # [T]
    delta_B = 0.01  # Magnetic field precision [T]
    t = 200e-9  # [m]
    delta_t = 5e-9  # thickness measurement precision [m]
    
    R_H, delta_R_H, n, delta_n, rel_unc = hall_measurement_uncertainty(
        V_H, delta_V_H, I, delta_I, B, delta_B, t, delta_t
    )
    
    print("Uncertainty analysis of Hall measurement:")
    print("=" * 60)
    print("Measured values:")
    print(f"  Hall voltage: ({V_H * 1e3:.2f} ± {delta_V_H * 1e3:.2f}) mV")
    print(f"  current: ({I * 1e6:.1f} ± {delta_I * 1e6:.2f}) μA")
    print(f"  magnetic field: ({B:.2f} ± {delta_B:.3f}) T")
    print(f"  thickness: ({t * 1e9:.1f} ± {delta_t * 1e9:.1f}) nm")
    print("\nResults:")
    print(f"  Hall coefficient: ({R_H:.3e} ± {delta_R_H:.3e}) m³/C")
    print(f"  Relative uncertainty: {rel_unc * 100:.2f}%")
    print(f"  Carrier density: ({n:.3e} ± {delta_n:.3e}) m⁻³")
    print(f"                = ({n / 1e6:.3e} ± {delta_n / 1e6:.3e}) cm⁻³")
    
    # Visualize uncertainty contributions
    contributions = {
        'V_H': (delta_V_H / np.abs(V_H))**2,
        't': (delta_t / t)**2,
        'I': (delta_I / I)**2,
        'B': (delta_B / B)**2
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(contributions.keys())
    values = [np.sqrt(v) * 100 for v in contributions.values()]
    
    bars = ax.bar(labels, values, color=['#f093fb', '#f5576c', '#ffa500', '#99ccff'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Relative Uncertainty Contribution [%]', fontsize=12)
    ax.set_title('Uncertainty Budget for Hall Measurement', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Display values above bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add total uncertainty
    ax.axhline(rel_unc * 100, color='red', linestyle='--', linewidth=2, label=f'Total: {rel_unc * 100:.2f}%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    

## 2.7 Exercises

### Exercise 2-1: Calculation of Hall Voltage (Easy)

Easy **Problem** :Carrier density $n = 1 \times 10^{22}$ m$^{-3}$, thickness $t = 100$ nm, current $I = 1$ mA, magnetic field $B = 0.5$ T , when, Hall voltage $V_H$ 。

**Show Solution**
    
    
    e = 1.60218e-19  # [C]
    n = 1e22  # [m^-3]
    t = 100e-9  # [m]
    I = 1e-3  # [A]
    B = 0.5  # [T]
    
    V_H = I * B / (n * e * t)
    print(f"Hall voltage V_H = {V_H:.3e} V = {V_H * 1e3:.2f} mV")
    

**Solution** :V$_H$ = 3.12 × 10$^{-3}$ V = 3.12 mV

### Exercise 2-2: Calculation of Carrier Density (Easy)

Easy **Problem** :Hall coefficient $R_H = -1.5 \times 10^{-3}$ m$^3$/C was measured。Carrier typeとCarrier density。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: EasyProblem:Hall coefficient $R_H = -1.5 \times 10^{-3}$ m$^
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    e = 1.60218e-19  # [C]
    R_H = -1.5e-3  # [m^3/C]
    
    carrier_type = 'electron' if R_H < 0 else 'hole'
    n = 1 / (np.abs(R_H) * e)
    
    print(f"Carrier type: {carrier_type}")
    print(f"Carrier density n = {n:.3e} m⁻³ = {n / 1e6:.3e} cm⁻³")
    

**Solution** :electron（n-type）, n = 4.16 × 10$^{21}$ m$^{-3}$ = 4.16 × 10$^{15}$ cm$^{-3}$

### Exercise 2-3: Calculation of Mobility (Easy)

Easy **Problem** :Electrical conductivity $\sigma = 1 \times 10^4$ S/m, Hall coefficient $R_H = -2 \times 10^{-3}$ m$^3$/C , when, Mobility $\mu$ 。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: EasyProblem:Electrical conductivity $\sigma = 1 \times 10^4$
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    sigma = 1e4  # [S/m]
    R_H = -2e-3  # [m^3/C]
    
    mu = sigma * np.abs(R_H)
    print(f"Mobility μ = {mu:.2f} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**Solution** :μ = 20 m$^2$/(V·s) = 200,000 cm$^2$/(V·s) (unrealistically high → measurement review needed)

### Exercise 2-4: Analysis of van der Pauw Hall Configuration (Medium)

Medium **Problem** :van der Pauwmeasurementで, $R_{\text{AB,CD}} = 950$ Ω, $R_{\text{BC,DA}} = 1050$ Ω, $V_{24}^{+B} = -4.5$ mV, $V_{24}^{-B} = +4.3$ mV, $I = 100$ μA, $B = 0.5$ T, $t = 300$ nm was obtained。$\sigma$, $R_H$, $n$, $\mu$ 。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MediumProblem:van der Pauwmeasurementで, $R_{\text{AB,CD}} = 
    
    Purpose: Demonstrate optimization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_Rs(R1, R2):
        def eq(Rs):
            return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
        R_init = (R1 + R2) / 2 * np.pi / np.log(2)
        return fsolve(eq, R_init)[0]
    
    # Parameters
    R1, R2 = 950, 1050  # [Ω]
    V_pos, V_neg = -4.5e-3, 4.3e-3  # [V]
    I = 100e-6  # [A]
    B = 0.5  # [T]
    t = 300e-9  # [m]
    e = 1.60218e-19  # [C]
    
    # Sheet resistance
    R_s = van_der_pauw_Rs(R1, R2)
    sigma = 1 / (R_s * t)
    
    # Hall analysis
    V_H = 0.5 * (V_pos - V_neg)
    R_H = V_H * t / (I * B)
    n = 1 / (np.abs(R_H) * e)
    mu = sigma * np.abs(R_H)
    
    print(f"Sheet resistance R_s = {R_s:.2f} Ω/sq")
    print(f"Electrical conductivity σ = {sigma:.2e} S/m")
    print(f"Hall coefficient R_H = {R_H:.3e} m³/C")
    print(f"Carrier density n = {n:.2e} m⁻³ = {n / 1e6:.2e} cm⁻³")
    print(f"Mobility μ = {mu:.2e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**Solution** :R$_s$ ≈ 1370 Ω/sq, σ ≈ 2.43 × 10$^3$ S/m, R$_H$ ≈ -2.64 × 10$^{-2}$ m$^3$/C, n ≈ 2.36 × 10$^{20}$ m$^{-3}$, μ ≈ 0.064 m$^2$/(V·s) = 640 cm$^2$/(V·s)

### Exercise 2-5: Analysis of Two-Band Model (Medium)

Medium **Problem** :$n_e = 1 \times 10^{22}$ m$^{-3}$, $\mu_e = 0.5$ m$^2$/(V·s), $n_h = 5 \times 10^{21}$ m$^{-3}$, $\mu_h = 0.05$ m$^2$/(V·s) , when, Electrical conductivity $\sigma$ とHall Coefficient $R_H$ 。

**Show Solution**
    
    
    e = 1.60218e-19  # [C]
    n_e = 1e22  # [m^-3]
    mu_e = 0.5  # [m^2/(V·s)]
    n_h = 5e21  # [m^-3]
    mu_h = 0.05  # [m^2/(V·s)]
    
    # Electrical conductivity
    sigma = n_e * e * mu_e + n_h * e * mu_h
    print(f"Electrical conductivity σ = {sigma:.2e} S/m")
    
    # Hall coefficient
    numerator = n_h * mu_h**2 - n_e * mu_e**2
    denominator = (n_h * mu_h + n_e * mu_e)**2
    R_H = numerator / (e * denominator)
    print(f"Hall coefficient R_H = {R_H:.3e} m³/C")
    
    # 見かけのCarrier density
    n_apparent = 1 / (abs(R_H) * e)
    print(f"見かけのCarrier density: {n_apparent:.2e} m⁻³ = {n_apparent / 1e6:.2e} cm⁻³")
    print(f"  （trueelectrondensity {n_e / 1e6:.2e} cm⁻³different from）")
    

**Solution** :σ ≈ 8.41 × 10$^2$ S/m, R$_H$ ≈ -1.37 × 10$^{-3}$ m$^3$/C, apparent n ≈ 4.56 × 10$^{21}$ m$^{-3}$(different from true value)

### Exercise 2-6: Temperature Dependence Fitting (Medium)

Medium **Problem** :Mobilityが T = 100 K で 5000 cm$^2$/(V·s), 300 K で 1500 cm$^2$/(V·s) であった。$\mu \propto T^{-\alpha}$ モデルで指数 $\alpha$ 。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MediumProblem:Mobilityが T = 100 K で 5000 cm$^2$/(V·s), 300 K
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    T1, mu1 = 100, 5000  # [K], [cm^2/(V·s)]
    T2, mu2 = 300, 1500  # [K], [cm^2/(V·s)]
    
    # μ ∝ T^(-α) より, log(μ) = -α log(T) + const
    # log(mu1/mu2) = -α log(T1/T2)
    # α = -log(mu1/mu2) / log(T1/T2)
    
    alpha = -np.log(mu1 / mu2) / np.log(T1 / T2)
    print(f"指数 α = {alpha:.2f}")
    print(f"モデル: μ ∝ T^(-{alpha:.2f})")
    print(f"\nInterpretation: α ≈ 1.5 → Acoustic phonon scattering is dominant")
    

**Solution** :α ≈ 1.10（理論値 3/2 に近いが, 複数のScattering Mechanismが寄与している可能性）

### Exercise 2-7: Analysis of Magnetic Field Dependence (Hard)

Hard **Problem** :Hall voltageがmagnetic field B = 0, 0.2, 0.4, 0.6, 0.8, 1.0 T で V$_H$ = 0, 1.0, 2.1, 3.0, 4.1, 5.0 mV とmeasurementされた（current I = 100 μA, thickness t = 200 nm）。線形Fittingで Hall coefficientを求め, 非線形性を評価せよ。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: HardProblem:Hall voltageがmagnetic field B = 0, 0.2, 0.4, 0.6
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    B = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # [T]
    V_H = np.array([0, 1.0, 2.1, 3.0, 4.1, 5.0]) * 1e-3  # [V]
    I = 100e-6  # [A]
    t = 200e-9  # [m]
    
    # 線形Fitting
    slope, intercept, r_value, _, std_err = linregress(B, V_H)
    
    # Hall coefficient
    R_H = slope * t / I
    print(f"線形Fitting: V_H = {slope * 1e3:.3f} mV/T × B + {intercept * 1e3:.3f} mV")
    print(f"Hall coefficient R_H = {R_H:.3e} m³/C")
    print(f"決定係数 R² = {r_value**2:.4f}")
    
    # 非線形性評価
    V_H_fit = slope * B + intercept
    residuals = V_H - V_H_fit
    rel_residuals = residuals / V_H_fit[1:] * 100  # 最初の点（B=0）を除く
    
    print(f"\n非線形性評価:")
    print(f"  最大残差: {np.max(np.abs(residuals[1:])) * 1e6:.2f} μV")
    print(f"  平均相対残差: {np.mean(np.abs(rel_residuals)):.2f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(B, V_H * 1e3, s=100, edgecolors='black', linewidths=2, label='Measured', color='#f093fb', zorder=5)
    ax1.plot(B, V_H_fit * 1e3, linewidth=2.5, label=f'Fit: {slope * 1e3:.3f} mV/T', color='#f5576c', linestyle='--')
    ax1.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax1.set_ylabel('Hall Voltage V$_H$ [mV]', fontsize=12)
    ax1.set_title('Hall Voltage vs Magnetic Field', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.scatter(B[1:], residuals[1:] * 1e6, s=100, edgecolors='black', linewidths=2, color='#ffa500')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax2.set_ylabel('Residuals [μV]', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Solution** :R$_H$ ≈ 1.00 × 10$^{-2}$ m$^3$/C, R$^2$ ≈ 0.9996(good linearity), 最大残差 < 50 μV（measurement precision内）

### Exercise 2-8: Uncertainty Propagation (Hard)

Hard **Problem** :Hall voltage $V_H = (5.0 \pm 0.2)$ mV, current $I = (100 \pm 1)$ μA, magnetic field $B = (0.50 \pm 0.02)$ T, thickness $t = (200 \pm 10)$ nm , when, Carrier density $n$ の不確かさ $\Delta n$ 。どのParametersが最も影響するか評価せよ。

**Show Solution**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: HardProblem:Hall voltage $V_H = (5.0 \pm 0.2)$ mV, current $
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    V_H, dV_H = 5.0e-3, 0.2e-3  # [V]
    I, dI = 100e-6, 1e-6  # [A]
    B, dB = 0.50, 0.02  # [T]
    t, dt = 200e-9, 10e-9  # [m]
    e = 1.60218e-19  # [C]
    
    # n = 1 / (e * R_H) = I * B / (e * V_H * t)
    n = I * B / (e * V_H * t)
    
    # Relative uncertainty contribution
    rel_V_H = (dV_H / V_H)**2
    rel_I = (dI / I)**2
    rel_B = (dB / B)**2
    rel_t = (dt / t)**2
    
    rel_unc_total = np.sqrt(rel_V_H + rel_I + rel_B + rel_t)
    dn = n * rel_unc_total
    
    print(f"Carrier density: n = {n:.3e} m⁻³ = {n / 1e6:.3e} cm⁻³")
    print(f"不確かさ: Δn = {dn:.3e} m⁻³ = {dn / 1e6:.3e} cm⁻³")
    print(f"相対不確かさ: {rel_unc_total * 100:.2f}%")
    print(f"\nUncertainty contributions:")
    print(f"  V_H: {np.sqrt(rel_V_H) * 100:.2f}%")
    print(f"  I:   {np.sqrt(rel_I) * 100:.2f}%")
    print(f"  B:   {np.sqrt(rel_B) * 100:.2f}%")
    print(f"  t:   {np.sqrt(rel_t) * 100:.2f}%")
    print(f"\nConclusion: Uncertainty in thickness t has the largest impact ({np.sqrt(rel_t) * 100:.1f}%)")
    

**Solution** :n = (6.24 ± 0.42) × 10$^{21}$ m$^{-3}$, 相対不確かさ 6.7%, thicknessmeasurementが最も重要（5.0%寄与）

### Exercise 2-9: Experimental Planning (Hard)

Hard **Problem** :未知の薄膜半導体（thickness 100 nm）のCarrier type, density, Mobilityを決定する実験計画を立案せよ。必要なmeasurement, temperaturerange, magnetic fieldrange, データ解析手法を具体的に説明せよ。

**Show Solution**

**Experimental Plan** :

  1. **Sample Preparation** : 
     * van der Pauw配置:四隅に8接点（current用4つ, voltage用4つ）
     * Contact size < 0.5 mm, Verify ohmic contact（I-V characteristic is linear）
  2. **Room Temperature Measurement** (T = 300 K): 
     * **Sheet resistancemeasurement** （B = 0）:$R_{\text{AB,CD}}$, $R_{\text{BC,DA}}$ → $R_s$, $\sigma$ 計算
     * **Hallmeasurement** :B = ±0.5 T で $V_H$ measurement（オフセット除去）
     * → Carrier type（$R_H$ の符号）, Carrier density $n$, Mobility $\mu$ を決定
  3. **Magnetic Field Dependence Measurement** (room temperature): 
     * B = 0 → 1 T to 0.1 T step Hall voltagemeasurement
     * Verify linearity（非線形 → 多キャリア系possibility）
  4. **Temperature Dependence Measurement** : 
     * temperaturerange:77 K（liquid nitrogen）〜 400 K
     * measurement点:20-25 K interval, 各temperatureで熱平衡待機（10-15minutes）
     * 各temperatureで:Sheet resistance + Hallmeasurement（B = ±0.5 T）
  5. **Data Analysis** : 
     * $n(T)$, $\mu(T)$ のPlot作成
     * 半導体or metal determination（$\rho$ のtemperature依存性）
     * 半導体 case:ArrheniusPlot（$\ln n$ vs $1/T$）→ activation energy
     * Temperature Dependence of MobilityFitting（Matthiessen's rule, $\mu \propto T^{-\alpha}$）
     * Scattering Mechanismidentification（音響フォノン, 不純物, grain boundaries, etc.）

**Expected Results** :

  * **n-type semiconductor** :$R_H < 0$, $n \sim 10^{15}$-10$^{18}$ cm$^{-3}$, $\mu \sim 100$-5000 cm$^2$/(V·s), temperature上昇で $n$ increases（thermal excitation）
  * **p-type semiconductor** :$R_H > 0$, 同様のdensity・Mobilityrange
  * **Metallic material** :$n \sim 10^{21}$-10$^{23}$ cm$^{-3}$, $\mu$ はtemperature上昇で減少（フォノン散乱）

## 2.8 Learning Check

Check your understanding with the following checklist:

### Basic Understanding

  * Can derive Hall voltage equation from Lorentz force
  * Understand the relationship between Hall coefficient $R_H = 1/(ne)$ and carrier density
  * Can explain the physical meaning of mobility $\mu = \sigma R_H$
  * Understand the measurement procedure of van der Pauw Hall configuration
  * Can explain the purpose of reversed magnetic field measurement (offset removal)

### Practical Skills

  * Hall voltageからCalculate carrier densityできる
  * Can completely analyze van der Pauw Hall measurement data ($\sigma$, $R_H$, $n$, $\mu$)
  * PythonでComplete Hall Data Processing Workflowcan implement
  * Can calculate uncertainty propagation and evaluate measurement accuracy
  * temperature依存性データからScattering Mechanismcan estimate

### Application Skills

  * Can analyze multi-carrier systems with two-band model
  * Temperature Dependence of MobilityからScattering Mechanismcan identify
  * Can evaluate measurement nonlinearity and estimate causes
  * Can plan experiments and determine appropriate measurement conditions

## 2.9 References

  1. Hall, E. H. (1879). _On a New Action of the Magnet on Electric Currents_. American Journal of Mathematics, 2(3), 287-292. - Hall effect discovery paper
  2. van der Pauw, L. J. (1958). _A method of measuring the resistivity and Hall coefficient on lamellae of arbitrary shape_. Philips Technical Review, 20(8), 220-224. - van der Pauw Hallmeasurement法の原論文
  3. Look, D. C. (1989). _Electrical Characterization of GaAs Materials and Devices_. Wiley. - 半導体Hallmeasurementの実践的テキスト
  4. Putley, E. H. (1960). _The Hall Effect and Related Phenomena_. Butterworths. - Comprehensive explanation of Hall effect
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_ (Chapter 2: The Sommerfeld Theory of Metals). Holt, Rinehart and Winston. - Drude model and carrier transport theory
  6. Schroder, D. K. (2006). _Semiconductor Material and Device Characterization_ (3rd ed., Chapter 2: Resistivity). Wiley-Interscience. - Hallmeasurement技術の詳細
  7. Popović, R. S. (2004). _Hall Effect Devices_ (2nd ed.). Institute of Physics Publishing. - Hall効果デバイスとmeasurement技術

## 2.10 To Next Chapter

In the next chapter, you will learn the principles and practice of **magnetic measurements**. You will master magnetization measurements using VSM (Vibrating Sample Magnetometer) and SQUID (Superconducting Quantum Interference Device), M-H curve analysis, magnetic anisotropy evaluation, and integrated measurement techniques using PPMS (Physical Property Measurement System).
