---
title: "Chapter 3: Grand Canonical Ensemble and Chemical Potential"
chapter_title: "Chapter 3: Grand Canonical Ensemble and Chemical Potential"
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/classical-statistical-mechanics/chapter-3.html>) | Last sync: 2025-11-16

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Classical Statistical Mechanics](<index.html>) > Chapter 3 

## 🎯 Learning Objectives

  * Understand the concept of the grand canonical ensemble and situations where it applies
  * Learn the relationship between the grand partition function and chemical potential
  * Master methods for calculating particle number fluctuations
  * Derive the Langmuir adsorption isotherm and learn applications to materials
  * Understand the correspondence between the lattice gas model and Ising model
  * Learn the fundamentals of quantum statistics (Fermi-Dirac distribution, Bose-Einstein distribution)
  * Understand applications to chemical reaction equilibrium
  * Simulate adsorption phenomena in materials science

## 📖 What is the Grand Canonical Ensemble

### Grand Canonical Ensemble

A system with fixed temperature \\(T\\) and chemical potential \\(\mu\\) that can exchange particles and energy with a particle reservoir. In this ensemble, volume \\(V\\), temperature \\(T\\), and chemical potential \\(\mu\\) are fixed, while particle number \\(N\\) and energy \\(E\\) fluctuate. This ensemble is particularly important for describing adsorption, chemical reactions, and open systems.

**Grand partition function** :

\\[ \Xi(\mu, V, T) = \sum_{N=0}^\infty \sum_i e^{\beta(\mu N - E_i)} = \sum_{N=0}^\infty e^{\beta\mu N} Z(N, V, T) \\]

where \\(\beta = 1/(k_B T)\\) and \\(Z(N, V, T)\\) is the canonical partition function.

### Grand Potential and Chemical Potential

Grand potential:

\\[ \Omega = -k_B T \ln \Xi = F - \mu N \\]

Derivation of thermodynamic quantities: The **average particle number** is given by \\(\langle N \rangle = -\frac{\partial \Omega}{\partial \mu} = \frac{1}{\beta}\frac{\partial \ln \Xi}{\partial \mu}\\), the **pressure** by \\(P = -\frac{\partial \Omega}{\partial V}\\), and the **particle number fluctuation** by \\(\langle (\Delta N)^2 \rangle = k_B T \left(\frac{\partial \langle N \rangle}{\partial \mu}\right)_{V,T}\\).

The **chemical potential** \\(\mu\\) is "the free energy required to add one particle".

## 💻 Example 3.1: Grand Partition Function of an Ideal Gas

### Grand Canonical Calculation for an Ideal Gas

Using the single-particle partition function \\(z_1 = V/\lambda_T^3\\) (where \\(\lambda_T\\) is the thermal de Broglie wavelength):

\\[ \Xi = \sum_{N=0}^\infty \frac{(z_1 e^{\beta\mu})^N}{N!} = \exp(z_1 e^{\beta\mu}) \\]

Average particle number:

\\[ \langle N \rangle = z_1 e^{\beta\mu} \\]

Solving for the chemical potential:

\\[ \mu = k_B T \ln\left(\frac{\langle N \rangle \lambda_T^3}{V}\right) \\]

Python Implementation: Chemical Potential of an Ideal Gas
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    k_B = 1.380649e-23  # J/K
    h = 6.62607015e-34  # J·s
    m_Ar = 6.63e-26  # Mass of Ar atom (kg)
    
    def thermal_wavelength(T, m, h, k_B):
        """Thermal de Broglie wavelength"""
        return h / np.sqrt(2 * np.pi * m * k_B * T)
    
    def chemical_potential_ideal_gas(N, V, T, m, h, k_B):
        """Chemical potential of ideal gas"""
        lambda_T = thermal_wavelength(T, m, h, k_B)
        return k_B * T * np.log((N / V) * lambda_T**3)
    
    def fugacity(mu, T, k_B):
        """Fugacity z = exp(βμ)"""
        return np.exp(mu / (k_B * T))
    
    # Ar gas near standard conditions
    N_A = 6.022e23
    N_molar = N_A
    V_molar = 0.0224  # m³
    
    T_range = np.linspace(100, 1000, 100)
    
    # Density dependence (fixed temperature)
    T_fixed = 300  # K
    V_range = np.linspace(0.001, 0.1, 100)  # m³
    n_range = N_molar / V_range  # Number density
    
    mu_vs_V = [chemical_potential_ideal_gas(N_molar, V, T_fixed, m_Ar, h, k_B)
               for V in V_range]
    
    # Temperature dependence (fixed volume)
    mu_vs_T = [chemical_potential_ideal_gas(N_molar, V_molar, T, m_Ar, h, k_B)
               for T in T_range]
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chemical potential density dependence
    ax1 = axes[0, 0]
    ax1.plot(n_range / 1e25, np.array(mu_vs_V) / (k_B * T_fixed), 'b-', linewidth=2)
    ax1.set_xlabel('Number density (10²⁵ m⁻³)')
    ax1.set_ylabel('μ / (k_B T)')
    ax1.set_title(f'Chemical Potential Density Dependence (T = {T_fixed} K)')
    ax1.grid(True, alpha=0.3)
    
    # Chemical potential temperature dependence
    ax2 = axes[0, 1]
    ax2.plot(T_range, np.array(mu_vs_T) / 1e-21, 'r-', linewidth=2)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('μ (10⁻²¹ J)')
    ax2.set_title('Chemical Potential Temperature Dependence (V = 22.4 L)')
    ax2.grid(True, alpha=0.3)
    
    # Fugacity
    ax3 = axes[1, 0]
    z_vals = [fugacity(mu, T_fixed, k_B) for mu in mu_vs_V]
    ax3.semilogy(n_range / 1e25, z_vals, 'g-', linewidth=2)
    ax3.set_xlabel('Number density (10²⁵ m⁻³)')
    ax3.set_ylabel('Fugacity z = exp(βμ)')
    ax3.set_title(f'Fugacity (T = {T_fixed} K)')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Particle number fluctuation
    ax4 = axes[1, 1]
    # <(ΔN)²> = k_B T (∂<n>/∂μ) = <n> (ideal gas)
    fluctuation_ratio = 1 / np.sqrt(n_range * V_molar)  # σ_N / <n> = 1/√<n>
    ax4.loglog(n_range / 1e25, fluctuation_ratio, 'm-', linewidth=2)
    ax4.set_xlabel('Number density (10²⁵ m⁻³)')
    ax4.set_ylabel('σ_N / <n>')
    ax4.set_title('Relative Fluctuation')
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('stat_mech_grand_canonical_ideal_gas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numerical results
    print("=== Grand Canonical Statistics of Ideal Gas ===\n")
    print(f"Ar gas (1 mol, near standard conditions)\n")
    
    T_std = 298.15  # K
    V_std = V_molar
    mu_std = chemical_potential_ideal_gas(N_molar, V_std, T_std, m_Ar, h, k_B)
    lambda_T_std = thermal_wavelength(T_std, m_Ar, h, k_B)
    
    print(f"T = {T_std} K, V = {V_std*1000:.1f} L")
    print(f"Thermal de Broglie wavelength: λ_T = {lambda_T_std*1e9:.4f} nm")
    print(f"Chemical potential: μ = {mu_std:.6e} J")
    print(f"                  μ/(k_B T) = {mu_std/(k_B*T_std):.4f}")
    print(f"Fugacity: z = {fugacity(mu_std, T_std, k_B):.6e}")
    print(f"\nRelative fluctuation: σ_N/<n> = 1/√<n> = {1/np.sqrt(N_molar):.6e}")
    print(f"            = {1/np.sqrt(N_molar):.2e} (extremely small)")
    </n></n></n></n></n></n></n>

## 💻 Example 3.2: Langmuir Adsorption Isotherm

### Langmuir Adsorption Model

There are \\(M\\) adsorption sites, each able to adsorb at most one particle:

Grand partition function (per site):

\\[ \xi = 1 + e^{\beta(\mu - \varepsilon)} \\]

where \\(\varepsilon\\) is the adsorption energy (negative value).

Coverage \\(\theta\\):

\\[ \theta = \frac{\langle N \rangle}{M} = \frac{e^{\beta(\mu - \varepsilon)}}{1 + e^{\beta(\mu - \varepsilon)}} = \frac{1}{1 + e^{-\beta(\mu - \varepsilon)}} \\]

Relationship with gas phase pressure \\(P\\) (where \\(\mu = k_B T \ln(P/P_0)\\)):

\\[ \theta = \frac{KP}{1 + KP}, \quad K = e^{-\beta\varepsilon} / P_0 \\]

This is the **Langmuir adsorption isotherm**.

Python Implementation: Langmuir Adsorption Isotherm
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    k_B = 1.380649e-23  # J/K
    
    def langmuir_coverage(P, K):
        """Langmuir adsorption isotherm"""
        return K * P / (1 + K * P)
    
    def freundlich_isotherm(P, k_F, n):
        """Freundlich adsorption isotherm (empirical formula, multilayer adsorption)"""
        return k_F * P**(1/n)
    
    def bet_isotherm(P, P0, c):
        """BET adsorption isotherm (multilayer adsorption)"""
        x = P / P0
        return (c * x) / ((1 - x) * (1 + (c - 1) * x))
    
    # Different adsorption energies
    T = 300  # K
    epsilon_values = [-0.1, -0.3, -0.5, -0.8]  # eV
    epsilon_J = [eps * 1.602e-19 for eps in epsilon_values]  # J
    P0 = 1e5  # Pa (standard pressure)
    
    K_values = [np.exp(-eps / (k_B * T)) / P0 for eps in epsilon_J]
    
    P_range = np.logspace(-2, 6, 200)  # Pa
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Langmuir isotherm (linear plot)
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for epsilon, K, color in zip(epsilon_values, K_values, colors):
        theta = langmuir_coverage(P_range, K)
        ax1.plot(P_range / 1e3, theta, color=color, linewidth=2,
                 label=f'ε = {epsilon} eV')
    
    ax1.set_xlabel('Pressure (kPa)')
    ax1.set_ylabel('Coverage θ')
    ax1.set_title('Langmuir Adsorption Isotherm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    
    # Langmuir isotherm (log plot)
    ax2 = axes[0, 1]
    for epsilon, K, color in zip(epsilon_values, K_values, colors):
        theta = langmuir_coverage(P_range, K)
        ax2.semilogx(P_range, theta, color=color, linewidth=2,
                     label=f'ε = {epsilon} eV')
    
    ax2.set_xlabel('Pressure (Pa)')
    ax2.set_ylabel('Coverage θ')
    ax2.set_title('Langmuir Isotherm (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Langmuir plot (P/θ vs P)
    ax3 = axes[1, 0]
    K_demo = K_values[2]
    theta_demo = langmuir_coverage(P_range, K_demo)
    # P/θ = 1/K + P (becomes linear)
    P_over_theta = P_range / theta_demo
    
    ax3.plot(P_range / 1e3, P_over_theta / 1e3, 'b-', linewidth=2)
    ax3.set_xlabel('Pressure (kPa)')
    ax3.set_ylabel('P/θ (kPa)')
    ax3.set_title('Langmuir Plot (Linearity Check)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 100])
    
    # Temperature dependence
    ax4 = axes[1, 1]
    temperatures = [250, 300, 350, 400]
    epsilon_fixed = epsilon_J[2]
    P_fixed_range = np.logspace(2, 5, 100)
    
    for T_val, color in zip(temperatures, colors):
        K_T = np.exp(-epsilon_fixed / (k_B * T_val)) / P0
        theta_T = langmuir_coverage(P_fixed_range, K_T)
        ax4.semilogx(P_fixed_range, theta_T, color=color, linewidth=2,
                     label=f'T = {T_val} K')
    
    ax4.set_xlabel('Pressure (Pa)')
    ax4.set_ylabel('Coverage θ')
    ax4.set_title(f'Temperature Dependence (ε = {epsilon_values[2]} eV)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_langmuir_adsorption.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numerical results
    print("=== Langmuir Adsorption Isotherm ===\n")
    print(f"Temperature T = {T} K\n")
    
    for epsilon, K in zip(epsilon_values, K_values):
        print(f"Adsorption energy ε = {epsilon} eV:")
        print(f"  Adsorption equilibrium constant K = {K:.6e} Pa⁻¹")
    
        # Half-coverage pressure (θ = 0.5)
        P_half = 1 / K
        print(f"  Half-coverage pressure P₁/₂ = {P_half:.2e} Pa = {P_half/1e3:.2f} kPa\n")
    
    print("Characteristics of Langmuir isotherm:")
    print("  - Low pressure: θ ≈ KP (Henry's law)")
    print("  - High pressure: θ → 1 (saturation)")
    print("  - Assumes monolayer adsorption")
    

## 💻 Example 3.3: Lattice Gas Model

### Lattice Gas Model

There are \\(M\\) lattice sites, and each site has two states: occupied (particle present) or empty (no particle):

Energy: \\(E = -\varepsilon_0 N - \frac{1}{2}J \sum_{\langle i,j \rangle} n_i n_j\\)

where \\(n_i = 0, 1\\) is the occupation number and \\(J\\) is the nearest-neighbor interaction.

**Correspondence with Ising model** : Substituting \\(s_i = 2n_i - 1\\) reduces to the Ising model of a spin system.

In mean field approximation, with \\(\langle n \rangle = \theta\\) (coverage):

\\[ \theta = \frac{1}{1 + e^{-\beta(\mu - \varepsilon_0 + Jz\theta)}} \\]

where \\(z\\) is the coordination number.

Python Implementation: Mean Field Theory of Lattice Gas
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    k_B = 1.380649e-23  # J/K
    
    def mean_field_equation(theta, mu, epsilon_0, J, z, T, k_B):
        """Mean field equation"""
        beta = 1 / (k_B * T)
        effective_mu = mu - epsilon_0 + J * z * theta
        return theta - 1 / (1 + np.exp(-beta * effective_mu))
    
    def solve_mean_field_coverage(mu, epsilon_0, J, z, T, k_B, theta_init=0.5):
        """Solve mean field equation to obtain coverage"""
        sol = fsolve(mean_field_equation, theta_init,
                      args=(mu, epsilon_0, J, z, T, k_B))
        return sol[0]
    
    # Parameters
    epsilon_0 = -0.5 * 1.602e-19  # Adsorption energy (J)
    z = 4  # Coordination number of 2D square lattice
    T = 300  # K
    
    # Cases with different interaction strengths
    J_values = [0, -0.1e-19, -0.2e-19, -0.3e-19]  # J (attractive)
    J_eV = [J / 1.602e-19 for J in J_values]
    colors = ['blue', 'green', 'orange', 'red']
    
    # Chemical potential range
    mu_range = np.linspace(-1.0e-19, 0.5e-19, 200)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coverage vs chemical potential
    ax1 = axes[0, 0]
    for J, J_ev, color in zip(J_values, J_eV, colors):
        theta_list = []
        for mu in mu_range:
            theta = solve_mean_field_coverage(mu, epsilon_0, J, z, T, k_B)
            theta_list.append(theta)
    
        ax1.plot(mu_range / 1.602e-19, theta_list, color=color, linewidth=2,
                 label=f'J = {J_ev:.2f} eV')
    
    ax1.set_xlabel('Chemical potential μ (eV)')
    ax1.set_ylabel('Coverage θ')
    ax1.set_title('Coverage (Mean Field Theory)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature dependence
    ax2 = axes[0, 1]
    temperatures = [200, 300, 400, 500]
    J_fixed = J_values[2]
    mu_fixed = -0.3e-19  # J
    
    for T_val, color in zip(temperatures, colors):
        theta_T_list = []
        for mu in mu_range:
            theta = solve_mean_field_coverage(mu, epsilon_0, J_fixed, z, T_val, k_B)
            theta_T_list.append(theta)
    
        ax2.plot(mu_range / 1.602e-19, theta_T_list, color=color, linewidth=2,
                 label=f'T = {T_val} K')
    
    ax2.set_xlabel('Chemical potential μ (eV)')
    ax2.set_ylabel('Coverage θ')
    ax2.set_title(f'Temperature Dependence (J = {J_fixed/1.602e-19:.2f} eV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase transition due to interaction (low temperature)
    ax3 = axes[1, 0]
    T_low = 200  # K
    J_strong = -0.4e-19  # Strong attraction
    
    # Multiple solutions of self-consistent equation
    theta_init_values = [0.1, 0.5, 0.9]
    markers = ['o', 's', '^']
    
    for mu in mu_range[::10]:
        for theta_init, marker in zip(theta_init_values, markers):
            try:
                theta_sol = solve_mean_field_coverage(mu, epsilon_0, J_strong, z, T_low, k_B, theta_init)
                ax3.plot(mu / 1.602e-19, theta_sol, marker, color='blue', markersize=4)
            except:
                pass
    
    ax3.set_xlabel('Chemical potential μ (eV)')
    ax3.set_ylabel('Coverage θ')
    ax3.set_title(f'Phase Transition (T = {T_low} K, J = {J_strong/1.602e-19:.2f} eV)')
    ax3.grid(True, alpha=0.3)
    
    # Compressibility
    ax4 = axes[1, 1]
    J_demo = J_values[1]
    mu_demo_range = np.linspace(-0.8e-19, 0.2e-19, 100)
    theta_demo = [solve_mean_field_coverage(mu, epsilon_0, J_demo, z, T, k_B)
                  for mu in mu_demo_range]
    
    # Compressibility κ ∝ ∂θ/∂μ
    dtheta_dmu = np.gradient(theta_demo, mu_demo_range)
    
    ax4.plot(mu_demo_range / 1.602e-19, dtheta_dmu * 1.602e-19, 'purple', linewidth=2)
    ax4.set_xlabel('Chemical potential μ (eV)')
    ax4.set_ylabel('∂θ/∂μ (eV⁻¹)')
    ax4.set_title('Response Function (Compressibility)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_lattice_gas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Lattice Gas Model (Mean Field Theory) ===\n")
    print(f"Adsorption energy ε₀ = {epsilon_0/1.602e-19:.2f} eV")
    print(f"Coordination number z = {z}")
    print(f"Temperature T = {T} K\n")
    
    for J, J_ev in zip(J_values, J_eV):
        print(f"Interaction J = {J_ev:.2f} eV:")
    
        # Coverage at μ = 0
        theta_0 = solve_mean_field_coverage(0, epsilon_0, J, z, T, k_B)
        print(f"  Coverage at μ=0: θ = {theta_0:.4f}\n")
    
    print("Correspondence with Ising model:")
    print("  n_i = 0, 1 → s_i = 2n_i - 1 = -1, +1")
    print("  Lattice gas ↔ Spin system")
    

## 💻 Example 3.4: Fermi-Dirac and Bose-Einstein Distributions

### Quantum Statistics

**Fermi-Dirac distribution** (fermions):

\\[ \langle n_i \rangle = \frac{1}{e^{\beta(\varepsilon_i - \mu)} + 1} \\]

Due to the Pauli exclusion principle, each state can have at most one particle.

**Bose-Einstein distribution** (bosons):

\\[ \langle n_i \rangle = \frac{1}{e^{\beta(\varepsilon_i - \mu)} - 1} \\]

Multiple particles can occupy each state.

**Classical limit** (high temperature or low density):

\\[ \langle n_i \rangle \approx e^{-\beta(\varepsilon_i - \mu)} \quad (\text{Maxwell-Boltzmann distribution}) \\]

Python Implementation: Comparison of Quantum Statistical Distributions
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    k_B = 1.380649e-23  # J/K
    
    def fermi_dirac(epsilon, mu, T, k_B):
        """Fermi-Dirac distribution"""
        beta = 1 / (k_B * T)
        x = beta * (epsilon - mu)
        # Avoid overflow
        if x > 100:
            return 0
        elif x < -100:
            return 1
        return 1 / (np.exp(x) + 1)
    
    def bose_einstein(epsilon, mu, T, k_B):
        """Bose-Einstein distribution"""
        beta = 1 / (k_B * T)
        x = beta * (epsilon - mu)
        if x <= 0:
            return np.inf  # μ < ε required
        if x < 0.01:
            return 1 / x  # Approximation
        return 1 / (np.exp(x) - 1)
    
    def maxwell_boltzmann(epsilon, mu, T, k_B):
        """Maxwell-Boltzmann distribution"""
        beta = 1 / (k_B * T)
        x = beta * (epsilon - mu)
        if x > 100:
            return 0
        return np.exp(-x)
    
    # Energy range (chemical potential set to 0)
    mu = 0
    epsilon_range = np.linspace(-5, 5, 500)  # In units of k_B T
    
    # Different temperatures (normalized in k_B T units)
    k_B_T = 1  # Normalized unit
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution at T = 1 (in k_B T units)
    ax1 = axes[0, 0]
    n_FD = [fermi_dirac(eps * k_B_T, mu, 1, k_B) for eps in epsilon_range]
    n_BE = [bose_einstein(eps * k_B_T, mu, 1, k_B) if eps > 0 else 0 for eps in epsilon_range]
    n_MB = [maxwell_boltzmann(eps * k_B_T, mu, 1, k_B) for eps in epsilon_range]
    
    ax1.plot(epsilon_range, n_FD, 'b-', linewidth=2, label='Fermi-Dirac')
    ax1.plot(epsilon_range, n_BE, 'r-', linewidth=2, label='Bose-Einstein')
    ax1.plot(epsilon_range, n_MB, 'g--', linewidth=2, label='Maxwell-Boltzmann')
    ax1.axvline(0, color='k', linestyle=':', linewidth=1, label='μ = 0')
    ax1.set_xlabel('(ε - μ) / (k_B T)')
    ax1.set_ylabel('Occupation number ⟨n⟩')
    ax1.set_title('Quantum Statistical Distributions (T = 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 3])
    
    # Fermi-Dirac at low temperature
    ax2 = axes[0, 1]
    temperatures_FD = [0.1, 0.5, 1.0, 2.0]
    colors_FD = ['blue', 'green', 'orange', 'red']
    
    for T_val, color in zip(temperatures_FD, colors_FD):
        n_FD_T = [fermi_dirac(eps * k_B_T, mu, T_val, k_B) for eps in epsilon_range]
        ax2.plot(epsilon_range, n_FD_T, color=color, linewidth=2,
                 label=f'k_B T = {T_val}')
    
    ax2.axvline(0, color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel('(ε - μ) / (E_F)')
    ax2.set_ylabel('⟨n⟩')
    ax2.set_title('Temperature Dependence of Fermi-Dirac Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bose-Einstein distribution and Bose condensation
    ax3 = axes[1, 0]
    epsilon_BE = np.linspace(0.01, 5, 100)
    temperatures_BE = [0.5, 1.0, 2.0, 5.0]
    
    for T_val, color in zip(temperatures_BE, colors_FD):
        n_BE_T = [bose_einstein(eps * k_B_T, mu * 0.9, T_val, k_B) for eps in epsilon_BE]
        ax3.semilogy(epsilon_BE, n_BE_T, color=color, linewidth=2,
                     label=f'k_B T = {T_val}')
    
    ax3.set_xlabel('(ε - μ) / (k_B T)')
    ax3.set_ylabel('⟨n⟩ (log scale)')
    ax3.set_title('Bose-Einstein Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # Classical limit verification
    ax4 = axes[1, 1]
    epsilon_classical = np.linspace(0, 10, 100)
    T_high = 5  # High temperature
    
    n_FD_high = [fermi_dirac(eps * k_B_T, mu, T_high, k_B) for eps in epsilon_classical]
    n_BE_high = [bose_einstein(eps * k_B_T, mu * 0.5, T_high, k_B) for eps in epsilon_classical]
    n_MB_high = [maxwell_boltzmann(eps * k_B_T, mu * 0.5, T_high, k_B) for eps in epsilon_classical]
    
    ax4.semilogy(epsilon_classical, n_FD_high, 'b-', linewidth=2, label='Fermi-Dirac')
    ax4.semilogy(epsilon_classical, n_BE_high, 'r-', linewidth=2, label='Bose-Einstein')
    ax4.semilogy(epsilon_classical, n_MB_high, 'g--', linewidth=2, label='Maxwell-Boltzmann')
    ax4.set_xlabel('(ε - μ) / (k_B T)')
    ax4.set_ylabel('⟨n⟩ (log scale)')
    ax4.set_title(f'Classical Limit (k_B T = {T_high})')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('stat_mech_quantum_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Quantum Statistics ===\n")
    
    print("Fermi-Dirac distribution:")
    print("  - Fermions (electrons, protons, neutrons, etc.)")
    print("  - Pauli exclusion principle: ⟨n⟩ ≤ 1")
    print("  - Step function at T = 0 (Fermi surface)\n")
    
    print("Bose-Einstein distribution:")
    print("  - Bosons (photons, phonons, He-4, etc.)")
    print("  - Multiple occupation possible: ⟨n⟩ ≥ 0")
    print("  - Bose condensation at low temperature (μ → ε₀)\n")
    
    print("Classical limit (high temperature):")
    print("  - When e^{β(ε-μ)} >> 1")
    print("  - FD ≈ BE ≈ MB")
    print("  - Quantum statistics → classical statistics\n")
    
    # Numerical example
    epsilon_test = 2 * k_B_T
    T_test = 300  # K
    print(f"Numerical example (ε-μ = 2k_BT, T = {T_test} K):")
    n_FD_test = fermi_dirac(epsilon_test, mu, T_test, k_B)
    n_BE_test = bose_einstein(epsilon_test, mu * 0.5, T_test, k_B)
    n_MB_test = maxwell_boltzmann(epsilon_test, mu * 0.5, T_test, k_B)
    print(f"  Fermi-Dirac: ⟨n⟩ = {n_FD_test:.6f}")
    print(f"  Bose-Einstein: ⟨n⟩ = {n_BE_test:.6f}")
    print(f"  Maxwell-Boltzmann: ⟨n⟩ = {n_MB_test:.6f}")
    

## 📚 Summary

1\. The **grand canonical ensemble** describes open systems with fluctuating particle numbers, where chemical potential is an important variable.

2\. From the **grand partition function** \\(\Xi\\), all thermodynamic quantities can be derived through the grand potential \\(\Omega\\).

3\. The **chemical potential** is the free energy change when adding a particle and determines equilibrium conditions.

4\. The **Langmuir adsorption isotherm** describes monolayer adsorption and is widely applied in materials science.

5\. The **lattice gas model** is equivalent to the Ising model and describes phase transitions in interacting systems.

6\. The **Fermi-Dirac distribution** is the quantum statistics for fermions, essential for describing electron systems and metals.

7\. The **Bose-Einstein distribution** is the quantum statistics for bosons, describing phonon and photon systems.

8\. In the high-temperature limit, quantum statistics reduce to the classical Maxwell-Boltzmann distribution.

### 💡 Exercises

  1. **Particle number fluctuation** : For an ideal gas in the grand canonical ensemble, calculate \\(\langle (\Delta N)^2 \rangle / \langle N \rangle\\) and show that the relative fluctuation approaches zero as \\(N \to \infty\\).
  2. **BET adsorption** : Implement the BET theory for multilayer adsorption and visualize the differences from the Langmuir isotherm.
  3. **Phase transition in lattice gas** : Find the critical temperature \\(T_c\\) using mean field theory and investigate its dependence on \\(J\\) and \\(z\\).
  4. **Fermi energy** : Calculate the Fermi energy \\(E_F\\) of a 3D electron gas as a function of density and compare with typical values for metals.
  5. **Planck distribution** : Derive the Planck distribution for blackbody radiation from the Bose-Einstein distribution of photons (with \\(\mu = 0\\)).

[← Chapter 2: Canonical Ensemble and Partition Function](<chapter-2.html>) [Chapter 4: Interacting Systems and Phase Transitions →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
