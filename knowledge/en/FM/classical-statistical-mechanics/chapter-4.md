---
title: "Chapter 4: Interacting Systems and Phase Transitions"
chapter_title: "Chapter 4: Interacting Systems and Phase Transitions"
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/classical-statistical-mechanics/chapter-4.html>) | Last sync: 2025-11-16

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Classical Statistical Mechanics](<index.html>) > Chapter 4 

## 🎯 Learning Objectives

  * Understand the statistical mechanics of the Ising model and ferromagnetism
  * Master the mean field approximation method
  * Learn the differences between first-order and second-order phase transitions
  * Understand critical phenomena and critical exponents
  * Learn Landau theory's description of order-disorder transitions
  * Understand van der Waals gas and liquid-gas phase transitions
  * Master the concept of order parameters
  * Understand phase transition phenomena in materials science

## 📖 Ising Model and Ferromagnetism

### Ising Model

A model with spins \\(s_i = \pm 1\\) arranged on lattice sites:

\\[ H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i \\]

Here, \\(J > 0\\) corresponds to ferromagnetic interaction where parallel alignment is favorable, while \\(J < 0\\) represents antiferromagnetic interaction where antiparallel alignment is favorable. The parameter \\(h\\) denotes the external magnetic field, and \\(\langle i,j \rangle\\) indicates the sum over nearest neighbor pairs.

**Magnetization (order parameter)** :

\\[ m = \frac{1}{N}\sum_i \langle s_i \rangle \\]

At low temperature \\(m \neq 0\\) (ferromagnetic phase), at high temperature \\(m = 0\\) (paramagnetic phase).

### Mean Field Approximation

Approximation where each spin feels a mean field \\(h_{\text{eff}} = Jz\langle s \rangle + h\\):

\\[ \langle s_i \rangle = \tanh(\beta h_{\text{eff}}) = \tanh(\beta(Jzm + h)) \\]

Here \\(z\\) is the coordination number and \\(m = \langle s_i \rangle\\) is the magnetization.

**Self-consistency equation** :

\\[ m = \tanh(\beta Jzm + \beta h) \\]

**Critical temperature** (\\(h = 0\\)):

\\[ T_c = \frac{Jz}{k_B} \\]

For \\(T < T_c\\), spontaneous magnetization \\(m \neq 0\\) appears.

## 💻 Example 4.1: Ising Mean Field Theory and Spontaneous Magnetization

### Temperature Dependence of Spontaneous Magnetization

Magnetization at \\(h = 0\\):

\\[ m = \tanh(\beta Jzm) \\]

Near the critical temperature \\(m \approx \sqrt{3(1 - T/T_c)}\\) (critical exponent \\(\beta = 1/2\\)).

Python Implementation: Ising Mean Field Theory
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    k_B = 1.380649e-23  # J/K
    
    def mean_field_ising_equation(m, T, J, z, h, k_B):
        """Mean field equation"""
        if T == 0:
            return m - np.sign(J*z*m + h)
        beta = 1 / (k_B * T)
        return m - np.tanh(beta * (J * z * m + h))
    
    def solve_magnetization(T, J, z, h, k_B, m_init=0.5):
        """Solve the self-consistency equation"""
        sol = fsolve(mean_field_ising_equation, m_init,
                      args=(T, J, z, h, k_B))
        return sol[0]
    
    # Parameters
    J = 1e-21  # J (interaction strength)
    z = 4  # 2D square lattice
    T_c = J * z / k_B  # Critical temperature
    
    # Temperature range
    T_range = np.linspace(0.01, 2.5 * T_c, 200)
    
    # Spontaneous magnetization at h = 0
    m_positive = []
    m_negative = []
    
    for T in T_range:
        # Positive solution
        m_pos = solve_magnetization(T, J, z, 0, k_B, m_init=0.9)
        m_positive.append(m_pos)
    
        # Negative solution
        m_neg = solve_magnetization(T, J, z, 0, k_B, m_init=-0.9)
        m_negative.append(m_neg)
    
    # Case with external field
    h_values = [0, 0.1e-21, 0.3e-21, 0.5e-21]
    h_eV = [h / 1.602e-19 for h in h_values]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature dependence of spontaneous magnetization
    ax1 = axes[0, 0]
    ax1.plot(T_range / T_c, m_positive, 'b-', linewidth=2, label='m > 0')
    ax1.plot(T_range / T_c, m_negative, 'r--', linewidth=2, label='m < 0')
    ax1.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c')
    ax1.set_xlabel('T / T_c')
    ax1.set_ylabel('Magnetization m')
    ax1.set_title('Spontaneous Magnetization (h = 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Effect of external field
    ax2 = axes[0, 1]
    for h, h_ev, color in zip(h_values, h_eV, colors):
        m_h = [solve_magnetization(T, J, z, h, k_B) for T in T_range]
        ax2.plot(T_range / T_c, m_h, color=color, linewidth=2,
                 label=f'h = {h_ev:.2f} eV')
    
    ax2.axvline(1, color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel('T / T_c')
    ax2.set_ylabel('Magnetization m')
    ax2.set_title('Magnetization by External Field')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Magnetic susceptibility (response at h = 0)
    ax3 = axes[1, 0]
    # χ = ∂m/∂h |_{h=0}
    epsilon = 1e-23  # Small field
    chi_values = []
    
    for T in T_range:
        m0 = solve_magnetization(T, J, z, 0, k_B)
        m_eps = solve_magnetization(T, J, z, epsilon, k_B)
        chi = (m_eps - m0) / epsilon
        chi_values.append(chi)
    
    ax3.plot(T_range / T_c, np.array(chi_values) * 1e21, 'purple', linewidth=2)
    ax3.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c')
    ax3.set_xlabel('T / T_c')
    ax3.set_ylabel('χ (10⁻²¹ J⁻¹)')
    ax3.set_title('Magnetic Susceptibility (Diverges at T_c)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Enlarged critical region
    ax4 = axes[1, 1]
    T_critical = np.linspace(0.5 * T_c, 1.5 * T_c, 100)
    m_critical = [solve_magnetization(T, J, z, 0, k_B, m_init=0.9) for T in T_critical]
    
    ax4.plot(T_critical / T_c, m_critical, 'b-', linewidth=2)
    ax4.axvline(1, color='k', linestyle=':', linewidth=2, label='T_c')
    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('Magnetization m')
    ax4.set_title('Magnetization in Critical Region')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_ising_mean_field.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numerical results
    print("=== Ising Mean Field Theory ===\n")
    print(f"Interaction strength J = {J:.2e} J = {J/1.602e-19:.4f} eV")
    print(f"Coordination number z = {z}")
    print(f"Critical temperature T_c = {T_c:.2f} K\n")
    
    # Magnetization at different temperatures
    test_temps = [0.5 * T_c, 0.9 * T_c, 1.0 * T_c, 1.5 * T_c]
    for T_test in test_temps:
        m_test = solve_magnetization(T_test, J, z, 0, k_B, m_init=0.9)
        print(f"T = {T_test:.2f} K (T/T_c = {T_test/T_c:.2f}): m = {m_test:.4f}")
    
    print("\nCritical exponents:")
    print("  Magnetization: m ~ (T_c - T)^β, β = 1/2 (mean field)")
    print("  Susceptibility: χ ~ |T - T_c|^{-γ}, γ = 1 (mean field)")
    

## 💻 Example 4.2: Classification of Phase Transitions and Critical Phenomena

### Classification of Phase Transitions (Ehrenfest classification)

**First-order phase transition** : In this type of transition, the first derivative of free energy (entropy, volume) is discontinuous, and latent heat exists. Examples include liquid-gas transition, melting, and evaporation.

**Second-order phase transition** : Here, the second derivative of free energy (specific heat, compressibility) is discontinuous or diverges. There is no latent heat, and the transition is continuous. Examples include ferromagnetic transition, superconducting transition, and superfluid transition.

**Critical exponents** characterize second-order phase transitions: \\(\beta\\) describes the order parameter as \\(m \sim |T - T_c|^\beta\\), \\(\gamma\\) describes the susceptibility as \\(\chi \sim |T - T_c|^{-\gamma}\\), \\(\alpha\\) describes the specific heat as \\(C \sim |T - T_c|^{-\alpha}\\), and \\(\delta\\) describes the critical isotherm as \\(m \sim h^{1/\delta}\\) at \\(T = T_c\\).

Mean field theory: \\(\beta = 1/2, \gamma = 1, \alpha = 0, \delta = 3\\)

Python Implementation: Calculation of Critical Exponents
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve, curve_fit
    
    k_B = 1.380649e-23
    
    def solve_m(T, J, z, h, k_B):
        """Solve mean field equation"""
        if T < 1e-10:
            return np.sign(h) if h != 0 else 1.0
        beta = 1 / (k_B * T)
        eq = lambda m: m - np.tanh(beta * (J * z * m + h))
        return fsolve(eq, 0.5 if h >= 0 else -0.5)[0]
    
    J = 1e-21
    z = 4
    T_c = J * z / k_B
    
    # Calculation of critical exponent β
    T_below_Tc = np.linspace(0.5 * T_c, 0.999 * T_c, 50)
    m_beta = [solve_m(T, J, z, 0, k_B) for T in T_below_Tc]
    reduced_temp_beta = (T_c - T_below_Tc) / T_c
    
    # Power law fitting
    def power_law_beta(t, beta_exp):
        return t**beta_exp
    
    # Find exponent by log fitting
    log_t = np.log(reduced_temp_beta)
    log_m = np.log(np.abs(m_beta))
    poly_beta = np.polyfit(log_t, log_m, 1)
    beta_fitted = poly_beta[0]
    
    # Calculation of critical exponent γ (susceptibility)
    T_chi = np.linspace(1.01 * T_c, 2 * T_c, 50)
    epsilon_h = 1e-23
    chi_gamma = []
    
    for T in T_chi:
        m0 = solve_m(T, J, z, 0, k_B)
        m_h = solve_m(T, J, z, epsilon_h, k_B)
        chi = (m_h - m0) / epsilon_h
        chi_gamma.append(chi)
    
    reduced_temp_gamma = (T_chi - T_c) / T_c
    log_t_gamma = np.log(reduced_temp_gamma)
    log_chi = np.log(chi_gamma)
    poly_gamma = np.polyfit(log_t_gamma, log_chi, 1)
    gamma_fitted = -poly_gamma[0]
    
    # Calculation of critical exponent δ (T = T_c)
    h_range = np.logspace(-24, -20, 30)
    m_delta = [solve_m(T_c, J, z, h, k_B) for h in h_range]
    
    log_h = np.log(h_range)
    log_m_delta = np.log(np.abs(m_delta))
    poly_delta = np.polyfit(log_h, log_m_delta, 1)
    delta_fitted = 1 / poly_delta[0]
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Critical exponent β
    ax1 = axes[0, 0]
    ax1.loglog(reduced_temp_beta, m_beta, 'bo', markersize=6, label='Data')
    ax1.loglog(reduced_temp_beta, reduced_temp_beta**0.5, 'r--', linewidth=2,
               label=f'β = 0.5 (theory)')
    ax1.loglog(reduced_temp_beta, reduced_temp_beta**beta_fitted, 'g-', linewidth=2,
               label=f'β = {beta_fitted:.3f} (fit)')
    ax1.set_xlabel('(T_c - T) / T_c')
    ax1.set_ylabel('Magnetization m')
    ax1.set_title('Critical Exponent β')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Critical exponent γ
    ax2 = axes[0, 1]
    ax2.loglog(reduced_temp_gamma, np.array(chi_gamma) * 1e21, 'go', markersize=6, label='Data')
    ax2.loglog(reduced_temp_gamma, 1e21 * (reduced_temp_gamma)**(-1), 'r--', linewidth=2,
               label='γ = 1 (theory)')
    ax2.loglog(reduced_temp_gamma, 1e21 * (reduced_temp_gamma)**(-gamma_fitted), 'b-', linewidth=2,
               label=f'γ = {gamma_fitted:.3f} (fit)')
    ax2.set_xlabel('(T - T_c) / T_c')
    ax2.set_ylabel('χ (10⁻²¹ J⁻¹)')
    ax2.set_title('Critical Exponent γ')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Critical exponent δ
    ax3 = axes[1, 0]
    ax3.loglog(h_range / 1.602e-19, np.abs(m_delta), 'mo', markersize=6, label='Data')
    ax3.loglog(h_range / 1.602e-19, (h_range/1e-21)**(1/3), 'r--', linewidth=2,
               label='δ = 3 (theory)')
    ax3.loglog(h_range / 1.602e-19, (h_range/1e-21)**(1/delta_fitted), 'c-', linewidth=2,
               label=f'δ = {delta_fitted:.3f} (fit)')
    ax3.set_xlabel('h (eV)')
    ax3.set_ylabel('m (at T = T_c)')
    ax3.set_title('Critical Exponent δ')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # Specific heat (numerical differentiation)
    ax4 = axes[1, 1]
    T_heat = np.linspace(0.5 * T_c, 1.5 * T_c, 100)
    dT = T_heat[1] - T_heat[0]
    free_energy = []
    
    for T in T_heat:
        m_eq = solve_m(T, J, z, 0, k_B)
        # Free energy (mean field)
        if T > 0:
            beta = 1 / (k_B * T)
            f = -J * z * m_eq**2 / 2 - k_B * T * np.log(2 * np.cosh(beta * J * z * m_eq))
        else:
            f = -J * z
        free_energy.append(f)
    
    entropy = -np.gradient(free_energy, dT)
    heat_capacity = T_heat * np.gradient(entropy, dT)
    
    ax4.plot(T_heat / T_c, -heat_capacity / k_B, 'r-', linewidth=2)
    ax4.axvline(1, color='k', linestyle=':', linewidth=2, label='T_c')
    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('C / k_B')
    ax4.set_title('Specific Heat (Discontinuous at T_c)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_critical_exponents.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Numerical results
    print("=== Critical Exponents (Mean Field Theory) ===\n")
    print(f"Theoretical values (mean field):")
    print(f"  β = 1/2 (magnetization)")
    print(f"  γ = 1   (susceptibility)")
    print(f"  δ = 3   (critical isotherm)")
    print(f"  α = 0   (specific heat, discontinuous)\n")
    
    print(f"Numerical fits:")
    print(f"  β = {beta_fitted:.4f}")
    print(f"  γ = {gamma_fitted:.4f}")
    print(f"  δ = {delta_fitted:.4f}\n")
    
    print("Experimental values (3D Ising):")
    print("  β ≈ 0.325")
    print("  γ ≈ 1.24")
    print("  δ ≈ 4.8")
    print("  α ≈ 0.11")
    

## 💻 Example 4.3: Landau Theory

### Landau Free Energy

Expand the free energy as a function of the order parameter \\(m\\):

\\[ F(m, T) = F_0 + a(T) m^2 + b m^4 - hm \\]

Here \\(a(T) = a_0(T - T_c)\\), \\(b > 0\\).

**Equilibrium condition** : \\(\frac{\partial F}{\partial m} = 0\\)

\\[ 2a(T)m + 4bm^3 = h \\]

When \\(h = 0\\): For \\(T > T_c\\), only \\(m = 0\\) is stable, corresponding to the paramagnetic phase. For \\(T < T_c\\), the solutions \\(m = \pm\sqrt{-a(T)/(2b)}\\) become stable, corresponding to the ferromagnetic phase.

Python Implementation: Landau Theory
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    
    def landau_free_energy(m, T, T_c, a0, b, h):
        """Landau free energy"""
        a = a0 * (T - T_c)
        return a * m**2 + b * m**4 - h * m
    
    def equilibrium_magnetization(T, T_c, a0, b, h):
        """Equilibrium magnetization (free energy minimization)"""
        # Search from multiple initial values
        m_candidates = []
        for m_init in [-1, 0, 1]:
            result = minimize_scalar(lambda m: landau_free_energy(m, T, T_c, a0, b, h),
                                      bounds=(-2, 2), method='bounded')
            m_candidates.append(result.x)
    
        # Select solution that gives minimum free energy
        F_values = [landau_free_energy(m, T, T_c, a0, b, h) for m in m_candidates]
        return m_candidates[np.argmin(F_values)]
    
    # Parameters
    T_c = 500  # K
    a0 = 1e-4
    b = 1e-6
    h_values = [0, 0.01, 0.05, 0.1]
    
    T_range = np.linspace(300, 700, 100)
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Shape of Landau potential (different temperatures)
    ax1 = axes[0, 0]
    m_plot = np.linspace(-1.5, 1.5, 200)
    temperatures_plot = [0.8*T_c, 0.95*T_c, T_c, 1.2*T_c]
    
    for T_val, color in zip(temperatures_plot, colors):
        F_plot = [landau_free_energy(m, T_val, T_c, a0, b, 0) for m in m_plot]
        ax1.plot(m_plot, F_plot, color=color, linewidth=2,
                 label=f'T/T_c = {T_val/T_c:.2f}')
    
    ax1.set_xlabel('Order parameter m')
    ax1.set_ylabel('Free energy F(m)')
    ax1.set_title('Landau Potential (h = 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature dependence of equilibrium magnetization
    ax2 = axes[0, 1]
    for h, color in zip(h_values, colors):
        m_eq = [equilibrium_magnetization(T, T_c, a0, b, h) for T in T_range]
        ax2.plot(T_range / T_c, m_eq, color=color, linewidth=2,
                 label=f'h = {h:.2f}')
    
    ax2.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c')
    ax2.set_xlabel('T / T_c')
    ax2.set_ylabel('Magnetization m')
    ax2.set_title('Equilibrium Magnetization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # First-order transition (case with negative b)
    ax3 = axes[1, 0]
    # F = a m² + b m⁴ + c m⁶ (first-order transition when b < 0)
    b_first_order = -2e-6
    c = 1e-7
    
    def landau_first_order(m, T, T_c, a0, b, c, h):
        a = a0 * (T - T_c)
        return a * m**2 + b * m**4 + c * m**6 - h * m
    
    m_1st = np.linspace(-1.5, 1.5, 200)
    T_1st = [0.95*T_c, 0.98*T_c, T_c, 1.02*T_c]
    
    for T_val, color in zip(T_1st, colors):
        F_1st = [landau_first_order(m, T_val, T_c, a0, b_first_order, c, 0) for m in m_1st]
        ax3.plot(m_1st, F_1st, color=color, linewidth=2,
                 label=f'T/T_c = {T_val/T_c:.2f}')
    
    ax3.set_xlabel('Order parameter m')
    ax3.set_ylabel('Free energy F(m)')
    ax3.set_title('First-Order Phase Transition (Double-Well Potential)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Magnetic susceptibility
    ax4 = axes[1, 1]
    epsilon_h = 0.001
    chi_landau = []
    
    for T in T_range:
        m0 = equilibrium_magnetization(T, T_c, a0, b, 0)
        m_h = equilibrium_magnetization(T, T_c, a0, b, epsilon_h)
        chi = (m_h - m0) / epsilon_h
        chi_landau.append(chi)
    
    ax4.plot(T_range / T_c, chi_landau, 'purple', linewidth=2)
    ax4.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c')
    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('χ')
    ax4.set_title('Magnetic Susceptibility (Diverges at T_c)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_landau_theory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Landau Theory ===\n")
    print(f"Parameters:")
    print(f"  T_c = {T_c} K")
    print(f"  a₀ = {a0}")
    print(f"  b = {b}\n")
    
    print("Second-order phase transition (b > 0):")
    print(f"  T > T_c: m = 0 (single well)")
    print(f"  T < T_c: m = ±√(-a/2b) (double well)")
    print(f"  Critical exponent β = 1/2\n")
    
    print("First-order phase transition (b < 0, c > 0):")
    print(f"  Double-well potential")
    print(f"  Magnetization jumps discontinuously at transition")
    print(f"  Latent heat exists")
    

## 💻 Example 4.4: van der Waals Gas and Liquid-Gas Phase Transition

### van der Waals Equation of State

\\[ \left(P + \frac{a}{V^2}\right)(V - b) = k_B T \\]

Here, \\(a\\) represents intermolecular attraction (cohesion) and \\(b\\) represents the excluded volume of molecules.

**Critical point** :

\\[ T_c = \frac{8a}{27k_B b}, \quad P_c = \frac{a}{27b^2}, \quad V_c = 3b \\]

In reduced variables \\(t = T/T_c, p = P/P_c, v = V/V_c\\):

\\[ \left(p + \frac{3}{v^2}\right)(3v - 1) = 8t \\]

For \\(T < T_c\\), liquid-gas coexistence (Maxwell construction).

Python Implementation: van der Waals Gas
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    k_B = 1.380649e-23
    
    def vdw_pressure(V, T, a, b, k_B):
        """van der Waals pressure"""
        return k_B * T / (V - b) - a / V**2
    
    def vdw_equation(V, P, T, a, b, k_B):
        """van der Waals equation (find volume)"""
        return P - vdw_pressure(V, T, a, b, k_B)
    
    # van der Waals parameters (Ar gas)
    a = 1.355e-49  # J·m³
    b = 3.201e-29  # m³
    
    T_c = 8 * a / (27 * k_B * b)
    P_c = a / (27 * b**2)
    V_c = 3 * b
    
    print(f"Critical constants:")
    print(f"  T_c = {T_c:.2f} K")
    print(f"  P_c = {P_c/1e6:.2f} MPa")
    print(f"  V_c = {V_c*1e30:.2f} ų\n")
    
    # Isotherms (reduced variables)
    v_range = np.linspace(0.4, 5, 300)
    temperatures_reduced = [0.85, 0.95, 1.0, 1.1, 1.3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures_reduced)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P-V diagram (reduced variables)
    ax1 = axes[0, 0]
    for t_r, color in zip(temperatures_reduced, colors):
        p_values = [(3 / v**2) * (8*t_r / (3*v - 1) - 1) for v in v_range if 3*v > 1]
        v_plot = [v for v in v_range if 3*v > 1]
        ax1.plot(v_plot, p_values, color=color, linewidth=2,
                 label=f't = {t_r:.2f}')
    
    ax1.axhline(1, color='k', linestyle=':', linewidth=1)
    ax1.axvline(1, color='k', linestyle=':', linewidth=1)
    ax1.plot(1, 1, 'ro', markersize=10, label='Critical point')
    ax1.set_xlabel('v = V/V_c')
    ax1.set_ylabel('p = P/P_c')
    ax1.set_title('van der Waals Isotherms (Reduced Variables)')
    ax1.set_xlim([0.4, 3])
    ax1.set_ylim([0, 2])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Maxwell equal area rule
    ax2 = axes[0, 1]
    t_coexist = 0.90
    v_fine = np.linspace(0.4, 3, 500)
    p_fine = [(3 / v**2) * (8*t_coexist / (3*v - 1) - 1) for v in v_fine if 3*v > 1]
    v_plot_fine = [v for v in v_fine if 3*v > 1]
    
    ax2.plot(v_plot_fine, p_fine, 'b-', linewidth=2, label=f't = {t_coexist}')
    
    # Equilibrium pressure estimation (simplified)
    p_equilibrium = 0.7  # Adjusted by visual inspection
    ax2.axhline(p_equilibrium, color='r', linestyle='--', linewidth=2,
                label='Maxwell construction')
    
    ax2.set_xlabel('v = V/V_c')
    ax2.set_ylabel('p = P/P_c')
    ax2.set_title('Maxwell Equal Area Rule')
    ax2.set_xlim([0.4, 3])
    ax2.set_ylim([0.4, 1.2])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Compressibility
    ax3 = axes[1, 0]
    for t_r, color in zip(temperatures_reduced, colors):
        # κ = -1/V (∂V/∂P)_T
        v_comp = np.linspace(0.5, 3, 100)
        dv = v_comp[1] - v_comp[0]
        p_comp = [(3 / v**2) * (8*t_r / (3*v - 1) - 1) for v in v_comp if 3*v > 1]
        v_comp_valid = [v for v in v_comp if 3*v > 1]
    
        if len(p_comp) > 2:
            dP_dV = np.gradient(p_comp, dv)
            kappa = -1 / (np.array(v_comp_valid) * np.array(dP_dV))
            ax3.plot(v_comp_valid, kappa, color=color, linewidth=2,
                     label=f't = {t_r:.2f}')
    
    ax3.set_xlabel('v = V/V_c')
    ax3.set_ylabel('κ (reduced)')
    ax3.set_title('Isothermal Compressibility')
    ax3.set_ylim([0, 10])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Phase diagram (T-P plane)
    ax4 = axes[1, 1]
    # Vapor pressure curve (simplified estimation)
    T_sat = np.linspace(0.6 * T_c, T_c, 50)
    # Clausius-Clapeyron approximation
    P_sat = P_c * np.exp(-1.5 * (1 - T_sat / T_c))
    
    ax4.plot(T_sat / T_c, P_sat / P_c, 'b-', linewidth=3, label='Coexistence curve')
    ax4.plot(1, 1, 'ro', markersize=12, label='Critical point')
    ax4.fill_between(T_sat / T_c, 0, P_sat / P_c, alpha=0.3, color='blue', label='Liquid')
    ax4.fill_between(T_sat / T_c, P_sat / P_c, 2, alpha=0.3, color='red', label='Gas')
    
    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('P / P_c')
    ax4.set_title('Phase Diagram (Liquid-Gas Coexistence Curve)')
    ax4.set_xlim([0.6, 1.2])
    ax4.set_ylim([0, 1.5])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stat_mech_vdw_phase_transition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Features of van der Waals gas:")
    print("  - Liquid-gas coexistence below critical point")
    print("  - Maxwell equal area rule determines equilibrium pressure")
    print("  - Isothermal compressibility diverges at critical point")
    print("  - Mathematically equivalent to lattice gas model")
    

## 📚 Summary

1\. The **Ising model** is the most fundamental model for interacting spin systems, describing ferromagnetic transitions.

2\. The **mean field approximation** is a method that reduces many-body problems to one-body problems, predicting critical temperature and spontaneous magnetization.

3\. In **second-order phase transitions** , the order parameter changes continuously and is characterized by critical exponents.

4\. In **first-order phase transitions** , the order parameter changes discontinuously and latent heat exists.

5\. **Landau theory** is a phenomenological description of order parameters, capturing universal properties of phase transitions.

6\. **Critical exponents** can be calculated by mean field theory but systematically deviate from experimental values (dimension dependence).

7\. The **van der Waals gas** describes liquid-gas phase transitions and is mathematically equivalent to the lattice gas model.

8\. Phase transition theory is essential for understanding ferromagnetism, superconductivity, and structural phase transitions in materials science.

### 💡 Exercises

  1. **Antiferromagnetic Ising** : Develop mean field theory for the antiferromagnetic Ising model with \\(J < 0\\), considering two sublattices, and find the Néel temperature.
  2. **3D Ising** : Find the critical temperature for the ferromagnetic Ising model on a simple cubic lattice (\\(z = 6\\)) and compare with the experimental value (\\(T_c \approx 4.51 J/k_B\\)).
  3. **Widom scaling** : Verify the scaling relations \\(\alpha + 2\beta + \gamma = 2\\) and \\(\gamma = \beta(\delta - 1)\\) that hold among critical exponents in mean field theory.
  4. **Landau-Ginzburg** : Consider the Landau-Ginzburg free energy including spatial dependence and calculate the interface energy.
  5. **van der Waals real gas** : Using van der Waals parameters for CO₂, calculate the critical point and phase diagram.

[← Chapter 3: Grand Canonical Ensemble and Chemical Potential](<chapter-3.html>) [Chapter 5: Statistical Mechanics Simulations →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
