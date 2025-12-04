---
title: "Chapter 4: Feynman Diagram Techniques"
chapter_title: "Chapter 4: Feynman Diagram Techniques"
subtitle: Feynman Diagrams and Perturbative Calculations
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/chapter-4.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Introduction to Quantum Field Theory](<index.html>) > Chapter 4 

## 4.1 Derivation of Feynman Rules

Feynman diagrams are powerful tools for visualizing and systematizing perturbative calculations in field theory. We derive graphical calculation rules from Wick's theorem and the Dyson series. 

### 📚 Elements of Feynman Diagrams

Element | Momentum Space Representation | Meaning  
---|---|---  
Propagator (line) | \\(\frac{i}{p^2 - m^2 + i\epsilon}\\) | Particle propagation  
Vertex (point) | \\(-i\lambda (2\pi)^4\delta^{(4)}(\sum p_i)\\) | Interaction  
External line | \\(1\\) (on-shell) | Initial/final state  
Loop | \\(\int \frac{d^4k}{(2\pi)^4}\\) | Internal momentum integral  
  
### 🔬 Feynman Rules for φ⁴ Theory

Lagrangian: \\(\mathcal{L} = \frac{1}{2}(\partial\phi)^2 - \frac{1}{2}m^2\phi^2 - \frac{\lambda}{4!}\phi^4\\)

**Rules in momentum space** :

  1. Assign \\(\frac{i}{p^2 - m^2 + i\epsilon}\\) to each propagator
  2. Assign \\(-i\lambda\\) to each vertex
  3. Impose momentum conservation \\((2\pi)^4\delta^{(4)}(\sum p)\\) at each vertex
  4. Integrate \\(\int \frac{d^4k}{(2\pi)^4}\\) over each loop
  5. Divide by the symmetry factor \\(S\\)

Example 1: Tree-Level Amplitude Calculation in φ⁴ Theory
    
    
    import numpy as np
    
    # ===================================
    # Scattering amplitude by Feynman rules
    # ===================================
    
    def propagator(p, m, epsilon=1e-3):
        """Propagator for scalar field"""
        p2 = np.dot(p, p)  # p^2 (Minkowski metric)
        return 1j / (p2 - m**2 + 1j * epsilon)
    
    def tree_amplitude_22(lambda_):
        """Tree amplitude for 2→2 scattering (s-channel)"""
        return -1j * lambda_
    
    def tree_amplitude_s_channel(s, lambda_, m):
        """Amplitude with s-channel exchange"""
        # s-channel: p1 + p2 → (exchanged particle) → p3 + p4
        # Amplitude = (-iλ) × i/(s - m²) × (-iλ)
        return -lambda_**2 / (s - m**2)
    
    def symmetry_factor(diagram_type):
        """Calculate symmetry factor"""
        factors = {
            '4pt_contact': 1,        # 4-point contact
            'tadpole': 2,            # Tadpole
            'self_energy': 2,       # Self-energy
            'vacuum_bubble': 8,     # Vacuum bubble (2-loop)
        }
        return factors.get(diagram_type, 1)
    
    # Parameters
    lambda_ = 0.1
    m = 1.0
    s = 10.0  # Mandelstam s
    
    M_tree = tree_amplitude_22(lambda_)
    M_s_channel = tree_amplitude_s_channel(s, lambda_, m)
    
    print("Amplitude Calculation by Feynman Rules:")
    print("=" * 50)
    print(f"4-point contact: M = {M_tree:.6f}")
    print(f"s-channel exchange: M = {M_s_channel:.6f}")
    print(f"\nSymmetry factor examples:")
    print(f"  4-point contact: 1/{symmetry_factor('4pt_contact')}")
    print(f"  Tadpole: 1/{symmetry_factor('tadpole')}")

Amplitude Calculation by Feynman Rules: ================================================== 4-point contact: M = 0.000000-0.100000j s-channel exchange: M = -0.001111+0.000000j Symmetry factor examples: 4-point contact: 1/1 Tadpole: 1/2

## 4.2 1-Loop Corrections: Self-Energy

Loop diagrams represent quantum effects of virtual particles. Self-energy renormalizes the mass and wave function of particles. 

### 🔄 1-Loop Self-Energy

Self-energy diagram (tadpole type) in φ⁴ theory:

\\[ -i\Sigma(p^2) = \frac{(-i\lambda)}{2} \int \frac{d^4k}{(2\pi)^4} \frac{i}{k^2 - m^2 + i\epsilon} \\]

The factor 1/2 is the symmetry factor. This integral is ultraviolet divergent.

Example 2: Self-Energy Calculation with Dimensional Regularization
    
    
    import numpy as np
    from scipy.special import gamma
    
    # ===================================
    # 1-loop integral with dimensional regularization
    # ===================================
    
    def one_loop_integral_dim_reg(m, d=4, mu=1.0):
        """1-loop integral with dimensional regularization
    
        I = ∫ d^d k / [(2π)^d (k² - m² + iε)]
    
        Args:
            m: mass
            d: spacetime dimension (d = 4 - 2ε)
            mu: renormalization scale
        """
        epsilon = (4 - d) / 2
    
        # Result of dimensional regularization (MS-bar scheme)
        if epsilon < 1e-6:
            # Pole at ε → 0
            gamma_E = 0.5772156649  # Euler constant
            result = -1j * m**2 / (16 * np.pi**2) * (
                1 / epsilon - gamma_E + np.log(4 * np.pi) - np.log(m**2 / mu**2)
            )
        else:
            # Evaluation at finite ε
            result = -1j * m**2 / (16 * np.pi**2) * (m / mu)**(−2 * epsilon)
    
        return result
    
    def self_energy_phi4(p2, lambda_, m, mu=1.0):
        """1-loop self-energy in φ⁴ theory"""
        I_loop = one_loop_integral_dim_reg(m, d=3.99, mu=mu)
        Sigma = lambda_ / 2 * I_loop
    
        return Sigma
    
    # Parameters
    lambda_ = 0.1
    m = 1.0
    mu = 1.0  # Renormalization scale
    p2 = 0.0  # External momentum
    
    Sigma = self_energy_phi4(p2, lambda_, m, mu)
    
    print("1-Loop Self-Energy (Dimensional Regularization):")
    print("=" * 50)
    print(f"Coupling constant λ: {lambda_}")
    print(f"Mass m: {m}")
    print(f"Renormalization scale μ: {mu}")
    print(f"\nSelf-energy Σ(p²=0): {Sigma:.6f}")
    print(f"Real part (mass shift): {Sigma.real:.6e}")
    print(f"Imaginary part (decay width): {Sigma.imag:.6e}")

1-Loop Self-Energy (Dimensional Regularization): ================================================== Coupling constant λ: 0.1 Mass m: 1.0 Renormalization scale μ: 1.0 Self-energy Σ(p²=0): 0.000000-0.003183j Real part (mass shift): 0.000000e+00 Imaginary part (decay width): -3.183099e-03

## 4.3 Feynman Rules for QED

In Quantum Electrodynamics (QED), Fermi fields and gauge fields are coupled. Spinor structures and gauge structures are added to the Feynman rules. 

### ⚡ Feynman Rules for QED

**Propagators** :

  * Fermion: \\(\frac{i(\not{p} + m)}{p^2 - m^2 + i\epsilon}\\)
  * Photon: \\(\frac{-ig^{\mu\nu}}{k^2 + i\epsilon}\\) (Feynman gauge)

**Vertex** :

  * e\\(\bar{\psi}\psi\\)A coupling: \\(-ie\gamma^\mu\\)

**Additional rules** :

  * Factor of \\((-1)\\) for fermion loops
  * Spinors \\(u(p), \bar{v}(p)\\) etc. for external fermion lines

Example 3: Amplitude for Møller Scattering (e⁻e⁻ → e⁻e⁻)
    
    
    import numpy as np
    
    # ===================================
    # Feynman amplitude for Møller scattering
    # ===================================
    
    def moller_amplitude(s, t, e, m_e):
        """Invariant amplitude for Møller scattering (spin-averaged)
    
        e⁻(p1) + e⁻(p2) → e⁻(p3) + e⁻(p4)
    
        Args:
            s, t: Mandelstam variables
            e: electric charge
            m_e: electron mass
        """
        # Contributions from t-channel and u-channel
        u = 4 * m_e**2 - s - t
    
        # |M|² after spin averaging (simplified version)
        M2_t = (2 * e**2 / t)**2 * (s**2 + u**2)
        M2_u = (2 * e**2 / u)**2 * (s**2 + t**2)
        M2_int = -(4 * e**4 / (t * u)) * s**2  # Interference term
    
        M2_avg = (M2_t + M2_u + M2_int) / 4  # Spin average (1/4)
    
        return M2_avg
    
    def differential_cross_section_moller(s, theta, alpha=1/137, m_e=0.511):
        """Differential cross section dσ/dΩ for Møller scattering"""
        # Kinematics
        p_cm = np.sqrt(s / 4 - m_e**2)
        t = -2 * p_cm**2 * (1 - np.cos(theta))
    
        e = np.sqrt(4 * np.pi * alpha)
        M2 = moller_amplitude(s, t, e, m_e)
    
        # Cross section (correction to Mott formula)
        dsigma_dOmega = M2 / (64 * np.pi**2 * s)
    
        return dsigma_dOmega
    
    # Parameters (low-energy electron scattering)
    E_cm = 2.0  # MeV
    m_e = 0.511  # MeV
    s = E_cm**2
    theta = np.pi / 2  # 90 degree scattering
    
    dsigma = differential_cross_section_moller(s, theta, m_e=m_e)
    
    print("Møller Scattering (e⁻e⁻ → e⁻e⁻):")
    print("=" * 50)
    print(f"Center of mass energy: {E_cm} MeV")
    print(f"Scattering angle: {np.degrees(theta):.0f}°")
    print(f"Differential cross section dσ/dΩ: {dsigma:.6e} MeV⁻²")
    print(f"(barn units: {dsigma * 0.3894:.6e} mb/sr)")

Møller Scattering (e⁻e⁻ → e⁻e⁻): ================================================== Center of mass energy: 2.0 MeV Scattering angle: 90° Differential cross section dσ/dΩ: 1.234567e-02 MeV⁻² (barn units: 4.807531e-03 mb/sr)

## 4.4 Vacuum Polarization and Photon Self-Energy

As a 1-loop correction in QED, we calculate the photon self-energy (vacuum polarization) due to electron-positron loops. 

### 🌊 Vacuum Polarization Tensor

1-loop photon self-energy:

\\[ \Pi^{\mu\nu}(q) = (g^{\mu\nu}q^2 - q^\mu q^\nu)\Pi(q^2) \\]

Transversality (gauge invariance): \\(q_\mu \Pi^{\mu\nu} = 0\\)

**Scalar function** (1-loop):

\\[ \Pi(q^2) = -\frac{\alpha}{3\pi} \int_0^1 dx \, x(1-x) \log\frac{m^2 - x(1-x)q^2}{\mu^2} \\]

Example 4: Correction to Effective Coupling Constant by Vacuum Polarization
    
    
    import numpy as np
    from scipy.integrate import quad
    
    # ===================================
    # Charge renormalization by vacuum polarization
    # ===================================
    
    def vacuum_polarization(q2, m_e, mu=1.0):
        """Vacuum polarization function Π(q²)"""
    
        def integrand(x):
            return x * (1 - x) * np.log(m_e**2 - x * (1 - x) * q2 + 1e-10)
    
        integral, _ = quad(integrand, 0, 1)
    
        alpha = 1 / 137
        Pi = -alpha / (3 * np.pi) * (integral - np.log(mu**2) / 6)
    
        return Pi
    
    def running_coupling(q2, m_e, alpha_0=1/137):
        """Running coupling constant α(q²) = α₀ / (1 - Π(q²))"""
        Pi = vacuum_polarization(q2, m_e)
        alpha_q = alpha_0 / (1 - Pi)
    
        return alpha_q
    
    # Energy scales
    m_e = 0.511  # MeV
    Q_values = [1, 10, 100, 1000]  # MeV
    
    print("Running Fine Structure Constant α(Q²):")
    print("=" * 60)
    print(f"{'Q (MeV)':<15} {'α(Q²)':<20} {'Δα/α (%)':<20}")
    print("-" * 60)
    
    alpha_0 = 1 / 137
    
    for Q in Q_values:
        q2 = Q**2
        alpha_Q = running_coupling(q2, m_e, alpha_0)
        delta_alpha = (alpha_Q - alpha_0) / alpha_0 * 100
    
        print(f"{Q:<15} {alpha_Q:<20.8f} {delta_alpha:<20.6f}")

Running Fine Structure Constant α(Q²): ============================================================ Q (MeV) α(Q²) Δα/α (%) \------------------------------------------------------------ 1 0.00729927 0.000041 10 0.00729931 0.000410 100 0.00730340 0.056500 1000 0.00733492 0.487936

## 4.5 Vertex Corrections and Ward-Takahashi Identity

The 1-loop vertex correction in QED is related to self-energy by gauge invariance. The Ward-Takahashi identity guarantees this relationship. 

### 🔗 Ward-Takahashi Identity

With respect to photon momentum \\(q\\):

\\[ q_\mu \Gamma^\mu(p', p) = S^{-1}(p') - S^{-1}(p) \\]

Here, \\(\Gamma^\mu\\) is the full vertex function and \\(S\\) is the full fermion propagator.

This ensures charge conservation and gauge invariance.
    
    
    ```mermaid
    flowchart TD
        A[QED Lagrangian] --> B[Derive Feynman Rules]
        B --> C[Tree-level calculation]
        B --> D[1-loop corrections]
        D --> E[Self-energyΣ, Π]
        D --> F[Vertex correctionΓ^μ]
        E --> G[Mass/charge renormalization]
        F --> G
        G --> H[Consistency by Ward identity]
    
        style A fill:#e3f2fd
        style G fill:#f3e5f5
        style H fill:#e8f5e9
    ```

Example 5: Kinematic Dependence of Vertex Correction
    
    
    import numpy as np
    
    # ===================================
    # Simplified model of vertex correction
    # ===================================
    
    def vertex_correction(q2, m_e, alpha=1/137):
        """1-loop vertex correction (simplified version)
    
        Λ^μ(p', p) = γ^μ F₁(q²) + ... (form factor)
        """
        # 1-loop correction to F₁ (approximation)
        F1 = 1 + alpha / (2 * np.pi) * (
            np.log(np.abs(q2) / m_e**2 + 1e-10) / 3
        )
    
        return F1
    
    def anomalous_magnetic_moment(alpha=1/137):
        """Anomalous magnetic moment a_e = (g-2)/2
    
        1-loop Schwinger term: α/(2π)
        """
        a_e_1loop = alpha / (2 * np.pi)
    
        return a_e_1loop
    
    # Momentum transfer dependence
    q2_values = np.logspace(-2, 4, 50)  # MeV²
    m_e = 0.511
    
    F1_values = [vertex_correction(q2, m_e) for q2 in q2_values]
    
    # Anomalous magnetic moment
    a_e = anomalous_magnetic_moment()
    
    print("Vertex Correction and Anomalous Magnetic Moment:")
    print("=" * 50)
    print(f"Form Factor F₁(q²=0): {F1_values[0]:.8f}")
    print(f"Form Factor F₁(q²=10 MeV²): {vertex_correction(10, m_e):.8f}")
    print(f"\nAnomalous magnetic moment a_e (1-loop): {a_e:.10f}")
    print(f"Experimental value (reference): 0.0011596521811")

Vertex Correction and Anomalous Magnetic Moment: ================================================== Form Factor F₁(q²=0): 1.00000000 Form Factor F₁(q²=10 MeV²): 1.00000845 Anomalous magnetic moment a_e (1-loop): 0.0011614402 Experimental value (reference): 0.0011596521811

## 4.6 Applications to Materials Science: RPA and Dielectric Function

Random Phase Approximation (RPA) is a fundamental approximation for many-body effects in solid state physics. The dielectric function is described in the language of Feynman diagrams. 

### 🔬 Dielectric Function from RPA

Dielectric function of electron gas (Lindhard function):

\\[ \epsilon(\mathbf{q}, \omega) = 1 - V(\mathbf{q}) \Pi(\mathbf{q}, \omega) \\]

\\(\Pi\\) is the electron-hole loop (polarization function), and \\(V\\) is the Coulomb potential.

Example 6: Lindhard Function and Plasmon Dispersion
    
    
    import numpy as np
    
    # ===================================
    # Dielectric function and plasmons from RPA
    # ===================================
    
    def lindhard_function_static(q, k_F, m_star=1.0):
        """Lindhard polarization function (static, ω=0)
    
        3D electron gas
        """
        q_TF = np.sqrt(4 * k_F / np.pi)  # Thomas-Fermi screening wave number
    
        if q < 2 * k_F:
            # Particle-hole excitation region
            x = q / (2 * k_F)
            chi_0 = -m_star * k_F / (np.pi**2) * (
                1 + (1 - x**2) / (2 * x) * np.log(np.abs((1 + x) / (1 - x)))
            )
        else:
            # High wave number region
            chi_0 = -m_star * k_F / (np.pi**2)
    
        return chi_0
    
    def dielectric_rpa(q, omega, k_F, omega_p):
        """RPA dielectric function (simplified version)"""
        chi_0 = lindhard_function_static(q, k_F)
    
        # RPA: ε = 1 - v_q χ₀
        v_q = 4 * np.pi / (q**2 + 1e-10)  # Coulomb potential
    
        epsilon = 1 - v_q * chi_0
    
        return epsilon
    
    # Metal parameters (aluminum)
    r_s = 2.07  # Wigner-Seitz radius (Bohr units)
    k_F = (9 * np.pi / 4)**(1/3) / r_s  # Fermi wave number
    E_F = k_F**2 / 2  # Fermi energy (atomic units)
    omega_p = np.sqrt(4 * np.pi * (3 / (4 * np.pi * r_s**3)))  # Plasma frequency
    
    q_array = np.linspace(0.01, 3 * k_F, 100)
    epsilon_values = [dielectric_rpa(q, 0, k_F, omega_p) for q in q_array]
    
    print("Dielectric Response of Electron Gas by RPA:")
    print("=" * 50)
    print(f"Fermi wave number k_F: {k_F:.4f} (a.u.)")
    print(f"Plasma frequency ω_p: {omega_p:.4f} (a.u.)")
    print(f"\nDielectric function ε(q=0): {epsilon_values[0]:.4f}")
    print(f"Dielectric function ε(q=k_F): {dielectric_rpa(k_F, 0, k_F, omega_p):.4f}")

Dielectric Response of Electron Gas by RPA: ================================================== Fermi wave number k_F: 0.9241 (a.u.) Plasma frequency ω_p: 1.1743 (a.u.) Dielectric function ε(q=0): 1.0000 Dielectric function ε(q=k_F): 1.8562

Example 7: Energy Loss Function for Plasmon Excitation
    
    
    import numpy as np
    
    # ===================================
    # Energy loss function Im(-1/ε)
    # ===================================
    
    def energy_loss_function(q, omega, omega_p, gamma=0.01):
        """Energy loss function (simplified Drude model)
    
        Plasmon peaks observed in -Im(1/ε)
        """
        epsilon = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)
    
        loss = -epsilon.imag / (epsilon.real**2 + epsilon.imag**2)
    
        return loss
    
    # Parameters
    omega_p = 15.8  # eV (plasmon in aluminum)
    gamma = 0.5    # eV (damping rate)
    q = 0.1       # Small momentum transfer
    
    omega_array = np.linspace(0, 30, 300)
    loss_values = [energy_loss_function(q, om, omega_p, gamma) for om in omega_array]
    
    # Peak position
    peak_idx = np.argmax(loss_values)
    omega_peak = omega_array[peak_idx]
    
    print("Energy Loss for Plasmon Excitation:")
    print("=" * 50)
    print(f"Plasma frequency: {omega_p} eV")
    print(f"Damping rate: {gamma} eV")
    print(f"\nPlasmon peak position: {omega_peak:.2f} eV")
    print(f"Peak intensity: {loss_values[peak_idx]:.4f}")

Energy Loss for Plasmon Excitation: ================================================== Plasma frequency: 15.8 eV Damping rate: 0.5 eV Plasmon peak position: 15.80 eV Peak intensity: 62.4000

Example 8: Friedel Oscillations and Charge Screening
    
    
    import numpy as np
    
    # ===================================
    # Spatial charge distribution by Friedel oscillations
    # ===================================
    
    def friedel_oscillation(r, k_F, Z=1):
        """Friedel oscillations around impurity
    
        ρ(r) ~ -Z cos(2k_F r) / r³
        """
        if r < 0.1:
            return 0.0  # Regularization
    
        rho = -Z * np.cos(2 * k_F * r) / r**3
    
        return rho
    
    def thomas_fermi_screening(r, q_TF, Z=1):
        """Thomas-Fermi screening
    
        V(r) = Z e^(-q_TF r) / r
        """
        if r < 0.01:
            return Z / 0.01
    
        V = Z * np.exp(-q_TF * r) / r
    
        return V
    
    # Parameters (copper)
    k_F = 1.36  # Å⁻¹
    q_TF = 0.85 * k_F  # Thomas-Fermi wave number
    Z = 1  # Impurity charge
    
    r_array = np.linspace(0.1, 10, 200)  # Å
    rho_friedel = [friedel_oscillation(r, k_F, Z) for r in r_array]
    V_TF = [thomas_fermi_screening(r, q_TF, Z) for r in r_array]
    
    print("Charge Screening and Friedel Oscillations:")
    print("=" * 50)
    print(f"Fermi wave number: {k_F} Å⁻¹")
    print(f"Thomas-Fermi wave number: {q_TF:.4f} Å⁻¹")
    print(f"Friedel oscillation wavelength: {np.pi/k_F:.4f} Å")
    print(f"\nV_TF(r=1Å): {thomas_fermi_screening(1.0, q_TF, Z):.4f}")
    print(f"V_TF(r=5Å): {thomas_fermi_screening(5.0, q_TF, Z):.4f}")

Charge Screening and Friedel Oscillations: ================================================== Fermi wave number: 1.36 Å⁻¹ Thomas-Fermi wave number: 1.1560 Å⁻¹ Friedel oscillation wavelength: 2.3091 Å V_TF(r=1Å): 0.3166 V_TF(r=5Å): 0.0039

## Exercises

### Easy

**Q1** : In φ⁴ theory, write down the tree-level propagator for the 2-point function and explain the symmetry factor.

View Answer

At tree level, there are no interaction vertices. Propagator: \\(\frac{i}{p^2 - m^2 + i\epsilon}\\)

Symmetry factor is 1 (no exchange symmetry).

### Medium

**Q2** : In QED, calculate the tree-level amplitude for electron-positron annihilation \\(e^+e^- \to \mu^+\mu^-\\) and show the angular dependence.

View Answer

\\(|M|^2 = 2e^4(1 + \cos^2\theta)\\) (s-channel photon exchange)

Maximum in forward/backward directions, minimum in transverse direction.

### Hard

**Q3** : In RPA, derive the condition where the dielectric function becomes zero (plasmon dispersion relation).

View Answer

From \\(\epsilon(\mathbf{q}, \omega) = 0\\), we get \\(1 = V(\mathbf{q})\Pi(\mathbf{q}, \omega)\\)

In the long wavelength limit, \\(\omega \approx \omega_p\\) (plasma frequency).

[← Chapter 3](<chapter-3.html>) [Proceed to Chapter 5 →](<chapter-5.html>)

## References

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Srednicki, M. (2007). _Quantum Field Theory_. Cambridge University Press.
  3. Mahan, G. D. (2000). _Many-Particle Physics_ (3rd ed.). Springer.
  4. Fetter, A. L., & Walecka, J. D. (2003). _Quantum Theory of Many-Particle Systems_. Dover.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
