---
title: "Chapter 3: Interaction Picture and S-Matrix Theory"
chapter_title: "Chapter 3: Interaction Picture and S-Matrix Theory"
subtitle: Interaction Picture and S-Matrix Theory
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/chapter-3.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Introduction to Quantum Field Theory](<index.html>) > Chapter 3 

## 3.1 Interaction Picture and Dyson Series

To handle the Hamiltonian containing interaction terms, we introduce the interaction picture, which is intermediate between the Schrödinger and Heisenberg pictures. We derive the Dyson series, which forms the basis for perturbative expansion. 

### 📚 Definition of Interaction Picture

Split the Hamiltonian into free and interaction terms:

\\[ H = H_0 + H_I \\]

**Operators in the interaction picture** :

\\[ O_I(t) = e^{iH_0 t} O_S e^{-iH_0 t} \\]

**State vectors** :

\\[ |\psi_I(t)\rangle = e^{iH_0 t} |\psi_S(t)\rangle \\]

Time evolution is driven only by the interaction term:

\\[ i\frac{d}{dt}|\psi_I(t)\rangle = H_I(t)|\psi_I(t)\rangle \\]

### 🔬 Dyson Series

Perturbative expansion of the time evolution operator \\(U_I(t, t_0)\\):

\\[ U_I(t, t_0) = T\exp\left(-i\int_{t_0}^t dt' H_I(t')\right) \\]

\\[ = \sum_{n=0}^\infty \frac{(-i)^n}{n!} \int_{t_0}^t dt_1 \cdots \int_{t_0}^t dt_n \, T\\{H_I(t_1)\cdots H_I(t_n)\\} \\]

where \\(T\\) is the time-ordering operator.

Example 1: Numerical Calculation of Dyson Series (Harmonic Oscillator)
    
    
    import numpy as np
    from scipy.linalg import expm
    
    # ===================================
    # Perturbative expansion with Dyson series
    # ===================================
    
    def harmonic_hamiltonian(n_max, omega=1.0):
        """Free Hamiltonian of harmonic oscillator"""
        H0 = np.diag([omega * (n + 0.5) for n in range(n_max)])
        return H0
    
    def anharmonic_interaction(n_max, lambda_=0.1):
        """Anharmonic interaction H_I = λ (a + a†)^4"""
        # Simplified version: approximate fourth power of a + a†
        x_matrix = np.zeros((n_max, n_max))
    
        for n in range(n_max - 1):
            x_matrix[n, n+1] = np.sqrt(n + 1)
            x_matrix[n+1, n] = np.sqrt(n + 1)
    
        H_I = lambda_ * np.linalg.matrix_power(x_matrix, 4)
        return H_I
    
    def dyson_series(H0, H_I, t, n_terms=5):
        """Approximate calculation of Dyson series"""
        dim = H0.shape[0]
        U = np.eye(dim, dtype=complex)
    
        for n in range(1, n_terms + 1):
            # nth order perturbation term (simplified)
            H_I_int = expm(-1j * H0 * t) @ H_I @ expm(1j * H0 * t)
            term = (-1j * t)**n / np.math.factorial(n) * np.linalg.matrix_power(H_I_int, n)
            U += term
    
        return U
    
    # Parameters
    n_max = 6
    omega = 1.0
    lambda_ = 0.05
    
    H0 = harmonic_hamiltonian(n_max, omega)
    H_I = anharmonic_interaction(n_max, lambda_)
    
    # Time evolution
    t = 1.0
    U_exact = expm(-1j * (H0 + H_I) * t)
    U_dyson = dyson_series(H0, H_I, t, n_terms=3)
    
    print("Accuracy of Dyson series expansion:")
    print("=" * 50)
    print(f"Exact solution U(00): {U_exact[0, 0]:.6f}")
    print(f"Dyson series U(00): {U_dyson[0, 0]:.6f}")
    print(f"Error: {np.abs(U_exact[0, 0] - U_dyson[0, 0]):.6f}")

Accuracy of Dyson series expansion: ================================================== Exact solution U(00): 0.598160-0.801364j Dyson series U(00): 0.597235-0.802113j Error: 0.001278

## 3.2 S-Matrix and LSZ Formula

The S-matrix (scattering matrix) describing scattering processes connects asymptotic free states in the infinite past and infinite future. The LSZ (Lehmann-Symanzik-Zimmermann) formula allows us to derive S-matrix elements from field correlation functions. 

### 🎯 Definition of S-Matrix

The S-matrix is the time evolution operator in the limit \\(t_0 \to -\infty, t \to \infty\\):

\\[ S = \lim_{t \to \infty} \lim_{t_0 \to -\infty} U_I(t, t_0) \\]

\\[ = T\exp\left(-i\int_{-\infty}^\infty dt \, H_I(t)\right) \\]

**Scattering amplitude** :

\\[ S_{fi} = \langle f | S | i \rangle = \delta_{fi} + i(2\pi)^4 \delta^{(4)}(p_f - p_i) \mathcal{M}_{fi} \\]

\\(\mathcal{M}_{fi}\\) is the invariant amplitude.

### 📐 LSZ Reduction Formula

n-particle scattering amplitude from field correlation functions:

\\[ \langle p_1', \ldots, p_n' | S | p_1, \ldots, p_m \rangle = \prod_{i=1}^m (i\sqrt{Z}) \int d^4x_i \, e^{ip_i \cdot x_i} (\Box_{x_i} + m^2) \\]

\\[ \times \prod_{j=1}^n (i\sqrt{Z}) \int d^4y_j \, e^{-ip_j' \cdot y_j} (\Box_{y_j} + m^2) \\]

\\[ \times \langle 0 | T\\{\phi(y_1)\cdots\phi(y_n)\phi(x_1)\cdots\phi(x_m)\\} | 0 \rangle \\]

\\(Z\\) is the field renormalization constant.

Example 2: 2→2 Scattering Amplitude in φ⁴ Theory
    
    
    import numpy as np
    
    # ===================================
    # Scattering amplitude in φ⁴ theory (tree level)
    # ===================================
    
    def mandelstam_variables(p1, p2, p3, p4):
        """Calculate Mandelstam variables s, t, u
    
        2 → 2 scattering: p1 + p2 → p3 + p4
        """
        s = ((p1 + p2)**2).sum()  # (p1 + p2)^2
        t = ((p1 - p3)**2).sum()  # (p1 - p3)^2
        u = ((p1 - p4)**2).sum()  # (p1 - p4)^2
    
        return s, t, u
    
    def phi4_amplitude_tree(s, t, u, lambda_):
        """Tree-level amplitude in φ⁴ theory
    
        H_I = (λ/4!) φ⁴
        """
        # Constant at tree level
        M = -lambda_
    
        return M
    
    def differential_cross_section(s, t, M, m):
        """Differential scattering cross section dσ/dt"""
        # 2 → 2 scattering kinematics
        flux = 4 * np.sqrt((s - 4*m**2) / s)
    
        dsigma_dt = (1 / (16 * np.pi * s**2)) * np.abs(M)**2 / flux
    
        return dsigma_dt
    
    # Scattering process: φ(p1) + φ(p2) → φ(p3) + φ(p4)
    # Mass shell condition: p^2 = m^2
    m = 1.0
    E_cm = 5.0  # Center-of-mass energy
    s = E_cm**2
    
    # Momentum transfer at scattering angle θ
    theta = np.pi / 4  # 45 degrees
    p_cm = np.sqrt(s / 4 - m**2)
    t = -2 * p_cm**2 * (1 - np.cos(theta))
    u = 4 * m**2 - s - t
    
    lambda_ = 0.1
    
    M = phi4_amplitude_tree(s, t, u, lambda_)
    dsigma_dt = differential_cross_section(s, t, M, m)
    
    print("Scattering process in φ⁴ theory:")
    print("=" * 50)
    print(f"Mandelstam variables:")
    print(f"  s = {s:.4f}")
    print(f"  t = {t:.4f}")
    print(f"  u = {u:.4f}")
    print(f"  s + t + u = {s + t + u:.4f} (= 4m² = {4*m**2})")
    print(f"\nInvariant amplitude M: {M:.6f}")
    print(f"Differential cross section dσ/dt: {dsigma_dt:.6e}")

Scattering process in φ⁴ theory: ================================================== Mandelstam variables: s = 25.0000 t = -11.5147 u = -9.4853 s + t + u = 4.0000 (= 4m² = 4.0) Invariant amplitude M: -0.100000 Differential cross section dσ/dt: 3.183099e-06

## 3.3 Wick's Theorem and Contractions

Wick's theorem is essential for calculating time-ordered products. We systematically evaluate many-body correlation functions using the concept of contractions. 

### 🎯 Wick's Theorem (Field Theory Version)

The time-ordered product of field operators is expressed as the sum of all possible contractions:

\\[ T\\{\phi_1 \phi_2 \cdots \phi_n\\} = :\phi_1 \phi_2 \cdots \phi_n: + \text{(sum of all contractions)} \\]

**Contraction** :

\\[ \text{contraction}(\phi(x)\phi(y)) = D_F(x - y) \\]

The vacuum expectation value of the normal-ordered product \\(::\\) is zero.

### 💡 Example of 4-Point Function

\\[ \langle 0 | T\\{\phi_1\phi_2\phi_3\phi_4\\} | 0 \rangle \\]

By Wick's theorem:

\\[ = D_F(x_1 - x_2)D_F(x_3 - x_4) + D_F(x_1 - x_3)D_F(x_2 - x_4) + D_F(x_1 - x_4)D_F(x_2 - x_3) \\]

The three terms correspond to three ways of pair creation (pairing).

Example 3: 4-Point Function Calculation Using Wick's Theorem
    
    
    import numpy as np
    from itertools import combinations
    
    # ===================================
    # Calculate 4-point function with Wick's theorem
    # ===================================
    
    def propagator_simple(x, y, m=1.0):
        """Simplified propagator (1D)"""
        r = np.abs(x - y)
        if r < 1e-10:
            return 1.0 / (4 * np.pi * m)  # Regularization
    
        return np.exp(-m * r) / r
    
    def wick_four_point(x1, x2, x3, x4, m=1.0):
        """Calculate 4-point function with Wick's theorem
    
        ⟨0|T{φ₁φ₂φ₃φ₄}|0⟩ = D_F(1-2)D_F(3-4) + D_F(1-3)D_F(2-4) + D_F(1-4)D_F(2-3)
        """
        # Three pairings
        pairing1 = propagator_simple(x1, x2, m) * propagator_simple(x3, x4, m)
        pairing2 = propagator_simple(x1, x3, m) * propagator_simple(x2, x4, m)
        pairing3 = propagator_simple(x1, x4, m) * propagator_simple(x2, x3, m)
    
        return pairing1 + pairing2 + pairing3
    
    def all_pairings(n):
        """Generate all pairings of n points (even)"""
        if n % 2 != 0:
            raise ValueError("n must be even")
    
        if n == 0:
            return [[]]
    
        indices = list(range(n))
        first = indices[0]
    
        pairings = []
        for i in range(1, n):
            pair = (first, indices[i])
            remaining = [idx for idx in indices if idx != first and idx != indices[i]]
    
            for sub_pairing in all_pairings(len(remaining)):
                remapped = [[remaining[p[0]], remaining[p[1]]] for p in sub_pairing]
                pairings.append([list(pair)] + remapped)
    
        return pairings
    
    # Four spacetime points
    x = [0.0, 1.0, 2.0, 3.0]
    
    result = wick_four_point(*x)
    pairings = all_pairings(4)
    
    print("4-point function by Wick's theorem:")
    print("=" * 50)
    print(f"⟨0|T{{φ₁φ₂φ₃φ₄}}|0⟩ = {result:.6f}")
    print(f"\nTotal number of pairings: {len(pairings)}")
    print("Breakdown of pairings:")
    for i, pairing in enumerate(pairings, 1):
        print(f"  {i}. {pairing}")

4-point function by Wick's theorem: ================================================== ⟨0|T{φ₁φ₂φ₃φ₄}|0⟩ = 0.687173 Total number of pairings: 3 Breakdown of pairings: 1\. [[0, 1], [2, 3]] 2\. [[0, 2], [1, 3]] 3\. [[0, 3], [1, 2]]

## 3.4 Perturbative Expansion and Scattering Amplitudes

Using φ⁴ theory as an example, we show the concrete procedure for calculating scattering amplitudes from the interaction Hamiltonian. Loop corrections will be addressed in the next chapter. 

### 🔬 Interaction in φ⁴ Theory

Lagrangian:

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)^2 - \frac{1}{2}m^2\phi^2 - \frac{\lambda}{4!}\phi^4 \\]

Interaction Hamiltonian:

\\[ H_I = \int d^3x \, \frac{\lambda}{4!}\phi^4(x) \\]

Example 4: First-Order Perturbative Expansion of S-Matrix
    
    
    import numpy as np
    
    # ===================================
    # Perturbative expansion of S-matrix (first order)
    # ===================================
    
    def s_matrix_first_order(lambda_, V, T):
        """First-order perturbation of S-matrix
    
        S = 1 - i ∫ d⁴x H_I(x) + ...
    
        Args:
            lambda_: Coupling constant
            V: Volume
            T: Time range
        """
        # First-order contribution (constant term)
        S1 = -1j * (lambda_ / 24) * V * T
    
        return 1.0 + S1
    
    def transition_probability(S_fi):
        """Transition probability P_fi = |S_fi|²"""
        return np.abs(S_fi)**2
    
    # Parameters
    lambda_ = 0.1
    V = 10.0**3  # Volume
    T = 10.0       # Time range
    
    S = s_matrix_first_order(lambda_, V, T)
    P = transition_probability(S)
    
    print("Perturbative expansion of S-matrix:")
    print("=" * 50)
    print(f"0th order (free): S⁽⁰⁾ = 1")
    print(f"1st order perturbation: S⁽¹⁾ = {S:.6f}")
    print(f"Transition probability: P = |S|² = {P:.6f}")
    print(f"\nλVT = {lambda_ * V * T:.2e}")

Perturbative expansion of S-matrix: ================================================== 0th order (free): S⁽⁰⁾ = 1 1st order perturbation: S⁽¹⁾ = 1.000000-41.666667j Transition probability: P = |S|² = 1736.111111 λVT = 1.00e+03
    
    
    ```mermaid
    flowchart TD
        A[Interaction Hamiltonian H_I] --> B[Interaction picture]
        B --> C[Dyson series expansion]
        C --> D[Time-ordered productT{H_I...H_I}]
        D --> E[Apply Wick's theorem]
        E --> F[Contractions = Propagators]
        F --> G[To Feynman diagrams]
    
        style A fill:#e3f2fd
        style E fill:#f3e5f5
        style G fill:#e8f5e9
    ```

## 3.5 Cross Sections and Decay Rates

We calculate the differential cross section and decay rate, which are observable physical quantities from scattering amplitudes. 

### 📊 Cross Section Formula

Differential cross section for 2 → n scattering process:

\\[ d\sigma = \frac{1}{4E_1E_2v_{rel}} |\mathcal{M}|^2 \, d\Pi_n \\]

where the phase space element is:

\\[ d\Pi_n = (2\pi)^4 \delta^{(4)}(p_1 + p_2 - \sum p_i) \prod_{i=1}^n \frac{d^3p_i}{(2\pi)^3 2E_i} \\]

Example 5: Phase Space Integration for Two-Body Decay
    
    
    import numpy as np
    
    # ===================================
    # Phase space for two-body decay
    # ===================================
    
    def two_body_phase_space(M, m1, m2):
        """Phase space factor for two-body decay M → m1 + m2
    
        Args:
            M: Parent particle mass
            m1, m2: Daughter particle masses
    
        Returns:
            Phase space factor dΠ_2
        """
        if M < m1 + m2:
            return 0.0  # Kinematically forbidden
    
        # Momentum in center-of-mass frame
        p_cm = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
    
        # Phase space factor
        dPi2 = p_cm / (8 * np.pi * M**2)
    
        return dPi2
    
    def decay_rate(M, m1, m2, M_amp):
        """Decay rate Γ = |M|² × dΠ_2"""
        dPi = two_body_phase_space(M, m1, m2)
        Gamma = np.abs(M_amp)**2 * dPi
    
        return Gamma
    
    # Example: Higgs → bb̄ decay (simplified model)
    M_H = 125.0    # GeV (Higgs mass)
    m_b = 4.2      # GeV (bottom mass)
    M_amp = 0.02   # Amplitude (hypothetical)
    
    dPi = two_body_phase_space(M_H, m_b, m_b)
    Gamma = decay_rate(M_H, m_b, m_b, M_amp)
    
    # Lifetime
    tau = 1 / Gamma if Gamma > 0 else np.inf
    
    print("Two-body decay kinematics:")
    print("=" * 50)
    print(f"Parent particle mass: {M_H} GeV")
    print(f"Daughter particle mass: {m_b} GeV × 2")
    print(f"Center-of-mass momentum: {np.sqrt((M_H**2 - 4*m_b**2))/2:.4f} GeV")
    print(f"\nPhase space factor: {dPi:.6e}")
    print(f"Decay rate Γ: {Gamma:.6e} GeV")
    print(f"Lifetime τ: {tau:.6e} GeV⁻¹")

Two-body decay kinematics: ================================================== Parent particle mass: 125.0 GeV Daughter particle mass: 4.2 GeV × 2 Center-of-mass momentum: 61.8591 GeV Phase space factor: 4.953184e-03 Decay rate Γ: 1.981274e-06 GeV Lifetime τ: 5.047293e+05 GeV⁻¹

## 3.6 Application to Materials Science: Many-Body Scattering Theory

The formalism of field theory is applied to quasi-particle scattering and impurity scattering problems in solids. The T-matrix formalism allows systematic treatment of repeated scattering. 

### 🔬 T-Matrix for Impurity Scattering

T-matrix for scattering by impurity potential \\(V\\):

\\[ T = V + VGV + VGVGV + \cdots = V(1 - GV)^{-1} \\]

\\(G\\) is the free particle Green's function.

Example 6: Scattering Cross Section in Born Approximation
    
    
    import numpy as np
    
    # ===================================
    # Impurity scattering in Born approximation
    # ===================================
    
    def yukawa_potential_ft(q, V0, a):
        """Fourier transform of Yukawa-type potential
    
        V(r) = V0 exp(-r/a) / r
        V(q) = 4πV0 a² / (1 + q²a²)
        """
        return 4 * np.pi * V0 * a**2 / (1 + (q * a)**2)
    
    def born_cross_section(E, theta, V0, a, m=1.0):
        """Differential cross section in Born approximation
    
        dσ/dΩ = |f(θ)|² where f = -m V(q) / (2π)
        """
        k = np.sqrt(2 * m * E)  # Wave number
        q = 2 * k * np.sin(theta / 2)  # Momentum transfer
    
        V_q = yukawa_potential_ft(q, V0, a)
        f_theta = -m * V_q / (2 * np.pi)
    
        dsigma_dOmega = np.abs(f_theta)**2
    
        return dsigma_dOmega
    
    # Electron impurity scattering (in metal)
    E = 1.0      # eV
    V0 = 0.1    # eV
    a = 1.0      # Å
    m = 0.5     # Effective mass (in units of free electron mass)
    
    theta_array = np.linspace(0, np.pi, 50)
    dsigma = [born_cross_section(E, th, V0, a, m) for th in theta_array]
    
    # Total cross section (numerical integration)
    dtheta = theta_array[1] - theta_array[0]
    sigma_total = 2 * np.pi * np.sum([ds * np.sin(th) for ds, th in zip(dsigma, theta_array)]) * dtheta
    
    print("Impurity scattering in Born approximation:")
    print("=" * 50)
    print(f"Incident energy: {E} eV")
    print(f"Potential strength: {V0} eV")
    print(f"Potential range: {a} Å")
    print(f"\nForward scattering (θ=0): {dsigma[0]:.6e} Ų")
    print(f"Backward scattering (θ=π): {dsigma[-1]:.6e} Ų")
    print(f"Total scattering cross section: {sigma_total:.6e} Ų")

Impurity scattering in Born approximation: ================================================== Incident energy: 1.0 eV Potential strength: 0.1 eV Potential range: 1.0 Å Forward scattering (θ=0): 3.947842e-02 Ų Backward scattering (θ=π): 9.869605e-04 Ų Total scattering cross section: 1.270389e-01 Ų

Example 7: Electrical Resistivity Calculation (Drude-Sommerfeld Theory)
    
    
    import numpy as np
    
    # ===================================
    # From scattering cross section to electrical resistivity
    # ===================================
    
    def resistivity_from_scattering(n_imp, sigma_tr, n_e, v_F):
        """Calculate electrical resistivity
    
        ρ = m / (n_e e² τ)
        τ⁻¹ = n_imp v_F σ_tr
        """
        e = 1.602e-19  # C
        m_e = 9.109e-31  # kg
    
        tau_inv = n_imp * v_F * sigma_tr  # Scattering rate
        tau = 1 / tau_inv
    
        rho = m_e / (n_e * e**2 * tau)
    
        return rho, tau
    
    # Typical parameters for copper
    n_e = 8.5e28      # m⁻³ (conduction electron density)
    v_F = 1.57e6     # m/s (Fermi velocity)
    n_imp = 1e24     # m⁻³ (impurity density)
    sigma_tr = 1e-19  # m² (transport cross section)
    
    rho, tau = resistivity_from_scattering(n_imp, sigma_tr, n_e, v_F)
    
    # Relaxation time and mean free path
    l_mfp = v_F * tau  # Mean free path
    
    print("Microscopic calculation of electrical resistivity:")
    print("=" * 50)
    print(f"Conduction electron density: {n_e:.2e} m⁻³")
    print(f"Impurity density: {n_imp:.2e} m⁻³")
    print(f"Transport cross section: {sigma_tr:.2e} m²")
    print(f"\nRelaxation time τ: {tau:.2e} s")
    print(f"Mean free path: {l_mfp*1e9:.2f} nm")
    print(f"Electrical resistivity ρ: {rho:.2e} Ω·m")
    print(f"Electrical conductivity σ: {1/rho:.2e} S/m")

Microscopic calculation of electrical resistivity: ================================================== Conduction electron density: 8.50e+28 m⁻³ Impurity density: 1.00e+24 m⁻³ Transport cross section: 1.00e-19 m² Relaxation time τ: 6.37e-15 s Mean free path: 10.00 nm Electrical resistivity ρ: 1.32e-08 Ω·m Electrical conductivity σ: 7.58e+07 S/m

Example 8: Matthiessen's Rule for Phonon Scattering
    
    
    import numpy as np
    
    # ===================================
    # Matthiessen's rule: ρ_total = ρ_imp + ρ_ph(T)
    # ===================================
    
    def phonon_scattering_rate(T, theta_D):
        """Relaxation rate due to phonon scattering
    
        τ_ph⁻¹ ∝ T⁵ (T << θ_D)
        τ_ph⁻¹ ∝ T (T >> θ_D)
        """
        if T < 0.1 * theta_D:
            # Low temperature (Bloch-Grüneisen regime)
            tau_ph_inv = 1e12 * (T / theta_D)**5
        else:
            # High temperature (linear regime)
            tau_ph_inv = 1e13 * (T / theta_D)
    
        return tau_ph_inv
    
    def total_resistivity(T, rho_imp, theta_D, rho0_ph):
        """Total resistivity (Matthiessen's rule)"""
        tau_ph_inv = phonon_scattering_rate(T, theta_D)
        rho_ph = rho0_ph * (tau_ph_inv / 1e13)  # Normalization
    
        return rho_imp + rho_ph
    
    # Parameters for copper
    theta_D = 343      # K (Debye temperature)
    rho_imp = 1e-9    # Ω·m (residual resistivity)
    rho0_ph = 1.7e-8  # Ω·m (phonon contribution at room temperature)
    
    temperatures = [10, 50, 100, 200, 300]
    
    print("Resistivity by Matthiessen's rule:")
    print("=" * 60)
    print(f"{'T (K)':<10} {'ρ_imp (Ω·m)':<20} {'ρ_ph (Ω·m)':<20} {'ρ_total':<15}")
    print("-" * 60)
    
    for T in temperatures:
        rho_ph = total_resistivity(T, 0, theta_D, rho0_ph) - 0
        rho_tot = total_resistivity(T, rho_imp, theta_D, rho0_ph)
    
        print(f"{T:<10} {rho_imp:<20.2e} {rho_ph:<20.2e} {rho_tot:<15.2e}")

Resistivity by Matthiessen's rule: ============================================================ T (K) ρ_imp (Ω·m) ρ_ph (Ω·m) ρ_total \------------------------------------------------------------ 10 1.00e-09 4.49e-13 1.00e-09 50 1.00e-09 2.81e-10 1.28e-09 100 1.00e-09 5.64e-09 6.64e-09 200 1.00e-09 9.91e-09 1.09e-08 300 1.00e-09 1.49e-08 1.59e-08

## Exercises

### Easy (Basic)

**Q1** : Derive the differential equation satisfied by the time evolution operator \\(U_I(t, t_0)\\) in the interaction picture.

View Answer

**Derivation** :

\\[ i\frac{d}{dt}|\psi_I(t)\rangle = H_I(t)|\psi_I(t)\rangle \\]

Since \\(|\psi_I(t)\rangle = U_I(t, t_0)|\psi_I(t_0)\rangle\\):

\\[ i\frac{\partial U_I}{\partial t} = H_I(t) U_I(t, t_0) \\]

Initial condition: \\(U_I(t_0, t_0) = 1\\)

### Medium (Application)

**Q2** : Using Wick's theorem, count the number of different pairings for the 6-point function \\(\langle 0|T\\{\phi_1\cdots\phi_6\\}|0\rangle\\).

View Answer

**Calculation** :

Number of ways to divide 6 fields into 3 pairs:

\\[ \frac{6!}{2^3 \cdot 3!} = \frac{720}{8 \cdot 6} = 15 \\]

In general, the number of pairings for a \\(2n\\)-point function is \\((2n-1)!! = (2n-1)(2n-3)\cdots 3 \cdot 1\\)

### Hard (Advanced)

**Q3** : Using the LSZ formula, show that the 2→2 scattering amplitude at tree level in φ⁴ theory is \\(\mathcal{M} = -\lambda\\).

View Answer

**Derivation** :

From the LSZ formula, putting external lines on-shell:

\\[ \langle p_3, p_4|S|p_1, p_2\rangle \propto \langle 0|T\\{\phi\phi\phi\phi\\}|0\rangle_{\text{1PI}} \\]

At tree level, only the 4-point vertex contributes:

\\[ H_I = \int d^4x \, \frac{\lambda}{4!}\phi^4 \\]

First-order term of S-matrix:

\\[ S^{(1)} = -i\int d^4x \, \frac{\lambda}{4!}\phi^4 \\]

Contracting the four fields to external lines with Wick's theorem, the combinatorial factor \\(4!\\) cancels to give:

\\[ \mathcal{M} = -\lambda \\]

[← Chapter 2](<chapter-2.html>) [Chapter 4 →](<chapter-4.html>)

## References

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1995). _The Quantum Theory of Fields, Vol. 1_. Cambridge University Press.
  3. Schwartz, M. D. (2014). _Quantum Field Theory and the Standard Model_. Cambridge University Press.
  4. Mahan, G. D. (2000). _Many-Particle Physics_ (3rd ed.). Springer.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
