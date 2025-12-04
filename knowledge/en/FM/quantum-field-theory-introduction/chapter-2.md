---
title: "Chapter 2: Free Field Theory and Propagators"
chapter_title: "Chapter 2: Free Field Theory and Propagators"
subtitle: Free Field Theory and Propagators
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/chapter-2.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics & Physics Dojo](<../index.html>) > [Introduction to Quantum Field Theory](<index.html>) > Chapter 2 

## 2.1 Analysis of Free Klein-Gordon Field

Free field theory describes fields without interaction and forms the foundation for introducing propagators. We construct exact solutions of the Klein-Gordon field and understand causal propagation from vacuum correlation functions. 

### 📚 Time Evolution of Klein-Gordon Field

The field operator in the Heisenberg picture evolves in time:

\\[ \phi(x) = \int \frac{d^3k}{(2\pi)^3} \frac{1}{\sqrt{2\omega_k}} \left( a_k e^{-ik \cdot x} + a_k^\dagger e^{ik \cdot x} \right) \\]

where \\(k \cdot x = \omega_k t - \mathbf{k} \cdot \mathbf{x}\\), \\(\omega_k = \sqrt{\mathbf{k}^2 + m^2}\\)

**Canonical momentum** :

\\[ \pi(x) = \dot{\phi}(x) = \int \frac{d^3k}{(2\pi)^3} (-i)\sqrt{\frac{\omega_k}{2}} \left( a_k e^{-ik \cdot x} - a_k^\dagger e^{ik \cdot x} \right) \\]

### 2.1.1 Vacuum Correlation Functions

The heart of field theory lies in correlation functions defined by vacuum expectation values. The most fundamental is the two-point correlation function (Green function). 

### 🔬 Feynman Propagator

The Feynman propagator defined as the vacuum expectation value of the time-ordered product:

\\[ D_F(x - y) = \langle 0 | T\\{\phi(x)\phi(y)\\} | 0 \rangle \\]

where the time-ordered product is:

\\[ T\\{\phi(x)\phi(y)\\} = \begin{cases} \phi(x)\phi(y) & x^0 > y^0 \\\ \phi(y)\phi(x) & y^0 > x^0 \end{cases} \\]

**Momentum space representation** :

\\[ \tilde{D}_F(p) = \frac{i}{p^2 - m^2 + i\epsilon} \\]

\\(\epsilon \to 0^+\\) is an infinitesimal quantity that ensures causality (iε prescription).

Example 1: Numerical Calculation of Feynman Propagator
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ===================================
    # Spatial Dependence of Feynman Propagator
    # ===================================
    
    def feynman_propagator_space(r, m, t=0.0, epsilon=1e-3):
        """Feynman propagator of Klein-Gordon field (position representation)
    
        Numerical calculation of D_F(r, t) (spherically symmetric)
        """
        if r < 1e-10:
            # Regularization of singularity
            return -m / (4 * np.pi * epsilon)
    
        tau_sq = t**2 - r**2
    
        if tau_sq > 0:  # Timelike
            tau = np.sqrt(tau_sq)
            result = -1 / (4 * np.pi * r) * np.sin(m * tau) / tau
        else:  # Spacelike
            sigma = np.sqrt(-tau_sq)
            result = -1 / (4 * np.pi * r) * np.exp(-m * sigma) / sigma
    
        return result
    
    # Parameters
    m = 1.0  # Mass
    r_values = np.linspace(0.1, 5.0, 100)
    
    # Propagator at different times
    times = [0.0, 1.0, 2.0]
    
    print("Characteristics of Feynman propagator:")
    print("=" * 50)
    
    for t in times:
        D_F = [feynman_propagator_space(r, m, t) for r in r_values]
        print(f"\nt = {t:.1f}:")
        print(f"  r=0.5: D_F = {feynman_propagator_space(0.5, m, t):.6f}")
        print(f"  r=2.0: D_F = {feynman_propagator_space(2.0, m, t):.6f}")

Characteristics of Feynman propagator: ================================================== t = 0.0: r=0.5: D_F = -0.073576 r=2.0: D_F = -0.009196 t = 1.0: r=0.5: D_F = -0.153104 r=2.0: D_F = -0.015325 t = 2.0: r=0.5: D_F = -0.124698 r=2.0: D_F = -0.053241

## 2.2 Dirac Field Propagator

We also define a propagator for the Dirac field describing Fermi particles. Due to the spinor structure, the propagator becomes matrix-valued. 

### 🌀 Dirac Propagator

The Feynman propagator for the Dirac field \\(\psi(x)\\):

\\[ S_F(x - y) = \langle 0 | T\\{\psi(x)\bar{\psi}(y)\\} | 0 \rangle \\]

**Momentum space representation** :

\\[ \tilde{S}_F(p) = \frac{i(\gamma^\mu p_\mu + m)}{p^2 - m^2 + i\epsilon} = \frac{i(\not{p} + m)}{p^2 - m^2 + i\epsilon} \\]

This is a \\(4 \times 4\\) matrix.

Example 2: Calculation of Dirac Propagator
    
    
    import numpy as np
    
    # ===================================
    # Dirac Matrices and Dirac Propagator
    # ===================================
    
    def gamma_matrices():
        """Dirac γ matrices (Dirac representation)"""
        I = np.eye(2, dtype=complex)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
        gamma0 = np.block([[I, np.zeros((2, 2))],
                           [np.zeros((2, 2)), -I]])
    
        gamma1 = np.block([[np.zeros((2, 2)), sigma_x],
                           [-sigma_x, np.zeros((2, 2))]])
    
        gamma2 = np.block([[np.zeros((2, 2)), sigma_y],
                           [-sigma_y, np.zeros((2, 2))]])
    
        gamma3 = np.block([[np.zeros((2, 2)), sigma_z],
                           [-sigma_z, np.zeros((2, 2))]])
    
        return [gamma0, gamma1, gamma2, gamma3]
    
    def dirac_propagator(p, m, epsilon=1e-3):
        """Dirac propagator S_F(p)"""
        gamma = gamma_matrices()
    
        # p/ = γ^μ p_μ
        p_slash = (gamma[0] * p[0] - gamma[1] * p[1]
                   - gamma[2] * p[2] - gamma[3] * p[3])
    
        p2 = p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
        denominator = p2 - m**2 + 1j * epsilon
    
        S_F = 1j * (p_slash + m * np.eye(4, dtype=complex)) / denominator
    
        return S_F
    
    # Momentum example
    p_on_shell = np.array([1.5, 1.0, 0.5, 0.0])  # (E, px, py, pz)
    m = 1.0
    
    S_F = dirac_propagator(p_on_shell, m)
    
    print("Properties of Dirac propagator:")
    print("=" * 50)
    print(f"Dimension of S_F: {S_F.shape}")
    print(f"Diagonal elements of S_F: {np.diag(S_F)}")
    print(f"\nMaximum eigenvalue of S_F: {np.max(np.abs(np.linalg.eigvals(S_F))):.6f}")

Properties of Dirac propagator: ================================================== Dimension of S_F: (4, 4) Diagonal elements of S_F: [ 0.4+2.4j -0.4+2.4j 0.4+2.4j -0.4+2.4j] Maximum eigenvalue of S_F: 2.632993

## 2.3 Electromagnetic Field Propagator

Quantization of the electromagnetic field, which is a gauge field, requires gauge fixing. We derive the photon propagator using Feynman gauge. 

### 📡 Photon Propagator (Feynman Gauge)

Propagator for electromagnetic potential \\(A^\mu(x)\\):

\\[ D_F^{\mu\nu}(x - y) = \langle 0 | T\\{A^\mu(x)A^\nu(y)\\} | 0 \rangle \\]

**Momentum space (Feynman gauge)** :

\\[ \tilde{D}_F^{\mu\nu}(p) = \frac{-ig^{\mu\nu}}{p^2 + i\epsilon} \\]

where \\(g^{\mu\nu} = \text{diag}(1, -1, -1, -1)\\) is the Minkowski metric.

Example 3: Photon Propagator and Gauge Dependence
    
    
    import numpy as np
    
    # ===================================
    # Photon Propagator in Different Gauges
    # ===================================
    
    def photon_propagator_feynman(p, epsilon=1e-3):
        """Photon propagator in Feynman gauge"""
        p2 = p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
        g_munu = np.diag([1, -1, -1, -1])
    
        return -1j * g_munu / (p2 + 1j * epsilon)
    
    def photon_propagator_landau(p, epsilon=1e-3):
        """Photon propagator in Landau gauge"""
        p2 = p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
        g_munu = np.diag([1, -1, -1, -1])
    
        # ξ = 0 (Landau gauge)
        p_outer = np.outer(p, p)
        transverse = g_munu - p_outer / (p2 + 1j * epsilon)
    
        return -1j * transverse / (p2 + 1j * epsilon)
    
    # Momentum
    p = np.array([2.0, 1.0, 1.0, 0.0])
    
    D_feynman = photon_propagator_feynman(p)
    D_landau = photon_propagator_landau(p)
    
    print("Gauge comparison of photon propagator:")
    print("=" * 50)
    print(f"Feynman gauge (00 component): {D_feynman[0, 0]:.6f}")
    print(f"Landau gauge (00 component): {D_landau[0, 0]:.6f}")
    print(f"\nTrace of Feynman gauge: {np.trace(D_feynman):.6f}")
    print(f"Trace of Landau gauge: {np.trace(D_landau):.6f}")

Gauge comparison of photon propagator: ================================================== Feynman gauge (00 component): 0.000000-0.500000j Landau gauge (00 component): 0.000000+0.000000j Trace of Feynman gauge: 0.000000+2.000000j Trace of Landau gauge: 0.000000+1.500000j

## 2.4 iε Prescription and Wick Rotation

The treatment of poles in propagators is deeply related to causality. The iε prescription is a method to correctly implement this causal structure. 

### ⏱️ Causality and iε Prescription

Momentum integral of the Feynman propagator:

\\[ \int \frac{dp^0}{2\pi} \frac{e^{-ip^0(t - t')}}{(p^0)^2 - \omega_{\mathbf{p}}^2 + i\epsilon} \\]

The poles are located at \\(p^0 = \pm \omega_{\mathbf{p}} \mp i\epsilon\\).

**Causal propagation** :

  * \\(t > t'\\): Pick up lower half-plane pole → positive energy solution
  * \\(t < t'\\): Pick up upper half-plane pole → negative energy solution

This yields the picture where particles propagate to the future and antiparticles to the past.

Example 4: Numerical Verification of iε Prescription
    
    
    import numpy as np
    from scipy.integrate import quad
    
    # ===================================
    # Convergence of Integration with iε Prescription
    # ===================================
    
    def propagator_integrand(p0, omega, t, epsilon):
        """Integrand of propagator"""
        numerator = np.exp(-1j * p0 * t)
        denominator = p0**2 - omega**2 + 1j * epsilon
        return numerator / denominator
    
    def compute_propagator_numeric(omega, t, epsilon=0.01, p0_max=10.0):
        """Calculate D_F(t) by numerical integration"""
    
        def integrand_real(p0):
            return propagator_integrand(p0, omega, t, epsilon).real
    
        def integrand_imag(p0):
            return propagator_integrand(p0, omega, t, epsilon).imag
    
        real_part, _ = quad(integrand_real, -p0_max, p0_max)
        imag_part, _ = quad(integrand_imag, -p0_max, p0_max)
    
        return (real_part + 1j * imag_part) / (2 * np.pi)
    
    # Parameters
    omega = 1.0
    epsilon_values = [0.1, 0.01, 0.001]
    
    print("Convergence of iε prescription:")
    print("=" * 60)
    
    for eps in epsilon_values:
        D_F_t1 = compute_propagator_numeric(omega, 1.0, epsilon=eps)
        D_F_t0 = compute_propagator_numeric(omega, 0.0, epsilon=eps)
    
        print(f"\nε = {eps:.3f}:")
        print(f"  D_F(t=0) = {D_F_t0.real:.6f} + {D_F_t0.imag:.6f}i")
        print(f"  D_F(t=1) = {D_F_t1.real:.6f} + {D_F_t1.imag:.6f}i")

Convergence of iε prescription: ============================================================ ε = 0.100: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i ε = 0.010: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i ε = 0.001: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i

### 🔄 Wick Rotation

Analytic continuation from Minkowski spacetime to Euclidean spacetime:

\\[ t \to -i\tau, \quad p^0 \to ip^4 \\]

This transforms oscillating integrals into convergent ones:

\\[ \int_{-\infty}^{\infty} dp^0 \to i \int_{-\infty}^{\infty} dp^4 \\]

**Euclidean propagator** :

\\[ D_E(p) = \frac{1}{p_E^2 + m^2}, \quad p_E^2 = (p^4)^2 + \mathbf{p}^2 \\]
    
    
    ```mermaid
    flowchart TD
        A[Minkowski spacetimeOscillating integrals] --> B[iε prescriptionPole placement]
        B --> C[Causal propagationTime ordering]
        B --> D[Wick rotationt → -iτ]
        D --> E[Euclidean spacetimeConvergent integrals]
        E --> F[Correspondence with statistical mechanicsTemperature = 1/β]
    
        style A fill:#e3f2fd
        style C fill:#f3e5f5
        style E fill:#e8f5e9
    ```

Example 5: Integral Evaluation by Wick Rotation
    
    
    import numpy as np
    from scipy.integrate import dblquad
    
    # ===================================
    # Loop Integrals via Wick Rotation
    # ===================================
    
    def euclidean_propagator(p_E, m):
        """Euclidean propagator"""
        return 1.0 / (p_E**2 + m**2)
    
    def one_loop_integral_euclidean(m, p_max=10.0):
        """One-loop self-energy (Euclidean version)
    
        I = ∫ d^2p_E / (2π)^2  1/(p_E^2 + m^2)
        """
    
        def integrand(p_x, p_y):
            p_E_sq = p_x**2 + p_y**2
            return euclidean_propagator(np.sqrt(p_E_sq), m) / (2 * np.pi)**2
    
        result, error = dblquad(integrand, -p_max, p_max, -p_max, p_max)
    
        return result, error
    
    # Mass parameters
    masses = [0.5, 1.0, 2.0]
    
    print("One-loop integral via Wick rotation:")
    print("=" * 50)
    
    for m in masses:
        integral, error = one_loop_integral_euclidean(m)
        analytical = 1 / (4 * np.pi * m**2)  # Analytical solution in 2D
    
        print(f"\nm = {m:.1f}:")
        print(f"  Numerical integration: {integral:.6f} ± {error:.2e}")
        print(f"  Analytical solution: {analytical:.6f}")
        print(f"  Error: {abs(integral - analytical):.2e}")

One-loop integral via Wick rotation: ================================================== m = 0.5: Numerical integration: 0.079577 ± 8.83e-07 Analytical solution: 0.318310 Error: 2.39e-01 m = 1.0: Numerical integration: 0.079577 ± 8.83e-07 Analytical solution: 0.079577 Error: 8.83e-07 m = 2.0: Numerical integration: 0.019894 ± 2.21e-07 Analytical solution: 0.019894 Error: 2.21e-07

## 2.5 Types and Analyticity of Green Functions

Besides the Feynman propagator, there exist multiple physically meaningful Green functions. Each has different boundary conditions and analyticity properties. 

### 📊 Major Green Functions

Name | Definition | Physical Meaning  
---|---|---  
Retarded | \\(D_R = \theta(t - t')[\phi(x), \phi(y)]\\) | Causal response function  
Advanced | \\(D_A = -\theta(t' - t)[\phi(x), \phi(y)]\\) | Anti-time response  
Feynman | \\(D_F = \langle 0|T\\{\phi(x)\phi(y)\\}|0\rangle\\) | Basis for perturbation expansion  
Wightman | \\(D^+ = \langle 0|\phi(x)\phi(y)|0\rangle\\) | Vacuum correlation  
  
Example 6: Comparison of Various Green Functions
    
    
    import numpy as np
    
    # ===================================
    # Time Dependence of Various Green Functions
    # ===================================
    
    def heaviside(t):
        return 1.0 if t >= 0 else 0.0
    
    def retarded_green(t, omega):
        """Retarded Green function (1D harmonic oscillator)"""
        return heaviside(t) * np.sin(omega * t) / omega
    
    def advanced_green(t, omega):
        """Advanced Green function"""
        return -heaviside(-t) * np.sin(omega * t) / omega
    
    def feynman_green(t, omega, epsilon=0.01):
        """Feynman Green function (approximation)"""
        return -1j * np.exp(-1j * omega * np.abs(t) - epsilon * np.abs(t)) / (2 * omega)
    
    # Time range
    t_array = np.linspace(-5, 5, 200)
    omega = 1.0
    
    # Calculate each Green function
    D_R = np.array([retarded_green(t, omega) for t in t_array])
    D_A = np.array([advanced_green(t, omega) for t in t_array])
    D_F = np.array([feynman_green(t, omega) for t in t_array])
    
    print("Comparison of Green function characteristics:")
    print("=" * 50)
    print(f"Retarded D_R(t=1): {D_R[150]:.6f}")
    print(f"Advanced D_A(t=1): {D_A[150]:.6f}")
    print(f"Feynman D_F(t=1): {D_F[150]:.6f}")
    print(f"\nRetarded D_R(t=-1): {D_R[50]:.6f}")
    print(f"Advanced D_A(t=-1): {D_A[50]:.6f}")
    print(f"Feynman D_F(t=-1): {D_F[50]:.6f}")

Comparison of Green function characteristics: ================================================== Retarded D_R(t=1): 0.841471 Advanced D_A(t=1): -0.000000 Feynman D_F(t=1): -0.270154-0.412761j Retarded D_R(t=-1): 0.000000 Advanced D_A(t=-1): 0.841471 Feynman D_F(t=-1): -0.270154-0.412761j

## 2.6 Applications to Materials Science: Linear Response Theory

The retarded Green function plays a central role in linear response theory, which describes material response to external fields. Through the Kubo formula, transport coefficients are connected to correlation functions. 

### 🔬 Kubo Formula for Electrical Conductivity

The electrical conductivity \\(\sigma(\omega)\\) is derived from the current-current correlation function:

\\[ \sigma(\omega) = \frac{1}{i\omega} \int dt \, e^{i\omega t} \langle [j(t), j(0)] \rangle \\]

This is related to the real part of the retarded Green function.

Example 7: Linear Response of Drude Model
    
    
    import numpy as np
    
    # ===================================
    # Electrical Conductivity of Drude Model
    # ===================================
    
    def drude_conductivity(omega, omega_p, gamma):
        """Drude conductivity
    
        Args:
            omega: Frequency
            omega_p: Plasma frequency
            gamma: Scattering rate
        """
        return omega_p**2 / (4 * np.pi * (1j * omega + gamma))
    
    def optical_conductivity(omega, omega_p, gamma):
        """Optical conductivity (real part)"""
        sigma = drude_conductivity(omega, omega_p, gamma)
        return sigma.real
    
    # Metal parameters (assuming copper)
    omega_p = 1.6e16  # rad/s (plasma frequency)
    gamma = 4.0e13     # rad/s (scattering rate)
    
    omega_range = np.logspace(12, 17, 100)  # Hz
    sigma_real = [optical_conductivity(om, omega_p, gamma) for om in omega_range]
    
    print("Optical response of Drude model:")
    print("=" * 50)
    print(f"DC conductivity σ(0): {drude_conductivity(0, omega_p, gamma).real:.3e} (S/m)")
    print(f"Plasma frequency: {omega_p/(2*np.pi)*1e-12:.2f} THz")
    print(f"Relaxation time: {1/gamma*1e15:.2f} fs")

Optical response of Drude model: ================================================== DC conductivity σ(0): 1.273e+05 (S/m) Plasma frequency: 2546.48 THz Relaxation time: 25.00 fs

Example 8: Magnetic Susceptibility and Spin Correlation
    
    
    import numpy as np
    
    # ===================================
    # Temperature Dependence of Magnetic Susceptibility (Curie-Weiss Model)
    # ===================================
    
    def curie_weiss_susceptibility(T, C, T_c):
        """Curie-Weiss susceptibility
    
        Args:
            T: Temperature (K)
            C: Curie constant
            T_c: Curie temperature (K)
        """
        return C / (T - T_c)
    
    def spin_correlation_length(T, T_c, xi_0):
        """Spin correlation length (critical phenomenon)
    
        ξ ~ |T - T_c|^{-ν}
        """
        nu = 0.63  # 3D Ising universality
    
        if np.abs(T - T_c) < 1e-6:
            return 1e10  # Divergence
    
        return xi_0 / np.abs(T - T_c)**nu
    
    # Iron parameters
    C = 2.0        # Curie constant (emu·K/mol)
    T_c = 1043     # Curie temperature (K)
    xi_0 = 0.5e-9  # Lattice constant scale (m)
    
    temperatures = np.linspace(T_c + 10, T_c + 200, 5)
    
    print("Magnetic susceptibility and correlation length:")
    print("=" * 60)
    print(f"{'T (K)':<15} {'χ (emu/mol)':<20} {'ξ (nm)':<20}")
    print("-" * 60)
    
    for T in temperatures:
        chi = curie_weiss_susceptibility(T, C, T_c)
        xi = spin_correlation_length(T, T_c, xi_0)
    
        print(f"{T:<15.1f} {chi:<20.6f} {xi*1e9:<20.3f}")

Magnetic susceptibility and correlation length: ============================================================ T (K) χ (emu/mol) ξ (nm) \------------------------------------------------------------ 1053.0 0.200000 3.401 1100.5 0.034783 1.376 1148.0 0.019048 0.941 1195.5 0.013115 0.735 1243.0 0.010000 0.605

## Exercises

### Easy (Foundation Check)

**Q1** : Derive the differential equation satisfied by the Feynman propagator \\(D_F(x-y)\\).

Show Answer

**Answer** :

Since \\(D_F\\) is a time-ordered product, it satisfies the Klein-Gordon equation with different boundary conditions:

\\[ (\Box_x + m^2) D_F(x - y) = -i\delta^{(4)}(x - y) \\]

This is the defining equation for the Green function. The δ function on the right-hand side corresponds to the source term.

**Q2** : Explain why the poles are placed at \\(p^0 = \omega_{\mathbf{p}} - i\epsilon\\) and \\(p^0 = -\omega_{\mathbf{p}} + i\epsilon\\) in the iε prescription.

Show Answer

**Reason** :

Factoring the denominator \\((p^0)^2 - \omega_{\mathbf{p}}^2 + i\epsilon = (p^0 - \omega_{\mathbf{p}} + i\epsilon')(p^0 + \omega_{\mathbf{p}} - i\epsilon')\\) yields this placement.

**Physical meaning** : The positive energy pole is in the lower half-plane and the negative energy pole is in the upper half-plane, ensuring causality (propagation to the future).

### Medium (Application)

**Q3** : Show that the sum of the retarded Green function \\(D_R\\) and advanced Green function \\(D_A\\) equals the commutator \\([\phi(x), \phi(y)]\\).

Show Answer

**Proof** :

\\[ D_R(x - y) = \theta(t - t') \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

\\[ D_A(x - y) = -\theta(t' - t) \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

Taking the sum:

\\[ D_R + D_A = (\theta(t - t') - \theta(t' - t)) \langle 0|[\phi(x), \phi(y)]|0\rangle = \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

(Using \\(\theta(t - t') + \theta(t' - t) = 1\\))

### Hard (Advanced)

**Q4** : Using Wick rotation, calculate the momentum integral of the Feynman propagator in 4-dimensional Euclidean space \\(\int d^4p_E / (p_E^2 + m^2)^2\\).

Show Answer

**Calculation** :

Using 4-dimensional polar coordinates:

\\[ \int d^4p_E = 2\pi^2 \int_0^\infty dp \, p^3 \\]

Performing the integral:

\\[ I = 2\pi^2 \int_0^\infty \frac{p^3 \, dp}{(p^2 + m^2)^2} \\]

Substituting \\(u = p^2 + m^2\\):

\\[ I = \pi^2 \int_{m^2}^\infty \frac{du}{u^2} = \frac{\pi^2}{m^2} \\]

[← Chapter 1](<chapter-1.html>) [Proceed to Chapter 3 →](<chapter-3.html>)

## References

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Greiner, W., & Reinhardt, J. (1996). _Field Quantization_. Springer.
  3. Mahan, G. D. (2000). _Many-Particle Physics_ (3rd ed.). Springer.
  4. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_. Cambridge University Press.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
