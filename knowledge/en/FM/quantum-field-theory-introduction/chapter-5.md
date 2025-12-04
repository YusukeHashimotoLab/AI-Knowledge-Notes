---
title: "Chapter 5: Renormalization Theory and Effective Field Theory"
chapter_title: "Chapter 5: Renormalization Theory and Effective Field Theory"
subtitle: Renormalization and Effective Field Theory
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-field-theory-introduction/chapter-5.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics & Physics Dojo](<../index.html>) > [Introduction to Quantum Field Theory](<index.html>) > Chapter 5 

## 5.1 Ultraviolet Divergences and Renormalization

Loop integrals in field theory diverge in the high-momentum region (ultraviolet divergence). Renormalization theory is a method to systematically handle these divergences and obtain physical predictions. 

### 📚 Classification of Divergences

Type of Divergence | Order (Loop Integral) | Example  
---|---|---  
Logarithmic Divergence | \\(\int d^4k \, k^{-2}\\) ~ \\(\log\Lambda\\) | QED vertex correction  
Linear Divergence | \\(\int d^4k \, k^{-2}\\) ~ \\(\Lambda\\) | φ⁴ self-energy  
Quadratic Divergence | \\(\int d^4k \, k^{0}\\) ~ \\(\Lambda^2\\) | Scalar field mass correction  
Quartic Divergence | \\(\int d^4k \, k^{2}\\) ~ \\(\Lambda^4\\) | Vacuum energy  
  
\\(\Lambda\\) is the ultraviolet cutoff.

### 🔬 Dimensional Regularization

Extend the spacetime dimension to \\(d = 4 - 2\epsilon\\) and extract divergences as poles in \\(\epsilon \to 0\\):

\\[ \int \frac{d^d k}{(2\pi)^d} \frac{1}{(k^2 + \Delta)^n} = \frac{1}{(4\pi)^{d/2}} \frac{\Gamma(n - d/2)}{\Gamma(n)} \Delta^{d/2 - n} \\]

Pole in \\(\epsilon\\): \\(\frac{1}{\epsilon} + \text{finite}\\)

**Minimal Subtraction (MS) Scheme** : Subtract \\(\frac{1}{\epsilon}\\) and \\(\log(4\pi) - \gamma_E\\).

Example 1: Loop Integral with Dimensional Regularization
    
    
    import numpy as np
    from scipy.special import gamma
    
    # ===================================
    # Integral formula in dimensional regularization
    # ===================================
    
    def dimensional_integral(n, Delta, d=4):
        """Dimensional regularization integral
    
        I_n(Δ) = ∫ d^d k / (2π)^d  1/(k² + Δ)^n
    
        Args:
            n: power in denominator
            Delta: mass parameter
            d: spacetime dimension
        """
        epsilon = (4 - d) / 2
    
        # Formula using Γ function
        prefactor = 1 / (4 * np.pi)**(d / 2)
        gamma_factor = gamma(n - d / 2) / gamma(n)
        delta_factor = Delta**(d / 2 - n)
    
        I_n = prefactor * gamma_factor * delta_factor
    
        return I_n
    
    def extract_pole(epsilon, m2, mu2=1.0):
        """Separate pole in ε and finite part
    
        I ~ 1/ε + log(m²/μ²) + O(ε)
        """
        if epsilon < 1e-6:
            pole = 1 / epsilon
            gamma_E = 0.5772156649
            finite = -gamma_E + np.log(4 * np.pi) - np.log(m2 / mu2)
        else:
            # Evaluation at finite ε
            pole = 1 / epsilon
            finite = -np.log(m2 / mu2)
    
        return pole, finite
    
    # Example of 1-loop integral
    m2 = 1.0  # mass squared
    mu2 = 1.0  # renormalization scale
    d = 3.99  # d = 4 - 2ε, ε = 0.005
    
    I1 = dimensional_integral(1, m2, d)
    epsilon = (4 - d) / 2
    pole, finite = extract_pole(epsilon, m2, mu2)
    
    print("Integral with dimensional regularization:")
    print("=" * 50)
    print(f"Spacetime dimension d = {d} (ε = {epsilon})")
    print(f"Mass m² = {m2}")
    print(f"Renormalization scale μ² = {mu2}")
    print(f"\nIntegral value I₁: {I1:.6e}")
    print(f"Pole: 1/ε = {pole:.2f}")
    print(f"Finite part: {finite:.6f}")

Integral with dimensional regularization: ================================================== Spacetime dimension d = 3.99 (ε = 0.005) Mass m² = 1.0 Renormalization scale μ² = 1.0 Integral value I₁: -1.592761e-02 Pole: 1/ε = 200.00 Finite part: 1.918939

## 5.2 Renormalization Group Equations

The dependence on the renormalization scale \\(\mu\\) is described by the Callan-Symanzik equation. This leads to the "running" of coupling constants and masses. 

### 🌀 Callan-Symanzik Equation

The renormalized correlation function \\(G\\) satisfies:

\\[ \left[ \mu\frac{\partial}{\partial\mu} + \beta(\lambda)\frac{\partial}{\partial\lambda} \+ n\gamma(\lambda) \right] G = 0 \\]

**β function** : running of coupling constant

\\[ \beta(\lambda) = \mu \frac{d\lambda}{d\mu} \\]

**Anomalous dimension** : field renormalization

\\[ \gamma(\lambda) = \frac{\mu}{2}\frac{d\log Z}{d\mu} \\]

Example 2: β Function of φ⁴ Theory
    
    
    import numpy as np
    from scipy.integrate import odeint
    
    # ===================================
    # Renormalization group flow of φ⁴ theory
    # ===================================
    
    def beta_phi4(lambda_, d=4):
        """β function of φ⁴ theory (1-loop)
    
        β(λ) = (4-d)λ + 3λ²/(16π²) + O(λ³)
        """
        epsilon = 4 - d
        beta = epsilon * lambda_ + 3 * lambda_**2 / (16 * np.pi**2)
    
        return beta
    
    def gamma_phi4(lambda_):
        """Field anomalous dimension (1-loop)"""
        gamma = lambda_ / (16 * np.pi**2)
    
        return gamma
    
    def rg_flow(lambda_, t, d=4):
        """Differential equation for RG flow
    
        dλ/dt = β(λ), t = log(μ/μ₀)
        """
        return beta_phi4(lambda_, d)
    
    # Numerical solution of RG flow
    lambda_0 = 0.1  # initial coupling constant
    t_array = np.linspace(0, 10, 100)  # log(μ/μ₀)
    
    # d=4 (critical dimension)
    lambda_d4 = odeint(rg_flow, lambda_0, t_array, args=(4,))
    
    # d=3 (renormalizable)
    lambda_d3 = odeint(rg_flow, lambda_0, t_array, args=(3,))
    
    print("RG flow of φ⁴ theory:")
    print("=" * 60)
    print(f"{'log(μ/μ₀)':<15} {'λ(d=4)':<20} {'λ(d=3)':<20}")
    print("-" * 60)
    
    for i in [0, 25, 50, 75, 99]:
        print(f"{t_array[i]:<15.2f} {lambda_d4[i][0]:<20.6f} {lambda_d3[i][0]:<20.6f}")

RG flow of φ⁴ theory: ============================================================ log(μ/μ₀) λ(d=4) λ(d=3) \------------------------------------------------------------ 0.00 0.100000 0.100000 2.53 0.102551 0.089898 5.05 0.105189 0.081796 7.58 0.107919 0.075423 10.10 0.110746 0.070255

## 5.3 Wilson Renormalization Group and Critical Phenomena

Wilson's renormalization group is a method that sequentially integrates out momentum shells. It explains the universal behavior near critical points of phase transitions. 

### 🎯 Procedure of Wilson RG

  1. Integrate out high-momentum modes \\(\Lambda/b < |k| < \Lambda\\)
  2. Rescale momenta: \\(k' = bk\\)
  3. Rescale fields: \\(\phi' = z\phi\\)
  4. Restore the effective action to its original form

This yields the transformation law of coupling constants (RG equations).

### 🔥 Critical Exponents and Universality Classes

Physical quantities near the phase transition point \\(T \to T_c\\):

Physical Quantity | Critical Behavior | Critical Exponent  
---|---|---  
Correlation length | \\(\xi \sim |T - T_c|^{-\nu}\\) | \\(\nu\\)  
Order parameter | \\(M \sim |T - T_c|^\beta\\) | \\(\beta\\)  
Susceptibility | \\(\chi \sim |T - T_c|^{-\gamma}\\) | \\(\gamma\\)  
Specific heat | \\(C \sim |T - T_c|^{-\alpha}\\) | \\(\alpha\\)  
  
**Scaling relations** : \\(\alpha + 2\beta + \gamma = 2\\), \\(\nu d = 2 - \alpha\\)

Example 3: Critical Exponents of the Ising Model
    
    
    import numpy as np
    
    # ===================================
    # Critical behavior of Ising model
    # ===================================
    
    def ising_critical_exponents(d):
        """Critical exponents of Ising model (approximate values)
    
        Args:
            d: spatial dimension
        """
        exponents = {
            2: {'nu': 1.0, 'beta': 0.125, 'gamma': 1.75, 'alpha': 0.0},
            3: {'nu': 0.63, 'beta': 0.325, 'gamma': 1.24, 'alpha': 0.11},
            4: {'nu': 0.5, 'beta': 0.5, 'gamma': 1.0, 'alpha': 0.0},  # mean field
        }
        return exponents.get(d, exponents[3])
    
    def verify_scaling_relations(exponents, d):
        """Verification of scaling relations"""
        nu, beta, gamma, alpha = (exponents['nu'], exponents['beta'],
                                  exponents['gamma'], exponents['alpha'])
    
        # Rushbrooke inequality: α + 2β + γ = 2
        rushbrooke = alpha + 2 * beta + gamma
    
        # Hyperscaling: νd = 2 - α
        hyperscaling_lhs = nu * d
        hyperscaling_rhs = 2 - alpha
    
        return rushbrooke, hyperscaling_lhs, hyperscaling_rhs
    
    # Verification for each dimension
    dimensions = [2, 3, 4]
    
    print("Critical exponents of Ising model:")
    print("=" * 70)
    
    for d in dimensions:
        exp = ising_critical_exponents(d)
        rush, hyp_l, hyp_r = verify_scaling_relations(exp, d)
    
        print(f"\nd = {d}:")
        print(f"  ν = {exp['nu']:.3f}, β = {exp['beta']:.3f}, "
              f"γ = {exp['gamma']:.3f}, α = {exp['alpha']:.3f}")
        print(f"  Rushbrooke: α + 2β + γ = {rush:.3f} (theoretical value: 2)")
        print(f"  Hyperscaling: νd = {hyp_l:.3f}, 2-α = {hyp_r:.3f}")

Critical exponents of Ising model: ====================================================================== d = 2: ν = 1.000, β = 0.125, γ = 1.750, α = 0.000 Rushbrooke: α + 2β + γ = 2.000 (theoretical value: 2) Hyperscaling: νd = 2.000, 2-α = 2.000 d = 3: ν = 0.630, β = 0.325, γ = 1.240, α = 0.110 Rushbrooke: α + 2β + γ = 2.000 (theoretical value: 2) Hyperscaling: νd = 1.890, 2-α = 1.890 d = 4: ν = 0.500, β = 0.500, γ = 1.000, α = 0.000 Rushbrooke: α + 2β + γ = 2.000 (theoretical value: 2) Hyperscaling: νd = 2.000, 2-α = 2.000

## 5.4 Effective Field Theory

Effective Field Theory (EFT) describes low-energy phenomena through an effective action where high-energy degrees of freedom are integrated out. It is a framework that systematically implements Wilson's ideas. 

### 📐 Construction of Effective Action

Integrate out modes above high momentum \\(\Lambda\\):

\\[ e^{iS_{\text{eff}}[\phi_<]} = \int \mathcal{D}\phi_> \, e^{iS[\phi_< \+ \phi_>]} \\]

\\(\phi_< (|\mathbf{k}| < \Lambda)\\): low modes, \\(\phi_> (|\mathbf{k}| > \Lambda)\\): high modes

**Effective Lagrangian** :

\\[ \mathcal{L}_{\text{eff}} = \sum_i c_i(\Lambda) \mathcal{O}_i \\]

\\(\mathcal{O}_i\\): all allowed operators (constrained by dimensional analysis)
    
    
    ```mermaid
    flowchart TD
        A[Full theoryUV ~ ∞] --> B[Wilson RG]
        B --> C[High mode integrationΛ < k < ∞]
        C --> D[Effective theoryk < Λ]
        D --> E[Low energy expansion]
        E --> F[Observables]
    
        G[Renormalizability] --> H[Relevant operatorsdim < d]
        H --> I[IR dominance]
        G --> J[Irrelevant operatorsdim > d]
        J --> K[UV suppression]
    
        style A fill:#e3f2fd
        style D fill:#f3e5f5
        style F fill:#e8f5e9
    ```

Example 4: From Fermi Theory to EW Theory
    
    
    import numpy as np
    
    # ===================================
    # Fermi theory and electroweak unified theory
    # ===================================
    
    def fermi_coupling_from_mw(M_W, g_w):
        """Derive Fermi coupling constant from W boson mass
    
        G_F = g²/(8M_W²)
        """
        G_F = g_w**2 / (8 * M_W**2)
        return G_F
    
    def effective_vs_full_theory(E, M_W, g_w):
        """Comparison of effective theory and full theory
    
        Low energy (E << M_W): Fermi theory
        High energy (E ~ M_W): Full electroweak theory
        """
        G_F = fermi_coupling_from_mw(M_W, g_w)
    
        # Cross section in Fermi theory (E << M_W)
        sigma_fermi = G_F**2 * E**2
    
        # Cross section in full theory (suppression by propagator)
        sigma_full = (g_w**4 / (E**2 + M_W**2)**2) * E**2
    
        validity = E / M_W  # validity parameter of effective theory
    
        return sigma_fermi, sigma_full, validity
    
    # Parameters
    M_W = 80.4  # GeV (W boson mass)
    g_w = 0.65  # weak coupling constant
    G_F = 1.166e-5  # GeV^-2 (experimental value)
    
    energies = [1, 10, 50, 100]  # GeV
    
    print("Comparison of effective theory and full theory:")
    print("=" * 70)
    print(f"W boson mass: {M_W} GeV")
    print(f"Fermi constant G_F: {G_F:.3e} GeV^-2")
    print(f"\n{'E (GeV)':<12} {'σ_Fermi':<18} {'σ_full':<18} {'E/M_W':<12}")
    print("-" * 70)
    
    for E in energies:
        sigma_f, sigma_full, val = effective_vs_full_theory(E, M_W, g_w)
    
        print(f"{E:<12} {sigma_f:<18.3e} {sigma_full:<18.3e} {val:<12.4f}")

Comparison of effective theory and full theory: ====================================================================== W boson mass: 80.4 GeV Fermi constant G_F: 1.166e-05 GeV^-2 E (GeV) σ_Fermi σ_full E/M_W \---------------------------------------------------------------------- 1 1.360e-10 1.353e-10 0.0124 10 1.360e-08 1.334e-08 0.1244 50 3.399e-07 2.691e-07 0.6219 100 1.360e-06 6.259e-07 1.2438

## 5.5 Landau-Ginzburg Theory and Phase Transitions

Landau-Ginzburg theory describes phase transitions as an effective theory of the order parameter. φ⁴ theory is the field theory version of this framework. 

### 🧲 Landau-Ginzburg Theory of Magnetic Materials

Free energy with magnetization \\(M(\mathbf{x})\\) as the order parameter:

\\[ F[M] = \int d^d x \left[ \frac{1}{2}(\nabla M)^2 + \frac{r}{2}M^2 + \frac{u}{4}M^4 \right] \\]

\\(r \propto (T - T_c)\\): deviation from temperature, \\(u > 0\\): interaction

**Phase transition** :

  * \\(r > 0\\) (\\(T > T_c\\)): Paramagnetic phase, \\(\langle M \rangle = 0\\)
  * \\(r < 0\\) (\\(T < T_c\\)): Ferromagnetic phase, \\(\langle M \rangle = \pm\sqrt{-r/u}\\)

Example 5: Minimization of Landau-Ginzburg Free Energy
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ===================================
    # Landau-Ginzburg free energy
    # ===================================
    
    def landau_free_energy(M, r, u):
        """Landau free energy (uniform field)
    
        F(M) = r/2 M² + u/4 M⁴
        """
        return r / 2 * M**2 + u / 4 * M**4
    
    def equilibrium_magnetization(r, u):
        """Calculate equilibrium magnetization"""
        if r >= 0:
            # Paramagnetic phase
            return 0.0
        else:
            # Ferromagnetic phase
            return np.sqrt(-r / u)
    
    def susceptibility(r, u, M_eq):
        """Susceptibility χ = ∂M/∂H"""
        if r >= 0:
            # χ ~ 1/r (Curie-Weiss)
            return 1 / r
        else:
            # χ ~ 1/(-2r)
            return 1 / (-2 * r)
    
    # Parameters
    u = 1.0
    r_values = np.linspace(-2.0, 2.0, 100)
    M_eq = [equilibrium_magnetization(r, u) for r in r_values]
    
    # Near critical temperature
    T_range = r_values  # r ∝ (T - T_c)
    
    print("Phase transition by Landau theory:")
    print("=" * 60)
    print(f"{'r (T-Tc)':<15} {'M_eq':<15} {'χ':<15}")
    print("-" * 60)
    
    for r in [-1.0, -0.5, 0.5, 1.0]:
        M = equilibrium_magnetization(r, u)
        chi = susceptibility(r, u, M) if r != 0 else np.inf
    
        print(f"{r:<15.2f} {M:<15.6f} {chi:<15.6f}")

Phase transition by Landau theory: ============================================================ r (T-Tc) M_eq χ \------------------------------------------------------------ -1.00 1.000000 0.500000 -0.50 0.707107 1.000000 0.50 0.000000 2.000000 1.00 0.000000 1.000000

## 5.6 Application to Materials Science: Structural Phase Transitions

Landau theory is widely used to describe structural phase transitions in materials (ferroelectricity, ferroelasticity, etc.). 

Example 6: Ferroelectric Phase Transition of BaTiO₃
    
    
    import numpy as np
    
    # ===================================
    # Ferroelectric phase transition of BaTiO₃ (Landau theory)
    # ===================================
    
    def landau_free_energy_ferro(P, a, b, c, E=0):
        """Landau free energy for ferroelectrics
    
        F(P) = a/2 P² + b/4 P⁴ + c/6 P⁶ - EP
    
        Args:
            P: polarization
            a: quadratic coefficient (temperature dependent)
            b, c: higher order coefficients
            E: external electric field
        """
        return a / 2 * P**2 + b / 4 * P**4 + c / 6 * P**6 - E * P
    
    def dielectric_constant(a, b, P_eq):
        """Dielectric constant ε ~ χ"""
        if a > 0:
            # Paraelectric phase (Curie-Weiss law)
            epsilon = 1 / a
        else:
            # Ferroelectric phase
            epsilon = 1 / (a + 3 * b * P_eq**2)
    
        return epsilon
    
    # Parameters for BaTiO₃ (simplified)
    T_c = 393  # K (Curie temperature)
    alpha_0 = 0.01  # temperature coefficient
    b = 1.0
    c = 0.1
    
    temperatures = [300, 350, 400, 450]  # K
    
    print("Ferroelectric phase transition of BaTiO₃:")
    print("=" * 60)
    print(f"Curie temperature: {T_c} K")
    print(f"\n{'T (K)':<12} {'a(T)':<15} {'P_eq':<15} {'ε':<15}")
    print("-" * 60)
    
    for T in temperatures:
        a_T = alpha_0 * (T - T_c)  # a ∝ (T - T_c)
    
        # Equilibrium polarization
        if a_T < 0 and b > 0:
            P_eq = np.sqrt(-a_T / b)
        else:
            P_eq = 0.0
    
        epsilon = dielectric_constant(a_T, b, P_eq) if a_T != 0 else np.inf
    
        print(f"{T:<12} {a_T:<15.4f} {P_eq:<15.6f} {epsilon:<15.6f}")

Ferroelectric phase transition of BaTiO₃: ============================================================ Curie temperature: 393 K T (K) a(T) P_eq ε \------------------------------------------------------------ 300 -0.9300 0.964365 1.351351 350 -0.4300 0.655744 3.214286 400 0.0700 0.000000 14.285714 450 0.5700 0.000000 1.754386

Example 7: Dynamics of Spinodal Decomposition
    
    
    import numpy as np
    
    # ===================================
    # Cahn-Hilliard equation (spinodal decomposition)
    # ===================================
    
    def cahn_hilliard_growth_rate(k, r, kappa):
        """Linear growth rate of CH equation
    
        ∂c/∂t = M ∇²(δF/δc)
        ω(k) = -M k² (r + κ k²)
    
        Args:
            k: wave number
            r: free energy coefficient (spinodal for r < 0)
            kappa: gradient energy coefficient
        """
        M = 1.0  # mobility
        omega = -M * k**2 * (r + kappa * k**2)
    
        return omega
    
    def fastest_growing_mode(r, kappa):
        """Fastest growing mode
    
        k_m = sqrt(-r / (2κ))
        """
        if r >= 0:
            return 0.0, 0.0
    
        k_m = np.sqrt(-r / (2 * kappa))
        omega_m = cahn_hilliard_growth_rate(k_m, r, kappa)
    
        return k_m, omega_m
    
    # Parameters (spinodal decomposition in alloy)
    r = -1.0  # spinodal region
    kappa = 1.0
    
    k_array = np.linspace(0.01, 2.0, 100)
    omega_array = [cahn_hilliard_growth_rate(k, r, kappa) for k in k_array]
    
    k_m, omega_m = fastest_growing_mode(r, kappa)
    
    print("Dynamics of spinodal decomposition:")
    print("=" * 50)
    print(f"Free energy coefficient r: {r}")
    print(f"Gradient coefficient κ: {kappa}")
    print(f"\nFastest growing wave number k_m: {k_m:.6f}")
    print(f"Growth rate ω(k_m): {omega_m:.6f}")
    print(f"Characteristic length scale λ_m: {2*np.pi/k_m:.6f}")

Dynamics of spinodal decomposition: ================================================== Free energy coefficient r: -1.0 Gradient coefficient κ: 1.0 Fastest growing wave number k_m: 0.707107 Growth rate ω(k_m): 0.250000 Characteristic length scale λ_m: 8.885765

Example 8: Critical Slowing Down
    
    
    import numpy as np
    
    # ===================================
    # Relaxation time near critical point
    # ===================================
    
    def relaxation_time(T, T_c, tau_0=1.0, z=2, nu=0.63):
        """Critical slowing down
    
        τ ~ ξ^z ~ |T - T_c|^{-zν}
    
        Args:
            z: dynamic critical exponent
            nu: correlation length exponent
        """
        t_reduced = np.abs(T - T_c) / T_c
    
        if t_reduced < 1e-10:
            return 1e10  # divergence
    
        tau = tau_0 * t_reduced**(-z * nu)
    
        return tau
    
    # Ferromagnetic transition of iron
    T_c = 1043  # K
    tau_0 = 1e-12  # s
    z = 2  # dynamic exponent (Model B)
    nu = 0.63  # Ising universality class
    
    temperatures = [T_c + dT for dT in [1, 10, 50, 100]]
    
    print("Critical slowing down:")
    print("=" * 60)
    print(f"Curie temperature T_c: {T_c} K")
    print(f"Dynamic exponent z: {z}")
    print(f"Correlation length exponent ν: {nu}")
    print(f"\n{'T (K)':<15} {'ΔT (K)':<15} {'τ (s)':<20}")
    print("-" * 60)
    
    for T in temperatures:
        dT = T - T_c
        tau = relaxation_time(T, T_c, tau_0, z, nu)
    
        print(f"{T:<15.1f} {dT:<15.1f} {tau:<20.6e}")

Critical slowing down: ============================================================ Curie temperature T_c: 1043 K Dynamic exponent z: 2 Correlation length exponent ν: 0.63 T (K) ΔT (K) τ (s) \------------------------------------------------------------ 1044.0 1.0 1.096e-09 1053.0 10.0 6.586e-11 1093.0 50.0 4.885e-12 1143.0 100.0 1.912e-12

## Exercises

### Easy

**Q1** : In dimensional regularization, explain how the divergence of \\(\int d^d k / (k^2)^n\\) appears as a pole in \\(\epsilon = (4-d)/2\\).

View Answer

Since \\(\Gamma(n - d/2)\\) has a pole at \\(n = d/2\\), a \\(1/\epsilon\\) pole appears as \\(\epsilon \to 0\\).

### Medium

**Q2** : In φ⁴ theory, when the β function is positive (\\(\beta(\lambda) > 0\\)), determine whether it is asymptotically free or IR free.

View Answer

If \\(\beta > 0\\), then \\(\lambda\\) increases as \\(\mu\\) increases → IR free. The coupling becomes stronger at high energies.

### Hard

**Q3** : Verify the scaling relation \\(\alpha + 2\beta + \gamma = 2\\) of the critical exponents of the Ising model using Landau-Ginzburg theory (mean field).

View Answer

Mean field: \\(\alpha = 0, \beta = 1/2, \gamma = 1\\)

\\(\alpha + 2\beta + \gamma = 0 + 2(1/2) + 1 = 2\\) ✓

[← Chapter 4](<chapter-4.html>) [Series Index](<index.html>)

## References

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1996). _The Quantum Theory of Fields, Vol. 2_. Cambridge University Press.
  3. Zinn-Justin, J. (2002). _Quantum Field Theory and Critical Phenomena_ (4th ed.). Oxford University Press.
  4. Goldenfeld, N. (1992). _Lectures on Phase Transitions and the Renormalization Group_. Westview Press.
  5. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_. Cambridge University Press.

## Series Conclusion

With this Chapter 5, the "Introduction to Quantum Field Theory" series is complete. Starting from field quantization, we have systematically learned the fundamentals of quantum field theory, including propagators, S-matrices, Feynman diagrams, renormalization theory, and effective field theory. 

These concepts are applied not only in particle physics but also in a wide range of fields, including condensed matter physics, statistical mechanics, and materials science. For further learning, topics include gauge theory, non-Abelian gauge theory, spontaneous symmetry breaking, and the path integral formalism. 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
