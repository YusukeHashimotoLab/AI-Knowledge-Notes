---
title: "Chapter 1: Fundamental Laws and Thermodynamic Potentials"
chapter_title: "Chapter 1: Fundamental Laws and Thermodynamic Potentials"
subtitle: 
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 0
exercises: 0
version: 1.0
created_at: "by:"
---

This chapter covers the fundamentals of Fundamental Laws and Thermodynamic Potentials, which 📖 fundamental laws of thermodynamics. You will learn definition of internal energy \\(U\\), physical meaning of enthalpy \\(H\\), and Helmholtz free energy \\(F\\).

[Fundamentals Dojo](<../index.html>) > [Equilibrium Thermodynamics and Phase Transitions](<index.html>) > Chapter 1 

## 🎯 Learning Objectives

  * Understand the meaning and applications of the zeroth through third laws of thermodynamics
  * Learn the definition of internal energy \\(U\\) and the first law of thermodynamics
  * Grasp the physical meaning of enthalpy \\(H\\) and its role in isobaric processes
  * Understand Helmholtz free energy \\(F\\) and useful work in isothermal processes
  * Apply Gibbs free energy \\(G\\) to chemical and phase equilibria
  * Choose and use the four thermodynamic potentials appropriately
  * Compute and visualize thermodynamic functions with Python

## 📖 Fundamental Laws of Thermodynamics

### The Four Laws of Thermodynamics

Thermodynamics is built on four universal laws derived from experimental facts.

#### Zeroth Law (Thermal Equilibrium)

“If A and B are in thermal equilibrium, and B and C are in thermal equilibrium, then A and C are also in thermal equilibrium.”

**Meaning** : Establishes the concept of temperature—the basis for thermometry.

#### First Law (Energy Conservation)

\\[ dU = \delta Q - \delta W \\]

The change in internal energy \\(dU\\) equals heat supplied to the system \\(\delta Q\\) minus work done by the system \\(\delta W\\).

**Meaning** : Energy is conserved; a perpetual-motion machine of the first kind is impossible.

#### Second Law (Entropy Increase)

\\[ dS \geq \frac{\delta Q}{T} \\]

The entropy of an isolated system never decreases (equality holds for reversible processes).

**Meaning** : Explains irreversibility—heat flows spontaneously from hot to cold; perpetual-motion machines of the second kind are impossible.

#### Third Law (Nernst Theorem)

\\[ \lim_{T \to 0} S(T) = 0 \\]

The entropy of a perfect crystal approaches zero at absolute zero.

**Meaning** : Absolute zero cannot be reached through a finite number of operations.

### Quasi-static and Reversible Processes

**Quasi-static process** : Proceeds infinitely slowly so the system remains arbitrarily close to equilibrium.

**Reversible process** : Quasi-static and free from dissipation; reversing the path restores both system and surroundings.

**Irreversible process** : Real-world processes involving friction, heat conduction, diffusion, etc.

## 💻 Code Example 1.1: Isothermal vs. Adiabatic Paths of an Ideal Gas

Python Implementation: Visualizing Reversible Paths on a PV Diagram
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    R = 8.314  # J/(mol·K)
    n = 1.0    # mol
    gamma = 1.4  # Heat capacity ratio (air)
    
    # Initial state
    P1 = 1e5  # Pa (1 atm)
    V1 = 0.0224  # m³ (STP)
    T1 = 273.15  # K
    
    # Volume range
    V_range = np.linspace(0.01, 0.05, 200)
    
    # Isothermal process (PV = nRT = const)
    def isothermal_process(V, n, R, T):
        """Isothermal path: PV = nRT"""
        return n * R * T / V
    
    # Adiabatic process (PV^γ = const)
    def adiabatic_process(V, P1, V1, gamma):
        """Adiabatic path: PV^γ = const"""
        return P1 * (V1 / V)**gamma
    
    # Isobaric process (constant P)
    def isobaric_process(V, P):
        """Isobaric path: P = const"""
        return P * np.ones_like(V)
    
    # Calculate curves
    P_isothermal = isothermal_process(V_range, n, R, T1)
    P_adiabatic = adiabatic_process(V_range, P1, V1, gamma)
    P_isobaric = isobaric_process(V_range, P1)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PV-diagram
    ax1 = axes[0]
    ax1.plot(V_range * 1000, P_isothermal / 1e5, 'b-', linewidth=2,
             label=f'Isothermal (T = {T1:.1f} K)')
    ax1.plot(V_range * 1000, P_adiabatic / 1e5, 'r-', linewidth=2,
             label=f'Adiabatic (γ = {gamma})')
    ax1.plot(V_range * 1000, P_isobaric / 1e5, 'g--', linewidth=2,
             label=f'Isobaric (P = {P1/1e5:.1f} bar)')
    ax1.scatter([V1 * 1000], [P1 / 1e5], color='black', s=100, zorder=5,
                label='Initial state')
    ax1.set_xlabel('Volume (L)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('PV Diagram: Reversible Processes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Work comparison (V1 → V2)
    V2 = 0.04  # m³
    W_isothermal = n * R * T1 * np.log(V2 / V1)
    W_adiabatic = (P1 * V1**gamma) * (V2**(1-gamma) - V1**(1-gamma)) / (1 - gamma)
    W_isobaric = P1 * (V2 - V1)
    
    ax2 = axes[1]
    processes = ['Isothermal', 'Adiabatic', 'Isobaric']
    works = [W_isothermal, W_adiabatic, W_isobaric]
    colors = ['blue', 'red', 'green']
    
    ax2.bar(processes, works, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Work done by gas (J)')
    ax2.set_title(f'Expansion Work (V: {V1*1000:.1f} L → {V2*1000:.1f} L)')
    ax2.grid(True, axis='y', alpha=0.3)
    for i, (process, work) in enumerate(zip(processes, works)):
        ax2.text(i, work + 50, f'{work:.1f} J', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('thermo_reversible_processes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Work Extracted from Reversible Paths ===")
    print(f"Initial state: V = {V1*1000:.1f} L, P = {P1/1e5:.2f} bar, T = {T1:.1f} K")
    print(f"Final state:   V = {V2*1000:.1f} L\n")
    print(f"Isothermal: W = nRT ln(V2/V1) = {W_isothermal:.2f} J")
    print(f"Adiabatic:  W = (P1V1^γ)(V2^(1-γ) - V1^(1-γ))/(1-γ) = {W_adiabatic:.2f} J")
    print(f"Isobaric:   W = P(V2 - V1) = {W_isobaric:.2f} J")
    print("\n→ The isothermal path delivers the largest work because it absorbs heat during expansion.")
    

## 📊 Thermodynamic Potentials

### The Four Fundamental Potentials

State functions used to describe equilibrium systems:

Potential | Definition | Natural Variables | Primary Use  
---|---|---|---  
**Internal Energy (U)** | \\(U = U(S, V, N)\\) | \\(S, V, N\\) | Isolated/adiabatic systems  
**Enthalpy (H)** | \\(H = U + PV\\) | \\(S, P, N\\) | Isobaric processes, reaction heat  
**Helmholtz Free Energy (F)** | \\(F = U - TS\\) | \\(T, V, N\\) | Isothermal/isochoric processes, statistical mechanics  
**Gibbs Free Energy (G)** | \\(G = H - TS = U + PV - TS\\) | \\(T, P, N\\) | Isothermal/isobaric conditions, chemical & phase equilibria  
  
#### Differential Forms

\\[ \begin{aligned} dU &= TdS - PdV + \mu dN \\\ dH &= TdS + VdP + \mu dN \\\ dF &= -SdT - PdV + \mu dN \\\ dG &= -SdT + VdP + \mu dN \end{aligned} \\]

\\(\mu\\) is the chemical potential.

### Choosing the Right Potential

  * **Internal energy \\(U\\)** : Systems with controllable \\(S\\) and \\(V\\) (e.g., adiabatic rigid containers)
  * **Enthalpy \\(H\\)** : Constant-pressure processes (typical chemical reactions)
  * **Helmholtz free energy \\(F\\)** : Constant-temperature systems (in contact with a heat bath) and links to statistical mechanics
  * **Gibbs free energy \\(G\\)** : Systems under constant \\(T\\) and \\(P\\) (most experimental conditions)

## 💻 Code Example 1.2: Thermodynamic Potentials of an Ideal Gas

Python Implementation: Computing and Comparing Four Potentials
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Physical constants
    R = 8.314  # J/(mol·K)
    n = 1.0    # mol
    
    # Thermodynamic potentials of an ideal monatomic gas
    def ideal_gas_potentials(T, V, P, n, R):
        """Return U, H, F, G, and S for an ideal gas."""
        # Internal energy (monatomic ideal gas, Cv = (3/2)R)
        U = (3/2) * n * R * T
    
        # Enthalpy H = U + PV
        H = U + P * V
    
        # Helmholtz free energy F = U - TS
        # Use a qualitative Sackur–Tetrode-like entropy expression
        S = n * R * ((3/2) * np.log(T) + np.log(V) + 10)
        F = U - T * S
    
        # Gibbs free energy G = H - TS
        G = H - T * S
    
        return U, H, F, G, S
    
    # Temperature sweep at fixed volume/pressure
    V_fixed = 0.0224  # m³
    P_fixed = 1e5     # Pa
    T_range = np.linspace(100, 500, 100)
    
    potentials_vs_T = {'U': [], 'H': [], 'F': [], 'G': [], 'S': []}
    
    for T in T_range:
        U, H, F, G, S = ideal_gas_potentials(T, V_fixed, P_fixed, n, R)
        potentials_vs_T['U'].append(U)
        potentials_vs_T['H'].append(H)
        potentials_vs_T['F'].append(F)
        potentials_vs_T['G'].append(G)
        potentials_vs_T['S'].append(S)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(T_range, potentials_vs_T['U'], 'b-', linewidth=2)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('U (J)')
    ax1.set_title('Internal Energy U(T)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(T_range, potentials_vs_T['H'], 'r-', linewidth=2)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('H (J)')
    ax2.set_title('Enthalpy H(T)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(T_range, potentials_vs_T['F'], 'g-', linewidth=2)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('F (J)')
    ax3.set_title('Helmholtz Free Energy F(T)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(T_range, potentials_vs_T['G'], 'purple', linewidth=2)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('G (J)')
    ax4.set_title('Gibbs Free Energy G(T)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermo_potentials.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Values at a reference temperature
    T_ref = 300  # K
    U, H, F, G, S = ideal_gas_potentials(T_ref, V_fixed, P_fixed, n, R)
    
    print(f"=== Ideal-Gas Potentials at T = {T_ref} K ===")
    print(f"Volume  V = {V_fixed * 1000:.2f} L")
    print(f"Pressure P = {P_fixed / 1e5:.2f} bar\n")
    print(f"Internal energy  U = {U:.2f} J")
    print(f"Enthalpy        H = U + PV = {H:.2f} J")
    print(f"Helmholtz free energy F = U - TS = {F:.2f} J")
    print(f"Gibbs free energy    G = H - TS = {G:.2f} J")
    print(f"Entropy             S = {S:.2f} J/K")
    print("\nConsistency checks:")
    print(f"H - U = PV = {H - U:.2f} J (theoretical: {P_fixed * V_fixed:.2f} J)")
    print(f"U - F = TS = {U - F:.2f} J (theoretical: {T_ref * S:.2f} J)")
    print(f"H - G = TS = {H - G:.2f} J (theoretical: {T_ref * S:.2f} J)")
    

## 💻 Code Example 1.3: Gibbs Free Energy Under Isothermal–Isobaric Conditions

### Gibbs Free Energy and Spontaneity

At constant \\(T\\) and \\(P\\), Gibbs free energy governs equilibrium:

\\[ dG = -SdT + VdP \\]

When \\(T\\) and \\(P\\) are fixed, \\(dG = 0\\) characterizes equilibrium.

**Spontaneity criteria** :

  * \\(\Delta G < 0\\): proceeds spontaneously (exergonic)
  * \\(\Delta G = 0\\): equilibrium
  * \\(\Delta G > 0\\): non-spontaneous (endergonic)

Python Implementation: Gibbs Free Energy of a Reaction
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Hypothetical reaction: A ⇌ B
    # ΔG = ΔH - TΔS
    
    delta_H = -50000  # J/mol (exothermic)
    delta_S = -100    # J/(mol·K) (entropy decreases)
    
    def gibbs_free_energy_change(delta_H, delta_S, T):
        """Return ΔG for given ΔH, ΔS, and temperature."""
        return delta_H - T * delta_S
    
    T_range = np.linspace(200, 800, 100)
    delta_G_range = [gibbs_free_energy_change(delta_H, delta_S, T) for T in T_range]
    
    # Equilibrium temperature where ΔG = 0
    T_eq = delta_H / delta_S
    delta_G_eq = gibbs_free_energy_change(delta_H, delta_S, T_eq)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ΔG vs. temperature
    ax1 = axes[0]
    ax1.plot(T_range, delta_G_range, 'b-', linewidth=2, label='ΔG(T)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, label='ΔG = 0')
    ax1.axvline(T_eq, color='red', linestyle='--', linewidth=1.5,
                label=f'Equilibrium temperature ({T_eq:.1f} K)')
    ax1.fill_between(T_range, delta_G_range, 0, where=(np.array(delta_G_range) < 0),
                     alpha=0.3, color='green', label='Spontaneous (ΔG < 0)')
    ax1.fill_between(T_range, delta_G_range, 0, where=(np.array(delta_G_range) > 0),
                     alpha=0.3, color='red', label='Non-spontaneous (ΔG > 0)')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('ΔG (J/mol)')
    ax1.set_title('Temperature Dependence of ΔG')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Enthalpy vs. entropy contributions
    ax2 = axes[1]
    enthalpy_term = delta_H * np.ones_like(T_range)
    entropy_term = -T_range * delta_S
    
    ax2.plot(T_range, enthalpy_term, 'r-', linewidth=2, label='ΔH (enthalpy term)')
    ax2.plot(T_range, entropy_term, 'b-', linewidth=2, label='-TΔS (entropy term)')
    ax2.plot(T_range, delta_G_range, 'purple', linewidth=2.5, label='ΔG = ΔH - TΔS')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Energy (J/mol)')
    ax2.set_title('Contribution of Enthalpy and Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermo_gibbs_reaction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Gibbs Free Energy Analysis ===")
    print("Reaction: A ⇌ B")
    print(f"ΔH = {delta_H / 1000:.1f} kJ/mol (exothermic)")
    print(f"ΔS = {delta_S:.1f} J/(mol·K) (entropy decreases)\n")
    print(f"Equilibrium temperature: T_eq = ΔH/ΔS = {T_eq:.1f} K\n")
    
    for T in [300, T_eq, 700]:
        dG = gibbs_free_energy_change(delta_H, delta_S, T)
        if dG < 0:
            spontaneity = "Spontaneous (A → B)"
        elif dG > 0:
            spontaneity = "Non-spontaneous (reverse favored)"
        else:
            spontaneity = "At equilibrium"
        print(f"T = {T:.1f} K: ΔG = {dG / 1000:.2f} kJ/mol → {spontaneity}")
    
    print("\nInterpretation:")
    print("- Low T: ΔH dominates → exothermic reaction proceeds (ΔG < 0).")
    print("- High T: -TΔS dominates → entropy decrease penalizes forward direction (ΔG > 0).")
    print(f"- T = {T_eq:.1f} K: enthalpy and entropy balance → equilibrium.")
    

## 💻 Code Example 1.4: Legendre Transformations Between Potentials

### Legendre Transform

Thermodynamic potentials are related through Legendre transforms.

For a function \\(f(x)\\), its Legendre transform \\(g(p)\\) is defined by:

\\[ g(p) = px - f(x), \quad p = \frac{df}{dx} \\]

#### Applications to Thermodynamics

  * \\(H = U + PV\\) (transform \\(V \to P\\), with \\(P = -\partial U/\partial V\\))
  * \\(F = U - TS\\) (transform \\(S \to T\\), with \\(T = \partial U/\partial S\\))
  * \\(G = U - TS + PV = F + PV = H - TS\\) (transform both \\(S\\) and \\(V\\))

Python Implementation: Symbolic Legendre Transform
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    
    # SymPy example: f(x) = x^2
    x = sp.Symbol('x', real=True, positive=True)
    f = x**2
    df_dx = sp.diff(f, x)
    
    print("=== Analytic Legendre Transform ===")
    print(f"Original function: f(x) = {f}")
    print(f"df/dx = {df_dx}")
    
    p = sp.Symbol('p', real=True, positive=True)
    x_of_p = sp.solve(df_dx - p, x)[0]
    print(f"From p = df/dx we obtain x(p) = {x_of_p}")
    
    g = (p * x_of_p - f.subs(x, x_of_p)).simplify()
    print(f"Legendre transform: g(p) = px - f(x) = {g}\n")
    
    # Numerical plots
    x_vals = np.linspace(0.1, 3, 100)
    f_vals = x_vals**2
    
    p_vals = 2 * x_vals  # p = df/dx = 2x
    g_vals = p_vals**2 / 4  # g(p) = p²/4
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.plot(x_vals, f_vals, 'b-', linewidth=2, label='f(x) = x²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Original Function f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(p_vals, g_vals, 'r-', linewidth=2, label='g(p) = p²/4')
    ax2.set_xlabel('p = df/dx')
    ax2.set_ylabel('g(p)')
    ax2.set_title('Legendre Transform g(p)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermo_legendre_transform.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Application to Thermodynamic Potentials ===")
    print("Internal energy U(S, V):")
    print("  dU = TdS - PdV")
    print("  → T = (∂U/∂S)_V,  P = -(∂U/∂V)_S")
    print()
    print("Legendre transforms:")
    print("  S → T: F = U - TS  (Helmholtz free energy)")
    print("  V → P: H = U + PV  (Enthalpy)")
    print("  Both: G = U - TS + PV  (Gibbs free energy)")
    print()
    print("Legendre transforms let us construct potentials whose natural variables")
    print("match the experimental control parameters (T, P).")
    

## 💻 Code Example 1.5: Third Law of Thermodynamics and Nernst Theorem

### Third Law (Nernst Theorem)

Entropy at absolute zero satisfies:

\\[ \lim_{T \to 0} S(T) = 0 \\]

**Implications** :

  * A perfect crystal has zero entropy at absolute zero.
  * The system freezes into a unique ground state (\\(\Omega = 1\\)).
  * Reaching absolute zero requires infinitely many steps.

**Consequence** : Heat capacity follows \\(C_V \propto T^3\\) at low temperatures (Debye law).

Python Implementation: Low-Temperature Entropy and Heat Capacity
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Debye model for entropy and heat capacity
    def debye_entropy(T, theta_D, R):
        """Entropy from a simplified Debye model."""
        x = theta_D / T
        if T < 0.01:
            return 0
        if x > 10:
            return (12/5) * np.pi**4 * R * (T / theta_D)**3
        else:
            return 3 * R * (np.log(T / theta_D) + 4/3)
    
    def debye_heat_capacity(T, theta_D, R):
        """Constant-volume heat capacity from the Debye model."""
        x = theta_D / T
        if x > 10:
            return (12/5) * np.pi**4 * R * (T / theta_D)**3
        else:
            return 3 * R
    
    R = 8.314  # J/(mol·K)
    theta_D_Cu = 343  # Debye temperature of Cu (K)
    
    T_range = np.logspace(-1, 3, 200)  # 0.1 K to 1000 K
    S_vals = [debye_entropy(T, theta_D_Cu, R) for T in T_range]
    C_vals = [debye_heat_capacity(T, theta_D_Cu, R) for T in T_range]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.loglog(T_range, S_vals, 'b-', linewidth=2)
    ax1.axvline(theta_D_Cu, color='red', linestyle='--', linewidth=1.5,
                label=f'Debye temperature ({theta_D_Cu} K)')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Entropy (J/(mol·K))')
    ax1.set_title('Entropy vs. Temperature (Cu)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2 = axes[1]
    ax2.loglog(T_range, C_vals, 'r-', linewidth=2, label='Debye model')
    ax2.axhline(3 * R, color='green', linestyle='--', linewidth=1.5,
                label='Dulong–Petit limit (3R)')
    ax2.axvline(theta_D_Cu, color='black', linestyle='--', linewidth=1.5,
                label='Debye temperature')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Heat capacity C_V (J/(mol·K))')
    ax2.set_title('Heat Capacity vs. Temperature (Cu)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('thermo_third_law.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Third-Law Verification (Cu) ===")
    print(f"Debye temperature: θ_D = {theta_D_Cu} K\n")
    
    T_low = [0.1, 1, 10, 100]
    print("Low-temperature entropy:")
    for T in T_low:
        S = debye_entropy(T, theta_D_Cu, R)
        print(f"T = {T:.1f} K → S = {S:.4e} J/(mol·K)")
    
    print("\nLow-temperature heat capacity:")
    for T in T_low:
        C = debye_heat_capacity(T, theta_D_Cu, R)
        print(f"T = {T:.1f} K → C_V = {C:.4e} J/(mol·K)")
    
    print("\n→ Both entropy and heat capacity drop toward zero as T³, consistent with the third law.")
    

## 💻 Code Example 1.6: Equation of State and Internal Energy of a van der Waals Gas

### van der Waals Equation

\\[ \left(P + \frac{an^2}{V^2}\right)(V - nb) = nRT \\]

**Corrections** :

  * \\(a\\): attractive interaction correction
  * \\(b\\): finite molecular size

Internal energy:

\\[ U = nC_VT - \frac{an^2}{V} \\]

Unlike an ideal gas, \\(U\\) now depends on volume.

Python Implementation: Analyzing a van der Waals Gas
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    # van der Waals constants for N₂
    a_N2 = 0.1408    # Pa·m⁶/mol²
    b_N2 = 3.913e-5  # m³/mol
    R = 8.314        # J/(mol·K)
    C_V = 2.5 * R    # Diatomic gas
    
    def van_der_waals_pressure(V, n, T, a, b, R):
        """Pressure given by the van der Waals EOS."""
        return n * R * T / (V - n * b) - a * n**2 / V**2
    
    def van_der_waals_internal_energy(V, n, T, a, C_V):
        """Internal energy including attraction correction."""
        return n * C_V * T - a * n**2 / V
    
    n = 1.0  # mol
    T = 300  # K
    
    V_range = np.linspace(0.001, 0.1, 200)
    
    P_vdw = [van_der_waals_pressure(V, n, T, a_N2, b_N2, R) for V in V_range]
    U_vdw = [van_der_waals_internal_energy(V, n, T, a_N2, C_V) for V in V_range]
    
    P_ideal = [n * R * T / V for V in V_range]
    U_ideal = n * C_V * T * np.ones_like(V_range)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.plot(V_range * 1000, np.array(P_vdw) / 1e5, 'b-', linewidth=2, label='van der Waals')
    ax1.plot(V_range * 1000, np.array(P_ideal) / 1e5, 'r--', linewidth=2, label='Ideal gas')
    ax1.set_xlabel('Volume (L)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title(f'PV Isotherm (T = {T} K, N₂)')
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 50])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(V_range * 1000, U_vdw, 'b-', linewidth=2, label='van der Waals')
    ax2.plot(V_range * 1000, U_ideal, 'r--', linewidth=2, label='Ideal gas')
    ax2.set_xlabel('Volume (L)')
    ax2.set_ylabel('Internal energy U (J)')
    ax2.set_title(f'Internal Energy (T = {T} K, N₂)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermo_van_der_waals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    V_test = 0.0224  # m³ (STP)
    P_vdw_val = van_der_waals_pressure(V_test, n, T, a_N2, b_N2, R)
    P_ideal_val = n * R * T / V_test
    U_vdw_val = van_der_waals_internal_energy(V_test, n, T, a_N2, C_V)
    U_ideal_val = n * C_V * T
    
    print("=== van der Waals vs. Ideal Gas (N₂, T = 300 K) ===")
    print(f"Volume V = {V_test * 1000:.2f} L\n")
    print("Pressure:")
    print(f"  van der Waals: {P_vdw_val / 1e5:.4f} bar")
    print(f"  Ideal gas:    {P_ideal_val / 1e5:.4f} bar")
    print(f"  Difference:   {(P_vdw_val - P_ideal_val) / 1e5:.4f} bar\n")
    print("Internal energy:")
    print(f"  van der Waals: U = {U_vdw_val:.2f} J")
    print(f"  Ideal gas:     U = {U_ideal_val:.2f} J")
    print(f"  Difference:    {U_vdw_val - U_ideal_val:.2f} J\n")
    print("Corrections:")
    print(f"  Attraction term  (-an²/V) = {-a_N2 * n**2 / V_test:.2f} J")
    print(f"  Volume correction (nb) = {n * b_N2 * 1000:.4f} L")
    

## 💻 Code Example 1.7: Materials Application—Equation of State and Thermal Expansion

Python Implementation: Solid Thermal Expansion via the Grüneisen Relation
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Grüneisen relation: α = γ C_V / (V K_T)
    
    def thermal_expansion_coefficient(T, gamma, C_V, V, K_T):
        """Thermal expansion coefficient α = γ C_V / (V K_T)."""
        return gamma * C_V(T) / (V * K_T)
    
    def debye_heat_capacity_solid(T, theta_D, R):
        """Debye heat capacity for a solid."""
        x = theta_D / T
        if x > 10:
            return (12/5) * np.pi**4 * R * (T / theta_D)**3
        else:
            return 3 * R
    
    # Material constants for Cu
    theta_D_Cu = 343  # K
    gamma_Cu = 2.0
    V_Cu = 7.11e-6  # m³/mol
    K_T_Cu = 1.4e11  # Pa
    R = 8.314  # J/(mol·K)
    
    T_range = np.linspace(10, 1000, 200)
    
    alpha_vals = []
    C_V_vals = []
    
    for T in T_range:
        C_V = debye_heat_capacity_solid(T, theta_D_Cu, R)
        alpha = thermal_expansion_coefficient(T, gamma_Cu, lambda _: C_V, V_Cu, K_T_Cu)
        alpha_vals.append(alpha)
        C_V_vals.append(C_V)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.plot(T_range, np.array(alpha_vals) * 1e6, 'b-', linewidth=2)
    ax1.axvline(theta_D_Cu, color='red', linestyle='--', linewidth=1.5,
                label=f'Debye temperature ({theta_D_Cu} K)')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Thermal expansion α (10⁻⁶ K⁻¹)')
    ax1.set_title('Temperature Dependence of α (Cu)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    scatter = axes[1].scatter(C_V_vals, np.array(alpha_vals) * 1e6,
                              c=T_range, cmap='viridis', s=20)
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Temperature (K)')
    axes[1].set_xlabel('Heat capacity C_V (J/(mol·K))')
    axes[1].set_ylabel('Thermal expansion α (10⁻⁶ K⁻¹)')
    axes[1].set_title('Correlation Between α and C_V (Grüneisen relation)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermo_thermal_expansion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Thermal Expansion of Cu ===")
    print(f"Debye temperature: θ_D = {theta_D_Cu} K")
    print(f"Grüneisen parameter: γ = {gamma_Cu}")
    print(f"Isothermal bulk modulus: K_T = {K_T_Cu:.2e} Pa\n")
    
    for T in [100, 300, 500, 1000]:
        C_V = debye_heat_capacity_solid(T, theta_D_Cu, R)
        alpha = thermal_expansion_coefficient(T, gamma_Cu, lambda _: C_V, V_Cu, K_T_Cu)
        print(f"T = {T:4d} K: C_V = {C_V:.2f} J/(mol·K), α = {alpha*1e6:.2f} × 10⁻⁶ K⁻¹")
    
    print("\nGrüneisen relation: α = γ C_V / (V K_T)")
    print("→ Thermal expansion tracks heat capacity (both → 0 at low T).")
    print("→ Critical for evaluating thermal stress in materials design.")
    

## 📚 Summary

  * The four laws define temperature, energy conservation, entropy increase, and the nature of absolute zero.
  * Internal energy \\(U\\) follows \\(dU = \delta Q - \delta W\\) and is the base state function.
  * Enthalpy \\(H\\) is suited for isobaric processes and reaction heat.
  * Helmholtz free energy \\(F\\) captures useful work under isothermal conditions and links to statistical mechanics.
  * Gibbs free energy \\(G\\) is the key criterion for chemical and phase equilibria at constant \\(T\\) and \\(P\\).
  * Potentials are connected via Legendre transforms.
  * The third law predicts \\(S, C_V \to 0\\) as \\(T^3\\) when \\(T \to 0\\).
  * Realistic equations of state (van der Waals) and thermal expansion models are essential for materials applications.

### 💡 Practice Problems

  1. **Carnot efficiency** : For \\(T_H = 600\,\text{K}\\) and \\(T_C = 300\,\text{K}\\), compute \\(\eta = 1 - T_C/T_H\\) and relate it to the second law.
  2. **Differentials of potentials** : Starting from \\(G = H - TS\\), derive \\(dG = -SdT + VdP\\) (with constant \\(N\\)).
  3. **Chemical equilibrium constant** : Using \\(\Delta G^\circ = -RT \ln K_{eq}\\), find \\(K_{eq}\\) at \\(T = 298\,\text{K}\\) for \\(\Delta G^\circ = -10\,\text{kJ/mol}\\).
  4. **van der Waals critical point** : From \\((\partial P/\partial V)_T = 0\\) and \\((\partial^2 P/\partial V^2)_T = 0\\), derive \\(T_c = 8a/(27Rb)\\).
  5. **Estimating Debye temperature** : Implement a method to fit low-temperature heat capacity data (\\(C_V \propto T^3\\)) and extract \\(\theta_D\\).

[← Series Overview](<index.html>) [Chapter 2: Maxwell Relations and Thermodynamic Identities →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
