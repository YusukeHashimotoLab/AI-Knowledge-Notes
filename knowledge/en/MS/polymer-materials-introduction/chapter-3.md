---
title: "Chapter 3: Polymer Physical Properties"
chapter_title: "Chapter 3: Polymer Physical Properties"
---

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>):Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/MS/polymer-materials-introduction/chapter-3.html>) | Last sync: 2025-11-16

  * [Table of Contents](<index.html>)
  * [� Chapter 2](<chapter-2.html>)
  * [Chapter 3](<chapter-3.html>)
  * [Chapter 4 ’](<chapter-4.html>)
  * [Chapter 5](<chapter-4.html>)

This chapter covers Polymer Physical Properties. You will learn essential concepts and techniques.

### Learning Objectives

**Beginner:**

  * Understand the key features of stress-strain curves (yield, fracture)
  * Explain the basic concept of viscoelasticity (intermediate behavior between elasticity and viscosity)
  * Understand the difference between creep and stress relaxation

**Intermediate:**

  * Simulate viscoelasticity using Maxwell and Voigt models
  * Perform time-temperature superposition using the WLF equation
  * Calculate E', E'', and tan ´ from dynamic mechanical analysis (DMA) data

**Advanced:**

  * Model complex rheological behavior
  * Create master curves and predict long-term behavior
  * Precisely determine glass transition temperature from tan ´ peaks

## 3.1 Mechanical Properties

The **mechanical properties** of polymer materials are evaluated through stress-strain testing. In tensile testing, a load is applied to the specimen and strain is measured to obtain characteristic values such as **Young's modulus (elastic modulus)** , **yield stress** , **fracture stress** , and **elongation at break**. 
    
    
    ```mermaid
    flowchart TD
                        A[Stress-Strain Test] --> B[Elastic Region]
                        A --> C[Yield Point]
                        A --> D[Plastic Deformation]
                        A --> E[Fracture Point]
                        B --> F[Young's Modulus EÃ = E × µ]
                        C --> G[Yield Stress ÃyPlastic Deformation Onset]
                        D --> H[NeckingLocal Constriction]
                        E --> I[Fracture Stress ÃbElongation at Break µb]
                        F --> J[Application: Stiffness Evaluation]
                        G --> K[Application: Load-Bearing Design]
                        I --> L[Application: Ductility Evaluation]
    ```

### 3.1.1 Stress-Strain Curve Simulation

Stress-strain curves of polymers vary greatly depending on the material type (glassy, rubbery, semicrystalline). The following simulates three typical patterns. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Stress-Strain Curve Simulation
    def simulate_stress_strain_curves():
        """
        Simulate typical stress-strain curves for polymer materials
    
        Returns:
        - strain: Strain (%)
        - stresses: Stress for each material type (MPa)
        """
        # Strain range (%)
        strain = np.linspace(0, 100, 500)
    
        # 1. Glassy polymer (PMMA, PS): High stiffness, low ductility
        def glassy_polymer(eps):
            """Stress-strain for glassy polymer"""
            E = 3000  # MPa (high Young's modulus)
            sigma_y = 70  # MPa (yield stress)
            eps_y = sigma_y / E * 100  # Yield strain (%)
            eps_b = 5  # % (fracture strain: brittle)
    
            sigma = np.zeros_like(eps)
            for i, e in enumerate(eps):
                if e <= eps_y:
                    sigma[i] = E * e / 100  # Elastic region
                elif e <= eps_b:
                    sigma[i] = sigma_y + (e - eps_y) * 2  # Slight plastic deformation
                else:
                    sigma[i] = 0  # Fracture
            return sigma
    
        # 2. Rubbery polymer (Natural rubber, Silicone rubber): Low stiffness, high ductility
        def rubbery_polymer(eps):
            """Stress-strain for rubbery polymer (nonlinear elasticity)"""
            G = 2  # MPa (low shear modulus)
            # Rubber elasticity theory: Ã = G(» - »^-2), » = 1 + µ
            lambda_ratio = 1 + eps / 100
            sigma = G * (lambda_ratio - lambda_ratio**(-2))
            return sigma
    
        # 3. Semicrystalline polymer (PE, PP): Medium stiffness, high ductility, necking
        def semicrystalline_polymer(eps):
            """Stress-strain for semicrystalline polymer"""
            E = 1200  # MPa
            sigma_y = 25  # MPa
            eps_y = sigma_y / E * 100
            eps_neck_end = 30  # Necking end strain
            eps_b = 80  # Fracture strain
    
            sigma = np.zeros_like(eps)
            for i, e in enumerate(eps):
                if e <= eps_y:
                    sigma[i] = E * e / 100  # Elastic region
                elif e <= eps_neck_end:
                    # Stress is nearly constant during necking
                    sigma[i] = sigma_y - 3 + (e - eps_y) * 0.1
                elif e <= eps_b:
                    # Strain hardening (strengthening due to orientation)
                    sigma[i] = 22 + (e - eps_neck_end) * 0.4
                else:
                    sigma[i] = 0  # Fracture
            return sigma
    
        # Stress calculation
        stress_glassy = glassy_polymer(strain)
        stress_rubbery = rubbery_polymer(strain)
        stress_semicryst = semicrystalline_polymer(strain)
    
        # Visualization
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(strain[stress_glassy > 0], stress_glassy[stress_glassy > 0],
                 'b-', linewidth=2, label='Glassy (PMMA)')
        plt.plot(strain[stress_rubbery > 0], stress_rubbery[stress_rubbery > 0],
                 'r-', linewidth=2, label='Rubbery (Natural Rubber)')
        plt.plot(strain[stress_semicryst > 0], stress_semicryst[stress_semicryst > 0],
                 'g-', linewidth=2, label='Semicrystalline (PE)')
    
        plt.xlabel('Strain (%)', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Stress-Strain Curves for Different Polymers', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0, 100)
    
        # Subplot 2: Young's modulus comparison
        plt.subplot(1, 2, 2)
        materials = ['Glassy\n(PMMA)', 'Semicryst.\n(PE)', 'Rubbery\n(NR)']
        youngs_moduli = [3000, 1200, 2]  # MPa
        colors = ['#4A90E2', '#50C878', '#E74C3C']
    
        bars = plt.bar(materials, youngs_moduli, color=colors, edgecolor='black', linewidth=2)
        plt.ylabel('Young\'s Modulus (MPa)', fontsize=12)
        plt.title('Comparison of Young\'s Moduli', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(alpha=0.3, axis='y')
    
        # Numerical labels
        for bar, val in zip(bars, youngs_moduli):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val} MPa', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('stress_strain_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== Stress-Strain Characteristics Comparison ===")
        print("\n1. Glassy Polymer (PMMA):")
        print("   Young's modulus: 3000 MPa, Yield stress: 70 MPa, Fracture strain: 5%")
        print("   Features: High stiffness, brittleness, transparency")
    
        print("\n2. Rubbery Polymer (Natural Rubber):")
        print("   Young's modulus: 2 MPa, Fracture strain: >500%")
        print("   Features: Low stiffness, high ductility, entropic elasticity")
    
        print("\n3. Semicrystalline Polymer (Polyethylene):")
        print("   Young's modulus: 1200 MPa, Yield stress: 25 MPa, Fracture strain: 80%")
        print("   Features: Medium stiffness, ductility, necking phenomenon")
    
        return strain, stress_glassy, stress_rubbery, stress_semicryst
    
    # Execute
    simulate_stress_strain_curves()
    

## 3.2 Fundamentals of Viscoelasticity

Polymers exhibit **viscoelasticity** , showing intermediate behavior between elastic solids and viscous fluids. They respond elastically instantaneously, but viscous flow occurs over long periods. Viscoelasticity is described by the **Maxwell model** and **Voigt model**. 

### Maxwell Model and Voigt Model

**Maxwell Model:** **Series connection** of spring and dashpot. Represents stress relaxation. Under constant stress, strain increases over time (creep). 

**Voigt Model:** **Parallel connection** of spring and dashpot. Represents creep. Under constant strain, stress does not relax (delayed elasticity). 

### 3.2.1 Maxwell Model Simulation

Stress relaxation in the Maxwell model is described by the following differential equation: 

\\[ \sigma(t) = \sigma_0 e^{-t/\tau} \\] 

where Ä = ·/E is the relaxation time, · is viscosity, and E is elastic modulus. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Maxwell Model Simulation
    def simulate_maxwell_model(relaxation_times=[1, 10, 100], strain0=0.1):
        """
        Simulate stress relaxation using Maxwell model
    
        Parameters:
        - relaxation_times: List of relaxation times Ä (seconds)
        - strain0: Initial strain
    
        Returns:
        - time: Time (seconds)
        - stresses: Stress (for each relaxation time)
        """
        # Time range (logarithmic scale)
        time = np.logspace(-2, 3, 500)  # 0.01 to 1000 seconds
    
        E = 1000  # MPa (elastic modulus)
        sigma0 = E * strain0  # Initial stress (MPa)
    
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Stress relaxation curves
        plt.subplot(1, 3, 1)
        for tau in relaxation_times:
            # Maxwell stress relaxation: Ã(t) = Ã0 * exp(-t/Ä)
            stress = sigma0 * np.exp(-time / tau)
            plt.plot(time, stress, linewidth=2, label=f'Ä = {tau} s')
    
        plt.xscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Stress Ã(t) (MPa)', fontsize=12)
        plt.title('Maxwell Model: Stress Relaxation', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.axhline(0, color='k', linewidth=0.8)
    
        # Subplot 2: Relaxation modulus
        plt.subplot(1, 3, 2)
        for tau in relaxation_times:
            # Relaxation modulus: E(t) = E0 * exp(-t/Ä)
            E_t = E * np.exp(-time / tau)
            plt.plot(time, E_t, linewidth=2, label=f'Ä = {tau} s')
    
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Relaxation Modulus E(t) (MPa)', fontsize=12)
        plt.title('Relaxation Modulus vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Creep compliance
        plt.subplot(1, 3, 3)
        sigma_const = 10  # MPa (constant stress)
        for tau in relaxation_times:
            # Maxwell creep: µ(t) = Ã/E + Ãt/· = Ã/E(1 + t/Ä)
            strain = (sigma_const / E) * (1 + time / tau) * 100  # %
            plt.plot(time, strain, linewidth=2, label=f'Ä = {tau} s')
    
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Strain µ(t) (%)', fontsize=12)
        plt.title('Maxwell Model: Creep Behavior', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('maxwell_model.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== Maxwell Model Analysis Results ===")
        print(f"Initial strain: {strain0*100:.1f}%")
        print(f"Initial stress: {sigma0:.1f} MPa")
        print(f"Elastic modulus: {E} MPa\n")
    
        for tau in relaxation_times:
            eta = E * tau  # Viscosity (MPa·s)
            t_half = tau * np.log(2)  # Half-life
            print(f"Relaxation time Ä = {tau} s:")
            print(f"  Viscosity · = {eta} MPa·s")
            print(f"  Stress half-life = {t_half:.2f} s")
    
        return time, relaxation_times
    
    # Execute
    simulate_maxwell_model()
    

### 3.2.2 Voigt Model Simulation

Creep in the Voigt model is described by the following equation: 

\\[ \varepsilon(t) = \frac{\sigma_0}{E} \left(1 - e^{-t/\tau}\right) \\] 

where Ä = ·/E is the retardation time. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Voigt Model Simulation
    def simulate_voigt_model(retardation_times=[1, 10, 100], stress0=10):
        """
        Simulate creep using Voigt model
    
        Parameters:
        - retardation_times: List of retardation times Ä (seconds)
        - stress0: Constant stress (MPa)
    
        Returns:
        - time: Time (seconds)
        - strains: Strain (for each retardation time)
        """
        # Time range
        time = np.logspace(-2, 3, 500)  # 0.01 to 1000 seconds
    
        E = 1000  # MPa (elastic modulus)
        epsilon_eq = stress0 / E  # Equilibrium strain
    
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Creep curves
        plt.subplot(1, 3, 1)
        for tau in retardation_times:
            # Voigt creep: µ(t) = (Ã0/E)(1 - exp(-t/Ä))
            strain = epsilon_eq * (1 - np.exp(-time / tau)) * 100  # %
            plt.plot(time, strain, linewidth=2, label=f'Ä = {tau} s')
    
        plt.xscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Strain µ(t) (%)', fontsize=12)
        plt.title('Voigt Model: Creep Behavior', fontsize=14, fontweight='bold')
        plt.axhline(epsilon_eq * 100, color='red', linestyle='--',
                    linewidth=1.5, label=f'Equilibrium ({epsilon_eq*100:.2f}%)')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Creep compliance
        plt.subplot(1, 3, 2)
        for tau in retardation_times:
            # Creep compliance: J(t) = (1/E)(1 - exp(-t/Ä))
            J_t = (1 / E) * (1 - np.exp(-time / tau)) * 1000  # 1/GPa
            plt.plot(time, J_t, linewidth=2, label=f'Ä = {tau} s')
    
        plt.xscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Creep Compliance J(t) (1/GPa)', fontsize=12)
        plt.title('Creep Compliance vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Recovery curve (after load removal)
        plt.subplot(1, 3, 3)
        time_loading = 100  # Loading time (seconds)
        time_total = np.linspace(0, 300, 500)
    
        for tau in retardation_times:
            strain_recovery = np.zeros_like(time_total)
            for i, t in enumerate(time_total):
                if t <= time_loading:
                    # During loading: Creep
                    strain_recovery[i] = epsilon_eq * (1 - np.exp(-t / tau))
                else:
                    # After load removal: Recovery
                    t_unload = t - time_loading
                    strain_at_unload = epsilon_eq * (1 - np.exp(-time_loading / tau))
                    strain_recovery[i] = strain_at_unload * np.exp(-t_unload / tau)
            plt.plot(time_total, strain_recovery * 100, linewidth=2, label=f'Ä = {tau} s')
    
        plt.axvline(time_loading, color='red', linestyle='--', linewidth=1.5, label='Unload')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Strain µ(t) (%)', fontsize=12)
        plt.title('Voigt Model: Recovery after Unloading', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('voigt_model.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== Voigt Model Analysis Results ===")
        print(f"Constant stress: {stress0} MPa")
        print(f"Equilibrium strain: {epsilon_eq*100:.2f}%")
        print(f"Elastic modulus: {E} MPa\n")
    
        for tau in retardation_times:
            eta = E * tau  # Viscosity (MPa·s)
            t_90 = -tau * np.log(0.1)  # 90% approach time
            print(f"Retardation time Ä = {tau} s:")
            print(f"  Viscosity · = {eta} MPa·s")
            print(f"  90% creep approach time = {t_90:.2f} s")
    
        return time, retardation_times
    
    # Execute
    simulate_voigt_model()
    

## 3.3 Creep and Stress Relaxation

**Creep** is the time-dependent increase in strain under constant stress, while **stress relaxation** is the time-dependent decrease in stress under constant strain. Both are important for understanding the viscoelastic behavior of polymers. 
    
    
    ```mermaid
    flowchart LR
                        A[Viscoelastic Test] --> B[Creep Test]
                        A --> C[Stress Relaxation Test]
                        B --> D[Constant Stress Ã0]
                        D --> E[Measure Strain µ t]
                        E --> F[Creep ComplianceJ t = µ t /Ã0]
                        C --> G[Constant Strain µ0]
                        G --> H[Measure Stress Ã t]
                        H --> I[Relaxation ModulusE t = Ã t /µ0]
    ```

### 3.3.1 Creep Compliance Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Creep Compliance Calculation
    def calculate_creep_compliance(stress=10, times=None):
        """
        Calculate creep compliance from experimental creep data
    
        Parameters:
        - stress: Constant stress (MPa)
        - times: Time array (seconds)
    
        Returns:
        - times: Time (seconds)
        - compliance: Creep compliance (1/GPa)
        """
        if times is None:
            times = np.logspace(-1, 4, 100)  # 0.1 to 10000 seconds
    
        # Simulate experimental creep strain (4-element model)
        # µ(t) = Ã0[J0 + J1(1-exp(-t/Ä1)) + J2(1-exp(-t/Ä2)) + t/·0]
        J0 = 0.2e-3  # Instantaneous compliance (1/GPa)
        J1 = 0.5e-3  # Retarded compliance 1
        tau1 = 10    # Retardation time 1 (seconds)
        J2 = 0.3e-3  # Retarded compliance 2
        tau2 = 1000  # Retardation time 2 (seconds)
        eta0 = 1e6   # Steady-state flow viscosity (GPa·s)
    
        # Creep strain (%)
        strain = stress * (J0 + J1 * (1 - np.exp(-times / tau1)) +
                           J2 * (1 - np.exp(-times / tau2)) +
                           times / eta0) * 100
    
        # Creep compliance (1/GPa)
        compliance = strain / (stress * 100) * 1000
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Creep strain
        plt.subplot(1, 3, 1)
        plt.plot(times, strain, 'b-', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Creep Strain (%)', fontsize=12)
        plt.title(f'Creep Curve (Ã = {stress} MPa)', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # Subplot 2: Creep compliance
        plt.subplot(1, 3, 2)
        plt.plot(times, compliance, 'r-', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Creep Compliance J(t) (1/GPa)', fontsize=12)
        plt.title('Creep Compliance vs Time', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # Subplot 3: Creep rate (dµ/dt)
        plt.subplot(1, 3, 3)
        # Numerical differentiation
        creep_rate = np.gradient(strain, times)
        plt.plot(times, creep_rate, 'g-', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Creep Rate dµ/dt (%/s)', fontsize=12)
        plt.title('Creep Rate vs Time', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('creep_compliance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== Creep Compliance Analysis ===")
        print(f"Constant stress: {stress} MPa")
        print(f"Instantaneous compliance J0: {J0*1000:.3f} 1/GPa")
        print(f"Retardation time 1: {tau1} s, J1: {J1*1000:.3f} 1/GPa")
        print(f"Retardation time 2: {tau2} s, J2: {J2*1000:.3f} 1/GPa")
        print(f"Steady-state flow viscosity: {eta0:.2e} GPa·s")
    
        # Creep strain at specific times
        for t in [1, 10, 100, 1000]:
            idx = np.argmin(np.abs(times - t))
            print(f"\nt = {t} s:")
            print(f"  Creep strain: {strain[idx]:.3f}%")
            print(f"  Compliance: {compliance[idx]:.3f} 1/GPa")
    
        return times, compliance
    
    # Execute
    calculate_creep_compliance()
    

## 3.4 WLF Equation and Time-Temperature Superposition

The viscoelasticity of polymers is strongly temperature-dependent. The **Williams-Landel-Ferry (WLF) equation** enables time-temperature superposition near the glass transition temperature Tg: 

\\[ \log a_T = \frac{-C_1 (T - T_g)}{C_2 + (T - T_g)} \\] 

where aT is the shift factor, and C1 and C2 are material-specific constants (universal constants: C1 = 17.44, C2 = 51.6 K). 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # WLF Time-Temperature Superposition
    def apply_wlf_time_temperature_superposition(tg=373, temperatures=None):
        """
        Perform time-temperature superposition using WLF equation to create master curve
    
        Parameters:
        - tg: Glass transition temperature (K)
        - temperatures: List of measurement temperatures (K)
    
        Returns:
        - shift_factors: Shift factors
        - master_curve: Master curve
        """
        if temperatures is None:
            # Measurement temperatures (±50K around Tg)
            temperatures = tg + np.array([-40, -20, 0, 20, 40])
    
        # WLF constants (universal constants)
        C1 = 17.44
        C2 = 51.6  # K
    
        # Reference temperature (usually Tg)
        T_ref = tg
    
        # Calculate shift factors
        shift_factors = {}
        for T in temperatures:
            log_aT = -C1 * (T - T_ref) / (C2 + (T - T_ref))
            aT = 10**log_aT
            shift_factors[T] = aT
    
        # Generate relaxation modulus at each temperature
        time_base = np.logspace(-5, 5, 100)  # Reference time scale
    
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Relaxation modulus at each temperature
        plt.subplot(1, 3, 1)
        for T in temperatures:
            # Simple relaxation modulus (single relaxation time model)
            tau = 1.0  # Reference relaxation time (seconds)
            E_inf = 1  # MPa (equilibrium modulus)
            E0 = 1000  # MPa (glassy modulus)
            E_t = E_inf + (E0 - E_inf) * np.exp(-time_base / tau)
            plt.plot(time_base, E_t, linewidth=2, label=f'{T-273.15:.0f}°C')
    
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Relaxation Modulus E(t) (MPa)', fontsize=12)
        plt.title('E(t) at Different Temperatures', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Master curve (after time axis shift)
        plt.subplot(1, 3, 2)
        for T in temperatures:
            aT = shift_factors[T]
            time_shifted = time_base * aT  # Shift time axis
    
            # Relaxation modulus (behavior at reference temperature)
            tau_ref = 1.0
            E_inf = 1
            E0 = 1000
            E_t = E_inf + (E0 - E_inf) * np.exp(-time_base / tau_ref)
    
            plt.plot(time_shifted, E_t, linewidth=2, label=f'{T-273.15:.0f}°C')
    
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'Reduced Time (s) at Tref = {T_ref-273.15:.0f}°C', fontsize=12)
        plt.ylabel('Relaxation Modulus E(t) (MPa)', fontsize=12)
        plt.title('Master Curve by WLF Superposition', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Shift factor vs temperature
        plt.subplot(1, 3, 3)
        T_range = np.linspace(tg - 50, tg + 50, 100)
        log_aT_range = -C1 * (T_range - T_ref) / (C2 + (T_range - T_ref))
    
        plt.plot(T_range - 273.15, log_aT_range, 'b-', linewidth=2, label='WLF Equation')
        plt.scatter([T - 273.15 for T in temperatures],
                    [np.log10(shift_factors[T]) for T in temperatures],
                    s=100, c='red', edgecolors='black', linewidths=2, zorder=5, label='Measured')
        plt.axvline(tg - 273.15, color='green', linestyle='--', linewidth=1.5, label=f'Tg = {tg-273.15:.0f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('log(aT)', fontsize=12)
        plt.title('WLF Shift Factor vs Temperature', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('wlf_time_temperature.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== WLF Time-Temperature Superposition Results ===")
        print(f"Glass transition temperature Tg: {tg - 273.15:.0f}°C")
        print(f"WLF constants: C1 = {C1}, C2 = {C2} K")
        print(f"Reference temperature: {T_ref - 273.15:.0f}°C\n")
    
        for T in sorted(temperatures):
            aT = shift_factors[T]
            print(f"Temperature {T - 273.15:.0f}°C:")
            print(f"  Shift factor aT = {aT:.2e}")
            print(f"  log(aT) = {np.log10(aT):.2f}")
    
        return shift_factors, T_range
    
    # Execution example: Polymer with Tg = 100°C (373 K)
    apply_wlf_time_temperature_superposition(tg=373)
    

## 3.5 Dynamic Mechanical Analysis (DMA)

In **Dynamic Mechanical Analysis (DMA)** , oscillatory stress is applied to measure storage modulus (E'), loss modulus (E''), and loss tangent (tan ´). These are useful for analyzing glass transition and molecular motion modes. 

### 3.5.1 Dynamic Viscoelastic Parameter Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Dynamic Viscoelasticity (DMA) Simulation
    def simulate_dma_measurement(tg=373, frequency=1.0):
        """
        Simulate temperature sweep DMA measurement
    
        Parameters:
        - tg: Glass transition temperature (K)
        - frequency: Measurement frequency (Hz)
    
        Returns:
        - temperatures: Temperature (K)
        - E_prime: Storage modulus (MPa)
        - E_double_prime: Loss modulus (MPa)
        - tan_delta: Loss tangent
        """
        # Temperature range (±100K around Tg)
        temperatures = np.linspace(tg - 100, tg + 100, 200)
    
        # Glassy and rubbery moduli
        E_glassy = 3000  # MPa
        E_rubbery = 10   # MPa
    
        # Transition width
        transition_width = 20  # K
    
        # Storage modulus E' (approximated by sigmoid function)
        def sigmoid(T, Tg, width):
            """Sigmoid function"""
            return 1 / (1 + np.exp((T - Tg) / width))
    
        E_prime = E_rubbery + (E_glassy - E_rubbery) * sigmoid(temperatures, tg, transition_width)
    
        # Loss modulus E'' (proportional to derivative of E')
        # Peak is maximum at Tg
        def gaussian_peak(T, Tg, width, amplitude):
            """Gaussian peak"""
            return amplitude * np.exp(-0.5 * ((T - Tg) / width)**2)
    
        E_double_prime = gaussian_peak(temperatures, tg, transition_width, 300)
    
        # Loss tangent tan(´) = E''/E'
        tan_delta = E_double_prime / E_prime
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: E' and E''
        plt.subplot(1, 3, 1)
        plt.plot(temperatures - 273.15, E_prime, 'b-', linewidth=2, label="E' (Storage Modulus)")
        plt.plot(temperatures - 273.15, E_double_prime, 'r-', linewidth=2, label='E" (Loss Modulus)')
        plt.axvline(tg - 273.15, color='green', linestyle='--', linewidth=1.5, label=f'Tg = {tg-273.15:.0f}°C')
        plt.yscale('log')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Modulus (MPa)', fontsize=12)
        plt.title(f'DMA: E\' and E" vs Temperature (f = {frequency} Hz)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: tan ´
        plt.subplot(1, 3, 2)
        plt.plot(temperatures - 273.15, tan_delta, 'purple', linewidth=2)
        plt.axvline(tg - 273.15, color='green', linestyle='--', linewidth=1.5, label=f'Tg = {tg-273.15:.0f}°C')
        # Detect tan ´ peak position
        tg_from_tan_delta = temperatures[np.argmax(tan_delta)]
        plt.axvline(tg_from_tan_delta - 273.15, color='red', linestyle=':', linewidth=1.5,
                    label=f'Tg (tan ´ peak) = {tg_from_tan_delta-273.15:.0f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('tan ´', fontsize=12)
        plt.title('Loss Tangent vs Temperature', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Frequency dependence
        plt.subplot(1, 3, 3)
        frequencies = [0.1, 1.0, 10.0]  # Hz
        for freq in frequencies:
            # Higher frequency shifts Tg to higher temperature (time-temperature superposition)
            # Simply modeled as Tg_app = Tg + k*log(f)
            k = 5  # K/decade
            tg_app = tg + k * np.log10(freq / 1.0)
            tan_delta_freq = E_double_prime / (E_rubbery + (E_glassy - E_rubbery) *
                                                sigmoid(temperatures, tg_app, transition_width))
            plt.plot(temperatures - 273.15, tan_delta_freq, linewidth=2, label=f'{freq} Hz')
    
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('tan ´', fontsize=12)
        plt.title('Frequency Dependence of tan ´', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('dma_dynamic_mechanical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== DMA Analysis Results ===")
        print(f"Measurement frequency: {frequency} Hz")
        print(f"Glass transition temperature Tg: {tg - 273.15:.0f}°C")
        print(f"Glassy modulus E': {E_glassy} MPa")
        print(f"Rubbery modulus E': {E_rubbery} MPa")
        print(f"\ntan ´ peak position: {tg_from_tan_delta - 273.15:.1f}°C")
        print(f"Maximum tan ´ value: {np.max(tan_delta):.3f}")
    
        # Values at specific temperatures
        for T_target in [tg - 50, tg, tg + 50]:
            idx = np.argmin(np.abs(temperatures - T_target))
            print(f"\nTemperature {T_target - 273.15:.0f}°C:")
            print(f"  E' = {E_prime[idx]:.1f} MPa")
            print(f"  E'' = {E_double_prime[idx]:.1f} MPa")
            print(f"  tan ´ = {tan_delta[idx]:.3f}")
    
        return temperatures, E_prime, E_double_prime, tan_delta
    
    # Execute
    simulate_dma_measurement(tg=373, frequency=1.0)
    

### 3.5.2 Rheological Flow Curves

The rheological behavior of polymer melts is characterized by the relationship between shear rate and viscosity (flow curve). Many polymers exhibit **shear-thinning**. 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Rheological Flow Curve
    def simulate_rheological_flow_curve():
        """
        Simulate flow curves for polymer melts (Cross/Carreau-Yasuda model)
    
        Returns:
        - shear_rates: Shear rate (1/s)
        - viscosities: Viscosity (Pa·s)
        """
        # Shear rate range (logarithmic scale)
        shear_rates = np.logspace(-3, 3, 100)  # 0.001 to 1000 1/s
    
        # Cross model: ·(³) = ·_inf + (·0 - ·_inf) / (1 + (»³)^m)
        eta0 = 10000    # Zero-shear viscosity (Pa·s)
        eta_inf = 100   # Infinite-shear viscosity (Pa·s)
        lambda_c = 1.0  # Relaxation time (s)
        m = 0.7         # Power index
    
        viscosity_cross = eta_inf + (eta0 - eta_inf) / (1 + (lambda_c * shear_rates)**m)
    
        # Power law model: · = K * ³^(n-1)
        K = 1000  # Consistency coefficient (Pa·s^n)
        n = 0.3   # Power index (n < 1 for shear-thinning)
        viscosity_power_law = K * shear_rates**(n - 1)
    
        # Visualization
        plt.figure(figsize=(14, 5))
    
        # Subplot 1: Viscosity-shear rate curve
        plt.subplot(1, 3, 1)
        plt.plot(shear_rates, viscosity_cross, 'b-', linewidth=2, label='Cross Model')
        plt.plot(shear_rates, viscosity_power_law, 'r--', linewidth=2, label='Power Law Model')
        plt.axhline(eta0, color='green', linestyle=':', linewidth=1.5, label=f'·0 = {eta0} Pa·s')
        plt.axhline(eta_inf, color='purple', linestyle=':', linewidth=1.5, label=f'· = {eta_inf} Pa·s')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Shear Rate ³ (1/s)', fontsize=12)
        plt.ylabel('Viscosity · (Pa·s)', fontsize=12)
        plt.title('Flow Curve: Viscosity vs Shear Rate', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 2: Shear stress-shear rate
        plt.subplot(1, 3, 2)
        shear_stress_cross = viscosity_cross * shear_rates
        shear_stress_power_law = viscosity_power_law * shear_rates
        plt.plot(shear_rates, shear_stress_cross, 'b-', linewidth=2, label='Cross Model')
        plt.plot(shear_rates, shear_stress_power_law, 'r--', linewidth=2, label='Power Law Model')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Shear Rate ³ (1/s)', fontsize=12)
        plt.ylabel('Shear Stress Ä (Pa)', fontsize=12)
        plt.title('Shear Stress vs Shear Rate', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Subplot 3: Temperature dependence (Arrhenius equation)
        plt.subplot(1, 3, 3)
        temperatures = np.linspace(150, 250, 50) + 273.15  # K
        Ea = 50000  # Activation energy (J/mol)
        R = 8.314   # Gas constant
        T_ref = 200 + 273.15  # Reference temperature (K)
        eta_ref = 1000  # Reference viscosity (Pa·s)
    
        # Arrhenius equation: ·(T) = ·_ref * exp(Ea/R * (1/T - 1/T_ref))
        viscosity_temp = eta_ref * np.exp(Ea / R * (1 / temperatures - 1 / T_ref))
    
        plt.plot(temperatures - 273.15, viscosity_temp, 'g-', linewidth=2)
        plt.axvline(T_ref - 273.15, color='red', linestyle='--', linewidth=1.5,
                    label=f'Tref = {T_ref-273.15:.0f}°C')
        plt.yscale('log')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Viscosity · (Pa·s)', fontsize=12)
        plt.title('Temperature Dependence (Arrhenius)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('rheology_flow_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Results output
        print("=== Rheological Flow Curve Analysis ===")
        print("Cross model parameters:")
        print(f"  Zero-shear viscosity ·0: {eta0} Pa·s")
        print(f"  Infinite-shear viscosity ·: {eta_inf} Pa·s")
        print(f"  Relaxation time »: {lambda_c} s")
        print(f"  Power index m: {m}")
    
        print("\nPower law model parameters:")
        print(f"  Consistency coefficient K: {K} Pa·s^n")
        print(f"  Power index n: {n} (shear-thinning)")
    
        # Values at specific shear rates
        for gamma_dot in [0.01, 1.0, 100]:
            idx = np.argmin(np.abs(shear_rates - gamma_dot))
            print(f"\nShear rate {gamma_dot} 1/s:")
            print(f"  Viscosity (Cross): {viscosity_cross[idx]:.1f} Pa·s")
            print(f"  Shear stress: {shear_stress_cross[idx]:.1f} Pa")
    
        return shear_rates, viscosity_cross
    
    # Execute
    simulate_rheological_flow_curve()
    

## Exercises

#### Exercise 1: Young's Modulus Calculation (Easy)

Calculate Young's modulus when stress is 10 MPa and strain is 0.5%. 

View Solution

**Solution:**
    
    
    stress = 10  # MPa
    strain = 0.5 / 100  # Convert % to decimal
    
    E = stress / strain
    print(f"Young's modulus E = {E} MPa = {E/1000} GPa")
    # Output: Young's modulus E = 2000 MPa = 2.0 GPa

This value corresponds to semicrystalline polymers (PE, PP).

#### Exercise 2: Maxwell Relaxation Time (Easy)

Calculate the Maxwell relaxation time Ä when E = 1000 MPa and · = 10,000 MPa·s. 

View Solution

**Solution:**
    
    
    E = 1000  # MPa
    eta = 10000  # MPa·s
    
    tau = eta / E
    print(f"Relaxation time Ä = {tau} s")
    # Output: Relaxation time Ä = 10 s

Stress decreases to approximately 37% (1/e) in 10 seconds.

#### Exercise 3: WLF Shift Factor (Easy)

Calculate the WLF shift factor aT when Tg = 100°C, T = 120°C, C1 = 17.44, C2 = 51.6 K. 

View Solution

**Solution:**
    
    
    Tg = 373  # K
    T = 393  # K
    C1 = 17.44
    C2 = 51.6  # K
    
    log_aT = -C1 * (T - Tg) / (C2 + (T - Tg))
    aT = 10**log_aT
    
    print(f"log(aT) = {log_aT:.3f}")
    print(f"Shift factor aT = {aT:.3e}")
    # Output: log(aT) H -4.88, aT H 1.3e-5

At 120°C, relaxation is approximately 100,000 times faster.

#### Exercise 4: Creep Compliance (Medium)

Under constant stress of 5 MPa, the strain after 10 seconds was 0.8%. Calculate the creep compliance J(10s) (units: 1/GPa). 

View Solution

**Solution:**
    
    
    stress = 5  # MPa
    strain = 0.8 / 100  # Decimal
    time = 10  # s
    
    # J(t) = µ(t) / Ã
    J_t = strain / stress  # 1/MPa
    J_t_GPa = J_t * 1000   # 1/GPa
    
    print(f"Creep compliance J(10s) = {J_t_GPa:.3f} 1/GPa")
    # Output: J(10s) = 0.160 1/GPa

#### Exercise 5: tan ´ and Tg (Medium)

In DMA measurement, the tan ´ peak was observed at 95°C. Estimate the glass transition temperature Tg (at 1 Hz frequency). 

View Solution

**Solution:**

The tan ´ peak temperature typically appears 5-10°C higher than Tg (at 1 Hz frequency). Therefore, Tg H 85-90°C is estimated. This approximately matches Tg measured by DSC (heating rate 10 K/min).

#### Exercise 6: Shear Viscosity Calculation (Medium)

Calculate the apparent viscosity when shear rate is 10 1/s and shear stress is 5000 Pa. 

View Solution

**Solution:**
    
    
    shear_rate = 10  # 1/s
    shear_stress = 5000  # Pa
    
    # · = Ä / ³
    viscosity = shear_stress / shear_rate
    
    print(f"Apparent viscosity · = {viscosity} Pa·s")
    # Output: Apparent viscosity · = 500 Pa·s

This is a moderate viscosity, suitable for injection molding.

#### Exercise 7: Voigt Recovery Time (Medium)

For a Voigt model (E = 1000 MPa, · = 5000 MPa·s), calculate the time required for strain to recover to 10% of its initial value after load removal. 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    E = 1000  # MPa
    eta = 5000  # MPa·s
    tau = eta / E  # Retardation time
    
    # Recovery: µ(t) = µ0 * exp(-t/Ä)
    # µ(t) / µ0 = 0.1 ’ exp(-t/Ä) = 0.1
    # t = -Ä * ln(0.1)
    
    import numpy as np
    t_recovery = -tau * np.log(0.1)
    
    print(f"Retardation time Ä = {tau} s")
    print(f"90% recovery time = {t_recovery:.2f} s")
    # Output: Retardation time Ä = 5 s, 90% recovery time = 11.51 s

#### Exercise 8: Master Curve Creation (Hard)

Create a master curve by shifting relaxation modulus data measured at three temperatures (80°C, 100°C, 120°C) using the WLF equation with Tg = 100°C as reference. Use C1 = 17.44, C2 = 51.6 K. 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Parameters
    Tg = 373  # K (100°C)
    C1, C2 = 17.44, 51.6
    temperatures = [353, 373, 393]  # K (80, 100, 120°C)
    
    # Reference time scale
    time_base = np.logspace(-2, 4, 100)
    
    # WLF shift factor
    def wlf_shift(T, Tref):
        return 10**(-C1 * (T - Tref) / (C2 + (T - Tref)))
    
    # Master curve plot
    plt.figure(figsize=(10, 6))
    for T in temperatures:
        aT = wlf_shift(T, Tg)
        time_shifted = time_base * aT
        # Simple relaxation modulus
        E_t = 10 + 2990 * np.exp(-time_base / 1.0)
        plt.plot(time_shifted, E_t, 'o-', label=f'{T-273.15:.0f}°C (aT={aT:.2e})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reduced Time (s)', fontsize=12)
    plt.ylabel('E(t) (MPa)', fontsize=12)
    plt.title('Master Curve at Tref = 100°C', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("If all data collapse onto a single curve, master curve creation is successful")

#### Exercise 9: Tg Determination by DMA (Hard)

The tan ´ peak temperatures measured at frequencies 0.1, 1.0, 10 Hz were 85, 95, 105°C. Estimate Tg by extrapolating to 0 Hz frequency. 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Experimental data
    frequencies = np.array([0.1, 1.0, 10.0])  # Hz
    tan_delta_peaks = np.array([85, 95, 105])  # °C
    
    # Linear fitting: Tg_app = Tg + k*log10(f)
    def linear_model(log_f, Tg, k):
        return Tg + k * log_f
    
    log_frequencies = np.log10(frequencies)
    params, _ = curve_fit(linear_model, log_frequencies, tan_delta_peaks)
    Tg_extrapolated, k = params
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(log_frequencies, tan_delta_peaks, s=100, c='red', edgecolors='black', zorder=5)
    log_f_fit = np.linspace(-2, 2, 100)
    Tg_fit = linear_model(log_f_fit, Tg_extrapolated, k)
    plt.plot(log_f_fit, Tg_fit, 'b-', linewidth=2)
    plt.axhline(Tg_extrapolated, color='green', linestyle='--', label=f'Tg (f’0) = {Tg_extrapolated:.1f}°C')
    plt.xlabel('log(Frequency) [Hz]', fontsize=12)
    plt.ylabel('tan ´ Peak Temperature (°C)', fontsize=12)
    plt.title('Frequency Dependence of Tg', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Extrapolated Tg (f’0): {Tg_extrapolated:.1f}°C")
    print(f"Frequency dependence k: {k:.2f} K/decade")
    # Example output: Tg H 75°C, k H 10 K/decade

#### Exercise 10: Rheology Optimization (Hard)

For injection molding with shear rate range 100-1000 1/s, optimize Cross model parameters (·0, », m) to control viscosity in the range 500-1000 Pa·s. 

View Solution

**Solution:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    # Target: viscosity 500-1000 Pa·s at 100-1000 1/s
    target_shear_rates = np.array([100, 1000])
    target_viscosities = np.array([1000, 500])
    
    # Cross model: · = ·_inf + (·0 - ·_inf) / (1 + (»³)^m)
    eta_inf = 100  # Fixed
    
    def cross_viscosity(gamma_dot, params):
        eta0, lambda_c, m = params
        return eta_inf + (eta0 - eta_inf) / (1 + (lambda_c * gamma_dot)**m)
    
    # Optimization: minimize residual with target viscosity
    def objective(params):
        predicted = cross_viscosity(target_shear_rates, params)
        return np.sum((predicted - target_viscosities)**2)
    
    # Initial values and optimization
    initial_params = [5000, 0.01, 0.7]
    result = minimize(objective, initial_params, bounds=[(1000, 20000), (0.001, 1.0), (0.3, 1.0)])
    eta0_opt, lambda_opt, m_opt = result.x
    
    print("=== Optimization Results ===")
    print(f"·0 = {eta0_opt:.0f} Pa·s")
    print(f"» = {lambda_opt:.4f} s")
    print(f"m = {m_opt:.3f}")
    
    # Verification plot
    gamma_range = np.logspace(1, 3, 100)
    eta_optimized = cross_viscosity(gamma_range, result.x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_range, eta_optimized, 'b-', linewidth=2, label='Optimized Cross Model')
    plt.scatter(target_shear_rates, target_viscosities, s=100, c='red', edgecolors='black',
                zorder=5, label='Target Values')
    plt.fill_between(gamma_range, 500, 1000, alpha=0.2, color='green', label='Target Range')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Shear Rate (1/s)', fontsize=12)
    plt.ylabel('Viscosity (Pa·s)', fontsize=12)
    plt.title('Optimized Rheology for Injection Molding', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## References

  1. Ward, I. M., & Sweeney, J. (2012). _An Introduction to the Mechanical Properties of Solid Polymers_ (3rd ed.). Wiley. pp. 1-105, 220-295.
  2. Ferry, J. D. (1980). _Viscoelastic Properties of Polymers_ (3rd ed.). Wiley. pp. 30-125, 280-340.
  3. Osswald, T. A., & Rudolph, N. (2015). _Polymer Rheology: Fundamentals and Applications_. Hanser. pp. 15-90.
  4. Menard, K. P., & Menard, N. (2008). _Dynamic Mechanical Analysis: A Practical Introduction_ (2nd ed.). CRC Press. pp. 1-75.
  5. Dealy, J. M., & Wissbrun, K. F. (1990). _Melt Rheology and Its Role in Plastics Processing_. Springer. pp. 50-145.
  6. Williams, M. L., Landel, R. F., & Ferry, J. D. (1955). _J. Am. Chem. Soc._ , 77, 3701-3707. (WLF equation)

### Connection to Next Chapter

In Chapter 4, you will learn about **functional polymers** such as conductive polymers, biocompatible polymers, and stimuli-responsive polymers. The knowledge of viscoelasticity learned in this chapter directly relates to understanding the behavior of biomaterials as soft matter and analyzing phase transitions in stimuli-responsive polymers. It is also applied to the design of conductive polymers that balance both electrical and mechanical properties. 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
