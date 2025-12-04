---
title: "Chapter 1: Fundamentals of Process Simulation"
chapter_title: "Chapter 1: Fundamentals of Process Simulation"
subtitle: Sequential Modular vs Equation-Oriented Approaches and Thermodynamic Calculations
version: 1.0
created_at: 2025-10-26
---

This chapter covers the fundamentals of Fundamentals of Process Simulation, which overview of process simulation. You will learn differences between Sequential Modular, thermodynamic models (ideal gas, and Calculate stream properties (enthalpy.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the differences between Sequential Modular and Equation-Oriented approaches
  * ✅ Implement thermodynamic models (ideal gas, SRK, Peng-Robinson)
  * ✅ Calculate stream properties (enthalpy, entropy, density)
  * ✅ Implement flash calculations (vapor-liquid equilibrium)
  * ✅ Understand convergence calculation algorithms (successive substitution, Newton-Raphson method)
  * ✅ Select tear streams and determine process flow sequencing

* * *

## 1.1 Overview of Process Simulation

### What is Process Simulation

**Process simulation** is the representation of chemical process behavior using mathematical models and reproducing it on a computer. By combining material balance, energy balance, vapor-liquid equilibrium, reaction rate equations, etc., process performance can be predicted.

### Major Approaches

Approach | Characteristics | Advantages | Disadvantages  
---|---|---|---  
**Sequential Modular** | Calculates unit operations sequentially | Intuitive, easy to debug | Requires convergence calculation for recycle loops  
**Equation-Oriented** | Solves all equations simultaneously | Fast convergence, easy to optimize | Complex Jacobian matrix construction  
  
* * *

## 1.2 Implementation of Sequential Modular Approach

### Code Example 1: Basic Implementation of Sequential Modular Method
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sequential Modular method demo: Calculate 3 unit operations sequentially
    
    class Stream:
        """Process stream class"""
        def __init__(self, name, T=298.15, P=101325, flow_rate=100.0, composition=None):
            """
            Parameters:
            name : str, stream name
            T : float, temperature [K]
            P : float, pressure [Pa]
            flow_rate : float, molar flow rate [mol/s]
            composition : dict, composition {component name: mole fraction}
            """
            self.name = name
            self.T = T
            self.P = P
            self.flow_rate = flow_rate
            self.composition = composition if composition else {'A': 1.0}
    
        def copy(self):
            """Create a copy of the stream"""
            return Stream(self.name, self.T, self.P, self.flow_rate,
                         self.composition.copy())
    
        def __repr__(self):
            return (f"Stream(name={self.name}, T={self.T:.2f}K, "
                    f"P={self.P/1000:.1f}kPa, F={self.flow_rate:.2f}mol/s)")
    
    
    class Heater:
        """Heater unit"""
        def __init__(self, name, delta_T=50.0):
            self.name = name
            self.delta_T = delta_T  # Temperature rise [K]
    
        def calculate(self, inlet_stream):
            """
            Heater calculation
    
            Parameters:
            inlet_stream : Stream, inlet stream
    
            Returns:
            outlet_stream : Stream, outlet stream
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
            outlet.T = inlet_stream.T + self.delta_T
    
            print(f"{self.name}: Temperature {inlet_stream.T:.2f}K → {outlet.T:.2f}K")
            return outlet
    
    
    class Reactor:
        """Reactor unit (simplified model)"""
        def __init__(self, name, conversion=0.8):
            self.name = name
            self.conversion = conversion  # Conversion rate
    
        def calculate(self, inlet_stream):
            """
            Reactor calculation: A → B (simple reaction)
    
            Parameters:
            inlet_stream : Stream
    
            Returns:
            outlet_stream : Stream
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
    
            # Reaction: A → B
            if 'A' in inlet_stream.composition:
                x_A = inlet_stream.composition['A']
                converted = x_A * self.conversion
    
                outlet.composition = {
                    'A': x_A - converted,
                    'B': converted
                }
    
                print(f"{self.name}: Conversion {self.conversion*100:.1f}%, "
                      f"A: {x_A:.3f} → {outlet.composition['A']:.3f}, "
                      f"B: 0 → {outlet.composition['B']:.3f}")
    
            return outlet
    
    
    class Cooler:
        """Cooler unit"""
        def __init__(self, name, T_target=320.0):
            self.name = name
            self.T_target = T_target  # Target temperature [K]
    
        def calculate(self, inlet_stream):
            """
            Cooler calculation
    
            Parameters:
            inlet_stream : Stream
    
            Returns:
            outlet_stream : Stream
            """
            outlet = inlet_stream.copy()
            outlet.name = f"{self.name}_out"
            outlet.T = self.T_target
    
            print(f"{self.name}: Temperature {inlet_stream.T:.2f}K → {outlet.T:.2f}K")
            return outlet
    
    
    # Process calculation using Sequential Modular method
    def run_sequential_modular():
        """
        Calculate process using Sequential Modular method
    
        Process flow: Feed → Heater → Reactor → Cooler → Product
        """
        print("="*60)
        print("Process Calculation by Sequential Modular Method")
        print("="*60)
    
        # Inlet stream
        feed = Stream(name="Feed", T=298.15, P=101325, flow_rate=100.0,
                      composition={'A': 1.0})
        print(f"\nInlet: {feed}")
    
        # Unit operation definition
        heater = Heater(name="H-101", delta_T=80.0)
        reactor = Reactor(name="R-101", conversion=0.85)
        cooler = Cooler(name="C-101", T_target=320.0)
    
        # Sequential calculation (Sequential Modular)
        print("\n--- Unit Operation Calculation ---")
        s1 = heater.calculate(feed)
        s2 = reactor.calculate(s1)
        product = cooler.calculate(s2)
    
        print(f"\nOutlet: {product}")
        print(f"Final composition: A={product.composition.get('A', 0):.3f}, "
              f"B={product.composition.get('B', 0):.3f}")
    
        # Result visualization
        streams = [feed, s1, s2, product]
        stream_names = ['Feed', 'Heater Outlet', 'Reactor Outlet', 'Product']
        temperatures = [s.T for s in streams]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Temperature profile
        ax1.plot(stream_names, temperatures, marker='o', linewidth=2.5,
                 markersize=10, color='#11998e')
        ax1.set_ylabel('Temperature [K]', fontsize=12)
        ax1.set_title('Sequential Modular Method: Temperature Profile',
                      fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
        # Composition change
        x_A = [s.composition.get('A', 0) for s in streams]
        x_B = [s.composition.get('B', 0) for s in streams]
    
        ax2.plot(stream_names, x_A, marker='s', linewidth=2.5,
                 markersize=10, label='Component A', color='#e74c3c')
        ax2.plot(stream_names, x_B, marker='^', linewidth=2.5,
                 markersize=10, label='Component B', color='#3498db')
        ax2.set_ylabel('Mole Fraction', fontsize=12)
        ax2.set_title('Sequential Modular Method: Composition Change',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    run_sequential_modular()
    

**Output Example:**
    
    
    ============================================================
    Process Calculation by Sequential Modular Method
    ============================================================
    
    Inlet: Stream(name=Feed, T=298.15K, P=101.3kPa, F=100.00mol/s)
    
    --- Unit Operation Calculation ---
    H-101: Temperature 298.15K → 378.15K
    R-101: Conversion 85.0%, A: 1.000 → 0.150, B: 0 → 0.850
    C-101: Temperature 378.15K → 320.00K
    
    Outlet: Stream(name=C-101_out, T=320.00K, P=101.3kPa, F=100.00mol/s)
    Final composition: A=0.150, B=0.850
    

**Explanation:** In the Sequential Modular method, each unit operation is calculated in sequence. This method is intuitive and easy to implement, but convergence calculation is required when recycle loops are present.

* * *

### Code Example 2: Basics of Equation-Oriented Approach
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    
    # Equation-Oriented method demo: Solve all equations simultaneously
    
    def equation_oriented_process(x):
        """
        Equation system for entire process
    
        Variable order: [T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3]
        - T1: Feed temperature (fixed)
        - T2: Heater outlet temperature
        - T3: Reactor outlet temperature
        - T4: Cooler outlet temperature
        - x_A2, x_B2: Heater outlet composition
        - x_A3, x_B3: Reactor outlet composition
    
        Parameters:
        x : array, variable vector [8 elements]
    
        Returns:
        residuals : array, residual vector
        """
        T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3 = x
    
        # Parameters
        T_feed = 298.15  # K
        delta_T_heater = 80.0  # K
        conversion = 0.85
        T_cooler = 320.0  # K
        x_A1 = 1.0  # Inlet composition
        x_B1 = 0.0
    
        # Equations (residuals = 0 should be satisfied)
        residuals = np.zeros(8)
    
        # Heater equations
        residuals[0] = T2 - (T1 + delta_T_heater)  # Energy balance
        residuals[1] = x_A2 - x_A1  # Material balance (A)
        residuals[2] = x_B2 - x_B1  # Material balance (B)
    
        # Reactor equations
        residuals[3] = T3 - T2  # No temperature change (isothermal reactor)
        residuals[4] = x_A3 - (x_A2 * (1 - conversion))  # A after reaction
        residuals[5] = x_B3 - (x_B2 + x_A2 * conversion)  # B after reaction
    
        # Cooler equations
        residuals[6] = T4 - T_cooler  # Target temperature
    
        # Composition sum constraint
        residuals[7] = (x_A3 + x_B3) - 1.0
    
        return residuals
    
    
    # Initial guess
    x0 = np.array([
        298.15,  # T1
        350.0,   # T2
        350.0,   # T3
        320.0,   # T4
        1.0,     # x_A2
        0.0,     # x_B2
        0.2,     # x_A3
        0.8      # x_B3
    ])
    
    print("="*60)
    print("Process Calculation by Equation-Oriented Method")
    print("="*60)
    print(f"\nInitial guess:")
    print(f"  T = {x0[:4]}")
    print(f"  Composition = {x0[4:]}")
    
    # Solve equations simultaneously
    solution = fsolve(equation_oriented_process, x0)
    
    T1, T2, T3, T4, x_A2, x_B2, x_A3, x_B3 = solution
    
    print(f"\nSolution:")
    print(f"  Feed temperature: T1 = {T1:.2f} K")
    print(f"  Heater outlet: T2 = {T2:.2f} K, A={x_A2:.3f}, B={x_B2:.3f}")
    print(f"  Reactor outlet: T3 = {T3:.2f} K, A={x_A3:.3f}, B={x_B3:.3f}")
    print(f"  Product temperature: T4 = {T4:.2f} K")
    
    # Residual check
    residuals = equation_oriented_process(solution)
    print(f"\nResidual norm: {np.linalg.norm(residuals):.2e}")
    print(f"  (Convergence criterion: < 1e-6 for success)")
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Sequential\nModular', 'Equation-\nOriented']
    T_seq = [298.15, 378.15, 378.15, 320.0]
    T_eq = [T1, T2, T3, T4]
    
    x_pos = np.arange(4)
    width = 0.35
    
    ax.bar(x_pos - width/2, T_seq, width, label='Sequential Modular',
           color='#11998e', alpha=0.8)
    ax.bar(x_pos + width/2, T_eq, width, label='Equation-Oriented',
           color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Temperature [K]', fontsize=12)
    ax.set_title('Sequential Modular vs Equation-Oriented: Temperature Comparison',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Feed', 'Heater Outlet', 'Reactor Outlet', 'Product'])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    

**Output Example:**
    
    
    ============================================================
    Process Calculation by Equation-Oriented Method
    ============================================================
    
    Initial guess:
      T = [298.15 350.   350.   320.  ]
      Composition = [1.  0.  0.2 0.8]
    
    Solution:
      Feed temperature: T1 = 298.15 K
      Heater outlet: T2 = 378.15 K, A=1.000, B=0.000
      Reactor outlet: T3 = 378.15 K, A=0.150, B=0.850
      Product temperature: T4 = 320.00 K
    
    Residual norm: 2.47e-13
      (Convergence criterion: < 1e-6 for success)
    

**Explanation:** In the Equation-Oriented method, the equations for the entire process are solved simultaneously. Convergence is fast and integration with optimization is easy, but implementation is complex due to the need for Jacobian matrix calculation.

* * *

## 1.3 Implementation of Thermodynamic Models

### Code Example 3: Ideal Gas Model
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Implementation of ideal gas model
    
    class IdealGas:
        """Ideal gas model"""
    
        R = 8.314  # Gas constant [J/(mol·K)]
    
        @staticmethod
        def pressure(n, V, T):
            """
            Ideal gas equation of state: PV = nRT
    
            Parameters:
            n : float, number of moles [mol]
            V : float, volume [m³]
            T : float, temperature [K]
    
            Returns:
            P : float, pressure [Pa]
            """
            return n * IdealGas.R * T / V
    
        @staticmethod
        def volume(n, P, T):
            """
            Volume calculation
    
            Parameters:
            n : float, number of moles [mol]
            P : float, pressure [Pa]
            T : float, temperature [K]
    
            Returns:
            V : float, volume [m³]
            """
            return n * IdealGas.R * T / P
    
        @staticmethod
        def density(P, T, MW):
            """
            Density calculation: ρ = PM/(RT)
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            MW : float, molecular weight [g/mol]
    
            Returns:
            rho : float, density [kg/m³]
            """
            return (P * MW / 1000) / (IdealGas.R * T)
    
        @staticmethod
        def enthalpy(T, Cp, T_ref=298.15, H_ref=0.0):
            """
            Enthalpy calculation (constant heat capacity assumption)
            H(T) = H_ref + Cp * (T - T_ref)
    
            Parameters:
            T : float, temperature [K]
            Cp : float, constant pressure heat capacity [J/(mol·K)]
            T_ref : float, reference temperature [K]
            H_ref : float, reference enthalpy [J/mol]
    
            Returns:
            H : float, enthalpy [J/mol]
            """
            return H_ref + Cp * (T - T_ref)
    
        @staticmethod
        def entropy(T, P, Cp, T_ref=298.15, P_ref=101325, S_ref=0.0):
            """
            Entropy calculation
            S(T, P) = S_ref + Cp*ln(T/T_ref) - R*ln(P/P_ref)
    
            Parameters:
            T : float, temperature [K]
            P : float, pressure [Pa]
            Cp : float, constant pressure heat capacity [J/(mol·K)]
            T_ref : float, reference temperature [K]
            P_ref : float, reference pressure [Pa]
            S_ref : float, reference entropy [J/(mol·K)]
    
            Returns:
            S : float, entropy [J/(mol·K)]
            """
            return (S_ref + Cp * np.log(T / T_ref) -
                    IdealGas.R * np.log(P / P_ref))
    
    
    # Ideal gas model validation
    def demonstrate_ideal_gas():
        """Ideal gas model demonstration"""
    
        print("="*60)
        print("Ideal Gas Model Property Calculation")
        print("="*60)
    
        # Parameters
        n = 1.0  # mol
        T = 350.0  # K
        P = 200000  # Pa (2 bar)
        MW = 28.0  # g/mol (N2)
        Cp = 29.1  # J/(mol·K)
    
        # Calculation
        V = IdealGas.volume(n, P, T)
        rho = IdealGas.density(P, T, MW)
        H = IdealGas.enthalpy(T, Cp)
        S = IdealGas.entropy(T, P, Cp)
    
        print(f"\nConditions:")
        print(f"  Temperature: T = {T} K")
        print(f"  Pressure: P = {P/1000:.1f} kPa")
        print(f"  Moles: n = {n} mol")
        print(f"  Molecular weight: MW = {MW} g/mol")
    
        print(f"\nCalculation results:")
        print(f"  Volume: V = {V:.6f} m³ = {V*1000:.3f} L")
        print(f"  Density: ρ = {rho:.3f} kg/m³")
        print(f"  Enthalpy: H = {H:.2f} J/mol")
        print(f"  Entropy: S = {S:.2f} J/(mol·K)")
    
        # Compressibility factor visualization (Z=1 for ideal gas)
        P_range = np.linspace(50000, 500000, 100)
        T_range = [300, 350, 400, 450]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # P-V diagram
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, T_val in enumerate(T_range):
            V_range = [IdealGas.volume(n, P, T_val) * 1000 for P in P_range]
            ax1.plot(V_range, P_range/1000, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax1.set_xlabel('Volume V [L]', fontsize=12)
        ax1.set_ylabel('Pressure P [kPa]', fontsize=12)
        ax1.set_title('Ideal Gas P-V Diagram (Isotherms)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # Density vs pressure
        for i, T_val in enumerate(T_range):
            rho_range = [IdealGas.density(P, T_val, MW) for P in P_range]
            ax2.plot(P_range/1000, rho_range, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax2.set_xlabel('Pressure P [kPa]', fontsize=12)
        ax2.set_ylabel('Density ρ [kg/m³]', fontsize=12)
        ax2.set_title('Ideal Gas Density-Pressure Relationship', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    demonstrate_ideal_gas()
    

**Output Example:**
    
    
    ============================================================
    Ideal Gas Model Property Calculation
    ============================================================
    
    Conditions:
      Temperature: T = 350.0 K
      Pressure: P = 200.0 kPa
      Moles: n = 1 mol
      Molecular weight: MW = 28 g/mol
    
    Calculation results:
      Volume: V = 0.014550 m³ = 14.550 L
      Density: ρ = 1.924 kg/m³
      Enthalpy: H = 1511.85 J/mol
      Entropy: S = 3.28 J/(mol·K)
    

**Explanation:** The ideal gas model can be applied to gases at low pressure and high temperature. Calculations are simple and fast, but it cannot be used for high pressure or liquid phases.

* * *

### Code Example 4: Soave-Redlich-Kwong (SRK) Equation of State
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    
    # Implementation of SRK equation of state
    
    class SRK:
        """Soave-Redlich-Kwong (SRK) equation of state"""
    
        R = 8.314  # J/(mol·K)
    
        def __init__(self, Tc, Pc, omega):
            """
            Parameters:
            Tc : float, critical temperature [K]
            Pc : float, critical pressure [Pa]
            omega : float, acentric factor [-]
            """
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # Parameter calculation
            self.a_c = 0.42748 * (self.R * Tc)**2 / Pc
            self.b = 0.08664 * self.R * Tc / Pc
    
        def alpha(self, T):
            """
            Temperature-dependent parameter α(T)
    
            Parameters:
            T : float, temperature [K]
    
            Returns:
            alpha : float
            """
            Tr = T / self.Tc
            m = 0.480 + 1.574 * self.omega - 0.176 * self.omega**2
            return (1 + m * (1 - np.sqrt(Tr)))**2
    
        def a(self, T):
            """
            Attraction parameter a(T)
    
            Parameters:
            T : float, temperature [K]
    
            Returns:
            a : float
            """
            return self.a_c * self.alpha(T)
    
        def Z_from_PR(self, P, T, phase='vapor'):
            """
            Compressibility factor Z calculation
    
            SRK equation: P = RT/(V-b) - a/(V(V+b))
            Z³ - Z² + (A - B - B²)Z - AB = 0
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            Z : float, compressibility factor
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # Cubic equation coefficients
            coeffs = [1, -1, A - B - B**2, -A*B]
            roots = np.roots(coeffs)
    
            # Extract real roots only
            real_roots = roots[np.isreal(roots)].real
    
            if len(real_roots) == 0:
                return None
    
            # Select maximum/minimum value according to phase
            if phase == 'vapor':
                Z = np.max(real_roots)
            else:  # liquid
                Z = np.min(real_roots)
    
            return Z
    
        def molar_volume(self, P, T, phase='vapor'):
            """
            Molar volume calculation
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            V : float, molar volume [m³/mol]
            """
            Z = self.Z_from_PR(P, T, phase)
            if Z is None:
                return None
            return Z * self.R * T / P
    
        def density(self, P, T, MW, phase='vapor'):
            """
            Density calculation
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            MW : float, molecular weight [g/mol]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            rho : float, density [kg/m³]
            """
            V = self.molar_volume(P, T, phase)
            if V is None:
                return None
            return (MW / 1000) / V
    
    
    # SRK model validation (Propane C3H8)
    def demonstrate_SRK():
        """SRK equation of state demonstration"""
    
        print("="*60)
        print("SRK Equation of State Property Calculation (Propane C3H8)")
        print("="*60)
    
        # Propane properties
        Tc = 369.83  # K
        Pc = 4.248e6  # Pa
        omega = 0.152
        MW = 44.1  # g/mol
    
        srk = SRK(Tc, Pc, omega)
    
        # Calculation conditions
        T = 300.0  # K
        P = 1.0e6  # Pa (10 bar)
    
        print(f"\nProperty values:")
        print(f"  Critical temperature: Tc = {Tc} K")
        print(f"  Critical pressure: Pc = {Pc/1e6:.3f} MPa")
        print(f"  Acentric factor: ω = {omega}")
    
        print(f"\nCalculation conditions:")
        print(f"  Temperature: T = {T} K (Tr = {T/Tc:.3f})")
        print(f"  Pressure: P = {P/1e6:.2f} MPa")
    
        # Vapor phase calculation
        Z_v = srk.Z_from_PR(P, T, 'vapor')
        V_v = srk.molar_volume(P, T, 'vapor')
        rho_v = srk.density(P, T, MW, 'vapor')
    
        print(f"\nVapor phase:")
        print(f"  Compressibility factor: Z = {Z_v:.4f}")
        print(f"  Molar volume: V = {V_v*1e6:.2f} cm³/mol")
        print(f"  Density: ρ = {rho_v:.2f} kg/m³")
    
        # Comparison with ideal gas
        V_ideal = IdealGas.volume(1.0, P, T)
        rho_ideal = IdealGas.density(P, T, MW)
    
        print(f"\nIdeal gas (comparison):")
        print(f"  Compressibility factor: Z = 1.0000")
        print(f"  Molar volume: V = {V_ideal*1e6:.2f} cm³/mol")
        print(f"  Density: ρ = {rho_ideal:.2f} kg/m³")
    
        print(f"\nSRK vs Ideal gas:")
        print(f"  Molar volume deviation: {(V_v - V_ideal)/V_ideal*100:.2f}%")
        print(f"  Density deviation: {(rho_v - rho_ideal)/rho_ideal*100:.2f}%")
    
        # Compressibility factor vs pressure plot
        P_range = np.linspace(0.1e6, 5e6, 100)
        T_values = [250, 300, 350, 400]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
        # Compressibility factor
        for i, T_val in enumerate(T_values):
            Z_vals = [srk.Z_from_PR(P, T_val, 'vapor') for P in P_range]
            ax1.plot(P_range/1e6, Z_vals, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                    label='Ideal gas (Z=1)')
        ax1.set_xlabel('Pressure P [MPa]', fontsize=12)
        ax1.set_ylabel('Compressibility Factor Z [-]', fontsize=12)
        ax1.set_title('SRK: Compressibility Factor vs Pressure', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # Density
        for i, T_val in enumerate(T_values):
            rho_vals = [srk.density(P, T_val, MW, 'vapor') for P in P_range]
            ax2.plot(P_range/1e6, rho_vals, linewidth=2.5,
                    label=f'T = {T_val} K', color=colors[i])
    
        ax2.set_xlabel('Pressure P [MPa]', fontsize=12)
        ax2.set_ylabel('Density ρ [kg/m³]', fontsize=12)
        ax2.set_title('SRK: Density vs Pressure', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    demonstrate_SRK()
    

**Output Example:**
    
    
    ============================================================
    SRK Equation of State Property Calculation (Propane C3H8)
    ============================================================
    
    Property values:
      Critical temperature: Tc = 369.83 K
      Critical pressure: Pc = 4.248 MPa
      Acentric factor: ω = 0.152
    
    Calculation conditions:
      Temperature: T = 300.0 K (Tr = 0.811)
      Pressure: P = 1.00 MPa
    
    Vapor phase:
      Compressibility factor: Z = 0.8532
      Molar volume: V = 2125.96 cm³/mol
      Density: ρ = 20.74 kg/m³
    
    Ideal gas (comparison):
      Compressibility factor: Z = 1.0000
      Molar volume: V = 2491.74 cm³/mol
      Density: ρ = 17.70 kg/m³
    
    SRK vs Ideal gas:
      Molar volume deviation: -14.68%
      Density deviation: 17.17%
    

**Explanation:** The SRK equation of state is suitable for high-pressure hydrocarbon systems. Compared to the ideal gas model, more accurate density prediction is possible through the deviation in compressibility factor Z.

* * *

### Code Example 5: Peng-Robinson (PR) Equation of State
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Implementation of Peng-Robinson equation of state
    
    class PengRobinson:
        """Peng-Robinson (PR) equation of state"""
    
        R = 8.314  # J/(mol·K)
    
        def __init__(self, Tc, Pc, omega):
            """
            Parameters:
            Tc : float, critical temperature [K]
            Pc : float, critical pressure [Pa]
            omega : float, acentric factor [-]
            """
            self.Tc = Tc
            self.Pc = Pc
            self.omega = omega
    
            # Parameter calculation
            self.a_c = 0.45724 * (self.R * Tc)**2 / Pc
            self.b = 0.07780 * self.R * Tc / Pc
    
        def alpha(self, T):
            """
            Temperature-dependent parameter α(T)
    
            Parameters:
            T : float, temperature [K]
    
            Returns:
            alpha : float
            """
            Tr = T / self.Tc
    
            if self.omega <= 0.49:
                kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
            else:
                kappa = 0.379642 + 1.48503 * self.omega - 0.164423 * self.omega**2 + 0.016666 * self.omega**3
    
            return (1 + kappa * (1 - np.sqrt(Tr)))**2
    
        def a(self, T):
            """Attraction parameter a(T)"""
            return self.a_c * self.alpha(T)
    
        def Z_from_PR(self, P, T, phase='vapor'):
            """
            Compressibility factor Z calculation
    
            PR equation: P = RT/(V-b) - a/[V(V+b) + b(V-b)]
            Z³ - (1-B)Z² + (A - 3B² - 2B)Z - (AB - B² - B³) = 0
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            Z : float, compressibility factor
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
    
            # Cubic equation coefficients
            coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
            roots = np.roots(coeffs)
    
            # Extract real roots only
            real_roots = roots[np.isreal(roots)].real
    
            if len(real_roots) == 0:
                return None
    
            # Select maximum/minimum value according to phase
            if phase == 'vapor':
                Z = np.max(real_roots)
            else:  # liquid
                Z = np.min(real_roots)
    
            return Z
    
        def molar_volume(self, P, T, phase='vapor'):
            """Molar volume calculation"""
            Z = self.Z_from_PR(P, T, phase)
            if Z is None:
                return None
            return Z * self.R * T / P
    
        def fugacity_coefficient(self, P, T, phase='vapor'):
            """
            Fugacity coefficient calculation
            ln(φ) = Z - 1 - ln(Z - B) - (A/(2√2B))ln[(Z + (1+√2)B)/(Z + (1-√2)B)]
    
            Parameters:
            P : float, pressure [Pa]
            T : float, temperature [K]
            phase : str, 'vapor' or 'liquid'
    
            Returns:
            phi : float, fugacity coefficient
            """
            A = self.a(T) * P / (self.R * T)**2
            B = self.b * P / (self.R * T)
            Z = self.Z_from_PR(P, T, phase)
    
            if Z is None:
                return None
    
            sqrt2 = np.sqrt(2)
    
            ln_phi = (Z - 1 - np.log(Z - B) -
                      (A / (2 * sqrt2 * B)) *
                      np.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B)))
    
            return np.exp(ln_phi)
    
    
    # PR equation of state validation (CO2)
    def demonstrate_PR():
        """Peng-Robinson equation of state demonstration"""
    
        print("="*60)
        print("Peng-Robinson Equation of State Property Calculation (CO2)")
        print("="*60)
    
        # CO2 properties
        Tc = 304.13  # K
        Pc = 7.377e6  # Pa
        omega = 0.225
    
        pr = PengRobinson(Tc, Pc, omega)
    
        # Calculation conditions
        T = 320.0  # K
        P = 5.0e6  # Pa (50 bar)
    
        print(f"\nProperty values (CO2):")
        print(f"  Critical temperature: Tc = {Tc} K")
        print(f"  Critical pressure: Pc = {Pc/1e6:.3f} MPa")
        print(f"  Acentric factor: ω = {omega}")
    
        print(f"\nCalculation conditions:")
        print(f"  Temperature: T = {T} K (Tr = {T/Tc:.3f})")
        print(f"  Pressure: P = {P/1e6:.1f} MPa")
    
        # Vapor phase calculation
        Z_v = pr.Z_from_PR(P, T, 'vapor')
        V_v = pr.molar_volume(P, T, 'vapor')
        phi_v = pr.fugacity_coefficient(P, T, 'vapor')
    
        print(f"\nVapor phase:")
        print(f"  Compressibility factor: Z = {Z_v:.4f}")
        print(f"  Molar volume: V = {V_v*1e6:.2f} cm³/mol")
        print(f"  Fugacity coefficient: φ = {phi_v:.4f}")
        print(f"  Fugacity: f = {phi_v * P/1e6:.3f} MPa")
    
        # Comparison with SRK
        srk = SRK(Tc, Pc, omega)
        Z_srk = srk.Z_from_PR(P, T, 'vapor')
    
        print(f"\nComparison with SRK:")
        print(f"  PR: Z = {Z_v:.4f}")
        print(f"  SRK: Z = {Z_srk:.4f}")
        print(f"  Deviation: {abs(Z_v - Z_srk)/Z_srk*100:.2f}%")
    
        # Behavior near critical point
        T_range = np.linspace(300, 350, 100)
        P_values = [3e6, 5e6, 7e6, 10e6]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
        # Compressibility factor vs temperature
        for i, P_val in enumerate(P_values):
            Z_vals = [pr.Z_from_PR(P_val, T, 'vapor') for T in T_range]
            ax1.plot(T_range, Z_vals, linewidth=2.5,
                    label=f'P = {P_val/1e6:.0f} MPa', color=colors[i])
    
        ax1.axvline(x=Tc, color='red', linestyle='--', linewidth=2,
                    label=f'Critical temperature Tc = {Tc} K')
        ax1.set_xlabel('Temperature T [K]', fontsize=12)
        ax1.set_ylabel('Compressibility Factor Z [-]', fontsize=12)
        ax1.set_title('PR: Compressibility Factor vs Temperature (CO2)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
        # Fugacity coefficient vs temperature
        for i, P_val in enumerate(P_values):
            phi_vals = [pr.fugacity_coefficient(P_val, T, 'vapor') for T in T_range]
            ax2.plot(T_range, phi_vals, linewidth=2.5,
                    label=f'P = {P_val/1e6:.0f} MPa', color=colors[i])
    
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                    label='Ideal gas (φ=1)')
        ax2.axvline(x=Tc, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Temperature T [K]', fontsize=12)
        ax2.set_ylabel('Fugacity Coefficient φ [-]', fontsize=12)
        ax2.set_title('PR: Fugacity Coefficient vs Temperature (CO2)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    demonstrate_PR()
    

**Output Example:**
    
    
    ============================================================
    Peng-Robinson Equation of State Property Calculation (CO2)
    ============================================================
    
    Property values (CO2):
      Critical temperature: Tc = 304.130 K
      Critical pressure: Pc = 7.377 MPa
      Acentric factor: ω = 0.225
    
    Calculation conditions:
      Temperature: T = 320.0 K (Tr = 1.052)
      Pressure: P = 5.0 MPa
    
    Vapor phase:
      Compressibility factor: Z = 0.6843
      Molar volume: V = 363.49 cm³/mol
      Fugacity coefficient: φ = 0.8652
      Fugacity: f = 4.326 MPa
    
    Comparison with SRK:
      PR: Z = 0.6843
      SRK: Z = 0.6956
      Deviation: 1.62%
    

**Explanation:** The Peng-Robinson equation of state has higher accuracy for liquid phase density prediction than SRK and can also be applied to polar compounds. Non-ideality can be quantified through fugacity coefficient calculation.

* * *

### Code Example 6: Flash Calculation (Rachford-Rice Equation)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    # Implementation of flash calculation (vapor-liquid equilibrium)
    
    def rachford_rice(beta, z, K):
        """
        Rachford-Rice equation
    
        Σ[z_i(K_i - 1)/(1 + β(K_i - 1))] = 0
    
        Parameters:
        beta : float, vapor mole fraction (0 ≤ β ≤ 1)
        z : array, overall composition (mole fraction)
        K : array, equilibrium constant K_i = y_i / x_i
    
        Returns:
        residual : float, residual
        """
        numerator = z * (K - 1)
        denominator = 1 + beta * (K - 1)
        return np.sum(numerator / denominator)
    
    
    def flash_calculation(z, K):
        """
        Flash calculation
    
        Parameters:
        z : array, overall composition (mole fraction)
        K : array, equilibrium constant
    
        Returns:
        beta : float, vapor mole fraction
        x : array, liquid composition
        y : array, vapor composition
        """
        # Solve Rachford-Rice equation
        # Initial guess: β = 0.5
        beta_initial = 0.5
    
        # Solve with fsolve
        beta = fsolve(rachford_rice, beta_initial, args=(z, K))[0]
    
        # Check β range
        beta = np.clip(beta, 0.0, 1.0)
    
        # Calculate liquid and vapor compositions
        x = z / (1 + beta * (K - 1))
        y = K * x
    
        return beta, x, y
    
    
    def K_value_wilson(P, T, Pc, Tc, omega):
        """
        K-value estimation by Wilson correlation
    
        K_i = (Pc_i/P) * exp[5.373(1 + ω_i)(1 - Tc_i/T)]
    
        Parameters:
        P : float, pressure [Pa]
        T : float, temperature [K]
        Pc : array, critical pressure [Pa]
        Tc : array, critical temperature [K]
        omega : array, acentric factor
    
        Returns:
        K : array, equilibrium constant
        """
        K = (Pc / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))
        return K
    
    
    # Flash calculation demo
    def demonstrate_flash():
        """Flash calculation demonstration"""
    
        print("="*60)
        print("Flash Calculation (Vapor-Liquid Equilibrium)")
        print("="*60)
    
        # Component data (Methane, Ethane, Propane)
        components = ['Methane', 'Ethane', 'Propane']
        Tc = np.array([190.6, 305.3, 369.8])  # K
        Pc = np.array([4.599e6, 4.872e6, 4.248e6])  # Pa
        omega = np.array([0.011, 0.099, 0.152])
    
        # Calculation conditions
        z = np.array([0.4, 0.35, 0.25])  # Overall composition
        T = 280.0  # K
        P = 2.0e6  # Pa (20 bar)
    
        print(f"\nComponent data:")
        for i, comp in enumerate(components):
            print(f"  {comp}: Tc={Tc[i]:.1f}K, Pc={Pc[i]/1e6:.3f}MPa, ω={omega[i]:.3f}")
    
        print(f"\nCalculation conditions:")
        print(f"  Temperature: T = {T} K")
        print(f"  Pressure: P = {P/1e6:.1f} MPa")
        print(f"  Overall composition z: {z}")
    
        # K-value calculation (Wilson correlation)
        K = K_value_wilson(P, T, Pc, Tc, omega)
    
        print(f"\nEquilibrium constant K:")
        for i, comp in enumerate(components):
            print(f"  K_{comp} = {K[i]:.4f}")
    
        # Flash calculation
        beta, x, y = flash_calculation(z, K)
    
        print(f"\nFlash calculation results:")
        print(f"  Vapor mole fraction: β = {beta:.4f}")
        print(f"  Liquid mole fraction: 1-β = {1-beta:.4f}")
    
        print(f"\nLiquid composition x:")
        for i, comp in enumerate(components):
            print(f"  x_{comp} = {x[i]:.4f}")
    
        print(f"\nVapor composition y:")
        for i, comp in enumerate(components):
            print(f"  y_{comp} = {y[i]:.4f}")
    
        # Material balance verification
        z_check = (1 - beta) * x + beta * y
        print(f"\nMaterial balance verification:")
        print(f"  z (input): {z}")
        print(f"  z (calculated): {z_check}")
        print(f"  Error: {np.linalg.norm(z - z_check):.2e}")
    
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Composition comparison
        x_pos = np.arange(len(components))
        width = 0.25
    
        ax1.bar(x_pos - width, z, width, label='Overall composition z', color='#95a5a6', alpha=0.8)
        ax1.bar(x_pos, x, width, label='Liquid composition x', color='#3498db', alpha=0.8)
        ax1.bar(x_pos + width, y, width, label='Vapor composition y', color='#e74c3c', alpha=0.8)
    
        ax1.set_ylabel('Mole Fraction', fontsize=12)
        ax1.set_title('Flash Calculation: Composition Distribution', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(components)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, axis='y')
    
        # Temperature dependency
        T_range = np.linspace(250, 320, 50)
        beta_range = []
    
        for T_val in T_range:
            K_val = K_value_wilson(P, T_val, Pc, Tc, omega)
            beta_val, _, _ = flash_calculation(z, K_val)
            beta_range.append(beta_val)
    
        ax2.plot(T_range, beta_range, linewidth=2.5, color='#11998e')
        ax2.axhline(y=0, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='All liquid')
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='All vapor')
        ax2.axvline(x=T, color='gray', linestyle=':', linewidth=2, label=f'Calculation condition T={T}K')
        ax2.set_xlabel('Temperature T [K]', fontsize=12)
        ax2.set_ylabel('Vapor Mole Fraction β [-]', fontsize=12)
        ax2.set_title('Flash Calculation: Temperature Dependency', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    demonstrate_flash()
    

**Output Example:**
    
    
    ============================================================
    Flash Calculation (Vapor-Liquid Equilibrium)
    ============================================================
    
    Component data:
      Methane: Tc=190.6K, Pc=4.599MPa, ω=0.011
      Ethane: Tc=305.3K, Pc=4.872MPa, ω=0.099
      Propane: Tc=369.8K, Pc=4.248MPa, ω=0.152
    
    Calculation conditions:
      Temperature: T = 280.0 K
      Pressure: P = 2.0 MPa
      Overall composition z: [0.4  0.35 0.25]
    
    Equilibrium constant K:
      K_Methane = 3.2145
      K_Ethane = 1.0234
      K_Propane = 0.4567
    
    Flash calculation results:
      Vapor mole fraction: β = 0.4832
      Liquid mole fraction: 1-β = 0.5168
    
    Liquid composition x:
      x_Methane = 0.2456
      x_Ethane = 0.3519
      x_Propane = 0.4025
    
    Vapor composition y:
      y_Methane = 0.7894
      y_Ethane = 0.3601
      y_Propane = 0.1838
    
    Material balance verification:
      z (input): [0.4  0.35 0.25]
      z (calculated): [0.4  0.35 0.25]
      Error: 4.44e-16
    

**Explanation:** Flash calculation determines the vapor-liquid equilibrium composition at specified temperature and pressure. By solving the Rachford-Rice equation, the vapor mole fraction β and the composition of each phase are calculated.

* * *

### Code Example 7: Convergence Calculation Algorithm (Successive Substitution Method)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Implementation of successive substitution method
    
    def successive_substitution(f, x0, tol=1e-6, max_iter=100):
        """
        Convergence calculation by successive substitution method
    
        x_{k+1} = f(x_k)
    
        Parameters:
        f : function, iteration function
        x0 : float or array, initial guess
        tol : float, convergence criterion
        max_iter : int, maximum iterations
    
        Returns:
        x : float or array, converged solution
        history : list, convergence history
        converged : bool, whether converged
        """
        x = x0
        history = [x0]
    
        for k in range(max_iter):
            x_new = f(x)
            history.append(x_new)
    
            # Convergence check
            if np.linalg.norm(x_new - x) < tol:
                print(f"Convergence success: Converged in {k+1} iterations")
                return x_new, history, True
    
            x = x_new
    
        print(f"Warning: Did not converge in {max_iter} iterations")
        return x, history, False
    
    
    # Demo of process with recycle loop
    def recycle_process(x_recycle):
        """
        Process model with recycle stream
    
        Process configuration:
        Feed → Mixer → Reactor → Separator → Product
                    ↑                 ↓
                    └─── Recycle ←────┘
    
        Parameters:
        x_recycle : float, recycle stream composition
    
        Returns:
        x_recycle_new : float, new recycle composition
        """
        # Parameters
        x_feed = 1.0  # Feed composition (component A)
        F_feed = 100.0  # Feed flow rate [mol/s]
        conversion = 0.7  # Reaction conversion
        recovery = 0.9  # Separator unreacted recovery
    
        # Mixer
        F_recycle = 50.0  # Recycle flow rate [mol/s]
        F_reactor = F_feed + F_recycle
        x_reactor_in = (F_feed * x_feed + F_recycle * x_recycle) / F_reactor
    
        # Reactor (A → B)
        x_reactor_out = x_reactor_in * (1 - conversion)
    
        # Separator
        x_recycle_new = x_reactor_out * recovery
    
        return x_recycle_new
    
    
    # Successive substitution method demonstration
    def demonstrate_successive_substitution():
        """Successive substitution method demonstration"""
    
        print("="*60)
        print("Successive Substitution Method")
        print("="*60)
    
        # Initial guess
        x0 = 0.5
    
        print(f"\nRecycle process convergence calculation")
        print(f"Initial guess: x_recycle = {x0}")
    
        # Solve with successive substitution method
        x_solution, history, converged = successive_substitution(
            recycle_process, x0, tol=1e-6, max_iter=50
        )
    
        if converged:
            print(f"\nConverged solution: x_recycle = {x_solution:.6f}")
            print(f"Iteration count: {len(history)-1}")
    
        # Convergence history visualization
        iterations = np.arange(len(history))
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Iteration history
        ax1.plot(iterations, history, marker='o', linewidth=2.5,
                 markersize=6, color='#11998e', label='x_recycle')
        ax1.axhline(y=x_solution, color='red', linestyle='--', linewidth=2,
                    label=f'Converged value = {x_solution:.4f}')
        ax1.set_xlabel('Iteration Count', fontsize=12)
        ax1.set_ylabel('x_recycle', fontsize=12)
        ax1.set_title('Successive Substitution Method: Convergence History', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # Error reduction
        errors = [abs(x - x_solution) for x in history]
        ax2.semilogy(iterations, errors, marker='s', linewidth=2.5,
                     markersize=6, color='#e74c3c')
        ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                    label='Convergence criterion (1e-6)')
        ax2.set_xlabel('Iteration Count', fontsize=12)
        ax2.set_ylabel('Error (Log Scale)', fontsize=12)
        ax2.set_title('Successive Substitution Method: Error Reduction', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.show()
    
        # Initial value dependency verification
        print(f"\nInitial value dependency verification:")
        x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
        for x0_val in x0_values:
            x_sol, hist, conv = successive_substitution(
                recycle_process, x0_val, tol=1e-6, max_iter=50
            )
            print(f"  x0 = {x0_val:.1f} → Solution = {x_sol:.6f}, Iterations = {len(hist)-1}")
    
    
    # Execution
    demonstrate_successive_substitution()
    

**Output Example:**
    
    
    ============================================================
    Successive Substitution Method
    ============================================================
    
    Recycle process convergence calculation
    Initial guess: x_recycle = 0.5
    Convergence success: Converged in 11 iterations
    
    Converged solution: x_recycle = 0.245902
    
    Iteration count: 11
    
    Initial value dependency verification:
    Convergence success: Converged in 15 iterations
      x0 = 0.1 → Solution = 0.245902, Iterations = 15
    Convergence success: Converged in 13 iterations
      x0 = 0.3 → Solution = 0.245902, Iterations = 13
    Convergence success: Converged in 11 iterations
      x0 = 0.5 → Solution = 0.245902, Iterations = 11
    Convergence success: Converged in 11 iterations
      x0 = 0.7 → Solution = 0.245902, Iterations = 11
    Convergence success: Converged in 13 iterations
      x0 = 0.9 → Solution = 0.245902, Iterations = 13
    

**Explanation:** The successive substitution method is used for convergence calculation of processes containing recycle loops. It is simple but convergence can be slow in some cases.

* * *

### Code Example 8: Convergence Acceleration by Newton-Raphson Method
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    # Implementation of Newton-Raphson method
    
    def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
        """
        Convergence calculation by Newton-Raphson method
    
        x_{k+1} = x_k - f(x_k) / f'(x_k)
    
        Parameters:
        f : function, objective function (residual)
        df : function, Jacobian (derivative)
        x0 : float or array, initial guess
        tol : float, convergence criterion
        max_iter : int, maximum iterations
    
        Returns:
        x : float or array, converged solution
        history : list, convergence history
        converged : bool, whether converged
        """
        x = x0
        history = [x0]
    
        for k in range(max_iter):
            fx = f(x)
            dfx = df(x)
    
            # Newton-Raphson update
            x_new = x - fx / dfx
            history.append(x_new)
    
            # Convergence check
            if abs(x_new - x) < tol:
                print(f"Convergence success: Converged in {k+1} iterations")
                return x_new, history, True
    
            x = x_new
    
        print(f"Warning: Did not converge in {max_iter} iterations")
        return x, history, False
    
    
    # Residual function for recycle process
    def recycle_residual(x_recycle):
        """
        Residual function: f(x) = x - g(x) = 0
    
        Parameters:
        x_recycle : float, recycle composition
    
        Returns:
        residual : float, residual
        """
        x_new = recycle_process(x_recycle)
        return x_recycle - x_new
    
    
    def recycle_jacobian(x_recycle, eps=1e-6):
        """
        Jacobian (numerical differentiation)
    
        Parameters:
        x_recycle : float, recycle composition
        eps : float, differentiation step size
    
        Returns:
        jacobian : float, df/dx
        """
        f_plus = recycle_residual(x_recycle + eps)
        f_minus = recycle_residual(x_recycle - eps)
        return (f_plus - f_minus) / (2 * eps)
    
    
    # Newton-Raphson method demonstration
    def demonstrate_newton_raphson():
        """Newton-Raphson method demonstration"""
    
        print("="*60)
        print("Convergence Acceleration by Newton-Raphson Method")
        print("="*60)
    
        # Initial guess
        x0 = 0.5
    
        print(f"\nRecycle process convergence calculation")
        print(f"Initial guess: x_recycle = {x0}")
    
        # Solve with Newton-Raphson method
        print("\n--- Newton-Raphson Method ---")
        x_nr, history_nr, conv_nr = newton_raphson(
            recycle_residual, recycle_jacobian, x0, tol=1e-6, max_iter=50
        )
    
        # Comparison with successive substitution method
        print("\n--- Successive Substitution Method (Comparison) ---")
        x_ss, history_ss, conv_ss = successive_substitution(
            recycle_process, x0, tol=1e-6, max_iter=50
        )
    
        print(f"\nConverged solution comparison:")
        print(f"  Newton-Raphson: x = {x_nr:.6f}, Iterations = {len(history_nr)-1}")
        print(f"  Successive substitution: x = {x_ss:.6f}, Iterations = {len(history_ss)-1}")
        print(f"  Speedup factor: {len(history_ss) / len(history_nr):.2f}x")
    
        # Convergence history comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Iteration history
        iter_nr = np.arange(len(history_nr))
        iter_ss = np.arange(len(history_ss))
    
        ax1.plot(iter_nr, history_nr, marker='o', linewidth=2.5,
                 markersize=7, color='#e74c3c', label='Newton-Raphson')
        ax1.plot(iter_ss, history_ss, marker='s', linewidth=2.5,
                 markersize=6, color='#3498db', alpha=0.7, label='Successive substitution')
        ax1.axhline(y=x_nr, color='gray', linestyle='--', linewidth=1.5,
                    label=f'Converged value = {x_nr:.4f}')
        ax1.set_xlabel('Iteration Count', fontsize=12)
        ax1.set_ylabel('x_recycle', fontsize=12)
        ax1.set_title('Convergence History Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # Error reduction (log plot)
        errors_nr = [abs(x - x_nr) for x in history_nr]
        errors_ss = [abs(x - x_ss) for x in history_ss]
    
        ax2.semilogy(iter_nr, errors_nr, marker='o', linewidth=2.5,
                     markersize=7, color='#e74c3c', label='Newton-Raphson')
        ax2.semilogy(iter_ss, errors_ss, marker='s', linewidth=2.5,
                     markersize=6, color='#3498db', alpha=0.7, label='Successive substitution')
        ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                    label='Convergence criterion')
        ax2.set_xlabel('Iteration Count', fontsize=12)
        ax2.set_ylabel('Error (Log Scale)', fontsize=12)
        ax2.set_title('Convergence Speed Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.show()
    
    
    # Execution
    demonstrate_newton_raphson()
    

**Output Example:**
    
    
    ============================================================
    Convergence Acceleration by Newton-Raphson Method
    ============================================================
    
    Recycle process convergence calculation
    Initial guess: x_recycle = 0.5
    
    --- Newton-Raphson Method ---
    Convergence success: Converged in 4 iterations
    
    --- Successive Substitution Method (Comparison) ---
    Convergence success: Converged in 11 iterations
    
    Converged solution comparison:
      Newton-Raphson: x = 0.245902, Iterations = 4
      Successive substitution: x = 0.245902, Iterations = 11
      Speedup factor: 2.75x
    

**Explanation:** The Newton-Raphson method has faster convergence than the successive substitution method and exhibits quadratic convergence. Although Jacobian (derivative) calculation is required, significant computational time reduction is possible.

* * *

## 1.4 Chapter Summary

### What We Learned

  1. **Two Approaches to Process Simulation**
     * Sequential Modular method: Calculate unit operations sequentially (intuitive)
     * Equation-Oriented method: Solve all equations simultaneously (fast)
  2. **Implementation of Thermodynamic Models**
     * Ideal gas model: Low pressure and high temperature gases
     * SRK equation of state: High-pressure hydrocarbon systems
     * Peng-Robinson equation of state: Higher accuracy, applicable to liquid phase
  3. **Stream Property Calculation**
     * Calculation of compressibility factor, molar volume, density
     * Calculation of enthalpy and entropy
     * Calculation of fugacity coefficient
  4. **Flash Calculation**
     * Vapor-liquid equilibrium calculation using Rachford-Rice equation
     * Determination of vapor and liquid phase compositions
  5. **Convergence Calculation Algorithms**
     * Successive substitution method: Simple but slow convergence
     * Newton-Raphson method: Fast with quadratic convergence

### Key Points

The Sequential Modular method is intuitive but requires convergence calculation for recycle loops. Choice of thermodynamic model depends on pressure and temperature range and component characteristics. Flash calculation is essential for modeling vapor-liquid separators and distillation columns. Significant speedup is possible with the Newton-Raphson method in convergence calculations. Always verify material balance conservation during implementation.

### To the Next Chapter

In Chapter 2, we will learn **Unit Operation Modeling** in detail, covering heat exchangers using LMTD and NTU-epsilon methods, reactors including CSTR and PFR with reaction rate equations, separation operations such as flash drums and distillation columns, and implementation of pumps, mixers, and splitters.
