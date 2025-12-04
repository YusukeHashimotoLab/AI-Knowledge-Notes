---
title: "Chapter 5: Case Study - Optimal Operating Conditions for Chemical Processes"
chapter_title: "Chapter 5: Case Study - Optimal Operating Conditions for Chemical Processes"
subtitle: "Complete Industrial Optimization Workflow: Economic Objective Functions, Process Modeling, Robust Optimization, Real-Time Optimization"
---

This chapter covers Case Study. You will learn essential concepts and techniques.

## Introduction

In this final chapter, we integrate all the techniques learned so far to tackle real-world optimal operating condition search for chemical processes. We will cover complete optimization workflows, economic objective function design, sensitivity analysis, robust optimization, and real-time optimization frameworks—all directly applicable to industrial practice.

#### What You'll Learn in This Chapter

  * **Complete Optimization Workflow** : From problem definition to result validation
  * **Complete CSTR Optimization Implementation** : Multivariable optimization and economic objective functions
  * **Sensitivity Analysis** : Robustness evaluation of optimal solutions against parameter variations
  * **Robust Optimization** : Optimization under uncertainty
  * **Real-Time Optimization** : Adaptive optimization based on online data
  * **Distillation Column Optimization** : Comprehensive case study

## 5.1 Complete Optimization Workflow

Industrial process optimization follows a systematic workflow:
    
    
    flowchart TD
        A[1. Problem Definition and Goal Setting] --> B[2. Process Model Development]
        B --> C[3. Constraint Definition]
        C --> D[4. Optimization Problem Formulation]
        D --> E[5. Algorithm Selection and Execution]
        E --> F[6. Result Validation and Interpretation]
        F --> G{Goals Achieved?}
        G -->|No| H[7. Model Improvement]
        H --> B
        G -->|Yes| I[8. Sensitivity Analysis]
        I --> J[9. Implementation and Monitoring]
    
        style A fill:#e8f5e9
        style D fill:#fff9c4
        style E fill:#ffe0b2
        style F fill:#f8bbd0
        style I fill:#c5cae9
        style J fill:#b2dfdb
    

### Detailed Steps

**Step 1: Problem Definition and Goal Setting**

  * Clarify optimization objectives (cost reduction, yield improvement, energy efficiency improvement, etc.)
  * Define KPIs (Key Performance Indicators)
  * Measure baseline performance
  * Set target values (quantitative and achievable goals)

**Step 2: Process Model Development**

  * First-principles models (material balance, energy balance, reaction rate equations)
  * Data-driven models (machine learning, statistical models)
  * Hybrid models (first-principles + data-driven)
  * Model validation and verification

**Step 3: Constraint Definition**

  * Safety constraints (temperature, pressure, flow rate upper and lower limits)
  * Product specification constraints (purity, quality indicators)
  * Environmental constraints (emission standards, energy consumption)
  * Operational constraints (equipment capacity, physical limitations)

**Steps 4-9** : The remaining steps will be explained in detail in subsequent sections.

## 5.2 Complete CSTR Optimization Implementation

Let's learn optimal operating condition search for a Continuous Stirred Tank Reactor (CSTR) through a complete implementation example.

### Problem Setting

**Reaction System** : Simple reversible reaction A → B (exothermic reaction)

**Objective** : Maximize profit (product revenue - raw material cost - energy cost)

**Decision Variables** :

  * **T** : Reaction temperature [°C] (50-300)
  * **τ** : Residence time [min] (30-180)
  * **C A0**: Feed concentration [mol/L] (1-5)

**Constraints** :

  * Safety constraint: T ≤ 350°C
  * Purity constraint: XA (conversion) ≥ 0.95
  * Flow rate constraint: 100 ≤ F ≤ 400 L/h

#### Code Example 1: Complete CSTR Problem Formulation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import pandas as pd
    
    class CSTROptimization:
        """
        Continuous Stirred Tank Reactor (CSTR) Optimization Class
    
        Reaction: A → B (exothermic reaction)
        Objective: Maximize profit (revenue - raw material cost - energy cost)
        """
    
        def __init__(self):
            # Economic parameters (realistic pricing)
            self.price_B = 1200.0      # Product B price [¥/kg]
            self.price_A = 600.0       # Raw material A price [¥/kg]
            self.energy_cost = 12.0    # Energy cost [¥/kWh]
    
            # Physical property parameters
            self.MW_A = 60.0           # Molecular weight A [g/mol]
            self.MW_B = 60.0           # Molecular weight B [g/mol]
            self.rho = 1000.0          # Density [kg/m³]
            self.Cp = 4.18             # Specific heat [kJ/kg·K]
    
            # Reaction rate parameters (Arrhenius equation)
            self.k0 = 1e10             # Frequency factor [1/min]
            self.Ea = 80000.0          # Activation energy [J/mol]
            self.R = 8.314             # Gas constant [J/mol·K]
            self.delta_H = -50000.0    # Heat of reaction [J/mol] (exothermic)
    
            # Constraint parameters
            self.T_max = 350.0         # Maximum allowable temperature [°C]
            self.X_min = 0.95          # Minimum conversion [-]
            self.F_min = 100.0         # Minimum flow rate [L/h]
            self.F_max = 400.0         # Maximum flow rate [L/h]
    
        def reaction_rate_constant(self, T_celsius):
            """
            Calculate reaction rate constant (Arrhenius equation)
    
            Parameters:
            -----------
            T_celsius : float
                Temperature [°C]
    
            Returns:
            --------
            k : float
                Reaction rate constant [1/min]
            """
            T_K = T_celsius + 273.15
            k = self.k0 * np.exp(-self.Ea / (self.R * T_K))
            return k
    
        def conversion(self, T, tau):
            """
            Calculate CSTR conversion
    
            Parameters:
            -----------
            T : float
                Temperature [°C]
            tau : float
                Residence time [min]
    
            Returns:
            --------
            X_A : float
                Conversion [-]
            """
            k = self.reaction_rate_constant(T)
            X_A = (k * tau) / (1 + k * tau)  # CSTR conversion equation
            return X_A
    
        def profit(self, x):
            """
            Profit function (to be maximized)
    
            Parameters:
            -----------
            x : array_like
                [T, tau, C_A0]
                T: Temperature [°C]
                tau: Residence time [min]
                C_A0: Feed concentration [mol/L]
    
            Returns:
            --------
            profit : float
                Profit [¥/h] (negative value returned for scipy.minimize)
            """
            T, tau, C_A0 = x
    
            # Calculate conversion
            X_A = self.conversion(T, tau)
    
            # Reactor volume and flow rate
            V = 1000.0  # Fixed volume [L] (1 m³)
            F = V / tau  # Volumetric flow rate [L/h]
    
            # Production rate [mol/h]
            production_rate_B = F * C_A0 * X_A
    
            # Revenue [¥/h]
            revenue = production_rate_B * (self.MW_B / 1000.0) * self.price_B
    
            # Raw material cost [¥/h]
            raw_material_cost = F * C_A0 * (self.MW_A / 1000.0) * self.price_A
    
            # Energy cost [¥/h] (cooling for reaction heat removal)
            Q_reaction = abs(self.delta_H) * production_rate_B  # [J/h]
            energy_cost = (Q_reaction / 3.6e6) * self.energy_cost  # [¥/h] (J→kWh conversion)
    
            # Profit = Revenue - Raw material cost - Energy cost
            profit = revenue - raw_material_cost - energy_cost
    
            # Return negative value for minimization
            return -profit
    
        def constraints(self, x):
            """
            Define constraints
    
            Returns:
            --------
            constraints : list of dict
                Constraints in scipy.optimize format
            """
            T, tau, C_A0 = x
            V = 1000.0
            F = V / tau
            X_A = self.conversion(T, tau)
    
            cons = [
                {'type': 'ineq', 'fun': lambda x: self.T_max - x[0]},           # T ≤ 350°C
                {'type': 'ineq', 'fun': lambda x: self.conversion(x[0], x[1]) - self.X_min},  # X_A ≥ 0.95
                {'type': 'ineq', 'fun': lambda x: V / x[1] - self.F_min},       # F ≥ 100 L/h
                {'type': 'ineq', 'fun': lambda x: self.F_max - V / x[1]}        # F ≤ 400 L/h
            ]
    
            return cons
    
    # Instantiation and testing
    cstr = CSTROptimization()
    
    # Example operating conditions: T=200°C, τ=60min, C_A0=3.0 mol/L
    test_conditions = [200.0, 60.0, 3.0]
    profit_value = -cstr.profit(test_conditions)
    conversion_value = cstr.conversion(test_conditions[0], test_conditions[1])
    
    print("=" * 60)
    print("CSTR Optimization Problem Formulation")
    print("=" * 60)
    print(f"\nOperating conditions: T={test_conditions[0]:.1f}°C, τ={test_conditions[1]:.1f}min, C_A0={test_conditions[2]:.1f}mol/L")
    print(f"Conversion: {conversion_value:.3f} ({conversion_value*100:.1f}%)")
    print(f"Profit: ¥{profit_value:,.0f}/h")
    print("\nObjective: Maximize profit")
    print("Constraints: T≤350°C, X_A≥0.95, 100≤F≤400 L/h")
    

#### Code Example 2: Process Model Development (Reaction Rate Equations and Material Balance)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    class CSTRProcessModel:
        """
        CSTR Process Model: Reaction rate equations and material balance
    
        Model:
        - Arrhenius rate equation: k(T) = k0 * exp(-Ea/RT)
        - CSTR material balance: X_A = (k*τ) / (1 + k*τ)
        - Energy balance: Q_reaction + Q_cooling = 0
        """
    
        def __init__(self):
            self.k0 = 1e10
            self.Ea = 80000.0
            self.R = 8.314
    
        def k(self, T_celsius):
            """Reaction rate constant [1/min]"""
            T_K = T_celsius + 273.15
            return self.k0 * np.exp(-self.Ea / (self.R * T_K))
    
        def conversion_vs_temperature(self, tau=60.0):
            """Plot temperature vs conversion"""
            T_range = np.linspace(50, 300, 100)
            X_A = np.array([self.k(T) * tau / (1 + self.k(T) * tau) for T in T_range])
    
            plt.figure(figsize=(10, 6))
            plt.plot(T_range, X_A * 100, 'b-', linewidth=2, label=f'τ = {tau} min')
            plt.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='Target conversion 95%')
            plt.axvline(x=350, color='orange', linestyle='--', linewidth=1.5, label='Max temperature 350°C')
    
            plt.xlabel('Temperature T [°C]', fontsize=12, fontweight='bold')
            plt.ylabel('Conversion X_A [%]', fontsize=12, fontweight='bold')
            plt.title('CSTR Conversion vs Reaction Temperature', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig('cstr_conversion_vs_temp.png', dpi=150, bbox_inches='tight')
            plt.show()
    
        def conversion_surface(self):
            """3D surface plot of temperature, residence time vs conversion"""
            T_range = np.linspace(50, 300, 50)
            tau_range = np.linspace(30, 180, 50)
            T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    
            X_A_grid = np.zeros_like(T_grid)
            for i in range(len(T_range)):
                for j in range(len(tau_range)):
                    k_val = self.k(T_grid[j, i])
                    X_A_grid[j, i] = (k_val * tau_grid[j, i]) / (1 + k_val * tau_grid[j, i])
    
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
    
            surf = ax.plot_surface(T_grid, tau_grid, X_A_grid * 100, cmap='viridis',
                                   alpha=0.9, edgecolor='none')
    
            # Contour line for 95% conversion
            contour = ax.contour(T_grid, tau_grid, X_A_grid * 100, levels=[95],
                                 colors='red', linewidths=3)
    
            ax.set_xlabel('Temperature T [°C]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residence Time τ [min]', fontsize=12, fontweight='bold')
            ax.set_zlabel('Conversion X_A [%]', fontsize=12, fontweight='bold')
            ax.set_title('CSTR Conversion 3D Surface Plot', fontsize=14, fontweight='bold')
    
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Conversion [%]')
            plt.tight_layout()
            plt.savefig('cstr_conversion_3d.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("=" * 60)
            print("CSTR Process Model Visualization")
            print("=" * 60)
            print("\nOperating conditions to achieve 95% conversion:")
            print("- High temperature + Short residence time")
            print("- Low temperature + Long residence time")
            print("\nOptimization selects the optimal point considering economics")
    
    # Execution
    model = CSTRProcessModel()
    model.conversion_vs_temperature(tau=60.0)
    model.conversion_surface()
    

#### Code Example 3: Multivariable Optimization Execution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from scipy.optimize import minimize
    import numpy as np
    
    class CSTRMultivariableOptimization:
        """
        CSTR Multivariable Optimization Execution and Result Interpretation
    
        Decision variables:
        - T: Temperature [°C]
        - τ: Residence time [min]
        - C_A0: Feed concentration [mol/L]
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
        def optimize(self):
            """Execute optimization"""
    
            # Initial guess
            x0 = np.array([200.0, 60.0, 3.0])  # [T, tau, C_A0]
    
            # Variable bounds
            bounds = [
                (50.0, 300.0),    # T [°C]
                (30.0, 180.0),    # tau [min]
                (1.0, 5.0)        # C_A0 [mol/L]
            ]
    
            # Constraints
            constraints = self.cstr.constraints(x0)
    
            print("=" * 60)
            print("CSTR Multivariable Optimization Execution")
            print("=" * 60)
            print(f"\nInitial guess:")
            print(f"  T   = {x0[0]:.1f} °C")
            print(f"  τ   = {x0[1]:.1f} min")
            print(f"  C_A0 = {x0[2]:.2f} mol/L")
            print(f"\nInitial profit: ¥{-self.cstr.profit(x0):,.0f}/h")
    
            # Optimize using SLSQP method
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': True, 'maxiter': 100}
            )
    
            # Interpret results
            self.interpret_results(result)
    
            return result
    
        def interpret_results(self, result):
            """Interpret and output optimization results"""
    
            print("\n" + "=" * 60)
            print("Optimization Results")
            print("=" * 60)
    
            if result.success:
                print("\n✓ Optimization successful!")
            else:
                print("\n✗ Optimization failed")
                print(f"Reason: {result.message}")
                return
    
            T_opt, tau_opt, C_A0_opt = result.x
            profit_opt = -result.fun
            X_A_opt = self.cstr.conversion(T_opt, tau_opt)
            V = 1000.0
            F_opt = V / tau_opt
    
            print(f"\n【Optimal Operating Conditions】")
            print(f"  Temperature T*        : {T_opt:.2f} °C")
            print(f"  Residence time τ*    : {tau_opt:.2f} min")
            print(f"  Feed concentration C_A0* : {C_A0_opt:.3f} mol/L")
            print(f"  Flow rate F*        : {F_opt:.2f} L/h")
    
            print(f"\n【Process Performance】")
            print(f"  Conversion X_A     : {X_A_opt:.4f} ({X_A_opt*100:.2f}%)")
            print(f"  Maximum profit       : ¥{profit_opt:,.0f}/h")
    
            # Annual profit conversion
            annual_profit = profit_opt * 24 * 365
            print(f"  Annual profit estimate   : ¥{annual_profit:,.0f}/year")
    
            # Check constraint margins
            print(f"\n【Constraint Margins】")
            print(f"  Temperature constraint   : T* = {T_opt:.1f}°C ≤ {self.cstr.T_max}°C (Margin: {self.cstr.T_max - T_opt:.1f}°C)")
            print(f"  Conversion constraint : X_A* = {X_A_opt:.3f} ≥ {self.cstr.X_min} (Margin: {X_A_opt - self.cstr.X_min:.4f})")
            print(f"  Flow rate constraint   : F* = {F_opt:.1f} L/h ∈ [{self.cstr.F_min}, {self.cstr.F_max}]")
    
            # Iteration count
            print(f"\n【Optimization Statistics】")
            print(f"  Iteration count       : {result.nit}")
            print(f"  Function evaluation count   : {result.nfev}")
    
    # Execution
    optimizer = CSTRMultivariableOptimization()
    result_optimal = optimizer.optimize()
    

#### Code Example 4: Comprehensive Constraint Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import minimize
    
    class CSTRComprehensiveConstraints:
        """
        CSTR Optimization with Comprehensive Constraints
    
        Constraints:
        1. Safety constraints: Temperature limit, pressure limit
        2. Purity constraints: Product purity, conversion
        3. Flow rate constraints: Minimum and maximum flow rates
        4. Energy constraints: Cooling capacity
        5. Environmental constraints: Emission standards
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
            # Additional constraint parameters
            self.Q_cooling_max = 500000.0  # Maximum cooling capacity [J/h]
            self.emission_limit = 10.0     # CO2 emission limit [kg/h]
            self.purity_min = 0.98         # Product purity lower limit [-]
    
        def comprehensive_constraints(self, x):
            """Comprehensive constraints"""
            T, tau, C_A0 = x
            V = 1000.0
            F = V / tau
            X_A = self.cstr.conversion(T, tau)
    
            # Production rate
            production_rate_B = F * C_A0 * X_A
    
            # Reaction heat
            Q_reaction = abs(self.cstr.delta_H) * production_rate_B
    
            # CO2 emissions (assumed proportional to energy use)
            CO2_emission = (Q_reaction / 3.6e6) * 0.5  # [kg/h] (assumption: 0.5 kg-CO2/kWh)
    
            # Product purity (simplified: proportional to conversion)
            purity = 0.90 + 0.10 * X_A  # Purity 100% at X_A=1.0
    
            constraints = [
                # Existing constraints
                {'type': 'ineq', 'fun': lambda x: self.cstr.T_max - x[0]},
                {'type': 'ineq', 'fun': lambda x: self.cstr.conversion(x[0], x[1]) - self.cstr.X_min},
                {'type': 'ineq', 'fun': lambda x: V / x[1] - self.cstr.F_min},
                {'type': 'ineq', 'fun': lambda x: self.cstr.F_max - V / x[1]},
    
                # Additional constraints
                {'type': 'ineq', 'fun': lambda x: self.Q_cooling_max - abs(self.cstr.delta_H) * V / x[1] * x[2] * self.cstr.conversion(x[0], x[1])},  # Cooling capacity
                {'type': 'ineq', 'fun': lambda x: self.emission_limit - (abs(self.cstr.delta_H) * V / x[1] * x[2] * self.cstr.conversion(x[0], x[1])) / 3.6e6 * 0.5},  # CO2 emission
                {'type': 'ineq', 'fun': lambda x: (0.90 + 0.10 * self.cstr.conversion(x[0], x[1])) - self.purity_min}  # Product purity
            ]
    
            return constraints
    
        def optimize_with_comprehensive_constraints(self):
            """Optimization under comprehensive constraints"""
    
            x0 = np.array([200.0, 60.0, 3.0])
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
    
            constraints = self.comprehensive_constraints(x0)
    
            print("=" * 60)
            print("CSTR Optimization Under Comprehensive Constraints")
            print("=" * 60)
            print("\nConstraints:")
            print("  1. Safety constraint: T ≤ 350°C")
            print("  2. Conversion constraint: X_A ≥ 0.95")
            print("  3. Flow rate constraint: 100 ≤ F ≤ 400 L/h")
            print("  4. Cooling capacity constraint: Q_reaction ≤ 500,000 J/h")
            print("  5. Environmental constraint: CO2 emission ≤ 10 kg/h")
            print("  6. Purity constraint: Product purity ≥ 0.98")
    
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200}
            )
    
            if result.success:
                T_opt, tau_opt, C_A0_opt = result.x
                profit_opt = -result.fun
                X_A_opt = self.cstr.conversion(T_opt, tau_opt)
                purity_opt = 0.90 + 0.10 * X_A_opt
                V = 1000.0
                F_opt = V / tau_opt
                Q_reaction = abs(self.cstr.delta_H) * F_opt * C_A0_opt * X_A_opt
                CO2 = (Q_reaction / 3.6e6) * 0.5
    
                print(f"\n✓ Optimization successful")
                print(f"\n【Optimal Operating Conditions】")
                print(f"  Temperature: {T_opt:.2f}°C, Residence time: {tau_opt:.2f}min, Feed concentration: {C_A0_opt:.3f}mol/L")
                print(f"\n【Performance Indicators】")
                print(f"  Profit: ¥{profit_opt:,.0f}/h")
                print(f"  Conversion: {X_A_opt*100:.2f}%")
                print(f"  Product purity: {purity_opt*100:.2f}%")
                print(f"  CO2 emission: {CO2:.2f} kg/h")
                print(f"  Cooling load: {Q_reaction:,.0f} J/h ({Q_reaction/self.Q_cooling_max*100:.1f}% of max)")
            else:
                print(f"\n✗ Optimization failed: {result.message}")
    
            return result
    
    # Execution
    comprehensive_opt = CSTRComprehensiveConstraints()
    result_comp = comprehensive_opt.optimize_with_comprehensive_constraints()
    

## 5.3 Sensitivity Analysis and Robust Optimization

In real processes, parameters (raw material prices, energy costs, etc.) vary. Sensitivity analysis evaluates how robust the optimal solution is to these variations.

#### Code Example 5: Optimization Execution and Result Interpretation

(This code is already implemented in Code Example 3)

#### Code Example 6: Sensitivity Analysis (Parameter Perturbation)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    class SensitivityAnalysis:
        """
        Parameter Sensitivity Analysis of Optimal Solutions
    
        Analysis targets:
        - Raw material price variation (±10%)
        - Energy cost variation (±20%)
        - Reaction rate parameter uncertainty (±15%)
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
            self.base_price_A = self.cstr.price_A
            self.base_energy_cost = self.cstr.energy_cost
            self.base_k0 = self.cstr.k0
    
        def sensitivity_raw_material_price(self, optimal_x):
            """Raw material price sensitivity analysis"""
    
            # Price variation range: ±10%
            price_variation = np.linspace(0.90, 1.10, 21)
            profits = []
    
            for factor in price_variation:
                self.cstr.price_A = self.base_price_A * factor
                profit = -self.cstr.profit(optimal_x)
                profits.append(profit)
    
            # Reset
            self.cstr.price_A = self.base_price_A
    
            return price_variation, profits
    
        def sensitivity_energy_cost(self, optimal_x):
            """Energy cost sensitivity analysis"""
    
            # Cost variation range: ±20%
            cost_variation = np.linspace(0.80, 1.20, 21)
            profits = []
    
            for factor in cost_variation:
                self.cstr.energy_cost = self.base_energy_cost * factor
                profit = -self.cstr.profit(optimal_x)
                profits.append(profit)
    
            # Reset
            self.cstr.energy_cost = self.base_energy_cost
    
            return cost_variation, profits
    
        def sensitivity_reaction_rate(self, optimal_x):
            """Reaction rate parameter sensitivity analysis"""
    
            # Parameter variation range: ±15%
            k0_variation = np.linspace(0.85, 1.15, 21)
            profits = []
            conversions = []
    
            for factor in k0_variation:
                self.cstr.k0 = self.base_k0 * factor
                profit = -self.cstr.profit(optimal_x)
                X_A = self.cstr.conversion(optimal_x[0], optimal_x[1])
                profits.append(profit)
                conversions.append(X_A)
    
            # Reset
            self.cstr.k0 = self.base_k0
    
            return k0_variation, profits, conversions
    
        def run_sensitivity_analysis(self):
            """Execute and visualize sensitivity analysis"""
    
            # First execute optimization to get optimal solution
            optimizer = CSTRMultivariableOptimization()
            result = optimizer.optimize()
    
            if not result.success:
                print("Cannot run sensitivity analysis due to optimization failure")
                return
    
            optimal_x = result.x
            base_profit = -result.fun
    
            print("\n" + "=" * 60)
            print("Sensitivity Analysis")
            print("=" * 60)
            print(f"\nBase optimal solution: T={optimal_x[0]:.1f}°C, τ={optimal_x[1]:.1f}min, C_A0={optimal_x[2]:.2f}mol/L")
            print(f"Base profit: ¥{base_profit:,.0f}/h")
    
            # Sensitivity analysis for each parameter
            price_var, price_profits = self.sensitivity_raw_material_price(optimal_x)
            energy_var, energy_profits = self.sensitivity_energy_cost(optimal_x)
            k0_var, k0_profits, k0_conversions = self.sensitivity_reaction_rate(optimal_x)
    
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # 1. Raw material price sensitivity
            ax1 = axes[0, 0]
            ax1.plot((price_var - 1) * 100, price_profits, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='Base profit')
            ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax1.set_xlabel('Raw material price variation [%]', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Profit [¥/h]', fontsize=11, fontweight='bold')
            ax1.set_title('Raw Material Price Sensitivity Analysis', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
            # 2. Energy cost sensitivity
            ax2 = axes[0, 1]
            ax2.plot((energy_var - 1) * 100, energy_profits, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='Base profit')
            ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax2.set_xlabel('Energy cost variation [%]', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Profit [¥/h]', fontsize=11, fontweight='bold')
            ax2.set_title('Energy Cost Sensitivity Analysis', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            # 3. Reaction rate parameter sensitivity (profit)
            ax3 = axes[1, 0]
            ax3.plot((k0_var - 1) * 100, k0_profits, 'm-', linewidth=2, marker='^', markersize=4)
            ax3.axhline(y=base_profit, color='r', linestyle='--', linewidth=1.5, label='Base profit')
            ax3.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax3.set_xlabel('Reaction rate constant k0 variation [%]', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Profit [¥/h]', fontsize=11, fontweight='bold')
            ax3.set_title('Reaction Rate Parameter Sensitivity (Profit)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
    
            # 4. Reaction rate parameter sensitivity (conversion)
            ax4 = axes[1, 1]
            ax4.plot((k0_var - 1) * 100, np.array(k0_conversions) * 100, 'c-', linewidth=2, marker='d', markersize=4)
            ax4.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='Minimum conversion 95%')
            ax4.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax4.set_xlabel('Reaction rate constant k0 variation [%]', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Conversion [%]', fontsize=11, fontweight='bold')
            ax4.set_title('Reaction Rate Parameter Sensitivity (Conversion)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
    
            plt.tight_layout()
            plt.savefig('cstr_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            # Calculate sensitivity coefficients
            print("\n【Sensitivity Coefficients】(Profit variation per 1% parameter variation)")
    
            # Raw material price sensitivity
            d_profit_price = (price_profits[11] - price_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  Raw material price: {d_profit_price:.3f} (1% increase → profit {abs(d_profit_price):.2f}% decrease)")
    
            # Energy cost sensitivity
            d_profit_energy = (energy_profits[11] - energy_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  Energy cost: {d_profit_energy:.3f} (1% increase → profit {abs(d_profit_energy):.2f}% decrease)")
    
            # Reaction rate sensitivity
            d_profit_k0 = (k0_profits[11] - k0_profits[9]) / (2 * 0.01 * base_profit)
            print(f"  Reaction rate constant: {d_profit_k0:.3f} (1% increase → profit {d_profit_k0:.2f}% increase)")
    
            print("\nConclusion:")
            print("  - Raw material price has the greatest impact")
            print("  - Energy cost impact is relatively small")
            print("  - Attention needed for reaction rate parameter uncertainty")
    
    # Execution
    sensitivity = SensitivityAnalysis()
    sensitivity.run_sensitivity_analysis()
    

#### Code Example 7: Robust Optimization (Optimization Under Uncertainty)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    class RobustOptimization:
        """
        Robust Optimization: Optimization under uncertainty
    
        Approaches:
        - Worst-case optimization
        - Expected value optimization (Monte Carlo method)
        - Probabilistic constraint handling
        """
    
        def __init__(self, n_samples=100):
            self.cstr = CSTROptimization()
            self.n_samples = n_samples
    
            # Uncertainty parameters (standard deviation)
            self.sigma_price_A = 0.05 * self.cstr.price_A      # Raw material price uncertainty ±5%
            self.sigma_energy = 0.10 * self.cstr.energy_cost   # Energy cost uncertainty ±10%
            self.sigma_k0 = 0.10 * self.cstr.k0                # Reaction rate uncertainty ±10%
    
        def expected_profit(self, x):
            """
            Expected profit (Monte Carlo method)
    
            Parameters:
            -----------
            x : array_like
                [T, tau, C_A0]
    
            Returns:
            --------
            expected_profit : float
                Negative expected profit (for minimization)
            """
            np.random.seed(42)  # For reproducibility
    
            profits = []
    
            for _ in range(self.n_samples):
                # Sample parameters (normal distribution)
                price_A_sample = np.random.normal(self.cstr.price_A, self.sigma_price_A)
                energy_cost_sample = np.random.normal(self.cstr.energy_cost, self.sigma_energy)
                k0_sample = np.random.normal(self.cstr.k0, self.sigma_k0)
    
                # Temporarily update parameters
                original_price_A = self.cstr.price_A
                original_energy_cost = self.cstr.energy_cost
                original_k0 = self.cstr.k0
    
                self.cstr.price_A = max(0, price_A_sample)
                self.cstr.energy_cost = max(0, energy_cost_sample)
                self.cstr.k0 = max(0, k0_sample)
    
                # Calculate profit
                profit = -self.cstr.profit(x)
                profits.append(profit)
    
                # Reset parameters
                self.cstr.price_A = original_price_A
                self.cstr.energy_cost = original_energy_cost
                self.cstr.k0 = original_k0
    
            # Calculate expected value
            expected_profit_value = np.mean(profits)
    
            return -expected_profit_value  # Negative for minimization
    
        def worst_case_profit(self, x):
            """
            Worst-case profit
    
            Assumes worst combination of parameters (minimum profit)
            """
            # Worst case: Raw material price↑, Energy cost↑, Reaction rate↓
            original_price_A = self.cstr.price_A
            original_energy_cost = self.cstr.energy_cost
            original_k0 = self.cstr.k0
    
            self.cstr.price_A = self.cstr.price_A + 2 * self.sigma_price_A
            self.cstr.energy_cost = self.cstr.energy_cost + 2 * self.sigma_energy
            self.cstr.k0 = self.cstr.k0 - 2 * self.sigma_k0
    
            profit_worst = -self.cstr.profit(x)
    
            # Reset
            self.cstr.price_A = original_price_A
            self.cstr.energy_cost = original_energy_cost
            self.cstr.k0 = original_k0
    
            return -profit_worst  # Negative for minimization
    
        def optimize_robust(self, method='expected'):
            """
            Execute robust optimization
    
            Parameters:
            -----------
            method : str
                'expected' or 'worst_case'
            """
    
            x0 = np.array([200.0, 60.0, 3.0])
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
    
            constraints = self.cstr.constraints(x0)
    
            print("=" * 60)
            print(f"Robust Optimization: {method} method")
            print("=" * 60)
            print(f"\nUncertainty:")
            print(f"  Raw material price: ±{self.sigma_price_A/self.cstr.price_A*100:.1f}%")
            print(f"  Energy cost: ±{self.sigma_energy/self.cstr.energy_cost*100:.1f}%")
            print(f"  Reaction rate constant: ±{self.sigma_k0/self.cstr.k0*100:.1f}%")
    
            if method == 'expected':
                objective = self.expected_profit
                print(f"\nObjective: Maximize expected profit (Monte Carlo samples: {self.n_samples})")
            elif method == 'worst_case':
                objective = self.worst_case_profit
                print(f"\nObjective: Maximize worst-case profit")
            else:
                raise ValueError("method must be 'expected' or 'worst_case'")
    
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
    
            if result.success:
                T_opt, tau_opt, C_A0_opt = result.x
    
                # Evaluate profit under various scenarios
                profit_nominal = -self.cstr.profit(result.x)
                profit_expected = -self.expected_profit(result.x)
                profit_worst = -self.worst_case_profit(result.x)
    
                print(f"\n✓ Robust optimization successful")
                print(f"\n【Robust Optimal Operating Conditions】")
                print(f"  Temperature: {T_opt:.2f}°C")
                print(f"  Residence time: {tau_opt:.2f}min")
                print(f"  Feed concentration: {C_A0_opt:.3f}mol/L")
    
                print(f"\n【Profit Under Each Scenario】")
                print(f"  Nominal case: ¥{profit_nominal:,.0f}/h")
                print(f"  Expected value: ¥{profit_expected:,.0f}/h")
                print(f"  Worst case: ¥{profit_worst:,.0f}/h")
    
                profit_range = profit_nominal - profit_worst
                print(f"\n  Profit variation range: ¥{profit_range:,.0f}/h ({profit_range/profit_nominal*100:.1f}%)")
            else:
                print(f"\n✗ Optimization failed: {result.message}")
    
            return result
    
        def compare_robust_strategies(self):
            """Compare robust optimization strategies"""
    
            print("\n" + "=" * 60)
            print("Robust Optimization Strategy Comparison")
            print("=" * 60)
    
            # Normal optimization (without considering uncertainty)
            optimizer_nominal = CSTRMultivariableOptimization()
            result_nominal = optimizer_nominal.optimize()
    
            # Expected value robust optimization
            result_expected = self.optimize_robust(method='expected')
    
            # Worst-case robust optimization
            result_worst = self.optimize_robust(method='worst_case')
    
            # Create comparison table
            strategies = ['Normal optimization', 'Expected value robust', 'Worst-case robust']
            results = [result_nominal, result_expected, result_worst]
    
            comparison_data = []
            for strategy, result in zip(strategies, results):
                if result.success:
                    T, tau, C_A0 = result.x
                    profit_nom = -self.cstr.profit(result.x)
                    profit_exp = -self.expected_profit(result.x)
                    profit_worst = -self.worst_case_profit(result.x)
    
                    comparison_data.append({
                        'Strategy': strategy,
                        'T [°C]': f"{T:.1f}",
                        'τ [min]': f"{tau:.1f}",
                        'C_A0 [mol/L]': f"{C_A0:.2f}",
                        'Nominal profit [¥/h]': f"{profit_nom:,.0f}",
                        'Expected profit [¥/h]': f"{profit_exp:,.0f}",
                        'Worst profit [¥/h]': f"{profit_worst:,.0f}"
                    })
    
            df_comparison = pd.DataFrame(comparison_data)
            print("\n")
            print(df_comparison.to_string(index=False))
    
            print("\n【Recommendation】")
            print("  - Emphasize stable operation: Worst-case robust optimization")
            print("  - Emphasize average profit: Expected value robust optimization")
            print("  - Accurate parameters: Normal optimization")
    
    # Execution
    robust_opt = RobustOptimization(n_samples=100)
    robust_opt.compare_robust_strategies()
    

## 5.4 Real-Time Optimization Framework

Real-Time Optimization (RTO) is an adaptive optimization method that updates model parameters based on online process data and periodically re-executes optimization.
    
    
    flowchart LR
        A[Process Operation] -->|Measurement Data| B[Data Collection]
        B --> C[Model Parameter Update]
        C --> D[Optimization Execution]
        D --> E[Optimal Operating Conditions]
        E -->|Control Commands| A
    
        D --> F{Goals Achieved?}
        F -->|No| G[Model Re-evaluation]
        G --> C
        F -->|Yes| H[Next Cycle]
        H --> B
    
        style A fill:#e8f5e9
        style D fill:#fff9c4
        style E fill:#ffe0b2
        style C fill:#f8bbd0
    

#### Code Example 8: Real-Time Optimization Framework
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import time
    
    class RealTimeOptimization:
        """
        Real-Time Optimization (RTO) Framework
    
        Features:
        1. Online data acquisition
        2. Model parameter update
        3. Re-execution of optimization
        4. Control command generation
        """
    
        def __init__(self):
            self.cstr = CSTROptimization()
    
            # RTO parameters
            self.update_interval = 60  # Update interval [min]
            self.n_iterations = 10     # Simulation iteration count
    
            # Data history
            self.time_history = []
            self.profit_history = []
            self.T_history = []
            self.tau_history = []
            self.C_A0_history = []
            self.X_A_history = []
    
            # Current operating conditions
            self.current_x = np.array([200.0, 60.0, 3.0])
    
        def simulate_online_data(self, iteration):
            """
            Simulate online data
    
            In real processes:
            - Acquire data from DCS (Distributed Control System)
            - Filter sensor measurements
            - Detect and remove outliers
            """
            # Simulate parameter drift
            # In reality, reaction rate changes due to aging, catalyst deactivation, etc.
            k0_drift = 1.0 - 0.05 * (iteration / self.n_iterations)  # 5% degradation
    
            # Add noise (measurement error)
            noise = np.random.normal(0, 0.02)
            k0_measured = self.cstr.k0 * k0_drift * (1 + noise)
    
            return {'k0': k0_measured, 'drift_factor': k0_drift}
    
        def update_model_parameters(self, online_data):
            """
            Update model parameters
    
            Implementation:
            - Moving Horizon Estimation (MHE)
            - Recursive Least Squares (RLS)
            - Kalman Filter
            """
            # Simplified: Directly update parameters with measurements
            self.cstr.k0 = online_data['k0']
    
            print(f"  Model parameter updated: k0 = {self.cstr.k0:.3e} (degradation rate: {(1-online_data['drift_factor'])*100:.1f}%)")
    
        def optimize_current_conditions(self):
            """Execute optimization under current conditions"""
    
            x0 = self.current_x
            bounds = [(50.0, 300.0), (30.0, 180.0), (1.0, 5.0)]
            constraints = self.cstr.constraints(x0)
    
            result = minimize(
                self.cstr.profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50}
            )
    
            if result.success:
                return result.x
            else:
                print(f"  Warning: Optimization failed ({result.message}), maintaining previous conditions")
                return self.current_x
    
        def run_rto_cycle(self, iteration):
            """Execute one RTO cycle"""
    
            print(f"\n--- RTO Cycle {iteration+1}/{self.n_iterations} (Time: {iteration*self.update_interval}min) ---")
    
            # Step 1: Online data acquisition
            online_data = self.simulate_online_data(iteration)
    
            # Step 2: Model parameter update
            self.update_model_parameters(online_data)
    
            # Step 3: Optimization execution
            optimal_x = self.optimize_current_conditions()
    
            # Step 4: Control command generation (gradual transition in implementation)
            # In reality, gradually change setpoints using MPC or PID
            self.current_x = optimal_x
    
            # Step 5: Performance evaluation
            profit = -self.cstr.profit(optimal_x)
            X_A = self.cstr.conversion(optimal_x[0], optimal_x[1])
    
            print(f"  Optimal operating conditions: T={optimal_x[0]:.1f}°C, τ={optimal_x[1]:.1f}min, C_A0={optimal_x[2]:.2f}mol/L")
            print(f"  Performance: Profit=¥{profit:,.0f}/h, Conversion={X_A*100:.2f}%")
    
            # Record to history
            self.time_history.append(iteration * self.update_interval)
            self.profit_history.append(profit)
            self.T_history.append(optimal_x[0])
            self.tau_history.append(optimal_x[1])
            self.C_A0_history.append(optimal_x[2])
            self.X_A_history.append(X_A)
    
        def run_rto_simulation(self):
            """Run entire RTO simulation"""
    
            print("=" * 60)
            print("Real-Time Optimization (RTO) Simulation")
            print("=" * 60)
            print(f"\nSettings:")
            print(f"  Update interval: {self.update_interval}min")
            print(f"  Simulation iterations: {self.n_iterations}")
            print(f"  Total operation time: {self.n_iterations * self.update_interval}min")
    
            for iteration in range(self.n_iterations):
                self.run_rto_cycle(iteration)
    
            # Visualize results
            self.visualize_rto_results()
    
        def visualize_rto_results(self):
            """Visualize RTO results"""
    
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
            # 1. Profit evolution
            ax1 = axes[0, 0]
            ax1.plot(self.time_history, self.profit_history, 'b-o', linewidth=2, markersize=5)
            ax1.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Profit [¥/h]', fontsize=11, fontweight='bold')
            ax1.set_title('Profit Time Evolution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
    
            # 2. Conversion evolution
            ax2 = axes[0, 1]
            ax2.plot(self.time_history, np.array(self.X_A_history) * 100, 'g-o', linewidth=2, markersize=5)
            ax2.axhline(y=95, color='r', linestyle='--', linewidth=1.5, label='Minimum conversion 95%')
            ax2.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Conversion [%]', fontsize=11, fontweight='bold')
            ax2.set_title('Conversion Time Evolution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            # 3. Temperature evolution
            ax3 = axes[1, 0]
            ax3.plot(self.time_history, self.T_history, 'm-o', linewidth=2, markersize=5)
            ax3.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Temperature [°C]', fontsize=11, fontweight='bold')
            ax3.set_title('Reaction Temperature Time Evolution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
            # 4. Residence time evolution
            ax4 = axes[1, 1]
            ax4.plot(self.time_history, self.tau_history, 'c-o', linewidth=2, markersize=5)
            ax4.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Residence Time [min]', fontsize=11, fontweight='bold')
            ax4.set_title('Residence Time Evolution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
    
            # 5. Feed concentration evolution
            ax5 = axes[2, 0]
            ax5.plot(self.time_history, self.C_A0_history, 'y-o', linewidth=2, markersize=5)
            ax5.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Feed Concentration [mol/L]', fontsize=11, fontweight='bold')
            ax5.set_title('Feed Concentration Time Evolution', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
            # 6. Cumulative profit
            ax6 = axes[2, 1]
            cumulative_profit = np.cumsum(self.profit_history) * (self.update_interval / 60)  # [¥]
            ax6.plot(self.time_history, cumulative_profit, 'r-o', linewidth=2, markersize=5)
            ax6.set_xlabel('Time [min]', fontsize=11, fontweight='bold')
            ax6.set_ylabel('Cumulative Profit [¥]', fontsize=11, fontweight='bold')
            ax6.set_title('Cumulative Profit', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('rto_simulation_results.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("\n" + "=" * 60)
            print("RTO Simulation Results Summary")
            print("=" * 60)
            print(f"\nInitial profit: ¥{self.profit_history[0]:,.0f}/h")
            print(f"Final profit: ¥{self.profit_history[-1]:,.0f}/h")
            print(f"Average profit: ¥{np.mean(self.profit_history):,.0f}/h")
            print(f"Profit variation: {(self.profit_history[-1] - self.profit_history[0]) / self.profit_history[0] * 100:.1f}%")
            print(f"\nTotal cumulative profit: ¥{cumulative_profit[-1]:,.0f}")
            print(f"\nConclusion: RTO adapts to catalyst degradation and maintains optimal operating conditions")
    
    # Execution
    rto = RealTimeOptimization()
    rto.run_rto_simulation()
    

## 5.5 Comprehensive Case Study: Multicomponent Distillation Column Optimization

Finally, we tackle the more complex distillation column optimization as a comprehensive case study.

### Problem Setting

**System** : Continuous distillation column for 5-component mixture

**Objective** : Minimize energy cost while meeting product purity specifications

**Decision Variables** :

  * **R** : Reflux Ratio [-] (1.5-4.0)
  * **Q R**: Reboiler heat duty [kW] (500-2000)
  * **N stages**: Number of stages [-] (20-50, integer)

**Constraints** :

  * Product purity: xproduct ≥ 0.98
  * Recovery: recovery ≥ 0.95
  * Pressure drop: ΔP ≤ 50 kPa
  * Material balance: F = D + B

#### Code Example 9: Complete Distillation Column Optimization Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    import matplotlib.pyplot as plt
    
    class DistillationColumnOptimization:
        """
        Multicomponent Distillation Column Optimization
    
        Objective: Minimize energy cost
        Constraints: Product purity, recovery, pressure drop
        """
    
        def __init__(self):
            # Economic parameters
            self.energy_cost = 15.0         # Steam cost [¥/kWh]
            self.cooling_cost = 3.0         # Cooling water cost [¥/kWh]
            self.product_price = 2000.0     # Product price [¥/kg]
    
            # Process parameters
            self.F = 1000.0                 # Feed flow rate [kg/h]
            self.x_F = 0.50                 # Feed composition (light component) [-]
            self.MW_avg = 80.0              # Average molecular weight [g/mol]
    
            # Physical property parameters (simplified)
            self.lambda_vap = 300.0         # Latent heat of vaporization [kJ/kg]
            self.Cp = 2.5                   # Specific heat [kJ/kg·K]
    
            # Constraint parameters
            self.purity_min = 0.98          # Minimum product purity [-]
            self.recovery_min = 0.95        # Minimum recovery [-]
            self.dP_max = 50.0              # Maximum pressure drop [kPa]
    
        def column_model(self, R, Q_R, N_stages):
            """
            Distillation column model (simplified)
    
            In reality:
            - MESH equations (Material, Equilibrium, Summation, Heat balance)
            - Equation of state (VLE: Vapor-Liquid Equilibrium)
            - Calculation using simulators like Aspen
    
            Here we use simplified shortcut method
            """
            # Simplified calculation based on Fenske-Underwood-Gilliland method
    
            # Minimum reflux ratio (simplified Underwood equation)
            alpha = 2.5  # Relative volatility (simplified)
            R_min = (alpha - 1) / alpha
    
            # Stage efficiency (empirical equation)
            efficiency = 0.60 + 0.008 * N_stages  # Higher efficiency with more stages
            efficiency = min(efficiency, 0.85)
    
            # Effective stages
            N_eff = N_stages * efficiency
    
            # Product purity (empirical equation, in reality calculated by MESH equations)
            purity = 1.0 - np.exp(-(N_eff / 30.0) * (R / R_min - 1.0))
            purity = min(purity, 0.995)  # Physical upper limit
    
            # Recovery
            recovery = 0.85 + 0.10 * (R / (R + 1))
            recovery = min(recovery, 0.98)
    
            # Distillate flow rate (material balance)
            D = self.F * self.x_F * recovery / purity  # [kg/h]
            B = self.F - D  # [kg/h]
    
            # Pressure drop (empirical equation)
            dP = 0.5 * N_stages + 10.0  # [kPa]
    
            # Condenser heat duty
            Q_C = (R + 1) * D * self.lambda_vap / 3600.0  # [kW] (kg/h→kg/s conversion)
    
            return {
                'purity': purity,
                'recovery': recovery,
                'D': D,
                'B': B,
                'Q_C': Q_C,
                'Q_R': Q_R,
                'dP': dP
            }
    
        def objective(self, x):
            """
            Objective function: Minimize total operating cost
    
            Cost = Reboiler cost + Condenser cost
            """
            R, Q_R, N_stages = x
            N_stages = int(round(N_stages))  # Integerization
    
            # Model calculation
            results = self.column_model(R, Q_R, N_stages)
    
            # Operating cost [¥/h]
            reboiler_cost = Q_R * self.energy_cost
            condenser_cost = results['Q_C'] * self.cooling_cost
    
            total_cost = reboiler_cost + condenser_cost
    
            return total_cost
    
        def constraints_func(self, x):
            """Constraints"""
            R, Q_R, N_stages = x
            N_stages = int(round(N_stages))
    
            results = self.column_model(R, Q_R, N_stages)
    
            # Inequality constraints (defined as ≥0)
            constraints = []
    
            # Purity constraint
            constraints.append(results['purity'] - self.purity_min)
    
            # Recovery constraint
            constraints.append(results['recovery'] - self.recovery_min)
    
            # Pressure drop constraint
            constraints.append(self.dP_max - results['dP'])
    
            # Physical constraint: Q_R must be sufficient for distillate
            Q_R_min = results['D'] * self.lambda_vap / 3600.0 * 0.5
            constraints.append(Q_R - Q_R_min)
    
            return constraints
    
        def optimize_distillation(self):
            """Execute distillation column optimization"""
    
            print("=" * 60)
            print("Multicomponent Distillation Column Optimization")
            print("=" * 60)
            print(f"\nObjective: Minimize energy cost")
            print(f"\nFeed conditions:")
            print(f"  Flow rate: {self.F:.0f} kg/h")
            print(f"  Composition (light component): {self.x_F*100:.0f}%")
    
            print(f"\nConstraints:")
            print(f"  Product purity ≥ {self.purity_min*100:.0f}%")
            print(f"  Recovery ≥ {self.recovery_min*100:.0f}%")
            print(f"  Pressure drop ≤ {self.dP_max:.0f} kPa")
    
            # Initial guess
            x0 = np.array([2.5, 1000.0, 30.0])
    
            # Variable bounds
            bounds = [
                (1.5, 4.0),       # R
                (500.0, 2000.0),  # Q_R [kW]
                (20, 50)          # N_stages
            ]
    
            # Constraints (scipy format)
            constraints = [
                {'type': 'ineq', 'fun': lambda x: self.constraints_func(x)[i]}
                for i in range(4)
            ]
    
            print(f"\nStarting optimization...")
    
            # Mixed-Integer Nonlinear Programming (MINLP)
            # Use differential_evolution for integer N_stages (global optimization)
            result = differential_evolution(
                self.objective,
                bounds,
                strategy='best1bin',
                maxiter=100,
                popsize=15,
                constraints=constraints,
                seed=42
            )
    
            # Interpret results
            self.interpret_distillation_results(result)
    
            return result
    
        def interpret_distillation_results(self, result):
            """Interpret distillation column optimization results"""
    
            print("\n" + "=" * 60)
            print("Distillation Column Optimization Results")
            print("=" * 60)
    
            if result.success:
                print("\n✓ Optimization successful")
            else:
                print(f"\n✗ Optimization failed: {result.message}")
                return
    
            R_opt, Q_R_opt, N_stages_opt = result.x
            N_stages_opt = int(round(N_stages_opt))
    
            results = self.column_model(R_opt, Q_R_opt, N_stages_opt)
    
            total_cost = result.fun
            reboiler_cost = Q_R_opt * self.energy_cost
            condenser_cost = results['Q_C'] * self.cooling_cost
    
            print(f"\n【Optimal Design and Operating Conditions】")
            print(f"  Reflux ratio R*: {R_opt:.3f}")
            print(f"  Reboiler heat duty Q_R*: {Q_R_opt:.1f} kW")
            print(f"  Number of stages N*: {N_stages_opt}")
    
            print(f"\n【Process Performance】")
            print(f"  Product purity: {results['purity']*100:.2f}% (requirement: ≥{self.purity_min*100:.0f}%)")
            print(f"  Recovery: {results['recovery']*100:.2f}% (requirement: ≥{self.recovery_min*100:.0f}%)")
            print(f"  Distillate: {results['D']:.1f} kg/h")
            print(f"  Bottoms: {results['B']:.1f} kg/h")
            print(f"  Pressure drop: {results['dP']:.1f} kPa (limit: {self.dP_max:.0f} kPa)")
    
            print(f"\n【Energy Cost】")
            print(f"  Reboiler cost: ¥{reboiler_cost:,.0f}/h")
            print(f"  Condenser cost: ¥{condenser_cost:,.0f}/h")
            print(f"  Total operating cost: ¥{total_cost:,.0f}/h")
    
            # Annual cost
            annual_cost = total_cost * 24 * 365
            print(f"  Annual operating cost: ¥{annual_cost:,.0f}/year")
    
            # Product value
            annual_product_value = results['D'] * 24 * 365 * self.product_price
            annual_profit = annual_product_value - annual_cost
    
            print(f"\n【Economic Evaluation】")
            print(f"  Annual product value: ¥{annual_product_value:,.0f}/year")
            print(f"  Annual profit (gross margin): ¥{annual_profit:,.0f}/year")
    
            # Sensitivity analysis hints
            print(f"\n【Optimization Insights】")
            print(f"  - Reflux ratio R*={R_opt:.2f}: Balance between energy cost and separation performance")
            print(f"  - Number of stages N*={N_stages_opt}: Trade-off between capital cost (CAPEX) and operating cost (OPEX)")
            print(f"  - Further improvements: Heat integration, intermediate heating, pressure optimization")
    
            # Visualization
            self.visualize_optimization_landscape(R_opt, Q_R_opt, N_stages_opt)
    
        def visualize_optimization_landscape(self, R_opt, Q_R_opt, N_stages_opt):
            """Visualize optimization landscape"""
    
            # Plot reflux ratio vs cost (Q_R and N_stages fixed at optimal values)
            R_range = np.linspace(1.5, 4.0, 30)
            costs_R = []
            purities_R = []
    
            for R in R_range:
                try:
                    cost = self.objective([R, Q_R_opt, N_stages_opt])
                    results = self.column_model(R, Q_R_opt, int(N_stages_opt))
                    costs_R.append(cost)
                    purities_R.append(results['purity'] * 100)
                except:
                    costs_R.append(np.nan)
                    purities_R.append(np.nan)
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
            # 1. Reflux ratio vs cost
            ax1.plot(R_range, costs_R, 'b-', linewidth=2)
            ax1.axvline(x=R_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal value R*={R_opt:.2f}')
            ax1.set_xlabel('Reflux Ratio R [-]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total Operating Cost [¥/h]', fontsize=12, fontweight='bold')
            ax1.set_title('Reflux Ratio vs Operating Cost', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
            # 2. Reflux ratio vs product purity
            ax2.plot(R_range, purities_R, 'g-', linewidth=2)
            ax2.axhline(y=self.purity_min*100, color='r', linestyle='--', linewidth=2, label=f'Minimum purity {self.purity_min*100:.0f}%')
            ax2.axvline(x=R_opt, color='orange', linestyle='--', linewidth=2, label=f'Optimal value R*={R_opt:.2f}')
            ax2.set_xlabel('Reflux Ratio R [-]', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Product Purity [%]', fontsize=12, fontweight='bold')
            ax2.set_title('Reflux Ratio vs Product Purity', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
            plt.tight_layout()
            plt.savefig('distillation_optimization_landscape.png', dpi=150, bbox_inches='tight')
            plt.show()
    
            print("\nVisualization complete: distillation_optimization_landscape.png")
    
    # Execution
    distillation_opt = DistillationColumnOptimization()
    result_distillation = distillation_opt.optimize_distillation()
    

## Summary

#### What We Learned in Chapter 5

  * **Complete Optimization Workflow** : Systematic process from problem definition to model development, optimization execution, and result validation
  * **CSTR Optimization** : Implementation of economic objective functions, multivariable optimization, and comprehensive constraints
  * **Sensitivity Analysis** : Robustness evaluation of optimal solutions against parameter variations
  * **Robust Optimization** : Expected value optimization and worst-case optimization under uncertainty
  * **Real-Time Optimization** : Adaptive optimization framework based on online data
  * **Distillation Column Optimization** : Comprehensive case study of complex multivariable constrained optimization

### Key Points

  1. **Economic Objective Function Design** : Profit maximization = Revenue - Raw material cost - Energy cost
  2. **Realistic Constraints** : Comprehensive implementation of safety constraints, product specifications, environmental standards, and operational constraints
  3. **Importance of Sensitivity Analysis** : Quantitative evaluation of the impact of parameter uncertainty on optimal solutions
  4. **Robust Optimization** : Search for operating conditions that guarantee performance under uncertainty
  5. **Real-Time Optimization (RTO)** : Dynamic optimization approach that adapts to process variations

### Practical Application

**Short-term Actions (1-3 months)**

  * Evaluate optimization opportunities in your processes (energy cost, yield, quality)
  * Start with simple single-variable optimization (temperature, pressure, flow rate adjustment)
  * Measure current performance baseline using DCS/PLC data

**Medium-term Actions (3-12 months)**

  * Develop process models (first-principles models or data-driven models)
  * Implement and validate multivariable optimization (simulation environment)
  * Practice sensitivity analysis and robust optimization

**Long-term Actions (1-2 years)**

  * Build and implement Real-Time Optimization (RTO) systems
  * Integrated optimization of entire plant (coordination of multiple units)
  * Integration with Model Predictive Control (MPC)

### Further Learning Resources

**Recommended Books**

  * **Biegler, L.T. (2010)** : "Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes"
  * **Edgar, T.F., et al. (2001)** : "Optimization of Chemical Processes"
  * **Seborg, D.E., et al. (2016)** : "Process Dynamics and Control" (MPC related)

**Next Steps**

  * **Dynamic Optimization** : Batch process, startup optimization
  * **Model Predictive Control (MPC)** : Integration of real-time control and optimization
  * **Bayesian Optimization** : Optimization of black-box processes
  * **Machine Learning Integration** : Surrogate models, reinforcement learning

#### Implementation Tips

**Phased Approach** : First verify effectiveness with offline optimization → Small-scale pilot testing → Phased full implementation

**Safety First** : All optimizations must prioritize safety constraints and implement emergency shutdown logic

**Continuous Improvement** : Make periodic model updates and optimization result validation a habit

## Conclusion

Through all five chapters of this "Process Optimization Introduction Series," you have acquired comprehensive knowledge and skills from optimization problem formulation to optimal operating condition search for real chemical processes.

Optimization is not just a mathematical technique but a powerful tool to enhance competitiveness and achieve sustainability in the process industries. We hope you will apply the techniques learned in this series to practical work and contribute to improving process economics, energy efficiency, and reducing environmental impact.

**The journey of optimization does not end here—it begins!**

[← Chapter 4: Constrained Optimization](<chapter-4.html>) [Return to Series Top →](<./index.html>)
