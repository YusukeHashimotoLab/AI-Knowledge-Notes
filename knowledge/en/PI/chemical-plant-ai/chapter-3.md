---
title: "Chapter 3: Real-Time Optimization and APC"
chapter_title: "Chapter 3: Real-Time Optimization and APC"
subtitle: Implementation of Economic Optimization and Model Predictive Control in Chemical Plants
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›[Chemical Plant Ai](<../../PI/chemical-plant-ai/index.html>)›Chapter 3

🌐 EN | [🇯🇵 JP](<../../../jp/PI/chemical-plant-ai/chapter-3.html>) | Last sync: 2025-11-16

**What you will learn in this chapter:**

In chemical plant operations, Real-Time Optimization (RTO) and Advanced Process Control (APC) are critical technologies for achieving both economic efficiency and safety. This chapter covers optimization using SciPy and Pyomo, implementation of Model Predictive Control (MPC), and next-generation process control using deep reinforcement learning (DQN, PPO) at the implementation level.

  * **Economic Optimization** ：Achieve product value maximization and utility cost minimization through mathematical optimization
  * **Model Predictive Control** ：Calculate optimal manipulated variables under constraints while predicting future behavior
  * **Reinforcement Learning-Based Control** ：Autonomous control of batch and continuous processes using DQN and PPO
  * **Hierarchical Control Systems** ：Practical plant control architecture integrating RTO and APC layers

## 3.1 Hierarchical Structure of Process Control

In modern chemical plants, control systems are organized hierarchically. Each layer operates on different time scales, with lower layers achieving the optimization objectives set by upper layers.
    
    
    ```mermaid
    graph TB
        subgraph "Hierarchical Process Control System"
            RTO[Real-Time Optimization LayerReal-Time OptimizationTime Scale: Hours to Days]
            APC[Advanced Control LayerAdvanced Process ControlTime Scale: Minutes to Hours]
            REG[Regulatory Control LayerRegulatory ControlTime Scale: Seconds to Minutes]
            PROC[ProcessChemical Plant]
        end
    
        RTO -->|Optimal operating conditions| APC
        APC -->|Setpoints| REG
        REG -->|Manipulated variable| PROC
        PROC -->|Measured Values| REG
        PROC -->|State| APC
        PROC -->|Economic Indicators| RTO
    
        style RTO fill:#e3f2fd
        style APC fill:#fff3e0
        style REG fill:#e8f5e9
        style PROC fill:#f3e5f5
    ```

🎯 Hierarchical Control Example

**Case of Petroleum Refining Plant (FCC Unit):**

  * **RTO Layer (6-hour cycle)** ：Determine optimal light fraction ratio based on crude oil price and gasoline demand
  * **APC Layer (5-minute cycle)** ：Control reaction temperature and catalyst circulation with MPC to achieve target fraction ratio
  * **Regulatory Control Layer（1Second(s)Cycle）** ：Instantly adjust temperature, pressure, and flow rate with PID controllers

This hierarchical approach has demonstrated improvements in plant revenue by hundreds of millions of yen annually.

## 3.2 Online Optimization (SciPy)

In real-time optimization, the current plant state is used as input to calculate optimal operating conditions that satisfy economic objective functions (such as profit maximization). We implement an example using continuous stirred tank reactor (CSTR) operation optimization.

Example 1: Online Optimization (SciPy) - CSTR Operating Condition Optimization
    
    
    """
    ===================================
    Example 1: OnlineOptimization（SciPy）
    ===================================
    
    Continuous Stirred Tank Reactor（CSTR）inTemperature and Flow rate Optimization。
    Reaction rate and Selectivity Trade-off 考慮し、単位WhenBetweenあたり Revenue Maximum。
    
    Objective: ProductValue Maximumしな 、Energy cost and Feed cost Minimization
    """
    
    import numpy as np
    from scipy.optimize import minimize, NonlinearConstraint
    from typing import Dict, Tuple
    import pandas as pd
    
    
    class CSTROptimizer:
        """Continuous Stirred Tank Reactor Real-timeOptimization"""
    
        def __init__(self):
            # ProcessParameters
            self.volume = 10.0  # Reactor volume [m³]
            self.heat_capacity = 4.18  # Heat capacity [kJ/kg·K]
            self.density = 1000.0  # Density [kg/m³]
    
            # Reaction rate constant (ArrheniusEquation)
            self.A1 = 1.2e10  # Frequency factor（Main reaction） [1/h]
            self.E1 = 75000.0  # Activation energy（Main reaction） [J/mol]
            self.A2 = 3.5e9   # Frequency factor（Side reaction） [1/h]
            self.E2 = 68000.0  # Activation energy（Side reaction） [J/mol]
    
            # Economic parameters
            self.product_price = 150.0  # Product price [$/kg]
            self.byproduct_price = 40.0  # Byproduct price [$/kg]
            self.feed_cost = 50.0  # Feed cost [$/kg]
            self.energy_cost = 0.08  # Energy cost [$/kWh]
    
            # Physical constraints
            self.T_min, self.T_max = 320.0, 380.0  # Temperature range [K]
            self.F_min, self.F_max = 0.5, 5.0      # Flow rate range [m³/h]
            self.T_feed = 298.0  # Feed temperature [K]
    
        def reaction_rates(self, T: float) -> Tuple[float, float]:
            """Reaction rate constant Calculate（ArrheniusEquation）
    
            Args:
                T: Reaction temperature [K]
    
            Returns:
                (主Reaction rate constant, 副Reaction rate constant) [1/h]
            """
            R = 8.314  # Gas constant [J/mol·K]
            k1 = self.A1 * np.exp(-self.E1 / (R * T))
            k2 = self.A2 * np.exp(-self.E2 / (R * T))
            return k1, k2
    
        def conversion_selectivity(self, T: float, tau: float) -> Tuple[float, float]:
            """ReactionConversion and Selectivity Calculate
    
            Args:
                T: Reaction temperature [K]
                tau: Residence time [h]
    
            Returns:
                (Conversion, Selectivity)
            """
            k1, k2 = self.reaction_rates(T)
    
            # 1NextReaction Conversion
            conversion = 1.0 - np.exp(-(k1 + k2) * tau)
    
            # Selectivity（主生成物 / All生成物）
            selectivity = k1 / (k1 + k2)
    
            return conversion, selectivity
    
        def heating_power(self, T: float, F: float) -> float:
            """Calculate power required for heating
    
            Args:
                T: Reaction temperature [K]
                F: Flow rate [m³/h]
    
            Returns:
                Heating power [kW]
            """
            delta_T = T - self.T_feed
            mass_flow = F * self.density  # [kg/h]
            heat_duty = mass_flow * self.heat_capacity * delta_T  # [kJ/h]
            return heat_duty / 3600.0  # [kW]
    
        def objective(self, x: np.ndarray) -> float:
            """Objective function：Profit Negative value Calculate（MinimizationProblem Conversion）
    
            Args:
                x: [Temperature [K], Flow rate [m³/h]]
    
            Returns:
                -Profit [$/h]
            """
            T, F = x
            tau = self.volume / F  # Residence time [h]
    
            # Conversion and Selectivity
            conversion, selectivity = self.conversion_selectivity(T, tau)
    
            # Product generation
            feed_mass = F * self.density  # [kg/h]
            product_mass = feed_mass * conversion * selectivity
            byproduct_mass = feed_mass * conversion * (1 - selectivity)
    
            # Revenue
            revenue = (product_mass * self.product_price +
                      byproduct_mass * self.byproduct_price)
    
            # Cost
            feed_cost_total = feed_mass * self.feed_cost
            energy_cost_total = self.heating_power(T, F) * self.energy_cost
    
            profit = revenue - feed_cost_total - energy_cost_total
    
            return -profit  # Minimization Negative value
    
        def optimize(self, initial_guess: np.ndarray = None) -> Dict:
            """Optimization Execute
    
            Args:
                initial_guess: Initial guess [Temperature, Flow rate]
    
            Returns:
                OptimizationResults Dictionary
            """
            if initial_guess is None:
                initial_guess = np.array([350.0, 2.0])  # [K, m³/h]
    
            # Boundary constraints
            bounds = [(self.T_min, self.T_max),
                     (self.F_min, self.F_max)]
    
            # NonlinearConstraint：Conversion 0.85 or more（安All・品質要件）
            def conversion_constraint(x):
                T, F = x
                tau = self.volume / F
                conversion, _ = self.conversion_selectivity(T, tau)
                return conversion - 0.85
    
            nlc = NonlinearConstraint(conversion_constraint, 0, np.inf)
    
            # OptimizationExecute
            result = minimize(
                self.objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[nlc],
                options={'ftol': 1e-6, 'disp': False}
            )
    
            # Organize results
            T_opt, F_opt = result.x
            tau_opt = self.volume / F_opt
            conversion, selectivity = self.conversion_selectivity(T_opt, tau_opt)
    
            return {
                'success': result.success,
                'temperature': T_opt,
                'flow_rate': F_opt,
                'residence_time': tau_opt,
                'conversion': conversion,
                'selectivity': selectivity,
                'profit_per_hour': -result.fun,
                'heating_power': self.heating_power(T_opt, F_opt)
            }
    
    
    # ===================================
    # Execution Example
    # ===================================
    if __name__ == "__main__":
        optimizer = CSTROptimizer()
    
        print("="*70)
        print("CSTR Real-timeOptimization")
        print("="*70)
    
        # OptimizationExecute
        result = optimizer.optimize()
    
        if result['success']:
            print("\n【Optimal operating conditions】")
            print(f"  Reaction temperature: {result['temperature']:.1f} K ({result['temperature']-273.15:.1f} °C)")
            print(f"  Flow rate: {result['flow_rate']:.2f} m³/h")
            print(f"  Residence time: {result['residence_time']:.2f} h")
            print(f"\n【ProcessPerformance】")
            print(f"  Conversion: {result['conversion']:.1%}")
            print(f"  Selectivity: {result['selectivity']:.1%}")
            print(f"  Heating power: {result['heating_power']:.1f} kW")
            print(f"\n【Economics】")
            print(f"  Profit: ${result['profit_per_hour']:.2f}/h")
            print(f"  YearBetweenProfit: ${result['profit_per_hour'] * 8760:.0f}/year")
        else:
            print("Optimization Failedしました。")
    
        # Sensitivity Analysis：Feed価格 Variation 対OptimalCondition Change
        print("\n" + "="*70)
        print("Sensitivity Analysis：Impact of Feedstock Price")
        print("="*70)
    
        feed_costs = [40, 50, 60, 70]
        results = []
    
        for cost in feed_costs:
            optimizer.feed_cost = cost
            res = optimizer.optimize()
            results.append({
                'Feed cost [$/kg]': cost,
                'OptimalTemperature [K]': res['temperature'],
                'OptimalFlow rate [m³/h]': res['flow_rate'],
                'WhenBetweenProfit [$/h]': res['profit_per_hour']
            })
    
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    
        print("\n✓ Feed価格IncreaseWhen High temperature・LowFlow rate Optimization（SelectivityFocus on）")
    

## 3.3 Economic Optimization（Pyomo）

For more complex optimization problems, we use the algebraic modeling language Pyomo. We implement a formulation that simultaneously considers product value maximization and utility cost minimization.

Example 2: Economic Optimization（Pyomo） - ProductValueMaximum and CostMinimization
    
    
    """
    ===================================
    Example 2: Economic Optimization（Pyomo）
    ===================================
    
    Pyomo UsingChemical Plant EconomicOptimization。
    Multiple Product 持つProcess おいて、ProductValue Maximumしな 
    ーUtilityCost（Steam、Power、Cooling water） Minimization。
    
    Pyomo Actual  Installー Necessary  、Here  概念な実装 示s。
    """
    
    import numpy as np
    from scipy.optimize import minimize
    from typing import Dict, List
    import pandas as pd
    
    
    class EconomicOptimizer:
        """Chemical Plant EconomicOptimizationSystem"""
    
        def __init__(self):
            # Product price [$/ton]
            self.product_prices = {
                'ProductA': 800.0,   # High value-added product
                'ProductB': 500.0,   # Intermediate product
                'ProductC': 300.0    # Commodity product
            }
    
            # ーUtilityCost
            self.steam_cost = 25.0    # Steam [$/ton]
            self.power_cost = 0.10    # Power [$/kWh]
            self.cooling_cost = 0.5   # Cooling water [$/ton]
    
            # ProcessConstraintParameters
            self.max_capacity = 100.0  # Maximum processing capacity [ton/h]
            self.min_turndown = 0.4    # Minimum turndown ratio
    
        def production_model(self, feed_rate: float, temperature: float,
                            pressure: float) -> Dict[str, float]:
            """Production model: Calculate product yields from operating conditions
    
            Args:
                feed_rate: FeedFlow rate [ton/h]
                temperature: Reaction temperature [K]
                pressure: Reaction pressure [bar]
    
            Returns:
                Production of each product [ton/h]
            """
            # Simplified yield model (actual implementation uses detailed reaction model)
            T_ref = 400.0  # 基準Temperature [K]
            P_ref = 20.0   # Reference pressure [bar]
    
            # Temperature・Pressure Impact因子
            temp_factor = np.exp(-0.005 * (temperature - T_ref)**2)
            press_factor = 1.0 + 0.02 * (pressure - P_ref)
    
            # Base yield for each product
            yield_A_base = 0.35 * temp_factor * press_factor
            yield_B_base = 0.45 * (2.0 - temp_factor)
            yield_C_base = 0.20
    
            # Yield constraint (sum ≤ 1.0)
            total_yield = yield_A_base + yield_B_base + yield_C_base
            if total_yield > 1.0:
                scale = 1.0 / total_yield
                yield_A_base *= scale
                yield_B_base *= scale
                yield_C_base *= scale
    
            return {
                'ProductA': feed_rate * yield_A_base,
                'ProductB': feed_rate * yield_B_base,
                'ProductC': feed_rate * yield_C_base
            }
    
        def utility_consumption(self, feed_rate: float, temperature: float,
                               pressure: float) -> Dict[str, float]:
            """Calculate utility consumption
    
            Args:
                feed_rate: FeedFlow rate [ton/h]
                temperature: Reaction temperature [K]
                pressure: Reaction pressure [bar]
    
            Returns:
                Utility consumption
            """
            # SteamConsumption（Proportional to heating load）
            T_feed = 298.0  # Feed temperature [K]
            heating_load = feed_rate * 2.5 * (temperature - T_feed)  # Simplified equation
            steam = heating_load / 2000.0  # [ton/h]
    
            # PowerConsumption（Compressor, pump, agitation）
            compressor_power = 50.0 * (pressure / 20.0)**0.8  # [kW]
            pump_power = 10.0 * feed_rate
            agitator_power = 15.0
            power = compressor_power + pump_power + agitator_power
    
            # Cooling water（Exothermic heat removal）
            exothermic_heat = feed_rate * 500.0  # [kW] (Assumption)
            cooling_water = exothermic_heat / 40.0  # [ton/h]
    
            return {
                'steam': steam,
                'power': power,
                'cooling': cooling_water
            }
    
        def objective_function(self, x: np.ndarray) -> float:
            """Objective function：Profit Negative value（MinimizationProblem）
    
            Args:
                x: [feed_rate, temperature, pressure]
    
            Returns:
                -Profit [$/h]
            """
            feed_rate, temperature, pressure = x
    
            # Product Production
            products = self.production_model(feed_rate, temperature, pressure)
    
            # Revenue
            revenue = sum(
                products[prod] * self.product_prices[prod]
                for prod in products
            )
    
            # Utility consumption
            utilities = self.utility_consumption(feed_rate, temperature, pressure)
    
            # Cost
            utility_cost = (
                utilities['steam'] * self.steam_cost +
                utilities['power'] * self.power_cost +
                utilities['cooling'] * self.cooling_cost
            )
    
            # Feed cost（Assumption: $200/ton）
            feed_cost = feed_rate * 200.0
    
            profit = revenue - utility_cost - feed_cost
    
            return -profit
    
        def optimize_economics(self) -> Dict:
            """EconomicOptimization Execute
    
            Returns:
                OptimizationResults
            """
            # Initial guess: [feed_rate, temperature, pressure]
            x0 = np.array([60.0, 400.0, 20.0])
    
            # Boundary constraints
            bounds = [
                (self.max_capacity * self.min_turndown, self.max_capacity),  # feed_rate
                (350.0, 450.0),  # temperature [K]
                (10.0, 40.0)     # pressure [bar]
            ]
    
            # Optimization
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-6}
            )
    
            feed_opt, temp_opt, press_opt = result.x
    
            # Organize results
            products = self.production_model(feed_opt, temp_opt, press_opt)
            utilities = self.utility_consumption(feed_opt, temp_opt, press_opt)
    
            return {
                'success': result.success,
                'feed_rate': feed_opt,
                'temperature': temp_opt,
                'pressure': press_opt,
                'products': products,
                'utilities': utilities,
                'profit_per_hour': -result.fun
            }
    
    
    # ===================================
    # Execution Example
    # ===================================
    if __name__ == "__main__":
        optimizer = EconomicOptimizer()
    
        print("="*70)
        print("Chemical PlantEconomicOptimization")
        print("="*70)
    
        result = optimizer.optimize_economics()
    
        if result['success']:
            print("\n【Optimal operating conditions】")
            print(f"  FeedFlow rate: {result['feed_rate']:.1f} ton/h")
            print(f"  Reaction temperature: {result['temperature']:.1f} K ({result['temperature']-273.15:.1f} °C)")
            print(f"  Reaction pressure: {result['pressure']:.1f} bar")
    
            print("\n【Product Production】")
            for prod, amount in result['products'].items():
                price = optimizer.product_prices[prod]
                value = amount * price
                print(f"  {prod}: {amount:.2f} ton/h (Value: ${value:.2f}/h)")
    
            print("\n【Utility consumption】")
            util = result['utilities']
            print(f"  Steam: {util['steam']:.2f} ton/h (${util['steam'] * optimizer.steam_cost:.2f}/h)")
            print(f"  Power: {util['power']:.1f} kW (${util['power'] * optimizer.power_cost:.2f}/h)")
            print(f"  Cooling water: {util['cooling']:.2f} ton/h (${util['cooling'] * optimizer.cooling_cost:.2f}/h)")
    
            print("\n【Economics】")
            print(f"  WhenBetweenProfit: ${result['profit_per_hour']:.2f}/h")
            print(f"  DayBetweenProfit: ${result['profit_per_hour'] * 24:.2f}/day")
            print(f"  YearBetweenProfit: ${result['profit_per_hour'] * 8760:.0f}/year")
    
        # Product price Variation 対Sensitivity Analysis
        print("\n" + "="*70)
        print("Sensitivity Analysis：Impact of Product A Price")
        print("="*70)
    
        original_price = optimizer.product_prices['ProductA']
        price_scenarios = [600, 700, 800, 900, 1000]
        results_table = []
    
        for price in price_scenarios:
            optimizer.product_prices['ProductA'] = price
            res = optimizer.optimize_economics()
            results_table.append({
                'Product A Price [$/ton]': price,
                'OptimalFlow rate [ton/h]': res['feed_rate'],
                'OptimalTemperature [K]': res['temperature'],
                'Product A Production [ton/h]': res['products']['ProductA'],
                'WhenBetweenProfit [$/h]': res['profit_per_hour']
            })
    
        df = pd.DataFrame(results_table)
        print(df.to_string(index=False))
    
        print("\n✓ High value-added product 価格IncreaseWhen、High temperatureOperation Yield Maximum")
    

## 3.4 Model Predictive Control (MPC)

Model Predictive Control (MPC) is an advanced control technique that uses a dynamic model of the process to predict future behavior and optimize manipulated variables while satisfying constraints. We demonstrate a basic implementation using distillation column temperature control as an example.

Example 3: Basic MPC Implementation - Distillation Column Temperature Control
    
    
    """
    ===================================
    Example 3: MPC 基礎実装
    ===================================
    
    Distillation columninTemperatureControl Model Predictive Control（MPC） 適用。
    未来 挙動 Predictionしな 、RefluxRatio and RefluxFlow rate Optimization。
    
    MPC 特長：
    - 多VariableControl（Multiple OperationVariable・ControlVariable）
    - Constraints 明示な考慮
    - OutsideDisturbance Prediction and 補償
    """
    
    import numpy as np
    from scipy.optimize import minimize
    from typing import List, Tuple
    import matplotlib.pyplot as plt
    
    
    class DistillationMPC:
        """Model Predictive Control of Distillation Column"""
    
        def __init__(self, prediction_horizon: int = 10, control_horizon: int = 5):
            """
            Args:
                prediction_horizon: Prediction horizon (number of steps)
                control_horizon: Control horizon (number of steps)
            """
            self.Np = prediction_horizon
            self.Nc = control_horizon
            self.dt = 1.0  # Sampling time [min]
    
            # ProcessModel（State空BetweenModel）
            # x[k+1] = A*x[k] + B*u[k]
            # y[k] = C*x[k]
            # State: x = [TopTemperatureDeviation, BottomTemperatureDeviation]
            # Input: u = [RefluxRatioChange, RefluxFlow rateChange]
            # Output: y = [TopTemperature, BottomTemperature]
    
            self.A = np.array([
                [0.85, 0.10],
                [0.05, 0.90]
            ])
            self.B = np.array([
                [0.5, 0.1],
                [0.1, 0.4]
            ])
            self.C = np.eye(2)
    
            # Constraints
            self.u_min = np.array([-2.0, -5.0])  # [RefluxRatio, RefluxFlow rate kg/min]
            self.u_max = np.array([2.0, 5.0])
            self.delta_u_max = np.array([0.5, 1.0])  # Rate of change constraints
    
            # Weight matrices
            self.Q = np.diag([10.0, 8.0])   # OutputTracking Weight
            self.R = np.diag([1.0, 1.0])    # InputChange Weight
    
        def predict(self, x0: np.ndarray, u_sequence: np.ndarray) -> np.ndarray:
            """Predictionホライズン わたるOutput Prediction
    
            Args:
                x0: Current State
                u_sequence: InputSequence (Nc x 2)
    
            Returns:
                PredictionOutputSequence (Np x 2)
            """
            x = x0.copy()
            y_pred = np.zeros((self.Np, 2))
    
            for k in range(self.Np):
                # Control horizonInside OptimizationVariable、それ and after Last Value Hold
                if k < self.Nc:
                    u = u_sequence[k]
                else:
                    u = u_sequence[-1]
    
                # StateUpdate
                x = self.A @ x + self.B @ u
    
                # OutputCalculate
                y_pred[k] = self.C @ x
    
            return y_pred
    
        def mpc_objective(self, u_flat: np.ndarray, x0: np.ndarray,
                         r: np.ndarray, u_prev: np.ndarray) -> float:
            """MPCObjective function
    
            Args:
                u_flat: 平坦されたInputSequence (Nc*2,)
                x0: Current State
                r: Setpoint sequence (Np x 2)
                u_prev: FrontStep Input
    
            Returns:
                Objective function value
            """
            # InputSequence 再構成
            u_sequence = u_flat.reshape(self.Nc, 2)
    
            # Prediction
            y_pred = self.predict(x0, u_sequence)
    
            # Tracking error
            tracking_error = 0.0
            for k in range(self.Np):
                e = y_pred[k] - r[k]
                tracking_error += e.T @ self.Q @ e
    
            # InputChangePenalty
            control_effort = 0.0
            for k in range(self.Nc):
                if k == 0:
                    du = u_sequence[k] - u_prev
                else:
                    du = u_sequence[k] - u_sequence[k-1]
                control_effort += du.T @ self.R @ du
    
            return tracking_error + control_effort
    
        def solve(self, x0: np.ndarray, r: np.ndarray,
                 u_prev: np.ndarray) -> np.ndarray:
            """MPCOptimizationProblem 解く
    
            Args:
                x0: Current State
                r: Setpoint sequence (Np x 2)
                u_prev: FrontStep Input
    
            Returns:
                OptimalInputSequence First Value (2,)
            """
            # Initial guess
            u0_flat = np.zeros(self.Nc * 2)
    
            # Boundary constraints
            bounds = []
            for _ in range(self.Nc):
                bounds.extend([
                    (self.u_min[0], self.u_max[0]),
                    (self.u_min[1], self.u_max[1])
                ])
    
            # Rate of change constraints
            def delta_u_constraint(u_flat):
                u_seq = u_flat.reshape(self.Nc, 2)
                violations = []
                for k in range(self.Nc):
                    if k == 0:
                        du = np.abs(u_seq[k] - u_prev)
                    else:
                        du = np.abs(u_seq[k] - u_seq[k-1])
                    violations.extend((self.delta_u_max - du).tolist())
                return np.array(violations)
    
            from scipy.optimize import NonlinearConstraint
            nlc = NonlinearConstraint(delta_u_constraint, 0, np.inf)
    
            # Optimization
            result = minimize(
                lambda u: self.mpc_objective(u, x0, r, u_prev),
                u0_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=[nlc],
                options={'ftol': 1e-4, 'disp': False}
            )
    
            u_opt = result.x.reshape(self.Nc, 2)
            return u_opt[0]  # First Step みExecute
    
    
    # ===================================
    # Simulation Execution
    # ===================================
    if __name__ == "__main__":
        mpc = DistillationMPC(prediction_horizon=10, control_horizon=5)
    
        # Simulation settings
        T_sim = 50  # Simulation time [min]
        x = np.zeros(2)  # InitialState（Setpoint from  Deviation）
        u = np.zeros(2)  # InitialInput
    
        # Setpoint changes (step changes)
        r_top = np.zeros(T_sim)
        r_bottom = np.zeros(T_sim)
        r_top[10:] = -1.5  # 10minBack TopTemperature 1.5℃Downげる
        r_bottom[30:] = 1.0  # 30minBack BottomTemperature 1.0℃Upげる
    
        # For recording
        x_history = [x.copy()]
        u_history = [u.copy()]
    
        print("="*70)
        print("Distillation Column MPC Control Simulation")
        print("="*70)
    
        for k in range(T_sim):
            # Setpoint sequence（Predictionホライズンmin）
            r_horizon = np.zeros((mpc.Np, 2))
            for i in range(mpc.Np):
                if k + i < T_sim:
                    r_horizon[i] = [r_top[k+i], r_bottom[k+i]]
                else:
                    r_horizon[i] = [r_top[-1], r_bottom[-1]]
    
            # MPC OptimalInput Calculate
            u = mpc.solve(x, r_horizon, u)
    
            # ProcessUpdate（OutsideDisturbance 加える）
            disturbance = np.random.randn(2) * 0.05
            x = mpc.A @ x + mpc.B @ u + disturbance
    
            # Record
            x_history.append(x.copy())
            u_history.append(u.copy())
    
            if k % 10 == 0:
                print(f"Time {k:2d}min: Top deviation={x[0]:+.2f}℃, Bottom deviation={x[1]:+.2f}℃, "
                      f"Reflux={u[0]:+.2f}, Reflux flow={u[1]:+.2f} kg/min")
    
        # Convert results to array
        x_history = np.array(x_history)
        u_history = np.array(u_history)
    
        print("\n" + "="*70)
        print("Control Performance Evaluation")
        print("="*70)
    
        # Steady-state deviation
        steady_state_error_top = np.abs(x_history[-10:, 0] - r_top[-1]).mean()
        steady_state_error_bottom = np.abs(x_history[-10:, 1] - r_bottom[-1]).mean()
    
        print(f"TopTemperature Steady-state deviation: {steady_state_error_top:.3f} ℃")
        print(f"BottomTemperature Steady-state deviation: {steady_state_error_bottom:.3f} ℃")
        print(f"\n✓ MPC  ConstraintsDown High精度なTemperatureControl Achieve")
        print(f"✓ Multiple Setpoints変更 対して 良好なTrackingPerformance")
    

## 3.5 Nonlinear MPC (CasADi)

For nonlinear processes, nonlinear MPC is required. We show a nonlinear MPC implementation for a reactor using CasADi (conceptual implementation).

Example 4: Nonlinear MPC (CasADi) - Nonlinear Control of Reactor
    
    
    """
    ===================================
    Example 4: Nonlinear MPC (CasADi)
    ===================================
    
    CasADi UsingNonlinearModelPredictionControl。
    Continuous Stirred Tank Reactor（CSTR） Concentration and Temperature 同WhenControl。
    
    NonlinearProcessModel：
    - Mass balance: dC/dt = (C_in - C)/tau - k*C
    - Energy balance: dT/dt = (T_in - T)/tau + (-ΔH)*k*C/(ρ*Cp) + Q/(V*ρ*Cp)
    
    CasADi Actual  Installー Necessary  、Here  数Valueな実装 代替s。
    """
    
    import numpy as np
    from scipy.integrate import odeint
    from scipy.optimize import minimize
    from typing import Tuple
    
    
    class NonlinearCSTRMPC:
        """MPC for nonlinear CSTR reactor"""
    
        def __init__(self, prediction_horizon: int = 20, control_horizon: int = 10):
            self.Np = prediction_horizon
            self.Nc = control_horizon
            self.dt = 0.5  # Sampling time [min]
    
            # ProcessParameters
            self.V = 1.0  # Reactor volume [m³]
            self.rho = 1000.0  # Density [kg/m³]
            self.Cp = 4.18  # Specific heat [kJ/kg·K]
            self.delta_H = -50000.0  # Heat of reaction [kJ/kmol]
    
            # ArrheniusReaction rate
            self.A = 1.0e10  # Frequency factor [1/min]
            self.Ea = 75000.0  # Activation energy [J/mol]
            self.R = 8.314  # Gas constant [J/mol·K]
    
            # OperationVariable Constraint
            self.F_min, self.F_max = 0.02, 0.20  # Flow rate [m³/min]
            self.Q_min, self.Q_max = -5000, 5000  # Heating/cooling [kJ/min]
    
            # Input
            self.C_in = 10.0  # Inlet concentration [kmol/m³]
            self.T_in = 300.0  # 入口Temperature [K]
    
        def reaction_rate(self, C: float, T: float) -> float:
            """Reaction rate constant（ArrheniusEquation）"""
            return self.A * np.exp(-self.Ea / (self.R * T)) * C
    
        def cstr_model(self, state: np.ndarray, t: float,
                       F: float, Q: float) -> np.ndarray:
            """CSTRReactor 微min方程Equation
    
            Args:
                state: [Concentration C, Temperature T]
                t: WhenBetween
                F: Flow rate [m³/min]
                Q: Heating rate [kJ/min]
    
            Returns:
                [dC/dt, dT/dt]
            """
            C, T = state
            tau = self.V / F  # Residence time
    
            # Reaction rate
            r = self.reaction_rate(C, T)
    
            # Mass balance
            dC_dt = (self.C_in - C) / tau - r
    
            # Energy balance
            heat_reaction = (-self.delta_H) * r / (self.rho * self.Cp)
            heat_exchange = Q / (self.V * self.rho * self.Cp)
            dT_dt = (self.T_in - T) / tau + heat_reaction + heat_exchange
    
            return np.array([dC_dt, dT_dt])
    
        def simulate_step(self, state: np.ndarray,
                         F: float, Q: float) -> np.ndarray:
            """1Stepmin Simulation
    
            Args:
                state: [C, T]
                F: Flow rate
                Q: Heating rate
    
            Returns:
                Next Step State
            """
            t_span = [0, self.dt]
            result = odeint(self.cstr_model, state, t_span, args=(F, Q))
            return result[-1]
    
        def predict_trajectory(self, state0: np.ndarray,
                              u_sequence: np.ndarray) -> np.ndarray:
            """PredictionTrajectory Calculate
    
            Args:
                state0: InitialState [C, T]
                u_sequence: InputSequence (Nc x 2) [F, Q]
    
            Returns:
                PredictionStateSequence (Np x 2)
            """
            state = state0.copy()
            trajectory = np.zeros((self.Np, 2))
    
            for k in range(self.Np):
                if k < self.Nc:
                    F, Q = u_sequence[k]
                else:
                    F, Q = u_sequence[-1]
    
                state = self.simulate_step(state, F, Q)
                trajectory[k] = state
    
            return trajectory
    
        def mpc_objective(self, u_flat: np.ndarray, state0: np.ndarray,
                         setpoint: np.ndarray) -> float:
            """NonlinearMPCObjective function
    
            Args:
                u_flat: 平坦されたInputSequence
                state0: Current State
                setpoint: Setpoint [C_sp, T_sp]
    
            Returns:
                Objective function value
            """
            u_sequence = u_flat.reshape(self.Nc, 2)
    
            # PredictionTrajectory
            trajectory = self.predict_trajectory(state0, u_sequence)
    
            # Tracking error（Weight付き二乗和）
            Q = np.diag([10.0, 5.0])  # Concentration and Temperature Weight
            tracking_cost = 0.0
            for k in range(self.Np):
                error = trajectory[k] - setpoint
                tracking_cost += error.T @ Q @ error
    
            # InputChange Penalty
            R = np.diag([100.0, 0.01])  # Flow rate and Heating rate Weight
            control_cost = 0.0
            for k in range(self.Nc):
                if k > 0:
                    du = u_sequence[k] - u_sequence[k-1]
                    control_cost += du.T @ R @ du
    
            return tracking_cost + control_cost
    
        def solve(self, state0: np.ndarray, setpoint: np.ndarray,
                 u_prev: np.ndarray) -> np.ndarray:
            """NonlinearMPCOptimization
    
            Args:
                state0: Current State
                setpoint: Setpoint
                u_prev: FrontStep Input
    
            Returns:
                OptimalInput
            """
            # Initial guess（Front回 Input Hold）
            u0_flat = np.tile(u_prev, self.Nc)
    
            # Boundary constraints
            bounds = []
            for _ in range(self.Nc):
                bounds.append((self.F_min, self.F_max))
                bounds.append((self.Q_min, self.Q_max))
    
            # Optimization
            result = minimize(
                lambda u: self.mpc_objective(u, state0, setpoint),
                u0_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-3, 'maxiter': 50}
            )
    
            u_opt = result.x.reshape(self.Nc, 2)
            return u_opt[0]
    
    
    # ===================================
    # Simulation Execution
    # ===================================
    if __name__ == "__main__":
        mpc = NonlinearCSTRMPC(prediction_horizon=20, control_horizon=10)
    
        # InitialState
        state = np.array([2.0, 350.0])  # [C=2 kmol/m³, T=350 K]
        u = np.array([0.1, 0.0])  # [F=0.1 m³/min, Q=0 kJ/min]
    
        # Setpoint
        C_setpoint = 5.0  # kmol/m³
        T_setpoint = 360.0  # K
        setpoint = np.array([C_setpoint, T_setpoint])
    
        # Simulation
        T_sim = 100  # Number of steps
        state_history = [state.copy()]
        u_history = [u.copy()]
    
        print("="*70)
        print("MPC for nonlinear CSTR reactorControl")
        print("="*70)
        print(f"Setpoint: C={C_setpoint} kmol/m³, T={T_setpoint} K\n")
    
        for k in range(T_sim):
            # OutsideDisturbance（Inlet concentration Variation）
            if k == 40:
                mpc.C_in = 12.0  # Concentration Increase
                print(f"Time {k*mpc.dt:.1f}min: OutsideDisturbance発生（Inlet concentration 10→12 kmol/m³）\n")
    
            # MPC OptimalInput Calculate
            u = mpc.solve(state, setpoint, u)
    
            # ProcessUpdate
            state = mpc.simulate_step(state, u[0], u[1])
    
            # Add noise
            state += np.random.randn(2) * [0.05, 0.5]
    
            # Record
            state_history.append(state.copy())
            u_history.append(u.copy())
    
            if k % 20 == 0:
                print(f"Time {k*mpc.dt:5.1f}min: C={state[0]:.2f} kmol/m³, T={state[1]:.1f} K, "
                      f"F={u[0]:.3f} m³/min, Q={u[1]:+6.1f} kJ/min")
    
        state_history = np.array(state_history)
        u_history = np.array(u_history)
    
        print("\n" + "="*70)
        print("Control Performance")
        print("="*70)
    
        # Steady-stateState  Evaluation（Last 20Step）
        C_error = np.abs(state_history[-20:, 0] - C_setpoint).mean()
        T_error = np.abs(state_history[-20:, 1] - T_setpoint).mean()
    
        print(f"ConcentrationError（Steady-state）: {C_error:.3f} kmol/m³")
        print(f"TemperatureError（Steady-state）: {T_error:.3f} K")
        print(f"\n✓ NonlinearProcess 対して High精度なControl Achieve")
        print(f"✓ Rapid rejection of disturbances")
    

## 3.6 Control Using Deep Reinforcement Learning

Reinforcement learning learns optimal control policies through trial and error. We implement trajectory optimization for a batch reactor using Deep Q-Network (DQN).

Example 5: DQN よるバッチProcessControl - バッチReactorTrajectoryOptimization
    
    
    """
    ===================================
    Example 5: DQN よるバッチProcessControl
    ===================================
    
    Deep Q-Network（DQN） UsingバッチReactor TemperatureTrajectoryOptimization。
    Target: Product純度 Maximumしな 、バッチWhenBetween Minimization。
    
    強学習 定Equation:
    - State: [ConcentrationA, ConcentrationB, Temperature, 経過WhenBetween]
    - 行動: Temperature 増減（離散）
    - 報酬: ProductYield - WhenBetweenPenalty
    """
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random
    from typing import Tuple, List
    
    
    class BatchReactorEnv:
        """Batch reactor environment"""
    
        def __init__(self):
            self.dt = 1.0  # Time step [min]
            self.max_time = 120.0  # Maximum batch time [min]
    
            # Reaction rateParameters
            self.A1 = 1.0e8  # A→B（Desired reaction）
            self.E1 = 70000.0
            self.A2 = 5.0e7  # B→C（Side reaction）
            self.E2 = 65000.0
            self.R = 8.314
    
            # TemperatureConstraint
            self.T_min, self.T_max = 320.0, 380.0  # [K]
    
            self.reset()
    
        def reset(self) -> np.ndarray:
            """Reset environment"""
            self.C_A = 10.0  # InitialConcentrationA [mol/L]
            self.C_B = 0.0   # InitialConcentrationB [mol/L]
            self.C_C = 0.0   # InitialConcentrationC [mol/L]
            self.T = 340.0   # InitialTemperature [K]
            self.time = 0.0
    
            return self._get_state()
    
        def _get_state(self) -> np.ndarray:
            """Stateベクト 取得"""
            return np.array([
                self.C_A / 10.0,   # Normalization
                self.C_B / 10.0,
                (self.T - 350.0) / 30.0,
                self.time / self.max_time
            ], dtype=np.float32)
    
        def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
            """Execute 1 step
    
            Args:
                action: 0=cooling, 1=hold, 2=heating
    
            Returns:
                (next_state, reward, done)
            """
            # TemperatureChange
            delta_T = [-2.0, 0.0, 2.0][action]
            self.T = np.clip(self.T + delta_T, self.T_min, self.T_max)
    
            # Reaction rate constant
            k1 = self.A1 * np.exp(-self.E1 / (self.R * self.T))
            k2 = self.A2 * np.exp(-self.E2 / (self.R * self.T))
    
            # ConcentrationUpdate（1NextReaction）
            dC_A = -k1 * self.C_A * self.dt
            dC_B = (k1 * self.C_A - k2 * self.C_B) * self.dt
            dC_C = k2 * self.C_B * self.dt
    
            self.C_A += dC_A
            self.C_B += dC_B
            self.C_C += dC_C
    
            self.time += self.dt
    
            # Reward calculation
            # ProductBConcentration Maximum（Target: 8 mol/L or more）
            product_reward = self.C_B
    
            # Penalty for byproduct C
            byproduct_penalty = -0.5 * self.C_C
    
            # Time penalty (want to finish quickly)
            time_penalty = -0.01 * self.time
    
            reward = product_reward + byproduct_penalty + time_penalty
    
            # Termination condition
            done = (self.time >= self.max_time) or (self.C_A < 0.5)
    
            # Bonus: When target achieved
            if done and self.C_B >= 7.5:
                reward += 50.0
    
            return self._get_state(), reward, done
    
    
    class DQN(nn.Module):
        """Deep Q-Network"""
    
        def __init__(self, state_dim: int, action_dim: int):
            super(DQN, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
    
        def forward(self, x):
            return self.fc(x)
    
    
    class DQNAgent:
        """DQN Agent"""
    
        def __init__(self, state_dim: int, action_dim: int):
            self.state_dim = state_dim
            self.action_dim = action_dim
    
            self.q_network = DQN(state_dim, action_dim)
            self.target_network = DQN(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
            self.memory = deque(maxlen=10000)
            self.batch_size = 64
            self.gamma = 0.99
    
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995
    
        def select_action(self, state: np.ndarray) -> int:
            """ε-greedy action selection"""
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
    
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_t)
                return q_values.argmax().item()
    
        def store_transition(self, state, action, reward, next_state, done):
            """Store experience"""
            self.memory.append((state, action, reward, next_state, done))
    
        def train(self):
            """Train network"""
            if len(self.memory) < self.batch_size:
                return
    
            # Batch sampling
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
    
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
    
            # Current Q value
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
    
            # Target Q value
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
    
            # Loss calculation and update
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # ε decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
        def update_target_network(self):
            """Update target network"""
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    
    # ===================================
    # Training Execution
    # ===================================
    if __name__ == "__main__":
        env = BatchReactorEnv()
        agent = DQNAgent(state_dim=4, action_dim=3)
    
        num_episodes = 200
        rewards_history = []
    
        print("="*70)
        print("Learning Batch Reactor Control with DQN")
        print("="*70)
    
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
    
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
    
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
    
                state = next_state
                episode_reward += reward
    
            rewards_history.append(episode_reward)
    
            # ターゲットネットワークUpdate
            if episode % 10 == 0:
                agent.update_target_network()
    
            if episode % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:])
                print(f"Episode {episode:3d}: Average reward={avg_reward:6.2f}, "
                      f"ε={agent.epsilon:.3f}, Final C_B={env.C_B:.2f} mol/L")
    
        print("\n" + "="*70)
        print("Training Complete - Test Execution")
        print("="*70)
    
        # Test execution (greedy policy)
        agent.epsilon = 0.0
        state = env.reset()
        done = False
    
        trajectory = []
        while not done:
            action = agent.select_action(state)
            trajectory.append({
                'time': env.time,
                'C_A': env.C_A,
                'C_B': env.C_B,
                'C_C': env.C_C,
                'T': env.T,
                'action': ['Cooling', 'Hold', 'Heating'][action]
            })
            next_state, reward, done = env.step(action)
            state = next_state
    
        print(f"\nBatch completion time: {env.time:.1f} min")
        print(f"最終ProductConcentration C_B: {env.C_B:.2f} mol/L")
        print(f"Byproduct C_C: {env.C_C:.2f} mol/L")
        print(f"Selectivity: {env.C_B / (env.C_B + env.C_C):.1%}")
    
        print("\nOptimalTemperatureTrajectory（First 10Step）:")
        for i in range(min(10, len(trajectory))):
            t = trajectory[i]
            print(f"  {t['time']:5.1f}min: T={t['T']:.1f}K, C_B={t['C_B']:.2f}, Operation={t['action']}")
    
        print("\n✓ Learn optimal trajectory balancing batch time and product purity with DQN")
    

## 3.7 Application of Reinforcement Learning to Continuous Processes

For processes with continuous action spaces, Proximal Policy Optimization (PPO) is effective. We show an application example for CSTR control.

Example 6: PPO よる連続ProcessControl - CSTRTemperature・Flow rateControl
    
    
    """
    ===================================
    Example 6: PPO よる連続ProcessControl
    ===================================
    
    Proximal Policy Optimization（PPO） UsingContinuous Stirred Tank Reactor Control。
    連続Value 行動空Between（Temperature、Flow rate） 扱う、Actor-Criticアーキテクチャ 使用。
    
    Target: ProductConcentration Setpoint 維持しな 、EnergyーConsumption Minimization
    """
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    from typing import Tuple
    
    
    class CSTREnv:
        """Continuous Stirred Tank Reactor環境"""
    
        def __init__(self):
            self.dt = 0.5  # Sampling time [min]
            self.V = 1.0   # Reactor volume [m³]
    
            # Reaction rateParameters
            self.A = 1.0e9
            self.Ea = 72000.0
            self.R = 8.314
    
            # Setpoint
            self.C_target = 3.0  # TargetProductConcentration [mol/L]
    
            self.reset()
    
        def reset(self) -> np.ndarray:
            """Reset environment"""
            self.C = 2.0 + np.random.randn() * 0.5  # Concentration [mol/L]
            self.T = 350.0 + np.random.randn() * 5.0  # Temperature [K]
            self.C_in = 8.0  # Inlet concentration [mol/L]
    
            return self._get_state()
    
        def _get_state(self) -> np.ndarray:
            """Stateベクト"""
            return np.array([
                (self.C - self.C_target) / self.C_target,  # ConcentrationDeviation（Normalization）
                (self.T - 350.0) / 30.0  # TemperatureDeviation（Normalization）
            ], dtype=np.float32)
    
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute 1 step
    
            Args:
                action: [Flow rateChange率, TemperatureChange]（-1〜1 Normalization）
    
            Returns:
                (next_state, reward, done)
            """
            # Convert actions to actual physical quantities
            F = 0.1 + 0.05 * action[0]  # Flow rate 0.05〜0.15 [m³/min]
            delta_T = 5.0 * action[1]    # TemperatureChange -5〜+5 [K]
    
            self.T = np.clip(self.T + delta_T, 320.0, 380.0)
    
            # Reaction rate constant
            k = self.A * np.exp(-self.Ea / (self.R * self.T))
    
            # ConcentrationUpdate（CSTR物質収支）
            tau = self.V / F
            dC = ((self.C_in - self.C) / tau - k * self.C) * self.dt
            self.C += dC
    
            # Noise (disturbance)
            self.C += np.random.randn() * 0.05
            self.T += np.random.randn() * 1.0
    
            # Reward calculation
            # 1. ConcentrationTracking（主Objective）
            error = abs(self.C - self.C_target)
            tracking_reward = -10.0 * error
    
            # 2. EnergyーPenalty（HeatingCost）
            heating_cost = -0.01 * abs(delta_T)
    
            # 3. Flow rateChangePenalty（スムーズなOperation）
            flow_penalty = -0.1 * abs(action[0])
    
            reward = tracking_reward + heating_cost + flow_penalty
    
            # Bonus: Within target range (±0.2 mol/L)
            if error < 0.2:
                reward += 5.0
    
            done = False  # 連続Processな  終了なし
    
            return self._get_state(), reward, done
    
    
    class ActorCritic(nn.Module):
        """Actor-Critic network"""
    
        def __init__(self, state_dim: int, action_dim: int):
            super(ActorCritic, self).__init__()
    
            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
    
            # Actor (policy)
            self.actor_mean = nn.Linear(64, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
    
            # Critic（ValueFunction）
            self.critic = nn.Linear(64, 1)
    
        def forward(self, state):
            """Forward propagation"""
            shared_features = self.shared(state)
    
            # Actor: Positive規min布 Parameters
            action_mean = torch.tanh(self.actor_mean(shared_features))
            action_std = torch.exp(self.actor_log_std)
    
            # Critic: StateValue
            value = self.critic(shared_features)
    
            return action_mean, action_std, value
    
        def get_action(self, state):
            """Sample action"""
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
    
            return action, log_prob, value
    
    
    class PPOAgent:
        """PPO Agent"""
    
        def __init__(self, state_dim: int, action_dim: int):
            self.ac = ActorCritic(state_dim, action_dim)
            self.optimizer = optim.Adam(self.ac.parameters(), lr=3e-4)
    
            self.gamma = 0.99
            self.lam = 0.95  # GAE lambda
            self.clip_epsilon = 0.2
            self.epochs = 10
    
        def compute_gae(self, rewards, values, dones):
            """Generalized Advantage Estimation"""
            advantages = []
            gae = 0
    
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
    
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.lam * gae
                advantages.insert(0, gae)
    
            return advantages
    
        def update(self, states, actions, old_log_probs, returns, advantages):
            """PPO update"""
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(old_log_probs)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
    
            # Normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            for _ in range(self.epochs):
                action_mean, action_std, values = self.ac(states)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(-1)
    
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
    
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
    
                # Value loss
                critic_loss = nn.MSELoss()(values.squeeze(), returns)
    
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    
    # ===================================
    # Training Execution
    # ===================================
    if __name__ == "__main__":
        env = CSTREnv()
        agent = PPOAgent(state_dim=2, action_dim=2)
    
        num_episodes = 300
        steps_per_episode = 200
    
        print("="*70)
        print("Learning CSTR Continuous Control with PPO")
        print("="*70)
    
        for episode in range(num_episodes):
            state = env.reset()
    
            states, actions, log_probs, rewards, values = [], [], [], [], []
    
            for step in range(steps_per_episode):
                state_t = torch.FloatTensor(state)
                action, log_prob, value = agent.ac.get_action(state_t.unsqueeze(0))
    
                action_np = action.squeeze().detach().numpy()
                next_state, reward, done = env.step(action_np)
    
                states.append(state)
                actions.append(action_np)
                log_probs.append(log_prob.item())
                rewards.append(reward)
                values.append(value.item())
    
                state = next_state
    
            # GAE calculation
            advantages = agent.compute_gae(rewards, values, [False] * len(rewards))
            returns = [adv + val for adv, val in zip(advantages, values)]
    
            # PPO update
            agent.update(states, actions, log_probs, returns, advantages)
    
            if episode % 30 == 0:
                avg_reward = np.mean(rewards)
                final_error = abs(env.C - env.C_target)
                print(f"Episode {episode:3d}: Average reward={avg_reward:6.2f}, "
                      f"最終Error={final_error:.3f} mol/L")
    
        print("\n" + "="*70)
        print("Training Complete - Test Execution")
        print("="*70)
    
        # テストExecute
        state = env.reset()
        trajectory = []
    
        for step in range(100):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                action_mean, _, _ = agent.ac(state_t.unsqueeze(0))
                action = action_mean.squeeze().numpy()
    
            trajectory.append({
                'step': step,
                'C': env.C,
                'T': env.T,
                'action_flow': action[0],
                'action_temp': action[1]
            })
    
            next_state, reward, done = env.step(action)
            state = next_state
    
        # Performance evaluation
        concentrations = [t['C'] for t in trajectory]
        mean_error = np.mean([abs(c - env.C_target) for c in concentrations])
        std_error = np.std([abs(c - env.C_target) for c in concentrations])
    
        print(f"\nAverageTracking error: {mean_error:.3f} mol/L")
        print(f"Error standard deviation: {std_error:.3f} mol/L")
    
        print("\nControl trajectory (first 10 steps):")
        for i in range(10):
            t = trajectory[i]
            print(f"  Step {t['step']:3d}: C={t['C']:.2f} mol/L, T={t['T']:.1f}K, "
                  f"Flow action={t['action_flow']:+.2f}, Temp action={t['action_temp']:+.2f}")
    
        print("\n✓ PPO  連続ValueControl 学習し、SetpointTracking Achieve")
    

## 3.8 Multi-Objective Optimization

In chemical plants, multiple objectives such as yield and energy consumption are in trade-off relationships. We implement multi-objective optimization using NSGA-II.

Example 7: Multi-Objective Optimization (NSGA-II) - Yield and Energy Trade-off
    
    
    """
    ===================================
    Example 7: Multi-objectiveOptimization（NSGA-II）
    ===================================
    
    NSGA-II（Non-dominated Sorting Genetic Algorithm II） Using
    学Process Multi-objectiveOptimization。
    
    Objective:
    1. ProductYield Maximum
    2. EnergyーConsumption Minimization
    
    これ 互い Trade-off関係 あり、Pareto optimal solutionsSet 求める。
    
    pymoo Actual  Installー Necessary  、Here  基本な遺伝アゴリズム 代替実装s。
    """
    
    import numpy as np
    from typing import List, Tuple
    import pandas as pd
    
    
    class MultiObjectiveOptimizer:
        """Multi-objectiveOptimization器（NSGA-II風）"""
    
        def __init__(self, population_size: int = 50, generations: int = 100):
            self.pop_size = population_size
            self.n_gen = generations
    
            # Decision variable ranges
            # x = [Temperature [K], Pressure [bar], Flow rate [m³/h], Catalyst amount [kg]]
            self.bounds_lower = np.array([350.0, 10.0, 1.0, 50.0])
            self.bounds_upper = np.array([420.0, 50.0, 5.0, 200.0])
    
        def process_model(self, x: np.ndarray) -> Tuple[float, float]:
            """ProcessModel：Yield and Energyー Calculate
    
            Args:
                x: [Temperature, Pressure, Flow rate, Catalyst amount]
    
            Returns:
                (Yield, EnergyーConsumption)
            """
            T, P, F, Cat = x
    
            # Yield model (simplified)
            # High temperature・High圧・多Catalyst Yield向Up、 however 飽和あり
            T_factor = 1.0 / (1.0 + np.exp(-(T - 380.0) / 10.0))
            P_factor = np.log(P / 10.0) / np.log(5.0)
            Cat_factor = np.sqrt(Cat / 50.0)
    
            yield_fraction = 0.5 + 0.4 * T_factor + 0.2 * P_factor + 0.15 * Cat_factor
    
            # Flow rate Ratio例した生産量
            productivity = F * yield_fraction  # [m³/h * fraction]
    
            # Energy consumption model
            # HeatingEnergyー（Temperature Ratio例）
            heating_energy = F * 1000.0 * 4.18 * (T - 300.0) / 3600.0  # [kW]
    
            # 圧縮Energyー（Pressure Nonlinear）
            compression_energy = 10.0 * F * (P / 10.0)**1.2  # [kW]
    
            total_energy = heating_energy + compression_energy
    
            return productivity, total_energy
    
        def objectives(self, x: np.ndarray) -> np.ndarray:
            """Objective function（MinimizationProblem 統一）
    
            Returns:
                [-Yield（Maximum→Minimization）, EnergyーConsumption（Minimization）]
            """
            productivity, energy = self.process_model(x)
            return np.array([-productivity, energy])
    
        def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
            """Determine if obj1 dominates obj2"""
            return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
        def non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
            """Non-dominated sorting
    
            Args:
                objectives: (pop_size x 2) Objective functionValue
    
            Returns:
                Index list for each front
            """
            pop_size = len(objectives)
            domination_count = np.zeros(pop_size, dtype=int)
            dominated_solutions = [[] for _ in range(pop_size)]
    
            fronts = [[]]
    
            for i in range(pop_size):
                for j in range(i + 1, pop_size):
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                        domination_count[j] += 1
                    elif self.dominates(objectives[j], objectives[i]):
                        dominated_solutions[j].append(i)
                        domination_count[i] += 1
    
            for i in range(pop_size):
                if domination_count[i] == 0:
                    fronts[0].append(i)
    
            current_front = 0
            while len(fronts[current_front]) > 0:
                next_front = []
                for i in fronts[current_front]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front += 1
                fronts.append(next_front)
    
            return fronts[:-1]  # Last 空リスト 除く
    
        def crowding_distance(self, objectives: np.ndarray, front: List[int]) -> np.ndarray:
            """Calculate crowding distance"""
            n = len(front)
            if n <= 2:
                return np.full(n, np.inf)
    
            distances = np.zeros(n)
    
            for m in range(objectives.shape[1]):  # EachObjective ついて
                sorted_idx = np.argsort(objectives[front, m])
    
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
    
                obj_range = objectives[front[sorted_idx[-1]], m] - objectives[front[sorted_idx[0]], m]
                if obj_range == 0:
                    continue
    
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        (objectives[front[sorted_idx[i + 1]], m] -
                         objectives[front[sorted_idx[i - 1]], m]) / obj_range
                    )
    
            return distances
    
        def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
            """NSGA-IIOptimization Execute
    
            Returns:
                (パレート解Set, Objective functionValue)
            """
            # Generate initial population
            population = np.random.uniform(
                self.bounds_lower,
                self.bounds_upper,
                (self.pop_size, len(self.bounds_lower))
            )
    
            for generation in range(self.n_gen):
                # Objective functionEvaluation
                objectives = np.array([self.objectives(ind) for ind in population])
    
                # Non-dominated sorting
                fronts = self.non_dominated_sort(objectives)
    
                # Select next generation
                next_population = []
                for front in fronts:
                    if len(next_population) + len(front) <= self.pop_size:
                        next_population.extend(front)
                    else:
                        # Sort by crowding distance
                        distances = self.crowding_distance(objectives, front)
                        sorted_idx = np.argsort(distances)[::-1]
                        remaining = self.pop_size - len(next_population)
                        next_population.extend([front[i] for i in sorted_idx[:remaining]])
                        break
    
                # Crossover and mutation
                selected = population[next_population]
                offspring = []
    
                for i in range(0, len(selected) - 1, 2):
                    # SBX crossover (simplified)
                    alpha = np.random.rand(len(self.bounds_lower))
                    child1 = alpha * selected[i] + (1 - alpha) * selected[i + 1]
                    child2 = (1 - alpha) * selected[i] + alpha * selected[i + 1]
    
                    # Polynomial mutation (simplified)
                    if np.random.rand() < 0.1:
                        child1 += np.random.randn(len(self.bounds_lower)) * 0.1 * (self.bounds_upper - self.bounds_lower)
                    if np.random.rand() < 0.1:
                        child2 += np.random.randn(len(self.bounds_lower)) * 0.1 * (self.bounds_upper - self.bounds_lower)
    
                    # Boundary constraints
                    child1 = np.clip(child1, self.bounds_lower, self.bounds_upper)
                    child2 = np.clip(child2, self.bounds_lower, self.bounds_upper)
    
                    offspring.extend([child1, child2])
    
                population = np.array(offspring[:self.pop_size])
    
                if generation % 20 == 0:
                    print(f"Generation {generation}: Pareto front size={len(fronts[0])}")
    
            # Final evaluation
            objectives = np.array([self.objectives(ind) for ind in population])
            fronts = self.non_dominated_sort(objectives)
            pareto_front = fronts[0]
    
            return population[pareto_front], objectives[pareto_front]
    
    
    # ===================================
    # Execution Example
    # ===================================
    if __name__ == "__main__":
        optimizer = MultiObjectiveOptimizer(population_size=50, generations=100)
    
        print("="*70)
        print("Multi-objectiveOptimization（NSGA-II）")
        print("="*70)
        print("Objective 1: Maximize product yield")
        print("Objective2: EnergyーConsumption Minimization\n")
    
        # OptimizationExecute
        pareto_solutions, pareto_objectives = optimizer.optimize()
    
        print("\n" + "="*70)
        print(f"Pareto optimal solutions: {len(pareto_solutions)} solutions")
        print("="*70)
    
        # Organize results
        results = []
        for i, (sol, obj) in enumerate(zip(pareto_solutions, pareto_objectives)):
            T, P, F, Cat = sol
            productivity = -obj[0]  # MinimizationProblem from 戻
            energy = obj[1]
    
            results.append({
                'Solution No.': i + 1,
                'Temperature [K]': T,
                'Pressure [bar]': P,
                'Flow rate [m³/h]': F,
                'Catalyst [kg]': Cat,
                'Yield Productivity [m³/h]': productivity,
                'Energy [kW]': energy,
                'Energy Intensity [kW/(m³/h)]': energy / productivity
            })
    
        df = pd.DataFrame(results)
    
        # 代表な解 表示（YieldFocus on、バランス、省エネFocus on）
        print("\n【Representative Pareto Solutions】")
        print("\n1. Yield-focused (high energy consumption):")
        idx_max_yield = df['Yield Productivity [m³/h]'].idxmax()
        print(df.loc[idx_max_yield].to_string())
    
        print("\n2. Balanced:")
        df['バランススコア'] = (df['Yield Productivity [m³/h]'].rank() + (1 / df['Energy [kW]']).rank()) / 2
        idx_balanced = df['バランススコア'].idxmax()
        print(df.loc[idx_balanced].to_string())
    
        print("\n3. Energy-saving focused (lower yield):")
        idx_min_energy = df['Energy [kW]'].idxmin()
        print(df.loc[idx_min_energy].to_string())
    
        print("\n" + "="*70)
        print("Trade-offmin析")
        print("="*70)
    
        # Yield10%向Up  Energy cost増加
        sorted_df = df.sort_values('Yield Productivity [m³/h]')
        if len(sorted_df) > 1:
            yield_range = sorted_df['Yield Productivity [m³/h]'].max() - sorted_df['Yield Productivity [m³/h]'].min()
            energy_range = sorted_df['Energy [kW]'].max() - sorted_df['Energy [kW]'].min()
    
            print(f"Energy increase for 10% yield improvement: approximately{energy_range / yield_range * 0.1 * sorted_df['Yield Productivity [m³/h]'].mean():.1f} kW")
    
        print("\n✓ Pareto front enables decision-makers to select yield-energy trade-offs")
    

## 3.9 Integrated APC+Optimization System

Finally, we show an implementation example of a practical plant control system integrating RTO and APC layers.

Example 8: Integrated APC+Optimization System - Hierarchical Plant Control
    
    
    """
    ===================================
    Example 8: 統合APC+OptimizationSystem
    ===================================
    
    Real-timeOptimization（RTO）Layer and Advanced Process Control（APC）Layer 
    統合した階Layer型プラントControlSystem。
    
    階Layer構造:
    - RTOLayer（Up位）: Economic Optimization  Optimal operating conditions 決定
    - APCLayer（Down位）: MPC RTO Setpoint High精度 Tracking
    
    Chemical Plant（Distillation column+Reactor） to  適用例
    """
    
    import numpy as np
    from scipy.optimize import minimize
    from typing import Dict, Tuple
    import pandas as pd
    
    
    class RTOLayer:
        """Real-Time Optimization Layer"""
    
        def __init__(self):
            # Economic parameters
            self.product_price = 120.0  # $/ton
            self.feed_cost = 50.0  # $/ton
            self.steam_cost = 25.0  # $/ton
            self.power_cost = 0.10  # $/kWh
    
        def steady_state_model(self, x: np.ndarray) -> Dict:
            """Steady-stateStateProcessModel
    
            Args:
                x: [Reaction temperature, 蒸留Reflux flowRatio]
    
            Returns:
                ProcessOutput Dictionary
            """
            T_reactor, reflux_ratio = x
    
            # ReactionYield（Temperature依存）
            yield_base = 0.75
            temp_effect = 0.002 * (T_reactor - 370.0)
            yield_fraction = yield_base + temp_effect
    
            # Product純度（Reflux flowRatio依存）
            purity_base = 0.90
            reflux_effect = 0.15 * (1.0 - np.exp(-0.5 * (reflux_ratio - 2.0)))
            purity = min(0.99, purity_base + reflux_effect)
    
            # Utility consumption
            reactor_heat = 50.0 + 0.5 * (T_reactor - 350.0)**2  # kW
            steam_consumption = 2.0 + 0.8 * reflux_ratio  # ton/h
            power = 30.0 + 5.0 * reflux_ratio  # kW
    
            return {
                'yield': yield_fraction,
                'purity': purity,
                'reactor_heat': reactor_heat,
                'steam': steam_consumption,
                'power': power
            }
    
        def economic_objective(self, x: np.ndarray, feed_rate: float) -> float:
            """EconomicObjective function：Profit Negative value
    
            Args:
                x: [Reaction temperature, Reflux flowRatio]
                feed_rate: FeedFlow rate [ton/h]
    
            Returns:
                -Profit [$/h]
            """
            outputs = self.steady_state_model(x)
    
            # Product Production
            product_rate = feed_rate * outputs['yield'] * outputs['purity']
    
            # Revenue
            revenue = product_rate * self.product_price
    
            # Cost
            feed_cost = feed_rate * self.feed_cost
            steam_cost = outputs['steam'] * self.steam_cost
            power_cost = outputs['power'] * self.power_cost
    
            profit = revenue - feed_cost - steam_cost - power_cost
    
            return -profit  # Minimization Negative value
    
        def optimize(self, feed_rate: float) -> Dict:
            """Execute RTO
    
            Args:
                feed_rate: Current FeedFlow rate [ton/h]
    
            Returns:
                Optimal operating conditions
            """
            # Initial guess
            x0 = np.array([370.0, 3.0])
    
            # Boundary constraints
            bounds = [
                (350.0, 390.0),  # Reaction temperature [K]
                (2.0, 5.0)       # Reflux flowRatio
            ]
    
            # Constraint: Product purity ≥95%
            def purity_constraint(x):
                outputs = self.steady_state_model(x)
                return outputs['purity'] - 0.95
    
            from scipy.optimize import NonlinearConstraint
            nlc = NonlinearConstraint(purity_constraint, 0, np.inf)
    
            # Optimization
            result = minimize(
                lambda x: self.economic_objective(x, feed_rate),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[nlc]
            )
    
            T_opt, reflux_opt = result.x
            outputs = self.steady_state_model(result.x)
    
            return {
                'T_reactor_sp': T_opt,
                'reflux_ratio_sp': reflux_opt,
                'predicted_yield': outputs['yield'],
                'predicted_purity': outputs['purity'],
                'predicted_profit': -result.fun
            }
    
    
    class APCLayer:
        """Advanced Control Layer（MPC）"""
    
        def __init__(self):
            # ProcessModel（線形近似）
            # State: [Reaction temperatureDeviation, Reflux flowRatioDeviation]
            self.A = np.array([
                [0.90, 0.05],
                [0.00, 0.85]
            ])
            self.B = np.array([
                [0.8, 0.0],
                [0.0, 0.6]
            ])
    
            # Control horizon
            self.Np = 15
            self.Nc = 8
    
        def mpc_control(self, current_state: np.ndarray,
                        setpoint: np.ndarray) -> np.ndarray:
            """MPC control
    
            Args:
                current_state: Current StateDeviation [TDeviation, Reflux flowRatioDeviation]
                setpoint: SetpointDeviation（RTO from  指令）
    
            Returns:
                OptimalManipulated variable
            """
            # Simplified MPC (actual implementation requires more detail)
            # Here substituted with proportional control
            Kp = np.array([2.0, 1.5])
            u = Kp * (setpoint - current_state)
    
            # Manipulated variableConstraint
            u = np.clip(u, [-5.0, -0.5], [5.0, 0.5])
    
            return u
    
    
    class IntegratedControlSystem:
        """Integrated control system"""
    
        def __init__(self):
            self.rto = RTOLayer()
            self.apc = APCLayer()
    
            # RTO execution interval (longer than APC)
            self.rto_interval = 30  # 30 times APC cycle
    
            # Current State
            self.T_reactor = 365.0
            self.reflux_ratio = 3.2
    
        def run_rto(self, feed_rate: float) -> Dict:
            """Execute RTO layer"""
            print("\n" + "="*70)
            print("RTOLayer: EconomicOptimization Execute")
            print("="*70)
    
            result = self.rto.optimize(feed_rate)
    
            print(f"  OptimalReaction temperature: {result['T_reactor_sp']:.1f} K")
            print(f"  OptimalReflux flowRatio: {result['reflux_ratio_sp']:.2f}")
            print(f"  PredictionYield: {result['predicted_yield']:.1%}")
            print(f"  Prediction純度: {result['predicted_purity']:.1%}")
            print(f"  PredictionProfit: ${result['predicted_profit']:.2f}/h")
    
            return result
    
        def run_apc(self, rto_setpoint: Dict) -> np.ndarray:
            """Execute APC layer"""
            # Current deviation
            current_state = np.array([
                self.T_reactor - rto_setpoint['T_reactor_sp'],
                self.reflux_ratio - rto_setpoint['reflux_ratio_sp']
            ])
    
            # Target deviation (zero)
            setpoint = np.array([0.0, 0.0])
    
            # MPC control
            u = self.apc.mpc_control(current_state, setpoint)
    
            return u
    
        def simulate(self, feed_rate: float, simulation_steps: int = 100):
            """Simulate integrated system
    
            Args:
                feed_rate: FeedFlow rate [ton/h]
                simulation_steps: SimulationNumber of steps
            """
            print("="*70)
            print("統合APC+OptimizationSystem Simulation")
            print("="*70)
    
            # Initial RTO execution
            rto_result = self.run_rto(feed_rate)
    
            history = []
    
            for step in range(simulation_steps):
                # RTO update (periodic)
                if step % self.rto_interval == 0 and step > 0:
                    rto_result = self.run_rto(feed_rate)
    
                # APC execution (every step)
                u = self.run_apc(rto_result)
    
                # ProcessUpdate（簡略）
                self.T_reactor += u[0] * 0.5 + np.random.randn() * 0.5
                self.reflux_ratio += u[1] * 0.3 + np.random.randn() * 0.05
    
                # Physical constraints
                self.T_reactor = np.clip(self.T_reactor, 350.0, 390.0)
                self.reflux_ratio = np.clip(self.reflux_ratio, 2.0, 5.0)
    
                # Record
                outputs = self.rto.steady_state_model(
                    [self.T_reactor, self.reflux_ratio]
                )
    
                history.append({
                    'step': step,
                    'T_reactor': self.T_reactor,
                    'T_setpoint': rto_result['T_reactor_sp'],
                    'reflux': self.reflux_ratio,
                    'reflux_setpoint': rto_result['reflux_ratio_sp'],
                    'yield': outputs['yield'],
                    'purity': outputs['purity']
                })
    
                if step % 20 == 0:
                    print(f"\nStep {step:3d}:")
                    print(f"  Reaction temperature: {self.T_reactor:.1f}K (SP: {rto_result['T_reactor_sp']:.1f}K)")
                    print(f"  Reflux flowRatio: {self.reflux_ratio:.2f} (SP: {rto_result['reflux_ratio_sp']:.2f})")
                    print(f"  Yield: {outputs['yield']:.1%}, 純度: {outputs['purity']:.1%}")
    
            return pd.DataFrame(history)
    
    
    # ===================================
    # Execution Example
    # ===================================
    if __name__ == "__main__":
        system = IntegratedControlSystem()
    
        # FeedFlow rate
        feed_rate = 10.0  # ton/h
    
        # Simulation Execution
        df = system.simulate(feed_rate, simulation_steps=100)
    
        print("\n" + "="*70)
        print("Control Performance Evaluation")
        print("="*70)
    
        # TrackingPerformance
        T_error = np.abs(df['T_reactor'] - df['T_setpoint']).mean()
        reflux_error = np.abs(df['reflux'] - df['reflux_setpoint']).mean()
    
        print(f"\nReaction temperature AverageTracking error: {T_error:.2f} K")
        print(f"Reflux flowRatio AverageTracking error: {reflux_error:.3f}")
    
        # ProcessPerformance
        avg_yield = df['yield'].mean()
        avg_purity = df['purity'].mean()
    
        print(f"\nAverage yield: {avg_yield:.1%}")
        print(f"Average purity: {avg_purity:.1%}")
    
        print("\n" + "="*70)
        print("System Features")
        print("="*70)
        print("✓ RTOLayer EconomicOptimal operating conditions 決定")
        print("✓ APCLayer（MPC） High精度なSetpointTracking Achieve")
        print("✓ 階Layer  CalculateNegative荷 min散、実WhenBetweenControl Achieve")
        print("✓ プラントAll体 Economics Maximumしな 品質 保証")
    

## Summary

本章 、Chemical PlantinReal-timeOptimization and Advanced Process Control 実装技術 学びました。主要なポイント or less 通り ：

### Review of Learning Content

  1. **OnlineOptimization（SciPy）** ：CSTROperationCondition EconomicTargetFunction Optimizationし、ProfitMaximum Achieve
  2. **Economic Optimization（Pyomo）** ：ProductValue and ーUtilityCost Trade-off 考慮した複雑なOptimizationProblem 定Equation
  3. **Basic MPC Implementation** ：Distillation column TemperatureControl おいて、未来Prediction and Constraints 考慮したOptimalOperation Calculate
  4. **Nonlinear MPC (CasADi)** ：Reactor Nonlinearダイナミクス 対応したHigh度なModelPredictionControl 実装
  5. **DQN Batch Control** ：深Layer強学習 バッチReactor OptimalTemperatureTrajectory 自律 学習
  6. **PPO Continuous Control** ：連続Value行動空Between 扱うPPO 、CSTRConcentrationControl Optimal方策 獲得
  7. **Multi-objectiveOptimization** ：NSGA-II Yield and Energyー Pareto optimal solutionsSet 探索
  8. **Integrated APC+RTO** ：Hierarchical Control Systems EconomicOptimization and High精度TrackingControl 両立

🎯 Application to Practice

**Examples of Implementation Effects:**

  * **Company A (Petroleum Refining)** ：FCC装置 to RTO+MPC導入 、YearBetweenRevenue 3.5%改善（約15億円/Year）
  * **Company B (Ethylene Plant)** ：Improved cracker yield by 1.2% and reduced energy consumption by 8% with nonlinear MPC
  * **Company C (Polymer Manufacturing)** ：Reduced lot time by 15% and halved product quality variation with reinforcement learning-based batch control
  * **Company D (Chemical Manufacturing)** ：Multi-objectiveOptimization 、Yield95%維持しな Energy cost12%削減

### Connection to Next Chapter

Next章 、プラントAll体 サプライチェーンOptimization and デジタツイン ついて学びま。需要Prediction、生産計画、在庫Optimization、さ Real-timeSimulation よるプラント運用支援 to 、実装レベ 習得s。

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
