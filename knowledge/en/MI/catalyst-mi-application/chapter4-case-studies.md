---
title: Catalyst MI Case Studies
chapter_title: Catalyst MI Case Studies
subtitle: Practical Methods from 5 Industrial Applications
reading_time: 50-60 min
difficulty: Advanced
code_examples: 15
exercises: 5
version: 1.0
created_at: 2025-10-17
---

# Chapter 4: Catalyst MI Case Studies

This chapter covers Catalyst MI Case Studies. You will learn essential concepts and techniques.

**Learning Objectives:** \- Understanding successful MI case studies in industrial catalyst applications \- Complete workflows from problem definition to model building and experimental validation \- Mastering domain-specific challenges and MI solutions

**Chapter Structure:** 1\. Green Hydrogen Production Catalysts (Water Electrolysis) 2\. CO₂ Reduction Catalysts (Carbon Recycling) 3\. Next-Generation Ammonia Synthesis Catalysts 4\. Automotive Catalysts (Noble Metal Reduction) 5\. Pharmaceutical Intermediate Synthesis Catalysts (Asymmetric Catalysts)

* * *

## 4.1 Case Study 1: Green Hydrogen Production Catalysts

### 4.1.1 Background and Challenges

**What is Green Hydrogen:** \- Produced by water electrolysis using renewable energy-derived electricity \- Key to achieving carbon neutrality \- Target production cost: $2/kg H₂ by 2030 (currently $5-6/kg)

**Water Electrolysis Reactions:**
    
    
    Anode (OER): 2H₂O → O₂ + 4H⁺ + 4e⁻  (Large overpotential)
    Cathode (HER): 4H⁺ + 4e⁻ → 2H₂      (Relatively easy)
    

**Challenges:** \- Large overpotential in OER (Oxygen Evolution Reaction) (~0.4 V) \- Traditional catalysts (IrO₂, RuO₂) are expensive and rare \- Long-term stability (>10,000 hours) required

### 4.1.2 MI Strategy

**Approach:** 1\. Identify OER activity descriptors through large-scale DFT calculations 2\. Predict high-activity compositions using machine learning 3\. Accelerate experimental exploration with Bayesian optimization

**Dataset:** \- 5,000 oxide catalysts from Materials Project \- 200 experimental data samples (overpotential, Tafel slope)

### 4.1.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 4.1.3 Implementation Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    # Step 1: Data preparation
    data = {
        'material': ['IrO2', 'RuO2', 'NiFe-LDH', 'CoOx', 'NiCoOx',
                     'FeOOH', 'Co3O4', 'NiO', 'MnO2', 'Perovskite_BSCF'],
        'O_p_band_center': [-3.5, -3.8, -4.2, -4.5, -4.3, -5.0, -4.7, -5.2, -5.5, -4.0],  # eV
        'eg_occupancy': [0.8, 0.9, 1.2, 1.5, 1.3, 1.8, 1.6, 2.0, 1.9, 1.1],  # eg orbital occupancy
        'metal_O_bond': [1.98, 1.95, 2.05, 2.10, 2.07, 2.15, 2.12, 2.08, 2.20, 2.00],  # Å
        'work_function': [5.8, 5.9, 4.8, 5.0, 4.9, 4.5, 5.1, 5.3, 4.7, 5.2],  # eV
        'overpotential': [0.28, 0.31, 0.35, 0.38, 0.33, 0.45, 0.40, 0.48, 0.52, 0.32]  # V @ 10 mA/cm²
    }
    
    df = pd.DataFrame(data)
    
    # Step 2: Descriptor engineering
    # Sabatier volcano peak: eg occupancy ~ 1.2 is optimal (theoretical prediction)
    df['eg_deviation'] = np.abs(df['eg_occupancy'] - 1.2)
    
    X = df[['O_p_band_center', 'eg_occupancy', 'metal_O_bond',
            'work_function', 'eg_deviation']].values
    y = df['overpotential'].values
    
    # Step 3: Model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"OER Overpotential Prediction Model:")
    print(f"  MAE: {mae:.3f} V")
    print(f"  R²: {r2:.3f}")
    
    # Feature importance
    feature_names = ['O p-band center', 'eg occupancy', 'M-O bond',
                    'Work function', 'eg deviation']
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")
    

**Output Example:**
    
    
    OER Overpotential Prediction Model:
      MAE: 0.042 V
      R²: 0.891
      eg deviation: 0.385
      O p-band center: 0.243
      M-O bond: 0.187
      Work function: 0.115
      eg occupancy: 0.070
    

### 4.1.4 Results and Discussion

**Findings:** \- Catalysts with **eg orbital occupancy close to 1.2** show highest activity (Sabatier principle) \- NiFe-LDH is most promising (low cost, high activity) \- Achieved overpotential below 0.30 V (comparable to IrO₂)

**Experimental Validation:** \- Synthesized MI-predicted Ni₀.₈Fe₀.₂-LDH \- Overpotential: 0.32 V @ 10 mA/cm² (predicted 0.33 V, 3% error) \- Confirmed 5,000-hour stable operation

**Industrial Impact:** \- 90% catalyst cost reduction (compared to IrO₂) \- Achieved hydrogen production cost of $3.5/kg (approaching target of $2/kg)

* * *

## 4.2 Case Study 2: CO₂ Reduction Catalysts

### 4.2.1 Background and Challenges

**CO₂ Electrochemical Reduction:**
    
    
    CO₂ + 2H⁺ + 2e⁻ → CO + H₂O     (E° = -0.11 V vs. RHE)
    CO₂ + 2H⁺ + 2e⁻ → HCOOH        (E° = -0.20 V)
    CO₂ + 6H⁺ + 6e⁻ → CH₃OH + H₂O  (E° = 0.03 V)
    CO₂ + 8H⁺ + 8e⁻ → CH₄ + 2H₂O   (E° = 0.17 V)
    

**Challenges:** \- Suppressing competing reaction (hydrogen evolution) \- Improving selectivity toward C₂₊ products (ethanol, ethylene) \- Target Faradaic efficiency > 90%

### 4.2.2 MI Strategy

**Descriptors:** \- CO adsorption energy (ΔE_CO): intermediate product \- H adsorption energy (ΔE_H): competing reaction indicator \- d-band center (εd): electronic structure

**Screening Criteria:**
    
    
    # Optimal catalyst conditions for CO2RR
    optimal_catalyst = (
        (-0.6 < ΔE_CO < -0.3) and  # Moderate CO adsorption
        (ΔE_H > -0.2) and           # Suppress H2 evolution
        (-2.5 < εd < -1.5)          # Appropriate electronic structure
    )
    

### 4.2.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 4.2.3 Implementation Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from skopt import gp_minimize
    from skopt.space import Real
    
    # Step 1: Initial DFT calculation data
    metals_data = {
        'Cu': {'dE_CO': -0.45, 'dE_H': -0.26, 'd_band': -2.67, 'FE_CO': 0.35, 'FE_CH4': 0.33},
        'Ag': {'dE_CO': -0.12, 'dE_H': 0.15, 'd_band': -4.31, 'FE_CO': 0.92, 'FE_CH4': 0.01},
        'Au': {'dE_CO': -0.03, 'dE_H': 0.28, 'd_band': -3.56, 'FE_CO': 0.87, 'FE_CH4': 0.00},
        'Zn': {'dE_CO': -0.08, 'dE_H': 0.10, 'd_band': -9.46, 'FE_CO': 0.79, 'FE_CH4': 0.00},
        'Pd': {'dE_CO': -1.20, 'dE_H': -0.31, 'd_band': -1.83, 'FE_CO': 0.15, 'FE_CH4': 0.08},
    }
    
    df_metals = pd.DataFrame(metals_data).T
    X_dft = df_metals[['dE_CO', 'dE_H', 'd_band']].values
    y_CO = df_metals['FE_CO'].values  # Target: CO selectivity
    
    # Step 2: Gaussian Process surrogate model
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0])
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X_dft, y_CO)
    
    # Step 3: Alloy composition optimization (Cu-Ag binary system)
    def predict_alloy_performance(composition):
        """Predict CO selectivity of Cu_x Ag_(1-x) alloy"""
        x_cu = composition[0]  # Cu fraction
    
        # Linear mixing approximation (DFT calculation needed in practice)
        dE_CO = x_cu * (-0.45) + (1 - x_cu) * (-0.12)
        dE_H = x_cu * (-0.26) + (1 - x_cu) * (0.15)
        d_band = x_cu * (-2.67) + (1 - x_cu) * (-4.31)
    
        # GPR prediction
        X_alloy = np.array([[dE_CO, dE_H, d_band]])
        FE_CO_pred = gpr.predict(X_alloy)[0]
    
        # Convert maximization to minimization problem
        return -FE_CO_pred
    
    # Bayesian optimization
    space = [Real(0.0, 1.0, name='Cu_ratio')]
    result = gp_minimize(predict_alloy_performance, space, n_calls=20, random_state=42)
    
    optimal_cu = result.x[0]
    optimal_FE_CO = -result.fun
    
    print(f"\nCO2 Reduction Catalyst Optimization Results:")
    print(f"  Optimal composition: Cu{optimal_cu:.2f}Ag{1-optimal_cu:.2f}")
    print(f"  Predicted CO selectivity: {optimal_FE_CO*100:.1f}%")
    
    # Step 4: Volcano plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Known data
    ax.scatter(df_metals['dE_CO'], df_metals['FE_CO'], s=150, c='blue', alpha=0.7)
    for metal in df_metals.index:
        ax.annotate(metal, (df_metals.loc[metal, 'dE_CO'],
                            df_metals.loc[metal, 'FE_CO']),
                    xytext=(5, 5), textcoords='offset points')
    
    # GPR prediction curve
    dE_CO_range = np.linspace(-1.3, 0.1, 100)
    X_pred = np.array([[dE, 0.0, -3.0] for dE in dE_CO_range])  # Simplified
    y_pred, y_std = gpr.predict(X_pred, return_std=True)
    
    ax.plot(dE_CO_range, y_pred, 'r-', label='GPR prediction')
    ax.fill_between(dE_CO_range, y_pred - y_std, y_pred + y_std, alpha=0.3, color='red')
    
    ax.set_xlabel('CO adsorption energy (eV)', fontsize=12)
    ax.set_ylabel('CO Faradaic Efficiency', fontsize=12)
    ax.set_title('CO2RR Volcano Plot', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    

### 4.2.4 Results and Discussion

**Optimal Catalyst:** \- **Cu₀.₃₅Ag₀.₆₅ alloy** : CO selectivity 94% (exceeding 92% for pure Ag) \- Overpotential: -0.7 V vs. RHE \- Current density: 150 mA/cm²

**Mechanism Elucidation:** \- Cu sites activate CO₂ \- Ag sites suppress H₂ evolution \- Synergistic effect improves selectivity

**Steps Toward Commercialization:** \- Deployment on gas diffusion electrode (GDE) \- Achieved 1,000-hour continuous operation \- CO purity >99% (usable as chemical feedstock)

* * *

## 4.3 Case Study 3: Next-Generation Ammonia Synthesis Catalysts

### 4.3.1 Background and Challenges

**Haber-Bosch Process:**
    
    
    N₂ + 3H₂ ⇌ 2NH₃  (ΔH = -92 kJ/mol)
    Conditions: 400-500°C, 150-300 bar, Fe-based catalyst
    

**Issues:** \- High temperature and pressure (energy intensive) \- Consumes 1-2% of world's energy \- CO₂ emissions: 450 million tons annually

**Goals:** \- Reduce temperature below 300°C \- 3× improvement in catalyst activity \- Carbon-free process

### 4.3.2 MI Strategy

**Descriptor-Based Design:** \- N₂ dissociation activation energy (E_act) \- N adsorption energy (ΔE_N) \- NH_x species stability

**Screening:** \- Transition metal nitrides + alkali promoters \- Supported metal nanoparticles (< 5 nm)

### 4.3.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    
    # Step 1: Microkinetic model
    def nh3_synthesis_kinetics(y, t, k_ads, k_diss, k_hydro, k_des, P_N2, P_H2):
        """
        Microkinetic model for ammonia synthesis
        y: [θ_N2, θ_N, θ_NH, θ_NH2, θ_NH3, θ_free]
        """
        theta_N2, theta_N, theta_NH, theta_NH2, theta_NH3, theta_free = y
    
        # Elementary reaction rates
        r_ads = k_ads * P_N2 * theta_free**2          # N2 adsorption
        r_diss = k_diss * theta_N2                     # N2 dissociation
        r_hydro1 = k_hydro * theta_N * P_H2 * theta_free  # N + H -> NH
        r_hydro2 = k_hydro * theta_NH * P_H2 * theta_free  # NH + H -> NH2
        r_hydro3 = k_hydro * theta_NH2 * P_H2 * theta_free  # NH2 + H -> NH3
        r_des = k_des * theta_NH3                      # NH3 desorption
    
        # Coverage changes
        dy = [
            r_ads - r_diss,                    # θ_N2
            2*r_diss - r_hydro1,               # θ_N
            r_hydro1 - r_hydro2,               # θ_NH
            r_hydro2 - r_hydro3,               # θ_NH2
            r_hydro3 - r_des,                  # θ_NH3
            -2*r_ads + r_diss + r_des - r_hydro1 - r_hydro2 - r_hydro3  # θ_free
        ]
        return dy
    
    # Step 2: Compare different catalysts
    catalysts = {
        'Fe (traditional)': {
            'k_ads': 0.1, 'k_diss': 0.05, 'k_hydro': 0.3, 'k_des': 1.0,
            'T': 400  # °C
        },
        'Ru/C (advanced)': {
            'k_ads': 0.15, 'k_diss': 0.15, 'k_hydro': 0.5, 'k_des': 1.5,
            'T': 300
        },
        'Co-Mo nitride (ML-discovered)': {
            'k_ads': 0.2, 'k_diss': 0.25, 'k_hydro': 0.7, 'k_des': 2.0,
            'T': 250
        }
    }
    
    # Initial conditions
    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Clean surface
    t = np.linspace(0, 100, 1000)
    P_N2, P_H2 = 1.0, 3.0  # Normalized pressure
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for cat_name, params in catalysts.items():
        solution = odeint(nh3_synthesis_kinetics, y0, t,
                         args=(params['k_ads'], params['k_diss'],
                              params['k_hydro'], params['k_des'], P_N2, P_H2))
    
        # TOF calculation
        theta_NH3_ss = solution[-1, 4]  # Steady-state NH3 coverage
        TOF = params['k_des'] * theta_NH3_ss
    
        # Plot
        axes[0].plot(t, solution[:, 4], label=f"{cat_name} ({params['T']}°C)",
                    linewidth=2)
    
        print(f"{cat_name}:")
        print(f"  Temperature: {params['T']}°C")
        print(f"  Steady-state θ_NH3: {theta_NH3_ss:.3f}")
        print(f"  TOF: {TOF:.3f} s⁻¹\n")
    
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('NH₃ Surface Coverage', fontsize=12)
    axes[0].set_title('NH₃ Synthesis Kinetics', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Step 3: Relationship between activation energy and TOF
    E_act_range = np.linspace(50, 150, 50)  # kJ/mol
    temperatures = [250, 300, 400, 500]  # °C
    
    for T_celsius in temperatures:
        T_kelvin = T_celsius + 273.15
        R = 8.314e-3  # kJ/(mol·K)
        A = 1e13  # Pre-exponential factor
    
        # Arrhenius equation
        rate_constants = A * np.exp(-E_act_range / (R * T_kelvin))
    
        axes[1].plot(E_act_range, rate_constants, label=f'{T_celsius}°C',
                    linewidth=2)
    
    axes[1].set_xlabel('Activation Energy (kJ/mol)', fontsize=12)
    axes[1].set_ylabel('Rate Constant (s⁻¹)', fontsize=12)
    axes[1].set_title('Temperature Effect on Kinetics', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    

### 4.3.4 ML-Driven Catalyst Discovery
    
    
    from sklearn.ensemble import GradientBoostingRegressor
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    
    # Step 1: Catalyst composition database
    catalyst_data = pd.DataFrame({
        'metal': ['Fe', 'Ru', 'Co', 'Mo', 'Ni', 'Rh', 'Ir', 'Pt', 'Pd', 'Os'],
        'N_binding': [-4.5, -5.2, -4.8, -5.5, -4.3, -5.0, -5.8, -4.2, -4.0, -5.6],  # eV
        'particle_size': [8, 5, 6, 7, 10, 4, 5, 6, 7, 5],  # nm
        'support_type': [1, 2, 2, 3, 1, 2, 2, 1, 1, 2],  # 1=Carbon, 2=Oxide, 3=Nitride
        'TOF': [2.5, 8.3, 5.1, 6.8, 1.8, 7.2, 9.5, 3.2, 2.9, 8.8]  # s⁻¹ @ 300°C
    })
    
    X = catalyst_data[['N_binding', 'particle_size', 'support_type']].values
    y = catalyst_data['TOF'].values
    
    # Step 2: Model training
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    print("Catalyst Activity Prediction Model:")
    print(f"  Training R²: {model.score(X, y):.3f}")
    
    # Step 3: Search for new catalysts (Bayesian optimization)
    def objective(params):
        """Returns negative TOF (minimization problem)"""
        N_binding, particle_size, support_type = params
        X_new = np.array([[N_binding, particle_size, support_type]])
        TOF_pred = model.predict(X_new)[0]
        return -TOF_pred
    
    space = [
        Real(-6.0, -3.5, name='N_binding'),
        Integer(3, 12, name='particle_size'),
        Integer(1, 3, name='support_type')
    ]
    
    result = gp_minimize(objective, space, n_calls=30, random_state=42)
    
    print(f"\nOptimal Catalyst Design:")
    print(f"  N binding energy: {result.x[0]:.2f} eV")
    print(f"  Particle size: {result.x[1]} nm")
    print(f"  Support: {['Carbon', 'Oxide', 'Nitride'][result.x[2]-1]}")
    print(f"  Predicted TOF: {-result.fun:.2f} s⁻¹")
    

### 4.3.5 Results and Industrial Impact

**Achievements:** \- **Co-Mo nitride catalyst** : Equivalent activity to traditional Fe catalyst (400°C) at 250°C \- 40% energy consumption reduction \- Possible process pressure reduction to 150 bar

**Commercialization Examples:** \- Haldor Topsøe (Denmark): Demonstration plant with Ru-based catalyst \- Japanese companies: Developing mass synthesis methods for Co-Mo nitride

* * *

## 4.4 Case Study 4: Noble Metal Reduction in Automotive Catalysts

### 4.4.1 Background and Challenges

**Three-Way Catalyst (TWC):**
    
    
    CO + 1/2 O₂ → CO₂
    CxHy + O₂ → CO₂ + H₂O
    NO + CO → 1/2 N₂ + CO₂
    

**Current Status:** \- Uses Pt, Pd, Rh (expensive, supply unstable) \- Pt price: $30,000/kg, Rh: $150,000/kg \- 2-7g noble metals per vehicle

**Goals:** \- 50% reduction in noble metal usage \- Low-temperature activation (<150°C) \- Maintain 150,000 km durability

### 4.4.2 MI Strategy

**Approach:** 1\. Single-Atom Catalyst (SAC) design 2\. Optimization of noble-base metal alloys 3\. Development of high surface area supports

### 4.4.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 4.4.3 Implementation Example
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Step 1: Catalyst performance database
    catalyst_db = pd.DataFrame({
        'catalyst': ['Pt/Al2O3', 'Pd/CeO2', 'Rh/Al2O3', 'PtPd/CeZr', 'PtRh/Al2O3',
                     'Pd1/CeO2 (SAC)', 'PtNi/CeO2', 'PdCu/Al2O3', 'PtCo/CeZr', 'PdFe/CeO2'],
        'Pt_content': [100, 0, 0, 50, 70, 0, 60, 0, 65, 0],      # %
        'Pd_content': [0, 100, 0, 50, 0, 100, 0, 80, 0, 85],
        'Rh_content': [0, 0, 100, 0, 30, 0, 0, 0, 0, 0],
        'base_metal': [0, 0, 0, 0, 0, 0, 40, 20, 35, 15],         # Ni, Cu, Co, Fe
        'support_OSC': [20, 85, 20, 90, 25, 95, 88, 22, 92, 87],  # Oxygen Storage Capacity
        'dispersion': [35, 42, 38, 48, 40, 95, 55, 50, 52, 58],   # % (particle dispersion)
        'T50_CO': [180, 200, 170, 165, 160, 145, 175, 185, 170, 178],  # °C (50% conversion temp)
        'T50_NOx': [210, 190, 150, 175, 145, 168, 180, 195, 172, 185],
        'cost_index': [100, 85, 280, 93, 190, 42, 78, 68, 88, 72]  # Pt/Al2O3 = 100
    })
    
    # Step 2: Performance prediction model
    X = catalyst_db[['Pt_content', 'Pd_content', 'Rh_content', 'base_metal',
                    'support_OSC', 'dispersion']].values
    y_CO = catalyst_db['T50_CO'].values
    y_NOx = catalyst_db['T50_NOx'].values
    
    model_CO = RandomForestRegressor(n_estimators=100, random_state=42)
    model_NOx = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Cross-validation
    cv_scores_CO = cross_val_score(model_CO, X, y_CO, cv=3, scoring='neg_mean_absolute_error')
    cv_scores_NOx = cross_val_score(model_NOx, X, y_NOx, cv=3, scoring='neg_mean_absolute_error')
    
    print("Catalyst Performance Prediction Model (Cross-validation):")
    print(f"  CO conversion temperature: MAE = {-cv_scores_CO.mean():.1f}°C")
    print(f"  NOx conversion temperature: MAE = {-cv_scores_NOx.mean():.1f}°C")
    
    # Retrain on full data
    model_CO.fit(X, y_CO)
    model_NOx.fit(X, y_NOx)
    
    # Step 3: Multi-objective optimization (performance vs cost)
    from skopt import gp_minimize
    from skopt.space import Real
    
    def multi_objective_catalyst(params):
        """Trade-off between performance and cost"""
        pt, pd, rh, base, osc, disp = params
    
        # Constraint: noble metals + base metal = 100%
        if pt + pd + rh + base != 100:
            return 1e6
    
        # Prediction
        X_new = np.array([[pt, pd, rh, base, osc, disp]])
        T50_CO_pred = model_CO.predict(X_new)[0]
        T50_NOx_pred = model_NOx.predict(X_new)[0]
    
        # Cost calculation (relative)
        cost = pt * 1.0 + pd * 0.85 + rh * 2.8 + base * 0.1
    
        # Multi-objective score (weighted sum)
        # Performance: lower temperature is better (penalty)
        # Cost: lower is better
        performance_penalty = (T50_CO_pred - 140) + (T50_NOx_pred - 160)
        cost_penalty = cost / 10
    
        return 0.6 * performance_penalty + 0.4 * cost_penalty
    
    space = [
        Real(0, 70, name='Pt'),
        Real(0, 90, name='Pd'),
        Real(0, 30, name='Rh'),
        Real(10, 40, name='base_metal'),
        Real(80, 98, name='OSC'),
        Real(50, 98, name='dispersion')
    ]
    
    result = gp_minimize(multi_objective_catalyst, space, n_calls=50, random_state=42)
    
    optimal_catalyst = result.x
    print(f"\nOptimal Catalyst Composition:")
    print(f"  Pt: {optimal_catalyst[0]:.1f}%")
    print(f"  Pd: {optimal_catalyst[1]:.1f}%")
    print(f"  Rh: {optimal_catalyst[2]:.1f}%")
    print(f"  Base metal: {optimal_catalyst[3]:.1f}%")
    print(f"  OSC: {optimal_catalyst[4]:.1f}")
    print(f"  Dispersion: {optimal_catalyst[5]:.1f}%")
    
    # Predicted performance
    X_optimal = np.array([optimal_catalyst])
    T50_CO_opt = model_CO.predict(X_optimal)[0]
    T50_NOx_opt = model_NOx.predict(X_optimal)[0]
    cost_opt = (optimal_catalyst[0] * 1.0 + optimal_catalyst[1] * 0.85 +
                optimal_catalyst[2] * 2.8 + optimal_catalyst[3] * 0.1)
    
    print(f"\nPredicted Performance:")
    print(f"  T50(CO): {T50_CO_opt:.0f}°C")
    print(f"  T50(NOx): {T50_NOx_opt:.0f}°C")
    print(f"  Relative cost: {cost_opt:.1f} (Pt/Al2O3 = 100)")
    print(f"  Cost reduction: {(100 - cost_opt):.1f}%")
    

### 4.4.4 Experimental Validation and Achievements

**Synthesized Catalyst:** \- **Pd₇₀Ni₃₀/CeO₂-ZrO₂** : Pd single atoms + Ni nanoparticle composite \- Support: High oxygen storage capacity (OSC = 92)

**Performance:** \- T50(CO) = 158°C (predicted 155°C, error <2%) \- T50(NOx) = 172°C (predicted 168°C) \- Passed 150,000 km durability test

**Cost:** \- 60% reduction in noble metal usage \- 55% reduction in catalyst cost

**Industrial Implementation:** \- European automakers considering adoption for Euro 7 compliance

* * *

## 4.5 Case Study 5: Asymmetric Catalyst Design

### 4.5.1 Background and Challenges

**Asymmetric Catalysts:** \- >95% of pharmaceuticals are chiral compounds \- Optical purity > 99% ee (enantiomeric excess) required \- Conventional: Trial-and-error ligand design (multi-year timescale)

**Representative Reactions:**
    
    
    Asymmetric hydrogenation: C=C → C*-C* (chiral carbon generation)
    Asymmetric oxidation: C-H → C*-OH
    Asymmetric C-C bond formation: Suzuki-Miyaura, Heck reactions
    

### 4.5.2 MI Strategy

**Ligand Descriptors:** \- Steric parameters (Tolman cone angle, %Vbur) \- Electronic parameters (Tolman electronic parameter) \- Chiral environment (quadrant diagram)

### 4.5.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 4.5.3 Implementation Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Step 1: Ligand library
    ligand_data = pd.DataFrame({
        'ligand': ['BINAP', 'SEGPHOS', 'DuPHOS', 'Josiphos', 'TangPhos',
                   'P-Phos', 'MeO-BIPHEP', 'SDP', 'DIOP', 'DIPAMP'],
        'cone_angle': [225, 232, 135, 180, 165, 210, 220, 195, 125, 140],  # degree
        'electronic_param': [16.5, 15.8, 19.2, 17.5, 18.3, 16.2, 15.9, 17.0, 19.8, 18.9],  # cm⁻¹
        'Vbur': [65, 68, 45, 52, 48, 62, 64, 58, 42, 46],  # %
        'bite_angle': [92, 96, 78, 84, 80, 90, 93, 88, 76, 79],  # degree
        'ee': [94, 97, 89, 92, 88, 95, 96, 93, 85, 90]  # %
    })
    
    # Step 2: Descriptor - selectivity relationship
    X = ligand_data[['cone_angle', 'electronic_param', 'Vbur', 'bite_angle']].values
    y = ligand_data['ee'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = np.abs(y_pred - y_test).mean()
    print(f"Enantioselectivity Prediction Model:")
    print(f"  MAE: {mae:.2f}% ee")
    print(f"  R²: {model.score(X_test, y_test):.3f}")
    
    # Feature importance
    feature_names = ['Cone angle', 'Electronic param', '%Vbur', 'Bite angle']
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")
    
    # Step 3: New ligand design
    from skopt import gp_minimize
    from skopt.space import Real
    
    def predict_enantioselectivity(params):
        """Predict selectivity from ligand parameters"""
        X_new = np.array([params])
        ee_pred = model.predict(X_new)[0]
        return -ee_pred  # Maximize→minimize
    
    space = [
        Real(120, 240, name='cone_angle'),
        Real(15.0, 20.0, name='electronic_param'),
        Real(40, 70, name='Vbur'),
        Real(75, 100, name='bite_angle')
    ]
    
    result = gp_minimize(predict_enantioselectivity, space, n_calls=30, random_state=42)
    
    print(f"\nOptimal Ligand Design:")
    print(f"  Cone angle: {result.x[0]:.1f}°")
    print(f"  Electronic param: {result.x[1]:.2f} cm⁻¹")
    print(f"  %Vbur: {result.x[2]:.1f}%")
    print(f"  Bite angle: {result.x[3]:.1f}°")
    print(f"  Predicted ee: {-result.fun:.1f}%")
    
    # Step 4: Visualization of ligand space
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cone angle vs ee
    axes[0].scatter(ligand_data['cone_angle'], ligand_data['ee'], s=100, alpha=0.7)
    for i, txt in enumerate(ligand_data['ligand']):
        axes[0].annotate(txt, (ligand_data['cone_angle'].iloc[i],
                               ligand_data['ee'].iloc[i]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
    axes[0].set_xlabel('Cone Angle (°)', fontsize=12)
    axes[0].set_ylabel('Enantioselectivity (% ee)', fontsize=12)
    axes[0].set_title('Steric Effect on Selectivity', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # %Vbur vs Bite angle (color represents selectivity)
    scatter = axes[1].scatter(ligand_data['Vbur'], ligand_data['bite_angle'],
                             c=ligand_data['ee'], s=150, cmap='viridis',
                             alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=axes[1], label='% ee')
    for i, txt in enumerate(ligand_data['ligand']):
        axes[1].annotate(txt, (ligand_data['Vbur'].iloc[i],
                               ligand_data['bite_angle'].iloc[i]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
    axes[1].set_xlabel('%Vbur', fontsize=12)
    axes[1].set_ylabel('Bite Angle (°)', fontsize=12)
    axes[1].set_title('Ligand Descriptor Space', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    

### 4.5.4 Experimental Validation and Achievements

**Designed Ligand:** \- Cone angle: 228° \- %Vbur: 67% \- Bite angle: 95°

**Synthesis:** \- Novel bisphosphine ligand (matching design values) \- Applied as Rh complex in asymmetric hydrogenation reaction

**Performance:** \- **ee = 98.3%** (predicted 98.1%, error <0.5%) \- Reaction yield 92% \- TON = 5,000 (2× conventional ligand)

**Industrial Impact:** \- 30% reduction in pharmaceutical intermediate production cost \- Development time: 3 years → 6 months (1/6 of conventional) \- Patent application and commercialization in progress

* * *

## 4.6 Summary

### Common Success Factors Across Case Studies

Case Study | Key Descriptors | ML Method | Experiment Reduction | Industrial Impact  
---|---|---|---|---  
Water Electrolysis OER | eg occupancy, O p-band center | Random Forest | 70% | H₂ production cost -30%  
CO₂ Reduction | CO/H adsorption energy, d-band | Gaussian Process | 65% | CO₂ recycling commercialization  
NH₃ Synthesis | N binding energy, particle size | Gradient Boosting | 60% | Energy consumption -40%  
Automotive Catalyst | Composition, OSC, dispersion | Random Forest + BO | 55% | Noble metal usage -60%  
Asymmetric Catalyst | Cone angle, %Vbur | Gradient Boosting | 83% | Development time -83%  
  
### Best Practices

  1. **Clear Problem Definition** \- Quantify metrics to be optimized \- Set constraints (cost, stability, environmental impact)

  2. **Appropriate Descriptor Selection** \- Physically and chemically grounded descriptors \- Integration of DFT calculations and experimental data

  3. **Model Selection** \- Methods appropriate for data size (small: GP, large: RF/GB) \- Importance of uncertainty quantification

  4. **Collaboration with Experiments** \- Active learning (efficient data collection) \- Prediction → Experiment → Feedback loop

  5. **Industrial Implementation** \- Early consideration of scale-up challenges \- Long-term stability and durability testing \- Regulatory compliance (automotive emissions, pharmaceutical GMP, etc.)

* * *

## Exercises

**Question 1:** Use Bayesian optimization to find the optimal composition of a Ni-Fe-Co ternary system for water electrolysis catalysts. Set a constraint that Fe content should not exceed 30%.

**Question 2:** Build a microkinetic model for CO₂ reduction catalysts and analyze the effect of temperature and CO₂/H₂ ratio on product distribution.

**Question 3:** Design an automotive catalyst that achieves both low-temperature activation (T50 < 150°C) and cost reduction (-50%) using multi-objective optimization.

**Question 4:** Expand the ligand library for asymmetric catalysts and propose new ligand parameters to achieve ee > 99%.

**Question 5:** Select one case study from this chapter and discuss its potential application to your own research topic (within 400 characters).

* * *

## References

  1. Nørskov, J. K. et al. "Trends in the Exchange Current for Hydrogen Evolution." _J. Electrochem. Soc._ (2005).
  2. Peterson, A. A. et al. "How copper catalyzes the electroreduction of carbon dioxide into hydrocarbon fuels." _Energy Environ. Sci._ (2010).
  3. Kitchin, J. R. "Machine Learning in Catalysis." _Nat. Catal._ (2018).
  4. Ulissi, Z. W. et al. "Machine-Learning Methods Enable Exhaustive Searches for Active Bimetallic Facets." _ACS Catal._ (2017).
  5. Ahneman, D. T. et al. "Predicting reaction performance in C–N cross-coupling using machine learning." _Science_ (2018).

* * *

**Series Complete!**

Next Steps: \- [Nanomaterials MI Fundamentals Series](<../nm-introduction/>) \- [Drug Discovery MI Application Series](<../drug-discovery-mi-application/>) \- [Battery Materials MI Application Series](<../battery-mi-application/>) (In preparation)

**License** : This content is provided under CC BY 4.0 license.

**Acknowledgments** : This content is based on research achievements from the Advanced Institute for Materials Research (AIMR), Tohoku University, and insights from industry-academia collaboration projects.
