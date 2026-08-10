---
title: "Chapter 4: Processing and Heat Treatment of Metallic Materials"
chapter_title: "Chapter 4: Processing and Heat Treatment of Metallic Materials"
subtitle: Metal Processing and Heat Treatment
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/metallic-materials-introduction/chapter-4.html>) | Last sync: 2025-11-16

# Chapter 4: Processing and Heat Treatment of Metallic Materials

Metal Processing and Heat Treatment

[MS Dojo Top](<../index.html>)>[Introduction to Metallic Materials](<index.html>)> Chapter 4

## Introduction

The characteristics of metallic materials are determined by chemical composition, processing, and heat treatment. In this chapter, we study heat treatment processes such as plastic processing, annealing, quenching and tempering, and age hardening, and understand phase transformations using TTT diagrams (Time-Temperature-Transformation) and CCT diagrams (Continuous-Cooling-Transformation). We also learn how to analyze the relationship between heat treatment conditions and microstructure/properties through Python simulations.

### Learning Objectives

  * Understand the principles of plastic processing (rolling, forging, extrusion, drawing) and the formation of processing microstructures
  * Explain the mechanisms of recrystallization, recovery, and grain growth, and the effects of annealing treatment
  * Understand steel quenching and tempering processes and martensite transformation
  * Learn the principles of age hardening treatment (Al alloys, stainless steels) and precipitation strengthening
  * Read TTT and CCT diagrams and predict microstructures obtainable under various cooling conditions
  * Simulate transformation rates, cooling curves, and age hardening curves using Python

## 4.1 Plastic Processing

### 4.1.1 Fundamentals of Plastic Processing

Plastic processing is a processing method that applies stress greater than the yield stress of a material to cause permanent deformation and obtain the desired shape. The main plastic processing methods are as follows:

  * **Rolling** : Passing material between rollers to reduce thickness and extend length
  * **Forging** : Applying compressive force to material to shape it through hammering or pressing
  * **Extrusion** : Pushing material through a die to create long products with constant cross-section
  * **Drawing** : Pulling material through a die to reduce cross-section and produce wire or tubes

### 4.1.2 Processing Microstructure and Work Hardening

During plastic processing, crystal grains deform and dislocation density increases. This causes work hardening, which improves the strength of the material. The relationship between the degree of processing (processing strain) and strength is expressed by the following empirical equation:

\\[\sigma = \sigma_0 + K \varepsilon^n \\]

where \\(\sigma\\) is stress, \\(\sigma_0\\) is initial yield stress, \\(K\\) is the strength coefficient, \\(\varepsilon\\) is strain, and \\(n\\) is the work hardening index.

**Code Example 1: Simulation of Work Hardening Curve**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Work hardening curve simulation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def work_hardening_curve(strain, sigma_0, K, n): """ Work HardeningCurve Calculation Parameters: ----------- strain: array-like Strain sigma_0: float initialYield Stress [MPa] K: float Strength Coefficient [MPa] n: float Work Hardening Index (0.1-0.5) Returns: -------- stress: array-like Stress [MPa] """ return sigma_0 + K * strain**n # materialParameter（Low Carbon Steel of example） sigma_0 = 200 # MPa K = 500 # MPa n = 0.25 # Work Hardening Index # Strainrange strain = np.linspace(0, 0.5, 100) # differentWork Hardening Index of Ratiocomparison n_values = [0.1, 0.25, 0.4] colors = ['blue', 'green', 'red'] plt.figure(figsize=(10, 6)) for n_val, color in zip(n_values, colors): stress = work_hardening_curve(strain, sigma_0, K, n_val) plt.plot(strain, stress, color=color, linewidth=2, label=f'n = {n_val}') plt.xlabel('True Strain ε', fontsize=12) plt.ylabel('True Stress σ [MPa]', fontsize=12) plt.title('Work Hardening Behavior (σ = σ₀ + K·εⁿ)', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('work_hardening.png', dpi=300, bbox_inches='tight') plt.show() print(f"Initial yield stress: {sigma_0} MPa") print(f"Strength coefficient K: {K} MPa") print(f"At 20% strain (ε=0.2), stress = {work_hardening_curve(0.2, sigma_0, K, 0.25):.1f} MPa")

### 4.1.3 Recrystallization and Recovery

When processed materials are heated, dislocation rearrangement (recovery) and the formation and growth of strain-free new grains (recrystallization) occur. The recrystallization temperature is approximately 0.3 to 0.5 of the melting point \\(T_m\\) (in Kelvin), varying by material. Through recrystallization, work-hardened materials soften and ductility is recovered.

## 4.2 Annealing Treatment

### 4.2.1 Types and Purposes of Annealing

Annealing is a heat treatment that involves heating a material to a specific temperature, holding it, and then slowly cooling it. The main purposes and types are:

  * **Stress Relief Annealing** : Removes residual stress formed during processing (approximately 0.3Tm)
  * **Recrystallization Annealing** : Recrystallizes processed microstructure for softening (approximately 0.5Tm)
  * **Full Annealing** : Heating to austenite region followed by slow cooling to obtain ferrite + pearlite microstructure (for steel)
  * **Homogenization Annealing** : Eliminates composition segregation in the microstructure (high temperature, long time, approximately 0.8Tm)

### 4.2.2 Recrystallization Rate and Johnson-Mehl-Avrami (JMA) Equation

The progress of recrystallization \\(X\\) (transformation fraction) as a function of time \\(t\\) and temperature \\(T\\) is expressed by the JMA equation:

\\[X(t) = 1 - \exp\left(-kt^n\right) \\]

where \\(k\\) is the rate constant (with Arrhenius temperature dependence), and \\(n\\) is the Avrami index (which reflects nucleation and growth mechanisms, typically 1 to 4).

**Code Example 2: Recrystallization Rate Simulation Using JMA Equation**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: JMA equation recrystallization rate simulation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def jma_transformation(time, k, n): """ Johnson-Mehl-AvramiequationbytransformationrateCalculation Parameters: ----------- time: array-like time [s] k: float rateConstant [1/s^n] n: float AvramiIndex (1-4) Returns: -------- X: array-like transformationrate（0～1） """ return 1 - np.exp(-k * time**n) # timerange time = np.logspace(-2, 4, 200) # 0.01s to 10000s # differentTemperature of Simulation（rateConstantk change） temps = [500, 550, 600, 650] # °C k_values = [1e-6, 5e-6, 2e-5, 8e-5] # high n = 2.5 # AvramiIndex（3 ） fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # Figure: differentTemperature of recrystallizationCurve for temp, k in zip(temps, k_values): X = jma_transformation(time, k, n) ax1.semilogx(time, X * 100, linewidth=2, label=f'{temp}°C') # 50%transformationtime Calculation t_50 = (np.log(2) / k)**(1/n) ax1.plot(t_50, 50, 'o', markersize=8) ax1.set_xlabel('Time [s]', fontsize=12) ax1.set_ylabel('Recrystallized Fraction [%]', fontsize=12) ax1.set_title('Recrystallization Kinetics (JMA Model)', fontsize=14) ax1.legend(fontsize=11) ax1.grid(True, alpha=0.3) ax1.set_ylim(0, 105) # Figure: AvramiIndex of influence（600°C） n_values = [1, 2, 3, 4] k_ref = 2e-5 for n_val in n_values: X = jma_transformation(time, k_ref, n_val) ax2.semilogx(time, X * 100, linewidth=2, label=f'n = {n_val}') ax2.set_xlabel('Time [s]', fontsize=12) ax2.set_ylabel('Recrystallized Fraction [%]', fontsize=12) ax2.set_title('Effect of Avrami Exponent n', fontsize=14) ax2.legend(fontsize=11) ax2.grid(True, alpha=0.3) ax2.set_ylim(0, 105) plt.tight_layout() plt.savefig('jma_recrystallization.png', dpi=300, bbox_inches='tight') plt.show() # 50%recrystallizationtime of Calculationexample t_50_calc = (np.log(2) / k_values[2])**(1/n) print(f"At 600°C, 50% recrystallization time: {t_50_calc:.2f} seconds")

## 4.3 Quenching and Tempering (Heat Treatment of Steel)

### 4.3.1 Quenching Process

Quenching is a heat treatment in which steel is heated to the austenite region (above the A3 point, typically 850-950C) and then rapidly cooled to obtain a martensite microstructure. Martensite is a hard microstructure formed by diffusionless transformation from austenite. The martensite start temperature \\(M_s\\) and finish temperature \\(M_f\\) depend on carbon content \\(C\\) (wt%) according to the following empirical equation:

\\[M_s (°\text{C}) \approx 539 - 423C - 30.4\text{Mn} - 17.7\text{Ni} - 12.1\text{Cr} \\]

### 4.3.2 Tempering Treatment

Quenched martensite is hard but brittle, so tempering is performed. By heating and holding at 150-650C, carbide precipitation occurs, internal stress is relieved, and toughness improves. Depending on the tempering temperature, different microstructures such as tempered martensite, troostite, and sorbite can be obtained.

**Code Example 3: Ms Temperature Calculation and Cooling Curve Simulation**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Ms temperature calculation and cooling curve simulation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def calculate_ms_temperature(C, Mn=0, Ni=0, Cr=0): """ Martensitetransformation startTemperatureMs of Calculation（Andrewsequation） Parameters: ----------- C: float Carbon Content [wt%] Mn, Ni, Cr: float AlloyelementContent [wt%] Returns: -------- Ms: float Martensitetransformation startTemperature [°C] """ Ms = 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr return Ms def cooling_curve(time, T0, T_env, cooling_rate): """ CoolingCurve of Simulation（NewtonCoolingrule of simplifiedmoderu） Parameters: ----------- time: array-like time [s] T0: float initialTemperature [°C] T_env: float Temperature（Cooling Temperature） [°C] cooling_rate: float CoolingrateParameter [1/s] Returns: -------- T: array-like Temperature [°C] """ return T_env + (T0 - T_env) * np.exp(-cooling_rate * time) # carbonSteel of MsTemperatureCalculation carbon_contents = [0.2, 0.4, 0.6, 0.8, 1.0] print("=== Martensite Start Temperature (Ms) ===") for C in carbon_contents: Ms = calculate_ms_temperature(C) print(f"C = {C:.1f} wt%: Ms = {Ms:.1f}°C") # CoolingCurveSimulation time = np.linspace(0, 100, 500) # 0-100second T0 = 850 # AusteniteizationTemperature # differentCooling （Coolingrate different） cooling_media = { 'Water': (20, 0.08), # water cooling（ ） 'Oil': (60, 0.03), # oil cooling（ ） 'Air': (25, 0.005) # （ ） } plt.figure(figsize=(12, 6)) for media, (T_env, rate) in cooling_media.items(): T = cooling_curve(time, T0, T_env, rate) plt.plot(time, T, linewidth=2, label=media) # MsTemperatureline （0.4%CSteel of example） Ms_04 = calculate_ms_temperature(0.4) plt.axhline(y=Ms_04, color='red', linestyle='--', linewidth=2, label=f'Ms (0.4%C) = {Ms_04:.0f}°C') # A1Temperatureline（Approximately727°C、Ferrite+Pearlitetransformation start） plt.axhline(y=727, color='orange', linestyle='--', linewidth=2, label='A1 = 727°C') plt.xlabel('Time [s]', fontsize=12) plt.ylabel('Temperature [°C]', fontsize=12) plt.title('Cooling Curves for Different Quenching Media', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.xlim(0, 100) plt.ylim(0, 900) plt.tight_layout() plt.savefig('cooling_curves.png', dpi=300, bbox_inches='tight') plt.show() # Coolingrate of evaluate print("\n=== Cooling Rate Analysis ===") for media, (T_env, rate) in cooling_media.items(): T_at_10s = cooling_curve(10, T0, T_env, rate) avg_rate = (T0 - T_at_10s) / 10 print(f"{media}: Average cooling rate (0-10s) = {avg_rate:.1f}°C/s")

### 4.3.3 Hardenability

Hardenability refers to the depth (or ease) of hardening that can be achieved through quenching. It is determined by the chemical composition and grain size of the material, and is evaluated using the Jominy end-quench test. Addition of alloying elements (Mn, Cr, Mo, Ni, etc.) improves hardenability.

## 4.4 Age Hardening Treatment

### 4.4.1 Principles of Age Hardening

Age hardening (also known as precipitation hardening) is a precipitation strengthening heat treatment used for Al alloys, stainless steels, and other alloys. The process consists of three steps:

  1. **Solution Treatment** : Heating to high temperature (500-550C for Al-Cu alloys) to dissolve solute elements into solid solution
  2. **Quenching** : Rapid cooling to obtain a supersaturated solid solution
  3. **Aging Treatment** : Holding at moderate temperature (150-200C) to form fine precipitates (GP zones to theta'' to theta' to theta)

Precipitates hinder dislocation motion and improve strength (Orowan mechanism, see Chapter 3). Depending on aging time, peak aging (maximum hardness) and overaging (hardness decrease) can occur.

**Code Example 4: Simulation of Age Hardening Curve**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Age hardening curve simulation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    def age_hardening_curve(time, H0, H_max, k_peak, k_over, t_peak):
        """
        Modelling the age hardening curve (simple empirical expression)

        Parameters:
        -----------
        time : array-like
            Aging time [hours]
        H0 : float
            Initial hardness (as-quenched) [HV]
        H_max : float
            Maximum hardness (peak aging) [HV]
        k_peak : float
            Hardening rate parameter
        k_over : float
            Over-aging softening rate parameter
        t_peak : float
            Peak aging time [hours]

        Returns:
        --------
        hardness : array-like
            Hardness [HV]
        """
        # Hardening up to peak aging
        H_peak = H0 + (H_max - H0) * (1 - np.exp(-k_peak * time))

        # Softening caused by over-aging
        H_over = H_max - (H_max - H0) * 0.3 * (1 - np.exp(-k_over * (time - t_peak)))

        # Switch between the two regimes by time
        hardness = np.where(time < t_peak, H_peak, H_over)
        return hardness

    # Age hardening simulation for an Al-Cu alloy (2024 series)
    time = np.logspace(-1, 3, 300)  # 0.1 to 1000 hours

    # Simulation at different aging temperatures
    aging_temps = [150, 175, 200]  # °C
    colors = ['blue', 'green', 'red']

    # The higher the temperature, the faster the hardening but the lower the peak hardness
    params = {
        150: {'H0': 70, 'H_max': 140, 'k_peak': 0.05, 'k_over': 0.01, 't_peak': 50},
        175: {'H0': 70, 'H_max': 135, 'k_peak': 0.15, 'k_over': 0.03, 't_peak': 15},
        200: {'H0': 70, 'H_max': 125, 'k_peak': 0.30, 'k_over': 0.08, 't_peak': 5}
    }

    plt.figure(figsize=(12, 6))

    for temp, color in zip(aging_temps, colors):
        p = params[temp]
        hardness = age_hardening_curve(time, p['H0'], p['H_max'],
                                         p['k_peak'], p['k_over'], p['t_peak'])
        plt.semilogx(time, hardness, color=color, linewidth=2,
                     label=f'{temp}°C (Peak: {p["H_max"]} HV at {p["t_peak"]}h)')

        # Mark the peak aging point
        plt.plot(p['t_peak'], p['H_max'], 'o', color=color, markersize=10)

    # Add natural aging (room temperature, 25°C)
    natural_aging = 70 + (95 - 70) * (1 - np.exp(-0.005 * time))
    plt.semilogx(time, natural_aging, 'k--', linewidth=2,
                 label='Natural Aging (25°C)')

    plt.xlabel('Aging Time [hours]', fontsize=12)
    plt.ylabel('Hardness [HV]', fontsize=12)
    plt.title('Age Hardening Curves for Al-Cu Alloy (2024)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 1000)
    plt.ylim(60, 150)
    plt.tight_layout()
    plt.savefig('age_hardening.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Recommended conditions for T6 treatment (solution treatment + artificial aging)
    print("=== Recommended T6 Heat Treatment ===")
    print("Solution Treatment: 495-505°C, 1-2 hours")
    print("Quenching: Water (rapid cooling to RT)")
    print("Aging: 175-190°C, 8-12 hours")
    print(f"Expected Peak Hardness: ~135 HV (at 175°C, 15 hours)")

### 4.4.2 Precipitate Evolution and Strengthening Mechanisms

In the Al-Cu system, precipitates evolve as: GP zones (coherent, several nm) to theta'' (coherent) to theta' (semi-coherent) to theta (equilibrium phase, Al2Cu). Maximum strength is obtained when fine theta'' and theta' precipitates are at high density (peak aging). During overaging, precipitates coarsen and particle spacing increases, leading to decreased strength.

## 4.5 TTT and CCT Diagrams

### 4.5.1 TTT Diagram (Time-Temperature-Transformation Diagram)

A TTT diagram shows the start and completion times of phase transformations when austenite is held at a constant temperature. Depending on steel type, pearlite transformation curves (at higher temperatures) and bainite transformation curves (at lower temperatures) appear, and there is a temperature where transformation rate is fastest (nose temperature). Below the Ms line, martensitic transformation occurs.

### 4.5.2 CCT Diagram (Continuous-Cooling-Transformation Diagram)

A CCT diagram shows transformations during continuous cooling and more closely represents actual heat treatment conditions. Compared to TTT diagrams, transformation curves shift to lower temperatures and longer times. The points where a cooling curve crosses transformation curves on the CCT diagram determine the resulting microstructure.
    
    
    ```mermaid
    graph TD
        A[Austenitizing850-950C] --> B{Cooling Rate}
        B -->|Rapid CoolingWater Quench| C[MartensiteMaximum Hardness]
        B -->|Medium CoolingOil Quench| D[BainiteIntermediate Hardness]
        B -->|Slow CoolingFurnace Cool| E[Ferrite+PearliteSoft]
        C --> F[Tempering150-650C]
        F --> G[Tempered MartensiteHigh Strength+Toughness]

        style A fill:#f093fb
        style C fill:#f5576c
        style G fill:#4ade80
    ```

**Code Example 5: Simplified TTT Diagram Visualization**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Simplified TTT diagram visualization
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def ttt_curve_pearlite(time): """ PearlitetransformationCurve（simplifiedmoderu） Parameters: ----------- time: array-like time [s] Returns: -------- T_start, T_end: array-like transformation start・ Temperature [°C] """ # C ca bu of moderuization（experimentalfit） t_nose = 1 # no zu of time（second） T_nose = 550 # no zuTemperature（°C） # transformation startCurve T_start = T_nose + 150 * (np.log10(time / t_nose))**2 T_start = np.clip(T_start, 400, 727) # transformation Curve（startthan ） T_end = T_nose + 150 * (np.log10(time / (t_nose * 10)))**2 T_end = np.clip(T_end, 400, 727) return T_start, T_end def ttt_curve_bainite(time): """ BainitetransformationCurve（simplifiedmoderu） """ t_nose = 10 T_nose = 350 T_start = T_nose + 80 * (np.log10(time / t_nose))**2 T_start = np.clip(T_start, 250, 500) T_end = T_nose + 80 * (np.log10(time / (t_nose * 10)))**2 T_end = np.clip(T_end, 250, 500) return T_start, T_end # timerange（ numberske ru） time = np.logspace(-1, 5, 200) # 0.1s to 100000s # TTTCurve of Calculation P_start, P_end = ttt_curve_pearlite(time) B_start, B_end = ttt_curve_bainite(time) # MsTemperature（Martensitetransformation start） Ms = 350 # 0.4%CSteel of example # TTTFigure of creation fig, ax = plt.subplots(figsize=(12, 8)) # Pearlitetransformationregion ax.fill_betweenx(P_start, time, 1e5, alpha=0.3, color='orange', label='Austenite') ax.fill_betweenx(P_start, time, time * 0 + 0.1, alpha=0.4, color='yellow', where=(P_start>400), label='Pearlite Region') # PearlitetransformationCurve ax.semilogx(time, P_start, 'r-', linewidth=2, label='Pearlite Start') ax.semilogx(time, P_end, 'r--', linewidth=2, label='Pearlite End') # BainitetransformationCurve ax.semilogx(time, B_start, 'b-', linewidth=2, label='Bainite Start') ax.semilogx(time, B_end, 'b--', linewidth=2, label='Bainite End') # Martensitetransformationline ax.axhline(y=Ms, color='green', linestyle='-', linewidth=2.5, label=f'Ms = {Ms}°C (Martensite)') ax.axhline(y=Ms - 150, color='green', linestyle='--', linewidth=2, label=f'Mf = {Ms-150}°C') # CoolingCurve of example（Rapid Cooling・Slow Cooling） time_cool = np.logspace(-1, 3, 100) T_quench = 850 - 800 * (1 - np.exp(-0.05 * time_cool)) # water cooling T_slow = 850 - 820 * (1 - np.exp(-0.0005 * time_cool)) # ax.semilogx(time_cool, T_quench, 'k-', linewidth=2.5, label='Quench (Water)', alpha=0.7) ax.semilogx(time_cool, T_slow, 'k--', linewidth=2.5, label='Slow Cool (Furnace)', alpha=0.7) ax.set_xlabel('Time [s]', fontsize=13) ax.set_ylabel('Temperature [°C]', fontsize=13) ax.set_title('TTT Diagram for Eutectoid Steel (0.8% C)', fontsize=15, fontweight='bold') ax.legend(fontsize=10, loc='upper right') ax.grid(True, alpha=0.3, which='both') ax.set_xlim(0.1, 1e5) ax.set_ylim(0, 900) # A1Temperatureline ax.axhline(y=727, color='purple', linestyle=':', linewidth=1.5, alpha=0.5) ax.text(1e4, 735, 'A1 (727°C)', fontsize=10, color='purple') plt.tight_layout() plt.savefig('ttt_diagram.png', dpi=300, bbox_inches='tight') plt.show() print("=== TTT Diagram Interpretation ===") print("Water quench: Crosses below Ms → Full martensite") print("Furnace cool: Passes through pearlite region → Ferrite + Pearlite") print("Oil quench (intermediate): May produce bainite or mixed structure")

### 4.5.3 Critical Cooling Rate

The critical cooling rate is the minimum cooling rate required to obtain a complete martensitic microstructure. On the CCT diagram, it is the minimum cooling rate at which the cooling curve does not intersect the pearlite or bainite transformation curves. Steels with higher hardenability have lower critical cooling rates.

## Exercises

📝 Exercise 1: Determination of Work Hardening IndexEasy

**Problem** : In a tensile test of low carbon steel, true stress of 300 MPa was obtained at true strain of 0.1, and 420 MPa at strain of 0.3. Determine the constants K and n in the work hardening equation \\(\sigma = K \varepsilon^n\\) (assume initial yield stress \\(\sigma_0\\) is negligible).

Click to Show/Hide Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Work hardening parameter fitting
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np from scipy.optimize import curve_fit # Experimental Data strain_exp = np.array([0.1, 0.3]) stress_exp = np.array([300, 420]) # Work Hardeningmoderu（σ = K * ε^n） def work_hardening_model(strain, K, n): return K * strain**n # line Minimumtwo Parameterfit params, covariance = curve_fit(work_hardening_model, strain_exp, stress_exp, p0=[500, 0.3]) K_fit, n_fit = params print("=== Work Hardening Parameter Fitting ===") print(f"Strength coefficient K = {K_fit:.1f} MPa") print(f"Work hardening exponent n = {n_fit:.3f}") #: fitCurve and experimentvalue of Ratiocomparison stress_fit = work_hardening_model(strain_exp, K_fit, n_fit) print("\n=== Verification ===") for i in range(len(strain_exp)): print(f"ε = {strain_exp[i]:.1f}: Experimental σ = {stress_exp[i]} MPa, " f"Fitted σ = {stress_fit[i]:.1f} MPa") # numberal （2pointfromCalculation） # σ1 = K * ε1^n, σ2 = K * ε2^n # σ2/σ1 = (ε2/ε1)^n → n = log(σ2/σ1) / log(ε2/ε1) n_algebraic = np.log(stress_exp[1] / stress_exp[0]) / np.log(strain_exp[1] / strain_exp[0]) K_algebraic = stress_exp[0] / (strain_exp[0]**n_algebraic) print("\n=== Algebraic Solution (2-point method) ===") print(f"n = log({stress_exp[1]}/{stress_exp[0]}) / log({strain_exp[1]}/{strain_exp[0]}) = {n_algebraic:.3f}") print(f"K = {stress_exp[0]} / {strain_exp[0]}^{n_algebraic:.3f} = {K_algebraic:.1f} MPa") # StressPrediction（ε = 0.5 of value） strain_pred = 0.5 stress_pred = work_hardening_model(strain_pred, K_fit, n_fit) print(f"\n=== Prediction ===") print(f"At ε = {strain_pred}, predicted σ = {stress_pred:.1f} MPa")

**Expected output** :
    
    
    Strength coefficient K = 530-550 MPa Work hardening exponent n = 0.18-0.20

**Explanation** : From two data points, we can solve algebraically or use curve_fit for least-squares fitting. An n value around 0.2 is typical for moderately work-hardening materials like low carbon steel.

📝 Exercise 2: Estimation of Recrystallization TemperatureEasy

**Problem** : The melting point of copper (Cu) is 1085C (1358 K). Assuming the recrystallization temperature is approximately 0.4 times the melting point (in K), estimate the recrystallization temperature. Also, determine whether recrystallization will occur if stress relief annealing is performed at 200C.

Click to Show/Hide Solution
    
    
    # Copper of Melting Point T_m_Cu_K = 1358 # Kelvin T_m_Cu_C = 1085 # Celsius # recrystallizationTemperature of （T_rex ≈ 0.4 * T_m） T_rex_K = 0.4 * T_m_Cu_K T_rex_C = T_rex_K - 273.15 print("=== Recrystallization Temperature Estimation ===") print(f"Melting point of Cu: {T_m_Cu_K} K ({T_m_Cu_C}°C)") print(f"Estimated recrystallization temperature: {T_rex_K:.0f} K ({T_rex_C:.0f}°C)") # Stress annealingTemperature of stress_relief_temp = 200 # °C print(f"\n=== Heat Treatment at {stress_relief_temp}°C ===") if stress_relief_temp

**Expected output** :
    
    
    Estimated recrystallization temperature: 543 K (270C) At 200C: Recovery occurs, but NO recrystallization

**Explanation** : The recrystallization temperature varies by material and is typically 0.3 to 0.5 times the melting point. For copper, recrystallization occurs above approximately 270C. At 200C, only recovery (dislocation rearrangement) occurs, but new strain-free grains do not form.

📝 Exercise 3: Ms Temperature and Quenched Microstructure PredictionMedium

**Problem** : After austenitizing a carbon steel (C: 0.6 wt%, Mn: 0.8 wt%, Ni: 0, Cr: 0), it is water quenched (rapid cooling to room temperature 20C). Calculate the Ms temperature and predict the obtainable microstructure. Also, discuss how the microstructure would change if the steel were oil cooled (cooled to 60C followed by slow cooling).

Click to Show/Hide Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Ms temperature calculation and microstructure prediction
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np

    def calculate_ms_temperature(C, Mn=0, Ni=0, Cr=0):
        """
        Martensite start temperature Ms (Andrews equation)
        Ms = 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr
        """
        Ms = 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr
        return Ms

    # Chemical composition of the steel
    C_content = 0.6   # wt%
    Mn_content = 0.8  # wt%
    Ni_content = 0
    Cr_content = 0

    # Ms temperature
    Ms_temp = calculate_ms_temperature(C_content, Mn_content, Ni_content, Cr_content)

    print("=== Steel Composition ===")
    print(f"C: {C_content} wt%, Mn: {Mn_content} wt%, Ni: {Ni_content} wt%, Cr: {Cr_content} wt%")
    print(f"\n=== Martensite Start Temperature ===")
    print(f"Ms = 539 - 423×{C_content} - 30.4×{Mn_content} - 17.7×{Ni_content} - 12.1×{Cr_content}")
    print(f"Ms = {Ms_temp:.1f}°C")

    # Mf temperature estimate (empirically Ms - 215°C)
    Mf_temp = Ms_temp - 215
    print(f"Mf (estimated) ≈ {Mf_temp:.1f}°C")

    # Water quenching
    quench_temp_water = 20  # °C
    print(f"\n=== Water Quenching (to {quench_temp_water}°C) ===")
    if quench_temp_water < Ms_temp:
        martensite_fraction = 100  # Fully martensitic (simplified estimate)
        print(f"Final temperature ({quench_temp_water}°C) < Ms ({Ms_temp:.1f}°C)")
        print(f"→ Martensite transformation occurs during quenching")
        print(f"→ Expected structure: ~{martensite_fraction:.0f}% Martensite (high hardness)")
        print(f"→ Hardness: ~60-65 HRC (extremely hard but brittle)")
    else:
        print(f"Final temperature ({quench_temp_water}°C) ≥ Ms ({Ms_temp:.1f}°C)")
        print(f"→ Austenite retained (not recommended)")

    # Oil quenching
    quench_temp_oil = 60  # °C
    print(f"\n=== Oil Quenching (to {quench_temp_oil}°C, then slow cooling) ===")
    if quench_temp_oil < Ms_temp:
        # Oil cools more slowly than water, so partial pearlite/bainite transformation is possible
        print(f"Cooling rate: Slower than water quenching")
        print(f"→ Possible pearlite/bainite transformation at higher temperatures")
        print(f"→ Final structure: Martensite + Bainite (mixed)")
        print(f"→ Hardness: ~50-55 HRC (slightly lower than water quench)")
        print(f"→ Better toughness than full martensite")
    else:
        print(f"Final temperature ({quench_temp_oil}°C) ≥ Ms ({Ms_temp:.1f}°C)")
        print(f"→ Austenite may transform to pearlite/bainite")

    # Recommended heat treatment
    print(f"\n=== Recommended Heat Treatment ===")
    print(f"1. Austenitizing: 820-850°C, 30-60 min")
    print(f"2. Quenching: Oil quench (to avoid cracking)")
    print(f"3. Tempering: 200-400°C, 1-2 hours")
    print(f"   → Reduces brittleness, improves toughness")
    print(f"   → Hardness after tempering: ~45-55 HRC")

    # Effect of alloying elements on the Ms temperature
    print(f"\n=== Effect of Alloying Elements on Ms ===")
    elements = {'C': -423, 'Mn': -30.4, 'Ni': -17.7, 'Cr': -12.1}
    for elem, coeff in elements.items():
        print(f"{elem}: {coeff:.1f}°C per wt% (negative = lowers Ms)")

**Expected output** :
    
    
    Ms = 261C, Mf = 46C Water quenching: ~100% Martensite (high hardness, brittle) Oil quenching: Martensite + Bainite (mixed, better toughness)

**Explanation** : For steel with 0.6% C, the Ms temperature is approximately 260C, and cooling to room temperature results in martensite formation. Water quenching produces complete martensite due to rapid cooling, while oil quenching has a slower cooling rate and may allow some bainite formation. After quenching, tempering is important to improve toughness.

📝 Exercise 4: Calculation of Recrystallization Time Using JMA EquationMedium

**Problem** : During annealing at 600C, recrystallization follows the JMA equation \\(X = 1 - \exp(-kt^n)\\), with \\(k = 0.01\\) (1/min^n) and \\(n = 2.5\\). Calculate the time \\(t_{50}\\) required to achieve 50% recrystallization. Also determine the time \\(t_{90}\\) for 90% recrystallization.

Click to Show/Hide Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: JMA equation recrystallization time calculation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np import matplotlib.pyplot as plt def calculate_jma_time(X_target, k, n): """ JMAequationfromta gettransformationrate dotime Calculation X = 1 - exp(-k*t^n) → t = (ln(1/(1-X)) / k)^(1/n) Parameters: ----------- X_target: float transformationrate（0～1） k: float rateConstant [1/min^n] n: float AvramiIndex Returns: -------- t: float time [min] """ if X_target>= 1.0: return np.inf t = (np.log(1 / (1 - X_target)) / k)**(1 / n) return t # Parameter k = 0.01 # 1/min^n n = 2.5 # AvramiIndex # 50%recrystallizationtime X_50 = 0.5 t_50 = calculate_jma_time(X_50, k, n) # 90%recrystallizationtime X_90 = 0.9 t_90 = calculate_jma_time(X_90, k, n) print("=== JMA Recrystallization Kinetics ===") print(f"Parameters: k = {k} (1/min^{n}), n = {n}") print(f"\n=== Calculation Results ===") print(f"50% recrystallization time (t_50):") print(f" t_50 = (ln(1/(1-{X_50})) / {k})^(1/{n})") print(f" t_50 = (ln(2) / {k})^(1/{n})") print(f" t_50 = {t_50:.2f} minutes ({t_50/60:.2f} hours)") print(f"\n90% recrystallization time (t_90):") print(f" t_90 = (ln(1/(1-{X_90})) / {k})^(1/{n})") print(f" t_90 = (ln(10) / {k})^(1/{n})") print(f" t_90 = {t_90:.2f} minutes ({t_90/60:.2f} hours)") # time of Ratiocomparison ratio = t_90 / t_50 print(f"\n=== Time Ratio ===") print(f"t_90 / t_50 = {ratio:.2f}") print(f"→ 90% recrystallization takes {ratio:.2f}× longer than 50%") # recrystallizationCurve of Plot time = np.linspace(0, t_90 * 1.5, 200) X = 1 - np.exp(-k * time**n) plt.figure(figsize=(10, 6)) plt.plot(time, X * 100, 'b-', linewidth=2, label='Recrystallization Curve') plt.axhline(y=50, color='green', linestyle='--', linewidth=1.5, label=f't_50 = {t_50:.1f} min') plt.axvline(x=t_50, color='green', linestyle='--', linewidth=1.5, alpha=0.5) plt.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label=f't_90 = {t_90:.1f} min') plt.axvline(x=t_90, color='red', linestyle='--', linewidth=1.5, alpha=0.5) plt.plot(t_50, 50, 'go', markersize=10) plt.plot(t_90, 90, 'ro', markersize=10) plt.xlabel('Time [minutes]', fontsize=12) plt.ylabel('Recrystallized Fraction [%]', fontsize=12) plt.title(f'JMA Recrystallization Curve (k={k}, n={n})', fontsize=14) plt.legend(fontsize=11) plt.grid(True, alpha=0.3) plt.xlim(0, time[-1]) plt.ylim(0, 100) plt.tight_layout() plt.savefig('jma_calculation.png', dpi=300, bbox_inches='tight') plt.show() # differenttransformationrate of timeCalculation print("\n=== Time for Various Recrystallization Fractions ===") X_values = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] for X in X_values: t = calculate_jma_time(X, k, n) print(f"X = {X*100:5.1f}% → t = {t:7.2f} min ({t/60:5.2f} hours)")

**Expected output** :
    
    
    t_50 = 5.28 minutes (0.09 hours) t_90 = 13.00 minutes (0.22 hours) t_90 / t_50 = 2.46

**Explanation** : To solve for time from the JMA equation, use \\(t = [\ln(1/(1-X)) / k]^{1/n}\\). The time ratio from 50% transformation (\\(\ln 2\\)) to 90% transformation (\\(\ln 10\\)) depends on the Avrami index. For n = 2.5, it takes approximately 2.46 times longer.

📝 Exercise 5: Optimization of Age Hardening ConditionsMedium

**Problem** : In age hardening treatment of an Al-Cu alloy (2024 series), aging treatment was performed at three temperatures: 150C, 175C, and 200C. From the following data, determine the maximum hardness and time to reach it, and recommend the optimal aging condition.

  * 150C: 10h = 120 HV, 50h = 140 HV, 200h = 135 HV
  * 175C: 3h = 120 HV, 15h = 135 HV, 50h = 125 HV
  * 200C: 1h = 115 HV, 5h = 125 HV, 20h = 115 HV

Click to Show/Hide Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Age hardening condition optimization
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Experimental data
    data = {
        150: {'time': [10, 50, 200], 'hardness': [120, 140, 135]},
        175: {'time': [3, 15, 50], 'hardness': [120, 135, 125]},
        200: {'time': [1, 5, 20], 'hardness': [115, 125, 115]}
    }

    # Determine the maximum hardness and the peak time
    print("=== Age Hardening Analysis ===")
    print(f"{'Temp [°C]':<12} {'Max Hardness [HV]':<20} {'Peak Time [h]':<15}")
    print("-" * 50)

    results = {}
    for temp, values in data.items():
        times = np.array(values['time'])
        hardness = np.array(values['hardness'])

        max_hardness = np.max(hardness)
        peak_time = times[np.argmax(hardness)]

        results[temp] = {'max_hardness': max_hardness, 'peak_time': peak_time}

        print(f"{temp:<12} {max_hardness:<20} {peak_time:<15}")

    # Determine the recommended conditions
    print("\n=== Recommendation ===")
    best_temp = max(results, key=lambda x: results[x]['max_hardness'])
    print(f"Best temperature: {best_temp}°C")
    print(f"Maximum hardness: {results[best_temp]['max_hardness']} HV")
    print(f"Time to peak: {results[best_temp]['peak_time']} hours")

    # Practical recommendation (productivity also taken into account)
    print("\n=== Practical Recommendation (T6 Treatment) ===")
    print(f"Option 1 (Maximum hardness): {best_temp}°C, {results[best_temp]['peak_time']} hours")
    print(f"  → Highest hardness ({results[best_temp]['max_hardness']} HV)")
    print(f"  → Suitable for maximum strength applications")

    print(f"\nOption 2 (Balanced): 175°C, 15 hours")
    print(f"  → Good hardness (135 HV, ~96% of max)")
    print(f"  → Shorter processing time (3.3× faster than 150°C)")
    print(f"  → Better productivity")

    print(f"\nOption 3 (Rapid): 200°C, 5 hours")
    print(f"  → Moderate hardness (125 HV, ~89% of max)")
    print(f"  → Very fast processing (10× faster than 150°C)")
    print(f"  → Suitable for rapid production")

    # Plot the age hardening curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: experimental data and interpolated curves
    colors = {150: 'blue', 175: 'green', 200: 'red'}
    for temp, values in data.items():
        times = np.array(values['time'])
        hardness = np.array(values['hardness'])

        # Interpolation (cubic spline)
        time_interp = np.linspace(times[0], times[-1], 100)
        f_interp = interp1d(times, hardness, kind='quadratic', fill_value='extrapolate')
        hardness_interp = f_interp(time_interp)

        ax1.plot(time_interp, hardness_interp, color=colors[temp], linewidth=2,
                 label=f'{temp}°C', alpha=0.7)
        ax1.plot(times, hardness, 'o', color=colors[temp], markersize=8)

        # Mark the peak point
        peak_idx = np.argmax(hardness)
        ax1.plot(times[peak_idx], hardness[peak_idx], '*', color=colors[temp],
                 markersize=15, markeredgecolor='black', markeredgewidth=1)

    ax1.set_xlabel('Aging Time [hours]', fontsize=12)
    ax1.set_ylabel('Hardness [HV]', fontsize=12)
    ax1.set_title('Age Hardening Curves for Al-Cu Alloy', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Right panel: relation between maximum hardness and the time to reach it
    temps_list = sorted(results.keys())
    max_hardness_list = [results[t]['max_hardness'] for t in temps_list]
    peak_time_list = [results[t]['peak_time'] for t in temps_list]

    ax2_twin = ax2.twinx()
    ax2.plot(temps_list, max_hardness_list, 'bo-', linewidth=2, markersize=10,
             label='Max Hardness')
    ax2_twin.plot(temps_list, peak_time_list, 'rs-', linewidth=2, markersize=10,
                  label='Peak Time')

    ax2.set_xlabel('Aging Temperature [°C]', fontsize=12)
    ax2.set_ylabel('Maximum Hardness [HV]', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Time to Peak [hours]', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Temperature vs. Max Hardness & Peak Time', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('age_hardening_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Evaluate over-aging
    print("\n=== Over-aging Analysis ===")
    for temp in [150, 175, 200]:
        hardness_data = data[temp]['hardness']
        if hardness_data[-1] < max(hardness_data):
            loss = max(hardness_data) - hardness_data[-1]
            loss_pct = (loss / max(hardness_data)) * 100
            print(f"{temp}°C: Over-aging detected (loss = {loss} HV, {loss_pct:.1f}%)")
        else:
            print(f"{temp}°C: No over-aging in observed time range")

**Expected output** :
    
    
    Best temperature: 150C (Max hardness: 140 HV at 50 hours) Practical recommendation: 175C, 15 hours (135 HV, faster processing) Over-aging detected at all temperatures (hardness decreases after peak)

**Explanation** : At lower temperature (150C), maximum hardness is highest but requires long time. For practical purposes, 175C for 15 hours offers a good balance between hardness (135 HV) and productivity. Holding for too long results in overaging and decreased hardness, so proper aging time control is important.

📝 Exercise 6: Microstructure Prediction Using TTT DiagramHard

**Problem** : For eutectoid steel (0.8% C) cooled from 850C, predict the resulting microstructure under the following cooling conditions. On the TTT diagram, assume the pearlite transformation nose is at 550C/1 second, the bainite transformation nose is at 350C/10 seconds, and Ms = 220C.  
(a) Cool to 550C in 1 second, then hold at 550C for 10 seconds  
(b) Rapid cool to 200C in 0.1 second  
(c) Cool to 350C in 10 seconds, then hold at 350C for 100 seconds

Click to Show/Hide Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: TTT diagram microstructure prediction
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    # TTT diagram parameters
    Ms_temp = 220  # Martensite start temperature [°C]
    pearlite_nose = {'temp': 550, 'time': 1}      # Pearlite nose [°C, s]
    bainite_nose = {'temp': 350, 'time': 10}      # Bainite nose [°C, s]

    # Cooling scenarios
    scenarios = {
        'a': {'cool_time': 1, 'cool_to': 550, 'hold_temp': 550, 'hold_time': 10},
        'b': {'cool_time': 0.1, 'cool_to': 200, 'hold_temp': None, 'hold_time': 0},
        'c': {'cool_time': 10, 'cool_to': 350, 'hold_temp': 350, 'hold_time': 100}
    }

    def predict_structure(scenario_name, scenario):
        """
        Predict the microstructure from the TTT diagram
        """
        print(f"\n=== Scenario {scenario_name.upper()} ===")
        print(f"Cooling: {scenario['cool_time']}s to {scenario['cool_to']}°C")
        if scenario['hold_temp']:
            print(f"Holding: {scenario['hold_temp']}°C for {scenario['hold_time']}s")

        cool_time = scenario['cool_time']
        cool_to = scenario['cool_to']
        hold_temp = scenario['hold_temp']
        hold_time = scenario['hold_time']

        # Transformation during cooling
        if cool_to == pearlite_nose['temp'] and cool_time >= pearlite_nose['time']:
            print(f"→ Crosses pearlite nose during cooling")
            print(f"→ Partial pearlite transformation likely")
            partial_pearlite = True
        else:
            partial_pearlite = False

        # Transformation during isothermal holding
        if hold_temp == pearlite_nose['temp'] and hold_time >= pearlite_nose['time']:
            print(f"→ Isothermal holding at pearlite nose temperature")
            print(f"→ COMPLETE pearlite transformation")
            structure = "100% Pearlite"
            hardness = "~20-25 HRC (soft, ductile)"

        elif hold_temp == bainite_nose['temp'] and hold_time >= bainite_nose['time']:
            print(f"→ Isothermal holding at bainite nose temperature")
            print(f"→ COMPLETE bainite transformation")
            structure = "100% Bainite"
            hardness = "~40-50 HRC (moderate hardness, good toughness)"

        elif cool_to < Ms_temp and hold_temp is None:
            print(f"→ Rapid cooling below Ms ({Ms_temp}°C)")
            print(f"→ Martensite transformation")
            structure = "~100% Martensite"
            hardness = "~63-65 HRC (very hard, brittle)"

        else:
            print(f"→ Intermediate cooling")
            if partial_pearlite:
                structure = "Pearlite + Martensite (mixed)"
                hardness = "~35-45 HRC"
            else:
                structure = "Bainite + Martensite (mixed)"
                hardness = "~50-55 HRC"

        print(f"\n【Predicted Structure】: {structure}")
        print(f"【Expected Hardness】: {hardness}")

        return structure, hardness

    # Analyze each scenario
    results = {}
    for name, scenario in scenarios.items():
        structure, hardness = predict_structure(name, scenario)
        results[name] = {'structure': structure, 'hardness': hardness}

    # Comparison table of results
    print("\n" + "="*70)
    print("=== SUMMARY: Structure Prediction Results ===")
    print("="*70)
    print(f"{'Scenario':<12} {'Structure':<35} {'Hardness':<20}")
    print("-"*70)
    for name in ['a', 'b', 'c']:
        print(f"{name.upper():<12} {results[name]['structure']:<35} {results[name]['hardness']:<20}")

    # Simplified TTT diagram plot (conceptual)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Time axis (logarithmic)
    time = np.logspace(-1, 4, 200)

    # Pearlite transformation curve (simplified C-curve)
    P_start = 550 + 150 * (np.log10(time / pearlite_nose['time']))**2
    P_start = np.clip(P_start, 400, 727)

    # Bainite transformation curve
    B_start = 350 + 80 * (np.log10(time / bainite_nose['time']))**2
    B_start = np.clip(B_start, 250, 500)

    ax.semilogx(time, P_start, 'r-', linewidth=2, label='Pearlite Start')
    ax.semilogx(time, B_start, 'b-', linewidth=2, label='Bainite Start')
    ax.axhline(y=Ms_temp, color='green', linestyle='-', linewidth=2.5, label=f'Ms = {Ms_temp}°C')

    # Plot the cooling curve for each scenario
    # Scenario A
    cool_time_a = scenarios['a']['cool_time']
    hold_time_a = scenarios['a']['hold_time']
    time_a = [0, cool_time_a, cool_time_a + hold_time_a]
    temp_a = [850, 550, 550]
    ax.plot(time_a, temp_a, 'ko-', linewidth=2.5, markersize=8, label='Scenario A', alpha=0.7)

    # Scenario B
    time_b = [0, scenarios['b']['cool_time']]
    temp_b = [850, 200]
    ax.plot(time_b, temp_b, 'mo-', linewidth=2.5, markersize=8, label='Scenario B', alpha=0.7)

    # Scenario C
    cool_time_c = scenarios['c']['cool_time']
    hold_time_c = scenarios['c']['hold_time']
    time_c = [0, cool_time_c, cool_time_c + hold_time_c]
    temp_c = [850, 350, 350]
    ax.plot(time_c, temp_c, 'co-', linewidth=2.5, markersize=8, label='Scenario C', alpha=0.7)

    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Temperature [°C]', fontsize=13)
    ax.set_title('TTT Diagram and Cooling Paths for Eutectoid Steel', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0, 900)

    plt.tight_layout()
    plt.savefig('ttt_scenarios.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Recommended applications
    print("\n=== Recommended Applications ===")
    print("Scenario A (Pearlite): Rails, wire ropes (high ductility, wear resistance)")
    print("Scenario B (Martensite): Cutting tools, bearings (maximum hardness)")
    print("Scenario C (Bainite): Gears, springs (balanced hardness and toughness)")

**Expected output** :
    
    
    Scenario A: 100% Pearlite (~20-25 HRC) Scenario B: ~100% Martensite (~63-65 HRC) Scenario C: 100% Bainite (~40-50 HRC)

**Explanation** : (a) Holding at the pearlite nose temperature results in complete pearlite transformation. (b) Rapid cooling below Ms produces a martensitic microstructure. (c) Holding at the bainite nose temperature for sufficient time completes the bainite transformation. By confirming where the cooling curve crosses transformation curves on the TTT diagram, microstructure prediction is possible.

## References

**[1]** Dieter, G.E., Bacon, D. (2013). _Mechanical Metallurgy_ , SI Metric Edition. McGraw-Hill Education, pp. 289-345, 412-467.

**[2]** Porter, D.A., Easterling, K.E., Sherif, M.Y. (2009). _Phase Transformations in Metals and Alloys_ , 3rd Edition. CRC Press, pp. 1-35, 189-245, 315-378.

**[3]** Cottrell, A.H. (1953). _Dislocations and Plastic Flow in Crystals_. Oxford University Press, pp. 111-156.

**[4]** Askeland, D.R., Wright, W.J. (2015). _The Science and Engineering of Materials_ , 7th Edition. Cengage Learning, pp. 345-412, 523-589.

**[5]** Honeycombe, R.W.K., Bhadeshia, H.K.D.H. (2017). _Steels: Microstructure and Properties_ , 4th Edition. Butterworth-Heinemann, pp. 67-134, 201-267.

**[6]** Reed-Hill, R.E., Abbaschian, R. (1994). _Physical Metallurgy Principles_ , 3rd Edition. PWS Publishing, pp. 289-356, 445-512.

**[7]** Japan Institute of Metals (2015). Heat Treatment Handbook, Revised Edition. Maruzen, pp. 123-189, 267-334.

**[8]** Christian, J.W. (2002). _The Theory of Transformations in Metals and Alloys_ , 3rd Edition. Pergamon, pp. 1-45, 123-189.

## Learning Objectives Checklist

### Skills and Knowledge

#### Level 1: Basic Understanding (Knowledge)

  * Explain the types (rolling, forging, extrusion, drawing) and principles of plastic processing
  * Understand work hardening, recovery, and recrystallization phenomena
  * Distinguish types of annealing (stress relief, recrystallization, full, homogenization)
  * Understand basic quenching and tempering processes for steel and martensite transformation
  * Explain the three steps of age hardening (solution treatment, quenching, aging)
  * Understand the meaning and use of TTT and CCT diagrams

#### Level 2: Practical Skills (Application)

  * Calculate stress using the work hardening equation \\(\sigma = \sigma_0 + K \varepsilon^n\\)
  * Calculate recrystallization rate and transformation rate using the JMA equation
  * Estimate martensite start temperature using the Ms temperature calculation equation
  * Simulate cooling curves and evaluate the effects of quenching media
  * Plot age hardening curves and determine peak aging conditions
  * Predict obtainable microstructures from cooling conditions using TTT diagrams
  * Create and execute Python code for heat treatment simulations

#### Level 3: Advanced Application (Problem Solving)

  * Design processing and heat treatment conditions based on material application requirements
  * Combine multiple heat treatment processes appropriately
  * Determine work hardening parameters and recrystallization rate constants from experimental data
  * Estimate practical critical cooling rates and resulting microstructure/properties using CCT diagrams
  * Recommend optimal conditions (temperature/time) from age hardening data
  * Understand and predict heat treatment behavior for different steels and alloys
  * Analyze and optimize experimental data using Python tools

**Confirmation** : Please verify your understanding by completing Exercises 1-6 above. In particular, Exercise 5 (age hardening optimization) and Exercise 6 (microstructure prediction using TTT diagram) are important for evaluating practical problem-solving skills.

[Chapter 3: Strengthening Mechanisms](<chapter-3.html>)[Chapter 5: Python Practice](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
