---
title: "Chapter 3: Creep and Stress Relaxation"
chapter_title: "Chapter 3: Creep and Stress Relaxation"
subtitle: Time-Dependent Deformation Behavior
---

🌐 EN | 日本語 (準備中) Last sync: 2025-11-16

[Materials Science Dojo](<../index.html>) > [Mechanical Testing Introduction](<index.html>) > Chapter 3 

## 3.1 Creep Fundamentals

Creep is time-dependent plastic deformation under constant stress at elevated temperature. Critical for high-temperature applications like turbines and reactors.

**📐 Creep Strain Rate:** $$\dot{\epsilon} = A\sigma^n \exp\left(-\frac{Q}{RT}\right)$$ where $A$ is constant, $n$ is stress exponent, $Q$ is activation energy. 

### 💻 Code Example 1: Creep Curve Generation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def creep_curve(time, stress, temperature, Q=200000, A=1e-15, n=5):
        """Generate creep strain vs time curve"""
        R = 8.314
        T = temperature + 273
        
        # Primary creep
        t1 = time[time <= 100]
        eps1 = 0.01 * np.sqrt(t1)
        
        # Secondary creep (steady-state)
        strain_rate_ss = A * stress**n * np.exp(-Q/(R*T))
        t2 = time[(time > 100) & (time <= 900)]
        eps2 = eps1[-1] + strain_rate_ss * (t2 - 100)
        
        # Tertiary creep
        t3 = time[time > 900]
        eps3 = eps2[-1] + strain_rate_ss * (t3 - 900) * np.exp((t3-900)/100)
        
        strain = np.concatenate([eps1, eps2, eps3])
        return strain
    
    time = np.linspace(0, 1000, 500)
    strain = creep_curve(time, stress=100, temperature=600)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, strain*100, 'b-', linewidth=2)
    plt.axvline(100, color='r', linestyle='--', alpha=0.5, label='Primary→Secondary')
    plt.axvline(900, color='g', linestyle='--', alpha=0.5, label='Secondary→Tertiary')
    plt.xlabel('Time (hours)')
    plt.ylabel('Creep Strain (%)')
    plt.title('Creep Curve: Three Stages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

## 3.2 Larson-Miller Parameter

Larson-Miller parameter enables extrapolation of short-term creep data to long-term performance.

**📐 Larson-Miller Parameter:** $$P_{LM} = T(C + \log t_r)$$ where $T$ is temperature (K), $t_r$ is rupture time (hours), $C$ ≈ 20. 

### 💻 Code Example 2: Larson-Miller Analysis
    
    
    def larson_miller_parameter(temperature, rupture_time, C=20):
        """Calculate Larson-Miller parameter"""
        T = temperature + 273  # Convert to K
        P_LM = T * (C + np.log10(rupture_time))
        return P_LM
    
    def predict_life(stress, P_LM, temperature, C=20):
        """Predict rupture time from P_LM"""
        T = temperature + 273
        log_t = (P_LM / T) - C
        t_r = 10**log_t
        return t_r
    
    # Example
    T1, t1 = 650, 1000
    P_LM = larson_miller_parameter(T1, t1)
    t2_predicted = predict_life(stress=100, P_LM=P_LM, temperature=600)
    
    print(f"Larson-Miller Parameter: {P_LM:.0f}")
    print(f"Predicted life at 600°C: {t2_predicted:.1f} hours")

## 3.3 Stress Relaxation

Stress relaxation is decrease in stress under constant strain, opposite of creep.

**📐 Stress Relaxation:** $$\sigma(t) = \sigma_0 \exp\left(-\frac{t}{\tau}\right)$$ where $\tau$ is relaxation time constant. 

### 💻 Code Example 3: Stress Relaxation Modeling
    
    
    def stress_relaxation(time, sigma_0, tau):
        """Model stress relaxation"""
        return sigma_0 * np.exp(-time / tau)
    
    time = np.linspace(0, 500, 200)
    sigma = stress_relaxation(time, sigma_0=300, tau=100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, sigma, 'b-', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress Relaxation Curve')
    plt.grid(True, alpha=0.3)
    plt.show()

## 3.4 Creep Testing Standards

ASTM E139 specifies procedures for creep and creep-rupture testing.

### 💻 Code Example 4: Creep Test Analysis
    
    
    class CreepTest:
        """Creep test data analysis"""
        
        def minimum_creep_rate(self, time, strain):
            """Find minimum (steady-state) creep rate"""
            strain_rate = np.gradient(strain, time)
            min_idx = np.argmin(strain_rate[100:]) + 100
            return strain_rate[min_idx]
        
        def time_to_rupture(self, time, strain, failure_strain=0.20):
            """Estimate time to rupture"""
            idx = np.argmin(np.abs(strain - failure_strain))
            return time[idx]
    
    test = CreepTest()
    time = np.linspace(0, 1000, 500)
    strain = creep_curve(time, 100, 600)
    
    mcr = test.minimum_creep_rate(time, strain)
    ttr = test.time_to_rupture(time, strain)
    
    print(f"Minimum Creep Rate: {mcr:.2e} /hour")
    print(f"Time to Rupture: {ttr:.1f} hours")

## 3.5 Temperature and Stress Effects

Creep rate increases exponentially with temperature and as power law with stress.

### 💻 Code Example 5: Parametric Study
    
    
    temperatures = [500, 550, 600, 650]
    stresses = [50, 75, 100, 125]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature effect
    for T in temperatures:
        strain = creep_curve(time, stress=100, temperature=T)
        ax1.plot(time, strain*100, linewidth=2, label=f'{T}°C')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Creep Strain (%)')
    ax1.set_title('Temperature Effect on Creep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stress effect
    for sigma in stresses:
        strain = creep_curve(time, stress=sigma, temperature=600)
        ax2.plot(time, strain*100, linewidth=2, label=f'{sigma} MPa')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Creep Strain (%)')
    ax2.set_title('Stress Effect on Creep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 3.6 Creep Damage and Life Prediction

Creep damage accumulates over time, leading to eventual failure. Damage models predict remaining life.

### 💻 Code Example 6: Damage Accumulation
    
    
    def monkman_grant(time_to_rupture, min_creep_rate, m=1, C=0.05):
        """Monkman-Grant relationship"""
        # MCR * t_r^m = C
        predicted_tr = (C / min_creep_rate)**(1/m)
        return predicted_tr
    
    mcr = 1e-5
    predicted_life = monkman_grant(None, mcr)
    print(f"Predicted rupture life: {predicted_life:.1f} hours")

## 3.7 Applications and Design

Creep considerations critical for gas turbines, nuclear reactors, and high-temperature pipelines.

### 💻 Code Example 7: Design Allowable Stress
    
    
    def design_allowable_stress(temperature, design_life=100000):
        """Calculate allowable stress for design life"""
        # Use simplified master curve
        T = temperature + 273
        sigma_allow = 500 * np.exp(-0.003 * temperature) * (design_life / 100000)**(-0.15)
        return sigma_allow
    
    temps = np.linspace(400, 700, 50)
    sigma_allow = [design_allowable_stress(T) for T in temps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temps, sigma_allow, 'b-', linewidth=2)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Allowable Stress (MPa)')
    plt.title('Design Allowable Stress vs Temperature (100,000 hour life)')
    plt.grid(True, alpha=0.3)
    plt.show()

## 📝 Chapter Exercises

**✏️ Exercises**

  1. Calculate Larson-Miller parameter for creep rupture at 650°C after 5000 hours.
  2. Predict rupture life at 600°C using P_LM from exercise 1.
  3. Analyze creep curve to determine minimum creep rate and time to 5% strain.
  4. Compare creep rates at 550°C and 650°C for same stress using Arrhenius equation.
  5. Design allowable stress for 10-year service life at 600°C.

## Summary

  * Creep is time-dependent plastic deformation at elevated temperature
  * Three stages: primary (decreasing rate), secondary (constant rate), tertiary (accelerating rate)
  * Creep rate: $\dot{\epsilon} = A\sigma^n \exp(-Q/RT)$
  * Larson-Miller parameter enables life prediction from short-term data
  * Stress relaxation is stress decrease under constant strain
  * Critical for high-temperature applications: turbines, reactors, pipelines
  * Design must account for creep to ensure long-term structural integrity

[← Chapter 2: Hardness Testing](<chapter-2.html>) [Chapter 4: Fatigue & Fracture →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
