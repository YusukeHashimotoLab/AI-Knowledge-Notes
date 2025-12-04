---
title: "Chapter 1: Tensile Testing Fundamentals"
chapter_title: "Chapter 1: Tensile Testing Fundamentals"
subtitle: Stress-Strain Behavior and Mechanical Properties
---

🌐 EN | 日本語 (準備中) Last sync: 2025-11-16

[Materials Science Dojo](<../index.html>) > [Mechanical Testing Introduction](<index.html>) > Chapter 1 

## 1.1 Introduction to Tensile Testing

Tensile testing is the most fundamental mechanical test used to determine material properties including strength, ductility, and elastic modulus. A specimen is subjected to controlled tension until failure while measuring force and elongation. 

**=Ğ Definition: Engineering Stress and Strain**  
**Engineering stress:** $$\sigma = \frac{F}{A_0}$$ where $F$ is applied force and $A_0$ is original cross-sectional area.  
  
**Engineering strain:** $$\epsilon = \frac{\Delta L}{L_0} = \frac{L - L_0}{L_0}$$ where $L_0$ is original length and $L$ is current length. 

### =» Code Example 1: Stress-Strain Curve Generation

Python Implementation: Engineering Stress-Strain Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def generate_stress_strain_curve(material='steel'):
        """Generate engineering stress-strain curve"""
        materials = {
            'steel': {'E': 200e3, 'yield': 250, 'uts': 400, 'fracture_strain': 0.25},
            'aluminum': {'E': 70e3, 'yield': 100, 'uts': 200, 'fracture_strain': 0.15},
            'copper': {'E': 120e3, 'yield': 70, 'uts': 220, 'fracture_strain': 0.45}
        }
        props = materials[material]
    
        # Elastic region
        strain_elastic = np.linspace(0, props['yield']/props['E'], 100)
        stress_elastic = props['E'] * strain_elastic
    
        # Plastic region
        strain_plastic = np.linspace(props['yield']/props['E'], props['fracture_strain'], 200)
        K = props['uts'] * 1.1
        n = 0.2
        stress_plastic = K * strain_plastic**n
    
        strain = np.concatenate([strain_elastic, strain_plastic])
        stress = np.concatenate([stress_elastic, stress_plastic])
        return strain, stress, props
    
    # Visualize different materials
    fig, ax = plt.subplots(figsize=(10, 6))
    for material in ['steel', 'aluminum', 'copper']:
        strain, stress, props = generate_stress_strain_curve(material)
        ax.plot(strain * 100, stress, linewidth=2, label=material.capitalize())
        yield_idx = np.argmin(np.abs(stress - props['yield']))
        ax.plot(strain[yield_idx] * 100, stress[yield_idx], 'o', markersize=8)
    
    ax.set_xlabel('Engineering Strain (%)', fontsize=12)
    ax.set_ylabel('Engineering Stress (MPa)', fontsize=12)
    ax.set_title('Engineering Stress-Strain Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

## 1.2 True Stress and True Strain

Engineering stress and strain assume constant dimensions, but materials deform during testing. True stress and strain account for instantaneous dimensions. 

**=Ğ Definition: True Stress and True Strain**  
**True stress:** $\sigma_T = \frac{F}{A} = \sigma(1 + \epsilon)$  
**True strain:** $\epsilon_T = \ln\left(\frac{L}{L_0}\right) = \ln(1 + \epsilon)$ 

### =» Code Example 2: True vs Engineering Conversion

Python Implementation: Stress-Strain Conversion
    
    
    def engineering_to_true(eng_stress, eng_strain):
        """Convert engineering to true stress-strain"""
        true_stress = eng_stress * (1 + eng_strain)
        true_strain = np.log(1 + eng_strain)
        return true_stress, true_strain
    
    strain_eng, stress_eng, _ = generate_stress_strain_curve('steel')
    stress_true, strain_true = engineering_to_true(stress_eng, strain_eng)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(strain_eng * 100, stress_eng, 'b-', linewidth=2, label='Engineering')
    ax1.set_xlabel('Engineering Strain (%)')
    ax1.set_ylabel('Engineering Stress (MPa)')
    ax1.set_title('Engineering Stress-Strain')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(strain_true * 100, stress_true, 'r-', linewidth=2, label='True')
    ax2.set_xlabel('True Strain (%)')
    ax2.set_ylabel('True Stress (MPa)')
    ax2.set_title('True Stress-Strain')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()

## 1.3 Mechanical Properties Extraction

Tensile testing provides critical mechanical properties: elastic modulus, yield strength, ultimate tensile strength, and ductility. 

**=Ğ Key Mechanical Properties**  

  * **Elastic Modulus:** $E = \frac{\sigma}{\epsilon}$ in elastic region
  * **Yield Strength:** 0.2% offset method
  * **Ultimate Tensile Strength:** Maximum engineering stress
  * **Ductility:** $\delta = \frac{L_f - L_0}{L_0} \times 100\%$
  * **Toughness:** Area under stress-strain curve

### =» Code Example 3: Mechanical Property Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    from scipy import integrate
    from scipy.stats import linregress
    
    def calculate_mechanical_properties(strain, stress):
        """Extract mechanical properties from stress-strain curve"""
        properties = {}
    
        # Elastic Modulus
        elastic_idx = int(len(strain) * 0.05)
        slope, intercept, r_value, _, _ = linregress(strain[:elastic_idx], stress[:elastic_idx])
        properties['elastic_modulus'] = slope
    
        # Yield Strength (0.2% offset)
        offset_strain = 0.002
        offset_line = slope * (strain - offset_strain) + intercept
        diff = np.abs(stress - offset_line)
        yield_idx = elastic_idx + np.argmin(diff[elastic_idx:])
        properties['yield_strength'] = stress[yield_idx]
    
        # Ultimate Tensile Strength
        uts_idx = np.argmax(stress)
        properties['ultimate_tensile_strength'] = stress[uts_idx]
        properties['uniform_elongation'] = strain[uts_idx]
    
        # Ductility
        properties['elongation_percent'] = strain[-1] * 100
    
        # Toughness
        properties['toughness'] = integrate.trapz(stress, strain)
    
        return properties
    
    strain, stress, _ = generate_stress_strain_curve('steel')
    props = calculate_mechanical_properties(strain, stress)
    
    print(f"Elastic Modulus: {props['elastic_modulus']/1000:.1f} GPa")
    print(f"Yield Strength: {props['yield_strength']:.1f} MPa")
    print(f"UTS: {props['ultimate_tensile_strength']:.1f} MPa")
    print(f"Elongation: {props['elongation_percent']:.1f}%")
    print(f"Toughness: {props['toughness']:.1f} MJ/m³")

## 1.4 Testing Standards

ASTM E8 and ISO 6892 provide standardized procedures ensuring reproducibility. Key requirements include specimen geometry, strain rate, and environmental conditions. 

### =» Code Example 4: ASTM E8 Specimen Design
    
    
    class TensileSpecimen:
        """ASTM E8 tensile specimen calculator"""
    
        def calculate_dimensions(self, diameter=12.5):
            """Calculate specimen dimensions per ASTM E8"""
            area = np.pi * (diameter/2)**2
            gauge_length = 4 * np.sqrt(area)
    
            return {
                'diameter': diameter,
                'area': area,
                'gauge_length': gauge_length,
                'total_length': gauge_length * 1.5 + 60
            }
    
        def calculate_test_speed(self, gauge_length, strain_rate=0.005):
            """Calculate crosshead speed"""
            return strain_rate * gauge_length
    
    specimen = TensileSpecimen()
    dims = specimen.calculate_dimensions(diameter=12.5)
    speed = specimen.calculate_test_speed(dims['gauge_length'])
    
    print(f"Gauge Length: {dims['gauge_length']:.2f} mm")
    print(f"Test Speed: {speed:.3f} mm/min")

## 1.5 Strain Hardening

During plastic deformation, flow stress increases with strain (strain hardening), described by the Hollomon equation. 

**=Ğ Hollomon Equation:** $\sigma_T = K \epsilon_T^n$  
where $K$ is strength coefficient and $n$ is strain hardening exponent. 

### =» Code Example 5: Hollomon Equation Fitting
    
    
    def fit_hollomon_equation(true_strain, true_stress):
        """Fit Hollomon equation to data"""
        plastic_idx = 100
        strain_plastic = true_strain[plastic_idx:]
        stress_plastic = true_stress[plastic_idx:]
    
        log_strain = np.log(strain_plastic)
        log_stress = np.log(stress_plastic)
    
        n, log_K, r_value, _, _ = linregress(log_strain, log_stress)
        K = np.exp(log_K)
    
        return K, n, r_value**2
    
    strain_eng, stress_eng, _ = generate_stress_strain_curve('steel')
    stress_true, strain_true = engineering_to_true(stress_eng, strain_eng)
    K, n, R2 = fit_hollomon_equation(strain_true, stress_true)
    
    print(f"Hollomon: Ã = {K:.1f} * µ^{n:.3f}")
    print(f"R² = {R2:.4f}")
    print(f"Strain hardening exponent: n = {n:.3f}")

## 1.6 Temperature Effects

Mechanical properties vary with temperature. Higher temperatures reduce strength and increase ductility. 

### =» Code Example 6: Temperature Dependence
    
    
    def temperature_dependent_properties(T, T_ref=293):
        """Calculate temperature-dependent yield strength"""
        R = 8.314
        Q = 50000
        sigma_ref = 250
    
        sigma_y = sigma_ref * np.exp(Q/R * (1/T - 1/T_ref))
        E = 200e3 * (1 - 0.0005 * (T - T_ref))
        elongation = 25 * (T / T_ref)**0.5
    
        return sigma_y, E, elongation
    
    temperatures = np.linspace(200, 800, 100)
    results = [temperature_dependent_properties(T) for T in temperatures]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures - 273, [r[0] for r in results], label='Yield Strength')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Yield Strength (MPa)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

## 1.7 Necking and Instability

Necking occurs at UTS when localized deformation begins. The Considère criterion predicts necking onset. 

**=Ğ Considère Criterion:** Necking when $\frac{d\sigma_T}{d\epsilon_T} = \sigma_T$  
For Hollomon equation: $\epsilon_T^{neck} = n$ 

### =» Code Example 7: Necking Prediction
    
    
    def predict_necking(K, n):
        """Predict necking using Considère criterion"""
        necking_strain_true = n
        necking_stress_true = K * necking_strain_true**n
    
        necking_strain_eng = np.exp(necking_strain_true) - 1
        necking_stress_eng = necking_stress_true / (1 + necking_strain_eng)
    
        return {
            'true_strain': necking_strain_true,
            'true_stress': necking_stress_true,
            'eng_strain': necking_strain_eng,
            'eng_stress': necking_stress_eng
        }
    
    K, n = 550, 0.22
    necking = predict_necking(K, n)
    print(f"Necking at true strain: {necking['true_strain']:.3f}")
    print(f"Engineering strain: {necking['eng_strain']*100:.1f}%")

## =İ Chapter Exercises

** Exercises**

  1. Calculate engineering and true stress for a specimen with original diameter 12.5 mm loaded to 62 kN with diameter 10.8 mm.
  2. Extract elastic modulus, yield strength, and UTS from stress-strain data. Calculate toughness.
  3. Fit Hollomon equation to plastic region and predict necking strain.
  4. Design ASTM E8 rectangular specimen for 2 mm thick sheet.
  5. Analyze property changes from 25°C to 400°C for high-temperature application.

## Summary

  * Tensile testing determines fundamental mechanical properties
  * Engineering vs true stress-strain: original vs instantaneous dimensions
  * Key properties: E, yield strength, UTS, ductility, toughness
  * ASTM E8/ISO 6892 standardize testing procedures
  * Hollomon equation describes strain hardening: Ã = Kµ^n
  * Temperature and strain rate affect properties significantly
  * Necking predicted by Considère criterion

[� Series Overview](<index.html>) [Chapter 2: Hardness Testing ’](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
