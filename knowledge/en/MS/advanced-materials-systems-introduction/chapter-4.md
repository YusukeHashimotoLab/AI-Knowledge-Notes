---
title: "Chapter 4: Energy Materials"
chapter_title: "Chapter 4: Energy Materials"
subtitle: Lithium-Ion Batteries, Fuel Cells, Solar Cells - Design Principles for High Performance
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
---

[AI Terakoya Top](<../index.html>):[Materials Science](<../../index.html>):[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>):Chapter 4

🌐 EN | [🇯🇵 JP](<../../../jp/MS/advanced-materials-systems-introduction/chapter-4.html>) | Last sync: 2025-11-16

## Learning Objectives

After completing this chapter, you will be able to explain the following:

### Basic Understanding

  * Classification of energy conversion and storage devices and their key performance metrics (energy density, power density, conversion efficiency)
  * The cathode and anode materials of the lithium-ion battery (LIB) and the principles behind calculating theoretical capacity
  * The electromotive force of a fuel cell (Nernst equation) and its three overpotentials (activation, ohmic, concentration)
  * The pn-junction operation of a solar cell, the Shockley-Queisser limit, the IV characteristics, and the fill factor (FF)

### Practical Skills

  * Compute the theoretical gravimetric capacity (mAh/g) of various electrode materials from Faraday's law in Python
  * Model the polarization curve of a fuel cell and locate the maximum power point
  * Compute the IV curve, maximum power point, fill factor, and conversion efficiency of a solar cell with the single-diode model
  * Quantitatively evaluate the relationship between material composition and performance metrics to select materials for a given application

### Applied Competence

  * Design the optimal battery material system from application requirements (driving range, power, lifetime)
  * Predict how fuel-cell operating conditions (temperature, pressure) affect performance
  * Evaluate the trade-off between the band-gap choice and efficiency of a solar cell
  * Understand the technical challenges of next-generation energy materials (all-solid-state batteries, perovskite solar cells)

## 4.1 Fundamentals of Energy Materials

### 4.1.1 Classification of Energy Conversion and Storage Devices

Energy materials are **materials that carry the function of interconverting or storing chemical, optical, and electrical energy**. Toward a carbon-neutral society, these materials underpin key technologies such as electric vehicles, stationary energy storage, and renewable power generation. This chapter treats three representative devices:

  * **Lithium-ion battery (LIB)** : stores chemical energy <=> electrical energy (secondary battery)
  * **Fuel cell** : continuously converts chemical energy to electrical energy (power-generating device)
  * **Solar cell** : directly converts optical energy to electrical energy (photovoltaic device)

    
    
    flowchart LR
        A[Chemical energy] <--> B[Lithium-ion battery  
    storage]
        A --> C[Fuel cell  
    continuous conversion]
        D[Optical energy] --> E[Solar cell  
    photovoltaic]
        B --> F[Electrical energy]
        C --> F
        E --> F
    
        style A fill:#e3f2fd
        style D fill:#fff9c4
        style B fill:#e8f5e9
        style C fill:#fff3e0
        style E fill:#fce4ec
        style F fill:#f3e5f5
            

### 4.1.2 Key Performance Metrics

To compare device performance quantitatively, we use the following metrics. They define the target values that guide material design.

#### Energy Density and Power Density

Gravimetric energy density [Wh/kg] = capacity [Ah/kg] × average voltage [V] 

Energy density expresses "how much can be stored" (driving range), while power density [W/kg] expresses "how fast it can be delivered" (acceleration). The two are often in a trade-off relationship and are visualized in a Ragone plot.

#### Conversion Efficiency

Conversion efficiency η = useful energy extracted ÷ input energy × 100 [%] 

Device | Key metric | Representative value  
---|---|---  
Lithium-ion battery (LIB)| Gravimetric energy density| 150-260 Wh/kg (cell)  
Polymer electrolyte fuel cell (PEFC)| Power density| 0.5-1.2 W/cm²  
Crystalline-silicon solar cell| Conversion efficiency| 20-27 % (production module)  
  
**💡 Importance of Energy Density**

In electric vehicles, the gravimetric energy density of the battery pack directly determines the driving range. Compared with gasoline (about 12,000 Wh/kg), current LIBs reach only about 250 Wh/kg (around 150 Wh/kg at pack level). To close this gap, research on next-generation materials such as all-solid-state batteries and lithium-metal anodes is being pursued vigorously.

## 4.2 Lithium-Ion Batteries

### 4.2.1 Operating Principle

The lithium-ion battery (LIB) is **a secondary battery that charges and discharges by shuttling lithium ions (Li⁺) between the cathode and the anode**. On discharge, Li⁺ migrates from the anode to the cathode while electrons flow through the external circuit (the reverse occurs on charge). This "rocking-chair" operation gives high reversibility and long life.

Cathode (discharge): Li₁₋ₓCoO₂ + xLi⁺ + xe⁻ → LiCoO₂ 

Anode (discharge): LiₓC₆ → xLi⁺ + xe⁻ + 6C 

### 4.2.2 Cathode Materials

The cathode material is the most critical component, determining the cell's voltage, capacity, safety, and cost. The representative materials are compared below.

Material | Abbreviation | Average voltage | Characteristics  
---|---|---|---  
LiCoO₂| LCO| 3.9 V| High energy density but expensive Co and thermally unstable  
LiFePO₄| LFP| 3.4 V| Excellent safety and lifetime, low cost, somewhat lower capacity  
LiNi₀.₈Mn₀.₁Co₀.₁O₂| NMC811| 3.8 V| High capacity, high Ni fraction cuts Co, mainstream for EVs  
  
### 4.2.3 Anode Materials

Graphite remains the dominant anode. It inserts Li⁺ between its layers (intercalation), with a theoretical capacity of 372 mAh/g (LiC₆). Higher-capacity silicon (Si) anodes (theoretical capacity about 3,579 mAh/g) are being commercialized, but the volume expansion during cycling (about 300%) is a lifetime challenge.

### 4.2.4 Calculating Theoretical Capacity

The theoretical capacity of an electrode material can be calculated from Faraday's law. The charge carried by one mole of electrons is the Faraday constant, F = 96,485 C/mol.

Theoretical capacity Q [mAh/g] = (n × F) ÷ (M × 3.6) 

Here n is the number of electrons transferred, M is the molar mass [g/mol], and 3.6 is the conversion factor from C/g to mAh/g. We implement this in Python in Section 4.5.

**⚠️ Mechanisms of Capacity Degradation**

An LIB loses capacity when charged and discharged repeatedly. The main degradation factors are as follows:

  * **SEI growth** : the solid electrolyte interphase on the anode surface continuously consumes Li⁺
  * **Cathode structural collapse** : transition-metal dissolution and phase transitions in high-Ni systems
  * **Lithium plating** : metallic Li deposits during low-temperature or fast charging, risking internal short circuits

## 4.3 Fuel Cells

### 4.3.1 Operating Principle and Types

A fuel cell is **a power-generating device that converts chemical energy directly into electrical energy through the electrochemical reaction of a fuel such as hydrogen with oxygen**. Because it does not go through combustion, it is not bound by the Carnot efficiency and can achieve high theoretical efficiency. The overall reaction is the formation of water:

H₂ + ½O₂ → H₂O (ΔG = -237 kJ/mol, standard state) 

Type | Electrolyte | Operating temperature | Main applications  
---|---|---|---  
PEFC (polymer electrolyte)| Proton-conducting polymer membrane| 60-90 °C| Fuel-cell vehicles, residential  
SOFC (solid oxide)| Oxide-ion-conducting ceramic (YSZ)| 700-1000 °C| Stationary generation, cogeneration  
  
### 4.3.2 Electromotive Force and the Nernst Equation

The theoretical electromotive force (reversible potential) depends on the activities (partial pressures) of reactants and products and on temperature, and is given by the Nernst equation:

E = E° + (RT ÷ 2F) × ln(pH₂ · pO₂1/2) 

Here E° is the standard reversible potential (1.229 V at 25 °C), R is the gas constant, T is the absolute temperature, F is the Faraday constant, and p is each gas partial pressure. Using air (O₂ partial pressure 0.21 atm) lowers the EMF slightly relative to pure oxygen.

### 4.3.3 Voltage Losses from Polarization (Overpotential)

In practice, the cell voltage falls below the theoretical EMF once current flows. This voltage loss is called polarization, and it is split into three components.
    
    
    flowchart TD
        A[Reversible EMF E_rev] --> B[Activation overpotential  
    dominant at low current]
        B --> C[Ohmic overpotential  
    linear rise at mid current]
        C --> D[Concentration overpotential  
    surges at high current]
        D --> E[Actual cell voltage V]
    
        style A fill:#e8f5e9
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#fce4ec
        style E fill:#f3e5f5
            

  1. **Activation overpotential** : the barrier needed to drive the electrode reaction. Dominant at low current density and described by the Butler-Volmer equation (Tafel approximation). The catalyst (Pt) performance is key.
  2. **Ohmic overpotential** : loss from the ionic conduction resistance of the electrolyte membrane and electronic resistance. Proportional to current (V = iR); thinning the membrane helps.
  3. **Concentration overpotential** : loss when the reactant gas supply cannot keep up and is depleted at the electrode surface. It surges as the current approaches the limiting current density iL.

**💡 Reading the Polarization Curve**

The I-V curve with current density on the horizontal axis and cell voltage on the vertical axis is the polarization curve. Three regions appear: a steep initial drop at low current (activation), a linear slope at mid current (ohmic), and a sharp drop at high current (concentration). The product of voltage and current is the power density, and the maximum power point is set by the balance of these losses. We model it in Section 4.5.

## 4.4 Solar Cells

### 4.4.1 The pn Junction and the Photovoltaic Effect

A solar cell is **a photovoltaic device that generates electron-hole pairs by illuminating a semiconductor pn junction and extracts power**. Photons with energy greater than the band gap (Eg) excite electrons from the valence band to the conduction band, and the built-in field of the pn junction separates the carriers to produce an EMF.

### 4.4.2 The Shockley-Queisser Limit

The theoretical efficiency of a single-junction solar cell has an upper bound known as the Shockley-Queisser limit (the detailed-balance limit). A small band gap absorbs many photons but lowers the voltage, while a large band gap raises the voltage but reduces the absorbed photons, so the efficiency is maximized at an optimal band gap.

**💡 Optimal Band Gap and Theoretical Limit**

For the AM1.5G standard solar spectrum, the Shockley-Queisser limit peaks at about 33% for a band gap of about 1.34 eV. The limit for silicon (Eg = 1.12 eV) is about 29-33%, and being close to this optimum is one reason silicon is dominant. The main loss factors are the non-absorption of sub-band-gap photons and the thermal relaxation (thermalization) of excess energy. Multi-junction (tandem) cells are a means to exceed this limit.

### 4.4.3 IV Characteristics and the Fill Factor

Solar-cell performance is evaluated from the IV (current-voltage) characteristics. A real device is approximated by the single-diode model, which includes a series resistance Rs and a parallel (shunt) resistance Rsh:

I = IL \- I₀ [exp((V + I·Rs) ÷ (n·Vt)) - 1] - (V + I·Rs) ÷ Rsh

Here IL is the photogenerated current, I₀ is the reverse saturation current, n is the diode ideality factor, and Vt = kT/q is the thermal voltage. The key performance metrics are as follows:

  * **Short-circuit current I sc**: the current at V = 0 (corresponds to light absorption)
  * **Open-circuit voltage V oc**: the voltage at I = 0 (related to the material band gap)
  * **Fill factor FF** : FF = (Vmp·Imp) ÷ (Voc·Isc). It expresses the "squareness" of the IV curve and approaches 1 as resistive losses shrink.
  * **Conversion efficiency η** : η = (Voc·Isc·FF) ÷ Pin

**✅ The Rapid Rise of Perovskite Solar Cells**

In recent years, perovskite solar cells have drawn attention. Using an organic-inorganic hybrid perovskite structure (e.g., CH₃NH₃PbI₃) as the light-absorbing layer, the conversion efficiency has risen rapidly from 3.8% in 2009 to over 26% for single junctions today, and over 33% in tandems with silicon. Low-cost solution-based fabrication is possible, but long-term stability against humidity and heat, along with the environmental burden of lead, are challenges for commercialization.

## 4.5 Python Practice: Performance Calculations for Energy Materials

For the three devices studied so far, we now compute performance metrics in Python. All of the code below is self-contained and runs with NumPy alone.

### Example 1: Theoretical Capacity of Electrode Materials from Faraday's Law

From the molar mass and number of electrons transferred of cathode and anode materials, we compute the theoretical gravimetric capacity [mAh/g]. We further estimate the energy density of an LFP/graphite full cell.
    
    
    # ===================================
    # Example 1: Theoretical capacity from Faraday's law
    # ===================================
    
    import numpy as np
    
    # Faraday constant
    F = 96485.0  # C/mol
    
    # Electrode materials: molar mass [g/mol], electrons transferred per formula unit
    materials = {
        "LiCoO2 (LCO)":      {"M": 97.87,  "n": 0.5},   # ~0.5 Li extracted in practice
        "LiFePO4 (LFP)":     {"M": 157.76, "n": 1.0},
        "LiNi0.8Mn0.1Co0.1O2 (NMC811)": {"M": 96.72, "n": 0.8},
        "Graphite (LiC6)":   {"M": 72.06,  "n": 1.0},   # C6 basis, 1 Li per 6 C
        "Silicon (Li15Si4)": {"M": 28.09,  "n": 3.75},  # Li3.75Si
    }
    
    def theoretical_capacity(M, n):
        """Theoretical gravimetric capacity in mAh/g.
        Q = n * F / M (C/g), divided by 3.6 to convert to mAh/g.
        """
        Q_C_per_g = n * F / M          # C/g
        Q_mAh_per_g = Q_C_per_g / 3.6  # mAh/g
        return Q_mAh_per_g
    
    print("Theoretical gravimetric capacity of electrode materials")
    print("=" * 62)
    print(f"{'Material':<32}{'M [g/mol]':>10}{'n':>5}{'Q [mAh/g]':>12}")
    print("-" * 62)
    for name, p in materials.items():
        Q = theoretical_capacity(p["M"], p["n"])
        print(f"{name:<32}{p['M']:>10.2f}{p['n']:>5.2f}{Q:>12.1f}")
    print("=" * 62)
    
    # Full-cell energy density estimate (LFP cathode / graphite anode)
    Q_cat = theoretical_capacity(157.76, 1.0)
    Q_an = theoretical_capacity(72.06, 1.0)
    Q_cell = 1.0 / (1.0/Q_cat + 1.0/Q_an)  # series combination of specific capacities
    V_avg = 3.3  # V, LFP/graphite average voltage
    E_grav = Q_cell * V_avg  # Wh/kg (mAh/g * V = mWh/g = Wh/kg)
    print(f"\nLFP/graphite full cell (active material only):")
    print(f"  Cathode capacity : {Q_cat:.1f} mAh/g")
    print(f"  Anode capacity   : {Q_an:.1f} mAh/g")
    print(f"  Combined capacity: {Q_cell:.1f} mAh/g")
    print(f"  Average voltage  : {V_avg:.1f} V")
    print(f"  Energy density   : {E_grav:.1f} Wh/kg (active material basis)")

**Execution result:**
    
    
    # Theoretical gravimetric capacity of electrode materials
    # ==============================================================
    # Material                         M [g/mol]    n   Q [mAh/g]
    # --------------------------------------------------------------
    # LiCoO2 (LCO)                         97.87 0.50       136.9
    # LiFePO4 (LFP)                       157.76 1.00       169.9
    # LiNi0.8Mn0.1Co0.1O2 (NMC811)         96.72 0.80       221.7
    # Graphite (LiC6)                      72.06 1.00       371.9
    # Silicon (Li15Si4)                    28.09 3.75      3578.0
    # ==============================================================
    #
    # LFP/graphite full cell (active material only):
    #   Cathode capacity : 169.9 mAh/g
    #   Anode capacity   : 371.9 mAh/g
    #   Combined capacity: 116.6 mAh/g
    #   Average voltage  : 3.3 V
    #   Energy density   : 384.8 Wh/kg (active material basis)

The computed values agree well with literature values (LFP about 170 mAh/g, graphite 372 mAh/g, silicon about 3,579 mAh/g). Silicon's theoretical capacity being about ten times that of graphite is why Si-based anodes are anticipated. Note that the active-material-only estimate (384.8 Wh/kg) comes out higher than a real cell (150-180 Wh/kg), which also includes electrolyte, current collectors, and casing.

### Example 2: Fuel-Cell Polarization Curve Model

We compute the theoretical EMF from the Nernst equation, then subtract the activation, ohmic, and concentration overpotentials to obtain the PEFC polarization curve, and search for the maximum power point.
    
    
    # ===================================
    # Example 2: PEFC polarization curve model
    # ===================================
    
    import numpy as np
    
    # Physical constants
    R = 8.314      # J/(mol K)
    F = 96485.0    # C/mol
    T = 353.15     # K (80 C, typical PEFC operation)
    
    # Nernst / thermodynamics
    E0 = 1.229     # V, standard reversible potential of H2/O2 at 25 C
    p_H2 = 1.0     # hydrogen partial pressure [atm]
    p_O2 = 0.21    # oxygen partial pressure in air [atm]
    # Nernst equation for H2 + 1/2 O2 -> H2O
    E_rev = E0 - 8.5e-4 * (T - 298.15) + (R * T) / (2 * F) * np.log(p_H2 * p_O2**0.5)
    
    # Polarization model parameters (representative PEFC values)
    i0 = 1.0e-3    # A/cm^2, exchange current density (activation)
    alpha = 0.5    # charge transfer coefficient
    i_L = 1.6      # A/cm^2, limiting current density (concentration)
    R_ohm = 0.15   # ohm cm^2, area-specific ohmic resistance
    i_leak = 1e-3  # A/cm^2, internal/leak current
    
    def activation_loss(i):
        return (R * T) / (alpha * 2 * F) * np.log((i + i_leak) / i0)
    
    def ohmic_loss(i):
        return i * R_ohm
    
    def concentration_loss(i):
        return -(R * T) / (2 * F) * np.log(1.0 - i / i_L)
    
    def cell_voltage(i):
        return E_rev - activation_loss(i) - ohmic_loss(i) - concentration_loss(i)
    
    # Current density sweep
    i = np.linspace(1e-3, i_L * 0.995, 400)
    V = cell_voltage(i)
    P = V * i  # power density W/cm^2
    
    # Peak power point
    idx = np.argmax(P)
    print("PEFC polarization model (T = 80 C, air cathode)")
    print("=" * 55)
    print(f"Reversible voltage E_rev      : {E_rev:.3f} V")
    print(f"Cell voltage at i=1mA         : {cell_voltage(1e-3):.3f} V")
    print("-" * 55)
    for target in [0.2, 0.5, 1.0, 1.4]:
        j = np.argmin(np.abs(i - target))
        print(f"i = {i[j]:.2f} A/cm^2 -> V = {V[j]:.3f} V, "
              f"P = {P[j]:.3f} W/cm^2")
    print("-" * 55)
    print(f"Peak power density            : {P[idx]:.3f} W/cm^2")
    print(f"  at i = {i[idx]:.3f} A/cm^2, V = {V[idx]:.3f} V")
    eta = V[idx] / 1.482  # relative to higher heating value HHV = 1.482 V
    print(f"  voltage efficiency (vs HHV) : {eta*100:.1f} %")
    print("=" * 55)

**Execution result:**
    
    
    # PEFC polarization model (T = 80 C, air cathode)
    # =======================================================
    # Reversible voltage E_rev      : 1.170 V
    # Cell voltage at i=1mA         : 1.149 V
    # -------------------------------------------------------
    # i = 0.20 A/cm^2 -> V = 0.977 V, P = 0.196 W/cm^2
    # i = 0.50 A/cm^2 -> V = 0.901 V, P = 0.450 W/cm^2
    # i = 1.00 A/cm^2 -> V = 0.795 V, P = 0.796 W/cm^2
    # i = 1.40 A/cm^2 -> V = 0.708 V, P = 0.992 W/cm^2
    # -------------------------------------------------------
    # Peak power density            : 1.026 W/cm^2
    #   at i = 1.540 A/cm^2, V = 0.666 V
    #   voltage efficiency (vs HHV) : 44.9 %
    # =======================================================

As the current density rises, the cell voltage falls, and the power density peaks (about 1.03 W/cm²) just before the limiting current. The lower EMF of an air cathode relative to pure oxygen, and the voltage efficiency of only about 45% at the maximum power point, show that voltage losses are unavoidable under real operating conditions. Practical cells are operated at a voltage lower than the peak (0.6-0.7 V) to balance efficiency and power.

### Example 3: Solar-Cell IV Curve and Fill Factor

We solve the single-diode model by bisection and obtain the IV curve, maximum power point, fill factor, and conversion efficiency of a crystalline-silicon solar cell.
    
    
    # ===================================
    # Example 3: Single-diode solar-cell IV analysis
    # ===================================
    
    import numpy as np
    
    # Physical constants
    q = 1.602176634e-19  # C
    k = 1.380649e-23     # J/K
    T = 298.15           # K
    Vt = k * T / q       # thermal voltage ~0.0257 V
    
    # Single-diode model parameters (representative c-Si cell, 1 sun, per cm^2)
    I_L = 0.0400   # A/cm^2, photogenerated current (~40 mA/cm^2)
    I_0 = 1.0e-12  # A/cm^2, diode reverse saturation current
    n = 1.0        # diode ideality factor
    Rs = 0.5       # ohm cm^2, series resistance
    Rsh = 1000.0   # ohm cm^2, shunt (parallel) resistance
    
    def diode_current(V, I):
        # Residual of the single-diode model:
        # I = I_L - I0(exp((V+I Rs)/(n Vt))-1) - (V+I Rs)/Rsh
        return I_L - I_0 * (np.exp((V + I * Rs) / (n * Vt)) - 1.0) - (V + I * Rs) / Rsh - I
    
    def solve_current(V):
        # Solve for I at a given V by bisection
        lo, hi = -0.05, I_L
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if diode_current(V, mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    
    # Voltage sweep
    V = np.linspace(0.0, 0.75, 400)
    I = np.array([solve_current(v) for v in V])
    P = V * I  # W/cm^2
    
    # Key metrics
    Isc = solve_current(0.0)
    idx_voc = np.argmin(np.abs(I))
    Voc = V[idx_voc]              # voltage where I = 0
    idx_mpp = np.argmax(P)
    Vmp, Imp, Pmp = V[idx_mpp], I[idx_mpp], P[idx_mpp]
    FF = Pmp / (Isc * Voc)
    P_in = 0.100                  # W/cm^2, AM1.5G ~ 100 mW/cm^2
    eff = Pmp / P_in
    
    print("Single-diode c-Si solar cell (1 sun, AM1.5G, 25 C)")
    print("=" * 54)
    print(f"Thermal voltage Vt   : {Vt*1000:.2f} mV")
    print(f"Short-circuit Isc    : {Isc*1000:.2f} mA/cm^2")
    print(f"Open-circuit Voc     : {Voc:.3f} V")
    print(f"MPP voltage Vmp      : {Vmp:.3f} V")
    print(f"MPP current Imp      : {Imp*1000:.2f} mA/cm^2")
    print(f"Max power Pmp        : {Pmp*1000:.2f} mW/cm^2")
    print(f"Fill factor FF       : {FF:.3f}")
    print(f"Efficiency eta       : {eff*100:.2f} %")
    print("=" * 54)

**Execution result:**
    
    
    # Single-diode c-Si solar cell (1 sun, AM1.5G, 25 C)
    # ======================================================
    # Thermal voltage Vt   : 25.69 mV
    # Short-circuit Isc    : 39.98 mA/cm^2
    # Open-circuit Voc     : 0.626 V
    # MPP voltage Vmp      : 0.530 V
    # MPP current Imp      : 37.56 mA/cm^2
    # Max power Pmp        : 19.91 mW/cm^2
    # Fill factor FF       : 0.796
    # Efficiency eta       : 19.91 %
    # ======================================================

A short-circuit current of 40 mA/cm², an open-circuit voltage of 0.63 V, a fill factor of 0.80, and a conversion efficiency of about 19.9% are consistent with the typical values of a practical crystalline-silicon cell. The fill factor falls when the series resistance Rs is increased, and also when the shunt resistance Rsh is decreased. Recomputing with different parameters lets you quantitatively confirm how resistive losses affect efficiency.

## Exercises

Q4.1: Calculating Theoretical Capacity

Calculate the theoretical gravimetric capacity [mAh/g] of the spinel cathode LiMn₂O₄ (molar mass 180.81 g/mol, electrons transferred n = 1).

View answer

Q = n·F ÷ (M × 3.6) = 1 × 96485 ÷ (180.81 × 3.6) = 96485 ÷ 650.9 ≈ **148.2 mAh/g**

LiMn₂O₄ is about 148 mAh/g, lower than LFP (170 mAh/g), but it has the advantages of inexpensive Mn and high power.

Q4.2: Comparing Energy Density

For NMC811 (capacity 221.7 mAh/g, average voltage 3.8 V) and LFP (capacity 169.9 mAh/g, average voltage 3.4 V), find the energy density [Wh/kg] per cathode active material and compare them.

View answer

NMC811: 221.7 × 3.8 = **842.5 Wh/kg**

LFP: 169.9 × 3.4 = **577.7 Wh/kg**

NMC811 has about 1.46 times the energy density of LFP. This quantitatively explains why NMC-type cells are chosen for EVs that prioritize driving range, while LFP is chosen for applications that prioritize safety, lifetime, and cost.

Q4.3: EMF from the Nernst Equation

If a PEFC is operated with pure oxygen (pO₂ = 1.0 atm), how does the reversible voltage change at 80 °C (353.15 K)? Compare with the air operation (0.21 atm) of Example 2. Use R = 8.314 and F = 96485.

View answer

The difference in the Nernst term is (RT/2F)·ln(1.0^0.5 / 0.21^0.5) = (RT/2F)·(1/2)·ln(1/0.21).

RT/2F = 8.314 × 353.15 ÷ (2 × 96485) = 0.01521 V.

Difference = 0.01521 × 0.5 × ln(4.762) = 0.01521 × 0.5 × 1.561 = **+0.0119 V**

With pure oxygen the reversible voltage rises by about 12 mV. Pure oxygen raises the EMF, but air is normally used in automotive applications because of the cost of the supply infrastructure.

Q4.4: Identifying Overpotential Components

In a fuel-cell polarization curve, for the three regions where (a) the voltage drops steeply at low current, (b) the voltage falls linearly in proportion to current at mid current, and (c) the voltage drops sharply near the limiting current, which overpotential is dominant in each?

View answer

  * (a) Low current: **activation overpotential** (the electrode-reaction barrier, logarithmic Tafel term)
  * (b) Mid current: **ohmic overpotential** (linear resistive loss, V = iR)
  * (c) Near the limiting current: **concentration overpotential** (reactant-gas supply limitation)

Thinning the membrane reduces the ohmic loss, improving the catalyst reduces the activation loss, and designing the gas diffusion layer reduces the concentration loss.

Q4.5: Fill Factor and Efficiency

A solar cell has Voc = 0.65 V, Isc = 38 mA/cm², and FF = 0.78. With an incident irradiance of 100 mW/cm² (AM1.5G), find the conversion efficiency η.

View answer

Pmp = Voc × Isc × FF = 0.65 × 38 × 0.78 = 19.27 mW/cm²

η = Pmp ÷ Pin = 19.27 ÷ 100 = **19.3 %**

If the fill factor improves from 0.78 to 0.82 with everything else unchanged, the efficiency rises to about 20.3%. Lowering the series resistance and raising the shunt resistance are the keys to improving FF.

Q4.6: Understanding the Shockley-Queisser Limit

Both a material with too large a band gap and one with too small a band gap have reduced conversion efficiency. Explain why, from the standpoints of photon absorption and voltage.

View answer

**Too large a band gap** : the open-circuit voltage is higher, but photons below Eg cannot be absorbed, so the short-circuit current is small and the efficiency falls from lack of current.

**Too small a band gap** : many photons are absorbed and the short-circuit current is large, but the open-circuit voltage falls, and excess energy is lost as heat (thermalization), so the efficiency falls.

Balancing the two, the efficiency is maximized at about 1.34 eV (about 33%) under AM1.5G. A tandem structure is a means to exceed this single-junction limit.

## Checking the Learning Objectives

Let us confirm whether you have mastered the following through this chapter:

  * ✅ Classify energy conversion and storage devices and explain the meaning of energy density, power density, and conversion efficiency
  * ✅ Understand the characteristics of LIB cathode (LCO/NMC/LFP) and anode (graphite/silicon) materials and the theoretical capacity calculation from Faraday's law
  * ✅ Distinguish the fuel-cell EMF (Nernst equation) and the three overpotentials (activation, ohmic, concentration) and explain the polarization curve
  * ✅ Compute the solar-cell pn-junction operation, the Shockley-Queisser limit, and the IV characteristics, fill factor, and conversion efficiency
  * ✅ Compute the performance metrics of each device quantitatively in Python and apply them to material-selection decisions

## Summary

In this chapter we studied three energy materials that underpin a carbon-neutral society, through quantitative calculations of performance metrics. The key points are as follows:

  * **Lithium-ion batteries** : the theoretical capacity is set by Faraday's law, and the choice of cathode material (LCO/NMC/LFP) governs the trade-off among energy density, safety, and cost. Silicon anodes are a strong candidate for higher capacity.
  * **Fuel cells** : the EMF is given by the Nernst equation, and the actual cell voltage falls due to the three overpotentials (activation, ohmic, concentration). The maximum power point is set by the balance of losses.
  * **Solar cells** : single-junction efficiency is constrained by the Shockley-Queisser limit (about 33%). From the IV characteristics we can obtain the fill factor and conversion efficiency and evaluate the effect of resistive losses.

These models are simplified, but they are useful in practice as first approximations for material design. More precise analysis uses electrochemical models (such as PyBaMM) and device simulators.

## Next Steps

In Chapter 4, we studied the design principles and performance calculations of energy materials (lithium-ion batteries, fuel cells, solar cells). In the next Chapter 5, we proceed to more advanced topics in materials systems.

[← Back to Chapter 3](<./chapter-3.html>) [Proceed to Chapter 5 →](<./chapter-5.html>)

## References

  1. Goodenough, J. B., & Park, K. S. (2013). "The Li-ion rechargeable battery: A perspective." _Journal of the American Chemical Society_ , 135(4), 1167-1176. - A review by a Nobel laureate on the history and future prospects of lithium-ion batteries.
  2. Steele, B. C. H., & Heinzel, A. (2001). "Materials for fuel-cell technologies." _Nature_ , 414(6861), 345-352. - Material design guidelines for solid oxide (SOFC) and PEM fuel cells.
  3. Nelson, J. (2003). _The Physics of Solar Cells_. Imperial College Press. pp. 1-40, 143-200. - A systematic treatment of solar-cell physics, IV characteristics, and the single-diode model.
  4. Green, M. A., Ho-Baillie, A., & Snaith, H. J. (2014). "The emergence of perovskite solar cells." _Nature Photonics_ , 8(7), 506-514. - The rapid development and high-efficiency mechanisms of perovskite solar cells.
  5. Tarascon, J. M., & Armand, M. (2001). "Issues and challenges facing rechargeable lithium batteries." _Nature_ , 414(6861), 359-367. - Challenges of lithium-ion batteries and prospects for next-generation battery materials.
  6. Shockley, W., & Queisser, H. J. (1961). "Detailed balance limit of efficiency of p-n junction solar cells." _Journal of Applied Physics_ , 32(3), 510-519. - The classic paper deriving the theoretical efficiency limit of single-junction solar cells.
  7. PyBaMM Documentation. (2024). _Python Battery Mathematical Modeling_. <https://pybamm.org/> \- A Python library for battery physics and electrochemical simulation.

## Tools and Libraries Used

  * **NumPy** (v1.24+): Numerical computing library - <https://numpy.org/>
  * **SciPy** (v1.10+): Scientific computing library (optimize, integrate) - <https://scipy.org/>
  * **Matplotlib** (v3.7+): Data visualization library - <https://matplotlib.org/>
  * **PyBaMM** (v23+): Battery mathematical modeling library - <https://pybamm.org/>
  * **pymatgen** (v2023+): Materials science computing library - <https://pymatgen.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
