---
title: "Chapter 2: Material Extrusion (FDM/FFF)"
chapter_title: "Chapter 2: Material Extrusion (FDM/FFF)"
subtitle: Additive Manufacturing of Thermoplastics - The Science of Filament Fusion
reading_time: 40-45 minutes
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 2

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-2.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain the following:

### Basic Understanding (Level 1)

  * The basic principle of Material Extrusion (MEX) = FDM/FFF: filament feeding, melting in the hot end, nozzle extrusion, and cooling/solidification
  * The characteristics and appropriate use of the main thermoplastics (PLA, ABS, PETG, PC)
  * The difference between amorphous and semi-crystalline polymers, and the meaning of glass transition temperature (Tg) and melting temperature (Tm)
  * What each of nozzle temperature, bed temperature, and chamber temperature controls

### Practical Skills (Level 2)

  * Compute the extrusion volumetric flow rate in Python and check it against the hot-end flow limit
  * Understand and tune the roles of layer height, line width, infill, print speed, and retraction
  * Estimate interlayer bond strength using the polymer chain interdiffusion (reptation) model
  * Compute the cooling curve of a bead and its welding window

### Applied Ability (Level 3)

  * Quantitatively evaluate the mechanical anisotropy of parts (in-plane XY vs. interlayer Z) and choose a build orientation according to the load direction
  * Understand the mechanisms of warping and shrinkage and devise countermeasures suited to material and geometry
  * Diagnose typical print defects (stringing, delamination, warping, under-extrusion, etc.) and propose cause-based remedies
  * Explain the requirements for handling high-performance engineering polymers such as PEEK and ULTEM

**💡 Where this chapter fits**

Chapter 1 surveyed additive manufacturing (AM) as a whole through seven process categories. This chapter focuses on the most widespread of them, **Material Extrusion (MEX / FDM / FFF)** , and digs into how thermoplastics melt and stack, and why strength depends on layer orientation, from the perspectives of materials science, heat transfer, and polymer physics.

## 2.1 Principle of Material Extrusion (FDM/FFF)

### 2.1.1 What is FDM/FFF?

**Material Extrusion (MEX)** is a method that heats and melts a thermoplastic filament, extrudes it through a fine nozzle while scanning within a plane, and stacks it one layer at a time to build a solid. Owing to trademark reasons there are two names, **FDM (Fused Deposition Modeling)** and **FFF (Fused Filament Fabrication)** , but they refer to the same technology. FDM is a registered trademark of Stratasys, and the open-source community adopted FFF as an equivalent generic term.

This method is the most widespread in the world because the machines are inexpensive, the material is supplied in the easy-to-handle form of filament, and the process is intuitive. On the other hand, as discussed later, it has an inherent weakness: **the interface between layers tends to be the weak point in strength**.

#### Main components

  * **Filament** : A wire of thermoplastic resin, typically 1.75 mm (common) or 2.85 mm in diameter, supplied wound on a spool.
  * **Extruder (feed mechanism)** : A toothed gear grips the filament and pushes it into the hot end at a controlled rate. There is the "Bowden" type placed away from the nozzle and the "direct" type placed directly above the nozzle.
  * **Hot end** : A heated block that melts the filament, with temperature controlled by a heater and a temperature sensor (thermistor).
  * **Nozzle** : The tip that extrudes the molten resin. Orifice diameter 0.2–1.0 mm (standard 0.4 mm). Brass is common; hardened steel or ruby nozzles are used for abrasive filaments.
  * **Heat break** : A thin tube that blocks heat between the hot end and the feed mechanism. If the resin softens prematurely here it causes clogging (heat creep).
  * **Build plate (bed)** : The platform to which the part adheres. Many are heated, which is important for first-layer adhesion and warp suppression.

### 2.1.2 The build process flow

Following the physical process by which filament becomes a solid gives the following.
    
    
    flowchart TD
        A[Filament feed  
    solid, room temp] --> B[Heat break  
    gradual softening]
        B --> C[Hot-end melting  
    190-260 C]
        C --> D[Nozzle extrusion  
    shear flow]
        D --> E[Deposit on bed/prev layer  
    contact, heat transfer]
        E --> F[Welding by interdiffusion  
    chain entanglement]
        F --> G[Cooling/solidification  
    below Tg]
        G --> H[Extrude next layer  
    repeat]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style F fill:#e8f5e9
        style H fill:#f3e5f5
            

What matters here from a materials-science viewpoint is **steps E and F**. The extruded molten resin (bead) contacts the slightly cooler resin just below or beside it. Only while the interface temperature of the two stays above the glass transition temperature (see below) can polymer chains diffuse across the interface and entangle into a unified whole. How much of this "weldable time" can be secured governs the strength of the part.

### 2.1.3 The main temperature parameters

In FDM/FFF, three temperatures determine quality independently yet interdependently.

Temperature parameter | Typical range | Mainly controls  
---|---|---  
Nozzle temperature | 190–260°C (material-dependent) | Viscosity of the melt, ease of extrusion, interlayer bonding (higher welds more easily but risks stringing/over-melting)  
Bed temperature | 0–110°C (material-dependent) | First-layer adhesion, warp suppression at the base of the part (a guideline is to set it near the material's Tg)  
Chamber temperature | Room temp to 80°C+ | Prevents rapid cooling during the build, reducing interlayer temperature differences and residual stress (especially important for ABS/PC/PEEK)  
  
**💡 Temperature follows from "material properties"**

Nozzle and bed temperatures are not chosen by rules of thumb but are derived from each material's glass transition temperature, melting temperature, and thermal decomposition temperature. Organizing these properties in the next section reveals systematically why PLA is handled cold and ABS hot, and why PEEK requires a high-temperature chamber.

## 2.2 Thermoplastics Used in Additive Manufacturing

### 2.2.1 Amorphous vs. semi-crystalline polymers

Thermoplastics are broadly divided into **amorphous** and **semi-crystalline** types according to how the molecular chains are arranged in the solid state. This difference greatly influences FDM behavior.

  * **Glass Transition Temperature (Tg)** : The temperature at which chains in the amorphous region go from a frozen state to being mobile. Across it, the material changes from a hard "glassy state" to a rubbery "rubbery state." Interlayer welding proceeds roughly only while the interface temperature stays above Tg.
  * **Melting Temperature (Tm)** : The temperature at which crystalline regions melt. Only semi-crystalline polymers have one.

**Amorphous polymers (e.g., ABS, PC, PETG)** have no distinct melting point and soften gradually above Tg. Their volume change on cooling is gentle, so dimensions tend to be stable; however, materials with a high Tg tend to warp.

**Semi-crystalline polymers (e.g., PLA, nylon, PEEK)** shrink significantly on cooling due to crystallization. When this shrinkage occurs non-uniformly it produces strong warping and distortion, raising the difficulty of printing. On the other hand, crystallization yields high rigidity and chemical resistance.

Condition for welding to proceed (guideline): interface temperature T_interface > Tg 

### 2.2.2 The main printing materials

Here we organize the representative materials widely used from desktop to industrial applications. The values are representative and vary by manufacturer and grade.

Material | Class | Tg / Tm (°C) | Nozzle temp (°C) | Bed temp (°C) | Characteristics  
---|---|---|---|---|---  
**PLA**  
(polylactic acid) | Semi-crystalline | Tg 60 / Tm 170 | 190–220 | 20–60 | Easy to print, low warp, biodegradable. Low heat resistance and toughness. For beginners  
**ABS**  
(acrylonitrile butadiene styrene) | Amorphous | Tg 105 / - | 230–260 | 90–110 | Good heat resistance and toughness. Warps easily, needs an enclosure. Can be smoothed with acetone vapor  
**PETG**  
(glycol-modified polyethylene terephthalate) | Amorphous | Tg 80 / - | 230–250 | 70–90 | Good balance of strength, chemical and weather resistance, low warp. Prone to stringing  
**PC**  
(polycarbonate) | Amorphous | Tg 147 / - | 260–310 | 110–130 | High strength, heat resistance, and transparency. Highly hygroscopic; needs a high-temp chamber  
**ASA** | Amorphous | Tg 100 / - | 240–260 | 90–110 | ABS-equivalent performance plus weather (UV) resistance. For outdoor use  
**Nylon**  
(PA) | Semi-crystalline | Tg 50 / Tm 220 | 240–270 | 70–100 | High toughness and wear resistance. Strongly hygroscopic, needs drying. Warps easily  
**TPU**  
(thermoplastic polyurethane) | Amorphous-type | Tg -30 to -20 / - | 220–240 | 40–60 | Flexible elastomer. Requires slow printing; direct drive recommended  
  
### 2.2.3 High-performance engineering polymers (PEEK / ULTEM)

In aerospace and medical fields that demand near-metal specific strength or high heat resistance, **high-performance polymers** are used.

  * **PEEK (polyether ether ketone)** : A semi-crystalline high-performance polymer. Continuous-use temperature about 250°C, with high mechanical strength, chemical resistance, and biocompatibility. Nozzle temperature about 360–450°C, bed temperature 120–160°C, and a high-temperature chamber (80–120°C+) to control crystallization are essential.
  * **ULTEM (PEI: polyetherimide)** : An amorphous high-performance polymer. Continuous-use temperature about 170°C, with excellent flame retardancy (meets FAR requirements for aircraft interiors). Nozzle temperature about 350–390°C and a high-temperature chamber are required. ULTEM 9085 is used for aerospace parts on machines such as the Stratasys Fortus series.

**⚠️ The difficulty of high-performance polymers**

PEEK and ULTEM cannot simply be melted at high temperature and printed. In semi-crystalline PEEK, if the cooling rate is too fast, crystallization is insufficient and mechanical properties drop; if too slow or non-uniform, warping and distortion occur. A uniform high-temperature environment (heated chamber) and control of the optimal cooling profile are indispensable, requiring dedicated high-temperature machines (on the order of hundreds of thousands of dollars overall) and careful process development. Note that "having the material" does not mean you can print it.

### 2.2.4 Code Example 1: Volumetric flow rate and hot-end limit

Whether FDM can print stably depends on whether the **volumetric flow rate Q** extruded per unit time exceeds the hot end's melting/supply capability (the flow limit). The volumetric flow rate can be approximated as follows.

Q [mm³/s] = line width W × layer height H × travel speed v 

Let us compute several settings and check against a standard hot-end limit (about 12 mm³/s for PLA).
    
    
    import numpy as np
    
    def volumetric_flow(line_width, layer_height, speed):
        """Extrusion volumetric flow rate (mm^3/s): cross-section (w x h) x speed"""
        return line_width * layer_height * speed
    
    max_flow = 12.0  # typical hotend limit for standard PLA (mm^3/s)
    cases = [
        ("LH0.10 / LW0.40 / 60mm/s", 0.40, 0.10, 60),
        ("LH0.20 / LW0.40 / 60mm/s", 0.40, 0.20, 60),
        ("LH0.20 / LW0.45 / 100mm/s", 0.45, 0.20, 100),
        ("LH0.30 / LW0.60 / 80mm/s", 0.60, 0.30, 80),
    ]
    for name, w, h, v in cases:
        q = volumetric_flow(w, h, v)
        flag = "OK" if q <= max_flow else "EXCEEDS LIMIT"
        print(f"{name:26s}: Q = {q:5.2f} mm^3/s  [{flag}]")
    
    # Convert to 1.75mm filament feed rate
    d_fil = 1.75
    A_fil = np.pi * (d_fil / 2) ** 2
    q = volumetric_flow(0.45, 0.20, 100)
    print(f"\nFilament feed rate (Q={q:.2f}): {q / A_fil:.2f} mm/s")
    

**Execution result:**
    
    
    LH0.10 / LW0.40 / 60mm/s  : Q =  2.40 mm^3/s  [OK]
    LH0.20 / LW0.40 / 60mm/s  : Q =  4.80 mm^3/s  [OK]
    LH0.20 / LW0.45 / 100mm/s : Q =  9.00 mm^3/s  [OK]
    LH0.30 / LW0.60 / 80mm/s  : Q = 14.40 mm^3/s  [EXCEEDS LIMIT]
    
    Filament feed rate (Q=9.00): 3.74 mm/s
    

Even a seemingly gentle setting of 0.3 mm layer height and 80 mm/s with a 0.6 mm nozzle reaches a volumetric flow of 14.4 mm³/s, exceeding the standard hot-end limit. In this case the hot end cannot fully melt the resin and **under-extrusion** occurs. To print fast and at high flow, you must consider not only speed but also the hot end's melting capacity (e.g., swapping to a high-flow hot end).

## 2.3 Process Parameters

FDM print quality is determined by the combination of many parameters. Here we take up four with especially large influence.

### 2.3.1 Layer height and line width

  * **Layer height** : The thickness of one layer. The recommended range is 25–80% of nozzle diameter (0.1–0.32 mm for a 0.4 mm nozzle). Thinner layers make layer lines less visible and curved surfaces smoother, but increase the number of layers and print time.
  * **Line width** : The width of one extruded line. Usually 100–120% of nozzle diameter (0.4–0.48 mm for a 0.4 mm nozzle). Wider improves strength and print speed; narrower improves fine-detail reproduction.
  * **First layer** : The convention is to make only the first layer slightly thicker, slower, and hotter to ensure adhesion to the bed.

### 2.3.2 Infill

**Infill** is the structure that fills the interior of the part, specified by density and pattern. Chapter 1 covered pattern-specific characteristics (Grid, Honeycomb, Gyroid, etc.), so here we focus on the relationship between density and mechanics.

  * **0–15%** : Decorative and non-load-bearing parts. Saves material and time
  * **20–25%** : Standard for general prototypes
  * **40–60%** : Functional parts and parts under repeated loads
  * **100%** : Parts requiring maximum strength and water-tightness (print time increases greatly)

What is important is that **strength is not proportional to infill density and is nonlinear at the low-density end**. Because much of the load is carried by the shell (perimeters), increasing the number of shells often improves strength more efficiently than raising infill density.

### 2.3.3 Print speed and cooling

  * **Print speed** : Faster shortens print time, but as noted above, exceeding the volumetric-flow limit causes under-extrusion. At high speed the interlayer contact time also becomes short, making welding prone to being insufficient.
  * **Cooling fan** : Solidifying the extruded resin quickly improves the quality of overhangs and bridges. But if cooling is too strong, the interface drops below Tg immediately, the welding time is lost, and interlayer strength decreases. The basic rule is **strong for PLA, weak or off for ABS/PC/PEEK**.

**💡 The trade-off between appearance and strength**

"Visual cleanliness" and "mechanical strength" often point in opposite directions. Strong fan cooling makes protrusions and bridges crisp, but weakens interlayer bonding. It is important to first decide whether the application is a "model to be seen" or a "part under load," and to choose cooling, temperature, and speed accordingly.

### 2.3.4 Retraction (countermeasure against stringing)

When the nozzle travels over a region it is not printing, the molten resin can drip due to its own weight or nozzle internal pressure, leaving fine strings (stringing). To prevent this, the operation of pulling the filament back slightly just before a travel move is called **retraction**.

  * **Retraction distance** : about 0.5–2 mm for direct drive, 4–7 mm for Bowden
  * **Retraction speed** : about 25–60 mm/s
  * Excessive retraction causes filament grinding and clogging, so keep it to the necessary minimum.

## 2.4 Interlayer Bonding and Mechanical Anisotropy

The greatest feature and also the weakness of FDM parts is the **interface between layers (the interlayer)**. Here we understand why the interface becomes weak from the viewpoint of polymer physics, and estimate it quantitatively in Python.

### 2.4.1 Interdiffusion of polymer chains (reptation)

The phenomenon of two molten polymer surfaces contacting and unifying is explained by **interdiffusion**. A polymer is a long, entangled chain, and the model in which such a chain moves by crawling through a tube is called **reptation** theory (by de Gennes, Doi-Edwards, and others).

The weld strength at the interface is determined by the degree to which chains interpenetrate across it. Using the welding time t and the time required for chains to fully entangle (the reptation time t_rep), the interface strength is approximated as follows.

σ_interface / σ_bulk ≈ (t / t_rep)^(1/4) (healing degree, max 1.0) 

The reptation time depends strongly on temperature and can be expressed in Arrhenius form. The higher the temperature, the faster the chain motion, so t_rep is smaller (i.e., welding is faster).

t_rep = t₀ · exp( Eₐ / (R·T) ) 

### 2.4.2 Code Example 2: Interface temperature and interlayer strength

Using the model above, we estimate the interlayer (Z-direction) tensile strength as the interface temperature is varied. We assume a welding window (time the interface stays weldable) of 1 second and a bulk strength of 40 MPa.
    
    
    import numpy as np
    
    R = 8.314          # gas constant J/(mol K)
    t0 = 1e-9          # pre-exponential factor s
    Ea = 90e3          # apparent activation energy for reptation J/mol
    t_contact = 1.0    # time interface stays weldable (welding window) s
    sigma_bulk = 40.0  # bulk tensile strength MPa
    
    print(f"{'T_iface(C)':>10} {'t_rep(s)':>12} {'healing':>8} {'sigma_z(MPa)':>13}")
    for Tc in [180, 200, 220, 240, 260]:
        T = Tc + 273.15
        t_rep = t0 * np.exp(Ea / (R * T))          # reptation time (Arrhenius)
        healing = min((t_contact / t_rep) ** 0.25, 1.0)  # healing degree (max 1.0)
        sigma_z = healing * sigma_bulk
        print(f"{Tc:>10} {t_rep:>12.3e} {healing:>8.3f} {sigma_z:>13.2f}")
    

**Execution result:**
    
    
    T_iface(C)     t_rep(s)  healing  sigma_z(MPa)
           180    2.370e+01    0.453         18.13
           200    8.633e+00    0.583         23.34
           220    3.413e+00    0.736         29.43
           240    1.451e+00    0.911         36.45
           260    6.576e-01    1.000         40.00
    

At an interface temperature of 180°C only about 45% of the bulk strength (18 MPa) is achieved, whereas at 240°C it recovers to 91% (36 MPa). This is the physical basis for "raising the nozzle temperature improves interlayer strength." However, raising the temperature too far invites stringing, over-melting, and thermal decomposition, so the upper limit is set by the trade-off with the material's decomposition temperature.

**⚠️ Limitations of this model**

The healing-degree model, the activation energy (90 kJ/mol), and the pre-exponential factor used here are synthetic values for illustration. Actual reptation times and interface strengths depend strongly on the material's molecular-weight distribution, additives, and the real interface temperature history. The aim is to understand the **trend** —"higher temperature and longer contact time raise interlayer strength"—not the absolute values.

### 2.4.3 Code Example 3: Bead cooling curve and welding window

To gain interlayer strength, the key is how much of the "welding window" during which the interface is above Tg can be secured. We estimate how an extruded bead cools using the **lumped capacitance model**. Assuming the internal temperature of the object is uniform, cooling follows the equation below.

T(t) = T_env + (T₀ − T_env)·exp(−t/τ), τ = ρ·c·L_c / h 

Here L_c is the characteristic length (volume/surface area) and h is the heat transfer coefficient. We compute using an ABS bead (0.4 mm diameter) as an example.
    
    
    import numpy as np
    
    # Lumped capacitance model: T(t) = T_env + (T0 - T_env) exp(-t/tau)
    rho, c = 1040.0, 1900.0   # ABS: density kg/m^3, specific heat J/(kg K)
    h_conv = 60.0             # convective heat transfer coeff W/(m^2 K)
    d = 0.4e-3                # bead diameter m
    Lc = (d / 2) / 2          # characteristic length (cylinder radius/2)
    tau = rho * c * Lc / h_conv
    T0, T_env, Tg = 240.0, 50.0, 105.0   # extrusion, ambient, ABS Tg (C)
    
    print(f"Characteristic length Lc = {Lc*1e6:.1f} um, time constant tau = {tau:.3f} s")
    t_to_Tg = -tau * np.log((Tg - T_env) / (T0 - T_env))
    print(f"Cooling time from {T0:.0f}C to Tg={Tg:.0f}C: {t_to_Tg*1000:.1f} ms")
    for t in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        T = T_env + (T0 - T_env) * np.exp(-t / tau)
        print(f"  t={t*1000:6.0f} ms : T = {T:6.1f} C")
    

**Execution result:**
    
    
    Characteristic length Lc = 100.0 um, time constant tau = 3.293 s
    Cooling time from 240C to Tg=105C: 4082.7 ms
      t=     0 ms : T =  240.0 C
      t=    50 ms : T =  237.1 C
      t=   100 ms : T =  234.3 C
      t=   200 ms : T =  228.8 C
      t=   500 ms : T =  213.2 C
      t=  1000 ms : T =  190.2 C
    

In this model with an ambient temperature of 50°C, a relatively long welding window of about 4 seconds is obtained before the interface falls below Tg (105°C). This shows that **keeping the surroundings warm (an enclosure or heated chamber) favors welding**. Conversely, applying a strong cooling fan to lower the ambient temperature shortens this window sharply.

**⚠️ Note that this is a simple model**

This model considers only convective cooling and ignores heat conduction to the adjacent, already-cooled layer, so it comes out slower than reality. In a real machine the bead contacts the cold previous layer and is quenched by conduction, so the welding window is shorter than this. The goal here is not to predict the absolute time but to grasp the qualitative relationship that "ambient temperature and heat transfer govern the window."

### 2.4.4 Code Example 4: Evaluating mechanical anisotropy

Because the interlayer interface is weak, the strength of an FDM part changes greatly depending on **the direction of the load relative to the layers**. This is called **mechanical anisotropy**. For three representative build orientations, we evaluate the retention relative to the strength when well-bonded in-plane (50 MPa, PLA-equivalent).
    
    
    # Tensile strength anisotropy by print orientation (FDM)
    sigma_bulk = 50.0  # in-plane, well-bonded tensile strength (PLA, MPa)
    orient = {
        "Flat    (XY, in-plane load)":  1.00,
        "On-edge (XY, in-plane load)":  0.92,
        "Upright (Z, across layers)":   0.48,
    }
    print(f"{'Orientation':30s} {'sigma(MPa)':>11} {'retention':>10}")
    for name, k in orient.items():
        print(f"{name:30s} {sigma_bulk*k:>11.1f} {k*100:>9.0f}%")
    

**Execution result:**
    
    
    Orientation                     sigma(MPa)  retention
    Flat    (XY, in-plane load)           50.0       100%
    On-edge (XY, in-plane load)           46.0        92%
    Upright (Z, across layers)            24.0        48%
    

Strength is maximal when the load is along the layers (Flat), but drops to about half when the load pulls the layers apart (Upright, Z-direction). This leads to a golden rule of FDM design: **"choose a build orientation so that the principal stress does not cross the interlayer interface."** For a hook-like part, for example, the convention is to lay it down so the layers run along the load direction. The literature also reports Z-direction strength as roughly 40–80% of the XY direction, and the values here fall within that range.

## 2.5 Warping, Shrinkage, and Quality Control

### 2.5.1 Mechanism of warping

**Warping** is caused by residual stress that arises when a part shrinks as it cools while its base is constrained to the bed. The larger the thermal shrinkage on cooling, and the longer the part, the greater the force lifting the corners. The free thermal shrinkage strain is expressed as follows.

shrinkage strain ε = α · ΔT (α: linear expansion coefficient, ΔT: cooling temperature difference from Tg) 

#### Code Example 5: Relative evaluation of thermal shrinkage and warping

For PLA, PETG, and ABS, we compare the shrinkage of a 100 mm-long part as it cools from near Tg to the bed/ambient temperature (50°C). The tendency to warp is relatively evaluated by the product of shrinkage strain and elastic modulus (proportional to the stress stored when constrained).
    
    
    # Thermal shrinkage strain and relative warping tendency
    materials = {
        "PLA":  dict(alpha=68e-6,  Tg=60,  E=3500e6),
        "PETG": dict(alpha=70e-6,  Tg=80,  E=2100e6),
        "ABS":  dict(alpha=90e-6,  Tg=105, E=2300e6),
    }
    L = 100.0            # part length mm
    T_bed_ambient = 50.0 # bed/ambient temperature after cooling C
    
    print(f"Part length L = {L:.0f} mm, cooled from ~Tg to {T_bed_ambient:.0f}C")
    print(f"{'Mat':>6} {'dT(K)':>7} {'strain(%)':>10} {'dL(mm)':>8} {'warp_idx':>9}")
    for name, p in materials.items():
        dT = p["Tg"] - T_bed_ambient
        strain = p["alpha"] * dT           # free thermal shrinkage strain
        dL = strain * L
        warp_idx = strain * p["E"] / 1e6   # relative index ~ locked-in stress (MPa)
        print(f"{name:>6} {dT:>7.0f} {strain*100:>10.4f} {dL:>8.3f} {warp_idx:>9.2f}")
    

**Execution result:**
    
    
    Part length L = 100 mm, cooled from ~Tg to 50C
       Mat   dT(K)  strain(%)   dL(mm)  warp_idx
       PLA      10     0.0680    0.068      2.38
      PETG      30     0.2100    0.210      4.41
       ABS      55     0.4950    0.495     11.39
    

ABS's warp index is about 4.8 times that of PLA. ABS has a high Tg (105°C), so the difference ΔT from the bed/ambient temperature is large, making the shrinkage strain large; moreover, its high elastic modulus stores that shrinkage as stress. This is the quantitative reason behind the rules of thumb "ABS warps easily and needs an enclosure" and "PLA warps little and is easy to handle." To suppress warping, as the equation shows, the most effective step is to **reduce ΔT (i.e., warm the surroundings)**.

### 2.5.2 Countermeasures for warping and adhesion failure

  * **Heated bed** : Set the bed temperature near the material's Tg to keep the base of the part in the rubbery state and relax shrinkage stress
  * **Enclosure / heated chamber** : Raise the ambient temperature to reduce ΔT and interlayer temperature differences. Especially effective for ABS, PC, and PEEK
  * **Brim / raft** : Add contact area around or beneath the part to mechanically suppress corner lifting
  * **Bed surface treatment** : Ensure first-layer adhesion with PEI sheets, glue, or dedicated adhesives
  * **Geometry design** : Round sharp corners, avoid large flat areas, add mouse ears (auxiliary corner pads)

### 2.5.3 Typical print defects and countermeasures

We organize the defects frequently seen in FDM in terms of cause (often the physics covered in this chapter) and countermeasure.

Defect | Symptom | Main cause | Countermeasure  
---|---|---|---  
Warping | Corners of the part lift and peel | Shrinkage stress on cooling (large ΔT) | Bed heating, enclosure, brim/raft  
Delamination | Cracks or separation at layer boundaries | Poor welding from insufficient interface temperature (low healing) | Raise nozzle temperature, weaken cooling fan, reduce speed  
Stringing | Fine strings remain between parts | Resin oozing during travel, insufficient retraction | Adjust retraction, lower nozzle temperature, dry the material  
Under-extrusion | Gaps or missing sections in layers, thin lines | Volumetric flow exceeds the limit, nozzle clog | Reduce speed, raise temperature, clean nozzle, review flow  
Elephant foot | The bottom few layers bulge outward | Excessive squish of the first layer, hot bed | Adjust first-layer height, reduce first-layer flow, apply chamfer compensation  
Zipper / seam | The start point of each layer stands out as a vertical line | Retraction/pressure fluctuation at the layer start | Optimize seam position, use coasting settings  
  
**💡 The mindset of quality control**

Most FDM defects reduce to three physics covered in this chapter—**melt flow (volumetric flow)** , **interlayer welding (temperature and time)** , and **thermal shrinkage (ΔT and constraint)**. Rather than fixing symptoms ad hoc, identifying "which physics has broken down" lets you choose a countermeasure tied directly to the cause.

## 2.6 Applications

Owing to its ease and material diversity, material extrusion is used in a wide range of fields, as follows.

  * **Prototyping and design verification** : The most common use. Quickly check shape, fit, and function at low cost
  * **Jigs and tools** : Assembly jigs and inspection gauges for production lines. Lightweight and easy to customize
  * **End-use products and low-volume production** : Functional and end-use parts made from PETG, nylon, or carbon-fiber-reinforced materials
  * **Medical and welfare** : Prosthetics and orthotics shaped to each patient, models for surgical planning
  * **Aerospace** : Flame-retardant cabin parts and lightweight brackets made from ULTEM 9085 and similar
  * **Education** : Teaching materials that let students experience the whole flow from design to build safely and at low cost

## Confirming the Learning Objectives

Through this chapter, confirm that you can now explain the following.

### Basic understanding

  * ✅ Explain the principle of material extrusion (FDM/FFF) and its main components (hot end, nozzle, bed, etc.)
  * ✅ Explain the difference between amorphous and semi-crystalline polymers and the meaning of Tg and Tm
  * ✅ Explain the characteristics and appropriate temperature settings of PLA, ABS, PETG, and PC
  * ✅ Explain what nozzle, bed, and chamber temperature each control

### Practical skills

  * ✅ Compute the extrusion volumetric flow rate in Python and check it against the hot-end limit
  * ✅ Estimate the relationship between interface temperature and interlayer strength with the reptation model
  * ✅ Compute a bead's cooling curve and welding window with the lumped capacitance model
  * ✅ Evaluate mechanical anisotropy by build orientation

### Applied ability

  * ✅ Choose a build orientation according to the load direction
  * ✅ Understand the mechanism of warping and devise countermeasures suited to material and geometry
  * ✅ Diagnose and remedy typical print defects based on the physics

## Exercises

### Easy (basic check)

Q1: The relationship between FDM and FFF

Which is the correct statement about the terms "FDM" and "FFF"?

a) FDM is for metals and FFF is for plastics; they are entirely different technologies  
b) FDM and FFF refer to the same material-extrusion method; because FDM is a Stratasys trademark, FFF is used as a generic name  
c) FFF is a higher-precision successor to FDM  
d) FDM is photo-curing and FFF is thermal melting

Show answer

**Correct: b)**

**Explanation:** FDM (Fused Deposition Modeling) is a registered trademark of Stratasys. FFF (Fused Filament Fabrication) is the generic name the open-source community uses for the same "melt filament and stack" technology to avoid the trademark. The two refer to essentially the same process. It is neither for metals nor photo-curing (those are separate categories, PBF and VPP respectively).

Q2: The meaning of glass transition temperature

Regarding interlayer welding (bonding), what does the glass transition temperature (Tg) signify?

a) Polymer chains diffuse and weld only below Tg  
b) While the interface temperature is above Tg, polymer chains diffuse across the interface and welding proceeds  
c) Tg is unrelated to welding; it is a color-change temperature  
d) Above Tg the material always thermally decomposes

Show answer

**Correct: b)**

**Explanation:** The glass transition temperature Tg is the temperature at which chains in the amorphous region go from frozen to mobile. Only while the interface temperature is above Tg do chains interdiffuse across the interface and entangle, so welding proceeds. Once the interface falls below Tg, chain motion freezes and welding effectively stops. Therefore, keeping the "welding window = the time the interface is above Tg" long is the key to interlayer strength. Thermal decomposition is a separate phenomenon occurring at temperatures far above Tg.

Q3: Material selection

For an outdoor application exposed to UV, which material best combines ABS-equivalent toughness with weather resistance?

a) PLA b) ASA c) TPU d) PVA

Show answer

**Correct: b) ASA**

**Explanation:** ASA is an amorphous polymer that has mechanical properties equivalent to ABS while adding weather (UV) resistance, making it suitable for outdoor use. PLA has low heat and weather resistance and degrades easily outdoors; TPU is a flexible elastomer; and PVA is a water-soluble support material—none fit the application.

### Medium (application)

Q4: Judging the volumetric flow rate

Consider a 0.4 mm nozzle with line width 0.4 mm, layer height 0.25 mm, and print speed 120 mm/s. Compute the volumetric flow rate Q and check it against the standard hot-end limit (12 mm³/s).

Show answer

**Calculation:** Q = W × H × v = 0.4 × 0.25 × 120 = **12.0 mm³/s**

This exactly reaches the standard hot-end limit (12 mm³/s). It is effectively right at the upper limit—an "aggressive" setting that, depending on material and machine unit variation, risks under-extrusion. For stable printing, it is advisable to leave margin: lower the speed (e.g., 100 mm/s gives Q = 10 mm³/s), reduce the layer height, or switch to a high-flow hot end.

Q5: Designing the build orientation

You print a hook part loaded in bending in one direction, like a cantilever. To avoid failure by delamination, what build orientation should you choose? Explain based on this chapter's discussion of anisotropy.

Show answer

**Reasoning:** In FDM, Z-direction (interlayer) strength is only about 40–80% of the in-plane value (48% in this chapter's example). Under a bending load, tensile stress arises on the surface of the part. If this tensile stress is oriented to **cross the interlayer interface (pull the layers apart)** , delamination starts from the weak interface.

**Countermeasure:** Lay the part down so that the layers run **along the principal (tensile) stress direction (i.e., the extruded lines connect continuously)**. For a hook, lay it so that the hook's curvature lies within the build plane (XY plane), so the load is carried by continuous extruded lines rather than by interfaces. This avoids concentrating the principal stress on the weak Z-direction interface. Add supports to match that orientation if necessary.

### Hard (advanced)

Q6: Quantitative planning of warp countermeasures

Suppose you want to print a 150 mm-long ABS flat plate without warping. Based on this chapter's Code Example 5 (warp index ≈ ε·E, ε = α·ΔT), (1) why does ABS warp easily, (2) which parameter is most effective for reducing warping, and (3) list three concrete countermeasures.

Show answer

**(1) Why ABS warps easily:** ABS has a high Tg (105°C), so the difference ΔT from the bed/ambient temperature becomes large. Since the shrinkage strain ε = α·ΔT is proportional to ΔT, it becomes large, and its high elastic modulus E stores that shrinkage as large residual stress. In this chapter's calculation, ABS's warp index was about 4.8 times that of PLA. The longer the part (150 mm), the larger the absolute shrinkage dL = ε·L as well, increasing the corner-lifting force.

**(2) The most effective parameter:** From ε = α·ΔT, the material-intrinsic α cannot be changed, so **reducing ΔT** —that is, raising the ambient and bed temperatures to narrow the gap from Tg—is most effective. If a heated chamber can halve ΔT, the shrinkage strain and residual stress are also roughly halved.

**(3) Examples of concrete countermeasures:**

  1. Warm the surroundings with a **heated chamber / enclosure** to reduce ΔT and interlayer temperature differences (most important)
  2. Set the **bed temperature near Tg (100–110°C)** to keep the base of the part soft and relax shrinkage stress
  3. Increase contact area with a **brim or raft** to mechanically suppress corner lifting; geometry measures such as rounding corners and adding mouse ears also help

Weakening (or turning off) the cooling fan is also effective, as it prevents rapid interlayer quenching and lowers residual stress.

## Next Steps

In Chapter 2, we learned the principle of material extrusion (FDM/FFF), the physical properties of the thermoplastics used, the main process parameters, and mechanical behaviors such as interlayer bonding, anisotropy, and warping, from the perspectives of polymer physics and heat transfer. In the next Chapter 3, we will learn about high-resolution resin printing by vat photopolymerization (VPP: SLA/DLP) and the mechanism of photo-curing.

[← Back to Chapter 1](<./chapter-1.html>) [Proceed to Chapter 3 →](<./chapter-3.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2015). _Additive Manufacturing Technologies: 3D Printing, Rapid Prototyping, and Direct Digital Manufacturing_ (2nd ed.). Springer. pp. 147-186. - Systematic explanation of the material extrusion (MEX) principle and process parameters
  2. Turner, B.N., Strong, R., & Gold, S.A. (2014). "A review of melt extrusion additive manufacturing processes: I. Process design and modeling." _Rapid Prototyping Journal_ , 20(3), 192-204. - Review of flow and heat-transfer modeling of FDM
  3. Wool, R.P., & O'Connor, K.M. (1981). "A theory of crack healing in polymers." _Journal of Applied Physics_ , 52(10), 5953-5963. - Theory of polymer interface healing (basis of the (t/t_rep)^(1/4) law)
  4. Sun, Q., Rizvi, G.M., Bellehumeur, C.T., & Gu, P. (2008). "Effect of processing conditions on the bonding quality of FDM polymer filaments." _Rapid Prototyping Journal_ , 14(2), 72-80. - Experimental study of the relationship between processing conditions and interlayer bond strength
  5. Ahn, S.H., Montero, M., Odell, D., Roundy, S., & Wright, P.K. (2002). "Anisotropic material properties of fused deposition modeling ABS." _Rapid Prototyping Journal_ , 8(4), 248-257. - Representative study of the mechanical anisotropy of FDM parts
  6. de Gennes, P.G. (1971). "Reptation of a Polymer Chain in the Presence of Fixed Obstacles." _Journal of Chemical Physics_ , 55(2), 572-579. - The original paper on reptation theory

## Tools and Libraries Used

  * **NumPy** (v1.24+): Numerical computing library - <https://numpy.org/>
  * **Matplotlib** (v3.7+): Data visualization library - <https://matplotlib.org/>
  * **SciPy** (v1.10+): Scientific computing library - <https://scipy.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
