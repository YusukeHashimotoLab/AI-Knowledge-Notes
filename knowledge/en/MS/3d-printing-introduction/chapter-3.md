---
title: "Chapter 3: Vat Photopolymerization and Powder Bed Fusion — SLA/DLP/SLS/SLM"
chapter_title: "Chapter 3: Vat Photopolymerization and Powder Bed Fusion — SLA/DLP/SLS/SLM"
subtitle: Building with light and heat — the principles, materials, and applications of vat photopolymerization and powder bed fusion
reading_time: 40-50 minutes
difficulty: Intermediate
code_examples: 4
exercises: 8
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../../index.html>)›[Materials Science](<../../MS/index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 3

## Learning Objectives

Upon completing this chapter, you will be able to explain the following:

### Basic Understanding (Level 1)

  * The principle of vat photopolymerization (VPP) and the differences between SLA, DLP, and LCD
  * The photopolymerization reaction of photopolymer resins and the role of the photoinitiator
  * The principle of powder bed fusion (PBF) and the differences between SLS and SLM/DMLS
  * Metal-AM-specific concepts such as the melt pool, residual stress, and support strategy

### Practical Skills (Level 2)

  * Using the Jacobs equation to calculate cure depth from exposure
  * Estimating XY resolution from DLP/LCD pixel size or laser spot diameter
  * Calculating volumetric energy density and judging the process window
  * Using the Rosenthal equation to estimate the melt-pool cooling rate

### Applied Skills (Level 3)

  * Choosing between VPP and PBF for an application in terms of precision, strength, and material
  * Inferring the cause of build defects (lack of fusion, keyholing, warping) from process parameters
  * Explaining the material-process combinations used in applications such as dental, aerospace, and medical implants

**💡 Where this chapter fits**

Chapter 1 covered the big picture of additive manufacturing (AM) and the seven process categories of ISO/ASTM 52900; Chapter 2 covered material extrusion (FDM/FFF), the most widespread process. This chapter dives deep into two contrasting process families: **vat photopolymerization (VPP), which excels at high precision** , and **powder bed fusion (PBF), which excels at high strength and metal capability**. Studying photochemistry and thermal physics side by side gives your intuition for AM process selection a three-dimensional feel.

## 3.1 Principles of Vat Photopolymerization

**Vat photopolymerization (VPP)** is the umbrella term for processes that fill a vat with a liquid **photopolymer resin** and selectively cure it with ultraviolet (UV) or visible light to build up layers. The stereolithography (SLA) invented by Dr. Chuck Hull in 1986 is the origin of AM, and VPP has both the longest history and the highest surface quality of all AM processes.

### 3.1.1 The Chemistry of Photopolymer Resins

The key to understanding VPP is the chemical reaction that turns a liquid into a solid: **photopolymerization**. A photopolymer resin is composed mainly of the following components:

  * **Monomers / oligomers** : the molecules that cure to form the resin's backbone. Acrylate-based (fast curing, somewhat brittle) and epoxy-based (low shrinkage, high precision) are representative.
  * **Photoinitiator** : a molecule that absorbs light of a specific wavelength and generates **radicals (species with an unpaired electron)** or cations that pull the trigger on polymerization. VPP exposure wavelengths (355 nm, 385 nm, 405 nm, etc.) are designed to match the absorption band of this photoinitiator.
  * **Additives** : pigments/dyes (to tune the light's penetration depth), stabilizers (to suppress dark reactions during storage), UV absorbers (to limit overcuring), and so on.

In the case of radical polymerization, the reaction proceeds in three stages:

Initiation: photoinitiator + light (hν) → radical (R·)  
Propagation: R· + monomer → R−monomer· → … (chain growth into a polymer)  
Termination: recombination of radicals, or deactivation by oxygen 

An important phenomenon here is **oxygen inhibition**. Oxygen in the air deactivates radicals, so curing near the liquid surface is hindered. This is both a drawback and, as we will see, the operating principle behind the **dead zone (a thin, intentionally uncured layer)** used in the DLP-based continuous process (CLIP).

### 3.1.2 Comparing SLA, DLP, and LCD

VPP splits into three main approaches according to how the light is delivered. They share the same photochemistry, but the light source and drawing method differ, which changes the balance of speed, resolution, and cost.
    
    
    flowchart TD
        VPP[Vat Photopolymerization] --> SLA[SLA  
    UV laser point-scanning]
        VPP --> DLP[DLP  
    DMD projector area exposure]
        VPP --> LCD[LCD-MSLA  
    LCD-mask area exposure]
    
        SLA --> SLA_C[High precision, large builds  
    slow due to point scanning]
        DLP --> DLP_C[Fast, whole-area at once  
    resolution depends on pixel count]
        LCD --> LCD_C[Low cost, widespread  
    LCD lifetime is a concern]
    
        style VPP fill:#fff3e0
        style SLA fill:#e3f2fd
        style DLP fill:#e8f5e9
        style LCD fill:#f3e5f5
            

Approach | Light source / drawing | Speed | What sets the resolution | Cost range  
---|---|---|---|---  
**SLA**  
(Stereolithography) | UV laser (355 nm) point-scanned by a galvanometer mirror | Slow (point by point) | Laser spot diameter (50-150 μm) | Medium-high ($3,000-$250,000)  
**DLP**  
(Digital Light Processing) | A DMD (micro-mirror array) projects a pattern, exposing the whole area at once | Fast (whole layer at once) | Projector pixel size | Medium ($500-$50,000)  
**LCD-MSLA**  
(Masked SLA) | A UV-LED backlight plus an LCD mask exposes the whole area at once | Fast (whole layer at once) | LCD panel pixel size | Low ($200-$1,000)  
  
Because SLA scans point by point, its build time scales with "area × number of layers," but it can focus the laser very finely, making it suitable for large, high-precision builds. DLP and LCD expose an entire layer at once, so **build time does not change no matter how many parts you place on that layer** — the reason they are prized in dental mass production. The trade-off is that resolution is fixed by the pixel size, so widening the build area makes each pixel coarser.

### 3.1.3 Cure Depth and the Jacobs Equation (Code Example 1)

The most fundamental element of VPP process design is predicting the **cure depth (C d)**. As light travels through the resin it is absorbed and decays exponentially (the Beer–Lambert law). As a result, cure depth is proportional to the logarithm of exposure, expressed by the **Jacobs equation (Jacobs working curve)** :

Cd = Dp · ln( Emax / Ec ) 

The symbols mean the following:

  * **C d**: cure depth — the thickness of resin solidified by a single exposure.
  * **D p**: penetration depth — the depth at which light intensity decays to 1/e (about 37%); a material constant set by the resin and wavelength.
  * **E max**: exposure at the resin surface (mJ/cm²).
  * **E c**: critical exposure — the threshold above which gelation (solidification) first begins.

Cure depth is always set a little larger than the layer thickness. This excess is called **overcure** , and it is needed to bond firmly to the previous layer and prevent delamination. The code below computes cure depth as a function of exposure, and the exposure needed to reach a target cure depth.
    
    
    import numpy as np
    
    # Jacobs equation: Cd = Dp * ln(Emax / Ec)
    Dp = 0.14   # mm, resin penetration depth (~140 um, standard resin)
    Ec = 6.5    # mJ/cm^2, critical (gel) exposure
    exposures = [10, 20, 40, 80, 160]  # mJ/cm^2 at the surface
    print(f"Dp (penetration depth) = {Dp*1000:.0f} um, Ec (critical exposure) = {Ec} mJ/cm^2")
    print(f"{'Exposure Emax':>16s} | {'Cure depth Cd':>14s}")
    for E in exposures:
        Cd = Dp * np.log(E / Ec)
        print(f"{E:>10d} mJ/cm^2 | {Cd*1000:>10.1f} um")
    
    # Required exposure for a target cure depth (layer + overcure for bonding)
    layer = 0.050            # 50 um layer thickness
    overcure = 0.020         # 20 um extra to bond to the previous layer
    target = layer + overcure
    E_req = Ec * np.exp(target / Dp)
    print(f"\nTarget cure depth = {target*1000:.0f} um (layer 50 + overcure 20)")
    print(f"Required surface exposure Emax = {E_req:.1f} mJ/cm^2")
    

**Execution result:**
    
    
    Dp (penetration depth) = 140 um, Ec (critical exposure) = 6.5 mJ/cm^2
       Exposure Emax |  Cure depth Cd
            10 mJ/cm^2 |       60.3 um
            20 mJ/cm^2 |      157.4 um
            40 mJ/cm^2 |      254.4 um
            80 mJ/cm^2 |      351.4 um
           160 mJ/cm^2 |      448.5 um
    
    Target cure depth = 70 um (layer 50 + overcure 20)
    Required surface exposure Emax = 10.7 mJ/cm^2
    

Note that doubling the exposure only increases cure depth logarithmically. This is the essence of the Jacobs working curve. In practice, engineers plot cure depth against exposure on a semi-log scale, fit a line, and read Dp from the slope and Ec from the intercept. The calculation above shows that adding 20 μm of overcure to a 50 μm layer requires only 10.7 mJ/cm² of exposure.

### 3.1.4 Factors That Set Resolution, and Post-Curing (Code Example 2)

VPP's high resolution is set by several factors, and it is important that the XY (in-plane) and Z (build) directions are governed by different things:

  * **XY resolution** : for DLP/LCD it is the **pixel size** (build width ÷ pixel count); for SLA it is the **laser spot diameter**.
  * **Z resolution** : set by the layer thickness (25-100 μm) and the cure bleed caused by the resin's penetration depth Dp.
  * **Light scattering / bleed** : if pigments or scattering spread the light sideways, features cure thicker than designed. Adding a UV absorber suppresses this bleed.

The code below computes the pixel size (i.e., the lower bound of XY resolution) for representative DLP/LCD panels, and the SLA spot diameter.
    
    
    import numpy as np
    
    # DLP / LCD masked exposure: XY resolution is set by the projected pixel size
    panels = [
        ("Full-HD DLP", 1920, 120.0),    # px across, build width mm
        ("4K DLP",      3840, 192.0),
        ("8K mono LCD", 7680, 218.88),
    ]
    print(f"{'Panel':<14s} | {'px (X)':>7s} | {'build X':>8s} | {'pixel = XY res':>14s}")
    for name, px, build_x in panels:
        pixel_um = build_x / px * 1000.0
        print(f"{name:<14s} | {px:>7d} | {build_x:>6.1f}mm | {pixel_um:>11.1f} um")
    
    # SLA: XY resolution is governed by the laser spot diameter (Gaussian 1/e^2)
    spot_diam_um = 85.0     # typical 355 nm galvo-scanned spot
    min_feature = spot_diam_um  # positive features roughly track the spot
    print(f"\nSLA laser spot diameter = {spot_diam_um:.0f} um "
          f"(min. positive feature ~ {min_feature:.0f} um)")
    

**Execution result:**
    
    
    Panel          |  px (X) |  build X | pixel = XY res
    Full-HD DLP    |    1920 |  120.0mm |        62.5 um
    4K DLP         |    3840 |  192.0mm |        50.0 um
    8K mono LCD    |    7680 |  218.9mm |        28.5 um
    
    SLA laser spot diameter = 85 um (min. positive feature ~ 85 um)
    

Even an 8K LCD has a pixel size of about 28.5 μm, finer than the SLA spot diameter of 85 μm. However, an LCD's pixels become coarser as the build area grows, so SLA still has the advantage when large size and high precision are both required.

**⚠️ Do not skip post-curing**

A part straight off a VPP machine is called a **green part** ; polymerization is incomplete and it has only about 30-70% of its final mechanical properties. Correct post-processing is a three-step sequence:

  1. **Washing** : remove uncured resin with isopropyl alcohol (IPA) or similar. Residual resin causes surface tackiness and dimensional error.
  2. **Drying** : fully evaporate the solvent.
  3. **Post-curing** : expose the part to additional light in a UV chamber to complete polymerization. Strength, hardness, and heat resistance improve significantly.

Skip this step and the part will warp or become brittle over time. "Print and done" does not apply — a major difference from FDM.

## 3.2 Principles of Powder Bed Fusion

**Powder bed fusion (PBF)** spreads a thin layer of powder and selectively melts or sinters it with a laser or electron beam, then lets it solidify to build up layers. Because the surrounding unmelted powder supports the part, **supports are minimal** , and for metals the strength rivals forged material — its greatest characteristic. Here the star of the show is not photochemistry but **thermal physics (heat transfer and solidification)**.

### 3.2.1 SLS: Sintering Polymer Powder

**SLS (Selective Laser Sintering)** uses a laser to **sinter (partially bond particles together near the melting point)** polymer powder. The key point is that it does not fully melt the material but melts the particle surfaces to bond them.

  * **Representative materials** : PA12 (nylon 12) is the industry standard. Also PA11, TPU (flexible), glass-fiber-reinforced nylon, and others.
  * **No supports needed** : the unmelted powder acts as the support, so complex internal structures and moving parts can be built as one piece.
  * **Powder refresh** : unmelted powder degrades through its thermal history, so it is mixed with virgin powder for reuse (a refresh rate of 50-70% is typical).
  * **Importance of preheating** : the whole build chamber is preheated to just below the melting point (about 170°C for PA12), so the laser only needs to add a little energy to sinter. Insufficient preheat leads to warping.

### 3.2.2 SLM/DMLS: Fully Melting Metal Powder

**SLM (Selective Laser Melting)** and **DMLS (Direct Metal Laser Sintering)** **fully melt** metal powder to build high-density parts (relative density above 99%). The difference between the two is largely a matter of vendor naming, and today they are treated as nearly synonymous (collectively also called L-PBF: Laser Powder Bed Fusion).

  * **Representative materials** : Ti-6Al-4V (titanium alloy; aerospace, medical), AlSi10Mg (aluminum alloy; lightweight parts), 316L stainless steel, Inconel 718 (Ni superalloy; heat resistance), Co-Cr alloy (dental).
  * **Inert atmosphere** : building is done under Ar or N₂ to prevent oxidation. Titanium and aluminum have a high affinity for oxygen, and there is also a fire risk in handling the powder.
  * **Supports required** : for metals, supports are needed to conduct heat away and suppress warping from residual stress, especially on overhangs and large bottom faces (unlike SLS, powder alone cannot hold the part).

A related approach is **EBM (Electron Beam Melting)**. It uses an electron beam as the heat source and builds under vacuum while preheating to high temperature (650-1000°C), giving low residual stress and fast build speed — but it requires a vacuum and produces a rougher surface.

### 3.2.3 Why Powder Properties Matter

PBF quality depends heavily on **powder quality**. With the same machine and the same parameters, poor powder will not produce good parts. The main powder properties are:

Property | Typical value (metal L-PBF) | Effect on the build  
---|---|---  
Particle size distribution (PSD) | 15-45 μm (D50 ≈ 30 μm) | Too fine reduces flowability; too coarse increases surface roughness  
Particle shape (sphericity) | Spherical (gas-atomized powder) | The more spherical, the better the flowability and packing density  
Apparent / tap density | Tap density > 4.0 g/cm³ (steel) | Higher packs a denser layer and reduces defects  
Flowability (Hall flow) | 15-30 s/50g | Poor flow gives uneven layers and causes lack-of-fusion defects  
  
Metal powder is mainly produced by **gas atomization (breaking up molten metal into fine particles with a high-pressure gas jet)** , yielding nearly spherical particles. Gas pores trapped inside particles become seeds for post-build defects, so powder makers manage both sphericity and internal defects.

### 3.2.4 Laser-Material Interaction and Energy Density (Code Example 3)

The central concept in PBF process design is **volumetric energy density (VED)**. It represents the laser energy delivered per unit volume and is calculated from four main parameters:

E = P / ( v · h · t ) 

Here P is laser power (W), v is scan speed (mm/s), h is hatch spacing (the distance between adjacent scan lines, mm), and t is layer thickness (mm). If VED is too low, insufficient melting produces **lack-of-fusion porosity** ; if it is too high, the melt pool digs deep and produces **keyhole porosity (elongated voids caused by vaporization)**. Between them lies the **process window** where dense parts are obtained.
    
    
    import numpy as np
    
    def ved(P, v, h, t):
        # P [W], v [mm/s], h hatch [mm], t layer [mm] -> J/mm^3
        return P / (v * h * t)
    
    # (label, laser power W, scan speed mm/s, hatch mm, layer mm)
    params = [
        ("Lack-of-fusion", 170, 1400, 0.13, 0.03),
        ("Optimal Ti-6Al-4V", 280, 1200, 0.14, 0.03),
        ("Keyholing / over-melt", 370, 650, 0.10, 0.03),
    ]
    print(f"{'Regime':<22s} | {'P':>4s} | {'v':>5s} | {'h':>5s} | {'t':>5s} | {'VED':>8s}")
    for name, P, v, h, t in params:
        E = ved(P, v, h, t)
        if E < 40:
            tag = "too low -> pores"
        elif E <= 70:
            tag = "dense (>99%)"
        else:
            tag = "too high -> keyhole"
        print(f"{name:<22s} | {P:>4d} | {v:>5d} | {h:>4.2f} | {t:>4.2f} | "
              f"{E:>6.1f} J/mm^3  ({tag})")
    

**Execution result:**
    
    
    Regime                 |    P |     v |     h |     t |      VED
    Lack-of-fusion         |  170 |  1400 | 0.13 | 0.03 |   31.1 J/mm^3  (too low -> pores)
    Optimal Ti-6Al-4V      |  280 |  1200 | 0.14 | 0.03 |   55.6 J/mm^3  (dense (>99%))
    Keyholing / over-melt  |  370 |   650 | 0.10 | 0.03 |  189.7 J/mm^3  (too high -> keyhole)
    

The VED suited to densifying Ti-6Al-4V is roughly 40-70 J/mm³, and the "Optimal" condition above (55.6 J/mm³) falls within that range. VED is a convenient index, but the same VED can produce differently shaped melt pools depending on the power-speed combination, so it is only a starting point; in practice you vary power and speed independently to build a process map.

### 3.2.5 Melt-Pool Dynamics and Cooling Rate (Code Example 4)

When the laser melts the powder, a tiny **melt pool** tens to hundreds of μm in diameter and depth forms, moving and solidifying as the laser travels. The behavior of this melt pool determines the part's density, microstructure, and residual stress.

The temperature field of a moving point heat source can be approximated by the classic **Rosenthal equation**. Here we compute the "cooling rate at the melting isotherm," which is often used in practice. The thick-plate cooling rate is given by:

dT/dt = 2π · k · v · (Tm − T0)² / ( η · P ) 

k is thermal conductivity, v is scan speed, Tm is the melting point, T0 is the preheat temperature, η is the laser absorptivity, and P is power. The cooling rate increases in proportion to scan speed, and this extremely fast cooling (10⁵-10⁷ K/s) produces the fine, metastable microstructures characteristic of PBF.
    
    
    import numpy as np
    
    # Thick-plate Rosenthal approximation: cooling rate at the melting isotherm
    #   dT/dt = 2*pi*k*v*(Tm - T0)^2 / (eta * P)
    k = 7.0        # W/(m.K), thermal conductivity of Ti-6Al-4V (solid, avg)
    Tm = 1923.0    # K, melting point (~1650 C)
    T0 = 473.0     # K, build-plate preheat (~200 C)
    eta = 0.40     # laser absorptivity of the powder bed
    P = 280.0      # W
    print(f"k={k} W/mK, Tm={Tm:.0f} K, T0={T0:.0f} K, eta={eta}, P={P:.0f} W")
    print(f"{'scan speed v':>14s} | {'cooling rate dT/dt':>20s}")
    for v_mm in [400, 800, 1200, 1600]:
        v = v_mm / 1000.0  # m/s
        dTdt = 2 * np.pi * k * v * (Tm - T0) ** 2 / (eta * P)
        print(f"{v_mm:>10d} mm/s | {dTdt:>16.3e} K/s")
    
    # Melt-pool length grows with linear energy input (P/v)
    print("\nLinear energy input (P/v) and relative melt-pool length:")
    for v_mm in [400, 800, 1200, 1600]:
        lin = P / (v_mm / 1000.0)  # J/m
        print(f"  v={v_mm:>4d} mm/s -> P/v = {lin:>6.0f} J/m")
    

**Execution result:**
    
    
    k=7.0 W/mK, Tm=1923 K, T0=473 K, eta=0.4, P=280 W
      scan speed v |   cooling rate dT/dt
           400 mm/s |        3.303e+05 K/s
           800 mm/s |        6.605e+05 K/s
          1200 mm/s |        9.908e+05 K/s
          1600 mm/s |        1.321e+06 K/s
    
    Linear energy input (P/v) and relative melt-pool length:
      v= 400 mm/s -> P/v =    700 J/m
      v= 800 mm/s -> P/v =    350 J/m
      v=1200 mm/s -> P/v =    233 J/m
      v=1600 mm/s -> P/v =    175 J/m
    

You can read off the opposing relationship: raising the scan speed increases the cooling rate (finer microstructure) while lowering the linear energy input (a smaller, shallower melt pool). In practice, engineers look for a balance point that ensures enough melting for densification while giving the desired fine microstructure and low residual stress.

### 3.2.6 Thermal Stress, Residual Stress, and Support Strategy

The repeated cycle of rapid heating and cooling creates the biggest challenge of metal PBF: **residual stress**. As a molten layer solidifies and shrinks, it is constrained by the layer below and cannot contract freely, so tensile stress accumulates inside. When this exceeds a limit, it appears as **warping, delamination, or cracking**.

**💡 Main measures to reduce residual stress**

  * **Plate preheating** : eases the temperature gradient and reduces stress accumulation (the effect of raising T0 in the equation). This is why EBM's high-temperature preheat is advantageous.
  * **Scan strategy** : rotating the scan direction layer by layer (e.g., 67° rotation) or dividing it into islands (island scanning) to distribute stress.
  * **Support design** : acting as anchors that conduct heat away and fix the part, suppressing warping of overhangs and bottom faces.
  * **Post-build stress-relief annealing** : heat-treat the part before cutting it off the plate to release stress (cut it off first and it warps).

Supports here are not merely "props" as in VPP or FDM; their essential role is as a **heat sink (a heat-conduction path)**. But more supports mean more material, time, and post-processing, so it is important to design for minimal supports by optimizing the build orientation (keeping overhangs at 45° or steeper, reducing the surfaces that need supports).

## 3.3 Process Comparison and Materials

### 3.3.1 Comparing VPP and PBF

The two process families we have seen are contrasting in both principle and strengths. As axes for application selection, we organize the main viewpoints.

Viewpoint | Vat Photopolymerization (VPP) | Powder Bed Fusion (PBF)  
---|---|---  
Bonding principle | Photopolymerization (photochemistry) | Melting / sintering (thermal physics)  
Materials | Photopolymer resin (polymer) | Polymer powder (SLS), metal powder (SLM/DMLS)  
Precision / surface quality | Very high (Ra < 5 μm, XY 25-100 μm) | Moderate (Ra 5-20 μm, post-processing assumed)  
Mechanical strength | Medium-low (resin, improved by post-curing) | High (metals rival forged material, 500-1200 MPa)  
Supports | Required (to hold the resin's weight) | SLS none, SLM required (heat removal, stress)  
Post-processing | Wash → dry → post-cure | Depowder → stress-relief anneal → support removal → finishing  
Machine cost range | $200-$250,000 | $100,000-$1,500,000 (metal)  
  
Put very simply, the division of labor is **"VPP when looks and precision are paramount, PBF (metal) when strength and function are paramount."** That said, real projects tangle up material availability, certification, cost, and quantity, so use this table as a starting point and select from all seven processes of Chapter 1.

### 3.3.2 Representative Materials

#### VPP resins

  * **Standard resin (acrylate-based)** : general purpose. Fast curing and inexpensive, but somewhat brittle and prone to yellowing/degradation under UV.
  * **Tough / engineering resin** : formulated for ABS-like toughness. For functional-test prototypes.
  * **High-temperature resin** : heat-deflection temperature above 200°C. For prototyping molds and fixtures.
  * **Castable resin** : formulated to leave almost no ash when burned out. For jewelry casting patterns.
  * **Biocompatible resin** : formulations with biocompatibility certification. For dental surgical guides and dentures.

#### PBF powders

  * **PA12 (nylon 12)** : the standard SLS polymer. Excellent chemical resistance and toughness; enables one-piece builds with moving parts.
  * **Ti-6Al-4V** : a titanium alloy with excellent specific strength, corrosion resistance, and biocompatibility. The workhorse for aerospace parts and medical implants.
  * **AlSi10Mg** : a casting-aluminum composition. For lightweight brackets and heat sinks; high thermal conductivity favors thermal-management designs.
  * **316L / 17-4PH stainless steel** : fixtures and functional parts leveraging corrosion resistance (316L) or high strength (17-4PH).
  * **Inconel 718** : a Ni-based superalloy with excellent high-temperature strength. For heat-resistant parts such as turbines and combustors.

## 3.4 Application Areas

VPP and PBF are being commercialized in fields where their characteristics mesh with the requirements. Let us look at three representative areas.

### 3.4.1 Dental (mostly VPP, some PBF)

Dental is one of VPP's most successful application areas. Each patient's geometry differs (so customization is intrinsically required), and DLP/LCD area exposure can build many cases at once, achieving both productivity and customization.

  * **Orthodontic models** : plaster-replacement models for thermoforming clear aligners. On the order of millions per year.
  * **Surgical guides** : biocompatible-resin guides that accurately direct implant placement.
  * **Dentures / crowns** : resin interim restorations, or metal frameworks built in Co-Cr alloy by SLM.

### 3.4.2 Aerospace (mostly metal PBF)

Aerospace is the driver of metal PBF. Because **weight reduction translates directly into fuel economy and payload** , the benefits of topology-optimized complex geometry and part consolidation are enormous.

  * **Fuel injection nozzles** : GE Aviation's nozzle for the LEAP engine consolidated 20 conventional parts into one, achieving 25% weight reduction and improved durability.
  * **Lightweight brackets and structural parts** : 40-60% lighter than conventional designs in Ti-6Al-4V or AlSi10Mg.
  * **Heat exchangers** : one-piece structures with complex internal channels, achieving both performance and light weight.

### 3.4.3 Medical Implants (mostly metal PBF)

Medical implants bring together several strengths of metal PBF at once: patient-specific geometry, biocompatibility, and porous structures.

  * **Hip implants / spinal cages** : made of Ti-6Al-4V. A porous lattice structure is built on the surface to promote **osseointegration (the direct bonding of bone to the implant)** as bone grows into it.
  * **Cranial / maxillofacial implants** : fully custom shapes built to fit a patient's defect from CT data.
  * **Dental implants** : Ti fixtures, and Co-Cr alloy prosthetic frameworks.

**✅ Learning process selection in reverse from applications**

Tracing "why is that process chosen in that field?" builds your instinct for process selection. Dental uses DLP because of **customization × precision × productivity** ; aerospace uses metal PBF because of **weight reduction × strength** ; medical implants use SLM of Ti-6Al-4V because of **patient-specific geometry × biocompatibility × porous structures**. Decompose the requirements and match them against each process's physical strengths. Once you internalize this pattern of thinking, you can handle unfamiliar applications too.

## Exercises

Let us consolidate the principles of VPP and PBF learned in this chapter through calculation and discussion. Each question has a sample answer. Think it through yourself first, then check.

### Easy

Q1: Choosing among SLA, DLP, and LCD

For each of the three cases below, choose the most suitable VPP approach (SLA / DLP / LCD) and state your reason.

  1. A dental lab wants to mass-produce by placing 50 small models in a single batch.
  2. A hobbyist wants to print figurines on a low budget.
  3. A research lab wants to make a large (30 cm wide), high-precision optical-part prototype.

Show answer

  1. **DLP** : because area exposure means build time does not change no matter how many parts are on a layer, ideal for mass production.
  2. **LCD-MSLA** : the lowest cost ($200-$1,000) with precision sufficient for personal use.
  3. **SLA** : laser point-scanning keeps resolution from dropping as the build area grows, ideal for large and high-precision work (DLP/LCD pixels become coarse when scaled up).

Q2: Calculating cure depth

For a resin with penetration depth Dp = 0.10 mm and critical exposure Ec = 8.0 mJ/cm², calculate the cure depth Cd when a surface exposure of Emax = 60 mJ/cm² is applied, using the Jacobs equation.

Show answer

Cd = Dp · ln(Emax / Ec) = 0.10 · ln(60 / 8.0) = 0.10 · ln(7.5) = 0.10 · 2.015 = **0.2015 mm ≈ 202 μm**.

This gives ample overcure over a 50 μm layer, so interlayer bonding is fine.

Q3: The difference between SLS and SLM

Explain the difference between SLS (Selective Laser Sintering) and SLM (Selective Laser Melting) in three points: (1) target material, (2) the laser's action (sintering vs. full melting), and (3) whether supports are needed.

Show answer

  * **Target material** : SLS uses polymer powder (e.g., PA12); SLM uses metal powder (e.g., Ti-6Al-4V).
  * **Laser action** : SLS "sinters," melting particle surfaces just below the melting point to bond them; SLM "fully melts" the metal to obtain a dense body of over 99% relative density.
  * **Supports** : SLS needs no supports because unmelted powder holds the part; SLM requires supports for heat removal and residual-stress control.

### Medium

Q4: Calculating VED and judging the process

You build Ti-6Al-4V with laser power P = 250 W, scan speed v = 1000 mm/s, hatch spacing h = 0.12 mm, and layer thickness t = 0.03 mm. Calculate the volumetric energy density (VED) and judge whether it falls in the range suited to densification (40-70 J/mm³).

Show answer

E = P / (v · h · t) = 250 / (1000 · 0.12 · 0.03) = 250 / 3.6 = **69.4 J/mm³**.

It sits near the top of the 40-70 J/mm³ range, so it is suited to densification. However, being near the upper limit, raising power further or lowering speed would increase the risk of keyhole defects. To keep a margin, slightly raising the scan speed to bring VED to around 55-60 is the safer choice.

Q5: Inferring the cause of a build defect

A CT inspection of a part built by metal L-PBF found many elongated, irregular voids along the scan lines, between layers. Calculating the VED gives 28 J/mm³. Give (1) the type of defect, (2) the cause, and (3) two remedies.

Show answer

  * **Type of defect** : lack-of-fusion porosity. Characterized by irregular voids along the layers and scan lines.
  * **Cause** : VED of 28 J/mm³ is well below the 40-70 J/mm³ needed for densification; melting is insufficient and the material fails to fuse with adjacent beads and the layer below.
  * **Remedies (examples)** : (a) raise laser power, (b) lower scan speed, (c) narrow the hatch spacing, (d) reduce layer thickness — all raise VED. Also check whether poor powder flowability caused uneven spreading, another cause of lack of fusion, so verify powder quality and recoater settings.

Q6: Designing the post-processing sequence

Explain, step by step, the post-processing sequence needed to make a dental surgical guide DLP-printed in biocompatible resin ready for clinical use. State the purpose of each step.

Show answer

  1. **Support removal** : carefully remove supports from the part, minimizing marks on the contact surfaces.
  2. **Washing (IPA, etc.)** : remove uncured resin from the surface. Residual resin causes dimensional error and reduced biocompatibility; strictly follow the prescribed cleaning protocol for biomedical use.
  3. **Drying** : fully evaporate the solvent.
  4. **Post-curing** : complete polymerization in a UV chamber to ensure mechanical strength and biocompatibility. Follow the manufacturer's specified wavelength, time, and temperature.
  5. **(If needed) sterilization** : sterilize by autoclave or similar for clinical use, choosing a method matched to the resin's heat resistance.

Skipping post-curing leaves unpolymerized components, causing insufficient strength and biocompatibility problems, so it cannot be omitted.

### Hard

Q7: The relationship between cooling rate and microstructure

You build Ti-6Al-4V by L-PBF. Using the thick-plate Rosenthal approximation dT/dt = 2π·k·v·(Tm−T0)²/(η·P), with k = 7 W/mK, Tm = 1923 K, η = 0.4, and P = 280 W: (1) compute the cooling rate for no preheat (T0 = 300 K) at v = 1000 mm/s, (2) compute it with preheat (T0 = 473 K) under the same conditions, and (3) discuss the effect of preheating on residual stress in terms of cooling rate.

Show answer

**(1) No preheat (T 0 = 300 K):**  
(Tm−T0) = 1923 − 300 = 1623 K  
dT/dt = 2π · 7 · 1.0 · 1623² / (0.4 · 280)  
= 43.98 · 2,634,129 / 112  
= 115,858,000 / 112 ≈ **1.03 × 10⁶ K/s**

**(2) With preheat (T 0 = 473 K):**  
(Tm−T0) = 1923 − 473 = 1450 K  
dT/dt = 2π · 7 · 1.0 · 1450² / (0.4 · 280)  
= 43.98 · 2,102,500 / 112  
= 92,470,000 / 112 ≈ **8.26 × 10⁵ K/s**

**(3) Discussion:**  
Preheating lowers the cooling rate from about 1.03×10⁶ to 8.26×10⁵ K/s, roughly a 20% reduction. Because the temperature-difference term enters as a square, raising T0 eases both the cooling rate and the temperature gradient. A gentler temperature gradient suppresses the buildup of residual stress from differential thermal contraction between layers, lowering the risk of warping and cracking. This is why plate preheating and EBM's high-temperature preheat are effective against residual stress. However, a lower cooling rate makes the microstructure somewhat coarser, so stress reduction and microstructure refinement are in a trade-off relationship.

Q8: Overall process selection, VPP vs. PBF

For each of the two parts below, choose the optimal process (one VPP approach or one PBF approach), including the material, and give three reasons each.

  1. A patient-specific hip implant stem (requiring high strength, biocompatibility, and a porous surface that promotes osseointegration).
  2. A complex jewelry pattern (requiring an extremely smooth surface, fine detail, and leaving no ash on casting).

Show answer

**Part 1: Hip implant stem → SLM, material Ti-6Al-4V**

  1. **High strength** : full melting gives over 99% relative density and fatigue properties rivaling forged material, essential for a load-bearing implant.
  2. **Biocompatibility** : Ti-6Al-4V has an extensive track record of long-term implantation and excellent corrosion resistance and biocompatibility.
  3. **Ability to build porous structures** : lattice structures can be built as one piece, creating a porous surface that promotes osseointegration; patient-specific geometry from CT data is also achievable.

**Part 2: Jewelry pattern → SLA or DLP, material castable resin**

  1. **Extremely high surface quality** : VPP has Ra < 5 μm with layer lines barely visible, minimizing the polishing effort.
  2. **Fine detail** : the SLA spot diameter and DLP pixel size are tens of μm, reproducing intricate jewelry detail.
  3. **Castable resin** : formulations exist that leave almost no ash on burnout, usable directly as a lost-wax casting pattern. Metal PBF has insufficient surface quality and is unsuited to this use.

## Summary

In this chapter, we studied two contrasting AM process families:

  * **Vat photopolymerization (VPP)** : uses the photopolymerization of photopolymer resins to achieve high precision and surface quality across the SLA, DLP, and LCD approaches. You design cure depth with the Jacobs equation, and washing and post-curing govern quality. It shines in dental, jewelry, and medical models.
  * **Powder bed fusion (PBF)** : melts and sinters powder to build up layers. SLS handles polymers; SLM/DMLS handle metals. You design the process window with volumetric energy density, and controlling the melt-pool cooling rate and residual stress is the crux of metal building. End-use parts are mass-produced in aerospace and medical implants.
  * **Process selection** : start from "VPP for precision and looks, PBF for strength and function," then decide by matching against material, certification, cost, and quantity. Decomposing an application's requirements and comparing them against each process's physical strengths is a pattern of thinking that becomes your ability to handle unfamiliar applications.

## Next Steps

Chapter 3 covered the principles, materials, and applications of two processes that carry high precision and high strength: vat photopolymerization (VPP) and powder bed fusion (PBF). Chapter 4 moves on to optimizing build parameters, systematically understanding build defects, and the mindset of quality assurance.

[← Back to Chapter 2](<./chapter-2.html>) [On to Chapter 4 →](<./chapter-4.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2015). _Additive Manufacturing Technologies: 3D Printing, Rapid Prototyping, and Direct Digital Manufacturing_ (2nd ed.). Springer. pp. 63-106, 107-145. - A standard textbook systematically explaining the principles, machines, and materials of VPP and PBF
  2. Jacobs, P.F. (1992). _Rapid Prototyping & Manufacturing: Fundamentals of StereoLithography_. Society of Manufacturing Engineers. - The original source of the cure-depth working curve (Jacobs equation)
  3. Bourell, D., et al. (2017). "Materials for additive manufacturing." _CIRP Annals_ , 66(2), 659-681. - A comprehensive review of AM materials (resins, polymer powders, metal powders)
  4. DebRoy, T., et al. (2018). "Additive manufacturing of metallic components – Process, structure and properties." _Progress in Materials Science_ , 92, 112-224. - The definitive review of metal AM, covering melt pool, solidification, microstructure, and residual stress
  5. Rosenthal, D. (1946). "The theory of moving sources of heat and its application to metal treatments." _Transactions of the ASME_ , 68, 849-866. - The original source of the moving-heat-source temperature field (Rosenthal equation)
  6. King, W.E., et al. (2015). "Laser powder bed fusion additive manufacturing of metals; physics, computational, and materials challenges." _Applied Physics Reviews_ , 2, 041304. - An explanation of keyhole and lack-of-fusion defects and the process window in L-PBF
  7. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. - The international standard for AM terminology and process classification, including VPP and PBF

## Tools and Libraries Used

  * **NumPy** (v1.24+): a numerical computing library - <https://numpy.org/>
  * **Matplotlib** (v3.7+): a data visualization library - <https://matplotlib.org/>

### Disclaimer

  * This content is provided for educational, research, and informational purposes only, and does not constitute professional advice (legal, accounting, technical assurance, or otherwise).
  * This content and any accompanying code examples are provided "AS IS," without warranty of any kind, express or implied, including without limitation warranties of merchantability, fitness for a particular purpose, non-infringement, or accuracy/completeness of operation or safety.
  * The creator and Tohoku University assume no responsibility for the content, availability, or safety of external links or third-party data, tools, or libraries.
  * To the maximum extent permitted by applicable law, the creator and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content of this material may be changed, updated, or discontinued without notice.
  * The copyright and license of this content follow the terms specified (e.g., CC BY 4.0). Such licenses typically include a no-warranty clause.
