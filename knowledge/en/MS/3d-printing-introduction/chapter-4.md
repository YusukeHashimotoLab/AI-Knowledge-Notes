---
title: "Chapter 4: Material Jetting, Binder Jetting, and Other AM Technologies"
chapter_title: "Chapter 4: Material Jetting, Binder Jetting, and Other AM Technologies"
subtitle: Droplet deposition, powder binding, large-scale deposition, emerging methods, and how to select a process
reading_time: 40-45 minutes
difficulty: Intermediate to Advanced
code_examples: 3
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-4.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../../index.html>)›[Materials Science](<../../MS/index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 4

## Learning Objectives

By completing this chapter, you will be able to explain the following:

### Basic Understanding (Level 1)

  * The droplet-deposition principle of material jetting (MJ/PolyJet) and how multi-material and full-color builds work
  * The post-processing chain of binder jetting (BJ), from the green part (the unsintered formed body) through sintering and infiltration
  * The role of directed energy deposition (DED/LENS) and sheet lamination (LOM/UAM)
  * An overview of hybrid manufacturing (integrating additive and subtractive machining) and emerging technologies (bioprinting, 4D printing)

### Practical Skills (Level 2)

  * Compute the dimensionless numbers of an inkjet droplet (Ohnesorge number, Weber number, Z number) and judge printability
  * Estimate sintering shrinkage from relative density and determine the scaling factor for green-part compensation
  * Compare and select AM processes quantitatively using weighted scoring

### Applied Skills (Level 3)

  * Choose the optimal process for an application from the trade-offs among accuracy, surface quality, speed, material range, cost, and strength
  * Build a realistic manufacturing plan that accounts for each method's post-processing cost and yield risk
  * Assess the applicability and technical maturity of emerging technologies without overstatement

**💡 Where this chapter fits**

Chapter 1 covered material extrusion (MEX), and Chapters 2 and 3 covered vat photopolymerization (VPP) and powder bed fusion (PBF). This chapter takes a cross-cutting look at the remaining major processes: **material jetting, binder jetting, directed energy deposition, and sheet lamination** , and finishes by addressing the question of "which method to choose, and when." The goal is less about operating specific machines and more about acquiring the **axes for comparing processes**.

## 4.1 Material Jetting (MJ)

### 4.1.1 Principle: "printing" droplets to build layers

Material jetting (MJ) **jets liquid material as tiny droplets using the same principle as an inkjet printer, then cures them in place with ultraviolet (UV) light to build up layers**. It is also called **PolyJet** , after the Stratasys trademark. The print head carries hundreds to thousands of nozzles that deposit droplets of photocurable resin (photopolymer) exactly where needed on each layer, and a UV lamp immediately behind the head solidifies them at once.

Process: atomize resin into droplets → jet from head → impact and leveling → UV curing → next layer 

Because MJ handles an entire area with a nozzle array at once, it is faster than point-scanning SLA while remaining high-resolution thanks to the small droplet size. On the other hand, its essential constraints are that **usable materials are limited to photocurable resins** and that mechanical properties (strength, heat resistance) remain moderate.

### 4.1.2 Droplet physics: the dimensionless numbers that govern printability

Whether a droplet forms "cleanly as a single drop" is decided by the balance among viscosity, surface tension, and inertia. The following dimensionless numbers express this. Each term is defined on first use.

  * **Ohnesorge number (Oh)** : the ratio of viscous force to inertial and surface-tension forces. Oh = μ / √(ρ σ D), where μ is viscosity, ρ is density, σ is surface tension, and D is nozzle diameter.
  * **Z number** : the reciprocal of Oh (Z = 1/Oh). It is widely used in inkjet research as an index of printability.
  * **Weber number (We)** : the ratio of inertial force to surface-tension force. We = ρ v² D / σ, where v is droplet velocity. It indicates whether there is enough energy to eject a droplet.
  * **Reynolds number (Re)** : the ratio of inertial force to viscous force. Re = ρ v D / μ.

Empirically, the window for stable droplet formation is **roughly 1 < Z < 10** (up to about 14 depending on the source). If Z is too small (too viscous), the droplet does not detach; if it is too large, it trails a tail and produces **satellite droplets**. In addition, a sufficient We is needed for ejection.

**⚠️ Dimensionless numbers are a guide, not a guarantee**

The 1 < Z < 10 window is an empirical rule derived from many experiments; the actual window shifts with the resin's non-Newtonian behavior (shear-dependent viscosity), the drive waveform, and nozzle geometry. It is useful for early-stage screening, but the honest approach is to leave the final judgment to droplet observation on the real machine (a drop watcher).

### 4.1.3 Multi-material and full-color builds

MJ's greatest strength is that it can **assign different materials to different nozzle groups and switch materials or colors freely within a single build**. It can create "digital materials" that blend rigid and soft resins in a gradient to produce intermediate hardnesses from rubbery to rigid, and full-color builds of more than 10 million colors by combining CMYK plus white and clear resins. In medical anatomical models, bone (hard), soft tissue (soft), and blood vessels (clear) can be built into a single piece, which is valuable for surgical planning.

### 4.1.4 Support strategy

In MJ, a separate **support material** (often gel-like or wax-like) is jetted at the same time as the build material to support overhangs and hollows. Removal is mainly of two kinds:

  * **Water-soluble / water-disintegrable support** : dissolved with a water jet or alkaline solution. It can flush out complex internal channels, but residue tends to remain in fine features.
  * **Mechanically removed support** : peeled off by hand or with a water jet. Fast, but risks damaging fine features.

The cost of support material and the labor of removal directly affect the effective cost and yield of MJ. Even though the build itself is high-precision, it is important to evaluate it by **total cost including post-processing**.

## 4.2 Binder Jetting (BJ)

### 4.2.1 Principle: "gluing" powder into shape

Binder jetting (BJ) **jets a liquid binder onto a thin powder bed with an inkjet head, gluing the powder particles together to form each layer**. Because it uses no laser or heat source to melt the material, the build itself proceeds quickly at room temperature. The formed body straight off the build is called the **green part (an unsintered, fragile state)**.

Process: spread powder → jet binder (per layer) → cure → remove green part → debind → sinter / infiltrate 

### 4.2.2 The green part and post-processing

The green part lacks strength on its own, so it is densified in post-processing. For metals and ceramics, the main routes are:

  * **Debinding** : removing the binder by heat or solvent. Done too rapidly it cracks, so careful heating is required.
  * **Sintering** : at high temperature below the melting point (around 1200-1400°C for metals), particles diffusion-bond and densify. **Large shrinkage** occurs during this step.
  * **Infiltration** : instead of sintering, a low-melting-point metal (such as bronze) is drawn into the porous green/brown part by capillary action to fill the voids. Shrinkage is small and dimensional stability is excellent, but the material becomes a composite.

### 4.2.3 Estimating sintering shrinkage and compensating dimensions

During sintering, as the relative density (the fraction of theoretical density) increases, the volume decreases and the part shrinks. Assuming isotropic shrinkage, the linear shrinkage can be expressed as the cube root of the density ratio.

linear shrinkage = (1 − (ρ_green / ρ_sinter)^(1/3)) × 100 [%] 

Here ρ_green is the green relative density and ρ_sinter is the sintered relative density. To obtain the target dimension, you **design the green part larger** to anticipate this shrinkage; the scale factor is (ρ_sinter / ρ_green)^(1/3). It is not unusual for linear shrinkage in metal BJ to reach 15-20%, and without compensating for it the part is unusable. This calculation is carried out in Code Example 2 below.

**💡 Where BJ excels**

  * **Sand molds and cores for casting** : complex cooling and gating channels are formed in one piece with no sintering required. Already in production use for large castings such as engine blocks.
  * **Metal mass-production parts** : systems such as Desktop Metal and HP Metal Jet aim for unit costs close to injection molding.
  * **Full-color gypsum models** : for memorabilia and educational models. Low in strength but inexpensive and colorful.

## 4.3 Directed Energy Deposition (DED)

Directed energy deposition (DED) **feeds metal powder or wire while melting it with a laser, electron beam, or arc, building up material on a substrate**. Because the nozzle and energy source move together, mounting them on a multi-axis robotic arm imposes few limits on build size and can handle large parts. LENS (Laser Engineered Net Shaping) is a representative trademark for the laser-powder variant.

  * **High deposition rate** : 1-5 kg/h, 10-50 times that of PBF. But precision is coarse (±0.5-2 mm).
  * **Repair and cladding** : worn turbine blades and damaged mold sections can be regenerated by depositing directly onto the existing part. This is DED's greatest practical value.
  * **Functionally graded materials** : by continuously varying the ratio of the fed powders, composition can vary with position (e.g., a tough base and a wear-resistant surface).

Rather than "creating a precise shape from scratch," DED shows its real strength in **"depositing large and fast" and "repairing what is broken."** Because finishing presupposes machining, it is closely tied to the hybrid manufacturing discussed in the next section.

## 4.4 Sheet Lamination and Hybrid Manufacturing

### 4.4.1 Sheet Lamination (SL)

Sheet lamination (SL) **stacks sheet materials such as paper, metal foil, or plastic film, bonds or welds them, and cuts the contour of each layer**. The two representative technologies are:

  * **LOM (Laminated Object Manufacturing)** : adhesive-backed paper or film is stacked and the contour cut with a laser or blade. Large and low-cost, but the interior is solid and the use is mainly visual models.
  * **UAM (Ultrasonic Additive Manufacturing)** : metal foils are solid-state joined by ultrasound and shaped by CNC machining. Because the bonding is low-temperature, it has a feature no other method offers: **sensors and optical fibers can be embedded inside the build**.

### 4.4.2 Hybrid Manufacturing (integrating additive and subtractive)

Hybrid manufacturing is an approach that **performs additive processing (AM) and subtractive processing (machining) alternately within a single machine**. By depositing a preform with DED, milling the surface to a finish while it is still accessible, and then depositing again, it combines AM's geometric freedom with the surface and dimensional precision of CNC. The ability to perform deposition repair on an existing part plus finishing in one continuous process is another reason industrial adoption is growing.

## 4.5 How to Select a Process

None of the methods above is "universal." Selection is an **evaluation of trade-offs against the application requirements**. The main comparison axes are summarized below.

Process | Accuracy / surface | Speed | Main materials | Strength | Best-suited use  
---|---|---|---|---|---  
Material Jetting (MJ) | Very high | Medium | Photocurable resin (multi-material, full-color) | Low to medium | Appearance models, medical anatomical models  
Binder Jetting (BJ) | Medium | High | Metals, ceramics, sand, gypsum | Medium (after sintering) | Sand molds, metal mass production, full-color figures  
DED / LENS | Low (needs post-machining) | Very high (deposition) | Metal (powder / wire) | High | Repair, large parts, graded materials  
Sheet Lamination (LOM / UAM) | Medium | High | Paper, metal foil | Low to medium | Visual models, embedded sensors  
  
**⚠️ "Higher accuracy" does not mean "better"**

A common mistake in selection is deciding on a method by a single metric (e.g., accuracy) alone. Making a single sand mold does not need MJ's ultra-high precision, and BJ cannot be used for repair. The practical principle is to **choose the minimally sufficient method that meets the requirement, judged by total cost (material + post-processing + yield)**. Code Example 3 quantifies this idea as weighted scoring.

## 4.6 Emerging Trends

### 4.6.1 Bioprinting (overview)

Bioprinting **dispenses a "bio-ink" (living cells plus a hydrogel carrier) to construct tissue and organ models**. To keep cells viable, low-pressure, low-viscosity conditions that limit shear stress during jetting are required, so the droplet-physics knowledge covered in this chapter applies directly. At present the focus is on **tissue chips for drug screening and small pieces of skin or cartilage** ; transplantable organs are still at the research stage. It is important to avoid overhyping and to assess maturity honestly.

### 4.6.2 4D Printing (overview)

4D printing designs parts so that **their shape changes after the build in response to stimuli such as temperature, humidity, or light**. The "fourth dimension" is time, that is, **the shape changing over time**. Shape-memory polymers or materials that swell with moisture are placed with directionality to create structures that self-fold from flat to three-dimensional. Deployable antennas, self-assembling parts, and soft robots are candidate applications, but here too practical use remains limited to a few cases.

## Code Examples

Let us confirm the key points of this chapter with runnable Python code. All outputs below are actual results executed with `python3` (using NumPy).

### Code Example 1: Droplet printability map (Oh / Z / We / Re)

For representative jetting fluids, we compute the dimensionless numbers and judge whether they fall inside the printable window (1 < Z < 10 and We > 4).
    
    
    import numpy as np
    
    # Material Jetting droplet printability: Ohnesorge / Reynolds / Weber / Z number
    # Z = 1/Oh ; printable window commonly cited as 1 < Z < 10 (some report up to 14)
    # Oh = mu / sqrt(rho * sigma * D)
    
    def dimensionless(rho, mu, sigma, D, v):
        Oh = mu / np.sqrt(rho * sigma * D)
        Z = 1.0 / Oh
        Re = rho * v * D / mu
        We = rho * v**2 * D / sigma
        return Oh, Z, Re, We
    
    # Representative jetting fluids (SI units)
    # rho [kg/m3], mu [Pa.s], sigma [N/m], D nozzle [m], v drop [m/s]
    fluids = [
        ("UV acrylate resin (PolyJet)", 1100, 0.012, 0.030, 30e-6, 8.0),
        ("Molten wax (support)",         900, 0.020, 0.025, 30e-6, 6.0),
        ("Water-thin binder (BJ)",      1000, 0.001, 0.072, 40e-6, 9.0),
        ("Nanoparticle metal ink",      1500, 0.015, 0.035, 20e-6, 7.0),
        ("Over-viscous resin (fail)",   1150, 0.080, 0.030, 30e-6, 8.0),
    ]
    
    print(f"{'Fluid':32s}{'Oh':>8s}{'Z=1/Oh':>9s}{'Re':>8s}{'We':>8s}  Printable(1<Z<10)")
    print("-"*80)
    for name, rho, mu, sigma, D, v in fluids:
        Oh, Z, Re, We = dimensionless(rho, mu, sigma, D, v)
        ok = "YES" if (1.0 < Z < 10.0 and We > 4.0) else "NO"
        print(f"{name:32s}{Oh:8.3f}{Z:9.2f}{Re:8.1f}{We:8.1f}   {ok}")
    
    print()
    print("Interpretation:")
    print(" - Z < 1  : too viscous, droplet won't form cleanly")
    print(" - Z > 10 : satellite droplets / instability")
    print(" - We < 4 : insufficient energy to eject a droplet")

**Execution result:**
    
    
    Fluid                                 Oh   Z=1/Oh      Re      We  Printable(1<Z<10)
    --------------------------------------------------------------------------------
    UV acrylate resin (PolyJet)        0.381     2.62    22.0    70.4   YES
    Molten wax (support)               0.770     1.30     8.1    38.9   YES
    Water-thin binder (BJ)             0.019    53.67   360.0    45.0   NO
    Nanoparticle metal ink             0.463     2.16    14.0    42.0   YES
    Over-viscous resin (fail)          2.487     0.40     3.5    73.6   NO
    
    Interpretation:
     - Z < 1  : too viscous, droplet won't form cleanly
     - Z > 10 : satellite droplets / instability
     - We < 4 : insufficient energy to eject a droplet

The numbers confirm that an overly viscous resin (Z = 0.40) will not detach into droplets, while a low-viscosity binder like water (Z = 53.7) is prone to satellite droplets. The PolyJet resin and the nanoparticle ink fall inside the window.

### Code Example 2: Sintering shrinkage and dimensional compensation for binder jetting

From relative density we compute linear and volumetric shrinkage, and the green-design scale factor needed to obtain the target dimension.
    
    
    import numpy as np
    
    # Binder Jetting: green part -> sintered part shrinkage from densification.
    # Isotropic linear shrinkage from relative density change:
    #   L_sinter / L_green = (rho_green / rho_sinter)^(1/3)
    # Linear shrinkage (%) = (1 - (rho_g/rho_s)^(1/3)) * 100
    
    def linear_shrinkage(rho_green, rho_sinter):
        ratio = (rho_green / rho_sinter) ** (1.0/3.0)
        lin = (1.0 - ratio) * 100.0
        vol = (1.0 - rho_green / rho_sinter) * 100.0
        return lin, vol
    
    # rho values are RELATIVE density (fraction of theoretical)
    cases = [
        ("316L stainless (metal BJ)", 0.55, 0.98),
        ("Ti-6Al-4V (metal BJ)",      0.50, 0.96),
        ("Alumina ceramic",           0.45, 0.95),
        ("Bronze-infiltrated steel",  0.60, 0.90),
    ]
    
    print(f"{'System':30s}{'rho_green':>10s}{'rho_sint':>9s}{'Lin.shr%':>10s}{'Vol.shr%':>10s}")
    print("-"*70)
    for name, rg, rs in cases:
        lin, vol = linear_shrinkage(rg, rs)
        print(f"{name:30s}{rg:10.2f}{rs:9.2f}{lin:10.2f}{vol:10.2f}")
    
    # Compensation: to hit a 50.00 mm target after sintering, scale the green CAD.
    target = 50.00  # mm final dimension
    rg, rs = 0.55, 0.98
    scale = (rs / rg) ** (1.0/3.0)   # green must be LARGER by this factor
    green_dim = target * scale
    print()
    print(f"Design compensation (316L, rho_g=0.55 -> rho_s=0.98):")
    print(f"  required green scale factor = {scale:.4f}")
    print(f"  to obtain {target:.2f} mm final, model green part at {green_dim:.3f} mm")

**Execution result:**
    
    
    System                         rho_green rho_sint  Lin.shr%  Vol.shr%
    ----------------------------------------------------------------------
    316L stainless (metal BJ)           0.55     0.98     17.51     43.88
    Ti-6Al-4V (metal BJ)                0.50     0.96     19.54     47.92
    Alumina ceramic                     0.45     0.95     22.05     52.63
    Bronze-infiltrated steel            0.60     0.90     12.64     33.33
    
    Design compensation (316L, rho_g=0.55 -> rho_s=0.98):
      required green scale factor = 1.2123
      to obtain 50.00 mm final, model green part at 60.617 mm

For 316L stainless steel, linear shrinkage reaches about 17.5% and volumetric shrinkage about 44%. To obtain a final dimension of 50.00 mm, the green part must be designed at 60.6 mm, showing how essential shrinkage compensation is.

### Code Example 3: Weighted scoring of AM processes

We assign weights to six evaluation axes (accuracy, surface, speed, material range, cost efficiency, strength) and compare processes quantitatively per use case. We also show that changing the weights changes the recommendation (sensitivity).
    
    
    import numpy as np
    
    # AM process selection by weighted scoring.
    # Criteria scored 1-5 (5 = best for that criterion).
    criteria = ["accuracy", "surface", "speed", "material_range", "cost_eff", "strength"]
    weights  = np.array([0.25, 0.15, 0.15, 0.15, 0.15, 0.15])  # sums to 1.0
    
    # rows = processes, cols = criteria (expert-assigned 1-5)
    processes = {
        "Material Jetting (MJ)":   [5, 5, 3, 2, 2, 2],
        "Binder Jetting (BJ)":     [3, 3, 5, 4, 4, 3],
        "DED / LENS":              [2, 1, 4, 4, 3, 5],
        "Sheet Lamination (LOM)":  [2, 2, 4, 2, 5, 2],
        "PBF (SLM/SLS)":           [4, 3, 2, 4, 2, 5],
        "Material Extrusion (FDM)":[2, 2, 3, 3, 5, 3],
    }
    
    print(f"weights: {dict(zip(criteria, weights))}")
    print()
    print(f"{'Process':28s}{'Score':>7s}   Ranked criteria contribution")
    print("-"*70)
    results = []
    for name, sc in processes.items():
        sc = np.array(sc, dtype=float)
        total = float(np.dot(weights, sc))
        results.append((name, total))
    for name, total in sorted(results, key=lambda x: -x[1]):
        bar = "#" * int(round(total*6))
        print(f"{name:28s}{total:7.3f}   {bar}")
    
    best = max(results, key=lambda x: x[1])
    print()
    print(f"Recommended (accuracy-weighted use case): {best[0]}  (score {best[1]:.3f})")
    
    # Re-run with a "cheap large metal part" weighting to show sensitivity
    w2 = np.array([0.05, 0.05, 0.25, 0.15, 0.30, 0.20])
    print()
    print("Re-weighted for 'low-cost large metal part' (cost & speed heavy):")
    r2 = [(n, float(np.dot(w2, np.array(s, dtype=float)))) for n, s in processes.items()]
    for name, total in sorted(r2, key=lambda x: -x[1])[:3]:
        print(f"  {name:28s}{total:7.3f}")

**Execution result:**
    
    
    weights: {'accuracy': np.float64(0.25), 'surface': np.float64(0.15), 'speed': np.float64(0.15), 'material_range': np.float64(0.15), 'cost_eff': np.float64(0.15), 'strength': np.float64(0.15)}
    
    Process                       Score   Ranked criteria contribution
    ----------------------------------------------------------------------
    Binder Jetting (BJ)           3.600   ######################
    PBF (SLM/SLS)                 3.400   ####################
    Material Jetting (MJ)         3.350   ####################
    DED / LENS                    3.050   ##################
    Material Extrusion (FDM)      2.900   #################
    Sheet Lamination (LOM)        2.750   ################
    
    Recommended (accuracy-weighted use case): Binder Jetting (BJ)  (score 3.600)
    
    Re-weighted for 'low-cost large metal part' (cost & speed heavy):
      Binder Jetting (BJ)           3.950
      DED / LENS                    3.650
      Material Extrusion (FDM)      3.500

Under accuracy-weighted weights, binder jetting, PBF, and material jetting are close; but reweighting for a "low-cost, large metal part" by emphasizing cost and speed swaps the ranking. This sensitivity analysis plainly shows that **selection changes with the weights (that is, the requirements).**

## Exercises

These exercises check your understanding. Think it through yourself before opening the answers.

Exercise 1 (Basics): Matching processes

For each use below, choose the most suitable AM method from MJ / BJ / DED / SL, and add a one-line reason.  
(a) Repairing a worn turbine blade (b) A sand mold for an engine block (c) A medical anatomical model with distinct hard and soft regions (d) A structure with sensors embedded inside metal foil

Show answer

(a) **DED** : deposition repair onto an existing part is the only practical option.  
(b) **BJ** : forms sand molds fast and large with no sintering.  
(c) **MJ** : can build multiple hardnesses and clear material in a single build.  
(d) **SL (UAM)** : its low-temperature solid-state joining allows sensors to be embedded inside.

Exercise 2 (Calculation): Ohnesorge number and Z number

For a resin with density ρ = 1100 kg/m³, viscosity μ = 0.010 Pa·s, surface tension σ = 0.030 N/m, and nozzle diameter D = 30 μm, find the Ohnesorge number and Z number, and judge whether it falls in the printable window (1 < Z < 10).

Show answer

Oh = μ / √(ρ σ D) = 0.010 / √(1100 × 0.030 × 30e-6) = 0.010 / √(9.9e-4) = 0.010 / 0.03146 ≈ **0.318**.  
Z = 1/Oh ≈ **3.15**. Since 1 < 3.15 < 10, it is **inside the printable window**. You can check this by passing the same values to the `dimensionless` function in Code Example 1.

Exercise 3 (Calculation): Dimensional compensation for sintering shrinkage

For a material with green relative density 0.52 and sintered relative density 0.97, you want a final dimension of 40.0 mm. At what size should the green part be designed? Also find the linear shrinkage.

Show answer

Scale factor = (0.97/0.52)^(1/3) = (1.865)^(1/3) ≈ **1.231**. Green dimension = 40.0 × 1.231 ≈ **49.2 mm**.  
Linear shrinkage = (1 − (0.52/0.97)^(1/3)) × 100 = (1 − 0.812) × 100 ≈ **18.8%**.

Exercise 4 (Discussion): Satellite droplets

You warmed a resin to lower its viscosity, and the Z number rose to 12. What print-quality problem would you expect, and how could you address it?

Show answer

When Z exceeds the upper bound of the window (about 10), the droplet trails a tail and breaks up, making **satellite droplets** more likely. This leads to landing-position errors and mist contamination. Remedies include: (1) heat less to raise the viscosity slightly and bring Z back within the window, (2) tune the drive waveform for cleaner tail break-off, and (3) optimize the droplet velocity (We). However, the final check requires direct observation with a drop watcher.

Exercise 5 (Application): Designing selection weights

For the requirement "produce high-precision full-color dental models in low-volume, high-mix," how would you set the weights for the six axes in Code Example 3 (accuracy, surface, speed, material range, cost efficiency, strength), and which method do you expect to be chosen?

Show answer

Weight accuracy, surface, and material range (full color) heavily, and speed, strength, and cost lightly (e.g., accuracy 0.30 / surface 0.25 / material 0.20 / speed 0.10 / cost 0.05 / strength 0.10). Under these weights, **material jetting (MJ)** , which combines full color and high precision, is expected to rank at the top. Rewriting `weights` in Code Example 3 lets you feel how weight design drives the conclusion.

## Summary

In this chapter we learned the remaining major AM processes, following material extrusion, vat photopolymerization, and powder bed fusion, along with how to select among them. The key points are:

  * **Material jetting (MJ)** : builds by printing droplets. Multi-material and full-color are its greatest strengths. Printability can be estimated from Oh / Z / We, but the final judgment rests on direct observation.
  * **Binder jetting (BJ)** : forms a green part fast by gluing powder, then densifies it by sintering or infiltration. Dimensional compensation for sintering shrinkage is essential.
  * **DED / LENS** : a method for depositing large and fast and repairing broken parts. Precision is coarse, and use with machining is a prerequisite.
  * **Sheet lamination and hybrid manufacturing** : unique value in large builds, embedded sensors, and the integration of additive and subtractive processing.
  * **Selection** : choose not by a single metric but by total cost and trade-offs against the requirement. Weighted scoring is an effective decision aid.
  * **Emerging technologies** : bioprinting and 4D printing are promising, but their maturity must be assessed honestly, without overstatement.

**✅ Looking ahead**

With this, the overall picture of the major AM processes is complete. In the next chapter, we integrate the knowledge so far and take on simulation and analysis of 3D printing in Python.

## Next Steps

In Chapter 4, we took a cross-cutting look at material jetting, binder jetting, directed energy deposition, sheet lamination, and hybrid manufacturing, and surveyed a quantitative approach to process selection along with emerging trends. In the next chapter, Chapter 5, we take on simulation and practical analysis of 3D printing using Python.

[← Back to Chapter 3](<./chapter-3.html>) [Proceed to Chapter 5 →](<./chapter-5.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2021). _Additive Manufacturing Technologies_ (3rd ed.). Springer. - A standard textbook covering all processes, including material jetting, binder jetting, DED, and sheet lamination
  2. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. - The international standard for AM process classification and vocabulary
  3. Derby, B. (2010). "Inkjet Printing of Functional and Structural Materials: Fluid Property Requirements, Feature Stability, and Resolution." _Annual Review of Materials Research_ , 40, 395-414. - The theoretical basis of droplet printability and the Ohnesorge / Z numbers
  4. Ziaee, M., & Crane, N.B. (2019). "Binder Jetting: A Review of Process, Materials, and Methods." _Additive Manufacturing_ , 28, 781-801. - A comprehensive review of binder jetting process, materials, and sintering
  5. Dass, A., & Moridi, A. (2019). "State of the Art in Directed Energy Deposition: From Additive Manufacturing to Materials Design." _Coatings_ , 9(7), 418. - A survey of DED principles, repair, and functionally graded materials
  6. Murphy, S.V., & Atala, A. (2014). "3D Bioprinting of Tissues and Organs." _Nature Biotechnology_ , 32, 773-785. - A representative review of bioprinting principles and challenges
  7. Tibbits, S. (2014). "4D Printing: Multi-Material Shape Change." _Architectural Design_ , 84(1), 116-121. - The foundational paper introducing the concept of 4D printing

## Tools and Libraries Used

  * **NumPy** (v1.24+): numerical computing library - <https://numpy.org/>
  * **Matplotlib** (v3.7+): data visualization library - <https://matplotlib.org/>
  * **Python** (v3.10+): the runtime for this chapter's code examples - <https://www.python.org/>

### Disclaimer

  * This content is provided for educational, research, and informational purposes only, and does not constitute professional advice (legal, accounting, technical assurance, or otherwise).
  * This content and any accompanying code examples are provided "AS IS," without warranty of any kind, express or implied, including without limitation warranties of merchantability, fitness for a particular purpose, non-infringement, or accuracy/completeness of operation or safety.
  * The creator and Tohoku University assume no responsibility for the content, availability, or safety of external links or third-party data, tools, or libraries.
  * To the maximum extent permitted by applicable law, the creator and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content of this material may be changed, updated, or discontinued without notice.
  * The copyright and license of this content follow the terms specified (e.g., CC BY 4.0). Such licenses typically include a no-warranty clause.
