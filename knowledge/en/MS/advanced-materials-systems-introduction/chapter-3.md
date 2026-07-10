---
title: "Chapter 3: Nanomaterials"
chapter_title: "Chapter 3: Nanomaterials"
subtitle: Carbon Nanotubes, Graphene, and Quantum Dots - Design Principles for High Performance
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 3
exercises: 6
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/advanced-materials-systems-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI Terakoya Top](<../../index.html>)›[Materials Science](<../../MS/index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 3

## Learning Objectives

Upon completing this chapter, you will be able to explain:

### Fundamental Understanding

  * How the size effects and the increased surface-to-volume ratio characteristic of nanomaterials (Nanomaterials) affect their properties
  * The chirality (chirality) of carbon nanotubes (Carbon Nanotube, CNT) and the rule that determines metallic vs. semiconducting behavior
  * The band structure of graphene (Graphene) and the distinctive electronic properties arising from the Dirac point (Dirac point)
  * The quantum confinement effect in quantum dots (Quantum Dot, QD) and the size dependence of the band gap

### Practical Skills

  * Classify a CNT's diameter, chiral angle, and electronic character from its (n,m) indices in Python
  * Compute the emission wavelength of a quantum dot from its radius using the Brus equation
  * Evaluate the surface-atom fraction of a nanoparticle as a function of particle size

### Applied Ability

  * Select the appropriate nanomaterial (0D, 1D, or 2D) from application requirements
  * Design the emission color of quantum dots for display and bio-imaging applications
  * Explain the catalytic activity and reactivity of nanomaterials from the standpoint of surface effects

## 3.1 Fundamentals of Nanomaterials - Size Effects and Dimensionality

### 3.1.1 What Are Nanomaterials?

Nanomaterials (Nanomaterials) are **materials in which at least one dimension lies roughly in the 1-100 nm range**. In this size regime, qualitatively different properties emerge compared with bulk materials. The origins are mainly the following two:

  * **Quantum size effect (Quantum Size Effect)** : The motion of electrons is confined to atomic-scale spaces, discretizing the energy levels
  * **Surface effect (Surface Effect)** : The surface-to-volume ratio rises sharply, and the fraction of surface atoms among all atoms becomes dominant

**💡 Why properties change at the "nano" scale**

In a gold nanoparticle 10 nm in diameter, about 16% of all atoms are exposed at the surface (we compute this in Section 3.5). Whereas bulk gold (with a surface-atom fraction close to 0%) is chemically inert, gold nanoparticles catalyze CO oxidation at room temperature. Surface atoms are coordinatively unsaturated and highly reactive, and this difference gives rise to the catalytic, optical, and electronic properties of nanomaterials.

### 3.1.2 Classification by Dimensionality (0D, 1D, 2D)

Nanomaterials are classified by the "number of directions confined to the nanoscale." For each confined direction the electrons lose a degree of freedom, changing the shape of the density of states (Density of States, DOS).
    
    
    flowchart LR
        A[Dimensional classification] --> B[0D  
    Quantum dots]
        A --> C[1D  
    Nanotubes / nanowires]
        A --> D[2D  
    Graphene / nanosheets]
    
        B --> B1[Confined in 3 directions  
    discrete levels]
        C --> C1[Confined in 2 directions  
    conduction along 1]
        D --> D1[Confined in 1 direction  
    conduction in-plane]
    
        style A fill:#f093fb
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#fff3e0
            

Dimension | Confined directions | Representative examples | Density-of-states feature  
---|---|---|---  
0D (zero-dimensional) | All 3 directions | Quantum dots, fullerenes | Discrete, delta-function-like  
1D (one-dimensional) | 2 directions | Carbon nanotubes, nanowires | 1/√E divergence (van Hove singularities)  
2D (two-dimensional) | 1 direction | Graphene, transition-metal dichalcogenides | Step-function-like (constant)  
3D (bulk) | None | Crystalline / polycrystalline solids | Proportional to √E  
  
### 3.1.3 The Increase in Surface-to-Volume Ratio

For a sphere of radius r, the ratio of surface area S = 4πr² to volume V = (4/3)πr³ is given by:

S / V = 3 / r 

That is, **the specific surface area increases in inverse proportion to the radius**. Reducing the particle size by a factor of 10 increases the surface area per unit volume tenfold, dramatically raising the fraction of atoms exposed at the surface. This geometric fact underlies the high catalytic activity, solubility, and reactivity of nanomaterials. Quantitative calculations are covered in the Python practice of Section 3.5.

## 3.2 Carbon Nanotubes - One-Dimensional Carbon Materials

### 3.2.1 Structure and Chirality

A carbon nanotube (Carbon Nanotube, CNT) is **a one-dimensional structure formed by rolling a graphene sheet into a cylinder**. The direction of rolling is expressed by the chiral vector (chiral vector), specified as an integer pair (n, m) using the primitive translation vectors a₁, a₂ of graphene:

C_h = n·a₁ + m·a₂ (|a₁| = |a₂| = a = 0.246 nm) 

This (n, m) determines all of the CNT's geometry and electronic properties. The rolling direction gives three types:

  * **Armchair (armchair)** : n = m; chiral angle 30°
  * **Zigzag (zigzag)** : m = 0; chiral angle 0°
  * **Chiral (chiral)** : all others; wound helically

The diameter d and chiral angle θ follow analytically from (n, m):

d = (a / π)·√(n² + nm + m²) , θ = arctan[ √3·m / (2n + m) ] 

### 3.2.2 The Metallic / Semiconducting Rule

The most important property of CNTs is that **whether a tube is metallic or semiconducting is fixed by its geometry alone**. The rule can be stated as:

(n − m) is a multiple of 3 ⇒ metallic / otherwise ⇒ semiconducting 

This rule follows from the band structure of graphene (Section 3.3). Graphene is a semimetal whose valence and conduction bands touch at the K points; in a CNT the circumferential wavenumber is quantized, so metallicity depends on whether the allowed wavenumber lines pass through a K point. Statistically, about 1/3 of randomly synthesized CNTs are metallic and 2/3 are semiconducting (we verify this in the Python practice of Section 3.5).

**💡 The band gap of semiconducting CNTs**

The band gap E_g of a semiconducting CNT is roughly inversely proportional to the diameter d, on the order of E_g ≈ 0.8 eV / d[nm]. A CNT 1 nm in diameter has about 0.8 eV, close to silicon (1.1 eV), making it promising as a transistor channel material. The ability to design the band gap by choosing the diameter is one of the attractions of CNTs.

### 3.2.3 Mechanical and Electrical Properties

CNTs exhibit outstanding properties owing to their strong sp² carbon-carbon bonds:

  * **Young's modulus** : about 1 TPa (roughly 5× that of steel)
  * **Tensile strength** : 50-100 GPa (about 50× steel, at roughly 1/6 the density)
  * **Current-density tolerance** : up to about 10⁹ A/cm² (about 1000× copper)
  * **Thermal conductivity** : about 3000-3500 W/(m·K) along the axis (about 8× copper)

### 3.2.4 Synthesis (CVD)

The dominant industrial synthesis method is **chemical vapor deposition (Chemical Vapor Deposition, CVD)**. Hydrocarbon gases (methane, ethylene, etc.) are supplied over nanoparticles of a catalyst metal (Fe, Co, Ni) and decomposed and precipitated at 600-1000°C to grow CNTs.

  * **Diameter control via catalyst size** : the diameter of the catalyst nanoparticle largely sets the CNT diameter
  * **Choice of growth temperature and feed gas** : affects whether single-walled (SWCNT) or multi-walled (MWCNT) tubes form
  * **Substrate growth (vertical alignment)** : forest-like aligned CNTs can be obtained over large areas

**⚠️ The challenge of chirality control**

CVD can control the diameter to some degree, but making the chirality ((n,m)) uniform remains difficult. Obtaining only semiconducting tubes at high purity requires post-synthesis separation and purification (density-gradient ultracentrifugation, gel chromatography, etc.). Direct synthesis of single-chirality CNTs is still an active research topic.

## 3.3 Graphene - A Two-Dimensional Carbon Material

### 3.3.1 Structure and the Basics of the Band Structure

Graphene (Graphene) is **a two-dimensional material one atom thick in which carbon atoms form a honeycomb lattice (honeycomb lattice) through sp² bonding**. In 2004, Geim and Novoselov isolated it by mechanical exfoliation using adhesive tape, work that led to the 2010 Nobel Prize in Physics.

The honeycomb lattice has two carbon atoms per unit cell (the A and B sublattices). Solving it in the tight-binding approximation (tight-binding) shows that the valence and conduction bands touch at the six K points (and K' points) of the Brillouin zone.

### 3.3.2 The Dirac Point and Linear Dispersion

The decisive feature of graphene is that near the K points the energy disperses **linearly** with wavenumber:

E(k) ≈ ± ħ·v_F·|k| (v_F ≈ 1.0 × 10⁶ m/s) 

This touching point is called the **Dirac point (Dirac point)**. Whereas ordinary semiconductors have parabolic dispersion E ∝ k², graphene's linear dispersion makes electrons behave as if they were massless relativistic particles (Dirac fermions). The Fermi velocity v_F is about 1/300 of the speed of light.

**💡 Graphene is a "zero-gap semiconductor"**

Because graphene's valence and conduction bands touch at a single point, it is a semimetal (semimetal) with a band gap that is exactly zero. This yields high mobility, but is a weakness for transistors since no off state can be created. Research is underway to open a gap by forming nanoribbons, bilayers, or through interactions with the substrate.

### 3.3.3 Outstanding Properties

  * **Electron mobility** : about 200,000 cm²/(V·s) at room temperature (over 100× silicon)
  * **Mechanical strength** : tensile strength about 130 GPa, Young's modulus about 1 TPa (among the strongest known materials)
  * **Thermal conductivity** : about 5000 W/(m·K) (over 10× copper)
  * **Optical transmittance** : a single layer transmits about 97.7% of visible light (promising for transparent conductive films)
  * **Specific surface area** : theoretical value about 2630 m²/g (both faces exposed)

### 3.3.4 Production Methods

Method | Quality | Area / quantity | Main applications  
---|---|---|---  
Mechanical exfoliation (Scotch-tape method) | Highest quality, minimal defects | Tiny flakes only | Basic research, property measurement  
CVD (growth on Cu foil) | High quality, large area | Roll-to-roll capable | Transparent electrodes, flexible devices  
SiC thermal decomposition (epitaxial growth) | High quality | Wafer scale | High-frequency electronic devices  
Reduced graphene oxide (rGO) | More defects, low cost | Mass production, solution process | Conductive inks, composites, electrodes  
  
## 3.4 Quantum Dots - Zero-Dimensional Semiconductor Nanocrystals

### 3.4.1 The Quantum Confinement Effect

A quantum dot (Quantum Dot, QD) is **a semiconductor nanocrystal roughly 2-10 nm in diameter**. Because the motion of electrons and holes is confined in all three directions to the crystal size, the energy levels become discrete and the band gap varies with particle size. This phenomenon is called the **quantum confinement effect (Quantum Confinement Effect)**.

Confinement becomes pronounced when the crystal radius falls to about the **exciton Bohr radius (exciton Bohr radius)** or below. In CdSe the exciton Bohr radius is about 5.6 nm, and once the particle size drops below this the band gap widens markedly.

### 3.4.2 The "Particle in a Box" Model and the Brus Equation

The most basic picture is the quantum-mechanical "particle in a box (particle in a box)." The ground-state energy of a particle confined in a three-dimensional infinite well of side L increases as E ∝ 1/L². In other words, **the smaller the box, the higher the ground level** , so the smaller the particle, the wider the band gap.

The Brus equation (Brus equation) quantifies this intuition for semiconductor nanocrystals. The effective band gap of a spherical nanocrystal of radius R is:

E_g(R) = E_g(bulk) + ( h² / 8R² )·( 1/m_e + 1/m_h ) − 1.8·e² / (4πε₀ε_r R) 

The right-hand side has three terms:

  * **First term E_g(bulk)** : the band gap of the bulk semiconductor (a size-independent reference)
  * **Second term (quantum confinement term)** : scaling as 1/R², it widens the gap as the particle shrinks. m_e and m_h are the effective masses of the electron and hole
  * **Third term (Coulomb term)** : stabilization from the electrostatic attraction between electron and hole; scaling as 1/R, it slightly narrows the gap

As the particle shrinks, the second term (1/R²) grows faster than the third term (1/R), so the net band gap widens and the emission wavelength shifts toward shorter wavelengths (toward blue). We compute this relationship in Python in Section 3.5.

### 3.4.3 Optical Properties and Applications

Because the band gap is set by particle size, quantum dots have the striking feature that **the emission color can be tuned continuously simply by changing the particle size at fixed composition**. In CdSe quantum dots, changing the size from about 2 nm to 6 nm shifts the emission color from blue to red.

  * **Displays (QD-LED, quantum-dot TVs)** : high color purity and wide color gamut from a narrow emission spectrum (full width at half maximum 20-40 nm). Quantum dots convert a blue backlight into highly pure green and red
  * **Bio-imaging** : more resistant to photobleaching than organic fluorophores, and different sizes emit different colors, allowing multiple targets to be labeled simultaneously (multicolor fluorescence labeling)
  * **Solar cells and photodetectors** : the absorption edge can be designed via particle size, and films can be deposited by solution processing

**⚠️ The toxicity of cadmium-based quantum dots**

Many high-performance quantum dots contain cadmium, as in CdSe and CdTe, raising issues of toxicity and environmental regulation (RoHS, etc.). Cadmium-free alternatives such as InP-based dots, carbon dots, and perovskite quantum dots are being studied, but they can fall short of Cd-based dots in emission efficiency and stability, so materials development continues.

## 3.5 Python Practice: Computing Nanomaterial Properties

In this section we actually compute the three themes covered above. All code is self-contained using only NumPy, with execution results shown alongside.

### 3.5.1 CNT Chirality Classification and Diameter Calculation

From the (n, m) indices we determine the diameter, chiral angle, geometric type, and metallic/semiconducting character. We confirm the rule (n − m) mod 3 = 0 for metallic tubes, and that enumerating all 0 ≤ m ≤ n ≤ 20 gives a metallic fraction of about 1/3.
    
    
    # ===================================
    # Example 1: CNT chirality classification
    # ===================================
    
    import numpy as np
    
    a = 0.246  # graphene lattice constant [nm]
    
    def cnt_diameter(n, m):
        """Compute the CNT diameter from chiral indices (n, m) [nm]"""
        return a * np.sqrt(n**2 + n*m + m**2) / np.pi
    
    def chiral_angle(n, m):
        """Compute the chiral angle [degree]"""
        return np.degrees(np.arctan(np.sqrt(3)*m / (2*n + m)))
    
    def cnt_type(n, m):
        """Determine metallic vs. semiconducting character"""
        if (n - m) % 3 == 0:
            return "metallic"
        return "semiconducting"
    
    def cnt_class(n, m):
        """Geometric classification (armchair / zigzag / chiral)"""
        if m == 0:
            return "zigzag"
        if n == m:
            return "armchair"
        return "chiral"
    
    # Classify representative (n, m)
    examples = [(5,5), (9,0), (10,0), (7,3), (6,4), (10,10), (8,4), (11,7)]
    print("(n, m)   type            geometry   d [nm]   theta [deg]")
    print("-" * 58)
    for n, m in examples:
        print(f"({n:2d},{m:2d})  {cnt_type(n,m):14s}  {cnt_class(n,m):9s}  "
              f"{cnt_diameter(n,m):5.3f}    {chiral_angle(n,m):5.2f}")
    
    # Enumerate all (n, m) with 0 <= m <= n <= 20 and tally the metallic fraction
    total = 0
    metal = 0
    for n in range(1, 21):
        for m in range(0, n + 1):
            total += 1
            if (n - m) % 3 == 0:
                metal += 1
    print()
    print(f"Enumerated (n,m) with 0<=m<=n<=20: {total}")
    print(f"Metallic count: {metal}  ({100*metal/total:.1f}%)")
    
    # Execution result:
    # (n, m)   type            geometry   d [nm]   theta [deg]
    # ----------------------------------------------------------
    # ( 5, 5)  metallic        armchair   0.678    30.00
    # ( 9, 0)  metallic        zigzag     0.705     0.00
    # (10, 0)  semiconducting  zigzag     0.783     0.00
    # ( 7, 3)  semiconducting  chiral     0.696    17.00
    # ( 6, 4)  semiconducting  chiral     0.683    23.41
    # (10,10)  metallic        armchair   1.356    30.00
    # ( 8, 4)  semiconducting  chiral     0.829    19.11
    # (11, 7)  semiconducting  chiral     1.231    22.69
    #
    # Enumerated (n,m) with 0<=m<=n<=20: 230
    # Metallic count: 83  (36.1%)
    

The armchair tubes (5,5) and (10,10) are always metallic (n − m = 0), and the diameter of (10,10) is twice that of (5,5). Over the full enumeration, 83 of 230 tubes are metallic (36.1%), in good agreement with the theoretical "about 1/3 metallic."

### 3.5.2 Quantum-Dot Emission Wavelength via the Brus Equation

For CdSe quantum dots, we vary the radius from 1.5 to 4.0 nm and compute the band gap and emission wavelength. We confirm that smaller particles have a wider gap and emit at shorter (bluer) wavelengths.
    
    
    # ===================================
    # Example 2: Brus equation (quantum dots)
    # ===================================
    
    import numpy as np
    
    # Physical constants (SI units)
    h    = 6.62607015e-34   # Planck constant [J s]
    me0  = 9.1093837015e-31 # electron rest mass [kg]
    e    = 1.602176634e-19  # elementary charge [C]
    eps0 = 8.8541878128e-12 # vacuum permittivity [F/m]
    c    = 2.99792458e8     # speed of light [m/s]
    
    # CdSe parameters
    Eg_bulk = 1.74          # bulk band gap [eV]
    m_e = 0.13 * me0        # electron effective mass
    m_h = 0.45 * me0        # hole effective mass
    eps_r = 10.6            # relative permittivity
    
    def brus_gap_eV(R_nm):
        """Compute the effective band gap with the Brus equation [eV]"""
        R = R_nm * 1e-9
        confinement = (h**2 / (8 * R**2)) * (1/m_e + 1/m_h) / e   # quantum confinement term [eV]
        coulomb = 1.8 * e / (4 * np.pi * eps0 * eps_r * R)        # Coulomb term [eV]
        return Eg_bulk + confinement - coulomb
    
    def emission_nm(Eg_eV):
        """Compute the emission wavelength from the band gap [nm]"""
        return h * c / (Eg_eV * e) * 1e9
    
    print("CdSe quantum dot: Brus equation")
    print(f"{'radius [nm]':>11} {'diameter [nm]':>13} {'E_gap [eV]':>11} {'lambda [nm]':>12}")
    print("-" * 50)
    for R in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        Eg = brus_gap_eV(R)
        lam = emission_nm(Eg)
        print(f"{R:11.1f} {2*R:13.1f} {Eg:11.3f} {lam:12.1f}")
    
    print()
    print(f"Bulk CdSe gap {Eg_bulk} eV -> lambda {emission_nm(Eg_bulk):.1f} nm")
    
    # Execution result:
    # CdSe quantum dot: Brus equation
    # radius [nm] diameter [nm]  E_gap [eV]  lambda [nm]
    # --------------------------------------------------
    #         1.5           3.0       3.234        383.4
    #         2.0           4.0       2.550        486.3
    #         2.5           5.0       2.239        553.8
    #         3.0           6.0       2.073        598.2
    #         3.5           7.0       1.974        627.9
    #         4.0           8.0       1.912        648.5
    #
    # Bulk CdSe gap 1.74 eV -> lambda 712.6 nm
    

At a radius of 1.5 nm the emission wavelength is 383 nm (violet), and at 4.0 nm it is 649 nm (red), so particle size alone can cover the entire visible range. Compared with bulk CdSe (712 nm, near-infrared), nanocrystallization greatly widens the band gap. This monotonic size-emission relationship is the basis of color design in displays.

**💡 The range of validity of the Brus equation**

Because the Brus equation is based on the effective-mass approximation, it tends to overestimate the band gap in the strong-confinement regime where the radius drops below 1 nm. This is why the calculation above starts at a radius of 1.5 nm. Smaller particles require tight-binding or first-principles methods.

### 3.5.3 Surface-Atom Fraction of a Nanoparticle

For a spherical nanoparticle, we treat atoms within a shell one atomic layer thick from the surface as "surface atoms" and compute their fraction as a function of particle size. We confirm that the surface-atom fraction rises sharply as the particle shrinks.
    
    
    # ===================================
    # Example 3: Surface-atom fraction
    # ===================================
    
    import numpy as np
    
    # Shell model: atoms within one atomic diameter of the surface are "surface" atoms.
    # F_surface = 1 - ((R - t)/R)^3,  t = surface-shell thickness [nm]
    
    r_atom = 0.144      # metallic radius of gold [nm]
    t = 2 * r_atom      # surface-shell thickness (one atomic diameter) [nm]
    
    def surface_fraction(D_nm):
        """Surface-atom fraction of a spherical nanoparticle of diameter D"""
        R = D_nm / 2.0
        if R <= t:
            return 1.0
        return 1.0 - ((R - t) / R)**3
    
    def n_total(D_nm):
        """Approximate total atom count (assuming fcc with packing fraction 0.74)"""
        R = D_nm / 2.0
        return 0.74 * (R / r_atom)**3
    
    print("Gold nanoparticle: surface-atom fraction (r_atom = 0.144 nm)")
    print(f"{'diameter [nm]':>13} {'~N_total':>10} {'surface fraction':>17}")
    print("-" * 44)
    for D in [1, 2, 5, 10, 20, 50, 100]:
        print(f"{D:13d} {n_total(D):10.0f} {surface_fraction(D)*100:16.1f}%")
    
    # Execution result:
    # Gold nanoparticle: surface-atom fraction (r_atom = 0.144 nm)
    # diameter [nm]   ~N_total  surface fraction
    # --------------------------------------------
    #             1         31             92.4%
    #             2        248             63.9%
    #             5       3872             30.7%
    #            10      30978             16.3%
    #            20     247825              8.4%
    #            50    3872258              3.4%
    #           100   30978063              1.7%
    

At a diameter of 1 nm, more than 90% of all atoms are exposed at the surface, whereas at 100 nm only 1.7% are. This sharp rise in the surface-atom fraction underlies size-dependent properties of gold nanoparticles such as catalytic activity, melting-point depression, and plasmonic coloration. The relation S/V = 3/r from Section 3.1.3 is seen to act in the same way at the level of atom counts.

## Checking the Learning Objectives

Upon completing this chapter, you can explain the following:

### Fundamental Understanding

  * ✅ Explain that the properties of nanomaterials arise from size effects and surface effects
  * ✅ Understand the 0D/1D/2D dimensional classification and the differences in density of states
  * ✅ Explain a CNT's chirality (n, m) and the metallic/semiconducting rule
  * ✅ Understand the meaning of graphene's Dirac point and linear dispersion
  * ✅ Explain the quantum confinement effect and the size dependence of the band gap

### Practical Skills

  * ✅ Compute a CNT's diameter, chiral angle, and electronic character from (n, m)
  * ✅ Obtain a quantum dot's emission wavelength from its radius via the Brus equation
  * ✅ Evaluate a nanoparticle's surface-atom fraction as a function of particle size

### Applied Ability

  * ✅ Select the appropriate nanomaterial dimension, composition, and size for an application
  * ✅ Design the emission color of quantum dots for displays and bio-imaging
  * ✅ Explain the catalytic activity and reactivity of nanomaterials from surface effects

## Exercises

### Easy (Basic Check)

Q3.1: Determining CNT Metallicity

Is the carbon nanotube with chiral indices (12, 6) metallic or semiconducting? Answer using the rule.

View answer

**Correct answer: metallic**

**Explanation:**  
The rule is "metallic if (n − m) is a multiple of 3."  
n − m = 12 − 6 = 6 = 3 × 2, which is a multiple of 3. Therefore (12, 6) is judged **metallic**.

Note that even when n = 2m, only the value of n − m matters for the determination.

Q3.2: Dimensional Classification

Classify the following nanomaterials as 0D, 1D, or 2D: (a) graphene, (b) CdSe quantum dot, (c) single-walled carbon nanotube.

View answer

**Correct answer:**

  * (a) Graphene → **2D (two-dimensional)** : confined only in the thickness direction, conducts in-plane
  * (b) CdSe quantum dot → **0D (zero-dimensional)** : confined in all three directions, discrete levels
  * (c) Single-walled carbon nanotube → **1D (one-dimensional)** : confined circumferentially and radially, conducts along the axis

Corresponding to the number of confined directions (3, 1, 2), the number of freely moving directions is 0, 2, 1.

Q3.3: Quantum-Dot Emission Color

Two quantum dots made of the same CdSe are given: A (radius 2.0 nm) and B (radius 3.5 nm). Which emits at the shorter (bluer) wavelength? State the reason.

View answer

**Correct answer: A (radius 2.0 nm)**

**Explanation:**  
The quantum confinement term scales as 1/R², so the smaller the radius, the wider the band gap. A wider band gap means higher-energy emitted photons and thus shorter wavelength.

In the calculation of Section 3.5.2, the emission wavelength was 486 nm at radius 2.0 nm and 628 nm at radius 3.5 nm. Therefore the smaller dot A emits at the shorter wavelength (blue-green) and B at the longer wavelength (red).

### Medium (Application)

Q3.4: Calculating CNT Diameter

Compute the diameter of the armchair (10, 10) CNT, using the lattice constant a = 0.246 nm and the formula d = (a/π)·√(n² + nm + m²).

View answer

**Answer:**

Substitute n = m = 10.  
n² + nm + m² = 100 + 100 + 100 = 300  
√300 ≈ 17.32  
d = (0.246 / π) × 17.32 = 0.0783 × 17.32 ≈ **1.356 nm**

This matches the code output of Section 3.5.1 ((10,10) → 1.356 nm). For reference, (5,5) gives √75 ≈ 8.66 and d ≈ 0.678 nm, exactly half the diameter.

Q3.5: The Meaning of the Surface-Atom Fraction

A gold nanoparticle 10 nm in diameter has a surface-atom fraction of about 16%, and one 2 nm in diameter about 64%. When used as a catalyst, explain from the standpoint of surface effects why smaller particles tend to be advantageous. Also give one drawback of making them too small.

View answer

**Sample answer:**

**Why smaller particles are advantageous for catalysis:**  
Catalytic reactions proceed on the atoms exposed at the surface. The higher the surface-atom fraction, the more atoms of the same mass of gold can participate in the reaction, raising the activity per unit mass. Surface atoms also have low coordination numbers (coordinatively unsaturated) and readily adsorb and activate reactant molecules, so they act as active sites. By the relation S/V = 3/r, the smaller the particle, the larger the specific surface area.

**Drawback of making them too small (any one):**

  * High surface energy causes particles to aggregate and sinter, losing activity (thermally unstable)
  * Melting-point depression becomes large, so the structure cannot be maintained at operating temperature
  * Synthesis and classification become difficult, making size-distribution control hard

### Hard (Advanced)

Q3.6: Designing Quantum Dots for a Display

For a QD display that uses a blue backlight (450 nm), design CdSe quantum dots that emit green (about 530 nm) and red (about 630 nm). Referring to the results of Section 3.5.2, estimate the approximate radius required for each, and also state two advantages of using quantum dots in a display compared with organic phosphors.

View answer

**Answer:**

**Radius estimates (interpolated from the table in 3.5.2):**

Target emission wavelength| Nearby points in table| Estimated radius  
---|---|---  
Green ~530 nm| 2.0 nm→486 nm, 2.5 nm→554 nm| about 2.3-2.4 nm  
Red ~630 nm| 3.5 nm→628 nm| about 3.5 nm  
  
Green is obtained with dots of radius about 2.3 nm and red with radius about 3.5 nm. The scheme is that the green and red dots absorb the blue backlight and re-emit in their respective colors.

**Advantages over organic phosphors (any two):**

  * The emission spectrum has a narrow full width at half maximum (20-40 nm), giving high color purity and a wide color gamut
  * Any emission color can be obtained from a single material just by changing the particle size, simplifying materials design
  * Strong resistance to photobleaching, with excellent long-term stability and luminance retention

**Note:** In practice, cadmium-free versions (e.g., InP-based) and synthesis control to narrow the size distribution are challenges.

## Next Steps

In Chapter 3 we learned the fundamentals of nanomaterials (size effects and dimensionality), the structures and properties of carbon nanotubes, graphene, and quantum dots, and how to compute their properties in Python. In Chapter 4, we address the design principles for integrating these nanomaterials and advanced materials into real devices and systems.

[← Back to Chapter 2](<./chapter-2.html>) [Continue to Chapter 4 →](<./chapter-4.html>)

## References

  1. Dresselhaus, M. S., Dresselhaus, G., & Avouris, P. (2001). _Carbon Nanotubes: Synthesis, Structure, Properties, and Applications_. Springer. pp. 1-38, 111-165. - A comprehensive account of the structure, properties, and synthesis of carbon nanotubes
  2. Geim, A. K., & Novoselov, K. S. (2007). "The rise of graphene." _Nature Materials_ , 6(3), 183-191. - Nobel Prize-winning work on the discovery of graphene and its distinctive electronic properties
  3. Alivisatos, A. P. (1996). "Semiconductor clusters, nanocrystals, and quantum dots." _Science_ , 271(5251), 933-937. - Pioneering work on the electronic structure of quantum dots and the quantum confinement effect
  4. Burda, C., Chen, X., Narayanan, R., & El-Sayed, M. A. (2005). "Chemistry and properties of nanocrystals of different shapes." _Chemical Reviews_ , 105(4), 1025-1102. - A detailed review of shape-controlled synthesis and optical properties of metal nanoparticles
  5. Iijima, S. (1991). "Helical microtubules of graphitic carbon." _Nature_ , 354(6348), 56-58. - The historic paper on the discovery of carbon nanotubes
  6. Brus, L. E. (1984). "Electron-electron and electron-hole interactions in small semiconductor crystallites: The size dependence of the lowest excited electronic state." _Journal of Chemical Physics_ , 80(9), 4403-4409. - The theoretical basis for the size-dependent band gap of quantum dots
  7. ASE Documentation. (2024). _Atomic Simulation Environment_. <https://wiki.fysik.dtu.dk/ase/> \- A Python library for nanostructure simulation

## Tools and Libraries Used

  * **NumPy** (v1.24+): Numerical computing library - <https://numpy.org/>
  * **SciPy** (v1.10+): Scientific computing library - <https://scipy.org/>
  * **Matplotlib** (v3.7+): Data visualization library - <https://matplotlib.org/>
  * **ASE** (v3.22+): Atomic Simulation Environment - <https://wiki.fysik.dtu.dk/ase/>
  * **pymatgen** (v2023+): Materials science computing library - <https://pymatgen.org/>

### Disclaimer

  * This content is provided for educational, research, and informational purposes only and does not constitute professional advice (legal, accounting, technical warranty, or otherwise).
  * This content and any accompanying code examples are provided "AS IS," without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links or third-party data, tools, and libraries referenced herein.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content of this material may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are governed by the stated terms (e.g., CC BY 4.0). Such licenses typically include a no-warranty clause.
