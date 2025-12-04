---
title: "Chapter 2: Fundamentals of Additive Manufacturing"
chapter_title: "Chapter 2: Fundamentals of Additive Manufacturing"
subtitle: AM Technology Principles and Classification - 3D Printing Technology System
reading_time: 35-40 minutes
---

[AI Terakoya Top](<../index.html>)›[Materials Science](<../../index.html>)›[Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>)›Chapter 2

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-2.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain the following:

### Basic Understanding (Level 1)

  * Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard
  * Characteristics of seven AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)
  * Structure of STL file format (triangular mesh, normal vectors, vertex order)
  * History of AM (from 1986 stereolithography to modern systems)

### Practical Skills (Level 2)

  * Ability to read STL files in Python and calculate volume and surface area
  * Ability to validate and repair meshes using numpy-stl and trimesh
  * Understanding basic principles of slicing (layer height, shells, infill)
  * Ability to interpret basic G-code structure (G0/G1/G28/M104, etc.)

### Applied Competence (Level 3)

  * Ability to select optimal AM process according to application requirements
  * Ability to detect and correct mesh problems (non-manifold, flipped normals)
  * Ability to optimize build parameters (layer height, print speed, temperature)
  * Ability to assess STL file quality and determine printability

## 1.1 What is Additive Manufacturing (AM)?

### 1.1.1 Definition of Additive Manufacturing

Additive Manufacturing (AM) is **"a process of fabricating objects by joining materials layer by layer from 3D CAD data," as defined in ISO/ASTM 52900:2021 standard**. In contrast to conventional subtractive manufacturing (cutting/machining), AM adds material only where needed, offering the following innovative characteristics:

  * **Design Freedom** : Enables manufacturing of complex geometries impossible with conventional methods (hollow structures, lattice structures, topology-optimized shapes)
  * **Material Efficiency** : Material waste rate of 5-10% (conventional machining: 30-90% waste) by using material only where needed
  * **On-Demand Manufacturing** : Enables low-volume, high-variety production of customized products without tooling
  * **Integrated Manufacturing** : Produces structures as single pieces that conventionally required assembly of multiple parts, reducing assembly steps

**💡 Industrial Significance**

The AM market is experiencing rapid growth. According to Wohlers Report 2023:

  * Global AM market size: $18.3B (2023) → $83.9B projected (2030, CAGR 23.5%)
  * Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)
  * Key industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)
  * Material share: Polymers (55%), Metals (35%), Ceramics (7%), Others (3%)

### 1.1.2 History and Evolution of AM

Additive manufacturing technology has approximately 40 years of history, reaching the present through the following milestones:
    
    
     flowchart LR A[1986  
    SLA Invention  
    Chuck Hull] --> B[1988  
    SLS Introduction  
    Carl Deckard] B --> C[1992  
    FDM Patent  
    Stratasys] C --> D[2005  
    RepRap  
    Open Source] D --> E[2012  
    Metal AM Adoption  
    EBM/SLM] E --> F[2023  
    Industrial Acceleration  
    Large-scale & High-speed] style A fill:#e3f2fd style B fill:#fff3e0 style C fill:#e8f5e9 style D fill:#f3e5f5 style E fill:#fce4ec style F fill:#fff9c4 

  1. **1986: Invention of Stereolithography (SLA)** \- Dr. Chuck Hull (founder of 3D Systems) invented the first AM technology that cures photopolymer resin layer by layer (US Patent 4,575,330). The term '3D printing' was also coined during this period.
  2. **1988: Introduction of Selective Laser Sintering (SLS)** \- Dr. Carl Deckard (University of Texas) developed technology to sinter powder materials with a laser, opening possibilities for metal and ceramic applications.
  3. **1992: Fused Deposition Modeling (FDM) Patent** \- Stratasys commercialized FDM technology, establishing the foundation for the most widely adopted 3D printing method today.
  4. **2005: RepRap Project** \- Professor Adrian Bowyer announced the open-source 3D printer 'RepRap'. Combined with patent expirations, this accelerated cost reduction and democratization.
  5. **2012 onwards: Industrial Adoption of Metal AM** \- Electron Beam Melting (EBM) and Selective Laser Melting (SLM) became practical in aerospace and medical fields. GE Aviation began mass production of FUEL injection nozzles.
  6. **2023 Present: Era of Large-scale and High-speed** \- New technologies such as binder jetting, continuous fiber composite AM, and multi-material AM are entering industrial implementation stages.

### 1.1.3 Major Application Areas of AM

#### Application 1: Rapid Prototyping

AM's first major application, rapidly manufacturing prototypes for design validation, functional testing, and market evaluation:

  * **Lead Time Reduction** : Conventional prototyping (weeks to months) → AM enables hours to days
  * **Accelerated Design Iteration** : Prototype multiple versions at low cost to optimize design
  * **Improved Communication** : Physical models providing visual and tactile feedback align understanding among stakeholders
  * **Typical Examples** : Automotive design models, consumer electronics enclosure prototypes, presurgical simulation models for medical devices

#### Application 2: Tooling & Fixtures

Application of manufacturing jigs, tools, and molds used in production environments with AM:

  * **Custom Jigs** : Rapidly fabricate assembly and inspection jigs specialized for production lines
  * **Conformal Cooling Molds** : Injection molds with 3D cooling channels following product geometry rather than conventional straight channels (30-70% cooling time reduction)
  * **Lightweight Tools** : Lightweight end effectors using lattice structures to reduce operator burden
  * **Typical Examples** : BMW assembly line jigs (over 100,000 units annually manufactured with AM), TaylorMade golf driver molds

#### Application 3: End-Use Parts

Applications manufacturing end-use parts directly with AM have been rapidly increasing in recent years:

  * **Aerospace Components** : GE Aviation LEAP fuel injection nozzle (conventional 20 parts → AM integrated, 25% weight reduction, over 100,000 units produced annually)
  * **Medical Implants** : Titanium artificial hip joints and dental implants (optimized for patient-specific anatomical shapes, porous structures promoting bone integration)
  * **Custom Products** : Hearing aids (over 10 million units manufactured with AM annually), sports shoe midsoles (Adidas 4D, Carbon DLS technology)
  * **Spare Parts** : On-demand manufacturing of discontinued and rare parts (automotive, aircraft, industrial machinery)

**⚠️ AM Constraints and Challenges**

AM is not a panacea and has the following constraints:

  * **Build Speed** : Unsuitable for mass production (injection molding 1 part/seconds vs AM hours). Economic break-even typically below 1,000 units
  * **Build Size Limitations** : Large parts exceeding build volume (typically around 200×200×200mm for many systems) require segmented manufacturing
  * **Surface Quality** : Layer lines remain, requiring post-processing (polishing, machining) when high-precision surfaces are needed
  * **Material Anisotropy** : Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY-plane), especially in FDM
  * **Material Cost** : AM-grade materials are 2-10 times more expensive than commodity materials (though can be offset by material efficiency and design optimization)

## 1.2 Seven AM Process Classifications by ISO/ASTM 52900

### 1.2.1 Overview of AM Process Classifications

The ISO/ASTM 52900:2021 standard classifies all AM technologies into **seven process categories based on energy source and material delivery method**. Each process has unique advantages and disadvantages, requiring selection of the optimal technology according to application.
    
    
     flowchart TD AM[Additive Manufacturing  
    Seven Processes] --> MEX[Material Extrusion] AM --> VPP[Vat Photopolymerization] AM --> PBF[Powder Bed Fusion] AM --> MJ[Material Jetting] AM --> BJ[Binder Jetting] AM --> SL[Sheet Lamination] AM --> DED[Directed Energy Deposition] MEX --> MEX_EX[FDM/FFF  
    Low-cost & Widespread] VPP --> VPP_EX[SLA/DLP  
    High precision & Surface quality] PBF --> PBF_EX[SLS/SLM/EBM  
    High strength & Metal capable] style AM fill:#f093fb style MEX fill:#e3f2fd style VPP fill:#fff3e0 style PBF fill:#e8f5e9 style MJ fill:#f3e5f5 style BJ fill:#fce4ec style SL fill:#fff9c4 style DED fill:#fce4ec 

### 1.2.2 Material Extrusion (MEX)

**Principle** : Thermoplastic filament is heated and melted, extruded from a nozzle, and deposited layer by layer. The most widespread technology (also called FDM/FFF).

Process: Filament → Heated nozzle (190-260°C) → Melt extrusion → Cooling solidification → Next layer deposition 

**Characteristics:**

  * **Low Cost** : Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)
  * **Material Diversity** : PLA, ABS, PETG, nylon, PC, carbon fiber composites, PEEK (high-performance)
  * **Build Speed** : 20-150 mm³/s (moderate), layer height 0.1-0.4mm
  * **Accuracy** : ±0.2-0.5 mm (desktop), ±0.1 mm (industrial)
  * **Surface Quality** : Visible layer lines (improvable with post-processing)
  * **Material Anisotropy** : Z-axis (build direction) strength 20-80% lower (weak interlayer bonding)

**Applications:**

  * Prototyping (most common use, low-cost and fast)
  * Jigs and tools (used in manufacturing, lightweight and easily customizable)
  * Educational models (widely used in schools and universities, safe and low-cost)
  * End-use parts (custom hearing aids, orthotic devices, architectural models)

**💡 Representative FDM Systems**

  * **Ultimaker S5** : Dual head, build volume 330×240×300mm, $6,000
  * **Prusa i3 MK4** : Open-source based, high reliability, $1,200
  * **Stratasys Fortus 450mc** : Industrial, ULTEM 9085 compatible, $250,000
  * **Markforged X7** : Continuous carbon fiber composite capable, $100,000

### 1.2.3 Vat Photopolymerization (VPP)

**Principle** : Liquid photopolymer resin is selectively cured by ultraviolet (UV) laser or projector light and deposited layer by layer.

Process: UV irradiation → Photopolymerization → Solidification → Build platform raise → Next layer exposure 

**Two Main VPP Methods:**

  1. **SLA (Stereolithography)** : UV laser (355 nm) scanned with galvanometer mirrors, point-by-point curing. High precision but slow.
  2. **DLP (Digital Light Processing)** : Entire layer exposed simultaneously by projector. Fast but resolution dependent on projector pixels (Full HD: 1920×1080).
  3. **LCD-MSLA (Masked SLA)** : Uses LCD mask, similar to DLP but lower cost (many desktop units $200-$1,000).

**Characteristics:**

  * **High Precision** : XY resolution 25-100 μm, Z resolution 10-50 μm (highest level among all AM technologies)
  * **Surface Quality** : Smooth surface (Ra < 5 μm), layer lines barely visible
  * **Build Speed** : SLA (10-50 mm³/s), DLP/LCD (100-500 mm³/s, area-dependent)
  * **Material Limitations** : Photopolymer resins only (mechanical properties often inferior to FDM)
  * **Post-processing Required** : Washing (IPA etc.) → Post-curing (UV irradiation) → Support removal

**Applications:**

  * Dental applications (orthodontic models, surgical guides, dentures, millions produced annually)
  * Wax models for jewelry casting (high precision, complex geometries)
  * Medical models (presurgical planning, anatomical models, patient education)
  * Master models (for silicone molding, design validation)

### 1.2.4 Powder Bed Fusion (PBF)

**Principle** : Powder material is spread in thin layers, selectively melted/sintered by laser or electron beam, cooled and solidified layer by layer. Compatible with metals, polymers, and ceramics.

Process: Powder spreading → Laser/electron beam scanning → Melting/sintering → Solidification → Next layer powder spreading 

**Three Main PBF Methods:**

  1. **SLS (Selective Laser Sintering)** : Laser sinters polymer powder (PA12 nylon etc.). No support required (surrounding powder provides support).
  2. **SLM (Selective Laser Melting)** : Completely melts metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). Produces high-density parts (relative density >99%).
  3. **EBM (Electron Beam Melting)** : Melts metal powder with electron beam. High-temperature preheating (650-1000°C) reduces residual stress with faster build speed.

**Characteristics:**

  * **High Strength** : Melting and resolidification produces mechanical properties comparable to forged materials (tensile strength 500-1200 MPa)
  * **Complex Geometry Capable** : Overhang fabrication without support (powder provides support)
  * **Material Diversity** : Ti alloys, Al alloys, stainless steel, Ni superalloys, Co-Cr alloys, nylon
  * **High Cost** : Equipment price $200,000-$1,500,000, material cost $50-$500/kg
  * **Post-processing** : Support removal, heat treatment (stress relief), surface finishing (blasting, polishing)

**Applications:**

  * Aerospace components (weight reduction, integration, GE LEAP fuel nozzle etc.)
  * Medical implants (patient-specific geometry, porous structures, Ti-6Al-4V)
  * Molds (conformal cooling, complex geometries, H13 tool steel)
  * Automotive parts (lightweight brackets, custom engine components)

### 1.2.5 Material Jetting (MJ)

**Principle** : Similar to inkjet printers, droplets of material (photopolymer resin or wax) are jetted from heads and immediately cured by UV irradiation for layer-by-layer deposition.

**Characteristics:**

  * **Ultra-high Precision** : XY resolution 42-85 μm, Z resolution 16-32 μm
  * **Multi-material** : Can use multiple materials and colors in single build
  * **Full-color Fabrication** : Over 10 million colors expressible through CMYK resin combinations
  * **Surface Quality** : Extremely smooth (virtually no layer lines)
  * **High Cost** : Equipment $50,000-$300,000, material cost $200-$600/kg
  * **Material Limitations** : Photopolymer resins only, moderate mechanical properties

**Applications:** : medicalmodel（soft tissue・hard tissue differentmaterial againcurrent）、architecturetype、validatemodel

### 1.2.6 Binder Jetting (BJ)

**Principle** : Liquid binder (adhesive) is jetted inkjet-style onto powder bed to bond powder particles. Strength enhanced through sintering or infiltration after building.

**Characteristics:**

  * **Fast Fabrication** : No laser scanning required, entire layer processed simultaneously, build speed 100-500 mm³/s
  * **Material Diversity** : Metal powders, ceramics, sand molds (for casting), full-color (gypsum)
  * **No Support Required** : Surrounding powder provides support, recyclable after removal
  * **Low Density Issue** : Fragile before sintering (green density 50-60%), relative density 90-98% even after sintering
  * **Post-processing Required** : Debinding → Sintering (metal: 1200-1400°C) → Infiltration (copper/bronze)

**Applications:** : Sand casting molds (large castings like engine blocks), metal parts (Desktop Metal, HP Metal Jet), full-color objects (souvenirs, educational models)

### 1.2.7 Sheet Lamination (SL)

**Principle** : Sheet materials (paper, metal foil, plastic film) are laminated and bonded by adhesive or welding. Each layer contour-cut by laser or blade.

**Representative Technologies:**

  * **LOM (Laminated Object Manufacturing)** : Paper/plastic sheets, laminated with adhesive, laser cut
  * **UAM (Ultrasonic Additive Manufacturing)** : Metal foils ultrasonically welded, contour machined by CNC

**Characteristics:** typebuildpossible、materiallow cost、accuracyMediumabout、applicationlimited（mainlymodel、metalinsensorequal）

### 1.2.8 Directed Energy Deposition (DED)

**Principle** : Metal powder or wire is fed and melted by laser, electron beam, or arc, deposited on substrate. Used for large parts and repair of existing parts.

**Characteristics:**

  * **Fast Deposition** : Deposition rate 1-5 kg/h (10-50 times PBF)
  * **Large-scale Capable** : Minimal build volume limitations (using multi-axis robot arms)
  * **Repair & Coating**: Repair worn parts of existing components, form surface hardening layers
  * **Low Precision** : Accuracy ±0.5-2 mm, post-processing (machining) required

**Applications:** : Turbine blade repair, large aerospace components, wear-resistant coatings for tools

**⚠️ Process Selection Guidelines**

Optimal AM process varies according to application requirements:

  * **Precision Priority** → VPP (SLA/DLP) or MJ
  * **Low-cost & Widespread** → MEX (FDM/FFF)
  * **High-strength Metal Parts** → PBF (SLM/EBM)
  * **Mass Production (Sand molds)** → BJ
  * **Large-scale & Fast Deposition** → DED

## 1.3 STL File Format and Data Processing

### 1.3.1 Structure of STL Files

STL (STereoLithography) is **the most widely used 3D model file format in AM** , developed by 3D Systems in 1987. STL files represent object surfaces as **a collection of triangular meshes**.

#### Basic Structure of STL Files

STL file = Normal vector (n) + Three vertex coordinates (v1, v2, v3) × Number of triangles 

**ASCII STL Format Example:**
    
    
    solid cube facet normal 0 0 1 outer loop vertex 0 0 10 vertex 10 0 10 vertex 10 10 10 endloop endfacet facet normal 0 0 1 outer loop vertex 0 0 10 vertex 10 10 10 vertex 0 10 10 endloop endfacet ... endsolid cube 

**Two Types of STL Format:**

  1. **ASCII STL** : Human-readable text format. Large file size (10-20x Binary for same model). Useful for debugging and validation.
  2. **Binary STL** : Binary format, small file size, fast processing. Standard for industrial use. Structure: 80-byte header + 4 bytes (triangle count) + 50 bytes per triangle (12B normal + 36B vertices + 2B attribute).

### 1.3.2 Important Concepts of STL Files

#### 1\. Normal Vector

Each triangular facet has a **normal vector (outward direction)** defined, distinguishing object 'inside' from 'outside'. Normal direction is determined by the **right-hand rule** :

Normal n = (v2 - v1) × (v3 - v1) / |(v2 - v1) × (v3 - v1)| 

**Vertex Ordering Rule:** Vertices v1, v2, v3 are arranged counter-clockwise (CCW), and when viewed from outside, counter-clockwise order results in outward-facing normal.

#### 2\. Manifold Conditions

For an STL mesh to be 3D printable, it must be **manifold** :

  * **Edge Sharing** : Every edge must be shared by exactly two triangles
  * **Vertex Sharing** : Every vertex must belong to a continuous triangle fan
  * **Closed Surface** : Forms a completely closed surface without holes or openings
  * **No Self-Intersection** : Triangles do not intersect or penetrate each other

**⚠️ Non-Manifold Mesh Problems**

Non-manifold meshes are unprintable in 3D. Typical problems:

  * **Holes** : Unclosed surface, edges belonging to only one triangle
  * **T-junction** : Edge shared by three or more triangles
  * **Inverted Normals** : Triangles with normals facing inward mixed in
  * **Duplicate Vertices** : Multiple vertices existing at the same position
  * **Degenerate Triangles** : Triangles with zero or near-zero area

These problems cause errors in slicer software and lead to print failures.

### 1.3.3 STL File Quality Metrics

STL mesh quality is evaluated by the following metrics:

  1. **Triangle Count** : Typically 10,000-500,000. Avoid too few (coarse model) or too many (large file size, processing delay).
  2. **Edge Length Uniformity** : Mixture of extremely large and small triangles degrades build quality. Ideally 0.1-1.0 mm range.
  3. **Aspect Ratio** : Elongated triangles (high aspect ratio) cause numerical errors. Ideally aspect ratio < 10.
  4. **Normal Consistency** : All normals uniformly outward-facing. Mixed inverted normals cause inside/outside determination errors.

**💡 STL File Resolution Tradeoff**

STL mesh resolution (triangle count) involves a tradeoff between accuracy and file size:

  * **Low Resolution (1,000-10,000 triangles)** : Fast processing, small file, but curved surfaces appear faceted (visible faceting)
  * **Medium Resolution (10,000-100,000 triangles)** : Appropriate for most applications, good balance
  * **High Resolution (100,000-1,000,000 triangles)** : Smooth curved surfaces, but large file size (tens of MB), processing delay

When exporting STL from CAD software, control resolution with **Chordal Tolerance** or **Angle Tolerance**. Recommended values: chordal tolerance 0.01-0.1 mm, angle tolerance 5-15 degrees.

### 1.3.4 STL Processing with Python Libraries

Main libraries for handling STL files in Python:

  1. **numpy-stl** : Fast STL read/write, volume/surface area calculation, normal vector operations. Simple and lightweight.
  2. **trimesh** : Comprehensive 3D mesh processing library. Mesh repair, Boolean operations, raycasting, collision detection. Feature-rich but many dependencies.
  3. **PyMesh** : Advanced mesh processing (remeshing, subdivision, feature extraction). Somewhat complex installation.

**Basic numpy-stl Usage:**
    
    
    from stl import mesh import numpy as np # STL read your_mesh = mesh.Mesh.from_file('model.stl') # basicgeometric information volume, cog, inertia = your_mesh.get_mass_properties() print(f"Volume: {volume:.2f} mm³") print(f"Center of Gravity: {cog}") print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²") # triangle count print(f"Number of Triangles: {len(your_mesh.vectors)}") 

## 1.4 Slicing and Toolpath Generation

The process of converting STL files into commands (G-code) that 3D printers understand is called **slicing**. This section covers basic principles of slicing, toolpath strategies, and G-code fundamentals.

### 1.4.1 Basic Principles of Slicing

Slicing is the process of horizontally cutting a 3D model at constant heights (layer heights) and extracting contours of each layer:
    
    
     flowchart TD A[3D Model  
    STL File] --> B[Slice layer by layer  
    in Z-axis direction] B --> C[Extract layer contours  
    Contour Detection] C --> D[Generate shells  
    Perimeter Path] D --> E[Generate infill  
    Infill Path] E --> F[Add support  
    Support Structure] F --> G[Optimize toolpath  
    Retraction/Travel] G --> H[G-code Output] style A fill:#e3f2fd style H fill:#e8f5e9 

#### Layer Height Selection

Layer height is the most important parameter determining the tradeoff between build quality and build time:

Layer Height | Build Quality | Build Time | Typical Applications  
---|---|---|---  
0.1 mm (Extra Fine) | Very High (layer lines barely visible) | Very Long (×2-3x) | Figurines, medical models, end-use parts  
0.2 mm (Standard) | Good (layer lines visible but acceptable) | Standard | General prototypes, functional parts  
0.3 mm (Coarse) | Low (visible layer lines) | Short (×0.5x) | Initial prototypes, internal structural parts  
  
**⚠️ Layer Height approximately**

Layer Height nozzlediameter **25-80%** settingdonecessaryexists。Example0.4mmnozzle case、Layer Height 0.1-0.32mm recommended range。 exceeding and 、resin extrusion amount not、nozzle before layers do。

### 1.4.2 Shell and Infill Strategies

#### Shell (Perimeter) Generation

**Shell (Shell/Perimeter)** is the path forming the outer perimeter of each layer:

  * **Perimeter Count** : Typically 2-4. Affects external quality and strength. 
    * 1: Very weak, high transparency, decorative only
    * 2this: Standard（）
    * 3-4: High strength, improved surface quality, improved airtightness
  * **Shell Order** : Inside-Out is common. Outside-In used when prioritizing surface quality.

#### Infill Pattern

**Infill** forms internal structure, controlling strength and material usage:

Pattern | Strength | Print Speed | Material Usage | Features  
---|---|---|---|---  
Grid | Medium | Fast | Medium | 、isotropic、Standardselect  
Honeycomb | High | Slow | Medium | HighStrength、weightexcellent ratio、aerospaceapplication  
Gyroid | nonalways High | Medium | Medium | 3D isotropic, curved surfaces, latest recommendation  
Concentric | Low | Fast | Less | Flexibility priority, follows shell  
Lines | Low（anisotropic） | nonalways Fast | Less | Highfast printing、directionalStrength  
  
**💡 Infill Density Guidelines**

  * **0-10%** : Decorative, non-load-bearing parts (material saving priority)
  * **20%** : Standardprototype（）
  * **40-60%** : Functionpart、HighStrengthrequirement
  * **100%** : end-use product、watertightnessrequirement、HighStrength（Build Time×3-5times）

### 1.4.3 Support Structure Generation

Parts with overhang angles exceeding 45 degrees require **support structures** :

#### Support Types

  * **Linear Support（linesupport）** : support。 removaleasy but、Material Usagemany。
  * **Tree Support（support）** : tree-like minutesdosupport。Material Usage30-50%reduction、removal。CuraPrusaSlicer Standardsupport。
  * **Interface Layers** : Thin interface layers on support top. Easy to remove, improved surface quality. Typically 2-4 layers.

#### Important Support Parameters

Parameter | Recommended Value | Effect  
---|---|---  
Overhang Angle | 45-60° | Generate support above this angle  
Support Density | 10-20% | density Highstable removaldifficult  
Support Z Distance | 0.2-0.3 mm | Gap between support and part (ease of removal)  
Interface Layers | 2-4 layers | Number of interface layers (balance of surface quality and removability)  
  
### 1.4.4 G-code Fundamentals

**G-code** 、3DprinterCNC controldoStandardnumbervaluecontrol。each 1 Command tabledo：

#### Main G-code Commands

Command | Category | Function | Example  
---|---|---|---  
G0 | Movement | HighMovement（non-extrusion） | G0 X100 Y50 Z10 F6000  
G1 | Movement | lineMovement（） | G1 X120 Y60 E0.5 F1200  
G28 | Initialization | Return to home position | G28 (all axes), G28 Z (Z-axis only)  
M104 | Temperature | nozzleTemperaturesetting（no-wait） | M104 S200  
M109 | Temperature | nozzleTemperaturesetting（machine） | M109 S210  
M140 | Temperature | bedTemperaturesetting（no-wait） | M140 S60  
M190 | Temperature | bedTemperaturesetting（machine） | M190 S60  
  
#### G-code Example（buildstart sectionminutes）
    
    
    ; === Start G-code === M140 S60 ; bed 60°C heatingstart（no-wait） M104 S210 ; nozzle 210°C heatingstart（no-wait） G28 ; all G29 ; leveling（bedmesh） M190 S60 ; bedTemperaturereaching machine M109 S210 ; nozzleTemperaturereaching machine G92 E0 ; extrusion amount zero reset G1 Z2.0 F3000 ; Z 2mmabove（safety assurance） G1 X10 Y10 F5000 ; Movement G1 Z0.3 F3000 ; Z 0.3mmbelow（initiallayersHigh） G1 X100 E10 F1500 ; prime line（nozzleremoval） G92 E0 ; extrusion amount againdegreezero reset ; === buildstart === 

### 1.4.5 Major Slicing Software

Software | License | Features | recommendedapplication  
---|---|---|---  
Cura | Open Source | 、preset、Tree SupportStandard | beginner~Medium、FDMfor  
PrusaSlicer | Open Source | Highdegreesetting、numberLayer Height、custom support | Medium~advanced、optimization  
Slic3r | Open Source | Original PrusaSlicer, lightweight | Legacy systems, research applications  
Simplify3D | Commercial ($150) | Highslicing、process、detailedcontrol | Professional, industrial applications  
IdeaMaker | Free | Raise3Dfor versatilityHigh、intuitiveUI | Raise3D users, beginners  
  
### 1.4.6 Toolpath Optimization Strategies

efficienttool 、Build Time・quality・Material Usage improvedo：

  * **retraction（Retraction）** : Movementtime filament stringing（） 。 
    * Distance: 1-6mm (Bowden tube: 4-6mm, direct drive: 1-2mm)
    * Speed: 25-45 mm/s
    * Excessive retraction causes nozzle clogging
  * **Z-hop（Zaxis hop）** : Movementtime nozzle abovebuildobject and times。0.2-0.5mmabove。Build Time surfacequalityenhance。
  * **（Combing）** : Movement infillabove limit、surfacetoMovement Low。when appearance matters effective。
  * **Seam Position** : Strategy for aligning layer start/end points. 
    * Random: Random placement (less visible)
    * Aligned: Linear placement (easier to remove seam with post-processing)
    * Sharpest Corner: Placed at sharpest corner (less noticeable)

### Example 1: Reading STL Files and Obtaining Basic Information
    
    
    # =================================== # Example 1: Reading STL Files and Obtaining Basic Information # =================================== import numpy as np from stl import mesh # Read STL file your_mesh = mesh.Mesh.from_file('model.stl') # Get basic geometric information volume, cog, inertia = your_mesh.get_mass_properties() print("=== STL File Basic Information ===") print(f"Volume: {volume:.2f} mm³") print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²") print(f"Center of Gravity: [{cog[0]:.2f}, {cog[1]:.2f}, {cog[2]:.2f}] mm") print(f"Number of Triangles: {len(your_mesh.vectors)}") # Calculate bounding box (minimum enclosing cuboid) min_coords = your_mesh.vectors.min(axis=(0, 1)) max_coords = your_mesh.vectors.max(axis=(0, 1)) dimensions = max_coords - min_coords print(f"\n=== Bounding Box ===") print(f"X: {min_coords[0]:.2f} to {max_coords[0]:.2f} mm (Width: {dimensions[0]:.2f} mm)") print(f"Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} mm (Depth: {dimensions[1]:.2f} mm)") print(f"Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} mm (High: {dimensions[2]:.2f} mm)") # Build Time estimation（Layer Height0.2mm、speed50mm/s and assumption） layer_height = 0.2 # mm print_speed = 50 # mm/s num_layers = int(dimensions[2] / layer_height) # Simple calculation: estimation based on surface area estimated_path_length = your_mesh.areas.sum() / layer_height # mm estimated_time_seconds = estimated_path_length / print_speed estimated_time_minutes = estimated_time_seconds / 60 print(f"\n=== Build Estimation ===") print(f"Number of layers (0.2mm/layer): {num_layers} layers") print(f"estimationBuild Time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.2f} hours)") # outputExample: # === STL File Basic Information === # Volume: 12450.75 mm³ # Surface Area: 5832.42 mm² # Center of Gravity: [25.34, 18.92, 15.67] mm # Number of Triangles: 2456 # # === Bounding Box === # X: 0.00 to 50.00 mm (Width: 50.00 mm) # Y: 0.00 to 40.00 mm (Depth: 40.00 mm) # Z: 0.00 to 30.00 mm (High: 30.00 mm) # # === Build Estimation === # Number of layers (0.2mm/layer): 150 layers # estimationBuild Time: 97.2 minutes (1.62 hours) 

### Example 2: Mesh Normal Vector Validation
    
    
    # =================================== # Example 2: Mesh Normal Vector Validation # =================================== import numpy as np from stl import mesh def check_normals(mesh_data): """Check consistency of normal vectors in STL mesh Args: mesh_data: numpy-stl Meshobject Returns: tuple: (flipped_count, total_count, percentage) """ # Verify normal direction with right-hand rule flipped_count = 0 total_count = len(mesh_data.vectors) for i, facet in enumerate(mesh_data.vectors): v0, v1, v2 = facet # Calculate edge vectors edge1 = v1 - v0 edge2 = v2 - v0 # Calculate normal with cross product (right-hand system) calculated_normal = np.cross(edge1, edge2) # Normalize norm = np.linalg.norm(calculated_normal) if norm > 1e-10: # Confirm not zero vector calculated_normal = calculated_normal / norm else: continue # Skip degenerate triangles # Compare with normals stored in file stored_normal = mesh_data.normals[i] stored_norm = np.linalg.norm(stored_normal) if stored_norm > 1e-10: stored_normal = stored_normal / stored_norm # Check direction match with dot product dot_product = np.dot(calculated_normal, stored_normal) # If dot product negative, opposite direction if dot_product < 0: flipped_count += 1 percentage = (flipped_count / total_count) * 100 if total_count > 0 else 0 return flipped_count, total_count, percentage # STL read your_mesh = mesh.Mesh.from_file('model.stl') # Execute normal check flipped, total, percent = check_normals(your_mesh) print("=== Normal Vector Validation Results ===") print(f"Total triangle count: {total}") print(f"Flipped normal count: {flipped}") print(f"Flip rate: {percent:.2f}%") if flipped == 0: print("\n✅ All normals are correctly oriented") print(" This mesh is 3D printable") elif percent < 5: print("\n⚠️ Some normals are flipped (minor)") print(" selfcorrectdopossibleproperty High") else: print("\n❌ Many normals are flipped (critical)") print(" Recommend repair with mesh repair tools (Meshmixer, netfabb)") # outputExample: # === Normal Vector Validation Results === # Total triangle count: 2456 # Flipped normal count: 0 # Flip rate: 0.00% # # ✅ All normals are correctly oriented # This mesh is 3D printable 

### Example 3: Manifold Checking
    
    
    # =================================== # Example 3: Manifold (Watertight) Checking # =================================== import trimesh # Read STL file (trimesh attempts auto-repair) mesh = trimesh.load('model.stl') print("=== Mesh Quality Diagnosis ===") # Basic information print(f"Vertex count: {len(mesh.vertices)}") print(f"Face count: {len(mesh.faces)}") print(f"Volume: {mesh.volume:.2f} mm³") # Check manifold property print(f"\n=== 3D Printability Check ===") print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") print(f"Is valid: {mesh.is_valid}") # Diagnose problems in detail if not mesh.is_watertight: # Detect number of holes try: edges = mesh.edges_unique edges_sorted = mesh.edges_sorted duplicate_edges = len(edges_sorted) - len(edges) print(f"\n⚠️ Problems Detected:") print(f" - Mesh has holes") print(f" - Duplicate edge count: {duplicate_edges}") except: print(f"\n⚠️ Mesh structure has problems") # Attempt repair if not mesh.is_watertight or not mesh.is_winding_consistent: print(f"\n🔧 auto-repair executeMedium...") # Fix normals trimesh.repair.fix_normals(mesh) print(" ✓ Fixed normal vectors") # Fill holes trimesh.repair.fill_holes(mesh) print(" ✓ Filled holes") # Remove degenerate faces mesh.remove_degenerate_faces() print(" ✓ Removed degenerate faces") # Merge duplicate vertices mesh.merge_vertices() print(" ✓ Merged duplicate vertices") # Check state after repair print(f"\n=== State After Repair ===") print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") # Save repaired mesh if mesh.is_watertight: mesh.export('model_repaired.stl') print(f"\n✅ Repair complete! Saved as model_repaired.stl") else: print(f"\n❌ Auto-repair failed. Recommend dedicated tools like Meshmixer") else: print(f"\n✅ This mesh is 3D printable") # outputExample: # === Mesh Quality Diagnosis === # Vertex count: 1534 # Face count: 2456 # Volume: 12450.75 mm³ # # === 3D Printability Check === # Is watertight: True # Is winding consistent: True # Is valid: True # # ✅ This mesh is 3D printable 

### Example 4: Basic Slicing Algorithm
    
    
    # =================================== # Example 4: Basic Slicing Algorithm # =================================== import numpy as np from stl import mesh def slice_mesh_at_height(mesh_data, z_height): """Temperatureprof generate Args: t (array): hours [min] T_target (float): retentionTemperature [°C] heating_rate (float): heating rate [°C/min] hold_time (float): retentionhours [min] cooling_rate (float): coolingspeed [°C/min] Returns: array: Temperatureprof [°C] """ T_room = 25 # temperature T = np.zeros_like(t) # heatinghours t_heat = (T_target - T_room) / heating_rate # coolingstart time t_cool_start = t_heat + hold_time for i, time in enumerate(t): if time <= t_heat: # heating T[i] = T_room + heating_rate * time elif time <= t_cool_start: # retention T[i] = T_target else: # cooling T[i] = T_target - cooling_rate * (time - t_cool_start) T[i] = max(T[i], T_room) # temperatureor lessdoes not become return T def simulate_reaction_progress(T, t, Ea, D0, r0): """Temperatureprof basisreactionprogress calculation Args: T (array): Temperatureprof [°C] t (array): hours [min] Ea (float): activationenergy [J/mol] D0 (float): frequency factor [m²/s] r0 (float): particle radius [m] Returns: array: conversion rate """ R = 8.314 C0 = 10000 alpha = np.zeros_like(t) for i in range(1, len(t)): T_k = T[i] + 273.15 D = D0 * np.exp(-Ea / (R * T_k)) k = D * C0 / r0**2 dt = (t[i] - t[i-1]) * 60 # min → s # simple integrationminutes（hoursinreactionprogress） if alpha[i-1] < 0.99: dalpha = k * dt / (2 * (1 - (1-alpha[i-1])**(1/3))) alpha[i] = min(alpha[i-1] + dalpha, 1.0) else: alpha[i] = alpha[i-1] return alpha # Parametersetting T_target = 1200 # °C hold_time = 240 # min (4 hours) Ea = 300e3 # J/mol D0 = 5e-4 # m²/s r0 = 5e-6 # m # differentheating rateincomparison heating_rates = [2, 5, 10, 20] # °C/min cooling_rate = 3 # °C/min # hours t_max = 800 # min t = np.linspace(0, t_max, 2000) # plot fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) # Temperatureprof for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) ax1.plot(t/60, T_profile, linewidth=2, label=f'{hr}°C/min') ax1.set_xlabel('Time (hours)', fontsize=12) ax1.set_ylabel('Temperature (°C)', fontsize=12) ax1.set_title('Temperature Profiles', fontsize=14, fontweight='bold') ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) ax1.set_xlim([0, t_max/60]) # reactionprogress for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0) ax2.plot(t/60, alpha, linewidth=2, label=f'{hr}°C/min') ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='Target (95%)') ax2.set_xlabel('Time (hours)', fontsize=12) ax2.set_ylabel('Conversion', fontsize=12) ax2.set_title('Reaction Progress', fontsize=14, fontweight='bold') ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) ax2.set_xlim([0, t_max/60]) ax2.set_ylim([0, 1]) plt.tight_layout() plt.savefig('temperature_profile_optimization.png', dpi=300, bbox_inches='tight') plt.show() # eachheating ratein95%reactionreachinghours calculation print("\n95%reactionreachinghours comparison:") print("=" * 60) for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0) # 95%arrival time idx_95 = np.where(alpha >= 0.95)[0] if len(idx_95) > 0: t_95 = t[idx_95[0]] / 60 print(f"heating rate {hr:2d}°C/min: t₉₅ = {t_95:.1f} hours") else: print(f"heating rate {hr:2d}°C/min: reactionincomplete") # outputExample: # 95%reactionreachinghours comparison: # ============================================================ # heating rate 2°C/min: t₉₅ = 7.8 hours # heating rate 5°C/min: t₉₅ = 7.2 hours # heating rate 10°C/min: t₉₅ = 6.9 hours # heating rate 20°C/min: t₉₅ = 6.7 hours 

## Exercises

### 1.5.1 pycalphad and 

**pycalphad** 、CALPHAD（CALculation of PHAse Diagrams）method basismutualfigurecalculationfor Pythonlibrary。forcefromequilibrium phase calculation、reaction design useful。

**💡 CALPHADmethod point**

  * multicomponent system（3originsystemor more） complexmutualfigure calculationpossible
  * experiment Lesssystem possible
  * Temperature・set・pressuredependency comprehensive 

### 1.5.2 binary phasefigure calculationExample
    
    
    # =================================== # Example 5: pycalphad mutualfigurecalculation # =================================== # note: pycalphad necessary # pip install pycalphad from pycalphad import Database, equilibrium, variables as v import matplotlib.pyplot as plt import numpy as np # TDB read（insimplifiedExample） # actualforappropriateTDB necessary # Example: BaO-TiO2system # simplifiedTDBstring（actual thancomplex） tdb_string = """ $ BaO-TiO2 system (simplified) ELEMENT BA BCC_A2 137.327 ! ELEMENT TI HCP_A3 47.867 ! ELEMENT O GAS 15.999 ! FUNCTION GBCCBA 298.15 +GHSERBA; 6000 N ! FUNCTION GHCPTI 298.15 +GHSERTI; 6000 N ! FUNCTION GGASO 298.15 +GHSERO; 6000 N ! PHASE LIQUID:L % 1 1.0 ! PHASE BAO_CUBIC % 2 1 1 ! PHASE TIO2_RUTILE % 2 1 2 ! PHASE BATIO3 % 3 1 1 3 ! """ # note: actual calculationforpositiveequationTDB necessary # inexplanation print("pycalphadbymutualfigurecalculation :") print("=" * 60) print("1. TDB（force） ") print("2. Temperature・setrangescope setting") print("3. calculation execute") print("4. stablemutual possible") print() print("actual forExample:") print("- BaO-TiO2system: BaTiO3 Temperature・setrangescope") print("- Si-Nsystem: Si3N4 stableregion") print("- multicomponent systemceramics mutual") # plot（real basis） fig, ax = plt.subplots(figsize=(10, 7)) # Temperaturerangescope T = np.linspace(800, 1600, 100) # eachmutual stableregion（figure） # BaO + TiO2 → BaTiO3 reaction BaO_region = np.ones_like(T) * 0.3 TiO2_region = np.ones_like(T) * 0.7 BaTiO3_region = np.where((T > 1100) & (T < 1400), 0.5, np.nan) ax.fill_between(T, 0, BaO_region, alpha=0.3, color='blue', label='BaO + TiO2') ax.fill_between(T, BaO_region, TiO2_region, alpha=0.3, color='green', label='BaTiO3 stable') ax.fill_between(T, TiO2_region, 1, alpha=0.3, color='red', label='Liquid') ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='BaTiO3 composition') ax.axvline(x=1100, color='gray', linestyle=':', linewidth=1, alpha=0.5) ax.axvline(x=1400, color='gray', linestyle=':', linewidth=1, alpha=0.5) ax.set_xlabel('Temperature (°C)', fontsize=12) ax.set_ylabel('Composition (BaO mole fraction)', fontsize=12) ax.set_title('Conceptual Phase Diagram: BaO-TiO2', fontsize=14, fontweight='bold') ax.legend(fontsize=10, loc='upper right') ax.grid(True, alpha=0.3) ax.set_xlim([800, 1600]) ax.set_ylim([0, 1]) # note ax.text(1250, 0.5, 'BaTiO₃\nformation\nregion', fontsize=11, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)) plt.tight_layout() plt.savefig('phase_diagram_concept.png', dpi=300, bbox_inches='tight') plt.show() # actual usageExample（） """ # actual pycalphadusageExample db = Database('BaO-TiO2.tdb') # TDBread # calculation eq = equilibrium(db, ['BA', 'TI', 'O'], ['LIQUID', 'BATIO3'], {v.X('BA'): (0, 1, 0.01), v.T: (1000, 1600, 50), v.P: 101325}) # resultplot eq.plot() """ 

## 1.6 experimentmethod（DOE）byconditionsoptimization

### 1.6.1 DOE and 

experimentmethod（Design of Experiments, DOE） 、multiplenumber Parameter mutualmutualfordosystem 、 experimenttimesnumber optimalconditions method。

**mutualreaction optimizationnecessaryParameter：**

  * reactionTemperature（T）
  * retentionhours（t）
  * （r）
  * ratio（ratio）
  * scope（empty、、trueemptyetc.）

### 1.6.2 surfacemethod（Response Surface Methodology）
    
    
    # =================================== # Example 6: DOEbyconditionsoptimization # =================================== import numpy as np import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D from scipy.optimize import minimize # ifconversion ratemodel（Temperature and hours number） def reaction_yield(T, t, noise=0): """Temperature and hoursfromconversion rate calculation（ifmodel） Args: T (float): Temperature [°C] t (float): hours [hours] noise (float): Returns: float: conversion rate [%] """ # optimalvalue: T=1200°C, t=6 hours T_opt = 1200 t_opt = 6 # nextmodel（type） yield_val = 100 * np.exp(-((T-T_opt)/150)**2 - ((t-t_opt)/3)**2) # add if noise > 0: yield_val += np.random.normal(0, noise) return np.clip(yield_val, 0, 100) # experimentpoint（Mediummultiplemethod） T_levels = [1000, 1100, 1200, 1300, 1400] # °C t_levels = [2, 4, 6, 8, 10] # hours # experimentpoint T_grid, t_grid = np.meshgrid(T_levels, t_levels) yield_grid = np.zeros_like(T_grid, dtype=float) # eachexperimentpoint conversion rate measurement（） for i in range(len(t_levels)): for j in range(len(T_levels)): yield_grid[i, j] = reaction_yield(T_grid[i, j], t_grid[i, j], noise=2) # result display print("experimentmethodbyreactionconditionsoptimization") print("=" * 70) print(f"{'Temperature (°C)':<20} {'Time (hours)':<20} {'Yield (%)':<20}") print("-" * 70) for i in range(len(t_levels)): for j in range(len(T_levels)): print(f"{T_grid[i, j]:<20} {t_grid[i, j]:<20} {yield_grid[i, j]:<20.1f}") # conversion rate conditions max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape) T_best = T_grid[max_idx] t_best = t_grid[max_idx] yield_best = yield_grid[max_idx] print("-" * 70) print(f"optimalconditions: T = {T_best}°C, t = {t_best} hours") print(f"conversion rate: {yield_best:.1f}%") # 3Dplot fig = plt.figure(figsize=(14, 6)) # 3Dsurfaceplot ax1 = fig.add_subplot(121, projection='3d') T_fine = np.linspace(1000, 1400, 50) t_fine = np.linspace(2, 10, 50) T_mesh, t_mesh = np.meshgrid(T_fine, t_fine) yield_mesh = np.zeros_like(T_mesh) for i in range(len(t_fine)): for j in range(len(T_fine)): yield_mesh[i, j] = reaction_yield(T_mesh[i, j], t_mesh[i, j]) surf = ax1.plot_surface(T_mesh, t_mesh, yield_mesh, cmap='viridis', alpha=0.8, edgecolor='none') ax1.scatter(T_grid, t_grid, yield_grid, color='red', s=50, label='Experimental points') ax1.set_xlabel('Temperature (°C)', fontsize=10) ax1.set_ylabel('Time (hours)', fontsize=10) ax1.set_zlabel('Yield (%)', fontsize=10) ax1.set_title('Response Surface', fontsize=12, fontweight='bold') ax1.view_init(elev=25, azim=45) fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5) # equalHighlineplot ax2 = fig.add_subplot(122) contour = ax2.contourf(T_mesh, t_mesh, yield_mesh, levels=20, cmap='viridis') ax2.contour(T_mesh, t_mesh, yield_mesh, levels=10, colors='black', alpha=0.3, linewidths=0.5) ax2.scatter(T_grid, t_grid, c=yield_grid, s=100, edgecolors='red', linewidths=2, cmap='viridis') ax2.scatter(T_best, t_best, color='red', s=300, marker='*', edgecolors='white', linewidths=2, label='Optimum') ax2.set_xlabel('Temperature (°C)', fontsize=11) ax2.set_ylabel('Time (hours)', fontsize=11) ax2.set_title('Contour Map', fontsize=12, fontweight='bold') ax2.legend(fontsize=10) fig.colorbar(contour, ax=ax2, label='Yield (%)') plt.tight_layout() plt.savefig('doe_optimization.png', dpi=300, bbox_inches='tight') plt.show() 

### 1.6.3 experiment realapproach

actual mutualreactionin、or less order DOE fordo：

  1. **experiment** （2necessarybecausemethod）: effect largeParameter specific
  2. **surfacemethod** （Mediummultiplemethod）: optimalconditions 
  3. **confirmexperiment** : optimalconditions experiment、model validate

**✅ realExample: Li-ionpositiveLiCoO₂ synthesisoptimization**

research DOE forLiCoO₂ synthesisconditions optimizationresult：

  * experimenttimesnumber: conventionalmethod100times → DOEmethod25times（75%reduction）
  * optimalTemperature: 900°C（conventional 850°CthanHightemperature）
  * optimalretentionhours: 12hours（conventional 24hoursfromhalf）
  * quantity: 140 mAh/g → 155 mAh/g（11%enhance）

## 1.7 reactionspeedline fitting

### 1.7.1 experimentfrom speedconstantdecide
    
    
    # =================================== # Example 7: reactionspeedlinefitting # =================================== import numpy as np import matplotlib.pyplot as plt from scipy.optimize import curve_fit # experiment（hours vs conversion rate） # Example: BaTiO3synthesis @ 1200°C time_exp = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]) # hours conversion_exp = np.array([0, 0.15, 0.28, 0.38, 0.47, 0.60, 0.70, 0.78, 0.84, 0.90, 0.95]) # Janderequationmodel def jander_model(t, k): """Janderequationbyconversion ratecalculation Args: t (array): hours [hours] k (float): speedconstant Returns: array: conversion rate """ # [1 - (1-α)^(1/3)]² = kt α about kt = k * t alpha = 1 - (1 - np.sqrt(kt))**3 alpha = np.clip(alpha, 0, 1) # 0-1 rangescope limit return alpha # Ginstling-Brounshteinequation（separate diffusionmodel） def gb_model(t, k): """Ginstling-Brounshteinequation Args: t (array): hours k (float): speedconstant Returns: array: conversion rate """ # 1 - 2α/3 - (1-α)^(2/3) = kt # numbervalue necessary 、insimilarequation usage kt = k * t alpha = 1 - (1 - kt/2)**(3/2) alpha = np.clip(alpha, 0, 1) return alpha # Power law (equation) def power_law_model(t, k, n): """model Args: t (array): hours k (float): speedconstant n (float): number Returns: array: conversion rate """ alpha = k * t**n alpha = np.clip(alpha, 0, 1) return alpha # eachmodel fitting # Janderequation popt_jander, _ = curve_fit(jander_model, time_exp, conversion_exp, p0=[0.01]) k_jander = popt_jander[0] # Ginstling-Brounshteinequation popt_gb, _ = curve_fit(gb_model, time_exp, conversion_exp, p0=[0.01]) k_gb = popt_gb[0] # Power law popt_power, _ = curve_fit(power_law_model, time_exp, conversion_exp, p0=[0.1, 0.5]) k_power, n_power = popt_power # linegenerate t_fit = np.linspace(0, 20, 200) alpha_jander = jander_model(t_fit, k_jander) alpha_gb = gb_model(t_fit, k_gb) alpha_power = power_law_model(t_fit, k_power, n_power) # differencecalculation residuals_jander = conversion_exp - jander_model(time_exp, k_jander) residuals_gb = conversion_exp - gb_model(time_exp, k_gb) residuals_power = conversion_exp - power_law_model(time_exp, k_power, n_power) # R²calculation def r_squared(y_true, y_pred): ss_res = np.sum((y_true - y_pred)**2) ss_tot = np.sum((y_true - np.mean(y_true))**2) return 1 - (ss_res / ss_tot) r2_jander = r_squared(conversion_exp, jander_model(time_exp, k_jander)) r2_gb = r_squared(conversion_exp, gb_model(time_exp, k_gb)) r2_power = r_squared(conversion_exp, power_law_model(time_exp, k_power, n_power)) # plot fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # fittingresult ax1.plot(time_exp, conversion_exp, 'ko', markersize=8, label='Experimental data') ax1.plot(t_fit, alpha_jander, 'b-', linewidth=2, label=f'Jander (R²={r2_jander:.4f})') ax1.plot(t_fit, alpha_gb, 'r-', linewidth=2, label=f'Ginstling-Brounshtein (R²={r2_gb:.4f})') ax1.plot(t_fit, alpha_power, 'g-', linewidth=2, label=f'Power law (R²={r2_power:.4f})') ax1.set_xlabel('Time (hours)', fontsize=12) ax1.set_ylabel('Conversion', fontsize=12) ax1.set_title('Kinetic Model Fitting', fontsize=14, fontweight='bold') ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) ax1.set_xlim([0, 20]) ax1.set_ylim([0, 1]) # differenceplot ax2.plot(time_exp, residuals_jander, 'bo-', label='Jander') ax2.plot(time_exp, residuals_gb, 'ro-', label='Ginstling-Brounshtein') ax2.plot(time_exp, residuals_power, 'go-', label='Power law') ax2.axhline(y=0, color='black', linestyle='--', linewidth=1) ax2.set_xlabel('Time (hours)', fontsize=12) ax2.set_ylabel('Residuals', fontsize=12) ax2.set_title('Residual Plot', fontsize=14, fontweight='bold') ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('kinetic_fitting.png', dpi=300, bbox_inches='tight') plt.show() # result print("\nreactionspeedmodel fittingresult:") print("=" * 70) print(f"{'Model':<25} {'Parameter':<30} {'R²':<10}") print("-" * 70) print(f"{'Jander':<25} {'k = ' + f'{k_jander:.4f} h⁻¹':<30} {r2_jander:.4f}") print(f"{'Ginstling-Brounshtein':<25} {'k = ' + f'{k_gb:.4f} h⁻¹':<30} {r2_gb:.4f}") print(f"{'Power law':<25} {'k = ' + f'{k_power:.4f}, n = {n_power:.4f}':<30} {r2_power:.4f}") print("=" * 70) print(f"\noptimalmodel: {'Jander' if r2_jander == max(r2_jander, r2_gb, r2_power) else 'GB' if r2_gb == max(r2_jander, r2_gb, r2_power) else 'Power law'}") # outputExample: # reactionspeedmodel fittingresult: # ====================================================================== # Model Parameter R² # ---------------------------------------------------------------------- # Jander k = 0.0289 h⁻¹ 0.9953 # Ginstling-Brounshtein k = 0.0412 h⁻¹ 0.9867 # Power law k = 0.2156, n = 0.5234 0.9982 # ====================================================================== # # optimalmodel: Power law 

## 1.8 Highdegree: structuremanufacturingcontrol

### 1.8.1 grain growth 

mutualreactionin、Hightemperature・lengthhoursretention thangrain growth 。 dostrategy：

  * **Two-step sintering** : Hightemperature hoursretentionafter、Lowtemperature lengthhoursretention
  * **usage** : grain growth（Example: MgO, Al₂O₃） quantity
  * **Spark Plasma Sintering (SPS)** : heating・hourssintering

### 1.8.2 reaction machineactivation

method（Highenergy） than、mutualreaction temperature progressthatpossible：
    
    
    # =================================== # Example 8: grain growth # =================================== import numpy as np import matplotlib.pyplot as plt def grain_growth(t, T, D0, Ea, G0, n): """grain growth hours Burke-Turnbullequation: G^n - G0^n = k*t Args: t (array): hours [hours] T (float): Temperature [K] D0 (float): frequency factor Ea (float): activationenergy [J/mol] G0 (float): initialdiameter [μm] n (float): grain growthnumber（typically2-4） Returns: array: diameter [μm] """ R = 8.314 k = D0 * np.exp(-Ea / (R * T)) G = (G0**n + k * t * 3600)**(1/n) # hours → seconds return G # Parametersetting D0_grain = 1e8 # μm^n/s Ea_grain = 400e3 # J/mol G0 = 0.5 # μm n = 3 # Temperature effect temps_celsius = [1100, 1200, 1300] t_range = np.linspace(0, 12, 100) # 0-12 hours plt.figure(figsize=(12, 5)) # Temperaturedependency plt.subplot(1, 2, 1) for T_c in temps_celsius: T_k = T_c + 273.15 G = grain_growth(t_range, T_k, D0_grain, Ea_grain, G0, n) plt.plot(t_range, G, linewidth=2, label=f'{T_c}°C') plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Target grain size') plt.xlabel('Time (hours)', fontsize=12) plt.ylabel('Grain Size (μm)', fontsize=12) plt.title('Grain Growth at Different Temperatures', fontsize=14, fontweight='bold') plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.ylim([0, 5]) # Two-step sintering Effect plt.subplot(1, 2, 2) # Conventional sintering: 1300°C, 6 hours t_conv = np.linspace(0, 6, 100) T_conv = 1300 + 273.15 G_conv = grain_growth(t_conv, T_conv, D0_grain, Ea_grain, G0, n) # Two-step: 1300°C 1h → 1200°C 5h t1 = np.linspace(0, 1, 20) G1 = grain_growth(t1, 1300+273.15, D0_grain, Ea_grain, G0, n) G_intermediate = G1[-1] t2 = np.linspace(0, 5, 80) G2 = grain_growth(t2, 1200+273.15, D0_grain, Ea_grain, G_intermediate, n) t_two_step = np.concatenate([t1, t2 + 1]) G_two_step = np.concatenate([G1, G2]) plt.plot(t_conv, G_conv, 'r-', linewidth=2, label='Conventional (1300°C)') plt.plot(t_two_step, G_two_step, 'b-', linewidth=2, label='Two-step (1300°C→1200°C)') plt.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5) plt.xlabel('Time (hours)', fontsize=12) plt.ylabel('Grain Size (μm)', fontsize=12) plt.title('Two-Step Sintering Strategy', fontsize=14, fontweight='bold') plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.ylim([0, 5]) plt.tight_layout() plt.savefig('grain_growth_control.png', dpi=300, bbox_inches='tight') plt.show() # finaldiameter comparison G_final_conv = grain_growth(6, 1300+273.15, D0_grain, Ea_grain, G0, n) G_final_two_step = G_two_step[-1] print("\ngrain growth comparison:") print("=" * 50) print(f"Conventional (1300°C, 6h): {G_final_conv:.2f} μm") print(f"Two-step (1300°C 1h + 1200°C 5h): {G_final_two_step:.2f} μm") print(f"diameterEffect: {(1 - G_final_two_step/G_final_conv)*100:.1f}%") # outputExample: # grain growth comparison: # ================================================== # Conventional (1300°C, 6h): 4.23 μm # Two-step (1300°C 1h + 1200°C 5h): 2.87 μm # diameterEffect: 32.2% 

## Learning Objectives confirm

Upon completing this chapter, you will be able to explain the following:

### basisthis

  * ✅ mutualreaction 3 rate-limiting（generate・interface reaction・diffusion） explanationcan
  * ✅ Arrheniusequation object and Temperaturedependency is
  * ✅ Janderequation and Ginstling-Brounshteinequation explanationcan
  * ✅ Temperatureprof 3necessary（heating rate・retentionhours・coolingspeed） importantproperty is

### real

  * ✅ Python diffusionnumber Temperaturedependency can
  * ✅ Janderequation forreactionprogress can
  * ✅ Kissingermethod DSC/TGfromactivationenergy calculationcan
  * ✅ DOE（experimentmethod） reactionconditions optimizationcan
  * ✅ pycalphad formutualfigurecalculation basis is

### applicationforce

  * ✅ newceramicsmaterial synthesisprocess designcan
  * ✅ experimentfromreactionmechanism estimation、appropriatespeedequation selectcan
  * ✅ industryprocessinconditionsoptimizationstrategy can
  * ✅ grain growthcontrol strategy（Two-step sinteringequal） can

## Exercises

### Easy（basisconfirm）

Q1: STLformat 

STL ASCIIformat and Binaryformatabout、positiveexplanation ？

a) ASCIIformat small  
b) Binaryformat interval format  
c) Binaryformat typicallyASCIIformat 5-10timessmall  
d) Binaryformat ASCIIformatthanaccuracy Low

answer display

**positive: c) Binaryformat typicallyASCIIformat 5-10timessmall**

**:**

  * **ASCII STL** : format interval 。eachangle 7（facet、normal、3point、endfacet） is done。（numberMB~numberMB）。
  * **Binary STL** : format type。80 + 4triangle count + eachangle50。sameshape ASCII 1/5~1/10 。
  * accuracy format and same（32-bitnumberpointnumber）
  * current 3Dprinter format support、Binaryrecommended

**realExample:** 10,000angle model → ASCII: approximately7MB、Binary: approximately0.5MB

Q2: Build Time calculation

volume12,000 mm³、High30 mm buildobject 、Layer Height0.2 mm、Print Speed50 mm/s builddo。 Build Time ？（infill20%、2layers and assumption）

a) 30minutes  
b) 60minutes  
c) 90minutes  
d) 120minutes

answer display

**positive: c) 90minutes（approximately1.5hours）**

**calculationorder:**

  1. **number** : High30mm ÷ Layer Height0.2mm = 150layers
  2. **1layers length estimation** : 
     * volume12,000mm³ → 1layers80mm³
     * （）: approximately200mm/layers（nozzlediameter0.4mm and assumption）
     * infill20%: approximately100mm/layers
     * : approximately300mm/layers
  3. **length** : 300mm/layers × 150layers = 45,000mm = 45m
  4. **hours** : 45,000mm ÷ 50mm/s = 900second = 15minutes
  5. **actual hours** : Movementhours・retraction・ do and approximately5-6times → 75-90minutes

**:** doestimationhours 、・Movement・Temperaturestable purpose、singlecalculation 4-6timesabout 。

Q3: AMprocess select

next application optimalAMprocess ：「emptymachinepart alloyproductionnozzle、complexinsidepart、HighStrength・Highpropertyrequirement」

a) FDM (Fused Deposition Modeling)  
b) SLA (Stereolithography)  
c) SLM (Selective Laser Melting)  
d) Binder Jetting

answer display

**positive: c) SLM (Selective Laser Melting / Powder Bed Fusion for Metal)**

**reason:**

  * **SLM Features** : metalpowder（、、） laser completemelting。Highdensity（99.9%）、HighStrength、Highproperty。
  * **applicationproperty** : 
    * ✓ alloy（Ti-6Al-4V）pair
    * ✓ complexinsidepartmanufacturingpossible（supportremovalafter）
    * ✓ aerospace machineproperty
    * ✓ GE Aviation actual FUELnozzle SLM quantityproduction
  * **other select notreason** : 
    * FDM: plastic 、Strength・propertynot
    * SLA: resin 、Functionpartfornot
    * Binder Jetting: metalpossible 、sinteringafterdensity90-95% aerospacestandard 

**realExample:** GE Aviation LEAPnozzle（SLMproduction） 、conventional20part ingthing 1part 、weight25%reduction、property5timesenhance 。

### Medium（application）

Q4: Python STLmesh validate

or less Python 、STL property（watertight） validateplease。
    
    
    import trimesh mesh = trimesh.load('model.stl') # add: property 、 # auto-repair 、repairafter mesh # 'model_fixed.stl'assaveplease 

answer display

**answerExample:**
    
    
    import trimesh mesh = trimesh.load('model.stl') # Check manifold property print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") # case repair if not mesh.is_watertight or not mesh.is_winding_consistent: print("meshrepair executeMedium...") # Fix normals trimesh.repair.fix_normals(mesh) # Fill holes trimesh.repair.fill_holes(mesh) # Remove degenerate faces mesh.remove_degenerate_faces() # Merge duplicate vertices mesh.merge_vertices() # repairresult confirm print(f"repairafter watertight: {mesh.is_watertight}") # Save repaired mesh if mesh.is_watertight: mesh.export('model_fixed.stl') print("repair: model_fixed.stl assave") else: print("⚠️ auto-repair。Meshmixerequal usageplease") else: print("✓ mesh 3Dringpossible") 

**:**

  * `trimesh.repair.fix_normals()`: methodline 
  * `trimesh.repair.fill_holes()`: mesh 
  * `remove_degenerate_faces()`: surfaceproduct angle delete
  * `merge_vertices()`: multiplepoint 

**real:** trimesh repair complex 、Meshmixer、Netfabb、MeshLabetc. fortool necessary。

Q5: supportmaterial volumecalculation

diameter40mm、High30mm 、surfacefrom45degree angledegree builddo。supportdensity15%、Layer Height0.2mm and assumption、 supportmaterialvolume estimationplease。

answer display

**answerprocess:**

  1. **support necessaryregion specific** : 
     * 45degree → surface approximatelyhalfminutes （45degreeor more ）
     * 45degree and 、piece 
  2. **supportregion whatcalculation** : 
     * surfaceproduct: π × (20mm)² ≈ 1,257 mm²
     * 45degreetime supportnecessarysurfaceproduct: approximately1,257mm² × 0.5 = 629 mm²
     * supportHigh: approximately 30mm × sin(45°) ≈ 21mm
     * supportvolume（density100% and assumption）: 629mm² × 21mm ÷ 2（angleshape）≈ 6,600 mm³
  3. **supportdensity15%** : 
     * actual supportmaterial: 6,600mm³ × 0.15 = **approximately990 mm³**
  4. **validate** : 
     * this volume: π × 20² × 30 ≈ 37,700 mm³
     * support/thisratio: 990 / 37,700 ≈ 2.6%（relevantrangescope）

**: approximately1,000 mm³ (990 mm³)**

**real:**

  * build optimization 、support Widthreductionpossible（ Examplein buildsupportunnecessary）
  * Tree Support usage、furthermore30-50%materialreductionpossible
  * propertysupport（PVA、HIPS） usage、removal easy

Q6: Layer Height optimization

High60mm buildobject 、quality and hours builddo。Layer Height0.1mm、0.2mm、0.3mm 3 select case、 Build Timeratio and recommendedapplication explanationplease。

answer display

**answer:**

Layer Height | number | hoursratio | quality | recommendedapplication  
---|---|---|---|---  
0.1 mm | 600layers | ×3.0 | nonalways High | forFigurines, medical models, end-use parts  
0.2 mm | 300layers | ×1.0（standard） |  | General prototypes, functional parts  
0.3 mm | 200layers | ×0.67 | Low | initialprototype、Strength insidepartpart  
  
**hoursratio calculationbasis:**

  * number 1/2 and 、ZMovementtimesnumber1/2
  * BUT: eachlayers hours （1layers volume purpose）
  * for、Layer Height 「oppositeratioExample」（for0.9-1.1times number）

**realselectstandard:**

  1. **0.1mmrecommended** : 
     * surfacequality （、）
     * surface important（、linear）
     * productlayers 
  2. **0.2mmrecommended** : 
     * quality and hours （general）
     * Functiontestforprototype
     * degreesurfaceabove minutes
  3. **0.3mmrecommended** : 
     * speed（shapeconfirm ）
     * insidepartstructuremanufacturingpart（outsidenot）
     * typebuildobject（hoursreductionEffect）

**numberLayer Height（Advanced）:**  
PrusaSlicerCura numberLayer HeightFunction 、part 0.3mm、surfacepart 0.1mm and 、quality and hours possible。

Q7: AMprocessselect 

aerospacefor quantity（alloy、optimizationcomplexshape、HighStrength・quantityrequirement） manufacturing optimalAMprocess select、 reason 3。also、afterprocess 2。

answer display

**optimalprocess: LPBF (Laser Powder Bed Fusion) - SLM for Aluminum**

**selectreason（3）:**

  1. **Highdensity・HighStrength** : 
     * lasercompletemelting thanmutualpairdensity99.5%or more 
     * manufacturing domachineproperty（Strength、property）
     * aerospace（AS9100、Nadcap）possible
  2. **optimizationshape manufacturingfunctionforce** : 
     * complexstructuremanufacturing（0.5mmor less） Highaccuracy build
     * Mediumemptystructuremanufacturing、shapeetc.conventionalnotpossibleshape pair
     * supportremovalafter、insidepartstructuremanufacturingpossible
  3. **materialefficiency and quantity** : 
     * Buy-to-Flyratio（materialquantity/end-use productweight） 1/10~1/20
     * optimization conventionaldesignratio40-60%quantity
     * alloy（AlSi10Mg、Scalmalloy） ratioStrength

**necessaryafterprocess（2）:**

  1. **process（Heat Treatment）** : 
     * forceremoval（Stress Relief Annealing）: 300°C、2-4hours
     * : buildtime force removal、dimensionstablepropertyenhance
     * Effect: 30-50%enhance、oppositedeformation
  2. **surfaceprocess（Surface Finishing）** : 
     * machine（CNC）: surface、 Highaccuracy（Ra < 3.2μm）
     * （Electropolishing）: surfaceLow（Ra 10μm → 2μm）
     * （Shot Peening）: surfacelayers force 、propertyenhance
     * process: propertyenhance、property（aerospaceStandard）

**addsubsection:**

  * **build** : and productlayers （ZStrength 10-15%Low）
  * **supportdesign** : removalTree Support、surfaceproduct
  * **quality** : CT insidepartdefectinspection、Xlineinspection
  * ****: powder、buildParameter

**realExample: Airbus A350**  
conventional32part seting 1part 、weight55%reduction、65%、cost35%reduction 。

3 × 3 = **9times** （） 

**DOE point（conventionalmethod and comparison）:**

  1. **mutualfor possible**
     * conventionalmethod: Temperature effect、hours effect individual evaluation
     * DOE: 「Hightemperatureinhours can」 and mutualfor quantity
     * Example: 1300°Cin4hours minutes 、1100°Cin8hoursnecessary、etc.
  2. **experimenttimesnumber reduction**
     * conventionalmethod（OFAT: One Factor At a Time）: 
       * Temperature: 3times（hours）
       * hours: 3times（Temperature）
       * confirmexperiment: multiplenumbertimes
       * : 10timesor more
     * DOE: 9times （allconditions＋mutualfor）
     * furthermoreMediummultiplemethod 7times reductionpossible

**add point:**

  * exist can be（differenceevaluation possible）
  * surface structure 、not yetrealconditions possible
  * optimalconditions experimentrangescopeoutside case can

### Hard（）

Q7: complexreactionsystem design

next conditions Li₁.₂Ni₀.₂Mn₀.₆O₂（positivematerial） synthesisdoTemperatureprof designplease：

  * : Li₂CO₃, NiO, Mn₂O₃
  * : singlemutual、diameter < 5 μm、Li/metalratio precisecontrol
  * approximately: 900°Cor more Li₂O （Li ）

Temperatureprof（heating rate、retentionTemperature・hours、coolingspeed） and 、 designreason explanationplease。

answer 

**recommendedTemperatureprof:**

**Phase 1: heating（Li₂CO₃minutes）**

  * temperature → 500°C: 3°C/min
  * 500°Cretention: 2hours
  * **reason:** Li₂CO₃ minutes（~450°C） progress、CO₂ complete removal

**Phase 2: Mediumintervalheating（before）**

  * 500°C → 750°C: 5°C/min
  * 750°Cretention: 4hours
  * **reason:** Li₂MnO₃LiNiO₂etc. Mediumintervalmutual 。Li LessTemperature quality

**Phase 3: this（mutualsynthesis）**

  * 750°C → 850°C: 2°C/min（）
  * 850°Cretention: 12hours
  * **reason:**
    * Li₁.₂Ni₀.₂Mn₀.₆O₂ singlemutualforlengthhoursnecessary
    * 850°C limitLi （<900°Capproximately）
    * lengthhoursretention diffusion 、grain growth is doneTemperature

**Phase 4: cooling**

  * 850°C → temperature: 2°C/min
  * **reason:** thanpropertyenhance、forceby

**design important:**

  1. **Lipair:**
     * 900°Cor less limit（this approximately）
     * furthermore、Lipast（Li/TM = 1.25etc.） usage
     * Medium Li₂O minutes Low
  2. **diametercontrol ( < 5 μm):**
     * Lowtemperature（850°C）・lengthhours（12h） reaction 
     * Hightemperature・hours and grain growth past 
     * diameter1μmor less 
  3. **setuniformproperty:**
     * 750°CinMediumintervalretention important
     * metal minutes quality
     * necessary 、750°Cretentionafter degreecooling→→againheating

**allplacenecessaryhours:** approximately30hours（heating12h + retention18h）

**method :**

  * **Sol-gelmethod:** thanLowtemperature（600-700°C） synthesispossible、qualitypropertyenhance
  * **Spray pyrolysis:** diametercontrol easy
  * **Two-step sintering:** 900°C 1h → 800°C 10h grain growth

Q8: speed 

or less from、reactionmechanism estimation、activationenergy calculationplease。

**experiment:**

Temperature (°C) | 50%reactionreachinghours t₅₀ (hours)  
---|---  
1000| 18.5  
1100| 6.2  
1200| 2.5  
1300| 1.2  
  
Janderequation assumptioncase: [1-(1-0.5)^(1/3)]² = k·t₅₀

answer 

**answer:**

**1: speedconstantk calculation**

Janderequation α=0.5 when:  
[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424

therefore k = 0.0424 / t₅₀

T (°C) | T (K) | t₅₀ (h) | k (h⁻¹) | ln(k) | 1000/T (K⁻¹)  
---|---|---|---|---|---  
1000| 1273| 18.5| 0.00229| -6.080| 0.7855  
1100| 1373| 6.2| 0.00684| -4.985| 0.7284  
1200| 1473| 2.5| 0.01696| -4.077| 0.6788  
1300| 1573| 1.2| 0.03533| -3.343| 0.6357  
  
**2: Arrheniusplot**

ln(k) vs 1/T plot（lineartimes）

linear: ln(k) = A - Eₐ/(R·T)

= -Eₐ/R

lineartimescalculation:  
slope = Δ(ln k) / Δ(1000/T)  
= (-3.343 - (-6.080)) / (0.6357 - 0.7855)  
= 2.737 / (-0.1498)  
= -18.27

**3: activationenergycalculation**

slope = -Eₐ / (R × 1000)  
Eₐ = -slope × R × 1000  
Eₐ = 18.27 × 8.314 × 1000  
Eₐ = 151,899 J/mol ≈ **152 kJ/mol**

**4: reactionmechanism**

  * **activationenergy comparison:**
    * value: 152 kJ/mol
    * typemutualdiffusion: 200-400 kJ/mol
    * interface reaction: 50-150 kJ/mol
  * **estimationis donemechanism:**
    * value interface reaction and diffusion Mediuminterval
    * possibleproperty1: interface reaction rate-limiting（diffusion effect ）
    * possibleproperty2: diffusiondistance 、 Eₐ Low
    * possibleproperty3: rate-limiting（interface reaction and diffusion ）

**5: validatemethod**

  1. **dependency:** differentdiameter experiment、k ∝ 1/r₀² doconfirm 
     * → diffusionrate-limiting
     * not → interface reactionrate-limiting
  2. **other speedequationinfitting:**
     * Ginstling-Brounshteinequation（3nextorigindiffusion）
     * Contracting sphere model（interface reaction）
     * R² Highcomparison
  3. **structuremanufacturing:** SEM reactionboundarysurface 
     * thickgenerateobjectlayers → diffusionrate-limiting basis
     * thingenerateobjectlayers → interface reactionrate-limiting possibleproperty

**final:**  
activationenergy **Eₐ = 152 kJ/mol**  
estimationmechanism: **interface reactionrate-limiting、also systemindiffusionrate-limiting**  
addexperiment recommendedis done。

## next 

Chapter 2inproductlayersbuild（AM） basisas、ISO/ASTM 52900by7 processCategory、STLformat structuremanufacturing、slicing and G-code basisthis 。next Chapter 2in、material（FDM/FFF） detailedbuildprocess、materialproperty、processParameteroptimizationabout。

[← table of contents](<./index.html>) [Chapter 2 →](<chapter-2.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2015). _Additive Manufacturing Technologies: 3D Printing, Rapid Prototyping, and Direct Digital Manufacturing_ (2nd ed.). Springer. pp. 1-35, 89-145, 287-334. - AMtechnology comprehensive、7 process and STLprocess detailed
  2. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. International Organization for Standardization. - AMfor and processCategory whenStandardstandard、industryboundary is done
  3. Kruth, J.P., Leu, M.C., & Nakagawa, T. (1998). "Progress in Additive Manufacturing and Rapid Prototyping." _CIRP Annals - Manufacturing Technology_ , 47(2), 525-540. - selectlasersintering and mechanism basis
  4. Hull, C.W. (1986). _Apparatus for production of three-dimensional objects by stereolithography_. US Patent 4,575,330. - boundaryinitial AMtechnology（SLA） special、AMindustry and importantliterature
  5. Wohlers, T. (2023). _Wohlers Report 2023: 3D Printing and Additive Manufacturing Global State of the Industry_. Wohlers Associates, Inc. pp. 15-89, 156-234. - AMmarket and industryapplication latest、yearnextupdateis doneboundaryStandard
  6. 3D Systems, Inc. (1988). _StereoLithography Interface Specification_. - STLformat equationspecification、ASCII/Binary STLstructuremanufacturing 
  7. numpy-stl Documentation. (2024). _Python library for working with STL files_. <https://numpy-stl.readthedocs.io/> \- STLload・volumecalculationfor Pythonlibrary
  8. trimesh Documentation. (2024). _Python library for loading and using triangular meshes_. <https://trimsh.org/> \- meshrepair・・qualityevaluation comprehensivelibrary

## usagetool and library

  * **NumPy** (v1.24+): numbervaluecalculationlibrary - <https://numpy.org/>
  * **numpy-stl** (v3.0+): STLprocesslibrary - <https://numpy-stl.readthedocs.io/>
  * **trimesh** (v4.0+): 3Dmeshprocesslibrary（repair、validate、） - <https://trimsh.org/>
  * **Matplotlib** (v3.7+): possiblelibrary - <https://matplotlib.org/>
  * **SciPy** (v1.10+): technologycalculationlibrary（optimization、interval） - <https://scipy.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
