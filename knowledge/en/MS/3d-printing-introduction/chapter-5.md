---
title: "Chapter 5: Fundamentals of Additive Manufacturing"
chapter_title: "Chapter 5: Fundamentals of Additive Manufacturing"
subtitle: Principles and Classification of AM Technologies - 3D Printing Technology Systems
reading_time: 35-40 minutes
difficulty: Beginner to Intermediate
---

[AI Terakoya Home](<../index.html>) › [Materials Science](<../../index.html>) › [Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>) › Chapter 5

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-5.html>) | Last sync: 2025-11-16

## Learning Objectives

Upon completing this chapter, you will be able to explain:

### Basic Understanding (Level 1)

  * Definition of Additive Manufacturing (AM) and basic concepts of ISO/ASTM 52900 standard
  * Characteristics of seven AM process categories (MEX, VPP, PBF, MJ, BJ, SL, DED)
  * Structure of STL file format (triangle mesh, normal vectors, vertex order)
  * History of AM (from 1986 stereolithography to modern systems)

### Practical Skills (Level 2)

  * Load STL files in Python and calculate volume and surface area
  * Perform mesh validation and repair using numpy-stl and trimesh
  * Understand basic slicing principles (layer height, shell, infill)
  * Interpret basic G-code structure (G0/G1/G28/M104, etc.)

### Applied Competency (Level 3)

  * Select optimal AM process according to application requirements
  * Detect and fix mesh problems (non-manifold, inverted normals)
  * Optimize build parameters (layer height, print speed, temperature)
  * Evaluate STL file quality and assess printability

## 1.1 What is Additive Manufacturing (AM)

### 1.1.1 Definition of Additive Manufacturing

Additive Manufacturing (AM) is**defined by the ISO/ASTM 52900:2021 standard as "a process of joining materials to make objects from 3D CAD data, usually layer upon layer"**.In contrast to conventional subtractive machining, material is added only where needed, providing these innovative features:

  * **Design freedom** : Can manufacture complex geometries impossible with conventional methods (hollow structures, lattice structures, topology-optimized shapes)
  * **Material efficiency** : Material waste rate 5-10% as material is used only where needed (conventional machining wastes 30-90%)
  * **On-demand manufacturing** : Can produce customized products in low volume, high variety without molds
  * **Integrated manufacturing** : Consolidate structures previously assembled from multiple parts into single build, reducing assembly steps

**💡 Industrial Significance**

The AM market is growing rapidly. According to Wohlers Report 2023:

  * Global AM market size: $18.3B (2023) → $83.9B forecast (2030, 23.5% CAGR)
  * Application breakdown: Prototyping (38%), Tooling (27%), End-use parts (35%)
  * Major industries: Aerospace (26%), Medical (21%), Automotive (18%), Consumer goods (15%)
  * Material share: Polymers (55%), Metals (35%), Ceramics (7%), Others (3%)

### 1.1.2 History and Development of AM

Additive manufacturing technology has approximately 40 years of history, reaching the present through these milestones:
    
    
    flowchart LR A[1986  
    SLAInvention  
    Chuck Hull] -->B[1988  
    SLSEmergence  
    Carl Deckard] B -->C[1992  
    FDMPatent  
    Stratasys] C -->D[2005  
    RepRap  
    Open Source] D -->E[2012  
    MetalAMAdoption  
    EBM/SLM] E -->F[2023  
    industryaddfast  
    Large-scale・Highfast] style A fill:#e3f2fd style B fill:#fff3e0 style C fill:#e8f5e9 style D fill:#f3e5f5 style E fill:#fce4ec style F fill:#fff9c4

  1. **1986: Invention of Stereolithography (SLA)** \- Dr. Chuck Hull (founder of 3D Systems) invented the first AM technology to cure photopolymer resin in layers (US Patent 4,575,330). The term "3D printing" was also coined at this time.
  2. **1988: Emergence of Selective Laser Sintering (SLS)** \- Dr. Carl Deckard (University of Texas) developed technology to sinter powder materials with laser, opening possibilities for metal and ceramic applications.
  3. **1992: Fused Deposition Modeling (FDM) Patent** \- Stratasys commercialized FDM technology, establishing the foundation for the currently most widespread 3D printing method.
  4. **2005: RepRap Project** \- Professor Adrian Bowyer announced the open source 3D printer "RepRap". Combined with patent expiration, this led to cost reduction and democratization.
  5. **2012 onwards: Industrial Adoption of Metal AM** \- Electron Beam Melting (EBM) and Selective Laser Melting (SLM) commercialized in aerospace and medical fields. GE Aviation started mass production of FUEL injection nozzles.
  6. **2023 Present: Era of Larger Size and Higher Speed** \- New technologies such as binder jetting, continuous fiber composite AM, and multi-material AM entering industrial implementation stage.

### 1.1.3 Major Application Fields of AM

#### Application 1: Prototyping (Rapid Prototyping)

AM's first major application, rapidly manufacturing prototypes for design validation, functional testing, and market evaluation:

  * **Lead time reduction** : Conventional prototyping (weeks to months) → AM in hours to days
  * **Accelerated design iteration** : Prototype multiple versions at low cost to optimize design
  * **Improved communication** : Unify stakeholder understanding with visual and tactile physical models
  * **Typical examples** : Automotive styling models, consumer electronics housing prototypes, medical device pre-surgical simulation models

#### Application 2: Tooling (Tooling & Fixtures)

Application of manufacturing jigs, tools, and molds for production using AM:

  * **Custom jigs** : Rapidly produce assembly jigs and inspection fixtures specialized for production lines
  * **Conformal cooling molds** : Injection molds with 3D cooling channels following product shape instead of conventional straight channels (30-70% cooling time reduction)
  * **Lightweight tools** : Reduce worker burden with lightweight end effectors using lattice structures
  * **Typical examples** : BMW assembly line fixtures (over 100,000 units per year manufactured with AM), TaylorMade Golf driver molds

#### Application 3: End-Use Parts

Applications of direct manufacturing of end-use parts with AM are rapidly increasing in recent years:

  * **Aerospace components** : GE Aviation LEAP fuel injection nozzle (conventional 20 parts → AM consolidated, 25% weight reduction, over 100,000 units per year production)
  * **Medical implants** : Titanium artificial hip joints and dental implants (optimized to patient-specific anatomical shape, porous structure promotes bone integration)
  * **Custom products** : Hearing aids (over 10 million units per year manufactured with AM), sports shoe midsoles (Adidas 4D, Carbon DLS technology)
  * **Spare parts** : On-demand manufacturing of discontinued and rare parts (automotive, aircraft, industrial machinery)

**⚠️ AM Limitations and Challenges**

AM is not a panacea and has the following constraints:

  * **Build speed** : Unsuitable for mass production (injection molding 1 part/seconds vs AM hours). Economic break-even typically below 1,000 units
  * **Build size limitation** : Large parts exceeding build volume (typically around 200×200×200mm for many machines) require split manufacturing
  * **Surface quality** : Layer lines remain, so post-processing (polishing, machining) required when high-precision surface needed
  * **Material anisotropy** : Mechanical properties may differ between build direction (Z-axis) and in-plane direction (XY-plane), especially for FDM
  * **Material cost** : AM-grade materials 2-10 times more expensive than general purpose materials (but can be offset by material efficiency and design optimization)

## 1.2 Seven AM Process Categories by ISO/ASTM 52900

### 1.2.1 Overview of AM Process Classification

The ISO/ASTM 52900:2021 standard classifies all AM technologies into**seven process categories based on energy source and material delivery method**. Each process has unique advantages and disadvantages, requiring selection of optimal technology according to application.
    
    
    flowchart TD AM[Additive Manufacturing  
    7 Processes] -->MEX[Material Extrusion  
    Material Extrusion] AM -->VPP[Vat Photopolymerization  
    Vat Photopolymerization] AM -->PBF[Powder Bed Fusion  
    Powder Bed Fusion] AM -->MJ[Material Jetting  
    Material Jetting] AM -->BJ[Binder Jetting  
    Binder Jetting] AM -->SL[Sheet Lamination  
    Sheet Lamination] AM -->DED[Directed Energy Deposition  
    Directed Energy Deposition] MEX -->MEX_EX[FDM/FFF  
    Lowcost・Adoptiontype] VPP -->VPP_EX[SLA/DLP  
    Highprecision・Highsurfacequality] PBF -->PBF_EX[SLS/SLM/EBM  
    Highstrength・metalcompatible] style AM fill:#f093fb style MEX fill:#e3f2fd style VPP fill:#fff3e0 style PBF fill:#e8f5e9 style MJ fill:#f3e5f5 style BJ fill:#fce4ec style SL fill:#fff9c4 style DED fill:#fce4ec

### 1.2.2 Material Extrusion (MEX)

**Principle** : Thermoplastic resin filament is heated and melted, extruded through a nozzle and layered. Most widespread technology (also called FDM/FFF).

Process: Filament → Heated nozzle (190-260°C) → Melt extrusion → Cooling solidification → Next layer

**Characteristics:**

  * **Low cost** : Equipment price $200-$5,000 (desktop), $10,000-$100,000 (industrial)
  * **Material diversity** : PLA, ABS, PETG, nylon, PC, carbon fiber composites, PEEK (high performance)
  * **Build speed** : 20-150 mm³/s (medium), layer height 0.1-0.4mm
  * **Accuracy** : ±0.2-0.5 mm (desktop), ±0.1 mm (industrial)
  * **Surface quality** : Layer lines clearly visible (can be improved with post-processing)
  * **Material anisotropy** : Z-axis direction (build direction) strength 20-80% lower (layer adhesion is weakness)

**Application examples:**

  * Prototyping (most common application, low cost, high speed)
  * Jigs and tools (used in manufacturing sites, lightweight, easy to customize)
  * Educational models (widely used in schools and universities, safe, low cost)
  * End-use parts (custom hearing aids, prosthetics, architectural models)

**💡 FDM Representative Equipment**

  * **Ultimaker S5** : Dual head, build volume330×240×300mm、$6,000
  * **Prusa i3 MK4** : Open source system, high reliability、$1,200
  * **Stratasys Fortus 450mc** : Industrial, ULTEM 9085compatible、$250,000
  * **Markforged X7** : Continuous carbon fiber composite compatible、$100,000

### 1.2.3 Vat Photopolymerization (VPP)

**Principle** : Liquid photopolymer resin is selectively cured and layered by exposing to ultraviolet (UV) laser or projector light.

Process: UV exposure → Photopolymerization → Solidification → Build platform rise → Next layer exposure

**Two main VPP methods:**

  1. **SLA (Stereolithography)** : UV laser (355 nm) scanned with galvanometer mirrors, point-by-point curing. High precision but slow.
  2. **DLP (Digital Light Processing)** : Entire surface exposed at once with projector. Fast but resolution depends on projector pixels (Full HD: 1920×1080).
  3. **LCD-MSLA (Masked SLA)** : Uses LCD mask, similar to DLP but lower cost (many desktop machines $200-$1,000).

**Characteristics:**

  * **High precision** : XY resolution 25-100 μm, Z resolution 10-50 μm (highest level among all AM technologies)
  * **Surface quality** : Smooth surface (Ra< 5 μm), layer lines barely visible
  * **Build speed** : SLA (10-50 mm³/s), DLP/LCD (100-500 mm³/s, area dependent)
  * **Material constraints** : Photopolymer resins only (mechanical properties often inferior to FDM)
  * **Post-processing required** : Washing (IPA etc.) → Post-curing (UV exposure) → Support removal

**Application examples:**

  * Dental applications (orthodontic models, surgical guides, dentures, millions produced annually)
  * Jewelry casting wax models (high precision, complex shapes)
  * Medical models (pre-surgical planning, anatomical models, patient education)
  * Master models (for silicone molding, design verification)

### 1.2.4 Powder Bed Fusion (PBF)

**Principle** : Powder material is spread thinly, selectively melted/sintered with laser or electron beam, cooled and solidified in layers. Compatible with metals, polymers, and ceramics.

Process: Powder spreading → Laser/electron beam scanning → Melting/sintering → Solidification → Next layer powder spreading

**Three main PBF methods:**

  1. **SLS (Selective Laser Sintering)** : Laser sintering of polymer powder (PA12 nylon etc.). No support needed (surrounding powder provides support).
  2. **SLM (Selective Laser Melting)** : Complete melting of metal powder (Ti-6Al-4V, AlSi10Mg, Inconel 718 etc.). Can manufacture high-density parts (relative density >99%).
  3. **EBM (Electron Beam Melting)** : Electron beam melting of metal powder. High temperature preheat (650-1000°C) results in low residual stress and fast build speed.

**Characteristics:**

  * **High strength** : Melting and resolidification provides mechanical properties comparable to forged materials (tensile strength 500-1200 MPa)
  * **Complex geometry capability** : Can build overhangs without support (powder provides support)
  * **Material diversity** : Ti alloys, Al alloys, stainless steel, Ni superalloys, Co-Cr alloys, nylon
  * **High cost** : Equipment price $200,000-$1,500,000, material cost $50-$500/kg
  * **Post-processing** : Support removal, heat treatment (stress relief), surface finishing (blasting, polishing)

**Application examples:**

  * Aerospace components (weight reduction, consolidation, GE LEAP fuel nozzle etc.)
  * Medical implants (patient-specific shapes, porous structure, Ti-6Al-4V)
  * Molds (conformal cooling, complex shapes, H13 tool steel)
  * Automotive parts (lightweight brackets, custom engine components)

### 1.2.5 Material Jetting (MJ)

**Principle** : Similar to inkjet printers, droplet material (photopolymer resin or wax) is jetted from head and immediately cured with UV exposure and layered.

**Characteristics:**

  * **Ultra-high precision** : XY resolution 42-85 μm, Z resolution 16-32 μm
  * **Multi-material** : Can use multiple materials and colors in same build
  * **Full color** : Over 10 million colors with CMYK resin combinations
  * **Surface quality** : Extremely smooth (almost no layer lines)
  * **High cost** : Equipment $50,000-$300,000, material cost $200-$600/kg
  * **Material constraints** : Photopolymer resins only, moderate mechanical properties

**Applicationexamples：** : Medical anatomical models (reproduce soft/hard tissue with different materials), full-color architectural models, design verification models

### 1.2.6 Binder Jetting (BJ)

**Principle** : Liquid binder (adhesive) is jetted inkjet-style onto powder bed to bind powder particles. After build, strength is enhanced by sintering or infiltration.

**Characteristics:**

  * **High-speed build** : No laser scanning needed, entire surface processed at once, build speed 100-500 mm³/s
  * **Material diversity** : Metal powder, ceramics, sand molds (for casting), full color (gypsum)
  * **No support needed** : Surrounding powder provides support, recyclable after removal
  * **Low density issue** : Fragile before sintering (green density 50-60%), even after sintering relative density 90-98%
  * **Post-processing required** : Debinding → Sintering (metal: 1200-1400°C) → Infiltration (copper, bronze)

**Applicationexamples：** : Sand casting molds (large castings such as engine blocks), metal parts (Desktop Metal, HP Metal Jet), full-color objects (souvenirs, educational models)

### 1.2.7 Sheet Lamination (SL)

**Principle** : Stack sheet materials (paper, metal foil, plastic film) and bond by adhesion or welding. Cut contour of each layer with laser or blade.

**Representative technologies:**

  * **LOM (Laminated Object Manufacturing)** : Paper/plastic sheets, laminated with adhesive, laser cutting
  * **UAM (Ultrasonic Additive Manufacturing)** : Metal foil ultrasonic welding, CNC milling for contour

**Features：** Large build size possible, low material cost, medium accuracy, limited applications (mainly visual models, embedded sensors for metal)

### 1.2.8 Directed Energy Deposition (DED)

**Principle** : Metal powder or wire is fed while melting with laser/electron beam/arc and deposited on substrate. Used for large parts and repair of existing parts.

**Characteristics:**

  * **High-speed deposition** : Deposition rate 1-5 kg/h (10-50 times PBF)
  * **Large scale capability** : Less build volume limitation (using multi-axis robot arm)
  * **Repair and coating** : Repair worn parts, form surface hardened layer
  * **Low precision** : Accuracy ±0.5-2 mm, post-processing (machining) required

**Applicationexamples：** : Turbine blade repair, large aerospace parts, wear-resistant coating of tools

**⚠️ Guidelines for Process Selection**

Optimal AM process varies by application requirements:

  * **Precision priority** → VPP (SLA/DLP) or MJ
  * **Low cost, widespread type** → MEX (FDM/FFF)
  * **Metal high-strength parts** → PBF (SLM/EBM)
  * **Mass production (sand molds)** → BJ
  * **Large scale, high-speed deposition** → DED

## 1.3 STL File Format and Data Processing

### 1.3.1 Structure of STL Files

STL (STereoLithography) is**the most widely used 3D model file format in AM** , developed by 3D Systems in 1987.STL files represent object surfaces as**a collection of triangle meshes**.

#### Basic Structure of STL Files

STL file = Normal vector (n) + 3 vertex coordinates (v1, v2, v3) × number of triangles

**ASCII STL format example:**
    
    
    solid cube facet normal 0 0 1 outer loop vertex 0 0 10 vertex 10 0 10 vertex 10 10 10 endloop endfacet facet normal 0 0 1 outer loop vertex 0 0 10 vertex 10 10 10 vertex 0 10 10 endloop endfacet ... endsolid cube
    

**Two types of STL format:**

  1. **ASCII STL** : Human-readable text format. Large file size (10-20 times binary for same model). Useful for debugging and verification.
  2. **Binary STL** : Binary format, small file size, fast processing. Standard in industrial applications. Structure: 80-byte header + 4 bytes (triangle count) + 50 bytes per triangle (normal 12B + vertices 36B + attribute 2B).

### 1.3.2 Key Concepts of STL Files

#### 1\. Normal Vector

Each triangle face has a**normal vector (outward direction)** defined to distinguish object "inside" and "outside". Normal direction is determined by**right-hand rule** :

Normal n = (v2 - v1) × (v3 - v1) / |(v2 - v1) × (v3 - v1)|

**Vertex order rule:** Vertices v1, v2, v3 are arranged counter-clockwise (CCW), and when viewed from outside, the counter-clockwise order makes the normal point outward.

#### 2\. Manifold Conditions

For STL mesh to be 3D printable, it must be**manifold** :

  * **Edge sharing** : Every edge is shared by exactly two triangles
  * **Vertex sharing** : Every vertex belongs to a continuous triangle fan
  * **Closed surface** : Forms a completely closed surface without holes or openings
  * **No self-intersection** : Triangles do not intersect or penetrate each other

**⚠️ Non-Manifold Mesh Issues**

Non-manifold meshes are not 3D printable. Typical problems:

  * **Holes** : Open surface, edge belongs to only one triangle
  * **T-junction** : Edge shared by three or more triangles
  * **Inverted Normals** : Mixed triangles with normals pointing inward
  * **Duplicate Vertices** : Multiple vertices at the same position
  * **Degenerate Triangles** : Triangles with zero or near-zero area

These issues cause errors in slicer software and lead to build failures.

### 1.3.3 Quality Metrics for STL Files

STL mesh quality is evaluated by the following metrics:

  1. **Triangle Count** : Typically 10,000-500,000. Avoid too few (coarse model) or too many (large file size, processing delay).
  2. **Edge Length Uniformity** : Mixed extremely large and small triangles degrade build quality. Ideally 0.1-1.0 mm range.
  3. **Aspect Ratio** : Elongated triangles (high aspect ratio) cause numerical errors. Ideally aspect ratio< 10.
  4. **Normal Consistency** : All normals unified outward. Mixed inverted normals cause inside/outside determination errors.

**💡 STL File Resolution Trade-offs**

STL mesh resolution (triangle count) is a trade-off between accuracy and file size:

  * **Low resolution (1,000-10,000 triangles)** : Fast processing, small file, but curved surfaces appear faceted
  * **Medium resolution (10,000-100,000 triangles)** : Suitable for many applications, good balance
  * **High resolution (100,000-1,000,000 triangles)** : Smooth curved surfaces, but large file size (tens of MB), processing delay

When exporting STL from CAD software, control resolution with**Chordal Tolerance** or**Angle Tolerance**. Recommended values: chordal tolerance 0.01-0.1 mm, angular tolerance 5-15 degrees.

### 1.3.4 STL Processing with Python Libraries

Major libraries for handling STL files in Python:

  1. **numpy-stl** : Fast STL read/write, volume/surface area calculation, normal vector manipulation. Simple and lightweight.
  2. **trimesh** : Comprehensive 3D mesh processing library. Mesh repair, Boolean operations, ray casting, collision detection. Feature-rich but many dependencies.
  3. **PyMesh** : Advanced mesh processing (remeshing, subdivision, feature extraction). Installation somewhat complex.

**Basic usage of numpy-stl:**
    
    
    from stl import mesh import numpy as np # Load STL file your_mesh = mesh.Mesh.from_file('model.stl') # Basic geometric information volume, cog, inertia = your_mesh.get_mass_properties() print(f"Volume: {volume:.2f} mm³") print(f"Center of Gravity: {cog}") print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²") # Number of triangles print(f"Number of Triangles: {len(your_mesh.vectors)}")
    

## 1.4 Slicing and Toolpath Generation

The process of converting STL files into commands (G-code) that 3D printers can understand is called**slicing**. This section covers the basic principles of slicing, toolpath strategies, and G-code fundamentals.

### 1.4.1 Basic Principles of Slicing

Slicing is the process of horizontally cutting a 3D model at constant height (layer height) and extracting the contour of each layer:
    
    
    flowchart TD A[3D Model  
    STL File] -->B[Slice in layers  
    along Z-axis] B -->C[Extract contour of each layer  
    Contour Detection] C -->D[Generate shell  
    Perimeter Path] D -->E[Generate infill  
    Infill Path] E -->F[Add support  
    Support Structure] F -->G[Optimal toolpath  
    Retraction/Travel] G -->H[G-codeoutput] style A fill:#e3f2fd style H fill:#e8f5e9

#### Layer Height Selection

Layer height is the most important parameter determining the trade-off between build quality and build time:

Layer Height | Build Quality | Build Time | Typical Applications  
---|---|---|---  
0.1 mm (Ultra-fine) | Very high (layer lines almost invisible) | Very long (×2-3 times) | Figurines, medical models, end-use parts  
0.2 mm (Standard) | Good (layer lines visible but acceptable) | Standard | General prototypes, functional parts  
0.3 mm (Coarse) | Low (layer lines clearly visible) | Short (×0.5 times) | Initial prototypes, internal structure parts  
  
**⚠️ Layer Height Constraints**

Layer Height must be set to **25-80%** of nozzle diameter。For example, with a 0.4mm nozzle, Layer Height recommended range is 0.1-0.32mm。Exceeding this can result in insufficient resin extrusion and the nozzle dragging the previous layer。

### 1.4.2 Shell and Infill Strategies

#### Shell (Perimeter) Generation

**Shell (Perimeter)** is the path forming the outer perimeter of each layer:

  * **Perimeter Count** : Typically 2-4. Affects external quality and strength.
    * 1: Very weak, high transparency, decorative only
    * 2 lines: Standard (good balance)
    * 3-4: High strength, improved surface quality, improved airtightness
  * **Shell order** : Inside-out is common. Outside-in used when surface quality is priority.

#### Infill Pattern

**Infill** forms internal structure and controls strength and material usage:

Pattern | Strength | Print Speed | Material Usage | Features  
---|---|---|---|---  
Grid | Medium | Fast | Medium | Simple, square property, standard selection  
Honeycomb | High | Slow | Medium | High strength, excellent weight ratio, aerospace applications  
Gyroid | Very High | Medium | Medium | 3D isotropic, curved surfaces, latest recommendation  
Concentric | Low | Fast | few | Flexibility focused, follows shell  
Lines | Low（differentsquareproperty） | Very Fast | few | Highfastprinting、directionpropertyStrength  
  
**💡 Infill Density Guidelines**

  * **0-10%** : Decorative items, non-load bearing parts (material saving priority)
  * **20%** : Standard prototype (good balance)
  * **40-60%** : Functionparts、Highstrengthrequirement
  * **100%** : Final products, watertight requirement, highest strength (Build Time ×3-5 times)

### 1.4.3 Generation of Support Structures

Parts with overhang angle exceeding 45 degrees require**support structures** :

#### Support Types

  * **Linear Support (straight support)** : Vertical pillar-like support. Simple and easy to remove, but uses more material。
  * **Tree Support (tree support)** : Tree-like branching support. Material usage reduced by 30-50%, easy to remove. Standard in Cura and PrusaSlicer。
  * **Interface Layers** : Thin interface layer on support top. Easy to remove, improves surface quality. Typically 2-4 layers.

#### Important Support Parameters

Parameter | Recommended Value | Effect  
---|---|---  
Overhang Angle | 45-60° | Generate support above this angle  
Support Density | 10-20% | Higher density is more stable but difficult to remove  
Support Z Distance | 0.2-0.3 mm | Gap between support and part (ease of removal)  
Interface Layers | 2-4layer | Interface layers (balance between surface quality and removability)  
  
### 1.4.4 Fundamentals of G-code

**G-code** is the standard numerical control language for controlling 3D printers and CNC machines. Each line represents one command:

#### Major G-code Commands

Command | Category | Function | Example  
---|---|---|---  
G0 | Movement | Rapid movement (non-extrusion) | G0 X100 Y50 Z10 F6000  
G1 | Movement | Linear movement (with extrusion) | G1 X120 Y60 E0.5 F1200  
G28 | Initialize | Return to home position | G28 （all axes), G28 Z (Z-axis only）  
M104 | Temperature | Nozzle temperature setting (non-blocking) | M104 S200  
M109 | Temperature | Nozzle temperature setting (blocking) | M109 S210  
M140 | Temperature | Bed temperature setting (non-blocking) | M140 S60  
M190 | Temperature | Bed temperature setting (blocking) | M190 S60  
  
#### G-codeExample（Build startpart minutes）
    
    
    ; === Start G-code === M140 S60 ; Start heating bed to 60°C (non-blocking) M104 S210 ; Start heating nozzle to 210°C (non-blocking) G28 ; Home all axes G29 ; Auto-leveling (bed mesh measurement) M190 S60 ; Wait for bed temperature to reach target M109 S210 ; Wait for nozzle temperature to reach target G92 E0 ; Reset extrusion to zero G1 Z2.0 F3000 ; Raise Z-axis 2mm (safety) G1 X10 Y10 F5000 ; Move to prime position G1 Z0.3 F3000 ; Lower Z-axis to 0.3mm (first layer height) G1 X100 E10 F1500 ; Draw prime line (clear nozzle clog) G92 E0 ; Reset extrusion to zero again ; === Build start ===
    

### 1.4.5 Major Slicing Software

Software | License | Features | Recommended Use  
---|---|---|---  
Cura | Open Source | Easy to use, rich presets, Tree Support standard | Beginner to intermediate, general FDM  
PrusaSlicer | Open Source | Advanced settings, variable layer height, custom support | Intermediate to advanced, optimization focused  
Slic3r | Open Source | Original of PrusaSlicer, lightweight | Legacy systems, research applications  
Simplify3D | Commercial ($150) | High-speed slicing, multi-process, detailed control | Professional, industrial applications  
IdeaMaker | Free | Raise3D specific but high versatility, intuitive UI | Raise3D users, beginners  
  
### 1.4.6 Toolpath Optimization Strategies

Efficient toolpath improves build time, quality, and material usage:

  * **Retraction** : Retracting filament during movement to prevent stringing。
    * Distance: 1-6mm (Bowden tube 4-6mm, direct 1-2mm)
    * Speed: 25-45 mm/s
    * Excessive retraction causes nozzle clogging
  * **Z-hop (Z-axis hopping)** : Raising nozzle during movement to avoid collision with built object。0.2-0.5mm raise. Slightly increases build time but improves surface quality。
  * **Combing** : Restricting movement paths above infill to reduce surface movement marks。Effective when appearance is important。
  * **Seam Position** : Strategy for aligning layer start/end points.
    * Random: Random placement (inconspicuous)
    * Aligned: Aligned placement (easy to remove seam in post-processing)
    * Sharpest Corner: Place at sharpest corner (less noticeable)

### Example 1: Loading STL Files and Obtaining Basic Information
    
    
    # =================================== # Example 1: Load STL file and obtain basic information # =================================== import numpy as np from stl import mesh # Load STL file your_mesh = mesh.Mesh.from_file('model.stl') # Get basic geometric information volume, cog, inertia = your_mesh.get_mass_properties() print("=== STL File Basic Information ===") print(f"Volume: {volume:.2f} mm³") print(f"Surface Area: {your_mesh.areas.sum():.2f} mm²") print(f"Center of Gravity: [{cog[0]:.2f}, {cog[1]:.2f}, {cog[2]:.2f}] mm") print(f"Number of Triangles: {len(your_mesh.vectors)}") # Calculate bounding box min_coords = your_mesh.vectors.min(axis=(0, 1)) max_coords = your_mesh.vectors.max(axis=(0, 1)) dimensions = max_coords - min_coords print(f"\n=== Bounding Box ===") print(f"X: {min_coords[0]:.2f} to {max_coords[0]:.2f} mm (Width: {dimensions[0]:.2f} mm)") print(f"Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} mm (Depth: {dimensions[1]:.2f} mm)") print(f"Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} mm (Height: {dimensions[2]:.2f} mm)") # Build time simple estimation (Layer Height 0.2mm, speed 50mm/s assumed) layer_height = 0.2 # mm print_speed = 50 # mm/s num_layers = int(dimensions[2] / layer_height) # Simple calculation: estimation based on surface area estimated_path_length = your_mesh.areas.sum() / layer_height # mm estimated_time_seconds = estimated_path_length / print_speed estimated_time_minutes = estimated_time_seconds / 60 print(f"\n=== Build Estimation ===") print(f"Number of layers（0.2mm/layer）: {num_layers} layer") print(f"Estimated build time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.2f} hours)") # Output example: # === STL File Basic Information === # Volume: 12450.75 mm³ # Surface Area: 5832.42 mm² # Center of Gravity: [25.34, 18.92, 15.67] mm # Number of Triangles: 2456 # # === Bounding Box === # X: 0.00 to 50.00 mm (Width: 50.00 mm) # Y: 0.00 to 40.00 mm (Depth: 40.00 mm) # Z: 0.00 to 30.00 mm (Height: 30.00 mm) # # === Build Estimation === # Number of layers（0.2mm/layer）: 150 layer # Estimated build time: 97.2 minutes (1.62 hours)
    

### Example 2: Mesh Normal Vector Verification
    
    
    # =================================== # Example 2: Mesh normal vector verification # =================================== import numpy as np from stl import mesh def check_normals(mesh_data): """Check STL mesh normal vector consistency Args: mesh_data: numpy-stl Mesh object Returns: tuple: (flipped_count, total_count, percentage) """ # Confirm normal direction according to right-hand rule flipped_count = 0 total_count = len(mesh_data.vectors) for i, facet in enumerate(mesh_data.vectors): v0, v1, v2 = facet # Calculate edge vectors edge1 = v1 - v0 edge2 = v2 - v0 # Calculate normal by cross product (right-hand system) calculated_normal = np.cross(edge1, edge2) # Normalize norm = np.linalg.norm(calculated_normal) if norm >1e-10: # Confirm not zero vector calculated_normal = calculated_normal / norm else: continue # Skip degenerate triangles # Compare with stored normal in file stored_normal = mesh_data.normals[i] stored_norm = np.linalg.norm(stored_normal) if stored_norm >1e-10: stored_normal = stored_normal / stored_norm # Check direction match with dot product dot_product = np.dot(calculated_normal, stored_normal) # If dot product is negative, opposite orientation if dot_product< 0: flipped_count += 1 percentage = (flipped_count / total_count) * 100 if total_count >0 else 0 return flipped_count, total_count, percentage # Load STL file your_mesh = mesh.Mesh.from_file('model.stl') # Execute normal check flipped, total, percent = check_normals(your_mesh) print("=== Normal Vector Verification Results ===") print(f"Total triangles: {total}") print(f"Inverted normals: {flipped}") print(f"Inversion rate: {percent:.2f}%") if flipped == 0: print("\n✅ All normals are correctly oriented") print(" This mesh is 3D printable") elif percent< 5: print("\n⚠️ Some normals are inverted (minor)") print(" High possibility slicer can auto-correct") else: print("\n❌ Many normals are inverted (critical)") print(" Repair with mesh repair tools (Meshmixer, netfabb) recommended") # Output example: # === Normal Vector Verification Results === # Total triangles: 2456 # Inverted normals: 0 # Inversion rate: 0.00% # # ✅ All normals are correctly oriented # This mesh is 3D printable
    

### Example 3: Manifold Check
    
    
    # =================================== # Example 3: Manifold property (Watertight) check # =================================== import trimesh # Load STL file (trimesh attempts automatic repair) mesh = trimesh.load('model.stl') print("=== Mesh Quality Diagnostics ===") # Basic information print(f"Vertex count: {len(mesh.vertices)}") print(f"Face count: {len(mesh.faces)}") print(f"Volume: {mesh.volume:.2f} mm³") # Manifold property check print(f"\n=== 3D Printability Check ===") print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") print(f"Is valid: {mesh.is_valid}") # Detailed problem diagnosis if not mesh.is_watertight: # Detect number of holes try: edges = mesh.edges_unique edges_sorted = mesh.edges_sorted duplicate_edges = len(edges_sorted) - len(edges) print(f"\n⚠️ Problem detected:") print(f" - Mesh has holes") print(f" - Duplicate edge count: {duplicate_edges}") except: print(f"\n⚠️ Mesh structure problem exists") # Attempt repair if not mesh.is_watertight or not mesh.is_winding_consistent: print(f"\n🔧 Executing automatic repair...") # Correct normals trimesh.repair.fix_normals(mesh) print(" ✓ Fixed normal vectors") # Fill holes trimesh.repair.fill_holes(mesh) print(" ✓ Filled holes") # Remove degenerate triangles mesh.remove_degenerate_faces() print(" ✓ Removed degenerate faces") # Merge duplicate vertices mesh.merge_vertices() print(" ✓ Merged duplicate vertices") # Confirm state after repair print(f"\n=== Post-Repair Status ===") print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") # Save repaired mesh if mesh.is_watertight: mesh.export('model_repaired.stl') print(f"\n✅ Repair complete! Saved as model_repaired.stl") else: print(f"\n❌ Automatic repair failed. Specialized tools like Meshmixer recommended") else: print(f"\n✅ This mesh is 3D printable") # Output example: # === Mesh Quality Diagnostics === # Vertex count: 1534 # Face count: 2456 # Volume: 12450.75 mm³ # # === 3D Printability Check === # Is watertight: True # Is winding consistent: True # Is valid: True # # ✅ This mesh is 3D printable
    

### Example 4: Basic Slicing Algorithm
    
    
    # =================================== # Example 4: Basic slicing algorithm # =================================== import numpy as np from stl import mesh def slice_mesh_at_height(mesh_data, z_height): """Generate temperature profile Args: t (array): Time array [min] T_target (float): Hold temperature [°C] heating_rate (float): Heating rate [°C/min] hold_time (float): Hold time [min] cooling_rate (float): Cooling rate [°C/min] Returns: array: Temperature profile [°C] """ T_room = 25 # Room temperature T = np.zeros_like(t) # Heating time t_heat = (T_target - T_room) / heating_rate # Cooling start time t_cool_start = t_heat + hold_time for i, time in enumerate(t): if time<= t_heat: # Heating phase T[i] = T_room + heating_rate * time elif time<= t_cool_start: # Holding phase T[i] = T_target else: # Cooling phase T[i] = T_target - cooling_rate * (time - t_cool_start) T[i] = max(T[i], T_room) # Not below room temperature return T def simulate_reaction_progress(T, t, Ea, D0, r0): """Calculate reaction progress based on temperature profile Args: T (array): Temperature profile [°C] t (array): Time array [min] Ea (float): Activation energy [J/mol] D0 (float): Frequency factor [m²/s] r0 (float): Particle radius [m] Returns: array: Reaction rate """ R = 8.314 C0 = 10000 alpha = np.zeros_like(t) for i in range(1, len(t)): T_k = T[i] + 273.15 D = D0 * np.exp(-Ea / (R * T_k)) k = D * C0 / r0**2 dt = (t[i] - t[i-1]) * 60 # min → s # Simple integration (infinitesimal time reaction progress) if alpha[i-1]< 0.99: dalpha = k * dt / (2 * (1 - (1-alpha[i-1])**(1/3))) alpha[i] = min(alpha[i-1] + dalpha, 1.0) else: alpha[i] = alpha[i-1] return alpha # Parameter setting T_target = 1200 # °C hold_time = 240 # min (4 hours) Ea = 300e3 # J/mol D0 = 5e-4 # m²/s r0 = 5e-6 # m # Comparison at different heating rates heating_rates = [2, 5, 10, 20] # °C/min cooling_rate = 3 # °C/min # Time array t_max = 800 # min t = np.linspace(0, t_max, 2000) # Plot fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) # Temperature profile for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) ax1.plot(t/60, T_profile, linewidth=2, label=f'{hr}°C/min') ax1.set_xlabel('Time (hours)', fontsize=12) ax1.set_ylabel('Temperature (°C)', fontsize=12) ax1.set_title('Temperature Profiles', fontsize=14, fontweight='bold') ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) ax1.set_xlim([0, t_max/60]) # Reaction progress for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0) ax2.plot(t/60, alpha, linewidth=2, label=f'{hr}°C/min') ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='Target (95%)') ax2.set_xlabel('Time (hours)', fontsize=12) ax2.set_ylabel('Conversion', fontsize=12) ax2.set_title('Reaction Progress', fontsize=14, fontweight='bold') ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) ax2.set_xlim([0, t_max/60]) ax2.set_ylim([0, 1]) plt.tight_layout() plt.savefig('temperature_profile_optimization.png', dpi=300, bbox_inches='tight') plt.show() # Calculate time to reach 95% reaction for each heating rate print("\nComparison of time to reach 95% reaction:") print("=" * 60) for hr in heating_rates: T_profile = temperature_profile(t, T_target, hr, hold_time, cooling_rate) alpha = simulate_reaction_progress(T_profile, t, Ea, D0, r0) # Time to reach 95% idx_95 = np.where(alpha >= 0.95)[0] if len(idx_95) >0: t_95 = t[idx_95[0]] / 60 print(f"Heating rate {hr:2d}°C/min: t₉₅ = {t_95:.1f} hours") else: print(f"Heating rate {hr:2d}°C/min: Incomplete reaction") # Output example: # Comparison of time to reach 95% reaction: # ============================================================ # Heating rate 2°C/min: t₉₅ = 7.8 hours # Heating rate 5°C/min: t₉₅ = 7.2 hours # Heating rate 10°C/min: t₉₅ = 6.9 hours # Heating rate 20°C/min: t₉₅ = 6.7 hours
    

## Exercises

### 1.5.1 What is pycalphad

**pycalphad** is a Python library for phase diagram calculations based on the CALPHAD (CALculation of PHAse Diagrams) method. It calculates equilibrium phases from thermodynamic databases and is useful for reaction pathway design.

**💡 Advantages of CALPHAD Method**

  * Can calculate complex phase diagrams of multicomponent systems (ternary and higher)
  * Can predict systems with limited experimental data
  * Can comprehensively handle temperature, composition, and pressure dependencies

### 1.5.2 Binary Phase Diagram Calculation Example
    
    
    # =================================== # Example 5: Phase diagram calculation with pycalphad # =================================== # Note: pycalphad installation required # pip install pycalphad from pycalphad import Database, equilibrium, variables as v import matplotlib.pyplot as plt import numpy as np # Load TDB database (simplified example here) # Actually requires proper TDB file # Example: BaO-TiO2 system # Simplified TDB string (more complex in practice) tdb_string = """ $ BaO-TiO2 system (simplified) ELEMENT BA BCC_A2 137.327 ! ELEMENT TI HCP_A3 47.867 ! ELEMENT O GAS 15.999 ! FUNCTION GBCCBA 298.15 +GHSERBA; 6000 N ! FUNCTION GHCPTI 298.15 +GHSERTI; 6000 N ! FUNCTION GGASO 298.15 +GHSERO; 6000 N ! PHASE LIQUID:L % 1 1.0 ! PHASE BAO_CUBIC % 2 1 1 ! PHASE TIO2_RUTILE % 2 1 2 ! PHASE BATIO3 % 3 1 1 3 ! """ # Note: Actual calculations require official TDB file # Limited to conceptual explanation here print("Concept of phase diagram calculation with pycalphad:") print("=" * 60) print("1. Load TDB database (thermodynamic data)") print("2. Set temperature and composition range") print("3. Execute equilibrium calculation") print("4. Visualize stable phases") print() print("Practical application examples:") print("- BaO-TiO2 system: Temperature and composition range for BaTiO3 formation") print("- Si-Nsystem: Si3N4 stability region") print("- Phase relationships in multicomponent ceramics") # Conceptual plot (image based on actual database) fig, ax = plt.subplots(figsize=(10, 7)) # Temperature range T = np.linspace(800, 1600, 100) # Each phase stability region (conceptual diagram) # BaO + TiO2 → BaTiO3 reaction BaO_region = np.ones_like(T) * 0.3 TiO2_region = np.ones_like(T) * 0.7 BaTiO3_region = np.where((T >1100) & (T< 1400), 0.5, np.nan) ax.fill_between(T, 0, BaO_region, alpha=0.3, color='blue', label='BaO + TiO2') ax.fill_between(T, BaO_region, TiO2_region, alpha=0.3, color='green', label='BaTiO3 stable') ax.fill_between(T, TiO2_region, 1, alpha=0.3, color='red', label='Liquid') ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='BaTiO3 composition') ax.axvline(x=1100, color='gray', linestyle=':', linewidth=1, alpha=0.5) ax.axvline(x=1400, color='gray', linestyle=':', linewidth=1, alpha=0.5) ax.set_xlabel('Temperature (°C)', fontsize=12) ax.set_ylabel('Composition (BaO mole fraction)', fontsize=12) ax.set_title('Conceptual Phase Diagram: BaO-TiO2', fontsize=14, fontweight='bold') ax.legend(fontsize=10, loc='upper right') ax.grid(True, alpha=0.3) ax.set_xlim([800, 1600]) ax.set_ylim([0, 1]) # Text annotation ax.text(1250, 0.5, 'BaTiO₃\nformation\nregion', fontsize=11, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)) plt.tight_layout() plt.savefig('phase_diagram_concept.png', dpi=300, bbox_inches='tight') plt.show() # Practical usage example (commented out) """ # Practical example using pycalphad db = Database('BaO-TiO2.tdb') # Load TDB file # Equilibrium calculation eq = equilibrium(db, ['BA', 'TI', 'O'], ['LIQUID', 'BATIO3'], {v.X('BA'): (0, 1, 0.01), v.T: (1000, 1600, 50), v.P: 101325}) # Plot result eq.plot() """
    

## 1.6 Condition Optimization with Design of Experiments (DOE)

### 1.6.1 What is DOE

Design of Experiments (DOE) is a statistical technique to find optimal conditions with minimum number of experiments in systems with multiple interacting parameters。

**Main parameters to optimize in solid-state reactions:**

  * reactionTemperature（T）
  * Hold time (t)
  * Particle size (r)
  * Raw material ratio (molar ratio)
  * Atmosphere (air, nitrogen, vacuum, etc.)

### 1.6.2 Response Surface Methodology
    
    
    # =================================== # Example 6: Condition optimization with DOE # =================================== import numpy as np import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D from scipy.optimize import minimize # Virtual reaction yield model (Temperature and time function) def reaction_yield(T, t, noise=0): """Calculate reaction yield from temperature and time (virtual model) Args: T (float): Temperature [°C] t (float): Time [hours] noise (float): Noise level Returns: float: Reaction rate [%] """ # Optimal value: T=1200°C, t=6 hours T_opt = 1200 t_opt = 6 # 2nd order model (Gaussian type) yield_val = 100 * np.exp(-((T-T_opt)/150)**2 - ((t-t_opt)/3)**2) # Add noise if noise >0: yield_val += np.random.normal(0, noise) return np.clip(yield_val, 0, 100) # Experiment point placement (central composite design) T_levels = [1000, 1100, 1200, 1300, 1400] # °C t_levels = [2, 4, 6, 8, 10] # hours # Grid experiment point placement T_grid, t_grid = np.meshgrid(T_levels, t_levels) yield_grid = np.zeros_like(T_grid, dtype=float) # Measure reaction yield at each experiment point (simulation) for i in range(len(t_levels)): for j in range(len(T_levels)): yield_grid[i, j] = reaction_yield(T_grid[i, j], t_grid[i, j], noise=2) # Display results print("Reaction Condition Optimization with Design of Experiments") print("=" * 70) print(f"{'Temperature (°C)':<20} {'Time (hours)':<20} {'Yield (%)':<20}") print("-" * 70) for i in range(len(t_levels)): for j in range(len(T_levels)): print(f"{T_grid[i, j]:<20} {t_grid[i, j]:<20} {yield_grid[i, j]:<20.1f}") # Find condition with highest reaction yield max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape) T_best = T_grid[max_idx] t_best = t_grid[max_idx] yield_best = yield_grid[max_idx] print("-" * 70) print(f"Optimal condition: T = {T_best}°C, t = {t_best} hours") print(f"Highest reaction yield: {yield_best:.1f}%") # 3D plot fig = plt.figure(figsize=(14, 6)) # 3D surface plot ax1 = fig.add_subplot(121, projection='3d') T_fine = np.linspace(1000, 1400, 50) t_fine = np.linspace(2, 10, 50) T_mesh, t_mesh = np.meshgrid(T_fine, t_fine) yield_mesh = np.zeros_like(T_mesh) for i in range(len(t_fine)): for j in range(len(T_fine)): yield_mesh[i, j] = reaction_yield(T_mesh[i, j], t_mesh[i, j]) surf = ax1.plot_surface(T_mesh, t_mesh, yield_mesh, cmap='viridis', alpha=0.8, edgecolor='none') ax1.scatter(T_grid, t_grid, yield_grid, color='red', s=50, label='Experimental points') ax1.set_xlabel('Temperature (°C)', fontsize=10) ax1.set_ylabel('Time (hours)', fontsize=10) ax1.set_zlabel('Yield (%)', fontsize=10) ax1.set_title('Response Surface', fontsize=12, fontweight='bold') ax1.view_init(elev=25, azim=45) fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5) # Contour plot ax2 = fig.add_subplot(122) contour = ax2.contourf(T_mesh, t_mesh, yield_mesh, levels=20, cmap='viridis') ax2.contour(T_mesh, t_mesh, yield_mesh, levels=10, colors='black', alpha=0.3, linewidths=0.5) ax2.scatter(T_grid, t_grid, c=yield_grid, s=100, edgecolors='red', linewidths=2, cmap='viridis') ax2.scatter(T_best, t_best, color='red', s=300, marker='*', edgecolors='white', linewidths=2, label='Optimum') ax2.set_xlabel('Temperature (°C)', fontsize=11) ax2.set_ylabel('Time (hours)', fontsize=11) ax2.set_title('Contour Map', fontsize=12, fontweight='bold') ax2.legend(fontsize=10) fig.colorbar(contour, ax=ax2, label='Yield (%)') plt.tight_layout() plt.savefig('doe_optimization.png', dpi=300, bbox_inches='tight') plt.show()
    

### 1.6.3 Practical Approach to Experimental Design

In actual solid-state reactions, apply DOE with the following procedure:

  1. **Screening experiment** (two-level factorial design): Identify parameters with large influence
  2. **Response surface method** (central composite design): Search for optimal conditions
  3. **Confirmation experiment** : Conduct experiment at predicted optimal condition to verify model

**✅ Practical example: Optimization of LiCoO₂ synthesis for Li-ion battery positive electrode material**

A research group optimized LiCoO₂ synthesis conditions using DOE with the following results:

  * Number of experiments: Conventional method 100 times → DOE method 25 times (75% reduction)
  * Optimal temperature: 900°C (higher than conventional 850°C)
  * Optimal hold time: 12 hours (half of conventional 24 hours)
  * Battery capacity: 140 mAh/g → 155 mAh/g (11% improvement)

## 1.7 Fitting of Reaction Rate Curves

### 1.7.1 Determination of Rate Constants from Experimental Data
    
    
    # =================================== # Example 7: Reaction kinetics curve fitting # =================================== import numpy as np import matplotlib.pyplot as plt from scipy.optimize import curve_fit # Experimental data (time vs conversion) # Example: BaTiO3 synthesis @ 1200°C time_exp = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]) # hours conversion_exp = np.array([0, 0.15, 0.28, 0.38, 0.47, 0.60, 0.70, 0.78, 0.84, 0.90, 0.95]) # Jander equation model def jander_model(t, k): """Calculate reaction rate using Jander equation Args: t (array): Time [hours] k (float): Rate constant Returns: array: Reaction rate """ # [1 - (1-α)^(1/3)]² = kt solve for α kt = k * t alpha = 1 - (1 - np.sqrt(kt))**3 alpha = np.clip(alpha, 0, 1) # Limit to 0-1 range return alpha # Ginstling-Brounshtein equation (alternative diffusion model) def gb_model(t, k): """Ginstling-Brounshtein equation Args: t (array): hours k (float): Rate constant Returns: array: Reaction rate """ # 1 - 2α/3 - (1-α)^(2/3) = kt # Requires numerical solution, using approximate form here kt = k * t alpha = 1 - (1 - kt/2)**(3/2) alpha = np.clip(alpha, 0, 1) return alpha # Power law (empirical equation) def power_law_model(t, k, n): """Power law model Args: t (array): hours k (float): Rate constant n (float): Exponent Returns: array: Reaction rate """ alpha = k * t**n alpha = np.clip(alpha, 0, 1) return alpha # Fit each model # Jander equation popt_jander, _ = curve_fit(jander_model, time_exp, conversion_exp, p0=[0.01]) k_jander = popt_jander[0] # Ginstling-Brounshtein equation popt_gb, _ = curve_fit(gb_model, time_exp, conversion_exp, p0=[0.01]) k_gb = popt_gb[0] # Power law popt_power, _ = curve_fit(power_law_model, time_exp, conversion_exp, p0=[0.1, 0.5]) k_power, n_power = popt_power # Generate prediction curves t_fit = np.linspace(0, 20, 200) alpha_jander = jander_model(t_fit, k_jander) alpha_gb = gb_model(t_fit, k_gb) alpha_power = power_law_model(t_fit, k_power, n_power) # Calculate residuals residuals_jander = conversion_exp - jander_model(time_exp, k_jander) residuals_gb = conversion_exp - gb_model(time_exp, k_gb) residuals_power = conversion_exp - power_law_model(time_exp, k_power, n_power) # Calculate R² def r_squared(y_true, y_pred): ss_res = np.sum((y_true - y_pred)**2) ss_tot = np.sum((y_true - np.mean(y_true))**2) return 1 - (ss_res / ss_tot) r2_jander = r_squared(conversion_exp, jander_model(time_exp, k_jander)) r2_gb = r_squared(conversion_exp, gb_model(time_exp, k_gb)) r2_power = r_squared(conversion_exp, power_law_model(time_exp, k_power, n_power)) # Plot fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # Fitting results ax1.plot(time_exp, conversion_exp, 'ko', markersize=8, label='Experimental data') ax1.plot(t_fit, alpha_jander, 'b-', linewidth=2, label=f'Jander (R²={r2_jander:.4f})') ax1.plot(t_fit, alpha_gb, 'r-', linewidth=2, label=f'Ginstling-Brounshtein (R²={r2_gb:.4f})') ax1.plot(t_fit, alpha_power, 'g-', linewidth=2, label=f'Power law (R²={r2_power:.4f})') ax1.set_xlabel('Time (hours)', fontsize=12) ax1.set_ylabel('Conversion', fontsize=12) ax1.set_title('Kinetic Model Fitting', fontsize=14, fontweight='bold') ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) ax1.set_xlim([0, 20]) ax1.set_ylim([0, 1]) # Residual plot ax2.plot(time_exp, residuals_jander, 'bo-', label='Jander') ax2.plot(time_exp, residuals_gb, 'ro-', label='Ginstling-Brounshtein') ax2.plot(time_exp, residuals_power, 'go-', label='Power law') ax2.axhline(y=0, color='black', linestyle='--', linewidth=1) ax2.set_xlabel('Time (hours)', fontsize=12) ax2.set_ylabel('Residuals', fontsize=12) ax2.set_title('Residual Plot', fontsize=14, fontweight='bold') ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('kinetic_fitting.png', dpi=300, bbox_inches='tight') plt.show() # Result summary print("\nKinetic Model Fitting Results:") print("=" * 70) print(f"{'Model':<25} {'Parameter':<30} {'R²':<10}") print("-" * 70) print(f"{'Jander':<25} {'k = ' + f'{k_jander:.4f} h⁻¹':<30} {r2_jander:.4f}") print(f"{'Ginstling-Brounshtein':<25} {'k = ' + f'{k_gb:.4f} h⁻¹':<30} {r2_gb:.4f}") print(f"{'Power law':<25} {'k = ' + f'{k_power:.4f}, n = {n_power:.4f}':<30} {r2_power:.4f}") print("=" * 70) print(f"\nBest model: {'Jander' if r2_jander == max(r2_jander, r2_gb, r2_power) else 'GB' if r2_gb == max(r2_jander, r2_gb, r2_power) else 'Power law'}") # Output example: # Kinetic Model Fitting Results: # ====================================================================== # Model Parameter R² # ---------------------------------------------------------------------- # Jander k = 0.0289 h⁻¹ 0.9953 # Ginstling-Brounshtein k = 0.0412 h⁻¹ 0.9867 # Power law k = 0.2156, n = 0.5234 0.9982 # ====================================================================== # # Best model: Power law
    

## 1.8 Advanced Topics: Microstructure Control

### 1.8.1 Suppression of Grain Growth

In solid-state reactions, high temperature and long hold times can cause undesirable grain growth. Strategies to suppress this:

  * **Two-step sintering** : Short hold at high temperature followed by long hold at low temperature
  * **Use of additives** : Add small amounts of grain growth inhibitors (e.g., MgO, Al₂O₃)
  * **Spark Plasma Sintering (SPS)** : Rapid heating and short sintering time

### 1.8.2 Mechanochemical Activation of Reactions

Using mechanochemical methods (high-energy ball milling), solid-state reactions can also proceed near room temperature:
    
    
    # =================================== # Example 8: Grain growth simulation # =================================== import numpy as np import matplotlib.pyplot as plt def grain_growth(t, T, D0, Ea, G0, n): """Grain size time evolution Burke-Turnbull equation: G^n - G0^n = k*t Args: t (array): Time [hours] T (float): Temperature [K] D0 (float): Frequency factor Ea (float): Activation energy [J/mol] G0 (float): Initial grain size [μm] n (float): particleachievelongExponent（ usually2-4） Returns: array: Grain size [μm] """ R = 8.314 k = D0 * np.exp(-Ea / (R * T)) G = (G0**n + k * t * 3600)**(1/n) # hours → seconds return G # Parameter setting D0_grain = 1e8 # μm^n/s Ea_grain = 400e3 # J/mol G0 = 0.5 # μm n = 3 # Temperature influence temps_celsius = [1100, 1200, 1300] t_range = np.linspace(0, 12, 100) # 0-12 hours plt.figure(figsize=(12, 5)) # Temperature dependency plt.subplot(1, 2, 1) for T_c in temps_celsius: T_k = T_c + 273.15 G = grain_growth(t_range, T_k, D0_grain, Ea_grain, G0, n) plt.plot(t_range, G, linewidth=2, label=f'{T_c}°C') plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Target grain size') plt.xlabel('Time (hours)', fontsize=12) plt.ylabel('Grain Size (μm)', fontsize=12) plt.title('Grain Growth at Different Temperatures', fontsize=14, fontweight='bold') plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.ylim([0, 5]) # Two-step sintering effect plt.subplot(1, 2, 2) # Conventional sintering: 1300°C, 6 hours t_conv = np.linspace(0, 6, 100) T_conv = 1300 + 273.15 G_conv = grain_growth(t_conv, T_conv, D0_grain, Ea_grain, G0, n) # Two-step: 1300°C 1h → 1200°C 5h t1 = np.linspace(0, 1, 20) G1 = grain_growth(t1, 1300+273.15, D0_grain, Ea_grain, G0, n) G_intermediate = G1[-1] t2 = np.linspace(0, 5, 80) G2 = grain_growth(t2, 1200+273.15, D0_grain, Ea_grain, G_intermediate, n) t_two_step = np.concatenate([t1, t2 + 1]) G_two_step = np.concatenate([G1, G2]) plt.plot(t_conv, G_conv, 'r-', linewidth=2, label='Conventional (1300°C)') plt.plot(t_two_step, G_two_step, 'b-', linewidth=2, label='Two-step (1300°C→1200°C)') plt.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5) plt.xlabel('Time (hours)', fontsize=12) plt.ylabel('Grain Size (μm)', fontsize=12) plt.title('Two-Step Sintering Strategy', fontsize=14, fontweight='bold') plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.ylim([0, 5]) plt.tight_layout() plt.savefig('grain_growth_control.png', dpi=300, bbox_inches='tight') plt.show() # Final grain size comparison G_final_conv = grain_growth(6, 1300+273.15, D0_grain, Ea_grain, G0, n) G_final_two_step = G_two_step[-1] print("\nGrain growth comparison:") print("=" * 50) print(f"Conventional (1300°C, 6h): {G_final_conv:.2f} μm") print(f"Two-step (1300°C 1h + 1200°C 5h): {G_final_two_step:.2f} μm") print(f"Grain size suppression effect: {(1 - G_final_two_step/G_final_conv)*100:.1f}%") # Output example: # Grain growth comparison: # ================================================== # Conventional (1300°C, 6h): 4.23 μm # Two-step (1300°C 1h + 1200°C 5h): 2.87 μm # Grain size suppression effect: 32.2%
    

## Learning Objectivesconfirm

Upon completing this chapter, you will be able to explain:

### Basic Understanding

  * ✅ Can explain the three rate-limiting stages of solid-state reactions (nucleation, interface reaction, diffusion)
  * ✅ Understands the physical meaning of the Arrhenius equation and temperature dependency
  * ✅ Can explain the difference between Jander and Ginstling-Brounshtein equations
  * ✅ Understands the three essential elements of temperature profiles (heating rate, hold time, cooling rate))

### Practical Skills

  * ✅ Can simulate temperature dependency of diffusion coefficient with Python
  * ✅ Can predict reaction progress using Jander equation
  * ✅ Can calculate activation energy from DSC/TG data using Kissinger method
  * ✅ Can optimize reaction conditions using DOE (Design of Experiments)
  * ✅ Understands basics of phase diagram calculation using pycalphad

### Applied Expertise

  * ✅ Can design synthesis process for new ceramic materials
  * ✅ Can infer reaction mechanism from experimental data and select appropriate kinetic equation
  * ✅ Can propose optimization strategy for industrial process conditions
  * ✅ Can propose grain growth control strategies (two-step sintering, etc.)

## Exercises

### Easy (Fundamentals)

Q1: STL File Format Understanding

Which of the following is a correct description of ASCII and Binary formats for STL files?

a) ASCII format has smaller file size  
b) Binary format is a text format that humans can read directly  
c) Binary format usually has file size 5-10 times smaller than ASCII format  
d) Binary format has lower accuracy than ASCII format

Show Answer

**Correct Answer: c) Binary format usually has file size 5-10 times smaller than ASCII format**

**Explanation:**

  * **ASCII STL** : Text format readable by humans. Each triangle is described in 7 lines (facet, normal, 3 vertices, endfacet). Large file size (tens of MB to hundreds of MB)。
  * **Binary STL** : Binary format compact. 80-byte header + 4-byte triangle count + 50 bytes per triangle. Same shape as ASCII but 1/5 to 1/10 the size。
  * Accuracy is the same for both formats (32-bit floating point)
  * Modern 3D printer software supports both formats, Binary recommended

**Practical Example:** 10,000 triangle model → ASCII: approximately 7MB, Binary: approximately 0.5MB

Q2: Build Time Simple Calculation

A build object with volume 12,000 mm³, height 30 mm is being built with Layer Height 0.2 mm, Print Speed 50 mm/s. What is the approximate build time? (Assume 20% infill, 2 wall layers)

a) 30 minutes  
b) 60 minutes  
c) 90 minutes  
d) 120 minutes

Show Answer

**Correct Answer: c) 90 minutes (approximately 1.5 hours)**

**calculationprocedure:**

  1. **Number of layers** : Height 30mm ÷ Layer Height 0.2mm = 150 layers
  2. **Estimate path length per layer** :
     * Volume 12,000mm³ → Average 80mm³ per layer
     * Wall (shell): approximately 200mm/layer (assuming 0.4mm nozzle diameter)
     * infill20%: approximately100mm/layer
     * Total: approximately 300mm/layer
  3. **Total path length** : 300mm/layer × 150layer = 45,000mm = 45m
  4. **printinghours** : 45,000mm ÷ 50mm/s = 900 seconds = 15 minutes
  5. **Actual time** : Considering travel time, retraction, acceleration/deceleration approximately 5-6 times → 75-90 minutes

**Key Point:** Estimated time provided by slicer software includes acceleration/deceleration, travel moves, and temperature stabilization, so typically 4-6 times the simple calculation。

Q3: AMprocessselection

Select the most suitable AM process for the following application: "Titanium alloy fuel injection nozzle for aircraft engine parts, complex internal flow channels, high strength and high heat resistance requirements"

a) FDM (Fused Deposition Modeling)  
b) SLA (Stereolithography)  
c) SLM (Selective Laser Melting)  
d) Binder Jetting

Show Answer

**Correct Answer: c) SLM (Selective Laser Melting / Powder Bed Fusion for Metal)**

**reason:**

  * **SLM Features** : Metal powder (titanium, inconel, stainless steel) completely melted by laser. High density (99.9%), high strength, high heat resistance。
  * **Application Suitability** :
    * ✓ Compatible with titanium alloy (Ti-6Al-4V)
    * ✓ Can manufacture complex internal flow channels (after support removal)
    * ✓ Aerospace-grade mechanical properties
    * ✓ GE Aviation actually mass-produces fuel injection nozzles with SLM
  * **Why Other Options Are Unsuitable** :
    * FDM: Plastic only, insufficient strength and heat resistance
    * SLA: Resin only, unsuitable for functional parts
    * Binder Jetting: Metal possible, but density after sintering 90-95% does not meet aerospace standards

**Practical Example:** GE Aviation LEAP fuel nozzle (SLM-made), consolidated 20 welded parts into 1 part, achieved 25% weight reduction and 5x improvement in durability。

### Medium (Application)

Q4: PythonSTLmeshverification

Complete the Python code below to verify manifold property (watertight) of an STL file。
    
    
    import trimesh mesh = trimesh.load('model.stl') # Add code here: Check manifold property, # if problem exists, automatically repair, then # save repaired mesh as 'model_fixed.stl'
    

Show Answer

**Example Answer:**
    
    
    import trimesh mesh = trimesh.load('model.stl') # Manifold property check print(f"Is watertight: {mesh.is_watertight}") print(f"Is winding consistent: {mesh.is_winding_consistent}") # Repair if problem exists if not mesh.is_watertight or not mesh.is_winding_consistent: print("Executing mesh repair...") # Correct normals trimesh.repair.fix_normals(mesh) # Fill holes trimesh.repair.fill_holes(mesh) # Remove degenerate triangles mesh.remove_degenerate_faces() # Merge duplicate vertices mesh.merge_vertices() # Confirm repair result print(f"Watertight after repair: {mesh.is_watertight}") # Save repaired mesh if mesh.is_watertight: mesh.export('model_fixed.stl') print("Repair complete: Saved as model_fixed.stl") else: print("⚠️ Automatic repair failed. Please use Meshmixer or other tools") else: print("✓ Mesh is 3D printable")
    

**Explanation:**

  * `trimesh.repair.fix_normals()`: Unify normal vector orientation
  * `trimesh.repair.fill_holes()`: Fill holes in mesh
  * `remove_degenerate_faces()`: Remove degenerate triangles with zero area
  * `merge_vertices()`: Merge duplicate vertices

**Practical Point:** For complex problems that trimesh cannot repair, specialized tools like Meshmixer, Netfabb, MeshLab are required。

Q5: supportmaterialvolumecalculation

A cylinder with diameter 40mm and height 30mm is built tilted 45 degrees from the bottom face. Assuming support density 15% and Layer Height 0.2mm, estimate the approximate support material volume。

Show Answer

**Solution Process:**

  1. **Identify Support-Requiring Region** :
     * 45 degree tilt → approximately half of cylinder bottom face becomes overhang (45 degrees or more slope)
     * When cylinder is tilted 45 degrees, one side becomes suspended
  2. **Support Region Geometric Calculation** :
     * Cylinder projection area: π × (20mm)² ≈ 1,257 mm²
     * Support-requiring area at 45 degree tilt: approximately 1,257mm² × 0.5 = 629 mm²
     * supportHeight: mostlargeapproximately 30mm × sin(45°) ≈ 21mm
     * Support volume (assuming 100% density): 629mm² × 21mm ÷ 2 (triangular shape) ≈ 6,600 mm³
  3. **Considering 15% Support Density** :
     * Actual support material: 6,600mm³ × 0.15 = **approximately 990 mm³**
  4. **verification** :
     * Cylinder solid volume: π × 20² × 30 ≈ 37,700 mm³
     * Support/solid ratio: 990 / 37,700 ≈ 2.6% (reasonable range)

**Answer: approximately 1,000 mm³ (990 mm³)**

**Practical Considerations:**

  * With optimal build orientation, support can be greatly reduced (in this example, if cylinder is built upright, support is unnecessary)
  * Using Tree Support, further 30-50% material reduction is possible
  * Using water-soluble support material (PVA, HIPS), removal is easy

Q6: Layer Height Optimization

A build object with height 60mm is being built considering quality and time balance. For Layer Height options of 0.1mm, 0.2mm, and 0.3mm, explain the build time ratio and recommended use for each。

Show Answer

**Answer:**

Layer Height | Number of layers | hoursratio | quality | Recommended Use  
---|---|---|---|---  
0.1 mm | 600layer | ×3.0 | Very High | Display figurines, medical models, end-use parts  
0.2 mm | 300layer | ×1.0（standard） | Good | General prototypes, functional parts  
0.3 mm | 200layer | ×0.67 | Low | Early prototypes, strength-priority internal parts  
  
**Basis for Time Ratio Calculation:**

  * Number of layers1/2becomeand、ZaxisMovementtimesnumberalso1/2
  * BUT: Printing time per layer slightly increases (because volume per layer increases)
  * Overall, Layer Height is "approximately inversely proportional" (strictly speaking, 0.9-1.1x coefficient exists)

**Practical Selection Criteria:**

  1. **0.1mm Recommended Cases** :
     * Surface quality top priority (customer presentation, exhibition)
     * Curved surface smoothness important (faces, curved shapes)
     * Want to almost eliminate layer lines
  2. **0.2mm Recommended Cases** :
     * Quality and time balance emphasis (most common)
     * Functional test prototypes
     * Adequate surface finish acceptable
  3. **0.3mm Recommended Cases** :
     * Speed priority (shape confirmation only)
     * Internal structure parts (appearance not important)
     * Large-scalebuildthing（hoursreductionEffectlarge）

**changenumberLayer Height（Advanced）:**   
Using variable layer height function in PrusaSlicer and Cura, flat parts 0.3mm and curved parts 0.1mm can be mixed, achieving both quality and time。

Q7: AM Process Selection Comprehensive Problem

Select the most suitable AM process to manufacture a lightweight bracket for aerospace (aluminum alloy, topology-optimized complex shape, high strength and lightweight requirements), and list three reasons. Also, list two post-processing steps to consider。

Show Answer

**Optimal Process: LPBF (Laser Powder Bed Fusion) - SLM for Aluminum**

**selectionreason（three）:**

  1. **Highdensity・Highstrength** :
     * Laser complete melting achieves relative density of 99.5% or higher
     * Mechanical properties (tensile strength, fatigue properties) comparable to forged material
     * Aerospace certification (AS9100, Nadcap) obtainable
  2. **Topology-Optimized Shape Manufacturability** :
     * Complex lattice structures (thickness 0.5mm or less) built with high precision
     * Hollow structures, bionic shapes, and other shapes impossible with conventional machining are compatible
     * After support removal, internal structures are also accessible
  3. **materialeffectrateandlightamount** :
     * Buy-to-Fly ratio (material input/final product weight) is 1/10 to 1/20 of conventional machining
     * Topology optimization achieves 40-60% weight reduction compared to conventional design
     * Aluminum alloys (AlSi10Mg, Scalmalloy) have maximum specific strength

**Required Post-Processing (Two):**

  1. **heatprocessing（Heat Treatment）** :
     * Stress Relief Annealing: 300°C, 2-4 hours
     * Purpose: Remove residual stress from build, improve dimensional stability
     * Effect: 30-50% improvement in fatigue life, prevent warping and deformation
  2. **surfaceprocessing（Surface Finishing）** :
     * Machining (CNC): Mounting faces, bolt holes with high precision machining (Ra < 3.2μm)
     * Electropolishing: Reduce surface roughness (Ra 10μm → 2μm)
     * Shot Peening: Impart compressive residual stress on surface layer, improve fatigue properties
     * Anodizing: Improve corrosion resistance, impart insulation properties (aerospace standard)

**Additional Considerations:**

  * **Build Direction** : Consider load direction and layer direction (Z-direction strength 10-15% lower)
  * **Support Design** : Easy-to-remove Tree Support, minimum contact area
  * **Quality Management** : CT scan for internal defect inspection, X-ray inspection
  * **Traceability** : Powder lot management, build parameter records

**Practical Example: Airbus A350 Titanium Bracket**   
Consolidated bracket that was assembled from 32 parts into 1 part, achieving 55% weight reduction, 65% lead time shortening, and 35% cost reduction。

3 levels × 3 levels = **9 times** (full factorial design) 

**DOE Advantages (Compared to Conventional Methods):**

  1. **Can Detect Interactions**
     * Conventional method: Evaluate temperature influence and time influence separately
     * DOE: Can determine interactions like "shorter time works at high temperature"
     * Example: 1300°C4hours10 minutes、1100°C8hoursnecessary、etc.
  2. **experimenttimesnumberreduction**
     * Conventional method (OFAT: One Factor At a Time):
       * Temperature study: 3 times (fixed time)
       * Time study: 3 times (fixed temperature)
       * confirmexperiment: multipletimes
       * Total: 10+ times
     * DOE: Complete in 9 times (all conditions covered + interaction analysis)
     * Furthermore, using central composite design can reduce to 7 times

**Additional Advantages:**

  * Obtain statistically significant conclusions (can evaluate variance)
  * Construct response surface, can predict untested conditions
  * Can detect cases where optimal condition is outside experimental range

### Hard (Advanced)

Q7: Complex Reaction System Design

Design a temperature profile to synthesize Li₁.₂Ni₀.₂Mn₀.₆O₂ (lithium-rich positive electrode material) under the following conditions:

  * originalmaterial: Li₂CO₃, NiO, Mn₂O₃
  * Goal: Single phase, grain size < 5 μm, precise Li/transition metal ratio control
  * Constraint: Li₂O volatilization at 900°C or higher (Li deficiency risk)

Describe the temperature profile (heating rate, hold temperature/time, cooling rate) and explain the design rationale。

View Answer

**Recommended Temperature Profile:**

**Phase 1: Pre-heating (Li₂CO₃ Decomposition)**

  * Room temperature → 500°C: 3°C/min
  * 500°C hold: 2 hours
  * **Rationale:** Li₂CO₃ decomposes (~450°C) slowly, ensuring complete CO₂ removal

**Phase 2: Intermediate Heating (Precursor Formation)**

  * 500°C → 750°C: 5°C/min
  * 750°C hold: 4 hours
  * **Rationale:** Intermediate phases like Li₂MnO₃ and LiNiO₂ form. Temperature low enough to avoid Li volatilization, ensuring homogeneity

**Phase 3: Calcination (Target Phase Synthesis)**

  * 750°C → 850°C: 2°C/min (slow)
  * 850°C hold: 12 hours
  * **reason:**
    * Long time necessary for Li₁.₂Ni₀.₂Mn₀.₆O₂ single phase formation
    * Limiting to 850°C minimizes Li volatilization (<900°C constraint)
    * Long hold time promotes diffusion, but temperature low enough to suppress grain growth

**Phase 4: cooling**

  * 850°C → Room temperature: 2°C/min
  * **Rationale:** Slow cooling improves crystallinity, prevents cracks from thermal stress

**Important Design Points:**

  1. **Li Volatilization Countermeasures:**
     * Limit to 900°C or less (critical constraint)
     * Furthermore, use Li-excess raw material (Li/TM = 1.25, etc.)
     * Calcine in oxygen gas flow to reduce Li₂O partial pressure
  2. **Grain Size Control ( < 5 μm):**
     * Low temperature (850°C) and long time (12h) proceed reaction
     * Hightemperature・shorthoursandparticleachievelong excessivebecome
     * Raw material particle size also 1μm or less fine
  3. **Composition Homogeneity:**
     * 750°C intermediate hold is important
     * At this stage, transition metal distribution becomes homogeneous
     * If necessary, after 750°C hold, cool once → pulverize → reheat

**Total Required Time:** approximately 30 hours (heating 12h + holding 18h)

**Alternative Technique Considerations:**

  * **Sol-gel method:** Synthesis possible at lower temperature (600-700°C), improved homogeneity
  * **Spray pyrolysis:** Easy grain size control
  * **Two-step sintering:** 900°C 1h → 800°C 10h suppresses grain growth

Q8: Kinetics Analysis Comprehensive Problem

From the data below, infer the reaction mechanism and calculate the activation energy。

**experimentdata:**

Temperature (°C) | Time to reach 50% reaction t₅₀ (hours)  
---|---  
1000 | 18.5  
1100 | 6.2  
1200 | 2.5  
1300 | 1.2  
  
Assuming Jander equation: [1-(1-0.5)^(1/3)]² = k·t₅₀

View Answer

**Answer:**

**step1: Rate constantkcalculation**

Jander equation at α=0.5:  
[1-(1-0.5)^(1/3)]² = [1-0.794]² = 0.206² = 0.0424

Therefore k = 0.0424 / t₅₀

T (°C) | T (K) | t₅₀ (h) | k (h⁻¹) | ln(k) | 1000/T (K⁻¹)  
---|---|---|---|---|---  
1000 | 1273 | 18.5 | 0.00229 | -6.080 | 0.7855  
1100 | 1373 | 6.2 | 0.00684 | -4.985 | 0.7284  
1200 | 1473 | 2.5 | 0.01696 | -4.077 | 0.6788  
1300 | 1573 | 1.2 | 0.03533 | -3.343 | 0.6357  
  
**step2: ArrheniusPlot**

ln(k) vs 1/T Plot（lineshapetimesreturn）

Linear fit: ln(k) = A - Eₐ/(R·T)

Slope = -Eₐ/R

lineshapetimesreturncalculation:  
slope = Δ(ln k) / Δ(1000/T)  
= (-3.343 - (-6.080)) / (0.6357 - 0.7855)  
= 2.737 / (-0.1498)  
= -18.27

**Step 3: Activation Energy Calculation**

slope = -Eₐ / (R × 1000)  
Eₐ = -slope × R × 1000  
Eₐ = 18.27 × 8.314 × 1000  
Eₐ = 151,899 J/mol ≈**152 kJ/mol**

**Step 4: Reaction Mechanism Consideration**

  * **Activation Energy Comparison:**
    * obtainedvalue: 152 kJ/mol
    * Typical solid-state diffusion: 200-400 kJ/mol
    * boundaryfacereaction: 50-150 kJ/mol
  * **Inferred Mechanism:**
    * This value is intermediate between interface reaction and diffusion
    * Possibility 1: Interface reaction rate-limiting (diffusion influence small)
    * Possibility 2: Particles fine, diffusion distance short, apparent Eₐ low
    * Possibility 3: Mixed rate control (both interface reaction and diffusion contribute)

**Step 5: Verification Method Proposal**

  1. **Particle size dependency:** Experiment with different particle sizes and confirm if k ∝ 1/r₀² holds
     * Holds → diffusion rate-limiting
     * Does not hold → interface reaction rate-limiting
  2. **Other Kinetic Equation Fitting:**
     * Ginstling-Brounshteinformat（3nextorigindiffusion）
     * Contracting sphere model（boundaryfacereaction）
     * Compare which has higher R²
  3. **Microstructure observation:** SEM observation of reaction interface
     * Thick product layer → evidence of diffusion rate-limiting
     * Thin product layer → possibility of interface reaction rate-limiting

**Final Conclusion:**   
Activation energy **Eₐ = 152 kJ/mol**   
Inferred mechanism: **Interface reaction rate-limiting, or diffusion rate-limiting in fine particle system**   
Additional experiments recommended。

## nextstep

In Chapter 5, we learned the fundamentals of additive manufacturing (AM), including the seven process categories according to ISO/ASTM 52900, STL file format structure, slicing, and G-code basics. In Chapter 2, we will learn about the detailed build process, material properties, and process parameter optimization for Material Extrusion (FDM/FFF)。

[← Series Index](<./index.html>) [Back to Series Index →](<./index.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2015)._Additive Manufacturing Technologies: 3D Printing, Rapid Prototyping, and Direct Digital Manufacturing_(2nd ed.). Springer. pp. 1-35, 89-145, 287-334. - Comprehensive AM technology textbook with detailed explanation of seven process categories and STL data processing
  2. ISO/ASTM 52900:2021._Additive manufacturing — General principles — Fundamentals and vocabulary_. International Organization for Standardization. - AM terminology and process category international standard specification, widely referenced in manufacturing industry
  3. Kruth, J.P., Leu, M.C., & Nakagawa, T. (1998). "Progress in Additive Manufacturing and Rapid Prototyping."_CIRP Annals - Manufacturing Technology_ , 47(2), 525-540. - Theoretical basis of selective laser sintering and binding mechanisms
  4. Hull, C.W. (1986)._Apparatus for production of three-dimensional objects by stereolithography_. US Patent 4,575,330. - World's first AM technology (SLA) patent, important document that became the origin of AM industry
  5. Wohlers, T. (2023)._Wohlers Report 2023: 3D Printing and Additive Manufacturing Global State of the Industry_. Wohlers Associates, Inc. pp. 15-89, 156-234. - Latest statistical report on AM market trends and industrial applications, updated annually as industry standard reference
  6. 3D Systems, Inc. (1988)._StereoLithography Interface Specification_. - STL file format official specification document, defining ASCII/Binary STL structure
  7. numpy-stl Documentation. (2024)._Python library for working with STL files_.<https://numpy-stl.readthedocs.io/>\- Python library for STL file loading and volume calculation
  8. trimesh Documentation. (2024)._Python library for loading and using triangular meshes_.<https://trimsh.org/>\- Comprehensive library for mesh repair, Boolean operations, and quality evaluation

## usefortoolandlibrary

  * **NumPy**(v1.24+): numbervaluecalculationlibrary -<https://numpy.org/>
  * **numpy-stl**(v3.0+): STLfileprocessinglibrary -<https://numpy-stl.readthedocs.io/>
  * **trimesh**(v4.0+): 3D mesh processing library (repair, verification, Boolean operations) -<https://trimsh.org/>
  * **Matplotlib**(v3.7+): Data visualization library -<https://matplotlib.org/>
  * **SciPy**(v1.10+): Scientific computing library (optimization, interpolation) -<https://scipy.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
