---
title: "Chapter 1: What are Materials - Classification and History"
chapter_title: "Chapter 1: What are Materials - Classification and History"
subtitle: From the Fundamentals of Materials Science to Modern Data-Driven Approaches
reading_time: 25-30 minutes
difficulty: Introductory
code_examples: 5
version: 1.0
created_at: 2025-10-25
---

Learn what materials are, how they are classified, and how materials science has evolved throughout human history. Build the foundation that connects to modern Materials Informatics and Process Informatics. 

## Learning Objectives

By reading this chapter, you will be able to:

  *  Explain the definition of materials and the purpose of materials science
  *  Understand the four major classes of materials (metals, ceramics, polymers, composites) and their characteristics
  *  Explain the history of materials science and its impact on human civilization
  *  Understand the fundamental concepts of material properties (mechanical, electrical, thermal, optical)
  *  Understand the relationship between Materials Informatics (MI) and Process Informatics (PI)
  *  Visualize material properties data using Python

* * *

## 1.1 Definition and Classification of Materials

### What are Materials?

**Materials** are substances used to make something. More technically, they can be defined as follows:

> **Materials** are substances whose composition, structure, and properties are engineeringly useful and are utilized as components of products or systems. 

Materials Science is a field of study that investigates the relationships among the **structure** , **properties** , **synthesis/processing methods** , and **performance** of materials. This relationship is represented as the "Materials Science Tetrahedron":
    
    
    ```mermaid
    graph TD
        A[StructureStructure] --- B[PropertiesProperties]
        A --- C[ProcessingProcessing]
        A --- D[PerformancePerformance]
        B --- C
        B --- D
        C --- D
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

**Important perspective** : The goal of materials science is to understand the relationships among these four elements and to **design and manufacture materials with desired performance**.

### The Four Major Classes of Materials

Materials are primarily classified into four categories based on their bonding types and structures:

#### 1\. Metallic Materials (Metals)

**Characteristics** :

  * Free electrons present due to metallic bonding
  * High electrical and thermal conductivity
  * Excellent ductility and malleability (easy to deform)
  * Possess metallic luster

**Representative examples** :

  * **Iron (Fe)** : Structural materials, automobiles, construction
  * **Copper (Cu)** : Electric wires, electronic components
  * **Aluminum (Al)** : Lightweight structures, aircraft
  * **Titanium (Ti)** : Biomedical materials, aerospace

**Applications** : Structural materials, conductive materials, mechanical parts, tools

#### 2\. Ceramic Materials (Ceramics)

**Characteristics** :

  * Ionic or covalent bonding
  * High hardness and heat resistance
  * Brittle (weak against impact)
  * High electrical insulation (some are semiconductors or superconductors)

**Representative examples** :

  * **Alumina (Al‚Oƒ)** : Abrasives, refractories, substrates
  * **Silica (SiO‚)** : Glass, optical fibers
  * **Silicon carbide (SiC)** : Heat-resistant materials, semiconductors
  * **Zirconia (ZrO‚)** : Ceramic blades, solid electrolytes

**Applications** : Refractories, electronic components, cutting tools, biomedical materials

#### 3\. Polymeric Materials (Polymers)

**Characteristics** :

  * Large molecules (polymers) formed by covalent bonding chains
  * Lightweight with good processability
  * High electrical insulation
  * Thermoplastic (softens when heated) or thermosetting

**Representative examples** :

  * **Polyethylene (PE)** : Plastic bags, containers
  * **Polypropylene (PP)** : Automotive parts, containers
  * **Polystyrene (PS)** : Styrofoam, packaging materials
  * **Nylon (PA)** : Fibers, mechanical parts

**Applications** : Packaging materials, fibers, medical devices, electronic device housings

#### 4\. Composite Materials (Composites)

**Characteristics** :

  * Combination of two or more materials
  * Leverage strengths and compensate weaknesses of each material
  * Combination of matrix (base material) and reinforcement
  * Lightweight and high strength

**Representative examples** :

  * **CFRP (Carbon Fiber Reinforced Plastic)** : Aircraft, sporting goods
  * **GFRP (Glass Fiber Reinforced Plastic)** : Ship hulls, automotive parts
  * **Concrete** : Cement + sand + gravel, construction
  * **Metal Matrix Composites (MMC)** : Al + SiC, high-strength parts

**Applications** : Aerospace, automotive, sporting goods, construction

### Material Classification Comparison Table

Property | Metals | Ceramics | Polymers | Composites  
---|---|---|---|---  
**Bonding type** | Metallic bonding | Ionic/covalent bonding | Covalent bonding | Mixed  
**Density** | High (2-20 g/cm³) | Medium-High (2-6 g/cm³) | Low (0.9-2 g/cm³) | Low-Medium  
**Strength** | High | Very high | Low-Medium | Very high  
**Ductility** | High | Low (brittle) | Medium-High | Low-Medium  
**Electrical conductivity** | High | Low (insulator) | Low (insulator) | Variable  
**Heat resistance** | High (~3000°C) | Very high (~3500°C) | Low (~200°C) | Medium  
**Processability** | Good | Difficult | Very good | Medium  
**Cost** | Medium | Medium-High | Low | High  
  
* * *

## 1.2 History and Importance of Materials Science

### Human History and Materials Development

The history of humanity is also the history of materials. In fact, historical periods are named after materials:

Era | Period | Primary materials | Technological features  
---|---|---|---  
**Stone Age** | ~3000 BC | Stone, wood, bone | Use of natural materials  
**Bronze Age** | 3000-1200 BC | Bronze (Cu + Sn) | Metal smelting and alloying  
**Iron Age** | 1200 BC~ | Iron | High-temperature smelting techniques  
**Industrial Revolution** | 1760-1840 | Steel (iron + carbon) | Mass production, steam engines  
**Polymer Age** | 1900~ | Plastics, rubber | Organic chemistry, synthetic materials  
**Semiconductor Age** | 1950~ | Silicon, GaAs | Electronics revolution  
**Composite Materials Age** | 1960~ | CFRP, composites | Lightweight high-strength materials  
**Nanomaterials Age** | 1990~ | Nanoparticles, CNT | Nanoscale control  
**MI/PI Age** | 2010~ | Data-driven materials | AI and machine learning utilization  
  
### Importance of Materials Science

Materials science is a fundamental technology for modern society and is indispensable in the following fields:

#### 1\. Energy Field

  * **Solar cells** : Silicon, perovskite (improved photoelectric conversion efficiency)
  * **Lithium-ion batteries** : Lithium cobalt oxide (high energy density)
  * **Fuel cells** : Solid polymer electrolytes (high-efficiency power generation)
  * **Superconducting materials** : YBCO (zero transmission loss)

#### 2\. Information and Communication Field

  * **Semiconductors** : Silicon, GaN (high speed, low power consumption)
  * **Optical fibers** : High-purity silica glass (high-speed communication)
  * **Magnetic recording materials** : CoFe alloys (large-capacity storage)

#### 3\. Medical and Biomedical Field

  * **Biomedical materials** : Titanium alloys (artificial joints)
  * **Bioresorbable materials** : PLA (decomposes in the body)
  * **Drug delivery systems** : Nanoparticles (targeted therapy)

#### 4\. Environment and Sustainability Field

  * **Catalyst materials** : Zeolites, precious metals (exhaust gas purification)
  * **Separation membranes** : Polymer membranes (water treatment, desalination)
  * **Lightweight materials** : Aluminum alloys, CFRP (fuel efficiency improvement)

* * *

## 1.3 Material Properties and Application Fields

### Classification of Material Properties

Material properties are mainly classified into four categories:

#### 1\. Mechanical Properties

The response of materials to forces and deformation.

  * **Strength** : Maximum stress a material can withstand without breaking
  * **Hardness** : Resistance to surface scratching
  * **Ductility** : Ability to deform without breaking
  * **Toughness** : Energy that can be absorbed before fracture
  * **Elastic Modulus** : Resistance to deformation (Young's modulus)

These properties are evaluated using the **stress-strain curve** :

$$\text{Stress} \, \sigma = \frac{F}{A} \quad (\text{Unit: Pa, MPa})$$

$$\text{Strain} \, \epsilon = \frac{\Delta L}{L_0} \quad (\text{Dimensionless})$$

#### 2\. Electrical Properties

The response of materials to electric fields or currents.

  * **Electrical Conductivity** : Ease of current flow (Unit: S/m)
  * **Resistivity** : Reciprocal of electrical conductivity (Unit: ©·m)
  * **Band Gap** : Energy required for electron excitation in semiconductors (Unit: eV)
  * **Dielectric Constant** : Susceptibility to electric fields

Materials are classified into three categories based on electrical conductivity:

  * **Conductors** : Ã > 10v S/m (metals)
  * **Semiconductors** : 10{x < Ã < 10v S/m (Si, GaAs)
  * **Insulators** : Ã < 10{x S/m (ceramics, polymers)

#### 3\. Thermal Properties

The response of materials to heat.

  * **Thermal Conductivity** : Ease of heat transfer (Unit: W/(m·K))
  * **Thermal Expansion Coefficient** : Dimensional change due to temperature variation (Unit: K{¹)
  * **Specific Heat Capacity** : Heat required to raise temperature (Unit: J/(kg·K))
  * **Melting Point** : Temperature at which solid changes to liquid

#### 4\. Optical Properties

The response of materials to light.

  * **Refractive Index** : How light bends
  * **Transmittance** : How much light passes through
  * **Reflectance** : How much light is reflected
  * **Absorption Coefficient** : How much light is absorbed

* * *

## 1.4 Relationship between Materials Science and MI/PI

### Relationship with Materials Informatics (MI)

**Materials Informatics (MI)** is a data-driven approach to discover and design new materials. Knowledge of materials science forms the foundation of MI.

**Typical MI workflow** :
    
    
    ```mermaid
    graph LR
        A[Materials DatabaseConstruction] --> B[DescriptorDesign]
        B --> C[Machine LearningModel Building]
        C --> D[Materials ScreeningPrediction]
        D --> E[Experimental Validation]
        E --> A
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

**Why materials science knowledge is necessary** :

  * Understanding of material properties is necessary for selecting appropriate descriptors
  * Knowledge of structure-property relationships is needed to judge the validity of prediction results
  * Knowledge of materials synthesis and processing is necessary for planning experimental validation

### Relationship with Process Informatics (PI)

**Process Informatics (PI)** is a data-driven method for optimizing manufacturing processes. Knowledge of materials science is essential for process design and quality control.

**Where materials science knowledge is useful** :

  * **Process design** : Set appropriate processing conditions by understanding thermal and mechanical properties of materials
  * **Quality control** : Understand the relationship between material properties and manufacturing conditions to build quality prediction models
  * **Troubleshooting** : Identify root causes of anomalies using materials science knowledge

**Example** : Semiconductor manufacturing process

  * Materials science: Understand silicon crystal structure and impurity diffusion mechanisms
  * PI: Model the relationship between temperature profile and impurity concentration to optimize processes

* * *

## 1.5 Visualization of Material Properties Data Using Python

Let's use Python to visualize material properties data and visually understand the differences in material classifications.

### Environment Setup

Install required libraries:
    
    
    # Install required libraries
    pip install numpy matplotlib pandas plotly seaborn
    

### Code Example 1: Material Classification Properties Comparison (Radar Chart)

Compare the properties of four types of materials (metals, ceramics, polymers, composites) using a radar chart.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Compare the properties of four types of materials (metals, c
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi
    
    # Material properties data (0-10 scale, 10 is highest)
    categories = ['Strength', 'Ductility', 'Electrical\nConductivity', 'Heat\nResistance', 'Lightness', 'Processability', 'Cost\nEfficiency']
    N = len(categories)
    
    # Property values for each material (0-10 scale)
    metals = [8, 9, 10, 7, 3, 7, 6]        # Metals
    ceramics = [9, 2, 1, 10, 5, 3, 5]      # Ceramics
    polymers = [4, 8, 1, 2, 9, 10, 9]      # Polymers
    composites = [9, 5, 3, 6, 8, 5, 3]     # Composites
    
    # Calculate angles (close the circle by adding the first value at the end)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    metals += metals[:1]
    ceramics += ceramics[:1]
    polymers += polymers[:1]
    composites += composites[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each material
    ax.plot(angles, metals, 'o-', linewidth=2, label='Metals', color='#1f77b4')
    ax.fill(angles, metals, alpha=0.15, color='#1f77b4')
    
    ax.plot(angles, ceramics, 'o-', linewidth=2, label='Ceramics', color='#ff7f0e')
    ax.fill(angles, ceramics, alpha=0.15, color='#ff7f0e')
    
    ax.plot(angles, polymers, 'o-', linewidth=2, label='Polymers', color='#2ca02c')
    ax.fill(angles, polymers, alpha=0.15, color='#2ca02c')
    
    ax.plot(angles, composites, 'o-', linewidth=2, label='Composites', color='#d62728')
    ax.fill(angles, composites, alpha=0.15, color='#d62728')
    
    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True)
    
    # Title and legend
    plt.title('Material Classification Properties Comparison', size=16, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("Read the characteristics of each material from the radar chart:")
    print("- Metals: Well-balanced, especially high in ductility and electrical conductivity")
    print("- Ceramics: Excellent in strength and heat resistance, but low ductility (brittle)")
    print("- Polymers: Lightweight with excellent processability and cost, but low strength and heat resistance")
    print("- Composites: Combining strength and lightness, balanced type")
    

**Explanation** : This radar chart allows visual understanding of the characteristics of each material class. For example, metals have overwhelmingly high electrical conductivity, while ceramics excel in heat resistance but have low ductility (brittle).

### Code Example 2: Material Density and Strength Relationship (Scatter Plot)

Plot the relationship between density and tensile strength of representative materials to learn material selection perspectives.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Plot the relationship between density and tensile strength o
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material data (density g/cm³, tensile strength MPa)
    materials = {
        'Metals': {
            'Iron': (7.87, 400),
            'Copper': (8.96, 220),
            'Aluminum': (2.70, 90),
            'Titanium': (4.51, 240),
            'Magnesium': (1.74, 100),
            'Stainless steel': (8.00, 520),
        },
        'Ceramics': {
            'Alumina': (3.95, 300),
            'Silicon carbide': (3.21, 400),
            'Zirconia': (6.05, 900),
            'Silicon nitride': (3.44, 700),
        },
        'Polymers': {
            'Polyethylene': (0.95, 30),
            'Polypropylene': (0.90, 35),
            'Nylon': (1.14, 80),
            'PEEK': (1.32, 100),
        },
        'Composites': {
            'CFRP': (1.60, 600),
            'GFRP': (1.80, 200),
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Metals': '#1f77b4', 'Ceramics': '#ff7f0e',
              'Polymers': '#2ca02c', 'Composites': '#d62728'}
    
    # Plot for each material category
    for category, materials_dict in materials.items():
        densities = [v[0] for v in materials_dict.values()]
        strengths = [v[1] for v in materials_dict.values()]
        names = list(materials_dict.keys())
    
        ax.scatter(densities, strengths, s=150, alpha=0.7,
                   color=colors[category], label=category, edgecolors='black', linewidth=1.5)
    
        # Display material names as labels
        for name, x, y in zip(names, densities, strengths):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Axis labels and title
    ax.set_xlabel('Density (g/cm³)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Tensile Strength (MPa)', fontsize=13, fontweight='bold')
    ax.set_title('Material Density and Strength Relationship', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)
    
    # Specific strength (strength/density) guidelines
    x_line = np.linspace(0.5, 9, 100)
    for specific_strength in [50, 100, 200, 400]:
        y_line = specific_strength * x_line
        ax.plot(x_line, y_line, '--', alpha=0.3, color='gray', linewidth=0.8)
        ax.text(8.5, specific_strength * 8.5, f'{specific_strength}',
                fontsize=8, alpha=0.6, rotation=30)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate specific strength (strength-to-weight ratio)
    print("\nSpecific strength ranking (strength/density, unit: MPa/(g/cm³)):")
    all_materials = []
    for category, materials_dict in materials.items():
        for name, (density, strength) in materials_dict.items():
            specific_strength = strength / density
            all_materials.append((name, specific_strength, category))
    
    all_materials.sort(key=lambda x: x[1], reverse=True)
    for i, (name, ss, category) in enumerate(all_materials[:5], 1):
        print(f"{i}. {name} ({category}): {ss:.1f}")
    

**Output example** :
    
    
    Specific strength ranking (strength/density, unit: MPa/(g/cm³)):
    1. CFRP (Composites): 375.0
    2. Silicon nitride (Ceramics): 203.5
    3. Zirconia (Ceramics): 148.8
    4. Silicon carbide (Ceramics): 124.6
    5. GFRP (Composites): 111.1
    

**Explanation** : From this graph, we can see that composites such as CFRP are lightweight (low density) while having high strength. This is why composites are valued in the aerospace field.

### Code Example 3: Material Electrical Conductivity Comparison (Logarithmic Scale)

Since electrical conductivity of materials differs by orders of magnitude, we plot it on a logarithmic scale.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Since electrical conductivity of materials differs by orders
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material electrical conductivity data (unit: S/m)
    materials_conductivity = {
        'Silver': 6.3e7,
        'Copper': 5.96e7,
        'Gold': 4.1e7,
        'Aluminum': 3.5e7,
        'Tungsten': 1.8e7,
        'Stainless steel': 1.4e6,
        'Graphite': 1e5,
        'Germanium': 2.0,
        'Silicon': 1e-3,
        'Pure water': 5.5e-6,
        'Glass': 1e-11,
        'Teflon': 1e-16,
        'Polyethylene': 1e-17,
    }
    
    # Material classification
    categories_conductivity = {
        'Conductors': ['Silver', 'Copper', 'Gold', 'Aluminum', 'Tungsten', 'Stainless steel', 'Graphite'],
        'Semiconductors': ['Germanium', 'Silicon'],
        'Insulators': ['Pure water', 'Glass', 'Teflon', 'Polyethylene']
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors_conductivity = {'Conductors': '#1f77b4', 'Semiconductors': '#ff7f0e', 'Insulators': '#2ca02c'}
    
    y_pos = 0
    yticks = []
    yticklabels = []
    
    for category, material_list in categories_conductivity.items():
        for material in material_list:
            conductivity = materials_conductivity[material]
            ax.barh(y_pos, conductivity, color=colors_conductivity[category],
                    alpha=0.7, edgecolor='black', linewidth=1)
            yticks.append(y_pos)
            yticklabels.append(material)
            y_pos += 1
        y_pos += 0.5  # Space between categories
    
    # Set logarithmic scale
    ax.set_xscale('log')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_xlabel('Electrical Conductivity (S/m)', fontsize=12, fontweight='bold')
    ax.set_title('Material Electrical Conductivity Comparison (Logarithmic Scale)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend (manually created)
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, edgecolor='black')
                       for color in colors_conductivity.values()]
    ax.legend(legend_elements, categories_conductivity.keys(),
              loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("\nElectrical conductivity range:")
    print(f"Conductors (metals): 10v - 10x S/m")
    print(f"Semiconductors: 10{x - 10v S/m")
    print(f"Insulators: < 10{x S/m")
    print("\nMost conductive material: Silver (6.3×10w S/m)")
    print("Most insulating material: Polyethylene (10{¹w S/m)")
    print(f"Difference between them: About 10²t times!")
    

**Explanation** : Electrical conductivity differs by about 24 orders of magnitude between materials. This enormous difference allows materials to be classified into conductors, semiconductors, and insulators. Copper and aluminum are used for wires, silicon for semiconductor devices, and polyethylene for wire insulation coating.

### Code Example 4: Material Melting Point and Thermal Conductivity Relationship

Plot the relationship between melting point and thermal conductivity of materials to obtain guidelines for material selection.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Plot the relationship between melting point and thermal cond
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material data (melting point K, thermal conductivity W/(m·K))
    materials_thermal = {
        'Metals': {
            'Copper': (1358, 401),
            'Aluminum': (933, 237),
            'Iron': (1811, 80),
            'Titanium': (1941, 22),
            'Tungsten': (3695, 173),
            'Silver': (1235, 429),
        },
        'Ceramics': {
            'Alumina': (2345, 30),
            'Silicon nitride': (2173, 90),
            'Silicon carbide': (3103, 120),
            'Zirconia': (2988, 2),
            'Diamond': (3823, 2200),
        },
        'Polymers': {
            'Polyethylene': (408, 0.4),
            'Polypropylene': (433, 0.22),
            'PTFE': (600, 0.25),
            'PEEK': (616, 0.25),
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_thermal = {'Metals': '#1f77b4', 'Ceramics': '#ff7f0e', 'Polymers': '#2ca02c'}
    
    for category, materials_dict in materials_thermal.items():
        melting_points = [v[0] for v in materials_dict.values()]
        thermal_conductivities = [v[1] for v in materials_dict.values()]
        names = list(materials_dict.keys())
    
        ax.scatter(melting_points, thermal_conductivities, s=150, alpha=0.7,
                   color=colors_thermal[category], label=category,
                   edgecolors='black', linewidth=1.5)
    
        # Display material names as labels
        for name, x, y in zip(names, melting_points, thermal_conductivities):
            offset_x = 10 if name != 'Diamond' else -50
            offset_y = 10 if name != 'Diamond' else -100
            ax.annotate(name, (x, y), xytext=(offset_x, offset_y),
                        textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Logarithmic scale (thermal conductivity)
    ax.set_yscale('log')
    ax.set_xlabel('Melting Point (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Thermal Conductivity (W/(m·K))', fontsize=13, fontweight='bold')
    ax.set_title('Material Melting Point and Thermal Conductivity Relationship', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nMaterials suitable for high-temperature applications:")
    print("- Tungsten: Melting point 3695K, aircraft engines")
    print("- Silicon carbide: Melting point 3103K, heat-resistant parts")
    print("- Diamond: Melting point 3823K, phenomenal thermal conductivity (2200 W/(m·K))")
    print("\nThermal management applications:")
    print("- Copper/Silver: High thermal conductivity (400+ W/(m·K)), heat sinks")
    print("- Diamond: Highest thermal conductivity, semiconductor heat dissipation substrates")
    

**Explanation** : Diamond has an extremely high melting point and overwhelming thermal conductivity, making it ideal as a heat dissipation substrate for semiconductor devices. Metals generally have high thermal conductivity and are suitable for thermal management applications.

### Code Example 5: Material Selection Map (Ashby Chart Style)

Simplify the famous Ashby chart from materials science to learn material selection perspectives.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Simplify the famous Ashby chart from materials science to le
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Material data (Young's modulus GPa, density g/cm³)
    materials_ashby = {
        'Metals': {
            'Steel': (200, 7.85),
            'Aluminum': (70, 2.70),
            'Titanium': (110, 4.51),
            'Magnesium': (45, 1.74),
        },
        'Ceramics': {
            'Alumina': (380, 3.95),
            'Silicon carbide': (410, 3.21),
            'Silicon nitride': (310, 3.44),
        },
        'Polymers': {
            'Epoxy': (3, 1.2),
            'Nylon': (2.5, 1.14),
            'PEEK': (4, 1.32),
        },
        'Composites': {
            'CFRP': (150, 1.60),
            'GFRP': (40, 1.80),
        },
        'Natural materials': {
            'Wood': (11, 0.6),
            'Bone': (20, 1.9),
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors_ashby = {
        'Metals': '#1f77b4', 'Ceramics': '#ff7f0e', 'Polymers': '#2ca02c',
        'Composites': '#d62728', 'Natural materials': '#9467bd'
    }
    
    for category, materials_dict in materials_ashby.items():
        youngs_moduli = [v[0] for v in materials_dict.values()]
        densities = [v[1] for v in materials_dict.values()]
        names = list(materials_dict.keys())
    
        ax.scatter(densities, youngs_moduli, s=200, alpha=0.7,
                   color=colors_ashby[category], label=category,
                   edgecolors='black', linewidth=1.5)
    
        # Display material names as labels
        for name, x, y in zip(names, densities, youngs_moduli):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Logarithmic scale
    ax.set_yscale('log')
    ax.set_xlabel('Density (g/cm³)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Young\'s Modulus (GPa)', fontsize=13, fontweight='bold')
    ax.set_title('Material Selection Map (Ashby Chart Style)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, which='both')
    
    # Specific stiffness (Young's modulus/density) guidelines
    x_line = np.linspace(0.5, 8, 100)
    for specific_stiffness in [10, 30, 100]:
        y_line = specific_stiffness * x_line
        ax.plot(x_line, y_line, '--', alpha=0.3, color='gray', linewidth=1)
        ax.text(7, specific_stiffness * 7, f'E/Á={specific_stiffness}',
                fontsize=8, alpha=0.6, rotation=15)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate specific stiffness
    print("\nSpecific stiffness ranking (Young's modulus/density, unit: GPa/(g/cm³)):")
    all_materials_ashby = []
    for category, materials_dict in materials_ashby.items():
        for name, (youngs, density) in materials_dict.items():
            specific_stiffness = youngs / density
            all_materials_ashby.append((name, specific_stiffness, category))
    
    all_materials_ashby.sort(key=lambda x: x[1], reverse=True)
    for i, (name, ss, category) in enumerate(all_materials_ashby[:5], 1):
        print(f"{i}. {name} ({category}): {ss:.1f}")
    
    print("\nMaterial selection guidelines:")
    print("- Lightweight and high stiffness needed ’ CFRP, silicon carbide")
    print("- High-temperature environment ’ Ceramics")
    print("- Conductivity needed ’ Metals")
    print("- Low cost and processability ’ Polymers")
    

**Output example** :
    
    
    Specific stiffness ranking (Young's modulus/density, unit: GPa/(g/cm³)):
    1. Silicon carbide (Ceramics): 127.7
    2. Alumina (Ceramics): 96.2
    3. CFRP (Composites): 93.8
    4. Silicon nitride (Ceramics): 90.1
    5. Titanium (Metals): 24.4
    

**Explanation** : Ashby charts are powerful tools for material selection. Materials with high specific stiffness (Young's modulus/density) are suitable for applications requiring lightweight and high stiffness (such as aircraft structures).

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Definition and classification of materials**
     * Materials are understood through four elements: structure-properties-synthesis-performance
     * Four major classes: metals, ceramics, polymers, composites
     * Each material class has characteristic properties that determine applications
  2. **History of materials science**
     * Human history is the history of materials (Stone Age ’ Bronze Age ’ Iron Age ’ ...)
     * Modern era is the MI/PI age (data-driven materials development)
  3. **Four categories of material properties**
     * Mechanical properties: strength, hardness, ductility
     * Electrical properties: electrical conductivity, band gap
     * Thermal properties: thermal conductivity, melting point
     * Optical properties: refractive index, transmittance
  4. **Relationship with MI/PI**
     * Materials science knowledge is the foundation of MI (materials design) and PI (process optimization)
     * Understanding structure-property relationships is essential for descriptor design
  5. **Data visualization using Python**
     * Material property comparison using radar charts, scatter plots, and logarithmic plots
     * Material selection guidelines using Ashby charts

### Key Points

  * Material selection is key to **optimizing properties according to application**
  * Single materials have limitations, making **composite materials** important
  * Material properties **differ by tens of orders of magnitude** (e.g., electrical conductivity)
  * **Normalized indices** such as specific strength and specific stiffness are useful for material selection
  * Data-driven approaches (MI/PI) accelerate materials development

### To the Next Chapter

In Chapter 2, we will learn about **atomic structure and chemical bonding** :

  * Atomic structure and electron configuration
  * Types of chemical bonding (ionic bonding, covalent bonding, metallic bonding, intermolecular forces)
  * Relationship between bonding and material properties
  * Visualization of electron configuration using Python
  * Calculation of bonding energy and material properties
