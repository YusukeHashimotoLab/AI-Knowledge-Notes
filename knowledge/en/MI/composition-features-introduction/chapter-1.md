---
title: "Chapter 1: Fundamentals of Composition-Based Features"
chapter_title: "Chapter 1: Fundamentals of Composition-Based Features"
subtitle: Principles of Predicting Material Properties from Chemical Composition
---

This chapter covers the fundamentals of Fundamentals of Composition, which based features. You will learn essential concepts and techniques.

### 🎯 Learning Objectives for This Chapter

#### Fundamental Understanding

  * ✅ Explain the definition and role of composition-based features
  * ✅ Understand principles of statistical aggregation (mean, variance, max/min, range)
  * ✅ Understand prediction capabilities and limitations without structural information

#### Practical Skills

  * ✅ Parse chemical formulas and extract elemental information using pymatgen
  * ✅ Generate basic composition-based features using matminer
  * ✅ Utilize elemental property databases to calculate statistics

#### Application Abilities

  * ✅ Apply composition-based features to new material systems
  * ✅ Design prediction workflows integrated with machine learning models
  * ✅ Determine applicable and inappropriate tasks

## 1.1 Principles of Feature Extraction from Chemical Composition

### What Are Composition-Based Features?

In materials discovery, **chemical composition** (types and ratios of elements) is the most fundamental information. For example, the chemical formula "Fe2O3" for iron oxide means "2 iron atoms and 3 oxygen atoms." However, this string cannot be directly input into machine learning models.

**Composition-based features** are techniques for converting chemical composition into **numerical vectors**. Specifically, using periodic table information for elements (atomic radius, ionization energy, electronegativity, etc.), the conversion proceeds as follows:
    
    
    ```mermaid
    graph LR
        A["Chemical FormulaFe₂O₃"] --> B["Element ExtractionFe: 2 atomsO: 3 atoms"]
        B --> C["Elemental Property RetrievalFe: Atomic Radius=1.26Å, IE=7.9eVO: Atomic Radius=0.66Å, IE=13.6eV"]
        C --> D["Statistical AggregationMean Atomic Radius=0.92ÅMean IE=10.1eV..."]
        D --> E["Feature Vector[0.92, 10.1, ...](145 dimensions)"]
    ```

### Comparison of Information Content: Chemical Composition vs Crystal Structure

There are two major approaches to describing materials:

Approach | Required Information | Information Content | Prediction Accuracy | Computation Speed  
---|---|---|---|---  
**Composition-Based** | Chemical formula only (e.g., Fe₂O₃) | Low (~150 dimensions) | Medium (R²=0.7-0.85) | Fast (1 million compounds in 1 second)  
**Structure-Based** (GNN) | Atomic coordinates, bonding information | High (~thousands of dimensions) | High (R²=0.85-0.95) | Slow (1000 compounds in 1 minute)  
  
#### 💡 Three Cases Where Composition-Based Approach is Advantageous

  1. **Discovery of materials with unknown structures** : Crystal structure is unknown for pre-synthesis candidate screening
  2. **High-speed large-scale screening** : Formation energy prediction for 1 million compounds (100x faster than GNN)
  3. **Experimental data-driven approach** : Learning from experimental data where structural information is difficult to obtain

### Types and Materials Science Significance of Elemental Properties

Composition-based features utilize **elemental periodic table databases**. Representative 22 types of elemental properties:

Category | Elemental Property | Materials Science Significance  
---|---|---  
**Atomic Structure** | Atomic Number | Basic element identifier  
Period | Number of electron shells (indicator of chemical reactivity)  
Group | Number of valence electrons (indicator of bonding tendency)  
**Atomic Size** | Atomic Radius | Crystal lattice size, packing fraction  
Covalent Radius | Covalent bond length  
Ionic Radius | Lattice constant of ionic crystals  
Atomic Volume | Crystal density estimation  
**Electronic Properties** | Ionization Energy | Chemical bond strength, reactivity  
Electron Affinity | Oxidation/reduction tendency  
Electronegativity | Bond polarity, ionicity  
Valence Electrons | Type of chemical bonding  
Oxidation State | Charge balance in ionic bonding  
**Thermal & Physical Properties** | Melting Point | Crystal stability, synthesis temperature  
Boiling Point | Volatility  
Density | Crystal packing fraction  
Thermal Conductivity | Heat transport properties  
**Others** | Abundance (in Earth's crust) | Material cost, rarity  
Discovery Year | Historical background of element  
Specific Heat Capacity | Heat capacity estimation  
Electrical Resistivity | Conductivity indicator  
Magnetic Moment | Magnetic material design  
Polarizability | Dielectric properties  
  
### Statistical Aggregation Methods (Mean, Variance, Max/Min, Range)

For compounds consisting of multiple elements (e.g., Fe₂O₃), properties of each element are **statistically aggregated**. Representative 6 statistics:

  1. **Mean** : $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} w_i x_i$ (weight $w_i$ is composition ratio)
  2. **Variance** : $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} w_i (x_i - \bar{x})^2$
  3. **Standard Deviation** : $\sigma = \sqrt{\sigma^2}$
  4. **Maximum** : $\max(x_1, x_2, \ldots, x_n)$
  5. **Minimum** : $\min(x_1, x_2, \ldots, x_n)$
  6. **Range** : $\text{range} = \max - \min$

#### 🔍 Concrete Example: Atomic Radius Statistics for Fe₂O₃

  * Fe atomic radius: 1.26 Å (composition ratio: 2/5 = 0.4)
  * O atomic radius: 0.66 Å (composition ratio: 3/5 = 0.6)

**Calculation Results** :

  * Mean: $0.4 \times 1.26 + 0.6 \times 0.66 = 0.900$ Å
  * Variance: $0.4(1.26-0.90)^2 + 0.6(0.66-0.90)^2 = 0.086$ Ų
  * Max: 1.26 Å, Min: 0.66 Å, Range: 0.60 Å

### Code Example 1: Basic Operations with pymatgen Composition Class

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example1.ipynb>)
    
    
    # Parse chemical formula and extract elemental information
    from pymatgen.core import Composition
    
    # Create chemical formula
    comp = Composition("Fe2O3")
    
    # Retrieve basic information
    print(f"Chemical Formula: {comp}")
    print(f"Element Types: {comp.elements}")  # [Element Fe, Element O]
    print(f"Total Atoms: {comp.num_atoms}")  # 5.0
    print(f"Total Weight: {comp.weight:.2f} g/mol")  # 159.69 g/mol
    
    # Composition ratio for each element
    print("\nComposition ratio by element:")
    for element, fraction in comp.get_atomic_fraction().items():
        print(f"  {element}: {fraction:.3f} ({fraction*comp.num_atoms:.0f} atoms)")
    # Output:
    #   Fe: 0.400 (2 atoms)
    #   O: 0.600 (3 atoms)
    
    # Check fractional notation
    print(f"\nBefore reduction: {Composition('Fe4O6')}")  # Fe4 O6
    print(f"After reduction: {Composition('Fe4O6').reduced_composition}")  # Fe2 O3

### Code Example 2: Elemental Property Extraction and Visualization

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example2.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 2: Elemental Property Extraction and Visualizat
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Extract elemental properties and visualize with matplotlib
    import matplotlib.pyplot as plt
    from pymatgen.core import Element
    
    # Compare atomic radii of multiple elements
    elements = [Element("Fe"), Element("O"), Element("Cu"), Element("Si")]
    properties = {
        "Atomic Radius (Å)": [el.atomic_radius for el in elements],
        "Ionization Energy (eV)": [el.ionization_energy for el in elements],
        "Electronegativity (Pauling)": [el.X for el in elements]
    }
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    element_names = [el.symbol for el in elements]
    
    for ax, (prop_name, values) in zip(axes, properties.items()):
        ax.bar(element_names, values, color=['#f093fb', '#f5576c', '#feca57', '#48dbfb'])
        ax.set_ylabel(prop_name, fontsize=12)
        ax.set_title(f"Elemental Property Comparison: {prop_name}", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('element_properties.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Numerical output
    print("Elemental Property Data:")
    for prop_name, values in properties.items():
        print(f"\n{prop_name}:")
        for el, val in zip(element_names, values):
            print(f"  {el}: {val:.3f}")

### Code Example 3: Statistics Calculation (Mean, Standard Deviation, Range)

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example3.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # Manual calculation of composition-based statistics
    import numpy as np
    from pymatgen.core import Composition, Element
    
    def compute_weighted_stats(comp, property_name):
        """Calculate statistics weighted by composition ratio"""
        # Extract elements and composition ratios
        fractions = comp.get_atomic_fraction()
    
        # Retrieve elemental properties
        values = []
        weights = []
        for element, frac in fractions.items():
            prop_value = getattr(Element(element), property_name)
            values.append(prop_value)
            weights.append(frac)
    
        values = np.array(values)
        weights = np.array(weights)
    
        # Calculate statistics
        mean = np.sum(weights * values)
        variance = np.sum(weights * (values - mean)**2)
        std = np.sqrt(variance)
    
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'max': values.max(),
            'min': values.min(),
            'range': values.max() - values.min()
        }
    
    # Calculate atomic radius statistics for Fe2O3
    comp = Composition("Fe2O3")
    stats = compute_weighted_stats(comp, 'atomic_radius')
    
    print("Atomic Radius Statistics for Fe2O3:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value:.4f} Å")
    
    # Example output:
    #   mean: 0.9000 Å
    #   std: 0.2933 Å
    #   variance: 0.0860 Ų
    #   max: 1.2600 Å
    #   min: 0.6600 Å
    #   range: 0.6000 Å

## 1.2 Elemental Periodic Table and Material Properties

### Periodicity and Correlation with Material Properties

The periodic table is based on the periodic law, which states that **elemental properties change periodically**. This periodicity directly contributes to predicting material properties:
    
    
    ```mermaid
    graph TD
        A[Periodic Table] --> B[Period]
        A --> C[Group]
        B --> D[Number of Electron Shells→ Atomic Size]
        C --> E[Valence Electrons→ Bonding Nature]
        D --> F[Crystal Lattice ConstantDensity]
        E --> G[IonicityCovalency]
        F --> H[Material PropertiesFormation EnergyBand Gap]
        G --> H
    ```

#### Effect of Period

  * **As period increases** (top to bottom): 
    * Atomic radius increases → Crystal lattice becomes larger
    * Ionization energy decreases → Reactivity becomes higher
    * Metallic character increases → Conductivity improves

#### Effect of Group

  * **As group number increases** (left to right): 
    * Valence electron count increases → Diversity of oxidation states
    * Electronegativity increases → Polarity of ionic bonding
    * Non-metallic character increases → Insulating materials

### Trends in Element Groups (Transition Metals, Halogens, etc.)

Element Group | Representative Elements | Characteristic Properties | Material Application Examples  
---|---|---|---  
**Alkali Metals**  
(Group 1) | Li, Na, K | ・Low ionization energy  
・High reactivity  
・Lightweight | Lithium-ion batteries  
Sodium-ion batteries  
**Transition Metals**  
(Groups 3-12) | Fe, Co, Ni, Cu | ・Multiple oxidation states  
・d-orbital electrons  
・Magnetism | Catalysts, magnetic materials  
Structural materials  
**Halogens**  
(Group 17) | F, Cl, Br, I | ・High electronegativity  
・Strong oxidizing power  
・Ionic bond formation | Perovskites  
Halide electrolytes  
**Rare Earth Elements**  
(Lanthanides) | La, Ce, Nd, Gd | ・4f orbital electrons  
・Magnetic moment  
・Fluorescence properties | Permanent magnets, phosphors  
Catalysts  
**Metalloids**  
(Groups 13-16 boundary) | Si, Ge, As | ・Intermediate between metals and non-metals  
・Adjustable band gap | Semiconductors, solar cells  
Thermoelectric materials  
  
### Structural Chemistry Background (Bond Types, Coordination Number)

Elemental properties directly influence **chemical bond types** and **coordination structures** :

#### Relationship Between Bond Types and Elemental Properties

  1. **Ionic Bonding** : 
     * Between elements with large electronegativity difference (e.g., Na-Cl, Ca-O)
     * Ionicity dominates when electronegativity difference $\Delta X > 1.7$
     * Prediction: Large formation energy, hard, insulating
  2. **Covalent Bonding** : 
     * Between elements with similar electronegativity (e.g., Si-Si, C-C)
     * Strong covalent bonding when $\Delta X < 0.5$
     * Prediction: Directional bonding, semiconductor properties
  3. **Metallic Bonding** : 
     * Between metallic elements (e.g., Fe-Fe, Cu-Cu)
     * Sharing of free electrons
     * Prediction: High conductivity, malleability and ductility

#### Coordination Number and Atomic Radius Ratio

In ionic crystals, the **coordination number** (number of ions around the central ion) is determined by the atomic radius ratio $r_{\text{cation}}/r_{\text{anion}}$:

Radius Ratio | Coordination Number | Coordination Structure | Example  
---|---|---|---  
0.225 - 0.414 | 4 | Tetrahedral | ZnS (Zinc blende)  
0.414 - 0.732 | 6 | Octahedral | NaCl (Rock salt)  
0.732 - 1.000 | 8 | Cubic | CsCl (Cesium chloride)  
> 1.000 | 12 | Close-packed | Metallic crystals  
  
#### 🔬 Real Example: Predicting Coordination Structure of TiO₂

  * Ti⁴⁺ ionic radius: 0.605 Å
  * O²⁻ ionic radius: 1.40 Å
  * Radius ratio: $0.605 / 1.40 = 0.432$
  * **Prediction** : Coordination number 6 (octahedral structure)
  * **Reality** : Rutile-type TiO₂ indeed has Ti⁴⁺ in 6-fold coordination

### Code Example 4: Generating Composition-Based Feature Vectors

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example4.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: Generating Composition-Based Feature Vectors
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # Generate feature vectors from composition using matminer
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # Compound list
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO"]
    
    # ElementProperty featurizer (simplified version, 3 statistics only)
    featurizer = ElementProperty.from_preset("magpie")
    
    # Create dataframe
    df = pd.DataFrame({"composition": compounds})
    
    # Generate features (may take some time)
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Display part of generated features
    feature_cols = [col for col in df.columns if col != "composition"]
    print(f"Number of generated features: {len(feature_cols)}")
    print(f"\nFirst 5 features:")
    print(df[feature_cols[:5]].head())
    
    # Feature name examples
    print(f"\nFeature name examples:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {i+1}. {col}")
    
    # Example output:
    # Number of generated features: 145
    # Feature name examples:
    #   1. MagpieData mean Number
    #   2. MagpieData avg_dev Number
    #   3. MagpieData range Number
    #   ...

### Code Example 5: Periodic Table Mapping (seaborn heatmap)

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example5.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 5: Periodic Table Mapping (seaborn heatmap)
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Visualize periodic table with heatmap for specific elemental properties
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pymatgen.core import Element
    
    # Part of periodic table (major elements)
    periods = {
        2: ["Li", "Be", "B", "C", "N", "O", "F"],
        3: ["Na", "Mg", "Al", "Si", "P", "S", "Cl"],
        4: ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    }
    
    # Create heatmap data for electronegativity
    max_cols = max(len(row) for row in periods.values())
    heatmap_data = []
    yticks = []
    
    for period_num, elements in sorted(periods.items()):
        row = []
        for el_symbol in elements:
            el = Element(el_symbol)
            row.append(el.X if el.X is not None else 0)
        # Pad with zeros
        row.extend([0] * (max_cols - len(row)))
        heatmap_data.append(row)
        yticks.append(f"Period {period_num}")
    
    # Draw heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, annot=False, cmap="RdYlBu_r",
                cbar_kws={'label': 'Electronegativity (Pauling)'},
                yticklabels=yticks)
    plt.title("Periodic Table: Electronegativity Heatmap", fontsize=16, fontweight='bold')
    plt.xlabel("Elements (left to right)", fontsize=12)
    plt.ylabel("Period", fontsize=12)
    plt.tight_layout()
    plt.savefig('periodic_table_electronegativity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Trends observable from the heatmap:")
    print("- Top right (F, O, Cl) have high electronegativity (red)")
    print("- Bottom left (Li, Na, K) have low electronegativity (blue)")
    print("- Within the same group (vertical), decreases as you go down")

## 1.3 Success Cases of Composition-Based Prediction

### OQMD/Materials Project Formation Energy Prediction (R² ≥ 0.8)

The most successful application of composition-based features is **formation energy prediction**. Ward et al. (2016) in the Magpie paper reported the following achievements:

Dataset | Sample Size | Model | R² Score | MAE  
---|---|---|---|---  
OQMD (All compounds) | 435,000 | Random Forest | 0.92 | 0.10 eV/atom  
Materials Project | 60,000 | Gradient Boosting | 0.89 | 0.12 eV/atom  
OQMD (Oxides only) | 50,000 | Random Forest | 0.94 | 0.08 eV/atom  
  
#### 📊 What is Formation Energy?

Formation energy is the energy change when a compound is formed from constituent elements in their standard states:

$$\Delta H_f = E_{\text{compound}} - \sum_i n_i E_{\text{element}_i}$$

  * **Negative value** : Compound is stable (spontaneously forms)
  * **Positive value** : Compound is unstable (difficult to form)

In materials discovery, compounds with negative formation energy and large absolute values are prioritized as easier to synthesize.

### Band Gap Prediction (MAE < 0.5 eV)

The **band gap** , important for semiconductor material design, can also be predicted with composition-based features:

Study | Dataset | Sample Size | MAE | R²  
---|---|---|---|---  
Ward+ (2016) | Materials Project | 10,000 | 0.45 eV | 0.78  
Jha+ (2018) ElemNet | OQMD | 28,000 | 0.38 eV | 0.83  
Meredig+ (2014) | Experimental Data | 1,200 | 0.62 eV | 0.65  
  
**Caveat** : Band gap is a property with **strong structure dependence**. In comparison with DFT calculated values, composition-based approaches have limitations, and GNN (structure-based) methods achieve higher accuracy (MAE < 0.25 eV).

### Thermoelectric Property Prediction

Successful prediction of the **ZT value** , a performance indicator for thermoelectric materials (converting heat to electricity):

  * **Carrete et al. (2014)** : ZT prediction for half-Heusler compounds (R² = 0.72)
  * **Feature importance** : 
    1. Mean atomic mass (affects thermal conductivity)
    2. Variance of electronegativity (affects carrier mobility)
    3. Mean valence electron count (affects electron concentration)

### Code Example 6: Composition Normalization Processing

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example6.ipynb>)
    
    
    # Normalize composition formulas to per-atom basis
    from pymatgen.core import Composition
    
    # Various notations of composition formulas
    formulas = ["Fe4O6", "Fe2O3", "Fe0.5O0.75", "FeO1.5"]
    
    print("Composition Formula Normalization:")
    for formula in formulas:
        comp = Composition(formula)
        reduced = comp.reduced_composition
        fractional = comp.fractional_composition
    
        print(f"\nOriginal formula: {formula}")
        print(f"  After reduction (integer ratio): {reduced}")
        print(f"  Per atom: {fractional}")
        print(f"  Total atoms: {comp.num_atoms:.2f}")
        print(f"  Fe/O ratio: {comp['Fe']/comp['O']:.3f}")
    
    # Example output:
    # Original formula: Fe4O6
    #   After reduction (integer ratio): Fe2 O3
    #   Per atom: Fe0.4 O0.6
    #   Total atoms: 10.00
    #   Fe/O ratio: 0.667

## 1.4 Why Composition-Based? Prediction Capability Without Structural Information

### Advantages of High-Speed Screening

The greatest advantage of composition-based features is **computational speed**. Comparison with structure-based (GNN) approaches:

Task | Composition-Based (Magpie + RF) | Structure-Based (CGCNN) | Speed Ratio  
---|---|---|---  
Feature generation (1 compound) | 0.001 seconds | 0.1 seconds | 100x faster  
Inference (1 million compounds) | 10 minutes | 27 hours | 162x faster  
Model training (100k samples) | 5 minutes | 60 minutes | 12x faster  
  
#### ⚡ Practical Example of High-Speed Screening

**Scenario** : From 1 million candidate compounds, find stable compounds with formation energy ≤ -2 eV/atom

  * **Composition-based** : Complete in 10 minutes → Select 50,000 compounds → Proceed to experimental validation
  * **Structure-based (GNN)** : Takes 27 hours → Not practical

**Strategy** : Narrow down candidates with composition-based approach, then precise prediction with GNN (hybrid approach)

### Application to Materials with Unknown Crystal Structure

In new materials discovery, there are many cases where **crystal structure is unknown** :

  1. **Pre-synthesis candidate compounds** : Composition can be determined, but structure is unknown 
     * Example: Composition search for Li-Ni-Mn-Co-O battery cathode materials
     * Narrow down candidates with composition-based → Synthesize → Structural analysis
  2. **Metastable phases** : Structure prediction by DFT calculation is difficult 
     * Example: High-pressure synthesis materials, rapid quenching materials
     * Predict property trends from composition
  3. **Amorphous materials** : No long-range order 
     * Example: Metallic glasses, oxide glasses
     * Composition is the only descriptor

### Compatibility with Experimental Data

For experimental researchers, **composition information is the most easily obtainable** data:

Information Type | Experimental Acquisition Difficulty | Accuracy | Cost  
---|---|---|---  
Chemical Composition | Low (EDX, ICP-MS) | High (±1%) | Low (few thousand yen/sample)  
Crystal Structure | Medium (XRD, single crystal analysis) | Medium (Rietveld analysis required) | Medium (tens of thousands of yen/sample)  
Atomic Coordinates (Precise) | High (Single crystal XRD, neutron diffraction) | High (Å precision) | High (hundreds of thousands of yen/sample)  
  
**Typical workflow for experimental data-driven materials discovery** :
    
    
    ```mermaid
    graph LR
        A[Measure composition experimentally] --> B[Generate composition-based features]
        B --> C[Train machine learning model]
        C --> D[Predict properties of new compositions]
        D --> E[Experimental validation]
        E --> F{Performance improved?}
        F -->|Yes| G[Next-generation composition optimization]
        F -->|No| C
        G --> E
    ```

### Code Example 7: Feature Correlation Analysis (pandas, seaborn)

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example7.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 7: Feature Correlation Analysis (pandas, seabor
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Analyze correlations between features and identify redundant features
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matminer.featurizers.composition import ElementProperty
    
    # Sample data
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO", "ZnO", "MgO", "CaO"]
    df = pd.DataFrame({"composition": compounds})
    
    # Feature generation (simplified version, statistics only)
    featurizer = ElementProperty(features=["Number", "AtomicWeight", "Row", "Column"],
                                  stats=["mean", "std", "range"])
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Extract feature columns only
    feature_cols = [col for col in df.columns if col != "composition"]
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Draw heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Composition-Based Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Detect highly correlated pairs (threshold: |r| > 0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    print("\nHighly correlated feature pairs (|r| > 0.9):")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1} ↔ {feat2}: r = {corr_val:.3f}")

### Code Example 8: Simple Linear Regression Model Application (scikit-learn)

[Open in Google Colab](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example8.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 8: Simple Linear Regression Model Application (
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # Predict formation energy with composition-based features (simulation data)
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    import matplotlib.pyplot as plt
    
    # Simulation data (in reality, obtained from Materials Project, etc.)
    np.random.seed(42)
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO", "ZnO", "MgO", "CaO",
                 "NiO", "CoO", "MnO", "V2O5", "Cr2O3", "SnO2", "In2O3", "Ga2O3"]
    formation_energies = [-2.5, -3.1, -3.8, -2.9, -1.5, -1.8, -2.3, -2.7,
                          -1.4, -1.6, -1.9, -3.5, -2.8, -2.4, -2.1, -2.6]  # eV/atom (fictional values)
    
    # Create dataframe
    df = pd.DataFrame({"composition": compounds, "formation_energy": formation_energies})
    
    # Generate features
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ["composition", "formation_energy"]]
    X = df[feature_cols].values
    y = df["formation_energy"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MAE: {mae:.3f} eV/atom")
    print(f"  R²: {r2:.3f}")
    
    # Plot prediction vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='#f5576c', s=100, alpha=0.7, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Value (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Value (eV/atom)', fontsize=12)
    plt.title(f'Formation Energy Prediction (Linear Regression)\nMAE={mae:.3f}, R²={r2:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_regression_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Top 5 important features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())

## Exercises

### Easy (Fundamental Level)

Exercise 1-1: Chemical Formula Analysis

**Problem** : Using pymatgen, extract the following from the chemical formula "Li3Fe2(PO4)3":

  1. Element types and counts
  2. Total number of atoms
  3. Total weight (g/mol)
  4. Composition ratio (atomic fraction) for each element

**Sample Solution** :
    
    
    from pymatgen.core import Composition
    
    comp = Composition("Li3Fe2(PO4)3")
    
    # 1. Element types and counts
    print("Elements and counts:")
    for el, count in comp.get_el_amt_dict().items():
        print(f"  {el}: {count} atoms")
    
    # 2. Total atoms
    print(f"\nTotal atoms: {comp.num_atoms}")
    
    # 3. Total weight
    print(f"Total weight: {comp.weight:.2f} g/mol")
    
    # 4. Composition ratios
    print("\nComposition ratios (atomic fractions):")
    for el, frac in comp.get_atomic_fraction().items():
        print(f"  {el}: {frac:.4f}")
    
    # Output:
    # Elements and counts:
    #   Li: 3 atoms
    #   Fe: 2 atoms
    #   P: 3 atoms
    #   O: 12 atoms
    # Total atoms: 20.0
    # Total weight: 397.48 g/mol
    # Composition ratios:
    #   Li: 0.1500
    #   Fe: 0.1000
    #   P: 0.1500
    #   O: 0.6000

Exercise 1-2: Basic Statistics Calculation

**Problem** : Calculate the mean, standard deviation, and range of atomic radius for the chemical formula "NaCl" (weighted by composition ratio).

**Hint** : Na atomic radius=1.86Å, Cl atomic radius=1.75Å, composition ratio is 1:1

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    from pymatgen.core import Composition, Element
    
    comp = Composition("NaCl")
    fractions = comp.get_atomic_fraction()
    
    # Atomic radius data
    radii = []
    weights = []
    for el, frac in fractions.items():
        radii.append(Element(el).atomic_radius)
        weights.append(frac)
    
    radii = np.array(radii)
    weights = np.array(weights)
    
    # Calculate statistics
    mean = np.sum(weights * radii)
    variance = np.sum(weights * (radii - mean)**2)
    std = np.sqrt(variance)
    range_val = radii.max() - radii.min()
    
    print(f"Atomic Radius Statistics for NaCl:")
    print(f"  Mean: {mean:.4f} Å")
    print(f"  Standard Deviation: {std:.4f} Å")
    print(f"  Range: {range_val:.4f} Å")
    
    # Output:
    # Atomic Radius Statistics for NaCl:
    #   Mean: 1.8050 Å
    #   Standard Deviation: 0.0550 Å
    #   Range: 0.1100 Å

Exercise 1-3: Element Count

**Problem** : Count the number of element types in the following chemical formulas:

  * Fe2O3
  * CaTiO3
  * Li(Ni0.8Co0.15Al0.05)O2

**Sample Solution** :
    
    
    from pymatgen.core import Composition
    
    formulas = ["Fe2O3", "CaTiO3", "Li(Ni0.8Co0.15Al0.05)O2"]
    
    for formula in formulas:
        comp = Composition(formula)
        num_elements = len(comp.elements)
        print(f"{formula}: {num_elements} element types")
        print(f"  Elements: {[el.symbol for el in comp.elements]}\n")
    
    # Output:
    # Fe2O3: 2 element types
    #   Elements: ['Fe', 'O']
    # CaTiO3: 3 element types
    #   Elements: ['Ca', 'Ti', 'O']
    # Li(Ni0.8Co0.15Al0.05)O2: 5 element types
    #   Elements: ['Li', 'Ni', 'Co', 'Al', 'O']

### Medium (Intermediate Level)

Exercise 1-4: Feature Generation with matminer

**Problem** : Using matminer's ElementProperty featurizer, generate features for the following compounds:

  * Compounds: ["BaTiO3", "SrTiO3", "PbTiO3"]
  * Elemental properties: Number, AtomicWeight, Row
  * Statistics: mean, std, range

Display the number of generated features and the first 3 features.

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from matminer.featurizers.composition import ElementProperty
    
    # Data preparation
    compounds = ["BaTiO3", "SrTiO3", "PbTiO3"]
    df = pd.DataFrame({"composition": compounds})
    
    # Featurizer setup
    featurizer = ElementProperty(features=["Number", "AtomicWeight", "Row"],
                                  stats=["mean", "std", "range"])
    
    # Feature generation
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Display results
    feature_cols = [col for col in df.columns if col != "composition"]
    print(f"Number of generated features: {len(feature_cols)}")
    print(f"\nFirst 3 features:")
    print(df[feature_cols[:3]])
    
    # Example output:
    # Number of generated features: 9
    # First 3 features:
    #    mean Number  std Number  range Number
    # 0    30.4      23.35         48.0
    # 1    29.2      22.41         46.0
    # 2    41.4      27.93         74.0

Exercise 1-5: Periodic Table Visualization

**Problem** : Visualize ionization energy for the following element groups using bar charts:

  * Alkali metals: Li, Na, K, Rb, Cs
  * Halogens: F, Cl, Br, I

Display both groups side by side for comparison.

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    from pymatgen.core import Element
    
    # Element groups
    alkali = ["Li", "Na", "K", "Rb", "Cs"]
    halogens = ["F", "Cl", "Br", "I"]
    
    # Get ionization energies
    alkali_ie = [Element(el).ionization_energy for el in alkali]
    halogen_ie = [Element(el).ionization_energy for el in halogens]
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(alkali, alkali_ie, color='#f093fb', edgecolor='black')
    axes[0].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[0].set_title('Ionization Energy of Alkali Metals', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(halogens, halogen_ie, color='#f5576c', edgecolor='black')
    axes[1].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[1].set_title('Ionization Energy of Halogens', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ionization_energy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Observations:")
    print("- Alkali metals: Decreases with increasing period (K < Na < Li)")
    print("- Halogens: F is maximum, I is minimum")

Exercise 1-6: Correlation Analysis

**Problem** : Calculate correlation coefficients between the following elemental property pairs:

  * Atomic radius vs Ionization energy
  * Electronegativity vs Ionization energy

Target elements: Period 2 elements (Li, Be, B, C, N, O, F)

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from pymatgen.core import Element
    import matplotlib.pyplot as plt
    
    # Period 2 elements
    elements = ["Li", "Be", "B", "C", "N", "O", "F"]
    
    # Extract data
    atomic_radius = []
    ionization_energy = []
    electronegativity = []
    
    for el_symbol in elements:
        el = Element(el_symbol)
        atomic_radius.append(el.atomic_radius)
        ionization_energy.append(el.ionization_energy)
        electronegativity.append(el.X)
    
    # Convert to NumPy arrays
    ar = np.array(atomic_radius)
    ie = np.array(ionization_energy)
    en = np.array(electronegativity)
    
    # Calculate correlation coefficients
    corr_ar_ie = np.corrcoef(ar, ie)[0, 1]
    corr_en_ie = np.corrcoef(en, ie)[0, 1]
    
    print(f"Correlation coefficients:")
    print(f"  Atomic radius vs Ionization energy: {corr_ar_ie:.3f}")
    print(f"  Electronegativity vs Ionization energy: {corr_en_ie:.3f}")
    
    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(ar, ie, s=100, color='#f093fb', edgecolors='black')
    for i, el in enumerate(elements):
        axes[0].annotate(el, (ar[i], ie[i]), fontsize=12, ha='right')
    axes[0].set_xlabel('Atomic Radius (Å)', fontsize=12)
    axes[0].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[0].set_title(f'Correlation: {corr_ar_ie:.3f}', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(en, ie, s=100, color='#f5576c', edgecolors='black')
    for i, el in enumerate(elements):
        axes[1].annotate(el, (en[i], ie[i]), fontsize=12, ha='right')
    axes[1].set_xlabel('Electronegativity (Pauling)', fontsize=12)
    axes[1].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[1].set_title(f'Correlation: {corr_en_ie:.3f}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example output:
    # Correlation coefficients:
    #   Atomic radius vs Ionization energy: -0.985 (strong negative correlation)
    #   Electronegativity vs Ionization energy: 0.992 (strong positive correlation)

Exercise 1-7: Simple Model Application

**Problem** : Train a linear regression model with the following fictional data to predict band gaps:
    
    
    Compounds: ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    Band gaps (eV): [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
                

Evaluate MAE and R² with train/test split (80/20).

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    
    # Data preparation
    compounds = ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    bandgaps = [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
    
    df = pd.DataFrame({"composition": compounds, "bandgap": bandgaps})
    
    # Feature generation
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ["composition", "bandgap"]]
    X = df[feature_cols].values
    y = df["bandgap"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MAE: {mae:.3f} eV")
    print(f"  R²: {r2:.3f}")
    print(f"\nTest data predictions:")
    for i, (true_val, pred_val) in enumerate(zip(y_test, y_pred)):
        print(f"  Actual: {true_val:.1f} eV, Predicted: {pred_val:.1f} eV")

### Hard (Advanced Level)

Exercise 1-8: Feature Design for Multicomponent Materials

**Problem** : For the high-entropy alloy (HEA) "CoCrFeNiMn" (equimolar ratio), calculate the following:

  1. Mean, standard deviation, and range of atomic radius
  2. Mean, standard deviation, and range of electronegativity
  3. Mean, standard deviation, and range of valence electron count

Visualize these statistics and discuss implications for HEA design.

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pymatgen.core import Composition, Element
    
    # High-entropy alloy (equimolar ratio)
    comp = Composition("CoCrFeNiMn")
    fractions = comp.get_atomic_fraction()
    
    # Extract elemental properties
    properties = {
        'atomic_radius': [],
        'X': [],  # Electronegativity
        'nvalence': []  # Valence electrons
    }
    
    for el in comp.elements:
        properties['atomic_radius'].append(Element(el).atomic_radius)
        properties['X'].append(Element(el).X)
        properties['nvalence'].append(Element(el).nvalence)
    
    # Calculate statistics
    stats_results = {}
    for prop_name, values in properties.items():
        values = np.array(values)
        stats_results[prop_name] = {
            'mean': values.mean(),
            'std': values.std(),
            'range': values.max() - values.min()
        }
    
    # Display results
    print("Elemental Property Statistics for CoCrFeNiMn:\n")
    for prop_name, stats in stats_results.items():
        print(f"{prop_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
        print()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    prop_labels = ['Atomic Radius (Å)', 'Electronegativity', 'Valence Electrons']
    
    for ax, (prop_name, prop_label) in zip(axes, zip(properties.keys(), prop_labels)):
        values = properties[prop_name]
        stats = stats_results[prop_name]
    
        ax.bar(['Co', 'Cr', 'Fe', 'Ni', 'Mn'], values,
               color=['#f093fb', '#f5576c', '#feca57', '#48dbfb', '#00d2d3'],
               edgecolor='black')
        ax.axhline(stats['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_ylabel(prop_label, fontsize=12)
        ax.set_title(f'{prop_label}\nMean={stats["mean"]:.3f}, Std={stats["std"]:.3f}',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hea_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Implications for HEA design
    print("Implications for HEA design:")
    print("- Small standard deviation in atomic radius → Low lattice strain → High phase stability")
    print("- Small standard deviation in electronegativity → Uniform chemical affinity → Easy solid solution formation")
    print("- Appropriate mean valence electron count → Affects metallic bond strength")

Exercise 1-9: Cross-Validation

**Problem** : Using the data from Exercise 1-7, evaluate model generalization performance with 5-fold cross-validation. Display MAE and R² for each fold, and report mean ± standard deviation.

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold
    from matminer.featurizers.composition import ElementProperty
    
    # Data preparation
    compounds = ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    bandgaps = [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
    df = pd.DataFrame({"composition": compounds, "bandgap": bandgaps})
    
    # Feature generation
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Features and target
    feature_cols = [col for col in df.columns if col not in ["composition", "bandgap"]]
    X = df[feature_cols].values
    y = df["bandgap"].values
    
    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    
    # MAE evaluation
    mae_scores = -cross_val_score(model, X, y, cv=kfold,
                                   scoring='neg_mean_absolute_error')
    
    # R² evaluation
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    # Display results
    print("5-fold Cross-Validation Results:\n")
    print("MAE for each fold (eV):")
    for i, mae in enumerate(mae_scores):
        print(f"  Fold {i+1}: {mae:.3f}")
    print(f"Mean MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}\n")
    
    print("R² for each fold:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Fold {i+1}: {r2:.3f}")
    print(f"Mean R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

Exercise 1-10: Model Evaluation (Comprehensive Problem)

**Problem** : Build a formation energy prediction model using composition-based features with the following fictional OQMD data:
    
    
    Compounds: ["Li2O", "Na2O", "K2O", "MgO", "CaO", "SrO", "Al2O3", "Ga2O3", "In2O3", "TiO2"]
    Formation energies (eV/atom): [-2.9, -2.6, -2.3, -3.0, -3.2, -3.1, -3.5, -2.8, -2.5, -4.1]
                

  1. Generate features (Magpie preset)
  2. Train/test split (70/30)
  3. Train with Random Forest model (n_estimators=100)
  4. Evaluate on test set: MAE, RMSE, R²
  5. Display top 5 feature importances

**Sample Solution** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Sample Solution:
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    import matplotlib.pyplot as plt
    
    # Data preparation
    compounds = ["Li2O", "Na2O", "K2O", "MgO", "CaO", "SrO",
                 "Al2O3", "Ga2O3", "In2O3", "TiO2"]
    formation_energies = [-2.9, -2.6, -2.3, -3.0, -3.2, -3.1,
                          -3.5, -2.8, -2.5, -4.1]
    
    df = pd.DataFrame({"composition": compounds, "formation_energy": formation_energies})
    
    # Feature generation
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # Features and target
    feature_cols = [col for col in df.columns if col not in ["composition", "formation_energy"]]
    X = df[feature_cols].values
    y = df["formation_energy"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation Results:")
    print(f"  MAE: {mae:.3f} eV/atom")
    print(f"  RMSE: {rmse:.3f} eV/atom")
    print(f"  R²: {r2:.3f}\n")
    
    # Top 5 feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 5 Feature Importances:")
    print(feature_importance.head())
    
    # Prediction vs actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, s=100, color='#f5576c', alpha=0.7, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Value (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Value (eV/atom)', fontsize=12)
    plt.title(f'Formation Energy Prediction (Random Forest)\nMAE={mae:.3f}, R²={r2:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

## Summary

In this chapter, we learned the fundamentals of **composition-based features** :

  * ✅ Principles of converting chemical composition to numerical vectors
  * ✅ Types of elemental properties (atomic radius, ionization energy, electronegativity, etc.)
  * ✅ Statistical aggregation methods (mean, variance, max/min, range)
  * ✅ Implementation with pymatgen/matminer
  * ✅ Success stories (OQMD formation energy prediction, band gap prediction)
  * ✅ Prediction capabilities without structural information and advantages of high-speed screening

#### 🎓 Learning Objectives Achievement Check

Can you answer the following questions?

  * Can you explain what composition-based features are in 3 sentences?
  * Can you manually calculate atomic radius statistics for Fe₂O₃?
  * Can you determine whether composition-based or GNN is appropriate?
  * Can you generate a 145-dimensional vector from a chemical formula using matminer?

**If all are Yes** , proceed to Chapter 2 (Magpie Details)!

## References

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028. <https://doi.org/10.1038/npjcompumats.2016.28> (Original Magpie descriptor paper, pp. 1-7)
  2. Jha, D., Ward, L., Paul, A., Liao, W., Choudhary, A., Wolverton, C., & Agrawal, A. (2018). "ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition." _Scientific Reports_ , 8, 17593. <https://doi.org/10.1038/s41598-018-35934-y> (High-accuracy prediction from composition only, pp. 1-13)
  3. Ong, S.P., Richards, W.D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V.L., Persson, K.A., & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319. <https://doi.org/10.1016/j.commatsci.2012.10.028> (Pymatgen library foundation, pp. 314-319)
  4. Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N.E.R., Bajaj, S., Wang, Q., Montoya, J., Chen, J., Bystrom, K., Dylla, M., Chard, K., Asta, M., Persson, K.A., Snyder, G.J., Foster, I., & Jain, A. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69. <https://doi.org/10.1016/j.commatsci.2018.05.018> (Original matminer paper, feature generation toolkit, pp. 60-69)
  5. Meredig, B., Agrawal, A., Kirklin, S., Saal, J.E., Doak, J.W., Thompson, A., Zhang, K., Choudhary, A., & Wolverton, C. (2014). "Combinatorial screening for new materials in unconstrained composition space with machine learning." _Physical Review B_ , 89(9), 094104. <https://doi.org/10.1103/PhysRevB.89.094104> (Empirical study on composition space exploration, pp. 1-7)
  6. Himanen, L., Jäger, M.O.J., Morooka, E.V., Federici Canova, F., Ranawat, Y.S., Gao, D.Z., Rinke, P., & Foster, A.S. (2019). "DScribe: Library of descriptors for machine learning in materials science." _Computer Physics Communications_ , 247, 106949. <https://doi.org/10.1016/j.cpc.2019.106949> (Comprehensive review of descriptor libraries, pp. 1-15)
  7. Materials Project Documentation: matminer module. <https://docs.materialsproject.org/> (Official matminer documentation and usage examples)

[← Series Contents](<./index.html>) [Chapter 2: Magpie and Statistical Descriptors →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
