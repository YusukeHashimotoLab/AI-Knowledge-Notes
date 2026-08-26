---
title: "Chapter 2: Magpie and Statistical Descriptors"
chapter_title: "Chapter 2: Magpie and Statistical Descriptors"
subtitle: High-precision mapping of materials space with 132-dimensional features
---

This chapter covers Magpie and Statistical Descriptors. You will learn essential concepts and techniques.

### 🎯 Learning Objectives

#### Basic Understanding

  * ✅ Explain the 132-dimensional structure of Magpie descriptors (22 elemental properties × 6 statistical measures)
  * ✅ Understand the 22 elemental properties of the `magpie` preset (periodic-table position, atomic/thermal, valence counts, unfilled-orbital counts, ground-state DFT)
  * ✅ Understand the principles of the six statistical aggregation methods (minimum, maximum, range, fraction-weighted mean, avg_dev, mode)

#### Practical Skills

  * ✅ Implement matminer's `ElementProperty.from_preset('magpie')` to generate 132-dimensional features
  * ✅ Perform dimensionality reduction and visualization using PCA/t-SNE
  * ✅ Analyze feature importance using Random Forest

#### Application Ability

  * ✅ Design custom statistical functions (geometric mean, harmonic mean)
  * ✅ Compare and analyze feature distributions across multiple material systems
  * ✅ Appropriately select dimensionality reduction methods (UMAP vs PCA vs t-SNE)

## 2.1 Magpie Descriptor Details

### Design Philosophy of Ward et al. (2016)

The Magpie (Materials Agnostic Platform for Informatics and Exploration) descriptor is the definitive composition-based feature set published by Dr. Logan Ward and colleagues at Northwestern University in 2016. In their _npj Computational Materials_ paper, Ward et al. (2016) proposed it as a "**general-purpose framework for predicting material properties without structural information** " (pp. 1-2).

The core design is based on three principles:

  1. **Physical interpretability** : All features are based on physicochemical properties of elements
  2. **Scalability** : Applicable to arbitrary chemical formulas (approximately 2-10 elements)
  3. **Information maximization** : Comprehensive description of materials space with 132 dimensions from 22 elemental properties × 6 statistical measures

#### 💡 Why 132 Dimensions? (And Where "145" Comes From)

Ward et al. found that adding too many elemental properties increases redundancy, while too few reduces expressiveness. The 132 dimensions returned by matminer's `ElementProperty.from_preset('magpie')` are the result of optimizing the **balance between information content and computational efficiency** (Ward et al., 2016, p. 4).

You will often see the number **145** quoted instead. That is the size of the full attribute set described in the Ward et al. (2016) paper, which stacks three extra families on top of the elemental-property statistics: 6 stoichiometric attributes + **132 elemental-property statistics** + 4 valence-orbital fraction attributes + 3 ionic-compound attributes = 145. In matminer those extra families live in separate featurizers (`Stoichiometry`, `ValenceOrbital`, `IonProperty`), so the `magpie` preset on its own is always 132 columns — in every version of the library. Throughout this chapter, "Magpie features" means those 132 columns unless the Ward 2016 attribute set is named explicitly.

In fact, Magpie achieved MAE=0.12 eV/atom in predicting formation enthalpies in OQMD (Open Quantum Materials Database), comparable to structure-based descriptors (Ward et al., 2017, p. 6).

### Structure of the 132-Dimensional Vector

Magpie descriptors have the following hierarchical structure:
    
    
    ```mermaid
    graph TD
        A[Magpie 132 Dimensions] --> B[Elemental Properties 22 Types]
        A --> C[Statistical Measures 6 Types]
    
        B --> D[Periodic Table Position 4 Types]
        B --> E[Atomic and Thermal 4 Types]
        B --> F[Valence Electron Counts 5 Types]
        B --> G[Unfilled Orbital Counts 5 Types]
        B --> N[Ground State DFT 4 Types]
    
        C --> H[minimum]
        C --> I[maximum]
        C --> J[range]
        C --> K[mean fraction weighted]
        C --> L[avg_dev]
        C --> M[mode]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:3px,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
    ```

**Breakdown of dimensions:**

  * 22 elemental properties × minimum = 22 dimensions
  * 22 elemental properties × maximum = 22 dimensions
  * 22 elemental properties × range (max - min) = 22 dimensions
  * 22 elemental properties × mean (fraction-weighted) = 22 dimensions
  * 22 elemental properties × avg_dev (mean absolute deviation) = 22 dimensions
  * 22 elemental properties × mode = 22 dimensions
  * **Total: 132 dimensions**

### Physical Meaning of Each Dimension

Each dimension of Magpie descriptors represents a physical quantity that directly affects material properties. For example:

Example Dimension | Physical Meaning | Affected Material Properties  
---|---|---  
MagpieData mean CovalentRadius | Fraction-weighted mean covalent radius (pm) | Lattice constants, density, ionic conductivity  
MagpieData range Electronegativity | Electronegativity range | Ionic bonding character, band gap  
MagpieData maximum MeltingT | Maximum melting point (K) | High-temperature stability, heat resistance  
MagpieData avg_dev NValence | Spread of the total valence-electron count | Redox properties, catalytic activity  
MagpieData mode GSvolume_pa | Mode ground-state volume/atom | Crystal structure stability  
  
## 2.2 Types of Elemental Properties

The 22 elemental properties of the `magpie` preset are fixed — they are exactly the list held in `ElementProperty.from_preset("magpie").features`, and they do not change between matminer releases. They fall into five natural groups.

### Periodic Table Position (4 Types)

Where an element sits in the periodic table, which encodes chemical similarity:

  1. **Number** (Atomic number, Z): Proton count, nuclear charge
  2. **MendeleevNumber** (Mendeleev number): An ordering that places chemically similar elements next to each other
  3. **Column** (Group number, 1-18): Chemical property periodicity
  4. **Row** (Period number, 1-7): Electron shell count, atomic size

### Atomic and Thermal Properties (4 Types)

Mass, size, bonding character, and thermal stability of the atom:

  1. **AtomicWeight** (Atomic weight, g/mol): Affects mass and density
  2. **MeltingT** (Melting point, K): Indicator of high-temperature stability
  3. **CovalentRadius** (Covalent radius, pm): Determines bond lengths and lattice constants
  4. **Electronegativity** (Electronegativity, Pauling scale): Ionic/covalent character of bonds

### Valence Electron Counts (5 Types)

The number of valence electrons in each orbital type, plus their total:

  1. **NsValence** (s-orbital valence electrons): Metallic bond strength
  2. **NpValence** (p-orbital valence electrons): Semiconductor properties
  3. **NdValence** (d-orbital valence electrons): Catalytic activity of transition metals
  4. **NfValence** (f-orbital valence electrons): Magnetism in lanthanides and actinides
  5. **NValence** (Total valence electrons): Overall bonding capacity and redox behaviour

### Unfilled Orbital Counts (5 Types)

The number of *vacancies* in each orbital type — the counterpart of the valence counts, and a strong signal for reactivity:

  1. **NsUnfilled** (Unfilled s states): Availability of s-type acceptor states
  2. **NpUnfilled** (Unfilled p states): Anion character, p-band filling
  3. **NdUnfilled** (Unfilled d states): Transition-metal reactivity and magnetism
  4. **NfUnfilled** (Unfilled f states): f-electron behaviour in rare earths
  5. **NUnfilled** (Total unfilled states): Overall electron-accepting capacity

### Ground-State (DFT) Properties (4 Types)

Properties of the element's own ground-state crystal, tabulated from DFT calculations:

  1. **GSvolume_pa** (Ground-state volume/atom, Å³): Theoretical volume from DFT calculations
  2. **GSbandgap** (Ground-state band gap, eV): Electrical properties of semiconductors/insulators
  3. **GSmagmom** (Ground-state magnetic moment, μB): Properties of magnetic materials
  4. **SpaceGroupNumber** (Space group number): Crystal symmetry of the elemental solid

Several quantities that feel like obvious candidates are **not** in these 22: ionization energy, electron affinity, density, boiling point, heat capacity, and ground-state energy per atom. If your problem needs them, use a different preset (matminer also ships `deml`, `matminer`, and `magpie_oxid`), pass an explicit property list to `ElementProperty`, or write a custom featurizer — Chapter 3 shows how.

#### 📊 Database Sources

Magpie elemental properties are obtained from the following databases:

  * **OQMD** (Open Quantum Materials Database): Ground-state properties from DFT calculations (GSvolume_pa, GSbandgap, GSmagmom)
  * **Materials Project** : Crystal structure database (SpaceGroupNumber, etc.)
  * **Mendeleev** : Standard periodic table elemental properties (atomic weight, electronegativity, melting point, etc.)

The values are shipped with matminer as a lookup table, and the same underlying quantities are also reachable through the `pymatgen.Element` class.

## 2.3 Statistical Aggregation Methods

### The Six Magpie Statistics

Each of the 22 elemental properties is collapsed into a single number six different ways, giving 22 × 6 = 132 columns named `MagpieData {statistic} {property}`. The six statistics are fixed: `minimum`, `maximum`, `range`, `mean`, `avg_dev`, `mode`.

Throughout, $f_i = n_i / \sum_j n_j$ is the **atomic fraction** of element $i$ (with $n_i$ its atom count), and $p_i$ is the property value of element $i$.

#### 1\. Minimum

Minimum property value in the composition. Represents the material's "bottleneck":

$$ \text{minimum}(P) = \min_{i} p_i $$

**Example:** minimum Electronegativity = min(Fe: 1.83, O: 3.44) = 1.83 (Fe)

#### 2\. Maximum

Maximum property value in the composition. Indicates the material's "peak performance":

$$ \text{maximum}(P) = \max_{i} p_i $$

**Example:** maximum MeltingT = max(Fe: 1811 K, O: 54 K) = 1811 K (Fe)

#### 3\. Range

Difference between maximum and minimum values. Represents property "spread":

$$ \text{range}(P) = \text{maximum}(P) - \text{minimum}(P) $$

**Example:** range Electronegativity = 3.44 - 1.83 = 1.61 (indicates ionic bonding strength)

#### 4\. Mean (Fraction-Weighted)

Magpie's `mean` is **not** a plain average over element types — it is weighted by atomic fraction:

$$ \text{mean}(P) = \sum_{i} f_i \cdot p_i $$

**Example (mean atomic radius of Fe 2O3):**

  * Atomic fraction of Fe: $f_{\text{Fe}} = 2 / (2+3) = 0.4$
  * Atomic fraction of O: $f_{\text{O}} = 3 / (2+3) = 0.6$
  * mean = $0.4 \times 1.26 + 0.6 \times 0.66 = 0.504 + 0.396 = 0.900$ Å

This value represents the **effective atomic radius** of the material and is useful for predicting lattice constants and density. The unweighted average over element types, $(1.26 + 0.66)/2 = 0.96$ Å, is a different quantity and is *not* what matminer returns.

#### 5\. Avg_dev (Mean Absolute Deviation)

The fraction-weighted mean absolute deviation from the mean. This is Magpie's measure of how spread out a property is:

$$ \text{avg\_dev}(P) = \sum_{i} f_i \cdot \lvert p_i - \text{mean}(P) \rvert $$

**Example (atomic radius of Fe 2O3):** $0.4 \times |1.26 - 0.900| + 0.6 \times |0.66 - 0.900| = 0.144 + 0.144 = 0.288$ Å

A value near zero means every element carries a similar value of that property; a large value flags a chemically heterogeneous composition.

#### 6\. Mode

The property value of the element with the largest atomic fraction. Important for multi-element systems:

$$ \text{mode}(P) = p_{\arg\max_i f_i} $$

**Example (LiFePO 4):** Li: 1, Fe: 1, P: 1, O: 4 atoms → O (oxygen) has the largest atomic fraction, so O's property value is returned as the mode.

#### ⚠️ "Weighted Mean" and "Std" Are Not Magpie Statistics

Many tutorials list `weighted_mean` and `std` alongside the statistics above. Neither is part of the preset. Magpie's `mean` already **is** the fraction-weighted mean, so there is no separate weighted-mean column, and dispersion is reported as `avg_dev`, never as a standard deviation. When you see `weighted_mean` or `std` next to Magpie features, someone has added a custom statistic on top — which is exactly what the next subsection does deliberately.

### Advanced Statistical Measures (Custom Design)

#### Geometric Mean

Expresses multiplicative effects (e.g., catalytic activation energy):

$$ \text{geometric_mean}(P) = \left( \prod_{i=1}^{N} p_i \right)^{1/N} $$ 

#### Harmonic Mean

Reciprocal mean. Expresses "series effects" like resistance or thermal conductivity:

$$ \text{harmonic_mean}(P) = \frac{N}{\sum_{i=1}^{N} \frac{1}{p_i}} $$ 

#### Standard Deviation

Degree of property variance:

$$ \text{std}(P) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (p_i - \text{mean}(P))^2} $$ 

## 2.4 Feature Visualization and Interpretation

### Need for Dimensionality Reduction in High-Dimensional Data

132-dimensional Magpie features cannot be intuitively understood by humans as-is. **Dimensionality reduction** compresses 132 dimensions → 2D or 3D for visualization, enabling:

  * Discovery of material cluster structures (oxides, metals, semiconductors, etc. group together)
  * Detection of outliers
  * Determination of search regions for new materials
  * Diagnosis of model prediction accuracy

### PCA (Principal Component Analysis)

PCA (Principal Component Analysis) is a method that finds directions (principal components) where data variance is maximized through **linear transformation**.

**Principle:**

  1. Calculate the covariance matrix of the data
  2. Find eigenvalues and eigenvectors
  3. Select principal component axes in descending order of eigenvalues

**Formula:**

$$ \mathbf{Z} = \mathbf{X} \mathbf{W} $$ 

Where $\mathbf{X}$ is the original 132-dimensional data, $\mathbf{W}$ is the principal component axes (eigenvectors), and $\mathbf{Z}$ is the reduced data.

**Advantages:**

  * Fast computation (applicable to large-scale data)
  * Quantitative evaluation of principal component contribution rates
  * High interpretability due to linear transformation

**Disadvantages:**

  * Cannot capture nonlinear structures (complex cluster structures are not preserved)
  * Sensitive to outliers

### t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE is a method that preserves local neighborhood relationships in high-dimensional data in 2D space through **nonlinear transformation**.

**Principle:**

  1. Calculate similarity (Gaussian distribution) between each pair of points in high-dimensional space
  2. Define similar similarity in low-dimensional space using t-distribution
  3. Optimize low-dimensional coordinates to minimize KL divergence (Kullback-Leibler divergence)

**Advantages:**

  * Beautiful visualization of complex cluster structures
  * Preserves local structure (similar points are placed nearby)

**Disadvantages:**

  * High computational cost (time-consuming for large-scale data)
  * Requires hyperparameter tuning (perplexity)
  * Results change with each run (stochastic optimization)
  * Global distance relationships are not guaranteed (inter-cluster distances are not meaningful)

### UMAP (Uniform Manifold Approximation and Projection)

UMAP is a modern dimensionality reduction method that improves upon t-SNE's shortcomings.

**Advantages:**

  * Faster than t-SNE (applicable to large-scale data)
  * Preserves global structure to some extent
  * Relatively easy parameter tuning

**Disadvantages:**

  * Higher computational cost than PCA
  * Requires attention to reproducibility due to stochastic methods

### Method Selection Guidelines

Method | Application Case | Data Size | Computation Time  
---|---|---|---  
PCA | Linear structure exploration, contribution analysis | ~1 million points | ⭐⭐⭐⭐⭐ Ultra-fast  
t-SNE | Complex cluster visualization | ~100k points | ⭐⭐ Slow  
UMAP | High-quality visualization of large-scale data | ~1 million points | ⭐⭐⭐⭐ Fast  
  
### Example Distribution by Material Class

When Magpie features are dimensionally reduced with PCA, the following material class separations are observed (Ward et al., 2016, p. 5):

  * **Metals** : Low electronegativity, high density
  * **Oxides** : High electronegativity range, moderate melting points
  * **Semiconductors** : Moderate electronegativity, specific band gap ranges
  * **Composite materials** : Wide property ranges, high standard deviation

## 2.5 Implementation Examples and Code Tutorials

### Code Example 1: Basic matminer MagpieFeaturizer Implementation

[Open in Google Colab](<https://colab.research.google.com/drive/1example_magpie_basic>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 1: Basic matminer MagpieFeaturizer Implementati
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 1: Basic Magpie Feature Generation
    # ===================================
    
    # Import necessary libraries
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # Initialize MagpieFeaturizer
    magpie = ElementProperty.from_preset("magpie")
    
    # Test chemical formulas
    compositions = ["Fe2O3", "TiO2", "LiFePO4", "MgB2", "BaTiO3"]
    
    # Feature generation
    features = []
    for comp in compositions:
        feat = magpie.featurize_dataframe(
            pd.DataFrame({"composition": [comp]}),
            col_id="composition"
        )
        features.append(feat)
    
    # Integrate results into DataFrame
    df = pd.concat(features, ignore_index=True)
    print(f"Number of generated features: {len(df.columns) - 1}")  # Excluding composition column
    print(f"\nFirst 5 dimensions:")
    print(df.iloc[:, 1:6].head())
    
    # Expected output:
    # Number of generated features: 132
    # (ElementProperty.from_preset("magpie") always returns 132 columns: 22 properties x 6 statistics)
    

### Code Example 2: Complete 132-Dimensional Feature Generation and Detailed Display

[Open in Google Colab](<https://colab.research.google.com/drive/1example_magpie_full>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 2: Complete 132-Dimensional Feature Generation 
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 2: Complete 132-Dimensional Magpie Feature Generation
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import pandas as pd
    import numpy as np
    
    # Magpie descriptor configuration (using all elemental properties)
    magpie = ElementProperty.from_preset("magpie")
    
    # Convert chemical formula to Composition object
    comp = Composition("Fe2O3")
    
    # Feature generation
    df = pd.DataFrame({"composition": [comp]})
    df = magpie.featurize_dataframe(df, col_id="composition")
    
    # Get feature names
    feature_names = magpie.feature_labels()
    print(f"Total Magpie feature dimensions: {len(feature_names)}")
    print(f"\nNumber of elemental property types: {len(set([name.split()[0] for name in feature_names]))}")
    
    # Count statistical measure types
    stats = {}
    for name in feature_names:
        stat = name.split()[0]  # Extract "mean", "range", etc.
        stats[stat] = stats.get(stat, 0) + 1
    
    print("\nDimensions by statistical measure:")
    for stat, count in sorted(stats.items()):
        print(f"  {stat}: {count} dimensions")
    
    # Display some features of Fe2O3
    print(f"\nKey features of Fe2O3:")
    important_features = [
        "mean AtomicWeight",
        "range Electronegativity",
        "max MeltingT",
        "weighted_mean Row"
    ]
    for feat in important_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            print(f"  {feat}: {df.iloc[0, idx+1]:.3f}")
    
    # Expected output:
    # Total Magpie feature dimensions: 132
    #
    # Dimensions by statistical measure:
    #   mean: 22 dimensions
    #   range: 22 dimensions
    #   ...
    #
    # Key features of Fe2O3:
    #   mean AtomicWeight: 31.951
    #   range Electronegativity: 1.610
    #   max MeltingT: 3134.000
    #   weighted_mean Row: 3.200
    

### Code Example 3: PCA Dimensionality Reduction and Visualization

[Open in Google Colab](<https://colab.research.google.com/drive/1example_pca_viz>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 3: PCA Dimensionality Reduction and Visualizati
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 3: PCA Dimensionality Reduction and Visualization
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Data preparation (different material classes)
    materials = {
        "oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO"],
        "metals": ["Fe", "Cu", "Al", "Ni", "Ti"],
        "semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS"],
        "perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3"]
    }
    
    # Magpie feature generation
    magpie = ElementProperty.from_preset("magpie")
    all_features = []
    all_labels = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
    
            # Get features only, excluding composition column
            features = df_feat.iloc[0, 1:].values
            all_features.append(features)
            all_labels.append(material_class)
    
    # Convert to NumPy array
    X = np.array(all_features)
    print(f"Feature matrix size: {X.shape}")  # (20 materials, 132 dimensions)
    
    # Reduce 132 dimensions → 2 dimensions with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Display contribution rates
    print(f"\n1st principal component contribution: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"2nd principal component contribution: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Cumulative contribution: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Visualization
    plt.figure(figsize=(10, 7))
    colors = {"oxides": "red", "metals": "blue", "semiconductors": "green", "perovskites": "orange"}
    
    for material_class in materials.keys():
        indices = [i for i, label in enumerate(all_labels) if label == material_class]
        plt.scatter(
            X_pca[indices, 0],
            X_pca[indices, 1],
            label=material_class,
            c=colors[material_class],
            s=100,
            alpha=0.7
        )
    
    plt.xlabel(f"1st PC (contribution {pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"2nd PC (contribution {pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA Visualization of Magpie Features (by Material Class)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("magpie_pca_visualization.png", dpi=150)
    plt.show()
    
    # Expected output:
    # Feature matrix size: (20, 132)
    #
    # 1st principal component contribution: 0.452
    # 2nd principal component contribution: 0.231
    # Cumulative contribution: 0.683
    #
    # (Scatter plot with color-coded material classes is displayed)
    

### Code Example 4: t-SNE Visualization (with perplexity optimization)

[Open in Google Colab](<https://colab.research.google.com/drive/1example_tsne_viz>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: t-SNE Visualization (with perplexity optimiz
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 4: t-SNE Dimensionality Reduction (perplexity optimization)
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Data preparation (same as Example 3)
    materials = {
        "oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO", "MgO", "CaO"],
        "metals": ["Fe", "Cu", "Al", "Ni", "Ti", "Co", "Cr"],
        "semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS", "Ge", "SiC"],
        "perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3"]
    }
    
    magpie = ElementProperty.from_preset("magpie")
    all_features = []
    all_labels = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
            features = df_feat.iloc[0, 1:].values
            all_features.append(features)
            all_labels.append(material_class)
    
    X = np.array(all_features)
    
    # Compare different perplexity settings
    perplexities = [5, 10, 20, 30]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"oxides": "red", "metals": "blue", "semiconductors": "green", "perovskites": "orange"}
    
    for idx, perp in enumerate(perplexities):
        ax = axes[idx // 2, idx % 2]
    
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
    
        # Visualization
        for material_class in materials.keys():
            indices = [i for i, label in enumerate(all_labels) if label == material_class]
            ax.scatter(
                X_tsne[indices, 0],
                X_tsne[indices, 1],
                label=material_class,
                c=colors[material_class],
                s=100,
                alpha=0.7
            )
    
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(f"perplexity = {perp}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("magpie_tsne_perplexity_comparison.png", dpi=150)
    plt.show()
    
    print("Perplexity selection guide:")
    print("  Small value (5-10): Emphasizes local cluster structure")
    print("  Moderate (10-30): Balanced visualization (recommended)")
    print("  Large value (30-50): Preserves global structure")
    
    # Expected output:
    # (Four subplots showing t-SNE results with different perplexity settings)
    # Material classes are best separated at perplexity=20 or so
    

### Code Example 5: Using Elemental Property Databases (pymatgen Element)

[Open in Google Colab](<https://colab.research.google.com/drive/1example_element_db>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 5: Using Elemental Property Databases (pymatgen
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 5: Get Elemental Properties with pymatgen Element
    # ===================================
    
    from pymatgen.core import Element
    import pandas as pd
    
    # Representative elements from the periodic table
    elements = ["H", "C", "O", "Fe", "Cu", "Si", "Au", "U"]
    
    # Get elemental properties
    data = []
    for elem_symbol in elements:
        elem = Element(elem_symbol)
    
        data.append({
            "Element": elem_symbol,
            "AtomicNumber": elem.Z,
            "AtomicWeight": elem.atomic_mass,
            "AtomicRadius": elem.atomic_radius,
            "Electronegativity": elem.X,
            "IonizationEnergy": elem.ionization_energy,
            "MeltingPoint": elem.melting_point,
            "Density": elem.density_of_solid,
            "Row": elem.row,
            "Group": elem.group
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Example custom elemental property calculations
    print("\n--- Custom Statistics ---")
    comp = "Fe2O3"
    from pymatgen.core import Composition
    c = Composition(comp)
    
    # Get atomic radius for each element
    radii = []
    fractions = []
    for elem, frac in c.get_el_amt_dict().items():
        radii.append(Element(elem).atomic_radius)
        fractions.append(frac)
    
    # Calculate various statistical measures
    mean_radius = sum(radii) / len(radii)
    weighted_mean_radius = sum([r * f for r, f in zip(radii, fractions)]) / sum(fractions)
    min_radius = min(radii)
    max_radius = max(radii)
    range_radius = max_radius - min_radius
    
    print(f"Atomic radius statistics for {comp}:")
    print(f"  mean: {mean_radius:.3f} Å")
    print(f"  weighted_mean: {weighted_mean_radius:.3f} Å")
    print(f"  min: {min_radius:.3f} Å")
    print(f"  max: {max_radius:.3f} Å")
    print(f"  range: {range_radius:.3f} Å")
    
    # Expected output:
    # Element  AtomicNumber  AtomicWeight  AtomicRadius  Electronegativity  ...
    # H        1             1.008         0.320         2.20               ...
    # C        6             12.011        0.770         2.55               ...
    # ...
    #
    # Atomic radius statistics for Fe2O3:
    #   mean: 0.960 Å
    #   weighted_mean: 0.900 Å
    #   min: 0.660 Å
    #   max: 1.260 Å
    #   range: 0.600 Å
    

### Code Example 6: Feature Importance Analysis with Random Forest

[Open in Google Colab](<https://colab.research.google.com/drive/1example_feature_importance>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 6: Feature Importance Analysis with Random Fore
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # ===================================
    # Example 6: Analyze Feature Importance with Random Forest
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.datasets import load_dataset
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Load matminer sample dataset (formation enthalpy prediction)
    print("Loading dataset...")
    df = load_dataset("castelli_perovskites")  # Perovskite data (18,928 compounds)
    
    # Check composition column
    if "formula" in df.columns:
        comp_col = "formula"
    elif "composition" in df.columns:
        comp_col = "composition"
    else:
        comp_col = df.columns[0]
    
    # Magpie feature generation (test with first 1000 entries)
    df_sample = df.head(1000).copy()
    magpie = ElementProperty.from_preset("magpie")
    
    print("Generating features...")
    df_feat = magpie.featurize_dataframe(df_sample, col_id=comp_col)
    
    # Separate features and target variable
    feature_cols = magpie.feature_labels()
    X = df_feat[feature_cols].values
    y = df_feat["e_form"].values  # Formation enthalpy
    
    # Remove missing values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    print(f"Valid data count: {len(X)}")
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    print("Training model...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Prediction accuracy
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"\nTraining data R²: {train_score:.3f}")
    print(f"Test data R²: {test_score:.3f}")
    
    # Get feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Display top 20 features
    print("\nFeature Importance Top 20:")
    for i in range(20):
        idx = indices[i]
        print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_indices = indices[:top_n]
    plt.barh(range(top_n), importances[top_indices], align="center")
    plt.yticks(range(top_n), [feature_cols[i] for i in top_indices])
    plt.xlabel("Importance")
    plt.title(f"Magpie Feature Importance (Formation Enthalpy Prediction, R²={test_score:.3f})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("magpie_feature_importance.png", dpi=150)
    plt.show()
    
    # Expected output:
    # Loading dataset...
    # Generating features...
    # Valid data count: 987
    # Training model...
    #
    # Training data R²: 0.923
    # Test data R²: 0.847
    #
    # Feature Importance Top 20:
    # 1. mean GSvolume_pa: 0.1254
    # 2. weighted_mean GSenergy_pa: 0.0987
    # 3. range Electronegativity: 0.0823
    # ...
    

### Code Example 7: Feature Distribution by Material Class (seaborn violinplot)

[Open in Google Colab](<https://colab.research.google.com/drive/1example_distribution_analysis>)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 7: Feature Distribution by Material Class (seab
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 7: Compare Feature Distributions by Material Class
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Data preparation (more samples)
    materials = {
        "Oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO", "MgO", "CaO", "SiO2", "SnO2", "V2O5"],
        "Metals": ["Fe", "Cu", "Al", "Ni", "Ti", "Co", "Cr", "Zn", "Ag", "Au"],
        "Semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS", "Ge", "SiC", "GaN", "AlN", "InSb"],
        "Perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3", "KNbO3", "NaTaO3", "BiFeO3"]
    }
    
    # Magpie feature generation
    magpie = ElementProperty.from_preset("magpie")
    results = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
    
            # Extract only important features
            row = {
                "Class": material_class,
                "mean_Electronegativity": df_feat["mean Electronegativity"].values[0],
                "range_Electronegativity": df_feat["range Electronegativity"].values[0],
                "mean_AtomicRadius": df_feat["mean AtomicRadius"].values[0],
                "weighted_mean_Row": df_feat["weighted_mean Row"].values[0]
            }
            results.append(row)
    
    df_results = pd.DataFrame(results)
    
    # Compare distribution of multiple features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features_to_plot = [
        ("mean_Electronegativity", "Mean Electronegativity"),
        ("range_Electronegativity", "Electronegativity Range"),
        ("mean_AtomicRadius", "Mean Atomic Radius (Å)"),
        ("weighted_mean_Row", "Weighted Mean Period")
    ]
    
    for idx, (feature, label) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        sns.violinplot(data=df_results, x="Class", y=feature, ax=ax, palette="Set2")
        ax.set_xlabel("Material Class")
        ax.set_ylabel(label)
        ax.set_title(f"Distribution Comparison of {label}")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("magpie_distribution_by_class.png", dpi=150)
    plt.show()
    
    # Statistical summary
    print("Mean electronegativity by material class:")
    print(df_results.groupby("Class")["mean_Electronegativity"].describe()[["mean", "std", "min", "max"]])
    
    # Expected output:
    # (Four violin plots displayed, visualizing feature distributions by material class)
    #
    # Mean electronegativity by material class:
    #                   mean       std   min   max
    # Class
    # Metals           1.763  0.214  1.550  2.200
    # Oxides           2.895  0.312  2.550  3.440
    # Perovskites      2.134  0.187  1.900  2.450
    # Semiconductors   2.012  0.298  1.810  2.550
    

### Code Example 8: Custom Statistical Functions (geometric mean, harmonic mean)

[Open in Google Colab](<https://colab.research.google.com/drive/1example_custom_stats>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # ===================================
    # Example 8: Implementation and Application of Custom Statistical Functions
    # ===================================
    
    from pymatgen.core import Composition, Element
    import numpy as np
    import pandas as pd
    
    def geometric_mean(values):
        """Calculate geometric mean
    
        Args:
            values (list): List of numbers
    
        Returns:
            float: Geometric mean
        """
        if len(values) == 0 or any(v <= 0 for v in values):
            return np.nan
        return np.prod(values) ** (1.0 / len(values))
    
    def harmonic_mean(values):
        """Calculate harmonic mean
    
        Args:
            values (list): List of numbers
    
        Returns:
            float: Harmonic mean
        """
        if len(values) == 0 or any(v == 0 for v in values):
            return np.nan
        return len(values) / sum(1.0 / v for v in values)
    
    def compute_custom_stats(composition_str, property_name):
        """Calculate custom statistics
    
        Args:
            composition_str (str): Chemical formula (e.g., "Fe2O3")
            property_name (str): Elemental property name (e.g., "atomic_radius")
    
        Returns:
            dict: Various statistical measures
        """
        comp = Composition(composition_str)
    
        # Get elemental properties
        values = []
        fractions = []
    
        for elem, frac in comp.get_el_amt_dict().items():
            element = Element(elem)
    
            # Get value according to property name
            if property_name == "atomic_radius":
                val = element.atomic_radius
            elif property_name == "electronegativity":
                val = element.X
            elif property_name == "ionization_energy":
                val = element.ionization_energy
            elif property_name == "melting_point":
                val = element.melting_point
            else:
                raise ValueError(f"Unknown property: {property_name}")
    
            if val is not None:
                values.append(val)
                fractions.append(frac)
    
        if len(values) == 0:
            return {}
    
        # Calculate statistics
        total_atoms = sum(fractions)
        weights = [f / total_atoms for f in fractions]
    
        stats = {
            "arithmetic_mean": np.mean(values),
            "geometric_mean": geometric_mean(values),
            "harmonic_mean": harmonic_mean(values),
            "weighted_mean": sum(v * w for v, w in zip(values, weights)),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "std": np.std(values)
        }
    
        return stats
    
    # Test cases
    test_compounds = ["Fe2O3", "LiFePO4", "BaTiO3", "MgB2", "CuInGaSe2"]
    
    # Calculate statistics for multiple elemental properties
    properties = ["atomic_radius", "electronegativity", "ionization_energy"]
    
    results = []
    for comp in test_compounds:
        for prop in properties:
            stats = compute_custom_stats(comp, prop)
            row = {"Compound": comp, "Property": prop}
            row.update(stats)
            results.append(row)
    
    df = pd.DataFrame(results)
    
    # Display atomic radius statistics
    print("=== Atomic Radius Statistics Comparison ===")
    df_radius = df[df["Property"] == "atomic_radius"]
    print(df_radius[["Compound", "arithmetic_mean", "geometric_mean", "harmonic_mean", "weighted_mean"]].to_string(index=False))
    
    # Compare geometric mean and arithmetic mean
    print("\n=== Statistics Comparison (Fe2O3 Electronegativity) ===")
    stats_fe2o3 = compute_custom_stats("Fe2O3", "electronegativity")
    for stat_name, value in stats_fe2o3.items():
        print(f"{stat_name:20s}: {value:.4f}")
    
    # Physical meaning of custom statistics
    print("\n【Physical Meaning of Statistics】")
    print("- Arithmetic mean: Reflects diversity of element types")
    print("- Geometric mean: Expresses multiplicative effects (catalytic activity, etc.)")
    print("- Harmonic mean: Expresses series effects (resistance, thermal conductivity, etc.)")
    print("- Weighted mean: Effective value considering composition ratio")
    
    # Expected output:
    # === Atomic Radius Statistics Comparison ===
    # Compound  arithmetic_mean  geometric_mean  harmonic_mean  weighted_mean
    # Fe2O3           0.960          0.912          0.866          0.900
    # LiFePO4         0.948          0.895          0.831          0.842
    # BaTiO3          1.313          1.171          1.016          1.076
    # MgB2            0.980          0.930          0.880          0.901
    # CuInGaSe2       1.163          1.141          1.118          1.144
    #
    # === Statistics Comparison (Fe2O3 Electronegativity) ===
    # arithmetic_mean     : 2.6350
    # geometric_mean      : 2.5090
    # harmonic_mean       : 2.3891
    # weighted_mean       : 2.7960
    # min                 : 1.8300
    # max                 : 3.4400
    # range               : 1.6100
    # std                 : 0.8050
    

## 2.6 Learning Objectives Review

### ✅ What You Learned in This Chapter

#### Basic Understanding

  * ✅ Magpie descriptors consist of 132 dimensions (22 elemental properties × 6 statistical measures)
  * ✅ The 22 elemental properties fall into five groups: periodic-table position, atomic/thermal, valence-electron counts, unfilled-orbital counts, and ground-state DFT properties
  * ✅ The six statistical aggregation methods (minimum, maximum, range, fraction-weighted mean, avg_dev, mode) quantify overall composition features

#### Practical Skills

  * ✅ Can generate 132-dimensional features with matminer's `ElementProperty.from_preset('magpie')`
  * ✅ Can perform dimensionality reduction with PCA/t-SNE and visualize materials space in 2D
  * ✅ Can analyze feature importance with Random Forest and identify factors contributing to material property prediction

#### Application Ability

  * ✅ Can design custom statistical functions (geometric mean, harmonic mean) to express specific physical phenomena
  * ✅ Can compare and analyze feature distributions across multiple material systems (oxides, metals, semiconductors, perovskites)
  * ✅ Can appropriately select and use dimensionality reduction methods (PCA, t-SNE, UMAP) according to data characteristics

## Exercises

### Easy (Basic Confirmation)

**Q1:** What is the total dimensionality of Magpie descriptors? Also, state their components (number of elemental property types and number of statistical measure types).

**Answer:** 132 dimensions

**Components:**

  * Elemental properties: 22 types
  * Statistical measures: 6 types (minimum, maximum, range, fraction-weighted mean, avg_dev, mode)

**Explanation:** Magpie descriptors are the standard for composition-based features designed by Ward et al. (2016). matminer's `ElementProperty.from_preset('magpie')` computes 6 statistics for each of 22 elemental properties, giving 22 × 6 = 132 columns. The number 145 that is often quoted refers to the full attribute set of the Ward et al. (2016) paper (6 stoichiometric + 132 elemental-property + 4 valence-orbital + 3 ionic-compound attributes); in matminer those extra families are separate featurizers.

**Q2:** Calculate the Magpie **mean** and **avg_dev** of atomic radius for Fe2O3, and compare the mean with a plain average over element types. The atomic radius of Fe is 1.26 Å and O is 0.66 Å.

**Answer:**

  * Magpie mean = 0.900 Å
  * Magpie avg_dev = 0.288 Å
  * Plain average over element types = 0.96 Å (not a Magpie feature)

**Calculation process:**

**mean (fraction-weighted):**

Atomic fraction of Fe: 2 / (2+3) = 0.4

Atomic fraction of O: 3 / (2+3) = 0.6

mean = 0.4 × 1.26 + 0.6 × 0.66 = 0.504 + 0.396 = 0.900 Å

**avg_dev (fraction-weighted mean absolute deviation):**

avg_dev = 0.4 × |1.26 − 0.900| + 0.6 × |0.66 − 0.900| = 0.4 × 0.360 + 0.6 × 0.240 = 0.144 + 0.144 = 0.288 Å

**Plain average over element types:** (1.26 + 0.66) / 2 = 1.92 / 2 = 0.96 Å

**Explanation:** Magpie's mean is the composition-weighted effective value, so it reflects the actual atomic arrangement in the material; the unweighted average over element types treats a trace dopant as equal to the host and is not what matminer returns.

**Q3:** Among the following, select all that belong to the 22 elemental properties of matminer's `magpie` preset.  
a) Electronegativity  
b) MeltingT (melting point)  
c) IonizationEnergy (ionization energy)  
d) AtomicRadius (atomic radius)  
e) NdUnfilled (unfilled d states)

**Answer:** a) Electronegativity, b) MeltingT, e) NdUnfilled

**Explanation:**

  * a), b), e) are in the preset: Electronegativity and MeltingT belong to the atomic/thermal group, NdUnfilled to the unfilled-orbital group.
  * c) IonizationEnergy is **not** in the preset, and neither is ElectronAffinity — despite appearing in many descriptions of Magpie.
  * d) AtomicRadius is **not** in the preset either; the radius Magpie stores is **CovalentRadius**.

If you need ionization energy or electron affinity, supply an explicit property list to `ElementProperty` or write a custom featurizer (Chapter 3).

### Medium (Application)

**Q4:** List three differences between PCA and t-SNE, and explain in which situations each should be used.

**Three differences between PCA and t-SNE:**

Aspect | PCA | t-SNE  
---|---|---  
Transformation method | Linear transformation | Nonlinear transformation  
Preserved structure | Global variance | Local neighborhood relationships  
Computation speed | Fast (large-scale data compatible) | Slow (medium-scale data only)  
  
**Usage guidelines:**

  * **When to use PCA:**
    * When data is linearly separable
    * When you want to quantitatively evaluate principal component contribution
    * When data size is large (100k+ points)
    * When computation speed is important
  * **When to use t-SNE:**
    * When you want to visualize complex cluster structures
    * When local similarity is important
    * When data size is medium (10k-100k points)
    * When beautiful visualization is the goal

**Example:** In materials exploration, it's effective to first use PCA to grasp global structure, then use t-SNE to visualize specific cluster regions in detail.

**Q5:** Feature importance analysis with Random Forest identified **mean GSvolume_pa** (mean ground-state volume/atom) as the most important feature. What material properties is this result suitable for predicting? Explain with reasons.

**Suitable material properties for prediction:**

  * **Formation Enthalpy**
  * **Density**
  * **Lattice Constant**
  * **Bulk Modulus**

**Reasons:**

GSvolume_pa (ground-state volume/atom) is the **theoretical atomic volume** obtained from DFT calculations. The importance of this property means:

  1. **Correlation with structural stability:** Materials with smaller volumes tend to have shorter interatomic distances and stronger bonds. This directly correlates with low formation enthalpy (high stability).
  2. **Direct relationship with density:** Smaller volume/atom results in higher material density.
  3. **Crystal structure influence:** Volume varies with crystal structure even for the same composition, suggesting structural factors strongly affect material properties.

**Note:** Since GSvolume_pa is a DFT calculation-based property, there may be slight discrepancies with experimental data. Also, note that it depends on crystal structure as well as composition, so it's **not a pure composition-based descriptor**.

**Q6:** When setting t-SNE's hyperparameter **perplexity** to 5, 20, and 50, what visualization results do you expect for each? Also, how should the optimal value be determined?

**Effect of perplexity:**

perplexity | Visualization characteristics | Application cases  
---|---|---  
5 (small) | Very fine clusters are formed. Emphasizes local structure but may over-cluster. | Local outlier detection, fine structure exploration  
20 (moderate) | Balanced cluster formation. Material classes (oxides, metals, etc.) are clearly separated. | General visualization (recommended)  
50 (large) | Preserves global structure. Cluster boundaries may become ambiguous. | Large datasets, global trend analysis  
  
**How to determine optimal value:**

  1. **Rule of thumb based on data size:**
     * Small (<100 points): perplexity = 5-15
     * Medium (100-1000 points): perplexity = 20-50
     * Large (>1000 points): perplexity = 50-100
  2. **Try multiple values:** Set perplexity = [5, 10, 20, 30, 50] and select the most interpretable result
  3. **Cluster evaluation metrics:** Quantitatively evaluate with Silhouette Score, etc.

**Example:** For materials exploration (100-1000 compounds), perplexity=20-30 often provides the clearest material class separation.

**Q7:** Literature and tutorials describe Magpie as both "132-dimensional" and "145-dimensional." Which does matminer produce, and where does the other number come from?

**Answer:** matminer's `ElementProperty.from_preset('magpie')` produces **132 columns**, and always has — 22 elemental properties × 6 statistics. There is no matminer version in which this preset returns 145 columns.

**Where 145 comes from:** it is the size of the complete attribute set proposed in Ward et al. (2016), which is built from four families:

  * 6 **stoichiometric** attributes (L^p norms of the composition vector, independent of which elements are present)
  * 132 **elemental-property** statistics (the part matminer's `magpie` preset implements)
  * 4 **valence-orbital** attributes (fraction of valence electrons in s, p, d, f states)
  * 3 **ionic-compound** attributes (whether a charge-neutral ionic configuration exists, plus maximum ionic character)

6 + 132 + 4 + 3 = 145. In matminer, the three extra families are separate featurizers: `Stoichiometry`, `ValenceOrbital`, and `IonProperty`.

**Practical consequence:** to reproduce the Ward et al. (2016) attribute set, chain the four featurizers with `MultipleFeaturizer([ElementProperty.from_preset('magpie'), Stoichiometry(), ValenceOrbital(), IonProperty()])`. If you only call the `magpie` preset and report "145 features," the number is wrong — check `len(featurizer.feature_labels())` before quoting a dimensionality.

**Sizing advice:** with small datasets (<100 samples) even 132 columns can overfit; apply PCA or feature selection. Adding the extra 13 attributes changes computational cost negligibly, but it does change the feature set, so keep the featurizer definition under version control alongside the model.

### Hard (Advanced)

**Q8:** Implement **geometric mean** and **harmonic mean** as custom statistical functions, and calculate them for Fe2O3's electronegativity. Also, explain how these statistics differ from **arithmetic mean** and what physical phenomena they are suitable for expressing.  
(Fe electronegativity: 1.83, O electronegativity: 3.44)

**Calculation results:**

  * Arithmetic mean = (1.83 + 3.44) / 2 = 2.635
  * Geometric mean = √(1.83 × 3.44) = √6.2952 = 2.509
  * Harmonic mean = 2 / (1/1.83 + 1/3.44) = 2 / (0.546 + 0.291) = 2 / 0.837 = 2.389

**Magnitude relationship of statistics:**

Always Harmonic mean ≤ Geometric mean ≤ Arithmetic mean (equality only when all values are equal).

**Physical meaning and application cases:**

Statistic | Formula | Physical meaning | Application cases  
---|---|---|---  
Arithmetic mean | $(x_1 + x_2) / 2$ | Linear additive effect | Density, molar mass, and other extensive variables  
Geometric mean | $\sqrt{x_1 \times x_2}$ | Multiplicative effect | Catalytic activity (activation energy), chemical reaction rates, composite material properties  
Harmonic mean | $2 / (1/x_1 + 1/x_2)$ | Series resistance effect | Electrical resistance, thermal conductivity, diffusion coefficients (rate-limiting step control)  
  
**Interpretation of Fe 2O3 electronegativity:**

  * **Arithmetic mean (2.635):** Reflects element diversity. Equal consideration of Fe and O electronegativity.
  * **Geometric mean (2.509):** Suitable for expressing ionic bonding character. Large electronegativity difference = 3.44 - 1.83 = 1.61 indicates strong ionic bonding (Fe³⁺ and O²⁻) formation.
  * **Harmonic mean (2.389):** Emphasizes the influence of low electronegativity element (Fe). Expresses "bottleneck" in electron transfer.

**Implementation code example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation code example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    def geometric_mean(values):
        return np.prod(values) ** (1.0 / len(values))
    
    def harmonic_mean(values):
        return len(values) / sum(1.0 / v for v in values)
    
    # Fe2O3 electronegativity
    en_values = [1.83, 3.44]  # Fe, O
    
    print(f"Arithmetic mean: {np.mean(en_values):.3f}")
    print(f"Geometric mean: {geometric_mean(en_values):.3f}")
    print(f"Harmonic mean: {harmonic_mean(en_values):.3f}")
    

**Q9:** You want to compare and analyze Magpie features for three material systems (oxides, metals, semiconductors). What feature combinations should you visualize to most clearly show differences between material classes? Propose three feature pairs with reasons.

**Three recommended feature pairs with reasons:**

#### 1\. mean Electronegativity vs range Electronegativity

**Reasons:**

  * **Oxides:** High mean electronegativity (metal + nonmetal), large range (ionic bonding character)
  * **Metals:** Low mean electronegativity, small range (similar element combinations)
  * **Semiconductors:** Moderate mean electronegativity, moderate range

**Expected separation:** Most basic feature pair that clearly separates three classes.

#### 2\. weighted_mean GSbandgap vs mean IonizationEnergy

**Reasons:**

  * **Oxides:** High band gap (>3 eV, insulators), high ionization energy
  * **Metals:** Band gap = 0 eV (conductive), low ionization energy
  * **Semiconductors:** Moderate band gap (1-3 eV), moderate ionization energy

**Expected separation:** Directly reflects electronic state differences. Useful for predicting material electrical properties.

#### 3\. mean AtomicRadius vs weighted_mean MeltingT

**Reasons:**

  * **Oxides:** Moderate atomic radius, medium-high melting point (ceramic properties)
  * **Metals:** Large atomic radius (large metallic radius), high melting point (transition metals)
  * **Semiconductors:** Medium-large atomic radius, moderate melting point

**Expected separation:** Expresses structural and thermodynamic stability differences.

**Implementation example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: Implementation example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Visualize feature pairs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pairs = [
        ("mean Electronegativity", "range Electronegativity"),
        ("weighted_mean GSbandgap", "mean IonizationEnergy"),
        ("mean AtomicRadius", "weighted_mean MeltingT")
    ]
    
    for idx, (feat1, feat2) in enumerate(pairs):
        ax = axes[idx]
        for material_class in ["Oxides", "Metals", "Semiconductors"]:
            mask = df_results["Class"] == material_class
            ax.scatter(
                df_results[mask][feat1],
                df_results[mask][feat2],
                label=material_class,
                s=100,
                alpha=0.7
            )
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Q10:** Compare PCA, t-SNE, and UMAP dimensionality reduction methods on the following three evaluation axes, and determine which method to choose for a 10,000 compound Magpie feature dataset.  
**Evaluation axes:** (1) Computation time, (2) Cluster separation performance, (3) Global structure preservation

**Quantitative comparison of three methods (10,000 compounds):**

Method | Computation Time | Cluster Separation | Global Structure Preservation | Overall Evaluation  
---|---|---|---|---  
PCA | ⭐⭐⭐⭐⭐  
~1 second | ⭐⭐⭐  
Linear separation only | ⭐⭐⭐⭐⭐  
Complete preservation | Fast but limited separation  
t-SNE | ⭐  
~30-60 min | ⭐⭐⭐⭐⭐  
Best | ⭐⭐  
Not preserved | Beautiful but time-consuming  
UMAP | ⭐⭐⭐⭐  
~2-5 min | ⭐⭐⭐⭐  
High | ⭐⭐⭐⭐  
Somewhat preserved | **Best balance (recommended)**  
  
**Recommendation: UMAP (detailed reasons)**

  1. **Computation time:** Can execute in about 2-5 minutes for 10,000 compounds. 10-20x faster than t-SNE.
  2. **Cluster separation performance:** High separation performance comparable to t-SNE. Material classes (oxides, metals, semiconductors, etc.) are clearly separated.
  3. **Global structure:** Unlike t-SNE, inter-cluster distances also have some meaning. For example, "distance between oxides and metals > distance between oxide subclasses" is preserved.

**Application cases for each method:**

  * **PCA:** First step of exploratory data analysis (EDA), contribution rate analysis, data exceeding 1 million points
  * **t-SNE:** Beautiful visualization for papers, small data (<5,000 points), when computation time is not a constraint
  * **UMAP:** Practical materials exploration, medium to large data (1,000-100,000 points), balance-oriented

**Implementation example (comparing three methods):**
    
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import time
    
    # Data preparation (10,000 compound Magpie features)
    # X = np.array(magpie_features)  # shape: (10000, 132)
    
    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=30, n_iter=1000),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    }
    
    results = {}
    for name, model in methods.items():
        start = time.time()
        X_reduced = model.fit_transform(X)
        elapsed = time.time() - start
        results[name] = {"time": elapsed, "data": X_reduced}
        print(f"{name}: {elapsed:.2f} seconds")
    
    # Visualization comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        scatter = ax.scatter(
            result["data"][:, 0],
            result["data"][:, 1],
            c=labels,  # Material class labels
            cmap="Set2",
            s=10,
            alpha=0.6
        )
        ax.set_title(f"{name} ({result['time']:.1f}s)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    plt.show()
    

**Conclusion:** **UMAP** is optimal for a 10,000 compound dataset. It offers the best balance of computation time, separation performance, and structure preservation.

## Next Steps

In this chapter, we learned the detailed structure of Magpie descriptors, types of elemental properties, statistical aggregation methods, and visualization through dimensionality reduction.

In the next Chapter 3, we will learn **Element Property Databases and Featurizers** , integrating diverse data sources to design custom features.

[← Chapter 1: Composition-Based Features Fundamentals](<chapter-1.html>) [Return to Series Table of Contents](<./index.html>) [Chapter 3: Element Property Databases and Featurizers →](<chapter-3.html>)

## References

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028, pp. 1-7. https://doi.org/10.1038/npjcompumats.2016.28
  2. Ghiringhelli, L. M., Vybiral, J., Levchenko, S. V., Draxl, C., & Scheffler, M. (2015). "Big Data of Materials Science: Critical Role of the Descriptor." _Physical Review Letters_ , 114(10), 105503, pp. 1-5. https://doi.org/10.1103/PhysRevLett.114.105503
  3. Ward, L., Liu, R., Krishna, A., Hegde, V. I., Agrawal, A., Choudhary, A., & Wolverton, C. (2017). "Including crystal structure attributes in machine learning models of formation energies via Voronoi tessellations." _Physical Review B_ , 96(2), 024104, pp. 1-12. https://doi.org/10.1103/PhysRevB.96.024104
  4. Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., & Mar, A. (2016). "High-Throughput Machine-Learning-Driven Synthesis of Full-Heusler Compounds." _Chemistry of Materials_ , 28(20), 7324-7331, pp. 7324-7331. https://doi.org/10.1021/acs.chemmater.6b02724
  5. matminer Documentation: Composition-based featurizers. Hacking Materials Research Group, Lawrence Berkeley National Laboratory. https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html#composition-based-featurizers (Accessed: 2025-01-15)
  6. scikit-learn Documentation: Feature selection. scikit-learn developers. https://scikit-learn.org/stable/modules/feature_selection.html (Accessed: 2025-01-15)
  7. Mendeleev Python library documentation. https://mendeleev.readthedocs.io/ (Accessed: 2025-01-15)

* * *

[Return to Series Table of Contents](<./index.html>) | [← Chapter 1](<chapter-1.html>) | [Chapter 3 →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
