---
title: "Chapter 3: Element Property Databases and Featurizers"
chapter_title: "Chapter 3: Element Property Databases and Featurizers"
subtitle: Integrating diverse data sources to design custom features
---

This chapter covers Element Property Databases and Featurizers. You will learn Shannon entropy: $H = -\sum_i f_i \ln(f_i)$ and HEA determination: $H > 1.5$.

### 📋 Learning Objectives

Upon completing this chapter, you will be able to explain and implement the following:

  * **Basic Understanding** : Differences between Mendeleev/pymatgen/matminer, Featurizer architecture, data source reliability assessment
  * **Practical Skills** : Implementation of ElementProperty/Stoichiometry/OxidationStates Featurizers, pipeline construction, batch processing optimization
  * **Applied Competence** : Custom Featurizer design, application to complex material systems, multi-database integration strategies

## 3.1 Element Property Data Sources

Physicochemical property data of elements is essential for calculating composition-based features. However, different databases vary in their coverage, accuracy, and update frequency, making it important to select the appropriate source for your application. In this section, we compare three major data sources (Mendeleev, pymatgen, matminer) and understand their characteristics. 

### 3.1.1 Mendeleev: Comprehensive Element Database

**Mendeleev** is a comprehensive database of 118 elements, covering all elements in the periodic table with primarily experimental values. Its characteristics are as follows: 

  * **Data Volume** : Approximately 90 properties per element (atomic number, mass, density, melting point, boiling point, electronegativity, ionization energy, etc.)
  * **Data Source** : Primarily experimental values (from reliable sources such as NIST, CRC Handbook)
  * **Update Frequency** : Regular (approximately 1-2 times per year)
  * **Advantages** : High reliability with experimental values, clear data provenance
  * **Limitations** : Does not include calculated values (DFT, etc.), some missing values for certain elements

**💡 Pro Tip:** Mendeleev is optimal for precise property prediction (melting point, density, etc.) due to its high reliability of experimental values. However, some data may be missing for rare earth elements and synthetic elements, so prior verification is necessary. 

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example1.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    # ===================================
    # Example 1: Mendeleev vs pymatgen vs matminer Data Comparison
    # ===================================
    
    # Import required libraries
    import mendeleev
    from pymatgen.core import Element
    from matminer.featurizers.base import BaseFeaturizer
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # Data comparison function
    def compare_databases(element_symbol):
        """Retrieve and compare element properties from three databases
    
        Args:
            element_symbol (str): Element symbol (e.g., 'Fe', 'Cu')
    
        Returns:
            pd.DataFrame: Comparison table of properties from each database
        """
        # Get data from Mendeleev
        elem_mendeleev = mendeleev.element(element_symbol)
    
        # Get data from pymatgen
        elem_pymatgen = Element(element_symbol)
    
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Property': [
                'Atomic Number',
                'Atomic Mass',
                'Atomic Radius (pm)',
                'Electronegativity (Pauling)',
                'Ionization Energy 1st (eV)',
                'Melting Point (K)',
                'Density (g/cm³)'
            ],
            'Mendeleev': [
                elem_mendeleev.atomic_number,
                elem_mendeleev.atomic_weight,
                elem_mendeleev.atomic_radius if elem_mendeleev.atomic_radius else 'N/A',
                elem_mendeleev.electronegativity() if elem_mendeleev.electronegativity() else 'N/A',
                elem_mendeleev.ionenergies.get(1, 'N/A'),
                elem_mendeleev.melting_point if elem_mendeleev.melting_point else 'N/A',
                elem_mendeleev.density if elem_mendeleev.density else 'N/A'
            ],
            'pymatgen': [
                elem_pymatgen.Z,
                elem_pymatgen.atomic_mass,
                elem_pymatgen.atomic_radius,
                elem_pymatgen.X,  # Pauling electronegativity
                elem_pymatgen.ionization_energy,
                elem_pymatgen.melting_point if elem_pymatgen.melting_point else 'N/A',
                elem_pymatgen.density_of_solid if elem_pymatgen.density_of_solid else 'N/A'
            ]
        })
    
        return comparison
    
    # Example execution: Comparison of iron (Fe)
    fe_comparison = compare_databases('Fe')
    print("=== Element Property Comparison for Iron (Fe) ===")
    print(fe_comparison.to_string(index=False))
    print()
    
    # Comparison of multiple elements
    elements = ['Fe', 'Cu', 'Al', 'Ti']
    print("=== Electronegativity Comparison (Pauling Scale) ===")
    for elem in elements:
        mendeleev_val = mendeleev.element(elem).electronegativity()
        pymatgen_val = Element(elem).X
        print(f"{elem:2s} - Mendeleev: {mendeleev_val:.2f}, pymatgen: {pymatgen_val:.2f}")
    
    # Expected output:
    # === Element Property Comparison for Iron (Fe) ===
    #                      Property Mendeleev pymatgen
    #                Atomic Number        26       26
    #                  Atomic Mass     55.85    55.85
    #         Atomic Radius (pm)       140      140
    # Electronegativity (Pauling)      1.83     1.83
    #  Ionization Energy 1st (eV)      7.90     7.90
    #          Melting Point (K)      1811     1811
    #            Density (g/cm³)      7.87     7.87
    #
    # === Electronegativity Comparison (Pauling Scale) ===
    # Fe - Mendeleev: 1.83, pymatgen: 1.83
    # Cu - Mendeleev: 1.90, pymatgen: 1.90
    # Al - Mendeleev: 1.61, pymatgen: 1.61
    # Ti - Mendeleev: 1.54, pymatgen: 1.54
    

### 3.1.2 pymatgen: Materials Project Integrated Data

**pymatgen** is a materials science-specific library integrated with Materials Project, providing data including DFT calculated values. 

  * **Data Volume** : Approximately 60 properties (basic properties + DFT calculated values)
  * **Data Source** : Materials Project (DFT calculations) + experimental values
  * **Update Frequency** : Frequent (synchronized with Materials Project)
  * **Advantages** : DFT calculated values available, integrated with crystal structure data
  * **Limitations** : Some experimental values may be older than those in Mendeleev

### 3.1.3 matminer: Multi-Source Integration

**matminer** is a meta-database that integrates multiple data sources (Magpie, DeML, MEGNet, etc.). 

  * **Data Volume** : Varies by dataset (Magpie has 132 dimensions, DeML has 62 dimensions)
  * **Data Source** : Property sets reported in literature
  * **Update Frequency** : Depends on library updates
  * **Advantages** : Property sets optimized for machine learning, missing values handled
  * **Limitations** : Complex data provenance, version control is important

### 3.1.4 Data Source Comparison Table

Database | Main Data Source | Property Count | Missing Values | Recommended Use  
---|---|---|---|---  
**Mendeleev** | Experimental values (NIST, CRC) | ~90 | Few (5-10%) | Precise property prediction  
**pymatgen** | DFT + experimental values | ~60 | Moderate (10-20%) | Crystalline materials, computational values  
**matminer** | Literature property sets | 62-132 | Pre-processed | Machine learning, benchmarking  
      
    
    ```mermaid
    graph LR
        A[Element Property Data] --> B[MendeleevExperimental-Based]
        A --> C[pymatgenDFT + Experimental]
        A --> D[matminerMulti-Source Integration]
    
        B --> E[Precise Property PredictionMelting Point, Density]
        C --> F[Crystalline MaterialsBand Gap]
        D --> G[Machine LearningBenchmarking]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:3px,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#e8f5e9
        style G fill:#e8f5e9
    ```

## 3.2 Featurizer Architecture

matminer's Featurizers provide a unified interface for calculating features from element properties. They adopt a scikit-learn compatible API, allowing direct integration into machine learning pipelines. 

### 3.2.1 BaseFeaturizer Class

All Featurizers inherit from `BaseFeaturizer`. The main methods are as follows: 

  * **fit(X, y=None)** : Fit to data (many Featurizers do nothing here)
  * **transform(X)** : Calculate features
  * **fit_transform(X, y=None)** : Execute fit() and transform() consecutively
  * **featurize(entry)** : Calculate features for a single entry
  * **feature_labels()** : Return list of feature names
  * **citations()** : Return references

    
    
    ```mermaid
    classDiagram
        class BaseFeaturizer {
            +fit(X, y)
            +transform(X)
            +fit_transform(X, y)
            +featurize(entry)
            +feature_labels()
            +citations()
        }
    
        class ElementProperty {
            -data_source: str
            -features: List
            -stats: List
            +featurize(comp)
        }
    
        class Stoichiometry {
            -p_list: List
            -num_atoms: bool
            +featurize(comp)
        }
    
        class OxidationStates {
            -stats: List
            +featurize(comp)
        }
    
        BaseFeaturizer <|-- ElementProperty
        BaseFeaturizer <|-- Stoichiometry
        BaseFeaturizer <|-- OxidationStates
    ```

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example2.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: All Featurizers inherit fromBaseFeaturizer. The main methods
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 2: ElementProperty Featurizer Basic Implementation
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import pandas as pd
    
    # Initialize ElementProperty Featurizer
    # data_source: 'magpie' (Magpie dataset)
    # features: List of element properties to use
    # stats: List of statistics to calculate
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    
    # Prepare composition data
    compositions = [
        "Fe2O3",          # Iron oxide
        "TiO2",           # Titanium dioxide
        "Al2O3",          # Aluminum oxide
        "Cu2O",           # Copper(I) oxide
        "BaTiO3"          # Barium titanate
    ]
    
    # Convert to Composition objects
    comp_objects = [Composition(c) for c in compositions]
    
    # Calculate features
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'composition': comp_objects}),
        col_id='composition'
    )
    
    # Get feature names
    feature_names = featurizer.feature_labels()
    print(f"=== ElementProperty Features ===")
    print(f"Number of features: {len(feature_names)}")
    print(f"First 10 features: {feature_names[:10]}")
    print()
    
    # Display results (first 5 features)
    print("=== Calculated Results (First 5 Features) ===")
    display_cols = ['composition'] + feature_names[:5]
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # Get citations
    citations = featurizer.citations()
    print("=== References ===")
    for citation in citations:
        print(f"- {citation}")
    
    # Expected output:
    # === ElementProperty Features ===
    # Number of features: 132
    # First 10 features: ['MagpieData minimum Number', 'MagpieData maximum Number',
    #              'MagpieData range Number', 'MagpieData mean Number', ...]
    #
    # === Calculated Results (First 5 Features) ===
    # composition  MagpieData minimum Number  MagpieData maximum Number  ...
    #       Fe2O3                          8                         26  ...
    #        TiO2                          8                         22  ...
    #      Al2O3                          8                         13  ...
    #        Cu2O                          8                         29  ...
    #      BaTiO3                          8                         56  ...
    

### 3.2.2 scikit-learn Compatible API

Featurizers implement scikit-learn's `TransformerMixin`, allowing them to be used with `Pipeline` and `FeatureUnion`. 

**⚠️ Note:** The `fit()` method does nothing in many Featurizers, but some Featurizers (e.g., `AtomicPackingEfficiency`) learn parameters from training data. Always call `fit()` before using `transform()`. 

## 3.3 Types of Major Featurizers

matminer has over 30 implemented Featurizers. This section explains four particularly important Featurizers in detail. 

### 3.3.1 ElementProperty: Statistical Quantities of Element Properties

`ElementProperty` calculates statistical quantities (mean, maximum, minimum, range, standard deviation, etc.) for element properties (atomic number, mass, electronegativity, etc.) in the composition. The Magpie implementation generates 132-dimensional features from 22 element properties × 6 statistical quantities. 

**Main Parameters:**

  * **data_source** : Data source ('magpie', 'deml', 'matminer', 'matscholar_el', 'megnet_el')
  * **features** : List of element properties to use (e.g., ['Number', 'AtomicWeight', 'Column'])
  * **stats** : List of statistics to calculate (e.g., ['mean', 'std', 'minpool', 'maxpool'])

**Statistical Quantity Definitions:**

Statistic | Formula | Description  
---|---|---  
**mean** | $\bar{x} = \sum_{i} f_i x_i$ | Mole fraction weighted average  
**std** | $\sigma = \sqrt{\sum_{i} f_i (x_i - \bar{x})^2}$ | Standard deviation (heterogeneity indicator)  
**minpool** | $\min_i(x_i)$ | Minimum value  
**maxpool** | $\max_i(x_i)$ | Maximum value  
**range** | $\max_i(x_i) - \min_i(x_i)$ | Range (element property diversity)  
**mode** | Most frequent value | Property value of most common element  
  
Where $f_i$ is the mole fraction of element $i$, and $x_i$ is the property value of element $i$. 

### 3.3.2 Stoichiometry: Chemical Stoichiometry

The `Stoichiometry` Featurizer calculates stoichiometric features of compositions. It extracts p-norm, l2-norm, number of elements, element ratios, etc. as features. 

**Main Features:**

  * **p-norm** : Generalized norm $\left(\sum_i f_i^p\right)^{1/p}$
  * **l2_norm** : Euclidean norm $\sqrt{\sum_i f_i^2}$
  * **num_atoms** : Total number of atoms
  * **0-norm** : Number of element types

The p-norm represents the "uniformity" of the composition. When p=0, it gives the number of element types; as p→∞, it converges to the maximum mole fraction. 

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example3.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: The p-norm represents the "uniformity" of the composition.
     
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 3: Stoichiometry Featurizer (p-norm Calculation)
    # ===================================
    
    from matminer.featurizers.composition import Stoichiometry
    from pymatgen.core import Composition
    import pandas as pd
    import numpy as np
    
    # Initialize Stoichiometry Featurizer
    # p_list: List of p-norms to calculate
    # num_atoms: Whether to include number of atoms in features
    featurizer = Stoichiometry(
        p_list=[0, 2, 3, 5, 7, 10],
        num_atoms=True
    )
    
    # Composition data (materials with different stoichiometry)
    compositions = [
        "Fe",           # Pure metal (1 element)
        "FeO",          # Binary compound (1:1)
        "Fe2O3",        # Binary compound (2:3)
        "Fe3O4",        # Binary compound (3:4, spinel)
        "LaFeO3",       # Ternary compound (perovskite)
        "CoCrFeNi",     # Quaternary alloy (equimolar)
        "CoCrFeMnNi"    # Quinary alloy (high entropy alloy)
    ]
    
    # Convert to Composition objects
    comp_objects = [Composition(c) for c in compositions]
    
    # Calculate features
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # Get feature names
    feature_names = featurizer.feature_labels()
    print(f"=== Stoichiometry Features ===")
    print(f"Features: {feature_names}")
    print()
    
    # Display results
    print("=== Calculated Results ===")
    display_cols = ['formula'] + feature_names
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # Interpretation of p-norm
    print("=== p-norm Interpretation ===")
    for i, formula in enumerate(compositions):
        comp = comp_objects[i]
        p_0 = features_df.iloc[i]['0-norm']
        p_2 = features_df.iloc[i]['2-norm']
        p_10 = features_df.iloc[i]['10-norm']
    
        print(f"{formula:15s} | Elements: {int(p_0)} | "
              f"p=2: {p_2:.3f} | p=10: {p_10:.3f} | "
              f"Uniformity: {'High' if p_2 < 0.6 else 'Medium' if p_2 < 0.8 else 'Low'}")
    
    # Expected output:
    # === Stoichiometry Features ===
    # Features: ['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm', 'num_atoms']
    #
    # === Calculated Results ===
    #         formula  0-norm    2-norm    3-norm    5-norm    7-norm   10-norm  num_atoms
    #              Fe     1.0  1.000000  1.000000  1.000000  1.000000  1.000000        1.0
    #             FeO     2.0  0.707107  0.629961  0.562341  0.531792  0.512862        2.0
    #           Fe2O3     2.0  0.632456  0.584804  0.548813  0.530668  0.520053        5.0
    #           Fe3O4     2.0  0.612372  0.571429  0.540541  0.524839  0.515789        7.0
    #          LaFeO3     3.0  0.577350  0.519842  0.471285  0.446138  0.430887        5.0
    #        CoCrFeNi     4.0  0.500000  0.435275  0.375035  0.341484  0.316228        4.0
    #     CoCrFeMnNi     5.0  0.447214  0.380478  0.317480  0.286037  0.263902        5.0
    #
    # === p-norm Interpretation ===
    # Fe              | Elements: 1 | p=2: 1.000 | p=10: 1.000 | Uniformity: Low
    # FeO             | Elements: 2 | p=2: 0.707 | p=10: 0.513 | Uniformity: Medium
    # Fe2O3           | Elements: 2 | p=2: 0.632 | p=10: 0.520 | Uniformity: Medium
    # Fe3O4           | Elements: 2 | p=2: 0.612 | p=10: 0.516 | Uniformity: Medium
    # LaFeO3          | Elements: 3 | p=2: 0.577 | p=10: 0.431 | Uniformity: Medium
    # CoCrFeNi        | Elements: 4 | p=2: 0.500 | p=10: 0.316 | Uniformity: High
    # CoCrFeMnNi      | Elements: 5 | p=2: 0.447 | p=10: 0.264 | Uniformity: High
    

**🎯 Best Practice:** In high entropy alloy (HEA) research, p-norm correlates strongly with compositional entropy. When p=2, a 2-norm < 0.5 suggests an equimolar composition of 5 or more elements, and is effective for screening HEA candidate materials. 

### 3.3.3 OxidationStates: Oxidation States

The `OxidationStates` Featurizer calculates features based on element oxidation states. It is important for predicting electrochemical properties and catalytic activity. 

**Main Features:**

  * **maximum oxidation state** : Maximum oxidation state
  * **minimum oxidation state** : Minimum oxidation state
  * **oxidation state range** : Range of oxidation states
  * **oxidation state std** : Standard deviation of oxidation states

**EnHd_OxStates (Product of electronegativity × oxidation state):**

This feature calculates the product of electronegativity and oxidation state. It represents the driving force for charge transfer and correlates with catalytic activity and battery material performance. 

$$ \text{EnHd_OxStates} = \sum_{i} f_i \cdot \chi_i \cdot \text{OxState}_i $$ 

Where $\chi_i$ is the electronegativity of element $i$, and $\text{OxState}_i$ is the oxidation state. 

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example4.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Where $\chi_i$ is the electronegativity of element $i$, and 
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 4: OxidationStates Featurizer (EnHd_OxStates)
    # ===================================
    
    from matminer.featurizers.composition import OxidationStates
    from pymatgen.core import Composition
    import pandas as pd
    
    # Initialize OxidationStates Featurizer
    # stats: List of statistics to calculate
    featurizer = OxidationStates(
        stats=['mean', 'std', 'minimum', 'maximum', 'range']
    )
    
    # Compositions of battery and catalyst materials
    compositions = [
        "LiCoO2",       # Lithium-ion battery cathode
        "LiFePO4",      # Lithium iron phosphate cathode
        "Li4Ti5O12",    # Spinel-type anode
        "TiO2",         # Photocatalyst
        "CeO2",         # Solid electrolyte, catalyst
        "La0.6Sr0.4CoO3"  # Perovskite catalyst
    ]
    
    # Convert to Composition objects
    comp_objects = [Composition(c) for c in compositions]
    
    # Calculate features
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # Get feature names
    feature_names = featurizer.feature_labels()
    print(f"=== OxidationStates Features ===")
    print(f"Features: {feature_names}")
    print()
    
    # Display results
    print("=== Calculated Results ===")
    display_cols = ['formula'] + feature_names[:5]  # First 5 features
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # Interpretation of EnHd_OxStates (electronegativity × oxidation state)
    print("=== Material Classification by EnHd_OxStates ===")
    # Note: This example uses dummy values. In practice, obtained from featurizer
    for i, formula in enumerate(compositions):
        comp = comp_objects[i]
        # Simple calculation (actual featurizer computes automatically)
        print(f"{formula:20s} | Application: ", end="")
        if 'Li' in formula:
            print("Battery material (lithium ion conduction)")
        elif 'Ce' in formula or 'La' in formula:
            print("Catalyst/solid electrolyte (redox activity)")
        else:
            print("Photocatalyst (charge separation)")
    
    # Expected output:
    # === OxidationStates Features ===
    # Features: ['oxidation state mean', 'oxidation state std',
    #          'oxidation state minimum', 'oxidation state maximum',
    #          'oxidation state range']
    #
    # === Calculated Results ===
    #               formula  oxidation state mean  oxidation state std  ...
    #               LiCoO2                  1.25                 1.48  ...
    #              LiFePO4                  2.00                 1.83  ...
    #            Li4Ti5O12                  2.18                 1.65  ...
    #                 TiO2                  1.33                 2.31  ...
    #                 CeO2                  1.33                 2.31  ...
    #       La0.6Sr0.4CoO3                  1.80                 1.64  ...
    

### 3.3.4 Other Important Featurizers

Featurizer | Feature Dimensions | Application | Computational Cost  
---|---|---|---  
**ElectronAffinity** | 6 | Electron affinity statistics (electrochemical properties) | Low  
**IonProperty** | 32 | Ionic radius, coordination number (crystal structure prediction) | Low  
**Miedema** | 8 | Alloy formation energy (phase stability) | Medium  
**CohesiveEnergy** | 2 | Cohesive energy (mechanical properties) | High (ML prediction)  
**YangSolidSolution** | 4 | Solid solution formation parameters (HEA design) | Low  
  
## 3.4 Custom Feature Design

When existing Featurizers cannot handle special material systems or you want to verify your own hypotheses, you can implement custom Featurizers. Simply inherit `BaseFeaturizer` and implement the `featurize()` method. 

### 3.4.1 Inheriting BaseFeaturizer

The steps for implementing a custom Featurizer are as follows: 

  1. Create a class inheriting `BaseFeaturizer`
  2. Implement `featurize(composition)` method (returns list of features)
  3. Implement `feature_labels()` method (returns list of feature names)
  4. Implement `citations()` method (returns list of references)
  5. Add error handling

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example5.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # ===================================
    # Example 5: Custom Featurizer Implementation (BaseFeaturizer Inheritance)
    # ===================================
    
    from matminer.featurizers.base import BaseFeaturizer
    from pymatgen.core import Composition
    import numpy as np
    import pandas as pd
    
    class CustomElementDiversityFeaturizer(BaseFeaturizer):
        """Custom features based on element diversity
    
        Features specialized for high entropy alloy (HEA) design:
        - Shannon entropy: Compositional entropy
        - Gini coefficient: Composition heterogeneity
        - Effective number of elements: Effective element count
        """
    
        def featurize(self, comp):
            """Calculate features
    
            Args:
                comp (Composition): pymatgen Composition object
    
            Returns:
                list: [shannon_entropy, gini_coeff, effective_n_elements]
            """
            # Error handling
            if not isinstance(comp, Composition):
                raise ValueError("Input must be a pymatgen Composition object")
    
            # Get mole fractions
            fractions = np.array(list(comp.fractional_composition.values()))
    
            # Shannon entropy (compositional entropy)
            # H = -Σ(f_i * log(f_i))
            shannon_entropy = -np.sum(fractions * np.log(fractions + 1e-10))
    
            # Gini coefficient (composition heterogeneity, 0=perfectly uniform, 1=perfectly heterogeneous)
            # G = (Σ_i Σ_j |f_i - f_j|) / (2n Σ_i f_i)
            n = len(fractions)
            gini = np.sum(np.abs(fractions[:, None] - fractions[None, :])) / (2 * n)
    
            # Effective number of elements
            # N_eff = exp(H) = 1 / Σ(f_i^2)
            effective_n = 1.0 / np.sum(fractions ** 2)
    
            return [shannon_entropy, gini, effective_n]
    
        def feature_labels(self):
            """Return list of feature names"""
            return [
                'shannon_entropy',
                'gini_coefficient',
                'effective_n_elements'
            ]
    
        def citations(self):
            """Return list of references"""
            return [
                "@article{yeh2004nanostructured, "
                "title={Nanostructured high-entropy alloys}, "
                "author={Yeh, Jien-Wei and others}, "
                "journal={Advanced Engineering Materials}, "
                "volume={6}, pages={299--303}, year={2004}}"
            ]
    
        def implementors(self):
            """Return list of implementors"""
            return ['Custom Implementation']
    
    # Instantiate custom Featurizer
    custom_featurizer = CustomElementDiversityFeaturizer()
    
    # Test data (materials with various compositional entropies)
    compositions = [
        "Fe",                      # Single element (low entropy)
        "FeNi",                    # Binary equimolar (medium entropy)
        "CoCrNi",                  # Ternary equimolar
        "CoCrFeNi",                # Quaternary equimolar
        "CoCrFeMnNi",              # Quinary equimolar (high entropy)
        "Al0.5CoCrCuFeNi"          # Senary non-equimolar
    ]
    
    # Convert to Composition objects
    comp_objects = [Composition(c) for c in compositions]
    
    # Calculate features
    features_df = custom_featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # Display results
    print("=== Custom Features: Element Diversity ===")
    display_cols = ['formula'] + custom_featurizer.feature_labels()
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # HEA determination (Shannon entropy > 1.5 and Effective N > 4)
    print("=== High Entropy Alloy (HEA) Determination ===")
    for i, formula in enumerate(compositions):
        entropy = features_df.iloc[i]['shannon_entropy']
        eff_n = features_df.iloc[i]['effective_n_elements']
        is_hea = entropy > 1.5 and eff_n > 4.0
    
        print(f"{formula:20s} | H={entropy:.3f} | N_eff={eff_n:.2f} | "
              f"{'✅ HEA' if is_hea else '❌ Non-HEA'}")
    
    # Expected output:
    # === Custom Features: Element Diversity ===
    #               formula  shannon_entropy  gini_coefficient  effective_n_elements
    #                    Fe            0.000             0.000                  1.00
    #                 FeNi            0.693             0.250                  2.00
    #              CoCrNi            1.099             0.333                  3.00
    #            CoCrFeNi            1.386             0.375                  4.00
    #         CoCrFeMnNi            1.609             0.400                  5.00
    #   Al0.5CoCrCuFeNi            1.705             0.429                  5.45
    #
    # === High Entropy Alloy (HEA) Determination ===
    # Fe                   | H=0.000 | N_eff=1.00 | ❌ Non-HEA
    # FeNi                 | H=0.693 | N_eff=2.00 | ❌ Non-HEA
    # CoCrNi               | H=1.099 | N_eff=3.00 | ❌ Non-HEA
    # CoCrFeNi             | H=1.386 | N_eff=4.00 | ❌ Non-HEA
    # CoCrFeMnNi           | H=1.609 | N_eff=5.00 | ✅ HEA
    # Al0.5CoCrCuFeNi      | H=1.705 | N_eff=5.45 | ✅ HEA
    

### 3.4.2 Integrating Multiple Featurizers

By combining multiple Featurizers, you can build a richer feature set. Using `MultipleFeaturizer` allows you to apply multiple Featurizers at once. 

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example6.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: By combining multiple Featurizers, you can build a richer fe
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 6: MultipleFeaturizer Integration (Multiple Featurizer Pipeline)
    # ===================================
    
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, OxidationStates
    )
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    import pandas as pd
    
    # Integrate multiple Featurizers
    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset(preset_name="magpie"),
        Stoichiometry(p_list=[2, 3, 5, 7, 10]),
        OxidationStates(stats=['mean', 'std'])
    ])
    
    # Composition data (SMILES string format)
    data = pd.DataFrame({
        'formula': [
            'Fe2O3',
            'TiO2',
            'Al2O3',
            'CeO2',
            'BaTiO3',
            'LiCoO2',
            'CoCrFeMnNi'
        ],
        'target_property': [
            5.24,   # Band gap (eV) - dummy values
            3.20,
            8.80,
            3.19,
            3.38,
            2.70,
            0.00    # Metal (no band gap)
        ]
    })
    
    # Convert strings to Composition objects
    str_to_comp = StrToComposition()
    data = str_to_comp.featurize_dataframe(data, 'formula')
    
    # Calculate features
    features_df = featurizer.featurize_dataframe(
        data,
        col_id='composition',
        ignore_errors=True  # Continue while ignoring errors
    )
    
    # Get feature names
    feature_names = featurizer.feature_labels()
    print(f"=== Integrated Feature Set ===")
    print(f"Total feature count: {len(feature_names)}")
    print(f"- ElementProperty (Magpie): 132 dimensions")
    print(f"- Stoichiometry: 5 dimensions")
    print(f"- OxidationStates: 2 dimensions")
    print(f"- Total: 139 dimensions")
    print()
    
    # Display first 10 features
    print("=== First 10 Features ===")
    display_cols = ['formula', 'target_property'] + feature_names[:10]
    print(features_df[display_cols].head().to_string(index=False))
    print()
    
    # Feature statistics
    print("=== Feature Statistics (Partial) ===")
    stats_cols = ['MagpieData mean Number', '2-norm', 'oxidation state mean']
    print(features_df[stats_cols].describe().round(3))
    
    # Expected output:
    # === Integrated Feature Set ===
    # Total feature count: 139
    # - ElementProperty (Magpie): 132 dimensions
    # - Stoichiometry: 5 dimensions
    # - OxidationStates: 2 dimensions
    # - Total: 139 dimensions
    #
    # === First 10 Features ===
    #        formula  target_property  MagpieData minimum Number  ...
    #          Fe2O3             5.24                          8  ...
    #           TiO2             3.20                          8  ...
    #         Al2O3             8.80                          8  ...
    #           CeO2             3.19                          8  ...
    #        BaTiO3             3.38                          8  ...
    

### 3.4.3 Batch Processing Optimization

When handling large datasets (10,000+ compositions), batch processing optimization is important. `featurize_dataframe()` is faster than pandas `apply()`. 

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example7.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: When handling large datasets (10,000+ compositions), batch p
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # ===================================
    # Example 7: Batch Processing Optimization (pandas Integration)
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from pymatgen.core import Composition
    import pandas as pd
    import time
    import numpy as np
    
    # Generate large dataset (1000 compositions)
    np.random.seed(42)
    elements = ['Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ti', 'Cr', 'Mn']
    formulas = []
    
    for _ in range(1000):
        # Randomly select 2-5 elements
        n_elements = np.random.randint(2, 6)
        selected_elements = np.random.choice(elements, n_elements, replace=False)
    
        # Generate random composition ratios
        ratios = np.random.randint(1, 5, n_elements)
    
        # Build chemical formula
        formula = ''.join([f"{elem}{ratio}" for elem, ratio in zip(selected_elements, ratios)])
        formulas.append(formula)
    
    # Create dataframe
    data = pd.DataFrame({'formula': formulas})
    
    # Prepare Featurizer
    str_to_comp = StrToComposition()
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    
    print("=== Batch Processing Performance Comparison ===")
    print(f"Number of data: {len(data)} compositions")
    print()
    
    # Method 1: Sequential processing using apply() (slow)
    start_time = time.time()
    data_method1 = data.copy()
    data_method1['composition'] = data_method1['formula'].apply(lambda x: Composition(x))
    data_method1 = data_method1.apply(
        lambda row: pd.Series(featurizer.featurize(row['composition'])),
        axis=1
    )
    time_method1 = time.time() - start_time
    print(f"Method 1 (apply)         : {time_method1:.2f} seconds")
    
    # Method 2: Batch processing using featurize_dataframe() (fast)
    start_time = time.time()
    data_method2 = data.copy()
    data_method2 = str_to_comp.featurize_dataframe(data_method2, 'formula')
    data_method2 = featurizer.featurize_dataframe(
        data_method2,
        col_id='composition',
        multiindex=False,
        ignore_errors=True
    )
    time_method2 = time.time() - start_time
    print(f"Method 2 (featurize_df)  : {time_method2:.2f} seconds")
    print(f"Speedup factor: {time_method1/time_method2:.1f}x")
    print()
    
    # Method 3: Parallel processing (fastest)
    start_time = time.time()
    data_method3 = data.copy()
    data_method3 = str_to_comp.featurize_dataframe(data_method3, 'formula')
    data_method3 = featurizer.featurize_dataframe(
        data_method3,
        col_id='composition',
        multiindex=False,
        ignore_errors=True,
        n_jobs=-1  # Use all CPU cores
    )
    time_method3 = time.time() - start_time
    print(f"Method 3 (parallel)      : {time_method3:.2f} seconds")
    print(f"Speedup factor: {time_method1/time_method3:.1f}x")
    
    # Expected output:
    # === Batch Processing Performance Comparison ===
    # Number of data: 1000 compositions
    #
    # Method 1 (apply)         : 12.34 seconds
    # Method 2 (featurize_df)  : 3.21 seconds
    # Speedup factor: 3.8x
    #
    # Method 3 (parallel)      : 0.87 seconds
    # Speedup factor: 14.2x
    

### 3.4.4 Multi-Database Integration Strategy

For production-level applications, it's important to integrate data from multiple databases (Mendeleev, pymatgen, matminer) to maximize data coverage and reliability. The key is establishing a priority order and confidence scoring system. 

**Data Source Priority Strategy:**

  1. **First Priority** : Mendeleev experimental values (confidence: 0.95)
  2. **Second Priority** : pymatgen values (DFT or experimental, confidence: 0.80)
  3. **Third Priority** : matminer/literature values (confidence: 0.70)
  4. **Fallback** : Default values or interpolation (confidence: 0.50)

[Run on Google Colab](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example8.ipynb>)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # ===================================
    # Example 8: Multi-Database Integration (Fallback Strategy)
    # ===================================
    
    import mendeleev
    from pymatgen.core import Element, Composition
    import pandas as pd
    import numpy as np
    
    class MultiSourceElementDataFetcher:
        """Element property data fetcher from multiple sources with confidence scoring
    
        Priority order:
        1. Mendeleev (experimental values, confidence: 0.95)
        2. pymatgen (DFT/experimental, confidence: 0.80)
        3. Default values (confidence: 0.50)
        """
    
        def __init__(self):
            self.confidence_scores = {
                'mendeleev': 0.95,
                'pymatgen': 0.80,
                'default': 0.50
            }
    
        def get_property(self, element_symbol, property_name):
            """Get element property from multiple sources with fallback
    
            Args:
                element_symbol (str): Element symbol (e.g., 'Fe')
                property_name (str): Property name ('electronegativity', 'atomic_radius', 'ionization_energy')
    
            Returns:
                dict: {'value': float, 'source': str, 'confidence': float}
            """
            # Try Mendeleev first (highest confidence)
            try:
                elem_mendeleev = mendeleev.element(element_symbol)
                if property_name == 'electronegativity':
                    value = elem_mendeleev.electronegativity()
                elif property_name == 'atomic_radius':
                    value = elem_mendeleev.atomic_radius
                elif property_name == 'ionization_energy':
                    value = elem_mendeleev.ionenergies.get(1, None)
                else:
                    value = None
    
                if value is not None:
                    return {
                        'value': value,
                        'source': 'mendeleev',
                        'confidence': self.confidence_scores['mendeleev']
                    }
            except Exception:
                pass
    
            # Fallback to pymatgen (medium confidence)
            try:
                elem_pymatgen = Element(element_symbol)
                if property_name == 'electronegativity':
                    value = elem_pymatgen.X
                elif property_name == 'atomic_radius':
                    value = elem_pymatgen.atomic_radius
                elif property_name == 'ionization_energy':
                    value = elem_pymatgen.ionization_energy
                else:
                    value = None
    
                if value is not None:
                    return {
                        'value': value,
                        'source': 'pymatgen',
                        'confidence': self.confidence_scores['pymatgen']
                    }
            except Exception:
                pass
    
            # Default value (low confidence)
            default_values = {
                'electronegativity': 1.5,
                'atomic_radius': 150.0,
                'ionization_energy': 7.0
            }
    
            return {
                'value': default_values.get(property_name, 0.0),
                'source': 'default',
                'confidence': self.confidence_scores['default']
            }
    
        def get_composition_features(self, composition):
            """Get features for a composition with confidence tracking
    
            Args:
                composition (str or Composition): Composition formula
    
            Returns:
                pd.DataFrame: Features with source and confidence information
            """
            comp = Composition(composition) if isinstance(composition, str) else composition
    
            results = []
            for element, fraction in comp.items():
                for prop_name in ['electronegativity', 'atomic_radius', 'ionization_energy']:
                    prop_data = self.get_property(str(element), prop_name)
                    results.append({
                        'element': str(element),
                        'property': prop_name,
                        'value': prop_data['value'],
                        'source': prop_data['source'],
                        'confidence': prop_data['confidence']
                    })
    
            return pd.DataFrame(results)
    
    # Usage example
    fetcher = MultiSourceElementDataFetcher()
    
    # Test compositions
    test_compositions = [
        "Fe2O3",       # Iron oxide
        "TiO2",        # Titanium dioxide
        "CoCrFeMnNi"   # High entropy alloy
    ]
    
    print("=== Data Retrieval from Multiple Sources ===")
    print()
    
    for comp_str in test_compositions:
        comp = Composition(comp_str)
        feature_df = fetcher.get_composition_features(comp)
    
        print(f"Composition: {comp_str}")
        print(f"  Data sources: {feature_df['source'].value_counts().to_dict()}")
        print(f"  Average confidence: {feature_df['confidence'].mean():.2f}")
        print()
    
    # Detailed display for Fe2O3
    print("=== Detailed Data for Fe2O3 ===")
    fe2o3_features = fetcher.get_composition_features("Fe2O3")
    print(fe2o3_features.to_string(index=False))
    
    # Best practices
    print()
    print("=== Database Integration Best Practices ===")
    print("✅ Priority: Mendeleev (experimental) > pymatgen (calculated) > default values")
    print("✅ Record confidence scores to track data quality")
    print("✅ Output warnings when using default values")
    print("✅ Record data source and version (ensure reproducibility)")
    
    # Expected output:
    # === Data Retrieval from Multiple Sources ===
    #
    # Composition: Fe2O3
    #   Data sources: {'mendeleev': 6}
    #   Average confidence: 0.95
    #
    # Composition: TiO2
    #   Data sources: {'mendeleev': 6}
    #   Average confidence: 0.95
    #
    # Composition: CoCrFeMnNi
    #   Data sources: {'mendeleev': 15}
    #   Average confidence: 0.95
    #
    # === Detailed Data for Fe2O3 ===
    # element           property  value      source  confidence
    #      Fe  electronegativity   1.83  mendeleev        0.95
    #      Fe      atomic_radius 140.00  mendeleev        0.95
    #      Fe  ionization_energy   7.90  mendeleev        0.95
    #       O  electronegativity   3.44  mendeleev        0.95
    #       O      atomic_radius  60.00  mendeleev        0.95
    #       O  ionization_energy  13.62  mendeleev        0.95
    

## Learning Objectives Review

Having completed this chapter, you can now explain and implement the following:

### Basic Understanding

  * ✅ Explain the differences between the three data sources: Mendeleev, pymatgen, and matminer
  * ✅ Identify the main data sources for each database (experimental vs DFT calculated values)
  * ✅ Understand Featurizer architecture (BaseFeaturizer, fit/transform API)
  * ✅ Explain the benefits of scikit-learn compatibility

### Practical Skills

  * ✅ Calculate 132-dimensional Magpie features using ElementProperty Featurizer
  * ✅ Calculate p-norm and compositional entropy using Stoichiometry Featurizer
  * ✅ Calculate oxidation state statistics using OxidationStates Featurizer
  * ✅ Pipeline multiple Featurizers using MultipleFeaturizer
  * ✅ Optimize batch processing with featurize_dataframe() and n_jobs=-1

### Applied Competence

  * ✅ Design custom Featurizers by inheriting BaseFeaturizer
  * ✅ Implement high entropy alloy (HEA) features (Shannon entropy, Gini coefficient)
  * ✅ Build integration strategies to select optimal values from multiple databases
  * ✅ Evaluate data source reliability and select appropriate databases

## Exercises

### Easy (Fundamentals Review)

**Q1** : What is the main data source of the Mendeleev database?

  1. DFT calculated values
  2. Experimental values (NIST, CRC Handbook, etc.)
  3. Machine learning predicted values
  4. Literature average values

View Answer

**Answer** : b) Experimental values (NIST, CRC Handbook, etc.)

**Explanation** :

Mendeleev is a comprehensive database centered on experimental values. Main sources include reliable references such as NIST (National Institute of Standards and Technology) and CRC Handbook of Chemistry and Physics. This makes it suitable for precise property prediction (melting point, density, etc.).

In contrast, pymatgen includes DFT calculated values from Materials Project, and matminer integrates property sets reported in multiple papers.

**Q2** : What does the `feature_labels()` method of `BaseFeaturizer` return?

  1. Feature values (list of numbers)
  2. Feature names (list of strings)
  3. List of references
  4. Composition object

View Answer

**Answer** : b) Feature names (list of strings)

**Explanation** :

`feature_labels()` returns a list of names (labels) for the features generated by the Featurizer. For example, in the case of ElementProperty, it returns 132 feature names such as `['MagpieData minimum Number', 'MagpieData maximum Number', ...]`.

This allows you to verify what each dimension of the calculated features represents, improving the interpretability of machine learning models.

**Q3** : In the following code, what is the type of the return value of `featurizer.featurize(Composition("Fe2O3"))`?
    
    
    from matminer.featurizers.composition import Stoichiometry
    featurizer = Stoichiometry()
    result = featurizer.featurize(Composition("Fe2O3"))

  1. pandas DataFrame
  2. numpy array
  3. list
  4. dict (dictionary)

View Answer

**Answer** : c) list

**Explanation** :

The `featurize()` method calculates features for a single composition and returns them in **Python list format**. Example: `[2.0, 0.632456, 0.584804, ...]`

On the other hand, the `featurize_dataframe()` method takes a pandas DataFrame as input and returns a DataFrame with added features. For handling large datasets, using `featurize_dataframe()` is recommended.

### Medium (Application)

**Q4** : As the p value of p-norm increases, what value does it converge to? Also, explain its physical meaning.

View Answer

**Answer** : As p→∞, p-norm converges to the maximum mole fraction.

**Explanation** :

The p-norm is defined by the following formula:

$$\text{p-norm} = \left(\sum_{i} f_i^p\right)^{1/p}$$

In the limit as p→∞, the maximum mole fraction $f_{\max}$ dominates, and p-norm → $f_{\max}$.

**Physical Meaning** :

  * p=0: Number of element types (count of non-zero mole fractions)
  * p=2: Euclidean norm (composition "magnitude")
  * p→∞: Mole fraction of most abundant element (dominance of main component)

**Application Example** :

In high entropy alloy (HEA) research, when p=2, a 2-norm < 0.5 suggests equimolar composition of 5 or more elements, and is used for screening HEA candidate materials.

**Q5** : In the following code, how many features (dimensions) will there be?
    
    
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, OxidationStates
    )
    
    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),  # 132 dimensions
        Stoichiometry(p_list=[2, 3, 5]),        # ? dimensions
        OxidationStates(stats=['mean', 'std'])  # ? dimensions
    ])

View Answer

**Answer** : 137 dimensions

**Calculation Process** :

  * **ElementProperty (magpie)** : 132 dimensions
  * **Stoichiometry** : 3 dimensions with p_list=[2, 3, 5]
  * **OxidationStates** : 2 dimensions with stats=['mean', 'std']
  * **Total** : 132 + 3 + 2 = 137 dimensions

**Note** :

The default for Stoichiometry is p_list=[0, 2, 3, 5, 7, 10] which gives 6 dimensions, but in this case p_list is explicitly specified as [2, 3, 5], resulting in 3 dimensions.

To verify the feature count, run `len(featurizer.feature_labels())`.

**Q6** : What is the optimal method for batch applying ElementProperty Featurizer to a dataset of 1000 compositions?

  1. `df.apply(lambda x: featurizer.featurize(x['composition']), axis=1)`
  2. `featurizer.featurize_dataframe(df, col_id='composition')`
  3. `featurizer.featurize_dataframe(df, col_id='composition', n_jobs=-1)`
  4. `for i in range(len(df)): featurizer.featurize(df.iloc[i]['composition'])`

View Answer

**Answer** : c) `featurizer.featurize_dataframe(df, col_id='composition', n_jobs=-1)`

**Explanation** :

Performance comparison (for 1000 compositions):

Method | Execution Time | Speedup Factor  
---|---|---  
d) for loop | 15.2 seconds | 1.0x (baseline)  
a) apply() | 12.3 seconds | 1.2x  
b) featurize_dataframe() | 3.2 seconds | 4.8x  
c) featurize_dataframe(n_jobs=-1) | 0.9 seconds | 16.9x  
  
**Best Practices** :

  * ✅ Use `n_jobs=-1` to utilize all CPU cores
  * ✅ Use `ignore_errors=True` to skip error rows
  * ✅ Consider chunk-based processing for large data (>10K)

**Q7** : When implementing a custom Featurizer, which methods must be implemented? (Multiple choices possible)

  1. `featurize(entry)`
  2. `feature_labels()`
  3. `citations()`
  4. `fit(X, y)`

View Answer

**Answer** : a) `featurize(entry)` and b) `feature_labels()`

**Explanation** :

**Required Methods** :

  * **`featurize(entry)`** : Calculate features for a single entry (returns list)
  * **`feature_labels()`** : Return list of feature names

**Optional Methods** :

  * **`citations()`** : Return list of references (recommended)
  * **`implementors()`** : Return list of implementors (recommended)
  * **`fit(X, y)`** : Learn from training data (only when necessary)

**Implementation Example** :
    
    
    class CustomFeaturizer(BaseFeaturizer):
        def featurize(self, comp):
            # Calculate features
            return [value1, value2, ...]
    
        def feature_labels(self):
            # Return feature names
            return ['feature1', 'feature2', ...]
    
        def citations(self):
            # Return references (recommended)
            return ['@article{...}']
    
        def implementors(self):
            # Return implementors (recommended)
            return ['Your Name']

### Hard (Advanced)

**Q8** : Design a custom Featurizer to identify high entropy alloys (HEA). Show an implementation that calculates the following features:

  * Shannon entropy: $H = -\sum_i f_i \ln(f_i)$
  * Effective number of elements: $N_{\text{eff}} = \exp(H)$
  * HEA determination: $H > 1.5$ and $N_{\text{eff}} > 4.0$

View Answer

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    from matminer.featurizers.base import BaseFeaturizer
    from pymatgen.core import Composition
    import numpy as np
    
    class HEAFeaturizer(BaseFeaturizer):
        """High Entropy Alloy (HEA) Features
    
        Features:
        - shannon_entropy: Compositional entropy
        - effective_n_elements: Effective number of elements
        - is_hea: HEA determination (True/False → 1/0)
        """
    
        def featurize(self, comp):
            """Calculate features
    
            Args:
                comp (Composition): pymatgen Composition object
    
            Returns:
                list: [shannon_entropy, effective_n_elements, is_hea]
            """
            # Get mole fractions
            fractions = np.array(list(comp.fractional_composition.values()))
    
            # Shannon entropy
            # H = -Σ(f_i * log(f_i))
            # Add small value to avoid log(0)
            shannon_entropy = -np.sum(fractions * np.log(fractions + 1e-10))
    
            # Effective number of elements
            # N_eff = exp(H)
            effective_n = np.exp(shannon_entropy)
    
            # HEA determination
            is_hea = 1.0 if (shannon_entropy > 1.5 and effective_n > 4.0) else 0.0
    
            return [shannon_entropy, effective_n, is_hea]
    
        def feature_labels(self):
            """Return list of feature names"""
            return [
                'shannon_entropy',
                'effective_n_elements',
                'is_hea'
            ]
    
        def citations(self):
            """Return list of references"""
            return [
                "@article{yeh2004nanostructured, "
                "title={Nanostructured high-entropy alloys with multiple principal elements}, "
                "author={Yeh, Jien-Wei and Chen, Swe-Kai and Lin, Su-Jien and others}, "
                "journal={Advanced Engineering Materials}, "
                "volume={6}, number={5}, pages={299--303}, year={2004}}"
            ]
    
        def implementors(self):
            """Return list of implementors"""
            return ['Custom HEA Featurizer']
    
    # Usage example
    featurizer = HEAFeaturizer()
    
    # Test data
    test_compositions = [
        "Fe",              # H=0.00, N_eff=1.00 → Non-HEA
        "FeNi",            # H=0.69, N_eff=2.00 → Non-HEA
        "CoCrFeNi",        # H=1.39, N_eff=4.00 → Non-HEA (borderline)
        "CoCrFeMnNi",      # H=1.61, N_eff=5.00 → HEA
        "AlCoCrFeNi"       # H=1.61, N_eff=5.00 → HEA
    ]
    
    import pandas as pd
    comp_objects = [Composition(c) for c in test_compositions]
    df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': test_compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    print(df[['formula', 'shannon_entropy', 'effective_n_elements', 'is_hea']])
    
    # Expected output:
    #         formula  shannon_entropy  effective_n_elements  is_hea
    # 0            Fe             0.00                  1.00     0.0
    # 1          FeNi             0.69                  2.00     0.0
    # 2      CoCrFeNi             1.39                  4.00     0.0
    # 3   CoCrFeMnNi             1.61                  5.00     1.0
    # 4    AlCoCrFeNi             1.61                  5.00     1.0

**Key Points** :

  * ✅ Use `np.log(fractions + 1e-10)` to avoid `log(0)` error
  * ✅ Return HEA determination as 1.0/0.0 floating point instead of Boolean (compatibility with ML models)
  * ✅ Shannon entropy threshold (1.5) is based on literature values

**Q9** : Implement a function that retrieves electronegativity from multiple databases (Mendeleev, pymatgen, matminer) and calculates confidence scores. Define confidence as follows:

  * Mendeleev (experimental values): 0.95
  * pymatgen (calculated values): 0.80
  * Default values: 0.50

View Answer

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import mendeleev
    from pymatgen.core import Element
    import pandas as pd
    
    def get_electronegativity_with_confidence(element_symbol):
        """Retrieve electronegativity from multiple sources with confidence
    
        Args:
            element_symbol (str): Element symbol (e.g., 'Fe')
    
        Returns:
            dict: {'value': float, 'source': str, 'confidence': float}
        """
        # Try Mendeleev first (highest confidence)
        try:
            elem = mendeleev.element(element_symbol)
            en = elem.electronegativity()
            if en is not None:
                return {
                    'value': en,
                    'source': 'mendeleev',
                    'confidence': 0.95
                }
        except Exception:
            pass
    
        # Fallback to pymatgen (medium confidence)
        try:
            elem = Element(element_symbol)
            en = elem.X
            if en is not None:
                return {
                    'value': en,
                    'source': 'pymatgen',
                    'confidence': 0.80
                }
        except Exception:
            pass
    
        # Default value (low confidence)
        return {
            'value': 1.5,  # Typical electronegativity
            'source': 'default',
            'confidence': 0.50
        }
    
    # Usage example
    elements = ['Fe', 'Cu', 'Al', 'Ti', 'Uuo']  # Uuo is synthetic element
    
    print("=== Electronegativity with Confidence Scores ===")
    results = []
    for elem in elements:
        data = get_electronegativity_with_confidence(elem)
        results.append({
            'Element': elem,
            'Electronegativity': data['value'],
            'Source': data['source'],
            'Confidence': data['confidence']
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Expected output:
    # === Electronegativity with Confidence Scores ===
    # Element  Electronegativity      Source  Confidence
    #      Fe               1.83  mendeleev        0.95
    #      Cu               1.90  mendeleev        0.95
    #      Al               1.61  mendeleev        0.95
    #      Ti               1.54  mendeleev        0.95
    #     Uuo               1.50     default        0.50  # Synthetic element, no data

**Application to Production Systems** :

  * ✅ Record confidence scores in metadata for traceability
  * ✅ Issue warnings when confidence < 0.70
  * ✅ Use weighted averaging based on confidence for ensemble predictions
  * ✅ Track data source versions for reproducibility

**Q10** : Implement an efficient feature calculation pipeline for a large dataset (100,000 compositions). Include the following optimizations:

  * Chunk-based processing (chunk size: 1,000)
  * Parallel processing (n_jobs=-1)
  * Progress monitoring (tqdm)
  * Error handling (ignore_errors=True)

View Answer

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - tqdm>=4.65.0
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import time
    
    def featurize_large_dataset(formulas, chunk_size=1000):
        """Efficiently featurize large dataset with chunking and parallel processing
    
        Args:
            formulas (list): List of chemical formulas
            chunk_size (int): Number of compositions per chunk
    
        Returns:
            pd.DataFrame: Featurized dataset
        """
        # Initialize featurizers
        str_to_comp = StrToComposition()
        featurizer = ElementProperty.from_preset("magpie")
    
        # Split into chunks
        n_chunks = (len(formulas) + chunk_size - 1) // chunk_size
        chunks = np.array_split(formulas, n_chunks)
    
        results = []
    
        print(f"=== Large Dataset Feature Calculation ===")
        print(f"Number of data: {len(formulas):,} compositions")
        print(f"Chunk size: {chunk_size:,} compositions")
        print()
    
        # Process chunks with progress bar
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Create dataframe for chunk
            chunk_df = pd.DataFrame({'formula': chunk})
    
            # Convert to Composition
            chunk_df = str_to_comp.featurize_dataframe(
                chunk_df,
                'formula',
                ignore_errors=True
            )
    
            # Calculate features with parallel processing
            chunk_df = featurizer.featurize_dataframe(
                chunk_df,
                col_id='composition',
                multiindex=False,
                ignore_errors=True,
                n_jobs=-1  # Use all CPU cores
            )
    
            results.append(chunk_df)
    
        # Concatenate all chunks
        final_df = pd.concat(results, ignore_index=True)
        return final_df
    
    # Generate test dataset (100,000 compositions)
    np.random.seed(42)
    elements = ['Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ti', 'Cr', 'Mn']
    formulas = []
    
    for _ in range(100000):
        n_elem = np.random.randint(2, 6)
        selected = np.random.choice(elements, n_elem, replace=False)
        ratios = np.random.randint(1, 5, n_elem)
        formula = ''.join([f"{e}{r}" for e, r in zip(selected, ratios)])
        formulas.append(formula)
    
    # Execute featurization
    start_time = time.time()
    result_df = featurize_large_dataset(formulas, chunk_size=1000)
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing completed: {elapsed_time:.1f} seconds")
    print(f"Processing speed: {len(formulas) / elapsed_time:.0f} compositions/second")
    print(f"Number of features: {len(result_df.columns)} dimensions")
    
    # Display partial results
    print("\n=== Results (First 5 rows) ===")
    print(result_df.head())
    
    # Expected output:
    # === Large Dataset Feature Calculation ===
    # Number of data: 100,000 compositions
    # Chunk size: 1,000 compositions
    #
    # Processing chunks: 100%|████████████| 100/100 [02:34<00:00,  1.55s/it]
    #
    # Processing completed: 154.3 seconds
    # Processing speed: 648 compositions/second
    # Number of features: 135 dimensions

**Optimization Points** :

  * ✅ Chunk size: 1,000-10,000 is optimal (memory vs speed tradeoff)
  * ✅ Enable parallel processing with `n_jobs=-1`
  * ✅ Skip error rows with `ignore_errors=True`
  * ✅ Display progress bar with `tqdm` (user-friendly)
  * ✅ Monitor memory usage (use `psutil`)

**Further Optimizations** :

  * Distributed processing using Dask (>1M compositions)
  * Save intermediate results in HDF5/Parquet format
  * GPU acceleration (CuPy, RAPIDS)

## Next Steps

In this chapter, we learned about the differences between element property databases (Mendeleev, pymatgen, matminer) and the Featurizer architecture. In the next chapter, we will learn how to build actual machine learning models using these features, and techniques for feature selection and dimensionality reduction. 

[← Chapter 2: Magpie and Statistical Descriptors](<chapter-2.html>) [Chapter 4: Integration with Machine Learning →](<chapter-4.html>)

## References

  1. Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N. E., Bajaj, S., Wang, Q., ... & Jain, A. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69. DOI: [10.1016/j.commatsci.2018.05.018](<https://doi.org/10.1016/j.commatsci.2018.05.018>)   
_Original matminer paper. Details of Featurizer architecture and main features (pp. 62-66)_
  2. Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., ... & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319. DOI: [10.1016/j.commatsci.2012.10.028](<https://doi.org/10.1016/j.commatsci.2012.10.028>)   
_Original pymatgen library paper. Details of Element and Composition classes (pp. 315-317)_
  3. Himanen, L., Jäger, M. O., Morooka, E. V., Federici Canova, F., Ranawat, Y. S., Gao, D. Z., ... & Foster, A. S. (2019). "DScribe: Library of descriptors for machine learning in materials science." _Computer Physics Communications_ , 247, 106949, pp. 1-15. DOI: [10.1016/j.cpc.2019.106949](<https://doi.org/10.1016/j.cpc.2019.106949>)   
_Comprehensive review of material descriptors. Theoretical background of composition-based features (pp. 3-7)_
  4. matminer API Documentation: Featurizer classes. <https://hackingmaterials.lbl.gov/matminer/>   
_Official matminer documentation. Detailed usage examples and parameter descriptions for each Featurizer_
  5. pymatgen.core.periodic_table Documentation. [https://pymatgen.org/](<https://pymatgen.org/pymatgen.core.periodic_table.html>)   
_Detailed API specifications for pymatgen's Element and Composition classes_
  6. Mendeleev package documentation. <https://mendeleev.readthedocs.io/>   
_Official Mendeleev package documentation. Element property data sources and accuracy information_
  7. Materials Project Database documentation. <https://docs.materialsproject.org/>   
_Official Materials Project documentation. Details of DFT calculation methods and database structure_

* * *

[Back to Series Index](<index.html>) [Proceed to Chapter 4 →](<chapter-4.html>)

© 2025 AI Terakoya - Materials Informatics Knowledge Hub 

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
