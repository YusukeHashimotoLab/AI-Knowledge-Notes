---
title: "Chapter 2: Fundamentals of MI - Concepts, Methods, and Ecosystem"
chapter_title: "Chapter 2: Fundamentals of MI - Concepts, Methods, and Ecosystem"
subtitle: Theory and Practice of Data-Driven Materials Development
reading_time: 20-25 minutes
difficulty: Beginner to Intermediate
code_examples: 0
exercises: 0
version: 3.0
created_at: 2025-10-16
---

# Chapter 2: Fundamentals of MI - Concepts, Methods, and Ecosystem

Grasp the complete picture of material descriptors and major databases, and learn to retrieve actual data using pymatgen/MP API. Understand the end-to-end workflow from feature generation with matminer.

**💡 Tip:** Think of Materials Project as an "atlas of materials" and pymatgen as the "tools to read the map." Start small with data acquisition → featurization workflow to build familiarity.

## Learning Objectives

By reading this chapter, you will be able to: \- Explain the definition of MI and its differences from related fields (computational materials science, cheminformatics, etc.) \- Understand the characteristics and use cases of major materials databases (Materials Project, AFLOW, OQMD, JARVIS) \- Explain the 5-step MI workflow in detail (from problem formulation to validation) \- Understand the types and importance of material descriptors (composition-based, structure-based, property-based) \- Correctly use 20 key technical terms in the MI domain

* * *

## 2.1 What is MI: Definition and Related Fields

### 2.1.1 Etymology and History of Materials Informatics

The term **Materials Informatics (MI)** began to be used in the early 2000s. It gained worldwide attention particularly after the launch of the **U.S. Materials Genome Initiative (MGI) in 2011** [1].

**MGI Goals:** Reduce the development time for new materials to half of conventional approaches, significantly reduce development costs, and accelerate through integration of computation, experiments, and data.

This initiative was expected to fundamentally transform materials science, just as the Human Genome Project revolutionized biology.

### 2.1.2 Definition

**Materials Informatics (MI)** is an interdisciplinary field that integrates materials science and data science. It is a methodology that leverages large amounts of materials data and information science technologies such as machine learning to accelerate the discovery of new materials and prediction of material properties.

**Concise definition:**

> "The science of accelerating materials development using the power of data and AI"

**Core elements:** 1\. **Data** : Experimental data, computational data, knowledge from literature 2\. **Computation** : First-principles calculations, molecular dynamics simulations 3\. **Machine Learning** : Predictive models, optimization algorithms 4\. **Experimental Validation** : Verification of predictions and data augmentation

### 2.1.3 Comparison with Related Fields

MI is related to multiple fields, but each has a different focus.

Field | Target | Main Methods | Purpose | Relationship with MI  
---|---|---|---|---  
**Computational Materials Science** | Physical and chemical phenomena in materials | DFT, Molecular Dynamics | Theoretical prediction of material properties | MI utilizes this data  
**Cheminformatics** | Compounds, small molecules | QSAR, molecular descriptors | Drug design, molecular property prediction | Shares descriptor concepts  
**Bioinformatics** | Biomolecules, DNA/proteins | Sequence analysis, structure prediction | Decoding genetic information | Shares data-driven approaches  
**Materials Informatics (MI)** | Solid materials in general | Machine Learning, Bayesian Optimization | Discovery and design of new materials | -  
  
**Uniqueness of MI:** The **inverse design approach** designs materials from target properties (conventional methods calculate properties from materials). MI addresses **diverse material types** including metals, ceramics, semiconductors, and polymers. It also emphasizes **strong experimental linkage** with validation, not just computation.

### 2.1.4 Forward Design vs Inverse Design

**Conventional materials development (Forward Design):**
    
    
    Material composition → Calculate/measure structure & properties → Evaluate results
    

Researchers propose candidate materials through repeated trial and error, making the process time-consuming.

**MI Approach (Inverse Design):**
    
    
    Target properties → Machine learning predicts candidate materials → Experiment on top candidates
    

AI proposes optimal materials, enabling efficient screening of large number of candidates and significant time reduction.

**Concrete example of inverse design:** "Want a semiconductor material with bandgap of 2.0 eV" → MI system automatically generates candidate material list → Researchers experimentally validate only top 10 candidates

* * *

## 2.2 MI Glossary: 20 Essential Terms

This section summarizes technical terms frequently encountered in MI. For beginners, properly understanding these terms is the first step.

### Data and Model Related (1-7)

Term (English) | Description  
---|---  
**1\. Descriptor** | Numerical representation of material features. Example: electronegativity, atomic radius, lattice constant. Used as input to machine learning models.  
**2\. Feature Engineering** | The process of designing and selecting descriptors suitable for machine learning from raw data. Critical step that affects model performance.  
**3\. Screening** | Efficiently selecting materials with desired properties from a large pool of candidates. Computational screening can evaluate thousands to tens of thousands of materials in a short time.  
**4\. Overfitting** | Phenomenon where a model "memorizes" training data and prediction performance on unknown data degrades. Requires special attention in materials science with limited data.  
**5\. Cross-validation** | Method to evaluate model generalization performance. Data is divided into K folds, using one for testing and the rest for training, repeated K times.  
**6\. Ensemble Methods** | Technique to achieve more accurate predictions by combining predictions from multiple models. Examples: Random Forest, Gradient Boosting.  
**7\. Validation** | Process of confirming whether predicted results match actual material properties through experiments or high-precision calculations. Critical step to ensure MI reliability.  
  
### Computational Methods Related (8-13)

Term (English) | Description  
---|---  
**8\. Density Functional Theory (DFT)** | Method to calculate electronic states of materials based on quantum mechanics. Can theoretically predict material properties (bandgap, formation energy, etc.).  
**9\. Active Learning** | Learning method where the model proposes "which data to acquire next." Can improve the model while minimizing experimental costs.  
**10\. Bayesian Optimization** | Method to search for optimal materials while minimizing the number of experiments. Uses Gaussian processes to determine next experimental candidates.  
**11\. Transfer Learning** | Technique to apply a model trained on one material system to a related different material system. Enables accurate predictions even for new material systems with limited data.  
**12\. Graph Neural Networks (GNN)** | Neural networks that treat crystal structures as graphs (atoms=nodes, bonds=edges) and directly learn structural information. Recently gaining attention.  
**13\. High-throughput Computation** | Method to automatically execute first-principles calculations for large numbers of materials. Materials Project has evaluated over 140,000 materials using high-throughput computation.  
  
### Materials Science Related (14-20)

Term (English) | Description  
---|---  
**14\. Crystal Structure** | Structure where atoms are arranged in a regular pattern. Types include FCC (face-centered cubic), BCC (body-centered cubic), HCP (hexagonal close-packed).  
**15\. Space Group** | 230 mathematical groups that classify crystal structure symmetry. Closely related to material properties.  
**16\. Bandgap** | Energy difference between the electron-occupied valence band and empty conduction band in semiconductors or insulators. Important for solar cells and semiconductor device design.  
**17\. Formation Energy** | Energy change when a material is formed from its constituent elements. Negative values indicate stable materials.  
**18\. Phase Diagram** | Diagram showing which phase (solid, liquid, gas) a material exists in as a function of temperature, pressure, and composition. Essential for alloy design.  
**19\. Multi-objective Optimization** | Method to simultaneously optimize multiple properties (e.g., lightweight and strength). Balances properties that usually have trade-off relationships.  
**20\. Pareto Front** | In multi-objective optimization, the set of solutions that are not optimal in all objectives but cannot improve any objective. Represents candidate group of optimal materials.  
  
**Key points for term learning:** First prioritize understanding 1-7 (data and model related). Learn 8-13 (computational methods) in detail at intermediate level. Review 14-20 (materials science) alongside materials science fundamentals.

* * *

## 2.3 Overview of Materials Databases

This section provides detailed comparison of four major materials databases that form the foundation of MI.

### 2.3.1 Detailed Comparison of Major Databases

Database Name | Number of Materials | Data Source | Main Property Data | License | Access Method | Advantages | Use Cases  
---|---|---|---|---|---|---|---  
**Materials Project** | 140,000+ | DFT calculations (VASP) | Bandgap, formation energy, elastic constants, phase stability | **CC BY 4.0**  
(Academic/non-profit only) | Web UI, API (Python: `pymatgen`), registration required (free) | Largest scale, active community, rich tools | Battery materials, semiconductors, structural materials  
**AFLOW** | 3,500,000+ | DFT calculations (VASP) | Crystal structure, electronic structure, thermodynamic stability | **CC BY 4.0** | Web UI, API (RESTful), no registration required | Most crystal structure data, standardized naming convention | Crystal structure exploration, new structure prediction  
**OQMD** | 1,000,000+ | DFT calculations (VASP) | Formation energy, stability, phase diagrams | **ODbL 1.0**  
(Citation required) | Web UI, API (Python: `qmpy`), no registration required | Strong in phase diagram calculations, rich alloy data | Alloy design, phase stability evaluation  
**JARVIS** | 70,000+ | DFT calculations (VASP)  
Machine Learning predictions | Optical properties, mechanical properties, topological properties | **NIST Public Data**  
(Public domain) | Web UI, API (Python: `jarvis-tools`), no registration required | Diverse properties, provides machine learning models | Optical materials, topological materials  
  
### 2.3.2 Database Selection Guide

**1\. Materials Project** \- **When to use** : Battery materials, semiconductors, general inorganic materials exploration \- **Strengths** : \- Intuitive Web UI, beginner-friendly \- Rich Python library (`pymatgen`) \- Active community with abundant information \- **Weaknesses** : Low coverage for some structure types

**2\. AFLOW** \- **When to use** : New crystal structure exploration, structural similarity search \- **Strengths** : \- Most crystal structures (3.5 million types) \- Standardized crystal structure description (AFLOW prototype) \- Fast structural similarity search \- **Weaknesses** : Fewer types of property data compared to Materials Project

**3\. OQMD** \- **When to use** : Alloy phase diagram calculations, detailed phase stability evaluation \- **Strengths** : \- Specialized in phase diagram calculations \- Rich data for multi-component alloys \- Temperature-dependent evaluation possible \- **Weaknesses** : Web UI usability somewhat inferior

**4\. JARVIS** \- **When to use** : Optical materials, topological materials, using machine learning models \- **Strengths** : \- Integrated machine learning models \- Rich optical properties (dielectric constant, refractive index) \- Topological property calculations \- **Weaknesses** : Fewer materials than others

### 2.3.3 Practical Examples of Database Utilization

**Scenario 1: Want to find new cathode materials for lithium-ion batteries**

  1. **Search in Materials Project** : \- Conditions: Contains Li, voltage 3.5-4.5V, stable \- Candidates: 100 types found

  2. **Select top 10** : \- Evaluate by balance of capacity, voltage, and stability

  3. **Confirm phase stability in OQMD** : \- Check possibility of decomposition with temperature changes

  4. **Experimental validation** : \- Actually synthesize top 3 materials

**Scenario 2: Want to find transparent conducting materials (for solar cells)**

  1. **Search in JARVIS** : \- Conditions: Bandgap > 3.0 eV (transparent), high electrical conductivity \- Candidates: 50 types

  2. **Additional information from Materials Project** : \- Check formation energy and thermal stability

  3. **Search for similar structures in AFLOW** : \- Find structures similar to promising materials found

  4. **Experimental validation**

### 2.3.4 Example Database Access

**Materials Project API (Python) Usage Example:**
    
    
    from pymatgen.ext.matproj import MPRester
    
    # Get API key: https://materialsproject.org
    with MPRester("YOUR_API_KEY") as mpr:
        # Get information on LiCoO2
        data = mpr.get_data("mp-1234")  # material_id
    
        print(f"Chemical formula: {data[0]['pretty_formula']}")
        print(f"Bandgap: {data[0]['band_gap']} eV")
        print(f"Formation energy: {data[0]['formation_energy_per_atom']} eV/atom")
    

**Important points:** Each database has a complementary relationship, so using multiple databases rather than just one is recommended. To ensure data reliability, it is important to compare results across different databases.

### 2.3.5 Data Licenses and Citation Methods

Proper license understanding and citation are essential when using materials databases. Inappropriate use can cause legal and ethical issues.

#### License Details

**Materials Project - CC BY 4.0 (Creative Commons Attribution 4.0)**

  * **Permitted uses:**
  * ✅ Use in academic research
  * ✅ Use in papers and presentations
  * ✅ Educational purposes
  * ✅ Use in non-profit projects

  * **Restrictions:**

  * ⚠️ **Commercial use requires separate permission**
  * ⚠️ Contact in advance when using for corporate product development
  * ⚠️ Must credit source when redistributing data

  * **Citation method:** `Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." APL Materials, 1(1), 011002. DOI: 10.1063/1.4812323`

  * **API key acquisition:** https://materialsproject.org → Account registration (free) → API Keys

**AFLOW - CC BY 4.0**

  * **Permitted uses:**
  * ✅ Free use for both academic and commercial purposes
  * ✅ Data modification and redistribution also possible (with source attribution)

  * **Citation method:** `Curtarolo, S., Setyawan, W., Hart, G. L., Jahnatek, M., et al. (2012). "AFLOW: An automatic framework for high-throughput materials discovery." Computational Materials Science, 58, 218-226. DOI: 10.1016/j.commatsci.2012.02.005`

  * **Access:** No registration required, API also freely available

**OQMD - Open Database License (ODbL) 1.0**

  * **Permitted uses:**
  * ✅ Use for both academic and commercial purposes
  * ✅ Data modification and redistribution also possible

  * **Required conditions:**

  * ✅ **Must include citation**
  * ✅ Derived data must be published under same license (ODbL)

  * **Citation method:** `Saal, J. E., Kirklin, S., Aykol, M., Meredig, B., & Wolverton, C. (2013). "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)." JOM, 65(11), 1501-1509. DOI: 10.1007/s11837-013-0755-4`

  * **Access:** No registration required

**JARVIS - NIST Public Data (Public domain equivalent)**

  * **Permitted uses:**
  * ✅ Completely free to use (U.S. government data)
  * ✅ Unlimited for both academic and commercial purposes
  * ✅ Citation recommended but not required

  * **Recommended citation method:** `Choudhary, K., Garrity, K. F., Reid, A. C. E., et al. (2020). "The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design." npj Computational Materials, 6(1), 173. DOI: 10.1038/s41524-020-00440-1`

  * **Access:** No registration required

#### Commercial Use Considerations

**⚠️ Important: Commercial Use of Materials Project**

Materials Project uses CC BY 4.0 license for **academic and non-profit purposes only**. The following cases are considered commercial use:

  * ❌ Corporate product development (material selection, property prediction, etc.)
  * ❌ Data use in commercial services
  * ❌ Use in paid consulting work

**For commercial use:** 1\. Contact Materials Project operations team (contact@materialsproject.org) 2\. Explain usage purpose and scope 3\. Obtain commercial license (conditions negotiable)

**AFLOW, OQMD, JARVIS allow commercial use:** \- However, proper citation is mandatory \- Pay attention to license when publishing derived data

#### Redistribution Permissions

Database | Original Data Redistribution | Processed Data Redistribution | Conditions  
---|---|---|---  
**Materials Project** | ⚠️ Requires confirmation | ⚠️ Requires confirmation | Source attribution, academic purposes only  
**AFLOW** | ✅ Possible | ✅ Possible | Source attribution required  
**OQMD** | ✅ Possible | ✅ Possible | Maintain ODbL license, source attribution  
**JARVIS** | ✅ Possible | ✅ Possible | No restrictions (citation recommended)  
  
#### Citation Examples in Python Code

It is recommended to include citations not only in papers and reports but also within code:
    
    
    """
    Materials Project API Example
    
    Data source: Materials Project (https://materialsproject.org)
    License: CC BY 4.0 (academic use only)
    Citation: Jain et al. (2013), APL Materials, 1(1), 011002.
              DOI: 10.1063/1.4812323
    
    Commercial use requires separate permission from Materials Project.
    """
    
    from pymatgen.ext.matproj import MPRester
    
    # Recommended to load API key from environment variable
    import os
    API_KEY = os.getenv("MP_API_KEY")  # Do not hardcode for security
    
    with MPRester(API_KEY) as mpr:
        # Data retrieval code
        pass
    

#### Rate Limits and Etiquette

Each database has rate limits to avoid server overload:

Database | Rate Limit | Recommendations  
---|---|---  
**Materials Project** | ~5 requests/sec | Insert sleep() during bulk retrieval  
**AFLOW** | No explicit limit | Use within reasonable range  
**OQMD** | No explicit limit | Avoid parallel requests  
**JARVIS** | No explicit limit | Batch download recommended  
  
**Recommended code example (handling rate limits):**
    
    
    import time
    from pymatgen.ext.matproj import MPRester
    
    with MPRester(API_KEY) as mpr:
        material_ids = ["mp-1234", "mp-5678", ...]  # Large number of IDs
    
        results = []
        for i, mat_id in enumerate(material_ids):
            data = mpr.get_data(mat_id)
            results.append(data)
    
            # Wait 1 second every 100 requests (rate limit consideration)
            if (i + 1) % 100 == 0:
                time.sleep(1)
                print(f"Retrieved: {i+1}/{len(material_ids)}")
    

#### Checklist: Verification Before Using Database

  * [ ] Is the usage purpose academic or commercial?
  * [ ] For commercial use, does the license permit it?
  * [ ] Have you confirmed the citation method?
  * [ ] Have you obtained API key if required?
  * [ ] Have you understood rate limits and written appropriate code?
  * [ ] If planning to redistribute data, are you meeting license conditions?

* * *

## 2.4 MI Ecosystem: Data Flow

MI is not a single technology but an ecosystem where multiple elements collaborate. The following diagram shows the data flow in MI.
    
    
    ```mermaid
    flowchart TB
        subgraph "Data Generation"
            A[Experimental Data] --> D[Materials Database]
            B[First-Principles Calculations\nDFT] --> D
            C[Papers & Patents] --> D
        end
    
        subgraph "Data Processing"
            D --> E[Data Cleaning\nStandardization]
            E --> F[Descriptor Generation\nFeature Engineering]
        end
    
        subgraph "Machine Learning"
            F --> G[Model Training\nRegression & Classification]
            G --> H[Prediction & Screening\nThousands-Tens of thousands of candidates]
        end
    
        subgraph "Experimental Validation"
            H --> I[Candidate Material Selection\nTop 10-100]
            I --> J[Experimental Synthesis & Measurement]
            J --> K{Prediction\nAccurate?}
        end
    
        subgraph "Continuous Improvement"
            K -->|Yes| L[Add as New Data]
            K -->|No| M[Model Improvement & Retraining]
            L --> D
            M --> G
        end
    
        style D fill:#e3f2fd
        style F fill:#fff3e0
        style G fill:#f3e5f5
        style J fill:#e8f5e9
        style L fill:#fce4ec
    ```

**How to read the diagram:**

**1\. Data Generation** — Collect data from experiments, computations, and literature.

**2\. Data Processing** — Convert raw data into machine learning-suitable format.

**3\. Machine Learning** — Train models and predict large numbers of candidates.

**4\. Experimental Validation** — Experimentally verify promising candidates.

**5\. Continuous Improvement** — Add results to data and improve models.

**Importance of the feedback loop:** Accurate predictions lead to adding data to further improve the model. Inaccurate predictions require reviewing the model and changing descriptors or learning methods. By repeating this cycle, model accuracy improves.

* * *

## 2.5 MI Basic Workflow: Detailed Version

Chapter 1 introduced a 4-step workflow, but here we expand it to a more practical **5-step** workflow.

### 2.5.1 Overall Picture
    
    
    ```mermaid
    flowchart LR
        A[Step 0:\nProblem Formulation] --> B[Step 1:\nData Collection]
        B --> C[Step 2:\nModel Construction]
        C --> D[Step 3:\nPrediction & Screening]
        D --> E[Step 4:\nExperimental Validation]
        E --> F[Step 5:\nData Addition & Improvement]
        F - Continuous Improvement .-> B
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#fce4ec
    ```

### 2.5.2 Step 0: Problem Formulation (Most Important, Often Overlooked)

**What to do:** Clearly define the problem to solve, specify target properties and constraints, and set success criteria.

**Concrete example: Battery material development**

**Poor problem formulation:**

> "Want to find good battery materials"

**Good problem formulation:**

> "Discover lithium-ion battery cathode materials with the following properties: \- Theoretical capacity: ≥200 mAh/g \- Operating voltage: 3.5-4.5 V vs. Li/Li+ \- Cycle life: capacity retention ≥80% after 500 cycles \- Cost: ≤$50/kg (raw material basis) \- Safety: thermal runaway temperature ≥200°C \- Environmental constraints: minimize Co usage (ideally Co-free)"

**Problem formulation checklist:** (1) Are target properties quantitatively defined? (2) Are constraints (cost, environmental, safety) clear? (3) Are success criteria measurable? (4) Is the scope experimentally verifiable?

**Time estimate:** 1-2 weeks (including literature review and expert discussions)

**Common failures:** Vague goals changed multiple times later, ignoring constraints and searching for unrealizable materials, and no success criteria leading to endless search.

### 2.5.3 Step 1: Data Collection

**What to do:** Collect material information from existing experimental data and literature, download relevant data from materials databases, and add data through first-principles calculations if necessary.

**Data source priority:** 1\. **Existing databases** (Most efficient) \- Materials Project, AFLOW, OQMD \- High reliability, ready to use

  2. **Papers & patents** (Manual work required) \- Google Scholar, Web of Science \- May contain experimental data

  3. **Self-calculation/measurement** (Time-consuming) \- Generate data for new materials by DFT calculations \- Laboratory measurements

**Concrete example: Lithium-ion battery cathode materials**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Data collection for lithium-ion battery cathode materials
    
    Dependencies (libraries and versions):
    - Python: 3.9+
    - pymatgen: 2023.10.11 or later
    - pandas: 2.0+
    - numpy: 1.24+
    
    Environment (execution environment):
    - API key: Materials Project (https://materialsproject.org)
    - Rate limit: ~5 requests/sec
    """
    
    from pymatgen.ext.matproj import MPRester
    import pandas as pd
    import os
    
    # Load API key from environment variable (improved security)
    API_KEY = os.getenv("MP_API_KEY", "YOUR_API_KEY")
    
    # Search Li-containing oxides from Materials Project
    with MPRester(API_KEY) as mpr:
        # Search criteria
        criteria = {
            "elements": {"$all": ["Li", "O"]},  # Must contain Li and O
            "nelements": {"$gte": 2, "$lte": 4},  # 2-4 elements
            "e_above_hull": {"$lte": 0.05}  # Stable or metastable (within 50 meV/atom)
        }
    
        # Properties to retrieve
        properties = [
            "material_id",
            "pretty_formula",
            "formation_energy_per_atom",  # eV/atom
            "energy_above_hull",  # eV/atom (thermodynamic stability)
            "band_gap",  # eV (electronic structure)
            "density"  # g/cm³
        ]
    
        # Data retrieval
        results = mpr.query(criteria, properties)
    
        # Convert to DataFrame
        df = pd.DataFrame(results)
    
        print(f"Number of materials retrieved: {len(df)}")
        print(f"Data shape: {df.shape}")
        print(df.head())
    
    # Data validation (check NaN, outliers)
    print(f"\nNumber of missing values:\n{df.isnull().sum()}")
    print(f"\nBandgap range: {df['band_gap'].min():.2f} - {df['band_gap'].max():.2f} eV")
    

**Expected results:** Hundreds to thousands of candidate material data, with basic properties for each material (composition, formation energy, bandgap, etc.).

**Time estimate:** Database utilization takes several hours to days. Literature survey requires 1-2 weeks. DFT calculations take several weeks to months depending on the number of materials.

**Common issues:** Data missing (specific properties available only for certain materials), data discrepancies (values differ across databases), and data bias (biased toward specific material systems).

**Solutions:** Compare multiple databases to verify reliability. Consider missing value imputation methods (mean, machine learning estimation, etc.). Recognize data bias and clarify model applicability scope.

### 2.5.4 Step 2: Model Construction

**What to do:** Train machine learning models using collected data, select appropriate descriptors (features), and evaluate and optimize model performance.

**Sub-steps:**

**2.1 Descriptor Design**

Need to convert materials into numerical vectors.

**Types of descriptors:**

Type | Concrete Examples | Advantages | Disadvantages  
---|---|---|---  
**Composition-based** | Electronegativity, atomic radius, atomic weight | Simple calculation, easy to interpret | Ignores structural information  
**Structure-based** | Lattice constants, space group, coordination number | Captures structure-property relationships | Requires crystal structure data  
**Property-based** | Melting point, density, bandgap | Uses correlations between properties | Difficult to apply to unknown materials  
  
**Descriptor example: Numerical representation of LiCoO2**
    
    
    # Simple example: composition-based descriptor
    material = "LiCoO2"
    
    # Fraction of each element
    Li_fraction = 0.25  # 1/(1+1+2)
    Co_fraction = 0.25
    O_fraction = 0.50
    
    # Element properties (from periodic table)
    electronegativity_Li = 0.98
    electronegativity_Co = 1.88
    electronegativity_O = 3.44
    
    # Weighted average
    avg_electronegativity = (
        Li_fraction * electronegativity_Li +
        Co_fraction * electronegativity_Co +
        O_fraction * electronegativity_O
    )  # = 2.38
    
    # Vector representation
    descriptor_vector = [
        Li_fraction, Co_fraction, O_fraction,  # Composition
        avg_electronegativity,  # Electronegativity
        # ... add other properties
    ]
    

**In actual projects, use the`matminer` library:**
    
    
    from matminer.featurizers.composition import ElementProperty
    
    # Automatically generate many descriptors
    featurizer = ElementProperty.from_preset("magpie")
    features = featurizer.featurize_dataframe(df, col_id="composition")
    

**2.2 Model Selection**

**Beginner-level models:** **Linear Regression** is simple and easy to interpret. **Decision Trees** are visualizable and capture nonlinear relationships.

**Intermediate-level models:** **Random Forest** offers high accuracy and is resistant to overfitting. **Gradient Boosting (XGBoost, LightGBM)** achieves highest accuracy.

**Advanced-level models:** **Neural Networks** learn complex nonlinear relationships. **Graph Neural Networks (GNN)** directly learn crystal structures.

**2.3 Training and Evaluation**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Machine learning model training and evaluation
    
    Dependencies (libraries and versions):
    - Python: 3.9+
    - scikit-learn: 1.3+
    - numpy: 1.24+
    - pandas: 2.0+
    
    Reproducibility:
    - Random seed fixed: 42 (unified across all random operations)
    - Train/test split: 80/20
    - Cross-validation: 5-fold
    """
    
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    
    # Fix random seed (ensure reproducibility)
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Data split
    X = features  # Descriptors (e.g., 73-dimensional vector)
    y = df['target_property']  # e.g., voltage (V), bandgap (eV), etc.
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target variable range: {y.min():.2f} - {y.max():.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
    
    print(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")
    
    # Model training
    model = RandomForestRegressor(
        n_estimators=100,  # Number of decision trees
        max_depth=20,  # Maximum depth (prevent overfitting)
        min_samples_split=5,  # Minimum samples required for split
        random_state=RANDOM_SEED,  # Ensure reproducibility
        n_jobs=-1  # Use all CPU cores (speed up)
    )
    
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Coefficient of Determination
    
    print("\n===== Test Set Performance =====")
    print(f"MAE (Mean Absolute Error): {mae:.3f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"R² (Coefficient of Determination): {r2:.3f}")
    
    # Cross-validation (more reliable evaluation)
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    print(f"\n===== Cross-Validation Performance (5-fold) =====")
    print(f"CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Data leakage verification (compare training error vs test error)
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"\nTraining MAE: {train_mae:.3f}, Test MAE: {mae:.3f}")
    if train_mae < mae * 0.5:
        print("⚠️ Warning: Possible overfitting (training error significantly lower)")
    

**Performance guidelines:** **R squared > 0.8** is good, **R squared > 0.9** is excellent, and **R squared < 0.5** indicates the model needs review.

**Time estimate:** Descriptor design takes several days to 1 week. Model training and optimization requires 1-2 weeks.

**Common issues:** Overfitting (high accuracy on training data but low on test data) and descriptor selection mistakes (overlooking important features).

**Solutions:** Verify generalization performance with cross-validation, analyze feature importance and remove unnecessary descriptors, and introduce regularization (L1/L2).

### 2.5.5 Step 3: Prediction & Screening

**What to do:** Use the trained model to predict properties of unknown materials, evaluate large numbers of candidate materials (thousands to tens of thousands) in a short time, and select promising top candidates.

**Screening workflow:**
    
    
    Candidate materials: 10,000 types (generated by calculations)
      ↓ (Predict with machine learning: minutes)
    Rank by predicted values
      ↓
    Select top 1,000 (close to target properties)
      ↓ (Detailed calculation/evaluation: hours to days)
    Narrow down to top 100
      ↓
    Materials to experiment: Top 10 (most promising candidates)
    

**Concrete code example:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Concrete code example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Generate candidate material list (e.g., create candidates by varying composition)
    # Actually use more systematic methods
    candidate_compositions = [...]  # 10,000 candidates
    
    # Calculate descriptors for each candidate
    candidate_features = compute_descriptors(candidate_compositions)
    
    # Predict with model
    predicted_properties = model.predict(candidate_features)
    
    # Rank (e.g., by high voltage)
    ranked_indices = np.argsort(predicted_properties)[::-1]
    
    # Select top 100
    top_100 = [candidate_compositions[i] for i in ranked_indices[:100]]
    
    print("Top 10 candidates:")
    for i, comp in enumerate(top_100[:10]):
        pred_val = predicted_properties[ranked_indices[i]]
        print(f"{i+1}. {comp}: Predicted value = {pred_val:.2f}")
    

**Efficiency example:** **Conventional** methods experimentally evaluate 10,000 types taking approximately 30 years (1 per day). **MI** experimentally evaluates only 10 types in approximately 2 weeks, achieving a **time reduction rate** of 99.9%.

**Time estimate:** Prediction calculation takes minutes to hours depending on the number of candidate materials. Result analysis requires several days.

**Important notes:** Predictions are predictions, so always experimentally verify. For materials outside model applicability scope (significantly different from training data), prediction accuracy is low. Uncertainty evaluation (Bayesian methods) provides more reliability.

### 2.5.6 Step 4: Experimental Validation

**What to do:** Actually synthesize materials narrowed down by predictions, measure properties and verify prediction accuracy, and analyze discrepancies between predictions and measurements.

**Experiment priority:** 1\. **Materials with highest predicted values** (best case) 2\. **Materials with moderate prediction but low uncertainty** (safe choice) 3\. **Materials with high prediction but also high uncertainty** (high risk, high return)

**Validation checklist:** (1) Are synthesis conditions established? (2) Are measurement instruments available? (3) Does measurement accuracy meet target property requirements? (4) Reproducibility confirmation (multiple measurements).

**Time estimate:** Synthesis takes several days to weeks depending on the material. Measurement requires several days to 1 week. Total time for top 10 materials is 2-3 months.

**Success and failure assessment:**

Result | Assessment | Next Action  
---|---|---  
Prediction matches measurement | Success | Add to data, continue exploration  
Better than prediction | Great success | Analyze model and investigate why it underestimated  
Worse than prediction | Partial failure | Review descriptors and model  
Completely different | Failure | Possible out of model scope. Reconsider data and model  
  
**Important points:** Failures are also valuable data and should always be added to the database. By analyzing causes of prediction-measurement discrepancies, models improve.

### 2.5.7 Step 5: Data Addition & Model Improvement

**What to do:** Add experimental results (both successes and failures) to the database, retrain the model with new data, and verify prediction accuracy improvement.

**Continuous improvement cycle:**
    
    
    Initial model (R² = 0.75)
      ↓
    Add 10 experimental results
      ↓
    Model retraining (R² = 0.82)
      ↓
    10 more experiments
      ↓
    Model retraining (R² = 0.88)
      ↓
    Finally discover optimal material
    

**Utilizing active learning:**

In normal MI, materials with high predicted values are experimented on, but in **active learning** , the model proposes "materials with high uncertainty."
    
    
    # Estimate uncertainty with Random Forest
    predictions = []
    for tree in model.estimators_:
        pred = tree.predict(candidate_features)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    uncertainty = predictions.std(axis=0)  # Large standard deviation = high uncertainty
    
    # Prioritize materials with high uncertainty for experiments
    high_uncertainty_indices = np.argsort(uncertainty)[::-1]
    next_experiment = candidate_compositions[high_uncertainty_indices[0]]
    

**Time estimate:** 1-2 weeks per cycle

**Termination conditions:** Materials meeting target properties are found, prediction accuracy is sufficiently high (R squared > 0.9), or budget/time constraints are reached.

* * *

## 2.6 Material Descriptor Details

### 2.6.1 Types of Descriptors and Concrete Examples

**1\. Composition-based Descriptors**

**Features:** Calculable from chemical formula alone, usable even when crystal structure is unknown, with low computational cost.

**Concrete examples:**

Descriptor | Description | Example (LiCoO2)  
---|---|---  
Average electronegativity | Weighted average of each element's electronegativity | 2.38  
Average atomic radius | Weighted average of each element's atomic radius | 1.15 Å  
Number of element types | Number of constituent elements | 3 (Li, Co, O)  
Average atomic weight | Weighted average of each element's atomic weight | 30.8 g/mol  
Electronegativity difference | Difference between maximum and minimum electronegativity | 2.46 (O - Li)  
  
**2\. Structure-based Descriptors**

**Features:** Utilizes crystal structure information, captures structure-property relationships, but requires crystal structure data.

**Concrete examples:**

Descriptor | Description | Example (LiCoO2)  
---|---|---  
Lattice constants | Unit cell lengths a, b, c | a=2.82 Å, c=14.05 Å (hexagonal)  
Space group | Crystal symmetry | R-3m (166)  
Coordination number | Number of nearest neighbor atoms around an atom | Co: 6-coordinate (surrounded by oxygen)  
Bond distance | Distance between adjacent atoms | Co-O: 1.93 Å  
Density | Mass per unit volume | 5.06 g/cm³  
  
**3\. Property-based Descriptors**

**Features:** Predicts unknown properties from known properties and utilizes correlations between properties, though it is difficult to apply to unknown materials.

**Concrete examples:**

Descriptor | Description | Example (LiCoO2)  
---|---|---  
Melting point | Solid-to-liquid phase transition temperature | ~1200 K  
Bandgap | Electronic structure energy gap | ~2.7 eV (insulator)  
Formation energy | Energy when formed from elements | -2.5 eV/atom (stable)  
Elastic modulus | Material hardness/resistance to deformation | 150 GPa  
Thermal conductivity | Ease of heat transfer | 5 W/(m·K)  
  
### 2.6.2 Automatic Descriptor Generation (Utilizing Matminer)
    
    
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    from matminer.featurizers.structure import DensityFeatures
    from pymatgen.core import Composition
    
    # Automatic generation of composition-based descriptors
    comp = Composition("LiCoO2")
    
    # Example 1: Element property-based descriptors (73 types)
    element_featurizer = ElementProperty.from_preset("magpie")
    element_features = element_featurizer.featurize(comp)
    
    print(f"Number of generated descriptors: {len(element_features)}")
    print(f"Example descriptors: {element_features[:5]}")
    
    # Example 2: Stoichiometry-based descriptors
    stoich_featurizer = Stoichiometry()
    stoich_features = stoich_featurizer.featurize(comp)
    
    print(f"Stoichiometry descriptors: {stoich_features}")
    

**Descriptors generated by Matminer (partial list):** Average electronegativity, atomic radius, and atomic weight; element position on periodic table (group, period); electronic configuration (number of s-orbital electrons, p-orbital electrons, etc.); average and variance of oxidation states; and number of valence electrons for elements.

### 2.6.3 Descriptor Selection and Feature Engineering

**Not all descriptors are useful:** Irrelevant descriptors become noise and degrade model performance. Redundant descriptors waste computational cost.

**Descriptor selection methods:**

**1\. Feature Importance**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 1. Feature Importance
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Random Forest feature importance
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Visualize top 20
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), importances[indices[:20]])
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance Top 20")
    plt.tight_layout()
    plt.show()
    

**2\. Correlation Analysis**
    
    
    # Correlation matrix between features
    correlation_matrix = X.corr()
    
    # Remove feature pairs with high correlation (>0.9)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((correlation_matrix.columns[i],
                                       correlation_matrix.columns[j]))
    
    print(f"High correlation pairs: {len(high_corr_pairs)} pairs")
    

**3\. Recursive Feature Elimination (RFE)**
    
    
    from sklearn.feature_selection import RFE
    
    # Select best 50 features
    selector = RFE(model, n_features_to_select=50, step=1)
    selector.fit(X_train, y_train)
    
    selected_features = X.columns[selector.support_]
    print(f"Selected features: {list(selected_features)}")
    

* * *

## 2.8 Practical Pitfalls: Common Failures in MI Projects

To succeed in MI projects, avoiding practical pitfalls is as important as technical skills. This section introduces six pitfalls beginners commonly fall into and countermeasures.

### 2.8.1 Data Leakage

**Problem:**

Data leakage is a phenomenon where test data information leaks into the training process, leading to overestimation of model performance.

**Common cases:**

**Case 1: Preprocessing errors**
    
    
    # ❌ Wrong: Standardize all data then split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Test data info leaks!
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    # ✅ Correct: Split first, then standardize only on training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn only on training data
    X_test_scaled = scaler.transform(X_test)  # Only transform test data
    

**Case 2: Handling time series data**
    
    
    # ❌ Wrong: Random split (train on future data)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # ✅ Correct: Split by time order
    split_point = int(len(X) * 0.8)
    X_train = X[:split_point]  # Train on old data
    X_test = X[split_point:]  # Test on new data
    

**Countermeasures:** Preprocessing must learn only on training data and apply only to test data. For time series data, split by time order. Verify data leakage with cross-validation (GroupKFold, TimeSeriesSplit).

### 2.8.2 Duplicate Structures

**Problem:**

Materials databases sometimes have the same material registered multiple times with different IDs. This causes the same material to appear in both training and test data, overestimating performance.

**Detection method:**
    
    
    from pymatgen.analysis.structure_matcher import StructureMatcher
    
    # Determine structural similarity
    matcher = StructureMatcher()
    
    # Detect duplicates
    duplicates = []
    for i in range(len(structures)):
        for j in range(i+1, len(structures)):
            if matcher.fit(structures[i], structures[j]):
                duplicates.append((i, j))
                print(f"Duplicate found: {material_ids[i]} and {material_ids[j]}")
    
    print(f"Number of duplicates: {len(duplicates)}")
    

**Countermeasures:** Remove duplicates when constructing the dataset. Identify similar structures with structure matching. Cross-check across different databases.

### 2.8.3 Polymorphs with Same Chemical Formula

**Problem:**

Even with the same chemical formula, different crystal structures result in vastly different properties (e.g., diamond and graphite are both C). Using only composition-based descriptors cannot capture this difference.

**Concrete example: TiO2 polymorphs**

Polymorph | Space Group | Bandgap | Use  
---|---|---|---  
**Anatase** | I4₁/amd | 3.2 eV | Photocatalyst  
**Rutile** | P4₂/mnm | 3.0 eV | Pigment  
**Brookite** | Pbca | 3.4 eV | Research  
  
**Countermeasure:**
    
    
    from matminer.featurizers.structure import SiteStatsFingerprint
    
    # Add structure-based descriptors
    structure_featurizer = SiteStatsFingerprint.from_preset("LocalPropertyDifference")
    structure_features = structure_featurizer.featurize_dataframe(df, col_id="structure")
    
    # Use both composition and structure
    X = pd.concat([composition_features, structure_features], axis=1)
    

**Countermeasures:** Combine composition-based and structure-based descriptors. Check `energy_above_hull` in the database (select only stable phases). Ensure proper phase identification in experiments (XRD, etc.).

### 2.8.4 Unit Cell Normalization

**Problem:**

Crystal structures have two representations: conventional cell and primitive cell, which have different numbers of atoms even for the same material.

**Example: FCC metal case** — Conventional cell contains 4 atoms while primitive cell contains 1 atom.

**Countermeasure:**
    
    
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # Unify to primitive cell
    analyzer = SpacegroupAnalyzer(structure)
    primitive_structure = analyzer.get_primitive_standard_structure()
    
    # Or normalize by number of atoms
    formation_energy_per_atom = formation_energy / structure.num_sites
    

**Countermeasures:** Unify all structures to primitive cell. Always normalize energy per atom (eV/atom). Use descriptors that do not depend on number of atoms.

### 2.8.5 Handling Missing Values

**Problem:**

In materials databases, not all materials have all property data. Inappropriate missing value handling significantly degrades prediction accuracy.

**Check missing values:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Check missing values:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Visualize missing values
    missing = df.isnull().sum() / len(df) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.ylabel('Missing rate (%)')
    plt.title('Missing rate by property')
    plt.show()
    
    print(f"Properties with missing values: {len(missing)}/{len(df.columns)}")
    

**Choose countermeasure:**

Missing Rate | Recommended Countermeasure | Reason  
---|---|---  
< 5% | Mean/median imputation | Small impact  
5-20% | KNN imputation, iterative imputation | Use correlations with other properties  
20-50% | Remove that property | Low imputation reliability  
> 50% | Remove data | Unusable  
  
**Imputation example:**
    
    
    from sklearn.impute import KNNImputer
    
    # KNN imputation (use values from similar materials)
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_train)
    
    # Or use only properties without missing values
    df_clean = df.dropna(subset=['band_gap', 'formation_energy'])
    

**Countermeasures:** Always check missing rate. Analyze missing pattern (random or systematic). Validate validity of imputation method.

### 2.8.6 Outlier Impact

**Problem:**

Calculation errors, measurement mistakes, or extreme materials exist as outliers and distort model learning.

**Detection method:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Detection method:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Visualize outliers with box plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.boxplot(df['formation_energy_per_atom'])
    plt.title('Formation Energy (eV/atom)')
    
    plt.subplot(1, 3, 2)
    plt.boxplot(df['band_gap'])
    plt.title('Band Gap (eV)')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(df['density'])
    plt.title('Density (g/cm³)')
    
    plt.tight_layout()
    plt.show()
    
    # Detect outliers by IQR method
    Q1 = df['formation_energy_per_atom'].quantile(0.25)
    Q3 = df['formation_energy_per_atom'].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df['formation_energy_per_atom'] < Q1 - 1.5*IQR) |
                  (df['formation_energy_per_atom'] > Q3 + 1.5*IQR)]
    
    print(f"Number of outliers: {len(outliers)}/{len(df)} ({len(outliers)/len(df)*100:.1f}%)")
    print(outliers[['material_id', 'pretty_formula', 'formation_energy_per_atom']])
    

**Countermeasures:**

  1. **Identify cause:** \- Calculation error → Remove \- Measurement mistake → Remove \- Real extreme material → Retain (but evaluate impact)

  2. **Robust models:** ```python from sklearn.ensemble import RandomForestRegressor from sklearn.linear_model import HuberRegressor # Robust to outliers

# Random Forest is relatively robust to outliers model_rf = RandomForestRegressor()

# Huber regression (robust to outliers) model_huber = HuberRegressor(epsilon=1.5) ```

  3. **Transformation:** `python # Log transformation (normalize skewed distribution) df['log_formation_energy'] = np.log1p(np.abs(df['formation_energy_per_atom']))`

**Countermeasures:** Always check outliers. Check physical validity. Compare performance before and after outlier removal.

### 2.8.7 Practical Checklist

Items to check before starting MI projects:

**Data quality:** \- [ ] Confirmed proportion and distribution of missing values? \- [ ] Removed duplicate structures? \- [ ] Detected and addressed outliers? \- [ ] Eliminated possibility of data leakage?

**Model construction:** \- [ ] Did preprocessing learn only on training data? \- [ ] Performed appropriate data split? (time series, group, random) \- [ ] Evaluated generalization performance with cross-validation? \- [ ] Checked signs of overfitting?

**Result interpretation:** \- [ ] Confirmed physical validity of prediction results? \- [ ] Understood model applicability scope? \- [ ] Evaluated uncertainty?

**Reproducibility:** \- [ ] Fixed random seed? \- [ ] Recorded dependency library versions? \- [ ] Documented data preprocessing steps?

* * *

## 2.9 End-of-Chapter Checklist: Quality Assurance

Verify that you understand and can practice the content of this chapter.

### Conceptual Understanding

  * [ ] Can explain MI definition to others
  * [ ] Can explain differences from computational materials science and cheminformatics
  * [ ] Can show difference between forward and inverse design with concrete examples
  * [ ] Can correctly use at least 15 out of 20 MI terminology

### Database Utilization (Application)

  * [ ] Understand Materials Project license and commercial use restrictions
  * [ ] Can differentiate uses of 4 databases (MP, AFLOW, OQMD, JARVIS)
  * [ ] Obtained API key and can retrieve data with Python
  * [ ] Know data citation methods and can include in code

### Workflow Practice (Application)

  * [ ] Can set quantitative goals in problem formulation
  * [ ] Can check missing values and duplicates during data collection
  * [ ] Can select appropriate descriptors (composition/structure/property)
  * [ ] Can perform correct data split avoiding data leakage
  * [ ] Can evaluate generalization performance with cross-validation

### Code Quality (Quality Assurance)

  * [ ] All code includes dependency library versions
  * [ ] Fixed random seed to ensure reproducibility
  * [ ] Performed data validation (shape, dtype, NaN, range)
  * [ ] Can write API access code handling rate limits

### Practical Skills (Real-world Application)

  * [ ] Can recognize and avoid 5 data leakage patterns
  * [ ] Can detect and remove duplicate structures
  * [ ] Can select appropriate missing value handling methods
  * [ ] Can detect outliers and assess validity

### Next Steps

**If achievement rate <80%:** \- Re-read this chapter, focusing on poorly understood parts \- Solve exercises again \- Reinforce foundational knowledge with prerequisites.md

**If achievement rate 80-95%:** \- Ready to proceed to Chapter 3 (Practical section) \- Deepen understanding while coding in Chapter 3

**If achievement rate ≥95%:** \- Proceed to Chapter 3 and consolidate knowledge through actual code implementation \- If possible, start a simple MI project

* * *

## 2.7 Summary

### What You Learned in This Chapter

  1. **MI Definition and Position** \- Integration of materials science and data science \- Efficiency through inverse design approach \- Differences from computational materials science and cheminformatics

  2. **20 MI Terms** \- Data & model related (descriptor, feature engineering, overfitting, etc.) \- Computational methods related (DFT, Bayesian Optimization, GNN, etc.) \- Materials science related (crystal structure, bandgap, phase diagram, etc.)

  3. **Materials Databases** \- Materials Project: Largest scale, beginner-friendly \- AFLOW: Most crystal structure data \- OQMD: Strong in phase diagram calculations \- JARVIS: Machine learning model integration \- Using multiple databases concurrently is recommended

  4. **MI Ecosystem** \- Cycle of data generation → processing → machine learning → experimental validation → improvement \- Feedback loop is important

  5. **5-Step MI Workflow** \- Step 0: Problem formulation (most important) \- Step 1: Data collection (databases, papers, calculations) \- Step 2: Model construction (descriptor design, training, evaluation) \- Step 3: Prediction & screening (efficiently evaluate large candidates) \- Step 4: Experimental validation (synthesis & measurement of top candidates) \- Step 5: Data addition & improvement (continuous improvement cycle)

  6. **Material Descriptor Details** \- Three types: composition-based, structure-based, property-based \- Automatic generation with Matminer \- Importance of feature selection

### To the Next Chapter

In Chapter 3, we put this knowledge into practice. Using actual Python code, you will experience the entire workflow from data retrieval from materials databases, descriptor generation, machine learning model construction, to predictions.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Select 5 terms from the MI glossary and explain them in your own words.

Sample Answer **1. Descriptor** Numerical representation of material features that can be input to machine learning models. Examples include element electronegativity and atomic radius. **2. Screening** Efficiently narrowing down materials with desired properties from a large pool of candidates. MI allows computational evaluation of thousands to tens of thousands of materials in a short time. **3. Overfitting** Phenomenon where a machine learning model memorizes training data and prediction performance on new data degrades. Can be detected by cross-validation. **4. Bandgap** In semiconductors, the energy difference between the electron-occupied valence band and empty conduction band. Important metric for solar cell design. **5. Bayesian Optimization** Method to search for optimal materials while minimizing the number of experiments. AI proposes which materials to experiment on next. 

### Problem 2 (Difficulty: medium)

Regarding the use cases of Materials Project and AFLOW, answer which one to use for the following scenarios with reasons.

**Scenario A** : Want to explore new lithium-ion battery cathode materials. Need bandgap and formation energy data.

**Scenario B** : Want to find new materials with crystal structures similar to existing materials. Need structural similarity search.

Sample Answer **Scenario A: Use Materials Project** **Reasons:** \- Materials Project has over 140,000 materials data with both bandgap and formation energy available \- Rich tools specialized for battery material research (voltage, capacity calculations, etc.) \- Well-developed Python library (pymatgen), easy data retrieval \- Intuitive Web UI, easy for beginners to use **Scenario B: Use AFLOW** **Reasons:** \- AFLOW has the most crystal structure data (3.5 million types) \- Fast and accurate structural similarity search functionality \- Standardized structure description through AFLOW prototypes makes similar structure exploration easy \- Rich tools specialized for structure exploration **Summary:** Materials Project is suitable when emphasizing property data, AFLOW when emphasizing structure exploration. In actual projects, both are often used together. 

### Problem 3 (Difficulty: medium)

Explain why Step 0 (problem formulation) of the MI workflow is most important, with concrete examples.

Hint Vague problem formulation affects all subsequent steps. Consider the importance of clarifying target properties, constraints, and success criteria.  Sample Answer **Importance of problem formulation:** Insufficient problem formulation causes the following issues. **Bad example:** > "Want to find high-performance catalyst materials" **Problems:** \- "High-performance" not defined (reaction rate? Selectivity? Durability?) \- No constraints (cost, toxicity, availability) \- Unclear success criteria (when to stop exploration?) **Consequences:** 1\. Waste time in data collection (collect irrelevant data) 2\. Model optimizes wrong objectives 3\. Realize at experiment stage "actually different properties were important," need to start over **Good example:** > "Discover catalyst materials for hydrogen production with the following properties: > \- Reaction rate: ≥100 mol H2/(m²·h) > \- Selectivity: ≥95% (suppress by-products other than hydrogen) > \- Durability: activity ≥80% after 1000 hours continuous operation > \- Cost: ≤$100/kg > \- Constraint: Minimize precious metal use (Pt, Pd, etc.)" **Effects:** 1\. Clear data collection (prioritize reaction rate, selectivity, durability data) 2\. Model optimizes correct objectives 3\. Clear success criteria, easy to evaluate project progress 4\. Easy to decide experiment priorities **Value of time investment:** Spending 1-2 weeks on problem formulation significantly reduces risk of wasting months to years of subsequent work. 

### Problem 4 (Difficulty: hard)

There are three types of material descriptors: composition-based, structure-based, and property-based. List advantages and disadvantages of each, and explain in what situations they should be used.

Hint Consider the trade-offs between computational cost, required data, and prediction accuracy for each descriptor.  Sample Answer **Composition-based Descriptors** **Advantages:** \- Calculable from chemical formula alone (no crystal structure needed) \- Low computational cost (seconds) \- Applicable to unknown materials **Disadvantages:** \- Ignores structural information (same composition but different structure → different properties) \- Prediction accuracy may be lower than structure-based **Use when:** \- Many materials with unknown crystal structures \- Need high-speed screening (tens of thousands) \- Project initial stages (rough screening) **Structure-based Descriptors** **Advantages:** \- Captures structure-property relationships (more accurate predictions) \- Can distinguish different structures with same composition **Disadvantages:** \- Requires crystal structure data (determined experimentally or by DFT calculations) \- High computational cost \- For unknown materials, need to predict structure **Use when:** \- Crystal structure data available \- Need high-accuracy predictions (narrowing down final candidates) \- Want to understand structure-property correlations **Property-based Descriptors** **Advantages:** \- Uses correlations between properties (e.g., high melting point materials tend to be hard) \- High accuracy for known materials **Disadvantages:** \- Difficult to apply to unknown materials (need other properties to predict target property) \- Unclear causal relationships (why those properties are related) **Use when:** \- Inferring one property from another for known materials \- Material systems with abundant experimental data \- Exploring correlations between properties **Practical differentiation strategy:** 1\. **Initial screening (tens of thousands)**: Composition-based descriptors \- Quickly narrow down to ~1,000 candidates 2\. **Intermediate screening (1,000)**: Structure-based descriptors \- More accurately narrow down to ~100 3\. **Final selection (100)**: Property-based descriptors (if possible) \- Use known properties to determine final 10 4\. **Experimental validation (10)** This staged approach balances computational cost and prediction accuracy. 

* * *

## References

  1. **Materials Genome Initiative (MGI)** \- White House Office of Science and Technology Policy (2011) URL: https://www.mgi.gov _Materials development acceleration project launched by the U.S. in 2011. Became the catalyst for global MI proliferation._

  2. Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., & Kim, C. (2017). "Machine learning in materials informatics: recent applications and prospects." _npj Computational Materials_ , 3(1), 54. DOI: [10.1038/s41524-017-0056-5](<https://doi.org/10.1038/s41524-017-0056-5>)

  3. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." _APL Materials_ , 1(1), 011002. DOI: [10.1063/1.4812323](<https://doi.org/10.1063/1.4812323>) Materials Project: https://materialsproject.org

  4. Curtarolo, S., Setyawan, W., Hart, G. L., Jahnatek, M., Chepulskii, R. V., et al. (2012). "AFLOW: An automatic framework for high-throughput materials discovery." _Computational Materials Science_ , 58, 218-226. DOI: [10.1016/j.commatsci.2012.02.005](<https://doi.org/10.1016/j.commatsci.2012.02.005>) AFLOW: http://www.aflowlib.org

  5. Saal, J. E., Kirklin, S., Aykol, M., Meredig, B., & Wolverton, C. (2013). "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)." _JOM_ , 65(11), 1501-1509. DOI: [10.1007/s11837-013-0755-4](<https://doi.org/10.1007/s11837-013-0755-4>) OQMD: http://oqmd.org

  6. Choudhary, K., Garrity, K. F., Reid, A. C. E., DeCost, B., Biacchi, A. J., et al. (2020). "The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design." _npj Computational Materials_ , 6(1), 173. DOI: [10.1038/s41524-020-00440-1](<https://doi.org/10.1038/s41524-020-00440-1>) JARVIS: https://jarvis.nist.gov

* * *

**Author Information**

This article was created as part of the MI Knowledge Hub project under Dr. Yusuke Hashimoto at Tohoku University.

**Update History** \- 2025-10-16: v3.0 Initial version \- Expanded Section 2 from v2.1 (~2,000 words) to 4,000-5,000 words \- Added 20-term glossary \- Added detailed materials database comparison table \- Added MI ecosystem diagram (Mermaid) \- Added detailed explanation of 5-step workflow \- Added in-depth section on material descriptors \- Added 4 exercise problems (difficulty: 1 easy, 2 medium, 1 hard)
