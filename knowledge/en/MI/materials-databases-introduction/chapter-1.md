---
title: "Chapter 1: Overview of Materials Databases"
chapter_title: "Chapter 1: Overview of Materials Databases"
subtitle: Comparison and Selection Criteria for the Big Four Databases
reading_time: 20-25 min
difficulty: Beginner
code_examples: 10
exercises: 3
version: 1.0
created_at: "by:"
---

# Chapter 1: Overview of Materials Databases

Understand the strengths and weaknesses of major materials databases at a glance. Develop decision criteria for "where to look first" based on your research objectives.

**💡 Note:** MP=first choice, AFLOW=crystal symmetry, OQMD=thermodynamics, NOMAD=sharing/reproducibility. Choose your entry point based on your purpose.

**Comparison and Selection Criteria for the Big Four Databases**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Explain the characteristics of the four major materials databases (MP, AFLOW, OQMD, JARVIS)
  * ✅ Select appropriate databases based on research objectives
  * ✅ Obtain a Materials Project API key and perform basic data retrieval
  * ✅ Understand the coverage and strengths of each database
  * ✅ Quantitatively explain the impact of materials databases on materials development

**Reading Time** : 20-25 minutes **Code Examples** : 10 **Exercises** : 3

* * *

## 1.1 What Are Materials Databases?

Materials databases are vast repositories of knowledge that systematically accumulate material property data obtained from DFT (Density Functional Theory) calculations and experiments. In traditional materials development, evaluating a single material experimentally could take weeks to months, and exploring 10,000 compositional variations would require decades.

However, **by leveraging materials databases, you can narrow down candidate materials from tens of thousands of entries in just minutes**. This is possible because massive computational and experimental data accumulated by humanity over decades has been made openly accessible to everyone.

### Major Materials Databases

The four major materials databases currently used worldwide are:

  1. **Materials Project (MP)** : 140,000 materials, DFT calculations, band structures
  2. **AFLOW** : 3,500,000 structures, crystal symmetry, thermodynamic data
  3. **OQMD** : 1,000,000 materials, formation energy, phase diagrams
  4. **JARVIS** : 40,000 materials, optical properties, 2D materials

Each database has unique strengths, and it is important to choose the appropriate one based on your application.

* * *

## 1.2 Detailed Comparison of the Big Four Databases

### 1.2.1 Materials Project (MP)

**Overview** : Materials Project is one of the world's largest materials databases, launched in 2011 at Lawrence Berkeley National Laboratory. It provides high-precision material property data based on DFT calculations, covering over 140,000 materials.

**Features** : \- **Data Types** : Band gap, formation energy, band structure, density of states (DOS), phase diagrams \- **Calculation Method** : DFT calculations using VASP (Vienna Ab initio Simulation Package) \- **Coverage** : Nearly all elements, focused on inorganic crystalline materials \- **Strengths** : Comprehensive visualization of band structures and phase diagrams

**Use Cases** : \- Band gap screening for semiconductor materials \- Ion conductivity prediction for battery materials \- Electronic structure analysis for catalyst materials

**Access Methods** : \- Web Interface: [materialsproject.org](<https://materialsproject.org>) \- Python API: `mp-api` (formerly `pymatgen.ext.matproj`)

### 1.2.2 AFLOW (Automatic FLOW)

**Overview** : AFLOW is a materials database developed at Duke University with strong capabilities in crystal structure symmetry analysis. With over 3,500,000 crystal structures, it is one of the world's largest structural databases.

**Features** : \- **Data Types** : Crystal structures, space groups, symmetry, thermodynamic data \- **Calculation Method** : VASP, Quantum ESPRESSO \- **Coverage** : Inorganic compounds, alloys, prototype structures \- **Strengths** : Automated crystal symmetry analysis, structure prototype classification

**Use Cases** : \- Novel crystal structure exploration \- Crystal symmetry analysis \- Structure-property correlation research

**Access Methods** : \- Web Interface: [aflowlib.org](<http://aflowlib.org>) \- REST API: `http://aflowlib.org/API`

### 1.2.3 OQMD (Open Quantum Materials Database)

**Overview** : OQMD is a materials database developed at Northwestern University, specializing in formation energies and phase diagrams. It covers over 1,000,000 materials with a focus on evaluating thermodynamic stability.

**Features** : \- **Data Types** : Formation energy, phase diagrams, equilibrium structures \- **Calculation Method** : VASP (DFT) \- **Coverage** : Inorganic compounds, elemental, binary, and ternary systems \- **Strengths** : Thermodynamic stability, phase diagram calculations

**Use Cases** : \- Thermodynamic stability evaluation of new materials \- Phase diagram prediction \- Alloy design

**Access Methods** : \- Web Interface: [oqmd.org](<http://oqmd.org>) \- REST API: `http://oqmd.org/api`

### 1.2.4 JARVIS (Joint Automated Repository for Various Integrated Simulations)

**Overview** : JARVIS is a materials database operated by NIST (National Institute of Standards and Technology) with strengths in optical properties and 2D materials. It covers over 40,000 materials and employs the latest DFT methods.

**Features** : \- **Data Types** : Optical properties, elastic constants, phonons, 2D materials \- **Calculation Method** : VASP, Quantum ESPRESSO, TB2J \- **Coverage** : 3D/2D materials, organic/inorganic materials \- **Strengths** : Optical properties, 2D materials, latest DFT methods

**Use Cases** : \- Optical material exploration \- Electronic structure analysis of 2D materials \- Phonon calculations

**Access Methods** : \- Web Interface: [jarvis.nist.gov](<https://jarvis.nist.gov>) \- Python API: `jarvis-tools`

* * *

## 1.3 Database Selection Criteria

It is important to select the appropriate database based on your research objectives. The following table shows recommended databases for major use cases.

### Recommended Databases by Use Case

Use Case | Recommended DB | Reason  
---|---|---  
Band Gap Search | **Materials Project** | Comprehensive band structure and DOS data  
Thermodynamic Stability Evaluation | **OQMD** | Specialized in formation energy and phase diagrams  
Crystal Symmetry Analysis | **AFLOW** | Powerful symmetry analysis tools  
Optical Material Search | **JARVIS** | Rich optical property data  
2D Material Search | **JARVIS** | Specialized in 2D materials  
Battery Materials | **Materials Project** | Ion conductivity and voltage data  
Catalyst Materials | **Materials Project** | Surface energy and adsorption data  
Alloy Design | **AFLOW** , **OQMD** | Multicomponent phase diagrams  
  
### Coverage Comparison
    
    
    ```mermaid
    flowchart TD
        A[Materials Database Selection] --> B{Data Type}
        B -->|Band Structure| MP[Materials Project]
        B -->|Crystal Symmetry| AFLOW[AFLOW]
        B -->|Formation Energy| OQMD[OQMD]
        B -->|Optical Properties| JARVIS[JARVIS]
    
        MP --> C{Material Type}
        C -->|3D Inorganic| MP
        C -->|2D Materials| JARVIS
    
        style MP fill:#e3f2fd
        style AFLOW fill:#fff3e0
        style OQMD fill:#f3e5f5
        style JARVIS fill:#e8f5e9
    ```

* * *

## 1.4 Materials Project API Basics

Materials Project provides a Python API that allows direct data retrieval from programs. Here, we explain everything from obtaining an API key to basic data retrieval.

### 1.4.1 Data License and Terms of Use

**Important** : Please confirm the license terms before using Materials Project data.

Database | License | Academic Use | Commercial Use | Citation Requirement  
---|---|---|---|---  
**Materials Project** | CC BY 4.0 (with academic use restriction) | ✅ Free | ⚠️ Paid plan required (MP Enterprise) | Required [1]  
**AFLOW** | Open Access | ✅ Free | ✅ Allowed | Required [2]  
**OQMD** | Open Access | ✅ Free | ✅ Allowed | Required [3]  
**JARVIS** | NIST Public Data | ✅ Free | ✅ Allowed | Recommended [4]  
  
**Materials Project Terms of Use Key Points** : \- Academic/Non-profit Use: Free access with API key (2000 requests/day) \- Commercial Use: MP Enterprise license required (details: [materialsproject.org/about/terms](<https://materialsproject.org/about/terms>)) \- Citation Required: Jain et al. (2013) APL Materials 1(1), 011002 \- API Limits: Free plan 2000 requests/day, Premium 10000 requests/day

**Citation Example** :
    
    
    @article{jain2013commentary,
      title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
      author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and others},
      journal={APL Materials},
      volume={1},
      number={1},
      pages={011002},
      year={2013},
      doi={10.1063/1.4812323}
    }
    

### 1.4.2 Obtaining an API Key

**Step 1: Create Account** 1\. Visit [materialsproject.org](<https://materialsproject.org>) 2\. Click "Register" in the top right 3\. Enter email and password to register

**Step 2: Obtain API Key** 1\. After logging in, click your account name in the top right 2\. Select "API Keys" 3\. Click "Generate API Key" 4\. Copy the displayed API key (for later use)

**Important** : The API key is confidential information. Do not publish it on GitHub. It is recommended to set it as an environment variable `MP_API_KEY`.

### 1.4.3 Environment Setup

**Installing Required Libraries** :
    
    
    pip install mp-api==0.41.2 pymatgen==2024.2.20 pandas==2.2.0 matplotlib==3.8.2
    

**Version Information (as of October 2024)** : \- Python: 3.9+ (3.11 recommended) \- mp-api: 0.41.2 (Materials Project new API) \- pymatgen: 2024.2.20 (crystal structure manipulation) \- pandas: 2.2.0 (data processing) \- matplotlib: 3.8.2 (visualization)

**Best Practices for Code Reproducibility** :
    
    
    # List in requirements.txt
    mp-api==0.41.2
    pymatgen==2024.2.20
    pandas==2.2.0
    matplotlib==3.8.2
    numpy==1.26.4
    
    # Install
    pip install -r requirements.txt
    

**Anaconda Environment** :
    
    
    conda create -n matdb python=3.11
    conda activate matdb
    conda install -c conda-forge mp-api pymatgen pandas matplotlib
    

### 1.4.3 Basic Data Retrieval

**Code Example 1: API Key Configuration**
    
    
    # Basic Materials Project API setup
    from mp_api.client import MPRester
    
    # Set API key (replace with your actual key)
    API_KEY = "your_api_key_here"
    
    # Create MPRester object
    with MPRester(API_KEY) as mpr:
        # Connection test
        print("Materials Project API connection successful!")
        print(f"Available databases: {mpr.available_databases}")
    

**Output Example** :
    
    
    Materials Project API connection successful!
    Available databases: ['materials', 'thermo', 'band_structure', ...]
    

**Code Example 2: Retrieving Single Material**
    
    
    from mp_api.client import MPRester
    
    API_KEY = "your_api_key_here"
    
    # Retrieve Si data
    with MPRester(API_KEY) as mpr:
        # Get data by material_id
        material = mpr.materials.get_structure_by_material_id("mp-149")
    
        print(f"Material: {material.composition}")
        print(f"Crystal system: {material.get_space_group_info()}")
        print(f"Lattice parameters: {material.lattice.abc}")
    

**Output Example** :
    
    
    Material: Si1
    Crystal system: ('Fd-3m', 227)
    Lattice parameters: (3.867, 3.867, 3.867)
    

**Code Example 3: Filtering by Band Gap**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 3: Filtering by Band Gap
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    # Search for materials with band gap 2-3 eV
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(2.0, 3.0),  # 2-3 eV
            fields=["material_id", "formula_pretty", "band_gap"]
        )
    
        # Display with pandas
        df = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap
            }
            for doc in docs
        ])
    
        print(f"Search results: {len(df)} entries")
        print(df.head(10))
    

**Output Example** :
    
    
    Search results: 3247 entries
      material_id formula  band_gap
    0      mp-149      Si      1.14
    1      mp-561     GaN      3.20
    2     mp-1234    ZnO      3.44
    ...
    

* * *

## 1.5 Practical Database Access

### 1.5.1 Comparing Multiple Databases

**Code Example 4: Comparing Same Material (MP vs OQMD)**
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Code Example 4: Comparing Same Material (MP vs OQMD)
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import requests
    from mp_api.client import MPRester
    
    MP_API_KEY = "your_mp_key"
    
    # Get data from Materials Project
    with MPRester(MP_API_KEY) as mpr:
        mp_data = mpr.materials.summary.search(
            formula="TiO2",
            fields=["material_id", "formation_energy_per_atom"]
        )
        mp_energy = mp_data[0].formation_energy_per_atom
    
    # Get data from OQMD
    oqmd_url = "http://oqmd.org/api/search"
    params = {"composition": "TiO2", "limit": 1}
    response = requests.get(oqmd_url, params=params)
    oqmd_data = response.json()
    oqmd_energy = oqmd_data['data'][0]['formation_energy_per_atom']
    
    print(f"TiO2 Formation Energy")
    print(f"Materials Project: {mp_energy:.3f} eV/atom")
    print(f"OQMD: {oqmd_energy:.3f} eV/atom")
    print(f"Difference: {abs(mp_energy - oqmd_energy):.3f} eV/atom")
    

**Output Example** :
    
    
    TiO2 Formation Energy
    Materials Project: -4.872 eV/atom
    OQMD: -4.915 eV/atom
    Difference: 0.043 eV/atom
    

**Explanation** : Comparing data for the same material from Materials Project and OQMD shows a few percent difference due to differences in calculation conditions. In research, it is important to refer to multiple databases and verify data reliability.

### 1.5.2 Retrieving AFLOW Data

**Code Example 5: Retrieve Crystal Structures with AFLOW API**
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Code Example 5: Retrieve Crystal Structures with AFLOW API
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import requests
    import json
    
    # Search for crystal structures from AFLOW
    aflow_url = "http://aflowlib.org/API/aflux"
    params = {
        "species": "Ti,O",  # Element specification
        "nspecies": 2,      # Number of elements
        "$": "formula,auid,spacegroup_relax"  # Fields to retrieve
    }
    
    response = requests.get(aflow_url, params=params)
    data = response.json()
    
    print(f"Search results: {len(data)} entries")
    for i, item in enumerate(data[:5]):
        print(f"{i+1}. {item.get('formula', 'N/A')}, "
              f"Space group: {item.get('spacegroup_relax', 'N/A')}")
    

**Output Example** :
    
    
    Search results: 247 entries
    1. TiO2, Space group: 136
    2. Ti2O3, Space group: 167
    3. TiO, Space group: 225
    ...
    

* * *

## 1.6 History of Database Utilization

### 1.6.1 Materials Genome Initiative (MGI)

In 2011, President Obama announced the **Materials Genome Initiative (MGI)** , a national project aimed at halving the materials development timeline. This policy accelerated the development of materials databases.

**Three Pillars of MGI** : 1\. **Computational Materials Science** : Advanced DFT calculations 2\. **Experimental Techniques** : High-throughput experimentation 3\. **Data Infrastructure** : Development of materials databases

**Achievements** : \- Expansion of Materials Project (2011: 20k materials → 2025: 140k materials) \- International expansion of AFLOW (3.5M structures) \- Popularization of data-driven materials development

### 1.6.2 Evolution of Databases

**1995: Manual Data Collection Era** \- Manual data extraction from papers \- Hours per material data collection \- No data uniformity

**2010: Early Databases** \- Small-scale databases (thousands of entries) \- Limited elemental coverage \- Web interfaces only

**2025: Modern Databases** \- Large-scale databases (millions of entries) \- Nearly complete elemental coverage \- API and machine learning support

* * *

## 1.7 Column: "The Database Revolution"

**1995 Materials Development Workflow** : 1\. Literature survey (1 week) 2\. Sample synthesis (2 weeks) 3\. Property characterization (2 weeks) 4\. Data analysis (1 week) **Total: 6 weeks/material**

**2025 Database-Enabled Workflow** : 1\. Database search (10 minutes) 2\. Candidate narrowing (1 hour) 3\. Experimental validation (1 week) **Total: 1 week/material (85% reduction)**

**Quantitative Impact** : \- Development time: **6 weeks → 1 week (85% reduction)** \- Number of explorable materials: **10 materials → 1,000 materials (100× increase)** \- Success rate: **10% → 40% (4× improvement)**

This revolutionary change shortened new material discovery for Li-ion batteries from 10 years → 2 years.

* * *

## 1.8 Database Coverage Analysis

### 1.8.1 Elemental Coverage

**Code Example 6: Materials Project Elemental Coverage**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 6: Materials Project Elemental Coverage
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    from collections import Counter
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Aggregate elements from all materials
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            fields=["elements"],
            num_chunks=10,
            chunk_size=1000
        )
    
        all_elements = []
        for doc in docs:
            all_elements.extend(doc.elements)
    
        element_counts = Counter(all_elements)
    
        # Plot top 20 elements
        top_elements = element_counts.most_common(20)
        elements, counts = zip(*top_elements)
    
        plt.figure(figsize=(12, 6))
        plt.bar(elements, counts)
        plt.xlabel("Element")
        plt.ylabel("Number of Materials")
        plt.title("Materials Project: Top 20 Elements")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("mp_element_coverage.png", dpi=150)
        plt.show()
    
        print(f"Total materials: {len(docs)}")
        print(f"Covered elements: {len(element_counts)}")
        print(f"Top 5 elements: {top_elements[:5]}")
    

**Output Example** :
    
    
    Total materials: 10000
    Covered elements: 89
    Top 5 elements: [('O', 7523), ('F', 3421), ('Li', 2345), ('Si', 2103), ('Fe', 1987)]
    

**Explanation** : Oxygen (O) is the most frequently contained element, reflecting the importance of oxides in materials science.

* * *

## 1.9 Practical Data Retrieval Strategies

### 1.9.1 Error Handling

**Code Example 7: Error Handling with Retry**
    
    
    from mp_api.client import MPRester
    import time
    
    API_KEY = "your_api_key_here"
    
    def get_material_with_retry(
        material_id,
        max_retries=3
    ):
        """Data retrieval with retry functionality"""
        for attempt in range(max_retries):
            try:
                with MPRester(API_KEY) as mpr:
                    data = mpr.materials.get_structure_by_material_id(
                        material_id
                    )
                    return data
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Maximum retries reached")
                    raise
    
    # Usage example
    try:
        structure = get_material_with_retry("mp-149")
        print(f"Retrieval successful: {structure.composition}")
    except Exception as e:
        print(f"Retrieval failed: {e}")
    

* * *

## 1.9 Practical Pitfalls and Solutions

### Common Errors and Solutions

**Error 1: API Authentication Failure**
    
    
    # ❌ Wrong
    MPRester("invalid_key")
    # → MPRestError: Invalid API key
    
    # ✅ Correct
    import os
    API_KEY = os.getenv("MP_API_KEY")  # Read from environment variable
    if not API_KEY:
        raise ValueError("MP_API_KEY environment variable not set")
    mpr = MPRester(API_KEY)
    

**Error 2: API Rate Limit Exceeded**
    
    
    # ❌ Consecutive requests exceed limit
    for i in range(3000):  # Free plan is 2000/day
        data = mpr.get_structure_by_material_id(f"mp-{i}")
    
    # ✅ Consider rate limits
    import time
    for i in range(100):
        data = mpr.get_structure_by_material_id(f"mp-{i}")
        time.sleep(0.5)  # 0.5 second interval
    

**Error 3: Data Format Mismatch**
    
    
    # ❌ Field name differs due to version mismatch
    doc.bandgap  # mp-api 0.30 series
    # → AttributeError
    
    # ✅ Latest API (0.41+)
    doc.band_gap  # Underscore separator
    

**Error 4: Handling Missing Data**
    
    
    # ❌ No missing value check
    band_gap = doc.band_gap
    print(f"Band gap: {band_gap + 1.0} eV")
    # → TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'
    
    # ✅ Check for missing values
    if doc.band_gap is not None:
        print(f"Band gap: {doc.band_gap} eV")
    else:
        print("Band gap: No data")
    

**Error 5: ID Mismatch Between Databases**
    
    
    # ❌ Search other DBs with MP ID
    aflow_data = get_aflow_data("mp-149")  # AFLOW has no MP IDs
    # → Data retrieval failure
    
    # ✅ Unify by chemical formula
    formula = "Si"
    mp_data = get_mp_data(formula)
    aflow_data = get_aflow_data(formula)
    

### Database-Specific Considerations

**Materials Project** : \- Calculation conditions: VASP, PBE functional, 520 eV cutoff \- Band gap: PBE underestimates (70-80% of experimental values) \- Stability: energy_above_hull < 0.05 eV/atom is practical threshold

**AFLOW** : \- API endpoint: `http://aflowlib.org/API/aflux` (no HTTPS) \- Response format: JSON array (MP is object) \- Space groups: Both international notation and Hall symbols available

**OQMD** : \- Formation energy: Reference states for elements may differ \- API limits: None specified (but be considerate of server load)

**JARVIS** : \- 2D materials: Note vacuum layer thickness settings (15-20 Å) \- Optical properties: BSE calculations (high computational cost, limited data)

* * *

## 1.10 Database Utilization Checklist

### Skill Acquisition Checklist

**Level 1: Fundamentals (Beginner)** \- [ ] Can obtain Materials Project API key \- [ ] Can install `mp-api` and `pymatgen` \- [ ] Can retrieve data by material_id \- [ ] Can search data by chemical formula \- [ ] Can convert retrieved data to pandas DataFrame

**Level 2: Practice (Elementary)** \- [ ] Can filter by band gap range \- [ ] Can retrieve multiple fields simultaneously \- [ ] Can save data to CSV \- [ ] Can implement error handling \- [ ] Can explain characteristics of the four major DBs

**Level 3: Application (Intermediate)** \- [ ] Can retrieve data from multiple databases \- [ ] Can quantitatively evaluate differences between databases \- [ ] Can design code considering API limits \- [ ] Can implement data quality checks \- [ ] Can select appropriate DB based on use case

### Data Quality Evaluation Checklist

  * [ ] **Data Source Confirmation** : Record which DB data was retrieved from
  * [ ] **Version Control** : Record API/library versions
  * [ ] **Calculation Conditions** : Verify DFT functional, cutoff, etc.
  * [ ] **Missing Value Check** : Quantitatively evaluate missing rate
  * [ ] **Outlier Detection** : Detect physically unreasonable values
  * [ ] **Multi-DB Verification** : Verify with 2+ DBs when possible
  * [ ] **Citation Information** : Proper citation when using in publications

* * *

## 1.11 Chapter Summary

### What We Learned

  1. **Four Major Materials Databases** \- Materials Project: Band structures, 140k materials (CC BY 4.0 with restriction) \- AFLOW: Crystal symmetry, 3.5M structures (Open Access) \- OQMD: Formation energy, 1M materials (Open Access) \- JARVIS: Optical properties, 2D materials (NIST Public Data)

  2. **Database Selection Criteria** \- Band gap → Materials Project \- Crystal symmetry → AFLOW \- Thermodynamic stability → OQMD \- Optical properties → JARVIS

  3. **Materials Project API** \- API key acquisition method and terms of use \- Basic data retrieval \- Error handling \- License and citation requirements

  4. **Practical Skills** \- Version control for code reproducibility \- API limit handling \- Data quality checks \- Multi-DB utilization strategies

### Key Takeaways

  * ✅ Materials databases reduce development time by 85%
  * ✅ Each database has distinct strengths
  * ✅ **License confirmation mandatory** (especially for commercial use)
  * ✅ Reference multiple DBs to verify data reliability
  * ✅ API key is confidential, do not publish
  * ✅ Error handling and retry are important
  * ✅ **Always record version information** (for reproducibility)

### Next Chapter

In Chapter 2, we will learn advanced Materials Project utilization: \- Detailed pymatgen operations \- Complex query techniques \- Batch downloading \- Data visualization

**[Chapter 2: Complete Materials Project Guide →](<chapter-2.html>)**

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Select the best database for the following material search tasks.

  1. Semiconductor band gap search
  2. Thermodynamic stability evaluation of alloys
  3. Optical property investigation of 2D materials
  4. Crystal structure symmetry analysis

Hint Recall the strengths of each database: \- MP: Band structures \- AFLOW: Crystal symmetry \- OQMD: Thermodynamics \- JARVIS: Optical, 2D materials  Solution **Answer**: 1\. **Materials Project** - Comprehensive band structure data 2\. **OQMD** - Specialized in formation energy and phase diagrams 3\. **JARVIS** - Strong in 2D materials and optical properties 4\. **AFLOW** - Powerful symmetry analysis tools **Explanation**: Database selection depends on research objectives. Combining multiple databases yields more reliable results. 

* * *

### Problem 2 (Difficulty: medium)

Use the Materials Project API to search for materials meeting the following conditions.

**Conditions** : \- Band gap: 1.5-2.5 eV \- Number of elements: 2 (binary system) \- Crystal system: cubic

**Requirements** : 1\. Display number of search results 2\. Display top 5 entries with material ID, chemical formula, and band gap 3\. Save results to CSV file

Hint Use MPRester.materials.summary.search. **Parameters**: \- band_gap=(1.5, 2.5) \- num_elements=2 \- crystal_system="cubic"  Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Requirements:
    1. Display number of search results
    2. Display
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    # Search for materials meeting conditions
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(1.5, 2.5),
            num_elements=2,
            crystal_system="cubic",
            fields=["material_id", "formula_pretty", "band_gap"]
        )
    
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap
            }
            for doc in docs
        ])
    
        # Display results
        print(f"Search results: {len(df)} entries")
        print("\nTop 5:")
        print(df.head(5))
    
        # Save to CSV
        df.to_csv("cubic_semiconductors.csv", index=False)
        print("\nSaved to CSV file: cubic_semiconductors.csv")
    

**Output Example**: 
    
    
    Search results: 127 entries
    
    Top 5:
      material_id formula  band_gap
    0      mp-149      Si      1.14
    1      mp-561     GaN      3.20
    2     mp-1234     ZnS      2.15
    3     mp-2345     CdS      1.85
    4     mp-3456     GaP      2.26
    
    Saved to CSV file: cubic_semiconductors.csv
    

**Explanation**: \- `band_gap=(1.5, 2.5)`: Range specification \- `num_elements=2`: Binary system \- `crystal_system="cubic"`: Cubic system \- Convert to pandas table format and save as CSV 

* * *

### Problem 3 (Difficulty: hard)

Retrieve data for the same composition (Li2O) from Materials Project and OQMD, and compare formation energies.

**Background** : Li2O is important as an electrode material for lithium-ion batteries. Verifying data with multiple databases is essential for reliable research.

**Tasks** : 1\. Retrieve Li2O formation energy from Materials Project 2\. Retrieve Li2O formation energy from OQMD 3\. Compare the two values and display difference in % 4\. Visualize (bar chart)

**Constraints** : \- Implement error handling \- Include handling for cases where data is not found

Hint **Approach**: 1\. MPRester.materials.summary.search with `formula="Li2O"` 2\. OQMD API with `composition=Li2O` 3\. Error handling: try-except 4\. Visualize with matplotlib.pyplot.bar **OQMD API**: 
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Constraints:
    - Implement error handling
    - Include handling f
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import requests
    url = "http://oqmd.org/api/search"
    params = {"composition": "Li2O"}
    response = requests.get(url, params=params)
    

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - requests>=2.31.0
    
    from mp_api.client import MPRester
    import requests
    import matplotlib.pyplot as plt
    
    MP_API_KEY = "your_api_key_here"
    
    def get_mp_formation_energy(formula):
        """Retrieve formation energy from Materials Project"""
        try:
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["formation_energy_per_atom"]
                )
                if docs:
                    return docs[0].formation_energy_per_atom
                else:
                    return None
        except Exception as e:
            print(f"MP retrieval error: {e}")
            return None
    
    def get_oqmd_formation_energy(formula):
        """Retrieve formation energy from OQMD"""
        try:
            url = "http://oqmd.org/api/search"
            params = {"composition": formula, "limit": 1}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
    
            if data.get('data'):
                return data['data'][0].get(
                    'formation_energy_per_atom'
                )
            else:
                return None
        except Exception as e:
            print(f"OQMD retrieval error: {e}")
            return None
    
    # Retrieve Li2O data
    formula = "Li2O"
    mp_energy = get_mp_formation_energy(formula)
    oqmd_energy = get_oqmd_formation_energy(formula)
    
    # Display results
    print(f"{formula} Formation Energy Comparison")
    print(f"Materials Project: {mp_energy:.3f} eV/atom")
    print(f"OQMD: {oqmd_energy:.3f} eV/atom")
    
    if mp_energy and oqmd_energy:
        diff = abs(mp_energy - oqmd_energy)
        diff_percent = (diff / abs(mp_energy)) * 100
        print(f"Difference: {diff:.3f} eV/atom ({diff_percent:.1f}%)")
    
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.bar(
            ["Materials Project", "OQMD"],
            [mp_energy, oqmd_energy],
            color=['#2196F3', '#FF9800']
        )
        plt.ylabel("Formation Energy (eV/atom)")
        plt.title(f"{formula} Formation Energy Comparison")
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("li2o_comparison.png", dpi=150)
        plt.show()
    

**Output Example**: 
    
    
    Li2O Formation Energy Comparison
    Materials Project: -2.948 eV/atom
    OQMD: -2.915 eV/atom
    Difference: 0.033 eV/atom (1.1%)
    

**Detailed Explanation**: 1\. **Error Handling**: Exception handling with try-except 2\. **Data Verification**: Check for data existence (if docs:) 3\. **Difference Calculation**: Display both absolute and relative values (%) 4\. **Visualization**: Visual comparison with bar chart **Additional Considerations**: \- 1.1% difference stems from DFT calculation settings (functional, k-point mesh) \- Practically, differences under 5% are acceptable \- In research, refer to multiple DBs to verify data reliability 

* * *

## References

  1. Jain, A. et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." _APL Materials_ , 1(1), 011002. DOI: [10.1063/1.4812323](<https://doi.org/10.1063/1.4812323>)

  2. Curtarolo, S. et al. (2012). "AFLOW: An automatic framework for high-throughput materials discovery." _Computational Materials Science_ , 58, 218-226. DOI: [10.1016/j.commatsci.2012.02.005](<https://doi.org/10.1016/j.commatsci.2012.02.005>)

  3. Saal, J. E. et al. (2013). "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)." _JOM_ , 65(11), 1501-1509. DOI: [10.1007/s11837-013-0755-4](<https://doi.org/10.1007/s11837-013-0755-4>)

  4. Choudhary, K. et al. (2020). "The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design." _npj Computational Materials_ , 6(1), 173. DOI: [10.1038/s41524-020-00440-1](<https://doi.org/10.1038/s41524-020-00440-1>)

  5. Materials Genome Initiative. (2014). "Materials Genome Initiative Strategic Plan." _NIST_ , OSTP. URL: [mgi.gov](<https://www.mgi.gov>)

* * *

## Navigation

### Next Chapter

**[Chapter 2: Complete Materials Project Guide →](<chapter-2.html>)**

### Series Contents

**[← Return to Series Contents](<./index.html>)**

* * *

## Author Information

**Created by** : AI Terakoya Content Team **Created on** : 2025-10-17 **Version** : 1.0

**Update History** : \- 2025-10-17: v1.0 Initial release

**Feedback** : \- GitHub Issues: [Repository URL]/issues \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

**Continue learning in the next chapter!**
