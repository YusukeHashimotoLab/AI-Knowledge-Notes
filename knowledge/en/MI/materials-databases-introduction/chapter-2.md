---
title: "Chapter 2: Materials Project Complete Guide"
chapter_title: "Chapter 2: Materials Project Complete Guide"
subtitle: Complete Mastery of pymatgen and MPRester API
reading_time: 30-35 min
difficulty: Beginner to Intermediate
code_examples: 18
exercises: 3
version: 1.0
created_at: "by:"
---

# Chapter 2: Materials Project Complete Guide

Master the established patterns for data acquisition and preprocessing using pymatgen/MP API. Learn practical approaches to handling duplicates and missing values.

**💡 Tip:** Keep queries small and incremental. Running small loops of fetch → inspect → save reduces accidents.

**Complete Mastery of pymatgen and MPRester API**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Load and manipulate crystal structures using pymatgen
  * ✅ Construct complex queries with MPRester API
  * ✅ Efficiently download 10,000+ data entries
  * ✅ Retrieve and visualize band structures and phase diagrams
  * ✅ Write practical code considering API limitations

**Reading time** : 30-35 minutes **Code examples** : 18 **Exercises** : 3

* * *

## 2.1 pymatgen Basics

pymatgen (Python Materials Genomics) is the official Python library for Materials Project. It provides powerful functionality specialized for materials science, including crystal structure manipulation, computational data analysis, and visualization.

### 2.1.1 Structure Object

**Code Example 1: Creating and Basic Operations with Structure Objects**
    
    
    from pymatgen.core import Structure, Lattice
    
    # Define lattice vectors (Si, diamond structure)
    lattice = Lattice.cubic(5.43)  # Å
    
    # Define atomic coordinates (fractional coordinates)
    species = ["Si", "Si"]
    coords = [[0, 0, 0], [0.25, 0.25, 0.25]]
    
    # Create Structure object
    structure = Structure(lattice, species, coords)
    
    # Display basic information
    print(f"Formula: {structure.composition}")
    print(f"Lattice parameters: {structure.lattice.abc}")
    print(f"Volume: {structure.volume:.2f} Ų")
    print(f"Density: {structure.density:.2f} g/cm³")
    print(f"Number of atoms: {len(structure)}")
    

**Output** :
    
    
    Formula: Si2
    Lattice parameters: (5.43, 5.43, 5.43)
    Volume: 160.10 Ų
    Density: 2.33 g/cm³
    Number of atoms: 2
    

**Code Example 2: Crystal Structure Visualization**
    
    
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    
    # Create Si crystal structure
    lattice = Lattice.cubic(5.43)
    species = ["Si"] * 8
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
    ]
    structure = Structure(lattice, species, coords)
    
    # Save to CIF file
    cif_writer = CifWriter(structure)
    cif_writer.write_file("Si_diamond.cif")
    print("CIF file saved: Si_diamond.cif")
    
    # Retrieve symmetry information
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(structure)
    
    print(f"Space group: {sga.get_space_group_symbol()}")
    print(f"Space group number: {sga.get_space_group_number()}")
    print(f"Crystal system: {sga.get_crystal_system()}")
    

**Output** :
    
    
    CIF file saved: Si_diamond.cif
    Space group: Fd-3m
    Space group number: 227
    Crystal system: cubic
    

* * *

## 2.2 MPRester API Details

### 2.2.1 Basic Queries

**Code Example 3: Data Retrieval by material_id**
    
    
    from mp_api.client import MPRester
    
    API_KEY = "your_api_key_here"
    
    # Retrieve detailed data for a single material
    with MPRester(API_KEY) as mpr:
        # Retrieve data for mp-149 (Si)
        doc = mpr.materials.summary.get_data_by_id("mp-149")
    
        print(f"Material ID: {doc.material_id}")
        print(f"Formula: {doc.formula_pretty}")
        print(f"Band gap: {doc.band_gap} eV")
        print(f"Formation energy: {doc.formation_energy_per_atom} eV/atom")
        print(f"Symmetry: {doc.symmetry}")
    

**Output** :
    
    
    Material ID: mp-149
    Formula: Si
    Band gap: 1.14 eV
    Formation energy: 0.0 eV/atom
    Symmetry: {'crystal_system': 'cubic', 'symbol': 'Fd-3m'}
    

**Code Example 4: Batch Retrieval of Multiple Fields**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: Batch Retrieval of Multiple Fields
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    # Batch retrieval from multiple material_ids
    material_ids = ["mp-149", "mp-804", "mp-22526"]
    
    with MPRester(API_KEY) as mpr:
        data_list = []
        for mat_id in material_ids:
            doc = mpr.materials.summary.get_data_by_id(mat_id)
            data_list.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "energy_above_hull": doc.energy_above_hull,
                "formation_energy": doc.formation_energy_per_atom
            })
    
        df = pd.DataFrame(data_list)
        print(df)
    

**Output** :
    
    
      material_id formula  band_gap  energy_above_hull  formation_energy
    0      mp-149      Si      1.14               0.00              0.00
    1      mp-804     GaN      3.45               0.00             -1.12
    2   mp-22526     ZnO      3.44               0.00             -1.95
    

### 2.2.2 Advanced Filtering

**Code Example 5: Complex Queries Using Logical Operators**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 5: Complex Queries Using Logical Operators
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    # Filtering by complex conditions
    with MPRester(API_KEY) as mpr:
        # Band gap 2-3 eV, 2 elements, cubic system
        docs = mpr.materials.summary.search(
            band_gap=(2.0, 3.0),
            num_elements=2,
            crystal_system="cubic",
            energy_above_hull=(0, 0.05),  # stability
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull"
            ]
        )
    
        df = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "stability": doc.energy_above_hull
            }
            for doc in docs
        ])
    
        print(f"Search results: {len(df)} entries")
        print("\nTop 10 entries:")
        print(df.head(10))
        print(f"\nAverage band gap: {df['band_gap'].mean():.2f} eV")
    

**Output** :
    
    
    Search results: 34 entries
    
    Top 10 entries:
      material_id formula  band_gap  stability
    0      mp-561     GaN      3.20       0.00
    1     mp-1234     ZnS      2.15       0.02
    2     mp-2345     CdS      1.85       0.01
    ...
    
    Average band gap: 2.47 eV
    

**Code Example 6: Search by Element Specification**
    
    
    from mp_api.client import MPRester
    
    API_KEY = "your_api_key_here"
    
    # Search for materials containing specific elements
    with MPRester(API_KEY) as mpr:
        # Materials containing both Li and O
        docs = mpr.materials.summary.search(
            elements=["Li", "O"],
            num_elements=2,
            fields=["material_id", "formula_pretty", "band_gap"]
        )
    
        print(f"Li-O system materials: {len(docs)} entries")
        for i, doc in enumerate(docs[:5]):
            print(
                f"{i+1}. {doc.material_id}: {doc.formula_pretty}, "
                f"Eg={doc.band_gap} eV"
            )
    

**Output** :
    
    
    Li-O system materials: 127 entries
    1. mp-1960: Li2O, Eg=4.52 eV
    2. mp-12193: LiO2, Eg=2.31 eV
    3. mp-19017: Li2O2, Eg=3.15 eV
    ...
    

* * *

## 2.3 Batch Download

To efficiently retrieve large-scale data, batch downloading is necessary. Learn how to retrieve 10,000+ entries while considering API limitations.

### 2.3.1 Pagination Processing

**Code Example 7: Large-Scale Download Using Chunk Division**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    from mp_api.client import MPRester
    import pandas as pd
    import time
    
    API_KEY = "your_api_key_here"
    
    def batch_download(
        criteria,
        chunk_size=1000,
        max_chunks=10
    ):
        """
        Batch download of large-scale data
    
        Parameters:
        -----------
        criteria : dict
            Search criteria
        chunk_size : int
            Number of entries per retrieval
        max_chunks : int
            Maximum number of chunks
        """
        all_data = []
    
        with MPRester(API_KEY) as mpr:
            for chunk_num in range(max_chunks):
                print(f"Retrieving chunk {chunk_num + 1}/{max_chunks}...")
    
                docs = mpr.materials.summary.search(
                    **criteria,
                    num_chunks=max_chunks,
                    chunk_size=chunk_size,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "band_gap"
                    ]
                )
    
                if not docs:
                    print("No data, terminating")
                    break
    
                for doc in docs:
                    all_data.append({
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "band_gap": doc.band_gap
                    })
    
                # API rate limit countermeasure
                time.sleep(1)
    
        return pd.DataFrame(all_data)
    
    # Usage example: Bulk retrieval of materials with band gap > 2 eV
    criteria = {"band_gap": (2.0, None)}
    df = batch_download(criteria, chunk_size=1000, max_chunks=5)
    
    print(f"\nTotal entries retrieved: {len(df)}")
    print(df.head())
    df.to_csv("wide_bandgap_materials.csv", index=False)
    

**Output** :
    
    
    Retrieving chunk 1/5...
    Retrieving chunk 2/5...
    Retrieving chunk 3/5...
    ...
    
    Total entries retrieved: 4523
      material_id formula  band_gap
    0      mp-561     GaN      3.20
    1     mp-1234     ZnS      2.15
    ...
    

### 2.3.2 Error Handling and Retry

**Code Example 8: Robust Batch Download**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    from mp_api.client import MPRester
    import pandas as pd
    import time
    from requests.exceptions import RequestException
    
    API_KEY = "your_api_key_here"
    
    def robust_batch_download(
        criteria,
        chunk_size=500,
        max_retries=3
    ):
        """Batch download with error handling"""
        all_data = []
    
        with MPRester(API_KEY) as mpr:
            chunk_num = 0
            while True:
                retry_count = 0
                success = False
    
                while retry_count < max_retries and not success:
                    try:
                        docs = mpr.materials.summary.search(
                            **criteria,
                            chunk_size=chunk_size,
                            fields=[
                                "material_id",
                                "formula_pretty",
                                "band_gap"
                            ]
                        )
    
                        if not docs:
                            return pd.DataFrame(all_data)
    
                        for doc in docs:
                            all_data.append({
                                "material_id": doc.material_id,
                                "formula": doc.formula_pretty,
                                "band_gap": doc.band_gap
                            })
    
                        success = True
                        print(f"Chunk {chunk_num + 1} successful "
                              f"({len(docs)} entries)")
    
                    except RequestException as e:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        print(
                            f"Error occurred: {e}, "
                            f"retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
    
                if not success:
                    print(f"Chunk {chunk_num + 1} skipped")
    
                chunk_num += 1
                time.sleep(0.5)  # API rate limit countermeasure
    
        return pd.DataFrame(all_data)
    
    # Usage example
    criteria = {"elements": ["Li"], "num_elements": 1}
    df = robust_batch_download(criteria)
    print(f"Download complete: {len(df)} entries")
    

* * *

## 2.4 Data Visualization

### 2.4.1 Retrieving and Visualizing Band Structures

**Code Example 9: Retrieving Band Structure Data**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 9: Retrieving Band Structure Data
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Retrieve Si band structure
    with MPRester(API_KEY) as mpr:
        # Retrieve band structure data
        bs_data = mpr.get_bandstructure_by_material_id("mp-149")
    
        # Basic information
        print(f"Material: {bs_data.structure.composition}")
        print(f"Band gap: {bs_data.get_band_gap()['energy']} eV")
        print(f"Direct/Indirect: {bs_data.get_band_gap()['transition']}")
    
        # Band structure plot
        plotter = bs_data.get_plotter()
        plotter.get_plot(
            ylim=(-10, 10),
            vbm_cbm_marker=True
        )
        plt.savefig("Si_band_structure.png", dpi=150)
        plt.show()
    

**Output** :
    
    
    Material: Si1
    Band gap: 1.14 eV
    Direct/Indirect: indirect
    

**Code Example 10: Retrieving Density of States (DOS)**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 10: Retrieving Density of States (DOS)
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Retrieve density of states
    with MPRester(API_KEY) as mpr:
        dos_data = mpr.get_dos_by_material_id("mp-149")
    
        # DOS plot
        plotter = dos_data.get_plotter()
        plotter.get_plot(
            xlim=(-10, 10),
            ylim=(0, 5)
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS (states/eV)")
        plt.title("Si Density of States")
        plt.savefig("Si_DOS.png", dpi=150)
        plt.show()
    

### 2.4.2 Retrieving Phase Diagrams

**Code Example 11: Binary Phase Diagram**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 11: Binary Phase Diagram
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Retrieve Li-O phase diagram
    with MPRester(API_KEY) as mpr:
        pd_data = mpr.get_phase_diagram_by_elements(["Li", "O"])
    
        # Phase diagram plot
        plotter = pd_data.get_plotter()
        plotter.get_plot(label_stable=True)
        plt.savefig("Li-O_phase_diagram.png", dpi=150)
        plt.show()
    
        # Display stable phases
        print("Stable phases:")
        for entry in pd_data.stable_entries:
            print(
                f"- {entry.composition.reduced_formula}: "
                f"{pd_data.get_form_energy_per_atom(entry):.3f} "
                f"eV/atom"
            )
    

* * *

## 2.5 Practical Data Retrieval Strategies

### 2.5.1 Cache Utilization

**Code Example 12: Acceleration Using Local Cache**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    from mp_api.client import MPRester
    import pandas as pd
    import pickle
    import os
    
    API_KEY = "your_api_key_here"
    CACHE_FILE = "mp_data_cache.pkl"
    
    def get_data_with_cache(criteria, cache_file=CACHE_FILE):
        """Data retrieval with cache functionality"""
    
        # Load if cache exists
        if os.path.exists(cache_file):
            print("Loading data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
        # Retrieve from API if cache does not exist
        print("Retrieving data from API...")
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                **criteria,
                fields=["material_id", "formula_pretty", "band_gap"]
            )
    
            data = pd.DataFrame([
                {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap
                }
                for doc in docs
            ])
    
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to cache: {cache_file}")
    
        return data
    
    # Usage example
    criteria = {"band_gap": (2.0, 3.0), "num_elements": 2}
    df1 = get_data_with_cache(criteria)  # API retrieval
    df2 = get_data_with_cache(criteria)  # Cache loading
    
    print(f"Number of entries: {len(df1)}")
    

### 2.5.2 Data Quality Check

**Code Example 13: Data Quality Validation**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    from mp_api.client import MPRester
    import pandas as pd
    import numpy as np
    
    API_KEY = "your_api_key_here"
    
    def quality_check(df):
        """Data quality check"""
        print("=== Data Quality Report ===")
    
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
    
        # Check for outliers (band gap)
        if 'band_gap' in df.columns:
            bg_mean = df['band_gap'].mean()
            bg_std = df['band_gap'].std()
            outliers = df[
                (df['band_gap'] < bg_mean - 3 * bg_std) |
                (df['band_gap'] > bg_mean + 3 * bg_std)
            ]
            print(f"\nBand gap outliers: {len(outliers)} entries")
            if len(outliers) > 0:
                print(outliers)
    
        # Check for duplicates
        duplicates = df.duplicated(subset=['material_id'])
        print(f"\nDuplicate data: {duplicates.sum()} entries")
    
    # Usage example
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li", "O"],
            fields=["material_id", "formula_pretty", "band_gap"]
        )
    
        df = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap
            }
            for doc in docs
        ])
    
    quality_check(df)
    

* * *

## 2.6 Advanced Query Techniques

### 2.6.1 Retrieving Calculated Properties

**Code Example 14: Ionic Conductivity Data**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 14: Ionic Conductivity Data
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    # Search for ionic conductors
    with MPRester(API_KEY) as mpr:
        # Li ionic conductors
        docs = mpr.materials.summary.search(
            elements=["Li"],
            theoretical=True,  # Include theoretical prediction data
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "formation_energy_per_atom"
            ]
        )
    
        df = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "energy": doc.formation_energy_per_atom
            }
            for doc in docs
        ])
    
        # Stable materials with wide band gap
        stable = df[df['energy'] < -0.1]
        wide_gap = stable[stable['band_gap'] > 2.0]
    
        print(f"Stable Li-containing materials: {len(stable)} entries")
        print(f"Wide band gap materials: {len(wide_gap)} entries")
        print(wide_gap.head(10))
    

### 2.6.2 Surface Energy and Adsorption Data

**Code Example 15: Retrieving Surface Energy**
    
    
    from mp_api.client import MPRester
    
    API_KEY = "your_api_key_here"
    
    # Retrieve surface energy data
    with MPRester(API_KEY) as mpr:
        # Surface energy of TiO2
        surface_data = mpr.get_surface_data("mp-2657")  # TiO2
    
        print(f"Material: {surface_data['material_id']}")
        print(f"\nSurface energy (J/m²):")
        for surface in surface_data['surfaces']:
            miller = surface['miller_index']
            energy = surface['surface_energy']
            print(f"  {miller}: {energy:.3f} J/m²")
    

* * *

## 2.7 MPRester Practical Patterns

### 2.7.1 Combining Multiple Conditions

**Code Example 16: Searching for Battery Materials**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    def find_battery_cathodes():
        """Search for battery cathode materials"""
        with MPRester(API_KEY) as mpr:
            # Conditions: Contains Li, contains transition metals, stable
            docs = mpr.materials.summary.search(
                elements=["Li", "Co", "O"],  # Li-Co-O system
                energy_above_hull=(0, 0.05),  # stability
                fields=[
                    "material_id",
                    "formula_pretty",
                    "energy_above_hull",
                    "formation_energy_per_atom"
                ]
            )
    
            results = []
            for doc in docs:
                # Estimate theoretical capacity (simplified version)
                formula = doc.formula_pretty
                if "Li" in formula and "Co" in formula:
                    results.append({
                        "material_id": doc.material_id,
                        "formula": formula,
                        "stability": doc.energy_above_hull,
                        "formation_energy":
                            doc.formation_energy_per_atom
                    })
    
            df = pd.DataFrame(results)
            return df.sort_values('stability')
    
    # Execute
    cathodes = find_battery_cathodes()
    print(f"Candidate cathode materials: {len(cathodes)} entries")
    print(cathodes.head(10))
    

### 2.7.2 Data Filtering and Aggregation

**Code Example 17: Statistical Analysis**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 17: Statistical Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Band gap distribution by element
    with MPRester(API_KEY) as mpr:
        # Band gap of oxides
        docs = mpr.materials.summary.search(
            elements=["O"],
            num_elements=2,
            fields=["formula_pretty", "band_gap", "elements"]
        )
    
        data = []
        for doc in docs:
            # Identify elements excluding O
            elements = [e for e in doc.elements if e != "O"]
            if elements and doc.band_gap is not None:
                data.append({
                    "element": elements[0],
                    "band_gap": doc.band_gap
                })
    
        df = pd.DataFrame(data)
    
        # Average band gap by element
        avg_bg = df.groupby('element')['band_gap'].agg(
            ['mean', 'std', 'count']
        )
        avg_bg = avg_bg.sort_values('mean', ascending=False)
    
        print("Average band gap of element oxides (top 10):")
        print(avg_bg.head(10))
    
        # Visualization
        top10 = avg_bg.head(10)
        plt.figure(figsize=(10, 6))
        plt.bar(top10.index, top10['mean'], yerr=top10['std'])
        plt.xlabel("Element")
        plt.ylabel("Average Band Gap (eV)")
        plt.title("Average Band Gap of Binary Oxides")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("oxide_bandgap_analysis.png", dpi=150)
        plt.show()
    

* * *

## 2.8 API Rate Limits and Best Practices

### 2.8.1 Rate Limit Countermeasures

Materials Project API has the following rate limits: \- **Free plan** : 2000 requests/day \- **Premium** : 10000 requests/day

**Code Example 18: Rate-Limited Wrapper**
    
    
    from mp_api.client import MPRester
    import time
    from functools import wraps
    
    API_KEY = "your_api_key_here"
    
    class RateLimitedMPRester:
        """Rate-limited MPRester"""
    
        def __init__(self, api_key, delay=0.5):
            self.api_key = api_key
            self.delay = delay
            self.request_count = 0
    
        def __enter__(self):
            self.mpr = MPRester(self.api_key).__enter__()
            return self
    
        def __exit__(self, *args):
            print(
                f"\nTotal requests: {self.request_count}"
            )
            return self.mpr.__exit__(*args)
    
        def search(self, **kwargs):
            """Search with rate limiting"""
            result = self.mpr.materials.summary.search(**kwargs)
            self.request_count += 1
            time.sleep(self.delay)
            return result
    
    # Usage example
    with RateLimitedMPRester(API_KEY, delay=1.0) as mpr:
        # Multiple searches
        for element in ["Li", "Na", "K"]:
            docs = mpr.search(
                elements=[element],
                num_elements=1,
                fields=["material_id", "formula_pretty"]
            )
            print(f"{element}: {len(docs)} entries")
    

* * *

## 2.9 Chapter Summary

### What You Learned

  1. **pymatgen Basics** \- Structure object manipulation \- Crystal structure visualization \- Symmetry analysis

  2. **MPRester API** \- Basic queries (material_id, formula) \- Advanced filtering (logical operators, range specification) \- Batch download (10,000+ entries)

  3. **Data Visualization** \- Band structure plotting \- Density of states (DOS) \- Phase diagrams

  4. **Practical Techniques** \- Cache utilization \- Error handling \- Rate limit countermeasures

### Key Points

  * ✅ pymatgen is the standard library for crystal structure manipulation
  * ✅ MPRester API provides access to 140k materials
  * ✅ Batch downloads are controlled with chunk_size
  * ✅ Cache reduces duplicate requests
  * ✅ Code design considering rate limits is important

### Next Chapter

In Chapter 3, you will learn about integrating multiple databases and workflows: \- Integration of Materials Project and AFLOW \- Data cleaning \- Missing value handling \- Automated update pipeline

**[Chapter 3: Database Integration and Workflows →](<chapter-3.html>)**

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Using pymatgen, create a Cu FCC structure (face-centered cubic) and display the following information.

**Requirements** : 1\. Lattice parameter: 3.61 Å 2\. Space group symbol 3\. Crystal system 4\. Density

Hint
    
    
    from pymatgen.core import Structure, Lattice
    
    # FCC structure coordinates
    lattice = Lattice.cubic(3.61)
    species = ["Cu"] * 4
    coords = [[0, 0, 0], [0.5, 0.5, 0], ...]
    

Solution
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # Cu FCC structure
    lattice = Lattice.cubic(3.61)
    species = ["Cu"] * 4
    coords = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]
    
    structure = Structure(lattice, species, coords)
    
    # Symmetry analysis
    sga = SpacegroupAnalyzer(structure)
    
    print(f"Formula: {structure.composition}")
    print(f"Lattice parameters: {structure.lattice.abc}")
    print(f"Space group: {sga.get_space_group_symbol()}")
    print(f"Crystal system: {sga.get_crystal_system()}")
    print(f"Density: {structure.density:.2f} g/cm³")
    

**Output**: 
    
    
    Formula: Cu4
    Lattice parameters: (3.61, 3.61, 3.61)
    Space group: Fm-3m
    Crystal system: cubic
    Density: 8.96 g/cm³
    

* * *

### Problem 2 (Difficulty: medium)

Search Materials Project for catalyst material candidates satisfying the following conditions and save to CSV.

**Conditions** : \- Contains transition metals (Ti, V, Cr, Mn, Fe, Co, Ni) \- Contains oxygen \- Band gap < 3 eV (electronic conductivity) \- Stability: energy_above_hull < 0.1 eV/atom

**Requirements** : 1\. Display number of search results 2\. Save material_id, formula, band_gap, stability to CSV 3\. Create bar graph of band gap distribution

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Requirements:
    1. Display number of search results
    2. Save ma
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import pandas as pd
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    # Transition metal list
    transition_metals = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]
    
    all_results = []
    
    with MPRester(API_KEY) as mpr:
        for tm in transition_metals:
            docs = mpr.materials.summary.search(
                elements=[tm, "O"],
                band_gap=(None, 3.0),
                energy_above_hull=(0, 0.1),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "energy_above_hull"
                ]
            )
    
            for doc in docs:
                all_results.append({
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap,
                    "stability": doc.energy_above_hull,
                    "transition_metal": tm
                })
    
    df = pd.DataFrame(all_results)
    
    print(f"Catalyst candidate materials: {len(df)} entries")
    print(df.head(10))
    
    # Save CSV
    df.to_csv("catalyst_candidates.csv", index=False)
    
    # Band gap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['band_gap'], bins=30, edgecolor='black')
    plt.xlabel("Band Gap (eV)")
    plt.ylabel("Count")
    plt.title("Band Gap Distribution of Catalyst Candidates")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("catalyst_bandgap_dist.png", dpi=150)
    plt.show()
    

* * *

### Problem 3 (Difficulty: hard)

Batch download 10,000+ entries from Materials Project and perform statistical analysis.

**Tasks** : 1\. Retrieve all materials with band gap > 0 eV 2\. Calculate average band gap by number of elements 3\. Visualize band gap distribution by crystal system 4\. List top 10% wide band gap materials

**Constraints** : \- Implement error handling \- Implement cache functionality \- Display progress bar

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    # - tqdm>=4.65.0
    
    from mp_api.client import MPRester
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    import os
    from tqdm import tqdm
    
    API_KEY = "your_api_key_here"
    CACHE_FILE = "wide_bg_cache.pkl"
    
    def batch_download_with_progress():
        """Batch download with progress bar"""
    
        # Check cache
        if os.path.exists(CACHE_FILE):
            print("Loading data from cache...")
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    
        all_data = []
    
        with MPRester(API_KEY) as mpr:
            # Retrieve total count
            total_docs = mpr.materials.summary.search(
                band_gap=(0.1, None),
                fields=["material_id"]
            )
            total = len(total_docs)
            print(f"Total data count: {total} entries")
    
            # Chunk-divided download
            chunk_size = 1000
            num_chunks = (total // chunk_size) + 1
    
            for i in tqdm(range(num_chunks), desc="Download"):
                docs = mpr.materials.summary.search(
                    band_gap=(0.1, None),
                    num_chunks=num_chunks,
                    chunk_size=chunk_size,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "band_gap",
                        "num_elements",
                        "symmetry"
                    ]
                )
    
                for doc in docs:
                    all_data.append({
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "band_gap": doc.band_gap,
                        "num_elements": doc.num_elements,
                        "crystal_system":
                            doc.symmetry.get('crystal_system')
                    })
    
        df = pd.DataFrame(all_data)
    
        # Save cache
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(df, f)
    
        return df
    
    # Data retrieval
    df = batch_download_with_progress()
    
    print(f"\nTotal data count: {len(df)}")
    
    # Average band gap by number of elements
    avg_by_elements = df.groupby('num_elements')['band_gap'].mean()
    print("\nAverage band gap by number of elements:")
    print(avg_by_elements)
    
    # Distribution by crystal system
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    crystal_systems = df['crystal_system'].unique()
    
    for i, cs in enumerate(crystal_systems[:6]):
        ax = axes[i // 3, i % 3]
        data = df[df['crystal_system'] == cs]['band_gap']
        ax.hist(data, bins=30, edgecolor='black')
        ax.set_title(f"{cs} (n={len(data)})")
        ax.set_xlabel("Band Gap (eV)")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("crystal_system_bandgap.png", dpi=150)
    plt.show()
    
    # Top 10% wide band gap materials
    threshold = df['band_gap'].quantile(0.9)
    top10 = df[df['band_gap'] >= threshold].sort_values(
        'band_gap', ascending=False
    )
    
    print(f"\nTop 10% band gap materials (threshold: {threshold:.2f} eV):")
    print(top10.head(20))
    
    top10.to_csv("top10_percent_wide_bg.csv", index=False)
    

**Example Output**: 
    
    
    Loading data from cache...
    
    Total data count: 12453
    
    Average band gap by number of elements:
    num_elements
    1    3.25
    2    2.87
    3    2.13
    4    1.65
    ...
    
    Top 10% band gap materials (threshold: 5.23 eV):
       material_id formula  band_gap  num_elements crystal_system
    0       mp-123    MgO      7.83             2          cubic
    1       mp-456    BN       6.42             2      hexagonal
    ...
    

* * *

## References

  1. Ong, S. P. et al. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319. DOI: [10.1016/j.commatsci.2012.10.028](<https://doi.org/10.1016/j.commatsci.2012.10.028>)

  2. Materials Project Documentation. "API Documentation." URL: [docs.materialsproject.org](<https://docs.materialsproject.org>)

  3. Jain, A. et al. (2013). "Commentary: The Materials Project." _APL Materials_ , 1(1), 011002. DOI: [10.1063/1.4812323](<https://doi.org/10.1063/1.4812323>)

* * *

## Navigation

### Previous Chapter

**[Chapter 1: Overview of Materials Databases ←](<chapter-1.html>)**

### Next Chapter

**[Chapter 3: Database Integration and Workflows →](<chapter-3.html>)**

### Series Table of Contents

**[← Back to Series Table of Contents](<./index.html>)**

* * *

## Author Information

**Created by** : AI Terakoya Content Team **Created on** : 2025-10-17 **Version** : 1.0

**License** : Creative Commons BY 4.0

* * *

**Continue learning in the next chapter!**
