---
title: "Chapter 3: Database Integration and Workflows"
chapter_title: "Chapter 3: Database Integration and Workflows"
subtitle: Hands-on Multi-Database Integration and Data Cleaning
reading_time: 20-25 min
difficulty: Beginner to Intermediate
code_examples: 12
exercises: 3
version: 1.0
created_at: "by:"
---

# Chapter 3: Database Integration and Workflows

Learn how to integrate data by overcoming schema differences. Master the essentials of quality control and traceability.

**💡 Note:** Key alignment and unit normalization are critical. Maintaining provenance allows for later verification.

**Hands-on Multi-Database Integration and Data Cleaning**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Integrate Materials Project and AFLOW data
  * ✅ Apply standard data cleaning techniques
  * ✅ Handle missing values appropriately
  * ✅ Build automated update pipelines
  * ✅ Quantitatively evaluate data quality

**Reading Time** : 20-25 minutes **Code Examples** : 12 **Exercises** : 3

* * *

## 3.1 Multi-Database Integration

In materials research, combining multiple databases rather than relying on a single source leads to more reliable results. Here, you'll learn how to integrate Materials Project and AFLOW data.

### 3.1.1 Basic Database Integration Strategy
    
    
    ```mermaid
    flowchart TD
        A[Materials Project] --> C[Match with Common Identifiers]
        B[AFLOW] --> C
        C --> D[Data Merge]
        D --> E[Duplicate Removal]
        E --> F[Missing Value Handling]
        F --> G[Integrated Dataset]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style G fill:#e8f5e9
    ```

**Code Example 1: Fetching Materials Project and AFLOW Data**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    # - requests>=2.31.0
    
    from mp_api.client import MPRester
    import requests
    import pandas as pd
    
    MP_API_KEY = "your_mp_key"
    
    def get_mp_data(formula):
        """Fetch data from Materials Project"""
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_above_hull"
                ]
            )
    
            if docs:
                doc = docs[0]
                return {
                    "source": "Materials Project",
                    "id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap,
                    "formation_energy":
                        doc.formation_energy_per_atom,
                    "stability": doc.energy_above_hull
                }
        return None
    
    def get_aflow_data(formula):
        """Fetch data from AFLOW"""
        url = "http://aflowlib.org/API/aflux"
        params = {
            "species": formula,
            "$": "formula,Egap,enthalpy_formation_atom"
        }
    
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
    
            if data:
                item = data[0]
                return {
                    "source": "AFLOW",
                    "id": item.get("auid", "N/A"),
                    "formula": item.get("formula", formula),
                    "band_gap": item.get("Egap", None),
                    "formation_energy":
                        item.get("enthalpy_formation_atom", None),
                    "stability": None  # AFLOW doesn't have direct value
                }
        except Exception as e:
            print(f"AFLOW fetch error: {e}")
    
        return None
    
    # Fetch data for TiO2 from both sources
    formula = "TiO2"
    mp_data = get_mp_data(formula)
    aflow_data = get_aflow_data(formula)
    
    print(f"=== {formula} Data Comparison ===")
    print(f"\nMaterials Project:")
    print(mp_data)
    print(f"\nAFLOW:")
    print(aflow_data)
    

**Output Example** :
    
    
    === TiO2 Data Comparison ===
    
    Materials Project:
    {'source': 'Materials Project', 'id': 'mp-2657', 'formula': 'TiO2',
     'band_gap': 3.44, 'formation_energy': -4.872, 'stability': 0.0}
    
    AFLOW:
    {'source': 'AFLOW', 'id': 'aflow:123456', 'formula': 'TiO2',
     'band_gap': 3.38, 'formation_energy': -4.915, 'stability': None}
    

### 3.1.2 Data Join

**Code Example 2: Join Based on Chemical Formula**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    
    def merge_databases(formulas):
        """Merge data from multiple databases"""
        mp_data_list = []
        aflow_data_list = []
    
        for formula in formulas:
            mp_data = get_mp_data(formula)
            if mp_data:
                mp_data_list.append(mp_data)
    
            aflow_data = get_aflow_data(formula)
            if aflow_data:
                aflow_data_list.append(aflow_data)
    
        # Convert to DataFrames
        df_mp = pd.DataFrame(mp_data_list)
        df_aflow = pd.DataFrame(aflow_data_list)
    
        # Merge on chemical formula
        df_merged = pd.merge(
            df_mp,
            df_aflow,
            on="formula",
            how="outer",
            suffixes=("_mp", "_aflow")
        )
    
        return df_merged
    
    # Integrate multiple materials
    formulas = ["TiO2", "ZnO", "GaN", "SiC"]
    df = merge_databases(formulas)
    
    print("Integrated Dataset:")
    print(df)
    print(f"\nTotal Records: {len(df)}")
    

**Output Example** :
    
    
    Integrated Dataset:
      formula    id_mp  band_gap_mp  formation_energy_mp    id_aflow  band_gap_aflow  ...
    0    TiO2  mp-2657         3.44                -4.872  aflow:123              3.38  ...
    1     ZnO  mp-2133         3.44                -1.950  aflow:456              3.41  ...
    ...
    
    Total Records: 4
    

* * *

## 3.2 Data Cleaning

### 3.2.1 Duplicate Detection and Removal

**Code Example 3: Duplicate Detection**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    
    def detect_duplicates(df):
        """Detect duplicate data"""
        # Duplicates based on chemical formula
        duplicates = df[df.duplicated(subset=['formula'],
                                       keep=False)]
    
        print("=== Duplicate Detection ===")
        print(f"Total Records: {len(df)}")
        print(f"Duplicates: {len(duplicates)}")
    
        if len(duplicates) > 0:
            print("\nDuplicate Data:")
            print(duplicates[['formula', 'source', 'id']])
    
        # Remove duplicates (keep first entry)
        df_clean = df.drop_duplicates(subset=['formula'],
                                       keep='first')
    
        print(f"\nAfter Removal: {len(df_clean)} records")
        return df_clean
    
    # Usage example
    df_clean = detect_duplicates(df)
    

### 3.2.2 Outlier Detection

**Code Example 4: Outlier Detection (IQR Method)**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def detect_outliers(df, column):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
    
        # Outlier range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        # Detect outliers
        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]
    
        print(f"=== {column} Outlier Detection ===")
        print(f"Q1: {Q1:.3f}, Q3: {Q3:.3f}, IQR: {IQR:.3f}")
        print(f"Range: [{lower_bound:.3f}, {upper_bound:.3f}]")
        print(f"Outliers: {len(outliers)} records")
    
        if len(outliers) > 0:
            print("\nOutlier Data:")
            print(outliers[['formula', column]])
    
        return outliers
    
    # Detect band gap outliers
    outliers_bg = detect_outliers(df_clean, 'band_gap_mp')
    

* * *

## 3.3 Missing Value Handling

### 3.3.1 Missing Pattern Visualization

**Code Example 5: Missing Value Analysis**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def analyze_missing_values(df):
        """Detailed missing value analysis"""
        # Count missing values
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
    
        # Create report
        missing_df = pd.DataFrame({
            "Column": missing_count.index,
            "Missing Count": missing_count.values,
            "Missing %": missing_percent.values
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        missing_df = missing_df.sort_values("Missing %",
                                            ascending=False)
    
        print("=== Missing Value Report ===")
        print(missing_df)
    
        # Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df.isnull(),
            cbar=True,
            cmap='viridis',
            yticklabels=False
        )
        plt.title("Missing Values Heatmap")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("missing_values_heatmap.png", dpi=150)
        plt.show()
    
        return missing_df
    
    # Missing value analysis
    missing_report = analyze_missing_values(df_clean)
    

### 3.3.2 Missing Value Imputation

**Code Example 6: Imputation Strategies**
    
    
    from sklearn.impute import SimpleImputer, KNNImputer
    
    def impute_missing_values(df, method='mean'):
        """
        Missing value imputation
    
        Parameters:
        -----------
        method : str
            One of 'mean', 'median', 'knn'
        """
        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns
    
        df_imputed = df.copy()
    
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown method: {method}")
    
        # Impute numeric columns only
        df_imputed[numeric_cols] = imputer.fit_transform(
            df[numeric_cols]
        )
    
        print(f"=== Missing Value Imputation ({method} method) ===")
        print(f"Before: {df.isnull().sum().sum()} cells")
        print(f"After: {df_imputed.isnull().sum().sum()} cells")
    
        return df_imputed
    
    # Mean imputation
    df_imputed = impute_missing_values(df_clean, method='mean')
    
    # KNN imputation (more advanced)
    df_knn = impute_missing_values(df_clean, method='knn')
    

* * *

## 3.4 Automated Update Pipeline

### 3.4.1 Data Acquisition Pipeline

**Code Example 7: Automated Update Script**
    
    
    from datetime import datetime
    import json
    import os
    
    class DataUpdatePipeline:
        """Automated data update pipeline"""
    
        def __init__(self, output_dir="data"):
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
    
        def fetch_data(self, formulas):
            """Fetch data"""
            print(f"Starting data fetch: {datetime.now()}")
            df = merge_databases(formulas)
            return df
    
        def clean_data(self, df):
            """Data cleaning"""
            print("Cleaning data...")
            df = detect_duplicates(df)
            df = impute_missing_values(df, method='mean')
            return df
    
        def save_data(self, df, filename=None):
            """Save data"""
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"materials_data_{timestamp}.csv"
    
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Data saved: {filepath}")
    
            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_records": len(df),
                "columns": list(df.columns),
                "file": filename
            }
    
            meta_file = filepath.replace(".csv", "_meta.json")
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
            return filepath
    
        def run(self, formulas):
            """Execute pipeline"""
            # Fetch data
            df = self.fetch_data(formulas)
    
            # Clean
            df_clean = self.clean_data(df)
    
            # Save
            filepath = self.save_data(df_clean)
    
            print(f"\n=== Pipeline Complete ===")
            print(f"Records Fetched: {len(df_clean)}")
            print(f"File: {filepath}")
    
            return df_clean
    
    # Usage example
    pipeline = DataUpdatePipeline()
    formulas = ["TiO2", "ZnO", "GaN", "SiC", "Al2O3"]
    df_result = pipeline.run(formulas)
    

### 3.4.2 Scheduled Execution

**Code Example 8: Cron-Style Periodic Execution**
    
    
    import schedule
    import time
    
    def scheduled_update():
        """Scheduled update job"""
        print(f"\n{'='*50}")
        print(f"Starting scheduled update: {datetime.now()}")
        print(f"{'='*50}")
    
        # Execute pipeline
        pipeline = DataUpdatePipeline()
        formulas = ["TiO2", "ZnO", "GaN", "SiC"]
        pipeline.run(formulas)
    
    # Schedule setup
    schedule.every().day.at("09:00").do(scheduled_update)
    schedule.every().week.do(scheduled_update)
    
    # Demo: every 10 seconds
    schedule.every(10).seconds.do(scheduled_update)
    
    print("Scheduler started...")
    print("Press Ctrl+C to exit")
    
    # Infinite loop
    while True:
        schedule.run_pending()
        time.sleep(1)
    

* * *

## 3.5 Data Quality Management

### 3.5.1 Quality Metrics

**Code Example 9: Data Quality Assessment**
    
    
    def calculate_quality_metrics(df):
        """Calculate data quality metrics"""
    
        metrics = {}
    
        # Completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (
            (total_cells - missing_cells) / total_cells
        ) * 100
    
        # Consistency
        # Example: check if band gap is non-negative
        if 'band_gap_mp' in df.columns:
            invalid_bg = (df['band_gap_mp'] < 0).sum()
            consistency = (
                (len(df) - invalid_bg) / len(df)
            ) * 100
        else:
            consistency = 100.0
    
        # Accuracy
        # Example: discrepancy between MP and AFLOW
        if 'band_gap_mp' in df.columns and \
           'band_gap_aflow' in df.columns:
            diff = (
                df['band_gap_mp'] - df['band_gap_aflow']
            ).abs()
            avg_diff = diff.mean()
            accuracy = max(
                0, 100 - (avg_diff / df['band_gap_mp'].mean()) * 100
            )
        else:
            accuracy = None
    
        metrics = {
            "Completeness (%)": round(completeness, 2),
            "Consistency (%)": round(consistency, 2),
            "Accuracy (%)": round(accuracy, 2) if accuracy else "N/A",
            "Total Records": len(df),
            "Total Cells": total_cells,
            "Missing Cells": missing_cells
        }
    
        print("=== Data Quality Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
        return metrics
    
    # Quality assessment
    quality = calculate_quality_metrics(df_clean)
    

### 3.5.2 Validation Rules

**Code Example 10: Data Validation**
    
    
    def validate_materials_data(df):
        """Validate materials data"""
    
        validation_report = []
    
        # Rule 1: Band gap must be non-negative
        if 'band_gap_mp' in df.columns:
            invalid_bg = df[df['band_gap_mp'] < 0]
            if len(invalid_bg) > 0:
                validation_report.append({
                    "Rule": "Band Gap >= 0",
                    "Status": "FAIL",
                    "Failed Records": len(invalid_bg)
                })
            else:
                validation_report.append({
                    "Rule": "Band Gap >= 0",
                    "Status": "PASS",
                    "Failed Records": 0
                })
    
        # Rule 2: Formation energy should be negative
        if 'formation_energy_mp' in df.columns:
            invalid_fe = df[df['formation_energy_mp'] > 0]
            if len(invalid_fe) > 0:
                validation_report.append({
                    "Rule": "Formation Energy <= 0",
                    "Status": "WARNING",
                    "Failed Records": len(invalid_fe)
                })
            else:
                validation_report.append({
                    "Rule": "Formation Energy <= 0",
                    "Status": "PASS",
                    "Failed Records": 0
                })
    
        # Rule 3: Stability <= 0.1 eV/atom
        if 'stability_mp' in df.columns:
            unstable = df[df['stability_mp'] > 0.1]
            validation_report.append({
                "Rule": "Stability <= 0.1 eV/atom",
                "Status": "INFO",
                "Failed Records": len(unstable)
            })
    
        # Display report
        print("=== Validation Report ===")
        report_df = pd.DataFrame(validation_report)
        print(report_df)
    
        return report_df
    
    # Execute validation
    validation = validate_materials_data(df_clean)
    

* * *

## 3.6 Practical Case Studies

### 3.6.1 Battery Materials Discovery

**Code Example 11: Integrated Search for Li-ion Battery Cathode Materials**
    
    
    def find_battery_materials():
        """Search for battery materials from multiple databases"""
    
        # Search from Materials Project
        with MPRester(MP_API_KEY) as mpr:
            mp_docs = mpr.materials.summary.search(
                elements=["Li", "Co", "O"],
                energy_above_hull=(0, 0.05),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "energy_above_hull",
                    "formation_energy_per_atom"
                ]
            )
    
        mp_data = [{
            "source": "MP",
            "formula": doc.formula_pretty,
            "stability": doc.energy_above_hull,
            "formation_energy": doc.formation_energy_per_atom
        } for doc in mp_docs]
    
        # AFLOW (simplified version)
        # Actual AFLOW API call omitted
    
        # Integration
        df_battery = pd.DataFrame(mp_data)
    
        # Sort by stability
        df_battery = df_battery.sort_values('stability')
    
        print("=== Li-Co-O Battery Material Candidates ===")
        print(df_battery.head(10))
    
        return df_battery
    
    # Execute
    df_battery = find_battery_materials()
    

### 3.6.2 Catalyst Materials Screening

**Code Example 12: Integrated Screening of Transition Metal Oxide Catalysts**
    
    
    def screen_catalysts(
        transition_metals=["Ti", "V", "Cr", "Mn", "Fe"]
    ):
        """Integrated screening of catalyst materials"""
    
        all_results = []
    
        for tm in transition_metals:
            # Fetch from Materials Project
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(
                    elements=[tm, "O"],
                    band_gap=(0.1, 3.0),
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
                        "transition_metal": tm,
                        "formula": doc.formula_pretty,
                        "band_gap": doc.band_gap,
                        "stability": doc.energy_above_hull,
                        "source": "MP"
                    })
    
        df_catalysts = pd.DataFrame(all_results)
    
        # Data cleaning
        df_catalysts = df_catalysts.drop_duplicates(
            subset=['formula']
        )
    
        # Filter by band gap range
        df_ideal = df_catalysts[
            (df_catalysts['band_gap'] >= 1.5) &
            (df_catalysts['band_gap'] <= 2.5)
        ]
    
        print(f"=== Catalyst Candidate Materials ===")
        print(f"Total Candidates: {len(df_catalysts)} materials")
        print(f"Ideal Band Gap: {len(df_ideal)} materials")
        print("\nTop 10:")
        print(df_ideal.head(10))
    
        return df_ideal
    
    # Execute
    df_catalysts = screen_catalysts()
    

* * *

## 3.7 Workflow Visualization
    
    
    ```mermaid
    flowchart TD
        A[Data Acquisition] --> B[Materials Project]
        A --> C[AFLOW]
        A --> D[OQMD]
    
        B --> E[Data Merge]
        C --> E
        D --> E
    
        E --> F[Duplicate Removal]
        F --> G[Outlier Detection]
        G --> H[Missing Value Handling]
    
        H --> I[Quality Assessment]
        I --> J{Quality OK?}
    
        J -->|Yes| K[Save Data]
        J -->|No| L[Reprocess]
        L --> E
    
        K --> M[Periodic Update]
        M --> A
    
        style A fill:#e3f2fd
        style K fill:#e8f5e9
        style J fill:#fff3e0
    ```

* * *

## 3.8 Chapter Summary

### What You Learned

  1. **Database Integration** \- Integrating Materials Project and AFLOW \- Merging based on chemical formula keys \- Data integration using outer join

  2. **Data Cleaning** \- Detecting and removing duplicate data \- Outlier detection using IQR method \- Data type normalization

  3. **Missing Value Handling** \- Visualizing missing patterns \- Mean/median imputation \- KNN imputation

  4. **Automation Pipeline** \- Data fetch → clean → save pipeline \- Scheduled execution \- Metadata management

  5. **Quality Management** \- Assessing completeness, consistency, and accuracy \- Validation rules \- Automated quality report generation

### Key Takeaways

  * ✅ Multi-database integration improves reliability
  * ✅ Data cleaning is an essential process
  * ✅ Choose missing value handling based on use case
  * ✅ Automation enables continuous data updates
  * ✅ Quality metrics provide objective evaluation

### Next Chapter

In Chapter 4, you'll learn how to build custom databases: \- SQLite/PostgreSQL design \- Schema design \- CRUD operations \- Backup strategies

**[Chapter 4: Building Custom Databases →](<chapter-4.html>)**

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Fetch data for the same material (SiC) from both Materials Project and AFLOW, and compare the band gap and formation energy.

**Requirements** : 1\. Fetch SiC data from both databases 2\. Display band gap difference as percentage 3\. Display formation energy difference as percentage

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Requirements:
    1. Fetch SiC data from both databases
    2. Displ
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from mp_api.client import MPRester
    import requests
    
    MP_API_KEY = "your_api_key"
    
    # Fetch from Materials Project
    with MPRester(MP_API_KEY) as mpr:
        mp_docs = mpr.materials.summary.search(
            formula="SiC",
            fields=["band_gap", "formation_energy_per_atom"]
        )
        mp_bg = mp_docs[0].band_gap
        mp_fe = mp_docs[0].formation_energy_per_atom
    
    # Fetch from AFLOW (simplified)
    aflow_url = "http://aflowlib.org/API/aflux"
    params = {
        "species": "Si,C",
        "$": "Egap,enthalpy_formation_atom"
    }
    response = requests.get(aflow_url, params=params)
    aflow_data = response.json()[0]
    aflow_bg = aflow_data.get("Egap")
    aflow_fe = aflow_data.get("enthalpy_formation_atom")
    
    # Comparison
    bg_diff = abs(mp_bg - aflow_bg) / mp_bg * 100
    fe_diff = abs(mp_fe - aflow_fe) / abs(mp_fe) * 100
    
    print(f"SiC Data Comparison:")
    print(f"Band Gap: MP={mp_bg} eV, AFLOW={aflow_bg} eV")
    print(f"Difference: {bg_diff:.1f}%")
    print(f"\nFormation Energy: MP={mp_fe} eV/atom, "
          f"AFLOW={aflow_fe} eV/atom")
    print(f"Difference: {fe_diff:.1f}%")
    

* * *

### Exercise 2 (Difficulty: Medium)

Perform missing value handling on the following dataset.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Perform missing value handling on the following dataset.
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    # Sample data
    data = {
        'material_id': ['mp-1', 'mp-2', 'mp-3', 'mp-4', 'mp-5'],
        'formula': ['TiO2', 'ZnO', 'GaN', 'SiC', 'Al2O3'],
        'band_gap': [3.44, np.nan, 3.20, 2.36, np.nan],
        'formation_energy': [-4.87, -1.95, np.nan, -0.67, -3.45]
    }
    df = pd.DataFrame(data)
    

**Requirements** : 1\. Impute missing values with mean 2\. Impute missing values using KNN method 3\. Compare results from both methods

Solution
    
    
    from sklearn.impute import SimpleImputer, KNNImputer
    
    # Mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    df_mean = df.copy()
    df_mean[['band_gap', 'formation_energy']] = \
        imputer_mean.fit_transform(
            df[['band_gap', 'formation_energy']]
        )
    
    # KNN imputation
    imputer_knn = KNNImputer(n_neighbors=2)
    df_knn = df.copy()
    df_knn[['band_gap', 'formation_energy']] = \
        imputer_knn.fit_transform(
            df[['band_gap', 'formation_energy']]
        )
    
    # Comparison
    print("Original Data:")
    print(df)
    print("\nMean Imputation:")
    print(df_mean)
    print("\nKNN Imputation:")
    print(df_knn)
    
    # Difference analysis
    diff_bg = (
        df_mean['band_gap'] - df_knn['band_gap']
    ).abs().mean()
    diff_fe = (
        df_mean['formation_energy'] -
        df_knn['formation_energy']
    ).abs().mean()
    
    print(f"\nAverage Differences:")
    print(f"Band Gap: {diff_bg:.3f} eV")
    print(f"Formation Energy: {diff_fe:.3f} eV/atom")
    

* * *

### Exercise 3 (Difficulty: Hard)

Fetch 100+ materials from Materials Project and AFLOW, then build a complete pipeline for integration, cleaning, and quality assessment.

**Requirements** : 1\. Fetch 100+ oxide (O-containing) materials 2\. Data integration (outer join) 3\. Duplicate removal, outlier detection 4\. Missing value imputation 5\. Calculate quality metrics 6\. Save results as CSV and JSON

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - requests>=2.31.0
    
    from mp_api.client import MPRester
    import requests
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    import json
    from datetime import datetime
    
    MP_API_KEY = "your_api_key"
    
    class IntegratedDataPipeline:
        """Integrated data pipeline"""
    
        def __init__(self):
            self.df = None
    
        def fetch_mp_data(self, num_materials=100):
            """Fetch data from Materials Project"""
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(
                    elements=["O"],
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "band_gap",
                        "formation_energy_per_atom"
                    ]
                )[:num_materials]
    
                return pd.DataFrame([{
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap_mp": doc.band_gap,
                    "formation_energy_mp":
                        doc.formation_energy_per_atom,
                    "source": "MP"
                } for doc in docs])
    
        def merge_data(self, df_mp, df_aflow=None):
            """Data integration"""
            # Simplified: AFLOW data omitted
            return df_mp
    
        def clean_data(self, df):
            """Data cleaning"""
            # Remove duplicates
            df = df.drop_duplicates(subset=['formula'])
    
            # Outlier detection (IQR method)
            Q1 = df['band_gap_mp'].quantile(0.25)
            Q3 = df['band_gap_mp'].quantile(0.75)
            IQR = Q3 - Q1
            df = df[
                (df['band_gap_mp'] >= Q1 - 1.5 * IQR) &
                (df['band_gap_mp'] <= Q3 + 1.5 * IQR)
            ]
    
            return df
    
        def impute_missing(self, df):
            """Missing value imputation"""
            numeric_cols = [
                'band_gap_mp', 'formation_energy_mp'
            ]
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(
                df[numeric_cols]
            )
            return df
    
        def calculate_quality(self, df):
            """Quality metrics"""
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            completeness = (
                (total_cells - missing_cells) / total_cells
            ) * 100
    
            return {
                "completeness": completeness,
                "num_records": len(df),
                "timestamp": datetime.now().isoformat()
            }
    
        def save_results(self, df, quality):
            """Save results"""
            # Save CSV
            df.to_csv("integrated_data.csv", index=False)
    
            # Save metadata
            with open("quality_report.json", 'w') as f:
                json.dump(quality, f, indent=2)
    
            print("Data saved:")
            print("- integrated_data.csv")
            print("- quality_report.json")
    
        def run(self):
            """Execute pipeline"""
            print("=== Integrated Data Pipeline ===")
    
            # Fetch data
            print("1. Fetching data...")
            df_mp = self.fetch_mp_data(100)
    
            # Integration
            print("2. Integrating data...")
            df = self.merge_data(df_mp)
    
            # Cleaning
            print("3. Cleaning data...")
            df = self.clean_data(df)
    
            # Imputation
            print("4. Imputing missing values...")
            df = self.impute_missing(df)
    
            # Quality assessment
            print("5. Assessing quality...")
            quality = self.calculate_quality(df)
    
            # Save
            print("6. Saving results...")
            self.save_results(df, quality)
    
            print("\n=== Complete ===")
            print(f"Total Records: {len(df)}")
            print(f"Quality Score: {quality['completeness']:.2f}%")
    
            self.df = df
            return df, quality
    
    # Execute
    pipeline = IntegratedDataPipeline()
    df_result, quality_metrics = pipeline.run()
    

* * *

## References

  1. Wilkinson, M. D. et al. (2016). "The FAIR Guiding Principles for scientific data management and stewardship." _Scientific Data_ , 3, 160018. DOI: [10.1038/sdata.2016.18](<https://doi.org/10.1038/sdata.2016.18>)

  2. pandas Documentation. "Merge, join, concatenate and compare." URL: [pandas.pydata.org/docs](<https://pandas.pydata.org/docs>)

* * *

## Navigation

### Previous Chapter

**[Chapter 2: Materials Project Complete Guide ←](<chapter-2.html>)**

### Next Chapter

**[Chapter 4: Building Custom Databases →](<chapter-4.html>)**

### Series Table of Contents

**[← Back to Series Contents](<./index.html>)**

* * *

## Author Information

**Created by** : AI Terakoya Content Team **Date Created** : 2025-10-17 **Version** : 1.0

**License** : Creative Commons BY 4.0

* * *

**Continue your learning journey in the next chapter!**
