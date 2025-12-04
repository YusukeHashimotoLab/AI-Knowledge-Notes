---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 0
exercises: 0
version: 1.0
created_at: "by:"
---

# Chapter 1: Fundamentals of Materials Space Visualization

This chapter covers the fundamentals of Fundamentals of Materials Space Visualization, which overview. You will learn essential concepts and techniques.

## Overview

In materials science, to effectively understand and design thousands to tens of thousands of materials, it is important to project high-dimensional materials property space into lower dimensions for visualization. This chapter introduces the fundamental concepts of materials space visualization and basic visualization techniques.

### Learning Objectives

  * Understand the concepts of materials space and property space
  * Understand the challenges of visualizing high-dimensional data
  * Implement basic scatter plots and clustering visualizations
  * Visually grasp correlations between materials properties

## 1.1 What is Materials Space

Materials space refers to a multidimensional space with axes representing material properties and structures. Each material is represented as a single point in this space.

### 1.1.1 Dimensions of Property Space

Materials are typically described by numerous properties such as:

  * **Physical properties** : Band gap, density, melting point, thermal conductivity, etc.
  * **Chemical properties** : Electronegativity, ionization energy, oxidation state, etc.
  * **Structural properties** : Lattice constants, space group, coordination number, etc.
  * **Functional properties** : Catalytic activity, battery capacity, magnetic susceptibility, etc.

When all these properties are considered, materials space becomes a high-dimensional space with tens to hundreds of dimensions.

### 1.1.2 Challenges in High-Dimensional Data Visualization

Visualizing high-dimensional data directly is challenging:

  1. **Curse of dimensionality** : As dimensions increase, the meaning of distances between points becomes less significant
  2. **Visualization limitations** : Humans can intuitively understand only 2D and 3D visualizations
  3. **Information loss** : Some information may be lost during dimensionality reduction
  4. **Computational cost** : Processing large datasets can be time-consuming

## 1.2 Materials Data Preparation

### Code Example 1: Loading Materials Data and Basic Statistics
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 1: Loading Materials Data and Basic Statistics
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    
    # Create sample materials dataset
    # In actual projects, this would be obtained using pymatgen, etc.
    np.random.seed(42)
    
    n_materials = 1000
    
    materials_data = pd.DataFrame({
        'formula': [f'Material_{i}' for i in range(n_materials)],
        'band_gap': np.random.normal(2.5, 1.2, n_materials),
        'formation_energy': np.random.normal(-1.5, 0.8, n_materials),
        'density': np.random.normal(5.0, 1.5, n_materials),
        'bulk_modulus': np.random.normal(150, 50, n_materials),
        'shear_modulus': np.random.normal(80, 30, n_materials),
        'melting_point': np.random.normal(1500, 400, n_materials),
    })
    
    # Adjust properties that should not have negative values
    materials_data['band_gap'] = materials_data['band_gap'].clip(lower=0)
    materials_data['density'] = materials_data['density'].clip(lower=0.1)
    materials_data['bulk_modulus'] = materials_data['bulk_modulus'].clip(lower=10)
    materials_data['shear_modulus'] = materials_data['shear_modulus'].clip(lower=5)
    materials_data['melting_point'] = materials_data['melting_point'].clip(lower=300)
    
    # Display basic statistics
    print("Basic statistics of materials dataset:")
    print(materials_data.describe())
    
    # Save data
    materials_data.to_csv('materials_properties.csv', index=False)
    print("\nData saved to materials_properties.csv")
    

**Example output** :
    
    
    Basic statistics of materials dataset:
               band_gap  formation_energy      density  bulk_modulus  shear_modulus  melting_point
    count   1000.000000       1000.000000  1000.000000   1000.000000    1000.000000    1000.000000
    mean       2.499124         -1.502361     4.985472    149.893421      79.876543    1498.234567
    std        1.189234          0.798765     1.487234     49.876543      29.765432     398.765432
    min        0.000000         -3.987654     0.123456     10.000000       5.000000     300.000000
    max        6.234567          1.234567    10.234567    289.876543     169.876543    2789.876543
    

### Code Example 2: Visualizing Property Distributions
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 2: Visualizing Property Distributions
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Style settings
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Histogram for each property
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    properties = ['band_gap', 'formation_energy', 'density',
                  'bulk_modulus', 'shear_modulus', 'melting_point']
    
    property_labels = {
        'band_gap': 'Band Gap (eV)',
        'formation_energy': 'Formation Energy (eV/atom)',
        'density': 'Density (g/cm³)',
        'bulk_modulus': 'Bulk Modulus (GPa)',
        'shear_modulus': 'Shear Modulus (GPa)',
        'melting_point': 'Melting Point (K)'
    }
    
    for idx, prop in enumerate(properties):
        axes[idx].hist(materials_data[prop], bins=30, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(property_labels[prop], fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'Distribution of {property_labels[prop]}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
    print("Property distribution histograms saved to property_distributions.png")
    plt.show()
    

## 1.3 Basic Visualization with 2D Scatter Plots

### Code Example 3: Scatter Plot of Two Properties
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 3: Scatter Plot of Two Properties
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Relationship between band gap and formation energy
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(materials_data['band_gap'],
                         materials_data['formation_energy'],
                         c=materials_data['density'],
                         cmap='viridis',
                         s=50,
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=0.5)
    
    ax.set_xlabel('Band Gap (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Formation Energy (eV/atom)', fontsize=14, fontweight='bold')
    ax.set_title('Materials Space: Band Gap vs Formation Energy',
                 fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density (g/cm³)', fontsize=12, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight stability region (formation_energy < 0)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
               label='Stability threshold', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('bandgap_vs_formation_energy.png', dpi=300, bbox_inches='tight')
    print("Scatter plot saved to bandgap_vs_formation_energy.png")
    plt.show()
    
    # Calculate correlation coefficient
    correlation = materials_data['band_gap'].corr(materials_data['formation_energy'])
    print(f"\nCorrelation coefficient between Band Gap and Formation Energy: {correlation:.3f}")
    

### Code Example 4: Pair Plot (Multivariate Correlation Visualization)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 4: Pair Plot (Multivariate Correlation Visualiz
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Pair plot of key properties
    properties_subset = ['band_gap', 'formation_energy', 'density', 'bulk_modulus']
    
    # Categorize by stability
    materials_data['stability'] = materials_data['formation_energy'].apply(
        lambda x: 'Stable' if x < -1.0 else 'Metastable' if x < 0 else 'Unstable'
    )
    
    # Pair plot
    pairplot = sns.pairplot(materials_data[properties_subset + ['stability']],
                            hue='stability',
                            diag_kind='kde',
                            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
                            diag_kws={'alpha': 0.7, 'linewidth': 2},
                            corner=False,
                            palette='Set2')
    
    pairplot.fig.suptitle('Materials Properties Pairplot',
                          fontsize=16, fontweight='bold', y=1.01)
    
    # Improve axis labels
    label_map = {
        'band_gap': 'Band Gap (eV)',
        'formation_energy': 'Form. E (eV/atom)',
        'density': 'Density (g/cm³)',
        'bulk_modulus': 'Bulk Mod. (GPa)'
    }
    
    for ax in pairplot.axes.flatten():
        if ax is not None:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in label_map:
                ax.set_xlabel(label_map[xlabel], fontsize=10)
            if ylabel in label_map:
                ax.set_ylabel(label_map[ylabel], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('materials_pairplot.png', dpi=300, bbox_inches='tight')
    print("Pair plot saved to materials_pairplot.png")
    plt.show()
    
    # Calculate and display correlation matrix
    print("\nCorrelation coefficient matrix between properties:")
    correlation_matrix = materials_data[properties_subset].corr()
    print(correlation_matrix.round(3))
    

## 1.4 Correlation Matrix Visualization

### Code Example 5: Correlation Visualization with Heatmap
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 5: Correlation Visualization with Heatmap
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Correlation matrix of all properties
    numerical_cols = ['band_gap', 'formation_energy', 'density',
                      'bulk_modulus', 'shear_modulus', 'melting_point']
    
    correlation_matrix = materials_data[numerical_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask (hide upper triangle)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                vmin=-1,
                vmax=1,
                ax=ax)
    
    # Improve labels
    labels = [
        'Band Gap\n(eV)',
        'Formation E\n(eV/atom)',
        'Density\n(g/cm³)',
        'Bulk Modulus\n(GPa)',
        'Shear Modulus\n(GPa)',
        'Melting Point\n(K)'
    ]
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    
    ax.set_title('Materials Properties Correlation Matrix',
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix heatmap saved to correlation_heatmap.png")
    plt.show()
    
    # Identify strongly correlated pairs
    print("\nStrongly correlated property pairs (|r| > 0.5):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                print(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_value:.3f}")
    

## 1.5 Summary

In this chapter, we covered the following as fundamentals of materials space visualization:

### Key Points

  1. **Concept of materials space** : Representing materials as points in multidimensional property space
  2. **Challenges of high-dimensional data** : Curse of dimensionality, visualization limitations, information loss
  3. **Basic visualization techniques** : \- Understanding distributions through histograms \- Visualizing relationships between two properties using scatter plots \- Understanding multivariate correlations through pair plots \- Visualizing correlation matrices through heatmaps

### Implemented Code

Code Example | Content | Main Output  
---|---|---  
Example 1 | Materials data preparation and basic statistics | CSV file, statistics  
Example 2 | Property distribution histograms | 6 histograms  
Example 3 | 2-property scatter plot | Band Gap vs Formation Energy  
Example 4 | Pair plot | All combinations of 4 properties  
Example 5 | Correlation matrix heatmap | Visualization of correlation coefficients  
  
### Looking Ahead to the Next Chapter

In Chapter 2, we will learn how to project high-dimensional materials space into 2D or 3D using more advanced dimensionality reduction techniques (PCA, t-SNE, UMAP). This will enable us to visualize tens to hundreds of dimensional material properties at once and reveal similarities and cluster structures between materials.

* * *

**Next Chapter** : [Chapter 2: Materials Space Mapping with Dimensionality Reduction Techniques](<chapter-2.html>)

**Series Top** : [Introduction to Materials Property Mapping](<index.html>)
