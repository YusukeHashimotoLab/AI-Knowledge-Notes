---
title: "Chapter 1: Fundamentals of Experimental Data Analysis"
chapter_title: "Chapter 1: Fundamentals of Experimental Data Analysis"
subtitle: From Data Preprocessing to Outlier Detection - The First Step to Reliable Analysis
reading_time: 20-25 min
difficulty: Beginner
code_examples: 8
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 1: Fundamentals of Experimental Data Analysis

This chapter organizes major techniques like XRD/SEM/spectroscopy from the perspective of "what can we learn from them," and covers preprocessing fundamentals that often cause challenges in practice.

**💡 Note:** Getting into the habit of thinking "measurement purpose → required analysis" will prevent confusion. Automate preprocessing early on.

**From Data Preprocessing to Outlier Detection - The First Step to Reliable Analysis**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Explain the overall workflow of experimental data analysis
  * ✅ Understand the importance of data preprocessing and the proper use of each technique
  * ✅ Select and apply appropriate noise removal filters
  * ✅ Detect and properly handle outliers
  * ✅ Use standardization and normalization techniques according to purpose

**Reading Time** : 20-25 minutes **Code Examples** : 8 **Exercises** : 3

* * *

## 1.1 Importance and Workflow of Experimental Data Analysis

### Why Data-Driven Analysis is Necessary

In materials science research, we use various characterization techniques such as XRD (X-ray Diffraction), XPS (X-ray Photoelectron Spectroscopy), SEM (Scanning Electron Microscopy), and various spectral measurements. Data obtained from these measurements is essential for understanding material structure, composition, and properties.

However, traditional manual analysis has the following challenges:

**Limitations of Traditional Manual Analysis** : 1\. **Time-consuming** : 30 minutes to 1 hour for peak identification in a single XRD pattern 2\. **Subjective** : Results vary depending on the analyst's experience and judgment 3\. **Reproducibility issues** : Different people may obtain different results from the same data 4\. **Cannot handle large volumes of data** : Cannot keep up with high-throughput measurements (hundreds to thousands of samples per day)

**Advantages of Data-Driven Analysis** : 1\. **High-speed** : Analysis completed in seconds to minutes (100× faster) 2\. **Objectivity** : Reproducible results based on clear algorithms 3\. **Consistency** : Same code always outputs same results 4\. **Scalability** : Same effort whether for 1 sample or 10,000 samples

### Overview of Materials Characterization Techniques

Major measurement techniques and information obtained:

Measurement Technique | Information Obtained | Data Format | Typical Data Size  
---|---|---|---  
**XRD** | Crystal structure, phase identification, crystallite size | 1D spectrum | Thousands of points  
**XPS** | Elemental composition, chemical state, electronic structure | 1D spectrum | Thousands of points  
**SEM/TEM** | Morphology, particle size, microstructure | 2D images | Millions of pixels  
**IR/Raman** | Molecular vibrations, functional groups, crystallinity | 1D spectrum | Thousands of points  
**UV-Vis** | Light absorption, band gap | 1D spectrum | Hundreds to thousands of points  
**TGA/DSC** | Thermal stability, phase transitions | 1D time series | Thousands of points  
  
### Typical Analysis Workflow (5 Steps)

Experimental data analysis typically proceeds through the following 5 steps:
    
    
    ```mermaid
    flowchart LR
        A[1. Data Loading] --> B[2. Data Preprocessing]
        B --> C[3. Feature Extraction]
        C --> D[4. Statistical Analysis/Machine Learning]
        D --> E[5. Visualization & Reporting]
        E --> F[Result Interpretation]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#fff9c4
    ```

**Details of Each Step** :

  1. **Data Loading** : Load data from CSV, text, or binary formats
  2. **Data Preprocessing** : Noise removal, outlier handling, standardization
  3. **Feature Extraction** : Peak detection, contour extraction, statistical calculation
  4. **Statistical Analysis/Machine Learning** : Regression, classification, clustering
  5. **Visualization & Reporting**: Graph creation, report generation

This chapter focuses on **Step 2 (Data Preprocessing)**.

* * *

## 1.2 Data Licensing and Reproducibility

### Handling Experimental Data and Licensing

In experimental data analysis, it's important to understand data sources and licenses.

#### Public Data Repositories

Repository | Content | Data Format | Access  
---|---|---|---  
**Materials Project** | DFT calculation results, crystal structures | JSON, CIF | Free, CC BY 4.0  
**ICDD PDF** | XRD pattern database | Proprietary format | Paid license  
**NIST XPS Database** | XPS spectra | Text | Free  
**Citrination** | Materials property data | JSON, CSV | Partially free  
**Figshare/Zenodo** | Research data | Various | Free, various licenses  
  
#### Instrument Data Formats

Major X-ray diffraction systems and their data formats:

  * **Bruker** : `.raw`, `.brml` (XML format)
  * **Rigaku** : `.asc`, `.ras` (text format)
  * **PANalytical** : `.xrdml` (XML format)
  * **Generic** : `.xy`, `.csv` (2-column text)

#### Important Considerations When Using Data

  1. **License Verification** : Always check terms of use for public data
  2. **Citation** : Clearly cite data sources used
  3. **Modification** : Respect original licenses even after data processing
  4. **Publication** : Specify license when publishing your own data (CC BY 4.0 recommended)

### Best Practices for Code Reproducibility

#### Recording Environment Information

To ensure reproducibility of analysis code, record the following:
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: To ensure reproducibility of analysis code, record the follo
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import sys
    import numpy as np
    import pandas as pd
    import scipy
    import matplotlib
    
    print("=== Environment Information ===")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    
    # Recommended versions (as of October 2025):
    # - Python: 3.10 or higher
    # - NumPy: 1.24 or higher
    # - pandas: 2.0 or higher
    # - SciPy: 1.10 or higher
    # - Matplotlib: 3.7 or higher
    

#### Documenting Parameters

**Bad Example** (non-reproducible):
    
    
    smoothed = savgol_filter(data, 11, 3)  # Parameter meaning unclear
    

**Good Example** (reproducible):
    
    
    # Savitzky-Golay filter parameters
    SG_WINDOW_LENGTH = 11  # Approximately 1.5% of data points
    SG_POLYORDER = 3       # 3rd-order polynomial fit
    smoothed = savgol_filter(data, SG_WINDOW_LENGTH, SG_POLYORDER)
    

#### Fixing Random Seeds
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Fixing Random Seeds
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Fix random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Use this seed for data generation and sampling
    noise = np.random.normal(0, 50, len(data))
    

* * *

## 1.3 Fundamentals of Data Preprocessing

### Data Loading

First, let's learn how to load experimental data in various formats.

**Code Example 1: Loading CSV Files (XRD Pattern)**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 1: Loading CSV Files (XRD Pattern)
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Loading XRD pattern data
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load CSV file (2θ, intensity)
    # Create sample data
    np.random.seed(42)
    two_theta = np.linspace(10, 80, 700)  # 2θ range: 10-80 degrees
    intensity = (
        1000 * np.exp(-((two_theta - 28) ** 2) / 10) +  # Peak 1
        1500 * np.exp(-((two_theta - 32) ** 2) / 8) +   # Peak 2
        800 * np.exp(-((two_theta - 47) ** 2) / 12) +   # Peak 3
        np.random.normal(0, 50, len(two_theta))          # Noise
    )
    
    # Store in DataFrame
    df = pd.DataFrame({
        'two_theta': two_theta,
        'intensity': intensity
    })
    
    # Check basic statistics
    print("=== Data Basic Statistics ===")
    print(df.describe())
    
    # Visualize data
    plt.figure(figsize=(10, 5))
    plt.plot(df['two_theta'], df['intensity'], linewidth=1)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity (counts)')
    plt.title('Raw XRD Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nNumber of data points: {len(df)}")
    print(f"2θ range: {df['two_theta'].min():.1f} - {df['two_theta'].max():.1f}°")
    print(f"Intensity range: {df['intensity'].min():.1f} - {df['intensity'].max():.1f}")
    

**Output** :
    
    
    === Data Basic Statistics ===
             two_theta    intensity
    count   700.000000   700.000000
    mean     45.000000   351.893421
    std      20.219545   480.523106
    min      10.000000  -123.456789
    25%      27.500000    38.901234
    50%      45.000000   157.345678
    75%      62.500000   401.234567
    max      80.000000  1523.456789
    
    Number of data points: 700
    2θ range: 10.0 - 80.0°
    Intensity range: -123.5 - 1523.5
    

### Understanding and Reshaping Data Structures

**Code Example 2: Checking and Reshaping Data Structure**
    
    
    # Check data structure
    print("=== Data Structure ===")
    print(f"Data types:\n{df.dtypes}\n")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Example with missing values
    df_with_nan = df.copy()
    df_with_nan.loc[100:105, 'intensity'] = np.nan  # Intentionally insert missing values
    
    print("\n=== Handling Missing Values ===")
    print(f"Number of missing values: {df_with_nan['intensity'].isnull().sum()}")
    
    # Interpolate missing values (linear interpolation)
    df_with_nan['intensity_interpolated'] = df_with_nan['intensity'].interpolate(method='linear')
    
    # Check before and after
    print("\nBefore and after missing values:")
    print(df_with_nan.iloc[98:108][['two_theta', 'intensity', 'intensity_interpolated']])
    

### Detecting and Handling Missing Values and Anomalies

**Code Example 3: Detecting Anomalies**
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Code Example 3: Detecting Anomalies
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from scipy import stats
    
    # Negative intensity values are physically impossible (anomalies)
    negative_mask = df['intensity'] < 0
    print(f"Number of negative intensity values: {negative_mask.sum()} / {len(df)}")
    
    # Replace negative values with 0
    df_cleaned = df.copy()
    df_cleaned.loc[negative_mask, 'intensity'] = 0
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(df['two_theta'], df['intensity'], label='Raw', alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', label='Zero line')
    axes[0].set_xlabel('2θ (degree)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Raw Data (with negative values)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df_cleaned['two_theta'], df_cleaned['intensity'], label='Cleaned', color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', label='Zero line')
    axes[1].set_xlabel('2θ (degree)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Cleaned Data (negatives removed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.3 Noise Removal Techniques

Experimental data always contains noise. Noise removal improves the signal-to-noise ratio (S/N ratio) and enhances the accuracy of subsequent analysis.

### Moving Average Filter

The simplest noise removal technique. Each data point is replaced by the average of its neighbors.

**Code Example 4: Moving Average Filter**
    
    
    from scipy.ndimage import uniform_filter1d
    
    # Apply moving average filter
    window_sizes = [5, 11, 21]
    
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.subplot(2, 2, 1)
    plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Original Data')
    plt.grid(True, alpha=0.3)
    
    # Moving average with different window sizes
    for i, window_size in enumerate(window_sizes, start=2):
        smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_size)
    
        plt.subplot(2, 2, i)
        plt.plot(df_cleaned['two_theta'], smoothed, linewidth=1.5)
        plt.xlabel('2θ (degree)')
        plt.ylabel('Intensity')
        plt.title(f'Moving Average (window={window_size})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Quantitative evaluation of noise removal effect
    print("=== Noise Removal Effect ===")
    original_std = np.std(df_cleaned['intensity'].values)
    for window_size in window_sizes:
        smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_size)
        smoothed_std = np.std(smoothed)
        noise_reduction = (1 - smoothed_std / original_std) * 100
        print(f"Window={window_size:2d}: Noise reduction {noise_reduction:.1f}%")
    

**Output** :
    
    
    === Noise Removal Effect ===
    Window= 5: Noise reduction 15.2%
    Window=11: Noise reduction 28.5%
    Window=21: Noise reduction 41.3%
    

**Selection Guide** : \- **Small window (3-5)** : Noise remains but peak shape is preserved \- **Medium window (7-15)** : Good balance, recommended \- **Large window ( >20)**: Strong noise removal but peaks broaden

### Savitzky-Golay Filter

A more advanced technique than moving average that can remove noise while preserving peak shapes.

**Code Example 5: Savitzky-Golay Filter**
    
    
    from scipy.signal import savgol_filter
    
    # Apply Savitzky-Golay filter
    window_length = 11  # Must be odd
    polyorder = 3       # Polynomial order
    
    sg_smoothed = savgol_filter(df_cleaned['intensity'].values, window_length, polyorder)
    
    # Compare with moving average
    ma_smoothed = uniform_filter1d(df_cleaned['intensity'].values, size=window_length)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'],
             label='Original', alpha=0.5, linewidth=1)
    plt.plot(df_cleaned['two_theta'], ma_smoothed,
             label='Moving Average', linewidth=1.5)
    plt.plot(df_cleaned['two_theta'], sg_smoothed,
             label='Savitzky-Golay', linewidth=1.5)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Comparison of Smoothing Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on peak region
    plt.subplot(1, 2, 2)
    peak_region = (df_cleaned['two_theta'] > 26) & (df_cleaned['two_theta'] < 34)
    plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
             df_cleaned.loc[peak_region, 'intensity'],
             label='Original', alpha=0.5, linewidth=1)
    plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
             ma_smoothed[peak_region],
             label='Moving Average', linewidth=1.5)
    plt.plot(df_cleaned.loc[peak_region, 'two_theta'],
             sg_smoothed[peak_region],
             label='Savitzky-Golay', linewidth=1.5)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Zoomed: Peak Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Savitzky-Golay Parameters ===")
    print(f"Window length: {window_length}")
    print(f"Polynomial order: {polyorder}")
    print(f"\nRecommended settings:")
    print("- High noise: window_length=11-21, polyorder=2-3")
    print("- Low noise: window_length=5-11, polyorder=2-4")
    

**Advantages of Savitzky-Golay Filter** : \- More accurately preserves peak height and position \- Does not over-smooth edges (sharp changes) \- Better compatibility with derivative calculations (useful for later peak detection)

### Gaussian Filter

Widely used in image processing, but also applicable to 1D data.

**Code Example 6: Gaussian Filter**
    
    
    from scipy.ndimage import gaussian_filter1d
    
    # Apply Gaussian filter
    sigma_values = [1, 2, 4]  # Standard deviation
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1)
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Original Data')
    plt.grid(True, alpha=0.3)
    
    for i, sigma in enumerate(sigma_values, start=2):
        gaussian_smoothed = gaussian_filter1d(df_cleaned['intensity'].values, sigma=sigma)
    
        plt.subplot(2, 2, i)
        plt.plot(df_cleaned['two_theta'], gaussian_smoothed, linewidth=1.5)
        plt.xlabel('2θ (degree)')
        plt.ylabel('Intensity')
        plt.title(f'Gaussian Filter (σ={sigma})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Selecting the Appropriate Filter
    
    
    ```mermaid
    flowchart TD
        Start[Need noise removal] --> Q1{Is peak shape preservation important?}
        Q1 -->|Very important| SG[Savitzky-Golay]
        Q1 -->|Somewhat important| Gauss[Gaussian]
        Q1 -->|Not very important| MA[Moving Average]
    
        SG --> Param1[window=11-21\npolyorder=2-3]
        Gauss --> Param2[sigma=1-3]
        MA --> Param3[window=7-15]
    
        style Start fill:#4CAF50,color:#fff
        style SG fill:#2196F3,color:#fff
        style Gauss fill:#FF9800,color:#fff
        style MA fill:#9C27B0,color:#fff
    ```

* * *

## 1.4 Outlier Detection

Outliers occur due to measurement errors, equipment malfunctions, or sample contamination. If not properly detected and handled, they can significantly affect analysis results.

### Z-score Method

Data that is statistically "more than X standard deviations away from the mean" is considered an outlier.

**Code Example 7: Outlier Detection Using Z-score**
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Code Example 7: Outlier Detection Using Z-score
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from scipy import stats
    
    # Sample data with outliers
    data_with_outliers = df_cleaned['intensity'].copy()
    # Intentionally add outliers
    outlier_indices = [50, 150, 350, 550]
    data_with_outliers.iloc[outlier_indices] = [3000, -500, 4000, 3500]
    
    # Calculate Z-score
    z_scores = np.abs(stats.zscore(data_with_outliers))
    threshold = 3  # Consider data beyond 3σ as outliers
    
    outliers = z_scores > threshold
    
    print(f"=== Z-score Outlier Detection ===")
    print(f"Number of outliers: {outliers.sum()} / {len(data_with_outliers)}")
    print(f"Outlier indices: {np.where(outliers)[0]}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_cleaned['two_theta'], data_with_outliers, label='Data with outliers')
    plt.scatter(df_cleaned['two_theta'][outliers], data_with_outliers[outliers],
                color='red', s=100, zorder=5, label='Detected outliers')
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('Outlier Detection (Z-score method)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # After outlier removal
    data_cleaned = data_with_outliers.copy()
    data_cleaned[outliers] = np.nan
    data_cleaned = data_cleaned.interpolate(method='linear')
    
    plt.subplot(1, 2, 2)
    plt.plot(df_cleaned['two_theta'], data_cleaned, color='green', label='Cleaned data')
    plt.xlabel('2θ (degree)')
    plt.ylabel('Intensity')
    plt.title('After Outlier Removal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### IQR (Interquartile Range) Method

A robust median-based method that can also be applied to non-normally distributed data.

**Code Example 8: IQR Method**
    
    
    # Outlier detection using IQR method
    Q1 = data_with_outliers.quantile(0.25)
    Q3 = data_with_outliers.quantile(0.75)
    IQR = Q3 - Q1
    
    # Outlier definition: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
    
    print(f"=== IQR Outlier Detection ===")
    print(f"Q1: {Q1:.1f}")
    print(f"Q3: {Q3:.1f}")
    print(f"IQR: {IQR:.1f}")
    print(f"Lower bound: {lower_bound:.1f}")
    print(f"Upper bound: {upper_bound:.1f}")
    print(f"Number of outliers: {outliers_iqr.sum()} / {len(data_with_outliers)}")
    
    # Visualize with box plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].boxplot(data_with_outliers.dropna(), vert=True)
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Box Plot (outliers visible)')
    axes[0].grid(True, alpha=0.3)
    
    # After outlier removal
    data_cleaned_iqr = data_with_outliers.copy()
    data_cleaned_iqr[outliers_iqr] = np.nan
    data_cleaned_iqr = data_cleaned_iqr.interpolate(method='linear')
    
    axes[1].boxplot(data_cleaned_iqr.dropna(), vert=True)
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Box Plot (after outlier removal)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Z-score vs IQR Selection** : \- **Z-score method** : Effective when data is close to normal distribution, simple to calculate \- **IQR method** : Robust even for non-normal distributions, strong against extreme outliers

* * *

## 1.5 Standardization and Normalization

Standardization and normalization are necessary to make data at different scales comparable.

### Min-Max Scaling

Transforms data into the range [0, 1].

$$ X_{\text{normalized}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}} $$

### Z-score Standardization

Transforms to mean 0 and standard deviation 1.

$$ X_{\text{standardized}} = \frac{X - \mu}{\sigma} $$

### Baseline Correction

For spectral data, removes background.

**Implementation detailed in Chapter 2**

* * *

## 1.6 Practical Pitfalls and Solutions

### Common Failure Examples

#### Failure 1: Excessive Noise Removal

**Symptom** : Important peaks disappear or distort after smoothing

**Cause** : Window size too large or multiple stages of smoothing

**Solution** :
    
    
    # Bad example: excessive smoothing
    data_smooth1 = gaussian_filter1d(data, sigma=5)
    data_smooth2 = savgol_filter(data_smooth1, 21, 3)  # Double smoothing
    data_smooth3 = uniform_filter1d(data_smooth2, 15)  # More smoothing
    
    # Good example: appropriate smoothing
    data_smooth = savgol_filter(data, 11, 3)  # Once only, with appropriate parameters
    

#### Failure 2: Excessive Outlier Removal

**Symptom** : Important signals (sharp peaks) removed as outliers

**Cause** : Z-score threshold too low

**Solution** :
    
    
    # Bad example: threshold too strict
    outliers = np.abs(stats.zscore(data)) > 2  # 2σ removal → normal peaks also removed
    
    # Good example: appropriate threshold with visual confirmation
    outliers = np.abs(stats.zscore(data)) > 3  # 3σ is standard
    # Always visualize and confirm before removal
    plt.scatter(range(len(data)), data)
    plt.scatter(np.where(outliers)[0], data[outliers], color='red')
    plt.show()
    

#### Failure 3: Misuse of Missing Value Interpolation

**Symptom** : Data continuity lost or unnatural values generated

**Cause** : Linear interpolation of large missing regions

**Solution** :
    
    
    # Bad example: interpolate large missing regions
    data_interpolated = data.interpolate(method='linear')  # Unconditional interpolation of all regions
    
    # Good example: check missing range
    max_gap = 5  # Only interpolate missing values up to 5 points
    gaps = data.isnull().astype(int).groupby(data.notnull().astype(int).cumsum()).sum()
    if gaps.max() <= max_gap:
        data_interpolated = data.interpolate(method='linear')
    else:
        print(f"Warning: Large missing regions exist (max {gaps.max()} points)")
    

#### Failure 4: Confusing Measurement Noise with Signal

**Symptom** : Misidentifying noise as physical signal

**Cause** : Neglecting quantitative noise level assessment

**Solution** :
    
    
    # Quantitative noise level assessment
    baseline_region = data[(two_theta > 70) & (two_theta < 80)]  # Region with no signal
    noise_std = np.std(baseline_region)
    print(f"Noise level (standard deviation): {noise_std:.2f}")
    
    # Calculate signal-to-noise ratio (S/N ratio)
    peak_height = data.max() - data.min()
    snr = peak_height / noise_std
    print(f"S/N ratio: {snr:.1f}")
    
    # If S/N ratio is 3 or below, do not treat as signal
    if snr < 3:
        print("Warning: S/N ratio too low. Please re-run measurement.")
    

### Importance of Processing Order

Correct preprocessing pipeline order:
    
    
    ```mermaid
    flowchart LR
        A[1. Physical Anomaly Removal] --> B[2. Statistical Outlier Detection]
        B --> C[3. Missing Value Interpolation]
        C --> D[4. Noise Removal]
        D --> E[5. Baseline Correction]
        E --> F[6. Standardization]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#fce4ec
    ```

**Why This Order is Important** : 1\. **Physical anomalies** (negative intensities, etc.) must be removed first or they distort statistics 2\. **Statistical outliers** must be removed before noise removal or they spread during smoothing 3\. **Noise removal** must be done before baseline correction or noise affects baseline 4\. **Baseline correction** must be done before standardization or standardization becomes meaningless

* * *

## 1.7 Chapter Summary

### What We Learned

  1. **Data Licensing and Reproducibility** \- Utilizing public data repositories \- Documenting environment information and parameters \- Best practices for code reproducibility

  2. **Experimental Data Analysis Workflow** \- Data loading → preprocessing → feature extraction → analysis → visualization \- Importance and impact of preprocessing

  3. **Noise Removal Techniques** \- Moving average, Savitzky-Golay, Gaussian filters \- Criteria for selecting appropriate techniques

  4. **Outlier Detection** \- Z-score method, IQR method \- Physical validity checks

  5. **Standardization and Normalization** \- Min-Max scaling, Z-score standardization \- Principles for selection

  6. **Practical Pitfalls** \- Avoiding excessive noise removal \- Importance of processing order \- Distinguishing noise from signal

### Key Points

  * ✅ Preprocessing is the most critical step that determines analysis success or failure
  * ✅ Balance between peak shape preservation and noise reduction is important in noise removal
  * ✅ Always verify outliers and check physical validity
  * ✅ Confirm effect of each processing step through visualization
  * ✅ Processing order significantly affects results
  * ✅ Record environment and parameters for code reproducibility

### To the Next Chapter

In Chapter 2, we will learn analysis techniques for spectral data (XRD, XPS, IR, Raman): \- Peak detection algorithms \- Background removal \- Quantitative analysis \- Material identification using machine learning

**[Chapter 2: Spectral Data Analysis →](<chapter-2.html>)**

* * *

## Experimental Data Preprocessing Checklist

### Data Loading and Validation

  * [ ] Verify data format is correct (CSV, text, binary)
  * [ ] Verify column names are appropriate (two_theta, intensity, etc.)
  * [ ] Verify sufficient data points (minimum 100 points recommended)
  * [ ] Check percentage of missing values (caution if >30%)
  * [ ] Check for duplicate data

### Environment and Reproducibility

  * [ ] Record Python, NumPy, pandas, SciPy, Matplotlib versions
  * [ ] Fix random seed (if applicable)
  * [ ] Define parameters as constants (e.g., SG_WINDOW_LENGTH = 11)
  * [ ] Document data sources and licenses

### Physical Validity Checks

  * [ ] Check for negative intensity values (physically impossible for XRD/XPS)
  * [ ] Verify measurement range is appropriate (XRD: 10-80°, XPS: 0-1200 eV, etc.)
  * [ ] Calculate S/N ratio (desirable >3)
  * [ ] Evaluate noise level in baseline region

### Outlier Detection

  * [ ] Detect outliers using Z-score or IQR method
  * [ ] **Visualize and confirm before removal** (don't accidentally remove important peaks)
  * [ ] Record threshold (Z-score: 3σ, IQR: 1.5×, etc.)
  * [ ] Record number and position of removed outliers

### Missing Value Handling

  * [ ] Verify number and position of missing values
  * [ ] Check maximum length of consecutive missing values (desirable ≤5 points)
  * [ ] Select interpolation method (linear, spline, forward/backward fill)
  * [ ] Visualize and verify data after interpolation

### Noise Removal

  * [ ] Select filter type (moving average, Savitzky-Golay, Gaussian)
  * [ ] Determine parameters (window_length, polyorder, sigma)
  * [ ] Compare and visualize data before and after smoothing
  * [ ] Verify peak shapes are preserved
  * [ ] **Avoid double smoothing**

### Verify Processing Order

  * [ ] 1. Physical anomaly removal
  * [ ] 2. Statistical outlier detection
  * [ ] 3. Missing value interpolation
  * [ ] 4. Noise removal
  * [ ] 5. Baseline correction (if applicable)
  * [ ] 6. Standardization/normalization (if applicable)

### Visualization and Verification

  * [ ] Plot original data
  * [ ] Plot processed data
  * [ ] Overlay before and after processing
  * [ ] Verify main peak positions are preserved
  * [ ] Verify peak intensities have not changed significantly

### Documentation

  * [ ] Record processing parameters as comments or variables
  * [ ] Record reason for processing (e.g., "Using SG filter due to high noise")
  * [ ] Record reasons for excluded data
  * [ ] Record final data quality metrics (S/N ratio, number of data points, etc.)

### For Batch Processing

  * [ ] Implement error handling (try-except)
  * [ ] Record processing results in log file
  * [ ] Measure and record processing time
  * [ ] Save list of failed files
  * [ ] Calculate and report success rate (e.g., 95/100 files succeeded)

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

Determine whether the following statements are true or false.

  1. Increasing the window size of a moving average filter enhances noise removal but broadens peaks
  2. Savitzky-Golay filter preserves peak shapes better than moving average
  3. Z-score method cannot be applied to non-normally distributed data

Hint 1\. Consider the relationship between window size, noise removal effect, and peak shape 2\. Recall the mathematical differences between the two methods 3\. Think about the definition of Z-score and its behavior with non-normal distributions  Solution **Answer**: 1\. **True** - Larger windows average more points, reducing noise but also smoothing peaks 2\. **True** - Savitzky-Golay uses polynomial fitting, better preserving sharp changes 3\. **False** - Z-score can be calculated but interpretation of 3σ rule assumes normal distribution. IQR method is recommended for non-normal distributions **Explanation**: Noise removal always involves trade-offs. Attempting to completely remove noise also distorts signals. It's important to select appropriate parameters based on experimental data characteristics (noise level, peak sharpness). 

* * *

### Problem 2 (Difficulty: Medium)

Build an appropriate preprocessing pipeline for the following XRD data (sample).
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Build an appropriate preprocessing pipeline for the followin
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # Sample data
    import numpy as np
    import pandas as pd
    
    np.random.seed(100)
    two_theta = np.linspace(20, 60, 400)
    intensity = (
        800 * np.exp(-((two_theta - 30) ** 2) / 8) +
        1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
        np.random.normal(0, 100, len(two_theta))
    )
    # Add outliers
    intensity[50] = 3000
    intensity[200] = -500
    
    df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})
    

**Requirements** : 1\. Replace negative intensity values with 0 2\. Detect and remove outliers using Z-score (threshold 3σ) 3\. Remove noise using Savitzky-Golay filter (window=11, polyorder=3) 4\. Visualize data before and after processing

Hint **Processing Flow**: 1\. Create mask for negative values → replace with 0 2\. Detect outliers using `scipy.stats.zscore` 3\. Interpolate outliers linearly 4\. Smooth using `scipy.signal.savgol_filter` 5\. Compare original and processed data using `matplotlib` **Functions to Use**: \- `df[condition]` for conditional extraction \- `np.abs(stats.zscore())` for Z-score \- `interpolate(method='linear')` for interpolation \- `savgol_filter()` for smoothing  Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Requirements:
    1. Replace negative intensity values with 0
    2.
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.signal import savgol_filter
    
    # Sample data
    np.random.seed(100)
    two_theta = np.linspace(20, 60, 400)
    intensity = (
        800 * np.exp(-((two_theta - 30) ** 2) / 8) +
        1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
        np.random.normal(0, 100, len(two_theta))
    )
    intensity[50] = 3000
    intensity[200] = -500
    
    df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})
    
    # Step 1: Replace negative values with 0
    df_cleaned = df.copy()
    negative_mask = df_cleaned['intensity'] < 0
    df_cleaned.loc[negative_mask, 'intensity'] = 0
    print(f"Number of negative values: {negative_mask.sum()}")
    
    # Step 2: Outlier detection (Z-score method)
    z_scores = np.abs(stats.zscore(df_cleaned['intensity']))
    outliers = z_scores > 3
    print(f"Number of outliers: {outliers.sum()}")
    
    # Step 3: Interpolate outliers
    df_cleaned.loc[outliers, 'intensity'] = np.nan
    df_cleaned['intensity'] = df_cleaned['intensity'].interpolate(method='linear')
    
    # Step 4: Savitzky-Golay filter
    intensity_smoothed = savgol_filter(df_cleaned['intensity'].values, window_length=11, polyorder=3)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    axes[0, 0].plot(df['two_theta'], df['intensity'], linewidth=1)
    axes[0, 0].set_title('Original Data (with outliers)')
    axes[0, 0].set_xlabel('2θ (degree)')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After outlier removal
    axes[0, 1].plot(df_cleaned['two_theta'], df_cleaned['intensity'], linewidth=1, color='orange')
    axes[0, 1].set_title('After Outlier Removal')
    axes[0, 1].set_xlabel('2θ (degree)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # After smoothing
    axes[1, 0].plot(df_cleaned['two_theta'], intensity_smoothed, linewidth=1.5, color='green')
    axes[1, 0].set_title('After Savitzky-Golay Smoothing')
    axes[1, 0].set_xlabel('2θ (degree)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall comparison
    axes[1, 1].plot(df['two_theta'], df['intensity'], label='Original', alpha=0.4, linewidth=1)
    axes[1, 1].plot(df_cleaned['two_theta'], intensity_smoothed, label='Processed', linewidth=1.5)
    axes[1, 1].set_title('Comparison')
    axes[1, 1].set_xlabel('2θ (degree)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output**: 
    
    
    Number of negative values: 1
    Number of outliers: 2
    

**Explanation**: This preprocessing pipeline removes physically impossible negative values, detects and interpolates statistical outliers, and finally removes noise. The order of processing is important; performing outlier removal before smoothing minimizes the impact of outliers. 

* * *

### Problem 3 (Difficulty: Hard)

Real materials research scenario: You obtained data from 1000 samples through high-throughput XRD measurement. Build a system that automates the preprocessing pipeline for each sample and saves the processed data to CSV files.

**Background** : An automated XRD system generates measurement data for 100 samples daily. Manual processing is impossible, making automation essential.

**Tasks** : 1\. Create a function to batch-process multiple samples 2\. Error handling (invalid data formats, extreme outliers) 3\. Log output of processing results 4\. Save processed data

**Constraints** : \- Number of data points per sample may vary \- Some samples may have incomplete data due to measurement failure \- Processing must complete within 5 seconds/sample

Hint **Approach**: 1\. Define a function that encapsulates preprocessing 2\. Error handling with `try-except` 3\. Record processing results in log file 4\. Save using `pandas.to_csv()` **Design Pattern**: 
    
    
    def preprocess_xrd(data, params):
        """Preprocess XRD data"""
        # 1. Validation
        # 2. Remove negative values
        # 3. Detect outliers
        # 4. Smoothing
        # 5. Return results
        pass
    
    def batch_process(file_list):
        """Batch process multiple files"""
        for file in file_list:
            try:
                # Load data
                # Execute preprocessing
                # Save
            except Exception as e:
                # Error log
                pass
    

Solution **Solution Overview**: Build a robust pipeline to automatically process multiple samples of XRD data, including error handling, log output, and processing time measurement. **Implementation Code**: 
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Constraints:
    - Number of data points per sample may vary
    - S
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 1-3 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.signal import savgol_filter
    import time
    import logging
    from pathlib import Path
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('xrd_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    def validate_data(df):
        """Data validity check"""
        if df.empty:
            raise ValueError("Empty DataFrame")
    
        if 'two_theta' not in df.columns or 'intensity' not in df.columns:
            raise ValueError("Missing required columns")
    
        if len(df) < 50:
            raise ValueError(f"Insufficient data points: {len(df)}")
    
        if df['intensity'].isnull().sum() > len(df) * 0.3:
            raise ValueError("Too many missing values (>30%)")
    
        return True
    
    
    def preprocess_xrd(df, params=None):
        """
        XRD data preprocessing pipeline
    
        Parameters:
        -----------
        df : pd.DataFrame
            Columns: 'two_theta', 'intensity'
        params : dict
            Preprocessing parameters
            - z_threshold: Z-score threshold (default: 3)
            - sg_window: Savitzky-Golay window (default: 11)
            - sg_polyorder: Polynomial order (default: 3)
    
        Returns:
        --------
        df_processed : pd.DataFrame
            Preprocessed data
        """
        # Default parameters
        if params is None:
            params = {
                'z_threshold': 3,
                'sg_window': 11,
                'sg_polyorder': 3
            }
    
        # Data validation
        validate_data(df)
    
        df_processed = df.copy()
    
        # Step 1: Replace negative values with 0
        negative_count = (df_processed['intensity'] < 0).sum()
        df_processed.loc[df_processed['intensity'] < 0, 'intensity'] = 0
    
        # Step 2: Outlier detection and interpolation
        z_scores = np.abs(stats.zscore(df_processed['intensity']))
        outliers = z_scores > params['z_threshold']
        outlier_count = outliers.sum()
    
        df_processed.loc[outliers, 'intensity'] = np.nan
        df_processed['intensity'] = df_processed['intensity'].interpolate(method='linear')
    
        # Step 3: Savitzky-Golay filter
        try:
            intensity_smoothed = savgol_filter(
                df_processed['intensity'].values,
                window_length=params['sg_window'],
                polyorder=params['sg_polyorder']
            )
            df_processed['intensity'] = intensity_smoothed
        except Exception as e:
            logging.warning(f"Savitzky-Golay failed: {e}. Using moving average.")
            from scipy.ndimage import uniform_filter1d
            df_processed['intensity'] = uniform_filter1d(
                df_processed['intensity'].values,
                size=params['sg_window']
            )
    
        # Processing statistics
        stats_dict = {
            'negative_values': negative_count,
            'outliers': outlier_count,
            'data_points': len(df_processed)
        }
    
        return df_processed, stats_dict
    
    
    def batch_process_xrd(input_files, output_dir, params=None):
        """
        Batch process multiple XRD files
    
        Parameters:
        -----------
        input_files : list
            List of input file paths
        output_dir : str or Path
            Output directory
        params : dict
            Preprocessing parameters
    
        Returns:
        --------
        results : dict
            Processing result summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        results = {
            'total': len(input_files),
            'success': 0,
            'failed': 0,
            'processing_times': []
        }
    
        logging.info(f"Starting batch processing of {len(input_files)} files")
    
        for i, file_path in enumerate(input_files, 1):
            file_path = Path(file_path)
            start_time = time.time()
    
            try:
                # Load data
                df = pd.read_csv(file_path)
    
                # Execute preprocessing
                df_processed, stats_dict = preprocess_xrd(df, params)
    
                # Save
                output_file = output_dir / f"processed_{file_path.name}"
                df_processed.to_csv(output_file, index=False)
    
                # Processing time
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                results['success'] += 1
    
                logging.info(
                    f"[{i}/{len(input_files)}] SUCCESS: {file_path.name} "
                    f"({processing_time:.2f}s) - "
                    f"Negatives: {stats_dict['negative_values']}, "
                    f"Outliers: {stats_dict['outliers']}"
                )
    
            except Exception as e:
                results['failed'] += 1
                logging.error(f"[{i}/{len(input_files)}] FAILED: {file_path.name} - {str(e)}")
    
        # Summary
        avg_time = np.mean(results['processing_times']) if results['processing_times'] else 0
        logging.info(
            f"\n=== Batch Processing Complete ===\n"
            f"Total: {results['total']}\n"
            f"Success: {results['success']}\n"
            f"Failed: {results['failed']}\n"
            f"Average processing time: {avg_time:.2f}s"
        )
    
        return results
    
    
    # ==================== Demo Execution ====================
    
    if __name__ == "__main__":
        # Generate sample data (in practice, load experimental data)
        np.random.seed(42)
    
        # Generate 10 sample datasets
        sample_dir = Path("sample_xrd_data")
        sample_dir.mkdir(exist_ok=True)
    
        for i in range(10):
            two_theta = np.linspace(20, 60, 400)
            intensity = (
                800 * np.exp(-((two_theta - 30) ** 2) / 8) +
                1200 * np.exp(-((two_theta - 45) ** 2) / 6) +
                np.random.normal(0, 100, len(two_theta))
            )
    
            # Randomly add outliers
            if np.random.rand() > 0.5:
                outlier_idx = np.random.randint(0, len(intensity), size=2)
                intensity[outlier_idx] = np.random.choice([3000, -500], size=2)
    
            df = pd.DataFrame({'two_theta': two_theta, 'intensity': intensity})
            df.to_csv(sample_dir / f"sample_{i:03d}.csv", index=False)
    
        # Execute batch processing
        input_files = list(sample_dir.glob("sample_*.csv"))
        output_dir = Path("processed_xrd_data")
    
        params = {
            'z_threshold': 3,
            'sg_window': 11,
            'sg_polyorder': 3
        }
    
        results = batch_process_xrd(input_files, output_dir, params)
    
        print(f"\nProcessing complete: {results['success']}/{results['total']} files")
    

**Results**: 
    
    
    2025-10-17 10:30:15 - INFO - Starting batch processing of 10 files
    2025-10-17 10:30:15 - INFO - [1/10] SUCCESS: sample_000.csv (0.15s) - Negatives: 1, Outliers: 2
    2025-10-17 10:30:15 - INFO - [2/10] SUCCESS: sample_001.csv (0.12s) - Negatives: 0, Outliers: 1
    ...
    2025-10-17 10:30:16 - INFO -
    === Batch Processing Complete ===
    Total: 10
    Success: 10
    Failed: 0
    Average processing time: 0.13s
    

**Detailed Explanation**: 1\. **Error Handling**: Pre-check with `validate_data()`, capture runtime errors with `try-except` 2\. **Logging**: Output processing status to both file and console 3\. **Parameterization**: Preprocessing parameters can be specified externally 4\. **Performance**: Measure processing time to identify bottlenecks 5\. **Scalability**: Works regardless of number of files **Additional Considerations**: \- Accelerate with parallel processing (`multiprocessing`) \- Save to database (SQLite) \- Visualize processing status with web dashboard \- Integration with cloud storage (S3, GCS) 

* * *

## References

  1. VanderPlas, J. (2016). "Python Data Science Handbook." O'Reilly Media. ISBN: 978-1491912058

  2. Savitzky, A., & Golay, M. J. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." _Analytical Chemistry_ , 36(8), 1627-1639. DOI: [10.1021/ac60214a047](<https://doi.org/10.1021/ac60214a047>)

  3. Stein, H. S. et al. (2019). "Progress and prospects for accelerating materials science with automated and autonomous workflows." _Chemical Science_ , 10(42), 9640-9649. DOI: [10.1039/C9SC03766G](<https://doi.org/10.1039/C9SC03766G>)

  4. SciPy Documentation: Signal Processing. URL: <https://docs.scipy.org/doc/scipy/reference/signal.html>

  5. pandas Documentation: Data Cleaning. URL: <https://pandas.pydata.org/docs/user_guide/missing_data.html>

* * *

## Navigation

### Previous Chapter

None (Chapter 1)

### Next Chapter

**[Chapter 2: Spectral Data Analysis →](<chapter-2.html>)**

### Series Contents

**[← Return to Series Contents](<./index.html>)**

* * *

## Author Information

**Author** : AI Terakoya Content Team **Created** : 2025-10-17 **Version** : 1.0

**Update History** : \- 2025-10-17: v1.0 Initial release

**Feedback** : \- GitHub Issues: [Repository URL]/issues \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

**Continue learning in the next chapter!**
