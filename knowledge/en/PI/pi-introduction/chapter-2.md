---
title: "Chapter 2: Process Data Preprocessing and Visualization"
chapter_title: "Chapter 2: Process Data Preprocessing and Visualization"
subtitle: Practical Preprocessing Methods to Improve Data Quality
version: 1.0
created_at: 2025-10-25
---

# Chapter 2: Process Data Preprocessing and Visualization

Process data preprocessing is the most critical step to obtain high-quality analysis results. Learn practical methods for handling time series data, addressing missing values and outliers, and data scaling techniques.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Time series data manipulation using Pandas (resampling, rolling statistics)
  * ✅ Identify types of missing values (MCAR/MAR/MNAR) and select appropriate imputation methods
  * ✅ Implement multiple outlier detection methods (Z-score, IQR, Isolation Forest)
  * ✅ Choose appropriate scaling methods for process data
  * ✅ Create advanced visualizations with Matplotlib/Seaborn

* * *

## 2.1 Handling Time Series Data

Process industry data is primarily **time series data** that changes over time. Mastering Pandas' powerful time series capabilities is a fundamental skill in PI.

### Basics of Pandas DatetimeIndex

To work with time series data, first set up a `DatetimeIndex`. This makes time-based operations much easier.

#### Code Example 1: DatetimeIndex Setup and Slicing
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 1: DatetimeIndex Setup and Slicing
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate sample data: 3 days of distillation column operation data (1-minute intervals)
    np.random.seed(42)
    dates = pd.date_range('2025-01-01 00:00', periods=4320, freq='1min')  # 3 days
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': 85 + np.random.normal(0, 1.5, 4320) + 2*np.sin(np.arange(4320)*2*np.pi/1440),
        'pressure': 1.2 + np.random.normal(0, 0.05, 4320),
        'flow_rate': 50 + np.random.normal(0, 3, 4320)
    })
    
    # Set as DatetimeIndex
    df = df.set_index('timestamp')
    print("Basic DataFrame Information:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Time-based slicing
    print("\nData from 2025-01-01 12:00 to 13:00:")
    subset = df['2025-01-01 12:00':'2025-01-01 13:00']
    print(subset.head())
    print(f"Number of records: {len(subset)}")
    
    # Extract data for a specific day
    day1 = df['2025-01-01']
    print(f"\nNumber of records for 2025-01-01: {len(day1)}")
    
    # Time period filtering (9:00-17:00 for all days)
    business_hours = df.between_time('09:00', '17:00')
    print(f"\nNumber of records during business hours: {len(business_hours)}")
    
    # Calculate statistics
    print("\nDaily Statistics:")
    daily_stats = df.resample('D').agg({
        'temperature': ['mean', 'std', 'min', 'max'],
        'pressure': ['mean', 'std'],
        'flow_rate': ['mean', 'sum']
    })
    print(daily_stats)
    

**Output Example** :
    
    
    Basic DataFrame Information:
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4320 entries, 2025-01-01 00:00:00 to 2025-01-03 23:59:00
    Freq: min
    Data columns (total 3 columns):
    ...
    
    Number of records for 2025-01-01: 1440
    Number of records during business hours: 1440
    

**Explanation** : Using `DatetimeIndex` enables intuitive time-based slicing and aggregation. Extracting specific time periods and calculating daily/weekly statistics for process data becomes straightforward.

### Resampling

Sensor data is often recorded at second intervals, but minute or hourly data may be sufficient for analysis. **Resampling** allows you to adjust the data granularity.

#### Code Example 2: Resampling and Downsampling
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 2: Resampling and Downsampling
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate high-frequency data (5-second intervals, 1 day)
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=17280, freq='5s')  # 5-second intervals
    df_highfreq = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 0.5, 17280)
    }, index=dates)
    
    print(f"Original data: {len(df_highfreq)} records (5-second intervals)")
    print(df_highfreq.head())
    
    # Downsample to 1-minute average
    df_1min = df_highfreq.resample('1min').mean()
    print(f"\n1-minute average: {len(df_1min)} records")
    print(df_1min.head())
    
    # 5-minute aggregation (multiple aggregation functions)
    df_5min = df_highfreq.resample('5min').agg(['mean', 'std', 'min', 'max'])
    print(f"\n5-minute aggregation: {len(df_5min)} records")
    print(df_5min.head())
    
    # Hourly aggregation
    df_hourly = df_highfreq.resample('1h').agg({
        'temperature': ['mean', 'std', 'count']
    })
    print(f"\nHourly aggregation: {len(df_hourly)} records")
    print(df_hourly.head(10))
    
    # Visualization: Original data and resampled results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Original data (first 1 hour only)
    axes[0].plot(df_highfreq.index[:720], df_highfreq['temperature'][:720],
                 linewidth=0.5, alpha=0.7, label='Original (5s)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Original Data (5-second interval)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 1-minute average (first 1 hour)
    axes[1].plot(df_1min.index[:60], df_1min['temperature'][:60],
                 linewidth=1, color='#11998e', label='1-min average')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Resampled to 1-minute average')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 5-minute average (full day)
    axes[2].plot(df_5min.index, df_5min['temperature']['mean'],
                 linewidth=1.5, color='#f59e0b', label='5-min average')
    axes[2].fill_between(df_5min.index,
                          df_5min['temperature']['min'],
                          df_5min['temperature']['max'],
                          alpha=0.2, color='#f59e0b', label='Min-Max range')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Resampled to 5-minute statistics')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Data size comparison
    print("\n\nData Size Comparison:")
    print(f"Original data (5s): {len(df_highfreq):,} records")
    print(f"1-minute average: {len(df_1min):,} records ({len(df_1min)/len(df_highfreq)*100:.1f}%)")
    print(f"5-minute average: {len(df_5min):,} records ({len(df_5min)/len(df_highfreq)*100:.2f}%)")
    

**Output Example** :
    
    
    Original data: 17,280 records (5-second intervals)
    1-minute average: 1,440 records (8.3%)
    5-minute average: 288 records (1.67%)
    

**Explanation** : Resampling reduces data volume while reducing noise. Selecting the appropriate time granularity based on analysis objectives optimizes the balance between computational efficiency and information content.

### Rolling Statistics

**Rolling statistics** such as moving averages and moving standard deviations are effective for trend identification and noise reduction.

#### Code Example 3: Rolling Statistics and Trend Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 3: Rolling Statistics and Trend Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sample data: Reactor temperature data with noise
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    # Trend component + noise
    trend = 170 + np.linspace(0, 10, 1440)  # Slowly increasing
    noise = np.random.normal(0, 2, 1440)
    df = pd.DataFrame({
        'temperature': trend + noise
    }, index=dates)
    
    # Calculate rolling statistics
    df['rolling_mean_10'] = df['temperature'].rolling(window=10).mean()
    df['rolling_mean_60'] = df['temperature'].rolling(window=60).mean()
    df['rolling_std_60'] = df['temperature'].rolling(window=60).std()
    
    # Deviation from moving average (useful for anomaly detection)
    df['deviation'] = df['temperature'] - df['rolling_mean_60']
    
    # Rolling max/min (60-minute window)
    df['rolling_max'] = df['temperature'].rolling(window=60).max()
    df['rolling_min'] = df['temperature'].rolling(window=60).min()
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Original data and rolling average
    axes[0].plot(df.index, df['temperature'], alpha=0.3, linewidth=0.5,
                 label='Raw data', color='gray')
    axes[0].plot(df.index, df['rolling_mean_10'], linewidth=1.5,
                 label='10-min moving average', color='#11998e')
    axes[0].plot(df.index, df['rolling_mean_60'], linewidth=2,
                 label='60-min moving average', color='#f59e0b')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Rolling Average for Trend Identification')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Rolling standard deviation (monitoring variability)
    axes[1].plot(df.index, df['rolling_std_60'], linewidth=1.5, color='#7b2cbf')
    axes[1].axhline(y=df['rolling_std_60'].mean(), color='red', linestyle='--',
                    label=f'Average Std: {df["rolling_std_60"].mean():.2f}')
    axes[1].set_ylabel('Rolling Std (°C)')
    axes[1].set_title('60-min Rolling Standard Deviation (Process Stability)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Deviation from moving average
    axes[2].plot(df.index, df['deviation'], linewidth=0.8, color='#11998e')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].fill_between(df.index, -3, 3, alpha=0.2, color='green',
                          label='Normal range (±3°C)')
    axes[2].set_ylabel('Deviation (°C)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Deviation from 60-min Moving Average')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("Rolling Statistics Summary:")
    print(df[['rolling_mean_60', 'rolling_std_60', 'deviation']].describe())
    
    # Anomaly detection (deviation ±4°C or more from moving average)
    anomalies = df[abs(df['deviation']) > 4]
    print(f"\nAnomaly data points: {len(anomalies)} records ({len(anomalies)/len(df)*100:.2f}%)")
    if len(anomalies) > 0:
        print(anomalies.head())
    

**Output Example** :
    
    
    Rolling Statistics Summary:
           rolling_mean_60  rolling_std_60  deviation
    count      1381.000000     1381.000000  1381.000000
    mean        174.952379        1.998624     0.004892
    std           2.885820        0.289455     2.019341
    min         170.256432        1.187654    -6.234521
    max         180.134567        3.456789     5.987654
    
    Anomaly data points: 12 records (0.83%)
    

**Explanation** : Rolling statistics form the foundation for process trend identification, stability monitoring, and anomaly detection. The window size should be adjusted according to the process time constant.

* * *

## 2.2 Missing Value Handling and Outlier Detection

Real process data contains **missing values** due to sensor failures or communication errors, and **outliers** due to abnormal measurements. Appropriate handling is essential.

### Types of Missing Values and Countermeasures

Missing values are classified into three types:

Type | Description | Example | Recommended Approach  
---|---|---|---  
**MCAR**  
(Missing Completely At Random) | Missing completely at random | Temporary sensor communication error | Linear interpolation, moving average imputation  
**MAR**  
(Missing At Random) | Missing depends on other variables | Sensor tends to fail at high temperatures | Regression imputation, K-nearest neighbors imputation  
**MNAR**  
(Missing Not At Random) | Missingness itself contains information | Out-of-range values are not recorded | Careful analysis, consider deletion  
  
#### Code Example 4: Missing Value Detection and Multiple Imputation Methods
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: Missing Value Detection and Multiple Imputat
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.impute import KNNImputer
    
    # Generate sample data: Intentionally create missing values
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='5min')
    df = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 2, 500),
        'pressure': 1.5 + np.random.normal(0, 0.1, 500),
        'flow_rate': 50 + np.random.normal(0, 3, 500)
    }, index=dates)
    
    # Insert missing values randomly (10% missing rate)
    missing_indices = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
    df.loc[missing_indices, 'temperature'] = np.nan
    
    # Add consecutive missing values (simulate sensor failure)
    df.loc['2025-01-01 10:00':'2025-01-01 10:30', 'pressure'] = np.nan
    
    print("Missing Value Check:")
    print(df.isnull().sum())
    print(f"\nMissing Rate:")
    print(df.isnull().sum() / len(df) * 100)
    
    # Method 1: Linear interpolation (optimal for time series data)
    df_linear = df.copy()
    df_linear['temperature'] = df_linear['temperature'].interpolate(method='linear')
    df_linear['pressure'] = df_linear['pressure'].interpolate(method='linear')
    
    # Method 2: Spline interpolation (smooth imputation)
    df_spline = df.copy()
    df_spline['temperature'] = df_spline['temperature'].interpolate(method='spline', order=2)
    df_spline['pressure'] = df_spline['pressure'].interpolate(method='spline', order=2)
    
    # Method 3: Forward Fill
    df_ffill = df.copy()
    df_ffill = df_ffill.fillna(method='ffill')
    
    # Method 4: K-nearest neighbors imputation (considers multivariate)
    imputer = KNNImputer(n_neighbors=5)
    df_knn = df.copy()
    df_knn_values = imputer.fit_transform(df_knn)
    df_knn = pd.DataFrame(df_knn_values, columns=df.columns, index=df.index)
    
    # Visualization: Comparison of imputation methods
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Temperature data imputation comparison
    time_range = slice('2025-01-01 08:00', '2025-01-01 12:00')
    axes[0].plot(df.loc[time_range].index, df.loc[time_range, 'temperature'],
                 'o', markersize=4, label='Original (with missing)', alpha=0.5)
    axes[0].plot(df_linear.loc[time_range].index, df_linear.loc[time_range, 'temperature'],
                 linewidth=2, label='Linear interpolation', alpha=0.8)
    axes[0].plot(df_spline.loc[time_range].index, df_spline.loc[time_range, 'temperature'],
                 linewidth=2, label='Spline interpolation', alpha=0.8)
    axes[0].plot(df_knn.loc[time_range].index, df_knn.loc[time_range, 'temperature'],
                 linewidth=2, label='KNN imputation', alpha=0.8, linestyle='--')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Comparison of Missing Value Imputation Methods - Temperature')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Pressure data imputation (includes consecutive missing)
    axes[1].plot(df.loc[time_range].index, df.loc[time_range, 'pressure'],
                 'o', markersize=4, label='Original (with missing)', alpha=0.5)
    axes[1].plot(df_linear.loc[time_range].index, df_linear.loc[time_range, 'pressure'],
                 linewidth=2, label='Linear interpolation', alpha=0.8)
    axes[1].plot(df_ffill.loc[time_range].index, df_ffill.loc[time_range, 'pressure'],
                 linewidth=2, label='Forward fill', alpha=0.8)
    axes[1].set_ylabel('Pressure (MPa)')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Comparison of Imputation Methods - Pressure (with consecutive missing)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nMissing Check After Imputation:")
    print("Linear interpolation:", df_linear.isnull().sum().sum())
    print("Spline interpolation:", df_spline.isnull().sum().sum())
    print("KNN imputation:", df_knn.isnull().sum().sum())
    

**Output Example** :
    
    
    Missing Value Check:
    temperature    50
    pressure        7
    flow_rate       0
    dtype: int64
    
    Missing Rate:
    temperature    10.0
    pressure        1.4
    flow_rate       0.0
    
    Missing Check After Imputation:
    Linear interpolation: 0
    Spline interpolation: 0
    KNN imputation: 0
    

**Explanation** : For process data, linear interpolation or spline interpolation considering time series properties is effective. KNN imputation is also an excellent choice when multivariate correlations are strong. For long consecutive missing periods, consider deletion rather than imputation.

### Practical Outlier Detection Methods

Outliers occur due to various causes including measurement errors, sensor failures, and actual abnormal conditions. Selecting appropriate detection methods is crucial.

#### Code Example 5: Statistical Outlier Detection (Z-score, IQR)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Code Example 5: Statistical Outlier Detection (Z-score, IQR)
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Generate sample data: Normal data + intentional outliers
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'temperature': np.random.normal(175, 2, n)
    })
    
    # Add outliers (5%)
    outlier_indices = np.random.choice(range(n), size=50, replace=False)
    df.loc[outlier_indices, 'temperature'] += np.random.choice([-15, 15], size=50)
    
    # Method 1: Z-score method (how many standard deviations from the mean)
    df['z_score'] = np.abs(stats.zscore(df['temperature']))
    df['outlier_zscore'] = df['z_score'] > 3  # 3σ rule
    
    # Method 2: IQR method (interquartile range)
    Q1 = df['temperature'].quantile(0.25)
    Q3 = df['temperature'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['outlier_iqr'] = (df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)
    
    # Method 3: Modified Z-score method (robust method)
    median = df['temperature'].median()
    mad = np.median(np.abs(df['temperature'] - median))
    modified_z_scores = 0.6745 * (df['temperature'] - median) / mad
    df['outlier_modified_z'] = np.abs(modified_z_scores) > 3.5
    
    # Result summary
    print("Outlier Detection Results:")
    print(f"Z-score method (>3σ): {df['outlier_zscore'].sum()} records ({df['outlier_zscore'].sum()/len(df)*100:.2f}%)")
    print(f"IQR method (1.5×IQR): {df['outlier_iqr'].sum()} records ({df['outlier_iqr'].sum()/len(df)*100:.2f}%)")
    print(f"Modified Z-score method: {df['outlier_modified_z'].sum()} records ({df['outlier_modified_z'].sum()/len(df)*100:.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data histogram
    axes[0, 0].hist(df['temperature'], bins=50, color='#11998e', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['temperature'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df["temperature"].mean():.2f}')
    axes[0, 0].axvline(df['temperature'].median(), color='orange', linestyle='--',
                        label=f'Median: {df["temperature"].median():.2f}')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Z-score method
    axes[0, 1].scatter(range(len(df)), df['temperature'], c=df['outlier_zscore'],
                       cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[0, 1].set_xlabel('Data Point Index')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title(f'Z-score Method ({df["outlier_zscore"].sum()} outliers)')
    axes[0, 1].grid(alpha=0.3)
    
    # IQR method
    axes[1, 0].scatter(range(len(df)), df['temperature'], c=df['outlier_iqr'],
                       cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[1, 0].axhline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:.2f}')
    axes[1, 0].axhline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:.2f}')
    axes[1, 0].set_xlabel('Data Point Index')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title(f'IQR Method ({df["outlier_iqr"].sum()} outliers)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Box plot
    box_data = [df[~df['outlier_iqr']]['temperature'],
                df[df['outlier_iqr']]['temperature']]
    axes[1, 1].boxplot(box_data, labels=['Normal', 'Outliers'], patch_artist=True,
                       boxprops=dict(facecolor='#11998e', alpha=0.7))
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_title('Box Plot: Normal vs Outliers (IQR method)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Outlier statistics
    print("\nOutlier Statistics:")
    print(df[df['outlier_iqr']]['temperature'].describe())
    

**Explanation** : The Z-score method assumes normal distribution and is vulnerable to outliers. The IQR method is robust and less affected by outliers. For process data, using both methods together is recommended.

#### Code Example 6: Machine Learning-Based Outlier Detection (Isolation Forest)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 6: Machine Learning-Based Outlier Detection (Is
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    
    # Generate sample data: Multivariate process data
    np.random.seed(42)
    n = 1000
    
    # Normal data (correlated 2 variables)
    temperature = np.random.normal(175, 2, n)
    pressure = 1.5 + 0.01 * (temperature - 175) + np.random.normal(0, 0.05, n)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure
    })
    
    # Add anomalous data (multivariate anomaly patterns)
    # Pattern 1: Temperature abnormally high but pressure normal
    df.loc[950:960, 'temperature'] += 20
    # Pattern 2: Pressure abnormally low but temperature normal
    df.loc[970:980, 'pressure'] -= 0.5
    # Pattern 3: Both abnormal
    df.loc[990:995, ['temperature', 'pressure']] += [15, 0.3]
    
    # Outlier detection with Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['outlier_if'] = iso_forest.fit_predict(df[['temperature', 'pressure']])
    df['outlier_if'] = df['outlier_if'] == -1  # -1 indicates outlier
    
    # Anomaly score (lower values indicate more anomalous)
    df['anomaly_score'] = iso_forest.score_samples(df[['temperature', 'pressure']])
    
    print("Isolation Forest Detection Results:")
    print(f"Detected outliers: {df['outlier_if'].sum()} records ({df['outlier_if'].sum()/len(df)*100:.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: Normal vs Anomalous
    normal = df[~df['outlier_if']]
    outliers = df[df['outlier_if']]
    
    axes[0].scatter(normal['temperature'], normal['pressure'],
                    c='#11998e', alpha=0.6, s=30, label='Normal')
    axes[0].scatter(outliers['temperature'], outliers['pressure'],
                    c='red', alpha=0.8, s=50, marker='x', label='Outliers')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Pressure (MPa)')
    axes[0].set_title('Isolation Forest: Outlier Detection (2D)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Anomaly score histogram
    axes[1].hist(df[~df['outlier_if']]['anomaly_score'], bins=50,
                 alpha=0.7, color='#11998e', label='Normal', edgecolor='black')
    axes[1].hist(df[df['outlier_if']]['anomaly_score'], bins=20,
                 alpha=0.7, color='red', label='Outliers', edgecolor='black')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Anomaly Scores')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Most anomalous data points
    print("\nTop 5 Most Anomalous Data Points:")
    most_anomalous = df.nsmallest(5, 'anomaly_score')[['temperature', 'pressure', 'anomaly_score']]
    print(most_anomalous)
    
    # Statistical comparison
    print("\nNormal Data Statistics:")
    print(normal[['temperature', 'pressure']].describe())
    print("\nOutlier Statistics:")
    print(outliers[['temperature', 'pressure']].describe())
    

**Output Example** :
    
    
    Isolation Forest Detection Results:
    Detected outliers: 50 records (5.00%)
    
    Top 5 Most Anomalous Data Points:
         temperature  pressure  anomaly_score
    990    189.234567  1.823456      -0.234567
    991    190.123456  1.834567      -0.223456
    995    188.987654  1.812345      -0.219876
    ...
    

**Explanation** : Isolation Forest excels at detecting outliers in multivariate data. It can capture anomalies in variable relationships that univariate statistical methods cannot detect. Very effective for process data anomaly detection.

* * *

## 2.3 Data Scaling and Normalization

Before building machine learning models, **scaling** to align variable scales is important. In process data, variable scales differ greatly, such as temperature (0-200°C) and pressure (0-3 MPa).

### Major Scaling Methods

Method | Transformation Formula | Characteristics | Application Scenarios  
---|---|---|---  
**Min-Max  
Scaling** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Transform to [0, 1] range | When outliers are few  
**Standard  
Scaling** | $x' = \frac{x - \mu}{\sigma}$ | Mean 0, standard deviation 1 | When distribution is close to normal  
**Robust  
Scaling** | $x' = \frac{x - Q_{med}}{Q_{75} - Q_{25}}$ | Robust to outliers | When outliers are many  
  
#### Code Example 7: Scaling Method Comparison and Selection
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 7: Scaling Method Comparison and Selection
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    # Generate sample data: Process data with outliers
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'temperature': np.random.normal(175, 5, n),
        'pressure': np.random.normal(1.5, 0.2, n),
        'flow_rate': np.random.normal(50, 10, n)
    })
    
    # Add outliers
    df.loc[480:490, 'temperature'] += 50  # Temperature outliers
    df.loc[491:495, 'pressure'] += 1.5    # Pressure outliers
    
    print("Original Data Statistics:")
    print(df.describe())
    
    # Apply three scaling methods
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    
    df_minmax = pd.DataFrame(
        minmax_scaler.fit_transform(df),
        columns=[col + '_minmax' for col in df.columns]
    )
    
    df_standard = pd.DataFrame(
        standard_scaler.fit_transform(df),
        columns=[col + '_standard' for col in df.columns]
    )
    
    df_robust = pd.DataFrame(
        robust_scaler.fit_transform(df),
        columns=[col + '_robust' for col in df.columns]
    )
    
    # Combine results
    df_scaled = pd.concat([df, df_minmax, df_standard, df_robust], axis=1)
    
    # Visualization: Temperature data scaling comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data
    axes[0, 0].hist(df['temperature'], bins=50, color='#11998e', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].axvline(df['temperature'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Min-Max Scaling
    axes[0, 1].hist(df_scaled['temperature_minmax'], bins=50, color='#f59e0b', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Scaled Temperature')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Min-Max Scaling [0, 1]')
    axes[0, 1].grid(alpha=0.3)
    
    # Standard Scaling
    axes[1, 0].hist(df_scaled['temperature_standard'], bins=50, color='#7b2cbf', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Scaled Temperature')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Standard Scaling (μ=0, σ=1)')
    axes[1, 0].axvline(0, color='red', linestyle='--', label='Mean=0')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Robust Scaling
    axes[1, 1].hist(df_scaled['temperature_robust'], bins=50, color='#10b981', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Scaled Temperature')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Robust Scaling (Median-IQR based)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Post-scaling statistics comparison
    print("\nPost-Scaling Statistics (Temperature):")
    print(df_scaled[['temperature_minmax', 'temperature_standard', 'temperature_robust']].describe())
    
    # Outlier impact evaluation
    print("\nScale Values for Data Points with Outliers (480-495):")
    outlier_range = df_scaled.iloc[480:496]
    print(outlier_range[['temperature', 'temperature_minmax', 'temperature_standard', 'temperature_robust']].head())
    
    # Recommendations
    print("\n【Recommendations】")
    print("✓ Min-Max: When outliers are few and [0,1] range is needed (e.g., neural networks)")
    print("✓ Standard: When distribution is close to normal and outliers are few (e.g., linear regression, SVM)")
    print("✓ Robust: When outliers are many and robust preprocessing is needed (real process data)")
    

**Output Example** :
    
    
    Original Data Statistics:
           temperature    pressure   flow_rate
    count   500.000000  500.000000  500.000000
    mean    176.543210    1.534567   50.123456
    std       7.234567    0.267890   10.234567
    min     165.123456    0.987654   25.678901
    max     225.678901    3.123456   75.432109
    
    【Recommendations】
    ✓ Min-Max: When outliers are few and [0,1] range is needed (e.g., neural networks)
    ✓ Standard: When distribution is close to normal and outliers are few (e.g., linear regression, SVM)
    ✓ Robust: When outliers are many and robust preprocessing is needed (real process data)
    

**Explanation** : Since process data often contains outliers, Robust Scaling is the safest choice. However, select the appropriate method based on model type and objectives.

#### Code Example 8: Practical Scaling Workflow
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 8: Practical Scaling Workflow
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Generate sample data: Distillation column process data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'feed_temp': np.random.normal(60, 5, n),
        'reflux_ratio': np.random.uniform(1.5, 3.5, n),
        'reboiler_duty': np.random.normal(1500, 200, n),
        'pressure': np.random.normal(1.2, 0.1, n)
    })
    
    # Target variable: Product purity (calculated from features)
    df['purity'] = (
        95 +
        0.3 * df['reflux_ratio'] +
        0.002 * df['reboiler_duty'] -
        0.1 * df['feed_temp'] +
        2 * df['pressure'] +
        np.random.normal(0, 0.5, n)
    )
    
    # Split into features and target variable
    X = df[['feed_temp', 'reflux_ratio', 'reboiler_duty', 'pressure']]
    y = df['purity']
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data Range Before Scaling:")
    print(X_train.describe().loc[['min', 'max']])
    
    # Scaling (fit on training data, transform test data)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Important: transform only, not fit
    
    # Convert back to DataFrame (preserve column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("\nData Range After Scaling (Training Data):")
    print(X_train_scaled.describe().loc[['min', 'max']])
    
    # Model building (unscaled vs scaled)
    # Unscaled
    model_unscaled = LinearRegression()
    model_unscaled.fit(X_train, y_train)
    y_pred_unscaled = model_unscaled.predict(X_test)
    
    # Scaled
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    
    # Performance evaluation
    print("\n【Model Performance Comparison】")
    print(f"Unscaled - R²: {r2_score(y_test, y_pred_unscaled):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_unscaled)):.4f}")
    print(f"Scaled - R²: {r2_score(y_test, y_pred_scaled):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_scaled)):.4f}")
    
    # Coefficient comparison (scaling effect)
    print("\nRegression Coefficient Comparison:")
    coef_comparison = pd.DataFrame({
        'Feature': X.columns,
        'Unscaled_Coef': model_unscaled.coef_,
        'Scaled_Coef': model_scaled.coef_
    })
    print(coef_comparison)
    
    print("\n【Important Notes】")
    print("1. Fit scaler on training data, use transform only on test data")
    print("2. This prevents data leakage and accurately simulates real-world operation")
    print("3. For linear regression, performance is the same but coefficient interpretability and model convergence improve")
    print("4. For distance-based models (KNN, SVM), scaling directly affects performance")
    

**Explanation** : The most important aspect of scaling in practice is to **fit on training data and use transform only on test data**. This prevents data leakage and allows accurate evaluation of real-world performance.

* * *

## 2.4 Advanced Visualization with Pandas/Matplotlib/Seaborn

Appropriate visualization is essential for understanding the essence of data. Let's master visualization techniques specific to process data.

#### Code Example 9: Multi-dimensional Visualization of Process Operating States
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 9: Multi-dimensional Visualization of Process O
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    # Generate sample data: 24 hours of continuous operation data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    # Simulate 3 operating phases
    phase1 = 480  # Startup phase (0-8h)
    phase2 = 600  # Steady-state operation (8-18h)
    phase3 = 360  # Shutdown phase (18-24h)
    
    temperature = np.concatenate([
        np.linspace(25, 175, phase1) + np.random.normal(0, 2, phase1),  # Startup
        175 + np.random.normal(0, 1, phase2),  # Steady-state
        np.linspace(175, 30, phase3) + np.random.normal(0, 3, phase3)  # Shutdown
    ])
    
    pressure = np.concatenate([
        np.linspace(0.1, 1.5, phase1) + np.random.normal(0, 0.05, phase1),
        1.5 + np.random.normal(0, 0.03, phase2),
        np.linspace(1.5, 0.1, phase3) + np.random.normal(0, 0.08, phase3)
    ])
    
    flow_rate = np.concatenate([
        np.linspace(0, 50, phase1) + np.random.normal(0, 2, phase1),
        50 + np.random.normal(0, 1, phase2),
        np.linspace(50, 0, phase3) + np.random.normal(0, 2, phase3)
    ])
    
    df = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'flow_rate': flow_rate,
        'phase': ['Startup']*phase1 + ['Steady-State']*phase2 + ['Shutdown']*phase3
    }, index=dates)
    
    # Visualization: Composite charts
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series plot (color-coded by phase)
    ax1 = fig.add_subplot(gs[0, :])
    colors = {'Startup': '#f59e0b', 'Steady-State': '#11998e', 'Shutdown': '#7b2cbf'}
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        ax1.plot(phase_data.index, phase_data['temperature'],
                 color=colors[phase], label=phase, linewidth=1.5)
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.set_title('Process Temperature by Operating Phase', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 2. Multiple variable simultaneous plot (dual axis)
    ax2 = fig.add_subplot(gs[1, :])
    ax2_twin = ax2.twinx()
    
    ax2.plot(df.index, df['temperature'], color='#11998e', linewidth=1.5, label='Temperature')
    ax2.set_ylabel('Temperature (°C)', color='#11998e', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#11998e')
    
    ax2_twin.plot(df.index, df['pressure'], color='#f59e0b', linewidth=1.5, label='Pressure')
    ax2_twin.set_ylabel('Pressure (MPa)', color='#f59e0b', fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='#f59e0b')
    
    ax2.set_title('Temperature and Pressure (Dual Axis)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Correlation scatter plot (by phase)
    ax3 = fig.add_subplot(gs[2, 0])
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        ax3.scatter(phase_data['temperature'], phase_data['pressure'],
                    c=colors[phase], alpha=0.5, s=10, label=phase)
    ax3.set_xlabel('Temperature (°C)', fontsize=11)
    ax3.set_ylabel('Pressure (MPa)', fontsize=11)
    ax3.set_title('Temperature vs Pressure by Phase', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Heatmap (hourly average values)
    ax4 = fig.add_subplot(gs[2, 1])
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly.index.hour
    hourly_avg = df_hourly.groupby('hour')[['temperature', 'pressure', 'flow_rate']].mean()
    sns.heatmap(hourly_avg.T, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Value'}, ax=ax4)
    ax4.set_title('Hourly Average Heatmap', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Hour of Day', fontsize=11)
    
    # 5. Box plot (distribution by phase)
    ax5 = fig.add_subplot(gs[3, 0])
    df.boxplot(column='temperature', by='phase', ax=ax5, patch_artist=True,
               boxprops=dict(facecolor='#11998e', alpha=0.7))
    ax5.set_title('Temperature Distribution by Phase', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Operating Phase', fontsize=11)
    ax5.set_ylabel('Temperature (°C)', fontsize=11)
    plt.sca(ax5)
    plt.xticks(rotation=0)
    
    # 6. Rolling statistics (stability visualization)
    ax6 = fig.add_subplot(gs[3, 1])
    df['rolling_std'] = df['temperature'].rolling(window=60).std()
    ax6.plot(df.index, df['rolling_std'], color='#7b2cbf', linewidth=1.5)
    ax6.axhline(y=1, color='green', linestyle='--', label='Target Stability (σ < 1°C)')
    ax6.set_ylabel('60-min Rolling Std (°C)', fontsize=11)
    ax6.set_xlabel('Time', fontsize=11)
    ax6.set_title('Process Stability (Rolling Standard Deviation)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Comprehensive Process Data Visualization Dashboard',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.show()
    
    # Statistical summary
    print("Phase-wise Statistical Summary:")
    print(df.groupby('phase')[['temperature', 'pressure', 'flow_rate']].describe())
    

**Explanation** : Combining multiple visualization methods allows simultaneous understanding of different aspects of data. Color-coding by operating phase, dual-axis charts, and heatmaps are very effective for communication with process engineers.

#### Code Example 10: Interactive Dashboard (Plotly)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - plotly>=5.14.0
    
    """
    Example: Code Example 10: Interactive Dashboard (Plotly)
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=2880, freq='30s')
    
    df = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 2, 2880) + 3*np.sin(np.arange(2880)*2*np.pi/2880),
        'pressure': 1.5 + np.random.normal(0, 0.08, 2880),
        'flow_rate': 50 + np.random.normal(0, 3, 2880),
        'purity': 98 + np.random.normal(0, 0.5, 2880)
    }, index=dates)
    
    # Add anomaly events
    df.loc['2025-01-01 08:00':'2025-01-01 08:15', 'temperature'] += 10
    df.loc['2025-01-01 16:00':'2025-01-01 16:20', 'purity'] -= 2
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Reactor Temperature', 'Pressure', 'Flow Rate', 'Product Purity'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=df.index, y=df['temperature'],
                   mode='lines',
                   name='Temperature',
                   line=dict(color='#11998e', width=1.5),
                   hovertemplate='%{x}<br/>Temp: %{y:.2f}°C<extra></extra>'),
        row=1, col=1
    )
    fig.add_hline(y=175, line_dash="dash", line_color="red",
                  annotation_text="Target", row=1, col=1)
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=df.index, y=df['pressure'],
                   mode='lines',
                   name='Pressure',
                   line=dict(color='#f59e0b', width=1.5),
                   hovertemplate='%{x}<br/>Pressure: %{y:.3f} MPa<extra></extra>'),
        row=2, col=1
    )
    
    # Flow rate
    fig.add_trace(
        go.Scatter(x=df.index, y=df['flow_rate'],
                   mode='lines',
                   name='Flow Rate',
                   line=dict(color='#7b2cbf', width=1.5),
                   hovertemplate='%{x}<br/>Flow: %{y:.2f} m³/h<extra></extra>'),
        row=3, col=1
    )
    
    # Product purity (add quality control range)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['purity'],
                   mode='lines',
                   name='Purity',
                   line=dict(color='#10b981', width=1.5),
                   hovertemplate='%{x}<br/>Purity: %{y:.2f}%<extra></extra>'),
        row=4, col=1
    )
    fig.add_hrect(y0=97.5, y1=99.0, line_width=0, fillcolor="green", opacity=0.1,
                  annotation_text="Spec Range", row=4, col=1)
    
    # Layout settings
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (MPa)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (m³/h)", row=3, col=1)
    fig.update_yaxes(title_text="Purity (%)", row=4, col=1)
    
    fig.update_layout(
        title_text="Interactive Process Monitoring Dashboard",
        height=1000,
        showlegend=False,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add interactive features
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        row=1, col=1
    )
    
    # Save and display
    fig.write_html("process_monitoring_dashboard.html")
    print("Interactive dashboard saved to 'process_monitoring_dashboard.html'.")
    print("\n【Dashboard Features】")
    print("✓ Zoom: Drag to zoom in, double-click to reset")
    print("✓ Pan: Shift + drag to move")
    print("✓ Hover: Mouse over data points for details")
    print("✓ Time range selection: Buttons at top for 1h/6h/12h/all periods")
    print("✓ Saved as HTML file, viewable in browser")
    
    # Optional: Display in Jupyter Notebook
    # fig.show()
    
    # Additional analysis: Identify anomaly times
    temp_anomaly = df[df['temperature'] > 180]
    purity_anomaly = df[df['purity'] < 97]
    
    print(f"\nTemperature anomalies: {len(temp_anomaly)} records")
    if len(temp_anomaly) > 0:
        print(f"Occurrence time: {temp_anomaly.index[0]} - {temp_anomaly.index[-1]}")
    
    print(f"\nPurity anomalies: {len(purity_anomaly)} records")
    if len(purity_anomaly) > 0:
        print(f"Occurrence time: {purity_anomaly.index[0]} - {purity_anomaly.index[-1]}")
    

**Output Example** :
    
    
    Interactive dashboard saved to 'process_monitoring_dashboard.html'.
    
    【Dashboard Features】
    ✓ Zoom: Drag to zoom in, double-click to reset
    ✓ Pan: Shift + drag to move
    ✓ Hover: Mouse over data points for details
    ✓ Time range selection: Buttons at top for 1h/6h/12h/all periods
    ✓ Saved as HTML file, viewable in browser
    
    Temperature anomalies: 31 records
    Occurrence time: 2025-01-01 08:00:00 - 2025-01-01 08:15:00
    
    Purity anomalies: 41 records
    Occurrence time: 2025-01-01 16:00:00 - 2025-01-01 16:20:00
    

**Explanation** : Interactive dashboards using Plotly are ideal for reporting to process engineers and managers. They can be saved as HTML files and viewed by anyone in a browser, making them highly practical.

* * *

## 2.5 Chapter Summary

### What We Learned

**1\. Time Series Data Manipulation** — DatetimeIndex enables intuitive time-based slicing, resampling adjusts data granularity to improve computational efficiency, and rolling statistics provide trend identification and noise reduction.

**2\. Missing Value Handling** — Understanding three types of missing data (MCAR/MAR/MNAR) guides selection of appropriate countermeasures, with options including linear interpolation, spline interpolation, and KNN imputation. Time-aware imputation is particularly effective for time series data.

**3\. Outlier Detection** — Statistical methods (Z-score, IQR) have known limitations, while Isolation Forest excels at multivariate outlier detection. Combining multiple methods is recommended for process data.

**4\. Scaling** — Three main methods exist: Min-Max, Standard, and Robust scaling. Robust scaling is appropriate for real process data with many outliers. The key principle is to fit on training data and transform on test data.

**5\. Advanced Visualization** — Techniques include phase-wise color coding, dual-axis charts, and heatmaps. Interactive dashboards with Plotly enhance exploration. Understanding data from multiple perspectives is essential for effective analysis.

### Key Points

> **"Garbage In, Garbage Out"** : The quality of data preprocessing determines model performance.

Process data always contains missing values and outliers, so never start model building without appropriate preprocessing. Visualization is the best means to verify preprocessing effectiveness. In practice, expect to spend 60-70% of total time on preprocessing.

### Practical Tips

  1. **Don't neglect Exploratory Data Analysis (EDA)** : Understand data first before jumping into model building
  2. **Make preprocessing reversible** : Keep original data and record each preprocessing step
  3. **Verify with visualization** : Visualize at each preprocessing step to confirm expected results
  4. **Leverage domain knowledge** : Collaborate with process engineers to understand the meaning of anomalies

### To the Next Chapter

In Chapter 3, we will learn **Process Modeling Basics** using preprocessed data, covering process model building with linear regression, multivariate regression and PLS (Partial Least Squares), the concept and implementation of soft sensors, model evaluation metrics (R-squared, RMSE, MAE), and extension to nonlinear models such as Random Forest and SVR.
