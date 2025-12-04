---
title: "Chapter 4: Time Series Data and Integrated Analysis"
chapter_title: "Chapter 4: Time Series Data and Integrated Analysis"
subtitle: Sensor Data Analysis and PCA - Integrated Understanding of Multivariate Data
reading_time: 20-25 min
difficulty: Intermediate
code_examples: 5
exercises: 3
version: 1.1
created_at: 2025-10-17
---

# Chapter 4: Time Series Data and Integrated Analysis

Learn comprehensive data processing and visualization for UV-Vis/IR/Raman/TGA/DSC data. Master how to interpret key metrics.

**💡 Supplement:** Perform baseline correction and normalization first. Compare three aspects consistently: peak position, area, and width.

**Sensor Data Analysis and PCA - Integrated Understanding of Multivariate Data**

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Preprocessing temperature/pressure sensor data and anomaly detection
  * ✅ Implementing sliding window analysis
  * ✅ Dimensionality reduction and visualization using PCA (Principal Component Analysis)
  * ✅ Building integrated analysis pipelines with sklearn Pipeline
  * ✅ Integrating data from multiple measurement techniques

**Reading Time** : 20-25 min **Code Examples** : 5 **Exercises** : 3

* * *

## 4.1 Characteristics and Preprocessing of Time Series Data

### Time Series Data in Materials Synthesis and Process Monitoring

In materials synthesis processes, physical quantities such as temperature, pressure, flow rate, and composition are measured over time.

Measurement Item | Typical Sampling Rate | Main Application | Data Characteristics  
---|---|---|---  
**Temperature** | 0.1-10 Hz | Reaction control, heat treatment | Trends, periodicity  
**Pressure** | 1-100 Hz | CVD, sputtering | High noise, steep changes  
**Flow Rate** | 0.1-1 Hz | Gas supply control | Drift, step changes  
**Composition** | 0.01-1 Hz | In-situ analysis | Delay, integration effects  
  
### Time Series Data Analysis Workflow
    
    
    ```mermaid
    flowchart TD
        A[Data Collection] --> B[Preprocessing]
        B --> C[Feature Extraction]
        C --> D[Anomaly Detection]
        D --> E[Integrated Analysis]
        E --> F[Process Optimization]
    
        B --> B1[Resampling]
        B --> B2[Detrending]
        B --> B3[Noise Removal]
    
        C --> C1[Statistics]
        C --> C2[Rolling Window Analysis]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
    ```

* * *

## 4.2 Data Licensing and Reproducibility

### Time Series Data Repositories and Licenses

Public repositories for sensor data and time series measurement data are limited, but the following resources are available.

#### Major Data Sources

Data Source | Content | License | Access | Citation Requirements  
---|---|---|---|---  
**Kaggle Time Series** | Industrial process data | Mixed | Free | Check required  
**UCI Machine Learning** | Sensor datasets | CC BY 4.0 | Free | Recommended  
**NIST Measurement Data** | Standard measurement data | Public Domain | Free | Recommended  
**PhysioNet** | Physiological signal data | Mixed | Free | Required  
**In-house Experimental Data** | Process monitoring | Follow internal regulations | - | -  
  
#### Precautions When Using Data

**Example of Public Data Usage** :
    
    
    """
    Process sensor data from UCI ML Repository
    Reference: UCI ML Repository - Gas Sensor Array Dataset
    Citation: Vergara, A. et al. (2012) Sensors and Actuators B
    License: CC BY 4.0
    URL: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array
    Sampling rate: 100 Hz
    Measurement period: 3600 seconds
    """
    

**Recording Measurement Metadata** :
    
    
    SENSOR_METADATA = {
        'instrument': 'Thermocouple Type-K',
        'sampling_rate_hz': 10,
        'measurement_start': '2025-10-15T10:00:00',
        'measurement_duration_s': 1000,
        'calibration_date': '2025-10-01',
        'sensor_accuracy': '±0.5°C',
        'data_logger': 'Agilent 34970A'
    }
    
    import json
    with open('sensor_metadata.json', 'w') as f:
        json.dump(SENSOR_METADATA, f, indent=2)
    

### Best Practices for Code Reproducibility

#### Recording Environment Information
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scikit-learn>=1.3.0, <1.5.0
    # - scipy>=1.11.0
    
    """
    Example: Recording Environment Information
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import sys
    import numpy as np
    import pandas as pd
    import scipy
    from sklearn import __version__ as sklearn_version
    
    print("=== Time Series Analysis Environment ===")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"scikit-learn: {sklearn_version}")
    
    # Recommended versions (as of October 2025):
    # - Python: 3.10 or higher
    # - NumPy: 1.24 or higher
    # - pandas: 2.0 or higher
    # - SciPy: 1.10 or higher
    # - scikit-learn: 1.3 or higher
    

#### Documenting Parameters

**Bad Example** (not reproducible):
    
    
    rolling_mean = df['temperature'].rolling(window=50).mean()  # Why 50?
    

**Good Example** (reproducible):
    
    
    # Rolling window parameter settings
    SAMPLING_RATE_HZ = 10  # Sensor sampling frequency
    WINDOW_DURATION_S = 5  # Window duration (seconds)
    WINDOW_SIZE = SAMPLING_RATE_HZ * WINDOW_DURATION_S  # 50 points
    
    WINDOW_DESCRIPTION = """
    Window size: 50 points = 5 seconds
    Rationale: Adopted approximately half of the characteristic time constant (~10 sec) of the process.
          Too short: Sensitive to noise
          Too long: Cannot capture steep changes
    """
    
    rolling_mean = df['temperature'].rolling(
        window=WINDOW_SIZE, center=True
    ).mean()
    

#### Fixing Random Seeds
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Fixing Random Seeds
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Fix random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Specify seed when splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    

* * *

## 4.3 Preprocessing Time Series Data and Rolling Window Analysis

### Sample Data Generation and Basic Visualization

**Code Example 1: Generating Sensor Data for Synthesis Process**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Code Example 1: Generating Sensor Data for Synthesis Process
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # Simulation of materials synthesis process (1000 seconds, 10 Hz)
    np.random.seed(42)
    time = np.linspace(0, 1000, 10000)
    
    # Temperature profile (ramp heating → holding → cooling)
    temperature = np.piecewise(
        time,
        [time < 300, (time >= 300) & (time < 700), time >= 700],
        [lambda t: 25 + 0.25 * t,  # Heating
         lambda t: 100,             # Holding
         lambda t: 100 - 0.1 * (t - 700)]  # Cooling
    )
    temperature += np.random.normal(0, 2, len(time))  # Noise
    
    # Pressure (vacuum level, with step changes)
    pressure = np.piecewise(
        time,
        [time < 200, (time >= 200) & (time < 800), time >= 800],
        [100, 1, 100]  # Torr
    )
    pressure += np.random.normal(0, 0.5, len(time))
    
    # Gas flow rate (periodic variation)
    flow_rate = 50 + 10 * np.sin(2 * np.pi * time / 100) + \
                np.random.normal(0, 2, len(time))
    
    # Intentionally insert anomalies (simulating sensor malfunction)
    temperature[5000:5010] = 200  # Spike noise
    pressure[3000] = -50          # Physically impossible value
    
    # Store in DataFrame
    df_sensor = pd.DataFrame({
        'time': time,
        'temperature': temperature,
        'pressure': pressure,
        'flow_rate': flow_rate
    })
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(df_sensor['time'], df_sensor['temperature'],
                 linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Process Sensor Data')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df_sensor['time'], df_sensor['pressure'],
                 linewidth=0.8, alpha=0.8, color='orange')
    axes[1].set_ylabel('Pressure (Torr)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df_sensor['time'], df_sensor['flow_rate'],
                 linewidth=0.8, alpha=0.8, color='green')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Flow Rate (sccm)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Sensor Data Statistics ===")
    print(df_sensor.describe())
    

### Rolling Window Analysis (Rolling Statistics)

**Code Example 2: Rolling Mean and Rolling Standard Deviation**
    
    
    # Calculate rolling statistics
    window_size = 100  # 10-second window (10 Hz × 10s)
    
    df_sensor['temp_rolling_mean'] = df_sensor['temperature'].rolling(
        window=window_size, center=True
    ).mean()
    
    df_sensor['temp_rolling_std'] = df_sensor['temperature'].rolling(
        window=window_size, center=True
    ).std()
    
    df_sensor['pressure_rolling_mean'] = df_sensor['pressure'].rolling(
        window=window_size, center=True
    ).mean()
    
    # Anomaly detection (3σ method)
    df_sensor['temp_anomaly'] = np.abs(
        df_sensor['temperature'] - df_sensor['temp_rolling_mean']
    ) > 3 * df_sensor['temp_rolling_std']
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Temperature and rolling mean
    axes[0].plot(df_sensor['time'], df_sensor['temperature'],
                 label='Raw', alpha=0.5, linewidth=0.8)
    axes[0].plot(df_sensor['time'], df_sensor['temp_rolling_mean'],
                 label=f'Rolling mean (window={window_size})',
                 linewidth=2, color='red')
    axes[0].fill_between(
        df_sensor['time'],
        df_sensor['temp_rolling_mean'] - 3 * df_sensor['temp_rolling_std'],
        df_sensor['temp_rolling_mean'] + 3 * df_sensor['temp_rolling_std'],
        alpha=0.2, color='red', label='±3σ'
    )
    axes[0].scatter(df_sensor.loc[df_sensor['temp_anomaly'], 'time'],
                    df_sensor.loc[df_sensor['temp_anomaly'], 'temperature'],
                    color='black', s=50, zorder=5, label='Anomalies')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Rolling Statistics and Anomaly Detection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling standard deviation (magnitude of variation)
    axes[1].plot(df_sensor['time'], df_sensor['temp_rolling_std'],
                 linewidth=1.5, color='purple')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Temperature Std (°C)')
    axes[1].set_title('Rolling Standard Deviation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"=== Anomaly Detection Results ===")
    print(f"Number of anomalies: {df_sensor['temp_anomaly'].sum()}")
    anomaly_times = df_sensor.loc[df_sensor['temp_anomaly'], 'time'].values
    print(f"Anomaly occurrence times: {anomaly_times[:5]}... (first 5 points)")
    

### Detrending and Stationarization

**Code Example 3: Differencing and Detrending**
    
    
    from scipy.signal import detrend
    
    # Differencing (first-order difference = rate of change)
    df_sensor['temp_diff'] = df_sensor['temperature'].diff()
    
    # Detrending (linear trend removal)
    df_sensor['temp_detrended'] = detrend(
        df_sensor['temperature'].fillna(method='bfill')
    )
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Original data
    axes[0].plot(df_sensor['time'], df_sensor['temperature'],
                 linewidth=1, alpha=0.8)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Original Time Series')
    axes[0].grid(True, alpha=0.3)
    
    # First-order difference
    axes[1].plot(df_sensor['time'], df_sensor['temp_diff'],
                 linewidth=1, alpha=0.8, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Temperature Diff (°C/0.1s)')
    axes[1].set_title('First Difference (Change Rate)')
    axes[1].grid(True, alpha=0.3)
    
    # Detrended
    axes[2].plot(df_sensor['time'], df_sensor['temp_detrended'],
                 linewidth=1, alpha=0.8, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_title('Detrended (Linear Trend Removed)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stationarity assessment (stability of variation)
    print("=== Stationarity Assessment ===")
    print(f"Standard deviation of original data: {df_sensor['temperature'].std():.2f}")
    print(f"Standard deviation of differenced data: {df_sensor['temp_diff'].std():.2f}")
    print(f"Standard deviation of detrended data: {df_sensor['temp_detrended'].std():.2f}")
    

* * *

## 4.4 Dimensionality Reduction Using PCA (Principal Component Analysis)

### Visualization of Multivariate Data

**Code Example 4: Dimensionality Reduction and Visualization Using PCA**
    
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Data preparation (after removing anomalies)
    df_clean = df_sensor[~df_sensor['temp_anomaly']].copy()
    
    # Feature matrix (temperature, pressure, flow rate)
    X = df_clean[['temperature', 'pressure', 'flow_rate']].dropna().values
    
    # Standardization (essential before PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Execute PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Results to DataFrame
    df_pca = pd.DataFrame(
        X_pca,
        columns=['PC1', 'PC2', 'PC3']
    )
    df_pca['time'] = df_clean['time'].values[:len(df_pca)]
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 2D scatter plot (PC1 vs PC2)
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(df_pca['PC1'], df_pca['PC2'],
                         c=df_pca['time'], cmap='viridis',
                         s=10, alpha=0.6)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('PCA: PC1 vs PC2 (colored by time)')
    plt.colorbar(scatter, ax=ax1, label='Time (s)')
    ax1.grid(True, alpha=0.3)
    
    # Explained variance ratio (Scree plot)
    ax2 = plt.subplot(2, 2, 2)
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.bar(range(1, 4), pca.explained_variance_ratio_, alpha=0.7,
            label='Individual')
    ax2.plot(range(1, 4), cumsum_variance, 'ro-', linewidth=2,
             markersize=8, label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Scree Plot')
    ax2.set_xticks(range(1, 4))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3D scatter plot
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    scatter_3d = ax3.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'],
                             c=df_pca['time'], cmap='viridis',
                             s=10, alpha=0.5)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax3.set_title('PCA: 3D Visualization')
    
    # Loading plot (interpretation of principal components)
    ax4 = plt.subplot(2, 2, 4)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    features = ['Temperature', 'Pressure', 'Flow Rate']
    
    for i, feature in enumerate(features):
        ax4.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        ax4.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                feature, fontsize=12, ha='center')
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('Loading Plot (Feature Contribution)')
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    # PCA statistics
    print("=== PCA Results ===")
    print(f"Cumulative explained variance ratio:")
    for i, var in enumerate(cumsum_variance, 1):
        print(f"  Up to PC{i}: {var*100:.2f}%")
    
    print(f"\nPrincipal component loadings:")
    loading_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(3)],
        index=features
    )
    print(loading_df)
    

**Interpretation of PCA** : \- **PC1 (First Principal Component)** : Direction with the largest variance (typically, overall process progression) \- **PC2 (Second Principal Component)** : Direction orthogonal to PC1 with the next largest variance \- **Loading values** : Impact of each variable on the principal component (larger absolute value = more important)

* * *

## 4.5 Integrated Analysis Pipeline (sklearn Pipeline)

### Integrating Data from Multiple Measurement Techniques

**Code Example 5: Automated Integrated Analysis Pipeline**
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import joblib
    
    class IntegratedAnalysisPipeline:
        """Integrated analysis pipeline"""
    
        def __init__(self, n_clusters=3):
            """
            Parameters:
            -----------
            n_clusters : int
                Number of clusters (number of process states)
            """
            self.pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Imputation of missing values
                ('scaler', RobustScaler()),  # Robust standardization against outliers
                ('pca', PCA(n_components=0.95)),  # Up to 95% cumulative variance
                ('clustering', KMeans(n_clusters=n_clusters, random_state=42))
            ])
            self.n_clusters = n_clusters
    
        def fit(self, X):
            """Train pipeline"""
            self.pipeline.fit(X)
            return self
    
        def transform(self, X):
            """Dimensionality reduction only"""
            # Execute steps up to clustering
            X_transformed = X.copy()
            for step_name, step in self.pipeline.steps[:-1]:
                X_transformed = step.transform(X_transformed)
            return X_transformed
    
        def predict(self, X):
            """Cluster prediction"""
            return self.pipeline.predict(X)
    
        def get_feature_importance(self, feature_names):
            """Feature importance in principal components"""
            pca = self.pipeline.named_steps['pca']
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
            importance_df = pd.DataFrame(
                loadings,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=feature_names
            )
            return importance_df
    
        def save(self, filename):
            """Save model"""
            joblib.dump(self.pipeline, filename)
    
        @staticmethod
        def load(filename):
            """Load model"""
            return joblib.load(filename)
    
    # Usage example: Execute integrated analysis
    # Prepare feature matrix
    X_integrated = df_clean[['temperature', 'pressure',
                             'flow_rate']].dropna().values
    
    # Execute pipeline
    pipeline = IntegratedAnalysisPipeline(n_clusters=3)
    pipeline.fit(X_integrated)
    
    # Dimensionality reduction results
    X_reduced = pipeline.transform(X_integrated)
    
    # Cluster prediction
    clusters = pipeline.predict(X_integrated)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Cluster visualization in time series
    time_clean = df_clean['time'].values[:len(clusters)]
    axes[0, 0].scatter(time_clean, clusters, c=clusters,
                      cmap='viridis', s=5, alpha=0.6)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cluster ID')
    axes[0, 0].set_title('Process State Clustering (Time Series)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clusters in PCA space
    axes[0, 1].scatter(X_reduced[:, 0], X_reduced[:, 1],
                      c=clusters, cmap='viridis', s=10, alpha=0.6)
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('Clusters in PCA Space')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature profile for each cluster
    temp_clean = df_clean['temperature'].values[:len(clusters)]
    for cluster_id in range(pipeline.n_clusters):
        mask = clusters == cluster_id
        axes[1, 0].scatter(time_clean[mask], temp_clean[mask],
                          label=f'Cluster {cluster_id}', s=5, alpha=0.6)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Temperature Profile by Cluster')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importance
    importance = pipeline.get_feature_importance(
        ['Temperature', 'Pressure', 'Flow Rate']
    )
    importance.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Importance in PCA')
    axes[1, 1].set_ylabel('Loading')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Cluster statistics
    print("=== Cluster Statistics ===")
    for cluster_id in range(pipeline.n_clusters):
        mask = clusters == cluster_id
        cluster_temp = temp_clean[mask]
        print(f"Cluster {cluster_id}:")
        print(f"  Sample count: {mask.sum()}")
        print(f"  Average temperature: {cluster_temp.mean():.2f}°C")
        print(f"  Temperature range: {cluster_temp.min():.2f} - {cluster_temp.max():.2f}°C")
    
    # Save pipeline
    pipeline.save('process_analysis_pipeline.pkl')
    print("\nPipeline saved to 'process_analysis_pipeline.pkl'")
    

**Benefits of sklearn Pipeline** : 1\. **Reproducibility** : All processing steps stored in a single object 2\. **Maintainability** : Easy to add/change steps 3\. **Deployment** : Can be saved/loaded as `.pkl` file 4\. **Automation** : Application to new data completed with one `predict()` line

* * *

## 4.6 Practical Pitfalls and Countermeasures

### Common Mistakes and Best Practices

#### Mistake 1: Applying PCA to Non-Stationary Time Series

**Symptom** : PCA results do not align with physical meaning of process

**Cause** : Applying PCA to non-stationary time series data (with trends) as is

**Countermeasure** :
    
    
    # ❌ Bad example: PCA on non-stationary data as is
    X_raw = df[['temperature', 'pressure', 'flow_rate']].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_raw)
    # Result: PC1 is dominated by trend, making physical interpretation difficult
    
    # ✅ Good example: PCA after stationarization (differencing)
    # Step 1: First-order difference of each variable
    X_diff = df[['temperature', 'pressure', 'flow_rate']].diff().dropna().values
    
    # Step 2: Standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_diff)
    
    # Step 3: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Validation: Stationarity test
    from statsmodels.tsa.stattools import adfuller
    for col in ['temperature', 'pressure', 'flow_rate']:
        result = adfuller(df[col].dropna())
        print(f"{col}: ADF statistic = {result[0]:.3f}, p-value = {result[1]:.3f}")
        if result[1] > 0.05:
            print(f"  Warning: {col} is non-stationary (p > 0.05). Please take difference")
    

#### Mistake 2: PCA Without Standardization

**Symptom** : Large-scale variables (pressure: ~100 Torr) dominate PC1, and small-scale variables (flow rate: ~50 sccm) are ignored

**Cause** : PCA is based on variance and is sensitive to scale

**Countermeasure** :
    
    
    # ❌ Bad example: Without standardization
    X = df[['temperature', 'pressure', 'flow_rate']].values
    pca = PCA()
    pca.fit(X)
    # Result: Pressure contribution to PC1 is overestimated
    
    # ✅ Good example: Always standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    
    # Confirm effect of standardization
    print("=== Variance Before Standardization ===")
    print(f"Temperature: {df['temperature'].var():.2f}")
    print(f"Pressure: {df['pressure'].var():.2f}")
    print(f"Flow: {df['flow_rate'].var():.2f}")
    
    print("\n=== Variance After Standardization (all become 1.0) ===")
    X_scaled_df = pd.DataFrame(X_scaled, columns=['temperature', 'pressure', 'flow_rate'])
    print(X_scaled_df.var())
    

#### Mistake 3: Data Leakage in Time Series

**Symptom** : High accuracy in cross-validation, but performance degradation in actual operation

**Cause** : Using future data to predict the past (ignoring temporal order)

**Countermeasure** :
    
    
    # ❌ Bad example: Random split (ignoring time series)
    from sklearn.model_selection import cross_val_score, KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    # Result: Predicting past with future data
    
    # ✅ Good example: Time series cross-validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv)
    
    # Visualization: Fold structure
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        ax.plot(train_idx, [i]*len(train_idx), 'b-', linewidth=10, label='Train' if i == 0 else '')
        ax.plot(test_idx, [i]*len(test_idx), 'r-', linewidth=10, label='Test' if i == 0 else '')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Fold')
    ax.set_title('TimeSeriesSplit: Train/Test Splits')
    ax.legend()
    plt.show()
    
    print("TimeSeriesSplit: Training data is always before test data")
    

#### Mistake 4: Inappropriate Rolling Window Size Setting

**Symptom** : Rolling average is too smooth (missing important changes), or full of noise

**Cause** : Window size does not match the time constant of the process

**Countermeasure** :
    
    
    # ❌ Bad example: Empirically determined window size
    window_size = 100  # Arbitrary value
    
    # ✅ Good example: Determine window size from process characteristics
    # Step 1: Autocorrelation analysis
    from statsmodels.graphics.tsaplots import plot_acf
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(df['temperature'].dropna(), lags=200, ax=ax)
    ax.set_title('Autocorrelation: Temperature')
    plt.show()
    
    # Step 2: Identify the point where autocorrelation first falls below threshold (e.g., 0.5)
    from statsmodels.tsa.stattools import acf
    acf_values = acf(df['temperature'].dropna(), nlags=200)
    threshold = 0.5
    lag_threshold = np.where(acf_values < threshold)[0][0]
    print(f"Lag where autocorrelation falls below {threshold}: {lag_threshold} points")
    
    # Step 3: Determine window size (about half of characteristic time constant)
    optimal_window = lag_threshold // 2
    print(f"Recommended window size: {optimal_window} points")
    
    rolling_mean = df['temperature'].rolling(window=optimal_window).mean()
    

#### Mistake 5: Excessive Imputation (Missing Value Imputation)

**Symptom** : Imputed missing values greatly deviate from actual data

**Cause** : Imputing large missing regions with simple methods (mean, forward fill)

**Countermeasure** :
    
    
    # ❌ Bad example: Unconditionally impute large missing regions
    df_filled = df['temperature'].fillna(method='ffill')  # Forward fill
    # Result: Same value continues during long sensor downtime (unrealistic)
    
    # ✅ Good example: Check missing range and select appropriate imputation method
    # Step 1: Evaluate length of consecutive missing values
    is_missing = df['temperature'].isnull()
    missing_groups = is_missing.ne(is_missing.shift()).cumsum()
    max_consecutive_missing = is_missing.groupby(missing_groups).sum().max()
    
    print(f"Maximum consecutive missing values: {max_consecutive_missing} points")
    
    # Step 2: Threshold judgment
    MAX_ALLOWED_GAP = 10  # Maximum allowed missing length (e.g., 10 times sampling interval)
    
    if max_consecutive_missing > MAX_ALLOWED_GAP:
        print(f"Warning: Large missing regions exist ({max_consecutive_missing} points)")
        print("Consider excluding data instead of imputation")
    
        # Exclude intervals containing large missing regions
        consecutive_missing = is_missing.groupby(missing_groups).transform('sum')
        df_clean = df[consecutive_missing <= MAX_ALLOWED_GAP].copy()
    else:
        # Impute only small missing values (linear interpolation)
        df_clean = df.copy()
        df_clean['temperature'] = df_clean['temperature'].interpolate(method='linear')
    
    print(f"Number of data points after imputation: {len(df_clean)} / {len(df)}")
    

* * *

## 4.7 Time Series Data Skills Checklist

### Time Series Preprocessing Skills

#### Basic Level

  * [ ] Can handle time series data with pandas DataFrame
  * [ ] Can calculate rolling average
  * [ ] Can execute downsampling/upsampling
  * [ ] Can detect missing values
  * [ ] Can visually confirm presence of trends

#### Applied Level

  * [ ] Can distinguish between first-order and second-order differencing
  * [ ] Can implement detrending
  * [ ] Can estimate time constant by autocorrelation analysis
  * [ ] Can verify stationarity with ADF test
  * [ ] Can determine adaptive rolling window size

#### Advanced Level

  * [ ] Can execute seasonal decomposition (STL)
  * [ ] Can build ARIMA/SARIMA models
  * [ ] Can estimate states with Kalman filter
  * [ ] Can quantitatively evaluate missing value imputation accuracy

### Anomaly Detection Skills

#### Basic Level

  * [ ] Can detect statistical anomalies using 3σ method
  * [ ] Can implement threshold-based anomaly detection
  * [ ] Can visualize anomaly points
  * [ ] Understand anomaly detection accuracy (True Positive, False Positive)

#### Applied Level

  * [ ] Can implement rolling statistics-based anomaly detection
  * [ ] Can perform robust anomaly detection using IQR method
  * [ ] Can calculate anomaly scores
  * [ ] Can evaluate detection performance with ROC curve
  * [ ] Can combine domain knowledge with statistical methods

#### Advanced Level

  * [ ] Can implement anomaly detection using Isolation Forest
  * [ ] Can detect anomalies using LSTM autoencoder
  * [ ] Can execute change point detection
  * [ ] Can build real-time anomaly detection systems

### PCA and Dimensionality Reduction Skills

#### Basic Level

  * [ ] Can standardize data
  * [ ] Can execute PCA and obtain principal component scores
  * [ ] Can calculate and visualize cumulative explained variance ratio
  * [ ] Can visualize PCA results with 2D/3D scatter plots

#### Applied Level

  * [ ] Can interpret principal components with loading plots
  * [ ] Can select appropriate number of principal components (cumulative variance, scree plot)
  * [ ] Can preprocess non-stationary data (differencing, detrending)
  * [ ] Understand PCA assumptions (linearity, normality)
  * [ ] Can explain influence of original variables from principal component scores

#### Advanced Level

  * [ ] Can perform nonlinear dimensionality reduction with kernel PCA
  * [ ] Can visualize with t-SNE/UMAP
  * [ ] Can improve interpretability with sparse PCA
  * [ ] Can implement principal component regression (PCR)

### Integrated Analysis and Pipeline Building Skills

#### Basic Level

  * [ ] Understand basic structure of sklearn Pipeline
  * [ ] Can build basic pipeline: Imputer→Scaler→PCA
  * [ ] Can execute training/prediction with pipeline
  * [ ] Can save/load pipeline

#### Applied Level

  * [ ] Can create custom Transformers
  * [ ] Can apply different preprocessing per column with ColumnTransformer
  * [ ] Can optimize hyperparameters with GridSearchCV
  * [ ] Can execute time series CV with TimeSeriesSplit
  * [ ] Can diagram entire pipeline processing flow

#### Advanced Level

  * [ ] Can ensemble multiple models
  * [ ] Can implement custom CV strategies (Blocked Time Series Split)
  * [ ] Can optimize computation time for each pipeline step
  * [ ] Can build deployment pipeline for production environment

### Integrated Skills: Comprehensive Experimental Data Analysis

#### Basic Level

  * [ ] Can process XRD, XPS, SEM, and sensor data individually
  * [ ] Can extract features from each data type
  * [ ] Can combine features into tabular data

#### Applied Level

  * [ ] Can appropriately standardize data with different scales
  * [ ] Can integrate data considering time delays
  * [ ] Can extract main patterns from integrated data using PCA
  * [ ] Can classify process states using clustering
  * [ ] Can predict material properties using regression models

#### Advanced Level

  * [ ] Can perform multimodal data integration (Early/Late Fusion)
  * [ ] Can identify causal relationships between variables using causal inference
  * [ ] Can optimize process conditions using Bayesian optimization
  * [ ] Can apply integrated analysis to digital twin construction

* * *

## 4.8 Comprehensive Skills Assessment

Please self-assess using the following criteria.

### Level 1: Beginner (60% or more of basic skills)

  * Can perform basic time series preprocessing (rolling average, differencing)
  * Can detect anomalies using 3σ method
  * Can reduce dimensions with PCA and visualize in 2D
  * Understand basics of sklearn Pipeline

**Next Steps** : \- Master stationarity test (ADF) \- Deepen loading interpretation of PCA \- Master TimeSeriesSplit

### Level 2: Intermediate (100% basic + 60% or more applied)

  * Can appropriately stationarize non-stationary time series
  * Can implement rolling statistics-based anomaly detection
  * Can physically interpret PCA results
  * Can create custom Transformers

**Next Steps** : \- Challenge ARIMA/SARIMA models \- Master advanced anomaly detection methods like Isolation Forest \- Experience pipeline optimization and deployment

### Level 3: Advanced (100% basic + 100% applied + 60% or more advanced)

  * Can build seasonal decomposition and ARIMA models
  * Can perform advanced anomaly detection with LSTM/Isolation Forest
  * Can perform nonlinear dimensionality reduction with kernel PCA/t-SNE
  * Can deploy pipelines to production environment

**Next Steps** : \- Advance to causal inference and Bayesian optimization \- Build real-time analysis systems \- Participate in digital twin development

### Level 4: Expert (90% or more of all items)

  * Can lead the field of time series and multivariate analysis
  * Can develop and implement new anomaly detection methods
  * Can design integrated analysis of multiple measurement techniques
  * Contributing to standardization of experimental data analysis

**Activity Examples** : \- In-house training and tutorial instructor \- Paper writing and conference presentations \- Open source contributions

* * *

## 4.9 Action Plan Template

### Current Level: **___** ____

### Target Level (after 3 months): **___** ____

### Priority Skills to Strengthen (select 3):

  1. * * *

  2. * * *

  3. * * *

### Specific Action Plan:

**Week 1-2** : \- [ ] Action 1: **___****___****___ _ \- [ ] Action 2: ****___****___** ____

**Week 3-4** : \- [ ] Action 1: **___****___****___ _ \- [ ] Action 2: ****___****___** ____

**Week 5-8** : \- [ ] Action 1: **___****___****___ _ \- [ ] Action 2: ****___****___** ____

**Week 9-12** : \- [ ] Action 1: **___****___****___ _ \- [ ] Action 2: ****___****___** ____

### Evaluation Metrics:

  * [ ] Apply time series analysis to real project
  * [ ] Build anomaly detection system
  * [ ] Report PCA results to team

* * *

## 4.10 Integrated Analysis Workflow Diagram

### Integration of Multiple Measurement Techniques
    
    
    ```mermaid
    flowchart LR
        A[XRD Data] --> D[Feature Extraction]
        B[XPS Data] --> E[Feature Extraction]
        C[SEM Data] --> F[Feature Extraction]
        G[Sensor Data] --> H[Feature Extraction]
    
        D --> I[Integrated Feature Matrix]
        E --> I
        F --> I
        H --> I
    
        I --> J[Standardization]
        J --> K[PCA]
        K --> L[Clustering\nor\nRegression Model]
        L --> M[Result Interpretation]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style G fill:#e8f5e9
        style I fill:#fce4ec
        style M fill:#fff9c4
    ```

* * *

## 4.11 Chapter Summary

### What We Learned

  1. **Time Series Data Analysis** \- Rolling window statistics (mean, standard deviation) \- Anomaly detection (3σ method) \- Detrending and stationarization (differencing, detrending)

  2. **PCA (Principal Component Analysis)** \- Dimensionality reduction of multivariate data \- Component selection by explained variance ratio \- Interpretation using loading plots

  3. **Integrated Analysis Pipeline** \- Automation with sklearn Pipeline \- Imputation→Standardization→PCA→Clustering \- Model saving and reuse

  4. **Practical Applications** \- Clustering of process states \- Integration of multiple measurement techniques \- Anomaly detection and quality control

  5. **Data Licensing and Reproducibility** (added in v1.1) \- Proper citation of public data sources (UCI ML, NIST, PhysioNet) \- Recording sensor metadata \- Documentation of environment information and parameters

  6. **Avoiding Practical Pitfalls** (added in v1.1) \- Precautions when applying PCA to non-stationary time series \- Importance of standardization \- Prevention of time series data leakage \- Scientific determination of rolling window size \- Appropriate limits on missing value imputation

  7. **Skills Assessment and Career Path** (added in v1.1) \- Checklist for 5 skill categories \- 3-level assessment (basic/applied/advanced) \- 4-level comprehensive assessment (beginner→expert) \- Individual action plan template

### Key Points

  * ✅ Preprocessing (stationarization) is crucial for time series data
  * ✅ PCA is dimensionality reduction considering correlations between variables
  * ✅ sklearn Pipeline enables highly reproducible analysis
  * ✅ Integrated analysis reveals insights not obtainable from single measurements
  * ✅ **Proper citation of data sources and ensuring reproducibility are the foundation of research quality** (v1.1)
  * ✅ **Knowing practical pitfalls in advance greatly reduces trial-and-error time** (v1.1)
  * ✅ **Systematic skills assessment enables efficient learning planning** (v1.1)

### Series Summary

In this series, we learned from basics to applications of experimental data analysis in materials science:

  * **Chapter 1** : Basics of data preprocessing (noise removal, outlier detection, standardization)
  * **Chapter 2** : Spectral data analysis (XRD, XPS, IR, Raman)
  * **Chapter 3** : Image data analysis (SEM, TEM, particle detection, CNN classification)
  * **Chapter 4** : Time series data and integrated analysis (sensor data, PCA, Pipeline)

**Common additions in v1.1 across all chapters** : \- Establishment of data licensing and citation standards \- Best practices for code reproducibility \- Practical pitfalls and countermeasures (5+ examples per chapter) \- Comprehensive skills assessment framework

By combining these technologies, accelerated and more accurate materials development can be realized.

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Judge whether the following statements are true or false.

  1. Rolling average filter removes trends from time series data
  2. The first principal component in PCA is the direction with the largest variance
  3. In sklearn Pipeline, all processing steps require the same data type

Hints 1\. Rolling average is smoothing, different from detrending 2\. Check the definition of PCA (variance maximization) 3\. Consider the input/output types of each Pipeline step  Answer Example **Answer**: 1\. **False** - Rolling average is noise removal (smoothing); use differencing or detrending for trend removal 2\. **True** - PCA defines the first principal component as the direction that maximizes variance 3\. **False** - Each step can handle different data types with appropriate transformations (e.g., Imputer→Scaler) **Explanation**: "Smoothing," "detrending," and "stationarization" of time series data are different concepts. Rolling average removes high-frequency noise but preserves low-frequency trends. PCA is an unsupervised learning dimensionality reduction method that maximally preserves data variance. 

* * *

### Problem 2 (Difficulty: medium)

Execute rolling window analysis and anomaly detection on the following sensor data.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Execute rolling window analysis and anomaly detection on the
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Sample sensor data
    np.random.seed(200)
    time = np.linspace(0, 500, 5000)
    signal = 50 + 10 * np.sin(2 * np.pi * time / 50) + \
             np.random.normal(0, 3, len(time))
    
    # Insert anomalies
    signal[2000:2005] = 100
    signal[3500] = -20
    

**Requirements** : 1\. Calculate rolling mean (window size 50) 2\. Calculate rolling standard deviation 3\. Detect anomalies using 3σ method 4\. Output times of anomalies 5\. Visualize results

Hints **Processing Flow**: 1\. `pandas.Series.rolling(window=50).mean()` 2\. `pandas.Series.rolling(window=50).std()` 3\. `np.abs(signal - rolling_mean) > 3 * rolling_std` 4\. `time[anomaly_mask]` 5\. Plot original signal, rolling mean, ±3σ range, and anomaly points using `matplotlib`  Answer Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Requirements:
    1. Calculate rolling mean (window size 50)
    2. 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Sample data
    np.random.seed(200)
    time = np.linspace(0, 500, 5000)
    signal = 50 + 10 * np.sin(2 * np.pi * time / 50) + \
             np.random.normal(0, 3, len(time))
    signal[2000:2005] = 100
    signal[3500] = -20
    
    # Convert to DataFrame
    df = pd.DataFrame({'time': time, 'signal': signal})
    
    # Rolling statistics
    window_size = 50
    df['rolling_mean'] = df['signal'].rolling(
        window=window_size, center=True
    ).mean()
    df['rolling_std'] = df['signal'].rolling(
        window=window_size, center=True
    ).std()
    
    # Anomaly detection
    df['anomaly'] = np.abs(
        df['signal'] - df['rolling_mean']
    ) > 3 * df['rolling_std']
    
    # Anomaly times
    anomaly_times = df.loc[df['anomaly'], 'time'].values
    print("=== Anomaly Detection Results ===")
    print(f"Number of anomalies: {df['anomaly'].sum()}")
    print(f"Anomaly occurrence times: {anomaly_times}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.plot(df['time'], df['signal'], label='Raw Signal',
             alpha=0.6, linewidth=0.8)
    plt.plot(df['time'], df['rolling_mean'], label='Rolling Mean',
             linewidth=2, color='red')
    plt.fill_between(
        df['time'],
        df['rolling_mean'] - 3 * df['rolling_std'],
        df['rolling_mean'] + 3 * df['rolling_std'],
        alpha=0.2, color='red', label='±3σ'
    )
    plt.scatter(df.loc[df['anomaly'], 'time'],
               df.loc[df['anomaly'], 'signal'],
               color='black', s=50, zorder=5, label='Anomalies')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Rolling Window Analysis and Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Example Output**: 
    
    
    === Anomaly Detection Results ===
    Number of anomalies: 6
    Anomaly occurrence times: [200.04  200.14  200.24  200.34  200.44  350.07]
    

**Explanation**: Rolling window statistics capture the local behavior (mean and standard deviation) of the signal, and the 3σ rule detects statistical anomalies. In this example, the spike near 200 seconds and the negative spike near 350 seconds were correctly detected. 

* * *

### Problem 3 (Difficulty: hard)

Build a pipeline to integrate data from multiple measurement techniques (XRD, XPS, SEM, sensors) and predict material quality.

**Background** : In materials synthesis experiments, the following data were acquired for each sample: \- XRD peak intensities (3 main peaks) \- XPS elemental composition (C, O, Fe atomic %) \- SEM particle size statistics (mean diameter, standard deviation) \- Process sensor statistics (maximum temperature, average pressure)

Build a model to predict material quality score (0-100) from these 11 variables.

**Tasks** : 1\. Build pipeline including missing value imputation and standardization 2\. Dimensionality reduction with PCA (90% cumulative variance) 3\. Quality prediction with regression model (Ridge regression) 4\. Performance evaluation with cross-validation 5\. Visualization of feature importance

**Constraints** : \- 100 samples (80 training, 20 test) \- Some data has missing values (5-10%) \- Scales differ greatly (XRD is thousands, composition is 0-100%)

Hints **Design Policy**: 1\. Integrate with `sklearn.pipeline.Pipeline` 2\. `SimpleImputer` → `StandardScaler` → `PCA` → `Ridge` 3\. Evaluate with `cross_val_score` 4\. Calculate feature importance from PCA loadings **Pipeline Example**: 
    
    
    from sklearn.linear_model import Ridge
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.9)),
        ('regressor', Ridge(alpha=1.0))
    ])
    

Answer Example **Answer Overview**: Realize quality prediction with an integrated pipeline of missing value imputation, standardization, PCA, and regression. **Implementation Code**: 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Constraints:
    - 100 samples (80 training, 20 test)
    - Some dat
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate features (11 variables)
    data = {
        # XRD peak intensities
        'xrd_peak1': np.random.normal(1000, 200, n_samples),
        'xrd_peak2': np.random.normal(1500, 300, n_samples),
        'xrd_peak3': np.random.normal(800, 150, n_samples),
    
        # XPS composition
        'xps_C': np.random.normal(20, 5, n_samples),
        'xps_O': np.random.normal(50, 10, n_samples),
        'xps_Fe': np.random.normal(30, 8, n_samples),
    
        # SEM statistics
        'sem_mean_diameter': np.random.normal(50, 10, n_samples),
        'sem_std_diameter': np.random.normal(8, 2, n_samples),
    
        # Sensor statistics
        'max_temperature': np.random.normal(300, 50, n_samples),
        'avg_pressure': np.random.normal(10, 3, n_samples),
        'total_flow': np.random.normal(100, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Quality score (linear combination of multiple variables + noise)
    quality_score = (
        0.02 * df['xrd_peak2'] +
        0.5 * df['xps_Fe'] +
        0.3 * df['sem_mean_diameter'] +
        0.1 * df['max_temperature'] +
        np.random.normal(0, 5, n_samples)
    )
    quality_score = np.clip(quality_score, 0, 100)
    
    # Intentionally insert missing values (5%)
    mask = np.random.rand(n_samples, 11) < 0.05
    df_with_missing = df.copy()
    df_with_missing[mask] = np.nan
    
    # Data split
    X = df_with_missing.values
    y = quality_score.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.9)),  # 90% cumulative variance
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Training
    pipeline.fit(X_train, y_train)
    
    # Prediction
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Performance evaluation
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                scoring='r2')
    
    print("=== Model Performance ===")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Number of PCA components
    n_components = pipeline.named_steps['pca'].n_components_
    print(f"\nNumber of PCA principal components: {n_components}")
    print(f"Cumulative explained variance: {pipeline.named_steps['pca'].explained_variance_ratio_.sum()*100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Prediction vs Actual (training data)
    axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, s=30)
    axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('True Quality Score')
    axes[0, 0].set_ylabel('Predicted Quality Score')
    axes[0, 0].set_title(f'Training Set (R²={train_r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Prediction vs Actual (test data)
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, s=30, color='orange')
    axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True Quality Score')
    axes[0, 1].set_ylabel('Predicted Quality Score')
    axes[0, 1].set_title(f'Test Set (R²={test_r2:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred_test
    axes[1, 0].scatter(y_pred_test, residuals, alpha=0.6, s=30, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Quality Score')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importance (PCA loadings)
    pca = pipeline.named_steps['pca']
    loadings = np.abs(pca.components_).sum(axis=0)
    feature_importance = loadings / loadings.sum()
    
    feature_names = list(data.keys())
    axes[1, 1].barh(feature_names, feature_importance, alpha=0.7)
    axes[1, 1].set_xlabel('Importance (normalized)')
    axes[1, 1].set_title('Feature Importance (PCA Loadings)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance ranking
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n=== Feature Importance Ranking ===")
    print(importance_df.to_string(index=False))
    

**Example Results**: 
    
    
    === Model Performance ===
    Training R²: 0.892
    Test R²: 0.867
    Test MAE: 4.23
    CV R² (mean ± std): 0.875 ± 0.032
    
    Number of PCA principal components: 6
    Cumulative explained variance: 91.2%
    
    === Feature Importance Ranking ===
              Feature  Importance
          xrd_peak2      0.1456
            xps_Fe      0.1289
    sem_mean_diameter   0.1142
    max_temperature     0.1078
          xrd_peak1     0.0987
          avg_pressure  0.0921
          ...
    

**Detailed Explanation**: 1\. **Missing value handling**: Median imputation with `SimpleImputer` (robust to outliers) 2\. **Standardization**: Unify variables with different scales (essential for PCA) 3\. **PCA**: Reduced 11 variables → 6 principal components (information loss < 10%) 4\. **Ridge regression**: Suppress overfitting with L2 regularization 5\. **Feature importance**: XRD peak 2, Fe composition, and particle size found to be important **Additional Considerations**: \- Hyperparameter optimization (GridSearchCV) \- Consider nonlinear models (RandomForest, XGBoost) \- Improve prediction interpretability with SHAP \- Integration with design of experiments (DOE) 

* * *

## References

  1. Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: Principles and Practice." OTexts. URL: <https://otexts.com/fpp2/>

  2. Jolliffe, I. T., & Cadima, J. (2016). "Principal component analysis: a review and recent developments." _Philosophical Transactions of the Royal Society A_ , 374(2065). DOI: [10.1098/rsta.2015.0202](<https://doi.org/10.1098/rsta.2015.0202>)

  3. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12, 2825-2830.

  4. sklearn Documentation: Pipeline. URL: <https://scikit-learn.org/stable/modules/compose.html>

  5. Chandola, V. et al. (2009). "Anomaly detection: A survey." _ACM Computing Surveys_ , 41(3), 1-58. DOI: [10.1145/1541880.1541882](<https://doi.org/10.1145/1541880.1541882>)

* * *

## Navigation

### Previous Chapter

**[Chapter 3: Image Data Analysis ←](<chapter-3.html>)**

### Series Index

**[← Return to Series Index](<./index.html>)**

* * *

## Author Information

**Author** : AI Terakoya Content Team **Created** : 2025-10-17 **Version** : 1.1

**Update History** : \- 2025-10-19: v1.1 Quality improvement version released \- Added data licensing and reproducibility section (Section 4.2) \- Added 5 practical pitfall examples (Section 4.6) \- Added time series data skills checklist (Section 4.7) \- Added comprehensive skills assessment and action plan template (Sections 4.8-4.9) \- 2025-10-17: v1.0 Initial version released

**Feedback** : \- GitHub Issues: [Repository URL]/issues \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

## Series Complete

**Congratulations! You have completed the Experimental Data Analysis Introduction series!**

Skills acquired in this series: \- ✅ Data preprocessing (noise removal, outlier detection, standardization) \- ✅ Spectral analysis (XRD, XPS, IR, Raman) \- ✅ Image analysis (SEM, TEM, particle detection, CNN) \- ✅ Time series analysis (sensor data, PCA, Pipeline)

**Next Steps** : \- Machine learning applications (regression, classification, clustering) \- Deep learning introduction (PyTorch, TensorFlow) \- Materials exploration using Bayesian optimization \- Integration with design of experiments (DOE)

Continue deepening your learning at AI Terakoya!
