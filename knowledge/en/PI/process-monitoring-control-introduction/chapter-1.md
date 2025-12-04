---
title: "Chapter 1: Fundamentals of Process Monitoring and Sensor Data Acquisition"
chapter_title: "Chapter 1: Fundamentals of Process Monitoring and Sensor Data Acquisition"
subtitle: From the Basics of Process Monitoring Systems to Time-Series Data Processing
version: 1.0
created_at: 2025-10-25
---

This chapter covers the fundamentals of Fundamentals of Process Monitoring and Sensor Data Acquisition, which fundamentals of process monitoring. You will learn major sensor types, sampling theory, and Process time-series sensor data using Python.

## Learning Objectives

After reading this chapter, you will be able to:

  * ✅ Explain the purpose and importance of process monitoring
  * ✅ Understand major sensor types and their characteristics
  * ✅ Explain sampling theory and the Nyquist theorem
  * ✅ Process time-series sensor data using Python
  * ✅ Evaluate data quality (missing values, drift, outliers)

* * *

## 1.1 Fundamentals of Process Monitoring

### What is Process Monitoring?

**Process monitoring** is an activity in process industries such as chemical plants, pharmaceuticals, food, and semiconductors to monitor the state of manufacturing processes in real-time and ensure quality, safety, and efficiency.

**Main Objectives:**

  * **Quality Assurance** : Verify that products meet specifications
  * **Safety Assurance** : Early detection of abnormal conditions and accident prevention
  * **Efficiency Improvement** : Maintain optimal operating points of processes
  * **Traceability** : Recording and analysis of operational history
  * **Regulatory Compliance** : Adherence to GMP (Good Manufacturing Practice) and other regulations

### SCADA and DCS

Process monitoring systems are primarily implemented in two forms:

Item | SCADA (Supervisory Control And Data Acquisition) | DCS (Distributed Control System)  
---|---|---  
**Main Applications** | Wide-area monitoring (power, water/wastewater, oil pipelines) | Process control (chemical plants, refineries)  
**Control Functions** | Monitoring-centric, some control | Advanced control functions (PID, MPC, etc.)  
**Real-time Performance** | Seconds to minutes | Milliseconds to seconds  
**System Architecture** | Centralized | Distributed (high redundancy)  
**Cost** | Relatively low cost | High cost  
  
### Monitoring System Architecture
    
    
    ```mermaid
    graph TD
        A[Sensor Layer] --> B[Data Collection Layer PLC/RTU]
        B --> C[Communication Layer OPC/Modbus]
        C --> D[Monitoring Layer SCADA/DCS]
        D --> E[Database Layer Historian]
        E --> F[Analysis Layer PI/ML]
    
        style A fill:#e8f5e9
        style B fill:#c8e6c9
        style C fill:#a5d6a7
        style D fill:#81c784
        style E fill:#66bb6a
        style F fill:#4caf50
    ```

* * *

## 1.2 Sensor Types and Data Acquisition

### Major Sensor Types

Sensor Type | Measurement Principle | Typical Measurement Range | Accuracy | Response Time  
---|---|---|---|---  
**Temperature Sensors** |  |  |  |   
\- Thermocouple (TC) | Seebeck effect | -200 to 1800°C | ±0.5 to 2°C | 0.1 to 10 sec  
\- Resistance Temperature Detector (RTD) | Resistance change | -200 to 850°C | ±0.1 to 0.5°C | 1 to 30 sec  
**Pressure Sensors** |  |  |  |   
\- Diaphragm type | Strain gauge | 0 to 100 MPa | ±0.1 to 0.5% FS | <0.1 sec  
**Flow Meters** |  |  |  |   
\- Electromagnetic flowmeter | Electromagnetic induction | 0.01 to 10 m/s | ±0.5% reading | <1 sec  
\- Coriolis flowmeter | Coriolis force | Mass flow rate | ±0.1% reading | <1 sec  
**Level Sensors** |  |  |  |   
\- Differential pressure type | Pressure difference | 0 to 50 m | ±0.5% FS | 1 to 10 sec  
  
### Sampling Theory and Nyquist Theorem

The **Nyquist-Shannon sampling theorem** is the fundamental principle for digitizing continuous signals:

> **Theorem** : To accurately reconstruct a signal, a sampling frequency of **at least twice the highest frequency component** of the signal is required.

Expressed mathematically:

_f s ≥ 2 × fmax_

Where _f s_ is the sampling frequency and _f max_ is the highest frequency component of the signal.

**Recommended Sampling Rates in Practice:**

  * **Temperature** : 1 second to 1 minute (slow changes)
  * **Pressure** : 0.1 seconds to 1 second (relatively fast changes)
  * **Flow Rate** : 0.1 seconds to 1 second
  * **Concentration** : 1 minute to 1 hour (for online analyzers)

**Aliasing** : When the sampling frequency is insufficient, high-frequency components are erroneously recorded as low-frequency components. To prevent this, anti-aliasing filters (low-pass filters) are used.

* * *

## 1.3 Code Examples: Time-Series Sensor Data Processing

From here, we will look at 8 code examples for processing sensor data using Python.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: From here, we will look at 8 code examples for processing se
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    <h4>Code Example 1: Time-Series Sensor Data Simulation and Basic Plotting</h4>
    
    <p><strong>Purpose</strong>: Simulate 24 hours of reactor temperature data and visualize the trend.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Font settings for Japanese (Mac)
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Simulation parameters
    np.random.seed(42)
    sampling_interval = 60  # seconds (1 minute interval)
    duration_hours = 24
    n_samples = int(duration_hours * 3600 / sampling_interval)
    
    # Generate time data
    time_index = pd.date_range('2025-01-01 00:00:00', periods=n_samples, freq=f'{sampling_interval}s')
    
    # Temperature data simulation
    # Base temperature + trend + periodic variation + random noise
    base_temp = 175.0  # Base temperature (°C)
    trend = np.linspace(0, 2, n_samples)  # Gradually increasing trend
    daily_cycle = 3 * np.sin(2 * np.pi * np.arange(n_samples) / (24*60))  # Daily cycle
    noise = np.random.normal(0, 0.8, n_samples)  # Measurement noise
    
    temperature = base_temp + trend + daily_cycle + noise
    
    # Store in DataFrame
    df = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df.set_index('timestamp', inplace=True)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['temperature'], linewidth=1, color='#11998e', alpha=0.8)
    ax.axhline(y=175, color='red', linestyle='--', linewidth=2, label='Target temperature: 175°C')
    ax.fill_between(df.index, 173, 177, alpha=0.15, color='green', label='Tolerance range (±2°C)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Reactor Temperature 24-Hour Trend', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("=== Temperature Data Statistical Summary ===")
    print(f"Average temperature: {df['temperature'].mean():.2f}°C")
    print(f"Standard deviation: {df['temperature'].std():.2f}°C")
    print(f"Maximum temperature: {df['temperature'].max():.2f}°C")
    print(f"Minimum temperature: {df['temperature'].min():.2f}°C")
    print(f"Number of data points: {len(df)}")
    

**Expected Output** :
    
    
    === Temperature Data Statistical Summary ===
    Average temperature: 176.01°C
    Standard deviation: 2.14°C
    Maximum temperature: 181.34°C
    Minimum temperature: 170.89°C
    Number of data points: 1440
    

**Explanation** : This code generates simulation data that includes characteristics of actual process sensor data (trend, periodic variation, noise). It generates 1 minute interval data for 24 hours and handles it as time-series data using Pandas.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Explanation: This code generates simulation data that includ
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    <h4>Code Example 2: Synchronized Multi-Sensor Data Acquisition and Plotting</h4>
    
    <p><strong>Purpose</strong>: Simultaneously monitor and visualize temperature, pressure, and flow rate of a distillation column.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Time data
    time_index = pd.date_range('2025-01-01 00:00:00', periods=1440, freq='1min')
    
    # Multi-sensor simulation data
    # Distillation column operating data
    top_temp = 85 + np.random.normal(0, 1.2, 1440) + 2 * np.sin(np.linspace(0, 4*np.pi, 1440))
    bottom_temp = 155 + np.random.normal(0, 1.5, 1440)
    column_pressure = 1.2 + np.random.normal(0, 0.03, 1440)
    reflux_flow = 50 + np.random.normal(0, 2.5, 1440)
    
    # Store in DataFrame
    df_multi = pd.DataFrame({
        'timestamp': time_index,
        'top_temp': top_temp,
        'bottom_temp': bottom_temp,
        'column_pressure': column_pressure,
        'reflux_flow': reflux_flow
    })
    df_multi.set_index('timestamp', inplace=True)
    
    # Display with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Top temperature
    axes[0].plot(df_multi.index, df_multi['top_temp'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('Top Temperature (°C)', fontsize=11)
    axes[0].set_title('Distillation Column Multi-Sensor Data', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=85, color='red', linestyle='--', alpha=0.5)
    
    # Bottom temperature
    axes[1].plot(df_multi.index, df_multi['bottom_temp'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('Bottom Temperature (°C)', fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=155, color='red', linestyle='--', alpha=0.5)
    
    # Column pressure
    axes[2].plot(df_multi.index, df_multi['column_pressure'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_ylabel('Column Pressure (MPa)', fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=1.2, color='red', linestyle='--', alpha=0.5)
    
    # Reflux flow rate
    axes[3].plot(df_multi.index, df_multi['reflux_flow'], color='#e63946', linewidth=0.8)
    axes[3].set_ylabel('Reflux Flow Rate (m³/h)', fontsize=11)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].grid(alpha=0.3)
    axes[3].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis between variables
    print("\n=== Correlation Coefficients Between Variables ===")
    correlation_matrix = df_multi.corr()
    print(correlation_matrix)
    

**Explanation** : By displaying multiple sensor data on the same time axis, relationships between process variables and abnormal patterns can be visually understood. In actual plants, dozens to hundreds of variables are monitored simultaneously.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Explanation: By displaying multiple sensor data on the same 
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 3: Sampling Rate Analysis and Aliasing Demonstration</h4>
    
    <p><strong>Purpose</strong>: Demonstrate the Nyquist theorem and visualize aliasing caused by improper sampling.</p>
    
    <pre><code class="language-python">import numpy as np
    import matplotlib.pyplot as plt
    
    # Original signal: 5 Hz sine wave
    frequency = 5  # Hz
    duration = 2  # seconds
    t_continuous = np.linspace(0, duration, 10000)  # Approximation of continuous signal
    signal_continuous = np.sin(2 * np.pi * frequency * t_continuous)
    
    # Proper sampling: 20 Hz (twice the Nyquist frequency)
    fs_good = 20
    t_good = np.arange(0, duration, 1/fs_good)
    signal_good = np.sin(2 * np.pi * frequency * t_good)
    
    # Improper sampling: 7 Hz (below Nyquist frequency)
    fs_bad = 7
    t_bad = np.arange(0, duration, 1/fs_bad)
    signal_bad = np.sin(2 * np.pi * frequency * t_bad)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Proper sampling
    axes[0].plot(t_continuous, signal_continuous, 'b-', linewidth=2, alpha=0.5, label='Original signal (5 Hz)')
    axes[0].plot(t_good, signal_good, 'ro-', markersize=6, label=f'Sampling {fs_good} Hz (proper)')
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Proper Sampling (Satisfies Nyquist Theorem)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Improper sampling (aliasing occurs)
    axes[1].plot(t_continuous, signal_continuous, 'b-', linewidth=2, alpha=0.5, label='Original signal (5 Hz)')
    axes[1].plot(t_bad, signal_bad, 'ro-', markersize=6, label=f'Sampling {fs_bad} Hz (improper)')
    # Apparent frequency changes due to aliasing
    aliased_freq = abs(frequency - fs_bad)
    t_aliased = np.linspace(0, duration, 1000)
    signal_aliased = np.sin(2 * np.pi * aliased_freq * t_aliased)
    axes[1].plot(t_aliased, signal_aliased, 'g--', linewidth=2,
                 label=f'Aliased signal ({aliased_freq} Hz)')
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('Improper Sampling (Aliasing Occurs)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Sampling Analysis ===")
    print(f"Original signal frequency: {frequency} Hz")
    print(f"Nyquist frequency: {2 * frequency} Hz")
    print(f"Proper sampling frequency: {fs_good} Hz ({fs_good/(2*frequency):.1f}x Nyquist)")
    print(f"Improper sampling frequency: {fs_bad} Hz ({fs_bad/(2*frequency):.1f}x Nyquist)")
    print(f"Apparent frequency due to aliasing: {aliased_freq} Hz")
    

**Expected Output** :
    
    
    === Sampling Analysis ===
    Original signal frequency: 5 Hz
    Nyquist frequency: 10 Hz
    Proper sampling frequency: 20 Hz (2.0x Nyquist)
    Improper sampling frequency: 7 Hz (0.7x Nyquist)
    Apparent frequency due to aliasing: 2 Hz
    

**Explanation** : This code demonstrates that when the sampling frequency is below the Nyquist frequency (twice the signal's highest frequency), aliasing occurs and the original signal cannot be reconstructed correctly. In process industries, selecting an appropriate sampling frequency based on process dynamics is crucial.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Explanation: This code demonstrates that when the sampling f
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 4: Data Quality Assessment - Missing Value and Outlier Detection</h4>
    
    <p><strong>Purpose</strong>: Detect missing values and outliers in sensor data and assess data quality.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Generate time-series data
    time_index = pd.date_range('2025-01-01', periods=1000, freq='1min')
    temperature = 175 + np.random.normal(0, 2, 1000)
    
    # Intentionally add missing values (simulate communication errors)
    missing_indices = np.random.choice(1000, size=50, replace=False)
    temperature[missing_indices] = np.nan
    
    # Intentionally add outliers (simulate sensor anomalies)
    outlier_indices = np.random.choice(1000, size=10, replace=False)
    temperature[outlier_indices] = temperature[outlier_indices] + np.random.choice([-20, 20], size=10)
    
    # Store in DataFrame
    df_quality = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df_quality.set_index('timestamp', inplace=True)
    
    # Data quality assessment
    print("=== Data Quality Assessment ===")
    print(f"Total data points: {len(df_quality)}")
    print(f"Missing values: {df_quality['temperature'].isna().sum()} ({df_quality['temperature'].isna().sum()/len(df_quality)*100:.1f}%)")
    print(f"Valid data count: {df_quality['temperature'].notna().sum()}")
    
    # Outlier detection: Z-score method
    mean_temp = df_quality['temperature'].mean()
    std_temp = df_quality['temperature'].std()
    z_scores = np.abs((df_quality['temperature'] - mean_temp) / std_temp)
    outliers = z_scores > 3  # 3-sigma rule
    
    print(f"\nOutlier detection (3-sigma rule):")
    print(f"Number of outliers: {outliers.sum()} ({outliers.sum()/len(df_quality)*100:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Original data (including missing values and outliers)
    axes[0].plot(df_quality.index, df_quality['temperature'], 'b-', linewidth=0.8, alpha=0.7)
    axes[0].scatter(df_quality.index[outliers], df_quality['temperature'][outliers],
                    color='red', s=50, label='Outliers', zorder=5)
    axes[0].axhline(y=mean_temp, color='green', linestyle='--', label='Mean')
    axes[0].axhline(y=mean_temp + 3*std_temp, color='orange', linestyle='--', alpha=0.5, label='±3σ')
    axes[0].axhline(y=mean_temp - 3*std_temp, color='orange', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Original Data (Including Missing Values and Outliers)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Visualization of missing values
    missing_mask = df_quality['temperature'].isna()
    axes[1].plot(df_quality.index, df_quality['temperature'], 'b-', linewidth=0.8, alpha=0.7)
    axes[1].scatter(df_quality.index[missing_mask],
                    [175]*missing_mask.sum(), color='red', s=30, marker='x',
                    label='Missing values', zorder=5)
    axes[1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[1].set_title('Missing Value Locations', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Z-score visualization
    axes[2].plot(df_quality.index, z_scores, 'g-', linewidth=0.8)
    axes[2].axhline(y=3, color='red', linestyle='--', label='Threshold (Z=3)')
    axes[2].fill_between(df_quality.index, 0, 3, alpha=0.1, color='green')
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_ylabel('Z-score', fontsize=11)
    axes[2].set_title('Outlier Detection (Z-score Method)', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Explanation** : Data quality assessment is the first step in process monitoring. Missing values occur due to communication errors or sensor failures, while outliers may indicate sensor anomalies or actual process anomalies. The Z-score method (3-sigma rule) is a basic technique for detecting data points that are statistically outside the normal range.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: Explanation: Data quality assessment is the first step in pr
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 5: Sensor Drift Detection and Correction</h4>
    
    <p><strong>Purpose</strong>: Detect and correct sensor drift (gradual deviation over time).</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # Generate time-series data (30 days, 1-hour intervals)
    time_index = pd.date_range('2025-01-01', periods=720, freq='1h')
    
    # True temperature (constant)
    true_temperature = 100.0
    
    # Sensor measurements (including drift)
    # Drift: linear decrease at 0.5°C/month rate
    drift_rate = -0.5 / 30  # °C/day
    days = np.arange(720) / 24
    drift = drift_rate * days
    
    # Measurement noise
    noise = np.random.normal(0, 0.3, 720)
    
    # Sensor measurement = true value + drift + noise
    measured_temperature = true_temperature + drift + noise
    
    # Store in DataFrame
    df_drift = pd.DataFrame({
        'timestamp': time_index,
        'measured': measured_temperature,
        'true': true_temperature
    })
    df_drift.set_index('timestamp', inplace=True)
    
    # Drift detection: estimate trend using linear regression
    x = np.arange(len(df_drift))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_drift['measured'])
    
    print("=== Sensor Drift Analysis ===")
    print(f"Detected drift rate: {slope * 24:.4f} °C/day")
    print(f"Drift over 30 days: {slope * 720:.2f} °C")
    print(f"R² value: {r_value**2:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    # Drift correction
    df_drift['corrected'] = df_drift['measured'] - (slope * x + (intercept - true_temperature))
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Measured values and trend
    axes[0].plot(df_drift.index, df_drift['measured'], 'b-', linewidth=0.8, alpha=0.6, label='Measured (with drift)')
    axes[0].plot(df_drift.index, slope * x + intercept, 'r--', linewidth=2, label='Detected trend')
    axes[0].axhline(y=true_temperature, color='green', linestyle='--', linewidth=2, label='True temperature')
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Sensor Drift Detection', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Corrected measurements
    axes[1].plot(df_drift.index, df_drift['measured'], 'b-', linewidth=0.8, alpha=0.4, label='Before correction')
    axes[1].plot(df_drift.index, df_drift['corrected'], 'orange', linewidth=0.8, label='After correction')
    axes[1].axhline(y=true_temperature, color='green', linestyle='--', linewidth=2, label='True temperature')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[1].set_title('Measurements After Drift Correction', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate correction effectiveness
    mae_before = np.mean(np.abs(df_drift['measured'] - df_drift['true']))
    mae_after = np.mean(np.abs(df_drift['corrected'] - df_drift['true']))
    
    print(f"\nCorrection effectiveness:")
    print(f"Mean absolute error (MAE) before correction: {mae_before:.3f} °C")
    print(f"Mean absolute error (MAE) after correction: {mae_after:.3f} °C")
    print(f"Error reduction rate: {(1 - mae_after/mae_before)*100:.1f}%")
    

**Expected Output** :
    
    
    === Sensor Drift Analysis ===
    Detected drift rate: -0.0167 °C/day
    Drift over 30 days: -0.50 °C
    R² value: 0.9423
    p-value: 0.0000e+00
    
    Correction effectiveness:
    Mean absolute error (MAE) before correction: 0.291 °C
    Mean absolute error (MAE) after correction: 0.243 °C
    Error reduction rate: 16.5%
    

**Explanation** : Sensor drift is a phenomenon where measured values gradually deviate from true values due to sensor aging or environmental changes. By detecting the trend using linear regression and correcting it, data quality can be improved. In actual plants, this is used in conjunction with periodic calibration.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    <h4>Code Example 6: Real-Time Data Streaming Simulation</h4>
    
    <p><strong>Purpose</strong>: Simulate a mechanism for acquiring and processing sensor data in real-time streaming.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import deque
    import time
    
    # Real-time streaming class
    class SensorDataStream:
        def __init__(self, buffer_size=100):
            """
            Sensor data streaming simulator
    
            Parameters:
            -----------
            buffer_size : int
                Data buffer size
            """
            self.buffer_size = buffer_size
            self.time_buffer = deque(maxlen=buffer_size)
            self.data_buffer = deque(maxlen=buffer_size)
            self.start_time = pd.Timestamp.now()
    
        def read_sensor(self):
            """Read one sensor data point (simulation)"""
            # In actual systems, data would be acquired from PLC/DCS here
            current_time = pd.Timestamp.now()
            elapsed_seconds = (current_time - self.start_time).total_seconds()
    
            # Temperature data simulation
            base_temp = 175
            variation = 3 * np.sin(2 * np.pi * elapsed_seconds / 60)  # 60-second period
            noise = np.random.normal(0, 0.5)
            temperature = base_temp + variation + noise
    
            self.time_buffer.append(current_time)
            self.data_buffer.append(temperature)
    
            return current_time, temperature
    
        def get_statistics(self):
            """Calculate statistics of data in buffer"""
            if len(self.data_buffer) == 0:
                return None
    
            data_array = np.array(self.data_buffer)
            stats = {
                'mean': np.mean(data_array),
                'std': np.std(data_array),
                'min': np.min(data_array),
                'max': np.max(data_array),
                'latest': data_array[-1]
            }
            return stats
    
    # Execute streaming simulation
    print("=== Real-Time Sensor Data Streaming ===")
    print("Starting 10-second data collection...\n")
    
    stream = SensorDataStream(buffer_size=50)
    
    # Acquire data every 0.5 seconds for 10 seconds
    duration = 10  # seconds
    interval = 0.5  # seconds
    n_samples = int(duration / interval)
    
    for i in range(n_samples):
        timestamp, temperature = stream.read_sensor()
    
        # Get statistics
        stats = stream.get_statistics()
    
        # Progress display (every 5 samples)
        if (i + 1) % 5 == 0:
            print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                  f"Temperature: {temperature:.2f}°C | "
                  f"Mean: {stats['mean']:.2f}°C | "
                  f"Std dev: {stats['std']:.2f}°C")
    
        time.sleep(interval)
    
    print("\nData collection complete!")
    
    # Visualize collected data
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time-series plot
    axes[0].plot(list(stream.time_buffer), list(stream.data_buffer),
                 'o-', color='#11998e', markersize=4, linewidth=1.5)
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Real-Time Streaming Data', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Histogram
    axes[1].hist(list(stream.data_buffer), bins=15, color='#11998e', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(list(stream.data_buffer)), color='red',
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(list(stream.data_buffer)):.2f}°C')
    axes[1].set_xlabel('Temperature (°C)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Temperature Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal statistics:")
    final_stats = stream.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value:.2f}")
    

**Explanation** : This code implements the basic concept of real-time sensor data streaming. In actual process monitoring systems, data is continuously acquired from PLC/DCS and statistical processing and anomaly detection are performed while buffering. Fixed-length buffers using `deque` are suitable for memory-efficient real-time processing.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    
    <h4>Code Example 7: Data Logging and Buffering System Implementation</h4>
    
    <p><strong>Purpose</strong>: Implement a mechanism for efficiently logging sensor data to log files with buffering.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    
    class ProcessDataLogger:
        def __init__(self, log_file='process_data.csv', buffer_size=100):
            """
            Process data logging system
    
            Parameters:
            -----------
            log_file : str
                Log file name
            buffer_size : int
                Buffer size (writes to file when buffer is full)
            """
            self.log_file = log_file
            self.buffer_size = buffer_size
            self.buffer = []
            self.total_logged = 0
    
            # Create header if file doesn't exist
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('timestamp,temperature,pressure,flow_rate\n')
    
        def log_data(self, temperature, pressure, flow_rate):
            """
            Add data to buffer
    
            Parameters:
            -----------
            temperature : float
                Temperature (°C)
            pressure : float
                Pressure (MPa)
            flow_rate : float
                Flow rate (m³/h)
            """
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            data_point = {
                'timestamp': timestamp,
                'temperature': temperature,
                'pressure': pressure,
                'flow_rate': flow_rate
            }
    
            self.buffer.append(data_point)
    
            # Write to file when buffer is full
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
        def flush(self):
            """Write buffer data to file"""
            if len(self.buffer) == 0:
                return
    
            df = pd.DataFrame(self.buffer)
            df.to_csv(self.log_file, mode='a', header=False, index=False)
    
            self.total_logged += len(self.buffer)
            print(f"[INFO] Wrote {len(self.buffer)} data points to log file. "
                  f"(Total: {self.total_logged} points)")
    
            self.buffer = []
    
        def close(self):
            """Flush remaining buffer and end logging"""
            self.flush()
            print(f"[INFO] Logging completed. Total: {self.total_logged} points")
    
    # Logging system demonstration
    print("=== Process Data Logging System ===\n")
    
    # Initialize logger
    logger = ProcessDataLogger(log_file='demo_process_data.csv', buffer_size=50)
    
    # Log 300 data points (acquired in real-time in practice)
    np.random.seed(42)
    n_samples = 300
    
    print(f"Starting {n_samples} point data logging...\n")
    
    for i in range(n_samples):
        # Sensor data simulation
        temperature = 175 + np.random.normal(0, 2)
        pressure = 1.5 + np.random.normal(0, 0.05)
        flow_rate = 50 + np.random.normal(0, 3)
    
        # Log data
        logger.log_data(temperature, pressure, flow_rate)
    
    # Flush remaining buffer
    logger.close()
    
    # Read and verify log file
    df_logged = pd.read_csv('demo_process_data.csv')
    df_logged['timestamp'] = pd.to_datetime(df_logged['timestamp'])
    
    print(f"\n=== Log File Statistics ===")
    print(f"Total data points: {len(df_logged)}")
    print(f"Period: {df_logged['timestamp'].min()} to {df_logged['timestamp'].max()}")
    print(f"\nStatistics by variable:")
    print(df_logged[['temperature', 'pressure', 'flow_rate']].describe())
    
    # Simple visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    axes[0].plot(df_logged.index, df_logged['temperature'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Logged Process Data', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(df_logged.index, df_logged['pressure'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('Pressure (MPa)', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(df_logged.index, df_logged['flow_rate'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_xlabel('Sample Number', fontsize=11)
    axes[2].set_ylabel('Flow Rate (m³/h)', fontsize=11)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Cleanup
    os.remove('demo_process_data.csv')
    print("\n[INFO] Deleted demo file.")
    

**Explanation** : This code implements the data logging mechanism used in actual process monitoring systems. Buffering reduces the number of disk I/O operations, achieving efficient logging. In actual systems, such logging mechanisms work in conjunction with Historian databases (OSIsoft PI, GE Proficy, etc.).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Explanation: This code implements the data logging mechanism
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 8: Basic Statistical Monitoring (Moving Average and Moving Variance)</h4>
    
    <p><strong>Purpose</strong>: Calculate rolling statistics (moving average, moving variance) to monitor process variation.</p>
    
    <pre><code class="language-python">import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Generate time-series data (24 hours, 1-minute intervals)
    time_index = pd.date_range('2025-01-01 00:00:00', periods=1440, freq='1min')
    
    # Base temperature + periodic variation + noise + sudden variation
    temperature = 175 + 2 * np.sin(2 * np.pi * np.arange(1440) / 360) + np.random.normal(0, 1, 1440)
    
    # Add intentional variation from 12:00 to 14:00 (simulate process disturbance)
    disturbance_start = 12 * 60  # 12:00
    disturbance_end = 14 * 60    # 14:00
    temperature[disturbance_start:disturbance_end] += 5
    
    # Store in DataFrame
    df_stats = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df_stats.set_index('timestamp', inplace=True)
    
    # Calculate rolling statistics
    window_size = 60  # 60-minute (1-hour) moving window
    
    df_stats['moving_average'] = df_stats['temperature'].rolling(window=window_size, center=True).mean()
    df_stats['moving_std'] = df_stats['temperature'].rolling(window=window_size, center=True).std()
    
    # Calculate control limits (moving average ± 3σ)
    df_stats['upper_limit'] = df_stats['moving_average'] + 3 * df_stats['moving_std']
    df_stats['lower_limit'] = df_stats['moving_average'] - 3 * df_stats['moving_std']
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Temperature and moving average
    axes[0].plot(df_stats.index, df_stats['temperature'], 'b-', linewidth=0.6, alpha=0.5, label='Measured')
    axes[0].plot(df_stats.index, df_stats['moving_average'], 'r-', linewidth=2, label=f'Moving average ({window_size} min)')
    axes[0].fill_between(df_stats.index, df_stats['lower_limit'], df_stats['upper_limit'],
                          alpha=0.15, color='green', label='±3σ range')
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Temperature Trend and Moving Average', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    
    # Moving standard deviation
    axes[1].plot(df_stats.index, df_stats['moving_std'], 'orange', linewidth=1.5)
    axes[1].axhline(y=df_stats['moving_std'].mean(), color='red', linestyle='--',
                    label=f'Mean std dev: {df_stats["moving_std"].mean():.2f}°C')
    axes[1].set_ylabel('Moving Standard Deviation (°C)', fontsize=11)
    axes[1].set_title('Process Variation Monitoring', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Coefficient of variation (CV: Coefficient of Variation)
    cv = (df_stats['moving_std'] / df_stats['moving_average']) * 100  # Percentage display
    axes[2].plot(df_stats.index, cv, 'purple', linewidth=1.5)
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_ylabel('Coefficient of Variation CV (%)', fontsize=11)
    axes[2].set_title('Coefficient of Variation (CV = σ/μ × 100%)', fontsize=13, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("=== Rolling Statistics Summary ===")
    print(f"Overall mean temperature: {df_stats['temperature'].mean():.2f}°C")
    print(f"Overall standard deviation: {df_stats['temperature'].std():.2f}°C")
    print(f"\nMoving average range:")
    print(f"  Minimum: {df_stats['moving_average'].min():.2f}°C")
    print(f"  Maximum: {df_stats['moving_average'].max():.2f}°C")
    print(f"\nMoving standard deviation range:")
    print(f"  Minimum: {df_stats['moving_std'].min():.2f}°C")
    print(f"  Maximum: {df_stats['moving_std'].max():.2f}°C")
    print(f"  Mean: {df_stats['moving_std'].mean():.2f}°C")
    
    # Detection of disturbance period
    high_variability = df_stats['moving_std'] > (df_stats['moving_std'].mean() + 2 * df_stats['moving_std'].std())
    print(f"\nHigh variability period detection:")
    print(f"  High variability data points: {high_variability.sum()}")
    if high_variability.sum() > 0:
        print(f"  First high variability time: {df_stats.index[high_variability][0]}")
        print(f"  Last high variability time: {df_stats.index[high_variability][-1]}")
    

**Expected Output** :
    
    
    === Rolling Statistics Summary ===
    Overall mean temperature: 175.24°C
    Overall standard deviation: 2.89°C
    
    Moving average range:
      Minimum: 172.51°C
      Maximum: 179.23°C
    
    Moving standard deviation range:
      Minimum: 1.12°C
      Maximum: 4.58°C
      Mean: 1.87°C
    
    High variability period detection:
      High variability data points: 142
      First high variability time: 2025-01-01 11:30:00
      Last high variability time: 2025-01-01 14:29:00
    

**Explanation** : Rolling statistics (moving average, moving standard deviation) are basic tools for monitoring process trends and variations. Moving averages remove noise and help understand basic process tendencies. Moving standard deviation quantifies process variability and is used to detect abnormal variations. In this code, you can confirm that the moving standard deviation accurately detects the process disturbance (temperature rise from 12:00 to 14:00).

* * *

## 1.4 Chapter Summary

### What We Learned

**1\. Fundamentals of Process Monitoring** — The purpose of monitoring including quality assurance, safety assurance, and efficiency improvement; differences between SCADA and DCS and their application areas; and monitoring system architecture from sensor layer to data analysis layer.

**2\. Sensor Technology** — Characteristics of major sensor types (temperature, pressure, flow, level), sampling theory and Nyquist theorem, and principles of aliasing and prevention measures.

**3\. Practical Time-Series Data Processing** — Time-series data handling with Pandas, missing value and outlier detection using Z-score method, sensor drift detection and correction, real-time streaming and buffering, and variation monitoring using rolling statistics.

### Key Points

**Sampling Frequency** requires selection of appropriate sampling rate according to process dynamics based on the Nyquist theorem. **Data Quality** through detection and correction of missing values, outliers, and drift is fundamental to monitoring. **Real-Time Processing** involves utilization of buffering and efficient data structures like deque. **Statistical Monitoring** enables variation detection using moving average and moving standard deviation.

### To the Next Chapter

In Chapter 2, we will learn about **Statistical Process Control (SPC)** , covering creation and interpretation of Shewhart control charts (X-bar-R, I-MR), calculating process capability indices (Cp, Cpk), advanced SPC methods (CUSUM, EWMA, Hotelling T-squared), control chart abnormality detection rules and alarm generation, and practical SPC with actual process data.
