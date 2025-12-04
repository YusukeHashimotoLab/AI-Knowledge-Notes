---
title: "Chapter 3: Anomaly Detection and Process Monitoring"
chapter_title: "Chapter 3: Anomaly Detection and Process Monitoring"
subtitle: From Rule-Based to Machine Learning - Comprehensive Anomaly Detection Methods
version: 1.0
created_at: 2025-10-25
---

This chapter covers Anomaly Detection and Process Monitoring. You will learn statistical anomaly detection methods (Z-score and Design alarm management systems.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand implementation and limitations of rule-based anomaly detection
  * ✅ Apply statistical anomaly detection methods (Z-score, Modified Z-score)
  * ✅ Implement machine learning-based anomaly detection (Isolation Forest, One-Class SVM)
  * ✅ Detect time-series anomalies using deep learning (LSTM Autoencoder)
  * ✅ Design alarm management systems and reduce false alarms

* * *

## 3.1 Fundamentals of Anomaly Detection

### What is Anomaly Detection

**Anomaly Detection** is the process of identifying data points or patterns that deviate significantly from normal patterns. In process industries, it is essential for early detection of equipment failures, quality anomalies, and process abnormalities.

Anomaly Detection Method | Characteristics | Advantages | Disadvantages | Application Examples  
---|---|---|---|---  
**Rule-Based** | Threshold-based judgment | Simple, easy to interpret | Difficult to detect complex patterns | Temperature/pressure monitoring  
**Statistical Methods** | Deviation from statistical distribution | Theoretical foundation | Requires normal distribution assumption | Process variable monitoring  
**Machine Learning** | Pattern learning from data | Can detect complex patterns | Low interpretability, requires data | Multivariate anomaly detection  
**Deep Learning** | Deep feature extraction | Handles high-dimensional data | High computational cost, requires large data | Time-series anomaly detection  
  
### Anomaly Detection System Architecture
    
    
    ```mermaid
    graph TD
        A[Data Collection] --> B[Preprocessing & Normalization]
        B --> C[Feature Extraction]
        C --> D[Anomaly Detection Model]
        D --> E{Anomaly Judgment}
        E -->|Normal| F[Continue Normal Operation]
        E -->|Anomaly| G[Generate Alarm]
        G --> H[Root Cause Analysis]
        H --> I[Execute Countermeasures]
        I --> A
    
        style A fill:#e8f5e9
        style D fill:#b3e5fc
        style G fill:#ffcdd2
        style I fill:#fff9c4
    ```

* * *

## 3.2 Code Examples: Implementing Anomaly Detection Methods
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 3.2 Code Examples: Implementing Anomaly Detection Methods
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 1: Rule-Based Anomaly Detection (Threshold-Based)</h4>
    
    <p><strong>Purpose</strong>: Implement an anomaly detection system using simple threshold rules.</p>
    
    <pre><code class="language-python">import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Japanese font settings
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    np.random.seed(42)
    
    # Generate process data (reactor temperature, 200 samples)
    n_samples = 200
    time_index = pd.date_range('2025-01-01', periods=n_samples, freq='1min')
    
    # Normal data (mean 175°C, standard deviation 2°C)
    temperature = np.random.normal(175, 2, n_samples)
    
    # Insert intentional anomalies
    # Anomaly 1: Spike (sample 50)
    temperature[50] = 190
    
    # Anomaly 2: Gradual increase (samples 120-140)
    temperature[120:141] = 175 + np.linspace(0, 10, 21)
    
    # Anomaly 3: Low temperature anomaly (samples 180-190)
    temperature[180:191] = np.random.normal(165, 1.5, 11)
    
    # Store in DataFrame
    df = pd.DataFrame({
        'timestamp': time_index,
        'temperature': temperature
    })
    df.set_index('timestamp', inplace=True)
    
    class RuleBasedAnomalyDetector:
        """Rule-based anomaly detection system"""
    
        def __init__(self, rules):
            """
            Parameters:
            -----------
            rules : dict
                Dictionary of anomaly detection rules
                Example: {'upper_limit': 180, 'lower_limit': 170, ...}
            """
            self.rules = rules
            self.anomalies = []
    
        def detect(self, data):
            """Execute anomaly detection"""
            anomaly_flags = np.zeros(len(data), dtype=bool)
            anomaly_types = [''] * len(data)
    
            for i, value in enumerate(data):
                # Rule 1: Upper limit check
                if value > self.rules.get('upper_limit', float('inf')):
                    anomaly_flags[i] = True
                    anomaly_types[i] = 'Upper Limit Exceeded'
                    self.anomalies.append({
                        'index': i,
                        'value': value,
                        'type': 'Upper Limit Exceeded',
                        'severity': 'HIGH'
                    })
    
                # Rule 2: Lower limit check
                elif value < self.rules.get('lower_limit', float('-inf')):
                    anomaly_flags[i] = True
                    anomaly_types[i] = 'Below Lower Limit'
                    self.anomalies.append({
                        'index': i,
                        'value': value,
                        'type': 'Below Lower Limit',
                        'severity': 'HIGH'
                    })
    
                # Rule 3: Rate of change check (difference from previous sample)
                if i > 0:
                    rate_of_change = abs(value - data[i-1])
                    if rate_of_change > self.rules.get('max_change_rate', float('inf')):
                        anomaly_flags[i] = True
                        anomaly_types[i] = 'Rapid Change'
                        self.anomalies.append({
                            'index': i,
                            'value': value,
                            'type': 'Rapid Change',
                            'severity': 'MEDIUM'
                        })
    
                # Rule 4: Consecutive deviation check (warning range)
                # Implementation in the next loop (requires continuity judgment)
    
            # Rule 4 implementation: 5 consecutive samples exceed warning range
            warning_upper = self.rules.get('warning_upper', float('inf'))
            warning_lower = self.rules.get('warning_lower', float('-inf'))
    
            consecutive_warning = 0
            for i, value in enumerate(data):
                if value > warning_upper or value < warning_lower:
                    consecutive_warning += 1
                    if consecutive_warning >= 5 and not anomaly_flags[i]:
                        anomaly_flags[i] = True
                        anomaly_types[i] = 'Consecutive Warning'
                        self.anomalies.append({
                            'index': i,
                            'value': value,
                            'type': 'Consecutive Warning',
                            'severity': 'MEDIUM'
                        })
                else:
                    consecutive_warning = 0
    
            return anomaly_flags, anomaly_types
    
    # Configure rule-based anomaly detector
    rules = {
        'upper_limit': 180,        # Upper limit threshold
        'lower_limit': 170,        # Lower limit threshold
        'warning_upper': 178,      # Warning upper limit
        'warning_lower': 172,      # Warning lower limit
        'max_change_rate': 8       # Maximum change rate (°C/min)
    }
    
    detector = RuleBasedAnomalyDetector(rules)
    anomaly_flags, anomaly_types = detector.detect(df['temperature'].values)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Normal data
    normal_mask = ~anomaly_flags
    ax.plot(df.index[normal_mask], df['temperature'][normal_mask],
            'o', color='#11998e', markersize=4, alpha=0.6, label='Normal')
    
    # Anomalous data
    anomaly_mask = anomaly_flags
    ax.scatter(df.index[anomaly_mask], df['temperature'][anomaly_mask],
              color='red', s=80, marker='o', zorder=5, label='Anomaly')
    
    # Threshold lines
    ax.axhline(y=rules['upper_limit'], color='red', linestyle='--',
              linewidth=2, label=f'Upper Limit ({rules["upper_limit"]}°C)')
    ax.axhline(y=rules['lower_limit'], color='red', linestyle='--',
              linewidth=2, label=f'Lower Limit ({rules["lower_limit"]}°C)')
    ax.axhline(y=rules['warning_upper'], color='orange', linestyle=':',
              linewidth=1.5, alpha=0.7, label='Warning Range')
    ax.axhline(y=rules['warning_lower'], color='orange', linestyle=':',
              linewidth=1.5, alpha=0.7)
    
    # Target value
    target = 175
    ax.axhline(y=target, color='blue', linestyle='-', linewidth=2,
              label=f'Target ({target}°C)')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Rule-Based Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary of anomaly detection results
    print("=== Rule-Based Anomaly Detection Results ===")
    print(f"\nTotal data points: {len(df)}")
    print(f"Anomalies detected: {anomaly_flags.sum()} ({anomaly_flags.sum()/len(df)*100:.1f}%)")
    
    # Aggregate by anomaly type
    df_anomalies = pd.DataFrame(detector.anomalies)
    if len(df_anomalies) > 0:
        print(f"\n【Breakdown by Anomaly Type】")
        type_counts = df_anomalies['type'].value_counts()
        for anomaly_type, count in type_counts.items():
            print(f"  {anomaly_type}: {count} cases")
    
        print(f"\n【Breakdown by Severity】")
        severity_counts = df_anomalies['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count} cases")
    
        # Display first 5 anomalies
        print(f"\n【Detected Anomalies (First 5 Cases)】")
        for i, anomaly in enumerate(df_anomalies.head(5).to_dict('records')):
            print(f"{i+1}. Sample {anomaly['index']+1} | "
                  f"Value: {anomaly['value']:.2f}°C | "
                  f"Type: {anomaly['type']} | "
                  f"Severity: {anomaly['severity']}")
    

**Explanation** : Rule-based anomaly detection is the simplest method that judges anomalies based on clear thresholds and rate of change. It is easy to implement and has high interpretability, but it is difficult to detect complex patterns or context-dependent anomalies. In process industries, it is widely used as primary monitoring for safety-critical variables (temperature, pressure).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Explanation: Rule-based anomaly detection is the simplest me
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    <h4>Code Example 2: Statistical Anomaly Detection (Z-score and Modified Z-score)</h4>
    
    <p><strong>Purpose</strong>: Implement statistical anomaly detection using Z-score and Modified Z-score (Median Absolute Deviation).</p>
    
    <pre><code class="language-python">import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # Generate process data
    n_samples = 300
    data = np.random.normal(100, 5, n_samples)
    
    # Add outliers
    outlier_indices = [50, 100, 150, 200, 250]
    data[outlier_indices] = [130, 70, 125, 65, 135]
    
    def z_score_detection(data, threshold=3):
        """
        Anomaly detection using Z-score
    
        Parameters:
        -----------
        data : array-like
            Input data
        threshold : float
            Threshold for anomaly judgment (multiple of standard deviation)
    
        Returns:
        --------
        anomalies : boolean array
            Anomaly flags
        z_scores : array
            Z-score values
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > threshold
    
        return anomalies, z_scores
    
    def modified_z_score_detection(data, threshold=3.5):
        """
        Anomaly detection using Modified Z-score (MAD-based)
        Robust method against outliers
    
        Parameters:
        -----------
        data : array-like
            Input data
        threshold : float
            Threshold for anomaly judgment (default 3.5)
    
        Returns:
        --------
        anomalies : boolean array
            Anomaly flags
        modified_z_scores : array
            Modified Z-score values
        """
        median = np.median(data)
    
        # Calculate MAD (Median Absolute Deviation)
        mad = np.median(np.abs(data - median))
    
        # Calculate Modified Z-score
        # Constant 0.6745 is a coefficient to convert MAD to standard deviation
        modified_z_scores = 0.6745 * (data - median) / mad
    
        anomalies = np.abs(modified_z_scores) > threshold
    
        return anomalies, modified_z_scores
    
    # Detection using Z-score method
    anomalies_z, z_scores = z_score_detection(data, threshold=3)
    
    # Detection using Modified Z-score method
    anomalies_modified, modified_z_scores = modified_z_score_detection(data, threshold=3.5)
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Original data and anomaly detection results (Z-score method)
    axes[0].plot(range(n_samples), data, 'o', color='lightgray',
                markersize=4, alpha=0.5, label='All Data')
    axes[0].scatter(np.where(anomalies_z)[0], data[anomalies_z],
                   color='red', s=100, marker='o', zorder=5, label='Anomaly (Z-score)')
    axes[0].axhline(y=np.mean(data), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean = {np.mean(data):.2f}')
    axes[0].axhline(y=np.mean(data) + 3*np.std(data), color='orange',
                   linestyle='--', linewidth=2, label='±3σ')
    axes[0].axhline(y=np.mean(data) - 3*np.std(data), color='orange',
                   linestyle='--', linewidth=2)
    axes[0].set_ylabel('Data Value', fontsize=11)
    axes[0].set_title('Anomaly Detection Using Z-score Method', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Z-score visualization
    axes[1].plot(range(n_samples), z_scores, 'o-', color='#11998e',
                markersize=4, linewidth=1, alpha=0.7, label='Z-score')
    axes[1].axhline(y=3, color='red', linestyle='--', linewidth=2,
                   label='Threshold = 3')
    axes[1].scatter(np.where(anomalies_z)[0], z_scores[anomalies_z],
                   color='red', s=100, marker='o', zorder=5)
    axes[1].set_ylabel('Z-score', fontsize=11)
    axes[1].set_title('Z-score Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Modified Z-score visualization
    axes[2].plot(range(n_samples), np.abs(modified_z_scores), 'o-', color='#7b2cbf',
                markersize=4, linewidth=1, alpha=0.7, label='Modified Z-score')
    axes[2].axhline(y=3.5, color='red', linestyle='--', linewidth=2,
                   label='Threshold = 3.5')
    axes[2].scatter(np.where(anomalies_modified)[0],
                   np.abs(modified_z_scores[anomalies_modified]),
                   color='red', s=100, marker='o', zorder=5)
    axes[2].set_xlabel('Sample Number', fontsize=11)
    axes[2].set_ylabel('Modified Z-score (Absolute)', fontsize=11)
    axes[2].set_title('Modified Z-score Distribution (MAD-Based)', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("=== Statistical Anomaly Detection Results ===")
    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Standard Deviation: {np.std(data, ddof=1):.2f}")
    print(f"  MAD: {np.median(np.abs(data - np.median(data))):.2f}")
    
    print(f"\n【Z-score Method】")
    print(f"  Detected anomalies: {anomalies_z.sum()} ({anomalies_z.sum()/n_samples*100:.1f}%)")
    if anomalies_z.any():
        print(f"  Anomaly samples: {np.where(anomalies_z)[0] + 1}")
        print(f"  Anomaly values: {data[anomalies_z]}")
    
    print(f"\n【Modified Z-score Method (MAD-Based)】")
    print(f"  Detected anomalies: {anomalies_modified.sum()} "
          f"({anomalies_modified.sum()/n_samples*100:.1f}%)")
    if anomalies_modified.any():
        print(f"  Anomaly samples: {np.where(anomalies_modified)[0] + 1}")
        print(f"  Anomaly values: {data[anomalies_modified]}")
    
    # Method comparison
    print(f"\n【Method Comparison】")
    print(f"  Detected by both methods: {np.sum(anomalies_z & anomalies_modified)} cases")
    print(f"  Z-score only: {np.sum(anomalies_z & ~anomalies_modified)} cases")
    print(f"  Modified Z-score only: {np.sum(~anomalies_z & anomalies_modified)} cases")
    
    print(f"\n【Recommendations】")
    print("  - Z-score method: Effective when data follows normal distribution")
    print("  - Modified Z-score method: Robust when many outliers exist (MAD is less affected by outliers)")
    

**Explanation** : The Z-score method calculates how many standard deviations away from the mean a data point is, assuming the data follows a normal distribution. The Modified Z-score method uses the median instead of the mean and MAD (Median Absolute Deviation) instead of the standard deviation, making it a robust method less affected by outliers. For process data containing outliers, the Modified Z-score method is recommended.

## 3.3 Chapter Summary

### What We Learned

  1. **Fundamentals of Anomaly Detection**
     * Purpose and importance of anomaly detection
     * Four approaches: rule-based, statistical, machine learning, deep learning
     * Understanding advantages and disadvantages of each method
  2. **Statistical Anomaly Detection**
     * Z-score method: Outlier detection assuming normal distribution
     * Modified Z-score method (MAD): Robust method against outliers
     * Application conditions and usage

### Key Points

**Method Selection** involves choosing appropriate anomaly detection methods based on data characteristics and objectives. **False Alarm Reduction** techniques are essential for practical systems to maintain operator trust. **Root Cause Analysis** is important since detecting anomalies alone is insufficient without identifying their causes. A **Staged Approach** starting with simple methods and introducing advanced methods as needed proves most effective. **Integrated Systems** combining multiple methods provide the most robust and reliable anomaly detection.

### To the Next Chapter

In Chapter 4, we will learn about **Feedback Control and PID Control** , covering the fundamental theory of feedback control, first-order lag systems and step response, and implementation of PID controllers including P, PI, and PID variations. The chapter also presents Ziegler-Nichols tuning methods and addresses practical considerations such as anti-windup strategies and cascade control architectures.
