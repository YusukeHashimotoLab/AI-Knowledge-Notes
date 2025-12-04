---
title: "Chapter 2: Statistical Anomaly Detection"
chapter_title: "Chapter 2: Statistical Anomaly Detection"
subtitle: Fundamentals and Applications of Statistical Methods for Anomaly Detection
reading_time: 30-35 minutes
difficulty: Beginner to Intermediate
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Statistical Anomaly Detection. You will learn outlier detection using Z-score, Mahalanobis distance, and statistical hypothesis tests (Grubbs.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Implement outlier detection using Z-score and IQR
  * ✅ Understand Mahalanobis distance and multivariate Gaussian distribution
  * ✅ Apply statistical hypothesis tests (Grubbs, ESD)
  * ✅ Use anomaly detection methods for time series data
  * ✅ Build complete pipelines for statistical methods

* * *

## 2.1 Statistical Outlier Detection

### Z-score (Standardized Score)

The **Z-score** is a metric that indicates how many standard deviations a data point is away from the mean.

> Z-score = $\frac{x - \mu}{\sigma}$
> 
> Common threshold: $|Z| > 3$ is considered anomalous

#### Characteristics of Z-score

  * **Advantages** : Simple and easy to interpret, fast computation
  * **Disadvantages** : Assumes normal distribution, sensitive to outliers
  * **Applications** : Univariate data, data close to normal distribution

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: Characteristics of Z-score
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Data generation
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=300)
    outliers = np.array([5, -5, 6, -6, 7])
    data = np.concatenate([normal_data, outliers])
    
    # Z-score calculation
    z_scores = np.abs(stats.zscore(data))
    threshold = 3
    anomalies = z_scores > threshold
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data distribution
    axes[0].hist(data, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(x=data.mean() + 3*data.std(), color='red',
                    linestyle='--', linewidth=2, label='±3σ')
    axes[0].axvline(x=data.mean() - 3*data.std(), color='red',
                    linestyle='--', linewidth=2)
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Data Distribution and Z-score Threshold', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Z-score plot
    axes[1].scatter(range(len(data)), z_scores, alpha=0.6, s=30, c='blue')
    axes[1].scatter(np.where(anomalies)[0], z_scores[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold=3')
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('|Z-score|', fontsize=12)
    axes[1].set_title('Anomaly Detection with Z-score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Z-score Anomaly Detection Results ===")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly indices: {np.where(anomalies)[0]}")
    print(f"Anomaly values: {data[anomalies]}")
    

### IQR (Interquartile Range)

The **IQR method** is a robust detection technique against outliers.

> IQR = Q3 - Q1  
>  Anomaly criterion: $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: IQR = Q3 - Q1Anomaly criterion: $x  Q3 + 1.5 \times IQR$
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Data generation
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=300)
    outliers = np.array([100, 5, 110, 0])
    data = np.concatenate([normal_data, outliers])
    
    # IQR calculation
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Anomaly detection
    anomalies = (data < lower_bound) | (data > upper_bound)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    bp = axes[0].boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    axes[0].scatter([1]*anomalies.sum(), data[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Box Plot and IQR Anomaly Detection', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot
    axes[1].scatter(range(len(data)), data, alpha=0.6, s=30, c='blue', label='Normal')
    axes[1].scatter(np.where(anomalies)[0], data[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black')
    axes[1].axhline(y=upper_bound, color='red', linestyle='--', linewidth=2, label='IQR Boundary')
    axes[1].axhline(y=lower_bound, color='red', linestyle='--', linewidth=2)
    axes[1].axhline(y=Q1, color='green', linestyle=':', linewidth=1.5, label='Q1/Q3')
    axes[1].axhline(y=Q3, color='green', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Anomaly Detection with IQR Method', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== IQR Anomaly Detection Results ===")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly values: {data[anomalies]}")
    

> **Important** : Unlike Z-score, IQR does not assume normal distribution and is less sensitive to outliers, making it a robust method.

* * *

## 2.2 Probability Distribution-Based Anomaly Detection

### Mahalanobis Distance

**Mahalanobis distance** is a distance metric that accounts for covariance in multivariate data.

> $D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$
> 
> Where $\mu$ is the mean vector and $\Sigma$ is the covariance matrix

#### Characteristics

  * **Advantages** : Considers correlation between variables, scale-invariant
  * **Disadvantages** : Requires covariance matrix inversion, high computational cost
  * **Applications** : Multivariate data, correlated features

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Characteristics
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2
    
    # Generate correlated bivariate data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # Correlation coefficient 0.8
    normal_data = np.random.multivariate_normal(mean, cov, size=300)
    
    # Add anomalous data
    outliers = np.array([[4, 4], [-4, -4], [4, -4]])
    data = np.vstack([normal_data, outliers])
    
    # Mahalanobis distance calculation
    mean_vec = normal_data.mean(axis=0)
    cov_matrix = np.cov(normal_data.T)
    cov_inv = np.linalg.inv(cov_matrix)
    
    mahal_distances = np.array([mahalanobis(x, mean_vec, cov_inv) for x in data])
    
    # Threshold (99% point of chi-square distribution with df=2)
    threshold = np.sqrt(chi2.ppf(0.99, df=2))
    anomalies = mahal_distances > threshold
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data distribution
    axes[0].scatter(normal_data[:, 0], normal_data[:, 1],
                    alpha=0.6, s=50, c='blue', label='Normal', edgecolors='black')
    axes[0].scatter(outliers[:, 0], outliers[:, 1],
                    c='red', s=150, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    
    # Confidence ellipse (99%)
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * threshold * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean_vec, width, height, angle=angle,
                      edgecolor='red', facecolor='none', linewidth=2, linestyle='--', label='99% Confidence Ellipse')
    axes[0].add_patch(ellipse)
    
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].set_title('Anomaly Detection with Mahalanobis Distance', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Mahalanobis distance distribution
    axes[1].hist(mahal_distances, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
    axes[1].set_xlabel('Mahalanobis Distance', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Mahalanobis Distance Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Mahalanobis Distance Anomaly Detection Results ===")
    print(f"Threshold: {threshold:.3f}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly Mahalanobis distances: {mahal_distances[anomalies]}")
    

### Multivariate Gaussian Distribution

Anomaly detection using **multivariate Gaussian distribution** determines anomalies based on probability density.

> $p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$
> 
> Anomaly criterion: $p(x) < \epsilon$ (threshold)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Anomaly criterion: $p(x) < \epsilon$ (threshold)
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    
    # Data generation
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    normal_data = np.random.multivariate_normal(mean, cov, size=300)
    outliers = np.array([[5, 5], [-5, -5], [5, -5]])
    data = np.vstack([normal_data, outliers])
    
    # Multivariate Gaussian distribution parameter estimation
    mean_vec = normal_data.mean(axis=0)
    cov_matrix = np.cov(normal_data.T)
    mvn = multivariate_normal(mean=mean_vec, cov=cov_matrix)
    
    # Probability density calculation
    densities = mvn.pdf(data)
    
    # Threshold (1% point)
    threshold = np.percentile(densities, 1)
    anomalies = densities < threshold
    
    # Calculate probability density on grid (for heatmap)
    x_range = np.linspace(-6, 6, 200)
    y_range = np.linspace(-6, 6, 200)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.dstack((xx, yy))
    Z = mvn.pdf(positions)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Probability density heatmap
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.7)
    axes[0].scatter(normal_data[:, 0], normal_data[:, 1],
                    alpha=0.6, s=30, c='blue', label='Normal', edgecolors='black')
    axes[0].scatter(data[anomalies, 0], data[anomalies, 1],
                    c='red', s=150, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    plt.colorbar(contour, ax=axes[0], label='Probability Density')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].set_title('Anomaly Detection with Multivariate Gaussian Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Probability density histogram
    axes[1].hist(np.log(densities), bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=np.log(threshold), color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_xlabel('log(Probability Density)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Probability Density Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Multivariate Gaussian Distribution Anomaly Detection Results ===")
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly probability densities: {densities[anomalies]}")
    

> **Important** : Mahalanobis distance and multivariate Gaussian distribution are mathematically equivalent. The square of Mahalanobis distance is proportional to log probability density.

* * *

## 2.3 Statistical Hypothesis Testing

### Grubbs' Test

**Grubbs' Test** is a hypothesis test for detecting a single outlier.

> Null hypothesis $H_0$: No outliers exist  
>  Test statistic: $G = \frac{\max|x_i - \bar{x}|}{s}$  
>  Where $s$ is the standard deviation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def grubbs_test(data, alpha=0.05):
        """Outlier detection with Grubbs' Test"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
    
        # Calculate test statistic
        deviations = np.abs(data - mean)
        max_idx = np.argmax(deviations)
        G = deviations[max_idx] / std
    
        # Calculate critical value
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    
        is_outlier = G > G_critical
    
        return {
            'G': G,
            'G_critical': G_critical,
            'is_outlier': is_outlier,
            'outlier_idx': max_idx if is_outlier else None,
            'outlier_value': data[max_idx] if is_outlier else None,
            'p_value': 1 - stats.t.cdf(G * np.sqrt(n) / np.sqrt(n - 1), n - 2)
        }
    
    # Data generation
    np.random.seed(42)
    data = np.concatenate([np.random.normal(50, 5, size=30), [80]])  # 80 is an outlier
    
    # Execute Grubbs' Test
    result = grubbs_test(data, alpha=0.05)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data plot
    axes[0].scatter(range(len(data)), data, alpha=0.6, s=50, c='blue', label='Data')
    if result['is_outlier']:
        axes[0].scatter(result['outlier_idx'], result['outlier_value'],
                        c='red', s=200, marker='X', label='Outlier', zorder=5, edgecolors='black', linewidths=2)
    axes[0].axhline(y=data.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    axes[0].axhline(y=data.mean() + 3*data.std(), color='orange', linestyle=':', linewidth=1.5, label='±3σ')
    axes[0].axhline(y=data.mean() - 3*data.std(), color='orange', linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title("Grubbs' Test Results", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Test statistic comparison
    axes[1].bar(['G Statistic', 'Critical Value'], [result['G'], result['G_critical']],
                color=['steelblue', 'red'], edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Test Statistic vs Critical Value', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Grubbs' Test Results ===")
    print(f"G statistic: {result['G']:.3f}")
    print(f"Critical value: {result['G_critical']:.3f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Outlier detected: {'Yes' if result['is_outlier'] else 'No'}")
    if result['is_outlier']:
        print(f"Outlier: index={result['outlier_idx']}, value={result['outlier_value']:.2f}")
    

### ESD Test (Extreme Studentized Deviate Test)

The **Generalized ESD Test** is an extended version that can detect multiple outliers.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def generalized_esd_test(data, max_outliers, alpha=0.05):
        """Multiple outlier detection with Generalized ESD Test"""
        n = len(data)
        outliers = []
        data_copy = data.copy()
    
        for i in range(max_outliers):
            mean = np.mean(data_copy)
            std = np.std(data_copy, ddof=1)
    
            # Calculate test statistic
            deviations = np.abs(data_copy - mean)
            max_idx = np.argmax(deviations)
            R = deviations[max_idx] / std
    
            # Calculate critical value
            n_current = len(data_copy)
            p = 1 - alpha / (2 * (n_current - i))
            t_dist = stats.t.ppf(p, n_current - i - 2)
            lambda_critical = ((n_current - i - 1) * t_dist) / np.sqrt((n_current - i - 2 + t_dist**2) * (n_current - i))
    
            if R > lambda_critical:
                outlier_idx = np.where(data == data_copy[max_idx])[0][0]
                outliers.append({'index': outlier_idx, 'value': data_copy[max_idx], 'R': R})
                data_copy = np.delete(data_copy, max_idx)
            else:
                break
    
        return outliers
    
    # Data generation
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, size=30)
    outlier_values = [80, 85, 15]
    data = np.concatenate([normal_data, outlier_values])
    
    # Execute ESD Test
    outliers = generalized_esd_test(data, max_outliers=5, alpha=0.05)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(data)), data, alpha=0.6, s=50, c='blue', label='Normal Data')
    
    if outliers:
        outlier_indices = [o['index'] for o in outliers]
        outlier_values_detected = [o['value'] for o in outliers]
        plt.scatter(outlier_indices, outlier_values_detected,
                    c='red', s=200, marker='X', label='Outlier', zorder=5, edgecolors='black', linewidths=2)
    
    plt.axhline(y=data.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    plt.axhline(y=data.mean() + 3*data.std(), color='orange', linestyle=':', linewidth=1.5, label='±3σ')
    plt.axhline(y=data.mean() - 3*data.std(), color='orange', linestyle=':', linewidth=1.5)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Generalized ESD Test Results', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Generalized ESD Test Results ===")
    print(f"Number of outliers detected: {len(outliers)}")
    for i, o in enumerate(outliers, 1):
        print(f"Outlier {i}: index={o['index']}, value={o['value']:.2f}, R={o['R']:.3f}")
    

> **Important** : Grubbs' Test can only detect one outlier, while ESD Test can sequentially detect multiple outliers.

* * *

## 2.4 Time Series Anomaly Detection

### Anomaly Detection with Moving Average

The **moving average** captures trends in time series data and detects deviations.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Themoving averagecaptures trends in time series data and det
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Time series data generation
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    trend = 0.05 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 2, n_samples)
    data = trend + seasonal + noise
    
    # Add anomalies
    anomaly_indices = [50, 150, 250]
    data[anomaly_indices] += [20, -25, 30]
    
    # Calculate moving average
    window_size = 20
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    moving_std = np.array([data[max(0, i-window_size//2):min(len(data), i+window_size//2)].std()
                           for i in range(len(data))])
    
    # Anomaly detection (3σ rule)
    residuals = np.abs(data - moving_avg)
    threshold = 3 * moving_std
    anomalies = residuals > threshold
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series data
    axes[0].plot(time, data, alpha=0.7, linewidth=1, label='Original Data', color='blue')
    axes[0].plot(time, moving_avg, linewidth=2, label=f'Moving Average (window={window_size})', color='green')
    axes[0].fill_between(time, moving_avg - 3*moving_std, moving_avg + 3*moving_std,
                         alpha=0.2, color='green', label='±3σ')
    axes[0].scatter(time[anomalies], data[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Time Series Anomaly Detection with Moving Average', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    axes[1].plot(time, residuals, alpha=0.7, linewidth=1, color='blue')
    axes[1].plot(time, threshold, linestyle='--', linewidth=2, color='red', label='Threshold (3σ)')
    axes[1].scatter(time[anomalies], residuals[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Residuals and Anomaly Threshold', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Moving Average Anomaly Detection Results ===")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly indices: {np.where(anomalies)[0]}")
    

### Anomaly Detection with Seasonal Decomposition

**STL decomposition** (Seasonal and Trend decomposition using Loess) decomposes into seasonality, trend, and residual components.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: STL decomposition(Seasonal and Trend decomposition using Loe
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Time series data generation (clear seasonality)
    np.random.seed(42)
    n_samples = 200
    time = np.arange(n_samples)
    trend = 0.1 * time
    seasonal = 15 * np.sin(2 * np.pi * time / 30)  # Period 30
    noise = np.random.normal(0, 2, n_samples)
    data = trend + seasonal + noise
    
    # Add anomalies
    anomaly_indices = [50, 120, 180]
    data[anomaly_indices] += [30, -30, 25]
    
    # Seasonal decomposition
    result = seasonal_decompose(data, model='additive', period=30, extrapolate_trend='freq')
    
    # Anomaly detection from residuals
    residual = result.resid
    threshold = 3 * np.nanstd(residual)
    anomalies = np.abs(residual) > threshold
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original data
    axes[0].plot(time, data, linewidth=1, color='blue')
    axes[0].scatter(time[anomalies], data[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    axes[0].set_ylabel('Original Data', fontsize=11)
    axes[0].set_title('Time Series Anomaly Detection with Seasonal Decomposition', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(time, result.trend, linewidth=2, color='green')
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonality
    axes[2].plot(time, result.seasonal, linewidth=2, color='orange')
    axes[2].set_ylabel('Seasonality', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(time, residual, linewidth=1, color='blue')
    axes[3].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold (±3σ)')
    axes[3].axhline(y=-threshold, color='red', linestyle='--', linewidth=2)
    axes[3].scatter(time[anomalies], residual[anomalies],
                    c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Seasonal Decomposition Anomaly Detection Results ===")
    print(f"Period: 30")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly indices: {np.where(anomalies)[0]}")
    

> **Important** : Seasonal decomposition can clearly detect true anomalies in the residuals by removing trend and seasonality.

* * *

## 2.5 Implementation and Applications

### Complete Pipeline for Statistical Anomaly Detection

Build a practical statistical anomaly detection system.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    class StatisticalAnomalyDetector:
        """Integrated class for statistical anomaly detection"""
    
        def __init__(self, method='zscore', threshold=3.0, window_size=20):
            """
            Parameters:
            -----------
            method : str
                Detection method ('zscore', 'iqr', 'mahalanobis', 'moving_avg', 'seasonal')
            threshold : float
                Anomaly detection threshold
            window_size : int
                Window size for moving average
            """
            self.method = method
            self.threshold = threshold
            self.window_size = window_size
            self.fitted = False
    
        def fit(self, X):
            """Learn statistics from training data"""
            if self.method == 'zscore':
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
            elif self.method == 'iqr':
                self.q1_ = np.percentile(X, 25, axis=0)
                self.q3_ = np.percentile(X, 75, axis=0)
                self.iqr_ = self.q3_ - self.q1_
            elif self.method == 'mahalanobis':
                self.mean_ = np.mean(X, axis=0)
                self.cov_ = np.cov(X.T)
                self.cov_inv_ = np.linalg.inv(self.cov_)
    
            self.fitted = True
            return self
    
        def predict(self, X):
            """Calculate anomaly scores and detect anomalies"""
            if not self.fitted and self.method not in ['moving_avg', 'seasonal']:
                raise ValueError("Model is not fitted. Please run fit() first.")
    
            if self.method == 'zscore':
                scores = np.abs((X - self.mean_) / self.std_)
                anomalies = np.any(scores > self.threshold, axis=1)
    
            elif self.method == 'iqr':
                lower = self.q1_ - 1.5 * self.iqr_
                upper = self.q3_ + 1.5 * self.iqr_
                anomalies = np.any((X < lower) | (X > upper), axis=1)
                scores = np.max(np.abs(X - self.mean_) / self.std_, axis=1)
    
            elif self.method == 'mahalanobis':
                from scipy.spatial.distance import mahalanobis
                scores = np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])
                anomalies = scores > self.threshold
    
            elif self.method == 'moving_avg':
                # Supports 1D time series only
                moving_avg = np.convolve(X.flatten(), np.ones(self.window_size)/self.window_size, mode='same')
                moving_std = np.array([X.flatten()[max(0, i-self.window_size//2):min(len(X), i+self.window_size//2)].std()
                                       for i in range(len(X))])
                scores = np.abs(X.flatten() - moving_avg)
                anomalies = scores > self.threshold * moving_std
    
            elif self.method == 'seasonal':
                # Supports 1D time series only
                result = seasonal_decompose(X.flatten(), model='additive', period=self.window_size, extrapolate_trend='freq')
                scores = np.abs(result.resid)
                threshold_val = self.threshold * np.nanstd(result.resid)
                anomalies = scores > threshold_val
    
            return anomalies.astype(int), scores
    
        def fit_predict(self, X):
            """Execute training and prediction at once"""
            self.fit(X)
            return self.predict(X)
    
    # Demonstration
    np.random.seed(42)
    
    # Dataset generation
    n_samples = 300
    n_features = 2
    X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n_samples)
    X_outliers = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5]])
    X = np.vstack([X_normal, X_outliers])
    y_true = np.array([0]*n_samples + [1]*len(X_outliers))
    
    # Anomaly detection with each method
    methods = ['zscore', 'iqr', 'mahalanobis']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, method in enumerate(methods):
        detector = StatisticalAnomalyDetector(method=method, threshold=3.0)
        detector.fit(X_normal)  # Train on normal data only
        y_pred, scores = detector.predict(X)
    
        # Visualization
        axes[i].scatter(X[y_pred==0, 0], X[y_pred==0, 1],
                        alpha=0.6, s=50, c='blue', label='Normal', edgecolors='black')
        axes[i].scatter(X[y_pred==1, 0], X[y_pred==1, 1],
                        c='red', s=150, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
        axes[i].set_xlabel('Feature 1', fontsize=12)
        axes[i].set_ylabel('Feature 2', fontsize=12)
        axes[i].set_title(f'{method.upper()} Method', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
        # Evaluation
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"=== {method.upper()} Method ===")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")
    
    plt.tight_layout()
    plt.show()
    

> **Important** : In practice, ensembles combining multiple statistical methods are effective. By leveraging the strengths of each method, robust anomaly detection becomes possible.

* * *

## Chapter Summary

### What We Learned

  1. **Statistical Outlier Detection**

     * Z-score: Simple and fast, assumes normal distribution
     * IQR: Robust, distribution-independent
  2. **Probability Distribution-Based**

     * Mahalanobis distance: Multivariate, considers covariance
     * Multivariate Gaussian distribution: Probability density-based determination
  3. **Statistical Hypothesis Testing**

     * Grubbs' Test: Rigorous testing for single outliers
     * ESD Test: Sequential detection of multiple outliers
  4. **Time Series Anomaly Detection**

     * Moving average: Trend following and residual detection
     * Seasonal decomposition: Removing seasonality and trend
  5. **Implementation and Applications**

     * Integrated pipeline: Unified interface for multiple methods
     * Practical applications: Method selection based on domain

### Criteria for Selecting Statistical Methods

Method | Data Type | Advantages | Disadvantages  
---|---|---|---  
**Z-score** | Univariate, normal distribution | Simple, fast | High impact from outliers  
**IQR** | Univariate, any distribution | Robust | Not suitable for multivariate  
**Mahalanobis** | Multivariate, correlated | Considers covariance | High computational cost  
**Grubbs/ESD** | Univariate, normal distribution | Clear statistical basis | Sequential processing  
**Moving average** | Time series | Trend following | Has lag  
**Seasonal decomposition** | Seasonal time series | Removes seasonality | Requires prior knowledge of period  
  
### Next Chapter

In Chapter 3, we will learn about **machine learning-based anomaly detection** :

  * Isolation Forest, LOF
  * One-Class SVM
  * Clustering-based methods
  * Ensemble methods

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the differences between Z-score and IQR methods, and describe in which situations each is appropriate.

Sample Answer

**Answer** :

**Z-score** :

  * Formula: $(x - \mu) / \sigma$
  * Assumption: Data follows normal distribution
  * Threshold: Typically $|Z| > 3$
  * Characteristics: Uses mean and standard deviation, sensitive to outliers

**IQR method** :

  * Formula: $IQR = Q3 - Q1$, anomalies are $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$
  * Assumption: Distribution-independent (non-parametric)
  * Characteristics: Uses quartiles, robust against outliers

**Application scenarios** :

Situation | Recommended Method | Reason  
---|---|---  
Data close to normal distribution | Z-score | Clear statistical basis  
Unknown/non-normal distribution | IQR | No distribution assumption needed  
Outliers already mixed in | IQR | High robustness  
Fast processing needed | Z-score | Simple computation  
  
### Problem 2 (Difficulty: medium)

Explain the superiority of Mahalanobis distance over Euclidean distance using an example of correlated data. Include simple Python code.

Sample Answer

**Answer** :

**Advantages of Mahalanobis distance** :

  1. **Considers covariance** : Reflects correlation between variables
  2. **Scale-invariant** : Independent of feature scales
  3. **Elliptical boundary** : Adapts to data distribution shape

**Implementation example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import euclidean, mahalanobis
    
    # Generate highly correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.9], [0.9, 1]]  # Correlation coefficient 0.9
    data = np.random.multivariate_normal(mean, cov, size=300)
    
    # Test points (outside data distribution)
    test_point1 = np.array([2, 2])  # Point along correlation direction
    test_point2 = np.array([2, -2])  # Point perpendicular to correlation direction
    
    # Distance calculation
    mean_vec = data.mean(axis=0)
    cov_matrix = np.cov(data.T)
    cov_inv = np.linalg.inv(cov_matrix)
    
    euclidean_dist1 = euclidean(test_point1, mean_vec)
    euclidean_dist2 = euclidean(test_point2, mean_vec)
    mahal_dist1 = mahalanobis(test_point1, mean_vec, cov_inv)
    mahal_dist2 = mahalanobis(test_point2, mean_vec, cov_inv)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30, c='blue', label='Data')
    plt.scatter(*test_point1, c='red', s=200, marker='X', label='Point 1 (correlation direction)',
                edgecolors='black', linewidths=2, zorder=5)
    plt.scatter(*test_point2, c='orange', s=200, marker='X', label='Point 2 (perpendicular direction)',
                edgecolors='black', linewidths=2, zorder=5)
    plt.scatter(*mean_vec, c='green', s=200, marker='o', label='Center',
                edgecolors='black', linewidths=2, zorder=5)
    
    # Confidence ellipse
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 3 * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean_vec, width, height, angle=angle,
                      edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
    plt.gca().add_patch(ellipse)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Euclidean Distance vs Mahalanobis Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print("=== Distance Comparison ===")
    print(f"Point 1 (correlation direction):")
    print(f"  Euclidean distance: {euclidean_dist1:.3f}")
    print(f"  Mahalanobis distance: {mahal_dist1:.3f}")
    print(f"\nPoint 2 (perpendicular direction):")
    print(f"  Euclidean distance: {euclidean_dist2:.3f}")
    print(f"  Mahalanobis distance: {mahal_dist2:.3f}")
    print(f"\n→ Euclidean distances are equal, but Mahalanobis distance is farther for Point 2")
    print("  (correctly reflects data distribution shape)")
    

**Conclusion** :

  * Euclidean distance judges both points as equidistant ($\sqrt{8} \approx 2.83$)
  * Mahalanobis distance correctly identifies Point 2 as anomalous
  * By considering correlation, it captures the true distribution of data

### Problem 3 (Difficulty: medium)

Explain the differences between Grubbs' Test and Generalized ESD Test, and state which is more appropriate when multiple outliers are present.

Sample Answer

**Answer** :

**Grubbs' Test** :

  * **Purpose** : Detect a single outlier
  * **Procedure** : Test the most extreme single value
  * **Problem** : Fails to detect when multiple outliers exist due to masking effect
  * **Masking effect** : Multiple outliers distort each other's mean and standard deviation, hindering detection

**Generalized ESD Test** :

  * **Purpose** : Detect multiple outliers (up to k maximum)
  * **Procedure** : Sequentially remove outliers while repeating the test
  * **Advantage** : Avoids masking effect
  * **Note** : Requires specifying maximum number of outliers k in advance

**Recommendations** :

Situation | Recommended Method | Reason  
---|---|---  
Certain only one outlier exists | Grubbs' Test | Simple and clear  
Possibility of multiple outliers | Generalized ESD | Avoids masking  
Number of outliers unknown | Generalized ESD | Set k conservatively  
  
**Concrete example** :

Data: [50, 51, 49, 52, 48, 100, 105] (2 outliers: 100, 105)

  * Grubbs' Test: Detects only 105 (100 fails detection due to distorted mean)
  * ESD Test: Detects 105, removes it → detects 100 (succeeds through sequential processing)

### Problem 4 (Difficulty: hard)

Explain why setting the period parameter is important in time series anomaly detection using seasonal decomposition (STL), and show the problems when an incorrect period is set. Include implementation examples.

Sample Answer

**Answer** :

**Importance of period parameter** :

  1. **Accurate seasonal removal** : If not decomposed with correct period, seasonal components leak into residuals
  2. **Anomaly detection accuracy** : If seasonality remains in residuals, false positives (FP) increase
  3. **Trend estimation** : If period is inappropriate, trend component also gets distorted

**Problems with incorrect period settings** :

  * **Undersetting** (shorter than true period): Over-removes seasonality, misidentifies true trend as seasonality
  * **Oversetting** (longer than true period): Seasonality remains in residuals, judges normal seasonal variations as anomalies

**Implementation example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Implementation example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Generate time series data with true period 30
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    trend = 0.05 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 30)  # Period 30
    noise = np.random.normal(0, 1, n_samples)
    data = trend + seasonal + noise
    
    # Add anomalies
    data[100] += 25
    data[200] -= 25
    
    # Decompose with 3 different period settings
    periods = [15, 30, 60]  # Undersetting, accurate, oversetting
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    for i, period in enumerate(periods):
        result = seasonal_decompose(data, model='additive', period=period, extrapolate_trend='freq')
    
        # Original data
        axes[i, 0].plot(time, data, linewidth=1, alpha=0.7)
        axes[i, 0].set_ylabel(f'Period={period}\nOriginal Data', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
    
        # Seasonality
        axes[i, 1].plot(time, result.seasonal, linewidth=1, color='orange')
        axes[i, 1].set_ylabel('Seasonality', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)
    
        # Residual and anomaly detection
        residual = result.resid
        threshold = 3 * np.nanstd(residual)
        anomalies = np.abs(residual) > threshold
    
        axes[i, 2].plot(time, residual, linewidth=1, alpha=0.7)
        axes[i, 2].axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        axes[i, 2].axhline(y=-threshold, color='red', linestyle='--', linewidth=2)
        axes[i, 2].scatter(time[anomalies], residual[anomalies],
                           c='red', s=50, marker='X', zorder=5)
        axes[i, 2].set_ylabel('Residual', fontsize=10)
        axes[i, 2].grid(True, alpha=0.3)
    
        # Evaluation
        print(f"=== Period={period} ===")
        print(f"Anomalies detected: {anomalies.sum()}")
        print(f"Anomaly indices: {np.where(anomalies)[0]}\n")
    
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Seasonal Component', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Residual (Anomaly Detection)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Time', fontsize=11)
    axes[2, 1].set_xlabel('Time', fontsize=11)
    axes[2, 2].set_xlabel('Time', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Conclusion ===")
    print("Period=15 (undersetting): Seasonality becomes complex, false detections increase")
    print("Period=30 (accurate): Proper decomposition, accurately detects anomalies")
    print("Period=60 (oversetting): Seasonality remains in residuals, false detection of normal variations")
    

**Conclusion** :

  * Accurate period setting determines anomaly detection accuracy
  * Period is determined by prior knowledge (business cycles, seasonal patterns) or ACF/PACF analysis
  * When unknown, try multiple periods and select the one with minimum residual variance

### Problem 5 (Difficulty: hard)

Design an ensemble anomaly detection system that combines three statistical methods (Z-score, IQR, Mahalanobis distance). Implement majority voting or weighted voting, and demonstrate that performance improves over single methods.

Sample Answer

**Answer** :

**Ensemble anomaly detection design** :

  1. **Combining basic methods** : Integrate methods with different principles (Z-score, IQR, Mahalanobis)
  2. **Voting strategy** : Majority voting or soft voting (score averaging)
  3. **Weighting** : Weights according to reliability of each method

**Implementation** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.spatial.distance import mahalanobis
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    class EnsembleAnomalyDetector:
        """Ensemble anomaly detection"""
    
        def __init__(self, voting='hard', weights=None):
            """
            Parameters:
            -----------
            voting : str
                'hard' (majority voting) or 'soft' (score averaging)
            weights : list or None
                Weights for each method [zscore, iqr, mahalanobis]
            """
            self.voting = voting
            self.weights = weights if weights is not None else [1/3, 1/3, 1/3]
    
        def fit(self, X):
            """Learn statistics"""
            # For Z-score
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
    
            # For IQR
            self.q1_ = np.percentile(X, 25, axis=0)
            self.q3_ = np.percentile(X, 75, axis=0)
            self.iqr_ = self.q3_ - self.q1_
    
            # For Mahalanobis
            self.cov_ = np.cov(X.T)
            self.cov_inv_ = np.linalg.inv(self.cov_)
    
            return self
    
        def predict(self, X, threshold_zscore=3, threshold_mahal=3):
            """Ensemble prediction"""
            n_samples = len(X)
    
            # 1. Z-score
            z_scores = np.abs((X - self.mean_) / self.std_)
            z_anomalies = np.any(z_scores > threshold_zscore, axis=1).astype(int)
            z_scores_norm = np.max(z_scores, axis=1) / 5  # Normalization
    
            # 2. IQR
            lower = self.q1_ - 1.5 * self.iqr_
            upper = self.q3_ + 1.5 * self.iqr_
            iqr_anomalies = np.any((X < lower) | (X > upper), axis=1).astype(int)
            iqr_scores = np.max(np.abs(X - self.mean_) / (self.iqr_ + 1e-10), axis=1)
            iqr_scores_norm = np.clip(iqr_scores / 5, 0, 1)  # Normalization
    
            # 3. Mahalanobis distance
            mahal_scores = np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])
            mahal_anomalies = (mahal_scores > threshold_mahal).astype(int)
            mahal_scores_norm = mahal_scores / 10  # Normalization
    
            # Ensemble voting
            if self.voting == 'hard':
                # Majority voting (2/3 or more judge as anomaly)
                votes = z_anomalies + iqr_anomalies + mahal_anomalies
                predictions = (votes >= 2).astype(int)
                scores = votes / 3
    
            elif self.voting == 'soft':
                # Weighted average of scores
                scores = (self.weights[0] * z_scores_norm +
                         self.weights[1] * iqr_scores_norm +
                         self.weights[2] * mahal_scores_norm)
                predictions = (scores > 0.5).astype(int)
    
            return predictions, scores, {
                'zscore': z_anomalies,
                'iqr': iqr_anomalies,
                'mahalanobis': mahal_anomalies
            }
    
    # Evaluation experiment
    np.random.seed(42)
    
    # Data generation
    n_normal = 300
    n_anomaly = 30
    X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n_normal)
    X_anomaly = np.random.uniform(-5, 5, size=(n_anomaly, 2))
    X_anomaly += np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]]).mean(axis=0)  # Bias
    
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0]*n_normal + [1]*n_anomaly)
    
    # Model training (normal data only)
    ensemble = EnsembleAnomalyDetector(voting='soft', weights=[0.3, 0.3, 0.4])
    ensemble.fit(X_normal)
    
    # Prediction
    y_pred_ensemble, scores_ensemble, individual_preds = ensemble.predict(X)
    
    # Individual evaluation of each method
    results = {}
    for method_name, preds in individual_preds.items():
        results[method_name] = {
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'f1': f1_score(y_true, preds)
        }
    
    # Ensemble evaluation
    results['ensemble'] = {
        'precision': precision_score(y_true, y_pred_ensemble),
        'recall': recall_score(y_true, y_pred_ensemble),
        'f1': f1_score(y_true, y_pred_ensemble)
    }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    methods = ['zscore', 'iqr', 'mahalanobis', 'ensemble']
    titles = ['Z-score', 'IQR', 'Mahalanobis Distance', 'Ensemble']
    predictions_list = [individual_preds['zscore'], individual_preds['iqr'],
                        individual_preds['mahalanobis'], y_pred_ensemble]
    
    for i, (method, title, preds) in enumerate(zip(methods, titles, predictions_list)):
        ax = axes[i // 2, i % 2]
        ax.scatter(X[preds==0, 0], X[preds==0, 1],
                   alpha=0.6, s=50, c='blue', label='Normal', edgecolors='black')
        ax.scatter(X[preds==1, 0], X[preds==1, 1],
                   c='red', s=100, marker='X', label='Anomaly', zorder=5, edgecolors='black', linewidths=2)
    
        # Display performance
        r = results[method]
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(f'{title}\nF1={r["f1"]:.3f}, Precision={r["precision"]:.3f}, Recall={r["recall"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results summary
    print("=== Performance Comparison ===")
    for method, metrics in results.items():
        print(f"{method.upper():15s} - Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
    
    print("\n=== Conclusion ===")
    print("Ensemble combines strengths of each method, achieving more stable performance than single methods")
    

**Conclusion** :

  * Ensemble mitigates bias of single methods
  * Soft voting has high flexibility and can be optimized by weight adjustment
  * In practice, set weights based on domain knowledge

* * *

## References

  1. Rousseeuw, P. J., & Hubert, M. (2011). _Robust statistics for outlier detection_. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 1(1), 73-79.
  2. Barnett, V., & Lewis, T. (1994). _Outliers in statistical data_ (3rd ed.). John Wiley & Sons.
  3. Grubbs, F. E. (1969). _Procedures for detecting outlying observations in samples_. Technometrics, 11(1), 1-21.
  4. Rosner, B. (1983). _Percentage points for a generalized ESD many-outlier procedure_. Technometrics, 25(2), 165-172.
  5. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). _STL: A seasonal-trend decomposition procedure based on loess_. Journal of Official Statistics, 6(1), 3-73.
