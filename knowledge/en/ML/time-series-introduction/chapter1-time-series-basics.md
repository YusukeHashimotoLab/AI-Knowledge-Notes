---
title: "Chapter 1: Time Series Data Fundamentals"
chapter_title: "Chapter 1: Time Series Data Fundamentals"
subtitle: The Foundation of Time Series Analysis - Data Understanding and Preprocessing
reading_time: 25-30 minutes
difficulty: Beginner
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers the fundamentals of Time Series Data Fundamentals, which what is time series data. You will learn Handle time series data with pandas, Perform visualization, and concept of stationarity.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the definition and characteristics of time series data
  * ✅ Handle time series data with pandas
  * ✅ Perform visualization and exploratory analysis of time series
  * ✅ Understand the concept of stationarity and testing methods
  * ✅ Interpret autocorrelation and partial autocorrelation
  * ✅ Execute preprocessing of time series data

* * *

## 1.1 What is Time Series Data

### Definition and Characteristics of Time Series

**Time Series Data** is a collection of observations recorded in chronological order.

> An important characteristic of time series data is the existence of **temporal dependencies** between data points.

### Properties of Time Series Data

Property | Description | Example  
---|---|---  
**Temporal Order** | The order of data is important | Past stock prices influence the future  
**Autocorrelation** | Past values correlate with current values | Continuity in temperature  
**Trend** | Long-term upward or downward tendency | Sales growth  
**Seasonality** | Periodic patterns | Increased power consumption in summer  
**Non-stationarity** | Statistical properties change over time | Stock price volatility changes  
  
### Types of Time Series Data

Classification | Description | Example  
---|---|---  
**Regular** | Observed at constant intervals | Daily stock prices, hourly temperature  
**Irregular** | Observed at irregular intervals | Event logs, transaction data  
**Univariate** | Observing one variable | Temperature only  
**Multivariate** | Observing multiple variables simultaneously | Temperature, humidity, air pressure  
  
### Time Series Analysis in Business

Time series analysis has broad applications across industries. **Demand Forecasting** enables optimization of sales, inventory, and logistics operations. **Financial Analysis** supports stock price prediction and risk management. **Anomaly Detection** powers system monitoring and fraud detection capabilities. **Sensor Data** applications include IoT systems and quality control in manufacturing. **Economic Analysis** examines indicators such as GDP, unemployment rate, and inflation.

### Time Series Data Basics with pandas
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Time Series Data Basics with pandas
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create time series data
    np.random.seed(42)
    ts_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 500, len(dates)) + np.arange(len(dates)) * 0.5,
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.randn(len(dates)) * 2
    })
    
    # Set date column as index
    ts_data.set_index('date', inplace=True)
    
    print("=== Time Series Data Overview ===")
    print(ts_data.head(10))
    print(f"\nData types:\n{ts_data.dtypes}")
    print(f"\nIndex type: {type(ts_data.index)}")
    print(f"\nBasic statistics:\n{ts_data.describe()}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(ts_data.index, ts_data['sales'], color='blue', alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Daily Sales Trend', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(ts_data.index, ts_data['temperature'], color='red', alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Daily Temperature Trend', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Convenient datetime Operations
    
    
    # Parse dates
    date_str = '2023-01-15'
    parsed_date = pd.to_datetime(date_str)
    print(f"Parsed date: {parsed_date}")
    print(f"Type: {type(parsed_date)}")
    
    # Create date ranges
    # Daily
    daily = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    print(f"\nDaily: {daily[:5]}")
    
    # Weekly (Sunday start)
    weekly = pd.date_range('2023-01-01', periods=10, freq='W')
    print(f"Weekly: {weekly[:3]}")
    
    # Monthly (month end)
    monthly = pd.date_range('2023-01-01', periods=12, freq='M')
    print(f"Monthly: {monthly[:3]}")
    
    # Hourly
    hourly = pd.date_range('2023-01-01', periods=24, freq='H')
    print(f"Hourly: {hourly[:3]}")
    
    # Extract date components
    ts_data['year'] = ts_data.index.year
    ts_data['month'] = ts_data.index.month
    ts_data['day'] = ts_data.index.day
    ts_data['dayofweek'] = ts_data.index.dayofweek  # Monday=0
    ts_data['quarter'] = ts_data.index.quarter
    
    print("\n=== Date Component Extraction ===")
    print(ts_data.head())
    
    # Slicing
    print("\n=== Time Series Slicing ===")
    print(f"Data for January 2023:\n{ts_data['2023-01'].head()}")
    print(f"\nJanuary 1 to January 7:\n{ts_data['2023-01-01':'2023-01-07']}")
    

* * *

## 1.2 Time Series Visualization and Exploration

### Time Series Plots
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Time Series Plots
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate more complex time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Trend + Seasonality + Noise
    trend = np.arange(len(dates)) * 0.5
    seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.randn(len(dates)) * 20
    
    sales = 1000 + trend + seasonality + noise
    
    ts = pd.Series(sales, index=dates, name='sales')
    
    # Basic visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Full period
    axes[0].plot(ts.index, ts.values, color='steelblue', linewidth=1)
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Time Series Plot - Full Period', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 2023 only
    ts_2023 = ts['2023']
    axes[1].plot(ts_2023.index, ts_2023.values, color='coral', linewidth=1.5)
    axes[1].set_ylabel('Sales')
    axes[1].set_title('Time Series Plot - 2023', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Q1 2023 only
    ts_q1 = ts['2023-01':'2023-03']
    axes[2].plot(ts_q1.index, ts_q1.values, color='green', linewidth=2, marker='o', markersize=3)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Sales')
    axes[2].set_title('Time Series Plot - Q1 2023', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Basic Statistics ===")
    print(ts.describe())
    

### Rolling Statistics (Moving Average)
    
    
    # Moving average and moving standard deviation
    rolling_mean_7 = ts.rolling(window=7).mean()
    rolling_mean_30 = ts.rolling(window=30).mean()
    rolling_std_30 = ts.rolling(window=30).std()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Moving average
    axes[0].plot(ts.index, ts.values, label='Original Data', alpha=0.5, linewidth=0.8)
    axes[0].plot(rolling_mean_7.index, rolling_mean_7.values,
                 label='7-Day Moving Average', color='orange', linewidth=2)
    axes[0].plot(rolling_mean_30.index, rolling_mean_30.values,
                 label='30-Day Moving Average', color='red', linewidth=2)
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Moving Average', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Moving standard deviation
    axes[1].plot(rolling_std_30.index, rolling_std_30.values,
                 color='purple', linewidth=2)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('30-Day Moving Standard Deviation (Volatility)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Moving Statistics ===")
    print(f"7-Day Moving Average (Latest 5):\n{rolling_mean_7.tail()}")
    print(f"\n30-Day Moving Standard Deviation (Latest 5):\n{rolling_std_30.tail()}")
    

### Time Series Decomposition

Time series can be decomposed into the following components:

  * **Trend** : Long-term tendency
  * **Seasonality** : Periodic patterns
  * **Residual** : Random noise

Decomposition models:

  * **Additive model** : $y_t = T_t + S_t + R_t$
  * **Multiplicative model** : $y_t = T_t \times S_t \times R_t$

    
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Seasonal decomposition (additive model)
    decomposition = seasonal_decompose(ts, model='additive', period=365)
    
    # Get decomposition results
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original data
    axes[0].plot(ts.index, ts.values, color='blue', linewidth=1)
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Original Time Series Data', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(trend.index, trend.values, color='red', linewidth=2)
    axes[1].set_ylabel('Trend')
    axes[1].set_title('Trend Component', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonality
    axes[2].plot(seasonal.index, seasonal.values, color='green', linewidth=1)
    axes[2].set_ylabel('Seasonality')
    axes[2].set_title('Seasonal Component', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(residual.index, residual.values, color='purple', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Residual')
    axes[3].set_title('Residual Component', fontsize=14)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Decomposition Statistics ===")
    print(f"Trend:\n{trend.describe()}")
    print(f"\nSeasonality:\n{seasonal.describe()}")
    print(f"\nResidual:\n{residual.describe()}")
    

* * *

## 1.3 Stationarity

### Definition of Stationarity

**Stationarity** refers to the statistical properties of a time series not changing over time.

### Weak Stationarity

Satisfies the following three conditions:

  1. **Constant mean** : $E[y_t] = \mu$ (constant for all $t$)
  2. **Constant variance** : $\text{Var}[y_t] = \sigma^2$ (constant for all $t$)
  3. **Autocovariance depends only on lag** : $\text{Cov}(y_t, y_{t-k})$ depends only on $k$

### Strict Stationarity

For any set of time points $\\{t_1, t_2, \ldots, t_n\\}$ and any lag $k$,

$$ F(y_{t_1}, y_{t_2}, \ldots, y_{t_n}) = F(y_{t_1+k}, y_{t_2+k}, \ldots, y_{t_n+k}) $$ 

In practice, weak stationarity is treated as "stationarity".

### Importance of Stationarity

Stationarity is fundamental to time series analysis for several reasons. Many time series models such as ARIMA assume stationarity as a prerequisite. Non-stationary data leads to unstable predictions and unreliable parameter estimates. Stationarization through appropriate transformations improves prediction accuracy and model reliability.

### ADF Test (Augmented Dickey-Fuller Test)

**Null hypothesis** : The time series is non-stationary (has a unit root)

**Alternative hypothesis** : The time series is stationary
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    import numpy as np
    
    # Generate non-stationary data (random walk)
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(1000))
    
    # Generate stationary data (white noise)
    white_noise = np.random.randn(1000)
    
    def adf_test(series, name):
        """Execute ADF test and display results"""
        result = adfuller(series, autolag='AIC')
    
        print(f"\n=== ADF Test for {name} ===")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Number of Lags: {result[2]}")
        print(f"Number of Observations: {result[3]}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
    
        if result[1] < 0.05:
            print("Conclusion: Stationary (p < 0.05)")
        else:
            print("Conclusion: Non-stationary (p >= 0.05)")
    
        return result
    
    # Execute tests
    adf_random_walk = adf_test(random_walk, "Random Walk (Non-stationary)")
    adf_white_noise = adf_test(white_noise, "White Noise (Stationary)")
    adf_sales = adf_test(ts.values, "Sales Data")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(random_walk, color='red', linewidth=1)
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Random Walk (Non-stationary) - ADF p-value: {adf_random_walk[1]:.4f}',
                      fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(white_noise, color='green', linewidth=1)
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'White Noise (Stationary) - ADF p-value: {adf_white_noise[1]:.4f}',
                      fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ts.values, color='blue', linewidth=1)
    axes[2].set_xlabel('Time Point')
    axes[2].set_ylabel('Sales')
    axes[2].set_title(f'Sales Data - ADF p-value: {adf_sales[1]:.4f}', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin Test)

**Null hypothesis** : The time series is stationary

**Alternative hypothesis** : The time series is non-stationary

> **Note** : KPSS has the opposite null hypothesis from ADF. Using both tests together allows for more reliable judgment.
    
    
    from statsmodels.tsa.stattools import kpss
    
    def kpss_test(series, name):
        """Execute KPSS test and display results"""
        result = kpss(series, regression='c', nlags='auto')
    
        print(f"\n=== KPSS Test for {name} ===")
        print(f"KPSS Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Number of Lags: {result[2]}")
        print(f"Critical Values:")
        for key, value in result[3].items():
            print(f"  {key}: {value:.4f}")
    
        if result[1] < 0.05:
            print("Conclusion: Non-stationary (p < 0.05)")
        else:
            print("Conclusion: Stationary (p >= 0.05)")
    
        return result
    
    # Execute tests
    kpss_random_walk = kpss_test(random_walk, "Random Walk (Non-stationary)")
    kpss_white_noise = kpss_test(white_noise, "White Noise (Stationary)")
    
    # Integrated judgment of test results
    print("\n=== Integrated Judgment (ADF & KPSS) ===")
    results = [
        ("Random Walk", adf_random_walk[1], kpss_random_walk[1]),
        ("White Noise", adf_white_noise[1], kpss_white_noise[1])
    ]
    
    for name, adf_p, kpss_p in results:
        print(f"\n{name}:")
        print(f"  ADF p-value: {adf_p:.4f} ({'Stationary' if adf_p < 0.05 else 'Non-stationary'})")
        print(f"  KPSS p-value: {kpss_p:.4f} ({'Non-stationary' if kpss_p < 0.05 else 'Stationary'})")
    
        if adf_p < 0.05 and kpss_p >= 0.05:
            print("  → Conclusion: Stationary")
        elif adf_p >= 0.05 and kpss_p < 0.05:
            print("  → Conclusion: Non-stationary")
        else:
            print("  → Conclusion: Tests disagree (requires additional analysis)")
    

* * *

## 1.4 Autocorrelation and Partial Autocorrelation

### ACF (Autocorrelation Function)

**Autocorrelation** is the correlation coefficient between a time series and its lagged version.

$$ \text{ACF}(k) = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} $$ 

  * $k$: Lag (time difference)
  * Value range: $[-1, 1]$

### PACF (Partial Autocorrelation Function)

**Partial autocorrelation** is the correlation after removing the effects of intermediate lags.

  * $\text{PACF}(k)$: Direct correlation at lag $k$
  * Removes the influence of intermediate lags $1, 2, \ldots, k-1$

### ACF and PACF Plots
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: ACF and PACF Plots
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Generate different types of time series data
    np.random.seed(42)
    n = 500
    
    # 1. AR(1) process: y_t = 0.7 * y_{t-1} + ε_t
    ar1 = [0]
    for _ in range(n):
        ar1.append(0.7 * ar1[-1] + np.random.randn())
    ar1 = np.array(ar1[1:])
    
    # 2. MA(1) process: y_t = ε_t + 0.7 * ε_{t-1}
    ma1 = []
    epsilon = np.random.randn(n + 1)
    for i in range(n):
        ma1.append(epsilon[i] + 0.7 * epsilon[i-1])
    ma1 = np.array(ma1)
    
    # 3. White noise
    white_noise = np.random.randn(n)
    
    # ACF/PACF plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    series_list = [
        (ar1, 'AR(1) Process'),
        (ma1, 'MA(1) Process'),
        (white_noise, 'White Noise')
    ]
    
    for i, (series, name) in enumerate(series_list):
        # Time series plot
        axes[i, 0].plot(series, linewidth=1)
        axes[i, 0].set_title(name, fontsize=12)
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
    
        # ACF
        plot_acf(series, lags=40, ax=axes[i, 1], alpha=0.05)
        axes[i, 1].set_title(f'{name} - ACF', fontsize=12)
        axes[i, 1].grid(True, alpha=0.3)
    
        # PACF
        plot_pacf(series, lags=40, ax=axes[i, 2], alpha=0.05)
        axes[i, 2].set_title(f'{name} - PACF', fontsize=12)
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Interpretation of ACF/PACF Patterns ===")
    print("\nAR(1) Process:")
    print("  - ACF: Exponentially decaying")
    print("  - PACF: Cuts off after lag 1 (zero thereafter)")
    
    print("\nMA(1) Process:")
    print("  - ACF: Cuts off after lag 1 (zero thereafter)")
    print("  - PACF: Exponentially decaying")
    
    print("\nWhite Noise:")
    print("  - ACF: Near zero at all lags")
    print("  - PACF: Near zero at all lags")
    

### Correlogram

ACF/PACF analysis on real data:
    
    
    # ACF/PACF of sales data
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time series plot
    axes[0].plot(ts.index, ts.values, linewidth=1)
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Sales Data', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF
    plot_acf(ts.values, lags=100, ax=axes[1], alpha=0.05)
    axes[1].set_title('ACF (Autocorrelation)', fontsize=14)
    axes[1].set_xlabel('Lag')
    axes[1].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(ts.values, lags=100, ax=axes[2], alpha=0.05)
    axes[2].set_title('PACF (Partial Autocorrelation)', fontsize=14)
    axes[2].set_xlabel('Lag')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Guidelines for Lag Selection ===")
    print("Determine from ACF/PACF:")
    print("  - Number of significant lags → Determine model order")
    print("  - Seasonal patterns → Identify seasonal lags")
    print("  - Decay pattern → Determine AR or MA")
    

### Model Identification Guidelines

Model | ACF | PACF  
---|---|---  
**AR(p)** | Exponential decay or damped oscillation | Cuts off after lag $p$  
**MA(q)** | Cuts off after lag $q$ | Exponential decay or damped oscillation  
**ARMA(p,q)** | Decays after lag $q$ | Decays after lag $p$  
**White Noise** | Non-significant at all lags | Non-significant at all lags  
  
* * *

## 1.5 Data Preprocessing

### Handling Missing Values
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Handling Missing Values
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Time series data with missing values
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    values = 100 + np.cumsum(np.random.randn(365))
    
    ts_missing = pd.Series(values, index=dates)
    
    # Create random missing values
    missing_indices = np.random.choice(365, size=30, replace=False)
    ts_missing.iloc[missing_indices] = np.nan
    
    print("=== Missing Value Status ===")
    print(f"Missing count: {ts_missing.isnull().sum()}")
    print(f"Missing rate: {ts_missing.isnull().sum() / len(ts_missing) * 100:.2f}%")
    
    # Missing value handling methods
    
    # 1. Forward Fill
    ts_ffill = ts_missing.fillna(method='ffill')
    
    # 2. Backward Fill
    ts_bfill = ts_missing.fillna(method='bfill')
    
    # 3. Linear Interpolation
    ts_interpolate = ts_missing.interpolate(method='linear')
    
    # 4. Spline Interpolation
    ts_spline = ts_missing.interpolate(method='spline', order=2)
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    methods = [
        (ts_missing, 'Original Data (with missing values)'),
        (ts_ffill, 'Forward Fill'),
        (ts_bfill, 'Backward Fill'),
        (ts_interpolate, 'Linear Interpolation'),
        (ts_spline, 'Spline Interpolation'),
        (ts_missing.dropna(), 'Drop Missing')
    ]
    
    for ax, (data, title) in zip(axes.flat, methods):
        ax.plot(data.index, data.values, linewidth=1.5, marker='o' if title == 'Original Data (with missing values)' else '', markersize=2)
        ax.set_ylabel('Value')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Statistics for Each Imputation Method ===")
    print(f"Forward Fill: Mean={ts_ffill.mean():.2f}, Std={ts_ffill.std():.2f}")
    print(f"Backward Fill: Mean={ts_bfill.mean():.2f}, Std={ts_bfill.std():.2f}")
    print(f"Linear Interpolation: Mean={ts_interpolate.mean():.2f}, Std={ts_interpolate.std():.2f}")
    print(f"Spline: Mean={ts_spline.mean():.2f}, Std={ts_spline.std():.2f}")
    

### Differencing

**Differencing transformation** is a fundamental technique to make non-stationary time series stationary.

  * **First-order differencing** : $\Delta y_t = y_t - y_{t-1}$ (removes trend)
  * **Seasonal differencing** : $\Delta_s y_t = y_t - y_{t-s}$ (removes seasonality)

    
    
    from statsmodels.tsa.stattools import adfuller
    
    # Non-stationary data (with trend)
    trend_data = ts.copy()
    
    # First-order differencing
    diff_1 = trend_data.diff().dropna()
    
    # Second-order differencing
    diff_2 = trend_data.diff().diff().dropna()
    
    # Seasonal differencing (7-day cycle)
    diff_seasonal = trend_data.diff(7).dropna()
    
    # Confirm stationarity with ADF test
    def quick_adf(data, name):
        result = adfuller(data, autolag='AIC')
        print(f"{name}: ADF statistic={result[0]:.4f}, p-value={result[1]:.4f} → {'Stationary' if result[1] < 0.05 else 'Non-stationary'}")
    
    print("=== Stationarization by Differencing ===")
    quick_adf(trend_data.values, "Original Data")
    quick_adf(diff_1.values, "First-order Differencing")
    quick_adf(diff_2.values, "Second-order Differencing")
    quick_adf(diff_seasonal.values, "Seasonal Differencing(7)")
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    axes[0].plot(trend_data.index, trend_data.values, linewidth=1)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Original Data (Non-stationary)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(diff_1.index, diff_1.values, linewidth=1, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Differenced Value')
    axes[1].set_title('First-order Differencing', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(diff_2.index, diff_2.values, linewidth=1, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Differenced Value')
    axes[2].set_title('Second-order Differencing', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(diff_seasonal.index, diff_seasonal.values, linewidth=1, color='purple')
    axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Differenced Value')
    axes[3].set_title('Seasonal Differencing (7 days)', fontsize=14)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Log Transformation

**Log transformation** stabilizes variance and converts multiplicative seasonality to additive.
    
    
    # Data with multiplicative seasonality
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    multiplicative = 100 * (1 + 0.001 * np.arange(1000)) * (1 + 0.3 * np.sin(2 * np.pi * np.arange(1000) / 365)) * (1 + 0.1 * np.random.randn(1000))
    ts_mult = pd.Series(multiplicative, index=dates)
    
    # Log transformation
    ts_log = np.log(ts_mult)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    axes[0, 0].plot(ts_mult.index, ts_mult.values, linewidth=1)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Original Data (Multiplicative Seasonality)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # After log transformation
    axes[0, 1].plot(ts_log.index, ts_log.values, linewidth=1, color='orange')
    axes[0, 1].set_ylabel('log(Value)')
    axes[0, 1].set_title('After Log Transformation (Additive Seasonality)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Original data histogram
    axes[1, 0].hist(ts_mult.values, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Original Data Distribution', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-transformed histogram
    axes[1, 1].hist(ts_log.values, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('log(Value)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log-transformed Distribution (Closer to Normal)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Effect of Log Transformation ===")
    print(f"Original Data: Mean={ts_mult.mean():.2f}, Std={ts_mult.std():.2f}, Skewness={ts_mult.skew():.2f}")
    print(f"Log Transformation: Mean={ts_log.mean():.2f}, Std={ts_log.std():.2f}, Skewness={ts_log.skew():.2f}")
    

### Train-Test Split for Time Series

> **Important** : For time series data, we use a future period relative to the training data as test data. Random splitting is not used because it destroys temporal dependencies.
    
    
    # Split time series data
    train_size = int(len(ts) * 0.8)
    
    train = ts[:train_size]
    test = ts[train_size:]
    
    print("=== Train-Test Split ===")
    print(f"Total Data: {len(ts)} records")
    print(f"Training Data: {len(train)} records ({train.index[0]} ~ {train.index[-1]})")
    print(f"Test Data: {len(test)} records ({test.index[0]} ~ {test.index[-1]})")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train.values, label='Training Data', linewidth=1.5, color='blue')
    plt.plot(test.index, test.values, label='Test Data', linewidth=1.5, color='red')
    plt.axvline(x=train.index[-1], color='green', linestyle='--', linewidth=2, label='Split Point')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Train-Test Split for Time Series Data', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Time Series Cross-Validation
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n=== Time Series Cross-Validation Splits ===")
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts.values)):
        print(f"\nFold {i+1}:")
        print(f"  Train: Index {train_idx[0]} ~ {train_idx[-1]} ({len(train_idx)} records)")
        print(f"  Test: Index {test_idx[0]} ~ {test_idx[-1]} ({len(test_idx)} records)")
    
    # Visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts.values)):
        axes[i].plot(ts.index[train_idx], ts.values[train_idx], color='blue', linewidth=1, label='Train')
        axes[i].plot(ts.index[test_idx], ts.values[test_idx], color='red', linewidth=1, label='Test')
        axes[i].set_ylabel('Sales')
        axes[i].set_title(f'Fold {i+1}', fontsize=12)
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()
    

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Time Series Data Fundamentals**

     * Temporal dependencies exist
     * Components: trend, seasonality, residual
     * Date manipulation with pandas
  2. **Visualization and Exploration**

     * Understanding patterns through time series plots
     * Understanding trends with moving statistics
     * Separating components through decomposition
  3. **Stationarity**

     * Three conditions of weak stationarity
     * ADF test and KPSS test
     * Importance of stationarization
  4. **Autocorrelation**

     * ACF: Correlation with all lags
     * PACF: Direct correlation
     * Application to model identification
  5. **Preprocessing**

     * Missing value imputation methods
     * Stationarization through differencing
     * Variance stabilization through log transformation
     * Appropriate train-test split

### Basic Workflow for Time Series Analysis
    
    
    ```mermaid
    graph TD
        A[Raw Data] --> B[Exploratory Analysis]
        B --> C[Visualization & Statistical Examination]
        C --> D[Stationarity Test]
        D --> E{Stationary?}
        E -->|No| F[Differencing & Transformation]
        F --> D
        E -->|Yes| G[ACF/PACF Analysis]
        G --> H[Model Selection]
        H --> I[Prediction & Evaluation]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#fff9c4
        style F fill:#ffccbc
        style G fill:#e8f5e9
        style H fill:#fce4ec
        style I fill:#c8e6c9
    ```

### Next Chapter

In Chapter 2, we will learn about **ARIMA Models** , covering AR (Autoregressive) models, MA (Moving Average) models, and their combination in ARIMA (Autoregressive Integrated Moving Average) models. We will also explore Seasonal ARIMA models (SARIMA) for handling periodic patterns, along with model selection criteria and parameter estimation techniques.

* * *

## Practice Problems

### Problem 1 (Difficulty: easy)

Explain the difference between stationary and non-stationary time series, and describe why stationarity is important.

Sample Answer

**Answer** :

**Stationary Time Series** :

  * Constant mean
  * Constant variance
  * Autocovariance depends only on time lag
  * Examples: White noise, AR(1) (|φ| < 1)

**Non-stationary Time Series** :

  * Mean or variance changes over time
  * Contains trend or seasonality
  * Examples: Random walk, growing sales data

**Why Stationarity is Important** :

  1. **Prediction Stability** : Since statistical properties are constant, future predictions are reliable
  2. **Model Application** : Many time series models (ARIMA, etc.) assume stationarity
  3. **Statistical Inference** : Enables parameter estimation and hypothesis testing
  4. **Generalizability** : Past patterns can be applied to the future

### Problem 2 (Difficulty: medium)

Execute an ADF test on the following data and determine its stationarity.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Execute an ADF test on the following data and determine its 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    np.random.seed(123)
    data = np.cumsum(np.random.randn(200)) + 10
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Execute an ADF test on the following data and determine its 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from statsmodels.tsa.stattools import adfuller
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(123)
    data = np.cumsum(np.random.randn(200)) + 10
    
    # ADF test
    result = adfuller(data, autolag='AIC')
    
    print("=== ADF Test Results ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    print(f"\nCritical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nJudgment:")
    if result[1] < 0.05:
        print("  → Stationary (p < 0.05, reject null hypothesis)")
    else:
        print("  → Non-stationary (p >= 0.05, cannot reject null hypothesis)")
        print("  → Data is random walk (cumulative sum), so non-stationary is reasonable")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(data, linewidth=1.5)
    plt.axhline(y=data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.title(f'Time Series Data (Random Walk) - ADF p-value: {result[1]:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Stationarize with first-order differencing
    diff_data = np.diff(data)
    result_diff = adfuller(diff_data, autolag='AIC')
    
    print(f"\n=== ADF Test After First-order Differencing ===")
    print(f"ADF Statistic: {result_diff[0]:.4f}")
    print(f"p-value: {result_diff[1]:.4f}")
    print(f"Judgment: {'Stationary' if result_diff[1] < 0.05 else 'Non-stationary'}")
    

**Expected Output** :
    
    
    === ADF Test Results ===
    ADF Statistic: -1.2345
    p-value: 0.6543
    → Non-stationary (p >= 0.05)
    
    === ADF Test After First-order Differencing ===
    p-value: 0.0001
    Judgment: Stationary
    

### Problem 3 (Difficulty: medium)

Explain the difference between ACF and PACF, and describe the ACF/PACF patterns for AR(2) and MA(2) processes.

Sample Answer

**Answer** :

**Difference Between ACF and PACF** :

  * **ACF (Autocorrelation Function)** : 
    * Correlation between the time series and its lagged version
    * Includes effects of all intermediate lags
    * $\text{Corr}(y_t, y_{t-k})$
  * **PACF (Partial Autocorrelation Function)** : 
    * Correlation after removing effects of intermediate lags
    * Only the direct effect of lag $k$
    * $\text{Corr}(y_t, y_{t-k} \mid y_{t-1}, \ldots, y_{t-k+1})$

**Patterns** :

Model | ACF | PACF  
---|---|---  
**AR(2)** | Exponential decay or damped oscillation (continues indefinitely) | Cuts off after lag 2 (zero thereafter)  
**MA(2)** | Cuts off after lag 2 (zero thereafter) | Exponential decay or damped oscillation (continues indefinitely)  
  
**AR(2) Example** : $y_t = 0.5y_{t-1} + 0.3y_{t-2} + \epsilon_t$

  * ACF: Gradually decaying pattern
  * PACF: Significant at lags 1 and 2, zero after lag 3

**MA(2) Example** : $y_t = \epsilon_t + 0.5\epsilon_{t-1} + 0.3\epsilon_{t-2}$

  * ACF: Significant at lags 1 and 2, zero after lag 3
  * PACF: Gradually decaying pattern

### Problem 4 (Difficulty: hard)

For the following time series data, perform appropriate preprocessing (missing value handling, stationarization) and execute ADF tests before and after processing.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: For the following time series data, perform appropriate prep
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.arange(365) * 0.5
    seasonal = 50 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.randn(365) * 10
    data = 100 + trend + seasonal + noise
    
    ts = pd.Series(data, index=dates)
    
    # Add missing values
    ts.iloc[50:60] = np.nan
    ts.iloc[200:205] = np.nan
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: For the following time series data, perform appropriate prep
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.arange(365) * 0.5
    seasonal = 50 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.randn(365) * 10
    data = 100 + trend + seasonal + noise
    
    ts = pd.Series(data, index=dates)
    ts.iloc[50:60] = np.nan
    ts.iloc[200:205] = np.nan
    
    print("=== Preprocessing Steps ===\n")
    
    # Step 1: Check missing values
    print(f"1. Missing Value Check")
    print(f"   Missing count: {ts.isnull().sum()} ({ts.isnull().sum() / len(ts) * 100:.2f}%)")
    
    # Step 2: Impute missing values (linear interpolation)
    ts_filled = ts.interpolate(method='linear')
    print(f"\n2. Missing Value Imputation (Linear Interpolation)")
    print(f"   Missing count after imputation: {ts_filled.isnull().sum()}")
    
    # Step 3: Stationarity test (original data)
    result_original = adfuller(ts_filled.values, autolag='AIC')
    print(f"\n3. ADF Test on Original Data")
    print(f"   ADF Statistic: {result_original[0]:.4f}")
    print(f"   p-value: {result_original[1]:.4f}")
    print(f"   Judgment: {'Stationary' if result_original[1] < 0.05 else 'Non-stationary'}")
    
    # Step 4: Stationarization through first-order differencing
    ts_diff = ts_filled.diff().dropna()
    result_diff = adfuller(ts_diff.values, autolag='AIC')
    print(f"\n4. ADF Test After First-order Differencing")
    print(f"   ADF Statistic: {result_diff[0]:.4f}")
    print(f"   p-value: {result_diff[1]:.4f}")
    print(f"   Judgment: {'Stationary' if result_diff[1] < 0.05 else 'Non-stationary'}")
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original data (with missing values)
    axes[0].plot(ts.index, ts.values, linewidth=1, marker='o', markersize=2)
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Original Data (with missing values) - Missing count: {ts.isnull().sum()}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # After missing value imputation
    axes[1].plot(ts_filled.index, ts_filled.values, linewidth=1, color='orange')
    axes[1].set_ylabel('Value')
    axes[1].set_title('After Linear Interpolation', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # First-order differencing
    axes[2].plot(ts_diff.index, ts_diff.values, linewidth=1, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Differenced Value')
    axes[2].set_title(f'First-order Differencing (Stationarized) - ADF p-value: {result_diff[1]:.4f}', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Histogram comparison
    axes[3].hist(ts_filled.values, bins=30, alpha=0.5, label='Original Data', edgecolor='black')
    axes[3].hist(ts_diff.values, bins=30, alpha=0.5, label='First-order Differencing', edgecolor='black')
    axes[3].set_xlabel('Value')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Distribution Comparison', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Preprocessing Complete ===")
    print("✓ Missing value imputation complete")
    print("✓ Stationarization complete (first-order differencing)")
    print(f"✓ Processed data count: {len(ts_diff)}")
    

### Problem 5 (Difficulty: hard)

Explain why random splitting (like in regular machine learning) should not be used for train-test split in time series data. Also, demonstrate appropriate splitting methods.

Sample Answer

**Answer** :

**Reasons Why Random Splitting Should Not Be Used** :

  1. **Destruction of Temporal Dependencies**

     * Time series data has important temporal order
     * Random splitting mixes past and future
     * Autocorrelation structure is destroyed
  2. **Data Leakage**

     * Future data is used for training
     * Test data information leaks into training
     * Performance is overestimated
  3. **Divergence from Reality**

     * In practice, we predict the future
     * Correct approach is to train on past data only and predict the future
     * Random splitting does not reflect real operation

**Appropriate Splitting Methods** :

#### 1\. Simple Time Series Split
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 1. Simple Time Series Split
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    ts = pd.Series(np.random.randn(len(dates)).cumsum(), index=dates)
    
    # 80% training, 20% test
    split_point = int(len(ts) * 0.8)
    train = ts[:split_point]
    test = ts[split_point:]
    
    print("=== Simple Time Series Split ===")
    print(f"Train: {train.index[0]} ~ {train.index[-1]} ({len(train)} records)")
    print(f"Test: {test.index[0]} ~ {test.index[-1]} ({len(test)} records)")
    

#### 2\. Time Series Cross-Validation (TimeSeriesSplit)
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n=== Time Series Cross-Validation ===")
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts)):
        print(f"Fold {i+1}:")
        print(f"  Train: {len(train_idx)} records")
        print(f"  Test: {len(test_idx)} records")
    

#### 3\. Walk-Forward Validation
    
    
    # Fixed window size with rolling validation
    window_size = 365  # 1 year
    test_size = 30     # 30 days
    
    print("\n=== Walk-Forward Validation ===")
    for i in range(0, len(ts) - window_size - test_size, test_size):
        train_start = i
        train_end = i + window_size
        test_end = train_end + test_size
    
        train_fold = ts[train_start:train_end]
        test_fold = ts[train_end:test_end]
    
        print(f"\nFold {i//test_size + 1}:")
        print(f"  Train: {train_fold.index[0]} ~ {train_fold.index[-1]}")
        print(f"  Test: {test_fold.index[0]} ~ {test_fold.index[-1]}")
    

**Example of Incorrect Method** :
    
    
    # ❌ Should NEVER be done
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(ts, test_size=0.2, shuffle=True)  # NG!
    

**Principles of Correct Methods** :

  * Training data is always in the past relative to test data
  * Preserve temporal order
  * Mimic real operation (predict past → future)
  * Prevent data leakage

* * *

## References

  1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.
  2. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts. <https://otexts.com/fpp3/>
  3. Tsay, R. S. (2010). _Analysis of Financial Time Series_ (3rd ed.). Wiley.
  4. Hamilton, J. D. (1994). _Time Series Analysis_. Princeton University Press.
