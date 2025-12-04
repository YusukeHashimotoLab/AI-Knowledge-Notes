---
title: "Chapter 2: Statistical Time Series Models"
chapter_title: "Chapter 2: Statistical Time Series Models"
subtitle: Fundamentals of Time Series Forecasting with AR, MA, and ARIMA Models
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Statistical Time Series Models. You will learn structure of MA (Moving Average) models, ARIMA model framework, and seasonal ARIMA models (SARIMA).

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the definition and parameter estimation of AR (AutoRegressive) models
  * ✅ Understand the structure of MA (Moving Average) models and their relationship with white noise
  * ✅ Master the ARIMA model framework and model selection methods
  * ✅ Implement seasonal ARIMA models (SARIMA)
  * ✅ Properly execute model evaluation and diagnostics
  * ✅ Master the use of statsmodels and pmdarima (auto_arima)

* * *

## 2.1 AR (AutoRegressive) Model

### What is an AR Model

The **AR (AutoRegressive) model** is a linear model that predicts the current value using past values.

Definition of an AR(p) model:

$$ y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t $$

  * $y_t$: Value at time t
  * $\phi_1, \ldots, \phi_p$: AR parameters
  * $p$: AR order (number of lags)
  * $c$: Constant term
  * $\varepsilon_t$: White noise (mean 0, variance $\sigma^2$)

> **Intuitive Understanding** : "Today's temperature is influenced by yesterday's and the day before yesterday's temperatures."

### AR Model Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: AR Model Implementation
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 5-15 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import warnings
    warnings.filterwarnings('ignore')
    
    # Generate sample data (AR(2) process)
    np.random.seed(42)
    n = 200
    epsilon = np.random.normal(0, 1, n)
    
    # AR(2): y_t = 0.6*y_{t-1} - 0.2*y_{t-2} + epsilon_t
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t-1] - 0.2 * y[t-2] + epsilon[t]
    
    # Format as time series data
    ts_data = pd.Series(y, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    print("=== AR(2) Process Simulation ===")
    print(f"Number of data points: {len(ts_data)}")
    print(f"Mean: {ts_data.mean():.3f}")
    print(f"Standard deviation: {ts_data.std():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time series plot
    axes[0].plot(ts_data, linewidth=1.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')
    axes[0].set_title('AR(2) Process', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF (Autocorrelation Function)
    plot_acf(ts_data, lags=30, ax=axes[1])
    axes[1].set_title('Autocorrelation Function (ACF)', fontsize=14)
    
    # PACF (Partial Autocorrelation Function)
    plot_pacf(ts_data, lags=30, ax=axes[2], method='ywm')
    axes[2].set_title('Partial Autocorrelation Function (PACF)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === AR(2) Process Simulation ===
    Number of data points: 200
    Mean: -0.012
    Standard deviation: 1.456
    

> **Important** : By examining the PACF, you can estimate the AR order p. The PACF cuts off at lag p.

### Parameter Estimation and Model Fitting
    
    
    # Train-test data split
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    # Fit AR(2) model
    model_ar2 = AutoReg(train, lags=2, trend='c')
    fitted_ar2 = model_ar2.fit()
    
    print("\n=== AR(2) Model Parameters ===")
    print(fitted_ar2.summary())
    
    # Get parameters
    params = fitted_ar2.params
    print(f"\nEstimated parameters:")
    print(f"Constant: {params['const']:.4f}")
    print(f"phi_1: {params['y.L1']:.4f} (true value: 0.6)")
    print(f"phi_2: {params['y.L2']:.4f} (true value: -0.2)")
    
    # Forecast
    predictions = fitted_ar2.predict(start=len(train), end=len(train) + len(test) - 1)
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label='Training data', linewidth=1.5)
    plt.plot(test.index, test, label='Actual values', linewidth=1.5, color='green')
    plt.plot(test.index, predictions, label='Predictions (AR(2))',
             linewidth=2, linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecasting with AR(2) Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Prediction accuracy
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    
    print(f"\n=== Prediction Accuracy ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    

### Order Selection (AIC, BIC)

**AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)** are metrics that evaluate the balance between model complexity and goodness of fit.

$$ \text{AIC} = 2k - 2\ln(L) $$

$$ \text{BIC} = k\ln(n) - 2\ln(L) $$

  * $k$: Number of parameters
  * $n$: Sample size
  * $L$: Likelihood

    
    
    # Compare AR models with different orders
    max_lag = 10
    aic_values = []
    bic_values = []
    
    for p in range(1, max_lag + 1):
        model = AutoReg(train, lags=p, trend='c')
        fitted = model.fit()
        aic_values.append(fitted.aic)
        bic_values.append(fitted.bic)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(1, max_lag + 1), aic_values, marker='o', linewidth=2)
    axes[0].axvline(x=np.argmin(aic_values) + 1, color='red',
                    linestyle='--', label=f'Minimum AIC (p={np.argmin(aic_values) + 1})')
    axes[0].set_xlabel('AR order p')
    axes[0].set_ylabel('AIC')
    axes[0].set_title('Order Selection by AIC', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, max_lag + 1), bic_values, marker='s',
                 linewidth=2, color='orange')
    axes[1].axvline(x=np.argmin(bic_values) + 1, color='red',
                    linestyle='--', label=f'Minimum BIC (p={np.argmin(bic_values) + 1})')
    axes[1].set_xlabel('AR order p')
    axes[1].set_ylabel('BIC')
    axes[1].set_title('Order Selection by BIC', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Order Selection Results ===")
    print(f"Optimal order (AIC): {np.argmin(aic_values) + 1}")
    print(f"Optimal order (BIC): {np.argmin(bic_values) + 1}")
    print(f"True order: 2")
    

* * *

## 2.2 MA (Moving Average) Model

### What is an MA Model

The **MA (Moving Average) model** expresses the current value as a linear combination of past prediction errors (noise).

Definition of an MA(q) model:

$$ y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q} $$

  * $y_t$: Value at time t
  * $\theta_1, \ldots, \theta_q$: MA parameters
  * $q$: MA order
  * $\mu$: Mean
  * $\varepsilon_t$: White noise

> **Intuitive Understanding** : "Today's temperature is affected by the prediction errors of the past few days."

### Relationship with White Noise
    
    
    from statsmodels.tsa.arima.model import ARIMA
    
    # Simulate MA(1) process
    np.random.seed(42)
    n = 200
    mu = 0
    theta = 0.7
    epsilon = np.random.normal(0, 1, n)
    
    # MA(1): y_t = mu + epsilon_t + theta*epsilon_{t-1}
    y_ma = np.zeros(n)
    y_ma[0] = mu + epsilon[0]
    for t in range(1, n):
        y_ma[t] = mu + epsilon[t] + theta * epsilon[t-1]
    
    ts_ma = pd.Series(y_ma, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    print("=== MA(1) Process Simulation ===")
    print(f"Number of data points: {len(ts_ma)}")
    print(f"Mean: {ts_ma.mean():.3f}")
    print(f"Standard deviation: {ts_ma.std():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time series plot
    axes[0].plot(ts_ma, linewidth=1.5, color='green')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')
    axes[0].set_title('MA(1) Process', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF
    plot_acf(ts_ma, lags=30, ax=axes[1])
    axes[1].set_title('Autocorrelation Function (ACF) - Cuts off at lag 1 for MA(1)', fontsize=14)
    
    # PACF
    plot_pacf(ts_ma, lags=30, ax=axes[2], method='ywm')
    axes[2].set_title('Partial Autocorrelation Function (PACF)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

> **Important** : For MA models, the ACF cuts off at lag q. This is the main way to distinguish them from AR models.

### MA Parameter Estimation
    
    
    # Fit MA(1) model
    train_ma = ts_ma[:int(len(ts_ma) * 0.8)]
    test_ma = ts_ma[int(len(ts_ma) * 0.8):]
    
    # ARIMA(0,0,1) = MA(1)
    model_ma1 = ARIMA(train_ma, order=(0, 0, 1))
    fitted_ma1 = model_ma1.fit()
    
    print("\n=== MA(1) Model Parameters ===")
    print(fitted_ma1.summary())
    
    # Get parameters
    print(f"\nEstimated parameters:")
    print(f"theta: {fitted_ma1.params['ma.L1']:.4f} (true value: 0.7)")
    
    # Forecast
    forecast_ma = fitted_ma1.forecast(steps=len(test_ma))
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train_ma.index, train_ma, label='Training data', linewidth=1.5)
    plt.plot(test_ma.index, test_ma, label='Actual values', linewidth=1.5, color='green')
    plt.plot(test_ma.index, forecast_ma, label='Predictions (MA(1))',
             linewidth=2, linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecasting with MA(1) Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Complete Example: Comparing AR and MA
    
    
    # Fit AR and MA to the same data
    np.random.seed(42)
    n = 200
    # ARMA(1,1) process
    y_arma = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)
    
    phi = 0.5
    theta = 0.3
    
    for t in range(1, n):
        y_arma[t] = phi * y_arma[t-1] + epsilon[t] + theta * epsilon[t-1]
    
    ts_arma = pd.Series(y_arma, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    train_arma = ts_arma[:160]
    test_arma = ts_arma[160:]
    
    # Fit with AR(2)
    model_ar = ARIMA(train_arma, order=(2, 0, 0))
    fitted_ar = model_ar.fit()
    
    # Fit with MA(2)
    model_ma = ARIMA(train_arma, order=(0, 0, 2))
    fitted_ma = model_ma.fit()
    
    # Fit with ARMA(1,1) (correct model)
    model_arma = ARIMA(train_arma, order=(1, 0, 1))
    fitted_arma = model_arma.fit()
    
    # Forecast
    forecast_ar = fitted_ar.forecast(steps=len(test_arma))
    forecast_ma = fitted_ma.forecast(steps=len(test_arma))
    forecast_arma = fitted_arma.forecast(steps=len(test_arma))
    
    # Evaluation
    from sklearn.metrics import mean_squared_error
    
    rmse_ar = np.sqrt(mean_squared_error(test_arma, forecast_ar))
    rmse_ma = np.sqrt(mean_squared_error(test_arma, forecast_ma))
    rmse_arma = np.sqrt(mean_squared_error(test_arma, forecast_arma))
    
    print("=== Model Comparison ===")
    print(f"AR(2) - AIC: {fitted_ar.aic:.2f}, RMSE: {rmse_ar:.4f}")
    print(f"MA(2) - AIC: {fitted_ma.aic:.2f}, RMSE: {rmse_ma:.4f}")
    print(f"ARMA(1,1) - AIC: {fitted_arma.aic:.2f}, RMSE: {rmse_arma:.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(test_arma.index, test_arma, label='Actual values', linewidth=2, color='black')
    plt.plot(test_arma.index, forecast_ar, label='AR(2)', linewidth=1.5, linestyle='--')
    plt.plot(test_arma.index, forecast_ma, label='MA(2)', linewidth=1.5, linestyle='--')
    plt.plot(test_arma.index, forecast_arma, label='ARMA(1,1)', linewidth=2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('AR vs MA vs ARMA Model Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.3 ARIMA Model

### The ARIMA(p,d,q) Framework

The **ARIMA (AutoRegressive Integrated Moving Average) model** is a powerful framework for handling non-stationary time series.

Components of ARIMA(p,d,q):

Parameter | Meaning | Role  
---|---|---  
**p** | AR order | Dependence on past values  
**d** | Differencing order | Removing non-stationarity  
**q** | MA order | Dependence on past errors  
  
ARIMA(p,d,q) model:

$$ \phi(B)(1-B)^d y_t = \theta(B)\varepsilon_t $$

  * $B$: Backshift operator ($By_t = y_{t-1}$)
  * $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$: AR polynomial
  * $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$: MA polynomial

### Model Identification Procedure
    
    
    ```mermaid
    graph TD
        A[Time Series Data] --> B[Check Stationarity]
        B -->|Non-stationary| C[Apply Differencing d=1,2,...]
        B -->|Stationary| D[Analyze ACF/PACF]
        C --> D
        D --> E{Identify Pattern}
        E -->|PACF cuts off| F[AR Model p=?]
        E -->|ACF cuts off| G[MA Model q=?]
        E -->|Both decay| H[ARMA Model p=?, q=?]
        F --> I[Estimate Model]
        G --> I
        H --> I
        I --> J[Diagnostics: Residuals white noise?]
        J -->|No| K[Adjust Parameters]
        J -->|Yes| L[Final Model]
        K --> I
    
        style A fill:#ffebee
        style L fill:#c8e6c9
        style J fill:#fff3e0
    ```

### ARIMA Model Fitting
    
    
    # Generate non-stationary time series (trend + random walk)
    np.random.seed(42)
    n = 300
    trend = np.linspace(0, 10, n)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    ts_nonstationary = pd.Series(
        trend + random_walk,
        index=pd.date_range('2020-01-01', periods=n, freq='D')
    )
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original data
    axes[0].plot(ts_nonstationary, linewidth=1.5)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Non-stationary Time Series (Trend + Random Walk)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # First difference
    ts_diff1 = ts_nonstationary.diff().dropna()
    axes[1].plot(ts_diff1, linewidth=1.5, color='orange')
    axes[1].set_ylabel('1st Difference')
    axes[1].set_title('First Differenced Series (d=1)', fontsize=14)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Second difference
    ts_diff2 = ts_diff1.diff().dropna()
    axes[2].plot(ts_diff2, linewidth=1.5, color='green')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('2nd Difference')
    axes[2].set_title('Second Differenced Series (d=2)', fontsize=14)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stationarity test (ADF test)
    from statsmodels.tsa.stattools import adfuller
    
    def adf_test(series, name=''):
        result = adfuller(series.dropna())
        print(f'\n=== ADF Test: {name} ===')
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print(f'Critical values:')
        for key, value in result[4].items():
            print(f'  {key}: {value:.4f}')
        if result[1] <= 0.05:
            print("→ Stationary (reject null hypothesis at 5% significance level)")
        else:
            print("→ Non-stationary (cannot reject null hypothesis)")
    
    adf_test(ts_nonstationary, 'Original Data')
    adf_test(ts_diff1, '1st Difference')
    adf_test(ts_diff2, '2nd Difference')
    

### ARIMA Model Estimation and Forecasting
    
    
    # Train-test data split
    train_size = int(len(ts_nonstationary) * 0.8)
    train_ns = ts_nonstationary[:train_size]
    test_ns = ts_nonstationary[train_size:]
    
    # ARIMA(1,1,1) model
    model_arima = ARIMA(train_ns, order=(1, 1, 1))
    fitted_arima = model_arima.fit()
    
    print("\n=== ARIMA(1,1,1) Model ===")
    print(fitted_arima.summary())
    
    # Forecast
    forecast_arima = fitted_arima.forecast(steps=len(test_ns))
    forecast_ci = fitted_arima.get_forecast(steps=len(test_ns)).conf_int()
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train_ns.index, train_ns, label='Training data', linewidth=1.5)
    plt.plot(test_ns.index, test_ns, label='Actual values', linewidth=1.5, color='green')
    plt.plot(test_ns.index, forecast_arima, label='Predictions',
             linewidth=2, linestyle='--', color='red')
    plt.fill_between(test_ns.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecasting with ARIMA(1,1,1) Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Accuracy evaluation
    rmse = np.sqrt(mean_squared_error(test_ns, forecast_arima))
    mae = mean_absolute_error(test_ns, forecast_arima)
    
    print(f"\n=== Prediction Accuracy ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    

### Residual Diagnostics
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Residual Diagnostics
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Residual diagnostics
    residuals = fitted_arima.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals time series plot
    axes[0, 0].plot(residuals, linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Time Series Plot', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution (Normality Check)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals ACF
    plot_acf(residuals, lags=30, ax=axes[1, 0])
    axes[1, 0].set_title('Residuals ACF (White Noise Check)', fontsize=14)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test (residual autocorrelation test)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\n=== Ljung-Box Test (Residual Autocorrelation) ===")
    print(lb_test)
    print("\nIf p-value > 0.05, residuals are white noise (good model)")
    

* * *

## 2.4 Seasonal ARIMA (SARIMA)

### What is a SARIMA Model

The **SARIMA (Seasonal ARIMA) model** is a model for handling time series data with seasonality.

SARIMA(p,d,q)(P,D,Q)s notation:

  * **(p,d,q)** : ARIMA parameters for the non-seasonal component
  * **(P,D,Q)** : ARIMA parameters for the seasonal component
  * **s** : Seasonal period (12 for monthly data, 4 for quarterly)

### Seasonal Data Generation and Visualization
    
    
    # Generate time series data with seasonality
    np.random.seed(42)
    n = 240  # 20 years of monthly data
    t = np.arange(n)
    
    # Trend + Seasonality + Noise
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    noise = np.random.normal(0, 2, n)
    
    ts_seasonal = pd.Series(
        trend + seasonal + noise,
        index=pd.date_range('2000-01-01', periods=n, freq='MS')
    )
    
    print("=== Seasonal Time Series Data ===")
    print(f"Number of data points: {len(ts_seasonal)}")
    print(f"Period: 12 months (annual seasonality)")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    axes[0].plot(ts_seasonal, linewidth=1.5)
    axes[0].set_xlabel('Year-Month')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Time Series Data with Seasonality', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Seasonal decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(ts_seasonal, model='additive', period=12)
    
    # Display decomposition results
    fig2 = decomposition.plot()
    fig2.set_size_inches(14, 10)
    plt.tight_layout()
    plt.show()
    

### Seasonal Differencing
    
    
    # Seasonal differencing (s=12)
    ts_seasonal_diff = ts_seasonal.diff(12).dropna()
    
    # Apply regular differencing as well
    ts_seasonal_diff2 = ts_seasonal_diff.diff().dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original data
    axes[0].plot(ts_seasonal, linewidth=1.5)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Original Data (with seasonality)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # After seasonal differencing
    axes[1].plot(ts_seasonal_diff, linewidth=1.5, color='orange')
    axes[1].set_ylabel('Seasonal Difference')
    axes[1].set_title('After Seasonal Differencing (D=1, s=12)', fontsize=14)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal difference + regular difference
    axes[2].plot(ts_seasonal_diff2, linewidth=1.5, color='green')
    axes[2].set_xlabel('Year-Month')
    axes[2].set_ylabel('Difference')
    axes[2].set_title('Seasonal Difference + Regular Difference (D=1, d=1)', fontsize=14)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stationarity test
    adf_test(ts_seasonal, 'Original Data')
    adf_test(ts_seasonal_diff, 'After Seasonal Differencing')
    adf_test(ts_seasonal_diff2, 'Seasonal Difference + Regular Difference')
    

### SARIMA Model Fitting
    
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Train-test data split
    train_seasonal = ts_seasonal[:int(len(ts_seasonal) * 0.8)]
    test_seasonal = ts_seasonal[int(len(ts_seasonal) * 0.8):]
    
    # SARIMA(1,1,1)(1,1,1)12 model
    model_sarima = SARIMAX(
        train_seasonal,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_sarima = model_sarima.fit(disp=False)
    
    print("\n=== SARIMA(1,1,1)(1,1,1)12 Model ===")
    print(fitted_sarima.summary())
    
    # Forecast
    forecast_sarima = fitted_sarima.forecast(steps=len(test_seasonal))
    forecast_sarima_ci = fitted_sarima.get_forecast(steps=len(test_seasonal)).conf_int()
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train_seasonal.index, train_seasonal, label='Training data', linewidth=1.5)
    plt.plot(test_seasonal.index, test_seasonal, label='Actual values', linewidth=1.5, color='green')
    plt.plot(test_seasonal.index, forecast_sarima, label='Predictions (SARIMA)',
             linewidth=2, linestyle='--', color='red')
    plt.fill_between(test_seasonal.index,
                     forecast_sarima_ci.iloc[:, 0],
                     forecast_sarima_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    plt.xlabel('Year-Month')
    plt.ylabel('Value')
    plt.title('Forecasting with SARIMA(1,1,1)(1,1,1)12', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Accuracy evaluation
    rmse_sarima = np.sqrt(mean_squared_error(test_seasonal, forecast_sarima))
    mae_sarima = mean_absolute_error(test_seasonal, forecast_sarima)
    
    print(f"\n=== SARIMA Model Prediction Accuracy ===")
    print(f"RMSE: {rmse_sarima:.4f}")
    print(f"MAE: {mae_sarima:.4f}")
    

### Automatic Parameter Selection with auto_arima
    
    
    # Use auto_arima from pmdarima library
    # pip install pmdarima is required
    
    from pmdarima import auto_arima
    
    print("\n=== Optimal Parameter Search with auto_arima ===")
    print("Searching...")
    
    # auto_arima (with seasonality)
    auto_model = auto_arima(
        train_seasonal,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        m=12,  # Seasonal period
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        seasonal=True,
        d=None,  # Automatic estimation
        D=None,  # Automatic estimation
        trace=True,  # Display search process
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print("\n=== Optimal Model ===")
    print(auto_model.summary())
    
    # Forecast
    forecast_auto = auto_model.predict(n_periods=len(test_seasonal))
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(test_seasonal.index, test_seasonal, label='Actual values', linewidth=2, color='green')
    plt.plot(test_seasonal.index, forecast_sarima, label='Manual SARIMA',
             linewidth=1.5, linestyle='--', color='red')
    plt.plot(test_seasonal.index, forecast_auto, label='auto_arima',
             linewidth=1.5, linestyle='--', color='blue')
    plt.xlabel('Year-Month')
    plt.ylabel('Value')
    plt.title('SARIMA Model Comparison (Manual vs auto_arima)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Accuracy comparison
    rmse_auto = np.sqrt(mean_squared_error(test_seasonal, forecast_auto))
    print(f"\n=== Accuracy Comparison ===")
    print(f"Manual SARIMA RMSE: {rmse_sarima:.4f}")
    print(f"auto_arima RMSE: {rmse_auto:.4f}")
    

* * *

## 2.5 Model Evaluation and Selection

### In-sample vs Out-of-sample Evaluation

Evaluation Method | Description | Use Case  
---|---|---  
**In-sample** | Prediction accuracy on training data | Check model fit  
**Out-of-sample** | Prediction accuracy on test data | Evaluate generalization performance (important)  
  
### Evaluation Metrics

**1\. RMSE (Root Mean Squared Error)**

$$ \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$

**2\. MAE (Mean Absolute Error)**

$$ \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$

**3\. MAPE (Mean Absolute Percentage Error)**

$$ \text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| $$
    
    
    # Calculate multiple evaluation metrics
    def evaluate_forecast(y_true, y_pred, model_name='Model'):
        """Comprehensive forecast accuracy evaluation"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    
        # MAPE (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
        # R^2 score
        r2 = r2_score(y_true, y_pred)
    
        print(f"\n=== {model_name} Evaluation ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R^2: {r2:.4f}")
    
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    
    # Compare multiple models
    results = {}
    results['ARIMA'] = evaluate_forecast(test_ns, forecast_arima, 'ARIMA(1,1,1)')
    results['SARIMA'] = evaluate_forecast(test_seasonal, forecast_sarima, 'SARIMA(1,1,1)(1,1,1)12')
    results['auto_arima'] = evaluate_forecast(test_seasonal, forecast_auto, 'auto_arima')
    
    # Comparison table
    results_df = pd.DataFrame(results).T
    print("\n=== Model Performance Comparison Table ===")
    print(results_df)
    

### Time Series Cross-Validation

For time series data, **time-aware cross-validation** is necessary rather than regular cross-validation.
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    rmse_scores = []
    
    print("\n=== Time Series Cross-Validation ===")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_seasonal), 1):
        # Data split
        cv_train = ts_seasonal.iloc[train_idx]
        cv_test = ts_seasonal.iloc[test_idx]
    
        # Model fitting
        cv_model = SARIMAX(cv_train, order=(1,1,1), seasonal_order=(1,1,1,12))
        cv_fitted = cv_model.fit(disp=False)
    
        # Forecast
        cv_forecast = cv_fitted.forecast(steps=len(cv_test))
    
        # Evaluation
        cv_rmse = np.sqrt(mean_squared_error(cv_test, cv_forecast))
        rmse_scores.append(cv_rmse)
    
        print(f"Fold {fold}: Train={len(cv_train)}, Test={len(cv_test)}, RMSE={cv_rmse:.4f}")
    
    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linewidth=2, markersize=10)
    plt.axhline(y=np.mean(rmse_scores), color='red', linestyle='--',
                label=f'Average RMSE: {np.mean(rmse_scores):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE by Time Series Cross-Validation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Complete Model Comparison Example
    
    
    # Compare multiple ARIMA models
    candidates = [
        ('ARIMA(1,1,0)', (1, 1, 0)),
        ('ARIMA(0,1,1)', (0, 1, 1)),
        ('ARIMA(1,1,1)', (1, 1, 1)),
        ('ARIMA(2,1,1)', (2, 1, 1)),
        ('ARIMA(1,1,2)', (1, 1, 2)),
    ]
    
    comparison_results = []
    
    print("\n=== ARIMA Model Comparison ===")
    for name, order in candidates:
        # Model fitting
        model = ARIMA(train_ns, order=order)
        fitted = model.fit()
    
        # Forecast
        forecast = fitted.forecast(steps=len(test_ns))
    
        # Evaluation
        rmse = np.sqrt(mean_squared_error(test_ns, forecast))
        mae = mean_absolute_error(test_ns, forecast)
        aic = fitted.aic
        bic = fitted.bic
    
        comparison_results.append({
            'Model': name,
            'AIC': aic,
            'BIC': bic,
            'RMSE': rmse,
            'MAE': mae
        })
    
        print(f"{name}: AIC={aic:.2f}, BIC={bic:.2f}, RMSE={rmse:.4f}")
    
    # Results to DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("\n=== Final Comparison Table (sorted by RMSE) ===")
    print(comparison_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIC comparison
    axes[0].barh(comparison_df['Model'], comparison_df['AIC'], color='steelblue')
    axes[0].set_xlabel('AIC')
    axes[0].set_title('Model Comparison: AIC (lower is better)', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # RMSE comparison
    axes[1].barh(comparison_df['Model'], comparison_df['RMSE'], color='coral')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Model Comparison: RMSE (lower is better)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **AR (AutoRegressive) Model**

     * Linear prediction model using past values
     * Identifying order p using PACF
     * Optimal order selection using AIC/BIC
  2. **MA (Moving Average) Model**

     * Model using past prediction errors
     * Identifying order q using ACF
     * Relationship with white noise
  3. **ARIMA Model**

     * Unified framework for handling non-stationary time series
     * Removing trends through differencing
     * Checking model fit through residual diagnostics
  4. **SARIMA Model**

     * Handling time series with seasonality
     * Removing seasonal components through seasonal differencing
     * Automatic parameter selection using auto_arima
  5. **Model Evaluation**

     * Accuracy evaluation using RMSE, MAE, MAPE
     * Time series cross-validation
     * In-sample vs Out-of-sample evaluation

### Model Selection Guidelines

Data Characteristics | Recommended Model | Reason  
---|---|---  
Stationary, short-term memory | AR | Recent values are important  
Stationary, shock-dependent | MA | Past errors influence  
Non-stationary, trend | ARIMA | Make stationary through differencing  
With seasonality | SARIMA | Model seasonal components  
Complex seasonality | auto_arima | Automatic parameter search  
  
### Practical Considerations

Consideration | Description  
---|---  
**Check stationarity** | Always verify stationarity with ADF test before applying  
**Residual diagnostics** | Verify residuals are white noise (Ljung-Box test)  
**Avoid overfitting** | Don't make order too large (select with AIC/BIC)  
**Out-of-sample evaluation** | Check performance on test data, not just training data  
**Present confidence intervals** | Communicate uncertainty, not just point forecasts  
  
### To the Next Chapter

In Chapter 3, we will learn **machine learning-based time series forecasting** , covering feature engineering techniques such as lag features and rolling statistics, forecasting with ensemble methods like Random Forest and XGBoost, LSTM neural networks for sequence modeling, Facebook's Prophet library for automated forecasting, and comparative analysis between machine learning and statistical approaches.

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Explain the differences between AR and MA models, including the patterns of ACF and PACF.

Sample Answer

**Answer** :

**AR Model** :

  * Definition: Predicts current value as a linear combination of past values
  * ACF: Exponential decay
  * PACF: Cuts off at lag p
  * Example: For AR(2), PACF becomes zero after lag 2

**MA Model** :

  * Definition: Expresses current value as a linear combination of past prediction errors
  * ACF: Cuts off at lag q
  * PACF: Exponential decay
  * Example: For MA(1), ACF becomes zero after lag 1

Feature | AR Model | MA Model  
---|---|---  
Depends on | Past values | Past errors  
ACF pattern | Decay | Cuts off at q  
PACF pattern | Cuts off at p | Decay  
Identification method | Order p from PACF | Order q from ACF  
  
### Exercise 2 (Difficulty: medium)

For the following time series data, determine the appropriate ARIMA(p,d,q) order and fit the model.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: For the following time series data, determine the appropriat
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    np.random.seed(123)
    n = 150
    trend = 0.1 * np.arange(n)
    noise = np.random.normal(0, 1, n)
    ts = pd.Series(trend + np.cumsum(noise))
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: For the following time series data, determine the appropriat
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    
    # Generate data
    np.random.seed(123)
    n = 150
    trend = 0.1 * np.arange(n)
    noise = np.random.normal(0, 1, n)
    ts = pd.Series(trend + np.cumsum(noise))
    
    # Step 1: Stationarity test
    print("=== Step 1: Check Stationarity ===")
    result = adfuller(ts)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    if result[1] > 0.05:
        print("→ Non-stationary (differencing needed)")
    
        # First difference
        ts_diff = ts.diff().dropna()
        result_diff = adfuller(ts_diff)
        print(f"\np-value after 1st difference: {result_diff[1]:.4f}")
    
        if result_diff[1] <= 0.05:
            print("→ Made stationary with 1st difference (d=1)")
            d = 1
    else:
        print("→ Stationary (d=0)")
        d = 0
    
    # Step 2: Check ACF/PACF
    print("\n=== Step 2: Estimate Order from ACF/PACF ===")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    if d == 1:
        plot_data = ts.diff().dropna()
    else:
        plot_data = ts
    
    plot_acf(plot_data, lags=20, ax=axes[0])
    axes[0].set_title('ACF (Estimate MA order q)')
    
    plot_pacf(plot_data, lags=20, ax=axes[1], method='ywm')
    axes[1].set_title('PACF (Estimate AR order p)')
    
    plt.tight_layout()
    plt.show()
    
    # Estimate order from PACF and ACF (here, p=1, q=1 as example)
    p = 1
    q = 1
    
    print(f"\nEstimated order: ARIMA({p},{d},{q})")
    
    # Step 3: Model fitting
    print("\n=== Step 3: Model Fitting ===")
    model = ARIMA(ts, order=(p, d, q))
    fitted = model.fit()
    print(fitted.summary())
    
    # Step 4: Residual diagnostics
    print("\n=== Step 4: Residual Diagnostics ===")
    residuals = fitted.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals Time Series Plot')
    
    axes[0, 1].hist(residuals, bins=20, edgecolor='black')
    axes[0, 1].set_title('Residuals Distribution')
    
    plot_acf(residuals, ax=axes[1, 0], lags=20)
    axes[1, 0].set_title('Residuals ACF')
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f"\nLjung-Box test p-value: {lb_test['lb_pvalue'].values[0]:.4f}")
    if lb_test['lb_pvalue'].values[0] > 0.05:
        print("→ Residuals are white noise (good model)")
    else:
        print("→ Residuals have autocorrelation (room for improvement)")
    
    print(f"\n=== Final Model: ARIMA({p},{d},{q}) ===")
    print(f"AIC: {fitted.aic:.2f}")
    print(f"BIC: {fitted.bic:.2f}")
    

### Exercise 3 (Difficulty: medium)

Explain the differences between AIC and BIC, and describe in which situations each should be prioritized.

Sample Answer

**Answer** :

**AIC (Akaike Information Criterion)** :

$$ \text{AIC} = 2k - 2\ln(L) $$

  * Lighter penalty for number of parameters k
  * Emphasizes maximizing prediction accuracy
  * Tends to select slightly more complex models

**BIC (Bayesian Information Criterion)** :

$$ \text{BIC} = k\ln(n) - 2\ln(L) $$

  * Penalty depends on sample size n
  * Emphasizes model simplicity
  * Stronger penalty than AIC when n > 8

**Guidelines for Use** :

Situation | Recommended | Reason  
---|---|---  
Prediction is main goal | AIC | Prioritizes prediction accuracy  
Interpretation is main goal | BIC | Simpler models are easier to interpret  
Small sample size | AIC | BIC penalty is too strong  
Large sample size | BIC | Prevents overfitting  
Estimating true model | BIC | Has consistency property  
  
**Practical Recommendation** :

  * Calculate both and investigate if they differ significantly
  * Ultimately judge based on out-of-sample performance
  * Consider domain knowledge in model selection

### Exercise 4 (Difficulty: hard)

For the following monthly sales data, fit a SARIMA model and forecast the next 12 months. The seasonal period is 12 months.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: For the following monthly sales data, fit a SARIMA model and
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n = 60  # 5 years
    t = np.arange(n)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise
    ts_sales = pd.Series(sales, index=pd.date_range('2019-01-01', periods=n, freq='MS'))
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: For the following monthly sales data, fit a SARIMA model and
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Generate data
    np.random.seed(42)
    n = 60
    t = np.arange(n)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise
    ts_sales = pd.Series(sales, index=pd.date_range('2019-01-01', periods=n, freq='MS'))
    
    print("=== Step 1: Data Examination and Decomposition ===")
    print(f"Number of data points: {len(ts_sales)}")
    print(f"Period: {ts_sales.index[0]} to {ts_sales.index[-1]}")
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(ts_sales, model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    plt.show()
    
    # Step 2: Train-test data split
    train_size = 48  # 4 years for training, 1 year for testing
    train = ts_sales[:train_size]
    test = ts_sales[train_size:]
    
    print(f"\nTraining data: {len(train)} months")
    print(f"Test data: {len(test)} months")
    
    # Step 3: SARIMA model fitting
    # Try SARIMA(1,1,1)(1,1,1)12
    print("\n=== Step 2: SARIMA Model Fitting ===")
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted = model.fit(disp=False)
    print(fitted.summary())
    
    # Step 4: Forecast on test data
    forecast = fitted.forecast(steps=len(test))
    forecast_ci = fitted.get_forecast(steps=len(test)).conf_int()
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"\n=== Prediction Accuracy on Test Data ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label='Training data', linewidth=1.5)
    plt.plot(test.index, test, label='Actual values', linewidth=2, color='green', marker='o')
    plt.plot(test.index, forecast, label='SARIMA predictions',
             linewidth=2, linestyle='--', color='red', marker='s')
    plt.fill_between(test.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    plt.xlabel('Year-Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Forecasting with SARIMA(1,1,1)(1,1,1)12', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Step 5: Forecast next 12 months
    print("\n=== Step 3: Retrain on All Data and Forecast Next 12 Months ===")
    final_model = SARIMAX(
        ts_sales,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    final_fitted = final_model.fit(disp=False)
    future_forecast = final_fitted.forecast(steps=12)
    future_ci = final_fitted.get_forecast(steps=12).conf_int()
    
    # Future date index
    future_dates = pd.date_range(ts_sales.index[-1] + pd.DateOffset(months=1),
                                 periods=12, freq='MS')
    
    print("\nNext 12 months forecast:")
    for date, value, lower, upper in zip(future_dates, future_forecast,
                                          future_ci.iloc[:, 0], future_ci.iloc[:, 1]):
        print(f"{date.strftime('%Y-%m')}: {value:.2f} (95%CI: [{lower:.2f}, {upper:.2f}])")
    
    # Future forecast visualization
    plt.figure(figsize=(14, 6))
    plt.plot(ts_sales.index, ts_sales, label='Historical data', linewidth=2, color='blue')
    plt.plot(future_dates, future_forecast, label='12-month forecast',
             linewidth=2, linestyle='--', color='red', marker='o')
    plt.fill_between(future_dates,
                     future_ci.iloc[:, 0],
                     future_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    plt.axvline(x=ts_sales.index[-1], color='gray', linestyle='--', alpha=0.5, label='Forecast start')
    plt.xlabel('Year-Month')
    plt.ylabel('Sales')
    plt.title('12-Month Sales Forecast', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Residual diagnostics
    print("\n=== Step 4: Residual Diagnostics ===")
    residuals = final_fitted.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals Time Series Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    plot_acf(residuals, ax=axes[1, 0], lags=24)
    axes[1, 0].set_title('Residuals ACF')
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
    print("\nLjung-Box test:")
    print(lb_test)
    

### Exercise 5 (Difficulty: hard)

Explain why time series cross-validation (TimeSeriesSplit) differs from regular cross-validation (KFold), and describe what problems occur when using regular cross-validation on time series data.

Sample Answer

**Answer** :

**Characteristics of Time Series Cross-Validation** :

  1. **Preserve time order** : Training data is always before test data
  2. **Cumulative training** : Training data size gradually increases
  3. **Exclude future data** : Test data is not included in training

**Problems with Regular Cross-Validation (KFold)** :

Problem | Description | Impact  
---|---|---  
**Data leakage** | Training on future data, predicting past | Overestimation of performance  
**Ignore autocorrelation** | Does not consider temporal dependencies | Inappropriate splitting  
**Divergence from reality** | Future is unknown in actual operation | Performance degradation after deployment  
  
**Concrete Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Concrete Example:
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, TimeSeriesSplit
    import matplotlib.pyplot as plt
    
    # Time series data (with trend)
    n = 100
    ts = pd.Series(np.arange(n) + np.random.normal(0, 5, n))
    
    # Regular KFold (problematic)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # TimeSeriesSplit (correct)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # KFold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(ts), 1):
        axes[0].scatter(train_idx, [fold] * len(train_idx),
                        c='blue', marker='|', s=100, label='Train' if fold == 1 else '')
        axes[0].scatter(test_idx, [fold] * len(test_idx),
                        c='red', marker='|', s=100, label='Test' if fold == 1 else '')
    
    axes[0].set_ylabel('Fold')
    axes[0].set_xlabel('Time Index')
    axes[0].set_title('Regular KFold (Problematic: Training on future, predicting past)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TimeSeriesSplit
    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts), 1):
        axes[1].scatter(train_idx, [fold] * len(train_idx),
                        c='blue', marker='|', s=100, label='Train' if fold == 1 else '')
        axes[1].scatter(test_idx, [fold] * len(test_idx),
                        c='red', marker='|', s=100, label='Test' if fold == 1 else '')
    
    axes[1].set_ylabel('Fold')
    axes[1].set_xlabel('Time Index')
    axes[1].set_title('TimeSeriesSplit (Correct: Training on past, predicting future)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Conclusion** :

  * For time series data, always use **TimeSeriesSplit**
  * Preserve time order and prevent future data leakage
  * Can evaluate under the same conditions as actual operation

* * *

## References

  1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.
  2. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts. Available at: <https://otexts.com/fpp3/>
  3. Shumway, R. H., & Stoffer, D. S. (2017). _Time Series Analysis and Its Applications: With R Examples_ (4th ed.). Springer.
  4. Brockwell, P. J., & Davis, R. A. (2016). _Introduction to Time Series and Forecasting_ (3rd ed.). Springer.
  5. Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. _Proceedings of the 9th Python in Science Conference_.
