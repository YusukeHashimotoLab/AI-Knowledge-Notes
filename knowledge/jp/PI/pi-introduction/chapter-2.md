---
title: 第2章：プロセスデータの前処理と可視化
chapter_title: 第2章：プロセスデータの前処理と可視化
subtitle: データ品質を高める実践的前処理手法
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/SCxfASHrfoc"
    title="プロセス・インフォマティクス入門 第2章: プロセスデータの前処理と可視化"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

# 第2章：プロセスデータの前処理と可視化

プロセスデータの前処理は、高品質な分析結果を得るための最重要ステップです。時系列データの扱い方、欠損値・外れ値への対処、データスケーリングの実践手法を習得します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Pandasを使った時系列データの操作（リサンプリング、ローリング統計）
  * ✅ 欠損値の種類（MCAR/MAR/MNAR）と適切な補完手法を選択できる
  * ✅ 外れ値検出の複数手法（Z-score、IQR、Isolation Forest）を実装できる
  * ✅ プロセスデータに適したスケーリング手法を使い分けられる
  * ✅ Matplotlib/Seabornで高度な可視化を作成できる

* * *

## 2.1 時系列データの扱い方

プロセス産業のデータは、時間とともに変化する**時系列データ** が中心です。Pandasの強力な時系列機能を使いこなすことが、PIの基本スキルです。

### Pandas DatetimeIndexの基礎

時系列データを扱うには、まず`DatetimeIndex`を設定します。これにより、時間ベースの操作が簡単になります。

#### コード例1: DatetimeIndexの設定とスライシング
    
    
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
    day1 = df.loc['2025-01-01']
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
    

**出力例 / Output Example** :

```text
Basic DataFrame Information:
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 4320 entries, 2025-01-01 00:00:00 to 2025-01-03 23:59:00
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   temperature  4320 non-null   float64
 1   pressure     4320 non-null   float64
 2   flow_rate    4320 non-null   float64
dtypes: float64(3)
memory usage: 135.0 KB
None

First 5 rows:
                     temperature  pressure  flow_rate
timestamp
2025-01-01 00:00:00    85.745071  1.199006  54.851135
2025-01-01 00:01:00    84.801330  1.194637  47.688726
2025-01-01 00:02:00    85.988986  1.261014  53.430085
2025-01-01 00:03:00    87.310724  1.146166  50.554576
2025-01-01 00:04:00    84.683675  1.161874  50.756585

Data from 2025-01-01 12:00 to 13:00:
                     temperature  pressure  flow_rate
timestamp
2025-01-01 12:00:00    85.461703  1.224955  54.271862
2025-01-01 12:01:00    82.426021  1.115386  54.241309
2025-01-01 12:02:00    82.960269  1.191481  49.912482
2025-01-01 12:03:00    86.088717  1.159359  50.417959
2025-01-01 12:04:00    85.221393  1.241403  53.332733
Number of records: 61

Number of records for 2025-01-01: 1440

Number of records during business hours: 1443

Daily Statistics:
           temperature                       ...  pressure  flow_rate
                  mean       std        min  ...       std       mean           sum
timestamp                                    ...
2025-01-01   85.059766  2.005376  78.664007  ...  0.049050  49.999686  71999.547697
2025-01-02   85.039841  2.029739  79.071002  ...  0.052078  49.904699  71862.766038
2025-01-03   84.943501  2.123821  78.841984  ...  0.052231  50.015225  72021.924583

[3 rows x 8 columns]
```

**解説** : `DatetimeIndex`を使うことで、直感的な時間ベースのスライシングや集計が可能になります。プロセスデータの特定時間帯の抽出や、日別・週別の統計計算が簡単に実行できます。

### リサンプリング（Resampling）

センサーデータは秒単位で記録されることが多いですが、分析には分単位や時間単位のデータで十分な場合があります。**リサンプリング** でデータ粒度を調整できます。

#### コード例2: リサンプリングとダウンサンプリング
    
    
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


**出力例 / Output Example** :

```text
Original data: 17280 records (5-second intervals)
                     temperature
2025-01-01 00:00:00   175.248357
2025-01-01 00:00:05   174.930868
2025-01-01 00:00:10   175.323844
2025-01-01 00:00:15   175.761515
2025-01-01 00:00:20   174.882923

1-minute average: 1440 records
                     temperature
2025-01-01 00:00:00   175.147978
2025-01-01 00:01:00   174.704384
2025-01-01 00:02:00   174.903332
2025-01-01 00:03:00   174.833711
2025-01-01 00:04:00   175.023958

5-minute aggregation: 288 records
                    temperature
                           mean       std         min         max
2025-01-01 00:00:00  174.922673  0.454261  174.020165  175.926139
2025-01-01 00:05:00  174.998158  0.471682  173.690127  176.231621
2025-01-01 00:10:00  175.045957  0.498052  174.196258  176.360085
2025-01-01 00:15:00  175.028407  0.518181  173.987429  176.926366
2025-01-01 00:20:00  174.990933  0.522220  173.379366  176.066517

Hourly aggregation: 24 records
                    temperature
                           mean       std count
2025-01-01 00:00:00  174.994950  0.493165   720
2025-01-01 01:00:00  175.044894  0.493246   720
2025-01-01 02:00:00  175.013703  0.492465   720
2025-01-01 03:00:00  175.012857  0.485290   720
2025-01-01 04:00:00  175.000094  0.530220   720
2025-01-01 05:00:00  174.962240  0.495909   720
2025-01-01 06:00:00  174.994513  0.498281   720
2025-01-01 07:00:00  174.977928  0.482791   720
2025-01-01 08:00:00  174.975696  0.514387   720
2025-01-01 09:00:00  174.990110  0.527364   720


Data Size Comparison:
Original data (5s): 17,280 records
1-minute average: 1,440 records (8.3%)
5-minute average: 288 records (1.67%)
```

**解説** : リサンプリングにより、データ量を削減しつつノイズを低減できます。分析の目的に応じて適切な時間粒度を選択することで、計算効率と情報量のバランスを最適化できます。

### ローリング統計量（Rolling Statistics）

移動平均や移動標準偏差などの**ローリング統計** は、トレンド把握やノイズ除去に有効です。

#### コード例3: ローリング統計とトレンド分析
    
    
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
    

**出力例 / Output Example** :

```text
Rolling Statistics Summary:
       rolling_mean_60  rolling_std_60    deviation
count      1381.000000     1381.000000  1381.000000
mean        175.079631        1.975941     0.223317
std           2.834735        0.163323     1.964661
min         169.740273        1.383536    -6.407692
25%         172.708022        1.869136    -1.105379
50%         174.814646        1.985401     0.235690
75%         177.576527        2.085680     1.462422
max         180.329194        2.384197     7.566088

Anomaly data points: 62 records (4.31%)
                     temperature  rolling_mean_10  ...  rolling_max  rolling_min
2025-01-01 01:14:00   165.274756       171.045531  ...   173.919984   165.274756
2025-01-01 01:53:00   175.711752       171.366960  ...   175.711752   165.274756
2025-01-01 02:05:00   175.249570       171.507640  ...   175.711752   165.274756
2025-01-01 02:59:00   176.684258       171.929577  ...   176.684258   167.771830
2025-01-01 03:29:00   179.157860       172.648432  ...   179.157860   168.283718

[5 rows x 7 columns]
```

**解説** : ローリング統計は、プロセスのトレンド把握、安定性監視、異常検知の基礎となります。窓サイズ（window）は、プロセスの時定数に応じて調整します。

* * *

## 2.2 欠損値処理・外れ値検出

実際のプロセスデータには、センサー故障や通信エラーによる**欠損値** 、異常な測定値による**外れ値** が含まれます。適切な処理が必須です。

### 欠損値の種類と対処法

欠損値は3つのタイプに分類されます：

タイプ | 説明 | 例 | 推奨される対処法  
---|---|---|---  
**MCAR**  
(Missing Completely At Random) | 欠損が完全にランダム | センサーの一時的な通信エラー | 線形補間、移動平均補完  
**MAR**  
(Missing At Random) | 他の変数に依存して欠損 | 高温時にセンサーが故障しやすい | 回帰補完、K近傍法補完  
**MNAR**  
(Missing Not At Random) | 欠損自体が情報を持つ | 測定範囲外の値は記録されない | 慎重な分析、削除を検討  
  
#### コード例4: 欠損値の検出と複数の補完手法
    
    
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
    

**出力例 / Output Example** :

```text
Missing Value Check:
temperature    50
pressure        7
flow_rate       0
dtype: int64

Missing Rate:
temperature    10.0
pressure        1.4
flow_rate       0.0
dtype: float64

Missing Check After Imputation:
Linear interpolation: 0
Spline interpolation: 0
KNN imputation: 0
```

**解説** : プロセスデータでは、時系列性を考慮した線形補間やスプライン補間が有効です。多変量の相関が強い場合はKNN補完も優れた選択肢です。連続した長時間の欠損は、補完ではなく削除を検討します。

### 外れ値検出の実践手法

外れ値は、測定エラー、センサー故障、実際の異常状態など、様々な原因で発生します。適切な検出手法の選択が重要です。

#### コード例5: 統計的手法による外れ値検出（Z-score、IQR）
    
    
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
    axes[1, 1].boxplot(box_data, patch_artist=True,
                       boxprops=dict(facecolor='#11998e', alpha=0.7))
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_xticks([1, 2], ['Normal', 'Outliers'])
    axes[1, 1].set_title('Box Plot: Normal vs Outliers (IQR method)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Outlier statistics
    print("\nOutlier Statistics:")
    print(df[df['outlier_iqr']]['temperature'].describe())
    

**解説** : Z-score法は正規分布を仮定するため、外れ値に影響されやすい弱点があります。IQR法はロバストで、外れ値の影響を受けにくい特徴があります。プロセスデータでは、両方を併用して判断することを推奨します。

#### コード例6: 機械学習による外れ値検出（Isolation Forest）
    
    
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
    

**出力例 / Output Example** :

```text
Isolation Forest Detection Results:
Detected outliers: 50 records (5.00%)

Top 5 Most Anomalous Data Points:
     temperature  pressure  anomaly_score
977   170.152241  0.998899      -0.774414
990   190.416766  1.844188      -0.774336
992   189.505645  1.854502      -0.768594
995   189.437799  1.847886      -0.766517
975   171.220919  0.942783      -0.766424

Normal Data Statistics:
       temperature    pressure
count   950.000000  950.000000
mean    175.092547    1.505337
std       1.877916    0.049534
min     170.396158    1.362307
25%     173.793169    1.473577
50%     175.093417    1.503446
75%     176.307244    1.538584
max     181.157762    1.662240

Outlier Statistics:
       temperature   pressure
count    50.000000  50.000000
mean    180.214889   1.403165
std      10.030184   0.268729
min     168.517465   0.839360
25%     171.900104   1.344497
50%     174.942445   1.413490
75%     189.488684   1.568950
max     197.887529   1.854502
```

**解説** : Isolation Forestは、多変量データの外れ値検出に優れています。単変量の統計的手法では検出できない、変数間の関係性の異常を捉えることができます。プロセスデータの異常検知に非常に有効です。

* * *

## 2.3 データのスケーリングと正規化

機械学習モデルを構築する前に、変数のスケールを揃える**スケーリング** が重要です。プロセスデータでは、温度（0-200°C）と圧力（0-3 MPa）のように、変数のスケールが大きく異なります。

### 主要なスケーリング手法

手法 | 変換式 | 特徴 | 適用場面  
---|---|---|---  
**Min-Max  
Scaling** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | [0, 1]の範囲に変換 | 外れ値が少ない場合  
**Standard  
Scaling** | $x' = \frac{x - \mu}{\sigma}$ | 平均0、標準偏差1 | 正規分布に近い場合  
**Robust  
Scaling** | $x' = \frac{x - Q_{med}}{Q_{75} - Q_{25}}$ | 外れ値に頑健 | 外れ値が多い場合  
  
#### コード例7: スケーリング手法の比較と選択
    
    
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


**出力例 / Output Example** :

```text
Original Data Statistics:
       temperature    pressure   flow_rate
count   500.000000  500.000000  500.000000
mean    176.134190    1.521365   51.084845
std       8.605454    0.235102   10.102463
min     158.793663    0.960623   21.037446
25%     171.672148    1.384816   43.975704
50%     175.311276    1.509661   51.198058
75%     178.430518    1.637094   57.547384
max     233.435708    2.950565   76.016831

Post-Scaling Statistics (Temperature):
       temperature_minmax  temperature_standard  temperature_robust
count          500.000000          5.000000e+02          500.000000
mean             0.232316         -2.685852e-15            0.121762
std              0.115290          1.001002e+00            1.273303
min              0.000000         -2.017080e+00           -2.444023
25%              0.172537         -5.190325e-01           -0.538462
50%              0.221291         -9.572285e-02            0.000000
75%              0.263080          2.671129e-01            0.461538
max              1.000000          6.665413e+00            8.600363

Scale Values for Data Points with Outliers (480-495):
     temperature  temperature_minmax  temperature_standard  temperature_robust
480   224.360412            0.878416              5.609759            7.257539
481   220.222298            0.822976              5.128406            6.645245
482   216.967768            0.779374              4.749833            6.163689
483   226.017318            0.900614              5.802493            7.502703
484   221.218246            0.836319              5.244257            6.792610

【Recommendations】
✓ Min-Max: When outliers are few and [0,1] range is needed (e.g., neural networks)
✓ Standard: When distribution is close to normal and outliers are few (e.g., linear regression, SVM)
✓ Robust: When outliers are many and robust preprocessing is needed (real process data)
```

**解説** : プロセスデータには外れ値が含まれることが多いため、Robust Scalingが最も安全な選択肢です。ただし、モデルの種類や目的に応じて適切な手法を選びます。

#### コード例8: スケーリングの実践ワークフロー
    
    
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
    

**解説** : スケーリングの実践で最も重要なのは、**訓練データでfitし、テストデータではtransformのみ** を使うことです。これにより、データリークを防ぎ、実運用時の性能を正確に評価できます。

* * *

## 2.4 Pandas/Matplotlib/Seabornによる高度な可視化

データの本質を理解するには、適切な可視化が不可欠です。プロセスデータ特有の可視化テクニックを習得しましょう。

#### コード例9: プロセス運転状態の多次元可視化
    
    
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
    

**解説** : 複数の可視化手法を組み合わせることで、データの異なる側面を同時に把握できます。運転フェーズごとの色分け、2軸グラフ、ヒートマップなどは、プロセスエンジニアとのコミュニケーションに非常に有効です。

#### コード例10: インタラクティブダッシュボード（Plotly）
    
    
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
    

**出力例 / Output Example** :

```text
Interactive dashboard saved to 'process_monitoring_dashboard.html'.

【Dashboard Features】
✓ Zoom: Drag to zoom in, double-click to reset
✓ Pan: Shift + drag to move
✓ Hover: Mouse over data points for details
✓ Time range selection: Buttons at top for 1h/6h/12h/all periods
✓ Saved as HTML file, viewable in browser

Temperature anomalies: 164 records
Occurrence time: 2025-01-01 00:56:30 - 2025-01-01 13:27:30

Purity anomalies: 91 records
Occurrence time: 2025-01-01 00:03:00 - 2025-01-01 23:43:00
```

**解説** : Plotlyによるインタラクティブダッシュボードは、プロセスエンジニアや管理者への報告に最適です。HTMLファイルとして保存でき、ブラウザで誰でも閲覧できる点が実用的です。

* * *

## 2.5 本章のまとめ

### 学んだこと

  1. **時系列データの操作**
     * DatetimeIndexによる直感的な時間ベーススライシング
     * リサンプリングでデータ粒度を調整し、計算効率を向上
     * ローリング統計でトレンド把握とノイズ除去
  2. **欠損値処理**
     * MCAR/MAR/MNARの3タイプと適切な対処法
     * 線形補間、スプライン補間、KNN補完の使い分け
     * 時系列データには時間を考慮した補完が有効
  3. **外れ値検出**
     * 統計的手法（Z-score、IQR）とその限界
     * Isolation Forestによる多変量外れ値検出
     * プロセスデータでは複数手法の併用が推奨
  4. **スケーリング**
     * Min-Max、Standard、Robustの3手法
     * 外れ値が多い実プロセスデータにはRobustが適切
     * 訓練データでfit、テストデータでtransformの原則
  5. **高度な可視化**
     * フェーズ別色分け、2軸グラフ、ヒートマップ
     * Plotlyによるインタラクティブダッシュボード
     * 複数の視点からデータを理解する重要性

### 重要なポイント

> **"Garbage In, Garbage Out"** : データ前処理の品質が、モデルの性能を決定します。

  * プロセスデータには必ず欠損値・外れ値が含まれる
  * 適切な前処理なしにモデル構築を始めてはいけない
  * 可視化は前処理の効果を確認する最良の手段
  * 実務では、前処理に全体の60-70%の時間を費やす

### 実践のヒント

  1. **探索的データ分析（EDA）を怠らない** : いきなりモデル構築せず、まずデータを理解する
  2. **前処理は可逆的に** : 元データを保持し、前処理の各ステップを記録
  3. **可視化で検証** : 前処理の各ステップで可視化し、期待通りの結果か確認
  4. **ドメイン知識を活用** : プロセスエンジニアと協力し、異常値の意味を理解

### 次の章へ

第3章では、前処理したデータを使って**プロセスモデリングの基礎** を学びます：

  * 線形回帰によるプロセスモデル構築
  * 多変量回帰とPLS（偏最小二乗法）
  * ソフトセンサーの概念と実装
  * モデル評価指標（R², RMSE, MAE）
  * 非線形モデルへの拡張（Random Forest、SVR）
