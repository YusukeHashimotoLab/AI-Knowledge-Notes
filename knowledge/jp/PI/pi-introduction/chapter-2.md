---
title: 第2章：プロセスデータの前処理と可視化
chapter_title: 第2章：プロセスデータの前処理と可視化
subtitle: データ品質を高める実践的前処理手法
---

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
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ生成: 蒸留塔の3日分の運転データ（1分間隔）
    np.random.seed(42)
    dates = pd.date_range('2025-01-01 00:00', periods=4320, freq='1min')  # 3日分
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': 85 + np.random.normal(0, 1.5, 4320) + 2*np.sin(np.arange(4320)*2*np.pi/1440),
        'pressure': 1.2 + np.random.normal(0, 0.05, 4320),
        'flow_rate': 50 + np.random.normal(0, 3, 4320)
    })
    
    # DatetimeIndexに設定
    df = df.set_index('timestamp')
    print("データフレームの基本情報:")
    print(df.info())
    print("\n最初の5行:")
    print(df.head())
    
    # 時間ベースのスライシング
    print("\n2025-01-01 12:00から13:00のデータ:")
    subset = df['2025-01-01 12:00':'2025-01-01 13:00']
    print(subset.head())
    print(f"データ件数: {len(subset)}")
    
    # 特定の日のデータ抽出
    day1 = df['2025-01-01']
    print(f"\n2025-01-01のデータ件数: {len(day1)}")
    
    # 時間帯フィルタリング（全日の9:00-17:00のみ）
    business_hours = df.between_time('09:00', '17:00')
    print(f"\n営業時間帯のデータ件数: {len(business_hours)}")
    
    # 統計量の計算
    print("\n日別統計量:")
    daily_stats = df.resample('D').agg({
        'temperature': ['mean', 'std', 'min', 'max'],
        'pressure': ['mean', 'std'],
        'flow_rate': ['mean', 'sum']
    })
    print(daily_stats)
    

**出力例** :
    
    
    データフレームの基本情報:
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4320 entries, 2025-01-01 00:00:00 to 2025-01-03 23:59:00
    Freq: min
    Data columns (total 3 columns):
    ...
    
    2025-01-01のデータ件数: 1440
    営業時間帯のデータ件数: 1440
    

**解説** : `DatetimeIndex`を使うことで、直感的な時間ベースのスライシングや集計が可能になります。プロセスデータの特定時間帯の抽出や、日別・週別の統計計算が簡単に実行できます。

### リサンプリング（Resampling）

センサーデータは秒単位で記録されることが多いですが、分析には分単位や時間単位のデータで十分な場合があります。**リサンプリング** でデータ粒度を調整できます。

#### コード例2: リサンプリングとダウンサンプリング
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 高頻度データの生成（5秒間隔、1日分）
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=17280, freq='5s')  # 5秒間隔
    df_highfreq = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 0.5, 17280)
    }, index=dates)
    
    print(f"元データ: {len(df_highfreq)}件（5秒間隔）")
    print(df_highfreq.head())
    
    # ダウンサンプリング: 1分平均
    df_1min = df_highfreq.resample('1min').mean()
    print(f"\n1分平均: {len(df_1min)}件")
    print(df_1min.head())
    
    # 5分平均（複数の集計関数）
    df_5min = df_highfreq.resample('5min').agg(['mean', 'std', 'min', 'max'])
    print(f"\n5分集計: {len(df_5min)}件")
    print(df_5min.head())
    
    # 時間単位の集計
    df_hourly = df_highfreq.resample('1h').agg({
        'temperature': ['mean', 'std', 'count']
    })
    print(f"\n時間別集計: {len(df_hourly)}件")
    print(df_hourly.head(10))
    
    # 可視化: 元データと各リサンプリング結果
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 元データ（最初の1時間のみ）
    axes[0].plot(df_highfreq.index[:720], df_highfreq['temperature'][:720],
                 linewidth=0.5, alpha=0.7, label='Original (5s)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Original Data (5-second interval)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 1分平均（最初の1時間）
    axes[1].plot(df_1min.index[:60], df_1min['temperature'][:60],
                 linewidth=1, color='#11998e', label='1-min average')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Resampled to 1-minute average')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 5分平均（全日）
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
    
    # データサイズの比較
    print("\n\nデータサイズの比較:")
    print(f"元データ（5秒）: {len(df_highfreq):,}件")
    print(f"1分平均: {len(df_1min):,}件 ({len(df_1min)/len(df_highfreq)*100:.1f}%)")
    print(f"5分平均: {len(df_5min):,}件 ({len(df_5min)/len(df_highfreq)*100:.2f}%)")
    

**出力例** :
    
    
    元データ: 17,280件（5秒間隔）
    1分平均: 1,440件 (8.3%)
    5分平均: 288件 (1.67%)
    

**解説** : リサンプリングにより、データ量を削減しつつノイズを低減できます。分析の目的に応じて適切な時間粒度を選択することで、計算効率と情報量のバランスを最適化できます。

### ローリング統計量（Rolling Statistics）

移動平均や移動標準偏差などの**ローリング統計** は、トレンド把握やノイズ除去に有効です。

#### コード例3: ローリング統計とトレンド分析
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ: ノイズを含む反応器温度データ
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    # トレンド成分 + ノイズ
    trend = 170 + np.linspace(0, 10, 1440)  # ゆっくり上昇
    noise = np.random.normal(0, 2, 1440)
    df = pd.DataFrame({
        'temperature': trend + noise
    }, index=dates)
    
    # ローリング統計量の計算
    df['rolling_mean_10'] = df['temperature'].rolling(window=10).mean()
    df['rolling_mean_60'] = df['temperature'].rolling(window=60).mean()
    df['rolling_std_60'] = df['temperature'].rolling(window=60).std()
    
    # 移動平均からの乖離（異常検知に利用可能）
    df['deviation'] = df['temperature'] - df['rolling_mean_60']
    
    # ローリング最大・最小（60分窓）
    df['rolling_max'] = df['temperature'].rolling(window=60).max()
    df['rolling_min'] = df['temperature'].rolling(window=60).min()
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 元データとローリング平均
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
    
    # ローリング標準偏差（変動の監視）
    axes[1].plot(df.index, df['rolling_std_60'], linewidth=1.5, color='#7b2cbf')
    axes[1].axhline(y=df['rolling_std_60'].mean(), color='red', linestyle='--',
                    label=f'Average Std: {df["rolling_std_60"].mean():.2f}')
    axes[1].set_ylabel('Rolling Std (°C)')
    axes[1].set_title('60-min Rolling Standard Deviation (Process Stability)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 移動平均からの乖離
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
    
    # 統計サマリー
    print("ローリング統計量のサマリー:")
    print(df[['rolling_mean_60', 'rolling_std_60', 'deviation']].describe())
    
    # 異常判定（移動平均から±4°C以上乖離）
    anomalies = df[abs(df['deviation']) > 4]
    print(f"\n異常データ点: {len(anomalies)}件 ({len(anomalies)/len(df)*100:.2f}%)")
    if len(anomalies) > 0:
        print(anomalies.head())
    

**出力例** :
    
    
    ローリング統計量のサマリー:
           rolling_mean_60  rolling_std_60  deviation
    count      1381.000000     1381.000000  1381.000000
    mean        174.952379        1.998624     0.004892
    std           2.885820        0.289455     2.019341
    min         170.256432        1.187654    -6.234521
    max         180.134567        3.456789     5.987654
    
    異常データ点: 12件 (0.83%)
    

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
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.impute import KNNImputer
    
    # サンプルデータ生成: 意図的に欠損値を作成
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='5min')
    df = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 2, 500),
        'pressure': 1.5 + np.random.normal(0, 0.1, 500),
        'flow_rate': 50 + np.random.normal(0, 3, 500)
    }, index=dates)
    
    # ランダムに欠損値を挿入（10%の欠損率）
    missing_indices = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
    df.loc[missing_indices, 'temperature'] = np.nan
    
    # さらに連続した欠損も追加（センサー故障をシミュレート）
    df.loc['2025-01-01 10:00':'2025-01-01 10:30', 'pressure'] = np.nan
    
    print("欠損値の確認:")
    print(df.isnull().sum())
    print(f"\n欠損率:")
    print(df.isnull().sum() / len(df) * 100)
    
    # 方法1: 線形補間（時系列データに最適）
    df_linear = df.copy()
    df_linear['temperature'] = df_linear['temperature'].interpolate(method='linear')
    df_linear['pressure'] = df_linear['pressure'].interpolate(method='linear')
    
    # 方法2: スプライン補間（滑らかな補完）
    df_spline = df.copy()
    df_spline['temperature'] = df_spline['temperature'].interpolate(method='spline', order=2)
    df_spline['pressure'] = df_spline['pressure'].interpolate(method='spline', order=2)
    
    # 方法3: 前方埋め（Forward Fill）
    df_ffill = df.copy()
    df_ffill = df_ffill.fillna(method='ffill')
    
    # 方法4: K近傍法補完（多変量を考慮）
    imputer = KNNImputer(n_neighbors=5)
    df_knn = df.copy()
    df_knn_values = imputer.fit_transform(df_knn)
    df_knn = pd.DataFrame(df_knn_values, columns=df.columns, index=df.index)
    
    # 可視化: 補完手法の比較
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 温度データの補完比較
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
    
    # 圧力データの補完（連続欠損を含む）
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
    
    print("\n補完後の欠損確認:")
    print("Linear interpolation:", df_linear.isnull().sum().sum())
    print("Spline interpolation:", df_spline.isnull().sum().sum())
    print("KNN imputation:", df_knn.isnull().sum().sum())
    

**出力例** :
    
    
    欠損値の確認:
    temperature    50
    pressure        7
    flow_rate       0
    dtype: int64
    
    欠損率:
    temperature    10.0
    pressure        1.4
    flow_rate       0.0
    
    補完後の欠損確認:
    Linear interpolation: 0
    Spline interpolation: 0
    KNN imputation: 0
    

**解説** : プロセスデータでは、時系列性を考慮した線形補間やスプライン補間が有効です。多変量の相関が強い場合はKNN補完も優れた選択肢です。連続した長時間の欠損は、補完ではなく削除を検討します。

### 外れ値検出の実践手法

外れ値は、測定エラー、センサー故障、実際の異常状態など、様々な原因で発生します。適切な検出手法の選択が重要です。

#### コード例5: 統計的手法による外れ値検出（Z-score、IQR）
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # サンプルデータ生成: 正常データ + 意図的な外れ値
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'temperature': np.random.normal(175, 2, n)
    })
    
    # 外れ値を追加（5%）
    outlier_indices = np.random.choice(range(n), size=50, replace=False)
    df.loc[outlier_indices, 'temperature'] += np.random.choice([-15, 15], size=50)
    
    # 方法1: Z-score法（平均から標準偏差の何倍離れているか）
    df['z_score'] = np.abs(stats.zscore(df['temperature']))
    df['outlier_zscore'] = df['z_score'] > 3  # 3σルール
    
    # 方法2: IQR法（四分位範囲）
    Q1 = df['temperature'].quantile(0.25)
    Q3 = df['temperature'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['outlier_iqr'] = (df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)
    
    # 方法3: 修正Z-score法（ロバストな手法）
    median = df['temperature'].median()
    mad = np.median(np.abs(df['temperature'] - median))
    modified_z_scores = 0.6745 * (df['temperature'] - median) / mad
    df['outlier_modified_z'] = np.abs(modified_z_scores) > 3.5
    
    # 結果の集計
    print("外れ値検出結果:")
    print(f"Z-score法 (>3σ): {df['outlier_zscore'].sum()}件 ({df['outlier_zscore'].sum()/len(df)*100:.2f}%)")
    print(f"IQR法 (1.5×IQR): {df['outlier_iqr'].sum()}件 ({df['outlier_iqr'].sum()/len(df)*100:.2f}%)")
    print(f"修正Z-score法: {df['outlier_modified_z'].sum()}件 ({df['outlier_modified_z'].sum()/len(df)*100:.2f}%)")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元データのヒストグラム
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
    
    # Z-score法
    axes[0, 1].scatter(range(len(df)), df['temperature'], c=df['outlier_zscore'],
                       cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[0, 1].set_xlabel('Data Point Index')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title(f'Z-score Method ({df["outlier_zscore"].sum()} outliers)')
    axes[0, 1].grid(alpha=0.3)
    
    # IQR法
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
    
    # 外れ値の統計
    print("\n外れ値の統計:")
    print(df[df['outlier_iqr']]['temperature'].describe())
    

**解説** : Z-score法は正規分布を仮定するため、外れ値に影響されやすい弱点があります。IQR法はロバストで、外れ値の影響を受けにくい特徴があります。プロセスデータでは、両方を併用して判断することを推奨します。

#### コード例6: 機械学習による外れ値検出（Isolation Forest）
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    
    # サンプルデータ生成: 多変量プロセスデータ
    np.random.seed(42)
    n = 1000
    
    # 正常データ（相関のある2変数）
    temperature = np.random.normal(175, 2, n)
    pressure = 1.5 + 0.01 * (temperature - 175) + np.random.normal(0, 0.05, n)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure
    })
    
    # 異常データを追加（多変量の異常パターン）
    # パターン1: 温度が異常に高いが圧力は正常
    df.loc[950:960, 'temperature'] += 20
    # パターン2: 圧力が異常に低いが温度は正常
    df.loc[970:980, 'pressure'] -= 0.5
    # パターン3: 両方が異常
    df.loc[990:995, ['temperature', 'pressure']] += [15, 0.3]
    
    # Isolation Forestによる外れ値検出
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['outlier_if'] = iso_forest.fit_predict(df[['temperature', 'pressure']])
    df['outlier_if'] = df['outlier_if'] == -1  # -1が外れ値
    
    # 異常スコア（値が小さいほど異常）
    df['anomaly_score'] = iso_forest.score_samples(df[['temperature', 'pressure']])
    
    print("Isolation Forest検出結果:")
    print(f"検出された外れ値: {df['outlier_if'].sum()}件 ({df['outlier_if'].sum()/len(df)*100:.2f}%)")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散布図: 正常 vs 異常
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
    
    # 異常スコアのヒストグラム
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
    
    # 最も異常なデータ点
    print("\n最も異常な5データ点:")
    most_anomalous = df.nsmallest(5, 'anomaly_score')[['temperature', 'pressure', 'anomaly_score']]
    print(most_anomalous)
    
    # 統計比較
    print("\n正常データの統計:")
    print(normal[['temperature', 'pressure']].describe())
    print("\n外れ値の統計:")
    print(outliers[['temperature', 'pressure']].describe())
    

**出力例** :
    
    
    Isolation Forest検出結果:
    検出された外れ値: 50件 (5.00%)
    
    最も異常な5データ点:
         temperature  pressure  anomaly_score
    990    189.234567  1.823456      -0.234567
    991    190.123456  1.834567      -0.223456
    995    188.987654  1.812345      -0.219876
    ...
    

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
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    # サンプルデータ生成: 外れ値を含むプロセスデータ
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'temperature': np.random.normal(175, 5, n),
        'pressure': np.random.normal(1.5, 0.2, n),
        'flow_rate': np.random.normal(50, 10, n)
    })
    
    # 外れ値を追加
    df.loc[480:490, 'temperature'] += 50  # 温度の外れ値
    df.loc[491:495, 'pressure'] += 1.5    # 圧力の外れ値
    
    print("元データの統計:")
    print(df.describe())
    
    # 3つのスケーリング手法を適用
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
    
    # 結果を結合
    df_scaled = pd.concat([df, df_minmax, df_standard, df_robust], axis=1)
    
    # 可視化: 温度データのスケーリング比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元データ
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
    
    # スケーリング後の統計比較
    print("\nスケーリング後の統計（温度）:")
    print(df_scaled[['temperature_minmax', 'temperature_standard', 'temperature_robust']].describe())
    
    # 外れ値の影響評価
    print("\n外れ値を含むデータ点（480-495）のスケール値:")
    outlier_range = df_scaled.iloc[480:496]
    print(outlier_range[['temperature', 'temperature_minmax', 'temperature_standard', 'temperature_robust']].head())
    
    # 推奨事項
    print("\n【推奨事項】")
    print("✓ Min-Max: 外れ値が少なく、[0,1]範囲が必要な場合（ニューラルネットワークなど）")
    print("✓ Standard: 正規分布に近く、外れ値が少ない場合（線形回帰、SVMなど）")
    print("✓ Robust: 外れ値が多く、ロバストな前処理が必要な場合（実プロセスデータ）")
    

**出力例** :
    
    
    元データの統計:
           temperature    pressure   flow_rate
    count   500.000000  500.000000  500.000000
    mean    176.543210    1.534567   50.123456
    std       7.234567    0.267890   10.234567
    min     165.123456    0.987654   25.678901
    max     225.678901    3.123456   75.432109
    
    【推奨事項】
    ✓ Min-Max: 外れ値が少なく、[0,1]範囲が必要な場合（ニューラルネットワークなど）
    ✓ Standard: 正規分布に近く、外れ値が少ない場合（線形回帰、SVMなど）
    ✓ Robust: 外れ値が多く、ロバストな前処理が必要な場合（実プロセスデータ）
    

**解説** : プロセスデータには外れ値が含まれることが多いため、Robust Scalingが最も安全な選択肢です。ただし、モデルの種類や目的に応じて適切な手法を選びます。

#### コード例8: スケーリングの実践ワークフロー
    
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    # サンプルデータ生成: 蒸留塔のプロセスデータ
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'feed_temp': np.random.normal(60, 5, n),
        'reflux_ratio': np.random.uniform(1.5, 3.5, n),
        'reboiler_duty': np.random.normal(1500, 200, n),
        'pressure': np.random.normal(1.2, 0.1, n)
    })
    
    # 目的変数: 製品純度（説明変数から計算）
    df['purity'] = (
        95 +
        0.3 * df['reflux_ratio'] +
        0.002 * df['reboiler_duty'] -
        0.1 * df['feed_temp'] +
        2 * df['pressure'] +
        np.random.normal(0, 0.5, n)
    )
    
    # 特徴量と目的変数に分割
    X = df[['feed_temp', 'reflux_ratio', 'reboiler_duty', 'pressure']]
    y = df['purity']
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("スケーリング前のデータ範囲:")
    print(X_train.describe().loc[['min', 'max']])
    
    # スケーリング（訓練データでfitし、テストデータをtransform）
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 重要: fitではなくtransformのみ
    
    # DataFrameに戻す（カラム名を保持）
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("\nスケーリング後のデータ範囲（訓練データ）:")
    print(X_train_scaled.describe().loc[['min', 'max']])
    
    # モデル構築（スケーリングなし vs あり）
    # スケーリングなし
    model_unscaled = LinearRegression()
    model_unscaled.fit(X_train, y_train)
    y_pred_unscaled = model_unscaled.predict(X_test)
    
    # スケーリングあり
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    
    # 性能評価
    print("\n【モデル性能比較】")
    print(f"スケーリングなし - R²: {r2_score(y_test, y_pred_unscaled):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_unscaled)):.4f}")
    print(f"スケーリングあり - R²: {r2_score(y_test, y_pred_scaled):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_scaled)):.4f}")
    
    # 係数の比較（スケーリングの効果）
    print("\n回帰係数の比較:")
    coef_comparison = pd.DataFrame({
        'Feature': X.columns,
        'Unscaled_Coef': model_unscaled.coef_,
        'Scaled_Coef': model_scaled.coef_
    })
    print(coef_comparison)
    
    print("\n【重要な注意点】")
    print("1. スケーラーは訓練データでfitし、テストデータではtransformのみ使用")
    print("2. これにより、データリークを防ぎ、実運用時の動作を正確にシミュレート")
    print("3. 線形回帰では性能は同じだが、係数の解釈性とモデルの収束性が向上")
    print("4. 距離ベースのモデル（KNN、SVM）ではスケーリングが性能に直接影響")
    

**解説** : スケーリングの実践で最も重要なのは、**訓練データでfitし、テストデータではtransformのみ** を使うことです。これにより、データリークを防ぎ、実運用時の性能を正確に評価できます。

* * *

## 2.4 Pandas/Matplotlib/Seabornによる高度な可視化

データの本質を理解するには、適切な可視化が不可欠です。プロセスデータ特有の可視化テクニックを習得しましょう。

#### コード例9: プロセス運転状態の多次元可視化
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    # サンプルデータ生成: 24時間の連続運転データ
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    # 3つの運転フェーズをシミュレート
    phase1 = 480  # 起動フェーズ（0-8時）
    phase2 = 600  # 定常運転（8-18時）
    phase3 = 360  # 停止フェーズ（18-24時）
    
    temperature = np.concatenate([
        np.linspace(25, 175, phase1) + np.random.normal(0, 2, phase1),  # 起動
        175 + np.random.normal(0, 1, phase2),  # 定常
        np.linspace(175, 30, phase3) + np.random.normal(0, 3, phase3)  # 停止
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
    
    # 可視化: 複合グラフ
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. 時系列プロット（フェーズ別に色分け）
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
    
    # 2. 複数変数の同時プロット（2軸）
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
    
    # 3. 相関散布図（フェーズ別）
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
    
    # 4. ヒートマップ（時間帯別の平均値）
    ax4 = fig.add_subplot(gs[2, 1])
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly.index.hour
    hourly_avg = df_hourly.groupby('hour')[['temperature', 'pressure', 'flow_rate']].mean()
    sns.heatmap(hourly_avg.T, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Value'}, ax=ax4)
    ax4.set_title('Hourly Average Heatmap', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Hour of Day', fontsize=11)
    
    # 5. Box plot（フェーズ別の分布）
    ax5 = fig.add_subplot(gs[3, 0])
    df.boxplot(column='temperature', by='phase', ax=ax5, patch_artist=True,
               boxprops=dict(facecolor='#11998e', alpha=0.7))
    ax5.set_title('Temperature Distribution by Phase', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Operating Phase', fontsize=11)
    ax5.set_ylabel('Temperature (°C)', fontsize=11)
    plt.sca(ax5)
    plt.xticks(rotation=0)
    
    # 6. ローリング統計（安定性の可視化）
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
    
    # 統計サマリー
    print("フェーズ別統計サマリー:")
    print(df.groupby('phase')[['temperature', 'pressure', 'flow_rate']].describe())
    

**解説** : 複数の可視化手法を組み合わせることで、データの異なる側面を同時に把握できます。運転フェーズごとの色分け、2軸グラフ、ヒートマップなどは、プロセスエンジニアとのコミュニケーションに非常に有効です。

#### コード例10: インタラクティブダッシュボード（Plotly）
    
    
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # サンプルデータ生成
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=2880, freq='30s')
    
    df = pd.DataFrame({
        'temperature': 175 + np.random.normal(0, 2, 2880) + 3*np.sin(np.arange(2880)*2*np.pi/2880),
        'pressure': 1.5 + np.random.normal(0, 0.08, 2880),
        'flow_rate': 50 + np.random.normal(0, 3, 2880),
        'purity': 98 + np.random.normal(0, 0.5, 2880)
    }, index=dates)
    
    # 異常イベントを追加
    df.loc['2025-01-01 08:00':'2025-01-01 08:15', 'temperature'] += 10
    df.loc['2025-01-01 16:00':'2025-01-01 16:20', 'purity'] -= 2
    
    # インタラクティブダッシュボードの作成
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Reactor Temperature', 'Pressure', 'Flow Rate', 'Product Purity'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # 温度
    fig.add_trace(
        go.Scatter(x=df.index, y=df['temperature'],
                   mode='lines',
                   name='Temperature',
                   line=dict(color='#11998e', width=1.5),
                   hovertemplate='%{x}  
    Temp: %{y:.2f}°C'),
        row=1, col=1
    )
    fig.add_hline(y=175, line_dash="dash", line_color="red",
                  annotation_text="Target", row=1, col=1)
    
    # 圧力
    fig.add_trace(
        go.Scatter(x=df.index, y=df['pressure'],
                   mode='lines',
                   name='Pressure',
                   line=dict(color='#f59e0b', width=1.5),
                   hovertemplate='%{x}  
    Pressure: %{y:.3f} MPa'),
        row=2, col=1
    )
    
    # 流量
    fig.add_trace(
        go.Scatter(x=df.index, y=df['flow_rate'],
                   mode='lines',
                   name='Flow Rate',
                   line=dict(color='#7b2cbf', width=1.5),
                   hovertemplate='%{x}  
    Flow: %{y:.2f} m³/h'),
        row=3, col=1
    )
    
    # 製品純度（品質管理範囲を追加）
    fig.add_trace(
        go.Scatter(x=df.index, y=df['purity'],
                   mode='lines',
                   name='Purity',
                   line=dict(color='#10b981', width=1.5),
                   hovertemplate='%{x}  
    Purity: %{y:.2f}%'),
        row=4, col=1
    )
    fig.add_hrect(y0=97.5, y1=99.0, line_width=0, fillcolor="green", opacity=0.1,
                  annotation_text="Spec Range", row=4, col=1)
    
    # レイアウト設定
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
    
    # インタラクティブ機能の追加
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
    
    # 保存と表示
    fig.write_html("process_monitoring_dashboard.html")
    print("インタラクティブダッシュボードを 'process_monitoring_dashboard.html' に保存しました。")
    print("\n【ダッシュボードの機能】")
    print("✓ ズーム: ドラッグして拡大、ダブルクリックでリセット")
    print("✓ パン: Shiftを押しながらドラッグで移動")
    print("✓ ホバー: データポイントにマウスを合わせて詳細表示")
    print("✓ 時間範囲選択: 上部のボタンで1h/6h/12h/全期間を選択")
    print("✓ HTML形式で保存され、ブラウザで表示可能")
    
    # オプション: Jupyter Notebookで表示する場合
    # fig.show()
    
    # 追加の分析: 異常時刻の特定
    temp_anomaly = df[df['temperature'] > 180]
    purity_anomaly = df[df['purity'] < 97]
    
    print(f"\n温度異常: {len(temp_anomaly)}件")
    if len(temp_anomaly) > 0:
        print(f"発生時刻: {temp_anomaly.index[0]} - {temp_anomaly.index[-1]}")
    
    print(f"\n純度異常: {len(purity_anomaly)}件")
    if len(purity_anomaly) > 0:
        print(f"発生時刻: {purity_anomaly.index[0]} - {purity_anomaly.index[-1]}")
    

**出力例** :
    
    
    インタラクティブダッシュボードを 'process_monitoring_dashboard.html' に保存しました。
    
    【ダッシュボードの機能】
    ✓ ズーム: ドラッグして拡大、ダブルクリックでリセット
    ✓ パン: Shiftを押しながらドラッグで移動
    ✓ ホバー: データポイントにマウスを合わせて詳細表示
    ✓ 時間範囲選択: 上部のボタンで1h/6h/12h/全期間を選択
    ✓ HTML形式で保存され、ブラウザで表示可能
    
    温度異常: 31件
    発生時刻: 2025-01-01 08:00:00 - 2025-01-01 08:15:00
    
    純度異常: 41件
    発生時刻: 2025-01-01 16:00:00 - 2025-01-01 16:20:00
    

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
