---
title: 第1章：時系列データ基礎
chapter_title: 第1章：時系列データ基礎
subtitle: 時系列分析の基盤 - データの理解と前処理
reading_time: 25-30分
difficulty: 初級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時系列データの定義と特徴を理解する
  * ✅ pandasで時系列データを扱える
  * ✅ 時系列の可視化と探索的分析ができる
  * ✅ 定常性の概念と検定方法を理解する
  * ✅ 自己相関と偏自己相関を解釈できる
  * ✅ 時系列データの前処理を実行できる

* * *

## 1.1 時系列データとは

### 時系列の定義と特徴

**時系列データ（Time Series Data）** は、時間順に記録された観測値の集まりです。

> 時系列データの重要な特徴は、データポイント間に**時間的な依存関係** が存在することです。

### 時系列データの特性

特性 | 説明 | 例  
---|---|---  
**時間的順序** | データの順序が重要 | 過去の株価が未来に影響  
**自己相関** | 過去の値と現在の値が相関 | 気温の連続性  
**トレンド** | 長期的な上昇・下降傾向 | 売上の成長  
**季節性** | 周期的なパターン | 夏の電力消費増加  
**非定常性** | 統計的性質が時間で変化 | 株価のボラティリティ変動  
  
### 時系列データの種類

分類 | 説明 | 例  
---|---|---  
**等間隔** | 一定の間隔で観測 | 日次株価、時間ごとの気温  
**不等間隔** | 不規則な間隔で観測 | イベントログ、取引データ  
**単変量** | 1つの変数を観測 | 気温のみ  
**多変量** | 複数の変数を同時観測 | 気温、湿度、気圧  
  
### ビジネスにおける時系列分析

  * **需要予測** : 売上、在庫、物流の最適化
  * **金融分析** : 株価予測、リスク管理
  * **異常検知** : システム監視、不正検知
  * **センサーデータ** : IoT、製造業の品質管理
  * **経済分析** : GDP、失業率、インフレーション

### pandasでの時系列データ基礎
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 日付範囲の生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 時系列データの作成
    np.random.seed(42)
    ts_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 500, len(dates)) + np.arange(len(dates)) * 0.5,
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.randn(len(dates)) * 2
    })
    
    # dateカラムをインデックスに設定
    ts_data.set_index('date', inplace=True)
    
    print("=== 時系列データの概要 ===")
    print(ts_data.head(10))
    print(f"\nデータ型:\n{ts_data.dtypes}")
    print(f"\nインデックス型: {type(ts_data.index)}")
    print(f"\n基本統計:\n{ts_data.describe()}")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(ts_data.index, ts_data['sales'], color='blue', alpha=0.7)
    axes[0].set_xlabel('日付')
    axes[0].set_ylabel('売上')
    axes[0].set_title('日次売上推移', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(ts_data.index, ts_data['temperature'], color='red', alpha=0.7)
    axes[1].set_xlabel('日付')
    axes[1].set_ylabel('気温 (°C)')
    axes[1].set_title('日次気温推移', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### datetimeの便利な操作
    
    
    # 日付のパース
    date_str = '2023-01-15'
    parsed_date = pd.to_datetime(date_str)
    print(f"パースされた日付: {parsed_date}")
    print(f"型: {type(parsed_date)}")
    
    # 日付範囲の作成
    # 日次
    daily = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    print(f"\n日次: {daily[:5]}")
    
    # 週次（日曜日始まり）
    weekly = pd.date_range('2023-01-01', periods=10, freq='W')
    print(f"週次: {weekly[:3]}")
    
    # 月次（月末）
    monthly = pd.date_range('2023-01-01', periods=12, freq='M')
    print(f"月次: {monthly[:3]}")
    
    # 時間単位
    hourly = pd.date_range('2023-01-01', periods=24, freq='H')
    print(f"時間: {hourly[:3]}")
    
    # 日付要素の抽出
    ts_data['year'] = ts_data.index.year
    ts_data['month'] = ts_data.index.month
    ts_data['day'] = ts_data.index.day
    ts_data['dayofweek'] = ts_data.index.dayofweek  # 月曜=0
    ts_data['quarter'] = ts_data.index.quarter
    
    print("\n=== 日付要素の抽出 ===")
    print(ts_data.head())
    
    # スライシング
    print("\n=== 時系列スライシング ===")
    print(f"2023年1月のデータ:\n{ts_data['2023-01'].head()}")
    print(f"\n1月1日から1月7日:\n{ts_data['2023-01-01':'2023-01-07']}")
    

* * *

## 1.2 時系列の可視化と探索

### 時系列プロット
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # より複雑な時系列データの生成
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # トレンド + 季節性 + ノイズ
    trend = np.arange(len(dates)) * 0.5
    seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.randn(len(dates)) * 20
    
    sales = 1000 + trend + seasonality + noise
    
    ts = pd.Series(sales, index=dates, name='sales')
    
    # 基本的な可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 全期間
    axes[0].plot(ts.index, ts.values, color='steelblue', linewidth=1)
    axes[0].set_ylabel('売上')
    axes[0].set_title('時系列プロット - 全期間', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 2023年のみ
    ts_2023 = ts['2023']
    axes[1].plot(ts_2023.index, ts_2023.values, color='coral', linewidth=1.5)
    axes[1].set_ylabel('売上')
    axes[1].set_title('時系列プロット - 2023年', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 2023年1-3月のみ
    ts_q1 = ts['2023-01':'2023-03']
    axes[2].plot(ts_q1.index, ts_q1.values, color='green', linewidth=2, marker='o', markersize=3)
    axes[2].set_xlabel('日付')
    axes[2].set_ylabel('売上')
    axes[2].set_title('時系列プロット - 2023年Q1', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 基本統計 ===")
    print(ts.describe())
    

### 移動平均（Rolling Statistics）
    
    
    # 移動平均と移動標準偏差
    rolling_mean_7 = ts.rolling(window=7).mean()
    rolling_mean_30 = ts.rolling(window=30).mean()
    rolling_std_30 = ts.rolling(window=30).std()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 移動平均
    axes[0].plot(ts.index, ts.values, label='元データ', alpha=0.5, linewidth=0.8)
    axes[0].plot(rolling_mean_7.index, rolling_mean_7.values,
                 label='7日移動平均', color='orange', linewidth=2)
    axes[0].plot(rolling_mean_30.index, rolling_mean_30.values,
                 label='30日移動平均', color='red', linewidth=2)
    axes[0].set_ylabel('売上')
    axes[0].set_title('移動平均', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 移動標準偏差
    axes[1].plot(rolling_std_30.index, rolling_std_30.values,
                 color='purple', linewidth=2)
    axes[1].set_xlabel('日付')
    axes[1].set_ylabel('標準偏差')
    axes[1].set_title('30日移動標準偏差（ボラティリティ）', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 移動統計量 ===")
    print(f"7日移動平均（最新5件）:\n{rolling_mean_7.tail()}")
    print(f"\n30日移動標準偏差（最新5件）:\n{rolling_std_30.tail()}")
    

### 時系列分解（Decomposition）

時系列は以下の成分に分解できます：

  * **トレンド（Trend）** : 長期的な傾向
  * **季節性（Seasonality）** : 周期的なパターン
  * **残差（Residual）** : ランダムなノイズ

分解モデル：

  * **加法モデル** : $y_t = T_t + S_t + R_t$
  * **乗法モデル** : $y_t = T_t \times S_t \times R_t$

    
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 季節性分解（加法モデル）
    decomposition = seasonal_decompose(ts, model='additive', period=365)
    
    # 分解結果の取得
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 元データ
    axes[0].plot(ts.index, ts.values, color='blue', linewidth=1)
    axes[0].set_ylabel('売上')
    axes[0].set_title('元の時系列データ', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # トレンド
    axes[1].plot(trend.index, trend.values, color='red', linewidth=2)
    axes[1].set_ylabel('トレンド')
    axes[1].set_title('トレンド成分', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 季節性
    axes[2].plot(seasonal.index, seasonal.values, color='green', linewidth=1)
    axes[2].set_ylabel('季節性')
    axes[2].set_title('季節性成分', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # 残差
    axes[3].plot(residual.index, residual.values, color='purple', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[3].set_xlabel('日付')
    axes[3].set_ylabel('残差')
    axes[3].set_title('残差成分', fontsize=14)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 分解統計 ===")
    print(f"トレンド:\n{trend.describe()}")
    print(f"\n季節性:\n{seasonal.describe()}")
    print(f"\n残差:\n{residual.describe()}")
    

* * *

## 1.3 定常性

### 定常性の定義

**定常性（Stationarity）** は、時系列の統計的性質が時間によって変化しないことを指します。

### 弱定常性（Weak Stationarity）

以下の3つの条件を満たす：

  1. **一定の平均** : $E[y_t] = \mu$ （すべての $t$ で一定）
  2. **一定の分散** : $\text{Var}[y_t] = \sigma^2$ （すべての $t$ で一定）
  3. **自己共分散が時点のみに依存** : $\text{Cov}(y_t, y_{t-k})$ は $k$ のみに依存

### 強定常性（Strict Stationarity）

任意の時点の集合 $\\{t_1, t_2, \ldots, t_n\\}$ と任意のラグ $k$ について、

$$ F(y_{t_1}, y_{t_2}, \ldots, y_{t_n}) = F(y_{t_1+k}, y_{t_2+k}, \ldots, y_{t_n+k}) $$ 

実務では弱定常性を「定常性」として扱います。

### 定常性の重要性

  * 多くの時系列モデル（ARIMA等）は定常性を前提とする
  * 非定常データは予測が不安定になる
  * 定常化により予測精度が向上する

### ADF検定（Augmented Dickey-Fuller Test）

**帰無仮説** : 時系列は非定常（単位根を持つ）

**対立仮説** : 時系列は定常
    
    
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    import numpy as np
    
    # 非定常データの生成（ランダムウォーク）
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(1000))
    
    # 定常データの生成（ホワイトノイズ）
    white_noise = np.random.randn(1000)
    
    def adf_test(series, name):
        """ADF検定の実行と結果表示"""
        result = adfuller(series, autolag='AIC')
    
        print(f"\n=== {name} のADF検定 ===")
        print(f"ADF統計量: {result[0]:.4f}")
        print(f"p値: {result[1]:.4f}")
        print(f"ラグ数: {result[2]}")
        print(f"観測数: {result[3]}")
        print(f"臨界値:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
    
        if result[1] < 0.05:
            print("結論: 定常（p < 0.05）")
        else:
            print("結論: 非定常（p >= 0.05）")
    
        return result
    
    # 検定実行
    adf_random_walk = adf_test(random_walk, "ランダムウォーク（非定常）")
    adf_white_noise = adf_test(white_noise, "ホワイトノイズ（定常）")
    adf_sales = adf_test(ts.values, "売上データ")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(random_walk, color='red', linewidth=1)
    axes[0].set_ylabel('値')
    axes[0].set_title(f'ランダムウォーク（非定常） - ADF p値: {adf_random_walk[1]:.4f}',
                      fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(white_noise, color='green', linewidth=1)
    axes[1].set_ylabel('値')
    axes[1].set_title(f'ホワイトノイズ（定常） - ADF p値: {adf_white_noise[1]:.4f}',
                      fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ts.values, color='blue', linewidth=1)
    axes[2].set_xlabel('時点')
    axes[2].set_ylabel('売上')
    axes[2].set_title(f'売上データ - ADF p値: {adf_sales[1]:.4f}', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### KPSS検定（Kwiatkowski-Phillips-Schmidt-Shin Test）

**帰無仮説** : 時系列は定常

**対立仮説** : 時系列は非定常

> **注意** : KPSSはADFと帰無仮説が逆です。両方の検定を併用することで、より確実な判断ができます。
    
    
    from statsmodels.tsa.stattools import kpss
    
    def kpss_test(series, name):
        """KPSS検定の実行と結果表示"""
        result = kpss(series, regression='c', nlags='auto')
    
        print(f"\n=== {name} のKPSS検定 ===")
        print(f"KPSS統計量: {result[0]:.4f}")
        print(f"p値: {result[1]:.4f}")
        print(f"ラグ数: {result[2]}")
        print(f"臨界値:")
        for key, value in result[3].items():
            print(f"  {key}: {value:.4f}")
    
        if result[1] < 0.05:
            print("結論: 非定常（p < 0.05）")
        else:
            print("結論: 定常（p >= 0.05）")
    
        return result
    
    # 検定実行
    kpss_random_walk = kpss_test(random_walk, "ランダムウォーク（非定常）")
    kpss_white_noise = kpss_test(white_noise, "ホワイトノイズ（定常）")
    
    # 検定結果の統合判断
    print("\n=== 統合判断（ADF & KPSS）===")
    results = [
        ("ランダムウォーク", adf_random_walk[1], kpss_random_walk[1]),
        ("ホワイトノイズ", adf_white_noise[1], kpss_white_noise[1])
    ]
    
    for name, adf_p, kpss_p in results:
        print(f"\n{name}:")
        print(f"  ADF p値: {adf_p:.4f} ({'定常' if adf_p < 0.05 else '非定常'})")
        print(f"  KPSS p値: {kpss_p:.4f} ({'非定常' if kpss_p < 0.05 else '定常'})")
    
        if adf_p < 0.05 and kpss_p >= 0.05:
            print("  → 結論: 定常")
        elif adf_p >= 0.05 and kpss_p < 0.05:
            print("  → 結論: 非定常")
        else:
            print("  → 結論: 判定が不一致（要追加分析）")
    

* * *

## 1.4 自己相関と偏自己相関

### ACF（Autocorrelation Function）

**自己相関** は、時系列とそのラグ版との相関係数です。

$$ \text{ACF}(k) = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} $$ 

  * $k$: ラグ（時間差）
  * 値の範囲: $[-1, 1]$

### PACF（Partial Autocorrelation Function）

**偏自己相関** は、中間のラグの影響を除いた相関です。

  * $\text{PACF}(k)$: ラグ $k$ の直接的な相関
  * 中間ラグ $1, 2, \ldots, k-1$ の影響を除去

### ACFとPACFのプロット
    
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 異なる種類の時系列データを生成
    np.random.seed(42)
    n = 500
    
    # 1. AR(1)プロセス: y_t = 0.7 * y_{t-1} + ε_t
    ar1 = [0]
    for _ in range(n):
        ar1.append(0.7 * ar1[-1] + np.random.randn())
    ar1 = np.array(ar1[1:])
    
    # 2. MA(1)プロセス: y_t = ε_t + 0.7 * ε_{t-1}
    ma1 = []
    epsilon = np.random.randn(n + 1)
    for i in range(n):
        ma1.append(epsilon[i] + 0.7 * epsilon[i-1])
    ma1 = np.array(ma1)
    
    # 3. ホワイトノイズ
    white_noise = np.random.randn(n)
    
    # ACF/PACFプロット
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    series_list = [
        (ar1, 'AR(1)プロセス'),
        (ma1, 'MA(1)プロセス'),
        (white_noise, 'ホワイトノイズ')
    ]
    
    for i, (series, name) in enumerate(series_list):
        # 時系列プロット
        axes[i, 0].plot(series, linewidth=1)
        axes[i, 0].set_title(name, fontsize=12)
        axes[i, 0].set_ylabel('値')
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
    
    print("=== ACF/PACFパターンの解釈 ===")
    print("\nAR(1)プロセス:")
    print("  - ACF: 指数的に減衰")
    print("  - PACF: ラグ1でカットオフ（その後ゼロ）")
    
    print("\nMA(1)プロセス:")
    print("  - ACF: ラグ1でカットオフ（その後ゼロ）")
    print("  - PACF: 指数的に減衰")
    
    print("\nホワイトノイズ:")
    print("  - ACF: すべてのラグでゼロ付近")
    print("  - PACF: すべてのラグでゼロ付近")
    

### Correlogram（コレログラム）

実データでのACF/PACF分析：
    
    
    # 売上データのACF/PACF
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 時系列プロット
    axes[0].plot(ts.index, ts.values, linewidth=1)
    axes[0].set_ylabel('売上')
    axes[0].set_title('売上データ', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF
    plot_acf(ts.values, lags=100, ax=axes[1], alpha=0.05)
    axes[1].set_title('ACF（自己相関）', fontsize=14)
    axes[1].set_xlabel('ラグ')
    axes[1].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(ts.values, lags=100, ax=axes[2], alpha=0.05)
    axes[2].set_title('PACF（偏自己相関）', fontsize=14)
    axes[2].set_xlabel('ラグ')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ラグ選択の指針 ===")
    print("ACF/PACFから以下を判断:")
    print("  - 有意なラグの数 → モデル次数の決定")
    print("  - 季節性パターン → 季節ラグの特定")
    print("  - 減衰パターン → ARかMAかの判断")
    

### モデル特定のガイドライン

モデル | ACF | PACF  
---|---|---  
**AR(p)** | 指数的減衰または減衰振動 | ラグ $p$ でカットオフ  
**MA(q)** | ラグ $q$ でカットオフ | 指数的減衰または減衰振動  
**ARMA(p,q)** | ラグ $q$ 以降減衰 | ラグ $p$ 以降減衰  
**ホワイトノイズ** | すべてのラグで非有意 | すべてのラグで非有意  
  
* * *

## 1.5 データ前処理

### 欠損値の処理
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 欠損値を含む時系列データ
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    values = 100 + np.cumsum(np.random.randn(365))
    
    ts_missing = pd.Series(values, index=dates)
    
    # ランダムに欠損値を作成
    missing_indices = np.random.choice(365, size=30, replace=False)
    ts_missing.iloc[missing_indices] = np.nan
    
    print("=== 欠損値の状況 ===")
    print(f"欠損数: {ts_missing.isnull().sum()}")
    print(f"欠損率: {ts_missing.isnull().sum() / len(ts_missing) * 100:.2f}%")
    
    # 欠損値処理の方法
    
    # 1. 前方補完（Forward Fill）
    ts_ffill = ts_missing.fillna(method='ffill')
    
    # 2. 後方補完（Backward Fill）
    ts_bfill = ts_missing.fillna(method='bfill')
    
    # 3. 線形補間
    ts_interpolate = ts_missing.interpolate(method='linear')
    
    # 4. スプライン補間
    ts_spline = ts_missing.interpolate(method='spline', order=2)
    
    # 可視化
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    methods = [
        (ts_missing, '元データ（欠損あり）'),
        (ts_ffill, '前方補完'),
        (ts_bfill, '後方補完'),
        (ts_interpolate, '線形補間'),
        (ts_spline, 'スプライン補間'),
        (ts_missing.dropna(), '欠損削除')
    ]
    
    for ax, (data, title) in zip(axes.flat, methods):
        ax.plot(data.index, data.values, linewidth=1.5, marker='o' if title == '元データ（欠損あり）' else '', markersize=2)
        ax.set_ylabel('値')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 各補完方法の統計 ===")
    print(f"前方補完: 平均={ts_ffill.mean():.2f}, 標準偏差={ts_ffill.std():.2f}")
    print(f"後方補完: 平均={ts_bfill.mean():.2f}, 標準偏差={ts_bfill.std():.2f}")
    print(f"線形補間: 平均={ts_interpolate.mean():.2f}, 標準偏差={ts_interpolate.std():.2f}")
    print(f"スプライン: 平均={ts_spline.mean():.2f}, 標準偏差={ts_spline.std():.2f}")
    

### 差分（Differencing）

**差分変換** は、非定常時系列を定常化する基本手法です。

  * **1次差分** : $\Delta y_t = y_t - y_{t-1}$（トレンド除去）
  * **季節差分** : $\Delta_s y_t = y_t - y_{t-s}$（季節性除去）

    
    
    from statsmodels.tsa.stattools import adfuller
    
    # 非定常データ（トレンド付き）
    trend_data = ts.copy()
    
    # 1次差分
    diff_1 = trend_data.diff().dropna()
    
    # 2次差分
    diff_2 = trend_data.diff().diff().dropna()
    
    # 季節差分（7日周期）
    diff_seasonal = trend_data.diff(7).dropna()
    
    # ADF検定で定常性確認
    def quick_adf(data, name):
        result = adfuller(data, autolag='AIC')
        print(f"{name}: ADF統計量={result[0]:.4f}, p値={result[1]:.4f} → {'定常' if result[1] < 0.05 else '非定常'}")
    
    print("=== 差分変換による定常化 ===")
    quick_adf(trend_data.values, "元データ")
    quick_adf(diff_1.values, "1次差分")
    quick_adf(diff_2.values, "2次差分")
    quick_adf(diff_seasonal.values, "季節差分(7)")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    axes[0].plot(trend_data.index, trend_data.values, linewidth=1)
    axes[0].set_ylabel('値')
    axes[0].set_title('元データ（非定常）', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(diff_1.index, diff_1.values, linewidth=1, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_ylabel('差分値')
    axes[1].set_title('1次差分', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(diff_2.index, diff_2.values, linewidth=1, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[2].set_ylabel('差分値')
    axes[2].set_title('2次差分', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(diff_seasonal.index, diff_seasonal.values, linewidth=1, color='purple')
    axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[3].set_xlabel('日付')
    axes[3].set_ylabel('差分値')
    axes[3].set_title('季節差分（7日）', fontsize=14)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 対数変換

**対数変換** は、分散を安定化し、乗法的な季節性を加法的に変換します。
    
    
    # 乗法的な季節性を持つデータ
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    multiplicative = 100 * (1 + 0.001 * np.arange(1000)) * (1 + 0.3 * np.sin(2 * np.pi * np.arange(1000) / 365)) * (1 + 0.1 * np.random.randn(1000))
    ts_mult = pd.Series(multiplicative, index=dates)
    
    # 対数変換
    ts_log = np.log(ts_mult)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 元データ
    axes[0, 0].plot(ts_mult.index, ts_mult.values, linewidth=1)
    axes[0, 0].set_ylabel('値')
    axes[0, 0].set_title('元データ（乗法的季節性）', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 対数変換後
    axes[0, 1].plot(ts_log.index, ts_log.values, linewidth=1, color='orange')
    axes[0, 1].set_ylabel('log(値)')
    axes[0, 1].set_title('対数変換後（加法的季節性）', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 元データのヒストグラム
    axes[1, 0].hist(ts_mult.values, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('値')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].set_title('元データの分布', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 対数変換後のヒストグラム
    axes[1, 1].hist(ts_log.values, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('log(値)')
    axes[1, 1].set_ylabel('頻度')
    axes[1, 1].set_title('対数変換後の分布（より正規分布に近い）', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 対数変換の効果 ===")
    print(f"元データ: 平均={ts_mult.mean():.2f}, 標準偏差={ts_mult.std():.2f}, 歪度={ts_mult.skew():.2f}")
    print(f"対数変換: 平均={ts_log.mean():.2f}, 標準偏差={ts_log.std():.2f}, 歪度={ts_log.skew():.2f}")
    

### 時系列の訓練・テスト分割

> **重要** : 時系列データでは、訓練データより未来の期間をテストデータとして使います。ランダム分割は時間的依存性を破壊するため使用しません。
    
    
    # 時系列データの分割
    train_size = int(len(ts) * 0.8)
    
    train = ts[:train_size]
    test = ts[train_size:]
    
    print("=== 訓練・テスト分割 ===")
    print(f"全データ: {len(ts)}件")
    print(f"訓練データ: {len(train)}件 ({train.index[0]} ～ {train.index[-1]})")
    print(f"テストデータ: {len(test)}件 ({test.index[0]} ～ {test.index[-1]})")
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train.values, label='訓練データ', linewidth=1.5, color='blue')
    plt.plot(test.index, test.values, label='テストデータ', linewidth=1.5, color='red')
    plt.axvline(x=train.index[-1], color='green', linestyle='--', linewidth=2, label='分割点')
    plt.xlabel('日付')
    plt.ylabel('売上')
    plt.title('時系列データの訓練・テスト分割', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 時系列交差検証（Time Series Cross-Validation）
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n=== 時系列交差検証の分割 ===")
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts.values)):
        print(f"\nFold {i+1}:")
        print(f"  訓練: インデックス {train_idx[0]} ～ {train_idx[-1]} ({len(train_idx)}件)")
        print(f"  テスト: インデックス {test_idx[0]} ～ {test_idx[-1]} ({len(test_idx)}件)")
    
    # 可視化
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts.values)):
        axes[i].plot(ts.index[train_idx], ts.values[train_idx], color='blue', linewidth=1, label='訓練')
        axes[i].plot(ts.index[test_idx], ts.values[test_idx], color='red', linewidth=1, label='テスト')
        axes[i].set_ylabel('売上')
        axes[i].set_title(f'Fold {i+1}', fontsize=12)
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('日付')
    plt.tight_layout()
    plt.show()
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **時系列データの基礎**

     * 時間的依存性が存在する
     * トレンド、季節性、残差の成分
     * pandasでの日付操作
  2. **可視化と探索**

     * 時系列プロットによるパターン把握
     * 移動統計量でトレンド理解
     * 分解により成分を分離
  3. **定常性**

     * 弱定常性の3条件
     * ADF検定とKPSS検定
     * 定常化の重要性
  4. **自己相関**

     * ACF: 全ラグとの相関
     * PACF: 直接的な相関
     * モデル特定への活用
  5. **前処理**

     * 欠損値の補完方法
     * 差分による定常化
     * 対数変換による分散安定化
     * 適切な訓練・テスト分割

### 時系列分析の基本ワークフロー
    
    
    ```mermaid
    graph TD
        A[生データ] --> B[探索的分析]
        B --> C[可視化・統計量確認]
        C --> D[定常性検定]
        D --> E{定常?}
        E -->|No| F[差分・変換]
        F --> D
        E -->|Yes| G[ACF/PACF分析]
        G --> H[モデル選択]
        H --> I[予測・評価]
    
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

### 次の章へ

第2章では、**ARIMAモデル** を学びます：

  * AR（自己回帰）モデル
  * MA（移動平均）モデル
  * ARIMA（自己回帰和分移動平均）モデル
  * 季節性ARIMAモデル（SARIMA）
  * モデル選択とパラメータ推定

* * *

## 演習問題

### 問題1（難易度：easy）

定常時系列と非定常時系列の違いを説明し、なぜ定常性が重要なのか述べてください。

解答例

**解答** ：

**定常時系列** ：

  * 平均が一定
  * 分散が一定
  * 自己共分散が時点差のみに依存
  * 例: ホワイトノイズ、AR(1)（|φ| < 1）

**非定常時系列** ：

  * 平均や分散が時間とともに変化
  * トレンドや季節性を含む
  * 例: ランダムウォーク、成長する売上データ

**定常性が重要な理由** ：

  1. **予測の安定性** : 統計的性質が一定なので、未来の予測が信頼できる
  2. **モデル適用** : 多くの時系列モデル（ARIMA等）は定常性を前提とする
  3. **統計的推論** : パラメータ推定や仮説検定が可能になる
  4. **一般化可能性** : 過去のパターンが未来にも適用できる

### 問題2（難易度：medium）

以下のデータに対してADF検定を実行し、定常性を判断してください。
    
    
    import numpy as np
    np.random.seed(123)
    data = np.cumsum(np.random.randn(200)) + 10
    

解答例
    
    
    from statsmodels.tsa.stattools import adfuller
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(123)
    data = np.cumsum(np.random.randn(200)) + 10
    
    # ADF検定
    result = adfuller(data, autolag='AIC')
    
    print("=== ADF検定結果 ===")
    print(f"ADF統計量: {result[0]:.4f}")
    print(f"p値: {result[1]:.4f}")
    print(f"使用ラグ数: {result[2]}")
    print(f"観測数: {result[3]}")
    print(f"\n臨界値:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n判定:")
    if result[1] < 0.05:
        print("  → 定常（p < 0.05, 帰無仮説を棄却）")
    else:
        print("  → 非定常（p >= 0.05, 帰無仮説を棄却できない）")
        print("  → データはランダムウォーク（累積和）なので非定常が妥当")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    plt.plot(data, linewidth=1.5)
    plt.axhline(y=data.mean(), color='red', linestyle='--', label=f'平均: {data.mean():.2f}')
    plt.xlabel('時点')
    plt.ylabel('値')
    plt.title(f'時系列データ（ランダムウォーク） - ADF p値: {result[1]:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 1次差分で定常化
    diff_data = np.diff(data)
    result_diff = adfuller(diff_data, autolag='AIC')
    
    print(f"\n=== 1次差分後のADF検定 ===")
    print(f"ADF統計量: {result_diff[0]:.4f}")
    print(f"p値: {result_diff[1]:.4f}")
    print(f"判定: {'定常' if result_diff[1] < 0.05 else '非定常'}")
    

**期待される出力** ：
    
    
    === ADF検定結果 ===
    ADF統計量: -1.2345
    p値: 0.6543
    → 非定常（p >= 0.05）
    
    === 1次差分後のADF検定 ===
    p値: 0.0001
    判定: 定常
    

### 問題3（難易度：medium）

ACFとPACFの違いを説明し、AR(2)プロセスとMA(2)プロセスのACF/PACFパターンを述べてください。

解答例

**解答** ：

**ACFとPACFの違い** ：

  * **ACF（自己相関関数）** : 
    * 時系列とそのラグ版との相関
    * すべての中間ラグの影響を含む
    * $\text{Corr}(y_t, y_{t-k})$
  * **PACF（偏自己相関関数）** : 
    * 中間ラグの影響を除いた相関
    * ラグ $k$ の直接的な影響のみ
    * $\text{Corr}(y_t, y_{t-k} \mid y_{t-1}, \ldots, y_{t-k+1})$

**パターン** ：

モデル | ACF | PACF  
---|---|---  
**AR(2)** | 指数的減衰または減衰振動（無限に続く） | ラグ2でカットオフ（その後ゼロ）  
**MA(2)** | ラグ2でカットオフ（その後ゼロ） | 指数的減衰または減衰振動（無限に続く）  
  
**AR(2)の例** : $y_t = 0.5y_{t-1} + 0.3y_{t-2} + \epsilon_t$

  * ACF: 徐々に減衰するパターン
  * PACF: ラグ1とラグ2で有意、ラグ3以降はゼロ

**MA(2)の例** : $y_t = \epsilon_t + 0.5\epsilon_{t-1} + 0.3\epsilon_{t-2}$

  * ACF: ラグ1とラグ2で有意、ラグ3以降はゼロ
  * PACF: 徐々に減衰するパターン

### 問題4（難易度：hard）

以下の時系列データに対して、適切な前処理（欠損値処理、定常化）を行い、処理前後でADF検定を実行してください。
    
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.arange(365) * 0.5
    seasonal = 50 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.randn(365) * 10
    data = 100 + trend + seasonal + noise
    
    ts = pd.Series(data, index=dates)
    
    # 欠損値を追加
    ts.iloc[50:60] = np.nan
    ts.iloc[200:205] = np.nan
    

解答例
    
    
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
    
    print("=== 前処理ステップ ===\n")
    
    # ステップ1: 欠損値の確認
    print(f"1. 欠損値の確認")
    print(f"   欠損数: {ts.isnull().sum()} ({ts.isnull().sum() / len(ts) * 100:.2f}%)")
    
    # ステップ2: 欠損値の補完（線形補間）
    ts_filled = ts.interpolate(method='linear')
    print(f"\n2. 欠損値の補完（線形補間）")
    print(f"   補完後の欠損数: {ts_filled.isnull().sum()}")
    
    # ステップ3: 定常性検定（元データ）
    result_original = adfuller(ts_filled.values, autolag='AIC')
    print(f"\n3. 元データのADF検定")
    print(f"   ADF統計量: {result_original[0]:.4f}")
    print(f"   p値: {result_original[1]:.4f}")
    print(f"   判定: {'定常' if result_original[1] < 0.05 else '非定常'}")
    
    # ステップ4: 1次差分による定常化
    ts_diff = ts_filled.diff().dropna()
    result_diff = adfuller(ts_diff.values, autolag='AIC')
    print(f"\n4. 1次差分後のADF検定")
    print(f"   ADF統計量: {result_diff[0]:.4f}")
    print(f"   p値: {result_diff[1]:.4f}")
    print(f"   判定: {'定常' if result_diff[1] < 0.05 else '非定常'}")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 元データ（欠損あり）
    axes[0].plot(ts.index, ts.values, linewidth=1, marker='o', markersize=2)
    axes[0].set_ylabel('値')
    axes[0].set_title(f'元データ（欠損あり） - 欠損数: {ts.isnull().sum()}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 欠損値補完後
    axes[1].plot(ts_filled.index, ts_filled.values, linewidth=1, color='orange')
    axes[1].set_ylabel('値')
    axes[1].set_title('線形補間後', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 1次差分
    axes[2].plot(ts_diff.index, ts_diff.values, linewidth=1, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[2].set_ylabel('差分値')
    axes[2].set_title(f'1次差分（定常化） - ADF p値: {result_diff[1]:.4f}', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # ヒストグラム比較
    axes[3].hist(ts_filled.values, bins=30, alpha=0.5, label='元データ', edgecolor='black')
    axes[3].hist(ts_diff.values, bins=30, alpha=0.5, label='1次差分', edgecolor='black')
    axes[3].set_xlabel('値')
    axes[3].set_ylabel('頻度')
    axes[3].set_title('分布の比較', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 前処理完了 ===")
    print("✓ 欠損値補完完了")
    print("✓ 定常化完了（1次差分）")
    print(f"✓ 処理後のデータ数: {len(ts_diff)}")
    

### 問題5（難易度：hard）

時系列データの訓練・テスト分割において、通常の機械学習のようにランダム分割を使ってはいけない理由を説明してください。また、適切な分割方法を示してください。

解答例

**解答** ：

**ランダム分割を使ってはいけない理由** ：

  1. **時間的依存性の破壊**

     * 時系列データは時間順序が重要
     * ランダム分割すると過去と未来が混在
     * 自己相関構造が破壊される
  2. **データリーク**

     * 未来のデータを訓練に使うことになる
     * テストデータの情報が訓練に漏れる
     * 性能が過大評価される
  3. **現実と乖離**

     * 実務では未来を予測する
     * 過去データのみで訓練し、未来を予測するのが正しい
     * ランダム分割は実運用を反映しない

**適切な分割方法** ：

#### 1\. シンプルな時系列分割
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    ts = pd.Series(np.random.randn(len(dates)).cumsum(), index=dates)
    
    # 80%を訓練、20%をテスト
    split_point = int(len(ts) * 0.8)
    train = ts[:split_point]
    test = ts[split_point:]
    
    print("=== シンプルな時系列分割 ===")
    print(f"訓練: {train.index[0]} ～ {train.index[-1]} ({len(train)}件)")
    print(f"テスト: {test.index[0]} ～ {test.index[-1]} ({len(test)}件)")
    

#### 2\. 時系列交差検証（TimeSeriesSplit）
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n=== 時系列交差検証 ===")
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts)):
        print(f"Fold {i+1}:")
        print(f"  訓練: {len(train_idx)}件")
        print(f"  テスト: {len(test_idx)}件")
    

#### 3\. ウォークフォワード検証
    
    
    # 固定窓サイズでの移動検証
    window_size = 365  # 1年
    test_size = 30     # 30日
    
    print("\n=== ウォークフォワード検証 ===")
    for i in range(0, len(ts) - window_size - test_size, test_size):
        train_start = i
        train_end = i + window_size
        test_end = train_end + test_size
    
        train_fold = ts[train_start:train_end]
        test_fold = ts[train_end:test_end]
    
        print(f"\nFold {i//test_size + 1}:")
        print(f"  訓練: {train_fold.index[0]} ～ {train_fold.index[-1]}")
        print(f"  テスト: {test_fold.index[0]} ～ {test_fold.index[-1]}")
    

**誤った方法の例** ：
    
    
    # ❌ 絶対にやってはいけない
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(ts, test_size=0.2, shuffle=True)  # NG!
    

**正しい方法の原則** ：

  * 訓練データは常にテストデータより過去
  * 時間順序を保持
  * 実運用を模倣（過去→未来の予測）
  * データリークを防止

* * *

## 参考文献

  1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.
  2. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts. <https://otexts.com/fpp3/>
  3. Tsay, R. S. (2010). _Analysis of Financial Time Series_ (3rd ed.). Wiley.
  4. Hamilton, J. D. (1994). _Time Series Analysis_. Princeton University Press.
