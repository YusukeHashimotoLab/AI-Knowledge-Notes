---
title: 第2章：統計的時系列モデル
chapter_title: 第2章：統計的時系列モデル
subtitle: AR、MA、ARIMAモデルによる時系列予測の基礎
reading_time: 30-35分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ AR（自己回帰）モデルの定義とパラメータ推定を理解する
  * ✅ MA（移動平均）モデルの構造とホワイトノイズの関係を理解する
  * ✅ ARIMAモデルの枠組みとモデル選択手法を習得する
  * ✅ 季節性ARIMAモデル（SARIMA）を実装できる
  * ✅ モデルの評価と診断を適切に実行できる
  * ✅ statsmodelsとpmdarima（auto_arima）を使いこなせる

* * *

## 2.1 AR（自己回帰）モデル

### ARモデルとは

**AR（AutoRegressive）モデル** は、過去の値を使って現在の値を予測する線形モデルです。

AR(p)モデルの定義：

$$ y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t $$

  * $y_t$: 時刻tの値
  * $\phi_1, \ldots, \phi_p$: ARパラメータ
  * $p$: ARの次数（ラグの数）
  * $c$: 定数項
  * $\varepsilon_t$: ホワイトノイズ（平均0、分散$\sigma^2$）

> **直感的理解** : 「今日の気温は、昨日と一昨日の気温に影響される」という考え方です。

### ARモデルの実装
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import warnings
    warnings.filterwarnings('ignore')
    
    # サンプルデータ生成（AR(2)プロセス）
    np.random.seed(42)
    n = 200
    epsilon = np.random.normal(0, 1, n)
    
    # AR(2): y_t = 0.6*y_{t-1} - 0.2*y_{t-2} + epsilon_t
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t-1] - 0.2 * y[t-2] + epsilon[t]
    
    # 時系列データとして整形
    ts_data = pd.Series(y, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    print("=== AR(2)プロセスのシミュレーション ===")
    print(f"データ点数: {len(ts_data)}")
    print(f"平均: {ts_data.mean():.3f}")
    print(f"標準偏差: {ts_data.std():.3f}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 時系列プロット
    axes[0].plot(ts_data, linewidth=1.5)
    axes[0].set_xlabel('日付')
    axes[0].set_ylabel('値')
    axes[0].set_title('AR(2)プロセス', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF（自己相関関数）
    plot_acf(ts_data, lags=30, ax=axes[1])
    axes[1].set_title('自己相関関数（ACF）', fontsize=14)
    
    # PACF（偏自己相関関数）
    plot_pacf(ts_data, lags=30, ax=axes[2], method='ywm')
    axes[2].set_title('偏自己相関関数（PACF）', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === AR(2)プロセスのシミュレーション ===
    データ点数: 200
    平均: -0.012
    標準偏差: 1.456
    

> **重要** : PACFを見ることで、ARの次数pを推定できます。PACFがp次で切断（カットオフ）されます。

### パラメータ推定とモデルフィッティング
    
    
    # 訓練・テストデータ分割
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    # AR(2)モデルのフィッティング
    model_ar2 = AutoReg(train, lags=2, trend='c')
    fitted_ar2 = model_ar2.fit()
    
    print("\n=== AR(2)モデルのパラメータ ===")
    print(fitted_ar2.summary())
    
    # パラメータの取得
    params = fitted_ar2.params
    print(f"\n推定されたパラメータ:")
    print(f"定数項: {params['const']:.4f}")
    print(f"phi_1: {params['y.L1']:.4f} (真値: 0.6)")
    print(f"phi_2: {params['y.L2']:.4f} (真値: -0.2)")
    
    # 予測
    predictions = fitted_ar2.predict(start=len(train), end=len(train) + len(test) - 1)
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label='訓練データ', linewidth=1.5)
    plt.plot(test.index, test, label='実測値', linewidth=1.5, color='green')
    plt.plot(test.index, predictions, label='予測値（AR(2)）',
             linewidth=2, linestyle='--', color='red')
    plt.xlabel('日付')
    plt.ylabel('値')
    plt.title('AR(2)モデルによる予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 予測精度
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    
    print(f"\n=== 予測精度 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    

### 次数選択（AIC、BIC）

**AIC（赤池情報量規準）** と**BIC（ベイズ情報量規準）** は、モデルの複雑さとフィット度のバランスを評価する指標です。

$$ \text{AIC} = 2k - 2\ln(L) $$

$$ \text{BIC} = k\ln(n) - 2\ln(L) $$

  * $k$: パラメータ数
  * $n$: サンプルサイズ
  * $L$: 尤度

    
    
    # 異なる次数のARモデルを比較
    max_lag = 10
    aic_values = []
    bic_values = []
    
    for p in range(1, max_lag + 1):
        model = AutoReg(train, lags=p, trend='c')
        fitted = model.fit()
        aic_values.append(fitted.aic)
        bic_values.append(fitted.bic)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(1, max_lag + 1), aic_values, marker='o', linewidth=2)
    axes[0].axvline(x=np.argmin(aic_values) + 1, color='red',
                    linestyle='--', label=f'最小AIC (p={np.argmin(aic_values) + 1})')
    axes[0].set_xlabel('ARの次数 p')
    axes[0].set_ylabel('AIC')
    axes[0].set_title('AICによる次数選択', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, max_lag + 1), bic_values, marker='s',
                 linewidth=2, color='orange')
    axes[1].axvline(x=np.argmin(bic_values) + 1, color='red',
                    linestyle='--', label=f'最小BIC (p={np.argmin(bic_values) + 1})')
    axes[1].set_xlabel('ARの次数 p')
    axes[1].set_ylabel('BIC')
    axes[1].set_title('BICによる次数選択', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 次数選択結果 ===")
    print(f"最適次数（AIC）: {np.argmin(aic_values) + 1}")
    print(f"最適次数（BIC）: {np.argmin(bic_values) + 1}")
    print(f"真の次数: 2")
    

* * *

## 2.2 MA（移動平均）モデル

### MAモデルとは

**MA（Moving Average）モデル** は、過去の予測誤差（ノイズ）の線形結合で現在の値を表現します。

MA(q)モデルの定義：

$$ y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q} $$

  * $y_t$: 時刻tの値
  * $\theta_1, \ldots, \theta_q$: MAパラメータ
  * $q$: MAの次数
  * $\mu$: 平均
  * $\varepsilon_t$: ホワイトノイズ

> **直感的理解** : 「今日の気温は、過去数日間の予測誤差の影響を受ける」という考え方です。

### ホワイトノイズとの関係
    
    
    from statsmodels.tsa.arima.model import ARIMA
    
    # MA(1)プロセスのシミュレーション
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
    
    print("=== MA(1)プロセスのシミュレーション ===")
    print(f"データ点数: {len(ts_ma)}")
    print(f"平均: {ts_ma.mean():.3f}")
    print(f"標準偏差: {ts_ma.std():.3f}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 時系列プロット
    axes[0].plot(ts_ma, linewidth=1.5, color='green')
    axes[0].set_xlabel('日付')
    axes[0].set_ylabel('値')
    axes[0].set_title('MA(1)プロセス', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # ACF
    plot_acf(ts_ma, lags=30, ax=axes[1])
    axes[1].set_title('自己相関関数（ACF）- MA(1)では1次で切断', fontsize=14)
    
    # PACF
    plot_pacf(ts_ma, lags=30, ax=axes[2], method='ywm')
    axes[2].set_title('偏自己相関関数（PACF）', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

> **重要** : MAモデルでは、ACFがq次で切断されます。これがARモデルとの主な識別方法です。

### MAパラメータ推定
    
    
    # MA(1)モデルのフィッティング
    train_ma = ts_ma[:int(len(ts_ma) * 0.8)]
    test_ma = ts_ma[int(len(ts_ma) * 0.8):]
    
    # ARIMA(0,0,1) = MA(1)
    model_ma1 = ARIMA(train_ma, order=(0, 0, 1))
    fitted_ma1 = model_ma1.fit()
    
    print("\n=== MA(1)モデルのパラメータ ===")
    print(fitted_ma1.summary())
    
    # パラメータの取得
    print(f"\n推定されたパラメータ:")
    print(f"theta: {fitted_ma1.params['ma.L1']:.4f} (真値: 0.7)")
    
    # 予測
    forecast_ma = fitted_ma1.forecast(steps=len(test_ma))
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train_ma.index, train_ma, label='訓練データ', linewidth=1.5)
    plt.plot(test_ma.index, test_ma, label='実測値', linewidth=1.5, color='green')
    plt.plot(test_ma.index, forecast_ma, label='予測値（MA(1)）',
             linewidth=2, linestyle='--', color='red')
    plt.xlabel('日付')
    plt.ylabel('値')
    plt.title('MA(1)モデルによる予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 完全な例：ARとMAの比較
    
    
    # 同じデータに対してARとMAをフィッティング
    np.random.seed(42)
    n = 200
    # ARMA(1,1)プロセス
    y_arma = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)
    
    phi = 0.5
    theta = 0.3
    
    for t in range(1, n):
        y_arma[t] = phi * y_arma[t-1] + epsilon[t] + theta * epsilon[t-1]
    
    ts_arma = pd.Series(y_arma, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    train_arma = ts_arma[:160]
    test_arma = ts_arma[160:]
    
    # AR(2)でフィッティング
    model_ar = ARIMA(train_arma, order=(2, 0, 0))
    fitted_ar = model_ar.fit()
    
    # MA(2)でフィッティング
    model_ma = ARIMA(train_arma, order=(0, 0, 2))
    fitted_ma = model_ma.fit()
    
    # ARMA(1,1)でフィッティング（正しいモデル）
    model_arma = ARIMA(train_arma, order=(1, 0, 1))
    fitted_arma = model_arma.fit()
    
    # 予測
    forecast_ar = fitted_ar.forecast(steps=len(test_arma))
    forecast_ma = fitted_ma.forecast(steps=len(test_arma))
    forecast_arma = fitted_arma.forecast(steps=len(test_arma))
    
    # 評価
    from sklearn.metrics import mean_squared_error
    
    rmse_ar = np.sqrt(mean_squared_error(test_arma, forecast_ar))
    rmse_ma = np.sqrt(mean_squared_error(test_arma, forecast_ma))
    rmse_arma = np.sqrt(mean_squared_error(test_arma, forecast_arma))
    
    print("=== モデル比較 ===")
    print(f"AR(2) - AIC: {fitted_ar.aic:.2f}, RMSE: {rmse_ar:.4f}")
    print(f"MA(2) - AIC: {fitted_ma.aic:.2f}, RMSE: {rmse_ma:.4f}")
    print(f"ARMA(1,1) - AIC: {fitted_arma.aic:.2f}, RMSE: {rmse_arma:.4f}")
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(test_arma.index, test_arma, label='実測値', linewidth=2, color='black')
    plt.plot(test_arma.index, forecast_ar, label='AR(2)', linewidth=1.5, linestyle='--')
    plt.plot(test_arma.index, forecast_ma, label='MA(2)', linewidth=1.5, linestyle='--')
    plt.plot(test_arma.index, forecast_arma, label='ARMA(1,1)', linewidth=2, linestyle='--')
    plt.xlabel('日付')
    plt.ylabel('値')
    plt.title('AR vs MA vs ARMA モデル比較', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.3 ARIMAモデル

### ARIMA(p,d,q)の枠組み

**ARIMA（AutoRegressive Integrated Moving Average）モデル** は、非定常時系列を扱うための強力なフレームワークです。

ARIMA(p,d,q)の構成要素：

パラメータ | 意味 | 役割  
---|---|---  
**p** | AR次数 | 過去の値への依存  
**d** | 差分次数 | 非定常性の除去  
**q** | MA次数 | 過去の誤差への依存  
  
ARIMA(p,d,q)モデル：

$$ \phi(B)(1-B)^d y_t = \theta(B)\varepsilon_t $$

  * $B$: バックシフト演算子（$By_t = y_{t-1}$）
  * $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$: AR多項式
  * $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$: MA多項式

### モデル識別の手順
    
    
    ```mermaid
    graph TD
        A[時系列データ] --> B[定常性の確認]
        B -->|非定常| C[差分を取る d=1,2,...]
        B -->|定常| D[ACF/PACFの分析]
        C --> D
        D --> E{パターン識別}
        E -->|PACFが切断| F[ARモデル p=?]
        E -->|ACFが切断| G[MAモデル q=?]
        E -->|両方減衰| H[ARMAモデル p=?, q=?]
        F --> I[モデル推定]
        G --> I
        H --> I
        I --> J[診断: 残差が白色雑音?]
        J -->|No| K[パラメータ調整]
        J -->|Yes| L[最終モデル]
        K --> I
    
        style A fill:#ffebee
        style L fill:#c8e6c9
        style J fill:#fff3e0
    ```

### ARIMAモデルのフィッティング
    
    
    # 非定常時系列の生成（トレンド + ランダムウォーク）
    np.random.seed(42)
    n = 300
    trend = np.linspace(0, 10, n)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    ts_nonstationary = pd.Series(
        trend + random_walk,
        index=pd.date_range('2020-01-01', periods=n, freq='D')
    )
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 元データ
    axes[0].plot(ts_nonstationary, linewidth=1.5)
    axes[0].set_ylabel('値')
    axes[0].set_title('非定常時系列（トレンド + ランダムウォーク）', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 1次差分
    ts_diff1 = ts_nonstationary.diff().dropna()
    axes[1].plot(ts_diff1, linewidth=1.5, color='orange')
    axes[1].set_ylabel('1次差分')
    axes[1].set_title('1次差分系列（d=1）', fontsize=14)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # 2次差分
    ts_diff2 = ts_diff1.diff().dropna()
    axes[2].plot(ts_diff2, linewidth=1.5, color='green')
    axes[2].set_xlabel('日付')
    axes[2].set_ylabel('2次差分')
    axes[2].set_title('2次差分系列（d=2）', fontsize=14)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 定常性検定（ADF検定）
    from statsmodels.tsa.stattools import adfuller
    
    def adf_test(series, name=''):
        result = adfuller(series.dropna())
        print(f'\n=== ADF検定: {name} ===')
        print(f'ADF統計量: {result[0]:.4f}')
        print(f'p値: {result[1]:.4f}')
        print(f'臨界値:')
        for key, value in result[4].items():
            print(f'  {key}: {value:.4f}')
        if result[1] <= 0.05:
            print("→ 定常（5%有意水準で帰無仮説を棄却）")
        else:
            print("→ 非定常（帰無仮説を棄却できない）")
    
    adf_test(ts_nonstationary, '元データ')
    adf_test(ts_diff1, '1次差分')
    adf_test(ts_diff2, '2次差分')
    

### ARIMAモデルの推定と予測
    
    
    # 訓練・テストデータ分割
    train_size = int(len(ts_nonstationary) * 0.8)
    train_ns = ts_nonstationary[:train_size]
    test_ns = ts_nonstationary[train_size:]
    
    # ARIMA(1,1,1)モデル
    model_arima = ARIMA(train_ns, order=(1, 1, 1))
    fitted_arima = model_arima.fit()
    
    print("\n=== ARIMA(1,1,1)モデル ===")
    print(fitted_arima.summary())
    
    # 予測
    forecast_arima = fitted_arima.forecast(steps=len(test_ns))
    forecast_ci = fitted_arima.get_forecast(steps=len(test_ns)).conf_int()
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train_ns.index, train_ns, label='訓練データ', linewidth=1.5)
    plt.plot(test_ns.index, test_ns, label='実測値', linewidth=1.5, color='green')
    plt.plot(test_ns.index, forecast_arima, label='予測値',
             linewidth=2, linestyle='--', color='red')
    plt.fill_between(test_ns.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95%信頼区間')
    plt.xlabel('日付')
    plt.ylabel('値')
    plt.title('ARIMA(1,1,1)モデルによる予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 精度評価
    rmse = np.sqrt(mean_squared_error(test_ns, forecast_arima))
    mae = mean_absolute_error(test_ns, forecast_arima)
    
    print(f"\n=== 予測精度 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    

### 残差診断
    
    
    # 残差の診断
    residuals = fitted_arima.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 残差の時系列プロット
    axes[0, 0].plot(residuals, linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差の時系列プロット', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差のヒストグラム
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('残差')
    axes[0, 1].set_ylabel('頻度')
    axes[0, 1].set_title('残差の分布（正規性の確認）', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差のACF
    plot_acf(residuals, lags=30, ax=axes[1, 0])
    axes[1, 0].set_title('残差のACF（ホワイトノイズの確認）', fontsize=14)
    
    # Q-Qプロット
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Qプロット（正規性の確認）', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box検定（残差の自己相関検定）
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\n=== Ljung-Box検定（残差の自己相関） ===")
    print(lb_test)
    print("\np値 > 0.05 なら、残差は白色雑音（良いモデル）")
    

* * *

## 2.4 季節性ARIMA（SARIMA）

### SARIMAモデルとは

**SARIMA（Seasonal ARIMA）モデル** は、季節性を持つ時系列データを扱うためのモデルです。

SARIMA(p,d,q)(P,D,Q)sの表記：

  * **(p,d,q)** : 非季節成分のARIMAパラメータ
  * **(P,D,Q)** : 季節成分のARIMAパラメータ
  * **s** : 季節周期（月次データなら12、四半期なら4）

### 季節性データの生成と可視化
    
    
    # 季節性を持つ時系列データの生成
    np.random.seed(42)
    n = 240  # 20年分の月次データ
    t = np.arange(n)
    
    # トレンド + 季節性 + ノイズ
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # 年周期
    noise = np.random.normal(0, 2, n)
    
    ts_seasonal = pd.Series(
        trend + seasonal + noise,
        index=pd.date_range('2000-01-01', periods=n, freq='MS')
    )
    
    print("=== 季節性時系列データ ===")
    print(f"データ点数: {len(ts_seasonal)}")
    print(f"周期: 12ヶ月（年次季節性）")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 時系列プロット
    axes[0].plot(ts_seasonal, linewidth=1.5)
    axes[0].set_xlabel('年月')
    axes[0].set_ylabel('値')
    axes[0].set_title('季節性を持つ時系列データ', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 季節性分解
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(ts_seasonal, model='additive', period=12)
    
    # 分解結果の表示
    fig2 = decomposition.plot()
    fig2.set_size_inches(14, 10)
    plt.tight_layout()
    plt.show()
    

### 季節差分
    
    
    # 季節差分（s=12）
    ts_seasonal_diff = ts_seasonal.diff(12).dropna()
    
    # 通常の差分も適用
    ts_seasonal_diff2 = ts_seasonal_diff.diff().dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 元データ
    axes[0].plot(ts_seasonal, linewidth=1.5)
    axes[0].set_ylabel('値')
    axes[0].set_title('元データ（季節性あり）', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 季節差分後
    axes[1].plot(ts_seasonal_diff, linewidth=1.5, color='orange')
    axes[1].set_ylabel('季節差分')
    axes[1].set_title('季節差分後（D=1, s=12）', fontsize=14)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # 季節差分 + 通常の差分
    axes[2].plot(ts_seasonal_diff2, linewidth=1.5, color='green')
    axes[2].set_xlabel('年月')
    axes[2].set_ylabel('差分')
    axes[2].set_title('季節差分 + 通常差分（D=1, d=1）', fontsize=14)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 定常性検定
    adf_test(ts_seasonal, '元データ')
    adf_test(ts_seasonal_diff, '季節差分後')
    adf_test(ts_seasonal_diff2, '季節差分+通常差分')
    

### SARIMAモデルのフィッティング
    
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # 訓練・テストデータ分割
    train_seasonal = ts_seasonal[:int(len(ts_seasonal) * 0.8)]
    test_seasonal = ts_seasonal[int(len(ts_seasonal) * 0.8):]
    
    # SARIMA(1,1,1)(1,1,1)12モデル
    model_sarima = SARIMAX(
        train_seasonal,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_sarima = model_sarima.fit(disp=False)
    
    print("\n=== SARIMA(1,1,1)(1,1,1)12モデル ===")
    print(fitted_sarima.summary())
    
    # 予測
    forecast_sarima = fitted_sarima.forecast(steps=len(test_seasonal))
    forecast_sarima_ci = fitted_sarima.get_forecast(steps=len(test_seasonal)).conf_int()
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train_seasonal.index, train_seasonal, label='訓練データ', linewidth=1.5)
    plt.plot(test_seasonal.index, test_seasonal, label='実測値', linewidth=1.5, color='green')
    plt.plot(test_seasonal.index, forecast_sarima, label='予測値（SARIMA）',
             linewidth=2, linestyle='--', color='red')
    plt.fill_between(test_seasonal.index,
                     forecast_sarima_ci.iloc[:, 0],
                     forecast_sarima_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95%信頼区間')
    plt.xlabel('年月')
    plt.ylabel('値')
    plt.title('SARIMA(1,1,1)(1,1,1)12による予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 精度評価
    rmse_sarima = np.sqrt(mean_squared_error(test_seasonal, forecast_sarima))
    mae_sarima = mean_absolute_error(test_seasonal, forecast_sarima)
    
    print(f"\n=== SARIMAモデルの予測精度 ===")
    print(f"RMSE: {rmse_sarima:.4f}")
    print(f"MAE: {mae_sarima:.4f}")
    

### auto_arimaによる自動パラメータ選択
    
    
    # pmdarimaライブラリのauto_arimaを使用
    # pip install pmdarima が必要
    
    from pmdarima import auto_arima
    
    print("\n=== auto_arimaによる最適パラメータ探索 ===")
    print("探索中...")
    
    # auto_arima（季節性あり）
    auto_model = auto_arima(
        train_seasonal,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        m=12,  # 季節周期
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        seasonal=True,
        d=None,  # 自動推定
        D=None,  # 自動推定
        trace=True,  # 探索過程を表示
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print("\n=== 最適モデル ===")
    print(auto_model.summary())
    
    # 予測
    forecast_auto = auto_model.predict(n_periods=len(test_seasonal))
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(test_seasonal.index, test_seasonal, label='実測値', linewidth=2, color='green')
    plt.plot(test_seasonal.index, forecast_sarima, label='手動SARIMA',
             linewidth=1.5, linestyle='--', color='red')
    plt.plot(test_seasonal.index, forecast_auto, label='auto_arima',
             linewidth=1.5, linestyle='--', color='blue')
    plt.xlabel('年月')
    plt.ylabel('値')
    plt.title('SARIMAモデル比較（手動 vs auto_arima）', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 精度比較
    rmse_auto = np.sqrt(mean_squared_error(test_seasonal, forecast_auto))
    print(f"\n=== 精度比較 ===")
    print(f"手動SARIMA RMSE: {rmse_sarima:.4f}")
    print(f"auto_arima RMSE: {rmse_auto:.4f}")
    

* * *

## 2.5 モデル評価と選択

### In-sample vs Out-of-sample評価

評価方法 | 説明 | 用途  
---|---|---  
**In-sample** | 訓練データでの予測精度 | モデルのフィット度確認  
**Out-of-sample** | テストデータでの予測精度 | 汎化性能の評価（重要）  
  
### 評価指標

**1\. RMSE（Root Mean Squared Error）**

$$ \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$

**2\. MAE（Mean Absolute Error）**

$$ \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$

**3\. MAPE（Mean Absolute Percentage Error）**

$$ \text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| $$
    
    
    # 複数の評価指標を計算
    def evaluate_forecast(y_true, y_pred, model_name='Model'):
        """予測精度の総合評価"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    
        # MAPE（0除算を避ける）
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
        # R^2スコア
        r2 = r2_score(y_true, y_pred)
    
        print(f"\n=== {model_name}の評価 ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R^2: {r2:.4f}")
    
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    
    # 複数モデルの比較
    results = {}
    results['ARIMA'] = evaluate_forecast(test_ns, forecast_arima, 'ARIMA(1,1,1)')
    results['SARIMA'] = evaluate_forecast(test_seasonal, forecast_sarima, 'SARIMA(1,1,1)(1,1,1)12')
    results['auto_arima'] = evaluate_forecast(test_seasonal, forecast_auto, 'auto_arima')
    
    # 結果の比較表
    results_df = pd.DataFrame(results).T
    print("\n=== モデル性能比較表 ===")
    print(results_df)
    

### 時系列交差検証

時系列データでは、通常の交差検証ではなく、**時間を考慮した交差検証** が必要です。
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    # 時系列交差検証
    tscv = TimeSeriesSplit(n_splits=5)
    
    rmse_scores = []
    
    print("\n=== 時系列交差検証 ===")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_seasonal), 1):
        # データ分割
        cv_train = ts_seasonal.iloc[train_idx]
        cv_test = ts_seasonal.iloc[test_idx]
    
        # モデルフィッティング
        cv_model = SARIMAX(cv_train, order=(1,1,1), seasonal_order=(1,1,1,12))
        cv_fitted = cv_model.fit(disp=False)
    
        # 予測
        cv_forecast = cv_fitted.forecast(steps=len(cv_test))
    
        # 評価
        cv_rmse = np.sqrt(mean_squared_error(cv_test, cv_forecast))
        rmse_scores.append(cv_rmse)
    
        print(f"Fold {fold}: 訓練={len(cv_train)}, テスト={len(cv_test)}, RMSE={cv_rmse:.4f}")
    
    print(f"\n平均RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linewidth=2, markersize=10)
    plt.axhline(y=np.mean(rmse_scores), color='red', linestyle='--',
                label=f'平均RMSE: {np.mean(rmse_scores):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('時系列交差検証によるRMSE', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### モデル比較の完全な例
    
    
    # 複数のARIMAモデルを比較
    candidates = [
        ('ARIMA(1,1,0)', (1, 1, 0)),
        ('ARIMA(0,1,1)', (0, 1, 1)),
        ('ARIMA(1,1,1)', (1, 1, 1)),
        ('ARIMA(2,1,1)', (2, 1, 1)),
        ('ARIMA(1,1,2)', (1, 1, 2)),
    ]
    
    comparison_results = []
    
    print("\n=== ARIMAモデル比較 ===")
    for name, order in candidates:
        # モデルフィッティング
        model = ARIMA(train_ns, order=order)
        fitted = model.fit()
    
        # 予測
        forecast = fitted.forecast(steps=len(test_ns))
    
        # 評価
        rmse = np.sqrt(mean_squared_error(test_ns, forecast))
        mae = mean_absolute_error(test_ns, forecast)
        aic = fitted.aic
        bic = fitted.bic
    
        comparison_results.append({
            'モデル': name,
            'AIC': aic,
            'BIC': bic,
            'RMSE': rmse,
            'MAE': mae
        })
    
        print(f"{name}: AIC={aic:.2f}, BIC={bic:.2f}, RMSE={rmse:.4f}")
    
    # 結果をDataFrameに
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("\n=== 最終比較表（RMSEでソート）===")
    print(comparison_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIC比較
    axes[0].barh(comparison_df['モデル'], comparison_df['AIC'], color='steelblue')
    axes[0].set_xlabel('AIC')
    axes[0].set_title('モデル比較: AIC（小さいほど良い）', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # RMSE比較
    axes[1].barh(comparison_df['モデル'], comparison_df['RMSE'], color='coral')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('モデル比較: RMSE（小さいほど良い）', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **AR（自己回帰）モデル**

     * 過去の値を使った線形予測モデル
     * PACFによる次数pの識別
     * AIC/BICによる最適次数の選択
  2. **MA（移動平均）モデル**

     * 過去の予測誤差を使ったモデル
     * ACFによる次数qの識別
     * ホワイトノイズとの関係
  3. **ARIMAモデル**

     * 非定常時系列を扱う統一的枠組み
     * 差分によるトレンドの除去
     * 残差診断による適合度確認
  4. **SARIMAモデル**

     * 季節性を持つ時系列への対応
     * 季節差分による季節成分の除去
     * auto_arimaによる自動パラメータ選択
  5. **モデル評価**

     * RMSE、MAE、MAPEによる精度評価
     * 時系列交差検証
     * In-sample vs Out-of-sample評価

### モデル選択のガイドライン

データの特徴 | 推奨モデル | 理由  
---|---|---  
定常、短期記憶 | AR | 直近の値が重要  
定常、ショック依存 | MA | 過去の誤差が影響  
非定常、トレンド | ARIMA | 差分で定常化  
季節性あり | SARIMA | 季節成分をモデル化  
複雑な季節性 | auto_arima | 自動パラメータ探索  
  
### 実務での注意点

注意点 | 説明  
---|---  
**定常性の確認** | ADF検定で必ず定常性を確認してから適用  
**残差診断** | 残差が白色雑音であることを確認（Ljung-Box検定）  
**過学習回避** | 次数を大きくしすぎない（AIC/BICで選択）  
**Out-of-sample評価** | 訓練データだけでなく、テストデータで性能確認  
**信頼区間の提示** | 点予測だけでなく、不確実性も伝える  
  
### 次の章へ

第3章では、**機械学習による時系列予測** を学びます：

  * 特徴量エンジニアリング（ラグ特徴量、ローリング統計量）
  * ランダムフォレスト・XGBoostによる予測
  * LSTMニューラルネットワーク
  * Prophet（Facebook製）
  * 統計モデルとの比較

* * *

## 演習問題

### 問題1（難易度：easy）

ARモデルとMAモデルの違いを、ACFとPACFのパターンを含めて説明してください。

解答例

**解答** ：

**ARモデル** ：

  * 定義: 過去の値の線形結合で現在の値を予測
  * ACF: 指数的に減衰
  * PACF: p次で切断（カットオフ）
  * 例: AR(2)では、PACF が2次以降でゼロになる

**MAモデル** ：

  * 定義: 過去の予測誤差の線形結合で現在の値を表現
  * ACF: q次で切断（カットオフ）
  * PACF: 指数的に減衰
  * 例: MA(1)では、ACF が1次以降でゼロになる

特徴 | ARモデル | MAモデル  
---|---|---  
依存対象 | 過去の値 | 過去の誤差  
ACFパターン | 減衰 | q次で切断  
PACFパターン | p次で切断 | 減衰  
識別方法 | PACFから次数p | ACFから次数q  
  
### 問題2（難易度：medium）

以下の時系列データに対して、適切なARIMA(p,d,q)の次数を決定し、モデルをフィッティングしてください。
    
    
    import numpy as np
    import pandas as pd
    np.random.seed(123)
    n = 150
    trend = 0.1 * np.arange(n)
    noise = np.random.normal(0, 1, n)
    ts = pd.Series(trend + np.cumsum(noise))
    

解答例
    
    
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(123)
    n = 150
    trend = 0.1 * np.arange(n)
    noise = np.random.normal(0, 1, n)
    ts = pd.Series(trend + np.cumsum(noise))
    
    # ステップ1: 定常性検定
    print("=== ステップ1: 定常性の確認 ===")
    result = adfuller(ts)
    print(f"ADF統計量: {result[0]:.4f}")
    print(f"p値: {result[1]:.4f}")
    
    if result[1] > 0.05:
        print("→ 非定常（差分が必要）")
    
        # 1次差分
        ts_diff = ts.diff().dropna()
        result_diff = adfuller(ts_diff)
        print(f"\n1次差分後のp値: {result_diff[1]:.4f}")
    
        if result_diff[1] <= 0.05:
            print("→ 1次差分で定常化（d=1）")
            d = 1
    else:
        print("→ 定常（d=0）")
        d = 0
    
    # ステップ2: ACF/PACFの確認
    print("\n=== ステップ2: ACF/PACFによる次数の推定 ===")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    if d == 1:
        plot_data = ts.diff().dropna()
    else:
        plot_data = ts
    
    plot_acf(plot_data, lags=20, ax=axes[0])
    axes[0].set_title('ACF（MAの次数qを推定）')
    
    plot_pacf(plot_data, lags=20, ax=axes[1], method='ywm')
    axes[1].set_title('PACF（ARの次数pを推定）')
    
    plt.tight_layout()
    plt.show()
    
    # PACFとACFから次数を推定（ここでは例として p=1, q=1）
    p = 1
    q = 1
    
    print(f"\n推定された次数: ARIMA({p},{d},{q})")
    
    # ステップ3: モデルフィッティング
    print("\n=== ステップ3: モデルのフィッティング ===")
    model = ARIMA(ts, order=(p, d, q))
    fitted = model.fit()
    print(fitted.summary())
    
    # ステップ4: 残差診断
    print("\n=== ステップ4: 残差診断 ===")
    residuals = fitted.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('残差の時系列プロット')
    
    axes[0, 1].hist(residuals, bins=20, edgecolor='black')
    axes[0, 1].set_title('残差の分布')
    
    plot_acf(residuals, ax=axes[1, 0], lags=20)
    axes[1, 0].set_title('残差のACF')
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Qプロット')
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box検定
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f"\nLjung-Box検定 p値: {lb_test['lb_pvalue'].values[0]:.4f}")
    if lb_test['lb_pvalue'].values[0] > 0.05:
        print("→ 残差は白色雑音（良いモデル）")
    else:
        print("→ 残差に自己相関あり（モデル改善の余地）")
    
    print(f"\n=== 最終モデル: ARIMA({p},{d},{q}) ===")
    print(f"AIC: {fitted.aic:.2f}")
    print(f"BIC: {fitted.bic:.2f}")
    

### 問題3（難易度：medium）

AICとBICの違いを説明し、どのような場面でどちらを優先すべきか述べてください。

解答例

**解答** ：

**AIC（赤池情報量規準）** ：

$$ \text{AIC} = 2k - 2\ln(L) $$

  * パラメータ数kのペナルティが軽い
  * 予測精度の最大化を重視
  * やや複雑なモデルを選択しやすい

**BIC（ベイズ情報量規準）** ：

$$ \text{BIC} = k\ln(n) - 2\ln(L) $$

  * サンプルサイズnに依存したペナルティ
  * モデルの簡潔さを重視
  * n > 8 で AIC よりも強いペナルティ

**使い分けのガイドライン** ：

状況 | 推奨 | 理由  
---|---|---  
予測が主目的 | AIC | 予測精度を優先  
解釈が主目的 | BIC | シンプルなモデルが解釈しやすい  
サンプルサイズ小 | AIC | BICのペナルティが過度に強い  
サンプルサイズ大 | BIC | 過学習を防ぐ  
真のモデル推定 | BIC | 一致性を持つ  
  
**実務での推奨** ：

  * 両方を計算し、大きく異なる場合は理由を調査
  * 最終的には、Out-of-sample性能で判断
  * ドメイン知識も考慮してモデル選択

### 問題4（難易度：hard）

以下の月次売上データに対して、SARIMAモデルをフィッティングし、次の12ヶ月を予測してください。季節周期は12ヶ月です。
    
    
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n = 60  # 5年分
    t = np.arange(n)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise
    ts_sales = pd.Series(sales, index=pd.date_range('2019-01-01', periods=n, freq='MS'))
    

解答例
    
    
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # データ生成
    np.random.seed(42)
    n = 60
    t = np.arange(n)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    sales = trend + seasonal + noise
    ts_sales = pd.Series(sales, index=pd.date_range('2019-01-01', periods=n, freq='MS'))
    
    print("=== ステップ1: データの確認と分解 ===")
    print(f"データ点数: {len(ts_sales)}")
    print(f"期間: {ts_sales.index[0]} から {ts_sales.index[-1]}")
    
    # 季節性分解
    decomposition = seasonal_decompose(ts_sales, model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    plt.show()
    
    # ステップ2: 訓練・テストデータ分割
    train_size = 48  # 4年分を訓練、1年分をテスト
    train = ts_sales[:train_size]
    test = ts_sales[train_size:]
    
    print(f"\n訓練データ: {len(train)}ヶ月")
    print(f"テストデータ: {len(test)}ヶ月")
    
    # ステップ3: SARIMAモデルのフィッティング
    # SARIMA(1,1,1)(1,1,1)12 を試す
    print("\n=== ステップ2: SARIMAモデルのフィッティング ===")
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted = model.fit(disp=False)
    print(fitted.summary())
    
    # ステップ4: テストデータでの予測
    forecast = fitted.forecast(steps=len(test))
    forecast_ci = fitted.get_forecast(steps=len(test)).conf_int()
    
    # 評価
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"\n=== テストデータでの予測精度 ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label='訓練データ', linewidth=1.5)
    plt.plot(test.index, test, label='実測値', linewidth=2, color='green', marker='o')
    plt.plot(test.index, forecast, label='SARIMA予測',
             linewidth=2, linestyle='--', color='red', marker='s')
    plt.fill_between(test.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95%信頼区間')
    plt.xlabel('年月')
    plt.ylabel('売上')
    plt.title('SARIMA(1,1,1)(1,1,1)12による月次売上予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ステップ5: 次の12ヶ月の予測
    print("\n=== ステップ3: 全データで再学習し、次の12ヶ月を予測 ===")
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
    
    # 将来の日付インデックス
    future_dates = pd.date_range(ts_sales.index[-1] + pd.DateOffset(months=1),
                                 periods=12, freq='MS')
    
    print("\n次の12ヶ月の予測:")
    for date, value, lower, upper in zip(future_dates, future_forecast,
                                          future_ci.iloc[:, 0], future_ci.iloc[:, 1]):
        print(f"{date.strftime('%Y-%m')}: {value:.2f} (95%CI: [{lower:.2f}, {upper:.2f}])")
    
    # 将来予測の可視化
    plt.figure(figsize=(14, 6))
    plt.plot(ts_sales.index, ts_sales, label='実績データ', linewidth=2, color='blue')
    plt.plot(future_dates, future_forecast, label='12ヶ月先予測',
             linewidth=2, linestyle='--', color='red', marker='o')
    plt.fill_between(future_dates,
                     future_ci.iloc[:, 0],
                     future_ci.iloc[:, 1],
                     alpha=0.2, color='red', label='95%信頼区間')
    plt.axvline(x=ts_sales.index[-1], color='gray', linestyle='--', alpha=0.5, label='予測開始')
    plt.xlabel('年月')
    plt.ylabel('売上')
    plt.title('月次売上の12ヶ月先予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 残差診断
    print("\n=== ステップ4: 残差診断 ===")
    residuals = final_fitted.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('残差の時系列プロット')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('残差の分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    plot_acf(residuals, ax=axes[1, 0], lags=24)
    axes[1, 0].set_title('残差のACF')
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Qプロット')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
    print("\nLjung-Box検定:")
    print(lb_test)
    

### 問題5（難易度：hard）

時系列交差検証（TimeSeriesSplit）が通常の交差検証（KFold）と異なる理由を説明し、時系列データで通常の交差検証を使うとどのような問題が発生するか述べてください。

解答例

**解答** ：

**時系列交差検証の特徴** ：

  1. **時間順序の保持** : 訓練データは常にテストデータより過去
  2. **累積的な訓練** : 訓練データサイズが徐々に増加
  3. **未来データの排除** : テストデータが訓練に含まれない

**通常の交差検証（KFold）の問題** ：

問題 | 説明 | 影響  
---|---|---  
**データリーク** | 未来のデータで訓練、過去を予測 | 性能の過大評価  
**自己相関の無視** | 時間的依存性を考慮しない | 不適切な分割  
**現実との乖離** | 実運用では未来は不明 | デプロイ後の性能劣化  
  
**具体例** ：
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, TimeSeriesSplit
    import matplotlib.pyplot as plt
    
    # 時系列データ（トレンドあり）
    n = 100
    ts = pd.Series(np.arange(n) + np.random.normal(0, 5, n))
    
    # 通常のKFold（問題あり）
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # TimeSeriesSplit（正しい）
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # KFold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(ts), 1):
        axes[0].scatter(train_idx, [fold] * len(train_idx),
                        c='blue', marker='|', s=100, label='訓練' if fold == 1 else '')
        axes[0].scatter(test_idx, [fold] * len(test_idx),
                        c='red', marker='|', s=100, label='テスト' if fold == 1 else '')
    
    axes[0].set_ylabel('Fold')
    axes[0].set_xlabel('時間インデックス')
    axes[0].set_title('通常のKFold（問題あり: 未来で訓練、過去を予測）', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TimeSeriesSplit
    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts), 1):
        axes[1].scatter(train_idx, [fold] * len(train_idx),
                        c='blue', marker='|', s=100, label='訓練' if fold == 1 else '')
        axes[1].scatter(test_idx, [fold] * len(test_idx),
                        c='red', marker='|', s=100, label='テスト' if fold == 1 else '')
    
    axes[1].set_ylabel('Fold')
    axes[1].set_xlabel('時間インデックス')
    axes[1].set_title('TimeSeriesSplit（正しい: 過去で訓練、未来を予測）', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**結論** ：

  * 時系列データでは、必ず**TimeSeriesSplit** を使用
  * 時間順序を保持し、未来のデータリークを防ぐ
  * 実運用と同じ条件で評価できる

* * *

## 参考文献

  1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.
  2. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts. Available at: <https://otexts.com/fpp3/>
  3. Shumway, R. H., & Stoffer, D. S. (2017). _Time Series Analysis and Its Applications: With R Examples_ (4th ed.). Springer.
  4. Brockwell, P. J., & Davis, R. A. (2016). _Introduction to Time Series and Forecasting_ (3rd ed.). Springer.
  5. Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. _Proceedings of the 9th Python in Science Conference_.
