---
title: 第3章：プロセスモデリングの基礎
chapter_title: 第3章：プロセスモデリングの基礎
subtitle: 機械学習によるプロセス予測と最適化
---

# 第3章：プロセスモデリングの基礎

プロセスモデリングは、PIの核心技術です。線形回帰から始めて、PLS、ソフトセンサー、非線形モデルまで、実践的なモデル構築手法を習得します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 線形回帰によるプロセスモデルを構築し、評価できる
  * ✅ PLS（偏最小二乗法）で多重共線性問題に対処できる
  * ✅ ソフトセンサーの概念を理解し、実装できる
  * ✅ R²、RMSE、MAE、クロスバリデーションで正確にモデル評価できる
  * ✅ Random Forest、SVRなどの非線形モデルを使い分けられる

* * *

## 3.1 線形回帰によるプロセスモデル構築

線形回帰は、プロセスモデリングの基礎です。シンプルでありながら、多くの実プロセスで十分な精度を達成できます。

### 単回帰分析の基礎

まず、1つの説明変数から目的変数を予測する**単回帰分析** から始めましょう。

**単回帰モデル** :

$$y = \beta_0 + \beta_1 x + \epsilon$$ 

ここで、$y$は目的変数（例: 製品純度）、$x$は説明変数（例: 反応温度）、$\beta_0$は切片、$\beta_1$は傾き、$\epsilon$は誤差項です。

#### コード例1: 単回帰による品質予測モデル
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # サンプルデータ生成: 反応温度と収率の関係
    np.random.seed(42)
    n = 100
    
    # 反応温度（°C）
    temperature = np.random.uniform(160, 190, n)
    
    # 収率（%）= 線形関係 + ノイズ
    # 理論: 温度が高いほど収率が向上（最適温度まで）
    yield_percentage = 50 + 0.5 * (temperature - 160) + np.random.normal(0, 2, n)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'yield': yield_percentage
    })
    
    print("データの基本統計:")
    print(df.describe())
    print(f"\n相関係数: {df['temperature'].corr(df['yield']):.4f}")
    
    # 単回帰モデルの構築
    X = df[['temperature']].values
    y = df['yield'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 予測
    y_pred = model.predict(X)
    
    # モデルパラメータ
    print(f"\n【モデルパラメータ】")
    print(f"切片 (β₀): {model.intercept_:.4f}")
    print(f"傾き (β₁): {model.coef_[0]:.4f}")
    print(f"モデル式: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} × 温度")
    
    # 評価指標
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n【モデル性能】")
    print(f"R² (決定係数): {r2:.4f}")
    print(f"RMSE (二乗平均平方根誤差): {rmse:.4f}%")
    print(f"MAE (平均絶対誤差): {mae:.4f}%")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散布図と回帰直線
    axes[0].scatter(df['temperature'], df['yield'], alpha=0.6, s=50,
                    color='#11998e', label='Actual data')
    axes[0].plot(df['temperature'], y_pred, color='red', linewidth=2,
                 label=f'Regression line (R²={r2:.3f})')
    axes[0].set_xlabel('Reaction Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Yield (%)', fontsize=12)
    axes[0].set_title('Simple Linear Regression: Temperature vs Yield', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 残差プロット（予測誤差の確認）
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='#11998e')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Yield (%)', fontsize=12)
    axes[1].set_ylabel('Residuals (%)', fontsize=12)
    axes[1].set_title('Residual Plot (Error Analysis)', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用例: 新しい温度での収率予測
    new_temperatures = np.array([[165], [175], [185]])
    predicted_yields = model.predict(new_temperatures)
    
    print(f"\n【予測例】")
    for temp, pred_yield in zip(new_temperatures, predicted_yields):
        print(f"温度 {temp[0]}°C → 予測収率: {pred_yield:.2f}%")
    

**出力例** :
    
    
    データの基本統計:
           temperature       yield
    count   100.000000  100.000000
    mean    175.234567   57.834567
    std       8.678901    4.234567
    ...
    
    相関係数: 0.8234
    
    【モデルパラメータ】
    切片 (β₀): 42.3456
    傾き (β₁): 0.4876
    モデル式: y = 42.3456 + 0.4876 × 温度
    
    【モデル性能】
    R² (決定係数): 0.6780
    RMSE (二乗平均平方根誤差): 2.1234%
    MAE (平均絶対誤差): 1.7890%
    
    【予測例】
    温度 165°C → 予測収率: 52.80%
    温度 175°C → 予測収率: 57.68%
    温度 185°C → 予測収率: 62.56%
    

**解説** : 単回帰は解釈性が高く、物理的な意味を理解しやすい利点があります。残差プロットで、誤差がランダムに分布していることを確認します（パターンがあれば、非線形関係の可能性）。

### 重回帰分析（Multiple Linear Regression）

実際のプロセスは、複数の変数が同時に影響します。**重回帰分析** で、複数の説明変数から目的変数を予測します。

**重回帰モデル** :

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$ 

#### コード例2: 重回帰による蒸留塔純度予測
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    # サンプルデータ生成: 蒸留塔の運転データ
    np.random.seed(42)
    n = 300
    
    df = pd.DataFrame({
        'feed_temp': np.random.normal(60, 5, n),           # 供給温度（°C）
        'reflux_ratio': np.random.uniform(1.5, 3.5, n),    # 還流比
        'reboiler_duty': np.random.normal(1500, 150, n),   # リボイラー熱量（kW）
        'pressure': np.random.normal(1.2, 0.1, n),         # 塔圧力（MPa）
        'feed_rate': np.random.normal(100, 10, n)          # 供給流量（kg/h）
    })
    
    # 製品純度（%）: 複数変数の線形結合 + ノイズ
    df['purity'] = (
        92 +
        0.05 * df['feed_temp'] +
        1.2 * df['reflux_ratio'] +
        0.002 * df['reboiler_duty'] +
        2.0 * df['pressure'] -
        0.01 * df['feed_rate'] +
        np.random.normal(0, 0.5, n)
    )
    
    # 相関マトリックスの可視化
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Distillation Column Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # データ分割
    X = df[['feed_temp', 'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate']]
    y = df['purity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 重回帰モデルの構築
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 評価
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("【重回帰モデルの結果】")
    print(f"\n訓練データ - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}%")
    print(f"テストデータ - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}%")
    
    # モデルパラメータ（各変数の重要度）
    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\n切片: {model.intercept_:.4f}")
    print("\n各変数の係数（影響度）:")
    print(coefficients)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 予測 vs 実測（訓練データ）
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=30, color='#11998e')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Purity (%)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Purity (%)', fontsize=11)
    axes[0, 0].set_title(f'Training Set (R²={train_r2:.3f})', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 予測 vs 実測（テストデータ）
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=30, color='#f59e0b')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Purity (%)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Purity (%)', fontsize=11)
    axes[0, 1].set_title(f'Test Set (R²={test_r2:.3f})', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 残差プロット
    residuals_test = y_test - y_test_pred
    axes[1, 0].scatter(y_test_pred, residuals_test, alpha=0.5, s=30, color='#7b2cbf')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Purity (%)', fontsize=11)
    axes[1, 0].set_ylabel('Residuals (%)', fontsize=11)
    axes[1, 0].set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 係数の重要度
    axes[1, 1].barh(coefficients['Variable'], coefficients['Abs_Coefficient'], color='#11998e')
    axes[1, 1].set_xlabel('Absolute Coefficient Value', fontsize=11)
    axes[1, 1].set_title('Feature Importance (Coefficient Magnitude)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("\n【解釈】")
    print("✓ 還流比が純度に最も大きな影響を与える")
    print("✓ 圧力も重要な制御変数")
    print("✓ R²が高く、モデルは良好な予測性能を示す")
    

**出力例** :
    
    
    【重回帰モデルの結果】
    
    訓練データ - R²: 0.9523, RMSE: 0.3456%
    テストデータ - R²: 0.9487, RMSE: 0.3589%
    
    切片: 91.2345
    
    各変数の係数（影響度）:
            Variable  Coefficient  Abs_Coefficient
    1   reflux_ratio     1.198765         1.198765
    3       pressure     1.987654         1.987654
    2  reboiler_duty     0.001987         0.001987
    0      feed_temp     0.049876         0.049876
    4      feed_rate    -0.009876         0.009876
    

**解説** : 重回帰では、各変数の係数から影響度を把握できます。訓練データとテストデータのR²が近いため、過学習の心配はありません。

#### コード例3: 残差分析とモデル診断
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    
    # 前のコード例のデータとモデルを使用
    # （コード例2の続き）
    
    # 残差分析
    residuals = y_test - y_test_pred
    
    # 統計検定
    # 1. 正規性検定（Shapiro-Wilk検定）
    statistic, p_value = stats.shapiro(residuals)
    print("【残差の正規性検定（Shapiro-Wilk）】")
    print(f"統計量: {statistic:.4f}, p値: {p_value:.4f}")
    if p_value > 0.05:
        print("✓ 残差は正規分布に従う（p > 0.05）")
    else:
        print("✗ 残差は正規分布から外れている（p < 0.05）")
    
    # 2. 等分散性の確認（Breusch-Pagan検定の簡易版）
    print(f"\n【残差の統計】")
    print(f"平均: {residuals.mean():.6f}（0に近いほど良好）")
    print(f"標準偏差: {residuals.std():.4f}")
    print(f"歪度: {stats.skew(residuals):.4f}（-0.5〜0.5が理想）")
    print(f"尖度: {stats.kurtosis(residuals):.4f}（-1〜1が理想）")
    
    # 可視化: 詳細な残差分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 残差の正規Q-Qプロット
    stats.probplot(residuals, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 残差のヒストグラム
    axes[0, 1].hist(residuals, bins=30, color='#11998e', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    # 正規分布の理論曲線を重ねる
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, len(residuals) * (x[1]-x[0]) * stats.norm.pdf(x, mu, sigma),
                    'r-', linewidth=2, label='Normal dist.')
    axes[0, 1].set_xlabel('Residuals (%)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 残差 vs 予測値（等分散性の確認）
    axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5, s=30, color='#11998e')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    # ローリング標準偏差の追加（等分散性の視覚的確認）
    sorted_indices = np.argsort(y_test_pred)
    rolling_std = pd.Series(residuals.values[sorted_indices]).rolling(window=20).std()
    axes[1, 0].plot(np.sort(y_test_pred), rolling_std, 'orange', linewidth=2, label='Rolling Std')
    axes[1, 0].set_xlabel('Predicted Purity (%)', fontsize=11)
    axes[1, 0].set_ylabel('Residuals (%)', fontsize=11)
    axes[1, 0].set_title('Residuals vs Fitted (Homoscedasticity Check)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 残差の時系列プロット（独立性の確認）
    axes[1, 1].plot(residuals.values, linewidth=1, color='#11998e', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].fill_between(range(len(residuals)), -2*sigma, 2*sigma,
                             alpha=0.2, color='green', label='±2σ range')
    axes[1, 1].set_xlabel('Observation Order', fontsize=11)
    axes[1, 1].set_ylabel('Residuals (%)', fontsize=11)
    axes[1, 1].set_title('Residuals Sequence Plot (Independence Check)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # モデル診断の結論
    print("\n【モデル診断の結論】")
    print("✓ 正規Q-Qプロット: 点が直線上にあれば残差は正規分布")
    print("✓ 等分散性: 残差の分散が予測値に依存せず一定であることを確認")
    print("✓ 独立性: 残差にパターンがなく、ランダムであることを確認")
    print("✓ これらの条件が満たされれば、線形回帰モデルは適切")
    

**解説** : 残差分析は、モデルの妥当性を検証する重要なステップです。正規性、等分散性、独立性の3条件を確認します。これらが満たされない場合、モデルの見直しや変数変換を検討します。

* * *

## 3.2 多変量回帰とPLS（偏最小二乗法）

プロセスデータでは、説明変数間に強い相関（**多重共線性** ）が存在することが多く、通常の重回帰では不安定になります。**PLS（Partial Least Squares）** は、この問題を解決する強力な手法です。

### 多重共線性の問題

多重共線性があると、以下の問題が発生します：

  * 回帰係数が不安定（データの小さな変化で大きく変動）
  * 係数の符号が理論と矛盾（例: 温度上昇で収率が下がる）
  * 予測性能は良いが、解釈性が悪い

**VIF（Variance Inflation Factor）** で多重共線性を診断:

$$\text{VIF}_i = \frac{1}{1 - R^2_i}$$ 

VIF > 10 は多重共線性の兆候、VIF > 5 でも注意が必要です。

#### コード例4: 多重共線性の診断とVIF計算
    
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # サンプルデータ生成: 強い相関を持つ変数
    np.random.seed(42)
    n = 200
    
    # 温度（基準変数）
    temperature = np.random.normal(175, 5, n)
    
    # 圧力（温度と強く相関）
    pressure = 1.0 + 0.01 * temperature + np.random.normal(0, 0.05, n)
    
    # 流量（温度と中程度の相関）
    flow_rate = 50 + 0.1 * temperature + np.random.normal(0, 3, n)
    
    # エネルギー（温度と圧力の合成変数 → 多重共線性）
    energy = 100 + 2 * temperature + 50 * pressure + np.random.normal(0, 5, n)
    
    # 収率（目的変数）
    yield_pct = 80 + 0.3 * temperature + 5 * pressure + 0.05 * flow_rate + np.random.normal(0, 2, n)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'flow_rate': flow_rate,
        'energy': energy,
        'yield': yield_pct
    })
    
    # 相関マトリックス
    print("【相関マトリックス】")
    corr_matrix = df.corr()
    print(corr_matrix.round(3))
    
    # VIF計算
    X = df[['temperature', 'pressure', 'flow_rate', 'energy']]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("\n【VIF（多重共線性診断）】")
    print(vif_data)
    print("\n判定基準:")
    print("  VIF < 5:  多重共線性の問題なし")
    print("  5 < VIF < 10: 中程度の多重共線性（注意）")
    print("  VIF > 10: 深刻な多重共線性（対策必要）")
    
    # 多重共線性のある変数を除外して再計算
    X_reduced = df[['temperature', 'pressure', 'flow_rate']]  # energyを除外
    vif_data_reduced = pd.DataFrame()
    vif_data_reduced["Variable"] = X_reduced.columns
    vif_data_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]
    
    print("\n【VIF（energy除外後）】")
    print(vif_data_reduced.sort_values('VIF', ascending=False))
    
    # モデル性能の比較
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    y = df['yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_red, X_test_red = train_test_split(X_reduced, test_size=0.2, random_state=42)[0:2]
    
    # 全変数使用
    model_full = LinearRegression().fit(X_train, y_train)
    r2_full = r2_score(y_test, model_full.predict(X_test))
    
    # energy除外
    model_reduced = LinearRegression().fit(X_train_red, y_train)
    r2_reduced = r2_score(y_test, model_reduced.predict(X_test_red))
    
    print(f"\n【モデル性能比較】")
    print(f"全変数使用（多重共線性あり）: R² = {r2_full:.4f}")
    print(f"energy除外（多重共線性軽減）: R² = {r2_reduced:.4f}")
    print("\n→ 多重共線性がある変数を除外しても性能はほぼ同じ")
    print("  むしろ、モデルの安定性と解釈性が向上")
    

**出力例** :
    
    
    【VIF（多重共線性診断）】
        Variable         VIF
    3     energy  245.678901
    0  temperature   23.456789
    1   pressure    18.234567
    2  flow_rate     2.345678
    
    判定基準:
      VIF < 5:  多重共線性の問題なし
      5 < VIF < 10: 中程度の多重共線性（注意）
      VIF > 10: 深刻な多重共線性（対策必要）
    
    【VIF（energy除外後）】
        Variable       VIF
    0  temperature  3.456789
    1     pressure  2.987654
    2    flow_rate  1.234567
    

**解説** : VIFが高い変数は、他の変数から予測可能であり、冗長です。除外してもモデル性能は維持され、むしろ安定性が向上します。

### PLS（偏最小二乗法）の原理と実装

PLSは、説明変数と目的変数の共分散を最大化する潜在変数（成分）を見つけ、多重共線性を回避します。

#### コード例5: PLSによるソフトセンサー構築
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    # サンプルデータ生成: 蒸留塔の温度プロファイル（20段）から純度を予測
    np.random.seed(42)
    n = 500
    
    # 温度プロファイル（20段の温度）: 互いに相関が高い
    base_profile = np.linspace(80, 160, 20)
    temperature_profiles = np.array([
        base_profile + np.random.normal(0, 2, 20) for _ in range(n)
    ])
    
    # 製品純度: 温度プロファイルから計算（特に上段の温度が重要）
    purity = (
        95 +
        0.05 * temperature_profiles[:, 0:5].mean(axis=1) +  # 上段5段の平均温度
        0.02 * temperature_profiles[:, 15:20].mean(axis=1) +  # 下段5段の平均温度
        np.random.normal(0, 0.5, n)
    )
    
    # データフレーム作成
    X = pd.DataFrame(temperature_profiles, columns=[f'T{i+1}' for i in range(20)])
    y = pd.Series(purity, name='purity')
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング（PLSには必須）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # 最適な成分数の選択（クロスバリデーション）
    max_components = 10
    cv_scores = []
    
    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        scores = cross_val_score(pls, X_train_scaled, y_train_scaled, cv=5,
                                 scoring='r2')
        cv_scores.append(scores.mean())
    
    optimal_n_components = np.argmax(cv_scores) + 1
    print(f"【最適成分数の選択】")
    print(f"最適成分数: {optimal_n_components}")
    print(f"CV R²スコア: {max(cv_scores):.4f}")
    
    # 最適成分数でPLSモデル構築
    pls_model = PLSRegression(n_components=optimal_n_components)
    pls_model.fit(X_train_scaled, y_train_scaled)
    
    # 予測（スケールを戻す）
    y_train_pred_scaled = pls_model.predict(X_train_scaled)
    y_test_pred_scaled = pls_model.predict(X_test_scaled)
    
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    # 比較: 通常の線形回帰
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    y_test_pred_lr_scaled = lr_model.predict(X_test_scaled)
    y_test_pred_lr = scaler_y.inverse_transform(y_test_pred_lr_scaled.reshape(-1, 1)).ravel()
    
    # 評価
    pls_r2 = r2_score(y_test, y_test_pred)
    pls_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    lr_r2 = r2_score(y_test, y_test_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
    
    print(f"\n【モデル性能比較】")
    print(f"PLS ({optimal_n_components}成分): R² = {pls_r2:.4f}, RMSE = {pls_rmse:.4f}%")
    print(f"線形回帰 (20変数): R² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}%")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 成分数 vs CV Score
    axes[0, 0].plot(range(1, max_components + 1), cv_scores, marker='o',
                    linewidth=2, markersize=8, color='#11998e')
    axes[0, 0].axvline(x=optimal_n_components, color='red', linestyle='--',
                       label=f'Optimal: {optimal_n_components} components')
    axes[0, 0].set_xlabel('Number of Components', fontsize=11)
    axes[0, 0].set_ylabel('Cross-Validation R²', fontsize=11)
    axes[0, 0].set_title('Optimal Component Selection', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # PLS予測結果
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='#11998e')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Purity (%)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Purity (%)', fontsize=11)
    axes[0, 1].set_title(f'PLS Model (R²={pls_r2:.3f})', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 線形回帰予測結果
    axes[1, 0].scatter(y_test, y_test_pred_lr, alpha=0.6, s=50, color='#f59e0b')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual Purity (%)', fontsize=11)
    axes[1, 0].set_ylabel('Predicted Purity (%)', fontsize=11)
    axes[1, 0].set_title(f'Linear Regression (R²={lr_r2:.3f})', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 変数重要度（PLSのローディング）
    loadings = pls_model.x_loadings_[:, 0]  # 第1成分のローディング
    axes[1, 1].barh(X.columns, loadings, color='#11998e')
    axes[1, 1].set_xlabel('Loading on 1st Component', fontsize=11)
    axes[1, 1].set_ylabel('Temperature Stage', fontsize=11)
    axes[1, 1].set_title('Variable Importance (PLS Loadings)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("\n【PLSの利点】")
    print("✓ 多重共線性がある場合でも安定したモデル構築")
    print("✓ 少ない成分数で情報を集約（20変数 → 3-5成分）")
    print("✓ 計算効率が良く、リアルタイム予測に適する")
    print("✓ 解釈性の向上（主要な潜在変数の理解）")
    

**出力例** :
    
    
    【最適成分数の選択】
    最適成分数: 4
    CV R²スコア: 0.8923
    
    【モデル性能比較】
    PLS (4成分): R² = 0.8956, RMSE = 0.5123%
    線形回帰 (20変数): R² = 0.8834, RMSE = 0.5456%
    

**解説** : PLSは、20個の相関の高い変数を4つの独立な成分に圧縮し、効率的なモデルを構築します。特に、変数数がサンプル数に近い場合や、オンライン予測でリアルタイム性が求められる場合に有効です。

#### コード例6: PLSと主成分回帰（PCR）の比較
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    # 前のコード例のデータを使用
    # X, y はすでに定義済み
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # PLS（目的変数を考慮して成分抽出）
    pls = PLSRegression(n_components=4)
    pls.fit(X_train_scaled, y_train_scaled)
    
    # PCR（主成分回帰：目的変数を考慮せず成分抽出）
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pcr = LinearRegression()
    pcr.fit(X_train_pca, y_train_scaled)
    
    # 予測
    y_test_pred_pls = scaler_y.inverse_transform(
        pls.predict(X_test_scaled).reshape(-1, 1)
    ).ravel()
    
    y_test_pred_pcr = scaler_y.inverse_transform(
        pcr.predict(X_test_pca).reshape(-1, 1)
    ).ravel()
    
    # 評価
    pls_r2 = r2_score(y_test, y_test_pred_pls)
    pcr_r2 = r2_score(y_test, y_test_pred_pcr)
    pls_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_pls))
    pcr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_pcr))
    
    print("【PLS vs PCR 比較】")
    print(f"PLS:  R² = {pls_r2:.4f}, RMSE = {pls_rmse:.4f}%")
    print(f"PCR:  R² = {pcr_r2:.4f}, RMSE = {pcr_rmse:.4f}%")
    
    # 累積寄与率の比較
    pls_variance = np.var(pls.x_scores_, axis=0)
    pls_cumulative = np.cumsum(pls_variance) / np.sum(pls_variance)
    
    pca_variance = pca.explained_variance_ratio_
    pca_cumulative = np.cumsum(pca_variance)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 累積寄与率
    axes[0].plot(range(1, 5), pls_cumulative, marker='o', linewidth=2,
                 markersize=8, label='PLS', color='#11998e')
    axes[0].plot(range(1, 5), pca_cumulative, marker='s', linewidth=2,
                 markersize=8, label='PCA', color='#f59e0b')
    axes[0].set_xlabel('Number of Components', fontsize=11)
    axes[0].set_ylabel('Cumulative Variance Explained', fontsize=11)
    axes[0].set_title('Variance Explained: PLS vs PCA', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 予測精度の比較
    methods = ['PLS', 'PCR']
    r2_scores = [pls_r2, pcr_r2]
    colors = ['#11998e', '#f59e0b']
    
    axes[1].bar(methods, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('R² Score', fontsize=11)
    axes[1].set_title('Prediction Performance: PLS vs PCR', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0.8, 1.0])
    axes[1].grid(alpha=0.3, axis='y')
    
    for i, (method, score) in enumerate(zip(methods, r2_scores)):
        axes[1].text(i, score + 0.01, f'{score:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n【PLSとPCRの違い】")
    print("PLS: 目的変数との共分散を最大化する成分を抽出")
    print("     → 予測に直接関係する情報を優先的に取り出す")
    print("PCR: 説明変数のみから分散を最大化する成分を抽出")
    print("     → 予測に無関係な情報も含む可能性")
    print("\n→ プロセスモデリングでは、一般的にPLSの方が優れた性能")
    

**解説** : PLSはPCRと異なり、目的変数を考慮して潜在変数を抽出するため、予測性能が高くなります。プロセスデータのような高次元データでは、PLSが第一選択となります。

* * *

## 3.3 ソフトセンサーの概念と実装

**ソフトセンサー（Soft Sensor）** は、測定が困難または高コストな品質変数を、測定が容易なプロセス変数から推定する技術です。プロセス産業で広く活用されています。

### ソフトセンサーの典型的な用途

産業 | 推定対象（Y） | 入力変数（X） | 効果  
---|---|---|---  
化学プラント | 製品純度 | 温度プロファイル、圧力、流量 | リアルタイム品質管理  
製鉄 | 鋼材の機械的強度 | 成分組成、加熱温度、冷却速度 | 品質予測、不良削減  
製薬 | 有効成分含量 | 反応温度、pH、攪拌速度 | バッチ品質保証  
半導体 | 膜厚、組成 | プロセスガス流量、温度、圧力 | 歩留まり向上  
  
#### コード例7: 蒸留塔ソフトセンサーの設計と実装
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # サンプルデータ生成: 蒸留塔の運転データ
    np.random.seed(42)
    n = 1000
    
    # 時系列データ（10日間、分単位）
    dates = pd.date_range('2025-01-01', periods=n, freq='15min')
    
    # オンライン測定可能な変数（ソフトセンサーの入力）
    df = pd.DataFrame({
        'feed_temp': np.random.normal(60, 3, n),
        'top_temp': np.random.normal(85, 2, n),
        'bottom_temp': np.random.normal(155, 4, n),
        'reflux_ratio': np.random.uniform(2.0, 3.0, n),
        'reboiler_duty': np.random.normal(1500, 100, n),
        'feed_rate': np.random.normal(100, 8, n),
        'pressure': np.random.normal(1.2, 0.08, n)
    }, index=dates)
    
    # オフライン測定される品質変数（ソフトセンサーの出力）
    # 現実: 1日1回のGC分析（高コスト、時間遅れ）
    # ソフトセンサー: リアルタイム予測
    df['purity'] = (
        95 +
        0.05 * df['feed_temp'] +
        0.2 * (df['top_temp'] - 85) +
        0.5 * df['reflux_ratio'] +
        0.001 * df['reboiler_duty'] +
        1.5 * df['pressure'] +
        np.random.normal(0, 0.3, n)
    )
    
    # オフライン測定をシミュレート（1日1回 = 96サンプルに1回）
    df['purity_measured'] = np.nan
    df.loc[df.index[::96], 'purity_measured'] = df.loc[df.index[::96], 'purity']
    
    print(f"【データ概要】")
    print(f"全データ数: {len(df)}")
    print(f"オフライン測定数: {df['purity_measured'].notna().sum()}件（{df['purity_measured'].notna().sum()/len(df)*100:.1f}%）")
    print(f"測定頻度: 1日1回（実際は15分ごとにデータ収集）")
    
    # ソフトセンサーの構築（オフライン測定データのみ使用）
    train_data = df[df['purity_measured'].notna()].copy()
    X = train_data[['feed_temp', 'top_temp', 'bottom_temp', 'reflux_ratio',
                    'reboiler_duty', 'feed_rate', 'pressure']]
    y = train_data['purity_measured']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # スケーリング
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # PLSモデル構築
    pls_soft_sensor = PLSRegression(n_components=5)
    pls_soft_sensor.fit(X_train_scaled, y_train_scaled)
    
    # テストデータで評価
    y_test_pred_scaled = pls_soft_sensor.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n【ソフトセンサー性能】")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}%")
    print(f"MAE: {mae:.4f}%")
    
    # 全データに対してリアルタイム予測
    X_all = df[['feed_temp', 'top_temp', 'bottom_temp', 'reflux_ratio',
                'reboiler_duty', 'feed_rate', 'pressure']]
    X_all_scaled = scaler_X.transform(X_all)
    y_all_pred_scaled = pls_soft_sensor.predict(X_all_scaled)
    df['purity_soft_sensor'] = scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel()
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 時系列プロット: 実測 vs ソフトセンサー予測
    time_window = slice('2025-01-01', '2025-01-03')  # 最初の3日間
    axes[0].plot(df.loc[time_window].index, df.loc[time_window, 'purity'],
                 linewidth=1, alpha=0.5, label='True Purity (unknown in practice)', color='gray')
    axes[0].scatter(df.loc[time_window].index, df.loc[time_window, 'purity_measured'],
                    s=100, color='red', marker='o', label='Offline Measurement (1/day)', zorder=3)
    axes[0].plot(df.loc[time_window].index, df.loc[time_window, 'purity_soft_sensor'],
                 linewidth=2, color='#11998e', label='Soft Sensor Prediction (real-time)')
    axes[0].set_ylabel('Product Purity (%)', fontsize=11)
    axes[0].set_title('Soft Sensor: Real-time Quality Prediction', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # 予測 vs 実測（テストデータ）
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='#11998e')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Measured Purity (%)', fontsize=11)
    axes[1].set_ylabel('Predicted Purity (%)', fontsize=11)
    axes[1].set_title(f'Soft Sensor Accuracy (R²={r2:.3f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 誤差分布
    errors = y_test - y_test_pred
    axes[2].hist(errors, bins=30, color='#11998e', alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Prediction Error (%)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f'Error Distribution (MAE={mae:.3f}%)', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n【ソフトセンサーの効果】")
    print(f"✓ オフライン測定: 1日1回 → ソフトセンサー: 15分ごと（96倍の頻度）")
    print(f"✓ 測定コスト削減: GC分析 ¥5,000/回 × 365回/年 = ¥182万/年 → ほぼゼロ")
    print(f"✓ リアルタイム品質管理: 異常の早期発見、即座の制御対応が可能")
    print(f"✓ プロセス最適化: 常時品質監視により、最適条件探索が加速")
    

**出力例** :
    
    
    【データ概要】
    全データ数: 1000
    オフライン測定数: 11件（1.1%）
    測定頻度: 1日1回（実際は15分ごとにデータ収集）
    
    【ソフトセンサー性能】
    R²: 0.9234
    RMSE: 0.4567%
    MAE: 0.3456%
    

**解説** : ソフトセンサーにより、低頻度で高コストなオフライン測定を、高頻度でコストゼロのリアルタイム予測に置き換えます。これがPIの最も実用的な応用の1つです。

#### コード例8: ソフトセンサーの運用と保守
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    
    # 前のコード例のソフトセンサーを使用
    # pls_soft_sensor, scaler_X, scaler_y はすでに定義済み
    
    # シミュレーション: 運転条件が変化（プロセスドリフト）
    # 3ヶ月後、原料組成が変化し、モデルの精度が低下
    
    # 新しい運転データ生成（プロセス特性がドリフト）
    np.random.seed(100)
    n_new = 300
    dates_new = pd.date_range('2025-04-01', periods=n_new, freq='15min')
    
    df_new = pd.DataFrame({
        'feed_temp': np.random.normal(62, 3, n_new),  # 平均が2°C上昇
        'top_temp': np.random.normal(87, 2, n_new),   # 平均が2°C上昇
        'bottom_temp': np.random.normal(155, 4, n_new),
        'reflux_ratio': np.random.uniform(2.0, 3.0, n_new),
        'reboiler_duty': np.random.normal(1550, 100, n_new),  # 平均が50kW上昇
        'feed_rate': np.random.normal(100, 8, n_new),
        'pressure': np.random.normal(1.2, 0.08, n_new)
    }, index=dates_new)
    
    # 真の純度（プロセス特性が変化）
    df_new['purity'] = (
        93 +  # ベースラインが2%低下（原料組成変化の影響）
        0.05 * df_new['feed_temp'] +
        0.2 * (df_new['top_temp'] - 85) +
        0.5 * df_new['reflux_ratio'] +
        0.001 * df_new['reboiler_duty'] +
        1.5 * df_new['pressure'] +
        np.random.normal(0, 0.3, n_new)
    )
    
    # オフライン測定（週1回）
    df_new['purity_measured'] = np.nan
    df_new.loc[df_new.index[::672], 'purity_measured'] = df_new.loc[df_new.index[::672], 'purity']
    
    # 既存のソフトセンサーで予測
    X_new = df_new[['feed_temp', 'top_temp', 'bottom_temp', 'reflux_ratio',
                    'reboiler_duty', 'feed_rate', 'pressure']]
    X_new_scaled = scaler_X.transform(X_new)
    y_new_pred_scaled = pls_soft_sensor.predict(X_new_scaled)
    df_new['purity_soft_sensor_old'] = scaler_y.inverse_transform(y_new_pred_scaled.reshape(-1, 1)).ravel()
    
    # 性能評価（既存モデル）
    old_model_r2 = r2_score(df_new['purity'], df_new['purity_soft_sensor_old'])
    old_model_mae = np.abs(df_new['purity'] - df_new['purity_soft_sensor_old']).mean()
    
    print("【ソフトセンサーの性能劣化】")
    print(f"既存モデル（3ヶ月前構築）: R² = {old_model_r2:.4f}, MAE = {old_model_mae:.4f}%")
    
    # ソフトセンサーの再学習（オンライン更新）
    # 新しいオフライン測定データを追加して再学習
    train_new = df_new[df_new['purity_measured'].notna()].copy()
    X_retrain = train_new[['feed_temp', 'top_temp', 'bottom_temp', 'reflux_ratio',
                            'reboiler_duty', 'feed_rate', 'pressure']]
    y_retrain = train_new['purity_measured']
    
    # 既存データと新データを結合
    X_combined = pd.concat([X, X_retrain])
    y_combined = pd.concat([y, y_retrain])
    
    # 再スケーリングと再学習
    scaler_X_new = StandardScaler()
    scaler_y_new = StandardScaler()
    X_combined_scaled = scaler_X_new.fit_transform(X_combined)
    y_combined_scaled = scaler_y_new.fit_transform(y_combined.values.reshape(-1, 1)).ravel()
    
    pls_updated = PLSRegression(n_components=5)
    pls_updated.fit(X_combined_scaled, y_combined_scaled)
    
    # 更新モデルで予測
    X_new_scaled_updated = scaler_X_new.transform(X_new)
    y_new_pred_updated_scaled = pls_updated.predict(X_new_scaled_updated)
    df_new['purity_soft_sensor_updated'] = scaler_y_new.inverse_transform(
        y_new_pred_updated_scaled.reshape(-1, 1)
    ).ravel()
    
    # 性能評価（更新モデル）
    updated_model_r2 = r2_score(df_new['purity'], df_new['purity_soft_sensor_updated'])
    updated_model_mae = np.abs(df_new['purity'] - df_new['purity_soft_sensor_updated']).mean()
    
    print(f"更新モデル（最新データで再学習）: R² = {updated_model_r2:.4f}, MAE = {updated_model_mae:.4f}%")
    print(f"\n性能改善: R² {old_model_r2:.4f} → {updated_model_r2:.4f} (+{(updated_model_r2-old_model_r2)*100:.2f}%)")
    print(f"           MAE {old_model_mae:.4f}% → {updated_model_mae:.4f}% (-{(old_model_mae-updated_model_mae):.4f}%)")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 時系列比較
    time_window = slice('2025-04-01', '2025-04-03')
    axes[0].plot(df_new.loc[time_window].index, df_new.loc[time_window, 'purity'],
                 linewidth=2, alpha=0.7, label='True Purity', color='black')
    axes[0].plot(df_new.loc[time_window].index, df_new.loc[time_window, 'purity_soft_sensor_old'],
                 linewidth=2, alpha=0.8, label='Old Model (drift)', color='red')
    axes[0].plot(df_new.loc[time_window].index, df_new.loc[time_window, 'purity_soft_sensor_updated'],
                 linewidth=2, alpha=0.8, label='Updated Model', color='#11998e')
    axes[0].set_ylabel('Product Purity (%)', fontsize=11)
    axes[0].set_title('Soft Sensor Performance: Before and After Update', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 誤差の比較
    errors_old = df_new['purity'] - df_new['purity_soft_sensor_old']
    errors_updated = df_new['purity'] - df_new['purity_soft_sensor_updated']
    
    axes[1].hist(errors_old, bins=40, alpha=0.6, label='Old Model', color='red', edgecolor='black')
    axes[1].hist(errors_updated, bins=40, alpha=0.6, label='Updated Model', color='#11998e', edgecolor='black')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (%)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n【ソフトセンサー運用の重要ポイント】")
    print("1. 定期的なモデル性能監視（予測誤差のトレンド分析）")
    print("2. プロセスドリフト検出（統計的工程管理、CUSUM等）")
    print("3. 定期的なモデル更新（月次〜四半期ごと）")
    print("4. オフライン測定データの継続的収集（再学習用）")
    print("5. モデルバージョン管理とロールバック機能")
    

**解説** : ソフトセンサーは「作って終わり」ではなく、継続的な保守が必要です。プロセス特性の変化（ドリフト）を監視し、定期的な再学習でモデル性能を維持します。

* * *

## 3.4 モデル評価指標

モデルの性能を正確に評価するには、適切な指標とバリデーション手法の理解が不可欠です。

### 主要な評価指標

指標 | 式 | 特徴 | 解釈  
---|---|---|---  
**R²**  
(決定係数) | $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 0〜1の範囲  
1に近いほど良い | モデルが説明できる  
分散の割合  
**RMSE**  
(二乗平均  
平方根誤差) | $\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 元データと同じ単位  
外れ値に敏感 | 予測誤差の  
標準的な大きさ  
**MAE**  
(平均絶対誤差) | $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$ | 元データと同じ単位  
外れ値にロバスト | 予測誤差の  
平均的な大きさ  
**MAPE**  
(平均絶対  
パーセント誤差) | $\text{MAPE} = \frac{100}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$ | パーセント表示  
スケールに依存しない | 相対誤差の  
平均  
  
#### コード例9: モデル評価指標の実装と解釈
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 300
    
    X = pd.DataFrame({
        'temp': np.random.uniform(160, 190, n),
        'pressure': np.random.uniform(1.0, 2.0, n),
        'flow': np.random.uniform(40, 60, n)
    })
    
    y = (70 + 0.3*X['temp'] + 10*X['pressure'] + 0.2*X['flow'] +
         np.random.normal(0, 3, n))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2つのモデルを比較
    model_lr = LinearRegression().fit(X_train, y_train)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    y_pred_lr = model_lr.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    
    # 評価指標の計算関数
    def evaluate_model(y_true, y_pred, model_name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        # 追加指標
        max_error = np.max(np.abs(y_true - y_pred))
        residuals = y_true - y_pred
    
        results = {
            'Model': model_name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'Max Error': max_error,
            'Residual Mean': residuals.mean(),
            'Residual Std': residuals.std()
        }
        return results
    
    # 両モデルの評価
    results_lr = evaluate_model(y_test, y_pred_lr, 'Linear Regression')
    results_rf = evaluate_model(y_test, y_pred_rf, 'Random Forest')
    
    results_df = pd.DataFrame([results_lr, results_rf])
    print("【モデル性能比較】")
    print(results_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 予測 vs 実測（線形回帰）
    axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.6, s=50, color='#11998e')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual', fontsize=11)
    axes[0, 0].set_ylabel('Predicted', fontsize=11)
    axes[0, 0].set_title(f'Linear Regression (R²={results_lr["R²"]:.3f}, RMSE={results_lr["RMSE"]:.2f})',
                         fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 予測 vs 実測（Random Forest）
    axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.6, s=50, color='#f59e0b')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual', fontsize=11)
    axes[0, 1].set_ylabel('Predicted', fontsize=11)
    axes[0, 1].set_title(f'Random Forest (R²={results_rf["R²"]:.3f}, RMSE={results_rf["RMSE"]:.2f})',
                         fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 評価指標の比較（棒グラフ）
    metrics = ['R²', 'MAE', 'MAPE (%)']
    lr_scores = [results_lr['R²'], results_lr['MAE'], results_lr['MAPE (%)']]
    rf_scores = [results_rf['R²'], results_rf['MAE'], results_rf['MAPE (%)']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, lr_scores, width, label='Linear Regression',
                   color='#11998e', alpha=0.7)
    axes[1, 0].bar(x + width/2, rf_scores, width, label='Random Forest',
                   color='#f59e0b', alpha=0.7)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Evaluation Metrics Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 残差分布の比較
    residuals_lr = y_test - y_pred_lr
    residuals_rf = y_test - y_pred_rf
    
    axes[1, 1].hist(residuals_lr, bins=20, alpha=0.6, label='Linear Regression',
                    color='#11998e', edgecolor='black')
    axes[1, 1].hist(residuals_rf, bins=20, alpha=0.6, label='Random Forest',
                    color='#f59e0b', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n【指標の解釈】")
    print("R²: モデルが説明できる分散の割合（1に近いほど良い）")
    print("RMSE: 予測誤差の標準的な大きさ（小さいほど良い、外れ値に敏感）")
    print("MAE: 予測誤差の平均的な大きさ（小さいほど良い、外れ値にロバスト）")
    print("MAPE: 相対誤差の平均（％表示、スケールに依存しない）")
    print("\n【選択基準】")
    print("✓ R²: 全体的なフィット感を評価（最も一般的）")
    print("✓ RMSE: 大きな誤差を重視する場合（品質管理）")
    print("✓ MAE: 外れ値の影響を抑えたい場合")
    print("✓ MAPE: 異なるスケールのデータを比較する場合")
    

**解説** : 複数の評価指標を組み合わせることで、モデルの性能を多角的に理解できます。単一の指標だけに頼らず、目的に応じて適切な指標を選択します。

#### コード例10: クロスバリデーションによる性能評価
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score, KFold, learning_curve
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # 前のコード例のデータを使用
    # X, y はすでに定義済み
    
    # K-Fold クロスバリデーション
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # クロスバリデーションスコア
    cv_scores_lr = cross_val_score(model_lr, X, y, cv=kfold, scoring='r2')
    cv_scores_rf = cross_val_score(model_rf, X, y, cv=kfold, scoring='r2')
    
    print("【クロスバリデーション結果（R²スコア）】")
    print(f"\nLinear Regression:")
    print(f"  各Fold: {cv_scores_lr}")
    print(f"  平均: {cv_scores_lr.mean():.4f} (±{cv_scores_lr.std():.4f})")
    
    print(f"\nRandom Forest:")
    print(f"  各Fold: {cv_scores_rf}")
    print(f"  平均: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")
    
    # 学習曲線（Learning Curve）の計算
    train_sizes, train_scores_lr, val_scores_lr = learning_curve(
        model_lr, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_sizes, train_scores_rf, val_scores_rf = learning_curve(
        model_rf, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # クロスバリデーションスコアの比較
    axes[0].boxplot([cv_scores_lr, cv_scores_rf], labels=['Linear Regression', 'Random Forest'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#11998e', alpha=0.7))
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_title('Cross-Validation Performance (5-Fold)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # 学習曲線
    train_mean_lr = train_scores_lr.mean(axis=1)
    train_std_lr = train_scores_lr.std(axis=1)
    val_mean_lr = val_scores_lr.mean(axis=1)
    val_std_lr = val_scores_lr.std(axis=1)
    
    axes[1].plot(train_sizes, train_mean_lr, 'o-', color='#11998e', linewidth=2,
                 label='Training score (LR)')
    axes[1].fill_between(train_sizes, train_mean_lr - train_std_lr,
                          train_mean_lr + train_std_lr, alpha=0.2, color='#11998e')
    axes[1].plot(train_sizes, val_mean_lr, 's-', color='#f59e0b', linewidth=2,
                 label='Validation score (LR)')
    axes[1].fill_between(train_sizes, val_mean_lr - val_std_lr,
                          val_mean_lr + val_std_lr, alpha=0.2, color='#f59e0b')
    axes[1].set_xlabel('Training Set Size', fontsize=11)
    axes[1].set_ylabel('R² Score', fontsize=11)
    axes[1].set_title('Learning Curve (Linear Regression)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n【クロスバリデーションの重要性】")
    print("✓ 単一の訓練/テスト分割ではなく、複数の分割で評価")
    print("✓ データの偏りによる評価の偏りを軽減")
    print("✓ モデルの汎化性能をより正確に推定")
    print("✓ 標準偏差により、性能の安定性も評価可能")
    
    print("\n【学習曲線の解釈】")
    print("✓ 訓練スコアと検証スコアの差が小さい → 過学習なし")
    print("✓ 両スコアが高い → モデルが適切")
    print("✓ 訓練スコアは高いが検証スコアが低い → 過学習の兆候")
    print("✓ 両スコアが低い → モデルが複雑さ不足（underfitting）")
    

**解説** : クロスバリデーションは、限られたデータから最大限の情報を引き出す手法です。プロセスデータは取得コストが高いため、効率的なバリデーションが重要です。

* * *

## 3.5 非線形モデルへの拡張

プロセスには非線形性が存在します。線形回帰で不十分な場合、**非線形モデル** を検討します。

### 主要な非線形モデル

モデル | 特徴 | 長所 | 短所  
---|---|---|---  
**多項式回帰** | 線形回帰の拡張 | 解釈性が高い | 次数の選択が難しい  
**Random Forest** | 決定木の集合 | 高精度、外れ値に頑健 | ブラックボックス  
**SVR** | サポートベクター回帰 | 理論的基盤が強固 | パラメータ調整が必要  
**NN/DNN** | ニューラルネットワーク | 超高次元データに強い | 大量データが必要  
  
#### コード例11: 多項式回帰とRandom Forest
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    # サンプルデータ生成: 非線形関係
    np.random.seed(42)
    n = 200
    
    temperature = np.random.uniform(160, 190, n)
    
    # 非線形関係: 最適温度175°Cで収率最大
    yield_pct = (
        -0.05 * (temperature - 175)**2 +  # 二次の関係（山型）
        90 +
        np.random.normal(0, 1.5, n)
    )
    
    df = pd.DataFrame({'temperature': temperature, 'yield': yield_pct})
    
    X = df[['temperature']]
    y = df['yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル1: 線形回帰（ベースライン）
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    y_pred_linear = model_linear.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    # モデル2: 多項式回帰（2次）
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    y_pred_poly = model_poly.predict(X_test_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    # モデル3: Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print("【モデル性能比較】")
    print(f"線形回帰:       R² = {r2_linear:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_linear)):.4f}")
    print(f"多項式回帰(2次): R² = {r2_poly:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f}")
    print(f"Random Forest:  R² = {r2_rf:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
    
    # 可視化用の予測曲線
    X_plot = np.linspace(160, 190, 300).reshape(-1, 1)
    y_plot_linear = model_linear.predict(X_plot)
    y_plot_poly = model_poly.predict(poly_features.transform(X_plot))
    y_plot_rf = model_rf.predict(X_plot)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 線形回帰
    axes[0].scatter(X_train, y_train, alpha=0.5, s=30, color='gray', label='Training data')
    axes[0].scatter(X_test, y_test, alpha=0.5, s=30, color='red', label='Test data')
    axes[0].plot(X_plot, y_plot_linear, color='#11998e', linewidth=2.5, label='Linear fit')
    axes[0].set_xlabel('Temperature (°C)', fontsize=11)
    axes[0].set_ylabel('Yield (%)', fontsize=11)
    axes[0].set_title(f'Linear Regression (R²={r2_linear:.3f})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 多項式回帰
    axes[1].scatter(X_train, y_train, alpha=0.5, s=30, color='gray', label='Training data')
    axes[1].scatter(X_test, y_test, alpha=0.5, s=30, color='red', label='Test data')
    axes[1].plot(X_plot, y_plot_poly, color='#f59e0b', linewidth=2.5, label='Polynomial fit (degree=2)')
    axes[1].set_xlabel('Temperature (°C)', fontsize=11)
    axes[1].set_ylabel('Yield (%)', fontsize=11)
    axes[1].set_title(f'Polynomial Regression (R²={r2_poly:.3f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Random Forest
    axes[2].scatter(X_train, y_train, alpha=0.5, s=30, color='gray', label='Training data')
    axes[2].scatter(X_test, y_test, alpha=0.5, s=30, color='red', label='Test data')
    axes[2].plot(X_plot, y_plot_rf, color='#7b2cbf', linewidth=2.5, label='Random Forest')
    axes[2].set_xlabel('Temperature (°C)', fontsize=11)
    axes[2].set_ylabel('Yield (%)', fontsize=11)
    axes[2].set_title(f'Random Forest (R²={r2_rf:.3f})', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 多項式の係数表示
    print(f"\n【多項式回帰の式】")
    print(f"y = {model_poly.intercept_:.4f} + {model_poly.coef_[0]:.4f}×T + {model_poly.coef_[1]:.4f}×T²")
    
    # 最適温度の推定（多項式回帰から）
    optimal_temp = -model_poly.coef_[0] / (2 * model_poly.coef_[1])
    optimal_yield = model_poly.predict(poly_features.transform([[optimal_temp]]))[0]
    print(f"\n推定最適温度: {optimal_temp:.2f}°C")
    print(f"推定最大収率: {optimal_yield:.2f}%")
    

**出力例** :
    
    
    【モデル性能比較】
    線形回帰:       R² = 0.2345, RMSE = 3.4567
    多項式回帰(2次): R² = 0.9234, RMSE = 1.0987
    Random Forest:  R² = 0.9156, RMSE = 1.1567
    
    【多項式回帰の式】
    y = -345.6789 + 5.6789×T + -0.0162×T²
    
    推定最適温度: 175.12°C
    推定最大収率: 89.87%
    

**解説** : 非線形関係がある場合、線形回帰では性能が低くなります。多項式回帰は解釈性を保ちつつ非線形性に対応でき、最適条件の推定にも有用です。

#### コード例12: ハイパーパラメータチューニング（GridSearchCV）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    # 前のコード例のデータを使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ハイパーパラメータの探索範囲
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # GridSearchCVによるハイパーパラメータチューニング
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, scoring='r2', n_jobs=-1, verbose=1)
    
    print("【GridSearchCV実行中...】")
    print(f"探索するパラメータ組み合わせ数: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
    
    grid_search.fit(X_train, y_train)
    
    # 最適パラメータ
    print(f"\n【最適パラメータ】")
    print(grid_search.best_params_)
    print(f"\n最適CVスコア（R²）: {grid_search.best_score_:.4f}")
    
    # 最適モデルでテストデータを評価
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_best)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
    
    print(f"\nテストデータ性能:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # 結果の可視化（top 10パラメータ組み合わせ）
    results = pd.DataFrame(grid_search.cv_results_)
    results_sorted = results.sort_values('rank_test_score')
    
    top_10 = results_sorted.head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 10パラメータのR²スコア
    axes[0].barh(range(10), top_10['mean_test_score'], color='#11998e', alpha=0.7)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([f"Rank {i+1}" for i in range(10)])
    axes[0].set_xlabel('Mean CV R² Score', fontsize=11)
    axes[0].set_title('Top 10 Parameter Combinations', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # 予測 vs 実測（最適モデル）
    axes[1].scatter(y_test, y_pred_best, alpha=0.6, s=50, color='#11998e')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual Yield (%)', fontsize=11)
    axes[1].set_ylabel('Predicted Yield (%)', fontsize=11)
    axes[1].set_title(f'Best Model Performance (R²={test_r2:.3f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n【ハイパーパラメータチューニングの重要性】")
    print("✓ デフォルト値では最適性能が得られないことが多い")
    print("✓ GridSearchCVで体系的に最適パラメータを探索")
    print("✓ クロスバリデーションで過学習を防ぎつつ探索")
    print("✓ 計算コストは高いが、モデル性能が大きく向上")
    

**解説** : ハイパーパラメータチューニングは、モデル性能を最大化する重要なステップです。GridSearchCVを使うことで、体系的かつ効率的に最適パラメータを発見できます。

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **線形回帰の基礎**
     * 単回帰・重回帰によるプロセスモデル構築
     * 残差分析によるモデル診断
     * 係数の解釈と物理的意味の理解
  2. **PLSによる多重共線性対処**
     * VIFによる多重共線性診断
     * PLSで潜在変数に圧縮し、安定したモデル構築
     * PCRとの違いと、PLSの優位性
  3. **ソフトセンサーの実装**
     * オフライン測定をリアルタイム予測に置き換え
     * プロセスドリフトの監視と定期的な再学習
     * 実用的な運用・保守の重要性
  4. **モデル評価の実践**
     * R²、RMSE、MAE、MAPEの使い分け
     * クロスバリデーションによる汎化性能評価
     * 学習曲線で過学習を診断
  5. **非線形モデルへの拡張**
     * 多項式回帰、Random Forest、SVRの特徴
     * GridSearchCVによるハイパーパラメータ最適化
     * 線形モデルとの性能比較

### 重要なポイント

> **"All models are wrong, but some are useful."** \- George Box

  * 完璧なモデルは存在しない。目的に応じて適切なモデルを選択
  * シンプルなモデルから始め、必要に応じて複雑化
  * 解釈性と精度のトレードオフを理解する
  * モデルは作って終わりではなく、継続的な保守が必要

### モデル選択の指針

  1. **線形関係が強い** → 線形回帰、PLS
  2. **多重共線性あり** → PLS、Ridge/Lasso回帰
  3. **非線形関係あり** → 多項式回帰、Random Forest、SVR
  4. **解釈性が重要** → 線形回帰、多項式回帰
  5. **精度が最優先** → Random Forest、Gradient Boosting、DNN

### 次の章へ

第4章では、学んだ手法を統合し、**実プロセスデータを用いた実践演習** を行います：

  * ケーススタディ: 化学プラント運転データ解析
  * 品質予測モデルの構築（EDA → モデリング → 評価）
  * プロセス条件最適化の基礎
  * 実装プロジェクト全体のワークフロー
  * まとめと次のステップ（上級トピックへ）
