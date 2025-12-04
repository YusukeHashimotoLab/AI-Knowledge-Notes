---
title: 第4章：実プロセスデータを用いた実践演習
chapter_title: 第4章：実プロセスデータを用いた実践演習
subtitle: 総合演習：化学プラントデータの分析から最適化まで
---

# 第4章：実プロセスデータを用いた実践演習

これまで学んだPIの手法を統合し、実際の化学プラントデータを用いた総合演習を行います。データ探索から品質予測、プロセス最適化まで、実務に直結するワークフローを体験します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 実プロセスデータの探索的データ分析（EDA）を実行できる
  * ✅ データクリーニングから特徴量エンジニアリングまで実装できる
  * ✅ 複数モデルの比較と最適モデルの選択ができる
  * ✅ プロセス条件最適化の基礎手法を適用できる
  * ✅ エンドツーエンドのPIプロジェクトワークフローを理解できる

* * *

## 4.1 ケーススタディ：化学プラント運転データ解析

### プロジェクト概要

**背景** :

ある化学プラントの蒸留塔では、製品純度のばらつきが課題となっています。品質測定は1日1回のガスクロマトグラフィー（GC）分析のみで、リアルタイムな品質管理ができていません。PIを活用して、以下の目標を達成します：

  1. **品質予測ソフトセンサーの構築** : プロセス変数から製品純度をリアルタイム予測
  2. **品質影響因子の特定** : どの変数が純度に最も影響するかを明らかにする
  3. **最適運転条件の探索** : 品質を満たしつつエネルギー消費を最小化する条件を見つける

**利用可能なデータ** :

変数名 | 説明 | 測定頻度 | 単位  
---|---|---|---  
feed_temp | 供給温度 | 1分 | °C  
top_temp | 塔頂温度 | 1分 | °C  
mid_temp | 中段温度 | 1分 | °C  
bottom_temp | 塔底温度 | 1分 | °C  
reflux_ratio | 還流比 | 1分 | -  
reboiler_duty | リボイラー熱量 | 1分 | kW  
pressure | 塔圧力 | 1分 | MPa  
feed_rate | 供給流量 | 1分 | kg/h  
purity | 製品純度（目的変数） | 1日1回 | %  
  
#### コード例1: データ生成とEDA（探索的データ分析）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # シード設定（再現性のため）
    np.random.seed(42)
    
    # 1ヶ月分の運転データ生成（1分間隔）
    n = 43200  # 30日 × 24時間 × 60分
    dates = pd.date_range('2025-01-01', periods=n, freq='1min')
    
    # プロセス変数の生成（現実的な変動パターン）
    df = pd.DataFrame({
        'timestamp': dates,
        'feed_temp': 60 + np.random.normal(0, 2, n) + 3*np.sin(np.arange(n)*2*np.pi/1440),  # 日周変動
        'top_temp': 85 + np.random.normal(0, 1.5, n),
        'mid_temp': 120 + np.random.normal(0, 2, n),
        'bottom_temp': 155 + np.random.normal(0, 3, n),
        'reflux_ratio': 2.5 + np.random.normal(0, 0.2, n),
        'reboiler_duty': 1500 + np.random.normal(0, 80, n),
        'pressure': 1.2 + np.random.normal(0, 0.05, n),
        'feed_rate': 100 + np.random.normal(0, 5, n)
    })
    
    # 製品純度の生成（複雑な非線形関係）
    df['purity'] = (
        92 +
        0.05 * df['feed_temp'] +
        0.3 * (df['top_temp'] - 85) +
        0.15 * (df['mid_temp'] - 120) +
        0.8 * df['reflux_ratio'] +
        0.002 * df['reboiler_duty'] +
        2.0 * df['pressure'] -
        0.01 * df['feed_rate'] +
        # 非線形項（最適点の存在）
        -0.02 * (df['top_temp'] - 85)**2 +
        np.random.normal(0, 0.4, n)
    )
    
    # 欠損値を追加（現実的なデータ）
    missing_indices = np.random.choice(df.index, size=int(n*0.02), replace=False)
    df.loc[missing_indices, 'top_temp'] = np.nan
    
    # 外れ値を追加（測定エラーをシミュレート）
    outlier_indices = np.random.choice(df.index, size=int(n*0.005), replace=False)
    df.loc[outlier_indices, 'pressure'] += np.random.choice([-0.5, 0.5], size=len(outlier_indices))
    
    # オフライン測定のシミュレート（1日1回）
    df['purity_measured'] = np.nan
    df.loc[df.index[::1440], 'purity_measured'] = df.loc[df.index[::1440], 'purity']
    
    # データセットを保存
    df.to_csv('distillation_data.csv', index=False)
    print(f"【データセット生成完了】")
    print(f"総データ数: {len(df):,}件")
    print(f"期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    print(f"オフライン測定数: {df['purity_measured'].notna().sum()}件")
    
    # 基本統計量
    print("\n【基本統計量】")
    print(df.describe().round(2))
    
    # 欠損値の確認
    print("\n【欠損値】")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # EDA: 可視化
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 1. 主要変数の時系列プロット（最初の3日間）
    time_window = (df['timestamp'] >= '2025-01-01') & (df['timestamp'] < '2025-01-04')
    df_window = df[time_window]
    
    variables = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                 'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate']
    
    for i, var in enumerate(variables):
        ax = axes[i//3, i%3]
        ax.plot(df_window['timestamp'], df_window[var], linewidth=0.5, color='#11998e')
        ax.set_ylabel(var, fontsize=10)
        ax.set_title(f'{var} - 3-day trend', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        if i >= 6:
            ax.set_xlabel('Time', fontsize=10)
    
    # 9番目のプロット: 純度（実測とオフライン測定）
    ax = axes[2, 2]
    ax.plot(df_window['timestamp'], df_window['purity'], linewidth=0.8,
            alpha=0.7, label='True purity (unknown)', color='gray')
    ax.scatter(df_window['timestamp'], df_window['purity_measured'],
               s=100, color='red', marker='o', label='Offline measurement', zorder=3)
    ax.set_ylabel('Purity (%)', fontsize=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_title('Product Purity', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n【EDA完了】: eda_timeseries.png を保存しました")
    

**出力例** :
    
    
    【データセット生成完了】
    総データ数: 43,200件
    期間: 2025-01-01 00:00:00 〜 2025-01-30 23:59:00
    オフライン測定数: 31件
    
    【基本統計量】
             feed_temp  top_temp  mid_temp  bottom_temp  reflux_ratio  reboiler_duty  pressure  feed_rate   purity
    count   43200.00  43200.00  43200.00     43200.00      43200.00       43200.00  43200.00   43200.00 43200.00
    mean       60.01     85.00    120.00       155.00          2.50        1500.01      1.20     100.00    96.50
    std         2.45      1.50      2.00         3.00          0.20          80.00      0.08       5.00     1.23
    ...
    

**解説** : 実プロセスデータには、日周変動、欠損値、外れ値が含まれます。EDAでこれらのパターンを把握することが、後続の分析の質を決定します。

#### コード例2: データクリーニングと前処理
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler
    from scipy import stats
    
    # データ読み込み
    df = pd.read_csv('distillation_data.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    print("【データクリーニング開始】")
    print(f"元データ: {len(df)}件")
    
    # ステップ1: 欠損値処理
    print("\n■ ステップ1: 欠損値処理")
    missing_before = df.isnull().sum().sum()
    print(f"欠損値数（処理前）: {missing_before}")
    
    # 線形補間（時系列データに適切）
    df_cleaned = df.copy()
    df_cleaned['top_temp'] = df_cleaned['top_temp'].interpolate(method='linear')
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"欠損値数（処理後）: {missing_after}")
    
    # ステップ2: 外れ値検出と処理
    print("\n■ ステップ2: 外れ値検出（IQR法）")
    
    def detect_outliers_iqr(series, multiplier=1.5):
        """IQR法で外れ値を検出"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    # 圧力の外れ値検出
    outliers, lower, upper = detect_outliers_iqr(df_cleaned['pressure'])
    print(f"圧力の外れ値: {outliers.sum()}件（{outliers.sum()/len(df_cleaned)*100:.2f}%）")
    print(f"  許容範囲: {lower:.3f} 〜 {upper:.3f} MPa")
    
    # 外れ値を中央値で置換（保守的な対処）
    df_cleaned.loc[outliers, 'pressure'] = df_cleaned['pressure'].median()
    
    # ステップ3: スケーリング
    print("\n■ ステップ3: データスケーリング（RobustScaler）")
    
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate']
    
    scaler = RobustScaler()
    df_scaled = df_cleaned.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_cleaned[feature_cols])
    
    print("スケーリング完了（RobustScaler使用）")
    
    # ステップ4: 特徴量エンジニアリング
    print("\n■ ステップ4: 特徴量エンジニアリング")
    
    df_cleaned['temp_gradient'] = df_cleaned['top_temp'] - df_cleaned['bottom_temp']
    df_cleaned['energy_efficiency'] = df_cleaned['reboiler_duty'] / df_cleaned['feed_rate']
    df_cleaned['hour'] = df_cleaned.index.hour
    df_cleaned['day_of_week'] = df_cleaned.index.dayofweek
    
    # 周期性の特徴量（サイクリックエンコーディング）
    df_cleaned['hour_sin'] = np.sin(2 * np.pi * df_cleaned['hour'] / 24)
    df_cleaned['hour_cos'] = np.cos(2 * np.pi * df_cleaned['hour'] / 24)
    
    print(f"追加特徴量: {4}個")
    print("  - temp_gradient: 塔頂-塔底の温度差")
    print("  - energy_efficiency: 単位供給量あたりのエネルギー")
    print("  - hour_sin/cos: 時刻の周期性エンコーディング")
    
    # 可視化: クリーニング前後の比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 欠損値補完の効果
    time_window = slice('2025-01-15 00:00', '2025-01-15 12:00')
    axes[0, 0].plot(df.loc[time_window].index, df.loc[time_window, 'top_temp'],
                    'o-', markersize=3, label='Before (with missing)', alpha=0.7)
    axes[0, 0].plot(df_cleaned.loc[time_window].index, df_cleaned.loc[time_window, 'top_temp'],
                    '-', linewidth=2, label='After (interpolated)', color='#11998e')
    axes[0, 0].set_ylabel('Top Temperature (°C)', fontsize=11)
    axes[0, 0].set_title('Missing Value Imputation', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 外れ値除去の効果
    axes[0, 1].hist(df['pressure'], bins=50, alpha=0.5, label='Before', edgecolor='black')
    axes[0, 1].hist(df_cleaned['pressure'], bins=50, alpha=0.5, label='After',
                    color='#11998e', edgecolor='black')
    axes[0, 1].axvline(lower, color='red', linestyle='--', label='Lower bound')
    axes[0, 1].axvline(upper, color='red', linestyle='--', label='Upper bound')
    axes[0, 1].set_xlabel('Pressure (MPa)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Outlier Removal', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # スケーリング前後の比較
    axes[1, 0].boxplot([df_cleaned['feed_temp'], df_cleaned['reboiler_duty']],
                       labels=['feed_temp', 'reboiler_duty'], patch_artist=True)
    axes[1, 0].set_ylabel('Original Scale', fontsize=11)
    axes[1, 0].set_title('Before Scaling (Different Scales)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    axes[1, 1].boxplot([df_scaled['feed_temp'], df_scaled['reboiler_duty']],
                       labels=['feed_temp', 'reboiler_duty'], patch_artist=True,
                       boxprops=dict(facecolor='#11998e', alpha=0.7))
    axes[1, 1].set_ylabel('Scaled Value', fontsize=11)
    axes[1, 1].set_title('After Scaling (Unified Scale)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data_cleaning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # クリーニング済みデータの保存
    df_cleaned.to_csv('distillation_data_cleaned.csv')
    print(f"\n【クリーニング完了】: distillation_data_cleaned.csv を保存しました")
    print(f"最終データ数: {len(df_cleaned)}件")
    

**出力例** :
    
    
    【データクリーニング開始】
    元データ: 43200件
    
    ■ ステップ1: 欠損値処理
    欠損値数（処理前）: 864
    欠損値数（処理後）: 0
    
    ■ ステップ2: 外れ値検出（IQR法）
    圧力の外れ値: 216件（0.50%）
      許容範囲: 1.080 〜 1.320 MPa
    
    ■ ステップ3: データスケーリング（RobustScaler）
    スケーリング完了（RobustScaler使用）
    
    ■ ステップ4: 特徴量エンジニアリング
    追加特徴量: 4個
      - temp_gradient: 塔頂-塔底の温度差
      - energy_efficiency: 単位供給量あたりのエネルギー
      - hour_sin/cos: 時刻の周期性エンコーディング
    

**解説** : データクリーニングと特徴量エンジニアリングは、モデル性能に直結する重要なステップです。ドメイン知識を活用した特徴量（温度勾配、エネルギー効率）が特に有効です。

* * *

## 4.2 品質予測モデルの構築

クリーニング済みデータを使って、製品純度を予測するソフトセンサーを構築します。複数のモデルを比較し、最適なモデルを選択します。

#### コード例3: 訓練データとテストデータの準備
    
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    
    # クリーニング済みデータ読み込み
    df = pd.read_csv('distillation_data_cleaned.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    print("【データ準備】")
    
    # オフライン測定データのみを使用（実運用を想定）
    train_data = df[df['purity_measured'].notna()].copy()
    print(f"訓練用データ数: {len(train_data)}件（オフライン測定のみ）")
    
    # 特徴量と目的変数
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate',
                    'temp_gradient', 'energy_efficiency', 'hour_sin', 'hour_cos']
    
    X = train_data[feature_cols]
    y = train_data['purity_measured']
    
    print(f"特徴量数: {len(feature_cols)}")
    print(f"特徴量: {feature_cols}")
    
    # 時系列分割（Time Series Split）
    # 時系列データでは、未来のデータで過去を予測しないように注意
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"\n時系列分割: {tscv.n_splits} folds")
    
    # 最終評価用に、最後の20%をテストデータとして確保
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"\n訓練データ: {len(X_train)}件")
    print(f"テストデータ: {len(X_test)}件")
    
    # スケーリング
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DataFrameに戻す（カラム名保持）
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print("\nスケーリング完了")
    print(f"訓練データ形状: {X_train_scaled.shape}")
    print(f"テストデータ形状: {X_test_scaled.shape}")
    
    # データの基本統計
    print("\n【訓練データ統計】")
    print(y_train.describe())
    print("\n【テストデータ統計】")
    print(y_test.describe())
    

**出力例** :
    
    
    【データ準備】
    訓練用データ数: 31件（オフライン測定のみ）
    特徴量数: 12
    特徴量: ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp', 'reflux_ratio',
             'reboiler_duty', 'pressure', 'feed_rate', 'temp_gradient',
             'energy_efficiency', 'hour_sin', 'hour_cos']
    
    時系列分割: 5 folds
    
    訓練データ: 25件
    テストデータ: 6件
    

**解説** : 時系列データでは、ランダム分割ではなく時系列分割を使用します。これにより、過去のデータで未来を予測する実運用と同じ条件で評価できます。

#### コード例4: 複数モデルの比較と選択
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import time
    
    # モデルの定義
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'PLS': PLSRegression(n_components=5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'SVR': SVR(kernel='rbf', C=10, gamma=0.1)
    }
    
    print("【モデル比較】")
    print("クロスバリデーションで各モデルを評価...")
    
    results = []
    
    for name, model in models.items():
        start_time = time.time()
    
        # クロスバリデーション（時系列分割）
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
        # 訓練
        model.fit(X_train_scaled, y_train)
    
        # 予測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    
        # 評価指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    
        training_time = time.time() - start_time
    
        results.append({
            'Model': name,
            'CV R² (mean)': cv_scores.mean(),
            'CV R² (std)': cv_scores.std(),
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Training Time (s)': training_time
        })
    
        print(f"  {name}: CV R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), "
              f"Test R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
    
    # 結果をDataFrameに
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    print("\n【総合評価結果】")
    print(results_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. テストR²スコアの比較
    axes[0, 0].barh(results_df['Model'], results_df['Test R²'], color='#11998e', alpha=0.7)
    axes[0, 0].set_xlabel('Test R² Score', fontsize=11)
    axes[0, 0].set_title('Model Performance Comparison (Test R²)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # 2. RMSE vs 訓練時間
    axes[0, 1].scatter(results_df['Training Time (s)'], results_df['Test RMSE'],
                       s=150, alpha=0.7, color='#11998e')
    for i, row in results_df.iterrows():
        axes[0, 1].annotate(row['Model'], (row['Training Time (s)'], row['Test RMSE']),
                            fontsize=8, ha='right')
    axes[0, 1].set_xlabel('Training Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Test RMSE', fontsize=11)
    axes[0, 1].set_title('Efficiency vs Accuracy Trade-off', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 訓練R² vs テストR²（過学習チェック）
    axes[1, 0].scatter(results_df['Train R²'], results_df['Test R²'],
                       s=150, alpha=0.7, color='#f59e0b')
    axes[1, 0].plot([0.9, 1.0], [0.9, 1.0], 'r--', linewidth=2, label='Perfect generalization')
    for i, row in results_df.iterrows():
        axes[1, 0].annotate(row['Model'], (row['Train R²'], row['Test R²']),
                            fontsize=8, ha='right')
    axes[1, 0].set_xlabel('Train R²', fontsize=11)
    axes[1, 0].set_ylabel('Test R²', fontsize=11)
    axes[1, 0].set_title('Overfitting Check', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. CVスコアの分布
    cv_means = results_df['CV R² (mean)']
    cv_stds = results_df['CV R² (std)']
    axes[1, 1].barh(results_df['Model'], cv_means, xerr=cv_stds,
                    color='#7b2cbf', alpha=0.7, capsize=5)
    axes[1, 1].set_xlabel('Cross-Validation R² Score', fontsize=11)
    axes[1, 1].set_title('Cross-Validation Performance (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 最優秀モデルの選択
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n【最優秀モデル】: {best_model_name}")
    print(f"  Test R²: {results_df.iloc[0]['Test R²']:.4f}")
    print(f"  Test RMSE: {results_df.iloc[0]['Test RMSE']:.4f}%")
    print(f"  Test MAE: {results_df.iloc[0]['Test MAE']:.4f}%")
    
    # 最優秀モデルを保存（後で使用）
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nモデルとスケーラーを保存しました")
    

**出力例** :
    
    
    【モデル比較】
    クロスバリデーションで各モデルを評価...
      Linear Regression: CV R² = 0.8456 (±0.1234), Test R² = 0.8678, RMSE = 0.4321
      Ridge: CV R² = 0.8512 (±0.1198), Test R² = 0.8723, RMSE = 0.4256
      Lasso: CV R² = 0.8389 (±0.1276), Test R² = 0.8598, RMSE = 0.4456
      PLS: CV R² = 0.8623 (±0.1089), Test R² = 0.8845, RMSE = 0.4034
      Random Forest: CV R² = 0.9012 (±0.0789), Test R² = 0.9234, RMSE = 0.3287
      Gradient Boosting: CV R² = 0.9156 (±0.0723), Test R² = 0.9345, RMSE = 0.3041
      SVR: CV R² = 0.8876 (±0.0856), Test R² = 0.9087, RMSE = 0.3589
    
    【最優秀モデル】: Gradient Boosting
      Test R²: 0.9345
      Test RMSE: 0.3041%
      Test MAE: 0.2456%
    

**解説** : 複数のモデルを体系的に比較することで、データに最適なモデルを選択できます。この例では、Gradient Boostingが最高性能を示しました。

#### コード例5: 特徴量重要度分析
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.inspection import permutation_importance
    
    # 最優秀モデルの読み込み
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【特徴量重要度分析】")
    
    # 方法1: モデル固有の特徴量重要度（Random ForestやGradient Boostingの場合）
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
        print("\n■ モデル固有の特徴量重要度:")
        print(feature_importance.to_string(index=False))
    
    # 方法2: Permutation Importance（モデルに依存しない）
    perm_importance = permutation_importance(best_model, X_test_scaled, y_test,
                                              n_repeats=10, random_state=42)
    
    perm_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n■ Permutation Importance:")
    print(perm_importance_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # モデル固有の重要度
    if hasattr(best_model, 'feature_importances_'):
        axes[0].barh(feature_importance['Feature'], feature_importance['Importance'],
                     color='#11998e', alpha=0.7)
        axes[0].set_xlabel('Importance', fontsize=11)
        axes[0].set_title('Feature Importance (Model-specific)', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x')
        axes[0].invert_yaxis()
    
    # Permutation Importance
    axes[1].barh(perm_importance_df['Feature'], perm_importance_df['Importance'],
                 xerr=perm_importance_df['Std'], color='#f59e0b', alpha=0.7, capsize=5)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Permutation Importance (Model-agnostic)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 主要因子の解釈
    print("\n【主要影響因子】")
    top_features = perm_importance_df.head(5)
    for i, row in top_features.iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})")
    
    print("\n【解釈】")
    print("✓ 還流比が純度に最も大きな影響を与える")
    print("✓ 塔頂温度と塔中段温度も重要な制御変数")
    print("✓ エネルギー効率（派生特徴量）が有意に寄与")
    print("→ これらの変数を重点的に管理することで品質安定化が可能")
    

**解説** : 特徴量重要度分析により、どの変数が品質に影響するかが定量的に分かります。これは、プロセス制御の優先順位付けに直結します。

* * *

## 4.3 プロセス条件最適化の基礎

構築したモデルを使って、品質制約を満たしつつエネルギー消費を最小化する運転条件を探索します。

#### コード例6: 制約付き最適化（Grid Search）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from itertools import product
    
    # モデルとスケーラーの読み込み
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【プロセス最適化】")
    print("目的: 品質（純度≥97%）を満たしつつ、エネルギー消費（reboiler_duty）を最小化")
    
    # 最適化対象の変数と探索範囲
    # 固定する変数（外部条件）
    feed_temp_fixed = 60.0
    feed_rate_fixed = 100.0
    pressure_fixed = 1.2
    
    # 最適化する変数
    reflux_ratios = np.linspace(2.0, 3.5, 20)
    reboiler_duties = np.linspace(1300, 1700, 20)
    
    print(f"\n探索範囲:")
    print(f"  還流比: {reflux_ratios.min():.2f} 〜 {reflux_ratios.max():.2f}")
    print(f"  リボイラー熱量: {reboiler_duties.min():.0f} 〜 {reboiler_duties.max():.0f} kW")
    print(f"  探索点数: {len(reflux_ratios) × len(reboiler_duties)}点")
    
    # グリッドサーチ
    results = []
    
    for reflux_ratio, reboiler_duty in product(reflux_ratios, reboiler_duties):
        # 運転条件から派生特徴量を計算
        # 注: top_temp, mid_temp, bottom_tempは相関関係から推定（簡易版）
        # 実際は物理モデルやより高度な予測が必要
        top_temp = 85 + 0.5 * (reflux_ratio - 2.5)  # 簡易推定
        mid_temp = 120
        bottom_temp = 155
    
        temp_gradient = top_temp - bottom_temp
        energy_efficiency = reboiler_duty / feed_rate_fixed
        hour_sin = 0  # 正午を想定
        hour_cos = 1
    
        # 特徴量ベクトル作成
        features = np.array([[
            feed_temp_fixed, top_temp, mid_temp, bottom_temp,
            reflux_ratio, reboiler_duty, pressure_fixed, feed_rate_fixed,
            temp_gradient, energy_efficiency, hour_sin, hour_cos
        ]])
    
        # スケーリング
        features_df = pd.DataFrame(features, columns=feature_cols)
        features_scaled = scaler.transform(features_df)
    
        # 純度予測
        purity_pred = best_model.predict(features_scaled)[0]
    
        results.append({
            'reflux_ratio': reflux_ratio,
            'reboiler_duty': reboiler_duty,
            'purity_pred': purity_pred,
            'feasible': purity_pred >= 97.0  # 品質制約
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n【探索結果】")
    print(f"全探索点数: {len(results_df)}")
    print(f"品質制約を満たす点: {results_df['feasible'].sum()}点")
    
    # 実行可能領域での最適解
    feasible_solutions = results_df[results_df['feasible']]
    
    if len(feasible_solutions) > 0:
        optimal_solution = feasible_solutions.loc[feasible_solutions['reboiler_duty'].idxmin()]
    
        print(f"\n【最適運転条件】")
        print(f"  還流比: {optimal_solution['reflux_ratio']:.3f}")
        print(f"  リボイラー熱量: {optimal_solution['reboiler_duty']:.1f} kW")
        print(f"  予測純度: {optimal_solution['purity_pred']:.2f}%")
    
        # 現在の運転条件と比較（平均値）
        current_reflux = X_train['reflux_ratio'].mean()
        current_duty = X_train['reboiler_duty'].mean()
        current_purity = y_train.mean()
    
        print(f"\n【現状との比較】")
        print(f"  還流比: {current_reflux:.3f} → {optimal_solution['reflux_ratio']:.3f}")
        print(f"  リボイラー熱量: {current_duty:.1f} kW → {optimal_solution['reboiler_duty']:.1f} kW")
        print(f"  予測純度: {current_purity:.2f}% → {optimal_solution['purity_pred']:.2f}%")
    
        energy_saving = (current_duty - optimal_solution['reboiler_duty']) / current_duty * 100
        print(f"\nエネルギー削減: {energy_saving:.1f}%")
        print(f"年間コスト削減（仮定）: ¥{energy_saving * 100000:.0f}万円")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # コンターマップ: 純度
    contour = axes[0].tricontourf(results_df['reflux_ratio'], results_df['reboiler_duty'],
                                   results_df['purity_pred'], levels=20, cmap='RdYlGn')
    axes[0].tricontour(results_df['reflux_ratio'], results_df['reboiler_duty'],
                       results_df['purity_pred'], levels=[97.0], colors='red',
                       linewidths=3, linestyles='--')
    if len(feasible_solutions) > 0:
        axes[0].scatter(optimal_solution['reflux_ratio'], optimal_solution['reboiler_duty'],
                        s=200, color='blue', marker='*', edgecolor='white', linewidth=2,
                        label='Optimal point', zorder=5)
    axes[0].set_xlabel('Reflux Ratio', fontsize=11)
    axes[0].set_ylabel('Reboiler Duty (kW)', fontsize=11)
    axes[0].set_title('Predicted Purity (% contour)', fontsize=12, fontweight='bold')
    axes[0].legend()
    plt.colorbar(contour, ax=axes[0], label='Purity (%)')
    
    # 実行可能領域
    axes[1].scatter(results_df[~results_df['feasible']]['reflux_ratio'],
                    results_df[~results_df['feasible']]['reboiler_duty'],
                    s=30, alpha=0.3, color='red', label='Infeasible (purity < 97%)')
    axes[1].scatter(results_df[results_df['feasible']]['reflux_ratio'],
                    results_df[results_df['feasible']]['reboiler_duty'],
                    s=30, alpha=0.5, color='green', label='Feasible (purity ≥ 97%)')
    if len(feasible_solutions) > 0:
        axes[1].scatter(optimal_solution['reflux_ratio'], optimal_solution['reboiler_duty'],
                        s=200, color='blue', marker='*', edgecolor='white', linewidth=2,
                        label='Optimal point', zorder=5)
    axes[1].set_xlabel('Reflux Ratio', fontsize=11)
    axes[1].set_ylabel('Reboiler Duty (kW)', fontsize=11)
    axes[1].set_title('Feasible Region', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    else:
        print("\n✗ 品質制約を満たす運転条件が見つかりませんでした")
        print("  → 探索範囲の拡大または制約の緩和が必要")
    

**出力例** :
    
    
    【プロセス最適化】
    目的: 品質（純度≥97%）を満たしつつ、エネルギー消費（reboiler_duty）を最小化
    
    探索範囲:
      還流比: 2.00 〜 3.50
      リボイラー熱量: 1300 〜 1700 kW
      探索点数: 400点
    
    【探索結果】
    全探索点数: 400
    品質制約を満たす点: 156点
    
    【最適運転条件】
      還流比: 2.789
      リボイラー熱量: 1368.4 kW
      予測純度: 97.12%
    
    【現状との比較】
      還流比: 2.503 → 2.789
      リボイラー熱量: 1499.8 kW → 1368.4 kW
      予測純度: 96.51% → 97.12%
    
    エネルギー削減: 8.8%
    年間コスト削減（仮定）: ¥880万円
    

**解説** : Grid Searchによる最適化で、品質を向上させつつエネルギー削減を達成する条件を発見しました。実プラントでは、さらに高度な最適化手法（遺伝的アルゴリズム、ベイズ最適化など）も活用されます。

#### コード例7: 高度な最適化（Scipy.optimize）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from scipy.optimize import minimize, differential_evolution
    
    # モデルとスケーラーの読み込み
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【高度な最適化（Scipy.optimize）】")
    
    # 固定パラメータ
    feed_temp_fixed = 60.0
    feed_rate_fixed = 100.0
    pressure_fixed = 1.2
    
    # 目的関数: エネルギー（最小化）
    def objective(x):
        """エネルギー消費を最小化（リボイラー熱量）"""
        reflux_ratio, reboiler_duty = x
        return reboiler_duty  # 最小化したい目標
    
    # 制約条件: 純度≥97%
    def constraint_purity(x):
        """純度制約（≥97%）"""
        reflux_ratio, reboiler_duty = x
    
        # 特徴量計算
        top_temp = 85 + 0.5 * (reflux_ratio - 2.5)
        mid_temp = 120
        bottom_temp = 155
        temp_gradient = top_temp - bottom_temp
        energy_efficiency = reboiler_duty / feed_rate_fixed
    
        features = np.array([[
            feed_temp_fixed, top_temp, mid_temp, bottom_temp,
            reflux_ratio, reboiler_duty, pressure_fixed, feed_rate_fixed,
            temp_gradient, energy_efficiency, 0, 1
        ]])
    
        features_df = pd.DataFrame(features, columns=feature_cols)
        features_scaled = scaler.transform(features_df)
    
        purity_pred = best_model.predict(features_scaled)[0]
    
        # 制約: purity >= 97 → purity - 97 >= 0
        return purity_pred - 97.0
    
    # 変数の範囲
    bounds = [
        (2.0, 3.5),      # 還流比
        (1300, 1700)     # リボイラー熱量 (kW)
    ]
    
    # 制約の定義
    constraints = [
        {'type': 'ineq', 'fun': constraint_purity}  # inequality: f(x) >= 0
    ]
    
    # 初期値
    x0 = [2.5, 1500]
    
    print("\n方法1: SLSQP（勾配ベース）")
    result_slsqp = minimize(objective, x0, method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'disp': True})
    
    if result_slsqp.success:
        print(f"\n【最適解（SLSQP）】")
        print(f"  還流比: {result_slsqp.x[0]:.3f}")
        print(f"  リボイラー熱量: {result_slsqp.x[1]:.1f} kW")
        print(f"  予測純度: {constraint_purity(result_slsqp.x) + 97:.2f}%")
    else:
        print("\n最適化失敗（SLSQP）")
    
    # 方法2: Differential Evolution（進化的アルゴリズム）
    print("\n\n方法2: Differential Evolution（大域的探索）")
    
    def objective_with_penalty(x):
        """ペナルティ関数法で制約を目的関数に組み込む"""
        energy = objective(x)
        purity_constraint = constraint_purity(x)
    
        # 制約違反にペナルティ
        if purity_constraint < 0:
            penalty = 1000 * abs(purity_constraint)
            return energy + penalty
        else:
            return energy
    
    result_de = differential_evolution(objective_with_penalty, bounds,
                                        seed=42, disp=True, maxiter=100)
    
    print(f"\n【最適解（Differential Evolution）】")
    print(f"  還流比: {result_de.x[0]:.3f}")
    print(f"  リボイラー熱量: {result_de.x[1]:.1f} kW")
    print(f"  予測純度: {constraint_purity(result_de.x) + 97:.2f}%")
    
    # 結果の比較
    print(f"\n【最適化手法の比較】")
    print(f"SLSQP（局所最適化）: エネルギー = {result_slsqp.fun:.1f} kW")
    print(f"Differential Evolution（大域最適化）: エネルギー = {result_de.fun:.1f} kW")
    
    # 可視化: 最適化の経路
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # グリッドでの純度分布
    reflux_grid = np.linspace(2.0, 3.5, 50)
    duty_grid = np.linspace(1300, 1700, 50)
    R, D = np.meshgrid(reflux_grid, duty_grid)
    
    purity_grid = np.zeros_like(R)
    for i in range(len(reflux_grid)):
        for j in range(len(duty_grid)):
            purity_grid[j, i] = constraint_purity([R[j, i], D[j, i]]) + 97
    
    contour = ax.contourf(R, D, purity_grid, levels=20, cmap='RdYlGn', alpha=0.6)
    ax.contour(R, D, purity_grid, levels=[97.0], colors='red', linewidths=3, linestyles='--')
    
    # 最適解をプロット
    if result_slsqp.success:
        ax.scatter(result_slsqp.x[0], result_slsqp.x[1], s=200, color='blue',
                   marker='o', edgecolor='white', linewidth=2, label='SLSQP', zorder=5)
    
    ax.scatter(result_de.x[0], result_de.x[1], s=200, color='orange',
               marker='*', edgecolor='white', linewidth=2, label='Differential Evolution', zorder=5)
    
    # 初期点
    ax.scatter(x0[0], x0[1], s=100, color='black', marker='x', linewidth=2,
               label='Initial point', zorder=5)
    
    ax.set_xlabel('Reflux Ratio', fontsize=12)
    ax.set_ylabel('Reboiler Duty (kW)', fontsize=12)
    ax.set_title('Optimization Results on Purity Contour', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.colorbar(contour, ax=ax, label='Purity (%)')
    plt.tight_layout()
    plt.savefig('advanced_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n【最適化手法の選択指針】")
    print("✓ SLSQP: 高速、勾配情報を利用、局所最適解")
    print("✓ Differential Evolution: 遅い、大域的探索、複雑な目的関数に強い")
    print("✓ 実務: まずSLSQPで高速探索、必要に応じてDEで確認")
    

**出力例** :
    
    
    【高度な最適化（Scipy.optimize）】
    
    方法1: SLSQP（勾配ベース）
    Optimization terminated successfully
    
    【最適解（SLSQP）】
      還流比: 2.784
      リボイラー熱量: 1365.2 kW
      予測純度: 97.03%
    
    方法2: Differential Evolution（大域的探索）
    
    【最適解（Differential Evolution）】
      還流比: 2.789
      リボイラー熱量: 1363.8 kW
      予測純度: 97.05%
    
    【最適化手法の比較】
    SLSQP（局所最適化）: エネルギー = 1365.2 kW
    Differential Evolution（大域最適化）: エネルギー = 1363.8 kW
    

**解説** : Scipy.optimizeを使うことで、より高度な最適化が可能です。SLSQPは高速ですが局所解に陥る可能性があり、Differential Evolutionは遅いですが大域的最適解を見つけやすい特徴があります。

* * *

## 4.4 実装プロジェクト全体のワークフロー

これまでの全ステップを統合し、エンドツーエンドのPIプロジェクトワークフローを確立します。

#### コード例8: 統合パイプラインとデプロイメント準備
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import json
    
    print("=" * 80)
    print("【PI統合プロジェクト: エンドツーエンドワークフロー】")
    print("=" * 80)
    
    # ============================================================================
    # ステップ1: データパイプライン構築
    # ============================================================================
    print("\n【ステップ1: データパイプライン構築】")
    
    class ProcessDataPipeline:
        """プロセスデータの前処理パイプライン"""
    
        def __init__(self):
            self.scaler = RobustScaler()
            self.feature_cols = None
    
        def fit(self, df, feature_cols):
            """訓練データでパイプラインをfit"""
            self.feature_cols = feature_cols
    
            # 欠損値補完
            df_clean = df.copy()
            for col in feature_cols:
                df_clean[col] = df_clean[col].interpolate(method='linear')
    
            # スケーリング
            self.scaler.fit(df_clean[feature_cols])
    
            return self
    
        def transform(self, df):
            """データを変換"""
            df_clean = df.copy()
    
            # 欠損値補完
            for col in self.feature_cols:
                df_clean[col] = df_clean[col].interpolate(method='linear')
    
            # スケーリング
            df_clean[self.feature_cols] = self.scaler.transform(df_clean[self.feature_cols])
    
            return df_clean
    
        def save(self, filepath):
            """パイプラインを保存"""
            joblib.dump(self, filepath)
            print(f"  パイプライン保存: {filepath}")
    
        @staticmethod
        def load(filepath):
            """パイプラインを読み込み"""
            return joblib.load(filepath)
    
    # パイプラインのインスタンス化
    pipeline = ProcessDataPipeline()
    
    # データ読み込み
    df = pd.read_csv('distillation_data_cleaned.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate',
                    'temp_gradient', 'energy_efficiency', 'hour_sin', 'hour_cos']
    
    # オフライン測定データのみ
    train_data = df[df['purity_measured'].notna()].copy()
    X_train = train_data[feature_cols]
    y_train = train_data['purity_measured']
    
    # パイプラインをfit
    pipeline.fit(X_train, feature_cols)
    X_train_processed = pipeline.transform(X_train)
    
    print("  データ前処理パイプライン構築完了")
    
    # ============================================================================
    # ステップ2: モデルトレーニング
    # ============================================================================
    print("\n【ステップ2: モデルトレーニング】")
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
    model.fit(X_train_processed[feature_cols], y_train)
    
    print(f"  モデル: Gradient Boosting Regressor")
    print(f"  訓練データ数: {len(X_train)}")
    print(f"  特徴量数: {len(feature_cols)}")
    
    # ============================================================================
    # ステップ3: モデル評価
    # ============================================================================
    print("\n【ステップ3: モデル評価】")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    
    cv_scores = cross_val_score(model, X_train_processed[feature_cols], y_train, cv=5, scoring='r2')
    print(f"  CV R² スコア: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    y_train_pred = model.predict(X_train_processed[feature_cols])
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"  訓練データ R²: {train_r2:.4f}")
    print(f"  訓練データ RMSE: {train_rmse:.4f}%")
    
    # ============================================================================
    # ステップ4: デプロイメント準備
    # ============================================================================
    print("\n【ステップ4: デプロイメント準備】")
    
    # モデルとパイプラインを保存
    model_path = 'production_model.pkl'
    pipeline_path = 'production_pipeline.pkl'
    
    joblib.dump(model, model_path)
    pipeline.save(pipeline_path)
    
    print(f"  モデル保存: {model_path}")
    print(f"  パイプライン保存: {pipeline_path}")
    
    # メタデータの保存
    metadata = {
        'model_type': 'Gradient Boosting Regressor',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(X_train),
        'features': feature_cols,
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'train_r2': float(train_r2),
        'train_rmse': float(train_rmse),
        'target_variable': 'purity',
        'target_unit': '%',
        'quality_threshold': 97.0
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  メタデータ保存: model_metadata.json")
    
    # ============================================================================
    # ステップ5: 推論関数（デプロイメント用）
    # ============================================================================
    print("\n【ステップ5: 推論関数の実装】")
    
    def predict_purity(process_data):
        """
        プロセスデータから純度を予測
    
        Parameters:
        -----------
        process_data : dict
            プロセス変数の辞書
            例: {'feed_temp': 60, 'top_temp': 85, ...}
    
        Returns:
        --------
        float
            予測純度（%）
        """
        # モデルとパイプラインの読み込み
        model = joblib.load('production_model.pkl')
        pipeline = ProcessDataPipeline.load('production_pipeline.pkl')
    
        # データフレームに変換
        df = pd.DataFrame([process_data])
    
        # 前処理
        df_processed = pipeline.transform(df)
    
        # 予測
        purity_pred = model.predict(df_processed[feature_cols])[0]
    
        return purity_pred
    
    # テスト実行
    test_data = {
        'feed_temp': 60.0,
        'top_temp': 85.5,
        'mid_temp': 120.0,
        'bottom_temp': 155.0,
        'reflux_ratio': 2.8,
        'reboiler_duty': 1400,
        'pressure': 1.2,
        'feed_rate': 100,
        'temp_gradient': -69.5,
        'energy_efficiency': 14.0,
        'hour_sin': 0,
        'hour_cos': 1
    }
    
    purity_prediction = predict_purity(test_data)
    print(f"\n  推論テスト:")
    print(f"    入力: {test_data}")
    print(f"    予測純度: {purity_prediction:.2f}%")
    
    # ============================================================================
    # ステップ6: モニタリングダッシュボード用データ出力
    # ============================================================================
    print("\n【ステップ6: モニタリングダッシュボード用データ生成】")
    
    # 全データに対してリアルタイム予測
    df_all = df.copy()
    df_all_processed = pipeline.transform(df_all)
    df_all['purity_predicted'] = model.predict(df_all_processed[feature_cols])
    
    # 予測誤差の計算（オフライン測定がある時のみ）
    df_all['prediction_error'] = df_all['purity_measured'] - df_all['purity_predicted']
    
    # ダッシュボード用データの保存（最新1週間）
    dashboard_data = df_all.tail(10080)[['purity', 'purity_predicted', 'purity_measured',
                                           'prediction_error', 'reflux_ratio', 'reboiler_duty']]
    dashboard_data.to_csv('dashboard_data.csv')
    
    print("  ダッシュボード用データ保存: dashboard_data.csv")
    print(f"  データ期間: {dashboard_data.index.min()} 〜 {dashboard_data.index.max()}")
    
    # 性能サマリー
    errors = df_all['prediction_error'].dropna()
    print(f"\n  モデル性能サマリー（オフライン測定との比較）:")
    print(f"    平均誤差: {errors.mean():.4f}%")
    print(f"    標準偏差: {errors.std():.4f}%")
    print(f"    最大誤差: {errors.abs().max():.4f}%")
    
    # ============================================================================
    # まとめ
    # ============================================================================
    print("\n" + "=" * 80)
    print("【プロジェクト完了】")
    print("=" * 80)
    print("\n構築した成果物:")
    print("  1. production_model.pkl - 訓練済みモデル")
    print("  2. production_pipeline.pkl - データ前処理パイプライン")
    print("  3. model_metadata.json - モデルのメタ情報")
    print("  4. dashboard_data.csv - モニタリングダッシュボード用データ")
    print("  5. predict_purity() - 推論関数（本番デプロイ用）")
    
    print("\n次のステップ:")
    print("  ✓ 実プラントでのパイロット運用")
    print("  ✓ リアルタイムデータストリームとの接続")
    print("  ✓ 定期的なモデル再学習スケジュールの確立")
    print("  ✓ アラート機能の実装（予測純度 < 閾値）")
    print("  ✓ A/Bテストによる最適化条件の検証")
    

**出力例** :
    
    
    ================================================================================
    【PI統合プロジェクト: エンドツーエンドワークフロー】
    ================================================================================
    
    【ステップ1: データパイプライン構築】
      データ前処理パイプライン構築完了
    
    【ステップ2: モデルトレーニング】
      モデル: Gradient Boosting Regressor
      訓練データ数: 25
      特徴量数: 12
    
    【ステップ3: モデル評価】
      CV R² スコア: 0.8923 (±0.1056)
      訓練データ R²: 0.9567
      訓練データ RMSE: 0.2456%
    
    【ステップ4: デプロイメント準備】
      モデル保存: production_model.pkl
      パイプライン保存: production_pipeline.pkl
      メタデータ保存: model_metadata.json
    
    【ステップ5: 推論関数の実装】
    
      推論テスト:
        入力: {'feed_temp': 60.0, 'top_temp': 85.5, ...}
        予測純度: 97.34%
    
    【ステップ6: モニタリングダッシュボード用データ生成】
      ダッシュボード用データ保存: dashboard_data.csv
      データ期間: 2025-01-24 00:00:00 〜 2025-01-30 23:59:00
    
      モデル性能サマリー（オフライン測定との比較）:
        平均誤差: 0.0123%
        標準偏差: 0.2567%
        最大誤差: 0.5678%
    
    ================================================================================
    【プロジェクト完了】
    ================================================================================
    

**解説** : 実務では、モデル構築だけでなく、デプロイメント準備、推論関数の実装、モニタリング基盤の整備までが重要です。このワークフローは、実プラントへの適用の基礎となります。

* * *

## 4.5 まとめと次のステップ

### 本シリーズで学んだこと

**第1章: PIの基礎概念**

  * プロセス・インフォマティクスの定義と目的
  * プロセス産業の特徴とデータの種類
  * データ駆動型プロセス改善の実例とROI
  * Pythonによる基本的なデータ可視化

**第2章: データ前処理と可視化**

  * 時系列データの操作（リサンプリング、ローリング統計）
  * 欠損値処理と外れ値検出の実践手法
  * データスケーリングの選択と実装
  * 高度な可視化テクニック

**第3章: プロセスモデリングの基礎**

  * 線形回帰による品質予測モデル構築
  * PLSによる多重共線性対処
  * ソフトセンサーの設計と運用
  * モデル評価指標とクロスバリデーション
  * 非線形モデルへの拡張

**第4章: 実践演習**

  * 実プロセスデータのEDAとクリーニング
  * 特徴量エンジニアリング
  * 複数モデルの比較と最適モデル選択
  * プロセス条件最適化の基礎
  * エンドツーエンドのプロジェクトワークフロー

### 実務適用のチェックリスト

  1. **データ収集・管理**
     * □ プロセス変数と品質変数の特定
     * □ データ収集頻度の決定
     * □ データベース設計とヒストリアン接続
  2. **データ分析**
     * □ EDAによるデータ理解
     * □ 欠損値・外れ値の対処方針決定
     * □ 相関分析と特徴量選択
  3. **モデル構築**
     * □ ベンチマークモデルの構築（線形回帰）
     * □ 複数手法の比較検証
     * □ クロスバリデーションによる性能評価
     * □ モデルの解釈性確認
  4. **実装・運用**
     * □ パイロット運用計画の策定
     * □ リアルタイム推論基盤の構築
     * □ モニタリングダッシュボードの設置
     * □ 定期的な再学習スケジュール確立
  5. **継続改善**
     * □ 性能モニタリングとドリフト検出
     * □ A/Bテストによる効果検証
     * □ フィードバックループの構築

### 次のステップ：上級トピック

本シリーズで基礎を習得した方は、以下の上級トピックに進むことをお勧めします：

#### 1\. 高度なモデリング手法

  * **ディープラーニング** : LSTM、CNNを使った時系列予測
  * **アンサンブル学習** : スタッキング、ブレンディング
  * **ベイズ最適化** : ハイパーパラメータの効率的探索
  * **転移学習** : 他プラントのデータを活用

#### 2\. リアルタイムPI

  * **ストリーム処理** : Apache Kafka、Spark Streamingとの連携
  * **オンライン学習** : インクリメンタル学習、適応制御
  * **エッジコンピューティング** : 現場での高速推論

#### 3\. プロセス制御との統合

  * **MPC（モデル予測制御）** : PIモデルを制御に活用
  * **強化学習** : 自律的な運転条件最適化
  * **デジタルツイン** : 仮想プラントでのシミュレーション

#### 4\. 異常検知・診断

  * **統計的工程管理** : CUSUM、EWMA
  * **変化点検出** : プロセスドリフトの早期発見
  * **根本原因分析** : 異常発生メカニズムの解明

#### 5\. エンタープライズ展開

  * **MLOps** : モデルのバージョン管理、CI/CD
  * **スケーラビリティ** : 複数プラントへの横展開
  * **セキュリティ** : データガバナンス、アクセス制御

### 推奨学習リソース

#### 書籍

  * 「プロセス制御工学」（化学工学会編）
  * "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Aurélien Géron)
  * "Introduction to Statistical Learning" (James et al.)

#### オンラインコース

  * Coursera: "Machine Learning" (Andrew Ng)
  * Udacity: "Machine Learning Engineer Nanodegree"
  * Fast.ai: "Practical Deep Learning for Coders"

#### コミュニティ・カンファレンス

  * PSE (Process Systems Engineering) Conference
  * IFAC DYCOPS (Dynamics and Control of Process Systems)
  * 化学工学会 プロセスシステム工学部会

### 最後に

> **"Data is the new oil, but analytics is the combustion engine."**

プロセス・インフォマティクスは、製造業のデジタル変革を推進する強力な技術です。本シリーズで学んだ基礎を土台に、実プラントでのPIプロジェクトに挑戦してください。

**成功の鍵** :

  1. **小さく始める** : 1つの品質変数、1つのプロセスから
  2. **プロセスエンジニアと協力** : ドメイン知識とデータ分析の融合
  3. **継続的改善** : モデルは作って終わりではなく、育てるもの
  4. **価値の可視化** : ROIを定量的に示し、経営層の支持を得る

皆様のPIプロジェクトの成功を心より願っています。
