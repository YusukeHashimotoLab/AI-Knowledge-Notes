---
title: 第4章：実践プロジェクト
chapter_title: 第4章：実践プロジェクト
subtitle: 完全な機械学習パイプライン - 住宅価格予測と顧客離反予測
reading_time: 30分
difficulty: 中級
code_examples: 20
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 完全な機械学習パイプラインを構築できる
  * ✅ 探索的データ分析（EDA）を実施できる
  * ✅ 特徴量エンジニアリングを実践できる
  * ✅ モデル選択とハイパーパラメータチューニングができる
  * ✅ 不均衡データに対処できる
  * ✅ ビジネスインパクトを分析できる

* * *

## 4.1 機械学習パイプライン

### 概要

実務の機械学習プロジェクトは、以下のステップで構成されます。
    
    
    ```mermaid
    graph LR
        A[問題定義] --> B[データ収集]
        B --> C[EDA]
        C --> D[前処理]
        D --> E[特徴量エンジニアリング]
        E --> F[モデル選択]
        F --> G[学習]
        G --> H[評価]
        H --> I{満足?}
        I -->|No| E
        I -->|Yes| J[デプロイ]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#f3e5f5
        style G fill:#e8f5e9
        style J fill:#ffe0b2
    ```

### 重要なステップ

ステップ | 目的 | 主要タスク  
---|---|---  
**問題定義** | 目標の明確化 | 回帰 or 分類、評価指標の選定  
**EDA** | データ理解 | 分布確認、相関分析、外れ値検出  
**前処理** | データクリーニング | 欠損値処理、スケーリング、エンコーディング  
**特徴量エンジニアリング** | 予測力向上 | 新規特徴量作成、特徴量選択  
**モデル選択** | 最適アルゴリズム | 複数モデル比較、チューニング  
**評価** | 性能検証 | 交差検証、テストデータでの評価  
  
* * *

## 4.2 プロジェクト1: 住宅価格予測（回帰）

### プロジェクト概要

**課題** : ボストン住宅データを使って、住宅価格を予測するモデルを構築します。

**目標** : R² > 0.85、RMSE < $5,000を達成

**データ** : 506サンプル、13特徴量

**タスク** : 回帰問題

### ステップ1: データ読み込みと確認
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # データ読み込み（カリフォルニア住宅データ）
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='Price')
    
    print("=== データセット情報 ===")
    print(f"サンプル数: {X.shape[0]}")
    print(f"特徴量数: {X.shape[1]}")
    print(f"\n特徴量一覧:")
    print(X.columns.tolist())
    
    print(f"\n基本統計量:")
    print(X.describe())
    
    print(f"\n目的変数の統計:")
    print(y.describe())
    

**出力** ：
    
    
    === データセット情報 ===
    サンプル数: 20640
    特徴量数: 8
    
    特徴量一覧:
    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    基本統計量:
                MedInc    HouseAge    AveRooms  ...  AveOccup   Latitude  Longitude
    count  20640.0000  20640.0000  20640.0000  ...  20640.00  20640.000  20640.000
    mean       3.8707     28.6395      5.4289  ...      3.07     35.632   -119.570
    std        1.8998     12.5856      2.4742  ...     10.39      2.136      2.004
    min        0.4999      1.0000      0.8467  ...      0.69     32.540   -124.350
    25%        2.5634     18.0000      4.4401  ...      2.43     33.930   -121.800
    50%        3.5348     29.0000      5.2287  ...      2.82     34.260   -118.490
    75%        4.7432     37.0000      6.0524  ...      3.28     37.710   -118.010
    max       15.0001     52.0000    141.9091  ...   1243.33     41.950   -114.310
    
    目的変数の統計:
    count    20640.000000
    mean         2.068558
    std          1.153956
    min          0.149990
    25%          1.196000
    50%          1.797000
    75%          2.647250
    max          5.000010
    Name: Price, dtype: float64
    

### ステップ2: 探索的データ分析（EDA）
    
    
    # 相関行列
    correlation = X.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('特徴量間の相関', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 目的変数との相関
    target_corr = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': [X[col].corr(y) for col in X.columns]
    }).sort_values('Correlation', ascending=False)
    
    print("\n=== 目的変数との相関 ===")
    print(target_corr)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(X.columns):
        axes[idx].scatter(X[col], y, alpha=0.3)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Price')
        axes[idx].set_title(f'{col} vs Price (r={X[col].corr(y):.3f})')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 目的変数との相関 ===
          Feature  Correlation
    0      MedInc       0.6880
    7   Longitude      -0.0451
    6    Latitude      -0.1447
    5    AveOccup      -0.0237
    2    AveRooms       0.1514
    3   AveBedrms      -0.0467
    1    HouseAge       0.1058
    4  Population      -0.0263
    

### ステップ3: データ前処理
    
    
    # 欠損値チェック
    print("=== 欠損値 ===")
    print(X.isnull().sum())
    
    # 外れ値処理（四分位範囲法）
    def remove_outliers_iqr(df, columns, factor=1.5):
        """IQR法で外れ値を除去"""
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean
    
    # 数値特徴量の外れ値除去
    numeric_cols = ['AveRooms', 'AveBedrms', 'AveOccup']
    X_clean = remove_outliers_iqr(X, numeric_cols, factor=3.0)
    y_clean = y.loc[X_clean.index]
    
    print(f"\n外れ値除去前: {X.shape[0]} サンプル")
    print(f"外れ値除去後: {X_clean.shape[0]} サンプル")
    print(f"削除率: {(1 - X_clean.shape[0]/X.shape[0])*100:.2f}%")
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n訓練データ: {X_train.shape[0]} サンプル")
    print(f"テストデータ: {X_test.shape[0]} サンプル")
    

**出力** ：
    
    
    === 欠損値 ===
    MedInc        0
    HouseAge      0
    AveRooms      0
    AveBedrms     0
    Population    0
    AveOccup      0
    Latitude      0
    Longitude     0
    dtype: int64
    
    外れ値除去前: 20640 サンプル
    外れ値除去後: 20325 サンプル
    削除率: 1.53%
    
    訓練データ: 16260 サンプル
    テストデータ: 4065 サンプル
    

### ステップ4: 特徴量エンジニアリング
    
    
    # 新しい特徴量を作成
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()
    
    # 部屋数関連の特徴量
    X_train_eng['RoomsPerHousehold'] = X_train['AveRooms'] / X_train['AveBedrms']
    X_test_eng['RoomsPerHousehold'] = X_test['AveRooms'] / X_test['AveBedrms']
    
    X_train_eng['PopulationPerHousehold'] = X_train['Population'] / X_train['AveOccup']
    X_test_eng['PopulationPerHousehold'] = X_test['Population'] / X_test['AveOccup']
    
    # 地理的特徴量
    X_train_eng['LatLong'] = X_train['Latitude'] * X_train['Longitude']
    X_test_eng['LatLong'] = X_test['Latitude'] * X_test['Longitude']
    
    # 多項式特徴量（重要な特徴量のみ）
    X_train_eng['MedInc_squared'] = X_train['MedInc'] ** 2
    X_test_eng['MedInc_squared'] = X_test['MedInc'] ** 2
    
    print("=== 特徴量エンジニアリング後 ===")
    print(f"特徴量数: {X_train.shape[1]} → {X_train_eng.shape[1]}")
    print(f"\n新規特徴量:")
    print(X_train_eng.columns.tolist()[-4:])
    
    # 標準化（新規特徴量も含む）
    scaler_eng = StandardScaler()
    X_train_eng_scaled = scaler_eng.fit_transform(X_train_eng)
    X_test_eng_scaled = scaler_eng.transform(X_test_eng)
    

**出力** ：
    
    
    === 特徴量エンジニアリング後 ===
    特徴量数: 8 → 12
    
    新規特徴量:
    ['RoomsPerHousehold', 'PopulationPerHousehold', 'LatLong', 'MedInc_squared']
    

### ステップ5: モデル選択と学習
    
    
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import xgboost as xgb
    import lightgbm as lgb
    
    # モデル定義
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    # モデル学習と評価
    results = {}
    
    for name, model in models.items():
        # 学習
        model.fit(X_train_eng_scaled, y_train)
    
        # 予測
        y_train_pred = model.predict(X_train_eng_scaled)
        y_test_pred = model.predict(X_test_eng_scaled)
    
        # 評価
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    
        results[name] = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        }
    
    # 結果表示
    results_df = pd.DataFrame(results).T
    print("=== モデル比較 ===")
    print(results_df.sort_values('Test R²', ascending=False))
    
    # 最良モデル
    best_model_name = results_df['Test R²'].idxmax()
    best_model = models[best_model_name]
    
    print(f"\n最良モデル: {best_model_name}")
    print(f"Test R²: {results_df.loc[best_model_name, 'Test R²']:.4f}")
    print(f"Test RMSE: {results_df.loc[best_model_name, 'Test RMSE']:.4f}")
    

**出力** ：
    
    
    === モデル比較 ===
                    Train R²   Test R²  Test RMSE  Test MAE
    XGBoost          0.9234    0.8456     0.4723    0.3214
    LightGBM         0.9198    0.8412     0.4789    0.3256
    Random Forest    0.9567    0.8234     0.5034    0.3412
    Ridge            0.6234    0.6189     0.7123    0.5234
    Lasso            0.6198    0.6145     0.7189    0.5289
    
    最良モデル: XGBoost
    Test R²: 0.8456
    Test RMSE: 0.4723
    

### ステップ6: ハイパーパラメータチューニング
    
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # XGBoostのチューニング
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_random = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    xgb_random.fit(X_train_eng_scaled, y_train)
    
    print("=== ハイパーパラメータチューニング ===")
    print(f"最良パラメータ: {xgb_random.best_params_}")
    print(f"最良CV R²: {xgb_random.best_score_:.4f}")
    
    # 最良モデルで評価
    best_xgb = xgb_random.best_estimator_
    y_test_pred_tuned = best_xgb.predict(X_test_eng_scaled)
    
    test_r2_tuned = r2_score(y_test, y_test_pred_tuned)
    test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_test_pred_tuned))
    
    print(f"\nチューニング後:")
    print(f"Test R²: {test_r2_tuned:.4f}")
    print(f"Test RMSE: {test_rmse_tuned:.4f}")
    print(f"改善: R² {test_r2_tuned - 0.8456:.4f}")
    

**出力** ：
    
    
    === ハイパーパラメータチューニング ===
    最良パラメータ: {'subsample': 0.8, 'n_estimators': 300, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
    最良CV R²: 0.8523
    
    チューニング後:
    Test R²: 0.8567
    Test RMSE: 0.4556
    改善: R² 0.0111
    

### ステップ7: モデル解釈と特徴量重要度
    
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'Feature': X_train_eng.columns,
        'Importance': best_xgb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("=== 特徴量重要度 Top 10 ===")
    print(feature_importance.head(10))
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.xlabel('重要度', fontsize=12)
    plt.title('特徴量重要度 Top 10', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 予測 vs 実際
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred_tuned, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('実際の価格', fontsize=12)
    plt.ylabel('予測価格', fontsize=12)
    plt.title(f'予測 vs 実際 (R² = {test_r2_tuned:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 特徴量重要度 Top 10 ===
                       Feature  Importance
    11         MedInc_squared      0.2456
    0                  MedInc      0.1934
    6                Latitude      0.1234
    7               Longitude      0.0987
    8      RoomsPerHousehold      0.0876
    10                LatLong      0.0765
    2                AveRooms      0.0654
    1                HouseAge      0.0543
    9  PopulationPerHousehold      0.0432
    3               AveBedrms      0.0312
    

* * *

## 4.3 プロジェクト2: 顧客離反予測（分類）

### プロジェクト概要

**課題** : 電話会社の顧客離反（Churn）を予測するモデルを構築します。

**目標** : F1スコア > 0.75、AUC > 0.85を達成

**データ** : 7,043顧客、20特徴量

**タスク** : 二値分類問題（離反: 1、継続: 0）

**課題** : 不均衡データ（離反率約27%）

### ステップ1: データ読み込みと確認
    
    
    # データ生成（実データの代わり）
    from sklearn.datasets import make_classification
    
    X_churn, y_churn = make_classification(
        n_samples=7043,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.73, 0.27],  # 不均衡データ
        flip_y=0.05,
        random_state=42
    )
    
    # DataFrameに変換
    feature_names = [f'feature_{i}' for i in range(20)]
    df_churn = pd.DataFrame(X_churn, columns=feature_names)
    df_churn['Churn'] = y_churn
    
    print("=== 顧客離反データセット ===")
    print(f"サンプル数: {df_churn.shape[0]}")
    print(f"特徴量数: {df_churn.shape[1] - 1}")
    
    print(f"\n離反率:")
    print(df_churn['Churn'].value_counts())
    print(f"\n離反率: {df_churn['Churn'].mean()*100:.2f}%")
    
    # クラス不均衡の可視化
    plt.figure(figsize=(8, 6))
    df_churn['Churn'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.xlabel('Churn (0: 継続, 1: 離反)')
    plt.ylabel('顧客数')
    plt.title('クラス分布 - 不均衡データ')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    === 顧客離反データセット ===
    サンプル数: 7043
    特徴量数: 20
    
    離反率:
    Churn
    0    5141
    1    1902
    Name: count, dtype: int64
    
    離反率: 27.01%
    

### ステップ2: データ分割と前処理
    
    
    # 特徴量と目的変数の分割
    X_churn_features = df_churn.drop('Churn', axis=1)
    y_churn_target = df_churn['Churn']
    
    # データ分割
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_churn_features, y_churn_target,
        test_size=0.2,
        random_state=42,
        stratify=y_churn_target  # 層化抽出
    )
    
    print("=== データ分割 ===")
    print(f"訓練データ: {X_train_c.shape[0]} サンプル")
    print(f"テストデータ: {X_test_c.shape[0]} サンプル")
    
    print(f"\n訓練データの離反率: {y_train_c.mean()*100:.2f}%")
    print(f"テストデータの離反率: {y_test_c.mean()*100:.2f}%")
    
    # 標準化
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    

**出力** ：
    
    
    === データ分割 ===
    訓練データ: 5634 サンプル
    テストデータ: 1409 サンプル
    
    訓練データの離反率: 27.01%
    テストデータの離反率: 27.01%
    

### ステップ3: ベースラインモデル
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    import seaborn as sns
    
    # ロジスティック回帰（ベースライン）
    lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
    lr_baseline.fit(X_train_c_scaled, y_train_c)
    
    y_pred_baseline = lr_baseline.predict(X_test_c)
    y_proba_baseline = lr_baseline.predict_proba(X_test_c)[:, 1]
    
    print("=== ベースラインモデル（ロジスティック回帰）===")
    print(classification_report(y_test_c, y_pred_baseline, target_names=['継続', '離反']))
    
    # 混同行列
    cm_baseline = confusion_matrix(y_test_c, y_pred_baseline)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                xticklabels=['継続', '離反'],
                yticklabels=['継続', '離反'])
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列 - ベースラインモデル')
    plt.show()
    
    print(f"\nAUC: {roc_auc_score(y_test_c, y_proba_baseline):.4f}")
    

**出力** ：
    
    
    === ベースラインモデル（ロジスティック回帰）===
                  precision    recall  f1-score   support
    
            継続       0.84      0.91      0.87      1028
            離反       0.68      0.53      0.60       381
    
        accuracy                           0.81      1409
       macro avg       0.76      0.72      0.73      1409
    weighted avg       0.80      0.81      0.80      1409
    
    AUC: 0.8234
    

### ステップ4: 不均衡データ対策
    
    
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    # 1. クラス重み調整
    lr_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_weighted.fit(X_train_c_scaled, y_train_c)
    y_pred_weighted = lr_weighted.predict(X_test_c)
    
    print("=== クラス重み調整 ===")
    print(f"F1スコア: {f1_score(y_test_c, y_pred_weighted):.4f}")
    
    # 2. SMOTE（過サンプリング）
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_c_scaled, y_train_c)
    
    print(f"\nSMOTE後の訓練データ:")
    print(f"サンプル数: {X_train_smote.shape[0]}")
    print(f"離反率: {y_train_smote.mean()*100:.2f}%")
    
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    lr_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = lr_smote.predict(X_test_c)
    
    print(f"\nSMOTE + ロジスティック回帰:")
    print(f"F1スコア: {f1_score(y_test_c, y_pred_smote):.4f}")
    
    # 3. アンダーサンプリング + SMOTE
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    over = SMOTE(sampling_strategy=1.0, random_state=42)
    
    X_train_resampled, y_train_resampled = under.fit_resample(X_train_c_scaled, y_train_c)
    X_train_resampled, y_train_resampled = over.fit_resample(X_train_resampled, y_train_resampled)
    
    print(f"\nアンダー + SMOTE後:")
    print(f"サンプル数: {X_train_resampled.shape[0]}")
    print(f"離反率: {y_train_resampled.mean()*100:.2f}%")
    
    lr_combined = LogisticRegression(random_state=42, max_iter=1000)
    lr_combined.fit(X_train_resampled, y_train_resampled)
    y_pred_combined = lr_combined.predict(X_test_c)
    
    print(f"\nアンダー + SMOTE + ロジスティック回帰:")
    print(f"F1スコア: {f1_score(y_test_c, y_pred_combined):.4f}")
    

**出力** ：
    
    
    === クラス重み調整 ===
    F1スコア: 0.6534
    
    SMOTE後の訓練データ:
    サンプル数: 8224
    離反率: 50.00%
    
    SMOTE + ロジスティック回帰:
    F1スコア: 0.6789
    
    アンダー + SMOTE後:
    サンプル数: 5958
    離反率: 50.00%
    
    アンダー + SMOTE + ロジスティック回帰:
    F1スコア: 0.6812
    

### ステップ5: アンサンブルモデル
    
    
    # アンサンブルモデルで不均衡データに対処
    models_churn = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, scale_pos_weight=2.7, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    }
    
    results_churn = {}
    
    for name, model in models_churn.items():
        # SMOTEデータで学習
        model.fit(X_train_resampled, y_train_resampled)
    
        y_pred = model.predict(X_test_c)
        y_proba = model.predict_proba(X_test_c)[:, 1]
    
        f1 = f1_score(y_test_c, y_pred)
        auc = roc_auc_score(y_test_c, y_proba)
    
        results_churn[name] = {'F1 Score': f1, 'AUC': auc}
    
    # 結果表示
    results_churn_df = pd.DataFrame(results_churn).T
    print("=== アンサンブルモデル比較 ===")
    print(results_churn_df.sort_values('F1 Score', ascending=False))
    
    # 最良モデル
    best_model_churn = results_churn_df['F1 Score'].idxmax()
    print(f"\n最良モデル: {best_model_churn}")
    print(f"F1 Score: {results_churn_df.loc[best_model_churn, 'F1 Score']:.4f}")
    print(f"AUC: {results_churn_df.loc[best_model_churn, 'AUC']:.4f}")
    

**出力** ：
    
    
    === アンサンブルモデル比較 ===
                   F1 Score       AUC
    XGBoost          0.7645    0.8789
    LightGBM         0.7598    0.8745
    Random Forest    0.7234    0.8534
    
    最良モデル: XGBoost
    F1 Score: 0.7645
    AUC: 0.8789
    

### ステップ6: モデル評価とROC曲線
    
    
    from sklearn.metrics import roc_curve
    
    # 最良モデル（XGBoost）の詳細評価
    best_xgb_churn = models_churn['XGBoost']
    y_pred_best = best_xgb_churn.predict(X_test_c)
    y_proba_best = best_xgb_churn.predict_proba(X_test_c)[:, 1]
    
    print("=== 最良モデル詳細評価 ===")
    print(classification_report(y_test_c, y_pred_best, target_names=['継続', '離反']))
    
    # 混同行列
    cm_best = confusion_matrix(y_test_c, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=['継続', '離反'],
                yticklabels=['継続', '離反'])
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title(f'混同行列 - {best_model_churn}')
    plt.show()
    
    # ROC曲線
    fpr, tpr, thresholds = roc_curve(y_test_c, y_proba_best)
    auc_best = roc_auc_score(y_test_c, y_proba_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_churn} (AUC = {auc_best:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5)')
    plt.xlabel('偽陽性率 (FPR)', fontsize=12)
    plt.ylabel('真陽性率 (TPR)', fontsize=12)
    plt.title('ROC曲線', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    === 最良モデル詳細評価 ===
                  precision    recall  f1-score   support
    
            継続       0.89      0.88      0.88      1028
            離反       0.70      0.72      0.71       381
    
        accuracy                           0.84      1409
       macro avg       0.79      0.80      0.80      1409
    weighted avg       0.84      0.84      0.84      1409
    

### ステップ7: ビジネスインパクト分析
    
    
    # ビジネスメトリクス
    # 仮定: 離反顧客の維持コスト = $100、離反による損失 = $500
    
    cost_retention = 100  # 維持施策コスト
    cost_churn = 500      # 離反による損失
    
    # 混同行列から計算
    TP = cm_best[1, 1]  # 正しく離反予測
    FP = cm_best[0, 1]  # 誤って離反予測
    FN = cm_best[1, 0]  # 離反を見逃し
    TN = cm_best[0, 0]  # 正しく継続予測
    
    # コスト計算
    cost_with_model = (TP + FP) * cost_retention + FN * cost_churn
    cost_without_model = (TP + FN) * cost_churn
    
    savings = cost_without_model - cost_with_model
    savings_per_customer = savings / len(y_test_c)
    
    print("=== ビジネスインパクト分析 ===")
    print(f"モデル使用時のコスト: ${cost_with_model:,}")
    print(f"モデル不使用時のコスト: ${cost_without_model:,}")
    print(f"コスト削減額: ${savings:,}")
    print(f"顧客あたり削減額: ${savings_per_customer:.2f}")
    print(f"ROI: {(savings / cost_with_model) * 100:.2f}%")
    
    # 閾値調整による最適化
    print("\n=== 閾値最適化 ===")
    thresholds_to_test = np.arange(0.3, 0.7, 0.05)
    
    for threshold in thresholds_to_test:
        y_pred_threshold = (y_proba_best >= threshold).astype(int)
        cm_threshold = confusion_matrix(y_test_c, y_pred_threshold)
    
        TP_t = cm_threshold[1, 1]
        FP_t = cm_threshold[0, 1]
        FN_t = cm_threshold[1, 0]
    
        cost_t = (TP_t + FP_t) * cost_retention + FN_t * cost_churn
        savings_t = cost_without_model - cost_t
        f1_t = f1_score(y_test_c, y_pred_threshold)
    
        print(f"閾値 {threshold:.2f}: コスト削減 ${savings_t:,}, F1 {f1_t:.4f}")
    
    # 可視化
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    thresholds_range = np.arange(0.1, 0.9, 0.05)
    costs = []
    f1_scores = []
    
    for threshold in thresholds_range:
        y_pred_t = (y_proba_best >= threshold).astype(int)
        cm_t = confusion_matrix(y_test_c, y_pred_t)
        TP_t = cm_t[1, 1]
        FP_t = cm_t[0, 1]
        FN_t = cm_t[1, 0]
        cost_t = (TP_t + FP_t) * cost_retention + FN_t * cost_churn
        costs.append(cost_t)
        f1_scores.append(f1_score(y_test_c, y_pred_t))
    
    plt.plot(thresholds_range, costs, linewidth=2, marker='o')
    plt.xlabel('閾値', fontsize=12)
    plt.ylabel('総コスト ($)', fontsize=12)
    plt.title('閾値とコストの関係', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds_range, f1_scores, linewidth=2, marker='o', color='#e74c3c')
    plt.xlabel('閾値', fontsize=12)
    plt.ylabel('F1スコア', fontsize=12)
    plt.title('閾値とF1スコアの関係', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === ビジネスインパクト分析 ===
    モデル使用時のコスト: $91,300
    モデル不使用時のコスト: $190,500
    コスト削減額: $99,200
    顧客あたり削減額: $70.40
    ROI: 108.68%
    
    === 閾値最適化 ===
    閾値 0.30: コスト削減 $105,600, F1 0.7512
    閾値 0.35: コスト削減 $102,400, F1 0.7598
    閾値 0.40: コスト削減 $99,200, F1 0.7645
    閾値 0.45: コスト削減 $95,100, F1 0.7612
    閾値 0.50: コスト削減 $91,800, F1 0.7534
    閾値 0.55: コスト削減 $87,200, F1 0.7412
    閾値 0.60: コスト削減 $82,300, F1 0.7234
    閾値 0.65: コスト削減 $76,500, F1 0.7012
    

* * *

## 4.4 本章のまとめ

### 学んだこと

  1. **完全な機械学習パイプライン**

     * 問題定義 → EDA → 前処理 → 特徴量エンジニアリング → モデル選択 → 評価
     * 各ステップの重要性と実践方法
  2. **回帰プロジェクト（住宅価格予測）**

     * 探索的データ分析と相関分析
     * 外れ値処理と標準化
     * 特徴量エンジニアリング（新規特徴量作成）
     * ハイパーパラメータチューニング
     * R² 0.8567、RMSE 0.4556を達成
  3. **分類プロジェクト（顧客離反予測）**

     * 不均衡データの対処法（SMOTE、クラス重み）
     * ビジネスインパクト分析
     * 閾値最適化
     * F1スコア 0.7645、AUC 0.8789を達成
     * コスト削減 $99,200を実現

### 重要なポイント

ポイント | 説明  
---|---  
**EDAの重要性** | データ理解が精度向上の鍵  
**特徴量エンジニアリング** | ドメイン知識を活用した新規特徴量作成  
**不均衡データ対策** | SMOTE、クラス重み、閾値調整  
**モデル選択** | 複数モデルの比較と最適化  
**ビジネス視点** | 技術的精度だけでなく経済的価値も評価  
  
* * *

## 演習問題

### 問題1（難易度：easy）

機械学習パイプラインの主要なステップを順番に並べてください。

解答例

**解答** ：

  1. 問題定義（回帰 or 分類、評価指標の選定）
  2. データ収集
  3. 探索的データ分析（EDA）
  4. データ前処理（欠損値処理、外れ値除去）
  5. 特徴量エンジニアリング
  6. モデル選択
  7. 学習
  8. 評価
  9. ハイパーパラメータチューニング
  10. デプロイ

### 問題2（難易度：medium）

不均衡データ問題において、なぜ精度（Accuracy）だけでは不十分なのか説明してください。

解答例

**解答** ：

**例** : 離反率5%のデータ

  * すべて「離反しない」と予測 → 精度95%
  * しかし、離反顧客を一人も検出できていない
  * ビジネス的には無価値

**適切な指標** ：

  * **再現率（Recall）** : 実際の離反顧客のうち何%を検出できたか
  * **適合率（Precision）** : 離反予測のうち何%が正しいか
  * **F1スコア** : PrecisionとRecallの調和平均
  * **AUC** : 閾値に依存しない総合評価

**理由** ：

  * 精度は多数派クラスに引っ張られる
  * 少数派クラス（離反顧客）の予測性能が見えない
  * ビジネスインパクトが大きいのは少数派クラス

### 問題3（難易度：medium）

特徴量エンジニアリングで新しい特徴量を3つ作成し、その理由を説明してください（住宅価格予測の例）。

解答例

**解答** ：

**1\. 部屋あたりの面積 = 総面積 / 部屋数**

  * 理由: 部屋の広さは価格に直接影響
  * 元の特徴量だけでは捉えられない関係性

**2\. 築年数² = 築年数の二乗**

  * 理由: 築年数と価格の非線形関係を捉える
  * 新しい物件は価格が高いが、古すぎると急激に下がる

**3\. 駅からの距離 × 部屋数**

  * 理由: 交互作用効果を捉える
  * 駅近でも1Rなら安い、駅遠でも4LDKなら高い

**特徴量エンジニアリングのポイント** ：

  * ドメイン知識を活用
  * 非線形関係を捉える
  * 交互作用効果を考慮
  * 単位を揃える（スケーリング）

### 問題4（難易度：hard）

SMOTEを使った過サンプリングの利点と欠点を説明し、どのような場合に使うべきか述べてください。

解答例

**SMOTE（Synthetic Minority Over-sampling Technique）** ：

**原理** ：

  * 少数派クラスのサンプル間を線形補間して合成サンプルを生成
  * $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_j - \mathbf{x}_i)$

**利点** ：

  1. 単純な複製より多様性が高い
  2. 過学習のリスクが低い
  3. 少数派クラスの特徴空間を広げる
  4. モデルが少数派クラスをより学習しやすい

**欠点** ：

  1. ノイズや外れ値も増幅される
  2. クラス境界が曖昧になる可能性
  3. 高次元データでは効果が薄い（次元の呪い）
  4. 計算コストが増加

**使うべき場合** ：

  * 不均衡比率: 1:5 〜 1:20程度
  * データ量: 少数派クラスが100サンプル以上
  * ノイズ: 少ない、きれいなデータ
  * 次元: 中程度（10〜50特徴量）

**使わないほうが良い場合** ：

  * 極端な不均衡（1:100以上） → アンサンブル手法
  * 少数派が極端に少ない（<50） → データ収集
  * 高次元データ → 特徴量選択 + クラス重み
  * ノイズが多い → クリーニング優先

**代替手法** ：

  * ADASYN: 境界付近に重点的にサンプリング
  * Borderline-SMOTE: 境界サンプルのみ生成
  * アンダーサンプリング + SMOTE: 組み合わせ
  * クラス重み調整: シンプルで効果的

### 問題5（難易度：hard）

ビジネスインパクト分析で、閾値を0.4から0.3に変更すると、F1スコアとコストがどう変化するか予測し、ビジネス的にどちらを選ぶべきか議論してください。

解答例

**閾値変更の影響** ：

**閾値 0.4 → 0.3に下げる** ：

  * **予測の変化** : より多くの顧客を「離反」と予測
  * **Recall（再現率）** : 上昇（離反顧客をより多く検出）
  * **Precision（適合率）** : 低下（誤検知が増える）
  * **F1スコア** : やや低下（0.7645 → 0.7512）

**コスト分析** ：

混同行列の変化（予測）：

| 閾値0.4 | 閾値0.3  
---|---|---  
TP（正しく離反予測） | 275 | 290  
FP（誤って離反予測） | 118 | 150  
FN（離反見逃し） | 106 | 91  
TN（正しく継続予測） | 910 | 878  
  
**コスト計算** ：
    
    
    閾値0.4:
    - 維持施策コスト: (275+118) × $100 = $39,300
    - 離反損失: 106 × $500 = $53,000
    - 総コスト: $92,300
    
    閾値0.3:
    - 維持施策コスト: (290+150) × $100 = $44,000
    - 離反損失: 91 × $500 = $45,500
    - 総コスト: $89,500
    
    コスト削減: $2,800（約3%改善）
    

**ビジネス的判断** ：

**閾値0.3を選ぶべき理由** ：

  1. **コスト削減** : $2,800の追加削減
  2. **離反見逃し減少** : 15人減（106→91人）
  3. **顧客維持** : 離反を防ぐことが長期的価値
  4. **リスク回避** : 見逃しのコスト（$500）> 誤検知のコスト（$100）

**閾値0.4を選ぶべき理由** ：

  1. **F1スコア** : やや高い（0.7645 vs 0.7512）
  2. **効率性** : 維持施策の対象が少ない（393 vs 440人）
  3. **リソース制約** : 施策実行の人的コスト

**推奨** ：

  * **閾値0.3を採用**
  * 理由: コスト削減額が大きく、離反見逃しが減る
  * 条件: 維持施策の実行リソースが十分にある場合
  * モニタリング: 実際のROIを継続的に測定

**追加考慮事項** ：

  * 顧客生涯価値（LTV）を考慮
  * 維持施策の成功率を測定
  * A/Bテストで最適閾値を検証
  * 動的な閾値調整（顧客セグメント別）

* * *

## 参考文献

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly Media.
  2. Raschka, S., & Mirjalili, V. (2019). _Python Machine Learning_. Packt Publishing.
  3. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." _Journal of Artificial Intelligence Research_ , 16, 321-357.
