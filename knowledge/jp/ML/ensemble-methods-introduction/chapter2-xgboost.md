---
title: 第2章：XGBoost
chapter_title: 第2章：XGBoost
subtitle: 勾配ブースティングの最適化実装 - 高速で正確な予測モデル
reading_time: 25-30分
difficulty: 中級
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ XGBoostの原理と正則化技術を理解する
  * ✅ ハイパーパラメータの役割と調整方法を習得する
  * ✅ xgboostライブラリで効率的な訓練を実行できる
  * ✅ 特徴量重要度を解釈し、モデルを改善できる
  * ✅ GPU加速とパラメータチューニング戦略を実践できる

* * *

## 2.1 XGBoostの原理

### XGBoostとは

**XGBoost（eXtreme Gradient Boosting）** は、勾配ブースティング決定木の高速で効率的な実装です。

> 「XGBoostは多くのKaggleコンペティションで優勝モデルとして使用されています。」

### 主要な特徴

特徴 | 説明 | 利点  
---|---|---  
**正則化** | L1/L2正則化による過学習防止 | 汎化性能向上  
**木の剪定** | max_depth後の枝刈り | 効率的な学習  
**並列処理** | 列（特徴量）単位の並列化 | 高速な訓練  
**欠損値処理** | 自動で最適な方向を学習 | 前処理不要  
  
### XGBoostの目的関数

XGBoostは以下の目的関数を最小化します：

$$ \mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k) $$

  * $l$: 損失関数（二乗誤差、対数損失など）
  * $\Omega(f_k)$: 正則化項（木の複雑さペナルティ）

正則化項：

$$ \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 $$

  * $T$: 葉の数
  * $w_j$: 葉の重み
  * $\gamma$: 葉数のペナルティ
  * $\lambda$: L2正則化係数

### 基本的な実装
    
    
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # サンプルデータ生成
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # XGBoostモデルの構築
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    print("=== XGBoostの基本性能 ===")
    print(f"精度: {accuracy:.3f}")
    print(f"\n詳細レポート:")
    print(classification_report(y_test, y_pred))
    

**出力** ：
    
    
    === XGBoostの基本性能 ===
    精度: 0.935
    
    詳細レポート:
                  precision    recall  f1-score   support
    
               0       0.94      0.93      0.93       102
               1       0.93      0.94      0.93        98
    
        accuracy                           0.93       200
       macro avg       0.93      0.93      0.93       200
    weighted avg       0.93      0.93      0.93       200
    

* * *

## 2.2 ハイパーパラメータ

### 主要なハイパーパラメータ

パラメータ | 説明 | 推奨範囲 | デフォルト  
---|---|---|---  
**learning_rate (eta)** | 学習率、各木の寄与を縮小 | 0.01 - 0.3 | 0.3  
**max_depth** | 木の最大深さ | 3 - 10 | 6  
**n_estimators** | 木の数 | 100 - 1000 | 100  
**subsample** | 各木で使用する行の割合 | 0.5 - 1.0 | 1.0  
**colsample_bytree** | 各木で使用する列の割合 | 0.5 - 1.0 | 1.0  
**gamma** | 分岐の最小損失減少 | 0 - 5 | 0  
**reg_alpha** | L1正則化（重みの絶対値） | 0 - 1 | 0  
**reg_lambda** | L2正則化（重みの二乗） | 0 - 1 | 1  
  
### パラメータの影響を可視化
    
    
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    
    # データ準備
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    # learning_rateの影響
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    lr_scores = []
    
    for lr in learning_rates:
        model = xgb.XGBClassifier(
            learning_rate=lr,
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        lr_scores.append(scores.mean())
    
    # max_depthの影響
    max_depths = [2, 3, 4, 5, 6, 8, 10]
    depth_scores = []
    
    for depth in max_depths:
        model = xgb.XGBClassifier(
            max_depth=depth,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        depth_scores.append(scores.mean())
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(learning_rates, lr_scores, marker='o', linewidth=2)
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Cross-Validation Accuracy', fontsize=12)
    axes[0].set_title('Learning Rate vs Accuracy', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(max_depths, depth_scores, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Max Depth', fontsize=12)
    axes[1].set_ylabel('Cross-Validation Accuracy', fontsize=12)
    axes[1].set_title('Max Depth vs Accuracy', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== パラメータ調整結果 ===")
    print(f"最適learning_rate: {learning_rates[np.argmax(lr_scores)]}")
    print(f"最適max_depth: {max_depths[np.argmax(depth_scores)]}")
    

### subsampleとcolsampleの効果
    
    
    # subsampleとcolsample_bytreeの組み合わせ
    subsample_values = [0.5, 0.7, 0.9, 1.0]
    colsample_values = [0.5, 0.7, 0.9, 1.0]
    
    results = []
    
    for sub in subsample_values:
        for col in colsample_values:
            model = xgb.XGBClassifier(
                subsample=sub,
                colsample_bytree=col,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results.append({
                'subsample': sub,
                'colsample': col,
                'accuracy': scores.mean()
            })
    
    # 結果の可視化
    df_results = pd.DataFrame(results)
    pivot_table = df_results.pivot(
        index='subsample',
        columns='colsample',
        values='accuracy'
    )
    
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Subsample vs Colsample_bytree', fontsize=14)
    plt.xlabel('Colsample_bytree', fontsize=12)
    plt.ylabel('Subsample', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print("\n=== 最適な組み合わせ ===")
    best_idx = df_results['accuracy'].idxmax()
    print(f"subsample: {df_results.loc[best_idx, 'subsample']}")
    print(f"colsample_bytree: {df_results.loc[best_idx, 'colsample']}")
    print(f"精度: {df_results.loc[best_idx, 'accuracy']:.3f}")
    

* * *

## 2.3 実装と訓練

### DMatrix形式での効率的な訓練
    
    
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # データ読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # DMatrix形式に変換（高速化）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # パラメータ設定
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }
    
    # 訓練（評価セット付き）
    evals = [(dtrain, 'train'), (dtest, 'test')]
    num_rounds = 100
    
    print("=== XGBoost訓練（DMatrix形式）===")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    print(f"\n最適な反復回数: {model.best_iteration}")
    print(f"最良スコア: {model.best_score:.4f}")
    

### Early Stoppingによる過学習防止
    
    
    from sklearn.model_selection import train_test_split
    
    # 検証セットを含む分割
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # XGBoostモデル（sklearn API）
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    # Early Stopping付き訓練
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=20,
        verbose=50
    )
    
    # テストセットでの評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=== Early Stopping結果 ===")
    print(f"使用された木の数: {model.best_iteration}")
    print(f"検証精度: {accuracy:.3f}")
    
    # 学習曲線の可視化
    results = model.evals_result()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['validation_0']['logloss'], label='Train')
    plt.plot(results['validation_1']['logloss'], label='Validation')
    plt.axvline(x=model.best_iteration, color='r', linestyle='--',
                label=f'Best iteration: {model.best_iteration}')
    plt.xlabel('Number of Trees')
    plt.ylabel('Log Loss')
    plt.title('Learning Curve with Early Stopping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['validation_0']['logloss'][-100:], label='Train (last 100)')
    plt.plot(results['validation_1']['logloss'][-100:], label='Validation (last 100)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Log Loss')
    plt.title('Learning Curve (Detail)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Cross-Validationによる性能評価
    
    
    import xgboost as xgb
    
    # DMatrix形式でのデータ
    dtrain = xgb.DMatrix(X, label=y)
    
    # パラメータ設定
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }
    
    # Cross-Validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=200,
        nfold=5,
        metrics='logloss',
        early_stopping_rounds=20,
        seed=42,
        verbose_eval=50
    )
    
    print("\n=== Cross-Validation結果 ===")
    print(f"最適な反復回数: {len(cv_results)}")
    print(f"訓練 log loss: {cv_results['train-logloss-mean'].iloc[-1]:.4f}")
    print(f"検証 log loss: {cv_results['test-logloss-mean'].iloc[-1]:.4f}")
    print(f"標準偏差: {cv_results['test-logloss-std'].iloc[-1]:.4f}")
    
    # CV結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results['train-logloss-mean'], label='Train')
    plt.plot(cv_results['test-logloss-mean'], label='Test')
    plt.fill_between(
        range(len(cv_results)),
        cv_results['test-logloss-mean'] - cv_results['test-logloss-std'],
        cv_results['test-logloss-mean'] + cv_results['test-logloss-std'],
        alpha=0.2
    )
    plt.xlabel('Number of Trees')
    plt.ylabel('Log Loss')
    plt.title('Cross-Validation Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.4 特徴量重要度

### 重要度の種類

タイプ | 説明 | 解釈  
---|---|---  
**gain** | その特徴量による損失の平均改善量 | 予測精度への寄与  
**weight** | 特徴量が分岐に使用された回数 | 使用頻度  
**cover** | 分岐でカバーされたサンプル数 | 影響範囲  
  
### 特徴量重要度の計算と可視化
    
    
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    import matplotlib.pyplot as plt
    
    # データ読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # モデル訓練
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # 3つの重要度タイプを取得
    importance_gain = model.get_booster().get_score(importance_type='gain')
    importance_weight = model.get_booster().get_score(importance_type='weight')
    importance_cover = model.get_booster().get_score(importance_type='cover')
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gain
    xgb.plot_importance(model, importance_type='gain', max_num_features=10,
                        ax=axes[0], title='Feature Importance (Gain)')
    axes[0].set_xlabel('Gain')
    
    # Weight
    xgb.plot_importance(model, importance_type='weight', max_num_features=10,
                        ax=axes[1], title='Feature Importance (Weight)')
    axes[1].set_xlabel('Weight')
    
    # Cover
    xgb.plot_importance(model, importance_type='cover', max_num_features=10,
                        ax=axes[2], title='Feature Importance (Cover)')
    axes[2].set_xlabel('Cover')
    
    plt.tight_layout()
    plt.show()
    
    # 数値で確認
    print("=== Top 10特徴量（Gain基準）===")
    importance_dict = model.get_booster().get_score(importance_type='gain')
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_features[:10], 1):
        feature_idx = int(feature.replace('f', ''))
        print(f"{i}. {feature_names[feature_idx]}: {score:.2f}")
    

### SHAP値による詳細分析
    
    
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt
    
    # モデル訓練
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False)
    plt.title('SHAP Summary Plot', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # SHAP Feature Importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title('SHAP Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("=== SHAP値による重要度分析 ===")
    print("各特徴量の平均絶対SHAP値（Top 5）:")
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[-5:][::-1]
    for idx in top_indices:
        print(f"{feature_names[idx]}: {shap_importance[idx]:.4f}")
    

* * *

## 2.5 実践最適化

### GPU加速
    
    
    import xgboost as xgb
    import time
    
    # データ準備（大規模データ）
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_informative=40,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CPU訓練
    start_time = time.time()
    model_cpu = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',  # CPUでの高速化
        random_state=42
    )
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start_time
    
    # GPU訓練（GPUが利用可能な場合）
    try:
        start_time = time.time()
        model_gpu = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method='gpu_hist',  # GPU加速
            random_state=42
        )
        model_gpu.fit(X_train, y_train)
        gpu_time = time.time() - start_time
    
        print("=== GPU vs CPU性能比較 ===")
        print(f"CPU訓練時間: {cpu_time:.2f}秒")
        print(f"GPU訓練時間: {gpu_time:.2f}秒")
        print(f"速度向上: {cpu_time/gpu_time:.2f}x")
    except Exception as e:
        print("=== CPU訓練結果 ===")
        print(f"訓練時間: {cpu_time:.2f}秒")
        print(f"GPU利用不可: {str(e)}")
    
    # 性能評価
    y_pred = model_cpu.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nテスト精度: {accuracy:.3f}")
    

### グリッドサーチによるハイパーパラメータ最適化
    
    
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    
    # データ準備
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # パラメータグリッド
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    }
    
    # グリッドサーチ
    model = xgb.XGBClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    print("=== グリッドサーチ実行中 ===")
    grid_search.fit(X_train, y_train)
    
    # 最適パラメータ
    print("\n=== 最適パラメータ ===")
    print(grid_search.best_params_)
    print(f"\n最良CV精度: {grid_search.best_score_:.3f}")
    
    # テストセットでの評価
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"テスト精度: {test_accuracy:.3f}")
    
    # 上位5つの組み合わせ
    cv_results = pd.DataFrame(grid_search.cv_results_)
    top_5 = cv_results.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    print("\n=== Top 5パラメータ組み合わせ ===")
    for i, row in top_5.iterrows():
        print(f"\n{i+1}. スコア: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"   パラメータ: {row['params']}")
    

### ベイズ最適化による効率的な探索
    
    
    from sklearn.model_selection import cross_val_score
    from scipy.stats import uniform, randint
    
    # ランダムサーチ（ベイズ最適化の代替）
    from sklearn.model_selection import RandomizedSearchCV
    
    # パラメータ分布
    param_distributions = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),
        'n_estimators': randint(50, 300),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    # ランダムサーチ
    model = xgb.XGBClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=50,  # 試行回数
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("=== ランダムサーチ実行中 ===")
    random_search.fit(X_train, y_train)
    
    # 最適パラメータ
    print("\n=== 最適パラメータ（ランダムサーチ）===")
    print(random_search.best_params_)
    print(f"\n最良CV精度: {random_search.best_score_:.3f}")
    
    # テスト精度
    y_pred = random_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"テスト精度: {test_accuracy:.3f}")
    
    # 探索過程の可視化
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(cv_results)), cv_results['mean_test_score'],
                alpha=0.6, edgecolors='black')
    plt.axhline(y=random_search.best_score_, color='r', linestyle='--',
                label=f'Best: {random_search.best_score_:.3f}')
    plt.xlabel('Iteration')
    plt.ylabel('CV Score')
    plt.title('Random Search Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_scores = sorted(cv_results['mean_test_score'])
    plt.plot(sorted_scores, linewidth=2)
    plt.xlabel('Rank')
    plt.ylabel('CV Score')
    plt.title('Sorted CV Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **XGBoostの原理**

     * 勾配ブースティングの最適化実装
     * 正則化による過学習防止
     * 効率的な木の剪定アルゴリズム
  2. **ハイパーパラメータ**

     * learning_rate: 学習の安定性と速度
     * max_depth: モデルの複雑さ制御
     * subsample/colsample: ランダム性導入
     * gamma, reg_alpha, reg_lambda: 正則化
  3. **実装と訓練**

     * DMatrix形式での効率的な処理
     * Early Stoppingによる自動最適化
     * Cross-Validationでの堅牢な評価
  4. **特徴量重要度**

     * gain, weight, coverの3種類
     * SHAP値による詳細な解釈
     * モデル改善への活用
  5. **実践最適化**

     * GPU加速による高速化
     * グリッドサーチとランダムサーチ
     * 効率的なパラメータ探索戦略

### XGBoost活用のベストプラクティス

原則 | 説明  
---|---  
**小さいlearning_rate** | 0.01-0.1で安定した学習、n_estimatorsを増やす  
**Early Stopping活用** | 過学習防止と訓練時間短縮  
**subsample導入** | 0.7-0.9でランダム性と汎化性能向上  
**特徴量重要度確認** | 不要な特徴量削除でモデル簡素化  
**CV評価** | 堅牢な性能評価とパラメータ選択  
  
### 次の章へ

第3章では、**LightGBM** を学びます：

  * Leaf-wiseアルゴリズム
  * カテゴリカル変数の直接処理
  * 超高速訓練の実現
  * XGBoostとの比較

* * *

## 演習問題

### 問題1（難易度：easy）

XGBoostの正則化項 $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ において、各項の意味を説明してください。

解答例

**解答** ：

  * **$\gamma T$** : 葉の数に対するペナルティ

    * $\gamma$: 葉数のペナルティ係数
    * $T$: 木の葉の数
    * 効果: 葉が多いほどペナルティが増加し、木が複雑になりすぎるのを防ぐ
  * **$\frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$** : 葉の重みに対するL2正則化

    * $\lambda$: L2正則化係数
    * $w_j$: 各葉の重み（予測値）
    * 効果: 重みが大きくなりすぎるのを防ぎ、滑らかな予測を促進

これらの正則化により、XGBoostは過学習を防ぎ、汎化性能を向上させます。

### 問題2（難易度：medium）

以下のデータに対してXGBoostモデルを訓練し、Early Stoppingを使用して最適な木の数を見つけてください。
    
    
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

解答例
    
    
    import xgboost as xgb
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # データ読み込み
    data = load_diabetes()
    X, y = data.data, data.target
    
    # データ分割（訓練、検証、テスト）
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # XGBoostモデル（回帰）
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    
    # Early Stopping付き訓練
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    # 結果
    print("\n=== Early Stopping結果 ===")
    print(f"最適な木の数: {model.best_iteration}")
    print(f"訓練時のRMSE: {model.best_score:.2f}")
    
    # テストセットでの評価
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nテストセット性能:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"R²: {r2:.3f}")
    
    # 学習曲線の可視化
    import matplotlib.pyplot as plt
    
    results = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['rmse'], label='Train')
    plt.plot(results['validation_1']['rmse'], label='Validation')
    plt.axvline(x=model.best_iteration, color='r', linestyle='--',
                label=f'Best: {model.best_iteration}')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('Early Stopping - Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 問題3（難易度：medium）

learning_rateとn_estimatorsの関係を説明し、最適な組み合わせを選ぶ戦略を述べてください。

解答例

**解答** ：

**learning_rateとn_estimatorsの関係** ：

  * **learning_rate（学習率）** : 各木の寄与を縮小する係数 
    * 小さい値（0.01-0.05）: 安定した学習、過学習しにくい
    * 大きい値（0.2-0.3）: 高速な学習、過学習リスク
  * **n_estimators（木の数）** : アンサンブルする木の総数 
    * 多い（500-1000）: 複雑なパターン学習、計算コスト大
    * 少ない（50-100）: 高速、単純なモデル

**トレードオフ** ：

  * learning_rateを小さくすると、同じ性能に到達するためにn_estimatorsを増やす必要がある
  * 目安: `learning_rate × n_estimators ≈ 一定`

**最適化戦略** ：

  1. **初期探索** （速度重視）

     * learning_rate = 0.1, n_estimators = 100
     * ベースライン性能を確認
  2. **Early Stopping活用**

     * learning_rate = 0.05, n_estimators = 1000（大きめ）
     * Early Stoppingで最適な木の数を自動決定
  3. **精密調整** （精度重視）

     * learning_rate = 0.01, n_estimators = 500-2000
     * 最終モデルで最高精度を追求
  4. **本番環境** （バランス）

     * learning_rate = 0.03-0.05
     * Early Stoppingで決定されたn_estimators使用

**実装例** ：
    
    
    # 戦略1: 速度重視
    model_fast = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100)
    
    # 戦略2: Early Stopping（推奨）
    model_auto = xgb.XGBClassifier(learning_rate=0.05, n_estimators=1000)
    model_auto.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   early_stopping_rounds=50)
    
    # 戦略3: 精度重視
    model_accurate = xgb.XGBClassifier(learning_rate=0.01, n_estimators=1500)
    

### 問題4（難易度：hard）

以下のデータに対してグリッドサーチを実行し、最適なハイパーパラメータを見つけてください。探索すべきパラメータとその範囲も提案してください。
    
    
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=5000, n_features=30, n_informative=20, random_state=42)
    

解答例
    
    
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    
    # データ生成
    X, y = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # パラメータグリッド（段階的探索）
    # フェーズ1: 主要パラメータの粗探索
    param_grid_phase1 = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = xgb.XGBClassifier(random_state=42)
    
    print("=== フェーズ1: 粗探索 ===")
    grid_search_phase1 = GridSearchCV(
        model,
        param_grid_phase1,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_phase1.fit(X_train, y_train)
    print(f"\nフェーズ1最適パラメータ: {grid_search_phase1.best_params_}")
    print(f"フェーズ1最良CV精度: {grid_search_phase1.best_score_:.4f}")
    
    # フェーズ2: 正則化パラメータの精密調整
    best_params = grid_search_phase1.best_params_
    
    param_grid_phase2 = {
        'max_depth': [best_params['max_depth']],
        'learning_rate': [best_params['learning_rate']],
        'n_estimators': [best_params['n_estimators']],
        'subsample': [best_params['subsample']],
        'colsample_bytree': [best_params['colsample_bytree']],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    print("\n=== フェーズ2: 正則化調整 ===")
    grid_search_phase2 = GridSearchCV(
        model,
        param_grid_phase2,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_phase2.fit(X_train, y_train)
    print(f"\nフェーズ2最適パラメータ: {grid_search_phase2.best_params_}")
    print(f"フェーズ2最良CV精度: {grid_search_phase2.best_score_:.4f}")
    
    # テストセットでの最終評価
    final_model = grid_search_phase2.best_estimator_
    y_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=== 最終モデル性能 ===")
    print(f"テスト精度: {test_accuracy:.4f}")
    print(f"\n分類レポート:")
    print(classification_report(y_test, y_pred))
    
    # 探索結果の分析
    cv_results = pd.DataFrame(grid_search_phase2.cv_results_)
    top_10 = cv_results.nlargest(10, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    
    print("\n=== Top 10パラメータ組み合わせ ===")
    for i, row in top_10.iterrows():
        print(f"\n{i+1}. CV精度: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"   gamma: {row['params']['gamma']}")
        print(f"   reg_alpha: {row['params']['reg_alpha']}")
        print(f"   reg_lambda: {row['params']['reg_lambda']}")
    
    # 可視化: パラメータの影響
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, param in zip(axes, ['gamma', 'reg_alpha', 'reg_lambda']):
        param_values = cv_results['param_' + param].values
        scores = cv_results['mean_test_score'].values
    
        ax.scatter(param_values, scores, alpha=0.6, edgecolors='black')
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('CV Accuracy', fontsize=12)
        ax.set_title(f'Impact of {param}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**推奨パラメータ範囲** ：

パラメータ | 初期探索 | 精密調整  
---|---|---  
max_depth | 3, 5, 7 | ±1範囲  
learning_rate | 0.01, 0.1, 0.3 | 最適値周辺  
n_estimators | 100, 200 | Early Stopping推奨  
subsample | 0.8, 1.0 | 0.7-1.0  
colsample_bytree | 0.8, 1.0 | 0.7-1.0  
gamma | - | 0, 0.1, 0.5, 1  
reg_alpha | - | 0, 0.1, 0.5  
reg_lambda | - | 1, 1.5, 2  
  
### 問題5（難易度：hard）

XGBoostにおける特徴量重要度のgain, weight, coverの違いを説明し、それぞれをどのような場面で使用すべきか述べてください。

解答例

**解答** ：

**3つの重要度指標** ：

  1. **Gain（利得）**

     * 定義: その特徴量による損失関数の平均改善量
     * 計算: 分岐時の損失減少の合計を、その特徴量の使用回数で割った値
     * 意味: 予測精度への直接的な寄与度
     * 数式: $\text{Gain}_f = \sum_{t \in \text{splits}(f)} \Delta L_t / |\text{splits}(f)|$
  2. **Weight（重み）**

     * 定義: その特徴量が分岐に使用された回数
     * 計算: 全ての木でその特徴量が分岐に使われた総回数
     * 意味: モデル構築における使用頻度
     * 数式: $\text{Weight}_f = |\text{splits}(f)|$
  3. **Cover（カバレッジ）**

     * 定義: 分岐でカバーされたサンプル数の合計
     * 計算: その特徴量で分岐した際に影響を受けたサンプル数の合計
     * 意味: 特徴量が影響するデータの範囲
     * 数式: $\text{Cover}_f = \sum_{t \in \text{splits}(f)} n_t$（$n_t$は分岐のサンプル数）

**使い分けガイドライン** ：

場面 | 推奨指標 | 理由  
---|---|---  
特徴選択 | Gain | 精度への寄与が直接的にわかる  
モデル解釈 | Gain | ビジネス価値を説明しやすい  
計算効率化 | Weight | 頻繁に使う特徴を優先的に計算  
データ影響範囲 | Cover | どれだけのサンプルに影響するか  
不均衡データ | Cover | 少数クラスへの影響を評価  
一般的な用途 | Gain | 最も解釈しやすく実用的  
  
**実装例** ：
    
    
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    import matplotlib.pyplot as plt
    
    # データ読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # モデル訓練
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # 3つの重要度を取得
    importance_gain = model.get_booster().get_score(importance_type='gain')
    importance_weight = model.get_booster().get_score(importance_type='weight')
    importance_cover = model.get_booster().get_score(importance_type='cover')
    
    # Top 5を比較
    print("=== Top 5特徴量の比較 ===\n")
    
    for imp_type, imp_dict in [('Gain', importance_gain),
                                ('Weight', importance_weight),
                                ('Cover', importance_cover)]:
        print(f"{imp_type}基準:")
        sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features[:5], 1):
            feature_idx = int(feature.replace('f', ''))
            print(f"  {i}. {data.feature_names[feature_idx]}: {score:.2f}")
        print()
    
    # 可視化: 3つの重要度の相関
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gain vs Weight
    common_features = set(importance_gain.keys()) & set(importance_weight.keys())
    gain_vals = [importance_gain[f] for f in common_features]
    weight_vals = [importance_weight[f] for f in common_features]
    
    axes[0].scatter(weight_vals, gain_vals, alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Gain')
    axes[0].set_title('Gain vs Weight')
    axes[0].grid(True, alpha=0.3)
    
    # Gain vs Cover
    cover_vals = [importance_cover[f] for f in common_features]
    
    axes[1].scatter(cover_vals, gain_vals, alpha=0.6, edgecolors='black', color='orange')
    axes[1].set_xlabel('Cover')
    axes[1].set_ylabel('Gain')
    axes[1].set_title('Gain vs Cover')
    axes[1].grid(True, alpha=0.3)
    
    # Weight vs Cover
    axes[2].scatter(weight_vals, cover_vals, alpha=0.6, edgecolors='black', color='green')
    axes[2].set_xlabel('Weight')
    axes[2].set_ylabel('Cover')
    axes[2].set_title('Weight vs Cover')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**結論** ：

  * 一般的には**Gain** を使用（予測精度への寄与が明確）
  * 計算効率やデータ構造の分析には**Weight** や**Cover** も有用
  * 複数の指標を組み合わせて総合的に判断することが重要

* * *

## 参考文献

  1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_.
  2. Chen, T., He, T., Benesty, M., et al. (2023). _XGBoost Documentation_. https://xgboost.readthedocs.io/
  3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_ (2nd ed.). Springer.
  4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. _Advances in Neural Information Processing Systems_.
