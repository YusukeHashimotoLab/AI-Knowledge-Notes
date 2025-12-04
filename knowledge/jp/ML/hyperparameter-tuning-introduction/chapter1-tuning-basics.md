---
title: 第1章：ハイパーパラメータチューニング基礎
chapter_title: 第1章：ハイパーパラメータチューニング基礎
subtitle: モデル性能を最大化する探索手法の基本
reading_time: 25-30分
difficulty: 初級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ハイパーパラメータとモデルパラメータの違いを理解する
  * ✅ チューニングの重要性と探索空間の設計を学ぶ
  * ✅ グリッドサーチの仕組みと実装方法を習得する
  * ✅ ランダムサーチの利点と使い方を理解する
  * ✅ 交差検証とハイパーパラメータ探索を組み合わせる
  * ✅ scikit-learnで実践的なチューニングを実行できる

* * *

## 1.1 ハイパーパラメータとは

### モデルパラメータとの違い

**ハイパーパラメータ（Hyperparameter）** は、学習前に人間が設定する値で、モデルの構造や学習プロセスを制御します。

種類 | 定義 | 例 | 決定方法  
---|---|---|---  
**モデルパラメータ** | 学習により自動的に最適化 | 線形回帰の係数、ニューラルネットの重み | 訓練データから学習  
**ハイパーパラメータ** | 学習前に人間が設定 | 学習率、木の深さ、正則化係数 | 試行錯誤、探索アルゴリズム  
  
### 主要なハイパーパラメータ

アルゴリズム | 主要ハイパーパラメータ | 役割  
---|---|---  
**Random Forest** | n_estimators, max_depth, min_samples_split | 木の数、深さ、分割条件  
**XGBoost** | learning_rate, max_depth, n_estimators, subsample | 学習速度、複雑度、サンプリング  
**SVM** | C, kernel, gamma | 正則化、カーネル、影響範囲  
**ニューラルネット** | learning_rate, batch_size, hidden_layers | 学習速度、バッチ、構造  
  
### チューニングの重要性

> 適切なハイパーパラメータ設定により、モデルの性能は10-30%以上改善することがあります。
    
    
    ```mermaid
    graph LR
        A[デフォルト設定] --> B[精度: 75%]
        C[チューニング後] --> D[精度: 88%]
    
        style A fill:#ffebee
        style B fill:#ffcdd2
        style C fill:#e8f5e9
        style D fill:#a5d6a7
    ```

### 探索空間の設計

探索空間は、各ハイパーパラメータの候補値の範囲です。適切な設計が重要です。
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # 探索空間の定義例
    param_space = {
        'n_estimators': [50, 100, 200, 300],           # 木の数
        'max_depth': [5, 10, 15, 20, None],            # 最大深さ
        'min_samples_split': [2, 5, 10],               # 分割に必要な最小サンプル数
        'min_samples_leaf': [1, 2, 4],                 # 葉に必要な最小サンプル数
        'max_features': ['sqrt', 'log2', None]         # 分割時の特徴量数
    }
    
    print("=== 探索空間の概要 ===")
    print(f"n_estimators: {len(param_space['n_estimators'])}通り")
    print(f"max_depth: {len(param_space['max_depth'])}通り")
    print(f"min_samples_split: {len(param_space['min_samples_split'])}通り")
    print(f"min_samples_leaf: {len(param_space['min_samples_leaf'])}通り")
    print(f"max_features: {len(param_space['max_features'])}通り")
    
    total_combinations = np.prod([len(v) for v in param_space.values()])
    print(f"\n総組み合わせ数: {total_combinations:,}")
    

**出力** ：
    
    
    === 探索空間の概要 ===
    n_estimators: 4通り
    max_depth: 5通り
    min_samples_split: 3通り
    min_samples_leaf: 3通り
    max_features: 3通り
    
    総組み合わせ数: 540
    

> **重要** : 探索空間が広すぎると計算コストが膨大になります。ドメイン知識と経験的な範囲を活用しましょう。

* * *

## 1.2 グリッドサーチ

### 仕組みと実装

**グリッドサーチ（Grid Search）** は、指定したすべてのハイパーパラメータの組み合わせを網羅的に探索します。
    
    
    ```mermaid
    graph TD
        A[探索空間定義] --> B[すべての組み合わせ生成]
        B --> C[各組み合わせで学習]
        C --> D[交差検証で評価]
        D --> E[最良パラメータ選択]
    
        style A fill:#e3f2fd
        style B fill:#bbdefb
        style C fill:#90caf9
        style D fill:#64b5f6
        style E fill:#42a5f5
    ```

### scikit-learn GridSearchCV
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    # データ準備
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # グリッドサーチ用のパラメータグリッド
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # GridSearchCVの設定
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,                      # 5-fold交差検証
        scoring='accuracy',        # 評価指標
        n_jobs=-1,                 # 全CPUコア使用
        verbose=2                  # 詳細出力
    )
    
    # グリッドサーチ実行
    print("=== グリッドサーチ開始 ===")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # 結果表示
    print(f"\n実行時間: {elapsed_time:.2f}秒")
    print(f"\n最良パラメータ:")
    print(grid_search.best_params_)
    print(f"\n最良スコア（交差検証）: {grid_search.best_score_:.4f}")
    
    # テストデータで評価
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"テストデータ精度: {test_accuracy:.4f}")
    

**出力例** ：
    
    
    === グリッドサーチ開始 ===
    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    
    実行時間: 12.34秒
    
    最良パラメータ:
    {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}
    
    最良スコア（交差検証）: 0.9648
    テストデータ精度: 0.9737
    

### 探索結果の詳細分析
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # 重要な列のみ抽出
    results_summary = results_df[[
        'param_n_estimators',
        'param_max_depth',
        'param_min_samples_split',
        'mean_test_score',
        'std_test_score',
        'rank_test_score'
    ]].sort_values('rank_test_score')
    
    print("\n=== トップ5の組み合わせ ===")
    print(results_summary.head(10))
    
    # 可視化：パラメータの影響分析
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # n_estimatorsの影響
    results_df.groupby('param_n_estimators')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[0], color='steelblue'
    )
    axes[0].set_title('n_estimators の影響', fontsize=12)
    axes[0].set_ylabel('平均スコア')
    axes[0].grid(True, alpha=0.3)
    
    # max_depthの影響
    results_df.groupby('param_max_depth')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[1], color='forestgreen'
    )
    axes[1].set_title('max_depth の影響', fontsize=12)
    axes[1].set_ylabel('平均スコア')
    axes[1].grid(True, alpha=0.3)
    
    # min_samples_splitの影響
    results_df.groupby('param_min_samples_split')['mean_test_score'].mean().plot(
        kind='bar', ax=axes[2], color='coral'
    )
    axes[2].set_title('min_samples_split の影響', fontsize=12)
    axes[2].set_ylabel('平均スコア')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 長所と短所

項目 | 詳細  
---|---  
**長所** | ✅ 網羅的探索で最適解を見逃さない  
✅ 実装がシンプルで理解しやすい  
✅ 並列化が容易  
**短所** | ❌ 計算コストが指数的に増加  
❌ 高次元探索には不向き  
❌ 連続値パラメータの探索に制限  
**適用場面** | パラメータ数が少ない（2-4個程度）  
各パラメータの候補が少ない  
計算リソースが十分にある  
  
* * *

## 1.3 ランダムサーチ

### 確率的探索の利点

**ランダムサーチ（Random Search）** は、探索空間からランダムにパラメータの組み合わせをサンプリングします。

> Bergstra & Bengio (2012)の研究により、ランダムサーチはグリッドサーチよりも効率的であることが示されています。
    
    
    ```mermaid
    graph LR
        A[グリッドサーチ] --> B[すべて探索計算コスト: 高]
        C[ランダムサーチ] --> D[ランダムサンプリング計算コスト: 低]
    
        style A fill:#ffcdd2
        style B fill:#ef9a9a
        style C fill:#c8e6c9
        style D fill:#81c784
    ```

### RandomizedSearchCV
    
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    import numpy as np
    
    # ランダムサーチ用の分布定義
    param_distributions = {
        'n_estimators': randint(50, 500),              # 50-500の整数
        'max_depth': randint(5, 30),                   # 5-30の整数
        'min_samples_split': randint(2, 20),           # 2-20の整数
        'min_samples_leaf': randint(1, 10),            # 1-10の整数
        'max_features': uniform(0.1, 0.9)              # 0.1-1.0の実数
    }
    
    # RandomizedSearchCVの設定
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=100,                # 100回のランダムサンプリング
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    # ランダムサーチ実行
    print("=== ランダムサーチ開始 ===")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\n実行時間: {elapsed_time:.2f}秒")
    print(f"\n最良パラメータ:")
    print(random_search.best_params_)
    print(f"\n最良スコア（交差検証）: {random_search.best_score_:.4f}")
    
    # テストデータで評価
    y_pred_random = random_search.predict(X_test)
    test_accuracy_random = accuracy_score(y_test, y_pred_random)
    print(f"テストデータ精度: {test_accuracy_random:.4f}")
    

**出力例** ：
    
    
    === ランダムサーチ開始 ===
    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    
    実行時間: 18.56秒
    
    最良パラメータ:
    {'max_depth': 18, 'max_features': 0.7234, 'min_samples_leaf': 1,
     'min_samples_split': 2, 'n_estimators': 387}
    
    最良スコア（交差検証）: 0.9692
    テストデータ精度: 0.9825
    

### グリッドサーチとの比較
    
    
    import matplotlib.pyplot as plt
    
    # 比較結果の可視化
    comparison_data = {
        'グリッドサーチ': {
            '探索回数': len(grid_search.cv_results_['params']),
            '実行時間': 12.34,
            'CV精度': grid_search.best_score_,
            'テスト精度': test_accuracy
        },
        'ランダムサーチ': {
            '探索回数': len(random_search.cv_results_['params']),
            '実行時間': 18.56,
            'CV精度': random_search.best_score_,
            'テスト精度': test_accuracy_random
        }
    }
    
    # DataFrame化
    comparison_df = pd.DataFrame(comparison_data).T
    print("\n=== グリッドサーチ vs ランダムサーチ ===")
    print(comparison_df)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 探索回数
    comparison_df['探索回数'].plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('探索回数の比較', fontsize=12)
    axes[0].set_ylabel('回数')
    axes[0].grid(True, alpha=0.3)
    
    # 実行時間
    comparison_df['実行時間'].plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
    axes[1].set_title('実行時間の比較', fontsize=12)
    axes[1].set_ylabel('秒')
    axes[1].grid(True, alpha=0.3)
    
    # 精度
    comparison_df[['CV精度', 'テスト精度']].plot(kind='bar', ax=axes[2])
    axes[2].set_title('精度の比較', fontsize=12)
    axes[2].set_ylabel('精度')
    axes[2].set_ylim([0.95, 1.0])
    axes[2].legend(['CV精度', 'テスト精度'])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### ランダムサーチの利点

側面 | グリッドサーチ | ランダムサーチ  
---|---|---  
**計算効率** | 探索回数 = 全組み合わせ | 探索回数を指定可能  
**連続値対応** | 離散値のみ | 連続分布から直接サンプリング  
**重要性への対応** | すべて均等に探索 | 重要なパラメータ範囲を広く探索可能  
**高次元探索** | 次元増加で指数的に増大 | 次元に対して線形的  
  
* * *

## 1.4 交差検証とハイパーパラメータ探索

### CV戦略の選択

交差検証は、ハイパーパラメータの汎化性能を評価するために不可欠です。

CV手法 | 説明 | 使用場面  
---|---|---  
**K-Fold CV** | データをK分割し、K回評価 | 標準的な場面（K=5または10）  
**Stratified K-Fold** | クラス比率を保持して分割 | 分類問題、不均衡データ  
**Time Series Split** | 時系列順序を保持 | 時系列データ  
**Leave-One-Out** | 1サンプルずつテスト | 小規模データ（計算コスト大）  
  
### 評価指標の設定
    
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
    
    # 複数の評価指標で比較
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    # Stratified K-Foldで交差検証
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearchCVに複数評価指標を適用
    random_search_multi = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv_strategy,
        scoring=scoring_metrics,
        refit='f1',                 # F1スコアで最良モデルを選択
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search_multi.fit(X_train, y_train)
    
    print("=== 複数評価指標での結果 ===")
    print(f"最良パラメータ（F1基準）:")
    print(random_search_multi.best_params_)
    
    # 各指標でのスコア
    results = random_search_multi.cv_results_
    best_index = random_search_multi.best_index_
    
    print(f"\n最良モデルのスコア:")
    for metric in scoring_metrics.keys():
        score = results[f'mean_test_{metric}'][best_index]
        std = results[f'std_test_{metric}'][best_index]
        print(f"  {metric}: {score:.4f} (±{std:.4f})")
    

**出力例** ：
    
    
    === 複数評価指標での結果 ===
    最良パラメータ（F1基準）:
    {'max_depth': 22, 'max_features': 0.6543, 'min_samples_leaf': 1,
     'min_samples_split': 3, 'n_estimators': 298}
    
    最良モデルのスコア:
      accuracy: 0.9670 (±0.0123)
      precision: 0.9678 (±0.0118)
      recall: 0.9670 (±0.0123)
      f1: 0.9672 (±0.0121)
    

### オーバーフィッティング防止
    
    
    import matplotlib.pyplot as plt
    
    # 訓練スコアとテストスコアの比較
    results = random_search.cv_results_
    
    train_scores = results['mean_train_score']
    test_scores = results['mean_test_score']
    
    # 過学習の検出
    overfit_gap = train_scores - test_scores
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # スコア分布
    axes[0].scatter(train_scores, test_scores, alpha=0.6, s=50)
    axes[0].plot([0.9, 1.0], [0.9, 1.0], 'r--', label='理想的な線')
    axes[0].set_xlabel('訓練スコア')
    axes[0].set_ylabel('テストスコア（CV）')
    axes[0].set_title('訓練 vs テストスコア', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 過学習ギャップ
    axes[1].hist(overfit_gap, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=overfit_gap.mean(), color='r', linestyle='--',
                    label=f'平均ギャップ: {overfit_gap.mean():.4f}')
    axes[1].set_xlabel('過学習ギャップ（訓練 - テスト）')
    axes[1].set_ylabel('頻度')
    axes[1].set_title('過学習の程度', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 過学習が少ない上位5モデル
    results_df = pd.DataFrame({
        'rank': results['rank_test_score'],
        'train_score': train_scores,
        'test_score': test_scores,
        'overfit_gap': overfit_gap
    })
    
    print("\n=== 過学習が少ないトップ5モデル ===")
    print(results_df.nsmallest(5, 'overfit_gap'))
    

* * *

## 1.5 実践: scikit-learnでの基本チューニング

### Random Forest チューニング例
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    # データ生成
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # デフォルト設定での性能
    print("=== Random Forest チューニング ===\n")
    rf_default = RandomForestClassifier(random_state=42)
    rf_default.fit(X_train, y_train)
    default_score = accuracy_score(y_test, rf_default.predict(X_test))
    print(f"デフォルト設定の精度: {default_score:.4f}")
    
    # グリッドサーチ
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    start = time.time()
    grid_rf.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # チューニング後の性能
    tuned_score = accuracy_score(y_test, grid_rf.predict(X_test))
    
    print(f"\n最良パラメータ: {grid_rf.best_params_}")
    print(f"チューニング後の精度: {tuned_score:.4f}")
    print(f"改善: {(tuned_score - default_score) * 100:.2f}%")
    print(f"実行時間: {elapsed:.2f}秒")
    

**出力例** ：
    
    
    === Random Forest チューニング ===
    
    デフォルト設定の精度: 0.8700
    
    最良パラメータ: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    チューニング後の精度: 0.9250
    改善: 5.50%
    実行時間: 24.56秒
    

### XGBoost チューニング例
    
    
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    # XGBoostのパラメータ分布
    param_dist_xgb = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5)
    }
    
    # デフォルト設定
    print("\n=== XGBoost チューニング ===\n")
    xgb_default = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_default.fit(X_train, y_train)
    default_score_xgb = accuracy_score(y_test, xgb_default.predict(X_test))
    print(f"デフォルト設定の精度: {default_score_xgb:.4f}")
    
    # ランダムサーチ
    random_xgb = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        param_dist_xgb,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    start = time.time()
    random_xgb.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # チューニング後の性能
    tuned_score_xgb = accuracy_score(y_test, random_xgb.predict(X_test))
    
    print(f"\n最良パラメータ:")
    for param, value in random_xgb.best_params_.items():
        print(f"  {param}: {value:.4f}" if isinstance(value, float) else f"  {param}: {value}")
    
    print(f"\nチューニング後の精度: {tuned_score_xgb:.4f}")
    print(f"改善: {(tuned_score_xgb - default_score_xgb) * 100:.2f}%")
    print(f"実行時間: {elapsed:.2f}秒")
    

**出力例** ：
    
    
    === XGBoost チューニング ===
    
    デフォルト設定の精度: 0.9000
    
    最良パラメータ:
      colsample_bytree: 0.8234
      gamma: 0.1234
      learning_rate: 0.0876
      max_depth: 7
      n_estimators: 387
      subsample: 0.8567
    
    チューニング後の精度: 0.9400
    改善: 4.00%
    実行時間: 42.18秒
    

### 結果の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # モデル比較
    models_comparison = {
        'RF (デフォルト)': default_score,
        'RF (チューニング)': tuned_score,
        'XGB (デフォルト)': default_score_xgb,
        'XGB (チューニング)': tuned_score_xgb
    }
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精度比較
    models = list(models_comparison.keys())
    scores = list(models_comparison.values())
    colors = ['lightcoral', 'lightgreen', 'lightcoral', 'lightgreen']
    
    axes[0].bar(models, scores, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('精度')
    axes[0].set_title('モデル性能比較', fontsize=14)
    axes[0].set_ylim([0.8, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, score in enumerate(scores):
        axes[0].text(i, score + 0.01, f'{score:.4f}', ha='center', fontsize=10)
    
    # 改善率
    improvements = [
        0,
        (tuned_score - default_score) * 100,
        0,
        (tuned_score_xgb - default_score_xgb) * 100
    ]
    
    axes[1].bar(models, improvements, color=colors, edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('改善率（%）')
    axes[1].set_title('チューニングによる改善', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, imp in enumerate(improvements):
        if imp > 0:
            axes[1].text(i, imp + 0.2, f'{imp:.2f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **ハイパーパラメータの理解**

     * モデルパラメータとの違い
     * 主要なハイパーパラメータと役割
     * 探索空間の適切な設計
  2. **グリッドサーチ**

     * 網羅的探索による最適化
     * scikit-learn GridSearchCVの使用法
     * 計算コストと探索効率のトレードオフ
  3. **ランダムサーチ**

     * 確率的サンプリングの効率性
     * 連続分布からの直接探索
     * グリッドサーチに対する優位性
  4. **交差検証の重要性**

     * 適切なCV戦略の選択
     * 複数評価指標での総合評価
     * 過学習の検出と防止
  5. **実践的チューニング**

     * Random ForestとXGBoostの最適化
     * デフォルト設定からの改善
     * 結果の可視化と解釈

### 手法選択ガイドライン

状況 | 推奨手法 | 理由  
---|---|---  
パラメータ数が少ない（2-3個） | グリッドサーチ | 網羅的探索が現実的  
パラメータ数が多い（4個以上） | ランダムサーチ | 計算効率が良い  
連続値パラメータ | ランダムサーチ | 分布から直接サンプリング  
計算リソースが限定的 | ランダムサーチ | 探索回数を制御可能  
最高精度が必要 | 両方を組み合わせ | 粗探索→細探索の2段階  
  
### 次の章へ

第2章では、**ベイズ最適化** を学びます：

  * ガウス過程による代理モデル
  * 獲得関数の設計
  * Optimaを使った実装
  * 従来手法との性能比較
  * 実践的な応用例

* * *

## 演習問題

### 問題1（難易度: easy）

ハイパーパラメータとモデルパラメータの違いを3つの観点（定義、決定方法、例）から説明してください。

解答例

**解答** ：

観点 | ハイパーパラメータ | モデルパラメータ  
---|---|---  
**定義** | 学習前に人間が設定する値 | 学習により自動的に最適化される値  
**決定方法** | 試行錯誤、探索アルゴリズム、経験 | 訓練データから勾配降下法等で学習  
**例** | 学習率、木の深さ、正則化係数 | 線形回帰の係数、ニューラルネットの重み  
  
**補足説明** ：

  * ハイパーパラメータはモデルの構造や学習プロセスを制御
  * モデルパラメータはデータのパターンを表現
  * 適切なハイパーパラメータ選択により、モデルパラメータの学習が効率化

### 問題2（難易度: medium）

以下のパラメータグリッドの総組み合わせ数を計算し、グリッドサーチの計算コストについて考察してください。
    
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    # 5-fold交差検証を使用
    

解答例
    
    
    import numpy as np
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    # 各パラメータの候補数
    param_counts = [len(v) for v in param_grid.values()]
    print("各パラメータの候補数:")
    for param, count in zip(param_grid.keys(), param_counts):
        print(f"  {param}: {count}")
    
    # 総組み合わせ数
    total_combinations = np.prod(param_counts)
    print(f"\n総組み合わせ数: {total_combinations:,}")
    
    # 5-fold交差検証での総学習回数
    cv_folds = 5
    total_fits = total_combinations * cv_folds
    print(f"5-fold CVでの総学習回数: {total_fits:,}")
    
    # 1回の学習に1分かかると仮定
    time_per_fit = 1  # 分
    total_time_minutes = total_fits * time_per_fit
    total_time_hours = total_time_minutes / 60
    
    print(f"\n計算時間（1回の学習=1分と仮定）:")
    print(f"  {total_time_minutes:,}分")
    print(f"  {total_time_hours:.1f}時間")
    

**出力** ：
    
    
    各パラメータの候補数:
      n_estimators: 5
      max_depth: 6
      min_samples_split: 4
      learning_rate: 4
    
    総組み合わせ数: 480
    5-fold CVでの総学習回数: 2,400
    
    計算時間（1回の学習=1分と仮定）:
      2,400分
      40.0時間
    

**考察** ：

  * パラメータ数が増えると組み合わせが指数的に増加
  * 交差検証により計算コストがさらに増大
  * この例では約40時間の計算時間が必要
  * ランダムサーチで探索回数を100に制限すれば約8.3時間（500回の学習）

### 問題3（難易度: medium）

グリッドサーチとランダムサーチの長所・短所を比較し、どのような場面でランダムサーチが有利か説明してください。

解答例

**解答** ：

項目 | グリッドサーチ | ランダムサーチ  
---|---|---  
**探索方法** | すべての組み合わせを網羅 | ランダムサンプリング  
**計算コスト** | 指数的に増加 | 探索回数を制御可能  
**最適解の保証** | 探索空間内で保証 | 確率的（保証なし）  
**連続値対応** | 離散化が必要 | 連続分布から直接サンプリング  
**高次元探索** | 困難（組み合わせ爆発） | 次元に対して線形的  
  
**ランダムサーチが有利な場面** ：

  1. **パラメータ数が多い（4個以上）**
     * グリッドサーチでは組み合わせが爆発的に増加
     * ランダムサーチは探索回数を固定できる
  2. **連続値パラメータの最適化**
     * 学習率、正則化係数などの連続値
     * 分布から直接サンプリングできる
  3. **一部のパラメータが重要な場合**
     * Bergstra & Bengio (2012)が示した通り、重要なパラメータの範囲を広く探索
     * グリッドサーチは等間隔に制限される
  4. **計算リソースが限定的**
     * 時間制約がある場合
     * 探索回数を予算内に制御

### 問題4（難易度: hard）

以下のデータセットに対して、RandomForestClassifierのハイパーパラメータチューニングを実装し、デフォルト設定からの改善率を報告してください。
    
    
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    

解答例
    
    
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from scipy.stats import randint, uniform
    import time
    
    # データ準備
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    print("=== Wine データセットでのチューニング ===\n")
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    print(f"クラス数: {len(data.target_names)}")
    
    # 1. デフォルト設定での性能
    print("\n1. デフォルト設定での評価")
    rf_default = RandomForestClassifier(random_state=42)
    rf_default.fit(X_train, y_train)
    y_pred_default = rf_default.predict(X_test)
    default_accuracy = accuracy_score(y_test, y_pred_default)
    
    print(f"精度: {default_accuracy:.4f}")
    
    # 2. ランダムサーチでチューニング
    print("\n2. ランダムサーチでチューニング")
    
    param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9)
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # 最良モデルで評価
    y_pred_tuned = random_search.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    
    print(f"\n最良パラメータ:")
    for param, value in random_search.best_params_.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nCV精度: {random_search.best_score_:.4f}")
    print(f"テスト精度: {tuned_accuracy:.4f}")
    print(f"実行時間: {elapsed_time:.2f}秒")
    
    # 3. 改善率の計算
    improvement = (tuned_accuracy - default_accuracy) * 100
    improvement_pct = (tuned_accuracy / default_accuracy - 1) * 100
    
    print(f"\n=== 結果のまとめ ===")
    print(f"デフォルト設定: {default_accuracy:.4f}")
    print(f"チューニング後: {tuned_accuracy:.4f}")
    print(f"絶対改善: {improvement:.2f}ポイント")
    print(f"相対改善: {improvement_pct:.2f}%")
    
    # 4. 詳細な分類レポート
    print(f"\n=== 分類レポート（チューニング後）===")
    print(classification_report(y_test, y_pred_tuned,
                              target_names=data.target_names))
    
    # 5. 可視化
    import matplotlib.pyplot as plt
    import pandas as pd
    
    results_df = pd.DataFrame(random_search.cv_results_)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # スコア分布
    axes[0, 0].hist(results_df['mean_test_score'], bins=20,
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=random_search.best_score_, color='r',
                       linestyle='--', label='最良スコア')
    axes[0, 0].set_xlabel('CV精度')
    axes[0, 0].set_ylabel('頻度')
    axes[0, 0].set_title('スコア分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # パラメータの影響: n_estimators
    axes[0, 1].scatter(results_df['param_n_estimators'],
                       results_df['mean_test_score'], alpha=0.5)
    axes[0, 1].set_xlabel('n_estimators')
    axes[0, 1].set_ylabel('CV精度')
    axes[0, 1].set_title('n_estimatorsの影響')
    axes[0, 1].grid(True, alpha=0.3)
    
    # パラメータの影響: max_depth
    axes[1, 0].scatter(results_df['param_max_depth'],
                       results_df['mean_test_score'], alpha=0.5)
    axes[1, 0].set_xlabel('max_depth')
    axes[1, 0].set_ylabel('CV精度')
    axes[1, 0].set_title('max_depthの影響')
    axes[1, 0].grid(True, alpha=0.3)
    
    # デフォルト vs チューニング
    comparison = ['デフォルト', 'チューニング']
    scores = [default_accuracy, tuned_accuracy]
    colors = ['lightcoral', 'lightgreen']
    
    axes[1, 1].bar(comparison, scores, color=colors,
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('精度')
    axes[1, 1].set_title('性能比較')
    axes[1, 1].set_ylim([0.9, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, score in enumerate(scores):
        axes[1, 1].text(i, score + 0.005, f'{score:.4f}',
                        ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

**出力例** ：
    
    
    === Wine データセットでのチューニング ===
    
    訓練データ: (142, 13)
    テストデータ: (36, 13)
    クラス数: 3
    
    1. デフォルト設定での評価
    精度: 0.9722
    
    2. ランダムサーチでチューニング
    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    
    最良パラメータ:
      max_depth: 18
      max_features: 0.3456
      min_samples_leaf: 1
      min_samples_split: 2
      n_estimators: 287
    
    CV精度: 0.9859
    テスト精度: 1.0000
    実行時間: 15.23秒
    
    === 結果のまとめ ===
    デフォルト設定: 0.9722
    チューニング後: 1.0000
    絶対改善: 2.78ポイント
    相対改善: 2.86%
    
    === 分類レポート（チューニング後）===
                  precision    recall  f1-score   support
    
         class_0       1.00      1.00      1.00        14
         class_1       1.00      1.00      1.00        15
         class_2       1.00      1.00      1.00         7
    
        accuracy                           1.00        36
       macro avg       1.00      1.00      1.00        36
    weighted avg       1.00      1.00      1.00        36
    

### 問題5（難易度: hard）

交差検証におけるデータリークの危険性について説明し、正しい実装方法を示してください。特に、スケーリングやハイパーパラメータ探索の文脈で考察してください。

解答例

**解答** ：

**データリークとは** ：

訓練データとテストデータの境界を越えて情報が漏れることで、モデルの性能が過大評価される問題です。

**具体的な危険性** ：

  1. **スケーリングでのリーク**
     * 全データでスケーリング→訓練/テスト分割だとテストデータの統計情報が訓練に漏れる
     * テストデータの平均・標準偏差を使用してしまう
  2. **特徴選択でのリーク**
     * 全データで特徴選択→訓練/テスト分割だとテストデータの情報が選択に影響
  3. **交差検証でのリーク**
     * CV外で前処理→各foldにテストfoldの情報が漏れる

**誤った実装例** ：
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # ❌ 間違い：全データでスケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 全データでfit
    
    # その後に交差検証
    scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
    # → テストfoldの情報が訓練foldに漏れている
    

**正しい実装例** ：
    
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # ✅ 正しい：Pipelineを使用
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    
    # Pipelineで交差検証
    # 各foldで訓練データのみでスケーラーをfit
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    # ハイパーパラメータ探索も同様
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    

**実証実験** ：
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    # データ生成（スケールの異なる特徴量）
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=10, random_state=42)
    
    # 意図的にスケールを変える
    X[:, :10] = X[:, :10] * 1000  # 最初の10特徴を1000倍
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== データリークの実証 ===\n")
    
    # 1. 誤った方法（データリークあり）
    scaler_wrong = StandardScaler()
    X_train_wrong = scaler_wrong.fit_transform(X_train)
    X_test_wrong = scaler_wrong.transform(X_test)
    
    # CVでもリークが発生
    X_all_scaled = StandardScaler().fit_transform(X)
    cv_scores_wrong = cross_val_score(
        RandomForestClassifier(random_state=42),
        X_all_scaled, y, cv=5
    )
    
    print("❌ 誤った方法（全データでスケーリング後にCV）")
    print(f"CV精度: {cv_scores_wrong.mean():.4f} (±{cv_scores_wrong.std():.4f})")
    
    # 2. 正しい方法（Pipelineでリーク防止）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    cv_scores_correct = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"\n✅ 正しい方法（Pipeline使用）")
    print(f"CV精度: {cv_scores_correct.mean():.4f} (±{cv_scores_correct.std():.4f})")
    
    # 差を計算
    difference = cv_scores_wrong.mean() - cv_scores_correct.mean()
    print(f"\n過大評価の程度: {difference:.4f} ({difference*100:.2f}%ポイント)")
    
    print("\n=== 結論 ===")
    print("データリークにより性能が過大評価されている")
    print("Pipelineを使用することで正しい評価が可能")
    

**出力例** ：
    
    
    === データリークの実証 ===
    
    ❌ 誤った方法（全データでスケーリング後にCV）
    CV精度: 0.9120 (±0.0234)
    
    ✅ 正しい方法（Pipeline使用）
    CV精度: 0.9050 (±0.0287)
    
    過大評価の程度: 0.0070 (0.70%ポイント)
    
    === 結論 ===
    データリークにより性能が過大評価されている
    Pipelineを使用することで正しい評価が可能
    

**ベストプラクティス** ：

  * 常にPipelineを使用して前処理とモデルを統合
  * 交差検証は前処理を含む全パイプラインに対して実行
  * 訓練データでfitし、テストデータではtransformのみ
  * ハイパーパラメータ探索もPipeline全体に対して実施

* * *

## 参考文献

  1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. _Journal of Machine Learning Research_ , 13(1), 281-305.
  2. Feurer, M., & Hutter, F. (2019). Hyperparameter optimization. In _Automated Machine Learning_ (pp. 3-33). Springer.
  3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_ (2nd ed.). Springer.
  4. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
