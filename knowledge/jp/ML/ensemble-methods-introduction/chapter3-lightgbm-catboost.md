---
title: 第3章：LightGBMとCatBoost
chapter_title: 第3章：LightGBMとCatBoost
subtitle: 次世代勾配ブースティング - 高速化とカテゴリカル変数処理
reading_time: 25-30分
difficulty: 中級
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ LightGBMの高速化技術（GOSS、EFB、Histogram-based）を理解する
  * ✅ LightGBMを実装し、パラメータチューニングができる
  * ✅ CatBoostのOrdered BoostingとCategorical Features処理を理解する
  * ✅ CatBoostを実装し、エンコーディング戦略を選択できる
  * ✅ XGBoost、LightGBM、CatBoostの特性を比較し、使い分けられる

* * *

## 3.1 LightGBM - 高速化の仕組み

### LightGBMとは

**LightGBM（Light Gradient Boosting Machine）** は、Microsoftが開発した高速で効率的な勾配ブースティングフレームワークです。

> 「Light」の名の通り、XGBoostよりも軽量で高速、大規模データセットに適しています。

### 主要な技術革新

#### 1\. Histogram-based Algorithm（ヒストグラムベースアルゴリズム）

連続値を離散化（ビニング）することで、計算量を大幅削減します。

手法 | 計算量 | メモリ | 精度  
---|---|---|---  
**Pre-sorted** （XGBoost） | $O(n \log n)$ | 高 | 高  
**Histogram-based** （LightGBM） | $O(n \times k)$ | 低 | ほぼ同等  
  
$k$: ビン数（通常255）、$n$: データ数
    
    
    ```mermaid
    graph LR
        A[連続値データ] --> B[ヒストグラム化]
        B --> C[255ビンに離散化]
        C --> D[高速な分岐探索]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#f3e5f5
        style D fill:#c8e6c9
    ```

#### 2\. GOSS（Gradient-based One-Side Sampling）

**GOSS** は、勾配の大きいデータを重視し、小さいデータをサンプリングすることで学習を高速化します。

アルゴリズム：

  1. 勾配の絶対値でデータをソート
  2. 上位 $a\%$ （大きい勾配）を全て保持
  3. 残り $(1-a)\%$ から $b\%$ をランダムサンプリング
  4. サンプルされたデータの重みを $(1-a)/b$ 倍に調整

#### 3\. EFB（Exclusive Feature Bundling）

**EFB** は、互いに排他的な特徴量（同時に非ゼロにならない）を束ねて次元を削減します。

例：One-Hot Encoding された特徴量
    
    
    color_red:   [1, 0, 0, 1, 0]
    color_blue:  [0, 1, 0, 0, 1]
    color_green: [0, 0, 1, 0, 0]
    → 1つの特徴量に統合可能
    

### Leaf-wise vs Level-wise 成長戦略

戦略 | 説明 | 使用 | 長所 | 短所  
---|---|---|---|---  
**Level-wise** | 深さ優先で全ノードを分割 | XGBoost | バランスの取れた木 | 情報利得が低いノードも分割  
**Leaf-wise** | 最大情報利得のリーフを分割 | LightGBM | 効率的、高精度 | 過学習しやすい  
      
    
    ```mermaid
    graph TD
        A[Level-wise: XGBoost] --> B1[レベル1: 全て分割]
        B1 --> C1[レベル2: 全て分割]
    
        D[Leaf-wise: LightGBM] --> E1[最大利得ノードのみ分割]
        E1 --> F1[次の最大利得ノードを分割]
    
        style A fill:#e3f2fd
        style D fill:#f3e5f5
    ```

* * *

## 3.2 LightGBM実装

### 基本的な使い方
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import lightgbm as lgb
    
    # データ生成
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # LightGBMモデルの構築
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )
    
    # 学習
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("=== LightGBM 基本実装 ===")
    print(f"精度: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    

**出力** ：
    
    
    === LightGBM 基本実装 ===
    精度: 0.9350
    AUC: 0.9712
    

### 重要なパラメータ

パラメータ | 説明 | 推奨値  
---|---|---  
`num_leaves` | 木の最大リーフ数 | 31-255（デフォルト: 31）  
`max_depth` | 木の最大深さ（過学習制御） | 3-10（デフォルト: -1=無制限）  
`learning_rate` | 学習率 | 0.01-0.1  
`n_estimators` | 木の本数 | 100-1000  
`min_child_samples` | リーフの最小サンプル数 | 20-100  
`subsample` | データサンプリング比率 | 0.7-1.0  
`colsample_bytree` | 特徴量サンプリング比率 | 0.7-1.0  
`reg_alpha` | L1正則化 | 0-1  
`reg_lambda` | L2正則化 | 0-1  
  
### 早期停止とバリデーション
    
    
    # 訓練データをさらに分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 早期停止付きで学習
    model_early = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )
    
    model_early.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    print(f"\n=== 早期停止 ===")
    print(f"最適なイテレーション数: {model_early.best_iteration_}")
    print(f"バリデーションAUC: {model_early.best_score_['valid_0']['auc']:.4f}")
    
    # テストデータで評価
    y_pred_early = model_early.predict(X_test)
    accuracy_early = accuracy_score(y_test, y_pred_early)
    print(f"テスト精度: {accuracy_early:.4f}")
    

### 特徴量重要度の可視化
    
    
    import matplotlib.pyplot as plt
    
    # 特徴量重要度の取得
    feature_importance = model.feature_importances_
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # DataFrameに変換
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== 特徴量重要度 Top 10 ===")
    print(importance_df.head(10))
    
    # 可視化
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain')
    plt.title('LightGBM 特徴量重要度（Gain）', fontsize=14)
    plt.tight_layout()
    plt.show()
    

### GPU サポート
    
    
    # GPU使用（CUDA環境が必要）
    model_gpu = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        device='gpu',  # GPU使用
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=42
    )
    
    # 学習（GPUで高速化）
    # model_gpu.fit(X_train, y_train)
    
    print("\n=== GPU サポート ===")
    print("LightGBMはGPU（CUDA）をサポートしており、大規模データで10-30倍高速化可能")
    print("device='gpu' パラメータで有効化")
    

> **注意** : GPU版を使うには、LightGBMをGPUサポート付きでビルドする必要があります。

* * *

## 3.3 CatBoost - Ordered Boostingとカテゴリカル変数処理

### CatBoostとは

**CatBoost（Categorical Boosting）** は、Yandexが開発した勾配ブースティングフレームワークで、カテゴリカル変数の自動処理が特徴です。

### 主要な技術革新

#### 1\. Ordered Boosting

**Ordered Boosting** は、予測シフト（prediction shift）を防ぐための手法です。

**問題** : 従来のブースティングでは、同じデータで勾配計算と学習を行うため、過学習しやすい。

**解決策** :

  1. データをランダムに並べ替え
  2. 各サンプル $i$ の予測に、サンプル $1, ..., i-1$ のみを使用
  3. 異なる順序で複数のモデルを構築

    
    
    ```mermaid
    graph LR
        A[従来のブースティング] --> B[全データで学習]
        B --> C[同じデータで予測]
        C --> D[予測シフト発生]
    
        E[Ordered Boosting] --> F[過去データのみで学習]
        F --> G[未来データで予測]
        G --> H[予測シフト防止]
    
        style D fill:#ffebee
        style H fill:#c8e6c9
    ```

#### 2\. Categorical Features の自動処理

CatBoostは、カテゴリカル変数を自動でエンコーディングします。

**Target Statistics** （ターゲット統計量）の計算：

$$ \text{TS}(x_i) = \frac{\sum_{j=1}^{i-1} \mathbb{1}_{x_j = x_i} \cdot y_j + a \cdot P}{\sum_{j=1}^{i-1} \mathbb{1}_{x_j = x_i} + a} $$ 

  * $x_i$: カテゴリ値
  * $y_j$: ターゲット値
  * $a$: 平滑化パラメータ
  * $P$: 事前確率

この手法により、以下の利点があります：

  * One-Hot Encodingが不要
  * 高カーディナリティ（多数のカテゴリ）に対応
  * ターゲットリークを防止

### 対称木（Oblivious Trees）

CatBoostは**対称木** （Oblivious Decision Trees）を使用します。

特性 | 通常の決定木 | 対称木（CatBoost）  
---|---|---  
分割条件 | 各ノードで異なる | 同じレベルで同じ条件  
構造 | 非対称 | 完全対称  
過学習 | しやすい | しにくい  
予測速度 | 普通 | 非常に高速  
  
* * *

## 3.4 CatBoost実装

### 基本的な使い方
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from catboost import CatBoostClassifier
    
    # データ生成
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CatBoostモデルの構築
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        random_seed=42,
        verbose=0
    )
    
    # 学習
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("=== CatBoost 基本実装 ===")
    print(f"精度: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    

**出力** ：
    
    
    === CatBoost 基本実装 ===
    精度: 0.9365
    AUC: 0.9721
    

### カテゴリカル変数の扱い
    
    
    # カテゴリカル変数を含むデータセット生成
    np.random.seed(42)
    n = 5000
    
    df = pd.DataFrame({
        'num_feature1': np.random.randn(n),
        'num_feature2': np.random.uniform(0, 100, n),
        'cat_feature1': np.random.choice(['A', 'B', 'C', 'D'], n),
        'cat_feature2': np.random.choice(['Low', 'Medium', 'High'], n),
        'cat_feature3': np.random.choice([f'Cat_{i}' for i in range(50)], n)  # 高カーディナリティ
    })
    
    # ターゲット変数（カテゴリに依存）
    df['target'] = (
        (df['cat_feature1'].isin(['A', 'B'])) &
        (df['num_feature1'] > 0) &
        (df['num_feature2'] > 50)
    ).astype(int)
    
    # 特徴量とターゲットの分離
    X = df.drop('target', axis=1)
    y = df['target']
    
    # カテゴリカル変数の列を指定
    cat_features = ['cat_feature1', 'cat_feature2', 'cat_feature3']
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== カテゴリカル変数を含むデータ ===")
    print(f"データ形状: {X.shape}")
    print(f"カテゴリカル変数: {cat_features}")
    print(f"\n各カテゴリのユニーク数:")
    for col in cat_features:
        print(f"  {col}: {X[col].nunique()}")
    
    # CatBoostで学習（カテゴリカル変数を自動処理）
    model_cat = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,  # カテゴリカル変数を指定
        random_seed=42,
        verbose=0
    )
    
    model_cat.fit(X_train, y_train)
    
    # 評価
    y_pred_cat = model_cat.predict(X_test)
    y_proba_cat = model_cat.predict_proba(X_test)[:, 1]
    
    accuracy_cat = accuracy_score(y_test, y_pred_cat)
    auc_cat = roc_auc_score(y_test, y_proba_cat)
    
    print(f"\n=== カテゴリカル変数処理結果 ===")
    print(f"精度: {accuracy_cat:.4f}")
    print(f"AUC: {auc_cat:.4f}")
    print("✓ One-Hot Encoding不要で高カーディナリティに対応")
    

### エンコーディング戦略

CatBoostは複数のエンコーディングモードをサポートします：

モード | 説明 | 用途  
---|---|---  
`Ordered` | Ordered Target Statistics | 過学習防止（デフォルト）  
`GreedyLogSum` | 貪欲なログ和 | 大規模データ  
`OneHot` | One-Hot Encoding | 低カーディナリティ（≤10）  
      
    
    # エンコーディング戦略の比較
    from catboost import Pool
    
    # CatBoost Pool作成（効率的なデータ構造）
    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )
    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_features
    )
    
    # 異なるエンコーディング戦略
    strategies = {
        'Ordered': 'Ordered',
        'GreedyLogSum': 'GreedyLogSum',
        'OneHot': {'one_hot_max_size': 10}  # カーディナリティ≤10でOne-Hot
    }
    
    print("\n=== エンコーディング戦略の比較 ===")
    for name, strategy in strategies.items():
        model_strategy = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features,
            random_seed=42,
            verbose=0
        )
    
        if name == 'OneHot':
            model_strategy.set_params(**strategy)
    
        model_strategy.fit(train_pool)
        y_pred = model_strategy.predict(test_pool)
        accuracy = accuracy_score(y_test, y_pred)
    
        print(f"{name:15s}: 精度 = {accuracy:.4f}")
    

### 早期停止とバリデーション
    
    
    # 訓練データをさらに分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 早期停止付きで学習
    model_early = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100
    )
    
    model_early.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    print(f"\n=== 早期停止 ===")
    print(f"最適なイテレーション数: {model_early.get_best_iteration()}")
    print(f"最良スコア: {model_early.get_best_score()}")
    
    # テストデータで評価
    y_pred_early = model_early.predict(X_test)
    accuracy_early = accuracy_score(y_test, y_pred_early)
    print(f"テスト精度: {accuracy_early:.4f}")
    

* * *

## 3.5 XGBoost、LightGBM、CatBoostの比較

### アルゴリズムの特性比較

特性 | XGBoost | LightGBM | CatBoost  
---|---|---|---  
**開発元** | Tianqi Chen（DMLC） | Microsoft | Yandex  
**分割アルゴリズム** | Pre-sorted | Histogram-based | Histogram-based  
**木の成長戦略** | Level-wise | Leaf-wise | Level-wise（対称木）  
**速度** | 普通 | 高速 | やや遅い  
**メモリ効率** | 普通 | 高効率 | 普通  
**カテゴリカル処理** | 手動エンコーディング必要 | 手動エンコーディング必要 | 自動処理  
**過学習耐性** | 高 | 中（Leaf-wiseで注意） | 非常に高  
**GPUサポート** | あり | あり | あり  
**ハイパーパラメータ調整** | やや複雑 | やや複雑 | シンプル  
  
### パフォーマンス比較実験
    
    
    import time
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # 大規模データ生成
    X_large, y_large = make_classification(
        n_samples=50000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    
    X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(
        X_large, y_large, test_size=0.2, random_state=42
    )
    
    # 共通パラメータ
    common_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    # モデル定義
    models = {
        'XGBoost': XGBClassifier(**common_params, verbosity=0),
        'LightGBM': LGBMClassifier(**common_params, verbose=-1),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0
        )
    }
    
    print("=== 性能比較（50,000サンプル、50特徴量）===\n")
    results = []
    
    for name, model in models.items():
        # 学習時間測定
        start_time = time.time()
        model.fit(X_train_lg, y_train_lg)
        train_time = time.time() - start_time
    
        # 予測時間測定
        start_time = time.time()
        y_pred = model.predict(X_test_lg)
        y_proba = model.predict_proba(X_test_lg)[:, 1]
        pred_time = time.time() - start_time
    
        # 評価
        accuracy = accuracy_score(y_test_lg, y_pred)
        auc = roc_auc_score(y_test_lg, y_proba)
    
        results.append({
            'Model': name,
            'Train Time (s)': train_time,
            'Predict Time (s)': pred_time,
            'Accuracy': accuracy,
            'AUC': auc
        })
    
        print(f"{name}:")
        print(f"  学習時間: {train_time:.3f}秒")
        print(f"  予測時間: {pred_time:.3f}秒")
        print(f"  精度: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}\n")
    
    # 結果をDataFrameで表示
    results_df = pd.DataFrame(results)
    print("=== 結果サマリー ===")
    print(results_df.to_string(index=False))
    

### メモリ使用量の比較
    
    
    import sys
    
    print("\n=== メモリ使用量の推定 ===")
    for name, model in models.items():
        # モデルのメモリサイズ（概算）
        model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
        print(f"{name:10s}: 約 {model_size:.2f} MB")
    
    print("\n特徴:")
    print("• LightGBM: Histogram化により最小メモリ")
    print("• XGBoost: Pre-sorted法で中程度メモリ")
    print("• CatBoost: 対称木でコンパクト")
    

### 使い分けガイドライン

状況 | 推奨 | 理由  
---|---|---  
**大規模データ（ >100万行）** | LightGBM | 最高速、低メモリ  
**カテゴリカル変数多数** | CatBoost | 自動処理、高精度  
**高カーディナリティ** | CatBoost | Target Statistics  
**過学習が心配** | CatBoost | Ordered Boosting  
**バランスの取れた性能** | XGBoost | 安定、豊富な実績  
**速度優先** | LightGBM | Leaf-wise + Histogram  
**精度優先** | CatBoost | 過学習耐性  
**チューニング時間が限られる** | CatBoost | デフォルトで良好  
**GPUで高速化** | 全て対応 | 環境に応じて選択  
  
### 実務での選択フローチャート
    
    
    ```mermaid
    graph TD
        A[勾配ブースティングが必要] --> B{カテゴリカル変数が多い?}
        B -->|Yes| C[CatBoost]
        B -->|No| D{データサイズは?}
        D -->|大規模 >100万行| E[LightGBM]
        D -->|中小規模| F{何を重視?}
        F -->|速度| E
        F -->|精度| C
        F -->|バランス| G[XGBoost]
    
        style C fill:#c8e6c9
        style E fill:#fff9c4
        style G fill:#e1bee7
    ```

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **LightGBMの技術革新**

     * Histogram-based Algorithm: 計算量削減
     * GOSS: 勾配ベースサンプリング
     * EFB: 排他的特徴量の束ね
     * Leaf-wise成長: 効率的な木構築
  2. **LightGBM実装**

     * 高速で効率的な学習
     * GPUサポートによる更なる高速化
     * 豊富なパラメータで柔軟な調整
  3. **CatBoostの技術革新**

     * Ordered Boosting: 予測シフト防止
     * カテゴリカル変数の自動処理
     * 対称木: 過学習耐性と高速予測
  4. **CatBoost実装**

     * カテゴリカル変数を直接扱える
     * 高カーディナリティに対応
     * デフォルトパラメータで高性能
  5. **3ツールの比較**

     * XGBoost: バランスと実績
     * LightGBM: 速度とメモリ効率
     * CatBoost: カテゴリカル処理と精度

### 選択のポイント

重視項目 | 第1候補 | 第2候補  
---|---|---  
学習速度 | LightGBM | XGBoost  
予測精度 | CatBoost | XGBoost  
メモリ効率 | LightGBM | CatBoost  
カテゴリカル処理 | CatBoost | -  
チューニングのしやすさ | CatBoost | XGBoost  
安定性 | XGBoost | CatBoost  
  
### 次のステップ

  * ハイパーパラメータの自動チューニング（Optuna、Hyperopt）
  * アンサンブル手法の組み合わせ（スタッキング、ブレンディング）
  * 特徴量エンジニアリングとの統合
  * モデルの解釈性向上（SHAP、LIME）

* * *

## 演習問題

### 問題1（難易度：easy）

LightGBMの3つの主要な高速化技術（Histogram-based、GOSS、EFB）をそれぞれ説明してください。

解答例

**解答** ：

  1. **Histogram-based Algorithm（ヒストグラムベースアルゴリズム）**

     * 説明: 連続値を固定数のビン（通常255）に離散化
     * 効果: 計算量を $O(n \log n)$ から $O(n \times k)$ に削減
     * 利点: メモリ効率向上、分岐探索の高速化
  2. **GOSS（Gradient-based One-Side Sampling）**

     * 説明: 勾配の大きいデータを優先的に使用
     * 手順: 勾配上位 $a\%$ 全保持 + 残りから $b\%$ サンプリング
     * 利点: データ削減による高速化、精度維持
  3. **EFB（Exclusive Feature Bundling）**

     * 説明: 排他的な特徴量（同時に非ゼロにならない）を束ねる
     * 例: One-Hot Encodingされた変数を1つにまとめる
     * 利点: 特徴量数削減による高速化

### 問題2（難易度：medium）

Level-wise（XGBoost）とLeaf-wise（LightGBM）の木成長戦略の違いを説明し、それぞれの長所と短所を述べてください。

解答例

**解答** ：

**Level-wise（レベル方向）** ：

  * 戦略: 深さ優先で、同じレベルの全ノードを分割
  * 長所: バランスの取れた木、過学習しにくい
  * 短所: 情報利得の低いノードも分割するため非効率
  * 使用: XGBoost、CatBoost

**Leaf-wise（リーフ方向）** ：

  * 戦略: 最大情報利得のリーフのみを分割
  * 長所: 効率的、高精度、高速
  * 短所: 過学習しやすい（深さ制限が重要）
  * 使用: LightGBM

**比較表** ：

項目 | Level-wise | Leaf-wise  
---|---|---  
効率 | 普通 | 高  
精度 | 安定 | 高いが過学習注意  
木の形状 | 対称 | 非対称  
過学習耐性 | 高 | 中（深さ制限必要）  
  
### 問題3（難易度：medium）

CatBoostのOrdered Boostingが、なぜ予測シフト（prediction shift）を防げるのか説明してください。

解答例

**解答** ：

**予測シフトの問題** ：

従来のブースティングでは、以下の問題が発生します：

  1. 全データで勾配を計算
  2. 同じデータで次の弱学習器を学習
  3. 訓練データに過剰適合（同じデータを見て予測と学習）
  4. テストデータで性能低下

**Ordered Boostingの解決策** ：

  1. **データの順序付け** : データをランダムに並べ替え
  2. **過去データのみ使用** : サンプル $i$ の予測に、サンプル $1, ..., i-1$ のみを使用
  3. **未来データで検証** : 学習に使ったデータで予測しない
  4. **複数モデル** : 異なる順序で複数モデルを構築し平均化

**効果** ：

  * 訓練とテストで同じ条件（過去データのみで予測）
  * 予測シフトの防止
  * 汎化性能の向上
  * 過学習の抑制

**数式** ：

サンプル $i$ の予測値 $\hat{y}_i$ は：

$$ \hat{y}_i = M(\\{(x_j, y_j)\\}_{j=1}^{i-1}) $$ 

つまり、$i$ より前のデータのみで学習したモデル $M$ を使用。

### 問題4（難易度：hard）

以下のデータに対して、LightGBMとCatBoostで学習し、性能を比較してください。カテゴリカル変数の処理方法の違いに注目してください。
    
    
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n = 10000
    
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'cat2': np.random.choice([f'Cat_{i}' for i in range(100)], n),  # 高カーディナリティ
        'target': np.random.choice([0, 1], n)
    })
    

解答例
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    # データ生成
    np.random.seed(42)
    n = 10000
    
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'cat2': np.random.choice([f'Cat_{i}' for i in range(100)], n),
    })
    
    # ターゲット生成（カテゴリに依存）
    df['target'] = (
        (df['cat1'].isin(['A', 'B'])) &
        (df['num1'] > 0)
    ).astype(int)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== データ情報 ===")
    print(f"サンプル数: {n}")
    print(f"カテゴリカル変数:")
    print(f"  cat1: {X['cat1'].nunique()} ユニーク値")
    print(f"  cat2: {X['cat2'].nunique()} ユニーク値（高カーディナリティ）")
    
    # ===== LightGBM: Label Encodingが必要 =====
    print("\n=== LightGBM（Label Encoding使用）===")
    
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    
    # Label Encoding
    le_cat1 = LabelEncoder()
    le_cat2 = LabelEncoder()
    
    X_train_lgb['cat1'] = le_cat1.fit_transform(X_train_lgb['cat1'])
    X_test_lgb['cat1'] = le_cat1.transform(X_test_lgb['cat1'])
    
    X_train_lgb['cat2'] = le_cat2.fit_transform(X_train_lgb['cat2'])
    # テストデータに未知カテゴリがある可能性に対処
    X_test_lgb['cat2'] = X_test_lgb['cat2'].map(
        {v: k for k, v in enumerate(le_cat2.classes_)}
    ).fillna(-1).astype(int)
    
    model_lgb = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    model_lgb.fit(X_train_lgb, y_train)
    y_pred_lgb = model_lgb.predict(X_test_lgb)
    y_proba_lgb = model_lgb.predict_proba(X_test_lgb)[:, 1]
    
    acc_lgb = accuracy_score(y_test, y_pred_lgb)
    auc_lgb = roc_auc_score(y_test, y_proba_lgb)
    
    print(f"精度: {acc_lgb:.4f}")
    print(f"AUC: {auc_lgb:.4f}")
    print("処理: Label Encodingで数値化（順序情報なし）")
    
    # ===== CatBoost: カテゴリカル変数を直接扱える =====
    print("\n=== CatBoost（自動カテゴリカル処理）===")
    
    cat_features = ['cat1', 'cat2']
    
    model_cat = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        verbose=0
    )
    
    model_cat.fit(X_train, y_train)
    y_pred_cat = model_cat.predict(X_test)
    y_proba_cat = model_cat.predict_proba(X_test)[:, 1]
    
    acc_cat = accuracy_score(y_test, y_pred_cat)
    auc_cat = roc_auc_score(y_test, y_proba_cat)
    
    print(f"精度: {acc_cat:.4f}")
    print(f"AUC: {auc_cat:.4f}")
    print("処理: Target Statisticsで自動エンコーディング")
    
    # ===== 比較 =====
    print("\n=== 比較結果 ===")
    comparison = pd.DataFrame({
        'Model': ['LightGBM', 'CatBoost'],
        'Accuracy': [acc_lgb, acc_cat],
        'AUC': [auc_lgb, auc_cat],
        'Categorical Handling': ['Manual (Label Encoding)', 'Automatic (Target Statistics)']
    })
    print(comparison.to_string(index=False))
    
    print("\n=== 考察 ===")
    print("• LightGBM: Label Encodingで順序情報がない数値化（準最適）")
    print("• CatBoost: Target Statisticsで意味のあるエンコーディング")
    print("• 高カーディナリティでCatBoostが有利")
    print("• One-Hot Encodingは次元爆発で非実用的（100カテゴリ）")
    

### 問題5（難易度：hard）

XGBoost、LightGBM、CatBoostのそれぞれで、以下の状況に最適なものを選び、理由を述べてください：

  1. 1億行、100特徴量のデータセット
  2. 100カテゴリを持つ高カーディナリティ変数が5つ
  3. 小規模データ（10,000行）で精度を最大化したい

解答例

**解答** ：

**1\. 1億行、100特徴量のデータセット**

  * **推奨** : LightGBM
  * **理由** : 
    * Histogram-based Algorithmで最速
    * GOSSによるデータサンプリングで更に高速化
    * メモリ効率が最高（大規模データに不可欠）
    * GPUサポートで更に加速可能
  * **代替** : XGBoost（GPUモード）も選択肢だが、LightGBMより遅い

**2\. 100カテゴリを持つ高カーディナリティ変数が5つ**

  * **推奨** : CatBoost
  * **理由** : 
    * カテゴリカル変数を自動で処理（Target Statistics）
    * One-Hot Encodingが不要（100カテゴリ×5 = 500次元の爆発を回避）
    * Ordered Boostingでターゲットリーク防止
    * 高カーディナリティに最適化された設計
  * **他の選択肢の問題点** : 
    * XGBoost: Label Encodingでは順序情報が無意味、One-Hotは次元爆発
    * LightGBM: 同上

**3\. 小規模データ（10,000行）で精度を最大化したい**

  * **推奨** : CatBoost
  * **理由** : 
    * Ordered Boostingで過学習を防ぎ、汎化性能が高い
    * 小規模データでの精度に定評
    * デフォルトパラメータが優秀（チューニング時間削減）
    * 対称木で安定した学習
    * 速度は問題にならない（データが小さい）
  * **代替** : XGBoost（バランスが良く安定）
  * **LightGBMの問題** : Leaf-wiseで小規模データでは過学習しやすい

**まとめ表** ：

状況 | 第1選択 | 第2選択 | キーファクター  
---|---|---|---  
大規模データ | LightGBM | XGBoost (GPU) | 速度、メモリ  
高カーディナリティ | CatBoost | - | 自動カテゴリ処理  
小規模・高精度 | CatBoost | XGBoost | 過学習耐性  
  
* * *

## 参考文献

  1. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." _Advances in Neural Information Processing Systems_ 30.
  2. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." _Advances in Neural Information Processing Systems_ 31.
  3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." _Proceedings of the 22nd ACM SIGKDD_.
  4. Microsoft LightGBM Documentation: <https://lightgbm.readthedocs.io/>
  5. Yandex CatBoost Documentation: <https://catboost.ai/docs/>
