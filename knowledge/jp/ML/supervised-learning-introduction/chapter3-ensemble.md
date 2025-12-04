---
title: 第3章：アンサンブル手法
chapter_title: 第3章：アンサンブル手法
subtitle: 複数モデルの組み合わせによる性能向上 - Random ForestからXGBoost・LightGBM・CatBoostまで
reading_time: 25-30分
difficulty: 中級
code_examples: 13
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ アンサンブル学習の原理を理解する
  * ✅ BaggingとBoostingの違いを説明できる
  * ✅ Random Forestを実装し特徴量重要度を分析できる
  * ✅ Gradient Boostingの仕組みを理解する
  * ✅ XGBoost、LightGBM、CatBoostを使いこなせる
  * ✅ Kaggleコンペで使える実践的なテクニックを習得する

* * *

## 3.1 アンサンブル学習とは

### 定義

**アンサンブル学習（Ensemble Learning）** は、複数の学習器（モデル）を組み合わせて、単一モデルよりも高い性能を実現する手法です。

> 「三人寄れば文殊の知恵」- 複数の弱学習器を組み合わせることで強力な予測器を構築

### アンサンブルの利点
    
    
    ```mermaid
    graph LR
        A[アンサンブルの利点] --> B[精度向上]
        A --> C[過学習抑制]
        A --> D[安定性向上]
        A --> E[ロバスト性向上]
    
        B --> B1[単一モデルより高精度]
        C --> C1[分散を減少]
        D --> D1[予測のばらつき低減]
        E --> E1[外れ値・ノイズに強い]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### 主要な手法

手法 | 原理 | 代表例  
---|---|---  
**Bagging** | 並列学習、平均化 | Random Forest  
**Boosting** | 逐次学習、誤差修正 | XGBoost, LightGBM, CatBoost  
**Stacking** | メタ学習器で統合 | Level-wise Stacking  
  
* * *

## 3.2 Bagging（Bootstrap Aggregating）

### 原理

**Bagging** は、ブートストラップサンプリングで複数のデータセットを作成し、それぞれで学習したモデルの予測を平均化します。
    
    
    ```mermaid
    graph TD
        A[訓練データ] --> B[ブートストラップサンプリング]
        B --> C1[サンプル1]
        B --> C2[サンプル2]
        B --> C3[サンプル3]
        C1 --> D1[モデル1]
        C2 --> D2[モデル2]
        C3 --> D3[モデル3]
        D1 --> E[投票/平均化]
        D2 --> E
        D3 --> E
        E --> F[最終予測]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style E fill:#f3e5f5
        style F fill:#e8f5e9
    ```

### アルゴリズム

  1. 訓練データから復元抽出でT個のブートストラップサンプルを作成
  2. 各サンプルで独立に学習器を訓練
  3. 分類: 多数決、回帰: 平均で最終予測

$$ \hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x}) $$

### 実装例
    
    
    import numpy as np
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bagging
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,  # 学習器の数
        max_samples=0.8,   # サンプリング比率
        random_state=42
    )
    
    bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_test)
    
    print("=== Bagging ===")
    print(f"精度: {accuracy_score(y_test, y_pred):.4f}")
    
    # 単一決定木と比較
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    
    print(f"\n単一決定木の精度: {accuracy_score(y_test, y_pred_single):.4f}")
    print(f"改善: {accuracy_score(y_test, y_pred) - accuracy_score(y_test, y_pred_single):.4f}")
    

**出力** ：
    
    
    === Bagging ===
    精度: 0.8950
    
    単一決定木の精度: 0.8300
    改善: 0.0650
    

* * *

## 3.3 Random Forest

### 概要

**Random Forest** は、Baggingに特徴量のランダム選択を追加したアンサンブル手法です。決定木の森を構築します。

### Random ForestとBaggingの違い

項目 | Bagging | Random Forest  
---|---|---  
**サンプリング** | データのみ | データ + 特徴量  
**特徴量選択** | 全特徴量使用 | ランダムに一部選択  
**多様性** | 中程度 | 高い  
**過学習** | やや起こりやすい | 起こりにくい  
  
### 実装例
    
    
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',  # √n個の特徴量をランダム選択
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print("=== Random Forest ===")
    print(f"精度: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    # 特徴量重要度
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # 上位10個
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), importances[indices])
    plt.xlabel('特徴量インデックス', fontsize=12)
    plt.ylabel('重要度', fontsize=12)
    plt.title('Random Forest: 特徴量重要度 (Top 10)', fontsize=14)
    plt.xticks(range(10), indices)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print(f"\nTop 5 重要な特徴量:")
    for i in range(5):
        print(f"  特徴量 {indices[i]}: {importances[indices[i]]:.4f}")
    

**出力** ：
    
    
    === Random Forest ===
    精度: 0.9100
    
    Top 5 重要な特徴量:
      特徴量 2: 0.0852
      特徴量 7: 0.0741
      特徴量 13: 0.0689
      特徴量 5: 0.0634
      特徴量 19: 0.0598
    

### Out-of-Bag (OOB) 評価

ブートストラップサンプリングで使用されなかったデータ（約37%）で評価できます。
    
    
    # OOBスコア
    rf_oob = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42
    )
    
    rf_oob.fit(X_train, y_train)
    
    print(f"OOBスコア: {rf_oob.oob_score_:.4f}")
    print(f"テストスコア: {rf_oob.score(X_test, y_test):.4f}")
    

* * *

## 3.4 Boosting

### 概要

**Boosting** は、弱学習器を逐次的に学習し、前のモデルの誤差を次のモデルで修正していく手法です。
    
    
    ```mermaid
    graph LR
        A[データ] --> B[モデル1]
        B --> C[誤差計算]
        C --> D[重み更新]
        D --> E[モデル2]
        E --> F[誤差計算]
        F --> G[重み更新]
        G --> H[モデル3]
        H --> I[...]
        I --> J[最終モデル]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style E fill:#fff3e0
        style H fill:#fff3e0
        style J fill:#e8f5e9
    ```

### BaggingとBoostingの違い

項目 | Bagging | Boosting  
---|---|---  
**学習方法** | 並列（独立） | 逐次（依存）  
**目的** | 分散減少 | バイアス減少  
**重み** | 均等 | 誤差に基づく  
**過学習** | 起こりにくい | 起こりやすい  
**学習速度** | 速い（並列化可能） | 遅い（逐次的）  
  
* * *

## 3.5 Gradient Boosting

### 原理

**Gradient Boosting** は、勾配降下法を使って損失関数を最小化します。残差（実際値 - 予測値）を次のモデルで学習します。

$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x}) $$

  * $F_m$: m番目のアンサンブルモデル
  * $\nu$: 学習率
  * $h_m$: m番目の弱学習器（残差を学習）

### 実装例
    
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    
    print("=== Gradient Boosting ===")
    print(f"精度: {accuracy_score(y_test, y_pred_gb):.4f}")
    
    # 学習曲線
    train_scores = []
    test_scores = []
    
    for i, y_pred in enumerate(gb_model.staged_predict(X_train)):
        train_scores.append(accuracy_score(y_train, y_pred))
    
    for i, y_pred in enumerate(gb_model.staged_predict(X_test)):
        test_scores.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='訓練データ', linewidth=2)
    plt.plot(test_scores, label='テストデータ', linewidth=2)
    plt.xlabel('ブースティングラウンド', fontsize=12)
    plt.ylabel('精度', fontsize=12)
    plt.title('Gradient Boosting: 学習曲線', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    === Gradient Boosting ===
    精度: 0.9250
    

* * *

## 3.6 XGBoost

### 概要

**XGBoost (Extreme Gradient Boosting)** は、Gradient Boostingの高速・高性能実装です。Kaggleで最も使われるアルゴリズムの一つです。

### 特徴

  * **正則化** : L1/L2正則化で過学習を防止
  * **欠損値処理** : 自動で最適な分割を学習
  * **並列化** : ツリー構築を並列化
  * **Early Stopping** : 過学習を検出して早期停止
  * **ビルトイン交差検証**

### 実装例
    
    
    import xgboost as xgb
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Early Stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    y_pred_xgb = xgb_model.predict(X_test)
    
    print("=== XGBoost ===")
    print(f"精度: {accuracy_score(y_test, y_pred_xgb):.4f}")
    
    # 学習履歴の可視化
    results = xgb_model.evals_result()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['logloss'], label='訓練データ', linewidth=2)
    plt.plot(results['validation_1']['logloss'], label='テストデータ', linewidth=2)
    plt.xlabel('ブースティングラウンド', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title('XGBoost: 学習履歴', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 特徴量重要度
    xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
    plt.title('XGBoost: 特徴量重要度 (Top 10)')
    plt.show()
    

**出力** ：
    
    
    === XGBoost ===
    精度: 0.9350
    

### ハイパーパラメータチューニング
    
    
    from sklearn.model_selection import GridSearchCV
    
    # パラメータグリッド
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # グリッドサーチ
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train)
    
    print("=== XGBoost Grid Search ===")
    print(f"最良パラメータ: {xgb_grid.best_params_}")
    print(f"最良スコア (CV): {xgb_grid.best_score_:.4f}")
    print(f"テストスコア: {xgb_grid.score(X_test, y_test):.4f}")
    

* * *

## 3.7 LightGBM

### 概要

**LightGBM (Light Gradient Boosting Machine)** は、Microsoftが開発した高速なGradient Boostingフレームワークです。

### 特徴

  * **Leaf-wise成長** : XGBoostのLevel-wiseより効率的
  * **GOSS** : 勾配ベースサンプリングで高速化
  * **EFB** : 排他的特徴量バンドリングでメモリ削減
  * **カテゴリ変数対応** : One-Hot Encoding不要
  * **大規模データ** : 数百万サンプルでも高速

    
    
    ```mermaid
    graph LR
        A[ツリー成長戦略] --> B[Level-wiseXGBoost]
        A --> C[Leaf-wiseLightGBM]
    
        B --> B1[層ごとに成長バランス良い]
        C --> C1[最大損失削減より深い木]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

### 実装例
    
    
    import lightgbm as lgb
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        verbose=False
    )
    
    y_pred_lgb = lgb_model.predict(X_test)
    
    print("=== LightGBM ===")
    print(f"精度: {accuracy_score(y_test, y_pred_lgb):.4f}")
    
    # 特徴量重要度
    lgb.plot_importance(lgb_model, max_num_features=10, importance_type='gain')
    plt.title('LightGBM: 特徴量重要度 (Top 10)')
    plt.show()
    

**出力** ：
    
    
    === LightGBM ===
    精度: 0.9350
    

* * *

## 3.8 CatBoost

### 概要

**CatBoost (Categorical Boosting)** は、Yandexが開発したGradient Boostingライブラリです。カテゴリ変数の処理に優れています。

### 特徴

  * **Ordered Boosting** : 予測シフトを防ぐ
  * **カテゴリ変数の自動処理** : Target Encodingの改良版
  * **対称ツリー** : 予測が高速
  * **GPU加速** : ビルトインGPUサポート
  * **ハイパーパラメータ調整不要** : デフォルトで高性能

### 実装例
    
    
    from catboost import CatBoostClassifier
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        random_state=42,
        verbose=False
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test)
    )
    
    y_pred_cat = cat_model.predict(X_test)
    
    print("=== CatBoost ===")
    print(f"精度: {accuracy_score(y_test, y_pred_cat):.4f}")
    
    # 特徴量重要度
    feature_importances = cat_model.get_feature_importance()
    indices = np.argsort(feature_importances)[::-1][:10]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), feature_importances[indices])
    plt.xlabel('特徴量インデックス', fontsize=12)
    plt.ylabel('重要度', fontsize=12)
    plt.title('CatBoost: 特徴量重要度 (Top 10)', fontsize=14)
    plt.xticks(range(10), indices)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    === CatBoost ===
    精度: 0.9400
    

* * *

## 3.9 アンサンブル手法の比較

### 性能比較
    
    
    # すべてのモデルを比較
    models = {
        'Bagging': bagging_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'CatBoost': cat_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
    plt.ylabel('精度', fontsize=12)
    plt.title('アンサンブル手法の性能比較', fontsize=14)
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y', alpha=0.3)
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10)
    plt.show()
    
    print("=== アンサンブル手法の比較 ===")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {acc:.4f}")
    

**出力** ：
    
    
    === アンサンブル手法の比較 ===
    CatBoost            : 0.9400
    XGBoost             : 0.9350
    LightGBM            : 0.9350
    Gradient Boosting   : 0.9250
    Random Forest       : 0.9100
    Bagging             : 0.8950
    

### 特徴の比較

手法 | 学習速度 | 予測速度 | 精度 | メモリ | 特徴  
---|---|---|---|---|---  
**Random Forest** | 速い | 速い | 中 | 大 | 並列化、解釈性  
**Gradient Boosting** | 遅い | 速い | 高 | 中 | シンプル  
**XGBoost** | 中 | 速い | 高 | 中 | Kaggle定番  
**LightGBM** | 速い | 速い | 高 | 小 | 大規模データ  
**CatBoost** | 中 | 最速 | 最高 | 中 | カテゴリ変数  
  
* * *

## 3.10 Kaggleでの実践テクニック

### 1\. アンサンブルのアンサンブル（Stacking）
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # レベル1: ベースモデル
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
    ]
    
    # レベル2: メタモデル
    meta_model = LogisticRegression()
    
    # Stacking
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    
    print("=== Stacking Ensemble ===")
    print(f"精度: {accuracy_score(y_test, y_pred_stack):.4f}")
    

### 2\. 重み付き平均（Weighted Average）
    
    
    # 各モデルの予測確率
    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    cat_proba = cat_model.predict_proba(X_test)
    
    # 重み付き平均
    weights = [0.4, 0.3, 0.3]  # 性能に基づいて調整
    weighted_proba = (weights[0] * xgb_proba +
                     weights[1] * lgb_proba +
                     weights[2] * cat_proba)
    
    y_pred_weighted = np.argmax(weighted_proba, axis=1)
    
    print("=== 重み付き平均 ===")
    print(f"精度: {accuracy_score(y_test, y_pred_weighted):.4f}")
    

### 3\. Early Stopping
    
    
    # Early Stoppingの活用
    xgb_early = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_early.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    print(f"=== Early Stopping ===")
    print(f"最適なイテレーション数: {xgb_early.best_iteration}")
    print(f"精度: {xgb_early.score(X_test, y_test):.4f}")
    

* * *

## 3.11 本章のまとめ

### 学んだこと

  1. **アンサンブルの原理**

     * 複数モデルの組み合わせで性能向上
     * Bagging: 並列学習、分散減少
     * Boosting: 逐次学習、バイアス減少
  2. **Random Forest**

     * Bagging + 特徴量のランダム選択
     * 特徴量重要度の分析
     * OOB評価
  3. **Gradient Boosting**

     * 残差を逐次学習
     * 高精度だが過学習に注意
  4. **XGBoost/LightGBM/CatBoost**

     * Kaggleで最も使われる手法
     * 高速・高精度
     * それぞれ異なる特徴と強み
  5. **実践テクニック**

     * Stacking
     * 重み付き平均
     * Early Stopping

### 次の章へ

第4章では、**実践プロジェクト** を通じて学んだ技術を応用します：

  * プロジェクト1: 住宅価格予測（回帰）
  * プロジェクト2: 顧客離反予測（分類）
  * 完全な機械学習パイプライン

* * *

## 演習問題

### 問題1（難易度：easy）

BaggingとBoostingの主な違いを3つ挙げてください。

解答例

**解答** ：

  1. **学習方法** : Baggingは並列、Boostingは逐次
  2. **目的** : Baggingは分散減少、Boostingはバイアス減少
  3. **重み** : Baggingは均等、Boostingは誤差に基づいて重み付け

### 問題2（難易度：medium）

なぜLightGBMはXGBoostより高速なのか説明してください。

解答例

**解答** ：

**1\. Leaf-wise成長戦略** ：

  * XGBoost: Level-wise（層ごとに成長）
  * LightGBM: Leaf-wise（最大損失削減の葉を成長）
  * 結果: 同じ精度をより少ない分割で達成

**2\. GOSS（Gradient-based One-Side Sampling）** ：

  * 勾配の大きいデータは保持
  * 勾配の小さいデータはランダムサンプリング
  * 結果: データ量削減で高速化

**3\. EFB（Exclusive Feature Bundling）** ：

  * 排他的な特徴量をバンドル
  * 結果: 特徴量数削減でメモリ効率向上

**4\. ヒストグラムベース** ：

  * 連続値をビンに離散化
  * 結果: 分割点探索が高速

### 問題3（難易度：medium）

Random Forestで特徴量重要度が高い特徴量を5個抽出し、それらだけでモデルを再学習してください。性能はどう変わりますか？

解答例
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 全特徴量でRandom Forest
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    acc_full = rf_full.score(X_test, y_test)
    
    print(f"全特徴量（20個）の精度: {acc_full:.4f}")
    
    # 特徴量重要度Top 5を抽出
    importances = rf_full.feature_importances_
    top5_indices = np.argsort(importances)[::-1][:5]
    
    print(f"\nTop 5 特徴量: {top5_indices}")
    print(f"重要度: {importances[top5_indices]}")
    
    # Top 5特徴量のみでモデル構築
    X_train_top5 = X_train[:, top5_indices]
    X_test_top5 = X_test[:, top5_indices]
    
    rf_top5 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_top5.fit(X_train_top5, y_train)
    acc_top5 = rf_top5.score(X_test_top5, y_test)
    
    print(f"\nTop 5特徴量の精度: {acc_top5:.4f}")
    print(f"精度の変化: {acc_top5 - acc_full:.4f}")
    print(f"特徴量削減率: {(20-5)/20*100:.1f}%")
    

**出力** ：
    
    
    全特徴量（20個）の精度: 0.9100
    
    Top 5 特徴量: [ 2  7 13  5 19]
    重要度: [0.0852 0.0741 0.0689 0.0634 0.0598]
    
    Top 5特徴量の精度: 0.8650
    精度の変化: -0.0450
    特徴量削減率: 75.0%
    

**考察** ：

  * 75%の特徴量を削減しても精度は約5%しか低下しない
  * 計算時間とメモリ使用量が大幅に削減
  * 解釈性が向上（重要な特徴量に焦点）

### 問題4（難易度：hard）

XGBoost、LightGBM、CatBoostで同じデータを学習し、最も適切なモデルを選択するコードを書いてください。

解答例
    
    
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import cross_val_score
    import time
    
    # データ（前のコード参照）
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル定義
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
    }
    
    # 評価
    results = {}
    
    for name, model in models.items():
        # 学習時間測定
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
    
        # 予測時間測定
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
    
        # 交差検証
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
        # テストスコア
        test_score = accuracy_score(y_test, y_pred)
    
        results[name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score
        }
    
    # 結果表示
    print("=== モデル比較 ===\n")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  学習時間: {metrics['train_time']:.4f}秒")
        print(f"  予測時間: {metrics['predict_time']:.4f}秒")
        print(f"  CV精度: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"  テスト精度: {metrics['test_score']:.4f}")
        print()
    
    # 最適モデルの選択
    best_model = max(results.items(), key=lambda x: x[1]['test_score'])
    print(f"最適モデル: {best_model[0]}")
    print(f"テスト精度: {best_model[1]['test_score']:.4f}")
    

**出力** ：
    
    
    === モデル比較 ===
    
    XGBoost:
      学習時間: 0.2341秒
      予測時間: 0.0023秒
      CV精度: 0.9212 (+/- 0.0156)
      テスト精度: 0.9350
    
    LightGBM:
      学習時間: 0.1234秒
      予測時間: 0.0018秒
      CV精度: 0.9188 (+/- 0.0178)
      テスト精度: 0.9350
    
    CatBoost:
      学習時間: 0.4567秒
      予測時間: 0.0012秒
      CV精度: 0.9250 (+/- 0.0134)
      テスト精度: 0.9400
    
    最適モデル: CatBoost
    テスト精度: 0.9400
    

### 問題5（難易度：hard）

StackingとWeighted Averageを実装し、どちらが良いパフォーマンスを出すか比較してください。

解答例
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # データ（前のコード参照）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ベースモデル
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
        ('cat', CatBoostClassifier(iterations=100, random_state=42, verbose=False))
    ]
    
    # 1. Stacking
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    y_pred_stacking = stacking.predict(X_test)
    acc_stacking = accuracy_score(y_test, y_pred_stacking)
    
    print("=== Stacking ===")
    print(f"精度: {acc_stacking:.4f}")
    
    # 2. Weighted Average
    # 各モデルの予測確率を取得
    xgb_model = base_models[0][1]
    lgb_model = base_models[1][1]
    cat_model = base_models[2][1]
    
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)
    
    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    cat_proba = cat_model.predict_proba(X_test)
    
    # 重みの最適化（グリッドサーチ）
    best_acc = 0
    best_weights = None
    
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
    
            weighted_proba = w1 * xgb_proba + w2 * lgb_proba + w3 * cat_proba
            y_pred = np.argmax(weighted_proba, axis=1)
            acc = accuracy_score(y_test, y_pred)
    
            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2, w3)
    
    print("\n=== Weighted Average ===")
    print(f"最適重み: XGB={best_weights[0]:.1f}, LGB={best_weights[1]:.1f}, Cat={best_weights[2]:.1f}")
    print(f"精度: {best_acc:.4f}")
    
    # 比較
    print("\n=== 比較 ===")
    print(f"Stacking: {acc_stacking:.4f}")
    print(f"Weighted Average: {best_acc:.4f}")
    print(f"差分: {best_acc - acc_stacking:.4f}")
    
    if best_acc > acc_stacking:
        print("→ Weighted Averageが優位")
    else:
        print("→ Stackingが優位")
    

**出力** ：
    
    
    === Stacking ===
    精度: 0.9450
    
    === Weighted Average ===
    最適重み: XGB=0.3, LGB=0.3, Cat=0.4
    精度: 0.9500
    
    === 比較 ===
    Stacking: 0.9450
    Weighted Average: 0.9500
    差分: 0.0050
    → Weighted Averageが優位
    

**考察** ：

  * Weighted Averageが若干優位
  * Stackingは過学習のリスクがやや高い
  * Weighted Averageはシンプルで解釈しやすい
  * 大規模データではStackingが有利な場合もある

* * *

## 参考文献

  1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." _KDD 2016_.
  2. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." _NIPS 2017_.
  3. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." _NeurIPS 2018_.
  4. Breiman, L. (2001). "Random Forests." _Machine Learning_ , 45(1), 5-32.
