---
title: 第1章：アンサンブル学習の基礎
chapter_title: 第1章：アンサンブル学習の基礎
subtitle: 複数モデルの組み合わせによる予測精度向上 - バギング、ブースティング、スタッキングの原理
reading_time: 20-25分
difficulty: 初級〜中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ アンサンブル学習の原理を理解する
  * ✅ バイアス・バリアンス分解の概念を説明できる
  * ✅ バギング（Random Forest、Extra Trees）を実装できる
  * ✅ ブースティング（AdaBoost、Gradient Boosting）の仕組みを理解する
  * ✅ スタッキングとメタ学習器を使いこなせる
  * ✅ 各手法の使い分けができる

* * *

## 1.1 アンサンブル学習とは

### 定義

**アンサンブル学習（Ensemble Learning）** は、複数の弱学習器（weak learner）を組み合わせて、より強力な予測モデルを構築する機械学習手法です。

> 「複数のモデルを組み合わせることで、単一モデルより高い性能を実現する」

### なぜ複数のモデルを組み合わせるのか
    
    
    ```mermaid
    graph LR
        A[単一モデルの限界] --> B[過学習しやすい]
        A --> C[バイアスが高い]
        A --> D[ノイズに敏感]
    
        E[アンサンブル] --> F[分散を減らす]
        E --> G[バイアスを減らす]
        E --> H[安定性向上]
    
        style A fill:#ffebee
        style E fill:#e8f5e9
    ```

### アンサンブルの効果

**例** ：3つのモデルがそれぞれ70%の精度で独立に予測する場合

多数決による精度：

$$ P(\text{正解}) = P(\text{2つ以上正解}) = \binom{3}{2}(0.7)^2(0.3) + \binom{3}{3}(0.7)^3 = 0.784 $$

単一モデル（70%）より高い精度（78.4%）を達成！

### バイアス・バリアンス分解

予測誤差は以下のように分解されます：

$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

要素 | 意味 | 対策  
---|---|---  
**バイアス** | モデルの単純化による誤差 | 複雑なモデル、ブースティング  
**バリアンス** | 訓練データの変動への敏感さ | バギング、平均化  
**既約誤差** | データに含まれるノイズ | 削減不可能  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    
    # データ生成
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 異なる深さの決定木で予測
    depths = [1, 3, 10]
    plt.figure(figsize=(15, 4))
    
    for i, depth in enumerate(depths, 1):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
    
        X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_pred = model.predict(X_plot)
    
        plt.subplot(1, 3, i)
        plt.scatter(X_train, y_train, alpha=0.5, label='訓練データ')
        plt.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'深さ={depth}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'深さ={depth}: {"高バイアス" if depth==1 else "高バリアンス" if depth==10 else "バランス"}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.2 バギング（Bagging）

### 概要

**バギング（Bootstrap Aggregating）** は、訓練データをブートストラップサンプリングして複数のモデルを訓練し、予測を平均化（回帰）または多数決（分類）する手法です。

### アルゴリズム

  1. 訓練データから**ブートストラップサンプル** を $B$ 個生成
  2. 各サンプルで独立にモデルを訓練
  3. 予測を集約： 
     * 回帰: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)$
     * 分類: 多数決

    
    
    ```mermaid
    graph TD
        A[訓練データ] --> B1[ブートストラップ1]
        A --> B2[ブートストラップ2]
        A --> B3[ブートストラップ3]
    
        B1 --> M1[モデル1]
        B2 --> M2[モデル2]
        B3 --> M3[モデル3]
    
        M1 --> AGG[集約]
        M2 --> AGG
        M3 --> AGG
    
        AGG --> PRED[最終予測]
    
        style A fill:#e3f2fd
        style AGG fill:#fff3e0
        style PRED fill:#e8f5e9
    ```

### Random Forest

**Random Forest** は、バギング + ランダム特徴量選択を組み合わせた手法です。

**特徴** ：

  * 各分割でランダムに選んだ特徴量のサブセットから最適な分割を選択
  * モデル間の相関を減らし、多様性を向上

    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 単一の決定木
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print("=== バギングの効果 ===")
    print(f"決定木（単一）: {dt_acc:.4f}")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"改善: {(rf_acc - dt_acc):.4f}")
    

**出力** ：
    
    
    === バギングの効果 ===
    決定木（単一）: 0.8600
    Random Forest: 0.9250
    改善: 0.0650
    

### Extra Trees

**Extra Trees（Extremely Randomized Trees）** は、Random Forestのさらにランダム化されたバージョンです。

**相違点** ：

  * 分割の閾値もランダムに選択
  * ブートストラップサンプリングを使わない（全データを使用）

    
    
    from sklearn.ensemble import ExtraTreesClassifier
    
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_train, y_train)
    et_acc = accuracy_score(y_test, et.predict(X_test))
    
    print("=== Extra Trees vs Random Forest ===")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Extra Trees: {et_acc:.4f}")
    

* * *

## 1.3 ブースティング（Boosting）

### 概要

**ブースティング** は、弱学習器を順次訓練し、前のモデルの誤りを次のモデルで修正する手法です。
    
    
    ```mermaid
    graph LR
        A[データ] --> M1[モデル1]
        M1 --> W1[重み更新]
        W1 --> M2[モデル2]
        M2 --> W2[重み更新]
        W2 --> M3[モデル3]
        M3 --> F[加重和]
    
        style A fill:#e3f2fd
        style F fill:#e8f5e9
    ```

### AdaBoost

**AdaBoost（Adaptive Boosting）** は、誤分類されたサンプルの重みを増やしながら順次モデルを訓練します。

**アルゴリズム** ：

  1. すべてのサンプルの重み $w_i = \frac{1}{m}$ で初期化
  2. 各反復 $t = 1, ..., T$: 
     * 重み付きデータで弱学習器 $h_t$ を訓練
     * 誤差率 $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
     * モデル重み $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
     * サンプル重み更新
  3. 最終予測: $H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$

    
    
    from sklearn.ensemble import AdaBoostClassifier
    
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada.predict(X_test))
    
    print("=== AdaBoost ===")
    print(f"精度: {ada_acc:.4f}")
    
    # 反復ごとの精度推移
    from sklearn.metrics import accuracy_score
    
    n_trees = [1, 5, 10, 25, 50, 100]
    train_scores = []
    test_scores = []
    
    for n in n_trees:
        ada_temp = AdaBoostClassifier(n_estimators=n, random_state=42)
        ada_temp.fit(X_train, y_train)
        train_scores.append(ada_temp.score(X_train, y_train))
        test_scores.append(ada_temp.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, train_scores, 'o-', label='訓練データ', linewidth=2)
    plt.plot(n_trees, test_scores, 's-', label='テストデータ', linewidth=2)
    plt.xlabel('弱学習器の数', fontsize=12)
    plt.ylabel('精度', fontsize=12)
    plt.title('AdaBoost: 学習器数と精度の関係', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### Gradient Boosting基礎

**Gradient Boosting** は、損失関数の勾配方向にモデルを追加していく手法です。

**アルゴリズム** ：

  1. 初期予測 $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{m} L(y_i, \gamma)$
  2. 各反復 $t = 1, ..., T$: 
     * 残差（負の勾配）を計算: $r_i = -\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$
     * 残差に対して弱学習器 $h_t$ を訓練
     * モデル更新: $F_t(x) = F_{t-1}(x) + \nu \cdot h_t(x)$

    
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    
    print("=== Gradient Boosting ===")
    print(f"精度: {gb_acc:.4f}")
    

* * *

## 1.4 スタッキング（Stacking）

### 概要

**スタッキング** は、複数の異なるモデル（ベースモデル）の予測を入力として、メタ学習器が最終予測を行う手法です。
    
    
    ```mermaid
    graph TD
        A[訓練データ] --> M1[モデル1: ロジスティック回帰]
        A --> M2[モデル2: Random Forest]
        A --> M3[モデル3: SVM]
    
        M1 --> P1[予測1]
        M2 --> P2[予測2]
        M3 --> P3[予測3]
    
        P1 --> META[メタ学習器]
        P2 --> META
        P3 --> META
    
        META --> FINAL[最終予測]
    
        style A fill:#e3f2fd
        style META fill:#fff3e0
        style FINAL fill:#e8f5e9
    ```

### メタ学習器

メタ学習器（Meta-learner）は、ベースモデルの予測を特徴量として学習します。

**一般的なメタ学習器** ：

  * ロジスティック回帰
  * Ridge回帰
  * ニューラルネットワーク

### 交差検証戦略

過学習を防ぐため、**K-Fold交差検証** を使ってベースモデルの予測を生成します。
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # ベースモデル
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=50, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
    ]
    
    # スタッキング
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    stack.fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stack.predict(X_test))
    
    print("=== スタッキング ===")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Extra Trees: {et_acc:.4f}")
    print(f"AdaBoost: {ada_acc:.4f}")
    print(f"スタッキング: {stack_acc:.4f}")
    

* * *

## 1.5 比較と使い分け

### 性能比較

手法 | バリアンス削減 | バイアス削減 | 並列化 | 訓練速度  
---|---|---|---|---  
**バギング** | ✓ | - | 可能 | 速い  
**Random Forest** | ✓✓ | - | 可能 | 速い  
**AdaBoost** | - | ✓ | 不可 | 中程度  
**Gradient Boosting** | - | ✓✓ | 不可 | 遅い  
**スタッキング** | ✓ | ✓ | 可能 | 遅い  
  
### 適用場面

状況 | 推奨手法 | 理由  
---|---|---  
**高バリアンスモデル** | バギング、Random Forest | 分散を効果的に削減  
**高バイアスモデル** | ブースティング | 複雑なパターンを学習  
**大規模データ** | Random Forest | 並列化可能で高速  
**不均衡データ** | AdaBoost | 誤分類サンプルに注目  
**最高性能追求** | スタッキング、GB | 複数手法の長所を統合  
      
    
    # 全手法の比較
    results = {
        '決定木': dt_acc,
        'Random Forest': rf_acc,
        'Extra Trees': et_acc,
        'AdaBoost': ada_acc,
        'Gradient Boosting': gb_acc,
        'スタッキング': stack_acc
    }
    
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(methods, accuracies, color=['#e74c3c', '#3498db', '#2ecc71',
                                               '#f39c12', '#9b59b6', '#1abc9c'])
    plt.ylabel('精度', fontsize=12)
    plt.title('アンサンブル手法の性能比較', fontsize=14)
    plt.xticks(rotation=15, ha='right')
    plt.ylim([0.8, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 最終結果 ===")
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:20s}: {acc:.4f}")
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **アンサンブル学習の原理**

     * 複数のモデルを組み合わせて性能向上
     * バイアス・バリアンス分解
  2. **バギング**

     * ブートストラップサンプリングと平均化
     * Random Forest: ランダム特徴量選択で多様性向上
     * Extra Trees: さらなるランダム化
  3. **ブースティング**

     * AdaBoost: 誤分類サンプルに注目
     * Gradient Boosting: 勾配方向への最適化
  4. **スタッキング**

     * メタ学習器による統合
     * 交差検証による過学習防止
  5. **使い分け**

     * バギング: バリアンス削減、並列化
     * ブースティング: バイアス削減、高精度
     * スタッキング: 最高性能追求

### 次の章へ

第2章では、**勾配ブースティングの発展** を学びます：

  * XGBoost
  * LightGBM
  * CatBoost
  * ハイパーパラメータチューニング

* * *

## 演習問題

### 問題1（難易度：easy）

バイアスとバリアンスの違いを説明し、それぞれを削減するアンサンブル手法を挙げてください。

解答例

**バイアス（Bias）** ：

  * モデルの単純化による誤差
  * 訓練データに対しても高い誤差
  * 例: 線形モデルで非線形データを予測

**バリアンス（Variance）** ：

  * 訓練データの変動への敏感さ
  * 訓練データには良いがテストデータに悪い
  * 例: 深い決定木の過学習

**削減手法** ：

  * **バリアンス削減** : バギング、Random Forest（平均化により分散を削減）
  * **バイアス削減** : ブースティング（逐次的に複雑なパターンを学習）

### 問題2（難易度：medium）

Random ForestとExtra Treesの違いを2つ挙げ、それぞれの特徴を説明してください。

解答例

**違い1: サンプリング**

  * **Random Forest** : ブートストラップサンプリング（復元抽出）
  * **Extra Trees** : 全データを使用（サンプリングなし）

**違い2: 分割方法**

  * **Random Forest** : ランダムな特徴量サブセットから最適な分割を選択
  * **Extra Trees** : 特徴量も閾値もランダムに選択

**特徴** ：

  * **Random Forest** : バリアンス削減、訓練にやや時間
  * **Extra Trees** : より高速、さらなるランダム化により多様性向上

### 問題3（難易度：medium）

AdaBoostで誤分類されたサンプルの重みが増加する理由を、アルゴリズムの観点から説明してください。

解答例

**理由** ：

  * AdaBoostは、各反復で前のモデルが**苦手なサンプルに注目** するように設計されている
  * 誤分類サンプルの重みを増やすことで、次のモデルはそれらを正しく分類しようとする

**アルゴリズム** ：
    
    
    誤分類されたサンプル i の重み更新:
    w_i ← w_i * exp(α_t)
    
    正しく分類されたサンプル j の重み更新:
    w_j ← w_j * exp(-α_t)
    
    ここで α_t = 0.5 * ln((1 - ε_t) / ε_t) > 0
    

**効果** ：

  * 弱学習器が順次、困難なサンプルを学習
  * 最終的に複雑な決定境界を形成

### 問題4（難易度：hard）

スタッキングでK-Fold交差検証を使う理由を、過学習の観点から説明してください。

解答例

**問題** ：訓練データ全体でベースモデルを訓練し、同じデータで予測を生成すると：

  * メタ学習器が訓練データに過学習
  * ベースモデルの予測が「見たことがあるデータ」に最適化

**K-Fold交差検証の解決策** ：
    
    
    1. データをK個に分割
    2. 各Fold k について:
       - Fold k 以外でベースモデルを訓練
       - Fold k の予測を生成（未知データへの予測）
    3. 全Foldの予測を結合してメタ学習器を訓練
    

**効果** ：

  * メタ学習器への入力は「未知データへの予測」
  * 汎化性能が向上
  * 過学習を防止

**実装例** ：
    
    
    StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5  # 5-Fold交差検証
    )
    

### 問題5（難易度：hard）

irisデータセットを使い、Random ForestとGradient Boostingを実装・比較してください。訓練時間と精度を報告してください。

解答例
    
    
    import time
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report
    
    # データ読み込み
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest
    print("=== Random Forest ===")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    
    rf_acc = rf.score(X_test, y_test)
    cv_rf = cross_val_score(rf, X, y, cv=5).mean()
    
    print(f"訓練時間: {rf_time:.4f}秒")
    print(f"テスト精度: {rf_acc:.4f}")
    print(f"交差検証精度: {cv_rf:.4f}")
    
    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    start = time.time()
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_time = time.time() - start
    
    gb_acc = gb.score(X_test, y_test)
    cv_gb = cross_val_score(gb, X, y, cv=5).mean()
    
    print(f"訓練時間: {gb_time:.4f}秒")
    print(f"テスト精度: {gb_acc:.4f}")
    print(f"交差検証精度: {cv_gb:.4f}")
    
    # 比較
    print("\n=== 比較 ===")
    print(f"精度: RF={rf_acc:.4f} vs GB={gb_acc:.4f}")
    print(f"訓練時間: RF={rf_time:.4f}秒 vs GB={gb_time:.4f}秒")
    print(f"速度比: GB/RF = {gb_time/rf_time:.2f}倍")
    

**出力例** ：
    
    
    === Random Forest ===
    訓練時間: 0.0523秒
    テスト精度: 1.0000
    交差検証精度: 0.9533
    
    === Gradient Boosting ===
    訓練時間: 0.1245秒
    テスト精度: 1.0000
    交差検証精度: 0.9467
    
    === 比較 ===
    精度: RF=1.0000 vs GB=1.0000
    訓練時間: RF=0.0523秒 vs GB=0.1245秒
    速度比: GB/RF = 2.38倍
    

**考察** ：

  * 精度はほぼ同等
  * Random Forestの方が訓練が高速（並列化可能）
  * データが小規模で単純なため、両手法とも高精度

* * *

## 参考文献

  1. Breiman, L. (1996). _Bagging predictors_. Machine Learning, 24(2), 123-140.
  2. Breiman, L. (2001). _Random forests_. Machine Learning, 45(1), 5-32.
  3. Freund, Y., & Schapire, R. E. (1997). _A decision-theoretic generalization of on-line learning and an application to boosting_. Journal of Computer and System Sciences, 55(1), 119-139.
  4. Friedman, J. H. (2001). _Greedy function approximation: A gradient boosting machine_. Annals of Statistics, 1189-1232.
