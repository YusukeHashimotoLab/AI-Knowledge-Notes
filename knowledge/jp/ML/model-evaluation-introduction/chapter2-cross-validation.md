---
title: 第2章：交差検証とデータ分割
chapter_title: 第2章：交差検証とデータ分割
subtitle: モデルの汎化性能を正しく評価するためのデータ分割戦略
reading_time: 20-25分
difficulty: 初級〜中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Train/Validation/Testセットの役割と分割方法を理解する
  * ✅ Hold-out法の問題点とその解決策を説明できる
  * ✅ K-Fold交差検証の原理と実装ができる
  * ✅ Stratified K-Foldの必要性を理解する
  * ✅ 時系列データにおける交差検証の特殊性を把握する
  * ✅ データリーケージを防ぐ前処理の方法を実践できる
  * ✅ Nested Cross-Validationでハイパーパラメータ調整ができる

* * *

## 2.1 データ分割の基礎

### なぜデータを分割するのか

機械学習モデルの最終目的は、**未知のデータに対して正確な予測を行うこと** です。そのため、モデルの性能を評価する際には、訓練データとは別の「見たことのないデータ」でテストする必要があります。

> 「訓練データでの性能が良くても、新しいデータで性能が悪ければ、そのモデルは実用的ではない」

### 3つのデータセット

セット名 | 英語名 | 役割 | 典型的な割合  
---|---|---|---  
**訓練セット** | Training Set | モデルのパラメータを学習する | 60-80%  
**検証セット** | Validation Set | ハイパーパラメータの調整とモデル選択 | 10-20%  
**テストセット** | Test Set | 最終的な汎化性能の評価 | 10-20%  
      
    
    ```mermaid
    graph LR
        A[全データセット] --> B[訓練セット 60-80%]
        A --> C[検証セット 10-20%]
        A --> D[テストセット 10-20%]
    
        B --> E[パラメータ学習]
        C --> F[ハイパーパラメータ調整]
        D --> G[最終性能評価]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
    ```

### Hold-out法：基本的なデータ分割

**Hold-out法** は、データを一度だけ訓練セットとテストセットに分割する最もシンプルな方法です。
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # サンプルデータ読み込み
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 訓練データとテストデータに分割（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"訓練データサイズ: {X_train.shape[0]}")
    print(f"テストデータサイズ: {X_test.shape[0]}")
    
    # モデル訓練
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 評価
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n訓練精度: {train_score:.4f}")
    print(f"テスト精度: {test_score:.4f}")
    

### Hold-out法の問題点

Hold-out法にはいくつかの**重大な問題** があります：

  1. **データの偏り** ：たまたま難しいサンプルがテストセットに集中する可能性
  2. **データの無駄** ：テストセットとして分離したデータは訓練に使えない
  3. **不安定性** ：分割方法によって評価結果が大きく変動する
  4. **小規模データでの問題** ：データが少ない場合、テストセットが極端に小さくなる

**実験：Hold-out法の不安定性を確認**
    
    
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # データ読み込み
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 異なるrandom_stateで10回実験
    test_scores = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
    
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        test_scores.append(score)
    
    print("=== Hold-out法の不安定性 ===")
    print(f"テスト精度の平均: {np.mean(test_scores):.4f}")
    print(f"テスト精度の標準偏差: {np.std(test_scores):.4f}")
    print(f"最小値: {np.min(test_scores):.4f}")
    print(f"最大値: {np.max(test_scores):.4f}")
    print(f"\n精度の変動幅: {np.max(test_scores) - np.min(test_scores):.4f}")
    

出力例：
    
    
    === Hold-out法の不安定性 ===
    テスト精度の平均: 0.9533
    テスト精度の標準偏差: 0.0356
    最小値: 0.9000
    最大値: 1.0000
    
    精度の変動幅: 0.1000
    

このように、同じアルゴリズムでもデータ分割方法によって**最大10%もの精度差** が生じることがあります。

* * *

## 2.2 K-Fold交差検証

### K-Fold Cross-Validationとは

**K-Fold交差検証（K-Fold Cross-Validation）** は、データをK個のサブセット（Fold）に分割し、各Foldを一度ずつテストセットとして使用する手法です。
    
    
    ```mermaid
    graph TD
        A[全データ] --> B[Fold 1]
        A --> C[Fold 2]
        A --> D[Fold 3]
        A --> E[Fold 4]
        A --> F[Fold 5]
    
        G[Round 1] --> H[Test: Fold 1Train: Fold 2,3,4,5]
        I[Round 2] --> J[Test: Fold 2Train: Fold 1,3,4,5]
        K[Round 3] --> L[Test: Fold 3Train: Fold 1,2,4,5]
        M[Round 4] --> N[Test: Fold 4Train: Fold 1,2,3,5]
        O[Round 5] --> P[Test: Fold 5Train: Fold 1,2,3,4]
    
        H --> Q[Score 1]
        J --> R[Score 2]
        L --> S[Score 3]
        N --> T[Score 4]
        P --> U[Score 5]
    
        Q --> V[平均スコア]
        R --> V
        S --> V
        T --> V
        U --> V
    
        style A fill:#e1f5ff
        style V fill:#4fc3f7
    ```

### K-Fold交差検証のアルゴリズム

  1. データを$K$個のサブセット（Fold）にランダム分割
  2. 各Fold $i$ ($i = 1, 2, ..., K$) に対して： 
     * Fold $i$ をテストセットとする
     * 残りの $K-1$ 個のFoldを訓練セットとする
     * モデルを訓練し、Fold $i$ で評価してスコア $S_i$ を得る
  3. $K$個のスコアの平均を最終評価とする： $$ \text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} S_i $$ 

### 基本的な実装
    
    
    import numpy as np
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    
    # データ読み込み
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # モデル作成
    model = LogisticRegression(max_iter=200)
    
    # 5-Fold交差検証
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    print("=== 5-Fold交差検証の結果 ===")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    
    print(f"\n平均精度: {scores.mean():.4f}")
    print(f"標準偏差: {scores.std():.4f}")
    print(f"95%信頼区間: [{scores.mean() - 1.96*scores.std():.4f}, "
          f"{scores.mean() + 1.96*scores.std():.4f}]")
    

### K-Fold交差検証の利点

  * **安定した評価** ：複数回の評価の平均を取るため、偶然の影響を減らせる
  * **データの有効活用** ：すべてのデータが訓練とテストの両方に使われる
  * **信頼区間の推定** ：標準偏差からモデルの安定性を評価できる
  * **小規模データでも有効** ：限られたデータでも信頼性の高い評価が可能

### K値の選択

K値 | 訓練データ割合 | メリット | デメリット | 推奨ケース  
---|---|---|---|---  
3 | 67% | 計算が高速 | 評価が不安定 | 大規模データ、初期実験  
5 | 80% | バランスが良い | - | 標準的な選択  
10 | 90% | 評価が安定 | 計算コストが高い | 中規模データ  
N (LOOCV) | 100% - 1 | バイアスが最小 | 計算コストが極めて高い | 小規模データ（N < 100）  
  
> **経験則** ：実務では**K=5** または**K=10** が最もよく使われます。K=5は計算効率と評価の安定性のバランスが良いため、多くの場合に推奨されます。

* * *

## 2.3 Stratified K-Fold

### クラス不均衡問題

通常のK-Fold交差検証では、各Foldにクラスが均等に分配される保証がありません。特に**クラス不均衡データ** （例：正例10%、負例90%）では、以下の問題が発生します：

  * あるFoldに特定のクラスがほとんど含まれない
  * 少数クラスがテストセットに全く含まれない
  * 評価指標が不安定になる

### Stratified K-Foldの原理

**Stratified K-Fold** は、各Foldで**クラスの比率を元データと同じに保つ** ように分割します。
    
    
    ```mermaid
    graph TD
        A[元データClass A: 70%Class B: 30%] --> B[Stratified分割]
    
        B --> C[Fold 1Class A: 70%Class B: 30%]
        B --> D[Fold 2Class A: 70%Class B: 30%]
        B --> E[Fold 3Class A: 70%Class B: 30%]
        B --> F[Fold 4Class A: 70%Class B: 30%]
        B --> G[Fold 5Class A: 70%Class B: 30%]
    
        style A fill:#e1f5ff
        style C fill:#b3e5fc
        style D fill:#b3e5fc
        style E fill:#b3e5fc
        style F fill:#b3e5fc
        style G fill:#b3e5fc
    ```

### 実装と比較
    
    
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    
    # 不均衡データを作成（正例:負例 = 1:9）
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],  # 90% vs 10%
        random_state=42
    )
    
    print(f"クラス分布: Class 0 = {np.sum(y==0)}, Class 1 = {np.sum(y==1)}")
    print(f"クラス比率: {np.sum(y==1)/len(y):.2%} が Class 1\n")
    
    # モデル作成
    model = LogisticRegression(max_iter=200)
    
    # 通常のK-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='f1')
    
    # Stratified K-Fold
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_stratified = cross_val_score(model, X, y, cv=stratified_kfold, scoring='f1')
    
    print("=== K-Fold vs Stratified K-Fold ===")
    print(f"通常のK-Fold:")
    print(f"  平均F1スコア: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
    
    print(f"\nStratified K-Fold:")
    print(f"  平均F1スコア: {scores_stratified.mean():.4f} ± {scores_stratified.std():.4f}")
    
    print(f"\n改善率: {(scores_stratified.mean() - scores_kfold.mean()) / scores_kfold.mean() * 100:.2f}%")
    

### 各Foldのクラス分布を確認
    
    
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.datasets import make_classification
    
    # 不均衡データ作成
    X, y = make_classification(
        n_samples=1000,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # 通常のK-Fold
    print("=== 通常のK-Fold のクラス分布 ===")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        y_test = y[test_idx]
        class_1_ratio = np.sum(y_test == 1) / len(y_test)
        print(f"Fold {i}: Class 1 の割合 = {class_1_ratio:.2%}")
    
    # Stratified K-Fold
    print("\n=== Stratified K-Fold のクラス分布 ===")
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y), 1):
        y_test = y[test_idx]
        class_1_ratio = np.sum(y_test == 1) / len(y_test)
        print(f"Fold {i}: Class 1 の割合 = {class_1_ratio:.2%}")
    

### いつStratified K-Foldを使うべきか

  * **必ず使うべき** ：分類問題でクラスが不均衡な場合
  * **推奨** ：すべての分類問題（バランスが良くても安定性向上）
  * **使えない** ：回帰問題（目的変数が連続値）

> **ベストプラクティス** ：分類問題では、デフォルトで`StratifiedKFold`を使いましょう。クラスがバランスしていても、評価の安定性が向上します。

* * *

## 2.4 Leave-One-Out交差検証

### Leave-One-Out Cross-Validation (LOOCV)

**LOOCV** は、K-Foldの特殊ケースで、$K = N$（データ数）とした極端な交差検証です。

  * 各イテレーションで**1つのサンプルだけ** をテストセットとする
  * 残りの$N-1$個のサンプルで訓練
  * $N$回の訓練と評価を実行

### LOOCVの特徴

項目 | LOOCV | K-Fold (K=5)  
---|---|---  
訓練データ割合 | $(N-1)/N$ ≈ 100% | 80%  
評価回数 | $N$回 | 5回  
バイアス | 極めて低い | やや高い  
分散 | 高い | 低い  
計算コスト | 非常に高い | 低い  
推奨データサイズ | $N < 100$ | 任意  
  
### 実装例
    
    
    import numpy as np
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import time
    
    # 小規模データで実験
    iris = load_iris()
    X, y = iris.data[:50], iris.target[:50]  # 50サンプルのみ
    
    model = LogisticRegression(max_iter=200)
    
    # LOOCV
    loo = LeaveOneOut()
    start_time = time.time()
    scores_loo = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
    loo_time = time.time() - start_time
    
    print("=== Leave-One-Out交差検証 ===")
    print(f"評価回数: {len(scores_loo)}回")
    print(f"平均精度: {scores_loo.mean():.4f}")
    print(f"標準偏差: {scores_loo.std():.4f}")
    print(f"実行時間: {loo_time:.3f}秒")
    
    # 5-Fold CVと比較
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    start_time = time.time()
    scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    kfold_time = time.time() - start_time
    
    print("\n=== 5-Fold交差検証（比較） ===")
    print(f"評価回数: {len(scores_kfold)}回")
    print(f"平均精度: {scores_kfold.mean():.4f}")
    print(f"標準偏差: {scores_kfold.std():.4f}")
    print(f"実行時間: {kfold_time:.3f}秒")
    
    print(f"\n速度比: LOOCV は 5-Fold の {loo_time/kfold_time:.1f}倍遅い")
    

### LOOCVのメリットとデメリット

**メリット** ：

  * 訓練データを最大限活用できる（バイアスが最小）
  * 決定論的（ランダム性がない）
  * 小規模データで有効

**デメリット** ：

  * 計算コストが極めて高い（$N$回の訓練）
  * 分散が大きい（各テストが1サンプルのみ）
  * 大規模データでは実用的でない

> **実務での使い分け** ：データ数が100未満の場合のみLOOCVを検討し、それ以外はK-Foldを使用するのが現実的です。

* * *

## 2.5 時系列データの交差検証

### 時系列データの特殊性

時系列データでは、**時間的な順序が重要** です。通常のK-Foldをランダムに適用すると、以下の問題が発生します：

  * **未来の情報リーク** ：訓練データに未来のデータが含まれる
  * **時間依存性の無視** ：過去→現在→未来という因果関係が崩れる
  * **非現実的な評価** ：実運用では常に未来を予測するのに、過去も未来も混ぜて訓練してしまう

    
    
    ```mermaid
    graph LR
        A[❌ 通常のK-Fold] --> B[時間順序を無視]
        B --> C[未来データで訓練過去データでテスト]
        C --> D[過大評価]
    
        E[✅ Time Series Split] --> F[時間順序を保持]
        F --> G[過去データで訓練未来データでテスト]
        G --> H[正しい評価]
    
        style A fill:#ffcdd2
        style E fill:#c8e6c9
    ```

### TimeSeriesSplit

**TimeSeriesSplit** は、時系列データのために設計された交差検証で、以下の特徴があります：

  * 訓練セットは常に**テストセットより前** の時間のデータ
  * 各Foldで訓練セットが**拡大** していく（累積型）
  * テストセットは常に未来の一定期間

    
    
    ```mermaid
    graph TD
        A[時系列データ: t1, t2, t3, t4, t5, t6, t7, t8, t9] --> B[Fold 1]
        A --> C[Fold 2]
        A --> D[Fold 3]
        A --> E[Fold 4]
    
        B --> F[Train: t1,t2,t3 | Test: t4,t5]
        C --> G[Train: t1,t2,t3,t4,t5 | Test: t6,t7]
        D --> H[Train: t1,t2,t3,t4,t5,t6,t7 | Test: t8,t9]
    
        style A fill:#e1f5ff
        style F fill:#b3e5fc
        style G fill:#81d4fa
        style H fill:#4fc3f7
    ```

### 実装例
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    
    # 時系列データ生成（日次売上データをシミュレート）
    np.random.seed(42)
    n_samples = 365  # 1年間の日次データ
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # トレンド + 季節性 + ノイズ
    trend = np.linspace(100, 150, n_samples)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # 週次周期
    noise = np.random.randn(n_samples) * 5
    y = trend + seasonality + noise
    
    # 特徴量作成（過去7日間の移動平均など）
    X = np.column_stack([
        np.roll(y, 1),  # 1日前
        np.roll(y, 7),  # 7日前
        pd.Series(y).rolling(7).mean().fillna(method='bfill'),  # 7日移動平均
    ])
    
    # 最初の7日間を除去（rollによるデータ不足を回避）
    X, y = X[7:], y[7:]
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("=== TimeSeriesSplit の分割パターン ===")
    for i, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"Fold {i}:")
        print(f"  訓練: {len(train_index)}サンプル (index {train_index[0]} ~ {train_index[-1]})")
        print(f"  テスト: {len(test_index)}サンプル (index {test_index[0]} ~ {test_index[-1]})")
    
    # モデル評価
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    print(f"\n=== 各Foldの性能 ===")
    for i, rmse in enumerate(rmse_scores, 1):
        print(f"Fold {i} RMSE: {rmse:.2f}")
    
    print(f"\n平均RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    

### 通常のK-Foldとの比較
    
    
    import numpy as np
    from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
    from sklearn.linear_model import Ridge
    
    # 同じデータで通常のK-Foldを実行
    kfold = KFold(n_splits=5, shuffle=False)  # shuffle=Falseで時系列順序を維持
    scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    rmse_kfold = np.sqrt(-scores_kfold)
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    scores_tscv = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse_tscv = np.sqrt(-scores_tscv)
    
    print("=== K-Fold vs TimeSeriesSplit ===")
    print(f"K-Fold 平均RMSE: {rmse_kfold.mean():.2f} ± {rmse_kfold.std():.2f}")
    print(f"TimeSeriesSplit 平均RMSE: {rmse_tscv.mean():.2f} ± {rmse_tscv.std():.2f}")
    
    print(f"\n差分: {rmse_kfold.mean() - rmse_tscv.mean():.2f}")
    print("（K-Foldの方が良い結果 = 未来情報リークによる過大評価の可能性）")
    

### 時系列CVのベストプラクティス

  * **必須** ：時系列データには必ずTimeSeriesSplitを使う
  * **推奨** ：テストセット期間を実運用と同じ長さに設定する
  * **注意** ：特徴量作成時にも未来情報を使わない（ラグ特徴量を使用）
  * **検討** ：Walking Forward Validation（段階的検証）も検討する

* * *

## 2.6 データリーケージの防止

### データリーケージとは

**データリーケージ（Data Leakage）** とは、訓練データに**本来知り得ない情報** が混入し、モデルが過大評価される現象です。

> 「訓練時に未来やテストデータの情報を使ってしまい、実運用では再現できない高性能が出てしまう」

### よくあるデータリーケージのパターン

リーケージの種類 | 原因 | 結果  
---|---|---  
**前処理リーケージ** | 全データで正規化してから分割 | テストセットの統計量を訓練に使用  
**時間的リーケージ** | 未来のデータで訓練 | 因果関係が逆転  
**ターゲットリーケージ** | 目的変数から作った特徴量 | テスト時に利用不可能な情報  
**重複データ** | 同じデータが訓練とテストに存在 | 記憶による過大評価  
  
### 間違った前処理の例
    
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # ❌ 間違い：分割前に全データで正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 全データの平均・標準偏差を使用
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    score_wrong = accuracy_score(y_test, model.predict(X_test))
    
    print(f"❌ リーケージあり（間違い）: テスト精度 = {score_wrong:.4f}")
    

### 正しい前処理の例
    
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # ✅ 正しい：先に分割してから訓練データのみで正規化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 訓練データのみで学習
    X_test_scaled = scaler.transform(X_test)  # 訓練データの統計量を適用
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    score_correct = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print(f"✅ リーケージなし（正しい）: テスト精度 = {score_correct:.4f}")
    

### Pipelineを使った安全な実装

`sklearn.pipeline.Pipeline`を使うと、前処理とモデルを一体化し、リーケージを防げます。
    
    
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.datasets import make_classification
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Pipelineで前処理とモデルを統合
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=200))
    ])
    
    # 交差検証（各Foldで自動的に正しく前処理される）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    print("=== Pipeline + Cross-Validation ===")
    print(f"各Foldで訓練データのみを使って正規化が実行されます")
    print(f"平均精度: {scores.mean():.4f} ± {scores.std():.4f}")
    

### Pipelineのメリット

  * **安全性** ：リーケージを構造的に防止
  * **簡潔性** ：コードが短く読みやすい
  * **再現性** ：前処理パラメータが一緒に保存される
  * **ハイパーパラメータ調整** ：GridSearchCVでも正しく動作

> **ベストプラクティス** ：実務では必ず`Pipeline`を使い、前処理をモデルと一体化させましょう。これにより、データリーケージを防ぎ、コードの保守性も向上します。

* * *

## 2.7 Group K-Fold

### グループ構造を持つデータ

実世界のデータには、**同じグループに属するサンプルが複数存在** するケースがあります：

  * **医療データ** ：同じ患者から複数の測定値
  * **画像データ** ：同じ人物の複数枚の写真
  * **時系列データ** ：同じ店舗の複数日のデータ

通常のK-Foldでは、**同じグループのデータが訓練とテストに分散** してしまい、過大評価につながります。

### Group K-Foldの原理

**Group K-Fold** は、同じグループのデータを必ず同じFoldに配置します。
    
    
    ```mermaid
    graph TD
        A[データPatient A: 5枚Patient B: 3枚Patient C: 4枚] --> B[Group K-Fold]
    
        B --> C[Fold 1Patient A の全データ]
        B --> D[Fold 2Patient B の全データ]
        B --> E[Fold 3Patient C の全データ]
    
        F[❌ 通常のK-Fold] --> G[Patient A のデータが訓練とテストに分散]
    
        style A fill:#e1f5ff
        style B fill:#c8e6c9
        style F fill:#ffcdd2
    ```

### 実装例
    
    
    import numpy as np
    from sklearn.model_selection import GroupKFold, KFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # グループ構造を持つデータ作成
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # グループID（例：患者ID、店舗IDなど）
    # 患者1が30サンプル、患者2が25サンプル、患者3が20サンプル...
    groups = np.array([1]*30 + [2]*25 + [3]*20 + [4]*15 + [5]*10)
    
    print(f"グループ数: {len(np.unique(groups))}")
    print(f"各グループのサンプル数: {[np.sum(groups==g) for g in np.unique(groups)]}\n")
    
    # モデル
    model = LogisticRegression(max_iter=200)
    
    # Group K-Fold
    group_kfold = GroupKFold(n_splits=5)
    scores_group = cross_val_score(model, X, y, cv=group_kfold.split(X, y, groups),
                                    scoring='accuracy')
    
    print("=== Group K-Fold ===")
    print(f"平均精度: {scores_group.mean():.4f} ± {scores_group.std():.4f}")
    
    # 通常のK-Fold（比較）
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    print("\n=== 通常のK-Fold（参考） ===")
    print(f"平均精度: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
    
    print(f"\n差分: {scores_kfold.mean() - scores_group.mean():.4f}")
    print("（K-Foldの方が良い = グループリーケージによる過大評価）")
    

### 各Foldのグループ構成を確認
    
    
    import numpy as np
    from sklearn.model_selection import GroupKFold
    
    # グループK-Foldの分割パターンを確認
    group_kfold = GroupKFold(n_splits=5)
    
    print("=== Group K-Fold の分割パターン ===")
    for i, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
    
        print(f"Fold {i}:")
        print(f"  訓練グループ: {train_groups}")
        print(f"  テストグループ: {test_groups}")
        print(f"  訓練サンプル数: {len(train_idx)}, テストサンプル数: {len(test_idx)}")
    

### いつGroup K-Foldを使うべきか

  * **必須** ：同じエンティティから複数のサンプルがある場合
  * **推奨** ：医療データ、時系列の複数観測、画像の複数ショット
  * **注意** ：グループ数が少ない（<10）場合は評価が不安定

* * *

## 2.8 Nested Cross-Validation

### ハイパーパラメータ調整の問題

ハイパーパラメータをチューニングする際、以下のような**間違ったアプローチ** をすると、テストセットにフィットしてしまいます：
    
    
    # ❌ 間違ったアプローチ
    from sklearn.model_selection import GridSearchCV, train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # テストセットを使ってハイパーパラメータ調整
    best_params = try_different_hyperparameters(X_train, y_train, X_test, y_test)
    
    # 同じテストセットで最終評価 → 過大評価！
    final_score = evaluate(best_model, X_test, y_test)
    

### Nested CVの原理

**Nested Cross-Validation（入れ子交差検証）** は、交差検証を2段階に分けます：

  * **外側のCV** ：汎化性能の評価
  * **内側のCV** ：ハイパーパラメータの選択

    
    
    ```mermaid
    graph TD
        A[全データ] --> B[外側CV: 5-Fold]
    
        B --> C[Fold 1]
        B --> D[Fold 2]
        B --> E[Fold 3]
    
        C --> F[訓練データ80%]
        C --> G[テストデータ20%]
    
        F --> H[内側CV: 5-Foldハイパーパラメータ調整]
        H --> I[最良パラメータ]
        I --> J[全訓練データで再学習]
        J --> K[外側テストで評価]
    
        style A fill:#e1f5ff
        style H fill:#fff9c4
        style K fill:#c8e6c9
    ```

### 実装例
    
    
    import numpy as np
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    
    # データ生成
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    # モデルとハイパーパラメータ空間
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # 内側CV：ハイパーパラメータ調整
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # 外側CV：汎化性能評価
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='accuracy')
    
    print("=== Nested Cross-Validation ===")
    print(f"外側CVの各Foldスコア: {nested_scores}")
    print(f"平均精度: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    
    # 比較：通常のCV（ハイパーパラメータ固定）
    simple_model = SVC(C=1.0, gamma=0.01, kernel='rbf')
    simple_scores = cross_val_score(simple_model, X, y, cv=outer_cv, scoring='accuracy')
    
    print("\n=== 通常のCV（C=1.0, gamma=0.01 固定） ===")
    print(f"平均精度: {simple_scores.mean():.4f} ± {simple_scores.std():.4f}")
    

### Nested CVのベストプラクティス

  * **外側CV** ：K=5または10（汎化性能の信頼性の高い推定）
  * **内側CV** ：K=3または5（計算コストとのバランス）
  * **総訓練回数** ：外側K × 内側K × ハイパーパラメータ数
  * **並列化** ：`n_jobs=-1`で高速化

### 計算コストの例

設定 | 訓練回数 | 推奨ケース  
---|---|---  
外側5 × 内側3 | 15回 × パラメータ数 | 標準的な設定  
外側5 × 内側5 | 25回 × パラメータ数 | 高精度が必要な場合  
外側10 × 内側5 | 50回 × パラメータ数 | 小〜中規模データ  
  
> **注意** ：Nested CVは計算コストが高いため、大規模データやディープラーニングでは実用的でない場合があります。その場合は、Hold-out法で検証セットを明示的に分離し、テストセットは最後まで触らない戦略を取ります。

* * *

## 2.9 交差検証のベストプラクティス

### タスク別の交差検証選択フローチャート
    
    
    ```mermaid
    graph TD
        A[データ分析開始] --> B{時系列データ?}
        B -->|Yes| C[TimeSeriesSplit]
        B -->|No| D{グループ構造?}
    
        D -->|Yes| E[GroupKFold]
        D -->|No| F{分類タスク?}
    
        F -->|Yes| G{クラス不均衡?}
        G -->|Yes| H[StratifiedKFold]
        G -->|No| I{データサイズ?}
    
        F -->|No| J[回帰タスク]
        J --> I
    
        I -->|N < 100| K[LeaveOneOut]
        I -->|N >= 100| L[KFold K=5 or 10]
    
        H --> M{ハイパーパラメータ調整?}
        L --> M
        K --> M
        C --> M
        E --> M
    
        M -->|Yes| N[Nested CV]
        M -->|No| O[通常のCV]
    
        style A fill:#e1f5ff
        style C fill:#b3e5fc
        style E fill:#81d4fa
        style H fill:#4fc3f7
        style N fill:#c8e6c9
    ```

### チェックリスト：データリーケージ防止

  * ✅ **前処理** ：分割後に訓練データのみで学習（StandardScaler等）
  * ✅ **特徴量選択** ：各Fold内で独立して実行
  * ✅ **欠損値補完** ：訓練データの統計量のみ使用
  * ✅ **Pipeline使用** ：前処理とモデルを統合
  * ✅ **時系列** ：TimeSeriesSplitを使用
  * ✅ **グループ** ：GroupKFoldで同一グループを分離
  * ✅ **テストセット** ：最終評価まで一切触らない

### パフォーマンス最適化
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
    
    # n_jobs=-1 で並列化（全CPUコアを使用）
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # cross_val_score でも並列化可能
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 並列実行（推奨）
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"並列実行: 平均精度 = {scores.mean():.4f}")
    

### 実務での交差検証戦略

フェーズ | 推奨手法 | 理由  
---|---|---  
**初期探索** | 3-Fold CV | 高速にモデルの方向性を確認  
**モデル開発** | 5-Fold Stratified CV | バランスと計算コストの最適化  
**ハイパーパラメータ調整** | Nested CV (5×3) | 過学習を防ぎながらチューニング  
**最終評価** | Hold-out Test Set | 未使用データで真の汎化性能測定  
  
### まとめ：推奨される実装パターン
    
    
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.datasets import make_classification
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # ベストプラクティス：Pipeline + Stratified K-Fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 前処理
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Stratified K-Fold（分類タスクの標準）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 交差検証実行
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1  # 並列化
    )
    
    print("=== 推奨実装パターン ===")
    print(f"平均精度: {scores.mean():.4f}")
    print(f"標準偏差: {scores.std():.4f}")
    print(f"95%信頼区間: [{scores.mean() - 1.96*scores.std():.4f}, "
          f"{scores.mean() + 1.96*scores.std():.4f}]")
    

* * *

## 演習問題

**演習1：Hold-out法とK-Foldの比較**

以下のコードを完成させ、Hold-out法（テスト20%）と5-Fold CVの性能を比較してください。どちらがより安定した評価を与えますか？
    
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.tree import DecisionTreeClassifier
    
    # データ読み込み
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # TODO: Hold-out法で評価
    # ヒント: train_test_split を使用
    
    # TODO: 5-Fold CVで評価
    # ヒント: cross_val_score を使用
    
    # TODO: 結果を比較
    

**演習2：Stratified K-Foldの重要性**

不均衡データ（正例5%）を作成し、通常のK-FoldとStratified K-Foldでクラス分布がどう異なるか確認してください。
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import KFold, StratifiedKFold
    import numpy as np
    
    # 不均衡データ生成
    X, y = make_classification(
        n_samples=1000,
        weights=[0.95, 0.05],  # 5% positive class
        random_state=42
    )
    
    # TODO: 各Foldのクラス分布を確認
    # ヒント: 各Foldのy_testでクラス1の割合を計算
    

**演習3：時系列データの交差検証**

時系列データで通常のK-FoldとTimeSeriesSplitを比較し、情報リーケージの影響を確認してください。
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, TimeSeriesSplit
    from sklearn.linear_model import Ridge
    
    # 時系列データ生成
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    y = 0.5 * t + 10 * np.sin(t / 10) + np.random.randn(n) * 5
    
    # 特徴量（過去の値）
    X = np.column_stack([np.roll(y, i) for i in range(1, 6)])
    X, y = X[5:], y[5:]
    
    # TODO: K-FoldとTimeSeriesSplitで評価を比較
    # どちらの方がRMSEが良いですか？その理由は？
    

**演習4：データリーケージの検出**

以下のコードにはデータリーケージがあります。どこが問題で、どう修正すべきですか？
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    # 全データで正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 交差検証
    model = LogisticRegression(max_iter=200)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv)
    
    print(f"精度: {scores.mean():.4f}")
    
    # TODO: この実装の問題点を指摘し、Pipelineを使って修正してください
    

**演習5：Nested CVの実装**

SVMのハイパーパラメータ（CとGamma）をNested CVで調整し、真の汎化性能を推定してください。
    
    
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    
    # データ読み込み
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # TODO: Nested CVを実装
    # 外側CV: 5-Fold
    # 内側CV: 3-Fold
    # パラメータ: C=[0.1, 1, 10], gamma=[0.001, 0.01, 0.1]
    
    # ヒント: GridSearchCVをcross_val_scoreに渡す
    

* * *

## まとめ

この章では、モデル評価の要となる**交差検証とデータ分割** について学びました。

### 重要ポイント

  * **Hold-out法** は簡単だが不安定。K-Fold CVで安定した評価を得る
  * **K-Fold CV** ：データを有効活用し、複数回の評価で信頼性を向上
  * **Stratified K-Fold** ：分類問題では必須。クラス比率を保持
  * **TimeSeriesSplit** ：時系列データでは時間順序を守る
  * **Group K-Fold** ：同じエンティティのデータを分離
  * **データリーケージ** ：Pipelineで防止。前処理は訓練データのみで学習
  * **Nested CV** ：ハイパーパラメータ調整と性能評価を分離

### 次のステップ

次章では、交差検証で得たスコアをもとに、さまざまな**評価指標** （精度、適合率、再現率、F1スコア、AUC-ROCなど）を学び、タスクに応じた適切な指標選択を習得します。
