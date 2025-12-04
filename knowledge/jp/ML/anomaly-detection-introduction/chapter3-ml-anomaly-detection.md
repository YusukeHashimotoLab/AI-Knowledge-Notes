---
title: 第3章：機械学習ベース異常検知
chapter_title: 第3章：機械学習ベース異常検知
subtitle: Isolation Forest、LOF、One-Class SVMによる異常検出
reading_time: 70-80分
difficulty: 中級
code_examples: 10
exercises: 5
---

# 第3章：機械学習ベース異常検知

## 学習目標

  * Isolation Forestのアルゴリズム原理を理解する
  * LOF（Local Outlier Factor）で局所的な異常を検出できる
  * One-Class SVMによる正常データの境界学習を習得する
  * DBSCANやその他の手法を異常検知に適用できる
  * アンサンブル異常検知の実装方法を学ぶ

**読了時間** : 70-80分

* * *

## 3.1 Isolation Forest（孤立森林）

Isolation Forestは、異常データが正常データより「分離しやすい」という性質を利用した異常検知アルゴリズムです。2008年にLiu et al.によって提案され、高次元データにも効果的に適用できます。

### 3.1.1 アルゴリズム原理

**基本アイデア:**

  * 異常データは数が少なく、正常データとは異なる特徴値を持つ
  * ランダムに選んだ特徴量で分割を繰り返すと、異常データはより早く孤立する
  * 孤立までの分割回数（パス長）が短いほど異常度が高い

**アルゴリズムステップ:**
    
    
    1. ランダムに特徴量を選択
    2. その特徴量の最小値と最大値の間でランダムに分割点を選ぶ
    3. データを2つのグループに分ける
    4. 各グループに対して再帰的に1-3を繰り返す
    5. 各データポイントが孤立するまでのパス長を記録
    6. 複数の木（森）を構築し、平均パス長から異常スコアを計算
    

### 3.1.2 パス長と異常スコア

**パス長（Path Length）:**

データポイント $x$ が孤立するまでの分割回数を $h(x)$ とすると、正常データは深い位置（大きな $h(x)$）、異常データは浅い位置（小さな $h(x)$）に孤立します。

**異常スコアの計算:**

$$ s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}} $$ 

ここで:

  * $E[h(x)]$: 複数の木における平均パス長
  * $c(n)$: サンプルサイズ $n$ における平均パス長の正規化定数
  * $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ （$H(i)$ は調和数）

**スコアの解釈:**

  * $s \approx 1$: 異常（明確な異常値）
  * $s \approx 0.5$: 正常（平均的なパス長）
  * $s < 0.5$: 正常（平均より深い位置）

### 3.1.3 ハイパーパラメータ調整

**主要パラメータ:**

パラメータ | 説明 | 推奨値  
---|---|---  
`n_estimators` | 木の数 | 100-200（デフォルト: 100）  
`max_samples` | 各木でサンプリングするデータ数 | 256（デフォルト: auto）  
`contamination` | 異常データの割合 | 0.1（データに依存）  
`max_features` | 各分割で考慮する特徴量数 | 1.0（全特徴量）  
  
**パラメータ選択のガイドライン:**

  * `n_estimators`: 多いほど安定するが計算コスト増加（100-200で十分）
  * `max_samples`: 256が推奨（論文のデフォルト）、大規模データでは小さくしてスピードアップ
  * `contamination`: 事前に異常率がわかっていればそれを設定、不明なら0.1

### 3.1.4 scikit-learn実装

**基本実装:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    
    # サンプルデータ生成（正常データ + 異常データ）
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
    X_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))  # 異常データ
    X = np.vstack([X_normal, X_anomaly])
    
    # Isolation Forestモデル
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.1,  # 10%が異常と想定
        random_state=42
    )
    
    # 学習と予測
    y_pred = iso_forest.fit_predict(X)  # -1: 異常、1: 正常
    scores = iso_forest.score_samples(X)  # 異常スコア（低いほど異常）
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('Isolation Forest: 異常検知結果')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', edgecolors='k')
    plt.title('Isolation Forest: 異常スコア')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly Score')
    
    plt.tight_layout()
    plt.show()
    
    print(f"検出された異常データ数: {np.sum(y_pred == -1)}")
    print(f"異常スコア範囲: [{scores.min():.3f}, {scores.max():.3f}]")
    

**出力例:**
    
    
    検出された異常データ数: 32
    異常スコア範囲: [-0.234, 0.178]
    

**実データへの適用例（クレジットカード不正検知）:**
    
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # データ読み込み（仮想的な例）
    # 実際のデータは Kaggle Credit Card Fraud Detection などを使用
    # URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
    
    # サンプルデータ生成（実データの代替）
    np.random.seed(42)
    n_normal = 1000
    n_fraud = 50
    
    # 正常取引（金額小、回数多、地理的に集中）
    normal_features = np.random.randn(n_normal, 5) * [10, 5, 2, 1, 0.5]
    normal_labels = np.zeros(n_normal)
    
    # 不正取引（金額大、回数少、地理的に分散）
    fraud_features = np.random.randn(n_fraud, 5) * [50, 1, 10, 5, 3] + [100, 0, 50, 20, 10]
    fraud_labels = np.ones(n_fraud)
    
    X = np.vstack([normal_features, fraud_features])
    y = np.hstack([normal_labels, fraud_labels])
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Isolation Forest（正常データのみで学習）
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # 5%が不正と想定
        random_state=42
    )
    
    # 訓練データで学習
    iso_forest.fit(X_train)
    
    # テストデータで予測
    y_pred = iso_forest.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1を1（不正）に変換
    
    # 評価
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    

**出力例:**
    
    
    Confusion Matrix:
    [[285  15]
     [  3  12]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
          Normal       0.99      0.95      0.97       300
           Fraud       0.44      0.80      0.57        15
    
        accuracy                           0.94       315
       macro avg       0.72      0.88      0.77       315
    weighted avg       0.96      0.94      0.95       315
    

* * *

## 3.2 LOF（Local Outlier Factor）

LOFは、各データポイントの局所的な密度に基づいて異常を検出する手法です。2000年にBreunig et al.によって提案されました。

### 3.2.1 密度ベース異常検知

**基本原理:**

  * 正常データは高密度領域に存在する
  * 異常データは低密度領域に存在する
  * 各点の密度を近傍点の密度と比較して異常度を計算

**なぜ「局所的」か:**

  * グローバルな密度では検出できない異常も検出可能
  * 密度が異なる複数のクラスタが存在する場合に有効
  * 各点の周辺環境を考慮した相対的な異常度を算出

### 3.2.2 局所到達可能密度（Local Reachability Density）

**k距離（k-distance）:**

点 $p$ から k番目に近い点までの距離を $d_k(p)$ とします。

**到達可能距離（Reachability Distance）:**

$$ \text{reach-dist}_k(p, o) = \max\\{d_k(o), d(p, o)\\} $$ 

  * $d(p, o)$: 点 $p$ と $o$ 間の実際の距離
  * 近傍点 $o$ が密な場合、到達可能距離は $d_k(o)$ で下限が設定される

**局所到達可能密度（LRD）:**

$$ \text{LRD}_k(p) = \frac{1}{\frac{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}{|N_k(p)|}} $$ 

  * $N_k(p)$: 点 $p$ の k近傍点の集合
  * 到達可能距離の平均の逆数 = 密度

### 3.2.3 LOFスコアの計算

**LOF（Local Outlier Factor）:**

$$ \text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{LRD}_k(o)}{\text{LRD}_k(p)}}{|N_k(p)|} $$ 

**スコアの解釈:**

  * $\text{LOF} \approx 1$: 正常（近傍と同程度の密度）
  * $\text{LOF} \gg 1$: 異常（近傍より密度が低い）
  * $\text{LOF} < 1$: 正常（近傍より密度が高い）

一般的に $\text{LOF} > 1.5$ を異常とみなします。

### 3.2.4 完全な実装例

**基本実装:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.datasets import make_moons
    
    # サンプルデータ生成（月型データ + 異常点）
    np.random.seed(42)
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X_outliers = np.random.uniform(low=-1, high=2, size=(20, 2))
    X = np.vstack([X, X_outliers])
    
    # LOFモデル
    lof = LocalOutlierFactor(
        n_neighbors=20,  # 近傍点数
        contamination=0.1,  # 異常率
        novelty=False  # 新規データ予測にはTrue
    )
    
    # 予測
    y_pred = lof.fit_predict(X)  # -1: 異常、1: 正常
    scores = lof.negative_outlier_factor_  # 負の異常度スコア（低いほど異常）
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('LOF: 異常検知結果')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', edgecolors='k')
    plt.title('LOF: 異常スコア')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Negative Outlier Factor')
    
    plt.tight_layout()
    plt.show()
    
    print(f"検出された異常データ数: {np.sum(y_pred == -1)}")
    print(f"異常スコア範囲: [{scores.min():.3f}, {scores.max():.3f}]")
    

**n_neighborsパラメータの影響:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor
    
    # データ生成
    np.random.seed(42)
    X_normal = np.random.randn(200, 2) * 0.5
    X_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # 異なるn_neighborsで比較
    n_neighbors_list = [5, 20, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n_neighbors in enumerate(n_neighbors_list):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        y_pred = lof.fit_predict(X)
    
        axes[idx].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
        axes[idx].set_title(f'LOF (n_neighbors={n_neighbors})')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
        anomaly_count = np.sum(y_pred == -1)
        axes[idx].text(0.05, 0.95, f'Anomalies: {anomaly_count}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    

**新規データの異常検知（novelty=True）:**
    
    
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.model_selection import train_test_split
    
    # データ準備
    np.random.seed(42)
    X_train = np.random.randn(500, 2) * 0.5  # 正常データのみ
    X_test_normal = np.random.randn(100, 2) * 0.5
    X_test_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X_test = np.vstack([X_test_normal, X_test_outliers])
    
    # LOF（novelty=True: 新規データ予測モード）
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_train)  # 正常データのみで学習
    
    # 新規データの予測
    y_pred = lof.predict(X_test)
    scores = lof.score_samples(X_test)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.3, label='Training Data', color='blue')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm',
                edgecolors='k', s=100, label='Test Data')
    plt.title('LOF: 新規データの異常検知（novelty=True）')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    plt.show()
    
    print(f"検出された異常データ数: {np.sum(y_pred == -1)}/{len(y_pred)}")
    

* * *

## 3.3 One-Class SVM

One-Class SVMは、正常データの境界を学習し、その境界の外側にあるデータを異常として検出する手法です。

### 3.3.1 最大マージン超平面

**基本原理:**

  * 正常データを原点から最もよく分離する超平面を見つける
  * 超平面とデータ点の間のマージンを最大化
  * カーネルトリックで非線形な境界を学習

**数式定義:**

決定関数:

$$ f(x) = \text{sign}(w \cdot \phi(x) - \rho) $$ 

  * $w$: 法線ベクトル
  * $\phi(x)$: カーネル変換後の特徴ベクトル
  * $\rho$: バイアス項

最適化問題:

$$ \min_{w, \rho, \xi} \frac{1}{2} \|w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho $$ 

制約:

$$ w \cdot \phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0 $$ 

### 3.3.2 カーネルトリック

**線形カーネル:**

$$ K(x, x') = x \cdot x' $$ 

  * 高速、解釈しやすい
  * 線形分離可能なデータに適用

**RBF（ガウス）カーネル:**

$$ K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right) $$ 

  * 非線形境界を学習可能
  * 最もよく使われるカーネル
  * $\gamma$: カーネル幅（大きいほど複雑な境界）

**多項式カーネル:**

$$ K(x, x') = (\gamma x \cdot x' + r)^d $$ 

  * 次数 $d$ の多項式境界
  * RBFより制約が強い

### 3.3.3 nuパラメータ

**nuの意味:**

$\nu \in (0, 1]$ は以下の2つの量の上限と下限を制御します:

  * 訓練データにおける異常値の割合の**上限**
  * サポートベクターの割合の**下限**

**推奨値:**

  * $\nu = 0.1$: 10%が異常と想定
  * $\nu = 0.05$: 5%が異常と想定
  * $\nu = 0.01$: 1%が異常と想定

**注意点:**

  * $\nu$ を小さくしすぎると、ほとんど異常が検出されない
  * $\nu$ を大きくしすぎると、正常データも異常と判定される
  * ドメイン知識や事前の異常率から設定

### 3.3.4 scikit-learn実装

**基本実装:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    
    # データ生成
    np.random.seed(42)
    X_train = np.random.randn(200, 2) * 0.5  # 正常データ
    X_test_normal = np.random.randn(50, 2) * 0.5
    X_test_outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
    X_test = np.vstack([X_test_normal, X_test_outliers])
    
    # One-Class SVM
    oc_svm = OneClassSVM(
        kernel='rbf',  # RBFカーネル
        gamma='auto',  # gamma = 1 / n_features
        nu=0.1  # 10%が異常と想定
    )
    
    # 学習
    oc_svm.fit(X_train)
    
    # 予測
    y_pred_train = oc_svm.predict(X_train)
    y_pred_test = oc_svm.predict(X_test)
    scores_test = oc_svm.decision_function(X_test)
    
    # 決定境界の可視化
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolors='k', label='Training')
    plt.title('One-Class SVM: 決定境界')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm',
                edgecolors='k', s=100)
    plt.title('One-Class SVM: テストデータ予測')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"訓練データの異常数: {np.sum(y_pred_train == -1)}/{len(y_pred_train)}")
    print(f"テストデータの異常数: {np.sum(y_pred_test == -1)}/{len(y_pred_test)}")
    

**gammaパラメータの影響:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    
    # データ生成
    np.random.seed(42)
    X_train = np.random.randn(200, 2) * 0.5
    
    # 異なるgammaで比較
    gamma_list = [0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, gamma in enumerate(gamma_list):
        oc_svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=0.1)
        oc_svm.fit(X_train)
    
        # 決定境界
        xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
        Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        axes[idx].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        axes[idx].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        axes[idx].scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolors='k')
        axes[idx].set_title(f'One-Class SVM (gamma={gamma})')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.4 その他の機械学習手法

### 3.4.1 DBSCAN（密度ベースクラスタリング）

**原理:**

  * 密度の高い領域をクラスタとして検出
  * どのクラスタにも属さない点をノイズ（異常）とみなす
  * クラスタ数を事前に指定する必要がない

**主要パラメータ:**

  * `eps`: 近傍の半径（距離の閾値）
  * `min_samples`: コア点になるための最小近傍点数

**実装例:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    
    # データ生成
    np.random.seed(42)
    X_cluster1 = np.random.randn(100, 2) * 0.3 + [0, 0]
    X_cluster2 = np.random.randn(100, 2) * 0.3 + [3, 3]
    X_outliers = np.random.uniform(low=-2, high=5, size=(20, 2))
    X = np.vstack([X_cluster1, X_cluster2, X_outliers])
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    # ラベル-1がノイズ（異常）
    outliers = labels == -1
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X[~outliers, 0], X[~outliers, 1], c=labels[~outliers],
                cmap='viridis', edgecolors='k', label='Clusters')
    plt.scatter(X[outliers, 0], X[outliers, 1], c='red', marker='x',
                s=100, label='Outliers (Anomalies)')
    plt.title('DBSCAN: 密度ベース異常検知')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
    print(f"検出されたクラスタ数: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"異常データ数: {np.sum(outliers)}")
    

### 3.4.2 Elliptic Envelope（楕円エンベロープ）

**原理:**

  * 正規分布を仮定し、データの中心と共分散を推定
  * マハラノビス距離で異常を検出
  * ロバスト推定（Minimum Covariance Determinant）で外れ値の影響を抑える

**実装例:**
    
    
    from sklearn.covariance import EllipticEnvelope
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    X_normal = np.random.randn(200, 2)
    X_outliers = np.random.uniform(low=-5, high=5, size=(10, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # Elliptic Envelope
    elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
    y_pred = elliptic.fit_predict(X)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    plt.title('Elliptic Envelope: 異常検知')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
    plt.show()
    
    print(f"検出された異常データ数: {np.sum(y_pred == -1)}")
    

### 3.4.3 Robust Covariance（ロバスト共分散推定）

**Minimum Covariance Determinant（MCD）:**

  * 共分散行列の行列式を最小化するサブセットを探索
  * 外れ値に対してロバストな推定
  * マハラノビス距離の計算に使用

    
    
    from sklearn.covariance import MinCovDet
    import numpy as np
    
    # データ生成
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:5] = X[:5] + 5  # 外れ値追加
    
    # MCD推定
    mcd = MinCovDet(random_state=42)
    mcd.fit(X)
    
    # マハラノビス距離の計算
    distances = mcd.mahalanobis(X)
    
    # 異常判定（カイ二乗分布の95パーセンタイル）
    from scipy import stats
    threshold = stats.chi2.ppf(0.95, df=2)
    outliers = distances > threshold
    
    print(f"異常データ数: {np.sum(outliers)}")
    print(f"距離の閾値: {threshold:.2f}")
    

### 3.4.4 PyODライブラリ

**PyOD（Python Outlier Detection）** は、異常検知専門のライブラリで40以上のアルゴリズムを提供します。

**インストール:**
    
    
    pip install pyod
    

**使用例:**
    
    
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    from pyod.utils.utility import standardizer
    import numpy as np
    
    # データ生成
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=2,
        contamination=0.1, random_state=42
    )
    
    # データ標準化
    X_train = standardizer(X_train)
    X_test = standardizer(X_test)
    
    # 複数モデルの比較
    models = {
        'KNN': KNN(contamination=0.1),
        'IForest': IForest(contamination=0.1, random_state=42),
        'LOF': LOF(contamination=0.1)
    }
    
    for name, model in models.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)
    
        # 評価（仮想的な正解ラベルで）
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, scores)
    
        print(f"{name}:")
        print(f"  AUC-ROC: {auc:.3f}")
        print(f"  検出異常数: {np.sum(y_pred == 1)}")
        print()
    

**出力例:**
    
    
    KNN:
      AUC-ROC: 0.892
      検出異常数: 10
    
    IForest:
      AUC-ROC: 0.915
      検出異常数: 10
    
    LOF:
      AUC-ROC: 0.903
      検出異常数: 10
    

* * *

## 3.5 アンサンブル異常検知

複数の異常検知アルゴリズムを組み合わせることで、より高精度で安定した異常検知が可能になります。

### 3.5.1 Feature Bagging

**原理:**

  * 特徴量のサブセットをランダムに選択
  * 各サブセットで異常検知モデルを訓練
  * 複数モデルの予測を集約

**実装例:**
    
    
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    from sklearn.metrics import roc_auc_score
    
    # データ生成
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=10,
        contamination=0.1, random_state=42
    )
    
    # Feature Bagging（ベースモデル: LOF）
    fb = FeatureBagging(
        base_estimator=LOF(),
        n_estimators=10,  # モデル数
        contamination=0.1,
        random_state=42
    )
    
    # 学習と予測
    fb.fit(X_train)
    y_pred = fb.predict(X_test)
    scores = fb.decision_function(X_test)
    
    # 評価
    auc = roc_auc_score(y_test, scores)
    print(f"Feature Bagging AUC-ROC: {auc:.3f}")
    print(f"検出異常数: {np.sum(y_pred == 1)}")
    

### 3.5.2 モデル平均化（Model Averaging）

**原理:**

  * 複数の異なるアルゴリズムを訓練
  * 各モデルの異常スコアを平均化
  * 単一モデルより頑健な予測

**実装例:**
    
    
    from pyod.models.combination import average, maximization
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    import numpy as np
    
    # データ生成
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=5,
        contamination=0.1, random_state=42
    )
    
    # 複数モデルの訓練
    models = [
        KNN(contamination=0.1),
        IForest(contamination=0.1, random_state=42),
        LOF(contamination=0.1)
    ]
    
    # 各モデルのスコアを計算
    scores_list = []
    for model in models:
        model.fit(X_train)
        scores = model.decision_function(X_test)
        scores_list.append(scores)
    
    scores_array = np.array(scores_list)
    
    # スコア集約（平均）
    scores_avg = average(scores_array)
    
    # スコア集約（最大値）
    scores_max = maximization(scores_array)
    
    # 評価
    from sklearn.metrics import roc_auc_score
    auc_avg = roc_auc_score(y_test, scores_avg)
    auc_max = roc_auc_score(y_test, scores_max)
    
    print(f"Average Combination AUC-ROC: {auc_avg:.3f}")
    print(f"Maximum Combination AUC-ROC: {auc_max:.3f}")
    

### 3.5.3 Isolation-Based Ensemble

**LSCP（Locally Selective Combination in Parallel）:**

  * 各テストサンプルに対して局所的に最適なモデルを選択
  * 近傍での性能に基づいてモデルを重み付け
  * グローバルな平均化より高精度

    
    
    from pyod.models.lscp import LSCP
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.utils.data import generate_data
    
    # データ生成
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, n_features=5,
        contamination=0.1, random_state=42
    )
    
    # ベースモデルのリスト
    detector_list = [
        KNN(),
        IForest(random_state=42),
        LOF()
    ]
    
    # LSCP
    lscp = LSCP(detector_list, contamination=0.1, random_state=42)
    lscp.fit(X_train)
    
    # 予測
    y_pred = lscp.predict(X_test)
    scores = lscp.decision_function(X_test)
    
    # 評価
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, scores)
    print(f"LSCP AUC-ROC: {auc:.3f}")
    print(f"検出異常数: {np.sum(y_pred == 1)}")
    

### 3.5.4 完全なパイプライン例

**データ前処理 → 複数モデル訓練 → アンサンブル → 評価:**
    
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.combination import average
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    
    # データ生成（実データの代替）
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    contamination = 0.05
    
    # 正常データ
    X_normal = np.random.randn(int(n_samples * (1 - contamination)), n_features)
    # 異常データ
    X_anomaly = np.random.uniform(low=-5, high=5, size=(int(n_samples * contamination), n_features))
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # データ標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 複数モデルの訓練
    models = {
        'KNN': KNN(contamination=contamination),
        'IForest': IForest(contamination=contamination, random_state=42),
        'LOF': LOF(contamination=contamination),
        'OCSVM': OCSVM(contamination=contamination)
    }
    
    scores_dict = {}
    predictions_dict = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled)
        scores = model.decision_function(X_test_scaled)
        y_pred = model.predict(X_test_scaled)
    
        scores_dict[name] = scores
        predictions_dict[name] = y_pred
    
        auc = roc_auc_score(y_test, scores)
        print(f"{name} AUC-ROC: {auc:.3f}")
    
    # アンサンブル（平均）
    scores_list = [scores_dict[name] for name in models.keys()]
    scores_ensemble = average(np.array(scores_list))
    auc_ensemble = roc_auc_score(y_test, scores_ensemble)
    print(f"\nEnsemble AUC-ROC: {auc_ensemble:.3f}")
    
    # ROC曲線の可視化
    plt.figure(figsize=(10, 6))
    
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    
    # アンサンブルのROC
    fpr_ens, tpr_ens, _ = roc_curve(y_test, scores_ensemble)
    plt.plot(fpr_ens, tpr_ens, 'k--', linewidth=2,
             label=f'Ensemble (AUC={auc_ensemble:.3f})')
    
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: 複数モデルとアンサンブル')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

* * *

## 3.6 まとめ

### 本章で学んだこと

  1. **Isolation Forest:**

     * ランダム分離による異常検知
     * パス長から異常スコアを計算
     * ハイパーパラメータ（n_estimators, max_samples, contamination）
     * 高次元データに効果的
  2. **LOF（Local Outlier Factor）:**

     * 局所密度に基づく異常検知
     * 到達可能距離と局所到達可能密度
     * LOFスコアの計算と解釈
     * 密度が異なるクラスタに対応
  3. **One-Class SVM:**

     * 最大マージン超平面による境界学習
     * カーネルトリック（RBF, 線形, 多項式）
     * nuパラメータによる異常率制御
     * 非線形境界の学習
  4. **その他の手法:**

     * DBSCAN（密度ベースクラスタリング）
     * Elliptic Envelope（楕円エンベロープ）
     * Robust Covariance（ロバスト共分散推定）
     * PyODライブラリ（40以上のアルゴリズム）
  5. **アンサンブル異常検知:**

     * Feature Bagging（特徴量サブセット）
     * Model Averaging（スコア平均化）
     * Isolation-Based Ensemble（LSCP）
     * 複数モデルの組み合わせによる精度向上

### 手法の使い分け

手法 | 適用場面 | 長所 | 短所  
---|---|---|---  
Isolation Forest | 高次元データ、大規模データ | 高速、スケーラブル | パラメータ調整が必要  
LOF | 密度が異なるクラスタ | 局所的な異常を検出 | 計算コストが高い  
One-Class SVM | 非線形境界、理論的保証 | 頑健、理論的基盤 | 大規模データで遅い  
DBSCAN | クラスタリング + 異常検知 | クラスタ数不要 | パラメータに敏感  
アンサンブル | 高精度が必要な場面 | 頑健、高精度 | 計算コスト増加  
  
### 次のステップ

第4章では、深層学習による異常検知を学びます：

  * Autoencoder（再構成誤差ベース）
  * VAE（Variational Autoencoder）
  * GAN（Generative Adversarial Network）
  * LSTM Autoencoder（時系列異常検知）
  * Transformer（Attention機構）

* * *

## 演習問題

**問1:** Isolation Forestで、異常スコア $s(x, n) = 0.8$ のデータポイントは異常と判定すべきか？理由とともに答えよ。

**解答:**

はい、異常と判定すべきです。

**理由:**

  * 異常スコア $s \approx 1$ は明確な異常を示す
  * $s \approx 0.5$ は正常（平均的なパス長）
  * $s = 0.8$ は1に近く、通常より早く孤立していることを意味する
  * 一般的に $s > 0.6$ を異常とみなす閾値として使用

**問2:** LOFスコアが $\text{LOF}_k(p) = 2.5$ の場合、この点は異常か？また、このスコアが意味することを説明せよ。

**解答:**

はい、異常です。

**意味:**

  * $\text{LOF} \approx 1$ は近傍と同程度の密度（正常）
  * $\text{LOF} > 1$ は近傍より密度が低い（異常の可能性）
  * $\text{LOF} = 2.5$ は、この点の密度が近傍の平均密度の約1/2.5であることを示す
  * 一般的に $\text{LOF} > 1.5$ を異常とみなすため、2.5は明確な異常

**問3:** One-Class SVMのnuパラメータを0.05に設定した場合、訓練データの何%が異常と判定されるか？また、nuを大きくした場合の影響を説明せよ。

**解答:**

訓練データの最大5%が異常と判定されます。

**nuを大きくした場合の影響:**

  * $\nu = 0.1$: 最大10%が異常と判定される
  * $\nu = 0.2$: 最大20%が異常と判定される
  * nuを大きくすると、より多くのデータが異常と判定される
  * 正常データも異常と誤判定されるリスクが増加（偽陽性増加）
  * 異常検知の感度が高くなるが、精度は低下する可能性

**問4:** DBSCANで異常検知を行う際、epsとmin_samplesパラメータをどのように選択すべきか？具体的な選択方法を3つ述べよ。

**解答:**

  1. **K距離グラフ法:**

     * 各点のk番目の最近傍距離を計算（kはmin_samplesの候補）
     * 距離を降順にソートしてプロット
     * 急激に増加する点（エルボー点）をepsとして選択
  2. **ドメイン知識に基づく選択:**

     * データの性質から適切な近傍サイズを推定
     * 例: 2次元データなら min_samples=4、高次元なら min_samples=2×次元数
     * epsはデータのスケールに応じて調整
  3. **グリッドサーチ:**

     * 複数の(eps, min_samples)の組み合わせを試す
     * シルエットスコアやクラスタ数で評価
     * 最適な組み合わせを選択

**問5:** アンサンブル異常検知で、Feature Baggingとモデル平均化の違いを説明し、それぞれがどのような状況で有効か論じよ（300字以内）。

**解答例:**

Feature Baggingは特徴量のサブセットで複数モデルを訓練し、高次元データにおける特徴量間の相関や冗長性に対処します。特徴量が多く相関が強い場合に有効です。一方、モデル平均化は異なるアルゴリズム（KNN、Isolation Forest、LOFなど）の予測を集約し、各手法の長所を活かします。データの性質が不明確で、どの手法が最適か事前にわからない場合に有効です。Feature Baggingは同一アルゴリズムの多様性を高め、モデル平均化はアルゴリズムの多様性を活用する点が主な違いです。実務では両方を組み合わせることで、より頑健な異常検知システムを構築できます。

* * *

## 参考文献

  1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." _IEEE International Conference on Data Mining (ICDM)_.
  2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). "LOF: Identifying Density-Based Local Outliers." _ACM SIGMOD International Conference on Management of Data_.
  3. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). "Estimating the Support of a High-Dimensional Distribution." _Neural Computation_ , 13(7), 1443-1471.
  4. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters." _KDD_ , 96(34), 226-231.
  5. Zhao, Y., Nasrullah, Z., & Li, Z. (2019). "PyOD: A Python Toolbox for Scalable Outlier Detection." _Journal of Machine Learning Research_ , 20(96), 1-7.

* * *

**次章** : [第4章：深層学習による異常検知](<chapter4-deep-learning-anomaly.html>)

**ライセンス** : このコンテンツはCC BY 4.0ライセンスの下で提供されています。
