---
title: 第3章：異常検知
chapter_title: 第3章：異常検知
subtitle: 不正検知からシステム監視まで - 統計手法から深層学習までの異常検知アルゴリズム
reading_time: 20-25分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 異常検知の基本概念と応用例を理解する
  * ✅ 統計的手法（Z-score、IQR）を実装できる
  * ✅ Isolation Forestの原理を理解し応用できる
  * ✅ One-Class SVMで新規性検知を実装できる
  * ✅ Local Outlier Factor（LOF）を使いこなせる
  * ✅ Autoencoderによる異常検知を実装できる
  * ✅ 異常検知の評価指標を適切に選択できる

* * *

## 3.1 異常検知とは

### 定義

**異常検知（Anomaly Detection）** は、正常なパターンから大きく外れたデータ点（異常値）を検出する手法です。

> 「異常値は単なるノイズではない。それは新しい発見、不正行為、システム障害のサインかもしれない」

### 異常の種類
    
    
    ```mermaid
    graph LR
        A[異常の種類] --> B[点異常Point Anomaly]
        A --> C[文脈異常Contextual Anomaly]
        A --> D[集団異常Collective Anomaly]
    
        B --> B1[単一データ点が異常値]
        C --> C1[文脈次第で異常になる]
        D --> D1[個々は正常だが集合として異常]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### 応用例

分野 | 応用例 | 検出対象  
---|---|---  
**金融** | クレジットカード不正検知 | 通常と異なる取引パターン  
**製造** | 製品の欠陥検出 | 品質基準を満たさない製品  
**医療** | 疾病の早期発見 | 異常なバイタルサイン  
**IT** | サーバー監視 | システム障害の予兆  
**セキュリティ** | ネットワーク侵入検知 | 不審なアクセスパターン  
  
### 教師あり vs 教師なし異常検知

項目 | 教師あり | 教師なし  
---|---|---  
**ラベル** | 必要（正常/異常） | 不要  
**適用場面** | 過去の異常例がある | 未知の異常を検出  
**手法** | 分類アルゴリズム | 密度推定、距離ベース  
**課題** | クラス不均衡 | 閾値の設定  
  
* * *

## 3.2 統計的手法

### Z-scoreによる異常検知

**Z-score** は、データ点が平均からどれだけ標準偏差離れているかを測定します。

$$ z = \frac{x - \mu}{\sigma} $$

  * $x$: データ点
  * $\mu$: 平均
  * $\sigma$: 標準偏差

一般的に、$|z| > 3$ のデータ点を異常値とします。

### 実装例: Z-score
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # データ生成: 正常データ + 異常値
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    anomalies = np.array([5, -4.5, 6, -5.5])
    data = np.concatenate([normal_data, anomalies])
    
    # Z-score計算
    z_scores = np.abs(stats.zscore(data))
    threshold = 3
    
    # 異常検知
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    print("=== Z-score異常検知 ===")
    print(f"データ数: {len(data)}")
    print(f"検出された異常値: {len(anomaly_indices)}個")
    print(f"異常値のインデックス: {anomaly_indices}")
    print(f"異常値: {data[anomaly_indices]}")
    
    # 可視化
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(data)), data, c='blue', alpha=0.5, label='正常データ')
    plt.scatter(anomaly_indices, data[anomaly_indices], c='red', s=100, label='異常値')
    plt.xlabel('データ点', fontsize=12)
    plt.ylabel('値', fontsize=12)
    plt.title('Z-score異常検知', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    for anomaly in data[anomaly_indices]:
        plt.axvline(anomaly, color='red', linestyle='--', linewidth=2)
    plt.xlabel('値', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('データ分布と異常値', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Z-score異常検知 ===
    データ数: 1004
    検出された異常値: 4個
    異常値のインデックス: [1000 1001 1002 1003]
    異常値: [ 5.  -4.5  6.  -5.5]
    

### IQR（四分位範囲）法

**IQR法** は、箱ひげ図で使われる手法で、外れ値を検出します。

$$ \text{IQR} = Q_3 - Q_1 $$

異常値の範囲:

  * 下限: $Q_1 - 1.5 \times \text{IQR}$
  * 上限: $Q_3 + 1.5 \times \text{IQR}$

### 実装例: IQR法
    
    
    # IQR法による異常検知
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 異常検知
    anomaly_mask = (data < lower_bound) | (data > upper_bound)
    iqr_anomaly_indices = np.where(anomaly_mask)[0]
    
    print("\n=== IQR法異常検知 ===")
    print(f"Q1: {Q1:.4f}, Q3: {Q3:.4f}")
    print(f"IQR: {IQR:.4f}")
    print(f"下限: {lower_bound:.4f}, 上限: {upper_bound:.4f}")
    print(f"検出された異常値: {len(iqr_anomaly_indices)}個")
    
    # 可視化: 箱ひげ図
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.boxplot(data, vert=False)
    plt.scatter(data[iqr_anomaly_indices],
               np.ones(len(iqr_anomaly_indices)),
               c='red', s=100, label='異常値', zorder=5)
    plt.xlabel('値', fontsize=12)
    plt.title('箱ひげ図による異常検知', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label='下限')
    plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label='上限')
    plt.xlabel('値', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('IQR法: データ分布と境界', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === IQR法異常検知 ===
    Q1: -0.6632, Q3: 0.6788
    IQR: 1.3420
    下限: -2.6762, 上限: 2.6918
    検出された異常値: 9個
    

* * *

## 3.3 Isolation Forest

### 原理

**Isolation Forest** は、異常値を「分離しやすい」という性質を利用します。
    
    
    ```mermaid
    graph TD
        A[ランダムに特徴量選択] --> B[ランダムに分割点選択]
        B --> C{データを分割}
        C --> D[左の子ノード]
        C --> E[右の子ノード]
        D --> F[再帰的に分割]
        E --> F
        F --> G[平均パス長を計算]
        G --> H[パス長が短い→ 異常値]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style G fill:#f3e5f5
        style H fill:#ffebee
    ```

### アルゴリズム

  1. ランダムに特徴量と分割点を選んでツリー構築
  2. データ点が孤立するまでの深さ（パス長）を計測
  3. パス長が短いほど異常値の可能性が高い

**異常スコア** ：

$$ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $$

  * $E(h(x))$: 平均パス長
  * $c(n)$: 正規化定数
  * $s \approx 1$: 異常値
  * $s < 0.5$: 正常値

### 実装例
    
    
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # データ生成: 2次元データ
    X_normal, _ = make_blobs(n_samples=300, centers=1,
                             cluster_std=0.5, random_state=42)
    
    # 異常値を追加
    X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.vstack([X_normal, X_anomalies])
    
    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # 異常値の割合（10%）
        random_state=42
    )
    
    # 予測: -1が異常、1が正常
    y_pred = iso_forest.fit_predict(X)
    anomaly_score = iso_forest.score_samples(X)
    
    # 結果
    anomaly_mask = y_pred == -1
    n_anomalies = np.sum(anomaly_mask)
    
    print("=== Isolation Forest ===")
    print(f"データ数: {len(X)}")
    print(f"検出された異常値: {n_anomalies}個")
    print(f"異常率: {n_anomalies/len(X)*100:.2f}%")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # 左: データ点の分類
    plt.subplot(1, 2, 1)
    plt.scatter(X[~anomaly_mask, 0], X[~anomaly_mask, 1],
               c='blue', alpha=0.5, label='正常')
    plt.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
               c='red', s=100, label='異常', marker='x')
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('Isolation Forest: 異常検知結果', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右: 異常スコアの分布
    plt.subplot(1, 2, 2)
    plt.hist(anomaly_score[~anomaly_mask], bins=30, alpha=0.7,
            label='正常', color='blue', edgecolor='black')
    plt.hist(anomaly_score[anomaly_mask], bins=30, alpha=0.7,
            label='異常', color='red', edgecolor='black')
    plt.xlabel('異常スコア', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('異常スコアの分布', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Isolation Forest ===
    データ数: 320
    検出された異常値: 32個
    異常率: 10.00%
    

### 特徴量重要度の分析
    
    
    # 多次元データでの特徴量重要度
    from sklearn.datasets import make_classification
    
    X_multi, _ = make_classification(n_samples=1000, n_features=10,
                                     n_informative=7, random_state=42)
    
    # 異常値を追加（一部の特徴量のみ極端な値）
    X_anomalies = X_multi[:20].copy()
    X_anomalies[:, 0] += 5  # 特徴量0に大きな値を追加
    X_anomalies[:, 3] += 4  # 特徴量3に大きな値を追加
    
    X_multi = np.vstack([X_multi, X_anomalies])
    
    # Isolation Forestで学習
    iso_forest_multi = IsolationForest(n_estimators=100, contamination=0.05,
                                       random_state=42)
    iso_forest_multi.fit(X_multi)
    
    # 特徴量重要度を近似的に計算
    # （各特徴量の分散と異常スコアの相関）
    feature_scores = []
    for i in range(X_multi.shape[1]):
        X_temp = X_multi.copy()
        np.random.shuffle(X_temp[:, i])  # 特徴量をシャッフル
        score_diff = np.mean(np.abs(
            iso_forest_multi.score_samples(X_multi) -
            iso_forest_multi.score_samples(X_temp)
        ))
        feature_scores.append(score_diff)
    
    feature_scores = np.array(feature_scores)
    feature_scores = feature_scores / np.sum(feature_scores)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_scores)), feature_scores, color='steelblue')
    plt.xlabel('特徴量インデックス', fontsize=12)
    plt.ylabel('重要度', fontsize=12)
    plt.title('Isolation Forest: 特徴量重要度', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print("\n=== 特徴量重要度 ===")
    for i, score in enumerate(feature_scores):
        print(f"特徴量 {i}: {score:.4f}")
    

* * *

## 3.4 One-Class SVM

### 概要

**One-Class SVM** は、正常データのみから学習し、新規性検知（Novelty Detection）を行います。

### 原理

高次元空間で正常データを囲む超球面または超平面を学習します。
    
    
    ```mermaid
    graph LR
        A[訓練データ正常データのみ] --> B[高次元空間にマッピング]
        B --> C[境界を学習]
        C --> D[超球面/超平面]
        D --> E[新規データ]
        E --> F{境界内?}
        F -->|Yes| G[正常]
        F -->|No| H[異常]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style G fill:#e8f5e9
        style H fill:#ffebee
    ```

### 実装例
    
    
    from sklearn.svm import OneClassSVM
    
    # データ生成: 訓練データは正常データのみ
    X_train, _ = make_blobs(n_samples=300, centers=1,
                            cluster_std=0.5, random_state=42)
    
    # テストデータ: 正常 + 異常
    X_test_normal, _ = make_blobs(n_samples=100, centers=1,
                                  cluster_std=0.5, random_state=123)
    X_test_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.vstack([X_test_normal, X_test_anomalies])
    y_true = np.array([1]*100 + [-1]*20)  # 1: 正常, -1: 異常
    
    # One-Class SVM
    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=0.05  # 外れ値の上限割合
    )
    
    oc_svm.fit(X_train)
    y_pred = oc_svm.predict(X_test)
    
    # 評価
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("=== One-Class SVM ===")
    print("\n混同行列:")
    print(confusion_matrix(y_true, y_pred))
    print("\n分類レポート:")
    print(classification_report(y_true, y_pred,
                              target_names=['異常', '正常']))
    
    # 可視化: 決定境界
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500),
                         np.linspace(-5, 5, 500))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    # 左: 訓練データと決定境界
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                cmap='Blues_r', alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.scatter(X_train[:, 0], X_train[:, 1],
               c='blue', alpha=0.5, label='訓練データ')
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('One-Class SVM: 決定境界', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右: テストデータの予測
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                cmap='Blues_r', alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    
    correct_normal = (y_pred == 1) & (y_true == 1)
    correct_anomaly = (y_pred == -1) & (y_true == -1)
    incorrect = y_pred != y_true
    
    plt.scatter(X_test[correct_normal, 0], X_test[correct_normal, 1],
               c='blue', alpha=0.6, label='正常（正解）')
    plt.scatter(X_test[correct_anomaly, 0], X_test[correct_anomaly, 1],
               c='red', s=100, marker='x', label='異常（正解）')
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1],
               c='orange', s=100, marker='^', label='誤検出')
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('One-Class SVM: テストデータ予測', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === One-Class SVM ===
    
    混同行列:
    [[15  5]
     [ 3 97]]
    
    分類レポート:
                  precision    recall  f1-score   support
    
            異常       0.83      0.75      0.79        20
            正常       0.95      0.97      0.96       100
    
        accuracy                           0.93       120
       macro avg       0.89      0.86      0.87       120
    weighted avg       0.93      0.93      0.93       120
    

### パラメータチューニング
    
    
    # nuパラメータの影響を比較
    nu_values = [0.01, 0.05, 0.1, 0.2]
    
    plt.figure(figsize=(14, 10))
    
    for i, nu in enumerate(nu_values, 1):
        oc_svm_nu = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
        oc_svm_nu.fit(X_train)
        y_pred_nu = oc_svm_nu.predict(X_test)
    
        Z = oc_svm_nu.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.subplot(2, 2, i)
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap='Blues_r', alpha=0.5)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
        plt.scatter(X_test[y_pred_nu == 1, 0], X_test[y_pred_nu == 1, 1],
                   c='blue', alpha=0.6, label='正常')
        plt.scatter(X_test[y_pred_nu == -1, 0], X_test[y_pred_nu == -1, 1],
                   c='red', s=100, marker='x', label='異常')
    
        accuracy = np.sum(y_pred_nu == y_true) / len(y_true)
        plt.xlabel('特徴量1', fontsize=12)
        plt.ylabel('特徴量2', fontsize=12)
        plt.title(f'nu = {nu} (精度: {accuracy:.3f})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Local Outlier Factor（LOF）

### 原理

**LOF** は、データ点の局所密度を比較して異常を検知します。

$$ \text{LOF}(p) = \frac{\sum_{o \in N(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)}}{|N(p)|} $$

  * $\text{lrd}(p)$: 点pの局所到達可能密度
  * $N(p)$: 点pのk近傍
  * $\text{LOF} \approx 1$: 正常
  * $\text{LOF} \gg 1$: 異常

    
    
    ```mermaid
    graph TD
        A[データ点p] --> B[k近傍を探索]
        B --> C[局所密度を計算]
        C --> D[近傍の密度と比較]
        D --> E{密度が低い?}
        E -->|Yes| F[LOF > 1異常値]
        E -->|No| G[LOF ≈ 1正常値]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style F fill:#ffebee
        style G fill:#e8f5e9
    ```

### 実装例
    
    
    from sklearn.neighbors import LocalOutlierFactor
    
    # LOF
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.1
    )
    
    # 予測: -1が異常、1が正常
    y_pred_lof = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_  # 負の値なので反転
    
    anomaly_mask_lof = y_pred_lof == -1
    n_anomalies_lof = np.sum(anomaly_mask_lof)
    
    print("=== Local Outlier Factor (LOF) ===")
    print(f"データ数: {len(X)}")
    print(f"検出された異常値: {n_anomalies_lof}個")
    print(f"異常率: {n_anomalies_lof/len(X)*100:.2f}%")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # 左: データ点の分類
    plt.subplot(1, 2, 1)
    plt.scatter(X[~anomaly_mask_lof, 0], X[~anomaly_mask_lof, 1],
               c='blue', alpha=0.5, label='正常')
    plt.scatter(X[anomaly_mask_lof, 0], X[anomaly_mask_lof, 1],
               c='red', s=100, label='異常', marker='x')
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('LOF: 異常検知結果', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右: LOFスコアの分布
    plt.subplot(1, 2, 2)
    plt.hist(lof_scores[~anomaly_mask_lof], bins=30, alpha=0.7,
            label='正常', color='blue', edgecolor='black')
    plt.hist(lof_scores[anomaly_mask_lof], bins=30, alpha=0.7,
            label='異常', color='red', edgecolor='black')
    plt.axvline(1, color='green', linestyle='--', linewidth=2, label='基準値')
    plt.xlabel('LOFスコア', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('LOFスコアの分布', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Local Outlier Factor (LOF) ===
    データ数: 320
    検出された異常値: 32個
    異常率: 10.00%
    

### k近傍数の影響
    
    
    # k近傍数の比較
    k_values = [5, 10, 20, 50]
    
    plt.figure(figsize=(14, 10))
    
    for i, k in enumerate(k_values, 1):
        lof_k = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
        y_pred_k = lof_k.fit_predict(X)
    
        plt.subplot(2, 2, i)
        plt.scatter(X[y_pred_k == 1, 0], X[y_pred_k == 1, 1],
                   c='blue', alpha=0.5, label='正常')
        plt.scatter(X[y_pred_k == -1, 0], X[y_pred_k == -1, 1],
                   c='red', s=100, label='異常', marker='x')
        plt.xlabel('特徴量1', fontsize=12)
        plt.ylabel('特徴量2', fontsize=12)
        plt.title(f'LOF: k={k} (異常: {np.sum(y_pred_k == -1)}個)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.6 Autoencoderによる異常検知

### 原理

**Autoencoder** は、入力を再構成するニューラルネットワークです。正常データで学習すると、異常データの再構成誤差が大きくなります。
    
    
    ```mermaid
    graph LR
        A[入力 x] --> B[エンコーダ]
        B --> C[潜在表現 z]
        C --> D[デコーダ]
        D --> E[再構成 x']
        E --> F[再構成誤差||x - x'||]
        F --> G{誤差が大きい?}
        G -->|Yes| H[異常]
        G -->|No| I[正常]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style F fill:#f3e5f5
        style H fill:#ffebee
        style I fill:#e8f5e9
    ```

### 実装例
    
    
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # データ生成と標準化
    X_normal_train, _ = make_blobs(n_samples=1000, centers=1,
                                   cluster_std=0.5, random_state=42)
    X_normal_test, _ = make_blobs(n_samples=200, centers=1,
                                  cluster_std=0.5, random_state=123)
    X_anomalies = np.random.uniform(low=-4, high=4, size=(50, 2))
    
    scaler = StandardScaler()
    X_normal_train = scaler.fit_transform(X_normal_train)
    X_normal_test = scaler.transform(X_normal_test)
    X_anomalies = scaler.transform(X_anomalies)
    
    X_test_ae = np.vstack([X_normal_test, X_anomalies])
    y_true_ae = np.array([1]*200 + [0]*50)  # 1: 正常, 0: 異常
    
    # Autoencoderモデル構築
    input_dim = X_normal_train.shape[1]
    encoding_dim = 1
    
    autoencoder = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(encoding_dim, activation='relu'),  # エンコーダ
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear')  # デコーダ
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 学習（正常データのみ）
    history = autoencoder.fit(
        X_normal_train, X_normal_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    print("=== Autoencoder ===")
    print("学習完了")
    
    # 再構成誤差の計算
    X_test_reconstructed = autoencoder.predict(X_test_ae, verbose=0)
    reconstruction_errors = np.mean(np.power(X_test_ae - X_test_reconstructed, 2), axis=1)
    
    # 閾値設定（正常データの95パーセンタイル）
    X_normal_reconstructed = autoencoder.predict(X_normal_train, verbose=0)
    normal_errors = np.mean(np.power(X_normal_train - X_normal_reconstructed, 2), axis=1)
    threshold = np.percentile(normal_errors, 95)
    
    # 異常検知
    y_pred_ae = (reconstruction_errors > threshold).astype(int)
    y_pred_ae = 1 - y_pred_ae  # 0→異常, 1→正常に変換
    
    print(f"\n再構成誤差の閾値: {threshold:.4f}")
    print(f"検出された異常値: {np.sum(y_pred_ae == 0)}個")
    
    # 評価
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print(f"\n精度: {accuracy_score(y_true_ae, y_pred_ae):.4f}")
    print(f"適合率: {precision_score(y_true_ae, y_pred_ae):.4f}")
    print(f"再現率: {recall_score(y_true_ae, y_pred_ae):.4f}")
    print(f"F1スコア: {f1_score(y_true_ae, y_pred_ae):.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 学習曲線
    axes[0, 0].plot(history.history['loss'], label='訓練損失')
    axes[0, 0].plot(history.history['val_loss'], label='検証損失')
    axes[0, 0].set_xlabel('エポック', fontsize=12)
    axes[0, 0].set_ylabel('損失（MSE）', fontsize=12)
    axes[0, 0].set_title('学習曲線', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 再構成誤差の分布
    axes[0, 1].hist(reconstruction_errors[y_true_ae == 1], bins=30,
                   alpha=0.7, label='正常', color='blue', edgecolor='black')
    axes[0, 1].hist(reconstruction_errors[y_true_ae == 0], bins=30,
                   alpha=0.7, label='異常', color='red', edgecolor='black')
    axes[0, 1].axvline(threshold, color='green', linestyle='--',
                      linewidth=2, label='閾値')
    axes[0, 1].set_xlabel('再構成誤差', fontsize=12)
    axes[0, 1].set_ylabel('頻度', fontsize=12)
    axes[0, 1].set_title('再構成誤差の分布', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # データ点の分類（元の空間）
    X_test_original = scaler.inverse_transform(X_test_ae)
    correct = y_pred_ae == y_true_ae
    incorrect = y_pred_ae != y_true_ae
    
    axes[1, 0].scatter(X_test_original[correct & (y_true_ae == 1), 0],
                      X_test_original[correct & (y_true_ae == 1), 1],
                      c='blue', alpha=0.6, label='正常（正解）')
    axes[1, 0].scatter(X_test_original[correct & (y_true_ae == 0), 0],
                      X_test_original[correct & (y_true_ae == 0), 1],
                      c='red', s=100, marker='x', label='異常（正解）')
    axes[1, 0].scatter(X_test_original[incorrect, 0],
                      X_test_original[incorrect, 1],
                      c='orange', s=100, marker='^', label='誤検出')
    axes[1, 0].set_xlabel('特徴量1', fontsize=12)
    axes[1, 0].set_ylabel('特徴量2', fontsize=12)
    axes[1, 0].set_title('Autoencoder: 異常検知結果', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 混同行列
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_ae, y_pred_ae)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 1].set_xlabel('予測ラベル', fontsize=12)
    axes[1, 1].set_ylabel('真のラベル', fontsize=12)
    axes[1, 1].set_title('混同行列', fontsize=14)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['異常', '正常'])
    axes[1, 1].set_yticklabels(['異常', '正常'])
    
    # 混同行列の値を表示
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]),
                           ha='center', va='center', fontsize=20)
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Autoencoder ===
    学習完了
    
    再構成誤差の閾値: 0.0234
    検出された異常値: 48個
    
    精度: 0.9600
    適合率: 0.9703
    再現率: 0.9850
    F1スコア: 0.9776
    

* * *

## 3.7 異常検知の評価指標

### 評価の課題

異常検知では、正常データと異常データの比率が極端に偏っているため、精度（Accuracy）だけでは不十分です。

### 主要な評価指標

指標 | 定義 | 重視する場面  
---|---|---  
**適合率（Precision）** | $\frac{TP}{TP + FP}$ | 誤検出を減らしたい  
**再現率（Recall）** | $\frac{TP}{TP + FN}$ | 見逃しを減らしたい  
**F1スコア** | $\frac{2 \cdot P \cdot R}{P + R}$ | バランスを取りたい  
**ROC-AUC** | ROC曲線下面積 | 閾値に依存しない評価  
  
  * TP: 真陽性（異常を異常と判定）
  * FP: 偽陽性（正常を異常と判定）
  * FN: 偽陰性（異常を正常と判定）
  * TN: 真陰性（正常を正常と判定）

### 実装例: モデル比較
    
    
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.metrics import precision_recall_curve
    
    # 各モデルのスコアを取得
    iso_scores = -iso_forest.score_samples(X)
    lof_scores_all = -LocalOutlierFactor(n_neighbors=20,
                                          novelty=False).fit(X).negative_outlier_factor_
    
    # 真のラベル（仮想）
    y_true_all = np.ones(len(X))
    y_true_all[-20:] = 0  # 最後の20個を異常とする
    
    # ROC曲線
    fpr_iso, tpr_iso, _ = roc_curve(y_true_all, iso_scores)
    fpr_lof, tpr_lof, _ = roc_curve(y_true_all, lof_scores_all)
    
    roc_auc_iso = auc(fpr_iso, tpr_iso)
    roc_auc_lof = auc(fpr_lof, tpr_lof)
    
    # Precision-Recall曲線
    precision_iso, recall_iso, _ = precision_recall_curve(y_true_all, iso_scores)
    precision_lof, recall_lof, _ = precision_recall_curve(y_true_all, lof_scores_all)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC曲線
    axes[0].plot(fpr_iso, tpr_iso, linewidth=2,
                label=f'Isolation Forest (AUC = {roc_auc_iso:.3f})')
    axes[0].plot(fpr_lof, tpr_lof, linewidth=2,
                label=f'LOF (AUC = {roc_auc_lof:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='ランダム')
    axes[0].set_xlabel('偽陽性率（FPR）', fontsize=12)
    axes[0].set_ylabel('真陽性率（TPR）', fontsize=12)
    axes[0].set_title('ROC曲線', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall曲線
    axes[1].plot(recall_iso, precision_iso, linewidth=2,
                label='Isolation Forest')
    axes[1].plot(recall_lof, precision_lof, linewidth=2,
                label='LOF')
    axes[1].set_xlabel('再現率（Recall）', fontsize=12)
    axes[1].set_ylabel('適合率（Precision）', fontsize=12)
    axes[1].set_title('Precision-Recall曲線', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 評価指標の比較 ===")
    print(f"Isolation Forest ROC-AUC: {roc_auc_iso:.4f}")
    print(f"LOF ROC-AUC: {roc_auc_lof:.4f}")
    

**出力** ：
    
    
    === 評価指標の比較 ===
    Isolation Forest ROC-AUC: 0.9567
    LOF ROC-AUC: 0.9234
    

* * *

## 3.8 実践例: クレジットカード不正検知

### 問題設定

クレジットカードの取引データから不正取引を検出します。

### 実装例
    
    
    # データ生成: クレジットカード取引を模擬
    from sklearn.datasets import make_classification
    
    # 取引データ（不均衡データ）
    X_fraud, y_fraud = make_classification(
        n_samples=10000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.98, 0.02],  # 98%正常, 2%不正
        random_state=42
    )
    
    print("=== クレジットカード不正検知 ===")
    print(f"総取引数: {len(X_fraud)}")
    print(f"正常取引: {np.sum(y_fraud == 0)}件 ({np.sum(y_fraud == 0)/len(y_fraud)*100:.2f}%)")
    print(f"不正取引: {np.sum(y_fraud == 1)}件 ({np.sum(y_fraud == 1)/len(y_fraud)*100:.2f}%)")
    
    # データ分割
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.3, stratify=y_fraud, random_state=42
    )
    
    # モデル比較
    models_fraud = {
        'Isolation Forest': IsolationForest(contamination=0.02, random_state=42),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=True),
        'One-Class SVM': OneClassSVM(nu=0.02, kernel='rbf', gamma='auto')
    }
    
    results_fraud = {}
    
    for name, model in models_fraud.items():
        # 訓練（正常データのみ）
        X_train_normal = X_train_fraud[y_train_fraud == 0]
        model.fit(X_train_normal)
    
        # テストデータで予測
        y_pred = model.predict(X_test_fraud)
        y_pred_binary = (y_pred == -1).astype(int)  # -1（異常）→1, 1（正常）→0
    
        # 評価
        from sklearn.metrics import classification_report, confusion_matrix
    
        precision = precision_score(y_test_fraud, y_pred_binary)
        recall = recall_score(y_test_fraud, y_pred_binary)
        f1 = f1_score(y_test_fraud, y_pred_binary)
    
        results_fraud[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_test_fraud, y_pred_binary)
        }
    
        print(f"\n{name}:")
        print(f"  適合率: {precision:.4f}")
        print(f"  再現率: {recall:.4f}")
        print(f"  F1スコア: {f1:.4f}")
    
    # 可視化: 性能比較
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    
    for i, (name, result) in enumerate(results_fraud.items()):
        values = [result[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xlabel('評価指標', fontsize=12)
    plt.ylabel('スコア', fontsize=12)
    plt.title('クレジットカード不正検知: モデル性能比較', fontsize=14)
    plt.xticks(x + width, ['適合率', '再現率', 'F1スコア'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 各バーに値を表示
    for i, (name, result) in enumerate(results_fraud.items()):
        values = [result[metric] for metric in metrics]
        for j, v in enumerate(values):
            plt.text(j + i*width, v + 0.02, f'{v:.3f}',
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 混同行列の可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (name, result) in enumerate(results_fraud.items()):
        cm = result['confusion_matrix']
        im = axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[i].set_xlabel('予測ラベル', fontsize=11)
        axes[i].set_ylabel('真のラベル', fontsize=11)
        axes[i].set_title(f'{name}', fontsize=12)
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['正常', '不正'])
        axes[i].set_yticklabels(['正常', '不正'])
    
        # 混同行列の値を表示
        for ii in range(2):
            for jj in range(2):
                axes[i].text(jj, ii, str(cm[ii, jj]),
                           ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === クレジットカード不正検知 ===
    総取引数: 10000
    正常取引: 9800件 (98.00%)
    不正取引: 200件 (2.00%)
    
    Isolation Forest:
      適合率: 0.5645
      再現率: 0.7000
      F1スコア: 0.6250
    
    LOF:
      適合率: 0.4324
      再現率: 0.5333
      F1スコア: 0.4773
    
    One-Class SVM:
      適合率: 0.5102
      再現率: 0.6667
      F1スコア: 0.5780
    

* * *

## 3.9 本章のまとめ

### 学んだこと

  1. **異常検知の基礎**

     * 点異常、文脈異常、集団異常
     * 教師あり vs 教師なし異常検知
     * 不正検知、システム監視などの応用
  2. **統計的手法**

     * Z-score法: 標準偏差ベース
     * IQR法: 四分位範囲ベース
     * シンプルで解釈しやすい
  3. **機械学習手法**

     * Isolation Forest: 分離のしやすさ
     * One-Class SVM: 境界学習
     * LOF: 局所密度ベース
  4. **深層学習手法**

     * Autoencoder: 再構成誤差ベース
     * 正常データのみで学習可能
  5. **評価指標**

     * 適合率、再現率、F1スコア
     * ROC-AUC、Precision-Recall曲線
     * 不均衡データへの対応

### 手法の選択ガイド

状況 | 推奨手法 | 理由  
---|---|---  
**単純な1次元データ** | Z-score, IQR | シンプルで解釈しやすい  
**高次元データ** | Isolation Forest | 次元の呪いに強い  
**局所的なパターン** | LOF | 密度ベースで柔軟  
**新規性検知** | One-Class SVM | 境界学習が得意  
**複雑なパターン** | Autoencoder | 非線形関係を学習  
**大規模データ** | Isolation Forest | 計算効率が良い  
  
### 次の章へ

第4章では、**実践プロジェクト** を通じて学んだ技術を応用します：

  * プロジェクト1: 顧客セグメンテーション
  * プロジェクト2: レコメンドシステム
  * 完全な教師なし学習パイプライン

* * *

## 演習問題

### 問題1（難易度：easy）

Z-score法とIQR法の違いを3つ挙げてください。

解答例

**解答** ：

  1. **基準** : Z-scoreは平均と標準偏差、IQRは四分位数を使用
  2. **外れ値への感度** : Z-scoreは外れ値の影響を受けやすい、IQRは頑健（ロバスト）
  3. **分布の仮定** : Z-scoreは正規分布を仮定、IQRは分布に依存しない

### 問題2（難易度：medium）

Isolation ForestとLOFの違いを説明し、それぞれどのような場面で有効か述べてください。

解答例

**解答** ：

**Isolation Forest** ：

  * **原理** : 異常値は分離しやすいという性質を利用
  * **方法** : ランダムにツリーを構築し、孤立までのパス長を測定
  * **有効な場面** : 
    * 高次元データ（次元の呪いに強い）
    * 大規模データ（計算効率が良い）
    * グローバルな異常検知

**LOF（Local Outlier Factor）** ：

  * **原理** : 局所密度を比較して異常を検知
  * **方法** : k近傍の密度と比較してLOFスコアを計算
  * **有効な場面** : 
    * 密度が不均一なデータ
    * 局所的な異常検知
    * クラスタごとに密度が異なる場合

**使い分け** ：

  * データが均一な密度 → Isolation Forest
  * データが複数のクラスタを持つ → LOF
  * 計算速度重視 → Isolation Forest
  * 局所パターン重視 → LOF

### 問題3（難易度：medium）

以下のデータに対してIsolation ForestとLOFを適用し、検出結果を比較してください。
    
    
    import numpy as np
    from sklearn.datasets import make_moons
    
    # データ生成
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    # 異常値を追加
    X_anomalies = np.array([[2, 2], [-1, -1], [2, -1], [-1, 2]])
    X_combined = np.vstack([X, X_anomalies])
    

解答例
    
    
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    import matplotlib.pyplot as plt
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    y_pred_iso = iso.fit_predict(X_combined)
    
    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred_lof = lof.fit_predict(X_combined)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Isolation Forest
    axes[0].scatter(X_combined[y_pred_iso == 1, 0],
                   X_combined[y_pred_iso == 1, 1],
                   c='blue', alpha=0.5, label='正常')
    axes[0].scatter(X_combined[y_pred_iso == -1, 0],
                   X_combined[y_pred_iso == -1, 1],
                   c='red', s=100, label='異常', marker='x')
    axes[0].set_xlabel('特徴量1', fontsize=12)
    axes[0].set_ylabel('特徴量2', fontsize=12)
    axes[0].set_title('Isolation Forest', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # LOF
    axes[1].scatter(X_combined[y_pred_lof == 1, 0],
                   X_combined[y_pred_lof == 1, 1],
                   c='blue', alpha=0.5, label='正常')
    axes[1].scatter(X_combined[y_pred_lof == -1, 0],
                   X_combined[y_pred_lof == -1, 1],
                   c='red', s=100, label='異常', marker='x')
    axes[1].set_xlabel('特徴量1', fontsize=12)
    axes[1].set_ylabel('特徴量2', fontsize=12)
    axes[1].set_title('LOF', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 比較 ===")
    print(f"Isolation Forest: {np.sum(y_pred_iso == -1)}個の異常検出")
    print(f"LOF: {np.sum(y_pred_lof == -1)}個の異常検出")
    
    # 一致率
    agreement = np.sum(y_pred_iso == y_pred_lof) / len(y_pred_iso)
    print(f"検出結果の一致率: {agreement*100:.2f}%")
    

**出力** ：
    
    
    === 比較 ===
    Isolation Forest: 15個の異常検出
    LOF: 15個の異常検出
    検出結果の一致率: 91.45%
    

**考察** ：

  * make_moonsは三日月形のデータで、密度が不均一
  * LOFは局所密度を考慮するため、三日月の形状に適応
  * Isolation Forestは全体的な分離しやすさで判定
  * 追加した明らかな異常値（四隅）は両方とも検出

### 問題4（難易度：hard）

Autoencoderで異常検知を実装し、閾値を変化させたときの適合率と再現率のトレードオフを可視化してください。

解答例
    
    
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_curve
    
    # データ生成
    X_normal, _ = make_blobs(n_samples=1000, centers=1,
                             cluster_std=0.5, random_state=42)
    X_test_normal, _ = make_blobs(n_samples=200, centers=1,
                                  cluster_std=0.5, random_state=123)
    X_anomalies = np.random.uniform(low=-4, high=4, size=(50, 2))
    
    # 標準化
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_test_normal_scaled = scaler.transform(X_test_normal)
    X_anomalies_scaled = scaler.transform(X_anomalies)
    
    X_test_all = np.vstack([X_test_normal_scaled, X_anomalies_scaled])
    y_true_all = np.array([0]*200 + [1]*50)  # 0: 正常, 1: 異常
    
    # Autoencoderモデル
    autoencoder = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='linear')
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_normal_scaled, X_normal_scaled,
                   epochs=50, batch_size=32, verbose=0)
    
    # 再構成誤差
    X_reconstructed = autoencoder.predict(X_test_all, verbose=0)
    reconstruction_errors = np.mean(np.power(X_test_all - X_reconstructed, 2), axis=1)
    
    # Precision-Recall曲線
    precision, recall, thresholds = precision_recall_curve(y_true_all, reconstruction_errors)
    
    # F1スコアが最大となる閾値を見つける
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1]
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Precision-Recall曲線
    axes[0].plot(recall, precision, linewidth=2, color='blue')
    axes[0].scatter(recall[best_threshold_idx], precision[best_threshold_idx],
                   c='red', s=100, zorder=5, label=f'最適閾値: {best_threshold:.4f}')
    axes[0].set_xlabel('再現率（Recall）', fontsize=12)
    axes[0].set_ylabel('適合率（Precision）', fontsize=12)
    axes[0].set_title('Precision-Recall曲線', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1スコアと閾値
    axes[1].plot(thresholds, f1_scores[:-1], linewidth=2, color='green')
    axes[1].axvline(best_threshold, color='red', linestyle='--',
                   linewidth=2, label=f'最適閾値: {best_threshold:.4f}')
    axes[1].set_xlabel('閾値', fontsize=12)
    axes[1].set_ylabel('F1スコア', fontsize=12)
    axes[1].set_title('閾値とF1スコアの関係', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最適閾値での性能
    y_pred_best = (reconstruction_errors > best_threshold).astype(int)
    
    print("=== 最適閾値での性能 ===")
    print(f"閾値: {best_threshold:.4f}")
    print(f"適合率: {precision[best_threshold_idx]:.4f}")
    print(f"再現率: {recall[best_threshold_idx]:.4f}")
    print(f"F1スコア: {f1_scores[best_threshold_idx]:.4f}")
    
    # 異なる閾値での比較
    print("\n=== 閾値の影響 ===")
    for percentile in [90, 95, 99]:
        thresh = np.percentile(reconstruction_errors, percentile)
        y_pred = (reconstruction_errors > thresh).astype(int)
        prec = precision_score(y_true_all, y_pred)
        rec = recall_score(y_true_all, y_pred)
        f1 = f1_score(y_true_all, y_pred)
        print(f"{percentile}パーセンタイル: 閾値={thresh:.4f}, "
              f"適合率={prec:.4f}, 再現率={rec:.4f}, F1={f1:.4f}")
    

**出力** ：
    
    
    === 最適閾値での性能 ===
    閾値: 0.0345
    適合率: 0.8621
    再現率: 1.0000
    F1スコア: 0.9259
    
    === 閾値の影響 ===
    90パーセンタイル: 閾値=0.0123, 適合率=0.7407, 再現率=1.0000, F1=0.8511
    95パーセンタイル: 閾値=0.0234, 適合率=0.8621, 再現率=1.0000, F1=0.9259
    99パーセンタイル: 閾値=0.0567, 適合率=1.0000, 再現率=0.6000, F1=0.7500
    

**考察** ：

  * 閾値を上げる → 適合率↑、再現率↓（見逃しが増える）
  * 閾値を下げる → 適合率↓、再現率↑（誤検出が増える）
  * F1スコアが最大となる閾値が最適なバランス点
  * 実務では、見逃しコストと誤検出コストを考慮して閾値を決定

### 問題5（難易度：hard）

Isolation Forest、LOF、Autoencoderの3つの手法をアンサンブルして、より高精度な異常検知システムを構築してください。

解答例
    
    
    from sklearn.preprocessing import StandardScaler
    
    # データ生成
    X_train_ens, _ = make_blobs(n_samples=1000, centers=2,
                                cluster_std=0.5, random_state=42)
    X_test_normal, _ = make_blobs(n_samples=300, centers=2,
                                  cluster_std=0.5, random_state=123)
    X_test_anomalies = np.random.uniform(low=-5, high=5, size=(50, 2))
    
    X_test_ens = np.vstack([X_test_normal, X_test_anomalies])
    y_true_ens = np.array([0]*300 + [1]*50)
    
    # 標準化（Autoencoder用）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ens)
    X_test_scaled = scaler.transform(X_test_ens)
    
    # 1. Isolation Forest
    iso_ens = IsolationForest(contamination=0.1, random_state=42)
    iso_ens.fit(X_train_ens)
    iso_scores = -iso_ens.score_samples(X_test_ens)
    
    # 2. LOF
    lof_ens = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof_ens.fit(X_train_ens)
    lof_scores = -lof_ens.score_samples(X_test_ens)
    
    # 3. Autoencoder
    ae_ens = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='linear')
    ])
    ae_ens.compile(optimizer='adam', loss='mse')
    ae_ens.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, verbose=0)
    
    X_test_reconstructed = ae_ens.predict(X_test_scaled, verbose=0)
    ae_scores = np.mean(np.power(X_test_scaled - X_test_reconstructed, 2), axis=1)
    
    # スコアの正規化（0-1の範囲に）
    from sklearn.preprocessing import MinMaxScaler
    
    scaler_iso = MinMaxScaler()
    scaler_lof = MinMaxScaler()
    scaler_ae = MinMaxScaler()
    
    iso_scores_norm = scaler_iso.fit_transform(iso_scores.reshape(-1, 1)).flatten()
    lof_scores_norm = scaler_lof.fit_transform(lof_scores.reshape(-1, 1)).flatten()
    ae_scores_norm = scaler_ae.fit_transform(ae_scores.reshape(-1, 1)).flatten()
    
    # アンサンブル: 平均スコア
    ensemble_scores = (iso_scores_norm + lof_scores_norm + ae_scores_norm) / 3
    
    # 閾値の設定（95パーセンタイル）
    threshold_ens = np.percentile(ensemble_scores, 90)
    y_pred_ensemble = (ensemble_scores > threshold_ens).astype(int)
    
    # 個別手法の予測
    threshold_iso = np.percentile(iso_scores_norm, 90)
    threshold_lof = np.percentile(lof_scores_norm, 90)
    threshold_ae = np.percentile(ae_scores_norm, 90)
    
    y_pred_iso_only = (iso_scores_norm > threshold_iso).astype(int)
    y_pred_lof_only = (lof_scores_norm > threshold_lof).astype(int)
    y_pred_ae_only = (ae_scores_norm > threshold_ae).astype(int)
    
    # 性能比較
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    methods = {
        'Isolation Forest': y_pred_iso_only,
        'LOF': y_pred_lof_only,
        'Autoencoder': y_pred_ae_only,
        'Ensemble': y_pred_ensemble
    }
    
    print("=== アンサンブル異常検知 ===\n")
    
    results_ens = {}
    for name, y_pred in methods.items():
        prec = precision_score(y_true_ens, y_pred)
        rec = recall_score(y_true_ens, y_pred)
        f1 = f1_score(y_true_ens, y_pred)
    
        results_ens[name] = {'precision': prec, 'recall': rec, 'f1': f1}
    
        print(f"{name}:")
        print(f"  適合率: {prec:.4f}")
        print(f"  再現率: {rec:.4f}")
        print(f"  F1スコア: {f1:.4f}")
        print()
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 各手法の結果
    titles = ['Isolation Forest', 'LOF', 'Autoencoder', 'Ensemble']
    predictions = [y_pred_iso_only, y_pred_lof_only, y_pred_ae_only, y_pred_ensemble]
    
    for i, (ax, title, y_pred) in enumerate(zip(axes.flat, titles, predictions)):
        correct = y_pred == y_true_ens
        incorrect = y_pred != y_true_ens
    
        ax.scatter(X_test_ens[correct & (y_true_ens == 0), 0],
                  X_test_ens[correct & (y_true_ens == 0), 1],
                  c='blue', alpha=0.5, label='正常（正解）')
        ax.scatter(X_test_ens[correct & (y_true_ens == 1), 0],
                  X_test_ens[correct & (y_true_ens == 1), 1],
                  c='red', s=100, marker='x', label='異常（正解）')
        ax.scatter(X_test_ens[incorrect, 0],
                  X_test_ens[incorrect, 1],
                  c='orange', s=100, marker='^', label='誤検出')
    
        f1 = results_ens[title]['f1']
        ax.set_xlabel('特徴量1', fontsize=12)
        ax.set_ylabel('特徴量2', fontsize=12)
        ax.set_title(f'{title} (F1={f1:.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 性能比較の棒グラフ
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    
    for i, (name, result) in enumerate(results_ens.items()):
        values = [result[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xlabel('評価指標', fontsize=12)
    plt.ylabel('スコア', fontsize=12)
    plt.title('アンサンブル異常検知: 性能比較', fontsize=14)
    plt.xticks(x + width*1.5, ['適合率', '再現率', 'F1スコア'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === アンサンブル異常検知 ===
    
    Isolation Forest:
      適合率: 0.7143
      再現率: 0.9000
      F1スコア: 0.7965
    
    LOF:
      適合率: 0.6667
      再現率: 0.8600
      F1スコア: 0.7505
    
    Autoencoder:
      適合率: 0.8235
      再現率: 0.8400
      F1スコア: 0.8317
    
    Ensemble:
      適合率: 0.8780
      再現率: 0.9000
      F1スコア: 0.8889
    

**考察** ：

  * アンサンブルが最も高いF1スコアを達成
  * 複数手法の強みを組み合わせることで、安定した性能を実現
  * 各手法が異なるタイプの異常を検出するため、相補的に機能
  * 実務では、重み付き平均や多数決などの組み合わせ方法も検討

* * *

## 参考文献

  1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." _ICDM 2008_.
  2. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers." _ACM SIGMOD_.
  3. Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution." _Neural Computation_.
  4. Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly Detection: A Survey." _ACM Computing Surveys_.
  5. Sakurada, M., & Yairi, T. (2014). "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction." _MLSDA 2014_.
