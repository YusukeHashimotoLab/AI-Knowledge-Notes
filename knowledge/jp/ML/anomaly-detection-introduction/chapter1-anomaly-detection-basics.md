---
title: 第1章：異常検知基礎
chapter_title: 第1章：異常検知基礎
subtitle: 異常検知の基本概念とタスク設計
reading_time: 25-30分
difficulty: 初級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 異常検知の定義と種類を理解する
  * ✅ Point, Contextual, Collective異常を区別できる
  * ✅ 異常検知のタスク分類と選択基準を理解する
  * ✅ 適切な評価指標を選択できる
  * ✅ 代表的なデータセットと可視化手法を使える
  * ✅ 異常検知の課題と対処法を理解する

* * *

## 1.1 異常検知とは

### 異常の定義

**異常検知（Anomaly Detection）** は、正常なパターンから大きく逸脱したデータポイントを識別するタスクです。

> 「異常（Anomaly）」とは、多数の正常なデータとは異なる、稀で予期しないパターンを持つ観測値です。

### 異常の3つのタイプ

タイプ | 説明 | 例  
---|---|---  
**Point Anomaly**  
（点異常） | 個別のデータポイントが異常 | クレジットカードで突然の高額決済  
**Contextual Anomaly**  
（文脈的異常） | 特定の文脈でのみ異常 | 夏の気温35℃は正常、冬は異常  
**Collective Anomaly**  
（集団異常） | データの集合が異常パターン | 心電図の異常波形の連続  
  
### 異常検知の応用例

#### 1\. 不正検知（Fraud Detection）

  * クレジットカード詐欺の検出
  * 保険金請求の不正検知
  * マネーロンダリング検知

#### 2\. 製造業（Manufacturing）

  * 製品の不良品検出
  * 設備の故障予知
  * 品質管理の異常検知

#### 3\. 医療（Healthcare）

  * 疾病の早期発見
  * 医療画像での腫瘍検出
  * バイタルサインの異常検知

#### 4\. ITシステム（Cybersecurity & Operations）

  * ネットワーク侵入検知
  * サーバー障害の予測
  * 異常トラフィックの検出

### 異常検知のビジネス価値
    
    
    ```mermaid
    graph LR
        A[異常検知] --> B[コスト削減]
        A --> C[リスク低減]
        A --> D[収益向上]
    
        B --> B1[故障前の予防保守]
        B --> B2[不良品の早期発見]
    
        C --> C1[セキュリティ侵害防止]
        C --> C2[不正取引の防止]
    
        D --> D1[ダウンタイム削減]
        D --> D2[顧客満足度向上]
    
        style A fill:#7b2cbf,color:#fff
        style B fill:#e8f5e9
        style C fill:#fff3e0
        style D fill:#e3f2fd
    ```

### 実例：異常検知の基本
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    # 正常データの生成
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1,
                             cluster_std=1.0, center_box=(0, 0))
    
    # 異常データの追加（3種類）
    # 1. Point Anomaly: 離れた点
    point_anomalies = np.array([[8, 8], [-8, -8], [8, -8]])
    
    # 2. Contextual Anomaly: 正常範囲だが文脈的に異常
    # （例：時系列データでの季節外れの値）
    contextual_anomalies = np.array([[2, 2], [-2, -2]])
    
    # 3. Collective Anomaly: 集団として異常
    collective_anomalies = np.random.normal(loc=[5, 5], scale=0.3, size=(10, 2))
    
    # 全データの結合
    X_all = np.vstack([X_normal, point_anomalies,
                       contextual_anomalies, collective_anomalies])
    
    # 可視化
    plt.figure(figsize=(12, 8))
    
    plt.scatter(X_normal[:, 0], X_normal[:, 1],
                c='blue', alpha=0.5, s=50, label='正常データ', edgecolors='black')
    plt.scatter(point_anomalies[:, 0], point_anomalies[:, 1],
                c='red', s=200, marker='X', label='Point Anomaly',
                edgecolors='black', linewidths=2)
    plt.scatter(contextual_anomalies[:, 0], contextual_anomalies[:, 1],
                c='orange', s=200, marker='s', label='Contextual Anomaly',
                edgecolors='black', linewidths=2)
    plt.scatter(collective_anomalies[:, 0], collective_anomalies[:, 1],
                c='purple', s=100, marker='^', label='Collective Anomaly',
                edgecolors='black', linewidths=2)
    
    plt.xlabel('特徴量 1', fontsize=12)
    plt.ylabel('特徴量 2', fontsize=12)
    plt.title('異常の3つのタイプ', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== データ統計 ===")
    print(f"正常データ: {len(X_normal)}個")
    print(f"Point Anomaly: {len(point_anomalies)}個")
    print(f"Contextual Anomaly: {len(contextual_anomalies)}個")
    print(f"Collective Anomaly: {len(collective_anomalies)}個")
    print(f"異常率: {(len(point_anomalies) + len(contextual_anomalies) + len(collective_anomalies)) / len(X_all) * 100:.1f}%")
    

> **重要** : 異常検知では、正常データが大多数を占め、異常データは少数（通常1-5%）です。

* * *

## 1.2 異常検知のタスク分類

### 1\. 学習方法による分類

タイプ | ラベル情報 | 使用場面 | アルゴリズム例  
---|---|---|---  
**教師あり学習**  
(Supervised) | 正常・異常ラベル両方 | ラベル付きデータが豊富 | Random Forest, SVM  
**半教師あり学習**  
(Semi-supervised) | 正常ラベルのみ | 正常データのみラベル化 | One-Class SVM, Autoencoder  
**教師なし学習**  
(Unsupervised) | ラベルなし | ラベル取得が困難 | Isolation Forest, LOF, DBSCAN  
  
### 2\. Novelty Detection vs Outlier Detection

タイプ | 訓練データ | 目的 | 例  
---|---|---|---  
**Novelty Detection**  
（新規性検知） | 正常データのみ | 新しいパターンの検出 | 新種のマルウェア検出  
**Outlier Detection**  
（外れ値検出） | 正常+異常混在 | 既存データ内の異常検出 | センサーデータのノイズ除去  
  
### 3\. Online vs Offline Detection

タイプ | 処理タイミング | 特徴 | 応用例  
---|---|---|---  
**Online Detection**  
（リアルタイム） | データ到着時 | 低遅延、逐次更新 | ネットワーク侵入検知  
**Offline Detection**  
（バッチ処理） | 一括処理 | 高精度、全体最適化 | 月次レポートの異常分析  
  
### タスク選択の決定フロー
    
    
    ```mermaid
    graph TD
        A[異常検知タスク設計] --> B{ラベルデータはある?}
        B -->|両方ある| C[教師あり学習]
        B -->|正常のみ| D[半教師あり学習 / Novelty Detection]
        B -->|なし| E[教師なし学習 / Outlier Detection]
    
        C --> F{リアルタイム?}
        D --> F
        E --> F
    
        F -->|Yes| G[Online Detection]
        F -->|No| H[Offline Detection]
    
        G --> I[手法選択]
        H --> I
    
        style A fill:#7b2cbf,color:#fff
        style C fill:#e8f5e9
        style D fill:#fff3e0
        style E fill:#ffebee
        style G fill:#e3f2fd
        style H fill:#f3e5f5
    ```

### 実例：3つのアプローチの比較
    
    
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import classification_report, accuracy_score
    
    # データ生成（不均衡データ）
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                               n_redundant=2, n_classes=2, weights=[0.95, 0.05],
                               flip_y=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("=== データ分布 ===")
    print(f"訓練データ: {len(y_train)}個 (正常: {(y_train==0).sum()}, 異常: {(y_train==1).sum()})")
    print(f"テストデータ: {len(y_test)}個 (正常: {(y_test==0).sum()}, 異常: {(y_test==1).sum()})")
    
    # 1. 教師あり学習（ラベルあり）
    print("\n=== 1. 教師あり学習 (Supervised) ===")
    clf_supervised = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_supervised.fit(X_train, y_train)
    y_pred_supervised = clf_supervised.predict(X_test)
    acc_supervised = accuracy_score(y_test, y_pred_supervised)
    print(f"精度: {acc_supervised:.3f}")
    print(classification_report(y_test, y_pred_supervised, target_names=['正常', '異常']))
    
    # 2. 半教師あり学習（正常データのみで訓練）
    print("\n=== 2. 半教師あり学習 (Semi-supervised / Novelty Detection) ===")
    X_train_normal = X_train[y_train == 0]  # 正常データのみ
    clf_novelty = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    clf_novelty.fit(X_train_normal)
    y_pred_novelty = clf_novelty.predict(X_test)
    # One-Class SVMの出力: 1=正常, -1=異常 → 0=正常, 1=異常に変換
    y_pred_novelty = (y_pred_novelty == -1).astype(int)
    acc_novelty = accuracy_score(y_test, y_pred_novelty)
    print(f"精度: {acc_novelty:.3f}")
    print(classification_report(y_test, y_pred_novelty, target_names=['正常', '異常']))
    
    # 3. 教師なし学習（ラベルなし）
    print("\n=== 3. 教師なし学習 (Unsupervised / Outlier Detection) ===")
    clf_unsupervised = IsolationForest(contamination=0.05, random_state=42)
    clf_unsupervised.fit(X_train)
    y_pred_unsupervised = clf_unsupervised.predict(X_test)
    # Isolation Forestの出力: 1=正常, -1=異常 → 0=正常, 1=異常に変換
    y_pred_unsupervised = (y_pred_unsupervised == -1).astype(int)
    acc_unsupervised = accuracy_score(y_test, y_pred_unsupervised)
    print(f"精度: {acc_unsupervised:.3f}")
    print(classification_report(y_test, y_pred_unsupervised, target_names=['正常', '異常']))
    
    # 比較サマリ
    print("\n=== 精度比較 ===")
    print(f"教師あり学習:   {acc_supervised:.3f}")
    print(f"半教師あり学習: {acc_novelty:.3f}")
    print(f"教師なし学習:   {acc_unsupervised:.3f}")
    

> **重要** : 教師あり学習が最も高精度ですが、ラベル付きデータが必要です。実際のビジネスでは、ラベル取得コストを考慮して手法を選択します。

* * *

## 1.3 評価指標

### クラス不均衡問題

異常検知では、正常データが圧倒的に多いため、精度（Accuracy）だけでは不十分です。
    
    
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # 例：正常95%, 異常5%のデータ
    y_true = np.array([0]*95 + [1]*5)
    
    # ダメな予測器：全て正常と予測
    y_pred_bad = np.array([0]*100)
    
    # 良い予測器：異常を正しく検出
    y_pred_good = np.concatenate([np.array([0]*95), np.array([1]*5)])
    
    print("=== ダメな予測器（全て正常と予測）===")
    print(f"精度: {accuracy_score(y_true, y_pred_bad):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred_bad, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred_bad, zero_division=0):.3f}")
    print(f"F1: {f1_score(y_true, y_pred_bad, zero_division=0):.3f}")
    
    print("\n=== 良い予測器（異常を正しく検出）===")
    print(f"精度: {accuracy_score(y_true, y_pred_good):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred_good):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred_good):.3f}")
    print(f"F1: {f1_score(y_true, y_pred_good):.3f}")
    

**出力** ：
    
    
    === ダメな予測器（全て正常と予測）===
    精度: 0.950
    Precision: 0.000
    Recall: 0.000
    F1: 0.000
    
    === 良い予測器（異常を正しく検出）===
    精度: 1.000
    Precision: 1.000
    Recall: 1.000
    F1: 1.000
    

> **教訓** : 精度95%でも、異常を1つも検出できていない場合があります。

### 混同行列と主要指標

指標 | 計算式 | 意味  
---|---|---  
**Precision（適合率）** | $\frac{TP}{TP + FP}$ | 異常と予測したうち実際に異常の割合  
**Recall（再現率）** | $\frac{TP}{TP + FN}$ | 実際の異常のうち検出できた割合  
**F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | PrecisionとRecallの調和平均  
**ROC-AUC** | ROC曲線下の面積 | 閾値に依存しない総合性能  
**PR-AUC** | PR曲線下の面積 | 不均衡データでのROC-AUCより適切  
  
### ROC-AUC vs PR-AUC
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    # 不均衡データの生成
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.95, 0.05],
                               random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # モデルの訓練
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 予測確率の取得
    y_scores = clf.predict_proba(X_test)[:, 1]
    
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR曲線
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC曲線
    axes[0].plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC曲線 (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                 label='ランダム予測')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC曲線', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # PR曲線
    axes[1].plot(recall, precision, color='green', lw=2,
                 label=f'PR曲線 (AUC = {pr_auc:.3f})')
    baseline = (y_test == 1).sum() / len(y_test)
    axes[1].axhline(y=baseline, color='gray', lw=1, linestyle='--',
                    label=f'ベースライン ({baseline:.3f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall曲線', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 評価指標 ===")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"異常データの割合: {baseline:.3f}")
    

> **重要** : 不均衡データでは、PR-AUCがROC-AUCより適切な指標です。ROC-AUCは正常データが多いため楽観的になりがちです。

### ドメイン特化評価指標

ドメイン | 重視する指標 | 理由  
---|---|---  
**医療診断** | Recall（高） | 見逃しを最小化（FN削減）  
**スパムフィルタ** | Precision（高） | 誤検出を最小化（FP削減）  
**不正検知** | F1, PR-AUC | バランス重視  
**予防保守** | Recall（高） | 故障の見逃し防止  
  
* * *

## 1.4 データセットと可視化

### Synthetic Datasets（合成データ）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs, make_moons
    from scipy.stats import multivariate_normal
    
    np.random.seed(42)
    
    # データセット1: ガウス分布
    X_gaussian, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0,
                               center_box=(0, 0), random_state=42)
    outliers_gaussian = np.random.uniform(low=-8, high=8, size=(15, 2))
    X1 = np.vstack([X_gaussian, outliers_gaussian])
    y1 = np.array([0]*300 + [1]*15)
    
    # データセット2: 三日月型
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    outliers_moons = np.random.uniform(low=-2, high=3, size=(15, 2))
    X2 = np.vstack([X_moons, outliers_moons])
    y2 = np.array([0]*300 + [1]*15)
    
    # データセット3: ドーナツ型
    theta = np.linspace(0, 2*np.pi, 300)
    r = 3 + np.random.normal(0, 0.3, 300)
    X_donut = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    outliers_donut = np.random.normal(0, 1, size=(15, 2))
    X3 = np.vstack([X_donut, outliers_donut])
    y3 = np.array([0]*300 + [1]*15)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    datasets = [
        (X1, y1, 'ガウス分布'),
        (X2, y2, '三日月型'),
        (X3, y3, 'ドーナツ型')
    ]
    
    for ax, (X, y, title) in zip(axes, datasets):
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6,
                   s=50, label='正常', edgecolors='black')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.9,
                   s=150, marker='X', label='異常', edgecolors='black', linewidths=2)
        ax.set_xlabel('特徴量 1', fontsize=11)
        ax.set_ylabel('特徴量 2', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Synthetic Datasetsの特徴 ===")
    print("1. ガウス分布: 線形分離可能、統計的手法に適合")
    print("2. 三日月型: 非線形パターン、境界が複雑")
    print("3. ドーナツ型: 密度ベースの手法が有効")
    

### Real-world Datasets

データセット | ドメイン | サンプル数 | 異常率  
---|---|---|---  
**Credit Card Fraud** | 金融 | 284,807 | 0.17%  
**KDD Cup 99** | ネットワーク | 4,898,431 | 19.7%  
**MNIST (異常検知版)** | 画像 | 70,000 | 可変  
**Thyroid Disease** | 医療 | 3,772 | 2.5%  
**NASA Bearing** | 製造 | 時系列 | 可変  
  
### 可視化手法

#### 1\. 次元削減による可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # 高次元データの生成（20次元）
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.95, 0.05],
                               random_state=42)
    
    # PCAによる次元削減
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # t-SNEによる次元削減
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA
    axes[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', alpha=0.6,
                    s=50, label='正常', edgecolors='black')
    axes[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', alpha=0.9,
                    s=150, marker='X', label='異常', edgecolors='black', linewidths=2)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    axes[0].set_title('PCA可視化', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    axes[1].scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='blue', alpha=0.6,
                    s=50, label='正常', edgecolors='black')
    axes[1].scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='red', alpha=0.9,
                    s=150, marker='X', label='異常', edgecolors='black', linewidths=2)
    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title('t-SNE可視化', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 次元削減の比較 ===")
    print(f"PCA累積寄与率（2成分）: {pca.explained_variance_ratio_.sum():.2%}")
    print("t-SNE: 非線形構造の保持に優れる（局所構造重視）")
    

#### 2\. 異常スコアの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs
    
    # データ生成
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0,
                             center_box=(0, 0), random_state=42)
    X_outliers = np.random.uniform(low=-8, high=8, size=(15, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # Isolation Forestで異常スコア計算
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)
    anomaly_scores = -clf.score_samples(X)  # 負の値を正に変換
    
    # グリッド上でスコアを計算（ヒートマップ用）
    xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    Z = -clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ヒートマップ
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.7)
    axes[0].scatter(X[:, 0], X[:, 1], c=anomaly_scores, cmap='RdYlBu_r',
                    s=50, edgecolors='black', linewidths=1)
    plt.colorbar(contour, ax=axes[0], label='異常スコア')
    axes[0].set_xlabel('特徴量 1', fontsize=12)
    axes[0].set_ylabel('特徴量 2', fontsize=12)
    axes[0].set_title('異常スコアのヒートマップ', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # ヒストグラム
    axes[1].hist(anomaly_scores, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=np.percentile(anomaly_scores, 95), color='red',
                    linestyle='--', linewidth=2, label='95%点（閾値）')
    axes[1].set_xlabel('異常スコア', fontsize=12)
    axes[1].set_ylabel('頻度', fontsize=12)
    axes[1].set_title('異常スコアの分布', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 異常スコア統計 ===")
    print(f"最小値: {anomaly_scores.min():.3f}")
    print(f"最大値: {anomaly_scores.max():.3f}")
    print(f"平均: {anomaly_scores.mean():.3f}")
    print(f"95%点（閾値候補）: {np.percentile(anomaly_scores, 95):.3f}")
    

* * *

## 1.5 異常検知の課題

### 1\. Label Scarcity（ラベル不足）

**問題** : 異常データのラベル付けは高コストで困難

**対処法** :

  * 教師なし学習（Isolation Forest, LOF）
  * 半教師あり学習（One-Class SVM, Autoencoder）
  * Active Learning（重要サンプルのみラベル付け）
  * Weak Supervision（ノイズラベルの活用）

### 2\. High Dimensionality（高次元性）

**問題** : 次元の呪い（距離が意味を失う）

**対処法** :

  * 次元削減（PCA, Autoencoder）
  * 特徴選択（重要特徴量のみ使用）
  * 部分空間法（Subspace methods）

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    
    # 次元の呪いの実験
    dimensions = [2, 5, 10, 20, 50, 100, 200]
    avg_distances = []
    
    np.random.seed(42)
    for d in dimensions:
        # ランダムな点を生成
        X = np.random.uniform(0, 1, size=(100, d))
        # 全ペア間の距離を計算
        distances = pdist(X, metric='euclidean')
        avg_distances.append(distances.mean())
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, avg_distances, marker='o', linewidth=2, markersize=8)
    plt.xlabel('次元数', fontsize=12)
    plt.ylabel('平均ユークリッド距離', fontsize=12)
    plt.title('次元の呪い：次元数と距離の関係', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== 次元の呪い ===")
    for d, dist in zip(dimensions, avg_distances):
        print(f"次元数 {d:3d}: 平均距離 = {dist:.3f}")
    print("\n→ 高次元では全ての点が等距離に見える（距離が意味を失う）")
    

### 3\. Concept Drift（概念ドリフト）

**問題** : 時間とともに正常パターンが変化

**対処法** :

  * Online Learning（逐次更新）
  * Sliding Window（直近データで再訓練）
  * Ensemble Methods（複数時期のモデル）
  * Adaptive Thresholds（閾値の動的調整）

### 4\. Interpretability（解釈可能性）

**問題** : なぜ異常と判定されたか説明が困難

**対処法** :

  * Rule-based methods（ルールベース手法）
  * Feature importance（特徴量重要度）
  * SHAP values（Shapley値による説明）
  * Attention mechanisms（注目メカニズム）

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.tree import DecisionTreeClassifier
    
    # サンプルデータ
    np.random.seed(42)
    X_normal = np.random.normal(0, 1, size=(100, 5))
    X_anomaly = np.random.normal(5, 1, size=(5, 5))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0]*100 + [1]*5)
    
    # Isolation Forestで異常検知
    clf_if = IsolationForest(contamination=0.05, random_state=42)
    clf_if.fit(X)
    predictions = clf_if.predict(X)
    
    # 異常サンプルの特徴量重要度を分析
    # 簡易的に各特徴量の偏差を計算
    X_mean = X_normal.mean(axis=0)
    X_std = X_normal.std(axis=0)
    
    anomaly_idx = np.where(predictions == -1)[0][:3]  # 最初の3つの異常サンプル
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, idx in enumerate(anomaly_idx):
        deviations = np.abs((X[idx] - X_mean) / X_std)
        axes[i].bar(range(5), deviations, color='steelblue', edgecolor='black')
        axes[i].axhline(y=2, color='red', linestyle='--', linewidth=2, label='2σ')
        axes[i].set_xlabel('特徴量', fontsize=11)
        axes[i].set_ylabel('標準偏差', fontsize=11)
        axes[i].set_title(f'異常サンプル {idx}の偏差', fontsize=12, fontweight='bold')
        axes[i].set_xticks(range(5))
        axes[i].set_xticklabels([f'F{j}' for j in range(5)])
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== 解釈可能性の例 ===")
    for i, idx in enumerate(anomaly_idx):
        deviations = np.abs((X[idx] - X_mean) / X_std)
        max_dev_feature = deviations.argmax()
        print(f"異常サンプル {idx}: 特徴量 {max_dev_feature} が最も異常 ({deviations[max_dev_feature]:.2f}σ)")
    

### 課題の優先順位

課題 | 影響度 | 対処難易度 | 優先度  
---|---|---|---  
**Label Scarcity** | 高 | 中 | 高  
**Concept Drift** | 高 | 高 | 高  
**High Dimensionality** | 中 | 中 | 中  
**Interpretability** | 中 | 高 | 中  
  
* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **異常検知の基本**

     * Point, Contextual, Collective異常の3タイプ
     * 不正検知、製造、医療、ITシステムへの応用
     * ビジネス価値の理解
  2. **タスク分類**

     * 教師あり・半教師あり・教師なし学習
     * Novelty Detection vs Outlier Detection
     * Online vs Offline Detection
  3. **評価指標**

     * Precision, Recall, F1の使い分け
     * ROC-AUC vs PR-AUC
     * クラス不均衡問題への対処
  4. **データセットと可視化**

     * Syntheticデータでの検証
     * Real-worldデータの特性
     * PCA, t-SNE, 異常スコアの可視化
  5. **課題と対処法**

     * Label Scarcity: 教師なし・半教師あり学習
     * High Dimensionality: 次元削減
     * Concept Drift: Online Learning
     * Interpretability: SHAP, Feature Importance

### 異常検知の原則

原則 | 説明  
---|---  
**ドメイン知識活用** | 業務知識を手法選択と閾値設定に反映  
**適切な評価指標** | 不均衡データではPR-AUC, F1を優先  
**継続的監視** | Concept Driftに対応する再訓練  
**解釈可能性重視** | 本番運用には説明可能なモデルが必要  
**コスト意識** | FP, FNのビジネスコストを考慮  
  
### 次の章へ

第2章では、**統計的異常検知** を学びます：

  * Z-score, Grubbs Test
  * Gaussian Mixture Models
  * Statistical Process Control
  * Bayesian Anomaly Detection
  * 時系列データへの応用

* * *

## 演習問題

### 問題1（難易度：easy）

Point Anomaly, Contextual Anomaly, Collective Anomalyの違いを説明し、それぞれの具体例を挙げてください。

解答例

**解答** ：

  1. **Point Anomaly（点異常）**

     * 説明: 個別のデータポイントが、他の全てのデータと大きく異なる
     * 例: クレジットカードで突然10万円の高額決済が発生
  2. **Contextual Anomaly（文脈的異常）**

     * 説明: 特定の文脈（時間、場所など）においてのみ異常とみなされる
     * 例: 気温35℃は夏は正常だが、冬は異常。午前3時のオフィスビルへのアクセスは異常
  3. **Collective Anomaly（集団異常）**

     * 説明: 個々のデータは正常だが、集合として異常なパターンを形成
     * 例: 心電図の一時的な異常波形の連続、Webサーバーへの分散DoS攻撃

### 問題2（難易度：medium）

以下のシナリオに対して、適切な異常検知のタスク設定（教師あり/半教師あり/教師なし）を選択し、理由を述べてください。

**シナリオ** : 製造ラインで製品画像から不良品を検出したい。正常品の画像は大量にあるが、不良品の画像は数枚しかない。

解答例

**解答** ：

**推奨タスク** : **半教師あり学習（Novelty Detection）**

**理由** ：

  1. **ラベル状況**

     * 正常品の画像は大量にある（ラベル付き）
     * 不良品の画像は数枚（教師あり学習には不十分）
  2. **タスクの性質**

     * 正常品のパターンを学習し、それから逸脱するものを異常とする
     * Novelty Detection（新規性検知）の典型的なユースケース
  3. **具体的手法**

     * One-Class SVM: 正常データのみで境界を学習
     * Autoencoder: 正常画像の再構成誤差で異常判定
     * Deep SVDD: 深層学習による正常データの超球面表現

**教師あり学習が不適な理由** :

  * 不良品のサンプルが少なすぎる（数枚では汎化困難）
  * 不良品の種類が未知（訓練データに含まれない不良パターンを検出できない）

**教師なし学習が不適な理由** :

  * 正常品のラベルがあるのに活用しないのは非効率
  * 半教師あり学習の方が高精度

### 問題3（難易度：medium）

異常検知において、なぜAccuracy（精度）だけでは評価が不十分なのか説明し、代わりに使うべき指標を提案してください。

解答例

**解答** ：

**Accuracyが不十分な理由** ：

異常検知では、正常データが圧倒的多数（95-99%）を占めるクラス不均衡問題があります。

**具体例** :

  * データ: 正常95%, 異常5%
  * 予測器A: 全てを正常と予測 → Accuracy = 95%（異常を1つも検出できていない）
  * 予測器B: 異常を全て正しく検出 → Accuracy = 100%

予測器Aは役に立たないのに、Accuracyでは95%の高評価を得てしまいます。

**推奨する評価指標** ：

指標 | 推奨理由 | 使用場面  
---|---|---  
**PR-AUC** | 不均衡データに適切、閾値非依存 | 総合評価  
**F1 Score** | Precision/Recallのバランス | 単一閾値での評価  
**Recall** | 異常の見逃し最小化 | 医療、予防保守  
**Precision** | 誤検出最小化 | スパムフィルタ  
  
**計算式** :

  * Precision = TP / (TP + FP): 異常と予測したうち実際に異常の割合
  * Recall = TP / (TP + FN): 実際の異常のうち検出できた割合
  * F1 = 2 × (Precision × Recall) / (Precision + Recall)

### 問題4（難易度：hard）

次元の呪いが異常検知に与える影響を説明し、対処法を3つ挙げてください。Pythonコードで次元数と距離の関係を示してください。

解答例

**解答** ：

**次元の呪いの影響** ：

高次元空間では、全てのデータポイント間の距離が似通ってくるため、距離ベースの異常検知手法（KNN, LOFなど）が機能しなくなります。

**具体的な問題** :

  1. 最近傍と最遠点の距離が収束する
  2. 異常データと正常データの距離差が小さくなる
  3. ユークリッド距離が意味を失う

**対処法** ：

  1. **次元削減**

     * PCA: 主成分分析で重要な軸のみ残す
     * Autoencoder: 非線形な次元削減
     * t-SNE/UMAP: 可視化と構造保持
  2. **特徴選択**

     * Mutual Information: 異常検知に寄与する特徴量を選択
     * L1正則化: 不要な特徴量の重みを0にする
     * ドメイン知識: 専門家による特徴量選定
  3. **部分空間法**

     * Subspace methods: 複数の低次元部分空間で異常検知
     * Random Projection: ランダムな低次元射影を複数使用

**Pythonコード** ：
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    
    # 次元の呪いの実験
    np.random.seed(42)
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]
    results = []
    
    for d in dimensions:
        # 均一分布からランダムな点を生成
        X = np.random.uniform(0, 1, size=(100, d))
    
        # 全ペア間の距離を計算
        distances = pdist(X, metric='euclidean')
    
        # 統計を記録
        results.append({
            'dim': d,
            'mean': distances.mean(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max()
        })
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 平均距離と標準偏差
    dims = [r['dim'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    
    axes[0].plot(dims, means, marker='o', linewidth=2, markersize=8, label='平均距離')
    axes[0].fill_between(dims,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          alpha=0.3, label='±1σ')
    axes[0].set_xlabel('次元数', fontsize=12)
    axes[0].set_ylabel('ユークリッド距離', fontsize=12)
    axes[0].set_title('次元数と距離の関係', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 最小距離と最大距離の比率
    ratios = [r['min'] / r['max'] for r in results]
    axes[1].plot(dims, ratios, marker='s', linewidth=2, markersize=8, color='red')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', label='完全一致')
    axes[1].set_xlabel('次元数', fontsize=12)
    axes[1].set_ylabel('最小距離 / 最大距離', fontsize=12)
    axes[1].set_title('距離の相対差の消失', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 次元の呪い：距離の統計 ===")
    for r in results:
        print(f"次元数 {r['dim']:3d}: 平均={r['mean']:.3f}, "
              f"標準偏差={r['std']:.3f}, 最小/最大比={r['min']/r['max']:.3f}")
    
    print("\n→ 次元数が増えると:")
    print("  1. 平均距離が増加（スケール効果）")
    print("  2. 相対的な距離差が縮小（最小/最大比が1に近づく）")
    print("  3. 全ての点が等距離に見える（異常検知が困難）")
    

**結論** :

  * 高次元では距離の差異が小さくなり、異常検知の精度が低下
  * 次元削減または特徴選択により、意味のある距離を保持
  * ドメイン知識を活用した特徴量エンジニアリングが重要

### 問題5（難易度：hard）

Concept Drift（概念ドリフト）が異常検知に与える影響を説明し、Online Learningによる対処法を示してください。時系列データでの簡単な実装例も含めてください。

解答例

**解答** ：

**Concept Driftの影響** ：

Concept Driftとは、時間とともに正常データの分布が変化する現象です。これにより、過去のデータで訓練したモデルが現在のデータに適合しなくなります。

**具体例** :

  * Eコマース: 季節変動（夏と冬で購買パターンが変化）
  * 製造業: 設備の経年劣化で正常な振動パターンが変化
  * ネットワーク: トラフィックパターンの進化

**問題点** :

  1. 過去のモデルが古くなり、誤検出（FP）が増加
  2. 新しい正常パターンを異常と誤判定
  3. 検知性能の経時的な劣化

**Online Learningによる対処法** ：

  1. **Sliding Window Approach**

     * 直近Nサンプルのみでモデルを再訓練
     * 古いデータを破棄し、新しいパターンに適応
  2. **Incremental Learning**

     * 新しいデータでモデルを逐次更新
     * 全データを再訓練せず効率的
  3. **Adaptive Thresholds**

     * 異常判定の閾値を動的に調整
     * 最近のデータ分布に基づく閾値更新

**実装例（Sliding Window）** ：
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    
    # 時系列データの生成（概念ドリフトあり）
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    
    # 正常データ: 平均が時間とともに変化（Concept Drift）
    mean_shift = time / 200  # 徐々に平均が増加
    X = np.random.normal(loc=mean_shift, scale=1.0, size=(n_samples, 5))
    
    # 異常データを一部追加
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    X[anomaly_indices] += np.random.uniform(5, 10, size=(50, 5))
    
    # 真のラベル
    y_true = np.zeros(n_samples)
    y_true[anomaly_indices] = 1
    
    # 1. 静的モデル（初期データのみで訓練）
    print("=== 1. 静的モデル（Concept Drift未対応）===")
    static_model = IsolationForest(contamination=0.05, random_state=42)
    static_model.fit(X[:200])  # 初期200サンプルのみ
    
    static_predictions = static_model.predict(X)
    static_predictions = (static_predictions == -1).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    static_precision = precision_score(y_true, static_predictions)
    static_recall = recall_score(y_true, static_predictions)
    static_f1 = f1_score(y_true, static_predictions)
    
    print(f"Precision: {static_precision:.3f}")
    print(f"Recall: {static_recall:.3f}")
    print(f"F1 Score: {static_f1:.3f}")
    
    # 2. Online Learning（Sliding Window）
    print("\n=== 2. Online Learning（Sliding Window, window=200）===")
    window_size = 200
    online_predictions = np.zeros(n_samples)
    
    for i in range(window_size, n_samples):
        # 直近window_sizeサンプルでモデルを訓練
        window_data = X[i-window_size:i]
        online_model = IsolationForest(contamination=0.05, random_state=42)
        online_model.fit(window_data)
    
        # 現在のサンプルを予測
        pred = online_model.predict(X[i:i+1])
        online_predictions[i] = (pred == -1).astype(int)
    
    online_precision = precision_score(y_true[window_size:], online_predictions[window_size:])
    online_recall = recall_score(y_true[window_size:], online_predictions[window_size:])
    online_f1 = f1_score(y_true[window_size:], online_predictions[window_size:])
    
    print(f"Precision: {online_precision:.3f}")
    print(f"Recall: {online_recall:.3f}")
    print(f"F1 Score: {online_f1:.3f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # データの平均の変化（Concept Drift）
    axes[0].plot(time, X.mean(axis=1), alpha=0.7, label='データ平均')
    axes[0].scatter(anomaly_indices, X[anomaly_indices].mean(axis=1),
                    c='red', s=50, marker='X', label='異常データ', zorder=5)
    axes[0].set_xlabel('時刻', fontsize=12)
    axes[0].set_ylabel('平均値', fontsize=12)
    axes[0].set_title('Concept Drift: 時間とともに正常データの分布が変化',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 予測の比較
    axes[1].scatter(time, static_predictions, alpha=0.5, label='静的モデル', s=10)
    axes[1].scatter(time, online_predictions, alpha=0.5, label='Online Learning', s=10)
    axes[1].scatter(anomaly_indices, y_true[anomaly_indices],
                    c='red', marker='X', s=100, label='真の異常', zorder=5, edgecolors='black')
    axes[1].set_xlabel('時刻', fontsize=12)
    axes[1].set_ylabel('異常フラグ', fontsize=12)
    axes[1].set_title('静的モデル vs Online Learning', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 性能比較 ===")
    print(f"静的モデル:      F1={static_f1:.3f}")
    print(f"Online Learning: F1={online_f1:.3f}")
    print(f"改善: {(online_f1 - static_f1):.3f}")
    

**結論** :

  * Concept Driftがある環境では、静的モデルは性能が劣化
  * Sliding Windowによるオンライン学習で、新しいパターンに適応
  * ウィンドウサイズは、安定性（大）と適応速度（小）のトレードオフ

* * *

## 参考文献

  1. Chandola, V., Banerjee, A., & Kumar, V. (2009). _Anomaly detection: A survey_. ACM computing surveys (CSUR), 41(3), 1-58.
  2. Aggarwal, C. C. (2017). _Outlier analysis_ (2nd ed.). Springer.
  3. Goldstein, M., & Uchida, S. (2016). _A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data_. PloS one, 11(4), e0152173.
  4. Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021). _Deep learning for anomaly detection: A review_. ACM Computing Surveys (CSUR), 54(2), 1-38.
  5. Rousseeuw, P. J., & Hubert, M. (2011). _Robust statistics for outlier detection_. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 1(1), 73-79.
