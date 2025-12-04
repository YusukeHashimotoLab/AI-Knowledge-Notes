---
title: 第2章：分類問題の基礎
chapter_title: 第2章：分類問題の基礎
subtitle: カテゴリ予測の理論と実装 - ロジスティック回帰から決定木・SVMまで
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 分類問題の定義と応用例を理解する
  * ✅ ロジスティック回帰の理論と実装ができる
  * ✅ シグモイド関数と確率解釈を説明できる
  * ✅ 決定木の仕組みと実装ができる
  * ✅ k-NN、SVMを適用できる
  * ✅ 混同行列、精度、再現率、F1スコアで評価できる
  * ✅ ROC曲線とAUCを理解し使いこなせる

* * *

## 2.1 分類問題とは

### 定義

**分類問題（Classification）** は、入力変数から**離散値（カテゴリ）** の出力を予測する教師あり学習のタスクです。

> 「特徴量 $X$ から離散的なクラスラベル $y \in \\{1, 2, ..., K\\}$ を予測する関数 $f: X \rightarrow y$ を学習する」

### 分類のタイプ

タイプ | クラス数 | 例  
---|---|---  
**二値分類** | 2クラス | スパム判定、疾病診断、顧客離反予測  
**多クラス分類** | 3+クラス | 手書き数字認識、画像分類、感情分析  
**多ラベル分類** | 複数ラベル | タグ付け、遺伝子機能予測  
  
### 実世界の応用例
    
    
    ```mermaid
    graph LR
        A[分類問題の応用] --> B[医療: 疾病診断]
        A --> C[金融: クレジット審査]
        A --> D[マーケティング: 顧客セグメント]
        A --> E[セキュリティ: 不正検知]
        A --> F[画像: 物体認識]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 2.2 ロジスティック回帰（Logistic Regression）

### 概要

**ロジスティック回帰** は、二値分類に使われる線形モデルです。線形回帰にシグモイド関数を適用して確率を出力します。

### シグモイド関数

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

特徴：

  * 出力範囲: $[0, 1]$（確率として解釈可能）
  * $z = 0$ で $\sigma(z) = 0.5$
  * 滑らかな S字カーブ

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # シグモイド関数
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # 可視化
    z = np.linspace(-10, 10, 100)
    y = sigmoid(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, y, linewidth=2, label='σ(z)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='閾値 0.5')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('z = w^T x', fontsize=12)
    plt.ylabel('σ(z)', fontsize=12)
    plt.title('シグモイド関数', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    

### モデルの定義

$$ P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}} $$

予測：

$$ \hat{y} = \begin{cases} 1 & \text{if } P(y=1 | \mathbf{x}) \geq 0.5 \\\ 0 & \text{otherwise} \end{cases} $$

### 損失関数：交差エントロピー

$$ J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] $$

### 実装例
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ロジスティック回帰
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("=== ロジスティック回帰 ===")
    print(f"精度: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n重み: {model.coef_[0]}")
    print(f"切片: {model.intercept_[0]:.4f}")
    
    # 決定境界の可視化
    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o',
                    edgecolors='k', s=80, label='クラス 0')
        plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s',
                    edgecolors='k', s=80, label='クラス 1')
        plt.xlabel('特徴量 1', fontsize=12)
        plt.ylabel('特徴量 2', fontsize=12)
        plt.title('ロジスティック回帰の決定境界', fontsize=14)
        plt.legend()
        plt.show()
    
    plot_decision_boundary(model, X_test, y_test)
    

**出力** ：
    
    
    === ロジスティック回帰 ===
    精度: 0.9550
    
    重み: [2.14532851 1.87653214]
    切片: -0.2341
    

* * *

## 2.3 決定木（Decision Tree）

### 概要

**決定木** は、if-then-elseルールの階層構造で分類を行います。特徴量を基準に再帰的にデータを分割します。
    
    
    ```mermaid
    graph TD
        A[特徴量1 <= 0.5] -->|Yes| B[特徴量2 <= 1.2]
        A -->|No| C[クラス 1]
        B -->|Yes| D[クラス 0]
        B -->|No| E[クラス 1]
    
        style A fill:#fff3e0
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#e3f2fd
        style E fill:#e8f5e9
    ```

### 分割基準

**1\. ジニ不純度（Gini Impurity）** ：

$$ \text{Gini}(S) = 1 - \sum_{i=1}^{K} p_i^2 $$

  * $p_i$: クラス $i$ の割合
  * 値が小さいほど純粋（一つのクラスに偏っている）

**2\. エントロピー（Entropy）** ：

$$ \text{Entropy}(S) = -\sum_{i=1}^{K} p_i \log_2(p_i) $$

### 実装例
    
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    
    # 決定木モデル
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 予測
    y_pred_dt = dt_model.predict(X_test)
    
    print("=== 決定木 ===")
    print(f"精度: {accuracy_score(y_test, y_pred_dt):.4f}")
    
    # 決定木の可視化
    plt.figure(figsize=(16, 10))
    plot_tree(dt_model, filled=True, feature_names=['特徴量1', '特徴量2'],
              class_names=['クラス0', 'クラス1'], fontsize=10)
    plt.title('決定木の構造', fontsize=16)
    plt.show()
    
    # 決定境界
    plot_decision_boundary(dt_model, X_test, y_test)
    

**出力** ：
    
    
    === 決定木 ===
    精度: 0.9450
    

### 特徴量重要度
    
    
    # 特徴量重要度
    importances = dt_model.feature_importances_
    
    plt.figure(figsize=(8, 6))
    plt.bar(['特徴量1', '特徴量2'], importances, color=['#3498db', '#e74c3c'])
    plt.ylabel('重要度', fontsize=12)
    plt.title('特徴量重要度', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print(f"\n特徴量1の重要度: {importances[0]:.4f}")
    print(f"特徴量2の重要度: {importances[1]:.4f}")
    

* * *

## 2.4 k-近傍法（k-Nearest Neighbors, k-NN）

### 概要

**k-NN** は、最も近い $k$ 個の訓練データの多数決で分類します。

### アルゴリズム

  1. テストデータ $\mathbf{x}$ に対して、訓練データとの距離を計算
  2. 最も近い $k$ 個のデータを選択
  3. 多数決で最も多いクラスを予測

### 距離の種類

距離 | 数式  
---|---  
**ユークリッド距離** | $\sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$  
**マンハッタン距離** | $\sum_{i=1}^{n} |x_i - y_i|$  
**ミンコフスキー距離** | $\left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$  
  
### 実装例
    
    
    from sklearn.neighbors import KNeighborsClassifier
    
    # k-NNモデル
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    # 予測
    y_pred_knn = knn_model.predict(X_test)
    
    print("=== k-NN (k=5) ===")
    print(f"精度: {accuracy_score(y_test, y_pred_knn):.4f}")
    
    # 決定境界
    plot_decision_boundary(knn_model, X_test, y_test)
    

**出力** ：
    
    
    === k-NN (k=5) ===
    精度: 0.9400
    

### kの選択
    
    
    # 異なるkでの精度比較
    k_range = range(1, 31)
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
    
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_scores, 'o-', label='訓練データ', linewidth=2)
    plt.plot(k_range, test_scores, 's-', label='テストデータ', linewidth=2)
    plt.xlabel('k (近傍数)', fontsize=12)
    plt.ylabel('精度', fontsize=12)
    plt.title('k-NN: kの値と精度の関係', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_k = k_range[np.argmax(test_scores)]
    print(f"\n最適なk: {best_k}")
    print(f"最高精度: {max(test_scores):.4f}")
    

* * *

## 2.5 サポートベクターマシン（SVM）

### 概要

**SVM** は、マージンを最大化する決定境界を見つけます。
    
    
    ```mermaid
    graph LR
        A[SVM] --> B[線形SVM]
        A --> C[非線形SVM カーネル法]
    
        B --> B1[線形分離可能なデータ]
        C --> C1[RBFカーネル]
        C --> C2[多項式カーネル]
        C --> C3[シグモイドカーネル]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

### マージン最大化

$$ \text{maximize} \quad \frac{2}{||\mathbf{w}||} \quad \text{subject to} \quad y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 $$

### カーネルトリック

**RBF（ガウシアン）カーネル** ：

$$ K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\sigma^2}\right) $$

### 実装例
    
    
    from sklearn.svm import SVC
    
    # 線形SVM
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    
    # RBF SVM
    svm_rbf = SVC(kernel='rbf', gamma='auto')
    svm_rbf.fit(X_train, y_train)
    
    print("=== SVM (線形カーネル) ===")
    print(f"精度: {svm_linear.score(X_test, y_test):.4f}")
    
    print("\n=== SVM (RBFカーネル) ===")
    print(f"精度: {svm_rbf.score(X_test, y_test):.4f}")
    
    # 決定境界の比較
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, model, title in zip(axes, [svm_linear, svm_rbf],
                                ['線形SVM', 'RBF SVM']):
        h = 0.02
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                  c='blue', marker='o', edgecolors='k', s=80, label='クラス 0')
        ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                  c='red', marker='s', edgecolors='k', s=80, label='クラス 1')
        ax.set_xlabel('特徴量 1')
        ax.set_ylabel('特徴量 2')
        ax.set_title(title, fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === SVM (線形カーネル) ===
    精度: 0.9550
    
    === SVM (RBFカーネル) ===
    精度: 0.9650
    

* * *

## 2.6 分類モデルの評価

### 混同行列（Confusion Matrix）

| 予測: 陽性 | 予測: 陰性  
---|---|---  
**実際: 陽性** | TP (真陽性) | FN (偽陰性)  
**実際: 陰性** | FP (偽陽性) | TN (真陰性)  
  
### 評価指標

指標 | 数式 | 意味  
---|---|---  
**精度**  
(Accuracy) | $\frac{TP + TN}{TP + TN + FP + FN}$ | 全体の正解率  
**適合率**  
(Precision) | $\frac{TP}{TP + FP}$ | 陽性予測の正確さ  
**再現率**  
(Recall) | $\frac{TP}{TP + FN}$ | 実際の陽性を捕捉する率  
**F1スコア** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | PrecisionとRecallの調和平均  
  
### 実装例
    
    
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['クラス 0', 'クラス 1'],
                yticklabels=['クラス 0', 'クラス 1'])
    plt.xlabel('予測', fontsize=12)
    plt.ylabel('実際', fontsize=12)
    plt.title('混同行列', fontsize=14)
    plt.show()
    
    # 詳細な評価レポート
    print("\n=== 分類レポート ===")
    print(classification_report(y_test, y_pred,
                              target_names=['クラス 0', 'クラス 1']))
    

**出力** ：
    
    
    === 分類レポート ===
                  precision    recall  f1-score   support
    
        クラス 0       0.96      0.95      0.95        99
        クラス 1       0.95      0.96      0.96       101
    
        accuracy                           0.96       200
       macro avg       0.96      0.96      0.96       200
    weighted avg       0.96      0.96      0.96       200
    

### ROC曲線とAUC

**ROC（Receiver Operating Characteristic）曲線** は、閾値を変えたときのTPR（真陽性率）とFPR（偽陽性率）の関係を示します。

$$ \text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN} $$
    
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    auc = roc_auc_score(y_test, y_proba[:, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲線 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='ランダム (AUC = 0.5)')
    plt.xlabel('偽陽性率 (FPR)', fontsize=12)
    plt.ylabel('真陽性率 (TPR)', fontsize=12)
    plt.title('ROC曲線', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"AUC: {auc:.4f}")
    

**出力** ：
    
    
    AUC: 0.9876
    

> **AUC（Area Under the Curve）** : ROC曲線の下の面積。1に近いほど良いモデル。

* * *

## 2.7 本章のまとめ

### 学んだこと

  1. **分類問題の定義**

     * 離散値（カテゴリ）の予測タスク
     * 二値分類、多クラス分類、多ラベル分類
  2. **ロジスティック回帰**

     * シグモイド関数による確率出力
     * 交差エントロピー損失
  3. **決定木**

     * if-then-elseルールの階層構造
     * ジニ不純度、エントロピー
     * 特徴量重要度
  4. **k-NN**

     * 最近傍の多数決
     * kの選択の重要性
  5. **SVM**

     * マージン最大化
     * カーネルトリック
  6. **評価指標**

     * 混同行列、精度、適合率、再現率、F1スコア
     * ROC曲線とAUC

### 次の章へ

第3章では、**アンサンブル手法** を学びます：

  * Baggingの原理
  * Random Forest
  * Boosting（Gradient Boosting、XGBoost、LightGBM、CatBoost）

* * *

## 演習問題

### 問題1（難易度：easy）

精度（Accuracy）が高くても不適切な場合があります。どのような状況か説明してください。

解答例

**解答** ：

**不均衡データ（Imbalanced Data）** の場合、精度は不適切です。

**例** ：

  * がん診断データ: 陽性1%、陰性99%
  * すべて「陰性」と予測すると精度99%だが、意味がない
  * 陽性を見逃すと重大な結果に

**適切な指標** ：

  * 再現率（Recall）: 陽性を見逃さない
  * F1スコア: PrecisionとRecallのバランス
  * AUC: 閾値に依存しない評価

### 問題2（難易度：medium）

以下の混同行列から、精度、適合率、再現率、F1スコアを計算してください。

| 予測: 陽性 | 予測: 陰性  
---|---|---  
実際: 陽性 | 80 | 20  
実際: 陰性 | 10 | 90  
解答例

**解答** ：
    
    
    TP = 80, FN = 20, FP = 10, TN = 90
    
    精度 (Accuracy) = (TP + TN) / (TP + TN + FP + FN)
                    = (80 + 90) / (80 + 90 + 10 + 20)
                    = 170 / 200 = 0.85 = 85%
    
    適合率 (Precision) = TP / (TP + FP)
                       = 80 / (80 + 10)
                       = 80 / 90 = 0.8889 = 88.89%
    
    再現率 (Recall) = TP / (TP + FN)
                    = 80 / (80 + 20)
                    = 80 / 100 = 0.80 = 80%
    
    F1スコア = 2 * (Precision * Recall) / (Precision + Recall)
            = 2 * (0.8889 * 0.80) / (0.8889 + 0.80)
            = 2 * 0.7111 / 1.6889
            = 0.8421 = 84.21%
    

### 問題3（難易度：medium）

k-NNで最適なkを選ぶ際、kが小さすぎる場合と大きすぎる場合の問題を説明してください。

解答例

**kが小さすぎる場合（例: k=1）** ：

  * **過学習（Overfitting）** : ノイズに敏感
  * 訓練データの精度は高いが、テストデータの精度は低い
  * 決定境界が複雑でギザギザ

**kが大きすぎる場合（例: k=全データ数）** ：

  * **過度な単純化** : すべて多数派クラスに分類
  * 決定境界が単純すぎる
  * 訓練・テスト両方の精度が低い

**最適なk** ：

  * 交差検証で選択
  * 通常 $\sqrt{N}$ 付近（$N$はデータ数）
  * 奇数を選ぶ（二値分類の同票を避ける）

### 問題4（難易度：hard）

SVMのカーネルトリックを使う意義を、計算量の観点から説明してください。

解答例

**カーネルトリックの意義** ：

**問題** ：非線形分離可能なデータを分類するには、高次元空間に変換する必要がある。

**直接的な方法** ：

  * 特徴量を明示的に高次元に変換: $\phi(\mathbf{x})$
  * 計算量: $O(d^2)$ または $O(d^3)$（$d$は次元）
  * 次元が高いと計算不可能

**カーネルトリック** ：

  * 内積 $\langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ をカーネル関数 $K(\mathbf{x}, \mathbf{x}')$ で直接計算
  * 高次元変換を明示的に行わない
  * 計算量: $O(d)$（元の次元のまま）

**例（RBFカーネル）** ：

  * 無限次元への変換を $O(d)$ で計算
  * $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$

**結論** ：カーネルトリックにより、高次元での計算を低次元で効率的に実行可能。

### 問題5（難易度：hard）

ロジスティック回帰を実装し、irisデータセットの二値分類（setosa vs versicolor）を行ってください。

解答例
    
    
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # データ読み込み
    iris = load_iris()
    X = iris.data[iris.target != 2]  # setosa (0) と versicolor (1)のみ
    y = iris.target[iris.target != 2]
    
    # 最初の2つの特徴量のみ使用（可視化のため）
    X = X[:, :2]
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ロジスティック回帰
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # 評価
    print("=== 分類レポート ===")
    print(classification_report(y_test, y_pred,
                              target_names=['setosa', 'versicolor']))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['setosa', 'versicolor'],
                yticklabels=['setosa', 'versicolor'])
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列')
    plt.show()
    
    # 決定境界の可視化
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train_scaled[y_train==0, 0], X_train_scaled[y_train==0, 1],
               c='blue', marker='o', edgecolors='k', s=80, label='setosa')
    plt.scatter(X_train_scaled[y_train==1, 0], X_train_scaled[y_train==1, 1],
               c='red', marker='s', edgecolors='k', s=80, label='versicolor')
    plt.xlabel('Sepal length (標準化)')
    plt.ylabel('Sepal width (標準化)')
    plt.title('ロジスティック回帰の決定境界')
    plt.legend()
    plt.show()
    
    print(f"\n精度: {model.score(X_test_scaled, y_test):.4f}")
    

**出力** ：
    
    
    === 分類レポート ===
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        10
      versicolor       1.00      1.00      1.00        10
    
        accuracy                           1.00        20
       macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20
    
    精度: 1.0000
    

* * *

## 参考文献

  1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
  2. Murphy, K. P. (2012). _Machine Learning: A Probabilistic Perspective_. MIT Press.
  3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). _An Introduction to Statistical Learning_. Springer.
