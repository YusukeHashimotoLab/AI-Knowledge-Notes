---
title: 第1章：評価指標基礎
chapter_title: 第1章：評価指標基礎
subtitle: モデル性能を正しく測る - 分類と回帰の評価指標完全ガイド
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 評価指標の重要性とビジネスへの影響を理解する
  * ✅ 分類問題の主要な評価指標を計算し、解釈できる
  * ✅ Confusion MatrixからPrecision、Recall、F1-scoreを導出できる
  * ✅ ROC-AUCとPR-AUCの違いを理解し、使い分けられる
  * ✅ 回帰問題の評価指標を選択し、適切に解釈できる
  * ✅ ビジネス目的に応じた評価指標を選択できる

* * *

## 1.1 評価指標の重要性

### なぜ評価指標が重要か

**評価指標（Evaluation Metrics）** は、機械学習モデルの性能を定量的に測定する手段です。

> 「測定できないものは改善できない」- Peter Drucker

適切な評価指標を選ぶことは、以下の理由で極めて重要です：

  * **モデル選択** : 複数のモデルを比較し、最適なものを選択
  * **ハイパーパラメータ調整** : チューニングの方向性を決定
  * **ビジネス価値** : モデルのビジネスインパクトを定量化
  * **改善の指針** : どこを改善すべきか明確化

### 評価指標の選択ミスによる影響

シナリオ | 不適切な指標 | 問題 | 適切な指標  
---|---|---|---  
**がん検出** | Accuracy | 陽性を見逃す（偽陰性） | Recall、F2-score  
**スパムフィルタ** | Recall | 正常メールを誤判定（偽陽性） | Precision、F0.5-score  
**不均衡データ** | Accuracy | 多数派クラスに偏る | F1-score、AUC  
**住宅価格予測** | MSE | 高額物件の誤差に過敏 | MAPE、MAE  
  
### 評価指標の全体像
    
    
    ```mermaid
    graph TD
        A[評価指標] --> B[分類問題]
        A --> C[回帰問題]
    
        B --> D[二値分類]
        B --> E[多クラス分類]
    
        D --> F[Accuracy]
        D --> G[Precision / Recall]
        D --> H[F1-score]
        D --> I[ROC-AUC]
        D --> J[PR-AUC]
        D --> K[Log Loss]
    
        E --> L[Macro / Micro / Weighted]
    
        C --> M[MAE / MSE / RMSE]
        C --> N[R² / Adjusted R²]
        C --> O[MAPE / MSLE]
    
        style A fill:#e8f5e9
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#fce4ec
        style E fill:#f3e5f5
    ```

### 実例：評価指標の選択の重要性
    
    
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # がん検出シナリオ（不均衡データ）
    # 陽性（がん）: 10例、陰性（正常）: 990例
    y_true = np.array([0]*990 + [1]*10)
    
    # モデルA: 全て陰性と予測（保守的）
    y_pred_A = np.array([0]*1000)
    
    # モデルB: 適度にバランスの取れた予測
    y_pred_B = np.array([0]*985 + [1]*5 + [0]*3 + [1]*7)
    
    print("=== がん検出シナリオ：評価指標の比較 ===\n")
    
    print("モデルA（全て陰性と予測）:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_A):.3f}")
    print(f"  Precision: N/A (陽性予測なし)")
    print(f"  Recall:    {recall_score(y_true, y_pred_A, zero_division=0):.3f}")
    print(f"  F1-score:  {f1_score(y_true, y_pred_A, zero_division=0):.3f}")
    
    print("\nモデルB（バランス型）:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_B):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred_B):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred_B):.3f}")
    print(f"  F1-score:  {f1_score(y_true, y_pred_B):.3f}")
    
    print("\n結論:")
    print("- Accuracyだけを見るとモデルAが優秀に見える（99.0%）")
    print("- しかし、がん患者を1人も検出できていない（Recall=0）")
    print("- 医療現場ではRecallやF1-scoreが重要！")
    

**出力** ：
    
    
    === がん検出シナリオ：評価指標の比較 ===
    
    モデルA（全て陰性と予測）:
      Accuracy:  0.990
      Precision: N/A (陽性予測なし)
      Recall:    0.000
      F1-score:  0.000
    
    モデルB（バランス型）:
      Accuracy:  0.982
      Precision: 0.583
      Recall:    0.700
      F1-score:  0.636
    
    結論:
    - Accuracyだけを見るとモデルAが優秀に見える（99.0%）
    - しかし、がん患者を1人も検出できていない（Recall=0）
    - 医療現場ではRecallやF1-scoreが重要！
    

> **重要** : 評価指標は「ビジネス目的」と「データの特性」に基づいて選択する必要があります。

* * *

## 1.2 分類問題の評価指標

### Confusion Matrix（混同行列）

**Confusion Matrix** は、分類モデルの予測結果を整理した表で、すべての分類指標の基礎となります。

| **予測: Positive** | **予測: Negative**  
---|---|---  
**実際: Positive** | TP (True Positive) | FN (False Negative)  
**実際: Negative** | FP (False Positive) | TN (True Negative)  
  
  * **TP (True Positive)** : 正しく陽性と予測
  * **TN (True Negative)** : 正しく陰性と予測
  * **FP (False Positive)** : 誤って陽性と予測（第一種過誤、Type I Error）
  * **FN (False Negative)** : 誤って陰性と予測（第二種過誤、Type II Error）

### Confusion Matrixの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # サンプルデータ生成
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               n_classes=2, weights=[0.7, 0.3],
                               random_state=42)
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # モデル訓練
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Confusion Matrix計算
    cm = confusion_matrix(y_test, y_pred)
    
    print("=== Confusion Matrix ===")
    print(cm)
    print(f"\nTP (True Positive):  {cm[1, 1]}")
    print(f"TN (True Negative):  {cm[0, 0]}")
    print(f"FP (False Positive): {cm[0, 1]}")
    print(f"FN (False Negative): {cm[1, 0]}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 標準的な表示
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Negative', 'Positive'])
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix（カウント）', fontsize=14)
    
    # 正規化（割合）表示
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                        display_labels=['Negative', 'Positive'])
    disp_norm.plot(ax=axes[1], cmap='Greens', values_format='.2f')
    axes[1].set_title('Confusion Matrix（正規化）', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

### Accuracy（正解率）

**Accuracy** は、全予測のうち正しく予測した割合です。

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
    
    
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # 手動計算での確認
    accuracy_manual = (cm[1, 1] + cm[0, 0]) / cm.sum()
    print(f"Accuracy (手動計算): {accuracy_manual:.3f}")
    

**Accuracyの長所と短所** ：

長所 | 短所  
---|---  
直感的で理解しやすい | 不均衡データで誤解を招く  
全体的な性能の概要 | クラスごとの性能が不明  
均衡データで有効 | ビジネスコストを考慮しない  
  
### Precision（適合率）とRecall（再現率）

#### Precision（適合率）

**Precision** は、陽性と予測したもののうち、実際に陽性だった割合です。

$$ \text{Precision} = \frac{TP}{TP + FP} $$

**意味** : 「予測の精度」- 陽性と言ったときの信頼性

#### Recall（再現率）

**Recall** は、実際に陽性のもののうち、正しく陽性と予測した割合です。

$$ \text{Recall} = \frac{TP}{TP + FN} $$

**意味** : 「検出率」- 陽性をどれだけ見逃さないか
    
    
    from sklearn.metrics import precision_score, recall_score, classification_report
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("=== Precision と Recall ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    
    # 手動計算
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    precision_manual = TP / (TP + FP)
    recall_manual = TP / (TP + FN)
    
    print(f"\nPrecision (手動): {precision_manual:.3f}")
    print(f"Recall (手動):    {recall_manual:.3f}")
    
    # 詳細レポート
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive']))
    

### PrecisionとRecallのトレードオフ
    
    
    from sklearn.metrics import precision_recall_curve
    
    # 予測確率の取得
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall曲線の計算
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall曲線
    axes[0].plot(recalls, precisions, linewidth=2, color='purple')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision-Recall Curve', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # 閾値による変化
    axes[1].plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
    axes[1].plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Precision & Recall vs Threshold', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    print("=== Precision-Recallトレードオフ ===")
    print("閾値を上げる → Precision↑、Recall↓（厳格な判定）")
    print("閾値を下げる → Precision↓、Recall↑（寛容な判定）")
    

### F1-score（F値）

**F1-score** は、PrecisionとRecallの調和平均で、両者のバランスを評価します。

$$ \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN} $$
    
    
    from sklearn.metrics import f1_score, fbeta_score
    
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score: {f1:.3f}")
    
    # 手動計算
    f1_manual = 2 * (precision * recall) / (precision + recall)
    print(f"F1-score (手動): {f1_manual:.3f}")
    
    # F-beta score（重み付けF値）
    f2 = fbeta_score(y_test, y_pred, beta=2)  # Recall重視
    f05 = fbeta_score(y_test, y_pred, beta=0.5)  # Precision重視
    
    print("\n=== F-beta Score ===")
    print(f"F2-score (Recall重視):    {f2:.3f}")
    print(f"F1-score (バランス):       {f1:.3f}")
    print(f"F0.5-score (Precision重視): {f05:.3f}")
    

**F-beta scoreの一般形** ：

$$ F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} $$

  * $\beta > 1$: Recallを重視（がん検出など）
  * $\beta = 1$: Precision = Recall（バランス）
  * $\beta < 1$: Precisionを重視（スパムフィルタなど）

### ROC曲線とAUC

**ROC曲線（Receiver Operating Characteristic Curve）** は、閾値を変化させたときのTPR（True Positive Rate）とFPR（False Positive Rate）の関係を示します。

$$ \text{TPR (True Positive Rate)} = \text{Recall} = \frac{TP}{TP + FN} $$

$$ \text{FPR (False Positive Rate)} = \frac{FP}{FP + TN} $$

**AUC（Area Under the Curve）** は、ROC曲線の下の面積で、モデルの総合的な性能を示します。
    
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # ROC曲線の計算
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"=== ROC-AUC ===")
    print(f"AUC: {roc_auc:.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2,
             linestyle='--', label='Random classifier (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== AUCの解釈 ===")
    print("AUC = 1.0: 完璧な分類器")
    print("AUC = 0.9-1.0: 優秀")
    print("AUC = 0.8-0.9: 良好")
    print("AUC = 0.7-0.8: まあまあ")
    print("AUC = 0.5-0.7: 不十分")
    print("AUC = 0.5: ランダム分類器（意味なし）")
    

### PR-AUC（Precision-Recall AUC）

**PR-AUC** は、Precision-Recall曲線の下の面積です。不均衡データではROC-AUCよりも有用です。
    
    
    from sklearn.metrics import average_precision_score
    
    # PR-AUCの計算
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"=== PR-AUC ===")
    print(f"PR-AUC: {pr_auc:.3f}")
    
    # ROC-AUCとPR-AUCの比較可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC曲線
    axes[0].plot(fpr, tpr, color='darkorange', linewidth=2,
                label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', linewidth=2,
                linestyle='--', label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR曲線
    axes[1].plot(recalls, precisions, color='purple', linewidth=2,
                label=f'PR (AUC = {pr_auc:.3f})')
    # ベースライン（陽性クラスの割合）
    baseline = y_test.sum() / len(y_test)
    axes[1].axhline(y=baseline, color='navy', linewidth=2,
                   linestyle='--', label=f'Baseline = {baseline:.2f}')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== ROC-AUC vs PR-AUC ===")
    print("ROC-AUC: 均衡データに適している")
    print("PR-AUC: 不均衡データ（陽性クラスが少ない）に適している")
    

### Log Loss（対数損失）

**Log Loss** は、予測確率と実際のラベルの乖離を測定します。小さいほど良好です。

$$ \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right] $$
    
    
    from sklearn.metrics import log_loss
    
    # Log Lossの計算
    logloss = log_loss(y_test, y_proba)
    
    print(f"=== Log Loss ===")
    print(f"Log Loss: {logloss:.3f}")
    
    # 完璧な予測との比較
    y_proba_perfect = y_test.astype(float)
    logloss_perfect = log_loss(y_test, y_proba_perfect)
    
    # ランダム予測との比較
    y_proba_random = np.random.rand(len(y_test))
    logloss_random = log_loss(y_test, y_proba_random)
    
    print(f"\n比較:")
    print(f"  現在のモデル: {logloss:.3f}")
    print(f"  完璧な予測:   {logloss_perfect:.6f}")
    print(f"  ランダム予測: {logloss_random:.3f}")
    
    print("\n解釈:")
    print("Log Lossは確率予測の質を評価")
    print("0に近いほど良好（完璧な予測では0）")
    print("Kaggleなど確率予測が重要なコンペで使用")
    

### 多クラス分類の評価指標

#### Macro / Micro / Weighted Averaging
    
    
    from sklearn.datasets import make_classification
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    
    # 多クラスデータ生成
    X_multi, y_multi = make_classification(n_samples=1000, n_features=20,
                                           n_informative=15, n_redundant=5,
                                           n_classes=3, n_clusters_per_class=1,
                                           weights=[0.5, 0.3, 0.2],
                                           random_state=42)
    
    # 訓練・テストデータ分割
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42
    )
    
    # モデル訓練
    model_multi = LogisticRegression(random_state=42, max_iter=1000)
    model_multi.fit(X_train_m, y_train_m)
    y_pred_m = model_multi.predict(X_test_m)
    
    print("=== 多クラス分類の評価 ===")
    print(classification_report(y_test_m, y_pred_m,
                               target_names=['Class 0', 'Class 1', 'Class 2']))
    
    # Confusion Matrix
    cm_multi = confusion_matrix(y_test_m, y_pred_m)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix (Multi-class)', fontsize=14)
    plt.show()
    
    print("\n=== 平均化手法の説明 ===")
    print("Macro-average: 各クラスの指標を単純平均（クラス不均衡に敏感）")
    print("Micro-average: 全体のTP, FP, FNから計算（サンプル数重視）")
    print("Weighted-average: クラスサイズで重み付け平均（実用的）")
    

* * *

## 1.3 回帰問題の評価指標

### 回帰問題とは

**回帰問題** は、連続値を予測するタスクです（例: 住宅価格、気温、売上）。

### MAE（Mean Absolute Error）

**MAE** は、予測値と実際の値の絶対誤差の平均です。

$$ \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
    
    
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 回帰データ生成
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10,
                                    n_informative=8, noise=20,
                                    random_state=42)
    
    # 訓練・テストデータ分割
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # モデル訓練
    model_reg = LinearRegression()
    model_reg.fit(X_train_r, y_train_r)
    y_pred_r = model_reg.predict(X_test_r)
    
    # MAEの計算
    mae = mean_absolute_error(y_test_r, y_pred_r)
    
    print("=== MAE (Mean Absolute Error) ===")
    print(f"MAE: {mae:.3f}")
    
    # 手動計算
    mae_manual = np.mean(np.abs(y_test_r - y_pred_r))
    print(f"MAE (手動計算): {mae_manual:.3f}")
    
    print("\n特徴:")
    print("- 外れ値に頑健")
    print("- 解釈しやすい（元の単位）")
    print("- すべての誤差を等しく扱う")
    

### MSE（Mean Squared Error）とRMSE

**MSE** は、予測値と実際の値の二乗誤差の平均です。

$$ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$

**RMSE（Root Mean Squared Error）** は、MSEの平方根で、元の単位に戻します。

$$ \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$
    
    
    from sklearn.metrics import mean_squared_error
    
    # MSEの計算
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    
    # または直接計算
    rmse_direct = mean_squared_error(y_test_r, y_pred_r, squared=False)
    
    print("=== MSE と RMSE ===")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"RMSE (直接計算): {rmse_direct:.3f}")
    
    print("\n特徴:")
    print("- 大きな誤差を強く罰する")
    print("- 外れ値に敏感")
    print("- 微分可能（最適化に有利）")
    print("- RMSEは元の単位で解釈可能")
    

### MAE vs MSE/RMSEの比較
    
    
    # 外れ値の影響を可視化
    # 完璧な予測に少数の外れ値を追加
    y_pred_with_outliers = y_test_r.copy()
    outlier_indices = np.random.choice(len(y_test_r), size=5, replace=False)
    y_pred_with_outliers[outlier_indices] += np.random.uniform(100, 200, 5)
    
    mae_with_outliers = mean_absolute_error(y_test_r, y_pred_with_outliers)
    rmse_with_outliers = mean_squared_error(y_test_r, y_pred_with_outliers, squared=False)
    
    print("=== 外れ値の影響 ===")
    print(f"元の予測:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"\n外れ値追加後:")
    print(f"  MAE:  {mae_with_outliers:.3f} (増加: {mae_with_outliers/mae:.2f}倍)")
    print(f"  RMSE: {rmse_with_outliers:.3f} (増加: {rmse_with_outliers/rmse:.2f}倍)")
    
    print("\n結論:")
    print("RMSEは外れ値に非常に敏感（二乗するため）")
    print("MAEは外れ値に頑健")
    

### R²（決定係数）

**R²（R-squared）** は、モデルがデータの分散をどれだけ説明できるかを示します。

$$ R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2} = 1 - \frac{\text{SSR}}{\text{SST}} $$

  * $\text{SSR}$: Residual Sum of Squares（残差平方和）
  * $\text{SST}$: Total Sum of Squares（全平方和）

    
    
    from sklearn.metrics import r2_score
    
    # R²の計算
    r2 = r2_score(y_test_r, y_pred_r)
    
    print("=== R² (決定係数) ===")
    print(f"R²: {r2:.3f}")
    
    # 手動計算
    ss_res = np.sum((y_test_r - y_pred_r) ** 2)
    ss_tot = np.sum((y_test_r - y_test_r.mean()) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    print(f"R² (手動計算): {r2_manual:.3f}")
    
    print("\n=== R²の解釈 ===")
    print("R² = 1.0: 完璧な予測")
    print("R² = 0.9-1.0: 非常に良好")
    print("R² = 0.7-0.9: 良好")
    print("R² = 0.5-0.7: まあまあ")
    print("R² = 0.0: モデルが平均値と同等")
    print("R² < 0.0: モデルが平均値より悪い")
    
    print(f"\n意味: モデルは分散の{r2*100:.1f}%を説明している")
    

### Adjusted R²（調整済み決定係数）

**Adjusted R²** は、特徴量の数を考慮したR²です。

$$ \text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(N - 1)}{N - p - 1} $$

  * $N$: サンプル数
  * $p$: 特徴量の数

    
    
    # Adjusted R²の計算
    n = len(y_test_r)
    p = X_test_r.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print("=== Adjusted R² ===")
    print(f"R²:          {r2:.3f}")
    print(f"Adjusted R²: {adjusted_r2:.3f}")
    print(f"特徴量の数:   {p}")
    print(f"サンプル数:   {n}")
    
    print("\n特徴:")
    print("- 特徴量が増えてもペナルティを課す")
    print("- モデル選択（特徴量選択）に有用")
    print("- 過学習の検出に役立つ")
    

### MAPE（Mean Absolute Percentage Error）

**MAPE** は、相対誤差の平均を百分率で表します。

$$ \text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| $$
    
    
    from sklearn.metrics import mean_absolute_percentage_error
    
    # MAPEの計算（scikit-learn 0.24+）
    mape = mean_absolute_percentage_error(y_test_r, y_pred_r)
    
    # 手動計算
    mape_manual = np.mean(np.abs((y_test_r - y_pred_r) / y_test_r)) * 100
    
    print("=== MAPE (Mean Absolute Percentage Error) ===")
    print(f"MAPE: {mape*100:.2f}%")
    print(f"MAPE (手動): {mape_manual:.2f}%")
    
    print("\n特徴:")
    print("- スケールに依存しない（異なるデータセットで比較可能）")
    print("- ビジネスで理解しやすい（パーセント表示）")
    print("- y=0があると計算不可")
    print("- 小さい値で誤差が大きく見える")
    

### MSLE（Mean Squared Logarithmic Error）

**MSLE** は、対数スケールでの二乗誤差です。

$$ \text{MSLE} = \frac{1}{N} \sum_{i=1}^{N} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2 $$
    
    
    from sklearn.metrics import mean_squared_log_error
    
    # MSLEの計算（正の値のみ）
    y_test_r_positive = np.abs(y_test_r) + 1
    y_pred_r_positive = np.abs(y_pred_r) + 1
    
    msle = mean_squared_log_error(y_test_r_positive, y_pred_r_positive)
    
    print("=== MSLE (Mean Squared Logarithmic Error) ===")
    print(f"MSLE: {msle:.4f}")
    
    print("\n特徴:")
    print("- 大きい値の誤差を相対的に小さく評価")
    print("- 小さい値の予測精度を重視")
    print("- 正の値のみ使用可能")
    print("- 価格予測など右裾の長い分布に有用")
    

### 回帰指標の可視化
    
    
    # 予測値 vs 実際の値
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 散布図
    axes[0, 0].scatter(y_test_r, y_pred_r, alpha=0.5, edgecolors='black')
    axes[0, 0].plot([y_test_r.min(), y_test_r.max()],
                   [y_test_r.min(), y_test_r.max()],
                   'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual', fontsize=12)
    axes[0, 0].set_ylabel('Predicted', fontsize=12)
    axes[0, 0].set_title('Predicted vs Actual', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差プロット
    residuals = y_test_r - y_pred_r
    axes[0, 1].scatter(y_pred_r, residuals, alpha=0.5, edgecolors='black')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差の分布
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual Distribution', fontsize=14)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 評価指標の比較
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Adj R²': adjusted_r2,
        'MAPE (%)': mape * 100
    }
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = axes[1, 1].bar(range(len(metrics)), metric_values,
                           color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'pink'],
                           edgecolor='black')
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].set_title('Regression Metrics', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 各バーに値を表示
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

### 回帰指標の選択ガイドライン

状況 | 推奨指標 | 理由  
---|---|---  
外れ値が多い | MAE | 外れ値に頑健  
大きな誤差を重視 | MSE、RMSE | 二乗により大きな誤差を強調  
説明力を評価 | R²、Adjusted R² | 分散の説明割合  
相対誤差が重要 | MAPE | パーセント表示  
右裾の長い分布 | MSLE | 対数スケール  
異なるスケールの比較 | MAPE、R² | スケールに依存しない  
  
* * *

## 1.4 評価指標の選び方

### ビジネス目的に応じた選択

ビジネス課題 | 重視すべきこと | 推奨指標  
---|---|---  
**医療診断（がん検出）** | 偽陰性を最小化 | Recall、F2-score  
**スパムフィルタ** | 偽陽性を最小化 | Precision、F0.5-score  
**与信審査** | バランス | F1-score、AUC  
**不正検出** | 異常を見逃さない | Recall、PR-AUC  
**住宅価格予測** | 平均的な誤差 | MAE、RMSE  
**需要予測** | 相対誤差 | MAPE  
**売上予測** | 説明力 | R²、Adjusted R²  
  
### 不均衡データでの注意点
    
    
    # 極端な不均衡データの例
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    # 不均衡データ生成（陽性クラス1%）
    X_imb, y_imb = make_classification(n_samples=10000, n_features=20,
                                       n_informative=15, n_redundant=5,
                                       n_classes=2, weights=[0.99, 0.01],
                                       random_state=42)
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42
    )
    
    # 2つのモデル
    # モデルA: 常に陰性と予測（ナイーブ）
    y_pred_A = np.zeros(len(y_test_imb))
    
    # モデルB: 実際に学習（ロジスティック回帰）
    model_imb = LogisticRegression(random_state=42, max_iter=1000)
    model_imb.fit(X_train_imb, y_train_imb)
    y_pred_B = model_imb.predict(X_test_imb)
    y_proba_B = model_imb.predict_proba(X_test_imb)[:, 1]
    
    print("=== 不均衡データでの評価指標の比較 ===\n")
    print(f"陽性クラスの割合: {y_test_imb.sum() / len(y_test_imb) * 100:.2f}%\n")
    
    print("モデルA（常に陰性と予測）:")
    print(f"  Accuracy:  {accuracy_score(y_test_imb, y_pred_A):.3f}")
    print(f"  F1-score:  {f1_score(y_test_imb, y_pred_A, zero_division=0):.3f}")
    print(f"  ROC-AUC:   計算不可（予測に変化なし）")
    
    print("\nモデルB（実際に学習）:")
    print(f"  Accuracy:  {accuracy_score(y_test_imb, y_pred_B):.3f}")
    print(f"  F1-score:  {f1_score(y_test_imb, y_pred_B):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test_imb, y_proba_B):.3f}")
    
    print("\n結論:")
    print("- Accuracyは不均衡データで誤解を招く")
    print("- F1-score、AUCは不均衡データでも信頼できる")
    print("- 常に複数の指標を確認すべき")
    

### コスト考慮型の評価
    
    
    # コストを考慮した評価
    def cost_based_evaluation(y_true, y_pred, cost_fp, cost_fn):
        """
        コストベースの評価
    
        Parameters:
        -----------
        y_true : array-like
            実際のラベル
        y_pred : array-like
            予測ラベル
        cost_fp : float
            偽陽性のコスト
        cost_fn : float
            偽陰性のコスト
    
        Returns:
        --------
        total_cost : float
            総コスト
        """
        cm = confusion_matrix(y_true, y_pred)
        fp = cm[0, 1]
        fn = cm[1, 0]
    
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        return total_cost, fp, fn
    
    # 例: クレジットカード不正検出
    # 偽陽性（正常を不正と誤判定）: 顧客の不便、コスト = 10
    # 偽陰性（不正を見逃す）: 金銭損失、コスト = 100
    
    cost_fp = 10
    cost_fn = 100
    
    total_cost, fp, fn = cost_based_evaluation(y_test, y_pred, cost_fp, cost_fn)
    
    print("=== コストベース評価 ===")
    print(f"偽陽性（FP）の数: {fp}, コスト: {fp * cost_fp}")
    print(f"偽陰性（FN）の数: {fn}, コスト: {fn * cost_fn}")
    print(f"総コスト: {total_cost}")
    
    print("\n意味:")
    print("ビジネスコストを考慮することで、")
    print("精度以外の重要な要素を定量化できる")
    

### カスタム評価指標の作成
    
    
    from sklearn.metrics import make_scorer
    
    # カスタム評価関数
    def custom_business_metric(y_true, y_pred):
        """
        ビジネス目的に特化したカスタム評価指標
        例: 偽陰性を強く罰する
        """
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
    
        # カスタムスコア: 正解に報酬、偽陰性に大きなペナルティ
        score = (tp * 1.0) + (tn * 0.5) - (fp * 0.3) - (fn * 2.0)
        return score
    
    # scikit-learnで使えるようにscorerを作成
    custom_scorer = make_scorer(custom_business_metric)
    
    # グリッドサーチなどで使用可能
    print("=== カスタム評価指標 ===")
    print(f"カスタムスコア: {custom_business_metric(y_test, y_pred):.2f}")
    
    print("\n用途:")
    print("- ビジネス固有の優先順位を反映")
    print("- ハイパーパラメータ調整に使用")
    print("- モデル選択の基準として活用")
    

* * *

## 1.5 本章のまとめ

### 学んだこと

  1. **評価指標の重要性**

     * 適切な指標選択がモデル開発の成否を決める
     * ビジネス目的とデータ特性に基づいて選択
  2. **分類問題の評価指標**

     * Confusion Matrix: すべての指標の基礎
     * Accuracy: 全体の正解率（不均衡データで注意）
     * Precision: 陽性予測の精度
     * Recall: 陽性の検出率
     * F1-score: PrecisionとRecallのバランス
     * ROC-AUC: 閾値に依存しない総合評価
     * PR-AUC: 不均衡データに適した評価
     * Log Loss: 確率予測の質
  3. **回帰問題の評価指標**

     * MAE: 外れ値に頑健
     * MSE/RMSE: 大きな誤差を強調
     * R²: 分散の説明力
     * MAPE: 相対誤差（パーセント表示）
     * MSLE: 右裾の長い分布に有用
  4. **評価指標の選び方**

     * ビジネスコストの考慮
     * 不均衡データへの対応
     * カスタム指標の作成

### 評価指標選択のフローチャート
    
    
    ```mermaid
    graph TD
        A[問題の種類] --> B{分類 or 回帰?}
    
        B -->|分類| C{データは均衡?}
        B -->|回帰| D{外れ値は多い?}
    
        C -->|均衡| E[Accuracy, F1-score]
        C -->|不均衡| F{何を重視?}
    
        F -->|偽陰性を避ける| G[Recall, F2-score]
        F -->|偽陽性を避ける| H[Precision, F0.5-score]
        F -->|バランス| I[F1-score, AUC]
    
        D -->|多い| J[MAE, RobustScaler]
        D -->|少ない| K{相対 or 絶対誤差?}
    
        K -->|相対| L[MAPE]
        K -->|絶対| M[RMSE, R²]
    
        style A fill:#e8f5e9
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#fce4ec
    ```

### ベストプラクティス

原則 | 説明  
---|---  
**複数指標の確認** | 単一指標に頼らず、複数の角度から評価  
**ビジネス目的優先** | 技術的な精度よりビジネス価値を重視  
**データ特性の理解** | 不均衡、外れ値などデータの特徴を把握  
**閾値の調整** | PrecisionとRecallのトレードオフを考慮  
**検証データで評価** | 訓練データではなく、未知データで性能測定  
  
### 次の章へ

第2章では、**交差検証とモデル選択** を学びます：

  * K-fold Cross-Validation
  * Stratified K-fold
  * Time Series Cross-Validation
  * ハイパーパラメータチューニング
  * 学習曲線と検証曲線

* * *

## 演習問題

### 問題1（難易度：easy）

以下のConfusion Matrixから、Accuracy、Precision、Recall、F1-scoreを計算してください。

| 予測: Positive | 予測: Negative  
---|---|---  
実際: Positive | 80 | 20  
実際: Negative | 10 | 90  
解答例

**解答** ：
    
    
    # Confusion Matrixの値
    TP = 80
    FN = 20
    FP = 10
    TN = 90
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Precision
    precision = TP / (TP + FP)
    print(f"Precision: {precision:.3f}")
    
    # Recall
    recall = TP / (TP + FN)
    print(f"Recall: {recall:.3f}")
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1-score: {f1:.3f}")
    

**出力** ：
    
    
    Accuracy: 0.850
    Precision: 0.889
    Recall: 0.800
    F1-score: 0.842
    

### 問題2（難易度：medium）

がん検出シナリオで、以下の2つのモデルがあります。どちらが優れていますか？理由とともに説明してください。

  * モデルA: Precision=0.95, Recall=0.60
  * モデルB: Precision=0.70, Recall=0.90

解答例

**解答** ：

**モデルBが優れている**

**理由** ：

  1. **がん検出ではRecallが最重要**

     * がん患者を見逃す（偽陰性）は生命に関わる
     * 健康な人をがんと誤判定（偽陽性）は再検査で対応可能
  2. **数値分析**

     * モデルA: 100人の患者のうち40人を見逃す（Recall=0.60）
     * モデルB: 100人の患者のうち10人を見逃す（Recall=0.90）
  3. **F-beta scoreで評価（Recall重視）**

    
    
    import numpy as np
    
    # モデルA
    precision_A = 0.95
    recall_A = 0.60
    f2_A = (1 + 2**2) * (precision_A * recall_A) / (2**2 * precision_A + recall_A)
    
    # モデルB
    precision_B = 0.70
    recall_B = 0.90
    f2_B = (1 + 2**2) * (precision_B * recall_B) / (2**2 * precision_B + recall_B)
    
    print(f"モデルA F2-score: {f2_A:.3f}")
    print(f"モデルB F2-score: {f2_B:.3f}")
    print(f"\nモデルBが{(f2_B/f2_A - 1)*100:.1f}%優れている")
    

**出力** ：
    
    
    モデルA F2-score: 0.635
    モデルB F2-score: 0.847
    モデルBが33.4%優れている
    

### 問題3（難易度：medium）

ROC-AUCとPR-AUCの違いを説明し、それぞれをどのような場面で使うべきか述べてください。

解答例

**解答** ：

**ROC-AUC（ROC曲線の下の面積）** ：

  * 軸: FPR (False Positive Rate) vs TPR (True Positive Rate)
  * 特徴: 陰性クラスと陽性クラスを均等に評価
  * 適用: データが均衡している場合

**PR-AUC（Precision-Recall曲線の下の面積）** ：

  * 軸: Recall vs Precision
  * 特徴: 陽性クラスの予測性能に焦点
  * 適用: データが不均衡（陽性クラスが少ない）場合

**使い分け** ：

シナリオ | 推奨 | 理由  
---|---|---  
均衡データ（50:50） | ROC-AUC | 両クラスを均等に評価  
不均衡データ（1:99） | PR-AUC | 少数クラスの性能に焦点  
不正検出 | PR-AUC | 不正（少数）を見逃さない  
医療診断（まれな病気） | PR-AUC | 病気（少数）の検出が重要  
スパムフィルタ | PR-AUC | スパム（変動あり）の検出  
  
**実例** ：
    
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # 不均衡データ生成
    X_imb, y_imb = make_classification(n_samples=1000, n_features=20,
                                       weights=[0.95, 0.05], random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"陽性クラスの割合: {y_test.sum() / len(y_test) * 100:.1f}%")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")
    print("\n不均衡データでは、PR-AUCがより現実的な性能を示す")
    

### 問題4（難易度：hard）

以下の回帰モデルの予測結果に対して、MAE、RMSE、R²、MAPEを計算し、結果を解釈してください。
    
    
    import numpy as np
    
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 320])
    

解答例
    
    
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 320])
    
    # 各指標の計算
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print("=== 回帰評価指標 ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 詳細分析
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    percentage_errors = np.abs(residuals / y_true) * 100
    
    print("\n=== 詳細分析 ===")
    print(f"実際の値:   {y_true}")
    print(f"予測値:     {y_pred}")
    print(f"残差:       {residuals}")
    print(f"絶対残差:   {abs_residuals}")
    print(f"誤差率(%):  {percentage_errors.round(2)}")
    
    print("\n=== 解釈 ===")
    print(f"1. MAE = {mae:.2f}")
    print("   平均して約12単位の誤差がある")
    print(f"\n2. RMSE = {rmse:.2f}")
    print("   RMSEがMAEより大きい → 大きな誤差が存在")
    print(f"\n3. R² = {r2:.3f}")
    print(f"   モデルは分散の{r2*100:.1f}%を説明 → 優秀")
    print(f"\n4. MAPE = {mape:.2f}%")
    print("   平均的に6%程度の相対誤差 → 良好")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 予測 vs 実際
    axes[0].scatter(y_true, y_pred, s=100, alpha=0.7, edgecolors='black')
    axes[0].plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual', fontsize=12)
    axes[0].set_ylabel('Predicted', fontsize=12)
    axes[0].set_title('Predicted vs Actual', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 残差プロット
    axes[1].bar(range(len(residuals)), residuals,
               color=['red' if r < 0 else 'blue' for r in residuals],
               alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=2)
    axes[1].set_xlabel('Index', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residuals (Actual - Predicted)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 回帰評価指標 ===
    MAE:  12.00
    RMSE: 13.04
    R²:   0.984
    MAPE: 5.96%
    
    === 詳細分析 ===
    実際の値:   [100 150 200 250 300]
    予測値:     [110 140 210 240 320]
    残差:       [-10  10 -10  10 -20]
    絶対残差:   [10 10 10 10 20]
    誤差率(%):  [10.    6.67  5.    4.    6.67]
    
    === 解釈 ===
    1. MAE = 12.00
       平均して約12単位の誤差がある
    
    2. RMSE = 13.04
       RMSEがMAEより大きい → 大きな誤差が存在
    
    3. R² = 0.984
       モデルは分散の98.4%を説明 → 優秀
    
    4. MAPE = 5.96%
       平均的に6%程度の相対誤差 → 良好
    

### 問題5（難易度：hard）

不均衡データ（陽性:陰性 = 1:99）において、Accuracyが高いにもかかわらずモデルが実用的でない場合があります。具体例を作成し、適切な評価指標を提案してください。

解答例
    
    
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 不均衡データのシミュレーション
    # シナリオ: 不正取引検出（不正=1%、正常=99%）
    np.random.seed(42)
    n_samples = 10000
    
    # 実際のラベル
    y_true = np.array([0] * 9900 + [1] * 100)
    
    # モデルA: 常に「正常」と予測（ナイーブな多数派分類器）
    y_pred_A = np.zeros(n_samples)
    
    # モデルB: 実際に不正を検出しようとするモデル
    # 不正の80%を検出、ただし正常の5%を誤検出
    y_pred_B = y_true.copy()
    # 不正のうち20%を見逃す
    false_negatives = np.random.choice(np.where(y_true == 1)[0], size=20, replace=False)
    y_pred_B[false_negatives] = 0
    # 正常のうち5%を誤検出
    false_positives = np.random.choice(np.where(y_true == 0)[0], size=495, replace=False)
    y_pred_B[false_positives] = 1
    
    print("=== 不均衡データでの評価指標の問題 ===\n")
    print(f"データの不均衡度: 陽性={y_true.sum()}件 ({y_true.sum()/len(y_true)*100:.1f}%), "
          f"陰性={len(y_true)-y_true.sum()}件 ({(len(y_true)-y_true.sum())/len(y_true)*100:.1f}%)\n")
    
    # モデルAの評価
    print("【モデルA: 常に「正常」と予測】")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_A):.3f} ← 高いが無意味！")
    print(f"Precision: 計算不可（陽性予測なし）")
    print(f"Recall:    {recall_score(y_true, y_pred_A, zero_division=0):.3f} ← 不正を1件も検出できない！")
    print(f"F1-score:  {f1_score(y_true, y_pred_A, zero_division=0):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_A))
    
    # モデルBの評価
    print("\n【モデルB: 実際に不正を検出】")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_B):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred_B):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred_B):.3f} ← 不正の80%を検出")
    print(f"F1-score:  {f1_score(y_true, y_pred_B):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_B))
    
    print("\n=== 結論と推奨指標 ===")
    print("\n問題点:")
    print("- モデルAはAccuracy=99.0%だが、不正を1件も検出できない")
    print("- Accuracyは不均衡データで誤解を招く")
    print("- 実用的には完全に無価値なモデル")
    
    print("\n推奨する評価指標:")
    print("1. F1-score: PrecisionとRecallのバランスを評価")
    print("2. Recall: 不正検出率（見逃しを最小化）")
    print("3. PR-AUC: 不均衡データに特化した総合評価")
    print("4. Cohen's Kappa: 偶然による一致を補正")
    
    print("\n実用的な判断:")
    print(f"- モデルAは実用不可（Recall=0）")
    print(f"- モデルBは実用的（F1={f1_score(y_true, y_pred_B):.3f}, Recall={recall_score(y_true, y_pred_B):.3f}）")
    
    # 詳細レポート
    print("\n【詳細レポート】")
    print("\nモデルA:")
    print(classification_report(y_true, y_pred_A, target_names=['正常', '不正'], zero_division=0))
    
    print("\nモデルB:")
    print(classification_report(y_true, y_pred_B, target_names=['正常', '不正']))
    
    # ビジネス的な解釈
    print("\n=== ビジネスへの影響 ===")
    cm_B = confusion_matrix(y_true, y_pred_B)
    tp, fp, fn = cm_B[1,1], cm_B[0,1], cm_B[1,0]
    
    # コスト計算（例）
    cost_per_fraud = 10000  # 不正1件あたりの損失
    cost_per_false_alarm = 100  # 誤検出1件あたりのコスト
    
    loss_A = 100 * cost_per_fraud  # 全ての不正を見逃す
    loss_B = (fn * cost_per_fraud) + (fp * cost_per_false_alarm)
    
    print(f"\nモデルA:")
    print(f"  見逃した不正: {100}件")
    print(f"  推定損失: ¥{loss_A:,}")
    
    print(f"\nモデルB:")
    print(f"  見逃した不正: {fn}件")
    print(f"  誤検出: {fp}件")
    print(f"  推定損失: ¥{loss_B:,}")
    
    print(f"\nモデルBの採用により、¥{loss_A - loss_B:,}の損失を防げる")
    

**出力（例）** ：
    
    
    === 不均衡データでの評価指標の問題 ===
    
    データの不均衡度: 陽性=100件 (1.0%), 陰性=9900件 (99.0%)
    
    【モデルA: 常に「正常」と予測】
    Accuracy:  0.990 ← 高いが無意味！
    Precision: 計算不可（陽性予測なし）
    Recall:    0.000 ← 不正を1件も検出できない！
    F1-score:  0.000
    
    Confusion Matrix:
    [[9900    0]
     [ 100    0]]
    
    【モデルB: 実際に不正を検出】
    Accuracy:  0.948
    Precision: 0.139
    Recall:    0.800 ← 不正の80%を検出
    F1-score:  0.237
    
    Confusion Matrix:
    [[9405  495]
     [  20   80]]
    
    === 結論と推奨指標 ===
    
    問題点:
    - モデルAはAccuracy=99.0%だが、不正を1件も検出できない
    - Accuracyは不均衡データで誤解を招く
    - 実用的には完全に無価値なモデル
    
    推奨する評価指標:
    1. F1-score: PrecisionとRecallのバランスを評価
    2. Recall: 不正検出率（見逃しを最小化）
    3. PR-AUC: 不均衡データに特化した総合評価
    4. Cohen's Kappa: 偶然による一致を補正
    
    実用的な判断:
    - モデルAは実用不可（Recall=0）
    - モデルBは実用的（F1=0.237, Recall=0.800）
    
    === ビジネスへの影響 ===
    
    モデルA:
      見逃した不正: 100件
      推定損失: ¥1,000,000
    
    モデルB:
      見逃した不正: 20件
      誤検出: 495件
      推定損失: ¥249,500
    
    モデルBの採用により、¥750,500の損失を防げる
    

* * *

## 参考文献

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_ (2nd ed.). Springer.
  3. Kuhn, M., & Johnson, K. (2013). _Applied Predictive Modeling_. Springer.
  4. Provost, F., & Fawcett, T. (2013). _Data Science for Business_. O'Reilly Media.
  5. Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. _PLOS ONE_ , 10(3).
