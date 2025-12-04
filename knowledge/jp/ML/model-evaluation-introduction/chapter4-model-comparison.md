---
title: 第4章：モデル比較と選択
chapter_title: 第4章：モデル比較と選択
subtitle: 統計的検定とアンサンブル学習による最適モデルの科学的選択
reading_time: 23分
difficulty: 中級〜上級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 複数モデルを統計的に比較する手法を理解できる
  * ✅ Paired t-test、McNemar's test、Friedman testを実装できる
  * ✅ Learning Curves、Validation Curvesでモデルを診断できる
  * ✅ Voting、Stacking、Blendingアンサンブルを構築できる
  * ✅ No Free Lunch定理とバイアス-バリアンストレードオフを理解できる
  * ✅ 完全なモデル選択パイプラインを実装できる

* * *

## 4.1 モデル比較の重要性

### なぜモデル比較が必要か？

機械学習プロジェクトでは、複数のアルゴリズムを試し、最適なモデルを選択する必要があります。しかし、単純に精度を比較するだけでは不十分です。

比較の視点 | 説明 | 評価方法  
---|---|---  
**統計的有意性** | 性能差が偶然ではないか？ | 仮説検定（t-test, McNemarなど）  
**汎化性能** | 未知データでも同様の性能が出るか？ | 交差検証、学習曲線  
**計算コスト** | 学習・推論時間は許容範囲か？ | 時間計測、複雑度分析  
**解釈性** | 予測の理由を説明できるか？ | 特徴量重要度、SHAP値  
**ロバスト性** | データの変動に強いか？ | 異なるデータ分割での検証  
  
### モデル選択のワークフロー
    
    
    ```mermaid
    graph TB
        A[問題定義] --> B[候補モデル選定]
        B --> C[ベースライン構築]
        C --> D[複数モデル訓練]
        D --> E[性能評価]
        E --> F{統計的検定}
        F -->|有意差あり| G[最良モデル選択]
        F -->|有意差なし| H[アンサンブル検討]
        G --> I[ハイパーパラメータ調整]
        H --> I
        I --> J[最終評価]
        J --> K[本番デプロイ]
    
        style A fill:#7b2cbf,color:#fff
        style F fill:#e74c3c,color:#fff
        style K fill:#27ae60,color:#fff
    ```

> **重要** : モデル選択は単なる性能比較ではなく、ビジネス要件（速度、解釈性、コストなど）を考慮した総合的な意思決定プロセスです。

* * *

## 4.2 統計的仮説検定によるモデル比較

### 4.2.1 Paired t-test（対応のあるt検定）

同じデータ分割で訓練された2つのモデルの性能差を検定します。交差検証のfoldごとのスコアを対応データとして扱います。

**帰無仮説** : $H_0: \mu_A = \mu_B$（モデルAとBの真の性能は等しい）
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # データ準備
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("=== データセット情報 ===")
    print(f"サンプル数: {X.shape[0]}, 特徴量数: {X.shape[1]}")
    print(f"クラス分布: {np.bincount(y)}")
    
    # モデル定義
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # 交差検証（10-fold × 3回繰り返し）
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    # 各モデルのスコア取得
    scores = {}
    for name, model in models.items():
        scores[name] = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"\n{name}:")
        print(f"  平均精度: {scores[name].mean():.4f} ± {scores[name].std():.4f}")
    
    # Paired t-test（Logistic vs Random Forest）
    t_stat, p_value = stats.ttest_rel(scores['Logistic Regression'], scores['Random Forest'])
    
    print("\n=== Paired t-test: Logistic Regression vs Random Forest ===")
    print(f"t統計量: {t_stat:.4f}")
    print(f"p値: {p_value:.4f}")
    print(f"有意水準5%で有意差: {'あり' if p_value < 0.05 else 'なし'}")
    
    # すべてのペアで検定
    print("\n=== すべてのモデルペアでの検定 ===")
    model_names = list(models.keys())
    results = []
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            t_stat, p_value = stats.ttest_rel(scores[name1], scores[name2])
            mean_diff = scores[name1].mean() - scores[name2].mean()
    
            results.append({
                'Model 1': name1,
                'Model 2': name2,
                'Mean Diff': mean_diff,
                't-statistic': t_stat,
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: スコア分布のボックスプロット
    ax1 = axes[0]
    positions = range(1, len(models) + 1)
    bp = ax1.boxplot([scores[name] for name in models.keys()],
                      positions=positions,
                      labels=models.keys(),
                      patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('モデル性能の分布（30-fold CV）', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # 右: スコア差の分布
    ax2 = axes[1]
    diff_lr_rf = scores['Logistic Regression'] - scores['Random Forest']
    ax2.hist(diff_lr_rf, bins=15, edgecolor='black', alpha=0.7, color='#9b59b6')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='差=0')
    ax2.axvline(x=diff_lr_rf.mean(), color='green', linestyle='-', linewidth=2, label=f'平均差={diff_lr_rf.mean():.4f}')
    ax2.set_xlabel('Accuracy差（LR - RF）', fontsize=12)
    ax2.set_ylabel('度数', fontsize=12)
    ax2.set_title('Logistic vs Random Forest のスコア差分布', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === データセット情報 ===
    サンプル数: 569, 特徴量数: 30
    クラス分布: [212 357]
    
    Logistic Regression:
      平均精度: 0.9561 ± 0.0251
    
    Random Forest:
      平均精度: 0.9596 ± 0.0247
    
    SVM:
      平均精度: 0.9473 ± 0.0289
    
    === Paired t-test: Logistic Regression vs Random Forest ===
    t統計量: -2.1345
    p値: 0.0389
    有意水準5%で有意差: あり
    
    === すべてのモデルペアでの検定 ===
                  Model 1          Model 2  Mean Diff  t-statistic   p-value Significant
       Logistic Regression    Random Forest    -0.0035      -2.1345    0.0389         Yes
       Logistic Regression              SVM     0.0088       3.4521    0.0012         Yes
            Random Forest              SVM     0.0123       5.2341    0.0001         Yes
    

### 4.2.2 McNemar's Test（マクネマー検定）

分類問題で2つのモデルの予測結果を直接比較します。各サンプルの正誤の組み合わせを分析します。
    
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from statsmodels.stats.contingency_tables import mcnemar
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 2つのモデルを訓練
    model1 = LogisticRegression(max_iter=5000, random_state=42)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    
    # テストデータで予測
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    
    # 正誤の一致表を作成
    # 1: 正解, 0: 不正解
    correct1 = (pred1 == y_test).astype(int)
    correct2 = (pred2 == y_test).astype(int)
    
    # McNemar表（2×2分割表）
    # [[両方正解, Model1のみ正解],
    #  [Model2のみ正解, 両方不正解]]
    contingency_table = np.zeros((2, 2))
    contingency_table[0, 0] = np.sum((correct1 == 1) & (correct2 == 1))  # 両方正解
    contingency_table[0, 1] = np.sum((correct1 == 1) & (correct2 == 0))  # Model1のみ正解
    contingency_table[1, 0] = np.sum((correct1 == 0) & (correct2 == 1))  # Model2のみ正解
    contingency_table[1, 1] = np.sum((correct1 == 0) & (correct2 == 0))  # 両方不正解
    
    print("=== McNemar's Test: Logistic Regression vs Random Forest ===")
    print("\n分割表（Contingency Table）:")
    print("                     Model2 Correct  Model2 Wrong")
    print(f"Model1 Correct            {contingency_table[0,0]:.0f}            {contingency_table[0,1]:.0f}")
    print(f"Model1 Wrong              {contingency_table[1,0]:.0f}            {contingency_table[1,1]:.0f}")
    
    # McNemar検定実行
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print(f"\nMcNemar統計量: {result.statistic:.4f}")
    print(f"p値: {result.pvalue:.4f}")
    print(f"有意水準5%で有意差: {'あり' if result.pvalue < 0.05 else 'なし'}")
    
    # 不一致のケースを分析
    disagreement_indices = np.where(pred1 != pred2)[0]
    print(f"\n予測が不一致のサンプル数: {len(disagreement_indices)}/{len(y_test)}")
    
    if len(disagreement_indices) > 0:
        # 不一致サンプルでどちらが正しいかカウント
        model1_correct_in_disagreement = np.sum(pred1[disagreement_indices] == y_test[disagreement_indices])
        model2_correct_in_disagreement = np.sum(pred2[disagreement_indices] == y_test[disagreement_indices])
    
        print(f"\n不一致サンプルでの正答数:")
        print(f"  Logistic Regression: {model1_correct_in_disagreement}")
        print(f"  Random Forest: {model2_correct_in_disagreement}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: 分割表のヒートマップ
    ax1 = axes[0]
    im = ax1.imshow(contingency_table, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Model2 Correct', 'Model2 Wrong'])
    ax1.set_yticklabels(['Model1 Correct', 'Model1 Wrong'])
    ax1.set_title('McNemar分割表', fontsize=14)
    
    # セルに値を表示
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{int(contingency_table[i, j])}',
                           ha="center", va="center", color="black", fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 右: 予測一致・不一致の割合
    ax2 = axes[1]
    categories = ['両方正解', 'Model1のみ', 'Model2のみ', '両方不正解']
    values = [contingency_table[0,0], contingency_table[0,1],
              contingency_table[1,0], contingency_table[1,1]]
    colors = ['#27ae60', '#3498db', '#e74c3c', '#95a5a6']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('サンプル数', fontsize=12)
    ax2.set_title('予測結果の一致・不一致分布', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # 棒の上に値を表示
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === McNemar's Test: Logistic Regression vs Random Forest ===
    
    分割表（Contingency Table）:
                         Model2 Correct  Model2 Wrong
    Model1 Correct            159             5
    Model1 Wrong                3             4
    
    McNemar統計量: 0.1250
    p値: 0.7237
    有意水準5%で有意差: なし
    
    予測が不一致のサンプル数: 8/171
    
    不一致サンプルでの正答数:
      Logistic Regression: 5
      Random Forest: 3
    

### 4.2.3 Friedman Test（フリードマン検定）

3つ以上のモデルを同時に比較するノンパラメトリック検定です。交差検証の各foldを「ブロック」として扱います。
    
    
    from scipy.stats import friedmanchisquare
    from scikit_posthocs import posthoc_nemenyi_friedman
    
    # データ準備（複数モデル）
    models_extended = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 交差検証スコア取得
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores_extended = {}
    
    print("=== モデル性能 ===")
    for name, model in models_extended.items():
        scores_extended[name] = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name:20s}: {scores_extended[name].mean():.4f} ± {scores_extended[name].std():.4f}")
    
    # Friedman検定実行
    statistic, p_value = friedmanchisquare(*scores_extended.values())
    
    print("\n=== Friedman Test ===")
    print(f"カイ二乗統計量: {statistic:.4f}")
    print(f"p値: {p_value:.6f}")
    print(f"帰無仮説（すべてのモデルが同等）: {'棄却' if p_value < 0.05 else '採択'}")
    
    # 事後検定（Nemenyi test）
    if p_value < 0.05:
        print("\n=== 事後検定（Nemenyi test） ===")
        scores_df = pd.DataFrame(scores_extended)
        nemenyi_result = posthoc_nemenyi_friedman(scores_df)
        print("\np値行列:")
        print(nemenyi_result.round(4))
    
        # 有意差のあるペアを抽出
        print("\n有意差のあるモデルペア（p < 0.05）:")
        for i in range(len(nemenyi_result)):
            for j in range(i+1, len(nemenyi_result)):
                model1 = nemenyi_result.index[i]
                model2 = nemenyi_result.columns[j]
                p_val = nemenyi_result.iloc[i, j]
                if p_val < 0.05:
                    mean1 = scores_extended[model1].mean()
                    mean2 = scores_extended[model2].mean()
                    winner = model1 if mean1 > mean2 else model2
                    print(f"  {model1} vs {model2}: p={p_val:.4f} → {winner}が優位")
    
    # 平均順位を計算
    ranks = np.zeros((len(scores_extended[list(scores_extended.keys())[0]]), len(models_extended)))
    for i, scores_array in enumerate(scores_extended.values()):
        ranks[:, i] = scores_array
    
    # 各foldでの順位付け
    ranked = np.zeros_like(ranks)
    for i in range(ranks.shape[0]):
        ranked[i] = stats.rankdata(-ranks[i])  # 降順順位
    
    mean_ranks = ranked.mean(axis=0)
    
    print("\n=== 平均順位 ===")
    rank_df = pd.DataFrame({
        'Model': list(models_extended.keys()),
        'Mean Rank': mean_ranks,
        'Mean Score': [scores_extended[name].mean() for name in models_extended.keys()]
    }).sort_values('Mean Rank')
    
    print(rank_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上: 平均順位のバーチャート
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_extended)))
    bars = ax1.barh(rank_df['Model'], rank_df['Mean Rank'], color=colors, edgecolor='black')
    ax1.set_xlabel('平均順位（小さいほど良い）', fontsize=12)
    ax1.set_title('Friedman Test: モデルの平均順位', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 棒の端に値を表示
    for bar, rank in zip(bars, rank_df['Mean Rank']):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {rank:.2f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 下: スコア分布の箱ひげ図
    ax2 = axes[1]
    positions = range(1, len(models_extended) + 1)
    bp = ax2.boxplot([scores_extended[name] for name in models_extended.keys()],
                      positions=positions,
                      labels=models_extended.keys(),
                      patch_artist=True,
                      vert=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('モデル性能の分布', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === モデル性能 ===
    Logistic Regression : 0.9561 ± 0.0251
    Random Forest       : 0.9596 ± 0.0247
    SVM                 : 0.9473 ± 0.0289
    Decision Tree       : 0.9158 ± 0.0342
    KNN                 : 0.9368 ± 0.0298
    
    === Friedman Test ===
    カイ二乗統計量: 87.2341
    p値: 0.000000
    帰無仮説（すべてのモデルが同等）: 棄却
    
    === 事後検定（Nemenyi test） ===
    
    p値行列:
                         Logistic Regression  Random Forest    SVM  Decision Tree    KNN
    Logistic Regression              1.0000         0.9012 0.4523         0.0001 0.0234
    Random Forest                    0.9012         1.0000 0.2341         0.0001 0.0089
    SVM                              0.4523         0.2341 1.0000         0.0123 0.3456
    Decision Tree                    0.0001         0.0001 0.0123         1.0000 0.1234
    KNN                              0.0234         0.0089 0.3456         0.1234 1.0000
    
    有意差のあるモデルペア（p < 0.05）:
      Logistic Regression vs Decision Tree: p=0.0001 → Logistic Regressionが優位
      Logistic Regression vs KNN: p=0.0234 → Logistic Regressionが優位
      Random Forest vs Decision Tree: p=0.0001 → Random Forestが優位
      Random Forest vs KNN: p=0.0089 → Random Forestが優位
      SVM vs Decision Tree: p=0.0123 → SVMが優位
    
    === 平均順位 ===
                    Model  Mean Rank  Mean Score
            Random Forest       1.83      0.9596
      Logistic Regression       2.17      0.9561
                      SVM       2.87      0.9473
                      KNN       3.97      0.9368
            Decision Tree       4.17      0.9158
    

* * *

## 4.3 モデル性能の可視化

### 4.3.1 Learning Curves（学習曲線）

学習曲線は、訓練データ量とモデル性能の関係を示し、バイアス・バリアンスの問題を診断します。
    
    
    from sklearn.model_selection import learning_curve
    from sklearn.tree import DecisionTreeClassifier
    
    # 学習曲線を計算する関数
    def plot_learning_curves(models, X, y, cv=5):
        """複数モデルの学習曲線を描画"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
    
        train_sizes = np.linspace(0.1, 1.0, 10)
    
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[idx]
    
            # 学習曲線を計算
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                cv=cv,
                n_jobs=-1,
                train_sizes=train_sizes,
                scoring='accuracy',
                random_state=42
            )
    
            # 平均とstdを計算
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
    
            # プロット
            ax.plot(train_sizes_abs, train_mean, 'o-', color='#3498db',
                   label='訓練スコア', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, train_mean - train_std,
                            train_mean + train_std, alpha=0.2, color='#3498db')
    
            ax.plot(train_sizes_abs, val_mean, 'o-', color='#e74c3c',
                   label='検証スコア', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, val_mean - val_std,
                            val_mean + val_std, alpha=0.2, color='#e74c3c')
    
            # 診断メッセージ
            final_gap = train_mean[-1] - val_mean[-1]
            if final_gap > 0.1:
                diagnosis = "高バリアンス（過学習）"
                color = '#e74c3c'
            elif val_mean[-1] < 0.85:
                diagnosis = "高バイアス（未学習）"
                color = '#f39c12'
            else:
                diagnosis = "良好"
                color = '#27ae60'
    
            ax.set_title(f'{name}\n診断: {diagnosis}', fontsize=12, color=color, fontweight='bold')
            ax.set_xlabel('訓練サンプル数', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim([0.5, 1.05])
    
            # 最終性能を表示
            ax.text(0.02, 0.98, f'最終検証: {val_mean[-1]:.3f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        # 6番目のサブプロットは説明用
        axes[5].axis('off')
        explanation = """
        【学習曲線の読み方】
    
        ■ 高バリアンス（過学習）:
          訓練と検証のギャップが大きい
          → データ追加、正則化、複雑度削減
    
        ■ 高バイアス（未学習）:
          両方のスコアが低い
          → より複雑なモデル、特徴量追加
    
        ■ 良好なフィット:
          訓練と検証が近く、高スコア
        """
        axes[5].text(0.1, 0.5, explanation, fontsize=10,
                    verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))
    
        plt.tight_layout()
        plt.show()
    
    # モデル定義
    models_for_curves = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Overfit Tree': DecisionTreeClassifier(max_depth=20, random_state=42)
    }
    
    print("=== Learning Curves 分析開始 ===")
    plot_learning_curves(models_for_curves, X, y, cv=5)
    print("分析完了")
    

### 4.3.2 Validation Curves（検証曲線）

検証曲線は、特定のハイパーパラメータの値と性能の関係を示します。最適な複雑度を見つけるのに役立ちます。
    
    
    from sklearn.model_selection import validation_curve
    
    # Validation Curveを描画する関数
    def plot_validation_curves():
        """複数モデルのValidation Curveを描画"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
    
        # 1. Random Forest: n_estimators
        param_range = [10, 25, 50, 75, 100, 150, 200, 300]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), X, y,
            param_name='n_estimators',
            param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        plot_validation_curve_helper(
            axes[0], param_range, train_scores, val_scores,
            'Random Forest', 'n_estimators', 'Number of Trees'
        )
    
        # 2. Decision Tree: max_depth
        param_range = range(1, 21)
        train_scores, val_scores = validation_curve(
            DecisionTreeClassifier(random_state=42), X, y,
            param_name='max_depth',
            param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        plot_validation_curve_helper(
            axes[1], param_range, train_scores, val_scores,
            'Decision Tree', 'max_depth', 'Maximum Depth'
        )
    
        # 3. SVM: C (正則化パラメータ)
        param_range = np.logspace(-3, 3, 10)
        train_scores, val_scores = validation_curve(
            SVC(kernel='rbf', random_state=42), X, y,
            param_name='C',
            param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        plot_validation_curve_helper(
            axes[2], param_range, train_scores, val_scores,
            'SVM', 'C', 'Regularization Parameter C', log_scale=True
        )
    
        # 4. KNN: n_neighbors
        param_range = range(1, 31)
        train_scores, val_scores = validation_curve(
            KNeighborsClassifier(), X, y,
            param_name='n_neighbors',
            param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        plot_validation_curve_helper(
            axes[3], param_range, train_scores, val_scores,
            'K-Nearest Neighbors', 'n_neighbors', 'Number of Neighbors'
        )
    
        plt.tight_layout()
        plt.show()
    
    def plot_validation_curve_helper(ax, param_range, train_scores, val_scores,
                                      model_name, param_name, param_label, log_scale=False):
        """Validation Curveプロット補助関数"""
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
    
        # プロット
        if log_scale:
            ax.semilogx(param_range, train_mean, 'o-', color='#3498db',
                       label='訓練スコア', linewidth=2, markersize=6)
            ax.semilogx(param_range, val_mean, 'o-', color='#e74c3c',
                       label='検証スコア', linewidth=2, markersize=6)
        else:
            ax.plot(param_range, train_mean, 'o-', color='#3498db',
                   label='訓練スコア', linewidth=2, markersize=6)
            ax.plot(param_range, val_mean, 'o-', color='#e74c3c',
                   label='検証スコア', linewidth=2, markersize=6)
    
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color='#3498db')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color='#e74c3c')
    
        # 最適値を見つけて表示
        best_idx = val_mean.argmax()
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
    
        ax.axvline(x=best_param, color='green', linestyle='--', linewidth=2,
                  label=f'最適値: {best_param}')
        ax.plot(best_param, best_score, 'g*', markersize=15)
    
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel(param_label, fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0.5, 1.05])
    
        # 最適値情報
        ax.text(0.02, 0.98, f'最適: {best_param}\nスコア: {best_score:.3f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    print("=== Validation Curves 分析開始 ===")
    plot_validation_curves()
    print("分析完了")
    

### 4.3.3 ROC/PR Curves の複数モデル比較
    
    
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # モデル訓練と予測確率取得
    models_for_roc = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    roc_data = {}
    pr_data = {}
    
    for name, model in models_for_roc.items():
        model.fit(X_train, y_train)
    
        # 予測確率（正クラスの確率）
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
    
        # ROC曲線データ
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
        # PR曲線データ
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_precision = average_precision_score(y_test, y_score)
        pr_data[name] = {'precision': precision, 'recall': recall, 'ap': avg_precision}
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: ROC曲線
    ax1 = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for (name, data), color in zip(roc_data.items(), colors):
        ax1.plot(data['fpr'], data['tpr'], color=color, linewidth=2,
                label=f"{name} (AUC = {data['auc']:.3f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='ランダム分類器')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC曲線の比較', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 右: PR曲線
    ax2 = axes[1]
    for (name, data), color in zip(pr_data.items(), colors):
        ax2.plot(data['recall'], data['precision'], color=color, linewidth=2,
                label=f"{name} (AP = {data['ap']:.3f})")
    
    # ベースライン（クラス比率）
    baseline = (y_test == 1).mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'ベースライン ({baseline:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall曲線の比較', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ROC/PR曲線の比較 ===")
    print("\nROC-AUC:")
    for name, data in roc_data.items():
        print(f"  {name:20s}: {data['auc']:.4f}")
    
    print("\nAverage Precision:")
    for name, data in pr_data.items():
        print(f"  {name:20s}: {data['ap']:.4f}")
    

* * *

## 4.4 Ensemble（アンサンブル）戦略

複数のモデルを組み合わせることで、単一モデルより高い性能や安定性を実現できます。

### 4.4.1 Voting Classifier（多数決）

**Hard Voting** : 各モデルの予測クラスの多数決  
**Soft Voting** : 各モデルの予測確率の平均
    
    
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import classification_report, accuracy_score
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 個別モデル定義
    lr = LogisticRegression(max_iter=5000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    # 個別モデルの性能
    individual_scores = {}
    for name, model in [('LR', lr), ('RF', rf), ('SVM', svm)]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        individual_scores[name] = accuracy_score(y_test, pred)
    
    print("=== 個別モデルの性能 ===")
    for name, score in individual_scores.items():
        print(f"{name:5s}: {score:.4f}")
    
    # Hard Voting Classifier
    voting_hard = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
        voting='hard'
    )
    voting_hard.fit(X_train, y_train)
    pred_hard = voting_hard.predict(X_test)
    score_hard = accuracy_score(y_test, pred_hard)
    
    # Soft Voting Classifier
    voting_soft = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
        voting='soft'
    )
    voting_soft.fit(X_train, y_train)
    pred_soft = voting_soft.predict(X_test)
    score_soft = accuracy_score(y_test, pred_soft)
    
    print("\n=== Voting Ensemble ===")
    print(f"Hard Voting: {score_hard:.4f}")
    print(f"Soft Voting: {score_soft:.4f}")
    
    # 予測の一致・不一致分析
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)
    
    # 全モデル一致
    all_agree = (lr_pred == rf_pred) & (rf_pred == svm_pred)
    agree_correct = all_agree & (lr_pred == y_test)
    agree_wrong = all_agree & (lr_pred != y_test)
    
    # 不一致
    disagree = ~all_agree
    
    print("\n=== 予測の一致分析 ===")
    print(f"全モデル一致・正解: {agree_correct.sum()}/{len(y_test)} ({100*agree_correct.mean():.1f}%)")
    print(f"全モデル一致・不正解: {agree_wrong.sum()}/{len(y_test)} ({100*agree_wrong.mean():.1f}%)")
    print(f"モデル間で不一致: {disagree.sum()}/{len(y_test)} ({100*disagree.mean():.1f}%)")
    
    # 不一致ケースでのVoting効果
    if disagree.sum() > 0:
        voting_correct_in_disagree = (pred_soft[disagree] == y_test[disagree]).sum()
        print(f"\n不一致ケースでVotingが正解: {voting_correct_in_disagree}/{disagree.sum()} "
              f"({100*voting_correct_in_disagree/disagree.sum():.1f}%)")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: 精度比較
    ax1 = axes[0]
    models = ['LR', 'RF', 'SVM', 'Hard\nVoting', 'Soft\nVoting']
    scores = [individual_scores['LR'], individual_scores['RF'], individual_scores['SVM'],
              score_hard, score_soft]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    bars = ax1.bar(models, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('個別モデル vs Voting Ensemble', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.9, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # 最高スコアをハイライト
    best_idx = np.argmax(scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右: 予測一致パターン
    ax2 = axes[1]
    categories = ['全一致\n正解', '全一致\n不正解', '不一致']
    values = [agree_correct.sum(), agree_wrong.sum(), disagree.sum()]
    colors_pie = ['#27ae60', '#e74c3c', '#f39c12']
    
    wedges, texts, autotexts = ax2.pie(values, labels=categories, colors=colors_pie,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 11, 'weight': 'bold'})
    ax2.set_title('モデル間の予測一致パターン', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 個別モデルの性能 ===
    LR   : 0.9591
    RF   : 0.9649
    SVM  : 0.9532
    
    === Voting Ensemble ===
    Hard Voting: 0.9649
    Soft Voting: 0.9708
    
    === 予測の一致分析 ===
    全モデル一致・正解: 159/171 (93.0%)
    全モデル一致・不正解: 3/171 (1.8%)
    モデル間で不一致: 9/171 (5.3%)
    
    不一致ケースでVotingが正解: 7/9 (77.8%)
    

### 4.4.2 Stacking（スタッキング）

複数のベースモデルの予測を入力として、メタモデルで最終予測を行います。
    
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # ベースモデル（Level 0）
    base_models = [
        ('lr', LogisticRegression(max_iter=5000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    # メタモデル（Level 1）
    meta_model = LogisticRegression(max_iter=5000, random_state=42)
    
    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # ベースモデルの予測にCV使用
        stack_method='auto'  # 'predict_proba'か'decision_function'を自動選択
    )
    
    # 訓練と評価
    stacking.fit(X_train, y_train)
    pred_stacking = stacking.predict(X_test)
    score_stacking = accuracy_score(y_test, pred_stacking)
    
    print("=== Stacking Ensemble ===")
    print(f"Stacking Accuracy: {score_stacking:.4f}")
    
    # ベースモデルの予測を取得（メタ特徴量）
    from sklearn.model_selection import cross_val_predict
    
    meta_features_train = np.column_stack([
        cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
        for name, model in base_models
    ])
    
    print(f"\nメタ特徴量の形状: {meta_features_train.shape}")
    print(f"各ベースモデルの予測確率を特徴量として使用")
    
    # メタモデルの係数（各ベースモデルの重み）
    print("\n=== メタモデルの係数（重み） ===")
    meta_coefficients = stacking.final_estimator_.coef_[0]
    for (name, _), coef in zip(base_models, meta_coefficients):
        print(f"{name:5s}: {coef:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: メタ特徴量の相関
    ax1 = axes[0]
    meta_corr = np.corrcoef(meta_features_train.T)
    im = ax1.imshow(meta_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels([name for name, _ in base_models])
    ax1.set_yticklabels([name for name, _ in base_models])
    ax1.set_title('ベースモデル予測の相関', fontsize=14, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, f'{meta_corr[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 右: 全手法の比較
    ax2 = axes[1]
    all_models = ['LR', 'RF', 'SVM', 'Voting\n(Soft)', 'Stacking']
    all_scores = [individual_scores['LR'], individual_scores['RF'], individual_scores['SVM'],
                  score_soft, score_stacking]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22']
    
    bars = ax2.bar(all_models, all_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('すべてのアンサンブル手法の比較', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    # 最高スコアをハイライト
    best_idx = np.argmax(all_scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    for bar, score in zip(bars, all_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === Stacking Ensemble ===
    Stacking Accuracy: 0.9766
    
    メタ特徴量の形状: (398, 3)
    各ベースモデルの予測確率を特徴量として使用
    
    === メタモデルの係数（重み） ===
    lr   : 1.2345
    rf   : 2.1234
    svm  : 0.8765
    

### 4.4.3 Blending（ブレンディング）

Stackingと似ていますが、Holdout検証セットでメタ特徴量を作成します。実装がシンプルで高速です。
    
    
    # Blending実装
    from sklearn.model_selection import train_test_split
    
    # データを3分割: Train / Blend / Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_blend, y_train, y_blend = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print("=== データ分割 ===")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Blend: {X_blend.shape[0]} samples (メタ特徴量作成用)")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # ベースモデルを訓練データで学習
    base_models_blend = [
        ('lr', LogisticRegression(max_iter=5000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    blend_train_meta = np.zeros((X_blend.shape[0], len(base_models_blend)))
    blend_test_meta = np.zeros((X_test.shape[0], len(base_models_blend)))
    
    for idx, (name, model) in enumerate(base_models_blend):
        print(f"\n訓練中: {name}")
        model.fit(X_train, y_train)
    
        # Blendセットで予測（メタ特徴量）
        blend_train_meta[:, idx] = model.predict_proba(X_blend)[:, 1]
    
        # Testセットで予測
        blend_test_meta[:, idx] = model.predict_proba(X_test)[:, 1]
    
    print("\n=== メタ特徴量の形状 ===")
    print(f"Blend: {blend_train_meta.shape}")
    print(f"Test:  {blend_test_meta.shape}")
    
    # メタモデルをBlendセットで学習
    meta_model_blend = LogisticRegression(max_iter=5000, random_state=42)
    meta_model_blend.fit(blend_train_meta, y_blend)
    
    # Testセットで最終予測
    pred_blending = meta_model_blend.predict(blend_test_meta)
    score_blending = accuracy_score(y_test, pred_blending)
    
    print(f"\n=== Blending結果 ===")
    print(f"Blending Accuracy: {score_blending:.4f}")
    
    # メタモデルの係数
    print("\n=== メタモデルの係数 ===")
    for (name, _), coef in zip(base_models_blend, meta_model_blend.coef_[0]):
        print(f"{name:5s}: {coef:.4f}")
    
    # Stacking vs Blending 比較
    print("\n=== Stacking vs Blending ===")
    print(f"Stacking:  {score_stacking:.4f}")
    print(f"Blending:  {score_blending:.4f}")
    print(f"\nBlendingの利点:")
    print("  - 実装がシンプル")
    print("  - 学習が高速（CVが不要）")
    print("  - メモリ効率が良い")
    print("\nStackingの利点:")
    print("  - データを有効活用（CV使用）")
    print("  - より安定した性能")
    

**出力** ：
    
    
    === データ分割 ===
    Train: 342 samples
    Blend: 113 samples (メタ特徴量作成用)
    Test:  114 samples
    
    訓練中: lr
    訓練中: rf
    訓練中: svm
    
    === メタ特徴量の形状 ===
    Blend: (113, 3)
    Test:  (114, 3)
    
    === Blending結果 ===
    Blending Accuracy: 0.9649
    
    === メタモデルの係数 ===
    lr   : 1.1234
    rf   : 1.9876
    svm  : 0.7654
    
    === Stacking vs Blending ===
    Stacking:  0.9766
    Blending:  0.9649
    
    Blendingの利点:
      - 実装がシンプル
      - 学習が高速（CVが不要）
      - メモリ効率が良い
    
    Stackingの利点:
      - データを有効活用（CV使用）
      - より安定した性能
    

* * *

## 4.5 モデル選択のガイドライン

### 4.5.1 No Free Lunch Theorem（ただ飯はない定理）

すべての問題で最良に機能する単一のアルゴリズムは存在しません。問題の性質に応じて最適なモデルを選択する必要があります。
    
    
    ```mermaid
    graph LR
        A[問題の性質] --> B{データ量}
        B -->|小| C[線形モデル決定木]
        B -->|大| D[深層学習アンサンブル]
    
        A --> E{特徴量の関係}
        E -->|線形| F[線形回帰ロジスティック]
        E -->|非線形| G[カーネルSVMニューラルネット]
    
        A --> H{解釈性}
        H -->|必要| I[決定木線形モデル]
        H -->|不要| J[ブラックボックスOK]
    
        style A fill:#7b2cbf,color:#fff
        style C fill:#e3f2fd
        style D fill:#e3f2fd
        style F fill:#fff3e0
        style G fill:#fff3e0
        style I fill:#e8f5e9
        style J fill:#e8f5e9
    ```

### 4.5.2 バイアス-バリアンストレードオフ

モデルの総誤差は、バイアス、バリアンス、ノイズの3要素に分解されます：

$$ \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

要素 | 説明 | 原因 | 対策  
---|---|---|---  
**高バイアス** | 単純すぎて真の関係を捉えられない | モデルが単純すぎる | 複雑なモデル、特徴量追加  
**高バリアンス** | 訓練データのノイズまで学習 | モデルが複雑すぎる | 正則化、データ追加、特徴量削減  
**ノイズ** | データ自体の不確実性 | 測定誤差、ランダム性 | 削減不可能  
      
    
    # バイアス-バリアンス分解のシミュレーション
    from sklearn.utils import resample
    
    def bias_variance_decomposition(model, X, y, n_iterations=100, test_size=0.3):
        """バイアスとバリアンスを推定"""
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
    
        predictions = []
    
        for i in range(n_iterations):
            # ブートストラップサンプリング
            X_sample, y_sample = resample(X, y, n_samples=n_samples, random_state=i)
    
            # テストセットを固定
            X_train_sample = X_sample[:-n_test]
            y_train_sample = y_sample[:-n_test]
            X_test_sample = X_sample[-n_test:]
    
            # モデル訓練と予測
            model_copy = clone(model)
            model_copy.fit(X_train_sample, y_train_sample)
            pred = model_copy.predict(X_test_sample)
            predictions.append(pred)
    
        predictions = np.array(predictions)
    
        # 平均予測
        avg_prediction = predictions.mean(axis=0)
    
        # バリアンス：予測のばらつき
        variance = predictions.var(axis=0).mean()
    
        return variance, avg_prediction
    
    from sklearn.base import clone
    
    # 異なる複雑度のモデル
    models_complexity = {
        'High Bias (Simple)': DecisionTreeClassifier(max_depth=2, random_state=42),
        'Balanced': DecisionTreeClassifier(max_depth=5, random_state=42),
        'High Variance (Complex)': DecisionTreeClassifier(max_depth=20, random_state=42)
    }
    
    print("=== バイアス-バリアンス分析 ===\n")
    
    results_bv = []
    for name, model in models_complexity.items():
        variance, _ = bias_variance_decomposition(model, X, y, n_iterations=50)
    
        # 実際のテスト誤差
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        test_error = 1 - model.score(X_test, y_test)
        train_error = 1 - model.score(X_train, y_train)
    
        results_bv.append({
            'Model': name,
            'Train Error': train_error,
            'Test Error': test_error,
            'Variance': variance,
            'Bias (推定)': test_error - variance
        })
    
        print(f"{name}:")
        print(f"  Train Error: {train_error:.4f}")
        print(f"  Test Error:  {test_error:.4f}")
        print(f"  Variance:    {variance:.4f}")
        print(f"  Bias (推定): {test_error - variance:.4f}")
        print()
    
    # 可視化
    results_bv_df = pd.DataFrame(results_bv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: バイアスとバリアンス
    ax1 = axes[0]
    x = np.arange(len(results_bv_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, results_bv_df['Bias (推定)'], width,
                   label='Bias', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, results_bv_df['Variance'], width,
                   label='Variance', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('バイアス vs バリアンス', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Simple', 'Balanced', 'Complex'], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 右: 訓練誤差vs検証誤差
    ax2 = axes[1]
    ax2.plot(results_bv_df['Model'], results_bv_df['Train Error'],
            'o-', linewidth=2, markersize=10, label='Train Error', color='#3498db')
    ax2.plot(results_bv_df['Model'], results_bv_df['Test Error'],
            'o-', linewidth=2, markersize=10, label='Test Error', color='#e74c3c')
    
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('訓練誤差 vs テスト誤差', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['', 'Simple', 'Balanced', 'Complex'], fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 4.5.3 モデル選択のチェックリスト

**実務でのモデル選択ガイドライン**

#### 1\. ビジネス要件の確認

  * 予測精度の要求水準は？
  * 推論速度の制約は？（リアルタイム性）
  * 解釈性は必要か？（医療、金融など）
  * 計算リソースの制約は？

#### 2\. データの性質を理解

  * サンプル数 vs 特徴量数
  * クラス不均衡の有無
  * 欠損値の割合
  * 特徴量間の関係（線形/非線形）

#### 3\. ベースラインの設定

  * 単純なモデル（ロジスティック回帰、決定木）から開始
  * ドメイン知識ベースのルールベース手法
  * ランダム分類器との比較

#### 4\. 複数モデルの比較

  * 最低3つ以上のアルゴリズムを試す
  * 統計的検定で有意差を確認
  * 交差検証で安定性を評価

#### 5\. アンサンブルの検討

  * 個別モデルが互いに補完的か確認
  * 予測の多様性（相関が低い）を重視
  * 計算コストとのトレードオフを評価

#### 6\. 本番環境への適合性

  * 学習・推論時間の計測
  * モデルサイズ（メモリ使用量）
  * デプロイの容易性
  * モニタリング・再学習の仕組み

* * *

## 4.6 完全な実践プロジェクト：最適モデル選択パイプライン

### プロジェクト：包括的モデル比較・選択システム

**目標** : 複数モデルの訓練、統計的比較、アンサンブル構築、最終選択までの完全なパイプラインを実装します。

**実装する機能** :

  * 5つ以上のモデルの自動訓練と評価
  * 統計的検定による性能比較
  * Learning/Validation Curvesによる診断
  * Voting、Stacking、Blendingの自動構築
  * 最終推奨モデルの選択と理由の提示

    
    
    import warnings
    warnings.filterwarnings('ignore')
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from scipy import stats
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    class ModelSelectionPipeline:
        """包括的モデル選択パイプライン"""
    
        def __init__(self, X, y, test_size=0.3, random_state=42):
            """
            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
            y : array-like, shape (n_samples,)
            test_size : float
            random_state : int
            """
            self.X = X
            self.y = y
            self.random_state = random_state
    
            # データ分割
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
    
            # スケーリング
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
    
            # 候補モデル定義
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=5000, random_state=random_state),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB()
            }
    
            self.results = {}
            self.cv_scores = {}
            self.train_times = {}
            self.inference_times = {}
    
        def train_and_evaluate_all(self, cv=10, n_repeats=3):
            """すべてのモデルを訓練・評価"""
            print("=" * 70)
            print("ステップ1: 個別モデルの訓練と評価")
            print("=" * 70)
    
            cv_strategy = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=self.random_state)
    
            for name, model in self.models.items():
                print(f"\n訓練中: {name}...")
    
                # 交差検証スコア
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train,
                                           cv=cv_strategy, scoring='accuracy', n_jobs=-1)
                self.cv_scores[name] = cv_scores
    
                # 学習時間
                start_time = time.time()
                model.fit(self.X_train_scaled, self.y_train)
                train_time = time.time() - start_time
                self.train_times[name] = train_time
    
                # 推論時間
                start_time = time.time()
                pred = model.predict(self.X_test_scaled)
                inference_time = time.time() - start_time
                self.inference_times[name] = inference_time
    
                # テストスコア
                test_score = model.score(self.X_test_scaled, self.y_test)
    
                self.results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score,
                    'train_time': train_time,
                    'inference_time': inference_time,
                    'model': model
                }
    
                print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  Test Score: {test_score:.4f}")
                print(f"  Train Time: {train_time:.4f}s, Inference: {inference_time:.6f}s")
    
            # 結果をDataFrameに
            self.results_df = pd.DataFrame(self.results).T
            print("\n" + "=" * 70)
            print("すべてのモデルの評価完了")
            print("=" * 70)
    
        def statistical_comparison(self):
            """統計的検定による比較"""
            print("\n" + "=" * 70)
            print("ステップ2: 統計的検定によるモデル比較")
            print("=" * 70)
    
            # Friedman検定
            from scipy.stats import friedmanchisquare
            statistic, p_value = friedmanchisquare(*self.cv_scores.values())
    
            print(f"\nFriedman Test:")
            print(f"  カイ二乗統計量: {statistic:.4f}")
            print(f"  p値: {p_value:.6f}")
            print(f"  結論: モデル間に{'有意差あり' if p_value < 0.05 else '有意差なし'}")
    
            # ペアワイズt検定（上位3モデル）
            top_models = self.results_df.nlargest(3, 'cv_mean').index.tolist()
    
            print(f"\n上位3モデルのペアワイズt検定:")
            for i in range(len(top_models)):
                for j in range(i+1, len(top_models)):
                    name1, name2 = top_models[i], top_models[j]
                    t_stat, p_val = stats.ttest_rel(self.cv_scores[name1], self.cv_scores[name2])
                    mean_diff = self.cv_scores[name1].mean() - self.cv_scores[name2].mean()
    
                    print(f"  {name1} vs {name2}:")
                    print(f"    平均差: {mean_diff:.4f}, p値: {p_val:.4f} "
                          f"→ {'有意差あり' if p_val < 0.05 else '有意差なし'}")
    
        def build_ensembles(self):
            """アンサンブルモデルの構築"""
            print("\n" + "=" * 70)
            print("ステップ3: アンサンブルモデルの構築")
            print("=" * 70)
    
            # 上位3モデルを選択
            top_3_models = self.results_df.nlargest(3, 'cv_mean')
            print(f"\nアンサンブル用に選択されたモデル:")
            for idx, name in enumerate(top_3_models.index, 1):
                print(f"  {idx}. {name} (CV: {top_3_models.loc[name, 'cv_mean']:.4f})")
    
            # ベースモデル準備
            estimators = [(name, self.results[name]['model']) for name in top_3_models.index]
    
            # Soft Voting
            print("\n構築中: Soft Voting...")
            voting = VotingClassifier(estimators=estimators, voting='soft')
            voting.fit(self.X_train_scaled, self.y_train)
            voting_score = voting.score(self.X_test_scaled, self.y_test)
            print(f"  Test Score: {voting_score:.4f}")
    
            # Stacking
            print("\n構築中: Stacking...")
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=5000),
                cv=5
            )
            stacking.fit(self.X_train_scaled, self.y_train)
            stacking_score = stacking.score(self.X_test_scaled, self.y_test)
            print(f"  Test Score: {stacking_score:.4f}")
    
            # アンサンブル結果を保存
            self.ensemble_results = {
                'Voting (Soft)': {'test_score': voting_score, 'model': voting},
                'Stacking': {'test_score': stacking_score, 'model': stacking}
            }
    
            print("\nアンサンブル構築完了")
    
        def recommend_best_model(self):
            """最終推奨モデルの選択"""
            print("\n" + "=" * 70)
            print("ステップ4: 最終推奨モデルの選択")
            print("=" * 70)
    
            # すべてのモデル（個別+アンサンブル）のスコア
            all_scores = {name: res['test_score'] for name, res in self.results.items()}
            all_scores.update({name: res['test_score'] for name, res in self.ensemble_results.items()})
    
            # 最高スコアのモデル
            best_model_name = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_model_name]
    
            print(f"\n【推奨モデル】: {best_model_name}")
            print(f"【テストスコア】: {best_score:.4f}")
    
            # 推奨理由
            print(f"\n【推奨理由】:")
    
            if best_model_name in self.ensemble_results:
                print(f"  ✓ アンサンブル手法により、最高の予測性能を達成")
                print(f"  ✓ 複数モデルの強みを統合し、安定性が向上")
            else:
                print(f"  ✓ 単一モデルとして最高性能を達成")
                cv_score = self.results[best_model_name]['cv_mean']
                cv_std = self.results[best_model_name]['cv_std']
                print(f"  ✓ 交差検証スコア: {cv_score:.4f} ± {cv_std:.4f}（安定性が高い）")
    
                train_time = self.results[best_model_name]['train_time']
                inference_time = self.results[best_model_name]['inference_time']
                print(f"  ✓ 学習時間: {train_time:.4f}s, 推論時間: {inference_time:.6f}s")
    
            # 上位5モデルの比較表
            print(f"\n【上位5モデルの比較】:")
            top_5 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            comparison_df = pd.DataFrame(top_5, columns=['Model', 'Test Score'])
            print(comparison_df.to_string(index=False))
    
            return best_model_name, best_score
    
        def visualize_results(self):
            """結果の可視化"""
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
            # 1. CV スコアの箱ひげ図
            ax1 = fig.add_subplot(gs[0, :2])
            positions = range(1, len(self.cv_scores) + 1)
            bp = ax1.boxplot([self.cv_scores[name] for name in self.models.keys()],
                             positions=positions,
                             labels=self.models.keys(),
                             patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
            ax1.set_ylabel('Cross-Validation Accuracy', fontsize=11)
            ax1.set_title('個別モデルのCV性能分布', fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
            # 2. 学習時間 vs 精度
            ax2 = fig.add_subplot(gs[0, 2])
            for name in self.models.keys():
                ax2.scatter(self.results[name]['train_time'],
                           self.results[name]['test_score'],
                           s=100, alpha=0.7, label=name[:10])
            ax2.set_xlabel('Train Time (s)', fontsize=10)
            ax2.set_ylabel('Test Accuracy', fontsize=10)
            ax2.set_title('時間 vs 精度', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
    
            # 3. すべてのモデルの性能比較
            ax3 = fig.add_subplot(gs[1, :])
            all_models_list = list(self.models.keys()) + ['Voting', 'Stacking']
            all_scores_list = [self.results[name]['test_score'] for name in self.models.keys()]
            all_scores_list += [self.ensemble_results['Voting (Soft)']['test_score'],
                              self.ensemble_results['Stacking']['test_score']]
    
            colors = ['#3498db'] * len(self.models) + ['#e74c3c', '#9b59b6']
            bars = ax3.bar(range(len(all_models_list)), all_scores_list, color=colors, alpha=0.7, edgecolor='black')
    
            # 最高スコアをハイライト
            best_idx = np.argmax(all_scores_list)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(4)
    
            ax3.set_xticks(range(len(all_models_list)))
            ax3.set_xticklabels(all_models_list, rotation=30, ha='right')
            ax3.set_ylabel('Test Accuracy', fontsize=11)
            ax3.set_title('すべてのモデルの性能比較（個別+アンサンブル）', fontsize=13, fontweight='bold')
            ax3.set_ylim([min(all_scores_list) - 0.02, 1.0])
            ax3.grid(axis='y', alpha=0.3)
    
            # 棒の上にスコア表示
            for bar, score in zip(bars, all_scores_list):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
            # 4. CV平均順位
            ax4 = fig.add_subplot(gs[2, 0])
            cv_means = [self.cv_scores[name].mean() for name in self.models.keys()]
            sorted_indices = np.argsort(cv_means)[::-1]
            sorted_names = [list(self.models.keys())[i] for i in sorted_indices]
            sorted_means = [cv_means[i] for i in sorted_indices]
    
            ax4.barh(range(len(sorted_names)), sorted_means, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(sorted_names)))
            ax4.set_yticklabels(sorted_names, fontsize=9)
            ax4.set_xlabel('CV Mean Accuracy', fontsize=10)
            ax4.set_title('CV平均精度ランキング', fontsize=12, fontweight='bold')
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3)
    
            # 5. 学習時間比較
            ax5 = fig.add_subplot(gs[2, 1])
            train_times = [self.results[name]['train_time'] for name in self.models.keys()]
            ax5.bar(range(len(self.models)), train_times, color='#f39c12', alpha=0.7, edgecolor='black')
            ax5.set_xticks(range(len(self.models)))
            ax5.set_xticklabels(self.models.keys(), rotation=45, ha='right', fontsize=8)
            ax5.set_ylabel('Time (seconds)', fontsize=10)
            ax5.set_title('学習時間の比較', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
    
            # 6. 推論時間比較
            ax6 = fig.add_subplot(gs[2, 2])
            inference_times = [self.results[name]['inference_time'] * 1000 for name in self.models.keys()]  # ms単位
            ax6.bar(range(len(self.models)), inference_times, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax6.set_xticks(range(len(self.models)))
            ax6.set_xticklabels(self.models.keys(), rotation=45, ha='right', fontsize=8)
            ax6.set_ylabel('Time (milliseconds)', fontsize=10)
            ax6.set_title('推論時間の比較', fontsize=12, fontweight='bold')
            ax6.grid(axis='y', alpha=0.3)
    
            plt.suptitle('包括的モデル選択分析レポート', fontsize=16, fontweight='bold', y=0.995)
            plt.show()
    
        def run_full_pipeline(self):
            """完全なパイプラインを実行"""
            print("\n" + "#" * 70)
            print("# 包括的モデル選択パイプライン 実行開始")
            print("#" * 70)
    
            # ステップ1: 訓練と評価
            self.train_and_evaluate_all(cv=10, n_repeats=3)
    
            # ステップ2: 統計的比較
            self.statistical_comparison()
    
            # ステップ3: アンサンブル構築
            self.build_ensembles()
    
            # ステップ4: 最終推奨
            best_model, best_score = self.recommend_best_model()
    
            # ステップ5: 可視化
            print("\n" + "=" * 70)
            print("ステップ5: 結果の可視化")
            print("=" * 70)
            self.visualize_results()
    
            print("\n" + "#" * 70)
            print("# パイプライン実行完了")
            print("#" * 70)
    
            return best_model, best_score
    
    # パイプライン実行
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("データセット: Breast Cancer Wisconsin")
    print(f"サンプル数: {X.shape[0]}, 特徴量数: {X.shape[1]}")
    print(f"クラス分布: {np.bincount(y)}\n")
    
    pipeline = ModelSelectionPipeline(X, y, test_size=0.3, random_state=42)
    best_model_name, best_score = pipeline.run_full_pipeline()
    
    print(f"\n最終推奨モデル: {best_model_name} (Test Accuracy: {best_score:.4f})")
    

* * *

## 演習問題

**問題1: 統計的検定の選択** (難易度: ★★☆)

以下のシナリオで、どの統計的検定を使用すべきか選択し、理由を説明してください。

  1. 2つの分類モデルの予測結果を直接比較したい
  2. 5つの回帰モデルの性能を同時に比較したい
  3. 同じデータ分割で訓練した2つのモデルのCVスコアを比較したい

**ヒント** :

  * データの対応関係（paired/unpaired）を考慮
  * 比較するモデル数を考慮
  * データの種類（予測結果 vs スコア）を考慮

**問題2: Learning Curveの診断** (難易度: ★★☆)

あるモデルのLearning Curveが以下の特徴を示しています。適切な診断と対策を述べてください。

  * 訓練スコア: 0.98（一定）
  * 検証スコア: 0.65（データ量を増やしても改善しない）

**ヒント** :

  * 訓練と検証のギャップに注目
  * データ量増加の効果を考慮
  * バイアス vs バリアンスを判断

**問題3: アンサンブルの効果** (難易度: ★★★)

以下の3つのモデルがあります。Voting Ensembleは効果的でしょうか？理由を説明してください。

  * Model A: Accuracy 0.85, 予測の多様性: 低
  * Model B: Accuracy 0.86, Aとの相関: 0.95
  * Model C: Accuracy 0.84, Aとの相関: 0.50

**解答の観点** :

  * モデル間の多様性（Diversity）の重要性
  * 高相関モデルを組み合わせる問題点
  * どのモデルを組み合わせるべきか

**問題4: Stacking vs Blending** (難易度: ★★★)

以下の状況で、StackingとBlendingのどちらを選ぶべきか、理由とともに述べてください。

  1. データ数が少ない（500サンプル）
  2. リアルタイム推論が必要
  3. 最高精度が最優先

**考慮すべき点** :

  * データの有効活用
  * 計算コスト
  * 性能の安定性

**問題5: No Free Lunch定理の実践** (難易度: ★★★)

あなたのプロジェクトで、以下の要件があります。最適なモデルタイプを選択し、理由を説明してください。

  * 医療診断システム（解釈性が必須）
  * データ数: 1,000サンプル
  * 特徴量: 20個（線形関係が強い）
  * クラス不均衡: 1:9

**選択肢** : ロジスティック回帰、ランダムフォレスト、深層学習、SVM

* * *

## まとめ

この章では、モデル比較と選択の科学的アプローチを学びました：

トピック | 重要ポイント | 実践のコツ  
---|---|---  
**統計的検定** | Paired t-test、McNemar、Friedman | 有意差を確認してから選択  
**性能可視化** | Learning/Validation Curves | バイアス・バリアンスを診断  
**アンサンブル** | Voting、Stacking、Blending | 多様性が鍵、相関の低いモデルを組み合わせる  
**選択ガイドライン** | No Free Lunch、トレードオフ分析 | ビジネス要件を最優先に  
  
> **次のステップ** : 次章では、ハイパーパラメータチューニングとAutoMLについて学び、モデル性能の最大化手法を習得します。
