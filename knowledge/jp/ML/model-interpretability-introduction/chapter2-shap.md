---
title: 第2章：SHAP (SHapley Additive exPlanations)
chapter_title: 第2章：SHAP (SHapley Additive exPlanations)
subtitle: ゲーム理論に基づく統一的特徴量重要度
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ゲーム理論におけるShapley値の概念と性質を理解する
  * ✅ SHAP値の数学的定式化と公理的特性を説明できる
  * ✅ TreeSHAP、KernelSHAPなどのアルゴリズムを理解する
  * ✅ SHAPライブラリを用いた実装と可視化ができる
  * ✅ Waterfall、Force、Summary plotsなどの解釈手法を使える
  * ✅ SHAPの応用範囲と限界を理解する

* * *

## 2.1 Shapley値の理論

### ゲーム理論の基礎

**Shapley値** は、協力ゲーム理論（Cooperative Game Theory）から生まれた概念で、プレイヤーの貢献度を公平に評価する手法です。

#### 協力ゲームの定義

協力ゲーム $(N, v)$ は以下で定義されます：

  * $N = \\{1, 2, \ldots, n\\}$：プレイヤーの集合
  * $v: 2^N \rightarrow \mathbb{R}$：特性関数（coalitional value function）
  * $v(S)$：連携 $S \subseteq N$ の価値

条件：

  * $v(\emptyset) = 0$（空集合の価値は0）
  * $v(N)$：全員が協力したときの総価値

#### 機械学習への適用

機械学習における解釈問題では：

  * **プレイヤー** ：特徴量（features）
  * **価値** ：予測値（prediction）
  * **連携** ：特徴量の部分集合

    
    
    ```mermaid
    graph TB
        GameTheory["協力ゲーム理論N: プレイヤー, v: 価値関数"] --> Shapley["Shapley値公平な貢献度分配"]
        Shapley --> ML["機械学習解釈特徴量の寄与度"]
        ML --> SHAP["SHAP統一的解釈フレームワーク"]
    
        style GameTheory fill:#b3e5fc
        style Shapley fill:#c5e1a5
        style ML fill:#fff9c4
        style SHAP fill:#ffab91
    ```

### Shapley値の定義

プレイヤー $i$ の**Shapley値** は、すべての可能な連携への貢献の平均として定義されます：

$$ \phi_i(v) = \sum_{S \subseteq N \setminus \\{i\\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \\{i\\}) - v(S) \right] $$ 

ここで：

  * $S$：プレイヤー $i$ を含まない連携
  * $v(S \cup \\{i\\}) - v(S)$：プレイヤー $i$ の限界貢献（marginal contribution）
  * $\frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!}$：重み（すべての順序を考慮）

### Shapley値の性質（公理）

Shapley値は、以下の4つの公理を満たす**唯一の解** です：

公理 | 数学的表現 | 意味  
---|---|---  
**効率性（Efficiency）** | $\sum_{i=1}^{n} \phi_i(v) = v(N)$ | 全プレイヤーの貢献の合計 = 総価値  
**対称性（Symmetry）** | $v(S \cup \\{i\\}) = v(S \cup \\{j\\})$ ならば $\phi_i = \phi_j$ | 同じ貢献なら同じ報酬  
**ダミー性（Dummy）** | $v(S \cup \\{i\\}) = v(S)$ ならば $\phi_i = 0$ | 貢献がないなら報酬もゼロ  
**加法性（Additivity）** | $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$ | 独立したゲームは分解可能  
  
### 計算の複雑性

Shapley値の正確な計算は、$2^n$ 個の連携を評価する必要があるため、**指数時間計算量** $O(2^n)$ となります。
    
    
    import numpy as np
    from itertools import combinations
    
    def shapley_value_exact(n, value_function):
        """
        Shapley値の厳密計算（小規模な特徴量数のみ）
    
        Args:
            n: 特徴量数
            value_function: 部分集合 S に対する価値を返す関数
    
        Returns:
            shapley_values: (n,) Shapley値
        """
        players = list(range(n))
        shapley_values = np.zeros(n)
    
        # 各プレイヤーについて
        for i in players:
            # プレイヤーiを除く他のプレイヤー
            others = [p for p in players if p != i]
    
            # すべての可能な連携Sについて
            for r in range(len(others) + 1):
                for S in combinations(others, r):
                    S = list(S)
    
                    # 限界貢献: v(S ∪ {i}) - v(S)
                    marginal_contribution = value_function(S + [i]) - value_function(S)
    
                    # 重み: |S|! * (n - |S| - 1)! / n!
                    weight = (np.math.factorial(len(S)) *
                             np.math.factorial(n - len(S) - 1) /
                             np.math.factorial(n))
    
                    shapley_values[i] += weight * marginal_contribution
    
        return shapley_values
    
    
    # 簡単な例：3特徴量
    print("=== Shapley値の厳密計算 ===")
    n = 3
    
    # 価値関数の例（線形モデル）
    def value_func(S):
        """特徴量の部分集合Sに対する予測値"""
        # 簡略化：各特徴量の重みが [1, 2, 3]
        weights = np.array([1.0, 2.0, 3.0])
        if len(S) == 0:
            return 0.0
        return weights[S].sum()
    
    # Shapley値を計算
    shapley = shapley_value_exact(n, value_func)
    
    print(f"特徴量数: {n}")
    print(f"Shapley値: {shapley}")
    print(f"合計: {shapley.sum()} (= v(N) = {value_func([0, 1, 2])})")
    print("\n→ 効率性公理を満たす：合計 = 総価値")
    

### 計算量の問題
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 計算量の可視化
    n_features = np.arange(1, 21)
    num_coalitions = 2 ** n_features
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(n_features, num_coalitions, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Number of Coalitions (log scale)', fontsize=12)
    plt.title('Computational Complexity of Exact Shapley Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 参考線
    plt.axhline(y=1e6, color='r', linestyle='--', label='1 million')
    plt.axhline(y=1e9, color='orange', linestyle='--', label='1 billion')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('shapley_complexity.png', dpi=150, bbox_inches='tight')
    print("計算量の図を保存: shapley_complexity.png")
    plt.close()
    
    print("\n=== 計算量の例 ===")
    for n in [5, 10, 15, 20, 30]:
        coalitions = 2 ** n
        print(f"特徴量数 {n:2d}: {coalitions:,} 個の連携")
    
    print("\n→ 特徴量が増えると計算が不可能になる")
    print("→ 近似アルゴリズム（KernelSHAP, TreeSHAPなど）が必要")
    

* * *

## 2.2 SHAP値

### Additive Feature Attribution

**SHAP (SHapley Additive exPlanations)** は、Shapley値を機械学習の解釈に適用したフレームワークです。

予測値 $f(x)$ に対する説明モデル $g$ は、以下の形式を持ちます：

$$ g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i $$ 

ここで：

  * $z' \in \\{0, 1\\}^M$：簡略化された入力（特徴量の有無）
  * $\phi_i$：特徴量 $i$ のSHAP値（寄与度）
  * $\phi_0 = \mathbb{E}[f(X)]$：ベース値（全体の期待値）

### SHAP値の定義

入力 $x$ に対する特徴量 $i$ のSHAP値は：

$$ \phi_i(f, x) = \sum_{S \subseteq F \setminus \\{i\\}} \frac{|S|! \cdot (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \\{i\\}) - f_x(S) \right] $$ 

ここで：

  * $F = \\{1, 2, \ldots, M\\}$：すべての特徴量の集合
  * $f_x(S)$：特徴量の部分集合 $S$ のみを使った予測（他は期待値で置換）

### SHAP値の計算

実際には、欠損した特徴量を**期待値で条件付けて** 計算します：

$$ f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S] $$ 

この期待値を計算する方法によって、異なるSHAPアルゴリズムが存在します。
    
    
    ```mermaid
    graph TB
        SHAP["SHAP統一フレームワーク"] --> TreeSHAP["TreeSHAP決定木モデル多項式時間"]
        SHAP --> KernelSHAP["KernelSHAP任意のモデルサンプリング近似"]
        SHAP --> DeepSHAP["DeepSHAPディープラーニング勾配ベース"]
        SHAP --> LinearSHAP["LinearSHAP線形モデル解析的計算"]
    
        style SHAP fill:#ffab91
        style TreeSHAP fill:#c5e1a5
        style KernelSHAP fill:#fff9c4
        style DeepSHAP fill:#b3e5fc
    ```

### TreeSHAPアルゴリズム

**TreeSHAP** は、決定木ベースのモデル（決定木、ランダムフォレスト、XGBoost、LightGBMなど）に対する効率的なアルゴリズムです。

特徴：

  * 計算量：$O(TLD^2)$（$T$: 木の数、$L$: 葉の数、$D$: 深さ）
  * 厳密なShapley値を計算
  * 木構造を利用した高速化

### KernelSHAP

**KernelSHAP** は、任意のモデルに適用可能な近似アルゴリズムです。

アイデア：

  1. ランダムに特徴量の部分集合をサンプリング
  2. 各部分集合で予測を計算
  3. 重み付き線形回帰でSHAP値を推定

重み関数：

$$ \pi_{x}(z') = \frac{(M-1)}{\binom{M}{|z'|} |z'|(M - |z'|)} $$ 

* * *

## 2.3 SHAP可視化

### Waterfall Plots

**Waterfall plot** は、1つのサンプルについて、ベース値から最終予測値までの各特徴量の寄与を滝のように表示します。
    
    
    import shap
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    
    # データセットの読み込み
    print("=== Breast Cancerデータセットの準備 ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    print(f"データサイズ: {X.shape}")
    print(f"特徴量数: {X.shape[1]}")
    
    # モデルの訓練
    print("\n=== ランダムフォレストの訓練 ===")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)
    
    train_acc = model.score(X, y)
    print(f"訓練精度: {train_acc:.4f}")
    
    # TreeExplainerの作成
    print("\n=== TreeExplainerの作成 ===")
    explainer = shap.TreeExplainer(model)
    
    # 1つのサンプルのSHAP値を計算
    sample_idx = 0
    shap_values = explainer(X.iloc[[sample_idx]])
    
    print(f"サンプル {sample_idx} のSHAP値:")
    print(f"  形状: {shap_values.values.shape}")
    print(f"  ベース値: {shap_values.base_values[0]:.4f}")
    print(f"  予測値: {shap_values.base_values[0] + shap_values.values[0].sum():.4f}")
    
    # Waterfall plotの作成
    print("\n=== Waterfall Plotの作成 ===")
    shap.plots.waterfall(shap_values[0], show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
    print("Waterfall plotを保存: shap_waterfall.png")
    plt.close()
    

### Force Plots

**Force plot** は、Waterfall plotと同様に1サンプルの説明を視覚化しますが、正負の寄与を横に並べて表示します。
    
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Force Plotの作成 ===")
    
    # Force plot（静的版）
    shap.plots.force(
        shap_values.base_values[0],
        shap_values.values[0],
        X.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_force.png', dpi=150, bbox_inches='tight')
    print("Force plotを保存: shap_force.png")
    plt.close()
    
    print("\n→ 赤色: 予測を増加させる特徴量")
    print("→ 青色: 予測を減少させる特徴量")
    

### Summary Plots

**Summary plot** は、すべてのサンプルのSHAP値を集約して、各特徴量の重要度と効果を可視化します。
    
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Summary Plotの作成 ===")
    
    # 全サンプルのSHAP値を計算（時間がかかる場合はサブセット使用）
    shap_values_all = explainer(X[:100])  # 最初の100サンプル
    
    # Summary plot（bee swarm plot）
    shap.summary_plot(shap_values_all, X[:100], show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    print("Summary plotを保存: shap_summary.png")
    plt.close()
    
    print("\n→ 各点は1サンプル")
    print("→ 横軸: SHAP値（寄与度）")
    print("→ 色: 特徴量の値（赤=高、青=低）")
    

### Bar Plots（特徴量重要度）
    
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Bar Plot（特徴量重要度）===")
    
    # 平均絶対SHAP値でソート
    shap.plots.bar(shap_values_all, show=False)
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
    print("Bar plotを保存: shap_bar.png")
    plt.close()
    
    # 手動で計算
    mean_abs_shap = np.abs(shap_values_all.values).mean(axis=0)
    feature_importance = sorted(
        zip(X.columns, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n特徴量重要度（上位10）:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
    

### Dependence Plots

**Dependence plot** は、1つの特徴量の値とそのSHAP値の関係を散布図で表示します。
    
    
    import shap
    import matplotlib.pyplot as plt
    
    print("\n=== Dependence Plotの作成 ===")
    
    # 最も重要な特徴量を選択
    top_feature = feature_importance[0][0]
    print(f"分析対象: {top_feature}")
    
    # Dependence plot
    shap.dependence_plot(
        top_feature,
        shap_values_all.values,
        X[:100],
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
    print("Dependence plotを保存: shap_dependence.png")
    plt.close()
    
    print("\n→ 横軸: 特徴量の値")
    print("→ 縦軸: SHAP値（その特徴量の寄与度）")
    print("→ 非線形関係や相互作用を発見できる")
    

* * *

## 2.4 SHAPライブラリの実践

### TreeExplainer（決定木モデル）

**TreeExplainer** は、決定木ベースのモデル（RandomForest、XGBoost、LightGBM、CatBoostなど）に対する最も効率的なExplainerです。
    
    
    import shap
    import xgboost as xgb
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # データ準備（Boston Housing - 回帰タスク）
    print("=== Boston Housingデータセット ===")
    # Note: load_boston is deprecated, using California housing as alternative
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    
    print(f"データサイズ: {X.shape}")
    print(f"特徴量: {list(X.columns)}")
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # XGBoostモデルの訓練
    print("\n=== XGBoostモデルの訓練 ===")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"訓練 R²: {train_score:.4f}")
    print(f"テスト R²: {test_score:.4f}")
    
    # TreeExplainerの使用
    print("\n=== TreeExplainerの使用 ===")
    explainer = shap.TreeExplainer(model)
    
    # SHAP値の計算
    shap_values = explainer(X_test)
    
    print(f"SHAP値の形状: {shap_values.values.shape}")
    print(f"ベース値: {shap_values.base_values[0]:.4f}")
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig('xgboost_shap_summary.png', dpi=150, bbox_inches='tight')
    print("Summary plotを保存: xgboost_shap_summary.png")
    plt.close()
    

### LinearExplainer（線形モデル）

**LinearExplainer** は、線形モデル（線形回帰、ロジスティック回帰など）に対する解析的なExplainerです。
    
    
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("\n=== LinearExplainer（ロジスティック回帰）===")
    
    # Irisデータセット
    X, y = load_iris(return_X_y=True, as_frame=True)
    # 2クラス分類に簡略化
    X = X[y != 2]
    y = y[y != 2]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # ロジスティック回帰
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    print(f"テスト精度: {model.score(X_test, y_test):.4f}")
    
    # LinearExplainer
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)
    
    print(f"SHAP値の形状: {shap_values.values.shape}")
    
    # Waterfall plot（1サンプル）
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig('linear_shap_waterfall.png', dpi=150, bbox_inches='tight')
    print("Waterfall plotを保存: linear_shap_waterfall.png")
    plt.close()
    

### KernelExplainer（任意のモデル）

**KernelExplainer** は、任意のモデル（ブラックボックスモデル含む）に適用できますが、計算コストが高いです。
    
    
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_wine
    import numpy as np
    
    print("\n=== KernelExplainer（任意のモデル）===")
    
    # Wineデータセット
    X, y = load_wine(return_X_y=True, as_frame=True)
    
    # モデル訓練
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    print(f"訓練精度: {model.score(X, y):.4f}")
    
    # KernelExplainer（計算コストが高いので小サンプル使用）
    print("\n=== KernelExplainerの作成（背景データ: 50サンプル）===")
    background = shap.sample(X, 50)  # 背景データ
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # SHAP値の計算（3サンプルのみ）
    print("SHAP値を計算中（時間がかかります）...")
    test_samples = X.iloc[:3]
    shap_values = explainer.shap_values(test_samples)
    
    print(f"SHAP値の形状: {np.array(shap_values).shape}")
    print("→ (クラス数, サンプル数, 特徴量数)")
    
    # Force plot（クラス0）
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        test_samples.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('kernel_shap_force.png', dpi=150, bbox_inches='tight')
    print("Force plotを保存: kernel_shap_force.png")
    plt.close()
    
    print("\n→ KernelSHAPは遅いが、任意のモデルに適用可能")
    

### DeepExplainer（ニューラルネットワーク）

**DeepExplainer** は、ディープラーニングモデル（TensorFlow、PyTorchなど）に対する効率的なExplainerです。
    
    
    import shap
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("\n=== DeepExplainer（PyTorchニューラルネットワーク）===")
    
    # データ生成
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # PyTorchデータに変換
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    
    # 簡単なニューラルネットワーク
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNN(input_dim=20)
    
    # 訓練（簡略版）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # 評価
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
        print(f"テスト精度: {accuracy:.4f}")
    
    # DeepExplainer
    print("\n=== DeepExplainerの使用 ===")
    background = X_train_t[:100]  # 背景データ
    explainer = shap.DeepExplainer(model, background)
    
    # SHAP値の計算
    test_samples = X_test_t[:10]
    shap_values = explainer.shap_values(test_samples)
    
    print(f"SHAP値の形状: {np.array(shap_values).shape}")
    
    # Summary plot（クラス1）
    shap.summary_plot(
        shap_values[1],
        test_samples.numpy(),
        show=False
    )
    plt.tight_layout()
    plt.savefig('deep_shap_summary.png', dpi=150, bbox_inches='tight')
    print("Summary plotを保存: deep_shap_summary.png")
    plt.close()
    

* * *

## 2.5 SHAPの応用と限界

### モデル診断

SHAPを使って、モデルの問題点を診断できます：
    
    
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("=== SHAPによるモデル診断 ===")
    
    # 意図的にバイアスのあるデータを作成
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    
    # 特徴量0と1だけが真に重要
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # ノイズ特徴量を追加（高い相関を持つ）
    X[:, 2] = X[:, 0] + np.random.randn(n_samples) * 0.1  # 特徴量0と相関
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"訓練精度: {model.score(X, y):.4f}")
    
    # SHAP分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # 特徴量重要度
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), mean_abs_shap)
    plt.xlabel('Feature Index')
    plt.ylabel('Mean |SHAP value|')
    plt.title('Feature Importance via SHAP')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shap_diagnosis.png', dpi=150, bbox_inches='tight')
    print("診断結果を保存: shap_diagnosis.png")
    plt.close()
    
    print("\n特徴量重要度:")
    for i, importance in enumerate(mean_abs_shap):
        print(f"  特徴量 {i}: {importance:.4f}")
    
    print("\n→ 特徴量2（相関ノイズ）が重要と誤認される可能性")
    print("→ SHAPで多重共線性の問題を発見")
    

### 特徴量選択

SHAPを使って、真に重要な特徴量を選択できます：
    
    
    import shap
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import cross_val_score
    
    print("\n=== SHAPによる特徴量選択 ===")
    
    # データ生成（20特徴量、うち5つだけが重要）
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=5,
        noise=10.0,
        random_state=42
    )
    
    # 初期モデル
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # SHAP値で特徴量重要度を計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # 重要度でソート
    feature_importance = sorted(
        enumerate(mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("特徴量重要度（SHAP）:")
    for i, (idx, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. 特徴量 {idx:2d}: {importance:.4f}")
    
    # 上位k個の特徴量で再訓練
    for k in [5, 10, 15, 20]:
        top_features = [idx for idx, _ in feature_importance[:k]]
        X_selected = X[:, top_features]
    
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X_selected, y, cv=5, scoring='r2'
        )
    
        print(f"\n上位 {k:2d} 特徴量: CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    
    print("\n→ 上位5特徴量で十分な性能")
    print("→ SHAPで効率的な特徴量選択が可能")
    

### 計算コスト

SHAPの計算コストは、アルゴリズムとモデルによって大きく異なります：
    
    
    import shap
    import numpy as np
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("\n=== SHAPの計算コスト比較 ===")
    
    # データサイズを変えて計算時間を測定
    results = []
    
    for n_samples in [100, 500, 1000, 2000]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            random_state=42
        )
    
        # モデル訓練
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
    
        # TreeExplainer
        explainer_tree = shap.TreeExplainer(model)
        start = time.time()
        shap_values_tree = explainer_tree(X)
        time_tree = time.time() - start
    
        # KernelExplainer（小サンプルのみ）
        if n_samples <= 500:
            background = shap.sample(X, 50)
            explainer_kernel = shap.KernelExplainer(model.predict_proba, background)
            test_samples = X[:10]  # 10サンプルのみ
            start = time.time()
            shap_values_kernel = explainer_kernel.shap_values(test_samples)
            time_kernel = time.time() - start
        else:
            time_kernel = np.nan
    
        results.append((n_samples, time_tree, time_kernel))
        print(f"サンプル数 {n_samples:4d}: TreeSHAP={time_tree:.3f}s, KernelSHAP={time_kernel:.3f}s")
    
    print("\n→ TreeSHAPは大規模データでも高速")
    print("→ KernelSHAPは小規模データのみ実用的")
    

### 解釈の注意点

注意点 | 説明 | 対策  
---|---|---  
**多重共線性** | 相関する特徴量間でSHAP値が不安定 | 相関分析と併用、特徴量の事前選択  
**背景データの選択** | KernelSHAPの結果が背景データに依存 | 代表的なサンプルを選択、複数の背景で検証  
**外挿** | 訓練データ外の領域で信頼性低下 | 予測の信頼区間と併用  
**因果関係** | SHAP値は相関であり因果ではない | 因果推論手法と併用  
  
* * *

## 演習問題

**演習1：TreeSHAPとKernelSHAPの比較**

同じデータセットとモデルで、TreeSHAPとKernelSHAPのSHAP値を計算し、結果を比較してください。どの程度一致しますか？
    
    
    import shap
    from sklearn.ensemble import RandomForestClassifier
    
    # TODO: データとモデルを準備
    # TODO: TreeSHAPとKernelSHAPで同じサンプルのSHAP値を計算
    # TODO: SHAP値の相関係数を計算
    # TODO: 散布図で比較
    # 期待: ほぼ一致するが、KernelSHAPは近似のため若干異なる
    

**演習2：多重共線性の影響調査**

意図的に相関する特徴量を作成し、SHAPでどのように解釈されるか調査してください。
    
    
    import numpy as np
    import shap
    
    # TODO: X1とX2を作成（X2 = X1 + ノイズ）
    # TODO: y = f(X1)として、X2は真には関係ない
    # TODO: モデルを訓練し、SHAPで分析
    # TODO: X1とX2のSHAP値を比較
    # 分析: 相関する特徴量間でSHAP値が分散される
    

**演習3：SHAP値による異常検知**

正常サンプルと異常サンプルのSHAP値の分布を比較し、異常の原因となる特徴量を特定してください。
    
    
    import shap
    from sklearn.ensemble import IsolationForest
    
    # TODO: 正常データと異常データを作成
    # TODO: IsolationForestで異常検知
    # TODO: 異常サンプルのSHAP値を分析
    # TODO: どの特徴量が異常の原因か特定
    # ヒント: Summary plotやDependence plotを活用
    

**演習4：時系列データでのSHAP応用**

時系列データ（例：過去N日のデータから翌日を予測）に対してSHAPを適用し、どの時点の情報が重要か分析してください。
    
    
    import numpy as np
    import shap
    
    # TODO: 時系列データを生成（例：株価、気温など）
    # TODO: ラグ特徴量を作成（t-1, t-2, ..., t-N）
    # TODO: モデルで予測
    # TODO: SHAPで各時点の重要度を分析
    # 期待: 直近の時点がより重要（一般的に）
    

**演習5：SHAPとPFI（Permutation Feature Importance）の比較**

SHAPによる特徴量重要度と、Permutation Feature Importanceを比較し、違いを分析してください。
    
    
    import shap
    from sklearn.inspection import permutation_importance
    
    # TODO: モデルを訓練
    # TODO: SHAPで特徴量重要度を計算
    # TODO: Permutation Importanceを計算
    # TODO: 2つの重要度の相関を分析
    # TODO: 違いが大きい特徴量を調査
    # 分析: SHAPは局所的、PFIは大域的な重要度
    

* * *

## まとめ

この章では、SHAP (SHapley Additive exPlanations)の理論と実践を学びました。

### 重要ポイント

  * **Shapley値** ：協力ゲーム理論に基づく公平な貢献度評価
  * **公理的性質** ：効率性、対称性、ダミー性、加法性を満たす唯一の解
  * **SHAP** ：Shapley値を機械学習の解釈に適用した統一フレームワーク
  * **TreeSHAP** ：決定木モデルに対する効率的な厳密計算
  * **KernelSHAP** ：任意のモデルに適用可能な近似手法
  * **可視化** ：Waterfall、Force、Summary、Dependence plotsで多角的に分析
  * **応用** ：モデル診断、特徴量選択、異常検知など
  * **限界** ：計算コスト、多重共線性、因果関係の解釈には注意が必要

### SHAPの利点と限界

項目 | 利点 | 限界  
---|---|---  
**理論的基盤** | ゲーム理論の確固たる基礎 | 計算量の問題（厳密計算は困難）  
**一貫性** | 公理的性質により一貫した解釈 | 多重共線性で不安定  
**適用範囲** | 任意のモデルに適用可能 | モデルによって計算コストが大きく異なる  
**解釈性** | 局所的説明と大域的重要度の両方 | 相関と因果の混同リスク  
  
### 次のステップ

次章では、**Integrated Gradients** と**Attention機構** について学びます。ディープラーニングモデルの解釈手法、勾配ベースの属性分析、Transformerモデルの注意機構の可視化など、より高度な解釈技術を習得します。

### 参考文献

  * Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." _NeurIPS_.
  * Shapley, L. S. (1953). "A value for n-person games." _Contributions to the Theory of Games_ , 2(28), 307-317.
  * Lundberg, S. M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." _Nature Machine Intelligence_ , 2(1), 2522-5839.
  * Molnar, C. (2022). "Interpretable Machine Learning." <https://christophm.github.io/interpretable-ml-book/>
