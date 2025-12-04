---
title: 第3章：LIMEとその他の解釈手法
chapter_title: 第3章：LIMEとその他の解釈手法
subtitle: 局所的・大域的解釈のための多様なアプローチ
reading_time: 30-35分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ LIMEの原理と局所線形近似の仕組みを理解する
  * ✅ Permutation Importanceでモデル非依存の特徴量重要度を計算できる
  * ✅ Partial Dependence Plots（PDP）で特徴量の影響を可視化できる
  * ✅ Anchorsや反事実的説明など最新の手法を活用できる
  * ✅ SHAP vs LIMEなど手法の使い分けができる
  * ✅ 計算コストと解釈の精度のトレードオフを理解する

* * *

## 3.1 LIME（Local Interpretable Model-agnostic Explanations）

### LIMEとは

**LIME（Local Interpretable Model-agnostic Explanations）** は、任意のブラックボックスモデルの個別予測を、局所的に解釈可能なモデルで近似する手法です。

> 「複雑なモデルも、ある1点の周辺では単純な線形モデルで近似できる」

### LIMEの基本原理
    
    
    ```mermaid
    graph LR
        A[元のデータポイント] --> B[周辺サンプリング]
        B --> C[ブラックボックスモデルで予測]
        C --> D[距離重み付け]
        D --> E[線形モデルで近似]
        E --> F[特徴量の重要度]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#c8e6c9
    ```

#### 数学的定式化

LIMEは以下の最適化問題を解きます：

$$ \xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

  * $f$: ブラックボックスモデル
  * $g$: 解釈可能なモデル（線形モデルなど）
  * $\mathcal{L}$: 損失関数（$f$と$g$の予測の違い）
  * $\pi_x$: 元のデータ点$x$からの距離に基づく重み
  * $\Omega(g)$: モデルの複雑さのペナルティ

### LIME実装：表形式データ
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from lime import lime_tabular
    
    # データ準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ブラックボックスモデル（ランダムフォレスト）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"モデル精度: {model.score(X_test, y_test):.3f}")
    
    # LIME Explainerの作成
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    
    # 1つのサンプルを説明
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    
    # 説明を生成
    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    print("\n=== LIME説明 ===")
    print(f"予測クラス: {data.target_names[model.predict([sample])[0]]}")
    print(f"予測確率: {model.predict_proba([sample])[0]}")
    print("\n特徴量の寄与:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:+.4f}")
    
    # 可視化
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    モデル精度: 0.965
    
    === LIME説明 ===
    予測クラス: benign
    予測確率: [0.03 0.97]
    
    特徴量の寄与:
      worst concave points <= 0.10: +0.2845
      worst radius <= 13.43: +0.1234
      worst perimeter <= 86.60: +0.0987
      mean concave points <= 0.05: +0.0765
      worst area <= 549.20: +0.0543
    

> **重要** : LIMEは局所的な説明であり、モデル全体の挙動を示すものではありません。

### LIMEのサンプリング手法

#### サンプリングプロセス
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # シンプルな2次元データ
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 説明したいサンプル
    sample = X[0]
    
    # LIMEスタイルのサンプリング（正規分布で近傍生成）
    n_samples = 1000
    noise_scale = 0.5
    
    # サンプル周辺でサンプリング
    samples = np.random.normal(
        loc=sample,
        scale=noise_scale,
        size=(n_samples, 2)
    )
    
    # ブラックボックスモデルで予測
    predictions = model.predict_proba(samples)[:, 1]
    
    # 距離計算（ユークリッド距離）
    distances = np.sqrt(np.sum((samples - sample)**2, axis=1))
    
    # カーネル重み（距離に反比例）
    kernel_width = 0.75
    weights = np.exp(-(distances**2) / (kernel_width**2))
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 元のデータ空間
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                    alpha=0.5, edgecolors='black')
    axes[0].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2,
                    label='説明対象サンプル')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('元のデータ空間', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # サンプリングされた点
    scatter = axes[1].scatter(samples[:, 0], samples[:, 1],
                             c=predictions, cmap='coolwarm',
                             alpha=0.4, s=20, edgecolors='none')
    axes[1].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title('サンプリングされた近傍点', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='予測確率')
    
    # 重み付けされた点
    scatter2 = axes[2].scatter(samples[:, 0], samples[:, 1],
                              c=predictions, cmap='coolwarm',
                              alpha=weights, s=weights*100, edgecolors='none')
    axes[2].scatter(sample[0], sample[1], color='green',
                    s=300, marker='*', edgecolors='black', linewidth=2)
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].set_title('距離重み付け（近いほど大きく）', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[2], label='予測確率')
    
    plt.tight_layout()
    plt.show()
    
    print(f"サンプリング数: {n_samples}")
    print(f"平均距離: {distances.mean():.3f}")
    print(f"最小/最大重み: {weights.min():.4f} / {weights.max():.4f}")
    

### 完全なLIME実装例
    
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics.pairwise import rbf_kernel
    
    class SimpleLIME:
        """シンプルなLIME実装"""
    
        def __init__(self, kernel_width=0.75, n_samples=5000):
            self.kernel_width = kernel_width
            self.n_samples = n_samples
    
        def explain_instance(self, model, instance, X_train,
                            feature_names=None, n_features=10):
            """
            個別のインスタンスを説明
    
            Parameters:
            -----------
            model : 訓練済みモデル（predict_probaメソッド必須）
            instance : 説明したいサンプル（1D array）
            X_train : 訓練データ（統計情報用）
            feature_names : 特徴量名リスト
            n_features : 上位何個の特徴量を返すか
    
            Returns:
            --------
            explanations : 特徴量と重要度のリスト
            """
            # 近傍サンプリング
            samples = self._sample_around_instance(instance, X_train)
    
            # モデルで予測
            predictions = model.predict_proba(samples)[:, 1]
    
            # 距離ベースの重み計算
            distances = np.sqrt(np.sum((samples - instance)**2, axis=1))
            weights = np.exp(-(distances**2) / (self.kernel_width**2))
    
            # 線形モデルで近似
            linear_model = Ridge(alpha=1.0)
            linear_model.fit(samples, predictions, sample_weight=weights)
    
            # 特徴量の重要度を取得
            feature_importance = linear_model.coef_
    
            # 特徴量名の設定
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(instance))]
    
            # 重要度でソート
            sorted_idx = np.argsort(np.abs(feature_importance))[::-1][:n_features]
    
            explanations = [
                (feature_names[idx], feature_importance[idx])
                for idx in sorted_idx
            ]
    
            return explanations, linear_model.score(samples, predictions,
                                                   sample_weight=weights)
    
        def _sample_around_instance(self, instance, X_train):
            """インスタンス周辺でサンプリング"""
            # 訓練データの統計を使用
            means = X_train.mean(axis=0)
            stds = X_train.std(axis=0)
    
            # 正規分布でサンプリング
            samples = np.random.normal(
                loc=instance,
                scale=stds * 0.5,  # スケール調整
                size=(self.n_samples, len(instance))
            )
    
            return samples
    
    # 使用例
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # データ準備
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SimpleLIMEで説明
    lime_explainer = SimpleLIME(kernel_width=0.75, n_samples=5000)
    sample = X_test[0]
    
    explanations, r2_score = lime_explainer.explain_instance(
        model=model,
        instance=sample,
        X_train=X_train,
        feature_names=data.feature_names,
        n_features=10
    )
    
    print("=== SimpleLIME説明 ===")
    print(f"局所モデルのR²スコア: {r2_score:.3f}")
    print("\n特徴量の重要度:")
    for feature, importance in explanations:
        print(f"  {feature}: {importance:+.4f}")
    

* * *

## 3.2 Permutation Importance

### Permutation Importanceとは

**Permutation Importance（順列重要度）** は、各特徴量をランダムにシャッフルした際のモデル性能の低下を測定する、モデル非依存の特徴量重要度計算手法です。

#### アルゴリズム

  1. ベースラインのモデル性能を計算
  2. 各特徴量について： 
     * その特徴量の値をシャッフル
     * モデル性能を再計算
     * 性能の低下量 = 重要度
  3. 元に戻して次の特徴量へ

    
    
    ```mermaid
    graph TD
        A[元のデータ] --> B[ベースライン性能測定]
        B --> C[特徴量1をシャッフル]
        C --> D[性能測定]
        D --> E[重要度 = 性能低下]
        E --> F[特徴量2をシャッフル]
        F --> G[...]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#c8e6c9
    ```

### scikit-learnでの実装
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # データ準備（糖尿病データセット）
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ベースライン性能
    baseline_score = model.score(X_test, y_test)
    print(f"ベースラインR²スコア: {baseline_score:.3f}")
    
    # Permutation Importanceの計算
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,  # シャッフルを30回繰り返し
        random_state=42,
        n_jobs=-1
    )
    
    # 結果の整理
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n=== Permutation Importance ===")
    print(importance_df)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance_mean'],
            xerr=importance_df['importance_std'],
            align='center', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² decrease)')
    ax.set_title('Permutation Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

### 独自実装：Permutation Importance
    
    
    import numpy as np
    from sklearn.metrics import r2_score, accuracy_score
    
    def custom_permutation_importance(model, X, y, metric='r2', n_repeats=10):
        """
        Permutation Importanceのカスタム実装
    
        Parameters:
        -----------
        model : 訓練済みモデル
        X : 特徴量（DataFrame or ndarray）
        y : ターゲット
        metric : 評価指標 ('r2' or 'accuracy')
        n_repeats : 各特徴量のシャッフル繰り返し回数
    
        Returns:
        --------
        importances : 各特徴量の重要度（平均と標準偏差）
        """
        X_array = X.values if hasattr(X, 'values') else X
        n_features = X_array.shape[1]
    
        # メトリック関数の選択
        if metric == 'r2':
            score_func = r2_score
            predictions = model.predict(X_array)
        elif metric == 'accuracy':
            score_func = accuracy_score
            predictions = model.predict(X_array)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
        # ベースラインスコア
        baseline_score = score_func(y, predictions)
    
        # 各特徴量の重要度を計算
        importances = np.zeros((n_features, n_repeats))
    
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # 特徴量をシャッフル
                X_permuted = X_array.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
    
                # 予測と評価
                if metric == 'r2':
                    perm_predictions = model.predict(X_permuted)
                else:
                    perm_predictions = model.predict(X_permuted)
    
                perm_score = score_func(y, perm_predictions)
    
                # スコアの低下 = 重要度
                importances[feature_idx, repeat] = baseline_score - perm_score
    
        # 統計量を計算
        result = {
            'importances_mean': importances.mean(axis=1),
            'importances_std': importances.std(axis=1),
            'importances': importances
        }
    
        return result
    
    # 使用例
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # カスタム実装で計算
    custom_perm = custom_permutation_importance(
        model, X_test, y_test,
        metric='accuracy',
        n_repeats=30
    )
    
    # 結果の整理
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': custom_perm['importances_mean'],
        'importance_std': custom_perm['importances_std']
    }).sort_values('importance_mean', ascending=False)
    
    print("=== カスタムPermutation Importance ===")
    print(results_df.head(10))
    

### Permutation ImportanceとSHAPの比較
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    import shap
    
    # データとモデル
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 1. Permutation Importance
    perm_imp = permutation_importance(
        model, X, y, n_repeats=30, random_state=42
    )
    
    # 2. SHAP値
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # 3. Tree Feature Importance（比較用）
    tree_importance = model.feature_importances_
    
    # 比較可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = [
        ('Permutation\nImportance', perm_imp.importances_mean),
        ('SHAP\nImportance', shap_importance),
        ('Tree Feature\nImportance', tree_importance)
    ]
    
    for ax, (title, importance) in zip(axes, methods):
        sorted_idx = np.argsort(importance)
        y_pos = np.arange(len(sorted_idx))
    
        ax.barh(y_pos, importance[sorted_idx], alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(X.columns[sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # 相関分析
    print("=== 手法間の相関 ===")
    comparison_df = pd.DataFrame({
        'Permutation': perm_imp.importances_mean,
        'SHAP': shap_importance,
        'Tree': tree_importance
    })
    print(comparison_df.corr())
    

* * *

## 3.3 Partial Dependence Plots（PDP）

### PDPとは

**Partial Dependence Plot（部分依存プロット）** は、特徴量がモデルの予測に与える平均的な影響を可視化する手法です。

#### 数学的定義

特徴量$x_S$に対する部分依存関数：

$$ \hat{f}_{x_S}(x_S) = \mathbb{E}_{x_C}[\hat{f}(x_S, x_C)] = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_S, x_C^{(i)}) $$

  * $x_S$: 対象の特徴量
  * $x_C$: その他の特徴量
  * $\hat{f}$: モデルの予測関数

### 1次元PDPの実装
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import PartialDependenceDisplay
    
    # データ準備
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # モデル訓練
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # PDPの計算と可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features_to_plot = ['age', 'bmi', 's5', 'bp', 's1', 's3']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes.flatten()[idx]
    
        # PDPの表示
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            ax=ax, kind='average'
        )
    
        ax.set_title(f'PDP: {feature}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Partial Dependence Plot生成完了 ===")
    print("各特徴量の平均的な影響を可視化")
    

### 2D PDP（相互作用の可視化）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    
    # 2つの特徴量の相互作用を可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2D PDPの例1: bmi vs s5
    display1 = PartialDependenceDisplay.from_estimator(
        model, X, features=[('bmi', 's5')],
        ax=axes[0], kind='average'
    )
    axes[0].set_title('2D PDP: BMI vs S5 (相互作用)', fontsize=14)
    
    # 2D PDPの例2: age vs bmi
    display2 = PartialDependenceDisplay.from_estimator(
        model, X, features=[('age', 'bmi')],
        ax=axes[1], kind='average'
    )
    axes[1].set_title('2D PDP: Age vs BMI (相互作用)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

### ICE（Individual Conditional Expectation）

**ICE** は、各サンプルごとの条件付き期待値を可視化し、異質性を捉えます。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    
    # ICEプロット（個別の条件付き期待値）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    features_for_ice = ['bmi', 's5', 'bp']
    
    for ax, feature in zip(axes, features_for_ice):
        # ICEプロット（individual=True）
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            kind='individual',  # 個別の線
            ax=ax,
            subsample=50,  # 50サンプルのみ表示
            random_state=42
        )
    
        # PDPも重ねて表示
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature],
            kind='average',  # 平均線
            ax=ax,
            line_kw={'color': 'red', 'linewidth': 3, 'label': 'PDP (average)'}
        )
    
        ax.set_title(f'ICE + PDP: {feature}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ICEプロット説明 ===")
    print("- 細い線: 個別サンプルの条件付き期待値（ICE）")
    print("- 太い赤線: 平均的な効果（PDP）")
    print("- 線のばらつき = 異質性（個体差）")
    

### カスタムPDP実装
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def compute_partial_dependence(model, X, feature_idx, grid_resolution=50):
        """
        部分依存を計算
    
        Parameters:
        -----------
        model : 訓練済みモデル
        X : 特徴量データ
        feature_idx : 対象特徴量のインデックス
        grid_resolution : グリッドの解像度
    
        Returns:
        --------
        grid_values : グリッド点の値
        pd_values : 部分依存の値
        """
        X_array = X.values if hasattr(X, 'values') else X
    
        # 対象特徴量の範囲でグリッド生成
        feature_min = X_array[:, feature_idx].min()
        feature_max = X_array[:, feature_idx].max()
        grid_values = np.linspace(feature_min, feature_max, grid_resolution)
    
        # 部分依存の計算
        pd_values = []
    
        for grid_value in grid_values:
            # 全サンプルの対象特徴量をgrid_valueに固定
            X_modified = X_array.copy()
            X_modified[:, feature_idx] = grid_value
    
            # 予測の平均
            predictions = model.predict(X_modified)
            pd_values.append(predictions.mean())
    
        return grid_values, np.array(pd_values)
    
    # 使用例
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # カスタム実装でPDP計算
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (ax, feature) in enumerate(zip(axes.flatten(), X.columns[:6])):
        grid, pd_vals = compute_partial_dependence(
            model, X, feature_idx=idx, grid_resolution=100
        )
    
        ax.plot(grid, pd_vals, linewidth=2, color='blue')
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'Custom PDP: {feature}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.4 その他の解釈手法

### Anchors

**Anchors** は、予測を保証する最小限のルールセットを見つける手法です。

> 「この条件が満たされれば、95%以上の確率で同じ予測になる」
    
    
    from anchor import anchor_tabular
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # データ準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Anchors Explainer作成
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=data.target_names,
        feature_names=data.feature_names,
        train_data=X_train.values
    )
    
    # 説明生成
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    
    explanation = explainer.explain_instance(
        data_row=sample,
        classifier_fn=model.predict,
        threshold=0.95  # 95%の信頼度
    )
    
    print("=== Anchors説明 ===")
    print(f"予測: {data.target_names[model.predict([sample])[0]]}")
    print(f"\nAnchor (精度={explanation.precision():.2f}):")
    print('AND'.join(explanation.names()))
    print(f"\nカバレッジ: {explanation.coverage():.2%}")
    

### Counterfactual Explanations（反事実的説明）

**Counterfactual Explanations** は、「何を変えれば予測が変わるか」を示します。
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # データ準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    def find_counterfactual(model, instance, target_class,
                           X_train, max_iterations=1000,
                           step_size=0.1):
        """
        反事実的説明を探索（シンプルな勾配ベース）
    
        Parameters:
        -----------
        model : 訓練済みモデル
        instance : 元のインスタンス
        target_class : 目標クラス
        X_train : 訓練データ（範囲の参考用）
        max_iterations : 最大反復回数
        step_size : ステップサイズ
    
        Returns:
        --------
        counterfactual : 反事実的インスタンス
        changes : 変更内容
        """
        counterfactual = instance.copy()
    
        for iteration in range(max_iterations):
            # 現在の予測
            pred_class = model.predict([counterfactual])[0]
    
            if pred_class == target_class:
                break
    
            # ランダムに特徴量を選んで変更
            feature_idx = np.random.randint(0, len(counterfactual))
    
            # 訓練データの範囲内でランダム変更
            feature_range = X_train.iloc[:, feature_idx]
            new_value = np.random.uniform(
                feature_range.min(),
                feature_range.max()
            )
    
            counterfactual[feature_idx] = new_value
    
        # 変更内容を計算
        changes = {}
        for idx, (orig, cf) in enumerate(zip(instance, counterfactual)):
            if not np.isclose(orig, cf):
                changes[X.columns[idx]] = {
                    'original': orig,
                    'counterfactual': cf,
                    'change': cf - orig
                }
    
        return counterfactual, changes
    
    # 使用例
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    original_pred = model.predict([sample])[0]
    target = 1 - original_pred  # 反対のクラス
    
    counterfactual, changes = find_counterfactual(
        model, sample, target, X_train,
        max_iterations=5000
    )
    
    print("=== Counterfactual Explanation ===")
    print(f"元の予測: {data.target_names[original_pred]}")
    print(f"目標予測: {data.target_names[target]}")
    print(f"反事実後: {data.target_names[model.predict([counterfactual])[0]]}")
    print(f"\n変更が必要な特徴量（上位5個）:")
    
    sorted_changes = sorted(
        changes.items(),
        key=lambda x: abs(x[1]['change']),
        reverse=True
    )[:5]
    
    for feature, change_info in sorted_changes:
        print(f"\n{feature}:")
        print(f"  元の値: {change_info['original']:.2f}")
        print(f"  変更後: {change_info['counterfactual']:.2f}")
        print(f"  変化量: {change_info['change']:+.2f}")
    

### Feature Ablation（特徴量削除）

**Feature Ablation** は、特徴量を削除した際の性能変化を測定します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    
    # データ準備
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # ベースライン性能
    baseline_score = r2_score(y, model.predict(X))
    
    # 各特徴量を削除した際の性能
    ablation_results = []
    
    for feature in X.columns:
        # 特徴量を削除
        X_ablated = X.drop(columns=[feature])
    
        # 新しいモデルを訓練
        model_ablated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_ablated.fit(X_ablated, y)
    
        # 性能測定
        score = r2_score(y, model_ablated.predict(X_ablated))
        importance = baseline_score - score
    
        ablation_results.append({
            'feature': feature,
            'score_without': score,
            'importance': importance
        })
    
    # 結果の整理
    ablation_df = pd.DataFrame(ablation_results).sort_values(
        'importance', ascending=False
    )
    
    print("=== Feature Ablation Results ===")
    print(f"Baseline R²: {baseline_score:.3f}\n")
    print(ablation_df)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(ablation_df))
    ax.barh(y_pos, ablation_df['importance'], alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ablation_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² decrease when removed)')
    ax.set_title('Feature Ablation Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 手法の比較とベストプラクティス

### SHAP vs LIME

観点 | SHAP | LIME  
---|---|---  
**理論的基盤** | ゲーム理論（Shapley値） | 局所線形近似  
**一貫性** | 高い（数学的保証あり） | 低い（サンプリングに依存）  
**計算コスト** | 高い（特にKernelSHAP） | 中程度  
**解釈の粒度** | 局所・大域両方 | 主に局所  
**モデル非依存性** | 完全に非依存 | 完全に非依存  
**再現性** | 高い | 中程度（ランダムサンプリング）  
**使いやすさ** | 非常に高い | 高い  
  
### グローバル vs ローカル解釈

手法 | タイプ | 用途  
---|---|---  
**LIME** | ローカル | 個別予測の説明  
**SHAP** | 両方 | 個別と全体の理解  
**Permutation Importance** | グローバル | 全体的な特徴量重要度  
**PDP/ICE** | グローバル | 特徴量の平均的影響  
**Anchors** | ローカル | ルールベースの説明  
**Counterfactuals** | ローカル | 変更の提案  
  
### 計算コストの比較
    
    
    import time
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import shap
    from lime import lime_tabular
    
    # データ準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 1サンプルの説明にかかる時間を測定
    sample = X_test.iloc[0].values
    
    results = {}
    
    # SHAP TreeExplainer
    start = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:10])
    results['SHAP (Tree)'] = time.time() - start
    
    # SHAP KernelExplainer（遅い）
    start = time.time()
    explainer_kernel = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_train, 50)
    )
    shap_kernel = explainer_kernel.shap_values(X_test.iloc[:5])
    results['SHAP (Kernel)'] = time.time() - start
    
    # LIME
    start = time.time()
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    for i in range(10):
        exp = lime_explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=10
        )
    results['LIME'] = time.time() - start
    
    # Permutation Importance
    start = time.time()
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42
    )
    results['Permutation'] = time.time() - start
    
    # 結果表示
    print("=== 計算時間の比較（10サンプル）===")
    for method, duration in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:20s}: {duration:6.2f}秒")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    times = list(results.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.barh(methods, times, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('計算時間（秒）', fontsize=12)
    ax.set_title('解釈手法の計算コスト比較', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 値を表示
    for bar, time_val in zip(bars, times):
        ax.text(time_val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{time_val:.2f}s', va='center')
    
    plt.tight_layout()
    plt.show()
    

### 使い分けガイド

状況 | 推奨手法 | 理由  
---|---|---  
個別予測の詳細説明 | SHAP (Tree) | 高速で一貫性あり  
任意モデルの個別説明 | LIME, KernelSHAP | モデル非依存  
全体的な特徴量重要度 | Permutation, SHAP | グローバルな理解  
特徴量の影響の可視化 | PDP/ICE | 直感的な理解  
ルールベースの説明 | Anchors | if-then形式  
変更提案 | Counterfactuals | 実用的なアクション  
計算時間が限られる | LIME, Tree SHAP | 比較的高速  
高精度が必要 | SHAP | 理論的保証  
  
### ベストプラクティス

  1. **複数手法の併用**

     * SHAP（大域）+ LIME（局所）で多角的に理解
     * PDP（平均）+ ICE（個別）で異質性を把握
  2. **計算リソースの考慮**

     * 本番環境: TreeSHAP, LIME
     * 研究・分析: KernelSHAP, 全手法併用
  3. **ドメイン知識の統合**

     * 解釈結果を業務知識で検証
     * 不自然な説明は要調査
  4. **可視化の工夫**

     * 非専門家向けには簡潔に
     * 専門家向けには詳細に
  5. **再現性の確保**

     * random_stateの固定
     * 説明の保存と共有

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **LIME**

     * 局所線形近似によるブラックボックス解釈
     * サンプリングと重み付けの仕組み
     * 実装と可視化の方法
  2. **Permutation Importance**

     * モデル非依存の特徴量重要度
     * シャッフルによる性能低下の測定
     * SHAPやTree Importanceとの違い
  3. **Partial Dependence Plots**

     * 特徴量の平均的影響の可視化
     * 2D PDPによる相互作用の理解
     * ICEによる異質性の捉え方
  4. **その他の手法**

     * Anchors: ルールベースの説明
     * Counterfactuals: 変更提案
     * Feature Ablation: 削除による重要度測定
  5. **手法の使い分け**

     * SHAP vs LIMEの特性比較
     * 計算コストと精度のトレードオフ
     * 状況に応じた最適な手法選択

### 主要な手法の特性まとめ

手法 | 強み | 弱み | 適用場面  
---|---|---|---  
**LIME** | 理解しやすい、高速 | 不安定、局所のみ | 個別予測の簡易説明  
**SHAP** | 理論的保証、一貫性 | 計算コスト（Kernel） | 精密な解釈が必要  
**Permutation** | シンプル、直感的 | 相関特徴で不安定 | 全体的重要度把握  
**PDP/ICE** | 可視化が直感的 | 相互作用の限界 | 特徴量影響の理解  
**Anchors** | ルール形式、明確 | カバレッジ制限 | ルールベース説明  
  
### 次の章へ

第4章では、**画像・テキストデータの解釈** を学びます：

  * Grad-CAMによる画像解釈
  * Attention機構の可視化
  * BERTモデルの解釈
  * Integrated Gradients
  * 実用的な応用例

* * *

## 演習問題

### 問題1（難易度：easy）

LIMEとSHAPの主な違いを3つ挙げて説明してください。

解答例

**解答** ：

  1. **理論的基盤**

     * LIME: 局所線形近似（サンプリング＋線形モデル）
     * SHAP: ゲーム理論のShapley値（公理的アプローチ）
  2. **一貫性と再現性**

     * LIME: サンプリングに依存するため、実行ごとに結果が変わる可能性
     * SHAP: 数学的に一貫した結果が保証される
  3. **適用範囲**

     * LIME: 主に局所的な説明（個別サンプル）
     * SHAP: 局所と大域の両方（個別＋全体の重要度）

**選択の目安** ：

  * 速度重視・簡易説明 → LIME
  * 精度重視・理論的保証 → SHAP
  * 理想的には両方を使って多角的に理解

### 問題2（難易度：medium）

Permutation Importanceを手動で実装し、scikit-learnの結果と比較してください。

解答例
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    
    # データ準備
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 手動実装
    def manual_permutation_importance(model, X, y, n_repeats=10):
        """Permutation Importanceの手動実装"""
        baseline_score = accuracy_score(y, model.predict(X))
        n_features = X.shape[1]
        importances = np.zeros((n_features, n_repeats))
    
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # 特徴量をシャッフル
                X_permuted = X.copy()
                X_permuted.iloc[:, feature_idx] = np.random.permutation(
                    X_permuted.iloc[:, feature_idx]
                )
    
                # スコア計算
                perm_score = accuracy_score(y, model.predict(X_permuted))
                importances[feature_idx, repeat] = baseline_score - perm_score
    
        return {
            'importances_mean': importances.mean(axis=1),
            'importances_std': importances.std(axis=1)
        }
    
    # 手動実装で計算
    manual_result = manual_permutation_importance(
        model, X_test, y_test, n_repeats=30
    )
    
    # scikit-learn実装
    sklearn_result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42
    )
    
    # 比較
    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'Manual_Mean': manual_result['importances_mean'],
        'Manual_Std': manual_result['importances_std'],
        'Sklearn_Mean': sklearn_result.importances_mean,
        'Sklearn_Std': sklearn_result.importances_std
    }).sort_values('Manual_Mean', ascending=False)
    
    print("=== Permutation Importance比較 ===")
    print(comparison_df)
    
    # 相関チェック
    correlation = np.corrcoef(
        manual_result['importances_mean'],
        sklearn_result.importances_mean
    )[0, 1]
    print(f"\n手動実装とsklearn実装の相関: {correlation:.4f}")
    print("（1に近いほど一致）")
    

**出力例** ：
    
    
    === Permutation Importance比較 ===
                       Feature  Manual_Mean  Manual_Std  Sklearn_Mean  Sklearn_Std
    2       petal length (cm)     0.3156      0.0289        0.3200       0.0265
    3        petal width (cm)     0.2933      0.0312        0.2867       0.0298
    0       sepal length (cm)     0.0089      0.0145        0.0111       0.0134
    1        sepal width (cm)     0.0067      0.0123        0.0044       0.0098
    
    手動実装とsklearn実装の相関: 0.9987
    （1に近いほど一致）
    

### 問題3（難易度：medium）

Partial Dependence PlotとICEプロットの違いを説明し、どのような場合にICEが有用か述べてください。

解答例

**解答** ：

**違い** ：

  * **PDP（Partial Dependence Plot）**

    * 全サンプルの平均的な効果を示す
    * 式: $\hat{f}_{PDP}(x_s) = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_s, x_c^{(i)})$
    * 1本の線で表現
  * **ICE（Individual Conditional Expectation）**

    * 各サンプルごとの効果を個別に示す
    * 式: $\hat{f}^{(i)}_{ICE}(x_s) = \hat{f}(x_s, x_c^{(i)})$
    * n本の線（サンプル数分）

**ICEが有用な場合** ：

  1. **異質性の検出**

     * サブグループで異なる効果がある場合
     * 例: 年齢の影響が性別で異なる
  2. **相互作用の発見**

     * 特徴量間の複雑な相互作用
     * PDPでは平均化されて見えない
  3. **非線形性の理解**

     * 個別のパターンが多様
     * 平均では単純に見えても実は複雑

**可視化例** ：
    
    
    from sklearn.inspection import PartialDependenceDisplay
    
    # PDPのみ（平均）
    PartialDependenceDisplay.from_estimator(
        model, X, features=['age'], kind='average'
    )
    
    # ICE + PDP（個別＋平均）
    PartialDependenceDisplay.from_estimator(
        model, X, features=['age'],
        kind='both',  # 両方表示
        subsample=50
    )
    

### 問題4（難易度：hard）

以下のデータに対して、LIME、SHAP、Permutation Importanceの3手法を適用し、結果を比較してください。特徴量の重要度ランキングが異なる場合、その理由を考察してください。
    
    
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    

解答例
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    import shap
    from lime import lime_tabular
    
    # データ準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 1. SHAP（グローバル）
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_test)
    shap_importance = np.abs(shap_values[1]).mean(axis=0)
    
    # 2. LIME（複数サンプルの平均）
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    
    lime_importances = []
    for i in range(min(50, len(X_test))):  # 50サンプル
        exp = explainer_lime.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=len(X.columns)
        )
        weights = dict(exp.as_list())
        # 特徴量名を抽出（条件部分を除去）
        feature_weights = {}
        for key, val in weights.items():
            feature_name = key.split('<=')[0].split('>')[0].strip()
            if feature_name in X.columns:
                feature_weights[feature_name] = abs(val)
        lime_importances.append(feature_weights)
    
    # LIME重要度の平均
    lime_importance_mean = pd.DataFrame(lime_importances).mean().reindex(X.columns).fillna(0)
    
    # 3. Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42
    )
    
    # 結果の統合
    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': shap_importance,
        'LIME': lime_importance_mean.values,
        'Permutation': perm_importance.importances_mean
    })
    
    # 正規化（比較のため）
    for col in ['SHAP', 'LIME', 'Permutation']:
        comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
    
    # ランキング
    comparison_df['SHAP_Rank'] = comparison_df['SHAP'].rank(ascending=False)
    comparison_df['LIME_Rank'] = comparison_df['LIME'].rank(ascending=False)
    comparison_df['Perm_Rank'] = comparison_df['Permutation'].rank(ascending=False)
    
    print("=== 3手法の特徴量重要度比較（上位10個）===\n")
    top_features = comparison_df.nlargest(10, 'SHAP')[
        ['Feature', 'SHAP', 'LIME', 'Permutation',
         'SHAP_Rank', 'LIME_Rank', 'Perm_Rank']
    ]
    print(top_features)
    
    # 相関分析
    print("\n=== 手法間の相関 ===")
    correlation_matrix = comparison_df[['SHAP', 'LIME', 'Permutation']].corr()
    print(correlation_matrix)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, method in zip(axes, ['SHAP', 'LIME', 'Permutation']):
        sorted_df = comparison_df.sort_values(method, ascending=True).tail(10)
    
        ax.barh(range(len(sorted_df)), sorted_df[method],
                alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Feature'])
        ax.set_xlabel('正規化重要度')
        ax.set_title(f'{method}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # ランク差分析
    print("\n=== ランクの大きな差異がある特徴量 ===")
    comparison_df['Rank_Std'] = comparison_df[
        ['SHAP_Rank', 'LIME_Rank', 'Perm_Rank']
    ].std(axis=1)
    
    disagreement = comparison_df.nlargest(5, 'Rank_Std')[
        ['Feature', 'SHAP_Rank', 'LIME_Rank', 'Perm_Rank', 'Rank_Std']
    ]
    print(disagreement)
    
    print("\n=== 考察 ===")
    print("""
    ランキングの違いが生じる理由：
    
    1. **測定対象の違い**
       - SHAP: 各特徴量の寄与（ゲーム理論）
       - LIME: 局所線形近似の係数
       - Permutation: シャッフル時の性能低下
    
    2. **局所 vs 大域**
       - LIME: 局所的（選択したサンプル周辺）
       - SHAP/Permutation: より大域的
    
    3. **特徴量の相関**
       - 相関が高い特徴量群では手法により重要度が分散
       - Permutationは相関を考慮しない
    
    4. **計算方法の違い**
       - SHAP: すべての特徴量の組み合わせを考慮
       - Permutation: 1つずつ独立に評価
       - LIME: サンプリングベース（ランダム性あり）
    
    推奨：複数手法を併用して多角的に解釈
    """)
    

### 問題5（難易度：hard）

Counterfactual Explanationを使って、予測が変わる最小の変更を見つけるアルゴリズムを実装してください。変更の妥当性をどのように評価すべきか考察してください。

解答例
    
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    class CounterfactualExplainer:
        """最小変更の反事実的説明を生成"""
    
        def __init__(self, model, X_train):
            self.model = model
            self.X_train = X_train
            self.feature_ranges = {
                'min': X_train.min(axis=0),
                'max': X_train.max(axis=0),
                'median': X_train.median(axis=0)
            }
    
        def find_minimal_counterfactual(self, instance, target_class,
                                       max_iterations=1000,
                                       change_penalty=0.1):
            """
            最小の変更で目標クラスを達成する反事実を探索
    
            Parameters:
            -----------
            instance : 元のインスタンス
            target_class : 目標クラス
            max_iterations : 最大反復回数
            change_penalty : 変更に対するペナルティ
    
            Returns:
            --------
            best_counterfactual : 最良の反事実的インスタンス
            changes : 変更内容
            metadata : メタ情報
            """
            best_counterfactual = None
            best_distance = float('inf')
    
            current = instance.copy()
    
            for iteration in range(max_iterations):
                # 現在の予測
                pred_class = self.model.predict([current])[0]
    
                if pred_class == target_class:
                    # 目標達成：距離を計算
                    distance = self._compute_distance(instance, current)
    
                    if distance < best_distance:
                        best_distance = distance
                        best_counterfactual = current.copy()
    
                # ランダムに1つの特徴量を変更
                feature_idx = np.random.randint(0, len(current))
    
                # 実現可能な範囲で変更
                feature_range = (
                    self.feature_ranges['min'][feature_idx],
                    self.feature_ranges['max'][feature_idx]
                )
    
                # 元の値に近い値を優先（正規分布）
                new_value = np.random.normal(
                    loc=instance[feature_idx],
                    scale=(feature_range[1] - feature_range[0]) * 0.1
                )
    
                # 範囲内にクリップ
                new_value = np.clip(new_value, feature_range[0], feature_range[1])
                current[feature_idx] = new_value
    
            if best_counterfactual is None:
                return None, None, {'success': False}
    
            # 変更内容の分析
            changes = self._analyze_changes(instance, best_counterfactual)
    
            # メタ情報
            metadata = {
                'success': True,
                'distance': best_distance,
                'n_changes': len(changes),
                'validity': self._check_validity(best_counterfactual)
            }
    
            return best_counterfactual, changes, metadata
    
        def _compute_distance(self, instance1, instance2):
            """L2距離（正規化）"""
            # 各特徴量を範囲で正規化
            ranges = self.feature_ranges['max'] - self.feature_ranges['min']
            normalized_diff = (instance1 - instance2) / ranges
            return np.sqrt(np.sum(normalized_diff**2))
    
        def _analyze_changes(self, original, counterfactual, threshold=0.01):
            """変更を分析"""
            changes = {}
            for idx, (orig, cf) in enumerate(zip(original, counterfactual)):
                relative_change = abs((cf - orig) / (orig + 1e-10))
    
                if relative_change > threshold:
                    changes[idx] = {
                        'original': orig,
                        'counterfactual': cf,
                        'absolute_change': cf - orig,
                        'relative_change': relative_change
                    }
            return changes
    
        def _check_validity(self, instance):
            """妥当性チェック（範囲内か）"""
            within_range = (
                (instance >= self.feature_ranges['min']).all() and
                (instance <= self.feature_ranges['max']).all()
            )
            return within_range
    
    # 使用例
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Counterfactual Explainer
    cf_explainer = CounterfactualExplainer(model, X_train)
    
    # テストサンプルで試行
    sample_idx = 0
    sample = X_test.iloc[sample_idx].values
    original_pred = model.predict([sample])[0]
    target_class = 1 - original_pred
    
    print("=== Counterfactual Explanation ===")
    print(f"元の予測: {data.target_names[original_pred]}")
    print(f"目標予測: {data.target_names[target_class]}\n")
    
    counterfactual, changes, metadata = cf_explainer.find_minimal_counterfactual(
        sample, target_class, max_iterations=5000
    )
    
    if metadata['success']:
        print(f"✓ 反事実的インスタンス発見")
        print(f"  距離: {metadata['distance']:.4f}")
        print(f"  変更数: {metadata['n_changes']}")
        print(f"  妥当性: {metadata['validity']}")
    
        cf_pred = model.predict([counterfactual])[0]
        print(f"  反事実後の予測: {data.target_names[cf_pred]}")
    
        print(f"\n必要な変更（上位5個）:")
        sorted_changes = sorted(
            changes.items(),
            key=lambda x: abs(x[1]['absolute_change']),
            reverse=True
        )[:5]
    
        for idx, change_info in sorted_changes:
            feature_name = X.columns[idx]
            print(f"\n{feature_name}:")
            print(f"  元: {change_info['original']:.2f}")
            print(f"  変更後: {change_info['counterfactual']:.2f}")
            print(f"  変化: {change_info['absolute_change']:+.2f} "
                  f"({change_info['relative_change']:.1%})")
    
        # 妥当性評価
        print("\n=== 妥当性評価 ===")
        print("1. 実現可能性: 訓練データの範囲内か")
        print(f"   → {metadata['validity']}")
    
        print("\n2. 最小性: 変更の数が少ないか")
        print(f"   → {metadata['n_changes']}/{len(sample)}特徴量を変更")
    
        print("\n3. 実用性: 変更が実行可能か")
        print("   → ドメイン知識で検証必要")
        print("   （例: 年齢を若くするのは不可能）")
    
        print("\n4. 近接性: 元のインスタンスに近いか")
        print(f"   → 正規化距離: {metadata['distance']:.4f}")
    
    else:
        print("✗ 反事実的インスタンスが見つかりませんでした")
    
    print("\n=== 評価基準のまとめ ===")
    print("""
    反事実的説明の妥当性評価：
    
    1. **実現可能性（Feasibility）**
       - 訓練データの分布内に存在するか
       - 物理的/論理的に可能な値か
    
    2. **最小性（Minimality）**
       - 変更する特徴量の数が最小か
       - 変更の大きさが最小か
    
    3. **実用性（Actionability）**
       - 実際に変更可能な特徴量か
       - コストが現実的か
    
    4. **近接性（Proximity）**
       - 元のインスタンスに近いか
       - 解釈が容易か
    
    5. **多様性（Diversity）**
       - 複数の解決策を提示できるか
       - ユーザーの選択肢があるか
    """)
    

* * *

## 参考文献

  1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. _KDD_.
  2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. _NeurIPS_.
  3. Molnar, C. (2022). _Interpretable Machine Learning_ (2nd ed.). Available at: https://christophm.github.io/interpretable-ml-book/
  4. Goldstein, A., et al. (2015). Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation. _Journal of Computational and Graphical Statistics_.
  5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision Model-Agnostic Explanations. _AAAI_.
  6. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual Explanations without Opening the Black Box. _Harvard Journal of Law & Technology_.
  7. Breiman, L. (2001). Random Forests. _Machine Learning_ , 45(1), 5-32.
