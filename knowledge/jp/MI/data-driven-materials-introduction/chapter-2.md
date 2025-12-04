---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25分
difficulty: 初級
code_examples: 0
exercises: 0
---

# Chapter 2: 特徴量エンジニアリング

* * *

## 学習目標

この章を読むことで、以下を習得できます：

✅ 材料記述子（組成・構造・電子構造）の選択と設計 ✅ matminerを活用した材料特徴量の自動生成 ✅ 特徴量変換（正規化、対数変換、多項式特徴量）の実践 ✅ 次元削減（PCA、t-SNE、UMAP）による可視化と解釈 ✅ 特徴量選択（Filter/Wrapper/Embedded/SHAP-based）の使い分け ✅ バンドギャップ予測における200次元→20次元への効果的削減

* * *

## 2.1 材料記述子の選択と設計

材料の性質を機械学習で予測するには、適切な**材料記述子（Material Descriptors）** が必要です。

### 組成記述子
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def calculate_composition_descriptors(formula_dict):
        """
        組成記述子の計算
    
        Parameters:
        -----------
        formula_dict : dict
            {'元素記号': 割合} 例: {'Fe': 0.7, 'Ni': 0.3}
    
        Returns:
        --------
        dict : 組成記述子
        """
        # 元素の物性値（簡略版）
        element_properties = {
            'Fe': {'atomic_mass': 55.845, 'electronegativity': 1.83,
                   'atomic_radius': 1.26},
            'Ni': {'atomic_mass': 58.693, 'electronegativity': 1.91,
                   'atomic_radius': 1.24},
            'Cu': {'atomic_mass': 63.546, 'electronegativity': 1.90,
                   'atomic_radius': 1.28},
            'Zn': {'atomic_mass': 65.38, 'electronegativity': 1.65,
                   'atomic_radius': 1.34}
        }
    
        descriptors = {}
    
        # 平均原子量
        avg_mass = sum(
            element_properties[el]['atomic_mass'] * frac
            for el, frac in formula_dict.items()
        )
        descriptors['平均原子量'] = avg_mass
    
        # 平均電気陰性度
        avg_electronegativity = sum(
            element_properties[el]['electronegativity'] * frac
            for el, frac in formula_dict.items()
        )
        descriptors['平均電気陰性度'] = avg_electronegativity
    
        # 電気陰性度差（最大 - 最小）
        electronegativities = [
            element_properties[el]['electronegativity']
            for el in formula_dict.keys()
        ]
        descriptors['電気陰性度差'] = max(electronegativities) - min(electronegativities)
    
        # 平均原子半径
        avg_radius = sum(
            element_properties[el]['atomic_radius'] * frac
            for el, frac in formula_dict.items()
        )
        descriptors['平均原子半径'] = avg_radius
    
        return descriptors
    
    # 例：Fe-Ni合金
    formula = {'Fe': 0.7, 'Ni': 0.3}
    descriptors = calculate_composition_descriptors(formula)
    
    print("組成記述子（Fe₀.₇Ni₀.₃）：")
    for key, value in descriptors.items():
        print(f"  {key}: {value:.4f}")
    

**出力** ：
    
    
    組成記述子（Fe₀.₇Ni₀.₃）：
      平均原子量: 56.6984
      平均電気陰性度: 1.8540
      電気陰性度差: 0.0800
      平均原子半径: 1.2540
    

### matminerの活用
    
    
    # matminerによる材料記述子の自動生成
    # !pip install matminer pymatgen
    
    from matminer.featurizers.composition import (
        ElementProperty,
        Stoichiometry,
        ValenceOrbital,
        IonProperty
    )
    from pymatgen.core import Composition
    
    def generate_matminer_features(formula_str):
        """
        matminerで材料記述子を生成
    
        Parameters:
        -----------
        formula_str : str
            化学式（例: "Fe2O3"）
    
        Returns:
        --------
        pd.DataFrame : 特徴量
        """
        comp = Composition(formula_str)
    
        # 元素物性記述子
        ep_feat = ElementProperty.from_preset("magpie")
        features_ep = ep_feat.featurize(comp)
    
        # 化学量論記述子
        stoich_feat = Stoichiometry()
        features_stoich = stoich_feat.featurize(comp)
    
        # 価電子軌道記述子
        valence_feat = ValenceOrbital()
        features_valence = valence_feat.featurize(comp)
    
        # 特徴量名取得
        feature_names = (
            ep_feat.feature_labels() +
            stoich_feat.feature_labels() +
            valence_feat.feature_labels()
        )
    
        # DataFrameに変換
        all_features = features_ep + features_stoich + features_valence
        df = pd.DataFrame([all_features], columns=feature_names)
    
        return df
    
    # 例：酸化鉄
    formula = "Fe2O3"
    features = generate_matminer_features(formula)
    
    print(f"matminerによる特徴量生成（{formula}）：")
    print(f"特徴量数: {features.shape[1]}")
    print(f"\n最初の10特徴量：")
    print(features.iloc[:, :10].T)
    

**matminerの主な記述子** ：
    
    
    # 記述子タイプの比較
    descriptor_types = pd.DataFrame({
        '記述子タイプ': [
            'ElementProperty',
            'Stoichiometry',
            'ValenceOrbital',
            'IonProperty',
            'OxidationStates',
            'ElectronAffinity'
        ],
        '特徴量数': [132, 7, 10, 32, 3, 1],
        '用途': [
            '元素の物理化学的性質',
            '化学量論比',
            '価電子軌道',
            'イオン特性',
            '酸化状態',
            '電子親和力'
        ]
    })
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(descriptor_types['記述子タイプ'],
            descriptor_types['特徴量数'],
            color='steelblue', alpha=0.7)
    ax.set_xlabel('特徴量数', fontsize=12)
    ax.set_title('matminerの記述子タイプ', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for idx, row in descriptor_types.iterrows():
        ax.text(row['特徴量数'] + 5, idx, row['用途'],
                va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    print(descriptor_types.to_string(index=False))
    

### 構造記述子
    
    
    def calculate_structure_descriptors(lattice_params):
        """
        結晶構造記述子
    
        Parameters:
        -----------
        lattice_params : dict
            {'a': float, 'b': float, 'c': float,
             'alpha': float, 'beta': float, 'gamma': float}
    
        Returns:
        --------
        dict : 構造記述子
        """
        a = lattice_params['a']
        b = lattice_params['b']
        c = lattice_params['c']
        alpha = np.radians(lattice_params['alpha'])
        beta = np.radians(lattice_params['beta'])
        gamma = np.radians(lattice_params['gamma'])
    
        # 体積
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
    
        # パッキング密度（簡略化）
        packing_density = 0.74  # 例：FCCの場合
    
        descriptors = {
            '格子定数a': a,
            '格子定数b': b,
            '格子定数c': c,
            '体積': volume,
            'パッキング密度': packing_density
        }
    
        return descriptors
    
    # 例：立方晶
    lattice = {'a': 5.43, 'b': 5.43, 'c': 5.43,
               'alpha': 90, 'beta': 90, 'gamma': 90}
    struct_desc = calculate_structure_descriptors(lattice)
    
    print("構造記述子（立方晶）：")
    for key, value in struct_desc.items():
        print(f"  {key}: {value:.4f}")
    

### 電子構造記述子
    
    
    # 電子構造記述子のシミュレーション
    def simulate_electronic_descriptors(n_samples=100):
        """
        電子構造記述子のサンプルデータ生成
        """
        np.random.seed(42)
    
        data = pd.DataFrame({
            'バンドギャップ': np.random.uniform(0, 5, n_samples),
            'フェルミエネルギー': np.random.uniform(-5, 5, n_samples),
            '状態密度_価電子帯': np.random.uniform(10, 100, n_samples),
            '状態密度_伝導帯': np.random.uniform(5, 50, n_samples),
            '有効質量_電子': np.random.uniform(0.1, 2, n_samples),
            '有効質量_正孔': np.random.uniform(0.1, 2, n_samples)
        })
    
        return data
    
    # 生成
    electronic_data = simulate_electronic_descriptors(100)
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(electronic_data.columns):
        axes[idx].hist(electronic_data[col], bins=20,
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(col, fontsize=11)
        axes[idx].set_ylabel('頻度', fontsize=11)
        axes[idx].set_title(f'{col}の分布', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("電子構造記述子の統計：")
    print(electronic_data.describe())
    

* * *

## 2.2 特徴量変換

生の特徴量を機械学習モデルに適した形に変換します。

### 正規化（Min-Max, Z-score）
    
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    def compare_normalization(data):
        """
        正規化手法の比較
        """
        # Min-Max正規化（0-1）
        minmax_scaler = MinMaxScaler()
        data_minmax = pd.DataFrame(
            minmax_scaler.fit_transform(data),
            columns=data.columns
        )
    
        # Z-score正規化（平均0、標準偏差1）
        standard_scaler = StandardScaler()
        data_standard = pd.DataFrame(
            standard_scaler.fit_transform(data),
            columns=data.columns
        )
    
        return data_minmax, data_standard
    
    # サンプルデータ
    np.random.seed(42)
    sample_data = pd.DataFrame({
        '格子定数': np.random.uniform(3, 7, 100),
        '電気伝導度': np.random.lognormal(10, 2, 100),
        'バンドギャップ': np.random.uniform(0, 3, 100)
    })
    
    # 正規化
    data_minmax, data_standard = compare_normalization(sample_data)
    
    # 可視化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for idx, col in enumerate(sample_data.columns):
        # 元データ
        axes[idx, 0].hist(sample_data[col], bins=20,
                          color='gray', alpha=0.7, edgecolor='black')
        axes[idx, 0].set_title(f'元データ: {col}', fontsize=11, fontweight='bold')
        axes[idx, 0].set_ylabel('頻度', fontsize=10)
    
        # Min-Max
        axes[idx, 1].hist(data_minmax[col], bins=20,
                          color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx, 1].set_title(f'Min-Max: {col}', fontsize=11, fontweight='bold')
    
        # Z-score
        axes[idx, 2].hist(data_standard[col], bins=20,
                          color='coral', alpha=0.7, edgecolor='black')
        axes[idx, 2].set_title(f'Z-score: {col}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("正規化後の統計：")
    print("\nMin-Max正規化：")
    print(data_minmax.describe())
    print("\nZ-score正規化：")
    print(data_standard.describe())
    

### 対数変換、Box-Cox変換
    
    
    from scipy.stats import boxcox
    
    def apply_transformations(data):
        """
        各種変換の適用
        """
        # 対数変換
        data_log = np.log1p(data)  # log(1+x)で0を扱える
    
        # Box-Cox変換（正値のみ）
        data_boxcox, lambda_param = boxcox(data + 1)  # +1でゼロを回避
    
        return data_log, data_boxcox, lambda_param
    
    # 偏ったデータ（電気伝導度など）
    np.random.seed(42)
    conductivity = np.random.lognormal(10, 2, 100)
    
    # 変換
    cond_log, cond_boxcox, lambda_val = apply_transformations(conductivity)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元データ
    axes[0].hist(conductivity, bins=30, color='gray',
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('電気伝導度 (S/m)', fontsize=11)
    axes[0].set_ylabel('頻度', fontsize=11)
    axes[0].set_title('元データ（歪度あり）', fontsize=12, fontweight='bold')
    
    # 対数変換
    axes[1].hist(cond_log, bins=30, color='steelblue',
                 alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('log(電気伝導度+1)', fontsize=11)
    axes[1].set_ylabel('頻度', fontsize=11)
    axes[1].set_title('対数変換', fontsize=12, fontweight='bold')
    
    # Box-Cox変換
    axes[2].hist(cond_boxcox, bins=30, color='coral',
                 alpha=0.7, edgecolor='black')
    axes[2].set_xlabel(f'Box-Cox (λ={lambda_val:.3f})', fontsize=11)
    axes[2].set_ylabel('頻度', fontsize=11)
    axes[2].set_title('Box-Cox変換', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 歪度の比較
    from scipy.stats import skew
    print(f"元データの歪度: {skew(conductivity):.3f}")
    print(f"対数変換後の歪度: {skew(cond_log):.3f}")
    print(f"Box-Cox変換後の歪度: {skew(cond_boxcox):.3f}")
    

### 多項式特徴量
    
    
    from sklearn.preprocessing import PolynomialFeatures
    
    def create_polynomial_features(X, degree=2):
        """
        多項式特徴量の生成
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        degree : int
            多項式の次数
    
        Returns:
        --------
        X_poly : 多項式特徴量
        feature_names : 特徴量名
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out()
    
        return X_poly, feature_names
    
    # サンプルデータ
    np.random.seed(42)
    X_original = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 50),
        'x2': np.random.uniform(0, 1, 50)
    })
    
    # 2次多項式特徴量
    X_poly, feature_names = create_polynomial_features(
        X_original.values, degree=2
    )
    
    print(f"元の特徴量数: {X_original.shape[1]}")
    print(f"多項式特徴量数: {X_poly.shape[1]}")
    print(f"\n生成された特徴量：")
    for name in feature_names:
        print(f"  {name}")
    
    # 非線形関係の学習例
    # y = 2*x1^2 + 3*x1*x2 - x2^2 + noise
    y_true = (
        2 * X_original['x1']**2 +
        3 * X_original['x1'] * X_original['x2'] -
        X_original['x2']**2 +
        np.random.normal(0, 0.1, 50)
    )
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # 線形モデル（元の特徴量）
    model_linear = LinearRegression()
    model_linear.fit(X_original, y_true)
    y_pred_linear = model_linear.predict(X_original)
    r2_linear = r2_score(y_true, y_pred_linear)
    
    # 線形モデル（多項式特徴量）
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y_true)
    y_pred_poly = model_poly.predict(X_poly)
    r2_poly = r2_score(y_true, y_pred_poly)
    
    print(f"\n線形モデル（元特徴量）R²: {r2_linear:.4f}")
    print(f"線形モデル（多項式特徴量）R²: {r2_poly:.4f}")
    print(f"改善率: {(r2_poly - r2_linear) / r2_linear * 100:.1f}%")
    

### 交互作用項の生成
    
    
    from itertools import combinations
    
    def create_interaction_features(df):
        """
        交互作用項の生成
    
        Parameters:
        -----------
        df : pd.DataFrame
            元の特徴量
    
        Returns:
        --------
        df_with_interactions : 交互作用項を追加したDataFrame
        """
        df_new = df.copy()
    
        # 2変数の交互作用（積）
        for col1, col2 in combinations(df.columns, 2):
            interaction_name = f"{col1}×{col2}"
            df_new[interaction_name] = df[col1] * df[col2]
    
        return df_new
    
    # 例：熱電材料の特徴量
    thermoelectric_features = pd.DataFrame({
        '電気伝導度': np.random.lognormal(8, 1, 50),
        'ゼーベック係数': np.random.normal(200, 50, 50),
        '熱伝導度': np.random.uniform(1, 10, 50)
    })
    
    # 交互作用項追加
    features_with_interactions = create_interaction_features(
        thermoelectric_features
    )
    
    print(f"元の特徴量: {thermoelectric_features.columns.tolist()}")
    print(f"\n追加された交互作用項:")
    new_cols = [col for col in features_with_interactions.columns
                if col not in thermoelectric_features.columns]
    for col in new_cols:
        print(f"  {col}")
    
    # ZT値予測（ZT = σ*S²/κ に近い）
    thermoelectric_features['ZT'] = (
        thermoelectric_features['電気伝導度'] *
        thermoelectric_features['ゼーベック係数']**2 /
        thermoelectric_features['熱伝導度'] / 1e6 +
        np.random.normal(0, 0.1, 50)
    )
    
    print(f"\n相関分析（交互作用項との相関）：")
    correlations = features_with_interactions.corrwith(
        thermoelectric_features['ZT']
    ).sort_values(ascending=False)
    print(correlations)
    

* * *

## 2.3 次元削減

高次元データを低次元に圧縮し、可視化・解釈を容易にします。

### PCA (Principal Component Analysis)
    
    
    from sklearn.decomposition import PCA
    
    def apply_pca(X, n_components=2):
        """
        PCAによる次元削減
    
        Parameters:
        -----------
        X : array-like
            元の特徴量
        n_components : int
            削減後の次元数
    
        Returns:
        --------
        X_pca : 主成分得点
        pca : PCお object
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
    
        return X_pca, pca
    
    # 高次元データ生成（100次元）
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # 潜在的な2次元構造を持つデータ
    latent = np.random.randn(n_samples, 2)
    X_high_dim = latent @ np.random.randn(2, n_features) + np.random.randn(n_samples, n_features) * 0.5
    
    # PCA適用
    X_pca, pca_model = apply_pca(X_high_dim, n_components=10)
    
    # 寄与率
    explained_var = pca_model.explained_variance_ratio_
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 寄与率
    axes[0].bar(range(1, 11), explained_var * 100,
                color='steelblue', alpha=0.7)
    axes[0].plot(range(1, 11), np.cumsum(explained_var) * 100,
                 'ro-', linewidth=2, label='累積寄与率')
    axes[0].set_xlabel('主成分', fontsize=12)
    axes[0].set_ylabel('寄与率 (%)', fontsize=12)
    axes[0].set_title('PCA寄与率', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2次元プロット
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                    c='steelblue', s=50, alpha=0.6, edgecolors='k')
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].set_title('PCA可視化（2次元）', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"元の次元数: {n_features}")
    print(f"削減後の次元数: {X_pca.shape[1]}")
    print(f"PC1-PC2の累積寄与率: {np.sum(explained_var[:2]) * 100:.2f}%")
    print(f"PC1-PC10の累積寄与率: {np.sum(explained_var) * 100:.2f}%")
    

### t-SNE, UMAP
    
    
    from sklearn.manifold import TSNE
    # !pip install umap-learn
    from umap import UMAP
    
    def compare_dimensionality_reduction(X, labels=None):
        """
        PCA, t-SNE, UMAPの比較
        """
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
    
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
    
        # UMAP
        umap_model = UMAP(n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(X)
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        # PCA
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                        c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[0].set_xlabel('PC1', fontsize=11)
        axes[0].set_ylabel('PC2', fontsize=11)
        axes[0].set_title('PCA', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
    
        # t-SNE
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                        c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[1].set_xlabel('t-SNE1', fontsize=11)
        axes[1].set_ylabel('t-SNE2', fontsize=11)
        axes[1].set_title('t-SNE', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
    
        # UMAP
        im = axes[2].scatter(X_umap[:, 0], X_umap[:, 1],
                             c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[2].set_xlabel('UMAP1', fontsize=11)
        axes[2].set_ylabel('UMAP2', fontsize=11)
        axes[2].set_title('UMAP', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3)
    
        if labels is not None:
            plt.colorbar(im, ax=axes[2], label='ラベル')
    
        plt.tight_layout()
        plt.show()
    
    # サンプルデータ（3クラス）
    np.random.seed(42)
    class1 = np.random.randn(100, 50) + [2, 2] + np.zeros(48)
    class2 = np.random.randn(100, 50) + [-2, 2] + np.zeros(48)
    class3 = np.random.randn(100, 50) + [0, -2] + np.zeros(48)
    
    X_multi_class = np.vstack([class1, class2, class3])
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    # 比較
    compare_dimensionality_reduction(X_multi_class, labels)
    
    print("次元削減手法の特徴：")
    print("PCA: 線形変換、大域的構造保持、高速")
    print("t-SNE: 非線形変換、局所的構造保持、遅い")
    print("UMAP: 非線形変換、大域+局所構造保持、中速")
    

### LDA (Linear Discriminant Analysis)
    
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    def apply_lda(X, y, n_components=2):
        """
        LDAによる次元削減（教師あり）
    
        Parameters:
        -----------
        X : array-like
            特徴量
        y : array-like
            ラベル
    
        Returns:
        --------
        X_lda : LDA変換後の特徴量
        lda : LDAモデル
        """
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
    
        return X_lda, lda
    
    # LDA適用
    X_lda, lda_model = apply_lda(X_multi_class, labels, n_components=2)
    
    # PCA vs LDA比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA（教師なし）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_multi_class)
    
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                    c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[0].set_xlabel('PC1', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    axes[0].set_title('PCA（教師なし）', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # LDA（教師あり）
    im = axes[1].scatter(X_lda[:, 0], X_lda[:, 1],
                         c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[1].set_xlabel('LD1', fontsize=11)
    axes[1].set_ylabel('LD2', fontsize=11)
    axes[1].set_title('LDA（教師あり）', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.colorbar(im, ax=axes[1], label='クラス')
    plt.tight_layout()
    plt.show()
    
    print("LDAの利点：")
    print("- クラス分離を最大化する射影軸を見つける")
    print("- 分類問題に適している")
    print(f"- 最大次元数: min(n_features, n_classes-1) = {lda_model.n_components}")
    

* * *

## 2.4 特徴量選択

重要な特徴量のみを選択し、モデルの精度と解釈性を向上させます。

### Filter法：相関係数、分散分析
    
    
    from sklearn.feature_selection import (
        VarianceThreshold,
        SelectKBest,
        f_regression,
        mutual_info_regression
    )
    
    def filter_method_selection(X, y, k=10):
        """
        Filter法による特徴量選択
    
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量
        y : array-like
            目的変数
    
        Returns:
        --------
        selected_features : 選択された特徴量名
        scores : 各特徴量のスコア
        """
        # 低分散特徴量除去
        var_threshold = VarianceThreshold(threshold=0.01)
        X_var = var_threshold.fit_transform(X)
        selected_by_var = X.columns[var_threshold.get_support()]
    
        # F値統計量
        selector_f = SelectKBest(f_regression, k=k)
        selector_f.fit(X, y)
        scores_f = selector_f.scores_
        selected_by_f = X.columns[selector_f.get_support()]
    
        # 相互情報量
        selector_mi = SelectKBest(mutual_info_regression, k=k)
        selector_mi.fit(X, y)
        scores_mi = selector_mi.scores_
        selected_by_mi = X.columns[selector_mi.get_support()]
    
        return {
            'variance': selected_by_var,
            'f_stat': selected_by_f,
            'mutual_info': selected_by_mi,
            'scores_f': scores_f,
            'scores_mi': scores_mi
        }
    
    # サンプルデータ
    np.random.seed(42)
    n_samples = 200
    X_data = pd.DataFrame(
        np.random.randn(n_samples, 30),
        columns=[f'feature_{i}' for i in range(30)]
    )
    
    # 目的変数（一部の特徴量のみ関連）
    y_data = (
        2 * X_data['feature_0'] +
        3 * X_data['feature_5'] -
        1.5 * X_data['feature_10'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Filter法実行
    selection_results = filter_method_selection(X_data, y_data, k=10)
    
    # スコア可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F値スコア
    axes[0].bar(range(len(selection_results['scores_f'])),
                selection_results['scores_f'],
                color='steelblue', alpha=0.7)
    axes[0].set_xlabel('特徴量インデックス', fontsize=11)
    axes[0].set_ylabel('F値スコア', fontsize=11)
    axes[0].set_title('F値統計量による特徴量評価', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 相互情報量スコア
    axes[1].bar(range(len(selection_results['scores_mi'])),
                selection_results['scores_mi'],
                color='coral', alpha=0.7)
    axes[1].set_xlabel('特徴量インデックス', fontsize=11)
    axes[1].set_ylabel('相互情報量', fontsize=11)
    axes[1].set_title('相互情報量による特徴量評価', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("F値統計量で選択された特徴量:")
    print(selection_results['f_stat'].tolist())
    print("\n相互情報量で選択された特徴量:")
    print(selection_results['mutual_info'].tolist())
    

### Wrapper法：RFE
    
    
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    
    def rfe_selection(X, y, n_features_to_select=10):
        """
        RFE（Recursive Feature Elimination）
        """
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(X, y)
    
        selected_features = X.columns[selector.support_]
        feature_ranking = selector.ranking_
    
        return selected_features, feature_ranking
    
    # RFE実行
    selected_rfe, ranking_rfe = rfe_selection(X_data, y_data, n_features_to_select=10)
    
    # ランキング可視化
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ranking_rfe)), ranking_rfe,
            color='steelblue', alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--',
                label='選択された特徴量 (rank=1)', linewidth=2)
    plt.xlabel('特徴量インデックス', fontsize=12)
    plt.ylabel('ランキング（低いほど重要）', fontsize=12)
    plt.title('RFEによる特徴量ランキング', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("RFEで選択された特徴量:")
    print(selected_rfe.tolist())
    

### Embedded法：Lasso, Random Forest importances
    
    
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestRegressor
    
    def embedded_selection(X, y):
        """
        Embedded法（Lasso + Random Forest）
        """
        # Lasso（L1正則化）
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X, y)
        lasso_coefs = np.abs(lasso.coef_)
        selected_lasso = X.columns[lasso_coefs > 0]
    
        # Random Forest importances
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importances = rf.feature_importances_
        # 上位10個選択
        top_10_idx = np.argsort(rf_importances)[-10:]
        selected_rf = X.columns[top_10_idx]
    
        return {
            'lasso': selected_lasso,
            'lasso_coefs': lasso_coefs,
            'rf': selected_rf,
            'rf_importances': rf_importances
        }
    
    # Embedded法実行
    embedded_results = embedded_selection(X_data, y_data)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lasso係数
    axes[0].bar(range(len(embedded_results['lasso_coefs'])),
                embedded_results['lasso_coefs'],
                color='steelblue', alpha=0.7)
    axes[0].set_xlabel('特徴量インデックス', fontsize=11)
    axes[0].set_ylabel('|Lasso係数|', fontsize=11)
    axes[0].set_title('Lasso特徴量選択', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Random Forest importances
    axes[1].bar(range(len(embedded_results['rf_importances'])),
                embedded_results['rf_importances'],
                color='coral', alpha=0.7)
    axes[1].set_xlabel('特徴量インデックス', fontsize=11)
    axes[1].set_ylabel('Feature Importance', fontsize=11)
    axes[1].set_title('Random Forest重要度', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Lassoで選択された特徴量:")
    print(embedded_results['lasso'].tolist())
    print("\nRandom Forestで選択された特徴量（上位10）:")
    print(embedded_results['rf'].tolist())
    

### SHAP-based selection
    
    
    import shap
    
    def shap_based_selection(X, y, top_k=10):
        """
        SHAPによる特徴量選択
        """
        # モデル訓練
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
    
        # SHAP値計算
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    
        # 平均絶対SHAP値で重要度評価
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
        # 上位k個選択
        top_k_idx = np.argsort(mean_abs_shap)[-top_k:]
        selected_features = X.columns[top_k_idx]
    
        return selected_features, mean_abs_shap, shap_values
    
    # SHAP選択
    selected_shap, mean_shap, shap_vals = shap_based_selection(X_data, y_data, top_k=10)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_data, plot_type="bar", show=False)
    plt.title('SHAP特徴量重要度', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("SHAPで選択された特徴量（上位10）:")
    print(selected_shap.tolist())
    
    # 手法比較
    print("\n特徴量選択手法の比較：")
    print(f"Filter法（F値）: {len(selection_results['f_stat'])} 特徴量")
    print(f"Wrapper法（RFE）: {len(selected_rfe)} 特徴量")
    print(f"Embedded法（Lasso）: {len(embedded_results['lasso'])} 特徴量")
    print(f"SHAP-based: {len(selected_shap)} 特徴量")
    

* * *

## 2.5 ケーススタディ：バンドギャップ予測

実際のバンドギャップ予測タスクで、200次元から20次元への効果的な削減を実践します。
    
    
    # バンドギャップデータセット（シミュレーション）
    np.random.seed(42)
    n_materials = 500
    
    # 200次元の材料記述子（組成・構造・電子構造）
    descriptor_names = (
        [f'組成_{i}' for i in range(80)] +
        [f'構造_{i}' for i in range(60)] +
        [f'電子_{i}' for i in range(60)]
    )
    
    X_bandgap = pd.DataFrame(
        np.random.randn(n_materials, 200),
        columns=descriptor_names
    )
    
    # バンドギャップ（一部の記述子のみ依存）
    important_features = [
        '組成_5', '組成_12', '組成_25',
        '構造_10', '構造_23',
        '電子_8', '電子_15', '電子_30'
    ]
    
    y_bandgap = np.zeros(n_materials)
    for feat in important_features:
        idx = descriptor_names.index(feat)
        y_bandgap += np.random.uniform(0.5, 1.5) * X_bandgap[feat]
    
    y_bandgap = np.abs(y_bandgap) + np.random.normal(0, 0.3, n_materials)
    y_bandgap = np.clip(y_bandgap, 0, 6)  # 0-6 eVの範囲
    
    print("=== バンドギャップ予測データセット ===")
    print(f"材料数: {n_materials}")
    print(f"特徴量数: {X_bandgap.shape[1]}")
    print(f"目標: 200次元 → 20次元に削減")
    

### Step 1: Filter法で100次元に削減
    
    
    # 相互情報量でトップ100選択
    selector_mi = SelectKBest(mutual_info_regression, k=100)
    X_filtered = selector_mi.fit_transform(X_bandgap, y_bandgap)
    selected_features_100 = X_bandgap.columns[selector_mi.get_support()]
    
    print(f"\nStep 1: Filter法（相互情報量）")
    print(f"200次元 → {X_filtered.shape[1]}次元")
    

### Step 2: PCAで50次元に削減
    
    
    # PCA適用
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_filtered)
    
    # 累積寄与率
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # 90%累積寄与率を達成する次元数
    n_components_90 = np.argmax(cumsum_var >= 0.90) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), cumsum_var * 100, 'b-', linewidth=2)
    plt.axhline(y=90, color='red', linestyle='--',
                label='90%累積寄与率', linewidth=2)
    plt.axvline(x=n_components_90, color='green', linestyle='--',
                label=f'{n_components_90}次元で90%達成', linewidth=2)
    plt.xlabel('主成分数', fontsize=12)
    plt.ylabel('累積寄与率 (%)', fontsize=12)
    plt.title('PCAによる次元削減', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nStep 2: PCA")
    print(f"100次元 → {X_pca.shape[1]}次元")
    print(f"90%累積寄与率達成: {n_components_90}次元")
    

### Step 3: Random Forest Importanceで20次元に削減
    
    
    # 50次元のPCA特徴量でRandom Forest訓練
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(50)])
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_pca_df, y_bandgap)
    
    # 重要度トップ20選択
    importances = rf.feature_importances_
    top_20_idx = np.argsort(importances)[-20:]
    X_final = X_pca_df.iloc[:, top_20_idx]
    
    # 重要度可視化
    plt.figure(figsize=(12, 6))
    plt.bar(range(50), importances, color='steelblue', alpha=0.7)
    plt.bar(top_20_idx, importances[top_20_idx],
            color='coral', alpha=0.9, label='選択された20次元')
    plt.xlabel('主成分インデックス', fontsize=12)
    plt.ylabel('Feature Importance', fontsize=12)
    plt.title('Random Forestによる最終選択', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nStep 3: Random Forest Importance")
    print(f"50次元 → {X_final.shape[1]}次元")
    print(f"\n最終選択された主成分:")
    print(X_final.columns.tolist())
    

### Step 4: 予測性能の検証
    
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    
    def evaluate_dimension_reduction(X, y, name):
        """
        次元削減後の予測性能評価
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        # 交差検証
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()
    
        return {
            'name': name,
            'dimensions': X.shape[1],
            'MAE': mae,
            'R2': r2,
            'CV_MAE': cv_mae
        }
    
    # 各段階の性能評価
    results = []
    
    # 元データ（200次元）
    results.append(evaluate_dimension_reduction(X_bandgap, y_bandgap, '元データ'))
    
    # Filter後（100次元）
    X_filtered_df = pd.DataFrame(X_filtered)
    results.append(evaluate_dimension_reduction(X_filtered_df, y_bandgap, 'Filter法'))
    
    # PCA後（50次元）
    results.append(evaluate_dimension_reduction(X_pca_df, y_bandgap, 'PCA'))
    
    # 最終（20次元）
    results.append(evaluate_dimension_reduction(X_final, y_bandgap, '最終選択'))
    
    # 結果表示
    results_df = pd.DataFrame(results)
    print("\n=== 次元削減の影響評価 ===")
    print(results_df.to_string(index=False))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE比較
    axes[0].bar(results_df['name'], results_df['MAE'],
                color=['gray', 'steelblue', 'coral', 'green'], alpha=0.7)
    axes[0].set_ylabel('MAE (eV)', fontsize=12)
    axes[0].set_title('予測誤差（MAE）', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)
    
    # R²比較
    axes[1].bar(results_df['name'], results_df['R2'],
                color=['gray', 'steelblue', 'coral', 'green'], alpha=0.7)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('決定係数（R²）', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n性能維持率（最終20次元 vs 元200次元）:")
    print(f"R²維持率: {results_df.iloc[3]['R2'] / results_df.iloc[0]['R2'] * 100:.1f}%")
    print(f"次元削減率: {(1 - 20/200) * 100:.0f}%")
    

### 物理的意味づけ
    
    
    # 選択された20次元と元の記述子の対応を分析
    def interpret_selected_components(pca_model, original_features, selected_pcs):
        """
        選択された主成分の物理的解釈
        """
        # PCA loadings（主成分負荷量）
        loadings = pca_model.components_.T
    
        interpretations = []
    
        for pc_name in selected_pcs:
            pc_idx = int(pc_name.replace('PC', '')) - 1
    
            # この主成分への寄与が大きい元の特徴量
            loading_vector = np.abs(loadings[:, pc_idx])
            top_5_idx = np.argsort(loading_vector)[-5:]
    
            top_features = [original_features[i] for i in top_5_idx]
            top_loadings = loading_vector[top_5_idx]
    
            interpretations.append({
                'PC': pc_name,
                'Top_Features': top_features,
                'Loadings': top_loadings
            })
    
        return interpretations
    
    # 解釈実行
    selected_pc_names = X_final.columns.tolist()
    interpretations = interpret_selected_components(
        pca,
        selected_features_100.tolist(),
        selected_pc_names[:5]  # 最初の5個のみ表示
    )
    
    print("\n=== 選択された主成分の物理的解釈 ===")
    for interp in interpretations:
        print(f"\n{interp['PC']}:")
        for feat, loading in zip(interp['Top_Features'], interp['Loadings']):
            print(f"  {feat}: {loading:.3f}")
    

* * *

## 2.6 特徴量エンジニアリングの実践ガイド

### matminerライブラリのバージョン管理
    
    
    # matminerとpymatgenのバージョン確認
    import matminer
    import pymatgen
    
    print("=== 特徴量エンジニアリング環境 ===")
    print(f"matminer: {matminer.__version__}")
    print(f"pymatgen: {pymatgen.__version__}")
    
    # 推奨バージョン
    print("\n【推奨環境】")
    recommended = {
        'matminer': '0.9.0',
        'pymatgen': '2023.9.10',
        'scikit-learn': '1.3.0',
        'numpy': '1.24.3'
    }
    
    for package, version in recommended.items():
        print(f"{package}>={version}")
    
    print("\n【インストールコマンド】")
    print("```bash")
    print("pip install matminer==0.9.0 pymatgen==2023.9.10")
    print("```")
    
    print("\n【注意事項】")
    print("⚠️ matminerのプリセット（magpie, demlなど）はバージョンで特徴量数が変わる")
    print("⚠️ 論文再現時は必ずバージョンを明記")
    

### ベンチマークデータセットの活用
    
    
    # 材料科学で一般的なベンチマークデータセット
    benchmark_datasets = pd.DataFrame({
        'データセット': [
            'Matbench',
            'JARVIS-DFT',
            'Materials Project',
            'OQMD',
            'Expt Gap (Zhuo et al.)'
        ],
        'タスク': [
            '13種類の回帰/分類',
            '55,000材料の物性予測',
            'バンドギャップ、形成エネルギー',
            '安定性予測、相図',
            '実験バンドギャップ（6,354材料）'
        ],
        'サンプル数': [
            '数百〜数万',
            '55,000+',
            '150,000+',
            '1,000,000+',
            '6,354'
        ],
        '用途': [
            'モデル性能比較',
            'GNN・深層学習',
            '汎用材料探索',
            '安定性評価',
            '実験データ検証'
        ],
        'URL': [
            'https://matbench.materialsproject.org/',
            'https://jarvis.nist.gov/',
            'https://materialsproject.org/',
            'http://oqmd.org/',
            'DOI: 10.1021/acs.jpclett.8b00124'
        ]
    })
    
    print("=== ベンチマークデータセット ===")
    print(benchmark_datasets.to_string(index=False))
    
    print("\n【使用例：Matbench】")
    print("```python")
    print("from matbench.bench import MatbenchBenchmark")
    print("mb = MatbenchBenchmark(autoload=False)")
    print("for task in mb.tasks:")
    print("    task.load()")
    print("    print(f'{task.dataset_name}: {len(task.df)} samples')")
    print("```")
    

### 実践的な落とし穴（特徴量エンジニアリング編）
    
    
    print("=== 特徴量エンジニアリングの落とし穴 ===\n")
    
    print("【落とし穴1: Target Leakage（目的変数リーク）】")
    print("❌ 悪い例：目的変数から特徴量を生成")
    print("```python")
    print("# バンドギャップ予測で、バンドギャップに直接関連する値を特徴量化")
    print("X['bandgap_proxy'] = y_bandgap * 0.9 + noise  # NG！")
    print("```")
    
    print("\n✅ 正しい例：独立な物性のみを特徴量化")
    print("```python")
    print("# 組成、構造、電子構造など、目的変数とは独立な記述子")
    print("X_features = generate_composition_features(formulas)")
    print("```")
    
    print("\n【落とし穴2: 特徴量スケールの不統一】")
    print("⚠️ 格子定数（3-7Å）と電気伝導度（10³-10⁶ S/m）を混在")
    print("→ 距離ベースのアルゴリズム（KNN、SVM）で性能低下")
    
    print("\n✅ 対策：StandardScalerで正規化")
    print("```python")
    print("from sklearn.preprocessing import StandardScaler")
    print("scaler = StandardScaler()")
    print("X_scaled = scaler.fit_transform(X)")
    print("```")
    
    print("\n【落とし穴3: 次元削減での情報損失】")
    print("⚠️ PCAで95%寄与率 → 残り5%に重要情報がある可能性")
    
    print("\n✅ 対策：複数の寄与率で性能比較")
    print("```python")
    print("for var_ratio in [0.90, 0.95, 0.99]:")
    print("    pca = PCA(n_components=var_ratio)")
    print("    X_pca = pca.fit_transform(X)")
    print("    # モデル性能評価")
    print("```")
    
    print("\n【落とし穴4: matminerプリセットの無批判な使用】")
    print("⚠️ magpieプリセット（132特徴量）をそのまま使用")
    print("→ 冗長・無関係な特徴量が多数含まれる")
    
    print("\n✅ 対策：ドメイン知識で特徴量を厳選")
    print("```python")
    print("# バンドギャップ予測に関連する特徴量のみ選択")
    print("relevant_features = [")
    print("    'MagpieData mean Electronegativity',")
    print("    'MagpieData range Electronegativity',")
    print("    'MagpieData mean NValence'")
    print("]")
    print("X_selected = X_all[relevant_features]")
    print("```")
    
    print("\n【落とし穴5: 組成記述子の非規格化】")
    print("⚠️ Li₀.₉CoO₂とLiCoO₂で異なる特徴量値")
    print("→ 組成正規化が必要")
    
    print("\n✅ 対策：組成を規格化（合計=1）")
    print("```python")
    print("from pymatgen.core import Composition")
    print("comp = Composition('Li0.9CoO2')")
    print("comp_normalized = comp.fractional_composition  # Li₀.₃₁Co₀.₃₄O₀.₆₉")
    print("```")
    

* * *

## 演習問題

### 問題1（難易度: easy）

matminerを使って、化学式"Fe2O3"と"TiO2"の材料記述子を生成し、どの記述子が最も異なるかを比較してください。

ヒント 1\. `ElementProperty.from_preset("magpie")`を使用 2\. 各化学式で特徴量生成 3\. 差の絶対値を計算し、上位10個を表示  解答例
    
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    
    # 記述子生成
    ep_feat = ElementProperty.from_preset("magpie")
    
    comp1 = Composition("Fe2O3")
    comp2 = Composition("TiO2")
    
    features1 = ep_feat.featurize(comp1)
    features2 = ep_feat.featurize(comp2)
    
    feature_names = ep_feat.feature_labels()
    
    # 差分計算
    df_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Fe2O3': features1,
        'TiO2': features2,
        'Difference': np.abs(np.array(features1) - np.array(features2))
    })
    
    df_comparison_sorted = df_comparison.sort_values(
        'Difference', ascending=False
    )
    
    print("最も異なる記述子（上位10）:")
    print(df_comparison_sorted.head(10).to_string(index=False))
    

### 問題2（難易度: medium）

PCAとUMAPを用いて、高次元材料データを2次元に削減し、どちらがクラスタ構造をより明確に可視化できるか比較してください。

解答例
    
    
    from sklearn.decomposition import PCA
    from umap import UMAP
    from sklearn.datasets import make_blobs
    
    # クラスタ構造を持つ高次元データ生成
    X, y = make_blobs(n_samples=300, n_features=50,
                      centers=3, random_state=42)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # UMAP
    umap_model = UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                    c=y, cmap='viridis', s=50, alpha=0.6)
    axes[0].set_title('PCA', fontsize=12, fontweight='bold')
    
    axes[1].scatter(X_umap[:, 0], X_umap[:, 1],
                    c=y, cmap='viridis', s=50, alpha=0.6)
    axes[1].set_title('UMAP', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

### 問題3（難易度: hard）

Filter法、Wrapper法、Embedded法、SHAP-basedの4つの特徴量選択手法を用いて、同じデータセットに対してそれぞれ上位10特徴量を選択し、選択された特徴量の重複度を分析してください。

解答例
    
    
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso
    import shap
    
    # サンプルデータ
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 30),
                     columns=[f'feat_{i}' for i in range(30)])
    y = (2*X['feat_0'] + 3*X['feat_5'] - X['feat_10'] +
         np.random.normal(0, 0.5, 200))
    
    # 1. Filter法
    selector_filter = SelectKBest(f_regression, k=10)
    selector_filter.fit(X, y)
    selected_filter = set(X.columns[selector_filter.get_support()])
    
    # 2. Wrapper法（RFE）
    model_rfe = RandomForestRegressor(n_estimators=50, random_state=42)
    selector_rfe = RFE(model_rfe, n_features_to_select=10)
    selector_rfe.fit(X, y)
    selected_rfe = set(X.columns[selector_rfe.support_])
    
    # 3. Embedded法（Lasso）
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X, y)
    lasso_coefs = np.abs(lasso.coef_)
    top_10_lasso_idx = np.argsort(lasso_coefs)[-10:]
    selected_lasso = set(X.columns[top_10_lasso_idx])
    
    # 4. SHAP-based
    rf_shap = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_shap.fit(X, y)
    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_10_shap_idx = np.argsort(mean_abs_shap)[-10:]
    selected_shap = set(X.columns[top_10_shap_idx])
    
    # 重複分析
    all_methods = {
        'Filter': selected_filter,
        'Wrapper': selected_rfe,
        'Embedded': selected_lasso,
        'SHAP': selected_shap
    }
    
    # ベン図的な重複計算
    common_all = selected_filter & selected_rfe & selected_lasso & selected_shap
    
    print("各手法で選択された特徴量:")
    for method, features in all_methods.items():
        print(f"{method}: {sorted(features)}")
    
    print(f"\n全手法共通: {sorted(common_all)}")
    print(f"共通特徴量数: {len(common_all)} / 10")
    

* * *

## まとめ

この章では、**特徴量エンジニアリング** の実践手法を学びました。

**重要ポイント** ：

  1. **材料記述子** ：組成・構造・電子構造記述子をmatminerで効率的に生成
  2. **特徴量変換** ：正規化・対数変換・多項式特徴量で非線形関係を捉える
  3. **次元削減** ：PCA、t-SNE、UMAPで可視化と計算効率化
  4. **特徴量選択** ：Filter < Wrapper < Embedded < SHAP の順で精度向上
  5. **実践事例** ：200次元→20次元で性能を維持しつつ解釈性向上
  6. **ライブラリ管理** ：matminer、pymatgenのバージョン管理と再現性確保
  7. **ベンチマークデータ** ：Matbench、JARVIS-DFTなど標準データセットの活用
  8. **実践的落とし穴** ：Target leakage、スケール不統一、情報損失、組成規格化

**次章予告** ： Chapter 3では、最適なモデル選択とハイパーパラメータ最適化を学びます。Optunaを用いた自動最適化とアンサンブル学習で予測精度を最大化します。

* * *

## Chapter 2 チェックリスト

### 特徴量生成（組成記述子）

  * [ ] **matminer活用**
  * [ ] ElementProperty.from_preset("magpie")で132特徴量生成
  * [ ] Stoichiometry()で化学量論記述子（7特徴量）
  * [ ] ValenceOrbital()で価電子軌道記述子（10特徴量）
  * [ ] バージョン記録（matminer 0.9.0、pymatgen 2023.9.10）

  * [ ] **組成の規格化**

  * [ ] Composition.fractional_compositionで合計=1に正規化
  * [ ] Li₀.₉CoO₂ とLiCoO₂の扱いを統一
  * [ ] ドーパント濃度の適切な表現

  * [ ] **ドメイン知識の統合**

  * [ ] 電気陰性度、原子半径、価電子数などの化学的意味を理解
  * [ ] バンドギャップ予測には電子構造記述子を優先
  * [ ] 機械的特性予測には結晶構造記述子を優先

### 特徴量生成（構造・電子構造記述子）

  * [ ] **結晶構造記述子**
  * [ ] 格子定数（a, b, c）、格子角度（α, β, γ）
  * [ ] 体積、パッキング密度
  * [ ] 空間群、対称性

  * [ ] **電子構造記述子**

  * [ ] バンドギャップ、フェルミエネルギー
  * [ ] 状態密度（価電子帯・伝導帯）
  * [ ] 有効質量（電子・正孔）

  * [ ] **DFT計算データ連携**

  * [ ] Materials Project APIで形成エネルギー取得
  * [ ] OQMDで安定性データ取得
  * [ ] 計算条件（汎関数、カットオフエネルギー）を記録

### 特徴量変換

  * [ ] **正規化**
  * [ ] StandardScaler（平均0、標準偏差1）をデフォルト使用
  * [ ] MinMaxScaler（0-1範囲）は解釈性重視時
  * [ ] Train/Test分割**後** にfit（データリーク防止）

  * [ ] **対数変換**

  * [ ] 電気伝導度など桁数が大きい特徴量に適用
  * [ ] log1p（log(1+x)）でゼロ値を扱う
  * [ ] 歪度（skewness）を確認し、変換後に-0.5〜0.5に収める

  * [ ] **多項式特徴量**

  * [ ] 2次項で非線形関係をキャプチャ
  * [ ] 交互作用項（x₁×x₂）で相互作用を明示
  * [ ] 次数3以上は過学習リスク → 慎重に

### 次元削減

  * [ ] **PCA（主成分分析）**
  * [ ] 累積寄与率90-95%を目標
  * [ ] 主成分負荷量で物理的解釈
  * [ ] スクリープロット（寄与率の減衰）を可視化

  * [ ] **t-SNE / UMAP**

  * [ ] クラスタ可視化に有効（2次元プロット）
  * [ ] 予測モデルの特徴量には使用しない（非線形で不可逆）
  * [ ] パープレキシティ（t-SNE）、n_neighbors（UMAP）をチューニング

  * [ ] **LDA（線形判別分析）**

  * [ ] 分類問題でクラス分離を最大化
  * [ ] 次元数は min(n_features, n_classes-1)
  * [ ] 教師あり学習なので汎化性能に注意

### 特徴量選択

  * [ ] **Filter法**
  * [ ] VarianceThreshold（低分散特徴量除去）
  * [ ] SelectKBest + f_regression（F値統計量）
  * [ ] SelectKBest + mutual_info_regression（相互情報量）
  * [ ] 計算高速だが、特徴量間の相互作用を考慮しない

  * [ ] **Wrapper法**

  * [ ] RFE（Recursive Feature Elimination）
  * [ ] 計算コスト高いが精度高い
  * [ ] 10-20特徴量への削減に有効

  * [ ] **Embedded法**

  * [ ] Lasso（L1正則化）で係数ゼロの特徴量を自動除去
  * [ ] Random Forest feature_importances_
  * [ ] LightGBM feature_importances_
  * [ ] 訓練過程で特徴量選択が行われる

  * [ ] **SHAP-based選択**

  * [ ] mean(|SHAP値|)で重要度ランキング
  * [ ] Global解釈として最も信頼性が高い
  * [ ] 計算コストは中程度

### 実践的落とし穴の回避（特徴量エンジニアリング）

  * [ ] **Target Leakage防止**
  * [ ] 目的変数から特徴量を生成しない
  * [ ] DFT計算値（バンドギャップなど）を直接使わない
  * [ ] 組成・構造など独立な記述子のみ使用

  * [ ] **スケール統一**

  * [ ] 格子定数（Å）と電気伝導度（S/m）を正規化
  * [ ] 距離ベースモデル（KNN、SVM）で特に重要
  * [ ] 木ベースモデル（RF、XGBoost）では不要だが推奨

  * [ ] **情報損失の監視**

  * [ ] PCA寄与率を複数（90%, 95%, 99%）で比較
  * [ ] 次元削減前後のモデル性能を評価
  * [ ] 削減しすぎで性能低下なら次元数を増やす

  * [ ] **matminerプリセットの吟味**

  * [ ] magpie（132特徴量）は多すぎる場合あり
  * [ ] ドメイン知識で関連特徴量を選定
  * [ ] 相関行列で冗長な特徴量を除去

  * [ ] **組成規格化**

  * [ ] 組成合計が1になるよう正規化
  * [ ] 化学式表記の揺れ（Li₀.₉CoO₂ vs LiCoO₂）を統一
  * [ ] Composition.fractional_composition使用

### ベンチマークデータセットの活用

  * [ ] **Matbench**
  * [ ] 13種類のタスクで標準的性能評価
  * [ ] 論文比較時の共通基準
  * [ ] モデル選択の妥当性検証

  * [ ] **JARVIS-DFT**

  * [ ] 55,000材料のDFT計算データ
  * [ ] GNNやTransformerの訓練に最適
  * [ ] 構造記述子の豊富さが特徴

  * [ ] **実験データセット**

  * [ ] Expt Gap（実験バンドギャップ 6,354材料）
  * [ ] DFT vs 実験のギャップを検証
  * [ ] モデルの実用性評価

### 特徴量エンジニアリング品質指標

  * [ ] **特徴量の質**
  * [ ] 欠損率 < 5%（特徴量レベル）
  * [ ] 分散 > 0.01（低分散特徴量を除去）
  * [ ] 相関係数 |r| < 0.9（高相関ペアを削減）

  * [ ] **次元削減の効果**

  * [ ] PCA寄与率 ≥ 90%
  * [ ] 次元削減率 50-90%
  * [ ] 性能維持率 ≥ 95%（MAE、R²など）

  * [ ] **特徴量選択の効果**

  * [ ] 選択後の特徴量数 ≤ 20%（元の数）
  * [ ] 性能改善または維持（過学習防止）
  * [ ] 訓練時間短縮 ≥ 30%

### 再現性の確保

  * [ ] **バージョン管理**
  * [ ] matminer、pymatgenのバージョン明記
  * [ ] プリセット名（magpie、demlなど）を記録
  * [ ] 特徴量生成日時を記録

  * [ ] **特徴量リストの保存**

  * [ ] 生成された特徴量名を全てCSV保存
  * [ ] 特徴量の意味（説明）をドキュメント化
  * [ ] 選択された特徴量のサブセットも記録

  * [ ] **変換パラメータの保存**

  * [ ] StandardScalerのmean、stdをpickle保存
  * [ ] PCAのcomponents_を保存
  * [ ] 新データへの適用時に同じ変換を適用

* * *

## 参考文献

  1. **Ward, L., Dunn, A., Faghaninia, A., et al.** (2018). Matminer: An open source toolkit for materials data mining. _Computational Materials Science_ , 152, 60-69. [DOI: 10.1016/j.commatsci.2018.05.018](<https://doi.org/10.1016/j.commatsci.2018.05.018>)

  2. **Jolliffe, I. T. & Cadima, J.** (2016). Principal component analysis: a review and recent developments. _Philosophical Transactions of the Royal Society A_ , 374(2065), 20150202. [DOI: 10.1098/rsta.2015.0202](<https://doi.org/10.1098/rsta.2015.0202>)

  3. **McInnes, L., Healy, J., & Melville, J.** (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. _arXiv preprint arXiv:1802.03426_.

  4. **Guyon, I. & Elisseeff, A.** (2003). An introduction to variable and feature selection. _Journal of Machine Learning Research_ , 3, 1157-1182.

* * *

[← Chapter 1に戻る](<chapter-1.html>) | [Chapter 3へ進む →](<chapter-3.html>)
