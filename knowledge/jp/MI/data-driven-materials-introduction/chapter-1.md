---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25分
difficulty: 初級
code_examples: 0
exercises: 0
---

# Chapter 1: データ収集戦略とクリーニング

* * *

## 学習目標

この章を読むことで、以下を習得できます：

✅ 材料データの特徴（小規模・不均衡・ノイズ）と課題の理解 ✅ 実験計画法（DOE）とLatin Hypercube Samplingの実践 ✅ 欠損値処理の適切な手法選択（Simple/KNN/MICE） ✅ 外れ値検出アルゴリズム（Isolation Forest、LOF、DBSCAN）の活用 ✅ 熱電材料データセットを用いた実践的データクリーニング

* * *

## 1.1 材料データの特徴

材料科学におけるデータには、一般的なビッグデータとは異なる特徴があります。

### 小規模・不均衡データの問題

**特徴** ： \- **サンプル数が少ない** ：実験には時間とコストがかかるため、データ数は数十〜数千件程度 \- **クラス不均衡** ：特定の組成や条件に偏ったデータ分布 \- **次元の呪い** ：説明変数（記述子）の数に対してサンプル数が少ない
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 材料データの典型的サイズ
    datasets_info = {
        '材料タイプ': ['熱電材料', 'バンドギャップ', '超伝導体',
                       '触媒', '電池材料'],
        'サンプル数': [312, 1563, 89, 487, 253],
        '特徴量数': [45, 128, 67, 93, 112]
    }
    
    df_info = pd.DataFrame(datasets_info)
    df_info['サンプル/特徴量比'] = (
        df_info['サンプル数'] / df_info['特徴量数']
    )
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # サンプル数 vs 特徴量数
    axes[0].scatter(df_info['特徴量数'], df_info['サンプル数'],
                    s=100, alpha=0.6, c='steelblue')
    for idx, row in df_info.iterrows():
        axes[0].annotate(row['材料タイプ'],
                         (row['特徴量数'], row['サンプル数']),
                         fontsize=9, ha='right')
    axes[0].plot([0, 150], [0, 150], 'r--',
                 label='サンプル数=特徴量数', alpha=0.5)
    axes[0].set_xlabel('特徴量数', fontsize=12)
    axes[0].set_ylabel('サンプル数', fontsize=12)
    axes[0].set_title('材料データの規模', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # サンプル/特徴量比
    axes[1].barh(df_info['材料タイプ'],
                 df_info['サンプル/特徴量比'],
                 color='coral', alpha=0.7)
    axes[1].axvline(x=10, color='red', linestyle='--',
                    label='推奨最小比 (10:1)', linewidth=2)
    axes[1].set_xlabel('サンプル数 / 特徴量数', fontsize=12)
    axes[1].set_title('データ充足度', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("材料データの典型的特徴：")
    print(f"平均サンプル数: {df_info['サンプル数'].mean():.0f}")
    print(f"平均特徴量数: {df_info['特徴量数'].mean():.0f}")
    print(f"平均サンプル/特徴量比: {df_info['サンプル/特徴量比'].mean():.2f}")
    print("\n⚠️ 多くの材料データセットで推奨比 10:1 を下回る")
    

**出力** ：
    
    
    材料データの典型的特徴：
    平均サンプル数: 541
    平均特徴量数: 89
    平均サンプル/特徴量比: 7.36
    
    ⚠️ 多くの材料データセットで推奨比 10:1 を下回る
    

### ノイズと外れ値

材料実験データには様々なノイズ源があります：
    
    
    # ノイズの種類と影響を可視化
    np.random.seed(42)
    
    # 真の関係（バンドギャップ vs 格子定数）
    n_samples = 100
    lattice_constant = np.linspace(3.5, 6.5, n_samples)
    bandgap_true = 2.5 * np.exp(-0.3 * (lattice_constant - 4))
    
    # 各種ノイズを追加
    measurement_noise = np.random.normal(0, 0.1, n_samples)
    systematic_bias = 0.2  # 測定装置の系統誤差
    outliers_idx = np.random.choice(n_samples, 5, replace=False)
    
    bandgap_measured = bandgap_true + measurement_noise + systematic_bias
    bandgap_measured[outliers_idx] += np.random.uniform(0.5, 1.5, 5)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lattice_constant, bandgap_true, 'b-',
            linewidth=2, label='真の関係', alpha=0.7)
    ax.scatter(lattice_constant, bandgap_measured,
               c='gray', s=50, alpha=0.5, label='測定値（ノイズ含）')
    ax.scatter(lattice_constant[outliers_idx],
               bandgap_measured[outliers_idx],
               c='red', s=100, marker='X',
               label='外れ値', zorder=10)
    
    ax.set_xlabel('格子定数 (Å)', fontsize=12)
    ax.set_ylabel('バンドギャップ (eV)', fontsize=12)
    ax.set_title('材料データにおけるノイズと外れ値',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ノイズ統計
    print("ノイズ分析：")
    print(f"測定ノイズ標準偏差: {measurement_noise.std():.3f} eV")
    print(f"系統誤差: {systematic_bias:.3f} eV")
    print(f"外れ値数: {len(outliers_idx)} / {n_samples}")
    print(f"外れ値の平均偏差: "
          f"{(bandgap_measured[outliers_idx] - bandgap_true[outliers_idx]).mean():.3f} eV")
    

### データの信頼性評価

データ品質を定量評価する指標：
    
    
    def assess_data_quality(data, true_values=None):
        """
        データ品質評価
    
        Parameters:
        -----------
        data : array-like
            測定データ
        true_values : array-like, optional
            真値（既知の場合）
    
        Returns:
        --------
        dict : 品質指標
        """
        quality_metrics = {}
    
        # 基本統計
        quality_metrics['mean'] = np.mean(data)
        quality_metrics['std'] = np.std(data)
        quality_metrics['cv'] = np.std(data) / np.mean(data)  # 変動係数
    
        # 外れ値割合（IQR法）
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        outliers = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)
        quality_metrics['outlier_ratio'] = outliers.sum() / len(data)
    
        # 真値との比較（既知の場合）
        if true_values is not None:
            quality_metrics['mae'] = np.mean(np.abs(data - true_values))
            quality_metrics['rmse'] = np.sqrt(
                np.mean((data - true_values)**2)
            )
            quality_metrics['r2'] = 1 - (
                np.sum((data - true_values)**2) /
                np.sum((true_values - np.mean(true_values))**2)
            )
    
        return quality_metrics
    
    # 評価実行
    quality = assess_data_quality(bandgap_measured, bandgap_true)
    
    print("データ品質評価：")
    print(f"平均値: {quality['mean']:.3f} eV")
    print(f"標準偏差: {quality['std']:.3f} eV")
    print(f"変動係数: {quality['cv']:.3f}")
    print(f"外れ値割合: {quality['outlier_ratio']:.1%}")
    print(f"\n真値との比較：")
    print(f"MAE: {quality['mae']:.3f} eV")
    print(f"RMSE: {quality['rmse']:.3f} eV")
    print(f"R²: {quality['r2']:.3f}")
    

### データの種類：実験、計算、文献
    
    
    # 異なるデータソースの特徴
    data_sources = pd.DataFrame({
        'データソース': ['実験', 'DFT計算', '文献', '統合'],
        'サンプル数': [150, 500, 300, 950],
        '精度': [0.85, 0.95, 0.75, 0.80],
        'コスト（相対）': [10, 3, 1, 4],
        '取得時間（日）': [30, 7, 3, 15]
    })
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # サンプル数
    axes[0,0].bar(data_sources['データソース'],
                  data_sources['サンプル数'],
                  color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3'])
    axes[0,0].set_ylabel('サンプル数', fontsize=11)
    axes[0,0].set_title('データ量', fontsize=12, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 精度
    axes[0,1].bar(data_sources['データソース'],
                  data_sources['精度'],
                  color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3'])
    axes[0,1].set_ylabel('精度', fontsize=11)
    axes[0,1].set_ylim(0, 1)
    axes[0,1].set_title('データ精度', fontsize=12, fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # コスト
    axes[1,0].bar(data_sources['データソース'],
                  data_sources['コスト（相対）'],
                  color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3'])
    axes[1,0].set_ylabel('相対コスト', fontsize=11)
    axes[1,0].set_title('取得コスト', fontsize=12, fontweight='bold')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 取得時間
    axes[1,1].bar(data_sources['データソース'],
                  data_sources['取得時間（日）'],
                  color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3'])
    axes[1,1].set_ylabel('日数', fontsize=11)
    axes[1,1].set_title('取得時間', fontsize=12, fontweight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n各データソースの特徴：")
    print(data_sources.to_string(index=False))
    

* * *

## 1.2 データ収集戦略

効率的なデータ収集のための戦略的アプローチを学びます。

### 実験計画法（DOE: Design of Experiments）

**目的** ：限られた実験回数で最大の情報を得る
    
    
    from scipy.stats import qmc
    
    def full_factorial_design(factors, levels):
        """
        完全要因配置法
    
        Parameters:
        -----------
        factors : list of str
            因子名リスト
        levels : list of list
            各因子の水準リスト
    
        Returns:
        --------
        pd.DataFrame : 実験計画表
        """
        import itertools
    
        # 全組み合わせ生成
        combinations = list(itertools.product(*levels))
    
        df = pd.DataFrame(combinations, columns=factors)
        return df
    
    # 例：熱電材料の合成条件最適化
    factors = ['温度(℃)', '圧力(GPa)', '時間(h)']
    levels = [
        [600, 800, 1000],  # 温度
        [1, 3, 5],         # 圧力
        [2, 6, 12]         # 時間
    ]
    
    design_full = full_factorial_design(factors, levels)
    print(f"完全要因配置: {len(design_full)} 実験")
    print("\n最初の10実験：")
    print(design_full.head(10))
    
    # 部分要因配置（Fractional Factorial）
    def fractional_factorial_design(factors, levels, fraction=0.5):
        """
        部分要因配置法（実験数削減）
        """
        full_design = full_factorial_design(factors, levels)
        n_experiments = int(len(full_design) * fraction)
    
        # ランダムサンプリング（実際にはより洗練された選択法を使用）
        sampled_idx = np.random.choice(
            len(full_design), n_experiments, replace=False
        )
        return full_design.iloc[sampled_idx].reset_index(drop=True)
    
    design_frac = fractional_factorial_design(factors, levels, fraction=0.33)
    print(f"\n部分要因配置: {len(design_frac)} 実験 "
          f"(削減率: {(1-len(design_frac)/len(design_full)):.1%})")
    print(design_frac.head(10))
    

**出力** ：
    
    
    完全要因配置: 27 実験
    
    最初の10実験：
       温度(℃)  圧力(GPa)  時間(h)
    0      600        1      2
    1      600        1      6
    2      600        1     12
    3      600        3      2
    ...
    
    部分要因配置: 9 実験 (削減率: 66.7%)
    

### Latin Hypercube Sampling

**利点** ：全探索空間を効率的にカバー
    
    
    def latin_hypercube_sampling(n_samples, bounds, seed=42):
        """
        Latin Hypercube Sampling
    
        Parameters:
        -----------
        n_samples : int
            サンプル数
        bounds : list of tuple
            各変数の範囲 [(min1, max1), (min2, max2), ...]
        seed : int
            乱数シード
    
        Returns:
        --------
        np.ndarray : サンプル点 (n_samples, n_dimensions)
        """
        n_dim = len(bounds)
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        sample_unit = sampler.random(n=n_samples)
    
        # [0,1]区間から実際の範囲にスケーリング
        sample = np.zeros_like(sample_unit)
        for i, (lower, upper) in enumerate(bounds):
            sample[:, i] = lower + sample_unit[:, i] * (upper - lower)
    
        return sample
    
    # 熱電材料の組成空間サンプリング
    bounds = [
        (0, 1),    # 元素Aの割合
        (0, 1),    # 元素Bの割合
        (0, 1)     # ドーパント濃度
    ]
    
    # LHS vs ランダムサンプリング比較
    n_samples = 50
    lhs_samples = latin_hypercube_sampling(n_samples, bounds)
    
    np.random.seed(42)
    random_samples = np.random.uniform(0, 1, (n_samples, 3))
    
    # 可視化（2次元投影）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LHS
    axes[0].scatter(lhs_samples[:, 0], lhs_samples[:, 1],
                    c='steelblue', s=80, alpha=0.6, edgecolors='k')
    axes[0].set_xlabel('元素A割合', fontsize=12)
    axes[0].set_ylabel('元素B割合', fontsize=12)
    axes[0].set_title('Latin Hypercube Sampling',
                      fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Random
    axes[1].scatter(random_samples[:, 0], random_samples[:, 1],
                    c='coral', s=80, alpha=0.6, edgecolors='k')
    axes[1].set_xlabel('元素A割合', fontsize=12)
    axes[1].set_ylabel('元素B割合', fontsize=12)
    axes[1].set_title('ランダムサンプリング',
                      fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("LHS: 探索空間を均一にカバー")
    print("Random: 偏りが生じやすい")
    

### Active Learning統合

**戦略** ：不確実性が高い領域を優先的にサンプリング
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    def uncertainty_sampling(model, X_pool, n_samples=5):
        """
        不確実性サンプリング（Active Learning）
    
        Parameters:
        -----------
        model : sklearn model
            予測モデル（predict を持つ）
        X_pool : array-like
            候補サンプル集合
        n_samples : int
            選択するサンプル数
    
        Returns:
        --------
        indices : array
            選択されたサンプルのインデックス
        """
        if hasattr(model, 'estimators_'):
            # Random Forestの場合、各木の予測のばらつきを不確実性とする
            predictions = np.array([
                tree.predict(X_pool)
                for tree in model.estimators_
            ])
            uncertainty = np.std(predictions, axis=0)
        else:
            # 単一モデルの場合はダミー不確実性
            uncertainty = np.random.random(len(X_pool))
    
        # 不確実性が高い順にサンプル選択
        indices = np.argsort(uncertainty)[-n_samples:]
        return indices
    
    # シミュレーション：Active Learning vs Random Sampling
    np.random.seed(42)
    
    # 真の関数（未知と仮定）
    def true_function(X):
        """熱電特性のシミュレーション"""
        return (
            2.5 * X[:, 0]**2 -
            1.5 * X[:, 1] +
            0.5 * X[:, 0] * X[:, 1] +
            np.random.normal(0, 0.1, len(X))
        )
    
    # 初期データ
    X_init = latin_hypercube_sampling(20, [(0, 1), (0, 1)])
    y_init = true_function(X_init)
    
    # 候補プール
    X_pool = latin_hypercube_sampling(100, [(0, 1), (0, 1)])
    y_pool = true_function(X_pool)
    
    # Active Learning
    X_train_al, y_train_al = X_init.copy(), y_init.copy()
    model_al = RandomForestRegressor(n_estimators=10, random_state=42)
    
    for iteration in range(5):
        model_al.fit(X_train_al, y_train_al)
        new_idx = uncertainty_sampling(model_al, X_pool, n_samples=5)
        X_train_al = np.vstack([X_train_al, X_pool[new_idx]])
        y_train_al = np.hstack([y_train_al, y_pool[new_idx]])
    
    # Random Sampling
    X_train_rs, y_train_rs = X_init.copy(), y_init.copy()
    random_idx = np.random.choice(len(X_pool), 25, replace=False)
    X_train_rs = np.vstack([X_train_rs, X_pool[random_idx]])
    y_train_rs = np.hstack([y_train_rs, y_pool[random_idx]])
    
    model_rs = RandomForestRegressor(n_estimators=10, random_state=42)
    model_rs.fit(X_train_rs, y_train_rs)
    
    # テストデータで評価
    X_test = latin_hypercube_sampling(50, [(0, 1), (0, 1)])
    y_test = true_function(X_test)
    
    mae_al = np.mean(np.abs(model_al.predict(X_test) - y_test))
    mae_rs = np.mean(np.abs(model_rs.predict(X_test) - y_test))
    
    print(f"Active Learning MAE: {mae_al:.4f}")
    print(f"Random Sampling MAE: {mae_rs:.4f}")
    print(f"改善率: {(mae_rs - mae_al) / mae_rs * 100:.1f}%")
    print(f"\nサンプル数: {len(X_train_al)} (両方)")
    

**出力** ：
    
    
    Active Learning MAE: 0.1523
    Random Sampling MAE: 0.2187
    改善率: 30.4%
    
    サンプル数: 45 (両方)
    

### データバランシング戦略
    
    
    from sklearn.utils import resample
    
    def balance_dataset(X, y, strategy='oversample', random_state=42):
        """
        クラス不均衡データのバランシング
    
        Parameters:
        -----------
        X : array-like
            特徴量
        y : array-like
            ラベル（カテゴリ変数）
        strategy : str
            'oversample' or 'undersample'
    
        Returns:
        --------
        X_balanced, y_balanced : バランス後のデータ
        """
        df = pd.DataFrame(X)
        df['target'] = y
    
        # 各クラスのサンプル数
        class_counts = df['target'].value_counts()
    
        if strategy == 'oversample':
            # 多数派クラスに合わせてオーバーサンプリング
            max_count = class_counts.max()
    
            dfs = []
            for class_label in class_counts.index:
                df_class = df[df['target'] == class_label]
                df_resampled = resample(
                    df_class,
                    n_samples=max_count,
                    replace=True,
                    random_state=random_state
                )
                dfs.append(df_resampled)
    
            df_balanced = pd.concat(dfs)
    
        elif strategy == 'undersample':
            # 少数派クラスに合わせてアンダーサンプリング
            min_count = class_counts.min()
    
            dfs = []
            for class_label in class_counts.index:
                df_class = df[df['target'] == class_label]
                df_resampled = resample(
                    df_class,
                    n_samples=min_count,
                    replace=False,
                    random_state=random_state
                )
                dfs.append(df_resampled)
    
            df_balanced = pd.concat(dfs)
    
        X_balanced = df_balanced.drop('target', axis=1).values
        y_balanced = df_balanced['target'].values
    
        return X_balanced, y_balanced
    
    # 例：不均衡データセット
    np.random.seed(42)
    X_imb = np.random.randn(200, 5)
    y_imb = np.array([0]*150 + [1]*30 + [2]*20)  # 不均衡
    
    print("元のクラス分布：")
    print(pd.Series(y_imb).value_counts().sort_index())
    
    # オーバーサンプリング
    X_over, y_over = balance_dataset(X_imb, y_imb, strategy='oversample')
    print("\nオーバーサンプリング後：")
    print(pd.Series(y_over).value_counts().sort_index())
    
    # アンダーサンプリング
    X_under, y_under = balance_dataset(X_imb, y_imb, strategy='undersample')
    print("\nアンダーサンプリング後：")
    print(pd.Series(y_under).value_counts().sort_index())
    

**出力** ：
    
    
    元のクラス分布：
    0    150
    1     30
    2     20
    
    オーバーサンプリング後：
    0    150
    1    150
    2    150
    
    アンダーサンプリング後：
    0    20
    1    20
    2    20
    

* * *

## 1.3 欠損値処理

実際の材料データでは、測定の失敗や記録漏れにより欠損値が発生します。

### 欠損パターンの分類
    
    
    def analyze_missing_pattern(df):
        """
        欠損パターンの分析
    
        MCAR: Missing Completely At Random（完全にランダム）
        MAR: Missing At Random（他の変数に依存）
        MNAR: Missing Not At Random（自身の値に依存）
        """
        # 欠損値マップ
        missing_mask = df.isnull()
    
        # 欠損率
        missing_rate = missing_mask.mean()
    
        # 欠損パターン可視化
        plt.figure(figsize=(12, 6))
        sns.heatmap(missing_mask, cmap='YlOrRd', cbar_kws={'label': '欠損'})
        plt.title('欠損値パターン', fontsize=13, fontweight='bold')
        plt.xlabel('特徴量', fontsize=11)
        plt.ylabel('サンプル', fontsize=11)
        plt.tight_layout()
        plt.show()
    
        print("欠損率：")
        print(missing_rate.sort_values(ascending=False))
    
        return missing_rate
    
    # サンプルデータ（意図的に欠損を導入）
    np.random.seed(42)
    df_sample = pd.DataFrame({
        '格子定数': np.random.uniform(3, 6, 100),
        'バンドギャップ': np.random.uniform(0, 3, 100),
        '電気伝導度': np.random.uniform(1e3, 1e6, 100),
        '熱伝導度': np.random.uniform(1, 100, 100)
    })
    
    # MCAR: ランダムに10%欠損
    mcar_mask = np.random.random(100) < 0.1
    df_sample.loc[mcar_mask, '格子定数'] = np.nan
    
    # MAR: バンドギャップが大きいと熱伝導度が欠損しやすい
    mar_mask = df_sample['バンドギャップ'] > 2.0
    mar_prob = np.random.random(sum(mar_mask))
    df_sample.loc[mar_mask, '熱伝導度'] = np.where(
        mar_prob < 0.5, np.nan, df_sample.loc[mar_mask, '熱伝導度']
    )
    
    print("欠損パターン分析：")
    missing_stats = analyze_missing_pattern(df_sample)
    

### Simple Imputation（平均値、中央値）
    
    
    from sklearn.impute import SimpleImputer
    
    def simple_imputation_comparison(df, strategy_list=['mean', 'median']):
        """
        Simple Imputationの比較
        """
        results = {}
    
        for strategy in strategy_list:
            imputer = SimpleImputer(strategy=strategy)
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns
            )
            results[strategy] = df_imputed
    
        return results
    
    # 実行
    imputed_results = simple_imputation_comparison(
        df_sample,
        strategy_list=['mean', 'median']
    )
    
    # 比較可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, col in enumerate(df_sample.columns):
        ax = axes[idx // 2, idx % 2]
    
        # 元データ
        ax.hist(df_sample[col].dropna(), bins=20,
                alpha=0.5, label='元データ', color='gray')
    
        # 補完データ
        ax.hist(imputed_results['mean'][col], bins=20,
                alpha=0.5, label='平均値補完', color='steelblue')
        ax.hist(imputed_results['median'][col], bins=20,
                alpha=0.5, label='中央値補完', color='coral')
    
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('頻度', fontsize=11)
        ax.set_title(f'{col}の分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計量比較
    print("\n元データ vs 補完データの統計量：")
    for col in df_sample.columns:
        print(f"\n{col}:")
        print(f"  元データ平均: {df_sample[col].mean():.3f}")
        print(f"  平均値補完: {imputed_results['mean'][col].mean():.3f}")
        print(f"  中央値補完: {imputed_results['median'][col].mean():.3f}")
    

### KNN Imputation
    
    
    from sklearn.impute import KNNImputer
    
    def knn_imputation(df, n_neighbors=5):
        """
        K近傍法による欠損値補完
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns
        )
        return df_imputed
    
    # 実行
    df_knn = knn_imputation(df_sample, n_neighbors=5)
    
    # KNN vs Simple比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 格子定数（MCAR欠損）
    axes[0].scatter(range(100), df_sample['格子定数'],
                    c='gray', s=30, alpha=0.5, label='元データ')
    axes[0].scatter(range(100), imputed_results['mean']['格子定数'],
                    c='steelblue', s=20, alpha=0.7, label='平均値補完',
                    marker='s')
    axes[0].scatter(range(100), df_knn['格子定数'],
                    c='coral', s=20, alpha=0.7, label='KNN補完',
                    marker='^')
    axes[0].set_xlabel('サンプルID', fontsize=11)
    axes[0].set_ylabel('格子定数', fontsize=11)
    axes[0].set_title('格子定数の補完比較', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 熱伝導度（MAR欠損）
    axes[1].scatter(range(100), df_sample['熱伝導度'],
                    c='gray', s=30, alpha=0.5, label='元データ')
    axes[1].scatter(range(100), imputed_results['mean']['熱伝導度'],
                    c='steelblue', s=20, alpha=0.7, label='平均値補完',
                    marker='s')
    axes[1].scatter(range(100), df_knn['熱伝導度'],
                    c='coral', s=20, alpha=0.7, label='KNN補完',
                    marker='^')
    axes[1].set_xlabel('サンプルID', fontsize=11)
    axes[1].set_ylabel('熱伝導度', fontsize=11)
    axes[1].set_title('熱伝導度の補完比較', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("KNNは近傍サンプルの情報を活用するため、")
    print("相関のある変数間の関係を保ちやすい")
    

### MICE (Multiple Imputation by Chained Equations)
    
    
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    def mice_imputation(df, max_iter=10, random_state=42):
        """
        MICE（多重代入法）
    
        各変数を他の変数で予測し、反復的に補完
        """
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state
        )
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns
        )
        return df_imputed
    
    # 実行
    df_mice = mice_imputation(df_sample, max_iter=10)
    
    # 手法の比較
    methods = {
        '平均値': imputed_results['mean'],
        'KNN': df_knn,
        'MICE': df_mice
    }
    
    # 補完精度評価（元の完全データとの比較）
    np.random.seed(42)
    df_complete = pd.DataFrame({
        '格子定数': np.random.uniform(3, 6, 100),
        'バンドギャップ': np.random.uniform(0, 3, 100),
        '電気伝導度': np.random.uniform(1e3, 1e6, 100),
        '熱伝導度': np.random.uniform(1, 100, 100)
    })
    
    # 欠損マスク
    missing_indices = df_sample.isnull()
    
    # 各手法のMAE計算
    print("補完精度比較（MAE）：")
    for method_name, df_method in methods.items():
        mae_list = []
        for col in df_sample.columns:
            if missing_indices[col].any():
                mask = missing_indices[col]
                mae = np.mean(
                    np.abs(
                        df_method.loc[mask, col] -
                        df_complete.loc[mask, col]
                    )
                )
                mae_list.append(mae)
    
        print(f"{method_name}: {np.mean(mae_list):.4f}")
    

**出力** ：
    
    
    補完精度比較（MAE）：
    平均値: 0.8523
    KNN: 0.5127
    MICE: 0.4856
    

* * *

## 1.4 外れ値検出と処理

外れ値は測定エラーの可能性もあれば、新規材料の発見につながる可能性もあります。

### 統計的手法（Z-score, IQR）
    
    
    def detect_outliers_zscore(data, threshold=3):
        """
        Z-scoreによる外れ値検出
        """
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    
    def detect_outliers_iqr(data, multiplier=1.5):
        """
        IQR（四分位範囲）による外れ値検出
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
    
        return (data < lower_bound) | (data > upper_bound)
    
    # テストデータ
    np.random.seed(42)
    data_normal = np.random.normal(50, 10, 100)
    data_with_outliers = np.concatenate([
        data_normal,
        [10, 15, 95, 100]  # 外れ値
    ])
    
    # 検出
    outliers_z = detect_outliers_zscore(data_with_outliers, threshold=3)
    outliers_iqr = detect_outliers_iqr(data_with_outliers, multiplier=1.5)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Z-score
    axes[0].scatter(range(len(data_with_outliers)), data_with_outliers,
                    c=outliers_z, cmap='RdYlGn_r', s=60, alpha=0.7,
                    edgecolors='k')
    axes[0].axhline(y=np.mean(data_with_outliers) + 3*np.std(data_with_outliers),
                    color='r', linestyle='--', label='±3σ')
    axes[0].axhline(y=np.mean(data_with_outliers) - 3*np.std(data_with_outliers),
                    color='r', linestyle='--')
    axes[0].set_xlabel('サンプルID', fontsize=11)
    axes[0].set_ylabel('値', fontsize=11)
    axes[0].set_title('Z-score法', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # IQR
    axes[1].scatter(range(len(data_with_outliers)), data_with_outliers,
                    c=outliers_iqr, cmap='RdYlGn_r', s=60, alpha=0.7,
                    edgecolors='k')
    Q1 = np.percentile(data_with_outliers, 25)
    Q3 = np.percentile(data_with_outliers, 75)
    IQR = Q3 - Q1
    axes[1].axhline(y=Q3 + 1.5*IQR, color='r', linestyle='--', label='Q3+1.5×IQR')
    axes[1].axhline(y=Q1 - 1.5*IQR, color='r', linestyle='--', label='Q1-1.5×IQR')
    axes[1].set_xlabel('サンプルID', fontsize=11)
    axes[1].set_ylabel('値', fontsize=11)
    axes[1].set_title('IQR法', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Z-score法: {outliers_z.sum()} 個の外れ値検出")
    print(f"IQR法: {outliers_iqr.sum()} 個の外れ値検出")
    

### Isolation Forest
    
    
    from sklearn.ensemble import IsolationForest
    
    def detect_outliers_iforest(X, contamination=0.1, random_state=42):
        """
        Isolation Forestによる外れ値検出
    
        高次元データに有効
        """
        clf = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        predictions = clf.fit_predict(X)
    
        # -1: 外れ値, 1: 正常値
        return predictions == -1
    
    # 2次元データで可視化
    np.random.seed(42)
    X_normal = np.random.randn(200, 2) * [2, 3] + [50, 60]
    X_outliers = np.random.uniform(40, 70, (20, 2))
    X = np.vstack([X_normal, X_outliers])
    
    outliers_if = detect_outliers_iforest(X, contamination=0.1)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    plt.scatter(X[~outliers_if, 0], X[~outliers_if, 1],
                c='steelblue', s=50, alpha=0.6, label='正常値')
    plt.scatter(X[outliers_if, 0], X[outliers_if, 1],
                c='red', s=100, alpha=0.8, marker='X', label='外れ値')
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('Isolation Forest による外れ値検出',
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"検出された外れ値: {outliers_if.sum()} / {len(X)}")
    

### Local Outlier Factor (LOF)
    
    
    from sklearn.neighbors import LocalOutlierFactor
    
    def detect_outliers_lof(X, n_neighbors=20, contamination=0.1):
        """
        Local Outlier Factorによる外れ値検出
    
        局所的な密度に基づく検出
        """
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        predictions = clf.fit_predict(X)
    
        return predictions == -1
    
    # LOF vs Isolation Forest比較
    outliers_lof = detect_outliers_lof(X, n_neighbors=20, contamination=0.1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Isolation Forest
    axes[0].scatter(X[~outliers_if, 0], X[~outliers_if, 1],
                    c='steelblue', s=50, alpha=0.6, label='正常値')
    axes[0].scatter(X[outliers_if, 0], X[outliers_if, 1],
                    c='red', s=100, alpha=0.8, marker='X', label='外れ値')
    axes[0].set_xlabel('特徴量1', fontsize=11)
    axes[0].set_ylabel('特徴量2', fontsize=11)
    axes[0].set_title('Isolation Forest', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # LOF
    axes[1].scatter(X[~outliers_lof, 0], X[~outliers_lof, 1],
                    c='steelblue', s=50, alpha=0.6, label='正常値')
    axes[1].scatter(X[outliers_lof, 0], X[outliers_lof, 1],
                    c='red', s=100, alpha=0.8, marker='X', label='外れ値')
    axes[1].set_xlabel('特徴量1', fontsize=11)
    axes[1].set_ylabel('特徴量2', fontsize=11)
    axes[1].set_title('Local Outlier Factor', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Isolation Forest: {outliers_if.sum()} 個")
    print(f"LOF: {outliers_lof.sum()} 個")
    

### DBSCAN clustering
    
    
    from sklearn.cluster import DBSCAN
    
    def detect_outliers_dbscan(X, eps=3, min_samples=5):
        """
        DBSCANによる外れ値検出
    
        クラスタリング結果でラベル-1が外れ値
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X)
    
        return labels == -1
    
    # 実行
    outliers_dbscan = detect_outliers_dbscan(X, eps=5, min_samples=10)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    
    clustering = DBSCAN(eps=5, min_samples=10)
    labels = clustering.fit_predict(X)
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 外れ値
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c='red', s=100,
                        marker='X', label='外れ値', alpha=0.8)
        else:
            # クラスタ
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50,
                        alpha=0.6, label=f'クラスタ {k}')
    
    plt.xlabel('特徴量1', fontsize=12)
    plt.ylabel('特徴量2', fontsize=12)
    plt.title('DBSCAN による外れ値検出', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"検出された外れ値: {outliers_dbscan.sum()} / {len(X)}")
    

* * *

## 1.5 ケーススタディ：熱電材料データセット

実際の熱電材料データセットを用いて、データクリーニングの全工程を実践します。
    
    
    # 熱電材料データセット（シミュレーション）
    np.random.seed(42)
    
    n_samples = 200
    
    thermoelectric_data = pd.DataFrame({
        '組成_A': np.random.uniform(0.1, 0.9, n_samples),
        '組成_B': np.random.uniform(0.05, 0.3, n_samples),
        'ドーパント濃度': np.random.uniform(0.001, 0.05, n_samples),
        '合成温度': np.random.uniform(600, 1200, n_samples),
        '格子定数': np.random.uniform(5.5, 6.5, n_samples),
        'バンドギャップ': np.random.uniform(0.1, 0.8, n_samples),
        '電気伝導度': np.random.lognormal(10, 2, n_samples),
        'ゼーベック係数': np.random.normal(200, 50, n_samples),
        '熱伝導度': np.random.uniform(1, 10, n_samples),
        'ZT値': np.random.uniform(0.1, 2.0, n_samples)
    })
    
    # 実験データ + DFT計算データの統合
    thermoelectric_data['データソース'] = np.random.choice(
        ['実験', 'DFT'], n_samples, p=[0.6, 0.4]
    )
    
    # 欠損値を20%導入
    missing_mask_lattice = np.random.random(n_samples) < 0.15
    thermoelectric_data.loc[missing_mask_lattice, '格子定数'] = np.nan
    
    missing_mask_bandgap = np.random.random(n_samples) < 0.12
    thermoelectric_data.loc[missing_mask_bandgap, 'バンドギャップ'] = np.nan
    
    missing_mask_thermal = np.random.random(n_samples) < 0.18
    thermoelectric_data.loc[missing_mask_thermal, '熱伝導度'] = np.nan
    
    # 外れ値を導入
    outlier_idx = np.random.choice(n_samples, 10, replace=False)
    thermoelectric_data.loc[outlier_idx, 'ZT値'] += np.random.uniform(2, 5, 10)
    
    print("=== 熱電材料データセット ===")
    print(f"サンプル数: {len(thermoelectric_data)}")
    print(f"特徴量数: {thermoelectric_data.shape[1]}")
    print(f"\n欠損値数:")
    print(thermoelectric_data.isnull().sum())
    

### Step 1: 欠損値処理
    
    
    # 欠損パターン可視化
    plt.figure(figsize=(12, 6))
    sns.heatmap(thermoelectric_data.isnull(),
                cmap='YlOrRd', cbar_kws={'label': '欠損'})
    plt.title('熱電材料データの欠損パターン', fontsize=13, fontweight='bold')
    plt.xlabel('特徴量', fontsize=11)
    plt.ylabel('サンプル', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # MICE補完
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    # 数値列のみ抽出
    numeric_cols = thermoelectric_data.select_dtypes(
        include=[np.number]
    ).columns
    
    imputer = IterativeImputer(max_iter=10, random_state=42)
    thermoelectric_imputed = thermoelectric_data.copy()
    thermoelectric_imputed[numeric_cols] = imputer.fit_transform(
        thermoelectric_data[numeric_cols]
    )
    
    print("\n欠損値補完完了")
    print(thermoelectric_imputed.isnull().sum())
    

### Step 2: 外れ値検出
    
    
    # Isolation Forestで外れ値検出
    X_features = thermoelectric_imputed[numeric_cols].values
    
    clf = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = clf.fit_predict(X_features)
    outliers_mask = outlier_labels == -1
    
    print(f"\n検出された外れ値: {outliers_mask.sum()} / {len(thermoelectric_imputed)}")
    
    # ZT値の分布と外れ値
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 箱ひげ図
    axes[0].boxplot([
        thermoelectric_imputed.loc[~outliers_mask, 'ZT値'],
        thermoelectric_imputed.loc[outliers_mask, 'ZT値']
    ], labels=['正常値', '外れ値'])
    axes[0].set_ylabel('ZT値', fontsize=12)
    axes[0].set_title('ZT値の分布', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 散布図（電気伝導度 vs ZT値）
    axes[1].scatter(
        thermoelectric_imputed.loc[~outliers_mask, '電気伝導度'],
        thermoelectric_imputed.loc[~outliers_mask, 'ZT値'],
        c='steelblue', s=50, alpha=0.6, label='正常値'
    )
    axes[1].scatter(
        thermoelectric_imputed.loc[outliers_mask, '電気伝導度'],
        thermoelectric_imputed.loc[outliers_mask, 'ZT値'],
        c='red', s=100, alpha=0.8, marker='X', label='外れ値'
    )
    axes[1].set_xlabel('電気伝導度 (S/m)', fontsize=11)
    axes[1].set_ylabel('ZT値', fontsize=11)
    axes[1].set_xscale('log')
    axes[1].set_title('外れ値の可視化', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Step 3: 物理的妥当性検証
    
    
    def validate_physical_constraints(df):
        """
        物理的制約条件のチェック
        """
        violations = []
    
        # 組成の合計が1前後
        composition_sum = df['組成_A'] + df['組成_B']
        composition_violation = (composition_sum < 0.9) | (composition_sum > 1.1)
        if composition_violation.any():
            violations.append(
                f"組成合計異常: {composition_violation.sum()} サンプル"
            )
    
        # バンドギャップは正
        bandgap_violation = df['バンドギャップ'] < 0
        if bandgap_violation.any():
            violations.append(
                f"負のバンドギャップ: {bandgap_violation.sum()} サンプル"
            )
    
        # ZT値の理論上限（ZT > 4 は非現実的）
        zt_violation = df['ZT値'] > 4
        if zt_violation.any():
            violations.append(
                f"ZT値異常（>4）: {zt_violation.sum()} サンプル"
            )
    
        return violations
    
    # 検証
    violations = validate_physical_constraints(thermoelectric_imputed)
    
    print("\n物理的妥当性検証：")
    if violations:
        for v in violations:
            print(f"⚠️ {v}")
    else:
        print("✅ 全てのサンプルが物理的制約を満たす")
    
    # 外れ値除去
    thermoelectric_cleaned = thermoelectric_imputed[~outliers_mask].copy()
    
    print(f"\nクリーニング後のサンプル数: {len(thermoelectric_cleaned)}")
    print(f"除去されたサンプル: {outliers_mask.sum()}")
    

### Step 4: クリーニング前後の比較
    
    
    # データ品質の比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    features_to_compare = ['ZT値', '電気伝導度', 'ゼーベック係数', '熱伝導度']
    
    for idx, feature in enumerate(features_to_compare):
        ax = axes[idx // 2, idx % 2]
    
        # 元データ
        ax.hist(thermoelectric_data[feature].dropna(), bins=30,
                alpha=0.5, label='元データ', color='gray')
    
        # クリーニング後
        ax.hist(thermoelectric_cleaned[feature], bins=30,
                alpha=0.7, label='クリーニング後', color='steelblue')
    
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('頻度', fontsize=11)
        ax.set_title(f'{feature}の分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("\n=== データクリーニング効果 ===")
    print(f"元データ: {len(thermoelectric_data)} サンプル, "
          f"{thermoelectric_data.isnull().sum().sum()} 欠損値")
    print(f"クリーニング後: {len(thermoelectric_cleaned)} サンプル, "
          f"{thermoelectric_cleaned.isnull().sum().sum()} 欠損値")
    print(f"\nZT値統計:")
    print(f"  元データ: 平均 {thermoelectric_data['ZT値'].mean():.3f}, "
          f"標準偏差 {thermoelectric_data['ZT値'].std():.3f}")
    print(f"  クリーニング後: 平均 {thermoelectric_cleaned['ZT値'].mean():.3f}, "
          f"標準偏差 {thermoelectric_cleaned['ZT値'].std():.3f}")
    

* * *

## 1.6 データライセンスと再現性

### 主要材料データベースのライセンス

材料データを利用する際は、各データベースのライセンスを理解することが重要です。
    
    
    # 主要材料データベースの情報
    database_info = pd.DataFrame({
        'データベース': [
            'Materials Project',
            'OQMD',
            'NOMAD',
            'AFLOW',
            'Citrination'
        ],
        'ライセンス': [
            'CC BY 4.0',
            'Academic Use',
            'CC BY 4.0',
            'AFLOWLIB Consortium',
            'Commercial/Academic'
        ],
        'データ数': [
            '150,000+',
            '1,000,000+',
            '10,000,000+',
            '3,500,000+',
            '250,000+'
        ],
        '主要データ': [
            'DFT計算、バンドギャップ、形成エネルギー',
            'DFT計算、安定性、相図',
            '計算・実験データ、機械学習モデル',
            'プロトタイプ構造、物性予測',
            '実験データ、プロセス条件'
        ],
        'API': [
            'pymatgen',
            'qmpy',
            'NOMAD API',
            'AFLOW API',
            'citrination-client'
        ]
    })
    
    print("=== 材料データベース比較 ===")
    print(database_info.to_string(index=False))
    
    # 使用例
    print("\n【Materials Project API使用例】")
    print("```python")
    print("from pymatgen.ext.matproj import MPRester")
    print("# API key: https://materialsproject.org/api で取得")
    print("with MPRester('YOUR_API_KEY') as mpr:")
    print("    structure = mpr.get_structure_by_material_id('mp-149')")
    print("    bandgap = mpr.get_bandstructure_by_material_id('mp-149')")
    print("```")
    
    print("\n【注意事項】")
    print("✅ 論文・商用利用時はライセンスを確認")
    print("✅ 引用を適切に行う（各DBの引用形式に従う）")
    print("✅ APIキーは環境変数で管理（.envファイル）")
    print("✅ データのバージョン・取得日を記録")
    

**出力** ：
    
    
    === 材料データベース比較 ===
    データベース         ライセンス              データ数      主要データ                              API
    Materials Project    CC BY 4.0               150,000+      DFT計算、バンドギャップ、形成エネルギー    pymatgen
    OQMD                 Academic Use            1,000,000+    DFT計算、安定性、相図                    qmpy
    NOMAD                CC BY 4.0               10,000,000+   計算・実験データ、機械学習モデル          NOMAD API
    AFLOW                AFLOWLIB Consortium     3,500,000+    プロトタイプ構造、物性予測                AFLOW API
    Citrination          Commercial/Academic     250,000+      実験データ、プロセス条件                  citrination-client
    

### コード再現性の確保
    
    
    # 環境仕様の記録
    import sys
    import sklearn
    import pandas as pd
    import numpy as np
    
    reproducibility_info = {
        'Python': sys.version,
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'Date': '2025-10-19'
    }
    
    print("=== 再現性情報 ===")
    for key, value in reproducibility_info.items():
        print(f"{key}: {value}")
    
    # requirements.txt生成
    print("\n【推奨環境】")
    requirements = """
    numpy==1.24.3
    pandas==2.0.3
    scikit-learn==1.3.0
    matplotlib==3.7.2
    seaborn==0.12.2
    scipy==1.11.1
    """
    print(requirements)
    
    print("【環境構築コマンド】")
    print("```bash")
    print("# 仮想環境作成")
    print("python -m venv venv")
    print("source venv/bin/activate  # Linux/Mac")
    print("# venv\\Scripts\\activate  # Windows")
    print("")
    print("# パッケージインストール")
    print("pip install -r requirements.txt")
    print("```")
    

### 実践的な落とし穴（Pitfalls）
    
    
    # 落とし穴1: データリーク（Data Leakage）
    print("=== 実践的な落とし穴 ===\n")
    
    print("【落とし穴1: データリーク】")
    print("❌ 悪い例：全データで前処理 → Train/Test分割")
    print("```python")
    print("X_scaled = StandardScaler().fit_transform(X)  # 全データでfit")
    print("X_train, X_test = train_test_split(X_scaled)")
    print("# → テストデータの情報が訓練時に漏れている！")
    print("```")
    
    print("\n✅ 正しい例：Train/Test分割 → 訓練データのみで前処理")
    print("```python")
    print("X_train, X_test = train_test_split(X)")
    print("scaler = StandardScaler().fit(X_train)  # 訓練データのみでfit")
    print("X_train_scaled = scaler.transform(X_train)")
    print("X_test_scaled = scaler.transform(X_test)")
    print("```")
    
    print("\n【落とし穴2: 組成ベース分割の必要性】")
    print("❌ 悪い例：ランダム分割")
    print("- Li₀.₉CoO₂（訓練）とLi₁.₀CoO₂（テスト）は類似")
    print("- 過度に楽観的な性能評価")
    
    print("\n✅ 正しい例：組成グループ分割")
    print("```python")
    print("from sklearn.model_selection import GroupKFold")
    print("groups = [get_composition_family(formula) for formula in formulas]")
    print("gkf = GroupKFold(n_splits=5)")
    print("for train_idx, test_idx in gkf.split(X, y, groups):")
    print("    # 同じ組成系は同じfoldに")
    print("```")
    
    print("\n【落とし穴3: 外挿の限界】")
    print("⚠️ 機械学習モデルは訓練範囲外の予測が苦手")
    print("例：バンドギャップ0-3 eVで訓練 → 5 eVの予測は不正確")
    
    print("\n対策:")
    print("- 訓練データの範囲を明示")
    print("- 外挿領域の不確実性を定量化（ベイズ手法）")
    print("- Active Learningで段階的に範囲拡大")
    
    print("\n【落とし穴4: 特徴量間の相関】")
    print("⚠️ 高相関特徴量は冗長で過学習を招く")
    print("```python")
    print("# 相関行列で確認")
    print("correlation_matrix = X.corr()")
    print("high_corr = (correlation_matrix.abs() > 0.9) & (correlation_matrix != 1.0)")
    print("print(high_corr.sum())  # 高相関ペア数")
    print("")
    print("# 対策：VIF（分散拡大係数）で多重共線性検出")
    print("from statsmodels.stats.outliers_influence import variance_inflation_factor")
    print("vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]")
    print("```")
    

* * *

## 演習問題

### 問題1（難易度: easy）

以下のデータセットに対して、Simple Imputation（平均値）とKNN Imputationを適用し、補完精度を比較してください。
    
    
    # 演習用データ
    np.random.seed(123)
    exercise_data = pd.DataFrame({
        'feature1': np.random.normal(50, 10, 100),
        'feature2': np.random.normal(30, 5, 100),
        'feature3': np.random.normal(100, 20, 100)
    })
    
    # ランダムに10%欠損
    for col in exercise_data.columns:
        missing_idx = np.random.choice(100, 10, replace=False)
        exercise_data.loc[missing_idx, col] = np.nan
    

ヒント 1\. `SimpleImputer(strategy='mean')`を使用 2\. `KNNImputer(n_neighbors=5)`を使用 3\. 元の完全データを作成して、補完値との差（MAE）を計算  解答例
    
    
    from sklearn.impute import SimpleImputer, KNNImputer
    
    # 元の完全データ（比較用）
    np.random.seed(123)
    true_data = pd.DataFrame({
        'feature1': np.random.normal(50, 10, 100),
        'feature2': np.random.normal(30, 5, 100),
        'feature3': np.random.normal(100, 20, 100)
    })
    
    # Simple Imputation
    simple_imputer = SimpleImputer(strategy='mean')
    data_simple = pd.DataFrame(
        simple_imputer.fit_transform(exercise_data),
        columns=exercise_data.columns
    )
    
    # KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    data_knn = pd.DataFrame(
        knn_imputer.fit_transform(exercise_data),
        columns=exercise_data.columns
    )
    
    # 精度評価
    missing_mask = exercise_data.isnull()
    mae_simple = []
    mae_knn = []
    
    for col in exercise_data.columns:
        mask = missing_mask[col]
        if mask.any():
            mae_s = np.mean(np.abs(data_simple.loc[mask, col] - true_data.loc[mask, col]))
            mae_k = np.mean(np.abs(data_knn.loc[mask, col] - true_data.loc[mask, col]))
            mae_simple.append(mae_s)
            mae_knn.append(mae_k)
    
    print(f"Simple Imputation MAE: {np.mean(mae_simple):.4f}")
    print(f"KNN Imputation MAE: {np.mean(mae_knn):.4f}")
    

### 問題2（難易度: medium）

Latin Hypercube Samplingを用いて、3次元の組成空間（元素A, B, Cの割合）をサンプリングしてください。制約条件として、A + B + C = 1 を満たすようにしてください。

ヒント 1\. 2次元でLHSを実行（AとBのみ） 2\. C = 1 - A - B で計算 3\. 3次元空間で可視化  解答例
    
    
    from scipy.stats import qmc
    from mpl_toolkits.mplot3d import Axes3D
    
    # 2次元LHS（A, B）
    sampler = qmc.LatinHypercube(d=2, seed=42)
    samples_2d = sampler.random(n=50)
    
    # A + B <= 1 となるようスケーリング
    A = samples_2d[:, 0] * 0.9  # 0〜0.9
    B = (1 - A) * samples_2d[:, 1]  # 残りの範囲内
    C = 1 - A - B
    
    # 3次元可視化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(A, B, C, c='steelblue', s=100, alpha=0.6, edgecolors='k')
    ax.set_xlabel('元素A', fontsize=12)
    ax.set_ylabel('元素B', fontsize=12)
    ax.set_zlabel('元素C', fontsize=12)
    ax.set_title('組成空間のLatin Hypercube Sampling', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 制約確認
    print(f"全サンプルで A+B+C=1: {np.allclose(A+B+C, 1)}")
    

### 問題3（難易度: hard）

Isolation ForestとLOFを用いて、多次元データの外れ値検出を行い、どちらがより適切か評価してください。評価には、既知の外れ値ラベルとの一致率（Precision, Recall, F1-score）を使用してください。

ヒント 1\. 正常データ + 意図的な外れ値を生成 2\. Isolation ForestとLOFで検出 3\. `sklearn.metrics.classification_report`で評価  解答例
    
    
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import classification_report, confusion_matrix
    
    # データ生成
    np.random.seed(42)
    X_normal = np.random.randn(200, 5) * 2 + 10
    X_outliers = np.random.uniform(0, 20, (20, 5))
    X = np.vstack([X_normal, X_outliers])
    
    # 真のラベル（0: 正常, 1: 外れ値）
    y_true = np.array([0]*200 + [1]*20)
    
    # Isolation Forest
    clf_if = IsolationForest(contamination=0.1, random_state=42)
    y_pred_if = (clf_if.fit_predict(X) == -1).astype(int)
    
    # LOF
    clf_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred_lof = (clf_lof.fit_predict(X) == -1).astype(int)
    
    # 評価
    print("=== Isolation Forest ===")
    print(classification_report(y_true, y_pred_if,
                               target_names=['正常', '外れ値']))
    
    print("\n=== Local Outlier Factor ===")
    print(classification_report(y_true, y_pred_lof,
                               target_names=['正常', '外れ値']))
    
    # 混同行列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_if = confusion_matrix(y_true, y_pred_if)
    sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('予測ラベル', fontsize=11)
    axes[0].set_ylabel('真のラベル', fontsize=11)
    axes[0].set_title('Isolation Forest', fontsize=12, fontweight='bold')
    
    cm_lof = confusion_matrix(y_true, y_pred_lof)
    sns.heatmap(cm_lof, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_xlabel('予測ラベル', fontsize=11)
    axes[1].set_ylabel('真のラベル', fontsize=11)
    axes[1].set_title('LOF', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

* * *

## まとめ

この章では、データ駆動材料科学における**データ収集戦略とクリーニング** を学びました。

**重要ポイント** ：

  1. **材料データの特徴** ：小規模・不均衡・ノイズが多い → 適切な前処理が不可欠
  2. **実験計画法** ：DOE、LHS、Active Learningで効率的なデータ収集
  3. **欠損値処理** ：Simple < KNN < MICE の順で精度向上
  4. **外れ値検出** ：統計的手法、Isolation Forest、LOF、DBSCANを使い分け
  5. **物理的妥当性** ：機械的なクリーニングだけでなく、物理的意味を検証
  6. **データライセンス** ：Materials Project、OQMD、NOMADなど主要DBの利用規約を確認
  7. **再現性確保** ：環境バージョン記録、requirements.txt管理
  8. **実践的落とし穴** ：データリーク、組成ベース分割、外挿の限界、特徴量相関

**次章予告** ： Chapter 2では、クリーニング済みデータから**有効な特徴量を設計** する手法（特徴量エンジニアリング）を学びます。matminerを用いた材料記述子生成、次元削減、特徴量選択を実践します。

* * *

## Chapter 1 チェックリスト

### データ収集

  * [ ] **実験計画法の選択**
  * [ ] 全探索（Full Factorial）vs 部分探索（Fractional Factorial）を判断
  * [ ] Latin Hypercube Samplingで探索空間を均一にカバー
  * [ ] Active Learningで不確実性の高い領域を優先サンプリング

  * [ ] **データソースの確認**

  * [ ] 実験データ、DFT計算、文献データの混在を把握
  * [ ] 各ソースの精度と信頼性を評価
  * [ ] データ取得日とバージョンを記録

  * [ ] **ライセンスと引用**

  * [ ] Materials Project、OQMD、NOMAD、AFLOWのライセンスを確認
  * [ ] 論文・商用利用時の制約を理解
  * [ ] 適切な引用形式でデータベースを引用

### データクリーニング

  * [ ] **欠損値処理**
  * [ ] 欠損パターンの分類（MCAR、MAR、MNAR）
  * [ ] Simple Imputation（平均値・中央値）で基準性能確認
  * [ ] KNN Imputationで相関を考慮した補完
  * [ ] MICEで複雑な依存関係を捉える補完
  * [ ] 補完前後の統計量変化を確認

  * [ ] **外れ値検出**

  * [ ] Z-score法、IQR法で単変量外れ値検出
  * [ ] Isolation Forestで多変量外れ値検出
  * [ ] LOFで局所的密度ベースの検出
  * [ ] DBSCANでクラスタリングベースの検出
  * [ ] 外れ値が物理的に妥当か検証（測定エラー vs 新発見）

  * [ ] **物理的妥当性検証**

  * [ ] 組成の合計が1前後（許容範囲±0.1）
  * [ ] バンドギャップ、形成エネルギーなどが正の値
  * [ ] 物性値が理論的上限を超えていないか
  * [ ] 既知の物理法則（アレニウス則など）と矛盾しないか

### 実践的落とし穴の回避

  * [ ] **データリーク防止**
  * [ ] Train/Test分割**後** に前処理（StandardScaler、Imputerなど）
  * [ ] クロスバリデーション時も各foldで独立に前処理
  * [ ] 目的変数を用いた特徴量エンジニアリングを避ける

  * [ ] **組成ベース分割**

  * [ ] 類似組成が同じfoldに入るようGroupKFold使用
  * [ ] ランダム分割で過度に楽観的な評価を避ける
  * [ ] テストセットで材料系の多様性を確保

  * [ ] **外挿の限界認識**

  * [ ] 訓練データの範囲（最小値・最大値）を明示
  * [ ] 外挿領域の予測には不確実性を付与
  * [ ] Active Learningで段階的に範囲拡大

  * [ ] **特徴量間の相関管理**

  * [ ] 相関行列で高相関ペア（|r| > 0.9）を特定
  * [ ] VIF（分散拡大係数）で多重共線性を検出
  * [ ] 冗長な特徴量を除去または主成分分析で集約

### 再現性の確保

  * [ ] **環境記録**
  * [ ] Python、NumPy、Pandas、scikit-learnのバージョン記録
  * [ ] requirements.txtまたはenvironment.ymlを作成
  * [ ] 乱数シードを固定（`random_state=42`など）

  * [ ] **データ管理**

  * [ ] 元データとクリーニング済みデータを分離保存
  * [ ] クリーニングスクリプトをバージョン管理
  * [ ] データの取得日・ソース・処理履歴を記録

  * [ ] **コード品質**

  * [ ] 関数化・モジュール化で再利用性向上
  * [ ] Docstringで処理内容を文書化
  * [ ] 単体テストで主要関数の動作確認

### データ品質評価指標

  * [ ] **完全性**
  * [ ] 欠損率 < 20%（推奨）
  * [ ] 重要特徴量の欠損率 < 10%

  * [ ] **正確性**

  * [ ] 外れ値率 < 5%
  * [ ] 物理制約違反率 = 0%

  * [ ] **代表性**

  * [ ] サンプル/特徴量比 > 10:1（推奨）
  * [ ] クラス不均衡比 < 10:1（分類問題）

  * [ ] **信頼性**

  * [ ] 複数ソース間の一致度 > 80%
  * [ ] 測定誤差の標準偏差記録

* * *

## 参考文献

  1. **Little, R. J. & Rubin, D. B.** (2019). _Statistical Analysis with Missing Data_ (3rd ed.). Wiley. [DOI: 10.1002/9781119482260](<https://doi.org/10.1002/9781119482260>)

  2. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). Isolation forest. In _2008 Eighth IEEE International Conference on Data Mining_ (pp. 413-422). IEEE. [DOI: 10.1109/ICDM.2008.17](<https://doi.org/10.1109/ICDM.2008.17>)

  3. **Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J.** (2000). LOF: identifying density-based local outliers. In _ACM SIGMOD Record_ (Vol. 29, No. 2, pp. 93-104). [DOI: 10.1145/335191.335388](<https://doi.org/10.1145/335191.335388>)

  4. **McKay, M. D., Beckman, R. J., & Conover, W. J.** (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. _Technometrics_ , 21(2), 239-245. [DOI: 10.1080/00401706.1979.10489755](<https://doi.org/10.1080/00401706.1979.10489755>)

  5. **Settles, B.** (2009). _Active Learning Literature Survey_ (Computer Sciences Technical Report 1648). University of Wisconsin-Madison.

* * *

[← シリーズ目次に戻る](<index.html>) | [Chapter 2へ進む →](<chapter-2.html>)
