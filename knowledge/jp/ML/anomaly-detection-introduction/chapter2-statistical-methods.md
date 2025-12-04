---
title: 第2章：統計的異常検知
chapter_title: 第2章：統計的異常検知
subtitle: 統計的手法による異常検知の基礎と応用
reading_time: 30-35分
difficulty: 初級〜中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Z-scoreとIQRによる外れ値検出を実装できる
  * ✅ Mahalanobis距離と多変量ガウス分布を理解する
  * ✅ 統計的仮説検定（Grubbs, ESD）を適用できる
  * ✅ 時系列データの異常検知手法を使える
  * ✅ 統計的手法の完全なパイプラインを構築できる

* * *

## 2.1 統計的外れ値検出

### Z-score（標準化スコア）

**Z-score** は、データポイントが平均からどれだけ標準偏差離れているかを示す指標です。

> Z-score = $\frac{x - \mu}{\sigma}$
> 
> 一般的な閾値：$|Z| > 3$ を異常とする

#### Z-scoreの特徴

  * **利点** : シンプルで解釈しやすい、計算が高速
  * **欠点** : 正規分布を仮定、外れ値の影響を受けやすい
  * **適用場面** : 単変量データ、正規分布に近いデータ

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # データ生成
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=300)
    outliers = np.array([5, -5, 6, -6, 7])
    data = np.concatenate([normal_data, outliers])
    
    # Z-scoreの計算
    z_scores = np.abs(stats.zscore(data))
    threshold = 3
    anomalies = z_scores > threshold
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # データ分布
    axes[0].hist(data, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(x=data.mean() + 3*data.std(), color='red',
                    linestyle='--', linewidth=2, label='±3σ')
    axes[0].axvline(x=data.mean() - 3*data.std(), color='red',
                    linestyle='--', linewidth=2)
    axes[0].set_xlabel('値', fontsize=12)
    axes[0].set_ylabel('頻度', fontsize=12)
    axes[0].set_title('データ分布とZ-score閾値', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Z-scoreプロット
    axes[1].scatter(range(len(data)), z_scores, alpha=0.6, s=30, c='blue')
    axes[1].scatter(np.where(anomalies)[0], z_scores[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='閾値=3')
    axes[1].set_xlabel('サンプル番号', fontsize=12)
    axes[1].set_ylabel('|Z-score|', fontsize=12)
    axes[1].set_title('Z-scoreによる異常検知', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Z-score異常検知結果 ===")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常インデックス: {np.where(anomalies)[0]}")
    print(f"異常値: {data[anomalies]}")
    

### IQR（四分位範囲）

**IQR法** は、外れ値に対してロバストな検出手法です。

> IQR = Q3 - Q1  
>  異常判定：$x < Q1 - 1.5 \times IQR$ または $x > Q3 + 1.5 \times IQR$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=300)
    outliers = np.array([100, 5, 110, 0])
    data = np.concatenate([normal_data, outliers])
    
    # IQRの計算
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 異常検出
    anomalies = (data < lower_bound) | (data > upper_bound)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ボックスプロット
    bp = axes[0].boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    axes[0].scatter([1]*anomalies.sum(), data[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black')
    axes[0].set_ylabel('値', fontsize=12)
    axes[0].set_title('箱ひげ図とIQR異常検知', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 散布図
    axes[1].scatter(range(len(data)), data, alpha=0.6, s=30, c='blue', label='正常')
    axes[1].scatter(np.where(anomalies)[0], data[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black')
    axes[1].axhline(y=upper_bound, color='red', linestyle='--', linewidth=2, label='IQR境界')
    axes[1].axhline(y=lower_bound, color='red', linestyle='--', linewidth=2)
    axes[1].axhline(y=Q1, color='green', linestyle=':', linewidth=1.5, label='Q1/Q3')
    axes[1].axhline(y=Q3, color='green', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('サンプル番号', fontsize=12)
    axes[1].set_ylabel('値', fontsize=12)
    axes[1].set_title('IQR法による異常検知', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== IQR異常検知結果 ===")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"下限: {lower_bound:.2f}, 上限: {upper_bound:.2f}")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常値: {data[anomalies]}")
    

> **重要** : IQRはZ-scoreと異なり、正規分布を仮定せず、外れ値の影響を受けにくいロバストな手法です。

* * *

## 2.2 確率分布ベースの異常検知

### Mahalanobis距離

**Mahalanobis距離** は、多変量データにおいて、共分散を考慮した距離指標です。

> $D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$
> 
> ここで、$\mu$は平均ベクトル、$\Sigma$は共分散行列

#### 特徴

  * **利点** : 変数間の相関を考慮、スケール不変
  * **欠点** : 共分散行列の逆行列計算が必要、計算コストが高い
  * **適用場面** : 多変量データ、相関のある特徴量

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2
    
    # 相関のある2変量データ生成
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # 相関係数0.8
    normal_data = np.random.multivariate_normal(mean, cov, size=300)
    
    # 異常データの追加
    outliers = np.array([[4, 4], [-4, -4], [4, -4]])
    data = np.vstack([normal_data, outliers])
    
    # Mahalanobis距離の計算
    mean_vec = normal_data.mean(axis=0)
    cov_matrix = np.cov(normal_data.T)
    cov_inv = np.linalg.inv(cov_matrix)
    
    mahal_distances = np.array([mahalanobis(x, mean_vec, cov_inv) for x in data])
    
    # 閾値（自由度2のカイ二乗分布の99%点）
    threshold = np.sqrt(chi2.ppf(0.99, df=2))
    anomalies = mahal_distances > threshold
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # データ分布
    axes[0].scatter(normal_data[:, 0], normal_data[:, 1],
                    alpha=0.6, s=50, c='blue', label='正常', edgecolors='black')
    axes[0].scatter(outliers[:, 0], outliers[:, 1],
                    c='red', s=150, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    
    # 信頼楕円（99%）
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * threshold * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean_vec, width, height, angle=angle,
                      edgecolor='red', facecolor='none', linewidth=2, linestyle='--', label='99%信頼楕円')
    axes[0].add_patch(ellipse)
    
    axes[0].set_xlabel('特徴量 1', fontsize=12)
    axes[0].set_ylabel('特徴量 2', fontsize=12)
    axes[0].set_title('Mahalanobis距離による異常検知', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Mahalanobis距離分布
    axes[1].hist(mahal_distances, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'閾値={threshold:.2f}')
    axes[1].set_xlabel('Mahalanobis距離', fontsize=12)
    axes[1].set_ylabel('頻度', fontsize=12)
    axes[1].set_title('Mahalanobis距離の分布', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Mahalanobis距離異常検知結果 ===")
    print(f"閾値: {threshold:.3f}")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常のMahalanobis距離: {mahal_distances[anomalies]}")
    

### 多変量ガウス分布

**多変量ガウス分布** を用いた異常検知は、確率密度に基づいて異常を判定します。

> $p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$
> 
> 異常判定：$p(x) < \epsilon$ （閾値）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    
    # データ生成
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    normal_data = np.random.multivariate_normal(mean, cov, size=300)
    outliers = np.array([[5, 5], [-5, -5], [5, -5]])
    data = np.vstack([normal_data, outliers])
    
    # 多変量ガウス分布のパラメータ推定
    mean_vec = normal_data.mean(axis=0)
    cov_matrix = np.cov(normal_data.T)
    mvn = multivariate_normal(mean=mean_vec, cov=cov_matrix)
    
    # 確率密度の計算
    densities = mvn.pdf(data)
    
    # 閾値（1%点）
    threshold = np.percentile(densities, 1)
    anomalies = densities < threshold
    
    # グリッド上で確率密度を計算（ヒートマップ用）
    x_range = np.linspace(-6, 6, 200)
    y_range = np.linspace(-6, 6, 200)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.dstack((xx, yy))
    Z = mvn.pdf(positions)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 確率密度ヒートマップ
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.7)
    axes[0].scatter(normal_data[:, 0], normal_data[:, 1],
                    alpha=0.6, s=30, c='blue', label='正常', edgecolors='black')
    axes[0].scatter(data[anomalies, 0], data[anomalies, 1],
                    c='red', s=150, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    plt.colorbar(contour, ax=axes[0], label='確率密度')
    axes[0].set_xlabel('特徴量 1', fontsize=12)
    axes[0].set_ylabel('特徴量 2', fontsize=12)
    axes[0].set_title('多変量ガウス分布による異常検知', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 確率密度のヒストグラム
    axes[1].hist(np.log(densities), bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(x=np.log(threshold), color='red', linestyle='--', linewidth=2, label='閾値')
    axes[1].set_xlabel('log(確率密度)', fontsize=12)
    axes[1].set_ylabel('頻度', fontsize=12)
    axes[1].set_title('確率密度の分布', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 多変量ガウス分布異常検知結果 ===")
    print(f"閾値: {threshold:.6f}")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常の確率密度: {densities[anomalies]}")
    

> **重要** : Mahalanobis距離と多変量ガウス分布は数学的に等価です。Mahalanobis距離の二乗は、対数確率密度に比例します。

* * *

## 2.3 統計的仮説検定

### Grubbs' Test（グラブス検定）

**Grubbs' Test** は、単一の外れ値を検出する仮説検定です。

> 帰無仮説 $H_0$: 外れ値は存在しない  
>  検定統計量: $G = \frac{\max|x_i - \bar{x}|}{s}$  
>  ここで、$s$は標準偏差
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def grubbs_test(data, alpha=0.05):
        """Grubbs' Testによる外れ値検出"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
    
        # 検定統計量の計算
        deviations = np.abs(data - mean)
        max_idx = np.argmax(deviations)
        G = deviations[max_idx] / std
    
        # 臨界値の計算
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    
        is_outlier = G > G_critical
    
        return {
            'G': G,
            'G_critical': G_critical,
            'is_outlier': is_outlier,
            'outlier_idx': max_idx if is_outlier else None,
            'outlier_value': data[max_idx] if is_outlier else None,
            'p_value': 1 - stats.t.cdf(G * np.sqrt(n) / np.sqrt(n - 1), n - 2)
        }
    
    # データ生成
    np.random.seed(42)
    data = np.concatenate([np.random.normal(50, 5, size=30), [80]])  # 80が外れ値
    
    # Grubbs' Test実行
    result = grubbs_test(data, alpha=0.05)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # データプロット
    axes[0].scatter(range(len(data)), data, alpha=0.6, s=50, c='blue', label='データ')
    if result['is_outlier']:
        axes[0].scatter(result['outlier_idx'], result['outlier_value'],
                        c='red', s=200, marker='X', label='外れ値', zorder=5, edgecolors='black', linewidths=2)
    axes[0].axhline(y=data.mean(), color='green', linestyle='--', linewidth=2, label='平均')
    axes[0].axhline(y=data.mean() + 3*data.std(), color='orange', linestyle=':', linewidth=1.5, label='±3σ')
    axes[0].axhline(y=data.mean() - 3*data.std(), color='orange', linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('サンプル番号', fontsize=12)
    axes[0].set_ylabel('値', fontsize=12)
    axes[0].set_title("Grubbs' Test結果", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 検定統計量の比較
    axes[1].bar(['G統計量', '臨界値'], [result['G'], result['G_critical']],
                color=['steelblue', 'red'], edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('値', fontsize=12)
    axes[1].set_title('検定統計量 vs 臨界値', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Grubbs' Test結果 ===")
    print(f"G統計量: {result['G']:.3f}")
    print(f"臨界値: {result['G_critical']:.3f}")
    print(f"p値: {result['p_value']:.4f}")
    print(f"外れ値検出: {'Yes' if result['is_outlier'] else 'No'}")
    if result['is_outlier']:
        print(f"外れ値: index={result['outlier_idx']}, value={result['outlier_value']:.2f}")
    

### ESD Test（極端なスチューデント化偏差検定）

**Generalized ESD Test** は、複数の外れ値を検出できる拡張版です。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def generalized_esd_test(data, max_outliers, alpha=0.05):
        """Generalized ESD Testによる複数外れ値検出"""
        n = len(data)
        outliers = []
        data_copy = data.copy()
    
        for i in range(max_outliers):
            mean = np.mean(data_copy)
            std = np.std(data_copy, ddof=1)
    
            # 検定統計量の計算
            deviations = np.abs(data_copy - mean)
            max_idx = np.argmax(deviations)
            R = deviations[max_idx] / std
    
            # 臨界値の計算
            n_current = len(data_copy)
            p = 1 - alpha / (2 * (n_current - i))
            t_dist = stats.t.ppf(p, n_current - i - 2)
            lambda_critical = ((n_current - i - 1) * t_dist) / np.sqrt((n_current - i - 2 + t_dist**2) * (n_current - i))
    
            if R > lambda_critical:
                outlier_idx = np.where(data == data_copy[max_idx])[0][0]
                outliers.append({'index': outlier_idx, 'value': data_copy[max_idx], 'R': R})
                data_copy = np.delete(data_copy, max_idx)
            else:
                break
    
        return outliers
    
    # データ生成
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, size=30)
    outlier_values = [80, 85, 15]
    data = np.concatenate([normal_data, outlier_values])
    
    # ESD Test実行
    outliers = generalized_esd_test(data, max_outliers=5, alpha=0.05)
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(data)), data, alpha=0.6, s=50, c='blue', label='正常データ')
    
    if outliers:
        outlier_indices = [o['index'] for o in outliers]
        outlier_values_detected = [o['value'] for o in outliers]
        plt.scatter(outlier_indices, outlier_values_detected,
                    c='red', s=200, marker='X', label='外れ値', zorder=5, edgecolors='black', linewidths=2)
    
    plt.axhline(y=data.mean(), color='green', linestyle='--', linewidth=2, label='平均')
    plt.axhline(y=data.mean() + 3*data.std(), color='orange', linestyle=':', linewidth=1.5, label='±3σ')
    plt.axhline(y=data.mean() - 3*data.std(), color='orange', linestyle=':', linewidth=1.5)
    plt.xlabel('サンプル番号', fontsize=12)
    plt.ylabel('値', fontsize=12)
    plt.title('Generalized ESD Test結果', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Generalized ESD Test結果 ===")
    print(f"検出された外れ値数: {len(outliers)}個")
    for i, o in enumerate(outliers, 1):
        print(f"外れ値{i}: index={o['index']}, value={o['value']:.2f}, R={o['R']:.3f}")
    

> **重要** : Grubbs' Testは1つの外れ値のみ検出可能ですが、ESD Testは複数の外れ値を逐次的に検出できます。

* * *

## 2.4 時系列異常検知

### 移動平均による異常検知

**移動平均** は、時系列データの傾向を捉え、逸脱を検出します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 時系列データ生成
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    trend = 0.05 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 2, n_samples)
    data = trend + seasonal + noise
    
    # 異常を追加
    anomaly_indices = [50, 150, 250]
    data[anomaly_indices] += [20, -25, 30]
    
    # 移動平均の計算
    window_size = 20
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    moving_std = np.array([data[max(0, i-window_size//2):min(len(data), i+window_size//2)].std()
                           for i in range(len(data))])
    
    # 異常検出（3σルール）
    residuals = np.abs(data - moving_avg)
    threshold = 3 * moving_std
    anomalies = residuals > threshold
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 時系列データ
    axes[0].plot(time, data, alpha=0.7, linewidth=1, label='元データ', color='blue')
    axes[0].plot(time, moving_avg, linewidth=2, label=f'移動平均(window={window_size})', color='green')
    axes[0].fill_between(time, moving_avg - 3*moving_std, moving_avg + 3*moving_std,
                         alpha=0.2, color='green', label='±3σ')
    axes[0].scatter(time[anomalies], data[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    axes[0].set_xlabel('時刻', fontsize=12)
    axes[0].set_ylabel('値', fontsize=12)
    axes[0].set_title('移動平均による時系列異常検知', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 残差プロット
    axes[1].plot(time, residuals, alpha=0.7, linewidth=1, color='blue')
    axes[1].plot(time, threshold, linestyle='--', linewidth=2, color='red', label='閾値(3σ)')
    axes[1].scatter(time[anomalies], residuals[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    axes[1].set_xlabel('時刻', fontsize=12)
    axes[1].set_ylabel('残差', fontsize=12)
    axes[1].set_title('残差と異常閾値', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 移動平均異常検知結果 ===")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常インデックス: {np.where(anomalies)[0]}")
    

### 季節性分解による異常検知

**STL分解** （Seasonal and Trend decomposition using Loess）により、季節性・トレンド・残差に分解します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 時系列データ生成（明確な季節性）
    np.random.seed(42)
    n_samples = 200
    time = np.arange(n_samples)
    trend = 0.1 * time
    seasonal = 15 * np.sin(2 * np.pi * time / 30)  # 周期30
    noise = np.random.normal(0, 2, n_samples)
    data = trend + seasonal + noise
    
    # 異常を追加
    anomaly_indices = [50, 120, 180]
    data[anomaly_indices] += [30, -30, 25]
    
    # 季節性分解
    result = seasonal_decompose(data, model='additive', period=30, extrapolate_trend='freq')
    
    # 残差から異常検出
    residual = result.resid
    threshold = 3 * np.nanstd(residual)
    anomalies = np.abs(residual) > threshold
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 元データ
    axes[0].plot(time, data, linewidth=1, color='blue')
    axes[0].scatter(time[anomalies], data[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    axes[0].set_ylabel('元データ', fontsize=11)
    axes[0].set_title('季節性分解による時系列異常検知', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # トレンド
    axes[1].plot(time, result.trend, linewidth=2, color='green')
    axes[1].set_ylabel('トレンド', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 季節性
    axes[2].plot(time, result.seasonal, linewidth=2, color='orange')
    axes[2].set_ylabel('季節性', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # 残差
    axes[3].plot(time, residual, linewidth=1, color='blue')
    axes[3].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='閾値(±3σ)')
    axes[3].axhline(y=-threshold, color='red', linestyle='--', linewidth=2)
    axes[3].scatter(time[anomalies], residual[anomalies],
                    c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    axes[3].set_xlabel('時刻', fontsize=12)
    axes[3].set_ylabel('残差', fontsize=11)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 季節性分解異常検知結果 ===")
    print(f"周期: 30")
    print(f"異常検出数: {anomalies.sum()}個")
    print(f"異常インデックス: {np.where(anomalies)[0]}")
    

> **重要** : 季節性分解は、トレンドと季節性を除去することで、真の異常を残差として明確に検出できます。

* * *

## 2.5 実装と応用

### 統計的異常検知の完全パイプライン

実務で使える統計的異常検知システムを構築します。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    class StatisticalAnomalyDetector:
        """統計的異常検知の統合クラス"""
    
        def __init__(self, method='zscore', threshold=3.0, window_size=20):
            """
            Parameters:
            -----------
            method : str
                検出手法 ('zscore', 'iqr', 'mahalanobis', 'moving_avg', 'seasonal')
            threshold : float
                異常判定閾値
            window_size : int
                移動平均のウィンドウサイズ
            """
            self.method = method
            self.threshold = threshold
            self.window_size = window_size
            self.fitted = False
    
        def fit(self, X):
            """訓練データで統計量を学習"""
            if self.method == 'zscore':
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
            elif self.method == 'iqr':
                self.q1_ = np.percentile(X, 25, axis=0)
                self.q3_ = np.percentile(X, 75, axis=0)
                self.iqr_ = self.q3_ - self.q1_
            elif self.method == 'mahalanobis':
                self.mean_ = np.mean(X, axis=0)
                self.cov_ = np.cov(X.T)
                self.cov_inv_ = np.linalg.inv(self.cov_)
    
            self.fitted = True
            return self
    
        def predict(self, X):
            """異常スコアの計算と異常判定"""
            if not self.fitted and self.method not in ['moving_avg', 'seasonal']:
                raise ValueError("モデルが未学習です。fit()を先に実行してください。")
    
            if self.method == 'zscore':
                scores = np.abs((X - self.mean_) / self.std_)
                anomalies = np.any(scores > self.threshold, axis=1)
    
            elif self.method == 'iqr':
                lower = self.q1_ - 1.5 * self.iqr_
                upper = self.q3_ + 1.5 * self.iqr_
                anomalies = np.any((X < lower) | (X > upper), axis=1)
                scores = np.max(np.abs(X - self.mean_) / self.std_, axis=1)
    
            elif self.method == 'mahalanobis':
                from scipy.spatial.distance import mahalanobis
                scores = np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])
                anomalies = scores > self.threshold
    
            elif self.method == 'moving_avg':
                # 1次元時系列のみ対応
                moving_avg = np.convolve(X.flatten(), np.ones(self.window_size)/self.window_size, mode='same')
                moving_std = np.array([X.flatten()[max(0, i-self.window_size//2):min(len(X), i+self.window_size//2)].std()
                                       for i in range(len(X))])
                scores = np.abs(X.flatten() - moving_avg)
                anomalies = scores > self.threshold * moving_std
    
            elif self.method == 'seasonal':
                # 1次元時系列のみ対応
                result = seasonal_decompose(X.flatten(), model='additive', period=self.window_size, extrapolate_trend='freq')
                scores = np.abs(result.resid)
                threshold_val = self.threshold * np.nanstd(result.resid)
                anomalies = scores > threshold_val
    
            return anomalies.astype(int), scores
    
        def fit_predict(self, X):
            """学習と予測を一度に実行"""
            self.fit(X)
            return self.predict(X)
    
    # デモンストレーション
    np.random.seed(42)
    
    # データセット生成
    n_samples = 300
    n_features = 2
    X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n_samples)
    X_outliers = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5]])
    X = np.vstack([X_normal, X_outliers])
    y_true = np.array([0]*n_samples + [1]*len(X_outliers))
    
    # 各手法で異常検知
    methods = ['zscore', 'iqr', 'mahalanobis']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, method in enumerate(methods):
        detector = StatisticalAnomalyDetector(method=method, threshold=3.0)
        detector.fit(X_normal)  # 正常データのみで学習
        y_pred, scores = detector.predict(X)
    
        # 可視化
        axes[i].scatter(X[y_pred==0, 0], X[y_pred==0, 1],
                        alpha=0.6, s=50, c='blue', label='正常', edgecolors='black')
        axes[i].scatter(X[y_pred==1, 0], X[y_pred==1, 1],
                        c='red', s=150, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
        axes[i].set_xlabel('特徴量 1', fontsize=12)
        axes[i].set_ylabel('特徴量 2', fontsize=12)
        axes[i].set_title(f'{method.upper()}法', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
        # 評価
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"=== {method.upper()}法 ===")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")
    
    plt.tight_layout()
    plt.show()
    

> **重要** : 実務では、複数の統計的手法を組み合わせたアンサンブルが効果的です。各手法の強みを活かすことで、ロバストな異常検知が可能になります。

* * *

## 本章のまとめ

### 学んだこと

  1. **統計的外れ値検出**

     * Z-score: 簡単で高速、正規分布を仮定
     * IQR: ロバスト、分布に依存しない
  2. **確率分布ベース**

     * Mahalanobis距離: 多変量、共分散考慮
     * 多変量ガウス分布: 確率密度による判定
  3. **統計的仮説検定**

     * Grubbs' Test: 単一外れ値の厳密な検定
     * ESD Test: 複数外れ値の逐次検出
  4. **時系列異常検知**

     * 移動平均: トレンド追従と残差検出
     * 季節性分解: 季節性・トレンド除去
  5. **実装と応用**

     * 統合パイプライン: 複数手法の統一インターフェース
     * 実務適用: ドメインに応じた手法選択

### 統計的手法の選択基準

手法 | データタイプ | 利点 | 欠点  
---|---|---|---  
**Z-score** | 単変量、正規分布 | シンプル、高速 | 外れ値の影響大  
**IQR** | 単変量、任意分布 | ロバスト | 多変量に不向き  
**Mahalanobis** | 多変量、相関あり | 共分散考慮 | 計算コスト高  
**Grubbs/ESD** | 単変量、正規分布 | 統計的根拠明確 | 逐次的処理  
**移動平均** | 時系列 | トレンド追従 | ラグあり  
**季節性分解** | 季節性時系列 | 季節性除去 | 周期要事前知識  
  
### 次の章へ

第3章では、**機械学習による異常検知** を学びます：

  * Isolation Forest, LOF
  * One-Class SVM
  * クラスタリングベース手法
  * アンサンブル手法

* * *

## 演習問題

### 問題1（難易度：easy）

Z-scoreとIQR法の違いを説明し、それぞれがどのような場面で適しているか述べてください。

解答例

**解答** ：

**Z-score** :

  * 計算式: $(x - \mu) / \sigma$
  * 仮定: データが正規分布に従う
  * 閾値: 通常 $|Z| > 3$
  * 特徴: 平均と標準偏差を使用するため、外れ値の影響を受けやすい

**IQR法** :

  * 計算式: $IQR = Q3 - Q1$、異常は $x < Q1 - 1.5 \times IQR$ または $x > Q3 + 1.5 \times IQR$
  * 仮定: 分布に依存しない（ノンパラメトリック）
  * 特徴: 四分位数を使用するため、外れ値に対してロバスト

**適用場面** :

状況 | 推奨手法 | 理由  
---|---|---  
正規分布に近いデータ | Z-score | 統計的根拠が明確  
分布が未知・非正規 | IQR | 分布の仮定不要  
外れ値が既に混入 | IQR | ロバスト性が高い  
高速処理が必要 | Z-score | 計算が単純  
  
### 問題2（難易度：medium）

Mahalanobis距離がユークリッド距離より優れている点を、相関のあるデータの例を用いて説明してください。簡単なPythonコードを含めてください。

解答例

**解答** ：

**Mahalanobis距離の優位性** :

  1. **共分散を考慮** : 変数間の相関を反映
  2. **スケール不変** : 特徴量のスケールに依存しない
  3. **楕円状の境界** : データの分布形状に適応

**実装例** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import euclidean, mahalanobis
    
    # 相関の強いデータ生成
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.9], [0.9, 1]]  # 相関係数0.9
    data = np.random.multivariate_normal(mean, cov, size=300)
    
    # テストポイント（データ分布の外）
    test_point1 = np.array([2, 2])  # 相関方向に沿った点
    test_point2 = np.array([2, -2])  # 相関方向に垂直な点
    
    # 距離計算
    mean_vec = data.mean(axis=0)
    cov_matrix = np.cov(data.T)
    cov_inv = np.linalg.inv(cov_matrix)
    
    euclidean_dist1 = euclidean(test_point1, mean_vec)
    euclidean_dist2 = euclidean(test_point2, mean_vec)
    mahal_dist1 = mahalanobis(test_point1, mean_vec, cov_inv)
    mahal_dist2 = mahalanobis(test_point2, mean_vec, cov_inv)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30, c='blue', label='データ')
    plt.scatter(*test_point1, c='red', s=200, marker='X', label='ポイント1(相関方向)',
                edgecolors='black', linewidths=2, zorder=5)
    plt.scatter(*test_point2, c='orange', s=200, marker='X', label='ポイント2(垂直方向)',
                edgecolors='black', linewidths=2, zorder=5)
    plt.scatter(*mean_vec, c='green', s=200, marker='o', label='中心',
                edgecolors='black', linewidths=2, zorder=5)
    
    # 信頼楕円
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 3 * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean_vec, width, height, angle=angle,
                      edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
    plt.gca().add_patch(ellipse)
    
    plt.xlabel('特徴量 1', fontsize=12)
    plt.ylabel('特徴量 2', fontsize=12)
    plt.title('ユークリッド距離 vs Mahalanobis距離', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print("=== 距離の比較 ===")
    print(f"ポイント1（相関方向）:")
    print(f"  ユークリッド距離: {euclidean_dist1:.3f}")
    print(f"  Mahalanobis距離: {mahal_dist1:.3f}")
    print(f"\nポイント2（垂直方向）:")
    print(f"  ユークリッド距離: {euclidean_dist2:.3f}")
    print(f"  Mahalanobis距離: {mahal_dist2:.3f}")
    print(f"\n→ ユークリッド距離は等しいが、Mahalanobis距離はポイント2が遠い")
    print("  （データの分布形状を正しく反映）")
    

**結論** :

  * ユークリッド距離は両点とも等距離と判定（$\sqrt{8} \approx 2.83$）
  * Mahalanobis距離はポイント2を異常と正しく判定
  * 相関を考慮することで、データの真の分布を捉える

### 問題3（難易度：medium）

Grubbs' TestとGeneralized ESD Testの違いを説明し、複数の外れ値がある場合にどちらが適切か述べてください。

解答例

**解答** ：

**Grubbs' Test** :

  * **目的** : 単一の外れ値を検出
  * **手順** : 最も極端な値1つを検定
  * **問題点** : 複数外れ値がある場合、マスキング効果により検出失敗
  * **マスキング効果** : 複数の外れ値が互いに平均・標準偏差を歪め、検出を妨げる

**Generalized ESD Test** :

  * **目的** : 複数の外れ値を検出（最大k個まで）
  * **手順** : 逐次的に外れ値を除去しながら検定を繰り返す
  * **利点** : マスキング効果を回避
  * **注意点** : 最大外れ値数kを事前に指定必要

**推奨** :

状況 | 推奨手法 | 理由  
---|---|---  
外れ値が1個のみ確実 | Grubbs' Test | シンプルで明確  
複数外れ値の可能性 | Generalized ESD | マスキング回避  
外れ値数が未知 | Generalized ESD | 保守的にk設定  
  
**具体例** :

データ: [50, 51, 49, 52, 48, 100, 105]（外れ値2個：100, 105）

  * Grubbs' Test: 105のみ検出（100は平均が歪んで検出失敗）
  * ESD Test: 105を検出・除去 → 100を検出（逐次的処理で成功）

### 問題4（難易度：hard）

季節性分解（STL）を用いた時系列異常検知において、周期パラメータの設定が重要な理由を説明し、誤った周期を設定した場合の問題点を示してください。実装例を含めてください。

解答例

**解答** ：

**周期パラメータの重要性** :

  1. **正確な季節性除去** : 正しい周期で分解しないと、季節成分が残差に混入
  2. **異常検出精度** : 残差に季節性が残ると、誤検出（FP）が増加
  3. **トレンド推定** : 周期が不適切だと、トレンド成分も歪む

**誤った周期設定の問題** :

  * **過小設定** （真の周期より短い）: 季節性を過剰に除去、真のトレンドを季節性と誤認
  * **過大設定** （真の周期より長い）: 季節性が残差に残る、正常な季節変動を異常判定

**実装例** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 真の周期30の時系列データ生成
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    trend = 0.05 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 30)  # 周期30
    noise = np.random.normal(0, 1, n_samples)
    data = trend + seasonal + noise
    
    # 異常を追加
    data[100] += 25
    data[200] -= 25
    
    # 3種類の周期設定で分解
    periods = [15, 30, 60]  # 過小、正確、過大
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    for i, period in enumerate(periods):
        result = seasonal_decompose(data, model='additive', period=period, extrapolate_trend='freq')
    
        # 元データ
        axes[i, 0].plot(time, data, linewidth=1, alpha=0.7)
        axes[i, 0].set_ylabel(f'周期={period}\n元データ', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
    
        # 季節性
        axes[i, 1].plot(time, result.seasonal, linewidth=1, color='orange')
        axes[i, 1].set_ylabel('季節性', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)
    
        # 残差と異常検出
        residual = result.resid
        threshold = 3 * np.nanstd(residual)
        anomalies = np.abs(residual) > threshold
    
        axes[i, 2].plot(time, residual, linewidth=1, alpha=0.7)
        axes[i, 2].axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        axes[i, 2].axhline(y=-threshold, color='red', linestyle='--', linewidth=2)
        axes[i, 2].scatter(time[anomalies], residual[anomalies],
                           c='red', s=50, marker='X', zorder=5)
        axes[i, 2].set_ylabel('残差', fontsize=10)
        axes[i, 2].grid(True, alpha=0.3)
    
        # 評価
        print(f"=== 周期={period} ===")
        print(f"異常検出数: {anomalies.sum()}個")
        print(f"異常インデックス: {np.where(anomalies)[0]}\n")
    
    axes[0, 0].set_title('元データ', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('季節性成分', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('残差（異常検出）', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('時刻', fontsize=11)
    axes[2, 1].set_xlabel('時刻', fontsize=11)
    axes[2, 2].set_xlabel('時刻', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 結論 ===")
    print("周期=15（過小）: 季節性が複雑化、誤検出が増加")
    print("周期=30（正確）: 適切な分解、異常を正確に検出")
    print("周期=60（過大）: 季節性が残差に残る、正常変動を誤検出")
    

**結論** :

  * 正確な周期設定が異常検知の精度を決定
  * 周期は事前知識（ビジネスサイクル、季節パターン）またはACF/PACF分析で決定
  * 不明な場合は、複数周期を試して残差の分散が最小になる周期を選択

### 問題5（難易度：hard）

統計的異常検知の3つの手法（Z-score, IQR, Mahalanobis距離）を組み合わせたアンサンブル異常検知システムを設計してください。多数決または重み付け投票を実装し、単一手法より性能が向上することを示してください。

解答例

**解答** ：

**アンサンブル異常検知の設計** :

  1. **基本手法の組み合わせ** : 異なる原理の手法を統合（Z-score, IQR, Mahalanobis）
  2. **投票戦略** : 多数決またはソフト投票（スコアの平均）
  3. **重み付け** : 各手法の信頼性に応じた重み

**実装** :
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.spatial.distance import mahalanobis
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    class EnsembleAnomalyDetector:
        """アンサンブル異常検知"""
    
        def __init__(self, voting='hard', weights=None):
            """
            Parameters:
            -----------
            voting : str
                'hard' (多数決) または 'soft' (スコア平均)
            weights : list or None
                各手法の重み [zscore, iqr, mahalanobis]
            """
            self.voting = voting
            self.weights = weights if weights is not None else [1/3, 1/3, 1/3]
    
        def fit(self, X):
            """統計量の学習"""
            # Z-score用
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
    
            # IQR用
            self.q1_ = np.percentile(X, 25, axis=0)
            self.q3_ = np.percentile(X, 75, axis=0)
            self.iqr_ = self.q3_ - self.q1_
    
            # Mahalanobis用
            self.cov_ = np.cov(X.T)
            self.cov_inv_ = np.linalg.inv(self.cov_)
    
            return self
    
        def predict(self, X, threshold_zscore=3, threshold_mahal=3):
            """アンサンブル予測"""
            n_samples = len(X)
    
            # 1. Z-score
            z_scores = np.abs((X - self.mean_) / self.std_)
            z_anomalies = np.any(z_scores > threshold_zscore, axis=1).astype(int)
            z_scores_norm = np.max(z_scores, axis=1) / 5  # 正規化
    
            # 2. IQR
            lower = self.q1_ - 1.5 * self.iqr_
            upper = self.q3_ + 1.5 * self.iqr_
            iqr_anomalies = np.any((X < lower) | (X > upper), axis=1).astype(int)
            iqr_scores = np.max(np.abs(X - self.mean_) / (self.iqr_ + 1e-10), axis=1)
            iqr_scores_norm = np.clip(iqr_scores / 5, 0, 1)  # 正規化
    
            # 3. Mahalanobis距離
            mahal_scores = np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])
            mahal_anomalies = (mahal_scores > threshold_mahal).astype(int)
            mahal_scores_norm = mahal_scores / 10  # 正規化
    
            # アンサンブル投票
            if self.voting == 'hard':
                # 多数決（2/3以上が異常と判定）
                votes = z_anomalies + iqr_anomalies + mahal_anomalies
                predictions = (votes >= 2).astype(int)
                scores = votes / 3
    
            elif self.voting == 'soft':
                # スコアの重み付け平均
                scores = (self.weights[0] * z_scores_norm +
                         self.weights[1] * iqr_scores_norm +
                         self.weights[2] * mahal_scores_norm)
                predictions = (scores > 0.5).astype(int)
    
            return predictions, scores, {
                'zscore': z_anomalies,
                'iqr': iqr_anomalies,
                'mahalanobis': mahal_anomalies
            }
    
    # 評価実験
    np.random.seed(42)
    
    # データ生成
    n_normal = 300
    n_anomaly = 30
    X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n_normal)
    X_anomaly = np.random.uniform(-5, 5, size=(n_anomaly, 2))
    X_anomaly += np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]]).mean(axis=0)  # バイアス
    
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0]*n_normal + [1]*n_anomaly)
    
    # モデル訓練（正常データのみ）
    ensemble = EnsembleAnomalyDetector(voting='soft', weights=[0.3, 0.3, 0.4])
    ensemble.fit(X_normal)
    
    # 予測
    y_pred_ensemble, scores_ensemble, individual_preds = ensemble.predict(X)
    
    # 各手法の個別評価
    results = {}
    for method_name, preds in individual_preds.items():
        results[method_name] = {
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'f1': f1_score(y_true, preds)
        }
    
    # アンサンブルの評価
    results['ensemble'] = {
        'precision': precision_score(y_true, y_pred_ensemble),
        'recall': recall_score(y_true, y_pred_ensemble),
        'f1': f1_score(y_true, y_pred_ensemble)
    }
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    methods = ['zscore', 'iqr', 'mahalanobis', 'ensemble']
    titles = ['Z-score', 'IQR', 'Mahalanobis距離', 'アンサンブル']
    predictions_list = [individual_preds['zscore'], individual_preds['iqr'],
                        individual_preds['mahalanobis'], y_pred_ensemble]
    
    for i, (method, title, preds) in enumerate(zip(methods, titles, predictions_list)):
        ax = axes[i // 2, i % 2]
        ax.scatter(X[preds==0, 0], X[preds==0, 1],
                   alpha=0.6, s=50, c='blue', label='正常', edgecolors='black')
        ax.scatter(X[preds==1, 0], X[preds==1, 1],
                   c='red', s=100, marker='X', label='異常', zorder=5, edgecolors='black', linewidths=2)
    
        # 性能表示
        r = results[method]
        ax.set_xlabel('特徴量 1', fontsize=12)
        ax.set_ylabel('特徴量 2', fontsize=12)
        ax.set_title(f'{title}\nF1={r["f1"]:.3f}, Precision={r["precision"]:.3f}, Recall={r["recall"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果サマリ
    print("=== 性能比較 ===")
    for method, metrics in results.items():
        print(f"{method.upper():15s} - Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
    
    print("\n=== 結論 ===")
    print("アンサンブルは各手法の強みを組み合わせ、単一手法より安定した性能を実現")
    

**結論** :

  * アンサンブルは単一手法のバイアスを軽減
  * ソフト投票は柔軟性が高く、重み調整で最適化可能
  * 実務では、ドメイン知識に基づいて重みを設定

* * *

## 参考文献

  1. Rousseeuw, P. J., & Hubert, M. (2011). _Robust statistics for outlier detection_. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 1(1), 73-79.
  2. Barnett, V., & Lewis, T. (1994). _Outliers in statistical data_ (3rd ed.). John Wiley & Sons.
  3. Grubbs, F. E. (1969). _Procedures for detecting outlying observations in samples_. Technometrics, 11(1), 1-21.
  4. Rosner, B. (1983). _Percentage points for a generalized ESD many-outlier procedure_. Technometrics, 25(2), 165-172.
  5. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). _STL: A seasonal-trend decomposition procedure based on loess_. Journal of Official Statistics, 6(1), 3-73.
