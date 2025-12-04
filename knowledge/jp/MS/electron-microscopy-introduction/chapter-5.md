---
title: 第5章：EDS・EELS・EBSD分析実践
chapter_title: 第5章：EDS・EELS・EBSD分析実践
subtitle: HyperSpyワークフロー、PCA/ICA、機械学習分類、EBSD方位解析
reading_time: 30-40分
difficulty: 中級〜上級
code_examples: 7
---

本章では、EDS・EELS・EBSDデータをPythonで統合的に解析する実践的ワークフローを学びます。HyperSpyによるスペクトル処理、PCA/ICAによる次元削減、機械学習による相分類、EELSバックグラウンド処理、EBSD方位解析（KAM、GNDマップ）を習得し、実際の材料解析に応用できる力を身につけます。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ HyperSpyの基本操作（データ読込、可視化、前処理）ができる
  * ✅ EELSスペクトルのバックグラウンド除去とピークフィッティングができる
  * ✅ PCA/ICAで高次元スペクトルデータを次元削減できる
  * ✅ 機械学習（k-means, GMM, SVM）で相自動分類ができる
  * ✅ EBSD方位データをorixで読込み、方位マップを作成できる
  * ✅ KAM（局所方位差）、GND（幾何学的必要転位密度）を計算できる
  * ✅ 統合解析ワークフローを構築し、実データに適用できる

## 5.1 HyperSpyによるスペクトル解析基礎

### 5.1.1 HyperSpyとは

**HyperSpy** は、電子顕微鏡スペクトルデータ（EELS, EDS, CL, XRF等）の解析に特化したPythonライブラリです。

**主な機能** ：

  * 多次元スペクトルデータ（Spectrum Image）の読込・可視化
  * バックグラウンド除去、ピークフィッティング、定量分析
  * 多変量統計解析（PCA, ICA, NMF）
  * 機械学習統合（scikit-learn連携）
  * バッチ処理とスクリプト化

    
    
    ```mermaid
    flowchart LR
        A[Raw Datadm3, hspy, msa] --> B[HyperSpyLoad & Visualize]
        B --> C[PreprocessingAlign, Crop, Bin]
        C --> D[BackgroundRemoval]
        D --> E{Analysis Type}
        E -->|Quantification| F[Element Maps]
        E -->|Dimensionality| G[PCA/ICA]
        E -->|Machine Learning| H[Classification]
        G --> I[Component Maps]
        H --> I
        F --> I
        I --> J[IntegratedInterpretation]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style J fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
    ```

### 5.1.2 HyperSpy基本ワークフロー

#### コード例5-1: HyperSpy基本操作とEELSスペクトル可視化
    
    
    import hyperspy.api as hs
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ダミーEELSスペクトラムイメージ生成（実際のデータはhs.load()で読込）
    def create_dummy_eels_si(size=64, energy_range=(400, 1000)):
        """
        ダミーEELS Spectrum Imageを生成
    
        Parameters
        ----------
        size : int
            空間サイズ [pixels]
        energy_range : tuple
            エネルギー範囲 [eV]
    
        Returns
        -------
        s : hyperspy Signal1D
            EELS Spectrum Image
        """
        # エネルギー軸
        energy = np.linspace(energy_range[0], energy_range[1], 500)
    
        # 空間依存の模擬スペクトル
        # 領域1: Fe-L2,3 edge (708 eV)
        # 領域2: O-K edge (532 eV)
    
        data = np.zeros((size, size, len(energy)))
    
        for i in range(size):
            for j in range(size):
                # バックグラウンド
                bg = 1000 * (energy / energy[0])**(-3)
    
                # 領域依存のエッジ追加
                if i < size // 2:  # 左半分: Fe rich
                    fe_edge = energy >= 708
                    bg[fe_edge] += 200 * np.exp(-(energy[fe_edge] - 708) / 50)
                else:  # 右半分: O rich
                    o_edge = energy >= 532
                    bg[o_edge] += 150 * np.exp(-(energy[o_edge] - 532) / 40)
    
                # ノイズ
                data[i, j, :] = bg + np.random.poisson(lam=10, size=len(energy))
    
        # HyperSpy Signal1D作成
        s = hs.signals.Signal1D(data)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'pixels'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].units = 'pixels'
        s.axes_manager[2].name = 'Energy'
        s.axes_manager[2].units = 'eV'
        s.axes_manager[2].offset = energy_range[0]
        s.axes_manager[2].scale = (energy_range[1] - energy_range[0]) / len(energy)
    
        s.metadata.General.title = 'EELS Spectrum Image (Dummy)'
        s.metadata.Signal.signal_type = 'EELS'
    
        return s
    
    # ダミーデータ生成
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    print("HyperSpy Signal情報:")
    print(s)
    print(f"\nデータ形状: {s.data.shape}")
    print(f"空間サイズ: {s.axes_manager[0].size} × {s.axes_manager[1].size}")
    print(f"エネルギー点数: {s.axes_manager[2].size}")
    
    # 基本可視化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 左図: 平均スペクトル
    s_mean = s.mean()
    axes[0].plot(s_mean.axes_manager[0].axis, s_mean.data, linewidth=2)
    axes[0].set_xlabel('Energy Loss [eV]', fontsize=12)
    axes[0].set_ylabel('Intensity [counts]', fontsize=12)
    axes[0].set_title('Mean EELS Spectrum', fontsize=13, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    
    # 中図: 特定エネルギーでの空間マップ（532 eV: O-K）
    idx_o = int((532 - 400) / (1000 - 400) * s.axes_manager[2].size)
    im1 = axes[1].imshow(s.data[:, :, idx_o], cmap='viridis')
    axes[1].set_title('Spatial Map at 532 eV\n(O-K edge)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # 右図: 特定位置でのスペクトル比較
    pos1 = (10, 32)  # Fe rich
    pos2 = (50, 32)  # O rich
    
    s1 = s.inav[pos1[0], pos1[1]]
    s2 = s.inav[pos2[0], pos2[1]]
    
    axes[2].plot(s1.axes_manager[0].axis, s1.data, label=f'Position {pos1} (Fe rich)', linewidth=2)
    axes[2].plot(s2.axes_manager[0].axis, s2.data, label=f'Position {pos2} (O rich)', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('Energy Loss [eV]', fontsize=12)
    axes[2].set_ylabel('Intensity [counts]', fontsize=12)
    axes[2].set_title('Spectra at Different Positions', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存例（実際のワークフローで使用）
    # s.save('my_eels_data.hspy')  # HyperSpy形式
    # s.save('my_eels_data.msa')   # MSA形式（Digital Micrograph互換）
    

### 5.1.3 EELSバックグラウンド除去

EELSスペクトルのコアロスエッジを定量するには、エッジ前のバックグラウンドを除去する必要があります。HyperSpyではべき乗則フィッティングが標準的です：

$$ I_{\text{BG}}(E) = A \cdot E^{-r} $$ 

#### コード例5-2: EELSバックグラウンド除去とピーク積分
    
    
    import hyperspy.api as hs
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 前例のダミーデータを使用
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    # 特定位置のスペクトル抽出
    pos = (10, 32)  # Fe rich領域
    s_point = s.inav[pos[0], pos[1]]
    
    # バックグラウンド除去（Fe-L2,3 edge用）
    # エッジ前領域でフィット
    edge_onset = 708  # Fe-L2,3 edge [eV]
    fit_range = (650, 700)  # フィット範囲 [eV]
    
    # HyperSpyのバックグラウンド除去機能
    s_point_bg_removed = s_point.remove_background(
        signal_range=fit_range,
        background_type='PowerLaw',
        fast=False
    )
    
    # 積分窓設定（エッジ後50 eV）
    integration_window = (edge_onset, edge_onset + 50)
    
    # 積分強度計算（台形則）
    energy_axis = s_point_bg_removed.axes_manager[0].axis
    mask = (energy_axis >= integration_window[0]) & (energy_axis <= integration_window[1])
    integrated_intensity = np.trapz(s_point_bg_removed.data[mask], energy_axis[mask])
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上図: バックグラウンド除去前後
    ax1.plot(s_point.axes_manager[0].axis, s_point.data, 'b-', linewidth=2, label='Raw Spectrum')
    
    # バックグラウンド曲線を再計算して表示
    from hyperspy.components1d import PowerLaw
    bg_model = PowerLaw()
    bg_model.fit(s_point, fit_range[0], fit_range[1])
    bg_curve = bg_model.function(s_point.axes_manager[0].axis)
    
    ax1.plot(s_point.axes_manager[0].axis, bg_curve, 'r--', linewidth=2, label='Background Fit')
    ax1.axvspan(fit_range[0], fit_range[1], alpha=0.2, color='yellow', label='Fit Region')
    ax1.axvline(edge_onset, color='green', linestyle=':', linewidth=2, label=f'Fe-L edge ({edge_onset} eV)')
    
    ax1.set_xlabel('Energy Loss [eV]', fontsize=12)
    ax1.set_ylabel('Intensity [counts]', fontsize=12)
    ax1.set_title('EELS Spectrum: Raw + Background Fit', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(500, 900)
    
    # 下図: バックグラウンド除去後
    ax2.plot(s_point_bg_removed.axes_manager[0].axis, s_point_bg_removed.data, 'g-', linewidth=2, label='Background Removed')
    ax2.axvline(edge_onset, color='green', linestyle=':', linewidth=2, label=f'Fe-L edge')
    ax2.axvspan(integration_window[0], integration_window[1], alpha=0.3, color='lightgreen', label='Integration Window')
    
    ax2.set_xlabel('Energy Loss [eV]', fontsize=12)
    ax2.set_ylabel('Intensity [counts]', fontsize=12)
    ax2.set_title(f'Background-Removed Spectrum (Integrated Intensity: {integrated_intensity:.0f})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(500, 900)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"積分強度（Fe-L edge）: {integrated_intensity:.1f} counts")
    print(f"この値を断面積で補正して元素濃度を算出")
    

## 5.2 多変量統計解析（PCA/ICA）

### 5.2.1 主成分分析（PCA）の原理

高次元スペクトルデータ（例：64×64空間 × 500エネルギー点）を少数の主成分に次元削減することで、ノイズ除去と相分離が可能になります。

**PCAの利点** ：

  * データの分散を最大化する方向を抽出（主成分）
  * 上位数成分で元データの90%以上の情報を保持可能
  * ノイズは低次成分に分離されるため、S/N比向上
  * 成分マップで相の空間分布を可視化

#### コード例5-3: PCA/ICAによるスペクトル分離
    
    
    import hyperspy.api as hs
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA, FastICA
    
    # ダミーデータ生成（2相混合）
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    # PCA実行
    s.decomposition(algorithm='PCA', output_dimension=10)
    
    # スクリープロット（主成分の寄与率）
    s.plot_explained_variance_ratio(n=10)
    
    # 上位3主成分のロードスペクトルとスコアマップ
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i in range(3):
        # ロードスペクトル（主成分のスペクトル形状）
        loading = s.get_decomposition_loadings().inav[i]
    
        axes[i, 0].plot(loading.axes_manager[0].axis, loading.data, linewidth=2)
        axes[i, 0].set_xlabel('Energy Loss [eV]', fontsize=11)
        axes[i, 0].set_ylabel('Loading', fontsize=11)
        axes[i, 0].set_title(f'PC{i+1} Loading Spectrum', fontsize=12, fontweight='bold')
        axes[i, 0].grid(alpha=0.3)
    
        # スコアマップ（主成分の空間分布）
        factor = s.get_decomposition_factors().inav[i]
    
        im = axes[i, 1].imshow(factor.data, cmap='RdBu_r')
        axes[i, 1].set_title(f'PC{i+1} Score Map', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
    
        # 累積寄与率
        if i == 0:
            variance_ratio = s.get_explained_variance_ratio().data
            cumsum = np.cumsum(variance_ratio[:10])
    
            axes[i, 2].bar(range(1, 11), variance_ratio[:10], alpha=0.7, label='Individual')
            axes[i, 2].plot(range(1, 11), cumsum, 'ro-', linewidth=2, markersize=6, label='Cumulative')
            axes[i, 2].set_xlabel('Principal Component', fontsize=11)
            axes[i, 2].set_ylabel('Explained Variance Ratio', fontsize=11)
            axes[i, 2].set_title('Scree Plot', fontsize=12, fontweight='bold')
            axes[i, 2].legend(fontsize=10)
            axes[i, 2].grid(alpha=0.3)
        else:
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 上位3成分で再構成
    s_reconstructed = s.get_decomposition_model(components=3)
    
    print(f"元データサイズ: {s.data.nbytes / 1e6:.1f} MB")
    print(f"PCA3成分のサイズ: {s_reconstructed.data.nbytes / 1e6:.1f} MB")
    print(f"圧縮率: {s.data.nbytes / s_reconstructed.data.nbytes:.1f}倍")
    
    # ICA実行（独立成分分析）
    # ICAはPCAより物理的に意味のある成分を抽出しやすい
    s.decomposition(algorithm='PCA', output_dimension=5)  # 前処理
    s.blind_source_separation(number_of_components=3, algorithm='FastICA')
    
    # ICA成分のプロット
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10))
    
    for i in range(3):
        # ICAロードスペクトル
        loading_ica = s.get_bss_loadings().inav[i]
        axes2[i, 0].plot(loading_ica.axes_manager[0].axis, loading_ica.data, linewidth=2, color='purple')
        axes2[i, 0].set_xlabel('Energy Loss [eV]', fontsize=11)
        axes2[i, 0].set_ylabel('IC Loading', fontsize=11)
        axes2[i, 0].set_title(f'ICA Component {i+1} Spectrum', fontsize=12, fontweight='bold')
        axes2[i, 0].grid(alpha=0.3)
    
        # ICAスコアマップ
        factor_ica = s.get_bss_factors().inav[i]
        im2 = axes2[i, 1].imshow(factor_ica.data, cmap='viridis')
        axes2[i, 1].set_title(f'ICA Component {i+1} Map', fontsize=12, fontweight='bold')
        axes2[i, 1].axis('off')
        plt.colorbar(im2, ax=axes2[i, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    print("\nPCA: データの分散を最大化（数学的最適）")
    print("ICA: 統計的に独立な成分を抽出（物理的解釈が容易）")
    

## 5.3 機械学習による相自動分類

### 5.3.1 教師なし学習（k-means, GMM）

スペクトルデータを特徴空間でクラスタリングし、異なる相や領域を自動分類します。

手法 | 原理 | 利点 | 欠点  
---|---|---|---  
**k-means** | ユークリッド距離でk個のクラスタに分割 | 高速、実装が簡単 | 球状クラスタ仮定、クラスタ数を事前指定  
**GMM** | ガウス混合モデルで確率的分類 | 楕円クラスタ対応、所属確率を出力 | 計算コストがやや高い  
**階層的** | デンドログラムで類似度を階層化 | クラスタ数自動決定、視覚的理解容易 | 大規模データに不向き  
  
#### コード例5-4: k-meansとGMMによる相分類
    
    
    import hyperspy.api as hs
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    
    # ダミーデータ生成
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    # データ整形（空間次元を平坦化）
    data_reshaped = s.data.reshape(-1, s.data.shape[2])  # (N_pixels, N_energy)
    
    # 標準化（特徴量のスケールを揃える）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # k-meansクラスタリング
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(data_scaled)
    labels_kmeans_map = labels_kmeans.reshape(s.data.shape[0], s.data.shape[1])
    
    # クラスタ中心のスペクトル
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # GMMクラスタリング
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
    gmm.fit(data_scaled)
    labels_gmm = gmm.predict(data_scaled)
    labels_gmm_map = labels_gmm.reshape(s.data.shape[0], s.data.shape[1])
    
    # 所属確率マップ
    proba_gmm = gmm.predict_proba(data_scaled)
    proba_gmm_maps = proba_gmm.reshape(s.data.shape[0], s.data.shape[1], n_clusters)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # k-meansクラスタマップ
    im0 = axes[0, 0].imshow(labels_kmeans_map, cmap='tab10', vmin=0, vmax=n_clusters-1)
    axes[0, 0].set_title('k-means Cluster Map', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], ticks=range(n_clusters), fraction=0.046)
    
    # k-meansクラスタ中心スペクトル
    for i in range(n_clusters):
        axes[0, 1].plot(s.axes_manager[2].axis, cluster_centers[i], linewidth=2, label=f'Cluster {i}')
    axes[0, 1].set_xlabel('Energy Loss [eV]', fontsize=11)
    axes[0, 1].set_ylabel('Intensity [a.u.]', fontsize=11)
    axes[0, 1].set_title('k-means Cluster Centers', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # GMMクラスタマップ
    im1 = axes[0, 2].imshow(labels_gmm_map, cmap='tab10', vmin=0, vmax=n_clusters-1)
    axes[0, 2].set_title('GMM Cluster Map', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], ticks=range(n_clusters), fraction=0.046)
    
    # GMM平均スペクトル
    gmm_means = scaler.inverse_transform(gmm.means_)
    for i in range(n_clusters):
        axes[0, 3].plot(s.axes_manager[2].axis, gmm_means[i], linewidth=2, label=f'Component {i}')
    axes[0, 3].set_xlabel('Energy Loss [eV]', fontsize=11)
    axes[0, 3].set_ylabel('Intensity [a.u.]', fontsize=11)
    axes[0, 3].set_title('GMM Component Means', fontsize=13, fontweight='bold')
    axes[0, 3].legend(fontsize=10)
    axes[0, 3].grid(alpha=0.3)
    axes[0, 3].set_yscale('log')
    
    # GMM所属確率マップ（各成分）
    for i in range(n_clusters):
        im = axes[1, i].imshow(proba_gmm_maps[:, :, i], cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'GMM Component {i}\nProbability Map', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    # 最後の軸を空白に
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # クラスタリング品質評価（シルエットスコア）
    from sklearn.metrics import silhouette_score
    
    silhouette_kmeans = silhouette_score(data_scaled, labels_kmeans)
    silhouette_gmm = silhouette_score(data_scaled, labels_gmm)
    
    print(f"k-meansシルエットスコア: {silhouette_kmeans:.3f} (1に近いほど良好)")
    print(f"GMMシルエットスコア: {silhouette_gmm:.3f}")
    print("\nGMMの利点: 所属確率を出力するため、境界領域の解釈が容易")
    

### 5.3.2 教師あり学習（SVM, Random Forest）

既知の相のスペクトルで学習し、未知領域を分類する手法。EDS/EELSで元素比が既知の領域をトレーニングデータとして使用します。

#### コード例5-5: SVMによる相分類（教師あり）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    
    # ダミーデータ生成（明確な2相）
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    # 手動ラベリング（実際は既知領域から作成）
    labels_true = np.zeros((64, 64), dtype=int)
    labels_true[:, :32] = 0  # Phase A (Fe-rich)
    labels_true[:, 32:] = 1  # Phase B (O-rich)
    
    # データ整形
    data_reshaped = s.data.reshape(-1, s.data.shape[2])
    labels_flat = labels_true.flatten()
    
    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        data_reshaped, labels_flat, test_size=0.3, random_state=42, stratify=labels_flat
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM学習
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # テストデータで評価
    y_pred = svm.predict(X_test_scaled)
    
    # 全データで予測
    y_pred_all = svm.predict(scaler.transform(data_reshaped))
    pred_map = y_pred_all.reshape(64, 64)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 真のラベル
    im0 = axes[0, 0].imshow(labels_true, cmap='coolwarm', vmin=0, vmax=1)
    axes[0, 0].set_title('True Labels\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], ticks=[0, 1], fraction=0.046)
    cbar0.set_ticklabels(['Phase A', 'Phase B'])
    
    # SVM予測ラベル
    im1 = axes[0, 1].imshow(pred_map, cmap='coolwarm', vmin=0, vmax=1)
    axes[0, 1].set_title('SVM Predicted Labels\n(RBF kernel)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], ticks=[0, 1], fraction=0.046)
    cbar1.set_ticklabels(['Phase A', 'Phase B'])
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    im2 = axes[1, 0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[1, 0].set_title('Confusion Matrix\n(Test Data)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Phase A', 'Phase B'])
    axes[1, 0].set_yticklabels(['Phase A', 'Phase B'])
    
    # 混同行列の値を表示
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # 分類レポート
    report = classification_report(y_test, y_pred, target_names=['Phase A', 'Phase B'], output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    phase_a = [report['Phase A'][m] for m in metrics]
    phase_b = [report['Phase B'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, phase_a, width, label='Phase A', alpha=0.8)
    axes[1, 1].bar(x + width/2, phase_b, width, label='Phase B', alpha=0.8)
    
    axes[1, 1].set_xlabel('Metric', fontsize=11)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Classification Performance\n(Test Data)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=['Phase A', 'Phase B']))
    print(f"テスト精度: {svm.score(X_test_scaled, y_test):.3f}")
    

## 5.4 EBSD方位データ解析

### 5.4.1 orix/pyxemによるEBSDデータ処理

**orix** は、結晶方位データ（オイラー角、四元数、回転行列）を扱うPythonライブラリです。EBSDデータの読込、方位解析、結晶学的統計処理が可能です。

**主な機能** ：

  * EBSD方位データの読込（EDAX, Oxford, Bruker形式）
  * 方位表現の変換（オイラー角、四元数、回転行列）
  * 局所方位差（KAM: Kernel Average Misorientation）計算
  * 幾何学的必要転位密度（GND: Geometrically Necessary Dislocation）推定
  * 極点図、逆極点図の作成

#### コード例5-6: KAM（局所方位差）マップの作成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve
    
    def generate_dummy_ebsd_data(size=128, num_grains=20):
        """
        ダミーEBSDデータ生成（方位マップ）
    
        Parameters
        ----------
        size : int
            マップサイズ [pixels]
        num_grains : int
            結晶粒数
    
        Returns
        -------
        euler_map : ndarray
            オイラー角マップ (size, size, 3) [degrees]
        grain_map : ndarray
            結晶粒IDマップ (size, size)
        """
        from scipy.spatial import Voronoi
        from scipy.spatial.distance import cdist
    
        # ボロノイ分割で結晶粒生成
        np.random.seed(42)
        points = np.random.rand(num_grains, 2) * size
    
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        pixels = np.stack([x.ravel(), y.ravel()], axis=1)
    
        distances = cdist(pixels, points)
        grain_map = np.argmin(distances, axis=1).reshape(size, size)
    
        # 各結晶粒にランダムな方位を割り当て
        euler_angles_grains = np.random.rand(num_grains, 3) * [360, 180, 360]
    
        euler_map = np.zeros((size, size, 3))
        for grain_id in range(num_grains):
            mask = grain_map == grain_id
            euler_map[mask] = euler_angles_grains[grain_id]
    
        # 粒界近傍に格子回転勾配を追加（転位蓄積）
        from scipy.ndimage import distance_transform_edt, binary_erosion
    
        for grain_id in range(num_grains):
            mask = grain_map == grain_id
            # 粒界までの距離
            dist = distance_transform_edt(mask)
            # 粒界近傍（3ピクセル以内）で回転を追加
            boundary_zone = (dist > 0) & (dist < 3)
            if np.sum(boundary_zone) > 0:
                rotation_gradient = np.random.rand() * 5  # 最大5度の回転
                euler_map[boundary_zone, 0] += rotation_gradient * (3 - dist[boundary_zone]) / 3
    
        return euler_map, grain_map
    
    def calculate_kam(euler_map, max_misorientation=5):
        """
        KAM (Kernel Average Misorientation) を計算
    
        Parameters
        ----------
        euler_map : ndarray
            オイラー角マップ (H, W, 3) [degrees]
        max_misorientation : float
            KAM計算に含める最大方位差 [degrees]
    
        Returns
        -------
        kam_map : ndarray
            KAMマップ (H, W) [degrees]
        """
        h, w = euler_map.shape[:2]
        kam_map = np.zeros((h, w))
    
        # 最近傍8ピクセルとの方位差を計算（簡略版）
        for i in range(1, h-1):
            for j in range(1, w-1):
                center_euler = euler_map[i, j]
    
                # 最近傍のオイラー角
                neighbors = [
                    euler_map[i-1, j], euler_map[i+1, j],
                    euler_map[i, j-1], euler_map[i, j+1],
                    euler_map[i-1, j-1], euler_map[i-1, j+1],
                    euler_map[i+1, j-1], euler_map[i+1, j+1]
                ]
    
                misorientations = []
                for neighbor in neighbors:
                    # 簡略的な方位差計算（実際はロドリゲスベクトルで計算）
                    diff = np.abs(neighbor - center_euler)
                    # オイラー角の周期性を考慮
                    diff[0] = min(diff[0], 360 - diff[0])
                    diff[1] = min(diff[1], 180 - diff[1])
                    diff[2] = min(diff[2], 360 - diff[2])
    
                    # ノルム（簡略版）
                    misorientation = np.linalg.norm(diff)
    
                    if misorientation < max_misorientation:
                        misorientations.append(misorientation)
    
                if len(misorientations) > 0:
                    kam_map[i, j] = np.mean(misorientations)
    
        return kam_map
    
    # ダミーデータ生成
    euler_map, grain_map = generate_dummy_ebsd_data(size=128, num_grains=20)
    
    # KAM計算
    kam_map = calculate_kam(euler_map, max_misorientation=5)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 結晶粒マップ
    im0 = axes[0, 0].imshow(grain_map, cmap='tab20')
    axes[0, 0].set_title('Grain Map\n(Voronoi tessellation)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    # オイラー角マップ（φ1のみ表示）
    im1 = axes[0, 1].imshow(euler_map[:, :, 0], cmap='hsv', vmin=0, vmax=360)
    axes[0, 1].set_title('Euler Angle φ1 Map\n(Crystal Orientation)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    cbar1.set_label('φ1 [degrees]', fontsize=10)
    
    # KAMマップ
    im2 = axes[1, 0].imshow(kam_map, cmap='jet', vmin=0, vmax=3)
    axes[1, 0].set_title('KAM Map\n(Local Misorientation)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    cbar2.set_label('KAM [degrees]', fontsize=10)
    
    # KAMヒストグラム
    kam_flat = kam_map[kam_map > 0].flatten()
    
    axes[1, 1].hist(kam_flat, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(kam_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(kam_flat):.2f}°')
    axes[1, 1].axvline(np.median(kam_flat), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(kam_flat):.2f}°')
    axes[1, 1].set_xlabel('KAM [degrees]', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('KAM Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("KAM解釈:")
    print(f"  平均KAM: {np.mean(kam_flat):.3f}°")
    print(f"  高KAM領域（>2°）: 転位密度が高い、変形領域")
    print(f"  低KAM領域（<1°）: 再結晶粒、回復組織")
    

### 5.4.2 GND（幾何学的必要転位）密度推定

局所的な格子曲率から、幾何学的に必要な転位密度を推定します。Nye tensorから計算されます。

#### コード例5-7: GND密度マップの簡易計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import sobel
    
    def estimate_gnd_density(euler_map, step_size=0.1):
        """
        GND (Geometrically Necessary Dislocation) 密度を推定
    
        Parameters
        ----------
        euler_map : ndarray
            オイラー角マップ (H, W, 3) [degrees]
        step_size : float
            ピクセル間の物理距離 [μm]
    
        Returns
        -------
        gnd_density : ndarray
            GND密度マップ (H, W) [m^-2]
        """
        # 簡略版: 方位勾配からNye tensorの対角成分を推定
    
        # オイラー角をラジアンに変換
        euler_rad = np.deg2rad(euler_map)
    
        # φ1の勾配（回転勾配の代理指標）
        grad_x = sobel(euler_rad[:, :, 0], axis=1, mode='reflect')
        grad_y = sobel(euler_rad[:, :, 0], axis=0, mode='reflect')
    
        # 曲率の大きさ
        curvature = np.sqrt(grad_x**2 + grad_y**2)
    
        # Burger's vectorの大きさ（Al FCC: b ≈ 0.286 nm）
        b = 0.286e-9  # [m]
    
        # GND密度 ρ = κ / b （簡略式）
        # 曲率を物理単位に変換（rad/pixel → rad/μm）
        curvature_per_um = curvature / (step_size * 1e-6)  # [rad/m]
    
        gnd_density = curvature_per_um / b  # [m^-2]
    
        return gnd_density
    
    # 前例のEBSDデータを使用
    euler_map, grain_map = generate_dummy_ebsd_data(size=128, num_grains=20)
    
    # GND密度推定
    gnd_density = estimate_gnd_density(euler_map, step_size=0.1)  # 0.1 μm/pixel
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 結晶粒マップ
    im0 = axes[0, 0].imshow(grain_map, cmap='tab20')
    axes[0, 0].set_title('Grain Map', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    # KAMマップ
    kam_map = calculate_kam(euler_map, max_misorientation=5)
    im1 = axes[0, 1].imshow(kam_map, cmap='jet', vmin=0, vmax=3)
    axes[0, 1].set_title('KAM Map\n(Local Misorientation)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    cbar1.set_label('KAM [degrees]', fontsize=10)
    
    # GND密度マップ
    gnd_log = np.log10(gnd_density + 1e12)  # 対数スケール（ゼロ除算回避）
    im2 = axes[1, 0].imshow(gnd_log, cmap='viridis', vmin=12, vmax=15)
    axes[1, 0].set_title('GND Density Map\n(Log scale)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    cbar2.set_label('log₁₀(ρ [m⁻²])', fontsize=10)
    
    # KAM vs GND散布図
    kam_flat = kam_map[kam_map > 0].flatten()
    gnd_flat = gnd_density[kam_map > 0].flatten()
    
    axes[1, 1].scatter(kam_flat, gnd_flat, alpha=0.3, s=10)
    axes[1, 1].set_xlabel('KAM [degrees]', fontsize=12)
    axes[1, 1].set_ylabel('GND Density [m⁻²]', fontsize=12)
    axes[1, 1].set_title('KAM vs GND Correlation', fontsize=13, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    # 相関係数
    correlation = np.corrcoef(kam_flat, np.log10(gnd_flat + 1e12))[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print("GND密度の解釈:")
    print(f"  典型的範囲: 10^12 〜 10^15 m^-2")
    print(f"  10^12-10^13 m^-2: 焼鈍材、低変形")
    print(f"  10^14-10^15 m^-2: 冷間加工材、高変形")
    print(f"  粒界近傍でGND密度が高い: 格子曲率が大きい")
    

## 5.5 演習問題

### 演習5-1: HyperSpy基本操作（易）

**問題** ：HyperSpyで読み込んだEELSデータの平均スペクトルを計算し、700 eV付近のピーク強度を抽出せよ。

**解答例を表示**
    
    
    import hyperspy.api as hs
    import numpy as np
    
    # データ読込（実際のファイルパスに置き換え）
    # s = hs.load('my_eels_data.hspy')
    
    # ダミーデータで代替
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    
    # 平均スペクトル
    s_mean = s.mean()
    
    # 700 eV付近の強度抽出
    energy_axis = s_mean.axes_manager[0].axis
    idx_700 = np.argmin(np.abs(energy_axis - 700))
    intensity_700 = s_mean.data[idx_700]
    
    print(f"700 eV付近の強度: {intensity_700:.1f} counts")
    print(f"正確なエネルギー: {energy_axis[idx_700]:.1f} eV")
    

### 演習5-2: PCA成分数の決定（易）

**問題** ：PCAで累積寄与率が95%を超えるために必要な主成分数を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    # PCA実行
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    s.decomposition(algorithm='PCA', output_dimension=20)
    
    # 累積寄与率
    variance_ratio = s.get_explained_variance_ratio().data
    cumsum = np.cumsum(variance_ratio)
    
    # 95%を超える最小成分数
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    
    print(f"累積寄与率95%に必要な主成分数: {n_components_95}")
    print(f"上位{n_components_95}成分の累積寄与率: {cumsum[n_components_95-1]:.3f}")
    

### 演習5-3: クラスタ数の最適化（中）

**問題** ：k-meansでクラスタ数を2〜10に変化させ、エルボー法（慣性）とシルエットスコアで最適なクラスタ数を決定せよ。

**解答例を表示**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    data_reshaped = s.data.reshape(-1, s.data.shape[2])
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # クラスタ数を変化させて評価
    K_range = range(2, 11)
    inertias = []
    silhouettes = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
    
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data_scaled, labels))
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # エルボープロット
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # シルエットスコア
    ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    optimal_k = K_range[np.argmax(silhouettes)]
    ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2,
                label=f'Optimal k={optimal_k}')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print(f"シルエットスコアが最大となるクラスタ数: {optimal_k}")
    

### 演習5-4: EELSバックグラウンド除去（中）

**問題** ：O-K edge（532 eV）のEELSスペクトルでバックグラウンド除去を行い、積分強度を計算せよ。フィット範囲は480-520 eV、積分窓は532-582 eVとする。

**解答例を表示**
    
    
    import hyperspy.api as hs
    import numpy as np
    
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    s_point = s.inav[50, 32]  # O-rich領域
    
    # バックグラウンド除去
    edge_onset = 532
    fit_range = (480, 520)
    s_bg_removed = s_point.remove_background(
        signal_range=fit_range,
        background_type='PowerLaw',
        fast=False
    )
    
    # 積分
    integration_window = (532, 582)
    energy_axis = s_bg_removed.axes_manager[0].axis
    mask = (energy_axis >= integration_window[0]) & (energy_axis <= integration_window[1])
    integrated_intensity = np.trapz(s_bg_removed.data[mask], energy_axis[mask])
    
    print(f"O-K edge積分強度: {integrated_intensity:.1f} counts")
    

### 演習5-5: ICA vs PCA（中）

**問題** ：PCAとICAで上位3成分のスペクトルを抽出し、どちらが物理的に解釈しやすいか、理由とともに答えよ。

**解答例を表示**

**実装** ：コード例5-3参照。

**比較** ：

  * **PCA** ：第1主成分は平均スペクトルに近く、第2成分以降は差分（残差）を表現。数学的に最適だが、物理的意味が不明瞭
  * **ICA** ：統計的に独立な成分を抽出。各成分が特定の相やエッジに対応しやすく、物理的解釈が容易
  * **結論** ：相分離やエッジ分離にはICAが有効。ただしPCAを前処理として使用するのが一般的

### 演習5-6: SVM vs Random Forest（中）

**問題** ：SVMとRandom Forestで同じEELSデータを分類し、精度、計算時間、過学習のリスクを比較せよ。

**解答例を表示**
    
    
    import time
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    s = create_dummy_eels_si(size=64, energy_range=(400, 1000))
    data_reshaped = s.data.reshape(-1, s.data.shape[2])
    
    labels_true = np.zeros(64*64, dtype=int)
    labels_true[32*64:] = 1
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # SVM
    start = time.time()
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    scores_svm = cross_val_score(svm, data_scaled, labels_true, cv=5)
    time_svm = time.time() - start
    
    # Random Forest
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_rf = cross_val_score(rf, data_scaled, labels_true, cv=5)
    time_rf = time.time() - start
    
    print(f"SVM 精度: {scores_svm.mean():.3f} ± {scores_svm.std():.3f}, 時間: {time_svm:.2f}秒")
    print(f"Random Forest 精度: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}, 時間: {time_rf:.2f}秒")
    print("\n比較:")
    print("  SVM: 高精度、中〜高計算コスト、過学習リスク中")
    print("  Random Forest: 高精度、並列化可能、過学習に頑健、特徴量重要度が得られる")
    

### 演習5-7: KAMの物理的意味（易）

**問題** ：KAM値が高い領域は何を示唆するか、転位密度との関係を含めて説明せよ。

**解答例を表示**

**高KAM領域の物理的意味** ：

  * 局所的な格子回転が大きい → 転位が蓄積している
  * KAM > 2°：冷間加工、塑性変形、粒界近傍
  * KAM < 1°：再結晶粒、回復組織、低歪領域
  * **転位密度との関係** ：KAMは格子曲率を反映し、曲率は幾何学的必要転位（GND）密度に比例する。したがって、高KAM = 高GND密度 = 高転位密度

### 演習5-8: 統合解析ワークフロー（難）

**問題** ：Fe-Cr合金のSTEM-EELSとEBSDデータを統合解析する計画を立案せよ。手順、期待される結果、技術的課題を含めること。

**解答例を表示**

**統合解析ワークフロー** ：

  1. **STEM-EELS取得** ： 
     * 同一試料でSTEM-HAADFとEELSマッピング
     * Fe-L2,3、Cr-L2,3エッジのマッピング（空間分解能<10 nm）
     * HyperSpyでPCA/ICA分析 → 相分離
  2. **EBSD取得** ： 
     * SEMでEBSDマッピング（同一領域、ステップサイズ<0.5 μm）
     * orixで方位解析、KAM、GND密度計算
  3. **空間相関解析** ： 
     * 画像レジストレーション（SIFT/ORBで特徴点マッチング）
     * 元素濃度とKAMの空間的相関を評価
     * Cr濃化領域と高KAM領域の重なりを統計的に検証
  4. **機械学習統合** ： 
     * EELSスペクトル（特徴量：Fe/Cr比）とEBSD方位（特徴量：KAM、GND）を統合特徴ベクトルに
     * Random Forestで相分類（α相、σ相、χ相など）
     * 相分類マップと転位構造の関係を解明

**技術的課題** ：

  * **空間分解能のミスマッチ** ：STEM（<10 nm）とEBSD（>100 nm）の分解能差 → 対策：多重スケール解析、階層的モデリング
  * **試料ダメージ** ：EELS測定でビーム損傷 → 対策：低ドーズ測定、クライオ冷却
  * **データレジストレーション** ：異なる装置での位置合わせ → 対策：金マーカー、自動特徴点抽出

### 演習5-9: GND密度の妥当性検証（難）

**問題** ：計算したGND密度が物理的に妥当か検証する方法を3つ提案せよ。

**解答例を表示**

**GND密度の検証方法** ：

  1. **文献値との比較** ： 
     * 同一材料・同一加工条件のTEM直接観察によるρ測定値と比較
     * 典型的範囲：焼鈍材 10^12-10^13 m^-2、冷間加工材 10^14-10^15 m^-2
  2. **硬度との相関** ： 
     * Taylor強化式：Δσ ∝ √ρ（転位密度と強度の関係）
     * ナノインデンテーション硬度とGND密度の空間分布を比較
     * 高GND領域で硬度が高ければ妥当
  3. **TEM観察との整合性** ： 
     * 同一試料をTEMで観察し、転位密度を直接カウント
     * EBSDから推定したGND密度と比較（通常EBSDはGNDのみ検出、統計的転位は検出しない）
     * EBSD値がTEM値より1桁程度低ければ妥当

### 演習5-10: バッチ処理スクリプト作成（難）

**問題** ：複数のEELSファイル（100個）を自動でバックグラウンド除去し、Fe-L積分強度マップを生成するPythonスクリプトを設計せよ。

**解答例を表示**
    
    
    import hyperspy.api as hs
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    def batch_process_eels(file_list, output_dir, edge_onset=708,
                            fit_range=(650, 700), integration_window=(708, 758)):
        """
        複数EELSファイルをバッチ処理
    
        Parameters
        ----------
        file_list : list
            EELSファイルパスのリスト
        output_dir : Path
            出力ディレクトリ
        edge_onset : float
            エッジ開始エネルギー [eV]
        fit_range : tuple
            バックグラウンドフィット範囲 [eV]
        integration_window : tuple
            積分窓 [eV]
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
        for i, file_path in enumerate(file_list):
            print(f"Processing {i+1}/{len(file_list)}: {file_path.name}")
    
            try:
                # データ読込
                s = hs.load(file_path)
    
                # バックグラウンド除去（全ピクセル）
                s_bg_removed = s.remove_background(
                    signal_range=fit_range,
                    background_type='PowerLaw',
                    fast=True  # 高速モード
                )
    
                # 積分強度マップ生成
                energy_axis = s_bg_removed.axes_manager[-1].axis
                mask = (energy_axis >= integration_window[0]) & (energy_axis <= integration_window[1])
    
                # 各ピクセルで積分
                intensity_map = np.trapz(s_bg_removed.isig[mask].data,
                                          energy_axis[mask], axis=-1)
    
                # マップを保存
                output_file = output_dir / f"{file_path.stem}_FeL_map.npy"
                np.save(output_file, intensity_map)
    
                # PNG可視化も保存
                plt.figure(figsize=(8, 6))
                plt.imshow(intensity_map, cmap='hot')
                plt.colorbar(label='Integrated Intensity [counts]')
                plt.title(f'Fe-L Intensity Map: {file_path.stem}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_dir / f"{file_path.stem}_FeL_map.png", dpi=150)
                plt.close()
    
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
        print(f"\nBatch processing complete. Results saved to {output_dir}")
    
    # 使用例
    # file_list = list(Path('eels_data/').glob('*.hspy'))
    # batch_process_eels(file_list, output_dir='processed_maps/')
    

## 5.6 学習チェック

以下の質問に答えて、理解度を確認しましょう：

  1. HyperSpyのSignal1Dオブジェクトの構造（data, axes_manager, metadata）を理解していますか？
  2. EELSバックグラウンド除去でべき乗則フィットを使う理由を説明できますか？
  3. PCAとICAの違いと、それぞれの利点・欠点を理解していますか？
  4. k-meansとGMMのクラスタリング原理の違いを説明できますか？
  5. 教師あり学習と教師なし学習の使い分けを判断できますか？
  6. KAM（局所方位差）が高い領域の物理的意味を理解していますか？
  7. GND（幾何学的必要転位）密度の計算原理（Nye tensor）を理解していますか？
  8. EELS、EDS、EBSDデータを統合解析するワークフローを設計できますか？
  9. 大規模データセットに対するバッチ処理スクリプトを実装できますか？
  10. 機械学習モデルの精度評価（混同行列、シルエットスコア）ができますか？
  11. EELSスペクトルの主成分分析で得られた成分を物理的に解釈できますか？
  12. オイラー角と四元数の違いを理解し、方位データを適切に扱えますか？
  13. KAMマップとGND密度マップの相関を定量的に評価できますか？
  14. SVMのカーネル（RBF, linear, poly）の選択基準を理解していますか？
  15. クロスバリデーションで過学習を検出し、対策を講じることができますか？

## 5.7 参考文献

  1. de la Peña, F., et al. (2023). _HyperSpy: Multidimensional Data Analysis Toolbox_. <https://hyperspy.org> \- HyperSpy公式ドキュメント
  2. Johnstone, D. N., et al. (2020). "pyxem: Multidimensional diffraction microscopy in Python." _Journal of Open Source Software_ , 5(51), 2496. - 回折データ解析ライブラリ
  3. Langenhorst, M., et al. (2023). "orix: A Python library for crystallographic orientation data analysis." _Journal of Open Research Software_. - EBSD方位解析ライブラリ
  4. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12, 2825-2830. - scikit-learn公式論文
  5. Britton, T. B., et al. (2016). "Tutorial: Crystal orientations and EBSD—Or which way is up?" _Materials Characterization_ , 117, 113-126. - EBSD解析の教育的総説
  6. Pantelic, R. S., et al. (2010). "The discriminative bilateral filter: An enhanced denoising filter for electron microscopy." _Journal of Structural Biology_ , 171(3), 289-295. - ノイズ除去手法
  7. Shiga, M., et al. (2016). "Sparse modeling of EELS and EDX spectral imaging data by nonnegative matrix factorization." _Ultramicroscopy_ , 170, 43-59. - スペクトル分解の機械学習応用

## 5.8 まとめと次のステップ

本章では、電子顕微鏡分析の統合的Pythonワークフローを学びました。HyperSpyによるスペクトル処理、PCA/ICAによる次元削減、機械学習による相分類、EBSD方位解析の実践的手法を習得しました。

**実践への応用** ：

  * 実データでHyperSpyワークフローを構築し、バッチ処理を自動化
  * PCA/ICAで得られた成分を物理モデル（相図、状態図）と対応付け
  * 機械学習モデルを継続的に改善（新データで再学習）
  * EBSDとSTEMデータを空間的に統合し、多重スケール解析を実施

**発展的学習** ：

  * 深層学習（CNN, U-Net）による画像セグメンテーション
  * グラフニューラルネットワークによる結晶粒ネットワーク解析
  * 4D-STEMデータのリアルタイム解析（pyxem利用）
  * トモグラフィー3D再構成とFEMシミュレーションの統合
