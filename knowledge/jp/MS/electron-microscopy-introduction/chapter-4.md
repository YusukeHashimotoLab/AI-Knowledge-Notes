---
title: 第4章：STEMと分析技術
chapter_title: 第4章：STEMと分析技術
subtitle: Z-contrast像、EELS、元素マッピング、原子分解能分析
reading_time: 25-35分
difficulty: 中級〜上級
code_examples: 7
---

走査透過型電子顕微鏡（STEM）は、収束電子ビームを試料上で走査し、透過電子や非弾性散乱電子を検出して像を形成します。この章では、STEM原理、Z-contrast像（ADF/HAADF）、環状明視野像（ABF）、電子エネルギー損失分光（EELS）、元素マッピング、原子分解能分析、トモグラフィーの基礎と応用を学び、Pythonで定量解析を実践します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ STEM結像原理と従来TEMとの違いを理解する
  * ✅ Z-contrast像（ADF/HAADF-STEM）の形成機構と原子番号依存性を説明できる
  * ✅ 環状明視野（ABF）像の原理と軽元素観察への応用を理解する
  * ✅ 電子エネルギー損失分光（EELS）の原理と定量分析手法を習得する
  * ✅ スペクトル解析（コアロス解析、プラズモン解析）ができる
  * ✅ STEM元素マッピング（EELS/EDS）の取得と解釈ができる
  * ✅ 電子トモグラフィーの原理と3D再構成の基礎を理解する

## 4.1 STEM原理と検出器構成

### 4.1.1 STEMとTEMの違い

STEM（Scanning Transmission Electron Microscopy）は、収束電子ビームを試料上で走査（スキャン）し、各点で透過・散乱電子を検出して像を形成します。

項目 | TEM（透過型） | STEM（走査透過型）  
---|---|---  
**照射方式** | 平行ビーム照射（試料全面） | 収束ビーム走査（点ごと）  
**像形成** | レンズによる結像（像面） | 検出器信号の走査同期（マッピング）  
**分解能** | 対物レンズの収差で決まる | プローブサイズで決まる  
**検出器** | CCDカメラ、蛍光スクリーン | 環状検出器（ADF, HAADF, ABF）、EELS、EDS  
**同時信号取得** | 困難（像or回折） | 容易（複数検出器並列使用）  
**Z-contrast** | 直接的には困難 | HAADF検出器で容易に実現  
  
### 4.1.2 STEM検出器の種類

STEMでは、試料透過後の電子を角度別に検出する**環状検出器** が重要な役割を果たします。
    
    
    ```mermaid
    flowchart TD
        A[収束電子ビーム] --> B[試料]
        B --> C{散乱角度}
        C -->|0 mrad付近| D[BF検出器Bright Field]
        C -->|10-50 mrad| E[ABF検出器Annular BF]
        C -->|30-100 mrad| F[ADF検出器Annular DF]
        C -->|>50 mrad| G[HAADF検出器High-Angle ADF]
        C -->|全角度| H[EELS分光器]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#ffeb99,stroke:#ffa500
        style G fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style H fill:#99ccff,stroke:#0066cc
    ```

**各検出器の特徴** ：

  * **BF（Bright Field）** ：低角散乱電子を検出。従来TEMの明視野像に相当。位相コントラストが主
  * **ABF（Annular Bright Field）** ：中心部を遮蔽した環状明視野。軽元素（Li, H, Oなど）の原子位置を直接観察可能
  * **ADF（Annular Dark Field）** ：中角散乱電子を検出。質量厚さコントラスト
  * **HAADF（High-Angle ADF）** ：高角散乱電子を検出。**Z-contrast** （原子番号依存コントラスト）が得られる。非干渉性結像
  * **EELS（Electron Energy Loss Spectroscopy）** ：エネルギー損失した電子を分光。元素同定、化学状態、バンドギャップ測定

### 4.1.3 プローブサイズと電流のトレードオフ

STEMの分解能は**プローブサイズ** で決まりますが、プローブを小さくすると電流密度が減少し、S/N比が低下します。

$$ d_{\text{probe}} \approx 0.6 \frac{\lambda}{\alpha} $$ 

  * $d_{\text{probe}}$：プローブ直径
  * $\lambda$：電子波長
  * $\alpha$：収束半角（対物絞りで制御）

**実用的なトレードオフ** ：

  * 原子分解能（<1 Å）：収束角 20-30 mrad、プローブ電流 50-200 pA
  * 高S/N分析（~2 Å）：収束角 10-15 mrad、プローブ電流 >500 pA

#### コード例4-1: STEM検出器の散乱角度と信号強度のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rutherford_scattering_cross_section(Z, theta, E_keV=200):
        """
        ラザフォード散乱断面積（簡略モデル）
    
        Parameters
        ----------
        Z : int
            原子番号
        theta : array-like
            散乱角 [rad]
        E_keV : float
            電子エネルギー [keV]
    
        Returns
        -------
        sigma : ndarray
            微分散乱断面積 [任意単位]
        """
        # 簡略化：高角散乱はZ^2に比例、低角は位相コントラスト
        # Rutherford散乱：dσ/dΩ ∝ Z^2 / (sin^4(θ/2))
    
        # 低角領域での発散を避けるため最小角度を設定
        theta = np.maximum(theta, 0.001)
    
        sigma = (Z**2) / (np.sin(theta / 2 + 1e-6)**4)
    
        return sigma
    
    def plot_stem_detector_signals():
        """
        STEM検出器の散乱角度依存性をプロット
        """
        # 散乱角（mrad）
        theta_mrad = np.linspace(0.1, 200, 1000)
        theta_rad = theta_mrad * 1e-3
    
        # 異なる原子番号での散乱強度
        elements = [('C', 6), ('Al', 13), ('Fe', 26), ('Au', 79)]
        colors = ['green', 'blue', 'orange', 'red']
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
        # 上図：散乱強度の角度依存性（対数スケール）
        for (elem, Z), color in zip(elements, colors):
            sigma = rutherford_scattering_cross_section(Z, theta_rad)
            ax1.plot(theta_mrad, sigma, color=color, linewidth=2, label=f'{elem} (Z={Z})')
    
        # 検出器範囲を示す帯
        ax1.axvspan(0, 10, alpha=0.2, color='cyan', label='BF (0-10 mrad)')
        ax1.axvspan(10, 50, alpha=0.2, color='lightgreen', label='ABF (10-50 mrad)')
        ax1.axvspan(50, 100, alpha=0.2, color='yellow', label='ADF (50-100 mrad)')
        ax1.axvspan(100, 200, alpha=0.2, color='pink', label='HAADF (>100 mrad)')
    
        ax1.set_xlabel('Scattering Angle [mrad]', fontsize=12)
        ax1.set_ylabel('Scattering Intensity [a.u.]', fontsize=12)
        ax1.set_title('STEM Detector Ranges and Scattering Intensity\n(Z-dependence)', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 200)
    
        # 下図：各検出器での積分強度（Z依存性）
        Z_range = np.arange(1, 80)
        detector_ranges = {
            'BF': (0, 10),
            'ABF': (10, 50),
            'ADF': (50, 100),
            'HAADF': (100, 200)
        }
    
        for det_name, (theta_min, theta_max) in detector_ranges.items():
            intensities = []
            for Z in Z_range:
                theta_range = np.linspace(theta_min*1e-3, theta_max*1e-3, 100)
                sigma = rutherford_scattering_cross_section(Z, theta_range)
                # 積分（台形則）
                intensity = np.trapz(sigma, theta_range)
                intensities.append(intensity)
    
            ax2.plot(Z_range, intensities, linewidth=2, marker='o', markersize=3, label=det_name)
    
        ax2.set_xlabel('Atomic Number Z', fontsize=12)
        ax2.set_ylabel('Integrated Signal Intensity [a.u.]', fontsize=12)
        ax2.set_title('Z-Dependence of STEM Detector Signals', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(1, 79)
    
        plt.tight_layout()
        plt.show()
    
        print("HAADF信号はZ^2に近い依存性（Z-contrast）")
        print("低角度検出器（BF, ABF）は位相コントラスト成分も含む")
    
    # 実行
    plot_stem_detector_signals()
    

**観察ポイント** ：

  * HAADF信号は高角散乱を検出するため、重元素（高Z）で強い信号が得られる
  * Z-contrastの起源は、ラザフォード散乱断面積の $Z^2$ 依存性
  * ABFは中角領域で軽元素と重元素の散乱強度差が相対的に小さく、軽元素観察に有利

## 4.2 Z-contrast像（HAADF-STEM）

### 4.2.1 Z-contrast像の原理

HAADF-STEM像は、高角散乱電子（通常>50 mrad）を環状検出器で検出します。この散乱は主に**熱拡散散乱（TDS: Thermal Diffuse Scattering）** と非弾性散乱で、以下の特徴を持ちます：

  * **非干渉性結像** ：位相コントラストがなく、像の解釈が直感的
  * **Z^2依存性** ：散乱強度が原子番号の2乗に近い依存性を持つ（Z-contrast）
  * **厚さ依存性** ：試料厚さに比例して信号が増加
  * **デフォーカス非依存** ：CTFの影響を受けない（収差補正なしでも解釈が容易）

HAADF信号強度は次式で近似されます：

$$ I_{\text{HAADF}} \propto Z^{1.7-2.0} \cdot t $$ 

ここで、$Z$ は原子番号、$t$ は試料厚さです。

### 4.2.2 原子分解能Z-contrast像

収差補正STEMにより、プローブサイズを1 Å以下にすることで、**原子カラム** （原子列）を直接観察できます。

**応用例** ：

  * 界面の原子配列解析（半導体ヘテロ構造、金属/セラミックス接合）
  * ドーパント原子の位置同定（半導体デバイス）
  * ナノ粒子の表面原子構造（触媒）
  * 結晶粒界の原子レベル解析

#### コード例4-2: Z-contrast像のシミュレーション（原子カラム強度）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    def simulate_haadf_image(lattice_a=4.0, Z_matrix=13, Z_dopant=79, dopant_positions=None, size=256, pixel_size=0.1):
        """
        HAADF-STEM像をシミュレート（簡略モデル）
    
        Parameters
        ----------
        lattice_a : float
            格子定数 [Å]
        Z_matrix : int
            母相の原子番号
        Z_dopant : int
            ドーパントの原子番号
        dopant_positions : list of tuples
            ドーパント位置 [(x, y), ...]（格子単位）
        size : int
            画像サイズ [pixels]
        pixel_size : float
            ピクセルサイズ [Å/pixel]
    
        Returns
        -------
        image : ndarray
            HAADF像
        """
        image = np.zeros((size, size))
    
        # 格子点を生成（正方格子）
        grid_points_per_side = int(size * pixel_size / lattice_a)
    
        for i in range(grid_points_per_side):
            for j in range(grid_points_per_side):
                x_grid = i * lattice_a / pixel_size
                y_grid = j * lattice_a / pixel_size
    
                # 原子カラムの位置（ピクセル単位）
                x_px = int(x_grid)
                y_px = int(y_grid)
    
                if x_px < size and y_px < size:
                    # 原子番号の決定（ドーパントか母相か）
                    Z = Z_matrix
                    if dopant_positions is not None:
                        for (dx, dy) in dopant_positions:
                            if abs(i - dx) < 0.5 and abs(j - dy) < 0.5:
                                Z = Z_dopant
                                break
    
                    # HAADF信号強度 ∝ Z^1.7
                    intensity = Z**1.7
    
                    # プローブ形状（ガウシアン）で原子カラムを表現
                    probe_sigma = 0.5  # [pixels]（プローブサイズ～1 Å）
    
                    # ガウシアンを画像に追加
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            px = x_px + dx
                            py = y_px + dy
                            if 0 <= px < size and 0 <= py < size:
                                r = np.sqrt(dx**2 + dy**2)
                                image[py, px] += intensity * np.exp(-r**2 / (2 * probe_sigma**2))
    
        # ノイズ追加
        image += np.random.poisson(lam=10, size=image.shape)
    
        # 正規化
        image = (image - image.min()) / (image.max() - image.min())
    
        return image
    
    # シミュレーション実行
    # Al母相（Z=13）にAuドーパント（Z=79）を配置
    dopant_positions = [(10, 10), (15, 15), (20, 12), (12, 20)]
    
    image = simulate_haadf_image(lattice_a=4.0, Z_matrix=13, Z_dopant=79,
                                  dopant_positions=dopant_positions, size=256, pixel_size=0.2)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # HAADF像（全体）
    im1 = ax1.imshow(image, cmap='gray', extent=[0, 256*0.2, 0, 256*0.2])
    ax1.set_title('HAADF-STEM Image\n(Al matrix + Au dopants)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x [Å]', fontsize=11)
    ax1.set_ylabel('y [Å]', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Intensity [a.u.]', fontsize=10)
    
    # 拡大図（ドーパント周辺）
    zoom_center = (10, 10)
    zoom_size = 30
    zoomed = image[zoom_center[1]*5-zoom_size:zoom_center[1]*5+zoom_size,
                   zoom_center[0]*5-zoom_size:zoom_center[0]*5+zoom_size]
    
    im2 = ax2.imshow(zoomed, cmap='gray', extent=[0, zoom_size*2*0.2, 0, zoom_size*2*0.2])
    ax2.set_title('Zoomed View\n(Bright spot = Au dopant)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x [Å]', fontsize=11)
    ax2.set_ylabel('y [Å]', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Intensity [a.u.]', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Z-contrast: Au (Z=79) の原子カラムがAl (Z=13) より明るく見える")
    print("強度比: I(Au)/I(Al) ≈ (79/13)^1.7 ≈ 18倍")
    

## 4.3 環状明視野（ABF）像と軽元素観察

### 4.3.1 ABF像の原理

環状明視野（Annular Bright Field, ABF）像は、中心部を遮蔽した環状検出器で低〜中角散乱電子（10-50 mrad）を検出します。

**ABFの特徴** ：

  * 軽元素（Li, H, O, N）の原子カラムが**暗いスポット** として観察される
  * HAA DFでは見えない軽元素を、重元素と同時に観察可能
  * 位相コントラストと非干渉性コントラストの混合
  * デフォーカス依存性が比較的小さい

**応用例** ：

  * 酸化物中の酸素原子位置の決定（ペロブスカイト、スピネル構造）
  * リチウムイオン電池材料のLi分布
  * 窒化物半導体のN原子観察
  * 水素貯蔵材料のH原子配置

#### コード例4-3: ABF像とHAADF像の同時シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_abf_haadf_comparison():
        """
        ABF像とHAADF像の同時観察シミュレーション
        ペロブスカイト構造（重金属 + 酸素）を想定
        """
        size = 128
        pixel_size = 0.1  # [Å/pixel]
        lattice_a = 4.0  # [Å]
    
        image_haadf = np.zeros((size, size))
        image_abf = np.ones((size, size)) * 0.5  # ABFはバックグラウンドが中間値
    
        # ペロブスカイト構造の簡略モデル
        # 重金属（Sr, Ba, Pb など Z~80）：コーナーとボディセンター
        # 酸素（Z=8）：face-centered
    
        grid_size = int(size * pixel_size / lattice_a)
    
        for i in range(grid_size):
            for j in range(grid_size):
                # 重金属サイト（コーナー）
                x_heavy = i * lattice_a / pixel_size
                y_heavy = j * lattice_a / pixel_size
    
                # 酸素サイト（face-centered）
                oxygen_sites = [
                    (x_heavy + lattice_a/(2*pixel_size), y_heavy),
                    (x_heavy, y_heavy + lattice_a/(2*pixel_size))
                ]
    
                # 重金属をプロット（HAADF: 明るい、ABF: 暗い）
                add_atom_column(image_haadf, x_heavy, y_heavy, Z=80, detector='HAADF')
                add_atom_column(image_abf, x_heavy, y_heavy, Z=80, detector='ABF')
    
                # 酸素をプロット（HAADF: ほぼ見えない、ABF: 暗いスポット）
                for (ox, oy) in oxygen_sites:
                    if ox < size and oy < size:
                        add_atom_column(image_haadf, ox, oy, Z=8, detector='HAADF')
                        add_atom_column(image_abf, ox, oy, Z=8, detector='ABF')
    
        # ノイズ追加
        image_haadf += np.random.normal(0, 0.02, image_haadf.shape)
        image_abf += np.random.normal(0, 0.02, image_abf.shape)
    
        # 正規化
        image_haadf = np.clip(image_haadf, 0, 1)
        image_abf = np.clip(image_abf, 0, 1)
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
        # HAADF像
        im0 = axes[0].imshow(image_haadf, cmap='gray', extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[0].set_title('HAADF-STEM Image\n(Heavy atoms bright)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('x [Å]', fontsize=11)
        axes[0].set_ylabel('y [Å]', fontsize=11)
    
        # ABF像
        im1 = axes[1].imshow(image_abf, cmap='gray', extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[1].set_title('ABF-STEM Image\n(Both heavy and light atoms dark)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('x [Å]', fontsize=11)
        axes[1].set_ylabel('y [Å]', fontsize=11)
    
        # オーバーレイ（カラー合成）
        # HAADF（赤）+ ABF反転（緑）
        overlay = np.zeros((size, size, 3))
        overlay[:, :, 0] = image_haadf  # Red: HAADF
        overlay[:, :, 1] = 1 - image_abf  # Green: ABF inverted
        overlay[:, :, 2] = 0
    
        im2 = axes[2].imshow(overlay, extent=[0, size*pixel_size, 0, size*pixel_size])
        axes[2].set_title('Overlay: HAADF (Red) + ABF (Green)\n(Yellow = both present)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('x [Å]', fontsize=11)
        axes[2].set_ylabel('y [Å]', fontsize=11)
    
        plt.tight_layout()
        plt.show()
    
        print("HAADF: 重元素のみ明るく見える")
        print("ABF: 重元素も軽元素（酸素）も暗いスポットとして観察される")
    
    def add_atom_column(image, x, y, Z, detector='HAADF'):
        """
        原子カラムを画像に追加
    
        Parameters
        ----------
        image : ndarray
            画像配列
        x, y : float
            原子位置 [pixels]
        Z : int
            原子番号
        detector : str
            'HAADF' or 'ABF'
        """
        size = image.shape[0]
        x_int = int(x)
        y_int = int(y)
    
        if detector == 'HAADF':
            # HAADF: Z^1.7に比例
            intensity = (Z / 80.0)**1.7 * 0.8
            sign = +1
        else:  # ABF
            # ABF: 原子位置で暗くなる（負のコントラスト）
            intensity = (Z / 80.0)**0.5 * 0.3
            sign = -1
    
        probe_sigma = 0.5
    
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px = x_int + dx
                py = y_int + dy
                if 0 <= px < size and 0 <= py < size:
                    r = np.sqrt((px - x)**2 + (py - y)**2)
                    image[py, px] += sign * intensity * np.exp(-r**2 / (2 * probe_sigma**2))
    
    # 実行
    simulate_abf_haadf_comparison()
    

## 4.4 電子エネルギー損失分光（EELS）

### 4.4.1 EELS原理

電子エネルギー損失分光（Electron Energy Loss Spectroscopy, EELS）は、試料を透過した電子のエネルギー損失を測定し、元素組成、化学結合状態、バンドギャップを解析する手法です。

**EELSスペクトルの構成** ：

  * **ゼロロスピーク** （0 eV）：弾性散乱電子（エネルギー損失なし）
  * **低損失領域** （0-50 eV）：プラズモン励起、バンド間遷移
  * **コアロス領域** （>50 eV）：内殻電子励起。元素固有のエッジ（K, L, M殻）

### 4.4.2 コアロス解析と元素定量

コアロスエッジの積分強度から元素濃度を定量できます：

$$ \frac{N_A}{N_B} = \frac{I_A(\Delta, \beta)}{I_B(\Delta, \beta)} \cdot \frac{\sigma_B(\Delta, \beta)}{\sigma_A(\Delta, \beta)} $$ 

  * $N_A, N_B$：元素AとBの原子数密度
  * $I_A, I_B$：エッジ積分強度
  * $\sigma_A, \sigma_B$：部分イオン化断面積（理論値またはHartree-Slater計算）
  * $\Delta$：積分窓幅、$\beta$：コレクション半角

**EELS定量解析の手順** ：

  1. **バックグラウンド除去** ：エッジ前領域をべき乗則でフィット（$AE^{-r}$）
  2. **積分強度計算** ：エッジから一定エネルギー範囲（通常50-100 eV）を積分
  3. **断面積補正** ：HyperSpyなどのライブラリで自動計算
  4. **濃度算出** ：上式で元素比を計算

#### コード例4-4: EELSスペクトルのシミュレーションと定量解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    def simulate_eels_spectrum(elements=['C', 'O', 'Fe'], concentrations=[0.3, 0.5, 0.2],
                                energy_range=(200, 800), noise_level=0.05):
        """
        EELSスペクトルをシミュレート
    
        Parameters
        ----------
        elements : list
            元素リスト
        concentrations : list
            各元素の相対濃度
        energy_range : tuple
            エネルギー範囲 [eV]
        noise_level : float
            ノイズレベル
    
        Returns
        -------
        energy : ndarray
            エネルギー軸 [eV]
        spectrum : ndarray
            強度
        """
        energy = np.linspace(energy_range[0], energy_range[1], 1000)
    
        # 典型的なコアロスエッジのエネルギー（簡略版）
        edge_energies = {
            'C': 284,   # C-K
            'N': 401,   # N-K
            'O': 532,   # O-K
            'Fe': 708,  # Fe-L3
            'Al': 1560, # Al-K
            'Si': 1839  # Si-K
        }
    
        # バックグラウンド（べき乗則）
        background = 1000 * (energy / energy[0])**(-3)
    
        spectrum = background.copy()
    
        # 各元素のエッジを追加
        for elem, conc in zip(elements, concentrations):
            if elem in edge_energies:
                edge_E = edge_energies[elem]
                if energy_range[0] < edge_E < energy_range[1]:
                    # エッジジャンプ（ステップ関数 + 減衰）
                    edge_mask = energy >= edge_E
                    edge_intensity = conc * 300 * np.exp(-(energy[edge_mask] - edge_E) / 100)
                    spectrum[edge_mask] += edge_intensity
    
        # ノイズ追加（ポアソンノイズ）
        spectrum += np.random.poisson(lam=noise_level*spectrum.mean(), size=spectrum.shape)
    
        return energy, spectrum
    
    def quantify_eels_edges(energy, spectrum, edges_dict):
        """
        EELSスペクトルから元素を定量
    
        Parameters
        ----------
        energy : ndarray
            エネルギー軸 [eV]
        spectrum : ndarray
            強度
        edges_dict : dict
            {'Element': edge_energy}
    
        Returns
        -------
        results : dict
            {'Element': integrated_intensity}
        """
        results = {}
    
        for elem, edge_E in edges_dict.items():
            # エッジ前領域（バックグラウンドフィット）
            pre_edge_mask = (energy >= edge_E - 50) & (energy < edge_E)
            if np.sum(pre_edge_mask) < 10:
                continue
    
            # べき乗則フィット A * E^(-r)
            E_pre = energy[pre_edge_mask]
            I_pre = spectrum[pre_edge_mask]
    
            # 対数変換して線形フィット
            log_E = np.log(E_pre)
            log_I = np.log(I_pre + 1)  # ゼロ除算回避
    
            coeffs = np.polyfit(log_E, log_I, 1)
            r = -coeffs[0]
            A = np.exp(coeffs[1])
    
            # バックグラウンド除去
            background = A * energy**(-r)
            spectrum_bg_removed = spectrum - background
            spectrum_bg_removed = np.maximum(spectrum_bg_removed, 0)
    
            # エッジ後領域の積分（50 eV窓）
            post_edge_mask = (energy >= edge_E) & (energy < edge_E + 50)
            integrated_intensity = np.trapz(spectrum_bg_removed[post_edge_mask],
                                            energy[post_edge_mask])
    
            results[elem] = integrated_intensity
    
        return results
    
    # シミュレーション実行
    elements = ['C', 'O', 'Fe']
    concentrations = [0.3, 0.5, 0.2]
    
    energy, spectrum = simulate_eels_spectrum(elements, concentrations,
                                              energy_range=(200, 800), noise_level=0.05)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上図：生スペクトル
    ax1.plot(energy, spectrum, 'b-', linewidth=1.5, label='Measured Spectrum')
    
    # エッジ位置を示す
    edge_energies = {'C': 284, 'O': 532, 'Fe': 708}
    colors_edge = {'C': 'green', 'O': 'red', 'Fe': 'orange'}
    
    for elem, edge_E in edge_energies.items():
        if 200 < edge_E < 800:
            ax1.axvline(edge_E, color=colors_edge[elem], linestyle='--', linewidth=2, alpha=0.7,
                       label=f'{elem}-K edge ({edge_E} eV)')
    
    ax1.set_xlabel('Energy Loss [eV]', fontsize=12)
    ax1.set_ylabel('Intensity [counts]', fontsize=12)
    ax1.set_title('Simulated EELS Spectrum (C, O, Fe)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 下図：定量解析結果
    quantified = quantify_eels_edges(energy, spectrum, edge_energies)
    
    elem_list = list(quantified.keys())
    intensities = list(quantified.values())
    
    # 相対濃度に正規化
    total = sum(intensities)
    relative_conc = [I / total for I in intensities]
    
    x_pos = np.arange(len(elem_list))
    bars = ax2.bar(x_pos, relative_conc, color=['green', 'red', 'orange'], alpha=0.7, edgecolor='black')
    
    # 真の濃度（入力値）をプロット
    true_conc = concentrations
    ax2.scatter(x_pos, true_conc, color='blue', s=150, marker='D', label='True Concentration', zorder=5)
    
    ax2.set_xlabel('Element', fontsize=12)
    ax2.set_ylabel('Relative Concentration', fontsize=12)
    ax2.set_title('EELS Quantification Results', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(elem_list, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n定量結果:")
    for elem, conc in zip(elem_list, relative_conc):
        print(f"  {elem}: {conc:.2f}")
    

### 4.4.3 ELNES（エネルギー損失近端構造）

エッジ直後の微細構造（Energy Loss Near Edge Structure, ELNES）は、化学結合状態や局所原子配列を反映します。

**ELNES解析の応用** ：

  * 酸化数の決定（Fe2+ vs Fe3+のL3/L2比）
  * 配位数の推定（Oの1s→2p遷移の形状）
  * 結晶場分裂の観察（遷移金属酸化物）
  * バンドギャップの測定（半導体材料）

## 4.5 STEM元素マッピング

### 4.5.1 EELS/EDSマッピングの原理

STEMモードでビームを走査しながら、各点でEELSまたはEDSスペクトルを取得することで、**2D元素分布マップ** を作成できます。

**EELSマッピング** ：

  * **利点** ：空間分解能が高い（プローブサイズ制限）、軽元素に感度が高い、化学状態も同時取得
  * **欠点** ：薄い試料が必要（<100 nm）、データ量が大きい、測定に時間がかかる

**EDSマッピング** ：

  * **利点** ：厚い試料でも測定可能、多元素同時分析、測定が比較的速い
  * **欠点** ：空間分解能がやや低い（X線発生体積）、軽元素の検出が困難

#### コード例4-5: STEM元素マッピングのシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    def simulate_stem_mapping(size=128, num_particles=10):
        """
        STEM元素マッピングをシミュレート（2元系ナノ粒子）
    
        Parameters
        ----------
        size : int
            マップサイズ [pixels]
        num_particles : int
            粒子数
    
        Returns
        -------
        map_Fe : ndarray
            Fe元素マップ
        map_O : ndarray
            O元素マップ
        haadf : ndarray
            HAADF像
        """
        # Fe3O4ナノ粒子を炭素基板上に分散
        map_Fe = np.zeros((size, size))
        map_O = np.zeros((size, size))
        map_C = np.ones((size, size)) * 0.3  # 基板のC
    
        np.random.seed(42)
    
        for _ in range(num_particles):
            # 粒子の位置とサイズ
            x_center = np.random.randint(10, size - 10)
            y_center = np.random.randint(10, size - 10)
            radius = np.random.randint(5, 15)
    
            # 粒子領域の生成
            y, x = np.ogrid[:size, :size]
            mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    
            # Fe3O4の化学量論比（Fe:O = 3:4）
            map_Fe[mask] = 0.75 + np.random.normal(0, 0.05, np.sum(mask))
            map_O[mask] = 1.0 + np.random.normal(0, 0.05, np.sum(mask))
            map_C[mask] = 0  # 粒子内部にはC無し
    
        # プローブサイズによるぼかし
        map_Fe = gaussian_filter(map_Fe, sigma=1.0)
        map_O = gaussian_filter(map_O, sigma=1.0)
        map_C = gaussian_filter(map_C, sigma=1.0)
    
        # HAADF像（Z-contrastの近似）
        haadf = map_Fe * 26**1.7 + map_O * 8**1.7 + map_C * 6**1.7
        haadf = haadf / haadf.max()
    
        # ノイズ追加
        map_Fe += np.random.normal(0, 0.02, map_Fe.shape)
        map_O += np.random.normal(0, 0.02, map_O.shape)
        map_C += np.random.normal(0, 0.02, map_C.shape)
        haadf += np.random.normal(0, 0.02, haadf.shape)
    
        # クリッピング
        map_Fe = np.clip(map_Fe, 0, 1)
        map_O = np.clip(map_O, 0, 1)
        map_C = np.clip(map_C, 0, 1)
        haadf = np.clip(haadf, 0, 1)
    
        return map_Fe, map_O, map_C, haadf
    
    # シミュレーション実行
    map_Fe, map_O, map_C, haadf = simulate_stem_mapping(size=128, num_particles=8)
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # HAADF像
    im0 = axes[0, 0].imshow(haadf, cmap='gray')
    axes[0, 0].set_title('HAADF-STEM Image', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Fe マップ
    im1 = axes[0, 1].imshow(map_Fe, cmap='Reds')
    axes[0, 1].set_title('Fe Elemental Map\n(EELS Fe-L edge)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='Fe Intensity')
    
    # O マップ
    im2 = axes[0, 2].imshow(map_O, cmap='Blues')
    axes[0, 2].set_title('O Elemental Map\n(EELS O-K edge)', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='O Intensity')
    
    # C マップ
    im3 = axes[1, 0].imshow(map_C, cmap='Greens')
    axes[1, 0].set_title('C Elemental Map\n(Substrate)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, label='C Intensity')
    
    # RGB合成（Fe=Red, O=Blue, C=Green）
    rgb_composite = np.stack([map_Fe, map_C, map_O], axis=2)
    rgb_composite = rgb_composite / rgb_composite.max()
    
    im4 = axes[1, 1].imshow(rgb_composite)
    axes[1, 1].set_title('RGB Composite\n(Fe:Red, O:Blue, C:Green)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Fe/O比マップ（化学量論）
    fe_o_ratio = np.divide(map_Fe, map_O + 1e-6)  # ゼロ除算回避
    im5 = axes[1, 2].imshow(fe_o_ratio, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes[1, 2].set_title('Fe/O Ratio Map\n(Stoichiometry)', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')
    cbar5 = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    cbar5.set_label('Fe/O', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Fe3O4ナノ粒子のマッピング:")
    print("  - Fe, O, C の分布が同時に取得される")
    print("  - Fe/O比マップで化学量論的均一性を評価")
    

## 4.6 電子トモグラフィー

### 4.6.1 トモグラフィーの原理

電子トモグラフィー（Electron Tomography）は、試料を様々な角度から撮影し、3次元構造を再構成する手法です。

**基本手順** ：

  1. **傾斜シリーズ取得** ：試料を-70°〜+70°程度傾斜させながら、1-2°刻みで像を取得（80-140枚）
  2. **像アライメント** ：各傾斜角度の像を位置合わせ（金マーカーなどを利用）
  3. **3D再構成** ：投影定理に基づき、逆ラドン変換や反復再構成法で3D像を計算
  4. **セグメンテーション** ：3D像から特定の構造（ナノ粒子、孔など）を抽出
  5. **定量解析** ：体積、表面積、形状、空間分布を計算

**投影定理（Projection Theorem）** ：

3D物体 $f(x, y, z)$ のある方向からの投影 $P_\theta(x', y')$ のフーリエ変換は、3Dフーリエスペースのその方向の断面に対応します。

#### コード例4-6: 電子トモグラフィーの投影と再構成シミュレーション（2D）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import radon, iradon
    from scipy.ndimage import rotate
    
    def create_test_object_2d(size=128):
        """
        2Dテストオブジェクト（ナノ粒子）を生成
    
        Parameters
        ----------
        size : int
            画像サイズ [pixels]
    
        Returns
        -------
        obj : ndarray
            テストオブジェクト
        """
        obj = np.zeros((size, size))
    
        # 3つの円形粒子
        particles = [
            (40, 40, 15),
            (80, 70, 20),
            (60, 90, 12)
        ]
    
        y, x = np.ogrid[:size, :size]
    
        for (cx, cy, r) in particles:
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            obj[mask] = 1.0
    
        return obj
    
    def simulate_tilt_series(obj, angles):
        """
        傾斜シリーズをシミュレート（ラドン変換）
    
        Parameters
        ----------
        obj : ndarray
            2Dオブジェクト
        angles : array-like
            投影角度 [degrees]
    
        Returns
        -------
        sinogram : ndarray
            サイノグラム（全投影データ）
        """
        # ラドン変換（投影）
        sinogram = radon(obj, theta=angles, circle=True)
    
        return sinogram
    
    # テストオブジェクト生成
    obj_original = create_test_object_2d(size=128)
    
    # 傾斜シリーズ取得（-70°〜+70°, 2°刻み）
    angles = np.arange(-70, 71, 2)
    sinogram = simulate_tilt_series(obj_original, angles)
    
    # ノイズ追加（現実的な測定）
    sinogram_noisy = sinogram + np.random.normal(0, 0.05*sinogram.max(), sinogram.shape)
    
    # 3D再構成（逆ラドン変換）
    reconstruction = iradon(sinogram_noisy, theta=angles, circle=True, filter_name='ramp')
    
    # クリッピング
    reconstruction = np.clip(reconstruction, 0, 1)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # オリジナルオブジェクト
    im0 = axes[0, 0].imshow(obj_original, cmap='gray')
    axes[0, 0].set_title('Original Object\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # サイノグラム
    im1 = axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto',
                            extent=[angles.min(), angles.max(), 0, sinogram.shape[0]])
    axes[0, 1].set_title('Sinogram (Tilt Series)\nProjections at Different Angles', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Tilt Angle [degrees]', fontsize=11)
    axes[0, 1].set_ylabel('Projection Position', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # 再構成像
    im2 = axes[1, 0].imshow(reconstruction, cmap='gray')
    axes[1, 0].set_title('Reconstructed Object\n(Filtered Back-Projection)', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # 誤差マップ
    error = np.abs(obj_original - reconstruction)
    im3 = axes[1, 1].imshow(error, cmap='hot')
    axes[1, 1].set_title('Reconstruction Error\n(|Original - Reconstructed|)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    cbar3.set_label('Error', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 定量評価
    mse = np.mean((obj_original - reconstruction)**2)
    print(f"再構成の平均二乗誤差（MSE）: {mse:.4f}")
    print(f"投影角度範囲: {angles.min()}° 〜 {angles.max()}°")
    print(f"投影数: {len(angles)}")
    

**トモグラフィーの応用** ：

  * ナノ粒子触媒の3D形状と表面構造解析
  * 多孔質材料の孔径分布と連結性評価
  * バッテリー電極材料の3D微細構造観察
  * 生物細胞小器官の3D再構成

## 4.7 演習問題

### 演習4-1: Z-contrast強度比

**問題** ：HAADF-STEM像でSi（Z=14）とGe（Z=32）の原子カラム強度比を計算せよ（Z^1.7依存性を仮定）。

**解答例を表示**
    
    
    Z_Si = 14
    Z_Ge = 32
    
    I_ratio = (Z_Ge / Z_Si)**1.7
    
    print(f"Si原子カラム強度: I_Si ∝ {Z_Si}^1.7")
    print(f"Ge原子カラム強度: I_Ge ∝ {Z_Ge}^1.7")
    print(f"強度比 I_Ge/I_Si: {I_ratio:.2f}")
    print("GeはSiの約6倍明るく見える")
    

### 演習4-2: EELS定量

**問題** ：Al2O3のEELSスペクトルで、Al-K積分強度が1200、O-K積分強度が3000だった。部分イオン化断面積比σ(O-K)/σ(Al-K) = 1.5のとき、Al/O原子数比を求めよ。

**解答例を表示**
    
    
    I_Al = 1200
    I_O = 3000
    sigma_ratio = 1.5  # σ(O) / σ(Al)
    
    # N_Al / N_O = (I_Al / I_O) * (σ_O / σ_Al)
    atom_ratio = (I_Al / I_O) * sigma_ratio
    
    print(f"Al積分強度: {I_Al}")
    print(f"O積分強度: {I_O}")
    print(f"断面積比 σ(O)/σ(Al): {sigma_ratio}")
    print(f"原子数比 N_Al/N_O: {atom_ratio:.3f}")
    print(f"\n理論値（Al2O3）: 2/3 = 0.667")
    print(f"測定値: {atom_ratio:.3f} → ほぼ一致")
    

### 演習4-3: ABF像の応用

**問題** ：SrTiO3ペロブスカイト構造で、O原子位置を決定するためにABF像が有効な理由を説明せよ。

**解答例を表示**

**理由** ：

  * HAADF像では、Sr（Z=38）とTi（Z=22）は明るく見えるが、O（Z=8）は散乱強度が弱く検出困難
  * ABF像では、中角領域の散乱を検出するため、重元素と軽元素の散乱強度差が相対的に小さい
  * O原子カラムも暗いスポットとして明瞭に観察される
  * SrとTiの位置（HAADF）とOの位置（ABF）を同時取得することで、完全な原子配列が決定できる

### 演習4-4: プラズモン解析

**問題** ：AlのEELSスペクトルで15 eVにプラズモンピークが観測された。自由電子密度を計算せよ（プラズモンエネルギー $E_p = \hbar\omega_p = \hbar\sqrt{ne^2/m_e\epsilon_0}$）。

**解答例を表示**
    
    
    import numpy as np
    
    E_p = 15  # [eV]
    e = 1.60218e-19  # [C]
    m_e = 9.10938e-31  # [kg]
    epsilon_0 = 8.85419e-12  # [F/m]
    hbar = 1.05457e-34  # [J·s]
    
    # E_p = hbar * omega_p
    omega_p = E_p * e / hbar  # [rad/s]
    
    # omega_p = sqrt(n * e^2 / (m_e * epsilon_0))
    n = (omega_p**2) * m_e * epsilon_0 / (e**2)
    
    print(f"プラズモンエネルギー: {E_p} eV")
    print(f"プラズモン角周波数: {omega_p:.3e} rad/s")
    print(f"自由電子密度: {n:.3e} m^-3")
    print(f"              = {n/1e28:.2f} × 10^28 m^-3")
    
    # Alの理論値（3個/原子、格子定数4.05 Å）
    a = 4.05e-10  # [m]
    atoms_per_cell = 4  # FCC
    n_theory = (atoms_per_cell * 3) / a**3
    print(f"\n理論値（Al FCC, 3 valence e/atom）: {n_theory:.3e} m^-3")
    

### 演習4-5: トモグラフィー投影数

**問題** ：直径50 nmのナノ粒子を1 nm分解能でトモグラフィー再構成するために必要な最小投影数をCrowther criterionから推定せよ。

**解答例を表示**
    
    
    import numpy as np
    
    D = 50  # [nm] 粒子直径
    resolution = 1  # [nm] 目標分解能
    
    # Crowther criterion: N >= π * D / resolution
    N_min = np.pi * D / resolution
    
    print(f"粒子直径: {D} nm")
    print(f"目標分解能: {resolution} nm")
    print(f"Crowther criterionによる最小投影数: {N_min:.1f}")
    print(f"実用的には: {int(np.ceil(N_min * 1.5))} 投影以上を推奨")
    print(f"\n傾斜範囲-70°〜+70°（140°）で2°刻みなら: 70投影")
    print(f"1°刻みなら: 140投影 → 十分")
    

### 演習4-6: STEM検出器最適化

**問題** ：軽元素（C, N, O）と重元素（Pt）を同時に観察したい。どの検出器組み合わせが最適か、理由とともに答えよ。

**解答例を表示**

**最適な組み合わせ** ：

  * **HAADF + ABF 同時取得**

**理由** ：

  * **HAADF** ：Pt（Z=78）の位置を高コントラストで観察。Z^1.7依存性により、軽元素はほぼ見えない
  * **ABF** ：C, N, Oを暗いスポットとして検出。Ptも観察可能
  * **同時取得** ：STEMは1回の走査で複数検出器の信号を並列取得できるため、効率的
  * **相補的情報** ：HAADFで構造の骨格（重元素）を把握し、ABFで軽元素位置を補完

### 演習4-7: EELS厚さ測定

**問題** ：EELSのゼロロスピーク積分強度が10000、全スペクトル積分強度が15000だった。平均自由行程λを100 nmとすると、試料厚さtを推定せよ（$I_{\text{total}}/I_0 = \exp(t/\lambda)$）。

**解答例を表示**
    
    
    import numpy as np
    
    I_zero_loss = 10000
    I_total = 15000
    lambda_mfp = 100  # [nm] 平均自由行程
    
    # I_total / I_zero_loss = exp(t / lambda)
    ratio = I_total / I_zero_loss
    t = lambda_mfp * np.log(ratio)
    
    print(f"ゼロロスピーク積分: {I_zero_loss}")
    print(f"全積分強度: {I_total}")
    print(f"強度比: {ratio:.3f}")
    print(f"平均自由行程: {lambda_mfp} nm")
    print(f"試料厚さ: {t:.1f} nm")
    
    if t < 50:
        print("→ 薄い試料（EELS定量に適している）")
    elif t < 100:
        print("→ 中程度の厚さ（多重散乱の影響あり）")
    else:
        print("→ 厚い試料（EELS定量は困難、より薄くする必要あり）")
    

### 演習4-8: 実践課題

**問題** ：Fe-Cr-Ni ステンレス鋼のSTEM-EELS分析計画を立案せよ。測定すべき信号、予想される課題、対策を含めること。

**解答例を表示**

**測定計画** ：

  1. **試料作製** ： 
     * FIB（集束イオンビーム）で薄片作製（目標厚さ<50 nm）
     * 低加速イオン研磨で表面ダメージ除去
  2. **HAADF-STEM観察** ： 
     * 結晶粒、析出物、界面の形態観察
     * Z-contrastでCr, Feの濃度変化を定性評価
  3. **EELS マッピング** ： 
     * Fe-L2,3（708 eV）、Cr-L2,3（575 eV）、Ni-L2,3（855 eV）のマッピング
     * デュアルEELS（低損失 + コアロス同時取得）で厚さ補正
  4. **定量解析** ： 
     * HyperSpyで各エッジの積分強度抽出
     * Hartree-Slater断面積で濃度定量

**予想される課題と対策** ：

  * **課題1** ：Fe, Crのエッジが近接（708 vs 575 eV）→ **対策** ：高エネルギー分解能設定（<1 eV）、ピーク分離
  * **課題2** ：試料厚さの不均一性 → **対策** ：低損失スペクトルから各点の厚さを推定し補正
  * **課題3** ：ビーム損傷（特に低加速電圧では） → **対策** ：液体窒素冷却ホルダー使用、ビーム電流最小化

## 4.8 学習チェック

以下の質問に答えて、理解度を確認しましょう：

  1. STEMとTEMの像形成原理の違いを説明できますか？
  2. HAADF-STEM像のZ-contrast特性の物理的起源を理解していますか？
  3. ABF像が軽元素観察に適している理由を説明できますか？
  4. EELSスペクトルの構成（ゼロロス、低損失、コアロス）を理解していますか？
  5. EELS定量解析の手順（バックグラウンド除去、積分、断面積補正）を実行できますか？
  6. STEM元素マッピングでEELSとEDSの使い分けを判断できますか？
  7. 電子トモグラフィーの投影定理と3D再構成の原理を理解していますか？

## 4.9 参考文献

  1. Pennycook, S. J., & Nellist, P. D. (Eds.). (2011). _Scanning Transmission Electron Microscopy: Imaging and Analysis_. Springer. - STEM技術の包括的教科書
  2. Egerton, R. F. (2011). _Electron Energy-Loss Spectroscopy in the Electron Microscope_ (3rd ed.). Springer. - EELS解析のバイブル
  3. Findlay, S. D., et al. (2010). "Robust atomic resolution imaging of light elements using scanning transmission electron microscopy." _Applied Physics Letters_ , 95, 191913. - ABF像の原理論文
  4. Muller, D. A. (2009). "Structure and bonding at the atomic scale by scanning transmission electron microscopy." _Nature Materials_ , 8, 263-270. - 原子分解能STEM解析
  5. de Jonge, N., & Ross, F. M. (2011). "Electron microscopy of specimens in liquid." _Nature Nanotechnology_ , 6, 695-704. - 液中STEM観察
  6. Midgley, P. A., & Dunin-Borkowski, R. E. (2009). "Electron tomography and holography in materials science." _Nature Materials_ , 8, 271-280. - 電子トモグラフィーの応用
  7. Krivanek, O. L., et al. (2010). "Atom-by-atom structural and chemical analysis by annular dark-field electron microscopy." _Nature_ , 464, 571-574. - 単原子分析STEM

## 4.10 次章へ

次章では、EDS、EELS、EBSDデータの統合分析をPythonで実践します。HyperSpyライブラリを使ったスペクトル処理、機械学習による相分類、EBSD方位解析、トラブルシューティングを学び、実際の材料解析ワークフローを構築します。
