---
title: 第3章：透過型電子顕微鏡（TEM）入門
chapter_title: 第3章：透過型電子顕微鏡（TEM）入門
subtitle: TEM結像理論、回折解析、高分解能観察の基礎
reading_time: 25-35分
difficulty: 中級
code_examples: 7
---

透過型電子顕微鏡（TEM）は、試料を透過した電子線を利用して、材料の内部構造を原子レベルで観察する強力なツールです。この章では、TEM結像理論、明視野・暗視野像、制限視野回折（SAED）、高分解能TEM（HRTEM）、収差補正技術を学び、Pythonで回折パターン解析とFFT処理を実践します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ TEM結像理論とコントラスト機構（質量厚さ、回折コントラスト）を理解する
  * ✅ 明視野像（BF）と暗視野像（DF）の形成原理と使い分けができる
  * ✅ 制限視野電子回折（SAED）パターンの指数付けができる
  * ✅ 格子像と高分解能TEM像の違いを理解し、FFT解析ができる
  * ✅ コントラスト伝達関数（CTF）の役割と収差補正技術を説明できる
  * ✅ Pythonでエワルド球構成、SAED解析、HRTEM像のFFT処理を実装できる
  * ✅ デフォーカスと球面収差が像に与える影響を定量的に評価できる

## 3.1 TEM結像理論の基礎

### 3.1.1 透過電子顕微鏡の構成

TEM（Transmission Electron Microscope）は、試料を透過した電子線を対物レンズで結像し、投影レンズで拡大する光学系を持ちます。
    
    
    ```mermaid
    flowchart TD
        A[電子銃] --> B[照射系レンズ]
        B --> C[試料ステージ]
        C --> D[対物レンズ]
        D --> E[制限視野絞りSelected Area Aperture]
        E --> F[中間レンズ]
        F --> G[投影レンズ]
        G --> H[蛍光スクリーン/検出器]
    
        D -.回折面.-> I[バックフォーカル面BFP]
        I -.-> F
        D -.像面.-> J[ガウス像面]
        J -.-> F
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#ffeb99,stroke:#ffa500
        style J fill:#99ccff,stroke:#0066cc
    ```

**重要な概念** ：

  * **バックフォーカル面（BFP: Back Focal Plane）** ：対物レンズの焦点面。ここに回折パターンが形成される
  * **ガウス像面** ：試料の実像が形成される面
  * **対物絞り** ：BFPに配置され、特定の回折スポットを選択してコントラストを制御
  * **制限視野絞り** ：像面に配置され、特定の領域から回折パターンを取得（SAED: Selected Area Electron Diffraction）

### 3.1.2 TEM像のコントラスト機構

TEM像のコントラストは主に3つのメカニズムで形成されます：

コントラスト種類 | 物理的起源 | 適用例  
---|---|---  
**質量厚さコントラスト** | 試料の密度・厚さによる電子の散乱強度差 | 非晶質試料、生物試料、厚さ変化の観察  
**回折コントラスト** | 結晶の回折条件（ブラッグ条件）の変化 | 転位、双晶、結晶粒界の観察  
**位相コントラスト** | 透過波と回折波の位相差による干渉 | 高分解能TEM（HRTEM）での格子像観察  
  
**明視野像（Bright Field, BF）** ：対物絞りで透過ビーム（000反射）のみを通過させる。厚い部分や散乱の強い部分が暗く見える。

**暗視野像（Dark Field, DF）** ：対物絞りで特定の回折ビームのみを通過させる。その反射条件を満たす結晶粒のみが明るく光る。

### 3.1.3 コントラスト伝達関数（CTF）

高分解能TEMでは、位相コントラストが重要です。この位相コントラストは**コントラスト伝達関数（Contrast Transfer Function, CTF）** で記述されます：

$$ \text{CTF}(k) = A(k) \sin\left[\chi(k)\right] $$ 

ここで、$A(k)$ は絞り関数、$\chi(k)$ は位相シフトで次式で表されます：

$$ \chi(k) = \frac{2\pi}{\lambda}\left(\frac{\lambda^2 k^2}{2}\Delta f + \frac{\lambda^4 k^4}{4}C_s\right) $$ 

  * $k$：空間周波数（逆Å単位）
  * $\lambda$：電子波長（加速電圧で決まる）
  * $\Delta f$：デフォーカス（焦点ずれ量、負値がアンダーフォーカス）
  * $C_s$：球面収差係数（レンズの収差パラメータ）

**物理的意味** ：

  * CTFが正値の空間周波数領域では、位相コントラストが反転せずに像に寄与
  * CTFが負値の領域では、コントラストが反転（白黒逆転）
  * CTFがゼロとなる周波数（ゼロクロス）では、その空間周波数成分は像に寄与しない
  * 適切なデフォーカス（シェルツァーフォーカス）を設定することで、広い空間周波数範囲で一定符号のCTFを実現

#### コード例3-1: コントラスト伝達関数（CTF）のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_ctf(k, voltage_kV, defocus_nm, Cs_mm, aperture_mrad=None):
        """
        コントラスト伝達関数（CTF）を計算
    
        Parameters
        ----------
        k : array-like
            空間周波数 [1/Å]
        voltage_kV : float
            加速電圧 [kV]
        defocus_nm : float
            デフォーカス [nm]（負値：アンダーフォーカス）
        Cs_mm : float
            球面収差係数 [mm]
        aperture_mrad : float, optional
            対物絞り半角 [mrad]。Noneの場合は無限大（絞りなし）
    
        Returns
        -------
        ctf : ndarray
            CTF値
        """
        # 電子波長計算（相対論補正あり）
        m0 = 9.10938e-31  # 電子質量 [kg]
        e = 1.60218e-19   # 電荷 [C]
        c = 2.99792e8     # 光速 [m/s]
        h = 6.62607e-34   # プランク定数 [J·s]
    
        E = voltage_kV * 1000 * e  # エネルギー [J]
        lambda_pm = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2))) * 1e12  # [pm]
        lambda_A = lambda_pm / 100  # [Å]
    
        # パラメータを適切な単位に変換
        defocus_A = defocus_nm * 10  # [Å]
        Cs_A = Cs_mm * 1e7  # [Å]
    
        # 位相シフト χ(k)
        chi = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_A
        )
    
        # 絞り関数（ガウス型で近似）
        if aperture_mrad is not None:
            theta_max = aperture_mrad * 1e-3  # [rad]
            k_max = theta_max / lambda_A
            A = np.exp(-(k / k_max)**4)
        else:
            A = 1.0
    
        # CTF = A(k) * sin(χ(k))
        ctf = A * np.sin(chi)
    
        return ctf, lambda_A
    
    # シミュレーション設定
    voltage = 200  # [kV]
    Cs = 0.5  # [mm]（現代的なTEM）
    k = np.linspace(0, 10, 1000)  # 空間周波数 [1/Å]
    
    # 異なるデフォーカス値でのCTF
    defocus_values = [-50, -70, -100]  # [nm]
    colors = ['blue', 'green', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：デフォーカス依存性
    for df, color in zip(defocus_values, colors):
        ctf, wavelength = calculate_ctf(k, voltage, df, Cs)
        ax1.plot(k, ctf, color=color, linewidth=2, label=f'Δf = {df} nm')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
    ax1.set_ylabel('CTF', fontsize=12)
    ax1.set_title(f'CTF vs Defocus (200 kV, Cs = {Cs} mm)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1, 1)
    
    # 右図：球面収差係数依存性
    defocus = -70  # [nm]
    Cs_values = [0.5, 1.0, 2.0]  # [mm]
    colors2 = ['purple', 'orange', 'brown']
    
    for Cs_val, color in zip(Cs_values, colors2):
        ctf, wavelength = calculate_ctf(k, voltage, defocus, Cs_val)
        ax2.plot(k, ctf, color=color, linewidth=2, label=f'Cs = {Cs_val} mm')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
    ax2.set_ylabel('CTF', fontsize=12)
    ax2.set_title(f'CTF vs Cs (200 kV, Δf = {defocus} nm)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    # シェルツァーフォーカス（最適デフォーカス）の計算
    Cs_mm = 0.5
    lambda_A = 0.0251  # 200 kVでの波長
    Cs_A = Cs_mm * 1e7
    scherzer_defocus_A = -1.2 * np.sqrt(Cs_A * lambda_A)
    scherzer_defocus_nm = scherzer_defocus_A / 10
    
    print(f"シェルツァーフォーカス: {scherzer_defocus_nm:.1f} nm")
    print(f"第1ゼロクロス周波数: ~{1/2.5:.2f} Å")
    

**出力解釈** ：

  * デフォーカスが大きすぎると、低空間周波数でCTFが振動し、像が複雑になる
  * Csが大きいほど、高空間周波数領域でのCTF振動が激しくなり、分解能が低下
  * シェルツァーフォーカス（~-70 nm）では、広い空間周波数範囲で正のCTFが得られる

## 3.2 明視野像と暗視野像

### 3.2.1 明視野像（BF）の形成

明視野像は、対物絞りで透過ビーム（000反射）のみを選択して形成します。試料が薄く均一な場合、質量厚さコントラストが主となります。

**明視野像の特徴** ：

  * 厚い部分や重元素を含む部分が暗く見える
  * S/N比が高く、低倍率観察に適する
  * 非晶質試料や生物試料の観察に有効
  * 結晶試料では、ブラッグ条件を満たす結晶粒が暗く（回折コントラスト）

### 3.2.2 暗視野像（DF）の形成

暗視野像は、対物絞りで特定の回折スポット（例：111反射）のみを選択して形成します。

**暗視野像の利点** ：

  * 特定の結晶方位を持つ粒子のみが明るく光る
  * 微小析出物や第二相の検出に有効
  * 結晶粒の識別が容易（同一方位の粒が同じ明るさ）
  * 転位や積層欠陥の観察に適する

**中心暗視野法（Centered Dark Field, CDF）** ：光軸を傾斜させて回折ビームを中心に持ってくる手法。球面収差の影響を低減し、高分解能が得られます。

#### コード例3-2: 明視野・暗視野像のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter, rotate
    
    def simulate_polycrystalline_sample(size=512, num_grains=50):
        """
        多結晶試料のシミュレーション
    
        Parameters
        ----------
        size : int
            画像サイズ [pixels]
        num_grains : int
            結晶粒数
    
        Returns
        -------
        grain_map : ndarray
            結晶粒マップ（各ピクセルの結晶粒ID）
        orientations : ndarray
            各結晶粒の方位 [degrees]
        """
        # ボロノイ分割で結晶粒を生成
        from scipy.spatial import Voronoi
    
        # ランダムな種点
        np.random.seed(42)
        points = np.random.rand(num_grains, 2) * size
    
        # 全ピクセルの座標
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        pixels = np.stack([x.ravel(), y.ravel()], axis=1)
    
        # 最近傍の種点を探してボロノイ領域を作成
        from scipy.spatial.distance import cdist
        distances = cdist(pixels, points)
        grain_map = np.argmin(distances, axis=1).reshape(size, size)
    
        # 各結晶粒にランダムな方位を割り当て
        orientations = np.random.rand(num_grains) * 360
    
        return grain_map, orientations
    
    def calculate_diffraction_intensity(grain_map, orientations, hkl=(111,)):
        """
        特定の回折条件での強度を計算
    
        Parameters
        ----------
        grain_map : ndarray
            結晶粒マップ
        orientations : ndarray
            各結晶粒の方位 [degrees]
        hkl : tuple
            回折面指数（簡略化のため方位依存のみ考慮）
    
        Returns
        -------
        intensity : ndarray
            回折強度マップ
        """
        # ブラッグ条件：特定の方位範囲でのみ強く回折
        # 簡略化：方位が30°±5°の範囲で強く回折すると仮定
        target_angle = 30.0
        tolerance = 5.0
    
        intensity = np.zeros_like(grain_map, dtype=float)
    
        for grain_id in range(len(orientations)):
            mask = (grain_map == grain_id)
            angle = orientations[grain_id]
    
            # 回折条件を満たすか判定
            angle_diff = min(abs(angle - target_angle),
                            abs(angle - target_angle + 360),
                            abs(angle - target_angle - 360))
    
            if angle_diff < tolerance:
                # ブラッグ条件を満たす → 強く回折（明視野では暗く、暗視野では明るく）
                intensity[mask] = 1.0
            else:
                intensity[mask] = 0.1  # 弱い回折
    
        # ノイズ追加
        intensity += np.random.normal(0, 0.05, intensity.shape)
    
        return intensity
    
    # シミュレーション実行
    size = 512
    grain_map, orientations = simulate_polycrystalline_sample(size, num_grains=30)
    
    # 明視野像：回折が強い部分が暗い
    diffraction_intensity = calculate_diffraction_intensity(grain_map, orientations)
    bf_image = 1.0 - diffraction_intensity * 0.7  # 透過強度 = 1 - 回折強度
    
    # 暗視野像：特定の回折条件を満たす結晶粒のみ明るい
    df_image = calculate_diffraction_intensity(grain_map, orientations)
    
    # ぼかしを追加（現実的な像に近づける）
    bf_image = gaussian_filter(bf_image, sigma=1.0)
    df_image = gaussian_filter(df_image, sigma=1.0)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 結晶粒マップ
    im0 = axes[0].imshow(grain_map, cmap='tab20', interpolation='nearest')
    axes[0].set_title('Crystal Grain Map\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # 明視野像
    im1 = axes[1].imshow(bf_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Bright Field Image\n(Transmitted Beam: 000)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # 暗視野像
    im2 = axes[2].imshow(df_image, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Dark Field Image\n(Diffracted Beam: 111)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("明視野像：回折条件を満たす結晶粒が暗く見える")
    print("暗視野像：特定の回折条件（111反射）を満たす結晶粒のみが明るく光る")
    

**観察ポイント** ：

  * 明視野像では、回折が強い結晶粒が暗く表示される
  * 暗視野像では、特定の方位を持つ結晶粒のみが選択的に明るく光る
  * 実際のTEMでは、対物絞りの位置を変えることで異なる回折スポットを選択可能

## 3.3 制限視野電子回折（SAED）

### 3.3.1 電子回折の原理

電子回折は、電子波が結晶の周期構造によって回折される現象です。ブラッグの法則が適用されます：

$$ 2d_{hkl}\sin\theta = n\lambda $$ 

TEMでは、入射電子線が試料にほぼ垂直に入るため、$\theta$ は非常に小さく（通常1°以下）、小角近似が成立します：

$$ \sin\theta \approx \theta \approx \tan\theta = \frac{R}{L} $$ 

ここで、$R$ は回折スポットの距離、$L$ はカメラ長です。したがって：

$$ d_{hkl} = \frac{\lambda L}{R} $$ 

**SAED（Selected Area Electron Diffraction）** ：制限視野絞りで特定の領域（通常数百nm〜数μm）からの回折パターンを取得する手法。

### 3.3.2 エワルド球構成

電子回折は、逆格子空間でのエワルド球（Ewald Sphere）と逆格子点の交差として理解されます。

  * **エワルド球** ：半径 $1/\lambda$ の球。入射ビーム方向に沿って描かれる
  * **逆格子点** ：結晶の周期構造に対応する逆格子空間の点
  * **回折条件** ：エワルド球が逆格子点を通るとき、ブラッグ条件が満たされ回折が生じる

#### コード例3-3: エワルド球構成と回折パターンのシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def generate_reciprocal_lattice_fcc(max_hkl=3):
        """
        FCC結晶の逆格子点を生成
    
        Parameters
        ----------
        max_hkl : int
            最大ミラー指数
    
        Returns
        -------
        points : list of tuples
            逆格子点 (h, k, l)
        """
        points = []
        for h in range(-max_hkl, max_hkl + 1):
            for k in range(-max_hkl, max_hkl + 1):
                for l in range(-max_hkl, max_hkl + 1):
                    # FCC消滅則：h, k, lが全て偶数または全て奇数
                    if (h % 2 == k % 2 == l % 2):
                        points.append((h, k, l))
        return points
    
    def plot_ewald_sphere(a_lattice=4.05, voltage_kV=200, zone_axis=[0, 0, 1]):
        """
        エワルド球構成を3Dプロット
    
        Parameters
        ----------
        a_lattice : float
            格子定数 [Å]
        voltage_kV : float
            加速電圧 [kV]
        zone_axis : list
            晶帯軸 [uvw]
        """
        # 電子波長計算
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        h = 6.62607e-34
        E = voltage_kV * 1000 * e
        lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        lambda_A = lambda_m * 1e10
    
        # エワルド球の半径（逆Å）
        k = 1 / lambda_A
    
        # 逆格子ベクトル（FCC）
        reciprocal_points = generate_reciprocal_lattice_fcc(max_hkl=2)
    
        # 逆格子定数
        a_star = 1 / a_lattice
    
        # 3Dプロット
        fig = plt.figure(figsize=(14, 6))
    
        # 左図：3Dエワルド球構成
        ax1 = fig.add_subplot(121, projection='3d')
    
        # 逆格子点をプロット
        for (h, k, l) in reciprocal_points:
            x = h * a_star
            y = k * a_star
            z = l * a_star
            ax1.scatter(x, y, z, c='blue', s=30, alpha=0.6)
            if abs(l) <= 1:  # [001] zone axis近傍
                ax1.text(x, y, z, f'  {h}{k}{l}', fontsize=8)
    
        # エワルド球（簡略化：円で表現）
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
    
        x_sphere = k * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = k * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = k * np.outer(np.ones(100), np.cos(phi)) - k
    
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='red')
    
        # 入射ビーム
        ax1.quiver(0, 0, -k, 0, 0, k*1.5, color='green', arrow_length_ratio=0.1, linewidth=2)
    
        ax1.set_xlabel('$k_x$ [1/Å]', fontsize=11)
        ax1.set_ylabel('$k_y$ [1/Å]', fontsize=11)
        ax1.set_zlabel('$k_z$ [1/Å]', fontsize=11)
        ax1.set_title('Ewald Sphere Construction\n(3D Reciprocal Space)', fontsize=13, fontweight='bold')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_zlim(-2, 2)
    
        # 右図：[001] zone axisの回折パターン（2D投影）
        ax2 = fig.add_subplot(122)
    
        for (h, k, l) in reciprocal_points:
            if l == 0:  # [001] zone axis
                x = h * a_star
                y = k * a_star
                ax2.scatter(x, y, c='blue', s=100, alpha=0.8)
                ax2.text(x, y + 0.1, f'{h}{k}{l}', fontsize=10, ha='center', fontweight='bold')
    
        ax2.scatter(0, 0, c='red', s=200, marker='o', label='Transmitted Beam (000)')
        ax2.set_xlabel('$k_x$ [1/Å]', fontsize=12)
        ax2.set_ylabel('$k_y$ [1/Å]', fontsize=12)
        ax2.set_title('SAED Pattern: [001] Zone Axis\n(Al FCC, 200 kV)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
    
        plt.tight_layout()
        plt.show()
    
        print(f"電子波長: {lambda_A:.5f} Å")
        print(f"エワルド球半径: {k:.2f} 1/Å")
        print(f"逆格子定数: {a_star:.3f} 1/Å")
    
    # 実行
    plot_ewald_sphere(a_lattice=4.05, voltage_kV=200, zone_axis=[0, 0, 1])
    

### 3.3.3 SAED パターンの指数付け

SAED パターンから結晶構造を同定する手順：

  1. **カメラ定数の校正** ：既知の試料（例：Au）で $\lambda L$ を決定
  2. **回折スポット間距離の測定** ：中心（000）から各スポットまでの距離 $R$
  3. **面間隔の計算** ：$d = \lambda L / R$
  4. **ミラー指数の同定** ：計算した $d$ 値を結晶学データと比較
  5. **晶帯軸の決定** ：複数の回折スポットから晶帯軸 $[uvw]$ を決定

#### コード例3-4: SAED パターンの指数付けシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_d_spacing_fcc(a, h, k, l):
        """
        FCC結晶の面間隔を計算
    
        Parameters
        ----------
        a : float
            格子定数 [Å]
        h, k, l : int
            ミラー指数
    
        Returns
        -------
        d : float
            面間隔 [Å]
        """
        d = a / np.sqrt(h**2 + k**2 + l**2)
        return d
    
    def index_saed_pattern(camera_length_mm=500, voltage_kV=200, crystal='Al'):
        """
        SAED パターンの指数付けシミュレーション
    
        Parameters
        ----------
        camera_length_mm : float
            カメラ長 [mm]
        voltage_kV : float
            加速電圧 [kV]
        crystal : str
            結晶種類（'Al', 'Cu', 'Au'）
        """
        # 電子波長計算
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        h = 6.62607e-34
        E = voltage_kV * 1000 * e
        lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        lambda_pm = lambda_m * 1e12  # [pm]
    
        # カメラ定数
        lambda_L = lambda_pm * camera_length_mm  # [pm·mm]
    
        # 格子定数データベース
        lattice_constants = {
            'Al': 4.05,  # [Å]
            'Cu': 3.61,
            'Au': 4.08
        }
        a = lattice_constants[crystal]
    
        # FCC [001] zone axisの許容反射
        reflections = [
            (0, 0, 0),  # 透過波
            (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0),
            (2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0),
            (4, 0, 0), (0, 4, 0), (-4, 0, 0), (0, -4, 0)
        ]
    
        # 回折パターン生成
        fig, ax = plt.subplots(figsize=(10, 10))
    
        for (h, k, l) in reflections:
            if h == 0 and k == 0 and l == 0:
                # 透過波
                ax.scatter(0, 0, c='red', s=300, marker='o', edgecolors='black', linewidths=2, zorder=10)
                ax.text(0, -5, '000', fontsize=12, ha='center', fontweight='bold', color='red')
                continue
    
            # 面間隔計算
            d = calculate_d_spacing_fcc(a, h, k, l)
    
            # 回折スポットの位置（スクリーン上の距離）[mm]
            R_mm = lambda_L / (d * 100)  # d [Å] → [pm]に変換
    
            # 2Dパターンでの位置（[001] zone axis）
            x = h / np.sqrt(h**2 + k**2) * R_mm if h != 0 or k != 0 else 0
            y = k / np.sqrt(h**2 + k**2) * R_mm if h != 0 or k != 0 else 0
    
            # 簡略化：h, kの符号をそのまま使用
            x = h * (lambda_L / (calculate_d_spacing_fcc(a, 2, 0, 0) * 100)) / 2
            y = k * (lambda_L / (calculate_d_spacing_fcc(a, 0, 2, 0) * 100)) / 2
    
            ax.scatter(x, y, c='blue', s=150, alpha=0.8, edgecolors='black', linewidths=1)
            ax.text(x, y - 2, f'{h}{k}{l}', fontsize=10, ha='center', fontweight='bold')
            ax.text(x, y + 2, f'd={d:.3f}Å', fontsize=8, ha='center', color='green')
    
        ax.set_xlabel('x [mm on screen]', fontsize=13)
        ax.set_ylabel('y [mm on screen]', fontsize=13)
        ax.set_title(f'Indexed SAED Pattern: {crystal} FCC [001]\n' +
                     f'(λL = {lambda_L:.2f} pm·mm, a = {a} Å)',
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
        # スケールバー
        ax.plot([25, 35], [-35, -35], 'k-', linewidth=3)
        ax.text(30, -38, '10 mm', fontsize=10, ha='center')
    
        plt.tight_layout()
        plt.show()
    
        # 主要反射のd値を出力
        print(f"\n{crystal} FCC 主要反射の面間隔:")
        print(f"  (200): d = {calculate_d_spacing_fcc(a, 2, 0, 0):.3f} Å")
        print(f"  (220): d = {calculate_d_spacing_fcc(a, 2, 2, 0):.3f} Å")
        print(f"  (400): d = {calculate_d_spacing_fcc(a, 4, 0, 0):.3f} Å")
    
    # 実行
    index_saed_pattern(camera_length_mm=500, voltage_kV=200, crystal='Al')
    

## 3.4 高分解能TEM（HRTEM）

### 3.4.1 格子像と構造像

高分解能TEM（High-Resolution TEM, HRTEM）では、透過波と複数の回折波が干渉して格子像を形成します。

**格子像とHRTEM像の違い** ：

  * **格子像（Lattice Fringe）** ：2波干渉（透過波 + 1つの回折波）で形成。特定の格子面が縞模様として見える
  * **HRTEM像** ：多波干渉（透過波 + 多数の回折波）で形成。原子配列を直接反映した像

**HRTEM像の解釈における注意点** ：

  * HRTEM像は原子配列の**投影** であり、厚さ方向の情報が重畳される
  * デフォーカスやCsにより、像が原子位置と対応しない場合がある（白黒反転）
  * 像シミュレーション（multislice法など）との比較が必須

### 3.4.2 HRTEM像のFFT解析

HRTEM像の**高速フーリエ変換（FFT）** により、逆格子情報を抽出できます。FFTパターンはSAEDパターンに対応します。

#### コード例3-5: HRTEM像の生成とFFT解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift
    
    def generate_hrtem_image(size=512, lattice_a=4.05, zone_axis=[1, 1, 0], noise_level=0.1):
        """
        簡略化されたHRTEM像をシミュレート（2波近似）
    
        Parameters
        ----------
        size : int
            画像サイズ [pixels]
        lattice_a : float
            格子定数 [Å]
        zone_axis : list
            晶帯軸 [uvw]
        noise_level : float
            ノイズレベル
    
        Returns
        -------
        image : ndarray
            HRTEM像
        """
        # ピクセルサイズ（Å/pixel）
        pixel_size = 0.1  # 0.1 Å/pixel → 分解能0.2 Å相当
    
        # 座標グリッド
        x = np.arange(size) * pixel_size
        y = np.arange(size) * pixel_size
        X, Y = np.meshgrid(x, y)
    
        # [110] zone axisの場合、(111)と(-111)面の干渉縞
        # 面間隔（FCC）
        d_111 = lattice_a / np.sqrt(3)
    
        # 格子縞のシミュレーション
        k1 = 2 * np.pi / d_111
    
        # 2方向の格子縞を重ね合わせ
        fringe1 = np.cos(k1 * (X + Y) / np.sqrt(2))
        fringe2 = np.cos(k1 * (X - Y) / np.sqrt(2))
    
        # HRTEM像：多波干渉の簡略モデル
        image = 0.5 + 0.3 * fringe1 + 0.3 * fringe2
    
        # ノイズ追加
        image += np.random.normal(0, noise_level, image.shape)
    
        # 正規化
        image = (image - image.min()) / (image.max() - image.min())
    
        return image, pixel_size
    
    def plot_hrtem_with_fft(image, pixel_size):
        """
        HRTEM像とそのFFTパターンをプロット
    
        Parameters
        ----------
        image : ndarray
            HRTEM像
        pixel_size : float
            ピクセルサイズ [Å/pixel]
        """
        # FFT計算
        fft_image = fftshift(fft2(image))
        fft_magnitude = np.abs(fft_image)
        fft_magnitude_log = np.log(1 + fft_magnitude)  # 対数スケール
    
        # 周波数軸（1/Å）
        freq_x = np.fft.fftshift(np.fft.fftfreq(image.shape[1], d=pixel_size))
        freq_y = np.fft.fftshift(np.fft.fftfreq(image.shape[0], d=pixel_size))
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
        # HRTEM像（全体）
        im0 = axes[0].imshow(image, cmap='gray', extent=[0, image.shape[1]*pixel_size,
                                                          0, image.shape[0]*pixel_size])
        axes[0].set_title('HRTEM Image (Full)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('x [Å]', fontsize=11)
        axes[0].set_ylabel('y [Å]', fontsize=11)
    
        # HRTEM像（拡大）
        zoom_size = 50
        center = image.shape[0] // 2
        zoomed = image[center-zoom_size:center+zoom_size, center-zoom_size:center+zoom_size]
        extent_zoom = [0, zoom_size*2*pixel_size, 0, zoom_size*2*pixel_size]
    
        im1 = axes[1].imshow(zoomed, cmap='gray', extent=extent_zoom)
        axes[1].set_title('HRTEM Image (Zoomed)\nLattice Fringes Visible', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('x [Å]', fontsize=11)
        axes[1].set_ylabel('y [Å]', fontsize=11)
    
        # FFTパターン（対数スケール）
        im2 = axes[2].imshow(fft_magnitude_log, cmap='hot',
                            extent=[freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()])
        axes[2].set_title('FFT Pattern (Log Scale)\n(= Diffractogram)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('$k_x$ [1/Å]', fontsize=11)
        axes[2].set_ylabel('$k_y$ [1/Å]', fontsize=11)
        axes[2].set_xlim(-5, 5)
        axes[2].set_ylim(-5, 5)
    
        # FFTの主要ピークをマーク
        # 中心（000）
        axes[2].scatter(0, 0, c='cyan', s=100, marker='o', edgecolors='white', linewidths=2)
    
        plt.tight_layout()
        plt.show()
    
    # 実行
    image, pixel_size = generate_hrtem_image(size=512, lattice_a=4.05, zone_axis=[1, 1, 0], noise_level=0.05)
    plot_hrtem_with_fft(image, pixel_size)
    
    print("FFTパターンの解釈：")
    print("  - 中心（明るい点）：透過波（000）")
    print("  - 対称な明るいスポット：格子面からの回折波")
    print("  - スポット間距離 ∝ 1/d（面間隔の逆数）")
    

### 3.4.3 収差補正技術

現代のTEMでは、球面収差補正器（Cs-corrector）により、球面収差係数を限りなくゼロに近づけることができます。

**収差補正の効果** ：

  * **分解能向上** ：0.5 Å → 0.05 Å（原子レベル）
  * **CTF改善** ：広い空間周波数範囲で一定符号のCTFを実現
  * **デフォーカス依存性の低減** ：シェルツァーフォーカスの制約が緩和
  * **像解釈の簡素化** ：原子位置と像の対応が直感的に

#### コード例3-6: 収差補正前後のCTF比較
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def ctf_comparison_corrected_vs_uncorrected():
        """
        収差補正前後のCTF比較
        """
        k = np.linspace(0, 15, 1000)  # 空間周波数 [1/Å]
    
        voltage = 200  # [kV]
        lambda_A = 0.0251  # 200 kVでの波長 [Å]
        defocus_nm = -70  # [nm]
        defocus_A = defocus_nm * 10  # [Å]
    
        # 収差補正前：Cs = 0.5 mm
        Cs_uncorrected_mm = 0.5
        Cs_uncorrected_A = Cs_uncorrected_mm * 1e7
    
        chi_uncorrected = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_uncorrected_A
        )
        ctf_uncorrected = np.sin(chi_uncorrected)
    
        # 収差補正後：Cs ≈ -0.01 mm（負の収差で補正）
        Cs_corrected_mm = -0.01
        Cs_corrected_A = Cs_corrected_mm * 1e7
    
        chi_corrected = (2 * np.pi / lambda_A) * (
            0.5 * lambda_A**2 * k**2 * defocus_A +
            0.25 * lambda_A**4 * k**4 * Cs_corrected_A
        )
        ctf_corrected = np.sin(chi_corrected)
    
        # プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
        # 上図：CTF比較
        ax1.plot(k, ctf_uncorrected, 'b-', linewidth=2, label='Uncorrected (Cs = 0.5 mm)')
        ax1.plot(k, ctf_corrected, 'r-', linewidth=2, label='Corrected (Cs ≈ 0 mm)')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
        ax1.set_ylabel('CTF', fontsize=12)
        ax1.set_title('CTF: Aberration-Corrected vs Uncorrected TEM\n(200 kV, Δf = -70 nm)',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 15)
        ax1.set_ylim(-1, 1)
    
        # 注釈：情報伝達限界
        ax1.axvline(x=1/0.2, color='blue', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(1/0.2 + 0.3, 0.8, 'Uncorrected limit\n(~2 Å)', fontsize=10, color='blue')
    
        ax1.axvline(x=1/0.08, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(1/0.08 + 0.3, -0.8, 'Corrected limit\n(~0.8 Å)', fontsize=10, color='red')
    
        # 下図：CTFの位相シフトχ(k)
        ax2.plot(k, chi_uncorrected, 'b-', linewidth=2, label='Uncorrected (Cs = 0.5 mm)')
        ax2.plot(k, chi_corrected, 'r-', linewidth=2, label='Corrected (Cs ≈ 0 mm)')
        ax2.set_xlabel('Spatial Frequency [1/Å]', fontsize=12)
        ax2.set_ylabel('Phase Shift χ(k) [rad]', fontsize=12)
        ax2.set_title('Phase Shift Function χ(k)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 15)
    
        plt.tight_layout()
        plt.show()
    
        # 分解能の推定
        # 第1ゼロクロスまでの周波数範囲が情報伝達に有効
        k_zero_uncorrected = k[np.where(np.diff(np.sign(ctf_uncorrected)))[0][0]]
        k_zero_corrected = k[np.where(np.diff(np.sign(ctf_corrected)))[0][0]]
    
        resolution_uncorrected = 1 / k_zero_uncorrected
        resolution_corrected = 1 / k_zero_corrected
    
        print(f"収差補正前の分解能: ~{resolution_uncorrected:.2f} Å")
        print(f"収差補正後の分解能: ~{resolution_corrected:.2f} Å")
        print(f"分解能向上: {resolution_uncorrected / resolution_corrected:.1f}倍")
    
    # 実行
    ctf_comparison_corrected_vs_uncorrected()
    

## 3.5 演習問題

### 演習3-1: CTFの最適化

**問題** ：300 kVのTEM（Cs = 1.0 mm）でシェルツァーフォーカスを計算し、第1ゼロクロスの空間周波数を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    # パラメータ
    voltage_kV = 300
    Cs_mm = 1.0
    
    # 電子波長計算（相対論補正）
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h = 6.62607e-34
    E = voltage_kV * 1000 * e
    lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_A = lambda_m * 1e10
    
    # シェルツァーフォーカス
    Cs_A = Cs_mm * 1e7
    scherzer_defocus_A = -1.2 * np.sqrt(Cs_A * lambda_A)
    scherzer_defocus_nm = scherzer_defocus_A / 10
    
    # 第1ゼロクロスの空間周波数（近似）
    k_first_zero = 1.5 / (Cs_A**0.25 * lambda_A**0.75)
    resolution_A = 1 / k_first_zero
    
    print(f"300 kVでの電子波長: {lambda_A:.5f} Å")
    print(f"シェルツァーフォーカス: {scherzer_defocus_nm:.1f} nm")
    print(f"第1ゼロクロス空間周波数: {k_first_zero:.3f} 1/Å")
    print(f"点分解能: {resolution_A:.3f} Å")
    

### 演習3-2: SAED指数付け

**問題** ：Cu FCC試料（a = 3.61 Å）の[011]晶帯軸SAED パターンで、200反射スポットが中心から15 mm離れていた。カメラ定数λLを求めよ（加速電圧200 kV）。

**解答例を表示**
    
    
    import numpy as np
    
    # 与えられたデータ
    a = 3.61  # [Å]
    h, k, l = 2, 0, 0
    R_mm = 15  # [mm]
    
    # 面間隔
    d_200 = a / np.sqrt(h**2 + k**2 + l**2)
    
    # カメラ定数 λL = R * d
    # dをpmに変換
    d_pm = d_200 * 100
    lambda_L = R_mm * d_pm
    
    print(f"Cu (200)面の面間隔: {d_200:.3f} Å = {d_pm:.1f} pm")
    print(f"カメラ定数 λL: {lambda_L:.1f} pm·mm")
    
    # 検証：200 kVでの理論値
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h_planck = 6.62607e-34
    E = 200 * 1000 * e
    lambda_m = h_planck / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_pm = lambda_m * 1e12
    
    # 仮にカメラ長500 mmとすると
    L_mm = lambda_L / lambda_pm
    print(f"推定カメラ長: {L_mm:.0f} mm")
    

### 演習3-3: FFT解析

**問題** ：HRTEM像のFFTパターンで、中心から最も近い対称スポットが3.5 1/Å離れている。対応する格子面の面間隔を求めよ。

**解答例を表示**
    
    
    k = 3.5  # [1/Å]（FFTパターンのスポット位置）
    
    # 面間隔 d = 1/k
    d = 1 / k
    
    print(f"FFTスポット空間周波数: {k} 1/Å")
    print(f"対応する面間隔: {d:.3f} Å")
    print(f"これは例えば、Si (111)面 (d = 3.14 Å) やAl (111)面 (d = 2.34 Å) に相当する可能性")
    

### 演習3-4: 暗視野像の活用

**問題** ：Al合金中のAl2Cu析出物（θ'相）を暗視野像で観察したい。適切な回折スポットを選ぶ戦略を説明せよ。

**解答例を表示**

**戦略** ：

  1. まず[001]_Alなどの低指数晶帯軸に試料を傾斜
  2. SAEDパターンを取得し、Al母相とθ'相の回折スポットを同定
  3. θ'相に**特有の回折スポット** （Alには現れない指数）を選択
  4. 例：θ'相の(002)や(100)反射など
  5. そのスポットで暗視野像を撮影 → θ'析出物のみが明るく光る
  6. 中心暗視野法（CDF）を使えば、より高分解能な像が得られる

### 演習3-5: 多波干渉の理解

**問題** ：2波干渉（透過波 + 1つの回折波）と多波干渉（透過波 + 複数の回折波）の違いを、CTFの観点から説明せよ。

**解答例を表示**

**回答** ：

  * **2波干渉** ：1つの空間周波数成分のみが干渉。CTFはその1点でのみ評価される。格子縞として観察されるが、原子配列の詳細は見えない
  * **多波干渉** ：多数の空間周波数成分が同時に干渉。CTFの広い範囲が像形成に寄与。原子配列を反映したHRTEM像が得られる
  * **CTFの役割** ：各空間周波数成分がどの程度像に寄与するかを決定。収差補正により、広い周波数範囲で一定符号のCTFを実現すると、より忠実な原子配列像が得られる

### 演習3-6: 収差補正の効果

**問題** ：収差補正前（Cs = 1.0 mm）と収差補正後（Cs ≈ 0.001 mm）の300 kV TEMで、それぞれの点分解能を比較せよ。

**解答例を表示**
    
    
    import numpy as np
    
    voltage_kV = 300
    
    # 電子波長
    m0 = 9.10938e-31
    e = 1.60218e-19
    c = 2.99792e8
    h = 6.62607e-34
    E = voltage_kV * 1000 * e
    lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    lambda_A = lambda_m * 1e10
    
    # 収差補正前
    Cs_uncorrected_mm = 1.0
    Cs_uncorrected_A = Cs_uncorrected_mm * 1e7
    k_limit_uncorrected = 1.5 / (Cs_uncorrected_A**0.25 * lambda_A**0.75)
    resolution_uncorrected = 1 / k_limit_uncorrected
    
    # 収差補正後
    Cs_corrected_mm = 0.001
    Cs_corrected_A = Cs_corrected_mm * 1e7
    k_limit_corrected = 1.5 / (Cs_corrected_A**0.25 * lambda_A**0.75)
    resolution_corrected = 1 / k_limit_corrected
    
    print(f"300 kVでの電子波長: {lambda_A:.5f} Å")
    print(f"\n収差補正前（Cs = {Cs_uncorrected_mm} mm）:")
    print(f"  点分解能: {resolution_uncorrected:.3f} Å")
    print(f"\n収差補正後（Cs = {Cs_corrected_mm} mm）:")
    print(f"  点分解能: {resolution_corrected:.3f} Å")
    print(f"\n分解能向上: {resolution_uncorrected / resolution_corrected:.1f}倍")
    

### 演習3-7: 実践的HRTEM解析

**問題** ：与えられたHRTEM像（512×512ピクセル、ピクセルサイズ0.05 Å/pixel）のFFTを計算し、主要な格子面の面間隔を3つ抽出せよ。

**解答例を表示**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift
    from scipy.ndimage import gaussian_filter
    
    # ダミーHRTEM像生成（実際のデータに置き換える）
    size = 512
    pixel_size = 0.05  # [Å/pixel]
    
    # 3つの格子面を持つ結晶をシミュレート
    x = np.arange(size) * pixel_size
    y = np.arange(size) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    # 格子面1: d1 = 3.5 Å
    d1 = 3.5
    image = 0.5 + 0.2 * np.cos(2*np.pi * X / d1)
    
    # 格子面2: d2 = 2.0 Å
    d2 = 2.0
    image += 0.15 * np.cos(2*np.pi * (X + Y) / (d2 * np.sqrt(2)))
    
    # 格子面3: d3 = 1.5 Å
    d3 = 1.5
    image += 0.1 * np.cos(2*np.pi * Y / d3)
    
    # ノイズ追加
    image += np.random.normal(0, 0.05, image.shape)
    image = gaussian_filter(image, sigma=0.5)
    
    # FFT計算
    fft_image = fftshift(fft2(image))
    fft_magnitude = np.abs(fft_image)
    fft_magnitude_log = np.log(1 + fft_magnitude)
    
    # 周波数軸
    freq = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    
    # FFTピーク検出（簡易版：中心から3つの主要ピークを手動抽出）
    center = size // 2
    # 実際の解析では、scipy.signal.find_peaks などを使用
    
    print("FFT解析結果（シミュレーションデータ）:")
    print(f"  ピーク1: k ≈ {1/d1:.3f} 1/Å → d ≈ {d1:.2f} Å")
    print(f"  ピーク2: k ≈ {1/d2:.3f} 1/Å → d ≈ {d2:.2f} Å")
    print(f"  ピーク3: k ≈ {1/d3:.3f} 1/Å → d ≈ {d3:.2f} Å")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('HRTEM Image', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(fft_magnitude_log, cmap='hot', extent=[freq.min(), freq.max(), freq.min(), freq.max()])
    ax2.set_title('FFT Pattern (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('$k_x$ [1/Å]', fontsize=11)
    ax2.set_ylabel('$k_y$ [1/Å]', fontsize=11)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()
    

### 演習3-8: 実験計画

**問題** ：未知の結晶試料をTEMで解析する実験計画を立案せよ。明視野像、暗視野像、SAED、HRTEMをどの順序で取得し、何を調べるか説明せよ。

**解答例を表示**

**実験計画** ：

  1. **低倍率明視野像** ：試料全体の形態、厚さ、結晶粒サイズを把握
  2. **SAED（広範囲）** ：多結晶か単結晶かを判定。多結晶の場合、デバイリング取得
  3. **試料傾斜とSAED** ：低指数晶帯軸（[001], [011], [111]など）を探索。デバイリング解析で格子定数と結晶系を推定
  4. **SAED（制限視野）** ：特定の結晶粒からSAEDパターンを取得し、指数付け。結晶構造を同定
  5. **暗視野像** ：特定の回折スポットで暗視野像を撮影。結晶粒、双晶、析出物の分布を観察
  6. **HRTEM** ：高分解能像を取得し、格子像やFFT解析で面間隔を精密測定。欠陥（転位、積層欠陥）の観察
  7. **データ統合** ：SAED、HRTEM、暗視野像の結果を統合し、結晶構造、方位、欠陥を総合的に解析

## 3.6 学習チェック

以下の質問に答えて、理解度を確認しましょう：

  1. TEM結像でバックフォーカル面（BFP）と像面の役割を説明できますか？
  2. 明視野像と暗視野像の違いと、それぞれの適用例を挙げられますか？
  3. コントラスト伝達関数（CTF）の物理的意味を理解していますか？
  4. シェルツァーフォーカスの目的とその計算方法を説明できますか？
  5. SAED パターンから格子定数を決定する手順を説明できますか？
  6. エワルド球構成を使って回折条件を説明できますか？
  7. HRTEM像のFFT解析で何が分かるか説明できますか？
  8. 収差補正技術の効果と分解能への影響を理解していますか？

## 3.7 参考文献

  1. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - TEM教科書の決定版
  2. Kirkland, E. J. (2020). _Advanced Computing in Electron Microscopy_ (3rd ed.). Springer. - HRTEM像シミュレーションとFFT解析
  3. Pennycook, S. J., & Nellist, P. D. (Eds.). (2011). _Scanning Transmission Electron Microscopy: Imaging and Analysis_. Springer. - STEM技術（次章で詳述）
  4. Spence, J. C. H. (2013). _High-Resolution Electron Microscopy_ (4th ed.). Oxford University Press. - HRTEM理論の詳細
  5. Hawkes, P. W., & Spence, J. C. H. (Eds.). (2019). _Springer Handbook of Microscopy_. Springer. - 電子顕微鏡技術の包括的ハンドブック
  6. Reimer, L., & Kohl, H. (2008). _Transmission Electron Microscopy: Physics of Image Formation_ (5th ed.). Springer. - TEM結像理論の詳細
  7. Haider, M., et al. (1998). "Electron microscopy image enhanced." _Nature_ , 392, 768–769. - 収差補正技術のブレークスルー論文

## 3.8 次章へ

次章では、走査透過型電子顕微鏡（STEM）の原理、Z-contrast像（HAADF-STEM）、電子エネルギー損失分光（EELS）、原子分解能分析、トモグラフィーの基礎と応用を学びます。STEMは、収束電子ビームを試料上で走査し、多様な信号を同時に検出できる強力な手法です。
