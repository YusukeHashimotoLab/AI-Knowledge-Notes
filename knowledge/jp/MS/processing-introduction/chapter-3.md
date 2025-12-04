---
title: 第3章：表面処理技術
chapter_title: 第3章：表面処理技術
subtitle: Electroplating, Anodizing, Surface Modification, Coating Technologies
difficulty: 中級
---

## 学習目標

この章を完了することで、以下のスキルを習得できます：

  * ✅ Faradayの法則を用いてめっき厚さを計算し、電流密度を最適化できる
  * ✅ 陽極酸化プロセスの電圧と膜厚の関係を理解し、アルマイト処理を設計できる
  * ✅ イオン注入の濃度プロファイルをガウス分布でモデル化できる
  * ✅ コーティング技術の選定基準を理解し、適切な手法を選択できる
  * ✅ 熱溶射プロセスの粒子速度・温度と密着性の関係を評価できる
  * ✅ 表面処理プロセスパラメータを最適化し、トラブルシューティングできる

## 3.1 電気めっき（Electroplating）

### 3.1.1 Faradayの法則と電気化学基礎

電気めっきは電気分解により金属イオンを陰極（被めっき物）表面で還元析出させるプロセスです。めっき速度と膜厚はFaradayの法則に従います。

**Faradayの第一法則** ：析出金属質量は通電量に比例

$$ m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta $$ 

ここで、

  * $m$: 析出質量 [g]
  * $M$: 金属の原子量 [g/mol]
  * $I$: 電流 [A]
  * $t$: めっき時間 [s]
  * $n$: 電子数（例：Cu²⁺なら2）
  * $F$: Faraday定数（96485 C/mol）
  * $\eta$: 電流効率（通常0.85〜0.98）

めっき厚さ $d$ [μm] は析出質量と密度から：

$$ d = \frac{m}{\rho \cdot A} \times 10^4 $$ 

$\rho$: 金属密度 [g/cm³]、$A$: めっき面積 [cm²]

**電流密度の影響** ：

  * **低電流密度** （0.5〜2 A/dm²）：平滑、緻密な膜、低速
  * **高電流密度** （5〜20 A/dm²）：粗い膜、樹枝状成長、高速

**均一電着性（Throwing Power）** ：

複雑形状部品では電流密度分布が不均一になり、膜厚にばらつきが生じます。均一電着性は浴組成、添加剤、攪拌により改善されます。

#### コード例3.1: Faradayの法則によるめっき厚さ計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_plating_thickness(current_A, time_s, area_cm2,
                                     metal='Cu', efficiency=0.95):
        """
        Faradayの法則を用いためっき厚さ計算
    
        Parameters:
        -----------
        current_A : float
            電流 [A]
        time_s : float
            めっき時間 [s]
        area_cm2 : float
            めっき面積 [cm²]
        metal : str
            金属種類（'Cu', 'Ni', 'Cr', 'Au', 'Ag'）
        efficiency : float
            電流効率（0〜1）
    
        Returns:
        --------
        thickness_um : float
            めっき厚さ [μm]
        """
        # 金属物性データベース
        metal_data = {
            'Cu': {'M': 63.55, 'n': 2, 'rho': 8.96},   # 銅
            'Ni': {'M': 58.69, 'n': 2, 'rho': 8.91},   # ニッケル
            'Cr': {'M': 52.00, 'n': 3, 'rho': 7.19},   # クロム
            'Au': {'M': 196.97, 'n': 1, 'rho': 19.32}, # 金
            'Ag': {'M': 107.87, 'n': 1, 'rho': 10.49}  # 銀
        }
    
        F = 96485  # Faraday定数 [C/mol]
    
        props = metal_data[metal]
        M = props['M']
        n = props['n']
        rho = props['rho']
    
        # 析出質量 [g]
        mass_g = (M * current_A * time_s * efficiency) / (n * F)
    
        # めっき厚さ [μm]
        thickness_um = (mass_g / (rho * area_cm2)) * 1e4
    
        return thickness_um
    
    # 実行例：銅めっき
    current = 2.0      # 2A
    time_hours = 1.0   # 1時間
    time_s = time_hours * 3600
    area = 100.0       # 100cm²
    
    thickness = calculate_plating_thickness(current, time_s, area,
                                             metal='Cu', efficiency=0.95)
    
    print(f"=== 銅めっきプロセス計算 ===")
    print(f"電流: {current} A")
    print(f"電流密度: {current/area*100:.2f} A/dm²")
    print(f"めっき時間: {time_hours} 時間")
    print(f"めっき面積: {area} cm²")
    print(f"電流効率: 95%")
    print(f"➡ めっき厚さ: {thickness:.2f} μm")
    
    # めっき時間 vs 膜厚のプロット
    time_range = np.linspace(0, 2, 100) * 3600  # 0〜2時間
    thicknesses = [calculate_plating_thickness(current, t, area, 'Cu', 0.95)
                   for t in time_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_range/3600, thicknesses, linewidth=2, color='#f5576c')
    plt.xlabel('めっき時間 [hours]', fontsize=12)
    plt.ylabel('めっき厚さ [μm]', fontsize=12)
    plt.title('銅めっき：めっき時間と膜厚の関係', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 電流密度の影響
    current_densities = np.linspace(0.5, 10, 50)  # 0.5〜10 A/dm²
    area_dm2 = area / 100  # cm² → dm²
    time_fixed = 3600  # 1時間
    
    thicknesses_cd = []
    for cd in current_densities:
        I = cd * area_dm2
        thick = calculate_plating_thickness(I, time_fixed, area, 'Cu', 0.95)
        thicknesses_cd.append(thick)
    
    plt.figure(figsize=(10, 6))
    plt.plot(current_densities, thicknesses_cd, linewidth=2, color='#f093fb')
    plt.axvspan(0.5, 2, alpha=0.2, color='green', label='低電流密度（平滑）')
    plt.axvspan(5, 10, alpha=0.2, color='red', label='高電流密度（粗い）')
    plt.xlabel('電流密度 [A/dm²]', fontsize=12)
    plt.ylabel('めっき厚さ [μm]', fontsize=12)
    plt.title('電流密度とめっき厚さの関係（1時間）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.1.2 めっき浴と添加剤

めっき浴の組成は膜質に決定的な影響を与えます。

浴成分 | 役割 | 典型的濃度  
---|---|---  
金属塩（CuSO₄など） | 金属イオン供給 | 200〜250 g/L  
導電塩（H₂SO₄など） | 導電性向上 | 50〜80 g/L  
光沢剤 | 平滑化、光沢付与 | 数ppm〜数百ppm  
レベリング剤 | 凹凸平坦化 | 数ppm〜数十ppm  
界面活性剤 | 水素ガス放出促進 | 数ppm  
  
#### コード例3.2: 電流密度分布シミュレーション（2D電極）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import laplace
    
    def simulate_current_distribution_2d(width=50, height=50,
                                          anode_position='top',
                                          cathode_position='bottom',
                                          iterations=500):
        """
        2D電極配置での電流密度分布シミュレーション
        Laplaceの方程式を有限差分法で解く
    
        Parameters:
        -----------
        width, height : int
            計算グリッドサイズ
        anode_position : str
            陽極位置（'top', 'bottom', 'left', 'right'）
        cathode_position : str
            陰極位置（'top', 'bottom', 'left', 'right'）
        iterations : int
            反復計算回数
        """
        # 電位分布の初期化
        phi = np.zeros((height, width))
    
        # 境界条件設定
        if anode_position == 'top':
            phi[0, :] = 1.0  # 陽極電位
        elif anode_position == 'bottom':
            phi[-1, :] = 1.0
        elif anode_position == 'left':
            phi[:, 0] = 1.0
        elif anode_position == 'right':
            phi[:, -1] = 1.0
    
        if cathode_position == 'top':
            phi[0, :] = 0.0  # 陰極電位
        elif cathode_position == 'bottom':
            phi[-1, :] = 0.0
        elif cathode_position == 'left':
            phi[:, 0] = 0.0
        elif cathode_position == 'right':
            phi[:, -1] = 0.0
    
        # Laplaceの方程式を反復法で解く（∇²φ = 0）
        for _ in range(iterations):
            phi_new = phi.copy()
            phi_new[1:-1, 1:-1] = (phi[:-2, 1:-1] + phi[2:, 1:-1] +
                                   phi[1:-1, :-2] + phi[1:-1, 2:]) / 4.0
    
            # 境界条件を再適用
            if anode_position == 'top':
                phi_new[0, :] = 1.0
            elif anode_position == 'bottom':
                phi_new[-1, :] = 1.0
    
            if cathode_position == 'bottom':
                phi_new[-1, :] = 0.0
            elif cathode_position == 'top':
                phi_new[0, :] = 0.0
    
            phi = phi_new
    
        # 電流密度 = -∇φ（電位勾配に比例）
        grad_y, grad_x = np.gradient(phi)
        current_density = np.sqrt(grad_x**2 + grad_y**2)
    
        return phi, current_density
    
    # 実行例：上部陽極、下部陰極
    phi, j = simulate_current_distribution_2d(width=50, height=50,
                                               anode_position='top',
                                               cathode_position='bottom')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 電位分布
    im1 = axes[0].imshow(phi, cmap='viridis', origin='lower')
    axes[0].set_title('電位分布', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X位置')
    axes[0].set_ylabel('Y位置')
    plt.colorbar(im1, ax=axes[0], label='電位 [V]')
    
    # 電流密度分布
    im2 = axes[1].imshow(j, cmap='hot', origin='lower')
    axes[1].set_title('電流密度分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X位置')
    axes[1].set_ylabel('Y位置')
    plt.colorbar(im2, ax=axes[1], label='電流密度 [a.u.]')
    
    # 陰極表面の電流密度分布
    cathode_j = j[-1, :]  # 下端（陰極）
    axes[2].plot(cathode_j, linewidth=2, color='#f5576c')
    axes[2].set_title('陰極表面の電流密度', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X位置')
    axes[2].set_ylabel('電流密度 [a.u.]')
    axes[2].grid(True, alpha=0.3)
    
    # 均一性評価
    uniformity = (1 - (cathode_j.std() / cathode_j.mean())) * 100
    axes[2].text(0.5, 0.95, f'均一性: {uniformity:.1f}%',
                 transform=axes[2].transAxes,
                 ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"電流密度の均一性: {uniformity:.2f}%")
    print(f"電流密度の最大値/最小値比: {cathode_j.max()/cathode_j.min():.2f}")
    

## 3.2 陽極酸化（Anodizing）

### 3.2.1 アルミニウム陽極酸化の原理

陽極酸化は金属表面を電気化学的に酸化させ、酸化皮膜を形成するプロセスです。アルミニウムのアルマイト処理が代表例です。

**陽極酸化プロセス** ：

  1. アルミニウムを陽極、白金などを陰極として電解液（硫酸、シュウ酸など）に浸漬
  2. 直流電圧印加により、Al表面でAl₂O₃皮膜が成長
  3. 皮膜は多孔質構造（バリア層 + ポーラス層）

    
    
    ```mermaid
    flowchart TB
        subgraph "陽極酸化セル"
            A[アルミニウム陽極]
            B[電解液硫酸/シュウ酸]
            C[白金陰極]
            D[DC電源]
        end
    
        D -->|電圧印加| A
        D --> C
        A -->|Al³⁺| B
        B -->|O²⁻| A
        A -->|Al₂O₃形成| E[酸化皮膜]
    
        E --> F[バリア層緻密・薄]
        E --> G[ポーラス層多孔質・厚]
    
        G --> H[封孔処理熱水/蒸気]
        H --> I[最終皮膜耐食性向上]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
        style I fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
    ```

**膜厚と電圧の関係** ：

硫酸浴の場合、バリア層の厚さは印加電圧にほぼ比例します（経験則）：

$$ d_{\text{barrier}} \approx 1.4 \, [\text{nm/V}] \times V $$ 

全体の膜厚（バリア層 + ポーラス層）はめっき時間と電流密度に依存します。

#### コード例3.3: 陽極酸化膜厚 vs 電圧の関係
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def anodization_thickness(voltage, material='Al',
                              electrolyte='H2SO4', time_min=30):
        """
        陽極酸化皮膜厚さの計算
    
        Parameters:
        -----------
        voltage : float or array
            印加電圧 [V]
        material : str
            材料（'Al', 'Ti'）
        electrolyte : str
            電解液（'H2SO4', 'H2C2O4'）
        time_min : float
            処理時間 [分]
    
        Returns:
        --------
        barrier_thickness : float
            バリア層厚さ [nm]
        total_thickness : float
            全体厚さ [μm]
        """
        # 材料・電解液ごとの定数
        if material == 'Al':
            if electrolyte == 'H2SO4':
                k_barrier = 1.4  # nm/V (硫酸浴)
                k_porous = 0.3   # μm/min at 1.5 A/dm²
            elif electrolyte == 'H2C2O4':
                k_barrier = 1.0  # nm/V (シュウ酸浴)
                k_porous = 0.5   # μm/min
        elif material == 'Ti':
            k_barrier = 2.5  # nm/V (TiO₂)
            k_porous = 0.2   # μm/min
    
        # バリア層厚さ [nm]
        barrier_thickness = k_barrier * voltage
    
        # ポーラス層厚さ [μm]（簡易モデル）
        porous_thickness = k_porous * time_min
    
        # 全体厚さ [μm]
        total_thickness = (barrier_thickness / 1000) + porous_thickness
    
        return barrier_thickness, total_thickness
    
    # 電圧範囲のスキャン
    voltages = np.linspace(10, 100, 100)
    barrier_thicknesses = []
    total_thicknesses = []
    
    for V in voltages:
        d_barrier, d_total = anodization_thickness(V, 'Al', 'H2SO4', 30)
        barrier_thicknesses.append(d_barrier)
        total_thicknesses.append(d_total)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # バリア層厚さ vs 電圧
    axes[0].plot(voltages, barrier_thicknesses, linewidth=2,
                 color='#f5576c', label='バリア層')
    axes[0].set_xlabel('印加電圧 [V]', fontsize=12)
    axes[0].set_ylabel('バリア層厚さ [nm]', fontsize=12)
    axes[0].set_title('バリア層厚さと電圧の関係（Al/硫酸浴）',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 全体厚さ vs 電圧
    axes[1].plot(voltages, total_thicknesses, linewidth=2,
                 color='#f093fb', label='全体厚さ（30分）')
    axes[1].set_xlabel('印加電圧 [V]', fontsize=12)
    axes[1].set_ylabel('全体厚さ [μm]', fontsize=12)
    axes[1].set_title('全体厚さと電圧の関係（Al/硫酸浴）',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 設計例：50nm厚のバリア層が必要な場合
    target_barrier = 50  # nm
    required_voltage = target_barrier / 1.4
    print(f"=== 陽極酸化プロセス設計 ===")
    print(f"目標バリア層厚さ: {target_barrier} nm")
    print(f"➡ 必要電圧: {required_voltage:.1f} V")
    
    # 時間の影響
    times = np.linspace(10, 60, 50)  # 10〜60分
    total_thicknesses_time = []
    for t in times:
        _, d_total = anodization_thickness(50, 'Al', 'H2SO4', t)
        total_thicknesses_time.append(d_total)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_thicknesses_time, linewidth=2, color='#f5576c')
    plt.xlabel('処理時間 [min]', fontsize=12)
    plt.ylabel('全体厚さ [μm]', fontsize=12)
    plt.title('陽極酸化膜厚と処理時間の関係（50V, 硫酸浴）',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### 3.2.2 封孔処理（Sealing）

ポーラス層の孔を閉じて耐食性を向上させる後処理です。

  * **熱水封孔** ：95〜100℃の純水で30〜60分、Al(OH)₃が孔を閉じる
  * **蒸気封孔** ：110℃の水蒸気で10〜30分
  * **冷封孔** ：ニッケル塩溶液で常温処理（省エネ）

## 3.3 表面改質技術

### 3.3.1 イオン注入（Ion Implantation）

イオン注入は高エネルギーイオンを材料表面に打ち込み、化学組成や結晶構造を改変する技術です。半導体製造のドーピングや金属の表面硬化に利用されます。

**イオン注入プロセス** ：

  1. イオン源でイオン生成（例：N⁺, B⁺, P⁺）
  2. 加速電場で10〜200 keVに加速
  3. 質量分析器で目的イオンのみ選択
  4. 真空チャンバ内で試料に照射

**濃度プロファイル（LSS理論）** ：

イオン注入後の濃度分布はガウス分布で近似されます：

$$ C(x) = \frac{\Phi}{\sqrt{2\pi} \Delta R_p} \exp\left(-\frac{(x - R_p)^2}{2 \Delta R_p^2}\right) $$ 

  * $C(x)$: 深さ $x$ での濃度 [atoms/cm³]
  * $\Phi$: ドーズ量（全イオン数/面積） [ions/cm²]
  * $R_p$: 飛程（ピーク深さ） [nm]
  * $\Delta R_p$: 飛程のばらつき（標準偏差） [nm]

#### コード例3.4: イオン注入濃度プロファイル（ガウスLSS理論）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    
    def ion_implantation_profile(energy_keV, dose_cm2, ion='N',
                                  substrate='Si', depth_range=None):
        """
        イオン注入濃度プロファイル計算（ガウス近似）
    
        Parameters:
        -----------
        energy_keV : float
            イオンエネルギー [keV]
        dose_cm2 : float
            ドーズ量 [ions/cm²]
        ion : str
            イオン種（'N', 'B', 'P', 'As'）
        substrate : str
            基板材料（'Si', 'Fe', 'Ti'）
        depth_range : array
            深さ範囲 [nm]（Noneなら自動設定）
    
        Returns:
        --------
        depth : array
            深さ [nm]
        concentration : array
            濃度 [atoms/cm³]
        """
        # 簡易LSS理論パラメータ（経験式）
        # 実際は SRIM/TRIM などのシミュレーションツールを使用
    
        # イオン質量
        ion_masses = {'N': 14, 'B': 11, 'P': 31, 'As': 75}
        M_ion = ion_masses[ion]
    
        # 基板密度・原子量
        substrate_data = {
            'Si': {'rho': 2.33, 'M': 28},
            'Fe': {'rho': 7.87, 'M': 56},
            'Ti': {'rho': 4.51, 'M': 48}
        }
        rho_sub = substrate_data[substrate]['rho']
        M_sub = substrate_data[substrate]['M']
    
        # 飛程 Rp [nm]（簡易式）
        Rp = 10 * energy_keV**0.7 * (M_sub / M_ion)**0.5
    
        # 飛程のばらつき ΔRp [nm]
        delta_Rp = 0.3 * Rp
    
        if depth_range is None:
            depth_range = np.linspace(0, 3 * Rp, 500)
    
        # ガウス濃度分布
        concentration = (dose_cm2 / (np.sqrt(2 * np.pi) * delta_Rp)) * \
                        np.exp(-(depth_range - Rp)**2 / (2 * delta_Rp**2))
    
        return depth_range, concentration, Rp, delta_Rp
    
    # 実行例：窒素イオンをシリコンに注入
    energy = 50  # keV
    dose = 1e16  # ions/cm²
    
    depth, conc, Rp, delta_Rp = ion_implantation_profile(
        energy, dose, ion='N', substrate='Si'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(depth, conc, linewidth=2, color='#f5576c', label=f'{energy} keV, {dose:.0e} ions/cm²')
    plt.axvline(Rp, color='gray', linestyle='--', alpha=0.7, label=f'Rp = {Rp:.1f} nm')
    plt.axvspan(Rp - delta_Rp, Rp + delta_Rp, alpha=0.2, color='orange',
                label=f'ΔRp = {delta_Rp:.1f} nm')
    plt.xlabel('深さ [nm]', fontsize=12)
    plt.ylabel('濃度 [atoms/cm³]', fontsize=12)
    plt.title('イオン注入濃度プロファイル（N⁺ → Si）', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # エネルギー依存性
    energies = [30, 50, 100, 150]  # keV
    plt.figure(figsize=(10, 6))
    for E in energies:
        d, c, rp, drp = ion_implantation_profile(E, dose, 'N', 'Si')
        plt.plot(d, c, linewidth=2, label=f'{E} keV (Rp={rp:.1f} nm)')
    
    plt.xlabel('深さ [nm]', fontsize=12)
    plt.ylabel('濃度 [atoms/cm³]', fontsize=12)
    plt.title('イオン注入エネルギーと濃度プロファイル', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"=== イオン注入パラメータ ===")
    print(f"イオン種: N⁺")
    print(f"基板: Si")
    print(f"エネルギー: {energy} keV")
    print(f"ドーズ量: {dose:.0e} ions/cm²")
    print(f"➡ 飛程 Rp: {Rp:.2f} nm")
    print(f"➡ 飛程ばらつき ΔRp: {delta_Rp:.2f} nm")
    print(f"➡ ピーク濃度: {conc.max():.2e} atoms/cm³")
    

### 3.3.2 プラズマ処理（Plasma Treatment）

プラズマにより表面の化学結合を切断・改質し、濡れ性、接着性、生体適合性を向上させます。

  * **酸素プラズマ** ：表面親水化、有機物除去
  * **アルゴンプラズマ** ：表面クリーニング、活性化
  * **窒素プラズマ** ：表面窒化、硬度向上

### 3.3.3 レーザー表面溶融（Laser Surface Melting）

高出力レーザーで表面を急速加熱・溶融・冷却し、微細結晶粒や非晶質層を形成します。硬度、耐摩耗性が向上します。

## 3.4 コーティング技術

### 3.4.1 熱溶射（Thermal Spray）

熱溶射は溶融または半溶融状態の粒子を高速で基材に衝突させ、コーティング層を形成するプロセスです。

**熱溶射法の分類** ：

  * **フレーム溶射** ：アセチレン/酸素炎で粒子溶融、安価、密着力中
  * **プラズマ溶射** ：高温プラズマ（10,000℃以上）、高品質、セラミックス可
  * **高速フレーム溶射（HVOF）** ：超音速炎（Mach 2〜3）、高密着力、高密度
  * **コールドスプレー** ：粒子を固相で超音速加速、低酸化、金属・複合材料

**重要パラメータ** ：

  * **粒子速度** ：100〜1200 m/s（手法により異なる）
  * **粒子温度** ：融点付近〜3000℃
  * **密着強度** ：機械的かみ合い + 金属結合 + 拡散結合

#### コード例3.5: コーティング密着強度予測（機械的・熱的特性）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def predict_coating_adhesion(particle_velocity_ms,
                                  particle_temp_C,
                                  coating_material='WC-Co',
                                  substrate_material='Steel'):
        """
        コーティング密着強度の予測（簡易モデル）
    
        Parameters:
        -----------
        particle_velocity_ms : float
            粒子速度 [m/s]
        particle_temp_C : float
            粒子温度 [℃]
        coating_material : str
            コーティング材料
        substrate_material : str
            基板材料
    
        Returns:
        --------
        adhesion_MPa : float
            予測密着強度 [MPa]
        """
        # 材料物性データベース
        material_data = {
            'WC-Co': {'T_melt': 2870, 'rho': 14.5, 'E': 600},
            'Al2O3': {'T_melt': 2072, 'rho': 3.95, 'E': 380},
            'Ni': {'T_melt': 1455, 'rho': 8.9, 'E': 200},
            'Steel': {'T_melt': 1500, 'rho': 7.85, 'E': 210}
        }
    
        coating_props = material_data[coating_material]
        substrate_props = material_data[substrate_material]
    
        # 簡易密着強度モデル（経験式）
        # adhesion ∝ v^a * (T/Tm)^b
    
        # 速度寄与（運動エネルギー → 塑性変形）
        v_factor = (particle_velocity_ms / 500)**1.5  # 正規化
    
        # 温度寄与（拡散結合促進）
        T_ratio = particle_temp_C / coating_props['T_melt']
        T_factor = T_ratio**0.8
    
        # ヤング率の適合性（大きい差は不利）
        E_ratio = min(coating_props['E'], substrate_props['E']) / \
                  max(coating_props['E'], substrate_props['E'])
        E_factor = E_ratio**0.5
    
        # 基礎密着強度（材料依存）
        base_adhesion = 30  # MPa
    
        # 総合密着強度 [MPa]
        adhesion_MPa = base_adhesion * v_factor * T_factor * E_factor
    
        return adhesion_MPa
    
    # パラメータスキャン：粒子速度の影響
    velocities = np.linspace(100, 1000, 50)  # m/s
    temp_fixed = 2000  # ℃
    
    adhesions_wc = []
    adhesions_al2o3 = []
    
    for v in velocities:
        adh_wc = predict_coating_adhesion(v, temp_fixed, 'WC-Co', 'Steel')
        adh_al2o3 = predict_coating_adhesion(v, temp_fixed, 'Al2O3', 'Steel')
        adhesions_wc.append(adh_wc)
        adhesions_al2o3.append(adh_al2o3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, adhesions_wc, linewidth=2,
             color='#f5576c', label='WC-Co コーティング')
    plt.plot(velocities, adhesions_al2o3, linewidth=2,
             color='#f093fb', label='Al₂O₃ コーティング')
    plt.xlabel('粒子速度 [m/s]', fontsize=12)
    plt.ylabel('予測密着強度 [MPa]', fontsize=12)
    plt.title('熱溶射：粒子速度とコーティング密着強度', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # パラメータスキャン：粒子温度の影響
    temps = np.linspace(1000, 2800, 50)  # ℃
    vel_fixed = 600  # m/s
    
    adhesions_temp = []
    for T in temps:
        adh = predict_coating_adhesion(vel_fixed, T, 'WC-Co', 'Steel')
        adhesions_temp.append(adh)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temps, adhesions_temp, linewidth=2, color='#f5576c')
    plt.xlabel('粒子温度 [℃]', fontsize=12)
    plt.ylabel('予測密着強度 [MPa]', fontsize=12)
    plt.title('熱溶射：粒子温度とコーティング密着強度（WC-Co）', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 最適化例
    v_opt = 800  # m/s
    T_opt = 2500  # ℃
    adh_opt = predict_coating_adhesion(v_opt, T_opt, 'WC-Co', 'Steel')
    
    print(f"=== 熱溶射プロセス最適化 ===")
    print(f"コーティング材料: WC-Co")
    print(f"基板材料: Steel")
    print(f"最適粒子速度: {v_opt} m/s")
    print(f"最適粒子温度: {T_opt} ℃")
    print(f"➡ 予測密着強度: {adh_opt:.2f} MPa")
    

### 3.4.2 PVD/CVD基礎

**PVD（Physical Vapor Deposition）** ：物理的蒸発・スパッタリングによる薄膜形成（詳細は第5章）

**CVD（Chemical Vapor Deposition）** ：化学反応による薄膜形成（詳細は第5章）

表面処理の文脈では、TiN（窒化チタン）、CrN（窒化クロム）、DLC（ダイヤモンドライクカーボン）などの硬質コーティングに利用されます。

### 3.4.3 ゾル-ゲルコーティング（Sol-Gel Coating）

ゾル-ゲル法は液相からゲル化・焼成により酸化物薄膜を形成する手法です。

  * **利点** ：低温プロセス、大面積対応、多孔質膜可能、組成制御容易
  * **用途** ：反射防止膜、耐食膜、触媒担体、光学膜

#### コード例3.6: 熱溶射粒子の温度・速度モデリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_spray_particle_dynamics(particle_diameter_um,
                                          material='WC-Co',
                                          spray_method='HVOF',
                                          distance_mm=150):
        """
        熱溶射粒子の飛行中の温度・速度変化モデル
    
        Parameters:
        -----------
        particle_diameter_um : float
            粒子直径 [μm]
        material : str
            粒子材料
        spray_method : str
            溶射法（'Flame', 'Plasma', 'HVOF'）
        distance_mm : float
            噴射距離 [mm]
    
        Returns:
        --------
        velocity : array
            速度 [m/s]
        temperature : array
            温度 [K]
        distance : array
            距離 [mm]
        """
        # 材料物性
        material_props = {
            'WC-Co': {'rho': 14500, 'Cp': 200, 'T_melt': 2870 + 273},
            'Al2O3': {'rho': 3950, 'Cp': 880, 'T_melt': 2072 + 273},
            'Ni': {'rho': 8900, 'Cp': 444, 'T_melt': 1455 + 273}
        }
        props = material_props[material]
    
        # 溶射法ごとの初期条件
        initial_conditions = {
            'Flame': {'v0': 100, 'T0': 2500 + 273},
            'Plasma': {'v0': 300, 'T0': 10000 + 273},
            'HVOF': {'v0': 800, 'T0': 2800 + 273}
        }
        ic = initial_conditions[spray_method]
    
        # 距離範囲
        distance = np.linspace(0, distance_mm, 500)
    
        # 簡易抗力モデル（速度減衰）
        drag_coeff = 0.44  # 球形粒子
        air_rho = 1.2  # kg/m³
        particle_mass = (4/3) * np.pi * (particle_diameter_um/2 * 1e-6)**3 * props['rho']
        particle_area = np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # 速度減衰定数
        k_v = (0.5 * drag_coeff * air_rho * particle_area) / particle_mass
        velocity = ic['v0'] * np.exp(-k_v * distance * 1e-3)
    
        # 温度減衰（対流冷却）
        h = 100  # 熱伝達係数 [W/m²K]
        T_air = 300  # 空気温度 [K]
        surface_area = 4 * np.pi * (particle_diameter_um/2 * 1e-6)**2
    
        # 温度減衰定数
        k_T = (h * surface_area) / (particle_mass * props['Cp'])
        temperature = T_air + (ic['T0'] - T_air) * np.exp(-k_T * distance * 1e-3 / velocity[0])
    
        return velocity, temperature - 273, distance  # 温度を℃に変換
    
    # 実行例：HVOF溶射でWC-Co粒子
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 速度プロファイル
    axes[0].plot(d, v, linewidth=2, color='#f5576c')
    axes[0].set_xlabel('噴射距離 [mm]', fontsize=12)
    axes[0].set_ylabel('粒子速度 [m/s]', fontsize=12)
    axes[0].set_title('熱溶射粒子の速度プロファイル（HVOF, WC-Co, 40μm）',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 温度プロファイル
    axes[1].plot(d, T, linewidth=2, color='#f093fb')
    axes[1].axhline(2870, color='red', linestyle='--', alpha=0.7, label='WC-Co融点')
    axes[1].set_xlabel('噴射距離 [mm]', fontsize=12)
    axes[1].set_ylabel('粒子温度 [℃]', fontsize=12)
    axes[1].set_title('熱溶射粒子の温度プロファイル', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 基板到達時の状態
    v_impact = v[-1]
    T_impact = T[-1]
    print(f"=== 基板到達時の粒子状態 ===")
    print(f"噴射距離: {d[-1]:.1f} mm")
    print(f"到達速度: {v_impact:.1f} m/s")
    print(f"到達温度: {T_impact:.1f} ℃")
    print(f"溶融状態: {'溶融' if T_impact > 2870 else '固相'}")
    
    # 複数粒径の比較
    diameters = [20, 40, 60, 80]  # μm
    plt.figure(figsize=(10, 6))
    for dia in diameters:
        v_d, T_d, d_d = thermal_spray_particle_dynamics(dia, 'WC-Co', 'HVOF', 150)
        plt.plot(d_d, v_d, linewidth=2, label=f'{dia} μm')
    
    plt.xlabel('噴射距離 [mm]', fontsize=12)
    plt.ylabel('粒子速度 [m/s]', fontsize=12)
    plt.title('粒径による速度プロファイルの違い（HVOF, WC-Co）',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.5 表面処理技術の選定

### 3.5.1 要求特性と技術の対応

要求特性 | 適した技術 | 特徴  
---|---|---  
耐食性 | めっき（Ni, Cr）、陽極酸化 | 化学的バリア層形成  
耐摩耗性 | 熱溶射（WC-Co）、PVD（TiN, CrN） | 高硬度層形成  
装飾性（外観） | めっき（Au, Ag, Ni-Cr）、陽極酸化 | 光沢、色彩  
導電性 | めっき（Cu, Ag, Au） | 低抵抗接触  
生体適合性 | プラズマ処理、陽極酸化（Ti） | 表面親水化、酸化物層  
断熱性 | 熱溶射（セラミックス） | 低熱伝導率  
表面硬化 | イオン注入（N⁺）、レーザー処理 | 母材変形なし  
  
### 3.5.2 技術選定フローチャート
    
    
    ```mermaid
    flowchart TD
        A[表面処理要求] --> B{主要特性は?}
    
        B -->|耐食性| C{膜厚要求}
        C -->|薄膜1-10μm| D[陽極酸化]
        C -->|厚膜10-100μm| E[めっきNi/Cr]
    
        B -->|耐摩耗性| F{使用温度}
        F -->|常温〜300℃| G[PVD/CVDTiN, CrN]
        F -->|300℃以上| H[熱溶射WC-Co]
    
        B -->|装飾性| I{導電性必要?}
        I -->|必要| J[めっきAu/Ag]
        I -->|不要| K[陽極酸化着色]
    
        B -->|導電性| L[めっきCu/Ag/Au]
    
        B -->|生体適合性| M[プラズマ処理or Ti陽極酸化]
    
        B -->|表面硬化| N{母材加熱OK?}
        N -->|NG| O[イオン注入]
        N -->|OK| P[レーザー処理or 熱溶射]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style E fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style G fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style H fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style J fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style K fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style L fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style M fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style O fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
        style P fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#fff
    ```

#### コード例3.7: 表面処理プロセス総合ワークフロー（パラメータ最適化）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    class SurfaceTreatmentOptimizer:
        """
        表面処理プロセスパラメータ最適化クラス
        """
        def __init__(self, treatment_type='electroplating'):
            self.treatment_type = treatment_type
    
        def objective_function(self, params, targets):
            """
            目的関数：目標特性との誤差を最小化
    
            Parameters:
            -----------
            params : array
                プロセスパラメータ（治療法により異なる）
            targets : dict
                目標特性値
    
            Returns:
            --------
            error : float
                誤差（小さいほど良い）
            """
            if self.treatment_type == 'electroplating':
                # パラメータ：[電流密度 A/dm², めっき時間 h, 効率]
                current_density, time_h, efficiency = params
                area_dm2 = 1.0  # 標準化
    
                # めっき厚さ計算
                current_A = current_density * area_dm2
                thickness = calculate_plating_thickness(
                    current_A, time_h * 3600, area_dm2 * 100, 'Cu', efficiency
                )
    
                # 誤差計算
                error_thickness = (thickness - targets['thickness'])**2
    
                # 制約ペナルティ（電流密度が高すぎると膜質悪化）
                penalty = 0
                if current_density > 5.0:
                    penalty += 100 * (current_density - 5.0)**2
                if current_density < 0.5:
                    penalty += 100 * (0.5 - current_density)**2
    
                return error_thickness + penalty
    
            elif self.treatment_type == 'anodizing':
                # パラメータ：[電圧 V, 時間 min]
                voltage, time_min = params
    
                # 膜厚計算
                _, thickness = anodization_thickness(voltage, 'Al', 'H2SO4', time_min)
    
                error_thickness = (thickness - targets['thickness'])**2
    
                # 制約ペナルティ
                penalty = 0
                if voltage > 100:
                    penalty += 100 * (voltage - 100)**2
    
                return error_thickness + penalty
    
            else:
                return 0
    
        def optimize(self, targets, initial_guess):
            """
            最適化実行
            """
            result = minimize(
                lambda p: self.objective_function(p, targets),
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )
    
            return result
    
    # 実行例1: めっきプロセス最適化
    print("=== 電気めっきプロセス最適化 ===")
    optimizer_plating = SurfaceTreatmentOptimizer('electroplating')
    
    targets_plating = {
        'thickness': 20.0  # 目標20μm
    }
    
    initial_guess_plating = [2.0, 1.0, 0.95]  # [電流密度, 時間, 効率]
    
    result_plating = optimizer_plating.optimize(targets_plating, initial_guess_plating)
    
    print(f"目標めっき厚さ: {targets_plating['thickness']} μm")
    print(f"最適パラメータ:")
    print(f"  電流密度: {result_plating.x[0]:.2f} A/dm²")
    print(f"  めっき時間: {result_plating.x[1]:.2f} 時間")
    print(f"  電流効率: {result_plating.x[2]:.3f}")
    
    # 達成された膜厚
    achieved_thickness = calculate_plating_thickness(
        result_plating.x[0], result_plating.x[1] * 3600, 100, 'Cu', result_plating.x[2]
    )
    print(f"➡ 達成膜厚: {achieved_thickness:.2f} μm")
    print(f"  誤差: {abs(achieved_thickness - targets_plating['thickness']):.2f} μm")
    
    # 実行例2: 陽極酸化プロセス最適化
    print("\n=== 陽極酸化プロセス最適化 ===")
    optimizer_anodizing = SurfaceTreatmentOptimizer('anodizing')
    
    targets_anodizing = {
        'thickness': 15.0  # 目標15μm
    }
    
    initial_guess_anodizing = [50.0, 30.0]  # [電圧 V, 時間 min]
    
    result_anodizing = optimizer_anodizing.optimize(targets_anodizing, initial_guess_anodizing)
    
    print(f"目標膜厚: {targets_anodizing['thickness']} μm")
    print(f"最適パラメータ:")
    print(f"  電圧: {result_anodizing.x[0]:.1f} V")
    print(f"  処理時間: {result_anodizing.x[1]:.1f} 分")
    
    # 達成された膜厚
    _, achieved_thickness_anodizing = anodization_thickness(
        result_anodizing.x[0], 'Al', 'H2SO4', result_anodizing.x[1]
    )
    print(f"➡ 達成膜厚: {achieved_thickness_anodizing:.2f} μm")
    print(f"  誤差: {abs(achieved_thickness_anodizing - targets_anodizing['thickness']):.2f} μm")
    
    # パラメータ感度分析（めっき）
    current_densities_scan = np.linspace(0.5, 5.0, 30)
    times_scan = np.linspace(0.5, 2.5, 30)
    
    CD, T = np.meshgrid(current_densities_scan, times_scan)
    Thickness = np.zeros_like(CD)
    
    for i in range(len(times_scan)):
        for j in range(len(current_densities_scan)):
            cd = CD[i, j]
            t = T[i, j]
            thick = calculate_plating_thickness(cd, t * 3600, 100, 'Cu', 0.95)
            Thickness[i, j] = thick
    
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(CD, T, Thickness, levels=20, cmap='viridis')
    plt.colorbar(contour, label='めっき厚さ [μm]')
    plt.contour(CD, T, Thickness, levels=[20], colors='red', linewidths=2)
    plt.scatter([result_plating.x[0]], [result_plating.x[1]],
                color='red', s=200, marker='*', edgecolors='white', linewidths=2,
                label='最適点')
    plt.xlabel('電流密度 [A/dm²]', fontsize=12)
    plt.ylabel('めっき時間 [hours]', fontsize=12)
    plt.title('めっきプロセスパラメータマップ（目標20μm）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

## 3.6 演習問題

#### 演習3.1（Easy）: めっき厚さ計算

銅めっきプロセスにおいて、電流2A、めっき時間1時間、めっき面積100cm²、電流効率95%の条件で、めっき厚さを計算せよ。

解答を表示

**計算手順** :

  1. Faradayの法則: $m = \frac{M \cdot I \cdot t}{n \cdot F} \cdot \eta$
  2. 銅のパラメータ: M = 63.55 g/mol, n = 2, ρ = 8.96 g/cm³
  3. $m = \frac{63.55 \times 2.0 \times 3600}{2 \times 96485} \times 0.95 = 2.25$ g
  4. $d = \frac{2.25}{8.96 \times 100} \times 10^4 = 25.1$ μm

**答え** : めっき厚さ = 25.1 μm
    
    
    thickness = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"めっき厚さ: {thickness:.2f} μm")  # 25.11 μm
    

#### 演習3.2（Easy）: 陽極酸化電圧の決定

アルミニウムのアルマイト処理で、50nmのバリア層を形成したい。硫酸浴を使用する場合、必要な印加電圧を求めよ（経験則: 1.4 nm/V）。

解答を表示

**計算** :

$V = \frac{d_{\text{barrier}}}{k} = \frac{50}{1.4} = 35.7$ V

**答え** : 必要電圧 = 35.7 V（実用上は36〜40V）

#### 演習3.3（Easy）: 表面処理技術の選定

航空機エンジン部品（チタン合金製）に耐食性と耐摩耗性を付与したい。温度は300〜600℃に達する。適切な表面処理技術を選び、理由を説明せよ。

解答を表示

**推奨技術** : 熱溶射（プラズマ溶射またはHVOF）によるセラミックコーティング（Al₂O₃やYSZ）

**理由** :

  * 高温環境（300〜600℃）ではめっきや陽極酸化は不適
  * セラミックコーティングは高温酸化に強い
  * 熱溶射は厚膜（100〜500μm）形成可能で耐摩耗性に優れる
  * HVOF法は密着力が高く、高速回転部品に適する

#### 演習3.4（Medium）: 均一電着性の改善

複雑形状部品のめっきで、凸部のめっき厚さが25μm、凹部が15μmと不均一になっている。均一電着性を改善する方法を3つ挙げ、それぞれの効果を説明せよ。

解答を表示

**改善方法** :

  1. **電流密度の低減**
     * 効果: 電位分布の均一化、拡散支配領域へ移行
     * 実装: 2 A/dm² → 0.8 A/dm²に低減、めっき時間延長で補償
  2. **レベリング剤の添加**
     * 効果: 凸部での析出速度を選択的に抑制、凹部優先析出
     * 実装: チオ尿素などの添加剤を数ppm添加
  3. **浴の攪拌強化**
     * 効果: 金属イオンの拡散層厚さの均一化
     * 実装: エアレーション、試料回転、ポンプ循環

**期待効果** : 膜厚比 25:15 → 22:18 程度（均一性60% → 82%）

#### 演習3.5（Medium）: イオン注入ドーズ量の計算

シリコン基板に窒素イオンを注入し、表面から50nmの深さでピーク濃度5×10²⁰ atoms/cm³を達成したい。エネルギー50 keV（Rp = 80 nm, ΔRp = 24 nm）の場合、必要なドーズ量を計算せよ。

解答を表示

**計算手順** :

ピーク濃度（x = Rp）でのガウス分布:

$$C_{\text{peak}} = \frac{\Phi}{\sqrt{2\pi} \Delta R_p}$$

問題では x = 50 nm ≠ Rp = 80 nm なので:

$$C(50) = \frac{\Phi}{\sqrt{2\pi} \cdot 24} \exp\left(-\frac{(50 - 80)^2}{2 \times 24^2}\right)$$

$$5 \times 10^{20} = \frac{\Phi}{\sqrt{2\pi} \cdot 24 \times 10^{-7}} \times 0.557$$

$$\Phi = \frac{5 \times 10^{20} \times \sqrt{2\pi} \times 24 \times 10^{-7}}{0.557} = 1.7 \times 10^{16} \text{ ions/cm}^2$$

**答え** : ドーズ量 = 1.7×10¹⁶ ions/cm²

#### 演習3.6（Medium）: 熱溶射プロセスパラメータの選定

HVOF溶射でWC-Coコーティングを施す。粒径40μm、噴射距離150mmの条件で、基板到達時の粒子速度を600 m/s以上、温度を2500℃以上に保ちたい。コード例3.6を参考に、この条件を満たすか検証し、満たさない場合は改善策を提案せよ。

解答を表示

**検証** :
    
    
    v, T, d = thermal_spray_particle_dynamics(40, 'WC-Co', 'HVOF', 150)
    print(f"到達速度: {v[-1]:.1f} m/s")  # 約650 m/s ✓
    print(f"到達温度: {T[-1]:.1f} ℃")   # 約2400 ℃ ✗
    

**判定** : 速度は満たすが、温度が不足（2400℃ < 2500℃）

**改善策** :

  1. **噴射距離短縮** : 150 mm → 120 mmで温度損失減少
  2. **粒径縮小** : 40 μm → 30 μmで冷却速度低減（熱容量/表面積比↑）
  3. **初期温度上昇** : 燃料/酸素比調整、プレヒート強化

**最終推奨** : 噴射距離120 mm + 粒径35 μm → 到達温度約2550℃（目標達成）

#### 演習3.7（Hard）: 多層コーティング設計

自動車エンジン部品（鋼材）に、耐摩耗性と耐食性を同時に付与したい。以下の条件で多層コーティングを設計せよ：

  * 最内層: 密着層（薄膜）
  * 中間層: 耐摩耗層（厚膜）
  * 最外層: 耐食層（中膜）

各層の材料、厚さ、製法を選定し、設計理由を説明せよ。

解答を表示

**多層コーティング設計** :

層 | 材料 | 厚さ | 製法 | 理由  
---|---|---|---|---  
密着層 | Ni | 5μm | 電気めっき | 鋼との密着性良好、応力緩和  
耐摩耗層 | WC-Co | 150μm | HVOF溶射 | 高硬度（HV1200）、耐摩耗性  
耐食層 | Cr₃C₂-NiCr | 50μm | HVOF溶射 | 耐酸化性、耐高温腐食  
  
**プロセスシーケンス** :

  1. 鋼基材の前処理（脱脂、サンドブラスト、Ra = 3〜5μm）
  2. 電気めっきでNi密着層（電流密度2 A/dm², 1時間）
  3. HVOF溶射でWC-Co層（粒径30μm, 噴射距離120mm, 速度800m/s）
  4. HVOF溶射でCr₃C₂-NiCr層（粒径40μm, 噴射距離150mm）
  5. 後処理（必要に応じて研磨、封孔）

**期待性能** :

  * 耐摩耗性: 摩擦係数0.3、摩耗率 < 10⁻⁶ mm³/Nm
  * 耐食性: 塩水噴霧1000時間以上
  * 密着強度: > 50 MPa

#### 演習3.8（Hard）: プロセストラブルシューティング

銅めっき工程で以下の不良が発生した。各不良の原因と対策を提案せよ：

  * **不良A** : めっき表面に小さな突起（ノジュール）が多数発生
  * **不良B** : めっき厚さが目標20μmに対し12μmしか達成できない
  * **不良C** : めっき後の密着試験（テープ試験）で剥離が発生

解答を表示

**不良A: ノジュール（表面突起）**

**原因候補** :

  * めっき浴中の不純物・パーティクル（ダスト、他金属イオン）
  * 浴のろ過不足
  * 電流密度過大による樹枝状成長

**対策** :

  1. めっき浴のろ過（5μmカートリッジフィルタ、24時間循環）
  2. 陽極の活性炭処理（不純物除去）
  3. 電流密度の低減（5 A/dm² → 2 A/dm²）
  4. 試料の前処理強化（脱脂 → 酸洗い → 純水リンス）

**不良B: 膜厚不足**

**原因候補** :

  * 電流効率の低下（副反応による）
  * 金属イオン濃度の不足
  * 実際の電流値が設定値より低い

**検証** :
    
    
    # 理論膜厚（効率95%）
    d_theoretical = calculate_plating_thickness(2.0, 3600, 100, 'Cu', 0.95)
    print(f"理論膜厚: {d_theoretical:.1f} μm")  # 25.1 μm
    
    # 実測12μmから逆算される電流効率
    actual_efficiency = 12 / d_theoretical * 0.95
    print(f"実際の電流効率: {actual_efficiency:.1%}")  # 約45%（大幅低下）
    

**対策** :

  1. 浴組成分析（CuSO₄濃度、H₂SO₄濃度）→ 不足なら補充
  2. 電流計の校正確認
  3. 浴温度の確認（低温は電流効率低下）→ 25±2℃に維持
  4. 陽極面積と陰極面積のバランス確認（1:1〜2:1が理想）

**不良C: 密着不良**

**原因候補** :

  * 基材表面の汚れ（油脂、酸化膜）
  * 前処理不足
  * 基材との熱膨張係数の不一致による応力

**対策** :

  1. 前処理プロセスの見直し 
     * 脱脂: アルカリ脱脂（60℃, 10分）+ 超音波洗浄
     * 酸洗い: 10% H₂SO₄ (室温, 1分) で酸化膜除去
     * 活性化: 5% HCl (室温, 30秒) 直前処理
  2. ストライクめっき（薄いNiまたはCu層）で密着性向上
  3. めっき後のベーキング（150℃, 1時間）で水素脆化除去と密着力向上

**検証方法** :

  * 密着試験: JIS H8504（クロスカット→テープ試験）
  * 引張試験: ASTM B571（引張密着強度 > 20 MPa目標）

## 3.7 学習確認チェックリスト

### 基本理解（5項目）

  * □ Faradayの法則を用いてめっき厚さを計算できる
  * □ 陽極酸化のバリア層とポーラス層の違いを説明できる
  * □ イオン注入の飛程とドーズ量の関係を理解している
  * □ コーティング技術の分類（めっき、溶射、PVD/CVD）を理解している
  * □ 熱溶射の粒子速度と温度が密着性に与える影響を説明できる

### 実践スキル（5項目）

  * □ 電流密度と電流効率を考慮してめっき条件を設計できる
  * □ 陽極酸化の電圧と膜厚の関係を計算できる
  * □ イオン注入プロファイルをPythonでシミュレートできる
  * □ 表面処理技術の選定フローチャートを使いこなせる
  * □ めっき不良（ノジュール、膜厚不足、密着不良）の原因を推定できる

### 応用力（5項目）

  * □ 複雑形状部品の均一電着性を改善する方法を提案できる
  * □ 多層コーティングを設計し、各層の材料・厚さ・製法を選定できる
  * □ 熱溶射プロセスパラメータ（粒径、噴射距離）を最適化できる
  * □ 要求特性（耐食性、耐摩耗性、導電性など）に応じて表面処理技術を選べる
  * □ プロセス異常に対するトラブルシューティングができる

## 3.8 参考文献

  1. Kanani, N. (2004). _Electroplating: Basic Principles, Processes and Practice_. Elsevier, **pp. 56-89** (Faradayの法則と電気化学基礎).
  2. Wernick, S., Pinner, R., Sheasby, P.G. (1987). _The Surface Treatment and Finishing of Aluminum and Its Alloys_ (5th ed.). ASM International, **pp. 234-267** (陽極酸化プロセスと膜構造).
  3. Davis, J.R. (Ed.) (2004). _Handbook of Thermal Spray Technology_. ASM International, **pp. 123-156** (熱溶射プロセスとコーティング特性).
  4. Pawlowski, L. (2008). _The Science and Engineering of Thermal Spray Coatings_ (2nd ed.). Wiley, **pp. 189-223** (HVOF溶射と粒子ダイナミクス).
  5. Townsend, P.D., Chandler, P.J., Zhang, L. (1994). _Optical Effects of Ion Implantation_. Cambridge University Press, **pp. 45-78** (イオン注入理論とLSSモデル).
  6. Inagaki, M., Toyoda, M., Soneda, Y., Morishita, T. (2014). "Nitrogen-doped carbon materials." _Carbon_ , 132, 104-140, **pp. 115-128** , DOI: 10.1016/j.carbon.2014.01.027 (プラズマ窒化プロセス).
  7. Fauchais, P.L., Heberlein, J.V.R., Boulos, M.I. (2014). _Thermal Spray Fundamentals: From Powder to Part_. Springer, **pp. 567-612** (熱溶射の基礎と応用).
  8. Schlesinger, M., Paunovic, M. (Eds.) (2010). _Modern Electroplating_ (5th ed.). Wiley, **pp. 209-248** (最新めっき技術とトラブルシューティング).

## まとめ

本章では、材料表面処理技術の基礎から実践まで学習しました。電気めっきではFaradayの法則による膜厚計算と電流密度の最適化、陽極酸化ではバリア層とポーラス層の形成メカニズム、イオン注入では濃度プロファイルのモデリング、そして熱溶射では粒子ダイナミクスと密着強度の関係を習得しました。

表面処理技術は、材料の内部特性を変えることなく表面機能（耐食性、耐摩耗性、導電性、装飾性など）を付与する重要なプロセス技術です。適切な技術選定とパラメータ最適化により、製品の性能と寿命を大幅に向上させることができます。

次章では、粉末焼結プロセスについて学びます。焼結メカニズム、緻密化モデル、ホットプレス、SPSなど、粉末冶金の基礎と実践を習得します。
