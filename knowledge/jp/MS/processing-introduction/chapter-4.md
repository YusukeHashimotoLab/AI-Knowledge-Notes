---
title: 第4章：薄膜成長プロセス
chapter_title: 第4章：薄膜成長プロセス
subtitle: Sputtering, Evaporation, CVD, Epitaxy
reading_time: 30-40分
difficulty: 中級
code_examples: 7
---

薄膜成長プロセスは、半導体デバイス、光学コーティング、保護膜など、現代材料科学の基盤技術です。この章では、スパッタリング（Sputtering）、真空蒸着（Evaporation）、化学気相成長（CVD）、エピタキシャル成長の原理と実践を学び、Pythonで成膜パラメータの最適化と膜質予測を行います。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ スパッタリングの原理（DCスパッタ、RFスパッタ、マグネトロン配置）を理解する
  * ✅ 真空蒸着法（熱蒸着、電子ビーム蒸着、分子線エピタキシー）の特徴を説明できる
  * ✅ CVD（Chemical Vapor Deposition）の基礎とPECVD、MOCVD、ALDの応用を理解する
  * ✅ エピタキシャル成長における格子整合と歪緩和機構を理解する
  * ✅ Sigmundのスパッタリング収率式とKnudsenの余弦則を計算できる
  * ✅ Pythonで成膜速度、膜厚分布、エピタキシャル臨界厚さをシミュレートできる
  * ✅ 実際のプロセス条件最適化を実践できる

## 4.1 スパッタリング（Sputtering）

### 4.1.1 スパッタリングの原理

スパッタリング（Sputtering）は、高エネルギーイオン（通常はAr+）をターゲット材料に衝突させ、ターゲット原子を弾き飛ばして基板上に堆積させる物理的気相成長（PVD: Physical Vapor Deposition）法です。

**スパッタリング収率（Sputtering Yield）** は、入射イオン1個あたりに放出されるターゲット原子数で定義されます：

$$ Y = \frac{\text{放出原子数}}{\text{入射イオン数}} $$ 

**Sigmundの理論式** （低エネルギー領域）：

$$ Y = \frac{3}{4\pi^2} \frac{\alpha S_n(E)}{U_0 N} $$ 

  * $\alpha$：材料依存定数（0.15-0.3）
  * $S_n(E)$：核阻止能（Nuclear stopping power）
  * $U_0$：表面結合エネルギー（Sublimation energy）
  * $N$：ターゲット原子密度

**実用的な簡略式** （エネルギー500 eV - 5 keV）：

$$ Y \approx A \frac{E - E_{\text{th}}}{U_0} $$ 

  * $A$：材料定数
  * $E$：入射イオンエネルギー
  * $E_{\text{th}}$：閾値エネルギー（通常20-50 eV）

### 4.1.2 DCスパッタとRFスパッタ

項目 | DCスパッタ（直流） | RFスパッタ（高周波）  
---|---|---  
**電源** | DC（直流、-300 V ~ -1000 V） | RF（13.56 MHz、数100 V）  
**ターゲット材料** | 導電性材料（金属） | 導電性・絶縁性材料（酸化物、窒化物）  
**チャージアップ対策** | 不要（電荷が流れる） | RF周期で電荷中和  
**成膜速度** | 速い（1-10 nm/s） | やや遅い（0.5-5 nm/s）  
**用途** | 金属薄膜（Al, Cu, Ti） | 酸化物（ITO, SiO2）、窒化物（Si3N4）  
  
### 4.1.3 マグネトロンスパッタリング

マグネトロン配置では、ターゲット背後に磁石を配置し、電子を磁場で閉じ込めることでプラズマ密度を向上させます。これにより：

  * 成膜速度が5-10倍向上
  * 低圧動作が可能（0.1-1 Pa）→ 膜質向上
  * 基板へのイオン衝撃低減 → ダメージ軽減

    
    
    ```mermaid
    flowchart TD
        A[Arガス導入0.1-1 Pa] --> B[DCまたはRF電源300-1000 V]
        B --> C[プラズマ生成Ar+イオン化]
        C --> D[磁場によるe-閉じ込めマグネトロン効果]
        D --> E[Ar+イオンがターゲット衝突]
        E --> F[ターゲット原子スパッタ放出]
        F --> G[基板への堆積薄膜成長]
    
        style A fill:#99ccff,stroke:#0066cc
        style C fill:#ffeb99,stroke:#ffa500
        style G fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
    ```

#### コード例4-1: スパッタリング収率の計算（Sigmund理論）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def sigmund_yield(E, M_ion, M_target, Z_ion, Z_target, U0, alpha=0.2):
        """
        Sigmundのスパッタリング収率式（簡略版）
    
        Parameters
        ----------
        E : float or ndarray
            入射イオンエネルギー [eV]
        M_ion : float
            イオン質量 [amu]
        M_target : float
            ターゲット原子質量 [amu]
        Z_ion : int
            イオン原子番号
        Z_target : int
            ターゲット原子番号
        U0 : float
            表面結合エネルギー [eV]
        alpha : float
            材料定数（0.15-0.3）
    
        Returns
        -------
        Y : ndarray
            スパッタリング収率
        """
        # Lindhard-Scharff reduced energy
        epsilon = 32.53 * M_target * E / (Z_ion * Z_target * (M_ion + M_target) *
                                           (Z_ion**(2/3) + Z_target**(2/3))**(1/2))
    
        # Nuclear stopping power (Lindhard-Scharff)
        # 簡略式: Sn(epsilon) ≈ epsilon / (1 + 0.3*epsilon^0.6)
        Sn_reduced = epsilon / (1 + 0.3 * epsilon**0.6)
    
        # Sigmund yield
        Y = alpha * Sn_reduced * 4 * M_ion * M_target / ((M_ion + M_target)**2 * U0)
    
        # 閾値エネルギー以下はゼロ
        E_th = U0 * (1 + M_target/(5*M_ion))**2
        Y = np.where(E > E_th, Y, 0)
    
        return Y
    
    # ArイオンによるSi, Cu, Auのスパッタリング
    E_range = np.linspace(50, 2000, 200)  # [eV]
    
    # Arイオン
    M_Ar = 40  # [amu]
    Z_Ar = 18
    
    # ターゲット材料
    targets = {
        'Si': {'M': 28, 'Z': 14, 'U0': 4.7, 'color': 'blue'},
        'Cu': {'M': 64, 'Z': 29, 'U0': 3.5, 'color': 'orange'},
        'Au': {'M': 197, 'Z': 79, 'U0': 3.8, 'color': 'red'}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: スパッタリング収率 vs エネルギー
    for name, params in targets.items():
        Y = sigmund_yield(E_range, M_Ar, params['M'], Z_Ar, params['Z'], params['U0'])
        ax1.plot(E_range, Y, linewidth=2, color=params['color'], label=name)
    
    ax1.set_xlabel('Ion Energy [eV]', fontsize=12)
    ax1.set_ylabel('Sputtering Yield [atoms/ion]', fontsize=12)
    ax1.set_title('Sputtering Yield vs Ion Energy\n(Ar+ ions)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 2000)
    
    # 右図: 典型的なスパッタ条件での比較（500 eV）
    E_fixed = 500  # [eV]
    materials = list(targets.keys())
    yields = [sigmund_yield(E_fixed, M_Ar, targets[m]['M'], Z_Ar,
                            targets[m]['Z'], targets[m]['U0']) for m in materials]
    
    bars = ax2.bar(materials, yields, color=[targets[m]['color'] for m in materials],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Sputtering Yield [atoms/ion]', fontsize=12)
    ax2.set_title(f'Sputtering Yield at {E_fixed} eV\n(Typical DC Sputtering)',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, y_val in zip(bars, yields):
        ax2.text(bar.get_x() + bar.get_width()/2, y_val + 0.1,
                 f'{y_val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("スパッタリング収率の傾向:")
    print(f"  - Cuが最も高い収率（中程度の質量、低い結合エネルギー）")
    print(f"  - Siは低い収率（軽い、高い結合エネルギー）")
    print(f"  - Auは中程度（重いが高い結合エネルギー）")
    

### 4.1.4 成膜速度の計算

スパッタリング成膜速度 $R_{\text{dep}}$ は次式で表されます：

$$ R_{\text{dep}} = \frac{Y \cdot J_{\text{ion}} \cdot M}{N_A \cdot \rho \cdot e} $$ 

  * $Y$：スパッタリング収率
  * $J_{\text{ion}}$：イオン電流密度 [A/cm²]
  * $M$：ターゲット原子のモル質量 [g/mol]
  * $N_A$：アボガドロ数
  * $\rho$：ターゲット密度 [g/cm³]
  * $e$：電気素量

#### コード例4-2: スパッタリング成膜速度とパワー・圧力依存性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def deposition_rate(power, pressure, Y=2.5, target_area=100, target_distance=5):
        """
        スパッタリング成膜速度の計算（経験的モデル）
    
        Parameters
        ----------
        power : float or ndarray
            スパッタパワー [W]
        pressure : float or ndarray
            Ar圧力 [Pa]
        Y : float
            スパッタリング収率 [atoms/ion]
        target_area : float
            ターゲット面積 [cm^2]
        target_distance : float
            ターゲット-基板距離 [cm]
    
        Returns
        -------
        rate : ndarray
            成膜速度 [nm/min]
        """
        # イオン電流密度の推定（経験式）
        # J_ion ≈ power / (voltage * target_area)
        # 典型的なDCスパッタ電圧: 500 V
        voltage = 500  # [V]
        J_ion = power / (voltage * target_area)  # [A/cm^2]
    
        # 圧力依存性（平均自由行程効果）
        # 低圧: 散乱が少なく効率的、高圧: 散乱で効率低下
        pressure_factor = 1.0 / (1 + pressure / 0.5)  # 0.5 Paで半減
    
        # 成膜速度（簡略モデル）
        # Cu（M=63.5 g/mol, ρ=8.96 g/cm^3）を想定
        M = 63.5
        rho = 8.96
        e = 1.60218e-19
        N_A = 6.022e23
    
        # [nm/s]
        rate_nm_s = (Y * J_ion * M * 1e7) / (N_A * rho * e) * pressure_factor
    
        # [nm/min] に変換
        rate = rate_nm_s * 60
    
        return rate
    
    # パワー依存性（圧力固定）
    power_range = np.linspace(50, 500, 50)
    pressure_fixed = 0.3  # [Pa]
    
    rate_vs_power = deposition_rate(power_range, pressure_fixed)
    
    # 圧力依存性（パワー固定）
    pressure_range = np.linspace(0.1, 2.0, 50)
    power_fixed = 200  # [W]
    
    rate_vs_pressure = deposition_rate(power_fixed, pressure_range)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: パワー依存性
    ax1.plot(power_range, rate_vs_power, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Sputtering Power [W]', fontsize=12)
    ax1.set_ylabel('Deposition Rate [nm/min]', fontsize=12)
    ax1.set_title('Deposition Rate vs Power\n(Ar pressure = 0.3 Pa)',
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 線形フィット
    coeffs = np.polyfit(power_range, rate_vs_power, 1)
    ax1.plot(power_range, np.poly1d(coeffs)(power_range), 'r--', linewidth=2,
             label=f'Linear fit: {coeffs[0]:.2f}·P + {coeffs[1]:.1f}')
    ax1.legend(fontsize=10)
    
    # 右図: 圧力依存性
    ax2.plot(pressure_range, rate_vs_pressure, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Ar Pressure [Pa]', fontsize=12)
    ax2.set_ylabel('Deposition Rate [nm/min]', fontsize=12)
    ax2.set_title('Deposition Rate vs Pressure\n(Power = 200 W)',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 最適圧力を示す
    optimal_p = pressure_range[np.argmax(rate_vs_pressure)]
    ax2.axvline(optimal_p, color='red', linestyle='--', linewidth=2,
                label=f'Optimal: {optimal_p:.2f} Pa')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"パワー依存性: ほぼ線形（200 W → 400 W で2倍）")
    print(f"圧力依存性: 最適値存在（{optimal_p:.2f} Pa）")
    print(f"  - 低圧過ぎ: プラズマ不安定")
    print(f"  - 高圧過ぎ: ガス散乱で効率低下")
    

## 4.2 真空蒸着（Vacuum Evaporation）

### 4.2.1 熱蒸着（Thermal Evaporation）

熱蒸着は、抵抗加熱または電子ビームで材料を加熱し、蒸発させて基板に堆積させる方法です。

**蒸気圧とClausiusClapeyron式** ：

$$ P(T) = P_0 \exp\left(-\frac{\Delta H_{\text{vap}}}{R T}\right) $$ 

  * $P(T)$：温度$T$での蒸気圧
  * $\Delta H_{\text{vap}}$：蒸発エンタルピー
  * $R$：気体定数

**実用的には** 、$10^{-2}$ Pa以上の蒸気圧が必要（成膜速度0.1 nm/s以上）。

### 4.2.2 Knudsenの余弦則

蒸発源からの原子フラックス分布は、**Knudsenの余弦則** に従います：

$$ \Phi(\theta) = \Phi_0 \cos(\theta) $$ 

  * $\Phi(\theta)$：角度$\theta$方向のフラックス
  * $\Phi_0$：法線方向（$\theta=0$）のフラックス

これにより、基板上の膜厚分布が不均一になります（中心部が厚く、周辺部が薄い）。

#### コード例4-3: 熱蒸着フラックス分布（Knudsen余弦則）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def knudsen_flux_distribution(x, y, source_x=0, source_y=0, source_z=-10, flux0=1.0):
        """
        Knudsenの余弦則による蒸着フラックス分布
    
        Parameters
        ----------
        x, y : ndarray
            基板上の座標 [cm]
        source_x, source_y, source_z : float
            蒸発源の位置 [cm]（z < 0: 基板下方）
        flux0 : float
            法線方向の基準フラックス
    
        Returns
        -------
        flux : ndarray
            各点でのフラックス
        """
        # 蒸発源から各点への距離と角度
        dx = x - source_x
        dy = y - source_y
        dz = 0 - source_z  # 基板はz=0
    
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_theta = dz / r
    
        # Knudsenの余弦則: Φ(θ) = Φ0 * cos(θ) / r^2
        flux = flux0 * cos_theta / r**2
    
        return flux
    
    # 基板グリッド（10 cm x 10 cm）
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 蒸発源は中心の10 cm下
    flux = knudsen_flux_distribution(X, Y, source_x=0, source_y=0, source_z=-10, flux0=100)
    
    # 膜厚に変換（フラックス × 時間）
    time = 60  # [s]
    thickness = flux * time  # [任意単位]
    
    # 可視化
    fig = plt.figure(figsize=(16, 6))
    
    # 左図: 2Dフラックス分布
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.contourf(X, Y, flux, levels=20, cmap='hot')
    ax1.contour(X, Y, flux, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax1.set_xlabel('x [cm]', fontsize=11)
    ax1.set_ylabel('y [cm]', fontsize=11)
    ax1.set_title('Flux Distribution (Knudsen Cosine Law)\nSource at (0, 0, -10 cm)',
                  fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Flux [a.u.]', fontsize=10)
    
    # 中央図: 3D膜厚分布
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, thickness, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('x [cm]', fontsize=10)
    ax2.set_ylabel('y [cm]', fontsize=10)
    ax2.set_zlabel('Thickness [a.u.]', fontsize=10)
    ax2.set_title('Film Thickness Distribution\n(60 s deposition)', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    
    # 右図: 中心軸方向のプロファイル
    ax3 = fig.add_subplot(1, 3, 3)
    center_profile = thickness[50, :]  # y=0のライン
    ax3.plot(x, center_profile, 'b-', linewidth=2)
    ax3.set_xlabel('x [cm]', fontsize=12)
    ax3.set_ylabel('Thickness [a.u.]', fontsize=12)
    ax3.set_title('Thickness Profile along Center Line\n(y = 0)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 均一性評価
    thickness_center = thickness[50, 50]
    thickness_edge = thickness[50, 0]
    uniformity = (thickness_center - thickness_edge) / thickness_center * 100
    
    ax3.axhline(thickness_center, color='red', linestyle='--', linewidth=1.5,
                label=f'Center: {thickness_center:.1f}')
    ax3.axhline(thickness_edge, color='green', linestyle='--', linewidth=1.5,
                label=f'Edge: {thickness_edge:.1f}')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"膜厚均一性: {uniformity:.1f}% 差（中心-端）")
    print(f"改善策:")
    print(f"  - 基板回転（回転対称性で均一化）")
    print(f"  - 複数蒸発源配置")
    print(f"  - マスク使用（選択成膜）")
    

### 4.2.3 電子ビーム蒸着と分子線エピタキシー（MBE）

項目 | 熱蒸着 | 電子ビーム蒸着 | MBE  
---|---|---|---  
**加熱方法** | 抵抗加熱（ボート、タングステンワイヤー） | 電子ビーム照射（局所加熱） | Knudsenセル（個別加熱）  
**到達温度** | ～1500°C | ～3000°C | ～1500°C  
**適用材料** | 低融点金属（Al, Ag, Au） | 高融点材料（Ti, W, SiO2） | 半導体（GaAs, InP, Si/Ge）  
**成膜速度** | 0.1-10 nm/s | 0.5-50 nm/s | 0.01-1 nm/s（原子層制御）  
**真空度** | 10-3-10-5 Pa | 10-4-10-6 Pa | 10-8-10-10 Pa（超高真空）  
**膜質** | 多結晶、アモルファス | 多結晶 | 単結晶エピタキシー  
  
## 4.3 化学気相成長（CVD: Chemical Vapor Deposition）

### 4.3.1 CVDの基礎

CVDは、ガス状の原料（前駆体）を基板表面で化学反応させ、固体薄膜を成長させる方法です。

**CVDプロセスの基本ステップ** ：

  1. 原料ガスの輸送（拡散）
  2. 基板表面への吸着
  3. 表面反応（熱分解、還元、酸化）
  4. 副生成物の脱離
  5. 副生成物の排出

**成長速度の律速段階** ：

$$ R_{\text{growth}} = \min\left(R_{\text{diffusion}}, R_{\text{reaction}}\right) $$ 

  * **低温領域** ：表面反応律速（Arrhenius依存性）
  * **高温領域** ：拡散律速（温度依存性小）

**Arrheniusの式** ：

$$ R = A \exp\left(-\frac{E_a}{k_B T}\right) $$ 

  * $E_a$：活性化エネルギー
  * $k_B$：ボルツマン定数
  * $T$：温度

#### コード例4-4: CVD成長速度のArrhenius温度依存性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def cvd_growth_rate(T, Ea, A, regime='reaction'):
        """
        CVD成長速度のArrhenius式
    
        Parameters
        ----------
        T : float or ndarray
            温度 [K]
        Ea : float
            活性化エネルギー [eV]
        A : float
            前指数因子 [nm/min]
        regime : str
            'reaction' (反応律速) or 'diffusion' (拡散律速)
    
        Returns
        -------
        rate : ndarray
            成長速度 [nm/min]
        """
        kB = 8.617e-5  # [eV/K] ボルツマン定数
    
        if regime == 'reaction':
            # 反応律速: Arrhenius式
            rate = A * np.exp(-Ea / (kB * T))
        elif regime == 'diffusion':
            # 拡散律速: 温度依存性小（T^0.5程度）
            rate = A * (T / 1000)**0.5
    
        return rate
    
    # 温度範囲（300-1000°C）
    T_celsius = np.linspace(300, 1000, 100)
    T_kelvin = T_celsius + 273.15
    
    # パラメータ（SiO2 CVD from SiH4 + O2を想定）
    Ea = 1.5  # [eV]
    A_reaction = 1e6  # [nm/min]
    A_diffusion = 100  # [nm/min]
    
    # 反応律速と拡散律速の成長速度
    rate_reaction = cvd_growth_rate(T_kelvin, Ea, A_reaction, regime='reaction')
    rate_diffusion = cvd_growth_rate(T_kelvin, Ea, A_diffusion, regime='diffusion')
    
    # 実際の成長速度（律速段階の小さい方）
    rate_actual = np.minimum(rate_reaction, rate_diffusion)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: Arrheniusプロット（log(rate) vs 1/T）
    ax1.semilogy(1000/T_kelvin, rate_reaction, 'b-', linewidth=2, label='Reaction-limited')
    ax1.semilogy(1000/T_kelvin, rate_diffusion, 'r-', linewidth=2, label='Diffusion-limited')
    ax1.semilogy(1000/T_kelvin, rate_actual, 'k--', linewidth=2.5, label='Actual (minimum)')
    
    ax1.set_xlabel('1000/T [K⁻¹]', fontsize=12)
    ax1.set_ylabel('Growth Rate [nm/min]', fontsize=12)
    ax1.set_title('Arrhenius Plot: CVD Growth Rate\n(SiO₂ from SiH₄ + O₂)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower left')
    ax1.grid(alpha=0.3, which='both')
    ax1.invert_xaxis()
    
    # 上軸に温度表示
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    temp_ticks = [400, 500, 600, 700, 800, 900]
    ax1_top.set_xticks([1000/(t+273.15) for t in temp_ticks])
    ax1_top.set_xticklabels([f'{t}°C' for t in temp_ticks], fontsize=10)
    
    # 右図: 線形スケール
    ax2.plot(T_celsius, rate_reaction, 'b-', linewidth=2, label='Reaction-limited')
    ax2.plot(T_celsius, rate_diffusion, 'r-', linewidth=2, label='Diffusion-limited')
    ax2.plot(T_celsius, rate_actual, 'k--', linewidth=2.5, label='Actual rate')
    
    # 領域を塗り分け
    transition_idx = np.argmin(np.abs(rate_reaction - rate_diffusion))
    T_transition = T_celsius[transition_idx]
    
    ax2.axvspan(300, T_transition, alpha=0.2, color='blue', label='Reaction-limited regime')
    ax2.axvspan(T_transition, 1000, alpha=0.2, color='red', label='Diffusion-limited regime')
    ax2.axvline(T_transition, color='green', linestyle=':', linewidth=2,
                label=f'Transition: {T_transition:.0f}°C')
    
    ax2.set_xlabel('Temperature [°C]', fontsize=12)
    ax2.set_ylabel('Growth Rate [nm/min]', fontsize=12)
    ax2.set_title('CVD Growth Rate vs Temperature\n(Linear Scale)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"遷移温度: {T_transition:.0f}°C")
    print(f"低温領域（<{T_transition:.0f}°C）: 反応律速 → 温度に強く依存")
    print(f"高温領域（>{T_transition:.0f}°C）: 拡散律速 → 温度依存性小")
    print(f"最適成膜温度: 遷移温度付近（速度と均一性のバランス）")
    

### 4.3.2 PECVD（Plasma-Enhanced CVD）

PECVDは、プラズマを利用して低温（200-400°C）でCVD反応を促進します。

**PECVDの利点** ：

  * 低温成膜（樹脂基板、有機材料への適用可能）
  * 成膜速度向上（プラズマが反応を加速）
  * 膜質制御（イオン衝撃によるデンシフィケーション）

**用途** ：Si3N4、SiO2、a-Si、DLC（Diamond-Like Carbon）

### 4.3.3 ALD（Atomic Layer Deposition）

ALDは、原料ガスを交互にパルス供給し、自己制御的に単原子層ずつ成長させる究極的精密技術です。

**ALDサイクル** ：

  1. 前駆体Aパルス → 表面飽和吸着
  2. パージ（Ar, N2）
  3. 前駆体Bパルス → 化学反応で1層形成
  4. パージ

**特徴** ：

  * 原子層精度の膜厚制御（0.1 nm/cycle）
  * 完全なコンフォーマルコーティング（高アスペクト比構造）
  * 低温成膜（100-300°C）
  * 成膜速度が遅い（0.01-0.1 nm/s）

    
    
    ```mermaid
    flowchart LR
        A[前駆体Aパルス表面飽和吸着] --> B[パージ余剰ガス除去]
        B --> C[前駆体Bパルス反応・1層形成]
        C --> D[パージ副生成物除去]
        D --> E{目標膜厚到達?}
        E -->|No| A
        E -->|Yes| F[成膜完了]
    
        style A fill:#99ccff,stroke:#0066cc
        style C fill:#ffeb99,stroke:#ffa500
        style F fill:#f5576c,stroke:#f093fb,stroke-width:2px,color:#fff
    ```

## 4.4 エピタキシャル成長

### 4.4.1 エピタキシーの定義と種類

エピタキシャル成長（Epitaxy）は、単結晶基板上に結晶方位を揃えて単結晶薄膜を成長させる技術です。

**エピタキシーの種類** ：

  * **ホモエピタキシー** ：同一材料（Si基板上にSi成長）
  * **ヘテロエピタキシー** ：異種材料（GaAs上にAlGaAs成長）

### 4.4.2 格子整合と臨界厚さ

ヘテロエピタキシーでは、基板と薄膜の格子定数のミスマッチ（格子不整合）が重要です：

$$ f = \frac{a_{\text{film}} - a_{\text{sub}}}{a_{\text{sub}}} $$ 

  * $a_{\text{film}}$：薄膜の格子定数
  * $a_{\text{sub}}$：基板の格子定数
  * $f$：格子不整合度（mismatch）

**臨界厚さ（Critical Thickness）$h_c$** ：歪みエネルギーが転位生成エネルギーを超える膜厚

**Matthews-Blakesleeの式** ：

$$ h_c = \frac{b}{4\pi f(1+\nu)} \left[\ln\left(\frac{h_c}{b}\right) + 1\right] $$ 

  * $b$：バーガースベクトル（格子定数程度）
  * $\nu$：ポアソン比

**実用的な簡略式** ：

$$ h_c \approx \frac{a}{2\pi f} $$ 

#### コード例4-5: エピタキシャル臨界厚さの計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def critical_thickness(mismatch, a=5.65, nu=0.3):
        """
        エピタキシャル臨界厚さ（Matthews-Blakeslee式の簡略版）
    
        Parameters
        ----------
        mismatch : float or ndarray
            格子不整合度 f = (a_film - a_sub) / a_sub
        a : float
            格子定数 [Å]
        nu : float
            ポアソン比
    
        Returns
        -------
        h_c : ndarray
            臨界厚さ [nm]
        """
        # 簡略式: h_c ≈ a / (2π f)
        # Matthews-Blakesleeの完全な式は反復計算が必要だが、
        # 実用的にはこの近似で十分
    
        h_c = a / (2 * np.pi * np.abs(mismatch)) / 10  # [nm]
    
        return h_c
    
    def relaxation_fraction(thickness, h_c):
        """
        歪緩和率（経験的モデル）
    
        Parameters
        ----------
        thickness : float or ndarray
            薄膜厚さ [nm]
        h_c : float
            臨界厚さ [nm]
    
        Returns
        -------
        relaxation : ndarray
            歪緩和率（0-1）
        """
        # 臨界厚さ以下: 完全弾性歪み（緩和0）
        # 臨界厚さ以上: 転位導入により緩和
    
        relaxation = 1 - np.exp(-(thickness - h_c) / h_c)
        relaxation = np.where(thickness < h_c, 0, relaxation)
        relaxation = np.clip(relaxation, 0, 1)
    
        return relaxation
    
    # 代表的なヘテロ系の格子不整合
    hetero_systems = {
        'GaAs/Si': {'f': 0.04, 'a': 5.65, 'color': 'blue'},
        'InP/GaAs': {'f': 0.038, 'a': 5.87, 'color': 'green'},
        'SiGe/Si': {'f': 0.02, 'a': 5.43, 'color': 'orange'},  # Ge 50%
        'AlN/GaN': {'f': 0.024, 'a': 4.98, 'color': 'red'}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 臨界厚さ vs 格子不整合
    mismatch_range = np.linspace(0.001, 0.1, 100)
    h_c_range = critical_thickness(mismatch_range, a=5.5)
    
    ax1.loglog(mismatch_range * 100, h_c_range, 'k-', linewidth=2.5, label='Matthews-Blakeslee')
    
    # 各系をプロット
    for name, params in hetero_systems.items():
        h_c = critical_thickness(params['f'], a=params['a'])
        ax1.loglog(params['f'] * 100, h_c, 'o', markersize=10,
                  color=params['color'], label=name)
    
    ax1.set_xlabel('Lattice Mismatch [%]', fontsize=12)
    ax1.set_ylabel('Critical Thickness [nm]', fontsize=12)
    ax1.set_title('Critical Thickness vs Lattice Mismatch\n(Heteroepitaxy)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3, which='both')
    
    # 右図: SiGe/Siの歪緩和（膜厚依存性）
    thickness_range = np.logspace(0, 3, 100)  # [nm]
    sige_params = hetero_systems['SiGe/Si']
    h_c_sige = critical_thickness(sige_params['f'], a=sige_params['a'])
    
    relaxation = relaxation_fraction(thickness_range, h_c_sige)
    
    ax2.semilogx(thickness_range, relaxation * 100, linewidth=2.5, color='orange')
    ax2.axvline(h_c_sige, color='red', linestyle='--', linewidth=2,
                label=f'Critical thickness: {h_c_sige:.1f} nm')
    ax2.axhspan(0, 10, alpha=0.2, color='green', label='Pseudomorphic (<10% relaxation)')
    ax2.axhspan(90, 100, alpha=0.2, color='red', label='Fully relaxed (>90%)')
    
    ax2.set_xlabel('Film Thickness [nm]', fontsize=12)
    ax2.set_ylabel('Strain Relaxation [%]', fontsize=12)
    ax2.set_title('Strain Relaxation in SiGe/Si\n(50% Ge, f = 2%)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3, which='both')
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.show()
    
    print("臨界厚さの傾向:")
    for name, params in hetero_systems.items():
        h_c = critical_thickness(params['f'], a=params['a'])
        print(f"  {name}: f = {params['f']*100:.1f}%, h_c = {h_c:.1f} nm")
    print("\n設計指針:")
    print("  - 臨界厚さ以下: 歪みエピタキシー（高品質だが膜厚制限）")
    print("  - 臨界厚さ以上: 転位緩和（膜厚自由だが欠陥増加）")
    print("  - 緩衝層（グレーデッド SiGe など）で臨界厚さ拡大可能")
    

### 4.4.3 成長モード

エピタキシャル成長は、表面エネルギーと格子不整合により3つのモードに分類されます：

  * **Frank-van der Merwe（FM）モード** ：層状成長（完全濡れ）
  * **Volmer-Weber（VW）モード** ：島状成長（非濡れ）
  * **Stranski-Krastanov（SK）モード** ：層状成長後に島状遷移（中間）

## 4.5 統合例：薄膜プロセス最適化シミュレーション

#### コード例4-6: 膜厚分布の最適化（スパッタ+基板回転）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    def sputtering_thickness_with_rotation(r_substrate, theta_substrate,
                                            source_position, rotation_angle=0):
        """
        基板回転を考慮したスパッタ膜厚分布
    
        Parameters
        ----------
        r_substrate, theta_substrate : ndarray
            基板上の極座標 [cm], [rad]
        source_position : tuple
            スパッタソース位置 (r, theta) [cm], [rad]
        rotation_angle : float
            基板回転角 [rad]
    
        Returns
        -------
        thickness : ndarray
            膜厚分布
        """
        # 回転後の基板座標
        theta_rotated = theta_substrate - rotation_angle
    
        # スパッタソースからの距離と角度
        x_sub = r_substrate * np.cos(theta_rotated)
        y_sub = r_substrate * np.sin(theta_rotated)
    
        x_src, y_src = source_position
    
        dx = x_sub - x_src
        dy = y_sub - y_src
        dz = 10  # ソースは10 cm下
    
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_angle = dz / r
    
        # スパッタフラックス（1/r^2 と cos依存性）
        flux = 100 * cos_angle / r**2
    
        return flux
    
    # 基板グリッド（極座標）
    r = np.linspace(0, 5, 100)
    theta = np.linspace(0, 2*np.pi, 200)
    R, Theta = np.meshgrid(r, theta)
    
    # デカルト座標変換（可視化用）
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # ソース位置（中心から2 cmオフセット）
    source_pos = (2, 0)
    
    # 回転なしの場合
    thickness_no_rotation = sputtering_thickness_with_rotation(R, Theta, source_pos, rotation_angle=0)
    
    # 回転あり（10回転分を積算）
    num_rotations = 10
    thickness_with_rotation = np.zeros_like(R)
    
    for i in range(num_rotations):
        rotation_angle = 2 * np.pi * i / num_rotations
        thickness_with_rotation += sputtering_thickness_with_rotation(R, Theta, source_pos,
                                                                       rotation_angle=rotation_angle)
    
    thickness_with_rotation /= num_rotations
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 左図: 回転なし
    im1 = axes[0].contourf(X, Y, thickness_no_rotation, levels=20, cmap='viridis')
    axes[0].scatter(*source_pos, color='red', s=200, marker='x', linewidths=3, label='Sputter Source')
    axes[0].set_xlabel('x [cm]', fontsize=11)
    axes[0].set_ylabel('y [cm]', fontsize=11)
    axes[0].set_title('Without Rotation\n(Highly non-uniform)', fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=10)
    plt.colorbar(im1, ax=axes[0], label='Thickness [a.u.]')
    
    # 中央図: 回転あり
    im2 = axes[1].contourf(X, Y, thickness_with_rotation, levels=20, cmap='viridis')
    axes[1].scatter(*source_pos, color='red', s=200, marker='x', linewidths=3, label='Sputter Source')
    axes[1].set_xlabel('x [cm]', fontsize=11)
    axes[1].set_ylabel('y [cm]', fontsize=11)
    axes[1].set_title('With Rotation (10 steps)\n(Much improved uniformity)', fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=10)
    plt.colorbar(im2, ax=axes[1], label='Thickness [a.u.]')
    
    # 右図: 半径方向プロファイル比較
    r_profile = r
    thickness_no_rot_profile = thickness_no_rotation[:, 0]
    thickness_rot_profile = np.mean(thickness_with_rotation, axis=0)
    
    axes[2].plot(r_profile, thickness_no_rot_profile, 'b-', linewidth=2,
                marker='o', markersize=4, label='No rotation')
    axes[2].plot(r_profile, thickness_rot_profile, 'r-', linewidth=2,
                marker='s', markersize=4, label='With rotation')
    
    axes[2].set_xlabel('Radius [cm]', fontsize=12)
    axes[2].set_ylabel('Thickness [a.u.]', fontsize=12)
    axes[2].set_title('Radial Thickness Profile', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 均一性評価
    uniformity_no_rot = (np.max(thickness_no_rotation) - np.min(thickness_no_rotation)) / np.mean(thickness_no_rotation) * 100
    uniformity_rot = (np.max(thickness_with_rotation) - np.min(thickness_with_rotation)) / np.mean(thickness_with_rotation) * 100
    
    print(f"膜厚均一性（最大-最小）/平均:")
    print(f"  回転なし: {uniformity_no_rot:.1f}%")
    print(f"  回転あり: {uniformity_rot:.1f}%")
    print(f"改善率: {(1 - uniformity_rot/uniformity_no_rot)*100:.1f}%")
    

#### コード例4-7: 完全統合シミュレーション（多パラメータ最適化）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    def film_quality_metric(params, target_thickness=100, target_rate=5):
        """
        薄膜プロセスの総合品質評価関数
    
        Parameters
        ----------
        params : array
            [power, pressure, temperature, distance]
            - power [W]
            - pressure [Pa]
            - temperature [°C]
            - distance [cm]
        target_thickness : float
            目標膜厚 [nm]
        target_rate : float
            目標成膜速度 [nm/min]
    
        Returns
        -------
        quality : float
            品質スコア（低いほど良い）
        """
        power, pressure, temperature, distance = params
    
        # 物理モデル（簡略版）
    
        # 1. 成膜速度（スパッタモデル）
        Y = 2.5 * (power / 200)**0.8  # スパッタリング収率
        rate = Y * power / (pressure * distance**2) * 0.1  # [nm/min]
    
        # 2. 膜厚均一性（距離と圧力に依存）
        uniformity = 1 / (1 + (distance - 5)**2 / 10) * (1 - np.abs(pressure - 0.5) / 2)
    
        # 3. 膜質（温度と圧力に依存）
        # 低温: アモルファス、高温: 結晶化
        crystallinity = 1 / (1 + np.exp(-(temperature - 400) / 50))
    
        # 低圧: 高密度、高圧: 多孔質
        density = 1 / (1 + pressure / 0.5)
    
        film_quality = crystallinity * density
    
        # 4. プロセス安定性（圧力範囲）
        stability = 1 if 0.2 < pressure < 1.0 else 0.5
    
        # 総合品質スコア（ペナルティ関数）
        penalty = 0
    
        # 成膜速度のペナルティ
        penalty += ((rate - target_rate) / target_rate)**2 * 100
    
        # 均一性のペナルティ（高いほど良いのでマイナス）
        penalty += (1 - uniformity)**2 * 50
    
        # 膜質のペナルティ
        penalty += (1 - film_quality)**2 * 50
    
        # 安定性のペナルティ
        penalty += (1 - stability) * 100
    
        # パラメータ範囲外ペナルティ
        if not (50 <= power <= 500):
            penalty += 1000
        if not (0.1 <= pressure <= 2.0):
            penalty += 1000
        if not (200 <= temperature <= 600):
            penalty += 1000
        if not (3 <= distance <= 10):
            penalty += 1000
    
        return penalty
    
    # 最適化実行
    initial_guess = [200, 0.5, 400, 5]  # [power, pressure, temperature, distance]
    
    # 境界条件
    bounds = [(50, 500), (0.1, 2.0), (200, 600), (3, 10)]
    
    result = minimize(film_quality_metric, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    optimal_params = result.x
    optimal_quality = result.fun
    
    print("=" * 60)
    print("薄膜プロセス最適化結果")
    print("=" * 60)
    print(f"最適パラメータ:")
    print(f"  スパッタパワー: {optimal_params[0]:.1f} W")
    print(f"  Ar圧力: {optimal_params[1]:.3f} Pa")
    print(f"  基板温度: {optimal_params[2]:.1f} °C")
    print(f"  ターゲット-基板距離: {optimal_params[3]:.1f} cm")
    print(f"\n品質スコア: {optimal_quality:.2f}")
    print(f"最適化成功: {result.success}")
    print("=" * 60)
    
    # パラメータ空間の可視化（2Dスライス）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # パワー vs 圧力（温度・距離固定）
    power_range = np.linspace(50, 500, 50)
    pressure_range = np.linspace(0.1, 2.0, 50)
    Power, Pressure = np.meshgrid(power_range, pressure_range)
    
    Quality_map = np.zeros_like(Power)
    for i in range(Power.shape[0]):
        for j in range(Power.shape[1]):
            Quality_map[i, j] = film_quality_metric([Power[i, j], Pressure[i, j],
                                                     optimal_params[2], optimal_params[3]])
    
    im1 = axes[0, 0].contourf(Power, Pressure, Quality_map, levels=20, cmap='RdYlGn_r')
    axes[0, 0].scatter(optimal_params[0], optimal_params[1], color='red', s=200,
                       marker='*', edgecolors='black', linewidths=2, label='Optimal')
    axes[0, 0].set_xlabel('Power [W]', fontsize=11)
    axes[0, 0].set_ylabel('Pressure [Pa]', fontsize=11)
    axes[0, 0].set_title('Quality Map: Power vs Pressure', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    plt.colorbar(im1, ax=axes[0, 0], label='Quality Score (lower is better)')
    
    # 温度 vs 距離（パワー・圧力固定）
    temp_range = np.linspace(200, 600, 50)
    dist_range = np.linspace(3, 10, 50)
    Temp, Dist = np.meshgrid(temp_range, dist_range)
    
    Quality_map2 = np.zeros_like(Temp)
    for i in range(Temp.shape[0]):
        for j in range(Temp.shape[1]):
            Quality_map2[i, j] = film_quality_metric([optimal_params[0], optimal_params[1],
                                                      Temp[i, j], Dist[i, j]])
    
    im2 = axes[0, 1].contourf(Temp, Dist, Quality_map2, levels=20, cmap='RdYlGn_r')
    axes[0, 1].scatter(optimal_params[2], optimal_params[3], color='red', s=200,
                       marker='*', edgecolors='black', linewidths=2, label='Optimal')
    axes[0, 1].set_xlabel('Temperature [°C]', fontsize=11)
    axes[0, 1].set_ylabel('Distance [cm]', fontsize=11)
    axes[0, 1].set_title('Quality Map: Temperature vs Distance', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1], label='Quality Score')
    
    # 各パラメータの感度解析
    param_names = ['Power [W]', 'Pressure [Pa]', 'Temperature [°C]', 'Distance [cm]']
    param_ranges = [np.linspace(50, 500, 50),
                    np.linspace(0.1, 2.0, 50),
                    np.linspace(200, 600, 50),
                    np.linspace(3, 10, 50)]
    
    for idx, (name, param_range) in enumerate(zip(param_names, param_ranges)):
        qualities = []
        for val in param_range:
            test_params = optimal_params.copy()
            test_params[idx] = val
            qualities.append(film_quality_metric(test_params))
    
        row = 1 if idx >= 2 else 0
        col = idx % 2
    
        if row == 1:
            axes[row, col].plot(param_range, qualities, linewidth=2, color='blue')
            axes[row, col].axvline(optimal_params[idx], color='red', linestyle='--',
                                  linewidth=2, label=f'Optimal: {optimal_params[idx]:.2f}')
            axes[row, col].set_xlabel(name, fontsize=11)
            axes[row, col].set_ylabel('Quality Score', fontsize=11)
            axes[row, col].set_title(f'Sensitivity: {name}', fontsize=12, fontweight='bold')
            axes[row, col].legend(fontsize=10)
            axes[row, col].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n感度分析結果:")
    print("  - パワー: 線形的に品質向上（高パワーほど良い）")
    print("  - 圧力: 最適値存在（0.3-0.5 Pa付近）")
    print("  - 温度: 400°C付近で結晶化促進")
    print("  - 距離: 5 cm付近で均一性最大")
    

## 4.6 演習問題

### 演習4-1: スパッタリング収率（易）

**問題** ：Ar+イオン（500 eV）によるCuターゲットのスパッタリング収率が2.5 atoms/ionのとき、イオン電流1 mAで10分間成膜した場合の総ターゲット原子放出数を計算せよ。

**解答例を表示**
    
    
    Y = 2.5  # [atoms/ion]
    I_ion = 1e-3  # [A]
    t = 10 * 60  # [s]
    e = 1.60218e-19  # [C]
    
    # イオン数 = 電流 × 時間 / 電荷
    N_ions = I_ion * t / e
    
    # 放出原子数 = 収率 × イオン数
    N_atoms = Y * N_ions
    
    print(f"イオン電流: {I_ion*1e3:.1f} mA")
    print(f"成膜時間: {t/60:.1f} min")
    print(f"入射イオン数: {N_ions:.3e}")
    print(f"放出原子数: {N_atoms:.3e}")
    print(f"             = {N_atoms/6.022e23:.2e} mol")
    

### 演習4-2: CVD成長速度の活性化エネルギー（中）

**問題** ：SiO2 CVDの成長速度が400°Cで1 nm/min、500°Cで10 nm/minだった。活性化エネルギーを求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    T1 = 400 + 273.15  # [K]
    T2 = 500 + 273.15  # [K]
    R1 = 1  # [nm/min]
    R2 = 10  # [nm/min]
    
    kB = 8.617e-5  # [eV/K]
    
    # Arrhenius式: R = A * exp(-Ea / (kB * T))
    # ln(R2/R1) = -Ea/kB * (1/T2 - 1/T1)
    
    ln_ratio = np.log(R2 / R1)
    Ea = -ln_ratio * kB / (1/T2 - 1/T1)
    
    print(f"温度1: {T1-273.15:.0f}°C, 成長速度: {R1} nm/min")
    print(f"温度2: {T2-273.15:.0f}°C, 成長速度: {R2} nm/min")
    print(f"ln(R2/R1): {ln_ratio:.3f}")
    print(f"活性化エネルギー Ea: {Ea:.2f} eV")
    print(f"\n解釈: Ea ≈ 1.5 eVは典型的なCVD反応（熱分解）の値")
    

### 演習4-3: エピタキシャル臨界厚さ（中）

**問題** ：Ge0.2Si0.8をSi基板上にエピタキシャル成長させる。Geの格子定数は5.65 Å、Siは5.43 Å。臨界厚さを推定せよ。

**解答例を表示**
    
    
    import numpy as np
    
    a_Si = 5.43  # [Å]
    a_Ge = 5.65  # [Å]
    x_Ge = 0.2
    
    # Vegard's law: a_SiGe = (1-x)*a_Si + x*a_Ge
    a_SiGe = (1 - x_Ge) * a_Si + x_Ge * a_Ge
    
    # 格子不整合
    f = (a_SiGe - a_Si) / a_Si
    
    # 臨界厚さ（簡略式）
    h_c = a_Si / (2 * np.pi * np.abs(f)) / 10  # [nm]
    
    print(f"Si格子定数: {a_Si} Å")
    print(f"Ge格子定数: {a_Ge} Å")
    print(f"Ge組成: {x_Ge*100:.0f}%")
    print(f"SiGe格子定数: {a_SiGe:.3f} Å")
    print(f"格子不整合: {f*100:.2f}%")
    print(f"臨界厚さ: {h_c:.1f} nm")
    print(f"\n結論: {h_c:.1f} nm以下で歪みエピタキシー可能")
    

### 演習4-4: Knudsen余弦則の膜厚分布（中）

**問題** ：蒸発源の真上20 cmに基板を設置した。基板中心から5 cm離れた点での膜厚は、中心の何%になるか（Knudsen余弦則を仮定）。

**解答例を表示**
    
    
    import numpy as np
    
    d = 20  # [cm] 蒸発源-基板距離
    r = 5   # [cm] 中心からの距離
    
    # 中心（r=0）
    R_center = d
    cos_center = d / R_center
    flux_center = cos_center / R_center**2
    
    # 周辺（r=5）
    R_edge = np.sqrt(r**2 + d**2)
    cos_edge = d / R_edge
    flux_edge = cos_edge / R_edge**2
    
    # 膜厚比
    ratio = flux_edge / flux_center
    
    print(f"蒸発源-基板距離: {d} cm")
    print(f"中心からの距離: {r} cm")
    print(f"中心での角度: 0° (cos=1)")
    print(f"周辺での角度: {np.arccos(cos_edge)*180/np.pi:.1f}° (cos={cos_edge:.3f})")
    print(f"膜厚比（周辺/中心）: {ratio:.3f} = {ratio*100:.1f}%")
    print(f"\n結論: 中心から5 cm離れると膜厚は{(1-ratio)*100:.1f}%減少")
    

### 演習4-5: マグネトロンスパッタの優位性（易）

**問題** ：マグネトロンスパッタリングが通常のDCスパッタより高速な理由を、プラズマ物理の観点から説明せよ。

**解答例を表示**

**理由** ：

  * **電子閉じ込め効果** ：ターゲット背後の磁場が電子をE×Bドリフトで閉じ込める
  * **プラズマ密度向上** ：電子の滞在時間が長くなり、Ar原子との衝突確率増加 → Ar+イオン密度が5-10倍向上
  * **低圧動作** ：高密度プラズマにより0.1-1 Paの低圧で安定放電 → ガス散乱減少、スパッタ粒子の平均自由行程増加
  * **イオン電流増加** ：高いAr+密度により、同じ電圧でもイオン電流が増加 → 成膜速度向上

**定量的比較** ：
    
    
    print("通常DCスパッタ:")
    print("  - 動作圧力: 1-10 Pa")
    print("  - プラズマ密度: 10^9-10^10 cm^-3")
    print("  - 成膜速度: 0.5-2 nm/s")
    print("\nマグネトロンスパッタ:")
    print("  - 動作圧力: 0.1-1 Pa")
    print("  - プラズマ密度: 10^10-10^11 cm^-3（5-10倍）")
    print("  - 成膜速度: 3-10 nm/s（5-10倍）")
    

### 演習4-6: ALD vs CVDの使い分け（中）

**問題** ：以下のケースでALDとCVDのどちらが適切か、理由とともに答えよ。  
(a) 100 nm幅のトレンチ内壁への10 nm均一コーティング  
(b) 4インチウェハ全面への1 μm厚SiO2成膜

**解答例を表示**

**(a) 100 nm幅トレンチへの10 nm均一コーティング → ALD**

  * **理由** ： 
    * アスペクト比 = (深さ/幅) が高い（通常>5）構造では、CVDは底部まで均一に成膜困難（ガス輸送律速）
    * ALDは自己制限的な表面反応により、完全なコンフォーマルコーティング達成
    * 10 nmは薄いため、ALDの低速度（0.1 nm/cycle × 100 cycle = 10分程度）でも許容可能

**(b) 4インチウェハへの1 μm厚SiO 2 → CVD（PECVD推奨）**

  * **理由** ： 
    * 1 μmは厚いため、ALDでは時間がかかりすぎる（10000 cycle以上 = 数時間〜数日）
    * CVDは成膜速度が速く（1-10 nm/s）、1 μmを数分〜数十分で成膜可能
    * 平坦なウェハ表面ではコンフォーマル性は不要
    * PECVDなら低温（300°C程度）で高品質SiO2が得られる

### 演習4-7: スパッタ成膜速度の実測（難）

**問題** ：Cuターゲット（直径10 cm）、DCパワー300 W、Ar圧力0.5 Pa、ターゲット-基板距離5 cmでスパッタリングを行った。10分後の膜厚が150 nmだった。スパッタリング収率を逆算せよ（典型的なDC電圧500 Vを仮定）。

**解答例を表示**
    
    
    import numpy as np
    
    # 既知パラメータ
    power = 300  # [W]
    voltage = 500  # [V]
    time = 10 * 60  # [s]
    thickness = 150  # [nm]
    target_area = np.pi * (5)**2  # [cm^2]
    
    # Cu物性値
    M_Cu = 63.5  # [g/mol]
    rho_Cu = 8.96  # [g/cm^3]
    
    # 定数
    e = 1.60218e-19  # [C]
    N_A = 6.022e23  # [1/mol]
    
    # イオン電流
    I_ion = power / voltage  # [A]
    J_ion = I_ion / target_area  # [A/cm^2]
    
    # 成膜速度
    rate = thickness / (time / 60)  # [nm/min]
    
    # スパッタリング収率の逆算
    # rate = (Y * J_ion * M) / (N_A * rho * e) * 1e7
    # Y = rate * (N_A * rho * e) / (J_ion * M * 1e7)
    
    Y = rate * (N_A * rho_Cu * e) / (J_ion * M_Cu * 1e7)
    
    print(f"実験条件:")
    print(f"  パワー: {power} W")
    print(f"  電圧: {voltage} V")
    print(f"  成膜時間: {time/60:.1f} min")
    print(f"  膜厚: {thickness} nm")
    print(f"\n計算結果:")
    print(f"  イオン電流: {I_ion:.3f} A = {I_ion*1e3:.1f} mA")
    print(f"  イオン電流密度: {J_ion:.4f} A/cm²")
    print(f"  成膜速度: {rate:.1f} nm/min")
    print(f"  スパッタリング収率 Y: {Y:.2f} atoms/ion")
    print(f"\n評価: Y={Y:.2f}はCuの典型値（2-3）と一致")
    

### 演習4-8: エピタキシーの成長モード判定（難）

**問題** ：GaAs（格子定数5.65 Å）をSi（格子定数5.43 Å）上にエピタキシャル成長させる場合、成長モード（FM, VW, SK）を表面エネルギーから予測せよ。  
ヒント：γGaAs = 0.7 J/m², γSi = 1.2 J/m², γinterface = 0.8 J/m²

**解答例を表示**
    
    
    import numpy as np
    
    # 表面エネルギー
    gamma_GaAs = 0.7  # [J/m^2]
    gamma_Si = 1.2  # [J/m^2]
    gamma_interface = 0.8  # [J/m^2]
    
    # 格子不整合
    a_GaAs = 5.65  # [Å]
    a_Si = 5.43  # [Å]
    f = (a_GaAs - a_Si) / a_Si
    
    print("表面エネルギー:")
    print(f"  γ_GaAs: {gamma_GaAs} J/m²")
    print(f"  γ_Si: {gamma_Si} J/m²")
    print(f"  γ_interface: {gamma_interface} J/m²")
    print(f"格子不整合: {f*100:.1f}%")
    
    # 成長モード判定基準
    # FM (Frank-van der Merwe): γ_GaAs + γ_interface < γ_Si → 完全濡れ
    # VW (Volmer-Weber): γ_GaAs + γ_interface > γ_Si → 非濡れ（島状成長）
    # SK (Stranski-Krastanov): 初期FMだが歪エネルギー蓄積でVWに遷移
    
    delta_gamma = (gamma_GaAs + gamma_interface) - gamma_Si
    
    print(f"\nΔγ = (γ_GaAs + γ_interface) - γ_Si")
    print(f"    = ({gamma_GaAs} + {gamma_interface}) - {gamma_Si}")
    print(f"    = {delta_gamma:.2f} J/m²")
    
    if delta_gamma < 0:
        print("\nΔγ < 0 → FM mode (層状成長)の傾向")
    else:
        print("\nΔγ > 0 → VW mode (島状成長)の傾向")
    
    # 格子不整合の影響
    if np.abs(f) > 0.02:
        print(f"\nしかし、格子不整合が大きい（{f*100:.1f}% > 2%）")
        print("→ 歪エネルギーが蓄積し、数MLでSK mode（層状→島状遷移）になる可能性大")
        print("実際: GaAs/Si系はSK modeとして知られている")
    
    print("\n結論: Stranski-Krastanov (SK) mode")
    print("  - 初期数ML: 2D層状成長（歪み蓄積）")
    print("  - 臨界厚さ超過後: 3D島状成長（歪み緩和）")
    

## 4.7 学習の確認

### 基本理解度チェック

  1. スパッタリングとCVDの成膜機構の違いを説明できますか？
  2. Sigmundのスパッタリング収率式の物理的意味を理解していますか？
  3. マグネトロンスパッタリングの高速化原理を説明できますか？
  4. Knudsenの余弦則が膜厚分布に与える影響を理解していますか？
  5. CVDの反応律速と拡散律速の違いを説明できますか？
  6. PECVDとALDの特徴と使い分けを理解していますか？

### 実践スキル確認

  1. スパッタリング条件（パワー、圧力）から成膜速度を推定できますか？
  2. CVD成長速度のArrhenius解析から活性化エネルギーを求められますか？
  3. エピタキシャル成長の格子不整合から臨界厚さを計算できますか？
  4. 基板回転により膜厚均一性を改善する方法を設計できますか？
  5. 多パラメータ最適化（パワー、圧力、温度、距離）を実行できますか？

### 応用力確認

  1. 実際のデバイス製造で適切な薄膜プロセスを選択できますか？
  2. 膜質問題（密着性、応力、結晶性）の原因をプロセスパラメータから推定できますか？
  3. 新規材料の薄膜成長条件を文献と物性値から設計できますか？

## 4.8 参考文献

  1. Ohring, M. (2001). _Materials Science of Thin Films_ (2nd ed.). Academic Press. pp. 123-178 (Sputtering), pp. 234-289 (Evaporation).
  2. Mattox, D.M. (2010). _Handbook of Physical Vapor Deposition (PVD) Processing_ (2nd ed.). Elsevier. pp. 89-156 (Sputtering mechanisms), pp. 234-289 (Process optimization).
  3. Chapman, B. (1980). _Glow Discharge Processes_. Wiley. pp. 89-134 (Plasma physics and sputtering).
  4. Choy, K.L. (2003). "Chemical vapour deposition of coatings." _Progress in Materials Science_ , 48:57-170. DOI: 10.1016/S0079-6425(01)00009-3
  5. Herman, M.A., Sitter, H. (1996). _Molecular Beam Epitaxy: Fundamentals and Current Status_ (2nd ed.). Springer. pp. 45-89 (Growth modes and kinetics), pp. 156-198 (Heteroepitaxy).
  6. George, S.M. (2010). "Atomic layer deposition: An overview." _Chemical Reviews_ , 110(1):111-131. DOI: 10.1021/cr900056b
  7. Matthews, J.W., Blakeslee, A.E. (1974). "Defects in epitaxial multilayers: I. Misfit dislocations." _Journal of Crystal Growth_ , 27:118-125. DOI: 10.1016/S0022-0248(74)80055-2
  8. Sigmund, P. (1969). "Theory of sputtering. I. Sputtering yield of amorphous and polycrystalline targets." _Physical Review_ , 184(2):383-416. DOI: 10.1103/PhysRev.184.383

## 4.9 次章へ

次章では、プロセスデータの解析とPython実践を学びます。統計的プロセス制御（SPC）、実験計画法（DOE）、機械学習によるプロセス予測、自動レポート生成まで、実務で即使える完全統合ワークフローを構築します。
