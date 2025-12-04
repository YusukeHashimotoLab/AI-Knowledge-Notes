---
title: 第2章：SEM入門
chapter_title: 第2章：SEM入門
subtitle: 走査型電子顕微鏡の原理、二次電子像、後方散乱電子像、EDS分析
reading_time: 25-35分
difficulty: 初級〜中級
code_examples: 7
---

走査型電子顕微鏡（SEM）は、試料表面を電子線で走査し、二次電子（SE）や後方散乱電子（BSE）を検出して高分解能像を取得する装置です。この章では、SEMの動作原理、SE像とBSE像の形成機構、エネルギー分散型X線分析（EDS）による元素分析、画像処理の基礎を学び、Pythonで信号シミュレーション、定量分析、粒子解析を実践します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ SEMの走査原理と信号検出のメカニズムを理解する
  * ✅ 二次電子（SE）と後方散乱電子（BSE）の発生機構と特性を説明できる
  * ✅ SE像とBSE像の使い分けと、それぞれの長所・短所を理解する
  * ✅ EDS（エネルギー分散型X線分析）の原理と定量分析手法を習得する
  * ✅ 加速電圧と作動距離が空間分解能と信号強度に与える影響を評価できる
  * ✅ Pythonで電子収率、EDS定量補正、粒子サイズ分布解析を実装できる
  * ✅ SEM画像の取得条件の最適化と画像処理の基礎を理解する

## 2.1 SEMの基本原理

### 2.1.1 走査電子顕微鏡の構成

SEM（Scanning Electron Microscope）は、収束した電子線を試料表面上で走査し、発生する信号を同期して検出することで画像を形成します。
    
    
    ```mermaid
    flowchart TD
        A[電子銃Electron Gun] --> B[集束レンズ系Condenser Lenses]
        B --> C[対物レンズObjective Lens]
        C --> D[走査コイルScan Coils]
        D --> E[試料Sample]
    
        E --> F[二次電子SE]
        E --> G[後方散乱電子BSE]
        E --> H[特性X線X-ray]
    
        F --> I[ET検出器]
        G --> J[BSE検出器]
        H --> K[EDS検出器]
    
        I --> L[画像表示Display]
        J --> L
        K --> M[スペクトル解析Analysis]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style L fill:#99ccff,stroke:#0066cc,stroke-width:2px
    ```

**SEMの特徴** ：

  * **広い倍率範囲** ：10倍〜100万倍（光学顕微鏡とTEMの中間を埋める）
  * **大きな焦点深度** ：凹凸のある試料でも広い範囲で焦点が合う
  * **試料作製が容易** ：バルク試料をそのまま観察可能（導電処理のみ）
  * **多様な信号検出** ：SE、BSE、X線を同時に取得可能

### 2.1.2 電子線と試料の相互作用領域

電子線が試料に入射すると、**相互作用体積（Interaction Volume）** と呼ばれる領域で様々な信号が発生します。この体積は、加速電圧と試料の原子番号により変化します。

**相互作用体積の深さ（簡略式）** ：

$$ R_{\text{KO}} = \frac{0.0276 A E_0^{1.67}}{Z^{0.89} \rho} $$ 

ここで、$R_{\text{KO}}$はKanaya-Okayamaの飛程（μm）、$A$は原子量、$E_0$は加速電圧（kV）、$Z$は原子番号、$\rho$は密度（g/cm³）です。

#### コード例2-1: 相互作用体積の深さ計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def kanaya_okayama_range(Z, A, rho, E0_kV):
        """
        Kanaya-Okayama式による電子飛程を計算
    
        Parameters
        ----------
        Z : int or array-like
            原子番号
        A : float or array-like
            原子量 [g/mol]
        rho : float or array-like
            密度 [g/cm³]
        E0_kV : float or array-like
            加速電圧 [kV]
    
        Returns
        -------
        R_KO : float or array-like
            電子飛程 [μm]
        """
        R_KO = 0.0276 * A * (E0_kV ** 1.67) / ((Z ** 0.89) * rho)
        return R_KO
    
    # 代表的な材料
    materials = {
        'C': {'Z': 6, 'A': 12, 'rho': 2.26},
        'Al': {'Z': 13, 'A': 27, 'rho': 2.70},
        'Si': {'Z': 14, 'A': 28, 'rho': 2.33},
        'Fe': {'Z': 26, 'A': 56, 'rho': 7.87},
        'Cu': {'Z': 29, 'A': 64, 'rho': 8.96},
        'Au': {'Z': 79, 'A': 197, 'rho': 19.3}
    }
    
    # 加速電圧範囲
    E0_values = np.array([5, 10, 15, 20, 30])  # [kV]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：加速電圧依存性
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(materials)))
    for (name, props), color in zip(materials.items(), colors):
        ranges = kanaya_okayama_range(props['Z'], props['A'], props['rho'], E0_values)
        ax1.plot(E0_values, ranges, 'o-', linewidth=2.5, markersize=8,
                 label=f"{name} (Z={props['Z']})", color=color)
    
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Electron Range R$_{KO}$ [μm]', fontsize=12)
    ax1.set_title('Interaction Volume Depth vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 8)
    
    # 右図：原子番号依存性（固定加速電圧）
    E0_fixed = 15  # [kV]
    Z_values = np.array([mat['Z'] for mat in materials.values()])
    A_values = np.array([mat['A'] for mat in materials.values()])
    rho_values = np.array([mat['rho'] for mat in materials.values()])
    mat_names = list(materials.keys())
    
    ranges_Z = kanaya_okayama_range(Z_values, A_values, rho_values, E0_fixed)
    
    ax2.scatter(Z_values, ranges_Z, s=200, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=2, zorder=3)
    ax2.plot(Z_values, ranges_Z, 'k--', linewidth=1.5, alpha=0.5, zorder=1)
    
    for Z, R, name in zip(Z_values, ranges_Z, mat_names):
        ax2.text(Z, R + 0.15, name, fontsize=10, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Number Z', fontsize=12)
    ax2.set_ylabel('Electron Range R$_{KO}$ [μm]', fontsize=12)
    ax2.set_title(f'Interaction Volume Depth vs Z\n(E$_0$ = {E0_fixed} kV)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 4)
    
    plt.tight_layout()
    plt.show()
    
    # 具体的な数値を出力
    print(f"加速電圧15 kVでの電子飛程:")
    for name, props in materials.items():
        R = kanaya_okayama_range(props['Z'], props['A'], props['rho'], 15)
        print(f"  {name:2s}: {R:.2f} μm")
    

**重要な観察** ：

  * 加速電圧が高いほど、電子飛程が長くなる（深い領域から信号が発生）
  * 重元素（高Z）ほど、電子飛程が短くなる（表面敏感）
  * 空間分解能を高めるには、低加速電圧で観察する

## 2.2 二次電子像（SE像）

### 2.2.1 二次電子の発生機構

**二次電子（Secondary Electron, SE）** は、入射電子との非弾性散乱により試料表面近傍（数nm）から放出される低エネルギー電子（<50 eV）です。

**SE像の特徴** ：

  * **表面敏感** ：脱出深度が数nmと浅いため、表面形態を高コントラストで観察
  * **エッジ効果** ：試料の凸部や端部で二次電子収率が増加し、明るく見える
  * **高分解能** ：プローブサイズが空間分解能を決定（5-10 nm）
  * **チャージアップ** ：絶縁性試料では電荷が蓄積し、像が歪む

### 2.2.2 二次電子収率

二次電子収率$\delta$は、入射電子1個あたりに放出される二次電子の数です。$\delta$は加速電圧と試料の傾斜角に依存します：

$$ \delta = \delta_{\max} \exp\left[-\left(\frac{E_0 - E_{\max}}{E_{\max}}\right)^2\right] \cdot \frac{1}{\cos\theta} $$ 

ここで、$\delta_{\max}$は最大二次電子収率、$E_{\max}$は$\delta_{\max}$を与える加速電圧、$\theta$は試料の傾斜角です。

#### コード例2-2: 二次電子収率のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def se_yield(E0_kV, theta_deg, delta_max=1.5, E_max_kV=0.5):
        """
        二次電子収率を計算
    
        Parameters
        ----------
        E0_kV : float or array-like
            加速電圧 [kV]
        theta_deg : float or array-like
            試料傾斜角 [degrees]（0°：垂直入射）
        delta_max : float
            最大二次電子収率
        E_max_kV : float
            最大収率を与える加速電圧 [kV]
    
        Returns
        -------
        delta : float or array-like
            二次電子収率
        """
        theta_rad = np.deg2rad(theta_deg)
        delta = delta_max * np.exp(-((E0_kV - E_max_kV) / E_max_kV)**2) / np.cos(theta_rad)
        return delta
    
    # 加速電圧依存性
    E0_range = np.linspace(0.1, 30, 200)  # [kV]
    theta_values = [0, 30, 60, 75]  # [degrees]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：加速電圧依存性（異なる傾斜角）
    colors = ['#f093fb', '#f5576c', '#ffa500', '#ff6347']
    for theta, color in zip(theta_values, colors):
        delta = se_yield(E0_range, theta, delta_max=1.5, E_max_kV=0.5)
        ax1.plot(E0_range, delta, linewidth=2.5, label=f'θ = {theta}°', color=color)
    
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='δ = 1 (Charge Balance)')
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Secondary Electron Yield δ', fontsize=12)
    ax1.set_title('SE Yield vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 4)
    
    # 右図：傾斜角依存性（固定加速電圧）
    E0_fixed = 5  # [kV]
    theta_range = np.linspace(0, 80, 100)  # [degrees]
    
    delta_theta = se_yield(E0_fixed, theta_range, delta_max=1.5, E_max_kV=0.5)
    
    ax2.plot(theta_range, delta_theta, linewidth=3, color='#f093fb')
    ax2.fill_between(theta_range, 0, delta_theta, alpha=0.3, color='#f093fb')
    
    ax2.set_xlabel('Sample Tilt Angle θ [degrees]', fontsize=12)
    ax2.set_ylabel('Secondary Electron Yield δ', fontsize=12)
    ax2.set_title(f'SE Yield vs Tilt Angle\n(E$_0$ = {E0_fixed} kV)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 10)
    
    # エッジ効果の説明を追加
    ax2.text(60, 7, 'Edge Effect:\nHigher yield at steep angles',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("二次電子収率の特性:")
    print(f"  - 垂直入射（θ=0°）、5 kV: δ = {se_yield(5, 0):.2f}")
    print(f"  - 傾斜60°、5 kV: δ = {se_yield(5, 60):.2f}")
    print("  - 凸部や端部では収率が増加 → 明るく見える（エッジ効果）")
    

## 2.3 後方散乱電子像（BSE像）

### 2.3.1 後方散乱電子の特性

**後方散乱電子（Backscattered Electron, BSE）** は、試料との弾性散乱により試料表面から反射されて戻ってくる高エネルギー電子（>50 eV）です。

**BSE像の特徴** ：

  * **組成コントラスト** ：原子番号$Z$が大きいほどBSE収率が高く、明るく見える
  * **深い情報** ：相互作用体積全体からBSEが発生（数百nm〜数μm）
  * **トポグラフィー情報** ：傾斜した面からのBSE収率が変化
  * **空間分解能** ：SE像より劣る（相互作用体積が大きい）

### 2.3.2 後方散乱電子収率

BSE収率$\eta$は、原子番号$Z$にほぼ比例します（経験式）：

$$ \eta \approx -0.0254 + 0.016 Z - 1.86 \times 10^{-4} Z^2 + 8.3 \times 10^{-7} Z^3 $$ 

#### コード例2-3: 後方散乱電子収率の原子番号依存性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def bse_yield(Z):
        """
        後方散乱電子収率を計算（経験式）
    
        Parameters
        ----------
        Z : int or array-like
            原子番号
    
        Returns
        -------
        eta : float or array-like
            BSE収率
        """
        Z = np.asarray(Z)
        eta = -0.0254 + 0.016*Z - 1.86e-4*Z**2 + 8.3e-7*Z**3
        return eta
    
    # 元素ごとのBSE収率
    elements = {
        'C': 6, 'Al': 13, 'Si': 14, 'Ti': 22, 'Fe': 26,
        'Cu': 29, 'Zn': 30, 'Mo': 42, 'Ag': 47, 'W': 74, 'Pt': 78, 'Au': 79
    }
    
    Z_values = np.array(list(elements.values()))
    element_names = list(elements.keys())
    eta_values = bse_yield(Z_values)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：原子番号 vs BSE収率
    Z_range = np.linspace(1, 92, 200)
    eta_range = bse_yield(Z_range)
    
    ax1.plot(Z_range, eta_range, linewidth=3, color='#f093fb', label='Empirical Formula')
    ax1.scatter(Z_values, eta_values, s=150, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=1.5, zorder=3)
    
    # 代表的な元素にラベル
    for Z, eta, name in zip(Z_values[::2], eta_values[::2], element_names[::2]):
        ax1.text(Z, eta + 0.02, name, fontsize=9, ha='center', fontweight='bold')
    
    ax1.set_xlabel('Atomic Number Z', fontsize=12)
    ax1.set_ylabel('Backscattered Electron Yield η', fontsize=12)
    ax1.set_title('BSE Yield vs Atomic Number', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 0.6)
    
    # 右図：組成コントラストのシミュレーション
    # Fe-Cu合金のBSE像をシミュレート
    size = 256
    image = np.zeros((size, size))
    
    # Fe領域（Z=26）
    Z_Fe = 26
    eta_Fe = bse_yield(Z_Fe)
    image[:, :size//2] = eta_Fe
    
    # Cu領域（Z=29）
    Z_Cu = 29
    eta_Cu = bse_yield(Z_Cu)
    image[:, size//2:] = eta_Cu
    
    # ノイズ追加
    image += np.random.normal(0, 0.01, image.shape)
    
    im = ax2.imshow(image, cmap='gray', vmin=0.1, vmax=0.4)
    ax2.axvline(x=size//2, color='red', linestyle='--', linewidth=2)
    ax2.text(size//4, size*0.1, f'Fe (Z={Z_Fe})\nη={eta_Fe:.3f}',
             fontsize=11, ha='center', color='white', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(size*3//4, size*0.1, f'Cu (Z={Z_Cu})\nη={eta_Cu:.3f}',
             fontsize=11, ha='center', color='black', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.set_title('BSE Image Simulation: Fe-Cu Interface\n(Compositional Contrast)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('BSE Yield η', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("後方散乱電子収率の例:")
    print(f"  C  (Z=6):  η = {bse_yield(6):.3f}")
    print(f"  Fe (Z=26): η = {bse_yield(26):.3f}")
    print(f"  Cu (Z=29): η = {bse_yield(29):.3f}")
    print(f"  Au (Z=79): η = {bse_yield(79):.3f}")
    print(f"\nAuの輝度はCの約{bse_yield(79)/bse_yield(6):.1f}倍")
    

## 2.4 エネルギー分散型X線分析（EDS）

### 2.4.1 特性X線の発生

入射電子が内殻電子を励起すると、外殻電子が遷移する際に**特性X線** が放出されます。このX線のエネルギーは元素固有であり、元素同定と定量分析が可能です。

**主なX線系列** ：

  * **K系列** ：K殻（n=1）への遷移。Kα（L→K）、Kβ（M→K）
  * **L系列** ：L殻（n=2）への遷移。Lα、Lβ、Lγ
  * **M系列** ：M殻（n=3）への遷移。重元素で重要

**モーズリーの法則** ：特性X線エネルギーは原子番号$Z$の2乗に比例：

$$ E_{\text{K}\alpha} \approx 10.2 (Z - 1)^2 \text{ eV} $$ 

### 2.4.2 EDS定量分析とZAF補正

EDS定量分析では、測定したX線強度比から組成を求めますが、**ZAF補正** が必要です：

  * **Z補正（原子番号効果）** ：後方散乱と停止能の補正
  * **A補正（吸収補正）** ：試料内でのX線吸収の補正
  * **F補正（蛍光補正）** ：他元素のX線による二次励起の補正

補正後の質量濃度$C_i$は：

$$ C_i = \frac{k_i \cdot \text{ZAF}_i}{\sum_j k_j \cdot \text{ZAF}_j} $$ 

ここで、$k_i$は標準試料に対する強度比です。

#### コード例2-4: EDS定量分析のシミュレーション（ZAF補正）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def zaf_correction(Z, A, F=1.0):
        """
        簡略化されたZAF補正係数
    
        Parameters
        ----------
        Z : float
            原子番号補正係数
        A : float
            吸収補正係数
        F : float
            蛍光補正係数（通常1に近い）
    
        Returns
        -------
        ZAF : float
            総合補正係数
        """
        return Z * A * F
    
    def quantitative_eds_analysis(k_ratios, elements, Z_factors, A_factors):
        """
        EDS定量分析（ZAF補正）
    
        Parameters
        ----------
        k_ratios : array-like
            各元素の強度比（標準試料に対する比）
        elements : list
            元素名のリスト
        Z_factors : array-like
            原子番号補正係数
        A_factors : array-like
            吸収補正係数
    
        Returns
        -------
        concentrations : dict
            各元素の質量濃度 [wt%]
        """
        k_ratios = np.array(k_ratios)
        Z_factors = np.array(Z_factors)
        A_factors = np.array(A_factors)
    
        # ZAF補正
        zaf_factors = zaf_correction(Z_factors, A_factors)
        corrected_intensities = k_ratios * zaf_factors
    
        # 正規化
        total = np.sum(corrected_intensities)
        concentrations = {elem: (corr / total) * 100 for elem, corr in zip(elements, corrected_intensities)}
    
        return concentrations
    
    # Fe-Cr-Ni合金の定量分析例
    elements = ['Fe', 'Cr', 'Ni']
    k_ratios = [0.70, 0.18, 0.12]  # 測定強度比
    Z_factors = [1.00, 0.98, 1.03]  # 原子番号補正
    A_factors = [1.00, 0.95, 1.02]  # 吸収補正
    
    concentrations = quantitative_eds_analysis(k_ratios, elements, Z_factors, A_factors)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：補正前後の比較
    uncorrected = {elem: (k / sum(k_ratios)) * 100 for elem, k in zip(elements, k_ratios)}
    
    x = np.arange(len(elements))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, list(uncorrected.values()), width,
                    label='Before ZAF Correction', color='#f093fb', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, list(concentrations.values()), width,
                    label='After ZAF Correction', color='#f5576c', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Concentration [wt%]', fontsize=12)
    ax1.set_title('EDS Quantitative Analysis: Fe-Cr-Ni Alloy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 数値ラベル
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
                 ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
                 ha='center', fontsize=10, fontweight='bold')
    
    # 右図：EDSスペクトルのシミュレーション
    energy = np.linspace(0, 10, 1000)  # [keV]
    
    # 各元素の特性X線ピーク
    Fe_Ka = 6.40  # [keV]
    Cr_Ka = 5.41
    Ni_Ka = 7.47
    
    # ガウス型ピーク
    def gaussian_peak(E, E0, sigma, amplitude):
        return amplitude * np.exp(-((E - E0) / sigma)**2)
    
    spectrum = np.zeros_like(energy)
    spectrum += gaussian_peak(energy, Fe_Ka, 0.15, concentrations['Fe'] * 10)
    spectrum += gaussian_peak(energy, Cr_Ka, 0.15, concentrations['Cr'] * 10)
    spectrum += gaussian_peak(energy, Ni_Ka, 0.15, concentrations['Ni'] * 10)
    
    # バックグラウンド（制動X線）
    background = 500 * np.exp(-energy / 2)
    spectrum += background
    
    ax2.plot(energy, spectrum, linewidth=2, color='#2c3e50')
    ax2.fill_between(energy, 0, spectrum, alpha=0.3, color='#f093fb')
    
    # ピーク位置を矢印で示す
    ax2.annotate('Fe Kα', xy=(Fe_Ka, gaussian_peak(Fe_Ka, Fe_Ka, 0.15, concentrations['Fe']*10)),
                 xytext=(Fe_Ka-1, 800), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('Cr Kα', xy=(Cr_Ka, gaussian_peak(Cr_Ka, Cr_Ka, 0.15, concentrations['Cr']*10)),
                 xytext=(Cr_Ka-1.5, 600), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('Ni Kα', xy=(Ni_Ka, gaussian_peak(Ni_Ka, Ni_Ka, 0.15, concentrations['Ni']*10)),
                 xytext=(Ni_Ka+0.5, 600), fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax2.set_xlabel('Energy [keV]', fontsize=12)
    ax2.set_ylabel('Intensity [counts]', fontsize=12)
    ax2.set_title('Simulated EDS Spectrum', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1000)
    
    plt.tight_layout()
    plt.show()
    
    print("EDS定量分析結果（ZAF補正後）:")
    for elem, conc in concentrations.items():
        print(f"  {elem}: {conc:.2f} wt%")
    

## 2.5 SEM画像解析

### 2.5.1 粒子サイズ分布解析

SEM画像から粒子サイズ分布を定量的に解析することは、材料評価において重要です。画像処理の手順：

  1. **前処理** ：ノイズ除去（ガウシアンフィルタ）、コントラスト調整
  2. **二値化** ：閾値処理により粒子領域を抽出
  3. **ラベリング** ：各粒子に固有のIDを割り当て
  4. **特徴抽出** ：面積、周囲長、円形度などを計算
  5. **統計解析** ：サイズ分布、平均粒径、標準偏差を算出

#### コード例2-5: SEM画像からの粒子サイズ分布解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter, label
    from scipy import ndimage
    
    def generate_particle_image(size=512, num_particles=50):
        """
        粒子が分散したSEM画像をシミュレート
    
        Parameters
        ----------
        size : int
            画像サイズ [pixels]
        num_particles : int
            粒子数
    
        Returns
        -------
        image : ndarray
            シミュレートされた画像
        """
        image = np.ones((size, size)) * 0.2  # バックグラウンド
    
        np.random.seed(42)
        for _ in range(num_particles):
            # ランダムな位置とサイズ
            x = np.random.randint(20, size-20)
            y = np.random.randint(20, size-20)
            radius = np.random.randint(5, 25)
    
            # 円形粒子を描画
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= radius**2
            image[mask] = 0.8 + np.random.normal(0, 0.05)
    
        # ノイズ追加
        image += np.random.normal(0, 0.03, image.shape)
        image = np.clip(image, 0, 1)
    
        # ぼかし
        image = gaussian_filter(image, sigma=1.0)
    
        return image
    
    def analyze_particles(image, threshold=0.5, pixel_size_nm=10):
        """
        粒子サイズ分布を解析
    
        Parameters
        ----------
        image : ndarray
            入力画像
        threshold : float
            二値化閾値
        pixel_size_nm : float
            ピクセルサイズ [nm/pixel]
    
        Returns
        -------
        areas : list
            各粒子の面積 [nm²]
        diameters : list
            各粒子の相当直径 [nm]
        binary : ndarray
            二値化画像
        labeled : ndarray
            ラベル画像
        """
        # 二値化
        binary = image > threshold
    
        # ラベリング
        labeled, num_features = label(binary)
    
        # 各粒子の面積を計算
        areas = []
        diameters = []
    
        for i in range(1, num_features + 1):
            area_pixels = np.sum(labeled == i)
            area_nm2 = area_pixels * (pixel_size_nm ** 2)
    
            # 相当直径（円と同じ面積を持つ直径）
            diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)
    
            # 小さすぎる粒子は除外
            if area_pixels > 10:
                areas.append(area_nm2)
                diameters.append(diameter_nm)
    
        return areas, diameters, binary, labeled
    
    # シミュレーション実行
    image = generate_particle_image(size=512, num_particles=60)
    areas, diameters, binary, labeled = analyze_particles(image, threshold=0.5, pixel_size_nm=5)
    
    # プロット
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 元画像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original SEM Image', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # 二値化画像
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Binary Image\n(Threshold = 0.5)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    # ラベル画像
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(labeled, cmap='tab20')
    ax3.set_title(f'Labeled Image\n({len(diameters)} particles detected)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    
    # ヒストグラム：粒子直径分布
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(diameters, bins=20, color='#f093fb', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Particle Diameter [nm]', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Particle Size Distribution', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 統計情報を追加
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    ax4.axvline(mean_diameter, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diameter:.1f} nm')
    ax4.axvline(mean_diameter - std_diameter, color='orange', linestyle=':', linewidth=1.5, label=f'Std: ±{std_diameter:.1f} nm')
    ax4.axvline(mean_diameter + std_diameter, color='orange', linestyle=':', linewidth=1.5)
    ax4.legend(fontsize=11)
    
    # 統計テーブル
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_text = f"""
    Particle Analysis Results
    
    Number of particles: {len(diameters)}
    
    Diameter [nm]:
      Mean:   {mean_diameter:.2f}
      Median: {np.median(diameters):.2f}
      Std:    {std_diameter:.2f}
      Min:    {np.min(diameters):.2f}
      Max:    {np.max(diameters):.2f}
    
    Area [nm²]:
      Mean:   {np.mean(areas):.1f}
      Total:  {np.sum(areas):.1f}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"粒子解析結果:")
    print(f"  検出粒子数: {len(diameters)}")
    print(f"  平均直径: {mean_diameter:.2f} ± {std_diameter:.2f} nm")
    print(f"  粒径範囲: {np.min(diameters):.2f} - {np.max(diameters):.2f} nm")
    

## 2.6 演習問題

### 演習2-1: 電子飛程の計算（Easy）

**問題** ：Si試料（Z=14, A=28, ρ=2.33 g/cm³）を15 kVで観察する際の電子飛程を計算せよ。

**解答例を表示**
    
    
    Z = 14
    A = 28
    rho = 2.33
    E0 = 15
    
    R_KO = 0.0276 * A * (E0 ** 1.67) / ((Z ** 0.89) * rho)
    print(f"Si（15 kV）の電子飛程: {R_KO:.2f} μm")
    

### 演習2-2: 二次電子収率の評価（Medium）

**問題** ：5 kVの加速電圧で、試料を60°傾斜させた場合の二次電子収率を計算せよ（δ_max=1.5, E_max=0.5 kV）。垂直入射の場合と比較せよ。

**解答例を表示**
    
    
    import numpy as np
    
    E0 = 5
    delta_max = 1.5
    E_max = 0.5
    
    # 垂直入射（θ=0°）
    theta_0 = 0
    delta_0 = delta_max * np.exp(-((E0 - E_max) / E_max)**2) / np.cos(np.deg2rad(theta_0))
    
    # 傾斜60°
    theta_60 = 60
    delta_60 = delta_max * np.exp(-((E0 - E_max) / E_max)**2) / np.cos(np.deg2rad(theta_60))
    
    print(f"垂直入射（θ=0°）: δ = {delta_0:.3f}")
    print(f"傾斜60°: δ = {delta_60:.3f}")
    print(f"収率増加: {delta_60 / delta_0:.2f}倍")
    

### 演習2-3: BSE組成コントラストの定量化（Medium）

**問題** ：Ti（Z=22）とNi（Z=28）の境界をBSE像で観察する。両者のBSE収率を計算し、像コントラストを推定せよ。

**解答例を表示**
    
    
    Z_Ti = 22
    Z_Ni = 28
    
    eta_Ti = -0.0254 + 0.016*Z_Ti - 1.86e-4*Z_Ti**2 + 8.3e-7*Z_Ti**3
    eta_Ni = -0.0254 + 0.016*Z_Ni - 1.86e-4*Z_Ni**2 + 8.3e-7*Z_Ni**3
    
    contrast = (eta_Ni - eta_Ti) / eta_Ti * 100
    
    print(f"Ti (Z={Z_Ti}): η = {eta_Ti:.3f}")
    print(f"Ni (Z={Z_Ni}): η = {eta_Ni:.3f}")
    print(f"コントラスト: {contrast:.1f}%")
    print(f"Ni領域がTi領域より{contrast:.1f}%明るく見える")
    

### 演習2-4: EDS定量分析（Hard）

**問題** ：Al-Si合金のEDS分析で、k(Al)=0.65、k(Si)=0.35を得た。ZAF補正係数がAl: 1.02、Si: 0.98の場合、組成を求めよ。

**解答例を表示**
    
    
    k_Al = 0.65
    k_Si = 0.35
    ZAF_Al = 1.02
    ZAF_Si = 0.98
    
    corrected_Al = k_Al * ZAF_Al
    corrected_Si = k_Si * ZAF_Si
    
    total = corrected_Al + corrected_Si
    
    C_Al = (corrected_Al / total) * 100
    C_Si = (corrected_Si / total) * 100
    
    print(f"ZAF補正後の組成:")
    print(f"  Al: {C_Al:.2f} wt%")
    print(f"  Si: {C_Si:.2f} wt%")
    

### 演習2-5: 粒子サイズ統計（Hard）

**問題** ：10個の粒子の直径が[50, 55, 60, 52, 58, 62, 48, 54, 56, 61] nmである。平均粒径、標準偏差、相対標準偏差（RSD）を計算せよ。

**解答例を表示**
    
    
    import numpy as np
    
    diameters = np.array([50, 55, 60, 52, 58, 62, 48, 54, 56, 61])
    
    mean_d = np.mean(diameters)
    std_d = np.std(diameters, ddof=1)  # 不偏標準偏差
    rsd = (std_d / mean_d) * 100
    
    print(f"平均粒径: {mean_d:.2f} nm")
    print(f"標準偏差: {std_d:.2f} nm")
    print(f"相対標準偏差（RSD）: {rsd:.2f} %")
    

### 演習2-6: 加速電圧の最適化（Hard）

**問題** ：TiとFeの組成コントラストを最大化するために、5 kVと20 kVのどちらが適切か、BSE収率の差を計算して議論せよ。

**解答例を表示**

**解答のポイント** ：

  * BSE収率$\eta$は加速電圧にほとんど依存しない（原子番号に主に依存）
  * したがって、5 kVでも20 kVでも組成コントラスト自体は同程度
  * しかし、低加速電圧（5 kV）の方が表面敏感で、空間分解能が高い
  * **結論** ：表面の組成分布を高分解能で観察したい場合は5 kVが有利

### 演習2-7: X線空間分解能の評価（Hard）

**問題** ：Al試料を15 kVで観察する際、Al Kα線（1.49 keV）の発生領域の深さを推定せよ。電子飛程の70%程度と仮定する。

**解答例を表示**
    
    
    Z_Al = 13
    A_Al = 27
    rho_Al = 2.70
    E0 = 15
    
    R_KO = 0.0276 * A_Al * (E0 ** 1.67) / ((Z_Al ** 0.89) * rho_Al)
    X_ray_depth = R_KO * 0.7
    
    print(f"Al（15 kV）の電子飛程: {R_KO:.2f} μm")
    print(f"Al Kα線の発生深さ（推定）: {X_ray_depth:.2f} μm")
    print(f"EDSの空間分解能は約{X_ray_depth*1000:.0f} nm程度")
    

### 演習2-8: 実験計画（Hard）

**問題** ：Al合金中のMg2Si析出物（サイズ数十nm）を観察・定量する実験計画を立案せよ。SE像、BSE像、EDSの使い分けを説明せよ。

**解答例を表示**

**実験計画** ：

  1. **SE像（低倍率）** ：試料表面の全体観察、析出物の分布確認（5-10 kV）
  2. **BSE像（高倍率）** ：Mg2Si（軽元素）とAl母相の組成コントラスト観察（10-15 kV）。Mg2Siは暗く見える
  3. **SE像（高倍率）** ：析出物の形態と粒径測定（5 kV、表面敏感）
  4. **EDS点分析** ：析出物上と母相上でそれぞれスペクトル取得、Mg/Si/Al比を定量（15 kV）
  5. **EDSマッピング** ：Mg、Si、Alの元素分布を可視化（15 kV、長時間積算）
  6. **画像解析** ：SE像またはBSE像から粒径分布、数密度を定量

**理由** ：

  * SE像は表面形態観察に最適、BSE像は組成コントラストで析出物を識別
  * 15 kVは、Mg Kα（1.25 keV）とSi Kα（1.74 keV）の励起に十分
  * 低加速電圧（5 kV）は高分解能だが、X線励起効率が低下するため定量には不向き

## 2.7 学習チェック

以下の質問に答えて、理解度を確認しましょう：

  1. SEMの走査原理と画像形成のメカニズムを説明できますか？
  2. 二次電子と後方散乱電子の発生機構の違いを理解していますか？
  3. SE像とBSE像の特徴と使い分けを説明できますか？
  4. 電子飛程の原子番号依存性と加速電圧依存性を理解していますか？
  5. 特性X線の発生機構とモーズリーの法則を説明できますか？
  6. EDS定量分析におけるZAF補正の必要性を理解していますか？
  7. SEM画像からの粒子サイズ分布解析の手順を説明できますか？
  8. 加速電圧と作動距離の選択が像質に与える影響を理解していますか？

## 2.8 参考文献

  1. Goldstein, J. I., et al. (2017). _Scanning Electron Microscopy and X-Ray Microanalysis_ (4th ed.). Springer. - SEMとEDS分析の包括的教科書
  2. Reimer, L. (1998). _Scanning Electron Microscopy: Physics of Image Formation and Microanalysis_ (2nd ed.). Springer. - SEM結像理論の詳細
  3. Newbury, D. E., & Ritchie, N. W. M. (2013). "Is Scanning Electron Microscopy/Energy Dispersive X-ray Spectrometry (SEM/EDS) Quantitative?" _Scanning_ , 35, 141-168. - EDS定量分析の精度評価
  4. Joy, D. C. (1995). _Monte Carlo Modeling for Electron Microscopy and Microanalysis_. Oxford University Press. - 電子線シミュレーション
  5. Echlin, P. (2009). _Handbook of Sample Preparation for Scanning Electron Microscopy and X-Ray Microanalysis_. Springer. - 試料作製技術
  6. JEOL Application Notes. "SEM Basics and Applications" - メーカー技術資料
  7. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - 電子線と物質の相互作用（基礎理論）

## 2.9 次章へ

次章では、透過型電子顕微鏡（TEM）の原理、結像理論、明視野・暗視野像、制限視野電子回折（SAED）、高分解能TEM（HRTEM）、収差補正技術を学びます。TEMは、試料を透過した電子線を利用して、原子レベルの構造を観察する強力な手法です。
