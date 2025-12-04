---
title: "第1章:電子顕微鏡の基礎"
chapter_title: "第1章:電子顕微鏡の基礎"
subtitle: 電子線と物質の相互作用、分解能の理論、電子顕微鏡の原理
reading_time: 20-30分
difficulty: 初級〜中級
code_examples: 7
---

電子顕微鏡は、光学顕微鏡の分解能限界を超えて、原子・分子レベルの構造を観察できる強力なツールです。この章では、電子線と物質の相互作用、電子波の波長と分解能の関係、電子顕微鏡の基本原理を学び、Pythonで電子波長計算、分解能シミュレーション、散乱断面積計算を実践します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 電子波の波動性と粒子性を理解し、ド・ブロイ波長を計算できる
  * ✅ レイリー基準による分解能の定義と、電子顕微鏡の分解能向上原理を説明できる
  * ✅ 弾性散乱と非弾性散乱の違いと、各散乱信号の利用法を理解する
  * ✅ 加速電圧が電子波長と分解能に与える影響を定量的に評価できる
  * ✅ 電子顕微鏡の構成要素（電子銃、レンズ、検出器）の役割を説明できる
  * ✅ Pythonで散乱強度、コントラスト伝達関数、信号強度分布を計算できる
  * ✅ 光学顕微鏡と電子顕微鏡の本質的な違いを波長の観点から説明できる

## 1.1 電子波の基礎

### 1.1.1 波動と粒子の二重性

電子は**波動性** と**粒子性** の両方を持つ量子力学的粒子です。ルイ・ド・ブロイ（1924年）は、運動量$p$を持つ粒子が波長$\lambda$を持つことを提唱しました：

$$ \lambda = \frac{h}{p} $$ 

ここで、$h = 6.626 \times 10^{-34}$ J·s（プランク定数）、$p = mv$（運動量：質量$m$ × 速度$v$）です。

**相対論補正** ：電子が高速で加速される場合、相対論効果を考慮する必要があります。加速電圧$V$で加速された電子のエネルギーは：

$$ E = eV $$ 

ここで、$e = 1.602 \times 10^{-19}$ C（電荷素量）です。相対論的な運動量は次式で計算されます：

$$ p = \sqrt{\frac{2m_0 eV}{c^2}\left(1 + \frac{eV}{2m_0 c^2}\right)} $$ 

$m_0 = 9.109 \times 10^{-31}$ kg（電子の静止質量）、$c = 2.998 \times 10^8$ m/s（光速）です。

#### コード例1-1: 電子波長の計算（相対論補正あり）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_electron_wavelength(voltage_kV):
        """
        加速電圧から電子波長を計算（相対論補正あり）
    
        Parameters
        ----------
        voltage_kV : float or array-like
            加速電圧 [kV]
    
        Returns
        -------
        wavelength_pm : float or array-like
            電子波長 [pm]
        """
        # 物理定数
        h = 6.62607e-34    # プランク定数 [J·s]
        m0 = 9.10938e-31   # 電子質量 [kg]
        e = 1.60218e-19    # 電荷素量 [C]
        c = 2.99792e8      # 光速 [m/s]
    
        # 加速電圧をJに変換
        V = voltage_kV * 1000  # [V]
        E = e * V              # エネルギー [J]
    
        # 相対論的運動量
        p = np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    
        # ド・ブロイ波長
        wavelength_m = h / p
        wavelength_pm = wavelength_m * 1e12  # [pm]
    
        return wavelength_pm
    
    # 加速電圧範囲
    voltages = np.array([10, 50, 100, 200, 300, 500, 1000])  # [kV]
    wavelengths = calculate_electron_wavelength(voltages)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：波長 vs 加速電圧
    ax1.plot(voltages, wavelengths, 'o-', linewidth=2, markersize=8, color='#f093fb')
    ax1.set_xlabel('Acceleration Voltage [kV]', fontsize=12)
    ax1.set_ylabel('Electron Wavelength [pm]', fontsize=12)
    ax1.set_title('Electron Wavelength vs Acceleration Voltage', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 光学顕微鏡の可視光波長を参考線として追加
    ax1.axhline(y=500000, color='orange', linestyle='--', linewidth=2, label='Visible Light (~500 nm)')
    ax1.legend(fontsize=10)
    
    # 右図：代表的な加速電圧での波長比較
    selected_voltages = [100, 200, 300]
    selected_wavelengths = calculate_electron_wavelength(selected_voltages)
    
    ax2.bar([f'{v} kV' for v in selected_voltages], selected_wavelengths,
            color=['#f093fb', '#f5576c', '#d07be8'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Electron Wavelength [pm]', fontsize=12)
    ax2.set_title('Typical Operating Voltages', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (v, wl) in enumerate(zip(selected_voltages, selected_wavelengths)):
        ax2.text(i, wl + 0.1, f'{wl:.3f} pm', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 具体的な数値を出力
    print("加速電圧と電子波長の対応:")
    for v, wl in zip(voltages, wavelengths):
        print(f"  {v:4.0f} kV → λ = {wl:.4f} pm = {wl/100:.4f} Å")
    print(f"\n可視光（550 nm）に対する200 kVの電子波の波長比: {550000 / wavelengths[3]:.0f}:1")
    

**出力解釈** ：

  * 100 kVで約3.7 pm、200 kVで約2.5 pm、300 kVで約2.0 pmの波長
  * 可視光（~500 nm）と比較して**約20万倍短い波長**
  * 加速電圧が高いほど波長が短くなり、分解能が向上

### 1.1.2 分解能とレイリー基準

顕微鏡の**分解能（Resolution）** は、2つの点を識別できる最小距離として定義されます。レイリー基準によれば：

$$ d = \frac{0.61 \lambda}{n \sin\alpha} $$ 

ここで：

  * $d$：分解能（識別可能な最小距離）
  * $\lambda$：波長
  * $n \sin\alpha$：開口数（NA: Numerical Aperture）
  * $\alpha$：対物レンズの半開き角

**光学顕微鏡 vs 電子顕微鏡** ：

項目 | 光学顕微鏡 | 電子顕微鏡（200 kV）  
---|---|---  
波長 $\lambda$ | ~500 nm（可視光） | ~2.5 pm（電子波）  
分解能 $d$ | ~200 nm（理論限界） | ~0.05 nm（収差補正TEM）  
倍率 | ~2,000倍 | ~50,000,000倍  
  
#### コード例1-2: 分解能のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def resolution_rayleigh(wavelength_nm, alpha_deg, n=1.0):
        """
        レイリー基準による分解能を計算
    
        Parameters
        ----------
        wavelength_nm : float
            波長 [nm]
        alpha_deg : float
            対物レンズ半開き角 [degrees]
        n : float
            屈折率（電子顕微鏡では真空中なので1.0）
    
        Returns
        -------
        d_nm : float
            分解能 [nm]
        """
        alpha_rad = np.deg2rad(alpha_deg)
        d_nm = 0.61 * wavelength_nm / (n * np.sin(alpha_rad))
        return d_nm
    
    # 光学顕微鏡と電子顕微鏡の比較
    alpha_range = np.linspace(0.5, 30, 100)  # [degrees]
    
    # 光学顕微鏡（可視光）
    optical_wavelength = 550  # [nm]
    optical_resolution = resolution_rayleigh(optical_wavelength, alpha_range)
    
    # 電子顕微鏡（100 kV → 3.7 pm = 0.0037 nm）
    em_100kV_wavelength = 0.0037  # [nm]
    em_100kV_resolution = resolution_rayleigh(em_100kV_wavelength, alpha_range)
    
    # 電子顕微鏡（200 kV → 2.5 pm = 0.0025 nm）
    em_200kV_wavelength = 0.0025  # [nm]
    em_200kV_resolution = resolution_rayleigh(em_200kV_wavelength, alpha_range)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(alpha_range, optical_resolution, linewidth=2.5, label='Optical (550 nm)', color='orange')
    ax.plot(alpha_range, em_100kV_resolution, linewidth=2.5, label='EM 100 kV (3.7 pm)', color='#f093fb')
    ax.plot(alpha_range, em_200kV_resolution, linewidth=2.5, label='EM 200 kV (2.5 pm)', color='#f5576c')
    
    # 光学顕微鏡の実用的な分解能限界
    ax.axhline(y=200, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Optical Limit (~200 nm)')
    
    # 電子顕微鏡の実用的な分解能（球面収差などの影響）
    ax.axhline(y=0.1, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Practical EM Limit (~0.1 nm)')
    
    ax.set_xlabel('Objective Lens Half-Angle α [degrees]', fontsize=12)
    ax.set_ylabel('Resolution d [nm]', fontsize=12)
    ax.set_title('Resolution vs Lens Aperture (Rayleigh Criterion)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e3)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 実用的な値を計算
    alpha_typical = 10  # [degrees]（典型的な対物レンズ半開き角）
    print(f"対物レンズ半開き角 α = {alpha_typical}°での分解能:")
    print(f"  光学顕微鏡: {resolution_rayleigh(optical_wavelength, alpha_typical):.1f} nm")
    print(f"  電子顕微鏡 (100 kV): {resolution_rayleigh(em_100kV_wavelength, alpha_typical):.4f} nm")
    print(f"  電子顕微鏡 (200 kV): {resolution_rayleigh(em_200kV_wavelength, alpha_typical):.4f} nm")
    

## 1.2 電子線と物質の相互作用

### 1.2.1 散乱の種類

電子線が試料に入射すると、原子核や電子との相互作用により様々な**散乱現象** が生じます。
    
    
    ```mermaid
    flowchart TD
        A[入射電子線] --> B{試料中の相互作用}
        B --> C[弾性散乱Elastic Scattering]
        B --> D[非弾性散乱Inelastic Scattering]
    
        C --> E[後方散乱電子BSE]
        C --> F[透過電子TEM信号]
    
        D --> G[二次電子SE]
        D --> H[X線EDS/WDS]
        D --> I[オージェ電子AES]
        D --> J[エネルギー損失電子EELS]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style D fill:#ffeb99,stroke:#ffa500,stroke-width:2px
    ```

**弾性散乱（Elastic Scattering）** ：

  * 電子のエネルギー損失がほぼゼロ（$\Delta E \approx 0$）
  * 原子核のクーロンポテンシャルによる散乱
  * 散乱角は原子番号$Z$に依存（重い元素ほど大角度散乱）
  * **利用** ：TEM回折、HRTEM、HAADF-STEM（Z-contrast像）

**非弾性散乱（Inelastic Scattering）** ：

  * 電子がエネルギーを失う（$\Delta E > 0$）
  * 内殻電子励起、プラズモン励起、フォノン励起など
  * **利用** ：EDS（元素分析）、EELS（電子状態分析）、SE（表面形態観察）

### 1.2.2 散乱断面積と原子番号依存性

散乱の強度は**散乱断面積（Scattering Cross Section）** $\sigma$で表されます。ラザフォード散乱の近似では：

$$ \frac{d\sigma}{d\Omega} = \left(\frac{Ze^2}{4\pi\epsilon_0 E}\right)^2 \frac{1}{\sin^4(\theta/2)} $$ 

ここで、$Z$は原子番号、$E$は電子のエネルギー、$\theta$は散乱角です。

#### コード例1-3: 散乱断面積の原子番号依存性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rutherford_scattering_cross_section(Z, E_keV, theta_deg):
        """
        ラザフォード散乱断面積を計算
    
        Parameters
        ----------
        Z : int or array-like
            原子番号
        E_keV : float
            電子のエネルギー [keV]
        theta_deg : float
            散乱角 [degrees]
    
        Returns
        -------
        cross_section : float or array-like
            微分散乱断面積 [nm²/sr]
        """
        # 定数
        e = 1.60218e-19      # [C]
        epsilon_0 = 8.85419e-12  # [F/m]
        a0 = 0.0529          # ボーア半径 [nm]
    
        # エネルギーをJに変換
        E_J = E_keV * 1000 * e
    
        # 散乱角をラジアンに変換
        theta_rad = np.deg2rad(theta_deg)
    
        # ラザフォード散乱断面積（簡略式）
        # dσ/dΩ ∝ Z² / (E² sin⁴(θ/2))
        Z = np.asarray(Z)
        cross_section = (Z / E_keV)**2 / (np.sin(theta_rad / 2)**4)
    
        # 正規化係数（任意単位）
        cross_section = cross_section * 1e-3
    
        return cross_section
    
    # 代表的な元素
    elements = {
        'C': 6, 'Al': 13, 'Si': 14, 'Ti': 22,
        'Fe': 26, 'Cu': 29, 'Mo': 42, 'W': 74, 'Pt': 78, 'Au': 79
    }
    
    E_keV = 200  # 200 kV
    theta_range = np.linspace(1, 30, 100)  # [degrees]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：散乱角依存性
    selected_elements = ['C', 'Si', 'Fe', 'Au']
    colors = ['#d3d3d3', '#f093fb', '#ffa500', '#ffd700']
    
    for elem, color in zip(selected_elements, colors):
        Z = elements[elem]
        cs = rutherford_scattering_cross_section(Z, E_keV, theta_range)
        ax1.plot(theta_range, cs, linewidth=2.5, label=f'{elem} (Z={Z})', color=color)
    
    ax1.set_xlabel('Scattering Angle θ [degrees]', fontsize=12)
    ax1.set_ylabel('d$\sigma$/d$\Omega$ [arb. units]', fontsize=12)
    ax1.set_title(f'Scattering Cross Section vs Angle\n(E = {E_keV} keV)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2, 1e4)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # 右図：原子番号依存性（固定散乱角）
    Z_values = np.array(list(elements.values()))
    element_names = list(elements.keys())
    theta_fixed = 10  # [degrees]
    
    cs_Z = rutherford_scattering_cross_section(Z_values, E_keV, theta_fixed)
    
    ax2.scatter(Z_values, cs_Z, s=150, c=Z_values, cmap='plasma',
                edgecolors='black', linewidths=1.5, zorder=3)
    ax2.plot(Z_values, cs_Z, 'k--', linewidth=1, alpha=0.5, zorder=1)
    
    for i, (Z, cs, name) in enumerate(zip(Z_values, cs_Z, element_names)):
        ax2.text(Z, cs * 1.2, name, fontsize=9, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Number Z', fontsize=12)
    ax2.set_ylabel('d$\sigma$/d$\Omega$ [arb. units]', fontsize=12)
    ax2.set_title(f'Scattering Cross Section vs Z\n(θ = {theta_fixed}°, E = {E_keV} keV)',
                  fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print(f"散乱断面積の原子番号依存性（θ = {theta_fixed}°）:")
    print(f"  C (Z=6):   {rutherford_scattering_cross_section(6, E_keV, theta_fixed):.3f}")
    print(f"  Si (Z=14): {rutherford_scattering_cross_section(14, E_keV, theta_fixed):.3f}")
    print(f"  Fe (Z=26): {rutherford_scattering_cross_section(26, E_keV, theta_fixed):.3f}")
    print(f"  Au (Z=79): {rutherford_scattering_cross_section(79, E_keV, theta_fixed):.3f}")
    print(f"\nAuの散乱断面積はCの約{(79/6)**2:.0f}倍（Z²依存性）")
    

**重要な観察** ：

  * 散乱断面積は$Z^2$に比例 → 重元素ほど強く散乱
  * 小角散乱が支配的（$\sin^4(\theta/2)$の逆数に比例）
  * HAADF-STEM（高角環状暗視野）では、大角度散乱を検出してZ-contrast像を取得

## 1.3 電子顕微鏡の構成要素

### 1.3.1 電子銃（Electron Gun）

電子銃は、安定した電子線を生成する装置です。主な種類：

種類 | 原理 | 輝度 | エネルギー広がり | 用途  
---|---|---|---|---  
**タングステン熱電子銃** | 加熱によるエミッション | 低（~10⁵ A/cm²·sr） | ~2-3 eV | 汎用SEM、低コスト  
**LaB₆熱電子銃** | LaB₆フィラメント加熱 | 中（~10⁶ A/cm²·sr） | ~1-2 eV | 汎用TEM、バランス型  
**電界放出銃（FEG）** | 強電界による量子トンネル効果 | 高（~10⁸ A/cm²·sr） | ~0.3-0.7 eV | 高分解能TEM/STEM、EELS  
  
**電界放出銃（FEG: Field Emission Gun）の利点** ：

  * 高輝度 → 細いプローブ（0.1 nm以下）が可能
  * 狭いエネルギー広がり → EELSの高エネルギー分解能
  * 高いコヒーレンス → HRTEMの高分解能

### 1.3.2 電子レンズと収差

電子レンズは、磁場（電磁レンズ）または静電場により電子線を集束させます。

**主な収差** ：

  * **球面収差（Spherical Aberration, $C_s$）** ：レンズの端を通る電子線が焦点から外れる
  * **色収差（Chromatic Aberration, $C_c$）** ：エネルギー広がりによる焦点のずれ
  * **非点収差（Astigmatism）** ：レンズの非対称性による像の歪み

現代のTEMでは、**収差補正器（Cs-corrector）** により球面収差をほぼゼロに補正し、0.05 nm以下の分解能を実現しています。

#### コード例1-4: コントラスト伝達関数（CTF）の基礎
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def ctf_simple(k, wavelength_A, defocus_nm, Cs_mm):
        """
        簡略化されたコントラスト伝達関数（CTF）
    
        Parameters
        ----------
        k : array-like
            空間周波数 [1/Å]
        wavelength_A : float
            電子波長 [Å]
        defocus_nm : float
            デフォーカス [nm]（負値：アンダーフォーカス）
        Cs_mm : float
            球面収差係数 [mm]
    
        Returns
        -------
        ctf : array-like
            CTF値
        """
        # 単位変換
        defocus_A = defocus_nm * 10
        Cs_A = Cs_mm * 1e7
    
        # 位相シフト χ(k)
        chi = (np.pi * wavelength_A / 2) * k**2 * (defocus_A + 0.5 * wavelength_A**2 * k**2 * Cs_A)
    
        # CTF = sin(χ)
        ctf = np.sin(chi)
    
        return ctf
    
    # パラメータ
    voltage_kV = 200
    wavelength_pm = 2.508  # 200 kVでの波長
    wavelength_A = wavelength_pm / 100
    
    k = np.linspace(0, 10, 500)  # 空間周波数 [1/Å]
    
    # 異なるデフォーカスとCs値でのCTF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：デフォーカス依存性
    Cs = 1.0  # [mm]
    defocus_values = [-50, -100, -200]  # [nm]
    colors = ['#f093fb', '#f5576c', '#d07be8']
    
    for df, color in zip(defocus_values, colors):
        ctf = ctf_simple(k, wavelength_A, df, Cs)
        ax1.plot(k, ctf, linewidth=2, label=f'Δf = {df} nm', color=color)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Spatial Frequency k [1/Å]', fontsize=12)
    ax1.set_ylabel('CTF', fontsize=12)
    ax1.set_title(f'CTF vs Defocus\n(200 kV, Cs = {Cs} mm)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1, 1)
    
    # 右図：球面収差係数依存性
    defocus = -100  # [nm]
    Cs_values = [0.5, 1.0, 2.0]  # [mm]
    colors2 = ['#99ccff', '#ffa500', '#ff6347']
    
    for Cs_val, color in zip(Cs_values, colors2):
        ctf = ctf_simple(k, wavelength_A, defocus, Cs_val)
        ax2.plot(k, ctf, linewidth=2, label=f'Cs = {Cs_val} mm', color=color)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Spatial Frequency k [1/Å]', fontsize=12)
    ax2.set_ylabel('CTF', fontsize=12)
    ax2.set_title(f'CTF vs Cs\n(200 kV, Δf = {defocus} nm)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("CTFの解釈:")
    print("  - CTFが正の領域：その空間周波数成分が像に正のコントラストで寄与")
    print("  - CTFが負の領域：コントラストが反転")
    print("  - CTFがゼロ：その周波数成分が像に寄与しない（情報損失）")
    

### 1.3.3 検出器

電子顕微鏡では、様々な信号を検出するため、複数の検出器が使用されます：

  * **蛍光スクリーン** ：電子線を可視光に変換（リアルタイム観察）
  * **CCDカメラ** ：高感度、デジタル画像取得
  * **直接電子検出器（Direct Electron Detector, DED）** ：電子を直接検出、高速・高感度
  * **エネルギー分散型X線分析装置（EDS/EDX）** ：元素分析
  * **電子エネルギー損失分光器（EELS）** ：電子状態分析

## 1.4 信号強度と統計

### 1.4.1 信号対雑音比（S/N比）

電子顕微鏡像の品質は、**信号対雑音比（Signal-to-Noise Ratio, S/N）** で評価されます：

$$ \text{S/N} = \frac{I_{\text{signal}}}{\sigma_{\text{noise}}} $$ 

電子線はポアソン統計に従うため、検出電子数$N$の場合、雑音の標準偏差は$\sqrt{N}$です。したがって：

$$ \text{S/N} = \sqrt{N} $$ 

**S/N比向上の戦略** ：

  * 電子線量を増やす（$N$を増やす） → S/N $\propto \sqrt{N}$
  * 複数枚の画像を平均化 → S/N $\propto \sqrt{M}$（$M$：画像枚数）
  * ただし、試料損傷（ビームダメージ）に注意

#### コード例1-5: ポアソン統計と信号対雑音比
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_electron_image(size=256, signal_level=100, noise_factor=1.0):
        """
        ポアソン統計に従う電子顕微鏡像をシミュレート
    
        Parameters
        ----------
        size : int
            画像サイズ [pixels]
        signal_level : float
            平均電子数 [counts/pixel]
        noise_factor : float
            雑音スケーリング係数
    
        Returns
        -------
        image : ndarray
            シミュレートされた画像
        snr : float
            信号対雑音比
        """
        # ポアソン分布に従う電子数生成
        image = np.random.poisson(signal_level, (size, size)).astype(float)
    
        # 追加のガウス雑音（読み出し雑音など）
        image += np.random.normal(0, noise_factor * np.sqrt(signal_level), (size, size))
    
        # S/N比計算
        signal = signal_level
        noise = np.sqrt(signal_level + noise_factor**2 * signal_level)
        snr = signal / noise
    
        return image, snr
    
    # 異なる電子線量でのシミュレーション
    dose_levels = [10, 50, 200, 1000]  # [electrons/pixel]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, dose in enumerate(dose_levels):
        image, snr = simulate_electron_image(size=128, signal_level=dose, noise_factor=0.5)
    
        # 画像表示
        axes[0, i].imshow(image, cmap='gray', vmin=0, vmax=1200)
        axes[0, i].set_title(f'Dose: {dose} e⁻/px\nS/N: {snr:.2f}', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
    
        # ヒストグラム
        axes[1, i].hist(image.ravel(), bins=50, color='#f093fb', alpha=0.7, edgecolor='black')
        axes[1, i].set_xlabel('Intensity [counts]', fontsize=10)
        axes[1, i].set_ylabel('Frequency', fontsize=10)
        axes[1, i].set_title(f'Histogram (σ={np.std(image):.1f})', fontsize=10)
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # S/N比の理論値と実測値の比較
    print("電子線量とS/N比の関係:")
    for dose in dose_levels:
        _, snr = simulate_electron_image(size=128, signal_level=dose, noise_factor=0.5)
        theoretical_snr = np.sqrt(dose)
        print(f"  線量 {dose:4d} e⁻/px: 理論S/N={theoretical_snr:.2f}, 実測S/N={snr:.2f}")
    

## 1.5 加速電圧の選択と試料損傷

### 1.5.1 加速電圧の効果

加速電圧は、電子顕微鏡の性能と試料への影響に直接関わります：

加速電圧 | 波長 | 分解能 | 透過能力 | 試料損傷  
---|---|---|---|---  
低電圧（60-80 kV） | 長い | 低 | 低 | 小  
中電圧（100-200 kV） | 中 | 中 | 中 | 中  
高電圧（300-1000 kV） | 短い | 高 | 高 | 大  
  
**試料損傷のメカニズム** ：

  * **ノックオン損傷** ：高エネルギー電子が原子を弾き飛ばす（閾値エネルギー以上）
  * **ラジオリシス（放射線分解）** ：電子線照射による化学結合の切断
  * **熱損傷** ：非弾性散乱による試料加熱

#### コード例1-6: ノックオン損傷の閾値エネルギー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def knock_on_threshold_voltage(element_mass, displacement_energy_eV=25):
        """
        ノックオン損傷の閾値加速電圧を計算
    
        Parameters
        ----------
        element_mass : float
            元素の原子量 [u]
        displacement_energy_eV : float
            原子の変位エネルギー [eV]（典型値：25 eV）
    
        Returns
        -------
        V_threshold : float
            閾値加速電圧 [kV]
        """
        # 定数
        m_e = 9.10938e-31   # 電子質量 [kg]
        u = 1.66054e-27     # 原子質量単位 [kg]
        e = 1.60218e-19     # 電荷素量 [C]
        c = 2.99792e8       # 光速 [m/s]
    
        # 原子の質量 [kg]
        M = element_mass * u
    
        # ノックオン閾値エネルギー（相対論的）
        # E_threshold ≈ (M + 2m_e) / (2M) * E_d * (1 + E_d / (2 M c²))
        E_d = displacement_energy_eV * e  # [J]
    
        # 簡略式（非相対論的近似）
        E_threshold_J = (M / (2 * m_e)) * E_d
        V_threshold_kV = E_threshold_J / (e * 1000)
    
        return V_threshold_kV
    
    # 代表的な元素のノックオン閾値
    elements = {
        'C': 12, 'Si': 28, 'Fe': 56, 'Cu': 64, 'Mo': 96, 'W': 184, 'Au': 197
    }
    
    element_names = list(elements.keys())
    masses = list(elements.values())
    thresholds = [knock_on_threshold_voltage(m) for m in masses]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図：元素ごとの閾値
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(elements)))
    bars = ax1.bar(element_names, thresholds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 典型的な加速電圧を参考線として追加
    ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Low Voltage TEM (80 kV)')
    ax1.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Standard TEM (200 kV)')
    ax1.axhline(y=300, color='red', linestyle='--', linewidth=2, label='High Voltage TEM (300 kV)')
    
    ax1.set_ylabel('Knock-on Threshold Voltage [kV]', fontsize=12)
    ax1.set_title('Knock-on Damage Threshold for Elements', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 350)
    
    # 数値ラベル
    for bar, threshold in zip(bars, thresholds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 5, f'{threshold:.0f}',
                 ha='center', fontsize=9, fontweight='bold')
    
    # 右図：原子量 vs 閾値
    ax2.scatter(masses, thresholds, s=200, c=masses, cmap='plasma',
                edgecolors='black', linewidths=2, zorder=3)
    ax2.plot(masses, thresholds, 'k--', linewidth=1.5, alpha=0.5, zorder=1)
    
    for name, mass, threshold in zip(element_names, masses, thresholds):
        ax2.text(mass, threshold + 10, name, fontsize=10, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Atomic Mass [u]', fontsize=12)
    ax2.set_ylabel('Knock-on Threshold Voltage [kV]', fontsize=12)
    ax2.set_title('Threshold Voltage vs Atomic Mass', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("ノックオン損傷閾値電圧:")
    for name, mass, threshold in zip(element_names, masses, thresholds):
        print(f"  {name:2s} (M={mass:3.0f} u): {threshold:6.1f} kV")
    print("\n炭素（グラフェンなど）は80 kV以下で観察すれば損傷を低減できる")
    

## 1.6 演習問題

### 演習1-1: 電子波長の計算（Easy）

**問題** ：加速電圧100 kV、200 kV、300 kVでの電子波長を計算し、可視光（550 nm）との波長比を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    def calc_wavelength(V_kV):
        h = 6.62607e-34
        m0 = 9.10938e-31
        e = 1.60218e-19
        c = 2.99792e8
        E = V_kV * 1000 * e
        p = np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
        return (h / p) * 1e12  # [pm]
    
    voltages = [100, 200, 300]
    visible_light_nm = 550
    
    print("電子波長と可視光との比較:")
    for V in voltages:
        wl_pm = calc_wavelength(V)
        wl_nm = wl_pm / 1000
        ratio = visible_light_nm / wl_nm
        print(f"  {V} kV: λ = {wl_pm:.3f} pm = {wl_nm:.6f} nm, 比率 = {ratio:.0f}:1")
    

### 演習1-2: 分解能の比較（Medium）

**問題** ：光学顕微鏡（λ=550 nm, α=30°）と電子顕微鏡（200 kV, α=10°）の分解能をレイリー基準で比較せよ。

**解答例を表示**
    
    
    import numpy as np
    
    def resolution(wavelength_nm, alpha_deg):
        alpha_rad = np.deg2rad(alpha_deg)
        return 0.61 * wavelength_nm / np.sin(alpha_rad)
    
    # 光学顕微鏡
    optical_res = resolution(550, 30)
    
    # 電子顕微鏡（200 kV → 2.508 pm = 0.002508 nm）
    em_res = resolution(0.002508, 10)
    
    print(f"光学顕微鏡の分解能: {optical_res:.1f} nm")
    print(f"電子顕微鏡の分解能: {em_res:.5f} nm = {em_res * 10:.3f} Å")
    print(f"分解能向上: {optical_res / em_res:.0f}倍")
    

### 演習1-3: 散乱断面積の計算（Medium）

**問題** ：200 kVの電子線に対して、Si（Z=14）とAu（Z=79）の散乱角10°での散乱断面積比を求めよ。

**解答例を表示**
    
    
    Z_Si = 14
    Z_Au = 79
    E_keV = 200
    theta = 10
    
    # 散乱断面積は Z² に比例
    ratio = (Z_Au / Z_Si)**2
    
    print(f"Si (Z={Z_Si}) と Au (Z={Z_Au}) の散乱断面積比:")
    print(f"  σ_Au / σ_Si = ({Z_Au}/{Z_Si})² = {ratio:.1f}")
    print(f"  Auの散乱断面積はSiの約{ratio:.0f}倍")
    

### 演習1-4: 信号対雑音比の改善（Hard）

**問題** ：現在のS/N比が10である電子顕微鏡像を、S/N比100にするには、何倍の電子線量が必要か？また、10枚の画像を平均化した場合のS/N比改善効果を計算せよ。

**解答例を表示**
    
    
    import numpy as np
    
    # S/N = √N より、S/N比を10倍にするには線量を100倍にする必要がある
    current_snr = 10
    target_snr = 100
    dose_increase = (target_snr / current_snr)**2
    
    print(f"S/N比を{current_snr}から{target_snr}に改善するための線量増加:")
    print(f"  必要な線量増加: {dose_increase}倍")
    
    # 10枚平均化の効果
    num_images = 10
    snr_improvement = np.sqrt(num_images)
    new_snr = current_snr * snr_improvement
    
    print(f"\n10枚の画像を平均化した場合:")
    print(f"  S/N比改善係数: √{num_images} = {snr_improvement:.2f}")
    print(f"  新しいS/N比: {current_snr} × {snr_improvement:.2f} = {new_snr:.1f}")
    

### 演習1-5: ノックオン損傷の評価（Hard）

**問題** ：グラフェン（炭素、M=12）を200 kVのTEMで観察する際、ノックオン損傷が発生するか評価せよ。また、損傷を避けるための最適な加速電圧を提案せよ（変位エネルギー25 eV）。

**解答例を表示**
    
    
    import numpy as np
    
    def knock_on_threshold(M, E_d=25):
        m_e = 9.10938e-31
        u = 1.66054e-27
        e = 1.60218e-19
        M_kg = M * u
        E_d_J = E_d * e
        E_threshold = (M_kg / (2 * m_e)) * E_d_J
        return E_threshold / (e * 1000)  # [kV]
    
    M_C = 12
    threshold_kV = knock_on_threshold(M_C, E_d=25)
    operating_voltage = 200
    
    print(f"グラフェン（炭素、M={M_C}）のノックオン閾値:")
    print(f"  閾値電圧: {threshold_kV:.1f} kV")
    print(f"  観察電圧: {operating_voltage} kV")
    
    if operating_voltage > threshold_kV:
        print(f"  ⚠️ {operating_voltage} kVでの観察はノックオン損傷が発生する")
        print(f"  推奨電圧: {threshold_kV * 0.8:.0f} kV以下（安全マージン20%）")
    else:
        print(f"  ✅ {operating_voltage} kVでの観察は安全")
    

### 演習1-6: CTFのゼロクロス（Hard）

**問題** ：200 kVのTEM（Cs=1.0 mm、デフォーカス-100 nm）で、CTFの第1ゼロクロスの空間周波数を求めよ。これは実空間の分解能に対応する。

**解答例を表示**
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    voltage_kV = 200
    Cs_mm = 1.0
    defocus_nm = -100
    
    # 電子波長
    wavelength_pm = 2.508
    wavelength_A = wavelength_pm / 100
    
    # 単位変換
    Cs_A = Cs_mm * 1e7
    defocus_A = defocus_nm * 10
    
    # CTFの位相シフト χ(k) = 0となる最初の非自明な解を探す
    def chi(k):
        return (np.pi * wavelength_A / 2) * k**2 * (defocus_A + 0.5 * wavelength_A**2 * k**2 * Cs_A)
    
    # 第1ゼロクロス: sin(χ) = 0 → χ = π
    def equation(k):
        return chi(k) - np.pi
    
    k_zero = fsolve(equation, 2.0)[0]  # 初期推定値2.0
    resolution_A = 1 / k_zero
    
    print(f"200 kV TEM（Cs={Cs_mm} mm, Δf={defocus_nm} nm）:")
    print(f"  第1ゼロクロス空間周波数: {k_zero:.3f} 1/Å")
    print(f"  対応する実空間分解能: {resolution_A:.3f} Å = {resolution_A/10:.3f} nm")
    

### 演習1-7: 加速電圧の最適化（Hard）

**問題** ：厚さ100 nmのSi試料を観察する際、80 kVと200 kVの加速電圧でどちらがより良い透過像を得られるか、平均自由行程を考慮して議論せよ。

**解答例を表示**

**解答のポイント** ：

  * 平均自由行程$\lambda_{\text{mfp}}$は加速電圧が高いほど長くなる
  * 80 kV：$\lambda_{\text{mfp}} \approx 50$ nm → 多重散乱が発生、コントラストは高いが分解能は低下
  * 200 kV：$\lambda_{\text{mfp}} \approx 150$ nm → 単一散乱に近く、分解能が高い
  * **結論** ：高分解能が必要な場合は200 kV、コントラストが必要な場合は80 kVが有利

### 演習1-8: 実験計画（Hard）

**問題** ：未知の金属ナノ粒子（直径10 nm）を電子顕微鏡で解析する実験計画を立案せよ。加速電圧、観察モード（TEM/SEM/STEM）、分析手法（EDS/EELS）の選択理由を説明せよ。

**解答例を表示**

**実験計画** ：

  1. **SEM観察（低倍率）** ：ナノ粒子の分散状態、形態を確認（加速電圧5-10 kV）
  2. **TEM観察（明視野像）** ：粒子サイズ分布、結晶性の判定（加速電圧200 kV）
  3. **SAED（制限視野電子回折）** ：結晶構造の同定
  4. **HRTEM** ：格子像観察で面間隔を測定、結晶方位を決定
  5. **EDS分析** ：元素組成を定量（加速電圧200 kV、炭素支持膜上で観察）
  6. **EELS（オプション）** ：表面酸化層の有無、価数状態の確認

**理由** ：

  * 200 kVは、10 nmナノ粒子に対して十分な分解能と透過能力を持つ
  * 明視野像で形態、SAEDで構造、HRTEMで原子配列、EDSで組成を総合的に解析
  * EELSは表面敏感性が高く、酸化層の検出に有効

## 1.7 学習チェック

以下の質問に答えて、理解度を確認しましょう：

  1. ド・ブロイ波長の式を導出し、加速電圧との関係を説明できますか？
  2. レイリー基準による分解能の定義を理解し、光学顕微鏡と電子顕微鏡の違いを説明できますか？
  3. 弾性散乱と非弾性散乱の物理的メカニズムと検出信号を説明できますか？
  4. 散乱断面積の原子番号依存性（$Z^2$）を理解していますか？
  5. 電子銃の種類（タングステン、LaB₆、FEG）とその特徴を説明できますか？
  6. 球面収差と色収差が像に与える影響を理解していますか？
  7. 信号対雑音比とポアソン統計の関係を説明できますか？
  8. ノックオン損傷の閾値エネルギーと試料損傷低減の戦略を理解していますか？

## 1.8 参考文献

  1. Williams, D. B., & Carter, C. B. (2009). _Transmission Electron Microscopy: A Textbook for Materials Science_ (2nd ed.). Springer. - 電子顕微鏡学の包括的教科書
  2. Reimer, L., & Kohl, H. (2008). _Transmission Electron Microscopy: Physics of Image Formation_ (5th ed.). Springer. - 結像理論の詳細
  3. Goldstein, J. I., et al. (2017). _Scanning Electron Microscopy and X-Ray Microanalysis_ (4th ed.). Springer. - SEMとEDS分析の標準テキスト
  4. Egerton, R. F. (2011). _Electron Energy-Loss Spectroscopy in the Electron Microscope_ (3rd ed.). Springer. - EELS技術の詳細
  5. Spence, J. C. H. (2013). _High-Resolution Electron Microscopy_ (4th ed.). Oxford University Press. - HRTEM理論
  6. Kirkland, E. J. (2020). _Advanced Computing in Electron Microscopy_ (3rd ed.). Springer. - 像シミュレーション
  7. De Graef, M. (2003). _Introduction to Conventional Transmission Electron Microscopy_. Cambridge University Press. - TEM入門書

## 1.9 次章へ

次章では、走査型電子顕微鏡（SEM）の原理、二次電子像（SE）と後方散乱電子像（BSE）、エネルギー分散型X線分析（EDS）、画像解析の基礎を学びます。SEMは、試料表面の形態観察と元素分析を高い空間分解能で行える汎用性の高い手法です。
