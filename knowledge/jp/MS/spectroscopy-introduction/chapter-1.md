---
title: 第1章：分光法の基礎
chapter_title: 第1章：分光法の基礎
subtitle: 光と物質の相互作用から理解する分光分析の原理
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/pEHTAkGLAO0"
    title="分光分析入門 第1章: 分光法の基礎"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

## イントロダクション

分光法（Spectroscopy）は、光（電磁波）と物質の相互作用を利用して、物質の構造、組成、電子状態を解析する強力な分析手法です。材料科学において、分光法は非破壊的に試料の特性を評価できる点で極めて重要な役割を果たしています。

この章では、分光法を理解するための基礎として、光と物質の相互作用（吸収、発光、散乱）、電磁波スペクトルの各領域、エネルギー量子化と遷移、選択則と遷移確率、そしてスペクトル分解能と装置の基礎について学びます。Pythonを用いた波長・エネルギー変換やスペクトルプロットの実践的なコード例も紹介します。

**この章で学ぶこと**

  * 光と物質の相互作用の3つの基本過程（吸収、発光、散乱）
  * 電磁波スペクトルの各領域（紫外、可視、赤外、X線）の特徴
  * エネルギー量子化と遷移の量子力学的基礎
  * 選択則と遷移確率の概念
  * スペクトル分解能と測定装置の基礎
  * Pythonによる波長・エネルギー変換と基本的なスペクトルプロット

## 1\. 光と物質の相互作用

### 1.1 電磁波の基本性質

光は電磁波の一種であり、波動性と粒子性の二重性を持ちます。電磁波は電場と磁場が互いに直交しながら空間を伝播する横波です。

電磁波の基本的な物理量として、以下が重要です：

  * **波長（$\lambda$）** ：波の1周期の長さ。単位はnm、$\mu$m、mなど
  * **振動数（周波数、$\nu$）** ：単位時間あたりの振動回数。単位はHz（= s$^{-1}$）
  * **波数（$\tilde{\nu}$）** ：単位長さあたりの波の数。単位はcm$^{-1}$。特に赤外分光で多用される
  * **光子エネルギー（$E$）** ：光の粒子（光子）が持つエネルギー。単位はeV、Jなど

これらの物理量は以下の関係式で結ばれています：

$$c = \lambda \nu$$

$$E = h\nu = \frac{hc}{\lambda} = hc\tilde{\nu}$$

ここで、$c = 2.998 \times 10^8$ m/s（真空中の光速）、$h = 6.626 \times 10^{-34}$ J$\cdot$s（プランク定数）です。

#### コード例1：波長・振動数・波数・エネルギーの相互変換
    
    
    import numpy as np
    
    # 物理定数
    h = 6.62607015e-34  # プランク定数 (J*s)
    c = 2.99792458e8    # 光速 (m/s)
    eV_to_J = 1.602176634e-19  # 1 eV = この値 J
    
    def wavelength_to_frequency(wavelength_nm):
        """波長(nm) -> 振動数(Hz)"""
        wavelength_m = wavelength_nm * 1e-9
        return c / wavelength_m
    
    def wavelength_to_wavenumber(wavelength_nm):
        """波長(nm) -> 波数(cm^-1)"""
        wavelength_cm = wavelength_nm * 1e-7
        return 1.0 / wavelength_cm
    
    def wavelength_to_energy_eV(wavelength_nm):
        """波長(nm) -> エネルギー(eV)"""
        wavelength_m = wavelength_nm * 1e-9
        energy_J = h * c / wavelength_m
        return energy_J / eV_to_J
    
    def wavenumber_to_wavelength(wavenumber_cm):
        """波数(cm^-1) -> 波長(nm)"""
        wavelength_cm = 1.0 / wavenumber_cm
        return wavelength_cm * 1e7
    
    def energy_eV_to_wavelength(energy_eV):
        """エネルギー(eV) -> 波長(nm)"""
        energy_J = energy_eV * eV_to_J
        wavelength_m = h * c / energy_J
        return wavelength_m * 1e9
    
    # 使用例
    print("=== 波長・エネルギー変換の例 ===")
    print()
    
    # 可視光の例
    wavelengths = [400, 500, 600, 700]  # nm
    colors = ["紫", "青緑", "橙", "赤"]
    
    print("可視光領域：")
    print(f"{'波長(nm)':<12} {'色':<8} {'振動数(THz)':<15} {'波数(cm^-1)':<15} {'エネルギー(eV)':<15}")
    print("-" * 70)
    
    for wl, color in zip(wavelengths, colors):
        freq = wavelength_to_frequency(wl) / 1e12  # THz
        wn = wavelength_to_wavenumber(wl)
        energy = wavelength_to_energy_eV(wl)
        print(f"{wl:<12} {color:<8} {freq:<15.2f} {wn:<15.0f} {energy:<15.3f}")
    
    print()
    print("=== 赤外領域の例（波数 -> 波長）===")
    wavenumbers_ir = [4000, 3000, 1700, 1000, 500]  # cm^-1
    print(f"{'波数(cm^-1)':<15} {'波長(um)':<15} {'エネルギー(eV)':<15}")
    print("-" * 50)
    
    for wn in wavenumbers_ir:
        wl_nm = wavenumber_to_wavelength(wn)
        wl_um = wl_nm / 1000
        energy = wavelength_to_energy_eV(wl_nm)
        print(f"{wn:<15} {wl_um:<15.2f} {energy:<15.4f}")
    

### 1.2 吸収（Absorption）

吸収は、物質が光子を受け取り、そのエネルギーを使って低いエネルギー状態から高いエネルギー状態へ遷移する過程です。吸収が起こるためには、光子のエネルギー $E = h\nu$ が、物質の2つのエネルギー準位の差 $\Delta E = E_2 - E_1$ と一致する必要があります（共鳴条件）。
    
    
    ```mermaid
    graph TD
        A[基底状態 E1] -->|光子吸収| B[励起状態 E2]
        B -->|"条件: hv = E2 - E1"| A
    
        style A fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#333,stroke-width:2px,color:#fff
    ```

吸収の強さを定量的に記述するのがBeer-Lambert則（ベール・ランベールの法則）です：

$$A = \log_{10}\left(\frac{I_0}{I}\right) = \varepsilon c l$$

ここで、$A$は吸光度、$I_0$は入射光強度、$I$は透過光強度、$\varepsilon$はモル吸光係数（L mol$^{-1}$ cm$^{-1}$）、$c$は濃度（mol/L）、$l$は光路長（cm）です。透過率 $T$ は $T = I/I_0$ で定義され、$A = -\log_{10}(T)$ の関係があります。

### 1.3 発光（Emission）

発光は、励起状態にある物質が低いエネルギー状態へ遷移する際に、そのエネルギー差に相当する光子を放出する過程です。発光には主に以下の種類があります：

  * **蛍光（Fluorescence）** ：スピン許容遷移（一重項 -> 一重項）による発光。励起後、ナノ秒（$10^{-9}$ s）程度で発光
  * **燐光（Phosphorescence）** ：スピン禁制遷移（三重項 -> 一重項）による発光。ミリ秒から秒単位で持続
  * **化学発光（Chemiluminescence）** ：化学反応によって生成した励起種からの発光

### 1.4 散乱（Scattering）

散乱は、光が物質と相互作用して方向を変える過程です。散乱には以下の種類があります：

  * **レイリー散乱（Rayleigh scattering）** ：弾性散乱。散乱光の振動数は入射光と同じ。散乱強度は波長の4乗に反比例（$I \propto 1/\lambda^4$）。空が青い理由
  * **ラマン散乱（Raman scattering）** ：非弾性散乱。分子振動とのエネルギー交換により、散乱光の振動数が変化。ストークス散乱（エネルギー減少）とアンチストークス散乱（エネルギー増加）がある

    
    
    ```mermaid
    graph LR
        subgraph "光と物質の相互作用"
        A[入射光] --> B{物質}
        B --> C[吸収<br/>エネルギー吸収]
        B --> D[発光<br/>エネルギー放出]
        B --> E[散乱<br/>方向変化]
        E --> F[レイリー<br/>弾性]
        E --> G[ラマン<br/>非弾性]
        end
    
        style A fill:#4ecdc4,stroke:#333,stroke-width:2px
        style C fill:#f093fb,stroke:#333,stroke-width:2px
        style D fill:#f5576c,stroke:#333,stroke-width:2px
        style F fill:#ffe66d,stroke:#333,stroke-width:2px
        style G fill:#a8e6cf,stroke:#333,stroke-width:2px
    ```

#### コード例2：Beer-Lambert則のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def beer_lambert(I0, epsilon, concentration, path_length):
        """
        Beer-Lambert則による透過光強度の計算
    
        Parameters:
        -----------
        I0 : float
            入射光強度
        epsilon : float
            モル吸光係数 (L mol^-1 cm^-1)
        concentration : float or array
            濃度 (mol/L)
        path_length : float
            光路長 (cm)
    
        Returns:
        --------
        I : float or array
            透過光強度
        A : float or array
            吸光度
        T : float or array
            透過率 (0-1)
        """
        A = epsilon * concentration * path_length
        T = 10**(-A)
        I = I0 * T
        return I, A, T
    
    # パラメータ設定
    I0 = 1.0  # 入射光強度（規格化）
    epsilon = 10000  # モル吸光係数（典型的な有機色素の値）
    path_length = 1.0  # 光路長 1 cm
    
    # 濃度範囲
    concentrations = np.linspace(0, 1e-4, 100)  # 0 - 100 uM
    
    # 計算
    I, A, T = beer_lambert(I0, epsilon, concentrations, path_length)
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 透過光強度 vs 濃度
    axes[0].plot(concentrations * 1e6, I, 'b-', linewidth=2)
    axes[0].set_xlabel('Concentration (uM)', fontsize=12)
    axes[0].set_ylabel('Transmitted Intensity', fontsize=12)
    axes[0].set_title('Transmitted Intensity vs Concentration', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 吸光度 vs 濃度
    axes[1].plot(concentrations * 1e6, A, 'r-', linewidth=2)
    axes[1].set_xlabel('Concentration (uM)', fontsize=12)
    axes[1].set_ylabel('Absorbance', fontsize=12)
    axes[1].set_title('Absorbance vs Concentration (Linear)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 透過率 vs 濃度
    axes[2].plot(concentrations * 1e6, T * 100, 'g-', linewidth=2)
    axes[2].set_xlabel('Concentration (uM)', fontsize=12)
    axes[2].set_ylabel('Transmittance (%)', fontsize=12)
    axes[2].set_title('Transmittance vs Concentration', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('beer_lambert_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 数値例
    print("=== Beer-Lambert則の計算例 ===")
    test_concentrations = [10, 25, 50, 100]  # uM
    print(f"モル吸光係数: {epsilon} L mol^-1 cm^-1")
    print(f"光路長: {path_length} cm")
    print()
    print(f"{'濃度(uM)':<12} {'吸光度':<12} {'透過率(%)':<12}")
    print("-" * 36)
    for c_um in test_concentrations:
        c_mol = c_um * 1e-6
        _, A_val, T_val = beer_lambert(I0, epsilon, c_mol, path_length)
        print(f"{c_um:<12} {A_val:<12.3f} {T_val*100:<12.1f}")
    

## 2\. 電磁波スペクトル

### 2.1 電磁波スペクトルの領域

電磁波は波長（またはエネルギー）によって様々な領域に分類されます。各領域は物質の異なる性質を探るために使用されます。

領域 | 波長範囲 | エネルギー範囲 | 関連する遷移 | 主な分光法  
---|---|---|---|---  
ガンマ線 | < 0.01 nm | > 100 keV | 原子核遷移 | メスバウアー分光  
X線 | 0.01 - 10 nm | 0.1 - 100 keV | 内殻電子遷移 | XPS, XRF, XAFS  
真空紫外 | 10 - 200 nm | 6 - 120 eV | 価電子遷移 | VUV分光  
紫外（UV） | 200 - 400 nm | 3 - 6 eV | 電子遷移 | UV-Vis  
可視光 | 400 - 800 nm | 1.5 - 3 eV | 電子遷移 | UV-Vis, 蛍光  
近赤外（NIR） | 0.8 - 2.5 um | 0.5 - 1.5 eV | 倍音・結合音 | NIR分光  
中赤外（MIR） | 2.5 - 25 um | 0.05 - 0.5 eV | 分子振動 | IR, FTIR  
遠赤外（FIR） | 25 - 1000 um | 1 - 50 meV | 格子振動 | THz分光  
マイクロ波 | 1 mm - 1 m | 1 ueV - 1 meV | 分子回転 | マイクロ波分光  
  
#### コード例3：電磁波スペクトルの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    
    # 電磁波スペクトルのデータ
    spectrum_data = [
        {'name': 'Gamma Ray', 'lambda_min': 1e-5, 'lambda_max': 0.01, 'color': '#9b59b6'},
        {'name': 'X-ray', 'lambda_min': 0.01, 'lambda_max': 10, 'color': '#8e44ad'},
        {'name': 'VUV', 'lambda_min': 10, 'lambda_max': 200, 'color': '#3498db'},
        {'name': 'UV', 'lambda_min': 200, 'lambda_max': 400, 'color': '#9b59b6'},
        {'name': 'Visible', 'lambda_min': 400, 'lambda_max': 700, 'color': 'rainbow'},
        {'name': 'NIR', 'lambda_min': 700, 'lambda_max': 2500, 'color': '#e74c3c'},
        {'name': 'MIR', 'lambda_min': 2500, 'lambda_max': 25000, 'color': '#c0392b'},
        {'name': 'FIR', 'lambda_min': 25000, 'lambda_max': 1e6, 'color': '#95a5a6'},
        {'name': 'Microwave', 'lambda_min': 1e6, 'lambda_max': 1e9, 'color': '#7f8c8d'},
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 対数スケールで波長軸を設定
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1e9)
    ax.set_ylim(0, 1)
    
    # 各領域を描画
    y_bottom = 0.3
    y_height = 0.4
    
    for i, region in enumerate(spectrum_data):
        if region['name'] == 'Visible':
            # 可視光領域は虹色のグラデーション
            n_colors = 100
            wavelengths = np.linspace(region['lambda_min'], region['lambda_max'], n_colors)
            for j in range(n_colors - 1):
                # 波長から色を計算（簡易的な変換）
                wl = wavelengths[j]
                if wl < 450:
                    r, g, b = 0.5, 0, 1
                elif wl < 500:
                    r, g, b = 0, 0.5, 1
                elif wl < 550:
                    r, g, b = 0, 1, 0.5
                elif wl < 600:
                    r, g, b = 1, 1, 0
                elif wl < 650:
                    r, g, b = 1, 0.5, 0
                else:
                    r, g, b = 1, 0, 0
    
                rect = Rectangle((wavelengths[j], y_bottom),
                                wavelengths[j+1] - wavelengths[j],
                                y_height, color=(r, g, b), alpha=0.8)
                ax.add_patch(rect)
        else:
            rect = Rectangle((region['lambda_min'], y_bottom),
                             region['lambda_max'] - region['lambda_min'],
                             y_height, color=region['color'], alpha=0.7)
            ax.add_patch(rect)
    
        # ラベル
        center = np.sqrt(region['lambda_min'] * region['lambda_max'])
        ax.text(center, 0.8, region['name'], ha='center', va='bottom',
                fontsize=10, fontweight='bold', rotation=45)
    
    # 軸ラベル
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Electromagnetic Spectrum', fontsize=14, fontweight='bold')
    
    # エネルギー軸（上部）を追加
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(wavelength_to_energy_eV(1e9), wavelength_to_energy_eV(1e-5))
    ax2.set_xlabel('Photon Energy (eV)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('electromagnetic_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
    

### 2.2 各領域で観測される現象

  * **X線領域** ：内殻電子の励起。元素固有のエネルギーで吸収・発光が起こるため、元素分析に利用（XPS、XRF）
  * **紫外・可視領域** ：価電子の励起（HOMO-LUMO遷移など）。有機分子の共役系、遷移金属錯体のd-d遷移、半導体のバンド間遷移を観測
  * **赤外領域** ：分子振動の励起。官能基の同定、分子構造解析に利用
  * **マイクロ波領域** ：分子回転の励起。気相分子の構造決定に利用

## 3\. エネルギー量子化と遷移

### 3.1 量子化されたエネルギー準位

量子力学によれば、原子や分子のエネルギーは連続的ではなく、離散的な値（量子化されたエネルギー準位）のみを取ります。これが分光法の基礎となります。

分子のエネルギーは、電子エネルギー、振動エネルギー、回転エネルギーの和として表されます（Born-Oppenheimer近似）：

$$E_{\text{total}} = E_{\text{electronic}} + E_{\text{vibrational}} + E_{\text{rotational}}$$

各エネルギーのスケールは：

  * 電子エネルギー：1 - 10 eV（紫外・可視領域）
  * 振動エネルギー：0.01 - 0.5 eV（赤外領域）
  * 回転エネルギー：$10^{-5}$ - $10^{-3}$ eV（マイクロ波領域）

**調和振動子のエネルギー準位**

二原子分子の振動は調和振動子モデルで近似でき、そのエネルギー準位は：

$$E_v = \left(v + \frac{1}{2}\right)h\nu_0 \quad (v = 0, 1, 2, \ldots)$$

ここで、$v$は振動量子数、$\nu_0$は振動の固有振動数です。基底状態（$v = 0$）でも $\frac{1}{2}h\nu_0$ のゼロ点エネルギーを持つことが量子力学の特徴です。

### 3.2 遷移とスペクトル

光の吸収・発光は、物質のエネルギー準位間の遷移に対応します。遷移が起こるためには、光子のエネルギーが2つの準位のエネルギー差と一致する必要があります：

$$h\nu = |E_f - E_i|$$

ここで、$E_i$は始状態、$E_f$は終状態のエネルギーです。

#### コード例4：水素原子のエネルギー準位と遷移スペクトル
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hydrogen_energy(n):
        """
        水素原子のエネルギー準位（eV）
        E_n = -13.6 / n^2 eV
        """
        return -13.6 / n**2
    
    def transition_wavelength(n_i, n_f):
        """
        遷移に対応する波長（nm）を計算
        """
        E_i = hydrogen_energy(n_i)
        E_f = hydrogen_energy(n_f)
        delta_E = abs(E_f - E_i)  # eV
    
        # E = hc/lambda より lambda = hc/E
        wavelength_nm = 1239.8 / delta_E  # 1239.8 eV*nm = hc
        return wavelength_nm, delta_E
    
    # エネルギー準位図
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左：エネルギー準位図
    ax1 = axes[0]
    n_levels = 7
    
    for n in range(1, n_levels + 1):
        E = hydrogen_energy(n)
        ax1.hlines(E, 0.2, 0.8, colors='blue', linewidth=2)
        ax1.text(0.85, E, f'n = {n}', va='center', fontsize=10)
        ax1.text(0.1, E, f'{E:.2f} eV', va='center', ha='right', fontsize=9)
    
    # 遷移を矢印で表示
    transitions = [
        (2, 1, 'red', 'Lyman alpha'),
        (3, 1, 'purple', 'Lyman beta'),
        (3, 2, 'red', 'Balmer alpha (H-alpha)'),
        (4, 2, 'cyan', 'Balmer beta'),
        (4, 3, 'orange', 'Paschen alpha'),
    ]
    
    for n_i, n_f, color, label in transitions:
        E_i = hydrogen_energy(n_i)
        E_f = hydrogen_energy(n_f)
        x_pos = 0.35 + (n_f - 1) * 0.08
        ax1.annotate('', xy=(x_pos, E_f), xytext=(x_pos, E_i),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-15, 1)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('Hydrogen Atom Energy Levels', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右：遷移スペクトル
    ax2 = axes[1]
    
    # 様々な系列の遷移を計算
    series = {
        'Lyman': {'n_f': 1, 'n_i_range': range(2, 8), 'color': 'purple'},
        'Balmer': {'n_f': 2, 'n_i_range': range(3, 8), 'color': 'red'},
        'Paschen': {'n_f': 3, 'n_i_range': range(4, 8), 'color': 'orange'},
    }
    
    y_offset = 0
    for series_name, params in series.items():
        wavelengths = []
        intensities = []
        for n_i in params['n_i_range']:
            wl, dE = transition_wavelength(n_i, params['n_f'])
            wavelengths.append(wl)
            # 強度は遷移確率に比例（簡略化）
            intensities.append(1.0 / (n_i - params['n_f'])**2)
    
        # スペクトル線をプロット
        for wl, intensity in zip(wavelengths, intensities):
            ax2.vlines(wl, y_offset, y_offset + intensity * 0.8,
                      colors=params['color'], linewidth=2)
    
        ax2.text(max(wavelengths) + 50, y_offset + 0.4, series_name,
                fontsize=10, color=params['color'])
        y_offset += 1.2
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_title('Hydrogen Emission Spectrum Series', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(-0.2, 4)
    ax2.grid(True, alpha=0.3)
    
    # 可視光領域をハイライト
    ax2.axvspan(400, 700, alpha=0.2, color='yellow', label='Visible region')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('hydrogen_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 主要な遷移の波長を出力
    print("=== 水素原子の主要な遷移 ===")
    print()
    print("Balmer系列（可視光領域）：")
    print(f"{'遷移':<12} {'波長(nm)':<12} {'エネルギー(eV)':<15}")
    print("-" * 40)
    for n_i in range(3, 8):
        wl, dE = transition_wavelength(n_i, 2)
        print(f"n={n_i} -> n=2   {wl:<12.1f} {dE:<15.3f}")
    

## 4\. 選択則と遷移確率

### 4.1 遷移双極子モーメント

量子力学において、光と物質の相互作用は時間依存摂動論で記述されます。電気双極子近似の下で、状態 $|i\rangle$ から状態 $|f\rangle$ への遷移確率は、遷移双極子モーメント $\boldsymbol{\mu}_{fi}$ の2乗に比例します：

$$\boldsymbol{\mu}_{fi} = \langle f | \hat{\boldsymbol{\mu}} | i \rangle = \int \psi_f^* \hat{\boldsymbol{\mu}} \psi_i \, d\tau$$

ここで、$\hat{\boldsymbol{\mu}} = -e\sum_j \boldsymbol{r}_j$ は電気双極子モーメント演算子です。

### 4.2 選択則

選択則（Selection Rules）は、遷移双極子モーメントが非ゼロとなる条件を規定します。遷移双極子モーメントがゼロの遷移は「禁制遷移」、非ゼロの遷移は「許容遷移」と呼ばれます。

**主要な選択則**

  * **電気双極子遷移（原子）** ：$\Delta l = \pm 1$, $\Delta m_l = 0, \pm 1$, $\Delta s = 0$
  * **電気双極子遷移（分子振動）** ：$\Delta v = \pm 1$（調和振動子近似）
  * **Laporte則** ：中心対称を持つ分子では、g（偶）$\leftrightarrow$ u（奇）遷移のみ許容
  * **スピン選択則** ：$\Delta S = 0$（スピン多重度の変化なし）
  * **赤外活性** ：振動による双極子モーメントの変化 $(\partial \mu / \partial Q) \neq 0$
  * **ラマン活性** ：振動による分極率の変化 $(\partial \alpha / \partial Q) \neq 0$

### 4.3 Fermiの黄金則

遷移確率は、Fermiの黄金則によって定量的に記述されます：

$$W_{i \to f} = \frac{2\pi}{\hbar} |\langle f | \hat{H}' | i \rangle|^2 \rho(E_f)$$

ここで、$\hat{H}'$ は摂動ハミルトニアン（光と物質の相互作用）、$\rho(E_f)$ は終状態のエネルギーにおける状態密度です。

#### コード例5：調和振動子の選択則の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import hermite
    from math import factorial
    
    def harmonic_oscillator_wavefunction(x, n, alpha=1.0):
        """
        調和振動子の波動関数
        psi_n(x) = N_n * H_n(alpha*x) * exp(-alpha^2 * x^2 / 2)
        """
        H_n = hermite(n)
        normalization = (alpha / np.pi)**0.25 / np.sqrt(2**n * factorial(n))
        return normalization * H_n(alpha * x) * np.exp(-alpha**2 * x**2 / 2)
    
    def transition_dipole_moment(n_i, n_f, x, alpha=1.0):
        """
        調和振動子間の遷移双極子モーメント積分
        mu_fi = 
        """
        psi_i = harmonic_oscillator_wavefunction(x, n_i, alpha)
        psi_f = harmonic_oscillator_wavefunction(x, n_f, alpha)
        integrand = psi_f * x * psi_i
        return np.trapz(integrand, x)
    
    # 計算用のグリッド
    x = np.linspace(-6, 6, 1000)
    
    # 波動関数のプロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：波動関数
    ax1 = axes[0, 0]
    for n in range(5):
        psi = harmonic_oscillator_wavefunction(x, n)
        offset = n + 0.5  # 視覚的なオフセット
        ax1.plot(x, psi * 2 + offset, label=f'n = {n}', linewidth=2)
        ax1.axhline(y=offset, color='gray', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Energy Level / Wavefunction', fontsize=12)
    ax1.set_title('Harmonic Oscillator Wavefunctions', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右上：遷移双極子モーメントの行列
    ax2 = axes[0, 1]
    n_max = 6
    mu_matrix = np.zeros((n_max, n_max))
    
    for n_i in range(n_max):
        for n_f in range(n_max):
            mu_matrix[n_i, n_f] = transition_dipole_moment(n_i, n_f, x)
    
    im = ax2.imshow(np.abs(mu_matrix), cmap='hot', aspect='equal')
    ax2.set_xlabel('Final State n_f', fontsize=12)
    ax2.set_ylabel('Initial State n_i', fontsize=12)
    ax2.set_title('|Transition Dipole Moment| Matrix', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(n_max))
    ax2.set_yticks(range(n_max))
    plt.colorbar(im, ax=ax2, label='||')
    
    # 左下：許容遷移 (Delta n = +1) の可視化
    ax3 = axes[1, 0]
    n_i = 0
    n_f = 1
    psi_i = harmonic_oscillator_wavefunction(x, n_i)
    psi_f = harmonic_oscillator_wavefunction(x, n_f)
    integrand = psi_f * x * psi_i
    
    ax3.plot(x, psi_i, 'b-', linewidth=2, label=f'psi_{n_i}')
    ax3.plot(x, psi_f, 'r-', linewidth=2, label=f'psi_{n_f}')
    ax3.plot(x, integrand * 5, 'g-', linewidth=2, label=f'psi_{n_f}*x*psi_{n_i} (x5)')
    ax3.fill_between(x, integrand * 5, alpha=0.3, color='green')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Position x', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.set_title(f'Allowed Transition: n={n_i} -> n={n_f}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    mu = transition_dipole_moment(n_i, n_f, x)
    ax3.text(0.05, 0.95, f'Transition Dipole = {mu:.4f}',
            transform=ax3.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 右下：禁制遷移 (Delta n = +2) の可視化
    ax4 = axes[1, 1]
    n_i = 0
    n_f = 2
    psi_i = harmonic_oscillator_wavefunction(x, n_i)
    psi_f = harmonic_oscillator_wavefunction(x, n_f)
    integrand = psi_f * x * psi_i
    
    ax4.plot(x, psi_i, 'b-', linewidth=2, label=f'psi_{n_i}')
    ax4.plot(x, psi_f, 'r-', linewidth=2, label=f'psi_{n_f}')
    ax4.plot(x, integrand * 5, 'g-', linewidth=2, label=f'psi_{n_f}*x*psi_{n_i} (x5)')
    ax4.fill_between(x, integrand * 5, alpha=0.3, color='green')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Position x', fontsize=12)
    ax4.set_ylabel('Amplitude', fontsize=12)
    ax4.set_title(f'Forbidden Transition: n={n_i} -> n={n_f}', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    mu = transition_dipole_moment(n_i, n_f, x)
    ax4.text(0.05, 0.95, f'Transition Dipole = {mu:.6f} (~ 0)',
            transform=ax4.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('selection_rules.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 遷移双極子モーメントの数値確認
    print("=== 調和振動子の遷移双極子モーメント ===")
    print()
    print("選択則: Delta n = +/- 1 のみ許容")
    print()
    print(f"{'遷移':<15} {'||':<15} {'許容/禁制':<15}")
    print("-" * 45)
    for n_i in range(4):
        for n_f in range(4):
            if n_i != n_f:
                mu = abs(transition_dipole_moment(n_i, n_f, x))
                status = "許容" if abs(n_f - n_i) == 1 else "禁制"
                print(f"n={n_i} -> n={n_f}     {mu:<15.6f} {status:<15}")
    

## 5\. スペクトル分解能と装置の基礎

### 5.1 スペクトル分解能

スペクトル分解能（Spectral Resolution）は、近接した2つのスペクトル線を区別できる能力を表します。分解能は通常、以下のいずれかで定義されます：

  * **波長分解能** ：$\Delta\lambda$（nm）- 区別可能な最小波長差
  * **波数分解能** ：$\Delta\tilde{\nu}$（cm$^{-1}$）- 区別可能な最小波数差
  * **分解能パラメータ** ：$R = \lambda / \Delta\lambda$ - 無次元量

分解能は装置の設計（スリット幅、回折格子、干渉計のパス差など）によって決まります。

### 5.2 スペクトル線の広がり

実際のスペクトル線は理想的なデルタ関数ではなく、有限の幅を持ちます。この広がりには以下の要因があります：

  * **自然幅（Natural linewidth）** ：励起状態の有限寿命による広がり。$\Delta E \cdot \tau \geq \hbar/2$（不確定性原理）
  * **ドップラー広がり** ：熱運動による速度分布に起因。Gaussian型の線形
  * **衝突広がり（圧力広がり）** ：分子間衝突による位相緩和。Lorentzian型の線形
  * **装置関数** ：測定装置の分解能限界による広がり

**線形関数**

  * **Gaussian（ガウス関数）** ：$G(x) = A \exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$。ドップラー広がりに対応。FWHM = $2\sqrt{2\ln 2}\sigma \approx 2.355\sigma$
  * **Lorentzian（ローレンツ関数）** ：$L(x) = \frac{A\gamma^2}{(x-x_0)^2 + \gamma^2}$。自然幅・衝突広がりに対応。FWHM = $2\gamma$
  * **Voigt関数** ：GaussianとLorentzianの畳み込み。両方の広がり機構が共存する場合

#### コード例6：線形関数（Gaussian, Lorentzian, Voigt）の比較
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import voigt_profile
    
    def gaussian(x, amplitude, center, sigma):
        """Gaussian関数"""
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian関数"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    def voigt(x, amplitude, center, sigma, gamma):
        """Voigt関数（scipy使用）"""
        # voigt_profile は規格化されているので振幅調整
        v = voigt_profile(x - center, sigma, gamma)
        return amplitude * v / np.max(v)
    
    # パラメータ
    x = np.linspace(-10, 10, 1000)
    center = 0
    amplitude = 1.0
    sigma = 1.5  # Gaussianの幅パラメータ
    gamma = 1.0  # Lorentzianの幅パラメータ
    
    # 各線形を計算
    g = gaussian(x, amplitude, center, sigma)
    l = lorentzian(x, amplitude, center, gamma)
    v = voigt(x, amplitude, center, sigma, gamma)
    
    # FWHM計算
    fwhm_gaussian = 2.355 * sigma
    fwhm_lorentzian = 2 * gamma
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 線形比較
    ax1 = axes[0]
    ax1.plot(x, g, 'b-', linewidth=2, label=f'Gaussian (FWHM={fwhm_gaussian:.2f})')
    ax1.plot(x, l, 'r-', linewidth=2, label=f'Lorentzian (FWHM={fwhm_lorentzian:.2f})')
    ax1.plot(x, v, 'g-', linewidth=2, label='Voigt')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Half Maximum')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Lineshape Functions (Linear Scale)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 対数スケールで尾部の違いを強調
    ax2 = axes[1]
    ax2.semilogy(x, g, 'b-', linewidth=2, label='Gaussian')
    ax2.semilogy(x, l, 'r-', linewidth=2, label='Lorentzian')
    ax2.semilogy(x, v, 'g-', linewidth=2, label='Voigt')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Intensity (log scale)', fontsize=12)
    ax2.set_title('Lineshape Functions (Log Scale) - Wing Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(1e-4, 2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lineshape_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== 線形関数の特徴 ===")
    print()
    print("Gaussian: 中心付近で急峻、尾部が速く減衰")
    print("Lorentzian: 中心付近でやや緩やか、尾部が長い（裾野が広い）")
    print("Voigt: 両者の特徴を持つ（中間的な振る舞い）")
    

### 5.3 分光装置の基本構成

典型的な分光装置は以下の要素で構成されます：

  * **光源** ：連続光源（白色光、重水素ランプ、ハロゲンランプ）、線光源（水銀ランプ、レーザー）
  * **分光器** ：光を波長ごとに分離する装置。回折格子、プリズム、干渉計（FTIR用）など
  * **試料室** ：試料を設置し、光と相互作用させる部分
  * **検出器** ：光強度を電気信号に変換。PMT、CCD、MCT（赤外用）など

    
    
    ```mermaid
    graph LR
        A[光源] --> B[モノクロメーター<br/>または干渉計]
        B --> C[試料]
        C --> D[検出器]
        D --> E[データ処理]
    
        style A fill:#4ecdc4,stroke:#333,stroke-width:2px
        style B fill:#f093fb,stroke:#333,stroke-width:2px
        style C fill:#ffe66d,stroke:#333,stroke-width:2px
        style D fill:#a8e6cf,stroke:#333,stroke-width:2px
        style E fill:#f5576c,stroke:#333,stroke-width:2px
    ```

#### コード例7：単純な吸収スペクトルのシミュレーションとプロット
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_absorption_spectrum(wavelengths, peaks):
        """
        複数のGaussianピークからなる吸収スペクトルをシミュレート
    
        Parameters:
        -----------
        wavelengths : array
            波長配列（nm）
        peaks : list of dict
            各ピークの情報 {'center': 中心波長, 'amplitude': 振幅, 'width': 幅}
    
        Returns:
        --------
        spectrum : array
            吸光度スペクトル
        """
        spectrum = np.zeros_like(wavelengths, dtype=float)
        for peak in peaks:
            spectrum += gaussian(wavelengths,
                               peak['amplitude'],
                               peak['center'],
                               peak['width'])
        return spectrum
    
    def add_noise(spectrum, noise_level=0.01):
        """スペクトルにガウスノイズを付加"""
        noise = np.random.normal(0, noise_level, len(spectrum))
        return spectrum + noise
    
    # 波長範囲
    wavelengths = np.linspace(350, 750, 500)
    
    # 仮想的な分子のピーク（例：有機色素の吸収）
    peaks = [
        {'center': 420, 'amplitude': 0.8, 'width': 25},  # Soret帯的な吸収
        {'center': 520, 'amplitude': 0.3, 'width': 20},  # Q帯的な吸収
        {'center': 580, 'amplitude': 0.4, 'width': 22},
        {'center': 640, 'amplitude': 0.2, 'width': 18},
    ]
    
    # スペクトル生成
    clean_spectrum = simulate_absorption_spectrum(wavelengths, peaks)
    noisy_spectrum = add_noise(clean_spectrum, noise_level=0.02)
    
    # 透過率への変換
    transmittance = 10**(-noisy_spectrum) * 100
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：吸光度スペクトル
    ax1 = axes[0, 0]
    ax1.plot(wavelengths, noisy_spectrum, 'b-', linewidth=1, alpha=0.7, label='Measured')
    ax1.plot(wavelengths, clean_spectrum, 'r--', linewidth=2, label='True')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.set_title('Absorption Spectrum', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ピーク位置にマーカー
    for peak in peaks:
        ax1.axvline(x=peak['center'], color='gray', linestyle=':', alpha=0.5)
        ax1.text(peak['center'], peak['amplitude'] + 0.05,
                f"{peak['center']} nm", ha='center', fontsize=9)
    
    # 右上：透過率スペクトル
    ax2 = axes[0, 1]
    ax2.plot(wavelengths, transmittance, 'g-', linewidth=1.5)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Transmittance (%)', fontsize=12)
    ax2.set_title('Transmittance Spectrum', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 左下：個別のピーク成分
    ax3 = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, peak in enumerate(peaks):
        single_peak = gaussian(wavelengths, peak['amplitude'], peak['center'], peak['width'])
        ax3.fill_between(wavelengths, single_peak, alpha=0.4, color=colors[i],
                        label=f"Peak at {peak['center']} nm")
        ax3.plot(wavelengths, single_peak, color=colors[i], linewidth=1.5)
    
    ax3.set_xlabel('Wavelength (nm)', fontsize=12)
    ax3.set_ylabel('Absorbance', fontsize=12)
    ax3.set_title('Individual Peak Components', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下：エネルギー表示
    energies = wavelength_to_energy_eV(wavelengths)
    ax4 = axes[1, 1]
    ax4.plot(energies, noisy_spectrum, 'purple', linewidth=1.5)
    ax4.set_xlabel('Photon Energy (eV)', fontsize=12)
    ax4.set_ylabel('Absorbance', fontsize=12)
    ax4.set_title('Absorption Spectrum (Energy Scale)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()  # 低エネルギー側を右に
    
    plt.tight_layout()
    plt.savefig('absorption_spectrum_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ピーク情報のサマリー
    print("=== スペクトルピークのサマリー ===")
    print()
    print(f"{'ピーク':<8} {'波長(nm)':<12} {'エネルギー(eV)':<15} {'吸光度':<12} {'FWHM(nm)':<12}")
    print("-" * 60)
    for i, peak in enumerate(peaks, 1):
        energy = wavelength_to_energy_eV(peak['center'])
        fwhm = 2.355 * peak['width']
        print(f"{i:<8} {peak['center']:<12} {energy:<15.3f} {peak['amplitude']:<12.2f} {fwhm:<12.1f}")
    

#### コード例8：ピークフィッティングの基礎
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    
    def multi_gaussian(x, *params):
        """
        複数のGaussian関数の和
        params: [amp1, center1, sigma1, amp2, center2, sigma2, ...]
        """
        y = np.zeros_like(x, dtype=float)
        n_peaks = len(params) // 3
        for i in range(n_peaks):
            amp = params[3*i]
            center = params[3*i + 1]
            sigma = params[3*i + 2]
            y += amp * np.exp(-(x - center)**2 / (2 * sigma**2))
        return y
    
    def fit_spectrum(wavelengths, spectrum, n_peaks=None, initial_guess=None):
        """
        スペクトルにマルチGaussianフィッティングを行う
    
        Parameters:
        -----------
        wavelengths : array
            波長配列
        spectrum : array
            吸光度スペクトル
        n_peaks : int, optional
            フィッティングするピーク数（Noneの場合は自動検出）
        initial_guess : list, optional
            初期パラメータ
    
        Returns:
        --------
        popt : array
            最適化されたパラメータ
        pcov : array
            共分散行列
        """
        if n_peaks is None:
            # ピークを自動検出
            peak_indices, properties = find_peaks(spectrum, prominence=0.05, width=5)
            n_peaks = len(peak_indices)
            print(f"検出されたピーク数: {n_peaks}")
    
            # 初期推定値の生成
            initial_guess = []
            for idx in peak_indices:
                amp = spectrum[idx]
                center = wavelengths[idx]
                sigma = 15  # 初期幅の推定
                initial_guess.extend([amp, center, sigma])
    
        # フィッティング実行
        bounds_lower = []
        bounds_upper = []
        for i in range(n_peaks):
            bounds_lower.extend([0, wavelengths.min(), 1])  # amp, center, sigma
            bounds_upper.extend([10, wavelengths.max(), 100])
    
        popt, pcov = curve_fit(multi_gaussian, wavelengths, spectrum,
                              p0=initial_guess, bounds=(bounds_lower, bounds_upper),
                              maxfev=10000)
        return popt, pcov, n_peaks
    
    # テストデータの生成
    np.random.seed(42)
    wavelengths = np.linspace(400, 700, 300)
    
    # 真のパラメータ
    true_params = [0.8, 450, 20, 0.5, 550, 25, 0.6, 620, 18]
    true_spectrum = multi_gaussian(wavelengths, *true_params)
    noisy_spectrum = true_spectrum + np.random.normal(0, 0.03, len(wavelengths))
    
    # フィッティング
    popt, pcov, n_peaks = fit_spectrum(wavelengths, noisy_spectrum)
    
    # フィッティング結果
    fitted_spectrum = multi_gaussian(wavelengths, *popt)
    
    # 残差
    residuals = noisy_spectrum - fitted_spectrum
    
    # プロット
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # メインプロット
    ax1 = axes[0]
    ax1.plot(wavelengths, noisy_spectrum, 'k.', markersize=3, alpha=0.5, label='Data')
    ax1.plot(wavelengths, fitted_spectrum, 'r-', linewidth=2, label='Fitted')
    
    # 個別のピーク成分
    colors = ['blue', 'green', 'orange']
    for i in range(n_peaks):
        amp = popt[3*i]
        center = popt[3*i + 1]
        sigma = popt[3*i + 2]
        single_peak = amp * np.exp(-(wavelengths - center)**2 / (2 * sigma**2))
        ax1.fill_between(wavelengths, single_peak, alpha=0.3, color=colors[i % len(colors)],
                        label=f'Peak {i+1}: {center:.1f} nm')
    
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.set_title('Multi-Gaussian Peak Fitting', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 残差プロット
    ax2 = axes[1]
    ax2.plot(wavelengths, residuals, 'g-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(wavelengths, residuals, alpha=0.3, color='green')
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('peak_fitting_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # フィッティング結果のサマリー
    print("\n=== フィッティング結果 ===")
    print()
    print(f"{'ピーク':<8} {'中心(nm)':<12} {'振幅':<12} {'幅(nm)':<12} {'FWHM(nm)':<12}")
    print("-" * 60)
    for i in range(n_peaks):
        amp = popt[3*i]
        center = popt[3*i + 1]
        sigma = popt[3*i + 2]
        fwhm = 2.355 * sigma
        print(f"{i+1:<8} {center:<12.2f} {amp:<12.4f} {sigma:<12.2f} {fwhm:<12.2f}")
    
    # 真値との比較
    print("\n=== 真値との比較 ===")
    print()
    print(f"{'パラメータ':<15} {'真値':<12} {'フィット値':<12} {'誤差(%)':<12}")
    print("-" * 55)
    param_names = ['振幅', '中心', '幅']
    for i in range(n_peaks):
        for j, name in enumerate(param_names):
            true_val = true_params[3*i + j]
            fit_val = popt[3*i + j]
            error = abs(fit_val - true_val) / true_val * 100
            print(f"Peak{i+1} {name:<10} {true_val:<12.2f} {fit_val:<12.2f} {error:<12.2f}")
    
    # フィッティングの良さの評価
    rmse = np.sqrt(np.mean(residuals**2))
    r_squared = 1 - np.sum(residuals**2) / np.sum((noisy_spectrum - np.mean(noisy_spectrum))**2)
    print(f"\nRMSE: {rmse:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    

## 演習問題

**演習問題（クリックして展開）**

### 基礎問題

**問題1** ：以下の波長の光について、振動数（Hz）、波数（cm$^{-1}$）、光子エネルギー（eV）を計算してください。

  * (a) 254 nm（紫外線殺菌灯）
  * (b) 532 nm（Nd:YAGレーザーの第二高調波）
  * (c) 10.6 um（CO2レーザー）

解答を見る
    
    
    # 問題1の解答
    wavelengths_nm = [254, 532, 10600]  # 10.6 um = 10600 nm
    names = ["UV殺菌灯", "Nd:YAGレーザー", "CO2レーザー"]
    
    print("問題1の解答")
    print("=" * 70)
    for wl, name in zip(wavelengths_nm, names):
        freq = wavelength_to_frequency(wl)
        wn = wavelength_to_wavenumber(wl)
        energy = wavelength_to_energy_eV(wl)
        print(f"{name} ({wl} nm):")
        print(f"  振動数: {freq:.3e} Hz ({freq/1e12:.2f} THz)")
        print(f"  波数: {wn:.0f} cm^-1")
        print(f"  エネルギー: {energy:.3f} eV")
        print()
    

**問題2** ：ある有機色素のモル吸光係数が$\varepsilon = 50000$ L mol$^{-1}$ cm$^{-1}$で、1 cmのセルを使って測定したところ吸光度が0.8でした。溶液の濃度（mol/L）と透過率（%）を求めてください。

解答を見る
    
    
    # 問題2の解答
    epsilon = 50000  # L mol^-1 cm^-1
    A = 0.8
    l = 1.0  # cm
    
    # Beer-Lambert則: A = epsilon * c * l
    c = A / (epsilon * l)
    T = 10**(-A) * 100
    
    print("問題2の解答")
    print("=" * 40)
    print(f"濃度: {c:.2e} mol/L = {c*1e6:.2f} uM")
    print(f"透過率: {T:.2f} %")
    

**問題3** ：水素原子のBalmer系列（n=3,4,5,6からn=2への遷移）の発光波長を計算し、可視光領域（400-700 nm）に含まれるものを特定してください。

解答を見る
    
    
    # 問題3の解答
    print("問題3の解答：Balmer系列")
    print("=" * 50)
    
    for n_i in range(3, 7):
        wl, dE = transition_wavelength(n_i, 2)
        in_visible = 400 <= wl <= 700
        visible_str = "可視光" if in_visible else "可視光外"
        print(f"n={n_i} -> n=2: {wl:.1f} nm ({dE:.3f} eV) - {visible_str}")
    

### 応用問題

**問題4** ：調和振動子モデルにおいて、v=0からv=1への遷移（基本振動）は許容ですが、v=0からv=2への遷移（第一倍音）は禁制です。しかし、実際の分子では倍音遷移も観測されます。これはなぜでしょうか？理由を説明してください。

解答を見る

**解答** ：実際の分子のポテンシャルエネルギー曲線は、調和振動子（放物線）からずれており、非調和性を持ちます。非調和ポテンシャルでは、波動関数が調和振動子の波動関数から変形し、$\Delta v = \pm 2, \pm 3, \ldots$ の遷移に対する遷移双極子モーメントがゼロでなくなります。ただし、倍音遷移の強度は基本振動に比べて通常1-2桁弱くなります。

**問題5** ：以下のPythonコードを完成させて、与えられたスペクトルデータからピーク位置（波長）、ピーク高さ（吸光度）、半値全幅（FWHM）を自動的に抽出する関数を作成してください。

解答を見る
    
    
    # 問題5の解答
    from scipy.signal import find_peaks, peak_widths
    
    def analyze_peaks(wavelengths, spectrum, prominence=0.05):
        """
        スペクトルからピーク情報を自動抽出
    
        Parameters:
        -----------
        wavelengths : array
            波長配列
        spectrum : array
            吸光度スペクトル
        prominence : float
            ピーク検出の閾値
    
        Returns:
        --------
        peak_info : list of dict
            各ピークの情報
        """
        # ピーク検出
        peak_indices, properties = find_peaks(spectrum, prominence=prominence)
    
        # 半値幅の計算
        widths, width_heights, left_ips, right_ips = peak_widths(
            spectrum, peak_indices, rel_height=0.5
        )
    
        peak_info = []
        for i, idx in enumerate(peak_indices):
            # 波長間隔を考慮してFWHMを波長単位に変換
            wl_per_point = (wavelengths[-1] - wavelengths[0]) / len(wavelengths)
            fwhm_nm = widths[i] * wl_per_point
    
            info = {
                'peak_wavelength': wavelengths[idx],
                'peak_absorbance': spectrum[idx],
                'fwhm_nm': fwhm_nm,
                'peak_index': idx
            }
            peak_info.append(info)
    
        return peak_info
    
    # テスト
    wavelengths = np.linspace(400, 700, 300)
    test_spectrum = gaussian(wavelengths, 0.8, 480, 20) + gaussian(wavelengths, 0.5, 580, 25)
    test_spectrum += np.random.normal(0, 0.01, len(wavelengths))
    
    peaks = analyze_peaks(wavelengths, test_spectrum)
    
    print("問題5の解答：ピーク解析結果")
    print("=" * 60)
    for i, p in enumerate(peaks, 1):
        print(f"ピーク {i}:")
        print(f"  波長: {p['peak_wavelength']:.1f} nm")
        print(f"  吸光度: {p['peak_absorbance']:.4f}")
        print(f"  FWHM: {p['fwhm_nm']:.1f} nm")
    

**問題6** ：ドップラー広がりとローレンツ広がりの両方が存在する場合、スペクトル線はVoigt関数で表されます。温度300 Kの窒素分子（N$_2$、分子量28）が500 nmの光を放出する場合、ドップラー広がりによる線幅（FWHM）を計算してください。また、衝突広がりのFWHMが0.001 nmの場合、どちらの広がり機構が支配的か判定してください。

解答を見る
    
    
    # 問題6の解答
    import numpy as np
    
    # 定数
    k_B = 1.380649e-23  # ボルツマン定数 (J/K)
    c = 2.99792458e8    # 光速 (m/s)
    u = 1.66054e-27     # 原子質量単位 (kg)
    
    # パラメータ
    T = 300  # 温度 (K)
    M = 28   # 分子量
    lambda_0 = 500e-9  # 中心波長 (m)
    
    # ドップラー広がりの計算
    # FWHM_Doppler = (2 * lambda_0 / c) * sqrt(2 * k_B * T * ln(2) / m)
    m = M * u  # 質量 (kg)
    fwhm_doppler = (2 * lambda_0 / c) * np.sqrt(2 * k_B * T * np.log(2) / m)
    fwhm_doppler_nm = fwhm_doppler * 1e9
    
    # 衝突広がり
    fwhm_collision_nm = 0.001
    
    print("問題6の解答：線幅の計算")
    print("=" * 50)
    print(f"温度: {T} K")
    print(f"分子量: {M}")
    print(f"中心波長: {lambda_0*1e9} nm")
    print()
    print(f"ドップラー広がり FWHM: {fwhm_doppler_nm:.6f} nm")
    print(f"衝突広がり FWHM: {fwhm_collision_nm:.6f} nm")
    print()
    
    if fwhm_doppler_nm > fwhm_collision_nm:
        ratio = fwhm_doppler_nm / fwhm_collision_nm
        print(f"結論: ドップラー広がりが支配的（{ratio:.1f}倍大きい）")
    else:
        ratio = fwhm_collision_nm / fwhm_doppler_nm
        print(f"結論: 衝突広がりが支配的（{ratio:.1f}倍大きい）")
    

## まとめ

この章では、分光法の基礎として以下の内容を学びました：

  * **光と物質の相互作用** ：吸収、発光、散乱の3つの基本過程とBeer-Lambert則
  * **電磁波スペクトル** ：X線から赤外、マイクロ波までの各領域と対応する遷移の種類
  * **エネルギー量子化** ：電子、振動、回転エネルギー準位と量子力学的基礎
  * **選択則と遷移確率** ：遷移双極子モーメントと許容・禁制遷移の決定
  * **スペクトル分解能と装置** ：線形関数（Gaussian、Lorentzian、Voigt）と分光装置の基本構成

次章では、これらの基礎知識を踏まえて、UV-Vis分光法について詳しく学びます。電子遷移の種類、バンドギャップ測定、定量分析など、材料科学で重要な応用を扱います。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言（法律・会計・技術的保証など）を提供するものではありません。
  * 本コンテンツおよび付随するコード例は「現状有姿（AS IS）」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件（例: CC BY 4.0）に従います。当該ライセンスは通常、無保証条項を含みます。
