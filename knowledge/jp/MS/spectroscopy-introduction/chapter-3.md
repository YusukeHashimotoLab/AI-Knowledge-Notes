---
title: "第3章: 紫外可視分光法 (UV-Vis Spectroscopy)"
chapter_title: "第3章: 紫外可視分光法 (UV-Vis Spectroscopy)"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/spectroscopy-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Spectroscopy](<../../MS/spectroscopy-introduction/index.html>)›Chapter 3

# 第3章: 紫外可視分光法 (UV-Vis Spectroscopy)

**この章で学ぶこと:** 紫外可視分光法（UV-Vis Spectroscopy）は、物質の電子遷移を観測する分光法であり、材料科学における光学特性評価、バンドギャップ測定、配位化合物の電子状態解析に不可欠な手法です。本章では、電子遷移の理論的基礎から、Lambert-Beer則の実践的応用、Taucプロット法によるバンドギャップ決定法、配位子場理論による遷移金属錯体の解釈まで、UV-Vis分光法の基礎から実践までを体系的に学びます。

## 3.1 電子遷移とUV-Vis分光法の原理

### 3.1.1 電子遷移のエネルギー

UV-Vis分光法では、紫外線（10-400 nm）から可視光（400-800 nm）の領域における光吸収を測定します。この波長領域は、分子の電子遷移エネルギー（約1.5-6 eV）に対応します。

**電子遷移エネルギーと波長の関係:**

\\[ E = h\nu = \frac{hc}{\lambda} \\] 

ここで、\\( h = 6.626 \times 10^{-34} \\) J·s (Planck定数)、\\( c = 3.0 \times 10^8 \\) m/s (光速)、\\( \lambda \\)は波長です。

波長（nm）とエネルギー（eV）の変換式:

\\[ E\,(\text{eV}) = \frac{1239.8}{\lambda\,(\text{nm})} \\] 

### 3.1.2 電子遷移の種類

#### 主要な電子遷移

  * **σ → σ* 遷移:** 単結合のσ軌道から反結合性σ*軌道への遷移（遠紫外域、λ < 200 nm）
  * **n → σ* 遷移:** 非共有電子対から反結合性σ*軌道への遷移（150-250 nm）
  * **π → π* 遷移:** 二重結合のπ軌道からπ*軌道への遷移（200-400 nm、UV-Visの主要領域）
  * **n → π* 遷移:** 非共有電子対からπ*軌道への遷移（250-350 nm、弱い吸収）
  * **d → d 遷移:** 遷移金属錯体のd軌道間の遷移（可視域、配位子場理論で説明）
  * **電荷移動遷移 (CT):** 金属から配位子（MLCT）、配位子から金属（LMCT）への電荷移動（強い吸収）

### 3.1.3 HOMO-LUMO遷移とバンドギャップ

有機分子や半導体材料において、最も低エネルギーの電子遷移は最高被占軌道（HOMO）から最低空軌道（LUMO）への遷移です。この遷移エネルギーは、半導体のバンドギャップ \\( E_g \\) に相当します。

**バンドギャップとUV-Vis吸収端の関係:**

\\[ E_g = h\nu_{\text{onset}} = \frac{1239.8}{\lambda_{\text{onset}}\,(\text{nm})} \\] 

ここで、\\( \lambda_{\text{onset}} \\)は吸収スペクトルの立ち上がり波長（absorption onset）です。
    
    
    ```mermaid
    flowchart TD
            A[基底状態HOMO電子配置] -->|光吸収 hν| B[励起状態LUMO電子配置]
            B -->|蛍光発光| C[基底状態エネルギー放出]
            B -->|無放射失活| D[基底状態熱エネルギー]
    
            style A fill:#e3f2fd
            style B fill:#fff3e0
            style C fill:#e8f5e9
            style D fill:#fce4ec
        
    3.2 Lambert-Beer則の理論と応用
    3.2.1 Lambert-Beer則の数学的表現
    Lambert-Beer則は、溶液の光吸収と濃度の関係を記述する基本法則です。第1章で導入した式を、UV-Vis分光法の文脈で再考します。
    
    吸光度の定義:
            \[
            A = \log_{10}\left(\frac{I_0}{I}\right) = \epsilon c l
            \]
            ここで、\( A \)は吸光度、\( I_0 \)は入射光強度、\( I \)は透過光強度、\( \epsilon \)はモル吸光係数（L mol-1 cm-1）、\( c \)は濃度（mol/L）、\( l \)は光路長（cm）です。
    透過率との関係:
            \[
            T = \frac{I}{I_0} = 10^{-A}
            \]
            \[
            A = -\log_{10} T = 2 - \log_{10}(\%T)
            \]
        
    3.2.2 モル吸光係数の物理的意味
    モル吸光係数 \( \epsilon \) は、特定波長における物質の光吸収能力を表す固有の物性値です。大きな \( \epsilon \) 値（\( \epsilon > 10^4 \) L mol-1 cm-1）は許容遷移（allowed transition）を示し、小さな \( \epsilon \) 値（\( \epsilon < 10^3 \)）は禁制遷移（forbidden transition）を示します。
    
    
    
    遷移タイプ
    モル吸光係数 ε (L mol-1 cm-1)
    例
    
    
    
    
    π → π* (共役系)
    10,000 - 100,000
    ベンゼン、アントラセン
    
    
    n → π*
    10 - 1,000
    カルボニル化合物
    
    
    d → d (遷移金属)
    1 - 100
    Cu2+, Ni2+錯体
    
    
    電荷移動 (CT)
    1,000 - 50,000
    MnO4-, Fe-フェナントロリン
    
    
    
    3.2.3 検量線法による定量分析
    Lambert-Beer則の線形性を利用して、未知試料の濃度を決定できます。既知濃度の標準溶液系列を測定し、吸光度 \( A \) vs. 濃度 \( c \) の検量線（calibration curve）を作成します。
    コード例1: Lambert-Beer則による検量線の作成と定量分析
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    def create_calibration_curve(concentrations, absorbances):
        """
        検量線を作成し、線形回帰パラメータを返す
    
        Parameters:
        -----------
        concentrations : array-like
            標準溶液の濃度 (mol/L)
        absorbances : array-like
            各濃度における吸光度
    
        Returns:
        --------
        slope : float
            検量線の傾き（モル吸光係数 × 光路長）
        intercept : float
            切片（ゼロであるべき）
        r_value : float
            相関係数
        """
        # 線形回帰
        slope, intercept, r_value, p_value, std_err = linregress(concentrations, absorbances)
    
        # プロット
        plt.figure(figsize=(10, 6))
        plt.scatter(concentrations, absorbances, s=100, alpha=0.7, label='実測データ')
    
        # 回帰直線
        conc_fit = np.linspace(0, max(concentrations)*1.1, 100)
        abs_fit = slope * conc_fit + intercept
        plt.plot(conc_fit, abs_fit, 'r--', linewidth=2,
                 label=f'回帰直線: A = {slope:.3f}c + {intercept:.4f}\nR² = {r_value**2:.4f}')
    
        plt.xlabel('濃度 (mol/L)', fontsize=12)
        plt.ylabel('吸光度', fontsize=12)
        plt.title('Lambert-Beer則による検量線', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        return slope, intercept, r_value
    
    def determine_concentration(absorbance_sample, slope, intercept):
        """
        検量線から未知試料の濃度を決定
    
        Parameters:
        -----------
        absorbance_sample : float
            未知試料の吸光度
        slope : float
            検量線の傾き
        intercept : float
            検量線の切片
    
        Returns:
        --------
        concentration : float
            未知試料の濃度 (mol/L)
        """
        concentration = (absorbance_sample - intercept) / slope
        return concentration
    
    # 実行例：メチレンブルーの定量分析
    concentrations = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-5  # mol/L
    absorbances = np.array([0.12, 0.24, 0.48, 0.72, 0.96, 1.20])
    
    slope, intercept, r_value = create_calibration_curve(concentrations, absorbances)
    
    # 光路長 1 cm の場合、モル吸光係数を計算
    epsilon = slope  # L mol^-1 cm^-1
    print(f"モル吸光係数 ε = {epsilon:.2e} L mol⁻¹ cm⁻¹")
    
    # 未知試料の濃度決定
    A_unknown = 0.60
    c_unknown = determine_concentration(A_unknown, slope, intercept)
    print(f"未知試料の濃度: {c_unknown:.2e} mol/L")
    print(f"検量線の相関係数 R² = {r_value**2:.4f}")
    
    3.3 Taucプロット法によるバンドギャップ測定
    3.3.1 Tauc則の理論的背景
    半導体や絶縁体材料のバンドギャップを精密に決定するため、Jan Tauc（1968）が提案した解析法が広く用いられます。Tauc則は、吸収係数 \( \alpha \) と光子エネルギー \( h\nu \) の関係を記述します。
    
    Tauc則（直接遷移）:
            \[
            (\alpha h\nu)^2 = B(h\nu - E_g)
            \]
            ここで、\( B \)は材料定数、\( E_g \)はバンドギャップです。
    Tauc則（間接遷移）:
            \[
            (\alpha h\nu)^{1/2} = B(h\nu - E_g)
            \]
    
            吸収係数の計算:
            \[
            \alpha = \frac{2.303 \cdot A}{l}
            \]
            ここで、\( A \)は吸光度、\( l \)は試料厚さ（cm）です。
    
    3.3.2 Taucプロットの作成手順
    
    UV-Vis吸収スペクトルを測定し、波長 \( \lambda \) と吸光度 \( A \) を取得
    波長を光子エネルギー \( h\nu = 1239.8/\lambda \) (eV) に変換
    吸光度から吸収係数 \( \alpha = 2.303 \cdot A/l \) を計算
    直接遷移の場合: \( (\alpha h\nu)^2 \) vs. \( h\nu \) をプロット
    間接遷移の場合: \( (\alpha h\nu)^{1/2} \) vs. \( h\nu \) をプロット
    吸収端の線形領域を外挿し、横軸との交点からバンドギャップ \( E_g \) を決定
    
    コード例2: Taucプロット法によるバンドギャップ測定
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def tauc_plot_direct(wavelength, absorbance, thickness_cm, plot_range=(2.0, 4.0)):
        """
        直接遷移材料のTaucプロットを作成し、バンドギャップを決定
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        absorbance : array
            吸光度
        thickness_cm : float
            試料厚さ (cm)
        plot_range : tuple
            フィッティングに使用するエネルギー範囲 (eV)
    
        Returns:
        --------
        Eg : float
            バンドギャップ (eV)
        """
        # 波長を光子エネルギーに変換
        photon_energy = 1239.8 / wavelength  # eV
    
        # 吸収係数を計算
        alpha = 2.303 * absorbance / thickness_cm  # cm^-1
    
        # Taucプロット: (αhν)^2
        tauc_y = (alpha * photon_energy)**2
    
        # フィッティング範囲の選択
        mask = (photon_energy >= plot_range[0]) & (photon_energy <= plot_range[1])
        E_fit = photon_energy[mask]
        tauc_fit = tauc_y[mask]
    
        # 線形フィッティング
        def linear(x, B, Eg):
            return B * (x - Eg)
    
        popt, pcov = curve_fit(linear, E_fit, tauc_fit, p0=[1e10, 3.0])
        B, Eg = popt
    
        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(photon_energy, tauc_y, 'o-', label='実測データ', alpha=0.7)
    
        # フィッティング直線
        E_extended = np.linspace(Eg - 0.5, plot_range[1], 100)
        tauc_extended = linear(E_extended, B, Eg)
        plt.plot(E_extended, tauc_extended, 'r--', linewidth=2,
                 label=f'線形フィット\nEg = {Eg:.3f} eV')
    
        # バンドギャップ位置を強調
        plt.axvline(Eg, color='green', linestyle=':', linewidth=2, label=f'Bandgap: {Eg:.3f} eV')
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
        plt.xlabel('光子エネルギー (eV)', fontsize=12)
        plt.ylabel('(αhν)² (eV² cm⁻²)', fontsize=12)
        plt.title('Taucプロット（直接遷移）', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(photon_energy.min(), photon_energy.max())
        plt.tight_layout()
        plt.show()
    
        print(f"決定されたバンドギャップ: Eg = {Eg:.3f} eV")
        print(f"対応する波長: λ = {1239.8/Eg:.1f} nm")
    
        return Eg
    
    def tauc_plot_indirect(wavelength, absorbance, thickness_cm, plot_range=(1.5, 3.0)):
        """
        間接遷移材料のTaucプロットを作成し、バンドギャップを決定
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        absorbance : array
            吸光度
        thickness_cm : float
            試料厚さ (cm)
        plot_range : tuple
            フィッティングに使用するエネルギー範囲 (eV)
    
        Returns:
        --------
        Eg : float
            バンドギャップ (eV)
        """
        # 波長を光子エネルギーに変換
        photon_energy = 1239.8 / wavelength  # eV
    
        # 吸収係数を計算
        alpha = 2.303 * absorbance / thickness_cm  # cm^-1
    
        # Taucプロット: (αhν)^(1/2)
        tauc_y = np.sqrt(alpha * photon_energy)
    
        # フィッティング範囲の選択
        mask = (photon_energy >= plot_range[0]) & (photon_energy <= plot_range[1])
        E_fit = photon_energy[mask]
        tauc_fit = tauc_y[mask]
    
        # 線形フィッティング
        def linear(x, B, Eg):
            return B * (x - Eg)
    
        popt, pcov = curve_fit(linear, E_fit, tauc_fit, p0=[100, 2.0])
        B, Eg = popt
    
        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(photon_energy, tauc_y, 'o-', label='実測データ', alpha=0.7)
    
        # フィッティング直線
        E_extended = np.linspace(Eg - 0.3, plot_range[1], 100)
        tauc_extended = linear(E_extended, B, Eg)
        plt.plot(E_extended, tauc_extended, 'r--', linewidth=2,
                 label=f'線形フィット\nEg = {Eg:.3f} eV')
    
        # バンドギャップ位置を強調
        plt.axvline(Eg, color='green', linestyle=':', linewidth=2, label=f'Bandgap: {Eg:.3f} eV')
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
        plt.xlabel('光子エネルギー (eV)', fontsize=12)
        plt.ylabel('(αhν)^(1/2) (eV^(1/2) cm^(-1/2))', fontsize=12)
        plt.title('Taucプロット（間接遷移）', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(photon_energy.min(), photon_energy.max())
        plt.tight_layout()
        plt.show()
    
        print(f"決定されたバンドギャップ: Eg = {Eg:.3f} eV")
        print(f"対応する波長: λ = {1239.8/Eg:.1f} nm")
    
        return Eg
    
    # 実行例：TiO₂ナノ粒子（直接遷移、Eg ≈ 3.2 eV）
    wavelength_nm = np.linspace(300, 500, 200)
    # シミュレーションデータ（実際には分光光度計から取得）
    Eg_true = 3.2  # eV
    alpha_true = 1e4 * np.maximum(0, (1239.8/wavelength_nm - Eg_true))**2
    absorbance_sim = alpha_true * 0.01 / 2.303  # 試料厚さ 0.01 cm
    
    Eg_measured = tauc_plot_direct(wavelength_nm, absorbance_sim, thickness_cm=0.01)
    
    3.4 配位子場理論とd-d遷移
    3.4.1 配位子場分裂エネルギー
    遷移金属錯体（Cu2+, Ni2+, Co2+など）は、可視域に特徴的な色を示します。これは、配位子による静電場の影響でd軌道が分裂し、d-d遷移が可視光の吸収を引き起こすためです。
    
    八面体配位子場（Oh対称性）のd軌道分裂
    
    eg軌道（高エネルギー）: dz², dx²-y² （配位子と直接対向、反発大）
    t2g軌道（低エネルギー）: dxy, dxz, dyz （配位子との反発小）
    
    分裂エネルギー \( \Delta_o \)（10Dq）は、錯体の色と直接関係します：
            \[
            \Delta_o = h\nu = \frac{hc}{\lambda}
            \]
        
    
        flowchart TB
            A[自由イオン5つのd軌道縮退] -->|八面体配位子場| B[e_g軌道高エネルギー]
            A -->|八面体配位子場| C[t_2g軌道低エネルギー]
    
            B -.->|d-d遷移光吸収| C
    
            D[d電子配置基底状態] -->|可視光吸収| E[d電子配置励起状態]
    
            style A fill:#e3f2fd
            style B fill:#ffebee
            style C fill:#e8f5e9
            style D fill:#fff3e0
            style E fill:#fce4ec
        
    3.4.2 分光化学系列
    配位子の種類によって、分裂エネルギー \( \Delta_o \) の大きさが変化します。これを分光化学系列（spectrochemical series）と呼びます：
    
    分光化学系列（配位子の強度順）:
            \[
            \text{I}^- < \text{Br}^- < \text{Cl}^- < \text{F}^- < \text{OH}^- < \text{H}_2\text{O} < \text{NH}_3 < \text{en} < \text{NO}_2^- < \text{CN}^- < \text{CO}
            \]
            弱場配位子（左側）→ 小さい \( \Delta_o \)、長波長吸収（赤色・黄色）
    強場配位子（右側）→ 大きい \( \Delta_o \)、短波長吸収（青色・紫色）
    
    コード例3: 配位子場理論による遷移金属錯体の色予測
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    def predict_complex_color(delta_o_cm, d_electron_count, geometry='octahedral'):
        """
        配位子場分裂エネルギーからd-d遷移波長と錯体の色を予測
    
        Parameters:
        -----------
        delta_o_cm : float
            配位子場分裂エネルギー (cm^-1)
        d_electron_count : int
            d電子数（1-10）
        geometry : str
            配位構造（'octahedral' or 'tetrahedral'）
    
        Returns:
        --------
        wavelength_nm : float
            d-d遷移の波長 (nm)
        observed_color : str
            観測される錯体の色（補色）
        """
        # 波長に変換
        wavelength_nm = 1e7 / delta_o_cm  # nm
    
        # 吸収される光の色
        if wavelength_nm < 450:
            absorbed_color = '紫'
            observed_color = '黄緑'
        elif wavelength_nm < 495:
            absorbed_color = '青'
            observed_color = '黄色'
        elif wavelength_nm < 570:
            absorbed_color = '緑'
            observed_color = '赤紫'
        elif wavelength_nm < 590:
            absorbed_color = '黄'
            observed_color = '青紫'
        elif wavelength_nm < 620:
            absorbed_color = '橙'
            observed_color = '青'
        elif wavelength_nm < 750:
            absorbed_color = '赤'
            observed_color = '緑'
        else:
            absorbed_color = '赤外'
            observed_color = '無色（赤外吸収）'
    
        print(f"配位子場分裂エネルギー Δo = {delta_o_cm:.0f} cm⁻¹")
        print(f"d-d遷移波長: λ = {wavelength_nm:.1f} nm")
        print(f"吸収される光: {absorbed_color}（{wavelength_nm:.1f} nm）")
        print(f"観測される錯体の色: {observed_color}（補色）")
        print(f"d電子数: d^{d_electron_count}（{geometry}配位）")
    
        return wavelength_nm, observed_color
    
    def plot_spectrochemical_series():
        """
        代表的な遷移金属錯体の分光化学系列を可視化
        """
        # 代表的な錯体のΔoデータ（cm^-1）
        complexes = [
            '[Ti(H2O)6]3+',
            '[V(H2O)6]3+',
            '[Cr(H2O)6]3+',
            '[Mn(H2O)6]2+',
            '[Fe(H2O)6]2+',
            '[Co(H2O)6]2+',
            '[Ni(H2O)6]2+',
            '[Cu(H2O)6]2+',
            '[Co(NH3)6]3+',
            '[Cr(CN)6]3-'
        ]
    
        delta_o_values = np.array([20300, 18900, 17400, 21000, 10400, 9300, 8500, 12600, 22900, 26600])
        wavelengths = 1e7 / delta_o_values  # nm
    
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # 各錯体の吸収波長を色付きバーで表示
        colors_map = {
            (380, 450): ('#8B00FF', '黄緑'),
            (450, 495): ('#0000FF', '黄色'),
            (495, 570): ('#00FF00', '赤紫'),
            (570, 590): ('#FFFF00', '青紫'),
            (590, 620): ('#FFA500', '青'),
            (620, 750): ('#FF0000', '緑')
        }
    
        for i, (name, wl) in enumerate(zip(complexes, wavelengths)):
            # 吸収波長に対応する色
            color = '#808080'  # デフォルトグレー
            observed = '不明'
            for (wl_min, wl_max), (c, obs) in colors_map.items():
                if wl_min <= wl < wl_max:
                    color = c
                    observed = obs
                    break
    
            ax.barh(i, wl, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.text(wl + 20, i, f'{wl:.1f} nm\n観測色: {observed}',
                    va='center', fontsize=9, fontweight='bold')
    
        ax.set_yticks(range(len(complexes)))
        ax.set_yticklabels(complexes, fontsize=11)
        ax.set_xlabel('d-d遷移波長 (nm)', fontsize=12)
        ax.set_title('遷移金属錯体の分光化学系列とd-d遷移波長', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 800)
        ax.grid(axis='x', alpha=0.3)
    
        # 可視光領域を強調
        ax.axvspan(380, 750, alpha=0.1, color='yellow', label='可視光領域')
        ax.legend()
    
        plt.tight_layout()
        plt.show()
    
    # 実行例1: [Cu(H2O)6]2+の色予測（d^9電子配置）
    delta_o_cu = 12600  # cm^-1
    wavelength, color = predict_complex_color(delta_o_cu, d_electron_count=9)
    
    print("\n" + "="*50)
    # 実行例2: [Cr(NH3)6]3+の色予測（d^3電子配置、強場配位子）
    delta_o_cr = 21500  # cm^-1
    wavelength2, color2 = predict_complex_color(delta_o_cr, d_electron_count=3)
    
    # 分光化学系列のプロット
    plot_spectrochemical_series()
    
    3.5 電荷移動遷移（Charge Transfer Transition）
    3.5.1 LMCT遷移とMLCT遷移
    電荷移動遷移は、d-d遷移よりも大きなモル吸光係数（\( \epsilon > 10^4 \)）を持ち、強い色を示します。
    
    電荷移動遷移の分類
    
    LMCT（Ligand-to-Metal Charge Transfer）: 配位子の電子が金属イオンへ移動する遷移。例：MnO4-（紫色）、CrO42-（黄色）
    MLCT（Metal-to-Ligand Charge Transfer）: 金属イオンの電子が配位子へ移動する遷移。例：Fe(II)-フェナントロリン錯体（赤色）、Ru(bpy)32+（橙色）
    
    
    コード例4: 過マンガン酸イオンのLMCT遷移解析
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_lmct_spectrum(wavelength, lambda_max, epsilon_max, bandwidth):
        """
        LMCT遷移のUV-Visスペクトルをガウス関数でシミュレート
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        lambda_max : float
            吸収極大波長 (nm)
        epsilon_max : float
            最大モル吸光係数 (L mol^-1 cm^-1)
        bandwidth : float
            吸収バンドの幅（半値全幅, nm）
    
        Returns:
        --------
        epsilon : array
            各波長におけるモル吸光係数
        """
        sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
        epsilon = epsilon_max * np.exp(-((wavelength - lambda_max)**2) / (2 * sigma**2))
        return epsilon
    
    # MnO4^- のLMCT遷移スペクトルをシミュレート
    wavelength = np.linspace(400, 700, 300)
    
    # MnO4^- は526 nm（緑色）に強い吸収 → 紫色に見える
    epsilon_mno4 = simulate_lmct_spectrum(wavelength, lambda_max=526, epsilon_max=2300, bandwidth=80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # スペクトルプロット
    ax1.plot(wavelength, epsilon_mno4, linewidth=2, color='purple', label='MnO₄⁻ LMCT遷移')
    ax1.axvline(526, color='green', linestyle='--', linewidth=1.5, label='吸収極大 (526 nm)')
    ax1.fill_between(wavelength, epsilon_mno4, alpha=0.3, color='purple')
    ax1.set_xlabel('波長 (nm)', fontsize=12)
    ax1.set_ylabel('モル吸光係数 ε (L mol⁻¹ cm⁻¹)', fontsize=12)
    ax1.set_title('MnO₄⁻ の LMCT遷移スペクトル', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 可視光スペクトルと観測色
    visible_colors = [
        (380, 450, '#8B00FF'),
        (450, 495, '#0000FF'),
        (495, 570, '#00FF00'),
        (570, 590, '#FFFF00'),
        (590, 620, '#FFA500'),
        (620, 750, '#FF0000')
    ]
    
    for wl_min, wl_max, color in visible_colors:
        ax2.axvspan(wl_min, wl_max, color=color, alpha=0.7)
    
    ax2.axvline(526, color='black', linestyle='--', linewidth=2, label='MnO₄⁻ 吸収 (526 nm)')
    ax2.set_xlabel('波長 (nm)', fontsize=12)
    ax2.set_title('可視光スペクトルとMnO₄⁻の吸収', fontsize=14, fontweight='bold')
    ax2.set_xlim(380, 750)
    ax2.set_yticks([])
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("MnO₄⁻ イオン:")
    print("- 吸収極大: 526 nm（緑色）")
    print("- 観測色: 紫色（緑の補色）")
    print("- 遷移タイプ: LMCT（配位子O²⁻ → Mn⁷⁺）")
    print("- モル吸光係数: ε ≈ 2300 L mol⁻¹ cm⁻¹（強い吸収）")
    
    3.6 溶媒効果と溶媒シフト
    3.6.1 溶媒極性による吸収波長変化
    溶媒の極性は、溶質分子の電子状態を安定化または不安定化し、吸収波長をシフトさせます。
    
    溶媒シフトの分類
    
    Red shift（bathochromic shift）: 長波長側へのシフト。極性溶媒が励起状態をより安定化する場合。
    Blue shift（hypsochromic shift）: 短波長側へのシフト。極性溶媒が基底状態をより安定化する場合。
    
    
    コード例5: 溶媒極性による吸収スペクトルシフトのシミュレーション
    import numpy as np
    import matplotlib.pyplot as plt
    
    def simulate_solvent_shift(wavelength, lambda_max_nonpolar, shift_per_polarity_unit):
        """
        溶媒極性による吸収スペクトルのシフトをシミュレート
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        lambda_max_nonpolar : float
            非極性溶媒中の吸収極大波長 (nm)
        shift_per_polarity_unit : float
            溶媒極性1単位あたりのシフト量 (nm)
    
        Returns:
        --------
        spectra : dict
            各溶媒におけるスペクトル
        """
        solvents = {
            'ヘキサン': 0.0,
            'エタノール': 5.2,
            'メタノール': 6.6,
            '水': 9.0,
            'DMSO': 7.2
        }
    
        fig, ax = plt.subplots(figsize=(12, 7))
    
        colors = ['blue', 'green', 'orange', 'red', 'purple']
    
        for (solvent, polarity), color in zip(solvents.items(), colors):
            lambda_max = lambda_max_nonpolar + shift_per_polarity_unit * polarity
    
            # ガウス型吸収バンド
            sigma = 30
            absorbance = np.exp(-((wavelength - lambda_max)**2) / (2 * sigma**2))
    
            ax.plot(wavelength, absorbance, linewidth=2, label=f'{solvent} (λmax = {lambda_max:.1f} nm)', color=color)
            ax.axvline(lambda_max, linestyle='--', linewidth=1, color=color, alpha=0.5)
    
        ax.set_xlabel('波長 (nm)', fontsize=12)
        ax.set_ylabel('規格化吸光度', fontsize=12)
        ax.set_title('溶媒極性による吸収スペクトルのRed Shift', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # π → π* 遷移分子のsolvatochromism（溶媒発色）
    wavelength = np.linspace(300, 500, 200)
    simulate_solvent_shift(wavelength, lambda_max_nonpolar=350, shift_per_polarity_unit=3.0)
    
    print("溶媒極性による吸収波長シフト:")
    print("- 極性溶媒 → 励起状態安定化 → Red shift（長波長シフト）")
    print("- 非極性溶媒 → シフトなし")
    print("- π → π* 遷移（極性大）はRed shiftしやすい")
    print("- n → π* 遷移（極性小）はBlue shiftすることもある")
    
    3.7 ベースライン補正とスペクトル前処理
    3.7.1 散乱光補正
    固体試料や懸濁液の測定では、散乱光がベースラインを歪めます。適切なベースライン補正が必要です。
    コード例6: ベースライン補正と散乱光除去
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from scipy.interpolate import UnivariateSpline
    
    def baseline_correction_polynomial(wavelength, absorbance, baseline_region, poly_order=2):
        """
        多項式フィッティングによるベースライン補正
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        absorbance : array
            生の吸光度データ
        baseline_region : tuple
            ベースライン領域の波長範囲 (nm)
        poly_order : int
            多項式の次数
    
        Returns:
        --------
        corrected_absorbance : array
            補正後の吸光度
        """
        # ベースライン領域のデータを抽出
        mask = (wavelength >= baseline_region[0]) & (wavelength <= baseline_region[1])
        wl_base = wavelength[mask]
        abs_base = absorbance[mask]
    
        # 多項式フィッティング
        poly_coef = np.polyfit(wl_base, abs_base, poly_order)
        baseline = np.polyval(poly_coef, wavelength)
    
        # ベースライン減算
        corrected_absorbance = absorbance - baseline
    
        return corrected_absorbance, baseline
    
    def baseline_correction_spline(wavelength, absorbance, baseline_points):
        """
        スプライン補間によるベースライン補正
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        absorbance : array
            生の吸光度データ
        baseline_points : list of tuples
            ベースラインポイント [(wl1, abs1), (wl2, abs2), ...]
    
        Returns:
        --------
        corrected_absorbance : array
            補正後の吸光度
        """
        wl_base = np.array([p[0] for p in baseline_points])
        abs_base = np.array([p[1] for p in baseline_points])
    
        # スプライン補間
        spline = UnivariateSpline(wl_base, abs_base, s=0, k=3)
        baseline = spline(wavelength)
    
        # ベースライン減算
        corrected_absorbance = absorbance - baseline
    
        return corrected_absorbance, baseline
    
    # シミュレーションデータ：散乱光を含むスペクトル
    wavelength = np.linspace(300, 700, 400)
    
    # 真の吸収スペクトル（ガウス型ピーク）
    true_abs = 0.8 * np.exp(-((wavelength - 450)**2) / (2 * 50**2))
    
    # 散乱光によるベースライン（波長の逆べき乗に比例）
    scattering_baseline = 0.3 * (wavelength / 300)**(-4)
    
    # ノイズ
    noise = np.random.normal(0, 0.01, len(wavelength))
    
    # 観測スペクトル
    observed_abs = true_abs + scattering_baseline + noise
    
    # ベースライン補正（多項式）
    corrected_abs_poly, baseline_poly = baseline_correction_polynomial(
        wavelength, observed_abs, baseline_region=(600, 700), poly_order=3
    )
    
    # ベースライン補正（スプライン）
    baseline_points = [(300, observed_abs[0]), (380, observed_abs[80]),
                       (600, observed_abs[300]), (700, observed_abs[-1])]
    corrected_abs_spline, baseline_spline = baseline_correction_spline(
        wavelength, observed_abs, baseline_points
    )
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元のスペクトル
    axes[0, 0].plot(wavelength, observed_abs, label='観測スペクトル（散乱光含む）', color='blue')
    axes[0, 0].plot(wavelength, true_abs, '--', label='真のスペクトル', color='red', linewidth=2)
    axes[0, 0].plot(wavelength, scattering_baseline, ':', label='散乱ベースライン', color='green', linewidth=2)
    axes[0, 0].set_xlabel('波長 (nm)')
    axes[0, 0].set_ylabel('吸光度')
    axes[0, 0].set_title('散乱光を含む観測スペクトル')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 多項式ベースライン補正
    axes[0, 1].plot(wavelength, observed_abs, label='観測スペクトル', color='blue', alpha=0.5)
    axes[0, 1].plot(wavelength, baseline_poly, '--', label='多項式ベースライン', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('波長 (nm)')
    axes[0, 1].set_ylabel('吸光度')
    axes[0, 1].set_title('多項式フィッティング')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # スプラインベースライン補正
    axes[1, 0].plot(wavelength, observed_abs, label='観測スペクトル', color='blue', alpha=0.5)
    axes[1, 0].plot(wavelength, baseline_spline, '--', label='スプラインベースライン', color='purple', linewidth=2)
    for wl, abs_val in baseline_points:
        axes[1, 0].plot(wl, abs_val, 'ro', markersize=8)
    axes[1, 0].set_xlabel('波長 (nm)')
    axes[1, 0].set_ylabel('吸光度')
    axes[1, 0].set_title('スプライン補間')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 補正結果比較
    axes[1, 1].plot(wavelength, true_abs, '--', label='真のスペクトル', color='red', linewidth=2)
    axes[1, 1].plot(wavelength, corrected_abs_poly, label='多項式補正', color='orange', alpha=0.7)
    axes[1, 1].plot(wavelength, corrected_abs_spline, label='スプライン補正', color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('波長 (nm)')
    axes[1, 1].set_ylabel('吸光度')
    axes[1, 1].set_title('ベースライン補正結果')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("ベースライン補正の評価:")
    poly_error = np.mean((corrected_abs_poly - true_abs)**2)
    spline_error = np.mean((corrected_abs_spline - true_abs)**2)
    print(f"多項式補正の平均二乗誤差: {poly_error:.6f}")
    print(f"スプライン補正の平均二乗誤差: {spline_error:.6f}")
    
    3.8 多波長解析と多成分定量
    3.8.1 吸収加算性の原理
    複数の吸収種が共存する混合溶液では、Lambert-Beer則の加算性により、各成分の寄与を分離できます。
    
    多成分系のLambert-Beer則:
            \[
            A(\lambda) = \sum_{i=1}^{n} \epsilon_i(\lambda) \cdot c_i \cdot l
            \]
            行列表記（\( m \)波長、\( n \)成分）:
            \[
            \mathbf{A} = \mathbf{E} \mathbf{c} l
            \]
            ここで、\( \mathbf{A} \)は吸光度ベクトル（\( m \times 1 \)）、\( \mathbf{E} \)はモル吸光係数行列（\( m \times n \)）、\( \mathbf{c} \)は濃度ベクトル（\( n \times 1 \)）です。
    
    コード例7: 多波長解析による2成分混合物の定量
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import nnls  # Non-Negative Least Squares
    
    def multiwavelength_analysis(wavelength, absorbance_mixture, epsilon_matrix, path_length=1.0):
        """
        多波長解析により混合溶液中の各成分濃度を決定
    
        Parameters:
        -----------
        wavelength : array
            測定波長 (nm)
        absorbance_mixture : array
            混合溶液の吸光度スペクトル
        epsilon_matrix : 2D array
            各成分のモル吸光係数スペクトル (shape: n_wavelengths × n_components)
        path_length : float
            光路長 (cm)
    
        Returns:
        --------
        concentrations : array
            決定された各成分の濃度 (mol/L)
        """
        # 非負最小二乗法で濃度を決定（負の濃度を禁止）
        concentrations, residual = nnls(epsilon_matrix * path_length, absorbance_mixture)
    
        # 再構成スペクトル
        absorbance_reconstructed = epsilon_matrix @ concentrations * path_length
    
        return concentrations, absorbance_reconstructed, residual
    
    # シミュレーション：メチレンブルー（MB）とメチルオレンジ（MO）の混合溶液
    wavelength = np.linspace(400, 700, 300)
    
    # 成分1: メチレンブルー（λmax = 664 nm）
    epsilon_MB = 8e4 * np.exp(-((wavelength - 664)**2) / (2 * 40**2))
    
    # 成分2: メチルオレンジ（λmax = 464 nm）
    epsilon_MO = 2.7e4 * np.exp(-((wavelength - 464)**2) / (2 * 35**2))
    
    # モル吸光係数行列
    epsilon_matrix = np.column_stack([epsilon_MB, epsilon_MO])
    
    # 真の濃度（mol/L）
    c_MB_true = 1.5e-5
    c_MO_true = 3.0e-5
    
    # 混合溶液の吸光度（光路長 1 cm）
    absorbance_mixture = epsilon_MB * c_MB_true + epsilon_MO * c_MO_true
    absorbance_mixture += np.random.normal(0, 0.005, len(wavelength))  # ノイズ
    
    # 多波長解析
    concentrations, absorbance_recon, residual = multiwavelength_analysis(
        wavelength, absorbance_mixture, epsilon_matrix, path_length=1.0
    )
    
    c_MB_calc, c_MO_calc = concentrations
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 各成分のモル吸光係数スペクトル
    axes[0, 0].plot(wavelength, epsilon_MB, label='メチレンブルー（MB）', color='blue', linewidth=2)
    axes[0, 0].plot(wavelength, epsilon_MO, label='メチルオレンジ（MO）', color='orange', linewidth=2)
    axes[0, 0].set_xlabel('波長 (nm)')
    axes[0, 0].set_ylabel('モル吸光係数 (L mol⁻¹ cm⁻¹)')
    axes[0, 0].set_title('各成分のモル吸光係数スペクトル')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 混合溶液のスペクトル
    axes[0, 1].plot(wavelength, absorbance_mixture, 'o-', label='観測スペクトル',
                    color='purple', alpha=0.6, markersize=2)
    axes[0, 1].plot(wavelength, absorbance_recon, '--', label='再構成スペクトル',
                    color='red', linewidth=2)
    axes[0, 1].set_xlabel('波長 (nm)')
    axes[0, 1].set_ylabel('吸光度')
    axes[0, 1].set_title('混合溶液のスペクトルとフィッティング')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 各成分の寄与
    abs_MB_contrib = epsilon_MB * c_MB_calc
    abs_MO_contrib = epsilon_MO * c_MO_calc
    axes[1, 0].plot(wavelength, absorbance_mixture, label='総吸光度', color='black', linewidth=2)
    axes[1, 0].fill_between(wavelength, 0, abs_MB_contrib, alpha=0.5, color='blue', label='MB寄与')
    axes[1, 0].fill_between(wavelength, abs_MB_contrib, abs_MB_contrib + abs_MO_contrib,
                            alpha=0.5, color='orange', label='MO寄与')
    axes[1, 0].set_xlabel('波長 (nm)')
    axes[1, 0].set_ylabel('吸光度')
    axes[1, 0].set_title('各成分の吸光度寄与')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 濃度決定結果
    components = ['メチレンブルー', 'メチルオレンジ']
    concentrations_true = [c_MB_true, c_MO_true]
    concentrations_calc = [c_MB_calc, c_MO_calc]
    
    x = np.arange(len(components))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, np.array(concentrations_true)*1e5, width, label='真の濃度', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, np.array(concentrations_calc)*1e5, width, label='計算濃度', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('成分')
    axes[1, 1].set_ylabel('濃度 (×10⁻⁵ mol/L)')
    axes[1, 1].set_title('濃度決定結果')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("多波長解析結果:")
    print(f"メチレンブルー: 真の濃度 = {c_MB_true:.2e} mol/L, 計算濃度 = {c_MB_calc:.2e} mol/L")
    print(f"メチルオレンジ: 真の濃度 = {c_MO_true:.2e} mol/L, 計算濃度 = {c_MO_calc:.2e} mol/L")
    print(f"相対誤差（MB）: {abs(c_MB_calc - c_MB_true)/c_MB_true * 100:.2f}%")
    print(f"相対誤差（MO）: {abs(c_MO_calc - c_MO_true)/c_MO_true * 100:.2f}%")
    print(f"残差: {residual:.6f}")
    
    3.9 時間分解UV-Vis分光法
    3.9.1 反応速度論的解析
    UV-Vis分光法は、化学反応の進行をリアルタイムで追跡できます。吸光度の時間変化から反応速度定数を決定できます。
    
    1次反応の速度式:
            \[
            \frac{d[A]}{dt} = -k[A]
            \]
            積分形:
            \[
            [A]_t = [A]_0 e^{-kt}
            \]
            吸光度での表現（\( A_t = \epsilon [A]_t l \)）:
            \[
            A_t = A_0 e^{-kt}
            \]
            \[
            \ln A_t = \ln A_0 - kt
            \]
        
    コード例8: 時間分解UV-Vis分光法による1次反応速度定数の決定
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def first_order_kinetics(time, A0, k):
        """
        1次反応の吸光度時間変化
    
        Parameters:
        -----------
        time : array
            時間 (s)
        A0 : float
            初期吸光度
        k : float
            速度定数 (s^-1)
    
        Returns:
        --------
        absorbance : array
            時間における吸光度
        """
        return A0 * np.exp(-k * time)
    
    def determine_rate_constant(time, absorbance):
        """
        時間分解UV-Visデータから1次反応速度定数を決定
    
        Parameters:
        -----------
        time : array
            時間 (s)
        absorbance : array
            各時間における吸光度
    
        Returns:
        --------
        k : float
            速度定数 (s^-1)
        half_life : float
            半減期 (s)
        """
        # 非線形フィッティング
        popt, pcov = curve_fit(first_order_kinetics, time, absorbance, p0=[absorbance[0], 0.01])
        A0_fit, k_fit = popt
    
        # 半減期
        half_life = np.log(2) / k_fit
    
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 吸光度 vs 時間（指数プロット）
        ax1.plot(time, absorbance, 'o', label='実測データ', markersize=8, alpha=0.7)
        time_fit = np.linspace(0, time.max(), 200)
        abs_fit = first_order_kinetics(time_fit, A0_fit, k_fit)
        ax1.plot(time_fit, abs_fit, 'r--', linewidth=2,
                 label=f'フィット: A = {A0_fit:.3f} exp(-{k_fit:.4f}t)\nk = {k_fit:.4f} s⁻¹\nt₁/₂ = {half_life:.1f} s')
        ax1.set_xlabel('時間 (s)', fontsize=12)
        ax1.set_ylabel('吸光度', fontsize=12)
        ax1.set_title('1次反応の時間変化', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
        # ln(A) vs 時間（線形プロット）
        ln_abs = np.log(absorbance)
        ax2.plot(time, ln_abs, 'o', label='実測データ', markersize=8, alpha=0.7)
        ln_abs_fit = np.log(A0_fit) - k_fit * time_fit
        ax2.plot(time_fit, ln_abs_fit, 'r--', linewidth=2,
                 label=f'線形フィット\n傾き = -{k_fit:.4f} s⁻¹')
        ax2.set_xlabel('時間 (s)', fontsize=12)
        ax2.set_ylabel('ln(吸光度)', fontsize=12)
        ax2.set_title('1次プロット（対数）', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        return k_fit, half_life, A0_fit
    
    # シミュレーション：クリスタルバイオレットの塩基加水分解反応
    time_data = np.linspace(0, 300, 30)  # 0-300秒、30点測定
    k_true = 0.012  # s^-1
    A0_true = 1.2
    absorbance_data = first_order_kinetics(time_data, A0_true, k_true)
    absorbance_data += np.random.normal(0, 0.02, len(time_data))  # ノイズ
    
    # 速度定数決定
    k_calc, t_half, A0_calc = determine_rate_constant(time_data, absorbance_data)
    
    print("反応速度論的解析結果:")
    print(f"速度定数 k = {k_calc:.4f} s⁻¹（真値: {k_true:.4f} s⁻¹）")
    print(f"半減期 t₁/₂ = {t_half:.1f} s")
    print(f"初期吸光度 A₀ = {A0_calc:.3f}")
    print(f"相対誤差: {abs(k_calc - k_true)/k_true * 100:.2f}%")
    
    3.10 演習問題
    
    基礎問題（Easy）
    問題1: 波長とエネルギーの変換
    UV-Vis分光法で観測された吸収極大波長が450 nmの化合物がある。この吸収に対応する光子エネルギーを eV 単位で計算せよ。
    
    解答を見る
    
    解答:
    波長とエネルギーの変換式を使用：
                    \[
                    E\,(\text{eV}) = \frac{1239.8}{\lambda\,(\text{nm})} = \frac{1239.8}{450} = 2.755\,\text{eV}
                    \]
                    答え: 2.76 eV
    Pythonコード:
    lambda_nm = 450
    E_eV = 1239.8 / lambda_nm
    print(f"光子エネルギー: {E_eV:.3f} eV")
    
    
    
    問題2: Lambert-Beer則による濃度計算
    モル吸光係数 \( \epsilon = 1.5 \times 10^4 \) L mol-1 cm-1 の化合物の溶液（光路長 1 cm）の吸光度が 0.75 であった。この溶液の濃度（mol/L）を求めよ。
    
    解答を見る
    
    解答:
    Lambert-Beer則: \( A = \epsilon c l \) より、
                    \[
                    c = \frac{A}{\epsilon l} = \frac{0.75}{1.5 \times 10^4 \times 1} = 5.0 \times 10^{-5}\,\text{mol/L}
                    \]
                    答え: 5.0 × 10-5 mol/L
    Pythonコード:
    A = 0.75
    epsilon = 1.5e4  # L mol^-1 cm^-1
    l = 1.0  # cm
    c = A / (epsilon * l)
    print(f"濃度: {c:.2e} mol/L")
    
    
    
    問題3: 透過率と吸光度の変換
    溶液の透過率が 40% であった。この溶液の吸光度を計算せよ。
    
    解答を見る
    
    解答:
    吸光度と透過率の関係式: \( A = -\log_{10} T = 2 - \log_{10}(\%T) \)
                    \[
                    A = 2 - \log_{10}(40) = 2 - 1.602 = 0.398
                    \]
                    答え: 0.398
    Pythonコード:
    import numpy as np
    T_percent = 40
    A = 2 - np.log10(T_percent)
    print(f"吸光度: {A:.3f}")
    
    
    
    
    
    中級問題（Medium）
    問題4: Taucプロットによるバンドギャップ決定
    ある半導体材料のUV-Visスペクトルから、以下のデータが得られた。Taucプロット（直接遷移）を作成し、バンドギャップを決定せよ。試料厚さ: 0.01 cm
    
    波長 (nm)400420440460480500
    吸光度1.201.050.850.600.350.15
    
    
    解答を見る
    
    解答:
    1. 波長を光子エネルギーに変換: \( E = 1239.8 / \lambda \)
    2. 吸収係数を計算: \( \alpha = 2.303 \cdot A / l \)
    3. Taucプロット: \( (\alpha h\nu)^2 \) vs. \( h\nu \) の線形領域を外挿
    Pythonコード:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    wavelength = np.array([400, 420, 440, 460, 480, 500])
    absorbance = np.array([1.20, 1.05, 0.85, 0.60, 0.35, 0.15])
    thickness = 0.01  # cm
    
    # 光子エネルギー
    E = 1239.8 / wavelength  # eV
    
    # 吸収係数
    alpha = 2.303 * absorbance / thickness  # cm^-1
    
    # Taucプロット
    tauc_y = (alpha * E)**2
    
    # 線形領域のフィッティング（E > 2.7 eV）
    mask = E > 2.7
    slope, intercept, r_value, _, _ = linregress(E[mask], tauc_y[mask])
    
    # バンドギャップ（横軸切片）
    Eg = -intercept / slope
    
    plt.figure(figsize=(10, 6))
    plt.plot(E, tauc_y, 'o', markersize=10, label='実測データ')
    E_fit = np.linspace(Eg, E.max(), 100)
    tauc_fit = slope * E_fit + intercept
    plt.plot(E_fit, tauc_fit, 'r--', linewidth=2, label=f'Eg = {Eg:.3f} eV')
    plt.axvline(Eg, color='green', linestyle=':', linewidth=2)
    plt.xlabel('光子エネルギー (eV)', fontsize=12)
    plt.ylabel('(αhν)² (eV² cm⁻²)', fontsize=12)
    plt.title('Taucプロット', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"バンドギャップ Eg = {Eg:.3f} eV")
    print(f"対応波長: λ = {1239.8/Eg:.1f} nm")
    
    答え: Eg ≈ 2.6-2.7 eV（実際のフィッティング結果による）
    
    
    問題5: 配位子場理論による錯体の色予測
    [Co(H2O)6]2+ の配位子場分裂エネルギーが \( \Delta_o = 9300 \) cm-1 である。この錯体が吸収する光の波長と、観測される錯体の色を予測せよ。
    
    解答を見る
    
    解答:
    1. 吸収波長を計算:
                    \[
                    \lambda = \frac{1}{\Delta_o\,(\text{cm}^{-1})} \times 10^7\,(\text{nm}) = \frac{10^7}{9300} = 1075\,\text{nm}
                    \]
                    2. 1075 nm は近赤外域（可視域外）。しかし、Co2+ (d7) は複数のd-d遷移を持ち、実際には可視域にも吸収がある。
    3. [Co(H2O)6]2+ は約 510 nm（緑色）に強い吸収を持ち、ピンク色（緑の補色）に見える。
    Pythonコード:
    delta_o_cm = 9300
    wavelength_nm = 1e7 / delta_o_cm
    print(f"Δo対応波長: {wavelength_nm:.1f} nm（近赤外）")
    print("実際の[Co(H2O)6]2+錯体:")
    print("- 主吸収: 510 nm（緑色）")
    print("- 観測色: ピンク色（緑の補色）")
    
    答え: 錯体の色はピンク色
    
    
    問題6: 多波長解析による2成分定量
    メチレンブルー（MB、\( \epsilon_{664} = 8 \times 10^4 \) L mol-1 cm-1）とメチルオレンジ（MO、\( \epsilon_{464} = 2.7 \times 10^4 \)）の混合溶液を 1 cm セルで測定した結果、A664 = 0.40、A464 = 0.54 であった。各成分の濃度を求めよ。ただし、664 nm では MO は吸収せず、464 nm では MB の吸収は無視できるとする。
    
    解答を見る
    
    解答:
    664 nm（MB のみ吸収）:
                    \[
                    c_{\text{MB}} = \frac{A_{664}}{\epsilon_{\text{MB},664} \cdot l} = \frac{0.40}{8 \times 10^4 \times 1} = 5.0 \times 10^{-6}\,\text{mol/L}
                    \]
    
                    464 nm（MO のみ吸収）:
                    \[
                    c_{\text{MO}} = \frac{A_{464}}{\epsilon_{\text{MO},464} \cdot l} = \frac{0.54}{2.7 \times 10^4 \times 1} = 2.0 \times 10^{-5}\,\text{mol/L}
                    \]
    
                    Pythonコード:
    A_664 = 0.40
    A_464 = 0.54
    epsilon_MB_664 = 8e4  # L mol^-1 cm^-1
    epsilon_MO_464 = 2.7e4
    l = 1.0  # cm
    
    c_MB = A_664 / (epsilon_MB_664 * l)
    c_MO = A_464 / (epsilon_MO_464 * l)
    
    print(f"メチレンブルー濃度: {c_MB:.2e} mol/L")
    print(f"メチルオレンジ濃度: {c_MO:.2e} mol/L")
    
    答え: MB = 5.0 × 10-6 mol/L, MO = 2.0 × 10-5 mol/L
    
    
    
    
    上級問題（Hard）
    問題7: 温度依存性UV-Visスペクトルからの熱力学パラメータ決定
    ある平衡系 A ⇌ B の平衡定数 \( K \) を、異なる温度で UV-Vis 分光法により決定した。以下のデータから、van't Hoff プロットを作成し、反応のエンタルピー変化 \( \Delta H^\circ \) とエントロピー変化 \( \Delta S^\circ \) を求めよ。
    
    温度 (K)298308318328338
    平衡定数 K0.500.801.201.752.40
    
    
    解答を見る
    
    解答:
    van't Hoff 式:
                    \[
                    \ln K = -\frac{\Delta H^\circ}{R} \cdot \frac{1}{T} + \frac{\Delta S^\circ}{R}
                    \]
                    \( \ln K \) vs. \( 1/T \) のプロットの傾きから \( \Delta H^\circ \)、切片から \( \Delta S^\circ \) を決定。
    Pythonコード:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    T = np.array([298, 308, 318, 328, 338])  # K
    K = np.array([0.50, 0.80, 1.20, 1.75, 2.40])
    
    # van't Hoffプロット
    inv_T = 1 / T  # K^-1
    ln_K = np.log(K)
    
    # 線形回帰
    slope, intercept, r_value, _, _ = linregress(inv_T, ln_K)
    
    # 熱力学パラメータ
    R = 8.314  # J mol^-1 K^-1
    Delta_H = -slope * R  # J/mol
    Delta_S = intercept * R  # J/(mol K)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(inv_T * 1000, ln_K, 'o', markersize=10, label='実測データ')
    inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
    ln_K_fit = slope * inv_T_fit + intercept
    plt.plot(inv_T_fit * 1000, ln_K_fit, 'r--', linewidth=2,
             label=f'ΔH° = {Delta_H/1000:.2f} kJ/mol\nΔS° = {Delta_S:.2f} J/(mol·K)\nR² = {r_value**2:.4f}')
    plt.xlabel('1000/T (K⁻¹)', fontsize=12)
    plt.ylabel('ln K', fontsize=12)
    plt.title("van't Hoffプロット", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"エンタルピー変化: ΔH° = {Delta_H/1000:.2f} kJ/mol")
    print(f"エントロピー変化: ΔS° = {Delta_S:.2f} J/(mol·K)")
    print(f"相関係数: R² = {r_value**2:.4f}")
    
    答え: ΔH° ≈ 35-40 kJ/mol, ΔS° ≈ 90-100 J/(mol·K)（実データによる）
    
    
    問題8: 散乱光を含むスペクトルの高度なベースライン補正
    ナノ粒子懸濁液の UV-Vis スペクトルには、Rayleigh 散乱（\( \propto \lambda^{-4} \)）と Mie 散乱（\( \propto \lambda^{-n}, n < 4 \)）が重畳する。以下のスペクトルから、真の吸収スペクトルを抽出し、バンドギャップを決定する Python プログラムを作成せよ。
    
    解答を見る
    
    解答:
    ベースライン補正戦略:
    
    吸収端より長波長側のデータから散乱成分をフィッティング
    \( A_{\text{scattering}} = C \lambda^{-n} \) の形式でフィッティング
    全波長範囲で散乱成分を減算
    補正後のスペクトルから Tauc プロットでバンドギャップ決定
    
    Pythonコード:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def scattering_baseline(wavelength, C, n):
        """散乱ベースライン（べき乗則）"""
        return C * wavelength**(-n)
    
    def advanced_baseline_correction(wavelength, absorbance, scattering_region):
        """
        散乱光を含むスペクトルの高度なベースライン補正
    
        Parameters:
        -----------
        wavelength : array
            波長 (nm)
        absorbance : array
            観測吸光度
        scattering_region : tuple
            散乱フィッティング領域 (nm)
    
        Returns:
        --------
        corrected_absorbance : array
            補正後の吸光度
        """
        # 散乱領域のデータ
        mask = (wavelength >= scattering_region[0]) & (wavelength <= scattering_region[1])
        wl_scatter = wavelength[mask]
        abs_scatter = absorbance[mask]
    
        # べき乗則フィッティング
        popt, _ = curve_fit(scattering_baseline, wl_scatter, abs_scatter, p0=[1e7, 4.0], maxfev=5000)
        C, n = popt
    
        # 全波長範囲で散乱成分を計算
        baseline = scattering_baseline(wavelength, C, n)
    
        # ベースライン減算
        corrected = absorbance - baseline
        corrected = np.maximum(corrected, 0)  # 負の値を0にクリップ
    
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 元のスペクトル
        axes[0, 0].plot(wavelength, absorbance, label='観測スペクトル', color='blue')
        axes[0, 0].plot(wavelength, baseline, '--', label=f'散乱ベースライン\n(λ^-{n:.2f})', color='red', linewidth=2)
        axes[0, 0].set_xlabel('波長 (nm)')
        axes[0, 0].set_ylabel('吸光度')
        axes[0, 0].set_title('散乱光を含むスペクトル')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
        # 補正スペクトル
        axes[0, 1].plot(wavelength, corrected, label='補正後スペクトル', color='green', linewidth=2)
        axes[0, 1].set_xlabel('波長 (nm)')
        axes[0, 1].set_ylabel('吸光度')
        axes[0, 1].set_title('ベースライン補正後')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
        # Taucプロット
        E = 1239.8 / wavelength
        alpha = 2.303 * corrected / 0.01  # 仮定: 試料厚さ 0.01 cm
        tauc_y = (alpha * E)**2
    
        # バンドギャップ決定
        mask_tauc = (E > 2.5) & (E < 3.5)
        if np.sum(mask_tauc) > 5:
            from scipy.stats import linregress
            slope_t, intercept_t, _, _, _ = linregress(E[mask_tauc], tauc_y[mask_tauc])
            Eg = -intercept_t / slope_t if slope_t > 0 else np.nan
        else:
            Eg = np.nan
    
        axes[1, 0].plot(E, tauc_y, 'o-', label='Taucプロット')
        if not np.isnan(Eg):
            E_fit = np.linspace(Eg, E[mask_tauc].max(), 100)
            tauc_fit = slope_t * E_fit + intercept_t
            axes[1, 0].plot(E_fit, tauc_fit, 'r--', linewidth=2, label=f'Eg = {Eg:.3f} eV')
            axes[1, 0].axvline(Eg, color='green', linestyle=':', linewidth=2)
        axes[1, 0].set_xlabel('光子エネルギー (eV)')
        axes[1, 0].set_ylabel('(αhν)² (eV² cm⁻²)')
        axes[1, 0].set_title('Taucプロット（補正後）')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
        # 散乱指数の評価
        axes[1, 1].text(0.5, 0.5, f'散乱解析結果:\n\n散乱指数 n = {n:.2f}\n\nRayleigh散乱 (n=4): 小粒子\nMie散乱 (n<4): 大粒子\n\nバンドギャップ Eg = {Eg:.3f} eV',
                       transform=axes[1, 1].transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
    
        plt.tight_layout()
        plt.show()
    
        return corrected, baseline, Eg
    
    # シミュレーションデータ
    wavelength = np.linspace(300, 800, 500)
    # 真の吸収（TiO2, Eg=3.2 eV）
    Eg_true = 3.2
    alpha_true = 1e4 * np.maximum(0, (1239.8/wavelength - Eg_true))**2
    true_abs = alpha_true * 0.01 / 2.303
    
    # 散乱成分（Rayleigh + Mie）
    scattering = 0.5 * (wavelength / 300)**(-3.5)
    
    # 観測スペクトル
    observed = true_abs + scattering + np.random.normal(0, 0.01, len(wavelength))
    
    # 高度なベースライン補正
    corrected, baseline, Eg_calc = advanced_baseline_correction(
        wavelength, observed, scattering_region=(600, 800)
    )
    
    print(f"決定されたバンドギャップ: Eg = {Eg_calc:.3f} eV")
    print(f"真のバンドギャップ: Eg = {Eg_true:.3f} eV")
    print(f"誤差: {abs(Eg_calc - Eg_true):.3f} eV")
    
    答え: Egの精密決定（0.1 eV以内の誤差）
    
    
    問題9: 機械学習によるUV-Visスペクトルからの構造予測
    化合物の UV-Vis スペクトルから、その分子構造（共役系の長さ、官能基の種類）を予測する機械学習モデルを構築せよ。scikit-learn のランダムフォレスト回帰を用いて、吸収極大波長から共役二重結合の数を予測するプログラムを作成せよ。
    
    解答を見る
    
    解答:
    共役系の長さと吸収波長の関係（Woodward-Fieser則に基づく）:
                    \[
                    \lambda_{\max} = \lambda_{\text{base}} + \Delta \lambda \times n
                    \]
                    ここで、\( n \) は共役二重結合の数。
    Pythonコード:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # トレーニングデータ（共役ポリエンの吸収波長データ）
    # n: 共役二重結合の数, lambda_max: 吸収極大波長 (nm)
    n_conjugated = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lambda_max = np.array([165, 217, 258, 290, 315, 334, 349, 364, 377, 390])
    
    # 特徴量（吸収波長、モル吸光係数、吸収バンド幅などを含むことも可能）
    X = lambda_max.reshape(-1, 1)
    y = n_conjugated
    
    # トレーニング・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ランダムフォレスト回帰モデル
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 予測
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # 評価
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # プロット
    plt.figure(figsize=(12, 5))
    
    # 訓練データとモデル
    plt.subplot(1, 2, 1)
    lambda_range = np.linspace(150, 400, 200).reshape(-1, 1)
    n_predicted = rf_model.predict(lambda_range)
    plt.plot(lambda_range, n_predicted, 'r-', linewidth=2, label='RF予測モデル')
    plt.scatter(X_train, y_train, s=100, alpha=0.7, label='訓練データ', color='blue')
    plt.scatter(X_test, y_test, s=100, alpha=0.7, label='テストデータ', color='green')
    plt.xlabel('吸収極大波長 (nm)', fontsize=12)
    plt.ylabel('共役二重結合の数', fontsize=12)
    plt.title(f'ランダムフォレスト回帰\nR²(train) = {r2_train:.3f}, R²(test) = {r2_test:.3f}',
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 予測精度
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, s=100, alpha=0.7, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2, label='理想直線')
    plt.xlabel('真の共役二重結合数', fontsize=12)
    plt.ylabel('予測共役二重結合数', fontsize=12)
    plt.title(f'予測精度\nMAE = {mae_test:.2f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 未知試料の予測
    unknown_lambda = np.array([[280], [350], [400]])
    predicted_n = rf_model.predict(unknown_lambda)
    print("未知試料の構造予測:")
    for lam, n_pred in zip(unknown_lambda.flatten(), predicted_n):
        print(f"  λmax = {lam:.0f} nm → 予測共役数: {n_pred:.1f}")
    
    答え: R² > 0.95 の高精度予測モデル構築
    
    
    
    
    学習目標の確認
    以下の項目について、自己評価してください：
    レベル1: 基本理解
    
    UV-Vis分光法の原理と電子遷移の種類を説明できる
    Lambert-Beer則を用いた濃度計算ができる
    波長とエネルギーの変換ができる
    吸光度と透過率の関係を理解している
    
    レベル2: 実践スキル
    
    Taucプロット法でバンドギャップを決定できる
    検量線法による定量分析を実施できる
    配位子場理論を用いて遷移金属錯体の色を予測できる
    ベースライン補正とスペクトル前処理ができる
    多波長解析による多成分定量ができる
    
    レベル3: 応用力
    
    散乱光を含む複雑なスペクトルを補正できる
    時間分解UV-Vis分光法で反応速度定数を決定できる
    溶媒効果を考慮したスペクトル解析ができる
    機械学習を用いたスペクトルデータ解析ができる
    
    
    
    参考文献
    
    Atkins, P., de Paula, J. (2010). Physical Chemistry (9th ed.). Oxford University Press, pp. 465-468 (Beer-Lambert law), pp. 495-502 (electronic transitions), pp. 510-518 (ligand field theory). - UV-Vis分光法の量子力学的基礎、電子遷移の選択則、配位子場理論の詳細な解説
    Figgis, B.N., Hitchman, M.A. (2000). Ligand Field Theory and Its Applications. Wiley-VCH, pp. 85-105 (d-orbital splitting), pp. 120-135 (spectrochemical series), pp. 140-150 (electronic spectra of complexes). - 配位子場理論の体系的解説、d-d遷移と錯体の色、分光化学系列の理論的背景
    Tauc, J., Grigorovici, R., Vancu, A. (1966). Optical properties and electronic structure of amorphous germanium. Physica Status Solidi (b), 15(2), 627-637. DOI: 10.1002/pssb.19660150224 - Taucプロット法の原論文、半導体のバンドギャップ決定法の確立
    Perkampus, H.-H. (1992). UV-VIS Spectroscopy and Its Applications. Springer, pp. 1-18 (principles), pp. 32-48 (quantitative analysis), pp. 120-145 (solvent effects), pp. 165-180 (practical applications). - UV-Vis分光法の実践的応用、定量分析法、スペクトル解釈の実例集
    Casida, M. E. (1995). Time-dependent density functional response theory for molecules. In Recent Advances in Density Functional Methods (Part I), pp. 155-192. World Scientific, Singapore. - TDDFT法によるUV-Visスペクトル計算の理論的基礎
    SciPy 1.11 documentation. scipy.optimize.curve_fit, scipy.optimize.nnls. https://docs.scipy.org/doc/scipy/reference/optimize.html - 非線形最小二乗フィッティング、非負最小二乗法、スペクトルフィッティングへの応用
    Makuła, P., Pacia, M., Macyk, W. (2018). How to correctly determine the band gap energy of modified semiconductor photocatalysts based on UV–Vis spectra. Journal of Physical Chemistry Letters, 9(23), 6814-6817. DOI: 10.1021/acs.jpclett.8b02892 - Taucプロット法の正確な適用法、よくある誤りとその回避法
    scikit-learn 1.3 documentation. Ensemble methods (RandomForestRegressor). https://scikit-learn.org/stable/modules/ensemble.html#forest - ランダムフォレスト回帰、UV-Visスペクトルの機械学習解析への応用
    
    
    
    
    免責事項
    
    本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
    本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
    外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
    本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
    本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
    本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
    ```
