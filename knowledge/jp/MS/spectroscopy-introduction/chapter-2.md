---
title: 第2章：赤外・ラマン分光法
chapter_title: 第2章：赤外・ラマン分光法
subtitle: 振動分光で探る分子構造と化学結合
---

## イントロダクション

赤外分光（Infrared Spectroscopy, IR）とラマン分光（Raman Spectroscopy）は、分子の振動情報を通じて化学結合、官能基、結晶構造を解明する相補的な手法です。IRは赤外光の吸収を測定し、Ramanは散乱光の周波数シフトを観測します。両者は異なる選択則に従うため、IRで活性な振動がRamanで不活性、またはその逆という相補性を持ちます。

**IRとRamanの使い分け**  

  * **IR** : 極性基（C=O, O-H, N-H）の検出、有機物の官能基同定、固体・液体・気体すべてに適用可能
  * **Raman** : 対称振動（C=C, S-S）の検出、水溶液試料、結晶性評価（低波数領域）、非破壊・非接触測定

## 1\. 分子振動の基礎

### 1.1 調和振動子モデル

2原子分子の振動は調和振動子で近似できます。ポテンシャルエネルギーはHookeの法則に従います：

$$V(r) = \frac{1}{2}k(r - r_e)^2$$

ここで、$k$ は力の定数（N/m）、$r_e$ は平衡核間距離です。振動周波数 $\nu$ は以下で与えられます：

$$\nu = \frac{1}{2\pi}\sqrt{\frac{k}{\mu}}$$

$\mu = \frac{m_1 m_2}{m_1 + m_2}$ は換算質量です。振動エネルギー準位は量子化され、

$$E_v = h\nu \left(v + \frac{1}{2}\right), \quad v = 0, 1, 2, \ldots$$

調和振動子近似では、選択則は $\Delta v = \pm 1$ です（基本振動のみ許容）。実際の分子では非調和性により $\Delta v = \pm 2, \pm 3, \ldots$（倍音）も弱く観測されます。

#### コード例1: 調和振動子のエネルギー準位と振動周波数計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    h = 6.62607015e-34  # J·s
    c = 2.99792458e8    # m/s
    u = 1.66053906660e-27  # kg (atomic mass unit)
    
    def vibrational_frequency(k, m1, m2):
        """
        2原子分子の振動周波数（Hz）と波数（cm^-1）を計算
    
        Parameters:
        -----------
        k : float
            力の定数（N/m）
        m1, m2 : float
            原子の質量（amu）
    
        Returns:
        --------
        freq_Hz : float
            振動周波数（Hz）
        wavenumber : float
            波数（cm^-1）
        """
        # 換算質量
        mu = (m1 * m2) / (m1 + m2) * u  # kg
    
        # 振動周波数
        freq_Hz = (1 / (2 * np.pi)) * np.sqrt(k / mu)
    
        # 波数に変換
        wavenumber = freq_Hz / (c * 100)  # cm^-1
    
        return freq_Hz, wavenumber
    
    def energy_levels(v_max, freq_Hz):
        """
        調和振動子のエネルギー準位
    
        Parameters:
        -----------
        v_max : int
            最大振動量子数
        freq_Hz : float
            振動周波数（Hz）
    
        Returns:
        --------
        v : array
            振動量子数
        E : array
            エネルギー（eV）
        """
        v = np.arange(0, v_max + 1)
        E_J = h * freq_Hz * (v + 0.5)
        E_eV = E_J / 1.602176634e-19
        return v, E_eV
    
    # 典型的な化学結合の計算
    bonds = {
        'C-H': {'k': 500, 'm1': 12, 'm2': 1},
        'C=O': {'k': 1200, 'm1': 12, 'm2': 16},
        'C-C': {'k': 400, 'm1': 12, 'm2': 12},
        'O-H': {'k': 750, 'm1': 16, 'm2': 1}
    }
    
    print("=" * 70)
    print("典型的な化学結合の振動周波数")
    print("=" * 70)
    print(f"{'結合':<8} {'力の定数 (N/m)':<18} {'周波数 (Hz)':<18} {'波数 (cm⁻¹)':<15}")
    print("-" * 70)
    
    for bond, params in bonds.items():
        freq_Hz, wavenumber = vibrational_frequency(params['k'], params['m1'], params['m2'])
        print(f"{bond:<8} {params['k']:<18} {freq_Hz:.3e} {wavenumber:<15.1f}")
    
    # C=O伸縮振動のエネルギー準位図
    freq_Hz_CO, wn_CO = vibrational_frequency(1200, 12, 16)
    v, E = energy_levels(5, freq_Hz_CO)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # エネルギー準位図
    ax1.hlines(E, 0, 1, colors='#f093fb', linewidths=3)
    for i, (vi, Ei) in enumerate(zip(v, E)):
        ax1.text(1.1, Ei, f'v={vi}', fontsize=11, va='center')
        if i < len(v) - 1:
            # 遷移の矢印
            ax1.annotate('', xy=(0.5, E[i+1]), xytext=(0.5, E[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.set_xlim(-0.2, 1.5)
    ax1.set_ylim(E[0] - 0.05, E[-1] + 0.1)
    ax1.set_ylabel('エネルギー (eV)', fontsize=12)
    ax1.set_title('C=O伸縮振動のエネルギー準位', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(axis='y', alpha=0.3)
    
    # 同位体効果
    masses_C = np.array([12, 13, 14])
    wavenumbers = []
    for m_C in masses_C:
        _, wn = vibrational_frequency(1200, m_C, 16)
        wavenumbers.append(wn)
    
    ax2.bar([f'$^{{{int(m)}}}$C=O' for m in masses_C], wavenumbers,
            color=['#f093fb', '#f5576c', '#4ecdc4'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('波数 (cm⁻¹)', fontsize=12)
    ax2.set_title('同位体効果：C=O伸縮振動', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (m, wn) in enumerate(zip(masses_C, wavenumbers)):
        ax2.text(i, wn + 20, f'{wn:.0f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('vibrational_fundamentals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nC=O伸縮振動の波数: {wn_CO:.1f} cm⁻¹")
    print(f"基底状態(v=0)のゼロ点エネルギー: {E[0]:.4f} eV")
    print(f"v=0 → v=1 遷移エネルギー: {E[1] - E[0]:.4f} eV")
    

### 1.2 多原子分子の振動モード

$N$ 原子からなる分子は $3N$ 個の自由度を持ち、そのうち3つは並進、（非線形分子では）3つは回転、残りの $3N - 6$（線形分子では $3N - 5$）が振動の自由度です。
    
    
    ```mermaid
    flowchart TD
                A[全自由度 3N] --> B[並進 3]
                A --> C[回転]
                A --> D[振動]
    
                C --> C1[非線形分子: 3]
                C --> C2[線形分子: 2]
    
                D --> D1[非線形: 3N-6]
                D --> D2[線形: 3N-5]
    
                D1 --> E[H₂O: 3モードCO₂: 4モードベンゼン: 30モード]
    
                style A fill:#f093fb,color:#fff
                style D fill:#f5576c,color:#fff
                style E fill:#a8e6cf,color:#000
    ```

振動モードは**伸縮振動（stretching）** と**変角振動（bending）** に分類されます：

  * **伸縮振動** : 対称伸縮（symmetric stretch, νₛ）、非対称伸縮（asymmetric stretch, νₐₛ）
  * **変角振動** : はさみ振動（scissoring, δ）、横揺れ振動（rocking, ρ）、縦揺れ振動（wagging, ω）、ねじれ振動（twisting, τ）

#### コード例2: H₂O分子の3つの振動モードシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    def h2o_normal_modes():
        """
        H2O分子の3つの振動モード（対称伸縮、非対称伸縮、変角）の可視化
    
        Returns:
        --------
        fig : matplotlib figure
            振動モードの図
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # 平衡位置（O原子を原点）
        O = np.array([0, 0])
        H1 = np.array([-0.76, 0.59])  # 左のH
        H2 = np.array([0.76, 0.59])   # 右のH
    
        modes = [
            {
                'name': '対称伸縮 (νₛ)',
                'freq': '3657 cm⁻¹',
                'displacements': {
                    'O': np.array([0, 0]),
                    'H1': np.array([-0.1, 0.08]),
                    'H2': np.array([0.1, 0.08])
                }
            },
            {
                'name': '変角振動 (δ)',
                'freq': '1595 cm⁻¹',
                'displacements': {
                    'O': np.array([0, 0]),
                    'H1': np.array([-0.05, -0.1]),
                    'H2': np.array([0.05, -0.1])
                }
            },
            {
                'name': '非対称伸縮 (νₐₛ)',
                'freq': '3756 cm⁻¹',
                'displacements': {
                    'O': np.array([0, 0]),
                    'H1': np.array([-0.1, 0.08]),
                    'H2': np.array([0.1, -0.08])
                }
            }
        ]
    
        for ax, mode in zip(axes, modes):
            # 平衡位置
            ax.plot(*O, 'ro', markersize=15, label='O')
            ax.plot(*H1, 'bo', markersize=10, label='H')
            ax.plot(*H2, 'bo', markersize=10)
            ax.plot([O[0], H1[0]], [O[1], H1[1]], 'k-', linewidth=2)
            ax.plot([O[0], H2[0]], [O[1], H2[1]], 'k-', linewidth=2)
    
            # 振動の変位（拡大表示）
            scale = 2
            O_disp = O + scale * mode['displacements']['O']
            H1_disp = H1 + scale * mode['displacements']['H1']
            H2_disp = H2 + scale * mode['displacements']['H2']
    
            ax.plot(*O_disp, 'ro', markersize=15, alpha=0.3)
            ax.plot(*H1_disp, 'bo', markersize=10, alpha=0.3)
            ax.plot(*H2_disp, 'bo', markersize=10, alpha=0.3)
            ax.plot([O_disp[0], H1_disp[0]], [O_disp[1], H1_disp[1]], 'k--', linewidth=2, alpha=0.3)
            ax.plot([O_disp[0], H2_disp[0]], [O_disp[1], H2_disp[1]], 'k--', linewidth=2, alpha=0.3)
    
            # 変位ベクトル
            ax.arrow(*O, *mode['displacements']['O'], head_width=0.08, head_length=0.05, fc='red', ec='red')
            ax.arrow(*H1, *mode['displacements']['H1'], head_width=0.08, head_length=0.05, fc='blue', ec='blue')
            ax.arrow(*H2, *mode['displacements']['H2'], head_width=0.08, head_length=0.05, fc='blue', ec='blue')
    
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.5, 1)
            ax.set_aspect('equal')
            ax.set_title(f"{mode['name']}\n{mode['freq']}", fontsize=12, fontweight='bold')
            ax.axis('off')
    
        plt.tight_layout()
        plt.savefig('h2o_normal_modes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return fig
    
    # 実行
    fig = h2o_normal_modes()
    
    # 振動モードの特徴
    print("=" * 60)
    print("H₂O分子の3つの基本振動モード")
    print("=" * 60)
    print("1. 対称伸縮 (νₛ): 3657 cm⁻¹")
    print("   - 両方のO-H結合が同時に伸縮")
    print("   - IRで強い吸収（双極子モーメント変化大）")
    print("")
    print("2. 変角振動 (δ): 1595 cm⁻¹")
    print("   - H-O-H角が変化")
    print("   - 最も低い振動数（弱い力の定数）")
    print("")
    print("3. 非対称伸縮 (νₐₛ): 3756 cm⁻¹")
    print("   - 一方のO-Hが伸びるとき、他方は縮む")
    print("   - IRで最も強い吸収")
    

## 2\. 赤外分光法（IR）

### 2.1 IR吸収の選択則

IRで振動が活性（IR active）であるためには、振動に伴って**双極子モーメント $\boldsymbol{\mu}$ が変化** する必要があります：

$$\left(\frac{\partial \boldsymbol{\mu}}{\partial Q}\right)_0 \neq 0$$

ここで、$Q$ は振動の規準座標です。対称分子（CO₂など）の対称伸縮振動はIR不活性ですが、非対称伸縮や変角振動はIR活性です。

### 2.2 官能基と特性吸収

有機化合物の官能基は特定の波数領域に特性吸収を示します。これにより、IRスペクトルから分子構造を推定できます。

官能基 | 振動モード | 波数（cm⁻¹） | 強度  
---|---|---|---  
O-H（アルコール） | 伸縮 | 3200-3600 | 強、幅広  
N-H | 伸縮 | 3300-3500 | 中  
C-H（脂肪族） | 伸縮 | 2850-2960 | 強  
C≡N | 伸縮 | 2210-2260 | 中  
C=O（カルボニル） | 伸縮 | 1650-1750 | 非常に強  
C=C | 伸縮 | 1620-1680 | 弱〜中  
C-O | 伸縮 | 1000-1300 | 強  
芳香環 | 面外変角 | 690-900 | 強  
  
#### コード例3: IRスペクトルシミュレーション（エタノール）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lorentzian_peak(x, center, intensity, width):
        """Lorentzian線形関数"""
        return intensity * (width**2 / ((x - center)**2 + width**2))
    
    def simulate_ir_spectrum(peaks, x_range=(4000, 400)):
        """
        IRスペクトルのシミュレーション
    
        Parameters:
        -----------
        peaks : list of dict
            各ピークの情報 [{'center': cm-1, 'intensity': 0-1, 'width': cm-1, 'label': str}, ...]
        x_range : tuple
            波数範囲（cm⁻¹）
    
        Returns:
        --------
        wavenumbers : array
            波数軸
        transmittance : array
            透過率（%）
        """
        wavenumbers = np.linspace(x_range[0], x_range[1], 2000)
        absorbance = np.zeros_like(wavenumbers)
    
        for peak in peaks:
            absorbance += lorentzian_peak(wavenumbers, peak['center'],
                                           peak['intensity'], peak['width'])
    
        transmittance = 100 * np.exp(-absorbance)
        return wavenumbers, transmittance, peaks
    
    # エタノール (CH₃CH₂OH) のIRスペクトル
    ethanol_peaks = [
        {'center': 3350, 'intensity': 1.5, 'width': 100, 'label': 'O-H伸縮'},
        {'center': 2970, 'intensity': 0.8, 'width': 20, 'label': 'C-H伸縮（CH₃）'},
        {'center': 2930, 'intensity': 0.7, 'width': 20, 'label': 'C-H伸縮（CH₂）'},
        {'center': 1450, 'intensity': 0.4, 'width': 30, 'label': 'C-H変角'},
        {'center': 1380, 'intensity': 0.3, 'width': 20, 'label': 'C-H変角'},
        {'center': 1050, 'intensity': 1.0, 'width': 40, 'label': 'C-O伸縮'},
        {'center': 880, 'intensity': 0.5, 'width': 25, 'label': 'C-C伸縮'},
    ]
    
    wavenumbers, transmittance, peaks = simulate_ir_spectrum(ethanol_peaks)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(wavenumbers, transmittance, linewidth=1.5, color='#f093fb')
    ax.fill_between(wavenumbers, transmittance, 100, alpha=0.2, color='#f5576c')
    
    # ピークのラベル
    for peak in peaks:
        idx = np.argmin(np.abs(wavenumbers - peak['center']))
        y_pos = transmittance[idx]
        if y_pos < 50:
            ax.annotate(peak['label'], xy=(peak['center'], y_pos),
                       xytext=(peak['center'], y_pos - 15),
                       fontsize=9, ha='center', rotation=90,
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    ax.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax.set_ylabel('透過率 (%)', fontsize=12)
    ax.set_title('エタノール (CH₃CH₂OH) のIRスペクトル（シミュレーション）',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 105)
    ax.invert_xaxis()  # IRスペクトルは通常高波数側が左
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ethanol_ir_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ピーク帰属表
    print("=" * 70)
    print("エタノールのIRスペクトル帰属")
    print("=" * 70)
    print(f"{'波数 (cm⁻¹)':<15} {'強度':<10} {'帰属':<30}")
    print("-" * 70)
    for peak in sorted(peaks, key=lambda x: x['center'], reverse=True):
        intensity_str = '強' if peak['intensity'] > 1.0 else ('中' if peak['intensity'] > 0.5 else '弱')
        print(f"{peak['center']:<15} {intensity_str:<10} {peak['label']:<30}")
    

### 2.3 FTIR（フーリエ変換赤外分光法）

現代のIR分光計はマイケルソン干渉計を用いたFTIR（Fourier Transform Infrared Spectroscopy）が主流です。干渉計で得られたインターフェログラム（時間領域信号）をフーリエ変換して周波数領域のスペクトルを得ます。

**FTIRの利点**

  * **高速測定** : 全波数を同時測定（Fellgettの利点）
  * **高感度** : 光エネルギーの利用効率が高い（Jacquinotの利点）
  * **高波数精度** : He-Neレーザーによる内部校正
  * **多数回積算** : S/N比向上

#### コード例4: FTIRのインターフェログラムとフーリエ変換
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    def generate_interferogram(wavenumbers_cm, intensities, mirror_displacement_max=0.05):
        """
        IRスペクトルからインターフェログラムを生成（簡略化版）
    
        Parameters:
        -----------
        wavenumbers_cm : array
            波数（cm⁻¹）
        intensities : array
            各波数の強度
        mirror_displacement_max : float
            ミラー移動距離の最大値（cm）
    
        Returns:
        --------
        displacement : array
            ミラー変位
        interferogram : array
            インターフェログラム
        """
        # ミラー変位
        n_points = 2048
        displacement = np.linspace(-mirror_displacement_max, mirror_displacement_max, n_points)
    
        # インターフェログラム（各波数成分の干渉パターンの和）
        interferogram = np.zeros_like(displacement)
        for wn, intensity in zip(wavenumbers_cm, intensities):
            # 波数をcm^-1からcm単位の波長に変換
            # cos(2π * wavenumber * displacement)
            interferogram += intensity * np.cos(2 * np.pi * wn * displacement)
    
        # DC成分追加
        interferogram += np.sum(intensities)
    
        return displacement, interferogram
    
    def fourier_transform_spectrum(displacement, interferogram):
        """
        インターフェログラムをフーリエ変換してスペクトルを復元
    
        Parameters:
        -----------
        displacement : array
            ミラー変位（cm）
        interferogram : array
            インターフェログラム
    
        Returns:
        --------
        wavenumbers : array
            波数（cm⁻¹）
        spectrum : array
            復元されたスペクトル
        """
        # フーリエ変換
        N = len(interferogram)
        spectrum_complex = fft(interferogram)
        spectrum = np.abs(spectrum_complex[:N//2])
    
        # 波数軸（cm⁻¹）
        delta_x = displacement[1] - displacement[0]
        wavenumbers = fftfreq(N, delta_x)[:N//2]
    
        return wavenumbers, spectrum
    
    # シミュレーション：3つのIRピーク
    true_wavenumbers = np.array([1000, 1700, 2900])  # cm^-1
    true_intensities = np.array([0.5, 1.0, 0.7])
    
    # インターフェログラム生成
    displacement, interferogram = generate_interferogram(true_wavenumbers, true_intensities)
    
    # フーリエ変換でスペクトル復元
    wavenumbers_ft, spectrum_ft = fourier_transform_spectrum(displacement, interferogram)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # インターフェログラム
    ax1.plot(displacement, interferogram, linewidth=1.5, color='#f093fb')
    ax1.set_xlabel('ミラー変位 (cm)', fontsize=12)
    ax1.set_ylabel('インターフェログラム強度', fontsize=12)
    ax1.set_title('FTIRインターフェログラム（時間領域）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Path Difference')
    ax1.legend()
    
    # フーリエ変換後のスペクトル
    ax2.plot(wavenumbers_ft, spectrum_ft, linewidth=1.5, color='#f5576c')
    ax2.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('フーリエ変換後のIRスペクトル（周波数領域）', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 4000)
    ax2.grid(True, alpha=0.3)
    
    # 真のピーク位置をマーク
    for wn, intensity in zip(true_wavenumbers, true_intensities):
        ax2.axvline(x=wn, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.text(wn, max(spectrum_ft) * 0.9, f'{wn} cm⁻¹',
                rotation=90, va='bottom', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig('ftir_interferogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)
    print("FTIR測定の原理")
    print("=" * 60)
    print("1. マイケルソン干渉計でミラーを移動")
    print("2. 干渉パターン（インターフェログラム）を記録")
    print("3. フーリエ変換で周波数領域のスペクトルを復元")
    print("")
    print("真のピーク位置: ", true_wavenumbers, " cm⁻¹")
    print("復元されたピーク: フーリエ変換後のスペクトルで確認")
    

## 3\. ラマン分光法

### 3.1 Raman散乱の原理

Raman散乱は、光と分子の相互作用により、入射光（周波数 $\nu_0$）が分子の振動エネルギー（周波数 $\nu_m$）だけシフトした散乱光として観測される現象です：

  * **Rayleigh散乱** （弾性散乱）: $\nu_{\text{scatter}} = \nu_0$（大多数、106倍強い）
  * **Stokes Raman散乱** : $\nu_{\text{scatter}} = \nu_0 - \nu_m$（分子が励起）
  * **Anti-Stokes Raman散乱** : $\nu_{\text{scatter}} = \nu_0 + \nu_m$（既に励起状態にある分子が基底状態へ、弱い）

Raman散乱の選択則は、振動に伴って**分極率 $\alpha$ が変化** することです：

$$\left(\frac{\partial \alpha}{\partial Q}\right)_0 \neq 0$$

この選択則はIRとは異なり、対称振動がRaman活性、非対称振動がRaman不活性となる場合が多いです（相補則）。

#### コード例5: RamanスペクトルとStokes/Anti-Stokesの比
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def boltzmann_population(E_vib_cm, T=300):
        """
        Boltzmann分布による振動励起状態の占有率
    
        Parameters:
        -----------
        E_vib_cm : float
            振動エネルギー（cm⁻¹）
        T : float
            温度（K）
    
        Returns:
        --------
        ratio : float
            v=1とv=0の占有率比 n₁/n₀
        """
        k_B = 1.380649e-23  # J/K
        h = 6.62607015e-34  # J·s
        c = 2.99792458e8    # m/s
    
        E_J = h * c * E_vib_cm * 100  # cm^-1 to J
        ratio = np.exp(-E_J / (k_B * T))
        return ratio
    
    def raman_spectrum_simulation(laser_wavelength=532):
        """
        Ramanスペクトル（Stokes/Anti-Stokes）のシミュレーション
    
        Parameters:
        -----------
        laser_wavelength : float
            励起レーザー波長（nm）
    
        Returns:
        --------
        fig : matplotlib figure
        """
        # 振動モード
        vibrations = [
            {'mode': 'C-C伸縮', 'shift': 1000, 'intensity': 0.8},
            {'mode': 'C=O伸縮', 'shift': 1700, 'intensity': 1.0},
            {'mode': 'C-H伸縮', 'shift': 2900, 'intensity': 0.6}
        ]
    
        # レーザー周波数
        laser_freq = 1e7 / laser_wavelength  # cm^-1
    
        # Raman shift軸（通常は-3500 ~ +3500 cm^-1）
        raman_shift = np.linspace(-3500, 3500, 3000)
    
        # スペクトル初期化
        spectrum = np.zeros_like(raman_shift)
    
        # 各振動モードのピーク
        for vib in vibrations:
            shift = vib['shift']
            intensity = vib['intensity']
    
            # Stokesピーク（正のシフト）
            stokes = intensity * np.exp(-(raman_shift - shift)**2 / (2 * 30**2))
            spectrum += stokes
    
            # Anti-Stokesピーク（負のシフト）
            # Boltzmann因子で強度が減少
            boltzmann_ratio = boltzmann_population(shift, T=300)
            anti_stokes = intensity * boltzmann_ratio * np.exp(-(raman_shift + shift)**2 / (2 * 30**2))
            spectrum += anti_stokes
    
        # Rayleigh散乱（中心、非常に強い）
        rayleigh = 100 * np.exp(-(raman_shift)**2 / (2 * 20**2))
    
        # プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
        # 全スペクトル（Rayleigh含む）
        ax1.plot(raman_shift, spectrum + rayleigh, linewidth=1.5, color='#f093fb')
        ax1.fill_between(raman_shift, spectrum + rayleigh, alpha=0.3, color='#f5576c')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Rayleigh散乱')
        ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('強度 (a.u.)', fontsize=12)
        ax1.set_title(f'Ramanスペクトル全体（レーザー: {laser_wavelength} nm）',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_ylim(0.01, 150)
    
        # Stokes領域の拡大（実用的な測定範囲）
        ax2.plot(raman_shift, spectrum, linewidth=1.5, color='#f5576c')
        ax2.fill_between(raman_shift, spectrum, alpha=0.3, color='#f093fb')
        ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('強度 (a.u.)', fontsize=12)
        ax2.set_title('Ramanスペクトル（Rayleigh除去後）', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
    
        # ピーク帰属
        for vib in vibrations:
            shift = vib['shift']
            ax2.text(shift, vib['intensity'] * 1.1, vib['mode'],
                    ha='center', fontsize=10, rotation=45)
            ax2.text(-shift, vib['intensity'] * boltzmann_population(shift) * 1.1,
                    vib['mode'] + '\n(Anti-Stokes)',
                    ha='center', fontsize=9, rotation=45, alpha=0.7)
    
        plt.tight_layout()
        plt.savefig('raman_spectrum_stokes_antistokes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # Boltzmann比の温度依存性
        print("=" * 70)
        print("Stokes/Anti-Stokes強度比の温度依存性")
        print("=" * 70)
        print(f"{'振動モード':<20} {'Raman shift (cm⁻¹)':<25} {'I(Anti-Stokes)/I(Stokes) at 300K':<30}")
        print("-" * 70)
        for vib in vibrations:
            ratio = boltzmann_population(vib['shift'], T=300)
            print(f"{vib['mode']:<20} {vib['shift']:<25} {ratio:<30.4f}")
    
        return fig
    
    # 実行
    fig = raman_spectrum_simulation(laser_wavelength=532)
    
    # 温度依存性の計算
    temperatures = np.linspace(100, 800, 50)
    raman_shift_1000 = 1000  # cm^-1
    
    ratios = [boltzmann_population(raman_shift_1000, T) for T in temperatures]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, ratios, linewidth=2, color='#f093fb')
    plt.xlabel('温度 (K)', fontsize=12)
    plt.ylabel('I(Anti-Stokes) / I(Stokes)', fontsize=12)
    plt.title('Raman強度比の温度依存性（1000 cm⁻¹モード）', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=boltzmann_population(1000, 300), color='red', linestyle='--',
               label='室温 (300 K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('raman_temperature_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 3.2 IRとRamanの相補則

中心対称分子（例: CO₂、ベンゼン）では、IRとRamanの選択則が相補的です：

**相互排他則（Mutual Exclusion Rule）**  
中心対称分子において、IR活性な振動はRaman不活性、Raman活性な振動はIR不活性となります。これは対称性による選択則の違いから生じます。 

振動モード | 対称性 | IR活性 | Raman活性  
---|---|---|---  
CO₂対称伸縮 | Σg⁺ | 不活性 | 活性  
CO₂非対称伸縮 | Σu⁺ | 活性 | 不活性  
CO₂変角振動 | Πu | 活性 | 不活性  
  
### 3.3 Raman分光の応用

  * **結晶性評価** : 低波数領域（<200 cm⁻¹）の格子振動モードから結晶化度を評価
  * **水溶液試料** : 水のIR吸収が強いが、Ramanでは水の影響が少ない
  * **非接触・非破壊測定** : レーザーを集光して微小領域測定（ラマン顕微鏡）
  * **表面増強Raman（SERS）** : 金属ナノ粒子表面で106〜1014倍の増強

#### コード例6: 結晶性評価のためのRamanピークフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def crystallinity_analysis(raman_shift, intensity):
        """
        Ramanスペクトルから結晶化度を評価（ポリマーの例）
    
        Parameters:
        -----------
        raman_shift : array
            Raman shift（cm⁻¹）
        intensity : array
            Raman強度
    
        Returns:
        --------
        crystallinity : float
            結晶化度（%）
        fit_params : dict
            フィッティングパラメータ
        """
        def two_peak_model(x, A1, c1, w1, A2, c2, w2):
            """結晶ピークと非晶ピークの2成分モデル"""
            peak1 = A1 * np.exp(-(x - c1)**2 / (2 * w1**2))  # 結晶ピーク
            peak2 = A2 * np.exp(-(x - c2)**2 / (2 * w2**2))  # 非晶ピーク
            return peak1 + peak2
    
        # 初期推定値
        p0 = [100, 1095, 10, 80, 1080, 15]
    
        # フィッティング
        popt, pcov = curve_fit(two_peak_model, raman_shift, intensity, p0=p0)
    
        # 個別ピーク
        crystal_peak = popt[0] * np.exp(-(raman_shift - popt[1])**2 / (2 * popt[2]**2))
        amorphous_peak = popt[3] * np.exp(-(raman_shift - popt[4])**2 / (2 * popt[5]**2))
    
        # 結晶化度（ピーク面積比）
        crystal_area = popt[0] * popt[2] * np.sqrt(2 * np.pi)
        amorphous_area = popt[3] * popt[5] * np.sqrt(2 * np.pi)
        crystallinity = crystal_area / (crystal_area + amorphous_area) * 100
    
        fit_params = {
            'crystal_center': popt[1],
            'crystal_width': popt[2],
            'amorphous_center': popt[4],
            'amorphous_width': popt[5],
            'crystal_peak': crystal_peak,
            'amorphous_peak': amorphous_peak,
            'fitted_curve': two_peak_model(raman_shift, *popt)
        }
    
        return crystallinity, fit_params
    
    # 合成データ（半結晶性ポリマーのC-C伸縮領域）
    raman_shift = np.linspace(1050, 1130, 300)
    crystal_peak_true = 70 * np.exp(-(raman_shift - 1095)**2 / (2 * 8**2))
    amorphous_peak_true = 50 * np.exp(-(raman_shift - 1080)**2 / (2 * 12**2))
    intensity = crystal_peak_true + amorphous_peak_true + np.random.normal(0, 2, len(raman_shift))
    
    # 結晶化度解析
    crystallinity, fit_params = crystallinity_analysis(raman_shift, intensity)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ピーク分離
    ax1.plot(raman_shift, intensity, 'k.', markersize=4, alpha=0.5, label='実験データ')
    ax1.plot(raman_shift, fit_params['fitted_curve'], 'r-', linewidth=2, label='フィッティング')
    ax1.plot(raman_shift, fit_params['crystal_peak'], 'b--', linewidth=2, label='結晶成分')
    ax1.plot(raman_shift, fit_params['amorphous_peak'], 'g--', linewidth=2, label='非晶成分')
    ax1.fill_between(raman_shift, fit_params['crystal_peak'], alpha=0.3, color='blue')
    ax1.fill_between(raman_shift, fit_params['amorphous_peak'], alpha=0.3, color='green')
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('強度 (a.u.)', fontsize=12)
    ax1.set_title('ピーク分離による結晶化度解析', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 結晶化度の表示
    ax2.bar(['結晶成分', '非晶成分'],
           [crystallinity, 100 - crystallinity],
           color=['#4ecdc4', '#ffe66d'], edgecolor='black', linewidth=2)
    ax2.set_ylabel('割合 (%)', fontsize=12)
    ax2.set_title(f'結晶化度: {crystallinity:.1f}%', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate(zip(['結晶成分', '非晶成分'],
                                            [crystallinity, 100 - crystallinity])):
        ax2.text(i, value + 3, f'{value:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('raman_crystallinity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 結果の出力
    print("=" * 60)
    print("Raman結晶化度解析結果")
    print("=" * 60)
    print(f"結晶ピーク中心: {fit_params['crystal_center']:.1f} cm⁻¹")
    print(f"結晶ピーク幅（FWHM）: {2.355 * fit_params['crystal_width']:.1f} cm⁻¹")
    print(f"非晶ピーク中心: {fit_params['amorphous_center']:.1f} cm⁻¹")
    print(f"非晶ピーク幅（FWHM）: {2.355 * fit_params['amorphous_width']:.1f} cm⁻¹")
    print(f"\n結晶化度: {crystallinity:.1f}%")
    print(f"非晶化度: {100 - crystallinity:.1f}%")
    

## 4\. 群論と振動選択則

### 4.1 分子の対称性と既約表現

分子の振動モードは、分子の対称性（点群）によって分類されます。各振動モードは点群の既約表現に対応し、その対称性から選択則（IR活性、Raman活性）が決まります。

**主要な点群と既約表現の例**  

  * **C 2v**（H₂O）: A₁, A₂, B₁, B₂
  * **D 6h**（ベンゼン）: A1g, A2g, B1g, B2g, E1g, E2g, A1u, A2u, B1u, B2u, E1u, E2u
  * **T d**（CH₄）: A₁, A₂, E, T₁, T₂

#### コード例7: H₂O分子の振動モードと対称性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def h2o_symmetry_analysis():
        """
        H₂O分子（C2v点群）の振動モードと選択則
    
        Returns:
        --------
        table : dict
            振動モードの情報
        """
        # H2O分子の3つの基本振動モード
        modes = {
            'mode1': {
                'name': '対称伸縮',
                'symmetry': 'A₁',
                'wavenumber': 3657,
                'IR_active': True,
                'Raman_active': True,
                'description': '両O-H結合が同時に伸縮、対称'
            },
            'mode2': {
                'name': '変角振動',
                'symmetry': 'A₁',
                'wavenumber': 1595,
                'IR_active': True,
                'Raman_active': True,
                'description': 'H-O-H角が変化'
            },
            'mode3': {
                'name': '非対称伸縮',
                'symmetry': 'B₁',
                'wavenumber': 3756,
                'IR_active': True,
                'Raman_active': True,
                'description': '一方のO-Hが伸びる時、他方が縮む'
            }
        }
    
        # 表形式で表示
        print("=" * 80)
        print("H₂O分子（C₂v点群）の振動モードと選択則")
        print("=" * 80)
        print(f"{'モード':<12} {'対称性':<10} {'波数 (cm⁻¹)':<15} {'IR活性':<10} {'Raman活性':<12} {'説明':<30}")
        print("-" * 80)
    
        for mode_id, mode in modes.items():
            ir_str = '○' if mode['IR_active'] else '×'
            raman_str = '○' if mode['Raman_active'] else '×'
            print(f"{mode['name']:<12} {mode['symmetry']:<10} {mode['wavenumber']:<15} "
                  f"{ir_str:<10} {raman_str:<12} {mode['description']:<30}")
    
        print("\n" + "=" * 80)
        print("C₂v点群の指標表（Character Table）")
        print("=" * 80)
        print("  C₂v | E   C₂  σv  σv' | 基底関数")
        print("-" * 80)
        print("  A₁  | 1   1   1   1   | z, x², y², z²")
        print("  A₂  | 1   1  -1  -1   | Rz")
        print("  B₁  | 1  -1   1  -1   | x, Ry")
        print("  B₂  | 1  -1  -1   1   | y, Rx")
        print("\n選択則:")
        print("  IR活性: μx, μy, μz（双極子モーメント）が基底に含まれる")
        print("  Raman活性: αxx, αyy, αzz, αxy, αxz, αyz（分極率テンソル）が基底に含まれる")
        print("\nH₂Oの場合、A₁とB₁はいずれもIRとRaman両方で活性")
    
        # 可視化：エネルギーレベル図
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # 基底状態と励起状態
        y_ground = 0
        modes_sorted = sorted(modes.items(), key=lambda x: x[1]['wavenumber'])
    
        colors = ['#f093fb', '#f5576c', '#4ecdc4']
    
        for i, (mode_id, mode) in enumerate(modes_sorted):
            y_excited = mode['wavenumber'] / 100  # スケーリング
            ax.hlines(y_excited, i*0.5, i*0.5 + 0.4, colors=colors[i], linewidths=5,
                     label=f"{mode['name']} ({mode['symmetry']})")
            ax.text(i*0.5 + 0.45, y_excited, f"{mode['wavenumber']} cm⁻¹",
                   va='center', fontsize=10)
    
            # 遷移矢印
            ax.annotate('', xy=(i*0.5 + 0.2, y_excited), xytext=(i*0.5 + 0.2, y_ground),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
        ax.hlines(y_ground, -0.2, 1.5, colors='black', linewidths=3, label='基底状態 (v=0)')
        ax.set_xlim(-0.3, 1.6)
        ax.set_ylim(-2, 40)
        ax.set_ylabel('相対エネルギー (cm⁻¹ / 100)', fontsize=12)
        ax.set_title('H₂O分子の振動励起エネルギー準位', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('h2o_symmetry_modes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        return modes
    
    # 実行
    modes = h2o_symmetry_analysis()
    

### 4.2 選択則の決定

既約表現に対応する振動モードがIR活性またはRaman活性であるかは、以下の規則で判定されます：

  * **IR活性** : 振動モードの既約表現が、双極子モーメント成分（x, y, z）のいずれかと同じ対称性を持つ
  * **Raman活性** : 振動モードの既約表現が、分極率テンソル成分（x², y², z², xy, xz, yz）のいずれかと同じ対称性を持つ

## 5\. 実践的なスペクトル解析

#### コード例8: IRとRamanの統合解析ワークフロー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    class VibrationalSpectroscopyAnalyzer:
        """IR・Ramanスペクトルの統合解析クラス"""
    
        def __init__(self):
            # 官能基データベース（簡略版）
            self.functional_groups = {
                'O-H': {'IR': (3200, 3600), 'Raman': (3200, 3600), 'intensity_IR': 'strong'},
                'N-H': {'IR': (3300, 3500), 'Raman': (3300, 3500), 'intensity_IR': 'medium'},
                'C-H': {'IR': (2850, 3000), 'Raman': (2850, 3000), 'intensity_IR': 'strong'},
                'C=O': {'IR': (1650, 1750), 'Raman': (1650, 1750), 'intensity_IR': 'very strong'},
                'C=C': {'IR': (1620, 1680), 'Raman': (1620, 1680), 'intensity_Raman': 'strong'},
                'C-C': {'IR': (800, 1200), 'Raman': (800, 1200), 'intensity_Raman': 'medium'},
            }
    
        def identify_functional_groups(self, wavenumbers, intensity, threshold=0.3):
            """
            スペクトルから官能基を同定
    
            Parameters:
            -----------
            wavenumbers : array
                波数（cm⁻¹）
            intensity : array
                強度
            threshold : float
                ピーク検出の閾値（最大値に対する相対値）
    
            Returns:
            --------
            identified_groups : list
                同定された官能基のリスト
            """
            # ピーク検出
            peaks, properties = find_peaks(intensity, prominence=threshold * np.max(intensity))
    
            identified_groups = []
    
            for peak in peaks:
                peak_wn = wavenumbers[peak]
    
                # 官能基データベースと照合
                for group, ranges in self.functional_groups.items():
                    ir_range = ranges['IR']
                    if ir_range[0] <= peak_wn <= ir_range[1]:
                        identified_groups.append({
                            'functional_group': group,
                            'wavenumber': peak_wn,
                            'intensity': intensity[peak]
                        })
    
            return identified_groups
    
        def complementary_analysis(self, ir_spectrum, raman_spectrum):
            """
            IRとRamanの相補的解析
    
            Parameters:
            -----------
            ir_spectrum : dict
                {'wavenumbers': array, 'intensity': array}
            raman_spectrum : dict
                {'wavenumbers': array, 'intensity': array}
    
            Returns:
            --------
            analysis_result : dict
                統合解析結果
            """
            # IRで検出された官能基
            ir_groups = self.identify_functional_groups(ir_spectrum['wavenumbers'],
                                                         ir_spectrum['intensity'])
    
            # Ramanで検出された官能基
            raman_groups = self.identify_functional_groups(raman_spectrum['wavenumbers'],
                                                            raman_spectrum['intensity'])
    
            # 統合
            all_groups = set([g['functional_group'] for g in ir_groups] +
                            [g['functional_group'] for g in raman_groups])
    
            analysis_result = {
                'IR_only': [g for g in ir_groups if g['functional_group'] not in
                           [rg['functional_group'] for rg in raman_groups]],
                'Raman_only': [g for g in raman_groups if g['functional_group'] not in
                              [ig['functional_group'] for ig in ir_groups]],
                'Both': list(all_groups.intersection(set([g['functional_group'] for g in ir_groups]),
                                                     set([g['functional_group'] for g in raman_groups])))
            }
    
            return analysis_result
    
    # 実行例：アセトン（CH₃COCH₃）のIR・Raman統合解析
    analyzer = VibrationalSpectroscopyAnalyzer()
    
    # 合成IRスペクトル
    wn_ir = np.linspace(4000, 400, 2000)
    ir_intensity = (
        1.5 * np.exp(-(wn_ir - 2970)**2 / (2 * 20**2)) +  # C-H伸縮
        2.0 * np.exp(-(wn_ir - 1715)**2 / (2 * 30**2)) +  # C=O伸縮
        0.5 * np.exp(-(wn_ir - 1360)**2 / (2 * 25**2)) +  # C-H変角
        0.3 * np.random.random(len(wn_ir))  # ノイズ
    )
    
    # 合成Ramanスペクトル
    wn_raman = np.linspace(3500, 100, 2000)
    raman_intensity = (
        0.8 * np.exp(-(wn_raman - 2970)**2 / (2 * 20**2)) +  # C-H伸縮
        1.2 * np.exp(-(wn_raman - 1715)**2 / (2 * 30**2)) +  # C=O伸縮
        1.5 * np.exp(-(wn_raman - 900)**2 / (2 * 25**2)) +   # C-C伸縮
        0.2 * np.random.random(len(wn_raman))  # ノイズ
    )
    
    # 統合解析
    ir_spec = {'wavenumbers': wn_ir, 'intensity': ir_intensity}
    raman_spec = {'wavenumbers': wn_raman, 'intensity': raman_intensity}
    
    result = analyzer.complementary_analysis(ir_spec, raman_spec)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # IRスペクトル
    ax1.plot(wn_ir, ir_intensity, linewidth=1.5, color='#f093fb', label='IRスペクトル')
    ax1.fill_between(wn_ir, ir_intensity, alpha=0.3, color='#f5576c')
    ax1.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('吸光度 (a.u.)', fontsize=12)
    ax1.set_title('アセトンのIRスペクトル', fontsize=14, fontweight='bold')
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ramanスペクトル
    ax2.plot(wn_raman, raman_intensity, linewidth=1.5, color='#4ecdc4', label='Ramanスペクトル')
    ax2.fill_between(wn_raman, raman_intensity, alpha=0.3, color='#ffe66d')
    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('アセトンのRamanスペクトル', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acetone_ir_raman_complementary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 解析結果の表示
    print("=" * 70)
    print("IRとRamanの統合解析結果（アセトン）")
    print("=" * 70)
    print("\nIRのみで検出:")
    for group in result['IR_only']:
        print(f"  {group['functional_group']}: {group['wavenumber']:.0f} cm⁻¹")
    
    print("\nRamanのみで検出:")
    for group in result['Raman_only']:
        print(f"  {group['functional_group']}: {group['wavenumber']:.0f} cm⁻¹")
    
    print("\n両方で検出:")
    for group in result['Both']:
        print(f"  {group}")
    
    print("\n結論:")
    print("  - C=O伸縮: IRで非常に強い、Ramanでも観測")
    print("  - C-H伸縮: IR・Raman両方で強い")
    print("  - C-C伸縮: Ramanで強く観測（IRでは弱い）")
    print("  → IRとRamanを組み合わせることで、分子構造の全体像を把握")
    

## 演習問題

**演習問題（クリックして展開）**

### Easy レベル（基本計算）

**問題1** : C-O結合（力の定数 $k = 1000$ N/m）の振動周波数（cm⁻¹）を計算してください。炭素の質量は12 amu、酸素の質量は16 amuです。

解答を見る

**解答** :
    
    
    # コード例1の関数を使用
    freq_Hz, wavenumber = vibrational_frequency(k=1000, m1=12, m2=16)
    print(f"C-O伸縮振動の波数: {wavenumber:.1f} cm⁻¹")
    # 出力: 約1270 cm⁻¹
    

**問題2** : H₂O分子（非線形、3原子）の振動自由度の数を求めてください。

解答を見る

**解答** :

$$3N - 6 = 3 \times 3 - 6 = 3$$

H₂Oは3つの振動モード（対称伸縮、変角振動、非対称伸縮）を持ちます。

**問題3** : IRスペクトルで1715 cm⁻¹に強いピークが観測されました。この官能基は何ですか？

解答を見る

**解答** : カルボニル基（C=O）の伸縮振動。1650-1750 cm⁻¹の領域はC=Oの特性吸収です。

### Medium レベル（実践的計算）

**問題4** : 同位体効果を考慮し、¹²C=O と ¹³C=O の振動周波数の比を計算してください（力の定数は同じと仮定）。

解答を見る

**解答** :
    
    
    _, wn_12C = vibrational_frequency(1200, 12, 16)
    _, wn_13C = vibrational_frequency(1200, 13, 16)
    
    ratio = wn_12C / wn_13C
    print(f"¹²C=O波数: {wn_12C:.1f} cm⁻¹")
    print(f"¹³C=O波数: {wn_13C:.1f} cm⁻¹")
    print(f"比率: {ratio:.4f}")
    # 出力: 比率 ≈ 1.017（約1.7%のシフト）
    

**問題5** : Raman散乱において、室温（300 K）でStokes線とAnti-Stokes線の強度比を計算してください。振動モードは1500 cm⁻¹とします。

解答を見る

**解答** :
    
    
    # コード例5のboltzmann_population関数を使用
    ratio = boltzmann_population(1500, T=300)
    print(f"I(Anti-Stokes) / I(Stokes) = {ratio:.4f}")
    # 出力: 約0.023（Anti-StokesはStokesの約2.3%）
    

**問題6** : CO₂分子（線形、3原子）の振動自由度を求め、各振動モードの対称性（Σg⁺, Σu⁺, Πu）とIR/Raman活性を答えてください。

解答を見る

**解答** :

$$3N - 5 = 3 \times 3 - 5 = 4$$

モード| 対称性| 波数（cm⁻¹）| IR活性| Raman活性  
---|---|---|---|---  
対称伸縮| Σg⁺| 1340| 不活性| 活性  
非対称伸縮| Σu⁺| 2349| 活性| 不活性  
変角振動（2重縮退）| Πu| 667| 活性| 不活性  
  
CO₂は中心対称分子なので、相互排他則が成立します。

### Hard レベル（高度な解析）

**問題7** : 下記の実測IRスペクトルデータから、FTIRのインターフェログラムを逆算してください。その後、フーリエ変換で元のスペクトルが復元されることを確認してください。

解答を見る（完全なコード）
    
    
    # コード例4の関数を利用
    true_wavenumbers = np.array([1000, 1500, 2000, 2900])
    true_intensities = np.array([0.6, 1.0, 0.8, 0.7])
    
    # インターフェログラム生成
    displacement, interferogram = generate_interferogram(true_wavenumbers, true_intensities,
                                                         mirror_displacement_max=0.1)
    
    # フーリエ変換でスペクトル復元
    wavenumbers_ft, spectrum_ft = fourier_transform_spectrum(displacement, interferogram)
    
    # 検証プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(displacement, interferogram, linewidth=1.5, color='#f093fb')
    ax1.set_xlabel('ミラー変位 (cm)', fontsize=12)
    ax1.set_ylabel('干渉強度', fontsize=12)
    ax1.set_title('実測IRスペクトルから逆算したインターフェログラム', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wavenumbers_ft, spectrum_ft, linewidth=1.5, color='#f5576c')
    for wn in true_wavenumbers:
        ax2.axvline(x=wn, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('波数 (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('強度 (a.u.)', fontsize=12)
    ax2.set_title('フーリエ変換で復元されたスペクトル', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 4000)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("真のピーク位置:", true_wavenumbers)
    print("復元成功: フーリエ変換により元のスペクトルが再現されました")
    

**問題8** : ベンゼン（D6h点群）の振動モード（30個）のうち、IR活性なモードとRaman活性なモードの数を推定してください。対称性に基づいて考察してください。

解答を見る

**解答** :

ベンゼンは中心対称分子（D6h）なので、相互排他則が成立します。30個の振動モードのうち：

  * **IR活性** : u（ungerade）対称性のモード → 約4個のE1uモード
  * **Raman活性** : g（gerade）対称性のモード → 約7個のA1g, E1g, E2gモード
  * 残りのモードは不活性（A2g, A2u, B1u, B2uなど）

詳細な指標表による解析が必要ですが、IR活性モードとRaman活性モードは互いに排他的です。

**問題9** : ポリエチレンの結晶化度をRamanスペクトルから評価する際、結晶ピーク（1130 cm⁻¹）と非晶ピーク（1080 cm⁻¹）の強度比が2:1でした。ピーク面積比から結晶化度を計算してください（ピーク幅はそれぞれ10 cm⁻¹、15 cm⁻¹とします）。

解答を見る

**解答** :
    
    
    # Gaussian近似でピーク面積を計算
    I_crystal = 2.0  # 強度比
    I_amorphous = 1.0
    width_crystal = 10  # cm^-1
    width_amorphous = 15  # cm^-1
    
    # 面積 = 強度 × 幅 × sqrt(2π)（Gaussian近似）
    area_crystal = I_crystal * width_crystal * np.sqrt(2 * np.pi)
    area_amorphous = I_amorphous * width_amorphous * np.sqrt(2 * np.pi)
    
    crystallinity = area_crystal / (area_crystal + area_amorphous) * 100
    print(f"結晶ピーク面積: {area_crystal:.1f}")
    print(f"非晶ピーク面積: {area_amorphous:.1f}")
    print(f"結晶化度: {crystallinity:.1f}%")
    # 出力: 約51.3%
    

**問題10** : H₂O分子のC2v点群における3つの振動モード（A₁, A₁, B₁）が、なぜすべてIRとRaman両方で活性なのかを、指標表を用いて説明してください。

解答を見る

**解答** :

C2v点群の指標表より：

  * **A₁対称性** : 基底関数にz（双極子モーメント）とx², y², z²（分極率テンソル）が含まれる → IR活性かつRaman活性
  * **B₁対称性** : 基底関数にx（双極子モーメント）とxz（分極率テンソル）が含まれる → IR活性かつRaman活性

したがって、H₂Oの3つの振動モード（対称伸縮A₁、変角A₁、非対称伸縮B₁）はすべてIRとRaman両方で観測されます。中心対称分子ではないため、相互排他則は適用されません。

## 学習目標の確認

この章で学んだ内容を振り返り、以下の項目を確認してください。

### 基本理解

  * ✅ 調和振動子モデルと振動周波数の質量・力の定数依存性を説明できる
  * ✅ IRとRamanの選択則の違い（双極子モーメント変化 vs 分極率変化）を理解している
  * ✅ 主要な官能基の特性吸収波数を暗記している（C=O: 1700 cm⁻¹、O-H: 3400 cm⁻¹など）
  * ✅ 中心対称分子における相互排他則を説明できる

### 実践スキル

  * ✅ 振動周波数の計算と同位体効果の評価ができる
  * ✅ IRスペクトルから官能基を同定できる
  * ✅ RamanスペクトルからStokes/Anti-Stokes強度比を用いて温度推定ができる
  * ✅ 結晶化度評価のためのピーク分離ができる

### 応用力

  * ✅ IRとRamanの相補的情報を組み合わせて分子構造を決定できる
  * ✅ 群論を用いて振動モードの対称性とIR/Raman活性を判定できる
  * ✅ FTIRの原理（インターフェログラムとフーリエ変換）を理解し、実装できる

## 参考文献

  1. Raman, C. V., Krishnan, K. S. (1928). A new type of secondary radiation. _Nature_ , 121(3048), 501-502. DOI: 10.1038/121501c0 - Raman散乱効果の発見を報告した歴史的原著論文
  2. Nakamoto, K. (2008). _Infrared and Raman Spectra of Inorganic and Coordination Compounds_ (6th ed.). Wiley, pp. 25-31 (IR theory), pp. 78-95 (Raman theory), pp. 115-140 (group theory applications). - IR・Ramanスペクトルの包括的教科書と官能基帰属表
  3. Long, D. A. (2002). _The Raman Effect: A Unified Treatment of the Theory of Raman Scattering by Molecules_. Wiley, pp. 50-68 (classical theory), pp. 95-115 (quantum theory), pp. 145-160 (selection rules). - Raman散乱の量子力学的理論と選択則
  4. Wilson, E. B., Decius, J. C., Cross, P. C. (1980). _Molecular Vibrations: The Theory of Infrared and Raman Vibrational Spectra_. Dover Publications, pp. 25-42 (normal modes), pp. 65-85 (group theory), pp. 95-110 (selection rules). - 分子振動の古典的名著、群論と選択則
  5. Colthup, N. B., Daly, L. H., Wiberley, S. E. (1990). _Introduction to Infrared and Raman Spectroscopy_ (3rd ed.). Academic Press, pp. 100-125 (functional group frequencies), pp. 180-210 (spectral interpretation), pp. 220-240 (peak assignment). - 実践的なスペクトル解釈とピーク帰属
  6. Savitzky, A., Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. _Analytical Chemistry_ , 36(8), 1627-1639. DOI: 10.1021/ac60214a047 - Savitzky-Golay平滑化フィルタの原著論文（コード例で使用）
  7. SciPy 1.11 Signal Processing Documentation. _scipy.signal.find_peaks, scipy.signal.savgol_filter_. Available at: https://docs.scipy.org/doc/scipy/reference/signal.html - Pythonによるスペクトルデータ処理
  8. Smith, E., Dent, G. (2019). _Modern Raman Spectroscopy: A Practical Approach_ (2nd ed.). Wiley, pp. 15-28 (instrumentation), pp. 45-65 (sampling techniques), pp. 72-80 (data processing). - 現代Raman分光法の実践的技術
  9. Cotton, F. A. (1990). _Chemical Applications of Group Theory_ (3rd ed.). Wiley, pp. 250-275 (point groups), pp. 285-305 (vibrational modes), pp. 310-320 (selection rules). - 群論の化学応用と振動スペクトルの対称性解析

## 次のステップ

第2章では、赤外・ラマン分光法の原理、選択則、官能基同定、群論による対称性解析を学びました。調和振動子モデル、FTIRの原理、Stokes/Anti-Stokes散乱、結晶化度評価など、実践的なデータ解析スキルも習得しました。

**第3章** では、UV-Vis（紫外可視）分光法を学びます。電子遷移、Lambert-Beer則の応用、Tauc plotによるバンドギャップ測定、配位子場理論、Python実践による吸収スペクトル解析など、半導体・有機材料の電子状態解析の全てをカバーします。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
