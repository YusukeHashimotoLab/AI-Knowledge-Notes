---
title: 第3章：赤外分光法
chapter_title: 第3章：赤外分光法
subtitle: 分子振動から官能基を同定し、材料の化学構造を解明する
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/XF1VYwWRuMk"
    title="分光分析入門 第3章: 赤外分光法"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

**この章で学ぶこと**

赤外分光法（Infrared Spectroscopy, IR）は、分子の振動エネルギーに対応する赤外光の吸収を測定することで、化学結合や官能基の情報を得る分析手法です。本章では、分子振動の基礎理論から、FTIR（フーリエ変換赤外分光法）の原理、官能基同定の実践的手法、ATR-FTIRによる表面分析まで、IRスペクトル解析に必要な知識を体系的に学びます。Pythonを用いたスペクトルデータ解析の実装も行います。

## 3.1 分子振動の基礎

### 3.1.1 調和振動子モデル

分子振動を理解するための最も基本的なモデルは、2原子分子の調和振動子近似です。このモデルでは、2つの原子がバネでつながれた系として扱います。

**調和振動子の振動周波数：**

$$\nu = \frac{1}{2\pi}\sqrt{\frac{k}{\mu}}$$ 

ここで、$k$ は力の定数（結合の強さを表す、単位：N/m）、$\mu$ は換算質量です。

**換算質量：**

$$\mu = \frac{m_1 \cdot m_2}{m_1 + m_2}$$ 

$m_1$, $m_2$ は2つの原子の質量です。

赤外分光法では、振動周波数を波数（wavenumber）$\tilde{\nu}$（単位：cm$^{-1}$）で表すことが一般的です：

$$\tilde{\nu} = \frac{\nu}{c} = \frac{1}{2\pi c}\sqrt{\frac{k}{\mu}}$$ 

この式から、振動周波数は以下の特性を持つことがわかります：

  * **力の定数 $k$ が大きい（強い結合）** ほど、振動周波数が高い（例：C$\equiv$C > C=C > C-C）
  * **換算質量 $\mu$ が小さい** ほど、振動周波数が高い（例：C-H > C-D > C-C）

#### コード例1: 振動周波数の計算と同位体効果
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    c = 2.998e10  # 光速 (cm/s)
    u = 1.6605e-27  # 原子質量単位 (kg)
    
    def calculate_wavenumber(k, m1, m2):
        """
        2原子分子の振動波数を計算
    
        Parameters:
        -----------
        k : float
            力の定数 (N/m)
        m1, m2 : float
            原子質量 (amu)
    
        Returns:
        --------
        wavenumber : float
            振動波数 (cm^-1)
        """
        # 換算質量 (kg)
        mu = (m1 * m2) / (m1 + m2) * u
    
        # 振動周波数 (Hz)
        nu = (1 / (2 * np.pi)) * np.sqrt(k / mu)
    
        # 波数に変換 (cm^-1)
        wavenumber = nu / c
    
        return wavenumber
    
    # 代表的な化学結合の振動波数
    bonds = {
        'C-H': {'k': 500, 'm1': 12, 'm2': 1},
        'C-D': {'k': 500, 'm1': 12, 'm2': 2},  # 重水素同位体
        'O-H': {'k': 750, 'm1': 16, 'm2': 1},
        'C=O': {'k': 1200, 'm1': 12, 'm2': 16},
        'C-C': {'k': 400, 'm1': 12, 'm2': 12},
        'C=C': {'k': 900, 'm1': 12, 'm2': 12},
    }
    
    print("=" * 60)
    print("代表的な化学結合の振動波数")
    print("=" * 60)
    print(f"{'結合':<10} {'力の定数 (N/m)':<18} {'波数 (cm^-1)':<15}")
    print("-" * 60)
    
    for bond, params in bonds.items():
        wn = calculate_wavenumber(params['k'], params['m1'], params['m2'])
        print(f"{bond:<10} {params['k']:<18} {wn:.0f}")
    
    # 同位体効果の可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # C-H vs C-D の比較
    wn_CH = calculate_wavenumber(500, 12, 1)
    wn_CD = calculate_wavenumber(500, 12, 2)
    isotope_ratio = wn_CH / wn_CD
    
    bars = ax.bar(['C-H', 'C-D'], [wn_CH, wn_CD],
                  color=['#f093fb', '#f5576c'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('波数 (cm$^{-1}$)', fontsize=12)
    ax.set_title('同位体効果：C-H vs C-D 振動', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 3500)
    
    for bar, val in zip(bars, [wn_CH, wn_CD]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(0.5, 0.85, f'比率: {isotope_ratio:.3f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('isotope_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n同位体効果: C-H/C-D 波数比 = {isotope_ratio:.3f}")
    print(f"理論値 (sqrt(2)) = {np.sqrt(2):.3f}")

### 3.1.2 振動モードの種類

$N$ 個の原子からなる分子は $3N$ 個の自由度を持ちます。そのうち、並進（3）と回転（非線形分子：3、線形分子：2）を除いた残りが振動の自由度です：

  * **非線形分子** ：$3N - 6$ 個の振動モード
  * **線形分子** ：$3N - 5$ 個の振動モード

振動モードは大きく2種類に分類されます：

#### 伸縮振動（Stretching Vibration）

結合長が変化する振動です。

  * **対称伸縮振動（$\nu_s$）** ：複数の結合が同時に伸び縮みする
  * **非対称伸縮振動（$\nu_{as}$）** ：一方の結合が伸びるとき、他方が縮む

#### 変角振動（Bending Vibration）

結合角が変化する振動です。伸縮振動より低い周波数に現れます。

  * **はさみ振動（Scissoring, $\delta$）** ：面内で結合角が変化
  * **横揺れ振動（Rocking, $\rho$）** ：面内で原子が同方向に揺れる
  * **縦揺れ振動（Wagging, $\omega$）** ：面外で原子が同方向に動く
  * **ねじれ振動（Twisting, $\tau$）** ：面外で原子が逆方向に動く

    
    
    ```mermaid
    flowchart TD
                A[分子振動 3N-6 モード] --> B[伸縮振動]
                A --> C[変角振動]
    
                B --> B1[対称伸縮 νs]
                B --> B2[非対称伸縮 νas]
    
                C --> C1[面内変角]
                C --> C2[面外変角]
    
                C1 --> D1[はさみ δ]
                C1 --> D2[横揺れ ρ]
                C2 --> D3[縦揺れ ω]
                C2 --> D4[ねじれ τ]
    
                style A fill:#f093fb,color:#fff
                style B fill:#f5576c,color:#fff
                style C fill:#f5576c,color:#fff
    ```

#### コード例2: H2O分子の振動モードの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def visualize_h2o_modes():
        """
        H2O分子の3つの振動モードを可視化
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
        # 平衡位置（O原子を原点）
        O = np.array([0, 0])
        H1 = np.array([-0.76, 0.59])  # 左のH
        H2 = np.array([0.76, 0.59])   # 右のH
    
        modes = [
            {
                'name': '対称伸縮振動 (νs)',
                'freq': '3657 cm$^{-1}$',
                'disp_H1': np.array([-0.15, 0.12]),
                'disp_H2': np.array([0.15, 0.12]),
                'ir_active': True,
                'description': '両O-H結合が同時に伸縮'
            },
            {
                'name': '変角振動 (δ)',
                'freq': '1595 cm$^{-1}$',
                'disp_H1': np.array([0.08, -0.12]),
                'disp_H2': np.array([-0.08, -0.12]),
                'ir_active': True,
                'description': 'H-O-H角が変化'
            },
            {
                'name': '非対称伸縮振動 (νas)',
                'freq': '3756 cm$^{-1}$',
                'disp_H1': np.array([-0.15, 0.12]),
                'disp_H2': np.array([-0.15, -0.12]),
                'ir_active': True,
                'description': '一方が伸びると他方が縮む'
            }
        ]
    
        for ax, mode in zip(axes, modes):
            # 平衡位置の分子を描画
            ax.plot(*O, 'ro', markersize=20, label='O', zorder=5)
            ax.plot(*H1, 'bo', markersize=12, label='H', zorder=5)
            ax.plot(*H2, 'bo', markersize=12, zorder=5)
            ax.plot([O[0], H1[0]], [O[1], H1[1]], 'k-', linewidth=3)
            ax.plot([O[0], H2[0]], [O[1], H2[1]], 'k-', linewidth=3)
    
            # 変位ベクトルを矢印で表示
            scale = 1.5
            ax.annotate('', xy=H1 + scale*mode['disp_H1'], xytext=H1,
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.annotate('', xy=H2 + scale*mode['disp_H2'], xytext=H2,
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.8, 1.2)
            ax.set_aspect('equal')
            ax.set_title(f"{mode['name']}\n{mode['freq']}", fontsize=11, fontweight='bold')
            ax.text(0, -0.6, mode['description'], ha='center', fontsize=9)
            ax.axis('off')
    
        plt.suptitle('H$_2$O分子の基準振動モード（緑矢印：変位方向）',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('h2o_vibration_modes.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        # 振動モードの特徴を表示
        print("=" * 60)
        print("H2O分子の基準振動モード（3原子 → 3x3-6 = 3モード）")
        print("=" * 60)
        for mode in modes:
            ir_status = "IR活性" if mode['ir_active'] else "IR不活性"
            print(f"\n{mode['name']}: {mode['freq']}")
            print(f"  {mode['description']}")
            print(f"  {ir_status}")
    
    visualize_h2o_modes()

## 3.2 赤外選択則

### 3.2.1 双極子モーメントの変化

振動がIR活性（赤外光を吸収する）であるためには、その振動に伴って**分子の双極子モーメント $\boldsymbol{\mu}$ が変化する** 必要があります：

**赤外選択則：**

$$\left(\frac{\partial \boldsymbol{\mu}}{\partial Q}\right)_{Q=0} \neq 0$$ 

ここで、$Q$ は基準座標（振動の変位）です。

この選択則から、以下の重要な結論が導かれます：

  * **極性結合の振動** （O-H, N-H, C=O など）は一般にIR活性
  * **対称分子の対称伸縮振動** （CO$_2$の対称伸縮、O=C=Oなど）はIR不活性
  * **ホモ核二原子分子** （N$_2$, O$_2$, H$_2$など）は双極子モーメントを持たないためIR不活性

**IR活性とRaman活性の相補性**

分子が対称中心を持つ場合、IR活性な振動はRaman不活性、Raman活性な振動はIR不活性となります（相互排他則）。そのため、IRとRaman分光法を組み合わせることで、分子構造のより完全な情報が得られます。

#### コード例3: CO2分子のIR活性・不活性振動
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def visualize_co2_modes():
        """
        CO2分子の振動モードとIR活性を可視化
        線形3原子分子：3x3-5 = 4振動モード
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    
        # 平衡位置
        O1 = np.array([-1.2, 0])
        C = np.array([0, 0])
        O2 = np.array([1.2, 0])
    
        modes = [
            {
                'name': '対称伸縮振動 (νs)',
                'freq': '1388 cm$^{-1}$',
                'disp': {'O1': np.array([-0.2, 0]), 'C': np.array([0, 0]), 'O2': np.array([0.2, 0])},
                'ir_active': False,
                'raman_active': True,
                'description': '双極子モーメント変化なし → IR不活性'
            },
            {
                'name': '非対称伸縮振動 (νas)',
                'freq': '2349 cm$^{-1}$',
                'disp': {'O1': np.array([-0.15, 0]), 'C': np.array([0.1, 0]), 'O2': np.array([0.15, 0])},
                'ir_active': True,
                'raman_active': False,
                'description': '双極子モーメント変化あり → IR活性'
            },
            {
                'name': '変角振動 (δ) - 面内',
                'freq': '667 cm$^{-1}$',
                'disp': {'O1': np.array([0, 0.15]), 'C': np.array([0, -0.1]), 'O2': np.array([0, 0.15])},
                'ir_active': True,
                'raman_active': False,
                'description': '2重縮退、双極子モーメント変化あり'
            },
            {
                'name': '変角振動 (δ) - 面外',
                'freq': '667 cm$^{-1}$',
                'disp': {'O1': np.array([0, -0.15]), 'C': np.array([0, 0.1]), 'O2': np.array([0, 0.15])},
                'ir_active': True,
                'raman_active': False,
                'description': '2重縮退、双極子モーメント変化あり'
            }
        ]
    
        for ax, mode in zip(axes, modes):
            # 分子を描画
            ax.plot(*O1, 'ro', markersize=18, zorder=5)
            ax.plot(*C, 'ko', markersize=15, zorder=5)
            ax.plot(*O2, 'ro', markersize=18, zorder=5)
            ax.plot([O1[0], C[0]], [O1[1], C[1]], 'k-', linewidth=4)
            ax.plot([C[0], O2[0]], [C[1], O2[1]], 'k=', linewidth=4)
    
            # 変位ベクトル
            scale = 1.5
            for atom, pos in [('O1', O1), ('C', C), ('O2', O2)]:
                disp = mode['disp'][atom]
                if np.linalg.norm(disp) > 0.01:
                    ax.annotate('', xy=pos + scale*disp, xytext=pos,
                               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
            # 状態表示
            ir_color = '#4caf50' if mode['ir_active'] else '#f44336'
            raman_color = '#4caf50' if mode['raman_active'] else '#f44336'
            ir_text = 'IR活性' if mode['ir_active'] else 'IR不活性'
            raman_text = 'Raman活性' if mode['raman_active'] else 'Raman不活性'
    
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_title(f"{mode['name']}\n{mode['freq']}", fontsize=11, fontweight='bold')
    
            # 活性情報
            ax.text(0, -0.5, f"{ir_text} / {raman_text}", ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0, -0.75, mode['description'], ha='center', fontsize=8, style='italic')
            ax.axis('off')
    
        plt.suptitle('CO$_2$分子の振動モードと選択則', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('co2_vibration_modes.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    visualize_co2_modes()
    
    # 選択則のまとめ
    print("=" * 60)
    print("CO2分子の振動モードと選択則")
    print("=" * 60)
    print("対称中心を持つ分子では、IR活性とRaman活性は相互排他的")
    print("")
    print("νs (1388 cm-1): IR不活性, Raman活性")
    print("  → 対称振動で双極子モーメント変化なし")
    print("")
    print("νas (2349 cm-1): IR活性, Raman不活性")
    print("  → CO2の最も強いIR吸収")
    print("")
    print("δ (667 cm-1): IR活性, Raman不活性")
    print("  → 2重縮退（面内・面外で同じ周波数）")

## 3.3 FTIR（フーリエ変換赤外分光法）

### 3.3.1 マイケルソン干渉計の原理

現代のIR分光計の主流であるFTIR（Fourier Transform Infrared Spectroscopy）は、マイケルソン干渉計を用いて干渉パターン（インターフェログラム）を測定し、フーリエ変換によってスペクトルを得る手法です。
    
    
    ```mermaid
    flowchart LR
                A[IR光源] --> B[ビームスプリッター]
                B --> C[固定ミラー]
                B --> D[可動ミラー]
                C --> B
                D --> B
                B --> E[試料]
                E --> F[検出器]
                F --> G[インターフェログラム]
                G --> H[フーリエ変換]
                H --> I[IRスペクトル]
    
                style A fill:#ff9800,color:#fff
                style H fill:#f093fb,color:#fff
                style I fill:#4caf50,color:#fff
    ```

干渉計で得られるインターフェログラム $I(\delta)$ は、光路差 $\delta$ の関数として記録されます：

**インターフェログラム：**

$$I(\delta) = \int_0^\infty B(\tilde{\nu}) \cos(2\pi\tilde{\nu}\delta) \, d\tilde{\nu}$$ 

**フーリエ変換によるスペクトル復元：**

$$B(\tilde{\nu}) = \int_{-\infty}^\infty I(\delta) \cos(2\pi\tilde{\nu}\delta) \, d\delta$$ 

ここで、$B(\tilde{\nu})$ はスペクトル強度、$\tilde{\nu}$ は波数、$\delta$ は光路差です。

**FTIRの利点**

  * **Fellgettの利点（多重化）** ：全波数を同時測定できるため、測定時間が短縮される
  * **Jacquinotの利点（高スループット）** ：光エネルギーの利用効率が高く、S/N比が向上
  * **Connesの利点（高波数精度）** ：He-Neレーザーによる内部校正で高い波数精度
  * **積算によるS/N向上** ：短時間で多数回の測定・積算が可能

#### コード例4: FTIRのインターフェログラムとフーリエ変換
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    def generate_interferogram(peaks, delta_max=0.1, n_points=4096):
        """
        IRスペクトルからインターフェログラムを生成
    
        Parameters:
        -----------
        peaks : list of tuples
            (波数 cm^-1, 強度) のリスト
        delta_max : float
            光路差の最大値 (cm)
        n_points : int
            データ点数
    
        Returns:
        --------
        delta : array
            光路差 (cm)
        interferogram : array
            インターフェログラム強度
        """
        delta = np.linspace(0, delta_max, n_points)
        interferogram = np.zeros_like(delta)
    
        for wavenumber, intensity in peaks:
            # 各周波数成分の寄与を加算
            interferogram += intensity * np.cos(2 * np.pi * wavenumber * delta)
    
        return delta, interferogram
    
    def compute_spectrum(delta, interferogram):
        """
        インターフェログラムをフーリエ変換してスペクトルを復元
    
        Returns:
        --------
        wavenumbers : array
            波数 (cm^-1)
        spectrum : array
            スペクトル強度
        """
        N = len(interferogram)
        d_delta = delta[1] - delta[0]
    
        # フーリエ変換
        spectrum_complex = fft(interferogram)
        spectrum = np.abs(spectrum_complex[:N//2])
    
        # 波数軸
        wavenumbers = fftfreq(N, d_delta)[:N//2]
    
        return wavenumbers, spectrum
    
    # サンプルスペクトル（CO2 + H2O の模擬データ）
    peaks = [
        (667, 0.8),    # CO2 変角
        (1595, 0.3),   # H2O 変角
        (2349, 1.0),   # CO2 非対称伸縮
        (3657, 0.5),   # H2O 対称伸縮
        (3756, 0.6),   # H2O 非対称伸縮
    ]
    
    # インターフェログラム生成
    delta, interferogram = generate_interferogram(peaks, delta_max=0.2, n_points=8192)
    
    # フーリエ変換でスペクトル復元
    wavenumbers, spectrum = compute_spectrum(delta, interferogram)
    
    # プロット
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # インターフェログラム
    ax1 = axes[0]
    ax1.plot(delta * 10000, interferogram, color='#f093fb', linewidth=0.8)
    ax1.set_xlabel('光路差 ($\\mu$m)', fontsize=12)
    ax1.set_ylabel('強度', fontsize=12)
    ax1.set_title('インターフェログラム（時間領域）', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 500)
    ax1.grid(True, alpha=0.3)
    
    # 復元スペクトル
    ax2 = axes[1]
    ax2.plot(wavenumbers, spectrum, color='#f5576c', linewidth=1.2)
    ax2.set_xlabel('波数 (cm$^{-1}$)', fontsize=12)
    ax2.set_ylabel('強度', fontsize=12)
    ax2.set_title('復元IRスペクトル（周波数領域）', fontsize=14, fontweight='bold')
    ax2.set_xlim(400, 4000)
    
    # ピーク位置をマーク
    for wn, intensity in peaks:
        idx = np.argmin(np.abs(wavenumbers - wn))
        if wavenumbers[idx] < 4000:
            ax2.axvline(x=wn, color='gray', linestyle='--', alpha=0.5)
            ax2.text(wn, spectrum[idx] * 1.1, f'{wn}', ha='center', fontsize=9)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ftir_principle.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("FTIRの原理デモンストレーション")
    print("=" * 50)
    print("入力ピーク位置:")
    for wn, intensity in peaks:
        print(f"  {wn} cm^-1 (強度: {intensity})")
    print("\nフーリエ変換により周波数領域のスペクトルが復元されます")

## 3.4 官能基同定

### 3.4.1 特性吸収帯

有機化合物の官能基は、特定の波数領域に特徴的な吸収を示します。この「特性吸収帯」を利用して、IRスペクトルから分子構造を推定できます。

官能基 | 振動モード | 波数 (cm$^{-1}$) | 強度・特徴  
---|---|---|---  
O-H（アルコール） | 伸縮 | 3200-3600 | 強、幅広（水素結合）  
O-H（カルボン酸） | 伸縮 | 2500-3300 | 非常に幅広  
N-H | 伸縮 | 3300-3500 | 中、1級アミンは2本  
C-H（脂肪族） | 伸縮 | 2850-2960 | 強  
C-H（芳香族） | 伸縮 | 3000-3100 | 弱-中  
C$\equiv$N | 伸縮 | 2210-2260 | 強  
C$\equiv$C | 伸縮 | 2100-2260 | 弱（対称なら不活性）  
C=O（アルデヒド） | 伸縮 | 1720-1740 | 非常に強  
C=O（ケトン） | 伸縮 | 1705-1725 | 非常に強  
C=O（エステル） | 伸縮 | 1735-1750 | 非常に強  
C=O（アミド） | 伸縮 | 1630-1680 | 強（アミドI帯）  
C=C（アルケン） | 伸縮 | 1620-1680 | 弱-中  
芳香環 C=C | 伸縮 | 1450-1600 | 中（複数ピーク）  
C-O | 伸縮 | 1000-1300 | 強  
芳香環 | 面外変角 | 675-900 | 強（置換パターン判定）  
  
**IRスペクトル解析のコツ**

  * まず**4000-2500 cm$^{-1}$** （X-H伸縮領域）を確認：O-H, N-H, C-Hの存在
  * 次に**2500-1500 cm$^{-1}$** （三重結合、二重結合領域）：C$\equiv$N, C=O, C=Cなど
  * **1500-400 cm$^{-1}$** （指紋領域）：分子固有のパターン、構造異性体の区別に有用

#### コード例5: IRスペクトルシミュレーションと官能基同定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lorentzian(x, center, intensity, width):
        """ローレンツ型線形関数"""
        return intensity * (width**2 / ((x - center)**2 + width**2))
    
    def gaussian(x, center, intensity, width):
        """ガウス型線形関数（水素結合したO-Hなど幅広いピーク用）"""
        return intensity * np.exp(-((x - center)**2) / (2 * width**2))
    
    def simulate_ir_spectrum(compound_name, peaks, x_range=(4000, 400)):
        """
        化合物のIRスペクトルをシミュレーション
    
        Parameters:
        -----------
        compound_name : str
            化合物名
        peaks : list of dict
            ピーク情報 [{'center': cm-1, 'intensity': 0-1, 'width': cm-1,
                        'shape': 'lorentzian'/'gaussian', 'label': str}, ...]
    
        Returns:
        --------
        wavenumbers : array
            波数軸
        transmittance : array
            透過率 (%)
        """
        wavenumbers = np.linspace(x_range[0], x_range[1], 2000)
        absorbance = np.zeros_like(wavenumbers)
    
        for peak in peaks:
            shape = peak.get('shape', 'lorentzian')
            if shape == 'gaussian':
                absorbance += gaussian(wavenumbers, peak['center'],
                                      peak['intensity'], peak['width'])
            else:
                absorbance += lorentzian(wavenumbers, peak['center'],
                                        peak['intensity'], peak['width'])
    
        # 透過率に変換
        transmittance = 100 * np.exp(-absorbance)
    
        return wavenumbers, transmittance
    
    # 酢酸エチル (CH3COOCH2CH3) のIRスペクトル
    ethyl_acetate_peaks = [
        {'center': 2985, 'intensity': 0.5, 'width': 30, 'label': 'C-H伸縮 (CH3)'},
        {'center': 2940, 'intensity': 0.4, 'width': 25, 'label': 'C-H伸縮 (CH2)'},
        {'center': 1742, 'intensity': 1.5, 'width': 20, 'label': 'C=O伸縮 (エステル)'},
        {'center': 1465, 'intensity': 0.3, 'width': 20, 'label': 'C-H変角'},
        {'center': 1375, 'intensity': 0.4, 'width': 15, 'label': 'C-H変角 (CH3)'},
        {'center': 1240, 'intensity': 1.0, 'width': 40, 'label': 'C-O伸縮 (エステル)'},
        {'center': 1050, 'intensity': 0.8, 'width': 30, 'label': 'C-O伸縮'},
    ]
    
    # エタノール (CH3CH2OH) のIRスペクトル
    ethanol_peaks = [
        {'center': 3350, 'intensity': 1.2, 'width': 150, 'shape': 'gaussian', 'label': 'O-H伸縮'},
        {'center': 2975, 'intensity': 0.6, 'width': 25, 'label': 'C-H伸縮 (CH3)'},
        {'center': 2930, 'intensity': 0.5, 'width': 20, 'label': 'C-H伸縮 (CH2)'},
        {'center': 1450, 'intensity': 0.3, 'width': 25, 'label': 'C-H変角'},
        {'center': 1380, 'intensity': 0.25, 'width': 20, 'label': 'C-H変角'},
        {'center': 1050, 'intensity': 0.9, 'width': 35, 'label': 'C-O伸縮'},
        {'center': 880, 'intensity': 0.4, 'width': 25, 'label': 'C-C伸縮'},
    ]
    
    # 比較プロット
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    compounds = [
        ('酢酸エチル (CH$_3$COOCH$_2$CH$_3$)', ethyl_acetate_peaks, '#f093fb'),
        ('エタノール (CH$_3$CH$_2$OH)', ethanol_peaks, '#f5576c'),
    ]
    
    for ax, (name, peaks, color) in zip(axes, compounds):
        wn, trans = simulate_ir_spectrum(name, peaks)
    
        ax.plot(wn, trans, color=color, linewidth=1.5)
        ax.fill_between(wn, trans, 100, alpha=0.2, color=color)
    
        # ピークラベル
        for peak in peaks:
            idx = np.argmin(np.abs(wn - peak['center']))
            y = trans[idx]
            if y < 70:
                ax.annotate(peak['label'], xy=(peak['center'], y),
                           xytext=(peak['center'], y - 12),
                           fontsize=8, ha='center', rotation=45,
                           arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
        ax.set_xlim(4000, 400)
        ax.set_ylim(0, 105)
        ax.invert_xaxis()
        ax.set_xlabel('波数 (cm$^{-1}$)', fontsize=11)
        ax.set_ylabel('透過率 (%)', fontsize=11)
        ax.set_title(f'{name} のIRスペクトル', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
        # 領域区分
        ax.axvspan(4000, 2500, alpha=0.05, color='blue')
        ax.axvspan(2500, 1500, alpha=0.05, color='green')
        ax.axvspan(1500, 400, alpha=0.05, color='orange')
    
        ax.text(3200, 98, 'X-H領域', fontsize=9, ha='center', color='blue')
        ax.text(2000, 98, '不飽和領域', fontsize=9, ha='center', color='green')
        ax.text(900, 98, '指紋領域', fontsize=9, ha='center', color='orange')
    
    plt.tight_layout()
    plt.savefig('ir_functional_groups.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 構造判定のポイント
    print("=" * 70)
    print("IRスペクトルによる官能基同定")
    print("=" * 70)
    print("\n【酢酸エチル】")
    print("  - 1742 cm^-1: 強いC=O伸縮 → エステルカルボニル")
    print("  - 1240, 1050 cm^-1: C-O伸縮 → エステル結合確認")
    print("  - O-H伸縮なし → アルコール/カルボン酸ではない")
    print("\n【エタノール】")
    print("  - 3350 cm^-1: 幅広いO-H伸縮 → アルコール（水素結合）")
    print("  - 1050 cm^-1: C-O伸縮")
    print("  - C=O伸縮なし → カルボニル化合物ではない")

## 3.5 ATR-FTIR法による表面分析

### 3.5.1 全反射減衰の原理

ATR（Attenuated Total Reflection：全反射減衰）法は、試料の前処理がほとんど不要で、固体・液体・ペースト状試料を直接測定できる手法です。

高屈折率の結晶（ATR結晶）と低屈折率の試料界面で全反射が起こるとき、結晶表面からわずかに染み出す**エバネッセント波** が試料に吸収されます。

**浸透深さ（Penetration Depth）：**

$$d_p = \frac{\lambda}{2\pi n_1 \sqrt{\sin^2\theta - (n_2/n_1)^2}}$$ 

ここで、$\lambda$ は波長、$n_1$ は ATR結晶の屈折率、$n_2$ は試料の屈折率、$\theta$ は入射角です。

典型的な浸透深さは **0.5-5 $\mu$m** 程度で、表面近傍の情報が得られます。
    
    
    ```mermaid
    flowchart LR
                subgraph ATR結晶
                    A[IR光入射] --> B[全反射]
                    B --> C[全反射]
                    C --> D[検出器へ]
                end
    
                E[試料] --- B
                E --- C
    
                style E fill:#ffeb3b,stroke:#f57f17
    ```

ATR結晶 | 屈折率 | 波数範囲 (cm$^{-1}$) | 特徴  
---|---|---|---  
ダイヤモンド | 2.4 | 4000-400 | 最も汎用的、硬度が高い  
ZnSe | 2.4 | 4000-650 | 低コスト、酸に弱い  
Ge | 4.0 | 4000-700 | 浸透深さが浅い、表面敏感  
Si | 3.4 | 8000-1500 | 近赤外用  
  
#### コード例6: ATR浸透深さの計算とシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def penetration_depth(wavelength_um, n1, n2, theta_deg):
        """
        ATRのエバネッセント波浸透深さを計算
    
        Parameters:
        -----------
        wavelength_um : float or array
            波長 (um)
        n1 : float
            ATR結晶の屈折率
        n2 : float
            試料の屈折率
        theta_deg : float
            入射角 (度)
    
        Returns:
        --------
        dp : float or array
            浸透深さ (um)
        """
        theta = np.radians(theta_deg)
        n_ratio = n2 / n1
    
        # 全反射条件の確認
        critical_angle = np.degrees(np.arcsin(n_ratio))
        if theta_deg < critical_angle:
            print(f"警告: 入射角 ({theta_deg}度) が臨界角 ({critical_angle:.1f}度) 未満です")
            return None
    
        denominator = np.sqrt(np.sin(theta)**2 - n_ratio**2)
        dp = wavelength_um / (2 * np.pi * n1 * denominator)
    
        return dp
    
    # パラメータ設定
    crystals = {
        'ダイヤモンド': {'n': 2.4, 'color': '#2196f3'},
        'Ge': {'n': 4.0, 'color': '#4caf50'},
        'ZnSe': {'n': 2.4, 'color': '#ff9800'},
    }
    
    # 試料（ポリマー）の屈折率
    n_sample = 1.5
    theta = 45  # 入射角
    
    # 波数から波長への変換
    wavenumbers = np.linspace(4000, 500, 500)  # cm^-1
    wavelengths_um = 10000 / wavenumbers  # um (1 cm^-1 = 10^4 / um)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 浸透深さ vs 波数
    ax1 = axes[0]
    for crystal_name, props in crystals.items():
        dp = penetration_depth(wavelengths_um, props['n'], n_sample, theta)
        ax1.plot(wavenumbers, dp, label=crystal_name, color=props['color'], linewidth=2)
    
    ax1.set_xlabel('波数 (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('浸透深さ ($\\mu$m)', fontsize=12)
    ax1.set_title(f'ATR浸透深さの波数依存性（入射角={theta}度）', fontsize=14, fontweight='bold')
    ax1.set_xlim(4000, 500)
    ax1.invert_xaxis()
    ax1.set_ylim(0, 5)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 浸透深さ vs 入射角
    ax2 = axes[1]
    angles = np.linspace(30, 60, 100)
    wavelength_fixed = 5  # 2000 cm^-1 に相当
    
    for crystal_name, props in crystals.items():
        n1 = props['n']
        critical = np.degrees(np.arcsin(n_sample / n1))
    
        dp_angles = []
        valid_angles = []
        for angle in angles:
            if angle > critical:
                dp = penetration_depth(wavelength_fixed, n1, n_sample, angle)
                if dp is not None:
                    dp_angles.append(dp)
                    valid_angles.append(angle)
    
        if valid_angles:
            ax2.plot(valid_angles, dp_angles, label=f'{crystal_name} (臨界角:{critical:.1f}度)',
                    color=props['color'], linewidth=2)
            ax2.axvline(x=critical, color=props['color'], linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('入射角 (度)', fontsize=12)
    ax2.set_ylabel('浸透深さ ($\\mu$m)', fontsize=12)
    ax2.set_title(f'ATR浸透深さの入射角依存性（2000 cm$^{{-1}}$）', fontsize=14, fontweight='bold')
    ax2.set_xlim(30, 60)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('atr_penetration_depth.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 具体的な数値例
    print("=" * 60)
    print("ATR浸透深さの具体例（試料屈折率=1.5、入射角=45度）")
    print("=" * 60)
    print(f"{'結晶':<15} {'1000 cm^-1':<15} {'2000 cm^-1':<15} {'3000 cm^-1'}")
    print("-" * 60)
    for crystal_name, props in crystals.items():
        dp_1000 = penetration_depth(10, props['n'], 1.5, 45)
        dp_2000 = penetration_depth(5, props['n'], 1.5, 45)
        dp_3000 = penetration_depth(3.33, props['n'], 1.5, 45)
        print(f"{crystal_name:<15} {dp_1000:.2f} um {' '*7} {dp_2000:.2f} um {' '*7} {dp_3000:.2f} um")

## 3.6 IRスペクトルのデータ処理

### 3.6.1 ベースライン補正

実測IRスペクトルでは、光学系のドリフトや試料の不均一性により、ベースラインが傾斜したり曲がったりすることがあります。定量分析や正確なピーク位置決定のためには、ベースライン補正が必要です。

#### コード例7: ベースライン補正とピーク検出
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.signal import find_peaks, savgol_filter
    
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares (ALS) によるベースライン推定
    
        Parameters:
        -----------
        y : array
            入力スペクトル
        lam : float
            平滑化パラメータ（大きいほど滑らか）
        p : float
            非対称パラメータ（0-1、小さいほどベースラインが下に）
        niter : int
            反復回数
    
        Returns:
        --------
        z : array
            推定されたベースライン
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
    
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
    
        return z
    
    def detect_peaks_ir(wavenumbers, absorbance, height=0.05, prominence=0.02):
        """
        IRスペクトルのピーク検出
    
        Parameters:
        -----------
        wavenumbers : array
            波数軸
        absorbance : array
            吸光度（ベースライン補正済み）
        height : float
            最小ピーク高さ
        prominence : float
            最小プロミネンス
    
        Returns:
        --------
        peak_positions : array
            ピーク位置（波数）
        peak_heights : array
            ピーク高さ
        """
        # スムージング
        smoothed = savgol_filter(absorbance, window_length=11, polyorder=3)
    
        # ピーク検出
        peaks, properties = find_peaks(smoothed, height=height, prominence=prominence)
    
        peak_positions = wavenumbers[peaks]
        peak_heights = smoothed[peaks]
    
        return peak_positions, peak_heights, peaks
    
    # サンプルデータ生成（ノイズとベースラインドリフトを含む）
    np.random.seed(42)
    wavenumbers = np.linspace(4000, 400, 2000)
    
    # 真のスペクトル（複数のピーク）
    true_peaks = [
        (3400, 0.8, 80),   # O-H
        (2950, 0.5, 30),   # C-H
        (1720, 1.0, 25),   # C=O
        (1450, 0.3, 20),   # C-H変角
        (1050, 0.6, 40),   # C-O
    ]
    
    true_absorbance = np.zeros_like(wavenumbers)
    for center, intensity, width in true_peaks:
        true_absorbance += intensity * np.exp(-((wavenumbers - center)**2) / (2 * width**2))
    
    # ベースラインドリフトを追加
    baseline_drift = 0.2 * np.exp(-((wavenumbers - 2000)**2) / (2 * 1500**2)) + 0.1
    
    # ノイズを追加
    noise = 0.02 * np.random.randn(len(wavenumbers))
    
    # 観測スペクトル
    observed = true_absorbance + baseline_drift + noise
    
    # ベースライン補正
    estimated_baseline = baseline_als(observed, lam=1e6, p=0.001)
    corrected = observed - estimated_baseline
    
    # ピーク検出
    peak_wn, peak_h, peak_idx = detect_peaks_ir(wavenumbers, corrected, height=0.05, prominence=0.03)
    
    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 生データ
    ax1 = axes[0]
    ax1.plot(wavenumbers, observed, color='#666', linewidth=1, label='観測スペクトル')
    ax1.plot(wavenumbers, estimated_baseline, color='#f093fb', linewidth=2, linestyle='--', label='推定ベースライン')
    ax1.set_xlabel('波数 (cm$^{-1}$)', fontsize=11)
    ax1.set_ylabel('吸光度', fontsize=11)
    ax1.set_title('ベースライン推定（ALS法）', fontsize=13, fontweight='bold')
    ax1.set_xlim(4000, 400)
    ax1.invert_xaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 補正後
    ax2 = axes[1]
    ax2.plot(wavenumbers, corrected, color='#f5576c', linewidth=1.2, label='補正後スペクトル')
    ax2.plot(wavenumbers, true_absorbance, color='#4caf50', linewidth=1.5, linestyle=':', alpha=0.8, label='真のスペクトル')
    ax2.scatter(peak_wn, peak_h, color='blue', s=50, zorder=5, label='検出ピーク')
    ax2.set_xlabel('波数 (cm$^{-1}$)', fontsize=11)
    ax2.set_ylabel('吸光度', fontsize=11)
    ax2.set_title('ベースライン補正後とピーク検出', fontsize=13, fontweight='bold')
    ax2.set_xlim(4000, 400)
    ax2.invert_xaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 検出されたピーク位置
    for wn, h in zip(peak_wn, peak_h):
        ax2.annotate(f'{wn:.0f}', xy=(wn, h), xytext=(wn, h + 0.1),
                    fontsize=9, ha='center')
    
    # ピーク比較表
    ax3 = axes[2]
    ax3.axis('off')
    
    # 表形式でピーク比較
    table_data = [['検出波数 (cm$^{-1}$)', '真の波数 (cm$^{-1}$)', '誤差 (cm$^{-1}$)', '帰属']]
    assignments = {3400: 'O-H伸縮', 2950: 'C-H伸縮', 1720: 'C=O伸縮', 1450: 'C-H変角', 1050: 'C-O伸縮'}
    
    for det_wn in sorted(peak_wn, reverse=True):
        # 最も近い真のピークを探す
        true_wns = [p[0] for p in true_peaks]
        closest_idx = np.argmin(np.abs(np.array(true_wns) - det_wn))
        true_wn = true_wns[closest_idx]
        error = det_wn - true_wn
        assign = assignments.get(true_wn, '-')
        table_data.append([f'{det_wn:.1f}', f'{true_wn}', f'{error:+.1f}', assign])
    
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center',
                      colColours=['#f093fb']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax3.set_title('ピーク検出結果の評価', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ir_baseline_correction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)
    print("ピーク検出結果")
    print("=" * 60)
    for wn, h in zip(peak_wn, peak_h):
        print(f"  {wn:.1f} cm^-1 (強度: {h:.3f})")

### 3.6.2 スペクトルライブラリ検索

#### コード例8: 簡易スペクトルマッチング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cosine
    from scipy.interpolate import interp1d
    
    class IRSpectralLibrary:
        """
        IRスペクトルライブラリと検索機能
        """
    
        def __init__(self):
            self.library = {}
            self.wavenumber_range = (4000, 400)
            self.n_points = 1000
            self.wavenumbers = np.linspace(*self.wavenumber_range, self.n_points)
    
        def add_compound(self, name, peaks):
            """
            化合物のスペクトルをライブラリに追加
    
            Parameters:
            -----------
            name : str
                化合物名
            peaks : list of tuples
                (波数, 強度, 幅) のリスト
            """
            spectrum = np.zeros(self.n_points)
            for center, intensity, width in peaks:
                spectrum += intensity * np.exp(-((self.wavenumbers - center)**2) / (2 * width**2))
    
            # 正規化
            spectrum = spectrum / np.max(spectrum)
            self.library[name] = {
                'spectrum': spectrum,
                'peaks': peaks
            }
    
        def search(self, query_spectrum, query_wavenumbers=None, top_n=3):
            """
            クエリスペクトルに最も類似した化合物を検索
    
            Parameters:
            -----------
            query_spectrum : array
                クエリスペクトル
            query_wavenumbers : array, optional
                クエリの波数軸（異なる場合は補間）
            top_n : int
                返す結果の数
    
            Returns:
            --------
            results : list of tuples
                (化合物名, 類似度スコア) のリスト
            """
            # 波数軸が異なる場合は補間
            if query_wavenumbers is not None:
                f = interp1d(query_wavenumbers, query_spectrum,
                            bounds_error=False, fill_value=0)
                query = f(self.wavenumbers)
            else:
                query = query_spectrum
    
            # 正規化
            query = query / np.max(query) if np.max(query) > 0 else query
    
            # コサイン類似度で比較
            scores = []
            for name, data in self.library.items():
                ref = data['spectrum']
                similarity = 1 - cosine(query, ref)  # コサイン類似度
                scores.append((name, similarity))
    
            # スコア順にソート
            scores.sort(key=lambda x: x[1], reverse=True)
    
            return scores[:top_n]
    
        def plot_comparison(self, query_spectrum, query_wavenumbers, match_name):
            """
            クエリと参照スペクトルを比較表示
            """
            fig, ax = plt.subplots(figsize=(12, 6))
    
            # クエリ
            ax.plot(query_wavenumbers, query_spectrum / np.max(query_spectrum),
                   label='クエリスペクトル', color='#f5576c', linewidth=1.5)
    
            # 参照
            ref = self.library[match_name]['spectrum']
            ax.plot(self.wavenumbers, ref,
                   label=f'参照: {match_name}', color='#4caf50', linewidth=1.5, linestyle='--')
    
            ax.set_xlabel('波数 (cm$^{-1}$)', fontsize=12)
            ax.set_ylabel('正規化強度', fontsize=12)
            ax.set_title('スペクトルマッチング結果', fontsize=14, fontweight='bold')
            ax.set_xlim(4000, 400)
            ax.invert_xaxis()
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
    
            return fig
    
    # ライブラリ構築
    library = IRSpectralLibrary()
    
    # 代表的な化合物を登録
    compounds = {
        'エタノール': [
            (3350, 1.0, 100),  # O-H
            (2970, 0.6, 30),   # C-H
            (1050, 0.8, 40),   # C-O
        ],
        'アセトン': [
            (2970, 0.5, 25),   # C-H
            (1715, 1.0, 20),   # C=O
            (1360, 0.4, 20),   # C-H変角
        ],
        '酢酸': [
            (3000, 0.8, 200),  # O-H (カルボン酸)
            (1710, 1.0, 25),   # C=O
            (1280, 0.6, 40),   # C-O
        ],
        'トルエン': [
            (3030, 0.5, 25),   # 芳香族C-H
            (2920, 0.4, 20),   # 脂肪族C-H
            (1600, 0.6, 15),   # 芳香環C=C
            (1500, 0.7, 15),   # 芳香環C=C
            (730, 0.8, 20),    # 面外変角
        ],
        '酢酸エチル': [
            (2980, 0.5, 25),   # C-H
            (1740, 1.0, 20),   # C=O (エステル)
            (1240, 0.8, 40),   # C-O
            (1050, 0.6, 30),   # C-O
        ],
    }
    
    for name, peaks in compounds.items():
        library.add_compound(name, peaks)
    
    # 未知試料のスペクトル生成（酢酸エチル + ノイズ）
    np.random.seed(123)
    unknown_wn = np.linspace(4000, 400, 800)
    unknown_spectrum = np.zeros_like(unknown_wn)
    for center, intensity, width in compounds['酢酸エチル']:
        unknown_spectrum += intensity * np.exp(-((unknown_wn - center)**2) / (2 * width**2))
    unknown_spectrum += 0.05 * np.random.randn(len(unknown_wn))  # ノイズ追加
    
    # 検索実行
    results = library.search(unknown_spectrum, unknown_wn, top_n=5)
    
    print("=" * 60)
    print("スペクトルライブラリ検索結果")
    print("=" * 60)
    print(f"{'順位':<6} {'化合物':<15} {'類似度スコア'}")
    print("-" * 60)
    for i, (name, score) in enumerate(results, 1):
        print(f"{i:<6} {name:<15} {score:.4f}")
    
    # 最良マッチの可視化
    fig = library.plot_comparison(unknown_spectrum, unknown_wn, results[0][0])
    plt.savefig('ir_spectral_matching.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n最も可能性の高い化合物: {results[0][0]} (類似度: {results[0][1]:.4f})")

## 3.7 演習問題

#### 演習1: 振動周波数の計算

以下の化学結合について、調和振動子モデルを用いて振動周波数（波数 cm$^{-1}$）を計算してください。

  * C-F結合（力の定数: 600 N/m）
  * C-Cl結合（力の定数: 350 N/m）
  * C-Br結合（力の定数: 280 N/m）

結果から、ハロゲンの原子量と振動周波数の関係を考察してください。

解答例を見る
    
    
    # 原子質量 (amu)
    masses = {'C': 12, 'F': 19, 'Cl': 35, 'Br': 80}
    
    # C-X結合の振動波数計算
    bonds = [
        ('C-F', 600, 12, 19),
        ('C-Cl', 350, 12, 35),
        ('C-Br', 280, 12, 80),
    ]
    
    for bond, k, m1, m2 in bonds:
        wn = calculate_wavenumber(k, m1, m2)
        print(f"{bond}: {wn:.0f} cm^-1")
    
    # 結果:
    # C-F: 約1100 cm^-1
    # C-Cl: 約750 cm^-1
    # C-Br: 約560 cm^-1
    #
    # 考察: ハロゲンの原子量が大きくなるほど、
    # 換算質量が増加し、振動周波数は低下する

#### 演習2: IRスペクトル解析

以下のIRスペクトルの吸収帯から、試料に含まれる官能基を推定してください。

  * 3300 cm$^{-1}$: 中程度、鋭いピーク
  * 2120 cm$^{-1}$: 中程度
  * 1640 cm$^{-1}$: 弱い

解答例を見る

**推定構造: 末端アルキン（R-C$\equiv$C-H）**

  * 3300 cm$^{-1}$: $\equiv$C-H 伸縮振動（末端アルキン特有の鋭いピーク）
  * 2120 cm$^{-1}$: C$\equiv$C 伸縮振動
  * 1640 cm$^{-1}$: 可能性として C=C 伸縮（不純物または共役系）

末端アルキンでは、$\equiv$C-H の振動がIR活性であり、内部アルキン（R-C$\equiv$C-R'）よりも強い吸収を示します。

#### 演習3: ATR-FTIR測定条件の最適化

厚さ約2 $\mu$mのポリマー薄膜（屈折率 n=1.5）をATR-FTIRで測定します。C=O伸縮振動（1700 cm$^{-1}$付近）を十分な感度で検出するために、最適なATR結晶と入射角を選択してください。

解答例を見る
    
    
    # 1700 cm^-1 での波長
    wavelength = 10000 / 1700  # 約5.9 um
    
    # 各結晶での浸透深さ（入射角45度）
    for crystal, n in [('ダイヤモンド', 2.4), ('Ge', 4.0), ('ZnSe', 2.4)]:
        dp = penetration_depth(wavelength, n, 1.5, 45)
        print(f"{crystal}: {dp:.2f} um")
    
    # 結果:
    # ダイヤモンド: 約1.5 um
    # Ge: 約0.5 um
    # ZnSe: 約1.5 um
    #
    # 推奨: ダイヤモンドまたはZnSe結晶（入射角45度）
    # 浸透深さ約1.5 umで、2 um薄膜の情報を十分に取得できる
    # Ge結晶は浸透深さが浅すぎるため、薄膜表面のみの情報になる

#### 演習4: ベースライン補正の実践

提供されたIRスペクトルデータに対して、ALS法でベースライン補正を行い、主要ピークの位置と強度を求めてください。平滑化パラメータ（lam）と非対称パラメータ（p）を変えて、最適な補正結果を得てください。

解答のヒント

  * lam（平滑化パラメータ）: 10$^4$ - 10$^7$ の範囲で試す。大きいほど滑らかなベースライン
  * p（非対称パラメータ）: 0.001 - 0.1 の範囲で試す。小さいほどベースラインが下方に
  * 補正後のスペクトルでピークがマイナスにならないよう注意
  * ノイズが多い場合は、Savitzky-Golayフィルタで平滑化してからピーク検出

## まとめ

**この章で学んだこと**

  * **分子振動の基礎** ：調和振動子モデル、伸縮振動と変角振動の分類
  * **赤外選択則** ：双極子モーメントの変化がIR活性の条件
  * **FTIR原理** ：マイケルソン干渉計とフーリエ変換による高速・高感度測定
  * **官能基同定** ：特性吸収帯を利用した構造解析
  * **ATR-FTIR** ：エバネッセント波による表面敏感測定
  * **データ処理** ：ベースライン補正、ピーク検出、スペクトルマッチング

**次章の予告**

第4章では**ラマン分光法** を学びます。ラマン散乱の原理、IRとの相補性、結晶構造解析への応用、表面増強ラマン散乱（SERS）について解説します。
